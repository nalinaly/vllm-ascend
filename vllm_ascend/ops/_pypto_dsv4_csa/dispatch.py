# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Production custom-op dispatch owner for PyPTO DeepSeek V4 decode CSA.

The public vLLM custom-op schema remains unchanged. An explicitly installed
layer owner replaces only the implementation behind that op; unsupported
layers keep the native implementation because no owner is attached.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, field

import torch

from .adapter import PreparedDecodeCSACaches, prepare_decode_csa_caches
from .backend import TRB_RUNTIME, PyPTOBackendDependencies, PyPTODSABackend
from .config import DECODE_SEQ, FLASH
from .contract import DecodeCSAProgramSpec
from .native_metadata import (
    bind_decode_csa_native_metadata,
    derive_decode_csa_program_spec,
    validate_decode_csa_uniform_query_rows,
)
from .weights import PreparedDecodeCSAWeights, pack_decode_csa_weights

_LAYER_DISPATCH_ATTRIBUTE = "_pypto_dsv4_csa_dispatch"


class DecodeCSADispatchError(RuntimeError):
    """Raised when an installed PyPTO layer cannot safely service a call."""


def _spec_family_key(spec: DecodeCSAProgramSpec) -> tuple[object, ...]:
    # DecodeCSAProgramSpec.key starts with batch. The rest is the shared family.
    return spec.key[1:]


def _clone_spec_for_batch(spec: DecodeCSAProgramSpec, batch: int) -> DecodeCSAProgramSpec:
    return DecodeCSAProgramSpec(
        batch=batch,
        seq=spec.seq,
        swa_blocks=spec.swa_blocks,
        compressed_blocks=spec.compressed_blocks,
        main_state_blocks=spec.main_state_blocks,
        inner_state_blocks=spec.inner_state_blocks,
        indexer_blocks=spec.indexer_blocks,
        swa_table_width=spec.swa_table_width,
        compressed_table_width=spec.compressed_table_width,
        main_state_table_width=spec.main_state_table_width,
        inner_state_table_width=spec.inner_state_table_width,
        indexer_table_width=spec.indexer_table_width,
        physical_layout=spec.physical_layout,
    )


def _tensor_storage_key(tensor: torch.Tensor) -> tuple[object, ...]:
    return (
        int(tensor.data_ptr()),
        int(tensor.storage_offset()),
        tuple(int(extent) for extent in tensor.shape),
        tuple(int(stride) for stride in tensor.stride()),
        tensor.dtype,
        tensor.device,
    )


def _cache_storage_key(cache_tuple: Sequence[object]) -> tuple[tuple[object, ...], ...]:
    if not isinstance(cache_tuple, (tuple, list)) or len(cache_tuple) != 6:
        raise DecodeCSADispatchError("installed ratio-4 A2/A3 owner requires the original six-cache tuple")
    keys: list[tuple[object, ...]] = []
    for index, value in enumerate(cache_tuple):
        if not isinstance(value, torch.Tensor):
            raise DecodeCSADispatchError(f"cache[{index}] must be torch.Tensor, got {type(value).__name__}")
        keys.append(_tensor_storage_key(value))
    return tuple(keys)


def _require_output_storage_isolated(
    output: torch.Tensor,
    named_arguments: Mapping[str, object],
) -> None:
    """Reject an output that shares storage with any input or mutable state.

    The outlined program assumes ``attn_out`` is a pure ``Out`` parameter.  A
    same-storage view would violate that direction contract even when its
    logical range happened not to overlap today.  Checking storage identity is
    metadata-only and therefore does not synchronize the device.
    """
    output_storage = output.untyped_storage()._cdata
    for name, value in named_arguments.items():
        if name == "attn_out" or not isinstance(value, torch.Tensor):
            continue
        if value is output or value.untyped_storage()._cdata == output_storage:
            raise DecodeCSADispatchError(
                f"output must not alias tensor argument {name!r}; attn_out is a pure Out parameter"
            )


@dataclass(slots=True)
class DecodeCSADeviceOwner:
    """One shared backend/context for every enabled layer on a device."""

    device: int
    runtime: str
    backend: PyPTODSABackend
    specs: Mapping[int, DecodeCSAProgramSpec]
    family_key: tuple[object, ...]
    _warmup_enqueued: set[int] = field(default_factory=set, init=False)
    _capture_ready: set[int] = field(default_factory=set, init=False)
    _initialization_error: BaseException | None = field(default=None, init=False, repr=False)

    @property
    def cleanup_only(self) -> bool:
        """Whether initialization failed after this owner acquired resources."""
        return self._initialization_error is not None

    def _require_ready(self) -> None:
        if self._initialization_error is not None:
            raise DecodeCSADispatchError(
                "PyPTO decode CSA initialization previously failed; the retained "
                "owner is cleanup-only and cannot accept calls"
            ) from self._initialization_error

    def spec_for_batch(self, batch: int) -> DecodeCSAProgramSpec:
        self._require_ready()
        try:
            return self.specs[batch]
        except KeyError as error:
            raise DecodeCSADispatchError(
                f"decode batch {batch} has no prepared PyPTO specialization; prepared={tuple(self.specs)}"
            ) from error

    def warmup(self, binding: object, *, batch: int) -> object:
        self._require_ready()
        result = self.backend.warmup(binding)  # type: ignore[arg-type]
        self._warmup_enqueued.add(batch)
        self._capture_ready.discard(batch)
        return result

    def launch(self, binding: object) -> object:
        self._require_ready()
        return self.backend.launch(binding)  # type: ignore[arg-type]

    def is_warmed(self, batch: int) -> bool:
        return batch in self._warmup_enqueued

    def require_capture_ready(self, batch: int) -> None:
        if batch not in self._capture_ready:
            raise DecodeCSADispatchError(
                "PyPTO decode CSA was not externally quiesced after ordinary warmup; "
                "synchronize the caller outside the operator, then mark warmups "
                "quiesced before ACLGraph capture"
            )

    def mark_warmups_quiesced(self) -> None:
        """Mark already-enqueued warmups ready; this method never synchronizes."""
        self._require_ready()
        self._capture_ready.update(self._warmup_enqueued)

    def close(self) -> None:
        """Close through the retained backend; failures remain retryable there."""
        self.backend.close()


class DecodeCSADeviceOwnerRegistry:
    """Process-pinned device-owner registry; no implicit GC/atexit shutdown."""

    def __init__(
        self,
        *,
        backend_factory: Callable[..., PyPTODSABackend] = PyPTODSABackend,
    ) -> None:
        self._lock = threading.RLock()
        self._owners: dict[tuple[int, str], DecodeCSADeviceOwner] = {}
        self._backend_factory = backend_factory

    def get_or_create(
        self,
        *,
        device: int,
        runtime: str,
        base_spec: DecodeCSAProgramSpec,
        batch_buckets: Sequence[int],
        backend_dependencies: PyPTOBackendDependencies | None = None,
    ) -> DecodeCSADeviceOwner:
        buckets = tuple(dict.fromkeys(int(batch) for batch in batch_buckets))
        if not buckets:
            raise ValueError("batch_buckets must not be empty")
        specs = {batch: _clone_spec_for_batch(base_spec, batch) for batch in buckets}
        family_key = _spec_family_key(base_spec)
        key = (device, runtime)
        with self._lock:
            existing = self._owners.get(key)
            if existing is not None:
                existing._require_ready()
                if existing.family_key != family_key:
                    raise DecodeCSADispatchError(
                        f"device {device} already owns a different decode CSA cache/layout family"
                    )
                missing = tuple(batch for batch in buckets if batch not in existing.specs)
                if missing:
                    raise DecodeCSADispatchError(
                        "the shared PyPTO registry is already prepared and cannot add "
                        f"batch buckets {missing}; register the complete family first"
                    )
                return existing

            backend = self._backend_factory(
                device=device,
                runtime=runtime,
                dependencies=backend_dependencies,
            )
            for spec in specs.values():
                backend.compile(spec)
            owner = DecodeCSADeviceOwner(
                device=device,
                runtime=runtime,
                backend=backend,
                specs=specs,
                family_key=family_key,
            )
            try:
                backend.prepare()
            except BaseException as error:
                # Context construction can fail after PyPTO has created a
                # close-only cleanup context and claimed this device/runtime.
                # Keep the complete backend owner strongly reachable instead
                # of losing it with this stack frame.  The original exception
                # remains authoritative and receives a discoverable cleanup
                # owner for callers that do not hold the registry directly.
                owner._initialization_error = error
                self._owners[key] = owner
                with suppress(AttributeError, TypeError):
                    error.decode_csa_cleanup_owner = owner  # type: ignore[attr-defined]
                raise
            self._owners[key] = owner
            return owner

    def get(self, *, device: int, runtime: str) -> DecodeCSADeviceOwner | None:
        with self._lock:
            return self._owners.get((device, runtime))

    def mark_warmups_quiesced(self, *, device: int, runtime: str) -> None:
        owner = self.get(device=device, runtime=runtime)
        if owner is None:
            raise DecodeCSADispatchError(f"no PyPTO decode CSA owner for device={device}, runtime={runtime!r}")
        owner.mark_warmups_quiesced()


_PROCESS_DEVICE_OWNERS = DecodeCSADeviceOwnerRegistry()


@dataclass(slots=True)
class DecodeCSALayerOwner:
    """Stable owner attached to one AscendDeepseekSparseAttention layer."""

    layer: object
    device_owner: DecodeCSADeviceOwner
    weights: PreparedDecodeCSAWeights
    prepared_caches: Mapping[int, PreparedDecodeCSACaches]
    cache_key: tuple[tuple[object, ...], ...]

    def _require_io(self, hidden_states: torch.Tensor, output: torch.Tensor) -> int:
        if not isinstance(hidden_states, torch.Tensor) or not isinstance(output, torch.Tensor):
            raise DecodeCSADispatchError("hidden_states and output must be torch.Tensor")
        if hidden_states.device.type != "npu" or hidden_states.device.index != self.device_owner.device:
            raise DecodeCSADispatchError(
                f"hidden_states must be on npu:{self.device_owner.device}, got {hidden_states.device}"
            )
        if output.device != hidden_states.device:
            raise DecodeCSADispatchError("output must share hidden_states device")
        if hidden_states.ndim != 2 or hidden_states.shape[1] != FLASH.hidden_size:
            raise DecodeCSADispatchError(
                f"hidden_states must have shape [B*{DECODE_SEQ},{FLASH.hidden_size}], got {tuple(hidden_states.shape)}"
            )
        if output.shape != hidden_states.shape:
            raise DecodeCSADispatchError(
                f"output shape must equal hidden_states shape, got "
                f"{tuple(output.shape)} vs {tuple(hidden_states.shape)}"
            )
        if hidden_states.dtype != torch.bfloat16 or output.dtype != torch.bfloat16:
            raise DecodeCSADispatchError(f"hidden_states/output must be BF16, got {hidden_states.dtype}/{output.dtype}")
        if not hidden_states.is_contiguous() or not output.is_contiguous():
            raise DecodeCSADispatchError("hidden_states/output must use canonical contiguous layout")
        tokens = int(hidden_states.shape[0])
        if tokens % DECODE_SEQ:
            raise DecodeCSADispatchError(f"token count {tokens} is not divisible by decode sequence {DECODE_SEQ}")
        return tokens // DECODE_SEQ

    def __call__(
        self,
        *,
        forward_context: object,
        capturing: bool,
        hidden_states: torch.Tensor,
        need_gather_q_kv: bool,
        output: torch.Tensor,
        kv_cache: Sequence[object],
        attn_metadata: Sequence[object],
    ) -> torch.Tensor:
        if not isinstance(capturing, bool):
            raise DecodeCSADispatchError(f"capturing must be bool, got {type(capturing).__name__}")
        if need_gather_q_kv:
            raise DecodeCSADispatchError("FlashComm/need_gather_q_kv is unsupported by the single L1 CSA op")
        if _cache_storage_key(kv_cache) != self.cache_key:
            raise DecodeCSADispatchError(
                "DSA cache storage changed after PyPTO preparation; sleep/wake "
                "and cache rebind require explicit reinstall"
            )
        batch = self._require_io(hidden_states, output)
        spec = self.device_owner.spec_for_batch(batch)
        warmed = self.device_owner.is_warmed(batch)
        if capturing and not warmed:
            raise DecodeCSADispatchError(
                "PyPTO decode CSA specialization has not completed an ordinary-eager warmup before ACLGraph capture"
            )
        if capturing:
            self.device_owner.require_capture_ready(batch)

        native = bind_decode_csa_native_metadata(
            attn_metadata,
            spec,
            device=hidden_states.device,
        )
        arguments: dict[str, object] = dict(self.weights.for_launch())
        arguments.update(native.for_launch())
        arguments["hidden_states"] = hidden_states
        arguments["attn_out"] = output
        cache_arguments = self.prepared_caches[batch].for_launch()
        _require_output_storage_isolated(output, {**arguments, **cache_arguments})

        binding = self.device_owner.backend.bind(
            spec,
            arguments,
            prepared_caches=self.prepared_caches[batch],
            owners=(self, native),
            # Eager leases belong to taskQueue. Only captured snapshots stay pinned.
            retain=capturing,
        )
        if not warmed:
            return self.device_owner.warmup(binding, batch=batch)
        return self.device_owner.launch(binding)


def install_pypto_dsv4_decode_csa(
    layer: object,
    *,
    kv_cache: Sequence[object],
    attn_metadata: Sequence[object],
    device: int,
    runtime: str = TRB_RUNTIME,
    batch_buckets: Sequence[int] | None = None,
    registry: DecodeCSADeviceOwnerRegistry = _PROCESS_DEVICE_OWNERS,
    backend_dependencies: PyPTOBackendDependencies | None = None,
    prepared_weights: PreparedDecodeCSAWeights | None = None,
    uniform_query_rows_contract: bool = False,
) -> DecodeCSALayerOwner:
    """Cold-prepare and attach one owner; call only outside ACLGraph capture."""
    if getattr(layer, _LAYER_DISPATCH_ATTRIBUTE, None) is not None:
        raise DecodeCSADispatchError("layer already has an installed PyPTO decode CSA owner")
    if not uniform_query_rows_contract:
        raise DecodeCSADispatchError(
            "the current static ABI requires an explicit uniform_query_rows_contract=True "
            "guarantee that every decode request keeps query length S=8"
        )
    base_spec = derive_decode_csa_program_spec(kv_cache, attn_metadata)
    validate_decode_csa_uniform_query_rows(attn_metadata, base_spec)
    sample_device = torch.device("npu", device)
    native = bind_decode_csa_native_metadata(attn_metadata, base_spec, device=sample_device)
    weights = prepared_weights or pack_decode_csa_weights(
        layer,
        full_rope_cos=native.full_rope_cos,
        full_rope_sin=native.full_rope_sin,
        hadamard=native.hadamard,
    )
    buckets = (base_spec.batch,) if batch_buckets is None else tuple(batch_buckets)

    # Materialize all aliases before sealing the shared runtime registry.
    cache_bindings: dict[int, PreparedDecodeCSACaches] = {}
    for batch in buckets:
        spec = _clone_spec_for_batch(base_spec, int(batch))
        cache_bindings[int(batch)] = prepare_decode_csa_caches(kv_cache, spec)

    device_owner = registry.get_or_create(
        device=device,
        runtime=runtime,
        base_spec=base_spec,
        batch_buckets=buckets,
        backend_dependencies=backend_dependencies,
    )
    owner = DecodeCSALayerOwner(
        layer=layer,
        device_owner=device_owner,
        weights=weights,
        prepared_caches=cache_bindings,
        cache_key=_cache_storage_key(kv_cache),
    )
    setattr(layer, _LAYER_DISPATCH_ATTRIBUTE, owner)
    return owner


def uninstall_pypto_dsv4_decode_csa(layer: object) -> None:
    """Detach dispatch only; process-pinned runtime and graph owners stay alive."""
    if hasattr(layer, _LAYER_DISPATCH_ATTRIBUTE):
        delattr(layer, _LAYER_DISPATCH_ATTRIBUTE)


def mark_pypto_dsv4_decode_csa_warmups_quiesced(
    *,
    device: int,
    runtime: str = TRB_RUNTIME,
    registry: DecodeCSADeviceOwnerRegistry = _PROCESS_DEVICE_OWNERS,
) -> None:
    """Record an external sync boundary without issuing a sync internally."""
    registry.mark_warmups_quiesced(device=device, runtime=runtime)


__all__ = [
    "DecodeCSADispatchError",
    "DecodeCSADeviceOwner",
    "DecodeCSADeviceOwnerRegistry",
    "DecodeCSALayerOwner",
    "install_pypto_dsv4_decode_csa",
    "mark_pypto_dsv4_decode_csa_warmups_quiesced",
    "uninstall_pypto_dsv4_decode_csa",
]
