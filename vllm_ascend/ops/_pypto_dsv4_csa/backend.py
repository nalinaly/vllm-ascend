# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Owned PyPTO L1 backend for the DeepSeek V4 decode CSA operator.

The backend deliberately separates cold-path construction from asynchronous
execution:

``compile()``
    Builds one positive-static, annotation-only artifact for a
    :class:`DecodeCSAProgramSpec`.
``bind()``
    Validates and orders one address snapshot, and retains every tensor/cache
    owner needed by taskQueue or a captured ACLGraph.
``prepare()``
    Creates exactly one device/runtime/thread-affine L1 context and prepares
    every registered specialization.  Registration is sealed at this point.
``warmup()`` / ``launch()``
    Enqueue the already-bound snapshot.  These methods do not compile, create
    tensors, copy data, allocate outputs, or synchronize the caller.

PyPTO is imported only by the default cold-path hooks.  Tests can inject all
runtime boundaries and exercise the complete ownership state machine without
an NPU or an installed PyPTO package.
"""

from __future__ import annotations

import importlib
import inspect
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum

import torch

from .contract import (
    MUTABLE_ARGUMENT_NAMES,
    OUTPUT_ARGUMENT_NAME,
    DecodeCSAProgramSpec,
)

TRB_RUNTIME = "tensormap_and_ringbuffer"
HBG_RUNTIME = "host_build_graph"
SUPPORTED_L1_RUNTIMES = frozenset({TRB_RUNTIME, HBG_RUNTIME})

EXPECTED_PARAMETER_NAMES = (
    "hidden_states",
    "wq_a",
    "wq_a_scale",
    "wq_b",
    "wq_b_scale",
    "wkv",
    "wkv_scale",
    "gamma_cq",
    "gamma_ckv",
    "freqs_cos",
    "freqs_sin",
    "cmp_wkv",
    "cmp_wgate",
    "cmp_ape",
    "cmp_norm_w",
    "compress_state",
    "compress_state_block_table",
    "idx_wq_b",
    "idx_wq_b_scale",
    "weights_proj",
    "hadamard_idx",
    "inner_wkv",
    "inner_wgate",
    "inner_ape",
    "inner_norm_w",
    "inner_compress_state",
    "inner_compress_state_block_table",
    "kv_cache",
    "cmp_kv",
    "cmp_block_table",
    "idx_kv_cache",
    "idx_kv_scale",
    "idx_block_table",
    "swa_block_table",
    "start_positions",
    "kv_seq_lens",
    "attn_sink",
    "wo_a",
    "wo_b",
    OUTPUT_ARGUMENT_NAME,
    "main_state_page_stride",
    "inner_state_page_stride",
    "indexer_k_page_stride",
    "indexer_scale_page_stride",
)
EXPECTED_OUTPUT_INDICES = (39,)
EXPECTED_SCALAR_NAMES = EXPECTED_PARAMETER_NAMES[-4:]
EXPECTED_TENSOR_COUNT = 40
EXPECTED_SCALAR_COUNT = 4
EXPECTED_ABI_PARAMETER_COUNT = EXPECTED_TENSOR_COUNT + EXPECTED_SCALAR_COUNT

_UINT32_MAX = 2**32 - 1

# HBG keeps the complete host-built graph resident in a non-recycling ring and
# snapshots that ring into every CANN-owned launch/captured node.  Leaving this
# unset selects Simpler's historical 16384-task L2 capacity; for decode CSA that
# turns a small graph into an approximately 93 MB HostArgs package which the
# AICPU restores on every ACLGraph replay.  The current B4 artifact has 38
# top-level submits (42 generated kernel binaries include mixed-task AIC/AIV
# halves and are not a DAG-node count), but the HBG runtime expands the graph
# into at least 64 tasks in one resident scope;
# its safety rule is strict (``scope_tasks < task_window``), so a 64-slot window
# fails closed while appending task 64.  Device-0 validation therefore makes
# 128 the smallest proven safe power-of-two capacity.  A wider future graph
# fails closed during warmup and can opt into a larger capacity through the
# same environment override used by the generic PyPTO/Inductor path.
_HBG_L1_DEFAULT_TASK_WINDOW = 128


def _canonical_contiguous_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    """Return the exact row-major strides assumed by outlined AICore children."""
    result = [1] * len(shape)
    running = 1
    for axis in range(len(shape) - 1, -1, -1):
        result[axis] = running
        running *= shape[axis]
    return tuple(result)


class PyPTOBackendState(str, Enum):
    """Externally observable state of one L1 owner."""

    CREATED = "created"
    COMPILED = "compiled"
    PREPARED = "prepared"
    WARMED = "warmed"
    POISONED = "poisoned"
    CLOSED = "closed"


class PyPTOBackendContractError(ValueError):
    """Raised before enqueue when an artifact or binding violates the CSA ABI."""


@dataclass(frozen=True, slots=True)
class TensorBindingMetadata:
    """Framework-neutral tensor metadata used by the bind-time validator."""

    shape: tuple[int, ...]
    dtype: str
    strides: tuple[int, ...]
    device_type: str
    device_index: int


@dataclass(frozen=True, slots=True)
class PreparedDecodeCSAInvocation:
    """Immutable, capture-stable launch snapshot created outside capture.

    ``ordered_arguments`` and ``input_arguments`` are precomputed according to
    ``compiled.param_names`` and ``compiled.output_indices``.  Holding this
    object is sufficient to retain the original argument mapping, prepared
    physical-cache aliases, all tensor objects, and optional external storage
    owners.  The backend also retains every binding until successful close.
    """

    spec: DecodeCSAProgramSpec
    ordered_arguments: tuple[object, ...]
    input_arguments: tuple[object, ...]
    output_argument: object
    prepared_caches: object
    strong_owners: tuple[object, ...]
    _backend_token: object = field(repr=False)
    _record_key: tuple[object, ...] = field(repr=False)


@dataclass(frozen=True, slots=True)
class _DeviceClaim:
    runtime: str
    owner_thread: int
    owner_token: object


class DeviceRuntimeOwnerRegistry:
    """Process-lifetime guard against multiple L1 owners or runtime mixing.

    A successful claim is intentionally never released.  L1 binaries and
    captured graph-visible handles are process-pinned, so switching TRB/HBG on
    one device after ``close()`` is not a supported lifecycle.  Tests that need
    isolation inject a fresh registry; production TRB/HBG matrices use fresh
    processes.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._claims: dict[int, _DeviceClaim] = {}

    def claim(
        self,
        *,
        device: int,
        runtime: str,
        owner_thread: int,
        owner_token: object,
    ) -> None:
        with self._lock:
            existing = self._claims.get(device)
            if existing is None:
                self._claims[device] = _DeviceClaim(
                    runtime=runtime,
                    owner_thread=owner_thread,
                    owner_token=owner_token,
                )
                return
            if existing.owner_token is owner_token:
                return
            raise RuntimeError(
                f"device {device} already has a process-pinned PyPTO L1 owner "
                f"runtime={existing.runtime!r}, thread={existing.owner_thread}; "
                f"requested runtime={runtime!r}, thread={owner_thread}. "
                "TRB/HBG and distinct backend owners must use fresh processes."
            )


_PROCESS_DEVICE_OWNERS = DeviceRuntimeOwnerRegistry()


def _default_make_program(spec: DecodeCSAProgramSpec, runtime: str) -> object:
    # Importing kernel.py imports PyPTO; keep that dependency on compile's cold path.
    from .kernel import make_decode_csa_l1_program

    return make_decode_csa_l1_program(spec, runtime)


def _default_compile_program(program: object, device: int, runtime: str) -> object:
    # Annotation-only compile: no sample tensor or scalar is accepted by this hook.
    from pypto.runtime import RunConfig

    config = RunConfig(platform="a2a3", device_id=device, runtime=runtime)
    return program.compile(config=config)


def _default_create_context(
    programs: Sequence[object],
    device: int,
    use_task_queue: bool,
) -> object:
    from pypto.runtime.l1 import L1Config, pypto_init

    runtime_names = {getattr(program, "runtime_name", None) for program in programs}
    if len(runtime_names) != 1:
        raise RuntimeError(f"decode CSA L1 context requires one artifact runtime, got {runtime_names!r}")
    runtime = next(iter(runtime_names))
    ring_task_window = _HBG_L1_DEFAULT_TASK_WINDOW if runtime == HBG_RUNTIME else None
    return pypto_init(
        programs=programs,
        device=device,
        config=L1Config(
            ring_task_window=ring_task_window,
            use_task_queue=use_task_queue,
        ),
    )


def _default_current_device() -> int:
    torch_npu = importlib.import_module("torch_npu")
    return int(torch_npu.npu.current_device())


def _canonical_dtype(dtype: object) -> str:
    name = str(dtype).lower()
    aliases = {
        "torch.bfloat16": "bfloat16",
        "torch.float16": "fp16",
        "torch.half": "fp16",
        "torch.float32": "fp32",
        "torch.float": "fp32",
        "torch.float64": "fp64",
        "torch.double": "fp64",
        "torch.int8": "int8",
        "torch.int16": "int16",
        "torch.int32": "int32",
        "torch.int64": "int64",
        "torch.uint8": "uint8",
        "torch.bool": "bool",
    }
    return aliases.get(name, name)


def _default_inspect_tensor(value: object) -> TensorBindingMetadata:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"expected torch.Tensor, got {type(value).__name__}")
    device_type = str(value.device.type)
    device_index = value.device.index
    if device_type == "npu" and device_index is None:
        device_index = value.get_device()
    return TensorBindingMetadata(
        shape=tuple(int(extent) for extent in value.shape),
        dtype=_canonical_dtype(value.dtype),
        strides=tuple(int(stride) for stride in value.stride()),
        device_type=device_type,
        device_index=-1 if device_index is None else int(device_index),
    )


@dataclass(frozen=True, slots=True)
class PyPTOBackendDependencies:
    """Injectable cold/runtime boundaries for Host-only state-machine tests."""

    make_program: Callable[[DecodeCSAProgramSpec, str], object] = _default_make_program
    compile_program: Callable[[object, int, str], object] = _default_compile_program
    create_context: Callable[[Sequence[object], int, bool], object] = _default_create_context
    current_device: Callable[[], int] = _default_current_device
    inspect_tensor: Callable[[object], TensorBindingMetadata] = _default_inspect_tensor
    owner_registry: DeviceRuntimeOwnerRegistry = _PROCESS_DEVICE_OWNERS
    thread_ident: Callable[[], int] = threading.get_ident


@dataclass(slots=True)
class _ProgramRecord:
    spec: DecodeCSAProgramSpec
    source_program: object
    compiled: object
    param_infos: tuple[object, ...]
    output_indices: tuple[int, ...]
    operator: object | None = None
    bound_tensor_layout: tuple[TensorBindingMetadata, ...] | None = None
    warmed: bool = False


def _direction_name(direction: object) -> str:
    name = getattr(direction, "name", None)
    if name is None:
        name = str(direction).rsplit(".", maxsplit=1)[-1]
    return str(name).lower()


def _metadata_shape(info: object) -> tuple[int, ...] | None:
    shape = getattr(info, "shape", None)
    if shape is None:
        return None
    try:
        return tuple(int(extent) for extent in shape)
    except (TypeError, ValueError) as error:
        raise PyPTOBackendContractError(
            f"parameter {getattr(info, 'name', '<unknown>')!r} has non-integral metadata shape {shape!r}"
        ) from error


def _validate_source_annotations(program: object, param_infos: tuple[object, ...]) -> None:
    raw_function = getattr(program, "_func", None)
    if raw_function is None:
        return
    parameters = tuple(inspect.signature(raw_function).parameters.values())
    source_names = tuple(parameter.name for parameter in parameters)
    if source_names != EXPECTED_PARAMETER_NAMES:
        raise PyPTOBackendContractError(
            "source program parameter order does not match the decode CSA ABI: "
            f"expected={EXPECTED_PARAMETER_NAMES!r}, got={source_names!r}"
        )
    for parameter, info in zip(parameters, param_infos, strict=True):
        annotation = parameter.annotation
        annotation_shape = getattr(annotation, "shape", None)
        compiled_shape = _metadata_shape(info)
        if annotation_shape is None:
            if compiled_shape is not None:
                raise PyPTOBackendContractError(
                    f"{parameter.name!r} is scalar in the source but tensor in compiled metadata"
                )
        else:
            try:
                expected_shape = tuple(int(extent) for extent in annotation_shape)
            except (TypeError, ValueError) as error:
                raise PyPTOBackendContractError(
                    f"{parameter.name!r} retains a dynamic/non-integral source shape {annotation_shape!r}"
                ) from error
            if expected_shape != compiled_shape:
                raise PyPTOBackendContractError(
                    f"{parameter.name!r} compiled shape differs from its static annotation: "
                    f"source={expected_shape}, compiled={compiled_shape}"
                )
        annotation_dtype = getattr(annotation, "dtype", None)
        if annotation_dtype is not None and _canonical_dtype(annotation_dtype) != _canonical_dtype(
            getattr(info, "dtype", None)
        ):
            raise PyPTOBackendContractError(
                f"{parameter.name!r} compiled dtype differs from its source annotation: "
                f"source={annotation_dtype}, compiled={getattr(info, 'dtype', None)}"
            )


def _validate_compiled_program(
    compiled: object,
    source_program: object,
    *,
    runtime: str,
) -> tuple[tuple[object, ...], tuple[int, ...]]:
    param_names = tuple(getattr(compiled, "param_names", ()))
    if param_names != EXPECTED_PARAMETER_NAMES:
        raise PyPTOBackendContractError(
            "compiled parameter order does not match the "
            f"{EXPECTED_ABI_PARAMETER_COUNT}-slot decode CSA ABI: "
            f"expected={EXPECTED_PARAMETER_NAMES!r}, got={param_names!r}"
        )
    if len(param_names) != EXPECTED_ABI_PARAMETER_COUNT:
        raise PyPTOBackendContractError(
            f"decode CSA requires {EXPECTED_ABI_PARAMETER_COUNT} parameters, got {len(param_names)}"
        )

    metadata_method = getattr(compiled, "_get_metadata", None)
    if not callable(metadata_method):
        raise PyPTOBackendContractError("compiled program does not expose _get_metadata()")
    param_infos_raw, metadata_output_indices, _ = metadata_method()
    param_infos = tuple(param_infos_raw)
    output_indices = tuple(int(index) for index in metadata_output_indices)
    public_output_indices = tuple(int(index) for index in getattr(compiled, "output_indices", ()))
    if len(param_infos) != EXPECTED_ABI_PARAMETER_COUNT:
        raise PyPTOBackendContractError(
            f"compiled metadata requires {EXPECTED_ABI_PARAMETER_COUNT} entries, got {len(param_infos)}"
        )
    metadata_names = tuple(getattr(info, "name", None) for info in param_infos)
    if metadata_names != param_names:
        raise PyPTOBackendContractError(
            f"compiled.param_names and metadata order differ: {param_names!r} vs {metadata_names!r}"
        )
    if output_indices != EXPECTED_OUTPUT_INDICES or public_output_indices != EXPECTED_OUTPUT_INDICES:
        raise PyPTOBackendContractError(
            "decode CSA must expose only attn_out as pure output at index "
            f"{EXPECTED_OUTPUT_INDICES[0]}: "
            f"metadata={output_indices!r}, public={public_output_indices!r}"
        )

    tensor_infos = tuple(info for info in param_infos if _metadata_shape(info) is not None)
    scalar_infos = tuple(info for info in param_infos if _metadata_shape(info) is None)
    if len(tensor_infos) != EXPECTED_TENSOR_COUNT or len(scalar_infos) != EXPECTED_SCALAR_COUNT:
        raise PyPTOBackendContractError(
            "decode CSA L1 ABI must contain "
            f"{EXPECTED_TENSOR_COUNT} tensors and {EXPECTED_SCALAR_COUNT} scalars: "
            f"got tensors={len(tensor_infos)}, scalars={len(scalar_infos)}"
        )
    scalar_names = tuple(getattr(info, "name", None) for info in scalar_infos)
    if scalar_names != EXPECTED_SCALAR_NAMES:
        raise PyPTOBackendContractError(
            f"unexpected scalar ABI order: expected={EXPECTED_SCALAR_NAMES!r}, got={scalar_names!r}"
        )

    for index, info in enumerate(param_infos):
        name = param_names[index]
        shape = _metadata_shape(info)
        direction = _direction_name(getattr(info, "direction", ""))
        expected_direction = (
            "out" if name == OUTPUT_ARGUMENT_NAME else "inout" if name in MUTABLE_ARGUMENT_NAMES else "in"
        )
        if direction != expected_direction:
            raise PyPTOBackendContractError(
                f"{name!r} requires direction {expected_direction}, got {direction or '<missing>'}"
            )
        if shape is not None:
            if not shape or any(extent <= 0 for extent in shape):
                raise PyPTOBackendContractError(f"{name!r} requires a positive static tensor shape, got {shape!r}")
        elif _canonical_dtype(getattr(info, "dtype", None)) != "index":
            raise PyPTOBackendContractError(
                f"runtime scalar {name!r} must use index dtype, got {getattr(info, 'dtype', None)!r}"
            )

    compiled_runtime = getattr(compiled, "runtime_name", None)
    if compiled_runtime != runtime:
        raise PyPTOBackendContractError(
            f"compiled runtime mismatch: requested={runtime!r}, artifact={compiled_runtime!r}"
        )
    if getattr(compiled, "platform", None) != "a2a3":
        raise PyPTOBackendContractError(
            f"decode CSA L1 supports only a2a3 artifacts, got {getattr(compiled, 'platform', None)!r}"
        )
    runtime_config = getattr(compiled, "runtime_config", {})
    if callable(getattr(runtime_config, "get", None)) and bool(runtime_config.get("enable_sdma", False)):
        raise PyPTOBackendContractError("decode CSA L1 does not accept SDMA-enabled artifacts")

    _validate_source_annotations(source_program, param_infos)
    return param_infos, output_indices


class PyPTODSABackend:
    """One non-concurrent, process-pinned PyPTO L1 owner for decode CSA."""

    def __init__(
        self,
        *,
        device: int,
        runtime: str = TRB_RUNTIME,
        dependencies: PyPTOBackendDependencies | None = None,
    ) -> None:
        if isinstance(device, bool) or not isinstance(device, int):
            raise TypeError("device must be a non-bool integer")
        if device < 0:
            raise ValueError("device must be non-negative")
        if runtime not in SUPPORTED_L1_RUNTIMES:
            raise ValueError(
                f"unsupported PyPTO L1 runtime {runtime!r}; expected one of {sorted(SUPPORTED_L1_RUNTIMES)}"
            )
        self._dependencies = dependencies or PyPTOBackendDependencies()
        self._device = device
        self._runtime = runtime
        self._owner_thread = int(self._dependencies.thread_ident())
        self._owner_token = object()
        self._records: dict[tuple[object, ...], _ProgramRecord] = {}
        self._bindings: list[PreparedDecodeCSAInvocation] = []
        self._context: object | None = None
        self._prepared = False
        self._poisoned = False
        self._closed = False
        self._runtime_claimed = False
        self._invoke_lock = threading.Lock()

    @property
    def device(self) -> int:
        return self._device

    @property
    def runtime(self) -> str:
        return self._runtime

    @property
    def state(self) -> PyPTOBackendState:
        if self._closed:
            return PyPTOBackendState.CLOSED
        if self._poisoned:
            return PyPTOBackendState.POISONED
        if self._records and all(record.warmed for record in self._records.values()):
            return PyPTOBackendState.WARMED
        if self._prepared:
            return PyPTOBackendState.PREPARED
        if self._records:
            return PyPTOBackendState.COMPILED
        return PyPTOBackendState.CREATED

    @property
    def registered_specs(self) -> tuple[DecodeCSAProgramSpec, ...]:
        return tuple(record.spec for record in self._records.values())

    @property
    def prepared(self) -> bool:
        return self._prepared

    @property
    def closed(self) -> bool:
        return self._closed

    def _check_owner_thread(self) -> None:
        current = int(self._dependencies.thread_ident())
        if current != self._owner_thread:
            raise RuntimeError(
                f"PyPTO decode CSA backend is thread-affine: owner={self._owner_thread}, caller={current}"
            )

    def _check_current_device(self) -> None:
        current = int(self._dependencies.current_device())
        if current != self._device:
            raise RuntimeError(
                f"PyPTO decode CSA backend borrows current NPU device {self._device}, but current device is {current}"
            )

    def _check_usable(self) -> None:
        self._check_owner_thread()
        if self._closed:
            raise RuntimeError("PyPTO decode CSA backend is closed")
        if self._poisoned:
            raise RuntimeError("PyPTO decode CSA backend is poisoned; only close() may be retried")

    def compile(self, spec: DecodeCSAProgramSpec) -> object:
        """Compile and register one static spec without sample arguments."""
        self._check_usable()
        if self._prepared or self._context is not None:
            raise RuntimeError("static decode CSA registry is sealed after prepare()")
        if not isinstance(spec, DecodeCSAProgramSpec):
            raise TypeError("spec must be DecodeCSAProgramSpec")
        existing = self._records.get(spec.key)
        if existing is not None:
            return existing.compiled

        source_program = self._dependencies.make_program(spec, self._runtime)
        compiled = self._dependencies.compile_program(source_program, self._device, self._runtime)
        param_infos, output_indices = _validate_compiled_program(
            compiled,
            source_program,
            runtime=self._runtime,
        )
        self._records[spec.key] = _ProgramRecord(
            spec=spec,
            source_program=source_program,
            compiled=compiled,
            param_infos=param_infos,
            output_indices=output_indices,
        )
        return compiled

    def _record_for(self, spec: DecodeCSAProgramSpec) -> _ProgramRecord:
        record = self._records.get(spec.key)
        if record is None:
            raise RuntimeError("decode CSA specialization is not compiled; call backend.compile(spec) outside capture")
        return record

    @staticmethod
    def _merge_cache_arguments(
        arguments: Mapping[str, object],
        prepared_caches: object,
    ) -> dict[str, object]:
        cache_spec = getattr(prepared_caches, "spec", None)
        cache_mapping_method = getattr(prepared_caches, "for_launch", None)
        if cache_spec is None or not callable(cache_mapping_method):
            raise TypeError("prepared_caches must expose spec and for_launch()")
        cache_mapping = cache_mapping_method()
        if not isinstance(cache_mapping, Mapping):
            raise TypeError("prepared_caches.for_launch() must return a mapping")
        merged = dict(arguments)
        for name, value in cache_mapping.items():
            if name in merged:
                existing = merged[name]
                same_value = existing is value
                if isinstance(existing, int) and not isinstance(existing, bool) and isinstance(value, int):
                    same_value = existing == value
                if not same_value:
                    raise PyPTOBackendContractError(f"argument {name!r} conflicts with the prepared cache binding")
            merged[name] = value
        return merged

    def bind(
        self,
        spec: DecodeCSAProgramSpec,
        arguments: Mapping[str, object],
        *,
        prepared_caches: object,
        owners: Sequence[object] = (),
        retain: bool = True,
    ) -> PreparedDecodeCSAInvocation:
        """Validate, order, and strongly retain one launch address snapshot.

        This method may create Python metadata but never copies or materializes
        a tensor.  ``retain=True`` is for captured graph nodes whose address
        snapshot must stay pinned until context close.  Ordinary eager and
        warmup calls must use ``retain=False``: the PyTorch taskQueue adapter
        leases the tensors for the asynchronous enqueue, so retaining every
        freshly allocated output here would leak allocator storage.
        """
        self._check_usable()
        self._check_current_device()
        if not isinstance(arguments, Mapping):
            raise TypeError("arguments must be a name-to-value mapping")
        if not isinstance(retain, bool):
            raise TypeError("retain must be bool")
        record = self._record_for(spec)
        cache_spec = getattr(prepared_caches, "spec", None)
        if not isinstance(cache_spec, DecodeCSAProgramSpec) or cache_spec.key != spec.key:
            raise PyPTOBackendContractError("prepared cache spec does not match the requested program spec")
        merged = self._merge_cache_arguments(arguments, prepared_caches)
        expected_names = tuple(record.compiled.param_names)
        missing = tuple(name for name in expected_names if name not in merged)
        extras = tuple(name for name in merged if name not in set(expected_names))
        if missing or extras:
            raise PyPTOBackendContractError(
                f"decode CSA binding keys do not match the compiled ABI: missing={missing}, extras={extras}"
            )

        ordered_arguments = tuple(merged[name] for name in expected_names)
        expected_page_strides = {
            "main_state_page_stride": spec.physical_layout.main_state_strides[0],
            "inner_state_page_stride": spec.physical_layout.inner_state_strides[0],
            "indexer_k_page_stride": spec.physical_layout.indexer_k_strides[0],
            "indexer_scale_page_stride": spec.physical_layout.indexer_scale_strides[0],
        }
        tensor_layout: list[TensorBindingMetadata] = []
        for info, value in zip(record.param_infos, ordered_arguments, strict=True):
            name = info.name
            expected_shape = _metadata_shape(info)
            if expected_shape is None:
                if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                    raise PyPTOBackendContractError(
                        f"runtime scalar {name!r} must be a positive non-bool integer, got {value!r}"
                    )
                expected_scalar = expected_page_strides[name]
                if value != expected_scalar:
                    raise PyPTOBackendContractError(
                        f"runtime scalar {name!r} must match the static physical layout: "
                        f"expected={expected_scalar}, got={value}"
                    )
                continue
            try:
                metadata = self._dependencies.inspect_tensor(value)
            except (TypeError, ValueError) as error:
                raise PyPTOBackendContractError(f"invalid tensor argument {name!r}: {error}") from error
            if metadata.device_type != "npu" or metadata.device_index != self._device:
                raise PyPTOBackendContractError(
                    f"tensor {name!r} must be on npu:{self._device}, got {metadata.device_type}:{metadata.device_index}"
                )
            if metadata.shape != expected_shape:
                raise PyPTOBackendContractError(
                    f"tensor {name!r} shape mismatch: expected={expected_shape}, got={metadata.shape}"
                )
            expected_dtype = _canonical_dtype(getattr(info, "dtype", None))
            if _canonical_dtype(metadata.dtype) != expected_dtype:
                raise PyPTOBackendContractError(
                    f"tensor {name!r} dtype mismatch: expected={expected_dtype}, got={metadata.dtype}"
                )
            if len(metadata.strides) != len(metadata.shape) or any(
                stride <= 0 or stride > _UINT32_MAX for stride in metadata.strides
            ):
                raise PyPTOBackendContractError(
                    f"tensor {name!r} requires positive uint32 strides, got {metadata.strides}"
                )
            canonical_strides = _canonical_contiguous_strides(metadata.shape)
            if metadata.strides != canonical_strides:
                raise PyPTOBackendContractError(
                    f"tensor {name!r} must use canonical contiguous strides because "
                    "outlined AICore children do not consume arbitrary runtime Tensor strides: "
                    f"expected={canonical_strides}, got={metadata.strides}"
                )
            tensor_layout.append(metadata)

        layout_snapshot = tuple(tensor_layout)
        if record.bound_tensor_layout is None:
            record.bound_tensor_layout = layout_snapshot
        elif record.bound_tensor_layout != layout_snapshot:
            raise PyPTOBackendContractError(
                "tensor shape/dtype/stride family changed for an already-bound static specialization"
            )

        output_set = set(record.output_indices)
        input_arguments = tuple(value for index, value in enumerate(ordered_arguments) if index not in output_set)
        output_argument = ordered_arguments[record.output_indices[0]]
        binding = PreparedDecodeCSAInvocation(
            spec=spec,
            ordered_arguments=ordered_arguments,
            input_arguments=input_arguments,
            output_argument=output_argument,
            prepared_caches=prepared_caches,
            strong_owners=(arguments, prepared_caches, *tuple(owners)),
            _backend_token=self._owner_token,
            _record_key=spec.key,
        )
        if retain:
            # Captured graph nodes may outlive the Python caller's local
            # binding.  Their address snapshots remain pinned until close.
            self._bindings.append(binding)
        return binding

    def prepare(self) -> None:
        """Create the default-taskQueue context and asynchronously prepare it."""
        self._check_usable()
        self._check_current_device()
        if self._prepared:
            return
        if not self._records:
            raise RuntimeError("compile at least one decode CSA specialization before prepare()")

        if not self._runtime_claimed:
            self._dependencies.owner_registry.claim(
                device=self._device,
                runtime=self._runtime,
                owner_thread=self._owner_thread,
                owner_token=self._owner_token,
            )
            self._runtime_claimed = True

        programs = tuple(record.compiled for record in self._records.values())
        context: object | None = None
        try:
            # Production never falls back to use_task_queue=False.
            context = self._dependencies.create_context(programs, self._device, True)
            self._context = context
            for record in self._records.values():
                record.operator = context.operator(record.compiled)
            context.prepare()
        except BaseException as error:
            cleanup_context = getattr(error, "cleanup_context", None)
            if cleanup_context is not None:
                self._context = cleanup_context
            elif context is not None:
                self._context = context
            self._poisoned = True
            raise
        self._prepared = True

    def _validate_binding(self, binding: PreparedDecodeCSAInvocation) -> _ProgramRecord:
        if not isinstance(binding, PreparedDecodeCSAInvocation) or binding._backend_token is not self._owner_token:
            raise PyPTOBackendContractError("launch binding belongs to a different backend owner")
        record = self._records.get(binding._record_key)
        if record is None or record.spec.key != binding.spec.key:
            raise PyPTOBackendContractError("launch binding refers to an unknown static specialization")
        if record.operator is None:
            raise RuntimeError("decode CSA specialization is not prepared")
        return record

    def _invoke(self, binding: PreparedDecodeCSAInvocation, *, warmup: bool) -> object:
        self._check_usable()
        self._check_current_device()
        if not self._prepared:
            raise RuntimeError("call backend.prepare() outside capture before warmup/launch")
        record = self._validate_binding(binding)
        if not warmup and not record.warmed:
            raise RuntimeError(
                "decode CSA specialization is not warmed; call backend.warmup(binding) "
                "outside capture and synchronize externally before capture"
            )
        if not self._invoke_lock.acquire(blocking=False):
            raise RuntimeError("concurrent PyPTO decode CSA host invocation is unsupported")
        try:
            operator = record.operator
            assert operator is not None
            if warmup:
                result = operator.warmup(
                    *binding.input_arguments,
                    out=binding.output_argument,
                )
            else:
                result = operator(
                    *binding.input_arguments,
                    out=binding.output_argument,
                )
        except BaseException:
            # Enqueue failures can leave native ownership live.  Do not pretend
            # the owner vanished or silently create a replacement context.
            self._poisoned = True
            raise
        finally:
            self._invoke_lock.release()
        if result is not binding.output_argument:
            self._poisoned = True
            raise RuntimeError("PyPTO L1 operator did not return the bound attn_out tensor")
        if warmup:
            record.warmed = True
        return result

    def warmup(self, binding: PreparedDecodeCSAInvocation) -> object:
        """Enqueue one explicit ordinary-eager warmup; caller owns synchronization."""
        return self._invoke(binding, warmup=True)

    def launch(self, binding: PreparedDecodeCSAInvocation) -> object:
        """Enqueue one pre-bound asynchronous L1 call without cold-path work."""
        return self._invoke(binding, warmup=False)

    def close(self) -> None:
        """Retryably close the externally-quiesced context.

        The method never synchronizes and never changes device.  On failure the
        context and every tensor/cache owner remain retained so the same owner
        can be closed again after the caller fixes external quiescence.
        """
        self._check_owner_thread()
        if self._closed:
            return
        if self._context is not None:
            self._check_current_device()
            try:
                self._context.close()
            except BaseException:
                self._poisoned = True
                raise
        self._closed = True
        self._poisoned = False
        self._prepared = False
        self._context = None
        self._bindings.clear()


__all__ = [
    "DeviceRuntimeOwnerRegistry",
    "HBG_RUNTIME",
    "PreparedDecodeCSAInvocation",
    "PyPTOBackendContractError",
    "PyPTOBackendDependencies",
    "PyPTOBackendState",
    "PyPTODSABackend",
    "TRB_RUNTIME",
    "TensorBindingMetadata",
]
