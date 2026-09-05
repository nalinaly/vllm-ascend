# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Host-only orchestration model for a multi-layer decode CSA Capsule.

This module records ownership, routing, ordering, and address-binding evidence.
It intentionally performs no tensor mathematics and makes no native/PyPTO
accuracy claim. A later NPU runner can implement CapsuleExecutor while reusing
the same immutable topology and per-layer resource model.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, runtime_checkable


class CapsuleError(ValueError):
    """Raised when a Capsule topology or execution contract is invalid."""


class BackendKind(str, Enum):
    NATIVE = "native"
    PYPTO = "pypto"


class CapsuleTopology(str, Enum):
    NNNN = "NNNN"
    PPPP = "PPPP"
    NPNP = "NPNP"
    PNPN = "PNPN"
    PP_SAME = "PP_same"
    PP_DISTINCT = "PP_distinct"


class BridgeKind(str, Enum):
    IDENTITY_RESIDUAL = "identity_residual"
    LOW_RANK_FFN = "low_rank_ffn"


class CallStage(str, Enum):
    RESIDUAL_CLONE = "residual_clone"
    HC_PRE = "hc_pre"
    RMS_NORM = "rms_norm"
    DSA_FORWARD = "dsa_forward"
    HC_POST = "hc_post"
    IDENTITY_RESIDUAL_BRIDGE = "identity_residual_bridge"
    LOW_RANK_FFN_BRIDGE = "low_rank_ffn_bridge"


AddressToken = Hashable


@dataclass(frozen=True, slots=True)
class CapsuleConfig:
    """Static shape-independent orchestration choices for one Capsule."""

    num_layers: int = 4
    topology: CapsuleTopology = CapsuleTopology.PPPP
    bridge_kind: BridgeKind = BridgeKind.IDENTITY_RESIDUAL
    seed: int = 0
    layer_name_prefix: str = "model.layers"
    low_rank_ffn_rank: int = 16

    def __post_init__(self) -> None:
        try:
            topology = CapsuleTopology(self.topology)
        except ValueError as error:
            raise CapsuleError(f"unsupported Capsule topology {self.topology!r}") from error
        try:
            bridge_kind = BridgeKind(self.bridge_kind)
        except ValueError as error:
            raise CapsuleError(f"unsupported bridge kind {self.bridge_kind!r}") from error
        object.__setattr__(self, "topology", topology)
        object.__setattr__(self, "bridge_kind", bridge_kind)
        if self.num_layers not in {1, 2, 4}:
            raise CapsuleError("num_layers must be one of 1, 2, or 4")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise CapsuleError("seed must be an integer")
        if not self.layer_name_prefix:
            raise CapsuleError("layer_name_prefix must not be empty")
        if (
            isinstance(self.low_rank_ffn_rank, bool)
            or not isinstance(self.low_rank_ffn_rank, int)
            or self.low_rank_ffn_rank <= 0
        ):
            raise CapsuleError("low_rank_ffn_rank must be a positive integer")


@dataclass(frozen=True, slots=True)
class WeightOwner:
    """Immutable ownership handle; payload may later hold real NPU weights."""

    owner_id: str
    address_token: AddressToken
    layer_name: str
    payload: object | None = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        if not self.owner_id or not self.layer_name:
            raise CapsuleError("weight owner identity and layer name must not be empty")


@dataclass(frozen=True, slots=True)
class CacheMutation:
    """Host evidence that exactly one layer invocation touched an owner."""

    owner_local_invocation: int
    layer_name: str
    callable_id: str
    input_address: AddressToken


@dataclass(slots=True)
class MutableCacheOwner:
    """Per-layer mutable cache owner with an append-only Host audit trail."""

    owner_id: str
    address_token: AddressToken
    layer_name: str
    payload: object | None = field(default=None, repr=False)
    _mutations: list[CacheMutation] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.owner_id or not self.layer_name:
            raise CapsuleError("cache owner identity and layer name must not be empty")

    @property
    def mutations(self) -> tuple[CacheMutation, ...]:
        return tuple(self._mutations)

    @property
    def mutation_count(self) -> int:
        return len(self._mutations)

    def record_mutation(
        self,
        *,
        layer_name: str,
        callable_id: str,
        input_address: AddressToken,
    ) -> None:
        if layer_name != self.layer_name:
            raise CapsuleError(f"cache owner {self.owner_id} belongs to {self.layer_name}, not {layer_name}")
        self._mutations.append(
            CacheMutation(
                owner_local_invocation=len(self._mutations),
                layer_name=layer_name,
                callable_id=callable_id,
                input_address=input_address,
            )
        )


@dataclass(frozen=True, slots=True)
class CallableIdentity:
    """Logical callable identity, separate from weights and argument addresses."""

    callable_id: str
    logical_function_id: int
    backend: BackendKind
    payload: object | None = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        if not self.callable_id:
            raise CapsuleError("callable_id must not be empty")
        if (
            isinstance(self.logical_function_id, bool)
            or not isinstance(self.logical_function_id, int)
            or self.logical_function_id < 0
        ):
            raise CapsuleError("logical_function_id must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class BridgeSpec:
    """Deterministic parameters for one symbolic inter-layer bridge."""

    after_layer_index: int
    kind: BridgeKind
    bridge_id: str
    parameter_seed: int
    low_rank: int

    def __post_init__(self) -> None:
        if self.after_layer_index < 0 or not self.bridge_id:
            raise CapsuleError("bridge index and identity are invalid")
        if self.kind is BridgeKind.IDENTITY_RESIDUAL and self.low_rank != 0:
            raise CapsuleError("identity_residual bridge must not carry a low-rank width")
        if self.kind is BridgeKind.LOW_RANK_FFN and self.low_rank <= 0:
            raise CapsuleError("low_rank_ffn bridge requires a positive rank")

    @property
    def call_stage(self) -> CallStage:
        if self.kind is BridgeKind.IDENTITY_RESIDUAL:
            return CallStage.IDENTITY_RESIDUAL_BRIDGE
        return CallStage.LOW_RANK_FFN_BRIDGE


@dataclass(frozen=True, slots=True)
class CSALayerCapsule:
    """One layer shell with independent owners and a routed CSA backend."""

    layer_index: int
    layer_name: str
    backend: BackendKind
    weight_owner: WeightOwner
    cache_owner: MutableCacheOwner
    callable_identity: CallableIdentity

    def __post_init__(self) -> None:
        if self.layer_index < 0 or not self.layer_name:
            raise CapsuleError("layer index and name are invalid")
        if self.weight_owner.layer_name != self.layer_name:
            raise CapsuleError("weight owner belongs to a different layer")
        if self.cache_owner.layer_name != self.layer_name:
            raise CapsuleError("cache owner belongs to a different layer")
        if self.callable_identity.backend is not self.backend:
            raise CapsuleError("callable backend does not match the routed layer backend")


@dataclass(frozen=True, slots=True)
class HCPreResult:
    """Explicit dataflow produced by the production HC-pre boundary.

    DeepSeek V4 HC-pre does not return only the normalized-layer input.  It
    additionally produces the ``post`` and ``comb`` mixing tensors that the
    matching HC-post invocation must consume.  Keeping all three values in one
    result prevents an executor from hiding those graph-visible dependencies in
    mutable side state.
    """

    hidden: object
    post: object
    comb: object


@runtime_checkable
class CapsuleExecutor(Protocol):
    """Execution boundary implemented now by Host recording and later by NPU."""

    def address_token(self, value: object) -> AddressToken: ...

    def residual_clone(self, layer: CSALayerCapsule, value: object) -> object: ...

    def hc_pre(
        self,
        layer: CSALayerCapsule,
        value: object,
        residual: object,
    ) -> HCPreResult: ...

    def rms_norm(self, layer: CSALayerCapsule, value: object) -> object: ...

    def dsa_forward(self, layer: CSALayerCapsule, value: object) -> object: ...

    def hc_post(
        self,
        layer: CSALayerCapsule,
        value: object,
        residual: object,
        post: object,
        comb: object,
    ) -> object: ...

    def bridge(self, spec: BridgeSpec, value: object) -> object: ...


@dataclass(frozen=True, slots=True)
class HostActivation:
    """Opaque symbolic value used only for orchestration and address evidence."""

    address: str
    lineage: tuple[str, ...]


class HostRecordingExecutor:
    """Symbolic executor; it does not calculate or compare tensor values."""

    def __init__(self) -> None:
        self._next_address = 0

    def make_input(self, label: str = "input") -> HostActivation:
        if not label:
            raise CapsuleError("Host input label must not be empty")
        return self._emit((), f"input:{label}")

    def address_token(self, value: object) -> AddressToken:
        if not isinstance(value, HostActivation):
            raise TypeError("HostRecordingExecutor only accepts HostActivation values")
        return value.address

    def residual_clone(self, layer: CSALayerCapsule, value: object) -> HostActivation:
        return self._derive(value, f"{layer.layer_name}:residual_clone")

    def hc_pre(
        self,
        layer: CSALayerCapsule,
        value: object,
        residual: object,
    ) -> HCPreResult:
        self.address_token(residual)
        return HCPreResult(
            hidden=self._derive(value, f"{layer.layer_name}:hc_pre:hidden"),
            post=self._derive(value, f"{layer.layer_name}:hc_pre:post"),
            comb=self._derive(value, f"{layer.layer_name}:hc_pre:comb"),
        )

    def rms_norm(self, layer: CSALayerCapsule, value: object) -> HostActivation:
        return self._derive(value, f"{layer.layer_name}:rms_norm")

    def dsa_forward(self, layer: CSALayerCapsule, value: object) -> HostActivation:
        input_address = self.address_token(value)
        layer.cache_owner.record_mutation(
            layer_name=layer.layer_name,
            callable_id=layer.callable_identity.callable_id,
            input_address=input_address,
        )
        return self._derive(
            value,
            f"{layer.layer_name}:dsa_forward:{layer.backend.value}:{layer.callable_identity.callable_id}",
        )

    def hc_post(
        self,
        layer: CSALayerCapsule,
        value: object,
        residual: object,
        post: object,
        comb: object,
    ) -> HostActivation:
        residual_address = self.address_token(residual)
        post_address = self.address_token(post)
        comb_address = self.address_token(comb)
        return self._derive(
            value,
            (f"{layer.layer_name}:hc_post:residual={residual_address}:post={post_address}:comb={comb_address}"),
        )

    def bridge(self, spec: BridgeSpec, value: object) -> HostActivation:
        operation = f"{spec.bridge_id}:seed={spec.parameter_seed}:rank={spec.low_rank}"
        return self._derive(value, operation)

    def _derive(self, value: object, operation: str) -> HostActivation:
        if not isinstance(value, HostActivation):
            raise TypeError("HostRecordingExecutor only accepts HostActivation values")
        return self._emit(value.lineage, operation)

    def _emit(self, lineage: tuple[str, ...], operation: str) -> HostActivation:
        address = f"host-activation:{self._next_address}"
        self._next_address += 1
        return HostActivation(address=address, lineage=lineage + (operation,))


@dataclass(frozen=True, slots=True)
class CallRecord:
    """One ordered stage invocation and its immutable binding evidence."""

    ordinal: int
    stage: CallStage
    layer_index: int
    layer_name: str
    backend: BackendKind | None
    callable_id: str | None
    logical_function_id: int | None
    weight_owner_id: str | None
    cache_owner_id: str | None
    input_address: AddressToken
    output_address: AddressToken
    auxiliary_addresses: tuple[AddressToken, ...] = ()
    argument_addresses: tuple[AddressToken, ...] = ()
    bridge_id: str | None = None


@dataclass(frozen=True, slots=True)
class CapsuleRunResult:
    """Structural execution result; final_value remains executor-owned."""

    final_value: object = field(compare=False, repr=False)
    layer_outputs: tuple[object, ...] = field(compare=False, repr=False)
    call_records: tuple[CallRecord, ...]

    @property
    def csa_calls(self) -> tuple[CallRecord, ...]:
        return tuple(record for record in self.call_records if record.stage is CallStage.DSA_FORWARD)

    @property
    def bridge_calls(self) -> tuple[CallRecord, ...]:
        return tuple(
            record
            for record in self.call_records
            if record.stage
            in {
                CallStage.IDENTITY_RESIDUAL_BRIDGE,
                CallStage.LOW_RANK_FFN_BRIDGE,
            }
        )


@dataclass(frozen=True, slots=True)
class CSACapsuleRunner:
    """Reusable 1/2/4-layer orchestration graph."""

    config: CapsuleConfig
    layers: tuple[CSALayerCapsule, ...]
    bridges: tuple[BridgeSpec, ...]

    def __post_init__(self) -> None:
        if len(self.layers) != self.config.num_layers:
            raise CapsuleError("runner layer count does not match config")
        if len(self.bridges) != max(0, self.config.num_layers - 1):
            raise CapsuleError("runner must contain exactly one bridge between adjacent layers")
        if tuple(layer.layer_index for layer in self.layers) != tuple(range(self.config.num_layers)):
            raise CapsuleError("layer indices must be contiguous and zero-based")
        layer_names = tuple(layer.layer_name for layer in self.layers)
        weight_ids = tuple(layer.weight_owner.owner_id for layer in self.layers)
        cache_ids = tuple(layer.cache_owner.owner_id for layer in self.layers)
        weight_addresses = tuple(layer.weight_owner.address_token for layer in self.layers)
        cache_addresses = tuple(layer.cache_owner.address_token for layer in self.layers)
        for label, identities in (
            ("layer names", layer_names),
            ("weight owners", weight_ids),
            ("cache owners", cache_ids),
            ("weight addresses", weight_addresses),
            ("cache addresses", cache_addresses),
        ):
            if len(set(identities)) != len(identities):
                raise CapsuleError(f"{label} must be unique across layers")
        self._validate_topology()

    def run(self, initial_value: object, executor: CapsuleExecutor) -> CapsuleRunResult:
        """Execute the layer topology and retain an address-level call trace."""
        if not isinstance(executor, CapsuleExecutor):
            raise TypeError("executor does not implement the CapsuleExecutor protocol")
        records: list[CallRecord] = []
        layer_outputs: list[object] = []
        value = initial_value

        def append_record(
            *,
            stage: CallStage,
            layer: CSALayerCapsule,
            source: object,
            target: object,
            auxiliary: tuple[object, ...] = (),
            bridge_id: str | None = None,
        ) -> None:
            is_csa = stage is CallStage.DSA_FORWARD
            auxiliary_addresses = tuple(executor.address_token(item) for item in auxiliary)
            input_address = executor.address_token(source)
            output_address = executor.address_token(target)
            argument_addresses: tuple[AddressToken, ...] = ()
            if is_csa:
                argument_addresses = (
                    input_address,
                    output_address,
                    layer.weight_owner.address_token,
                    layer.cache_owner.address_token,
                )
            records.append(
                CallRecord(
                    ordinal=len(records),
                    stage=stage,
                    layer_index=layer.layer_index,
                    layer_name=layer.layer_name,
                    backend=layer.backend if is_csa else None,
                    callable_id=layer.callable_identity.callable_id if is_csa else None,
                    logical_function_id=(layer.callable_identity.logical_function_id if is_csa else None),
                    weight_owner_id=layer.weight_owner.owner_id if is_csa else None,
                    cache_owner_id=layer.cache_owner.owner_id if is_csa else None,
                    input_address=input_address,
                    output_address=output_address,
                    auxiliary_addresses=auxiliary_addresses,
                    argument_addresses=argument_addresses,
                    bridge_id=bridge_id,
                )
            )

        for layer_index, layer in enumerate(self.layers):
            residual = executor.residual_clone(layer, value)
            append_record(
                stage=CallStage.RESIDUAL_CLONE,
                layer=layer,
                source=value,
                target=residual,
            )
            hc_pre = executor.hc_pre(layer, value, residual)
            if not isinstance(hc_pre, HCPreResult):
                raise TypeError(
                    "CapsuleExecutor.hc_pre() must return HCPreResult with explicit hidden/post/comb values"
                )
            append_record(
                stage=CallStage.HC_PRE,
                layer=layer,
                source=value,
                target=hc_pre.hidden,
                auxiliary=(residual, hc_pre.post, hc_pre.comb),
            )
            normalized = executor.rms_norm(layer, hc_pre.hidden)
            append_record(
                stage=CallStage.RMS_NORM,
                layer=layer,
                source=hc_pre.hidden,
                target=normalized,
            )
            csa_output = executor.dsa_forward(layer, normalized)
            append_record(
                stage=CallStage.DSA_FORWARD,
                layer=layer,
                source=normalized,
                target=csa_output,
            )
            value = executor.hc_post(
                layer,
                csa_output,
                residual,
                hc_pre.post,
                hc_pre.comb,
            )
            append_record(
                stage=CallStage.HC_POST,
                layer=layer,
                source=csa_output,
                target=value,
                auxiliary=(residual, hc_pre.post, hc_pre.comb),
            )
            layer_outputs.append(value)

            if layer_index < len(self.bridges):
                bridge = self.bridges[layer_index]
                bridged = executor.bridge(bridge, value)
                append_record(
                    stage=bridge.call_stage,
                    layer=layer,
                    source=value,
                    target=bridged,
                    bridge_id=bridge.bridge_id,
                )
                value = bridged

        return CapsuleRunResult(
            final_value=value,
            layer_outputs=tuple(layer_outputs),
            call_records=tuple(records),
        )

    def _validate_topology(self) -> None:
        expected = topology_backends(self.config.topology, self.config.num_layers)
        actual = tuple(layer.backend for layer in self.layers)
        if actual != expected:
            raise CapsuleError(f"topology backend route mismatch: expected={expected}, actual={actual}")
        callable_ids = tuple(layer.callable_identity.callable_id for layer in self.layers)
        function_ids = tuple(layer.callable_identity.logical_function_id for layer in self.layers)
        callable_to_function: dict[str, int] = {}
        function_to_callable: dict[int, str] = {}
        for callable_id, function_id in zip(callable_ids, function_ids, strict=True):
            if callable_id in callable_to_function and callable_to_function[callable_id] != function_id:
                raise CapsuleError("one callable identity cannot map to multiple logical function IDs")
            if function_id in function_to_callable and function_to_callable[function_id] != callable_id:
                raise CapsuleError("distinct callable identities cannot alias one logical function ID")
            callable_to_function[callable_id] = function_id
            function_to_callable[function_id] = callable_id
        if self.config.topology is CapsuleTopology.PP_SAME and self.config.num_layers >= 2:
            if callable_ids[0] != callable_ids[1] or function_ids[0] != function_ids[1]:
                raise CapsuleError("PP_same requires layers 0 and 1 to share one callable identity")
            if len(set(callable_ids[1:])) != len(callable_ids[1:]):
                raise CapsuleError("PP_same requires layers 2+ to use independent callables")
        if self.config.topology is CapsuleTopology.PP_DISTINCT:
            if len(set(callable_ids)) != len(callable_ids) or len(set(function_ids)) != len(function_ids):
                raise CapsuleError("PP_distinct requires one callable identity per layer")


def topology_backends(
    topology: CapsuleTopology | str,
    num_layers: int,
) -> tuple[BackendKind, ...]:
    """Return the backend route, truncated to an allowed Capsule depth."""
    try:
        normalized = CapsuleTopology(topology)
    except ValueError as error:
        raise CapsuleError(f"unsupported Capsule topology {topology!r}") from error
    if num_layers not in {1, 2, 4}:
        raise CapsuleError("num_layers must be one of 1, 2, or 4")
    if normalized is CapsuleTopology.NNNN:
        return (BackendKind.NATIVE,) * num_layers
    if normalized in {
        CapsuleTopology.PPPP,
        CapsuleTopology.PP_SAME,
        CapsuleTopology.PP_DISTINCT,
    }:
        return (BackendKind.PYPTO,) * num_layers
    if normalized is CapsuleTopology.NPNP:
        return tuple(
            BackendKind.NATIVE if layer_index % 2 == 0 else BackendKind.PYPTO for layer_index in range(num_layers)
        )
    return tuple(BackendKind.PYPTO if layer_index % 2 == 0 else BackendKind.NATIVE for layer_index in range(num_layers))


def _default_callable_identities(config: CapsuleConfig) -> tuple[CallableIdentity, ...]:
    backends = topology_backends(config.topology, config.num_layers)
    keys: list[str] = []
    for layer_index, backend in enumerate(backends):
        if config.topology is CapsuleTopology.PP_SAME and layer_index < 2:
            key = "pypto:shared:layers-0-1"
        else:
            key = f"{backend.value}:{config.topology.value}:layer-{layer_index}"
        keys.append(key)

    identities_by_key: dict[str, CallableIdentity] = {}
    identities: list[CallableIdentity] = []
    for key, backend in zip(keys, backends, strict=True):
        if key not in identities_by_key:
            identities_by_key[key] = CallableIdentity(
                callable_id=key,
                logical_function_id=len(identities_by_key),
                backend=backend,
            )
        identities.append(identities_by_key[key])
    return tuple(identities)


def _default_bridge_specs(config: CapsuleConfig) -> tuple[BridgeSpec, ...]:
    result: list[BridgeSpec] = []
    for layer_index in range(config.num_layers - 1):
        parameter_seed = config.seed * 1_009 + layer_index * 9_173
        low_rank = config.low_rank_ffn_rank if config.bridge_kind is BridgeKind.LOW_RANK_FFN else 0
        result.append(
            BridgeSpec(
                after_layer_index=layer_index,
                kind=config.bridge_kind,
                bridge_id=f"{config.bridge_kind.value}:after-layer-{layer_index}",
                parameter_seed=parameter_seed,
                low_rank=low_rank,
            )
        )
    return tuple(result)


def build_capsule(
    config: CapsuleConfig,
    *,
    weight_owners: Sequence[WeightOwner] | None = None,
    cache_owners: Sequence[MutableCacheOwner] | None = None,
    callable_identities: Sequence[CallableIdentity] | None = None,
) -> CSACapsuleRunner:
    """Build a Capsule while allowing a later NPU harness to inject owners."""
    names = tuple(f"{config.layer_name_prefix}.{index}.dsa" for index in range(config.num_layers))
    weights = (
        tuple(weight_owners)
        if weight_owners is not None
        else tuple(
            WeightOwner(
                owner_id=f"weights:{name}",
                address_token=f"host-owner://weights/{index}",
                layer_name=name,
            )
            for index, name in enumerate(names)
        )
    )
    caches = (
        tuple(cache_owners)
        if cache_owners is not None
        else tuple(
            MutableCacheOwner(
                owner_id=f"cache:{name}",
                address_token=f"host-owner://cache/{index}",
                layer_name=name,
            )
            for index, name in enumerate(names)
        )
    )
    callables = tuple(callable_identities) if callable_identities is not None else _default_callable_identities(config)
    for label, values in (
        ("weight_owners", weights),
        ("cache_owners", caches),
        ("callable_identities", callables),
    ):
        if len(values) != config.num_layers:
            raise CapsuleError(f"{label} must contain exactly {config.num_layers} entries")
    backends = topology_backends(config.topology, config.num_layers)
    layers = tuple(
        CSALayerCapsule(
            layer_index=index,
            layer_name=names[index],
            backend=backends[index],
            weight_owner=weights[index],
            cache_owner=caches[index],
            callable_identity=callables[index],
        )
        for index in range(config.num_layers)
    )
    return CSACapsuleRunner(
        config=config,
        layers=layers,
        bridges=_default_bridge_specs(config),
    )


__all__ = [
    "AddressToken",
    "BackendKind",
    "BridgeKind",
    "BridgeSpec",
    "CSACapsuleRunner",
    "CSALayerCapsule",
    "CacheMutation",
    "CallRecord",
    "CallStage",
    "CallableIdentity",
    "CapsuleConfig",
    "CapsuleError",
    "CapsuleExecutor",
    "CapsuleRunResult",
    "CapsuleTopology",
    "HCPreResult",
    "HostActivation",
    "HostRecordingExecutor",
    "MutableCacheOwner",
    "WeightOwner",
    "build_capsule",
    "topology_backends",
]
