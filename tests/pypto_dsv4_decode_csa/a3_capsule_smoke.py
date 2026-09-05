# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Production custom-op A3 runner for a 1/2/4-layer PyPTO CSA Capsule.

The Host-visible plan mirrors :mod:`tests.pypto_dsv4_decode_csa.capsule` and is
safe to import without an NPU.  The execution entry lazily imports ``torch_npu``
and assembles every callable in one process-pinned backend/context.  This is
important for ``PP_distinct``: distinct static callable identities must not be
faked with multiple competing device runtime owners.

The current runner intentionally uses a small deterministic Torch residual and
bridge instead of claiming fidelity with the model's HC/MoE shell.  Its purpose
is stream ordering, address/owner isolation, repeated-callable binding, and a
single mixed Torch/PyPTO ACLGraph.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

import torch

from tests.pypto_dsv4_decode_csa.capsule import (
    BridgeKind,
    CapsuleConfig,
    CapsuleTopology,
    build_capsule,
)
from vllm_ascend.ops._pypto_dsv4_csa import DecodeCSAProgramSpec

_SUPPORTED_RUNTIMES = frozenset({"tensormap_and_ringbuffer", "host_build_graph"})
_SUPPORTED_TOPOLOGIES = frozenset(
    {
        CapsuleTopology.PPPP,
        CapsuleTopology.PP_SAME,
        CapsuleTopology.PP_DISTINCT,
    }
)
_METADATA_ARGUMENT_NAMES = (
    "compress_state_block_table",
    "inner_compress_state_block_table",
    "cmp_block_table",
    "idx_block_table",
    "swa_block_table",
    "start_positions",
    "kv_seq_lens",
)


@dataclass(frozen=True, slots=True)
class A3CapsuleSmokeConfig:
    runtime: str
    device: int = 0
    batch: int = 4
    num_layers: int = 4
    topology: CapsuleTopology = CapsuleTopology.PPPP
    seed: int = 17
    layer_name_prefix: str = "pypto.capsule.layers"
    replay_values: tuple[float, ...] = (0.25, -0.5, 1.0)

    def __post_init__(self) -> None:
        if self.runtime not in _SUPPORTED_RUNTIMES:
            raise ValueError(f"unsupported runtime {self.runtime!r}; expected {sorted(_SUPPORTED_RUNTIMES)}")
        if isinstance(self.device, bool) or not isinstance(self.device, int) or self.device < 0:
            raise ValueError("device must be a non-negative integer")
        try:
            topology = CapsuleTopology(self.topology)
        except ValueError as error:
            raise ValueError(f"unsupported Capsule topology {self.topology!r}") from error
        if topology not in _SUPPORTED_TOPOLOGIES:
            raise ValueError(f"A3 PyPTO Capsule runner does not support topology {topology.value}")
        object.__setattr__(self, "topology", topology)
        # Reuse the canonical depth/seed/name validation from the Host model.
        CapsuleConfig(
            num_layers=self.num_layers,
            topology=topology,
            bridge_kind=BridgeKind.IDENTITY_RESIDUAL,
            seed=self.seed,
            layer_name_prefix=self.layer_name_prefix,
        )
        DecodeCSAProgramSpec(batch=self.batch)
        if not self.replay_values:
            raise ValueError("replay_values must not be empty")
        if any(not isinstance(value, (int, float)) or isinstance(value, bool) for value in self.replay_values):
            raise TypeError("replay_values must contain only real scalars")
        if any(not math.isfinite(float(value)) for value in self.replay_values):
            raise ValueError("replay_values must be finite")


@dataclass(frozen=True, slots=True)
class A3CapsuleLayerPlan:
    layer_index: int
    layer_name: str
    callable_group: int
    callable_id: str
    spec: DecodeCSAProgramSpec
    weight_owner_token: str
    cache_owner_token: str
    bridge_scale: float | None
    bridge_bias: float | None

    def to_dict(self) -> dict[str, object]:
        return {
            "layer_index": self.layer_index,
            "layer_name": self.layer_name,
            "callable_group": self.callable_group,
            "callable_id": self.callable_id,
            "spec_key": repr(self.spec.key),
            "weight_owner_token": self.weight_owner_token,
            "cache_owner_token": self.cache_owner_token,
            "bridge_scale": self.bridge_scale,
            "bridge_bias": self.bridge_bias,
        }


@dataclass(frozen=True, slots=True)
class A3CapsuleExecutionPlan:
    config: A3CapsuleSmokeConfig
    layers: tuple[A3CapsuleLayerPlan, ...]

    @property
    def compiled_callable_count(self) -> int:
        return len({layer.callable_group for layer in self.layers})

    def to_dict(self) -> dict[str, object]:
        return {
            "runtime": self.config.runtime,
            "device": self.config.device,
            "batch": self.config.batch,
            "num_layers": self.config.num_layers,
            "topology": self.config.topology.value,
            "compiled_callable_count": self.compiled_callable_count,
            "layers": tuple(layer.to_dict() for layer in self.layers),
        }


def _spec_for_callable_group(base: DecodeCSAProgramSpec, callable_group: int) -> DecodeCSAProgramSpec:
    """Give each logical callable a harmless but real static ABI identity."""
    if callable_group == 0:
        return base
    return replace(
        base,
        swa_blocks=base.swa_blocks + callable_group,
        compressed_blocks=base.compressed_blocks + callable_group,
        main_state_blocks=base.main_state_blocks + callable_group,
        inner_state_blocks=base.inner_state_blocks + callable_group,
        indexer_blocks=base.indexer_blocks + callable_group,
    )


def build_a3_capsule_execution_plan(config: A3CapsuleSmokeConfig) -> A3CapsuleExecutionPlan:
    """Build the Host-auditable ownership and callable plan."""
    capsule = build_capsule(
        CapsuleConfig(
            num_layers=config.num_layers,
            topology=config.topology,
            bridge_kind=BridgeKind.IDENTITY_RESIDUAL,
            seed=config.seed,
            layer_name_prefix=config.layer_name_prefix,
        )
    )
    base = DecodeCSAProgramSpec(batch=config.batch)
    layers: list[A3CapsuleLayerPlan] = []
    for layer in capsule.layers:
        index = layer.layer_index
        has_bridge = index + 1 < config.num_layers
        # Scalars are exactly representable binary fractions, keeping eager and
        # graph comparison deterministic in BF16.
        bridge_scale = 1.0 + (index + 1) / 16.0 if has_bridge else None
        bridge_bias = ((config.seed + index * 13) % 9 - 4) / 32.0 if has_bridge else None
        callable_group = layer.callable_identity.logical_function_id
        layers.append(
            A3CapsuleLayerPlan(
                layer_index=index,
                layer_name=layer.layer_name,
                callable_group=callable_group,
                callable_id=layer.callable_identity.callable_id,
                spec=_spec_for_callable_group(base, callable_group),
                weight_owner_token=layer.weight_owner.owner_id,
                cache_owner_token=layer.cache_owner.owner_id,
                bridge_scale=bridge_scale,
                bridge_bias=bridge_bias,
            )
        )
    plan = A3CapsuleExecutionPlan(config=config, layers=tuple(layers))
    _validate_plan(plan)
    return plan


def _validate_plan(plan: A3CapsuleExecutionPlan) -> None:
    if len(plan.layers) != plan.config.num_layers:
        raise ValueError("Capsule plan layer count does not match its config")
    if tuple(layer.layer_index for layer in plan.layers) != tuple(range(plan.config.num_layers)):
        raise ValueError("Capsule plan layer indices must be contiguous")
    for field in ("layer_name", "weight_owner_token", "cache_owner_token"):
        values = tuple(getattr(layer, field) for layer in plan.layers)
        if len(set(values)) != len(values):
            raise ValueError(f"Capsule plan {field} values must be unique")
    group_to_key: dict[int, tuple[object, ...]] = {}
    key_to_group: dict[tuple[object, ...], int] = {}
    for layer in plan.layers:
        key = layer.spec.key
        existing_key = group_to_key.setdefault(layer.callable_group, key)
        if existing_key != key:
            raise ValueError("one callable group maps to multiple static specs")
        existing_group = key_to_group.setdefault(key, layer.callable_group)
        if existing_group != layer.callable_group:
            raise ValueError("distinct callable groups alias one static spec")


@dataclass(frozen=True, slots=True)
class A3CapsuleLayerAddressEvidence:
    layer_index: int
    layer_name: str
    callable_group: int
    fixture_owner_id: int
    weight_owner_id: int
    cache_owner_id: int
    layer_owner_id: int
    device_owner_id: int
    backend_owner_id: int
    context_owner_id: int
    weight_addresses: tuple[int, ...]
    metadata_addresses: tuple[int, ...]
    cache_addresses: tuple[int, ...]
    eager_input_address: int
    eager_output_address: int
    capture_input_address: int
    capture_output_address: int

    def to_dict(self) -> dict[str, object]:
        return {
            "layer_index": self.layer_index,
            "layer_name": self.layer_name,
            "callable_group": self.callable_group,
            "fixture_owner_id": self.fixture_owner_id,
            "weight_owner_id": self.weight_owner_id,
            "cache_owner_id": self.cache_owner_id,
            "layer_owner_id": self.layer_owner_id,
            "device_owner_id": self.device_owner_id,
            "backend_owner_id": self.backend_owner_id,
            "context_owner_id": self.context_owner_id,
            "weight_addresses": self.weight_addresses,
            "metadata_addresses": self.metadata_addresses,
            "cache_addresses": self.cache_addresses,
            "eager_input_address": self.eager_input_address,
            "eager_output_address": self.eager_output_address,
            "capture_input_address": self.capture_input_address,
            "capture_output_address": self.capture_output_address,
        }


def validate_a3_capsule_resource_independence(
    plan: A3CapsuleExecutionPlan,
    evidence: Sequence[A3CapsuleLayerAddressEvidence],
) -> None:
    """Fail if any layer-local owner or live tensor storage aliases another."""
    values = tuple(evidence)
    if len(values) != len(plan.layers):
        raise ValueError(f"expected {len(plan.layers)} layer evidence rows, got {len(values)}")
    for expected, actual in zip(plan.layers, values, strict=True):
        if (actual.layer_index, actual.layer_name, actual.callable_group) != (
            expected.layer_index,
            expected.layer_name,
            expected.callable_group,
        ):
            raise ValueError(f"layer evidence does not match plan at index {expected.layer_index}")
        if not actual.weight_addresses or not actual.metadata_addresses or not actual.cache_addresses:
            raise ValueError(f"layer {actual.layer_name} has empty weight/metadata/cache address evidence")
        if actual.eager_input_address == actual.capture_input_address:
            raise ValueError(f"layer {actual.layer_name} eager/capture inputs alias")
        if actual.eager_output_address == actual.capture_output_address:
            raise ValueError(f"layer {actual.layer_name} eager/capture outputs alias")

    for field in (
        "fixture_owner_id",
        "weight_owner_id",
        "cache_owner_id",
        "layer_owner_id",
        "eager_input_address",
        "eager_output_address",
        "capture_input_address",
        "capture_output_address",
    ):
        identities = tuple(getattr(item, field) for item in values)
        if len(set(identities)) != len(identities):
            raise ValueError(f"Capsule layer-local {field} values must be independent")

    for left in range(len(values)):
        for right in range(len(values)):
            same_callable = values[left].callable_group == values[right].callable_group
            same_device_owner = values[left].device_owner_id == values[right].device_owner_id
            if same_device_owner is not same_callable:
                raise ValueError(
                    "Capsule device-owner identity must match callable-group identity: "
                    f"layers={left}/{right}, same_callable={same_callable}"
                )

    for field in ("backend_owner_id", "context_owner_id"):
        identities = {getattr(item, field) for item in values}
        if len(identities) != 1:
            raise ValueError(f"Capsule layers must share one {field}")

    # A shared callable may patch every tensor address, but no live storage
    # owned by one logical layer may alias storage owned by another layer.
    # Check the union as well as each category so, for example, a metadata
    # table cannot silently alias another layer's weight or cache allocation.
    address_sets = tuple(
        set(item.weight_addresses)
        | set(item.metadata_addresses)
        | set(item.cache_addresses)
        | {
            item.eager_input_address,
            item.eager_output_address,
            item.capture_input_address,
            item.capture_output_address,
        }
        for item in values
    )
    for left in range(len(address_sets)):
        for right in range(left + 1, len(address_sets)):
            overlap = address_sets[left] & address_sets[right]
            if overlap:
                raise ValueError(f"Capsule layers {left}/{right} alias live tensor storage: {tuple(sorted(overlap))}")


@dataclass(frozen=True, slots=True)
class A3CapsuleReplayObservation:
    input_value: float
    eager_graph_max_abs_error: float
    dsa_output_max_abs: float

    def to_dict(self) -> dict[str, object]:
        return {
            "input_value": self.input_value,
            "eager_graph_max_abs_error": self.eager_graph_max_abs_error,
            "dsa_output_max_abs": self.dsa_output_max_abs,
        }


@dataclass(frozen=True, slots=True)
class A3CapsuleSmokeResult:
    runtime: str
    device: int
    batch: int
    num_layers: int
    topology: str
    graph_capture_count: int
    compiled_callable_count: int
    retained_capture_bindings: int
    replay_observations: tuple[A3CapsuleReplayObservation, ...]
    layer_evidence: tuple[A3CapsuleLayerAddressEvidence, ...]

    @property
    def max_replay_error(self) -> float:
        return max((item.eager_graph_max_abs_error for item in self.replay_observations), default=0.0)

    def to_dict(self) -> dict[str, object]:
        return {
            "runtime": self.runtime,
            "device": self.device,
            "batch": self.batch,
            "num_layers": self.num_layers,
            "topology": self.topology,
            "graph_capture_count": self.graph_capture_count,
            "compiled_callable_count": self.compiled_callable_count,
            "retained_capture_bindings": self.retained_capture_bindings,
            "max_replay_error": self.max_replay_error,
            "replay_observations": tuple(item.to_dict() for item in self.replay_observations),
            "layer_evidence": tuple(item.to_dict() for item in self.layer_evidence),
        }


@dataclass(slots=True)
class _CapsuleBuffers:
    initial: torch.Tensor
    dsa_outputs: tuple[torch.Tensor, ...]
    residuals: tuple[torch.Tensor, ...]
    bridges: tuple[torch.Tensor, ...]

    def layer_inputs(self) -> tuple[torch.Tensor, ...]:
        return (self.initial, *self.bridges)


@dataclass(slots=True)
class _RuntimeLayer:
    plan: A3CapsuleLayerPlan
    fixture: object
    metadata: tuple[object, ...]
    prepared_weights: object
    layer: object
    owner: object


def _make_buffers(
    *,
    num_layers: int,
    shape: torch.Size,
    dtype: torch.dtype,
    device: torch.device,
) -> _CapsuleBuffers:
    return _CapsuleBuffers(
        initial=torch.empty(shape, dtype=dtype, device=device),
        dsa_outputs=tuple(torch.empty(shape, dtype=dtype, device=device) for _ in range(num_layers)),
        residuals=tuple(torch.empty(shape, dtype=dtype, device=device) for _ in range(num_layers)),
        bridges=tuple(torch.empty(shape, dtype=dtype, device=device) for _ in range(num_layers - 1)),
    )


def _execute_capsule_chain(
    runtime_layers: Sequence[_RuntimeLayer],
    buffers: _CapsuleBuffers,
) -> torch.Tensor:
    value = buffers.initial
    for index, runtime_layer in enumerate(runtime_layers):
        dsa_output = buffers.dsa_outputs[index]
        dsa_output.fill_(31.0)
        torch.ops.vllm.dsa_forward(value, False, dsa_output, runtime_layer.plan.layer_name)
        residual = buffers.residuals[index]
        torch.add(value, dsa_output, out=residual)
        value = residual
        if index < len(buffers.bridges):
            scale = runtime_layer.plan.bridge_scale
            bias = runtime_layer.plan.bridge_bias
            assert scale is not None and bias is not None
            bridge = buffers.bridges[index]
            torch.mul(value, scale, out=bridge)
            torch.add(bridge, bias, out=bridge)
            value = bridge
    return value


def _tensor_addresses(values: Mapping[str, object]) -> tuple[int, ...]:
    return tuple(sorted(int(value.data_ptr()) for value in values.values() if isinstance(value, torch.Tensor)))


def _validate_runtime_layer_sharing(
    plan: A3CapsuleExecutionPlan,
    runtime_layers: Sequence[_RuntimeLayer],
    backend: object,
) -> None:
    """Prove that layer-local facades share exactly one prepared context.

    Each callable group owns one :class:`DecodeCSADeviceOwner` warmup-state
    facade.  Repeated layers in ``PP_same`` therefore warm one prepared
    operator exactly once, while every facade still points at the same backend
    and PyPTO L1 context.  Callable identity is checked on the prepared
    operator object, rather than inferred only from Host labels.
    """
    values = tuple(runtime_layers)
    if len(values) != len(plan.layers):
        raise AssertionError("runtime layer count does not match the Capsule plan")
    if len({id(item.owner) for item in values}) != len(values):
        raise AssertionError("Capsule layers unexpectedly share one layer owner")
    if any(item.owner.device_owner.backend is not backend for item in values):
        raise AssertionError("Capsule layers do not share the requested PyPTO backend")

    context = getattr(backend, "_context", None)
    if context is None:
        raise AssertionError("Capsule backend has no prepared PyPTO L1 context")
    if any(getattr(item.owner.device_owner.backend, "_context", None) is not context for item in values):
        raise AssertionError("Capsule layers do not share one PyPTO L1 context")

    records = getattr(backend, "_records", None)
    if not isinstance(records, dict):
        raise AssertionError("Capsule backend does not expose its prepared callable registry")
    operators: list[object] = []
    for layer in plan.layers:
        record = records.get(layer.spec.key)
        operator = getattr(record, "operator", None)
        if operator is None:
            raise AssertionError(f"Capsule callable group {layer.callable_group} has no prepared operator")
        operators.append(operator)
    for left in range(len(plan.layers)):
        for right in range(len(plan.layers)):
            same_group = plan.layers[left].callable_group == plan.layers[right].callable_group
            same_device_owner = values[left].owner.device_owner is values[right].owner.device_owner
            if same_device_owner is not same_group:
                raise AssertionError(
                    "warmup-state facade identity does not match the Capsule callable-group plan: "
                    f"layers={left}/{right}, same_group={same_group}"
                )
            if (operators[left] is operators[right]) is not same_group:
                raise AssertionError(
                    "prepared operator identity does not match the Capsule callable-group plan: "
                    f"layers={left}/{right}, same_group={same_group}"
                )


def _collect_layer_evidence(
    runtime_layers: Sequence[_RuntimeLayer],
    eager: _CapsuleBuffers,
    captured: _CapsuleBuffers,
) -> tuple[A3CapsuleLayerAddressEvidence, ...]:
    eager_inputs = eager.layer_inputs()
    capture_inputs = captured.layer_inputs()
    result: list[A3CapsuleLayerAddressEvidence] = []
    for index, runtime_layer in enumerate(runtime_layers):
        fixture = runtime_layer.fixture
        raw_caches = fixture.raw_cache_tuple
        backend = runtime_layer.owner.device_owner.backend
        context = getattr(backend, "_context", None)
        if context is None:
            raise AssertionError("cannot collect Capsule evidence without a prepared PyPTO context")
        metadata_arguments = {name: fixture.arguments[name] for name in _METADATA_ARGUMENT_NAMES}
        result.append(
            A3CapsuleLayerAddressEvidence(
                layer_index=index,
                layer_name=runtime_layer.plan.layer_name,
                callable_group=runtime_layer.plan.callable_group,
                fixture_owner_id=id(fixture),
                weight_owner_id=id(runtime_layer.prepared_weights),
                cache_owner_id=id(fixture.prepared_caches),
                layer_owner_id=id(runtime_layer.owner),
                device_owner_id=id(runtime_layer.owner.device_owner),
                backend_owner_id=id(backend),
                context_owner_id=id(context),
                weight_addresses=_tensor_addresses(runtime_layer.prepared_weights.for_launch()),
                metadata_addresses=_tensor_addresses(metadata_arguments),
                cache_addresses=tuple(sorted(int(tensor.data_ptr()) for tensor in raw_caches)),
                eager_input_address=int(eager_inputs[index].data_ptr()),
                eager_output_address=int(eager.dsa_outputs[index].data_ptr()),
                capture_input_address=int(capture_inputs[index].data_ptr()),
                capture_output_address=int(captured.dsa_outputs[index].data_ptr()),
            )
        )
    return tuple(result)


def run_a3_capsule_smoke(config: A3CapsuleSmokeConfig) -> A3CapsuleSmokeResult:
    """Run eager and one ACLGraph for a production custom-op Capsule.

    Call this in a fresh process for each runtime/topology case because the L1
    runtime owner and captured callable handles are deliberately process-pinned.
    """
    if not isinstance(config, A3CapsuleSmokeConfig):
        raise TypeError("config must be A3CapsuleSmokeConfig")
    plan = build_a3_capsule_execution_plan(config)

    import torch_npu
    from vllm.forward_context import ForwardContext, override_forward_context

    import vllm_ascend.ops._pypto_dsv4_csa.dispatch as dispatch_module
    import vllm_ascend.ops.dsa  # noqa: F401 -- registers torch.ops.vllm.dsa_forward
    from tests.pypto_dsv4_decode_csa.a3_dispatch_smoke import (
        _custom_op_layer,
        _metadata_dict,
        _native_metadata,
        _prepared_weights,
    )
    from tests.pypto_dsv4_decode_csa.fixtures import build_zero_fixture
    from vllm_ascend.ops._pypto_dsv4_csa.backend import PyPTODSABackend
    from vllm_ascend.ops._pypto_dsv4_csa.dispatch import (
        DecodeCSADeviceOwner,
        DecodeCSALayerOwner,
    )
    from vllm_ascend.ops._pypto_dsv4_csa.native_metadata import (
        derive_decode_csa_program_spec,
        validate_decode_csa_uniform_query_rows,
    )

    torch_npu.npu.set_device(config.device)
    target = torch.device(f"npu:{config.device}")
    backend = PyPTODSABackend(device=config.device, runtime=config.runtime)
    runtime_layers: list[_RuntimeLayer] = []
    graph = None
    try:
        fixtures = tuple(build_zero_fixture(layer.spec, device=target, runtime=config.runtime) for layer in plan.layers)
        metadata = tuple(_native_metadata(fixture) for fixture in fixtures)
        for layer_plan, fixture, layer_metadata in zip(plan.layers, fixtures, metadata, strict=True):
            derived = derive_decode_csa_program_spec(fixture.raw_cache_tuple, layer_metadata)
            if derived.key != layer_plan.spec.key:
                raise AssertionError(f"fixture/spec mismatch for {layer_plan.layer_name}")
            validate_decode_csa_uniform_query_rows(layer_metadata, layer_plan.spec)
        for layer_plan in plan.layers:
            backend.compile(layer_plan.spec)
        compiled_callable_count = len(backend.registered_specs)
        if compiled_callable_count != plan.compiled_callable_count:
            raise AssertionError(
                "compiled callable registry does not match the Host Capsule plan: "
                f"expected={plan.compiled_callable_count}, got={compiled_callable_count}"
            )
        backend.prepare()

        no_compile_layers: dict[str, object] = {}
        metadata_by_name: dict[str, object] = {}
        device_owners_by_group: dict[int, DecodeCSADeviceOwner] = {}
        for layer_plan, fixture, layer_metadata in zip(plan.layers, fixtures, metadata, strict=True):
            layer = _custom_op_layer(fixture, prefix=layer_plan.layer_name)
            weights = _prepared_weights(fixture, layer_metadata)
            device_owner = device_owners_by_group.get(layer_plan.callable_group)
            if device_owner is None:
                device_owner = DecodeCSADeviceOwner(
                    device=config.device,
                    runtime=config.runtime,
                    backend=backend,
                    specs={config.batch: layer_plan.spec},
                    family_key=layer_plan.spec.key[1:],
                )
                device_owners_by_group[layer_plan.callable_group] = device_owner
            owner = DecodeCSALayerOwner(
                layer=layer,
                device_owner=device_owner,
                weights=weights,
                prepared_caches={config.batch: fixture.prepared_caches},
                cache_key=dispatch_module._cache_storage_key(fixture.raw_cache_tuple),
            )
            layer._pypto_dsv4_csa_dispatch = owner
            no_compile_layers[layer_plan.layer_name] = layer
            metadata_by_name.update(_metadata_dict(layer_plan.layer_name, layer_metadata))
            runtime_layers.append(
                _RuntimeLayer(
                    plan=layer_plan,
                    fixture=fixture,
                    metadata=layer_metadata,
                    prepared_weights=weights,
                    layer=layer,
                    owner=owner,
                )
            )
        _validate_runtime_layer_sharing(plan, runtime_layers, backend)

        context = ForwardContext(
            no_compile_layers=no_compile_layers,
            attn_metadata=metadata_by_name,
            slot_mapping={},
        )
        context.capturing = False
        tensor_shape = torch.Size((plan.config.batch * 8, 4096))
        eager_buffers = _make_buffers(
            num_layers=config.num_layers,
            shape=tensor_shape,
            dtype=torch.bfloat16,
            device=target,
        )
        capture_buffers = _make_buffers(
            num_layers=config.num_layers,
            shape=tensor_shape,
            dtype=torch.bfloat16,
            device=target,
        )

        eager_buffers.initial.fill_(0.125)
        with override_forward_context(context):
            _execute_capsule_chain(runtime_layers, eager_buffers)
        torch_npu.npu.synchronize(config.device)
        for runtime_layer in runtime_layers:
            runtime_layer.owner.device_owner.mark_warmups_quiesced()

        capture_stream = torch_npu.npu.Stream(device=config.device)
        # Seed the capture input on the same stream that will consume it.  This
        # is deliberately outside capture but ordered before its first node;
        # no default-stream timing assumption is part of the smoke contract.
        with torch_npu.npu.stream(capture_stream):
            capture_buffers.initial.fill_(0.125)
        graph = torch_npu.npu.NPUGraph()
        context.capturing = True
        with override_forward_context(context), torch_npu.npu.graph(graph, stream=capture_stream):
            _execute_capsule_chain(runtime_layers, capture_buffers)
        # ACLGraph capture may execute its nodes once.  The default-stream
        # eager oracle below mutates the same layer caches, so establish an
        # explicit caller-owned cross-stream quiescence boundary first.
        capture_stream.synchronize()

        layer_evidence = _collect_layer_evidence(runtime_layers, eager_buffers, capture_buffers)
        validate_a3_capsule_resource_independence(plan, layer_evidence)
        retained = len(backend._bindings)
        if retained != config.num_layers:
            raise AssertionError(f"expected one retained graph binding per layer, got {retained}")

        observations: list[A3CapsuleReplayObservation] = []
        for value in config.replay_values:
            context.capturing = False
            eager_buffers.initial.fill_(float(value))
            with override_forward_context(context):
                eager_final = _execute_capsule_chain(runtime_layers, eager_buffers)
            torch_npu.npu.synchronize(config.device)

            with torch_npu.npu.stream(capture_stream):
                capture_buffers.initial.fill_(float(value))
                graph.replay()
            capture_stream.synchronize()
            eager_host = eager_final.detach().float().to(device="cpu")
            graph_host = capture_buffers.residuals[-1].detach().float().to(device="cpu")
            error = float((eager_host - graph_host).abs().max())
            # Check every layer in both executions.  Comparing only the final
            # residual could hide two compensating custom-op failures, and
            # checking only graph buffers would leave the eager oracle weak.
            dsa_output_max = max(
                float(output.detach().float().to(device="cpu").abs().max())
                for output in (*eager_buffers.dsa_outputs, *capture_buffers.dsa_outputs)
            )
            if error != 0.0:
                raise AssertionError(f"Capsule eager/graph mismatch for input {value}: {error}")
            if dsa_output_max != 0.0:
                raise AssertionError(f"zero-weight Capsule DSA output is non-zero: {dsa_output_max}")
            observations.append(
                A3CapsuleReplayObservation(
                    input_value=float(value),
                    eager_graph_max_abs_error=error,
                    dsa_output_max_abs=dsa_output_max,
                )
            )
    finally:
        torch_npu.npu.synchronize(config.device)
        if graph is not None:
            graph.reset()
        backend.close()

    return A3CapsuleSmokeResult(
        runtime=config.runtime,
        device=config.device,
        batch=config.batch,
        num_layers=config.num_layers,
        topology=config.topology.value,
        graph_capture_count=1,
        compiled_callable_count=compiled_callable_count,
        retained_capture_bindings=retained,
        replay_observations=tuple(observations),
        layer_evidence=layer_evidence,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", choices=tuple(sorted(_SUPPORTED_RUNTIMES)), required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batch", type=int, choices=(4, 8, 12, 16), default=4)
    parser.add_argument("--num-layers", type=int, choices=(1, 2, 4), default=4)
    parser.add_argument(
        "--topology",
        choices=tuple(topology.value for topology in sorted(_SUPPORTED_TOPOLOGIES, key=lambda item: item.value)),
        default=CapsuleTopology.PPPP.value,
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--layer-name-prefix", default="pypto.capsule.layers")
    parser.add_argument("--replay-values", type=float, nargs="+", default=(0.25, -0.5, 1.0))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_a3_capsule_smoke(
        A3CapsuleSmokeConfig(
            runtime=args.runtime,
            device=args.device,
            batch=args.batch,
            num_layers=args.num_layers,
            topology=CapsuleTopology(args.topology),
            seed=args.seed,
            layer_name_prefix=args.layer_name_prefix,
            replay_values=tuple(args.replay_values),
        )
    )
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "A3CapsuleExecutionPlan",
    "A3CapsuleLayerAddressEvidence",
    "A3CapsuleLayerPlan",
    "A3CapsuleReplayObservation",
    "A3CapsuleSmokeConfig",
    "A3CapsuleSmokeResult",
    "build_parser",
    "build_a3_capsule_execution_plan",
    "main",
    "run_a3_capsule_smoke",
    "validate_a3_capsule_resource_independence",
]
