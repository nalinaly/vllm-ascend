# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Nonzero NNNN-vs-candidate decode-CSA Capsule comparison.

The default fidelity shell preserves the production attention-half boundary:
``[T, 4, 4096] -> HC-pre-v2 -> RMSNorm -> DSA -> HC-post``.  The local native
oracle and candidate receive the exact same normalized DSA input and HC
``residual/post/comb`` tensors, so only the DSA backend differs.  The older
``[T, 4096]`` residual/affine shell remains available as an explicit
diagnostic mode; it is not DeepSeek V4 HC or MoE.

The reference NNNN chain and candidate PPPP/NPNP/PNPN chain share each logical
layer's real synthetic TP1 ratio-4 weights.  They do not share mutable state:
every layer owns three disjoint six-cache worlds.  NNNN is the end-to-end chain
reference.  A pristine native local-oracle world evaluates each candidate
layer on that layer's *actual candidate input*, so upstream numerical drift
cannot contaminate the strict output/state contract for a later layer.  The
third world belongs to the candidate backend and continues downstream.

Local correctness and accumulated chain drift are reported and accepted with
independent policies.  It is safe to import on a Host without an NPU;
device/bootstrap imports are confined to ``run_a3_native_capsule_compare``.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

import torch

from tests.pypto_dsv4_decode_csa.a3_native_compare import (
    SUPPORTED_RUNTIMES,
    TensorComparison,
    _custom_op_call,
    _destroy_tp1,
    _forward_context,
    _initialize_tp1,
    compare_state_snapshots,
    compare_tensors,
)
from tests.pypto_dsv4_decode_csa.capsule import (
    BackendKind,
    CapsuleTopology,
    topology_backends,
)
from tests.pypto_dsv4_decode_csa.indexer_quantized_comparison import (
    IndexerQuantizedComparison,
    compare_indexer_quantized,
)
from tests.pypto_dsv4_decode_csa.native_fixture import (
    STATE_NAMES,
    bind_state_world,
    build_decode_csa_state_world,
    build_real_native_metadata,
    build_real_ratio4_attention,
    build_synthetic_vllm_config,
    build_twin_decode_csa_state_worlds,
)
from vllm_ascend.ops._pypto_dsv4_csa import DecodeCSAProgramSpec
from vllm_ascend.ops._pypto_dsv4_csa.config import FLASH


class CapsuleShellKind(str, Enum):
    """Execution shell surrounding each DSA call."""

    FIDELITY = "fidelity"
    DETERMINISTIC_DIAGNOSTIC = "deterministic_diagnostic"


HC_PRE_OPERATOR = "torch.ops._C_ascend.npu_hc_pre_v2"
HC_POST_OPERATOR = "torch.ops._C_ascend.npu_hc_post"
ASCEND_RMS_NORM_OPERATOR = "AscendRMSNorm"
SHARED_STATIC_ARGUMENT_NAMES = frozenset({"freqs_cos", "freqs_sin", "hadamard_idx"})


@dataclass(frozen=True, slots=True)
class A3NativeCapsuleCompareConfig:
    """Static configuration for exactly one fresh-process comparison."""

    runtime: str
    device: int = 0
    batch: int = 4
    num_layers: int = 4
    start_position: int = 0
    seed: int = 20260902
    atol: float = 0.1
    rtol: float = 0.1
    chain_atol: float | None = None
    chain_rtol: float = 0.1
    layer_name_prefix: str = "model.layers"
    candidate_topology: CapsuleTopology = CapsuleTopology.PPPP
    shell_kind: CapsuleShellKind = CapsuleShellKind.FIDELITY

    def __post_init__(self) -> None:
        if self.runtime not in SUPPORTED_RUNTIMES:
            raise ValueError(f"runtime must be one of {SUPPORTED_RUNTIMES}, got {self.runtime!r}")
        if isinstance(self.device, bool) or not isinstance(self.device, int) or self.device < 0:
            raise ValueError("device must be a non-negative integer")
        if self.batch != 4:
            raise ValueError("the nonzero Capsule is intentionally fixed to TP1 B4")
        if (
            isinstance(self.num_layers, bool)
            or not isinstance(self.num_layers, int)
            or self.num_layers not in {1, 2, 4}
        ):
            raise ValueError("num_layers must be one of 1, 2, or 4")
        try:
            topology = CapsuleTopology(self.candidate_topology)
        except ValueError as error:
            raise ValueError(f"unsupported candidate topology {self.candidate_topology!r}") from error
        if topology not in {CapsuleTopology.PPPP, CapsuleTopology.NPNP, CapsuleTopology.PNPN}:
            raise ValueError("candidate_topology must be PPPP, NPNP, or PNPN")
        object.__setattr__(self, "candidate_topology", topology)
        try:
            shell_kind = CapsuleShellKind(self.shell_kind)
        except ValueError as error:
            raise ValueError(f"unsupported Capsule shell_kind {self.shell_kind!r}") from error
        object.__setattr__(self, "shell_kind", shell_kind)
        if isinstance(self.start_position, bool) or not isinstance(self.start_position, int):
            raise ValueError("start_position must be an integer")
        spec = DecodeCSAProgramSpec(batch=self.batch)
        if self.start_position < 0 or self.start_position + spec.seq > FLASH.max_position_embeddings:
            raise ValueError("start_position is outside the production RoPE table")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")
        if not self.layer_name_prefix:
            raise ValueError("layer_name_prefix must not be empty")
        if not math.isfinite(self.atol) or self.atol < 0:
            raise ValueError("atol must be finite and non-negative")
        if not math.isfinite(self.rtol) or self.rtol < 0:
            raise ValueError("rtol must be finite and non-negative")
        # End-to-end drift is deliberately separate from same-input local
        # correctness.  By default use a root-sum-square budget over all
        # layers; this does not affect cache, raw INT8, or local-output
        # acceptance.
        chain_atol = self.atol * math.sqrt(self.num_layers) if self.chain_atol is None else self.chain_atol
        if not math.isfinite(chain_atol) or chain_atol < 0:
            raise ValueError("chain_atol must be finite and non-negative")
        object.__setattr__(self, "chain_atol", chain_atol)
        if not math.isfinite(self.chain_rtol) or self.chain_rtol < 0:
            raise ValueError("chain_rtol must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class A3NativeCapsuleLayerPlan:
    layer_index: int
    model_layer_index: int
    layer_name: str
    weight_seed: int
    native_cache_owner_token: str
    local_oracle_cache_owner_token: str
    candidate_cache_owner_token: str
    candidate_backend: BackendKind
    bridge_scale: float | None
    bridge_bias: float | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class A3NativeCapsuleComparePlan:
    config: A3NativeCapsuleCompareConfig
    native_route: tuple[BackendKind, ...]
    candidate_route: tuple[BackendKind, ...]
    layers: tuple[A3NativeCapsuleLayerPlan, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "runtime": self.config.runtime,
            "shell_kind": self.config.shell_kind.value,
            "native_topology": CapsuleTopology.NNNN.value,
            "candidate_topology": self.config.candidate_topology.value,
            "local_atol": self.config.atol,
            "local_rtol": self.config.rtol,
            "chain_atol": self.config.chain_atol,
            "chain_rtol": self.config.chain_rtol,
            "native_route": tuple(item.value for item in self.native_route),
            "candidate_route": tuple(item.value for item in self.candidate_route),
            "layers": tuple(layer.to_dict() for layer in self.layers),
        }


def build_a3_native_capsule_compare_plan(
    config: A3NativeCapsuleCompareConfig,
) -> A3NativeCapsuleComparePlan:
    """Build the immutable 1/2/4-layer route and ownership plan."""
    if not isinstance(config, A3NativeCapsuleCompareConfig):
        raise TypeError("config must be A3NativeCapsuleCompareConfig")
    candidate_route = topology_backends(config.candidate_topology, config.num_layers)
    ratio4_layer_indices = tuple(index for index, ratio in enumerate(FLASH.compress_ratios) if ratio == 4)
    if len(ratio4_layer_indices) < config.num_layers:
        raise ValueError(f"synthetic DSV4 config does not contain {config.num_layers} independent ratio-4 layers")
    layers = []
    for index, model_layer_index in enumerate(ratio4_layer_indices[: config.num_layers]):
        has_bridge = index + 1 < config.num_layers
        layers.append(
            A3NativeCapsuleLayerPlan(
                layer_index=index,
                model_layer_index=model_layer_index,
                layer_name=f"{config.layer_name_prefix}.{model_layer_index}.self_attn",
                weight_seed=config.seed + index * 10_009,
                native_cache_owner_token=f"native-cache://layer/{index}",
                local_oracle_cache_owner_token=f"local-oracle-cache://layer/{index}",
                candidate_cache_owner_token=f"candidate-cache://{candidate_route[index].value}/layer/{index}",
                candidate_backend=candidate_route[index],
                # Exactly representable binary fractions keep bridge drift out
                # of the native/PyPTO CSA comparison.
                bridge_scale=(1.0 + (index + 1) / 16.0) if has_bridge else None,
                bridge_bias=(((config.seed + index * 13) % 9 - 4) / 32.0) if has_bridge else None,
            )
        )
    plan = A3NativeCapsuleComparePlan(
        config=config,
        native_route=topology_backends(CapsuleTopology.NNNN, config.num_layers),
        candidate_route=candidate_route,
        layers=tuple(layers),
    )
    _validate_compare_plan(plan)
    return plan


def _validate_compare_plan(plan: A3NativeCapsuleComparePlan) -> None:
    expected_indices = tuple(range(plan.config.num_layers))
    if tuple(layer.layer_index for layer in plan.layers) != expected_indices:
        raise ValueError("Capsule layer indices must be contiguous")
    if any(FLASH.compress_ratios[layer.model_layer_index] != 4 for layer in plan.layers):
        raise ValueError("every real Capsule layer must select a ratio-4 DSV4 model index")
    if plan.native_route != (BackendKind.NATIVE,) * plan.config.num_layers:
        raise ValueError("native comparison route must be exactly NNNN")
    if plan.candidate_route != topology_backends(plan.config.candidate_topology, plan.config.num_layers):
        raise ValueError("candidate route does not match candidate_topology")
    if tuple(layer.candidate_backend for layer in plan.layers) != plan.candidate_route:
        raise ValueError("layer candidate backends do not match the candidate route")
    for field in (
        "model_layer_index",
        "layer_name",
        "weight_seed",
        "native_cache_owner_token",
        "local_oracle_cache_owner_token",
        "candidate_cache_owner_token",
    ):
        values = tuple(getattr(layer, field) for layer in plan.layers)
        if len(set(values)) != len(values):
            raise ValueError(f"per-layer {field} values must be unique")
    if any(layer.bridge_scale is None or layer.bridge_bias is None for layer in plan.layers[:-1]):
        raise ValueError("every non-final layer requires a deterministic bridge")
    if plan.layers[-1].bridge_scale is not None or plan.layers[-1].bridge_bias is not None:
        raise ValueError("the final layer must not have a bridge")


@dataclass(frozen=True, slots=True)
class CapsuleChainTrace:
    """Device-agnostic trace of one Capsule chain."""

    dsa_outputs: tuple[torch.Tensor, ...]
    layer_outputs: tuple[torch.Tensor, ...]
    bridge_outputs: tuple[torch.Tensor, ...]
    final_output: torch.Tensor
    hc_layers: tuple[CapsuleHCLayerTrace, ...] = ()


@dataclass(frozen=True, slots=True)
class CapsuleHCLayerTrace:
    """Graph-visible tensors surrounding one fidelity-mode DSA call."""

    input_value: torch.Tensor
    residual: torch.Tensor
    hidden: torch.Tensor
    normalized: torch.Tensor
    post: torch.Tensor
    comb: torch.Tensor


@dataclass(frozen=True, slots=True)
class CapsuleLocalOracleTrace:
    """Per-layer native outputs evaluated on the actual candidate inputs.

    These values deliberately do not form a chain: layer ``i+1`` receives the
    actual candidate output, not the previous shadow output.
    """

    dsa_outputs: tuple[torch.Tensor, ...]
    layer_outputs: tuple[torch.Tensor, ...]
    hc_layers: tuple[CapsuleHCLayerTrace, ...] = ()


LayerCall = Callable[[torch.Tensor], torch.Tensor]
HCPreCall = Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
HCPostCall = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass(frozen=True, slots=True)
class HCFidelityLayerOps:
    """Injectable real-or-fake HC boundary used by device and Host runners."""

    hc_pre: HCPreCall
    rms_norm: LayerCall
    hc_post: HCPostCall


def _require_tensor_like(
    value: object,
    *,
    description: str,
    shape: torch.Size,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{description} must be torch.Tensor")
    if value.shape != shape or value.dtype != dtype or value.device != device:
        raise ValueError(
            f"{description} must have shape={tuple(shape)}, dtype={dtype}, device={device}; "
            f"got shape={tuple(value.shape)}, dtype={value.dtype}, device={value.device}"
        )
    return value


def _run_hc_pre_and_norm(
    *,
    layer_index: int,
    value: torch.Tensor,
    ops: HCFidelityLayerOps,
) -> CapsuleHCLayerTrace:
    if value.ndim != 3 or tuple(value.shape[1:]) != (FLASH.hc_mult, FLASH.hidden_size):
        raise ValueError(
            f"layer {layer_index} fidelity input must have shape [T, {FLASH.hc_mult}, {FLASH.hidden_size}]"
        )
    if value.dtype is not torch.bfloat16:
        raise ValueError(f"layer {layer_index} fidelity input must use torch.bfloat16")
    residual = value.clone()
    outputs = ops.hc_pre(value)
    if not isinstance(outputs, tuple) or len(outputs) != 3:
        raise TypeError(f"layer {layer_index} HC-pre must return exactly (hidden, post, comb)")
    hidden, post, comb = outputs
    tokens = value.shape[0]
    hidden = _require_tensor_like(
        hidden,
        description=f"layer {layer_index} HC-pre hidden",
        shape=torch.Size((tokens, FLASH.hidden_size)),
        dtype=torch.bfloat16,
        device=value.device,
    )
    post = _require_tensor_like(
        post,
        description=f"layer {layer_index} HC-pre post",
        shape=torch.Size((tokens, FLASH.hc_mult)),
        dtype=torch.float32,
        device=value.device,
    )
    comb = _require_tensor_like(
        comb,
        description=f"layer {layer_index} HC-pre comb",
        shape=torch.Size((tokens, FLASH.hc_mult, FLASH.hc_mult)),
        dtype=torch.float32,
        device=value.device,
    )
    normalized = _require_tensor_like(
        ops.rms_norm(hidden),
        description=f"layer {layer_index} RMSNorm output",
        shape=hidden.shape,
        dtype=hidden.dtype,
        device=hidden.device,
    )
    return CapsuleHCLayerTrace(
        input_value=value,
        residual=residual,
        hidden=hidden,
        normalized=normalized,
        post=post,
        comb=comb,
    )


def _run_hc_post(
    *,
    layer_index: int,
    ops: HCFidelityLayerOps,
    dsa_output: torch.Tensor,
    hc_trace: CapsuleHCLayerTrace,
    route: str,
) -> torch.Tensor:
    return _require_tensor_like(
        ops.hc_post(
            dsa_output,
            hc_trace.residual,
            hc_trace.post,
            hc_trace.comb,
        ),
        description=f"layer {layer_index} {route} HC-post output",
        shape=hc_trace.residual.shape,
        dtype=hc_trace.residual.dtype,
        device=hc_trace.residual.device,
    )


def execute_hc_fidelity_capsule_chain(
    plan: A3NativeCapsuleComparePlan,
    initial: torch.Tensor,
    layer_calls: Sequence[LayerCall],
    hc_layers: Sequence[HCFidelityLayerOps],
) -> CapsuleChainTrace:
    """Execute the real HC-pre/RMSNorm/DSA/HC-post attention-half shell."""
    calls = tuple(layer_calls)
    shells = tuple(hc_layers)
    if len(calls) != len(plan.layers) or len(shells) != len(plan.layers):
        raise ValueError(f"expected {len(plan.layers)} DSA calls and HC layer shells")
    if not isinstance(initial, torch.Tensor):
        raise TypeError("initial must be torch.Tensor")

    value = initial
    dsa_outputs: list[torch.Tensor] = []
    layer_outputs: list[torch.Tensor] = []
    bridge_outputs: list[torch.Tensor] = []
    hc_traces: list[CapsuleHCLayerTrace] = []
    for layer, call, shell in zip(plan.layers, calls, shells, strict=True):
        hc_trace = _run_hc_pre_and_norm(layer_index=layer.layer_index, value=value, ops=shell)
        dsa_output = _require_tensor_like(
            call(hc_trace.normalized),
            description=f"layer {layer.layer_index} DSA output",
            shape=hc_trace.normalized.shape,
            dtype=hc_trace.normalized.dtype,
            device=hc_trace.normalized.device,
        )
        layer_output = _run_hc_post(
            layer_index=layer.layer_index,
            ops=shell,
            dsa_output=dsa_output,
            hc_trace=hc_trace,
            route="chain",
        )
        hc_traces.append(hc_trace)
        dsa_outputs.append(dsa_output)
        layer_outputs.append(layer_output)
        if layer.bridge_scale is not None:
            assert layer.bridge_bias is not None
            value = torch.add(torch.mul(layer_output, layer.bridge_scale), layer.bridge_bias)
            bridge_outputs.append(value)
        else:
            value = layer_output
    return CapsuleChainTrace(
        dsa_outputs=tuple(dsa_outputs),
        layer_outputs=tuple(layer_outputs),
        bridge_outputs=tuple(bridge_outputs),
        final_output=value,
        hc_layers=tuple(hc_traces),
    )


def execute_hc_fidelity_capsule_with_local_oracles(
    plan: A3NativeCapsuleComparePlan,
    initial: torch.Tensor,
    local_oracle_calls: Sequence[LayerCall],
    candidate_calls: Sequence[LayerCall],
    hc_layers: Sequence[HCFidelityLayerOps],
) -> tuple[CapsuleLocalOracleTrace, CapsuleChainTrace]:
    """Execute same-input local oracles with one shared HC envelope per layer.

    HC-pre and RMSNorm run exactly once.  The local oracle and candidate DSA
    calls receive the same normalized tensor object, and both HC-post calls
    receive the same residual/post/comb objects.  Only the candidate HC-post
    result is propagated into the following bridge and layer.
    """
    oracle_calls = tuple(local_oracle_calls)
    actual_calls = tuple(candidate_calls)
    shells = tuple(hc_layers)
    if any(len(values) != len(plan.layers) for values in (oracle_calls, actual_calls, shells)):
        raise ValueError(f"expected {len(plan.layers)} local-oracle calls, candidate calls, and HC layer shells")
    if not isinstance(initial, torch.Tensor):
        raise TypeError("initial must be torch.Tensor")

    value = initial
    oracle_dsa_outputs: list[torch.Tensor] = []
    oracle_layer_outputs: list[torch.Tensor] = []
    candidate_dsa_outputs: list[torch.Tensor] = []
    candidate_layer_outputs: list[torch.Tensor] = []
    candidate_bridge_outputs: list[torch.Tensor] = []
    hc_traces: list[CapsuleHCLayerTrace] = []
    for layer, oracle_call, candidate_call, shell in zip(
        plan.layers,
        oracle_calls,
        actual_calls,
        shells,
        strict=True,
    ):
        hc_trace = _run_hc_pre_and_norm(layer_index=layer.layer_index, value=value, ops=shell)
        oracle_dsa = _require_tensor_like(
            oracle_call(hc_trace.normalized),
            description=f"layer {layer.layer_index} local oracle DSA output",
            shape=hc_trace.normalized.shape,
            dtype=hc_trace.normalized.dtype,
            device=hc_trace.normalized.device,
        )
        candidate_dsa = _require_tensor_like(
            candidate_call(hc_trace.normalized),
            description=f"layer {layer.layer_index} candidate DSA output",
            shape=hc_trace.normalized.shape,
            dtype=hc_trace.normalized.dtype,
            device=hc_trace.normalized.device,
        )
        oracle_layer = _run_hc_post(
            layer_index=layer.layer_index,
            ops=shell,
            dsa_output=oracle_dsa,
            hc_trace=hc_trace,
            route="local oracle",
        )
        candidate_layer = _run_hc_post(
            layer_index=layer.layer_index,
            ops=shell,
            dsa_output=candidate_dsa,
            hc_trace=hc_trace,
            route="candidate",
        )
        hc_traces.append(hc_trace)
        oracle_dsa_outputs.append(oracle_dsa)
        oracle_layer_outputs.append(oracle_layer)
        candidate_dsa_outputs.append(candidate_dsa)
        candidate_layer_outputs.append(candidate_layer)
        if layer.bridge_scale is not None:
            assert layer.bridge_bias is not None
            value = torch.add(torch.mul(candidate_layer, layer.bridge_scale), layer.bridge_bias)
            candidate_bridge_outputs.append(value)
        else:
            value = candidate_layer

    shared_hc_traces = tuple(hc_traces)
    return (
        CapsuleLocalOracleTrace(
            dsa_outputs=tuple(oracle_dsa_outputs),
            layer_outputs=tuple(oracle_layer_outputs),
            hc_layers=shared_hc_traces,
        ),
        CapsuleChainTrace(
            dsa_outputs=tuple(candidate_dsa_outputs),
            layer_outputs=tuple(candidate_layer_outputs),
            bridge_outputs=tuple(candidate_bridge_outputs),
            final_output=value,
            hc_layers=shared_hc_traces,
        ),
    )


def execute_deterministic_capsule_chain(
    plan: A3NativeCapsuleComparePlan,
    initial: torch.Tensor,
    layer_calls: Sequence[LayerCall],
) -> CapsuleChainTrace:
    """Execute the deliberately small residual/affine shell.

    This helper has no NPU dependency and is used by Host tests to prove that
    both routes receive identical ordering and bridge parameters.
    """
    calls = tuple(layer_calls)
    if len(calls) != len(plan.layers):
        raise ValueError(f"expected {len(plan.layers)} layer calls, got {len(calls)}")
    if not isinstance(initial, torch.Tensor):
        raise TypeError("initial must be torch.Tensor")
    value = initial
    dsa_outputs: list[torch.Tensor] = []
    layer_outputs: list[torch.Tensor] = []
    bridge_outputs: list[torch.Tensor] = []
    for layer, call in zip(plan.layers, calls, strict=True):
        dsa_output = call(value)
        if not isinstance(dsa_output, torch.Tensor):
            raise TypeError(f"layer {layer.layer_index} call must return torch.Tensor")
        if dsa_output.shape != value.shape or dsa_output.dtype != value.dtype or dsa_output.device != value.device:
            raise ValueError(f"layer {layer.layer_index} CSA output must match its input shape/dtype/device")
        layer_output = torch.add(value, dsa_output)
        dsa_outputs.append(dsa_output)
        layer_outputs.append(layer_output)
        if layer.bridge_scale is not None:
            assert layer.bridge_bias is not None
            value = torch.add(torch.mul(layer_output, layer.bridge_scale), layer.bridge_bias)
            bridge_outputs.append(value)
        else:
            value = layer_output
    return CapsuleChainTrace(
        dsa_outputs=tuple(dsa_outputs),
        layer_outputs=tuple(layer_outputs),
        bridge_outputs=tuple(bridge_outputs),
        final_output=value,
    )


def execute_capsule_with_local_oracles(
    plan: A3NativeCapsuleComparePlan,
    initial: torch.Tensor,
    local_oracle_calls: Sequence[LayerCall],
    candidate_calls: Sequence[LayerCall],
) -> tuple[CapsuleLocalOracleTrace, CapsuleChainTrace]:
    """Run a propagated candidate chain plus same-input native shadows.

    Each native shadow consumes exactly the tensor presented to the candidate
    layer.  Only the candidate output is propagated through the residual and
    affine bridge.  Consequently an accepted upstream BF16 difference remains
    visible in the end-to-end NNNN comparison but cannot invalidate a later
    layer's cache comparison merely because that later input changed.
    """
    oracle_calls = tuple(local_oracle_calls)
    actual_calls = tuple(candidate_calls)
    if len(oracle_calls) != len(plan.layers) or len(actual_calls) != len(plan.layers):
        raise ValueError(f"expected {len(plan.layers)} local-oracle and candidate calls")
    if not isinstance(initial, torch.Tensor):
        raise TypeError("initial must be torch.Tensor")

    value = initial
    oracle_dsa_outputs: list[torch.Tensor] = []
    oracle_layer_outputs: list[torch.Tensor] = []
    candidate_dsa_outputs: list[torch.Tensor] = []
    candidate_layer_outputs: list[torch.Tensor] = []
    candidate_bridge_outputs: list[torch.Tensor] = []
    for layer, oracle_call, candidate_call in zip(
        plan.layers,
        oracle_calls,
        actual_calls,
        strict=True,
    ):
        oracle_dsa = oracle_call(value)
        candidate_dsa = candidate_call(value)
        for route, output in (("local oracle", oracle_dsa), ("candidate", candidate_dsa)):
            if not isinstance(output, torch.Tensor):
                raise TypeError(f"layer {layer.layer_index} {route} call must return torch.Tensor")
            if output.shape != value.shape or output.dtype != value.dtype or output.device != value.device:
                raise ValueError(
                    f"layer {layer.layer_index} {route} CSA output must match its input shape/dtype/device"
                )

        oracle_layer = torch.add(value, oracle_dsa)
        candidate_layer = torch.add(value, candidate_dsa)
        oracle_dsa_outputs.append(oracle_dsa)
        oracle_layer_outputs.append(oracle_layer)
        candidate_dsa_outputs.append(candidate_dsa)
        candidate_layer_outputs.append(candidate_layer)
        if layer.bridge_scale is not None:
            assert layer.bridge_bias is not None
            value = torch.add(torch.mul(candidate_layer, layer.bridge_scale), layer.bridge_bias)
            candidate_bridge_outputs.append(value)
        else:
            value = candidate_layer

    return (
        CapsuleLocalOracleTrace(
            dsa_outputs=tuple(oracle_dsa_outputs),
            layer_outputs=tuple(oracle_layer_outputs),
        ),
        CapsuleChainTrace(
            dsa_outputs=tuple(candidate_dsa_outputs),
            layer_outputs=tuple(candidate_layer_outputs),
            bridge_outputs=tuple(candidate_bridge_outputs),
            final_output=value,
        ),
    )


@dataclass(frozen=True, slots=True)
class A3NativeCapsuleLayerEvidence:
    """Concrete proof of per-layer model/state ownership."""

    layer_index: int
    layer_name: str
    candidate_backend: BackendKind
    attention_owner_id: int
    wrapper_owner_id: int
    prepared_weight_owner_id: int | None
    pypto_device_owner_id: int | None
    pypto_backend_owner_id: int | None
    parameter_addresses: tuple[int, ...]
    packed_weight_addresses: tuple[int, ...]
    shared_static_argument_addresses: tuple[int, ...]
    native_cache_addresses: tuple[int, ...]
    local_oracle_cache_addresses: tuple[int, ...]
    candidate_cache_addresses: tuple[int, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class A3NativeCapsuleHCLayerEvidence:
    """Runtime evidence that one layer used the production HC boundary."""

    layer_index: int
    layer_name: str
    hc_pre_operator: str
    rms_norm_operator: str
    hc_post_operator: str
    input_shape: tuple[int, ...]
    hidden_shape: tuple[int, ...]
    normalized_shape: tuple[int, ...]
    post_shape: tuple[int, ...]
    comb_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    input_dtype: str
    input_address: int
    residual_address: int
    hidden_address: int
    normalized_address: int
    post_address: int
    comb_address: int
    output_address: int
    rms_norm_owner_id: int
    rms_norm_weight_address: int
    rms_norm_weight_requires_grad: bool
    hc_parameter_addresses: tuple[int, ...]
    local_oracle_same_normalized_input: bool
    local_oracle_shared_residual_post_comb: bool

    @property
    def complete(self) -> bool:
        tokens = self.input_shape[0] if len(self.input_shape) == 3 else -1
        return (
            self.hc_pre_operator == HC_PRE_OPERATOR
            and self.rms_norm_operator == ASCEND_RMS_NORM_OPERATOR
            and self.hc_post_operator == HC_POST_OPERATOR
            and self.input_shape == (tokens, FLASH.hc_mult, FLASH.hidden_size)
            and self.hidden_shape == (tokens, FLASH.hidden_size)
            and self.normalized_shape == self.hidden_shape
            and self.post_shape == (tokens, FLASH.hc_mult)
            and self.comb_shape == (tokens, FLASH.hc_mult, FLASH.hc_mult)
            and self.output_shape == self.input_shape
            and self.input_dtype == str(torch.bfloat16)
            and all(
                address > 0
                for address in (
                    self.input_address,
                    self.residual_address,
                    self.hidden_address,
                    self.normalized_address,
                    self.post_address,
                    self.comb_address,
                    self.output_address,
                    self.rms_norm_owner_id,
                    self.rms_norm_weight_address,
                    *self.hc_parameter_addresses,
                )
            )
            and len(self.hc_parameter_addresses) == 3
            and not self.rms_norm_weight_requires_grad
            and self.local_oracle_same_normalized_input
            and self.local_oracle_shared_residual_post_comb
        )

    def to_dict(self) -> dict[str, object]:
        return {**asdict(self), "complete": self.complete}


def validate_native_capsule_resource_independence(
    plan: A3NativeCapsuleComparePlan,
    evidence: Sequence[A3NativeCapsuleLayerEvidence],
) -> None:
    """Require independent models and three disjoint state worlds per layer."""
    expected = tuple((layer.layer_index, layer.layer_name, layer.candidate_backend) for layer in plan.layers)
    _validate_resource_evidence_rows(expected, evidence)


def _validate_resource_evidence_rows(
    expected: Sequence[tuple[int, str, BackendKind]],
    evidence: Sequence[A3NativeCapsuleLayerEvidence],
) -> None:
    """Validate concrete ownership evidence against an ordered layer route."""
    expected_rows = tuple(expected)
    rows = tuple(evidence)
    if len(rows) != len(expected_rows):
        raise ValueError(f"expected {len(expected_rows)} evidence rows, got {len(rows)}")
    for (layer_index, layer_name, candidate_backend), row in zip(expected_rows, rows, strict=True):
        if (row.layer_index, row.layer_name) != (layer_index, layer_name):
            raise ValueError(f"evidence row does not match layer {layer_index}")
        if row.candidate_backend is not candidate_backend:
            raise ValueError(f"evidence backend does not match layer {layer_index}")
        if not row.parameter_addresses:
            raise ValueError(f"layer {row.layer_name} has no real parameter address evidence")
        if any(address <= 0 for address in row.parameter_addresses):
            raise ValueError(f"layer {row.layer_name} has an invalid real parameter address")
        if len(set(row.parameter_addresses)) != len(row.parameter_addresses):
            raise ValueError(f"layer {row.layer_name} aliases its own real parameter addresses")
        for field in ("attention_owner_id", "wrapper_owner_id"):
            value = getattr(row, field)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"layer {row.layer_name} has an invalid {field}")
        if row.candidate_backend is BackendKind.PYPTO:
            if (
                row.prepared_weight_owner_id is None
                or row.pypto_device_owner_id is None
                or row.pypto_backend_owner_id is None
                or not row.packed_weight_addresses
                or len(row.shared_static_argument_addresses) != len(SHARED_STATIC_ARGUMENT_NAMES)
            ):
                raise ValueError(f"PyPTO layer {row.layer_name} lacks prepared weight/runtime evidence")
            if any(address <= 0 for address in row.packed_weight_addresses):
                raise ValueError(f"PyPTO layer {row.layer_name} has an invalid packed weight address")
            if any(address <= 0 for address in row.shared_static_argument_addresses):
                raise ValueError(f"PyPTO layer {row.layer_name} has an invalid shared static argument address")
            if len(set(row.packed_weight_addresses)) != len(row.packed_weight_addresses):
                raise ValueError(f"PyPTO layer {row.layer_name} aliases its own packed weight arguments")
            if len(set(row.shared_static_argument_addresses)) != len(row.shared_static_argument_addresses):
                raise ValueError(f"PyPTO layer {row.layer_name} aliases distinct shared static arguments")
            for field in ("prepared_weight_owner_id", "pypto_device_owner_id", "pypto_backend_owner_id"):
                value = getattr(row, field)
                if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                    raise ValueError(f"PyPTO layer {row.layer_name} has an invalid {field}")
        elif (
            row.prepared_weight_owner_id is not None
            or row.pypto_device_owner_id is not None
            or row.pypto_backend_owner_id is not None
            or row.packed_weight_addresses
            or row.shared_static_argument_addresses
        ):
            raise ValueError(f"native candidate layer {row.layer_name} must not claim a PyPTO owner")
        cache_address_groups = (
            row.native_cache_addresses,
            row.local_oracle_cache_addresses,
            row.candidate_cache_addresses,
        )
        if any(len(addresses) != len(STATE_NAMES) for addresses in cache_address_groups):
            raise ValueError(f"layer {row.layer_name} must expose all six cache addresses in all three worlds")
        if any(len(set(addresses)) != len(addresses) for addresses in cache_address_groups):
            raise ValueError(f"layer {row.layer_name} aliases cache families within one state world")
        if any(address <= 0 for addresses in cache_address_groups for address in addresses):
            raise ValueError(f"layer {row.layer_name} has an invalid cache address")

    pypto_rows = tuple(row for row in rows if row.candidate_backend is BackendKind.PYPTO)
    layer_local_owner_ids = tuple(
        owner_id
        for row in rows
        for owner_id in (
            row.attention_owner_id,
            row.wrapper_owner_id,
            *((row.prepared_weight_owner_id,) if row.prepared_weight_owner_id is not None else ()),
        )
    )
    if len(set(layer_local_owner_ids)) != len(layer_local_owner_ids):
        raise ValueError("attention, wrapper, and prepared-weight owners must be independent across layers")
    prepared_owner_ids = tuple(row.prepared_weight_owner_id for row in pypto_rows)
    if len(set(prepared_owner_ids)) != len(prepared_owner_ids):
        raise ValueError("candidate PyPTO layers must own independent prepared weights")
    for field in ("pypto_device_owner_id", "pypto_backend_owner_id"):
        if pypto_rows and len({getattr(row, field) for row in pypto_rows}) != 1:
            raise ValueError(f"all candidate PyPTO layers must share one prepared {field}")
    shared_runtime_owner_ids = {
        owner_id for row in pypto_rows for owner_id in (row.pypto_device_owner_id, row.pypto_backend_owner_id)
    }
    if pypto_rows and (len(shared_runtime_owner_ids) != 2 or shared_runtime_owner_ids & set(layer_local_owner_ids)):
        raise ValueError("shared PyPTO device/backend owners must be distinct from every layer-local owner")
    shared_static_sets = tuple(set(row.shared_static_argument_addresses) for row in pypto_rows)
    if shared_static_sets and any(addresses != shared_static_sets[0] for addresses in shared_static_sets[1:]):
        raise ValueError("candidate PyPTO layers must share the same immutable RoPE/Hadamard arguments")

    parameter_sets = tuple(set(row.parameter_addresses) for row in rows)
    for left in range(len(parameter_sets)):
        for right in range(left + 1, len(parameter_sets)):
            overlap = parameter_sets[left] & parameter_sets[right]
            if overlap:
                raise ValueError(f"layers {left}/{right} alias real parameter storage: {tuple(sorted(overlap))}")

    packed_weight_sets = tuple((row.layer_index, set(row.packed_weight_addresses)) for row in pypto_rows)
    for left in range(len(packed_weight_sets)):
        left_layer, left_addresses = packed_weight_sets[left]
        for right in range(left + 1, len(packed_weight_sets)):
            right_layer, right_addresses = packed_weight_sets[right]
            overlap = left_addresses & right_addresses
            if overlap:
                raise ValueError(
                    f"PyPTO layers alias packed weight storage: {left_layer}/{right_layer}={tuple(sorted(overlap))}"
                )

    cache_sets = tuple(
        (row.layer_index, route, set(addresses))
        for row in rows
        for route, addresses in (
            ("native", row.native_cache_addresses),
            ("local-oracle", row.local_oracle_cache_addresses),
            (f"candidate-{row.candidate_backend.value}", row.candidate_cache_addresses),
        )
    )
    for left in range(len(cache_sets)):
        left_layer, left_route, left_addresses = cache_sets[left]
        for right in range(left + 1, len(cache_sets)):
            right_layer, right_route, right_addresses = cache_sets[right]
            overlap = left_addresses & right_addresses
            if overlap:
                raise ValueError(
                    "mutable cache storage aliases across Capsule worlds: "
                    f"{left_route}[{left_layer}]/{right_route}[{right_layer}]={tuple(sorted(overlap))}"
                )


@dataclass(frozen=True, slots=True)
class A3NativeCapsuleLayerComparison:
    layer_index: int
    layer_name: str
    dsa_output: TensorComparison
    layer_output: TensorComparison
    states: Mapping[str, TensorComparison]
    indexer_quantized: IndexerQuantizedComparison

    @property
    def close(self) -> bool:
        non_quantized_states_close = all(
            comparison.close for name, comparison in self.states.items() if name not in {"indexer_k", "indexer_scale"}
        )
        return (
            self.dsa_output.close
            and self.layer_output.close
            and non_quantized_states_close
            and self.indexer_quantized.acceptable
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "layer_index": self.layer_index,
            "layer_name": self.layer_name,
            "dsa_output": asdict(self.dsa_output),
            "layer_output": asdict(self.layer_output),
            "states": {name: asdict(value) for name, value in self.states.items()},
            "indexer_quantized": self.indexer_quantized.to_dict(),
            "close": self.close,
        }


def compare_native_capsule_traces(
    plan: A3NativeCapsuleComparePlan,
    local_oracle_trace: CapsuleLocalOracleTrace,
    candidate_trace: CapsuleChainTrace,
    local_oracle_states: Sequence[Mapping[str, torch.Tensor]],
    candidate_states: Sequence[Mapping[str, torch.Tensor]],
    initial_states: Sequence[Mapping[str, torch.Tensor]],
) -> tuple[A3NativeCapsuleLayerComparison, ...]:
    """Compare each candidate layer with a native oracle on identical input."""
    expected = len(plan.layers)
    values: tuple[Sequence[object], ...] = (
        local_oracle_trace.dsa_outputs,
        candidate_trace.dsa_outputs,
        local_oracle_trace.layer_outputs,
        candidate_trace.layer_outputs,
        local_oracle_states,
        candidate_states,
        initial_states,
    )
    if any(len(value) != expected for value in values):
        raise ValueError(f"local-oracle/candidate traces and state snapshots must contain exactly {expected} layers")
    result = []
    for index, layer in enumerate(plan.layers):
        result.append(
            A3NativeCapsuleLayerComparison(
                layer_index=index,
                layer_name=layer.layer_name,
                dsa_output=compare_tensors(
                    local_oracle_trace.dsa_outputs[index],
                    candidate_trace.dsa_outputs[index],
                    atol=plan.config.atol,
                    rtol=plan.config.rtol,
                ),
                layer_output=compare_tensors(
                    local_oracle_trace.layer_outputs[index],
                    candidate_trace.layer_outputs[index],
                    atol=plan.config.atol,
                    rtol=plan.config.rtol,
                ),
                states=compare_state_snapshots(
                    local_oracle_states[index],
                    candidate_states[index],
                    initial_states[index],
                    atol=plan.config.atol,
                    rtol=plan.config.rtol,
                ),
                indexer_quantized=compare_indexer_quantized(
                    local_oracle_states[index]["indexer_k"],
                    candidate_states[index]["indexer_k"],
                    initial_states[index]["indexer_k"],
                    local_oracle_states[index]["indexer_scale"],
                    candidate_states[index]["indexer_scale"],
                    initial_states[index]["indexer_scale"],
                ),
            )
        )
    return tuple(result)


@dataclass(frozen=True, slots=True)
class A3NativeCapsuleCompareResult:
    runtime: str
    device: int
    batch: int
    num_layers: int
    native_topology: str
    candidate_topology: str
    shell_kind: str
    local_oracle_same_candidate_inputs: bool
    candidate_sacrificial_warmup_reset: bool
    local_atol: float
    local_rtol: float
    chain_atol: float
    chain_rtol: float
    layers: tuple[A3NativeCapsuleLayerComparison, ...]
    final_output: TensorComparison
    resource_evidence: tuple[A3NativeCapsuleLayerEvidence, ...]
    real_hc_evidence: tuple[A3NativeCapsuleHCLayerEvidence, ...]

    @property
    def deterministic_bridge_only(self) -> bool:
        """Compatibility marker for the explicit legacy diagnostic shell."""
        return self.shell_kind == CapsuleShellKind.DETERMINISTIC_DIAGNOSTIC.value

    @property
    def topology_evidence_complete(self) -> bool:
        """Whether the serialized route is the exact runner route."""
        if self.native_topology != CapsuleTopology.NNNN.value:
            return False
        if (
            isinstance(self.num_layers, bool)
            or not isinstance(self.num_layers, int)
            or self.num_layers not in {1, 2, 4}
        ):
            return False
        try:
            candidate = CapsuleTopology(self.candidate_topology)
        except ValueError:
            return False
        if candidate not in {
            CapsuleTopology.PPPP,
            CapsuleTopology.NPNP,
            CapsuleTopology.PNPN,
        }:
            return False
        try:
            route = topology_backends(candidate, self.num_layers)
        except ValueError:
            return False
        return (
            self.runtime in SUPPORTED_RUNTIMES
            and not isinstance(self.device, bool)
            and isinstance(self.device, int)
            and self.device >= 0
            and self.batch == 4
            and len(self.layers) == self.num_layers
            and tuple(layer.layer_index for layer in self.layers) == tuple(range(self.num_layers))
            and all(layer.layer_name for layer in self.layers)
            and len({layer.layer_name for layer in self.layers}) == self.num_layers
            and len(self.resource_evidence) == self.num_layers
            and tuple(row.candidate_backend for row in self.resource_evidence) == route
        )

    @property
    def resource_evidence_complete(self) -> bool:
        """Validate ownership evidence without trusting a precomputed flag."""
        if not self.topology_evidence_complete:
            return False
        route = topology_backends(CapsuleTopology(self.candidate_topology), self.num_layers)
        expected = tuple(
            (layer.layer_index, layer.layer_name, backend) for layer, backend in zip(self.layers, route, strict=True)
        )
        try:
            _validate_resource_evidence_rows(expected, self.resource_evidence)
        except (TypeError, ValueError):
            return False
        return True

    @property
    def real_hc_complete(self) -> bool:
        if self.shell_kind != CapsuleShellKind.FIDELITY.value:
            return False
        if len(self.real_hc_evidence) != self.num_layers or len(self.layers) != self.num_layers:
            return False
        for layer, row in zip(self.layers, self.real_hc_evidence, strict=True):
            if (row.layer_index, row.layer_name) != (layer.layer_index, layer.layer_name) or not row.complete:
                return False
            if len(set(row.hc_parameter_addresses)) != len(row.hc_parameter_addresses):
                return False
        if len({row.rms_norm_owner_id for row in self.real_hc_evidence}) != self.num_layers:
            return False
        if len({row.rms_norm_weight_address for row in self.real_hc_evidence}) != self.num_layers:
            return False
        parameter_sets = tuple(set(row.hc_parameter_addresses) for row in self.real_hc_evidence)
        return all(
            not parameter_sets[left] & parameter_sets[right]
            for left in range(len(parameter_sets))
            for right in range(left + 1, len(parameter_sets))
        )

    @property
    def shell_evidence_complete(self) -> bool:
        if self.shell_kind == CapsuleShellKind.DETERMINISTIC_DIAGNOSTIC.value:
            return not self.real_hc_evidence
        return self.real_hc_complete

    @property
    def execution_contract_complete(self) -> bool:
        return (
            self.topology_evidence_complete
            and self.resource_evidence_complete
            and self.local_oracle_same_candidate_inputs is True
            and self.candidate_sacrificial_warmup_reset is True
            and self.shell_evidence_complete
            and all(tuple(layer.states) == STATE_NAMES for layer in self.layers)
        )

    @property
    def close(self) -> bool:
        return (
            self.execution_contract_complete and self.final_output.close and all(layer.close for layer in self.layers)
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "runtime": self.runtime,
            "device": self.device,
            "batch": self.batch,
            "num_layers": self.num_layers,
            "native_topology": self.native_topology,
            "candidate_topology": self.candidate_topology,
            "shell_kind": self.shell_kind,
            "deterministic_bridge_only": self.deterministic_bridge_only,
            "topology_evidence_complete": self.topology_evidence_complete,
            "resource_evidence_complete": self.resource_evidence_complete,
            "real_hc_complete": self.real_hc_complete,
            "shell_evidence_complete": self.shell_evidence_complete,
            "execution_contract_complete": self.execution_contract_complete,
            "local_oracle_same_candidate_inputs": self.local_oracle_same_candidate_inputs,
            "candidate_sacrificial_warmup_reset": self.candidate_sacrificial_warmup_reset,
            "local_atol": self.local_atol,
            "local_rtol": self.local_rtol,
            "chain_atol": self.chain_atol,
            "chain_rtol": self.chain_rtol,
            "layers": tuple(layer.to_dict() for layer in self.layers),
            "final_output": asdict(self.final_output),
            "resource_evidence": tuple(row.to_dict() for row in self.resource_evidence),
            "real_hc_evidence": tuple(row.to_dict() for row in self.real_hc_evidence),
            "close": self.close,
        }


@dataclass(slots=True)
class _RealHCShellLayer:
    """Small attention-half owner using the same operators as DeepSeek V4."""

    rms_norm: Any
    hc_fn: torch.Tensor
    hc_scale: torch.Tensor
    hc_base: torch.Tensor

    @property
    def ops(self) -> HCFidelityLayerOps:
        return HCFidelityLayerOps(
            hc_pre=self.hc_pre,
            rms_norm=self.normalize,
            hc_post=self.hc_post,
        )

    @property
    def parameter_addresses(self) -> tuple[int, ...]:
        return tuple(int(value.data_ptr()) for value in (self.hc_fn, self.hc_scale, self.hc_base))

    def hc_pre(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return torch.ops._C_ascend.npu_hc_pre_v2(
            value,
            self.hc_fn,
            self.hc_scale,
            self.hc_base,
            FLASH.hc_mult,
            FLASH.hc_sinkhorn_iters,
            FLASH.rms_norm_eps,
            FLASH.hc_eps,
        )

    def normalize(self, value: torch.Tensor) -> torch.Tensor:
        result = self.rms_norm(value)
        if not isinstance(result, torch.Tensor):
            raise TypeError("attention RMSNorm without residual must return one torch.Tensor")
        return result

    def hc_post(
        self,
        value: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
    ) -> torch.Tensor:
        # Keep the production wrapper's 3D-to-4D convention exactly.  The C++
        # HC-post contract accepts [B,T,D] plus [B,T,HC,D].
        return torch.ops._C_ascend.npu_hc_post(
            value.unsqueeze(0),
            residual.unsqueeze(0),
            post.unsqueeze(0),
            comb.unsqueeze(0),
        ).squeeze(0)


def _build_real_hc_shell_layer(
    *,
    seed: int,
    target: torch.device,
) -> _RealHCShellLayer:
    from vllm.model_executor.layers.layernorm import RMSNorm

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    hc_fn = (
        torch.randn(
            (FLASH.mix_hc, FLASH.hc_dim),
            dtype=torch.float32,
            generator=generator,
        )
        .mul_(0.01)
        .to(target)
    )
    hc_scale = torch.randn((3,), dtype=torch.float32, generator=generator).mul_(0.01).to(target)
    hc_base = torch.randn((FLASH.mix_hc,), dtype=torch.float32, generator=generator).mul_(0.01).to(target)
    rms_norm = RMSNorm(
        FLASH.hidden_size,
        eps=FLASH.rms_norm_eps,
        dtype=torch.bfloat16,
    ).to(target)
    if type(rms_norm).__name__ != ASCEND_RMS_NORM_OPERATOR:
        raise AssertionError(f"fidelity shell requires {ASCEND_RMS_NORM_OPERATOR}, got {type(rms_norm).__name__}")
    weight = (
        torch.randn(
            (FLASH.hidden_size,),
            dtype=torch.float32,
            generator=generator,
        )
        .mul_(0.01)
        .add_(1.0)
        .to(dtype=torch.bfloat16, device=target)
    )
    with torch.no_grad():
        rms_norm.weight.copy_(weight)
    rms_norm.weight.requires_grad_(False)
    rms_norm.eval()
    return _RealHCShellLayer(
        rms_norm=rms_norm,
        hc_fn=hc_fn,
        hc_scale=hc_scale,
        hc_base=hc_base,
    )


@dataclass(slots=True)
class _DeviceLayer:
    plan: A3NativeCapsuleLayerPlan
    attention: Any
    wrapper: Any
    native_world: Any
    local_oracle_world: Any
    candidate_world: Any
    initial_state: Mapping[str, torch.Tensor]
    candidate_pristine: tuple[torch.Tensor, ...]
    hc_shell: _RealHCShellLayer | None = None
    owner: Any | None = None
    candidate_metadata: Any | None = None


def _restore_world(world: Any, pristine: Sequence[torch.Tensor]) -> None:
    if len(pristine) != len(STATE_NAMES):
        raise ValueError("pristine state must contain six cache families")
    for tensor, source in zip(world.raw_cache_tuple, pristine, strict=True):
        tensor.copy_(source)


def _native_impl_call(
    wrapper: Any,
    world: Any,
    metadata_owner: Any,
    context: Any,
    hidden_states: torch.Tensor,
    output: torch.Tensor,
) -> None:
    """Call the unchanged production native implementation explicitly.

    A PyPTO candidate installs a private dispatch hook on the shared wrapper.
    The local oracle must not accidentally traverse that hook, so it invokes
    the exact native ``impl.forward`` target used by ``ops.dsa_forward`` while
    retaining the real cache tuple and metadata builder.
    """
    from vllm.forward_context import override_forward_context

    with torch.inference_mode(), override_forward_context(context):
        wrapper.dsa_attn.impl.forward(
            wrapper.dsa_attn.layer_name,
            hidden_states,
            world.raw_cache_tuple,
            list(metadata_owner.metadata),
            False,
            output,
        )


def _tensor_addresses(values: Mapping[str, object]) -> tuple[int, ...]:
    return tuple(sorted(int(value.data_ptr()) for value in values.values() if isinstance(value, torch.Tensor)))


def _collect_device_evidence(
    layers: Sequence[_DeviceLayer],
) -> tuple[A3NativeCapsuleLayerEvidence, ...]:
    result = []
    for layer in layers:
        is_pypto = layer.plan.candidate_backend is BackendKind.PYPTO
        if is_pypto and layer.owner is None:
            raise AssertionError(f"layer {layer.plan.layer_name} has no installed PyPTO owner")
        if not is_pypto and layer.owner is not None:
            raise AssertionError(f"native candidate layer {layer.plan.layer_name} unexpectedly has a PyPTO owner")
        parameters = tuple(sorted({int(parameter.data_ptr()) for parameter in layer.attention.parameters()}))
        owner = layer.owner
        launch_arguments = owner.weights.for_launch() if owner is not None else {}
        packed_weight_arguments = {
            name: tensor for name, tensor in launch_arguments.items() if name not in SHARED_STATIC_ARGUMENT_NAMES
        }
        shared_static_arguments = {
            name: tensor for name, tensor in launch_arguments.items() if name in SHARED_STATIC_ARGUMENT_NAMES
        }
        result.append(
            A3NativeCapsuleLayerEvidence(
                layer_index=layer.plan.layer_index,
                layer_name=layer.plan.layer_name,
                candidate_backend=layer.plan.candidate_backend,
                attention_owner_id=id(layer.attention),
                wrapper_owner_id=id(layer.wrapper),
                prepared_weight_owner_id=id(owner.weights) if owner is not None else None,
                pypto_device_owner_id=id(owner.device_owner) if owner is not None else None,
                pypto_backend_owner_id=id(owner.device_owner.backend) if owner is not None else None,
                parameter_addresses=parameters,
                packed_weight_addresses=_tensor_addresses(packed_weight_arguments),
                shared_static_argument_addresses=_tensor_addresses(shared_static_arguments),
                native_cache_addresses=tuple(int(tensor.data_ptr()) for tensor in layer.native_world.raw_cache_tuple),
                local_oracle_cache_addresses=tuple(
                    int(tensor.data_ptr()) for tensor in layer.local_oracle_world.raw_cache_tuple
                ),
                candidate_cache_addresses=tuple(
                    int(tensor.data_ptr()) for tensor in layer.candidate_world.raw_cache_tuple
                ),
            )
        )
    return tuple(result)


def _collect_real_hc_evidence(
    layers: Sequence[_DeviceLayer],
    local_oracle_trace: CapsuleLocalOracleTrace,
    candidate_trace: CapsuleChainTrace,
) -> tuple[A3NativeCapsuleHCLayerEvidence, ...]:
    if local_oracle_trace.hc_layers is not candidate_trace.hc_layers:
        raise AssertionError("local oracle and candidate did not retain one shared HC trace tuple")
    if len(candidate_trace.hc_layers) != len(layers) or len(candidate_trace.layer_outputs) != len(layers):
        raise AssertionError("real HC trace does not contain one row per Capsule layer")

    result = []
    for layer, hc_trace, output in zip(
        layers,
        candidate_trace.hc_layers,
        candidate_trace.layer_outputs,
        strict=True,
    ):
        shell = layer.hc_shell
        if shell is None:
            raise AssertionError(f"fidelity layer {layer.plan.layer_index} has no real HC shell owner")
        result.append(
            A3NativeCapsuleHCLayerEvidence(
                layer_index=layer.plan.layer_index,
                layer_name=layer.plan.layer_name,
                hc_pre_operator=HC_PRE_OPERATOR,
                rms_norm_operator=type(shell.rms_norm).__name__,
                hc_post_operator=HC_POST_OPERATOR,
                input_shape=tuple(hc_trace.input_value.shape),
                hidden_shape=tuple(hc_trace.hidden.shape),
                normalized_shape=tuple(hc_trace.normalized.shape),
                post_shape=tuple(hc_trace.post.shape),
                comb_shape=tuple(hc_trace.comb.shape),
                output_shape=tuple(output.shape),
                input_dtype=str(hc_trace.input_value.dtype),
                input_address=int(hc_trace.input_value.data_ptr()),
                residual_address=int(hc_trace.residual.data_ptr()),
                hidden_address=int(hc_trace.hidden.data_ptr()),
                normalized_address=int(hc_trace.normalized.data_ptr()),
                post_address=int(hc_trace.post.data_ptr()),
                comb_address=int(hc_trace.comb.data_ptr()),
                output_address=int(output.data_ptr()),
                rms_norm_owner_id=id(shell.rms_norm),
                rms_norm_weight_address=int(shell.rms_norm.weight.data_ptr()),
                rms_norm_weight_requires_grad=bool(shell.rms_norm.weight.requires_grad),
                hc_parameter_addresses=shell.parameter_addresses,
                local_oracle_same_normalized_input=True,
                local_oracle_shared_residual_post_comb=True,
            )
        )

    rows = tuple(result)
    for field in ("rms_norm_owner_id", "rms_norm_weight_address"):
        values = tuple(getattr(row, field) for row in rows)
        if len(set(values)) != len(values):
            raise AssertionError(f"real HC layers alias {field}")
    parameter_sets = tuple(set(row.hc_parameter_addresses) for row in rows)
    for left in range(len(parameter_sets)):
        for right in range(left + 1, len(parameter_sets)):
            if parameter_sets[left] & parameter_sets[right]:
                raise AssertionError(f"real HC parameter storage aliases across layers {left}/{right}")
    if not all(row.complete for row in rows):
        raise AssertionError("real HC runtime evidence is incomplete")
    return rows


def _require_nonzero_finite_trace(trace: CapsuleChainTrace, *, route: str) -> None:
    for index, tensor in enumerate(trace.dsa_outputs):
        host = tensor.detach().float().cpu()
        if not torch.isfinite(host).all() or float(host.abs().max()) == 0.0:
            raise AssertionError(f"{route} layer {index} produced zero or non-finite CSA output")
    final = trace.final_output.detach().float().cpu()
    if not torch.isfinite(final).all() or float(final.abs().max()) == 0.0:
        raise AssertionError(f"{route} Capsule produced zero or non-finite final output")


@dataclass(frozen=True, slots=True)
class _CapsuleCleanupOutcome:
    error: BaseException | None
    cleanup_owner: Any | None
    owner_closed: bool

    @property
    def retry_owner(self) -> Any | None:
        return None if self.owner_closed else self.cleanup_owner


def _attach_exception_attribute(error: BaseException, name: str, value: object) -> None:
    with suppress(AttributeError, TypeError):
        setattr(error, name, value)


def _cleanup_a3_native_capsule_compare(
    *,
    tp_initialized: bool,
    tp_initialization_attempted: bool,
    layers: Sequence[_DeviceLayer],
    fallback_owner: Any | None,
    get_owner: Callable[[], Any | None],
    synchronize: Callable[[], None],
    uninstall: Callable[[object], None],
    close_config: Callable[[], None],
    destroy_tp1: Callable[[], None],
) -> _CapsuleCleanupOutcome:
    """Best-effort cleanup that never loses an unclosed runtime owner."""
    failures: list[tuple[str, BaseException]] = []

    def record(phase: str, error: BaseException) -> None:
        failures.append((phase, error))

    owner = fallback_owner
    try:
        discovered_owner = get_owner()
    except BaseException as error:
        record("lookup PyPTO cleanup owner", error)
    else:
        if discovered_owner is not None:
            owner = discovered_owner
    if owner is None:
        layer_device_owners = {
            id(layer.owner.device_owner): layer.owner.device_owner for layer in layers if layer.owner is not None
        }
        if len(layer_device_owners) == 1:
            owner = next(iter(layer_device_owners.values()))

    owner_closed = False
    quiesced = not tp_initialized
    hooks_detached = not tp_initialized
    if tp_initialized:
        try:
            synchronize()
        except BaseException as error:
            record("synchronize before PyPTO close", error)
        else:
            quiesced = True

        if quiesced:
            hooks_detached = True
            for layer in layers:
                if layer.owner is None:
                    continue
                try:
                    uninstall(layer.wrapper)
                except BaseException as error:
                    hooks_detached = False
                    record(f"uninstall PyPTO layer {layer.plan.layer_index}", error)

        if quiesced and hooks_detached and owner is not None:
            try:
                owner.close()
            except BaseException as error:
                record("close PyPTO cleanup owner", error)
            else:
                owner_closed = True

    try:
        close_config()
    except BaseException as error:
        record("close synthetic config", error)
    if tp_initialization_attempted:
        try:
            destroy_tp1()
        except BaseException as error:
            record("destroy TP1 environment", error)

    cleanup_error = failures[0][1] if failures else None
    if cleanup_error is not None:
        for phase, additional in failures[1:]:
            with suppress(AttributeError):
                cleanup_error.add_note(f"additional cleanup failure during {phase}: {additional!r}")
        _attach_exception_attribute(
            cleanup_error,
            "decode_csa_additional_cleanup_errors",
            tuple(error for _, error in failures[1:]),
        )
    return _CapsuleCleanupOutcome(
        error=cleanup_error,
        cleanup_owner=owner,
        owner_closed=owner_closed,
    )


def _apply_capsule_cleanup_outcome(
    primary_error: BaseException | None,
    outcome: _CapsuleCleanupOutcome,
) -> None:
    """Preserve the primary failure and expose every retryable cleanup owner."""
    target_error = primary_error if primary_error is not None else outcome.error
    if target_error is not None and outcome.retry_owner is not None:
        _attach_exception_attribute(
            target_error,
            "decode_csa_cleanup_owner",
            outcome.retry_owner,
        )
    if outcome.error is None:
        return
    if primary_error is None:
        raise outcome.error
    _attach_exception_attribute(primary_error, "decode_csa_cleanup_error", outcome.error)
    with suppress(AttributeError):
        primary_error.add_note(f"Capsule cleanup also failed: {outcome.error!r}")


def run_a3_native_capsule_compare(
    config: A3NativeCapsuleCompareConfig,
    *,
    master_port: int = 29671,
) -> A3NativeCapsuleCompareResult:
    """Run one eager NNNN chain followed by the configured candidate chain.

    Fidelity mode executes the production HC-pre-v2/RMSNorm/HC-post boundary.
    The explicit deterministic diagnostic mode retains the original reduced
    shell.  This Phase-3 slice intentionally does not capture an ACLGraph or
    claim PP-distinct callable coverage.
    """
    if not isinstance(config, A3NativeCapsuleCompareConfig):
        raise TypeError("config must be A3NativeCapsuleCompareConfig")
    if isinstance(master_port, bool) or not isinstance(master_port, int) or not 1 <= master_port <= 65535:
        raise ValueError("master_port must be an integer in [1, 65535]")
    plan = build_a3_native_capsule_compare_plan(config)

    import torch_npu
    from vllm.config import set_current_vllm_config

    import vllm_ascend.ops.dsa  # noqa: F401 -- register torch.ops.vllm.dsa_forward
    from vllm_ascend.ops._pypto_dsv4_csa import (
        DecodeCSADeviceOwnerRegistry,
        install_pypto_dsv4_decode_csa,
        uninstall_pypto_dsv4_decode_csa,
    )
    from vllm_ascend.utils import register_ascend_customop

    config_owner = build_synthetic_vllm_config(
        batch=config.batch,
        prefix=tuple(layer.layer_name for layer in plan.layers),
    )
    registry = DecodeCSADeviceOwnerRegistry()
    tp_initialized = False
    tp_initialization_attempted = False
    primary_error: BaseException | None = None
    layers: list[_DeviceLayer] = []
    target = torch.device(f"npu:{config.device}")
    try:
        with set_current_vllm_config(config_owner.vllm_config):
            tp_initialization_attempted = True
            _initialize_tp1(device=config.device, master_port=master_port)
            tp_initialized = True
            register_ascend_customop(config_owner.vllm_config)
            spec = DecodeCSAProgramSpec(batch=config.batch)
            starts = (config.start_position,) * config.batch
            for layer_plan in plan.layers:
                # Every layer is constructed under its final unique prefix.
                # The synthetic ModelSlim description contains every selected
                # layer key, so static_forward_context and native RoPE
                # registration agree from construction onward; no post-hoc
                # registry retag exists.
                attention = build_real_ratio4_attention(
                    config_owner,
                    device=target,
                    prefix=layer_plan.layer_name,
                    seed=layer_plan.weight_seed,
                )
                wrapper = attention.dsa_attn
                native_world, candidate_world = build_twin_decode_csa_state_worlds(
                    spec,
                    device=target,
                    start_positions=starts,
                )
                local_oracle_world = build_decode_csa_state_world(
                    spec,
                    device=target,
                    start_positions=starts,
                )
                for source, destination in zip(
                    native_world.raw_cache_tuple,
                    local_oracle_world.raw_cache_tuple,
                    strict=True,
                ):
                    destination.copy_(source)
                layers.append(
                    _DeviceLayer(
                        plan=layer_plan,
                        attention=attention,
                        wrapper=wrapper,
                        native_world=native_world,
                        local_oracle_world=local_oracle_world,
                        candidate_world=candidate_world,
                        initial_state=local_oracle_world.snapshot(),
                        candidate_pristine=tuple(tensor.detach().clone() for tensor in candidate_world.raw_cache_tuple),
                        hc_shell=(
                            _build_real_hc_shell_layer(
                                seed=layer_plan.weight_seed + 6_700_001,
                                target=target,
                            )
                            if config.shell_kind is CapsuleShellKind.FIDELITY
                            else None
                        ),
                    )
                )

            generator = torch.Generator(device="cpu")
            generator.manual_seed(config.seed + 1)
            input_shape = (
                (spec.tokens, FLASH.hc_mult, FLASH.hidden_size)
                if config.shell_kind is CapsuleShellKind.FIDELITY
                else (spec.tokens, FLASH.hidden_size)
            )
            base_hidden = (
                torch.randn(
                    input_shape,
                    dtype=torch.bfloat16,
                    generator=generator,
                )
                .mul_(0.125)
                .to(target)
            )

            native_metadata_owners: list[Any] = []
            native_calls: list[LayerCall] = []
            for layer in layers:
                bind_state_world(layer.wrapper, layer.native_world)

                def native_call(value: torch.Tensor, *, layer: _DeviceLayer = layer) -> torch.Tensor:
                    # Build and immediately consume metadata because native
                    # per-token RoPE fields view a process-global runtime buffer.
                    metadata = build_real_native_metadata(
                        layer.wrapper,
                        layer.native_world,
                        config_owner.vllm_config,
                    )
                    native_metadata_owners.append(metadata)
                    context = _forward_context(layer.wrapper, metadata)
                    output = torch.empty_like(value)
                    _custom_op_call(layer.wrapper, context, value, output)
                    return output

                native_calls.append(native_call)
            hc_ops = tuple(layer.hc_shell.ops for layer in layers if layer.hc_shell is not None)
            if config.shell_kind is CapsuleShellKind.FIDELITY:
                if len(hc_ops) != len(layers):
                    raise AssertionError("fidelity Capsule did not construct every real HC shell")
                native_trace = execute_hc_fidelity_capsule_chain(
                    plan,
                    base_hidden.clone(),
                    native_calls,
                    hc_ops,
                )
            else:
                native_trace = execute_deterministic_capsule_chain(plan, base_hidden.clone(), native_calls)
            torch_npu.npu.synchronize(config.device)
            _require_nonzero_finite_trace(native_trace, route=CapsuleTopology.NNNN.value)

            # Every candidate layer gets its own state world even when its
            # route is native. Only P layers attach a PyPTO dispatch owner;
            # native layers continue through the unchanged custom-op fallback.
            for layer in layers:
                bind_state_world(layer.wrapper, layer.candidate_world)
                if layer.plan.candidate_backend is BackendKind.PYPTO:
                    metadata = build_real_native_metadata(
                        layer.wrapper,
                        layer.candidate_world,
                        config_owner.vllm_config,
                    )
                    layer.candidate_metadata = metadata
                    layer.owner = install_pypto_dsv4_decode_csa(
                        layer.wrapper,
                        kv_cache=layer.candidate_world.raw_cache_tuple,
                        attn_metadata=metadata.metadata,
                        device=config.device,
                        runtime=config.runtime,
                        batch_buckets=(config.batch,),
                        registry=registry,
                        uniform_query_rows_contract=True,
                    )
                    grad_packed = tuple(
                        name for name, tensor in layer.owner.weights.launch_arguments.items() if tensor.requires_grad
                    )
                    if grad_packed:
                        raise AssertionError(
                            f"layer {layer.plan.layer_index} packed tensors retained autograd: {grad_packed}"
                        )

            candidate_metadata_owners: list[Any] = []
            local_oracle_metadata_owners: list[Any] = []
            local_oracle_calls: list[LayerCall] = []
            candidate_calls = []
            for layer in layers:

                def local_oracle_call(
                    value: torch.Tensor,
                    *,
                    layer: _DeviceLayer = layer,
                ) -> torch.Tensor:
                    bind_state_world(layer.wrapper, layer.local_oracle_world)
                    metadata = build_real_native_metadata(
                        layer.wrapper,
                        layer.local_oracle_world,
                        config_owner.vllm_config,
                    )
                    local_oracle_metadata_owners.append(metadata)
                    context = _forward_context(layer.wrapper, metadata)
                    output = torch.empty_like(value)
                    _native_impl_call(
                        layer.wrapper,
                        layer.local_oracle_world,
                        metadata,
                        context,
                        value,
                        output,
                    )
                    return output

                local_oracle_calls.append(local_oracle_call)
                if layer.plan.candidate_backend is BackendKind.PYPTO:
                    assert layer.candidate_metadata is not None
                    context = _forward_context(layer.wrapper, layer.candidate_metadata)

                    def candidate_pypto_call(
                        value: torch.Tensor,
                        *,
                        layer: _DeviceLayer = layer,
                        context: Any = context,
                    ) -> torch.Tensor:
                        bind_state_world(layer.wrapper, layer.candidate_world)
                        output = torch.empty_like(value)
                        _custom_op_call(layer.wrapper, context, value, output)
                        return output

                    candidate_calls.append(candidate_pypto_call)
                else:

                    def candidate_native_call(
                        value: torch.Tensor,
                        *,
                        layer: _DeviceLayer = layer,
                    ) -> torch.Tensor:
                        bind_state_world(layer.wrapper, layer.candidate_world)
                        metadata = build_real_native_metadata(
                            layer.wrapper,
                            layer.candidate_world,
                            config_owner.vllm_config,
                        )
                        candidate_metadata_owners.append(metadata)
                        context = _forward_context(layer.wrapper, metadata)
                        output = torch.empty_like(value)
                        _custom_op_call(layer.wrapper, context, value, output)
                        return output

                    candidate_calls.append(candidate_native_call)

            # One full sacrificial chain exercises every layer's weight/cache
            # binding.  The shared static B4 specialization warms only once;
            # later layers are ordinary address patches on the same callable.
            if config.shell_kind is CapsuleShellKind.FIDELITY:
                execute_hc_fidelity_capsule_chain(
                    plan,
                    base_hidden.clone(),
                    candidate_calls,
                    hc_ops,
                )
            else:
                execute_deterministic_capsule_chain(plan, base_hidden.clone(), candidate_calls)
            torch_npu.npu.synchronize(config.device)
            owner = registry.get(device=config.device, runtime=config.runtime)
            has_pypto_candidate = any(layer.plan.candidate_backend is BackendKind.PYPTO for layer in layers)
            if has_pypto_candidate and owner is None:
                raise AssertionError("candidate PyPTO layers did not retain a shared device owner")
            if owner is not None:
                owner.mark_warmups_quiesced()
            for layer in layers:
                _restore_world(layer.candidate_world, layer.candidate_pristine)
            torch_npu.npu.synchronize(config.device)

            if config.shell_kind is CapsuleShellKind.FIDELITY:
                local_oracle_trace, candidate_trace = execute_hc_fidelity_capsule_with_local_oracles(
                    plan,
                    base_hidden.clone(),
                    local_oracle_calls,
                    candidate_calls,
                    hc_ops,
                )
            else:
                local_oracle_trace, candidate_trace = execute_capsule_with_local_oracles(
                    plan,
                    base_hidden.clone(),
                    local_oracle_calls,
                    candidate_calls,
                )
            torch_npu.npu.synchronize(config.device)
            _require_nonzero_finite_trace(candidate_trace, route=config.candidate_topology.value)
            local_oracle_states = tuple(layer.local_oracle_world.snapshot() for layer in layers)
            candidate_states = tuple(layer.candidate_world.snapshot() for layer in layers)

            evidence = _collect_device_evidence(layers)
            validate_native_capsule_resource_independence(plan, evidence)
            real_hc_evidence = (
                _collect_real_hc_evidence(layers, local_oracle_trace, candidate_trace)
                if config.shell_kind is CapsuleShellKind.FIDELITY
                else ()
            )
            comparisons = compare_native_capsule_traces(
                plan,
                local_oracle_trace,
                candidate_trace,
                local_oracle_states,
                candidate_states,
                tuple(layer.initial_state for layer in layers),
            )
            result = A3NativeCapsuleCompareResult(
                runtime=config.runtime,
                device=config.device,
                batch=config.batch,
                num_layers=config.num_layers,
                native_topology=CapsuleTopology.NNNN.value,
                candidate_topology=config.candidate_topology.value,
                shell_kind=config.shell_kind.value,
                local_oracle_same_candidate_inputs=True,
                candidate_sacrificial_warmup_reset=True,
                local_atol=config.atol,
                local_rtol=config.rtol,
                chain_atol=float(config.chain_atol),
                chain_rtol=config.chain_rtol,
                layers=comparisons,
                final_output=compare_tensors(
                    native_trace.final_output,
                    candidate_trace.final_output,
                    atol=float(config.chain_atol),
                    rtol=config.chain_rtol,
                ),
                resource_evidence=evidence,
                real_hc_evidence=real_hc_evidence,
            )
            if not result.close:
                raise AssertionError(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
            return result
    except BaseException as error:
        primary_error = error
        raise
    finally:
        fallback_owner = getattr(primary_error, "decode_csa_cleanup_owner", None) if primary_error is not None else None
        outcome = _cleanup_a3_native_capsule_compare(
            tp_initialized=tp_initialized,
            tp_initialization_attempted=tp_initialization_attempted,
            layers=layers,
            fallback_owner=fallback_owner,
            get_owner=lambda: registry.get(device=config.device, runtime=config.runtime),
            synchronize=lambda: torch_npu.npu.synchronize(config.device),
            uninstall=uninstall_pypto_dsv4_decode_csa,
            close_config=config_owner.close,
            destroy_tp1=_destroy_tp1,
        )
        _apply_capsule_cleanup_outcome(primary_error, outcome)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", choices=SUPPORTED_RUNTIMES, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument(
        "--candidate-topology",
        choices=(CapsuleTopology.PPPP.value, CapsuleTopology.NPNP.value, CapsuleTopology.PNPN.value),
        default=CapsuleTopology.PPPP.value,
    )
    parser.add_argument(
        "--shell-kind",
        choices=tuple(kind.value for kind in CapsuleShellKind),
        default=CapsuleShellKind.FIDELITY.value,
        help="fidelity uses real HC-pre/RMSNorm/HC-post; deterministic_diagnostic keeps the legacy reduced shell",
    )
    parser.add_argument("--start-position", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--atol", type=float, default=0.1)
    parser.add_argument("--rtol", type=float, default=0.1)
    parser.add_argument(
        "--chain-atol",
        type=float,
        default=None,
        help="end-to-end absolute tolerance; default is local atol * sqrt(num_layers)",
    )
    parser.add_argument("--chain-rtol", type=float, default=0.1)
    parser.add_argument("--master-port", type=int, default=29671)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = A3NativeCapsuleCompareConfig(
        runtime=args.runtime,
        device=args.device,
        batch=args.batch,
        num_layers=args.num_layers,
        start_position=args.start_position,
        seed=args.seed,
        atol=args.atol,
        rtol=args.rtol,
        chain_atol=args.chain_atol,
        chain_rtol=args.chain_rtol,
        candidate_topology=CapsuleTopology(args.candidate_topology),
        shell_kind=CapsuleShellKind(args.shell_kind),
    )
    result = run_a3_native_capsule_compare(config, master_port=args.master_port)
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "A3NativeCapsuleCompareConfig",
    "A3NativeCapsuleComparePlan",
    "A3NativeCapsuleCompareResult",
    "A3NativeCapsuleHCLayerEvidence",
    "A3NativeCapsuleLayerComparison",
    "A3NativeCapsuleLayerEvidence",
    "A3NativeCapsuleLayerPlan",
    "CapsuleChainTrace",
    "CapsuleHCLayerTrace",
    "CapsuleLocalOracleTrace",
    "CapsuleShellKind",
    "HCFidelityLayerOps",
    "build_a3_native_capsule_compare_plan",
    "compare_native_capsule_traces",
    "execute_capsule_with_local_oracles",
    "execute_deterministic_capsule_chain",
    "execute_hc_fidelity_capsule_chain",
    "execute_hc_fidelity_capsule_with_local_oracles",
    "run_a3_native_capsule_compare",
    "validate_native_capsule_resource_independence",
]
