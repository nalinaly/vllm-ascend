# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Fresh-process A3 native-vs-PyPTO comparison for synthetic DSV4 CSA.

Run this module once per runtime.  Native execution is deliberately completed
before the PyPTO owner is installed: there is no fallback after mutable state
writes, and a fresh process avoids cross-runtime DSO/registry contamination.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Any

import torch

from tests.pypto_dsv4_decode_csa.indexer_quantized_comparison import (
    IndexerQuantizedComparison,
    IndexerQuantizedLongChainReport,
    compare_indexer_quantized,
    compare_indexer_quantized_long_chain,
)
from tests.pypto_dsv4_decode_csa.native_fixture import (
    DEFAULT_LAYER_PREFIX,
    STATE_NAMES,
    bind_state_world,
    build_real_native_metadata,
    build_real_ratio4_attention,
    build_synthetic_vllm_config,
    build_twin_decode_csa_state_worlds,
    retarget_decode_csa_state_world,
)
from vllm_ascend.ops._pypto_dsv4_csa import DecodeCSAProgramSpec
from vllm_ascend.ops._pypto_dsv4_csa.config import FLASH

SUPPORTED_RUNTIMES = ("tensormap_and_ringbuffer", "host_build_graph")
NATIVE_HOT_OPERATORS = (
    "torch_npu.npu_dynamic_quant",
    "torch_npu.npu_quant_matmul",
    "torch.ops._C_ascend.npu_rms_norm_dynamic_quant",
    "torch.ops._C_ascend.inplace_partial_rotary_mul",
    "torch.ops._C_ascend.compressor_metadata",
    "torch.ops._C_ascend.compressor",
    "torch.ops._C_ascend.npu_scatter_nd_update_v2",
    "torch.ops._C_ascend.npu_vllm_quant_lightning_indexer",
    "torch.ops._C_ascend.npu_sparse_attn_sharedkv",
    "torch_npu.npu_transpose_batchmatmul",
)


@dataclass(frozen=True, slots=True)
class TensorComparison:
    active_elements: int
    total_elements: int
    max_abs_error: float
    mean_abs_error: float
    max_relative_error: float
    reference_max_abs: float
    candidate_max_abs: float
    finite: bool
    close: bool


@dataclass(frozen=True, slots=True)
class A3NativeCompareResult:
    runtime: str
    device: int
    batch: int
    seq: int
    start_position: int
    correctness_steps: int
    sync_between_correctness_steps: bool
    diagnose_step_states: bool
    pypto_sacrificial_warmup_reset: bool
    rope_dtype: str
    native_ms: float
    pypto_ms: float
    speedup: float
    output: TensorComparison
    states: Mapping[str, TensorComparison]
    indexer_quantized: IndexerQuantizedComparison
    native_hot_operators: tuple[str, ...]
    indexer_quantized_long_chain: IndexerQuantizedLongChainReport | None = None

    @property
    def close(self) -> bool:
        non_quantized_states_close = all(
            comparison.close for name, comparison in self.states.items() if name not in {"indexer_k", "indexer_scale"}
        )
        return self.output.close and non_quantized_states_close and self.indexer_quantized.acceptable

    def to_dict(self) -> dict[str, Any]:
        return {
            "runtime": self.runtime,
            "device": self.device,
            "batch": self.batch,
            "seq": self.seq,
            "start_position": self.start_position,
            "correctness_steps": self.correctness_steps,
            "sync_between_correctness_steps": self.sync_between_correctness_steps,
            "diagnose_step_states": self.diagnose_step_states,
            "pypto_sacrificial_warmup_reset": self.pypto_sacrificial_warmup_reset,
            "rope_dtype": self.rope_dtype,
            "native_ms": self.native_ms,
            "pypto_ms": self.pypto_ms,
            "speedup": self.speedup,
            "output": asdict(self.output),
            "states": {name: asdict(value) for name, value in self.states.items()},
            "indexer_quantized": self.indexer_quantized.to_dict(),
            "indexer_quantized_long_chain": (
                self.indexer_quantized_long_chain.to_dict() if self.indexer_quantized_long_chain is not None else None
            ),
            "native_hot_operators": self.native_hot_operators,
            "close": self.close,
        }


def compare_tensors(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    *,
    initial: torch.Tensor | None = None,
    atol: float,
    rtol: float,
) -> TensorComparison:
    """Compare the union of elements changed by either implementation."""
    if reference.shape != candidate.shape or reference.dtype != candidate.dtype:
        raise ValueError(
            "comparison tensors must share shape/dtype, got "
            f"{tuple(reference.shape)}/{reference.dtype} and {tuple(candidate.shape)}/{candidate.dtype}"
        )
    reference_cpu = reference.detach().cpu()
    candidate_cpu = candidate.detach().cpu()
    if initial is None:
        active = torch.ones_like(reference_cpu, dtype=torch.bool)
    else:
        initial_cpu = initial.detach().cpu()
        if initial_cpu.shape != reference_cpu.shape or initial_cpu.dtype != reference_cpu.dtype:
            raise ValueError("initial tensor must share comparison shape and dtype")
        active = reference_cpu.ne(initial_cpu) | candidate_cpu.ne(initial_cpu)
    active_count = int(active.sum().item())
    total = reference_cpu.numel()
    if active_count == 0:
        reference_values = reference_cpu.reshape(-1).to(torch.float64)
        candidate_values = candidate_cpu.reshape(-1).to(torch.float64)
        error_values = torch.zeros(1, dtype=torch.float64)
        close = torch.equal(reference_cpu, candidate_cpu)
    else:
        reference_values = reference_cpu[active].to(torch.float64)
        candidate_values = candidate_cpu[active].to(torch.float64)
        error_values = (reference_values - candidate_values).abs()
        if reference_cpu.is_floating_point():
            close = torch.allclose(reference_values, candidate_values, atol=atol, rtol=rtol)
        else:
            close = torch.equal(reference_values, candidate_values)
    denominator = reference_values.abs().clamp_min(1.0e-12)
    relative = error_values / denominator if active_count else error_values
    finite = bool(torch.isfinite(reference_values).all() and torch.isfinite(candidate_values).all())
    return TensorComparison(
        active_elements=active_count,
        total_elements=total,
        max_abs_error=float(error_values.max().item()),
        mean_abs_error=float(error_values.mean().item()),
        max_relative_error=float(relative.max().item()),
        reference_max_abs=float(reference_values.abs().max().item()),
        candidate_max_abs=float(candidate_values.abs().max().item()),
        finite=finite,
        close=bool(close and finite),
    )


def compare_state_snapshots(
    reference: Mapping[str, torch.Tensor],
    candidate: Mapping[str, torch.Tensor],
    initial: Mapping[str, torch.Tensor],
    *,
    atol: float,
    rtol: float,
) -> Mapping[str, TensorComparison]:
    if tuple(reference) != STATE_NAMES or tuple(candidate) != STATE_NAMES or tuple(initial) != STATE_NAMES:
        raise ValueError(f"state snapshots must preserve the six-family order {STATE_NAMES}")
    return MappingProxyType(
        {
            name: compare_tensors(reference[name], candidate[name], initial=initial[name], atol=atol, rtol=rtol)
            for name in STATE_NAMES
        }
    )


def _error_summary(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float | int | bool]:
    reference_f64 = reference.to(torch.float64).reshape(-1)
    candidate_f64 = candidate.to(torch.float64).reshape(-1)
    if reference_f64.numel() == 0:
        return {
            "elements": 0,
            "max_abs_error": 0.0,
            "mean_abs_error": 0.0,
            "max_relative_error": 0.0,
            "finite": True,
        }
    absolute = (reference_f64 - candidate_f64).abs()
    relative = absolute / reference_f64.abs().clamp_min(1.0e-12)
    return {
        "elements": reference_f64.numel(),
        "max_abs_error": float(absolute.max().item()),
        "mean_abs_error": float(absolute.mean().item()),
        "max_relative_error": float(relative.max().item()),
        "finite": bool(torch.isfinite(reference_f64).all() and torch.isfinite(candidate_f64).all()),
    }


def _rope_row_summary(tensor: torch.Tensor, position: int) -> Mapping[str, Any]:
    """Return compact values that identify the exact qkv RoPE cache row."""
    row = tensor[position].detach().to(dtype=torch.float32, device="cpu").reshape(-1)
    return MappingProxyType(
        {
            "position": position,
            "elements": row.numel(),
            "min": float(row.min().item()),
            "max": float(row.max().item()),
            "mean": float(row.mean().item()),
            "first_eight": tuple(float(value) for value in row[:8].tolist()),
        }
    )


def _state_row_view(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().cpu().reshape(-1, tensor.shape[-1])


def expected_state_row_positions(world: Any) -> Mapping[str, Mapping[int, tuple[Mapping[str, int], ...]]]:
    """Reverse the fixture's physical rows back to request/token positions."""
    family_by_state = {
        "compressed_kv": "compressed",
        "swa_kv": "swa",
        "compressor_state": "main_state",
        "indexer_compressor_state": "inner_state",
        "indexer_k": "indexer",
        "indexer_scale": "indexer",
    }
    positions = world.positions.detach().cpu().reshape(-1)
    result: dict[str, Mapping[int, tuple[Mapping[str, int], ...]]] = {}
    for state_name, family in family_by_state.items():
        rows = world.linear_slot_mappings[family].detach().cpu().reshape(-1)
        grouped: dict[int, list[Mapping[str, int]]] = {}
        for token_index, (row, absolute_position) in enumerate(zip(rows.tolist(), positions.tolist(), strict=True)):
            grouped.setdefault(int(row), []).append(
                {
                    "request": token_index // world.spec.seq,
                    "token_offset": token_index % world.spec.seq,
                    "absolute_position": int(absolute_position),
                }
            )
        result[state_name] = MappingProxyType({row: tuple(locations) for row, locations in grouped.items()})
    return MappingProxyType(result)


def expected_state_row_positions_for_worlds(
    worlds: Sequence[Any],
) -> Mapping[str, Mapping[int, tuple[Mapping[str, int], ...]]]:
    """Merge physical-row provenance across consecutive decode metadata."""
    merged: dict[str, dict[int, list[Mapping[str, int]]]] = {name: {} for name in STATE_NAMES}
    for step, world in enumerate(worlds):
        for state_name, rows in expected_state_row_positions(world).items():
            for row, locations in rows.items():
                merged[state_name].setdefault(row, []).extend(
                    dict(location, decode_step=step) for location in locations
                )
    return MappingProxyType(
        {
            state_name: MappingProxyType({row: tuple(locations) for row, locations in rows.items()})
            for state_name, rows in merged.items()
        }
    )


def diagnose_state_snapshots(
    reference: Mapping[str, torch.Tensor],
    candidate: Mapping[str, torch.Tensor],
    initial: Mapping[str, torch.Tensor],
    *,
    row_positions: Mapping[str, Mapping[int, tuple[Mapping[str, int], ...]]] | None = None,
) -> Mapping[str, Mapping[str, Any]]:
    """Explain write-set and same-row errors without hiding them in raw maxima."""
    if tuple(reference) != STATE_NAMES or tuple(candidate) != STATE_NAMES or tuple(initial) != STATE_NAMES:
        raise ValueError(f"state snapshots must preserve the six-family order {STATE_NAMES}")
    diagnostics: dict[str, Mapping[str, Any]] = {}
    for name in STATE_NAMES:
        reference_rows = _state_row_view(reference[name])
        candidate_rows = _state_row_view(candidate[name])
        initial_rows = _state_row_view(initial[name])
        reference_written = reference_rows.ne(initial_rows).any(dim=1)
        candidate_written = candidate_rows.ne(initial_rows).any(dim=1)
        write_union = reference_written | candidate_written
        write_intersection = reference_written & candidate_written
        reference_only = reference_written & ~candidate_written
        candidate_only = candidate_written & ~reference_written
        differing = reference_rows.ne(candidate_rows).any(dim=1)
        first_row = int(torch.nonzero(differing, as_tuple=False)[0].item()) if bool(differing.any()) else None
        first_column = None
        first_reference = None
        first_candidate = None
        first_address = None
        if first_row is not None:
            first_column = int(
                torch.nonzero(reference_rows[first_row].ne(candidate_rows[first_row]), as_tuple=False)[0].item()
            )
            first_reference = reference_rows[first_row, first_column].item()
            first_candidate = candidate_rows[first_row, first_column].item()
            rows_per_block = 2 if name in {"compressor_state", "indexer_compressor_state"} else 32
            first_address = {
                "physical_block": first_row // rows_per_block,
                "block_offset": first_row % rows_per_block,
            }
        common_reference = reference_rows[write_intersection]
        common_candidate = candidate_rows[write_intersection]
        item: dict[str, Any] = {
            "reference_written_rows": int(reference_written.sum().item()),
            "candidate_written_rows": int(candidate_written.sum().item()),
            "write_union_rows": int(write_union.sum().item()),
            "write_intersection_rows": int(write_intersection.sum().item()),
            "reference_only_rows": int(reference_only.sum().item()),
            "candidate_only_rows": int(candidate_only.sum().item()),
            "reference_only_row_indices": tuple(
                int(value) for value in torch.nonzero(reference_only, as_tuple=False).reshape(-1).tolist()
            ),
            "candidate_only_row_indices": tuple(
                int(value) for value in torch.nonzero(candidate_only, as_tuple=False).reshape(-1).tolist()
            ),
            "first_differing_row": first_row,
            "first_differing_column": first_column,
            "first_reference_value": first_reference,
            "first_candidate_value": first_candidate,
            "first_differing_address": first_address,
            "first_expected_token_positions": (
                row_positions[name].get(first_row, ()) if row_positions is not None and first_row is not None else ()
            ),
            "same_written_rows": _error_summary(common_reference, common_candidate),
        }
        if name in {"compressed_kv", "swa_kv"}:
            active_reference = reference_rows[write_union]
            active_candidate = candidate_rows[write_union]
            item["segments"] = {
                "nope_0_448": _error_summary(active_reference[:, :448], active_candidate[:, :448]),
                "rope_448_512": _error_summary(active_reference[:, 448:], active_candidate[:, 448:]),
            }
        diagnostics[name] = MappingProxyType(item)

    # The dequantized view uses the union of K and scale writes; comparing raw
    # INT8 alone exaggerates a one-bin difference when the row scale is small.
    reference_k = _state_row_view(reference["indexer_k"]).to(torch.float32)
    candidate_k = _state_row_view(candidate["indexer_k"]).to(torch.float32)
    initial_k = _state_row_view(initial["indexer_k"])
    reference_scale = _state_row_view(reference["indexer_scale"]).to(torch.float32)
    candidate_scale = _state_row_view(candidate["indexer_scale"]).to(torch.float32)
    initial_scale = _state_row_view(initial["indexer_scale"])
    dequant_rows = (
        reference_k.ne(initial_k).any(dim=1)
        | candidate_k.ne(initial_k).any(dim=1)
        | reference_scale.ne(initial_scale).any(dim=1)
        | candidate_scale.ne(initial_scale).any(dim=1)
    )
    indexer_item = dict(diagnostics["indexer_k"])
    indexer_item["dequantized_int8_x_scale"] = _error_summary(
        reference_k[dequant_rows] * reference_scale[dequant_rows],
        candidate_k[dequant_rows] * candidate_scale[dequant_rows],
    )
    diagnostics["indexer_k"] = MappingProxyType(indexer_item)
    return MappingProxyType(diagnostics)


def diagnose_state_transitions(
    initial: Mapping[str, torch.Tensor],
    snapshots: Sequence[Mapping[str, torch.Tensor]],
    worlds: Sequence[Any],
) -> tuple[Mapping[str, Mapping[str, Any]], ...]:
    """Identify writes outside each decode step's physical target rows."""
    if len(snapshots) != len(worlds):
        raise ValueError("state snapshots and metadata worlds must have equal length")
    previous = initial
    transitions: list[Mapping[str, Mapping[str, Any]]] = []
    for step, (current, world) in enumerate(zip(snapshots, worlds, strict=True)):
        expected = expected_state_row_positions(world)
        families: dict[str, Mapping[str, Any]] = {}
        for name in STATE_NAMES:
            current_rows = _state_row_view(current[name])
            previous_rows = _state_row_view(previous[name])
            changed = current_rows.ne(previous_rows).any(dim=1)
            expected_rows = torch.zeros_like(changed)
            if expected[name]:
                expected_rows[list(expected[name])] = True
            outside = changed & ~expected_rows
            first_outside = int(torch.nonzero(outside, as_tuple=False)[0].item()) if bool(outside.any()) else None
            first_column = None
            previous_value = None
            current_value = None
            if first_outside is not None:
                first_column = int(
                    torch.nonzero(
                        current_rows[first_outside].ne(previous_rows[first_outside]),
                        as_tuple=False,
                    )[0].item()
                )
                previous_value = previous_rows[first_outside, first_column].item()
                current_value = current_rows[first_outside, first_column].item()
            families[name] = MappingProxyType(
                {
                    "decode_step": step,
                    "start_position": int(world.start_positions[0].item()),
                    "changed_rows": int(changed.sum().item()),
                    "expected_target_rows": len(expected[name]),
                    "changed_outside_target_rows": int(outside.sum().item()),
                    "outside_target_row_indices": tuple(
                        int(value) for value in torch.nonzero(outside, as_tuple=False).reshape(-1).tolist()
                    ),
                    "first_outside_row": first_outside,
                    "first_outside_column": first_column,
                    "first_outside_previous_value": previous_value,
                    "first_outside_current_value": current_value,
                }
            )
        transitions.append(MappingProxyType(families))
        previous = current
    return tuple(transitions)


def _initialize_tp1(*, device: int, master_port: int) -> None:
    import torch_npu
    from vllm.distributed import init_distributed_environment, initialize_model_parallel
    from vllm.distributed.parallel_state import model_parallel_is_initialized

    from vllm_ascend.utils import enable_custom_op

    if torch.distributed.is_initialized() or model_parallel_is_initialized():
        raise RuntimeError("a3_native_compare must start in a fresh process without an existing process group")
    os.environ.update(
        {
            "RANK": "0",
            "LOCAL_RANK": str(device),
            "WORLD_SIZE": "1",
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(master_port),
        }
    )
    torch_npu.npu.set_device(device)
    # This is the same bootstrap used by the production worker.  Host tests
    # stop at the object/layout contract; they must not replace these registered
    # metadata operators with hand-derived SimpleNamespace fixtures, because
    # the point of this runner is to execute the real builder protocol.
    if not enable_custom_op():
        raise RuntimeError("vllm_ascend custom-op bootstrap failed before native CSA metadata construction")
    init_distributed_environment(
        world_size=1,
        rank=0,
        local_rank=device,
        backend="hccl",
    )
    initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)


def _destroy_tp1() -> None:
    from vllm.distributed.parallel_state import destroy_distributed_environment, destroy_model_parallel

    destroy_model_parallel()
    destroy_distributed_environment()


def _forward_context(wrapper: Any, metadata: Any) -> Any:
    from vllm.forward_context import ForwardContext

    context = ForwardContext(
        no_compile_layers={wrapper.prefix: wrapper},
        attn_metadata=dict(metadata.by_name),
        slot_mapping={},
    )
    # set_ascend_forward_context normally mirrors this scheduler value onto
    # the V1 ForwardContext.  Ascend's _EXTRA_CTX proxy reads it while native
    # o_proj restores the flattened token extent.
    context.num_tokens = metadata.metadata[0].num_actual_tokens
    context.capturing = False
    return context


def _custom_op_call(
    wrapper: Any,
    context: Any,
    hidden_states: torch.Tensor,
    output: torch.Tensor,
) -> None:
    from vllm.forward_context import override_forward_context

    # AscendWorker.execute_model carries the same process-wide inference
    # boundary in production; retain it for both sides of this comparison.
    with torch.inference_mode(), override_forward_context(context):
        torch.ops.vllm.dsa_forward(hidden_states, False, output, wrapper.prefix)


def _serial_wall_time_ms(
    callback: Any,
    *,
    warmups: int,
    iterations: int,
    device: int,
) -> float:
    import torch_npu

    for _ in range(warmups):
        callback()
    torch_npu.npu.synchronize(device)
    start = time.perf_counter()
    for _ in range(iterations):
        callback()
    torch_npu.npu.synchronize(device)
    return (time.perf_counter() - start) * 1000.0 / iterations


def run_a3_native_compare(
    *,
    runtime: str,
    device: int = 0,
    batch: int = 4,
    warmups: int = 2,
    iterations: int = 10,
    atol: float = 0.1,
    rtol: float = 0.1,
    seed: int = 20260902,
    master_port: int = 29641,
    start_position: int = 0,
    correctness_steps: int = 1,
    sync_between_correctness_steps: bool = False,
    diagnose_step_states: bool = False,
) -> A3NativeCompareResult:
    """Run native first, then PyPTO, against independent equal cache worlds."""
    import torch_npu
    from vllm.config import set_current_vllm_config

    import vllm_ascend.ops.dsa  # noqa: F401 -- register torch.ops.vllm.dsa_forward
    from vllm_ascend.ops._pypto_dsv4_csa import (
        DecodeCSADeviceOwnerRegistry,
        install_pypto_dsv4_decode_csa,
        uninstall_pypto_dsv4_decode_csa,
    )
    from vllm_ascend.utils import register_ascend_customop

    if runtime not in SUPPORTED_RUNTIMES:
        raise ValueError(f"runtime must be one of {SUPPORTED_RUNTIMES}, got {runtime!r}")
    # The fixture, metadata builders, cache partitioner and production static
    # contract all support the complete decode bucket family.  Keep runtime
    # validation delegated to DecodeCSAProgramSpec below instead of preserving
    # the old B4-only bring-up restriction: B8/B12/B16 numerical runs are
    # required to prove the final callable ABI, not merely compilation.
    spec = DecodeCSAProgramSpec(batch=batch)
    if warmups < 0 or iterations <= 0:
        raise ValueError("warmups must be non-negative and iterations must be positive")
    if correctness_steps <= 0:
        raise ValueError("correctness_steps must be positive")

    config_owner = build_synthetic_vllm_config(batch=batch)
    registry = DecodeCSADeviceOwnerRegistry()
    owner = None
    tp_initialized = False
    target = torch.device(f"npu:{device}")
    try:
        with set_current_vllm_config(config_owner.vllm_config):
            _initialize_tp1(device=device, master_port=master_port)
            tp_initialized = True
            # Production AscendWorker registers the pluggable-layer classes
            # before constructing the model.  enable_custom_op() only loads the
            # C extension and is not a substitute for this class registry.
            register_ascend_customop(config_owner.vllm_config)
            attention = build_real_ratio4_attention(
                config_owner,
                device=target,
                prefix=DEFAULT_LAYER_PREFIX,
                seed=seed,
            )
            wrapper = attention.dsa_attn
            native_world, pypto_world = build_twin_decode_csa_state_worlds(
                spec,
                device=target,
                start_positions=(start_position,) * batch,
            )
            initial_state = native_world.snapshot()
            pypto_initial_device = tuple(tensor.detach().clone() for tensor in pypto_world.raw_cache_tuple)
            native_worlds = [native_world]
            pypto_worlds = [pypto_world]
            for step in range(1, correctness_steps):
                step_start = start_position + step * spec.seq
                native_worlds.append(
                    retarget_decode_csa_state_world(
                        native_world,
                        start_positions=(step_start,) * batch,
                    )
                )
                pypto_worlds.append(
                    retarget_decode_csa_state_world(
                        pypto_world,
                        start_positions=(step_start,) * batch,
                    )
                )
            host_generator = torch.Generator(device="cpu")
            host_generator.manual_seed(seed + 1)
            hidden_states_by_step = [
                (
                    torch.randn(
                        (spec.tokens, FLASH.hidden_size),
                        dtype=torch.bfloat16,
                        generator=host_generator,
                    )
                    .mul_(0.125)
                    .to(target)
                )
                for _ in range(correctness_steps)
            ]
            native_outputs = [torch.empty_like(hidden_states) for hidden_states in hidden_states_by_step]
            pypto_outputs = [torch.empty_like(hidden_states) for hidden_states in hidden_states_by_step]

            # Native correctness and timing must complete before owner install.
            bind_state_world(wrapper, native_world)
            # Decode metadata owns views into rope_dsv4's process-level runtime
            # buffer.  A later metadata build intentionally reuses that buffer,
            # exactly as the production runner does on the next iteration.  Do
            # not prebuild multiple decode steps: doing so would overwrite the
            # earlier step's per-token cos/sin before its native forward.
            native_metadatas = []
            native_calls = []
            native_step_states: list[Mapping[str, torch.Tensor]] = []
            for step, (world, hidden_states, output) in enumerate(
                zip(native_worlds, hidden_states_by_step, native_outputs, strict=True)
            ):
                metadata = build_real_native_metadata(wrapper, world, config_owner.vllm_config)
                native_metadatas.append(metadata)
                context = _forward_context(wrapper, metadata)
                native_call = lambda context=context, hidden_states=hidden_states, output=output: _custom_op_call(
                    wrapper,
                    context,
                    hidden_states,
                    output,
                )
                native_calls.append(native_call)
                native_call()
                if diagnose_step_states:
                    torch_npu.npu.synchronize(device)
                    native_step_states.append(native_world.snapshot())
                elif sync_between_correctness_steps and step + 1 < correctness_steps:
                    torch_npu.npu.synchronize(device)
            torch_npu.npu.synchronize(device)
            native_output_host = torch.cat(
                [output.detach().cpu() for output in native_outputs],
                dim=0,
            )
            native_state = native_world.snapshot()
            if not torch.isfinite(native_output_host.float()).all() or float(native_output_host.abs().max()) == 0.0:
                raise AssertionError("native synthetic CSA output must be finite and non-zero")
            native_ms = _serial_wall_time_ms(
                native_calls[-1],
                warmups=warmups,
                iterations=iterations,
                device=device,
            )

            bind_state_world(wrapper, pypto_world)
            # PyPTO consumes full RoPE tables plus device start_positions, not
            # the native per-token runtime-buffer views.  These bundles still
            # retain their real builders/buffers for all address snapshots.
            pypto_metadatas = [
                build_real_native_metadata(wrapper, world, config_owner.vllm_config) for world in pypto_worlds
            ]
            owner = install_pypto_dsv4_decode_csa(
                wrapper,
                kv_cache=pypto_world.raw_cache_tuple,
                attn_metadata=pypto_metadatas[0].metadata,
                device=device,
                runtime=runtime,
                batch_buckets=(batch,),
                registry=registry,
                uniform_query_rows_contract=True,
            )
            grad_packed = tuple(name for name, tensor in owner.weights.launch_arguments.items() if tensor.requires_grad)
            if grad_packed:
                raise AssertionError(f"packed inference tensors retained autograd: {grad_packed}")
            pypto_contexts = [_forward_context(wrapper, metadata) for metadata in pypto_metadatas]
            pypto_calls = [
                (
                    lambda context=context, hidden_states=hidden_states, output=output: _custom_op_call(
                        wrapper,
                        context,
                        hidden_states,
                        output,
                    )
                )
                for context, hidden_states, output in zip(
                    pypto_contexts,
                    hidden_states_by_step,
                    pypto_outputs,
                    strict=True,
                )
            ]
            # Warmup is deliberately sacrificial: PyPTO's contract requires
            # external quiescence after the first eager enqueue. Restore all
            # six cache families from independent device clones so the actual
            # consecutive-call check contains ordinary launches only.
            pypto_calls[0]()
            torch_npu.npu.synchronize(device)
            for cache, pristine in zip(
                pypto_world.raw_cache_tuple,
                pypto_initial_device,
                strict=True,
            ):
                cache.copy_(pristine)
            torch_npu.npu.synchronize(device)

            pypto_step_states: list[Mapping[str, torch.Tensor]] = []
            for step, pypto_call in enumerate(pypto_calls):
                pypto_call()
                if diagnose_step_states:
                    torch_npu.npu.synchronize(device)
                    pypto_step_states.append(pypto_world.snapshot())
                elif sync_between_correctness_steps and step + 1 < correctness_steps:
                    torch_npu.npu.synchronize(device)
            torch_npu.npu.synchronize(device)
            pypto_output_host = torch.cat(
                [output.detach().cpu() for output in pypto_outputs],
                dim=0,
            )
            pypto_state = pypto_world.snapshot()
            output_comparison = compare_tensors(
                native_output_host,
                pypto_output_host,
                atol=atol,
                rtol=rtol,
            )
            output_step_comparisons = tuple(
                compare_tensors(
                    native_output.detach().cpu(),
                    pypto_output.detach().cpu(),
                    atol=atol,
                    rtol=rtol,
                )
                for native_output, pypto_output in zip(
                    native_outputs,
                    pypto_outputs,
                    strict=True,
                )
            )
            state_comparisons = compare_state_snapshots(
                native_state,
                pypto_state,
                initial_state,
                atol=atol,
                rtol=rtol,
            )
            row_positions = expected_state_row_positions_for_worlds(native_worlds)
            state_diagnostics = diagnose_state_snapshots(
                native_state,
                pypto_state,
                initial_state,
                row_positions=row_positions,
            )
            indexer_quantized = compare_indexer_quantized(
                native_state["indexer_k"],
                pypto_state["indexer_k"],
                initial_state["indexer_k"],
                native_state["indexer_scale"],
                pypto_state["indexer_scale"],
                initial_state["indexer_scale"],
                row_positions=row_positions["indexer_k"],
            )
            indexer_quantized_long_chain = None
            if diagnose_step_states:
                indexer_quantized_long_chain = compare_indexer_quantized_long_chain(
                    native_step_states,
                    pypto_step_states,
                    initial_state,
                    row_positions_by_step=tuple(
                        expected_state_row_positions_for_worlds(native_worlds[: step + 1])["indexer_k"]
                        for step in range(correctness_steps)
                    ),
                    start_positions=tuple(start_position + step * spec.seq for step in range(correctness_steps)),
                )
            pypto_ms = _serial_wall_time_ms(
                pypto_calls[-1],
                warmups=warmups,
                iterations=iterations,
                device=device,
            )
            result = A3NativeCompareResult(
                runtime=runtime,
                device=device,
                batch=batch,
                seq=spec.seq,
                start_position=start_position,
                correctness_steps=correctness_steps,
                sync_between_correctness_steps=sync_between_correctness_steps,
                diagnose_step_states=diagnose_step_states,
                pypto_sacrificial_warmup_reset=True,
                rope_dtype="fp32-production",
                native_ms=native_ms,
                pypto_ms=pypto_ms,
                speedup=native_ms / pypto_ms,
                output=output_comparison,
                states=state_comparisons,
                indexer_quantized=indexer_quantized,
                native_hot_operators=NATIVE_HOT_OPERATORS,
                indexer_quantized_long_chain=indexer_quantized_long_chain,
            )
            if not result.close:
                payload = result.to_dict()
                payload["output_steps"] = [
                    {
                        "decode_step": step,
                        "start_position": start_position + step * spec.seq,
                        **asdict(comparison),
                    }
                    for step, comparison in enumerate(output_step_comparisons)
                ]
                freqs_cos = owner.weights.launch_arguments["freqs_cos"]
                freqs_sin = owner.weights.launch_arguments["freqs_sin"]
                payload["qkv_rope_rows"] = [
                    {
                        "decode_step": step,
                        "cos": dict(_rope_row_summary(freqs_cos, position)),
                        "sin": dict(_rope_row_summary(freqs_sin, position)),
                    }
                    for step, position in enumerate(
                        start_position + step * spec.seq for step in range(correctness_steps)
                    )
                ]
                if diagnose_step_states:
                    payload["step_state_comparisons"] = [
                        {
                            name: asdict(comparison)
                            for name, comparison in compare_state_snapshots(
                                native_step,
                                pypto_step,
                                initial_state,
                                atol=atol,
                                rtol=rtol,
                            ).items()
                        }
                        for native_step, pypto_step in zip(
                            native_step_states,
                            pypto_step_states,
                            strict=True,
                        )
                    ]
                    payload["native_state_transitions"] = [
                        {name: dict(item) for name, item in transition.items()}
                        for transition in diagnose_state_transitions(
                            initial_state,
                            native_step_states,
                            native_worlds,
                        )
                    ]
                    payload["pypto_state_transitions"] = [
                        {name: dict(item) for name, item in transition.items()}
                        for transition in diagnose_state_transitions(
                            initial_state,
                            pypto_step_states,
                            pypto_worlds,
                        )
                    ]
                payload["state_diagnostics"] = {name: dict(value) for name, value in state_diagnostics.items()}
                raise AssertionError(json.dumps(payload, ensure_ascii=False, indent=2))
            return result
    finally:
        try:
            if tp_initialized:
                torch_npu.npu.synchronize(device)
                if owner is not None:
                    uninstall_pypto_dsv4_decode_csa(owner.layer)
                    owner.device_owner.backend.close()
        finally:
            config_owner.close()
            if tp_initialized:
                _destroy_tp1()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", choices=SUPPORTED_RUNTIMES, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--atol", type=float, default=0.1)
    parser.add_argument("--rtol", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--master-port", type=int, default=29641)
    parser.add_argument("--start-position", type=int, default=0)
    parser.add_argument("--correctness-steps", type=int, default=1)
    parser.add_argument(
        "--sync-between-correctness-steps",
        action="store_true",
        help="diagnostic only: externally quiesce each correctness step",
    )
    parser.add_argument(
        "--diagnose-step-states",
        action="store_true",
        help="diagnostic only: quiesce and snapshot all six caches after every step",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_a3_native_compare(
        runtime=args.runtime,
        device=args.device,
        batch=args.batch,
        warmups=args.warmups,
        iterations=args.iterations,
        atol=args.atol,
        rtol=args.rtol,
        seed=args.seed,
        master_port=args.master_port,
        start_position=args.start_position,
        correctness_steps=args.correctness_steps,
        sync_between_correctness_steps=args.sync_between_correctness_steps,
        diagnose_step_states=args.diagnose_step_states,
    )
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "A3NativeCompareResult",
    "IndexerQuantizedComparison",
    "IndexerQuantizedLongChainReport",
    "NATIVE_HOT_OPERATORS",
    "SUPPORTED_RUNTIMES",
    "TensorComparison",
    "build_parser",
    "compare_state_snapshots",
    "compare_tensors",
    "compare_indexer_quantized",
    "compare_indexer_quantized_long_chain",
    "diagnose_state_transitions",
    "expected_state_row_positions_for_worlds",
    "main",
    "run_a3_native_compare",
]
