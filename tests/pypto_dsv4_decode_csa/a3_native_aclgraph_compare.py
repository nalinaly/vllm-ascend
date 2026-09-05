# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Fresh-process nonzero ACLGraph comparison of native and PyPTO DSV4 CSA.

The capture region on each side contains exactly one ``vllm::dsa_forward``
custom-op node.  Native and PyPTO use the same real TP1 ratio-4 layer and
weights, but own disjoint input/output buffers, graph streams and six-cache
state worlds.  Every correctness replay restores both worlds to the same
initial device snapshot before changing the fixed-address graph inputs.

Run this module once per runtime.  A process-pinned PyPTO owner and captured
graph-visible callables deliberately make cross-runtime reuse unsupported.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
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
from tests.pypto_dsv4_decode_csa.indexer_quantized_comparison import (
    IndexerQuantizedComparison,
    compare_indexer_quantized,
)
from tests.pypto_dsv4_decode_csa.native_fixture import (
    DEFAULT_LAYER_PREFIX,
    bind_state_world,
    build_real_native_metadata,
    build_real_ratio4_attention,
    build_synthetic_vllm_config,
    build_twin_decode_csa_state_worlds,
)
from vllm_ascend.ops._pypto_dsv4_csa import DecodeCSAProgramSpec
from vllm_ascend.ops._pypto_dsv4_csa.config import FLASH


@dataclass(frozen=True, slots=True)
class ACLGraphReplayComparison:
    """One fixed-address native/PyPTO replay and its mutable-state result."""

    input_scale: float
    output: TensorComparison
    states: Mapping[str, TensorComparison]
    indexer_quantized: IndexerQuantizedComparison

    @property
    def close(self) -> bool:
        non_quantized_states_close = all(
            value.close for name, value in self.states.items() if name not in {"indexer_k", "indexer_scale"}
        )
        return self.output.close and non_quantized_states_close and self.indexer_quantized.acceptable

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_scale": self.input_scale,
            "output": asdict(self.output),
            "states": {name: asdict(value) for name, value in self.states.items()},
            "indexer_quantized": self.indexer_quantized.to_dict(),
            "close": self.close,
        }


@dataclass(frozen=True, slots=True)
class A3NativeACLGraphCompareResult:
    """Inspectable result for two independent single-node ACLGraphs."""

    runtime: str
    device: int
    batch: int
    seq: int
    start_position: int
    replay_count: int
    native_graph_ms: float
    pypto_graph_ms: float
    speedup: float
    native_and_pypto_addresses_disjoint: bool
    native_capture_addresses_differ_from_warmup: bool
    pypto_capture_addresses_differ_from_warmup: bool
    pypto_capture_bindings_retained: int
    replays: tuple[ACLGraphReplayComparison, ...]

    @property
    def close(self) -> bool:
        return bool(self.replays) and all(replay.close for replay in self.replays)

    def to_dict(self) -> dict[str, Any]:
        return {
            "runtime": self.runtime,
            "device": self.device,
            "batch": self.batch,
            "seq": self.seq,
            "start_position": self.start_position,
            "replay_count": self.replay_count,
            "native_graph_ms": self.native_graph_ms,
            "pypto_graph_ms": self.pypto_graph_ms,
            "speedup": self.speedup,
            "native_and_pypto_addresses_disjoint": self.native_and_pypto_addresses_disjoint,
            "native_capture_addresses_differ_from_warmup": self.native_capture_addresses_differ_from_warmup,
            "pypto_capture_addresses_differ_from_warmup": self.pypto_capture_addresses_differ_from_warmup,
            "pypto_capture_bindings_retained": self.pypto_capture_bindings_retained,
            "replays": [replay.to_dict() for replay in self.replays],
            "close": self.close,
        }


def _validate_run_arguments(
    *,
    runtime: str,
    batch: int,
    warmups: int,
    iterations: int,
    replay_values: Sequence[float],
) -> tuple[float, ...]:
    if runtime not in SUPPORTED_RUNTIMES:
        raise ValueError(f"runtime must be one of {SUPPORTED_RUNTIMES}, got {runtime!r}")
    if batch != 4:
        raise ValueError("the first nonzero ACLGraph fixture is intentionally fixed to the B4 TP1 bucket")
    if warmups < 0 or iterations <= 0:
        raise ValueError("warmups must be non-negative and iterations must be positive")
    values = tuple(float(value) for value in replay_values)
    if not values:
        raise ValueError("at least one replay value is required")
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"replay values must be finite, got {values}")
    return values


def _restore_state_world(world: Any, pristine: Sequence[torch.Tensor]) -> None:
    """Restore six logical cache views without reallocating graph-visible storage."""
    if len(pristine) != len(world.raw_cache_tuple):
        raise ValueError("pristine state must contain all six cache families")
    for cache, initial in zip(world.raw_cache_tuple, pristine, strict=True):
        cache.copy_(initial)


def _graph_wall_time_ms(
    graph: Any,
    stream: Any,
    *,
    warmups: int,
    iterations: int,
) -> float:
    for _ in range(warmups):
        with torch.npu.stream(stream):
            graph.replay()
    stream.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        with torch.npu.stream(stream):
            graph.replay()
    stream.synchronize()
    return (time.perf_counter() - start) * 1000.0 / iterations


def run_a3_native_aclgraph_compare(
    *,
    runtime: str,
    device: int = 0,
    batch: int = 4,
    start_position: int = 0,
    replay_values: Sequence[float] = (0.5, -1.0, 1.75),
    warmups: int = 3,
    iterations: int = 10,
    atol: float = 0.1,
    rtol: float = 0.1,
    seed: int = 20260902,
    master_port: int = 29661,
) -> A3NativeACLGraphCompareResult:
    """Capture and compare one real native node and one PyPTO node per graph."""
    import torch_npu
    from vllm.config import set_current_vllm_config

    import vllm_ascend.ops.dsa  # noqa: F401 -- register torch.ops.vllm.dsa_forward
    from vllm_ascend.ops._pypto_dsv4_csa import (
        DecodeCSADeviceOwnerRegistry,
        install_pypto_dsv4_decode_csa,
        uninstall_pypto_dsv4_decode_csa,
    )
    from vllm_ascend.utils import register_ascend_customop

    values = _validate_run_arguments(
        runtime=runtime,
        batch=batch,
        warmups=warmups,
        iterations=iterations,
        replay_values=replay_values,
    )
    config_owner = build_synthetic_vllm_config(batch=batch)
    registry = DecodeCSADeviceOwnerRegistry()
    owner = None
    native_graph = None
    pypto_graph = None
    tp_initialized = False
    target = torch.device(f"npu:{device}")
    try:
        with set_current_vllm_config(config_owner.vllm_config):
            _initialize_tp1(device=device, master_port=master_port)
            tp_initialized = True
            register_ascend_customop(config_owner.vllm_config)
            attention = build_real_ratio4_attention(
                config_owner,
                device=target,
                prefix=DEFAULT_LAYER_PREFIX,
                seed=seed,
            )
            wrapper = attention.dsa_attn
            spec = DecodeCSAProgramSpec(batch=batch)
            native_world, pypto_world = build_twin_decode_csa_state_worlds(
                spec,
                device=target,
                start_positions=(start_position,) * batch,
            )
            initial_state = native_world.snapshot()
            native_pristine = tuple(tensor.detach().clone() for tensor in native_world.raw_cache_tuple)
            pypto_pristine = tuple(tensor.detach().clone() for tensor in pypto_world.raw_cache_tuple)

            host_generator = torch.Generator(device="cpu")
            host_generator.manual_seed(seed + 1)
            base_hidden = (
                torch.randn(
                    (spec.tokens, FLASH.hidden_size),
                    dtype=torch.bfloat16,
                    generator=host_generator,
                )
                .mul_(0.125)
                .to(target)
            )
            replay_inputs = tuple((base_hidden * value).contiguous() for value in values)

            # Native warmup and capture use the real metadata builder.  Build
            # immediately before use because its per-token RoPE fields are
            # views into RopeGlobalState.runtime_buffer.
            bind_state_world(wrapper, native_world)
            native_metadata = build_real_native_metadata(wrapper, native_world, config_owner.vllm_config)
            native_context = _forward_context(wrapper, native_metadata)
            native_eager_hidden = base_hidden.clone()
            native_eager_output = torch.empty_like(native_eager_hidden)
            _custom_op_call(wrapper, native_context, native_eager_hidden, native_eager_output)
            torch_npu.npu.synchronize(device)
            _restore_state_world(native_world, native_pristine)
            torch_npu.npu.synchronize(device)

            native_hidden = torch.empty_like(base_hidden)
            native_output = torch.empty_like(base_hidden)
            native_address_patch = (
                native_hidden.data_ptr() != native_eager_hidden.data_ptr()
                and native_output.data_ptr() != native_eager_output.data_ptr()
            )
            if not native_address_patch:
                raise AssertionError("native capture input/output addresses must differ from eager warmup")
            native_hidden.copy_(base_hidden)
            torch_npu.npu.synchronize(device)
            native_stream = torch_npu.npu.Stream(device=device)
            native_graph = torch_npu.npu.NPUGraph()
            native_context.capturing = True
            with torch_npu.npu.graph(native_graph, stream=native_stream):
                _custom_op_call(wrapper, native_context, native_hidden, native_output)
            native_stream.synchronize()
            native_context.capturing = False
            _restore_state_world(native_world, native_pristine)
            torch_npu.npu.synchronize(device)

            # Rebind the same real layer to an independent cache universe,
            # then warm and capture the production PyPTO custom-op dispatch.
            bind_state_world(wrapper, pypto_world)
            pypto_metadata = build_real_native_metadata(wrapper, pypto_world, config_owner.vllm_config)
            owner = install_pypto_dsv4_decode_csa(
                wrapper,
                kv_cache=pypto_world.raw_cache_tuple,
                attn_metadata=pypto_metadata.metadata,
                device=device,
                runtime=runtime,
                batch_buckets=(batch,),
                registry=registry,
                uniform_query_rows_contract=True,
            )
            grad_packed = tuple(name for name, tensor in owner.weights.launch_arguments.items() if tensor.requires_grad)
            if grad_packed:
                raise AssertionError(f"packed inference tensors retained autograd: {grad_packed}")
            pypto_context = _forward_context(wrapper, pypto_metadata)
            pypto_eager_hidden = base_hidden.clone()
            pypto_eager_output = torch.empty_like(pypto_eager_hidden)
            _custom_op_call(wrapper, pypto_context, pypto_eager_hidden, pypto_eager_output)
            torch_npu.npu.synchronize(device)
            owner.device_owner.mark_warmups_quiesced()
            _restore_state_world(pypto_world, pypto_pristine)
            torch_npu.npu.synchronize(device)

            pypto_hidden = torch.empty_like(base_hidden)
            pypto_output = torch.empty_like(base_hidden)
            pypto_address_patch = (
                pypto_hidden.data_ptr() != pypto_eager_hidden.data_ptr()
                and pypto_output.data_ptr() != pypto_eager_output.data_ptr()
            )
            if not pypto_address_patch:
                raise AssertionError("PyPTO capture input/output addresses must differ from eager warmup")
            pypto_hidden.copy_(base_hidden)
            torch_npu.npu.synchronize(device)
            pypto_stream = torch_npu.npu.Stream(device=device)
            pypto_graph = torch_npu.npu.NPUGraph()
            pypto_context.capturing = True
            with torch_npu.npu.graph(pypto_graph, stream=pypto_stream):
                _custom_op_call(wrapper, pypto_context, pypto_hidden, pypto_output)
            pypto_stream.synchronize()
            pypto_context.capturing = False
            retained = len(owner.device_owner.backend._bindings)
            if retained != 1:
                raise AssertionError(f"single PyPTO graph must retain exactly one address snapshot, got {retained}")
            _restore_state_world(pypto_world, pypto_pristine)
            torch_npu.npu.synchronize(device)

            disjoint = all(
                left.data_ptr() != right.data_ptr()
                for left, right in zip(
                    (native_hidden, native_output, *native_world.raw_cache_tuple),
                    (pypto_hidden, pypto_output, *pypto_world.raw_cache_tuple),
                    strict=True,
                )
            )
            if not disjoint:
                raise AssertionError("native and PyPTO graph-visible input/output/cache addresses must be disjoint")

            replay_results = []
            for value, replay_input in zip(values, replay_inputs, strict=True):
                _restore_state_world(native_world, native_pristine)
                _restore_state_world(pypto_world, pypto_pristine)
                native_hidden.copy_(replay_input)
                pypto_hidden.copy_(replay_input)
                torch_npu.npu.synchronize(device)

                with torch_npu.npu.stream(native_stream):
                    native_graph.replay()
                native_stream.synchronize()
                with torch_npu.npu.stream(pypto_stream):
                    pypto_graph.replay()
                pypto_stream.synchronize()

                native_output_host = native_output.detach().cpu()
                pypto_output_host = pypto_output.detach().cpu()
                if (
                    not torch.isfinite(native_output_host.float()).all()
                    or not torch.isfinite(pypto_output_host.float()).all()
                    or float(native_output_host.abs().max()) == 0.0
                    or float(pypto_output_host.abs().max()) == 0.0
                ):
                    raise AssertionError(f"ACLGraph replay input_scale={value} produced zero or non-finite output")
                native_state = native_world.snapshot()
                pypto_state = pypto_world.snapshot()
                replay_results.append(
                    ACLGraphReplayComparison(
                        input_scale=value,
                        output=compare_tensors(
                            native_output_host,
                            pypto_output_host,
                            atol=atol,
                            rtol=rtol,
                        ),
                        states=compare_state_snapshots(
                            native_state,
                            pypto_state,
                            initial_state,
                            atol=atol,
                            rtol=rtol,
                        ),
                        indexer_quantized=compare_indexer_quantized(
                            native_state["indexer_k"],
                            pypto_state["indexer_k"],
                            initial_state["indexer_k"],
                            native_state["indexer_scale"],
                            pypto_state["indexer_scale"],
                            initial_state["indexer_scale"],
                        ),
                    )
                )

            # Timing excludes cache restoration and input patching.  It is a
            # graph replay latency comparison at one fixed input/address set.
            _restore_state_world(native_world, native_pristine)
            _restore_state_world(pypto_world, pypto_pristine)
            native_hidden.copy_(replay_inputs[-1])
            pypto_hidden.copy_(replay_inputs[-1])
            torch_npu.npu.synchronize(device)
            native_ms = _graph_wall_time_ms(
                native_graph,
                native_stream,
                warmups=warmups,
                iterations=iterations,
            )
            pypto_ms = _graph_wall_time_ms(
                pypto_graph,
                pypto_stream,
                warmups=warmups,
                iterations=iterations,
            )
            result = A3NativeACLGraphCompareResult(
                runtime=runtime,
                device=device,
                batch=batch,
                seq=spec.seq,
                start_position=start_position,
                replay_count=len(values),
                native_graph_ms=native_ms,
                pypto_graph_ms=pypto_ms,
                speedup=native_ms / pypto_ms,
                native_and_pypto_addresses_disjoint=disjoint,
                native_capture_addresses_differ_from_warmup=native_address_patch,
                pypto_capture_addresses_differ_from_warmup=pypto_address_patch,
                pypto_capture_bindings_retained=retained,
                replays=tuple(replay_results),
            )
            if not result.close:
                raise AssertionError(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
            return result
    finally:
        try:
            if tp_initialized:
                torch_npu.npu.synchronize(device)
                for graph in (pypto_graph, native_graph):
                    if graph is not None:
                        graph.reset()
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
    parser.add_argument("--start-position", type=int, default=0)
    parser.add_argument("--replay-values", type=float, nargs="+", default=(0.5, -1.0, 1.75))
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--atol", type=float, default=0.1)
    parser.add_argument("--rtol", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--master-port", type=int, default=29661)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_a3_native_aclgraph_compare(
        runtime=args.runtime,
        device=args.device,
        batch=args.batch,
        start_position=args.start_position,
        replay_values=args.replay_values,
        warmups=args.warmups,
        iterations=args.iterations,
        atol=args.atol,
        rtol=args.rtol,
        seed=args.seed,
        master_port=args.master_port,
    )
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ACLGraphReplayComparison",
    "A3NativeACLGraphCompareResult",
    "build_parser",
    "main",
    "run_a3_native_aclgraph_compare",
]
