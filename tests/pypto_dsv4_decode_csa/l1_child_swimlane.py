# SPDX-License-Identifier: Apache-2.0
"""Diagnostic-only CSA L1 ACLGraph swimlanes; no formal latency measurement.

Run with the isolated simpler-l1-swimlane runtime installed. Production PyPTO
and CSA sources are unchanged. Existing fixture construction and numerical
comparisons are reused; each exported window contains exactly one replay.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch


@contextmanager
def recording_workers(root: Path):
    from simpler.task_interface import ChipWorker

    workers = []
    original = ChipWorker.init_l1

    def initialize(self, device_id, bins, config, log_level=None):
        config.enable_chip_swimlane = 2
        config.output_prefix = str(root)
        original(self, device_id, bins, config, log_level)
        workers.append(self)

    with patch.object(ChipWorker, "init_l1", initialize):
        yield workers


def checkpoint(workers, prefix: Path | None = None):
    if len(workers) != 1:
        raise RuntimeError(f"expected one L1 worker, got {len(workers)}")
    if prefix is not None:
        prefix.mkdir(parents=True, exist_ok=False)
    workers[0].l1_swimlane_checkpoint("" if prefix is None else str(prefix))


def smoke(root: Path, device: int):
    import pypto
    import pypto.language as pl
    import torch
    import torch_npu

    @pl.jit(execution="l1", runtime="tensormap_and_ringbuffer")
    def add(
        lhs: pl.Tensor[[64, 128], pl.FP32],
        rhs: pl.Tensor[[64, 128], pl.FP32],
        out: pl.Out[pl.Tensor[[64, 128], pl.FP32]],
    ):
        with pl.at(level=pl.Level.CORE_GROUP):
            a = pl.load(lhs, [0, 0], [64, 128])
            b = pl.load(rhs, [0, 0], [64, 128])
            pl.store(pl.add(a, b), [0, 0], out)
        return out

    torch_npu.npu.set_device(device)
    lhs = torch.ones((64, 128), device=f"npu:{device}")
    rhs = torch.ones_like(lhs)
    out = torch.empty_like(lhs)
    stream = torch_npu.npu.Stream(device=device)
    graph = torch_npu.npu.NPUGraph()
    with recording_workers(root) as workers:
        try:
            add(lhs, rhs, out=out)
            torch_npu.npu.synchronize(device)
            with torch_npu.npu.graph(graph, stream=stream):
                add(lhs, rhs, out=out)
            torch_npu.npu.synchronize(device)
            for index in range(3):
                checkpoint(workers)
                with torch_npu.npu.stream(stream):
                    lhs.fill_(index + 2)
                    graph.replay()
                stream.synchronize()
                checkpoint(workers, root / f"replay{index + 1}")
                torch.testing.assert_close(out.cpu(), torch.full((64, 128), float(index + 3)))
                print(
                    f"smoke replay {index + 1}: output correct, swimlane exported",
                    flush=True,
                )
        finally:
            torch_npu.npu.synchronize(device)
            graph.reset()
            pypto.l1.shutdown(device=device)


@dataclass
class DiagnosticResult:
    metadata: tuple = ()


def csa(root: Path, device: int):
    import torch
    import torch_npu

    from tests.pypto_dsv4_decode_csa import a3_single_op_benchmark as bench

    def profile_schedule(schedule, cases, *, quiesce, between_phases, **_):
        # Warm the exact captured graphs using the benchmark's ABBA schedule.
        for invocation in schedule.warmup_invocations:
            cases[invocation.key].enqueue()
            quiesce()
        between_phases()
        checkpoint(workers)
        native = next(case for key, case in cases.items() if key.backend.value == "native_production")
        pto = next(case for key, case in cases.items() if key.backend.value != "native_production")
        raw = root / "cann_raw"
        profiler = bench._create_torch_npu_profiler(torch_npu, raw, rounds=3)
        with profiler as active:
            index = 0
            for round_id in range(3):
                for label, case in (
                    ("native", native),
                    ("pto", pto),
                    ("pto", pto),
                    ("native", native),
                ):
                    if label == "pto":
                        checkpoint(workers)
                    with torch.profiler.record_function(f"CSA_L1_DIAGNOSTIC/{round_id}/{label}"):
                        case.enqueue()
                        quiesce()
                    if label == "pto":
                        index += 1
                        checkpoint(workers, root / f"replay{index}")
                        print(f"CSA replay {index}: child swimlane exported", flush=True)
                active.step()
        return DiagnosticResult()

    # The original performance runner pins device 0. The diagnostic process
    # selects the same alternate device for both paths without editing it.
    with patch.object(bench, "SUPPORTED_DEVICE", device):
        config = bench.A3SingleOpBenchmarkConfig(
            device=device,
            runtime="tensormap_and_ringbuffer",
            mode=bench.BenchmarkMode.ACLGRAPH,
            start_position=8191,
            seed=20260902,
            warmup_iterations=20,
            sample_iterations=100,
            enqueue_batch_size=20,
            master_port=29863,
            atol=0.001,
            rtol=0.01,
        )
        with (
            recording_workers(root) as workers,
            patch.object(bench, "execute_benchmark_schedule", profile_schedule),
        ):
            result = bench.run_a3_single_op_benchmark(config)
    # This is an execution/configuration note, never a formal benchmark report.
    metadata = dict(result.metadata)
    (root / "diagnostic_result.json").write_text(
        json.dumps(
            {
                "mode": "diagnostic L1 ACLGraph, per-replay external synchronization",
                "formal_latency_samples": 0,
                "device": device,
                "workload": "A3 TP1 B4/S8 C8191, seed 20260902, full dsa_forward CSA",
                "pre_profile_correctness": json.loads(metadata["pre_timing_correctness_gate"]),
                "post_profile_correctness": json.loads(metadata["post_timing_correctness_gate"]),
            },
            indent=2,
        )
        + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "csa"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--runtime-worktree", type=Path)
    args = parser.parse_args()
    if args.runtime_worktree is not None:
        editable = next((args.runtime_worktree / ".venv/lib").glob("python*/site-packages/_simpler_editable.py"))
        spec = importlib.util.spec_from_file_location("_csa_diagnostic_simpler", editable)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    args.output.mkdir(parents=True, exist_ok=False)
    (smoke if args.mode == "smoke" else csa)(args.output.resolve(), args.device)


if __name__ == "__main__":
    main()
