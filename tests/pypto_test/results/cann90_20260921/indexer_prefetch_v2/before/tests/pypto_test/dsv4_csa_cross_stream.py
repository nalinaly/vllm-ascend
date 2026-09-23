"""G08: delay Native metadata producers during the existing full CSA replay.

This driver reuses full_replay's Native/eager/graph comparisons and changes only
producer timing. Delays run on the real DeviceMetadataExecutor stream before
its external readiness events; no host sleep or extra consumer synchronization
is inserted. It covers one fixed batch with A-B-A metadata updates, not request
lifecycle or the complete P3 matrix.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import traceback
from pathlib import Path
from unittest.mock import patch

from dsv4_csa_env import activate, write_json

DELAY_MATRIX_SIZE = 8192
DELAY_LEFT_VALUE = 0.03125
DELAY_RIGHT_VALUE = 0.0625


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--history", type=int, default=131071)
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--stage", choices=("COMPRESSOR", "INDEXER", "ATTENTION"), required=True)
    parser.add_argument("--delay-iterations", type=int, nargs=3, default=(0, 4, 16))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.delay_iterations[0] != 0 or min(args.delay_iterations[1:]) <= 0:
        parser.error("--delay-iterations must start with zero, followed by two positive device work counts")
    activate()
    import torch
    from dsv4_csa_full_replay import main as replay_main

    from vllm_ascend.worker.device_metadata import DeviceMetadataExecutor, DeviceMetadataStage, DeviceMetadataTask

    report = {
        "status": "FAIL",
        "scope": "G08_reference_full_CSA_A_B_A_with_delayed_Native_metadata_producers",
        "batch": args.batch,
        "stage": args.stage,
        "requested_delay_iterations": list(args.delay_iterations),
        "delay_api": "torch.mm on independent BF16 buffers on DeviceMetadataExecutor.stream",
        "delay_matrix_shape": [DELAY_MATRIX_SIZE, DELAY_MATRIX_SIZE],
        "submissions": [],
    }
    write_json(args.output_dir / "cross_stream.json", report)
    original_submit = DeviceMetadataExecutor.submit
    selected_stage = DeviceMetadataStage[args.stage]
    submission_counts = {}
    delay_buffers = {}
    timings = []

    def delayed_submit(executor, tasks, batch_descriptor=None):
        # Native reference execution uses ordinary events. Only the full-graph
        # PTO executor receives timing perturbations; all producer math stays
        # in the original task.run callable.
        if batch_descriptor is None:
            return original_submit(executor, tasks, batch_descriptor)
        ordinal = submission_counts.get(id(executor), 0)
        submission_counts[id(executor)] = ordinal + 1
        variant = 0 if ordinal < 2 else (ordinal - 2) // 2
        iterations = args.delay_iterations[variant % len(args.delay_iterations)]
        if id(executor) not in delay_buffers:
            # Prepare and warm the independent work before the shared harness
            # captures its graph. Native CSA inputs/cache are never used here.
            left = torch.full(
                (DELAY_MATRIX_SIZE, DELAY_MATRIX_SIZE), DELAY_LEFT_VALUE,
                dtype=torch.bfloat16, device=executor.stream.device,
            )
            right = torch.full_like(left, DELAY_RIGHT_VALUE)
            scratch = torch.empty_like(left)
            torch.mm(left, right, out=scratch)
            delay_buffers[id(executor)] = (left, right, scratch)
        left, right, scratch = delay_buffers[id(executor)]
        record = {
            "ordinal": ordinal,
            "delay_iterations": iterations,
            "producer_stream": int(executor.stream.npu_stream),
            "consumer_stream": int(torch.npu.current_stream().npu_stream),
            "frontiers": [],
        }
        if record["producer_stream"] == record["consumer_stream"]:
            raise AssertionError("G08 requires distinct producer and consumer NPU streams")
        wrapped = []
        frontier_events = []
        for task in tasks:
            if task.stage != selected_stage:
                wrapped.append(task)
                continue
            begin = torch.npu.Event(enable_timing=True)
            end = torch.npu.Event(enable_timing=True)
            frontier = {"stage": task.stage.name, "group_id": task.group_id}
            record["frontiers"].append(frontier)
            frontier_events.append((frontier, end))
            timings.append((iterations, begin, end, frontier))

            def run(task=task, begin=begin, end=end, iterations=iterations):
                begin.record()
                for _ in range(iterations):
                    torch.mm(left, right, out=scratch)
                task.run()
                end.record()

            wrapped.append(DeviceMetadataTask(stage=task.stage, group_id=task.group_id, run=run))
        if not record["frontiers"]:
            raise AssertionError(f"No Native producer frontier for {args.stage}")
        result = original_submit(executor, wrapped, batch_descriptor)
        # Nonblocking event queries observe whether producer work is still
        # pending when control returns to the consumer; they never wait or
        # choose the metadata/CSA computation to execute.
        for frontier, end in frontier_events:
            frontier["finished_after_submit"] = end.query()
        record["uses_external_events"] = executor.uses_external_events
        if not record["uses_external_events"]:
            raise AssertionError("G08 lost Native external readiness events")
        report["submissions"].append(record)
        return result

    replay_arguments = [
        "dsv4_csa_full_replay.py",
        "--device", str(args.device),
        "--checkpoint", str(args.checkpoint),
        "--batch", str(args.batch),
        "--history", str(args.history),
        "--seed", str(args.seed),
        "--output-dir", str(args.output_dir),
    ]
    try:
        with patch.object(DeviceMetadataExecutor, "submit", delayed_submit), patch.object(sys, "argv", replay_arguments):
            replay_main()
        # The shared comparison already synchronized at its assertion boundary.
        # Read timing events only after all Native/eager/graph checks finish.
        by_iterations = {}
        for iterations, begin, end, frontier in timings:
            elapsed = begin.elapsed_time(end)
            frontier["producer_duration_ms"] = elapsed
            by_iterations.setdefault(iterations, []).append(elapsed)
        medians = {iterations: statistics.median(values) for iterations, values in by_iterations.items()}
        expected_work = torch.full(
            (DELAY_MATRIX_SIZE, DELAY_MATRIX_SIZE),
            DELAY_MATRIX_SIZE * DELAY_LEFT_VALUE * DELAY_RIGHT_VALUE,
            dtype=torch.bfloat16,
        )
        for _, _, scratch in delay_buffers.values():
            torch.testing.assert_close(scratch.cpu(), expected_work, atol=0, rtol=0)
        longest = max(medians)
        if medians[longest] <= medians[0]:
            raise AssertionError("Device event timings did not show the requested producer delay")
        replay = json.loads((args.output_dir / "full_replay.json").read_text())
        if replay["status"] != "PASS" or len(replay["cases"]) != 3:
            raise AssertionError("The shared full CSA A-B-A comparison did not pass")
        # The reused harness submits warmup, capture, then eager/graph for each
        # of three variants. Validate that all graph replays saw the schedule.
        if len(report["submissions"]) != 8:
            raise AssertionError("Unexpected shared replay submission sequence; inspect coverage")
        graph_submissions = [report["submissions"][ordinal] for ordinal in (3, 5, 7)]
        graph_delays = {item["delay_iterations"] for item in graph_submissions}
        if 0 not in graph_delays or not any(iterations > 0 for iterations in graph_delays):
            raise AssertionError("Graph replays must include both ordinary and delayed producers")
        largest_submissions = [
            item for item in report["submissions"] if item["delay_iterations"] == max(args.delay_iterations)
        ]
        if not largest_submissions or any(
            all(frontier["finished_after_submit"] for frontier in item["frontiers"])
            for item in largest_submissions
        ):
            raise AssertionError("Largest delay finished before consumer dispatch; increase device work")
        report.update(
            status="PASS",
            producer_duration_median_ms=medians,
            graph_submission_ordinals=[3, 5, 7],
            graph_delay_iterations=[item["delay_iterations"] for item in graph_submissions],
            full_replay_checks=sum(len(case["checks"]) for case in replay["cases"]),
            full_replay_report="full_replay.json",
            fixed_addresses=replay["fixed_addresses"],
            native_external_metadata_events=replay["native_external_metadata_events"],
            delay_work_exact=True,
            largest_delay_pending_after_submit=True,
        )
    except Exception:
        report["error"] = traceback.format_exc()
        raise
    finally:
        write_json(args.output_dir / "cross_stream.json", report)


if __name__ == "__main__":
    main()
