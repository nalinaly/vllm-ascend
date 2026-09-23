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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--history", type=int, default=131071)
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--stage", choices=("COMPRESSOR", "INDEXER", "ATTENTION"), required=True)
    parser.add_argument("--delay-cycles", type=int, nargs=3, default=(0, 1_000_000, 10_000_000))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.delay_cycles[0] != 0 or min(args.delay_cycles[1:]) <= 0:
        parser.error("--delay-cycles must start with zero, followed by two positive device delays")
    activate()
    import torch
    import torch_npu
    from dsv4_csa_full_replay import main as replay_main

    from vllm_ascend.worker.device_metadata import DeviceMetadataExecutor, DeviceMetadataStage, DeviceMetadataTask

    report = {
        "status": "FAIL",
        "scope": "G08_reference_full_CSA_A_B_A_with_delayed_Native_metadata_producers",
        "batch": args.batch,
        "stage": args.stage,
        "requested_delay_cycles": list(args.delay_cycles),
        "delay_api": "torch_npu.npu._sleep on DeviceMetadataExecutor.stream",
        "submissions": [],
    }
    write_json(args.output_dir / "cross_stream.json", report)
    original_submit = DeviceMetadataExecutor.submit
    selected_stage = DeviceMetadataStage[args.stage]
    submission_counts = {}
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
        cycles = args.delay_cycles[variant % len(args.delay_cycles)]
        record = {
            "ordinal": ordinal,
            "delay_cycles": cycles,
            "producer_stream": int(executor.stream.npu_stream),
            "consumer_stream": int(torch.npu.current_stream().npu_stream),
            "frontiers": [],
        }
        if record["producer_stream"] == record["consumer_stream"]:
            raise AssertionError("G08 requires distinct producer and consumer NPU streams")
        wrapped = []
        for task in tasks:
            if task.stage != selected_stage:
                wrapped.append(task)
                continue
            begin = torch.npu.Event(enable_timing=True)
            end = torch.npu.Event(enable_timing=True)
            frontier = {"stage": task.stage.name, "group_id": task.group_id}
            record["frontiers"].append(frontier)
            timings.append((cycles, begin, end, frontier))

            def run(task=task, begin=begin, end=end, cycles=cycles):
                begin.record()
                if cycles:
                    torch_npu.npu._sleep(cycles)
                task.run()
                end.record()

            wrapped.append(DeviceMetadataTask(stage=task.stage, group_id=task.group_id, run=run))
        if not record["frontiers"]:
            raise AssertionError(f"No Native producer frontier for {args.stage}")
        result = original_submit(executor, wrapped, batch_descriptor)
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
        by_cycles = {}
        for cycles, begin, end, frontier in timings:
            elapsed = begin.elapsed_time(end)
            frontier["producer_duration_ms"] = elapsed
            by_cycles.setdefault(cycles, []).append(elapsed)
        medians = {cycles: statistics.median(values) for cycles, values in by_cycles.items()}
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
        graph_delays = {item["delay_cycles"] for item in graph_submissions}
        if 0 not in graph_delays or not any(cycles > 0 for cycles in graph_delays):
            raise AssertionError("Graph replays must include both ordinary and delayed producers")
        report.update(
            status="PASS",
            producer_duration_median_ms=medians,
            graph_submission_ordinals=[3, 5, 7],
            graph_delay_cycles=[item["delay_cycles"] for item in graph_submissions],
            full_replay_checks=sum(len(case["checks"]) for case in replay["cases"]),
            full_replay_report="full_replay.json",
            fixed_addresses=replay["fixed_addresses"],
            native_external_metadata_events=replay["native_external_metadata_events"],
        )
    except Exception:
        report["error"] = traceback.format_exc()
        raise
    finally:
        write_json(args.output_dir / "cross_stream.json", report)


if __name__ == "__main__":
    main()
