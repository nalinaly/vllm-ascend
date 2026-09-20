#!/usr/bin/env python3
"""Compare traces emitted by qwen3_aclgraph_profile.py."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path
from typing import Any

SENTINEL_MODEL_ID = "4294967295"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Directory containing native/ and pypto/ profile outputs",
    )
    return parser.parse_args()


def _only(root: Path, name: str) -> Path:
    matches = list(root.rglob(name))
    if len(matches) != 1:
        raise RuntimeError(f"expected one {name} below {root}, got {matches}")
    return matches[0]


def _load_kernels(path: Path) -> list[dict[str, Any]]:
    kernels: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8-sig") as file:
        for row in csv.DictReader(file):
            kernels.append(
                {
                    "model_id": row["Model ID"],
                    "stream_id": row["Stream ID"],
                    "name": row["Name"],
                    "core": row["Accelerator Core"],
                    "start_us": Decimal(row["Start Time(us)"].strip()),
                    "duration_us": Decimal(row["Duration(us)"].strip()),
                }
            )
    return kernels


def _is_decode_task(variant: str, kernel: dict[str, Any]) -> bool:
    if kernel["model_id"] != SENTINEL_MODEL_ID:
        return True
    if variant != "pypto":
        return False
    name = kernel["name"]
    return name == "aicore_kernel_mode_0" or name.startswith("simpler_aicpu_kernel_exec_")


def _round(value: Decimal | float, digits: int = 3) -> float:
    return round(float(value), digits)


def summarize(root: Path, variant: str) -> dict[str, Any]:
    variant_dir = root / variant
    trace_path = _only(variant_dir / "torch_npu_trace", "trace_view.json")
    kernel_path = _only(variant_dir / "torch_npu_trace", "kernel_details.csv")
    metadata = json.loads((variant_dir / "run_metadata.json").read_text())
    events = json.loads(trace_path.read_text())
    kernels = _load_kernels(kernel_path)

    launches = sorted(
        (
            {
                "ts_us": Decimal(str(event["ts"])),
                "host_duration_us": Decimal(str(event["dur"])),
            }
            for event in events
            if event.get("name") == "AscendCL@aclmdlRIExecuteAsync"
        ),
        key=lambda item: item["ts_us"],
    )
    expected_replays = metadata["decode_aclgraph_replays"]
    if len(launches) != expected_replays:
        raise RuntimeError(f"{variant}: expected {expected_replays} ACL Graph launches, got {len(launches)}")

    decode_tasks = sorted(
        (kernel for kernel in kernels if _is_decode_task(variant, kernel)),
        key=lambda item: item["start_us"],
    )
    per_replay: list[dict[str, Any]] = []
    all_names: Counter[str] = Counter()
    all_cores: Counter[str] = Counter()
    name_durations: dict[str, list[float]] = defaultdict(list)
    for index, launch in enumerate(launches):
        next_ts = launches[index + 1]["ts_us"] if index + 1 < len(launches) else None
        tasks = [
            task
            for task in decode_tasks
            if task["start_us"] >= launch["ts_us"] and (next_ts is None or task["start_us"] < next_ts)
        ]
        if not tasks:
            raise RuntimeError(f"{variant}: replay {index + 1} has no device tasks")
        start = min(task["start_us"] for task in tasks)
        end = max(task["start_us"] + task["duration_us"] for task in tasks)
        names = Counter(task["name"] for task in tasks)
        cores = Counter(task["core"] for task in tasks)
        all_names.update(names)
        all_cores.update(cores)
        for task in tasks:
            name_durations[task["name"]].append(float(task["duration_us"]))
        per_replay.append(
            {
                "index": index + 1,
                "host_launch_duration_us": _round(launch["host_duration_us"]),
                "device_start_after_launch_us": _round(start - launch["ts_us"]),
                "device_span_us": _round(end - start),
                "device_task_count": len(tasks),
                "core_counts": dict(sorted(cores.items())),
            }
        )

    launch_periods = [
        float(launches[index + 1]["ts_us"] - launches[index]["ts_us"]) for index in range(len(launches) - 1)
    ]
    key_kernel_names = (
        "aclnnMatmul_MatMulCommon_MatMulV2",
        "RmsNorm",
        "AddRmsNormBias",
        "aclnnScatterPaKvCache_ScatterPaKvCache_ScatterPaKvCache",
        "FusedInferAttentionScore",
        "SwiGlu",
        "aicore_kernel_mode_0",
    )
    key_kernels: dict[str, Any] = {}
    for name in key_kernel_names:
        durations = name_durations.get(name, [])
        if durations:
            key_kernels[name] = {
                "total_count": all_names[name],
                "count_per_replay": all_names[name] / len(launches),
                "median_duration_us": round(statistics.median(durations), 3),
                "total_duration_us": round(sum(durations), 3),
            }
    aicpu_names = [name for name in all_names if name.startswith("simpler_aicpu_kernel_exec_")]
    aicpu_durations = [duration for name in aicpu_names for duration in name_durations[name]]
    if aicpu_durations:
        key_kernels["simpler_aicpu_kernel_exec_*"] = {
            "total_count": sum(all_names[name] for name in aicpu_names),
            "count_per_replay": sum(all_names[name] for name in aicpu_names) / len(launches),
            "median_duration_us": round(statistics.median(aicpu_durations), 3),
            "total_duration_us": round(sum(aicpu_durations), 3),
        }

    return {
        "variant": variant,
        "trace_json": str(trace_path),
        "trace_json_bytes": trace_path.stat().st_size,
        "trace_event_count": len(events),
        "profiled_token_ids": metadata["profiled_token_ids"],
        "profiled_text": metadata["profiled_text"],
        "profiled_generate_elapsed_ms": round(metadata["profiled_generate_elapsed_s"] * 1000, 3),
        "aclgraph_execute_count": len(launches),
        "host_launch_duration_us": [_round(launch["host_duration_us"]) for launch in launches],
        "decode_launch_period_us": [round(value, 3) for value in launch_periods],
        "decode_launch_period_median_us": round(statistics.median(launch_periods), 3),
        "device_span_median_us": round(statistics.median(replay["device_span_us"] for replay in per_replay), 3),
        "device_task_count_per_replay": [replay["device_task_count"] for replay in per_replay],
        "decode_core_counts_total": dict(sorted(all_cores.items())),
        "per_replay": per_replay,
        "key_decode_kernels": key_kernels,
    }


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    native = summarize(root, "native")
    pypto = summarize(root, "pypto")
    native_period = native["decode_launch_period_median_us"]
    pypto_period = pypto["decode_launch_period_median_us"]
    native_span = native["device_span_median_us"]
    pypto_span = pypto["device_span_median_us"]
    native_elapsed = native["profiled_generate_elapsed_ms"]
    pypto_elapsed = pypto["profiled_generate_elapsed_ms"]
    comparison = {
        "conditions_equal": {
            "token_ids_equal": native["profiled_token_ids"] == pypto["profiled_token_ids"],
            "aclgraph_execute_count_equal": native["aclgraph_execute_count"] == pypto["aclgraph_execute_count"],
            "tp": 1,
            "device": 0,
            "layout": "ND",
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "with_stack": False,
            "record_shapes": False,
        },
        "native": native,
        "pypto": pypto,
        "delta_pypto_vs_native": {
            "profiled_generate_elapsed_ms": round(pypto_elapsed - native_elapsed, 3),
            "profiled_generate_elapsed_percent": round((pypto_elapsed / native_elapsed - 1) * 100, 2),
            "decode_launch_period_us": round(pypto_period - native_period, 3),
            "decode_launch_period_percent": round((pypto_period / native_period - 1) * 100, 2),
            "device_span_us": round(pypto_span - native_span, 3),
            "device_span_percent": round((pypto_span / native_span - 1) * 100, 2),
        },
    }
    output = root / "comparison.json"
    output.write_text(json.dumps(comparison, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(comparison, indent=2, ensure_ascii=False))
    print(f"comparison written to {output}")


if __name__ == "__main__":
    main()
