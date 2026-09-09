# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Audit CSA ABBA samples and export device-only traces without changing timestamps.

This is an offline diagnostic utility: it never imports torch or touches an NPU.
PTO parent-kernel records are explicitly NOT a child-task/core swimlane.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import subprocess
from collections import Counter
from decimal import Decimal
from pathlib import Path


def percentile(values, q):
    ordered = sorted(values)
    if not ordered or not 0 <= q <= 1:
        raise ValueError("percentile requires samples and q in [0, 1]")
    index = (len(ordered) - 1) * q
    lower = math.floor(index)
    upper = math.ceil(index)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def audit_samples(paths):
    summaries, samples = [], []
    for path in paths:
        envelope = json.loads(path.read_text())
        assert envelope["status"] == "ok"
        report = envelope["benchmark"]
        assert report["sampling"] == {"order": "abba", "sample_iterations": 100, "warmup_iterations": 20}
        assert report["metadata"]["task_queue_enable"] == "1"
        assert report["metadata"]["native_multistream_dsv4_dsa_overlap"] == "true"
        assert len(report["cases"]) == 2
        assert {case["key"]["backend"] for case in report["cases"]} == {"native_production", "pypto_trb"}
        assert report["cases"][0]["key"]["workload"] == report["cases"][1]["key"]["workload"]
        for gate in ("pre_timing_correctness_gate", "post_timing_correctness_gate"):
            evidence = json.loads(report["metadata"][gate])
            assert evidence["close"] and evidence["output"]["finite"]
            assert len(evidence["states"]) == 6 and all(s["finite"] for s in evidence["states"].values())
        for case in report["cases"]:
            backend = case["key"]["backend"]
            assert case["key"]["mode"] == "aclgraph"
            measured = [m for m in case["measurements"] if m["metric"] == "device_span"]
            assert len(measured) == 1
            measured = measured[0]
            values = measured["samples_ns"]
            assert len(values) == 100 and len(measured["warmup_samples_ns"]) == 20
            assert all(isinstance(v, int) and v > 0 for v in values)
            for key, q in (("p50_ns", 0.5), ("p90_ns", 0.9), ("p99_ns", 0.99)):
                assert math.isclose(percentile(values, q), measured["statistics"][key], abs_tol=1e-6)
            summary = {"run": path.parent.name, "backend": backend, "count": len(values)}
            summary.update({f"p{int(q * 100)}_us": percentile(values, q) / 1000 for q in (0.5, 0.9, 0.99)})
            summary.update(mean_us=statistics.mean(values) / 1000, max_us=max(values) / 1000)
            # Per-backend chronological ordinals alternate within each ABBA round.
            summary["first_in_abba_p50_us"] = statistics.median(values[::2]) / 1000
            summary["second_in_abba_p50_us"] = statistics.median(values[1::2]) / 1000
            summaries.append(summary)
            for index, duration in enumerate(values):
                samples.append(
                    {
                        "run": path.parent.name,
                        "backend": backend,
                        "sample_ordinal": index + 1,
                        "abba_slot": ("A1", "A2")[index % 2]
                        if backend == "native_production"
                        else ("B1", "B2")[index % 2],
                        "duration_us": duration / 1000,
                    }
                )
    assert len({(s["run"], s["backend"], s["sample_ordinal"]) for s in samples}) == len(samples)
    return summaries, samples


def event_start(event):
    return Decimal(str(event["ts"]))


def event_end(event):
    return event_start(event) + Decimal(str(event.get("dur", 0)))


def union_duration(intervals):
    total = Decimal(0)
    cursor = None
    for start, end in sorted(intervals):
        assert end >= start
        if cursor is None or start >= cursor:
            total += end - start
        elif end > cursor:
            total += end - cursor
        cursor = end if cursor is None else max(cursor, end)
    return total


def split_device_traces(events, expected_replays=6):
    hardware = {
        e["pid"]
        for e in events
        if e.get("name") == "process_name" and e.get("args", {}).get("name") == "Ascend Hardware"
    }
    assert len(hardware) == 1
    pid = next(iter(hardware))
    device = [e for e in events if e.get("pid") == pid and e.get("ph") == "X"]
    pto_models = {e["args"]["Model Id"] for e in device if e["name"] == "aicore_kernel_0"}
    assert len(pto_models) == 1
    pto_model = next(iter(pto_models))
    models = {e["args"]["Model Id"] for e in device if e["name"] == "MODEL_EXECUTE"}
    assert pto_model in models and len(models) == 2
    native_model = next(iter(models - pto_models))
    result = {}
    for backend, model in (("native", native_model), ("pypto", pto_model)):
        selected = [e for e in device if e.get("args", {}).get("Model Id") == model]
        begins = sorted((e for e in selected if e["name"] == "MODEL_EXECUTE"), key=event_start)
        ends = sorted((e for e in selected if e["name"] == "MODEL_WAIT_COMPLETE"), key=event_start)
        assert len(begins) == len(ends) == expected_replays
        assert sum(e.get("name") == f"csa_steady_state.{backend}.enqueue" for e in events) == expected_replays
        windows = [(event_start(a), event_end(b)) for a, b in zip(begins, ends, strict=True)]
        assert all(a < b for a, b in windows)
        assert all(windows[i][1] <= windows[i + 1][0] for i in range(len(windows) - 1))
        assert all(sum(a <= event_start(e) and event_end(e) <= b for a, b in windows) == 1 for e in selected)
        unique = {(e["tid"], e["ts"], e["name"], str(e["dur"])) for e in selected}
        assert len(unique) == len(selected)
        metadata = [e for e in events if e.get("ph") == "M" and e.get("pid") == pid]
        result[backend] = {"model_id": model, "events": selected, "trace": metadata + selected, "windows": windows}
    return result


def save_csv(path, rows):
    assert rows
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def draw_preview(output, backend, records, kernel_rows):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    # Fixed selection, not a fastest/nearest-median replay cherry-pick.
    replay = 2
    origin, finish = records["windows"][replay]
    selected = [row for row in kernel_rows if origin <= Decimal(row["Start Time(us)"].strip()) < finish]
    streams = sorted({int(row["Stream ID"]) for row in selected})
    colors = {"AI_CPU": "#6B7280", "AI_VECTOR_CORE": "#BC8A28", "AI_CORE": "#2563A6", "MIX_AIC": "#2563A6"}
    fig, ax = plt.subplots(figsize=(14, 3.8))
    for index, row in enumerate(selected, 1):
        start = float(Decimal(row["Start Time(us)"].strip()) - origin)
        duration = float(row["Duration(us)"])
        lane = streams.index(int(row["Stream ID"]))
        color = colors.get(row["Accelerator Core"], "#2563A6")
        ax.broken_barh([(start, duration)], (lane - 0.28, 0.56), facecolors=color)
        if duration >= 10:
            label = str(index) if backend == "native" else row["Name"].replace("simpler_aicpu_", "")
            ax.text(start + duration / 2, lane, label, ha="center", va="center", fontsize=8, color="white")
    ax.set_yticks(range(len(streams)), [f"Device stream {stream}" for stream in streams])
    ax.set_ylim(len(streams) - 0.5, -0.5)
    ax.set_xlim(0, max(850, math.ceil(float(finish - origin) / 50) * 50))
    ax.set_xlabel("Device time from MODEL_EXECUTE (us); gaps and overlaps preserved")
    title = (
        "Native: per-kernel device timeline" if backend == "native" else "PTO L1: parent-kernel lanes (NOT child tasks)"
    )
    ax.set_title(
        f"{title}\nA3 / TP1 / B4-S8 / C8191 / TRB ACLGraph | replay 3 of 6 | diagnostic profiler ON",
        loc="left",
        fontsize=12,
    )
    ax.grid(axis="x", alpha=0.15)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(
        handles=[
            Patch(color="#2563A6", label="AICore / mixed"),
            Patch(color="#BC8A28", label="Vector"),
            Patch(color="#6B7280", label="AICPU"),
        ],
        loc="lower right",
        fontsize=8,
    )
    fig.text(
        0.01,
        0.01,
        "Synthetic random weights/input; mostly zero historical caches. Not serving latency. "
        + (
            "Bar numbers: replay-local kernel order in native_kernel_details.csv."
            if backend == "native"
            else "CANN cannot resolve PyPTO child-task/core scheduling; no L2 trace substituted."
        ),
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(output / f"{backend}_device_preview.svg")
    fig.savefig(output / f"{backend}_device_preview.png", dpi=160)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.root / "artifacts"
    output.mkdir(exist_ok=True)
    paths = [args.root / f"run{i}" / "result.json" for i in (1, 2, 3)]
    summaries, samples = audit_samples(paths)
    save_csv(output / "formal_latency_summary.csv", summaries)
    save_csv(output / "formal_device_samples.csv", samples)
    trace_path = next(args.profile.rglob("trace_view.json"))
    kernel_path = next(args.profile.rglob("kernel_details.csv"))
    split = split_device_traces(json.loads(trace_path.read_text()))
    with kernel_path.open() as handle:
        kernels = list(csv.DictReader(handle))
    # CANN CSV loses Model ID on custom/Triton kernels (UINT32_MAX), while
    # the device timeline retains it. Resolve by exact name + device timestamp,
    # never by CPU enqueue windows or by assuming CSV row adjacency.
    all_device_events = [e for records in split.values() for e in records["events"]]
    for row in kernels:
        matches = [
            e
            for e in all_device_events
            if e["name"] == row["Name"] and event_start(e) == Decimal(row["Start Time(us)"].strip())
        ]
        assert len(matches) == 1
        model = matches[0]["args"]["Model Id"]
        assert int(row["Model ID"]) in (model, 4294967295)
        row["Resolved Device Model ID"] = model
    coverage, replay_rows = {}, []
    for backend, records in split.items():
        rows = [r for r in kernels if r["Resolved Device Model ID"] == records["model_id"]]
        rows.sort(key=lambda r: Decimal(r["Start Time(us)"].strip()))
        expected_per_replay = 38 if backend == "native" else 2
        assert len(rows) == expected_per_replay * len(records["windows"])
        for row in rows:
            matches = [
                e
                for e in records["events"]
                if e["name"] == row["Name"] and event_start(e) == Decimal(row["Start Time(us)"].strip())
            ]
            assert len(matches) == 1
            assert Decimal(str(matches[0]["dur"])) == Decimal(row["Duration(us)"])
        for number, (start, end) in enumerate(records["windows"], 1):
            local = [r for r in rows if start <= Decimal(r["Start Time(us)"].strip()) < end]
            assert len(local) == expected_per_replay
            for index, row in enumerate(local, 1):
                row["Replay"] = number
                row["Replay Kernel"] = index
                row["Relative Start(us)"] = str(Decimal(row["Start Time(us)"].strip()) - start)
            intervals = [
                (
                    Decimal(r["Start Time(us)"].strip()),
                    Decimal(r["Start Time(us)"].strip()) + Decimal(r["Duration(us)"]),
                )
                for r in local
            ]
            span = max(b for _, b in intervals) - min(a for a, _ in intervals)
            union = union_duration(intervals)
            total = sum((b - a for a, b in intervals), Decimal(0))
            replay_rows.append(
                {
                    "backend": backend,
                    "replay": number,
                    "kernel_count": len(local),
                    "model_span_us": float(end - start),
                    "kernel_span_us": float(span),
                    "kernel_sum_us": float(total),
                    "kernel_union_us": float(union),
                    "kernel_overlap_us": float(total - union),
                    "kernel_global_idle_us": float(span - union),
                }
            )
        write_json(output / f"{backend}_device_trace.json", records["trace"])
        save_csv(output / f"{backend}_kernel_details.csv", rows)
        coverage[backend] = {
            "model_id": records["model_id"],
            "replays": len(records["windows"]),
            "device_events": len(records["events"]),
            "kernel_rows": len(rows),
            "csv_missing_model_ids_resolved_from_device": sum(int(r["Model ID"]) == 4294967295 for r in rows),
            "kernel_types": dict(Counter(r["Type"] for r in rows)),
            "detail": "native per-kernel" if backend == "native" else "L1 parent kernels only; no child-task swimlane",
        }
        draw_preview(output, backend, records, rows)
    save_csv(output / "diagnostic_replay_summary.csv", replay_rows)
    identity_path = args.root / "identity" / "fixture_identity.json"
    identity = json.loads(identity_path.read_text())
    assert identity["status"] == "ok" and identity["all_logical_bytes_equal"]
    profile_log = args.profile.parent / "run.log"
    log_text = profile_log.read_text()
    profile_report = json.loads(log_text[log_text.index('\n{\n  "cases":') + 1 :])
    assert profile_report["metadata"]["diagnostic_profile_formal_samples"] == "false"
    assert json.loads(profile_report["metadata"]["diagnostic_profile_correctness_gate"])["close"]
    write_json(output / "profile_benchmark_report.json", profile_report)
    inputs = paths + [trace_path, kernel_path, Path(__file__), identity_path, profile_log]
    write_json(
        output / "audit_manifest.json",
        {
            "coverage": coverage,
            "formal_samples_per_backend": 300,
            "outliers_removed": 0,
            "original_trace_timestamps_preserved": True,
            "profile_not_used_for_formal_percentiles": True,
            "hashes": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
        },
    )
    repository = Path(__file__).resolve().parents[2]
    pypto = repository.parent / "pto" / "pypto"
    runtime = pypto / "runtime"
    runtime_binaries = runtime / "build/lib/a2a3/onboard/tensormap_and_ringbuffer"
    source_paths = sorted((repository / "vllm_ascend/ops/_pypto_dsv4_csa").rglob("*.py"))
    source_paths += [
        repository / relative
        for relative in (
            "tests/pypto_dsv4_decode_csa/a3_single_op_benchmark.py",
            "tests/pypto_dsv4_decode_csa/standalone_csa_perf_case.py",
            "tests/pypto_dsv4_decode_csa/native_fixture.py",
            "vllm_ascend/ops/dsa.py",
            "vllm_ascend/attention/dsa_v1.py",
            "vllm_ascend/models/deepseek_v4.py",
        )
    ]
    source_paths += [
        runtime_binaries / name for name in ("aicore_kernel.o", "libhost_runtime.so", "libaicpu_kernel.so")
    ]
    write_json(
        output / "software_manifest.json",
        {
            "commits": {
                name: subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
                for name, path in (("vllm-ascend", repository), ("pypto", pypto), ("simpler", runtime))
            },
            "sha256": {
                str(path.relative_to(repository.parent)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in source_paths
            },
            "scope": "source and TRB binaries inspected after collection; no production source changed in this session",
        },
    )
    print(json.dumps({"formal": summaries, "coverage": coverage}, indent=2))


if __name__ == "__main__":
    main()
