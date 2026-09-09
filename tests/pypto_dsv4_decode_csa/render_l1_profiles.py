# SPDX-License-Identifier: Apache-2.0
"""Export native device kernels and per-core L1 child-task preview files.

The two timelines are independent replay instances, not clock-aligned. Native
starts at its first kernel; PTO starts at its first child dispatch. Profiler
and DFX overhead are present, so these previews are not latency benchmarks.
"""

import argparse
import csv
import json
import math
from decimal import Decimal
from pathlib import Path

BLUE = "#2563A6"
GOLD = "#BC8A28"
GREY = "#6B7280"


def save_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def native_files(root):
    source = next(root.glob("cann_raw/*/ASCEND_PROFILER_OUTPUT/trace_view.json"))
    data = json.loads(source.read_text())
    events = data if isinstance(data, list) else data["traceEvents"]
    hardware = {
        event["pid"]
        for event in events
        if event.get("name") == "process_name" and event.get("args", {}).get("name") == "Ascend Hardware"
    }
    device_events = [event for event in events if event.get("pid") in hardware and event.get("ph") == "X"]
    pto_model = next(event["args"]["Model Id"] for event in device_events if event["name"] == "aicore_kernel_0")
    native_model = next(
        event["args"]["Model Id"]
        for event in device_events
        if event["name"] == "MODEL_EXECUTE" and event["args"]["Model Id"] != pto_model
    )
    selected = [event for event in device_events if event["args"].get("Model Id") == native_model]
    metadata = [event for event in events if event.get("pid") in hardware and event.get("ph") == "M"]
    (root / "native_device_trace.json").write_text(json.dumps({"traceEvents": metadata + selected}))
    (root / "pto_parent_device_trace.json").write_text(
        json.dumps(
            {"traceEvents": metadata + [event for event in device_events if event["args"].get("Model Id") == pto_model]}
        )
    )
    event_keys = {(event["name"], Decimal(str(event["ts"]))) for event in selected}
    with source.with_name("kernel_details.csv").open() as handle:
        rows = [
            row for row in csv.DictReader(handle) if (row["Name"], Decimal(row["Start Time(us)"].strip())) in event_keys
        ]
    starts = sorted(Decimal(str(event["ts"])) for event in selected if event["name"] == "MODEL_EXECUTE")
    for row in rows:
        start = Decimal(row["Start Time(us)"].strip())
        row["Replay"] = sum(origin <= start for origin in starts)
    rows.sort(key=lambda row: Decimal(row["Start Time(us)"].strip()))
    counters = {}
    for row in rows:
        replay = row["Replay"]
        counters[replay] = counters.get(replay, 0) + 1
        row["ReplayKernel"] = counters[replay]
    save_csv(root / "native_kernel_details.csv", rows)
    print(f"native device kernels: {len(rows)} across {len(starts)} replays")
    return [row for row in rows if row["Replay"] == 3]


def plot(root, rows, tasks):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    origin = min(Decimal(row["Start Time(us)"].strip()) for row in rows)
    native_end = max(
        float(Decimal(row["Start Time(us)"].strip()) - origin) + float(row["Duration(us)"]) for row in rows
    )
    pto_end = max(task["finish_time_us"] for task in tasks)
    limit = math.ceil(max(native_end, pto_end) / 100) * 100
    streams = sorted({int(row["Stream ID"]) for row in rows})
    fig, ax = plt.subplots(figsize=(14, 3.7))
    for row in rows:
        start = float(Decimal(row["Start Time(us)"].strip()) - origin)
        duration = float(row["Duration(us)"])
        lane = streams.index(int(row["Stream ID"]))
        color = GOLD if "VECTOR" in row["Accelerator Core"] else BLUE
        ax.broken_barh([(start, duration)], (lane - 0.28, 0.56), facecolors=color)
        if duration >= 9:
            ax.text(
                start + duration / 2,
                lane,
                str(row["ReplayKernel"]),
                ha="center",
                va="center",
                fontsize=8,
                color="white",
            )
    ax.set_yticks(range(len(streams)), [f"Stream {stream}" for stream in streams])
    ax.set_ylim(len(streams) - 0.5, -0.5)
    ax.set_title(
        "Native dsa_forward: device kernels — replay 3\nA3 device 1 | TP1 | B4/S8 | C8191 | ACLGraph | profiler ON",
        loc="left",
        fontsize=12,
    )
    ax.set_xlabel("Time from first native device kernel (us)")
    ax.set_xlim(0, limit)
    ax.grid(axis="x", alpha=0.15)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(
        handles=[
            Patch(color=BLUE, label="AIC / mixed"),
            Patch(color=GOLD, label="Vector"),
        ],
        loc="upper right",
        fontsize=8,
    )
    fig.text(
        0.01,
        0.02,
        "Numbers index native_kernel_details.csv. Real gaps/overlap retained. "
        "Synthetic inputs/history; diagnostic timing, not serving latency.",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(root / "native_device_preview.png", dpi=160)
    fig.savefig(root / "native_device_preview.svg")
    plt.close(fig)

    cores = sorted({(task["core_type"], task["core_id"]) for task in tasks})
    fig, ax = plt.subplots(figsize=(14, 11))
    for lane, (kind, core) in enumerate(cores):
        intervals = [(task["start_time_us"], task["duration_us"]) for task in tasks if task["core_id"] == core]
        ax.broken_barh(intervals, (lane - 0.36, 0.72), facecolors=BLUE if kind == "aic" else GOLD)
    ax.set_yticks(
        range(len(cores)),
        [f"{kind.upper()} {core}" for kind, core in cores],
        fontsize=7,
    )
    ax.set_ylim(len(cores) - 0.7, -0.7)
    ax.set_xlim(0, limit)
    ax.set_xlabel("Time from first PTO child dispatch (us); graph startup/close excluded")
    ax.set_title(
        f"PyPTO CSA: L1 ACLGraph child-task swimlane — replay 3\n"
        f"A3 device 1 | TP1 | B4/S8 | C8191 | {len(tasks)} task executions | DFX level 2",
        loc="left",
        fontsize=12,
    )
    ax.grid(axis="x", alpha=0.15)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(
        0.01,
        0.02,
        "Each bar is one measured child execution, not a parent kernel. "
        "Task IDs available in Perfetto/CSV; function-name mapping and dependency arrows not collected.\n"
        "Independent replay origins; no cross-plot clock alignment or end-to-end speed comparison. "
        "Blue = AIC (Cube), gold = AIV (Vector).",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(root / "pto_child_swimlane_preview.png", dpi=160)
    fig.savefig(root / "pto_child_swimlane_preview.svg")
    plt.close(fig)


def main():
    from simpler_setup.tools.swimlane_converter import (
        generate_chrome_trace_json,
        read_perf_data,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    native = native_files(args.root)
    for replay in range(1, 7):
        folder = args.root / f"replay{replay}"
        data = read_perf_data(folder / "chip_swimlane_records.json")
        tasks = data["tasks"]
        generate_chrome_trace_json(tasks, str(folder / "swimlane.json"))
        save_csv(folder / "child_tasks.csv", tasks)
        print(f"PTO replay {replay}: {len(tasks)} child executions")
        if replay == 3:
            plot(args.root, native, tasks)


if __name__ == "__main__":
    main()
