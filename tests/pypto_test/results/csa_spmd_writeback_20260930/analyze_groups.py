"""按真实task ID区分SPMD组，统计核时、组跨度及同窗其他任务的执行重叠。"""

import argparse
import collections
import json
import statistics
from pathlib import Path


def analyze(path, prefixes):
    events = json.loads(path.read_text())["traceEvents"]
    pid = next(
        e["pid"]
        for e in events
        if e.get("ph") == "M" and e.get("name") == "process_name" and e["args"]["name"] == "Worker View"
    )
    cores = {
        e["tid"]: e["args"]["name"]
        for e in events
        if e.get("ph") == "M" and e.get("name") == "thread_name" and e.get("pid") == pid
    }
    workers = [
        e for e in events if e.get("ph") == "X" and e.get("pid") == pid and "kernel-duration-us" in e.get("args", {})
    ]
    origin = min(e["ts"] for e in workers)
    rows = [
        {
            "name": e["name"],
            "core": cores[e["tid"]],
            "start": e["ts"] + e["args"]["local_setup_us"] - origin,
            "end": e["ts"] + e["dur"] - origin,
            "kernel_us": e["args"]["kernel-duration-us"],
        }
        for e in workers
    ]
    groups = collections.defaultdict(list)
    for row in rows:
        groups[row["name"]].append(row)
    result = []
    for name, group in groups.items():
        if not name.startswith(prefixes):
            continue
        start, end = min(r["start"] for r in group), max(r["end"] for r in group)
        overlap = collections.defaultdict(float)
        boundaries = {start, end}
        relevant = []
        for row in rows:
            lo, hi = max(start, row["start"]), min(end, row["end"])
            if lo < hi:
                relevant.append(row)
                boundaries.update((lo, hi))
                if row["name"] != name:
                    overlap[row["name"]] += hi - lo
        area = collections.defaultdict(float)
        peak = collections.defaultdict(int)
        times = sorted(boundaries)
        for lo, hi in zip(times, times[1:]):
            active = {r["core"] for r in relevant if r["start"] < hi and r["end"] > lo}
            for kind in ("AIC", "AIV"):
                count = sum(core.startswith(kind + "_") for core in active)
                area[kind] += count * (hi - lo)
                peak[kind] = max(peak[kind], count)
        values = [r["kernel_us"] for r in group]
        result.append(
            {
                "task": name,
                "workers": len(group),
                "physical_cores": len({r["core"] for r in group}),
                "first_start_us": start,
                "last_end_us": end,
                "envelope_us": end - start,
                "kernel_min_us": min(values),
                "kernel_max_us": max(values),
                "kernel_mean_us": statistics.mean(values),
                "all_tasks_active_cores_mean": {kind: value / (end - start) for kind, value in area.items()},
                "all_tasks_active_cores_peak": dict(peak),
                "other_tasks_overlap_core_us": dict(sorted(overlap.items(), key=lambda item: -item[1])),
            }
        )
    return {"source": str(path), "groups": sorted(result, key=lambda r: r["first_start_us"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prefix", nargs="+", default=("csa_cache_writeback", "scatter_softmax_pool"))
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text())
    output = {
        "method": (
            "Worker View body intervals, grouped by complete name including task ID; "
            "active cores exclude local_setup"
        ),
        "limits": "observed execution overlap only; no proof of bandwidth contention, ready time, or causality",
        "cases": [],
    }
    for case in summary["cases"]:
        for name, side in case["variants"].items():
            if not side.get("windows"):
                continue
            output["cases"].append(
                {
                    "history": case["history"],
                    "batch": case["batch"],
                    "side": name,
                    "windows": [analyze(Path(w["source"]), tuple(args.prefix)) for w in side["windows"]],
                }
            )
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
