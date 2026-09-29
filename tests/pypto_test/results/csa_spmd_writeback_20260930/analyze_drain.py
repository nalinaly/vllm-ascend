"""从既有DFX量化跨线程drain并集；不把线程重叠或prepare子阶段重复计时。"""

import argparse
import collections
import json
import statistics
from pathlib import Path


def analyze(path):
    events = json.loads(path.read_text())["traceEvents"]
    pids = {e["args"]["name"]: e["pid"] for e in events if e.get("ph") == "M" and e.get("name") == "process_name"}
    workers = [
        e
        for e in events
        if e.get("ph") == "X" and e.get("pid") == pids["Worker View"] and "kernel-duration-us" in e.get("args", {})
    ]
    scheduler = [e for e in events if e.get("ph") == "X" and e.get("pid") == pids["Scheduler View"]]
    drains = [
        e
        for e in events
        if e.get("ph") == "X" and e.get("pid") == pids["AICPU Scheduler"] and e.get("args", {}).get("phase") == "drain"
    ]
    origin = min(e["ts"] for e in workers)
    merged = []
    for event in sorted(drains, key=lambda e: e["ts"]):
        start, end = event["ts"], event["ts"] + event["dur"]
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
            merged[-1][2].append(event)
        else:
            merged.append([start, end, [event]])
    intervals = []
    for start, end, phases in merged:
        overlapping = collections.Counter()
        dispatched = collections.Counter()
        for event in workers:
            kernel_start = event["ts"] + event["args"]["local_setup_us"]
            if kernel_start < end and event["ts"] + event["dur"] > start:
                overlapping[event["name"].split("(")[0]] += 1
        for event in scheduler:
            if start <= event["ts"] < end:
                dispatched[event["name"].split("(")[0]] += 1
        intervals.append(
            {
                "start_us": start - origin,
                "end_us": end - origin,
                "duration_us": end - start,
                "scheduler_threads": sorted({e["tid"] for e in phases}),
                "phase_records": len(phases),
                "tasks_processed": sum(e["args"]["tasks_processed"] for e in phases),
                "overlapping_worker_counts": dict(overlapping),
                "dispatches_with_timestamp_in_interval": dict(dispatched),
            }
        )
    return {
        "source": str(path),
        "origin_worker_ts_us": origin,
        "drain_union_us": sum(e["duration_us"] for e in intervals),
        "drain_connected_intervals": len(intervals),
        "phase_records": len(drains),
        "intervals": intervals,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {
        "method": "union of AICPU Scheduler phase=drain across threads; nested prepare/publish excluded",
        "limits": (
            "DFX-only wall intervals, not device idle or isolated overhead; "
            "connected intervals are not protocol rounds; "
            "timestamp association does not prove task attribution"
        ),
        "cases": [],
    }
    lines = [
        "# SPMD同步协调的既有泳道证据",
        "",
        "单位μs。跨调度线程取drain时间并集，不叠加prepare/publish子阶段。",
        "此值是DFX中的协调区间，不等于设备空转或无profiler CSA的净开销；",
        "相连区间数也不等于协议轮数。任务关联只由同轴时间观察，不能证明因果。",
        "",
        "| 实验 | 档位 | 配置 | 两窗drain并集 | 平均 | 两窗相连区间数 |",
        "| --- | --- | --- | --- | ---: | --- |",
    ]
    for summary in args.summary:
        data = json.loads(summary.read_text())
        for case in data["cases"]:
            for name, side in case["variants"].items():
                windows = [analyze(Path(w["source"])) for w in side.get("windows", [])]
                if not windows:
                    continue
                record = {
                    "summary": str(summary),
                    "history": case["history"],
                    "batch": case["batch"],
                    "side": name,
                    "mean_drain_union_us": statistics.mean(w["drain_union_us"] for w in windows),
                    "windows": windows,
                }
                result["cases"].append(record)
                totals = "/".join(f"{w['drain_union_us']:.2f}" for w in windows)
                counts = "/".join(str(w["drain_connected_intervals"]) for w in windows)
                lines.append(
                    f"| {summary.parent.name}/{summary.stem} | {case['history']}/B{case['batch']} "
                    f"| {name} | {totals} | {record['mean_drain_union_us']:.2f} | {counts} |"
                )
    args.output.with_suffix(".json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    args.output.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
