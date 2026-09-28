# SPDX-License-Identifier: Apache-2.0
"""汇总 HCA Worker 泳道；核内时间含等待，各任务跨度有重叠，不能相加。"""

import argparse
import collections
import json
import statistics
from pathlib import Path

from dsv4_csa_env import write_json


def summarize(path):
    events = json.loads(path.read_text())["traceEvents"]
    pids = {e["pid"] for e in events if e.get("ph") == "M" and e.get("name") == "process_name"
            and e.get("args", {}).get("name") == "Worker View"}
    workers = [e for e in events if e.get("pid") in pids and e.get("ph") == "X"
               and "kernel-duration-us" in e.get("args", {})]
    if not workers:
        raise ValueError(f"未找到 Worker 核内记录：{path}")
    origin = min(e["ts"] for e in workers)
    groups = collections.defaultdict(list)
    for event in workers:
        groups[event["name"].split("(")[0].removesuffix("_spmd")].append(event)
    tasks = {}
    for name, rows in groups.items():
        starts = [e["ts"] + e["args"]["local_setup_us"] - origin for e in rows]
        ends = [e["ts"] + e["dur"] - origin for e in rows]
        tasks[name] = {
            "blocks": len(rows), "first_start_us": min(starts), "last_start_us": max(starts),
            "last_end_us": max(ends), "start_spread_us": max(starts) - min(starts),
            "worker_envelope_us": max(ends) - min(e["ts"] - origin for e in rows),
            "kernel_mean_us": statistics.mean(e["args"]["kernel-duration-us"] for e in rows),
            "kernel_max_us": max(e["args"]["kernel-duration-us"] for e in rows),
        }
    return {"path": str(path), "worker_span_us": max(e["ts"] + e["dur"] for e in workers) - origin,
            "tasks": tasks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="含 swimlane_windows 的 report.json 所在目录")
    args = parser.parse_args()
    report = json.loads((args.directory / "report.json").read_text())
    windows = [summarize(Path(w["merged_swimlane"])) for w in report["swimlane_windows"]]
    write_json(args.directory / "worker_summary.json", {"windows": windows})
    for index, window in enumerate(windows):
        print(f"窗口 {index}：Worker 总跨度 {window['worker_span_us']:.2f} μs")
        for name, task in sorted(window["tasks"].items(), key=lambda item: item[1]["first_start_us"]):
            print(f"{name:42s} {task['blocks']:4d}核  "
                  f"{task['first_start_us']:7.2f}→{task['last_end_us']:7.2f} μs  "
                  f"核内均值 {task['kernel_mean_us']:7.2f} μs")


if __name__ == "__main__":
    main()
