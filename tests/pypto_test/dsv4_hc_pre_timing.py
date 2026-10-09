# SPDX-License-Identifier: Apache-2.0
"""HC_pre 的 incore 起止统计；不把运行时包络或多核累计时间作为算子耗时。"""

import collections
import json
import math
import statistics
from pathlib import Path


def interval_metrics(intervals):
    """并行区间只计一次；空隙仅标为无 incore 执行，不猜测其调度原因。"""
    ordered = sorted(intervals)
    if not ordered or any(not math.isfinite(v) for pair in ordered for v in pair):
        raise ValueError("incore 时间区间为空或含非有限值")
    if any(end <= start for start, end in ordered):
        raise ValueError("incore 时间区间长度必须为正")
    first, last = ordered[0][0], max(end for _, end in ordered)
    left, right = ordered[0]
    active = 0.0
    for start, end in ordered[1:]:
        if start > right:
            active += right - left
            left, right = start, end
        else:
            right = max(right, end)
    active += right - left
    return {"first_start_us": first, "last_end_us": last, "span_us": last - first,
            "active_union_us": active, "no_incore_gap_us": max(0.0, last - first - active)}


def summarize_swimlane(path, steps):
    events = json.loads(Path(path).read_text())["traceEvents"]
    workers = {e["pid"] for e in events if e.get("ph") == "M" and e.get("name") == "process_name"
               and e.get("args", {}).get("name") == "Worker View"}
    epochs = collections.defaultdict(list)
    for event in events:
        args = event.get("args", {})
        if event.get("pid") not in workers or event.get("ph") != "X" or "kernel-duration-us" not in args:
            continue
        if "launch_epoch" not in args:
            raise ValueError("泳道缺少 launch_epoch，不能混合多个 replay 的任务")
        end = event["ts"] + event["dur"]
        # Worker 条形从 receive 开始，需排除 local_setup，只保留 kernel 本身。
        start = end - args["kernel-duration-us"]
        if not math.isclose(start, event["ts"] + args["local_setup_us"], abs_tol=1e-6):
            raise ValueError("Worker 时间与 kernel/local_setup 字段不一致")
        epochs[args["launch_epoch"]].append((event, start, end))
    if len(epochs) != steps:
        raise ValueError(f"需要 {steps} 次 incore 执行，实际 {len(epochs)}")
    samples = []
    for epoch, rows in sorted(epochs.items()):
        tasks = collections.defaultdict(list)
        for event, start, end in rows:
            tasks[event["name"].split("(")[0].removesuffix("_spmd")].append((start, end))
        samples.append({"launch_epoch": epoch, **interval_metrics([(start, end) for _, start, end in rows]),
                        "blocks": len(rows), "tasks": {
                            name: {**interval_metrics(intervals), "blocks": len(intervals),
                                   "core_time_sum_us": sum(end - start for start, end in intervals)}
                            for name, intervals in tasks.items()}})
    return {"source": str(path), "level": 1, "steps": steps, "samples": samples,
            **{f"{name}_p50_us": statistics.median(sample[f"{name}_us"] for sample in samples)
               for name in ("span", "active_union", "no_incore_gap")}}


def summarize_native(path, steps):
    raw = json.loads(Path(path).read_text())
    events = raw["traceEvents"] if isinstance(raw, dict) else raw
    kernels = sorted([e for e in events if e.get("ph") == "X" and e.get("name") == "HcPre"],
                     key=lambda e: e["ts"])
    if len(kernels) != steps or any(not math.isfinite(e["dur"]) or e["dur"] <= 0 for e in kernels):
        raise ValueError(f"需要 {steps} 个有效 Native HcPre 设备事件，实际 {len(kernels)}")
    samples = [e["dur"] for e in kernels]
    return {"source": str(path), "steps": steps, "p50_us": statistics.median(samples), "samples_us": samples}
