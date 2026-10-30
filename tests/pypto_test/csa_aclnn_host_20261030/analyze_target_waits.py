# SPDX-License-Identifier: Apache-2.0
"""离线拆分真实主图及跨rank下发偏差；不改运行路径、不新增设备采样。"""

import argparse
import json
import statistics
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path


def stats(values):
    return {"min": min(values), "mean": statistics.mean(values), "max": max(values)}


def start(event):
    return Decimal(str(event["ts"]))


def end(event):
    return start(event) + Decimal(str(event["dur"]))


def graph_intervals(directory, rank):
    path, = (directory / "trace" / f"rank{rank}").glob("*/ASCEND_PROFILER_OUTPUT/trace_view.json")
    raw = json.loads(path.read_text())
    events = raw if isinstance(raw, list) else raw["traceEvents"]
    device = [e for e in events if e.get("ph") == "X" and e.get("args", {}).get("Task Type")]
    models = Counter(e["args"].get("Model Id") for e in device if e["name"] == "MODEL_EXECUTE")
    model, = [key for key, count in models.items() if key not in (None, 4294967295) and count == 3]
    rows = sorted([e for e in device if e["args"].get("Model Id") == model], key=start)
    markers = [i for i, e in enumerate(rows) if e["name"] == "MODEL_EXECUTE"] + [len(rows)]
    steps = []
    for begin, finish in zip(markers, markers[1:]):
        frame = rows[begin:finish]
        dispatches = [e for e in frame if "MoeDistributeDispatch" in e["name"]]
        combines = [e for e in frame if "MoeDistributeCombine" in e["name"]]
        if len(dispatches) != 43 or len(combines) != 43:
            raise ValueError(f"{path}: 需要43层完整MoE边界")
        if len({e["tid"] for e in dispatches + combines}) != 1:
            raise ValueError(f"{path}: MoE不在同一主流，不能按当前方式分割")
        parts = defaultdict(Decimal)
        previous = start(frame[0])
        for layer, (dispatch, combine) in enumerate(zip(dispatches, combines)):
            if not previous <= start(dispatch) <= end(dispatch) <= start(combine):
                raise ValueError(f"{path}: layer{layer}边界重叠或次序不符")
            # 前层combine结束到本层dispatch开始，可能包含融合后的前层HCpost、
            # 本层HCpre/attention/HCpost/FFN-pre/router；不称作纯CSA核耗时。
            kind = "csa_layer_attention_and_router" if layer >= 2 and layer % 2 == 0 else "other_layer_attention_and_router"
            parts[kind] += start(dispatch) - previous
            parts["dispatch_fused_compute_and_wait"] += end(dispatch) - start(dispatch)
            parts["dispatch_to_combine"] += start(combine) - end(dispatch)
            parts["combine_compute_and_wait"] += end(combine) - start(combine)
            previous = end(combine)
        parts["tail"] = max(map(end, frame)) - previous
        span = max(map(end, frame)) - start(frame[0])
        if sum(parts.values()) != span:
            raise ValueError(f"{path}: 分段未覆盖完整主图")
        steps.append({
            "span_ms": float(span / 1000),
            "parts_ms": {key: float(value / 1000) for key, value in parts.items()},
            "first_dispatch_ms": float(Decimal(str(dispatches[0]["dur"])) / 1000),
        })
    return {"rank": rank, "trace": str(path), "model_id": model, "steps": steps,
            "mean_parts_ms": {key: statistics.mean(s["parts_ms"][key] for s in steps) for key in steps[0]["parts_ms"]}}


def host_intervals(directory):
    intervals = defaultdict(list)
    events = []
    for rank in range(16):
        record = json.loads((directory / f"rank{rank}.performance.json").read_text())
        window, = record["steady_window"]
        if not window["sufficient"] or len(window["forward"]["samples_us"]) != 10:
            raise ValueError(f"{directory}: rank{rank}缺少10个有效稳态样本")
        for step in window["host_diagnostics"]["steps"]:
            events.append((step["forward_entry"]["monotonic_ns"], rank, step))
            for label, first, last in (
                ("inputs", "inputs_begin", "inputs_end"),
                ("batch_coordination", "batch_coordination_begin", "batch_coordination_end"),
                ("metadata", "attention_metadata_begin", "attention_metadata_end"),
                ("forward_submit", "event_record_ready", "forward_submitted"),
            ):
                for clock in ("monotonic_ns", "thread_cpu_ns"):
                    intervals[f"{label}_{clock}"].append((step[last][clock] - step[first][clock]) / 1e6)
    # 同机monotonic时钟聚类；不直接假定各rank第N个样本是同一个全局step。
    groups = []
    for event in sorted(events):
        if not groups or event[0] - groups[-1][-1][0] > 20_000_000:
            groups.append([])
        groups[-1].append(event)
    aligned = []
    for group in groups:
        if len(group) != 16 or len({rank for _, rank, _ in group}) != 16:
            raise ValueError(f"{directory}: Host时钟聚类不能匹配完整16rank，需人工核对")
        aligned.append({key: (max(s[key]["monotonic_ns"] for _, _, s in group)
                              - min(s[key]["monotonic_ns"] for _, _, s in group)) / 1e6
                        for key in ("execute_entry", "inputs_end", "batch_coordination_end",
                                    "attention_metadata_end", "forward_entry", "forward_submitted")})
    return {"intervals_ms": {key: stats(values) for key, values in intervals.items()},
            "aligned_step_spreads_ms": aligned,
            "spread_summary_ms": {key: stats([row[key] for row in aligned]) for key in aligned[0]},
            "alignment": "同机monotonic_ns按20ms自然间隔分组，每组恰好16个不同rank；未改计时或增加屏障。"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--pto", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {
        "scope": "已有三步独立profiling及16rank稳态10步Host诊断；两种采集窗口不同，不能逐项相减。"
                 "主图分段是互斥的时间区间，不是多流task耗时相加。MoE融合区间包含计算和等待，不是纯通信。"
                 "CSA层区间包含邻接HCpost/HCpre及router，不是纯CSA。Host偏差与通信等待有关联，未隔离因果。",
    }
    for side in ("native", "pto"):
        directory = getattr(args, side)
        result[side] = {"directory": str(directory), "host": host_intervals(directory),
                        "graphs": [graph_intervals(directory, rank) for rank in (0, 10)]}
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
