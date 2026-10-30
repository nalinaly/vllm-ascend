# SPDX-License-Identifier: Apache-2.0
"""离线记录真实主图、SK和metadata覆盖；不把重叠task耗时累加为网络时间。"""

import argparse
import json
import statistics
from collections import Counter
from decimal import Decimal
from pathlib import Path


def time_us(event):
    return Decimal(str(event["ts"]))


def analyze(path):
    raw = json.loads(path.read_text())
    events = raw if isinstance(raw, list) else raw["traceEvents"]
    hardware = [e for e in events if e.get("ph") == "X" and e.get("args", {}).get("Task Type")]
    executes = Counter(e["args"].get("Model Id") for e in hardware if e["name"] == "MODEL_EXECUTE")
    candidates = [model for model, count in executes.items() if model not in (None, 4294967295) and count == 3]
    if len(candidates) != 1:
        raise ValueError(f"{path}: 不能唯一识别三次重放的主图：{dict(executes)}")
    model = candidates[0]
    rows = sorted([e for e in hardware if e["args"].get("Model Id") == model], key=time_us)
    markers = [i for i, event in enumerate(rows) if event["name"] == "MODEL_EXECUTE"]
    if markers[0] != 0:
        raise ValueError(f"{path}: 首个主图task先于MODEL_EXECUTE，须人工核对窗口")
    steps = []
    for begin, end in zip(markers, markers[1:] + [len(rows)]):
        frame = rows[begin:end]
        kernels = [e for e in frame if e["args"]["Task Type"].startswith("KERNEL")]
        by_name = {}
        for event in kernels:
            by_name.setdefault(event["name"], []).append(event["dur"])
        steps.append({
            "main_graph_span_us": float(max(time_us(e) + Decimal(str(e["dur"])) for e in frame) - time_us(frame[0])),
            "sk_tasks": sum(e["name"].startswith("sk_") for e in kernels),
            "csa_named_tasks": sum("CsaSpmdTndDevicePto" in e["name"] for e in kernels),
            "csa_named_sk_tasks": sum(
                e["name"].startswith("sk_") and "CsaSpmdTndDevicePto" in e["name"] for e in kernels
            ),
            "moe_dispatch_named_tasks": sum("MoeDistributeDispatch" in e["name"] for e in kernels),
            "moe_combine_named_tasks": sum("MoeDistributeCombine" in e["name"] for e in kernels),
            "kernel_records": [{"name": name, "count": len(values), "mean_us": statistics.mean(values),
                                "sum_us": sum(values)} for name, values in by_name.items()],
        })
    metadata = {}
    for event in hardware:
        name = event["name"]
        if "Metadata" in name or "MetaData" in name:
            metadata.setdefault(name, []).append(event["dur"])
    return {
        "trace": str(path),
        "model_id": model,
        "steps": steps,
        "metadata_full_window": {name: {"count": len(values), "mean_us": statistics.mean(values)}
                                 for name, values in metadata.items()},
        "scope": "三步独立profile，包含采集开销；MODEL_EXECUTE区分主图，metadata统计包含draft等消费者。"
                 "命名SK任务可包含多个算子，首尾名称不能穷举内部算子。kernel sum包含多流重叠及通信等待，"
                 "不能当作网络时间或纯通信时间；稳态性能见独立事件结果。",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--rank", type=int, default=0)
    args = parser.parse_args()
    trace, = (args.directory / "trace" / f"rank{args.rank}").glob("*/ASCEND_PROFILER_OUTPUT/trace_view.json")
    report = analyze(trace)
    output = args.directory / f"rank{args.rank}.graph_coverage.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(output), "model_id": report["model_id"],
                      "steps": [{k: v for k, v in row.items() if k != "kernel_records"} for row in report["steps"]],
                      "metadata": report["metadata_full_window"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
