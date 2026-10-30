# SPDX-License-Identifier: Apache-2.0
"""离线对齐16rank已有三步主图、Host launch和MoE到达时间，使用整数ns。"""

import argparse
import json
import sqlite3
import statistics
from pathlib import Path


def load_rank(root, rank):
    path, = (root / "trace" / f"rank{rank}").glob("*/ASCEND_PROFILER_OUTPUT/ascend_pytorch_profiler_*.db")
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as db:
        models = db.execute(
            "select t.modelId,count(*) from TASK t join STRING_IDS n on n.id=t.taskType "
            "where n.value='MODEL_EXECUTE' group by t.modelId"
        ).fetchall()
        model, = [key for key, count in models if key != 4294967295 and count == 3]
        executes = db.execute(
            "select t.startNs,t.connectionId from TASK t join STRING_IDS n on n.id=t.taskType "
            "where n.value='MODEL_EXECUTE' and t.modelId=? order by t.startNs", (model,)
        ).fetchall()
        kernels = db.execute(
            "select t.startNs,t.endNs,t.streamId,n.value from TASK t "
            "join COMPUTE_TASK_INFO i using(globalTaskId) join STRING_IDS n on n.id=i.name "
            "where t.modelId=? order by t.startNs", (model,)
        ).fetchall()
        result = []
        for index, (begin, connection) in enumerate(executes):
            limit = executes[index + 1][0] if index + 1 < len(executes) else 2**63 - 1
            end, = db.execute("select max(endNs) from TASK where modelId=? and startNs>=? and startNs<?",
                              (model, begin, limit)).fetchone()
            api, = db.execute(
                "select a.startNs,a.endNs from CANN_API a join STRING_IDS n on n.id=a.name "
                "where a.connectionId=? and n.value='aclmdlRIExecuteAsync'", (connection,)
            ).fetchall()
            frame = [row for row in kernels if begin <= row[0] < limit]
            dispatches = [row for row in frame if "MoeDistributeDispatch" in row[3]]
            combines = [row for row in frame if "MoeDistributeCombine" in row[3]]
            if len(dispatches) != 43 or len(combines) != 43:
                raise ValueError(f"rank{rank} step{index}: 没有43组MoE边界")
            result.append({"rank": rank, "model_id": model, "host_begin_ns": api[0], "host_end_ns": api[1],
                           "graph_begin_ns": begin, "graph_end_ns": end,
                           "dispatches_ns": [[row[0], row[1]] for row in dispatches],
                           "combines_ns": [[row[0], row[1]] for row in combines]})
    return result


def spread_ms(values):
    return (max(values) - min(values)) / 1e6


def ranges(values):
    return {"min": min(values), "mean": statistics.mean(values), "max": max(values)}


def analyze(root):
    ranks = [load_rank(root, rank) for rank in range(16)]
    steps = []
    for step in range(3):
        rows = [rank[step] for rank in ranks]
        # 同一主图的三个窗口必须全rank互不重叠，才能按序号配对。
        if step < 2 and max(row["graph_end_ns"] for row in rows) >= min(
            rank[step + 1]["graph_begin_ns"] for rank in ranks
        ):
            raise ValueError("跨rank窗口重叠，不能按序号对齐，须人工匹配")
        layers = []
        for layer in range(43):
            ds = [row["dispatches_ns"][layer] for row in rows]
            cs = [row["combines_ns"][layer] for row in rows]
            layers.append({"layer": layer,
                           "dispatch_entry_spread_ms": spread_ms([x[0] for x in ds]),
                           "dispatch_exit_spread_ms": spread_ms([x[1] for x in ds]),
                           "dispatch_fused_duration_ms": ranges([(x[1] - x[0]) / 1e6 for x in ds]),
                           "combine_entry_spread_ms": spread_ms([x[0] for x in cs]),
                           "combine_exit_spread_ms": spread_ms([x[1] for x in cs]),
                           "combine_duration_ms": ranges([(x[1] - x[0]) / 1e6 for x in cs])})
        steps.append({"step": step,
                      "host_launch_entry_spread_ms": spread_ms([x["host_begin_ns"] for x in rows]),
                      "host_launch_exit_spread_ms": spread_ms([x["host_end_ns"] for x in rows]),
                      "device_graph_entry_spread_ms": spread_ms([x["graph_begin_ns"] for x in rows]),
                      "device_graph_exit_spread_ms": spread_ms([x["graph_end_ns"] for x in rows]),
                      "graph_span_ms": ranges([(x["graph_end_ns"] - x["graph_begin_ns"]) / 1e6 for x in rows]),
                      "after_first_dispatch_ms": ranges([
                          (x["graph_end_ns"] - x["dispatches_ns"][0][1]) / 1e6 for x in rows
                      ]),
                      "layers": layers, "ranks": rows})
    return {"directory": str(root), "steps": steps,
            "scope": "已有三步独立profiling，非稳态十步计时；首步含profiling初始化扰动。"
                     "Host API通过connectionId关联MODEL_EXECUTE；整数ns沿profiler归一化时钟对齐。"
                     "MoE融合task包含计算和通信等待，不能把持续时间或rank差异全部解释为通信。"
                     "after_first_dispatch保留后续完整设备链，只作诊断，不能作为无profiler性能下界。"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.directory)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps([{k: v for k, v in row.items() if k not in ("layers", "ranks")}
                      | {"first_moe": row["layers"][0]} for row in result["steps"]], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
