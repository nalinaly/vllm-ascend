# SPDX-License-Identifier: Apache-2.0
"""按功能对照 HCA 的 PTO incore task 与 Native 算子耗时。

Native 取同轮计时后的 profiler（kernel_details.csv），PTO 取“Native 之后”口径的 DFX 泳道。
Native 基本单流串行，算子时长即该功能耗时；PTO 任务之间有重叠，给出每组的任务跨度
（首块开始到末块结束）、单块核内均值/最大值与核时间总和，不把跨度相加当作整层时长。
"""

import argparse
import collections
import csv
import json
import re
from pathlib import Path

# (功能, Native 匹配规则, PTO 任务名前缀)；Native 规则为 (算子类型, 首个输入形状前缀)，形状前缀为空表示不限。
GROUPS = [
    ("残差加宽", [("TensorMove", "{T},4,4096", 0)], ["hca_hc_widen"]),
    ("mHC pre + 输入 RMSNorm", [("HcPre", "", None), ("RmsNorm", "{T},4096", None)],
     ["hc_pre_rms", "hc_pre_linear", "comb_sinkhorn", "mix_x_rms_norm"]),
    ("Q_A 投影", [("MatMulV2", "{T},4096;256,64", None)], ["qr_proj_matmul"]),
    ("QR RMSNorm+量化", [("RmsNormDynamicQuant", "", None)], ["qr_rms_norm_quant"]),
    ("KV 投影", [("MatMulV2", "{T},4096;256,32", None)], ["kv_proj_matmul"]),
    ("KV RMSNorm+RoPE", [("RmsNorm", "{T},512", None), ("InplacePartialRotaryMul", "{T},1,1,512", None)],
     ["kv_rms_norm_rope"]),
    ("SWA cache 写回", [("ScatterNdUpdateV2", "", "{T},2")], ["hca_raw_cache_write"]),
    ("Q_B 投影", [("QuantBatchMatmulV3", "{T},1024", None)], ["qproj_matmul"]),
    ("Q 反量化+head RMSNorm+RoPE", [("triton_rms_kernel_0", "", None), ("InplacePartialRotaryMul", "{T},1,64,512", 0)],
     ["qproj_dequant_rms_nope_rope"]),
    ("C128 Compressor", [("Compressor", "", None), ("ScatterNdUpdateV2", "", "{B},2")],
     ["hca_kv_score_proj", "hca_softmax_pool", "hca_norm_rope_write", "hca_state_commit"]),
    ("PTO 独有准备（RoPE 表/有效长度/compact 偏移）", [], ["q_rope_prepare", "hca_raw_valid", "hca_inverse_rope_sign",
                                                   "hca_compact_offsets"]),
    ("Sparse attention + 逆 RoPE",
     [("SparseAttnSharedkv", "", None), ("TensorMove", "{T},64,512", None), ("Neg", "", None),
      ("InplacePartialRotaryMul", "{T},1,64,512", 1)],
     ["hca_gather_kv", "hca_cmp_work_gather", "hca_short_attention", "hca_unified_attention", "hca_attention",
      "hca_raw_attention", "hca_cmp_attention", "hca_merge", "hca_publish"]),
    ("O 投影（O_A + 量化 + O_B）",
     [("TransposeBatchMatMul", "", None), ("DynamicQuant", "", None), ("QuantBatchMatmulV3", "{T},8192", None)],
     ["proj_a_mm", "quant", "_proj_b_mm", "proj_b_mm"]),
    ("O 反量化 + mHC post", [("TensorMove", "{T},4096", None), ("HcPost", "", None), ("TensorMove", "{T},4,4096", 1)],
     ["hca_oproj_hc_post"]),
    ("预热", [], ["hca_warm"]),
]


def native_ops(path):
    rows = list(csv.DictReader(open(path)))
    rows.sort(key=lambda r: float(r["Start Time(us)"]))
    return [(r["Type"], r["Input Shapes"].strip('"'), float(r["Duration(us)"]), float(r["Start Time(us)"]))
            for r in rows if r["Type"] != "CompressorMetadata"]


def pto_tasks(path):
    events = json.loads(Path(path).read_text())["traceEvents"]
    launch = [e for e in events if e.get("pid") == 7 and e.get("ph") == "X"][0]
    tasks = collections.defaultdict(lambda: {"start": 1e18, "end": 0.0, "kernel": []})
    for e in events:
        if e.get("pid") != 4 or e.get("ph") != "X" or "kernel-duration-us" not in e.get("args", {}):
            continue
        name = e["name"].split("(")[0].removesuffix("_spmd")
        name = re.sub(r"_\d+$", "", name)
        task = tasks[name]
        task["start"] = min(task["start"], e["ts"] - launch["ts"] + e["args"].get("local_setup_us", 0))
        task["end"] = max(task["end"], e["ts"] + e["dur"] - launch["ts"])
        task["kernel"].append(e["args"]["kernel-duration-us"])
    return launch["dur"], tasks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("native_kernel_details", type=Path)
    parser.add_argument("pto_swimlane", type=Path)
    parser.add_argument("--tokens", type=int, required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    ops = native_ops(args.native_kernel_details)
    launch, tasks = pto_tasks(args.pto_swimlane)
    used_ops, used_tasks, rows = set(), set(), []
    for title, rules, prefixes in GROUPS:
        native_us, native_names = 0.0, []
        for op_type, shape, which in rules:
            shape = shape.format(T=args.tokens, B=args.batch)
            matches = [i for i, (t, s, _, _) in enumerate(ops) if t == op_type and i not in used_ops
                       and (not shape or (s.startswith(shape) if isinstance(which, int) or which is None else True))
                       and (not isinstance(which, str) or which.format(T=args.tokens, B=args.batch) in s)]
            if isinstance(which, int):
                matches = [i for i in matches if not shape or ops[i][1].startswith(shape)]
            for i in matches[:1] if isinstance(which, int) else matches:
                used_ops.add(i)
                native_us += ops[i][2]
                native_names.append(f"{ops[i][0]} {ops[i][2]:.1f}")
        names = [n for n in tasks if any(n.startswith(p) for p in prefixes) and n not in used_tasks]
        used_tasks.update(names)
        if names:
            start = min(tasks[n]["start"] for n in names)
            end = max(tasks[n]["end"] for n in names)
            kernels = [k for n in names for k in tasks[n]["kernel"]]
            pto = {"span_us": end - start, "start_us": start, "end_us": end, "blocks": len(kernels),
                   "kernel_mean_us": sum(kernels) / len(kernels), "kernel_max_us": max(kernels),
                   "core_us": sum(kernels), "tasks": {n: {"blocks": len(tasks[n]["kernel"]),
                                                           "span_us": tasks[n]["end"] - tasks[n]["start"],
                                                           "kernel_mean_us": sum(tasks[n]["kernel"]) / len(tasks[n]["kernel"])}
                                                       for n in names}}
        else:
            pto = None
        rows.append({"function": title, "native_us": native_us, "native_ops": native_names, "pto": pto})
    leftover_ops = [f"{ops[i][0]} {ops[i][1][:30]} {ops[i][2]:.1f}" for i in range(len(ops)) if i not in used_ops]
    leftover_tasks = [n for n in tasks if n not in used_tasks]
    total_native = sum(op[2] for op in ops)
    print(f"Native 算子合计 {total_native:.1f} μs；PTO DFX launch {launch:.1f} μs")
    print("| 功能 | Native μs | PTO 跨度 μs | PTO 起止 μs | 块数 | 单块均值/最大 μs | 核时间 μs | 跨度−Native |")
    print("| --- | ---: | ---: | --- | ---: | --- | ---: | ---: |")
    for row in rows:
        p = row["pto"]
        if p:
            print(f"| {row['function']} | {row['native_us']:.1f} | {p['span_us']:.1f} | {p['start_us']:.0f}→{p['end_us']:.0f} "
                  f"| {p['blocks']} | {p['kernel_mean_us']:.1f}/{p['kernel_max_us']:.1f} | {p['core_us']:.0f} "
                  f"| {p['span_us'] - row['native_us']:+.1f} |")
        else:
            print(f"| {row['function']} | {row['native_us']:.1f} | — | — | — | — | — | — |")
    print()
    for row in rows:
        if row["pto"] and len(row["pto"]["tasks"]) > 1:
            detail = "；".join(f"{n} {t['blocks']}块 跨度{t['span_us']:.1f} 均值{t['kernel_mean_us']:.1f}"
                              for n, t in row["pto"]["tasks"].items())
            print(f"- {row['function']}：{detail}；Native {', '.join(row['native_ops'])}")
    print("未归类 Native：", leftover_ops)
    print("未归类 PTO：", leftover_tasks)
    if args.output:
        args.output.write_text(json.dumps({"native_total_us": total_native, "pto_launch_us": launch, "rows": rows,
                                           "leftover_native": leftover_ops, "leftover_pto": leftover_tasks},
                                          ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
