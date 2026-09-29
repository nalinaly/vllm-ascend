"""FP32残差试验：完整CSA/P95、首调用舍入差异、连续调用及长档任务时序。"""

import functools
import importlib.util
import json
import statistics
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[4]
STATES = {"x_out", "idx_topk", "swa.0", "compressed.0", "state.0", "indexer.0", "indexer.1", "indexer_state.0"}
TASKS = ("hc_widen_rms", "hc_rms", "hc_pre_rms", "hc_pre_linear", "hc_pre_linear_reduce",
         "split_pre_post", "comb_sinkhorn",
         "mix_x_rms_norm", "idx_qr_dequant_rope", "indexer_score_topk_native_pair_aiv", "qk_pv_aiv",
         "proj_b_act_hc_post")
sys.path.insert(0, str(ROOT.parents[1]))
from dsv4_csa_validation import compare_tensor, compare_topk  # noqa: E402


def read(path):
    return json.loads(path.read_text())


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    task = (ROOT / "task.txt").read_text().strip()
    status = subprocess.check_output(["task-submit", "--status", task], text=True).strip()
    if status != "completed (exit=0)":
        raise RuntimeError(f"等待同一任务成功完成：{status}")
    torch.set_num_threads(4)
    source = read(ROOT / "source.json")
    pair = load("fp32_residual_pair", ROOT.parent / "csa_compiled_pair_20260929/analyze.py")
    worker = load("fp32_residual_worker", ROOT.parent / "csa_scheduling_20260927/upstream_725/compare.py")
    worker.summarize = functools.partial(worker.summarize, publication_task="qk_pv_aiv")
    helper = load("fp32_residual_helper", WORKSPACE / "pypto-lib/.claude/skills/critical-path/scripts/report.py")
    evidence = {"task": task, "task_status": status, "source": source, "cases": []}
    for history, batch in source["cases"]:
        case = {"history": history, "batch": batch, "sides": {}}
        states, chain = {}, {}
        for side in ("baseline", "candidate"):
            folder = ROOT / f"h{history}_b{batch}/timing/{side}"
            report = read(folder / "report.json")
            dtype = "torch.bfloat16" if side == "baseline" else "torch.float32"
            if (report["status"], report["source"], report["variant"], report["history"], report["batch"],
                    report["residual_dtype"], report["output_dtype"]) != (
                    "MEASURED", source["source_prefix"] + "-" + side, source["variant"], history, batch, dtype, dtype):
                raise ValueError("私有来源、档位或残差类型不符")
            value = pair.analyze_side(folder)
            if value["topk"]["structural_errors"]:
                raise ValueError("Top-K结构错误")
            value = {k: v for k, v in value.items() if k in (
                "source", "device", "cann", "requested", "compiler", "mean_us", "us_p50", "us_p95", "us_max",
                "states", "guards", "profile_json", "topk")}
            value["samples_us"] = report["timing"]["samples_us"]
            value["stdev_us"] = statistics.stdev(value["samples_us"])
            value["over_p50_5pct"] = sum(t > 1.05 * value["us_p50"] for t in value["samples_us"])
            value["residual_dtype"] = dtype
            value["chain"] = report["residual_chain"]
            if value["chain"]["status"] != "PASS":
                raise ValueError("连续输入的自身图检查失败")
            if side == "candidate" and value["chain"]["non_bf16_values"] <= 0:
                raise ValueError("没有真正用到FP32低位，连续输入检查无效")
            states[side] = torch.load(folder / "states.pt", map_location="cpu", weights_only=True)
            chain[side] = torch.load(folder / "chain_states.pt", map_location="cpu", weights_only=True)
            if set(states[side]) != STATES or set(chain[side]) != STATES:
                raise ValueError("完整状态缺失")
            if [history, batch] in source["swimlane_cases"]:
                dfx_root = Path(source["reused_baseline_dfx"]) if side == "baseline" else ROOT
                dfx = read(dfx_root / f"h{history}_b{batch}/swimlane/{side}/report.json")
                if len(dfx["swimlane_windows"]) != source["windows"]:
                    raise ValueError("缺少独立DFX窗口")
                if any(v["status"] != "PASS" for v in dfx["pto_self"].values()):
                    raise ValueError("DFX自身状态变化")
                if any(v["status"] != "PASS" for checks in dfx["pto_guards"] for v in checks.values()):
                    raise ValueError("DFX保护区变化")
                value["windows"] = []
                for window in dfx["swimlane_windows"]:
                    path = Path(window["merged_swimlane"])
                    analysis = helper._build_analysis(path.parent, path.parent, 2)
                    summary = worker.summarize(path)
                    value["windows"].append({
                        "path": str(path), "joined_rows": len(analysis.rows),
                        "worker_span_us": summary["worker_span_us"], "worker_phases_us": summary["phases_us"],
                        "tasks": {n: t for n, t in summary["tasks"].items()
                                  if n in TASKS or n.startswith("proj_b_act_hc_post")},
                    })
                value["native_reference"] = {
                    "pto_native": dfx["pto_native"], "topk_selection": dfx["topk_selection"],
                    "scope": "FP32输出对Native BF16输出加宽后比较；不是相同舍入规则或模型验收",
                }
            case["sides"][side] = value
        a, b = (case["sides"][s] for s in ("baseline", "candidate"))
        for key in ("device", "cann", "requested"):
            if a[key] != b[key]:
                raise ValueError(f"同轮配置不同：{key}")
        case["first_call_checks"] = {
            n: compare_tensor(states["candidate"][n].bfloat16() if n == "x_out" else states["candidate"][n],
                              states["baseline"][n], 0, 0) for n in sorted(STATES)}
        case["first_output_without_rounding"] = compare_tensor(
            states["candidate"]["x_out"], states["baseline"]["x_out"].float(), 0, 0)
        case["first_call_status"] = (
            "PASS" if all(v["status"] == "PASS" for v in case["first_call_checks"].values()) else "FAIL")
        case["chain_cross_version"] = {
            n: compare_tensor(chain["candidate"][n].float() if n == "x_out" else chain["candidate"][n],
                              chain["baseline"][n].float() if n == "x_out" else chain["baseline"][n], 0, 0)
            for n in sorted(STATES)}
        visible = ((history + torch.arange(6, dtype=torch.int64) + 1) // 4).repeat(batch)
        case["chain_topk"] = compare_topk(chain["candidate"]["idx_topk"], chain["baseline"]["idx_topk"], visible)
        case["chain_cross_version_scope"] = "前次输出的舍入不同，下一次输入已不同；记录误差，不要求跨版本零差异"
        case["csa_change_pct"] = 100 * (b["mean_us"] / a["mean_us"] - 1)
        case["saved_us"] = a["mean_us"] - b["mean_us"]
        evidence["cases"].append(case)
        del states, chain
    evidence["weighted_8_2_csa_pct"] = sum(w * c["csa_change_pct"] for w, c in zip((.8, .2), evidence["cases"]))
    (ROOT / "evidence.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n")
    lines = ["# FP32残差流：单CSA实测", "", "同卡CANN9.2/mode2/atomic0，第二CSA层metadata复用。",
             "BF16同值输入一次加宽在计时外；输入输出FP32，内部无残差回转。5预热20事件；非整模型收益。", "",
             "| 档位 | BF16 μs | FP32 μs | 节省μs | 耗时变化 | BF16/FP32 P95μs |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for case in evidence["cases"]:
        a, b = (case["sides"][s] for s in ("baseline", "candidate"))
        lines.append(f"| {case['history']//1024}K/B{case['batch']} | {a['mean_us']:.3f} | {b['mean_us']:.3f} "
                     f"| {case['saved_us']:.3f} | {case['csa_change_pct']:+.3f}% "
                     f"| {a['us_p95']:.3f}/{b['us_p95']:.3f} |")
    lines += ["", f"两代表档8:2耗时变化：{evidence['weighted_8_2_csa_pct']:+.3f}%。", "",
              "首调用将FP32输出按BF16舍入再比旧版，其余七类完整状态直接比；逐元素结果见原始记录。",
              "性能后还将本次输出接回输入：两版本分别检查同输入eager/graph精确一致和保护区，",
              "FP32输入确实含BF16不能表示的低位。跨版本第二次输入已不同，另记误差，不能假称零差异。",
              "这是同层权重的两次attention-half调用，不包含FFN/HCA，也不是token/DSpark模型验收。",
              "[计时样本、逐元素状态和长档任务时序](evidence.json)。"]
    (ROOT / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    if any(c["first_call_status"] != "PASS" or c["chain_topk"]["structural_errors"] for c in evidence["cases"]):
        raise SystemExit("舍入以外的状态差异或连续调用Top-K结构错误")


if __name__ == "__main__":
    main()
