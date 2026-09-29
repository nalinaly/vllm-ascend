"""七档固定window_3的query/Score及O投影前置等待；只做CPU现有泳道分析。"""

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[4]
TASKS = (
    "qr_proj_matmul", "qr_rms_norm_quant", "qproj_matmul", "qproj_dequant_rms_nope_rope",
    "idx_qr_proj_matmul", "idx_qr_dequant_rope", "qr_hadamard_matmul", "qr_hadamard_quant",
    "indexer_head_coefficients", "indexer_score_topk_native_pair_aic", "indexer_topk_query_merge",
    "csa_slots_build_valid_qk_plan", "qk_pv_aic", "proj_a_mm", "quant", "proj_b_mm",
    "proj_b_act_hc_post",
)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    evidence = json.loads((ROOT / "evidence.json").read_text())
    if evidence["task_status"] != "completed (exit=0)" or len(evidence["cases"]) != 7:
        raise ValueError("Need all seven completed cases")
    worker = load("gather_seven_worker", ROOT.parent / "csa_scheduling_20260927/upstream_725/compare.py")
    helper = load("gather_seven_helper", WORKSPACE / "pypto-lib/.claude/skills/critical-path/scripts/report.py")
    out = {"operator_commit": evidence["source"]["operator_commit"], "window": 3, "cases": []}
    lines = ["# 七档调度下一步依据：固定window_3", "",
             "只分析已有level4泳道，不新增占卡。所有时刻按该窗口首个Worker start归零，单位μs。",
             "end→FIN为逻辑任务末核结束到最后结束通知；FIN→dispatch可因early为负。",
             "前置有无时戳的dummy则ready不可确认，不把部分前置当完整ready。",
             "dispatch→start和启动分散含依赖门控/多波/竞争，均不是纯AICPU软件开销。",
             "同名多行是不同逻辑任务（如O投影各组），原始task_id与直接前置见scheduling.json。",
             "独立DFX不能与正式CSA事件相减，也不用于解释未同时profiling的P95样本。", ""]
    for case in evidence["cases"]:
        window = case["worker_windows"][3]
        path = Path(window["path"])
        if path.parent.name != "window_3":
            raise ValueError("Fixed window changed")
        analysis = helper._build_analysis(path.parent, path.parent, 2)
        names = {t: worker.canonical(n) for t, n in analysis.graph.name.items()}
        for task, name in names.items():
            if name.startswith("proj_b_act_hc_post"):
                names[task] = "proj_b_act_hc_post"
        starts = helper._aggregate_min(analysis.rows_by_task, "start_time_us")
        ends = helper._aggregate_max(analysis.rows_by_task, "end_time_us")
        finishes = helper._aggregate_max(analysis.rows_by_task, "finish_time_us")
        dispatches = helper._aggregate_min(analysis.rows_by_task, "dispatch_time_us")
        origin = min(starts.values())
        rows = []
        for name in TASKS:
            ids = [t for t, n in names.items() if n == name and t in starts]
            if not ids:
                continue
            for task in ids:
                preds = [p for p in analysis.preds[task] if not helper._is_alloc(p, analysis)]
                missing = [p for p in preds if p not in ends]
                observed = [p for p in preds if p in ends]
                last = max(observed, key=finishes.get) if observed and not missing else None
                early, early_rows, total_rows = helper._observed_early(task, analysis, 2e6 / analysis.graph.freq)
                row = {"task": name, "task_id": task, "start_us": starts[task] - origin,
                       "end_us": ends[task] - origin, "end_to_finish_us": finishes[task] - ends[task],
                       "latest_predecessor_finish": names[last] if last else None,
                       "predecessor_finish_to_dispatch_us": dispatches[task] - finishes[last] if last else None,
                       "dispatch_to_start_us": starts[task] - dispatches[task],
                       "untimed_predecessors": [names.get(p, p) for p in missing],
                       "predecessors": [names.get(p, p) for p in preds],
                       "early": early, "early_rows": early_rows, "observed_rows": total_rows}
                rows.append(row)
        item = {"history": case["history"], "batch": case["batch"], "path": str(path), "tasks": rows,
                "worker_phases_us": window["worker_phases_us"], "worker_span_us": window["worker_span_us"]}
        out["cases"].append(item)
        lines += [f"## {case['history']//1024}K/B{case['batch']}", "",
                  "| Task | 首start | 末end | end→FIN | 最晚直接前置FIN | FIN→dispatch | dispatch→start | early |",
                  "| --- | ---: | ---: | ---: | --- | ---: | ---: | --- |"]
        for row in rows:
            latest = row["latest_predecessor_finish"] or (
                "未计时前置，无法确认" if row["untimed_predecessors"] else "—")
            gap = row["predecessor_finish_to_dispatch_us"]
            gap_text = "—" if gap is None else f"{gap:.2f}"
            lines.append(f"| {row['task']} | {row['start_us']:.2f} | {row['end_us']:.2f} "
                         f"| {row['end_to_finish_us']:.2f} | {latest} | {gap_text} "
                         f"| {row['dispatch_to_start_us']:.2f} | {row['early']} |")
        lines += ["", f"[固定原始窗口]({path})", ""]
    reference = ROOT.parent / "csa_scheduling_20260927/upstream_725/report.json"
    up = json.loads(reference.read_text())["upstream"]
    short = next(c for c in out["cases"] if (c["history"], c["batch"]) == (8192, 16))
    out["historical_upstream"] = {"path": up["path"], "worker_phases_us": up["phases_us"],
                                  "worker_span_us": up["worker_span_us"],
                                  "limit": "旧图缺源码/输入完整证明及Scheduler View；只对照组织，不是同输入A/B"}
    lines += ["## 727.98μs历史上游组织参照", "",
              "上游缺版本和Scheduler View，不能比较纯调度开销；此处使用当前8K/B16的固定window_3。",
              "上游merge_norm独立、当前融合在qk_pv，比较的两段终点都在最终发布之后。", "",
              "| 不重叠Worker分段 | 历史上游 | 当前8K/B16 |", "| --- | ---: | ---: |"]
    labels = ("首receive→norm结束", "norm结束→Sparse首receive", "Sparse首receive→最终发布", "发布→末Worker")
    for label, old, new in zip(labels, up["phases_us"].values(), short["worker_phases_us"].values()):
        lines.append(f"| {label} | {old:.2f} | {new:.2f} |")
    lines.append(f"| 总Worker窗口 | {up['worker_span_us']:.2f} | {short['worker_span_us']:.2f} |")
    (ROOT / "SCHEDULING.md").write_text("\n".join(lines) + "\n")
    (ROOT / "scheduling.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    print("Seven fixed-window scheduling reports written")


if __name__ == "__main__":
    main()
