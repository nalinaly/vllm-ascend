"""收集计划拆分的完整CSA/P95、关键链、总核时和状态，避免把等待转移当收益。"""

import importlib.util
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def handoff_window(path, expected_workers, helper, worker):
    analysis = helper._build_analysis(path.parent, path.parent, 2)
    summary = worker.summarize(path)
    tasks = summary["tasks"]
    if tasks["indexer_head_coefficients"]["blocks"] != expected_workers:
        raise ValueError("Unexpected coefficient worker count")
    names = {t: worker.canonical(n) for t, n in analysis.graph.name.items()}
    split = "csa_slots_window_rope_plan" in tasks
    plans = (["csa_slots_window_rope_plan", "csa_slots_compressed_plan"] if split else
             ["rope_cs", "csa_slots_build_valid_qk_plan"])
    selected = ["indexer_topk_query_merge", *plans, "qk_pv_aic"]
    timing, ids = {}, {}
    for name in selected:
        matches = [t for t, n in names.items() if n == name]
        if len(matches) != 1:
            raise ValueError(f"Expected one physical {name}: {matches}")
        task = ids[name] = matches[0]
        rows = analysis.rows_by_task[task]
        predecessors = [p for p in analysis.preds.get(task, []) if not helper._is_alloc(p, analysis)]
        if any(p not in analysis.rows_by_task for p in predecessors):
            raise ValueError(f"Untimed producer prevents ready attribution: {name}")
        timing[name] = {key: agg(float(r[field]) for r in rows) for key, field, agg in (
            ("dispatch_us", "dispatch_time_us", min), ("start_us", "start_time_us", min),
            ("end_us", "end_time_us", max), ("finish_us", "finish_time_us", max))}
        timing[name]["producers"] = [names[p] for p in predecessors]
        ready = max((float(r["finish_time_us"]) for p in predecessors
                     for r in analysis.rows_by_task[p]), default=None)
        timing[name]["last_finish_to_start_us"] = None if ready is None else timing[name]["start_us"] - ready
    for name in plans:
        if tasks[name]["blocks"] != 16:
            raise ValueError(f"Unexpected plan worker count: {name}")
    late = plans[-1]
    if "indexer_topk_query_merge" not in timing[late]["producers"]:
        raise ValueError("Compressed plan does not depend on Top-K publication")
    if late not in timing["qk_pv_aic"]["producers"]:
        raise ValueError("Sparse does not depend on the completed plan")
    if split:
        if "csa_slots_window_rope_plan" not in timing[late]["producers"]:
            raise ValueError("Shared validity DDR line lost its SWA-before-compressed dependency")
        if "indexer_topk_query_merge" in timing["csa_slots_window_rope_plan"]["producers"]:
            raise ValueError("SWA plan still waits for Top-K")
        if timing[late]["start_us"] < timing[plans[0]]["finish_us"]:
            raise ValueError("The two scalar writers overlap across tasks")
    bridge = timing["qk_pv_aic"]["start_us"] - timing["indexer_topk_query_merge"]["finish_us"]
    return {"path": str(path), "joined_rows": len(analysis.rows), "tasks": tasks,
            "worker_span_us": summary["worker_span_us"],
            "handoff": {"split": split, "timing": timing,
                        "merge_finish_to_sparse_start_us": bridge,
                        "plan_kernel_work_us": sum(tasks[n]["blocks"] * tasks[n]["kernel_mean_us"] for n in plans)}}


def main():
    common = load("plan_common", ROOT.parent / "csa_sparse_first_pv_20260929/collect.py")
    common.ROOT = ROOT
    common.TITLE = "复用RoPE符号任务承接SWA计划"
    common.DESCRIPTION = "不新增任务数；baseline核时合并原plan与rope_cs，candidate合并window+rope与压缩plan。"
    common.TARGETS = {"aic": "qk_pv_aic", "aiv": "qk_pv_aiv", "merge": "indexer_topk_query_merge"}
    original_load = common.load

    def load_with_handoff(name, path):
        module = original_load(name, path)
        if path.parent.name == "csa_score_segment_ub_20260929":
            module.schedule_window = handoff_window
        return module

    common.load = load_with_handoff
    common.main()
    evidence = json.loads((ROOT / "evidence.json").read_text())
    result = {"task": evidence["task"], "state_status": evidence["state_status"],
              "weighted_8_2_csa_pct": evidence["weighted_8_2_csa_pct"], "cases": []}
    for case in evidence["cases"]:
        sides = {s: [w["handoff"] for w in v["windows"]] for s, v in case["sides"].items()}
        changes = {k: 100 * (statistics.mean(w[k] for w in sides["candidate"]) /
                            statistics.mean(w[k] for w in sides["baseline"]) - 1)
                   for k in ("plan_kernel_work_us", "merge_finish_to_sparse_start_us")}
        result["cases"].append({"history": case["history"], "batch": case["batch"],
                                "sides": sides, "change_pct": changes})
    result["weighted_8_2_plan_work_pct"] = sum(w * c["change_pct"]["plan_kernel_work_us"]
                                              for w, c in zip((.8, .2), result["cases"]))
    (ROOT / "handoff.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    report = ROOT / "RESULTS.md"
    text = report.read_text().replace(
        "采用还需依据四窗口分布判断真实核内收益，状态失败则禁止采用；不是Native或模型token/DSpark验收。",
        "本项为调度拆分，采用依据同卡长短8:2的完整CSA与P95；合并计划/RoPE总核时和其他任务等待必须保留。不是Native或模型验收。")
    text += "\n| 档位 | 计划加RoPE总核时μs | Merge FIN→Sparse start μs |\n| --- | ---: | ---: |\n"
    for case in result["cases"]:
        cells = ['→'.join(f"{statistics.mean(w[k] for w in case['sides'][s]):.3f}"
                          for s in ("baseline", "candidate"))
                 for k in ("plan_kernel_work_us", "merge_finish_to_sparse_start_us")]
        text += f"| {case['history']//1024}K/B{case['batch']} | " + " | ".join(cells) + " |\n"
    text += "\n[每窗关键链、真实依赖与全部计划工作量](handoff.json)。\n"
    report.write_text(text)


if __name__ == "__main__":
    main()
