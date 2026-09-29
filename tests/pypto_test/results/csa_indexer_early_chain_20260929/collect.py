"""收集纯调度候选，核对完整CSA/P95、Score/Sparse及七标志是否实际解锁early。"""

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def schedule_window(path, expected_workers, helper, worker):
    analysis = helper._build_analysis(path.parent, path.parent, 2)
    summary = worker.summarize(path)
    if summary["tasks"]["indexer_head_coefficients"]["blocks"] != expected_workers:
        raise ValueError("Unexpected coefficient workers")
    names = {t: worker.canonical(n) for t, n in analysis.graph.name.items()}
    starts = helper._aggregate_min(analysis.rows_by_task, "start_time_us")
    finishes = helper._aggregate_max(analysis.rows_by_task, "finish_time_us")
    origin = min(starts.values())
    selected = {"scatter_softmax_pool_0", "indexer_boundary_init", "rmsnorm_rope", "kv_hadamard",
                "kv_and_cache_write", "idx_kv_scale_commit", "rmsnorm_rope_cache_write",
                "indexer_head_coefficients", "indexer_score_topk_native_pair_aic", "qk_pv_aic",
                "qproj_matmul", "qproj_dequant_rms_nope_rope"}
    chain = []
    for tid in sorted(analysis.rows_by_task, key=starts.get):
        if names[tid] not in selected:
            continue
        preds = [p for p in analysis.preds.get(tid, []) if not helper._is_alloc(p, analysis)]
        missing = [p for p in preds if p not in finishes]
        ready = max((finishes[p] for p in preds if p in finishes), default=None)
        observed, early_rows, rows = helper._observed_early(tid, analysis, 2e6 / analysis.graph.freq)
        chain.append({"name": names[tid], "start_us": starts[tid] - origin,
                      "finish_us": finishes[tid] - origin,
                      "eligible": helper._early_eligible(tid, analysis),
                      "observed_early": observed, "early_rows": early_rows, "rows": rows,
                      "last_finish_to_start_us": None if missing or ready is None else starts[tid] - ready,
                      "untimed_producers": missing,
                      "non_early_producers": [names.get(p, p) for p in preds
                                              if not helper._early_producer(p, analysis)]})
    return {"path": str(path), "joined_rows": len(analysis.rows), "tasks": summary["tasks"],
            "worker_span_us": summary["worker_span_us"], "chain": chain}


def main():
    common = load("early_common", ROOT.parent / "csa_sparse_first_pv_20260929/collect.py")
    common.ROOT = ROOT
    common.TITLE = "Indexer cache生产链补七处early标志"
    common.DESCRIPTION = "纯调度改动，不能把未改任务的核时波动记成独立incore优化；按完整CSA的8:2与P95取舍。"
    common.TARGETS = {"score_aic": "indexer_score_topk_native_pair_aic",
                      "score_aiv": "indexer_score_topk_native_pair_aiv",
                      "sparse_aic": "qk_pv_aic", "sparse_aiv": "qk_pv_aiv"}
    original_load = common.load

    def custom_load(name, path):
        module = original_load(name, path)
        if path.parent.name == "csa_score_segment_ub_20260929":
            module.schedule_window = schedule_window
        return module

    common.load = custom_load
    common.main()
    evidence = json.loads((ROOT / "evidence.json").read_text())
    chain = {"task": evidence["task"], "cases": []}
    for case in evidence["cases"]:
        chain["cases"].append({"history": case["history"], "batch": case["batch"],
                               "sides": {s: [{"path": w["path"], "chain": w["chain"]} for w in v["windows"]]
                                         for s, v in case["sides"].items()}})
    (ROOT / "chain.json").write_text(json.dumps(chain, ensure_ascii=False, indent=2) + "\n")
    report = ROOT / "RESULTS.md"
    report.write_text(report.read_text().replace(
        "采用还需依据四窗口分布判断真实核内收益，状态失败则禁止采用；不是Native或模型token/DSpark验收。",
        "调度候选以完整CSA与尾部决定，核时用于排查资源争用；状态失败则禁止采用。不是Native或模型验收。")
        + "\n[逐窗生产链的真实early及交接等待](chain.json)。\n")


if __name__ == "__main__":
    main()
