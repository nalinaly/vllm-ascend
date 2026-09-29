"""复用已验证7b296153的八个窗口检查early断点；不编译、不新增设备采样。"""

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[4]
SOURCE = ROOT.parent / "csa_ob_hc_scalar_fused_20260929/evidence.json"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    helper = load("early_review_helper", WORKSPACE / "pypto-lib/.claude/skills/critical-path/scripts/report.py")
    worker = load("early_review_worker", ROOT.parent / "csa_scheduling_20260927/upstream_725/compare.py")
    source = json.loads(SOURCE.read_text())
    result = {"source": str(SOURCE), "source_commit": "7b296153", "new_device_sampling": False, "cases": []}
    selected = {"hc_widen_rms", "hc_pre_linear", "kv_proj_matmul", "kv_rms_norm_rope",
                "csa_cache_writeback", "kv_score_proj", "scatter_softmax_pool", "compress_state_commit",
                "indexer_boundary_init", "rmsnorm_rope", "kv_hadamard", "kv_and_cache_write",
                "idx_kv_scale_commit", "rmsnorm_rope_cache_write", "qproj_matmul",
                "qproj_dequant_rms_nope_rope", "indexer_head_coefficients",
                "indexer_score_topk_native_pair_aic", "indexer_topk_query_merge", "qk_pv_aic"}
    for case in source["cases"]:
        item = {"history": case["history"], "batch": case["batch"], "windows": []}
        for window in case["sides"]["candidate"]["windows"]:
            path = Path(window["path"])
            analysis = helper._build_analysis(path.parent, path.parent, 2)
            names = {t: worker.canonical(n) for t, n in analysis.graph.name.items()}
            starts = helper._aggregate_min(analysis.rows_by_task, "start_time_us")
            finishes = helper._aggregate_max(analysis.rows_by_task, "finish_time_us")
            origin = min(starts.values())
            tasks = []
            for tid in sorted(analysis.rows_by_task, key=starts.get):
                name = names[tid]
                if name not in selected and name.removesuffix("_0") not in selected:
                    continue
                preds = [p for p in analysis.preds.get(tid, []) if not helper._is_alloc(p, analysis)]
                missing = [p for p in preds if p not in finishes]
                ready = max((finishes[p] for p in preds if p in finishes), default=None)
                observed, early_rows, rows = helper._observed_early(tid, analysis, 2e6 / analysis.graph.freq)
                tasks.append({
                    "task_id": tid, "name": name,
                    "allows_consumers_early": helper._early_producer(tid, analysis),
                    "eligible_from_producers": helper._early_eligible(tid, analysis),
                    "observed_early": observed, "early_rows": early_rows, "rows": rows,
                    "start_us": starts[tid] - origin, "finish_us": finishes[tid] - origin,
                    "last_finish_to_start_us": None if missing or ready is None else starts[tid] - ready,
                    "untimed_producers": missing,
                    "producers": [{"task_id": p, "name": names.get(p, p),
                                   "early": helper._early_producer(p, analysis)} for p in preds],
                })
            item["windows"].append({"path": str(path), "joined_rows": len(analysis.rows), "tasks": tasks})
        result["cases"].append(item)
    (ROOT / "existing_windows.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    for case in result["cases"]:
        print(f"{case['history']//1024}K/B{case['batch']} window 0")
        for task in case["windows"][0]["tasks"]:
            gap = task["last_finish_to_start_us"]
            print(task["name"], f"own={task['allows_consumers_early']}",
                  f"eligible={task['eligible_from_producers']}", task["observed_early"],
                  f"gap={None if gap is None else round(gap, 3)}",
                  "blocked_by=" + ",".join(p["name"] for p in task["producers"] if not p["early"]))


if __name__ == "__main__":
    main()
