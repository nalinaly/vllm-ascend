"""只解析已有level4窗口，核对RoPE前置、实际提前派发及O收尾占位。"""

import argparse
import importlib.util
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[4]
TASKS = ("csa_rope_sign", "idx_qr_proj_matmul", "idx_qr_dequant_rope", "qr_hadamard_matmul",
         "qr_hadamard_quant", "indexer_head_coefficients", "indexer_score_topk_native_pair_aic",
         "qproj_matmul", "qproj_dequant_rms_nope_rope")


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--existing", action="store_true")
    args = parser.parse_args()
    helper = load("rope_early_helper", WORKSPACE / "pypto-lib/.claude/skills/critical-path/scripts/report.py")
    worker = load("rope_early_worker", ROOT.parent / "csa_scheduling_20260927/upstream_725/compare.py")
    base = ROOT.parent / "csa_indexer_rope_flat_gather_20260929" if args.existing else ROOT
    sides = ("candidate",) if args.existing else ("baseline", "candidate")
    out = {"source": str(base), "origin": "每窗首个Worker start；μs", "cases": []}
    for history, batch in ((131072, 16), (8192, 24)):
        case = {"history": history, "batch": batch, "sides": {}}
        for side in sides:
            report = json.loads((base / f"h{history}_b{batch}/swimlane/{side}/report.json").read_text())
            windows = []
            for window in report["swimlane_windows"]:
                path = Path(window["merged_swimlane"])
                analysis = helper._build_analysis(path.parent, path.parent, 2)
                names = {t: worker.canonical(n) for t, n in analysis.graph.name.items()}
                # True/False特化产生两个函数名，每次只执行其中一个。
                names = {t: "csa_rope_sign" if n == "csa_rope_sign_0" else n for t, n in names.items()}
                starts = helper._aggregate_min(analysis.rows_by_task, "start_time_us")
                ends = helper._aggregate_max(analysis.rows_by_task, "end_time_us")
                fins = helper._aggregate_max(analysis.rows_by_task, "finish_time_us")
                dispatches = helper._aggregate_min(analysis.rows_by_task, "dispatch_time_us")
                origin = min(starts.values())
                tasks = {}
                for task, name in names.items():
                    if name not in TASKS or task not in starts:
                        continue
                    preds = [p for p in analysis.preds[task] if not helper._is_alloc(p, analysis)]
                    missing = [p for p in preds if p not in fins]
                    last = max(preds, key=fins.get) if preds and not missing else None
                    early, early_rows, total_rows = helper._observed_early(task, analysis, 2e6 / analysis.graph.freq)
                    tasks[name] = {
                        "task_id": task, "worker_count": len(analysis.rows_by_task[task]),
                        "producer_early_flag": analysis.task_table[task]["early_dispatch"],
                        "first_start_us": starts[task] - origin, "last_end_us": ends[task] - origin,
                        "end_to_finish_us": fins[task] - ends[task],
                        "pred_finish_to_dispatch_us": dispatches[task] - fins[last] if last else None,
                        "pred_finish_to_start_us": starts[task] - fins[last] if last else None,
                        "early": early, "early_rows": early_rows, "total_rows": total_rows,
                        "untimed_predecessors": [names.get(p, p) for p in missing],
                        "predecessors": [{"id": p, "name": names.get(p, p),
                                          "producer_early_flag": analysis.task_table.get(p, {}).get("early_dispatch")}
                                         for p in preds],
                    }
                quant = [r for t, rows in analysis.rows_by_task.items() if names[t] == "quant" for r in rows]
                final = [r for t, rows in analysis.rows_by_task.items()
                         if names[t].startswith("proj_b_act_hc_post") for r in rows]
                qend = max(float(r["end_time_us"]) for r in quant)
                name_ids = {n: t for t, n in names.items() if t in starts}
                hadamard = analysis.rows_by_task[name_ids["qr_hadamard_matmul"]]
                idx_finish = fins[name_ids["idx_qr_proj_matmul"]]
                qb_finish = fins[name_ids["qproj_matmul"]]
                windows.append({"path": str(path), "window": window["window"], "tasks": tasks,
                                "final_dispatched_before_last_quant_end": sum(
                                    float(r["dispatch_time_us"]) < qend for r in final),
                                "final_workers": len(final),
                                # Earlier receipt is evidence of occupancy, not proof of blocking another task.
                                "hadamard_received_before_idx_mm_finish": sum(
                                    float(r["receive_time_us"]) < idx_finish for r in hadamard),
                                "hadamard_received_before_q_b_finish": sum(
                                    float(r["receive_time_us"]) < qb_finish for r in hadamard),
                                "hadamard_mean_local_setup_us": statistics.mean(
                                    float(r["local_setup_us"]) for r in hadamard)})
            case["sides"][side] = {"windows": windows, "mean_us": {
                n: {field: statistics.mean(w["tasks"][n][field] for w in windows)
                    for field in ("first_start_us", "last_end_us", "end_to_finish_us")} for n in TASKS},
                "idx_dequant_early_windows": [w["tasks"]["idx_qr_dequant_rope"]["early"] for w in windows]}
        out["cases"].append(case)
    name = "existing_handoff.json" if args.existing else "handoff.json"
    (ROOT / name).write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    print(name, [{"history": c["history"], "batch": c["batch"],
                  "early": {s: v["idx_dequant_early_windows"] for s, v in c["sides"].items()}}
                 for c in out["cases"]])


if __name__ == "__main__":
    main()
