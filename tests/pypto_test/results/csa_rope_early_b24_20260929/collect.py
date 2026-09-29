"""收集128K/B24控制档：正式性能后检查完整状态，保留必要的前置时序。"""

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
TASKS = ("qproj_matmul", "qproj_dequant_rms_nope_rope", "idx_qr_proj_matmul", "idx_qr_dequant_rope",
         "qr_hadamard_matmul", "qr_hadamard_quant", "indexer_head_coefficients",
         "indexer_score_topk_native_pair_aic", "indexer_score_topk_native_pair_aiv",
         "indexer_topk_query_merge", "qk_pv_aic", "qk_pv_aiv", "proj_a_mm", "quant", "proj_b_mm")
sys.path.insert(0, str(ROOT.parents[1]))
from dsv4_csa_validation import compare_tensor  # noqa: E402


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
        raise RuntimeError(f"Wait for the same task: {status}")
    source = read(ROOT / "source.json")
    pair = load("rope_b24_pair", ROOT.parent / "csa_compiled_pair_20260929/analyze.py")
    worker = load("rope_b24_worker", ROOT.parent / "csa_scheduling_20260927/upstream_725/compare.py")
    worker.summarize = functools.partial(worker.summarize, publication_task="qk_pv_aiv")
    helper = load("rope_b24_helper", WORKSPACE / "pypto-lib/.claude/skills/critical-path/scripts/report.py")
    torch.set_num_threads(4)
    result = {"task": task, "task_status": status, "source": source, "history": 131072, "batch": 24, "sides": {}}
    states, native = {}, {}
    for side in ("baseline", "candidate"):
        folder = ROOT / f"h131072_b24/timing/{side}"
        report = read(folder / "report.json")
        if (report["source"], report["variant"], report["history"], report["batch"], report["side"]) != (
                source["source_prefix"] + "-" + side, source["variant"], 131072, 24, "pto"):
            raise ValueError("Frozen source or case mismatch")
        value = pair.analyze_side(folder)
        value["samples_us"] = report["timing"]["samples_us"]
        value["stdev_us"] = statistics.stdev(value["samples_us"])
        value["over_p50_5pct"] = sum(v > 1.05 * value["us_p50"] for v in value["samples_us"])
        if value["topk"]["structural_errors"]:
            raise ValueError("Timing Top-K structure failed")
        # Keep actual config and graph guard proof; omit repeated profile type lists.
        value = {k: v for k, v in value.items() if k in (
            "source", "device", "cann", "requested", "compiler", "mean_us", "us_p50", "us_p95", "us_max",
            "states", "guards", "profile_json", "samples_us", "stdev_us", "over_p50_5pct")}
        states[side] = torch.load(folder / "states.pt", map_location="cpu", weights_only=True)
        if set(states[side]) != STATES:
            raise ValueError("Full state coverage missing")
        path = ROOT / f"h131072_b24/swimlane/{side}/report.json"
        dfx = read(path)
        if (dfx["status"], dfx["history"], dfx["batch"], dfx["variant"], dfx["effective_weight_nz_mode"],
                dfx["deterministic_level"], dfx["pto_reduction"]["atomic_add"]) != (
                "MEASURED", 131072, 24, source["variant"], 2, 0, 0):
            raise ValueError("DFX source or configuration changed")
        if set(dfx["pto_self"]) != STATES or any(v["status"] != "PASS" for v in dfx["pto_self"].values()):
            raise ValueError("DFX graph self state check failed")
        if any(v["status"] != "PASS" for checks in dfx["pto_guards"] for v in checks.values()):
            raise ValueError("DFX guard violation")
        if dfx["topk_selection"]["structural_errors"]:
            raise ValueError("DFX Top-K structure failed")
        windows = dfx["swimlane_windows"]
        if len(windows) != 4 or any(not w["exported"] or w["execution"] != "graph_replay" for w in windows):
            raise ValueError("Missing four graph windows")
        value["windows"] = []
        for window in windows:
            swimlane = Path(window["merged_swimlane"])
            analysis = helper._build_analysis(swimlane.parent, swimlane.parent, 2)
            names = {t: worker.canonical(n) for t, n in analysis.graph.name.items()}
            ids = {n: t for t, n in names.items()}
            target = ids["idx_qr_dequant_rope"]
            early, early_rows, rows = helper._observed_early(target, analysis, 2e6 / analysis.graph.freq)
            preds = [p for p in analysis.preds[target] if not helper._is_alloc(p, analysis)]
            starts = helper._aggregate_min(analysis.rows_by_task, "start_time_us")
            finishes = helper._aggregate_max(analysis.rows_by_task, "finish_time_us")
            missing = [p for p in preds if p not in finishes]
            gap = None if missing else starts[target] - max(finishes[p] for p in preds)
            summary = worker.summarize(swimlane)
            if rows != 48 or summary["tasks"]["indexer_head_coefficients"]["blocks"] != 24:
                raise ValueError("Unexpected worker coverage")
            value["windows"].append({
                "path": str(swimlane), "joined_rows": len(analysis.rows), "worker_span_us": summary["worker_span_us"],
                "worker_phases_us": summary["phases_us"], "idx_dequant_early": early,
                "early_rows": early_rows, "idx_dequant_pred_finish_to_start_us": gap,
                "untimed_predecessors": [names.get(p, p) for p in missing],
                "rope_producer_early": analysis.task_table[ids["csa_rope_sign"]]["early_dispatch"],
                "tasks": {n: {k: summary["tasks"][n][k] for k in (
                    "blocks", "first_start_us", "last_end_us", "kernel_mean_us")} for n in TASKS},
            })
        native[side] = {"pto_native": dfx["pto_native"], "topk_selection": dfx["topk_selection"]}
        value["native_reference"] = {
            "source": str(path), "errors": {n: {k: v[k] for k in (
                "status", "max_abs", "rmse", "mismatches", "nonfinite")} for n, v in dfx["pto_native"].items()},
            "topk": {k: dfx["topk_selection"][k] for k in (
                "position_mismatches", "different_set_rows", "replaced_indices", "structural_errors")},
        }
        result["sides"][side] = value
    a, b = (result["sides"][s] for s in ("baseline", "candidate"))
    for key in ("device", "cann", "requested"):
        if a[key] != b[key]:
            raise ValueError(f"Within-pair configuration changed: {key}")
    result["state_checks"] = {k: compare_tensor(states["candidate"][k], states["baseline"][k], 0, 0)
                              for k in sorted(STATES)}
    result["state_status"] = "PASS" if all(v["status"] == "PASS" for v in result["state_checks"].values()) else "FAIL"
    result["native_metrics_equal"] = native["baseline"] == native["candidate"]
    result["csa_change_pct"] = 100 * (b["mean_us"] / a["mean_us"] - 1)
    (ROOT / "evidence.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    lines = ["# RoPE early：128K/B24控制档", "", "同卡、CANN9.2、mode2/atomic0/det0；5预热20次正式事件。",
             "复用原冻结pkg，仅补缺口；没有改变源码或把本档拼入旧七档。", "",
             "| 档位 | 基线→候选μs | 变化 | P95μs | maxμs |", "| --- | ---: | ---: | ---: | ---: |",
             f"| 128K/B24 | {a['mean_us']:.3f}→{b['mean_us']:.3f} | {result['csa_change_pct']:+.3f}% "
             f"| {a['us_p95']:.3f}→{b['us_p95']:.3f} | {a['us_max']:.3f}→{b['us_max']:.3f} |", "",
             f"性能后八类完整状态零容差：{result['state_status']}。图重放、保护区及八窗官方覆盖通过。",
             f"Native误差/Top-K指标新旧相同：{result['native_metrics_equal']}；不代表Native或EP16精度验收。", "",
             "| Task | 首start | 末end | 核内均值 |", "| --- | ---: | ---: | ---: |"]
    for name in TASKS:
        cells = ["→".join(f"{statistics.mean(w['tasks'][name][key] for w in v['windows']):.3f}" for v in (a, b))
                 for key in ("first_start_us", "last_end_us", "kernel_mean_us")]
        lines.append("| " + " | ".join([name, *cells]) + " |")
    lines += ["", "Task时刻以每窗首Worker receive归零；独立DFX不与正式事件相减，也不用于推算完整CSA。",
              "[原样本、配置和状态](evidence.json)。"]
    (ROOT / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    if result["state_status"] != "PASS":
        raise SystemExit("Full state comparison failed")


if __name__ == "__main__":
    main()
