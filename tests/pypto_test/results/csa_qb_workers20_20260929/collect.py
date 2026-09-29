"""Q_B分工变化比较整组总核时和跨度，避免用变大的单worker均值误判。"""

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


def main():
    chain = load("qb_chain", ROOT.parent / "csa_indexer_early_chain_20260929/collect.py")
    common = load("qb_common", ROOT.parent / "csa_sparse_first_pv_20260929/collect.py")
    common.ROOT = ROOT
    common.TITLE = "Q_B持久worker 24→20"
    common.DESCRIPTION = "分工改变但总列数及整数矩阵工作保持；核内判断用整组总核时，采用仍看完整CSA的8:2及P95。"
    common.TARGETS = {"score_aic": "indexer_score_topk_native_pair_aic",
                      "score_aiv": "indexer_score_topk_native_pair_aiv",
                      "sparse_aic": "qk_pv_aic", "sparse_aiv": "qk_pv_aiv"}
    original_load = common.load

    def custom_load(name, path):
        module = original_load(name, path)
        if path.parent.name == "csa_score_segment_ub_20260929":
            module.schedule_window = chain.schedule_window
        return module

    common.load = custom_load
    common.main()
    evidence = json.loads((ROOT / "evidence.json").read_text())
    result = {"task": evidence["task"], "cases": []}
    for case in evidence["cases"]:
        item = {"history": case["history"], "batch": case["batch"], "sides": {}}
        for side, value in case["sides"].items():
            windows = []
            for w in value["windows"]:
                q = w["tasks"]["qproj_matmul"]
                if q["blocks"] != (24 if side == "baseline" else 20):
                    raise ValueError("实际Q_B worker数未按候选生效")
                windows.append({"path": w["path"], "kernel_work_us": q["blocks"] * q["kernel_mean_us"],
                                "kernel_mean_us": q["kernel_mean_us"],
                                "first_start_us": q["first_start_us"], "last_end_us": q["last_end_us"],
                                "span_us": q["last_end_us"] - q["first_start_us"],
                                "start_spread_us": q["start_spread_us"],
                                "occupancy": w["qproj_occupancy"], "chain": w["chain"]})
            item["sides"][side] = windows
        item["change_pct"] = {k: 100 * (statistics.mean(w[k] for w in item["sides"]["candidate"]) /
                                       statistics.mean(w[k] for w in item["sides"]["baseline"]) - 1)
                              for k in ("kernel_work_us", "span_us")}
        result["cases"].append(item)
    result["weighted_8_2_kernel_work_pct"] = sum(w * c["change_pct"]["kernel_work_us"]
                                                for w, c in zip((.8, .2), result["cases"]))
    (ROOT / "qb.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    lines = ["", "| 档位 | Q_B总核时μs | Q_B跨度μs |", "| --- | ---: | ---: |"]
    for case in result["cases"]:
        cells = ["→".join(f"{statistics.mean(w[k] for w in case['sides'][s]):.3f}"
                          for s in ("baseline", "candidate")) for k in ("kernel_work_us", "span_us")]
        lines.append(f"| {case['history']//1024}K/B{case['batch']} | " + " | ".join(cells) + " |")
    lines += ["", "[每窗Q_B工作量、物理核复用及后续链路](qb.json)。"]
    report = ROOT / "RESULTS.md"
    report.write_text(report.read_text() + "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
