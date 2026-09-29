"""双head反量化保持48份工作，比较核内均值/分布、关键链、CSA/P95与完整状态。"""

import importlib.util
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TITLE = "Q反量化满行按双head合批"
DESCRIPTION = "48-worker和尾行不变；保持逐head 512列归约与量化顺序，另列目标核的四窗分布。"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    chain = load("dq_chain", ROOT.parent / "csa_indexer_early_chain_20260929/collect.py")
    common = load("dq_common", ROOT.parent / "csa_sparse_first_pv_20260929/collect.py")
    common.ROOT = ROOT
    common.TITLE = TITLE
    common.DESCRIPTION = DESCRIPTION
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
                q = w["tasks"]["qproj_dequant_rms_nope_rope"]
                if q["blocks"] != 48:
                    raise ValueError("Q反量化worker数不应改变")
                windows.append({"path": w["path"], "kernel_mean_us": q["kernel_mean_us"],
                                "first_start_us": q["first_start_us"], "last_end_us": q["last_end_us"],
                                "span_us": q["last_end_us"] - q["first_start_us"],
                                "start_spread_us": q["start_spread_us"], "chain": w["chain"]})
            item["sides"][side] = windows
        item["kernel_change_pct"] = 100 * (
            statistics.mean(w["kernel_mean_us"] for w in item["sides"]["candidate"]) /
            statistics.mean(w["kernel_mean_us"] for w in item["sides"]["baseline"]) - 1)
        result["cases"].append(item)
    result["weighted_8_2_kernel_pct"] = sum(w * c["kernel_change_pct"]
                                           for w, c in zip((.8, .2), result["cases"]))
    (ROOT / "dequant.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    lines = ["", "| 档位 | Q反量化核时μs | 变化 | 基线四窗范围 | 候选四窗范围 |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for case in result["cases"]:
        samples = [[w["kernel_mean_us"] for w in case["sides"][s]] for s in ("baseline", "candidate")]
        means = "→".join(f"{statistics.mean(v):.3f}" for v in samples)
        ranges = " | ".join(f"{min(v):.3f}–{max(v):.3f}" for v in samples)
        lines.append(f"| {case['history']//1024}K/B{case['batch']} | {means} "
                     f"| {case['kernel_change_pct']:+.3f}% | {ranges} |")
    lines += ["", f"目标核8:2变化{result['weighted_8_2_kernel_pct']:+.3f}%；"
              "[全部窗口与query关键链](dequant.json)。"]
    p = ROOT / "RESULTS.md"
    p.write_text(p.read_text() + "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
