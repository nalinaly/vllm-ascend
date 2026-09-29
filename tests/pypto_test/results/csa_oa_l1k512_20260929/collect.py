"""复用完整状态/官方四窗校验，单列相同64份O-A工作的核时与跨度。"""

import importlib.util
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    path = ROOT.parent / "csa_sparse_first_pv_20260929/collect.py"
    spec = importlib.util.spec_from_file_location("oa_k_collection", path)
    common = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = common
    spec.loader.exec_module(common)
    common.ROOT = ROOT
    common.TITLE = "NZ O-A小中档按K512预取L1，大档保持K256"
    common.TARGETS = {"oa": "proj_a_mm", "score_aic": "indexer_score_topk_native_pair_aic",
                      "score_aiv": "indexer_score_topk_native_pair_aiv",
                      "sparse_aic": "qk_pv_aic", "sparse_aiv": "qk_pv_aiv"}
    common.DESCRIPTION = (
        "本轮两侧均B16/T96，覆盖受影响的N128分支；64份O-A工作及依赖不变。"
        "大档N256保持K256，在同一套算子内按实际T分支，不外推大档性能。"
    )
    common.main()
    evidence = json.loads((ROOT / "evidence.json").read_text())
    details = {"task": evidence["task"], "cases": []}
    lines = ["", "| 档位 | O-A基线/候选四窗核时μs | 基线/候选跨度μs |",
             "| --- | --- | --- |"]
    for case in evidence["cases"]:
        row = {"history": case["history"], "batch": case["batch"], "sides": {}}
        for side, value in case["sides"].items():
            row["sides"][side] = [{k: w["tasks"]["proj_a_mm"][k] for k in (
                "blocks", "kernel_mean_us", "first_start_us", "last_end_us", "start_spread_us")}
                for w in value["windows"]]
        ranges, spans = [], []
        for side in ("baseline", "candidate"):
            windows = row["sides"][side]
            means = [w["kernel_mean_us"] for w in windows]
            ranges.append("/".join(f"{v:.3f}" for v in means))
            spans.append(f"{statistics.mean(w['last_end_us'] - w['first_start_us'] for w in windows):.3f}")
        lines.append(f"| {case['history']//1024}K/B{case['batch']} | {' → '.join(ranges)} "
                     f"| {' → '.join(spans)} |")
        details["cases"].append(row)
    (ROOT / "oa.json").write_text(json.dumps(details, ensure_ascii=False, indent=2) + "\n")
    p = ROOT / "RESULTS.md"
    p.write_text(p.read_text() + "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
