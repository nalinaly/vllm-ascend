"""双query实验：实际性能、完整状态、核内与编译容量失败分开记录。"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    sys.path.insert(0, str(ROOT.parent))
    from hca_residual_reuse_20260930.collect import incore

    root = ROOT.parent / "results" / ROOT.name
    pairs = {}
    for side in ("pair_v2", "pair_hybrid_reload", "pair_three_slots"):
        path = root / f"{side}_h131072_b16/report.json"
        report = json.loads(path.read_text())
        assert report["status"] == "MEASURED", report.get("error")
        assert all(report["cross_variant_exact"].values())
        assert all(all(v.values()) for v in report["state_exact"].values())
        assert all(v == "PASS" for v in report["guards"].values())
        pairs[side] = {k: report[k] for k in ("sources", "case", "device", "cann", "summary",
                                             "cross_variant_exact", "state_exact", "guards")}
        pairs[side]["report"] = str(path)
    dfx = {s: incore(root / f"h131072_b16/swimlane_{s}/dfx/merged_swimlane.json")
           for s in ("base", "pair_v2")}
    (ROOT / "incore_gm.json").write_text(json.dumps(dfx, indent=2) + "\n")
    summary = {
        "baseline_commit": "ad0e6bbe", "pairs": pairs, "incore": ["incore_gm.json"],
        "codegen_memory": "codegen_memory.json",
        "cpu_failures": {
            "pair": "ConvertToSSA: nested branch needed explicit outer yields; fixed locally in pair_v2",
            "pair_ub": "Vec240384 > configured188416 bytes; no NPU submission",
            "pair_hybrid": "same240384 peak: both spilled quarters remained live until final publish; "
                           "fixed by reloading only final H16 slice, compiled Vec174848 in pair_hybrid_reload",
        },
        "invariants": "per-query compressed-before-raw order, block max/quantization, both raw sliding windows, "
                      "sink and RoPE retained; changes remain in private packages",
        "limits": "128K/B16 only; no claim of supported-tail, seven-tier, Native or full-model acceptance",
    }
    refine = root / "dfx_refinements"
    if refine.exists():
        dfx_refine = {s: incore(refine / f"swimlane_{s}/dfx/merged_swimlane.json")
                      for s in ("base", "pair_hybrid_reload", "pair_three_slots")}
        (ROOT / "incore_refinements.json").write_text(json.dumps(dfx_refine, indent=2) + "\n")
        summary["incore"].append("incore_refinements.json")
    (ROOT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({side: report["summary"] for side, report in pairs.items()}, indent=2))


if __name__ == "__main__":
    main()
