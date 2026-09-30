"""记录核内改善和整段结果，边界短测不混入性能对比。"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    sys.path.insert(0, str(ROOT.parent))
    from hca_residual_reuse_20260930.collect import incore

    root = ROOT.parent / "results" / ROOT.name
    pairs = {}
    for case in ("h131072_b16", "h8192_b24", "h16507_b3"):
        path = root / f"pair_{case}/report.json"
        report = json.loads(path.read_text())
        assert report["status"] == "MEASURED"
        assert all(report["cross_variant_exact"].values())
        assert all(all(v.values()) for v in report["state_exact"].values())
        assert all(v == "PASS" for v in report["guards"].values())
        pairs[case] = {k: report[k] for k in (
            "sources", "device", "cann", "summary", "cross_variant_exact", "state_exact", "guards",
        )}
        pairs[case]["report"] = str(path)
        pairs[case]["performance_evidence"] = case != "h16507_b3"
        if "padding_graph" in report:
            graph = report["padding_graph"]
            assert graph["status"] == "PASS"
            pairs[case]["padding_graph"] = {
                "status": graph["status"], "scope": graph["scope"],
                "active_batches": [p["active_batch"] for p in graph["replays"]],
            }
    incore_data = {s: incore(root / f"h131072_b16/swimlane_{s}/dfx/merged_swimlane.json")
                   for s in ("base", "flat")}
    (ROOT / "incore.json").write_text(json.dumps(incore_data, indent=2) + "\n")
    weighted = sum(
        weight * (pairs[case]["summary"]["candidate"]["mean_us"]
                  / pairs[case]["summary"]["base"]["mean_us"] - 1)
        for weight, case in ((0.8, "h131072_b16"), (0.2, "h8192_b24"))
    )
    result = {
        "baseline_commit": "a7a9f314", "pairs": pairs,
        "weighted_mean_relative_delta_8_to_2": weighted,
        "incore": "incore.json",
        "codegen": "per-row gather loops removed; full eight-row TGATHER, arithmetic/rounding unchanged",
        "decision": "retain Q dequant incore improvement; whole long mean neutral and max higher, "
                    "short mean/max lower; stable whole-layer gain not established",
        "limits": "single-card representative and affected-tail checks; no new seven-tier or model token acceptance",
    }
    (ROOT / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"weighted_mean_delta": weighted,
                      "pairs": {k: v["summary"] for k, v in pairs.items() if v["performance_evidence"]}}, indent=2))


if __name__ == "__main__":
    main()
