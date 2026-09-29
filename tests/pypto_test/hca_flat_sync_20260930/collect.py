"""整块Gather基线上的同步联动：保留每轮自己的基线和多任务DFX。"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    sys.path.insert(0, str(ROOT.parent))
    from hca_residual_reuse_20260930.collect import incore

    root = ROOT.parent / "results" / ROOT.name
    pairs = {}
    for side in ("dq_sync", "both_sync"):
        path = root / f"pair_{side}_h131072_b16/report.json"
        report = json.loads(path.read_text())
        assert report["status"] == "MEASURED"
        assert all(report["cross_variant_exact"].values())
        assert all(all(x.values()) for x in report["state_exact"].values())
        assert all(v == "PASS" for v in report["guards"].values())
        pairs[side] = {k: report[k] for k in ("sources", "case", "device", "cann", "summary",
                                             "cross_variant_exact", "state_exact", "guards")}
        pairs[side]["report"] = str(path)
    dfx = {side: incore(root / f"h131072_b16/swimlane_{side}/dfx/merged_swimlane.json")
           for side in ("base", "dq_sync")}
    (ROOT / "incore.json").write_text(json.dumps(dfx, indent=2) + "\n")
    (ROOT / "summary.json").write_text(json.dumps({
        "baseline_commit": "ad0e6bbe", "pairs": pairs, "incore": "incore.json",
        "decision": "not adopted; dq_sync mean neutral with lower sampled max retained as a lead; "
                    "both_sync mean regresses. DFX shows shorter dequant span but longer gather span",
        "limits": "one long representative, no short expansion or new seven-tier/model acceptance; "
                  "one DFX per side does not prove a unique contention cause",
    }, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
