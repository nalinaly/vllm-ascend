"""保留各候选自己的同卡基线及完整状态检查，不跨轮拼性能。"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    sys.path.insert(0, str(ROOT.parent))
    from hca_residual_reuse_20260930.collect import incore

    root = ROOT.parent / "results" / ROOT.name
    for filename, folder, sides in (
        ("incore.json", "h131072_b16", ("base", "al1")),
        ("incore_slices_npipe.json", "dfx_slices_npipe", ("base", "slices", "npipe")),
    ):
        data = {side: incore(root / folder / f"swimlane_{side}/dfx/merged_swimlane.json") for side in sides}
        (ROOT / filename).write_text(json.dumps(data, indent=2) + "\n")
    pairs = {}
    for side, folder in (("al1", "pair_h131072_b16"),
                         ("slices", "pair_slices_h131072_b16"),
                         ("npipe", "pair_npipe_h131072_b16")):
        path = root / folder / "report.json"
        report = json.loads(path.read_text())
        assert report["status"] == "MEASURED"
        assert all(report["cross_variant_exact"].values())
        assert all(all(x.values()) for x in report["state_exact"].values())
        assert all(v == "PASS" for v in report["guards"].values())
        pairs[side] = {k: report[k] for k in (
            "sources", "case", "device", "cann", "summary", "cross_variant_exact", "state_exact", "guards",
        )}
        pairs[side]["report"] = str(path)
    summary = {
        "baseline_commit": "a7a9f314", "pairs": pairs,
        "incore": ["incore.json", "incore_slices_npipe.json"],
        "codegen": {
            "al1": "K1024 resident A slice changes Left CompactMode Null/TMOV to Normal/TEXTRACT; "
                   "both use K256, correcting the initial frozen comment about K128",
            "slices": "four independent K256 A tiles preserve Left Null/TMOV",
            "npipe": "Mat 384KiB, Left/Right two 32KiB slots, Acc one 64KiB slot; "
                     "stage2 did not generate two Acc slots",
        },
        "decision": "no production adoption: al1 and npipe Q incore regress; slices Q mean slightly improves "
                    "but group span and whole mean regress, retained as a local-reuse lead; "
                    "npipe lower sampled whole max is retained as a scheduling lead, not a Q incore gain",
        "limits": "single-card screening and one independent DFX per side; no new Native/model acceptance",
    }
    (ROOT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({side: pair["summary"] for side, pair in pairs.items()}, indent=2))


if __name__ == "__main__":
    main()
