"""收集跨query候选的交错筛选、受影响边界及完整服务入口/泳道证据。"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT.parent / "results" / ROOT.name


def main():
    pairs = {}
    for history, batch in ((131072, 16), (8192, 24), (16507, 6), (16384, 2)):
        suffix = "_fixed" if (history, batch) == (16384, 2) else ""
        path = RESULTS / f"pair_h{history}_b{batch}{suffix}/report.json"
        report = json.loads(path.read_text())
        assert report["status"] == "MEASURED", path
        assert all(report["cross_variant_exact"].values()), path
        assert all(all(values.values()) for values in report["state_exact"].values()), path
        assert all(value == "PASS" for value in report["guards"].values()), path
        if history in (16507, 16384):
            assert report["padding_graph"]["status"] == "PASS", path
        pairs[f"h{history}_b{batch}"] = {
            "source": str(path),
            "summary": report["summary"],
            "runtime_shapes": report["runtime_shapes"],
            "cross_variant_exact": report["cross_variant_exact"],
            "graph_state_and_guards": "PASS",
            "padding_graph": report.get("padding_graph", {}).get("status"),
            "padding_active_batches": [p["active_batch"] for p in report.get("padding_graph", {}).get("replays", [])],
            "performance_evidence": history in (131072, 8192),
        }
    long_case, short_case = (pairs[name]["summary"] for name in ("h131072_b16", "h8192_b24"))
    weighted = sum(
        weight * (case["candidate"]["mean_us"] / case["base"]["mean_us"] - 1)
        for weight, case in ((0.8, long_case), (0.2, short_case))
    )
    service = json.loads((ROOT / "result_h131072_b16.json").read_text())
    assert all(p["state_pass"] for p in service["passes"])
    diagnostic_path = RESULTS / "pair_h16384_b2_base_padding/failure.json"
    diagnostic = json.loads(diagnostic_path.read_text())
    assert diagnostic["status"] == "PADDING_FAILED" and diagnostic["padding_variant"] == "base"
    report = {
        "source": json.loads((ROOT / "source.json").read_text()),
        "pairs": pairs,
        "weighted_pair_mean_change_pct": weighted * 100,
        "service_summary": service["summary"],
        "service_full_state": "PASS",
        "service_samples_and_all_incore_tasks": str(ROOT / "result_h131072_b16.json"),
        "empty_metadata_test_fix": {
            "baseline_failure": str(diagnostic_path),
            "failed_check": diagnostic["padding_graph"]["replays"][0]["compact_metadata"],
            "resolution": "No newly compressed token: compare complete invalid slots, skip unused cos/sin rows.",
            "candidate_after_fix": pairs["h16384_b2"]["padding_graph"],
        },
        "limits": [
            "Single-card layer-3 synthetic inputs/history with real weights; not whole-model/Native acceptance.",
            "Short path unchanged; retain observed drift without assigning it to the long-path change.",
            "Boundary cycles=1 only check correctness; they are not performance evidence.",
            "DFX worker times include waits; all linked task groups retained to avoid single-task attribution.",
        ],
    }
    (ROOT / "result_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"weighted_pair_mean_change_pct": weighted * 100, "service": service["summary"]}, indent=2))


if __name__ == "__main__":
    main()
