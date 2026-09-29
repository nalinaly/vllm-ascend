"""收集现有ABBA及独立DFX，不占用NPU。"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, default=ROOT)
    args = parser.parse_args()
    experiment = args.experiment_root.resolve()
    sys.path.insert(0, str(ROOT.parent))
    from hca_residual_reuse_20260930.collect import incore

    results = ROOT.parent / "results" / experiment.name
    reports = {}
    for path in sorted(results.glob("*/report.json")):
        report = json.loads(path.read_text())
        if "cross_variant_exact" not in report:
            continue
        assert report["status"] == "MEASURED"
        assert all(report["cross_variant_exact"].values())
        assert all(all(v.values()) for v in report["state_exact"].values())
        assert all(v == "PASS" for v in report["guards"].values())
        row = {k: report[k] for k in (
            "sources", "case", "device", "cann", "summary", "cross_variant_exact", "state_exact", "guards",
        )}
        row["source_report"] = str(path)
        row["nz_mode"] = report.get("nz_mode", 2)
        row["purpose"] = "functional_only" if report["summary"]["base"]["count"] < 10 else "performance_screen"
        if "padding_graph" in report:
            padding = report["padding_graph"]
            assert padding["status"] == "PASS"
            for replay in padding["replays"]:
                for group in ("state_comparison", "compact_metadata", "guards"):
                    assert all(v["status"] == "PASS" for v in replay[group].values())
            row["padding_graph"] = {
                "status": padding["status"], "scope": padding["scope"],
                "active_batch_sequence": [r["active_batch"] for r in padding["replays"]],
            }
        reports[path.parent.name] = row
    dfx = {}
    for path in sorted((results / "dfx").glob("*/dfx/merged_swimlane.json")):
        dfx[path.parents[1].name] = incore(path)
    summary = {
        "source": json.loads((experiment / "source.json").read_text()), "reports": reports,
        "limits": "每候选各自同进程ABBA；DFX独立采样，组跨度变化不等于整段收益；未新增Native或整机验收",
    }
    (experiment / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    if dfx:
        (experiment / "incore.json").write_text(json.dumps(dfx, indent=2) + "\n")
    print(json.dumps({k: v["summary"] for k, v in reports.items()}, indent=2))


if __name__ == "__main__":
    main()
