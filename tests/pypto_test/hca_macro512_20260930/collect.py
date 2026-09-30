"""汇总512列候选的算术诊断、完整层计时和独立泳道，不把诊断当验收。"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from hca_residual_reuse_20260930.collect import incore  # noqa: E402


def main():
    results = ROOT.parent / "results" / ROOT.name
    reports = {}
    for path in sorted(results.glob("*/report.json")):
        report = json.loads(path.read_text())
        assert report["status"] == "MEASURED"
        diagnostic = report.get("output_arithmetic_diagnostic")
        if diagnostic:
            assert diagnostic["comparison"]["nonfinite"] == 0
        assert all(v for k, v in report["cross_variant_exact"].items() if k != "x_out" or not diagnostic)
        assert all(all(v.values()) for v in report["state_exact"].values())
        assert all(v == "PASS" for v in report["guards"].values())
        row = {k: report[k] for k in (
            "sources", "case", "device", "cann", "summary", "pairs", "order",
            "cross_variant_exact", "state_exact", "guards",
        )}
        if diagnostic:
            row["output_arithmetic_diagnostic"] = diagnostic
        row["source_report"] = str(path)
        row["span_samples_us"] = report["profile"]["span_us_all"]
        reports[path.parent.name] = row
    summary = {
        "source": json.loads((ROOT / "source.json").read_text()),
        "reports": reports,
        "probe": json.loads((ROOT / "probe_result.json").read_text()),
        "limits": "独立数值诊断和同卡性能筛选；没有新增Native、七档或整模型token验收",
    }
    (ROOT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    dfx = {
        p.parents[1].name: incore(p)
        for p in sorted((results / "dfx").glob("*/dfx/merged_swimlane.json"))
    }
    if dfx:
        (ROOT / "incore.json").write_text(json.dumps(dfx, indent=2) + "\n")
    print(json.dumps({k: v["summary"] for k, v in reports.items()}, indent=2))


if __name__ == "__main__":
    main()
