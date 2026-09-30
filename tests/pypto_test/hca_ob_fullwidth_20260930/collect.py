"""分别汇总相同算术功能验证、跨算术性能诊断和独立DFX。"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))


def main():
    from hca_residual_reuse_20260930.collect import incore

    result = ROOT.parent / "results" / ROOT.name
    rows = {}
    for path in sorted(result.glob("*/report.json")):
        report = json.loads(path.read_text())
        if "cross_variant_exact" not in report:
            continue
        assert report["status"] == "MEASURED"
        assert all(all(v.values()) for v in report["state_exact"].values())
        assert all(v == "PASS" for v in report["guards"].values())
        diag = report.get("output_arithmetic_diagnostic")
        if diag:
            assert diag["comparison"]["nonfinite"] == 0
            assert all(v for k, v in report["cross_variant_exact"].items() if k != "x_out")
        else:
            assert all(report["cross_variant_exact"].values())
        row = {k: report[k] for k in (
            "sources", "case", "device", "cann", "summary", "cross_variant_exact", "state_exact", "guards",
        )}
        row["source_report"] = str(path)
        row["purpose"] = "arithmetic_performance_diagnostic" if diag else "same_arithmetic_functional"
        if not diag and report["summary"]["base"]["count"] >= 10:
            row["purpose"] = "same_arithmetic_performance"
        if diag:
            row["output_arithmetic_diagnostic"] = diag
        rows[path.parent.name] = row
    dfx = {
        path.parents[1].name: incore(path)
        for path in sorted((result / "dfx").glob("*/dfx/merged_swimlane.json"))
    }
    (ROOT / "summary.json").write_text(json.dumps({
        "reports": rows,
        "limits": "完整K和分组参考共享整宽量化；不代表Native量化逐元素或整模型token验收，未接入生产",
    }, ensure_ascii=False, indent=2) + "\n")
    if dfx:
        (ROOT / "incore.json").write_text(json.dumps(dfx, indent=2) + "\n")
    print(json.dumps({k: v["summary"] for k, v in rows.items()}, indent=2))


if __name__ == "__main__":
    main()
