"""保留反转A/B的真实标签及全部样本，不以先测顺序替代候选身份。"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    result = ROOT.parent / "results" / ROOT.name
    rows = {}
    for path in sorted(result.glob("*/report.json")):
        report = json.loads(path.read_text())
        assert report["status"] == "MEASURED"
        assert all(all(v.values()) for v in report["state_exact"].values())
        assert all(v == "PASS" for v in report["guards"].values())
        assert all(v for k, v in report["cross_variant_exact"].items() if k != "x_out")
        if "output_arithmetic_diagnostic" in report:
            assert report["output_arithmetic_diagnostic"]["comparison"]["nonfinite"] == 0
        else:
            assert report["cross_variant_exact"]["x_out"]
        row = {k: report[k] for k in (
            "sources", "case", "device", "cann", "summary", "cross_variant_exact", "state_exact", "guards",
            "order", "pairs",
        )}
        row["span_us_all"] = report["profile"]["span_us_all"]
        row["source_report"] = str(path)
        row["implementation_labels"] = {key: Path(value).name for key, value in report["sources"].items()}
        if "output_arithmetic_diagnostic" in report:
            row["output_arithmetic_diagnostic"] = report["output_arithmetic_diagnostic"]
        rows[path.parent.name] = row
    (ROOT / "summary.json").write_text(json.dumps({
        "reports": rows,
        "limits": "long_reverse的base是online；不合并跨卡绝对值，不事后删除慢点；未通过模型token验收",
    }, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({name: value["summary"] for name, value in rows.items()}, indent=2))


if __name__ == "__main__":
    main()
