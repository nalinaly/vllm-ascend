"""只汇总五点的已有报告，保留算术诊断与正式精度检查之间的区别。"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    sys.path.insert(0, str(ROOT.parent))
    from hca_residual_reuse_20260930.collect import incore

    results = ROOT.parent / "results" / ROOT.name
    reports = {}
    for path in sorted(results.glob("*/report.json")):
        report = json.loads(path.read_text())
        if "cross_variant_exact" not in report:
            continue
        assert report["status"] == "MEASURED"
        assert all(all(side.values()) for side in report["state_exact"].values())
        assert all(value == "PASS" for value in report["guards"].values())
        exact = report["cross_variant_exact"]
        assert all(value for key, value in exact.items() if key != "x_out")
        if not exact["x_out"]:
            assert report["output_arithmetic_diagnostic"]["comparison"]["nonfinite"] == 0
        row = {key: report[key] for key in ("sources", "case", "device", "summary", "pairs", "cross_variant_exact")}
        row["source_report"] = str(path.resolve())
        row["same_variant_replay_and_guards"] = "PASS"
        row["purpose"] = "functional_only" if report["summary"]["base"]["count"] < 10 else "performance_screen"
        if "output_arithmetic_diagnostic" in report:
            row["arithmetic_diagnostic"] = report["output_arithmetic_diagnostic"]
        if "padding_graph" in report:
            assert report["padding_graph"]["status"] == "PASS"
            row["padding"] = {"status": "PASS", "active_batches": [
                replay["active_batch"] for replay in report["padding_graph"]["replays"]
            ]}
        base, candidate = (report["summary"][side]["mean_us"] for side in ("base", "candidate"))
        row["delta_mean_us"] = candidate - base
        row["delta_mean_percent"] = (candidate / base - 1) * 100
        row["improved_pair_means"] = sum(p["candidate_us"] < p["base_us"] for p in report["pairs"])
        reports[path.parent.name] = row
    dfx = {}
    for path in sorted((results / "dfx").glob("*/dfx/merged_swimlane.json")):
        dfx[path.parents[1].name] = incore(path)
    weighted = {}
    for point in range(1, 6):
        long_case = reports.get(f"p{point}_128k_b16")
        short_cases = [v for k, v in reports.items() if k.startswith(f"p{point}_8k_")]
        if long_case and len(short_cases) == 1:
            weighted[str(point)] = 0.8 * long_case["delta_mean_percent"] + 0.2 * short_cases[0]["delta_mean_percent"]
    for label, long_name, short_name in (
        ("point_2_on_new_schedule", "p2_shift_on_schedule_128k_b16", "p2_shift_on_schedule_8k_b16"),
        ("integrated_13", "integrated_128k_b16", "integrated_8k_b24"),
    ):
        if long_name in reports and short_name in reports:
            weighted[label] = (
                0.8 * reports[long_name]["delta_mean_percent"]
                + 0.2 * reports[short_name]["delta_mean_percent"]
            )
    summary = {
        "source": json.loads((ROOT / "source.json").read_text()),
        "reports": reports, "weighted_8_to_2_percent": weighted,
        "limits": (
            "各点只与自己的同窗控制比较；不同点不叠加百分比。DFX独立单次采样，"
            "核占用/24不是整段延迟；第二点不是同窗四方因子试验。无新Native/整机结论。"
        ),
    }
    (ROOT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    # 每个task的统计占一行，保留全部指标，避免数千行重复JSON框架干扰代码审查。
    incore_lines = []
    for name, record in dfx.items():
        task_lines = [f"      {json.dumps(key)}: {json.dumps(value)}" for key, value in record["tasks"].items()]
        incore_lines.append(
            f"  {json.dumps(name)}: {{\n    \"source\": {json.dumps(record['source'])},\n    \"tasks\": {{\n"
            + ",\n".join(task_lines) + "\n    }\n  }"
        )
    (ROOT / "incore.json").write_text("{\n" + ",\n".join(incore_lines) + "\n}\n")
    for name, row in reports.items():
        print(name, row["summary"], "improved", row["improved_pair_means"])


if __name__ == "__main__":
    main()
