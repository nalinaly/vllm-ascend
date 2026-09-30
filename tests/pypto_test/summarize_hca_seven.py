# SPDX-License-Identifier: Apache-2.0
"""汇总七档同卡对照；缺失或失败的用例不得计入性能达标。"""

import argparse
import json
from pathlib import Path


CASES = ((131072, 4), (131072, 8), (131072, 16),
         (8192, 16), (8192, 24), (8192, 32), (8192, 40))


def summarize(directory):
    rows = []
    for history, batch in CASES:
        name = f"h{history}_b{batch}"
        row = {"case": name, "status": "PENDING"}
        paths = {kind: directory / name / kind / "report.json" for kind in ("baseline", "candidate")}
        reports = {kind: json.loads(path.read_text()) for kind, path in paths.items() if path.is_file()}
        failed = {kind: report.get("error", "验证失败") for kind, report in reports.items()
                  if report.get("status") == "FAIL"}
        if failed:
            row.update(status="FAIL", errors=failed)
            rows.append(row)
            continue
        if len(reports) != len(paths):
            rows.append(row)
            continue
        base, candidate = reports["baseline"], reports["candidate"]
        try:
            if any(report["status"] != "MEASURED" for report in reports.values()):
                raise ValueError("基线或候选未完成验证和计时")
            if (candidate["history"], candidate["batch"]) != (history, batch):
                raise ValueError("报告档位与目录不一致")
            for key in ("batch", "history", "seed", "checkpoint", "weight_nz_mode", "atomic_add",
                        "deterministic_level", "task_device"):
                if base[key] != candidate[key]:
                    raise ValueError(f"两侧配置不一致：{key}")
            expected_reference = (paths["baseline"].parent / "states.pt").resolve()
            if Path(candidate["reference_state"]).resolve() != expected_reference:
                raise ValueError("候选未对照本轮同卡基线")
            checks = candidate["pto_reference"]
            if set(checks) != {"output", "swa", "compressed", "state"}:
                raise ValueError("输出或完整 allocation 检查缺失")
            if any(check["status"] != "PASS" for check in checks.values()):
                raise ValueError("输出或完整 allocation 对照失败")
            for kind, report in reports.items():
                row[kind] = {"report": str(paths[kind].relative_to(directory)),
                             "operator_source": report["operator_source"], "timing": {}}
                for engine in ("native", "pto"):
                    timing = report["timing"][engine]
                    row[kind]["timing"][engine] = {
                        key: timing[key] for key in ("us_p50", "us_mean", "us_p95", "us_min", "us_max")
                    }
                    row[kind]["timing"][engine]["samples"] = len(timing["samples_us"])
            old = base["timing"]["pto"]["us_p50"]
            new = candidate["timing"]["pto"]["us_p50"]
            native = candidate["timing"]["native"]["us_p50"]
            row.update(status="MEASURED", pto_change_pct=100 * (new / old - 1),
                       vs_native_pct=100 * (new / native - 1), target_met=new <= 0.8 * native)
        except (KeyError, ValueError, ZeroDivisionError) as exc:
            row.update(status="FAIL", error=str(exc))
        rows.append(row)
    complete = all(row["status"] == "MEASURED" for row in rows)
    return {"scope": "正式权重、合成历史的七档单卡 HCA 整层；不是整模型 token 验收",
            "status": "FAIL" if any(row["status"] == "FAIL" for row in rows) else
                      "MEASURED" if complete else "PENDING",
            "criterion": "每档候选 PTO P50 <= 本轮同卡 Native P50 × 0.80",
            "all_targets_met": all(row["target_met"] for row in rows) if complete else None,
            "cases": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    result = summarize(args.directory)
    (args.directory / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    lines = ["# HCA 七档单卡优化对照", "", result["scope"], "", result["criterion"], "",
             "| 档位 | 状态 | 旧 PTO P50 μs | 候选 PTO P50 μs | Native P50 μs | PTO 变化 | 相对 Native |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in result["cases"]:
        if row["status"] != "MEASURED":
            lines.append(f"| {row['case']} | {row['status']} | — | — | — | — | — |")
            continue
        old = row["baseline"]["timing"]["pto"]["us_p50"]
        new = row["candidate"]["timing"]["pto"]["us_p50"]
        native = row["candidate"]["timing"]["native"]["us_p50"]
        lines.append(f"| [{row['case']}]({row['candidate']['report']}) | 已测量 | {old:.2f} | {new:.2f} | "
                     f"{native:.2f} | {row['pto_change_pct']:+.2f}% | {row['vs_native_pct']:+.2f}% |")
    lines += ["", "百分比为延迟变化，负值表示更快。P95、最大值和来源路径保存在 summary.json；",
              "所有原始样本保存在各轮 report.json。两个版本的顺序为基线→候选，不能替代同卡 ABBA 归因。", ""]
    (args.directory / "SUMMARY.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
