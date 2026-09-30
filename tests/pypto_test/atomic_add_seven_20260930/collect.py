# SPDX-License-Identifier: Apache-2.0
"""汇总同卡 atomic0/1 的原始样本与输出差异，不混用历史结果。"""

import json
import statistics
import sys
from pathlib import Path


def main():
    experiment = Path(__file__).resolve().parent
    root = experiment.parent / "results" / experiment.name
    sys.path.insert(0, str(experiment.parent))
    import torch
    from dsv4_csa_validation import compare_tensor, compare_topk

    cases = [(131072, b) for b in (4, 8, 16, 24)] + [(8192, b) for b in (16, 24, 32)]
    results = []
    for history, batch in cases:
        directory = root / f"h{history}_b{batch}"
        reports = [json.loads((directory / f"atomic{a}/report.json").read_text()) for a in (0, 1)]
        for report in reports:
            assert report["status"] == "MEASURED", (history, batch, report)
        for field in ("task_device", "versions", "history", "batch", "source", "checkpoint",
                      "weight_nz_mode", "deterministic_level", "runtime", "warmup", "iterations"):
            assert reports[0][field] == reports[1][field], (history, batch, field)
        outputs = [torch.load(directory / f"atomic{a}/outputs.pt", weights_only=True, map_location="cpu")
                   for a in (0, 1)]
        difference = {name: compare_tensor(outputs[1][name], outputs[0][name], 0, 0)
                      for name in ("CSA", "HCA", "CSA_HCA")}
        visible = ((history + torch.arange(6) + 1) // 4).repeat(batch).to(torch.int32)
        difference["topk"] = compare_topk(outputs[1]["idx_topk"], outputs[0]["idx_topk"], visible)
        summaries = [{
            "report": str(directory / f"atomic{a}/report.json"),
            "status": report["status"], "timing_us": report["timing_us"],
            "nonfinite": report["nonfinite"], "replay_vs_eager": report["replay_vs_eager"],
            "guards_pass": all(check["status"] == "PASS" for group in report["guards"].values()
                               for check in group.values()),
        } for a, report in enumerate(reports)]
        row = {"history": history, "batch": batch, "device": reports[0]["task_device"],
               "atomic0": summaries[0], "atomic1": summaries[1], "output_difference": difference,
               "mean_change_percent": {name: (reports[1]["timing_us"][name]["mean"] /
                                               reports[0]["timing_us"][name]["mean"] - 1) * 100
                                       for name in ("CSA", "HCA", "CSA_HCA")}}
        results.append(row)
    weighted = {name: 0.8 * statistics.mean(r["mean_change_percent"][name] for r in results[:4]) +
                     0.2 * statistics.mean(r["mean_change_percent"][name] for r in results[4:])
                for name in ("CSA", "HCA", "CSA_HCA")}
    evidence = {"source": json.loads((experiment / "source.json").read_text()),
                "versions": reports[0]["versions"],
                "weighted_mean_change_percent_long_short_8_2": weighted, "cases": results}
    (experiment / "evidence.json").write_text(json.dumps(evidence, indent=2, ensure_ascii=False) + "\n")
    lines = ["# TMR atomic add 七档实测", "", "单位 μs；变化=(开启/关闭−1)×100%，负值表示开启更快。",
             "独立CSA/HCA使用各自固定输入；联合图为CSA输出直接接HCA。不包含MoE或整模型。", ""]
    for name in ("CSA", "HCA", "CSA_HCA"):
        lines += [f"## {name}", "",
                  "| 档位 | 关闭 min | 关闭 mean | 关闭 max | 开启 min | 开启 mean | 开启 max | 开启耗时变化 |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for row in results:
            values = [row[f"atomic{a}"]["timing_us"][name][metric]
                      for a in (0, 1) for metric in ("min", "mean", "max")]
            label = f"{row['history'] // 1024}K/B{row['batch']}"
            lines.append(f"| {label} | " + " | ".join(f"{v:.2f}" for v in values) +
                         f" | {row['mean_change_percent'][name]:+.2f}% |")
        lines += ["", f"长短档8:2加权的均值耗时变化：{weighted[name]:+.2f}%。", ""]
    lines += ["## 输出变化", "",
              "这里报告开启相对关闭的数值变化，不把零容差差异自动认定为不可接受，也不代表token/DSpark验收。", "",
              "| 档位 | CSA max_abs | HCA max_abs | 联合 max_abs | Top-K不同集合行数 |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for row in results:
        diff = row["output_difference"]
        lines.append(f"| {row['history'] // 1024}K/B{row['batch']} | " +
                     " | ".join(f"{diff[name]['max_abs']:.6g}" for name in ("CSA", "HCA", "CSA_HCA")) +
                     f" | {diff['topk']['different_set_rows']} |")
    lines += ["", "完整20次样本、P95、图/eager差异、保护区检查及原始路径见 [evidence.json](evidence.json)。", ""]
    (experiment / "RESULTS.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
