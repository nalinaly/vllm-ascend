# SPDX-License-Identifier: Apache-2.0
"""汇总同卡 ABBA 对照：按变体聚合各 pass 的 span 中位数。

用法: python summarize_hca_ab.py <run_hca_ab_same_card.sh 的结果目录>
"""
import json
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for d in sorted(root.glob("p*_*"), key=lambda p: int(p.name.split("_")[0][1:])):
    label = d.name.split("_", 1)[1]
    report = d / "report.json"
    if not report.is_file():
        rows.append((d.name, label, None, "没有 report.json"))
        continue
    data = json.load(open(report))
    if data.get("error"):
        rows.append((d.name, label, None, data["error"][:70]))
        continue
    profile = data.get("device_profile") or {}
    checks = data.get("timing", {}).get("eager_comparison", {})
    ok = bool(checks) and all(c["status"] == "PASS" for c in checks.values())
    rows.append((d.name, label, profile.get("span_us"),
                 f"σ={profile.get('span_us_stdev', 0):.2f} 精度={'OK' if ok else 'FAIL'} "
                 f"device={data.get('device')}"))

print(f"{'pass':>10s} {'变体':>12s} {'span_us':>9s}  备注")
for name, label, span, note in rows:
    print(f"{name:>10s} {label:>12s} {span if span is None else f'{span:9.2f}'}  {note}")

groups = {}
for _, label, span, _ in rows:
    if span is not None:
        groups.setdefault(label, []).append(span)
print(f"\n{'变体':>12s} {'样本':>18s} {'中位数':>9s} {'均值':>9s}")
summary = {}
for label, spans in groups.items():
    summary[label] = statistics.median(spans)
    print(f"{label:>12s} {str([round(s, 1) for s in spans]):>18s} "
          f"{statistics.median(spans):9.2f} {statistics.mean(spans):9.2f}")
if "base" in summary:
    print("\n相对 base（同卡 ABBA，正负号可信）:")
    for label, value in summary.items():
        if label != "base":
            print(f"  {label:>12s} {value - summary['base']:+8.2f} μs")
