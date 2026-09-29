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
# 用户 2026-09-29 要求：报最小／平均／最大三个，而不只是中位数。
# 极差（max-min）直接反映这张卡这一轮的噪声水平——本批卡实测可达 58 μs，
# 所以 Δ 必须和极差一起看，否则小于噪声的效应会被当成收益。
print(f"\n{'变体':>12s} {'样本':>3s} {'最小':>9s} {'平均':>9s} {'最大':>9s} {'极差':>8s}")
summary = {}
for label, spans in groups.items():
    summary[label] = statistics.mean(spans)
    print(f"{label:>12s} {len(spans):3d} {min(spans):9.2f} {statistics.mean(spans):9.2f} "
          f"{max(spans):9.2f} {max(spans) - min(spans):8.2f}")
for label, spans in groups.items():
    print(f"  {label:>12s} 样本 {[round(s, 1) for s in spans]}")
if "base" in summary:
    base_spans = groups["base"]
    base_range = max(base_spans) - min(base_spans)
    print("\n相对 base 的平均值之差（同卡 ABBA，正负号可信）:")
    for label, value in summary.items():
        if label == "base":
            continue
        delta = value - summary["base"]
        verdict = "低于 base 极差，不可判定" if abs(delta) < base_range else ""
        print(f"  {label:>12s} {delta:+8.2f} μs   (base 极差 {base_range:.2f}) {verdict}")
