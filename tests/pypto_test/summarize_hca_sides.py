# SPDX-License-Identifier: Apache-2.0
"""汇总同卡 Native vs PTO 验收对照，给出可信比值。

用法: python summarize_hca_sides.py <结果目录> [<结果目录> ...]
"""
import json
import statistics
import sys
from pathlib import Path


def one(root):
    spans = {"native": [], "pto": []}
    meta = {}
    for d in sorted(root.glob("p*_*"), key=lambda p: int(p.name.split("_")[0][1:])):
        side = d.name.split("_", 1)[1]
        report = d / "report.json"
        if not report.is_file():
            continue
        data = json.load(open(report))
        if data.get("error"):
            print(f"  {d.name}: 失败 {data['error'][:70]}")
            continue
        profile = data.get("device_profile") or {}
        if profile.get("span_us") is None:
            continue
        spans[side].append(profile["span_us"])
        checks = data.get("timing", {}).get("eager_comparison", {})
        meta.setdefault("device", data.get("device"))
        meta.setdefault("nz", data.get("weight_nz_mode"))
        # 只有 PTO 要求与同初态 eager 逐 bit 一致（Native 走另一套 kernel + SuperKernel，
        # case 脚本的 require_exact 也只对 PTO 生效），所以别拿 Native 的结果报警。
        if side == "pto" and checks and not all(c["status"] == "PASS" for c in checks.values()):
            print(f"  ⚠ {d.name} 逐 bit 对照 FAIL")
    if not spans["native"] or not spans["pto"]:
        return None
    native = statistics.median(spans["native"])
    pto = statistics.median(spans["pto"])
    return {"native": native, "pto": pto, "ratio": pto / native,
            "n_native": len(spans["native"]), "n_pto": len(spans["pto"]),
            "samples_native": spans["native"], "samples_pto": spans["pto"], **meta}


print(f"{'目录':>22s} {'卡':>3s} {'nz':>3s} {'Native':>8s} {'PTO':>8s} {'比值':>7s}  样本")
rows = []
for arg in sys.argv[1:]:
    root = Path(arg)
    result = one(root)
    if result is None:
        print(f"{root.name:>22s}  数据不全")
        continue
    rows.append((root.name, result))
    print(f"{root.name:>22s} {str(result.get('device')):>3s} {str(result.get('nz')):>3s} "
          f"{result['native']:8.2f} {result['pto']:8.2f} {result['ratio']:7.3f}  "
          f"N{[round(s) for s in result['samples_native']]} P{[round(s) for s in result['samples_pto']]}")
if rows:
        # 128K:8K = 8:2 加权（见记忆 hca-weight-128k-over-8k-8to2）
    long_t = [r[1]["ratio"] for r in rows if "131072" in r[0]]
    short_t = [r[1]["ratio"] for r in rows if "8192" in r[0]]
    print(f"\n比值均值 {statistics.mean(r[1]['ratio'] for r in rows):.3f}")
    if long_t and short_t:
        weighted = 0.8 * statistics.mean(long_t) + 0.2 * statistics.mean(short_t)
        print(f"128K 均值 {statistics.mean(long_t):.3f}  8K 均值 {statistics.mean(short_t):.3f}  "
              f"8:2 加权 {weighted:.3f}")
