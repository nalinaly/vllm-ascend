# SPDX-License-Identifier: Apache-2.0
"""离线汇总16个rank的target计时；不读取设备、不修改有效样本。"""

import argparse
import json
import math
import statistics
from pathlib import Path


def summarize(directory):
    rows = []
    for rank in range(16):
        path = directory / f"rank{rank}.performance.json"
        record = json.loads(path.read_text())
        window, = record["steady_window"]
        samples = window["forward"]["samples_us"]
        if not window["sufficient"] or len(samples) != 10:
            raise ValueError(f"{path}: 需要10个有效满批target样本，不能删长尾后汇总")
        if window["step_tokens"] != [96] * 10 or window["step_requests"] != [16] * 10:
            raise ValueError(f"{path}: 样本不满足B16/T96")
        rows.append({
            "rank": rank,
            "min_ms": min(samples) / 1000,
            "mean_ms": statistics.mean(samples) / 1000,
            "max_ms": max(samples) / 1000,
            "samples_ms": [value / 1000 for value in samples],
            "source": str(path),
            "stage": record["stage"],
            "request_keys": record["request_keys"],
            "request_histories": record["request_histories"],
            "spec_decode": record.get("spec_decode"),
        })
    slowest = max(rows, key=lambda row: row["mean_ms"])
    pooled = sorted(value for row in rows for value in row["samples_ms"])
    return {
        "ranks": rows,
        "slowest_rank": slowest["rank"],
        "slowest_rank_mean_ms": slowest["mean_ms"],
        "pooled_mean_ms": statistics.mean(pooled),
        "pooled_p95_ms": pooled[math.ceil(len(pooled) * 0.95) - 1],
        "pooled_max_ms": max(pooled),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--pto", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    native = summarize(args.native)
    report = {"native": native}
    lines = [
        "# 128K/B16 target网络计时",
        "",
        "每rank预热后10个满批T96 target `_model_forward`，单位ms。",
        "128K公共前缀加固定问题后缀，准确长度保留在JSON；不含draft和接受拒绝。",
        "不剔除长尾，不把各rank第N个样本假定为同一个全局step。",
        "",
    ]
    if args.pto:
        pto = summarize(args.pto)
        report["pto"] = pto
        for left, right in zip(native["ranks"], pto["ranks"]):
            for key in ("request_keys", "request_histories"):
                if left[key] != right[key]:
                    raise ValueError(f"rank{left['rank']}: 两侧{key}不同，不能作同输入对照")
        lines.extend([
            "| rank | Native min | Native mean | Native max | PTO min | PTO mean | PTO max | mean降幅 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ])
        for left, right in zip(native["ranks"], pto["ranks"]):
            gain = 100 * (1 - right["mean_ms"] / left["mean_ms"])
            values = [left[k] for k in ("min_ms", "mean_ms", "max_ms")]
            values += [right[k] for k in ("min_ms", "mean_ms", "max_ms")]
            lines.append(f"| {left['rank']} | " + " | ".join(f"{v:.3f}" for v in values) + f" | {gain:+.2f}% |")
        report["timing_threshold_met"] = pto["slowest_rank_mean_ms"] <= 48
    else:
        lines.extend([
            "| rank | Native min | Native mean | Native max |",
            "| --- | ---: | ---: | ---: |",
        ])
        for row in native["ranks"]:
            values = [row[k] for k in ("min_ms", "mean_ms", "max_ms")]
            lines.append(f"| {row['rank']} | " + " | ".join(f"{v:.3f}" for v in values) + " |")
    for side in ("native", "pto"):
        if side in report:
            item = report[side]
            lines.extend(["", f"{side}: 最慢rank {item['slowest_rank']}，mean "
                          f"{item['slowest_rank_mean_ms']:.3f} ms；全样本P95 "
                          f"{item['pooled_p95_ms']:.3f} ms，max {item['pooled_max_ms']:.3f} ms。"])
    lines.extend(["", "这里只判定计时结果。真实SK覆盖、metadata消除与正常replay需结合执行记录，不能由耗时推断。"])
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "target_timing.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    (args.output / "target_timing.md").write_text("\n".join(lines) + "\n")
    print(args.output / "target_timing.md")


if __name__ == "__main__":
    main()
