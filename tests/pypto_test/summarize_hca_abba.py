# SPDX-License-Identifier: Apache-2.0
"""汇总同卡 A→B→B→A 的全部样本，并保留 Native 同轮变化。"""

import argparse
import json
import math
import statistics
from pathlib import Path

from dsv4_csa_env import write_json


def summarize(directory):
    names = ("0_baseline", "1_candidate", "2_candidate", "3_baseline")
    reports = {name: json.loads((directory / name / "report.json").read_text()) for name in names}
    first = reports[names[0]]
    result = {"scope": "同卡 A→B→B→A，保留全部正式样本", "runs": [], "pooled": {}}
    samples = {kind: {engine: [] for engine in ("native", "pto")} for kind in ("baseline", "candidate")}
    for name, report in reports.items():
        if report["status"] != "MEASURED":
            raise ValueError(f"{name} 未完成：{report.get('error')}")
        for key in ("batch", "history", "seed", "checkpoint", "weight_nz_mode", "atomic_add",
                    "deterministic_level", "task_device"):
            if report[key] != first[key]:
                raise ValueError(f"{name} 的 {key} 与基线不一致")
        if name != names[0] and any(check["status"] != "PASS" for check in report["pto_reference"].values()):
            raise ValueError(f"{name} 输出或 allocation 精确对照失败")
        kind = name.split("_", 1)[1]
        timing = report["timing"]
        result["runs"].append({"run": name, "operator_source": report["operator_source"], **{
            engine: {key: value for key, value in timing[engine].items() if key.startswith("us_")}
            for engine in ("native", "pto")
        }})
        for engine in ("native", "pto"):
            samples[kind][engine].extend(timing[engine]["samples_us"])
    for kind, engines in samples.items():
        result["pooled"][kind] = {}
        for engine, values in engines.items():
            ordered = sorted(values)
            result["pooled"][kind][engine] = {
                "n": len(values), "us_p50": statistics.median(values), "us_mean": statistics.mean(values),
                "us_p95": ordered[math.ceil(0.95 * len(values)) - 1],
                "us_min": ordered[0], "us_max": ordered[-1],
            }
    result["change_pct"] = {engine: 100 * (
        result["pooled"]["candidate"][engine]["us_p50"] / result["pooled"]["baseline"][engine]["us_p50"] - 1
    ) for engine in ("native", "pto")}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    result = summarize(args.directory)
    write_json(args.directory / "summary.json", result)
    for engine in ("native", "pto"):
        before = result["pooled"]["baseline"][engine]
        after = result["pooled"]["candidate"][engine]
        print(f"{engine}: P50 {before['us_p50']:.2f}→{after['us_p50']:.2f} μs "
              f"({result['change_pct'][engine]:+.2f}%)，P95 {before['us_p95']:.2f}→{after['us_p95']:.2f} μs")


if __name__ == "__main__":
    main()
