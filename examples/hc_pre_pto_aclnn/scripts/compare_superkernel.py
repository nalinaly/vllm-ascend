# SPDX-License-Identifier: Apache-2.0
"""Compare two completed runs using device kernel records, without another NPU run."""

import argparse
import csv
import json
import statistics
from decimal import Decimal
from pathlib import Path


def summarize(values, calls):
    assert values and len(values) % calls == 0, "Incomplete profiler samples"
    batches = [statistics.mean(values[i : i + calls]) for i in range(0, len(values), calls)]
    return {
        "min_us": min(batches),
        "mean_us": statistics.mean(batches),
        "max_us": max(batches),
        "samples": len(batches),
        "calls_per_sample": calls,
        "samples_us": batches,
    }


def read_run(path, fused):
    report = json.loads((path / "result/report.json").read_text())
    assert report["numerical_passed"], report
    assert all(r["inputs_preserved"] and r["guards_preserved"] for r in report["validation"])
    if fused:
        assert report["fusion"]["passed"], report
    profiles = list((path / "result/profile").rglob("kernel_details.csv"))
    assert len(profiles) == 1, profiles
    with profiles[0].open() as stream:
        rows = list(csv.DictReader(stream))
    names = ("HcPreAdd", "HcPrePtoBnn", "HcPrePto", "HcPreMul")  # Also read historical reports.
    kernels = [r for r in rows if r["Name"].startswith("sk_0_hcpre_add_mul") or r["Name"] in names]
    per_call = 1 if fused else 3
    calls = report["timing"]["calls_per_sample"]
    count = report["timing"]["samples"] * calls
    assert len(kernels) == count * per_call, f"Expected {count * per_call} kernels, got {len(kernels)}"
    spans, sums = [], []
    for i in range(0, len(kernels), per_call):
        chain = kernels[i : i + per_call]
        if not fused:
            assert tuple(r["Name"] for r in chain) == names, chain
        start = Decimal(chain[0]["Start Time(us)"].strip())
        end = Decimal(chain[-1]["Start Time(us)"].strip()) + Decimal(chain[-1]["Duration(us)"])
        spans.append(float(end - start))
        sums.append(sum(float(r["Duration(us)"]) for r in chain))
    return {
        "report": str(path / "result/report.json"),
        "profiler": str(profiles[0]),
        "device_scope_span": summarize(spans, calls),
        "device_kernel_sum": summarize(sums, calls),
        "compiled_call_event_window": report["timing"],
    }, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("graph", type=Path)
    parser.add_argument("superkernel", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    graph, graph_report = read_run(args.graph.resolve(), False)
    fused, fused_report = read_run(args.superkernel.resolve(), True)
    assert graph_report["tokens"] == fused_report["tokens"] == 96
    assert graph_report["neighbours"] == fused_report["neighbours"] == "ascendc"
    assert graph_report["torch"] == fused_report["torch"]
    assert graph_report["torch_npu"] == fused_report["torch_npu"]
    assert graph_report["device"] == fused_report["device"]
    for key in ("static_kernel_compile", "clone_input"):
        assert graph_report["options"][key] == fused_report["options"][key]
    delta = {
        key: (fused[key]["mean_us"] / graph[key]["mean_us"] - 1) * 100
        for key in ("device_scope_span", "device_kernel_sum", "compiled_call_event_window")
    }
    result = {
        "graph": graph,
        "superkernel": fused,
        "change_percent": delta,
        "within_30_percent": all(abs(v) <= 30 for v in delta.values()),
        "measurement": "30 samples of 8 calls; profiler device scope excludes framework ForeachCopy; "
        "event windows include framework work and submission gaps. "
        "Profiler and event timing are separate capture phases.",
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"change_percent": delta, "within_30_percent": result["within_30_percent"]}, indent=2))
    assert result["within_30_percent"], result


if __name__ == "__main__":
    main()
