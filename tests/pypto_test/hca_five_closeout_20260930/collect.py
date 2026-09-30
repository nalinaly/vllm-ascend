# SPDX-License-Identifier: Apache-2.0
"""只汇总已完成的收尾数据；区分同卡性能、PTO状态等价及模型验收。"""

import argparse
import json
import os
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT.parent / "results" / ROOT.name
PASSES = ("p1_native", "p2_before", "p3_final", "p4_native")


def stats(values):
    return {
        "count": len(values),
        "min_us": min(values),
        "max_us": max(values),
        "mean_us": statistics.mean(values),
        "p50_us": statistics.median(values),
        "p95_us": sorted(values)[int(0.95 * (len(values) - 1))],
    }


def check_report(report):
    assert report["status"] == "MEASURED", report.get("error")
    assert "dynamic=False" in report["compile_entry"]
    options = report["backend_options"]
    assert options["inplace_pass"] and options["static_kernel_compile"]
    native = report["side"] == "native"
    assert bool(report["super_kernel"]) == native
    compiler = report["compiler"]
    if native:
        assert compiler["static_compile_results"] and all(compiler["static_compile_results"])
        assert compiler["static_super_flags"] and all(compiler["static_super_flags"])
        assert compiler["installed_static_packages"] > 0
    else:
        assert compiler["pto_dispatch_calls"] > 0
    timing = report["timing"]
    assert all(v["status"] == "PASS" for v in timing["guards"].values())
    assert all(v["nonfinite"] == 0 for v in timing["eager_comparison"].values())
    if not native:
        assert all(v["status"] == "PASS" for v in timing["eager_comparison"].values())
    profile = report["device_profile"]
    assert profile["available"] and profile["replays"] == 30
    assert len(profile["span_us_all"]) == 30
    assert len(set(profile["kernels_per_replay"])) == 1


def link_artifact(source, target):
    if not target.exists():
        os.link(source, target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-states", action="store_true", help="首次收尾时逐项比较完整CPU快照")
    parser.add_argument("--bundle", action="store_true", help="汇聚完整七档JSON，不覆盖历史数据")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "source.json").read_text())
    rows = []
    all_done = True
    for history, batch in manifest["seven_cases"]:
        name = f"h{history}_b{batch}"
        directory = RESULTS / "seven" / name
        paths = {p: directory / p / "report.json" for p in PASSES}
        if not all(p.exists() for p in paths.values()):
            rows.append({"case": name, "status": "PENDING"})
            all_done = False
            continue
        reports = {p: json.loads(path.read_text()) for p, path in paths.items()}
        for report in reports.values():
            check_report(report)
            assert (report["history"], report["batch"]) == (history, batch)
            assert report["weight_nz_mode"] == 2
        assert len({r["device"] for r in reports.values()}) == 1
        assert len({r["cann"] for r in reports.values()}) == 1
        assert reports["p2_before"]["operator_source"] == manifest["sources"]["before_qb"]
        assert reports["p3_final"]["operator_source"] == manifest["sources"]["production"]
        record = {
            "case": name,
            "history": history,
            "batch": batch,
            "status": "MEASURED",
            "device": reports["p3_final"]["device"],
            "cann": reports["p3_final"]["cann"],
            "passes": {
                p: {
                    "report": str(paths[p]),
                    "samples_us": r["device_profile"]["span_us_all"],
                    "stats": stats(r["device_profile"]["span_us_all"]),
                    "compiler": r["compiler"],
                    "native_eager_differences": {
                        k: {field: v[field] for field in ("mismatches", "max_abs", "rmse")}
                        for k, v in r["timing"]["eager_comparison"].items()
                        if v["status"] != "PASS"
                    },
                }
                for p, r in reports.items()
            },
        }
        record["native"] = stats(
            [v for p in ("p1_native", "p4_native") for v in reports[p]["device_profile"]["span_us_all"]]
        )
        for label, p in (("before", "p2_before"), ("final", "p3_final")):
            record[label] = stats(reports[p]["device_profile"]["span_us_all"])
        record["final_vs_native_mean_percent"] = (record["final"]["mean_us"] / record["native"]["mean_us"] - 1) * 100
        record["final_vs_before_mean_percent"] = (record["final"]["mean_us"] / record["before"]["mean_us"] - 1) * 100
        check_path = directory / "state_equivalence.json"
        if args.check_states and not check_path.exists():
            import torch

            torch.set_num_threads(4)
            states = {
                p: torch.load(directory / p / "states.pt", map_location="cpu", weights_only=True)
                for p in ("p2_before", "p3_final")
            }
            left, right = states["p2_before"], states["p3_final"]
            assert left.keys() == right.keys()
            exact = {k: bool(torch.equal(v, right[k])) for k, v in left.items()}
            check_path.write_text(
                json.dumps(
                    {"exact": exact, "status": "PASS" if all(exact.values()) else "FAIL"}, ensure_ascii=False, indent=2
                )
                + "\n"
            )
            del states, left, right
        if check_path.exists():
            record["state_equivalence"] = json.loads(check_path.read_text())
            assert record["state_equivalence"]["status"] == "PASS", name
        else:
            all_done = False
        swimlane = directory / "swimlane_final/dfx/merged_swimlane.json"
        record["swimlane"] = str(swimlane) if swimlane.exists() else None
        if not swimlane.exists():
            all_done = False
        rows.append(record)
    report = {
        "status": "COMPLETE" if all_done else "PENDING",
        "source": manifest,
        "cases": rows,
        "scope": (
            "正式第3层真实权重、合成历史、同卡设备span；PTO完整状态只比较优化前后。不是整网性能或跨Native数值验收。"
        ),
        "native_policy": "CANN9.2/NZ2/npugraph_ex dynamic=False/inplace/static/SuperKernel开启；PTO关闭SuperKernel",
        "p95_estimator": "sorted(samples)[floor(0.95*(n-1))]；native60个样本，before/final各30个",
    }
    if all_done:
        weighted = {}
        for field in ("final_vs_before_mean_percent", "final_vs_native_mean_percent"):
            weighted[field] = 0.8 * statistics.mean(
                r[field] for r in rows if r["history"] == 131072
            ) + 0.2 * statistics.mean(r[field] for r in rows if r["history"] == 8192)
        report["weighted_long_short_8_to_2"] = weighted
    if args.bundle:
        assert all_done, "七档状态/性能/泳道未齐，不能发布完整下载包"
        bundle = RESULTS / "download_hca_7cases_29917c4e"
        bundle.mkdir(exist_ok=True)
        for index, row in enumerate(rows, 1):
            directory = RESULTS / "seven" / row["case"]
            label = f"{index:02d}_{row['history'] // 1024}K_B{row['batch']}"
            for suffix, p in (
                ("01_Native_Before_PyTorch", "p1_native"),
                ("02_PTO_BeforeQBDoubleBuffer_PyTorch", "p2_before"),
                ("03_PTO_Final_PyTorch", "p3_final"),
                ("04_Native_After_PyTorch", "p4_native"),
            ):
                hits = list((directory / p / "profile").glob("*/ASCEND_PROFILER_OUTPUT/trace_view.json"))
                assert len(hits) == 1, (row["case"], p, hits)
                link_artifact(hits[0], bundle / f"{label}_{suffix}.json")
            link_artifact(Path(row["swimlane"]), bundle / f"{label}_05_PTO_Swimlane_SingleHCA_SyntheticHistory.json")
        (bundle / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        report["download_directory"] = str(bundle)
    (ROOT / "seven_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(report["status"])
    for row in rows:
        print(row["case"], row["status"], row.get("native"), row.get("final"))


if __name__ == "__main__":
    main()
