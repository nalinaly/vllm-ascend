# SPDX-License-Identifier: Apache-2.0
"""复用完整的Native SK0三步task明细和最终PTO图；不重写SK1性能基线。"""

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TESTS = ROOT.parent
SOURCE = TESTS / "hca_five_closeout_20260930" / "source.json"
RESULTS = TESTS / "results" / ROOT.name
NATIVE = TESTS / "results" / "hca_native_sk0_20260929"
BUNDLE = RESULTS / "download_hca_7cases_sk0_task_details_29917c4e"


def link(source, target):
    if not target.exists():
        os.link(source, target)


def main():
    BUNDLE.mkdir(parents=True, exist_ok=True)
    source = json.loads(SOURCE.read_text())
    old = TESTS / "results" / "hca_five_closeout_20260930" / "seven"
    rows = []
    for index, (history, batch) in enumerate(source["seven_cases"], 1):
        case = f"h{history}_b{batch}"
        prefix = f"{index:02d}_{history // 1024}K_B{batch}"
        row = {"history": history, "batch": batch, "case": case, "status": "PENDING"}
        pto = list((old / case / "p3_final/profile").glob("*/ASCEND_PROFILER_OUTPUT/trace_view.json"))
        assert len(pto) == 1, case
        row["pto_profile"] = f"{prefix}_02_PTO_Final_PyTorch_30Steps.json"
        row["pto_swimlane"] = f"{prefix}_03_PTO_Swimlane_SingleHCA_SyntheticHistory.json"
        link(pto[0], BUNDLE / row["pto_profile"])
        link(old / case / "swimlane_final/dfx/merged_swimlane.json", BUNDLE / row["pto_swimlane"])
        report_path = NATIVE / case / "report.json"
        if report_path.exists():
            report = json.loads(report_path.read_text())
            pto_report = json.loads((old / case / "p3_final/report.json").read_text())
            assert report["status"] == "MEASURED", (case, report.get("error"))
            assert report["side"] == "native" and report["super_kernel"] == 0
            assert (report["history"], report["batch"], report["weight_nz_mode"]) == (history, batch, 2)
            assert report["cann"] == pto_report["cann"]
            assert "dynamic=False" in report["compile_entry"]
            options = report["backend_options"]
            assert options["inplace_pass"] and options["static_kernel_compile"]
            compiler = report["compiler"]
            assert compiler["static_compile_results"] and all(compiler["static_compile_results"])
            assert compiler["static_super_flags"] and not any(compiler["static_super_flags"])
            assert compiler["installed_static_packages"] > 0
            profile = report["device_profile"]
            assert profile["available"] and profile["replays"] == 3
            assert len(profile["span_us_all"]) == 3
            assert len(profile["kernels_per_replay"]) == 3 and min(profile["kernels_per_replay"]) > 1
            assert len(set(profile["kernels_per_replay"])) == 1
            assert all(v["status"] == "PASS" for v in report["timing"]["guards"].values())
            assert all(v["nonfinite"] == 0 for v in report["timing"]["eager_comparison"].values())
            files = list((NATIVE / case / "profile").glob("*/ASCEND_PROFILER_OUTPUT/trace_view.json"))
            assert len(files) == 1, case
            row.update(
                status="COMPLETE",
                native_profile=f"{prefix}_01_Native_SK0_Static1_PyTorch_3Steps.json",
                native_report=str(report_path.resolve()),
                device=report["device"],
                cann=report["cann"],
                compiler=compiler,
                kernels_per_replay=profile["kernels_per_replay"],
                expanded_kernel_names=[entry["name"] for entry in profile["top"]],
            )
            link(files[0], BUNDLE / row["native_profile"])
        rows.append(row)
    report = {
        "status": "COMPLETE" if all(r["status"] == "COMPLETE" for r in rows) else "PENDING",
        "native_capture_date": "2026-09-29",
        "native_source_commit": None,
        "native_provenance_note": "原report未记录Git提交；按原source路径与实际编译配置留证，不推断commit。",
        "pto_source_commit": "29917c4e",
        "pto_capture_date": "2026-09-30",
        "scope": "复用9月29日Native SK0/static1的3步task细节、9月30日最终PTO的30步profiling和单次泳道。",
        "limits": "用于查看内部task，不替代Native SK1正式性能对照；两侧采集窗口不同。无16卡任务。",
        "download_directory": str(BUNDLE.resolve()),
        "cases": rows,
    }
    payload = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    (ROOT / "summary.json").write_text(payload)
    (BUNDLE / "manifest.json").write_text(payload)
    lines = [
        "# HCA七档task细节：Native SK0/static1，最终PTO 29917c4e",
        "",
        f"当前状态：{report['status']}。",
        "",
        "- 01 Native：CANN9.2/NZ2/npugraph_ex dynamic=False/inplace与static开，SuperKernel关。",
        "  复用2026-09-29已完成的预热后3个step，可展开kernel/task细节。本次未重新采集。",
        "- 02 PTO：复用29917c4e已完成的30-step PyTorch profiling，没有重复采集。",
        "- 03 PTO：复用同版本独立单次HCA泳道，SyntheticHistory。",
        "",
        "第3层真实权重，HC_pre+norm+HCA+HC_post。两侧时间窗不同，",
        "本包不替代Native SK1正式性能表，也不是CSA或整网forward数据。",
        "原排队的7个重复采集任务已在启动前取消，无需等待队列恢复。",
        "",
        "| 档位 | Native SK0 | PTO profiling/泳道 |",
        "| --- | --- | --- |",
    ]
    for row in rows:
        lines.append(f"| {row['history'] // 1024}K/B{row['batch']} | {row['status']} | 已齐 |")
    (BUNDLE / "README.md").write_text("\n".join(lines) + "\n")
    print(report["status"], report["download_directory"])
    for row in rows:
        print(row["case"], row["status"], row.get("kernels_per_replay"))


if __name__ == "__main__":
    main()
