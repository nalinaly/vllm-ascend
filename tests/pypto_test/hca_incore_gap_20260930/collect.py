"""读取已有Native PMU和当前生产PTO泳道，按功能及核数量比较占用。"""

import csv
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TESTS = ROOT.parent
NATIVE_ROOT = TESTS / "results/hca_native_sk0_20260929/h131072_b16"
PTO_WINDOWS = (
    "hca_ob_fullwidth_20260930",
    "hca_qgroups_sync_20260930",
    "hca_online_softmax_20260930",
    "hca_qgroups_workers_20260930",
)
# 同名MatMul/QuantMatmul必须按形状拆分；RoPE另按每次重放的顺序区分Q/输出。
STAGES = {
    "O-A": (18, "TransposeBatchMatMul", "96,8,4096;", "proj_a_mm"),
    "Q-B": (6, "QuantBatchMatmulV3", "96,1024;", "hca_qb_stream"),
    "Q-A": (2, "MatMulV2", "96,4096;256,64,16,16", "qr_proj_matmul"),
    "O-B": (20, "QuantBatchMatmulV3", "96,8192;", "_proj_b_mm"),
    "Attention": (14, "SparseAttnSharedkv", "96,64,512;", "hca_unified_attention_aic"),
}


def stats(values):
    return {"count": len(values), "min": min(values), "max": max(values), "mean": statistics.mean(values)}


def main():
    native_report = json.loads((NATIVE_ROOT / "report.json").read_text())
    native_csv = next((NATIVE_ROOT / "profile").glob("*/ASCEND_PROFILER_OUTPUT/kernel_details.csv"))
    with native_csv.open() as stream:
        native_rows = sorted(csv.DictReader(stream), key=lambda r: float(r["Start Time(us)"]))
    assert native_report["device_profile"]["kernels_per_replay"] == [24, 24, 24]
    assert len(native_rows) == 72
    assert native_report["compiler"]["static_compile_results"] == [True]
    assert native_report["compiler"]["static_super_flags"] == [False]
    windows = {}
    for name in PTO_WINDOWS:
        incore = json.loads((TESTS / name / "incore.json").read_text())["base"]
        report_path = TESTS / "results" / name / "dfx/base/report.json"
        report = json.loads(report_path.read_text())
        assert report["weight_nz_mode"] == 2 and not report["super_kernel"]
        assert (report["history"], report["batch"]) == (131072, 16)
        assert report["cann"] == native_report["cann"]
        windows[name] = {"incore": incore, "report": str(report_path), "device": report["device"],
                         "operator_source": report["operator_source"]}
    comparison = {}
    for stage, (index, op_type, shape, prefix) in STAGES.items():
        selected_native = native_rows[index::24]
        assert all(r["Type"] == op_type and r["Input Shapes"].strip('"').startswith(shape)
                   for r in selected_native)
        native_values = [float(r["aicore_time(us)"]) * int(r["Block Num"]) / 24 for r in selected_native]
        pto_values = []
        pto_details = []
        for name, window in windows.items():
            tasks = [v for k, v in window["incore"]["tasks"].items() if k.startswith(prefix)]
            count = sum(t["count"] for t in tasks)
            work = sum(t["count"] * t["mean_us"] for t in tasks)
            pto_values.append(work / 24)
            pto_details.append({"window": name, "blocks": count, "core_us": work,
                                "worker_min_us": min(t["min_us"] for t in tasks),
                                "worker_max_us": max(t["max_us"] for t in tasks),
                                "worker_mean_us": work / count,
                                "group_envelope_us": max(t["last_end_us"] for t in tasks)
                                - min(t["first_start_us"] for t in tasks)})
        comparison[stage] = {
            "native_blocks": int(selected_native[0]["Block Num"]),
            "native_equivalent_24core_us": stats(native_values),
            "pto_equivalent_24core_us": stats(pto_values),
            "difference_mean_us": statistics.mean(pto_values) - statistics.mean(native_values),
            "native_raw": selected_native,
            "pto_windows": pto_details,
        }
    result = {
        "case": [131072, 16], "current_production_operator_commit": "0742f07c",
        "native_report": str(NATIVE_ROOT / "report.json"), "native_csv": str(native_csv),
        "native_device": native_report["device"],
        "pto_sources": {k: {a: b for a, b in v.items() if a != "incore"} for k, v in windows.items()},
        "definition": (
            "每次捕获的总AIC核占用/24，μs；Native用PMU单核均值×Block Num，"
            "PTO用全部worker kernel-duration-us之和"
        ),
        "limits": (
            "Native 9月29日3次重放，PTO 9月30日相同生产版本4个独立DFX窗口；"
            "异卡异窗口与不同采集机制，只用于差距定位；不是整层延迟、纯MAC时间、可回收时间或配对优化收益"
        ),
        "comparison": comparison,
    }
    (ROOT / "comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    for stage, row in comparison.items():
        print(stage, row["native_equivalent_24core_us"], row["pto_equivalent_24core_us"],
              row["difference_mean_us"])


if __name__ == "__main__":
    main()
