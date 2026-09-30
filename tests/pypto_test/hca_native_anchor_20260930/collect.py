"""只汇总同一完整ABBA的真实设备重放；不混入失败轮或图外Event。"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))


def main():
    from hca_residual_reuse_20260930.collect import stats

    result = ROOT.parent / "results" / ROOT.name / "h131072_b16_compatible"
    passes = []
    for name in ("p1_native", "p2_pto", "p3_pto", "p4_native"):
        path = result / name / "report.json"
        report = json.loads(path.read_text())
        assert report["status"] == "MEASURED", report.get("error")
        options = report["backend_options"]
        assert options["static_kernel_compile"] and options["inplace_pass"]
        assert 'dynamic=False' in report["compile_entry"]
        native = report["side"] == "native"
        assert report["super_kernel"] == native
        compiler = report["compiler"]
        if native:
            assert compiler["static_compile_results"] and all(compiler["static_compile_results"])
            assert compiler["installed_static_packages"] > 0
            assert compiler["static_super_flags"] and all(compiler["static_super_flags"])
        else:
            assert compiler["pto_dispatch_calls"] > 0
        profile = report["device_profile"]
        assert profile["available"] and profile["replays"] == 9
        samples = profile["span_us_all"]
        assert len(samples) == 9
        assert len(set(profile["kernels_per_replay"])) == 1
        assert all(v["status"] == "PASS" for v in report["timing"]["guards"].values())
        assert all(v["nonfinite"] == 0 for v in report["timing"]["eager_comparison"].values())
        if not native:
            assert all(v["status"] == "PASS" for v in report["timing"]["eager_comparison"].values())
        passes.append({
            "pass": name, "report": str(path), "side": report["side"],
            "device": report["device"], "cann": report["cann"],
            "source": report["source"], "operator_source": report["operator_source"],
            "samples_us": samples, "statistics": stats(samples),
            "compiler": compiler, "backend_options": options,
            "eager_comparison": report["timing"]["eager_comparison"],
            "guards": report["timing"]["guards"],
        })
    assert len({p["device"] for p in passes}) == 1
    assert len({str(p["cann"]) for p in passes}) == 1
    summary = {side: stats([x for p in passes if p["side"] == side for x in p["samples_us"]])
               for side in ("native", "pto")}
    ratios = {key: summary["pto"][key] / summary["native"][key]
              for key in ("mean_us", "p50_us")}
    report = {
        "status": "MEASURED", "baseline_commit": "a7a9f314",
        "case": [131072, 16], "passes": passes, "summary": summary,
        "pto_over_native": ratios,
        "target_p50_us": summary["native"]["p50_us"] * 0.8,
        "timing_target_satisfied_this_case": ratios["p50_us"] <= 0.8,
        "scope": "real layer3 weights, mHC pre/norm/HCA/O/post, single card, 18 actual spans per side",
        "limits": "no new seven-tier or full-model/token acceptance; Native/PTO cross-implementation "
                  "elementwise accuracy is not checked by this timing runner",
        "native_eager_diagnostic": "both Native passes differ from eager at zero tolerance: "
                                   "x_out max_abs=0.015625, 15814 elements; swa max_abs=0.0078125, 9 elements. "
                                   "No nonfinite or guard failure; not promoted to accuracy acceptance",
        "failed_attempt": "task_20260930_032122_216888230045: p2_pto SK1 failed 107017 invalid funcHandle; "
                          "old p1_native excluded, complete ABBA rerun with PTO SK0",
    }
    (ROOT / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("summary", "pto_over_native", "target_p50_us")}, indent=2))


if __name__ == "__main__":
    main()
