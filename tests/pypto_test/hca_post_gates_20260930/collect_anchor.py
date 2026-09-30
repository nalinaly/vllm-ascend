"""汇总新组合与Native同卡ABBA；只使用真实设备span并核对实际编译模式。"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))


def main():
    from hca_residual_reuse_20260930.collect import stats

    result = ROOT.parent / "results" / ROOT.name / "native_anchor_h131072_b16"
    passes = []
    for name in ("p1_native", "p2_pto", "p3_pto", "p4_native"):
        path = result / name / "report.json"
        report = json.loads(path.read_text())
        assert report["status"] == "MEASURED", report.get("error")
        options, compiler = report["backend_options"], report["compiler"]
        assert options["static_kernel_compile"] and options["inplace_pass"]
        assert "dynamic=False" in report["compile_entry"]
        native = report["side"] == "native"
        assert report["super_kernel"] == native
        if native:
            assert compiler["static_compile_results"] and all(compiler["static_compile_results"])
            assert compiler["static_super_flags"] and all(compiler["static_super_flags"])
            assert compiler["installed_static_packages"] > 0
        else:
            assert compiler["pto_dispatch_calls"] > 0
        profile = report["device_profile"]
        assert profile["available"] and profile["replays"] == 9
        samples = profile["span_us_all"]
        assert len(samples) == 9 and len(set(profile["kernels_per_replay"])) == 1
        assert all(v["status"] == "PASS" for v in report["timing"]["guards"].values())
        eager = report["timing"]["eager_comparison"]
        assert all(v["nonfinite"] == 0 for v in eager.values())
        if not native:
            assert all(v["status"] == "PASS" for v in eager.values())
        passes.append({
            "pass": name, "source_report": str(path), "side": report["side"],
            "device": report["device"], "cann": report["cann"],
            "operator_source": report["operator_source"], "samples_us": samples,
            "statistics": stats(samples), "compiler": compiler, "backend_options": options,
            "eager_comparison": {
                k: {f: v[f] for f in ("status", "mismatches", "max_abs", "nonfinite")}
                for k, v in eager.items()
            },
            "guards": "PASS",
        })
    assert len({p["device"] for p in passes}) == 1
    assert len({str(p["cann"]) for p in passes}) == 1
    summary = {
        side: stats([x for p in passes if p["side"] == side for x in p["samples_us"]])
        for side in ("native", "pto")
    }
    ratios = {k: summary["pto"][k] / summary["native"][k] for k in ("mean_us", "p50_us")}
    output = {
        "status": "MEASURED", "case": [131072, 16], "passes": passes, "summary": summary,
        "pto_over_native": ratios,
        "target_mean_us": 0.8 * summary["native"]["mean_us"],
        "timing_target_satisfied_this_case": all(v <= 0.8 for v in ratios.values()),
        "scope": "single card, real layer3 weights, mHC pre/norm/HCA/O/post, 18 actual spans per side",
        "limits": "not seven-tier or full-model/token acceptance; no cross-implementation elementwise check",
        "native_eager_diagnostic": "zero-tolerance differences recorded per pass; not treated as token acceptance",
    }
    (ROOT / "native_anchor.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: output[k] for k in ("summary", "pto_over_native", "target_mean_us")}, indent=2))


if __name__ == "__main__":
    main()
