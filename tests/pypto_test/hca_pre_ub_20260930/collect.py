"""收集现有完整状态/真实span和单独DFX；无需分配NPU。"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    sys.path.insert(0, str(ROOT.parent))
    from hca_residual_reuse_20260930.collect import incore

    results = ROOT.parent / "results" / ROOT.name
    reports = {}
    for path in sorted(results.glob("*/report.json")):
        report = json.loads(path.read_text())
        if report["status"] != "MEASURED":
            continue
        if "cross_variant_exact" not in report:
            continue
        assert all(report["cross_variant_exact"].values())
        assert all(all(v.values()) for v in report["state_exact"].values())
        assert all(v == "PASS" for v in report["guards"].values())
        reports[path.parent.name] = {k: report[k] for k in (
            "sources", "case", "device", "cann", "summary", "cross_variant_exact", "state_exact", "guards",
        )}
        reports[path.parent.name]["source_report"] = str(path)
        reports[path.parent.name]["purpose"] = (
            "functional_only" if report["summary"]["base"]["count"] < 10 else "performance_screen"
        )
        if "padding_graph" in report:
            padding = report["padding_graph"]
            assert padding["status"] == "PASS"
            for replay in padding["replays"]:
                for group in ("state_comparison", "compact_metadata", "guards"):
                    assert all(v["status"] == "PASS" for v in replay[group].values())
            reports[path.parent.name]["padding_graph"] = {
                "status": padding["status"], "scope": padding["scope"],
                "active_batch_sequence": [r["active_batch"] for r in padding["replays"]],
            }
    memory = {}
    for side in ("mix_ub_v7", "mix_gm_bf16", "mix_ub_bf16", "mix_m4_box8_v3", "mix_m4_box8_d512_v3"):
        dump = ROOT / f"compiled/{side}/passes_dump/38_after_AllocateMemoryAddr.py"
        if not dump.exists():
            continue
        body = dump.read_text().split("    def mix_x_rms_norm(", 1)[1].split("    def ", 1)[0]
        allocations = re.findall(r"pl.MemRef\(mem_vec_\d+, pl.const\((\d+), pl.INT64\), (\d+)\)", body)
        memory[side] = {"Vec_static_allocation_end_bytes": max(int(o) + int(s) for o, s in allocations),
                        "source_dump": str(dump)}
    dfx = {}
    for path in sorted((results / "dfx").glob("*/dfx/merged_swimlane.json")):
        dfx[path.parents[1].name] = incore(path)
    summary = {"baseline_commit": "ad0e6bbe", "reports": reports, "codegen_memory": memory,
               "limits": "不同候选各自ABBA控制；DFX独立采样，核内与整段不直接相减归因；未新增整机验收"}
    (ROOT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    if dfx:
        (ROOT / "incore.json").write_text(json.dumps(dfx, indent=2) + "\n")
    print(json.dumps({side: report["summary"] for side, report in reports.items()}, indent=2))
    print(json.dumps(memory, indent=2))


if __name__ == "__main__":
    main()
