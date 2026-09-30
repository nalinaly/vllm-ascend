"""分别保留同进程筛选、服务入口ABBA与独立DFX，禁止混算。"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "hca_residual_reuse_20260930"))
from collect import incore, stats  # noqa: E402


def main():
    import torch

    torch.set_num_threads(4)
    data = ROOT.parent / "results" / ROOT.name
    output = {"source": json.loads((ROOT / "source_combo.json").read_text()), "pair": {}, "passes": []}
    for name in ("combo_h131072_b16", "combo_h16384_b4"):
        path = data / name / "report.json"
        report = json.loads(path.read_text())
        assert report["status"] == "MEASURED"
        assert all(report["cross_variant_exact"].values())
        output["pair"][name] = {
            "source_report": str(path),
            **{k: report[k] for k in ("case", "device", "runtime_shapes", "summary", "cross_variant_exact", "guards")},
            "spans_us": report["profile"]["span_us_all"],
        }
    assert output["pair"]["combo_h16384_b4"]["runtime_shapes"]["candidate"]["attention_path"] == "long"
    reference = torch.load(data / "combo_service_abba/p1_base/states.pt", map_location="cpu", weights_only=True)
    for folder in ("p1_base", "p2_ring", "p3_ring", "p4_base"):
        path = data / "combo_service_abba" / folder
        report = json.loads((path / "report.json").read_text())
        assert report["status"] == "MEASURED", report.get("error")
        side = folder.split("_")[1]
        assert report["operator_source"] == output["source"]["baseline" if side == "base" else "candidate"]
        state = torch.load(path / "states.pt", map_location="cpu", weights_only=True)
        assert state.keys() == reference.keys()
        exact = {k: bool(torch.equal(v, state[k])) for k, v in reference.items()}
        profile = report["device_profile"]
        assert profile["replays"] == 9 and set(profile["kernels_per_replay"]) == {2}
        assert all(exact.values())
        assert all(v["status"] == "PASS" for v in report["timing"]["guards"].values())
        assert all(v["status"] == "PASS" for v in report["timing"]["eager_comparison"].values())
        output["passes"].append(
            {
                "source_report": str(path / "report.json"),
                "side": side,
                "device": report["device"],
                "spans_us": profile["span_us_all"],
                "state_exact": exact,
                "guards": {k: v["status"] for k, v in report["timing"]["guards"].items()},
                "replay_checks": {k: v["status"] for k, v in report["timing"]["eager_comparison"].items()},
            }
        )
    assert len({p["device"] for p in output["passes"]}) == 1
    output["service_summary"] = {
        side: stats([s for p in output["passes"] if p["side"] == side for s in p["spans_us"]])
        for side in ("base", "ring")
    }
    output["incore_v2_before_ob"] = {}
    for side in ("base", "ring"):
        report = json.loads((data / f"h131072_b16/swimlane_{side}/report.json").read_text())
        output["incore_v2_before_ob"][side] = incore(Path(report["swimlane"]["merged_swimlane"]))
    (ROOT / "result_combo.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(output["service_summary"]))


if __name__ == "__main__":
    main()
