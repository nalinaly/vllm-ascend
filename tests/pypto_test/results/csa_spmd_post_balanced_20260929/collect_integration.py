"""核对最终形状分支的完整状态；跨卡计时不用于单项收益归因。"""

import json

from collect import ROOT, pass_checks, stats


def main():
    import torch

    torch.set_num_threads(4)
    manifest = json.loads((ROOT / "source.json").read_text())
    result = {
        "task": "task_20260929_232626_286499726622",
        "scope": "integration; no paired performance claim",
        "cases": [],
    }
    for history, batch in manifest["cases"]:
        case = ROOT / f"h{history}_b{batch}" / "timing"
        directory = case / "o_post_t96_sync"
        report = json.loads((directory / "report.json").read_text())
        assert report["status"] == "MEASURED"
        assert report["source"] == manifest["source_prefix"] + "-o_post_t96_sync"
        baseline = torch.load(case / "baseline_start/states.pt", map_location="cpu", weights_only=True)
        actual = torch.load(directory / "states.pt", map_location="cpu", weights_only=True)
        assert len(baseline) == 8 and set(actual) == set(baseline)
        state = {name: bool(torch.equal(baseline[name], actual[name])) for name in baseline}
        assert all(state.values())
        del baseline, actual
        result["cases"].append(
            {
                "history": history,
                "batch": batch,
                "device": report["device"],
                "source": report["source"],
                "report": str(directory / "report.json"),
                "reference": str(case / "baseline_start/states.pt"),
                "timing": stats(report["timing"]["samples_us"]),
                "samples_us": report["timing"]["samples_us"],
                "state_exact": state,
                "initial_guards": pass_checks(report["initial_guards"]),
                "graph_guards": pass_checks(report["timing"]["guards"]),
                "eager_comparison": pass_checks(report["timing"]["eager_comparison"]),
            }
        )
    (ROOT / "integration.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print("PASS: T96 strategy and T144 fallback, eight exact states and graph guards")


if __name__ == "__main__":
    main()
