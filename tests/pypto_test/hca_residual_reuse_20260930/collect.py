"""汇总全部重放与pass中位数，分开报告；一次性比较已保存的完整状态。"""

import argparse
import collections
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def stats(samples):
    return {
        "count": len(samples),
        "min_us": min(samples),
        "max_us": max(samples),
        "mean_us": statistics.mean(samples),
        "p50_us": statistics.median(samples),
    }


def incore(path):
    events = json.loads(path.read_text())["traceEvents"]
    pid = next(
        e["pid"]
        for e in events
        if e.get("ph") == "M" and e.get("name") == "process_name" and e["args"]["name"] == "Worker View"
    )
    rows = [
        e for e in events if e.get("pid") == pid and e.get("ph") == "X" and "kernel-duration-us" in e.get("args", {})
    ]
    origin = min(e["ts"] for e in rows)
    groups = collections.defaultdict(list)
    for event in rows:
        groups[event["name"].split("(")[0]].append(event)
    tasks = {}
    for name, group in groups.items():
        starts = [e["ts"] + e["args"]["local_setup_us"] - origin for e in group]
        ends = [e["ts"] + e["dur"] - origin for e in group]
        tasks[name] = {
            **stats([e["args"]["kernel-duration-us"] for e in group]),
            "first_start_us": min(starts),
            "last_end_us": max(ends),
            "envelope_us": max(ends) - min(starts),
        }
    return {"source": str(path), "tasks": tasks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=int, default=131072)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--experiment-root", type=Path, default=ROOT)
    parser.add_argument("--candidate-label", default="reuse")
    args = parser.parse_args()
    experiment = args.experiment_root.resolve()
    label = args.candidate_label
    result_root = experiment.parent / f"results/{experiment.name}/h{args.history}_b{args.batch}"
    import torch

    torch.set_num_threads(4)
    source = json.loads((experiment / "source.json").read_text())
    states = torch.load(result_root / "abba/p1_base/states.pt", map_location="cpu", weights_only=True)
    passes = []
    for folder in ("p1_base", f"p2_{label}", f"p3_{label}", "p4_base"):
        path = result_root / "abba" / folder
        report = json.loads((path / "report.json").read_text())
        assert report["status"] == "MEASURED", report.get("error")
        side = folder.split("_")[1]
        assert report["operator_source"] == source["baseline" if side == "base" else "candidate"]
        actual = torch.load(path / "states.pt", map_location="cpu", weights_only=True)
        assert actual.keys() == states.keys()
        exact = {key: bool(torch.equal(states[key], actual[key])) for key in states}
        del actual
        profile = report["device_profile"]
        assert profile["replays"] == 9 and len(profile["span_us_all"]) == 9
        assert len(set(profile["kernels_per_replay"])) == 1, profile["kernels_per_replay"]
        passes.append(
            {
                "directory": str(path),
                "side": side,
                "device": report["device"],
                "span_samples_us": profile["span_us_all"],
                "device_span": stats(profile["span_us_all"]),
                "event_samples_us": report["timing"]["samples_us"],
                "event": stats(report["timing"]["samples_us"]),
                "state_exact": exact,
                "state_pass": all(exact.values()),
                "replay_checks": {key: value["status"] for key, value in report["timing"]["eager_comparison"].items()},
                "guards": {key: value["status"] for key, value in report["timing"]["guards"].items()},
            }
        )
    assert len({p["device"] for p in passes}) == 1
    summary = {}
    for side in ("base", label):
        selected = [p for p in passes if p["side"] == side]
        summary[side] = {
            "all_device_replays": stats([s for p in selected for s in p["span_samples_us"]]),
            "pass_medians": stats([p["device_span"]["p50_us"] for p in selected]),
            "all_event_replays": stats([s for p in selected for s in p["event_samples_us"]]),
        }
    output = {
        "source": source,
        "case": [args.history, args.batch],
        "passes": passes,
        "summary": summary,
        "limits": (
            "single-card ABBA screen; no stable small-effect or model/Native acceptance claim; event includes host gap"
        ),
    }
    output["incore"] = {}
    for side in ("base", label):
        report = json.loads((result_root / f"swimlane_{side}/report.json").read_text())
        assert report["status"] == "MEASURED"
        output["incore"][side] = incore(Path(report["swimlane"]["merged_swimlane"]))
    (experiment / f"result_h{args.history}_b{args.batch}.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False))
    print("state_pass", all(p["state_pass"] for p in passes))


if __name__ == "__main__":
    main()
