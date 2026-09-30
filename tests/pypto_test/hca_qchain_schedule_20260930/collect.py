"""比较同步启动的整层、核内与链路时序；精度检查复用已保存状态。"""

import argparse
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read_worker_events(path):
    events = json.loads(path.read_text())["traceEvents"]
    pid = next(
        e["pid"]
        for e in events
        if e.get("ph") == "M" and e.get("name") == "process_name" and e["args"]["name"] == "Worker View"
    )
    selected = [
        e for e in events if e.get("pid") == pid and e.get("ph") == "X" and "kernel-duration-us" in e.get("args", {})
    ]
    origin = min(e["ts"] for e in selected)
    return [
        (e["name"].split("(")[0], e["ts"] + e["args"]["local_setup_us"] - origin, e["ts"] + e["dur"] - origin)
        for e in selected
    ]


def chain(path):
    rows = read_worker_events(path)
    tasks = {}
    for prefix in ("qproj_matmul", "qproj_dequant", "hca_unified_attention", "hca_raw_attn", "hca_cmp_qk_pv"):
        selected = [(start, end) for name, start, end in rows if name.startswith(prefix)]
        if not selected:
            continue
        first = min(start for start, end in selected)
        last = max(end for start, end in selected)
        tasks[prefix] = {
            "first_start_us": first,
            "last_end_us": last,
            "start_spread_us": max(start for start, end in selected) - first,
            "envelope_us": last - first,
        }
    q = tasks["qproj_matmul"]
    dq = tasks["qproj_dequant"]
    return {
        "tasks": tasks,
        "dq_first_minus_q_last_us": dq["first_start_us"] - q["last_end_us"],
        "attention_first_minus_dq_last_us": min(
            value["first_start_us"] for name, value in tasks.items() if name.startswith("hca_")
        )
        - dq["last_end_us"],
        "limits": "one independent DFX window; negative gap means overlap; timings alone do not prove causality",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=int, default=131072)
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()
    result_root = ROOT.parent / f"results/hca_qchain_schedule_20260930/h{args.history}_b{args.batch}"
    spec = importlib.util.spec_from_file_location(
        "residual_collect", ROOT.parent / "hca_residual_reuse_20260930/collect.py"
    )
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
    import torch

    torch.set_num_threads(4)
    source = json.loads((ROOT / "source.json").read_text())
    reference = torch.load(result_root / "abba/p1_base/states.pt", map_location="cpu", weights_only=True)
    passes = []
    sides = ("base", "syncq", "syncboth")
    for index, side in enumerate((*sides, *reversed(sides)), start=1):
        directory = result_root / "abba" / f"p{index}_{side}"
        report = json.loads((directory / "report.json").read_text())
        assert report["status"] == "MEASURED", report.get("error")
        assert report["operator_source"] == source["sources"][side]
        actual = torch.load(directory / "states.pt", map_location="cpu", weights_only=True)
        assert actual.keys() == reference.keys()
        exact = {name: bool(torch.equal(value, actual[name])) for name, value in reference.items()}
        del actual
        profile = report["device_profile"]
        assert profile["replays"] == len(profile["span_us_all"]) == 9
        assert len(set(profile["kernels_per_replay"])) == 1
        passes.append(
            {
                "directory": str(directory),
                "side": side,
                "device": report["device"],
                "span_samples_us": profile["span_us_all"],
                "device_span": helpers.stats(profile["span_us_all"]),
                "event_samples_us": report["timing"]["samples_us"],
                "state_exact": exact,
                "replay_checks": {k: v["status"] for k, v in report["timing"]["eager_comparison"].items()},
                "guards": {k: v["status"] for k, v in report["timing"]["guards"].items()},
            }
        )
    assert len({p["device"] for p in passes}) == 1
    summary = {}
    incore = {}
    for side in sides:
        selected = [p for p in passes if p["side"] == side]
        summary[side] = {
            "all_device_replays": helpers.stats([s for p in selected for s in p["span_samples_us"]]),
            "pass_medians": helpers.stats([p["device_span"]["p50_us"] for p in selected]),
            "all_event_replays": helpers.stats([s for p in selected for s in p["event_samples_us"]]),
        }
        report = json.loads((result_root / f"swimlane_{side}/report.json").read_text())
        assert report["status"] == "MEASURED"
        path = Path(report["swimlane"]["merged_swimlane"])
        incore[side] = {**helpers.incore(path), "q_chain": chain(path)}
    result = {
        "source": source,
        "case": [args.history, args.batch],
        "passes": passes,
        "summary": summary,
        "incore": incore,
    }
    (ROOT / f"result_h{args.history}_b{args.batch}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(summary))
    print("all_states_exact", all(all(p["state_exact"].values()) for p in passes))
    for side, values in incore.items():
        print(side, values["q_chain"])


if __name__ == "__main__":
    main()
