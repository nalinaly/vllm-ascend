"""汇总单任务用核/流水/sync试验；复用已经完成的状态检查，不重新占卡。"""

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def stats(values):
    return dict(max_us=max(values), min_us=min(values), mean_us=statistics.mean(values), count=len(values))


def pass_checks(checks):
    if isinstance(checks, list):
        return [pass_checks(group) for group in checks]
    if not checks or not all(item["status"] == "PASS" for item in checks.values()):
        raise ValueError("State/guard check failed")
    return {"status": "PASS", "items": len(checks)}


def window(path):
    events = json.loads(path.read_text())["traceEvents"]
    pids = {
        e["pid"]
        for e in events
        if e.get("ph") == "M" and e.get("name") == "process_name" and e.get("args", {}).get("name") == "Worker View"
    }
    workers = [
        e for e in events if e.get("pid") in pids and e.get("ph") == "X" and "kernel-duration-us" in e.get("args", {})
    ]
    rows = [e for e in workers if e["name"].startswith("proj_b_act_hc_post")]
    preceding = [e for e in workers if e["name"].startswith("_proj_b_mm_nz_kernel")]
    origin = min(e["ts"] for e in workers)
    starts = [e["ts"] + e["args"]["local_setup_us"] for e in rows]
    ends = [e["ts"] + e["dur"] for e in rows]
    values = [e["args"]["kernel-duration-us"] for e in rows]
    return {
        "source": str(path),
        "kernel": stats(values),
        "kernel_samples_us": values,
        "workers": len(rows),
        "physical_cores": len({e["tid"] for e in rows}),
        "first_start_us": min(starts) - origin,
        "last_end_us": max(ends) - origin,
        "start_spread_us": max(starts) - min(starts),
        "kernel_envelope_us": max(ends) - min(starts),
        "o_b_to_post_us": min(starts) - max(e["ts"] + e["dur"] for e in preceding),
        "worker_span_us": max(e["ts"] + e["dur"] for e in workers) - origin,
    }


def main():
    import torch

    torch.set_num_threads(4)
    source = json.loads((ROOT / "source.json").read_text())
    cached_long = json.loads((ROOT / "long_state_checks.json").read_text())
    result = {"source": source, "scope": "single-card CSA; no model acceptance", "cases": []}
    for history, batch in source["cases"]:
        folder = ROOT / f"h{history}_b{batch}"
        sides = {}
        reference = None
        for name in ("baseline_start", "o_post_balanced", "o_post_balanced_sync", "baseline_end"):
            path = folder / "timing" / name / "report.json"
            d = json.loads(path.read_text())
            assert d["status"] == "MEASURED"
            expected = "baseline" if name.startswith("baseline_") else name
            assert d["source"] == source["source_prefix"] + "-" + expected
            assert d["variant"] == source["variant"]
            values = d["timing"]["samples_us"]
            assert len(values) == 20
            side = {
                "report": str(path),
                "source": d["source"],
                "device": d["device"],
                "cann": d["cann"],
                **stats(values),
                "samples_us": values,
                "initial_guards": pass_checks(d["initial_guards"]),
                "graph_guards": pass_checks(d["timing"]["guards"]),
                "eager_comparison": pass_checks(d["timing"]["eager_comparison"]),
            }
            if name in ("o_post_balanced", "o_post_balanced_sync"):
                if history == 131072:
                    side["state_exact"] = cached_long[name]
                    side["state_check_source"] = str(ROOT / "long_state_checks.json")
                else:
                    if reference is None:
                        reference = torch.load(
                            folder / "timing/baseline_start/states.pt", map_location="cpu", weights_only=True
                        )
                    actual = torch.load(path.parent / "states.pt", map_location="cpu", weights_only=True)
                    assert len(reference) == 8 and set(actual) == set(reference)
                    side["state_exact"] = {key: bool(torch.equal(reference[key], actual[key])) for key in reference}
                    del actual
                assert len(side["state_exact"]) == 8 and all(side["state_exact"].values())
            dfx_path = folder / "swimlane" / name / "report.json"
            if dfx_path.exists():
                dfx = json.loads(dfx_path.read_text())
                side["dfx_guards"] = pass_checks(dfx["pto_guards"])
                side["dfx_self"] = pass_checks(dfx["pto_self"])
                side["windows"] = [window(Path(w["merged_swimlane"])) for w in dfx["swimlane_windows"] if w["exported"]]
                assert len(side["windows"]) == 2
                side["target_kernel"] = stats([v for w in side["windows"] for v in w["kernel_samples_us"]])
            sides[name] = side
        del reference
        assert len({s["device"] for s in sides.values()}) == 1
        for side in sides.values():
            side["delta_start_pct"] = 100 * (side["mean_us"] / sides["baseline_start"]["mean_us"] - 1)
            side["delta_end_pct"] = 100 * (side["mean_us"] / sides["baseline_end"]["mean_us"] - 1)
        result["cases"].append(
            {
                "history": history,
                "batch": batch,
                "variants": sides,
                "baseline_drift_pct": sides["baseline_end"]["delta_start_pct"],
            }
        )
    result["weighted_8_2_pct"] = {
        name: {
            control: sum(
                weight * case["variants"][name][f"delta_{control}_pct"]
                for weight, case in zip((0.8, 0.2), result["cases"])
            )
            for control in ("start", "end")
        }
        for name in ("o_post_balanced", "o_post_balanced_sync")
    }
    (ROOT / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    lines = [
        "# O收尾逐任务扩核与sync实测",
        "",
        "单位μs，最大/最小/平均，正式计时每项20样本，保留全部慢点。",
        "前后基线分别报告；两者之间的漂移不能冒充优化收益。DFX独立采集，不与无profiler事件相减。",
        "",
    ]
    for case in result["cases"]:
        lines += [
            f"## H{case['history']}/B{case['batch']}",
            "",
            "| 配置 | 完整CSA最大 | 最小 | 平均 | 对开始基线 | 对结束基线 |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for name, side in case["variants"].items():
            lines += [
                f"| {name} | {side['max_us']:.3f} | {side['min_us']:.3f} | {side['mean_us']:.3f} "
                f"| {side['delta_start_pct']:+.3f}% | {side['delta_end_pct']:+.3f}% |"
            ]
        lines += [
            "",
            "| 配置 | 收尾核时最大 | 最小 | 平均 | 每窗有效AIV | 两窗核时包络 |",
            "| --- | ---: | ---: | ---: | --- | --- |",
        ]
        for name, side in case["variants"].items():
            if "target_kernel" not in side:
                continue
            k = side["target_kernel"]
            cores = "/".join(str(w["physical_cores"]) for w in side["windows"])
            spans = "/".join(f"{w['kernel_envelope_us']:.3f}" for w in side["windows"])
            lines += [f"| {name} | {k['max_us']:.3f} | {k['min_us']:.3f} | {k['mean_us']:.3f} | {cores} | {spans} |"]
        lines += [
            "",
            "两个候选的八类完整状态与原版逐bit一致；自身eager/graph及保护区均通过。未做本轮模型token/DSpark验收。",
            "",
        ]
    lines += ["8:2均值变化（对开始/结束基线）：", ""]
    for name, changes in result["weighted_8_2_pct"].items():
        lines += [f"- {name}: {changes['start']:+.3f}% / {changes['end']:+.3f}%"]
    (ROOT / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(result["weighted_8_2_pct"], ensure_ascii=False))


if __name__ == "__main__":
    main()
