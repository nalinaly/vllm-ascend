"""检查KV分工、相关并发者与完整CSA；性能后一次性检查八类状态。"""

import argparse
import collections
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NAMES = ("baseline_start", "kv_n64", "kv_n64_sync", "kv_n64_linked", "baseline_end")


def stats(values):
    return dict(max_us=max(values), min_us=min(values), mean_us=statistics.mean(values), count=len(values))


def passed(checks):
    if isinstance(checks, list):
        return [passed(c) for c in checks]
    assert checks and all(value["status"] == "PASS" for value in checks.values()), checks
    return {"status": "PASS", "count": len(checks)}


def role(name):
    name = name.split("(")[0].removesuffix("_spmd")
    if name.startswith("kv_proj_matmul"):
        return "KV"
    if name == "kv_score_proj":
        return "Compressor"
    if name == "kv_score_proj_0":
        return "IndexerCompressor"
    for prefix in (
        "idx_qr_proj_matmul",
        "qr_proj_matmul",
        "qproj_matmul",
        "kv_rms_norm_rope",
        "indexer_score_topk_native_pair",
        "qk_pv",
        "proj_b_act_hc_post",
    ):
        if name.startswith(prefix):
            if prefix == "indexer_score_topk_native_pair":
                return prefix + ("_aic" if name.endswith("_aic") else "_aiv")
            return prefix
    return None


def window(path):
    events = json.loads(path.read_text())["traceEvents"]
    pids = {
        e["pid"]
        for e in events
        if e.get("ph") == "M" and e.get("name") == "process_name" and e.get("args", {}).get("name") == "Worker View"
    }
    rows = [
        e for e in events if e.get("ph") == "X" and e.get("pid") in pids and "kernel-duration-us" in e.get("args", {})
    ]
    origin = min(e["ts"] for e in rows)
    groups = collections.defaultdict(list)
    for row in rows:
        key = role(row["name"])
        if key is not None:
            groups[key].append(row)
    tasks = {}
    for name, group in groups.items():
        starts = [e["ts"] + e["args"]["local_setup_us"] - origin for e in group]
        ends = [e["ts"] + e["dur"] - origin for e in group]
        values = [e["args"]["kernel-duration-us"] for e in group]
        tasks[name] = {
            "kernel": stats(values),
            "core_us": sum(values),
            "workers": len(group),
            "physical_cores": len({e["tid"] for e in group}),
            "first_start_us": min(starts),
            "last_end_us": max(ends),
            "start_spread_us": max(starts) - min(starts),
            "kernel_envelope_us": max(ends) - min(starts),
        }
    assert {"KV", "Compressor", "IndexerCompressor"} <= tasks.keys()
    return {"source": str(path), "tasks": tasks, "worker_span_us": max(e["ts"] + e["dur"] for e in rows) - origin}


def combined_stats(windows, name):
    values = [w["tasks"][name]["kernel"] for w in windows]
    count = sum(v["count"] for v in values)
    return {
        "max_us": max(v["max_us"] for v in values),
        "min_us": min(v["min_us"] for v in values),
        "mean_us": sum(v["mean_us"] * v["count"] for v in values) / count,
        "count": count,
        "envelope": stats([w["tasks"][name]["kernel_envelope_us"] for w in windows]),
        "core_us_per_window": statistics.mean(w["tasks"][name]["core_us"] for w in windows),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partial", action="store_true")
    args = parser.parse_args()
    source = json.loads((ROOT / "source.json").read_text())
    if not args.partial:
        import torch

        torch.set_num_threads(4)
    result = {"source": source, "scope": "same-card per shape; no seven-case/model acceptance", "cases": []}
    for history, batch in source["cases"]:
        folder = ROOT / f"h{history}_b{batch}"
        sides = {}
        reference = None
        for name in NAMES:
            path = folder / "timing" / name / "report.json"
            if args.partial and not path.exists():
                continue
            report = json.loads(path.read_text())
            assert report["status"] == "MEASURED", (path, report.get("error"))
            expected = "baseline" if name.startswith("baseline_") else name
            assert report["source"] == source["source_prefix"] + "-" + expected
            assert report["variant"] == source["variant"]
            samples = report["timing"]["samples_us"]
            side = {
                "report": str(path),
                "device": report["device"],
                **stats(samples),
                "samples_us": samples,
                "initial_guards": passed(report["initial_guards"]),
                "graph_guards": passed(report["timing"]["guards"]),
                "self": passed(report["timing"]["eager_comparison"]),
            }
            if not args.partial and name not in ("baseline_start", "baseline_end"):
                if reference is None:
                    reference = torch.load(
                        folder / "timing/baseline_start/states.pt", map_location="cpu", weights_only=True
                    )
                actual = torch.load(path.parent / "states.pt", map_location="cpu", weights_only=True)
                assert len(actual) == 8 and set(actual) == set(reference)
                side["state_exact"] = {key: bool(torch.equal(actual[key], reference[key])) for key in reference}
                del actual
                # Record numerical failures as evidence; never silently accept them.
                side["state_pass"] = all(side["state_exact"].values())
            dfx_path = folder / "swimlane" / name / "report.json"
            if dfx_path.exists():
                dfx = json.loads(dfx_path.read_text())
                side["dfx_self"] = passed(dfx["pto_self"])
                side["dfx_guards"] = passed(dfx["pto_guards"])
                side["windows"] = [window(Path(w["merged_swimlane"])) for w in dfx["swimlane_windows"] if w["exported"]]
                assert len(side["windows"]) == 2
                side["tasks"] = {key: combined_stats(side["windows"], key) for key in side["windows"][0]["tasks"]}
            elif not args.partial and name != "baseline_end":
                raise ValueError(f"Missing DFX {dfx_path}")
            sides[name] = side
        del reference
        if "baseline_start" not in sides:
            continue
        assert len({s["device"] for s in sides.values()}) == 1
        for side in sides.values():
            for control in ("start", "end"):
                if "baseline_" + control in sides:
                    side[f"delta_{control}_pct"] = 100 * (side["mean_us"] / sides["baseline_" + control]["mean_us"] - 1)
                    side[f"delta_max_{control}_pct"] = 100 * (
                        side["max_us"] / sides["baseline_" + control]["max_us"] - 1
                    )
        result["cases"].append({"history": history, "batch": batch, "variants": sides})
    result["weighted_8_2_pct"] = {}
    result["weighted_max_8_2_pct"] = {}
    if len(result["cases"]) == 2:
        a, b = result["cases"]
        for name in set(a["variants"]) & set(b["variants"]):
            if name.startswith("baseline"):
                continue
            result["weighted_8_2_pct"][name] = {
                control: sum(
                    weight * case["variants"][name][f"delta_{control}_pct"]
                    for weight, case in zip((0.8, 0.2), result["cases"])
                )
                for control in ("start", "end")
                if all(f"delta_{control}_pct" in c["variants"][name] for c in result["cases"])
            }
            result["weighted_max_8_2_pct"][name] = {
                control: sum(
                    weight * case["variants"][name][f"delta_max_{control}_pct"]
                    for weight, case in zip((0.8, 0.2), result["cases"])
                )
                for control in ("start", "end")
                if all(f"delta_max_{control}_pct" in c["variants"][name] for c in result["cases"])
            }
    filename = "partial.json" if args.partial else "summary.json"
    (ROOT / filename).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    if not args.partial:
        write_markdown(result)
    print(
        json.dumps(
            {
                "cases": [
                    {
                        "history": c["history"],
                        "batch": c["batch"],
                        "variants": {
                            name: {key: side.get(key) for key in ("max_us", "min_us", "mean_us", "state_pass")}
                            for name, side in c["variants"].items()
                        },
                    }
                    for c in result["cases"]
                ],
                "weighted": result["weighted_8_2_pct"],
            },
            ensure_ascii=False,
        )
    )


def write_markdown(result):
    lines = [
        "# KV扩核与并发联动结果",
        "",
        "单位μs；最大/最小/平均，全部正式事件保留。",
        "每档内部同卡配对，开始/结束基线分别报告；独立DFX核时不与无profiler事件相减。",
        "",
    ]
    for case in result["cases"]:
        lines += [
            f"## H{case['history']}/B{case['batch']}",
            "",
            "| 配置 | 完整CSA最大 | 最小 | 平均 | 对开始基线 | 对结束基线 | 八类状态 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
        for name, side in case["variants"].items():
            state = "—" if "state_pass" not in side else str(side["state_pass"])
            lines += [
                f"| {name} | {side['max_us']:.3f} | {side['min_us']:.3f} | {side['mean_us']:.3f} "
                f"| {side['delta_start_pct']:+.3f}% | {side['delta_end_pct']:+.3f}% | {state} |"
            ]
        lines += [
            "",
            "最大值独立变化（各20样本）：",
            "",
            "| 配置 | 对开始基线max | 对结束基线max |",
            "| --- | ---: | ---: |",
        ]
        for name, side in case["variants"].items():
            lines += [f"| {name} | {side['delta_max_start_pct']:+.3f}% | {side['delta_max_end_pct']:+.3f}% |"]
        for key in ("KV", "Compressor", "IndexerCompressor", "idx_qr_proj_matmul"):
            lines += [
                "",
                f"### {key}",
                "",
                "| 配置 | 核时最大 | 最小 | 平均 | 包络最大/最小/平均 | 每窗core-us | worker数 |",
                "| --- | ---: | ---: | ---: | --- | ---: | --- |",
            ]
            for name, side in case["variants"].items():
                if "tasks" not in side:
                    continue
                task = side["tasks"][key]
                env = task["envelope"]
                workers = "/".join(str(w["tasks"][key]["workers"]) for w in side["windows"])
                lines += [
                    f"| {name} | {task['max_us']:.3f} | {task['min_us']:.3f} | {task['mean_us']:.3f} "
                    f"| {env['max_us']:.3f}/{env['min_us']:.3f}/{env['mean_us']:.3f} "
                    f"| {task['core_us_per_window']:.3f} | {workers} |"
                ]
        lines.append("")
    lines += ["8:2完整CSA对开始/结束基线：", ""]
    for name, values in result["weighted_8_2_pct"].items():
        lines += [f"- {name}: {values['start']:+.3f}% / {values['end']:+.3f}%"]
    lines += ["", "8:2最大值变化对开始/结束基线（与均值分别计算，不混合评分）：", ""]
    for name, values in result["weighted_max_8_2_pct"].items():
        lines += [f"- {name}: {values['start']:+.3f}% / {values['end']:+.3f}%"]
    lines += ["", "状态通过仅覆盖本轮单卡真实层权重与合成历史、图重放和保护区；没有新增整模型token/DSpark验收。"]
    (ROOT / "RESULTS.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
