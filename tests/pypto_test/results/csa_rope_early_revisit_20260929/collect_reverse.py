"""只收集一次反序正式计时；复用已完成精度证据，不再读取全状态或采DFX。"""

import importlib.util
import json
import statistics
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    folder = ROOT / "reverse"
    task = (folder / "task.txt").read_text().strip()
    status = subprocess.check_output(["task-submit", "--status", task], text=True).strip()
    if status != "completed (exit=0)":
        raise RuntimeError(f"Wait for the same reverse task: {status}")
    spec = importlib.util.spec_from_file_location(
        "rope_reverse_pair", ROOT.parent / "csa_compiled_pair_20260929/analyze.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = json.loads((ROOT / "source.json").read_text())
    result = {"task": task, "task_status": status, "source": source, "cases": [],
              "accuracy_reference": "../evidence.json", "new_dfx": False, "new_state_snapshots": False}
    for history, batch in source["cases"]:
        case = {"history": history, "batch": batch, "sides": {}}
        for side in ("baseline", "candidate"):
            path = folder / f"h{history}_b{batch}/timing/{side}"
            report = json.loads((path / "report.json").read_text())
            if (report["source"], report["variant"], report["history"], report["batch"], report["side"]) != (
                    source["source_prefix"] + "-" + side, source["variant"], history, batch, "pto"):
                raise ValueError("Frozen reverse source or case changed")
            value = module.analyze_side(path)
            value["samples_us"] = report["timing"]["samples_us"]
            value["stdev_us"] = statistics.stdev(value["samples_us"])
            value["over_p50_5pct"] = sum(v > 1.05 * value["us_p50"] for v in value["samples_us"])
            case["sides"][side] = value
        a, b = (case["sides"][s] for s in ("baseline", "candidate"))
        for field in ("device", "cann", "requested"):
            if a[field] != b[field]:
                raise ValueError(f"Within-pair configuration mismatch: {field}")
        case["csa_change_pct"] = 100 * (b["mean_us"] / a["mean_us"] - 1)
        result["cases"].append(case)
    result["weighted_8_2_csa_pct"] = sum(w * c["csa_change_pct"] for w, c in zip((.8, .2), result["cases"]))
    (folder / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print("Reverse 8:2", result["weighted_8_2_csa_pct"])
    for c in result["cases"]:
        print(c["history"], c["batch"], c["csa_change_pct"],
              {s: {k: v[k] for k in ("mean_us", "us_p50", "us_p95", "us_max", "over_p50_5pct")}
               for s, v in c["sides"].items()})


if __name__ == "__main__":
    main()
