# SPDX-License-Identifier: Apache-2.0
"""Compare fresh no-hooks, hooks-off and hooks-on SK runs without another NPU run."""

import argparse
import json
from pathlib import Path

import torch
from compare_superkernel import read_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("no_hooks", type=Path)
    parser.add_argument("hooks_off", type=Path)
    parser.add_argument("hooks_on", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = {name: getattr(args, name).resolve() for name in ("no_hooks", "hooks_off", "hooks_on")}
    runs, reports = {}, {}
    for name, path in paths.items():
        runs[name], reports[name] = read_run(path, True)
        report = reports[name]
        enabled = int(name == "hooks_on")
        assert report["fusion"]["early_start"] == enabled, report["fusion"]
        assert report["options"]["super_kernel_optimize_options"] == {"early_start": enabled}
        assert report["tokens"] == 96 and report["neighbours"] == "ascendc"
        assert report["timing"]["samples"] == 30 and report["timing"]["calls_per_sample"] == 8
        for key in ("torch", "torch_npu", "device"):
            assert report[key] == reports["no_hooks"][key], key
        for key in ("static_kernel_compile", "super_kernel_optimize", "clone_input"):
            assert report["options"][key] == reports["no_hooks"]["options"][key], key

    equality = []
    for replay in range(2):
        reference = torch.load(paths["no_hooks"] / f"result/replay-{replay}.pt", weights_only=True)
        for name in ("hooks_off", "hooks_on"):
            outputs = torch.load(paths[name] / f"result/replay-{replay}.pt", weights_only=True)
            checked = {key: torch.equal(outputs[key], reference[key]) for key in ("mixed", "post", "comb")}
            assert all(checked.values()), (name, replay, checked)
            equality.append({"mode": name, "replay": replay, "bitwise_equal": checked})

    metrics = ("device_scope_span", "device_kernel_sum", "compiled_call_event_window")
    changes = {
        f"{current}_vs_{base}": {
            metric: (runs[current][metric]["mean_us"] / runs[base][metric]["mean_us"] - 1) * 100 for metric in metrics
        }
        for current, base in (("hooks_off", "no_hooks"), ("hooks_on", "hooks_off"))
    }
    result = {
        "runs": runs,
        "change_percent": changes,
        "output_equality": equality,
        "within_30_percent_device_scope": all(abs(v["device_scope_span"]) <= 30 for v in changes.values()),
        "measurement": "30 samples of 8 calls; profiler SK duration excludes framework ForeachCopy. "
        "Event windows include framework work and submission gaps, in a separate capture phase. "
        "Native Early-Start capability/entry flags are audited, not direct pipeline-overlap measurements.",
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"change_percent": changes, "output_equality": equality}, indent=2))


if __name__ == "__main__":
    main()
