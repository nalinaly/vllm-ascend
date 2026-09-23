"""Validate and summarize the separate Native/PTO single-CSA profiles on CPU."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import shutil
import statistics
import subprocess
import sys
from collections import Counter
from decimal import Decimal
from pathlib import Path


def name_swimlane(root, kernel_config):
    """Use the actual capture's generated kernel table; never infer names by task order."""
    directory = root / "swimlane"
    tree = ast.parse(kernel_config.read_text())
    tables = [
        node.value for node in tree.body if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "KERNELS" for target in node.targets)
    ]
    assert len(tables) == 1 and isinstance(tables[0], ast.List)
    names = {}
    for node in tables[0].elts:
        values = {ast.literal_eval(key): value for key, value in zip(node.keys, node.values)}
        names[str(ast.literal_eval(values["func_id"]))] = ast.literal_eval(values["name"])
    deps = json.loads((directory / "deps.json").read_text())
    active_ids = {str(k) for task in deps["tasks"] for k in task["kernel_ids"] if k >= 0}
    assert active_ids <= names.keys(), active_ids - names.keys()
    copied_config = directory / "kernel_config_source.py"
    if kernel_config.resolve() != copied_config.resolve():
        shutil.copy2(kernel_config, copied_config)
    name_map = directory / "name_map.json"
    name_map.write_text(json.dumps({"callable_id_to_name": names}, indent=2) + "\n")
    merged = directory / "merged_swimlane.json"
    before = json.loads(merged.read_text())["traceEvents"]
    subprocess.run([
        sys.executable, "-m", "simpler_setup.tools.swimlane_converter",
        str(directory / "chip_swimlane_records.json"), "--func-names", str(name_map), "-o", str(merged),
    ], check=True)
    after = json.loads(merged.read_text())["traceEvents"]

    def timing(events):
        return [(e.get("ph"), e.get("pid"), e.get("tid"), e.get("ts"), e.get("dur")) for e in events]

    assert timing(before) == timing(after), "Adding names must not change recorded timing or dependencies"
    slices = [e for e in after if e.get("ph") == "X" and e.get("cat") == "event"]
    assert all(not e["name"].startswith(("func_", "task(")) for e in slices)
    return {
        "name_map": str(name_map), "kernel_config_source": str(copied_config),
        "kernel_config_sha256": hashlib.sha256(copied_config.read_bytes()).hexdigest(),
        "named_device_slices": len(slices), "timing_unchanged_by_name_mapping": True,
    }


def only(root, filename):
    paths = list(root.rglob(filename))
    if len(paths) != 1:
        raise ValueError(f"Expected one {filename} under {root}, found {paths}")
    return paths[0]


def weight_records_by_name(records):
    """Compare full records independently of loader set-iteration order."""
    result = {record["name"]: record for record in records}
    assert len(result) == len(records), "Duplicate weight names in profile metadata"
    return result


def summarize(root, variant):
    directory = root / variant
    metadata = json.loads((directory / "run_metadata.json").read_text())
    assert metadata["status"] == "PASS", metadata
    trace_path = Path(metadata["trace_json"])
    trace = json.loads(trace_path.read_text())
    events = trace if isinstance(trace, list) else trace["traceEvents"]
    markers = sorted(
        (event for event in events if event.get("ph") == "X" and event.get("name", "").startswith("csa.profile.replay.")),
        key=lambda event: Decimal(str(event["ts"])),
    )
    assert len(markers) == metadata["decode_aclgraph_replays"], len(markers)
    launches = [event for event in events if event.get("name") == "AscendCL@aclmdlRIExecuteAsync"]
    assert len(launches) == len(markers), (variant, len(launches), len(markers))
    kernel_path = only(directory / "torch_npu_trace", "kernel_details.csv")
    with kernel_path.open(encoding="utf-8-sig", newline="") as stream:
        kernels = list(csv.DictReader(stream))
    per_replay = []
    covered = 0
    for index, marker in enumerate(markers):
        start = Decimal(str(marker["ts"]))
        stop = start + Decimal(str(marker["dur"]))
        selected = [row for row in kernels if start <= Decimal(row["Start Time(us)"].strip()) < stop]
        assert selected, (variant, index)
        assert all(
            Decimal(row["Start Time(us)"].strip()) + Decimal(row["Duration(us)"].strip()) <= stop
            for row in selected
        ), "Replay marker must include stream synchronization"
        names = Counter(row["Name"] for row in selected)
        cores = Counter(row["Accelerator Core"] for row in selected)
        simpler_count = sum(count for name, count in names.items() if "simpler_aicpu_kernel_exec" in name)
        aicore_count = sum(count for name, count in names.items() if "aicore_kernel_mode" in name)
        assert (simpler_count, aicore_count) == ((1, 1) if variant == "pypto" else (0, 0))
        first = min(Decimal(row["Start Time(us)"].strip()) for row in selected)
        last = max(Decimal(row["Start Time(us)"].strip()) + Decimal(row["Duration(us)"].strip()) for row in selected)
        launches_here = [event for event in launches if start <= Decimal(str(event["ts"])) < stop]
        assert len(launches_here) == 1
        per_replay.append({
            "index": index, "device_span_us": float(last - first),
            "host_submit_to_sync_us": float(stop - start),
            "aclgraph_host_launch_us": float(launches_here[0]["dur"]),
            "device_task_count": len(selected), "core_counts": dict(cores),
            "kernel_counts": dict(names), "pto_submissions": simpler_count,
        })
        covered += len(selected)
    assert covered == len(kernels), (covered, len(kernels))
    return metadata, {
        "trace_json": str(trace_path), "trace_bytes": trace_path.stat().st_size,
        "trace_events": len(events), "kernel_details": str(kernel_path),
        "aclgraph_execute_count": len(launches), "per_replay": per_replay,
        "device_span_median_us": statistics.median(row["device_span_us"] for row in per_replay),
        "host_submit_to_sync_median_us": statistics.median(row["host_submit_to_sync_us"] for row in per_replay),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--swimlane-kernel-config", type=Path, help="Actual DFX capture's generated kernel_config.py")
    args = parser.parse_args()
    root = args.root.resolve()
    name_mapping = name_swimlane(root, args.swimlane_kernel_config) if args.swimlane_kernel_config else {}
    native_metadata, native = summarize(root, "native")
    pto_metadata, pto = summarize(root, "pypto")
    fields = (
        "checkpoint", "layer", "batch", "query_tokens_per_request", "history", "seed", "device", "tp",
        "layout", "hidden_dtype", "cudagraph_mode", "decode_aclgraph_replays", "dfx_level",
        "dependency_collection", "profiling_scope", "profiler_with_stack", "profiler_record_shapes",
        "profiler_with_memory", "profiler_level", "input_hashes", "weights", "quant_methods",
    )
    conditions = {field: native_metadata[field] == pto_metadata[field] for field in fields}
    conditions["weights"] = weight_records_by_name(native_metadata["weights"]) == weight_records_by_name(pto_metadata["weights"])
    assert all(conditions.values()), conditions
    import torch
    from dsv4_csa_full_compare import compare_tensor

    torch.set_num_threads(2)
    native_output = torch.load(root / "native/output.pt", map_location="cpu", weights_only=True)["output"]
    pto_output = torch.load(root / "pypto/output.pt", map_location="cpu", weights_only=True)["output"]
    numerical = compare_tensor(pto_output, native_output, 1e-2, 1e-2)
    assert numerical["status"] == "PASS", numerical
    swimlane = json.loads((root / "swimlane/run_metadata.json").read_text())
    assert swimlane["status"] == "PASS" and swimlane["captured_pypto_launches"] == 1
    assert swimlane["input_hashes"] == pto_metadata["input_hashes"]
    diagnostic_output = torch.load(root / "swimlane/output.pt", map_location="cpu", weights_only=True)["output"]
    assert torch.equal(diagnostic_output, pto_output)
    manifest = json.loads((root / "manifest.json").read_text())
    repo = Path(__file__).resolve().parents[2]
    source_match = all(hashlib.sha256((repo / path).read_bytes()).hexdigest() == sha for path, sha in manifest["sources"].items())
    assert source_match and all(not (repo / path).exists() for path in manifest["deleted_sources"])
    result = {
        "status": "PASS", "scope": native_metadata["scope"], "conditions_equal": conditions,
        "source_hashes_match": source_match, "native": native, "pypto": pto,
        "pto_vs_native_output": numerical,
        "device_span_pto_over_native": pto["device_span_median_us"] / native["device_span_median_us"],
        "swimlane": {key: swimlane[key] for key in (
            "captured_pypto_launches", "aicore_task_count", "converted_device_slices",
            "chip_swimlane_records", "deps_json", "merged_swimlane", "diagnostic_timing_only",
        )},
        "swimlane_name_mapping": name_mapping,
        "timing_method": "Three fixed-input single-layer replays; device span includes Native metadata producers, both streams, and CSA. Each marker ends after synchronize. DFX is a separate eager process and excluded from timing comparison.",
    }
    (root / "comparison.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
