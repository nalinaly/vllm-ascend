"""Compare preserved CSA profiles and numerical results on CPU."""

import hashlib
import json
import statistics
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[4]
BASELINE = ROOT.parent / "csa_profile_v1"
PROFILE = ROOT / "profile"
sys.path.insert(0, str(REPO / "tests/pypto_test"))
from compare_dsv4_csa_profiles import weight_records_by_name


def read(path):
    return json.loads(path.read_text())


def bitwise(left, right):
    return torch.equal(left.contiguous().view(torch.uint8), right.contiguous().view(torch.uint8))


def lanes(root):
    events = read(root / "swimlane/merged_swimlane.json")["traceEvents"]
    worker = next(e["pid"] for e in events if e.get("ph") == "M"
                  and e.get("name") == "process_name" and e["args"].get("name") == "Worker View")
    result = {}
    for name in ("indexer_score_topk_leaf_aic", "indexer_score_topk_leaf_aiv", "indexer_topk_query_merge"):
        selected = [e for e in events if e.get("ph") == "X" and e.get("pid") == worker
                    and e.get("name", "").startswith(name + "_spmd(")]
        assert selected, name
        durations = [e["args"]["kernel-duration-us"] for e in selected]
        result[name] = {
            "blocks": len(selected), "kernel_mean_us": statistics.mean(durations),
            "kernel_min_us": min(durations), "kernel_max_us": max(durations),
            "first_start_us": min(e["ts"] for e in selected),
            "last_finish_us": max(e["ts"] + e["dur"] for e in selected),
        }
    return result


def main():
    torch.set_num_threads(2)
    manifest = read(ROOT / "manifest.json")
    before_sources = read(ROOT / "before_sources.json")
    for name, expected in manifest["sources"].items():
        assert hashlib.sha256((REPO / name).read_bytes()).hexdigest() == expected, name
    for name, expected in manifest["postprocessing_sources"].items():
        assert hashlib.sha256((REPO / name).read_bytes()).hexdigest() == expected, name
    assert all(not (REPO / name).exists() for name in manifest["deleted_sources"])
    changed = [name for name, sha in before_sources.items() if name in manifest["sources"] and manifest["sources"][name] != sha]
    assert changed == ["vllm_ascend/ops/pypto/deepseek_v4_flash_dspark/decode_indexer.py"], changed
    before, after = read(BASELINE / "comparison.json"), read(PROFILE / "comparison.json")
    assert before["status"] == after["status"] == "PASS"
    old_meta, new_meta = read(BASELINE / "pypto/run_metadata.json"), read(PROFILE / "pypto/run_metadata.json")
    fields = ("checkpoint", "layer", "batch", "query_tokens_per_request", "history", "seed", "device",
              "tp", "layout", "hidden_dtype", "cudagraph_mode", "decode_aclgraph_replays", "dfx_level",
              "dependency_collection", "profiling_scope", "profiler_with_stack", "profiler_record_shapes",
              "profiler_with_memory", "profiler_level", "input_hashes", "weights", "quant_methods")
    conditions = {name: old_meta[name] == new_meta[name] for name in fields}
    conditions["weights"] = weight_records_by_name(old_meta["weights"]) == weight_records_by_name(new_meta["weights"])
    assert all(conditions.values()), conditions
    exact_outputs = {}
    for variant in ("native", "pypto", "swimlane"):
        a = torch.load(BASELINE / variant / "output.pt", map_location="cpu", weights_only=True)["output"]
        b = torch.load(PROFILE / variant / "output.pt", map_location="cpu", weights_only=True)["output"]
        exact_outputs[variant] = bitwise(a, b)
    assert all(exact_outputs.values()), exact_outputs
    numerical = read(ROOT / "numerical_comparison.json")
    assert len(numerical) == 2 and all(row["status"] == "PASS" and all(row["old_new_bitwise"].values()) for row in numerical)
    old_lanes, new_lanes = lanes(BASELINE), lanes(PROFILE)
    old_us, new_us = before["pypto"]["device_span_median_us"], after["pypto"]["device_span_median_us"]
    result = {
        "status": "PASS", "scope": manifest["scope"], "changed_sources": changed,
        "profile_conditions_equal": conditions, "profile_outputs_bitwise_equal": exact_outputs,
        "numerical_cases": numerical, "numerical_checks": sum(row["checks"] for row in numerical),
        "pto_device_span_median_us": {"before": old_us, "after": new_us},
        "pto_speedup": old_us / new_us, "pto_reduction_percent": (1 - new_us / old_us) * 100,
        "native_device_span_median_us": {"before": before["native"]["device_span_median_us"], "after": after["native"]["device_span_median_us"]},
        "current_pto_over_native": after["device_span_pto_over_native"],
        "swimlane_worker_kernels": {"before": old_lanes, "after": new_lanes},
        "key_gm_loads_per_score_tile": {"before": 384, "after": 12, "bytes_per_tile": 49152},
        "timing_scope": "Three fixed-input graph replays including Native metadata producers; DFX is a separate eager diagnostic capture.",
        "limitations": "Reference single layer only; no full-model, target-weight, continuous-100-step, or complete P3 acceptance claim.",
    }
    (ROOT / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
