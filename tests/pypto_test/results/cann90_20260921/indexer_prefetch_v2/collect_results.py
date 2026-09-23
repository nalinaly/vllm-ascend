"""Collect completed queue jobs and compare the frozen before/after CSA implementations."""

import hashlib
import json
import re
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[4]
sys.path.insert(0, str(REPO / "tests/pypto_test"))

from compare_dsv4_csa_profiles import summarize, weight_records_by_name


def read(path):
    return json.loads(path.read_text())


def main():
    manifest = read(ROOT / "manifest.json")
    before_sources = read(ROOT / "before_sources.json")
    queue = []
    for task in manifest["tasks"]:
        status = subprocess.run(["task-submit", "--status", task["task"]], text=True,
                                capture_output=True, check=True).stdout.strip()
        queue.append({"task": task["task"], "status": status})
    (ROOT / "queue_status.json").write_text(json.dumps(queue, indent=2) + "\n")
    if any("completed" not in task["status"] for task in queue):
        print(json.dumps({"status": "PENDING", "queue": queue}))
        return
    assert all("completed (exit=0)" in task["status"] for task in queue), queue
    for task in manifest["tasks"]:
        output = Path(task["output"])
        log = subprocess.run(["task-submit", "--log", task["task"]], text=True,
                             capture_output=True, check=True).stdout
        (output / "execution.log").write_text(log)
    for name, sha in manifest["sources"].items():
        assert hashlib.sha256((ROOT / "sources" / name).read_bytes()).hexdigest() == sha, name
    for name, sha in manifest["postprocessing_sources"].items():
        assert hashlib.sha256((REPO / name).read_bytes()).hexdigest() == sha, name
    assert all(not (ROOT / "sources" / name).exists() for name in manifest["deleted_sources"])
    profile = ROOT / "profile"
    log = (profile / "execution.log").read_text()
    builds = list(dict.fromkeys(re.findall(r"build_output/(_jit__decode_csa_tp1_attention_[^/\s]+)", log)))
    assert len(builds) == 3, builds
    mapping = REPO / "build_output" / builds[-1] / "kernel_config.py"
    with (profile / "comparison_named.log").open("w") as output:
        subprocess.run([sys.executable, "tests/pypto_test/compare_dsv4_csa_profiles.py",
                        "--root", str(profile), "--swimlane-kernel-config", str(mapping),
                        "--source-snapshot", str(ROOT / "sources")],
                       cwd=REPO, stdout=output, stderr=subprocess.STDOUT, check=True)
    (profile / "compile_sources.json").write_text(json.dumps({
        "before_profile": builds[0], "after_profile": builds[1], "after_dfx": builds[2],
    }, indent=2) + "\n")
    import torch
    torch.set_num_threads(2)
    cases = []
    for task in manifest["tasks"]:
        if task["phase"] != "full_compare":
            continue
        output = Path(task["output"])
        result = read(output / "full_compare.json")
        checks = [result["output"], result["topk"], *result["cache_comparisons"].values(),
                  *result["native_untouched"].values(), *result["pto_untouched"].values()]
        assert result["status"] == "PASS" and all(c["status"] == "PASS" for c in checks)
        old = torch.load(ROOT.parent / "indexer_page_load_v1" / output.name / "outputs.pt",
                         map_location="cpu", weights_only=False)
        new = torch.load(output / "outputs.pt", map_location="cpu", weights_only=False)
        same = {key: torch.equal(old[key].contiguous().view(torch.uint8), new[key].contiguous().view(torch.uint8))
                for key in ("pto", "pto_topk", "pto_scores")}
        assert all(same.values()), same
        cases.append({"case": output.name, "task": task["task"], "checks": len(checks),
                      "status": "PASS", "old_new_bitwise": same, "output": result["output"]})
    old_meta, old_perf = summarize(ROOT / "baseline", "pypto")
    new_meta, new_perf = summarize(profile, "pypto")
    conditions = {key: value == new_meta[key] for key, value in old_meta.items()
                  if key in new_meta and key not in ("weights", "trace_json")}
    conditions["weights"] = weight_records_by_name(old_meta["weights"]) == weight_records_by_name(new_meta["weights"])
    assert all(conditions.values()), {key: value for key, value in conditions.items() if not value}
    for path, expected in [(ROOT / "baseline/pypto", before_sources), (profile / "pypto", manifest["sources"]),
                           (profile / "swimlane", manifest["sources"])]:
        provenance = read(path / "source_snapshot.json")
        assert provenance["loaded_csa_sources"]
        assert all(expected[name] == sha for name, sha in provenance["loaded_csa_sources"].items())
    old_output = torch.load(ROOT / "baseline/pypto/output.pt", map_location="cpu", weights_only=True)["output"]
    new_output = torch.load(profile / "pypto/output.pt", map_location="cpu", weights_only=True)["output"]
    assert torch.equal(old_output.view(torch.uint8), new_output.view(torch.uint8))
    worker = []
    for directory in (ROOT.parent / "indexer_page_load_v1/profile", profile):
        events = read(directory / "swimlane/merged_swimlane.json")["traceEvents"]
        pid = next(e["pid"] for e in events if e.get("name") == "process_name" and e.get("ph") == "M"
                   and e["args"].get("name") == "Worker View")
        lanes = {}
        for name in ("indexer_score_topk_leaf_aic", "indexer_score_topk_leaf_aiv", "indexer_topk_query_merge"):
            durations = [e["args"]["kernel-duration-us"] for e in events if e.get("pid") == pid
                         and e.get("ph") == "X" and e["name"].startswith(name + "_spmd(")]
            assert durations, name
            lanes[name] = {"blocks": len(durations), "kernel_mean_us": statistics.mean(durations)}
        worker.append(lanes)
    before, after = old_perf["device_span_median_us"], new_perf["device_span_median_us"]
    result = {"status": "PASS", "status_scope": "Numerical and evidence consistency; see decision.json for the optimization decision.",
              "scope": manifest["scope"], "numerical_cases": cases,
              "numerical_checks": sum(c["checks"] for c in cases), "profile_conditions_equal": conditions,
              "profile_outputs_bitwise_equal": True,
              "same_session_profiles": {"before": old_perf, "after": new_perf},
              "device_span_median_us": {"before": before, "after": after},
              "speedup": before / after, "latency_reduction_percent": (1 - after / before) * 100,
              "swimlane_worker_kernels": {"previous_capture": worker[0], "after": worker[1]},
              "timing_note": "Both implementations captured in fresh processes on device 8 with five replays; previous DFX is separate diagnostic evidence."}
    (ROOT / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("status", "numerical_checks", "device_span_median_us", "speedup", "swimlane_worker_kernels")}, indent=2))


if __name__ == "__main__":
    main()
