"""CPU audit of redundancy cleanup against saved pre-change evidence."""

import csv
import hashlib
import json
import statistics
from pathlib import Path

import torch


def read(path):
    return json.loads(path.read_text())


def profile_counts(directory):
    files = list(directory.rglob("kernel_details.csv"))
    assert len(files) == 1, files
    with files[0].open() as stream:
        rows = list(csv.DictReader(stream))
    return {
        "path": str(files[0]),
        "simpler_aicpu": sum("simpler_aicpu_kernel_exec" in row["Name"] for row in rows),
        "aicore_kernel_mode": sum("aicore_kernel_mode" in row["Name"] for row in rows),
    }


def failures(report):
    return [
        {"phase": phase, "step": step["step"], "check": name,
         "mismatches": check.get("mismatches"), "positions": check.get("first_mismatches")}
        for phase, steps in report["phases"].items()
        for step in steps
        for name, check in step["checks"].items()
        if check["status"] != "PASS"
    ]


torch.set_num_threads(2)
root = Path(__file__).resolve().parent
base = root.parent
repo = root.parents[4]
manifest = read(root / "manifest.json")
states = {row["task"]: row["state"] for row in read(root / "queue_snapshot.json")}
report = {
    "scope": "Redundancy cleanup equivalence; existing continuous numerical failures are not P3 PASS",
    "source_hashes_match": all(
        hashlib.sha256((repo / path).read_bytes()).hexdigest() == expected
        for path, expected in manifest["sources"].items()
    ),
    "full_compare": [], "native_adapters": [], "continuous": [], "cross_stream": [], "pending": [],
}
for task in manifest["tasks"]:
    output = Path(task["output"])
    phase = task["phase"]
    state = states[task["task"]]
    if not state.startswith("completed"):
        report["pending"].append({"task": task["task"], "case": output.name, "state": state})
        continue
    row = {"case": output.name, "task": task["task"], "queue_state": state}
    if phase == "full_compare":
        current = read(output / "full_compare.json")
        old_dir = base / f"reference_matrix_v5/b{task['batch']}_h{task['history']}"
        old = torch.load(old_dir / "outputs.pt", map_location="cpu", weights_only=False)
        new = torch.load(output / "outputs.pt", map_location="cpu", weights_only=False)
        checks = [current["output"], current["topk"], *current["cache_comparisons"].values(),
                  *current["native_untouched"].values(), *current["pto_untouched"].values()]
        row.update(status=current["status"], checks=len(checks),
                   all_checks_pass=all(check["status"] == "PASS" for check in checks),
                   native_rope_zero_copy=current["native_rope_zero_copy"],
                   output=current["output"],
                   old_new_bitwise={key: torch.equal(old[key].contiguous().view(torch.uint8),
                                                     new[key].contiguous().view(torch.uint8))
                                    for key in ("pto", "pto_topk", "pto_scores")})
        row["equivalent"] = (state == "completed (exit=0)" and row["status"] == "PASS"
                             and row["all_checks_pass"] and row["native_rope_zero_copy"]
                             and all(row["old_new_bitwise"].values()))
    elif phase == "native_adapters":
        current = read(output / "native_adapters.json")
        row.update(results=current, equivalent=state == "completed (exit=0)" and current["status"] == "PASS")
    elif phase == "continuous":
        current = read(output / "continuous.json")
        old_dir = base / f"indexer_pool_order_v1/b{task['batch']}"
        old = read(old_dir / "continuous.json")
        matching = {key: current[key] == old[key] for key in
                    ("batch", "history", "seed", "steps", "fixture_capacity", "trajectory")}
        fingerprints = {}
        for mode, steps in current["phases"].items():
            previous = old["phases"][mode]
            fingerprints[mode] = {
                "steps": len(steps), "previous_steps": len(previous),
                "exact": len(steps) == len(previous) and all(
                    step["pto_fingerprint"] == before["pto_fingerprint"]
                    and step["starts"] == before["starts"] and step["accepted"] == before["accepted"]
                    for step, before in zip(steps, previous)
                ),
            }
        row.update(status=current["status"], previous_status=old["status"],
                   fixture_matches=matching, fingerprints=fingerprints,
                   failures=failures(current), previous_failures=failures(old))
        row["equivalent"] = (all(matching.values()) and all(item["exact"] for item in fingerprints.values())
                             and current["status"] == old["status"]
                             and row["failures"] == row["previous_failures"]
                             and state == f"completed (exit={0 if current['status'] == 'PASS' else 1})")
        if current["status"] == "PASS":
            row["profile_before"] = profile_counts(old_dir / "graph_profile")
            row["profile_after"] = profile_counts(output / "graph_profile")
            row["profile_reduction_verified"] = all(
                row["profile_before"][key] == 9 and row["profile_after"][key] == 8
                for key in ("simpler_aicpu", "aicore_kernel_mode")
            )
            row["equivalent"] &= row["profile_reduction_verified"]
    elif phase == "cross_stream":
        current = read(output / "cross_stream.json")
        full = read(output / "full_replay.json")
        checks = [check for case in full["cases"] for check in case["checks"].values()]
        row.update(status=current["status"], checks=len(checks),
                   all_checks_pass=all(check["status"] == "PASS" for check in checks),
                   largest_delay_pending_after_submit=current["largest_delay_pending_after_submit"],
                   delay_work_exact=current["delay_work_exact"],
                   median_ms={str(n): statistics.median(
                       frontier["producer_duration_ms"] for submission in current["submissions"][2:]
                       if submission["delay_iterations"] == n for frontier in submission["frontiers"]
                   ) for n in current["requested_delay_iterations"]})
        row["equivalent"] = (state == "completed (exit=0)" and current["status"] == "PASS"
                             and full["status"] == "PASS" and row["all_checks_pass"]
                             and row["largest_delay_pending_after_submit"] and row["delay_work_exact"])
    report[phase].append(row)

complete = [row for phase in ("full_compare", "native_adapters", "continuous", "cross_stream")
            for row in report[phase]]
report["redundancy_regression_status"] = (
    "FAIL" if not report["source_hashes_match"] or not all(row["equivalent"] for row in complete)
    else "PENDING" if report["pending"] else "PASS"
)
(root / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({"status": report["redundancy_regression_status"],
                  "source_hashes_match": report["source_hashes_match"],
                  "completed": len(complete), "pending": len(report["pending"]),
                  "cases": {phase: len(report[phase]) for phase in
                            ("full_compare", "native_adapters", "continuous", "cross_stream")},
                  "failed_equivalence": [row["case"] for row in complete if not row["equivalent"]]}, indent=2))
