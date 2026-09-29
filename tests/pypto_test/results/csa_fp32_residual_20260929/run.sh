#!/usr/bin/env bash
set -eo pipefail
: "${TASK_DEVICE:?Submit through task-submit --device auto}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
test -f "$root/compile_candidate.json"
for case_spec in 131072:16 8192:24; do
    history="${case_spec%:*}"
    batch="${case_spec#*:}"
    sides=(baseline candidate)
    if [[ "$history" == 8192 ]]; then sides=(candidate baseline); fi
    for side in "${sides[@]}"; do
        bash "$root/run_side.sh" "$side" timing "$history" "$batch"
    done
done
for side in baseline candidate; do
    bash "$root/run_side.sh" "$side" swimlane 131072 16
done
