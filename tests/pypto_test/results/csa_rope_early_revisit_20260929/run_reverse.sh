#!/usr/bin/env bash
set -eo pipefail
: "${TASK_DEVICE:?Submit through task-submit --device auto}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Only confirm the small formal timing gain; reuse the frozen kernels and
# completed accuracy/DFX proof, without new swimlanes or state snapshots.
for case_spec in 131072:16 8192:24; do
    history="${case_spec%:*}"
    batch="${case_spec#*:}"
    sides=(candidate baseline)
    if [[ "$history" == 8192 ]]; then sides=(baseline candidate); fi
    for side in "${sides[@]}"; do
        bash "$root/run_reverse_side.sh" "$side" timing "$history" "$batch"
    done
done
