#!/usr/bin/env bash
set -eo pipefail
: "${TASK_DEVICE:?Submit the complete pair through task-submit --device auto}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
test -f "$root/compat_pass.json"
for case_spec in 131072:16 8192:24; do
    history="${case_spec%:*}"
    batch="${case_spec#*:}"
    sides=(candidate baseline)
    if [[ "$history" == 8192 ]]; then sides=(baseline candidate); fi
    for phase in timing; do
        for side in "${sides[@]}"; do
            bash "$root/run_side.sh" "$side" "$phase" "$history" "$batch"
        done
    done
done
