#!/usr/bin/env bash
set -eo pipefail
: "${TASK_DEVICE:?Submit through task-submit --device auto}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 正式计时和跨版本完整状态已完成；这里只补同源核时与实际NZ绑定。
for case_spec in 131072:16 8192:24; do
    history="${case_spec%:*}"
    batch="${case_spec#*:}"
    sides=(baseline candidate)
    if [[ "$history" == 8192 ]]; then sides=(candidate baseline); fi
    for side in "${sides[@]}"; do
        bash "$root/run_side.sh" "$side" swimlane "$history" "$batch"
    done
done
