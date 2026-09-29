#!/usr/bin/env bash
set -eo pipefail
: "${TASK_DEVICE:?Submit through task-submit --device auto}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Fresh paired controls; never reuse the preceding N64 experiment's timing.
bash "$root/run_side.sh" baseline timing 131072 16 m16_baseline_start
bash "$root/run_side.sh" baseline swimlane 131072 16 m16_baseline_start
for name in kv_m16 kv_m16_sync; do
    bash "$root/run_side.sh" "$name" timing 131072 16
    bash "$root/run_side.sh" "$name" swimlane 131072 16
done
bash "$root/run_side.sh" baseline timing 131072 16 m16_baseline_end
