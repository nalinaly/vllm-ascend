#!/usr/bin/env bash
set -eo pipefail
: "${TASK_DEVICE:?Submit through task-submit --device auto}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for name in wb16_sync wb16_pool_sync; do
    test -f "$root/compile_${name}.json"
done
bash "$root/run_side.sh" baseline timing 131072 16 balanced_baseline_start
bash "$root/run_side.sh" baseline swimlane 131072 16 balanced_baseline_start
for name in wb16_sync wb16_pool_sync; do
    bash "$root/run_side.sh" "$name" timing 131072 16
    bash "$root/run_side.sh" "$name" swimlane 131072 16
done
bash "$root/run_side.sh" baseline timing 131072 16 balanced_baseline_end
