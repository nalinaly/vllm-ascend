#!/usr/bin/env bash
set -eo pipefail
: "${TASK_DEVICE:?Submit through task-submit --device auto}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Reuse the already compiled, immutable N64+sync package. Same T144 as 8K/B24:
# this resolves whether a token-shape branch also works at long context.
bash "$root/run_side.sh" baseline timing 131072 24 baseline_start
bash "$root/run_side.sh" baseline swimlane 131072 24 baseline_start
bash "$root/run_side.sh" kv_n64_sync timing 131072 24
bash "$root/run_side.sh" kv_n64_sync swimlane 131072 24
bash "$root/run_side.sh" baseline timing 131072 24 baseline_end
