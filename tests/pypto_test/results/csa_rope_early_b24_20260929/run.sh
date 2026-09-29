#!/usr/bin/env bash
set -eo pipefail
: "${TASK_DEVICE:?Submit through task-submit --device auto}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
test -f "$root/../csa_rope_early_revisit_20260929/compile_candidate.json"
# Same frozen pair and T=144 as the two measured 8K/B24 rounds.
# Only add the missing 128K/B24 control; no other source changes.
for phase in timing swimlane; do
    for side in baseline candidate; do
        bash "$root/run_side.sh" "$side" "$phase" 131072 24
    done
done
