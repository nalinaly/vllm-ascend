#!/usr/bin/env bash
set -eo pipefail
: "${TASK_DEVICE:?请通过task-submit自动排队}"
tests="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
side="${1:?候选名}"
history="${2:-131072}"
batch="${3:-16}"
shift "$(( $# >= 3 ? 3 : $# ))"
cache=/data/pyptouser/qinchuanyu/pto-eager/.cache
out="$tests/results/hca_pre_ub_20260930/${side}_h${history}_b${batch}"
bash "$tests/hca_pair_screen_20260930/run.sh" "$out" \
    --baseline "$cache/hca-flat-sync-ad0e6bbe-20260930/base" \
    --candidate "$cache/hca-pre-ub-ad0e6bbe-20260930/$side" \
    --history "$history" --batch "$batch" "$@"
