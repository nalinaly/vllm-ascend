#!/usr/bin/env bash
set -eo pipefail
: "${TASK_DEVICE:?请通过task-submit自动排队}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
tests="$(dirname "$root")"
cache=/data/pyptouser/qinchuanyu/pto-eager/.cache
for side in pair_hybrid_reload pair_three_slots; do
    test -f "$root/compile_$side.json"
    bash "$tests/hca_pair_screen_20260930/run.sh" \
        "$tests/results/hca_pair_query_20260930/${side}_h131072_b16" \
        --baseline "$cache/hca-flat-sync-ad0e6bbe-20260930/base" \
        --candidate "$cache/hca-pair-query-ad0e6bbe-20260930/$side" --history 131072 --batch 16
done
