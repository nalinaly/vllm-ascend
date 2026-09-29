#!/usr/bin/env bash
set -eo pipefail
: "${TASK_DEVICE:?请通过task-submit自动排队}"
tests="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cache=/data/pyptouser/qinchuanyu/pto-eager/.cache
for side in base pair_hybrid_reload pair_three_slots; do
    if [[ "$side" == base ]]; then
        source_dir="$cache/hca-flat-sync-ad0e6bbe-20260930/base"
    else
        source_dir="$cache/hca-pair-query-ad0e6bbe-20260930/$side"
    fi
    out="$tests/results/hca_pair_query_20260930/dfx_refinements/swimlane_$side"
    test ! -e "$out"
    bash "$tests/run_hca_compiled_case.sh" "$out" pto --history 131072 --batch 16 \
        --super-kernel 0 --operator-source "$source_dir" --iters 20 --warmup 5 --swimlane
done
