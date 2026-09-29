#!/usr/bin/env bash
set -eo pipefail
: "${TASK_DEVICE:?请通过task-submit自动排队}"
tests="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cache=/data/pyptouser/qinchuanyu/pto-eager/.cache
for side in base flat; do
    if [[ "$side" == base ]]; then
        source_dir="$cache/hca-q-al1-slices-a7a9f314-20260930/base"
    else
        source_dir="$cache/hca-q-flat-gather-a7a9f314-20260930/flat"
    fi
    out="$tests/results/hca_q_flat_gather_20260930/h131072_b16/swimlane_$side"
    test ! -e "$out"
    bash "$tests/run_hca_compiled_case.sh" "$out" pto \
        --history 131072 --batch 16 --super-kernel 0 --operator-source "$source_dir" \
        --iters 20 --warmup 5 --swimlane
done
