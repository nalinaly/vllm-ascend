#!/usr/bin/env bash
set -eo pipefail
: "${TASK_DEVICE:?请通过task-submit自动排队}"
tests="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
prefix=/data/pyptouser/qinchuanyu/pto-eager/.cache/hca-flat-sync-ad0e6bbe-20260930
for side in base dq_sync; do
    out="$tests/results/hca_flat_sync_20260930/h131072_b16/swimlane_$side"
    test ! -e "$out"
    bash "$tests/run_hca_compiled_case.sh" "$out" pto --history 131072 --batch 16 \
        --super-kernel 0 --operator-source "$prefix/$side" --iters 20 --warmup 5 --swimlane
done
