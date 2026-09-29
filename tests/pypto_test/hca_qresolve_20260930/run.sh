#!/usr/bin/env bash
set -eo pipefail
: "${TASK_DEVICE:?请通过task-submit自动排队}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
tests="$(dirname "$root")"
prefix=/data/pyptouser/qinchuanyu/pto-eager/.cache/hca-qresolve-a7a9f314-20260930
for side in late_dq early_q_late_dq; do
    test -f "$root/compile_$side.json"
    bash "$tests/hca_pair_screen_20260930/run.sh" \
        "$tests/results/hca_qresolve_20260930/pair_${side}_h131072_b16" \
        --baseline "$prefix/base" --candidate "$prefix/$side" --history 131072 --batch 16
done
