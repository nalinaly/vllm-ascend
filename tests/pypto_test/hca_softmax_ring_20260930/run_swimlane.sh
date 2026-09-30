#!/usr/bin/env bash
set -eo pipefail
: "${TASK_DEVICE:?请通过task-submit自动排队}"
tests="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
prefix=/data/pyptouser/qinchuanyu/pto-eager/.cache/hca-softmax-ring-v2-bef9f7fa-20260930
out="$tests/results/hca_softmax_ring_20260930/h131072_b16"
test ! -e "$out"
for side in base ring; do
    bash "$tests/run_hca_compiled_case.sh" "$out/swimlane_$side" pto \
        --history 131072 --batch 16 --super-kernel 0 --operator-source "$prefix/$side" --swimlane
done
