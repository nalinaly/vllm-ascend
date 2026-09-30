#!/usr/bin/env bash
set -eo pipefail
: "${TASK_DEVICE:?请通过task-submit自动排队}"
tests="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
prefix=/data/pyptouser/qinchuanyu/pto-eager/.cache/hca-ring-combo-4ffe0a53-20260930
for case in 131072:16:5 16384:4:1; do
    IFS=: read -r history batch cycles <<< "$case"
    bash "$tests/hca_pair_screen_20260930/run.sh" \
        "$tests/results/hca_softmax_ring_20260930/combo_h${history}_b${batch}" \
        --baseline "$prefix/base" --candidate "$prefix/ring" \
        --history "$history" --batch "$batch" --cycles "$cycles"
done
