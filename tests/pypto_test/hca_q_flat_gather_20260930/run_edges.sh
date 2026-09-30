#!/usr/bin/env bash
set -eo pipefail
: "${TASK_DEVICE:?请通过task-submit自动排队}"
tests="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cache=/data/pyptouser/qinchuanyu/pto-eager/.cache
common=(--baseline "$cache/hca-q-al1-slices-a7a9f314-20260930/base"
        --candidate "$cache/hca-q-flat-gather-a7a9f314-20260930/flat")
bash "$tests/hca_pair_screen_20260930/run.sh" \
    "$tests/results/hca_q_flat_gather_20260930/pair_h8192_b24" \
    "${common[@]}" --history 8192 --batch 24
# 18行含完整8行块与2行尾，图内活动请求3→2→1→3。
bash "$tests/hca_pair_screen_20260930/run.sh" \
    "$tests/results/hca_q_flat_gather_20260930/pair_h16507_b3" \
    "${common[@]}" --history 16507 --batch 3 --cycles 1 --padding-graph
