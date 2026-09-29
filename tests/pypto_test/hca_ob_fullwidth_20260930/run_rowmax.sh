#!/usr/bin/env bash
set -eo pipefail
: "${TASK_DEVICE:?请通过task-submit排队}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
tests="$(dirname "$root")"
prefix=/data/pyptouser/qinchuanyu/pto-eager/.cache/hca-ob-fullwidth-0742f07c-20260930-v7
test -f "$root/compile_full_m96_rowmax.json"
history="${1:?history}"
batch="${2:?batch}"
shift 2
bash "$tests/hca_pair_screen_20260930/run.sh" \
    "$tests/results/hca_ob_fullwidth_20260930/rowmax_reference_h${history}_b${batch}" \
    --baseline "$prefix/full_m96" --candidate "$prefix/full_m96_rowmax" \
    --history "$history" --batch "$batch" "$@"
bash "$tests/hca_pair_screen_20260930/run.sh" \
    "$tests/results/hca_ob_fullwidth_20260930/rowmax_performance_h${history}_b${batch}" \
    --baseline "$prefix/base" --candidate "$prefix/full_m96_rowmax" \
    --history "$history" --batch "$batch" --diagnostic-output-differences "$@"
