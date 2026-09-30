#!/usr/bin/env bash
set -eo pipefail
: "${TASK_DEVICE:?请通过task-submit排队}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
tests="$(dirname "$root")"
prefix=/data/pyptouser/qinchuanyu/pto-eager/.cache/hca-ob-fullwidth-0742f07c-20260930-v7
mode="${1:?reference或performance}"
history="${2:?history}"
batch="${3:?batch}"
shift 3
test -f "$root/compile_full_v7.json"
extra=()
if [[ "$mode" == reference ]]; then
    baseline="$prefix/grouped"
    test -f "$root/compile_grouped_v7.json"
    extra=(--cycles 1)
elif [[ "$mode" == performance ]]; then
    baseline="$prefix/base"
    extra=(--diagnostic-output-differences)
else
    exit 2
fi
bash "$tests/hca_pair_screen_20260930/run.sh" \
    "$tests/results/hca_ob_fullwidth_20260930/${mode}_h${history}_b${batch}" \
    --baseline "$baseline" --candidate "$prefix/full" \
    --history "$history" --batch "$batch" "${extra[@]}" "$@"
