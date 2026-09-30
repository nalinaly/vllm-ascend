#!/usr/bin/env bash
set -eo pipefail
: "${TASK_DEVICE:?请通过task-submit自动排队}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
tests="$(dirname "$root")"
prefix=/data/pyptouser/qinchuanyu/pto-eager/.cache/hca-post-gates-f14d8d90-20260930
side="${1:?候选名}"
history="${2:?history}"
batch="${3:?batch}"
shift 3
[[ "$history" =~ ^[0-9]+$ && "$batch" =~ ^[0-9]+$ ]]
test -f "$root/compile_$side.json"
baseline="$prefix/base"
if [[ "$side" == combo ]]; then
    baseline=/data/pyptouser/qinchuanyu/pto-eager/.cache/hca-qb-groups-f14d8d90-20260930/ready_v2
fi
bash "$tests/hca_pair_screen_20260930/run.sh" \
    "$tests/results/hca_post_gates_20260930/${side}_h${history}_b${batch}" \
    --baseline "$baseline" --candidate "$prefix/$side" \
    --history "$history" --batch "$batch" "$@"
