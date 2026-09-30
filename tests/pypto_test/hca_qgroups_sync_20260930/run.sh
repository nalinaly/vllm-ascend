#!/usr/bin/env bash
set -eo pipefail
: "${TASK_DEVICE:?请通过task-submit排队}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
tests="$(dirname "$root")"
prefix=/data/pyptouser/qinchuanyu/pto-eager/.cache/hca-qgroups-sync-0742f07c-20260930
side="${1:?候选}"
history="${2:?history}"
batch="${3:?batch}"
shift 3
test -f "$root/compile_$side.json"
bash "$tests/hca_pair_screen_20260930/run.sh" \
    "$tests/results/hca_qgroups_sync_20260930/${side}_h${history}_b${batch}" \
    --baseline "$prefix/base" --candidate "$prefix/$side" --history "$history" --batch "$batch" "$@"
