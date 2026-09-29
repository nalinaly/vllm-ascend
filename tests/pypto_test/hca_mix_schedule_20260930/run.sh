#!/usr/bin/env bash
set -eo pipefail
: "${TASK_DEVICE:?请通过task-submit自动排队}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
tests="$(dirname "$root")"
prefix=/data/pyptouser/qinchuanyu/pto-eager/.cache/hca-mix-schedule-9c2ede34-20260930
history="${1:?需要显式传入history，避免把队列附加的--device当作档位}"
batch="${2:?需要显式传入batch}"
[[ "$history" =~ ^[0-9]+$ && "$batch" =~ ^[0-9]+$ ]]
for side in mix_sync both_sync; do
    test -f "$root/compile_$side.json"
    bash "$tests/hca_pair_screen_20260930/run.sh" \
        "$tests/results/hca_mix_schedule_20260930/${side}_h${history}_b${batch}" \
        --baseline "$prefix/base" --candidate "$prefix/$side" --history "$history" --batch "$batch"
done
