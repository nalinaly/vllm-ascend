#!/usr/bin/env bash
set -eo pipefail
: "${TASK_DEVICE:?请通过task-submit自动排队}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
tests="$(dirname "$root")"
history="${1:-131072}"
batch="${2:-16}"
out="$tests/results/hca_qchain_schedule_20260930/h${history}_b${batch}"
prefix=/data/pyptouser/qinchuanyu/pto-eager/.cache/hca-qchain-c6792787-reuse-20260930
for side in base syncq syncboth; do test -f "$root/compile_$side.json"; done
test ! -e "$out"
CYCLES=1 bash "$tests/run_hca_ab_same_card.sh" "$out/abba" "$history" "$batch" \
    "base=$prefix/base" "syncq=$prefix/syncq" "syncboth=$prefix/syncboth" -- \
    --iters 20 --warmup 5 --save-state
for side in base syncq syncboth; do
    bash "$tests/run_hca_compiled_case.sh" "$out/swimlane_$side" pto \
        --history "$history" --batch "$batch" --super-kernel 0 --operator-source "$prefix/$side" --swimlane
done
