#!/usr/bin/env bash
set -eo pipefail
: "${TASK_DEVICE:?请通过task-submit自动排队}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
tests="$(dirname "$root")"
history="${1:-131072}"
batch="${2:-16}"
out="$tests/results/hca_fullmask_20260930/h${history}_b${batch}"
prefix=/data/pyptouser/qinchuanyu/pto-eager/.cache/hca-fullmask-v2-2cb8714b-20260930
test -f "$root/compile_fullmask.json"
test ! -e "$out"
CYCLES=1 bash "$tests/run_hca_ab_same_card.sh" "$out/abba" "$history" "$batch" \
    "base=$prefix/base" "fullmask=$prefix/fullmask" -- --iters 20 --warmup 5 --save-state
for side in base fullmask; do
    bash "$tests/run_hca_compiled_case.sh" "$out/swimlane_$side" pto \
        --history "$history" --batch "$batch" --super-kernel 0 --operator-source "$prefix/$side" --swimlane
done
