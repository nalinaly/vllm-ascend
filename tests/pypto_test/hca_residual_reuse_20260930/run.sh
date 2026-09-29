#!/usr/bin/env bash
set -eo pipefail
: "${TASK_DEVICE:?请通过task-submit自动排队}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
tests="$(dirname "$root")"
history="${1:-131072}"
batch="${2:-16}"
out="$tests/results/hca_residual_reuse_20260930/h${history}_b${batch}"
prefix=/data/pyptouser/qinchuanyu/pto-eager/.cache/hca-residual-c6792787-20260930
test -f "$root/compile_base.json"
test -f "$root/compile_reuse.json"
test ! -e "$out"
# 首轮只跑ABBA，不继承旧脚本每候选十个进程的默认迭代。
CYCLES=1 bash "$tests/run_hca_ab_same_card.sh" "$out/abba" "$history" "$batch" \
    "base=$prefix/base" "reuse=$prefix/reuse" -- --iters 20 --warmup 5 --save-state
for side in base reuse; do
    bash "$tests/run_hca_compiled_case.sh" "$out/swimlane_$side" pto \
        --history "$history" --batch "$batch" --super-kernel 0 --operator-source "$prefix/$side" --swimlane
done
