#!/usr/bin/env bash
set -eo pipefail
: "${TASK_DEVICE:?请通过task-submit自动排队}"
tests="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cache=/data/pyptouser/qinchuanyu/pto-eager/.cache
for side in base_m4 mix_m4_box8_v3 mix_m4_box8_d512_v3; do
    if [[ "$side" == base_m4 ]]; then
        source_dir="$cache/hca-flat-sync-ad0e6bbe-20260930/base"
    else
        source_dir="$cache/hca-pre-ub-ad0e6bbe-20260930/$side"
    fi
    out="$tests/results/hca_pre_ub_20260930/dfx/$side"
    test ! -e "$out"
    bash "$tests/run_hca_compiled_case.sh" "$out" pto --history 131072 --batch 16 \
        --super-kernel 0 --operator-source "$source_dir" --iters 20 --warmup 5 --swimlane
done
