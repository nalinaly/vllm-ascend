#!/usr/bin/env bash
# 同一卡按 Native/PTO/PTO/Native 取当前生产版本的长档对照。
set -eo pipefail
: "${TASK_DEVICE:?请通过task-submit自动排队}"
tests="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source_dir=/data/pyptouser/qinchuanyu/pto-eager/.cache/hca-q-al1-slices-a7a9f314-20260930/base
for pass in p1_native p2_pto p3_pto p4_native; do
    side="${pass#*_}"
    out="$tests/results/hca_native_anchor_20260930/h131072_b16_compatible/$pass"
    test ! -e "$out"
    args=()
    super_kernel=1
    if [[ "$side" == pto ]]; then
        args+=(--operator-source "$source_dir")
        # PyPTO kernel模式在SK图优化时报107017无效funcHandle；Native仍用完整部署口径。
        super_kernel=0
    fi
    bash "$tests/run_hca_compiled_case.sh" "$out" "$side" \
        --history 131072 --batch 16 --super-kernel "$super_kernel" --inplace-pass 1 \
        --iters 20 --warmup 5 --profile-replays 9 "${args[@]}"
done
