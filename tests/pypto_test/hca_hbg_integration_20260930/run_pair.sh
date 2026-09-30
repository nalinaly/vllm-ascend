#!/usr/bin/env bash
# 原版 ring 与独立 HBG 入口：同卡短/长档对照，源码须事先冻结。
set -eo pipefail
: "${TASK_DEVICE:?通过 task-submit 分配单卡}"
[[ "$TASK_DEVICE" != *,* ]]
experiment_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
case_root="$(realpath "${1:?结果根目录，包含 legacy_ops 与 hbg_ops}")"
history="${2:?历史长度}"
shift 2
export PTO_CSA_RING_HEAP_MB=320
export VLLM_ASCEND_ENABLE_NZ=2
for runtime in tensormap_and_ringbuffer host_build_graph; do
    if [[ "$runtime" == tensormap_and_ringbuffer ]]; then
        source_dir="$case_root/legacy_ops"
        side=ring
        reference=()
    else
        source_dir="$case_root/hbg_ops"
        side=hbg
        reference=(--reference-state "$case_root/b4_${history}_ring/states.pt")
    fi
    export PYPTO_PROG_BUILD_DIR="$case_root/b4_${history}_${side}/build"
    bash "$experiment_dir/../run_hca_single_layer.sh" "$case_root/b4_${history}_${side}" \
        --batch 4 --history "$history" --runtime "$runtime" --operator-source "$source_dir" \
        --weight-nz-mode 2 --service-graph --timing-iters 10 --timing-direct "${reference[@]}" "$@"
done
