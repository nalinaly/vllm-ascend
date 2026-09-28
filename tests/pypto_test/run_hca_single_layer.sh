#!/usr/bin/env bash
# 正式 HCA 单层对照；设备由队列分配。
set -eo pipefail
: "${TASK_DEVICE:?请通过 task-submit 提交}"
if [[ "$TASK_DEVICE" == *,* ]]; then
  printf 'HCA 单层对照只需要一张卡。\n' >&2
  exit 2
fi
hca_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$(dirname "$hca_repo")/env-dsv4-0251rc1.sh"
hca_output="$(realpath -m "${1:?指定结果目录}")"
shift
mkdir -p "$hca_output/ascend"
export ASCEND_RT_VISIBLE_DEVICES="$TASK_DEVICE"
export ASCEND_PROCESS_LOG_PATH="$hca_output/ascend"
export VLLM_ASCEND_ENABLE_NZ=1
export PTO_CSA_VARIANT=performance
export VLLM_ASCEND_PTO_CSA_ATOMIC_ADD=0
# 与上线 decode 口径对齐，取自
# vllm-ascend-main/tests/dsv4_perf_accuracy_20260827/runtime/decode/run_dp_template.sh。
# 这些都是进程级设置，Native 与 PTO 在同一进程里交替计时，因此两侧必然同配。
export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_BUFFSIZE=1800
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_BATCH_INVARIANT=0
cd "$hca_output"
exec python "$hca_repo/tests/pypto_test/dsv4_hca_single_layer.py" --output "$hca_output" "$@"
