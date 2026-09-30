#!/usr/bin/env bash
set -eo pipefail
experiment_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
experiment_repo="$(cd "$experiment_dir/../../.." && pwd)"
source "$(dirname "$experiment_repo")/env-dsv4-0251rc1.sh"
experiment_root="$experiment_repo/tests/pypto_test/results/atomic_add_seven_20260930"
experiment_output="$(realpath -m "${1:?output}")"
experiment_atomic="${2:?atomic}"
shift 2
mkdir -p "$experiment_output/ascend"
export VLLM_ASCEND_PTO_CSA_ATOMIC_ADD="$experiment_atomic"
export PTO_CSA_VARIANT=performance
export PTO_CSA_RUNTIME=tensormap_and_ringbuffer
export VLLM_ASCEND_ENABLE_NZ=2
export HCCL_DETERMINISTIC=false
export PTO_CSA_RING_HEAP_MB=320
export ASCEND_PROCESS_LOG_PATH="$experiment_output/ascend"
export PYPTO_PROG_BUILD_DIR="$experiment_root/build_atomic$experiment_atomic"
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_BUFFSIZE=1800
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_BATCH_INVARIANT=0
if [[ " $* " != *" --compile-only "* ]]; then
  : "${TASK_DEVICE:?必须通过task-submit分配单卡}"
  [[ "$TASK_DEVICE" != *,* ]]
  export ASCEND_RT_VISIBLE_DEVICES="$TASK_DEVICE"
fi
cd "$experiment_output"
exec python -u "$experiment_dir/run.py" --output "$experiment_output" \
  --source "$experiment_root/frozen_ops_pypto" --atomic "$experiment_atomic" "$@"
