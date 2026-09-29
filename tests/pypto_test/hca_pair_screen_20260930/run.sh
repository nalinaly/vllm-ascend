#!/usr/bin/env bash
set -eo pipefail
: "${TASK_DEVICE:?请通过task-submit自动排队}"
tests="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
repo="$(cd "$tests/../.." && pwd)"
source "$(dirname "$repo")/env-dsv4-0251rc1.sh"
out="$(realpath -m "${1:?结果目录}")"
shift
test ! -e "$out/report.json"
mkdir -p "$out/ascend"
export ASCEND_PROCESS_LOG_PATH="$out/ascend"
export VLLM_ASCEND_ENABLE_NZ=2
export PTO_CSA_VARIANT=performance
export VLLM_ASCEND_PTO_CSA_ATOMIC_ADD=0
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_BUFFSIZE=1800
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_BATCH_INVARIANT=0
export ASCEND_OPP_PATH="$(python "$tests/dsv4_hca_prepare_opp.py" --destination "$out/opp_root")"
mkdir -p "$ASCEND_OPP_PATH/static_kernel"
cd "$out"
exec python "$tests/hca_pair_screen_20260930/pair.py" --output "$out" "$@"
