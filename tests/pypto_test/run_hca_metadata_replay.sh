#!/usr/bin/env bash
# 单卡服务图同地址 metadata A→B→A；设备由队列分配。
set -eo pipefail
: "${TASK_DEVICE:?请通过 task-submit 提交}"
hca_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$(dirname "$hca_repo")/env-dsv4-0251rc1.sh"
hca_output="$(realpath -m "${1:?指定结果目录}")"
shift
mkdir -p "$hca_output/ascend"
export ASCEND_RT_VISIBLE_DEVICES="$TASK_DEVICE"
export ASCEND_PROCESS_LOG_PATH="$hca_output/ascend"
export VLLM_ASCEND_ENABLE_NZ=2
export PTO_CSA_VARIANT=performance
export VLLM_ASCEND_PTO_CSA_ATOMIC_ADD=0
cd "$hca_output"
exec python "$hca_repo/tests/pypto_test/dsv4_hca_metadata_replay.py" --output "$hca_output" "$@"
