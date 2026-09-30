#!/usr/bin/env bash
set -eo pipefail
: "${TASK_DEVICE:?请通过 task-submit 分配单卡}"
[[ "$TASK_DEVICE" != *,* ]]
source /data/pyptouser/qinchuanyu/pto-eager/env-dsv4-0251rc1.sh
case_output="$(realpath -m "${1:?结果目录}")"
shift
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$case_output/ascend"
export ASCEND_RT_VISIBLE_DEVICES="$TASK_DEVICE"
export ASCEND_PROCESS_LOG_PATH="$case_output/ascend"
export PYPTO_PROG_BUILD_DIR="$case_output/build"
export PTO_CSA_RING_HEAP_MB=320
export PTO_CSA_RUNTIME=host_build_graph
export PTO_CSA_VARIANT=performance
export VLLM_ASCEND_ENABLE_NZ=2
export VLLM_ASCEND_PTO_CSA_ATOMIC_ADD=0
export HCCL_DETERMINISTIC=true
cd "$case_output"
exec python -u "$script_dir/run.py" --output "$case_output" "$@"
