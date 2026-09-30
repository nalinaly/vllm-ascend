#!/usr/bin/env bash
set -eo pipefail
: "${TASK_DEVICE:?必须通过 task-submit 分配单卡}"
[[ "$TASK_DEVICE" != *,* ]]
source /data/pyptouser/qinchuanyu/pto-eager/env-dsv4-0251rc1.sh
experiment_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
case_output="$(realpath "${1:?指定已冻结源码的结果目录}")"
shift
export ASCEND_RT_VISIBLE_DEVICES="$TASK_DEVICE"
export ASCEND_PROCESS_LOG_PATH="$case_output/ascend"
export PYPTO_PROG_BUILD_DIR="$case_output/build"
mkdir -p "$ASCEND_PROCESS_LOG_PATH"
exec python -u "$experiment_dir/run.py" --output "$case_output" "$@"
