#!/usr/bin/env bash
set -eo pipefail
: "${TASK_DEVICE:?必须通过 task-submit 自动分配单卡}"
[[ "$TASK_DEVICE" != *,* ]]
source /data/pyptouser/qinchuanyu/pto-eager/env-dsv4-0251rc1.sh
case_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ASCEND_RT_VISIBLE_DEVICES="$TASK_DEVICE"
export ASCEND_PROCESS_LOG_PATH="$case_dir/device_run/ascend"
export PYPTO_PROG_BUILD_DIR="$case_dir/device_run/build"
mkdir -p "$ASCEND_PROCESS_LOG_PATH"
cd "$case_dir"
exec python -u "$case_dir/run_single_card.py" > "$case_dir/device.log" 2>&1
