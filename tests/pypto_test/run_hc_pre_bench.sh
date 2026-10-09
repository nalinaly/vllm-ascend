#!/usr/bin/env bash
# 独立 HC_pre 单卡对照，设备由队列分配。
set -eo pipefail
: "${TASK_DEVICE:?通过 task-submit --device auto 提交}"
[[ "$TASK_DEVICE" =~ ^[0-9]+$ ]] || { printf '本用例只使用一张卡。\n' >&2; exit 2; }
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$(dirname "$repo_root")/env-dsv4-0251rc1.sh"
output_dir="${1:?指定新的输出目录}"
shift
mkdir -p "$output_dir"
output_dir="$(realpath "$output_dir")"
export ASCEND_RT_VISIBLE_DEVICES="$TASK_DEVICE"
export ASCEND_PROCESS_LOG_PATH="$output_dir/ascend"
mkdir -p "$ASCEND_PROCESS_LOG_PATH"
cd "$output_dir"
exec python "$repo_root/tests/pypto_test/dsv4_hc_pre_bench.py" --output "$output_dir" "$@"
