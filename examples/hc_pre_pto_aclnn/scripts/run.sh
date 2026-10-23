#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${TASK_DEVICE:?Submit with task-submit --device auto}"
[[ "$TASK_DEVICE" =~ ^[0-9]+$ ]] || { echo 'One device required' >&2; exit 2; }
package="$(realpath "${1:?Usage: run.sh VENDOR_DIR INPUT_DIR OUTPUT_DIR [BUILD_DIR]}")"
input="$(realpath "${2:?INPUT_DIR}")"
output="${3:?OUTPUT_DIR}"
build="${4:-$root/build}"
[[ ! -e "$output" ]] || { echo 'Use a new output directory' >&2; exit 2; }
export ASCEND_CUSTOM_OPP_PATH="$package"
export LD_LIBRARY_PATH="$package/op_api/lib:${LD_LIBRARY_PATH:-}"
export ASCEND_RT_VISIBLE_DEVICES="$TASK_DEVICE"
"$build/run_aclnn" 0 "$input" "$output"
python "$root/tests/check_outputs.py" "$input" --output "$output"
