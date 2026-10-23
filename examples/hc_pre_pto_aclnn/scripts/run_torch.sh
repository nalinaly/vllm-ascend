#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${TASK_DEVICE:?Submit with task-submit --device auto}"
[[ "$TASK_DEVICE" =~ ^[0-9]+$ ]] || exit 2
vendor="$(realpath "${1:?Usage: run_torch.sh VENDOR INPUT OUTPUT [NATIVE_VENDOR NATIVE_LIBRARY]}")"
input="$(realpath "${2:?INPUT}")"
output="${3:?OUTPUT}"
args=()
export ASCEND_CUSTOM_OPP_PATH="$vendor"
if [[ $# == 5 ]]; then
    export ASCEND_CUSTOM_OPP_PATH="$vendor:$(realpath "$4")"
    args+=(--native-library "$(realpath "$5")")
fi
export LD_LIBRARY_PATH="$vendor/op_api/lib:${LD_LIBRARY_PATH:-}"
export ASCEND_RT_VISIBLE_DEVICES="$TASK_DEVICE"
exec python "$root/tests/test_torch.py" --binding "$root/build/torch/bytenn_pto_ops_npu.so" \
    --input "$input" --output "$output" "${args[@]}"
