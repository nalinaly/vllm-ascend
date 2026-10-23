#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${TASK_DEVICE:?Submit with task-submit --device auto}"
[[ "$TASK_DEVICE" =~ ^[0-9]+$ ]] || exit 2
vendor="$(realpath "${1:?Usage: run_all.sh PTO_VENDOR NATIVE_VENDOR NATIVE_LIBRARY OUTPUT [--checkpoint CHECKPOINT]}")"
native="$(realpath "${2:?NATIVE_VENDOR}")"
export ASCEND_CUSTOM_OPP_PATH="$vendor:$native"
export LD_LIBRARY_PATH="$vendor/op_api/lib:${LD_LIBRARY_PATH:-}"
export ASCEND_RT_VISIBLE_DEVICES="$TASK_DEVICE"
library="${3:?NATIVE_LIBRARY}"
output="${4:?OUTPUT}"
shift 4
status=0
python "$root/tests/bench_all.py" --binding "$root/build/torch/bytenn_pto_ops_npu.so" \
    --native-library "$library" --output "$output" "$@" || status=$?
if [[ -f "$output/report.json" ]]; then
    python "$root/scripts/export_all.py" "$output/report.json"
    python "$root/scripts/summarize_all.py" "$output/report.json" --output "$output/summary.md"
fi
exit "$status"
