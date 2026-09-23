#!/usr/bin/env bash
set -eo pipefail
: "${TASK_DEVICE:?Submit this script through task-submit with one NPU device}"
if [[ "$TASK_DEVICE" == *,* ]]; then
  printf 'This reproducer requires one NPU device.\n' >&2
  exit 2
fi
workspace_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
source "$workspace_root/env.sh"
repro_output="${1:?Pass a new output directory}"
mkdir -p "$repro_output"
repro_output="$(realpath "$repro_output")"
python "$workspace_root/vllm-ascend-dsv4-pto/tests/pypto_test/repro_tdiv_high_precision.py" \
  --device "$TASK_DEVICE" --output-dir "$repro_output" \
  --isa-root "$workspace_root/simpler/build/pto-isa" \
  2>&1 | tee "$repro_output/run.log"
