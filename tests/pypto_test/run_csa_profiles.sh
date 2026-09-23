#!/usr/bin/env bash
# Run three fresh processes sequentially on the same queue-allocated NPU.
set -eo pipefail
: "${TASK_DEVICE:?Submit through task-submit}"
if [[ "$TASK_DEVICE" == *,* ]]; then
  printf 'CSA profiling uses one device.\n' >&2
  exit 2
fi
workspace_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
source "$workspace_root/env.sh"
cd "$workspace_root/vllm-ascend-dsv4-pto"
output_root="${1:?Pass an output directory}"
checkpoint="${2:?Pass the reference checkpoint}"
mkdir -p "$output_root/plog"
output_root="$(realpath "$output_root")"
export ASCEND_PROCESS_LOG_PATH="$output_root/plog"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
python tests/pypto_test/dsv4_csa_profile.py --variant native --device "$TASK_DEVICE" \
  --checkpoint "$checkpoint" --output-dir "$output_root/native" "${@:3}"
python tests/pypto_test/dsv4_csa_profile.py --variant pypto --device "$TASK_DEVICE" \
  --checkpoint "$checkpoint" --output-dir "$output_root/pypto" "${@:3}"
python tests/pypto_test/dsv4_csa_profile.py --variant pypto --swimlane --device "$TASK_DEVICE" \
  --checkpoint "$checkpoint" --output-dir "$output_root/swimlane" "${@:3}"
