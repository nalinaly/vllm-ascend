#!/usr/bin/env bash
# Run the existing PyPTO kernel integration cases under a task-submit allocation.
set -eo pipefail

: "${TASK_DEVICE:?Submit this script through task-submit with one NPU device}"
workspace_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
source "$workspace_root/env.sh"
if [[ "$TASK_DEVICE" == *,* ]]; then
  printf 'This runtime check requires exactly one allocated device.\n' >&2
  exit 2
fi
output_dir="${1:?Pass an output directory for logs and JUnit results}"
mkdir -p "$output_dir/plog"
output_dir="$(realpath "$output_dir")"

cd "$PTO_EAGER_ROOT/pypto"
source .claude/skills/testing/load-env.sh
source .github/scripts/kernel-mode-cases.sh capture
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export PYTHONPATH="$PWD/python:$PWD/build/torch_npu_tests${PYTHONPATH:+:$PYTHONPATH}"
export ASCEND_PROCESS_LOG_PATH="$output_dir/plog"

"$PTO_EAGER_ROOT/.venv/bin/python" -m pytest \
  tests/st/runtime/kernel/test_kernel_shutdown.py \
  tests/st/runtime/kernel/test_jit_eager.py \
  'tests/st/runtime/kernel/test_torch_launch.py::test_torch_kernel_launch[delayed-1]' \
  'tests/st/runtime/kernel/test_torch_ops.py::test_torch_ops[1-compile]' \
  "${cases[@]}" \
  --platform a2a3 --device "$TASK_DEVICE" -q \
  --junitxml="$output_dir/kernel-runtime.xml" \
  "${@:2}" \
  2>&1 | tee "$output_dir/kernel-runtime.log"
