#!/usr/bin/env bash
set -eo pipefail
: "${TASK_DEVICE:?Submit through task-submit}"
source /data/pyptouser/qinchuanyu/pto-eager/env.sh
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
cd /data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto
exec python tests/pypto_test/dsv4_csa_swa_window.py --device "$TASK_DEVICE" "$@"
