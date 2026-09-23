#!/usr/bin/env bash
set -eo pipefail
: "${TASK_DEVICE:?task-submit required}"
source /data/pyptouser/qinchuanyu/pto-eager/env.sh
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
cd /data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto
python tests/pypto_test/dsv4_csa_swa_rms_boundary.py --device "$TASK_DEVICE" --checkpoint /data/model/DeepSeek-V4-Flash-0731-w8a8 --output-dir tests/pypto_test/results/cann90_20260921/formal_swa_rms_v3 --steps 14 18 19 32 34 42 46
