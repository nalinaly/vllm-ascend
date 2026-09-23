#!/usr/bin/env bash
set -eo pipefail
source /data/pyptouser/qinchuanyu/pto-eager/env.sh
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
cd /data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto
python tests/pypto_test/dsv4_csa_main_compressor_replay.py --device "$TASK_DEVICE" --capture tests/pypto_test/results/cann90_20260921/step46_residual_v1/compressor_capture/b40/main_compressor_boundary.pt --output-dir "$1" "${@:2}"
