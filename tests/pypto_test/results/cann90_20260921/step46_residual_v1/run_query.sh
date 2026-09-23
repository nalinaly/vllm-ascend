#!/usr/bin/env bash
set -eo pipefail
source /data/pyptouser/qinchuanyu/pto-eager/env.sh
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
cd /data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto
python tests/pypto_test/dsv4_csa_main_query_boundary.py --device "$TASK_DEVICE" --checkpoint /data/model/dsv4-flash-0731-dspark-w8a8 --capture tests/pypto_test/results/cann90_20260921/step46_residual_v1/capture/output_boundary.pt --output-dir tests/pypto_test/results/cann90_20260921/step46_residual_v1/query_dequant
