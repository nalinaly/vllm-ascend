#!/usr/bin/env bash
set -eo pipefail
: "${TASK_DEVICE:?Submit through task-submit}"
source /data/pyptouser/qinchuanyu/pto-eager/env.sh
cd /data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export ASCEND_PROCESS_LOG_PATH=/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto/tests/pypto_test/results/cann90_20260921/full_cross_stream_v2/b40_indexer/plog
python /data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto/tests/pypto_test/dsv4_csa_cross_stream.py --checkpoint /data/model/dsv4-flash-0731-dspark-w8a8 --batch 40 --stage INDEXER --output-dir /data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto/tests/pypto_test/results/cann90_20260921/full_cross_stream_v2/b40_indexer --device "$TASK_DEVICE"
