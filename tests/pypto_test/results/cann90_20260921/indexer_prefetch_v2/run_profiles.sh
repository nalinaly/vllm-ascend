#!/usr/bin/env bash
set -eo pipefail
: "${TASK_DEVICE:?Submit through task-submit}"
source /data/pyptouser/qinchuanyu/pto-eager/env.sh
cd /data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
mkdir -p /data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto/tests/pypto_test/results/cann90_20260921/indexer_prefetch_v2/baseline/pypto/plog
export ASCEND_PROCESS_LOG_PATH=/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto/tests/pypto_test/results/cann90_20260921/indexer_prefetch_v2/baseline/pypto/plog
python /data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto/tests/pypto_test/results/cann90_20260921/indexer_prefetch_v2/profile_snapshot.py --snapshot /data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto/tests/pypto_test/results/cann90_20260921/indexer_prefetch_v2/before --variant pypto --device "$TASK_DEVICE" --checkpoint /data/model/dsv4-flash-0731-dspark-w8a8 --batch 40 --history 131073 --replays 5 --output-dir /data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto/tests/pypto_test/results/cann90_20260921/indexer_prefetch_v2/baseline/pypto
mkdir -p /data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto/tests/pypto_test/results/cann90_20260921/indexer_prefetch_v2/profile/pypto/plog
export ASCEND_PROCESS_LOG_PATH=/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto/tests/pypto_test/results/cann90_20260921/indexer_prefetch_v2/profile/pypto/plog
python /data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto/tests/pypto_test/results/cann90_20260921/indexer_prefetch_v2/profile_snapshot.py --snapshot /data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto/tests/pypto_test/results/cann90_20260921/indexer_prefetch_v2/sources --variant pypto --device "$TASK_DEVICE" --checkpoint /data/model/dsv4-flash-0731-dspark-w8a8 --batch 40 --history 131073 --replays 5 --output-dir /data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto/tests/pypto_test/results/cann90_20260921/indexer_prefetch_v2/profile/pypto
mkdir -p /data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto/tests/pypto_test/results/cann90_20260921/indexer_prefetch_v2/profile/native/plog
export ASCEND_PROCESS_LOG_PATH=/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto/tests/pypto_test/results/cann90_20260921/indexer_prefetch_v2/profile/native/plog
python tests/pypto_test/dsv4_csa_profile.py --variant native --device "$TASK_DEVICE" --checkpoint /data/model/dsv4-flash-0731-dspark-w8a8 --batch 40 --history 131073 --replays 5 --output-dir /data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto/tests/pypto_test/results/cann90_20260921/indexer_prefetch_v2/profile/native
mkdir -p /data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto/tests/pypto_test/results/cann90_20260921/indexer_prefetch_v2/profile/swimlane/plog
export ASCEND_PROCESS_LOG_PATH=/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto/tests/pypto_test/results/cann90_20260921/indexer_prefetch_v2/profile/swimlane/plog
python /data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto/tests/pypto_test/results/cann90_20260921/indexer_prefetch_v2/profile_snapshot.py --snapshot /data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto/tests/pypto_test/results/cann90_20260921/indexer_prefetch_v2/sources --variant pypto --device "$TASK_DEVICE" --checkpoint /data/model/dsv4-flash-0731-dspark-w8a8 --batch 40 --history 131073 --replays 5 --output-dir /data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto/tests/pypto_test/results/cann90_20260921/indexer_prefetch_v2/profile/swimlane --swimlane
