#!/usr/bin/env bash
set -eo pipefail
pair_script="/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/atomic_add_seven_20260930/run_pair.sh"
bash "$pair_script" 131072 8 0
bash "$pair_script" 8192 16 1
bash "$pair_script" 8192 32 0
