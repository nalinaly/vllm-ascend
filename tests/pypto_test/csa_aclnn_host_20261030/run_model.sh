#!/usr/bin/env bash
set -eo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo=$(cd "$here/../../.." && pwd)
workspace=$(dirname "$repo")
job=${1:?plan或prefill}
shift
if [[ ${1:-} == --device ]]; then shift 2; fi
source "$workspace/env-dsv4-0251rc1.sh"
source "$repo/tests/pypto_test/csa_shared_toolchain/env.sh"
frozen="$here/results/model_source"
bank="$here/results/kv/prefix128k"
output="$here/results/model_${job}"
cache="$here/results/model_prefill"
export PYTHONPATH="$frozen:$frozen/tests/pypto_test:$workspace/csa-repro-20261024-current/vllm:$PYTHONPATH"
export LD_LIBRARY_PATH="$repo/.cache/csa/native-install:$LD_LIBRARY_PATH"
export VLLM_ASCEND_ENABLE_NZ=0 PTO_CSA_VARIANT=spmd_tnd
export HCCL_OP_EXPANSION_MODE=AIV HCCL_DETERMINISTIC=true OMP_PROC_BIND=false OMP_NUM_THREADS=4
export ASCEND_WORK_PATH="$output/ascend" ASCEND_PROCESS_LOG_PATH="$output/ascend/log"
export VLLM_CACHE_ROOT="$cache/vllm_cache" ASCEND_CACHE_PATH="$cache/ascend_cache"
mkdir -p "$ASCEND_PROCESS_LOG_PATH" "$VLLM_CACHE_ROOT" "$ASCEND_CACHE_PATH"
case "$job" in
    plan) args=(plan --histories 131072) ;;
    prefill*)
        : "${TASK_DEVICE:?需要队列分配16卡}"
        args=(prefill --output "$output" --host 192.168.0.106 --nic enp23s0f3
              --backend native --gpu-memory-utilization 0.92 --weight-nz-mode 0
              --deterministic --event-work-mode 1)
        ;;
    *) exit 2 ;;
esac
cd "$frozen"
exec python tests/pypto_test/csa_toolchain_repro_20261019/bootstrap.py \
    --shared-toolchain --workspace-root "$workspace" --vllm-root "$workspace/csa-repro-20261024-current/vllm" \
    tests/pypto_test/offline_pd/run.py "${args[@]}" \
    --model /data/model/DeepSeek-V4-Flash-0731-w8a8 --bank "$bank"
