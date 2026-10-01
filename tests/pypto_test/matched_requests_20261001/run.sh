#!/usr/bin/env bash
# 参数之后可能有队列追加的--device；只使用明确的固定位置参数。
set -eo pipefail
: "${TASK_DEVICE:?请通过统一队列分配卡}"
experiment_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$experiment_dir/../../.." && pwd)"
workspace="$(dirname "$repo")"
action="${1:?guard或model}"
output="$(realpath -m "${2:?新结果目录}")"
snapshot="$(realpath "${3:?冻结源码}")"
if [[ "$action" == model ]]; then
    bank="$(realpath "${4:?所选bank}")"
    configuration="${5:?三方配置之一}"
    exec bash "$repo/tests/pypto_test/low_acceptance_20261001/run_mixed.sh" \
        "$output" "$snapshot" "$bank" 9 "$configuration"
fi
[[ "$action" == guard && "$TASK_DEVICE" != *,* ]]
source "$workspace/env-dsv4-0251rc1.sh"
source "$repo/tests/pypto_test/results/csa_native_template_20260929/env.sh"
export ASCEND_RT_VISIBLE_DEVICES="$TASK_DEVICE"
export PYTHONPATH="$snapshot/tests/pypto_test:$snapshot:$workspace/.cache/migration-v0.25.1rc1/vllm:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$repo/.cache/csa/native-install:$LD_LIBRARY_PATH"
export VLLM_ASCEND_PTO_CSA_ATOMIC_ADD=0 PTO_CSA_RUNTIME=tensormap_and_ringbuffer
export HCCL_DETERMINISTIC=true HCCL_OP_EXPANSION_MODE=AIV HCCL_BUFFSIZE=1800
export OMP_PROC_BIND=false OMP_NUM_THREADS=10 VLLM_BATCH_INVARIANT=0
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
mkdir "$output"
for variant in performance precision; do
    current="$output/$variant"
    mkdir "$current"
    export PYPTO_PROG_BUILD_DIR="$current/build" ASCEND_PROCESS_LOG_PATH="$current/ascend"
    export PTO_CSA_VARIANT="$variant"
    cd "$current"
    python "$snapshot/tests/pypto_test/dsv4_csa_single_layer.py" --output "$current" \
        --checkpoint /data/model/DeepSeek-V4-Flash-0731-w8a8 \
        --batch 9 --history "${4:?最大历史长度}" --variant "$variant" --weight-nz-mode 2 \
        --atomic-add 0 --deterministic-level 1 --device 0 --graph --padding-graph \
        > "$current/run.log" 2>&1
done
