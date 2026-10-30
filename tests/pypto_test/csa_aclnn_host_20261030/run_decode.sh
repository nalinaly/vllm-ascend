#!/usr/bin/env bash
set -eo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo=$(cd "$here/../../.." && pwd)
workspace=$(dirname "$repo")
stage=${1:?suffix或native或pto或profile_native或profile_pto}
label=${2:?独立结果目录名}
shift 2
if [[ ${1:-} == --device ]]; then shift 2; fi
source "$workspace/env-dsv4-0251rc1.sh"
source "$repo/tests/pypto_test/csa_shared_toolchain/env.sh"
frozen="$here/results/model_decode_source_v3"
site="$here/results/aclnn_site"
side=${stage##*_}
cache_side=$side
if [[ "$side" == pto ]]; then
    site="$here/results/aclnn_rebased_site"
    cache_side=pto_ndptr
fi
output="$here/results/$label"
bank="$here/results/kv/mixed128k_fixed"
export PYTHONPATH="$frozen:$frozen/tests/pypto_test:$workspace/csa-repro-20261024-current/vllm:$site:$PYTHONPATH"
# Native 图融合规则注册会执行 AddRmsNormBias；它不在 attention 专用 vendor 内。
native_template="$repo/.cache/csa/native-template-cann92-stage/packages/vendors/csa_template_transformer"
export ASCEND_CUSTOM_OPP_PATH="$site/csa_device_pto_ops/ascend/vendors/csa_spmd_tnd_device_pto:$native_template:$here/results/static_vendor/custom_transformer"
export LD_LIBRARY_PATH="$repo/.cache/csa/native-install:$LD_LIBRARY_PATH"
export VLLM_ASCEND_ENABLE_NZ=2 PTO_CSA_VARIANT=spmd_tnd VLLM_ASCEND_PTO_CSA_BACKEND=aclnn
export PTO_CSA_RUNTIME=tensormap_and_ringbuffer VLLM_ASCEND_PTO_CSA_ATOMIC_ADD=0
export ASCEND_GLOBAL_LOG_LEVEL=3 HCCL_OP_EXPANSION_MODE=AIV OMP_PROC_BIND=false OMP_NUM_THREADS=10
export ASCEND_WORK_PATH="$output/ascend" ASCEND_PROCESS_LOG_PATH="$output/ascend/log"
export PYPTO_PROG_BUILD_DIR="$output/unused_pypto_build"
cache="$here/results/decode_cache_$cache_side"
export VLLM_CACHE_ROOT="$cache/vllm" ASCEND_CACHE_PATH="$cache/ascend"
mkdir -p "$ASCEND_PROCESS_LOG_PATH" "$VLLM_CACHE_ROOT" "$ASCEND_CACHE_PATH" "$cache/compile"
export ASCEND_OPP_PATH=$(python "$repo/tests/pypto_test/dsv4_hca_prepare_opp.py" --destination "$cache/private_opp")
args=(--host 192.168.0.106 --nic enp23s0f3 --decode-dp 16 --batch 16 --mixed-requests
      --gpu-memory-utilization 0.92 --weight-nz-mode 2 --event-work-mode 1 --port 30881
      --output "$output" --pto-attention csa)
case "$stage" in
    suffix)
        export HCCL_DETERMINISTIC=true
        bank="$here/results/kv/mixed128k"
        command=decode
        args+=(--backend native --save-extended-bank "$here/results/kv/mixed128k_fixed"
               --decode-tokens 1 --graph-mode eager --no-static-kernel --deterministic)
        ;;
    native|pto|profile_native|profile_pto)
        export HCCL_DETERMINISTIC=false
        command=performance
        [[ "$stage" == profile_* ]] && command=profile
        args+=(--backend "$side" --decode-tokens 192 --warmup-tokens 96 --warmup-steps 8
               --steady-cycles 10 --profile-steps 3 --graph-mode full_decode_only
               --static-kernel --super-kernel --capture-sizes 6 96)
        [[ "$command" == performance ]] && args+=(--forward-host-diagnostics)
        ;;
    *) exit 2 ;;
esac
: "${TASK_DEVICE:?需要队列分配16卡}"
cd "$cache/compile"
exec python "$frozen/tests/pypto_test/csa_toolchain_repro_20261019/bootstrap.py" \
    --shared-toolchain --workspace-root "$workspace" --vllm-root "$workspace/csa-repro-20261024-current/vllm" \
    "$frozen/tests/pypto_test/offline_pd/run.py" "$command" "${args[@]}" \
    --model /data/model/DeepSeek-V4-Flash-0731-w8a8 --bank "$bank"
