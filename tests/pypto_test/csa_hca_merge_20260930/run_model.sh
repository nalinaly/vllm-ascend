#!/usr/bin/env bash
# 复用 P TP4×DP4 离线 KV，对照 Native 与 CSA/HCA 同时启用的 D TP1×DP/EP16。
set -eo pipefail
: "${TASK_DEVICE:?请通过 task-submit 分配 16 张卡}"
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
workspace="$(dirname "$repo")"
source "$workspace/env-dsv4-0251rc1.sh"
source_repo="$workspace/vllm-ascend-dsv4-pto-0251rc1"
# 整网 norm/quant 融合需要已构建的 AddRmsNormBias 补充包；公共环境只含 CSA 基础包。
# 两侧使用相同 vendor，随后各自创建私有可写 OPP 根。
source "$source_repo/tests/pypto_test/results/csa_native_template_20260929/env.sh"
output="$(realpath -m "${1:?指定新的结果目录}")"
history="${2:-131072}"
batch="${3:-16}"
bank="$source_repo/tests/pypto_test/results/release_offline_pd_20260923/h${history}_bank"
export LD_LIBRARY_PATH="$source_repo/.cache/csa/native-install:$LD_LIBRARY_PATH"
export PYTHONPATH="$workspace/.cache/migration-v0.25.1rc1/vllm:$repo:$repo/tests/pypto_test:${PYTHONPATH:-}"
export PTO_CSA_VARIANT=performance
export VLLM_ASCEND_PTO_CSA_ATOMIC_ADD=0
export HCCL_DETERMINISTIC=false
capture_sizes=()
for size in 6 24 48 96 144 192 240; do
    if (( size <= batch * 6 )); then capture_sizes+=("$size"); fi
done
mkdir -p "$output"
cd "$output"
for backend in pto native; do
    mkdir -p "$output/$backend/ascend"
    export ASCEND_PROCESS_LOG_PATH="$output/$backend/ascend"
    # 各侧独占可写静态编译目录，避免修改共用 CANN 安装。
    opp_workspace="$(mktemp -d "$workspace/.cache/merge-opp-${backend}-XXXXXX")"
    export ASCEND_OPP_PATH="$(python "$repo/tests/pypto_test/dsv4_hca_prepare_opp.py" --destination "$opp_workspace")"
    printf '%s\n' "$ASCEND_OPP_PATH" > "$output/$backend/opp_path.txt"
    python "$repo/tests/pypto_test/offline_pd/run.py" performance \
        --bank "$bank" --output "$output/$backend" --backend "$backend" --pto-attention both \
        --decode-dp 16 --gpu-memory-utilization 0.95 --batch "$batch" \
        --decode-tokens 192 --max-num-batched-tokens 400 --weight-nz-mode 2 \
        --graph-mode full_decode_only --capture-sizes "${capture_sizes[@]}" --port 30631 \
        --warmup-rounds 1 --warmup-tokens 96 --warmup-steps 8 --steady-cycles 10 \
        --profile-start-step 8 --profile-steps 3 > "$output/${backend}_launch.log" 2>&1
done
python "$repo/tests/pypto_test/offline_pd/performance.py" \
    --root "$output" --bank "$bank" --mode 2 --batch "$batch" --decode-tokens 192 \
    --pto-attention both --token-only > "$output/comparison.log" 2>&1
