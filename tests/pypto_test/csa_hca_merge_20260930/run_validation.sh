#!/usr/bin/env bash
# 按顺序验证联合整网功能和输出 token 精度，不采集 profiler 或性能样本。
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
mkdir -p "$(dirname "$output")"
mkdir "$output"
cd "$output"
source_check=(--repo "$repo" --manifest "$output/source.json" --bank "$bank" --history "$history" --batch "$batch")
python "$repo/tests/pypto_test/csa_hca_merge_20260930/source.py" record "${source_check[@]}"
for backend in pto native; do
    mkdir -p "$output/$backend/ascend"
    export ASCEND_PROCESS_LOG_PATH="$output/$backend/ascend"
    # 各侧独占可写静态编译目录，避免修改共用 CANN 安装。
    opp_workspace="$(mktemp -d "$workspace/.cache/merge-opp-${backend}-XXXXXX")"
    export ASCEND_OPP_PATH="$(python "$repo/tests/pypto_test/dsv4_hca_prepare_opp.py" --destination "$opp_workspace")"
    printf '%s\n' "$ASCEND_OPP_PATH" > "$output/$backend/opp_path.txt"
    python "$repo/tests/pypto_test/offline_pd/run.py" decode \
        --bank "$bank" --output "$output/$backend" --backend "$backend" --pto-attention both \
        --decode-dp 16 --gpu-memory-utilization 0.95 --batch "$batch" \
        --decode-tokens 192 --max-num-batched-tokens 400 --weight-nz-mode 2 \
        --graph-mode full_decode_only --capture-sizes "${capture_sizes[@]}" --port 30631 > "$output/${backend}_launch.log" 2>&1
    # 两侧必须运行同一份执行代码；途中更新源码时保留结果，但不放行下一阶段。
    python "$repo/tests/pypto_test/csa_hca_merge_20260930/source.py" verify "${source_check[@]}"
    if [[ "$backend" == pto ]]; then
        # 功能闸门通过后才启动 Native，复用本轮 PTO 输出进行精度比较。
        python "$repo/tests/pypto_test/csa_hca_merge_20260930/functional.py" \
            --root "$output/pto" --bank "$bank" --batch "$batch" --decode-tokens 192 \
            --output "$output/functional.json" > "$output/functional.log" 2>&1
    fi
done
python "$repo/tests/pypto_test/offline_pd/compare.py" \
    --native "$output/native" --pto "$output/pto" --bank "$bank" --batch "$batch" \
    --decode-tokens 192 --ranks 16 --token-only --output "$output/token_comparison.json" \
    > "$output/comparison.log" 2>&1
