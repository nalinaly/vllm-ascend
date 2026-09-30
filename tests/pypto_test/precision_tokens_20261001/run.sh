#!/usr/bin/env bash
# 两侧只切换 CSA 算术版本；相同 HCA、TMR、确定性、权重及历史缓存。
set -eo pipefail
: "${TASK_DEVICE:?请通过 task-submit 分配16张卡}"
experiment_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$experiment_dir/../../.." && pwd)"
workspace="$(dirname "$repo")"
source "$workspace/env-dsv4-0251rc1.sh"
source "$repo/tests/pypto_test/results/csa_native_template_20260929/env.sh"
output="$(realpath -m "${1:?指定新的结果目录}")"
snapshot="$(realpath "${2:?指定冻结源码目录}")"
bank="$repo/tests/pypto_test/results/release_offline_pd_20260923/h131072_bank"
mkdir "$output"
# CANN 会把 cwd 编入 shape_info 文件名；长结果路径会超过单个文件名255字节。
compile_cwd="$(mktemp -d "$workspace/.cache/csa-tokens-XXXXXX")"
printf '%s\n' "$compile_cwd" > "$output/compile_workdir.txt"
cd "$compile_cwd"
export LD_LIBRARY_PATH="$repo/.cache/csa/native-install:$LD_LIBRARY_PATH"
export PYTHONPATH="$workspace/.cache/migration-v0.25.1rc1/vllm:$snapshot:$snapshot/tests/pypto_test:${PYTHONPATH:-}"
export VLLM_ASCEND_PTO_CSA_ATOMIC_ADD=0
export PTO_CSA_RUNTIME=tensormap_and_ringbuffer
export HCCL_DETERMINISTIC=true
python -c 'import vllm_ascend; print(vllm_ascend.__file__)' > "$output/import_path.txt"
grep -F "$snapshot/vllm_ascend/__init__.py" "$output/import_path.txt"
for variant in performance precision; do
    export PTO_CSA_VARIANT="$variant"
    mkdir -p "$output/$variant/ascend"
    export ASCEND_PROCESS_LOG_PATH="$output/$variant/ascend"
    export PYPTO_PROG_BUILD_DIR="$output/$variant/build"
    opp_workspace="$(mktemp -d "$workspace/.cache/precision-tokens-opp-${variant}-XXXXXX")"
    export ASCEND_OPP_PATH="$(python "$snapshot/tests/pypto_test/dsv4_hca_prepare_opp.py" --destination "$opp_workspace")"
    printf '%s\n' "$ASCEND_OPP_PATH" > "$output/$variant/opp_path.txt"
    python "$snapshot/tests/pypto_test/offline_pd/run.py" decode \
        --bank "$bank" --output "$output/$variant" --backend pto --pto-attention both \
        --decode-dp 16 --gpu-memory-utilization 0.95 --batch 16 --deterministic \
        --decode-tokens 192 --max-num-batched-tokens 400 --weight-nz-mode 2 \
        --graph-mode full_decode_only --capture-sizes 6 24 48 96 --port 30741 \
        > "$output/${variant}_launch.log" 2>&1
    if grep -q "execute op_compiler error" "$output/$variant/rank0.log"; then
        echo "静态kernel打包失败，不能将requested开关记作实际成功" >&2
        exit 1
    fi
    python "$snapshot/tests/pypto_test/csa_hca_merge_20260930/functional.py" \
        --root "$output/$variant" --bank "$bank" --batch 16 --decode-tokens 192 \
        --output "$output/${variant}_functional.json"
done
