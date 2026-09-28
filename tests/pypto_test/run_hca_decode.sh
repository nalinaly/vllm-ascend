#!/usr/bin/env bash
# 复用 P 阶段离线 KV，在真实 D TP1×DP/EP16 上比较最终 token。
set -eo pipefail
: "${TASK_DEVICE:?请通过 task-submit 分配所需设备}"
hca_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
workspace="$(dirname "$hca_repo")"
source "$workspace/env-dsv4-0251rc1.sh"
source_repo="$workspace/vllm-ascend-dsv4-pto-0251rc1"
output="$(realpath -m "${1:?指定新的结果目录}")"
history="${2:-8192}"
batch="${3:-16}"
decode_dp="${4:-16}"
native_reference="${5:-}"
backends=(pto native)
if [[ -n "$native_reference" ]]; then
    native_reference="$(realpath -e "$native_reference")"
    backends=(pto)
fi
port=30561
memory_utilization=0.9
if (( decode_dp == 8 )); then
    port=30562
    memory_utilization=0.95
fi
bank="$source_repo/tests/pypto_test/results/release_offline_pd_20260923/h${history}_bank"
capture_sizes=()
for size in 6 24 48 96 144 192 240; do
    if (( size <= batch * 6 )); then capture_sizes+=("$size"); fi
done
export LD_LIBRARY_PATH="$source_repo/.cache/csa/native-install:$LD_LIBRARY_PATH"
export PYTHONPATH="$workspace/.cache/migration-v0.25.1rc1/vllm:$hca_repo:$hca_repo/tests/pypto_test:${PYTHONPATH:-}"
export PTO_CSA_VARIANT=performance
export VLLM_ASCEND_PTO_CSA_ATOMIC_ADD=0
export HCCL_DETERMINISTIC=false
mkdir -p "$output"
cd "$output"
for backend in "${backends[@]}"; do
    mkdir -p "$output/$backend/ascend"
    export ASCEND_PROCESS_LOG_PATH="$output/$backend/ascend"
    python "$hca_repo/tests/pypto_test/offline_pd/run.py" decode \
        --bank "$bank" --output "$output/$backend" --backend "$backend" --pto-attention hca \
        --decode-dp "$decode_dp" \
        --gpu-memory-utilization "$memory_utilization" \
        --batch "$batch" --decode-tokens 256 --max-num-batched-tokens 400 \
        --weight-nz-mode 2 --graph-mode full_decode_only --capture-sizes "${capture_sizes[@]}" \
        --port "$port" > "$output/${backend}_launch.log" 2>&1
done
python "$hca_repo/tests/pypto_test/offline_pd/compare.py" \
    --native "${native_reference:-$output/native}" --pto "$output/pto" --bank "$bank" \
    --batch "$batch" --decode-tokens 256 --ranks "$decode_dp" --token-only --output "$output/token_comparison.json"
