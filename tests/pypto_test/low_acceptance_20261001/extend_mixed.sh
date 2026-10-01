#!/usr/bin/env bash
set -eo pipefail
: "${TASK_DEVICE:?请通过task-submit分配16卡}"
experiment_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$experiment_dir/../../.." && pwd)"
workspace="$(dirname "$repo")"
source "$workspace/env-dsv4-0251rc1.sh"
source "$repo/tests/pypto_test/results/csa_native_template_20260929/env.sh"
output="$(realpath -m "${1:?新结果目录}")"
snapshot="$(realpath "${2:?冻结源码}")"
bank="$(realpath "${3:?输入bank}")"
destination="$(realpath "${4:?已准备的输出bank}")"
batch="${5:?batch}"
mkdir "$output"
cd "$(mktemp -d "$workspace/.cache/csa-mix-p-XXXXXX")"
export LD_LIBRARY_PATH="$repo/.cache/csa/native-install:$LD_LIBRARY_PATH"
export PYTHONPATH="$workspace/.cache/migration-v0.25.1rc1/vllm:$snapshot:$snapshot/tests/pypto_test:${PYTHONPATH:-}"
export HCCL_DETERMINISTIC=true
python "$snapshot/tests/pypto_test/offline_pd/run.py" decode \
    --bank "$bank" --save-extended-bank "$destination" --output "$output" \
    --backend native --pto-attention both --mixed-requests --batch "$batch" \
    --decode-dp 16 --decode-tokens 1 --max-num-batched-tokens 400 --weight-nz-mode 2 \
    --gpu-memory-utilization 0.95 --graph-mode eager --deterministic --port 30871 \
    > "$output/launch.log" 2>&1
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python "$snapshot/tests/pypto_test/low_acceptance_20261001/fixed_bank.py" \
    audit --bank "$destination" > "$output/bank_audit.log"
