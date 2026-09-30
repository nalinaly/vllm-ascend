#!/usr/bin/env bash
set -eo pipefail
: "${TASK_DEVICE:?Submit through task-submit --device auto}"
workspace=/data/pyptouser/qinchuanyu/pto-eager
repo="$workspace/vllm-ascend-dsv4-pto-0251rc1"
root="$repo/tests/pypto_test/results/csa_native_sk_seven_20260930"
super_kernel="${1:?0 or 1}"
history="${2:?history length}"
batch="${3:?batch size}"
source "$workspace/env-dsv4-0251rc1.sh"
source "$repo/tests/pypto_test/results/csa_native_template_20260929/env.sh"
source_repo="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["source"])' "$root/source.json")"
export PTO_CSA_VARIANT=pkg:dsv4_csa_coefficients_seven_20260929
export LD_LIBRARY_PATH="$repo/.cache/csa/native-install:$LD_LIBRARY_PATH"
export PYTHONPATH="$source_repo/tests/pypto_test:${PYTHONPATH:-}"
export VLLM_ASCEND_PTO_CSA_ATOMIC_ADD=0 VLLM_ASCEND_ENABLE_NZ=2
export PTO_CSA_RING_HEAP_MB=256,128,256,32 PTO_CSA_RING_TASK_WINDOW=4096
export OMP_NUM_THREADS=10 OMP_PROC_BIND=false VLLM_BATCH_INVARIANT=0
export LOCAL_WORLD_SIZE=1
export HCCL_OP_EXPANSION_MODE=AIV HCCL_BUFFSIZE=1800 HCCL_DETERMINISTIC=false
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export DYNAMIC_EPLB=false EXPERT_MAP_RECORD=false
[[ "$ASCEND_HOME_PATH" == */cann-9.2.0-beta.2 ]]
out="$root/h${history}_b${batch}/super${super_kernel}_${RUN_SLOT:?a or b}"
test ! -e "$out/report.json"
mkdir -p "$out/ascend" "$out/ascend_cache"
export ASCEND_PROCESS_LOG_PATH="$out/ascend" ASCEND_CACHE_PATH="$out/ascend_cache"
export VLLM_CACHE_ROOT="$out/vllm_cache"
ASCEND_OPP_PATH="$(python "$source_repo/tests/pypto_test/coefficients_seven_experiment/prepare_opp.py" \
    --destination "$out/opp_env")"
export ASCEND_OPP_PATH
# static_kernel 编译器把 CWD 完整路径展平当生成文件名，结果目录一深就
# OSError: [Errno 36] File name too long；换到短路径编译，report 由 --output 写回。
work="$workspace/.cache/a/nsk_${history}_${batch}_${super_kernel}"; rm -rf "$work"; mkdir -p "$work"
cd "$work"
exec python "$source_repo/tests/pypto_test/coefficients_seven_experiment/compiled_case.py" \
    --side native --super-kernel "$super_kernel" --history "$history" --batch "$batch" \
    --save-state --output "$out" --device "$TASK_DEVICE" > "$out/run.log" 2>&1
