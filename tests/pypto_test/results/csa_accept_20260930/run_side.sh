#!/usr/bin/env bash
# 同轮同卡 Native↔PTO 交替的七档验收。已落地生产（9a983dae，含 early3）。
# Native 开 SuperKernel（它最好的形态）；PTO 结构上开不了、恒为 0。
set -eo pipefail
: "${TASK_DEVICE:?Submit through task-submit --device auto}"
workspace=/data/pyptouser/qinchuanyu/pto-eager
repo="$workspace/vllm-ascend-dsv4-pto-0251rc1"
root="$repo/tests/pypto_test/results/csa_accept_20260930"
side="${1:?native or pto}"; history="${2:?history}"; batch="${3:?batch}"; tag="${4:?tag}"
source "$workspace/env-dsv4-0251rc1.sh"
source "$repo/tests/pypto_test/results/csa_native_template_20260929/env.sh"
source_repo="$workspace/.cache/csa-accept-9a983dae"
export PTO_CSA_VARIANT=pkg:dsv4_csa_accept_9a983dae
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
out="$root/h${history}_b${batch}/$tag"
test ! -e "$out/report.json"
mkdir -p "$out/ascend" "$out/ascend_cache"
export ASCEND_PROCESS_LOG_PATH="$out/ascend" ASCEND_CACHE_PATH="$out/ascend_cache"
export VLLM_CACHE_ROOT="$out/vllm_cache"
ASCEND_OPP_PATH="$(python "$source_repo/tests/pypto_test/coefficients_seven_experiment/prepare_opp.py" \
    --destination "$out/opp_env")"
export ASCEND_OPP_PATH
# static_kernel 编译器把 CWD 的完整路径展平（/→_）当生成文件名，结果目录太深会
# 触发 OSError: [Errno 36] File name too long，进而 static_compile_results=[False]、
# installed_static_packages=0，Native 侧直接判编译失败。所以换到一个短路径下编译，
# report 仍由 --output 写回结果目录。
work="$workspace/.cache/a/${history}_${batch}_${tag}"
rm -rf "$work"; mkdir -p "$work"
cd "$work"
# 两侧同一个编译入口：vllm 的 @support_torch_compile（backend_options
# force_eager/inplace_pass/static_kernel_compile 由 vllm config 统一给两侧）。
# ⚠ 这条路径下 Native **没有** super_kernel_optimize。改用 4ffccb7b 那份带
# --super-kernel 的 runner 试过，报 "Cannot prepare for replay during capturing
# stage"——那是旧冻结的 runner 配新源，两代 harness 不兼容，不在本轮解决。
# 所以本目录给出的是"同入口、Native 不开 SuperKernel"的比值；
# 要与既有的 Native-with-SuperKernel 基线对齐，见 RESULTS 里的换算。
# Native 开 SuperKernel（既有基线的口径）；PTO 侧不开——它结构上用不上。
# Native 的 SuperKernel 由 ACCEPT_NATIVE_SK 控制（默认 1）。交付 profiling 时用 0：
# SuperKernel 会把多个算子融进一个 sk_* kernel，逐算子耗时无法归属，
# 关掉才能拿到 Native 各独立算子的时长用于与 PTO 的 incore task 对照。
if [[ "$side" == native && "${ACCEPT_NATIVE_SK:-1}" == 1 ]]; then
  export ACCEPT_SUPER_KERNEL=1
else
  export ACCEPT_SUPER_KERNEL=0
fi
exec python "$root/compiled_case.py" --side "$side" \
    --history "$history" --batch "$batch" --save-state --output "$out" --device "$TASK_DEVICE" > "$out/run.log" 2>&1
