#!/usr/bin/env bash
# HCA 单算子的上线编译口径对照，一次只跑一侧。
#
# 与 run_hca_single_layer.sh 的区别：这条路径走 torch.compile(backend="npugraph_ex")，
# 因此 static_kernel 与 SuperKernel 才真正生效；代价是编译期要往
# $ASCEND_OPP_PATH/static_kernel 装包，两侧必须各自从干净目录开始，
# 所以不能在同一进程里交替 A/B。
#
# 用法: run_hca_compiled_case.sh <结果目录> <native|pto> [dsv4_hca_compiled_case.py 的其它参数]
set -eo pipefail
: "${TASK_DEVICE:?请通过 task-submit 提交}"
if [[ "$TASK_DEVICE" == *,* ]]; then
  printf '单算子对照只需要一张卡。\n' >&2
  exit 2
fi
hca_tests="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
hca_repo="$(cd "$hca_tests/../.." && pwd)"
source "$(dirname "$hca_repo")/env-dsv4-0251rc1.sh"
hca_output="$(realpath -m "${1:?指定结果目录}")"
shift
hca_side="${1:?指定 native 或 pto}"
shift
mkdir -p "$hca_output/ascend"
# 不做 ASCEND_RT_VISIBLE_DEVICES 重映射：直接用队列分配的那张卡（TASK_DEVICE），
# 与 CSA 的 compiled_case 一致。早先又重映射又强行传 --device 0，两者打架。
export ASCEND_PROCESS_LOG_PATH="$hca_output/ascend"
# kernel 侧的 nz_mode.py 读的是这个环境变量（BF16_WEIGHT_NZ = 值 >= 2），
# 而 --weight-nz-mode 只进 vLLM 的 additional_config（主机侧权重存储格式）。
# 两者必须一致，否则主机把 BF16 权重存成 FRACTAL_NZ 而 kernel 按 ND 寻址，
# 只能靠 recast 出私有副本来兜，量到的就不是生产路径。允许外部覆盖以便对照。
# 默认 2：kernel 侧 BF16_WEIGHT_NZ = 值 >= 2，必须与 --weight-nz-mode（默认 2，
# 决定主机侧权重存储格式）一致。2026-09-29 之前这里是 1，导致主机把 BF16 权重
# 存成 FRACTAL_NZ 而 kernel 却按 ND 编译（走 _proj_a_mm_nd），量到的不是生产路径。
export VLLM_ASCEND_ENABLE_NZ="${VLLM_ASCEND_ENABLE_NZ:-2}"
export PTO_CSA_VARIANT=performance
export VLLM_ASCEND_PTO_CSA_ATOMIC_ADD=0
# 与上线 decode 口径对齐，取自
# vllm-ascend-main/tests/dsv4_perf_accuracy_20260827/runtime/decode/run_dp_template.sh。
# 都是进程级设置，两侧同配。
export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_BUFFSIZE=1800
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_BATCH_INVARIANT=0
# 私有可写 OPP 根：static_kernel 留空，本侧生成的静态包只落在这里。
# 必须在 source 环境之后覆盖 ASCEND_OPP_PATH；ASCEND_CUSTOM_OPP_PATH（Native 自定义算子包）
# 保持环境给的值不动。
hca_opp_root="$hca_output/opp_root"
rm -rf "$hca_opp_root"
ASCEND_OPP_PATH="$(python "$hca_tests/dsv4_hca_prepare_opp.py" --destination "$hca_opp_root")"
export ASCEND_OPP_PATH
mkdir -p "$ASCEND_OPP_PATH/static_kernel"
cd "$hca_output"
# --device 不传：dsv4_hca_compiled_case.py 默认取 TASK_DEVICE，即队列分给本任务的卡。
exec python "$hca_tests/dsv4_hca_compiled_case.py" \
  --side "$hca_side" --output "$hca_output" "$@"
