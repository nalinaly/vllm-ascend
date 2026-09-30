#!/usr/bin/env bash
# 在**同一张卡**上顺序跑若干个 PTO 变体，按 ABBA…BA 顺序抵消卡内漂移。
#
# 为什么必须这样：`task-submit --device auto` 每个任务拿到的卡不同，
# 而同配置换卡实测能差 45 μs（2026-09-29：allow_early_resolve 在
# 11/13/15 三张卡上测出的 128K/B16 Δ 从 −23.75 翻成 +21.00）。
# 轮内 9 次重放取中位数只压掉了轮内方差，跨卡差异压不掉，
# 所以任何 A/B 都必须落在同一张卡上，并且正反各跑一遍。
#
# 用法:
#   run_hca_ab_same_card.sh <结果目录> <history> <batch> <变体1> [变体2 ...] [-- 额外参数]
# 变体名 base 表示用仓库里的生产源码；其它名字表示
# <tests>/variants_*/<名字> 下的冻结快照（用 --operator-source 指过去）。
# 变体名也可以写成 `<标签>=<绝对路径>`。
set -eo pipefail
hca_tests="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ab_output="$(realpath -m "${1:?指定结果目录}")"; shift
ab_history="${1:?指定 history}"; shift
ab_batch="${1:?指定 batch}"; shift
ab_variants=()
# 变体名收集到第一个以 `-` 开头的参数为止。task-submit 会在命令末尾自动追加
# `--device N`，不加这道判断它会被当成变体名。剩下的参数全部透传给 case 脚本。
while [ "$#" -gt 0 ] && [ "$1" != "--" ] && [ "${1#-}" = "$1" ]; do ab_variants+=("$1"); shift; done
[ "${1:-}" = "--" ] && shift
[ "${#ab_variants[@]}" -ge 1 ] || { printf '至少给一个变体\n' >&2; exit 2; }
mkdir -p "$ab_output"
printf '卡: %s  档位: %s/%s  变体: %s\n' "${TASK_DEVICE:-?}" "$ab_history" "$ab_batch" "${ab_variants[*]}" \
  | tee "$ab_output/plan.txt"

# 解析每个变体的 --operator-source 参数
declare -A ab_source
for v in "${ab_variants[@]}"; do
  label="${v%%=*}"
  if [ "$v" = base ]; then
    ab_source[base]=""
  elif [ "$label" != "$v" ]; then
    ab_source[$label]="--operator-source ${v#*=}"
  elif [[ "$v" =~ ^nz[0-9]$ ]] || [ "$v" = dbc ] || [ "$v" = hbg ] || [[ "$v" =~ ^at[2-5]$ ]]; then
    ab_source[$v]=""   # 只改环境/编译开关，源码用仓库生产版
  else
    hit="$(ls -d "$hca_tests"/variants_*/"$v" 2>/dev/null | head -1)"
    [ -n "$hit" ] || { printf '找不到变体 %s\n' "$v" >&2; exit 2; }
    ab_source[$v]="--operator-source $hit"
  fi
done
ab_labels=()
for v in "${ab_variants[@]}"; do ab_labels+=("${v%%=*}"); done

# ABBA：正序一遍、逆序一遍，重复 CYCLES 轮。每个变体因此有 2×CYCLES 个样本。
# 卡内漂移实测 14.5 μs（base 在第 1 遍 627.75、第 4 遍 642.25），ABBA 只抵消线性部分，
# 要分辨 10 μs 以下的效应就得多轮。
ab_cycles="${CYCLES:-1}"
ab_order=()
for ((c=0; c<ab_cycles; c++)); do
  ab_order+=("${ab_labels[@]}")
  for ((i=${#ab_labels[@]}-1; i>=0; i--)); do ab_order+=("${ab_labels[$i]}"); done
done

pass=0
for label in "${ab_order[@]}"; do
  pass=$((pass + 1))
  target="$ab_output/p${pass}_${label}"
  printf '=== pass %s: %s ===\n' "$pass" "$label"
  # 变体名形如 nz1 / nz2 时只切 kernel 侧的 NZ 档（BF16_WEIGHT_NZ = 值 >= 2），
  # 同时把 --weight-nz-mode 跟着设成同一个值，两侧保持一致。
  ab_env=(); ab_extra=()
  if [[ "$label" =~ ^nz([0-9])$ ]]; then
    ab_env=("VLLM_ASCEND_ENABLE_NZ=${BASH_REMATCH[1]}")
    ab_extra=(--weight-nz-mode "${BASH_REMATCH[1]}")
  elif [ "$label" = dbc ]; then
    # 只切编译期的 L0C 双缓冲开关，源码用仓库生产版。
    ab_extra=(--pypto-dbc)
  elif [ "$label" = hbg ]; then
    # 只切 Simpler 运行时 ABI：主机提前建图，取代 AICPU 上的 TensorMap 建图。
    ab_extra=(--pypto-runtime host_build_graph)
  elif [[ "$label" =~ ^at([2-5])$ ]]; then
    # 只切 AICPU 线程数（init 的 aicpu_thread_num，合法值 2..5）。
    ab_extra=(--aicpu-threads "${BASH_REMATCH[1]}")
  fi
  # 不用 exec：要在同一个任务（同一张卡）里接着跑下一个变体。
  env "${ab_env[@]}" bash "$hca_tests/run_hca_compiled_case.sh" "$target" pto \
    --history "$ab_history" --batch "$ab_batch" --super-kernel 0 \
    ${ab_source[$label]} "${ab_extra[@]}" "$@" || printf 'pass %s (%s) 失败\n' "$pass" "$label" >&2
done
printf '完成，结果在 %s\n' "$ab_output"
