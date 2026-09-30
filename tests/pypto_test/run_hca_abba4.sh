#!/usr/bin/env bash
# 同卡 ABBA 的 4 个周期（共 16 轮），用于分辨 5～15 μs 量级的候选。
#
# 为什么是加轮数而不是加采样数：null A/A 实测中，单轮 100 个样本的 p50 标准误只有
# 1.4～2.4 μs，而轮与轮之间的 p50 stdev 有 3.6～13.1 μs——噪声几乎全在轮间
# （ABBA 的每一轮都是独立进程，重新加载模型，L2/显存布局与时钟状态都不同）。
# 因此噪声按 1/sqrt(轮数) 下降：4 个周期把 ±13 μs 压到约 ±6.5 μs。
set -eo pipefail
: "${TASK_DEVICE:?请通过 task-submit 提交}"
hca_tests="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
hca_output="$(realpath -m "${1:?指定结果目录}")"
hca_history="${2:?指定历史长度}"
hca_batch="${3:?指定 batch}"
hca_baseline="$(realpath "${4:?指定基线源码}")"
hca_candidate="$(realpath "${5:?指定候选源码}")"
hca_cycles="${6:-4}"
hca_first=""
for cycle in $(seq 0 $((hca_cycles - 1))); do
  for hca_run in 0_baseline 1_candidate 2_candidate 3_baseline; do
    hca_source="$hca_baseline"
    if [[ "$hca_run" == *candidate ]]; then hca_source="$hca_candidate"; fi
    hca_dir="$hca_output/c${cycle}_$hca_run"
    hca_reference=()
    if [[ -n "$hca_first" ]]; then hca_reference=(--reference-state "$hca_first/states.pt"); fi
    bash "$hca_tests/run_hca_single_layer.sh" "$hca_dir" \
      --history "$hca_history" --batch "$hca_batch" --weight-nz-mode 2 --deterministic-level 0 \
      --operator-source "$hca_source" --device 0 --timing-iters 100 --timing-warmup 10 \
      "${hca_reference[@]}"
    if [[ -z "$hca_first" ]]; then hca_first="$hca_dir"; fi
  done
done
