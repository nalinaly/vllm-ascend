#!/usr/bin/env bash
# 单卡检查短窗口、C128 写回和六个 token 的 Vector 尾行。
set -eo pipefail
: "${TASK_DEVICE:?请通过 task-submit 提交}"
hca_tests="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
hca_output="$(realpath -m "${1:?指定结果目录}")"
hca_baseline="$(realpath "${2:?指定基线源码}")"
hca_candidate="$(realpath "${3:?指定候选源码}")"
for hca_history in 0 124; do
  hca_args=(--batch 1 --history "$hca_history" --weight-nz-mode 2 --deterministic-level 0
            --device 0 --service-graph --poison-unused-compressed)
  bash "$hca_tests/run_hca_single_layer.sh" "$hca_output/h$hca_history/baseline" \
    "${hca_args[@]}" --operator-source "$hca_baseline"
  bash "$hca_tests/run_hca_single_layer.sh" "$hca_output/h$hca_history/candidate" \
    "${hca_args[@]}" --operator-source "$hca_candidate" \
    --reference-state "$hca_output/h$hca_history/baseline/states.pt"
done
