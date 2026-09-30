#!/usr/bin/env bash
# 同卡按 A→B→B→A 比较两个源码快照，排除跨卡和单次运行漂移。
set -eo pipefail
: "${TASK_DEVICE:?请通过 task-submit 提交}"
hca_tests="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
hca_output="$(realpath -m "${1:?指定结果目录}")"
hca_history="${2:?指定历史长度}"
hca_batch="${3:?指定 batch}"
hca_baseline="$(realpath "${4:?指定基线源码}")"
hca_candidate="$(realpath "${5:?指定候选源码}")"
for hca_run in 0_baseline 1_candidate 2_candidate 3_baseline; do
  hca_source="$hca_baseline"
  if [[ "$hca_run" == *candidate ]]; then hca_source="$hca_candidate"; fi
  hca_reference=()
  if [[ "$hca_run" != 0_baseline ]]; then
    hca_reference=(--reference-state "$hca_output/0_baseline/states.pt")
  fi
  bash "$hca_tests/run_hca_single_layer.sh" "$hca_output/$hca_run" \
    --history "$hca_history" --batch "$hca_batch" --weight-nz-mode 2 --deterministic-level 0 \
    --operator-source "$hca_source" --device 0 --timing-iters 100 --timing-warmup 10 \
    "${hca_reference[@]}"
done
if [[ "${6:-}" == swimlane ]]; then
  bash "$hca_tests/run_hca_single_layer.sh" "$hca_output/swimlane" \
    --history "$hca_history" --batch "$hca_batch" --weight-nz-mode 2 --deterministic-level 0 \
    --operator-source "$hca_candidate" --device 0 --swimlane \
    --reference-state "$hca_output/0_baseline/states.pt"
fi
