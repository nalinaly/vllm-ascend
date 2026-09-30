#!/usr/bin/env bash
# 单卡对照两份冻结源码；可选计时，不采 profiler。
set -eo pipefail
: "${TASK_DEVICE:?请通过 task-submit 提交}"
hca_tests="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
hca_output="$(realpath -m "${1:?指定结果目录}")"
hca_history="${2:?指定历史长度}"
hca_batch="${3:?指定 batch}"
hca_baseline="$(realpath "${4:?指定基线源码}")"
hca_candidate="$(realpath "${5:?指定候选源码}")"
hca_args=(--history "$hca_history" --batch "$hca_batch" --weight-nz-mode 2
          --deterministic-level 0 --device 0)
if [[ "${6:-}" == timing ]]; then
  hca_args+=(--timing-iters 50 --timing-warmup 10)
fi
bash "$hca_tests/run_hca_single_layer.sh" "$hca_output/baseline" \
  "${hca_args[@]}" --operator-source "$hca_baseline"
bash "$hca_tests/run_hca_single_layer.sh" "$hca_output/candidate" \
  "${hca_args[@]}" --operator-source "$hca_candidate" \
  --reference-state "$hca_output/baseline/states.pt"
