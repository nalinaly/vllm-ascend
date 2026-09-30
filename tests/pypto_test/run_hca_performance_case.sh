#!/usr/bin/env bash
# 单卡同配置计时与独立泳道；源码快照使排队中的基线不受后续编辑影响。
set -eo pipefail
: "${TASK_DEVICE:?请通过 task-submit 提交}"
hca_test_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
hca_output="$(realpath -m "${1:?指定结果目录}")"
hca_history="${2:?指定历史长度}"
hca_batch="${3:?指定 batch}"
hca_source="$(realpath "${4:?指定冻结的 ops/pypto 源码目录}")"
hca_args=(--history "$hca_history" --batch "$hca_batch" --weight-nz-mode 2
          --deterministic-level 0 --operator-source "$hca_source" --device 0)
if [[ -n "${5:-}" ]]; then
  hca_args+=(--reference-state "$(realpath "$5")")
fi
bash "$hca_test_dir/run_hca_single_layer.sh" "$hca_output/timing" \
  "${hca_args[@]}" --timing-iters 50 --timing-warmup 10 --profile
bash "$hca_test_dir/run_hca_single_layer.sh" "$hca_output/swimlane" \
  "${hca_args[@]}" --swimlane
