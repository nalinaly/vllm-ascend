#!/usr/bin/env bash
# 单卡候选：先对旧快照精确门禁并计时，再在独立进程分别采冷 L2 与“Native 之后”的泳道。
set -eo pipefail
: "${TASK_DEVICE:?请通过 task-submit 提交}"
hca_tests="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
hca_output="$(realpath -m "${1:?指定结果目录}")"
hca_history="${2:?指定历史长度}"
hca_batch="${3:?指定 batch}"
hca_source="$(realpath "${4:?指定冻结的 ops/pypto 源码目录}")"
hca_reference="$(realpath "${5:?指定旧 PTO 快照 states.pt}")"
hca_args=(--history "$hca_history" --batch "$hca_batch" --weight-nz-mode 2
          --deterministic-level 0 --device 0 --operator-source "$hca_source")
bash "$hca_tests/run_hca_single_layer.sh" "$hca_output/timing" "${hca_args[@]}" \
  --reference-state "$hca_reference" --timing-iters 50 --timing-warmup 10
bash "$hca_tests/run_hca_single_layer.sh" "$hca_output/cold" "${hca_args[@]}" \
  --swimlane --swimlane-cold-l2
bash "$hca_tests/run_hca_single_layer.sh" "$hca_output/after_native" "${hca_args[@]}" \
  --swimlane --swimlane-after-native
