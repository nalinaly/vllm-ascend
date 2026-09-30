#!/usr/bin/env bash
set -eo pipefail
pair_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
pair_repo="$(cd "$pair_dir/../../.." && pwd)"
pair_root="$pair_repo/tests/pypto_test/results/atomic_add_seven_20260930"
pair_history="${1:?history}"
pair_batch="${2:?batch}"
pair_first="${3:-0}"
# task-submit 会在命令末尾附加 --device N；设备由 TASK_DEVICE 读取。
if [[ "$pair_first" == --device ]]; then pair_first=0; fi
[[ "$pair_first" == 0 || "$pair_first" == 1 ]]
pair_order=("$pair_first" "$((1 - pair_first))")
for pair_atomic in "${pair_order[@]}"; do
  bash "$pair_dir/run.sh" "$pair_root/h${pair_history}_b${pair_batch}/atomic${pair_atomic}" \
    "$pair_atomic" --history "$pair_history" --batch "$pair_batch"
done
