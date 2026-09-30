#!/usr/bin/env bash
set -eo pipefail
experiment_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
experiment_repo="$(cd "$experiment_dir/../../.." && pwd)"
experiment_root="$experiment_repo/tests/pypto_test/results/precision_port_20260930"
experiment_history="${1:?history}"
experiment_batch="${2:?batch}"
experiment_candidate="${3:?snapshot name}"
experiment_mode="${4:?guard or matrix}"
shift 4
if [[ "$experiment_mode" == guard ]]; then
  bash "$experiment_dir/run.sh" "$experiment_root/h${experiment_history}_b${experiment_batch}/baseline" precision "$experiment_root/baseline_ops" --history "$experiment_history" --batch "$experiment_batch" "$@"
  bash "$experiment_dir/run.sh" "$experiment_root/h${experiment_history}_b${experiment_batch}/$experiment_candidate" precision "$experiment_root/$experiment_candidate" --history "$experiment_history" --batch "$experiment_batch" "$@"
else
  bash "$experiment_dir/run.sh" "$experiment_root/matrix/h${experiment_history}_b${experiment_batch}/performance" performance "$experiment_root/$experiment_candidate" --history "$experiment_history" --batch "$experiment_batch" "$@"
  bash "$experiment_dir/run.sh" "$experiment_root/matrix/h${experiment_history}_b${experiment_batch}/precision" precision "$experiment_root/$experiment_candidate" --history "$experiment_history" --batch "$experiment_batch" "$@"
fi
