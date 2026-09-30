#!/usr/bin/env bash
set -eo pipefail
experiment_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
experiment_group="${1:?a or b}"
shift
if [[ "$experiment_group" == a ]]; then
  bash "$experiment_dir/pair.sh" 131072 4 final_ops_v2 matrix "$@"
  bash "$experiment_dir/pair.sh" 8192 16 final_ops_v2 matrix "$@"
  bash "$experiment_dir/pair.sh" 131072 16 final_ops_v2 matrix "$@"
  bash "$experiment_dir/pair.sh" 8192 32 final_ops_v2 matrix "$@"
elif [[ "$experiment_group" == b ]]; then
  bash "$experiment_dir/pair.sh" 131072 8 final_ops_v2 matrix "$@"
  bash "$experiment_dir/pair.sh" 8192 24 final_ops_v2 matrix "$@"
  bash "$experiment_dir/pair.sh" 131072 24 final_ops_v2 matrix "$@"
else
  exit 2
fi
