#!/usr/bin/env bash
set -eo pipefail
result_root="$(realpath -m "${1:?结果根目录，须含 frozen_ops_pypto}")"
case_operator="${2:?csa/hca}"
case_history="${3:?history}"
case_label="${4:?label}"
shift 4
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "$script_dir/run.sh" "$result_root/${case_label}_ring" \
  --operator "$case_operator" --runtime tensormap_and_ringbuffer --history "$case_history" \
  --source "$result_root/frozen_ops_pypto" "$@"
bash "$script_dir/run.sh" "$result_root/${case_label}_hbg" \
  --operator "$case_operator" --runtime host_build_graph --history "$case_history" \
  --source "$result_root/frozen_ops_pypto" --reference "$result_root/${case_label}_ring/states.pt" "$@"
