#!/usr/bin/env bash
# Run standalone CSA validation under devices allocated by task-submit.
set -eo pipefail

: "${TASK_DEVICE:?Submit this script through task-submit with NPU devices}"
phase="${1:?Pass native_ops, native_layout, metadata, native_forward, native_adapters, full_compare, full_replay, continuous, qr_boundary, weights_boundary, scatter_offset, or dp_metadata}"
IFS=, read -r -a allocated_devices <<< "$TASK_DEVICE"
required_devices=1
if [[ "$phase" == dp_metadata ]]; then
  required_devices=2
fi
if [[ "${#allocated_devices[@]}" -ne "$required_devices" ]]; then
  printf '%s requires exactly %s allocated devices.\n' "$phase" "$required_devices" >&2
  exit 2
fi
workspace_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
source "$workspace_root/env.sh"
output_dir="${2:?Pass an output directory}"
checkpoint="${3:-}"
mkdir -p "$output_dir/plog"
output_dir="$(realpath "$output_dir")"
export ASCEND_PROCESS_LOG_PATH="$output_dir/plog"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
cd "$workspace_root/vllm-ascend-dsv4-pto"

case "$phase" in
  sparse_boundary)
    python tests/pypto_test/dsv4_csa_sparse_boundary.py \
      --device "$TASK_DEVICE" --capture "${checkpoint:?Pass the captured output_boundary.pt}" \
      --output-dir "$output_dir" "${@:4}"
    ;;
  weights_boundary)
    python tests/pypto_test/dsv4_csa_weights_boundary.py \
      --device "$TASK_DEVICE" --capture "${checkpoint:?Pass the captured query_boundary.pt}" \
      --output-dir "$output_dir" "${@:4}"
    ;;
  scatter_offset)
    python tests/pypto_test/dsv4_csa_scatter_offset.py \
      --device "$TASK_DEVICE" --output-dir "$output_dir"
    ;;
  dp_metadata)
    : "${checkpoint:?This phase requires a checkpoint directory}"
    export ASCEND_RT_VISIBLE_DEVICES="$TASK_DEVICE"
    python tests/pypto_test/dsv4_csa_dp_metadata.py \
      --checkpoint "$checkpoint" --output-dir "$output_dir" "${@:4}"
    ;;
  native_ops)
    for op in compressor qli sas; do
      python tests/pypto_test/dsv4_csa_native_ops_smoke.py \
        --device "$TASK_DEVICE" --op "$op" --output-dir "$output_dir"
    done
    ;;
  native_layout|metadata|native_forward|native_adapters|full_compare|full_replay|continuous|qr_boundary)
    : "${checkpoint:?This phase requires a checkpoint directory}"
    case "$phase" in
      native_layout) entry=dsv4_csa_native_layout.py ;;
      metadata) entry=dsv4_csa_all_groups.py ;;
      native_forward) entry=dsv4_csa_native_forward.py ;;
      native_adapters) entry=dsv4_csa_native_adapters.py ;;
      full_compare) entry=dsv4_csa_full_compare.py ;;
      full_replay) entry=dsv4_csa_full_replay.py ;;
      continuous) entry=dsv4_csa_continuous.py ;;
      qr_boundary) entry=dsv4_csa_qr_boundary.py ;;
    esac
    python "tests/pypto_test/$entry" --device "$TASK_DEVICE" \
      --checkpoint "$checkpoint" --output-dir "$output_dir" "${@:4}"
    ;;
  *)
    printf 'Unknown CSA validation phase: %s\n' "$phase" >&2
    exit 2
    ;;
esac
