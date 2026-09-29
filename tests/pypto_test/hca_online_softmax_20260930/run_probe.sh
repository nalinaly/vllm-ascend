#!/usr/bin/env bash
set -eo pipefail
: "${TASK_DEVICE:?请通过task-submit自动排队}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$root/../../.." && pwd)"
source "$repo/../env-dsv4-0251rc1.sh"
export OMP_NUM_THREADS=4
export VLLM_ASCEND_ENABLE_NZ=2
export VLLM_ASCEND_PTO_CSA_ATOMIC_ADD=0
exec python "$root/probe.py" "$@"
