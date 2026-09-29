#!/usr/bin/env bash
set -eo pipefail
: "${TASK_DEVICE:?请通过task-submit自动排队}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for side in mix_gm_bf16 mix_ub_bf16; do
    bash "$here/run.sh" "$side" 131072 16
done
