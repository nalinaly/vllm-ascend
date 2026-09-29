#!/usr/bin/env bash
set -eo pipefail
: "${TASK_DEVICE:?请通过task-submit排队}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
history="${1:?history}"
batch="${2:?batch}"
shift 2
bash "$root/run.sh" reference "$history" "$batch" "$@"
bash "$root/run.sh" performance "$history" "$batch" "$@"
