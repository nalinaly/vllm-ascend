#!/usr/bin/env bash
set -eo pipefail
: "${TASK_DEVICE:?Submit through task-submit --device auto}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
history="${1:?history}"
batch="${2:?batch}"
mapfile -t candidates < <(python - "$root/compile_results.json" <<'PY'
import json
import sys
for item in json.load(open(sys.argv[1])):
    if item['exit_code'] == 0 and item['name'] != 'baseline':
        print(item['name'])
PY
)
bash "$root/run_side.sh" baseline timing "$history" "$batch" baseline_start
bash "$root/run_side.sh" baseline swimlane "$history" "$batch" baseline_start
sequence=("${candidates[@]}")
if [[ "$history" == 8192 ]]; then
    sequence=()
    for ((i=${#candidates[@]}-1; i>=0; i--)); do sequence+=("${candidates[i]}"); done
fi
for name in "${sequence[@]}"; do
    bash "$root/run_side.sh" "$name" timing "$history" "$batch"
    bash "$root/run_side.sh" "$name" swimlane "$history" "$batch"
done
bash "$root/run_side.sh" baseline timing "$history" "$batch" baseline_end
