#!/usr/bin/env bash
# 五点收尾：固定最终版本，同卡Native/旧PTO/最终PTO/Native，再独立导出泳道。
set -eo pipefail
: "${TASK_DEVICE:?请通过task-submit自动排队}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
tests="$(dirname "$here")"
history="${1:?history}"
batch="${2:?batch}"
case_name="h${history}_b${batch}"
out="$tests/results/hca_five_closeout_20260930/seven/$case_name"
test ! -e "$out"
for pass in p1_native p2_before p3_final p4_native; do
    side=native
    super_kernel=1
    args=()
    if [[ "$pass" == p2_before || "$pass" == p3_final ]]; then
        side=pto
        super_kernel=0
        key=before_qb
        [[ "$pass" == p3_final ]] && key=production
        source_dir="$(python - "$here/source.json" "$key" <<'PY'
import json
import sys
from pathlib import Path
print(json.loads(Path(sys.argv[1]).read_text())["sources"][sys.argv[2]])
PY
)"
        args+=(--operator-source "$source_dir" --save-state)
    fi
    bash "$tests/run_hca_compiled_case.sh" "$out/$pass" "$side" \
        --history "$history" --batch "$batch" --super-kernel "$super_kernel" \
        --inplace-pass 1 --iters 20 --warmup 5 --profile-replays 30 "${args[@]}"
done
final_source="$(python - "$here/source.json" <<'PY'
import json
import sys
from pathlib import Path
print(json.loads(Path(sys.argv[1]).read_text())["sources"]["production"])
PY
)"
bash "$tests/run_hca_compiled_case.sh" "$out/swimlane_final" pto \
    --history "$history" --batch "$batch" --super-kernel 0 --inplace-pass 1 \
    --operator-source "$final_source" --iters 20 --warmup 5 --swimlane
