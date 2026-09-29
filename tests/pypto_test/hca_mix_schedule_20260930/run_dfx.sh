#!/usr/bin/env bash
set -eo pipefail
: "${TASK_DEVICE:?请通过task-submit自动排队}"
tests="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
experiment="${1:?实验目录名}"
shift
for side in "$@"; do
    [[ "$side" == --device ]] && break
    source_dir="$(python - "$tests/$experiment/source.json" "$side" <<'PY'
import json
import sys
from pathlib import Path
print(json.loads(Path(sys.argv[1]).read_text())["sources"][sys.argv[2]])
PY
)"
    out="$tests/results/$experiment/dfx/$side"
    test ! -e "$out"
    bash "$tests/run_hca_compiled_case.sh" "$out" pto --history 131072 --batch 16 \
        --super-kernel 0 --operator-source "$source_dir" --iters 20 --warmup 5 --swimlane
done
