#!/usr/bin/env bash
set -eo pipefail
: "${TASK_DEVICE:?请通过task-submit自动排队}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
tests="$(cd "$root/.." && pwd)"

bash "$tests/hca_online_softmax_20260930/run_probe.sh" \
    --experiment-root "$root" --output "$root/probe_result.json"

# 分页直读不改算术，不能沿用512列候选的浮点差异豁免。
python - "$root/probe_result.json" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text())
assert report["status"] == "DIAGNOSTIC_COMPLETE"
for case in report["cases"].values():
    difference = case["online_vs_base"]
    assert difference["nonfinite"] == 0 and difference["max_abs"] == 0, difference
print("PAGE_PROBE_CROSS_VARIANT_EXACT", flush=True)
PY

mapfile -t sources < <(python - "$root/source.json" <<'PY'
import json
import sys
from pathlib import Path

sources = json.loads(Path(sys.argv[1]).read_text())["sources"]
print(sources["base"])
print(sources["direct_store"])
PY
)
bash "$tests/hca_pair_screen_20260930/run.sh" \
    "$tests/results/hca_direct_pages_20260930/direct_long" \
    --baseline "${sources[0]}" --candidate "${sources[1]}" --history 131072 --batch 16
