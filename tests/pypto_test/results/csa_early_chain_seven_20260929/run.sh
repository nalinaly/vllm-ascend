#!/usr/bin/env bash
set -eo pipefail
: "${TASK_DEVICE:?通过 task-submit --device auto 提交五个缺口}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
test -f "$root/../csa_indexer_early_chain_20260929/production_parse.json"
test -f "$root/../csa_indexer_early_chain_20260929/boundary/summary.json"
# The exact frozen candidate already has 128K/B16 and 8K/B24 timing/DFX.
# Fill only the remaining cases; Native keeps its completed latest-standard baseline.
for case_spec in 131072:4 131072:8 131072:24 8192:16 8192:32; do
    for phase in timing swimlane; do
        bash "$root/run_side.sh" candidate "$phase" "${case_spec%:*}" "${case_spec#*:}"
    done
done
