#!/usr/bin/env bash
set -eo pipefail
: "${TASK_DEVICE:?请通过task-submit自动排队}"
tests="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
prefix=/data/pyptouser/qinchuanyu/pto-eager/.cache/hca-query-stream-bb4c2831-20260930
# 8K守护；长分支的短循环、压缩块边界、不同worker查询数和padding。
for case in 8192:24:5 16507:6:1 16384:2:1; do
    IFS=: read -r history batch cycles <<< "$case"
    extra=()
    suffix=""
    if [[ "$history" != 8192 ]]; then
        extra+=(--padding-graph)
    fi
    if [[ "$history" == 16384 ]]; then
        suffix="_fixed"
    fi
    bash "$tests/hca_pair_screen_20260930/run.sh" \
        "$tests/results/hca_query_stream_20260930/pair_h${history}_b${batch}${suffix}" \
        --baseline "$prefix/base" --candidate "$prefix/stream" \
        --history "$history" --batch "$batch" --cycles "$cycles" "${extra[@]}"
done
