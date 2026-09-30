#!/usr/bin/env bash
# 单档同卡内 sk0_a → sk1_a → sk1_b → sk0_b 的 ABBA，抵消卡内漂移。
set -eo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
h="${1:?history}"; b="${2:?batch}"
RUN_SLOT=a bash "$root/run_side.sh" 0 "$h" "$b"
RUN_SLOT=a bash "$root/run_side.sh" 1 "$h" "$b"
RUN_SLOT=b bash "$root/run_side.sh" 1 "$h" "$b"
RUN_SLOT=b bash "$root/run_side.sh" 0 "$h" "$b"
