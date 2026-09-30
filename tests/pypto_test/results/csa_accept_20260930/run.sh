#!/usr/bin/env bash
set -eo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
h="${1:?history}"; b="${2:?batch}"
# ABBA：同卡上 native→pto→pto→native，抵消卡内漂移
bash "$root/run_side.sh" native "$h" "$b" native_a
bash "$root/run_side.sh" pto    "$h" "$b" pto_a
bash "$root/run_side.sh" pto    "$h" "$b" pto_b
bash "$root/run_side.sh" native "$h" "$b" native_b
