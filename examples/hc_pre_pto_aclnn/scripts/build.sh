#!/usr/bin/env bash
set -euo pipefail
# CANN set_env.sh must already be sourced. No dependency checkout is changed.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
pto_isa="${1:?Usage: build.sh PTO_ISA_ROOT [BUILD_DIR]}"
build="${2:-$root/build}"
: "${ASCEND_HOME_PATH:?Source CANN set_env.sh first}"
cmake -S "$root" -B "$build" -DPTO_ISA_ROOT="$pto_isa"
cmake --build "$build" --target binary --parallel
cmake --build "$build" --parallel
cmake --build "$build" --target package --parallel
