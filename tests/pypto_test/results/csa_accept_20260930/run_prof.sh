#!/usr/bin/env bash
# 每档采一次 Native（不开 SuperKernel）与一次 PTO 的 PyTorch profiling。
# Native 不开 SK 的理由：SuperKernel 把多个算子融进一个 sk_* kernel，
# 逐算子耗时无法归属；关掉才能拿到 Native 各独立算子的时长与 PTO 对照。
set -eo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
h="${1:?history}"; b="${2:?batch}"
export ACCEPT_PROFILE=1 ACCEPT_NATIVE_SK=0
bash "$root/run_side.sh" native "$h" "$b" prof_native_nosk
bash "$root/run_side.sh" pto    "$h" "$b" prof_pto
