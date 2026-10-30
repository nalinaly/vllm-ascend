#!/usr/bin/env bash
set -eo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo=$(cd "$here/../../.." && pwd)
workspace=$(dirname "$repo")
source "$workspace/env-dsv4-0251rc1.sh"
source "$repo/tests/pypto_test/csa_shared_toolchain/env.sh"
export VLLM_ASCEND_ENABLE_NZ=2
export PYTHONPATH="$repo:$workspace/csa-repro-20261024-current/vllm:$repo/tests/pypto_test:$here/results/site:$repo/tests/pypto_test/csa_device_policy_20261029/probe/aclnn_installed_v1:$PYTHONPATH"
# 和普通自定义ACLNN相同，在进程初始化前配置OPP；不能等首次算子调用才补路径。
export ASCEND_CUSTOM_OPP_PATH="$repo/tests/pypto_test/csa_device_policy_20261029/probe/aclnn_installed_v1/csa_device_pto_ops/ascend/vendors/csa_spmd_tnd_device_pto:${ASCEND_CUSTOM_OPP_PATH:-}"
cd "$repo"
script=$1
shift
exec python tests/pypto_test/csa_toolchain_repro_20261019/bootstrap.py \
  --shared-toolchain --vllm-root "$workspace/csa-repro-20261024-current/vllm" \
  "$here/$script" "$@"
