#!/usr/bin/env bash
set -eo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo=$(cd "$here/../../.." && pwd)
workspace=$(dirname "$repo")
source_name=${1:?私有源码目录名}
script=${2:?本目录脚本名}
shift 2
source "$workspace/env-dsv4-0251rc1.sh"
source "$repo/tests/pypto_test/csa_shared_toolchain/env.sh"
export VLLM_ASCEND_ENABLE_NZ=2
frozen="$here/results/$source_name"
export PYTHONPATH="$frozen:$frozen/tests/pypto_test:$repo/tests/pypto_test:$workspace/csa-repro-20261024-current/vllm:$PYTHONPATH"
cd "$repo"
exec python "$frozen/tests/pypto_test/csa_toolchain_repro_20261019/bootstrap.py" \
    --shared-toolchain --workspace-root "$workspace" --vllm-root "$workspace/csa-repro-20261024-current/vllm" \
    "$here/$script" "$@"
