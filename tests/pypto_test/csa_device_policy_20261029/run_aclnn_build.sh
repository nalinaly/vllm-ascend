#!/usr/bin/env bash
set -eo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo=$(cd "$here/../../.." && pwd)
workspace=$(dirname "$repo")
source "$workspace/env-dsv4-0251rc1.sh"
source "$repo/tests/pypto_test/csa_shared_toolchain/env.sh"
src="$here/probe/aclnn_v1"
build="$here/probe/aclnn_build_v1"
if [[ ${1:-host} == kernel ]]; then
  python - "$src/build.json" <<'PY'
import json
import sys
import time
from pathlib import Path
while True:
    state = json.loads(Path(sys.argv[1]).read_text())
    if state['failed']:
        raise SystemExit('策略对象编译失败，停止CANN打包')
    if state['completed'] == 188:
        break
    time.sleep(5)
PY
  export CSA_DEVICE_POLICY_BUILD="$src"
  export PYTHONPATH="$here/aclnn_link_hook${PYTHONPATH:+:$PYTHONPATH}"
  cmake --build "$build" --target binary --parallel 32
  cmake --build "$build" --parallel 32
  cmake --build "$build" --target package --parallel 32
  exit
fi
cmake -S "$src" -B "$build" -DPTO_ISA_ROOT="$CSA_SHARED_SIMPLER_ROOT/build/pto-isa"
cmake --build "$build" --target cust_opapi cust_op_proto cust_optiling --parallel 32
