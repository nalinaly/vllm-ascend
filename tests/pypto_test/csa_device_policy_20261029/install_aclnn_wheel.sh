#!/usr/bin/env bash
set -eo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo=$(cd "$here/../../.." && pwd)
workspace=$(dirname "$repo")
source "$workspace/env-dsv4-0251rc1.sh"
source "$repo/tests/pypto_test/csa_shared_toolchain/env.sh"
destination="$here/probe/aclnn_installed_v1"
test ! -e "$destination"
python -m pip install --no-deps --no-compile --target "$destination" "$here"/probe/aclnn_package_v1/dist/*.whl
exec bash "$here/run_probe.sh" verify_aclnn_install.py --package "$destination"
