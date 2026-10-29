#!/usr/bin/env bash
set -eo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
vendor="$here/probe/aclnn_build_v1/_CPack_Packages/Linux/External/csa_spmd_tnd_device_pto.run/packages/vendors/csa_spmd_tnd_device_pto"
exec bash "$here/run_probe.sh" validate_cpp.py \
  --artifact "$here/probe/cpp94_v2" \
  --aclnn-library "$here/probe/aclnn_torch_v1/csa_device_pto_npu.so" \
  --aclnn-vendor "$vendor" "$@"
