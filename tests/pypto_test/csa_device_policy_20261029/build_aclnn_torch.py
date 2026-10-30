# SPDX-License-Identifier: Apache-2.0
"""构建独立CSA设备选档Torch ACLNN扩展；不替换现有csa_pto扩展。"""

import argparse
from pathlib import Path

import torch_npu
from csa_device_policy_20261029.nd_descriptor import normalize_nd_descriptors
from torch.utils.cpp_extension import load

parser = argparse.ArgumentParser()
parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
repo = Path(__file__).resolve().parents[3]
adapter = repo / "csrc/aclnn_torch_adapter"
npu = Path(torch_npu.__file__).resolve().parent
# Quotes in binding.cpp resolve this CSA-private header first. Native extensions
# keep using the original adapter; no shared source or SDK is patched.
(args.source / "torch/op_api_common.h").write_text(
    normalize_nd_descriptors((adapter / "op_api_common.h").read_text())
)
args.output.mkdir(parents=True, exist_ok=True)
load(
    name="csa_device_pto_npu",
    sources=[
        str(args.source / "torch/binding.cpp"),
        str(adapter / "NPUBridge.cpp"),
        str(adapter / "NPUStorageImpl.cpp"),
    ],
    extra_include_paths=[
        str(adapter),
        str(npu / "include"),
        str(npu / "include/third_party/op-plugin"),
        str(npu / "include/third_party/acl/inc"),
    ],
    extra_cflags=["-O2"],
    extra_ldflags=[f"-L{npu / 'lib'}", "-ltorch_npu"],
    build_directory=str(args.output),
    is_python_module=False,
    verbose=True,
)
print(args.output / "csa_device_pto_npu.so", flush=True)
