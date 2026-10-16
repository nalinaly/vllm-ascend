# SPDX-License-Identifier: Apache-2.0
"""Build the Torch binding on CPU. Source CANN set_env.sh first."""
from pathlib import Path

import torch_npu
from torch.utils.cpp_extension import load

root = Path(__file__).resolve().parents[1]
npu = Path(torch_npu.__file__).resolve().parent
adapter = root.parents[1] / "csrc/aclnn_torch_adapter"
build = root / "build" / "torch"
build.mkdir(parents=True, exist_ok=True)
load(name="hcpre_pto_torch", sources=[str(root / "torch/binding.cpp"), str(adapter / "NPUBridge.cpp"),
                                              str(adapter / "NPUStorageImpl.cpp")],
     extra_include_paths=[str(adapter), str(npu / "include"),
                          str(npu / "include/third_party/op-plugin"),
                          str(npu / "include/third_party/acl/inc")],
     extra_cflags=["-O2"],
     extra_ldflags=[f"-L{npu / 'lib'}", "-ltorch_npu"],
     build_directory=str(build), is_python_module=False, verbose=True)
print(build / "hcpre_pto_torch.so")
