# SPDX-License-Identifier: Apache-2.0
"""只重编Torch包装；复用已验证的设备二进制，不触碰公共安装。"""

import runpy
import shutil
import sys
from pathlib import Path

from csa_device_policy_20261029.prepare_aclnn import simplify_torch_run

here = Path(__file__).resolve().parent
previous = here.parent / "csa_device_policy_20261029/probe"
output = here / "results/wrapper"
source = output / "source"
source.mkdir(parents=True, exist_ok=True)
shutil.copytree(previous / "aclnn_v1/include", source / "include", dirs_exist_ok=True)
(source / "torch").mkdir(exist_ok=True)
(source / "torch/binding.cpp").write_text(simplify_torch_run((previous / "aclnn_v1/torch/binding.cpp").read_text()))
sys.argv = ["build_aclnn_torch.py", "--source", str(source), "--output", str(output / "build")]
runpy.run_path(str(previous.parent / "build_aclnn_torch.py"), run_name="__main__")
site = here / "results/site/csa_device_pto_ops"
site.mkdir(parents=True, exist_ok=True)
old = previous / "aclnn_installed_v1/csa_device_pto_ops"
for name in ("__init__.py", "interface.json", "package_manifest.json"):
    shutil.copy2(old / name, site / name)
if not (site / "ascend").exists():
    (site / "ascend").symlink_to(old / "ascend", target_is_directory=True)
(site / "lib").mkdir(exist_ok=True)
shutil.copy2(output / "build/csa_device_pto_npu.so", site / "lib/csa_device_pto_npu.so")
print(f"HOST_WRAPPER_READY {site}", flush=True)
