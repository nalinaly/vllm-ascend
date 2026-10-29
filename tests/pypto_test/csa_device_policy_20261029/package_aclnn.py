# SPDX-License-Identifier: Apache-2.0
"""把独立vendor和Torch库打包到csa_device_pto_ops，不覆盖旧62档包。"""

import argparse
import json
import platform
import shutil
import sys
from pathlib import Path

import torch
import torch_npu

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
here = Path(__file__).resolve().parent
project = here / "probe/aclnn_v1"
vendor = (
    here
    / "probe/aclnn_build_v1/_CPack_Packages/Linux/External/csa_spmd_tnd_device_pto.run"
    / "packages/vendors/csa_spmd_tnd_device_pto"
)
package = args.output / "csa_device_pto_ops"
package.mkdir(parents=True, exist_ok=False)
(package / "lib").mkdir()
shutil.copytree(vendor, package / "ascend/vendors/csa_spmd_tnd_device_pto")
shutil.copy2(here / "probe/aclnn_torch_v1/csa_device_pto_npu.so", package / "lib/csa_device_pto_npu.so")
shutil.copy2(project / "generated/interface.json", package / "interface.json")
(package / "package_manifest.json").write_text(
    json.dumps(
        dict(
            schema=1,
            policy_mode="device",
            policy_count=94,
            tiling_key=0,
            tensor_count=55,
            torch=torch.__version__,
            torch_npu=torch_npu.__version__,
            architecture=platform.machine(),
            python=list(sys.version_info[:2]),
            cann="9.2.0-beta.2",
            superkernel_validated=False,
        ),
        indent=2,
    )
    + "\n"
)
(package / "__init__.py").write_text("""# SPDX-License-Identifier: Apache-2.0
import ctypes
import os
from functools import lru_cache
from pathlib import Path

@lru_cache(maxsize=1)
def ensure_vendor():
    vendor = Path(__file__).resolve().parent / "ascend/vendors/csa_spmd_tnd_device_pto"
    previous = os.environ.get("ASCEND_CUSTOM_OPP_PATH", "")
    paths = [str(vendor), *[p for p in previous.split(":") if p and p != str(vendor)]]
    os.environ["ASCEND_CUSTOM_OPP_PATH"] = ":".join(paths)
    return ctypes.CDLL(str(vendor / "op_api/lib/libcust_opapi.so"), mode=ctypes.RTLD_GLOBAL)

__version__ = "0.2.0"
""")
(args.output / "setup.py").write_text("""from setuptools import setup, find_packages
from setuptools.dist import Distribution
class BinaryDistribution(Distribution):
    def has_ext_modules(self): return True
from pathlib import Path
root=Path(__file__).resolve().parent
files=[str(p.relative_to(root/"csa_device_pto_ops")) for p in (root/"csa_device_pto_ops").rglob("*") if p.is_file()]
setup(name="csa-device-pto-ops",version="0.2.0",packages=find_packages(),
      package_data={"csa_device_pto_ops":files},include_package_data=True,zip_safe=False,distclass=BinaryDistribution)
""")
print(args.output, flush=True)
