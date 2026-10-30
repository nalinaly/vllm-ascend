# SPDX-License-Identifier: Apache-2.0
"""把独立vendor和Torch库打包到csa_device_pto_ops，不覆盖旧62档包。"""

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
import torch_npu

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--torch-library", type=Path, help="默认使用已有Torch库；可指定去除重复校验的新库")
parser.add_argument("--version", default="0.3.0")
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
torch_library = args.torch_library or here / "probe/aclnn_torch_v1/csa_device_pto_npu.so"
shutil.copy2(torch_library, package / "lib/csa_device_pto_npu.so")
impl = package / "ascend/vendors/csa_spmd_tnd_device_pto/op_impl/ai_core/tbe/csa_spmd_tnd_device_pto_impl"
shutil.copytree(project / "objects", impl / "objects")
objcopy = Path(os.environ["ASCEND_HOME_PATH"]) / "bin/llvm-objcopy"


def prepare_policy_object(path):
    # 这些是__aicore__函数，真正核入口由CANN生成。其独立编译的全局核类型
    # record不能并入SK绑定表；DWARF/local调试符号也不应被四份split重复打包。
    subprocess.run(
        [str(objcopy), "--strip-debug", "--discard-all", "--remove-section=.ascend.meta", str(path)], check=True,
    )


with ThreadPoolExecutor(max_workers=16) as pool:
    list(pool.map(prepare_policy_object, (impl / "objects").glob("*.o")))
shutil.copy2(here / "aclnn_policy_compile.py", impl / "dynamic/aclnn_policy_compile.py")
# 打包与公共manifest相同版本的PTO-ISA头文件，静态编译不再依赖原机器绝对路径。
shared = json.loads((here.parent / "csa_shared_toolchain/manifest.json").read_text())
workspace = here.parents[3]
isa = workspace / shared["simpler"]["source"] / "build/pto-isa/include"
shutil.copytree(isa, impl / "include", dirs_exist_ok=True)
shutil.copytree(isa.parent / "pkg_inc", impl / "pkg_inc")
loader = impl / "dynamic/csa_spmd_tnd_device_pto.py"
code = loader.read_text()
code = code.replace("'-I" + str(isa) + "'", "'-I' + os.path.join(os.path.dirname(__file__), '../include')")
old = "    compile_op(src, origin_func_name, op_info, options, code_channel, '{}', {'valueDepend': {}})"
new = """    # CANN/TorchAir普通静态编译自动使用包内策略对象；无服务环境patch。
    import importlib.util
    helper_path = os.path.join(os.path.dirname(__file__), 'aclnn_policy_compile.py')
    helper_spec = importlib.util.spec_from_file_location('csa_device_policy_compile', helper_path)
    helper = importlib.util.module_from_spec(helper_spec)
    helper_spec.loader.exec_module(helper)
    helper.compile_with_policy_objects(
        compile_op, os.path.join(os.path.dirname(__file__), '../objects'),
        src, origin_func_name, op_info, options, code_channel, '{}', {'valueDepend': {}})"""
assert code.count(old) == 1
loader.write_text(code.replace(old, new))
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
            static_policy_link="package_compiler",
            shared_toolchain=shared["release"],
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

__version__ = "VERSION_PLACEHOLDER"
""")
(package / "__init__.py").write_text((package / "__init__.py").read_text().replace("VERSION_PLACEHOLDER", args.version))
(args.output / "setup.py").write_text("""from setuptools import setup, find_packages
from setuptools.dist import Distribution
class BinaryDistribution(Distribution):
    def has_ext_modules(self): return True
from pathlib import Path
root=Path(__file__).resolve().parent
files=[str(p.relative_to(root/"csa_device_pto_ops")) for p in (root/"csa_device_pto_ops").rglob("*") if p.is_file()]
setup(name="csa-device-pto-ops",version="VERSION_PLACEHOLDER",packages=find_packages(),
      package_data={"csa_device_pto_ops":files},include_package_data=True,zip_safe=False,distclass=BinaryDistribution)
""")
(args.output / "setup.py").write_text(
    (args.output / "setup.py").read_text().replace("VERSION_PLACEHOLDER", args.version)
)
print(args.output, flush=True)
