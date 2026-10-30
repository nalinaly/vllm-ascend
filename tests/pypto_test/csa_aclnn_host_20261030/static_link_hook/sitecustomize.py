# SPDX-License-Identifier: Apache-2.0
"""仅私有静态编译进程：将已编译策略链接进普通入口和SK的各核对象。"""

import importlib
import os
import subprocess
from pathlib import Path

root = os.environ.get("CSA_DEVICE_POLICY_BUILD")
if root:
    compiler = importlib.import_module("asc_op_compile_base.asc_op_compiler.compile_op")
    utility = importlib.import_module("asc_op_compile_base.asc_op_compiler.ascendc_common_utility")
    storage = importlib.import_module("asc_op_compile_base.asc_op_compiler.global_storage").global_var_storage
    objects = Path(root) / "objects"
    toolchain = Path(os.environ["ASCEND_HOME_PATH"]) / "bin"
    linker, objcopy = str(toolchain / "ld.lld"), str(toolchain / "llvm-objcopy")

    def link_into(target, roles, localize=False):
        target = Path(target)
        additions = [str(objects / f"{role}_{i:03d}.o") for role in roles for i in range(94)]
        temporary = target.with_suffix(".with_policy.o")
        subprocess.run([linker, "-r", "-o", str(temporary), str(target), *additions], check=True)
        if localize:
            # CANN将同一个核入口复制成SK的多份split对象，策略函数须随副本局部化。
            symbols = [f"--localize-symbol=csa_{role}_{i:03d}" for role in roles for i in range(94)]
            subprocess.run([objcopy, *symbols, str(temporary)], check=True)
        temporary.replace(target)

    original_split = compiler.split_sub_kernel_objs

    def split_with_policies(dst, tiling, info):
        if (
            info.op_type == "CsaSpmdTndDevicePto"
            and tiling.static_shape_flag
            and storage.get_variable("ascendc_enable_super_kernel")
        ):
            kernels = storage.get_variable("ascendc_sub_super_kernel_fun_names")
            for arch, kernel in kernels.items():
                role = "aic" if "cube" in arch else "aiv"
                link_into(kernel["obj_files"], (role,), localize=True)
        return original_split(dst, tiling, info)

    compiler.split_sub_kernel_objs = split_with_policies
    original_combine = compiler.link_sk_norm_combine

    def combine_with_policies(sk, normal, *args):
        if "CsaSpmdTndDevicePto" in str(sk):
            link_into(normal, ("aic", "aiv"))
        return original_combine(sk, normal, *args)

    compiler.link_sk_norm_combine = combine_with_policies
    original_run = utility.CommonUtility.run_cmd_inner

    def run_with_policies(command, stage, compile_log_path=None):
        if getattr(stage, "name", "") == "LINKRELOCATE" and any("CsaSpmdTndDevicePto" in str(part) for part in command):
            command = [*command, *[str(objects / f"{role}_{i:03d}.o") for role in ("aic", "aiv") for i in range(94)]]
        return original_run(command, stage, compile_log_path)

    utility.CommonUtility.run_cmd_inner = staticmethod(run_with_policies)
