# SPDX-License-Identifier: Apache-2.0
"""仅本构建子进程加载：给CANN9.2最终device链接补入独立编译的94份策略。

不修改安装的CANN文件。CANN继续生成MIX入口、tiling元数据和算子包；本目录
只在run_aclnn_build.sh的kernel阶段进入PYTHONPATH，不能用于运行时环境。
"""

import os
from pathlib import Path

root = os.environ.get("CSA_DEVICE_POLICY_BUILD")
if root:
    from asc_op_compile_base.asc_op_compiler.ascendc_common_utility import CommonUtility

    original = CommonUtility.run_cmd_inner

    def with_policy_objects(cmds, stage, compile_log_path=None):
        if getattr(stage, "name", "") == "LINKRELOCATE" and any(
            "csaspmdtnddevicepto" in "".join(c.lower() for c in str(arg) if c.isalnum()) for arg in cmds
        ):
            directory = Path(root)
            objects = [directory / f"objects/{role}_{i:03d}.o" for role in ("aic", "aiv") for i in range(94)]
            if not all(path.is_file() for path in objects):
                raise RuntimeError("CSA设备策略对象未完成，拒绝生成缺少分支的ACLNN包")
            cmds = [*cmds, *map(str, objects)]
            with (directory / "cann_link_commands.log").open("a") as stream:
                stream.write(" ".join(cmds) + "\n")
        return original(cmds, stage, compile_log_path)

    CommonUtility.run_cmd_inner = staticmethod(with_policy_objects)
