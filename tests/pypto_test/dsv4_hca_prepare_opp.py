# SPDX-License-Identifier: Apache-2.0
"""造一个可写的私有 OPP 根，不改动共用的 CANN 载荷。

静态 kernel 编译会把生成的 .run 包装进 `$ASCEND_OPP_PATH/static_kernel`，
而 Native 与 PTO 两侧必须各自从一个干净的 static_kernel 目录开始，否则装包互相污染、
比出来的数字不可归因。共用的 CANN 目录是只读且被其他会话共用的，不能直接往里写，
所以这里把 CANN 的 opp 子项逐个软链到私有目录，只把 static_kernel 留空。

移植自 CSA 会话的 coefficients_seven_experiment/prepare_opp.py，逻辑保持一致。
"""

import argparse
import os
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    cann = Path(os.environ["ASCEND_HOME_PATH"])
    base = args.destination.resolve()
    opp = base / "opp"
    opp.mkdir(parents=True, exist_ok=True)
    for source in (cann / "opp").iterdir():
        # static_kernel 必须留空：本次运行生成的静态包只能落在这里。
        if source.name == "static_kernel":
            continue
        destination = opp / source.name
        if not destination.exists():
            destination.symlink_to(source, target_is_directory=source.is_dir())
    platform = base / "aarch64-linux"
    if not platform.exists():
        platform.symlink_to(cann / "aarch64-linux", target_is_directory=True)
    # CANN 不会从内置的浅层软链里注册 tiling，这条目录链和它的两个库必须实体化（约 40 MiB）。
    tiling = Path("built-in/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64")
    for relative in [*reversed(tiling.parents[:-1]), tiling]:
        directory = opp / relative
        if directory.is_symlink():
            directory.unlink()
        directory.mkdir(exist_ok=True)
        for source in (cann / "opp" / relative).iterdir():
            target = directory / source.name
            if not target.exists():
                target.symlink_to(source, target_is_directory=source.is_dir())
    for source in (cann / "opp" / tiling).glob("*.so"):
        target = opp / tiling / source.name
        if target.is_symlink():
            target.unlink()
        if not target.exists():
            shutil.copyfile(source, target)
    print(opp)


if __name__ == "__main__":
    main()
