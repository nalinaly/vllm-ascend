# SPDX-License-Identifier: Apache-2.0
"""HCA 整层 CPU 编译，不初始化设备，也不宣称数值通过。"""

import argparse
from pathlib import Path

from dsv4_csa_env import activate, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--operator-source", type=Path, help="编译已冻结的 ops/pypto 源码目录")
    args = parser.parse_args()
    activate()
    if args.operator_source:
        import vllm_ascend.ops.pypto as operator_package

        source = args.operator_source.resolve()
        if not (source / "deepseek_v4_flash_hca/decode_hca.py").is_file():
            raise ValueError(f"无效的 HCA 源码目录：{source}")
        operator_package.__path__ = [str(source)]
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"scope": "HCA 整层 CPU 编译；未执行设备数值测试", "status": "RUNNING"}
    report["operator_source"] = str(args.operator_source.resolve()) if args.operator_source else "当前 worktree"
    try:
        from pypto.runtime import RunConfig

        from vllm_ascend.ops.pypto.deepseek_v4_flash_hca.decode_hca import decode_hca_tp1_layer_test

        if args.build:
            program = decode_hca_tp1_layer_test.warmup(config=RunConfig(
                platform="a2a3", save_kernels=True,
                save_kernels_dir=str((args.output / "build").resolve()),
            )).program
        else:
            program = decode_hca_tp1_layer_test.lower(config=RunConfig(platform="a2a3"))
        (args.output / "lowered.py").write_text(str(program))
        report.update(status="PASS", build=args.build, parameters=decode_hca_tp1_layer_test.param_names)
    except BaseException as exc:
        report.update(status="FAIL", error=repr(exc))
        raise
    finally:
        write_json(args.output / "report.json", report)


if __name__ == "__main__":
    main()
