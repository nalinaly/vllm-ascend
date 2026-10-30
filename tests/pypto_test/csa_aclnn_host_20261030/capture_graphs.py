# SPDX-License-Identifier: Apache-2.0
"""仅初始化期保存SK优化前后真实图，供离线审计融合成员。"""

import json
import os
from pathlib import Path


def install(directory):
    # 私有实验源码显式调用；不接入业务，也不在稳态replay增加回调。
    if int(os.environ.get("VLLM_DP_RANK", "-1")) != 0:
        return
    import torch

    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    original = torch.npu.NPUGraph.super_kernel_optimize
    sequence = 0

    def optimize(graph, *args, **kwargs):
        nonlocal sequence
        sequence += 1
        report = {"sequence": sequence, "pid": os.getpid()}

        def dump(phase):
            target = root / f"pid{os.getpid()}_{sequence:02d}_{phase}.json"
            try:
                graph.debug_dump(str(target))
                report[phase] = str(target)
            except RuntimeError as error:
                # 调试输出不支持时记录事实，不能让额外审计改变正常优化流程。
                report[phase + "_error"] = str(error)

        dump("before_sk")
        result = original(graph, *args, **kwargs)
        dump("after_sk")
        (root / f"pid{os.getpid()}_{sequence:02d}_record.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n"
        )
        return result

    torch.npu.NPUGraph.super_kernel_optimize = optimize
