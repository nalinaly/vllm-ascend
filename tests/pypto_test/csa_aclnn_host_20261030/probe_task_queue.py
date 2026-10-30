# SPDX-License-Identifier: Apache-2.0
"""读取已安装torch_npu的Host队列配置；不初始化设备、不修改配置。"""

import argparse
import json
import os
from pathlib import Path

import torch_npu
from offline_pd.event_mode import get_task_queue_enable


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    library = Path(torch_npu.__file__).parent / "lib/libtorch_npu.so"
    before = torch_npu.npu.is_initialized()
    mode = get_task_queue_enable()
    after = torch_npu.npu.is_initialized()
    report = {
        "torch_npu": torch_npu.__version__,
        "library": str(library),
        "environment": {key: os.environ.get(key) for key in ("TASK_QUEUE_ENABLE", "ASCEND_LAUNCH_BLOCKING")},
        "task_queue_enable": mode,
        "npu_initialized_before": before,
        "npu_initialized_after": after,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.output:
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    if before or after:
        raise RuntimeError("Host配置探针不应初始化NPU")


if __name__ == "__main__":
    main()
