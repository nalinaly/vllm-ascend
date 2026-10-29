# SPDX-License-Identifier: Apache-2.0
"""检查wheel安装后的正式包加载与ABI；不初始化NPU。"""

import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.package.resolve()))
    import torch

    from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd_tnd.backend import _load_kernel_package

    package = _load_kernel_package()
    assert package.device_policy and len(package.parameters) == 55
    assert not torch.npu.is_initialized()
    result = dict(
        status="PASS", library=str(package.library), policy_mode="device", parameters=55, npu_initialized=False
    )
    (args.package / "install_validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
