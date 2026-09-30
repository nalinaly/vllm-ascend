# SPDX-License-Identifier: Apache-2.0
"""Run the shared single-card harness with frozen kernels and the HBG ABI."""

import argparse
import runpy
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    tests = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(tests))
    from dsv4_csa_env import activate

    activate()
    import vllm_ascend.ops.pypto as ops

    frozen = args.output / "frozen_ops_pypto"
    if not frozen.is_dir():
        raise ValueError(f"Freeze vllm_ascend/ops/pypto before submission: {frozen}")
    ops.__path__ = [str(frozen)]
    sys.argv = [
        str(tests / "dsv4_csa_single_layer.py"),
        "--checkpoint", "/data/model/DeepSeek-V4-Flash-0731-w8a8",
        "--output", str(args.output), "--device", "0", "--layer-index", "2",
        "--variant", "performance", "--weight-nz-mode", "2", "--atomic-add", "0",
        "--runtime", "host_build_graph", *remaining,
    ]
    runpy.run_path(sys.argv[0], run_name="__main__")


if __name__ == "__main__":
    main()
