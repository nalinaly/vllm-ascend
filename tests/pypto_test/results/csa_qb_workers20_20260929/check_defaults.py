"""仅CPU编译共享Q_B的默认调用，确认ND/NZ均仍派发24，不改变精度版默认行为。"""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parents[4] / ".cache/csa-qb-workers20-f4861832-v3-candidate"
sys.path.insert(0, str(SOURCE / "tests/pypto_test"))
from dsv4_csa_env import activate  # noqa: E402

activate()
import pypto.language as pl  # noqa: E402
from pypto.runtime import RunConfig  # noqa: E402

from vllm_ascend.ops.pypto.deepseek_v4_flash_dspark.q_projection import (  # noqa: E402
    QUANT_WEIGHT_LAYOUT,
    q_proj_q_matmul,
)


@pl.jit
def default_qproj(
    weight: pl.Tensor[[1024, 32768], pl.INT8, QUANT_WEIGHT_LAYOUT],
    x: pl.Tensor[[512, 1024], pl.INT8],
    out: pl.Out[pl.Tensor[[64, 32768], pl.INT32]],
):
    ready = pl.system.task_dummy(deps=[])
    out, tid = q_proj_q_matmul(weight, x, out, 64, ready)
    return out


def main():
    mode = os.environ["VLLM_ASCEND_ENABLE_NZ"]
    destination = ROOT / f"compiled_default_v3_mode{mode}"
    compiled = default_qproj.compile(config=RunConfig(
        platform="a2a3", save_kernels=True, save_kernels_dir=str(destination)))
    compiled.load()
    paths = list((destination / "orchestration").glob("*.cpp"))
    text = "\n".join(path.read_text() for path in paths)
    if "set_block_num(24)" not in text or "set_block_num(20)" in text:
        raise ValueError("Default shared Q_B no longer dispatches 24 workers")
    report = {"status": "COMPILE_PASS", "mode": int(mode), "default_workers": 24,
              "source": str(SOURCE), "orchestration": [str(p) for p in paths],
              "scope": "CPU shared-helper default call, not precision model or device validation"}
    (ROOT / f"default_mode{mode}.json").write_text(json.dumps(report, indent=2) + "\n")
    print("DEFAULT_COMPILE_PASS", mode)


if __name__ == "__main__":
    main()
