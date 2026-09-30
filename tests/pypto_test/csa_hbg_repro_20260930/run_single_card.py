"""Run the existing real-weight CSA single-card case with HBG only overridden."""
import os
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TESTS = HERE.parents[1]
sys.path.insert(0, str(TESTS))
from dsv4_csa_env import activate

activate()
import vllm_ascend.ops.pypto as ops
ops.__path__ = [str(HERE / "frozen_ops_pypto")]
import pypto.torch

original_init = pypto.torch.init

def init_hbg(**kwargs):
    kwargs["runtime"] = "host_build_graph"
    print(f"CSA HBG repro: pypto.torch.init({kwargs!r})", flush=True)
    return original_init(**kwargs)

pypto.torch.init = init_hbg
sys.argv = [str(TESTS / "dsv4_csa_single_layer.py"),
            "--checkpoint", "/data/model/DeepSeek-V4-Flash-0731-w8a8",
            "--output", str(HERE / "device_run"),
            "--device", "0", "--batch", "4", "--history", "8192",
            "--layer-index", "2", "--variant", "performance",
            "--weight-nz-mode", "2", "--atomic-add", "0", "--timing-iters", "0"]
runpy.run_path(sys.argv[0], run_name="__main__")
