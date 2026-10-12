# SPDX-License-Identifier: Apache-2.0
"""Run a generated HC_pre T=10 buildout directly through Simpler.

Example:
    python tests/pypto_test/run_hc_pre_buildout.py \
        --build-dir /path/to/hc_pre_buildout -p a2a3 -d 0

The buildout must contain kernel_config.py, orchestration/, and kernels/.
This is a compile-and-run smoke test with synthetic inputs; it does not
compare the outputs with a reference implementation.
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import torch
from simpler_setup import SceneTestCase, TaskArgsBuilder, TensorArg, scene_test

TOKENS = 10
HEADS = 4
HIDDEN_SIZE = 4096


@scene_test(level=2, runtime="tensormap_and_ringbuffer")
class HcPreT10(SceneTestCase):
    CASES = [{"name": "hc_pre_t10", "platforms": ["a2a3"], "params": {}}]
    SKIP_GOLDEN = True

    def generate_args(self, _params):
        generator = torch.Generator().manual_seed(0)
        return TaskArgsBuilder(
            TensorArg("x", torch.randn(TOKENS, HEADS, HIDDEN_SIZE, generator=generator, dtype=torch.bfloat16)),
            TensorArg("hc_fn", torch.randn(24, HEADS * HIDDEN_SIZE, generator=generator) * 0.001),
            TensorArg("hc_scale", torch.ones(3, dtype=torch.float32)),
            TensorArg("hc_base", torch.zeros(24, dtype=torch.float32)),
            TensorArg("mixed", torch.zeros(TOKENS, HIDDEN_SIZE, dtype=torch.bfloat16)),
            TensorArg("post", torch.zeros(TOKENS, HEADS, dtype=torch.float32)),
            TensorArg("comb_out", torch.zeros(TOKENS, HEADS, HEADS, dtype=torch.float32)),
        )


def load_callable(build_dir: Path) -> dict:
    config_path = build_dir / "kernel_config.py"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing generated config: {config_path}")
    spec = importlib.util.spec_from_file_location("hc_pre_kernel_config", config_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import generated config: {config_path}")
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    incores = config.KERNELS
    names = {entry["name"] for entry in incores}
    if not {"hc_small_aic", "hc_small_aiv"} <= names:
        raise ValueError("Expected generated HC_pre small AIC/AIV kernels")
    orchestration = config.ORCHESTRATION
    for entry in [orchestration, *incores]:
        if not Path(entry["source"]).is_file():
            raise FileNotFoundError(f"Missing generated C++: {entry['source']}")
    return {"orchestration": orchestration, "incores": incores}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--build-dir", type=Path, required=True)
    options, simpler_args = parser.parse_known_args()
    HcPreT10.CALLABLE = load_callable(options.build_dir.resolve())
    sys.argv = [sys.argv[0], *simpler_args]
    SceneTestCase.run_module(__name__)
