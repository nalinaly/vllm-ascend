"""显式解析HCA依赖图并编译设备代码；无需分配NPU，不以注册代替编译。"""

import argparse
import importlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("side")
    parser.add_argument("--operator-source", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT)
    parser.add_argument("--nz-mode", type=int, choices=(0, 1, 2), default=2)
    args = parser.parse_args()
    if args.operator_source is not None:
        source = args.operator_source.resolve()
    else:
        if args.side not in ("base", "reuse"):
            parser.error("未指定operator-source时side必须为base或reuse")
        manifest = json.loads((ROOT / "source.json").read_text())
        source = Path(manifest["baseline" if args.side == "base" else "candidate"])
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.environ.update(
        VLLM_ASCEND_ENABLE_NZ=str(args.nz_mode), VLLM_ASCEND_PTO_CSA_ATOMIC_ADD="0", PTO_CSA_VARIANT="performance",
    )
    sys.path.insert(0, str(ROOT.parent))
    from dsv4_csa_env import activate

    activate()
    import vllm_ascend.ops.pypto as package

    package.__path__ = [str(source)]
    from pypto.runtime import RunConfig

    module = importlib.import_module("vllm_ascend.ops.pypto.deepseek_v4_flash_hca.decode_hca")
    graphs = {
        name: type(getattr(module, name)._get_dep_graph()).__name__
        for name in ("decode_hca_tp1_layer", "decode_hca_tp1_layer_test")
    }
    compiled = module.decode_hca_tp1_layer_test.compile(
        config=RunConfig(
            platform="a2a3",
            save_kernels=True,
            dump_passes=True,
            save_kernels_dir=str(output / "compiled" / args.side),
        )
    )
    compiled.load()
    report = {
        "status": "CPU_COMPILE_PASS",
        "source": str(source),
        "graphs": graphs,
        "cann": os.environ["ASCEND_HOME_PATH"],
        "device_execution": False,
        "nz_mode": args.nz_mode,
    }
    (output / f"compile_{args.side}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
