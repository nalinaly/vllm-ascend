# SPDX-License-Identifier: Apache-2.0
"""复用单层驱动，检查 HBG 与 ring 的完整输出及状态是否逐字节一致。"""

import argparse
import json
import runpy
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operator", choices=("csa", "hca"), required=True)
    parser.add_argument("--runtime", choices=("host_build_graph", "tensormap_and_ringbuffer"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--history", type=int, required=True)
    parser.add_argument("--device", type=int, choices=(0,), default=0)
    args = parser.parse_args()
    tests = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(tests))
    from dsv4_csa_env import activate, write_json

    activate()
    import vllm_ascend.ops.pypto as ops

    if not (args.source / "deepseek_v4_flash_hca/decode_hca.py").is_file():
        raise ValueError("排队前必须冻结完整算子源码")
    ops.__path__ = [str(args.source.resolve())]
    script = tests / f"dsv4_{args.operator}_single_layer.py"
    sys.argv = [
        str(script), "--output", str(args.output), "--batch", "4", "--history", str(args.history),
        "--device", "0", "--runtime", args.runtime, "--weight-nz-mode", "2", "--deterministic-level", "1",
        "--checkpoint", "/data/model/DeepSeek-V4-Flash-0731-w8a8",
    ]
    if args.operator == "csa":
        sys.argv += ["--layer-index", "2", "--variant", "performance", "--atomic-add", "0", "--save-state", "--graph"]
    else:
        sys.argv += ["--service-graph", "--operator-source", str(args.source.resolve())]
    runpy.run_path(str(script), run_name="__main__")
    if args.reference is None:
        return

    import torch
    from dsv4_csa_validation import compare_tensor

    current_report = json.loads((args.output / "report.json").read_text())
    reference_report = json.loads(args.reference.with_name("report.json").read_text())
    for key in ("checkpoint", "seed", "batch", "history", "weight_nz_mode", "deterministic_level"):
        if current_report.get(key) != reference_report.get(key):
            raise ValueError(f"参考配置不匹配：{key}")
    actual = torch.load(args.output / "states.pt", map_location="cpu", weights_only=False)["pto"]
    reference = torch.load(args.reference, map_location="cpu", weights_only=False)["pto"]
    expected_count = 8 if args.operator == "csa" else 4
    if actual.keys() != reference.keys() or len(actual) != expected_count:
        raise ValueError("参考快照缺少必要输出或状态")
    checks = {}
    for name, value in actual.items():
        check = compare_tensor(value, reference[name], 0, 0)
        raw, expected = value.contiguous().view(torch.uint8), reference[name].contiguous().view(torch.uint8)
        check["different_bytes"] = int((raw != expected).sum())
        if check["different_bytes"]:
            check["status"] = "FAIL"
        checks[name] = check
    passed = all(value["status"] == "PASS" for value in checks.values())
    write_json(args.output / "hbg_vs_ring.json", {
        "status": "PASS" if passed else "FAIL", "reference": str(args.reference.resolve()), "checks": checks,
    })
    if not passed:
        raise AssertionError("HBG 与 ring 的输出或完整状态不一致")
    print(f"HBG_RING_EXACT_PASS {args.operator} history={args.history} tensors={len(checks)}", flush=True)


if __name__ == "__main__":
    main()
