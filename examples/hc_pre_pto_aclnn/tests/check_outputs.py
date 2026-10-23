# SPDX-License-Identifier: Apache-2.0
"""Generate deterministic T96 inputs or check ACLNN outputs against CPU FP32."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch


def reference(x, w, scale, base):
    x = x.float()
    tokens = x.shape[0]
    flat = x.reshape(tokens, 16384)
    proj = (flat @ w.T) * torch.rsqrt(flat.square().mean(1, keepdim=True) + 1e-6)
    pre = torch.sigmoid(proj[:, :4] * scale[0] + base[:4]) + 1e-6
    post = 2 * torch.sigmoid(proj[:, 4:8] * scale[1] + base[4:8])
    mixed = (x * pre[:, :, None]).sum(1).to(torch.bfloat16).float()
    comb = torch.softmax((proj[:, 8:] * scale[2] + base[8:]).reshape(tokens, 4, 4), -1) + 1e-6
    comb = comb / (comb.sum(-2, keepdim=True) + 1e-6)
    for _ in range(19):
        comb = comb / (comb.sum(-1, keepdim=True) + 1e-6)
        comb = comb / (comb.sum(-2, keepdim=True) + 1e-6)
    return mixed, post, comb


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.generate:
        args.input.mkdir(parents=True, exist_ok=True)
        if any(args.input.glob("*.bin")):
            raise RuntimeError("Use an empty input directory")
        generator = torch.Generator().manual_seed(20262615)
        x = torch.randn((96, 4, 4096), generator=generator, dtype=torch.bfloat16)
        w = torch.randn((24, 16384), generator=generator) * 0.01
        scale = torch.tensor([1.0, 1.0, 1.0])
        base = torch.randn((24,), generator=generator) * 0.1
        x.view(torch.uint16).numpy().tofile(args.input / "x.bin")
        for name, tensor in [("weight", w), ("scale", scale), ("base", base)]:
            tensor.numpy().tofile(args.input / f"{name}.bin")
    if args.output is None:
        return
    def read(directory, name, bf16=False):
        a = np.fromfile(directory / f"{name}.bin", dtype=np.uint16 if bf16 else np.float32)
        t = torch.from_numpy(a)
        return t.view(torch.bfloat16).float() if bf16 else t
    x = read(args.input, "x", True).reshape(96, 4, 4096)
    w = read(args.input, "weight").reshape(24, 16384)
    scale, base = read(args.input, "scale"), read(args.input, "base")
    mixed, post, comb = reference(x, w, scale, base)
    report = {}
    for name, expected, bf16 in [("mixed", mixed, True), ("post", post, False), ("comb", comb, False)]:
        actual = read(args.output, name, bf16).reshape(expected.shape)
        atol, rtol = (0.01, 0.01) if bf16 else (1e-5, 1e-4)
        report[name] = {"passed": bool(torch.allclose(actual, expected, atol=atol, rtol=rtol)),
                        "max_abs": float((actual - expected).abs().max()), "atol": atol, "rtol": rtol}
    (args.output / "accuracy.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not all(v["passed"] for v in report.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
