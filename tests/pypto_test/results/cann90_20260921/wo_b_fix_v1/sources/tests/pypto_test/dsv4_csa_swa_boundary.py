"""Isolate captured SWA KV errors; identity RoPE permits a NoPE comparison."""

import argparse
import hashlib
import traceback
from pathlib import Path

from dsv4_csa_env import activate, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--kv-splits", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    repo = activate()
    report = {"status": "FAIL", "scope": "SWA projection/RMS diagnostic, no trajectory acceptance"}
    try:
        import pypto.torch
        import torch
        import torch_npu
        from dsv4_csa_full_compare import compare_tensor
        from importlib import import_module
        kv_module = import_module("vllm_ascend.ops.pypto.deepseek_v4_flash_dspark.qkv_proj_rope")
        # Process-local diagnostic specialization; no production source edit.
        kv_module.KV_OK = args.kv_splits
        kv_module.KV_SPLIT_K_TILE = kv_module.D // args.kv_splits
        from dsv4_csa_precision_kernels import diagnose_kv_identity_rope
        report["diagnostic_kv_splits"] = args.kv_splits

        torch.set_num_threads(2)
        torch.npu.set_device(args.device)
        device = torch.device(f"npu:{args.device}")
        pypto.torch.init(device=args.device, platform="a2a3", runtime="tensormap_and_ringbuffer")
        data = torch.load(args.capture, map_location="cpu", weights_only=True)
        # Repeat to 32 rows: the failed request used the production full RMS tile.
        x = data["x"].repeat(4, 1).to(device)
        weight = data["weight"].T.contiguous().to(device)
        gamma = data["gamma"].to(device)
        cos = torch.ones((32, 64), dtype=torch.float32, device=device)
        sin = torch.zeros_like(cos)
        pto = torch.empty((32, 512), dtype=torch.bfloat16, device=device)
        pto_given_projection = torch.empty_like(pto)
        with torch.inference_mode():
            native_rows = []
            original_weight = data["weight"].to(device)
            for origin in data["origins"]:
                hidden = torch.randn((240, 4096), generator=torch.Generator().manual_seed(1024 + origin["step"]),
                                     dtype=torch.bfloat16).to(device)
                # Same F.linear, original weight strides and batch as Native.
                projected = torch.nn.functional.linear(hidden, original_weight)
                native_rows.append(projected[origin["input_row"]].clone())
            native_projection = torch.stack(native_rows).repeat(4, 1)
            native_norm, native_rstd = torch_npu.npu_rms_norm(native_projection, gamma, 1e-6)
            diagnose_kv_identity_rope(x, weight, gamma, cos, sin, pto)
            # Feed the Native BF16 projection through an exact identity matmul
            # to isolate the *same production* RMS code, without copying it.
            identity_x = torch.zeros_like(x)
            identity_x[:, :512] = native_projection
            identity_w = torch.zeros_like(weight)
            identity_w[:512, :] = torch.eye(512, dtype=torch.bfloat16, device=device)
            diagnose_kv_identity_rope(identity_x, identity_w, gamma, cos, sin, pto_given_projection)
            torch.npu.synchronize()
        tensors = {"native_projection": native_projection.cpu(), "native_norm": native_norm.cpu(),
                   "native_rstd": native_rstd.cpu(), "pto": pto.cpu(),
                   "pto_given_native_projection": pto_given_projection.cpu()}
        for tensor in tensors.values():
            assert torch.equal(tensor, tensor[:8].repeat(4, *([1] * (tensor.ndim - 1))))
        report["rows"] = []
        for row, origin in enumerate(data["origins"]):
            item = dict(origin)
            for label, value, expected in (
                ("native_reproduces_saved_nope", tensors["native_norm"], data["native_selected"]),
                ("pto_reproduces_saved_nope", tensors["pto"], data["pto_selected"]),
                ("pto_vs_native_nope", tensors["pto"], data["native_selected"]),
                ("given_native_projection_vs_native_nope", tensors["pto_given_native_projection"], tensors["native_norm"]),
            ):
                item[label] = compare_tensor(value[row, :448], expected[row, :448], 0, 0)
            report["rows"].append(item)
        report["sources"] = {str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in [Path(__file__).resolve(), repo / "tests/pypto_test/dsv4_csa_precision_kernels.py",
                                       repo / "vllm_ascend/ops/pypto/deepseek_v4_flash_dspark/qkv_proj_rope.py"]}
        args.output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(tensors, args.output_dir / "swa_boundary.pt")
        report["status"] = "PASS"
    except BaseException:
        report["error"] = traceback.format_exc()
        raise
    finally:
        write_json(args.output_dir / "swa_boundary.json", report)


if __name__ == "__main__":
    main()
