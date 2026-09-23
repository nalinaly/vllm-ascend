"""Replay the production QR normalization against captured Native QA on NPU."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from dsv4_csa_env import activate, write_json
from dsv4_csa_full_compare import compare_tensor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--include-projection", action="store_true")
    parser.add_argument("--observe-native", action="store_true")
    parser.add_argument("--observe-k-rotation", action="store_true")
    parser.add_argument("--qa-rotation-step", type=int, choices=(128, 256), default=256)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    activate()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    import pypto.torch
    import torch
    import torch_npu  # noqa: F401
    from dsv4_csa_precision_kernels import QPROJ_T_PAD, diagnose_qa, diagnose_qr_normalize, diagnose_qr_rms
    from safetensors import safe_open

    if (args.checkpoint / "quant_model_description.json").exists():
        raise ValueError("Captured reference QA cannot validate a target ModelSlim checkpoint")
    captured = torch.load(args.capture, map_location="cpu", weights_only=True)
    weight_name = "layers.2.attn.q_norm.weight"
    weight_map = json.loads((args.checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    with safe_open(args.checkpoint / weight_map[weight_name], framework="pt", device="cpu") as reader:
        gamma = reader.get_tensor(weight_name)
    device = torch.device(f"npu:{args.device}")
    torch.npu.set_device(device)
    native_observations = {}
    if args.observe_native:
        projection_name = "layers.2.attn.wq_a.weight"
        with safe_open(args.checkpoint / weight_map[projection_name], framework="pt", device="cpu") as reader:
            native_weight = reader.get_tensor(projection_name).to(device)
        native_hidden = captured["hidden"].to(device)
        with (
            torch.inference_mode(),
            torch_npu.profiler.profile(
                activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
                record_shapes=True,
                experimental_config=torch_npu.profiler._ExperimentalConfig(
                    profiler_level=torch_npu.profiler.ProfilerLevel.Level1, record_op_args=True, op_attr=True
                ),
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(args.output_dir / "native_profile")),
            ),
        ):
            native_values = {
                "linear": torch.nn.functional.linear(native_hidden, native_weight),
                "contiguous_b": native_hidden @ native_weight.T.contiguous(),
                "chunks_m48": torch.cat(
                    [torch.nn.functional.linear(chunk, native_weight) for chunk in native_hidden.split(48)]
                ),
                "fp32": torch.nn.functional.linear(native_hidden.float(), native_weight.float()),
            }
            torch.npu.synchronize()
        native_observations = {
            name: compare_tensor(value.bfloat16(), captured["qa_bf16"], 0, 0) for name, value in native_values.items()
        }
        torch.save({name: value.cpu() for name, value in native_values.items()}, args.output_dir / "native_qa.pt")
    pypto.torch.init(device=args.device, platform="a2a3", runtime="tensormap_and_ringbuffer")
    with torch.inference_mode():
        if args.observe_k_rotation:
            # Reorder both operands identically; reuse the production QA math.
            # Native M240 uses L1 K256 and L0 K128; observe both granularities.
            projection_name = "layers.2.attn.wq_a.weight"
            with safe_open(args.checkpoint / weight_map[projection_name], framework="pt", device="cpu") as reader:
                rotation_weight = reader.get_tensor(projection_name).T.contiguous().to(device)
            rotation_hidden = captured["hidden"].to(device)
            rotation_output = torch.empty((QPROJ_T_PAD, 1024), dtype=torch.float32, device=device)
            rotations = {}
            rotation_counts = {}
            for start in range(4096 // args.qa_rotation_step):
                diagnose_qa(
                    torch.roll(rotation_hidden, shifts=-start * args.qa_rotation_step, dims=1),
                    torch.roll(rotation_weight, shifts=-start * args.qa_rotation_step, dims=0),
                    rotation_output,
                )
                value = rotation_output[: rotation_hidden.shape[0]].cpu()
                rotations[start] = value
                mismatch = value.bfloat16() != captured["qa_bf16"]
                rotation_counts[start] = {
                    "k_start": start * args.qa_rotation_step,
                    "total": int(mismatch.sum()),
                    "native_n96_blocks": [int(mismatch[:, col : col + 96].sum()) for col in range(0, 1024, 96)],
                }
                print("QA K rotation", start, rotation_counts[start], flush=True)
            torch.save(rotations, args.output_dir / "qa_k_rotations.pt")
            write_json(args.output_dir / "qa_k_rotations.json", rotation_counts)
        qa = captured["qa_bf16"].float().to(device)
        qa_fp32 = None
        if args.include_projection:
            projection_name = "layers.2.attn.wq_a.weight"
            with safe_open(args.checkpoint / weight_map[projection_name], framework="pt", device="cpu") as reader:
                projection = reader.get_tensor(projection_name).T.contiguous().to(device)
            qa_fp32 = torch.empty((QPROJ_T_PAD, 1024), dtype=torch.float32, device=device)
            diagnose_qa(captured["hidden"].to(device), projection, qa_fp32)
            qa = qa_fp32[: qa.shape[0]]
        qr = torch.empty_like(captured["qr"], device=device)
        scale = torch.empty((qa.shape[0], 1), dtype=torch.float32, device=device)
        square_sum, inverse_rms, rms = (torch.empty_like(scale) for _ in range(3))
        diagnose_qr_normalize(qa, gamma.to(device), qr, scale)
        diagnose_qr_rms(qa, square_sum, inverse_rms, rms)
        torch.npu.synchronize()
        output = {
            "qr": qr.cpu(),
            "qr_scale": scale.cpu(),
            "qr_square_sum": square_sum.cpu(),
            "qr_inverse_rms": inverse_rms.cpu(),
            "qr_rms": rms.cpu(),
        }
        if qa_fp32 is not None:
            output["qa_fp32"] = qa.cpu()
        torch.save(output, args.output_dir / "pto_debug.pt")
        report = {
            "scope": "Captured reference QA diagnostic; production normalization reused, not full CSA acceptance",
            "capture": str(args.capture),
            "capture_sha256": hashlib.sha256(args.capture.read_bytes()).hexdigest(),
            "include_projection": args.include_projection,
            "native_observations": native_observations,
            "qr": compare_tensor(qr, captured["qr"], 0, 0),
            "qr_scale": compare_tensor(scale, captured["qr_scale"].view(-1, 1), 0, 0),
        }
        checked = ["qr", "qr_scale"]
        if args.include_projection:
            report["qa_bf16"] = compare_tensor(qa.bfloat16(), captured["qa_bf16"], 0, 0)
            checked.append("qa_bf16")
        report["status"] = "PASS" if all(report[name]["status"] == "PASS" for name in checked) else "FAIL"
        write_json(args.output_dir / "qr_boundary.json", report)
        print({name: report[name]["mismatches"] for name in checked}, flush=True)
        if report["status"] != "PASS":
            raise AssertionError("QR normalization differs from the captured Native boundary")


if __name__ == "__main__":
    main()
