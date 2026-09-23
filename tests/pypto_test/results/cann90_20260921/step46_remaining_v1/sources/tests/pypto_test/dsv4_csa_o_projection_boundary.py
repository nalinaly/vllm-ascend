"""Isolate saved O-projection errors by passing Native WO-A through identity weights.

All device arithmetic uses diagnose_o_projection and the production function.
Identity weights expose its existing quantizer without copying that calculation.
This is a diagnostic input substitution, not a production change or acceptance.
"""

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
    args = parser.parse_args()
    repo = activate()
    report = {"status": "FAIL", "scope": "saved O-projection boundary only"}
    try:
        import torch
        import torch_npu
        import pypto.torch
        from dsv4_csa_full_compare import compare_tensor
        from dsv4_csa_precision_kernels import O_PROJ_T_PAD, diagnose_o_projection

        torch.set_num_threads(4)
        torch.npu.set_device(args.device)
        device = torch.device(f"npu:{args.device}")
        pypto.torch.init(device=args.device, platform="a2a3", runtime="tensormap_and_ringbuffer")
        data = torch.load(args.capture, map_location="cpu", weights_only=False)
        rows = data["native.output"].shape[0]
        packed = torch.zeros((8, O_PROJ_T_PAD, 4096), dtype=torch.bfloat16, device=device)
        packed[:, :rows] = data["native.projection_input"].reshape(rows, 8, 4096).transpose(0, 1).to(device)
        wa, wb, ws = (data[key].to(device) for key in ("wo_a", "wo_b", "wo_b_scale"))
        output = torch.empty((rows, 4096), dtype=torch.bfloat16, device=device)
        dump = {}
        with torch.inference_mode():
            diagnose_o_projection(packed.view(-1, 4096), wa, wb, ws, output)
            torch.npu.synchronize()
            dump["production_output"] = output.cpu()
            report["reproduces_saved_production"] = compare_tensor(
                dump["production_output"], data["pto.given_native_projection_input"], 0, 0)
            assert report["reproduces_saved_production"]["status"] == "PASS", report
            report["production_vs_native"] = compare_tensor(dump["production_output"], data["native.output"], 0, 0)

            # Identity WO-A makes each group output precisely its captured BF16
            # Native activation; original 240-row shape and quantizer are kept.
            packed.zero_()
            packed[:, :rows, :1024] = data["native.wo_a"].view(rows, 8, 1024).transpose(0, 1).to(device)
            identity_wa = torch.zeros_like(wa)
            identity_wa[:, :, :1024] = torch.eye(1024, dtype=torch.bfloat16, device=device)
            diagnose_o_projection(packed.view(-1, 4096), identity_wa, wb, ws, output)
            torch.npu.synchronize()
            dump["given_native_wo_a"] = output.cpu()
            report["given_native_wo_a"] = compare_tensor(dump["given_native_wo_a"], data["native.output"], 0, 0)
            quantized, scale = torch_npu.npu_dynamic_quant(data["native.wo_a"].to(device))
            report["native_quant_reproduces_capture"] = compare_tensor(quantized.cpu(), data["native.wo_b_quantized"], 0, 0)
            report["native_scale_reproduces_capture"] = compare_tensor(scale.cpu(), data["native.wo_b_scale"], 0, 0)
            assert report["native_quant_reproduces_capture"]["status"] == "PASS", report
            assert report["native_scale_reproduces_capture"]["status"] == "PASS", report

            # Selector WO-B exposes q * token_scale, rounded to BF16. Dividing
            # by the known positive scale recovers INT8 exactly: the BF16 error
            # is at most 0.25 quantization steps for |q| <= 127. Verify the
            # reconstruction against every returned BF16 element as well.
            recovered = []
            for half in range(2):
                selector = torch.zeros_like(wb)
                channels = torch.arange(4096, device=device)
                selector[channels, channels + half * 4096] = 1
                diagnose_o_projection(packed.view(-1, 4096), identity_wa, selector, torch.ones_like(ws), output)
                torch.npu.synchronize()
                observed = output.cpu()
                integers = torch.round(observed.float() / data["native.wo_b_scale"][:, None]).to(torch.int8)
                reconstructed = (integers.float() * data["native.wo_b_scale"][:, None]).bfloat16()
                assert torch.equal(reconstructed, observed), "selector quantization recovery is not exact"
                recovered.append(integers)
            pto_q = torch.cat(recovered, dim=1)
            dump["pto_quant_given_native_wo_a"] = pto_q
            report["quant_given_native_wo_a"] = compare_tensor(pto_q, data["native.wo_b_quantized"], 0, 0)
            report["quant_coordinates"] = (pto_q != data["native.wo_b_quantized"]).nonzero().tolist()

        # FP64 exactly accumulates these bounded integer products (< 2**53).
        native_acc = (data["native.wo_b_quantized"].double() @ data["wo_b"].T.double()).float()
        pto_acc = (pto_q.double() @ data["wo_b"].T.double()).float()
        token_scale = data["native.wo_b_scale"][:, None]
        weight_scale = data["wo_b_scale"]
        report["dequant_orders"] = {}
        for name, acc in (("native_quant", native_acc), ("pto_quant", pto_acc)):
            variants = {"activation_then_weight": (acc * token_scale) * weight_scale,
                        "weight_then_activation": (acc * weight_scale) * token_scale,
                        "merged_scales": acc * (token_scale * weight_scale)}
            report["dequant_orders"][name] = {label: compare_tensor(value.bfloat16(), data["native.output"], 0, 0)
                                              for label, value in variants.items()}
            if name == "pto_quant":
                report["reconstructs_identity_wo_a_device_output"] = compare_tensor(
                    variants["activation_then_weight"].bfloat16(), dump["given_native_wo_a"], 0, 0)
                assert report["reconstructs_identity_wo_a_device_output"]["status"] == "PASS", report
        args.output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(dump, args.output_dir / "o_projection_boundary.pt")
        paths = [Path(__file__).resolve(), repo / "tests/pypto_test/dsv4_csa_precision_kernels.py",
                 repo / "vllm_ascend/ops/pypto/deepseek_v4_flash_dspark/decode_o_proj.py"]
        report["sources"] = {str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        report["status"] = "DIAGNOSTIC_COMPLETE"
        print(report, flush=True)
    except BaseException:
        report["error"] = traceback.format_exc()
        raise
    finally:
        write_json(args.output_dir / "o_projection_boundary.json", report)


if __name__ == "__main__":
    main()
