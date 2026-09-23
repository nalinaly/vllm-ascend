"""Rebuild the complete failed B40 SWA window from its original producer inputs.

Uses production KV and Native F.linear/RMS/RoPE. No CSA trajectory is replaced:
this small replay isolates the arithmetic that produced the 128 selected rows.
"""
import argparse
import hashlib
import json
import traceback
from pathlib import Path

from dsv4_csa_env import activate, load_native_extension, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline", action="store_true")
    args = parser.parse_args()
    repo = activate()
    report = {"status": "FAIL", "scope": "B40 step46/query185 complete SWA producer replay only"}
    try:
        import pypto.torch
        import torch
        import torch_npu
        from dsv4_csa_continuous import accepted_counts
        from dsv4_csa_full_compare import compare_tensor
        from dsv4_csa_precision_kernels import diagnose_kv_identity_rope
        from vllm_ascend.ops.rope_dsv4 import ComplexExpRotaryEmbedding

        torch.set_num_threads(2)
        torch.npu.set_device(args.device)
        device = torch.device(f"npu:{args.device}")
        load_native_extension(repo)
        pypto.torch.init(device=args.device, platform="a2a3", runtime="tensormap_and_ringbuffer")
        captured = torch.load(args.capture, map_location="cpu", weights_only=True)
        weights = torch.load(args.weights, map_location="cpu", weights_only=True)
        row, batch, last_step, seed = 185, 40, 46, 1024
        failure = torch.load(args.capture.with_name("failure.pt"), map_location="cpu", weights_only=True)
        last_position = int(failure["positions"][row])
        starts = torch.full((batch,), 131071, dtype=torch.int32)
        writes, step_starts = {}, {}
        for step in range(last_step + 1):
            step_starts[step] = starts.clone()
            for token in range(6):
                writes[int(starts[row // 6]) + token] = (step, row // 6 * 6 + token)
            starts += accepted_counts(torch, "mixed", step, batch)
        origins = [writes[pos] for pos in range(last_position - 127, last_position + 1)]
        assert all(step <= last_step for step, _ in origins)
        original_weight = weights["weight"].to(device)
        pto_weight = original_weight.T.contiguous()
        gamma = weights["gamma"].to(device)
        pto_window = torch.empty((128, 512), dtype=torch.bfloat16)
        native_window = torch.empty_like(pto_window)
        with torch.device(device):
            inv_freq = ComplexExpRotaryEmbedding.precompute_freqs_cis(64, 65536, 65536, 160000, 16, 32, 1)
        with torch.inference_mode():
            for step in sorted({origin[0] for origin in origins}):
                hidden = torch.randn((240, 4096), generator=torch.Generator().manual_seed(seed + step),
                                     dtype=torch.bfloat16).to(device)
                positions = (step_starts[step][:, None] + torch.arange(6)).flatten().to(device)
                freqs = torch.einsum("i,j->ij", positions.float(), inv_freq)
                cosine = freqs.cos().repeat_interleave(2, dim=-1)
                sine = freqs.sin().repeat_interleave(2, dim=-1)
                if step == last_step:
                    for name, value in (("cos", cosine), ("sin", sine)):
                        saved = captured[name][row]
                        if captured.get("rope_layout") != "native_interleaved_fp32":
                            assert torch.equal(saved[:32], saved[32:])
                            saved = saved[:32].repeat_interleave(2)
                        assert torch.equal(value[row].cpu(), saved), name
                native_projection = torch.nn.functional.linear(hidden, original_weight)
                native_norm, _ = torch_npu.npu_rms_norm(native_projection, gamma, 1e-6)
                native_kv = native_norm.view(240, 1, 1, 512)
                torch.ops._C_ascend.inplace_partial_rotary_mul(
                    native_kv, cosine.view(240, 1, 1, 64), sine.view(240, 1, 1, 64),
                    rotary_mode="interleave", partial_slice=[448, 512])
                pto_kv = torch.empty((240, 512), dtype=torch.bfloat16, device=device)
                diagnose_kv_identity_rope(hidden, pto_weight, gamma, cosine, sine, pto_kv)
                torch.npu.synchronize()
                native_cpu, pto_cpu = native_kv.view(240, 512).cpu(), pto_kv.cpu()
                for index, (producer, token) in enumerate(origins):
                    if producer == step:
                        native_window[index] = native_cpu[token]
                        pto_window[index] = pto_cpu[token]
        report["native_reproduces_saved"] = compare_tensor(native_window, captured["native.selected_kv"][row, :128], 0, 0)
        report["pto_reproduces_saved"] = compare_tensor(pto_window, captured["pto.selected_kv"][row, :128], 0, 0)
        report["pto_vs_native"] = compare_tensor(pto_window, native_window, 0, 0)
        assert report["native_reproduces_saved"]["status"] == "PASS", report
        if args.baseline:
            assert report["pto_reproduces_saved"]["status"] == "PASS", report
        args.output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(dict(native=native_window, pto=pto_window, origins=origins), args.output_dir / "swa_window.pt")
        minimal = {}
        for key in ("native.query", "pto.query", "native.selected_kv", "pto.selected_kv", "selected_valid",
                    "cos", "sin", "native.output", "native.projection_input", "native.attention_output",
                    "pto.pto_query_pto_cache.projection_input", "pto.native_query_native_cache.projection_input"):
            minimal[key] = captured[key][row:row+1].clone()
        for key in ("wo_a", "wo_b", "wo_b_scale", "sink"):
            minimal[key] = captured[key]
        if "rope_layout" in captured:
            minimal["rope_layout"] = captured["rope_layout"]
        minimal["pto.selected_kv"][0, :128] = pto_window
        torch.save(minimal, args.output_dir / "attention_capture.pt")
        report["sources"] = {str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in (Path(__file__).resolve(), repo / "tests/pypto_test/dsv4_csa_precision_kernels.py",
                                       repo / "vllm_ascend/ops/pypto/deepseek_v4_flash_dspark/qkv_proj_rope.py")}
        report["producer_steps"] = sorted({origin[0] for origin in origins})
        report["status"] = "PASS"
        print(json.dumps(report), flush=True)
    except BaseException:
        report["error"] = traceback.format_exc()
        raise
    finally:
        write_json(args.output_dir / "swa_window.json", report)


if __name__ == "__main__":
    main()
