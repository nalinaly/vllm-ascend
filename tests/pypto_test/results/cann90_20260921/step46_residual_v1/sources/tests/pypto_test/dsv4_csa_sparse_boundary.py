"""Isolate a captured query's sparse attention using the production functions.

Only selected KV rows are remapped. The remapped result must reproduce the
captured production result exactly before any new diagnostic is accepted.
"""

import argparse
import hashlib
import traceback
from pathlib import Path

from dsv4_csa_env import activate, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--row", type=int, required=True)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate", action="store_true", help="Validate a changed kernel against captured Native")
    parser.add_argument("--input-source", choices=("native", "pto"), default="native")
    args = parser.parse_args()
    repo = activate()
    report = {"status": "FAIL", "capture": str(args.capture), "row": args.row,
              "scope": "captured sparse-attention arithmetic only; not a continuous-trajectory acceptance"}
    try:
        import pypto.torch
        import torch
        import torch_npu  # noqa: F401
        from dsv4_csa_full_compare import compare_tensor
        from dsv4_csa_precision_kernels import (
            O_PROJ_T_PAD, diagnose_sparse_accumulators, diagnose_sparse_attention,
            diagnose_sparse_qk,
            diagnose_o_projection,
        )

        torch.set_num_threads(2)
        torch.npu.set_device(args.device)
        device = torch.device(f"npu:{args.device}")
        pypto.torch.init(device=args.device, platform="a2a3", runtime="tensormap_and_ringbuffer")
        captured = torch.load(args.capture, map_location="cpu", weights_only=False)
        row, tokens = args.row, 24
        assert captured["selected_valid"][row].all(), "This isolation requires all 128+512 rows to be valid"
        query = captured[f"{args.input_source}.query"][row].expand(tokens, -1, -1).contiguous().to(device)
        selected = captured[f"{args.input_source}.selected_kv"][row]
        original = selected[:128].reshape(4, 32, 1, 512).contiguous().to(device)
        compressed = selected[128:].reshape(16, 32, 1, 512).contiguous().to(device)
        # Position 131071 gives an aligned 128-row window and permits all 512
        # remapped compressed indices. RoPE is supplied from the capture.
        original_table = torch.zeros((4, 4096), dtype=torch.int32, device=device)
        original_table[:, -4:] = torch.arange(4, dtype=torch.int32, device=device)
        compressed_table = torch.arange(16, dtype=torch.int32, device=device).expand(4, -1).contiguous()
        indices = torch.arange(512, dtype=torch.int32, device=device).expand(tokens, -1).contiguous()
        positions = torch.full((tokens, 1), 131071, dtype=torch.int64, device=device)
        frequencies = []
        for name in ("cos", "sin"):
            value = captured[name][row]
            if captured.get("rope_layout") != "native_interleaved_fp32":
                assert torch.equal(value[:32], value[32:]), "Unrecognized historical RoPE layout"
                value = value[:32].repeat_interleave(2)
            frequencies.append(value.expand(tokens, -1).contiguous().to(device))
        inputs = (query, original, original_table, compressed, compressed_table,
                  indices, positions, captured["sink"].to(device), *frequencies)
        packed = torch.empty((8 * O_PROJ_T_PAD, 4096), dtype=torch.bfloat16, device=device)
        with torch.inference_mode():
            diagnose_sparse_attention(*inputs, packed)
            torch.npu.synchronize()
            actual = packed.view(8, O_PROJ_T_PAD, 4096)[:, :tokens].transpose(0, 1).reshape(tokens, 64, 512).cpu()
            expected = captured[f"pto.{args.input_source}_query_{args.input_source}_cache.projection_input"][row].expand(tokens, -1, -1)
            report["reproduces_saved_production"] = compare_tensor(actual, expected, 0, 0)
            if not args.candidate:
                assert report["reproduces_saved_production"]["status"] == "PASS", report
            report["input_source"] = args.input_source
            report["native_projection_input_exact"] = compare_tensor(actual[0], captured["native.projection_input"][row], 0, 0)
            if args.candidate:
                output = torch.empty((tokens, 4096), dtype=torch.bfloat16, device=device)
                diagnose_o_projection(packed, captured["wo_a"].to(device), captured["wo_b"].to(device),
                                      captured["wo_b_scale"].to(device), output)
                torch.npu.synchronize()
                report["candidate_output"] = compare_tensor(output.cpu(), captured["native.output"][row].expand(tokens, -1), 1e-2, 1e-2)
            maximum = torch.empty((tokens * 64, 1), dtype=torch.float32, device=device)
            denominator = torch.empty_like(maximum)
            numerator = torch.empty((tokens * 64, 512), dtype=torch.float32, device=device)
            diagnose_sparse_accumulators(*inputs, maximum, denominator, numerator)
            scores = torch.empty((64, 640), dtype=torch.float32, device=device)
            diagnose_sparse_qk(query[0].contiguous(), selected.to(device), scores)
            torch.npu.synchronize()
        stats = {"maximum": maximum.cpu().view(tokens, 64, 1),
                 "denominator": denominator.cpu().view(tokens, 64, 1),
                 "numerator": numerator.cpu().view(tokens, 64, 512)}
        for name, tensor in stats.items():
            assert torch.equal(tensor, tensor[0:1].expand_as(tensor)), name
        args.output_dir.mkdir(parents=True, exist_ok=True)
        torch.save({**{name: tensor[0] for name, tensor in stats.items()},
                    "projection_input": actual[0], "native_projection_input": captured["native.projection_input"][row],
                    "native_attention_output": captured["native.attention_output"][row],
                    "query": captured[f"{args.input_source}.query"][row], "selected_kv": selected,
                    "sink": captured["sink"], "scores": scores.cpu()}, args.output_dir / "sparse_boundary.pt")
        report["sources"] = {str(path.relative_to(repo)): hashlib.sha256(path.read_bytes()).hexdigest()
                             for path in [Path(__file__).resolve(), repo / "tests/pypto_test/dsv4_csa_precision_kernels.py",
                                          repo / "vllm_ascend/ops/pypto/deepseek_v4_flash_dspark/decode_sparse_attn_csa.py"]}
        report["status"] = report.get("candidate_output", {"status": "PASS"})["status"]
        print(report, flush=True)
        assert report["status"] == "PASS", report
    except BaseException:
        report["error"] = traceback.format_exc()
        raise
    finally:
        write_json(args.output_dir / "sparse_boundary.json", report)


if __name__ == "__main__":
    main()
