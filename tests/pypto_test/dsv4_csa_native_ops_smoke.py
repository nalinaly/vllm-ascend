"""Execute native CSA metadata operators; availability smoke, not a P1 acceptance test."""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

from dsv4_csa_env import activate, load_native_extension, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--op", choices=("sas", "qli", "compressor"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    repo = activate()
    report = {
        "case_id": f"P0_NATIVE_{args.op.upper()}_METADATA_EXECUTION",
        "status": "FAIL",
        "scope": "native_operator_execution_only_not_full_builder_or_numeric_validation",
        "python": sys.executable,
        "device": args.device,
    }
    try:
        import torch
        import torch_npu  # noqa: F401

        torch.npu.set_device(args.device)
        report["extension"] = str(load_native_extension(repo))
        device = torch.device(f"npu:{args.device}")
        batch, query, ratio, page = 4, 6, 4, 32
        starts = torch.tensor([131071, 131072, 131073, 131078], dtype=torch.int32, device=device)
        bounds = torch.arange(batch + 1, dtype=torch.int32, device=device) * query
        seq_lens = starts + query
        if args.op == "sas":
            op = torch.ops._C_ascend.npu_sparse_attn_sharedkv_metadata
            result = op(
                num_heads_q=64,
                num_heads_kv=1,
                head_dim=512,
                cu_seqlens_q=bounds,
                seqused_kv=seq_lens,
                batch_size=batch,
                max_seqlen_q=query,
                max_seqlen_kv=131084,
                cmp_topk=512,
                cmp_ratio=ratio,
                ori_mask_mode=4,
                cmp_mask_mode=3,
                ori_win_left=127,
                ori_win_right=0,
                layout_q="TND",
                layout_kv="PA_ND",
                has_ori_kv=True,
                has_cmp_kv=True,
                device=str(device),
            )
        elif args.op == "qli":
            op = torch.ops._C_ascend.npu_vllm_quant_lightning_indexer_metadata
            result = op(
                actual_seq_lengths_query=bounds[1:],
                actual_seq_lengths_key=seq_lens,
                num_heads_q=64,
                num_heads_k=1,
                head_dim=128,
                query_quant_mode=0,
                key_quant_mode=0,
                batch_size=batch,
                max_seqlen_q=query,
                max_seqlen_k=131084,
                layout_query="TND",
                layout_key="PA_BSND",
                sparse_count=512,
                sparse_mode=3,
                pre_tokens=(1 << 63) - 1,
                next_tokens=(1 << 63) - 1,
                cmp_ratio=ratio,
            )
        else:
            op = torch.ops._C_ascend.compressor_metadata
            # Deterministic input to check execution at the target history length.
            rope_cos = torch.ones((131100, 64), dtype=torch.float32, device=device)
            rope_sin = torch.zeros_like(rope_cos)
            table = torch.arange(batch * 1040, dtype=torch.int32, device=device).reshape(batch, 1040)
            result = op(rope_cos, rope_sin, bounds, starts, table, page, 2, ratio, 10, batch)
        torch.npu.synchronize()
        outputs = result if isinstance(result, tuple) else (result,)
        report.update(
            status="PASS",
            schema=str(op.default._schema),
            outputs=[{"shape": list(t.shape), "dtype": str(t.dtype), "device": str(t.device)} for t in outputs],
        )
        if args.op == "compressor":
            report["slots"] = result[2].cpu().tolist()
    except Exception:
        report["error"] = traceback.format_exc()
        raise
    finally:
        path = args.output_dir / f"native_{args.op}_metadata_device{args.device}.json"
        write_json(path, report)
        print(f"{report['case_id']}: {report['status']} ({path})", flush=True)


if __name__ == "__main__":
    main()
