"""Native BlockTable -> native DSA builder smoke, without full model weights."""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

from dsv4_csa_env import activate, load_native_extension, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    repo = activate()
    report = {"case_id": "P1_NATIVE_BUILDER_SMOKE", "status": "FAIL", "python": sys.executable}
    try:
        import torch
        import torch_npu  # noqa: F401
        from dsv4_csa_native_fixture import make_config, make_swa_builder, native_session, register_rope

        load_native_extension(repo)
        import vllm_ascend.ops  # noqa: F401 -- initialize DeviceOperator before dsa_v1
        from vllm_ascend.attention.attention_v1 import AscendAttentionState
        from vllm_ascend.attention.utils import AscendCommonAttentionMetadata
        from vllm_ascend.worker.block_table import BlockTable

        config = make_config(args.checkpoint)
        device = torch.device(f"npu:{args.device}")
        with native_session(config, args.device):
            rope = register_rope(config, "model.layers.2.attn.attn")
            owner, spec, builder = make_swa_builder(config, device)
            batch, query, table_stride = 4, 6, 4112
            table = BlockTable(
                block_size=32,
                max_num_reqs=40,
                max_num_blocks_per_req=table_stride,
                max_num_batched_tokens=400,
                pin_memory=True,
                device=device,
                num_speculative_tokens=5,
            )
            for row in range(batch):
                table.add_row(list(range((row + 1) * table_stride, (row + 2) * table_stride)), row)
            table.commit_block_table(batch)
            starts = torch.tensor([131071, 131072, 131073, 131078], dtype=torch.int32)
            bounds = torch.arange(batch + 1, dtype=torch.int32) * query
            seq_cpu = starts + query
            positions_cpu = (starts[:, None] + torch.arange(query)).reshape(-1).to(torch.int64)
            bounds_gpu, seq_gpu, positions_gpu = [t.to(device) for t in (bounds, seq_cpu, positions_cpu)]
            table.compute_slot_mapping(batch, bounds_gpu, positions_gpu)
            common = AscendCommonAttentionMetadata(
                query_start_loc=bounds_gpu,
                query_start_loc_cpu=bounds,
                seq_lens=seq_gpu,
                _seq_lens_cpu=seq_cpu,
                seq_lens_cpu=seq_cpu,
                num_reqs=batch,
                num_actual_tokens=batch * query,
                num_input_tokens=batch * query,
                max_query_len=query,
                max_seq_len=int(seq_cpu.max()),
                block_table_tensor=table.block_table.gpu[:batch],
                slot_mapping=table.slot_mapping.gpu[: batch * query],
                positions=positions_gpu,
                attn_state=AscendAttentionState.SpecDecoding,
                causal=True,
            )
            metadata = builder.build(0, common, common_ratio_to_sas_metadata={}, num_actual_reqs=batch)
            torch.npu.synchronize()
            request = metadata.req_metadata
            slots = request.slot_mapping.cpu()
            expected_slots = torch.empty_like(slots)
            for row in range(batch):
                for q in range(query):
                    position = int(positions_cpu[row * query + q])
                    logical, offset = divmod(position, spec.block_size)
                    expected_slots[row * query + q] = torch.tensor(
                        [int(table.block_table.cpu[row, logical]), offset], dtype=torch.int32
                    )
            torch.testing.assert_close(slots, expected_slots, rtol=0, atol=0)
            torch.testing.assert_close(request.seq_lens.cpu(), seq_cpu, rtol=0, atol=0)
            torch.testing.assert_close(request.start_pos.cpu(), starts, rtol=0, atol=0)
            report.update(
                status="PASS",
                scope="SWA_group_only_real_native_block_table_and_builder_not_full_P1",
                config_quantization=config.model_config.quantization,
                checkpoint=str(args.checkpoint),
                drafter_weights_loaded=False,
                slot_exact=True,
                lengths_exact=True,
                start_pos_exact=True,
                spec=repr(spec),
                block_table_stride=list(request.block_table.stride()),
                slot_shape=list(request.slot_mapping.shape),
                slots=slots.tolist(),
            )
            del rope, owner
    except Exception:
        report["error"] = traceback.format_exc()
        raise
    finally:
        path = args.output_dir / f"native_builder_device{args.device}.json"
        write_json(path, report)
        print(f"{report['case_id']}: {report['status']} ({path})", flush=True)


if __name__ == "__main__":
    main()
