# SPDX-License-Identifier: Apache-2.0
"""单卡对照现有slot kernel和Native格式；只做整数功能验证，不声称整网收益。"""

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import torch_npu  # noqa: F401
from vllm.v1.worker.block_table import _compute_slot_mapping_kernel

import vllm_ascend.ops  # noqa: F401
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.ops.triton.dsa_slot_mapping import compute_dsa_slot_mapping_kernel
from vllm_ascend.worker.block_table import BlockTable


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int)
    parser.add_argument("--through-block-table", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    device = int(os.environ["TASK_DEVICE"])
    assert args.device is None or args.device == device
    torch.npu.set_device(device)
    target = torch.device(f"npu:{device}")
    cases = []
    for block_size in (2, 32, 128):
        for name, lengths, histories, capacity in (
            ("B16_S6_128K", [6] * 16, [131072 + i for i in range(16)], 112),
            ("mixed_and_padding", [6, 1, 0, 4], [131072, 31, 0, 127], 32),
            ("long_prefill_loop", [1025, 17], [512, 2048], 1056),
            ("empty", [], [], 16),
        ):
            print(f"case={name} block_size={block_size}", flush=True)
            tokens = sum(lengths)
            query = torch.tensor([0, *torch.tensor(lengths, dtype=torch.int32).cumsum(0).tolist()],
                                 dtype=torch.int32, device=target)
            positions_cpu = torch.tensor([h + j for h, q in zip(histories, lengths) for j in range(q)],
                                         dtype=torch.int64)
            positions = positions_cpu.to(target)
            columns = (int(positions_cpu.max()) // block_size + 1) if tokens else 1
            rows = max(1, len(lengths))
            # Preserve a noncontiguous row stride; page IDs need not be ordered.
            storage = (torch.arange(rows * (columns + 7), dtype=torch.int32) % 5003).reshape(rows, columns + 7)
            if tokens:
                storage[0, histories[0] // block_size] = -1
            table = storage.to(target)[:, :columns]
            reference = torch.full((capacity + 4,), -91, dtype=torch.int32, device=target)
            flat = torch.full_like(reference, -91)
            parts = torch.full((capacity + 4, 2), -91, dtype=torch.int32, device=target)
            secondary = torch.full_like(parts, -91)
            write_second = name != "mixed_and_padding"
            _compute_slot_mapping_kernel[(len(lengths) + 1,)](
                tokens, capacity, query, positions, table, table.stride(0), block_size, reference,
                TOTAL_CP_WORLD_SIZE=1, TOTAL_CP_RANK=0, CP_KV_CACHE_INTERLEAVE_SIZE=1,
                PAD_ID=-1, BLOCK_SIZE=1024,
            )
            if args.through_block_table:
                owner = BlockTable.__new__(BlockTable)
                owner.dcp_world_size = 1
                owner.use_hybrid_blocks = False
                owner.max_num_batched_tokens = capacity
                owner.block_size = block_size
                owner.block_table = SimpleNamespace(gpu=table)
                owner.slot_mapping = SimpleNamespace(gpu=flat)
                owner.dsa_slot_mappings = ()
                assert owner.bind_dsa_slot_mappings((parts, secondary) if write_second else (parts,))
                owner.compute_slot_mapping(len(lengths), query, positions)
            else:
                compute_dsa_slot_mapping_kernel[(len(lengths) + 1,)](
                    tokens, capacity, query, positions, table, table.stride(0), flat, parts, secondary,
                    CACHE_BLOCK_SIZE=block_size, BLOCK_SIZE=512, WRITE_SECOND=write_second,
                )
            torch.npu.synchronize()
            expected = reference.cpu()
            flat_cpu, parts_cpu = flat.cpu(), parts.cpu()
            native_parts = DeviceOperator.format_dsa_slot_mapping(expected[:capacity], block_size)
            secondary_cpu = secondary.cpu()
            row = {
                "case": name, "block_size": block_size, "tokens": tokens, "capacity": capacity,
                "flat_exact": torch.equal(flat_cpu, expected),
                "native_parts_exact": torch.equal(parts_cpu[:capacity], native_parts),
                "flat_guard_unchanged": bool((flat_cpu[capacity:] == -91).all()),
                "parts_guard_unchanged": bool((parts_cpu[capacity:] == -91).all()),
                "secondary_exact": torch.equal(secondary_cpu[:capacity], native_parts) if write_second
                                   else bool((secondary_cpu[:capacity] == -91).all()),
                "secondary_guard_unchanged": bool((secondary_cpu[capacity:] == -91).all()),
            }
            cases.append(row)
            assert all(row[key] for key in ("flat_exact", "native_parts_exact", "flat_guard_unchanged",
                                            "parts_guard_unchanged", "secondary_exact",
                                            "secondary_guard_unchanged")), row
    report = {"passed": True, "device": device, "scope": __doc__, "cases": cases,
              "through_block_table": args.through_block_table}
    (args.output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
