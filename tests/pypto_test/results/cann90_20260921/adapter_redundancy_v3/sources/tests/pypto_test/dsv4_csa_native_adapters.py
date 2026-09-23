"""Validate CSA adapters against real Native metadata and guarded cache views."""

from __future__ import annotations

import argparse
import traceback
from pathlib import Path

from dsv4_csa_env import activate, load_native_extension, write_json


def run(config, attention, fixture, report):
    import pypto.torch
    import torch

    from vllm_ascend.ops.pypto.deepseek_v4_flash_dspark import native_metadata as kernels
    from vllm_ascend.ops.pypto.deepseek_v4_flash_dspark.native_storage import (
        indexer_storage,
        physical_pages,
        table_storage,
    )

    groups = fixture["groups"]
    req = {name: fixture["metadata"][group["prefix"]].req_metadata for name, group in groups.items()}
    positions = fixture["positions"]
    device = positions.device
    batch = req["swa"].seq_lens.numel()
    tokens = positions.numel()
    executor = fixture["executor"]
    executor.submit(fixture["tasks"])
    for task in fixture["tasks"]:
        executor.wait(task.stage, task.group_id)

    def empty(shape, dtype):
        return torch.empty(shape, dtype=dtype, device=device)

    names = (
        "gather_state_window",
        "commit_state_window",
    )
    ops = {name: pypto.torch.register(getattr(kernels, name), f"dsv4_csa_adapters::{name}") for name in names}
    cpu_positions = positions.cpu()
    cpu_bounds = req["swa"].query_start_loc.cpu()
    ring_rows = torch.empty(tokens, dtype=torch.int64)
    for request in range(batch):
        begin, end = int(cpu_bounds[request]), int(cpu_bounds[request + 1])
        ring_rows[begin:end] = request * 14 + cpu_positions[begin:end] % 14

    main = req["compressed"]
    # Use exactly the named layer RoPE selected by Native attention.
    layer_name = groups["compressed"]["prefix"]
    cos = main.cos[layer_name][:tokens].view(tokens, 64)
    sin = main.sin[layer_name][:tokens].view(tokens, 64)
    for source in (cos, sin):
        native_rows = source.cpu().contiguous()
        # The removed adapter extracted even columns, then consumers duplicated
        # them again. Validate bit patterns against the actual Native producer.
        old_round_trip = native_rows[:, ::2].repeat_interleave(2, dim=-1)
        torch.testing.assert_close(native_rows.view(torch.int32), old_round_trip.view(torch.int32), atol=0, rtol=0)
    report["native_rope_direct_layout_exact"] = "PASS"

    # Compact metadata now goes directly to the production consumers; its
    # addressing is covered by full_compare/full_replay, not a removed adapter.
    for name in ("state", "indexer_state"):
        group, metadata = groups[name], req[name]
        view = group["views"][0]
        native = physical_pages(view)
        width = view.shape[-1]
        scratch = empty((batch * 7, 2, width), torch.float32)
        before = group["allocation"].cpu()
        source = view.cpu()
        table = metadata.block_table.cpu()
        lengths = metadata.seq_lens.cpu()
        expected = torch.zeros((batch * 14, width), dtype=torch.float32)
        for request in range(batch):
            start = int(lengths[request] - (cpu_bounds[request + 1] - cpu_bounds[request]))
            for absolute in range(max(start - 8, 0), start):
                expected[request * 14 + absolute % 14] = source[table[request, absolute // 2], absolute % 2, 0]
        ops["gather_state_window"](
            native, table_storage(metadata.block_table), metadata.query_start_loc, metadata.seq_lens, scratch
        )
        torch.testing.assert_close(scratch.cpu().view(batch * 14, width), expected, atol=0, rtol=0)
        torch.testing.assert_close(group["allocation"].cpu(), before, atol=0, rtol=0)
        # Commit distinct, exactly representable values. Whole-allocation
        # comparison proves page padding, old rows and guard bytes survive.
        updates = (torch.arange(tokens, dtype=torch.float32)[:, None] + 100).expand(tokens, width).contiguous()
        scratch.view(batch * 14, width)[ring_rows.to(device)] = updates.to(device)
        expected_allocation = before.clone()
        expected_view = expected_allocation.view(torch.float32).as_strided(
            view.shape, view.stride(), view.storage_offset()
        )
        native_slots = metadata.slot_mapping.cpu()
        for token in range(tokens):
            page, offset = native_slots[token].tolist()
            if page >= 0 and offset >= 0:
                expected_view[page, offset, 0] = updates[token]
        ops["commit_state_window"](
            scratch, metadata.slot_mapping, positions, metadata.query_start_loc, metadata.seq_lens, native
        )
        torch.testing.assert_close(group["allocation"].cpu(), expected_allocation, atol=0, rtol=0)
        report[f"{name}_gather_commit_and_guards"] = "PASS"

    key, scale = groups["indexer"]["views"]
    carrier = indexer_storage(key, scale)
    assert carrier.untyped_storage().data_ptr() == key.untyped_storage().data_ptr()
    assert carrier.data_ptr() == key.data_ptr()
    report["indexer_single_storage_descriptor"] = "PASS"
    executor.release()
    torch.npu.synchronize()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--history", type=int, default=131072)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    repo = activate()
    report = {
        "status": "FAIL",
        "batch": args.batch,
        "history": args.history,
        "scope": "Native metadata/state adapters; no attention computation or checkpoint acceptance",
    }
    try:
        import pypto.torch
        import torch
        import torch_npu  # noqa: F401
        from dsv4_csa_native_fixture import make_attention, make_config, native_session
        from dsv4_csa_native_forward import make_numerical_fixture

        load_native_extension(repo)
        import vllm_ascend.ops  # noqa: F401

        config = make_config(args.checkpoint)
        device = torch.device(f"npu:{args.device}")
        with native_session(config, args.device), torch.inference_mode():
            pypto.torch.init(device=args.device, platform="a2a3", runtime="tensormap_and_ringbuffer")
            attention = make_attention(config, device)
            fixture = make_numerical_fixture(config, device, attention, args.batch, args.history, 1024)
            run(config, attention, fixture, report)
            report["status"] = "PASS"
            print("Native metadata and state adapters: PASS", flush=True)
    except Exception:
        report["error"] = traceback.format_exc()
        raise
    finally:
        write_json(args.output_dir / "native_adapters.json", report)


if __name__ == "__main__":
    main()
