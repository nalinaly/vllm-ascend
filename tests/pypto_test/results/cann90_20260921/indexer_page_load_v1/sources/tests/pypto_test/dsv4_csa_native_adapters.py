"""Validate zero-copy CSA descriptors against real Native metadata and cache views."""

from __future__ import annotations

import argparse
import traceback
from pathlib import Path

from dsv4_csa_env import activate, load_native_extension, write_json


def run(config, attention, fixture, report):
    import torch

    from vllm_ascend.ops.pypto.deepseek_v4_flash_dspark.native_storage import (
        indexer_storage,
        physical_pages,
        table_storage,
    )

    groups = fixture["groups"]
    req = {name: fixture["metadata"][group["prefix"]].req_metadata for name, group in groups.items()}
    positions = fixture["positions"]
    tokens = positions.numel()
    executor = fixture["executor"]
    executor.submit(fixture["tasks"])
    for task in fixture["tasks"]:
        executor.wait(task.stage, task.group_id)

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

    # State now goes directly to CSA. Check the physical carrier's row pitch,
    # pointer and contents; writes are exercised by full_compare/full_replay.
    for name in ("state", "indexer_state"):
        view = groups[name]["views"][0]
        native = physical_pages(view)
        table = table_storage(req[name].block_table)
        assert native.data_ptr() == view.data_ptr()
        assert native.shape == (view.shape[0], view.stride(0))
        assert table.data_ptr() == req[name].block_table.data_ptr()
        torch.testing.assert_close(
            native[:, :view.shape[1] * view.shape[-1]].cpu().reshape(view.shape), view.cpu(), atol=0, rtol=0
        )
        report[f"{name}_native_storage_descriptor"] = "PASS"

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
        "scope": "Native zero-copy metadata/state descriptors; no attention computation or checkpoint acceptance",
    }
    try:
        import torch
        import torch_npu  # noqa: F401
        from dsv4_csa_native_fixture import make_attention, make_config, native_session
        from dsv4_csa_native_forward import make_numerical_fixture

        load_native_extension(repo)
        import vllm_ascend.ops  # noqa: F401

        config = make_config(args.checkpoint)
        device = torch.device(f"npu:{args.device}")
        with native_session(config, args.device), torch.inference_mode():
            attention = make_attention(config, device)
            fixture = make_numerical_fixture(config, device, attention, args.batch, args.history, 1024)
            run(config, attention, fixture, report)
            report["status"] = "PASS"
            print("Native metadata and state descriptors: PASS", flush=True)
    except Exception:
        report["error"] = traceback.format_exc()
        raise
    finally:
        write_json(args.output_dir / "native_adapters.json", report)


if __name__ == "__main__":
    main()
