"""Five native cache groups -> device slot adapter -> PTO, including async graph replay.

Cache payloads here are diagnostic int32 sentinels in real native allocations.
This is a metadata/layout test, not an attention numerical result.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

from dsv4_csa_env import activate, load_native_extension, write_json


def run_batch(config, device, batch, groups, probe, expand, steps=0, graph_capacity=None):
    import torch
    import torch_npu
    from dsv4_csa_contract import run_rejection_cases, validate_allocator_snapshot, validate_decode_contract
    from dsv4_csa_native_layout import allocate_native_cache
    from vllm.forward_context import BatchDescriptor, ForwardContext, get_forward_context, override_forward_context

    from vllm_ascend.attention.attention_v1 import AscendAttentionState
    from vllm_ascend.attention.utils import AscendCommonAttentionMetadata
    from vllm_ascend.worker.block_table import BlockTable
    from vllm_ascend.worker.device_metadata import DeviceMetadataExecutor

    query = 6
    actual = batch * query
    capacity = graph_capacity if graph_capacity is not None else (actual + 31) // 32 * 32
    if capacity < actual:
        raise ValueError("graph capacity is smaller than the local active token count")
    bounds_cpu = torch.arange(batch + 1, dtype=torch.int32) * query
    bounds = bounds_cpu.to(device)
    lengths = torch.empty(batch, dtype=torch.int32, device=device)
    positions = torch.empty(capacity, dtype=torch.int64, device=device)
    descriptor = BatchDescriptor(num_tokens=capacity, num_reqs=batch, uniform=True)
    executor = DeviceMetadataExecutor()
    context = ForwardContext(
        no_compile_layers=config.compilation_config.static_forward_context,
        attn_metadata={},
        slot_mapping={},
    )
    context.device_metadata_executor = executor
    layouts = {}
    snapshots = []
    negative_cases = []
    for group_index, (kind, group) in enumerate(groups.items()):
        # Different physical numbering and page contents for every group.
        # Nine pages/request suffice for accessed SWA/state/current compressed
        # positions. This test does not read the full 128K attention history.
        pages = batch * 9 + 7 + group_index
        layouts[kind] = allocate_native_cache(group, pages, device)
        spec = group["spec"]
        table_stride = (131104 + steps * query + spec.block_size - 1) // spec.block_size + 8
        table = BlockTable(
            block_size=spec.block_size,
            max_num_reqs=40,
            max_num_blocks_per_req=table_stride,
            max_num_batched_tokens=400,
            pin_memory=True,
            device=device,
            num_speculative_tokens=5,
        )
        cache = group["allocation"].view(torch.int32)
        seed = torch.arange(cache.numel(), dtype=torch.int32) + 10000000 * (group_index + 1)
        view = group["views"][0]
        token_stride = view.stride(1) * view.element_size() // 4
        group.update(
            table=table,
            index=group_index,
            table_stride=table_stride,
            cache=cache,
            seed=seed,
            token_stride=token_stride,
            cache_offset=view.storage_offset() * view.element_size() // 4,
            page_stride=spec.page_size_bytes // 4,
            output=torch.empty((capacity, 16), dtype=torch.int64, device=device),
            expanded=torch.empty((capacity, 2), dtype=torch.int32, device=device),
        )

    def prepare(variant, starts_override=None, starts_device=None):
        starts = (
            starts_override
            if starts_override is not None
            else torch.tensor([131071 + row % 3 + variant * (row % 4 + 1) for row in range(batch)], dtype=torch.int32)
        )
        seq = starts + query
        upper = seq + 5
        cpu_positions = torch.zeros(capacity, dtype=torch.int64)
        cpu_positions[:actual] = (starts[:, None] + torch.arange(query)).reshape(-1)
        if starts_device is None:
            lengths.copy_(seq)
            positions.copy_(cpu_positions)
        else:
            lengths.copy_(starts_device + query)
            positions.zero_()
            positions[:actual].copy_((starts_device[:, None] + torch.arange(query, device=device)).flatten())
        expected = {}
        tasks = []
        shared_metadata = {}
        pointers = {}
        for kind, group in groups.items():
            table, index = group["table"], group["index"]
            group["reference_seq_lens"] = seq.clone()
            for row in range(batch):
                # NumPy writes the native CPU allocator mirror in bulk.
                # The IDs and subsequent commit/slot calculation stay native.
                import numpy as np

                columns = np.arange(group["table_stride"])
                table.add_row((1 + index + row * 9 + (columns * 5 + variant * 2 + index) % 9).tolist(), row)
            # Exercise actual native row operations, including deletion/new row.
            if variant:
                table.swap_row(0, batch - 1)
                table.move_row(1, batch - 1)
                table.clear_row(1)
                table.add_row([1 + index + (column * 7 + 2) % 9 for column in range(group["table_stride"])], 1)
            table.commit_block_table(batch)
            table.compute_slot_mapping(batch, bounds, positions)
            table.slot_mapping.gpu[1] = -1
            common = AscendCommonAttentionMetadata(
                query_start_loc=bounds,
                query_start_loc_cpu=bounds_cpu,
                seq_lens=lengths,
                _seq_lens_cpu=upper,
                seq_lens_cpu=None,
                num_reqs=batch,
                num_actual_tokens=actual,
                num_input_tokens=capacity,
                max_query_len=query,
                max_seq_len=int(upper.max()),
                block_table_tensor=table.block_table.gpu[:batch],
                slot_mapping=table.slot_mapping.gpu[:capacity],
                positions=positions,
                attn_state=AscendAttentionState.SpecDecoding,
                causal=True,
            )
            builder = group["builder"]
            metadata = builder.build(0, common, common_ratio_to_sas_metadata=shared_metadata, num_actual_reqs=batch)
            validate_decode_contract(metadata, group, bounds_cpu, positions)
            validate_allocator_snapshot(group, batch, int(upper.max()))
            if not negative_cases:
                negative_cases.extend(run_rejection_cases(metadata, group, bounds_cpu, positions))
            context.attn_metadata[group["prefix"]] = metadata
            tasks.extend(builder.take_device_metadata_tasks())
            req = metadata.req_metadata
            compact = builder.compressor_ratio == 4
            slots = req.compressor_metadata[2] if compact else req.slot_mapping
            pointers[kind] = [
                req.block_table.data_ptr(),
                slots.data_ptr(),
                req.seq_lens.data_ptr(),
                req.start_pos.data_ptr(),
            ]
            expected_output = torch.full((capacity, 16), -1, dtype=torch.int64)
            expected_cache = group["seed"].clone()
            expanded_slots = torch.full((capacity, 2), -1, dtype=torch.int32)
            compact_rows = []
            for row in range(batch):
                for delta in range(query):
                    token = row * query + delta
                    position = int(cpu_positions[token])
                    if (compact and (position + 1) % 4) or (not compact and token == 1):
                        continue
                    cache_position = position // (4 if compact else 1)
                    logical, offset = divmod(cache_position, group["page_size"])
                    physical = int(table.block_table.cpu[row, logical])
                    flat = physical * group["page_size"] + offset
                    cache_index = (
                        group["cache_offset"] + physical * group["page_stride"] + offset * group["token_stride"]
                    )
                    expected_output[token] = torch.tensor(
                        [
                            row,
                            row * query,
                            (row + 1) * query,
                            int(seq[row]),
                            position,
                            int(starts[row]),
                            logical,
                            physical,
                            offset,
                            flat,
                            flat,
                            int(group["seed"][cache_index]),
                            1,
                            query,
                            delta,
                            int(seq[row]) // 4,
                        ],
                        dtype=torch.int64,
                    )
                    expected_cache[cache_index] = position % 1000000
                    expanded_slots[token] = torch.tensor([physical, offset], dtype=torch.int32)
                    if compact:
                        compact_rows.append([physical, offset])
            group["cache"].copy_(group["seed"])
            group["output"].fill_(-99)
            expected[kind] = (expected_output, expected_cache, expanded_slots, compact_rows)
        executor.submit(tasks, batch_descriptor=descriptor)
        snapshots.append({"variant": variant, "pointers": pointers})
        return expected, tasks

    def invoke(tasks):
        # Native stream/event dependency, with no synchronize between producer
        # and consumer. The external waits/resets are captured in the graph.
        for task in tasks:
            executor.wait(task.stage, task.group_id)
        for kind, group in groups.items():
            request = get_forward_context().attn_metadata[group["prefix"]].req_metadata
            if request.cache_group_key != group["prefix"]:
                raise ValueError(f"wrong cache group for {kind}")
            compact = group["builder"].compressor_ratio == 4
            if compact:
                expand(
                    request.query_start_loc,
                    request.seq_lens,
                    positions,
                    request.compressor_metadata[2],
                    group["expanded"],
                )
                slots = group["expanded"]
            else:
                slots = request.slot_mapping
            probe(
                request.query_start_loc,
                request.seq_lens,
                positions,
                group["table"].block_table.gpu.view(-1),
                slots,
                group["cache"],
                request.block_table.stride(0),
                request.block_table.storage_offset(),
                group["page_size"],
                group["page_stride"],
                group["cache_offset"],
                4 if compact else 1,
                group["token_stride"],
                group["output"],
            )

    def compare(expected):
        # Synchronization is only for assertion after the consumer/reuse fence.
        torch.npu.synchronize()
        for kind, (expected_output, expected_cache, expanded, compact_rows) in expected.items():
            group = groups[kind]
            torch.testing.assert_close(group["output"].cpu(), expected_output, rtol=0, atol=0, msg=kind + " output")
            torch.testing.assert_close(group["cache"].cpu(), expected_cache, rtol=0, atol=0, msg=kind + " cache/guards")
            if group["builder"].compressor_ratio == 4:
                request = context.attn_metadata[group["prefix"]].req_metadata
                torch.testing.assert_close(
                    request.qli_seqused_k.cpu(),
                    group["reference_seq_lens"] // 4,
                    rtol=0,
                    atol=0,
                    msg=kind + " QLI compressed lengths",
                )
                torch.testing.assert_close(
                    request.qli_cmp_residual_k.cpu(),
                    group["reference_seq_lens"] % 4,
                    rtol=0,
                    atol=0,
                    msg=kind + " QLI residual lengths",
                )
            if compact_rows:
                request = context.attn_metadata[group["prefix"]].req_metadata
                compact = request.compressor_metadata[2].cpu()
                torch.testing.assert_close(
                    compact[: len(compact_rows)], torch.tensor(compact_rows, dtype=torch.int32), rtol=0, atol=0
                )
                assert (compact[len(compact_rows) :, 0] == -1).all(), kind + " compact tail"
                torch.testing.assert_close(
                    group["expanded"].cpu(), expanded, rtol=0, atol=0, msg=kind + " expanded slots"
                )

    with override_forward_context(context):
        expected, tasks = prepare(0)
        invoke(tasks)
        executor.release()
        compare(expected)
        expected, tasks = prepare(0)
        graph = torch_npu.npu.NPUGraph()
        with torch_npu.npu.graph(graph):
            invoke(tasks)
        executor.release()
        torch.npu.synchronize()
        for variant in (0, 1, 0):
            expected, tasks = prepare(variant)
            graph.replay()
            executor.release()
            compare(expected)
        if steps:
            from vllm_ascend.spec_decode.utils import (
                correct_optimistic_seq_lens_cpu,
                update_num_computed_tokens_for_batch_change,
            )

            starts_reference = torch.tensor([131071 + row % 3 for row in range(batch)], dtype=torch.int32)
            computed = starts_reference.to(device)
            accepted = torch.ones(batch, dtype=torch.int32, device=device)
            previous_drafts = torch.full((batch,), 5, dtype=torch.int32, device=device)
            advances = []
            for step in range(steps):
                counts = torch.tensor([[1, 6, 2, 5, 5][(step + row) % 5] for row in range(batch)], dtype=torch.int32)
                previous_cpu = torch.arange(batch, dtype=torch.int64)
                if step in (25, 75):
                    previous_cpu = previous_cpu.flip(0)
                optimistic_cpu = starts_reference[previous_cpu] + query
                corrected_cpu = optimistic_cpu.clone()
                correct_optimistic_seq_lens_cpu(
                    corrected_cpu.numpy(),
                    previous_cpu.numpy(),
                    torch.full((batch,), 5, dtype=torch.int32).numpy(),
                    counts.numpy(),
                    batch,
                )
                update_num_computed_tokens_for_batch_change(
                    computed,
                    accepted,
                    previous_cpu.to(device),
                    counts.to(device),
                    previous_drafts,
                    optimistic_cpu.to(device),
                )
                starts_reference = starts_reference[previous_cpu] + counts[previous_cpu]
                torch.testing.assert_close(corrected_cpu, starts_reference, rtol=0, atol=0)
                expected, tasks = prepare(0, starts_reference, computed)
                graph.replay()
                executor.release()
                compare(expected)
                # Only test assertions transfer results back, after all kernels.
                torch.testing.assert_close(computed.cpu(), starts_reference, rtol=0, atol=0)
                torch.testing.assert_close(accepted.cpu(), counts[previous_cpu], rtol=0, atol=0)
                advances.extend(counts.tolist())
            assert sum(advances) / len(advances) == 3.8
        del graph
    assert all(snapshot["pointers"] == snapshots[0]["pointers"] for snapshot in snapshots)
    return {
        "status": "PASS",
        "batch": batch,
        "T_actual": actual,
        "T_capacity": capacity,
        "groups": layouts,
        "eager_exact": True,
        "graph_A_B_A_exact": True,
        "all_cache_bytes_and_guards_exact": True,
        "native_compact_slots_and_device_expansion_exact": True,
        "native_device_metadata_executor_external_events": True,
        "pointers_stable": True,
        "native_swap_move_clear_add_rows": True,
        "scope": "five_group_metadata_not_attention_forward",
        "snapshots": snapshots,
        "negative_cases": negative_cases,
        "acceptance_trace": {
            "steps": steps,
            "native_gpu_and_cpu_corrections": bool(steps),
            "injected_mean_advance": 3.8 if steps else None,
            "scope": "metadata_and_sentinel_only_full_CSA_state_trajectory_pending",
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batches", type=int, nargs="+", default=[4, 8, 16, 24, 32, 40])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=0)
    args = parser.parse_args()
    repo = activate()
    report = {"case_id": "P1_NATIVE_FIVE_GROUPS", "status": "FAIL", "python": sys.executable, "cases": []}
    try:
        import pypto.torch
        import torch
        import torch_npu  # noqa: F401
        from dsv4_csa_metadata_kernel import expand_compressed_slots, metadata_probe
        from dsv4_csa_native_fixture import make_attention, make_cache_groups, make_config, native_session

        load_native_extension(repo)
        import vllm_ascend.ops  # noqa: F401

        config = make_config(args.checkpoint)
        device = torch.device(f"npu:{args.device}")
        with native_session(config, args.device):
            attention = make_attention(config, device)
            groups = make_cache_groups(config, device, attention)
            pypto.torch.init(device=args.device, platform="a2a3", runtime="tensormap_and_ringbuffer")
            probe = pypto.torch.register(metadata_probe, "dsv4_csa_test::all_group_probe")
            expand = pypto.torch.register(expand_compressed_slots, "dsv4_csa_test::expand_slots")
            for batch in args.batches:
                result = run_batch(config, device, batch, groups, probe, expand, steps=args.steps)
                report["cases"].append(result)
                write_json(args.output_dir / f"native_groups_B{batch}_device{args.device}.json", result)
                print(f"native five-group transport B={batch}: PASS", flush=True)
            report.update(
                status="PASS",
                scope="native_metadata_transport_and_sentinel_only",
                rejection_cases="PASS",
                formal_case_mapping="DSV4_FLASH_CSA_VALIDATION_LOG.md#13",
            )
    except Exception:
        report["error"] = traceback.format_exc()
        raise
    finally:
        write_json(args.output_dir / f"native_groups_device{args.device}.json", report)


if __name__ == "__main__":
    main()
