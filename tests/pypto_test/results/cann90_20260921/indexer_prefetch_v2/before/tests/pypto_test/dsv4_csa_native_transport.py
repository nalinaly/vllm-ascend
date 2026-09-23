"""SWA metadata: native BlockTable/builder/ForwardContext -> PTO, eager and graph.

This matrix covers the raw-cache group. P1 remains incomplete until compressed
and both state groups, lifecycle and rejection cases are also covered.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

from dsv4_csa_env import activate, load_native_extension, write_json


def run_batch(config, device, batch: int, builder, prefix: str, op):
    import torch
    import torch_npu
    from vllm.forward_context import ForwardContext, get_forward_context, override_forward_context

    from vllm_ascend.attention.attention_v1 import AscendAttentionState
    from vllm_ascend.attention.utils import AscendCommonAttentionMetadata
    from vllm_ascend.worker.block_table import BlockTable

    query, page_size, cache_stride, table_stride = 6, 32, 40, 4112
    actual = batch * query
    capacity = (actual + 31) // 32 * 32
    table = BlockTable(
        block_size=page_size,
        max_num_reqs=40,
        max_num_blocks_per_req=table_stride,
        max_num_batched_tokens=400,
        pin_memory=True,
        device=device,
        num_speculative_tokens=5,
    )
    # Nine physical pages per request cover more than the live SWA window.
    # Old logical columns may reuse evicted pages; this is a metadata fixture,
    # not a compressed full-history numerical fixture.
    pages = batch * 9 + 1
    cache_cpu = torch.arange(pages * cache_stride, dtype=torch.int32) + 1234
    cache = cache_cpu.to(device)
    output = torch.empty((capacity, 16), dtype=torch.int64, device=device)
    positions = torch.zeros(capacity, dtype=torch.int64, device=device)
    bounds_cpu = torch.arange(batch + 1, dtype=torch.int32) * query
    bounds = bounds_cpu.to(device)
    lengths = torch.empty(batch, dtype=torch.int32, device=device)
    # Passing a flat view of the entire native allocation preserves row stride.
    table_storage = table.block_table.gpu.view(-1)
    context = ForwardContext(
        no_compile_layers=config.compilation_config.static_forward_context,
        attn_metadata={},
        slot_mapping={},
    )
    snapshots = []

    def prepare(variant: int):
        # A and B differ in exact lengths and allocator pages, while all device
        # input buffers retain the same addresses for replay.
        starts = torch.tensor([131071 + row % 3 + variant * (row % 4 + 1) for row in range(batch)], dtype=torch.int32)
        seq = starts + query
        host_upper_bound = seq + 5
        cpu_positions = torch.zeros(capacity, dtype=torch.int64)
        cpu_positions[:actual] = (starts[:, None] + torch.arange(query)).reshape(-1)
        for row in range(batch):
            table.add_row([1 + row * 9 + (column * 5 + variant * 2) % 9 for column in range(table_stride)], row)
        table.commit_block_table(batch)
        bounds.copy_(bounds_cpu)
        lengths.copy_(seq)
        positions.copy_(cpu_positions)
        table.compute_slot_mapping(batch, bounds, positions)
        # Native formatting must preserve an invalid slot on a real token too.
        table.slot_mapping.gpu[1] = -1
        common = AscendCommonAttentionMetadata(
            query_start_loc=bounds,
            query_start_loc_cpu=bounds_cpu,
            seq_lens=lengths,
            _seq_lens_cpu=host_upper_bound,
            seq_lens_cpu=None,
            num_reqs=batch,
            num_actual_tokens=actual,
            num_input_tokens=capacity,
            max_query_len=query,
            max_seq_len=int(host_upper_bound.max()),
            block_table_tensor=table.block_table.gpu[:batch],
            slot_mapping=table.slot_mapping.gpu[:capacity],
            positions=positions,
            attn_state=AscendAttentionState.SpecDecoding,
            causal=True,
        )
        metadata = builder.build(0, common, common_ratio_to_sas_metadata={}, num_actual_reqs=batch)
        context.attn_metadata = {prefix: metadata}
        context.slot_mapping = {prefix: table.slot_mapping.gpu[:capacity]}
        expected = torch.full((capacity, 16), -1, dtype=torch.int64)
        expected_cache = cache_cpu.clone()
        for row in range(batch):
            for delta in range(query):
                token = row * query + delta
                if token == 1:
                    continue
                position = int(cpu_positions[token])
                logical, offset = divmod(position, page_size)
                physical = int(table.block_table.cpu[row, logical])
                cache_index = physical * cache_stride + offset
                flat = physical * page_size + offset
                expected[token] = torch.tensor(
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
                        int(cache_cpu[cache_index]),
                        1,
                        query,
                        delta,
                        int(seq[row]) // 4,
                    ],
                    dtype=torch.int64,
                )
                expected_cache[cache_index] = position % 1000000
        cache.copy_(cache_cpu)
        output.fill_(-99)
        request = metadata.req_metadata
        snapshots.append(
            {
                "variant": variant,
                "cpu_upper_bound": host_upper_bound.tolist(),
                "device_lengths_expected": seq.tolist(),
                "pointers": {
                    "query_start_loc": request.query_start_loc.data_ptr(),
                    "seq_lens": request.seq_lens.data_ptr(),
                    "positions": positions.data_ptr(),
                    "block_table": request.block_table.data_ptr(),
                    "slots": request.slot_mapping.data_ptr(),
                },
            }
        )
        return expected, expected_cache

    def invoke_from_context():
        request = get_forward_context().attn_metadata[prefix].req_metadata
        if request.cache_group_key != prefix:
            raise ValueError("metadata bound to the wrong cache group")
        if request.query_start_loc.numel() != request.seq_lens.numel() + 1:
            raise ValueError("query boundaries and request lengths disagree")
        if request.slot_mapping.shape != (actual, 2):
            raise ValueError("unexpected native A3 slot shape")
        return op(
            request.query_start_loc,
            request.seq_lens,
            positions,
            table_storage,
            request.slot_mapping,
            cache,
            request.block_table.stride(0),
            request.block_table.storage_offset(),
            page_size,
            cache_stride,
            0,
            1,
            1,
            output,
        )

    def compare(expected, expected_cache):
        torch.npu.synchronize()
        torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
        torch.testing.assert_close(cache.cpu(), expected_cache, rtol=0, atol=0)

    with override_forward_context(context):
        expected, expected_cache = prepare(0)
        invoke_from_context()
        compare(expected, expected_cache)
        # Refill after warmup, then capture only the consumer. Native metadata
        # production remains outside this minimal graph, as in the intended
        # producer/consumer boundary; all test copies precede replay on stream.
        prepare(0)
        graph = torch_npu.npu.NPUGraph()
        with torch_npu.npu.graph(graph):
            invoke_from_context()
        torch.npu.synchronize()
        for variant in (0, 1, 0):
            expected, expected_cache = prepare(variant)
            graph.replay()
            compare(expected, expected_cache)
        del graph
    assert all(snapshot["pointers"] == snapshots[0]["pointers"] for snapshot in snapshots)
    return {
        "batch": batch,
        "query": query,
        "T_actual": actual,
        "T_padded": capacity,
        "status": "PASS",
        "scope": "SWA_group_only",
        "eager_exact": True,
        "graph_A_B_A_exact": True,
        "invalid_slot_and_padding_untouched": True,
        "cache_storage_exact": True,
        "host_optimistic_device_actual_differ": True,
        "native_table_row_stride": table.block_table.gpu.stride(0),
        "snapshots": snapshots,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batches", type=int, nargs="+", default=[4, 8, 16, 24, 32, 40])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    repo = activate()
    report = {"case_id": "P1_NATIVE_SWA_TRANSPORT", "status": "FAIL", "python": sys.executable, "cases": []}
    try:
        import pypto.torch
        import torch
        import torch_npu  # noqa: F401
        from dsv4_csa_metadata_kernel import metadata_probe
        from dsv4_csa_native_fixture import make_config, make_swa_builder, native_session, register_rope

        load_native_extension(repo)
        import vllm_ascend.ops  # noqa: F401

        config = make_config(args.checkpoint)
        device = torch.device(f"npu:{args.device}")
        with native_session(config, args.device):
            rope = register_rope(config, "model.layers.2.attn.attn")
            owner, spec, builder = make_swa_builder(config, device)
            pypto.torch.init(device=args.device, platform="a2a3", runtime="tensormap_and_ringbuffer")
            op = pypto.torch.register(metadata_probe, "dsv4_csa_test::native_metadata_probe")
            for batch in args.batches:
                result = run_batch(config, device, batch, builder, owner.prefix, op)
                report["cases"].append(result)
                write_json(args.output_dir / f"native_swa_B{batch}_device{args.device}.json", result)
                print(f"native SWA transport B={batch}: PASS", flush=True)
            report.update(status="PASS", spec=repr(spec), full_P1_status="INCOMPLETE_OTHER_GROUPS_AND_CASES")
            del rope
    except Exception:
        report["error"] = traceback.format_exc()
        raise
    finally:
        write_json(args.output_dir / f"native_swa_transport_device{args.device}.json", report)


if __name__ == "__main__":
    main()
