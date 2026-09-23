"""PTO transport precheck. This does not substitute for the native-builder P1 suite."""

from __future__ import annotations

import argparse
import traceback
from pathlib import Path

from dsv4_csa_env import activate, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    activate()
    import pypto.torch
    import torch
    import torch_npu
    from dsv4_csa_metadata_kernel import metadata_probe

    report = {
        "case_id": "P1_TRANSPORT_PRECHECK",
        "phase": "P1",
        "status": "FAIL",
        "scope": "synthetic_metadata_only_native_builder_not_covered",
        "device": args.device,
    }
    try:
        torch.npu.set_device(args.device)
        pypto.torch.init(device=args.device, platform="a2a3", runtime="tensormap_and_ringbuffer")
        op = pypto.torch.register(metadata_probe, "dsv4_csa_test::metadata_probe")
        batch, query, capacity, page_size = 4, 6, 32, 32
        table_stride, cache_stride = 4112, 40
        starts = [131071, 131072, 131073, 131078]
        bounds = torch.arange(batch + 1, dtype=torch.int32) * query
        seq_lens = torch.tensor([p + query for p in starts], dtype=torch.int32)
        positions = torch.zeros(capacity, dtype=torch.int64)
        table = torch.full((batch, table_stride), -1, dtype=torch.int32)
        slots = torch.full((capacity, 2), -1, dtype=torch.int32)
        for request, start in enumerate(starts):
            for delta in range(query):
                token, position = request * query + delta, start + delta
                positions[token] = position
                logical_page, offset = divmod(position, page_size)
                physical_page = request * 3 + logical_page % 3
                table[request, logical_page] = physical_page
                slots[token] = torch.tensor([physical_page, offset])
        cache = torch.arange(batch * 3 * cache_stride, dtype=torch.int32) + 1000
        expected = torch.full((capacity, 16), -1, dtype=torch.int64)
        expected_cache = cache.clone()
        for request, start in enumerate(starts):
            for delta in range(query):
                token, position = request * query + delta, start + delta
                logical_page, offset = divmod(position, page_size)
                physical_page = int(table[request, logical_page])
                cache_index = physical_page * cache_stride + offset
                expected[token] = torch.tensor(
                    [
                        request,
                        request * query,
                        (request + 1) * query,
                        start + query,
                        position,
                        start,
                        logical_page,
                        physical_page,
                        offset,
                        physical_page * page_size + offset,
                        physical_page * page_size + offset,
                        int(cache[cache_index]),
                        1,
                        query,
                        delta,
                        (start + query) // 4,
                    ]
                )
                expected_cache[cache_index] = position % 1000000
        device_args = [tensor.to("npu") for tensor in (bounds, seq_lens, positions, table.flatten(), slots, cache)]
        out = torch.empty((capacity, 16), dtype=torch.int64, device="npu")
        call_args = (*device_args, table_stride, 0, page_size, cache_stride, 0, 1, 1, out)
        op(*call_args)
        torch.npu.synchronize()
        torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)
        torch.testing.assert_close(device_args[5].cpu(), expected_cache, rtol=0, atol=0)
        device_args[5].copy_(cache)
        graph = torch_npu.npu.NPUGraph()
        with torch_npu.npu.graph(graph):
            op(*call_args)
        torch.npu.synchronize()
        device_args[5].copy_(cache)
        out.fill_(-99)
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)
        torch.testing.assert_close(device_args[5].cpu(), expected_cache, rtol=0, atol=0)
        report.update(
            status="PASS",
            eager_exact=True,
            graph_exact=True,
            cache_and_padding_exact=True,
            torch_schema=str(op._schema),
        )
        del graph
    except Exception:
        report["error"] = traceback.format_exc()
        raise
    finally:
        write_json(args.output_dir / f"transport_device{args.device}.json", report)
        print(report)


if __name__ == "__main__":
    main()
