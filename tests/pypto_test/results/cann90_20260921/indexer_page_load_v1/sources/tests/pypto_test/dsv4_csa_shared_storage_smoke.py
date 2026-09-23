"""Bit-exact shared K/FP16-scale probe; no full attention claim."""

from __future__ import annotations

import argparse
import traceback
from pathlib import Path

from dsv4_csa_env import activate, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--page-bytes", type=int, choices=(4160, 32768), default=32768)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    activate()
    import pypto.torch
    import torch
    import torch_npu
    from dsv4_csa_shared_storage_kernel import shared_indexer_storage_probe

    report = {
        "case_id": "P1_SHARED_STORAGE_PRECHECK",
        "status": "FAIL",
        "device": args.device,
        "scope": "A3_indexer_page_layout_only_native_runner_not_covered",
    }
    try:
        torch.npu.set_device(args.device)
        pypto.torch.init(device=args.device, platform="a2a3", runtime="tensormap_and_ringbuffer")
        op = pypto.torch.register(shared_indexer_storage_probe, "dsv4_csa_test::shared_indexer_storage_probe")
        pages, page_bytes, prefix = 7, args.page_bytes, 128
        base = torch.full((prefix + pages * page_bytes + prefix,), 37, dtype=torch.int8)
        key = torch.as_strided(base, (pages, 32, 128), (page_bytes, 128, 1), prefix)
        scale = torch.as_strided(
            base.view(torch.float16), (pages, 32, 1), (page_bytes // 2, 1, 1), (prefix + 4096) // 2
        )
        key.copy_((torch.arange(pages * 4096).reshape_as(key) % 127 - 63).to(torch.int8))
        scale.copy_((torch.arange(pages * 32).reshape_as(scale) % 16 + 1).float() / 8)
        assert not key.is_contiguous() and not scale.is_contiguous()
        assert key.untyped_storage().data_ptr() == scale.untyped_storage().data_ptr()
        expected_out = (key.float() * scale.float()).reshape(pages, 4096)
        expected_base = base.clone()
        expected_key = torch.as_strided(expected_base, key.shape, key.stride(), key.storage_offset())
        expected_scale = torch.as_strided(
            expected_base.view(torch.float16), scale.shape, scale.stride(), scale.storage_offset()
        )
        expected_key.neg_()
        expected_scale.add_(0.5)
        device_base = base.to("npu")
        raw = device_base[prefix : prefix + pages * page_bytes].view(pages, page_bytes)
        assert raw.is_contiguous() and raw.storage_offset() == prefix
        observed = torch.empty((pages, 4096), dtype=torch.float32, device="npu")
        op(raw, observed)
        torch.npu.synchronize()
        torch.testing.assert_close(observed.cpu(), expected_out, rtol=0, atol=0)
        torch.testing.assert_close(device_base.cpu(), expected_base, rtol=0, atol=0)
        graph = torch_npu.npu.NPUGraph()
        device_base.copy_(base)
        with torch_npu.npu.graph(graph):
            op(raw, observed)
        torch.npu.synchronize()
        pointer = raw.data_ptr()
        # A -> B -> A at fixed addresses: B changes both quantized K and its
        # FP16 scale. Whole-allocation comparison includes prefix/tail guards.
        for variant in (0, 1, 0):
            source = base.clone()
            if variant:
                source_key = torch.as_strided(source, key.shape, key.stride(), key.storage_offset())
                source_scale = torch.as_strided(
                    source.view(torch.float16), scale.shape, scale.stride(), scale.storage_offset()
                )
                source_key.neg_()
                source_scale.add_(1)
            source_key = torch.as_strided(source, key.shape, key.stride(), key.storage_offset())
            source_scale = torch.as_strided(
                source.view(torch.float16), scale.shape, scale.stride(), scale.storage_offset()
            )
            want_out = (source_key.float() * source_scale.float()).reshape_as(expected_out)
            device_base.copy_(source)
            source_key.neg_()
            source_scale.add_(0.5)
            graph.replay()
            torch.npu.synchronize()
            assert raw.data_ptr() == pointer
            torch.testing.assert_close(observed.cpu(), want_out, rtol=0, atol=0)
            torch.testing.assert_close(device_base.cpu(), source, rtol=0, atol=0)
        report.update(
            status="PASS",
            eager_exact=True,
            graph_A_B_A_exact=True,
            guards_exact=True,
            page_bytes=page_bytes,
            input_offset_bytes=prefix,
            key_stride=list(key.stride()),
            scale_stride=list(scale.stride()),
            key_dtype=str(key.dtype),
            scale_dtype=str(scale.dtype),
            torch_schema=str(op._schema),
        )
        del graph
    except Exception:
        report["error"] = traceback.format_exc()
        raise
    finally:
        write_json(args.output_dir / f"shared_storage_device{args.device}_page{args.page_bytes}.json", report)
        print(report)


if __name__ == "__main__":
    main()
