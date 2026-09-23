"""Reproduce Native scatter writes outside a cache view with nonzero offset."""

from __future__ import annotations

import argparse
import traceback
from pathlib import Path

from dsv4_csa_env import activate, load_native_extension, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    repo = activate()
    report = {"status": "FAIL", "scope": "Native scatter offset diagnostic", "cases": []}
    try:
        import torch
        import torch_npu  # noqa: F401

        torch.npu.set_device(args.device)
        load_native_extension(repo)
        device = torch.device(f"npu:{args.device}")
        pages = 8
        for guard_extra in (0, 128):
            for name, dtype, width, region_offset, tokens, page_bytes in (
                ("compressed", torch.bfloat16, 512, 0, 32, 32768),
                ("indexer_key", torch.int8, 128, 0, 32, 32768),
                ("indexer_scale", torch.float16, 1, 4096, 32, 32768),
                ("wide_row_slice", torch.bfloat16, 131072, 0, 1, 262144),
            ):
                guard = page_bytes + guard_extra
                slots_cpu = torch.tensor(
                    [[3, 0], [-1, tokens - 1], [5, min(2, tokens - 1)], [-1, tokens - 1]], dtype=torch.int64
                )
                slots = slots_cpu.to(device)
                allocation = torch.full((guard + pages * page_bytes + 128,), 37, dtype=torch.int8, device=device)
                size = torch.empty((), dtype=dtype).element_size()
                view = allocation.view(dtype).as_strided(
                    (pages, tokens, 1, width),
                    (page_bytes // size, width, width, 1),
                    (guard + region_offset) // size,
                )
                updates = torch.full((len(slots_cpu), 1, width), 3, dtype=dtype, device=device)
                before = allocation.cpu()
                expected = before.clone()
                expected_view = expected.view(dtype).as_strided(view.shape, view.stride(), view.storage_offset())
                for row, (page, token) in enumerate(slots_cpu.tolist()):
                    if page >= 0:
                        expected_view[page, token].copy_(updates[row].cpu())
                torch.ops._C_ascend.npu_scatter_nd_update_sk(view, slots, updates)
                torch.npu.synchronize()
                actual = allocation.cpu()
                wrong = actual != expected
                changed = actual != before
                case = {
                    "name": name,
                    "guard": guard,
                    "storage_offset_bytes": view.storage_offset() * size,
                    "stride_bytes": [stride * size for stride in view.stride()],
                    "status": "PASS" if not bool(wrong.any()) else "FAIL",
                    "wrong_bytes": int(wrong.sum()),
                    "changed_bytes": int(changed.sum()),
                    "first_wrong_offsets": wrong.nonzero()[:12].flatten().tolist(),
                    "first_changed_offsets": changed.nonzero()[:12].flatten().tolist(),
                }
                report["cases"].append(case)
                print(case, flush=True)
        report["status"] = "PASS" if all(case["status"] == "PASS" for case in report["cases"]) else "FAIL"
        if report["status"] != "PASS":
            raise AssertionError("Native scatter offset check failed")
    except Exception:
        report["error"] = traceback.format_exc()
        raise
    finally:
        write_json(args.output_dir / "scatter_offset.json", report)


if __name__ == "__main__":
    main()
