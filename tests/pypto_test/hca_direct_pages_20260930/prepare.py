"""Cube直接按Native页表装入L1，AIV只准备有效mask与共享零页。"""

import ast
import difflib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PREFIX = ROOT.parents[3] / ".cache/hca-direct-pages-cba6c465-20260930"
RELATIVE = Path("deepseek_v4_flash_hca/decode_sparse_attn_hca.py")

METADATA = '''    # 页表/长度是本次调用输入；这里不读压缩cache，无需等待cache写回。
    cmp_zero_page = pl.create_tensor([CMP_STORAGE_BLOCK_SIZE, HEAD_DIM], dtype=pl.BF16)
    cmp_work_valid = pl.create_tensor([cmp_gather_count, CMP_ATTN_K_TILE], dtype=pl.FP32)
    gather_work_tile = (cmp_gather_count + CMP_GATHER_AIV_WAVE - 1) // CMP_GATHER_AIV_WAVE
    metadata_blocks = pl.max(1, (cmp_gather_count + gather_work_tile - 1) // gather_work_tile)
    with pl.spmd(metadata_blocks, name_hint="hca_cmp_page_metadata", allow_early_resolve=True) as cmp_gather_tid:
        metadata_worker = pl.tile.get_block_idx()
        if metadata_worker == 0:
            zero_page = pl.tile.full([CMP_STORAGE_BLOCK_SIZE, HEAD_DIM], dtype=pl.BF16, value=0.0)
            pl.store(zero_page, [0, 0], cmp_zero_page)
        page_indices = pl.cast(pl.tile.arange(0, [1, CMP_STORAGE_BLOCK_SIZE], dtype=pl.INT32), pl.FP32)
        for gather_item in pl.range(metadata_worker, cmp_gather_count, metadata_blocks):
            gather_request = gather_item // cmp_work_count
            gather_work = gather_item % cmp_work_count
            gather_rows = pl.min(HCA_MAX_COMPRESSED_ROWS, pl.max(
                pl.read(kv_seq_lens, [gather_request]), 0,
            ) // COMPRESS_RATIO)
            gather_mask = pl.tile.full([1, CMP_ATTN_K_TILE], dtype=pl.FP32, value=0.0)
            for gather_page in pl.unroll(CMP_PAGES_PER_WORK):
                page_column = gather_work * CMP_PAGES_PER_WORK + gather_page
                valid_rows = pl.min(CMP_STORAGE_BLOCK_SIZE, pl.max(
                    gather_rows - page_column * CMP_STORAGE_BLOCK_SIZE, 0,
                ))
                if page_column < cmp_table_blocks and valid_rows > 0:
                    physical_page = pl.read(cmp_block_table, [gather_request, page_column])
                    if physical_page >= 0 and physical_page < cmp_block_num:
                        page_mask = pl.minimum(pl.maximum(
                            pl.sub(pl.cast(valid_rows, pl.FP32), page_indices), 0.0,
                        ), 1.0)
                        gather_mask = pl.tile.assemble(
                            gather_mask, page_mask, [0, gather_page * CMP_STORAGE_BLOCK_SIZE],
                        )
            pl.store(gather_mask, [gather_item, 0], cmp_work_valid)

'''

PAGE_LOAD = '''                        for page_part in pl.range(CMP_PAGES_PER_WORK):
                            page_column = tick * CMP_PAGES_PER_WORK + page_part
                            dst_row = l1_row + page_part * CMP_STORAGE_BLOCK_SIZE
                            copy_rows = pl.min(CMP_STORAGE_BLOCK_SIZE, pl.max(
                                cmp_rows - page_column * CMP_STORAGE_BLOCK_SIZE, 0,
                            ))
                            if page_column < cmp_table_blocks and copy_rows > 0:
                                physical_page = pl.yield_(pl.cast(pl.read(
                                    cmp_block_table, [request, page_column],
                                ), pl.INDEX))
                            else:
                                physical_page = pl.yield_(pl.cast(-1, pl.INDEX))
                            if (
                                copy_rows == CMP_STORAGE_BLOCK_SIZE
                                and physical_page >= 0 and physical_page < cmp_block_num
                            ):
                                kv_l1 = pl.gather_row(
                                    kv_l1, cmp_kv_flat, [dst_row, 0],
                                    [physical_page * CMP_STORAGE_BLOCK_SIZE, 0],
                                    [CMP_STORAGE_BLOCK_SIZE, HEAD_DIM],
                                )
                            else:
                                # 部分页先置零，避免零概率乘以未初始化的NaN。
                                kv_l1 = pl.gather_row(
                                    kv_l1, cmp_zero_page, [dst_row, 0], [0, 0],
                                    [CMP_STORAGE_BLOCK_SIZE, HEAD_DIM],
                                )
                                if physical_page >= 0 and physical_page < cmp_block_num and copy_rows > 0:
                                    kv_l1 = pl.gather_row(
                                        kv_l1, cmp_kv_flat, [dst_row, 0],
                                        [physical_page * CMP_STORAGE_BLOCK_SIZE, 0],
                                        [CMP_STORAGE_BLOCK_SIZE, HEAD_DIM],
                                        valid_shape=[copy_rows, HEAD_DIM],
                                    )
'''


def once(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new)


def main():
    assert not PREFIX.exists()
    old = json.loads((ROOT.parent / "hca_qqueue_groups_20260930/source.json").read_text())
    base = Path(old["sources"]["qr_late"])
    target = PREFIX / "direct"
    shutil.copytree(base, target)
    before = (base / RELATIVE).read_text()
    start = before.index("def _long_sparse_attn_hca_tp1(")
    end = before.index("\n\n@pl.jit.inline", start)
    body = before[start:end]
    gather_start = body.index("    cmp_work_kv = pl.create_tensor(")
    gather_end = body.index("    # 全部历史共用一组 MIX 任务", gather_start)
    body = body[:gather_start] + METADATA + body[gather_end:]
    body = once(body,
                "deps=[raw_gather_tid, raw_valid_tid, cmp_gather_tid, rope_cs_tid, q_ready]",
                "deps=[raw_gather_tid, raw_valid_tid, cmp_gather_tid, cmp_cache_ready_dep, rope_cs_tid, q_ready]")
    body = once(body,
                "                        cmp_row = (request * cmp_work_count + tick) * ATTN_K_TILE\n"
                "                        kv_l1 = pl.gather_row(kv_l1, cmp_work_kv, [l1_row, 0], [cmp_row, 0],\n"
                "                                             [ATTN_K_TILE, HEAD_DIM])\n", PAGE_LOAD)
    after = before[:start] + body + before[end:]
    ast.parse(after)
    path = target / RELATIVE
    path.chmod(0o644)
    path.write_text(after)
    path.chmod(0o444)
    (ROOT / "direct.patch").write_text("".join(difflib.unified_diff(
        before.splitlines(True), after.splitlines(True), n=0,
        fromfile="a/" + str(RELATIVE), tofile="b/" + str(RELATIVE),
    )))
    (ROOT / "source.json").write_text(json.dumps({
        "sources": {"base": str(base), "direct": str(target), "production": old["sources"]["production"]},
        "baseline": "Q24 with QR early-resolve disabled, same block-local softmax as production; not adopted",
        "change": "Cube directly fills its existing 3-slot L1 ring from 32-row Native pages; no KV GM staging",
        "metadata": "AIV generates exact page-valid mask and one shared zero page before cache completion",
        "invariants": ("same QK/PV tiles, block-local BF16 probabilities, "
                       "per-query reduction order, public cache layout"),
        "dependencies": "attention explicitly waits for both metadata and compressed cache writer",
        "tradeoff": "four page-table lookups and segmented loads instead of one contiguous 128-row load",
    }, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
