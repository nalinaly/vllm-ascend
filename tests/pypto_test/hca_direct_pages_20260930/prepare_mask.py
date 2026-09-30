"""页mask用Tile在左侧的算术写法，避免scalar-left sub解析限制。"""

import argparse
import ast
import difflib
import json
import shutil
from pathlib import Path

from prepare import RELATIVE, ROOT, once


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--index-cast", action="store_true")
    group.add_argument("--page-stores", action="store_true")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "source.json").read_text())
    parent = "direct_cast" if args.page_stores else "direct_mask" if args.index_cast else "direct"
    side = "direct_store" if args.page_stores else "direct_cast" if args.index_cast else "direct_mask"
    base = Path(manifest["sources"][parent])
    target = base.parent / side
    assert not target.exists()
    shutil.copytree(base, target)
    before = (base / RELATIVE).read_text()
    if args.page_stores:
        start = before.index("    # 页表/长度是本次调用输入")
        end = before.index("    # 全部历史共用一组 MIX 任务", start)
        body = before[start:end]
        body = once(body,
                    "            gather_mask = pl.tile.full([1, CMP_ATTN_K_TILE], dtype=pl.FP32, value=0.0)\n", "")
        body = once(body,
                    "            for gather_page in pl.unroll(CMP_PAGES_PER_WORK):\n",
                    "            for gather_page in pl.unroll(CMP_PAGES_PER_WORK):\n"
                    "                page_mask = pl.tile.full([1, CMP_STORAGE_BLOCK_SIZE], dtype=pl.FP32, value=0.0)\n")
        body = once(body,
                    "                        gather_mask = pl.tile.assemble(\n"
                    "                            gather_mask, page_mask, [0, gather_page * CMP_STORAGE_BLOCK_SIZE],\n"
                    "                        )\n"
                    "            pl.store(gather_mask, [gather_item, 0], cmp_work_valid)\n",
                    "                pl.store(page_mask, [gather_item, gather_page * CMP_STORAGE_BLOCK_SIZE], "
                    "cmp_work_valid)\n")
        after = before[:start] + body + before[end:]
    elif args.index_cast:
        after = once(before, "pl.cast(valid_rows, pl.FP32)",
                     "pl.cast(pl.cast(valid_rows, pl.INT32), pl.FP32)")
    else:
        after = once(before, "pl.sub(pl.cast(valid_rows, pl.FP32), page_indices)",
                     "pl.neg(pl.sub(page_indices, pl.cast(valid_rows, pl.FP32)))")
    ast.parse(after)
    path = target / RELATIVE
    path.chmod(0o644)
    path.write_text(after)
    path.chmod(0o444)
    patch = "store_fix.patch" if args.page_stores else "cast_fix.patch" if args.index_cast else "mask_fix.patch"
    (ROOT / patch).write_text("".join(difflib.unified_diff(
        before.splitlines(True), after.splitlines(True), n=0,
        fromfile="a/" + str(RELATIVE), tofile="b/" + str(RELATIVE),
    )))
    manifest["sources"][side] = str(target)
    (ROOT / "source.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
