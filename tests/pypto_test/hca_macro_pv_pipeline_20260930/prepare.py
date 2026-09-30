"""在已获益的QK双缓冲上继续给PV微块增加局部加载及两级流水。"""

import ast
import difflib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PREFIX = ROOT.parents[3] / ".cache/hca-macro-pvpipe-070d081a-20260930"
RELATIVE = Path("deepseek_v4_flash_hca/decode_sparse_attn_hca.py")


def once(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new)


def main():
    assert not PREFIX.exists()
    old = json.loads((ROOT.parent / "hca_macro512_20260930/source.json").read_text())
    base = Path(old["sources"]["macro512_qkpipe"])
    target = PREFIX / "pvpipe"
    shutil.copytree(base, target)
    before = (base / RELATIVE).read_text()
    start = before.index("    # 全部历史共用一组 MIX 任务", before.index("def _long_sparse_attn_hca_tp1("))
    end = before.index("        for lane in pl.split_aiv(2, mode=pl.SplitMode.NONE):", start)
    body = before[start:end]
    body = once(body,
                "        pv_kv = pl.create_tile([ATTN_K_TILE, HEAD_DIM], dtype=pl.BF16, "
                "target_memory=pl.MemorySpace.Mat)\n", "")
    first = body.index("                    if prev_block < 0:")
    last = body.index("                    probability = pl.load(probs", first)
    body = body[:first] + '''                    if prev_block < 0:
                        first_kv = pl.yield_(pl.load(
                            raw_kv, [prev_request * REQUEST_KV_ROWS + prev_drop, 0],
                            [ATTN_K_TILE, HEAD_DIM], target_memory=pl.MemorySpace.Mat,
                        ))
                    else:
                        pv_source = (prev_request * cmp_work_count + prev_block * MACRO_PARTS) * ATTN_K_TILE
                        first_kv = pl.yield_(pl.load(
                            cmp_work_kv, [pv_source, 0], [ATTN_K_TILE, HEAD_DIM],
                            target_memory=pl.MemorySpace.Mat,
                        ))
''' + body[last:]
    for col in range(4):
        body = once(body, f"right{col} = pl.tile.extract(pv_kv,", f"right{col} = pl.tile.extract(first_kv,")
    body = once(body,
                "                    for part, (acc0, acc1, acc2, acc3) in pl.range(\n"
                "                        1, prev_parts, init_values=(first0, first1, first2, first3),",
                "                    for part, (acc0, acc1, acc2, acc3) in pl.pipeline(\n"
                "                        1, prev_parts, stage=2, init_values=(first0, first1, first2, first3),")
    body = once(body,
                "                        pv_kv = pl.gather_row(pv_kv, cmp_work_kv, [0, 0], "
                "[source_row, 0], [ATTN_K_TILE, HEAD_DIM])",
                "                        part_kv = pl.load(\n"
                "                            cmp_work_kv, [source_row, 0], [ATTN_K_TILE, HEAD_DIM],\n"
                "                            target_memory=pl.MemorySpace.Mat,\n"
                "                        )")
    for col in range(4):
        body = once(body, f"right_update{col} = pl.tile.extract(pv_kv,",
                    f"right_update{col} = pl.tile.extract(part_kv,")
    after = before[:start] + body + before[end:]
    ast.parse(after)
    path = target / RELATIVE
    path.chmod(0o644)
    path.write_text(after)
    path.chmod(0o444)
    (ROOT / "pvpipe.patch").write_text("".join(difflib.unified_diff(
        before.splitlines(True), after.splitlines(True), n=0,
        fromfile="a/" + str(RELATIVE), tofile="b/" + str(RELATIVE),
    )))
    (ROOT / "source.json").write_text(json.dumps({
        "sources": {"base": str(base), "pvpipe": str(target), "production": old["sources"]["production"]},
        "baseline": "unadopted 512-column macro with QK stage2 pipeline, not production",
        "change": "PV K128 blocks use local Mat loads and stage2 pipeline with four accumulator carries",
        "invariants": "same QK stage2, four L0C accumulation order, BF16 softmax, raw block and query stream",
        "risks_to_check": "zero-iteration raw PV loop, odd K-block tail, no L1/L0C double-buffer overflow",
    }, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
