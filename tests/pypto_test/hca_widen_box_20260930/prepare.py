"""扩大widen/RMS的有效并行度，物理M8保持A3列向量对齐和固定归约顺序。"""

import ast
import difflib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
BASE = REPO.parent / ".cache/hca-mix-schedule-9c2ede34-20260930/base"
PREFIX = REPO.parent / ".cache/hca-widen-box-9c2ede34-20260930"


def main():
    assert not PREFIX.exists(), PREFIX
    sources = {"base": str(BASE)}
    for side, width, stage in (("m4_k512", 512, 4), ("m4_k1024", 1024, 2)):
        destination = PREFIX / side
        shutil.copytree(BASE, destination, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        sources[side] = str(destination)
        relative = Path("deepseek_v4_flash_hca/decode_hca.py")
        path = destination / relative
        before = path.read_text()
        after = before.replace("WIDEN_ROWS = 8", "WIDEN_ROWS = 8\nWIDEN_ACTIVE_ROWS = 4\n"
                               f"WIDEN_LOAD_COLS = {width}\nWIDEN_PIPE_STAGE = {stage}")
        after = after.replace("widen_blocks = (tokens + WIDEN_ROWS - 1) // WIDEN_ROWS",
                              "widen_blocks = (tokens + WIDEN_ACTIVE_ROWS - 1) // WIDEN_ACTIVE_ROWS")
        after = after.replace("    tail = pl.create_tensor([WIDEN_ROWS, HC_DIM], dtype=pl.FP32)\n", "")
        start = after.index("            row = block * WIDEN_ROWS")
        end = after.index("    # 冷 L2", start)
        body = """            row = block * WIDEN_ACTIVE_ROWS
            count = pl.min(WIDEN_ACTIVE_ROWS, tokens - row)
            sq_sum = pl.tile.full([1, WIDEN_ROWS], dtype=pl.FP32, value=0.0)
            for col_block in pl.pipeline(HC_DIM // WIDEN_LOAD_COLS, stage=WIDEN_PIPE_STAGE):
                col = col_block * WIDEN_LOAD_COLS
                source = pl.load(x_flat, [row, col], [WIDEN_ROWS, WIDEN_LOAD_COLS],
                                 valid_shape=[count, WIDEN_LOAD_COLS])
                value = pl.cast(source, pl.FP32)
                # 只写本worker有效行，不能让两个4行worker的8行物理盒互相覆盖。
                valid_value = pl.set_validshape(value, count, WIDEN_LOAD_COLS)
                pl.store(valid_value, [row, col], x32_flat)
                clean = pl.fillpad(valid_value, pad_value=pl.PadValue.zero)
                squared = pl.mul(clean, clean)
                sum_tmp = pl.create_tile([WIDEN_ROWS, RMS_COLS], dtype=pl.FP32)
"""
        if width == 512:
            body += """                chunk_sum = pl.row_sum(squared, sum_tmp)
                sq_sum = pl.add(sq_sum, pl.reshape(chunk_sum, [1, WIDEN_ROWS]))
"""
        else:
            body += """                # 宽搬运仍按原来的两段512列顺序累加，不改成1024列整块归约。
                square_left = pl.tile.extract(
                    squared, 0, 0, [WIDEN_ROWS, RMS_COLS], target_memory=pl.MemorySpace.Vec,
                )
                left_sum = pl.row_sum(square_left, sum_tmp)
                sq_sum = pl.add(sq_sum, pl.reshape(left_sum, [1, WIDEN_ROWS]))
                square_right = pl.tile.extract(
                    squared, 0, RMS_COLS, [WIDEN_ROWS, RMS_COLS], target_memory=pl.MemorySpace.Vec,
                )
                right_sum = pl.row_sum(square_right, sum_tmp)
                sq_sum = pl.add(sq_sum, pl.reshape(right_sum, [1, WIDEN_ROWS]))
"""
        body += """            mean = pl.add(pl.mul(sq_sum, 1.0 / HC_DIM), NORM_EPS)
            inverse_tmp = pl.create_tile([1, WIDEN_ROWS], dtype=pl.FP32)
            inverse = pl.reshape(pl.tile.rsqrt(mean, inverse_tmp), [WIDEN_ROWS, 1])
            pl.store(pl.set_validshape(inverse, count, 1), [row, 0], inv_rms)
"""
        after = after[:start] + body + after[end:]
        ast.parse(after)
        path.chmod(0o644)
        path.write_text(after)
        for source in destination.rglob("*.py"):
            source.chmod(0o444)
        (ROOT / f"{side}.patch").write_text("".join(difflib.unified_diff(
            before.splitlines(True), after.splitlines(True),
            fromfile="a/" + str(relative), tofile="b/" + str(relative), n=0,
        )))
    (ROOT / "source.json").write_text(json.dumps({
        "baseline_commit": "9c2ede34", "sources": sources,
        "change": "widen physical M8/active M4; K512 stage4 versus K1024 stage2 with ordered K512 sums",
        "invariants": (
            "BF16-to-FP32 widening, RMS K512 order, high-precision rsqrt, downstream tensor layout and task deps"
        ),
        "native_reference": (
            "ops-transformer28f40354 MhcPreSinkhornPremixStage1: vector cast plus ordered chunk statistics"
        ),
    }, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
