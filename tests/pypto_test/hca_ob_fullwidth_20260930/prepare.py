"""从0742f07c冻结整包，生成整宽量化的分组参考与完整K候选。"""

import ast
import difflib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
PREFIX = REPO.parent / ".cache/hca-ob-fullwidth-0742f07c-20260930-v7"


def main():
    assert not PREFIX.exists()
    base = PREFIX / "base"
    shutil.copytree(REPO / "vllm_ascend/ops/pypto", base, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    relative = Path("deepseek_v4_flash_hca/o_proj_hc_post.py")
    before = (base / relative).read_text()
    sources = {"base": str(base)}
    for side in ("grouped", "full"):
        dest = PREFIX / side
        shutil.copytree(base, dest)
        (dest / "deepseek_v4_flash_hca/whole_o_projection.py").write_text((ROOT / "whole_o_projection.py").read_text())
        after = before.replace("    _decode_o_proj_tp1_parts,\n", "")
        after = after.replace(
            "T_DYN = pl.dynamic",
            f"from .whole_o_projection import o_parts_{side} as _decode_o_proj_tp1_parts\n\nT_DYN = pl.dynamic",
        )
        start = after.index("            acc = pl.tile.full")
        end = after.index("            # 保留原先 attention", start)
        if side == "grouped":
            block = '''            acc = pl.tile.full([TOKEN_TILE, COL_TILE], dtype=pl.INT32, value=0)
            for group in pl.pipeline(O_GROUPS, stage=2):
                part = pl.load(partials, [row, group * D + col0], [TOKEN_TILE, COL_TILE])
                acc = pl.add(acc, part)
'''
        else:
            block = "            acc = pl.load(partials, [row, col0], [TOKEN_TILE, COL_TILE])\n"
        block += '''            scale = pl.reshape(pl.load(scales, [0, row], [1, TOKEN_TILE]), [TOKEN_TILE, 1])
            attention = pl.row_expand_mul(pl.col_expand_mul(pl.cast(acc, pl.FP32), weight_scale), scale)
'''
        after = after[:start] + block + after[end:]
        ast.parse(after)
        (dest / relative).write_text(after)
        sources[side] = str(dest)
        (ROOT / f"{side}.patch").write_text("".join(difflib.unified_diff(
            before.splitlines(True), after.splitlines(True), fromfile="a/" + str(relative), tofile="b/" + str(relative),
        )))
    for file in PREFIX.rglob("*.py"):
        file.chmod(0o444)
    (ROOT / "source.json").write_text(json.dumps({
        "baseline_commit": "0742f07c", "sources": sources,
        "arithmetic": ("BF16 O-A output; whole8192 token amax; channel scale then token scale; "
                       "grouped reference sums INT32"),
        "candidate": "full K8192 INT32 matmul M32/N256/K512 stage2, no eight-part output reduction",
        "limits": "different arithmetic from performance baseline; state exact and separate grouped reference required",
    }, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
