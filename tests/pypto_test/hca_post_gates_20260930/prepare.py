"""mHC post按列整理门控，消除每个输出/输入组合的索引构造和Gather。"""

import ast
import difflib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
PREFIX = REPO.parent / ".cache/hca-post-gates-f14d8d90-20260930"


def main():
    assert not PREFIX.exists()
    base = PREFIX / "base"
    candidate = PREFIX / "transpose"
    shutil.copytree(REPO / "vllm_ascend/ops/pypto", base, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copytree(base, candidate)
    relative = Path("deepseek_v4_flash_hca/o_proj_hc_post.py")
    file = candidate / relative
    before = file.read_text()
    after = before.replace(
        "        row_ids = pl.cast(pl.tile.arange(0, [1, TOKEN_TILE], dtype=pl.INT32), pl.FP32)\n"
        "        gather_tmp = pl.create_tile([1, TOKEN_TILE], dtype=pl.INT32)\n",
        "",
    )
    start = after.index("            for out_h in pl.unroll(HC_MULT):")
    end = after.index('                result = pl.cast(value, pl.BF16, mode="rint")', start)
    block = """            post_columns = pl.transpose(post_rows, axis1=0, axis2=1)
            comb_columns = pl.transpose(comb_rows, axis1=0, axis2=1)
            for out_h in pl.unroll(HC_MULT):
                post_weight = pl.reshape(
                    pl.tile.slice(post_columns, [1, TOKEN_TILE], [out_h, 0]), [TOKEN_TILE, 1],
                )
                value = pl.row_expand_mul(attention, post_weight)
"""
    for group in range(4):
        block += f"""                coefficient_{group} = pl.reshape(
                    pl.tile.slice(comb_columns, [1, TOKEN_TILE], [{group} * HC_MULT + out_h, 0]),
                    [TOKEN_TILE, 1],
                )
                value = pl.add(value, pl.row_expand_mul(residual_{group}, coefficient_{group}))
"""
    after = after[:start] + block + after[end:]
    ast.parse(after)
    file.write_text(after)
    for py in PREFIX.rglob("*.py"):
        py.chmod(0o444)
    (ROOT / "transpose.patch").write_text(
        "".join(
            difflib.unified_diff(
                before.splitlines(True),
                after.splitlines(True),
                fromfile="a/" + str(relative),
                tofile="b/" + str(relative),
                n=0,
            )
        )
    )
    manifest = {
        "baseline_commit": "f14d8d90 (production operator unchanged from 9015d45a)",
        "sources": {"base": str(base), "transpose": str(candidate)},
        "reference": "ops-transformer28f40354 mhc/mhc_post/op_kernel/arch22/mhc_post_arch22.h:203-238",
        "change": "two UB transposes replace twenty coefficient gathers per eight token rows",
        "invariants": (
            "task partition, INT32 dequant reduction order, BF16 attention rounding, post sum order unchanged"
        ),
    }
    (ROOT / "source.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
