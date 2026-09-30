"""冻结算子包；按Native A3 permanent-X思路复用四份FP32 residual。"""

import ast
import difflib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
PREFIX = REPO.parent / ".cache/hca-residual-c6792787-20260930"
RELATIVE = Path("deepseek_v4_flash_hca/o_proj_hc_post.py")


def main():
    baseline = PREFIX / "base"
    candidate = PREFIX / "reuse"
    assert not PREFIX.exists(), PREFIX
    shutil.copytree(
        REPO / "vllm_ascend/ops/pypto",
        baseline,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    shutil.copytree(baseline, candidate)
    path = candidate / RELATIVE
    before = path.read_text()
    old = "            for out_h in pl.unroll(HC_MULT):\n"
    assert before.count(old) == 1
    loads = "            # 四个输入HC分量只加载、转FP32一次，供四个输出顺序复用。\n"
    for index in range(4):
        loads += (
            f"            residual_{index} = pl.cast(pl.load(\n"
            f"                residual_flat, [row, {index} * D + col0], [TOKEN_TILE, COL_TILE],\n"
            "                valid_shape=[valid, COL_TILE],\n"
            "            ), pl.FP32)\n"
        )
    text = before.replace(old, loads + old, 1)
    start = text.index("                for in_h in pl.pipeline(HC_MULT, stage=4):\n")
    end = text.index('                result = pl.cast(value, pl.BF16, mode="rint")', start)
    operations = ""
    for index in range(4):
        operations += (
            f"                comb_index_{index} = pl.cast(pl.add(\n"
            "                    pl.mul(row_ids, 16.0), "
            f"pl.cast(pl.cast({index} * HC_MULT + out_h, pl.INT32), pl.FP32),\n"
            "                ), pl.INT32)\n"
            f"                coefficient_{index} = pl.reshape(\n"
            f"                    pl.tile.gather(comb_rows, comb_index_{index}, gather_tmp), [TOKEN_TILE, 1],\n"
            "                )\n"
            f"                value = pl.add(value, pl.row_expand_mul(residual_{index}, coefficient_{index}))\n"
        )
    text = text[:start] + operations + text[end:]
    ast.parse(text)
    path.write_text(text)
    (ROOT / "reuse.patch").write_text(
        "".join(
            difflib.unified_diff(
                before.splitlines(True),
                text.splitlines(True),
                fromfile="a/" + str(RELATIVE),
                tofile="b/" + str(RELATIVE),
            )
        )
    )
    for folder in (baseline, candidate):
        for file in folder.rglob("*.py"):
            file.chmod(0o444)
    manifest = {
        "baseline_commit": "c6792787",
        "baseline": str(baseline),
        "candidate": str(candidate),
        "source_reference": "ops-transformer/mhc/mhc_post/op_kernel/arch22/mhc_post_arch22.h USE_PERMANENT_X",
        "change": "four residual loads/casts per row tile instead of sixteen; same output accumulation and rounding",
        "limits": "extra UB residency must fit; no performance or numerical claim until device validation",
    }
    (ROOT / "source.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
