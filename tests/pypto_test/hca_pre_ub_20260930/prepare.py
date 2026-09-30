"""冻结 mHC pre 混合值常驻 UB 候选，保持归约次序和 BF16 舍入点。"""

import ast
import difflib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
BASE = REPO.parent / ".cache/hca-flat-sync-ad0e6bbe-20260930/base"
DEST = REPO.parent / ".cache/hca-pre-ub-ad0e6bbe-20260930/mix_ub_v7"


def main():
    assert not DEST.exists(), DEST
    shutil.copytree(BASE, DEST, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    relative = Path("deepseek_v4_flash_hca/hc_pre_fused.py")
    path = DEST / relative
    before = path.read_text()
    after = before

    def replace(old, new):
        nonlocal after
        assert after.count(old) == 1, old
        after = after.replace(old, new)

    replace("    x_mixed = pl.create_tensor([t_pad, D], dtype=pl.BF16)\n", "")
    replace("        pre_tile = pre_val_store[t0:t0 + T_TILE, 0:HC_PAD]\n",
            "        pre_tile = pl.load(pre_val_store, [t0, 0], [T_TILE, HC_PAD])\n")
    for lane, col in enumerate(("d0", "D + d0", "2 * D + d0", "3 * D + d0")):
        replace(
            f"            x{lane} = pl.slice(x_flat, [T_TILE, D_TILE], [t0, {col}], "
            "valid_shape=[valid_rows, D_TILE])\n",
            f"            x{lane} = pl.load(x_flat, [t0, {col}], [T_TILE, D_TILE], "
            "valid_shape=[valid_rows, D_TILE])\n",
        )
    replace(
        "        sq_sum = pl.full([1, T_TILE], dtype=pl.FP32, value=0.0)\n",
        "        # 保留已量化的混合值，后续 RMS 不再经过 GM。\n"
        "        mixed_ub = pl.create_tile([D // D_TILE * T_TILE, D_TILE], dtype=pl.BF16)\n"
        "        sq_sum = pl.tile.full([1, T_TILE], dtype=pl.FP32, value=0.0)\n",
    )
    replace(
        "            x_mixed[t0:t0 + T_TILE, d0:d0 + D_TILE] = y_bf16\n",
        "            mixed_ub = pl.tile.assemble(mixed_ub, y_bf16, [mix_db * T_TILE, 0])\n",
    )
    replace("            y_sq_sum = pl.row_sum(y_sq)\n",
            "            sum_tmp = pl.create_tile([T_TILE, D_TILE], dtype=pl.FP32)\n"
            "            y_sq_sum = pl.row_sum(y_sq, sum_tmp)\n")
    start = after.index("        norm_sq_sum = pl.create_tensor([1, T_TILE], dtype=pl.FP32)")
    end = after.index("            if valid_rows == T_TILE:\n", start)
    after = after[:start] + """        # 与 rms_norm_inverse/apply 相同的运算与高精度 rsqrt，统计量留在 UB。
        mixed_variance = pl.add(pl.mul(sq_sum, 1.0 / D), NORM_EPS)
        inverse_tmp = pl.create_tile([1, T_TILE], dtype=pl.FP32)
        inverse_col = pl.reshape(pl.tile.rsqrt(mixed_variance, inverse_tmp), [T_TILE, 1])
        for norm_db in pl.pipeline(D // D_TILE, stage=2):
            d0 = norm_db * D_TILE
            mixed_input = pl.tile.extract(
                mixed_ub, norm_db * T_TILE, 0, [T_TILE, D_TILE], target_memory=pl.MemorySpace.Vec,
            )
            norm_w_input = pl.load(norm_w, [d0], [D_TILE])
            norm_w_row = pl.cast(pl.reshape(norm_w_input, [1, D_TILE]), pl.FP32)
            mixed_fp32 = pl.cast(mixed_input, pl.FP32)
            scaled = pl.row_expand_mul(mixed_fp32, inverse_col)
            normed_bf16 = pl.cast(pl.col_expand_mul(scaled, norm_w_row), pl.BF16, mode="rint")
""" + after[end:]
    replace("                x_normed[t0:t0 + T_TILE, d0:d0 + D_TILE] = normed_bf16\n",
            "                pl.store(normed_bf16, [t0, d0], x_normed)\n")
    replace("                x_normed_tail_store[0:T_TILE, d0:d0 + D_TILE] = normed_bf16\n",
            "                pl.store(normed_bf16, [0, d0], x_normed_tail_store)\n")
    ast.parse(after)
    path.chmod(0o644)
    path.write_text(after)
    for source in DEST.rglob("*.py"):
        source.chmod(0o444)
    (ROOT / "mix_ub.patch").write_text("".join(difflib.unified_diff(
        before.splitlines(True), after.splitlines(True),
        fromfile="a/" + str(relative), tofile="b/" + str(relative), n=0,
    )))
    (ROOT / "source.json").write_text(json.dumps({
        "baseline_commit": "ad0e6bbe", "baseline": str(BASE), "candidate": str(DEST),
        "change": "8x4096 BF16混合值按16个8x256连续块及RMS统计量常驻UB，消除两段循环间GM存取",
        "invariants": "相同8行分工、stage2、split0到3及D256归约顺序、BF16舍入、tail保护、依赖",
        "upstream": "pypto-lib2164563 hc_pre_norm仍存取GM；保留其算术，改变中间值存放位置",
    }, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
