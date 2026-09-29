"""Indexer query满行RoPE整块Gather；保持尾行、算术与任务依赖。"""

import ast
import difflib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[4]
BASE = WORKSPACE / ".cache/csa-sparse-rope-flat-gather-369ad2c1-v1-candidate"
PREFIX = WORKSPACE / ".cache/csa-indexer-rope-flat-gather-632dd00a-v1"
OLD_PACKAGE = "dsv4_csa_sparse_rope_flat_gather_369ad2c1_v1"
PACKAGE = "dsv4_csa_indexer_rope_flat_gather_632dd00a_v1"


def change(before):
    start = before.index("                qr_scale_tile = qr_scale[dq_t0 : dq_t0 + DEQUANT_T_TILE, :]")
    end = before.index("            else:\n                # At most seven rows.", start)
    replacement = """                qr_scale_tile = pl.load(qr_scale, [dq_t0, 0], [DEQUANT_T_TILE, 1])
                cos_tile = pl.load(cos, [dq_t0, 0], [DEQUANT_T_TILE, ROPE_HEAD_DIM])
                sin_tile = pl.load(sin, [dq_t0, 0], [DEQUANT_T_TILE, ROPE_HEAD_DIM])
                # The RoPE slice has a 128-element row stride. Gather from the
                # contiguous full head using absolute element indices.
                flat_i = pl.tile.ci(0, [1, ROPE_HEAD_DIM], dtype=pl.INT32)
                flat_tmp = pl.create_tile([1, ROPE_HEAD_DIM], dtype=pl.INT32)
                flat_lane = pl.tile.rems(flat_i, 2, flat_tmp)
                swap_row = pl.tile.adds(pl.tile.sub(flat_i, pl.tile.muls(flat_lane, 2)), IDX_NOPE_HEAD_DIM + 1)
                swap_base = pl.create_tile([DEQUANT_T_TILE, ROPE_HEAD_DIM], dtype=pl.INT32)
                swap_source = pl.col_expand(swap_base, swap_row)
                row_offsets = pl.tile.muls(pl.tile.ci(0, [1, DEQUANT_T_TILE], dtype=pl.INT32), IDX_HEAD_DIM)
                flat_swap = pl.reshape(
                    pl.row_expand_add(swap_source, pl.reshape(row_offsets, [DEQUANT_T_TILE, 1])),
                    [1, DEQUANT_T_TILE * ROPE_HEAD_DIM],
                )
                gather_tmp = pl.create_tile([1, DEQUANT_T_TILE * ROPE_HEAD_DIM], dtype=pl.INT32)
                for h_inner in pl.pipeline(DQ_ROPE_H_TILE, stage=2):
                    h0 = (hg + h_inner) * IDX_HEAD_DIM
                    wq_scale = pl.reshape(pl.load(wq_b_scale, [h0], [IDX_HEAD_DIM]), [1, IDX_HEAD_DIM])
                    acc_fp32 = pl.cast(
                        pl.load(qr_acc_pad, [dq_t0, h0], [DEQUANT_T_TILE, IDX_HEAD_DIM]),
                        target_type=pl.FP32,
                        mode="none",
                    )
                    qr_dequant = pl.col_expand_mul(pl.row_expand_mul(acc_fp32, qr_scale_tile), wq_scale)
                    qr_nope_bf16 = pl.cast(qr_dequant[:, 0:IDX_NOPE_HEAD_DIM], target_type=pl.BF16, mode="rint")
                    qr_rope_slice = qr_dequant[:, IDX_NOPE_HEAD_DIM:IDX_HEAD_DIM]
                    qr_swapped_flat = pl.tile.gather(
                        pl.reshape(qr_dequant, [1, DEQUANT_T_TILE * IDX_HEAD_DIM]),
                        flat_swap, gather_tmp,
                    )
                    qr_swapped = pl.reshape(qr_swapped_flat, [DEQUANT_T_TILE, ROPE_HEAD_DIM])
                    rope_rot = pl.add(pl.mul(qr_rope_slice, cos_tile), pl.mul(qr_swapped, sin_tile))
                    rope_bf16 = pl.cast(rope_rot, target_type=pl.BF16, mode="rint")
                    pl.store(qr_nope_bf16, [dq_t0, h0], qr_bf16_2d)
                    pl.store(rope_bf16, [dq_t0, h0 + IDX_NOPE_HEAD_DIM], qr_bf16_2d)
"""
    result = before[:start] + replacement + before[end:]
    ast.parse(result)
    return result


def main():
    assert not (ROOT / "task.txt").exists()
    for side in ("baseline", "candidate"):
        dest = Path(str(PREFIX) + "-" + side)
        assert not dest.exists(), dest
        shutil.copytree(BASE, dest, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "build_output"))
        packages = dest / "vllm_ascend/ops/pypto"
        (packages / OLD_PACKAGE).rename(packages / PACKAGE)
    relative = Path("vllm_ascend/ops/pypto") / PACKAGE / "decode_indexer.py"
    path = Path(str(PREFIX) + "-candidate") / relative
    before = path.read_text()
    after = change(before)
    path.chmod(0o644)
    path.write_text(after)
    (ROOT / "candidate.patch").write_text("".join(difflib.unified_diff(
        before.splitlines(True), after.splitlines(True), fromfile="a/" + str(relative), tofile="b/" + str(relative))))
    template = ROOT.parent / "csa_sparse_rope_flat_gather_20260929"
    for name in ("compile.py", "run.sh", "run_side.sh"):
        text = (template / name).read_text().replace(template.name, ROOT.name)
        text = text.replace("csa-sparse-rope-flat-gather-369ad2c1-v1", PREFIX.name).replace(OLD_PACKAGE, PACKAGE)
        (ROOT / name).write_text(text)
    (ROOT / "source.json").write_text(json.dumps({
        "baseline": "632dd00a", "base_source": str(BASE), "source_prefix": str(PREFIX),
        "variant": "pkg:" + PACKAGE, "cases": [[131072, 16], [8192, 24]],
        "change": "仅Indexer query满行反量化/RoPE的8x64逐行Gather展平1x512；尾行路径保持",
        "native_reference": ("ops-transformer28f40354/posembedding/rotary_position_embedding/"
                             "op_kernel/rotate_interleaved_split_bsn_pad.h:261,295"),
        "not_changed": "乘法顺序、BF16舍入、head/worker分工、Hadamard/量化、Score、依赖/early、精度版/工具链",
        "acceptance": ("CPU确认取消满行逐行TMOV后，两代表档核内/CSA/P95；性能后完整八类状态，"
                       "有收益再H4095/B3真实筛选+尾行/padding"),
    }, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
