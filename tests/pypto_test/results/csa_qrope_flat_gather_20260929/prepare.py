"""按AscendC整块Gather表达RoPE置换；保持逐head算术和48-worker。"""

import ast
import difflib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[4]
BASE = WORKSPACE / ".cache/csa-indexer-early-chain-7b296153-candidate"
TEMPLATE = ROOT.parent / "csa_indexer_early_chain_20260929"
OLD_PACKAGE = "dsv4_csa_indexer_early_chain_7b296153"
PACKAGE = "dsv4_csa_qrope_flat_gather_f4861832_v2"
PREFIX = WORKSPACE / ".cache/csa-qrope-flat-gather-f4861832-v2"


def change(before):
    start = before.index("                qr_scale_dq_t = qr_scale_pad_store[tg : tg + Q_ROPE_T_TILE, :]")
    end = before.index("            else:\n                valid_tail_rows = tile_rows - tg", start)
    replacement = '''                qr_scale_dq_t = pl.load(
                    qr_scale_pad_store, [tg, 0], [Q_ROPE_T_TILE, 1], target_memory=pl.MemorySpace.Vec,
                )
                q_cos_il = pl.load(
                    rope_cos_il, [out_tg, 0], [Q_ROPE_T_TILE, ROPE_DIM], target_memory=pl.MemorySpace.Vec,
                )
                q_sin_signed = pl.load(
                    rope_sin_signed, [out_tg, 0], [Q_ROPE_T_TILE, ROPE_DIM], target_memory=pl.MemorySpace.Vec,
                )
                q_swap_idx = pl.load(
                    rope_swap_idx, [out_tg, 0], [Q_ROPE_T_TILE, ROPE_DIM], target_memory=pl.MemorySpace.Vec,
                )
                # Preserve the caller's within-row indices; add row starts once per work tile.
                q_flat_i = pl.cast(
                    pl.tile.arange(0, [1, Q_ROPE_T_TILE * ROPE_DIM], dtype=pl.INT32), pl.FP32,
                )
                q_row_i = pl.cast(pl.mul(q_flat_i, 1.0 / ROPE_DIM), pl.INT32, mode="trunc")
                q_row_starts = pl.mul(pl.cast(q_row_i, pl.FP32), ROPE_DIM_SCALE)
                q_abs_swap = pl.cast(pl.add(
                    pl.cast(pl.reshape(q_swap_idx, [1, Q_ROPE_T_TILE * ROPE_DIM]), pl.FP32), q_row_starts,
                ), pl.INT32)
                q_flat_gather_tmp = pl.tile.create([1, Q_ROPE_T_TILE * ROPE_DIM], dtype=pl.INT32)
                for h_inner in pl.pipeline(dq_head_tile, stage=2):
                    h = hg + h_inner
                    h0 = h * HEAD_DIM
                    q_head_acc = pl.load(
                        q_proj_i32, [tg, h0], [Q_ROPE_T_TILE, HEAD_DIM], target_memory=pl.MemorySpace.Vec,
                    )
                    q_head_scale = pl.reshape(pl.load(
                        wq_b_scale, [h0], [HEAD_DIM], target_memory=pl.MemorySpace.Vec,
                    ), [1, HEAD_DIM])
                    q_head_acc_fp32 = pl.cast(q_head_acc, target_type=pl.FP32, mode="none")
                    q_head_row_scaled = pl.row_expand_mul(q_head_acc_fp32, qr_scale_dq_t)
                    q_head_dq = pl.col_expand_mul(q_head_row_scaled, q_head_scale)
                    q_head_sq = pl.mul(q_head_dq, q_head_dq)
                    q_head_sq_row = pl.row_sum(
                        q_head_sq, tmp_tile=pl.tile.create([Q_ROPE_T_TILE, HEAD_DIM], dtype=pl.FP32),
                    )
                    q_head_sq_sum = pl.reshape(q_head_sq_row, [1, Q_ROPE_T_TILE])
                    q_head_sq_mean = pl.mul(q_head_sq_sum, 1.0 / HEAD_DIM)
                    q_head_var = pl.add(q_head_sq_mean, EPS)
                    q_head_inv_rms = pl.tile.rsqrt(
                        q_head_var, pl.tile.create([1, Q_ROPE_T_TILE], dtype=pl.FP32),
                    )
                    q_head_inv_rms_t = pl.reshape(q_head_inv_rms, [Q_ROPE_T_TILE, 1])
                    q_nope_normed = pl.row_expand_mul(q_head_dq[:, 0:NOPE_DIM], q_head_inv_rms_t)
                    q_nope_bf16 = pl.cast(q_nope_normed, target_type=pl.BF16, mode="rint")
                    pl.store(q_nope_bf16, [out_tg, h0], q_flat)
                    q_rope_chunk_raw = q_head_dq[:, NOPE_DIM:HEAD_DIM]
                    q_rope_chunk = pl.row_expand_mul(q_rope_chunk_raw, q_head_inv_rms_t)
                    q_rope_flat = pl.reshape(q_rope_chunk, [1, Q_ROPE_T_TILE * ROPE_DIM])
                    q_swapped_flat = pl.tile.gather(q_rope_flat, q_abs_swap, q_flat_gather_tmp)
                    q_rope_swapped = pl.reshape(q_swapped_flat, [Q_ROPE_T_TILE, ROPE_DIM])
                    q_rope_base = pl.mul(q_rope_chunk, q_cos_il)
                    q_rope_delta = pl.mul(q_rope_swapped, q_sin_signed)
                    q_rope_rot = pl.add(q_rope_base, q_rope_delta)
                    q_rope_bf16 = pl.cast(q_rope_rot, target_type=pl.BF16, mode="rint")
                    pl.store(q_rope_bf16, [out_tg, h0 + NOPE_DIM], q_flat)
'''
    after = before[:start] + replacement + before[end:]
    ast.parse(after)
    return after


def main():
    if (ROOT / "task.txt").exists():
        raise RuntimeError("Submitted sources must stay immutable")
    for side in ("baseline", "candidate"):
        target = Path(str(PREFIX) + "-" + side)
        if target.exists():
            raise RuntimeError(f"Existing snapshot: {target}")
        shutil.copytree(BASE, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "build_output"))
        packages = target / "vllm_ascend/ops/pypto"
        (packages / OLD_PACKAGE).rename(packages / PACKAGE)
    relative = Path("vllm_ascend/ops/pypto") / PACKAGE / "qkv_proj_rope.py"
    before = (Path(str(PREFIX) + "-baseline") / relative).read_text()
    after = change(before)
    target = Path(str(PREFIX) + "-candidate") / relative
    target.chmod(0o644)
    target.write_text(after)
    (ROOT / "candidate.patch").write_text("".join(difflib.unified_diff(
        before.splitlines(True), after.splitlines(True), fromfile="a/" + str(relative), tofile="b/" + str(relative))))
    for name in ("compile.py", "run.sh", "run_side.sh"):
        text = (TEMPLATE / name).read_text().replace(TEMPLATE.name, ROOT.name)
        text = text.replace("csa-indexer-early-chain-7b296153", PREFIX.name).replace(OLD_PACKAGE, PACKAGE)
        (ROOT / name).write_text(text)
    source = {"baseline": "f4861832", "source_prefix": str(PREFIX), "base_source": str(BASE),
              "variant": "pkg:" + PACKAGE, "cases": [[131072, 16], [8192, 24]],
              "change": "满行RoPE先展平8×64，绝对元素索引单次tile.gather；保持逐head和48-worker",
              "reference": ("ops-transformer28f40354/rotary_position_embedding/"
                            "rotate_interleaved_split_bsn_pad.h:Compute"),
              "not_changed": "Q_B24、量化顺序/逐head RMS/舍入、依赖/early、尾行、精度版/工具链",
              "scope": "与在跑双head方案独立；先确认展平gather生成单行512元素指令且无逐行TMOV，再两档验证"}
    (ROOT / "source.json").write_text(json.dumps(source, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
