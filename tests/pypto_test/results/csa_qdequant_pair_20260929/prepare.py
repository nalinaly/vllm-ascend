"""按最新AscendC多行处理思路冻结Q反量化双head核内候选，保留48-worker及尾行。"""

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
PACKAGE = "dsv4_csa_qdequant_pair_f4861832_v2"
PREFIX = WORKSPACE / ".cache/csa-qdequant-pair-f4861832-v2"


def change(before):
    start = before.index("                for h_inner in pl.pipeline(dq_head_tile, stage=2):")
    end = before.index("            else:\n                valid_tail_rows = tile_rows - tg", start)
    replacement = '''                # Two adjacent heads share one wide dequant vector operation.
                # Flatten only after dequant: every RMS row still has HEAD_DIM columns.
                q_pair_cos = pl.reshape(pl.concat(q_cos_il, q_cos_il), [Q_ROPE_T_TILE * 2, ROPE_DIM])
                q_pair_sin = pl.reshape(pl.concat(q_sin_signed, q_sin_signed), [Q_ROPE_T_TILE * 2, ROPE_DIM])
                q_pair_swap = pl.reshape(pl.concat(q_swap_idx, q_swap_idx), [Q_ROPE_T_TILE * 2, ROPE_DIM])
                for h_inner in pl.pipeline(0, dq_head_tile, 2, stage=2):
                    h0 = (hg + h_inner) * HEAD_DIM
                    q_pair_acc = q_proj_i32[tg : tg + Q_ROPE_T_TILE, h0 : h0 + HEAD_DIM * 2]
                    q_pair_scale = pl.reshape(wq_b_scale[h0 : h0 + HEAD_DIM * 2], [1, HEAD_DIM * 2])
                    q_pair_fp32 = pl.cast(q_pair_acc, target_type=pl.FP32, mode="none")
                    q_pair_row_scaled = pl.row_expand_mul(q_pair_fp32, qr_scale_dq_t)
                    q_pair_dq_wide = pl.col_expand_mul(q_pair_row_scaled, q_pair_scale)
                    q_pair_dq = pl.reshape(q_pair_dq_wide, [Q_ROPE_T_TILE * 2, HEAD_DIM])
                    q_pair_sq = pl.mul(q_pair_dq, q_pair_dq)
                    q_pair_sq_sum = pl.reshape(pl.row_sum(q_pair_sq), [1, Q_ROPE_T_TILE * 2])
                    q_pair_var = pl.add(pl.mul(q_pair_sq_sum, 1.0 / HEAD_DIM), EPS)
                    q_pair_inv = pl.reshape(pl.rsqrt(q_pair_var, high_precision=True), [Q_ROPE_T_TILE * 2, 1])
                    q_pair_nope = pl.row_expand_mul(q_pair_dq[:, 0:NOPE_DIM], q_pair_inv)
                    q_pair_nope_bf16 = pl.cast(q_pair_nope, target_type=pl.BF16, mode="rint")
                    q_pair_rope = pl.row_expand_mul(q_pair_dq[:, NOPE_DIM:HEAD_DIM], q_pair_inv)
                    q_pair_swapped = pl.gather(q_pair_rope, dim=-1, index=q_pair_swap)
                    q_pair_rot = pl.add(pl.mul(q_pair_rope, q_pair_cos), pl.mul(q_pair_swapped, q_pair_sin))
                    q_pair_rope_bf16 = pl.cast(q_pair_rot, target_type=pl.BF16, mode="rint")
                    # Alias the paired rows; keep the original four strided GM stores.
                    # A3 TCONCAT would copy each output row in UB before a combined store.
                    q_nope_pair = pl.reshape(q_pair_nope_bf16, [Q_ROPE_T_TILE, NOPE_DIM * 2])
                    q_rope_pair = pl.reshape(q_pair_rope_bf16, [Q_ROPE_T_TILE, ROPE_DIM * 2])
                    q_flat[out_tg : out_tg + Q_ROPE_T_TILE, h0 : h0 + NOPE_DIM] = q_nope_pair[:, :NOPE_DIM]
                    q_flat[out_tg : out_tg + Q_ROPE_T_TILE, h0 + HEAD_DIM : h0 + HEAD_DIM + NOPE_DIM] = (
                        q_nope_pair[:, NOPE_DIM:NOPE_DIM * 2]
                    )
                    q_flat[out_tg : out_tg + Q_ROPE_T_TILE, h0 + NOPE_DIM : h0 + HEAD_DIM] = (
                        q_rope_pair[:, :ROPE_DIM]
                    )
                    q_flat[out_tg : out_tg + Q_ROPE_T_TILE, h0 + HEAD_DIM + NOPE_DIM : h0 + HEAD_DIM * 2] = (
                        q_rope_pair[:, ROPE_DIM:ROPE_DIM * 2]
                    )
'''
    replacement = replacement.replace("* 2", "* Q_DEQUANT_HEAD_BATCH").replace(
        "pl.pipeline(0, dq_head_tile, 2,", "pl.pipeline(0, dq_head_tile, Q_DEQUANT_HEAD_BATCH,",
    )
    after = (before[:start] + replacement + before[end:]).replace(
        "Q_DEQUANT_WORKERS = 48", "Q_DEQUANT_WORKERS = 48\n\nQ_DEQUANT_HEAD_BATCH = 2",
    )
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
        if name == "compile.py":
            text = text.replace('ROOT / "compiled" / args.side', 'ROOT / "compiled_v2" / args.side')
        (ROOT / name).write_text(text)
    source = {"baseline": "f4861832", "source_prefix": str(PREFIX), "base_source": str(BASE),
              "variant": "pkg:" + PACKAGE, "cases": [[131072, 16], [8192, 24]],
              "change": "Q反量化满8行按相邻2head合批；RMS列维512不变，尾行及48-worker保持",
              "reference": "ops-nn19614968/rms_norm_whole_reduce_sum.h:SubProcess910/ComputeRstd",
              "not_changed": "Q_B24、任务依赖/early、量化顺序、rsqrt精度与BF16舍入、精度版及工具链",
              "scope": "先查UB容量/reshape搬运及向量指令，再独立两档；不叠加在跑Q_B20候选"}
    (ROOT / "source.json").write_text(json.dumps(source, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
