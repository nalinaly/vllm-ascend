"""整块RoPE Gather替代逐行Gather，保持算术、任务划分及依赖。"""

import ast
import difflib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
RELATIVE = Path("deepseek_v4_flash_dspark_perf/qkv_proj_rope.py")


def main():
    baseline = REPO.parent / ".cache/hca-q-al1-slices-a7a9f314-20260930/base"
    candidate = REPO.parent / ".cache/hca-q-flat-gather-a7a9f314-20260930/flat"
    assert not candidate.exists(), candidate
    shutil.copytree(baseline, candidate)
    path = candidate / RELATIVE
    before = path.read_text()
    old = "                q_swap_idx = rope_swap_idx[out_tg : out_tg + Q_ROPE_T_TILE, :]\n"
    new = old + """                # Flat gather handles all eight rows in one vector operation.
                # Axis gather lowers to a scalar loop with one gather per row.
                q_gather_row_seed = pl.mul(
                    pl.cast(pl.arange(0, [1, Q_ROPE_T_TILE], dtype=pl.INT32), pl.FP32),
                    ROPE_DIM_SCALE,
                )
                q_gather_row_grid = pl.col_expand_mul(
                    pl.full([ROPE_DIM, Q_ROPE_T_TILE], dtype=pl.FP32, value=1.0),
                    q_gather_row_seed,
                )
                q_gather_row_offsets = pl.cast(
                    pl.transpose(q_gather_row_grid, axis1=0, axis2=1), pl.INT32,
                )
                q_swap_flat_idx = pl.add(q_swap_idx, q_gather_row_offsets)
"""
    assert before.count(old) == 1
    after = before.replace(old, new)
    old = "q_rope_swapped = pl.gather(q_rope_chunk, dim=-1, index=q_swap_idx)"
    assert after.count(old) == 1
    after = after.replace(old, "q_rope_swapped = pl.gather(q_rope_chunk, index=q_swap_flat_idx)")
    ast.parse(after)
    path.chmod(0o644)
    path.write_text(after)
    path.chmod(0o444)
    (ROOT / "flat.patch").write_text(
        "".join(difflib.unified_diff(
            before.splitlines(True), after.splitlines(True),
            fromfile="a/" + str(RELATIVE), tofile="b/" + str(RELATIVE), n=0,
        ))
    )
    (ROOT / "source.json").write_text(json.dumps({
        "baseline_commit": "a7a9f314",
        "baseline": str(baseline), "candidate": str(candidate),
        "change": "Q RoPE axis gather to flat gather over eight rows; row offsets prepared once per head group",
        "invariants": "same arithmetic/rounding, workers, head pipeline stage2, task dependencies; tail unchanged",
        "reference": "ops-transformer 28f40354 interleave_rope_b11d.h:127 batches vector work across factor rows; "
                     "not a claim that the installed Native selects this template",
    }, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
