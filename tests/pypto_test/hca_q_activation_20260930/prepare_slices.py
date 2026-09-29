"""Q激活四个K256块分别常驻，避免从整K1024 tile提取时改变L0布局。"""

import ast
import difflib
import json
import shutil
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
PREFIX = REPO.parent / ".cache/hca-q-al1-slices-a7a9f314-20260930"
RELATIVE = Path("deepseek_v4_flash_dspark/q_projection.py")


def main():
    assert not PREFIX.exists(), PREFIX
    source = REPO.parent / ".cache/hca-q-al1-a7a9f314-20260930/base"
    baseline, candidate = PREFIX / "base", PREFIX / "slices"
    shutil.copytree(source, baseline)
    shutil.copytree(source, candidate)
    path = candidate / RELATIVE
    before = path.read_text()
    prefix, body = before.split("def _q_proj_q_matmul_nz(", 1)
    start = body.index("        # Keep the column offset provably nonnegative")
    end = body.index("    return q_proj_i32, qproj_tid", start)
    replacement = textwrap.indent(
        textwrap.dedent("""\
        for t0 in pl.range(0, qproj_t_matmul, QPROJ_PIPE_M_TILE):
            m_rows = pl.min(QPROJ_PIPE_M_TILE, tile_rows - t0)
            # Retain four independent Mat tiles: L0 can use the baseline TMOV.
            a0 = pl.load(qr_i8_matmul, [t0, 0], [QPROJ_PIPE_M_TILE, QPROJ_PIPE_K_TILE],
                         valid_shape=[m_rows, QPROJ_PIPE_K_TILE], target_memory=pl.MemorySpace.Mat)
            a1 = pl.load(qr_i8_matmul, [t0, QPROJ_PIPE_K_TILE], [QPROJ_PIPE_M_TILE, QPROJ_PIPE_K_TILE],
                         valid_shape=[m_rows, QPROJ_PIPE_K_TILE], target_memory=pl.MemorySpace.Mat)
            a2 = pl.load(qr_i8_matmul, [t0, 2 * QPROJ_PIPE_K_TILE], [QPROJ_PIPE_M_TILE, QPROJ_PIPE_K_TILE],
                         valid_shape=[m_rows, QPROJ_PIPE_K_TILE], target_memory=pl.MemorySpace.Mat)
            a3 = pl.load(qr_i8_matmul, [t0, 3 * QPROJ_PIPE_K_TILE], [QPROJ_PIPE_M_TILE, QPROJ_PIPE_K_TILE],
                         valid_shape=[m_rows, QPROJ_PIPE_K_TILE], target_memory=pl.MemorySpace.Mat)
            for qproj_round in pl.range(0, (QPROJ_N_BLOCKS - qproj_worker + QPROJ_WORKERS - 1) // QPROJ_WORKERS):
                w_col0 = (qproj_worker + qproj_round * QPROJ_WORKERS) * QPROJ_MM_N_TILE
                w0 = pl.load(wq_b, [0, w_col0], [QPROJ_PIPE_K_TILE, QPROJ_MM_N_TILE],
                             target_memory=pl.MemorySpace.Mat)
                w1 = pl.load(wq_b, [QPROJ_PIPE_K_TILE, w_col0], [QPROJ_PIPE_K_TILE, QPROJ_MM_N_TILE],
                             target_memory=pl.MemorySpace.Mat)
                col_acc = pl.tile.matmul(a0, w0)
                w2 = pl.load(wq_b, [2 * QPROJ_PIPE_K_TILE, w_col0], [QPROJ_PIPE_K_TILE, QPROJ_MM_N_TILE],
                             target_memory=pl.MemorySpace.Mat)
                col_acc = pl.tile.matmul_acc(col_acc, a1, w1)
                w3 = pl.load(wq_b, [3 * QPROJ_PIPE_K_TILE, w_col0], [QPROJ_PIPE_K_TILE, QPROJ_MM_N_TILE],
                             target_memory=pl.MemorySpace.Mat)
                col_acc = pl.tile.matmul_acc(col_acc, a2, w2)
                col_acc = pl.tile.matmul_acc(col_acc, a3, w3)
                pl.store(col_acc, [t0, w_col0], q_proj_i32)
    """),
        " " * 8,
    )
    after = prefix + "def _q_proj_q_matmul_nz(" + body[:start] + replacement + body[end:]
    ast.parse(after)
    path.chmod(0o644)
    path.write_text(after)
    path.chmod(0o444)
    (ROOT / "slices.patch").write_text(
        "".join(
            difflib.unified_diff(
                before.splitlines(True),
                after.splitlines(True),
                fromfile="a/" + str(RELATIVE),
                tofile="b/" + str(RELATIVE),
                n=0,
            )
        )
    )
    (ROOT / "source_slices.json").write_text(
        json.dumps(
            {
                "baseline_commit": "a7a9f314",
                "baseline": str(baseline),
                "candidate": str(candidate),
                "change": "four K256 activation tiles resident in L1; preload B, preserve K order",
                "invariants": "same integer arithmetic, M128/N256/K256, 20 workers, dependencies and cache policy",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
