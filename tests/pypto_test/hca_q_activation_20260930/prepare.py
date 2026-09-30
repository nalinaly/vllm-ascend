"""Q投影按行块将K1024激活常驻L1，供每个worker的N列轮次复用。"""

import ast
import difflib
import json
import shutil
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
PREFIX = REPO.parent / ".cache/hca-q-al1-a7a9f314-20260930"
RELATIVE = Path("deepseek_v4_flash_dspark/q_projection.py")


def main():
    assert not PREFIX.exists(), PREFIX
    baseline, candidate = PREFIX / "base", PREFIX / "al1"
    shutil.copytree(REPO / "vllm_ascend/ops/pypto", baseline, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copytree(baseline, candidate)
    path = candidate / RELATIVE
    before = path.read_text()
    prefix, body = before.split("def _q_proj_q_matmul_nz(", 1)
    start = body.index("        # Keep the column offset provably nonnegative")
    end = body.index("    return q_proj_i32, qproj_tid", start)
    replacement = textwrap.indent(
        textwrap.dedent("""\
        # One complete activation tile serves every N round on this worker.
        # Keep Mat operands and the baseline K256 L0 matmul shape.
        for t0 in pl.range(0, qproj_t_matmul, QPROJ_PIPE_M_TILE):
            m_rows = pl.min(QPROJ_PIPE_M_TILE, tile_rows - t0)
            activation_l1 = pl.load(
                qr_i8_matmul, [t0, 0], [QPROJ_PIPE_M_TILE, Q_LORA],
                valid_shape=[m_rows, Q_LORA], target_memory=pl.MemorySpace.Mat,
            )
            # Keep NZ column offsets provably nonnegative after outlining.
            for qproj_round in pl.range(0, (QPROJ_N_BLOCKS - qproj_worker + QPROJ_WORKERS - 1) // QPROJ_WORKERS):
                qproj_n_idx = qproj_worker + qproj_round * QPROJ_WORKERS
                w_col0 = qproj_n_idx * QPROJ_MM_N_TILE
                qr_first = pl.tile.slice(activation_l1, [QPROJ_PIPE_M_TILE, QPROJ_PIPE_K_TILE], [0, 0])
                wq_first = pl.load(wq_b, [0, w_col0], [QPROJ_PIPE_K_TILE, QPROJ_MM_N_TILE],
                                   target_memory=pl.MemorySpace.Mat)
                col_acc = pl.tile.matmul(qr_first, wq_first)
                for k0 in pl.pipeline(QPROJ_PIPE_K_TILE, Q_LORA, QPROJ_PIPE_K_TILE, stage=2):
                    qr_chunk = pl.tile.slice(activation_l1, [QPROJ_PIPE_M_TILE, QPROJ_PIPE_K_TILE], [0, k0])
                    wq_chunk = pl.load(wq_b, [k0, w_col0], [QPROJ_PIPE_K_TILE, QPROJ_MM_N_TILE],
                                       target_memory=pl.MemorySpace.Mat)
                    col_acc = pl.tile.matmul_acc(col_acc, qr_chunk, wq_chunk)
                pl.store(col_acc, [t0, w_col0], q_proj_i32)
    """),
        " " * 8,
    )
    after = prefix + "def _q_proj_q_matmul_nz(" + body[:start] + replacement + body[end:]
    ast.parse(after)
    path.write_text(after)
    for package in (baseline, candidate):
        for file in package.rglob("*.py"):
            file.chmod(0o444)
    (ROOT / "al1.patch").write_text(
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
    (ROOT / "source.json").write_text(
        json.dumps(
            {
                "baseline_commit": "a7a9f314",
                "baseline": str(baseline),
                "candidate": str(candidate),
                "change": "Q projection activation AL1 full, N rounds reuse; retain K256 weight streaming and L0 shape",
                "invariants": "INT8 integer arithmetic, 20 workers, task dependencies, cache policy, ND path unchanged",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
