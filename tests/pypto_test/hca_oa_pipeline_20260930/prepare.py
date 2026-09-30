"""O-A N128显式L1 K512/L0 K128双缓冲；保留分组任务及N256原实现。"""

import ast
import difflib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
PREFIX = REPO.parent / ".cache/hca-oa-pipeline-f4dfef15-20260930"
RELATIVE = Path("deepseek_v4_flash_dspark_perf/decode_o_proj.py")

BODY = '''        if A_COL_TILE == 128:
            seed = pl.create_tile([PROJ_A_ROW_TILE, 128], dtype=pl.FP32, target_memory=pl.MemorySpace.Acc)
            for outer, (outer_acc,) in pl.pipeline(0, O_GROUP_IN // 512, stage=2, init_values=(seed,)):
                outer_k = outer * 512
                lhs = pl.load(o_packed, [pa_src0, outer_k], [PROJ_A_ROW_TILE, 512],
                              valid_shape=[pa_rows, 512], target_memory=pl.MemorySpace.Mat)
                rhs_group = pl.load(wo_a, [g, outer_k, n0], [1, 512, 128],
                                    target_memory=pl.MemorySpace.Mat)
                rhs = pl.reshape(rhs_group, [512, 128])
                for inner, (inner_acc,) in pl.pipeline(0, 4, stage=2, init_values=(outer_acc,)):
                    inner_k = inner * 128
                    lhs_part = pl.tile.extract(lhs, 0, inner_k, [PROJ_A_ROW_TILE, 128],
                                               target_memory=pl.MemorySpace.Left)
                    rhs_part = pl.tile.extract(rhs, inner_k, 0, [128, 128],
                                               target_memory=pl.MemorySpace.Right)
                    updated = pl.tile.matmul_acc(inner_acc, lhs_part, rhs_part,
                                                 init_cond=(outer == 0 and inner == 0))
                    inner_done = pl.yield_(updated)
                outer_done = pl.yield_(inner_done)
            stored = pl.store(outer_done, [pa_r0, out_col_g + n0], o_r_pad)
            o_r_pad = pl.yield_(stored)
        else:
__REFERENCE__
'''


def main():
    assert not PREFIX.exists()
    manifest = json.loads((ROOT.parent / "hca_qgroups_workers_20260930/source.json").read_text())
    base = Path(manifest["sources"]["base"])
    candidate = PREFIX / "nested"
    shutil.copytree(base, candidate)
    before = (base / RELATIVE).read_text()
    function = before.index("def _proj_a_mm_nz(")
    start = before.index("        xa_first = pl.slice(", function)
    end = before.index("    return o_r_pad, pa_tid", start)
    reference = before[start:end].replace(
        "        o_r_pad = pl.assemble(o_r_pad, acc_a, [pa_r0, out_col_g + n0])",
        "        stored = pl.assemble(o_r_pad, acc_a, [pa_r0, out_col_g + n0])\n"
        "        o_r_pad = pl.yield_(stored)",
    )
    reference = "\n".join("    " + line if line else line for line in reference.splitlines())
    after = before[:start] + BODY.replace("__REFERENCE__", reference) + before[end:]
    ast.parse(after)
    path = candidate / RELATIVE
    path.chmod(0o644)
    path.write_text(after)
    for file in candidate.rglob("*.py"):
        file.chmod(0o444)
    (ROOT / "nested.patch").write_text("".join(difflib.unified_diff(
        before.splitlines(True), after.splitlines(True), n=0,
        fromfile="a/" + str(RELATIVE), tofile="b/" + str(RELATIVE),
    )))
    (ROOT / "source.json").write_text(json.dumps({
        "sources": {"base": str(base), "nested": str(candidate)},
        "baseline": "0742f07c production operator, captured at f4dfef15",
        "change": "N128 O-A: L1 K512 stage2, explicit L0 K128 stage2, single init_cond accumulator",
        "invariants": "8 groups, same worker count/TaskIds/dependencies, Native NZ GM layout, K128 arithmetic order",
        "scope": "private HCA package only; N256 branch and public CSA operator unchanged",
        "reference": "ops-nn transpose_batch_mat_mul/pp_matmul_ein_sum_kernel.h L1 prefetch and L0 ping-pong",
        "expected_capacity_bytes": {"Mat": 524288, "Left": 65536, "Right": 65536, "Acc": 65536},
    }, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
