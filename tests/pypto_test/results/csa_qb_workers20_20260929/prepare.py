"""冻结Q_B 24→20候选，原五参数入口显式传24，保留精度版调用契约。"""

import ast
import difflib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[4]
TEMPLATE = ROOT.parent / "csa_indexer_early_chain_20260929"
BASE = WORKSPACE / ".cache/csa-indexer-early-chain-7b296153-candidate"
PREFIX = WORKSPACE / ".cache/csa-qb-workers20-f4861832-v3"
OLD_PACKAGE = "dsv4_csa_indexer_early_chain_7b296153"
PACKAGE = "dsv4_csa_qb_workers20_f4861832_v3"


def shared_worker_parameter(before):
    # dep调用不读取constexpr默认值，因此保留原五参数wrapper显式传24。
    start = before.index("@pl.jit.inline")
    header, functions = before[:start], before[start:]
    functions = functions.replace("QPROJ_WORKERS", "qproj_workers")
    anchor = "    qproj_dep: pl.Scalar[pl.TASK_ID],\n"
    if functions.count(anchor) != 2:
        raise ValueError("ND/NZ helper signatures changed")
    functions = functions.replace(anchor, anchor + "    qproj_workers: pl.constexpr,\n")
    alias = "q_proj_q_matmul = _q_proj_q_matmul_nz if QUANT_WEIGHT_NZ else _q_proj_q_matmul_nd"
    assert functions.count(alias) == 1
    functions = functions.replace(alias, alias.replace("q_proj_q_matmul =", "q_proj_q_matmul_with_workers ="))
    functions += '''

@pl.jit.inline(auto_scope=False)
def q_proj_q_matmul(
    wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8, QUANT_WEIGHT_LAYOUT],
    qr_i8_matmul: pl.Tensor[[QPROJ_T_PAD, Q_LORA], pl.INT8],
    q_proj_i32: pl.Tensor[[QPROJ_MM_T_DYN, H * HEAD_DIM], pl.INT32],
    tile_rows: pl.Scalar[pl.INDEX],
    qproj_dep: pl.Scalar[pl.TASK_ID],
):
    q_proj_i32, qproj_task = q_proj_q_matmul_with_workers(
        wq_b, qr_i8_matmul, q_proj_i32, tile_rows, qproj_dep, QPROJ_WORKERS,
    )
    return q_proj_i32, qproj_task
'''
    after = header + functions
    ast.parse(after)
    return after


def performance_workers(before):
    before = before.replace("    q_proj_q_matmul,\n", "    q_proj_q_matmul_with_workers,\n")
    before = before.replace("q_proj_q_matmul(\n", "q_proj_q_matmul_with_workers(\n")
    anchor = "                tile_rows,\n                qproj_dep,\n            )"
    if before.count(anchor) != 1:
        raise ValueError("Expected one performance Q_B call")
    after = before.replace(anchor, "                tile_rows,\n                qproj_dep,\n"
                           "                20,\n            )")
    ast.parse(after)
    return after


def main():
    if (ROOT / "task.txt").exists():
        raise RuntimeError("Never edit a submitted package")
    for side in ("baseline", "candidate"):
        destination = Path(str(PREFIX) + "-" + side)
        if destination.exists():
            raise RuntimeError(f"Existing package: {destination}")
        shutil.copytree(BASE, destination, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "build_output"))
        packages = destination / "vllm_ascend/ops/pypto"
        (packages / OLD_PACKAGE).rename(packages / PACKAGE)
    patch = []
    changes = {
        Path("vllm_ascend/ops/pypto/deepseek_v4_flash_dspark/q_projection.py"): shared_worker_parameter,
        Path("vllm_ascend/ops/pypto") / PACKAGE / "qkv_proj_rope.py": performance_workers,
    }
    for relative, change in changes.items():
        before = (Path(str(PREFIX) + "-baseline") / relative).read_text()
        after = change(before)
        target = Path(str(PREFIX) + "-candidate") / relative
        target.chmod(0o644)
        target.write_text(after)
        patch.extend(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
                                        fromfile="a/" + str(relative), tofile="b/" + str(relative)))
    (ROOT / "candidate.patch").write_text("".join(patch))
    for name in ("compile.py", "run.sh", "run_side.sh"):
        text = (TEMPLATE / name).read_text().replace(TEMPLATE.name, ROOT.name)
        text = text.replace("csa-indexer-early-chain-7b296153", PREFIX.name).replace(OLD_PACKAGE, PACKAGE)
        if name == "compile.py":
            text = text.replace('ROOT / "compiled" / args.side', 'ROOT / "compiled_v3" / args.side')
        (ROOT / name).write_text(text)
    partitions = {}
    for columns in (64, 128):
        partitions[str(columns)] = {}
        for workers in (24, 20):
            indices = [list(range(w, columns, workers)) for w in range(workers)]
            assert sorted(n for group in indices for n in group) == list(range(columns))
            partitions[str(columns)][str(workers)] = {"total_columns": sum(map(len, indices)),
                                                     "min_tiles": min(map(len, indices)),
                                                     "max_tiles": max(map(len, indices))}
    source = {"baseline": "f4861832", "base_source": str(BASE), "source_prefix": str(PREFIX),
              "variant": "pkg:" + PACKAGE, "cases": [[131072, 16], [8192, 24]],
              "change": "性能Q_B派发24→20；共享ND/NZ显式constexpr，原五参数wrapper显式传24保护精度调用",
              "column_coverage": partitions,
              "scope": "独立私有候选；不叠加Q_B early、dequant worker或其他算术/任务/依赖改动",
              "comparison_scope": "Q_B工作份数改变，必须看整组总核时/跨度与完整CSA，不能只比单worker均值"}
    (ROOT / "source.json").write_text(json.dumps(source, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
