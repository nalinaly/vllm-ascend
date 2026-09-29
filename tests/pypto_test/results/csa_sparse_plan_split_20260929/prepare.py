"""将SWA页表计划提前，压缩索引计划仍等待Top-K；不改变算术或Native cache。"""

import ast
import difflib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[4]
PREVIOUS = ROOT.parent / "csa_ob_hc_scalar_fused_20260929"
BASE = WORKSPACE / ".cache/csa-ob-hc-scalar-fused-9a868d26-candidate"
PREFIX = WORKSPACE / ".cache/csa-sparse-plan-split-7b296153"
OLD_PACKAGE = "dsv4_csa_ob_hc_scalar_fused_9a868d26"
PACKAGE = "dsv4_csa_sparse_plan_split_7b296153"


def split_plan(before):
    start = before.index('    with pl.spmd(CSA_PLAN_WORKERS, name_hint="csa_slots_build_valid_qk_plan"')
    body = before.index('            # 压缩索引这段', start)
    split = before.index('            # ---- 滑窗有效位：', body)
    end = before.index('    # Native cosine rows', split)
    compressed = before[body:split]
    window = before[split:end]
    common = '''        plan_worker = pl.tile.get_block_idx()
        for bias_t0 in pl.range(plan_worker * BIAS_T_TILE, t_dim, CSA_PLAN_WORKERS * BIAS_T_TILE):
            bias_rows = pl.min(BIAS_T_TILE, t_dim - bias_t0)
'''
    replacement = '''    # SWA positions/page tables are independent of the Indexer's Top-K.
    with pl.spmd(CSA_PLAN_WORKERS, name_hint="csa_slots_window_plan",
                 allow_early_resolve=True) as window_plan_tid:
''' + common + window + '''    # Both phases use scalar stores to different columns of the same DDR
    # cache line. Explicitly finish SWA before compressed validity writes.
    with pl.spmd(CSA_PLAN_WORKERS, name_hint="csa_slots_compressed_plan",
                 deps=[window_plan_tid], allow_early_resolve=True) as qk_plan_tid:
''' + common + compressed + '\n'
    after = before[:start] + replacement + before[end:]
    ast.parse(after)
    return after


def main():
    if (ROOT / "task.txt").exists():
        raise RuntimeError("Do not modify a submitted private package")
    for side in ("baseline", "candidate"):
        destination = Path(str(PREFIX) + "-" + side)
        if destination.exists():
            raise RuntimeError(f"Private package already exists: {destination}")
        shutil.copytree(BASE, destination, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "build_output"))
        packages = destination / "vllm_ascend/ops/pypto"
        (packages / OLD_PACKAGE).rename(packages / PACKAGE)
    filename = Path("vllm_ascend/ops/pypto") / PACKAGE / "decode_sparse_attn_csa.py"
    before = (Path(str(PREFIX) + "-baseline") / filename).read_text()
    after = split_plan(before)
    destination = Path(str(PREFIX) + "-candidate") / filename
    destination.chmod(0o644)
    destination.write_text(after)
    (ROOT / "candidate.patch").write_text(''.join(difflib.unified_diff(
        before.splitlines(True), after.splitlines(True), fromfile="a/" + str(filename), tofile="b/" + str(filename))))
    for name in ("compile.py", "run_side.sh"):
        text = (PREVIOUS / name).read_text().replace(PREVIOUS.name, ROOT.name)
        text = text.replace("csa-ob-hc-scalar-fused-9a868d26", PREFIX.name).replace(OLD_PACKAGE, PACKAGE)
        (ROOT / name).write_text(text)
    source = {"baseline": "7b296153", "base_source": str(BASE), "source_prefix": str(PREFIX),
              "variant": "pkg:" + PACKAGE, "cases": [[131072, 16], [8192, 24]],
              "change": "将不依赖Top-K的SWA计划移到独立任务；压缩计划显式依赖SWA以保护共享DDR行",
              "scope": "私有单文件调度候选；数学、cache、Sparse incore保持，未修改生产或工具链"}
    (ROOT / "source.json").write_text(json.dumps(source, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
