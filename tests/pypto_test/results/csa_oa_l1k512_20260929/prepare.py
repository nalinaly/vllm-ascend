"""按Native L1 depth思路合并NZ小中档K加载，保留大档与ND的K256。"""

import ast
import difflib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[4]
BASE = WORKSPACE / ".cache/csa-qrope-flat-gather-f4861832-v2-candidate"
TEMPLATE = ROOT.parent / "csa_qrope_flat_gather_20260929"
OLD_PACKAGE = "dsv4_csa_qrope_flat_gather_f4861832_v2"
PACKAGE = "dsv4_csa_oa_l1k512_369ad2c1_v1"
PREFIX = WORKSPACE / ".cache/csa-oa-l1k512-369ad2c1-v1"


def change(before):
    text = before.replace("A_K_TILE = 256\n", "A_K_TILE = 256\nPROJ_A_NZ_SMALL_K_TILE = 512\n", 1)
    # Explicit constexpr arguments: PyPTO dependency specialization ignores defaults.
    text = text.replace("    A_COL_TILE: pl.constexpr,\n", "    A_COL_TILE: pl.constexpr,\n"
                        "    A_PREFETCH_K: pl.constexpr,\n")
    assert text.count("    A_PREFETCH_K: pl.constexpr,\n") == 3
    start = text.index("def _proj_a_mm_nz(")
    end = text.index("@pl.jit.inline\ndef _proj_a_mm_nd(", start)
    nz = text[start:end].replace("A_K_TILE", "A_PREFETCH_K")
    nz = nz.replace('    """Parallelize row and column tiles as upstream; retain Native NZ weights."""',
                    '    """Use larger NZ L1 K panels when N128 leaves capacity; preserve K order."""')
    text = text[:start] + nz + text[end:]
    text = text.replace("t_dim, proj_a_rows, heads_dep, A_COL_TILE,\n",
                        "t_dim, proj_a_rows, heads_dep, A_COL_TILE, A_PREFETCH_K,\n", 1)
    for row in ("PROJ_B_SMALL_T_TILE", "PROJ_B_MEDIUM_T_TILE"):
        old = f"            {row}, PROJ_A_MM_N_TILE,\n"
        assert old in text
        text = text.replace(old, f"            {row}, PROJ_A_MM_N_TILE, PROJ_A_NZ_SMALL_K_TILE,\n", 1)
    text = text.replace("            PROJ_B_MM_T_TILE, PROJ_A_LARGE_N_TILE,\n",
                        "            PROJ_B_MM_T_TILE, PROJ_A_LARGE_N_TILE, A_K_TILE,\n", 1)
    ast.parse(text)
    return text


def main():
    if (ROOT / "task.txt").exists():
        raise RuntimeError("Submitted sources must remain immutable")
    for side in ("baseline", "candidate"):
        target = Path(str(PREFIX) + "-" + side)
        if target.exists():
            raise RuntimeError(f"Existing snapshot: {target}")
        shutil.copytree(BASE, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "build_output"))
        packages = target / "vllm_ascend/ops/pypto"
        (packages / OLD_PACKAGE).rename(packages / PACKAGE)
    relative = Path("vllm_ascend/ops/pypto") / PACKAGE / "decode_o_proj.py"
    before = (Path(str(PREFIX) + "-baseline") / relative).read_text()
    production = (ROOT.parents[3] / "vllm_ascend/ops/pypto/deepseek_v4_flash_dspark_perf/decode_o_proj.py").read_text()
    assert ast.dump(ast.parse(before)) == ast.dump(ast.parse(production))
    after = change(before)
    target = Path(str(PREFIX) + "-candidate") / relative
    target.chmod(0o644)
    target.write_text(after)
    (ROOT / "candidate.patch").write_text("".join(difflib.unified_diff(
        before.splitlines(True), after.splitlines(True), fromfile="a/" + str(relative), tofile="b/" + str(relative))))
    for name in ("compile.py", "run.sh", "run_side.sh"):
        text = (TEMPLATE / name).read_text().replace(TEMPLATE.name, ROOT.name)
        text = text.replace("csa-qrope-flat-gather-f4861832-v2", PREFIX.name).replace(OLD_PACKAGE, PACKAGE)
        text = text.replace("8192:24", "8192:16")
        (ROOT / name).write_text(text)
    source = {
        "baseline": "369ad2c1", "source_prefix": str(PREFIX), "base_source": str(BASE),
        "variant": "pkg:" + PACKAGE, "cases": [[131072, 16], [8192, 16]],
        "change": "NZ O-A实际T<=96/N128时L1 K256→512；N256和ND保持K256，L0与状态另验",
        "reference": "ops-nn19614968/transpose_batch_mat_mul_base_tiling.cpp:DoCommonTiling depthA1/depthB1",
        "not_changed": "任务/worker/依赖/early、O-B/量化、原K序、精度版及工具链",
        "scope": "两代表档均B16以覆盖受影响T96；大档在算子内选择原策略，CPU核对N256生成码",
        "historical_difference": "旧§115是CANN9.0/WO_A ND/atomic1整层单档；本轮仅当前Native NZ原地址路径",
    }
    (ROOT / "source.json").write_text(json.dumps(source, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
