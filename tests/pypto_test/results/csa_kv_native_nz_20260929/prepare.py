"""隔离KV原生NZ权重候选；保持分工和K累加顺序，先验证新鲜真实权重路径。"""

import ast
import difflib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[4]
BASE = WORKSPACE / ".cache/csa-qrope-flat-gather-f4861832-v2-candidate"
TEMPLATE = ROOT.parent / "csa_oa_l1k512_20260929"
OLD_PACKAGE = "dsv4_csa_qrope_flat_gather_f4861832_v2"
PACKAGE = "dsv4_csa_kv_native_nz_369ad2c1_v1"
PREFIX = WORKSPACE / ".cache/csa-kv-native-nz-369ad2c1-v1"


def replace_once(text, before, after):
    assert text.count(before) == 1, before
    return text.replace(before, after, 1)


def change_kernel(text):
    old = "wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16]"
    assert text.count(old) == 3
    text = text.replace(old, "wkv: pl.Tensor[[HEAD_DIM, D], pl.BF16, BF16_WEIGHT_LAYOUT]")
    for name, offset in (("dense_w", "dense_d0"), ("wkv_chunk", "d0")):
        text = replace_once(text,
                            f"{name} = wkv[{offset} : {offset} + KV_K_TILE, kv_col0 : kv_col0 + KV_N_TILE]",
                            f"{name} = wkv[kv_col0 : kv_col0 + KV_N_TILE, {offset} : {offset} + KV_K_TILE]")
    text = replace_once(text, "dense_x, dense_w, init_cond=(dense_k == 0)",
                        "dense_x, dense_w, b_trans=True, init_cond=(dense_k == 0)")
    text = replace_once(text, "kv_x_chunk_bf16, wkv_chunk, init_cond=(db == 0)",
                        "kv_x_chunk_bf16, wkv_chunk, b_trans=True, init_cond=(db == 0)")
    start = text.index("def _kv_project(")
    end = text.index("@pl.jit.inline(auto_scope=False)\ndef kv_proj_rope(", start)
    kernel = text[start:end]
    signature = kernel[:kernel.index('    """')]
    kernel = replace_once(kernel, "def _kv_project(", "def _kv_project_grouped(")
    kernel = replace_once(kernel, "    GROUP_ROWS: pl.constexpr,\n",
                          "    GROUP_ROWS: pl.constexpr,\n    KV_M_GROUPS: pl.constexpr,\n")
    kernel = replace_once(kernel, "    kv_m_groups = pl.min(KV_OM, pl.max(1, tile_rows // GROUP_ROWS))\n", "")
    # 常量直接写入核体；若先赋给编排局部变量，outline会把它变成未携带符号事实的动态入参。
    kernel = kernel.replace("kv_m_groups", "KV_M_GROUPS")
    wrapper = signature + '''    """Keep the existing 1/2/3 M partition; make its NZ divisor explicit."""
    kv_m_groups = pl.min(KV_OM, pl.max(1, tile_rows // GROUP_ROWS))
    if kv_m_groups == 3:
        kv_fp32 = _kv_project_grouped(
            x, wkv, kv_fp32, tile_base, tile_rows, t_matmul, late_dep, DENSE_M, GROUP_ROWS, 3,
        )
    elif kv_m_groups == 2:
        kv_fp32 = _kv_project_grouped(
            x, wkv, kv_fp32, tile_base, tile_rows, t_matmul, late_dep, DENSE_M, GROUP_ROWS, 2,
        )
    else:
        kv_fp32 = _kv_project_grouped(
            x, wkv, kv_fp32, tile_base, tile_rows, t_matmul, late_dep, DENSE_M, GROUP_ROWS, 1,
        )
    return kv_fp32


'''
    return text[:start] + kernel + "@pl.jit.inline(auto_scope=False)\n" + wrapper + text[end:]


def change_layout(text):
    text = replace_once(text, "from vllm_ascend import envs\n",
                        "from vllm_ascend import envs\n\nfrom .config import FLASH\n")
    return replace_once(text, '    for name in ("wq_a", "wq_b", "wo_a", "wo_b"):\n',
                        '    names = ("wq_a", "wq_b", "wo_a", "wo_b")\n'
                        '    # 精度版旧方向保持；仅原生几何的性能版KV进入直接绑定。\n'
                        '    if tuple(params["wkv"].annotation.shape) == (FLASH.head_dim, FLASH.hidden_size):\n'
                        '        names += ("wkv",)\n'
                        '    for name in names:\n')


def change_adapter(text):
    return replace_once(text, '        "wkv": weight(attention.wkv, (512, 4096), bf16, True),\n',
                        '        "wkv": (root_weight("wkv", (512, 4096), bf16) if "wkv" in layouts\n'
                        '                else weight(attention.wkv, (512, 4096), bf16, True)),\n')


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
    package = Path("vllm_ascend/ops/pypto")
    mutations = {
        package / PACKAGE / "qkv_proj_rope.py": change_kernel,
        package / PACKAGE / "decode_csa.py": lambda s: replace_once(
            s, "wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16]",
            "wkv: pl.Tensor[[HEAD_DIM, D], pl.BF16, BF16_WEIGHT_LAYOUT]"),
        package / "deepseek_v4_flash_dspark/nz_mode.py": change_layout,
        package / "deepseek_v4_flash_dspark/native_adapter.py": change_adapter,
    }
    patch = []
    for relative, mutation in mutations.items():
        before = (Path(str(PREFIX) + "-baseline") / relative).read_text()
        production_relative = str(relative).replace(PACKAGE, "deepseek_v4_flash_dspark_perf")
        production = (ROOT.parents[3] / production_relative).read_text()
        assert ast.dump(ast.parse(before)) == ast.dump(ast.parse(production))
        after = mutation(before)
        ast.parse(after)
        target = Path(str(PREFIX) + "-candidate") / relative
        target.chmod(0o644)
        target.write_text(after)
        patch.extend(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
                                         fromfile="a/" + str(relative), tofile="b/" + str(relative)))
    (ROOT / "candidate.patch").write_text("".join(patch))
    for name in ("compile.py", "run.sh", "run_side.sh"):
        text = (TEMPLATE / name).read_text().replace(TEMPLATE.name, ROOT.name)
        text = text.replace("csa-oa-l1k512-369ad2c1-v1", PREFIX.name)
        text = text.replace("dsv4_csa_oa_l1k512_369ad2c1_v1", PACKAGE)
        text = text.replace("8192:16", "8192:24")
        (ROOT / name).write_text(text)
    source = {
        "baseline": "369ad2c1", "source_prefix": str(PREFIX), "base_source": str(BASE),
        "variant": "pkg:" + PACKAGE, "cases": [[131072, 16], [8192, 24]],
        "change": "性能版KV复用Native [512,4096] NZ权重原地址，Cube b_trans=True；不在加载期转ND转置",
        "reference": "Native最新七档KV MatMulV2的NZ输入；ops-nn19614968按实际B布局选择加载路径",
        "not_changed": "M/N/K分块、完整K顺序、worker数量、依赖/early、缓存布局、精度版签名及工具链",
        "scope": "先测新鲜真实权重路径；若采用，另补旧快照方向迁移、ND/atomic/尾行兼容",
        "comparison_limit": "Native和PTO分工不同，不能将单核均值直接相减为同工作量性能差距",
        "nz_proof": "按原有M分组1/2/3显式constexpr分支，常量直接进入核体；任务数量和工作映射与旧公式一致",
    }
    (ROOT / "source.json").write_text(json.dumps(source, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
