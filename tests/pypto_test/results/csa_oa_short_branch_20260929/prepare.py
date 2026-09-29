"""按实际最长历史与T96分支，仅短档选择已有收益的NZ O-A K512。"""

import ast
import difflib
import importlib.util
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[4]
BASE = WORKSPACE / ".cache/csa-qrope-flat-gather-f4861832-v2-candidate"
TEMPLATE = ROOT.parent / "csa_oa_l1k512_20260929"
OLD_PACKAGE = "dsv4_csa_qrope_flat_gather_f4861832_v2"
PACKAGE = "dsv4_csa_oa_short_branch_369ad2c1_v1"
PREFIX = WORKSPACE / ".cache/csa-oa-short-branch-369ad2c1-v1"


def replace_once(text, before, after):
    assert text.count(before) == 1, before
    return text.replace(before, after, 1)


def executable_ast(text):
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef)) and ast.get_docstring(node) is not None:
            node.body = node.body[1:]
    return ast.dump(tree)


def change_oa(text):
    spec = importlib.util.spec_from_file_location("oa_l1_prepare", TEMPLATE / "prepare.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    text = module.change(text)
    start = text.index("def decode_o_proj_tp1(")
    prefix, entry = text[:start], text[start:]
    entry = replace_once(entry, "    heads_dep: pl.Scalar[pl.TASK_ID],\n",
                         "    heads_dep: pl.Scalar[pl.TASK_ID],\n    short_history: pl.Scalar[pl.INT32],\n")
    entry = replace_once(entry, "PROJ_B_SMALL_T_TILE, PROJ_A_MM_N_TILE, PROJ_A_NZ_SMALL_K_TILE,",
                         "PROJ_B_SMALL_T_TILE, PROJ_A_MM_N_TILE, A_K_TILE,")
    old = """    elif t_dim <= PROJ_B_MEDIUM_T_TILE:
        x_out = _decode_o_proj_tp1_tiled(
            o_packed, wo_a, wo_b, wo_b_scale, residual, post, comb, x_out, heads_dep,
            PROJ_B_MEDIUM_T_TILE, PROJ_A_MM_N_TILE, PROJ_A_NZ_SMALL_K_TILE,
        )
"""
    new = """    elif t_dim == PROJ_B_MEDIUM_T_TILE:
        # 同为T96，设备四窗显示长档K512回退、短档获益；复用Indexer的实际长度判据。
        if short_history == 1:
            x_out = _decode_o_proj_tp1_tiled(
                o_packed, wo_a, wo_b, wo_b_scale, residual, post, comb, x_out, heads_dep,
                PROJ_B_MEDIUM_T_TILE, PROJ_A_MM_N_TILE, PROJ_A_NZ_SMALL_K_TILE,
            )
        else:
            x_out = _decode_o_proj_tp1_tiled(
                o_packed, wo_a, wo_b, wo_b_scale, residual, post, comb, x_out, heads_dep,
                PROJ_B_MEDIUM_T_TILE, PROJ_A_MM_N_TILE, A_K_TILE,
            )
    elif t_dim < PROJ_B_MEDIUM_T_TILE:
        x_out = _decode_o_proj_tp1_tiled(
            o_packed, wo_a, wo_b, wo_b_scale, residual, post, comb, x_out, heads_dep,
            PROJ_B_MEDIUM_T_TILE, PROJ_A_MM_N_TILE, A_K_TILE,
        )
"""
    return prefix + replace_once(entry, old, new)


def change_indexer(text):
    for name in ("indexer_score_topk_forest", "indexer_weights_score", "indexer"):
        start = text.index("def " + name + "(")
        end = text.find("\n@pl.jit", start)
        if end == -1:
            end = len(text)
        part = replace_once(text[start:end], "    kv_seq_lens: pl.Tensor[[B_DYN], pl.INT32],\n",
                            "    kv_seq_lens: pl.Tensor[[B_DYN], pl.INT32],\n"
                            "    max_topk_cache_len: pl.Scalar[pl.INDEX],\n")
        if name == "indexer_score_topk_forest":
            part = replace_once(part, """    max_topk_cache_len = 0
    for topk_batch in pl.range(b_dim):
        max_topk_cache_len = pl.max(max_topk_cache_len, pl.read(kv_seq_lens, [topk_batch]) // COMPRESS_RATIO)
""", "")
            part = replace_once(part, """    max_topk_cache_len = 0
    for topk_batch in pl.range(b_dim):
        topk_cache_len = pl.read(kv_seq_lens, [topk_batch]) // COMPRESS_RATIO
        max_topk_cache_len = pl.max(max_topk_cache_len, topk_cache_len)
""", "")
        else:
            part = replace_once(part, "        kv_seq_lens,\n", "        kv_seq_lens,\n        max_topk_cache_len,\n")
        text = text[:start] + part + text[end:]
    return text


def change_root(text):
    text = replace_once(text, "from .decode_indexer import indexer\n",
                        "from .decode_indexer import TOPK_CANDIDATES_PER_LEAF, indexer\n")
    text = replace_once(text, "        # Bound indexer scratch to its own runtime scope.\n",
                        "        # Indexer与O-A共用实际最长历史；原Indexer两次相同扫描合为一次。\n"
                        "        max_indexer_cache_len = 0\n"
                        "        for length_batch in pl.range(pl.tensor.dim(kv_seq_lens, 0)):\n"
                        "            max_indexer_cache_len = pl.max(\n"
                        "                max_indexer_cache_len, "
                        "pl.read(kv_seq_lens, [length_batch]) // COMPRESS_RATIO,\n"
                        "            )\n"
                        "        # Bound indexer scratch to its own runtime scope.\n")
    text = replace_once(text, "                kv_seq_lens,\n                late_dep,\n",
                        "                kv_seq_lens,\n                max_indexer_cache_len,\n"
                        "                late_dep,\n")
    arguments = "                o_packed_heads, wo_a, wo_b, wo_b_scale, x_hc, post_t, comb_t, x_out, heads_dep,\n"
    return replace_once(text, arguments, arguments +
                        "                pl.cast(max_indexer_cache_len <= TOPK_CANDIDATES_PER_LEAF, pl.INT32),\n")


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
    patch = []
    for name, mutation in (("decode_o_proj.py", change_oa), ("decode_indexer.py", change_indexer),
                           ("decode_csa.py", change_root)):
        relative = Path("vllm_ascend/ops/pypto") / PACKAGE / name
        before = (Path(str(PREFIX) + "-baseline") / relative).read_text()
        production = (ROOT.parents[3] / "vllm_ascend/ops/pypto/deepseek_v4_flash_dspark_perf" / name).read_text()
        # 已测包的Indexer有两处旧说明文字，执行语句必须与生产完全对应。
        assert executable_ast(before) == executable_ast(production), name
        after = mutation(before)
        ast.parse(after)
        target = Path(str(PREFIX) + "-candidate") / relative
        target.chmod(0o644)
        target.write_text(after)
        patch.extend(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
                                         fromfile="a/" + str(relative), tofile="b/" + str(relative)))
    (ROOT / "candidate.patch").write_text("".join(patch))
    for name in ("compile.py", "run.sh", "run_side.sh", "collect.py"):
        text = (TEMPLATE / name).read_text().replace(TEMPLATE.name, ROOT.name)
        text = text.replace("csa-oa-l1k512-369ad2c1-v1", PREFIX.name)
        text = text.replace("dsv4_csa_oa_l1k512_369ad2c1_v1", PACKAGE)
        text = text.replace("NZ O-A小中档按K512预取L1，大档保持K256", "NZ O-A仅实际短历史T96分支K512")
        text = text.replace("本轮两侧均B16/T96，覆盖受影响的N128分支；64份O-A工作及依赖不变。",
                            "两档B16/T96，短档K512、长档保持K256；64份O-A工作及依赖不变。")
        (ROOT / name).write_text(text)
    source = {
        "baseline": "369ad2c1", "source_prefix": str(PREFIX), "base_source": str(BASE),
        "variant": "pkg:" + PACKAGE, "cases": [[131072, 16], [8192, 16]],
        "change": "仅T96且实际max(kv_seq_lens//4)<=8192时NZ O-A用K512，其余使用原K256",
        "reference": "csa_oa_l1k512_20260929长B16核时+6.663%、短B16-8.638%；按实际长度分支保留候选",
        "not_changed": "根ABI/Native适配、KV布局、worker/依赖/early、精度版及工具链",
        "scope": ("复用既有实际最大长度：上提至公共编排层，Indexer两次重复扫描合为一次；"
                  "不加主机读取/环境变量/扫描任务，混合batch按最长请求"),
    }
    (ROOT / "source.json").write_text(json.dumps(source, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
