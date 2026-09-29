"""清理HCA局部模块命名，ND继续调用原Q投影；冻结可接入包。"""

import ast
import difflib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    path = ROOT / "source.json"
    manifest = json.loads(path.read_text())
    source = Path(manifest["sources"]["flat4_c16"])
    destination = source.with_name("ready_v2")
    assert not destination.exists()
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    hca = destination / "deepseek_v4_flash_hca"
    file = hca / "qkv_mixed.py"
    before = file.read_text()
    projection = (hca / "q_projection_streamed.py").read_text()
    tree = ast.parse(projection)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "q_proj_q_streamed")
    text = ast.get_source_segment(projection, node)
    signature = text[: text.index("    t_dim = pl.tensor.dim(x, 0)")]
    signature = signature.replace("def q_proj_q_streamed(", "def q_proj_q_nd_view(")
    helper = (
        "@pl.jit.inline(auto_scope=False)\n"
        + signature
        + """    t_dim = pl.tensor.dim(x, 0)
    q_dense = pl.reshape(q, [t_dim, H, HEAD_DIM])
    q_proj_q_separate(
        x, wq_b, wq_b_scale, rope_cos_il, rope_sin_signed, rope_swap_idx,
        q_dense, qr_i8_matmul, qr_scale_pad_store, qproj_dep,
    )
    return q


q_proj_q = q_proj_q_streamed if QUANT_WEIGHT_NZ else q_proj_q_nd_view
"""
    )
    after = before.replace("q_proj_q = q_proj_q_streamed\n", helper)
    ast.parse(after)
    file.unlink()
    output = hca / "qkv_proj_rope.py"
    output.write_text(after)
    output.chmod(0o444)
    # MIX旧实现不在本候选调用链中，不把无关实验一起接入。
    mixed = hca / "q_projection_mixed.py"
    if mixed.exists():
        mixed.unlink()
    decode = hca / "decode_hca.py"
    text = decode.read_text()
    decode.chmod(0o644)
    decode.write_text(text.replace("from .qkv_mixed import", "from .qkv_proj_rope import"))
    decode.chmod(0o444)
    manifest["sources"]["ready_v2"] = str(destination)
    manifest["cpu_failures"] = {
        "ready": "inline if with nested dim/reshape fallback leaves original dequant Q metadata uninferred"
    }
    manifest["ready_scope"] = "NZ uses four groups/16 Cube; ND retains original Q projection via a tracked 3D view"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    (ROOT / "ready_wrapper.patch").write_text(
        "".join(
            difflib.unified_diff(
                before.splitlines(True),
                after.splitlines(True),
                fromfile="a/deepseek_v4_flash_hca/qkv_mixed.py",
                tofile="b/deepseek_v4_flash_hca/qkv_proj_rope.py",
                n=0,
            )
        )
    )


if __name__ == "__main__":
    main()
