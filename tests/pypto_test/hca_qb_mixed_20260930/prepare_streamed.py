"""将Q_B分成四条独立生产者/消费者链，不让MIX占住等待中的AIV。"""

import ast
import difflib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    path = ROOT / "source.json"
    manifest = json.loads(path.read_text())
    source = Path(manifest["sources"]["base"])
    destination = source.with_name("streamed4_const")
    assert not destination.exists()
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    perf = (source / "deepseek_v4_flash_dspark_perf/qkv_proj_rope.py").read_text()
    tree = ast.parse(perf)
    functions = {n.name: ast.get_source_segment(perf, n) for n in tree.body if isinstance(n, ast.FunctionDef)}
    dequant = functions["q_proj_q_dequant"]
    dequant = dequant.replace("def q_proj_q_dequant(", "def q_proj_q_dequant_stream(")
    dequant = dequant.replace(
        "q_proj_i32: pl.Tensor[[QPROJ_MM_T_DYN, H * HEAD_DIM]",
        "q_proj_i32: pl.Tensor[[QPROJ_MM_T_DYN, STREAM_HEADS * HEAD_DIM]",
    )
    dequant = dequant.replace(
        "    tile_rows: pl.Scalar[pl.INDEX],\n",
        "    tile_rows: pl.Scalar[pl.INDEX],\n    head_base: pl.Scalar[pl.INDEX],\n",
    )
    dequant = dequant.replace("Q_DEQUANT_WORKERS", "STREAM_VEC_WORKERS")
    dequant = dequant.replace("dq_head_tile = H", "dq_head_tile = STREAM_HEADS")
    dequant = dequant.replace("// H", "// STREAM_HEADS")
    dequant = dequant.replace("H // dq_head_tile", "STREAM_HEADS // dq_head_tile")
    dequant = dequant.replace(
        "h0 = h * HEAD_DIM", "source_h0 = h * HEAD_DIM\n                    h0 = head_base * HEAD_DIM + source_h0"
    )
    dequant = dequant.replace(
        "q_proj_i32[tg : tg + Q_ROPE_T_TILE, h0 : h0 + HEAD_DIM]",
        "q_proj_i32[tg : tg + Q_ROPE_T_TILE, source_h0 : source_h0 + HEAD_DIM]",
    )
    dequant = dequant.replace(
        "h0_tail = h_tail * HEAD_DIM",
        "source_h0_tail = h_tail * HEAD_DIM\n                    h0_tail = head_base * HEAD_DIM + source_h0_tail",
    )
    dequant = dequant.replace("[tg, h0_tail]", "[tg, source_h0_tail]")
    imports = '''"""Q_B每16个head独立发布；保持原整数矩阵分块和逐行数值策略。"""
import pypto.language as pl
from ..deepseek_v4_flash_dspark_perf.qkv_proj_rope import (
    D, EPS, H, HEAD_DIM, NOPE_DIM, PREFILL_DENSE_TILE, Q_LORA,
    QPROJ_MM_T_DYN, QPROJ_T_PAD, Q_ROPE_H_TILE, Q_ROPE_T_TILE,
    ROPE_DIM, ROPE_DIM_SCALE, T_DYN,
)
from ..deepseek_v4_flash_dspark_perf.nz_mode import QUANT_WEIGHT_LAYOUT

STREAM_GROUPS = 4
STREAM_HEADS = H // STREAM_GROUPS
STREAM_CUBE_WORKERS = 20 // STREAM_GROUPS
STREAM_VEC_WORKERS = 48 // STREAM_GROUPS
STREAM_M = 128
STREAM_N = 256
STREAM_K = 256
'''
    signature = functions["q_proj_q"].split('    """', 1)[0].replace("def q_proj_q(", "def q_proj_q_streamed(")
    body = """    t_dim = pl.tensor.dim(x, 0)
    for tile_base in pl.range(0, t_dim, PREFILL_DENSE_TILE):
        tile_rows = pl.min(PREFILL_DENSE_TILE, t_dim - tile_base)
        matrix_rows = ((tile_rows + STREAM_M - 1) // STREAM_M) * STREAM_M
        for head_group in pl.unroll(STREAM_GROUPS):
            with pl.scope():
                head_base = head_group * STREAM_HEADS
                q_proj_i32 = pl.create_tensor([matrix_rows, STREAM_HEADS * HEAD_DIM], dtype=pl.INT32)
                with pl.spmd(STREAM_CUBE_WORKERS, name_hint="hca_qb_stream", deps=[qproj_dep]):
                    pl.set_cache_policy(wq_b, pl.CachePolicy.BYPASS)
                    worker = pl.tile.get_block_idx()
                    n_blocks = STREAM_HEADS * HEAD_DIM // STREAM_N
                    for step in pl.range((n_blocks - worker + STREAM_CUBE_WORKERS - 1) // STREAM_CUBE_WORKERS):
                        col_local = (worker + step * STREAM_CUBE_WORKERS) * STREAM_N
                        col = head_base * HEAD_DIM + col_local
                        for t0 in pl.range(0, matrix_rows, STREAM_M):
                            count = pl.min(STREAM_M, tile_rows - t0)
                            qr_first = pl.slice(qr_i8_matmul, [STREAM_M, STREAM_K], [t0, 0],
                                                valid_shape=[count, STREAM_K])
                            weight_first = wq_b[0:STREAM_K, col:col + STREAM_N]
                            acc = pl.matmul(qr_first, weight_first, out_dtype=pl.INT32)
                            for k0 in pl.pipeline(STREAM_K, Q_LORA, STREAM_K, stage=2):
                                qr_part = pl.slice(qr_i8_matmul, [STREAM_M, STREAM_K], [t0, k0],
                                                   valid_shape=[count, STREAM_K])
                                weight_part = wq_b[k0:k0 + STREAM_K, col:col + STREAM_N]
                                acc = pl.matmul_acc(acc, qr_part, weight_part)
                            q_proj_i32[t0:t0 + STREAM_M, col_local:col_local + STREAM_N] = acc
                q_proj_q_dequant_stream(
                    wq_b_scale, rope_cos_il, rope_sin_signed, rope_swap_idx, q,
                    qr_scale_pad_store, q_proj_i32, tile_base, tile_rows, head_base,
                )
    return q
"""
    # 当前NZ证明不追踪orchestration外层循环，普通unroll也晚于该检查；
    # 用四段明确的列常量，避免将外层head_base作为未知符号传入AIC。
    start = body.index("        for head_group in pl.unroll(STREAM_GROUPS):\n")
    end = body.index("    return q\n", start)
    lines = body[start:end].splitlines(True)[1:]
    group_body = "".join(line[4:] if line.startswith("    ") else line for line in lines)
    groups = []
    for group in range(4):
        segment = group_body.replace("head_base = head_group * STREAM_HEADS", f"head_base = {group * 16}")
        segment = segment.replace("col = head_base * HEAD_DIM + col_local", f"col = {group * 8192} + col_local")
        groups.append(segment)
    body = body[:start] + "".join(groups) + body[end:]
    module = imports + "\n\n@pl.jit.inline(auto_scope=False)\n" + dequant
    module += "\n\n@pl.jit.inline(auto_scope=False)\n" + signature + body
    ast.parse(module)
    (ROOT / "q_projection_streamed.py").write_text(module)
    (destination / "deepseek_v4_flash_hca/q_projection_streamed.py").write_text(module)
    wrapper = (
        (ROOT / "qkv_mixed.py")
        .read_text()
        .replace(
            "from .q_projection_mixed import q_proj_q_mixed",
            "from .q_projection_streamed import q_proj_q_streamed",
        )
        .replace("q_proj_q = q_proj_q_mixed if", "q_proj_q = q_proj_q_streamed if")
    )
    (destination / "deepseek_v4_flash_hca/qkv_mixed.py").write_text(wrapper)
    relative = Path("deepseek_v4_flash_hca/decode_hca.py")
    file = destination / relative
    before = file.read_text()
    after = before.replace(
        "from ..deepseek_v4_flash_dspark_perf.qkv_proj_rope import qkv_proj_rope",
        "from .qkv_mixed import qkv_proj_rope",
    )
    file.chmod(0o644)
    file.write_text(after)
    for file in destination.rglob("*.py"):
        file.chmod(0o444)
    manifest["sources"]["streamed4_const"] = str(destination)
    manifest["streamed_change"] = "four head groups; each has 5 Cube and 12 AIV workers, linked by its own INT32 tensor"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    (ROOT / "streamed_entry.patch").write_text(
        "".join(
            difflib.unified_diff(
                before.splitlines(True),
                after.splitlines(True),
                fromfile="a/" + str(relative),
                tofile="b/" + str(relative),
                n=0,
            )
        )
    )


if __name__ == "__main__":
    main()
