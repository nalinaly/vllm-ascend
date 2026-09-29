"""四组Q共享原布局，通过直接TaskId边表达独立写入与最终读依赖。"""

import ast
import difflib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    path = ROOT / "source.json"
    manifest = json.loads(path.read_text())
    source = Path(manifest["sources"]["streamed4_const"])
    destination = source.with_name("streamed4_edges")
    assert not destination.exists()
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    patches = []

    def edit(name, transform):
        relative = Path("deepseek_v4_flash_hca") / name
        file = destination / relative
        before = file.read_text()
        after = transform(before)
        assert after != before
        ast.parse(after)
        file.chmod(0o644)
        file.write_text(after)
        file.chmod(0o444)
        patches.extend(
            difflib.unified_diff(
                before.splitlines(True),
                after.splitlines(True),
                fromfile="a/" + str(relative),
                tofile="b/" + str(relative),
                n=0,
            )
        )

    def projection(text):
        text = text.replace("for dq_worker in pl.spmd(", "with pl.spmd(", 1)
        text = text.replace(
            "        allow_early_resolve=True,\n    ):",
            "        allow_early_resolve=True,\n    ) as dq_tid:\n        dq_worker = pl.tile.get_block_idx()",
            1,
        )
        text = text.replace("    return q\n", "    return dq_tid\n", 1)
        text = text.replace(
            "    qproj_dep: pl.Scalar[pl.TASK_ID],\n",
            "    qproj_dep: pl.Scalar[pl.TASK_ID],\n    q_ready: pl.Array[4, pl.TASK_ID],\n",
        )
        assert text.count("            q_proj_q_dequant_stream(\n") == 4
        text = text.replace("            q_proj_q_dequant_stream(\n", "            dq_tid = q_proj_q_dequant_stream(\n")
        for group in range(4):
            start = text.index(f"            head_base = {group * 16}\n")
            end = text.index("            )\n", text.index("            dq_tid =", start)) + len("            )\n")
            segment = text[start:end]
            segment = segment.replace("deps=[qproj_dep]", f"deps=[qproj_dep, q_ready[{group}]]")
            segment += f"            q_ready[{group}] = dq_tid\n"
            text = text[:start] + segment + text[end:]
        return text

    def wrapper(text):
        text = text.replace(
            "q_proj_q = q_proj_q_streamed if QUANT_WEIGHT_NZ else q_proj_q_separate", "q_proj_q = q_proj_q_streamed"
        )
        text = text.replace(
            "    qr_scale: pl.Tensor[[T_DYN, 1], pl.FP32],\n",
            "    qr_scale: pl.Tensor[[T_DYN, 1], pl.FP32],\n    q_ready: pl.Array[4, pl.TASK_ID],\n",
        )
        text = text.replace("        q_seq_dep,\n", "        q_seq_dep,\n        q_ready,\n")
        text = text.replace("        qr_scale,\n    )", "        qr_scale,\n        q_ready,\n    )")
        return text

    def decode(text):
        text = text.replace(
            "q = pl.create_tensor([tokens, H, HEAD_DIM], dtype=pl.BF16)",
            "q = pl.create_tensor([tokens, H, HEAD_DIM], dtype=pl.BF16, manual_dep=True)\n"
            "    q_ready = pl.array.create(4, pl.TASK_ID)\n"
            "    for group in pl.unroll(4):\n"
            "        q_ready[group] = pl.system.task_invalid()",
        )
        text = text.replace(
            "gamma_cq, gamma_ckv, q, kv, qr, qr_scale, ready,",
            "gamma_cq, gamma_ckv, q, kv, qr, qr_scale, q_ready, ready,",
        )
        return text.replace(
            "freqs_cos, freqs_sin, packed, raw_tid, compressed_tid,",
            "freqs_cos, freqs_sin, packed, raw_tid, compressed_tid, q_ready,",
        )

    def attention(text):
        # 仅改当前入口及其长短实现，历史分支保持原样。
        start = text.index("def _short_sparse_attn_hca_tp1(")
        prefix, body = text[:start], text[start:]
        old = "    cmp_cache_ready_dep: pl.Scalar[pl.TASK_ID],\n"
        assert body.count(old) == 3
        body = body.replace(old, old + "    q_ready: pl.Array[4, pl.TASK_ID],\n")
        old_deps = "deps=[raw_gather_tid, raw_valid_tid, cmp_gather_tid, rope_cs_tid]"
        assert body.count(old_deps) == 2
        body = body.replace(old_deps, "deps=[raw_gather_tid, raw_valid_tid, cmp_gather_tid, rope_cs_tid, q_ready]")
        body = body.replace(
            "            raw_cache_ready_dep, cmp_cache_ready_dep,\n",
            "            raw_cache_ready_dep, cmp_cache_ready_dep, q_ready,\n",
        )
        return prefix + body

    edit("q_projection_streamed.py", projection)
    edit("qkv_mixed.py", wrapper)
    edit("decode_hca.py", decode)
    edit("decode_sparse_attn_hca.py", attention)
    manifest["sources"]["streamed4_edges"] = str(destination)
    manifest["cpu_failures"]["streamed4_valid"] = (
        "JIT dependency discovery only includes JITFunction; direct pl.function helper is unresolved"
    )
    manifest["streamed_edges"] = (
        "Q remains [T,64,512] with correct full stride; lifetime manual_dep on private Q only; "
        "four producer TaskIds directly feed both short/long attention; groups write disjoint heads"
    )
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    (ROOT / "streamed4_edges.patch").write_text("".join(patches))


if __name__ == "__main__":
    main()
