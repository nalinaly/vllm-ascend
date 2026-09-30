"""每组用明确的Q输出视图，避免四个消费者因整张Q的InOut产生假依赖。"""

import difflib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    path = ROOT / "source.json"
    manifest = json.loads(path.read_text())
    source = Path(manifest["sources"]["streamed4_const"])
    destination = source.with_name("streamed4_views")
    assert not destination.exists()
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    relative = Path("deepseek_v4_flash_hca/q_projection_streamed.py")
    file = destination / relative
    before = file.read_text()
    split = before.index("@pl.jit.inline(auto_scope=False)\ndef q_proj_q_streamed(")
    dequant, main = before[:split], before[split:]
    dequant = dequant.replace(
        "q: pl.Tensor[[T_DYN, H, HEAD_DIM], pl.BF16]", "q_view: pl.Tensor[[T_DYN, STREAM_HEADS * HEAD_DIM], pl.BF16]"
    )
    dequant = dequant.replace(
        "t_dim = pl.tensor.dim(q, 0)\n    q_flat = pl.reshape(q, [t_dim, H * HEAD_DIM])", "q_flat = q_view"
    )
    dequant = dequant.replace(
        "q_flat[out_tg : out_tg + Q_ROPE_T_TILE, h0 : h0 + NOPE_DIM]",
        "q_flat[out_tg : out_tg + Q_ROPE_T_TILE, source_h0 : source_h0 + NOPE_DIM]",
    )
    dequant = dequant.replace(
        "q_flat[out_tg : out_tg + Q_ROPE_T_TILE, h0 + NOPE_DIM : h0 + HEAD_DIM]",
        "q_flat[out_tg : out_tg + Q_ROPE_T_TILE, source_h0 + NOPE_DIM : source_h0 + HEAD_DIM]",
    )
    dequant = dequant.replace("[out_tg, h0_tail]", "[out_tg, source_h0_tail]")
    dequant = dequant.replace("[out_tg, h0_tail + NOPE_DIM]", "[out_tg, source_h0_tail + NOPE_DIM]")
    dequant = dequant.replace("    return q\n", "    return q_view\n")
    main = main.replace(
        "    t_dim = pl.tensor.dim(x, 0)\n",
        "    t_dim = pl.tensor.dim(x, 0)\n    q_flat = pl.reshape(q, [t_dim, H * HEAD_DIM])\n",
    )
    old = "            q_proj_q_dequant_stream(\n"
    assert main.count(old) == 4
    main = main.replace(
        old,
        """            q_group_view = pl.slice(q_flat, [t_dim, STREAM_HEADS * HEAD_DIM],
                                    [0, head_base * HEAD_DIM])
            q_proj_q_dequant_stream(
""",
    )
    main = main.replace(
        "wq_b_scale, rope_cos_il, rope_sin_signed, rope_swap_idx, q,",
        "wq_b_scale, rope_cos_il, rope_sin_signed, rope_swap_idx, q_group_view,",
    )
    after = dequant + main
    file.chmod(0o644)
    file.write_text(after)
    file.chmod(0o444)
    manifest["sources"]["streamed4_views"] = str(destination)
    manifest["cpu_failures"]["streamed4_unroll"] = "unroll still fails the NZ offset proof; explicit constants compiled"
    manifest["streamed_const_status"] = (
        "CPU compile passes, but orchestration has four InOut references to the full Q; not submitted to NPU"
    )
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    (ROOT / "streamed4_views.patch").write_text(
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
