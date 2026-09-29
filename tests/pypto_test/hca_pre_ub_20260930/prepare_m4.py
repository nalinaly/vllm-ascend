"""M4有效行、M8物理盒：扩大并行度并保持A3列向量32B对齐。"""

import ast
import difflib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    manifest = json.loads((ROOT / "source_ub_bf16.json").read_text())
    parent = Path(manifest["candidate"])
    destination = parent.parent / "mix_m4_box8_v3"
    assert not destination.exists(), destination
    shutil.copytree(parent, destination, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    relative = Path("deepseek_v4_flash_hca/hc_pre_fused.py")
    path = destination / relative
    before = path.read_text()
    prefix, body = before.split("    t_pad = ((t_dim + T_TILE - 1) // T_TILE) * T_TILE", 1)
    prefix = prefix.replace("T_TILE = 8  # other values miscompare", "T_TILE = 8\nMIX_ROWS = 4")
    body = "    t_pad = ((t_dim + MIX_ROWS - 1) // MIX_ROWS) * MIX_ROWS" + body
    body = body.replace("mixes_total", "pre_mixes_total").replace("mixes_piece", "pre_mixes_piece")
    body = body.replace("pl.spmd(t_pad // T_TILE,", "pl.spmd(t_pad // MIX_ROWS,")
    body = body.replace("t0 = pl.tile.get_block_idx() * T_TILE", "t0 = pl.tile.get_block_idx() * MIX_ROWS")
    body = body.replace("valid_rows = pl.min(T_TILE, t_dim - t0)", "valid_rows = pl.min(MIX_ROWS, t_dim - t0)")
    body = body.replace("[T_TILE, MIX_PAD])", "[T_TILE, MIX_PAD], valid_shape=[valid_rows, MIX_PAD])")
    body = body.replace("inv_col = pl.load(inv_rms, [t0, 0], [T_TILE, 1])",
                        "inv_col = pl.load(inv_rms, [t0, 0], [T_TILE, 1], valid_shape=[valid_rows, 1])")
    old = ("        pl.store(pre_val, [t0, 0], pre_val_store)\n"
           "        pre_tile = pl.load(pre_val_store, [t0, 0], [T_TILE, HC_PAD])")
    assert body.count(old) == 1
    body = body.replace(old, "        pre_tile = pre_val")
    # 多个worker都只有4个有效行，不能共用旧的单尾块scratch，也不能写完整8行。
    start = body.index("        if valid_rows == T_TILE:")
    end = body.index("        pre_tile_t = ", start)
    body = (body[:start] + "        pl.store(pl.set_validshape(post_pad, valid_rows, HC_MULT), [t0, 0], post)\n\n"
            + body[end:])
    start = body.index("        pre_tile_t = ")
    end = body.index("        sq_sum = ", start)
    gate = "        gate_rows = pl.cast(pl.tile.arange(0, [1, T_TILE], dtype=pl.INT32), pl.FP32)\n"
    for lane in range(4):
        gate += (
            f"        gate_idx{lane} = pl.cast(pl.add(pl.mul(gate_rows, HC_PAD), {lane}), pl.INT32)\n"
            f"        gate_tmp{lane} = pl.create_tile([1, T_TILE], dtype=pl.INT32)\n"
            f"        pre{lane} = pl.reshape(pl.tile.gather(pre_tile, gate_idx{lane}, gate_tmp{lane}), [T_TILE, 1])\n"
        )
    body = body[:start] + gate + body[end:]
    body = body.replace("    post_tail_store = ",
                        "    x_mixed = pl.create_tensor([t_pad, D], dtype=pl.BF16)\n    post_tail_store = ", 1)
    old = "            mixed_ub = pl.tile.assemble(mixed_ub, y_bf16, [mix_db * T_TILE, 0])"
    assert body.count(old) == 1
    body = body.replace(old, "            pl.store(pl.set_validshape(y_bf16, valid_rows, D_TILE), [t0, d0], x_mixed)")
    old = ("            mixed_input = pl.tile.extract(\n"
           "                mixed_ub, norm_db * T_TILE, 0, [T_TILE, D_TILE], target_memory=pl.MemorySpace.Vec,\n"
           "            )")
    assert body.count(old) == 1
    body = body.replace(old, "            mixed_input = pl.load(x_mixed, [t0, d0], [T_TILE, D_TILE], "
                            "valid_shape=[valid_rows, D_TILE])")
    start = body.index("            if valid_rows == T_TILE:")
    end = body.index("    return mixed_tid", start)
    body = body[:start] + (
        "            normed_valid = pl.set_validshape(normed_bf16, valid_rows, D_TILE)\n"
        "            pl.store(normed_valid, [t0, d0], x_normed)\n"
    ) + body[end:]
    after = prefix + body
    ast.parse(after)
    path.chmod(0o644)
    path.write_text(after)
    path.chmod(0o444)
    (ROOT / "m4.patch").write_text("".join(difflib.unified_diff(
        before.splitlines(True), after.splitlines(True),
        fromfile="a/" + str(relative), tofile="b/" + str(relative), n=0,
    )))
    manifest.update(parent_candidate=str(parent), candidate=str(destination), mixed_storage="GM",
                    change="mix有效4行/物理8行，24workers；门控留UB，所有输出恢复有效行，避免共享尾块竞争")
    (ROOT / "source_m4.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
