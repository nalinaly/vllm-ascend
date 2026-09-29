"""缩小PV更新临时块并将softmax scratch生命周期限制在实际归约内。"""

import ast
import difflib
import json
import shutil

from prepare import PREFIX, RELATIVE, ROOT, once


def main():
    side = "softmax16_publish8_n64_rings"
    target = PREFIX / side
    assert not target.exists()
    shutil.copytree(PREFIX / "softmax16_publish8", target)
    path = target / RELATIVE
    before = path.read_text()
    start = before.index("def _long_sparse_attn_hca_tp1(")
    split = before.index("        for lane in pl.split_aiv(2, mode=pl.SplitMode.NONE):", start)
    end = before.index("\n\n@pl.jit.inline", split)
    body = before[split:end]
    # A3不支持在同一tile的列方向做任意窄assemble；改成每H16一整行的UB环。
    body = body.replace("pl.tile.full([2, H],", "pl.tile.full([2 * (H // H_TILE), H_TILE],")
    assert body.count("[vec_tick % 2, softmax_head]") == 2
    body = body.replace("[vec_tick % 2, softmax_head]",
                        "[(vec_tick % 2) * (H // H_TILE) + softmax_part, 0]")
    body = once(body, "head_max_done, vec_tick % 2, 0, [1, H],",
                "head_max_done, (vec_tick % 2) * (H // H_TILE), 0, [H // H_TILE, H_TILE],")
    for name in ("updated_max", "updated_sum"):
        body = once(body, f"{name}, out_tick % 2, 0, [1, H],",
                    f"{name}, (out_tick % 2) * (H // H_TILE), 0, [H // H_TILE, H_TILE],")
    body = once(body, "            reduce_tmp = pl.create_tile([H_TILE, ATTN_K_TILE], dtype=pl.FP32)\n", "")
    body = once(body, "                                softmax_head = softmax_part * H_TILE",
                "                                reduce_tmp = pl.create_tile([H_TILE, ATTN_K_TILE], dtype=pl.FP32)\n"
                "                                softmax_head = softmax_part * H_TILE")
    seed_start = body.index("                seed_out0 =")
    seed_end = body.index("                for vec_tick,", seed_start)
    seeds = "".join(
        f"                seed_out{i} = pl.tile.full([H, PV_N_TILE // 2], dtype=pl.FP32, value=0.0)\n"
        for i in range(8)
    )
    body = body[:seed_start] + seeds + body[seed_end:]
    for template in ("out{}_iter", "seed_out{}", "active_o{}", "o{}_after", "o{}_done"):
        old = ", ".join(template.format(i) for i in range(4))
        assert old in body, old
        body = body.replace(old, ", ".join(template.format(i) for i in range(8)))
    # 四片更新变八片，每次32KiB PV加载/缩放暂存减为16KiB，保留全部FP32累积位。
    update_start = body.index("                            pv_part0 =")
    update_end = body.index("                            if out_tick == cmp_blocks + lane:", update_start)
    updates = "".join(
        f"                            pv_part{i} = pl.load(values, [out_row, {i} * (PV_N_TILE // 2)], "
        "[H, PV_N_TILE // 2])\n"
        f"                            next_out{i} = pl.add(pl.row_expand_mul(out{i}_iter, alpha), pv_part{i})\n"
        for i in range(8)
    )
    body = body[:update_start] + updates + body[update_end:]
    body = once(body, "next_out0, next_out1, next_out2, next_out3,",
                ", ".join(f"next_out{i}" for i in range(8)) + ",")
    publish_start = body.index("                                    merged_left = pl.concat(")
    publish_end = body.index("                                    normalized =", publish_start)
    publication = ""
    for group in range(4):
        publication += (
            f"                                    merged_quarter{group} = pl.concat(\n"
            f"                                        pl.tile.slice(next_out{2 * group}, "
            "[HEADS_PER_GROUP, PV_N_TILE // 2], [head, 0]),\n"
            f"                                        pl.tile.slice(next_out{2 * group + 1}, "
            "[HEADS_PER_GROUP, PV_N_TILE // 2], [head, 0]),\n"
            "                                    )\n"
        )
    publication += (
        "                                    merged_left = pl.concat(merged_quarter0, merged_quarter1)\n"
        "                                    merged_right = pl.concat(merged_quarter2, merged_quarter3)\n"
        "                                    merged = pl.concat(merged_left, merged_right)\n"
    )
    body = body[:publish_start] + publication + body[publish_end:]
    after = before[:split] + body + before[end:]
    ast.parse(after)
    path.chmod(0o644)
    path.write_text(after)
    path.chmod(0o444)
    (ROOT / f"{side}.patch").write_text("".join(difflib.unified_diff(
        before.splitlines(True), after.splitlines(True), n=0,
        fromfile="a/" + str(RELATIVE), tofile="b/" + str(RELATIVE),
    )))
    manifest = json.loads((ROOT / "source.json").read_text())
    manifest["sources"][side] = str(target)
    (ROOT / "source.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
