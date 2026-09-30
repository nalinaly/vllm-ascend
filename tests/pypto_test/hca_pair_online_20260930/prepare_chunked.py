"""借鉴Native按UB容量切Vector行块，保留Cube M128和完整FP32累积。"""

import argparse
import ast
import difflib
import json
import shutil

from prepare import PREFIX, RELATIVE, ROOT, once


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publish-eight", action="store_true")
    args = parser.parse_args()
    side = "softmax16_publish8" if args.publish_eight else "softmax16"
    target = PREFIX / side
    assert not target.exists()
    shutil.copytree(PREFIX / "full_ub", target)
    path = target / RELATIVE
    before = path.read_text()
    prefix, body = before.split("def _long_sparse_attn_hca_tp1(", 1)
    body = once(body, "reduce_tmp = pl.create_tile([H, ATTN_K_TILE], dtype=pl.FP32)",
                "reduce_tmp = pl.create_tile([H_TILE, ATTN_K_TILE], dtype=pl.FP32)")
    start = body.index("                            score = pl.load(scores,")
    stop_line = (
        "                            ready_max, ready_sum, ready_softmax = pl.yield_(next_max, next_sum, maximum)"
    )
    end = body.index(stop_line, start) + len(stop_line)
    chunk = body[start:end]
    chunk = chunk.replace("[vec_row, 0]", "[vec_row + softmax_head, 0]")
    chunk = chunk.replace("[H, ATTN_K_TILE]", "[H_TILE, ATTN_K_TILE]")
    chunk = chunk.replace("pl.maximum(softmax_m_iter,", "pl.maximum(previous_head_max,")
    chunk = chunk.replace("pl.set_validshape(cmp_scaled, H, valid_rows)",
                          "pl.set_validshape(cmp_scaled, H_TILE, valid_rows)")
    chunk = chunk.replace("[1, H]", "[1, H_TILE]")
    chunk = chunk.replace("pl.tile.assemble(max_ring,", "pl.tile.assemble(head_max_ring,")
    chunk = chunk.replace("pl.tile.assemble(sum_ring,", "pl.tile.assemble(head_sum_ring,")
    chunk = chunk.replace("[vec_tick % 2, 0]", "[vec_tick % 2, softmax_head]")
    chunk = chunk.replace(stop_line.strip(), "head_max_done, head_sum_done = pl.yield_(next_max, next_sum)")
    chunk = "\n".join("    " + line for line in chunk.splitlines())
    intro = '''                            for softmax_part, (head_max_ring, head_sum_ring) in pl.range(
                                H // H_TILE, init_values=(max_ring, sum_ring),
                            ):
                                softmax_head = softmax_part * H_TILE
                                previous_head_max = pl.tile.slice(softmax_m_iter, [H_TILE, 1], [softmax_head, 0])
'''
    ending = '''
                            softmax_complete = pl.reshape(pl.tile.extract(
                                head_max_done, vec_tick % 2, 0, [1, H], target_memory=pl.MemorySpace.Vec,
                            ), [H, 1])
                            ready_max, ready_sum, ready_softmax = pl.yield_(
                                head_max_done, head_sum_done, softmax_complete,
                            )'''
    body = body[:start] + intro + chunk + ending + body[end:]
    if args.publish_eight:
        start = body.index("                            if out_tick == cmp_blocks + lane:")
        end = body.index("                            active_m, active_l, active_o0", start)
        publish = body[start:end]
        # 每次只发布8个head，即一个O组；不改变每行归一化和RoPE运算顺序。
        publish = publish.replace("H_TILE", "HEADS_PER_GROUP")
        publish = publish.replace("[PUBLISH_GROUPS, O_GROUP_IN]", "[1, O_GROUP_IN]")
        publish = once(publish,
                       "                                    "
                       "pl.store(groups[1:2, :], [packed_row + T_PAD, 0], o_packed_heads)\n",
                       "")
        body = body[:start] + publish + body[end:]
    after = prefix + "def _long_sparse_attn_hca_tp1(" + body
    ast.parse(after)
    path.chmod(0o644)
    path.write_text(after)
    for file in target.rglob("*.py"):
        file.chmod(0o444)
    (ROOT / f"{side}.patch").write_text("".join(difflib.unified_diff(
        before.splitlines(True), after.splitlines(True), n=0,
        fromfile="a/" + str(RELATIVE), tofile="b/" + str(RELATIVE),
    )))
    manifest = json.loads((ROOT / "source.json").read_text())
    manifest["sources"][side] = str(target)
    (ROOT / "source.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
