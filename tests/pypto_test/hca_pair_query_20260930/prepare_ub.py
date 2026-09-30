"""双query的输出按四个N128片常驻UB，检查能否消除GM累加状态往返。"""

import ast
import difflib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
RELATIVE = Path("deepseek_v4_flash_hca/decode_sparse_attn_hca.py")


def main():
    prefix = REPO.parent / ".cache/hca-pair-query-ad0e6bbe-20260930"
    candidate = prefix / "pair_ub"
    assert not candidate.exists(), candidate
    shutil.copytree(prefix / "pair_v2", candidate)
    path = candidate / RELATIVE
    before = path.read_text()
    left, body = before.split("def _long_sparse_attn_hca_tp1(", 1)

    def replace(old, new):
        nonlocal body
        assert body.count(old) == 1, old
        body = body.replace(old, new)

    replace("    accum = pl.create_tensor([NUM_QK_CORES * pair_rows, HEAD_DIM], dtype=pl.FP32)\n", "")
    seeds = "".join(f"                seed_out{i} = pl.tile.full([H, PV_N_TILE], dtype=pl.FP32, value=0.0)\n"
                    for i in range(4))
    replace("                acc_row = (worker * 2 + lane) * H\n", seeds)
    replace("for vec_tick, (m_iter, l_iter, has_iter, max_ring, sum_ring) in pl.range(",
            "for vec_tick, (m_iter, l_iter, out0_iter, out1_iter, "
            "out2_iter, out3_iter, max_ring, sum_ring) in pl.range(")
    replace("init_values=(seed_m, seed_l, pl.cast(0, pl.INDEX), seed_max, seed_sum),",
            "init_values=(seed_m, seed_l, seed_out0, seed_out1, seed_out2, seed_out3, seed_max, seed_sum),")
    start = body.index("                            for out_col in pl.range(HEAD_DIM // PV_N_TILE):")
    end = body.index("                            if out_tick == cmp_blocks + lane:", start)
    quarters = "".join(
        f"                            pv_part{i} = pl.load(values, [out_row, {i} * PV_N_TILE], [H, PV_N_TILE])\n"
        f"                            next_out{i} = pl.add(pl.row_expand_mul(out{i}_iter, alpha),\n"
        f"                                                pl.row_expand_mul(pv_part{i}, beta))\n"
        for i in range(4)
    )
    body = body[:start] + quarters + body[end:]
    replace("                                    merged = pl.load(accum, [acc_row + head, 0], [H_TILE, HEAD_DIM])",
            "                                    merged_left = pl.concat(\n"
            "                                        pl.tile.slice(next_out0, [H_TILE, PV_N_TILE], [head, 0]),\n"
            "                                        pl.tile.slice(next_out1, [H_TILE, PV_N_TILE], [head, 0]),\n"
            "                                    )\n"
            "                                    merged_right = pl.concat(\n"
            "                                        pl.tile.slice(next_out2, [H_TILE, PV_N_TILE], [head, 0]),\n"
            "                                        pl.tile.slice(next_out3, [H_TILE, PV_N_TILE], [head, 0]),\n"
            "                                    )\n"
            "                                    merged = pl.concat(merged_left, merged_right)")
    replace("active_m, active_l, active_has = pl.yield_(next_m, next_l, pl.cast(1, pl.INDEX))",
            "active_m, active_l, active_o0, active_o1, active_o2, active_o3 = pl.yield_(\n"
            "                                next_m, next_l, next_out0, next_out1, next_out2, next_out3,\n"
            "                            )")
    replace("active_m, active_l, active_has = pl.yield_(m_iter, l_iter, has_iter)",
            "active_m, active_l, active_o0, active_o1, active_o2, active_o3 = pl.yield_(\n"
            "                                m_iter, l_iter, out0_iter, out1_iter, out2_iter, out3_iter,\n"
            "                            )")
    replace("m_after, l_after, has_after = pl.yield_(active_m, active_l, active_has)",
            "m_after, l_after, o0_after, o1_after, o2_after, o3_after = pl.yield_(\n"
            "                            active_m, active_l, active_o0, active_o1, active_o2, active_o3,\n"
            "                        )")
    replace("m_after, l_after, has_after = pl.yield_(m_iter, l_iter, has_iter)",
            "m_after, l_after, o0_after, o1_after, o2_after, o3_after = pl.yield_(\n"
            "                            m_iter, l_iter, out0_iter, out1_iter, out2_iter, out3_iter,\n"
            "                        )")
    replace("m_done, l_done, has_done, max_done, sum_done = pl.yield_(\n"
            "                        m_after, l_after, has_after, updated_max, updated_sum,",
            "m_done, l_done, o0_done, o1_done, o2_done, o3_done, max_done, sum_done = pl.yield_(\n"
            "                        m_after, l_after, o0_after, o1_after, "
            "o2_after, o3_after, updated_max, updated_sum,")
    after = left + "def _long_sparse_attn_hca_tp1(" + body
    ast.parse(after)
    path.chmod(0o644)
    path.write_text(after)
    path.chmod(0o444)
    (ROOT / "ub.patch").write_text("".join(difflib.unified_diff(
        before.splitlines(True), after.splitlines(True),
        fromfile="a/" + str(RELATIVE), tofile="b/" + str(RELATIVE), n=0,
    )))
    source = json.loads((ROOT / "source.json").read_text())
    source.update(parent_candidate=source["candidate"], candidate=str(candidate),
                  change="pair_v2 plus four N128 FP32 output tiles resident in UB; no GM accumulator",
                  limits="CPU capacity check required before NPU; both query masks and arithmetic unchanged")
    (ROOT / "source_ub.json").write_text(json.dumps(source, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
