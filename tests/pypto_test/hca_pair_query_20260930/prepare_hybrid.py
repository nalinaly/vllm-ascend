"""四片常驻超过Vec预算后，保留前两片在UB，后两片用FP32 GM。"""

import argparse
import ast
import difflib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
RELATIVE = Path("deepseek_v4_flash_hca/decode_sparse_attn_hca.py")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default="pair_hybrid_reload")
    args = parser.parse_args()
    prefix = REPO.parent / ".cache/hca-pair-query-ad0e6bbe-20260930"
    candidate = prefix / args.variant
    assert not candidate.exists(), candidate
    shutil.copytree(prefix / "pair_ub", candidate)
    path = candidate / RELATIVE
    before = path.read_text()
    left, body = before.split("def _long_sparse_attn_hca_tp1(", 1)

    def replace(old, new):
        nonlocal body
        assert body.count(old) == 1, old
        body = body.replace(old, new)

    replace("    ffts = pl.create_tensor([256], dtype=pl.INT64)",
            "    accum = pl.create_tensor([NUM_QK_CORES * pair_rows, HEAD_DIM // 2], dtype=pl.FP32)\n"
            "    ffts = pl.create_tensor([256], dtype=pl.INT64)")
    for i in (2, 3):
        replace(f"                seed_out{i} = pl.tile.full([H, PV_N_TILE], dtype=pl.FP32, value=0.0)\n", "")
    replace("                for vec_tick, (m_iter, l_iter, out0_iter, out1_iter, "
            "out2_iter, out3_iter, max_ring, sum_ring)",
            "                acc_row = (worker * 2 + lane) * H\n"
            "                for vec_tick, (m_iter, l_iter, out0_iter, out1_iter, has_iter, max_ring, sum_ring)")
    replace("seed_m, seed_l, seed_out0, seed_out1, seed_out2, seed_out3, seed_max, seed_sum",
            "seed_m, seed_l, seed_out0, seed_out1, pl.cast(0, pl.INDEX), seed_max, seed_sum")
    for i in (2, 3):
        old = (f"                            pv_part{i} = pl.load(values, [out_row, {i} * PV_N_TILE], [H, PV_N_TILE])\n"
               f"                            next_out{i} = pl.add(pl.row_expand_mul(out{i}_iter, alpha),\n"
               f"                                                pl.row_expand_mul(pv_part{i}, beta))\n")
        new = (f"                            if has_iter == 0:\n"
               f"                                zero_part{i} = "
               "pl.tile.full([H, PV_N_TILE], dtype=pl.FP32, value=0.0)\n"
               f"                                prior_part{i} = pl.yield_(zero_part{i})\n"
               f"                            else:\n"
               f"                                loaded_part{i} = "
               f"pl.load(accum, [acc_row, {i-2} * PV_N_TILE], [H, PV_N_TILE])\n"
               f"                                prior_part{i} = pl.yield_(loaded_part{i})\n"
               f"                            pv_part{i} = pl.load(values, [out_row, {i} * PV_N_TILE], [H, PV_N_TILE])\n"
               f"                            next_out{i} = pl.add(pl.row_expand_mul(prior_part{i}, alpha),\n"
               f"                                                pl.row_expand_mul(pv_part{i}, beta))\n"
               f"                            pl.store(next_out{i}, [acc_row, {i-2} * PV_N_TILE], accum)\n")
        replace(old, new)
    # Carry only the first half across loop iterations; the other half has an
    # explicit first-write guard and retains all FP32 bits in GM.
    body = body.replace("active_o0, active_o1, active_o2, active_o3", "active_o0, active_o1, active_has")
    body = body.replace("next_m, next_l, next_out0, next_out1, next_out2, next_out3,",
                        "next_m, next_l, next_out0, next_out1, pl.cast(1, pl.INDEX),")
    body = body.replace("m_iter, l_iter, out0_iter, out1_iter, out2_iter, out3_iter,",
                        "m_iter, l_iter, out0_iter, out1_iter, has_iter,")
    body = body.replace("o0_after, o1_after, o2_after, o3_after", "o0_after, o1_after, has_after")
    body = body.replace("o0_done, o1_done, o2_done, o3_done", "o0_done, o1_done, has_done")
    replace("                                    merged_right = pl.concat(\n"
            "                                        pl.tile.slice(next_out2, [H_TILE, PV_N_TILE], [head, 0]),\n"
            "                                        pl.tile.slice(next_out3, [H_TILE, PV_N_TILE], [head, 0]),\n"
            "                                    )",
            "                                    # Reload only the final H16 slice; retaining next_out2/3\n"
            "                                    # until publish keeps both H64 tiles live and defeats spilling.\n"
            "                                    merged_right = "
            "pl.load(accum, [acc_row + head, 0], [H_TILE, HEAD_DIM // 2])")
    after = left + "def _long_sparse_attn_hca_tp1(" + body
    ast.parse(after)
    path.chmod(0o644)
    path.write_text(after)
    path.chmod(0o444)
    (ROOT / "hybrid.patch").write_text("".join(difflib.unified_diff(
        before.splitlines(True), after.splitlines(True),
        fromfile="a/" + str(RELATIVE), tofile="b/" + str(RELATIVE), n=0,
    )))
    source = json.loads((ROOT / "source.json").read_text())
    source.update(parent_candidate=str(prefix / "pair_ub"), candidate=str(candidate),
                  change="pair query with two N128 FP32 accumulator tiles in UB and remaining half in GM",
                  limits="four-quarter UB candidate exceeded configured Vec limit; hybrid requires CPU/NPU validation")
    (ROOT / "source_hybrid.json").write_text(json.dumps(source, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
