"""双query完整UB累积＋累计max概率；检查新的活跃区间是否允许更宽M复用KV。"""

import ast
import difflib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
PREFIX = REPO.parent / ".cache/hca-pair-online-e8bd858e-20260930"
RELATIVE = Path("deepseek_v4_flash_hca/decode_sparse_attn_hca.py")


def once(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new)


def main():
    assert not PREFIX.exists()
    sources = json.loads((ROOT.parent / "hca_online_softmax_20260930/source.json").read_text())["sources"]
    base = Path(sources["online"])
    target = PREFIX / "full_ub"
    shutil.copytree(base, target)
    before = (base / RELATIVE).read_text()
    old_path = REPO.parent / ".cache/hca-pair-query-ad0e6bbe-20260930/pair_ub" / RELATIVE
    old = old_path.read_text()
    body = old[old.index("    pair_rows = 2 * H"):]
    body = body[:body.index("\n\n@pl.jit.inline")]
    body = once(body, "deps=[raw_gather_tid, raw_valid_tid, cmp_gather_tid, rope_cs_tid]",
                "deps=[raw_gather_tid, raw_valid_tid, cmp_gather_tid, rope_cs_tid, q_ready]")
    body = once(body, "                seed_m = pl.load(sink_col, [0, 0], [H, 1])",
                "                seed_m = pl.load(sink_col, [0, 0], [H, 1])\n"
                "                softmax_seed = pl.load(sink_col, [0, 0], [H, 1])")
    body = once(body, "out2_iter, out3_iter, max_ring, sum_ring) in pl.range(",
                "out2_iter, out3_iter, max_ring, sum_ring, softmax_m_iter) in pl.range(")
    body = once(body, "seed_out2, seed_out3, seed_max, seed_sum),",
                "seed_out2, seed_out3, seed_max, seed_sum, softmax_seed),")
    for prefix, score in (("raw", "raw_scaled"), ("cmp", "cmp_masked")):
        body = once(body, f"{prefix}_m = pl.row_max({score}, reduce_tmp)",
                    f"{prefix}_block_m = pl.row_max({score}, reduce_tmp)\n"
                    f"                                {prefix}_m = pl.maximum(softmax_m_iter, {prefix}_block_m)")
    body = once(body, "ready_max, ready_sum = pl.yield_(next_max, next_sum)",
                "ready_max, ready_sum, ready_softmax = pl.yield_(next_max, next_sum, maximum)")
    body = once(body, "ready_max, ready_sum = pl.yield_(max_ring, sum_ring)",
                "ready_max, ready_sum, ready_softmax = pl.yield_(max_ring, sum_ring, softmax_m_iter)")
    body = once(body, "updated_max, updated_sum = pl.yield_(ready_max, ready_sum)",
                "updated_max, updated_sum, softmax_after = pl.yield_(ready_max, ready_sum, ready_softmax)")
    body = once(body, "updated_max, updated_sum = pl.yield_(max_ring, sum_ring)",
                "updated_max, updated_sum, softmax_after = pl.yield_(max_ring, sum_ring, softmax_m_iter)")
    body = once(body,
                "next_m = pl.maximum(m_iter, out_m)\n"
                "                            alpha = pl.exp(pl.sub(m_iter, next_m))\n"
                "                            beta = pl.exp(pl.sub(out_m, next_m))\n"
                "                            next_l = pl.add(pl.mul(alpha, l_iter), pl.mul(beta, out_l))",
                "next_m = out_m\n"
                "                            alpha = pl.exp(pl.sub(m_iter, next_m))\n"
                "                            next_l = pl.add(pl.mul(alpha, l_iter), out_l)")
    for part in range(4):
        body = once(body, f"pl.row_expand_mul(pv_part{part}, beta)", f"pv_part{part}")
    body = once(body, "o3_done, max_done, sum_done = pl.yield_(",
                "o3_done, max_done, sum_done, softmax_done = pl.yield_(")
    body = once(body, "o2_after, o3_after, updated_max, updated_sum,",
                "o2_after, o3_after, updated_max, updated_sum, softmax_after,")
    start = before.index("    # 全部历史共用一组 MIX 任务", before.index("def _long_sparse_attn_hca_tp1("))
    end = before.index("\n\n@pl.jit.inline", start)
    after = before[:start] + body + before[end:]
    ast.parse(after)
    path = target / RELATIVE
    path.chmod(0o644)
    path.write_text(after)
    for file in target.rglob("*.py"):
        file.chmod(0o444)
    (ROOT / "full_ub.patch").write_text("".join(difflib.unified_diff(
        before.splitlines(True), after.splitlines(True), n=0,
        fromfile="a/" + str(RELATIVE), tofile="b/" + str(RELATIVE),
    )))
    (ROOT / "source.json").write_text(json.dumps({
        "sources": {"base": str(base), "full_ub": str(target)},
        "production": sources["base"], "baseline": "e8bd858e recorded online candidate; not production",
        "change": "two query M128 compressed QK/PV, two L1 KV slots, four FP32 N128 accumulator tiles in UB",
        "invariants": "same per-query online softmax, masks, BF16 rounding, cache and q_ready dependencies",
        "hypothesis": "shorten beta/PV temporary lifetimes to fit full UB and avoid GM accumulator traffic",
        "limitations": "CPU capacity required first; two-slot pipeline drains per pair unlike single-query baseline",
    }, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
