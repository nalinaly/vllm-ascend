"""双query全UB累积保留，三KV槽/两步预取通过Q按块重载让L1可容纳。"""

import ast
import difflib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PREFIX = ROOT.parents[3] / ".cache/hca-pair-depth-5c0fa709-20260930"
RELATIVE = Path("deepseek_v4_flash_hca/decode_sparse_attn_hca.py")


def once(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new)


def main():
    assert not PREFIX.exists()
    source = json.loads((ROOT.parent / "hca_pair_online_20260930/source.json").read_text())["sources"]
    base = Path(source["head_extract"])
    dest = PREFIX / "depth3"
    shutil.copytree(base, dest)
    before = (base / RELATIVE).read_text()
    start = before.index("    pair_rows = 2 * H", before.index("def _long_sparse_attn_hca_tp1("))
    end = before.index("\n\n@pl.jit.inline", start)
    body = before[start:end]
    body = once(body, "pair_slots = 2", "pair_slots = 3")
    body = once(body, "[2 * ATTN_K_TILE, HEAD_DIM]", "[3 * ATTN_K_TILE, HEAD_DIM]")
    load = (
        "            pair_query = pl.load(q_flat, [token0 * H, 0], [2 * H, HEAD_DIM], "
        "target_memory=pl.MemorySpace.Mat)\n"
    )
    body = once(body, load, "")
    body = once(body,
                "            # A two-slot ring leaves L1 capacity for M128 queries. Drain only\n"
                "            # once per pair; QK for the next block overlaps the preceding PV.\n",
                "            # Q在每块QK内重载；PV期间释放其L1区域，容纳三个KV槽。\n")
    body = once(body, "            for tick in pl.range(work_count + 1):",
                "            for tick in pl.range(work_count + 2):")
    body = once(body, "                if tick < work_count:\n",
                "                if tick < work_count:\n" + "        " + load)
    body = body.replace("tick % 2", "tick % 3")
    body = body.replace("worker * 2 +", "worker * 3 +")
    body = once(body, "                if tick >= 1:", "                if tick >= 2:")
    body = once(body, "pv_tick = tick - 1", "pv_tick = tick - 2")
    body = body.replace("[2 * (H // H_TILE), H_TILE]", "[3 * (H // H_TILE), H_TILE]")
    body = once(body, "                    cmp_blocks + 3,", "                    cmp_blocks + 4,")
    body = once(body, "                    if vec_tick >= 1:", "                    if vec_tick >= 2:")
    body = once(body, "out_tick = vec_tick - 1", "out_tick = vec_tick - 2")
    after = before[:start] + body + before[end:]
    ast.parse(after)
    path = dest / RELATIVE
    path.chmod(0o644)
    path.write_text(after)
    path.chmod(0o444)
    (ROOT / "depth3.patch").write_text("".join(difflib.unified_diff(
        before.splitlines(True), after.splitlines(True), n=0,
        fromfile="a/" + str(RELATIVE), tofile="b/" + str(RELATIVE),
    )))
    (ROOT / "source.json").write_text(json.dumps({
        "sources": {"base": str(base), "depth3": str(dest), "single_online": source["base"]},
        "baseline": "head_extract double-query full-UB candidate from 5c0fa709; not production",
        "change": "3 KV slots, 2-tick QK preload, Q reloaded per block instead of permanent L1 residency",
        "invariants": "full FP32 UB accumulation, dynamic head extract, online BF16, same per-query block order",
        "tradeoff": "extra Q MTE2 traffic; still drains once per pair; CPU capacity/load required before NPU",
    }, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
