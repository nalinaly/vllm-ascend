"""双query恢复两步预发射；Q按块加载，释放L1供概率复用，容纳三槽KV。"""

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
    candidate = prefix / "pair_three_slots"
    assert not candidate.exists(), candidate
    shutil.copytree(prefix / "pair_v2", candidate)
    path = candidate / RELATIVE
    before = path.read_text()
    left, body = before.split("def _long_sparse_attn_hca_tp1(", 1)

    def replace(old, new):
        nonlocal body
        assert body.count(old) == 1, old
        body = body.replace(old, new)

    replace("    pair_slots = 2", "    pair_slots = 3")
    replace("pl.create_tile([2 * ATTN_K_TILE, HEAD_DIM]", "pl.create_tile([3 * ATTN_K_TILE, HEAD_DIM]")
    replace("            pair_query = pl.load(q_flat, [token0 * H, 0], [2 * H, HEAD_DIM], "
            "target_memory=pl.MemorySpace.Mat)\n", "")
    replace("            # A two-slot ring leaves L1 capacity for M128 queries. Drain only\n"
            "            # once per pair; QK for the next block overlaps the preceding PV.\n"
            "            for tick in pl.range(work_count + 1):",
            "            # Q and probability have disjoint L1 lifetimes; load Q per block\n"
            "            # to share their storage and restore two-step QK/PV lookahead.\n"
            "            for tick in pl.range(work_count + 2):")
    replace("                        cmp_qk = pl.matmul(pair_query, cmp_key_t, out_dtype=pl.FP32)",
            "                        pair_query = pl.load(q_flat, [token0 * H, 0], [2 * H, HEAD_DIM],\n"
            "                                             target_memory=pl.MemorySpace.Mat)\n"
            "                        cmp_qk = pl.matmul(pair_query, cmp_key_t, out_dtype=pl.FP32)")
    replace("raw_query = pl.tile.slice(pair_query, [H, HEAD_DIM], [raw_lane * H, 0])",
            "raw_query = pl.load(q_flat, [(token0 + raw_lane) * H, 0], [H, HEAD_DIM],\n"
            "                                            target_memory=pl.MemorySpace.Mat)")
    replace("if tick >= 1:", "if tick >= 2:")
    replace("pv_tick = tick - 1", "pv_tick = tick - 2")
    replace("if vec_tick >= 1:", "if vec_tick >= 2:")
    replace("out_tick = vec_tick - 1", "out_tick = vec_tick - 2")
    replace("                    cmp_blocks + 3,", "                    cmp_blocks + 4,")
    for var in ("tick", "pv_tick", "vec_tick", "out_tick"):
        body = body.replace(f"{var} % 2", f"{var} % 3")
    # Only transfer-ring addressing changes; worker*2 query/accumulator mapping stays.
    body = body.replace("(worker * 2 + tick % 3)", "(worker * 3 + tick % 3)")
    body = body.replace("(worker * 2 + pv_tick % 3)", "(worker * 3 + pv_tick % 3)")
    body = body.replace("(worker * 2 + vec_tick % 3)", "(worker * 3 + vec_tick % 3)")
    body = body.replace("(worker * 2 + out_tick % 3)", "(worker * 3 + out_tick % 3)")
    body = body.replace("pl.tile.full([2, H]", "pl.tile.full([3, H]")
    after = left + "def _long_sparse_attn_hca_tp1(" + body
    ast.parse(after)
    path.chmod(0o644)
    path.write_text(after)
    path.chmod(0o444)
    (ROOT / "three_slots.patch").write_text("".join(difflib.unified_diff(
        before.splitlines(True), after.splitlines(True),
        fromfile="a/" + str(RELATIVE), tofile="b/" + str(RELATIVE), n=0,
    )))
    source = json.loads((ROOT / "source.json").read_text())
    source.update(parent_candidate=source["candidate"], candidate=str(candidate),
                  change="three KV slots and two-step lookahead; Q reloads per block to share L1 with probabilities",
                  limits="extra Q loads versus pair_v2; requires CPU capacity and NPU checks")
    (ROOT / "source_three_slots.json").write_text(json.dumps(source, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
