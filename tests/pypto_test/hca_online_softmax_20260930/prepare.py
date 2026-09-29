"""长档累计最大值先于BF16概率转换；保留跨query流水及独立PV归约状态。"""

import ast
import difflib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
PREFIX = REPO.parent / ".cache/hca-online-softmax-0742f07c-20260930-v2"


def change_once(text, before, after):
    assert text.count(before) == 1, before
    return text.replace(before, after)


def main():
    assert not PREFIX.exists()
    base = PREFIX / "base"
    candidate = PREFIX / "online"
    shutil.copytree(REPO / "vllm_ascend/ops/pypto", base, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copytree(base, candidate)
    relative = Path("deepseek_v4_flash_hca/decode_sparse_attn_hca.py")
    before = (base / relative).read_text()
    begin = before.index("def _long_sparse_attn_hca_tp1(")
    end = before.index("\n\n@pl.jit.inline", begin)
    part = before[begin:end]
    part = change_once(part,
        "for vec_tick, (m_iter, l_iter, left_iter, right_iter, max_ring, sum_ring) in pl.range(",
        "softmax_seed = pl.load(sink_col, [head0, 0], [H // 2, 1])\n"
        "                for vec_tick, (m_iter, l_iter, left_iter, right_iter, max_ring, sum_ring, "
        "softmax_m_iter) in pl.range(")
    part = change_once(part,
        "init_values=(m_query, l_query, left_query, right_query, max_query, sum_query),",
        "init_values=(m_query, l_query, left_query, right_query, max_query, sum_query, softmax_seed),")
    for prefix, score in (("raw", "raw_scaled"), ("cmp", "cmp_masked")):
        old = f"{prefix}_m = pl.row_max({score}, reduce_tmp)"
        new = (f"{prefix}_block_m = pl.row_max({score}, reduce_tmp)\n"
               f"                            {prefix}_m = pl.maximum(softmax_m_iter, {prefix}_block_m)")
        part = change_once(part, old, new)
    part = change_once(part,
        "updated_max_ring, updated_sum_ring = pl.yield_(next_max_ring, next_sum_ring)",
        "updated_max_ring, updated_sum_ring, softmax_m_after = pl.yield_(next_max_ring, next_sum_ring, maximum)")
    part = change_once(part,
        "updated_max_ring, updated_sum_ring = pl.yield_(max_ring, sum_ring)",
        "updated_max_ring, updated_sum_ring, softmax_m_after = pl.yield_(max_ring, sum_ring, softmax_m_iter)")
    part = change_once(part,
        "next_m = pl.maximum(current_m, out_m)\n"
        "                        alpha = pl.exp(pl.sub(current_m, next_m))\n"
        "                        beta = pl.exp(pl.sub(out_m, next_m))\n"
        "                        next_l = pl.add(pl.mul(alpha, current_l), pl.mul(beta, out_l))\n"
        "                        # 两个乘积先各自舍入为 FP32；有限值加法两侧交换不改变结果。",
        "# 概率已按累计max缩放；PV可直接加入，只缩放旧累积。\n"
        "                        next_m = out_m\n"
        "                        alpha = pl.exp(pl.sub(current_m, next_m))\n"
        "                        next_l = pl.add(pl.mul(alpha, current_l), out_l)")
    part = change_once(part, "pl.row_expand_mul(pv_left, beta)", "pv_left")
    part = change_once(part, "pl.row_expand_mul(pv_right, beta)", "pv_right")
    part = change_once(part,
        "running_m, running_l, running_left, running_right, max_ring_done, sum_ring_done = pl.yield_(\n"
        "                        m_after, l_after, left_after, right_after, updated_max_ring, updated_sum_ring,",
        "running_m, running_l, running_left, running_right, max_ring_done, sum_ring_done, softmax_m_done = pl.yield_(\n"
        "                        m_after, l_after, left_after, right_after, updated_max_ring, "
        "updated_sum_ring, softmax_m_after,")
    after = before[:begin] + part + before[end:]
    ast.parse(after)
    (candidate / relative).write_text(after)
    (ROOT / "online.patch").write_text("".join(difflib.unified_diff(
        before.splitlines(True), after.splitlines(True), fromfile="a/" + str(relative), tofile="b/" + str(relative),
    )))
    for path in PREFIX.rglob("*.py"):
        path.chmod(0o444)
    (ROOT / "source.json").write_text(json.dumps({
        "baseline_commit": "0742f07c", "sources": {"base": str(base), "online": str(candidate)},
        "invariants": "long-only; unchanged QK/PV Cube, three-slot ring, cross-query pipeline, masking and cache",
        "arithmetic_change": "cumulative maximum before probability BF16 cast; one PV scaling instead of two",
        "coupling": "softmax producer max resets per query independently of delayed PV consumer state",
        "acceptance": "diagnostic only until independent numerical reference and model token acceptance",
    }, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
