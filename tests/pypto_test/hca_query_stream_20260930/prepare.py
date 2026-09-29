"""跨query延续HCA QK/PV三槽流水；从基线提取原算术，不改mask/归约/量化顺序。"""

import ast
import difflib
import json
import re
import shutil
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
PREFIX = REPO.parent / ".cache/hca-query-stream-bb4c2831-20260930"
RELATIVE = Path("deepseek_v4_flash_hca/decode_sparse_attn_hca.py")


def rename(text, names):
    return re.sub(r"\b(?:" + "|".join(names) + r")\b", lambda m: names[m[0]], text)


def indent(text, spaces):
    return textwrap.indent(textwrap.dedent(text), " " * spaces)


def transform(before):
    left, rest = before.split("def _long_sparse_attn_hca_tp1(", 1)
    body, right = rest.split("def sparse_attn_hca_tp1(", 1)
    start = body.index("        for token in pl.range(worker, t_dim, NUM_QK_CORES):")
    end = body.index("    return o_packed_heads, attention_tid", start)
    original = body[start:end]
    metadata = original.split("            request =", 1)[1].split("            query =", 1)[0]
    metadata = "            request =" + metadata
    aic = original.split("            for tick in pl.range(work_count + QK_PRE_LAUNCH):\n", 1)[1]
    aic = aic.split("            for lane in pl.split_aiv", 1)[0]
    # QK addresses use the monotonic stream index; local tick still selects the
    # current query's compressed/raw block. PV consumes the older ring slot.
    aic = aic.replace("tick % QK_TRANSFER_SLOTS", "global_tick % QK_TRANSFER_SLOTS")
    aic = aic.replace("if tick >= QK_PRE_LAUNCH:", "if global_tick >= QK_PRE_LAUNCH:")
    aic = aic.replace("pv_work = tick - QK_PRE_LAUNCH", "pv_work = global_tick - QK_PRE_LAUNCH")
    vector = original.split("            for lane in pl.split_aiv(2, mode=pl.SplitMode.NONE):\n", 1)[1]
    initial = vector.split("                for vec_tick,", 1)[0]
    initial = rename(
        initial,
        {
            "running_m": "stream_seed_m",
            "running_l": "stream_seed_l",
            "running_left": "stream_seed_left",
            "running_right": "stream_seed_right",
        },
    )
    softmax = vector.split("                    if vec_tick < work_count:\n", 1)[1]
    softmax = softmax.split("                    if vec_tick >= QK_PRE_LAUNCH:", 1)[0]
    softmax = "                    if vec_tick < work_count:\n" + softmax
    softmax = softmax.replace("vec_tick % QK_TRANSFER_SLOTS", "vec_global % QK_TRANSFER_SLOTS")
    softmax = rename(softmax, {"token": "vec_token"})
    marker = "                        pl.system.sync_set(1, pipe=pl.PipeType.MTE3"
    position = softmax.index(marker)
    softmax = (
        softmax[:position]
        + textwrap.indent(
            textwrap.dedent("""\
        # Only this AIV reads these descriptors; no global metadata task.
        pl.tile.write(query_ring, [vec_global % QK_TRANSFER_SLOTS, 0], pl.cast(vec_token, pl.INT32))
        pl.tile.write(query_ring, [vec_global % QK_TRANSFER_SLOTS, 1], pl.cast(vec_tick, pl.INT32))
        pl.tile.write(query_ring, [vec_global % QK_TRANSFER_SLOTS, 2], pl.cast(work_count - vec_tick - 1, pl.INT32))
    """),
            " " * 24,
        )
        + softmax[position:]
    )
    merge = vector.split("                    if vec_tick >= QK_PRE_LAUNCH:\n", 1)[1]
    merge = merge.split("                    running_m, running_l,", 1)[0]
    merge = "                    if vec_global >= QK_PRE_LAUNCH:\n" + merge
    merge = merge.replace("out_work = vec_tick - QK_PRE_LAUNCH", "out_work = vec_global - QK_PRE_LAUNCH")
    insert_at = merge.index("                        # 长历史的 running_m")
    # Reset only when PV consumes the first block of a query. QK may already
    # have advanced to another query; its token is not the output token.
    reset = textwrap.indent(
        textwrap.dedent("""\
        out_local = pl.tile.read(query_ring, [out_work % QK_TRANSFER_SLOTS, 1])
        if out_local == 0:
            fresh_m = pl.load(sink_col, [head0, 0], [H // 2, 1])
            fresh_l = pl.mul(fresh_m, 0.0)
            fresh_left = pl.tile.full([H // 2, HEAD_DIM // 2], dtype=pl.FP32, value=0.0)
            fresh_right = pl.tile.full([H // 2, HEAD_DIM // 2], dtype=pl.FP32, value=0.0)
            current_m, current_l, current_left, current_right = pl.yield_(
                fresh_m, fresh_l, fresh_left, fresh_right,
            )
        else:
            current_m, current_l, current_left, current_right = pl.yield_(
                m_iter, l_iter, left_iter, right_iter,
            )
    """),
        " " * 24,
    )
    arithmetic = merge[insert_at:]
    until = arithmetic.index("                        m_after, l_after, left_after, right_after = pl.yield_(")
    arithmetic = (
        rename(
            arithmetic[:until],
            {
                "m_iter": "current_m",
                "l_iter": "current_l",
                "left_iter": "current_left",
                "right_iter": "current_right",
            },
        )
        + arithmetic[until:]
    )
    merge = merge[:insert_at] + reset + arithmetic
    publish = vector.split("                final_sink =", 1)[1]
    publish = "                final_sink =" + publish
    publish = rename(
        publish,
        {
            "running_m": "next_m",
            "running_l": "next_l",
            "running_left": "next_left",
            "running_right": "next_right",
            "token": "publish_token",
        },
    )
    insertion = merge.index("                        m_after, l_after, left_after, right_after = pl.yield_(")
    conditional_publish = (
        "                        remaining = pl.tile.read(query_ring, [out_work % QK_TRANSFER_SLOTS, 2])\n"
        "                        if remaining == 0:\n"
        "                            publish_token = pl.cast(pl.tile.read(\n"
        "                                query_ring, [out_work % QK_TRANSFER_SLOTS, 0],\n"
        "                            ), pl.INDEX)\n" + indent(publish, 28)
    )
    merge = merge[:insertion] + conditional_publish + merge[insertion:]
    cube = textwrap.dedent("""\
        # Persist KV slots while QK advances into the next query and PV drains
        # the previous one. Only the final query pays the pipeline tail.
        kv_l1 = pl.create_tile([QK_TRANSFER_SLOTS * ATTN_K_TILE, HEAD_DIM],
                              dtype=pl.BF16, target_memory=pl.MemorySpace.Mat)
        for token, (work_offset,) in pl.range(
            worker, t_dim, NUM_QK_CORES, init_values=(pl.cast(0, pl.INDEX),),
        ):
    """)
    cube += indent(metadata, 4)
    cube += textwrap.indent(
        textwrap.dedent("""\
        query = pl.load(q_flat, [token * H, 0], [H, HEAD_DIM], target_memory=pl.MemorySpace.Mat)
        if token + NUM_QK_CORES >= t_dim:
            extra = pl.yield_(QK_PRE_LAUNCH)
        else:
            extra = pl.yield_(0)
        for tick in pl.range(work_count + extra):
            global_tick = work_offset + tick
    """),
        " " * 4,
    )
    cube += indent(aic, 8)
    cube += "    work_offset_done = pl.yield_(work_offset + work_count)\n"
    vector_body = "for lane in pl.split_aiv(2, mode=pl.SplitMode.NONE):\n" + indent(initial, 4)
    vector_body += textwrap.indent(
        textwrap.dedent("""\
        query_ring = pl.tile.full([QK_TRANSFER_SLOTS, 8], dtype=pl.INT32, value=0)
        for vec_token, (vec_offset, m_query, l_query, left_query, right_query, max_query, sum_query) in pl.range(
            worker, t_dim, NUM_QK_CORES,
            init_values=(pl.cast(0, pl.INDEX), stream_seed_m, stream_seed_l,
                         stream_seed_left, stream_seed_right, max_ring_init, sum_ring_init),
        ):
    """),
        " " * 4,
    )
    vector_body += indent(rename(metadata, {"token": "vec_token"}), 8)
    vector_body += textwrap.indent(
        textwrap.dedent("""\
        if vec_token + NUM_QK_CORES >= t_dim:
            vec_extra = pl.yield_(QK_PRE_LAUNCH)
        else:
            vec_extra = pl.yield_(0)
        for vec_tick, (m_iter, l_iter, left_iter, right_iter, max_ring, sum_ring) in pl.range(
            work_count + vec_extra,
            init_values=(m_query, l_query, left_query, right_query, max_query, sum_query),
        ):
            vec_global = vec_offset + vec_tick
    """),
        " " * 8,
    )
    vector_body += indent(softmax, 12) + indent(merge, 12)
    vector_body += textwrap.indent(
        textwrap.dedent("""\
        running_m, running_l, running_left, running_right, max_ring_done, sum_ring_done = pl.yield_(
            m_after, l_after, left_after, right_after, updated_max_ring, updated_sum_ring,
        )
    """),
        " " * 12,
    )
    vector_body += textwrap.indent(
        textwrap.dedent("""\
        offset_done, m_done, l_done, left_done, right_done, max_done, sum_done = pl.yield_(
            vec_offset + work_count, running_m, running_l, running_left, running_right,
            max_ring_done, sum_ring_done,
        )
    """),
        " " * 8,
    )
    replacement = indent(cube, 8) + "\n" + indent(vector_body, 8)
    body = body[:start] + replacement + body[end:]
    return left + "def _long_sparse_attn_hca_tp1(" + body + "def sparse_attn_hca_tp1(" + right


def main():
    assert not PREFIX.exists(), PREFIX
    baseline, candidate = PREFIX / "base", PREFIX / "stream"
    shutil.copytree(REPO / "vllm_ascend/ops/pypto", baseline, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copytree(baseline, candidate)
    path = candidate / RELATIVE
    before = path.read_text()
    after = transform(before)
    ast.parse(after)
    path.write_text(after)
    for package in (baseline, candidate):
        for file in package.rglob("*.py"):
            file.chmod(0o444)
    (ROOT / "stream.patch").write_text(
        "".join(
            difflib.unified_diff(
                before.splitlines(True),
                after.splitlines(True),
                fromfile="a/" + str(RELATIVE),
                tofile="b/" + str(RELATIVE),
                n=0,
            )
        )
    )
    (ROOT / "source.json").write_text(
        json.dumps(
            {
                "baseline_commit": "bb4c2831",
                "baseline": str(baseline),
                "candidate": str(candidate),
                "change": "preserve QK/PV ring across queries; drain only at each worker's final query",
                "numerics": (
                    "unchanged block/quantization/merge order; per-slot query descriptors keep delayed PV correct"
                ),
                "scope": "long path only, same workers and task graph, no toolchain edits",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
