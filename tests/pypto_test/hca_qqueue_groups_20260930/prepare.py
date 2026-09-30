"""合并Cube提交粒度，保持四组独立Q反量化；观察SPMD跨调度线程分配。"""

import ast
import difflib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PREFIX = ROOT.parents[3] / ".cache/hca-qqueue-groups-5c0fa709-20260930"
RELATIVE = Path("deepseek_v4_flash_hca/q_projection_streamed.py")


def make_source(before, groups):
    head_count = 64 // groups
    prefix, body = before.split("    t_dim = pl.tensor.dim(x, 0)\n", 1)
    prefix = prefix.replace("STREAM_CUBE_WORKERS = 24 // STREAM_GROUPS",
                            f"STREAM_CUBE_WORKERS = 24 // {groups}\nCUBE_HEADS = {head_count}")
    prefix = prefix.replace("QPROJ_MM_T_DYN, STREAM_HEADS * HEAD_DIM", "QPROJ_MM_T_DYN, CUBE_HEADS * HEAD_DIM")
    prefix = prefix.replace("source_h0 = h * HEAD_DIM", "source_h0 = (head_base % CUBE_HEADS + h) * HEAD_DIM")
    prefix = prefix.replace("h0 = head_base * HEAD_DIM + source_h0", "h0 = (head_base + h) * HEAD_DIM")
    prefix = prefix.replace("source_h0_tail = h_tail * HEAD_DIM",
                            "source_h0_tail = (head_base % CUBE_HEADS + h_tail) * HEAD_DIM")
    prefix = prefix.replace("h0_tail = head_base * HEAD_DIM + source_h0_tail",
                            "h0_tail = (head_base + h_tail) * HEAD_DIM")
    intro = body[:body.index("        with pl.scope():")]
    original = body[body.index("        with pl.scope():"):]
    cube_end = original.index("            dq_tid = q_proj_q_dequant_stream(")
    cube = original[:cube_end]
    dq_end = original.index("            q_ready[0] = dq_tid") + len("            q_ready[0] = dq_tid\n")
    dq = original[cube_end:dq_end]
    pieces = []
    for group in range(groups):
        head_base = group * head_count
        members = range(head_base // 16, (head_base + head_count) // 16)
        chunk = cube.replace("            head_base = 0\n", "")
        chunk = chunk.replace("STREAM_HEADS * HEAD_DIM", "CUBE_HEADS * HEAD_DIM")
        chunk = chunk.replace("col = 0 + col_local", f"col = {head_base * 512} + col_local")
        chunk = chunk.replace("deps=[qproj_dep, q_ready[0]]",
                              "deps=[qproj_dep, " + ", ".join(f"q_ready[{i}]" for i in members) + "]")
        for member in members:
            consumer = dq.replace("                head_base,", f"                {member * 16},")
            consumer = consumer.replace("q_ready[0]", f"q_ready[{member}]")
            chunk += consumer
        pieces.append(chunk)
    after = prefix + "    t_dim = pl.tensor.dim(x, 0)\n" + intro + "".join(pieces) + "    return q\n"
    ast.parse(after)
    assert after.count('name_hint="hca_qb_stream"') == groups
    assert after.count("q_ready[") == 8
    return after


def main():
    assert not PREFIX.exists()
    manifest = json.loads((ROOT.parent / "hca_qgroups_workers_20260930/source.json").read_text())
    base = Path(manifest["sources"]["q24"])
    before = (base / RELATIVE).read_text()
    sources = {"base": str(base), "production": manifest["sources"]["base"]}
    for groups in (1, 2):
        side = f"cube_groups{groups}"
        dest = PREFIX / side
        shutil.copytree(base, dest)
        after = make_source(before, groups)
        path = dest / RELATIVE
        path.chmod(0o644)
        path.write_text(after)
        path.chmod(0o444)
        (ROOT / f"{side}.patch").write_text("".join(difflib.unified_diff(
            before.splitlines(True), after.splitlines(True), n=0,
            fromfile="a/" + str(RELATIVE), tofile="b/" + str(RELATIVE),
        )))
        sources[side] = str(dest)
    (ROOT / "source.json").write_text(json.dumps({
        "sources": sources,
        "baseline": "q24 candidate from 5c0fa709, not production",
        "invariants": "24 Cube total, M128/N256/K256 stage2, four independent 12-AIV dequant groups, exact arithmetic",
        "hypothesis": "larger SPMD ranges may distribute Q across scheduler partitions; avoid 6-worker local imbalance",
        "tradeoff": "fewer Cube producers postpone partial Q release; compare full HCA and parallel compressor/gather",
    }, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
