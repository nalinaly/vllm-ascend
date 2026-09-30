"""512列QK微块改为局部加载与两级流水，验证同一L1缓冲串行复用的代价。"""

import ast
import difflib
import json
import shutil
from pathlib import Path

from prepare import RELATIVE, ROOT, once


def main():
    manifest = json.loads((ROOT / "source.json").read_text())
    base = Path(manifest["sources"]["macro512_v2"])
    target = base.parent / "macro512_qkpipe"
    assert not target.exists()
    shutil.copytree(base, target)
    before = (base / RELATIVE).read_text()
    start = before.index("    # 全部历史共用一组 MIX 任务", before.index("def _long_sparse_attn_hca_tp1("))
    end = before.index("        for lane in pl.split_aiv(2, mode=pl.SplitMode.NONE):", start)
    body = before[start:end]
    body = once(body,
                "        qk_kv = pl.create_tile([ATTN_K_TILE, HEAD_DIM], dtype=pl.BF16, "
                "target_memory=pl.MemorySpace.Mat)\n", "")
    body = once(body,
                "                        raw_loaded = pl.gather_row(qk_kv, raw_kv, [0, 0],\n"
                "                                             [request * REQUEST_KV_ROWS + raw_drop, 0], "
                "[ATTN_K_TILE, HEAD_DIM])",
                "                        raw_loaded = pl.load(\n"
                "                            raw_kv, [request * REQUEST_KV_ROWS + raw_drop, 0],\n"
                "                            [ATTN_K_TILE, HEAD_DIM], target_memory=pl.MemorySpace.Mat,\n"
                "                        )")
    body = once(body, "                        for part in pl.range(parts):",
                "                        for part in pl.pipeline(parts, stage=2):")
    body = once(body,
                "                            cmp_loaded = pl.gather_row(qk_kv, cmp_work_kv, [0, 0], "
                "[cmp_row, 0], [ATTN_K_TILE, HEAD_DIM])",
                "                            cmp_loaded = pl.load(\n"
                "                                cmp_work_kv, [cmp_row, 0], [ATTN_K_TILE, HEAD_DIM],\n"
                "                                target_memory=pl.MemorySpace.Mat,\n"
                "                            )")
    after = before[:start] + body + before[end:]
    ast.parse(after)
    path = target / RELATIVE
    path.chmod(0o644)
    path.write_text(after)
    path.chmod(0o444)
    (ROOT / "qkpipe.patch").write_text("".join(difflib.unified_diff(
        before.splitlines(True), after.splitlines(True), n=0,
        fromfile="a/" + str(RELATIVE), tofile="b/" + str(RELATIVE),
    )))
    manifest["sources"]["macro512_qkpipe"] = str(target)
    manifest["qkpipe_parent"] = "macro512_v2; same arithmetic and PV, QK tile loads local to stage-2 pipeline"
    (ROOT / "source.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
