"""给QK分支使用独立局部变量，避免隐式tile回传与四标量yield冲突。"""

import ast
import difflib
import json
import shutil
from pathlib import Path

from prepare import RELATIVE, ROOT, once


def main():
    manifest = json.loads((ROOT / "source.json").read_text())
    base = Path(manifest["sources"]["macro512"])
    target = base.parent / "macro512_v2"
    assert not target.exists()
    shutil.copytree(base, target)
    before = (base / RELATIVE).read_text()
    start = before.index("    # 全部历史共用一组 MIX 任务", before.index("def _long_sparse_attn_hca_tp1("))
    end = before.index("        for lane in pl.split_aiv(2, mode=pl.SplitMode.NONE):", start)
    body = before[start:end]
    body = once(body, "qk_kv = pl.gather_row(qk_kv, raw_kv", "raw_loaded = pl.gather_row(qk_kv, raw_kv")
    body = once(body, "raw_key = pl.tile.transpose_view(qk_kv)", "raw_key = pl.tile.transpose_view(raw_loaded)")
    body = once(body, "qk_kv = pl.gather_row(qk_kv, cmp_work_kv", "cmp_loaded = pl.gather_row(qk_kv, cmp_work_kv")
    body = once(body, "key = pl.tile.transpose_view(qk_kv)", "key = pl.tile.transpose_view(cmp_loaded)")
    after = before[:start] + body + before[end:]
    ast.parse(after)
    path = target / RELATIVE
    path.chmod(0o644)
    path.write_text(after)
    path.chmod(0o444)
    (ROOT / "ssa_fix.patch").write_text("".join(difflib.unified_diff(
        before.splitlines(True), after.splitlines(True), n=0,
        fromfile="a/" + str(RELATIVE), tofile="b/" + str(RELATIVE),
    )))
    manifest["sources"]["macro512_v2"] = str(target)
    (ROOT / "source.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
