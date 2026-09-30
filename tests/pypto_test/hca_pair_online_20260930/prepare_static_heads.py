"""静态展开H16分组，使当前PyPTO生成码保留UB列向量slice的head偏移。"""

import ast
import difflib
import json
import shutil

from prepare import PREFIX, RELATIVE, ROOT, once


def main():
    side = "static_heads"
    target = PREFIX / side
    assert not target.exists()
    shutil.copytree(PREFIX / "softmax16_publish8_n64_rings", target)
    path = target / RELATIVE
    before = path.read_text()
    # unroll不接受loop-carried参数；每组按顺序重新绑定环状态，保留旧softmax_m_iter。
    old = '''                            for softmax_part, (head_max_ring, head_sum_ring) in pl.range(
                                H // H_TILE, init_values=(max_ring, sum_ring),
                            ):'''
    new = '''                            head_max_done = max_ring
                            head_sum_done = sum_ring
                            for softmax_part in pl.unroll(H // H_TILE):
                                head_max_ring = head_max_done
                                head_sum_ring = head_sum_done'''
    after = once(before, old, new)
    after = once(after, "                                head_max_done, head_sum_done = pl.yield_(next_max, next_sum)",
                 "                                head_max_done = next_max\n"
                 "                                head_sum_done = next_sum")
    ast.parse(after)
    path.chmod(0o644)
    path.write_text(after)
    path.chmod(0o444)
    (ROOT / "static_heads.patch").write_text("".join(difflib.unified_diff(
        before.splitlines(True), after.splitlines(True), n=0,
        fromfile="a/" + str(RELATIVE), tofile="b/" + str(RELATIVE),
    )))
    manifest = json.loads((ROOT / "source.json").read_text())
    manifest["sources"][side] = str(target)
    (ROOT / "source.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
