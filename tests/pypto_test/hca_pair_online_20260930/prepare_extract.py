"""以连续H16行为单位显式extract，避免动态tile.slice丢失head偏移。"""

import ast
import difflib
import json
import shutil

from prepare import PREFIX, RELATIVE, ROOT, once


def main():
    side = "head_extract"
    target = PREFIX / side
    assert not target.exists()
    shutil.copytree(PREFIX / "softmax16_publish8_n64_rings", target)
    path = target / RELATIVE
    before = path.read_text()
    old = "previous_head_max = pl.tile.slice(softmax_m_iter, [H_TILE, 1], [softmax_head, 0])"
    new = '''previous_head_max = pl.reshape(pl.tile.extract(
                                    pl.reshape(softmax_m_iter, [H // H_TILE, H_TILE]),
                                    softmax_part, 0, [1, H_TILE], target_memory=pl.MemorySpace.Vec,
                                ), [H_TILE, 1])'''
    after = once(before, old, new)
    ast.parse(after)
    path.chmod(0o644)
    path.write_text(after)
    path.chmod(0o444)
    (ROOT / "head_extract.patch").write_text("".join(difflib.unified_diff(
        before.splitlines(True), after.splitlines(True), n=0,
        fromfile="a/" + str(RELATIVE), tofile="b/" + str(RELATIVE),
    )))
    manifest = json.loads((ROOT / "source.json").read_text())
    manifest["sources"][side] = str(target)
    (ROOT / "source.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
