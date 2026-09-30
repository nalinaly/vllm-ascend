"""针对已观测到的六Cube组内迟启，叠加每组sync_start验证组合增量。"""

import difflib
import json
import shutil
from pathlib import Path

from prepare import PREFIX, ROOT


def main():
    dest = PREFIX / "q24_cube_sync"
    assert not dest.exists()
    shutil.copytree(PREFIX / "q24", dest)
    relative = Path("deepseek_v4_flash_hca/q_projection_streamed.py")
    path = dest / relative
    before = path.read_text()
    after = before
    for group in range(4):
        line = f'with pl.spmd(STREAM_CUBE_WORKERS, name_hint="hca_qb_stream", deps=[qproj_dep, q_ready[{group}]]):'
        assert after.count(line) == 1
        after = after.replace(line, line[:-2] + ", sync_start=True):")
    path.chmod(0o644)
    path.write_text(after)
    path.chmod(0o444)
    (ROOT / "q24_cube_sync.patch").write_text("""# Parent: q24 (not production)\n""" + "".join(difflib.unified_diff(
        before.splitlines(True), after.splitlines(True), n=0,
        fromfile="a/" + str(relative), tofile="b/" + str(relative),
    )))
    manifest = json.loads((ROOT / "source.json").read_text())
    manifest["sources"]["q24_cube_sync"] = str(dest)
    (ROOT / "source.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
