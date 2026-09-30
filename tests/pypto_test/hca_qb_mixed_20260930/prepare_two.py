"""四组改两组，保持总Cube/Vector分工，衡量发布粒度和调度开销。"""

import ast
import difflib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    path = ROOT / "source.json"
    manifest = json.loads(path.read_text())
    source = Path(manifest["sources"]["streamed4_edges"])
    destination = source.with_name("streamed2_edges")
    assert not destination.exists()
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    relative = Path("deepseek_v4_flash_hca/q_projection_streamed.py")
    file = destination / relative
    before = file.read_text()
    after = before.replace("STREAM_GROUPS = 4", "STREAM_GROUPS = 2")
    start = after.index("        with pl.scope():\n            head_base = 32\n")
    end = after.rindex("    return q\n")
    after = after[:start] + after[end:]
    after = after.replace("head_base = 16", "head_base = 32").replace(
        "col = 8192 + col_local", "col = 16384 + col_local"
    )
    ast.parse(after)
    file.chmod(0o644)
    file.write_text(after)
    file.chmod(0o444)
    manifest["sources"]["streamed2_edges"] = str(destination)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    (ROOT / "streamed2_edges.patch").write_text(
        "".join(
            difflib.unified_diff(
                before.splitlines(True),
                after.splitlines(True),
                fromfile="a/" + str(relative),
                tofile="b/" + str(relative),
                n=0,
            )
        )
    )


if __name__ == "__main__":
    main()
