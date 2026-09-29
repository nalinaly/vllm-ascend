"""按真实启动分布比较16组分工及关闭提前解析；不改运行中的包。"""

import difflib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    path = ROOT / "source.json"
    manifest = json.loads(path.read_text())
    relative = Path("deepseek_v4_flash_hca/q_projection_mixed.py")
    variants = (
        ("mixed16", "mixed20_v2", "QB_WORKERS = 20", "QB_WORKERS = 16"),
        ("mixed24_late", "mixed24", "allow_early_resolve=True", "allow_early_resolve=False"),
    )
    for side, parent, old, new in variants:
        source = Path(manifest["sources"][parent])
        destination = source.with_name(side)
        assert not destination.exists()
        shutil.copytree(source, destination, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        file = destination / relative
        before = file.read_text()
        assert before.count(old) == 1
        after = before.replace(old, new)
        file.chmod(0o644)
        file.write_text(after)
        file.chmod(0o444)
        manifest["sources"][side] = str(destination)
        (ROOT / f"{side}.patch").write_text(
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
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
