"""只读冻结首版，分别细化头部分工与Native同类MTE2回收信号。"""

import difflib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    manifest = json.loads((ROOT / "source.json").read_text())
    source = Path(manifest["sources"]["mixed20_v2"])
    relative = Path("deepseek_v4_flash_hca/q_projection_mixed.py")
    before = (source / relative).read_text()
    modifications = {
        "mixed24": ("QB_WORKERS = 20", "QB_WORKERS = 24"),
        "mixed20_mte2": (
            "sync_set(1, pipe=pl.PipeType.MTE3",
            "sync_set(1, pipe=pl.PipeType.MTE2",
        ),
    }
    for side, (old, new) in modifications.items():
        destination = source.parent / side
        assert not destination.exists()
        assert before.count(old) == 1
        after = before.replace(old, new)
        shutil.copytree(source, destination, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        path = destination / relative
        path.chmod(0o644)
        path.write_text(after)
        path.chmod(0o444)
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
    (ROOT / "source.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
