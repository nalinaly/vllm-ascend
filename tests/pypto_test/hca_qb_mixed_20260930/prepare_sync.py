"""针对DFX中MIX启动分散，仅给24组Q_B候选添加整组同步启动。"""

import difflib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    path = ROOT / "source.json"
    manifest = json.loads(path.read_text())
    source = Path(manifest["sources"]["mixed24"])
    destination = source.with_name("mixed24_sync")
    assert not destination.exists()
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    relative = Path("deepseek_v4_flash_hca/q_projection_mixed.py")
    file = destination / relative
    before = file.read_text()
    old = 'name_hint="hca_qb_mixed", allow_early_resolve=True, deps=[qproj_dep],'
    assert before.count(old) == 1
    after = before.replace(old, old + " sync_start=True,")
    file.chmod(0o644)
    file.write_text(after)
    file.chmod(0o444)
    manifest["sources"]["mixed24_sync"] = str(destination)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    (ROOT / "mixed24_sync.patch").write_text(
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
