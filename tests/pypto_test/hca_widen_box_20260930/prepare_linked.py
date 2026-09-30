"""基于已冻结M4/K512追加mix与Sinkhorn同步，检查两项收益能否叠加。"""

import difflib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    manifest = json.loads((ROOT / "source.json").read_text())
    source = Path(manifest["sources"]["m4_k512"])
    destination = source.parent / "m4_k512_linked"
    assert not destination.exists(), destination
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    relative = Path("deepseek_v4_flash_hca/hc_pre_fused.py")
    path = destination / relative
    before = path.read_text()
    after = before
    for name in ("mix_x_rms_norm", "comb_sinkhorn"):
        old = f'name_hint="{name}", allow_early_resolve=True'
        assert after.count(old) == 1
        after = after.replace(old, old + ", sync_start=True")
    path.chmod(0o644)
    path.write_text(after)
    for file in destination.rglob("*.py"):
        file.chmod(0o444)
    (ROOT / "m4_k512_linked.patch").write_text("".join(difflib.unified_diff(
        before.splitlines(True), after.splitlines(True),
        fromfile="a/" + str(relative), tofile="b/" + str(relative), n=0,
    )))
    manifest["sources"]["m4_k512_linked"] = str(destination)
    manifest["linked_change"] = "在m4_k512之上增加mix与Sinkhorn sync_start，显式评估组合效应"
    (ROOT / "source.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
