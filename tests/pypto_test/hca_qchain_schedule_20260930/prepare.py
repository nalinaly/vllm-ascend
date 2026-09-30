"""冻结Q投影同步启动及Q投影/反量化联动候选；不改变分块与流水。"""

import ast
import difflib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
PREFIX = REPO.parent / ".cache/hca-qchain-c6792787-reuse-20260930"


def main():
    source = REPO.parent / ".cache/hca-residual-c6792787-20260930/reuse"
    assert not PREFIX.exists(), PREFIX
    sources = {}
    for side in ("base", "syncq", "syncboth"):
        destination = PREFIX / side
        shutil.copytree(source, destination, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        sources[side] = str(destination)
        edits = {}
        if side != "base":
            edits[Path("deepseek_v4_flash_dspark/q_projection.py")] = [
                (
                    '    with pl.spmd(QPROJ_WORKERS, name_hint="qproj_matmul", deps=[qproj_dep]) as qproj_tid:',
                    '    with pl.spmd(QPROJ_WORKERS, name_hint="qproj_matmul", '
                    "deps=[qproj_dep], sync_start=True) as qproj_tid:",
                )
            ]
        if side == "syncboth":
            edits[Path("deepseek_v4_flash_dspark_perf/qkv_proj_rope.py")] = [
                (
                    '        name_hint="qproj_dequant_rms_nope_rope",\n        allow_early_resolve=True,',
                    '        name_hint="qproj_dequant_rms_nope_rope",\n'
                    "        allow_early_resolve=True,\n        sync_start=True,",
                )
            ]
        patches = []
        for relative, replacements in edits.items():
            path = destination / relative
            before = path.read_text()
            after = before
            for old, new in replacements:
                assert after.count(old) == 1, relative
                after = after.replace(old, new)
            ast.parse(after)
            path.chmod(0o644)
            path.write_text(after)
            patches.extend(
                difflib.unified_diff(
                    before.splitlines(True),
                    after.splitlines(True),
                    fromfile="a/" + str(relative),
                    tofile="b/" + str(relative),
                )
            )
        for path in destination.rglob("*.py"):
            path.chmod(0o444)
        if patches:
            (ROOT / f"{side}.patch").write_text("".join(patches))
    manifest = {
        "baseline": "c6792787 plus tested residual reuse; mode2 only",
        "sources": sources,
        "purpose": "compare Q Cube gang start alone and paired with 48-AIV dequant; retain worker counts and pipelines",
        "numerics": "no arithmetic, buffers or dependencies changed; validate saved full states",
    }
    (ROOT / "source.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
