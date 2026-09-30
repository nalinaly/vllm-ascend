"""显式L0累加器声明动态M所需的compact存法，冻结为新的私有候选。"""

import ast
import difflib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RELATIVE = Path("deepseek_v4_flash_dspark_perf/decode_o_proj.py")


def main():
    manifest_path = ROOT / "source.json"
    manifest = json.loads(manifest_path.read_text())
    parent = Path(manifest["sources"]["nested"])
    target = parent.parent / "compact"
    assert not target.exists()
    shutil.copytree(parent, target)
    path = target / RELATIVE
    before = path.read_text()
    old = "seed = pl.create_tile([PROJ_A_ROW_TILE, 128], dtype=pl.FP32, target_memory=pl.MemorySpace.Acc)"
    new = (
        "seed_storage = pl.create_tile([PROJ_A_ROW_TILE, 128], dtype=pl.FP32,\n"
        "                                          target_memory=pl.MemorySpace.Acc, compact=True)\n"
        "            seed = pl.tile.set_validshape(seed_storage, pa_rows, 128)"
    )
    assert before.count(old) == 1
    after = before.replace(old, new)
    ast.parse(after)
    path.chmod(0o644)
    path.write_text(after)
    path.chmod(0o444)
    (ROOT / "compact.patch").write_text("".join(difflib.unified_diff(
        before.splitlines(True), after.splitlines(True), n=0,
        fromfile="a/" + str(RELATIVE), tofile="b/" + str(RELATIVE),
    )))
    manifest["sources"]["compact"] = str(target)
    manifest["nested_failure"] = "AccCompactValid: dynamic valid M needs compact Acc storage; plain create is rejected"
    manifest["compact_note"] = (
        "Explicit L0 path uses compact=True plus validshape; currently an internal compiler API. "
        "Before production adoption prefer a tile.matmul-derived seed or an officially supported narrow seed."
    )
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
