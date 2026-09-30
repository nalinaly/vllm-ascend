"""四组Q各由4增至6 Cube，并比较长档压缩gather限24 AIV的联动。"""

import ast
import difflib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
PREFIX = REPO.parent / ".cache/hca-qgroups-workers-e8bd858e-20260930"


def main():
    assert not PREFIX.exists()
    source = json.loads((ROOT.parent / "hca_online_softmax_20260930/source.json").read_text())["sources"]["base"]
    sources = {"base": source}
    for side in ("q24", "q24_gather24"):
        dest = PREFIX / side
        shutil.copytree(source, dest)
        patches = []
        changes = {"q_projection_streamed.py": (
            "STREAM_CUBE_WORKERS = 16 // STREAM_GROUPS", "STREAM_CUBE_WORKERS = 24 // STREAM_GROUPS",
        )}
        if side == "q24_gather24":
            changes["decode_sparse_attn_hca.py"] = ("CMP_GATHER_AIV_WAVE = 48", "CMP_GATHER_AIV_WAVE = 24")
        for name, (old, new) in changes.items():
            relative = Path("deepseek_v4_flash_hca") / name
            path = dest / relative
            before = path.read_text()
            assert before.count(old) == 1
            after = before.replace(old, new)
            ast.parse(after)
            path.chmod(0o644)
            path.write_text(after)
            patches.append("".join(difflib.unified_diff(
                before.splitlines(True), after.splitlines(True), n=0,
                fromfile="a/" + str(relative), tofile="b/" + str(relative),
            )))
        for file in dest.rglob("*.py"):
            file.chmod(0o444)
        (ROOT / f"{side}.patch").write_text("".join(patches))
        sources[side] = str(dest)
    (ROOT / "source.json").write_text(json.dumps({
        "sources": sources, "baseline": "0742f07c production operator, recorded at e8bd858e",
        "invariants": "four Q groups x16 heads; M128/N256/K256 stage2; same arithmetic/TaskId/sync/early settings",
        "hypothesis": "shorten late Q groups; compressed gather at 24 AIV may overlap Q dequant with less reservation",
        "constraints": "measure concurrent compressor/gather and whole HCA; worker count does not prove utilization",
    }, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
