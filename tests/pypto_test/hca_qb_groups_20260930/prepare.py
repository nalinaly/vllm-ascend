"""在已消除reshape假依赖的Q分组实现上调整生产者与消费者粒度。"""

import ast
import difflib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
PREFIX = REPO.parent / ".cache/hca-qb-groups-f14d8d90-20260930"


def main():
    assert not PREFIX.exists()
    previous = json.loads((ROOT.parent / "hca_qb_mixed_20260930/source.json").read_text())
    source = Path(previous["sources"]["streamed4_flat"])
    base = PREFIX / "base"
    shutil.copytree(REPO / "vllm_ascend/ops/pypto", base, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    sources = {"base": str(base), "flat4": str(source)}
    relative = Path("deepseek_v4_flash_hca/q_projection_streamed.py")
    for side in ("flat2", "flat4_c16"):
        destination = PREFIX / side
        shutil.copytree(source, destination, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        file = destination / relative
        before = file.read_text()
        after = before
        if side == "flat2":
            after = after.replace("STREAM_GROUPS = 4", "STREAM_GROUPS = 2")
            start = after.index("        with pl.scope():\n            head_base = 32\n")
            end = after.rindex("    return q\n")
            after = after[:start] + after[end:]
            after = after.replace("head_base = 16", "head_base = 32")
            after = after.replace("col = 8192 + col_local", "col = 16384 + col_local")
        else:
            after = after.replace(
                "STREAM_CUBE_WORKERS = 20 // STREAM_GROUPS", "STREAM_CUBE_WORKERS = 16 // STREAM_GROUPS"
            )
        assert after != before
        ast.parse(after)
        file.chmod(0o644)
        file.write_text(after)
        file.chmod(0o444)
        sources[side] = str(destination)
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
    for file in base.rglob("*.py"):
        file.chmod(0o444)
    manifest = {
        "baseline_commit": "f14d8d90 (production operator unchanged from 9015d45a)",
        "parent_experiment": str(ROOT.parent / "hca_qb_mixed_20260930/source.json"),
        "sources": sources,
        "scope": "HCA private packages only; correct 2D Q manual_dep and explicit producer edges",
        "changes": {
            "flat2": "two groups, each 10 Cube and 24 AIV; four-slot dependency array keeps two invalid entries",
            "flat4_c16": "four groups, each 4 Cube and 12 AIV; preserve integer/RMS/RoPE arithmetic",
        },
    }
    (ROOT / "source.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
