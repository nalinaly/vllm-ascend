"""四块Q常驻配合N128外层流水；实际生成L0A/B双缓冲，L0C仍为单份。"""

import ast
import difflib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
RELATIVE = Path("deepseek_v4_flash_dspark/q_projection.py")


def main():
    source = json.loads((ROOT / "source_slices.json").read_text())
    baseline = Path(source["baseline"])
    candidate = REPO.parent / ".cache/hca-q-npipe-a7a9f314-20260930/npipe"
    assert not candidate.exists(), candidate
    shutil.copytree(Path(source["candidate"]), candidate)
    path = candidate / RELATIVE
    before = path.read_text()
    old = "QPROJ_MM_N_TILE = 256 if QUANT_WEIGHT_NZ else 512"
    assert before.count(old) == 1
    after = before.replace(old, "QPROJ_MM_N_TILE = 128 if QUANT_WEIGHT_NZ else 512")
    old = "for qproj_round in pl.range(0, (QPROJ_N_BLOCKS - qproj_worker + QPROJ_WORKERS - 1) // QPROJ_WORKERS):"
    new = (
        "for qproj_round in pl.pipeline(\n"
        "                0, (QPROJ_N_BLOCKS - qproj_worker + QPROJ_WORKERS - 1) // QPROJ_WORKERS, stage=2,\n"
        "            ):"
    )
    assert after.count(old) == 1
    after = after.replace(old, new)
    ast.parse(after)
    path.chmod(0o644)
    path.write_text(after)
    path.chmod(0o444)
    (ROOT / "npipe.patch").write_text(
        "".join(
            difflib.unified_diff(
                (baseline / RELATIVE).read_text().splitlines(True),
                after.splitlines(True),
                fromfile="a/" + str(RELATIVE),
                tofile="b/" + str(RELATIVE),
                n=0,
            )
        )
    )
    (ROOT / "source_npipe.json").write_text(
        json.dumps(
            {
                "baseline_commit": "a7a9f314",
                "baseline": str(baseline),
                "candidate": str(candidate),
                "parent_candidate": source["candidate"],
                "change": "four resident A tiles plus N128 outer pipeline stage2; inspect actual L0 buffers before NPU",
                "invariants": "INT8 arithmetic, M128/K256, 20 workers, task dependencies and cache policy",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
