"""分别限制Q反量化提前占核，以及联动开启Q投影提前解析；保持数据和任务依赖。"""

import ast
import difflib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
PREFIX = REPO.parent / ".cache/hca-qresolve-a7a9f314-20260930"


def main():
    assert not PREFIX.exists(), PREFIX
    sources = {}
    for side in ("base", "late_dq", "early_q_late_dq"):
        destination = PREFIX / side
        shutil.copytree(
            REPO / "vllm_ascend/ops/pypto", destination, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
        )
        sources[side] = str(destination)
        edits = {}
        if side != "base":
            edits[Path("deepseek_v4_flash_dspark_perf/qkv_proj_rope.py")] = (
                '        name_hint="qproj_dequant_rms_nope_rope",\n        allow_early_resolve=True,',
                '        name_hint="qproj_dequant_rms_nope_rope",\n        allow_early_resolve=False,',
            )
        if side == "early_q_late_dq":
            edits[Path("deepseek_v4_flash_dspark/q_projection.py")] = (
                '    with pl.spmd(QPROJ_WORKERS, name_hint="qproj_matmul", deps=[qproj_dep]) as qproj_tid:',
                '    with pl.spmd(QPROJ_WORKERS, name_hint="qproj_matmul", '
                "deps=[qproj_dep], allow_early_resolve=True) as qproj_tid:",
            )
        patches = []
        for relative, (old, new) in edits.items():
            path = destination / relative
            before = path.read_text()
            assert before.count(old) == 1, relative
            after = before.replace(old, new)
            ast.parse(after)
            path.write_text(after)
            patches.extend(
                difflib.unified_diff(
                    before.splitlines(True),
                    after.splitlines(True),
                    fromfile="a/" + str(relative),
                    tofile="b/" + str(relative),
                    n=0,
                )
            )
        for path in destination.rglob("*.py"):
            path.chmod(0o444)
        if patches:
            (ROOT / f"{side}.patch").write_text("".join(patches))
    (ROOT / "source.json").write_text(
        json.dumps(
            {
                "baseline_commit": "a7a9f314",
                "sources": sources,
                "change": "disable Q dequant early resolution; separately combine with early Q projection",
                "invariants": "same arithmetic, worker counts, tiling and task dependencies; no toolchain edits",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
