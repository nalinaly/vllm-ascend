"""单独评估Q投影提前解析；保持20核、依赖与核内分块。"""

import ast
import difflib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
PREFIX = REPO.parent / ".cache/hca-qearly-bef9f7fa-20260930"
RELATIVE = Path("deepseek_v4_flash_dspark/q_projection.py")


def main():
    assert not PREFIX.exists(), PREFIX
    baseline, candidate = PREFIX / "base", PREFIX / "qearly"
    source = REPO.parent / ".cache/hca-ob-al1-bef9f7fa-20260930/base"
    shutil.copytree(source, baseline, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copytree(baseline, candidate)
    path = candidate / RELATIVE
    before = path.read_text()
    old = '    with pl.spmd(QPROJ_WORKERS, name_hint="qproj_matmul", deps=[qproj_dep]) as qproj_tid:'
    new = (
        '    with pl.spmd(QPROJ_WORKERS, name_hint="qproj_matmul", '
        "deps=[qproj_dep], allow_early_resolve=True) as qproj_tid:"
    )
    assert before.count(old) == 1
    after = before.replace(old, new)
    ast.parse(after)
    path.chmod(0o644)
    path.write_text(after)
    path.chmod(0o444)
    (ROOT / "qearly.patch").write_text(
        "".join(
            difflib.unified_diff(
                before.splitlines(True),
                after.splitlines(True),
                fromfile="a/" + str(RELATIVE),
                tofile="b/" + str(RELATIVE),
            )
        )
    )
    (ROOT / "source.json").write_text(
        json.dumps(
            {
                "baseline_commit": "bef9f7fa",
                "baseline": str(baseline),
                "candidate": str(candidate),
                "change": "NZ Q projection allow_early_resolve=True; keep 20 workers and original dependencies",
                "interaction": "observe compressor overlap, Q completion, dequant and attention starts",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
