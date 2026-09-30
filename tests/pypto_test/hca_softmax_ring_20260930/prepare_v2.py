"""用显式Vec extract保留环形槽偏移，避免动态slice经reshape后丢失偏移。"""

import ast
import difflib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
PREFIX = REPO.parent / ".cache/hca-softmax-ring-v2-bef9f7fa-20260930"
RELATIVE = Path("deepseek_v4_flash_hca/decode_sparse_attn_hca.py")


def main():
    old_source = json.loads((ROOT / "source.json").read_text())
    assert not PREFIX.exists(), PREFIX
    baseline, candidate = PREFIX / "base", PREFIX / "ring"
    shutil.copytree(old_source["baseline"], baseline, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copytree(old_source["candidate"], candidate, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    path = candidate / RELATIVE
    before = path.read_text()
    after = before
    for name, ring in (("out_m", "max"), ("out_l", "sum")):
        old = (
            f"                        {name} = pl.reshape(pl.tile.slice(\n"
            f"                            updated_{ring}_ring, [1, H // 2], [out_work % QK_TRANSFER_SLOTS, 0],\n"
            "                        ), [H // 2, 1])\n"
        )
        new = (
            f"                        {name} = pl.reshape(pl.tile.extract(\n"
            f"                            updated_{ring}_ring, out_work % QK_TRANSFER_SLOTS, 0, [1, H // 2],\n"
            "                            target_memory=pl.MemorySpace.Vec,\n"
            "                        ), [H // 2, 1])\n"
        )
        assert after.count(old) == 1
        after = after.replace(old, new)
    ast.parse(after)
    path.chmod(0o644)
    path.write_text(after)
    path.chmod(0o444)
    (ROOT / "extract.patch").write_text(
        "".join(
            difflib.unified_diff(
                before.splitlines(True),
                after.splitlines(True),
                fromfile="v1/" + str(RELATIVE),
                tofile="v2/" + str(RELATIVE),
            )
        )
    )
    (ROOT / "source_v2.json").write_text(
        json.dumps(
            {
                **old_source,
                "baseline": str(baseline),
                "candidate": str(candidate),
                "v2": "explicit Vec extract with dynamic row offset; no compiler or runtime edits",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
