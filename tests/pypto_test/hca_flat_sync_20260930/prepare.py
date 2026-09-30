"""在flat Gather基线上检查Q反量化与压缩gather的同步联动。"""

import ast
import difflib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
PREFIX = REPO.parent / ".cache/hca-flat-sync-ad0e6bbe-20260930"


def main():
    assert not PREFIX.exists(), PREFIX
    sources = {}
    for side in ("base", "dq_sync", "both_sync"):
        destination = PREFIX / side
        shutil.copytree(REPO / "vllm_ascend/ops/pypto", destination,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        sources[side] = str(destination)
        patches = []
        if side != "base":
            relative = Path("deepseek_v4_flash_dspark_perf/qkv_proj_rope.py")
            path = destination / relative
            before = path.read_text()
            old = '        name_hint="qproj_dequant_rms_nope_rope",\n        allow_early_resolve=True,'
            assert before.count(old) == 1
            after = before.replace(old, old + "\n        sync_start=True,")
            ast.parse(after)
            path.write_text(after)
            patches.extend(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
                                               fromfile="a/" + str(relative), tofile="b/" + str(relative), n=0))
        if side == "both_sync":
            relative = Path("deepseek_v4_flash_hca/decode_sparse_attn_hca.py")
            path = destination / relative
            before = path.read_text()
            prefix, long_body = before.split("def _long_sparse_attn_hca_tp1(", 1)
            old = 'name_hint="hca_cmp_work_gather", allow_early_resolve=True, deps=[cmp_cache_ready_dep]'
            assert long_body.count(old) == 1
            after = prefix + "def _long_sparse_attn_hca_tp1(" + long_body.replace(
                old, old + ", sync_start=True",
            )
            ast.parse(after)
            path.write_text(after)
            patches.extend(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
                                               fromfile="a/" + str(relative), tofile="b/" + str(relative), n=0))
        for path in destination.rglob("*.py"):
            path.chmod(0o444)
        if patches:
            (ROOT / f"{side}.patch").write_text("".join(patches))
    (ROOT / "source.json").write_text(json.dumps({
        "baseline_commit": "ad0e6bbe", "sources": sources,
        "change": "Q dequant sync_start alone, then linked with long compressed gather sync_start",
        "invariants": "same arithmetic, full flat Gather, worker counts, head pipeline, dependencies and state layout",
    }, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
