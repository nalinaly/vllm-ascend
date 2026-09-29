"""当前四组Q上的Cube启动同步与Cube/Vector联动，保持算术不变。"""

import ast
import difflib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
PREFIX = REPO.parent / ".cache/hca-qgroups-sync-0742f07c-20260930"


def main():
    assert not PREFIX.exists()
    base = PREFIX / "base"
    shutil.copytree(REPO / "vllm_ascend/ops/pypto", base, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    relative = Path("deepseek_v4_flash_hca/q_projection_streamed.py")
    before = (base / relative).read_text()
    sources = {"base": str(base)}
    for side in ("cube_sync", "both_sync"):
        dest = PREFIX / side
        shutil.copytree(base, dest)
        after = before
        for group in range(4):
            line = f'with pl.spmd(STREAM_CUBE_WORKERS, name_hint="hca_qb_stream", deps=[qproj_dep, q_ready[{group}]]):'
            assert after.count(line) == 1
            after = after.replace(line, line[:-2] + ", sync_start=True):")
        if side == "both_sync":
            after = after.replace(
                '        name_hint="qproj_dequant_rms_nope_rope",\n        allow_early_resolve=True,',
                '        name_hint="qproj_dequant_rms_nope_rope",\n        allow_early_resolve=True, sync_start=True,',
            )
        ast.parse(after)
        (dest / relative).write_text(after)
        sources[side] = str(dest)
        (ROOT / f"{side}.patch").write_text("".join(difflib.unified_diff(
            before.splitlines(True), after.splitlines(True), fromfile="a/" + str(relative), tofile="b/" + str(relative),
        )))
    for file in PREFIX.rglob("*.py"):
        file.chmod(0o444)
    (ROOT / "source.json").write_text(json.dumps({
        "baseline_commit": "0742f07c", "sources": sources,
        "invariants": "four groups x4 Cube/x12 AIV; unchanged tiling, arithmetic, TaskId and allow_early_resolve",
        "hypothesis": ("groupwise start coordination; evaluate gather overlap and attention readiness; "
                       "measure complete span"),
    }, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
