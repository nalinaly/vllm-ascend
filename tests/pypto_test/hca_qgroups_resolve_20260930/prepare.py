"""在累计max候选上实测四组Q的提前解析，以及与消费者延后解析的组合增量。"""

import ast
import difflib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PREFIX = ROOT.parents[3] / ".cache/hca-qgroups-resolve-online-20260930"


def main():
    assert not PREFIX.exists()
    upstream = json.loads((ROOT.parent / "hca_online_softmax_20260930/source.json").read_text())
    source = Path(upstream["sources"]["online"])
    sources = {"base": str(source)}
    relative = Path("deepseek_v4_flash_hca/q_projection_streamed.py")
    before = (source / relative).read_text()
    for side in ("cube_early", "cube_early_dq_late"):
        dest = PREFIX / side
        shutil.copytree(source, dest)
        after = before
        for group in range(4):
            line = f'with pl.spmd(STREAM_CUBE_WORKERS, name_hint="hca_qb_stream", deps=[qproj_dep, q_ready[{group}]]):'
            assert after.count(line) == 1
            after = after.replace(line, line[:-2] + ", allow_early_resolve=True):")
        if side == "cube_early_dq_late":
            line = '        name_hint="qproj_dequant_rms_nope_rope",\n        allow_early_resolve=True,'
            assert after.count(line) == 1
            after = after.replace(line, line.replace("True", "False"))
        ast.parse(after)
        path = dest / relative
        path.chmod(0o644)
        path.write_text(after)
        for file in dest.rglob("*.py"):
            file.chmod(0o444)
        sources[side] = str(dest)
        (ROOT / f"{side}.patch").write_text("".join(difflib.unified_diff(
            before.splitlines(True), after.splitlines(True), fromfile="a/" + str(relative), tofile="b/" + str(relative),
        )))
    (ROOT / "source.json").write_text(json.dumps({
        "sources": sources, "baseline": "0742f07c + frozen online-softmax candidate (not production)",
        "invariants": "four x4 Cube/x12 AIV, unchanged arithmetic, tiling, sync and TaskId",
        "hypothesis": "publish Q completion early; DQ reservation may compete with compressed gather/Attention",
        "limits": "same-window combined increment; never sum separate experiments' deltas",
    }, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
