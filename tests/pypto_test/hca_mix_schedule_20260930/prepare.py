"""在M4/D512基线上检查mix与Sinkhorn同步启动联动，算术与分工不变。"""

import ast
import difflib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
PREFIX = REPO.parent / ".cache/hca-mix-schedule-9c2ede34-20260930"


def main():
    assert not PREFIX.exists(), PREFIX
    sources = {}
    relative = Path("deepseek_v4_flash_hca/hc_pre_fused.py")
    for side in ("base", "mix_sync", "both_sync"):
        destination = PREFIX / side
        shutil.copytree(REPO / "vllm_ascend/ops/pypto", destination,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        sources[side] = str(destination)
        path = destination / relative
        before = path.read_text()
        after = before
        if side != "base":
            old = 'name_hint="mix_x_rms_norm", allow_early_resolve=True'
            assert after.count(old) == 1
            after = after.replace(old, old + ", sync_start=True")
        if side == "both_sync":
            old = 'name_hint="comb_sinkhorn", allow_early_resolve=True'
            assert after.count(old) == 1
            after = after.replace(old, old + ", sync_start=True")
        ast.parse(after)
        path.write_text(after)
        if side != "base":
            (ROOT / f"{side}.patch").write_text("".join(difflib.unified_diff(
                before.splitlines(True), after.splitlines(True),
                fromfile="a/" + str(relative), tofile="b/" + str(relative), n=0,
            )))
        for source in destination.rglob("*.py"):
            source.chmod(0o444)
    (ROOT / "source.json").write_text(json.dumps({
        "baseline_commit": "9c2ede34", "sources": sources,
        "change": "mix sync_start alone, then linked with concurrent Sinkhorn sync_start",
        "invariants": "M4/D512 with D256 reductions, worker counts, stage2, arithmetic, buffer layout and dependencies",
    }, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
