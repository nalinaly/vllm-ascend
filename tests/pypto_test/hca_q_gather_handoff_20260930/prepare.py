"""Q24修复预置后，联动控制gather生产者提前解析及四组反量化同步。"""

import ast
import difflib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PREFIX = ROOT.parents[3] / ".cache/hca-q-gather-handoff-5a5b9bfb-20260930"


def main():
    assert not PREFIX.exists()
    old = json.loads((ROOT.parent / "hca_qqueue_groups_20260930/source.json").read_text())
    base = Path(old["sources"]["qr_late"])
    cache = Path("deepseek_v4_flash_hca/decode_compressor_ratio128.py")
    dq = Path("deepseek_v4_flash_hca/q_projection_streamed.py")
    cache_change = (
        cache,
        'name_hint="hca_norm_rope_write", allow_early_resolve=True',
        'name_hint="hca_norm_rope_write", allow_early_resolve=False',
    )
    dq_change = (
        dq,
        '        name_hint="qproj_dequant_rms_nope_rope",\n',
        '        name_hint="qproj_dequant_rms_nope_rope",\n        sync_start=True,\n',
    )
    variants = {"cache_late": [cache_change], "cache_late_dq_sync": [cache_change, dq_change]}
    sources = {"base": str(base), "production": old["sources"]["production"]}
    for side, changes in variants.items():
        target = PREFIX / side
        shutil.copytree(base, target)
        patches = []
        for relative, previous, replacement in changes:
            path = target / relative
            before = path.read_text()
            assert before.count(previous) == 1
            after = before.replace(previous, replacement)
            ast.parse(after)
            path.chmod(0o644)
            path.write_text(after)
            path.chmod(0o444)
            patches.extend(difflib.unified_diff(
                before.splitlines(True), after.splitlines(True), n=0,
                fromfile="a/" + str(relative), tofile="b/" + str(relative),
            ))
        (ROOT / f"{side}.patch").write_text("".join(patches))
        sources[side] = str(target)
    (ROOT / "source.json").write_text(json.dumps({
        "sources": sources,
        "baseline": "four-group Q24 with QR early-resolve disabled; unadopted candidate, not production",
        "basis": "Q spread improved to 24 physical cores, but DQ completion stayed late beside compressed gather",
        "changes": {
            "cache_late": "cache writer stops resolving compressed gather before completion",
            "cache_late_dq_sync": "cache_late plus sync_start for each of four 12-AIV DQ groups",
        },
        "invariants": "same kernels/arithmetic, block counts, buffers, cache/state and dependency edges",
        "limits": "same-card parent comparisons required; do not add independent deltas",
    }, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
