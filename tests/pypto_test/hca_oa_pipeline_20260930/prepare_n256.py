"""N256的显式L1 K256/L0 K64流水，单独核对分块与缓冲联动。"""

import ast
import difflib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RELATIVE = Path("deepseek_v4_flash_dspark_perf/decode_o_proj.py")


def main():
    manifest_path = ROOT / "source.json"
    manifest = json.loads(manifest_path.read_text())
    parent = Path(manifest["sources"]["compact"])
    target = parent.parent / "n256"
    assert not target.exists()
    shutil.copytree(parent, target)
    path = target / RELATIVE
    before = path.read_text()
    start = before.index("        if A_COL_TILE == 128:")
    end = before.index("        else:", start)
    body = before[start:end]
    replacements = (
        ("if A_COL_TILE == 128:", "if A_COL_TILE == 256:"),
        ("[PROJ_A_ROW_TILE, 128], dtype=pl.FP32", "[PROJ_A_ROW_TILE, 256], dtype=pl.FP32"),
        ("set_validshape(seed_storage, pa_rows, 128)", "set_validshape(seed_storage, pa_rows, 256)"),
        ("O_GROUP_IN // 512", "O_GROUP_IN // 256"),
        ("outer * 512", "outer * 256"),
        ("[PROJ_A_ROW_TILE, 512]", "[PROJ_A_ROW_TILE, 256]"),
        ("valid_shape=[pa_rows, 512]", "valid_shape=[pa_rows, 256]"),
        ("[1, 512, 128]", "[1, 256, 256]"),
        ("[512, 128]", "[256, 256]"),
        ("inner * 128", "inner * 64"),
        ("[PROJ_A_ROW_TILE, 128]", "[PROJ_A_ROW_TILE, 64]"),
        ("[128, 128]", "[64, 256]"),
    )
    for old, new in replacements:
        assert old in body, old
        body = body.replace(old, new)
    after = before[:start] + body + before[end:]
    after = after.replace("PROJ_A_MM_N_TILE = 128", "PROJ_A_MM_N_TILE = 256")
    ast.parse(after)
    path.chmod(0o644)
    path.write_text(after)
    path.chmod(0o444)
    (ROOT / "n256.patch").write_text("".join(difflib.unified_diff(
        before.splitlines(True), after.splitlines(True), n=0,
        fromfile="a/" + str(RELATIVE), tofile="b/" + str(RELATIVE),
    )))
    manifest["sources"]["n256"] = str(target)
    manifest["n256_change"] = (
        "O-A N256 / L1 K256 / explicit L0 K64 stage2; B16 goes 64->32 workers, "
        "halves repeated input reads across N blocks; same groups/quant/O-B dependencies"
    )
    manifest["n256_scope"] = (
        "private HCA package; both small/medium and large O-A branches use explicit N256 pipeline; "
        "K64 microstep changes require actual numerical comparison"
    )
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
