"""整宽量化按8段1024列归约，保留相同全行amax，避免1024行窄列归约。"""

import ast
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    file = ROOT / "source.json"
    manifest = json.loads(file.read_text())
    source = Path(manifest["sources"]["full_m96"])
    dest = source.with_name("full_m96_rowmax")
    assert not dest.exists()
    shutil.copytree(source, dest, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    kernel = dest / "deepseek_v4_flash_hca/whole_o_projection.py"
    before = kernel.read_text()
    after = before.replace(
        "                lanes = pl.reshape(pl.abs(value), [FULL_WIDTH // 8, 8])\n"
        "                maxima = pl.col_max(lanes)",
        "                lanes = pl.reshape(pl.abs(value), [8, FULL_WIDTH // 8])\n"
        "                lane_max = pl.row_max(lanes, pl.create_tile([8, FULL_WIDTH // 8], dtype=pl.FP32))\n"
        "                maxima = pl.reshape(lane_max, [1, 8])",
    )
    assert after != before
    ast.parse(after)
    kernel.chmod(0o644)
    kernel.write_text(after)
    kernel.chmod(0o444)
    manifest["sources"]["full_m96_rowmax"] = str(dest)
    manifest["full_m96_rowmax"] = (
        "same full-row max via row_max on8x1024, then8-lane reduction; no arithmetic relaxation"
    )
    file.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
