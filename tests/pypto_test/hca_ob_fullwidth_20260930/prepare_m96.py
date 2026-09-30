"""完整K候选改M96/N128：减少长B16下重复权重读取，仍比较真实全段。"""

import ast
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    file = ROOT / "source.json"
    manifest = json.loads(file.read_text())
    source = Path(manifest["sources"]["full"])
    dest = source.with_name("full_m96")
    assert not dest.exists()
    shutil.copytree(source, dest, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    kernel = dest / "deepseek_v4_flash_hca/whole_o_projection.py"
    content = kernel.read_text().replace("FULL_M = 32", "FULL_M = 96").replace("FULL_N = 256", "FULL_N = 128")
    content = content.replace(
        "    padded = (tokens + ROW_TILE - 1) // ROW_TILE * ROW_TILE",
        "    padded = pl.max((tokens + ROW_TILE - 1) // ROW_TILE * ROW_TILE, "
        "(tokens + FULL_M - 1) // FULL_M * FULL_M)",
    )
    ast.parse(content)
    kernel.chmod(0o644)
    kernel.write_text(content)
    kernel.chmod(0o444)
    manifest["sources"]["full_m96"] = str(dest)
    manifest["full_m96"] = "M96/N128/K512; one M block for96 tokens,32 Cube workers; unchanged arithmetic"
    file.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
