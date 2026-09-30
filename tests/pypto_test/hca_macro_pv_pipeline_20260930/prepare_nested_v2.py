"""使用四个显式if/else产出累加器，避开elif嵌套yield的SSA限制。"""

import ast
import difflib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RELATIVE = Path("deepseek_v4_flash_hca/decode_sparse_attn_hca.py")


def main():
    source_file = ROOT / "source.json"
    metadata = json.loads(source_file.read_text())
    parent = Path(metadata["sources"]["pvpipe_nested"])
    target = parent.parent / "pvpipe_nested_v2"
    assert not target.exists()
    shutil.copytree(parent, target)
    before = (parent / RELATIVE).read_text()
    start = before.index("                            if column == 0:")
    end = before.index("                        done0, done1, done2, done3 =", start)
    body = ""
    for index in range(4):
        body += (
            f"                            if column == {index}:\n"
            f"                                next{index} = pl.yield_(pl.tile.matmul_acc(\n"
            f"                                    col{index}, left_part, right_part,\n"
            "                                ))\n"
            "                            else:\n"
            f"                                next{index} = pl.yield_(col{index})\n"
        )
    body += "                            column0, column1, column2, column3 = pl.yield_(next0, next1, next2, next3)\n"
    after = before[:start] + body + before[end:]
    ast.parse(after)
    path = target / RELATIVE
    path.chmod(0o644)
    path.write_text(after)
    path.chmod(0o444)
    (ROOT / "pvpipe_nested_v2.patch").write_text("".join(difflib.unified_diff(
        before.splitlines(True), after.splitlines(True), n=0,
        fromfile="a/" + str(RELATIVE), tofile="b/" + str(RELATIVE),
    )))
    metadata["sources"]["pvpipe_nested_v2"] = str(target)
    metadata["nested_failure"] = "ConvertToSSA: elif with tuple yield leaves outer else without a final yield"
    metadata["nested_v2_change"] = (
        "four explicit single-result if/else branches; one accumulator updated for each N column"
    )
    source_file.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
