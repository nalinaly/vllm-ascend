"""M4加宽列读取，仍按256列依次计算平方和，避免改变RMS归约规则。"""

import ast
import difflib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    manifest = json.loads((ROOT / "source_m4.json").read_text())
    parent = Path(manifest["candidate"])
    destination = parent.parent / "mix_m4_box8_d512_v3"
    assert not destination.exists(), destination
    shutil.copytree(parent, destination, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    relative = Path("deepseek_v4_flash_hca/hc_pre_fused.py")
    path = destination / relative
    before = path.read_text()
    assert before.count("D_TILE = 256") == 1
    after = before.replace("D_TILE = 256", "D_TILE = 512\nRMS_REDUCE_COLS = 256")
    old = """            y_sq = pl.mul(y_rounded, y_rounded)
            sum_tmp = pl.create_tile([T_TILE, D_TILE], dtype=pl.FP32)
            y_sq_sum = pl.row_sum(y_sq, sum_tmp)
            y_sq_row = pl.reshape(y_sq_sum, [1, T_TILE])
            sq_sum = pl.add(sq_sum, y_sq_row)
"""
    assert after.count(old) == 1
    new = """            y_sq = pl.mul(y_rounded, y_rounded)
            sum_tmp = pl.create_tile([T_TILE, RMS_REDUCE_COLS], dtype=pl.FP32)
            # 读取/混合/归一化用512列，统计仍按原先0,256,512,...次序累加。
            left_sq = pl.tile.extract(y_sq, 0, 0, [T_TILE, RMS_REDUCE_COLS], target_memory=pl.MemorySpace.Vec)
            left_sum = pl.row_sum(left_sq, sum_tmp)
            sq_sum = pl.add(sq_sum, pl.reshape(left_sum, [1, T_TILE]))
            right_sq = pl.tile.extract(
                y_sq, 0, RMS_REDUCE_COLS, [T_TILE, RMS_REDUCE_COLS], target_memory=pl.MemorySpace.Vec,
            )
            right_sum = pl.row_sum(right_sq, sum_tmp)
            sq_sum = pl.add(sq_sum, pl.reshape(right_sum, [1, T_TILE]))
"""
    after = after.replace(old, new)
    ast.parse(after)
    path.chmod(0o644)
    path.write_text(after)
    path.chmod(0o444)
    (ROOT / "m4_wide.patch").write_text("".join(difflib.unified_diff(
        before.splitlines(True), after.splitlines(True),
        fromfile="a/" + str(relative), tofile="b/" + str(relative), n=0,
    )))
    manifest.update(parent_candidate=str(parent), candidate=str(destination),
                    change="M4读取/混合/归一化按512列，平方和仍按原256列顺序，保持stage2与24workers")
    (ROOT / "source_m4_wide.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
