"""将四次L0B抽取/计算放入列循环，避免K流水同时保活八份Right。"""

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
    parent = Path(metadata["sources"]["pvpipe"])
    target = parent.parent / "pvpipe_nested"
    assert not target.exists()
    shutil.copytree(parent, target)
    before = (parent / RELATIVE).read_text()
    start = before.index("                        right_update0 = pl.tile.extract(part_kv,")
    end = before.index("                    pl.store(done0,", start)
    body = '''                        # K块加载仍按stage2预取；N列循环顺序消费L0B，
                        # 避免四列×两级的全部Right操作数同时驻留。
                        for column, (col0, col1, col2, col3) in pl.range(
                            4, init_values=(acc0, acc1, acc2, acc3),
                        ):
                            right_part = pl.tile.extract(
                                part_kv, 0, column * PV_N_TILE, [ATTN_K_TILE, PV_N_TILE],
                                target_memory=pl.MemorySpace.Right,
                            )
                            if column == 0:
                                next0, next1, next2, next3 = pl.yield_(
                                    pl.tile.matmul_acc(col0, left_part, right_part), col1, col2, col3,
                                )
                            elif column == 1:
                                next0, next1, next2, next3 = pl.yield_(
                                    col0, pl.tile.matmul_acc(col1, left_part, right_part), col2, col3,
                                )
                            elif column == 2:
                                next0, next1, next2, next3 = pl.yield_(
                                    col0, col1, pl.tile.matmul_acc(col2, left_part, right_part), col3,
                                )
                            else:
                                next0, next1, next2, next3 = pl.yield_(
                                    col0, col1, col2, pl.tile.matmul_acc(col3, left_part, right_part),
                                )
                            column0, column1, column2, column3 = pl.yield_(next0, next1, next2, next3)
                        done0, done1, done2, done3 = pl.yield_(column0, column1, column2, column3)
'''
    after = before[:start] + body + before[end:]
    ast.parse(after)
    path = target / RELATIVE
    path.chmod(0o644)
    path.write_text(after)
    path.chmod(0o444)
    (ROOT / "pvpipe_nested.patch").write_text("".join(difflib.unified_diff(
        before.splitlines(True), after.splitlines(True), n=0,
        fromfile="a/" + str(RELATIVE), tofile="b/" + str(RELATIVE),
    )))
    metadata["sources"]["pvpipe_nested"] = str(target)
    metadata["pvpipe_failure"] = "Right buffer 262144 > 65536 bytes: K pipeline hoists all four N-column extracts"
    metadata["nested_change"] = "K stage2 prefetch with an inner four-column loop consuming one Right tile at a time"
    source_file.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
