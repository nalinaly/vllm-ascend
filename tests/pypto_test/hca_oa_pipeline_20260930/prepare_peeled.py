"""由首个真实matmul推导窄行累加器，移除compact内部参数与循环内初始化分支。"""

import ast
import difflib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RELATIVE = Path("deepseek_v4_flash_dspark_perf/decode_o_proj.py")

PREFIX = '''            first_lhs = pl.load(o_packed, [pa_src0, 0], [PROJ_A_ROW_TILE, 512],
                                    valid_shape=[pa_rows, 512], target_memory=pl.MemorySpace.Mat)
            first_rhs_group = pl.load(wo_a, [g, 0, n0], [1, 512, 128], target_memory=pl.MemorySpace.Mat)
            first_rhs = pl.reshape(first_rhs_group, [512, 128])
            first_left = pl.tile.extract(first_lhs, 0, 0, [PROJ_A_ROW_TILE, 128],
                                         target_memory=pl.MemorySpace.Left)
            first_right = pl.tile.extract(first_rhs, 0, 0, [128, 128], target_memory=pl.MemorySpace.Right)
            first_acc = pl.tile.matmul(first_left, first_right)
            for prefix_k, (prefix_acc,) in pl.pipeline(1, 4, stage=2, init_values=(first_acc,)):
                prefix_left = pl.tile.extract(first_lhs, 0, prefix_k * 128, [PROJ_A_ROW_TILE, 128],
                                              target_memory=pl.MemorySpace.Left)
                prefix_right = pl.tile.extract(first_rhs, prefix_k * 128, 0, [128, 128],
                                               target_memory=pl.MemorySpace.Right)
                prefix_updated = pl.tile.matmul_acc(prefix_acc, prefix_left, prefix_right)
                seed = pl.yield_(prefix_updated)
'''


def main():
    manifest_path = ROOT / "source.json"
    manifest = json.loads(manifest_path.read_text())
    parent = Path(manifest["sources"]["compact"])
    target = parent.parent / "peeled"
    assert not target.exists()
    shutil.copytree(parent, target)
    path = target / RELATIVE
    before = path.read_text()
    start = before.index("            seed_storage =")
    end = before.index("            stored = pl.store(outer_done", start)
    body = before[start:end]
    loop = body[body.index("            for outer,"):]
    loop = loop.replace("pl.pipeline(0, O_GROUP_IN // 512", "pl.pipeline(1, O_GROUP_IN // 512")
    loop = loop.replace(
        "pl.tile.matmul_acc(inner_acc, lhs_part, rhs_part,\n"
        "                                                 init_cond=(outer == 0 and inner == 0))",
        "pl.tile.matmul_acc(inner_acc, lhs_part, rhs_part)",
    )
    after = before[:start] + PREFIX + loop + before[end:]
    ast.parse(after)
    path.chmod(0o644)
    path.write_text(after)
    path.chmod(0o444)
    (ROOT / "peeled.patch").write_text("".join(difflib.unified_diff(
        before.splitlines(True), after.splitlines(True), n=0,
        fromfile="a/" + str(RELATIVE), tofile="b/" + str(RELATIVE),
    )))
    manifest["sources"]["peeled"] = str(target)
    manifest["peeled_change"] = (
        "first real K128 tile.matmul derives compact Acc; finish first K512 then pipeline remaining K512 chunks; "
        "same arithmetic order and no extra GM loads, no compact=True and no runtime init_cond"
    )
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
