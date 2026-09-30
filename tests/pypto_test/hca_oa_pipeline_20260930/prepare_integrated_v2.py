"""将策略分支放回同一SPMD内，避免JIT对未调用分支的tensor元数据推导限制。"""

import ast
import difflib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RELATIVE = Path("deepseek_v4_flash_dspark_perf/decode_o_proj.py")
HCA = Path("deepseek_v4_flash_hca/o_proj_hc_post.py")


def main():
    manifest_path = ROOT / "source.json"
    manifest = json.loads(manifest_path.read_text())
    base = Path(manifest["sources"]["base"])
    compact = Path(manifest["sources"]["compact"])
    target = compact.parent / "integrated_v2"
    assert not target.exists()
    shutil.copytree(base, target)
    before = (base / RELATIVE).read_text()
    after = (compact / RELATIVE).read_text()
    after = after.replace(
        "A_K_TILE = 256\n",
        "A_K_TILE = 256\nPROJ_A_PIPE_N_TILE = 128\nPROJ_A_L1_K_TILE = 512\nPROJ_A_L0_K_TILE = 128\n", 1,
    )
    begin = after.index("        if A_COL_TILE == 128:")
    end = after.index("        else:", begin)
    body = after[begin:end]
    body = body.replace("if A_COL_TILE == 128:", "if PIPELINE_OA and A_COL_TILE == PROJ_A_PIPE_N_TILE:")
    body = body.replace("[PROJ_A_ROW_TILE, 128], dtype=pl.FP32", "[PROJ_A_ROW_TILE, A_COL_TILE], dtype=pl.FP32")
    body = body.replace("pa_rows, 128)", "pa_rows, A_COL_TILE)")
    body = body.replace("[1, 512, 128]", "[1, 512, A_COL_TILE]")
    body = body.replace("[512, 128]", "[512, A_COL_TILE]")
    body = body.replace("[128, 128]", "[128, A_COL_TILE]")
    body = body.replace("512", "PROJ_A_L1_K_TILE").replace("128", "PROJ_A_L0_K_TILE")
    body = body.replace(
        "for inner, (inner_acc,) in pl.pipeline(0, 4, stage=2, init_values=(outer_acc,)):",
        "for inner, (inner_acc,) in pl.pipeline(\n"
        "                    0, PROJ_A_L1_K_TILE // PROJ_A_L0_K_TILE, stage=2, init_values=(outer_acc,),\n"
        "                ):",
    )
    body = body.replace(
        "            seed_storage =",
        "            # MAD动态M按有效行打包；低层Acc种子显式匹配硬件布局。\n            seed_storage =",
    )
    after = after[:begin] + body + after[end:]
    for name in ("_proj_a_mm_nz", "_proj_a_mm_nd", "_decode_o_proj_tp1_parts"):
        begin = after.index(f"def {name}(")
        end = after.index("\n):", begin)
        signature = after[begin:end]
        signature = signature.replace(
            "    A_COL_TILE: pl.constexpr,",
            "    A_COL_TILE: pl.constexpr,\n    PIPELINE_OA: pl.constexpr = False,", 1,
        )
        after = after[:begin] + signature + after[end:]
    call = "                t_dim, proj_a_rows, heads_dep, A_COL_TILE,\n"
    assert after.count(call) == 1
    after = after.replace(call, "                t_dim, proj_a_rows, heads_dep, A_COL_TILE, PIPELINE_OA,\n", 1)
    hca_before = (base / HCA).read_text()
    hca_after = hca_before.replace(
        "packed, wo_a, wo_b, tokens, heads_dep, MM_ROWS, MM_COLS,",
        "packed, wo_a, wo_b, tokens, heads_dep, MM_ROWS, MM_COLS, PIPELINE_OA=True,", 1,
    )
    patches = []
    for relative, old, new in ((RELATIVE, before, after), (HCA, hca_before, hca_after)):
        ast.parse(new)
        path = target / relative
        path.chmod(0o644)
        path.write_text(new)
        path.chmod(0o444)
        patches.append("".join(difflib.unified_diff(
            old.splitlines(True), new.splitlines(True), n=0,
            fromfile="a/" + str(relative), tofile="b/" + str(relative),
        )))
    (ROOT / "integrated_v2.patch").write_text("".join(patches))
    manifest["sources"]["integrated_v2"] = str(target)
    manifest["integrated_failure"] = "conditional inline call: missing inferred tensor metadata for o_r_pad"
    manifest["integrated_v2_scope"] = (
        "same kernel body/strategy as compact, constexpr flag inside SPMD; "
        "HCA=True, CSA default=False; ND/N256 unchanged"
    )
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
