"""准备仅HCA显式启用的N128流水接入，CSA调用继续默认使用原实现。"""

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
    target = compact.parent / "integrated"
    assert not target.exists()
    shutil.copytree(base, target)
    original = (compact / RELATIVE).read_text()
    start = original.index("@pl.jit.inline\ndef _proj_a_mm_nz(")
    stop = original.index("\n\n@pl.jit.inline\ndef _proj_a_mm_nd", start)
    function = original[start:stop]
    head, rest = function.split("        if A_COL_TILE == 128:\n", 1)
    body = rest[:rest.index("        else:\n")]
    body = "\n".join(line[4:] if line.startswith("    ") else line for line in body.splitlines()) + "\n"
    body = body.replace("        o_r_pad = pl.yield_(stored)\n", "        o_r_pad = stored\n")
    helper = head + body + "    return o_r_pad, pa_tid\n"
    helper = helper.replace("def _proj_a_mm_nz(", "def _proj_a_mm_nz_pipelined(")
    helper = helper.replace(
        '"""Parallelize row and column tiles as upstream; retain Native NZ weights."""',
        '"""HCA的N128 O-A：显式分离L1预取和L0双缓冲，保留组间流水。"""',
    )
    helper = helper.replace(
        "        seed_storage =",
        "        # 动态M的MAD按有效行打包；显式Acc种子必须声明同一紧凑布局。\n        seed_storage =",
    )
    helper = helper.replace("512", "PROJ_A_L1_K_TILE").replace("128", "PROJ_A_L0_K_TILE")
    helper = helper.replace("pl.pipeline(0, 4,", "pl.pipeline(0, PROJ_A_L1_K_TILE // PROJ_A_L0_K_TILE,")
    helper = helper.replace("HCA的NPROJ_A_L0_K_TILE", "HCA的N128")
    before = (base / RELATIVE).read_text()
    after = before.replace(
        "A_K_TILE = 256\n", "A_K_TILE = 256\nPROJ_A_L1_K_TILE = 512\nPROJ_A_L0_K_TILE = 128\n", 1,
    )
    marker = "\n\n@pl.jit.inline\ndef _proj_a_mm_nd("
    after = after.replace(marker, "\n\n" + helper + marker, 1)
    fn_start = after.index("def _decode_o_proj_tp1_parts(")
    fn_end = after.index("\n\n@pl.jit.inline", fn_start)
    function = after[fn_start:fn_end]
    function = function.replace(
        "    A_COL_TILE: pl.constexpr,\n", "    A_COL_TILE: pl.constexpr,\n    PIPELINE_OA: pl.constexpr = False,\n", 1,
    )
    call_start = function.index("            o_r_pad, pa_tid = proj_a_mm(")
    call_end = function.index("\n\n            col_g", call_start)
    call = function[call_start:call_end]
    old_call = "\n".join("    " + line for line in call.splitlines())
    new_call = old_call.replace("= proj_a_mm(", "= _proj_a_mm_nz_pipelined(")
    function = function[:call_start] + (
        "            if PIPELINE_OA and BF16_WEIGHT_NZ and A_COL_TILE == PROJ_A_L0_K_TILE:\n"
        + new_call + "\n            else:\n" + old_call
    ) + function[call_end:]
    after = after[:fn_start] + function + after[fn_end:]
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
    (ROOT / "integrated.patch").write_text("".join(patches))
    manifest["sources"]["integrated"] = str(target)
    manifest["integrated_scope"] = (
        "only HCA opts into PIPELINE_OA=True; shared helper default=False preserves CSA; "
        "only NZ N128 routes to explicit pipeline, N256 and ND retain originals"
    )
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
