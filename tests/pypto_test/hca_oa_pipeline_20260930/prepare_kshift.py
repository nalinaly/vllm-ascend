"""参考Native NZ路径的K-shift，按N块错开O-A的K512读取起点。"""

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
    source = Path(manifest["sources"]["integrated_v2"])
    target = source.parent / "kshift"
    if target.exists():
        raise FileExistsError(target)
    shutil.copytree(source, target)
    before = (source / RELATIVE).read_text()
    old = "                outer_k = outer * PROJ_A_L1_K_TILE\n"
    new = (
        "                # 每个N块循环遍历全部K段，只错开起点；FP32累加次序会改变。\n"
        "                outer_k = ((outer + nf) % (O_GROUP_IN // PROJ_A_L1_K_TILE)) * PROJ_A_L1_K_TILE\n"
    )
    assert before.count(old) == 1
    after = before.replace(old, new)
    ast.parse(after)
    path = target / RELATIVE
    path.chmod(0o644)
    path.write_text(after)
    path.chmod(0o444)
    (ROOT / "kshift.patch").write_text("".join(difflib.unified_diff(
        before.splitlines(True), after.splitlines(True), n=0,
        fromfile="a/" + str(RELATIVE), tofile="b/" + str(RELATIVE),
    )))
    manifest["sources"]["kshift"] = str(target)
    manifest["kshift_change"] = (
        "relative to integrated_v2, rotate K512 start by N-block index; all 8 K blocks exactly once; "
        "FP32 reduction order changes, output requires arithmetic diagnosis and later model token acceptance"
    )
    manifest["kshift_reference"] = (
        "ops-nn/matmul/transpose_batch_mat_mul/op_kernel/transpose_batch_mat_mul.cpp:71-88 NZ MM_CFG_K_SHIFT; "
        "strategy reference only, not proof of installed binary or equivalent scheduling"
    )
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
