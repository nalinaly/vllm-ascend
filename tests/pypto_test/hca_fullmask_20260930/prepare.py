"""仅对长度与页有效性均已证明的完整压缩块跳过mask算术。"""

import ast
import difflib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
PREFIX = REPO.parent / ".cache/hca-fullmask-v2-2cb8714b-20260930"
RELATIVE = Path("deepseek_v4_flash_hca/decode_sparse_attn_hca.py")


def replace_once(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new, 1)


def main():
    assert not PREFIX.exists(), PREFIX
    baseline = PREFIX / "base"
    candidate = PREFIX / "fullmask"
    shutil.copytree(
        REPO / "vllm_ascend/ops/pypto",
        baseline,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    shutil.copytree(baseline, candidate)
    path = candidate / RELATIVE
    before = path.read_text()
    start = before.index("def _long_sparse_attn_hca_tp1(")
    end = before.index("\n@pl.jit.inline", start)
    body = before[start:end]
    body = replace_once(
        body,
        "    cmp_work_valid = pl.create_tensor([cmp_gather_count, CMP_ATTN_K_TILE], dtype=pl.FP32)",
        "    cmp_work_valid = pl.create_tensor([cmp_gather_count, CMP_ATTN_K_TILE], dtype=pl.FP32)\n"
        "    # 由实际成功搬运的行数证明页有效性，不能只用页表容量或seq_len。\n"
        "    # 每项独占64字节，避免并发scalar write覆盖同cache line的邻项。\n"
        "    mask_line_i32 = 16\n"
        "    cmp_work_rows = pl.create_tensor([cmp_gather_count, mask_line_i32], dtype=pl.INT32)",
    )
    body = replace_once(
        body,
        "            for gather_page in pl.range(CMP_PAGES_PER_WORK):",
        "            copied_rows = 0\n            for gather_page in pl.range(CMP_PAGES_PER_WORK):",
    )
    body = replace_once(
        body,
        "                            gather_page_id = pl.cast(gather_page_i32, pl.INDEX)",
        "                            copied_rows = copied_rows + gather_valid_rows\n"
        "                            gather_page_id = pl.cast(gather_page_i32, pl.INDEX)",
    )
    body = replace_once(
        body,
        "            pl.store(gather_mask, [gather_item, 0], cmp_work_valid)",
        "            pl.store(gather_mask, [gather_item, 0], cmp_work_valid)\n"
        "            pl.write(cmp_work_rows, [gather_item, 0], pl.cast(copied_rows, pl.INT32))",
    )
    old_start = body.index("                            cmp_mask = pl.load(cmp_work_valid")
    old_end = body.index("                        total = pl.row_sum", old_start)
    generic = body[old_start:old_end]
    valid_line = "                            valid_rows = pl.min(ATTN_K_TILE, cmp_rows - cmp_work * ATTN_K_TILE)\n"
    assert generic.count(valid_line) == 1
    generic = generic.replace(valid_line, "").replace("maximum, exponent =", "selected_m, selected_e =")
    generic = "".join("    " + line if line.strip() else line for line in generic.splitlines(True))
    fast = (
        valid_line
        + "                            valid_page_rows = pl.read(\n"
        + "                                cmp_work_rows, [request * cmp_work_count + cmp_work, 0])\n"
        + "                            if valid_rows == ATTN_K_TILE and valid_page_rows == ATTN_K_TILE:\n"
        + "                                full_scaled = pl.mul(score, SOFTMAX_SCALE)\n"
        + "                                full_maximum = pl.row_max(full_scaled, reduce_tmp)\n"
        + "                                full_exponent = pl.exp(pl.row_expand_sub(full_scaled, full_maximum))\n"
        + "                                selected_m, selected_e = pl.yield_(full_maximum, full_exponent)\n"
        + "                            else:\n"
        + generic
        + "                            maximum, exponent = pl.yield_(selected_m, selected_e)\n"
    )
    body = body[:old_start] + fast + body[old_end:]
    after = before[:start] + body + before[end:]
    ast.parse(after)
    path.write_text(after)
    for folder in (baseline, candidate):
        for item in folder.rglob("*.py"):
            item.chmod(0o444)
    (ROOT / "fullmask.patch").write_text(
        "".join(
            difflib.unified_diff(
                before.splitlines(True),
                after.splitlines(True),
                fromfile="a/" + str(RELATIVE),
                tofile="b/" + str(RELATIVE),
            )
        )
    )
    (ROOT / "source.json").write_text(
        json.dumps(
            {
                "baseline_commit": "2cb8714b",
                "baseline": str(baseline),
                "candidate": str(candidate),
                "change": "long-attention full valid compressed blocks skip all-one mask operations; retain fallback",
                "reference": "ops-transformer sparse_attn_sharedkv/arch22 DealBmm1ResBaseBlock + SoftmaxFlashV2Compute",
                "limits": "CPU compile and device results pending; extra scalar metadata may outweigh vector savings",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
