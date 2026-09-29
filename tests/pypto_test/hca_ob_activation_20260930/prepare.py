"""只移植CSA已验证的O-B激活复用函数，不覆盖任务图及HCA收尾。"""

import ast
import difflib
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
PREFIX = REPO.parent / ".cache/hca-ob-al1-bef9f7fa-20260930"
RELATIVE = Path("deepseek_v4_flash_dspark_perf/decode_o_proj.py")
CSA_COMMIT = "5c66a59b"


def kernel_span(text):
    node = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == "_proj_b_mm_nz_kernel")
    lines = text.splitlines(True)
    start = min([node.lineno] + [d.lineno for d in node.decorator_list]) - 1
    return lines, start, node.end_lineno


def main():
    assert not PREFIX.exists(), PREFIX
    baseline, candidate = PREFIX / "base", PREFIX / "al1"
    shutil.copytree(REPO / "vllm_ascend/ops/pypto", baseline, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copytree(baseline, candidate)
    path = candidate / RELATIVE
    before = path.read_text()
    csa = subprocess.check_output(
        ["git", "show", f"{CSA_COMMIT}:vllm_ascend/ops/pypto/{RELATIVE}"],
        cwd=REPO.parent / "vllm-ascend-dsv4-pto-0251rc1",
        text=True,
    )
    src, left, right = kernel_span(csa)
    dst, start, end = kernel_span(before)
    after = "".join(dst[:start] + src[left:right] + dst[end:])
    ast.parse(after)
    path.write_text(after)
    for folder in (baseline, candidate):
        for item in folder.rglob("*.py"):
            item.chmod(0o444)
    (ROOT / "al1.patch").write_text(
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
                "baseline_commit": "bef9f7fa",
                "baseline": str(baseline),
                "candidate": str(candidate),
                "csa_reference_commit": CSA_COMMIT,
                "change": "ROW<=96 NZ O-B retains activation in L1 across two N256 outputs; preserve K128 L0 lowering",
                "limits": "only this incore function imported; task graph, quantization, ROW128 and ND unchanged",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
