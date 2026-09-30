"""冻结HCA专用Q_B MIX候选，不改共享CSA实现。"""

import ast
import difflib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
PREFIX = REPO.parent / ".cache/hca-qb-mixed-9015d45a-20260930"


def main():
    assert not PREFIX.exists()
    source_root = REPO / "vllm_ascend/ops/pypto"
    base = PREFIX / "base"
    shutil.copytree(source_root, base, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    candidate = PREFIX / "mixed20_v2"
    shutil.copytree(base, candidate)
    perf = (base / "deepseek_v4_flash_dspark_perf/qkv_proj_rope.py").read_text()
    tree = ast.parse(perf)
    names = {"q_proj_rope", "qkv_proj_rope"}
    functions = [
        ast.get_source_segment(perf, n) for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names
    ]
    imports = '''"""HCA专用入口，复用原QA/KV/rope准备，仅替换NZ的Q_B消费者链。"""
import pypto.language as pl
from ..deepseek_v4_flash_dspark_perf.qkv_proj_rope import (
    D, H, HEAD_DIM, Q_LORA, QPROJ_T_PAD, ROPE_DIM, T_DYN,
    kv_proj_rope, q_proj_q as q_proj_q_separate, q_proj_qr, rope_prepare,
)
from ..deepseek_v4_flash_dspark_perf.nz_mode import BF16_WEIGHT_LAYOUT, QUANT_WEIGHT_LAYOUT, QUANT_WEIGHT_NZ
from .q_projection_mixed import q_proj_q_mixed

q_proj_q = q_proj_q_mixed if QUANT_WEIGHT_NZ else q_proj_q_separate
'''
    wrapper = imports + "\n\n" + "\n\n".join("@pl.jit.inline(auto_scope=False)\n" + f for f in functions) + "\n"
    assert len(functions) == 2
    (candidate / "deepseek_v4_flash_hca/qkv_mixed.py").write_text(wrapper)
    (candidate / "deepseek_v4_flash_hca/q_projection_mixed.py").write_text((ROOT / "q_projection_mixed.py").read_text())
    relative = Path("deepseek_v4_flash_hca/decode_hca.py")
    path = candidate / relative
    before = path.read_text()
    old = "from ..deepseek_v4_flash_dspark_perf.qkv_proj_rope import qkv_proj_rope"
    assert before.count(old) == 1
    after = before.replace(old, "from .qkv_mixed import qkv_proj_rope")
    path.write_text(after)
    (ROOT / "entry.patch").write_text(
        "".join(
            difflib.unified_diff(
                before.splitlines(True),
                after.splitlines(True),
                fromfile="a/" + str(relative),
                tofile="b/" + str(relative),
                n=0,
            )
        )
    )
    (ROOT / "qkv_mixed.py").write_text(wrapper)
    for source in PREFIX.rglob("*.py"):
        source.chmod(0o444)
    (ROOT / "source.json").write_text(
        json.dumps(
            {
                "baseline_commit": "9015d45a",
                "sources": {"base": str(base), "mixed20_v2": str(candidate)},
                "change": (
                    "20 MIX groups, M128/N256/K256 integer Cube, two-slot GM head handoff, two AIV lanes per group"
                ),
                "invariants": (
                    "full INT32 dot product; row scale then channel scale; full-head RMS; BF16 and tail math unchanged"
                ),
                "native_reference": "ops-nn19614968 quant_batch_matmul_v3_pertoken_basic.h CV buffer and epilogue",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
