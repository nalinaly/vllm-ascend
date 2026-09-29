"""为已实测的T144分工选择RoPE early；其余形状保持原策略。"""

import ast
import difflib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[4]
BASE = WORKSPACE / ".cache/csa-rope-early-revisit-46cec3a0-v1-baseline"
PREFIX = WORKSPACE / ".cache/csa-rope-early-rows-46cec3a0-v1"
OLD_PACKAGE = "dsv4_csa_rope_early_revisit_46cec3a0_v1"
PACKAGE = "dsv4_csa_rope_early_rows_46cec3a0_v1"


def change(before):
    start = before.index("    rope_sign_blocks = (t_dim + CSA_ROPE_SIGN_T_TILE - 1)")
    end = before.index("\n    q = pl.create_tensor([t_dim, H, HEAD_DIM]", start)
    block = before[start:end].rstrip()
    assert block.count("deps=[offsets_tid]) as rope_tid:") == 1
    block = block.replace("deps=[offsets_tid]) as rope_tid:",
                          "deps=[offsets_tid], allow_early_resolve=ALLOW_EARLY) as rope_tid:")
    helper = '''@pl.jit.inline(auto_scope=False)
def _indexer_rope_sign(
    freqs_sin: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.FP32],
    idx_sin_signed: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.FP32],
    t_dim: pl.Scalar[pl.INDEX],
    offsets_tid: pl.Scalar[pl.TASK_ID],
    ALLOW_EARLY: pl.constexpr,
):
    """Keep one numerical implementation; specialize only the producer policy."""
''' + block + "\n    return idx_sin_signed, rope_tid\n\n\n"
    replacement = '''    # T144 (the B24/S6 bucket) benefited in both 8K and 128K measurements.
    # Do not extend the policy to unmeasured row counts or switch by test mode.
    if t_dim == CSA_ROPE_EARLY_QUERY_ROWS:
        idx_sin_signed, rope_tid = _indexer_rope_sign(
            freqs_sin, idx_sin_signed, t_dim, offsets_tid, True,
        )
    else:
        idx_sin_signed, rope_tid = _indexer_rope_sign(
            freqs_sin, idx_sin_signed, t_dim, offsets_tid, False,
        )
'''
    after = before[:start] + replacement + before[end:]
    anchor = "def _decode_csa_tp1_layer("
    assert after.count(anchor) == 1
    after = after.replace(anchor, helper + anchor, 1)
    after = after.replace(
        "CSA_ROPE_WORKERS = 16\n",
        "CSA_ROPE_WORKERS = 16\nCSA_ROPE_EARLY_QUERY_ROWS = 144  # Measured B24/S6 dispatch policy.\n", 1,
    )
    ast.parse(after)
    return after


def main():
    assert not (ROOT / "task.txt").exists()
    for side in ("baseline", "candidate"):
        dest = Path(str(PREFIX) + "-" + side)
        assert not dest.exists(), dest
        shutil.copytree(BASE, dest, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "build_output"))
        packages = dest / "vllm_ascend/ops/pypto"
        (packages / OLD_PACKAGE).rename(packages / PACKAGE)
    relative = Path("vllm_ascend/ops/pypto") / PACKAGE / "decode_csa.py"
    path = Path(str(PREFIX) + "-candidate") / relative
    before = path.read_text()
    after = change(before)
    path.chmod(0o644)
    path.write_text(after)
    (ROOT / "candidate.patch").write_text("".join(difflib.unified_diff(
        before.splitlines(True), after.splitlines(True), fromfile="a/" + str(relative), tofile="b/" + str(relative))))
    template = ROOT.parent / "csa_rope_early_revisit_20260929"
    for name in ("compile.py", "run.sh", "run_side.sh"):
        text = (template / name).read_text().replace(template.name, ROOT.name)
        text = text.replace("csa-rope-early-revisit-46cec3a0-v1", PREFIX.name).replace(OLD_PACKAGE, PACKAGE)
        (ROOT / name).write_text(text)
    (ROOT / "source.json").write_text(json.dumps({
        "baseline": "46cec3a0", "base_source": str(BASE), "source_prefix": str(PREFIX),
        "variant": "pkg:" + PACKAGE, "cases": [[131072, 16], [8192, 24]],
        "change": "提取原RoPE符号计算为同一inline helper，仅T144允许消费者提前派发；其他行数False",
        "basis": [str(template / "decision.json"), str(ROOT.parent / "csa_rope_early_b24_20260929/evidence.json")],
        "not_changed": "算术、任务数量/块数、真实依赖、sync_start、cache和权重布局、精度版、Native及工具链",
        "acceptance": "CPU完整编译后两代表档检验分支开销及收益，性能后八类状态零容差；有效再测T144真实筛选/padding",
    }, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
