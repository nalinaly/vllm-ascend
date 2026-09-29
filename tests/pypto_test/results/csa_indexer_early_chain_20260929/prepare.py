"""冻结七处生产者early标志的独立对照；沿用7b296153，不叠加Sparse候选。"""

import ast
import difflib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[4]
TEMPLATE = ROOT.parent / "csa_sparse_plan_rope_20260929"
BASE = WORKSPACE / ".cache/csa-ob-hc-scalar-fused-9a868d26-candidate"
OLD_PACKAGE = "dsv4_csa_ob_hc_scalar_fused_9a868d26"
PACKAGE = "dsv4_csa_indexer_early_chain_7b296153"
PREFIX = WORKSPACE / ".cache/csa-indexer-early-chain-7b296153"
CHANGES = {
    "decode_csa.py": ["csa_row_offsets"],
    "decode_compressor_ratio4.py": ["kv_score_proj"],
    "decode_indexer_compressor.py": ["kv_score_proj", "scatter_softmax_pool", "indexer_boundary_init",
                                      "rmsnorm_rope", "kv_hadamard"],
}


def add_flags(before, names):
    after = before
    for name in names:
        anchor = f'name_hint="{name}"'
        if after.count(anchor) != 1:
            raise ValueError(f"Expected one dispatch: {name}")
        after = after.replace(anchor, anchor + ", allow_early_resolve=True")
    # 确认七个调度标志之外的AST不变；不扫描无关文件或生成hash清单。
    tree = ast.parse(after)
    count = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        kwargs = {kw.arg: kw.value for kw in node.keywords}
        hint = kwargs.get("name_hint")
        if isinstance(hint, ast.Constant) and hint.value in names:
            assert ast.literal_eval(kwargs["allow_early_resolve"]) is True
            node.keywords = [kw for kw in node.keywords if kw.arg != "allow_early_resolve"]
            count += 1
    assert count == len(names) and ast.dump(tree) == ast.dump(ast.parse(before))
    return after


def main():
    if (ROOT / "task.txt").exists():
        raise RuntimeError("Submitted packages are immutable")
    for side in ("baseline", "candidate"):
        destination = Path(str(PREFIX) + "-" + side)
        if destination.exists():
            raise RuntimeError(f"Existing private package: {destination}")
        shutil.copytree(BASE, destination, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "build_output"))
        packages = destination / "vllm_ascend/ops/pypto"
        (packages / OLD_PACKAGE).rename(packages / PACKAGE)
    patch = []
    for filename, names in CHANGES.items():
        relative = Path("vllm_ascend/ops/pypto") / PACKAGE / filename
        before = (Path(str(PREFIX) + "-baseline") / relative).read_text()
        after = add_flags(before, names)
        target = Path(str(PREFIX) + "-candidate") / relative
        target.chmod(0o644)
        target.write_text(after)
        patch.extend(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
                                        fromfile="a/" + str(relative), tofile="b/" + str(relative)))
    (ROOT / "candidate.patch").write_text("".join(patch))
    for name in ("compile.py", "run.sh", "run_side.sh"):
        text = (TEMPLATE / name).read_text().replace(TEMPLATE.name, ROOT.name)
        text = text.replace("csa-sparse-plan-rope-7b296153", PREFIX.name)
        text = text.replace("dsv4_csa_sparse_plan_rope_7b296153", PACKAGE)
        (ROOT / name).write_text(text)
    source = {"baseline": "7b296153", "base_source": str(BASE), "source_prefix": str(PREFIX),
              "variant": "pkg:" + PACKAGE, "cases": [[131072, 16], [8192, 24]],
              "change": "只补Indexer cache生产链七处allow_early_resolve；含Attention projection和row offsets前置",
              "modified_dispatches": CHANGES, "only_seven_flags_ast_verified": True,
              "scope": "私有三文件调度候选；数学、任务数、依赖、worker、Score准入与长短early策略保持"}
    (ROOT / "source.json").write_text(json.dumps(source, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
