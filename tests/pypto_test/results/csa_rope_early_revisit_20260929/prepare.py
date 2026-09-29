"""在完整Gather基底仅重新评估RoPE符号生产者的提前派发资格。"""

import ast
import difflib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[4]
BASE = WORKSPACE / ".cache/csa-indexer-rope-flat-gather-632dd00a-v1-candidate"
PREFIX = WORKSPACE / ".cache/csa-rope-early-revisit-46cec3a0-v1"
OLD_PACKAGE = "dsv4_csa_indexer_rope_flat_gather_632dd00a_v1"
PACKAGE = "dsv4_csa_rope_early_revisit_46cec3a0_v1"


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
    old = ('    with pl.spmd(pl.min(rope_sign_blocks, CSA_ROPE_WORKERS), name_hint="csa_rope_sign",\n'
           '                 deps=[offsets_tid]) as rope_tid:')
    new = ('    with pl.spmd(pl.min(rope_sign_blocks, CSA_ROPE_WORKERS), name_hint="csa_rope_sign",\n'
           '                 deps=[offsets_tid], allow_early_resolve=True) as rope_tid:')
    assert before.count(old) == 1
    after = before.replace(old, new)
    ast.parse(after)
    path.chmod(0o644)
    path.write_text(after)
    (ROOT / "candidate.patch").write_text("".join(difflib.unified_diff(
        before.splitlines(True), after.splitlines(True), fromfile="a/" + str(relative), tofile="b/" + str(relative))))
    template = ROOT.parent / "csa_indexer_rope_flat_gather_20260929"
    for name in ("compile.py", "run.sh", "run_side.sh"):
        text = (template / name).read_text().replace(template.name, ROOT.name)
        text = text.replace("csa-indexer-rope-flat-gather-632dd00a-v1", PREFIX.name).replace(OLD_PACKAGE, PACKAGE)
        (ROOT / name).write_text(text)
    (ROOT / "source.json").write_text(json.dumps({
        "baseline": "46cec3a0", "base_source": str(BASE), "source_prefix": str(PREFIX),
        "variant": "pkg:" + PACKAGE, "cases": [[131072, 16], [8192, 24]],
        "change": "仅csa_rope_sign生产者allow_early_resolve由默认False改True；真实依赖保持",
        "previous_attempt": "验证日志§214，2dd51f15/atomic1/8K B16，完整CSA无收益，已撤回",
        "new_basis": ("当前46cec3a0已改Q/Sparse/Indexer Gather和七处early，atomic0/CANN9.2；"
                      "实际fanin仍确认csa_rope_sign禁止Indexer反量化预派发；复核未覆盖的128K主档"),
        "not_changed": "算术、任务数/分块、sync_start、cache/权重、精度版、Native与工具链",
        "acceptance": ("先完整CPU编译，再同卡两档8:2完整CSA/P95；性能后八类状态零容差，"
                       "四窗实际提前资格/资源等待；调度改善不得用随机核时下降替代"),
    }, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
