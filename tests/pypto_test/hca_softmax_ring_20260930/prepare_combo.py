"""把通过单独检查的UB环叠加至已提交O-B版本，验证组合及长分支短循环。"""

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
PREFIX = REPO.parent / ".cache/hca-ring-combo-4ffe0a53-20260930"


def main():
    assert not PREFIX.exists(), PREFIX
    baseline, candidate = PREFIX / "base", PREFIX / "ring"
    source = json.loads((ROOT / "source_v2.json").read_text())
    shutil.copytree(REPO / "vllm_ascend/ops/pypto", baseline, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copytree(baseline, candidate)
    relative = Path("deepseek_v4_flash_hca/decode_sparse_attn_hca.py")
    assert (baseline / relative).read_text() == (Path(source["baseline"]) / relative).read_text()
    shutil.copyfile(Path(source["candidate"]) / relative, candidate / relative)
    for folder in (baseline, candidate):
        for path in folder.rglob("*.py"):
            path.chmod(0o444)
    (ROOT / "source_combo.json").write_text(
        json.dumps(
            {
                "baseline_commit": "4ffe0a53",
                "baseline": str(baseline),
                "candidate": str(candidate),
                "change": "v2 UB softmax ring on top of committed O-B activation reuse",
                "cases": [[131072, 16], [16384, 4]],
                "scope": "same-process composition screen and long-path short-loop boundary; not Native acceptance",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
