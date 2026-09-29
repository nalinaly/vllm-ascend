"""从冻结48核写回候选派生16核同步及两pool联动，不修改在跑包。"""

import ast
import difflib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[4]
PREFIX = WORKSPACE / ".cache/csa-spmd-writeback-e110a886-v1"
PACKAGE = "dsv4_csa_spmd_writeback_e110a886_v1"
VARIANTS = ("wb16_sync", "wb16_pool_sync")


def replace_once(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new, 1)


def main():
    manifest = {
        "production": "e110a886",
        "case": [131072, 16],
        "variants": [],
        "rationale": (
            "16 writeback + 16 attention pool + 16 indexer pool can share 48 AIV; "
            "actual overlap and drain must be measured"
        ),
    }
    for name in VARIANTS:
        dest = Path(str(PREFIX) + "-" + name)
        assert not dest.exists(), dest
        shutil.copytree(
            Path(str(PREFIX) + "-wb48_sync"),
            dest,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "build_output"),
        )
        package = dest / "vllm_ascend/ops/pypto" / PACKAGE
        edits = []
        for filename in ("decode_csa.py", "decode_compressor_ratio4.py", "decode_indexer_compressor.py"):
            path = package / filename
            before = path.read_text()
            text = before
            if filename == "decode_csa.py":
                text = replace_once(
                    text, "import pypto.language as pl", "import pypto.language as pl\n\nWB_BALANCED_WORKERS = 16"
                )
                text = replace_once(
                    text,
                    "(t_dim + CSA_WB_WORKERS - 1) // CSA_WB_WORKERS",
                    "(t_dim + WB_BALANCED_WORKERS - 1) // WB_BALANCED_WORKERS",
                )
            elif name == "wb16_pool_sync":
                if filename == "decode_compressor_ratio4.py":
                    text = replace_once(
                        text,
                        '        name_hint="scatter_softmax_pool",\n',
                        '        name_hint="scatter_softmax_pool",\n        sync_start=True,\n',
                    )
                else:
                    text = replace_once(
                        text,
                        'with pl.spmd(pool_workers, name_hint="scatter_softmax_pool",\n',
                        'with pl.spmd(pool_workers, name_hint="scatter_softmax_pool", sync_start=True,\n',
                    )
            ast.parse(text)
            if text != before:
                path.chmod(0o644)
                path.write_text(text)
            baseline = Path(str(PREFIX) + "-baseline") / "vllm_ascend/ops/pypto" / PACKAGE / filename
            edits.extend(
                difflib.unified_diff(
                    baseline.read_text().splitlines(True),
                    text.splitlines(True),
                    fromfile="a/" + filename,
                    tofile="b/" + filename,
                )
            )
        (ROOT / f"{name}.patch").write_text("".join(edits))
        for path in dest.rglob("*.py"):
            path.chmod(0o444)
        manifest["variants"].append({"name": name, "source": str(dest)})
    (ROOT / "balanced_source.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
