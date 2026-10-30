# SPDX-License-Identifier: Apache-2.0
"""冻结模型测试源码；共用SDK及Native库，不复制或重建SDK。"""

import argparse
import json
import shutil
import subprocess
from pathlib import Path

here = Path(__file__).resolve().parent
repo = here.parents[2]
parser = argparse.ArgumentParser()
parser.add_argument("--name", default="model_source")
args = parser.parse_args()
destination = here / "results" / args.name
destination.mkdir(parents=True, exist_ok=False)
ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
shutil.copytree(repo / "vllm_ascend", destination / "vllm_ascend", ignore=ignore)
tests = destination / "tests/pypto_test"
tests.mkdir(parents=True)
for name in ("offline_pd", "low_acceptance_20261001", "csa_shared_toolchain", "csa_toolchain_repro_20261019"):
    shutil.copytree(
        repo / "tests/pypto_test" / name,
        tests / name,
        ignore=shutil.ignore_patterns("__pycache__", "results", "runs", "*.pyc"),
    )
for source in (repo / "tests/pypto_test").glob("*.py"):
    shutil.copy2(source, tests / source.name)
(tests / "csa_aclnn_backend_20261028").mkdir()
shutil.copy2(
    repo / "tests/pypto_test/csa_aclnn_backend_20261028/checks.py", tests / "csa_aclnn_backend_20261028/checks.py"
)
lib = destination / ".cache/csa/native-install"
lib.parent.mkdir(parents=True)
lib.symlink_to(repo / ".cache/csa/native-install", target_is_directory=True)
record = {
    "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
    "source": str(destination),
    "purpose": "Native真实128K前缀生成；不是随机KV",
    "shared_manifest": str(repo / "tests/pypto_test/csa_shared_toolchain/manifest.json"),
}
(destination / "source.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
(destination / "source.diff").write_bytes(subprocess.check_output(["git", "diff", "--", "vllm_ascend"], cwd=repo))
print(destination, flush=True)
