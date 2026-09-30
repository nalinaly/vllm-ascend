# SPDX-License-Identifier: Apache-2.0
"""将联合验证结果绑定到实际源码与负载；资料更新不触发重复验证。"""

import argparse
import json
import subprocess
from pathlib import Path


CODE_PATHS = (
    "vllm_ascend",
    "tests/pypto_test/offline_pd",
    ":(glob)tests/pypto_test/csa_hca_merge_20260930/*.py",
    ":(glob)tests/pypto_test/csa_hca_merge_20260930/*.sh",
    "tests/pypto_test/dsv4_hca_prepare_opp.py",
)


def git(repo, *args):
    return subprocess.check_output(
        ["git", "-C", str(repo), "--work-tree=.", *args], text=True,
    ).strip()


def current_revision(repo):
    if git(repo, "status", "--porcelain", "--untracked-files=all", "--", *CODE_PATHS):
        raise ValueError("执行代码有未提交修改，不能将结果绑定到 Git 提交")
    return git(repo, "rev-parse", "HEAD")


def record(repo, path, workload):
    data = {"revision": current_revision(repo), "workload": workload}
    # 新记录不能覆盖旧结果的来源。
    with path.open("x") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def verify(repo, path, workload):
    data = json.loads(path.read_text())
    if data.get("workload") != workload:
        raise ValueError("验证记录的 bank、history 或 batch 与本轮不同")
    revision = data.get("revision")
    if not isinstance(revision, str) or not revision:
        raise ValueError("验证记录缺少源码提交")
    current = current_revision(repo)
    changed = git(repo, "diff", "--name-only", revision, current, "--", *CODE_PATHS)
    if changed:
        raise ValueError(f"验证后执行代码已变化，需要重新验证：\n{changed}")
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("record", "verify"))
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--history", type=int, required=True)
    parser.add_argument("--batch", type=int, required=True)
    args = parser.parse_args()
    workload = {"bank": str(args.bank.resolve()), "history": args.history, "batch": args.batch}
    action = record if args.action == "record" else verify
    action(args.repo, args.manifest, workload)


if __name__ == "__main__":
    main()
