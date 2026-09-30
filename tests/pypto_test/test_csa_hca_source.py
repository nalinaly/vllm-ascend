# SPDX-License-Identifier: Apache-2.0
"""真实 Git 历史下验证旧结果不能放行新代码，资料提交仍可复用。"""

import subprocess

import pytest

from csa_hca_merge_20260930.source import record, verify


def test_validation_tracks_code_and_workload(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args):
        subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)

    def commit():
        git("add", ".")
        git("-c", "user.name=测试", "-c", "user.email=test@example.invalid", "commit", "-m", "测试提交")

    git("init")
    code = repo / "vllm_ascend" / "operator.py"
    code.parent.mkdir()
    code.write_text("VERSION = 1\n")
    commit()
    manifest = tmp_path / "source.json"
    workload = {"bank": "/bank", "history": 131072, "batch": 16}
    record(repo, manifest, workload)
    verify(repo, manifest, workload)
    with pytest.raises(FileExistsError):
        record(repo, manifest, workload)
    (repo / "README.md").write_text("资料更新\n")
    commit()
    verify(repo, manifest, workload)
    with pytest.raises(ValueError, match="本轮不同"):
        verify(repo, manifest, {**workload, "batch": 24})
    code.write_text("VERSION = 2\n")
    with pytest.raises(ValueError, match="未提交修改"):
        verify(repo, manifest, workload)
    commit()
    with pytest.raises(ValueError, match="执行代码已变化"):
        verify(repo, manifest, workload)
