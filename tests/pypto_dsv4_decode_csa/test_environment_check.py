# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Host-only tests for the read-only CSA environment preflight."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from importlib import machinery, metadata
from pathlib import Path

import pytest

from tests.pypto_dsv4_decode_csa.check_environment import (
    CAMPAIGN_DEVICE,
    EXPECTED_REPOSITORIES,
    CheckStatus,
    inspect_environment,
)


def _module_spec(module: str, root: Path) -> machinery.ModuleSpec:
    if module == "_task_interface":
        return machinery.ModuleSpec(module, loader=None, origin=str(root / f"{module}.so"))
    spec = machinery.ModuleSpec(module, loader=None, origin=str(root / module / "__init__.py"))
    spec.submodule_search_locations = [str(root / module)]
    return spec


def _complete_probe(tmp_path: Path):
    workspace = tmp_path / "workspace"
    repository = workspace / "vllm-ascend"
    repository.mkdir(parents=True)
    (repository / ".venv").mkdir()
    ptoas = workspace / "pto/PTOAS/build-v0.57-llvm21-cann9.2-clean/tools/ptoas/ptoas"
    ptoas.parent.mkdir(parents=True)
    ptoas.write_text("#!/bin/sh\n", encoding="utf-8")
    ptoas.chmod(0o755)
    gcc_runtime = workspace / "toolchains/gcc15/lib"
    gcc_runtime.mkdir(parents=True)

    repository_commits: dict[Path, str] = {}
    for expectation in EXPECTED_REPOSITORIES:
        path = workspace / expectation.relative_path
        path.mkdir(parents=True, exist_ok=True)
        repository_commits[path.resolve()] = expectation.commit

    source_roots = {
        "pypto": workspace / "pto/pypto/python",
        "simpler": workspace / "pto/pypto/runtime/python",
        "vllm": workspace,
        "vllm_ascend": repository,
        "torch": workspace / "site-packages",
        "torch_npu": workspace / "site-packages",
        "_task_interface": workspace / "site-packages",
    }

    def find_spec(module: str):
        return _module_spec(module, source_roots[module])

    versions = {
        "torch": "2.12.0+cpu",
        "torch-npu": "2.12.0+test",
        "pypto": "0.2.1",
        "simpler": "0.1.0",
    }

    def distribution_version(distribution: str) -> str:
        try:
            return versions[distribution]
        except KeyError as error:
            raise metadata.PackageNotFoundError(distribution) from error

    commands: list[tuple[str, ...]] = []

    def run_command(command, **kwargs):
        normalized = tuple(str(item) for item in command)
        commands.append(normalized)
        assert kwargs == {"check": False, "capture_output": True, "text": True}
        path = Path(normalized[2]).resolve()
        commit = repository_commits[path]
        return subprocess.CompletedProcess(normalized, 0, stdout=commit + "\n", stderr="")

    arguments = {
        "repository_root": repository,
        "workspace_root": workspace,
        "ptoas_path": ptoas,
        "gcc_runtime_path": gcc_runtime,
        "device": CAMPAIGN_DEVICE,
        "environ": {
            "PATH": str(ptoas.parent),
            "LD_LIBRARY_PATH": str(gcc_runtime),
        },
        "python_prefix": repository / ".venv",
        "python_version": (3, 11),
        "find_spec": find_spec,
        "distribution_version": distribution_version,
        "run_command": run_command,
        "which": lambda executable: str(ptoas) if executable == "ptoas" else None,
    }
    return arguments, commands


def test_complete_report_is_json_ready_and_never_executes_an_npu_command(tmp_path: Path) -> None:
    arguments, commands = _complete_probe(tmp_path)

    report = inspect_environment(**arguments)
    payload = report.to_dict()

    assert report.ok
    assert report.npu_probe_performed is False
    assert payload["device"] == 0
    assert payload["npu_probe_performed"] is False
    assert all(check.status is CheckStatus.PASS for check in report.checks)
    assert len(commands) == len(EXPECTED_REPOSITORIES)
    assert all(command[:2] == ("git", "-C") for command in commands)
    assert all(command[-2:] == ("rev-parse", "HEAD") for command in commands)
    assert not any("npu" in token.lower() or "davinci" in token.lower() for command in commands for token in command)
    json.dumps(payload)


@pytest.mark.parametrize(
    ("mutation", "failed_check"),
    (
        ("wrong_device", "campaign_device"),
        ("wrong_python", "python_version"),
        ("ptoas_root", "ptoas_root_unset"),
        ("missing_ptoas_path", "ptoas_path_precedence"),
        ("missing_gcc", "gcc15_runtime"),
    ),
)
def test_required_environment_mismatches_fail_without_device_access(
    tmp_path: Path,
    mutation: str,
    failed_check: str,
) -> None:
    arguments, _ = _complete_probe(tmp_path)
    if mutation == "wrong_device":
        arguments["device"] = 1
    elif mutation == "wrong_python":
        arguments["python_version"] = (3, 12)
    elif mutation == "ptoas_root":
        arguments["environ"] = {**arguments["environ"], "PTOAS_ROOT": "/stale/wrapper"}
    elif mutation == "missing_ptoas_path":
        arguments["which"] = lambda _executable: None
    elif mutation == "missing_gcc":
        arguments["environ"] = {"PATH": arguments["environ"]["PATH"]}
    else:  # pragma: no cover - parametrization is closed above
        raise AssertionError(mutation)

    report = inspect_environment(**arguments)
    failures = {check.name for check in report.checks if check.status is CheckStatus.FAIL}

    assert not report.ok
    assert failed_check in failures
    assert report.npu_probe_performed is False


def test_stale_source_origin_and_commit_are_reported_not_imported(tmp_path: Path) -> None:
    arguments, _ = _complete_probe(tmp_path)
    original_find_spec = arguments["find_spec"]

    def stale_find_spec(module: str):
        if module == "pypto":
            return _module_spec(module, tmp_path / "stale-user-site")
        return original_find_spec(module)

    original_run = arguments["run_command"]
    seen = 0

    def stale_run(command, **kwargs):
        nonlocal seen
        completed = original_run(command, **kwargs)
        if Path(command[2]).name == "vllm-ascend":
            seen += 1
            return subprocess.CompletedProcess(command, 0, stdout="deadbeef\n", stderr="")
        return completed

    arguments["find_spec"] = stale_find_spec
    arguments["run_command"] = stale_run

    report = inspect_environment(**arguments)
    failures = {check.name for check in report.checks if check.status is CheckStatus.FAIL}

    assert seen == 1
    assert "module:pypto" in failures
    assert "repository:vllm-ascend" in failures
    assert not report.ok


def test_importing_checker_does_not_import_accelerator_or_framework_packages() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    probe = r"""
import sys

forbidden = ("torch", "torch_npu", "pypto", "simpler", "vllm")
assert not any(name in sys.modules for name in forbidden)
import tests.pypto_dsv4_decode_csa.check_environment  # noqa: F401
loaded = [name for name in forbidden if name in sys.modules]
assert loaded == [], loaded
"""
    environment = os.environ.copy()
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = os.pathsep.join(part for part in (str(repository_root), existing) if part)

    completed = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=repository_root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"


def test_readme_keeps_fresh_process_device_and_evidence_boundaries_explicit() -> None:
    readme = (Path(__file__).with_name("README.md")).read_text(encoding="utf-8")

    assert "--device 0" in readme
    assert "tensormap_and_ringbuffer" in readme
    assert "host_build_graph" in readme
    assert "独立的新 Python 进程" in readme
    assert "不能" in readme
    assert "45-slot" in readme
    assert "最终 44-slot + writeback 依赖" in readme
    assert "四 graph 同存" in readme
    assert "partial bucket" in readme
    assert "zero-weight" in readme
    assert "B8/B12/B16 非零算法精度" in readme
    assert "320 ms" in readme
