# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Read-only Host preflight for the DeepSeek V4 decode CSA A3 campaign.

The checker deliberately uses only Python's standard library.  It locates
packages without importing them, reads repository HEADs with ``git rev-parse``,
and inspects files and environment variables.  It never imports ``torch_npu``,
opens an NPU device, invokes ``npu-smi``, builds an artifact, or edits the
workspace/system.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import Enum
from importlib.machinery import ModuleSpec
from pathlib import Path

ENVIRONMENT_REPORT_SCHEMA_VERSION = "1.0.0"
CAMPAIGN_DEVICE = 0
EXPECTED_PYTHON = (3, 11)


@dataclass(frozen=True, slots=True)
class RepositoryExpectation:
    name: str
    relative_path: str
    commit: str


EXPECTED_REPOSITORIES = (
    RepositoryExpectation("vllm-ascend", "vllm-ascend", "e7cb166290dfcbf2f997aa67f01b323be643fe0e"),
    RepositoryExpectation("vllm", "vllm", "6e448d0ea9bf3d88d898b65449ca6dc2aec170ac"),
    RepositoryExpectation("PyPTO", "pto/pypto", "9cece0b730a96fe1a52c2637537132f524ffe1ea"),
    RepositoryExpectation("simpler", "pto/pypto/runtime", "b6f905f63277597bd2547d672fd9d57b6013fca9"),
    RepositoryExpectation("pypto-lib", "pypto-lib", "0073f4228811eae687f9049b417be803c75c49e1"),
)

EXPECTED_DISTRIBUTION_PREFIXES = {
    "torch": "2.12.0",
    "torch-npu": "2.12.0",
    "pypto": "0.2.1",
    "simpler": "0.1.0",
}


class CheckStatus(str, Enum):
    PASS = "pass"
    WARNING = "warning"
    FAIL = "fail"


@dataclass(frozen=True, slots=True)
class EnvironmentCheck:
    name: str
    status: CheckStatus
    detail: str
    required: bool = True


@dataclass(frozen=True, slots=True)
class EnvironmentReport:
    repository_root: str
    workspace_root: str
    python_executable: str
    python_prefix: str
    device: int
    npu_probe_performed: bool
    checks: tuple[EnvironmentCheck, ...]
    schema_version: str = ENVIRONMENT_REPORT_SCHEMA_VERSION

    @property
    def ok(self) -> bool:
        return all(not check.required or check.status is not CheckStatus.FAIL for check in self.checks)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "ok": self.ok,
            "repository_root": self.repository_root,
            "workspace_root": self.workspace_root,
            "python_executable": self.python_executable,
            "python_prefix": self.python_prefix,
            "device": self.device,
            "npu_probe_performed": self.npu_probe_performed,
            "checks": [
                {
                    **asdict(check),
                    "status": check.status.value,
                }
                for check in self.checks
            ],
        }


FindSpec = Callable[[str], ModuleSpec | None]
DistributionVersion = Callable[[str], str]
RunCommand = Callable[..., subprocess.CompletedProcess[str]]
Which = Callable[[str], str | None]


def _is_within(candidate: str | os.PathLike[str], expected_root: Path) -> bool:
    try:
        Path(candidate).resolve().relative_to(expected_root.resolve())
    except (OSError, ValueError):
        return False
    return True


def _module_locations(spec: ModuleSpec) -> tuple[str, ...]:
    locations: list[str] = []
    if spec.origin and spec.origin not in {"built-in", "frozen"}:
        locations.append(spec.origin)
    if spec.submodule_search_locations is not None:
        locations.extend(str(path) for path in spec.submodule_search_locations)
    return tuple(dict.fromkeys(locations))


def _git_head(path: Path, run_command: RunCommand) -> tuple[str | None, str | None]:
    try:
        completed = run_command(
            ("git", "-C", str(path), "rev-parse", "HEAD"),
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as error:
        return None, str(error)
    if completed.returncode != 0:
        message = completed.stderr.strip() or completed.stdout.strip() or f"exit code {completed.returncode}"
        return None, message
    return completed.stdout.strip(), None


def inspect_environment(
    *,
    repository_root: Path | None = None,
    workspace_root: Path | None = None,
    ptoas_path: Path | None = None,
    gcc_runtime_path: Path | None = None,
    device: int = CAMPAIGN_DEVICE,
    environ: Mapping[str, str] | None = None,
    python_prefix: Path | None = None,
    python_version: tuple[int, int] | None = None,
    find_spec: FindSpec = importlib.util.find_spec,
    distribution_version: DistributionVersion = importlib.metadata.version,
    run_command: RunCommand = subprocess.run,
    which: Which = shutil.which,
) -> EnvironmentReport:
    """Return a read-only environment report without importing accelerator code."""
    repo = (repository_root or Path(__file__).resolve().parents[2]).resolve()
    workspace = (workspace_root or repo.parent).resolve()
    ptoas = (ptoas_path or workspace / "pto/PTOAS/build-v0.57-llvm21-cann9.2-clean/tools/ptoas/ptoas").resolve()
    gcc_runtime = (gcc_runtime_path or workspace / "toolchains/gcc15/lib").resolve()
    environment = dict(os.environ if environ is None else environ)
    prefix = (python_prefix or Path(sys.prefix)).resolve()
    version = python_version or (sys.version_info.major, sys.version_info.minor)
    checks: list[EnvironmentCheck] = []

    checks.append(
        EnvironmentCheck(
            name="inspection_policy",
            status=CheckStatus.PASS,
            detail="standard-library metadata/path/git inspection only; no NPU API or device probe",
        )
    )
    checks.append(
        EnvironmentCheck(
            name="campaign_device",
            status=CheckStatus.PASS if device == CAMPAIGN_DEVICE else CheckStatus.FAIL,
            detail=f"requested logical device={device}; this campaign is fixed to logical device {CAMPAIGN_DEVICE}",
        )
    )
    checks.append(
        EnvironmentCheck(
            name="python_version",
            status=CheckStatus.PASS if version == EXPECTED_PYTHON else CheckStatus.FAIL,
            detail=f"observed={version[0]}.{version[1]}, expected={EXPECTED_PYTHON[0]}.{EXPECTED_PYTHON[1]}",
        )
    )
    expected_prefix = (repo / ".venv").resolve()
    checks.append(
        EnvironmentCheck(
            name="python_venv",
            status=CheckStatus.PASS if prefix == expected_prefix else CheckStatus.FAIL,
            detail=f"observed={prefix}, expected={expected_prefix}",
        )
    )

    for expectation in EXPECTED_REPOSITORIES:
        path = workspace / expectation.relative_path
        if not path.is_dir():
            checks.append(
                EnvironmentCheck(
                    name=f"repository:{expectation.name}",
                    status=CheckStatus.FAIL,
                    detail=f"missing directory: {path}",
                )
            )
            continue
        head, error = _git_head(path, run_command)
        if error is not None:
            checks.append(
                EnvironmentCheck(
                    name=f"repository:{expectation.name}",
                    status=CheckStatus.FAIL,
                    detail=f"cannot read HEAD at {path}: {error}",
                )
            )
            continue
        checks.append(
            EnvironmentCheck(
                name=f"repository:{expectation.name}",
                status=CheckStatus.PASS if head == expectation.commit else CheckStatus.FAIL,
                detail=f"observed={head}, expected={expectation.commit}, path={path}",
            )
        )

    for distribution, expected_prefix in EXPECTED_DISTRIBUTION_PREFIXES.items():
        try:
            observed = distribution_version(distribution)
        except importlib.metadata.PackageNotFoundError:
            checks.append(
                EnvironmentCheck(
                    name=f"distribution:{distribution}",
                    status=CheckStatus.FAIL,
                    detail="distribution metadata not found",
                )
            )
            continue
        checks.append(
            EnvironmentCheck(
                name=f"distribution:{distribution}",
                status=CheckStatus.PASS if observed.startswith(expected_prefix) else CheckStatus.FAIL,
                detail=f"observed={observed}, expected prefix={expected_prefix}",
            )
        )

    module_roots: dict[str, Path | None] = {
        "torch": None,
        "torch_npu": None,
        "_task_interface": None,
        "pypto": workspace / "pto/pypto",
        "simpler": workspace / "pto/pypto/runtime",
        "vllm": workspace / "vllm",
        "vllm_ascend": repo,
    }
    for module, expected_root in module_roots.items():
        try:
            spec = find_spec(module)
        except (AttributeError, ImportError, ModuleNotFoundError, ValueError) as error:
            spec = None
            failure_detail = f"module spec lookup failed: {error}"
        else:
            failure_detail = "module spec not found"
        if spec is None:
            checks.append(
                EnvironmentCheck(
                    name=f"module:{module}",
                    status=CheckStatus.FAIL,
                    detail=failure_detail,
                )
            )
            continue
        locations = _module_locations(spec)
        if expected_root is None:
            status = CheckStatus.PASS
            detail = f"located without import: {locations}"
        else:
            matches_source = any(_is_within(location, expected_root) for location in locations)
            status = CheckStatus.PASS if matches_source else CheckStatus.FAIL
            detail = f"locations={locations}, expected source root={expected_root}"
        checks.append(
            EnvironmentCheck(
                name=f"module:{module}",
                status=status,
                detail=detail,
            )
        )

    ptoas_exists = ptoas.is_file() and os.access(ptoas, os.X_OK)
    checks.append(
        EnvironmentCheck(
            name="ptoas_executable",
            status=CheckStatus.PASS if ptoas_exists else CheckStatus.FAIL,
            detail=f"expected executable={ptoas}",
        )
    )
    selected_ptoas = which("ptoas")
    selected_matches = selected_ptoas is not None and Path(selected_ptoas).resolve() == ptoas
    checks.append(
        EnvironmentCheck(
            name="ptoas_path_precedence",
            status=CheckStatus.PASS if selected_matches else CheckStatus.FAIL,
            detail=f"PATH resolves ptoas to {selected_ptoas!r}; expected {ptoas}",
        )
    )
    ptoas_root = environment.get("PTOAS_ROOT")
    checks.append(
        EnvironmentCheck(
            name="ptoas_root_unset",
            status=CheckStatus.PASS if not ptoas_root else CheckStatus.FAIL,
            detail=f"PTOAS_ROOT={ptoas_root!r}; it must be unset for the pinned local PTOAS",
        )
    )

    ld_entries = tuple(filter(None, environment.get("LD_LIBRARY_PATH", "").split(os.pathsep)))
    gcc_selected = any(_is_within(entry, gcc_runtime) and Path(entry).resolve() == gcc_runtime for entry in ld_entries)
    checks.append(
        EnvironmentCheck(
            name="gcc15_runtime",
            status=CheckStatus.PASS if gcc_runtime.is_dir() and gcc_selected else CheckStatus.FAIL,
            detail=f"expected {gcc_runtime} in LD_LIBRARY_PATH",
        )
    )

    return EnvironmentReport(
        repository_root=str(repo),
        workspace_root=str(workspace),
        python_executable=sys.executable,
        python_prefix=str(prefix),
        device=device,
        npu_probe_performed=False,
        checks=tuple(checks),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=None)
    parser.add_argument("--workspace-root", type=Path, default=None)
    parser.add_argument("--ptoas", type=Path, default=None)
    parser.add_argument("--gcc-runtime", type=Path, default=None)
    parser.add_argument("--device", type=int, default=CAMPAIGN_DEVICE)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = inspect_environment(
        repository_root=args.repository_root,
        workspace_root=args.workspace_root,
        ptoas_path=args.ptoas,
        gcc_runtime_path=args.gcc_runtime,
        device=args.device,
    )
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CAMPAIGN_DEVICE",
    "ENVIRONMENT_REPORT_SCHEMA_VERSION",
    "CheckStatus",
    "EnvironmentCheck",
    "EnvironmentReport",
    "build_parser",
    "inspect_environment",
    "main",
]
