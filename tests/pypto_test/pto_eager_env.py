"""Keep local editable-install repair out of the vLLM-Ascend runtime."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path


def _finder_sources(finder: object) -> tuple[str, ...]:
    sources = getattr(finder, "known_source_files", None)
    return tuple(str(path) for path in sources.values()) if isinstance(sources, dict) else ()


def _load_editable_hook(path: Path, module_name: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"pto_eager editable hook not found: {path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load pto_eager editable hook: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)


def activate_pto_eager() -> None:
    """Select ``PTO_EAGER_ROOT`` in this process and future spawned workers."""
    root_text = os.environ.get("PTO_EAGER_ROOT")
    if not root_text:
        raise RuntimeError("source pto_eager/env.sh before running a PyPTO test")
    root = Path(root_text).resolve()

    expected_modules = {
        "pypto": root / "pypto",
        "simpler_setup": root / "simpler",
    }
    for name, expected_root in expected_modules.items():
        loaded = sys.modules.get(name)
        if loaded is None:
            continue
        loaded_path = Path(getattr(loaded, "__file__", "")).resolve()
        if expected_root not in loaded_path.parents:
            raise RuntimeError(f"{name} was already imported from {loaded_path}, expected {expected_root}")

    sys.meta_path[:] = [
        finder
        for finder in sys.meta_path
        if not (
            type(finder).__name__ == "ScikitBuildRedirectingFinder"
            and any("/pypto/" in source or "/simpler/" in source for source in _finder_sources(finder))
        )
    ]
    python_version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    site_packages = root / ".venv" / "lib" / python_version / "site-packages"
    _load_editable_hook(site_packages / "_pypto_editable.py", "_pto_eager_pypto_editable")
    _load_editable_hook(site_packages / "_simpler_editable.py", "_pto_eager_simpler_editable")

    # Python imports sitecustomize in every spawned interpreter. Keeping this
    # directory first makes the same repair apply to vLLM worker processes.
    helper_dir = str(Path(__file__).resolve().parent)
    python_path = os.environ.get("PYTHONPATH", "")
    entries = [entry for entry in python_path.split(os.pathsep) if entry and entry != helper_dir]
    os.environ["PYTHONPATH"] = os.pathsep.join([helper_dir, *entries])

    import pypto  # noqa: PLC0415
    import simpler_setup  # noqa: PLC0415

    for module, expected_root in ((pypto, root / "pypto"), (simpler_setup, root / "simpler")):
        loaded_path = Path(module.__file__).resolve()
        if expected_root not in loaded_path.parents:
            raise RuntimeError(f"selected {module.__name__} path is {loaded_path}, expected {expected_root}")
