"""Source selection for standalone CSA tests; never imported by production code."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path


def activate() -> Path:
    """Select PTO sources and pinned model sources in the chosen interpreter.

    The existing model environment is vllm-ascend/.venv. Loading PTO's
    toolchain and editable hooks does not require PTO's own interpreter.
    """
    root_text = os.environ.get("PTO_EAGER_ROOT")
    if not root_text:
        raise RuntimeError("source pto_eager/env.sh before running the CSA tests")
    root = Path(root_text).resolve()
    repo = Path(__file__).resolve().parents[2]
    vllm_source = root / ".cache/migration-v0.25.1rc1/vllm"
    expected_sources = [("vllm_ascend", repo), ("pypto", root / "pypto")]
    if vllm_source.is_dir():
        expected_sources.append(("vllm", vllm_source))
    for name, expected in expected_sources:
        loaded = sys.modules.get(name)
        if loaded is not None and expected not in Path(loaded.__file__).resolve().parents:
            raise RuntimeError(f"{name} already loaded from {loaded.__file__}, expected {expected}")
    sys.path.insert(0, str(repo))
    if vllm_source.is_dir():
        sys.path.insert(0, str(vllm_source))
    # vLLM inspects model classes in a fresh Python subprocess. sys.path alone
    # would leave that subprocess using the old editable vLLM checkout.
    source_paths = [str(repo)]
    if vllm_source.is_dir():
        source_paths.insert(0, str(vllm_source))
    inherited_paths = os.environ.get("PYTHONPATH", "").split(os.pathsep)
    os.environ["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(source_paths + inherited_paths))

    # Match the Qwen example's process-local editable-hook repair. The global
    # user site also has PyPTO/Simpler checkouts; their hooks must not win.
    def is_other_editable(finder: object) -> bool:
        sources = getattr(finder, "known_source_files", {})
        return type(finder).__name__ == "ScikitBuildRedirectingFinder" and any(
            "/pypto/" in str(source) or "/simpler/" in str(source) for source in sources.values()
        )

    sys.meta_path[:] = [finder for finder in sys.meta_path if not is_other_editable(finder)]
    site = root / ".venv-dsv4-0251rc1/lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    for name in ("pypto", "simpler"):
        candidates = [site / f"_{name}_editable.py", site / f"_editable_skbc_{name}.py"]
        installed = [candidate for candidate in candidates if candidate.is_file()]
        if len(installed) != 1:
            raise ImportError(f"expected one editable hook for {name}, found {installed} in {site}")
        path = installed[0]
        spec = importlib.util.spec_from_file_location(f"_csa_{name}_editable", path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    return repo


def load_native_extension(repo: Path, *, cann92_abi: bool = False) -> Path:
    """Load the isolated extension built from this checkout, without installing it."""
    name = "vllm_ascend.vllm_ascend_C"
    directory = "native-cann92-install" if cann92_abi else "native-install"
    candidates = sorted((repo / ".cache/csa" / directory).glob("vllm_ascend_C*.so"))
    if len(candidates) != 1:
        raise RuntimeError(f"expected one locally built native extension, found {candidates}")
    path = candidates[0].resolve()
    loaded = sys.modules.get(name)
    if loaded is not None:
        if Path(loaded.__file__).resolve() != path:
            raise RuntimeError(f"native extension already loaded from {loaded.__file__}")
        return path
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load native extension {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[name] = module
    return path


def write_json(path: Path, value: object) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
