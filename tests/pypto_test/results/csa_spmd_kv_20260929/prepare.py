"""冻结KV有效N向扩核、sync及真实并发Compressor联动候选。"""

import ast
import difflib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[3]
WORKSPACE = REPO.parent
BASE = WORKSPACE / ".cache/csa-spmd-pipeline-sync-46cec3a0-v1-o_post_t96_sync"
OLD_PACKAGE = "dsv4_csa_spmd_pipeline_sync_46cec3a0_v1"
PACKAGE = "dsv4_csa_spmd_kv_e110a886_v1"
PREFIX = WORKSPACE / ".cache/csa-spmd-kv-e110a886-v1"
VARIANTS = ("baseline", "kv_n64", "kv_n64_sync", "kv_n64_linked")


def add_sync(text, name, value="True"):
    marker = f'        name_hint="{name}",\n'
    assert text.count(marker) == 1
    return text.replace(marker, marker + f"        sync_start={value},\n", 1)


def main():
    manifest = {
        "production": "e110a886",
        "base_source": str(BASE),
        "source_prefix": str(PREFIX),
        "variant": "pkg:" + PACKAGE,
        "cases": [[131072, 16], [8192, 24]],
        "metrics": ["max_us", "min_us", "mean_us"],
        "variants": [],
    }
    for name in VARIANTS:
        dest = Path(str(PREFIX) + "-" + name)
        assert not dest.exists(), dest
        shutil.copytree(BASE, dest, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "build_output"))
        # Reuse only the test skeleton; all operator/shared code comes from HEAD.
        shutil.rmtree(dest / "vllm_ascend")
        shutil.copytree(
            REPO / "vllm_ascend", dest / "vllm_ascend", ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
        )
        packages = dest / "vllm_ascend/ops/pypto"
        shutil.copytree(packages / "deepseek_v4_flash_dspark_perf", packages / PACKAGE)
        patch = []
        for filename in ("qkv_proj_rope.py", "decode_compressor_ratio4.py", "decode_indexer_compressor.py"):
            path = packages / PACKAGE / filename
            before = path.read_text()
            text = before
            if filename == "qkv_proj_rope.py" and name != "baseline":
                assert text.count("KV_N_TILE = 128  #") == 1
                text = text.replace("KV_N_TILE = 128  #", "KV_N_TILE = 128 if ATOMIC_ADD else 64  #", 1)
                if name != "kv_n64":
                    text = text.replace("KV_OM = 3  #", "KV_SYNC_START = ATOMIC_ADD == 0\n\nKV_OM = 3  #", 1)
                    text = add_sync(text, "kv_proj_matmul", "KV_SYNC_START")
            if name == "kv_n64_linked" and filename in ("decode_compressor_ratio4.py", "decode_indexer_compressor.py"):
                text = add_sync(text, "kv_score_proj")
            ast.parse(text)
            path.chmod(0o644)
            path.write_text(text)
            patch.extend(
                difflib.unified_diff(
                    before.splitlines(True),
                    text.splitlines(True),
                    fromfile="a/vllm_ascend/ops/pypto/deepseek_v4_flash_dspark_perf/" + filename,
                    tofile="b/vllm_ascend/ops/pypto/deepseek_v4_flash_dspark_perf/" + filename,
                )
            )
        for path in dest.rglob("*.py"):
            path.chmod(0o444)
        if patch:
            (ROOT / f"{name}.patch").write_text("".join(patch))
        manifest["variants"].append({"name": name, "source": str(dest)})
    (ROOT / "source.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    previous = ROOT.parent / "csa_spmd_post_balanced_20260929"
    for filename in ("compile.py", "compiled_case.py", "run_side.sh"):
        text = (previous / filename).read_text().replace(previous.name, ROOT.name)
        text = text.replace("csa-spmd-pipeline-sync-46cec3a0-v1", "csa-spmd-kv-e110a886-v1")
        text = text.replace(OLD_PACKAGE, PACKAGE)
        if filename == "compiled_case.py":
            marker = "                    def record_options(backend_config, values):\n"
            text = text.replace(
                marker, marker + '                        values["options"]["inplace_pass"] = True\n', 1
            )
        (ROOT / filename).write_text(text)
    shutil.copyfile(ROOT.parent / "csa_spmd_pipeline_sync_20260929/compile_all.py", ROOT / "compile_all.py")


if __name__ == "__main__":
    main()
