"""冻结仅调整逐行cache写回分工的候选，保持搬运体及负slot保护。"""

import ast
import difflib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[3]
WORKSPACE = REPO.parent
PREVIOUS = ROOT.parent / "csa_spmd_kv_20260929"
BASE = WORKSPACE / ".cache/csa-spmd-kv-e110a886-v1-baseline"
PACKAGE = "dsv4_csa_spmd_writeback_e110a886_v1"
PREFIX = WORKSPACE / ".cache/csa-spmd-writeback-e110a886-v1"
VARIANTS = ("baseline", "wb48", "wb48_sync")


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
        packages = dest / "vllm_ascend/ops/pypto"
        shutil.copytree(packages / "deepseek_v4_flash_dspark_perf", packages / PACKAGE)
        path = packages / PACKAGE / "decode_csa.py"
        before = path.read_text()
        text = before
        if name != "baseline":
            old = "    wb_blocks = (t_dim + CSA_WB_TOKEN_TILE - 1) // CSA_WB_TOKEN_TILE\n"
            assert text.count(old) == 1
            text = text.replace(
                old,
                "    wb_rows_per_worker = (t_dim + CSA_WB_WORKERS - 1) // CSA_WB_WORKERS\n"
                "    wb_workers = (t_dim + wb_rows_per_worker - 1) // wb_rows_per_worker\n",
                1,
            )
            begin = text.index('        with pl.spmd(TP1_CSA_WB_WORKERS, name_hint="csa_cache_writeback"):\n')
            end = text.index("\n        # Keep Q_A ahead", begin)
            old_body = text[begin:end]
            inner = old_body[old_body.index("                    write_page =") :]
            inner = "\n".join(line[4:] if line.startswith("    ") else line for line in inner.split("\n"))
            sync = ", sync_start=True" if name == "wb48_sync" else ""
            new_body = (
                f'        with pl.spmd(wb_workers, name_hint="csa_cache_writeback"{sync}):\n'
                "            wb_worker = pl.tile.get_block_idx()\n"
                "            wb_t0 = wb_worker * wb_rows_per_worker\n"
                "            for write_dt in pl.range(pl.min(wb_rows_per_worker, t_dim - wb_t0)):\n"
                "                write_t = wb_t0 + write_dt\n" + inner
            )
            text = text[:begin] + new_body + text[end:]
        ast.parse(text)
        path.chmod(0o644)
        path.write_text(text)
        if text != before:
            (ROOT / f"{name}.patch").write_text(
                "".join(
                    difflib.unified_diff(
                        before.splitlines(True),
                        text.splitlines(True),
                        fromfile="a/vllm_ascend/ops/pypto/deepseek_v4_flash_dspark_perf/decode_csa.py",
                        tofile="b/vllm_ascend/ops/pypto/deepseek_v4_flash_dspark_perf/decode_csa.py",
                    )
                )
            )
        for path in dest.rglob("*.py"):
            path.chmod(0o444)
        manifest["variants"].append({"name": name, "source": str(dest)})
    (ROOT / "source.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    for filename in ("compile.py", "compile_all.py", "compiled_case.py", "run_side.sh", "run.sh", "collect.py"):
        text = (PREVIOUS / filename).read_text().replace(PREVIOUS.name, ROOT.name)
        text = text.replace("csa-spmd-kv-e110a886-v1", "csa-spmd-writeback-e110a886-v1")
        text = text.replace("dsv4_csa_spmd_kv_e110a886_v1", PACKAGE)
        if filename == "collect.py":
            text = text.replace('"kv_n64", "kv_n64_sync", "kv_n64_linked"', '"wb48", "wb48_sync"')
            text = text.replace(
                '    if name.startswith("kv_proj_matmul"):',
                '    if name.startswith("csa_cache_writeback"):\n        return "Writeback"\n'
                '    if name.startswith("scatter_softmax_pool"):\n        return name\n'
                '    if name.startswith("kv_proj_matmul"):',
            )
            text = text.replace(
                '("KV", "Compressor", "IndexerCompressor", "idx_qr_proj_matmul")',
                '("Writeback", "qproj_matmul", "Compressor", "scatter_softmax_pool")',
            )
            text = text.replace("KV扩核与并发联动结果", "Cache写回有效48核与整组启动结果")
            text = text.replace("检查KV分工、相关并发者", "检查写回分工、相关并发者")
        (ROOT / filename).write_text(text)


if __name__ == "__main__":
    main()
