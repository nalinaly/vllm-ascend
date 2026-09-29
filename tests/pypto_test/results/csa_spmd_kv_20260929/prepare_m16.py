"""长T96改沿M扩核；保留N128/K512，检查是否保持L0原分块。"""

import ast
import difflib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[4]
PREFIX = WORKSPACE / ".cache/csa-spmd-kv-e110a886-v1"
PACKAGE = "dsv4_csa_spmd_kv_e110a886_v1"


def main():
    source = Path(str(PREFIX) + "-baseline")
    records = []
    for name in ("kv_m16", "kv_m16_sync"):
        dest = Path(str(PREFIX) + "-" + name)
        assert not dest.exists()
        shutil.copytree(source, dest, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "build_output"))
        path = dest / "vllm_ascend/ops/pypto" / PACKAGE / "qkv_proj_rope.py"
        before = path.read_text()
        text = before.replace(
            "KV_N_TILE = 128  #", "KV_M16_T_ROWS = 96\nKV_M16_GROUP_LIMIT = 6\n\nKV_N_TILE = 128  #", 1
        )
        begin = text.index("def _kv_project(")
        end = text.index("\n\n@pl.jit.inline(auto_scope=False)\ndef kv_proj_rope", begin)
        body = text[begin:end].replace(
            "    GROUP_ROWS: pl.constexpr,\n",
            "    GROUP_ROWS: pl.constexpr,\n    GROUP_LIMIT: pl.constexpr,\n    SYNC_START: pl.constexpr,\n",
            1,
        )
        body = body.replace(
            "pl.min(KV_OM, pl.max(1, tile_rows // GROUP_ROWS))",
            "pl.min(GROUP_LIMIT, pl.max(1, tile_rows // GROUP_ROWS))",
            1,
        )
        body = body.replace(
            '        name_hint="kv_proj_matmul",\n',
            '        name_hint="kv_proj_matmul",\n        sync_start=SYNC_START,\n',
            1,
        )
        text = text[:begin] + body + text[end:]
        for arguments in (
            "KV_DENSE_M_TILE, 2 * KV_DENSE_M_TILE",
            "KV_FIXED_SMALL_M_TILE, KV_FIXED_SMALL_M_TILE",
            "KV_DENSE_M_TILE, KV_DENSE_M_TILE",
        ):
            old = "                    " + arguments + ",\n"
            assert text.count(old) == 1
            text = text.replace(old, "                    " + arguments + ", KV_OM, False,\n", 1)
        marker = "            elif tile_rows < KV_FIXED_SMALL_ROWS:\n"
        assert text.count(marker) == 1
        sync = "True" if name.endswith("_sync") else "False"
        text = text.replace(
            marker,
            "            elif t_dim == KV_M16_T_ROWS:\n"
            "                # Six M16 groups times four N128 columns: 24 nonempty AIC workers.\n"
            "                kv_fp32 = _kv_project(\n"
            "                    x, wkv, kv_fp32, tile_base, tile_rows, t_matmul, late_dep,\n"
            f"                    KV_M_TILE, KV_M_TILE, KV_M16_GROUP_LIMIT, {sync},\n"
            "                )\n" + marker,
            1,
        )
        ast.parse(text)
        path.chmod(0o644)
        path.write_text(text)
        path.chmod(0o444)
        patch = "".join(
            difflib.unified_diff(
                before.splitlines(True),
                text.splitlines(True),
                fromfile="a/vllm_ascend/ops/pypto/deepseek_v4_flash_dspark_perf/qkv_proj_rope.py",
                tofile="b/vllm_ascend/ops/pypto/deepseek_v4_flash_dspark_perf/qkv_proj_rope.py",
            )
        )
        (ROOT / f"{name}.patch").write_text(patch)
        records.append({"name": name, "source": str(dest)})
    (ROOT / "m16_source.json").write_text(
        json.dumps(
            {
                "baseline": str(source),
                "variant": "pkg:" + PACKAGE,
                "variants": records,
                "branch": "atomic0 and T96 only; all other shapes/atomic1 unchanged",
                "scope": "long-context first; changed T96 short context will follow if worthwhile",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
