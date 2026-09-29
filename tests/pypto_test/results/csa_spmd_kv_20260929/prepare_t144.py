"""预编译按T144隔离KV分工的候选；长B24实测前不写入生产。"""

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
    dest = Path(str(PREFIX) + "-kv_t144")
    assert not dest.exists()
    shutil.copytree(source, dest, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "build_output"))
    path = dest / "vllm_ascend/ops/pypto" / PACKAGE / "qkv_proj_rope.py"
    before = path.read_text()
    text = before.replace(
        "KV_N_TILE = 128  #", "KV_BALANCED_T_ROWS = 144\nKV_BALANCED_N_TILE = 64\n\nKV_N_TILE = 128  #", 1
    )
    begin = text.index("def _kv_project(")
    end = text.index("\n\n@pl.jit.inline(auto_scope=False)\ndef kv_proj_rope", begin)
    body = text[begin:end].replace("KV_N_TILE", "N_TILE")
    body = body.replace(
        "    GROUP_ROWS: pl.constexpr,\n",
        "    GROUP_ROWS: pl.constexpr,\n    N_TILE: pl.constexpr,\n    SYNC_START: pl.constexpr,\n",
        1,
    )
    body = body.replace(
        '        name_hint="kv_proj_matmul",\n',
        '        name_hint="kv_proj_matmul",\n        sync_start=SYNC_START,\n',
        1,
    )
    text = text[:begin] + body + text[end:]
    # The atomic and small-M paths keep exactly the old N128 unsynchronized policy.
    text = text.replace(
        "                    KV_DENSE_M_TILE, 2 * KV_DENSE_M_TILE,\n",
        "                    KV_DENSE_M_TILE, 2 * KV_DENSE_M_TILE, KV_N_TILE, False,\n",
        1,
    )
    text = text.replace(
        "                    KV_FIXED_SMALL_M_TILE, KV_FIXED_SMALL_M_TILE,\n",
        "                    KV_FIXED_SMALL_M_TILE, KV_FIXED_SMALL_M_TILE, KV_N_TILE, False,\n",
        1,
    )
    marker = "            else:\n                kv_fp32 = _kv_project(\n"
    assert text.count(marker) == 1
    text = text.replace(
        marker,
        "            elif t_dim == KV_BALANCED_T_ROWS:\n"
        "                # T144 keeps the M64 main tile and M16 tail; N64 gives 16 real workers.\n"
        "                kv_fp32 = _kv_project(\n"
        "                    x, wkv, kv_fp32, tile_base, tile_rows, t_matmul, late_dep,\n"
        "                    KV_DENSE_M_TILE, KV_DENSE_M_TILE, KV_BALANCED_N_TILE, True,\n"
        "                )\n" + marker,
        1,
    )
    text = text.replace(
        "                    KV_DENSE_M_TILE, KV_DENSE_M_TILE,\n",
        "                    KV_DENSE_M_TILE, KV_DENSE_M_TILE, KV_N_TILE, False,\n",
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
    (ROOT / "kv_t144.patch").write_text(patch)
    (ROOT / "t144_branch_source.json").write_text(
        json.dumps(
            {
                "baseline": str(source),
                "candidate": str(dest),
                "variant": "pkg:" + PACKAGE,
                "branch": "atomic0 and T144; all other shapes/atomic1 retain original N128/no-sync",
                "status": "private candidate only; long B24 acceptance pending",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
