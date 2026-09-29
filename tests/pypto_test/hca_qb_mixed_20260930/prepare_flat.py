"""直接分配二维Q，避免orchestration reshape重置manual_dep。"""

import ast
import difflib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    path = ROOT / "source.json"
    manifest = json.loads(path.read_text())
    source = Path(manifest["sources"]["streamed4_edges"])
    destination = source.with_name("streamed4_flat")
    assert not destination.exists()
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    patches = []
    for name in ("decode_hca.py", "qkv_mixed.py", "q_projection_streamed.py", "decode_sparse_attn_hca.py"):
        relative = Path("deepseek_v4_flash_hca") / name
        file = destination / relative
        before = file.read_text()
        after = before.replace(
            "q: pl.Tensor[[T_DYN, H, HEAD_DIM], pl.BF16]",
            "q: pl.Tensor[[T_DYN, H * HEAD_DIM], pl.BF16]",
        )
        after = after.replace(
            "q = pl.create_tensor([tokens, H, HEAD_DIM], dtype=pl.BF16, manual_dep=True)",
            "q = pl.create_tensor([tokens, H * HEAD_DIM], dtype=pl.BF16, manual_dep=True)",
        )
        if name == "q_projection_streamed.py":
            after = after.replace("q_flat = pl.reshape(q, [t_dim, H * HEAD_DIM])", "q_flat = q")
        assert after != before
        ast.parse(after)
        file.chmod(0o644)
        file.write_text(after)
        file.chmod(0o444)
        patches.extend(
            difflib.unified_diff(
                before.splitlines(True),
                after.splitlines(True),
                fromfile="a/" + str(relative),
                tofile="b/" + str(relative),
                n=0,
            )
        )
    manifest["sources"]["streamed4_flat"] = str(destination)
    manifest["streamed_edges_limit"] = (
        "Q root manual_dep is true but orchestration reshape resets it to false; "
        "DFX proves four dequant groups serialize, so edges/two-group results do not validate independent publishing"
    )
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    (ROOT / "streamed4_flat.patch").write_text("".join(patches))


if __name__ == "__main__":
    main()
