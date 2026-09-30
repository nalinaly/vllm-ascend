"""参考 Native ProcessY，混合阶段读取原始 BF16，保留线性投影 FP32 输入。"""

import argparse
import ast
import difflib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gm", action="store_true", help="保留GM混合值，独立检查BF16输入")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "source.json").read_text())
    parent = Path(manifest["baseline"] if args.gm else manifest["candidate"])
    destination = Path(manifest["candidate"]).parent / ("mix_gm_bf16" if args.gm else "mix_ub_bf16")
    assert not destination.exists(), destination
    shutil.copytree(parent, destination, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    patches = []
    for relative in (Path("deepseek_v4_flash_hca/hc_pre_fused.py"), Path("deepseek_v4_flash_hca/decode_hca.py")):
        path = destination / relative
        before = path.read_text()
        if relative.name == "hc_pre_fused.py":
            prefix, body = before.split("def hc_pre_norm(", 1)
            old = "    inv_rms: pl.Tensor[[HC_PAD_ROWS_DYN, 1], pl.FP32],\n"
            assert body.count(old) == 1
            body = body.replace(old, old + "    x_original: pl.Tensor[[T_DYN, HC_MULT, D], pl.BF16],\n")
            old = "    x_flat = pl.reshape(x, [t_dim, HC_DIM])"
            assert body.count(old) == 1
            body = body.replace(old, "    x_flat = pl.reshape(x_original, [t_dim, HC_DIM])")
            for lane, col in enumerate(("d0", "D + d0", "2 * D + d0", "3 * D + d0")):
                if args.gm:
                    old = (f"            x{lane} = pl.slice(x_flat, [T_TILE, D_TILE], [t0, {col}], "
                           "valid_shape=[valid_rows, D_TILE])")
                    new = old.replace("= pl.slice(", "= pl.cast(pl.slice(") + ", pl.FP32)"
                else:
                    old = (f"            x{lane} = pl.load(x_flat, [t0, {col}], [T_TILE, D_TILE], "
                           "valid_shape=[valid_rows, D_TILE])")
                    new = old.replace("= pl.load(", "= pl.cast(pl.load(") + ", pl.FP32)"
                assert body.count(old) == 1
                body = body.replace(old, new)
            after = prefix + "def hc_pre_norm(" + body
        else:
            old = "normalized, False, inv_rms)"
            assert before.count(old) == 1
            after = before.replace(old, "normalized, False, inv_rms, x_hc)")
        ast.parse(after)
        path.chmod(0o644)
        path.write_text(after)
        path.chmod(0o444)
        patches.extend(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
                                          fromfile="a/" + str(relative), tofile="b/" + str(relative), n=0))
    label = "gm_bf16" if args.gm else "ub_bf16"
    (ROOT / f"{label}.patch").write_text("".join(patches))
    manifest.update(parent_candidate=str(parent), candidate=str(destination),
                    change="mix直接读原始BF16并无损转FP32，线性投影输入不变", mixed_storage="GM" if args.gm else "UB")
    (ROOT / f"source_{label}.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
