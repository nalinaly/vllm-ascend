"""同请求两个query共享压缩KV；raw分别处理，FP32输出分块写回控制UB。"""

import argparse
import ast
import difflib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
RELATIVE = Path("deepseek_v4_flash_hca/decode_sparse_attn_hca.py")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default="pair")
    args = parser.parse_args()
    baseline = REPO.parent / ".cache/hca-flat-sync-ad0e6bbe-20260930/base"
    candidate = REPO.parent / ".cache/hca-pair-query-ad0e6bbe-20260930" / args.variant
    assert not candidate.exists(), candidate
    shutil.copytree(baseline, candidate)
    path = candidate / RELATIVE
    before = path.read_text()
    prefix, tail = before.split("def _long_sparse_attn_hca_tp1(", 1)
    body, suffix = tail.split("\n\n@pl.jit.inline(auto_scope=False)\ndef sparse_attn_hca_tp1(", 1)
    start = body.index("    # 全部历史共用一组 MIX 任务")
    replacement = (ROOT / "attention_body.py.inc").read_text()
    after = (prefix + "def _long_sparse_attn_hca_tp1(" + body[:start] + replacement
             + "\n\n@pl.jit.inline(auto_scope=False)\ndef sparse_attn_hca_tp1(" + suffix)
    ast.parse(after)
    path.chmod(0o644)
    path.write_text(after)
    path.chmod(0o444)
    (ROOT / "pair.patch").write_text("".join(difflib.unified_diff(
        before.splitlines(True), after.splitlines(True),
        fromfile="a/" + str(RELATIVE), tofile="b/" + str(RELATIVE), n=0,
    )))
    (ROOT / "source.json").write_text(json.dumps({
        "baseline_commit": "ad0e6bbe", "baseline": str(baseline), "candidate": str(candidate),
        "change": "two queries share compressed QK/PV M128; raw QK/PV M64 per query; "
                  "two L1 KV slots, one tick lookahead; FP32 output accumulation in GM by N128 chunks",
        "invariants": "same per-query compressed-before-raw order, block maxima, BF16 probabilities, "
                      "sink/causal masks and RoPE; 24 MIX groups and task dependencies unchanged",
        "reference": "ops-transformer 28f40354 SplitBalanced mBaseSize and "
                     "sparse_attn_sharedkv_swa_block_vector.h:757–826 GM FP32 accumulation",
        "limits": "experimental; requires CPU capacity, real-device precision and performance checks",
    }, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
