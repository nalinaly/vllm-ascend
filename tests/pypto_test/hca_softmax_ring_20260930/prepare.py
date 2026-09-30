"""长档attention的max/sum在同一AIV内用UB环传递，数值和跨核握手不改。"""

import ast
import difflib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
PREFIX = REPO.parent / ".cache/hca-softmax-ring-bef9f7fa-20260930"
RELATIVE = Path("deepseek_v4_flash_hca/decode_sparse_attn_hca.py")


def main():
    assert not PREFIX.exists(), PREFIX
    baseline, candidate = PREFIX / "base", PREFIX / "ring"
    source = REPO.parent / ".cache/hca-ob-al1-bef9f7fa-20260930/base"
    shutil.copytree(source, baseline, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copytree(baseline, candidate)
    path = candidate / RELATIVE
    before = path.read_text()
    left, tail = before.split("def _long_sparse_attn_hca_tp1(", 1)
    body, right = tail.split("def sparse_attn_hca_tp1(", 1)
    changes = [
        (
            "    maxima = pl.create_tensor([transfer_rows, 1], dtype=pl.FP32)\n"
            "    totals = pl.create_tensor([transfer_rows, 1], dtype=pl.FP32)\n",
            "",
        ),
        (
            "                for vec_tick, (m_iter, l_iter, left_iter, right_iter) in pl.range(\n"
            "                    work_count + QK_PRE_LAUNCH,\n"
            "                    init_values=(running_m, running_l, running_left, running_right),\n",
            "                # Native keeps softmaxMax/Sum in UB across the delayed PV stages.\n"
            "                # These values never cross cores; only scores/probs/values need GM.\n"
            "                max_ring_init = pl.tile.full([QK_TRANSFER_SLOTS, H // 2], dtype=pl.FP32, value=0.0)\n"
            "                sum_ring_init = pl.tile.full([QK_TRANSFER_SLOTS, H // 2], dtype=pl.FP32, value=0.0)\n"
            "                for vec_tick, (m_iter, l_iter, left_iter, right_iter, max_ring, sum_ring) in pl.range(\n"
            "                    work_count + QK_PRE_LAUNCH,\n"
            "                    init_values=(running_m, running_l, running_left, running_right,\n"
            "                                 max_ring_init, sum_ring_init),\n",
        ),
        (
            "                        pl.store(maximum, [vec_row, 0], maxima)\n"
            "                        pl.store(total, [vec_row, 0], totals)\n"
            "                        pl.system.sync_set(1, pipe=pl.PipeType.MTE3, "
            "ffts_mode=2, core_type=pl.KernelType.AIV)\n",
            "                        next_max_ring = pl.tile.assemble(\n"
            "                            max_ring, pl.reshape(maximum, [1, H // 2]), "
            "[vec_tick % QK_TRANSFER_SLOTS, 0],\n"
            "                        )\n"
            "                        next_sum_ring = pl.tile.assemble(\n"
            "                            sum_ring, pl.reshape(total, [1, H // 2]), [vec_tick % QK_TRANSFER_SLOTS, 0],\n"
            "                        )\n"
            "                        pl.system.sync_set(1, pipe=pl.PipeType.MTE3, "
            "ffts_mode=2, core_type=pl.KernelType.AIV)\n"
            "                        updated_max_ring, updated_sum_ring = pl.yield_(next_max_ring, next_sum_ring)\n"
            "                    else:\n"
            "                        updated_max_ring, updated_sum_ring = pl.yield_(max_ring, sum_ring)\n",
        ),
        (
            "                        out_m = pl.load(maxima, [out_row, 0], [H // 2, 1])\n"
            "                        out_l = pl.load(totals, [out_row, 0], [H // 2, 1])\n",
            "                        out_m = pl.reshape(pl.tile.slice(\n"
            "                            updated_max_ring, [1, H // 2], [out_work % QK_TRANSFER_SLOTS, 0],\n"
            "                        ), [H // 2, 1])\n"
            "                        out_l = pl.reshape(pl.tile.slice(\n"
            "                            updated_sum_ring, [1, H // 2], [out_work % QK_TRANSFER_SLOTS, 0],\n"
            "                        ), [H // 2, 1])\n",
        ),
        (
            "                    running_m, running_l, running_left, running_right = pl.yield_("
            "m_after, l_after, left_after, right_after)\n",
            "                    running_m, running_l, running_left, running_right, "
            "max_ring_done, sum_ring_done = pl.yield_(\n"
            "                        m_after, l_after, left_after, right_after, updated_max_ring, updated_sum_ring,\n"
            "                    )\n",
        ),
    ]
    for old, new in changes:
        assert body.count(old) == 1, old
        body = body.replace(old, new)
    after = left + "def _long_sparse_attn_hca_tp1(" + body + "def sparse_attn_hca_tp1(" + right
    ast.parse(after)
    path.chmod(0o644)
    path.write_text(after)
    path.chmod(0o444)
    (ROOT / "ring.patch").write_text(
        "".join(
            difflib.unified_diff(
                before.splitlines(True),
                after.splitlines(True),
                fromfile="a/" + str(RELATIVE),
                tofile="b/" + str(RELATIVE),
            )
        )
    )
    (ROOT / "source.json").write_text(
        json.dumps(
            {
                "baseline_commit": "bef9f7fa",
                "baseline": str(baseline),
                "candidate": str(candidate),
                "change": "long attention per-block max/sum moves from GM to per-AIV UB ring",
                "numerics": "same FP32 max/sum, BF16 probabilities, merge order and FFTS; short path unchanged",
                "reference": "ops-transformer SparseAttnSharedkv SWAVectorBlock softmaxMaxBuff/softmaxSumBuff",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
