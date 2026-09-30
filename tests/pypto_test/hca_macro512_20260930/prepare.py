"""参考Native S2=512，四个128列Cube块共用一次softmax/PV输出。"""

import ast
import difflib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PREFIX = ROOT.parents[3] / ".cache/hca-macro512-5a5b9bfb-20260930"
RELATIVE = Path("deepseek_v4_flash_hca/decode_sparse_attn_hca.py")

CUBE_BODY = '''        worker = pl.tile.get_block_idx()
        pl.system.set_ffts(ffts)
        qk_kv = pl.create_tile([ATTN_K_TILE, HEAD_DIM], dtype=pl.BF16, target_memory=pl.MemorySpace.Mat)
        pv_kv = pl.create_tile([ATTN_K_TILE, HEAD_DIM], dtype=pl.BF16, target_memory=pl.MemorySpace.Mat)
        for token, (work_offset, old_request, old_block, old_parts, old_drop) in pl.range(
            worker, t_dim, NUM_QK_CORES,
            init_values=(pl.cast(0, pl.INDEX), pl.cast(0, pl.INDEX), pl.cast(0, pl.INDEX),
                         pl.cast(1, pl.INDEX), pl.cast(0, pl.INDEX)),
        ):
            request = token // S
            length = pl.max(pl.read(kv_seq_lens, [request]), 0)
            position = pl.max(pl.read(position_ids, [token]), -1)
            cmp_rows = pl.cast(pl.min(HCA_MAX_COMPRESSED_ROWS, pl.min(
                (position + 1) // COMPRESS_RATIO, length // COMPRESS_RATIO,
            )), pl.INDEX)
            cmp_blocks = (cmp_rows + MACRO_K - 1) // MACRO_K
            work_count = cmp_blocks + 1
            first_length = pl.min(WIN, pl.max(pl.min(pl.read(position_ids, [request * S]) + 1, length), 0))
            raw_drop = pl.cast(pl.max(first_length + token % S - WIN, 0), pl.INDEX)
            query = pl.load(q_flat, [token * H, 0], [H, HEAD_DIM], target_memory=pl.MemorySpace.Mat)
            if token + NUM_QK_CORES >= t_dim:
                extra = pl.yield_(1)
            else:
                extra = pl.yield_(0)
            for tick, (prev_request, prev_block, prev_parts, prev_drop) in pl.range(
                work_count + extra, init_values=(old_request, old_block, old_parts, old_drop),
            ):
                global_tick = work_offset + tick
                if tick < work_count:
                    row = (worker * MACRO_SLOTS + global_tick % MACRO_SLOTS) * H
                    if tick == cmp_blocks:
                        qk_kv = pl.gather_row(qk_kv, raw_kv, [0, 0],
                                             [request * REQUEST_KV_ROWS + raw_drop, 0], [ATTN_K_TILE, HEAD_DIM])
                        raw_key = pl.tile.transpose_view(qk_kv)
                        raw_qk = pl.matmul(query, raw_key, out_dtype=pl.FP32)
                        pl.store(raw_qk, [row, 0], scores)
                        next_request, next_block, next_parts, next_drop = pl.yield_(
                            request, pl.cast(-1, pl.INDEX), pl.cast(1, pl.INDEX), raw_drop,
                        )
                    else:
                        parts = pl.min(MACRO_PARTS, (cmp_rows - tick * MACRO_K + ATTN_K_TILE - 1) // ATTN_K_TILE)
                        for part in pl.range(parts):
                            cmp_row = (request * cmp_work_count + tick * MACRO_PARTS + part) * ATTN_K_TILE
                            qk_kv = pl.gather_row(qk_kv, cmp_work_kv, [0, 0], [cmp_row, 0], [ATTN_K_TILE, HEAD_DIM])
                            key = pl.tile.transpose_view(qk_kv)
                            qk = pl.matmul(query, key, out_dtype=pl.FP32)
                            pl.store(qk, [row, part * ATTN_K_TILE], scores)
                        next_request, next_block, next_parts, next_drop = pl.yield_(request, tick, parts, raw_drop)
                    pl.system.sync_set(0, pipe=pl.PipeType.FIX, ffts_mode=2, core_type=pl.KernelType.AIC)
                    saved_request, saved_block, saved_parts, saved_drop = pl.yield_(
                        next_request, next_block, next_parts, next_drop,
                    )
                else:
                    saved_request, saved_block, saved_parts, saved_drop = pl.yield_(
                        prev_request, prev_block, prev_parts, prev_drop,
                    )
                if global_tick >= 1:
                    pv_row = (worker * MACRO_SLOTS + (global_tick - 1) % MACRO_SLOTS) * H
                    pl.system.sync_wait(1, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIC)
                    if prev_block < 0:
                        pv_kv = pl.gather_row(pv_kv, raw_kv, [0, 0],
                                             [prev_request * REQUEST_KV_ROWS + prev_drop, 0], [ATTN_K_TILE, HEAD_DIM])
                    else:
                        pv_source = (prev_request * cmp_work_count + prev_block * MACRO_PARTS) * ATTN_K_TILE
                        pv_kv = pl.gather_row(pv_kv, cmp_work_kv, [0, 0], [pv_source, 0], [ATTN_K_TILE, HEAD_DIM])
                    probability = pl.load(probs, [pv_row, 0], [H, ATTN_K_TILE], target_memory=pl.MemorySpace.Mat)
                    left = pl.tile.move(probability, target_memory=pl.MemorySpace.Left)
__INITIAL_ACC__
                    for part, (acc0, acc1, acc2, acc3) in pl.range(
                        1, prev_parts, init_values=(first0, first1, first2, first3),
                    ):
                        source_row = (prev_request * cmp_work_count + prev_block * MACRO_PARTS + part) * ATTN_K_TILE
                        pv_kv = pl.gather_row(pv_kv, cmp_work_kv, [0, 0], [source_row, 0], [ATTN_K_TILE, HEAD_DIM])
                        probability_part = pl.load(probs, [pv_row, part * ATTN_K_TILE], [H, ATTN_K_TILE],
                                                   target_memory=pl.MemorySpace.Mat)
                        left_part = pl.tile.move(probability_part, target_memory=pl.MemorySpace.Left)
__UPDATE_ACC__
                        done0, done1, done2, done3 = pl.yield_(update0, update1, update2, update3)
__STORE_ACC__
                    pl.system.sync_set(2, pipe=pl.PipeType.FIX, ffts_mode=2, core_type=pl.KernelType.AIC)
                end_request, end_block, end_parts, end_drop = pl.yield_(
                    saved_request, saved_block, saved_parts, saved_drop,
                )
            end_offset, carried_request, carried_block, carried_parts, carried_drop = pl.yield_(
                work_offset + work_count, end_request, end_block, end_parts, end_drop,
            )

'''

SOFTMAX = '''                        for head_part, (head_max_ring, head_sum_ring) in pl.range(
                            H // 2 // H_TILE, init_values=(max_ring, sum_ring),
                        ):
                            head_offset = head_part * H_TILE
                            previous_max = pl.reshape(pl.tile.extract(
                                pl.reshape(softmax_m_iter, [H // 2 // H_TILE, H_TILE]),
                                head_part, 0, [1, H_TILE], target_memory=pl.MemorySpace.Vec,
                            ), [H_TILE, 1])
                            if vec_tick == cmp_blocks:
                                raw_tmp = pl.create_tile([H_TILE, ATTN_K_TILE], dtype=pl.FP32)
                                raw_score = pl.load(scores, [vec_row + head_offset, 0], [H_TILE, ATTN_K_TILE])
                                raw_mask = pl.load(raw_valid, [vec_token, 0], [1, ATTN_K_TILE])
                                raw_scaled = pl.col_expand_add(pl.mul(raw_score, SOFTMAX_SCALE),
                                                              pl.mul(pl.sub(raw_mask, 1.0), -NEG_INF))
                                raw_max = pl.maximum(previous_max, pl.row_max(raw_scaled, raw_tmp))
                                raw_exp = pl.col_expand_mul(pl.exp(pl.row_expand_sub(raw_scaled, raw_max)), raw_mask)
                                raw_sum = pl.row_sum(raw_exp, raw_tmp)
                                pl.store(pl.cast(raw_exp, pl.BF16, mode="rint"), [vec_row + head_offset, 0], probs)
                                maximum, total = pl.yield_(raw_max, raw_sum)
                            else:
                                cmp_tmp = pl.create_tile([H_TILE, MACRO_K], dtype=pl.FP32)
                                valid_cols = pl.min(MACRO_K, cmp_rows - vec_tick * MACRO_K)
                                mask_parts = (valid_cols + ATTN_K_TILE - 1) // ATTN_K_TILE
                                mask_grid = pl.load(cmp_work_valid,
                                                    [request * cmp_work_count + vec_tick * MACRO_PARTS, 0],
                                                    [MACRO_PARTS, ATTN_K_TILE],
                                                    valid_shape=[mask_parts, ATTN_K_TILE])
                                mask_grid = pl.fillpad(mask_grid, pad_value=pl.PadValue.zero)
                                cmp_mask = pl.reshape(mask_grid, [1, MACRO_K])
                                score = pl.load(scores, [vec_row + head_offset, 0], [H_TILE, MACRO_K],
                                                valid_shape=[H_TILE, valid_cols])
                                scaled = pl.col_expand_add(pl.mul(score, SOFTMAX_SCALE),
                                                           pl.mul(pl.sub(cmp_mask, 1.0), -NEG_INF))
                                masked = pl.fillpad(pl.set_validshape(scaled, H_TILE, valid_cols),
                                                    pad_value=pl.PadValue.min)
                                cmp_max = pl.maximum(previous_max, pl.row_max(masked, cmp_tmp))
                                cmp_exp = pl.col_expand_mul(pl.exp(pl.row_expand_sub(masked, cmp_max)), cmp_mask)
                                cmp_sum = pl.row_sum(cmp_exp, cmp_tmp)
                                pl.store(pl.cast(cmp_exp, pl.BF16, mode="rint"), [vec_row + head_offset, 0], probs)
                                maximum, total = pl.yield_(cmp_max, cmp_sum)
                            row_slot = (vec_global % MACRO_SLOTS) * (H // 2 // H_TILE) + head_part
                            next_max_ring = pl.tile.assemble(
                                head_max_ring, pl.reshape(maximum, [1, H_TILE]), [row_slot, 0],
                            )
                            next_sum_ring = pl.tile.assemble(
                                head_sum_ring, pl.reshape(total, [1, H_TILE]), [row_slot, 0],
                            )
                            completed_max, completed_sum = pl.yield_(next_max_ring, next_sum_ring)
                        softmax_complete = pl.reshape(pl.tile.extract(
                            completed_max, (vec_global % MACRO_SLOTS) * (H // 2 // H_TILE), 0,
                            [H // 2 // H_TILE, H_TILE], target_memory=pl.MemorySpace.Vec,
                        ), [H // 2, 1])
'''


def once(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new)


def main():
    assert not PREFIX.exists()
    old_manifest = json.loads((ROOT.parent / "hca_online_softmax_20260930/source.json").read_text())
    base = Path(old_manifest["sources"]["online"])
    target = PREFIX / "macro512"
    shutil.copytree(base, target)
    before = (base / RELATIVE).read_text()
    start = before.index("    # 全部历史共用一组 MIX 任务", before.index("def _long_sparse_attn_hca_tp1("))
    cube_start = before.index("        worker = pl.tile.get_block_idx()", start)
    vec_start = before.index("        for lane in pl.split_aiv(2, mode=pl.SplitMode.NONE):", cube_start)
    end = before.index("\n\n@pl.jit.inline", vec_start)
    header = before[start:cube_start].replace("QK_TRANSFER_SLOTS", "MACRO_SLOTS")
    header = header.replace("[transfer_rows, ATTN_K_TILE]", "[transfer_rows, MACRO_K]")
    cube = CUBE_BODY
    initial, update, stores = [], [], []
    for col in range(4):
        initial.extend([
            f"                    right{col} = pl.tile.extract(pv_kv, 0, {col * 128}, [ATTN_K_TILE, PV_N_TILE],",
            "                                                   target_memory=pl.MemorySpace.Right)",
            f"                    first{col} = pl.tile.matmul(left, right{col})",
        ])
        update.extend([
            f"                        right_update{col} = pl.tile.extract("
            f"pv_kv, 0, {col * 128}, [ATTN_K_TILE, PV_N_TILE],",
            "                                                             target_memory=pl.MemorySpace.Right)",
            f"                        update{col} = pl.tile.matmul_acc(acc{col}, left_part, right_update{col})",
        ])
        stores.append(f"                    pl.store(done{col}, [pv_row, {col * 128}], values)")
    cube = cube.replace("__INITIAL_ACC__", "\n".join(initial))
    cube = cube.replace("__UPDATE_ACC__", "\n".join(update)).replace("__STORE_ACC__", "\n".join(stores))
    vec = before[vec_start:end].replace("QK_TRANSFER_SLOTS", "MACRO_SLOTS").replace("QK_PRE_LAUNCH", "MACRO_PRELAUNCH")
    vec = once(vec, "            reduce_tmp = pl.create_tile([H // 2, ATTN_K_TILE], dtype=pl.FP32)\n", "")
    vec = once(vec, "cmp_blocks = pl.min(cmp_work_count, (cmp_rows + ATTN_K_TILE - 1) // ATTN_K_TILE)",
               "cmp_blocks = (cmp_rows + MACRO_K - 1) // MACRO_K")
    vec = vec.replace("[MACRO_SLOTS, H // 2]", "[MACRO_SLOTS * (H // 2 // H_TILE), H_TILE]")
    first = vec.index("                        score = pl.load(scores,")
    last = vec.index("                        # 描述符仅供本AIV消费", first)
    vec = vec[:first] + SOFTMAX + vec[last:]
    vec = once(vec, "pl.yield_(next_max_ring, next_sum_ring, maximum)",
               "pl.yield_(completed_max, completed_sum, softmax_complete)")
    for name in ("updated_max_ring", "updated_sum_ring"):
        vec = once(vec, f"{name}, out_work % MACRO_SLOTS, 0, [1, H // 2],",
                   f"{name}, (out_work % MACRO_SLOTS) * (H // 2 // H_TILE), 0, [H // 2 // H_TILE, H_TILE],")
    after = before[:start] + header + cube + vec + before[end:]
    after = once(after, "PV_N_TILE = 128",
                 "PV_N_TILE = 128\nMACRO_K = 512\nMACRO_PARTS = 4\nMACRO_SLOTS = 2\nMACRO_PRELAUNCH = 1")
    ast.parse(after)
    path = target / RELATIVE
    path.chmod(0o644)
    path.write_text(after)
    path.chmod(0o444)
    (ROOT / "macro512.patch").write_text("".join(difflib.unified_diff(
        before.splitlines(True), after.splitlines(True), n=0,
        fromfile="a/" + str(RELATIVE), tofile="b/" + str(RELATIVE),
    )))
    (ROOT / "source.json").write_text(json.dumps({
        "sources": {"base": str(base), "macro512": str(target), "production": old_manifest["sources"]["base"]},
        "baseline": "single-query online candidate, not production",
        "change": ("S2=512 softmax; four 128-column QK chunks; "
                   "PV K128 accumulation in four L0C tiles; cross-query stream"),
        "invariants": "cache/state, masks, sink, per-query identity, inverse RoPE; no shared toolchain changes",
        "arithmetic": ("larger softmax block before BF16 and Cube PV accumulation change rounding; "
                       "diagnostic output only"),
        "tradeoff": "reload KV for PV instead of permanent L1 ring; cut PV GM round trips and FFTS handoffs",
    }, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
