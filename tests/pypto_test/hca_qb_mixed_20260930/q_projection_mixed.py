"""Q_B整数投影与反量化/RMS/RoPE的双槽Cube/Vector流水候选。"""

import pypto.language as pl

from ..deepseek_v4_flash_dspark_perf.nz_mode import QUANT_WEIGHT_LAYOUT
from ..deepseek_v4_flash_dspark_perf.qkv_proj_rope import (
    EPS,
    HEAD_DIM,
    NOPE_DIM,
    PREFILL_DENSE_TILE,
    Q_LORA,
    QPROJ_T_PAD,
    ROPE_DIM,
    ROPE_DIM_SCALE,
    T_DYN,
    D,
    H,
)

QB_WORKERS = 20
QB_M = 128
QB_N = 256
QB_K = 256
QB_VEC_ROWS = 8
QB_SLOTS = 2


@pl.jit.inline(auto_scope=False)
def q_proj_q_mixed(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8, QUANT_WEIGHT_LAYOUT],
    wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32],
    rope_cos_il: pl.Tensor[[T_DYN, ROPE_DIM], pl.FP32],
    rope_sin_signed: pl.Tensor[[T_DYN, ROPE_DIM], pl.FP32],
    rope_swap_idx: pl.Tensor[[T_DYN, ROPE_DIM], pl.INT32],
    q: pl.Tensor[[T_DYN, H, HEAD_DIM], pl.BF16],
    qr_i8_matmul: pl.Tensor[[QPROJ_T_PAD, Q_LORA], pl.INT8],
    qr_scale_pad_store: pl.Tensor[[QPROJ_T_PAD, 1], pl.FP32],
    qproj_dep: pl.Scalar[pl.TASK_ID],
):
    t_dim = pl.tensor.dim(x, 0)
    q_flat = pl.reshape(q, [t_dim, H * HEAD_DIM])
    for tile_base in pl.range(0, t_dim, PREFILL_DENSE_TILE):
        tile_rows = pl.min(PREFILL_DENSE_TILE, t_dim - tile_base)
        row_blocks = (tile_rows + QB_M - 1) // QB_M
        transfer = pl.create_tensor([QB_WORKERS * QB_SLOTS * QB_M, HEAD_DIM], dtype=pl.INT32)
        ffts = pl.create_tensor([256], dtype=pl.INT64)
        with pl.spmd(
            QB_WORKERS,
            name_hint="hca_qb_mixed",
            allow_early_resolve=True,
            deps=[qproj_dep],
        ):
            worker = pl.tile.get_block_idx()
            units = (H * row_blocks - worker + QB_WORKERS - 1) // QB_WORKERS
            pl.set_cache_policy(wq_b, pl.CachePolicy.BYPASS)
            pl.system.set_ffts(ffts)
            for work in pl.range(units):
                item = worker + work * QB_WORKERS
                head = item % H
                t0 = (item // H) * QB_M
                m_rows = pl.min(QB_M, tile_rows - t0)
                transfer_row = (worker * QB_SLOTS + work % QB_SLOTS) * QB_M
                if work >= QB_SLOTS:
                    pl.system.sync_wait(1, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIC)
                # 每个head的两个N256面板都完成后才发布，RMS仍覆盖完整512列。
                for half in pl.range(HEAD_DIM // QB_N):
                    col0 = head * HEAD_DIM + half * QB_N
                    qr_first = pl.slice(qr_i8_matmul, [QB_M, QB_K], [t0, 0], valid_shape=[m_rows, QB_K])
                    wq_first = wq_b[0:QB_K, col0 : col0 + QB_N]
                    acc = pl.matmul(qr_first, wq_first, out_dtype=pl.INT32)
                    for k0 in pl.pipeline(QB_K, Q_LORA, QB_K, stage=2):
                        qr_part = pl.slice(qr_i8_matmul, [QB_M, QB_K], [t0, k0], valid_shape=[m_rows, QB_K])
                        wq_part = wq_b[k0 : k0 + QB_K, col0 : col0 + QB_N]
                        acc = pl.matmul_acc(acc, qr_part, wq_part)
                    transfer[transfer_row : transfer_row + QB_M, half * QB_N : (half + 1) * QB_N] = acc
                pl.system.sync_set(0, pipe=pl.PipeType.FIX, ffts_mode=2, core_type=pl.KernelType.AIC)
            for drain in pl.range(pl.min(units, QB_SLOTS)):
                pl.system.sync_wait(1, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIC)

            for lane in pl.split_aiv(2, mode=pl.SplitMode.NONE):
                pl.system.set_ffts(ffts)
                row_seed = pl.reshape(
                    pl.mul(pl.cast(pl.tile.arange(0, [1, QB_VEC_ROWS], dtype=pl.INT32), pl.FP32), ROPE_DIM_SCALE),
                    [QB_VEC_ROWS, 1],
                )
                row_offsets = pl.cast(
                    pl.row_expand_add(
                        pl.tile.full([QB_VEC_ROWS, ROPE_DIM], dtype=pl.FP32, value=0.0),
                        row_seed,
                    ),
                    pl.INT32,
                )
                for vec_work in pl.range(units):
                    vec_item = worker + vec_work * QB_WORKERS
                    vec_head = vec_item % H
                    vec_t0 = (vec_item // H) * QB_M
                    vec_rows = pl.min(QB_M, tile_rows - vec_t0)
                    vec_transfer = (worker * QB_SLOTS + vec_work % QB_SLOTS) * QB_M
                    vec_col0 = vec_head * HEAD_DIM
                    channel_scale = pl.reshape(pl.load(wq_b_scale, [vec_col0], [HEAD_DIM]), [1, HEAD_DIM])
                    pl.system.sync_wait(0, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIV)
                    for row_group in pl.pipeline((vec_rows + 2 * QB_VEC_ROWS - 1) // (2 * QB_VEC_ROWS), stage=2):
                        row_local = (row_group * 2 + lane) * QB_VEC_ROWS
                        valid_rows = pl.min(QB_VEC_ROWS, vec_rows - row_local)
                        if valid_rows > 0:
                            tg = vec_t0 + row_local
                            out_tg = tile_base + tg
                            acc_input = pl.load(transfer, [vec_transfer + row_local, 0], [QB_VEC_ROWS, HEAD_DIM])
                            row_scale = pl.load(qr_scale_pad_store, [tg, 0], [QB_VEC_ROWS, 1])
                            cosine = pl.load(
                                rope_cos_il, [out_tg, 0], [QB_VEC_ROWS, ROPE_DIM], valid_shape=[valid_rows, ROPE_DIM]
                            )
                            sine = pl.load(
                                rope_sin_signed,
                                [out_tg, 0],
                                [QB_VEC_ROWS, ROPE_DIM],
                                valid_shape=[valid_rows, ROPE_DIM],
                            )
                            swap = pl.load(
                                rope_swap_idx, [out_tg, 0], [QB_VEC_ROWS, ROPE_DIM], valid_shape=[valid_rows, ROPE_DIM]
                            )
                            swap_flat = pl.add(swap, row_offsets)
                            acc_fp32 = pl.cast(acc_input, pl.FP32, mode="none")
                            row_scaled = pl.row_expand_mul(acc_fp32, row_scale)
                            dequant = pl.col_expand_mul(row_scaled, channel_scale)
                            squared = pl.mul(dequant, dequant)
                            reduce_tmp = pl.create_tile([QB_VEC_ROWS, HEAD_DIM], dtype=pl.FP32)
                            sum_square = pl.row_sum(squared, reduce_tmp)
                            variance = pl.add(pl.mul(sum_square, 1.0 / HEAD_DIM), EPS)
                            if valid_rows == QB_VEC_ROWS:
                                inverse_tmp = pl.create_tile([1, QB_VEC_ROWS], dtype=pl.FP32)
                                inverse_full = pl.tile.rsqrt(pl.reshape(variance, [1, QB_VEC_ROWS]), inverse_tmp)
                                inverse = pl.yield_(pl.reshape(inverse_full, [QB_VEC_ROWS, 1]))
                            else:
                                inverse = pl.yield_(pl.recip(pl.sqrt(variance)))
                            nope = pl.tile.extract(
                                dequant, 0, 0, [QB_VEC_ROWS, NOPE_DIM], target_memory=pl.MemorySpace.Vec
                            )
                            nope_out = pl.cast(pl.row_expand_mul(nope, inverse), pl.BF16, mode="rint")
                            pl.store(pl.set_validshape(nope_out, valid_rows, NOPE_DIM), [out_tg, vec_col0], q_flat)
                            rope_raw = pl.tile.extract(
                                dequant, 0, NOPE_DIM, [QB_VEC_ROWS, ROPE_DIM], target_memory=pl.MemorySpace.Vec
                            )
                            rope_value = pl.row_expand_mul(rope_raw, inverse)
                            gather_tmp = pl.create_tile([QB_VEC_ROWS, ROPE_DIM], dtype=pl.INT32)
                            swapped = pl.tile.gather(rope_value, swap_flat, gather_tmp)
                            rope_valid = pl.set_validshape(rope_value, valid_rows, ROPE_DIM)
                            swapped_valid = pl.set_validshape(swapped, valid_rows, ROPE_DIM)
                            rotated = pl.add(pl.mul(rope_valid, cosine), pl.mul(swapped_valid, sine))
                            rope_out = pl.cast(rotated, pl.BF16, mode="rint")
                            pl.store(
                                pl.set_validshape(rope_out, valid_rows, ROPE_DIM), [out_tg, vec_col0 + NOPE_DIM], q_flat
                            )
                    pl.system.sync_set(1, pipe=pl.PipeType.MTE3, ffts_mode=2, core_type=pl.KernelType.AIV)
    return q
