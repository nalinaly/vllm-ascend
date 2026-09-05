# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 KV Compressor (decode incremental, ratio=4 overlap).

Uses overlapping state layout with 8 slots.
Front slots 0-3 at columns [0:HEAD_DIM], back slots 4-7 at columns [HEAD_DIM:OUT_DIM].
Tree reduction for softmax+pool. State shift after compression."""

import pypto.language as pl

from .config import (
    BLOCK_SIZE,
    C4A_COMPRESSOR_BLOCK_SIZE,
    CSA_STATE_PHYSICAL_BLOCKS,
    DECODE_BATCH,
    DECODE_SEQ,
    FP32_NEG_INF,
    KV_CMP_BLOCK_NUM,
    KV_CMP_MAX_BLOCKS,
    TP,
)
from .config import (
    FLASH as M,
)

# Dynamic shape variables.

# model config
B = DECODE_BATCH // TP
S = DECODE_SEQ
EPS = M.rms_norm_eps
D = M.hidden_size
HEAD_DIM = M.head_dim
HEAD_DIM_INV = 1.0 / HEAD_DIM
ROPE_HEAD_DIM = M.qk_rope_head_dim
NOPE_HEAD_DIM = M.nope_head_dim
MAX_SEQ_LEN = M.max_position_embeddings

# kernel-local (ratio-4 overlapping compressor)
COMPRESS_RATIO = 4
OVERLAP = COMPRESS_RATIO == 4
COFF = 1 + int(OVERLAP)
OUT_DIM = COFF * HEAD_DIM
STATE_LEN = COFF * COMPRESS_RATIO
IDX_KV_LEN = MAX_SEQ_LEN // COMPRESS_RATIO
COMPRESS_STATE_BLOCK_SIZE = C4A_COMPRESSOR_BLOCK_SIZE
COMPRESS_STATE_PHYSICAL_BLOCKS = CSA_STATE_PHYSICAL_BLOCKS
COMPRESS_STATE_MAX_BLOCKS = (MAX_SEQ_LEN + COMPRESS_STATE_BLOCK_SIZE - 1) // COMPRESS_STATE_BLOCK_SIZE
COMPRESS_STATE_BLOCK_NUM = COMPRESS_STATE_PHYSICAL_BLOCKS
COMPRESS_STATE_DIM = 2 * OUT_DIM
CMP_MAX_BLOCKS = KV_CMP_MAX_BLOCKS
CMP_BLOCK_NUM = KV_CMP_BLOCK_NUM

# tiling
ROPE_TILE = 32
K_TILE = 512
OUT_TILE = 64
B_TILE = 8
MM_B_TILE = 16
BS_PAD = ((B * S + MM_B_TILE - 1) // MM_B_TILE) * MM_B_TILE
HEAD_TILE = 64
HEAD_DIM_TILE = 128
RMS_PAD_TILE = 16  # 16-row block of B (min M for FP32 vec ops)
CMP_WRITES_PER_REQUEST = S // COMPRESS_RATIO
assert CMP_WRITES_PER_REQUEST == 2
RMS_PAD_BLOCKS = (B * CMP_WRITES_PER_REQUEST + RMS_PAD_TILE - 1) // RMS_PAD_TILE
RMS_PAD_ROWS = RMS_PAD_BLOCKS * RMS_PAD_TILE


@pl.jit.inline
def _main_scatter_softmax_pool(
    compress_state: pl.Tensor,
    compress_state_block_table: pl.Tensor,
    cmp4_kv_proj_pad: pl.Tensor,
    cmp4_score_proj_pad: pl.Tensor,
    ape: pl.Tensor,
    position_ids: pl.Tensor,
    kv_seq_lens: pl.Tensor,
    state_page_stride: pl.Scalar[pl.INDEX],
):
    """Scatter and pool one independent request per AIV block."""
    b_dim = pl.tensor.dim(compress_state_block_table, 0)
    s_dim = pl.tensor.dim(position_ids, 0) // b_dim
    # scatter_softmax_pool: per batch, scatter the padded proj rows into compress_state, then
    # online-softmax pool that batch's window into pooled_kv. One region -- each batch's pool
    # reads only its own just-scattered state (per-batch block table), so no cross-task barrier.
    pooled_kv = pl.create_tensor([RMS_PAD_ROWS, HEAD_DIM], dtype=pl.FP32)
    for c_idx in pl.spmd(
        b_dim,
        name_hint="scatter_softmax_pool",
        allow_early_resolve=True,
    ):
        # Uniform ACLGraph padding leaves the physical table row zeroed.
        # The replay-varying device seq-lens buffer is the authoritative
        # active predicate.  Do not use a runtime ``continue`` here: the
        # current DSL lowers its conditional body but does not suppress
        # the remainder of this unrolled request iteration.
        if pl.read(kv_seq_lens, [c_idx]) <= 0:
            for write_index in pl.range(CMP_WRITES_PER_REQUEST):
                pool_row = c_idx * CMP_WRITES_PER_REQUEST + write_index
                pooled_kv[pool_row : pool_row + 1, 0:HEAD_DIM] = pl.full([1, HEAD_DIM], dtype=pl.FP32, value=0.0)
        for s_sc in pl.pipeline(s_dim, stage=2):
            token_pos = pl.read(position_ids, [c_idx * s_dim + s_sc])
            proj_row = c_idx * s_dim + s_sc
            token_ape_row = pl.cast(token_pos % COMPRESS_RATIO, target_type=pl.INDEX)
            state_logical_block = token_pos // COMPRESS_STATE_BLOCK_SIZE
            state_block_i32 = pl.read(
                compress_state_block_table,
                [c_idx, state_logical_block],
            )
            if pl.read(kv_seq_lens, [c_idx]) <= 0:
                state_block_i32 = pl.cast(-1, pl.INT32)
            if state_block_i32 >= 0:
                state_block = pl.cast(state_block_i32, pl.INDEX)
                state_intra = pl.cast(
                    token_pos % COMPRESS_STATE_BLOCK_SIZE,
                    pl.INDEX,
                )
                kv_tile = cmp4_kv_proj_pad[proj_row : proj_row + 1, 0:OUT_DIM]
                score_tile = cmp4_score_proj_pad[proj_row : proj_row + 1, 0:OUT_DIM]
                ape_tile = ape[token_ape_row : token_ape_row + 1, 0:OUT_DIM]
                score_tile = pl.add(score_tile, ape_tile)
                state_base = state_block * state_page_stride + state_intra * COMPRESS_STATE_DIM
                compress_state[:, state_base : state_base + OUT_DIM] = kv_tile
                compress_state[
                    :,
                    state_base + OUT_DIM : state_base + 2 * OUT_DIM,
                ] = score_tile

        first_pos_b = pl.read(position_ids, [c_idx * s_dim])
        pos_b = first_pos_b % COMPRESS_RATIO
        first_boundary_s = COMPRESS_RATIO - 1 - pos_b
        # S=8 closes exactly two ratio-4 windows for every request,
        # regardless of the first position's modulo.  The old decode
        # implementation only emitted the first one, which zero-weight
        # smoke tests could not expose.
        for write_index in pl.range(CMP_WRITES_PER_REQUEST):
            pool_row = c_idx * CMP_WRITES_PER_REQUEST + write_index
            boundary_s = first_boundary_s + write_index * COMPRESS_RATIO
            write_pos = first_pos_b + boundary_s
            cur_window_start = write_pos + 1 - COMPRESS_RATIO
            prev_window_start = cur_window_start - COMPRESS_RATIO

            last_blk_off = write_pos // COMPRESS_STATE_BLOCK_SIZE
            last_intra = write_pos % COMPRESS_STATE_BLOCK_SIZE
            last_blk_id = pl.cast(
                pl.read(
                    compress_state_block_table,
                    [c_idx, last_blk_off],
                ),
                pl.INDEX,
            )
            last_base = last_blk_id * state_page_stride + last_intra * COMPRESS_STATE_DIM
            mi = compress_state[
                :,
                last_base + OUT_DIM + HEAD_DIM : last_base + COMPRESS_STATE_DIM,
            ]
            li = pl.exp(pl.sub(mi, mi))
            oi = compress_state[
                :,
                last_base + HEAD_DIM : last_base + OUT_DIM,
            ]

            for pool_s in pl.range(COMPRESS_RATIO):
                prev_abs = prev_window_start + pool_s
                front_score = pl.full(
                    [1, HEAD_DIM],
                    dtype=pl.FP32,
                    value=FP32_NEG_INF,
                )
                front_kv = pl.full(
                    [1, HEAD_DIM],
                    dtype=pl.FP32,
                    value=0.0,
                )
                if write_pos >= 2 * COMPRESS_RATIO - 1:
                    prev_blk_off = prev_abs // COMPRESS_STATE_BLOCK_SIZE
                    prev_intra = prev_abs % COMPRESS_STATE_BLOCK_SIZE
                    prev_blk_id = pl.cast(
                        pl.read(
                            compress_state_block_table,
                            [c_idx, prev_blk_off],
                        ),
                        pl.INDEX,
                    )
                    prev_base = prev_blk_id * state_page_stride + prev_intra * COMPRESS_STATE_DIM
                    front_score = compress_state[
                        :,
                        prev_base + OUT_DIM : prev_base + OUT_DIM + HEAD_DIM,
                    ]
                    front_kv = compress_state[
                        :,
                        prev_base : prev_base + HEAD_DIM,
                    ]
                mi_next_front = pl.maximum(mi, front_score)
                alpha_front = pl.exp(pl.sub(mi, mi_next_front))
                beta_front = pl.exp(pl.sub(front_score, mi_next_front))
                li = pl.add(pl.mul(alpha_front, li), beta_front)
                oi = pl.add(
                    pl.mul(oi, alpha_front),
                    pl.mul(front_kv, beta_front),
                )
                mi = mi_next_front

            for pool_s in pl.range(COMPRESS_RATIO - 1):
                cur_abs = cur_window_start + pool_s
                cur_blk_off = cur_abs // COMPRESS_STATE_BLOCK_SIZE
                cur_intra = cur_abs % COMPRESS_STATE_BLOCK_SIZE
                cur_blk_id = pl.cast(
                    pl.read(
                        compress_state_block_table,
                        [c_idx, cur_blk_off],
                    ),
                    pl.INDEX,
                )
                cur_base = cur_blk_id * state_page_stride + cur_intra * COMPRESS_STATE_DIM
                back_score = compress_state[
                    :,
                    cur_base + OUT_DIM + HEAD_DIM : cur_base + COMPRESS_STATE_DIM,
                ]
                back_kv = compress_state[
                    :,
                    cur_base + HEAD_DIM : cur_base + OUT_DIM,
                ]
                mi_next_back = pl.maximum(mi, back_score)
                alpha_back = pl.exp(pl.sub(mi, mi_next_back))
                beta_back = pl.exp(pl.sub(back_score, mi_next_back))
                li = pl.add(pl.mul(alpha_back, li), beta_back)
                oi = pl.add(
                    pl.mul(oi, alpha_back),
                    pl.mul(back_kv, beta_back),
                )
                mi = mi_next_back

            pooled_kv[pool_row : pool_row + 1, 0:HEAD_DIM] = pl.div(
                oi,
                li,
            )

    return pooled_kv


@pl.jit.inline
def compressor_ratio4(
    x: pl.Tensor,
    kv: pl.Tensor,
    # Canonical [1, physical_span] storage alias.  Page padding is addressed
    # explicitly because outlined AICore children do not consume runtime
    # ChipTensor strides.
    compress_state: pl.Tensor,
    compress_state_block_table: pl.Tensor,
    wkv: pl.Tensor[[OUT_DIM, D], pl.BF16],
    wgate: pl.Tensor[[OUT_DIM, D], pl.BF16],
    ape: pl.Tensor[[COMPRESS_RATIO, OUT_DIM], pl.FP32],
    norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    # Native full-width interleaved cos and sign-folded sin, built once by the caller:
    #   cos[j] = cos_full[j];  sin[j] = sin_full[j] * sign[j], sign = [-1,+1,...]
    cos: pl.Tensor[[B * CMP_WRITES_PER_REQUEST, ROPE_HEAD_DIM], pl.FP32],
    sin: pl.Tensor[[B * CMP_WRITES_PER_REQUEST, ROPE_HEAD_DIM], pl.FP32],
    cmp_kv_cache: pl.Tensor,
    cmp_block_table: pl.Tensor,
    position_ids: pl.Tensor,
    kv_seq_lens: pl.Tensor,
    state_page_stride: pl.Scalar[pl.INDEX],
    late_dep: pl.Scalar[pl.TASK_ID],
):
    b_dim = pl.tensor.dim(compress_state_block_table, 0)
    bs = pl.tensor.dim(x, 0)
    s_dim = bs // b_dim
    t_matmul = ((bs + MM_B_TILE - 1) // MM_B_TILE) * MM_B_TILE  # ceil to whole 16-row cube tiles
    pool_rows = b_dim * CMP_WRITES_PER_REQUEST
    rms_blocks = (pool_rows + RMS_PAD_TILE - 1) // RMS_PAD_TILE
    x_flat = x
    cmp4_kv_proj_pad = pl.create_tensor([BS_PAD, OUT_DIM], dtype=pl.FP32)
    cmp4_score_proj_pad = pl.create_tensor([BS_PAD, OUT_DIM], dtype=pl.FP32)
    cmp_block_num = pl.tensor.dim(cmp_kv_cache, 0)
    kv_flat = kv
    cmp_kv_cache_flat = pl.reshape(cmp_kv_cache, [cmp_block_num * BLOCK_SIZE, HEAD_DIM])

    # Deferred behind the caller's rms_norm dummy barrier: qkv's qr_proj_matmul is the
    # critical path and must win the cores when rms_norm retires.
    with pl.spmd(
        t_matmul * OUT_DIM // (MM_B_TILE * OUT_TILE), name_hint="kv_score_proj", deps=[late_dep]
    ) as _kv_score_tid:
        idx = pl.tile.get_block_idx()
        global_row0 = (idx // (OUT_DIM // OUT_TILE)) * MM_B_TILE
        o0 = (idx % (OUT_DIM // OUT_TILE)) * OUT_TILE
        kv_acc = pl.create_tensor([MM_B_TILE, OUT_TILE], dtype=pl.FP32)
        score_acc = pl.create_tensor([MM_B_TILE, OUT_TILE], dtype=pl.FP32)
        for kb in pl.pipeline(0, D // K_TILE, stage=2):
            k0 = kb * K_TILE
            x_rows = pl.min(MM_B_TILE, bs - global_row0)
            x_tile = pl.slice(x_flat, [MM_B_TILE, K_TILE], [global_row0, k0], valid_shape=[x_rows, K_TILE])
            # Weights stored transposed [OUT_DIM, D] and consumed via b_trans=True so the
            # GM->L1 load is a DN2ZN (each [OUT_TILE, K_TILE] row is K-contiguous = long
            # bursts) instead of ND2NZ on [K_TILE, OUT_TILE] (K strided = many short
            # bursts). Cuts the transaction-bound MTE2 cost ~14% busy / ~7% compressor wall.
            wkv_tile = wkv[o0 : o0 + OUT_TILE, k0 : k0 + K_TILE]
            wgate_tile = wgate[o0 : o0 + OUT_TILE, k0 : k0 + K_TILE]
            if k0 == 0:
                kv_acc = pl.matmul(x_tile, wkv_tile, out_dtype=pl.FP32, b_trans=True)
                score_acc = pl.matmul(x_tile, wgate_tile, out_dtype=pl.FP32, b_trans=True)
            else:
                kv_acc = pl.matmul_acc(kv_acc, x_tile, wkv_tile, b_trans=True)
                score_acc = pl.matmul_acc(score_acc, x_tile, wgate_tile, b_trans=True)

        cmp4_kv_proj_pad[global_row0 : global_row0 + MM_B_TILE, o0 : o0 + OUT_TILE] = kv_acc
        cmp4_score_proj_pad[global_row0 : global_row0 + MM_B_TILE, o0 : o0 + OUT_TILE] = score_acc

    pooled_kv = _main_scatter_softmax_pool(
        compress_state,
        compress_state_block_table,
        cmp4_kv_proj_pad,
        cmp4_score_proj_pad,
        ape,
        position_ids,
        kv_seq_lens,
        state_page_stride,
    )
    normed_kv = pl.create_tensor([RMS_PAD_ROWS, HEAD_DIM], dtype=pl.FP32)
    norm_w_2d = pl.reshape(norm_w, [1, HEAD_DIM])
    for rms_blk in pl.spmd(rms_blocks, name_hint="rmsnorm_rope_cache_write", allow_early_resolve=True):
        # one 16-row block of B; rows rms_blk_rows..15 are pad on the tail block
        b0 = rms_blk * RMS_PAD_TILE
        rms_blk_rows = pl.min(RMS_PAD_TILE, pool_rows - b0)
        cos_b = pl.slice(cos, [RMS_PAD_TILE, ROPE_HEAD_DIM], [b0, 0], valid_shape=[rms_blk_rows, ROPE_HEAD_DIM])
        sin_b = pl.slice(sin, [RMS_PAD_TILE, ROPE_HEAD_DIM], [b0, 0], valid_shape=[rms_blk_rows, ROPE_HEAD_DIM])
        partial_sq = pl.full([1, RMS_PAD_TILE], dtype=pl.FP32, value=0.0)
        for k0 in pl.range(0, HEAD_DIM, HEAD_TILE):
            kv_rms_chunk = pooled_kv[b0 : b0 + RMS_PAD_TILE, k0 : k0 + HEAD_TILE]
            kv_rms_sq = pl.mul(kv_rms_chunk, kv_rms_chunk)
            kv_rms_rowsum = pl.reshape(pl.row_sum(kv_rms_sq), [1, RMS_PAD_TILE])
            partial_sq = pl.add(partial_sq, kv_rms_rowsum)

        variance = pl.reshape(pl.add(pl.mul(partial_sq, HEAD_DIM_INV), EPS), [RMS_PAD_TILE, 1])
        inv_rms = pl.recip(pl.sqrt(variance))
        for k0 in pl.range(0, NOPE_HEAD_DIM, HEAD_TILE):
            kv_norm_chunk = pooled_kv[b0 : b0 + RMS_PAD_TILE, k0 : k0 + HEAD_TILE]
            gamma = pl.cast(norm_w_2d[:, k0 : k0 + HEAD_TILE], pl.FP32)
            normed_chunk = pl.col_expand_mul(pl.row_expand_mul(kv_norm_chunk, inv_rms), gamma)
            normed_kv[b0 : b0 + RMS_PAD_TILE, k0 : k0 + HEAD_TILE] = normed_chunk

        kv_rope_norm = pooled_kv[b0 : b0 + RMS_PAD_TILE, NOPE_HEAD_DIM:HEAD_DIM]
        gamma_rope = pl.cast(norm_w_2d[:, NOPE_HEAD_DIM:HEAD_DIM], pl.FP32)
        # A3 interleaved swap-gather (same form as kv_rope_fused in qkv_proj_rope),
        # replacing the de-interleave gather + rotate + re-interleave scatter. gamma+inv_rms
        # are folded into rope_normed BEFORE the swap, so the swapped lane n[j^1] correctly
        # carries gamma[j^1]; inv_rms is per-row so it commutes. Only swap_idx (j^1) is built
        # in-kernel -- it permutes data, so no table can hold it; the interleaved cos and
        # sign-folded sin come in ready to use. normed_kv is FP32 -> write directly.
        #   out[j] = n[j]*cos_il[j] + n[j^1]*sin_il_signed[j]
        rope_normed = pl.col_expand_mul(pl.row_expand_mul(kv_rope_norm, inv_rms), gamma_rope)
        rope_ones = pl.full([RMS_PAD_TILE, ROPE_HEAD_DIM], dtype=pl.FP32, value=1.0)
        rope_col = pl.col_expand_mul(
            rope_ones, pl.cast(pl.arange(0, [1, ROPE_HEAD_DIM], dtype=pl.INT32), target_type=pl.FP32)
        )
        rope_dup_f = pl.cast(pl.cast(pl.mul(rope_col, 0.5), target_type=pl.INT32, mode="trunc"), target_type=pl.FP32)
        rope_lane = pl.sub(rope_col, pl.mul(rope_dup_f, 2.0))  # j%2
        rope_swap_idx = pl.cast(pl.sub(pl.add(rope_col, 1.0), pl.mul(rope_lane, 2.0)), target_type=pl.INT32)  # j^1
        swapped = pl.gather(rope_normed, dim=-1, index=rope_swap_idx)
        rope_rot = pl.add(pl.mul(rope_normed, cos_b), pl.mul(swapped, sin_b))
        normed_kv[b0 : b0 + RMS_PAD_TILE, NOPE_HEAD_DIM:HEAD_DIM] = rope_rot

        # cache write: reads back only this block's own normed_kv rows, so the normed_kv
        # RAW is intra-block -- no separate scope / cross-task barrier needed.
        for inner in pl.range(rms_blk_rows):
            pool_row = b0 + inner
            c_idx = pool_row // CMP_WRITES_PER_REQUEST
            if pl.read(kv_seq_lens, [c_idx]) > 0:
                write_index = pool_row - c_idx * CMP_WRITES_PER_REQUEST
                first_pos_b = pl.read(position_ids, [c_idx * s_dim])
                pos_b = first_pos_b % COMPRESS_RATIO
                boundary_s = COMPRESS_RATIO - 1 - pos_b + write_index * COMPRESS_RATIO
                kv_row_fp32 = normed_kv[pool_row : pool_row + 1, 0:HEAD_DIM]
                boundary_pos = first_pos_b + boundary_s
                compressed_pos = boundary_pos // COMPRESS_RATIO
                cache_block_i32 = pl.read(
                    cmp_block_table,
                    [c_idx, compressed_pos // BLOCK_SIZE],
                )
                if cache_block_i32 >= 0:
                    cache_block = pl.cast(cache_block_i32, pl.INDEX)
                    cache_row = cache_block * BLOCK_SIZE + pl.cast(compressed_pos % BLOCK_SIZE, pl.INDEX)
                    kv_flat[
                        c_idx * s_dim + boundary_s : c_idx * s_dim + boundary_s + 1,
                        :,
                    ] = kv_row_fp32
                    cmp_kv_cache_flat[cache_row : cache_row + 1, :] = pl.cast(
                        kv_row_fp32,
                        target_type=pl.BF16,
                        mode="rint",
                    )

    return kv
