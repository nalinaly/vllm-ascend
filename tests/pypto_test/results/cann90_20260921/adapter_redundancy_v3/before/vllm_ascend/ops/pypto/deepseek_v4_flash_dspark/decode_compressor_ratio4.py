# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""CSA integration dependency adapted from pypto-lib 205255b4/decode_compressor_ratio4.py."""

import pypto.language as pl

from .config import (
    BLOCK_SIZE,
    C4A_COMPRESSOR_BLOCK_SIZE,
    CSA_STATE_PHYSICAL_BLOCKS,
    DECODE_BATCH,
    DECODE_SEQ,
    FP32_NEG_INF,
    KV_CMP_BLOCK_NUM,
    TP,
)
from .config import (
    FLASH as M,
)

B_DYN = pl.dynamic("DECODE_CSA_C4_B_DYN")

S_DYN = pl.dynamic("DECODE_CSA_C4_S_DYN")

T_DYN = pl.dynamic("DECODE_CSA_C4_T_DYN")  # T = B * S

B = DECODE_BATCH // TP

S = DECODE_SEQ

EPS = M.rms_norm_eps

D = M.hidden_size

HEAD_DIM = M.head_dim

HEAD_DIM_INV = 1.0 / HEAD_DIM

ROPE_HEAD_DIM = M.qk_rope_head_dim

NOPE_HEAD_DIM = M.nope_head_dim

MAX_SEQ_LEN = M.max_position_embeddings

COMPRESS_RATIO = 4

OVERLAP = COMPRESS_RATIO == 4

COFF = 1 + int(OVERLAP)

OUT_DIM = COFF * HEAD_DIM

STATE_LEN = COFF * COMPRESS_RATIO

STATE_STORAGE_LEN = STATE_LEN + S

COMPRESS_STATE_BLOCK_SIZE = C4A_COMPRESSOR_BLOCK_SIZE

COMPRESS_STATE_MAX_BLOCKS = (STATE_STORAGE_LEN + COMPRESS_STATE_BLOCK_SIZE - 1) // COMPRESS_STATE_BLOCK_SIZE

COMPRESS_STATE_BLOCK_NUM = CSA_STATE_PHYSICAL_BLOCKS

COMPRESS_STATE_BLOCKS_PER_REQUEST = COMPRESS_STATE_BLOCK_NUM // DECODE_BATCH

COMPRESS_STATE_BLOCK_NUM_DYN = pl.dynamic("CSA_STATE_BLOCK_NUM_DYN")

COMPRESS_STATE_DIM = 2 * OUT_DIM

CMP_MAX_BLOCKS = (MAX_SEQ_LEN // COMPRESS_RATIO + BLOCK_SIZE - 1) // BLOCK_SIZE

CMP_BLOCK_NUM = KV_CMP_BLOCK_NUM

CMP_BLOCK_NUM_DYN = pl.dynamic("CMP_BLOCK_NUM_DYN")

K_TILE = 512

OUT_TILE = 64

MM_B_TILE = 64

KV_SCORE_WORKERS = 24  # KV-score projection workers

POOL_WORKERS = 48  # Pool workers

COMMIT_WORKERS = 48

GROUP_BS = DECODE_BATCH * DECODE_SEQ

BS_PAD = ((GROUP_BS + MM_B_TILE - 1) // MM_B_TILE) * MM_B_TILE

HEAD_TILE = 64

POOL_HEAD_TILE = HEAD_DIM

RMS_PAD_TILE = 16  # 16-row block of B (min M for FP32 vec ops)


@pl.jit.inline(auto_scope=False)
def compressor_ratio4_project(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    wkv: pl.Tensor[[OUT_DIM, D], pl.BF16],
    wgate: pl.Tensor[[OUT_DIM, D], pl.BF16],
    kv_proj_pad: pl.Tensor[[BS_PAD, OUT_DIM], pl.FP32],
    score_proj_pad: pl.Tensor[[BS_PAD, OUT_DIM], pl.FP32],
    late_dep: pl.Scalar[pl.TASK_ID],
):
    """Project token-local compressor values and scores in FP32."""
    bs = pl.tensor.dim(x, 0)
    t_matmul = ((bs + MM_B_TILE - 1) // MM_B_TILE) * MM_B_TILE
    x_flat = x

    cmp4_kv_proj_pad = kv_proj_pad
    cmp4_score_proj_pad = score_proj_pad

    # Caller-ordered KV and score projections.
    with pl.spmd(
        KV_SCORE_WORKERS,
        name_hint="kv_score_proj",
        deps=[late_dep],
    ) as _kv_score_tid:
        kv_worker = pl.tile.get_block_idx()
        for idx in pl.range(kv_worker, t_matmul * OUT_DIM // (MM_B_TILE * OUT_TILE), KV_SCORE_WORKERS):
            global_row0 = (idx // (OUT_DIM // OUT_TILE)) * MM_B_TILE
            o0 = (idx % (OUT_DIM // OUT_TILE)) * OUT_TILE
            x_rows = pl.min(MM_B_TILE, bs - global_row0)
            kv_acc = pl.create_tensor([MM_B_TILE, OUT_TILE], dtype=pl.FP32)
            score_acc = pl.create_tensor([MM_B_TILE, OUT_TILE], dtype=pl.FP32)
            for kb in pl.pipeline(0, D // K_TILE, stage=2):
                k0 = kb * K_TILE
                x_tile = pl.slice(x_flat, [MM_B_TILE, K_TILE], [global_row0, k0], valid_shape=[x_rows, K_TILE])
                # Transposed [OUT_DIM, D] projection weights.
                wkv_tile = wkv[o0 : o0 + OUT_TILE, k0 : k0 + K_TILE]
                wgate_tile = wgate[o0 : o0 + OUT_TILE, k0 : k0 + K_TILE]
                # This peel is NOT foldable into init_cond: x_tile narrows to a
                # runtime row count and MM_B_TILE spans four 16-row fractals, so
                # mad writes at pitch ceil(validRow/16)*16 while a create_tensor
                # accumulator is read back at 64. Only pl.matmul stamps the
                # accumulator compact, so dropping it fails AccCompactValid.
                if k0 == 0:
                    kv_acc = pl.matmul(x_tile, wkv_tile, out_dtype=pl.FP32, b_trans=True)
                    score_acc = pl.matmul(x_tile, wgate_tile, out_dtype=pl.FP32, b_trans=True)
                else:
                    kv_acc = pl.matmul_acc(kv_acc, x_tile, wkv_tile, b_trans=True)
                    score_acc = pl.matmul_acc(score_acc, x_tile, wgate_tile, b_trans=True)

            cmp4_kv_proj_pad[global_row0 : global_row0 + MM_B_TILE, o0 : o0 + OUT_TILE] = kv_acc
            cmp4_score_proj_pad[global_row0 : global_row0 + MM_B_TILE, o0 : o0 + OUT_TILE] = score_acc

    return _kv_score_tid


@pl.jit.inline(auto_scope=False)
def compressor_ratio4_pool_projected(
    compress_state: pl.Tensor[[COMPRESS_STATE_BLOCK_NUM_DYN, COMPRESS_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM], pl.FP32],
    ape: pl.Tensor[[COMPRESS_RATIO, OUT_DIM], pl.FP32],
    position_ids: pl.Tensor[[T_DYN], pl.INT64],
    pooled_kv: pl.Out[pl.Tensor[[BS_PAD, HEAD_DIM], pl.FP32]],
    kv_proj_pad: pl.Out[pl.Tensor[[BS_PAD, OUT_DIM], pl.FP32]],
    score_proj_pad: pl.Out[pl.Tensor[[BS_PAD, OUT_DIM], pl.FP32]],
    late_dep: pl.Scalar[pl.TASK_ID],
):
    """Pool full-stream projected values and scores against the state ring."""
    b_dim = pl.tensor.dim(compress_state, 0) // COMPRESS_STATE_MAX_BLOCKS
    bs = pl.tensor.dim(position_ids, 0)
    s_dim = bs // b_dim
    cmp4_kv_proj_pad = kv_proj_pad
    cmp4_score_proj_pad = score_proj_pad
    compress_state_block_num = pl.tensor.dim(compress_state, 0)
    compress_state_rows = compress_state_block_num * COMPRESS_STATE_BLOCK_SIZE
    compress_state_flat = pl.reshape(compress_state, [compress_state_rows, COMPRESS_STATE_DIM])

    _kv_score_tid = late_dep

    pool_workers = pl.min(b_dim, POOL_WORKERS)
    with pl.spmd(
        pool_workers,
        name_hint="scatter_softmax_pool",
        deps=[_kv_score_tid],
        allow_early_resolve=True,
    ) as pool_tid:
        pool_worker = pl.tile.get_block_idx()
        for c_idx in pl.range(pool_worker, b_dim, pool_workers):
            first_pos_b = pl.read(position_ids, [c_idx * s_dim])
            for s_idx in pl.range(s_dim):
                token = c_idx * s_dim + s_idx
                token_pos = pl.read(position_ids, [token])
                pooled_kv[token : token + 1, :] = pl.full([1, HEAD_DIM], dtype=pl.FP32, value=0.0)
                if (token_pos + 1) % COMPRESS_RATIO == 0:
                    window_start = token_pos - STATE_LEN + 1
                    for h0 in pl.range(0, HEAD_DIM, POOL_HEAD_TILE):
                        last_ape_row = pl.cast(token_pos % COMPRESS_RATIO, target_type=pl.INDEX)
                        mi = pl.add(
                            cmp4_score_proj_pad[
                                token : token + 1,
                                HEAD_DIM + h0 : HEAD_DIM + h0 + POOL_HEAD_TILE,
                            ],
                            ape[
                                last_ape_row : last_ape_row + 1,
                                HEAD_DIM + h0 : HEAD_DIM + h0 + POOL_HEAD_TILE,
                            ],
                        )
                        li = pl.exp(pl.sub(mi, mi))
                        oi = cmp4_kv_proj_pad[
                            token : token + 1,
                            HEAD_DIM + h0 : HEAD_DIM + h0 + POOL_HEAD_TILE,
                        ]
                        for state_idx in pl.range(STATE_LEN - 1):
                            logical_pos = window_start + state_idx
                            value = pl.full([1, POOL_HEAD_TILE], dtype=pl.FP32, value=0.0)
                            score = pl.full([1, POOL_HEAD_TILE], dtype=pl.FP32, value=FP32_NEG_INF)
                            state_half = 0
                            if state_idx >= COMPRESS_RATIO:
                                state_half = HEAD_DIM
                            if logical_pos >= 0 and logical_pos < first_pos_b:
                                ring_row = logical_pos % STATE_STORAGE_LEN
                                state_row = c_idx * STATE_STORAGE_LEN + ring_row
                                value = compress_state_flat[
                                    state_row : state_row + 1,
                                    state_half + h0 : state_half + h0 + POOL_HEAD_TILE,
                                ]
                                score = compress_state_flat[
                                    state_row : state_row + 1,
                                    OUT_DIM + state_half + h0 : OUT_DIM + state_half + h0 + POOL_HEAD_TILE,
                                ]
                            if logical_pos >= first_pos_b:
                                if logical_pos <= token_pos:
                                    overlay_token = c_idx * s_dim + logical_pos - first_pos_b
                                    ape_row = pl.cast(logical_pos % COMPRESS_RATIO, target_type=pl.INDEX)
                                    value = cmp4_kv_proj_pad[
                                        overlay_token : overlay_token + 1,
                                        state_half + h0 : state_half + h0 + POOL_HEAD_TILE,
                                    ]
                                    score = pl.add(
                                        cmp4_score_proj_pad[
                                            overlay_token : overlay_token + 1,
                                            state_half + h0 : state_half + h0 + POOL_HEAD_TILE,
                                        ],
                                        ape[ape_row : ape_row + 1, state_half + h0 : state_half + h0 + POOL_HEAD_TILE],
                                    )
                            mi_next = pl.maximum(mi, score)
                            alpha = pl.exp(pl.sub(mi, mi_next))
                            beta = pl.exp(pl.sub(score, mi_next))
                            li = pl.add(pl.mul(alpha, li), beta)
                            oi = pl.add(pl.mul(oi, alpha), pl.mul(value, beta))
                            mi = mi_next
                        pooled_kv[token : token + 1, h0 : h0 + POOL_HEAD_TILE] = pl.div(oi, li)

    return pool_tid, _kv_score_tid


@pl.jit.inline(auto_scope=False)
def compressor_ratio4_pool(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    compress_state: pl.Tensor[[COMPRESS_STATE_BLOCK_NUM_DYN, COMPRESS_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM], pl.FP32],
    wkv: pl.Tensor[[OUT_DIM, D], pl.BF16],
    wgate: pl.Tensor[[OUT_DIM, D], pl.BF16],
    ape: pl.Tensor[[COMPRESS_RATIO, OUT_DIM], pl.FP32],
    position_ids: pl.Tensor[[T_DYN], pl.INT64],
    pooled_kv: pl.Out[pl.Tensor[[BS_PAD, HEAD_DIM], pl.FP32]],
    kv_proj_pad: pl.Out[pl.Tensor[[BS_PAD, OUT_DIM], pl.FP32]],
    score_proj_pad: pl.Out[pl.Tensor[[BS_PAD, OUT_DIM], pl.FP32]],
    late_dep: pl.Scalar[pl.TASK_ID],
):
    """Project, then pool and commit the token stream."""
    projection_tid = compressor_ratio4_project(
        x,
        wkv,
        wgate,
        kv_proj_pad,
        score_proj_pad,
        late_dep,
    )
    pool_tid, projection_ready_tid = compressor_ratio4_pool_projected(
        compress_state,
        ape,
        position_ids,
        pooled_kv,
        kv_proj_pad,
        score_proj_pad,
        projection_tid,
    )
    return pool_tid, projection_ready_tid


@pl.jit.inline(auto_scope=False)
def compressor_ratio4_cache_write(
    kv: pl.Tensor[[T_DYN, HEAD_DIM], pl.FP32],
    pooled_kv: pl.Tensor[[BS_PAD, HEAD_DIM], pl.FP32],
    norm_w: pl.Tensor[[HEAD_DIM], pl.FP32],
    cos: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.FP32],
    sin: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.FP32],
    cmp_kv_cache: pl.Tensor[[CMP_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    cmp_slot_mapping: pl.Tensor[[T_DYN], pl.INT64],
    compress_state: pl.Tensor[[COMPRESS_STATE_BLOCK_NUM_DYN, COMPRESS_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM], pl.FP32],
    ape: pl.Tensor[[COMPRESS_RATIO, OUT_DIM], pl.FP32],
    kv_proj_pad: pl.Tensor[[BS_PAD, OUT_DIM], pl.FP32],
    score_proj_pad: pl.Tensor[[BS_PAD, OUT_DIM], pl.FP32],
    position_ids: pl.Tensor[[T_DYN], pl.INT64],
    state_slot_mapping: pl.Tensor[[T_DYN, 2], pl.INT32],
    pool_tid: pl.Scalar[pl.TASK_ID],
    late_write_dep: pl.Scalar[pl.TASK_ID],
):
    """State commit, RMSNorm + RoPE over the pooled rows, and the compressed KV cache write."""
    b_dim = pl.tensor.dim(compress_state, 0) // COMPRESS_STATE_MAX_BLOCKS
    bs = pl.tensor.dim(position_ids, 0)
    s_dim = bs // b_dim
    rms_blocks = (bs + RMS_PAD_TILE - 1) // RMS_PAD_TILE
    cmp_block_num = pl.tensor.dim(cmp_kv_cache, 0)
    kv_flat = kv
    cmp_kv_cache_flat = pl.reshape(cmp_kv_cache, [cmp_block_num * BLOCK_SIZE, HEAD_DIM])
    compress_state_block_num = pl.tensor.dim(compress_state, 0)
    compress_state_rows = compress_state_block_num * COMPRESS_STATE_BLOCK_SIZE
    compress_state_flat = pl.reshape(compress_state, [compress_state_rows, COMPRESS_STATE_DIM])
    cmp4_kv_proj_pad = kv_proj_pad
    cmp4_score_proj_pad = score_proj_pad

    # Recurrent state-ring commit.
    commit_workers = pl.min(b_dim, COMMIT_WORKERS)
    with pl.spmd(commit_workers, name_hint="compress_state_commit", deps=[pool_tid, late_write_dep]):
        commit_worker = pl.tile.get_block_idx()
        for c_idx in pl.range(commit_worker, b_dim, commit_workers):
            for s_idx in pl.range(s_dim):
                token = c_idx * s_dim + s_idx
                state_page = pl.read(state_slot_mapping, [token, 0])
                state_offset = pl.read(state_slot_mapping, [token, 1])
                if state_page >= 0 and state_offset >= 0:
                    token_pos = pl.read(position_ids, [token])
                    state_row = c_idx * STATE_STORAGE_LEN + pl.cast(token_pos % STATE_STORAGE_LEN, pl.INDEX)
                    ape_row = pl.cast(token_pos % COMPRESS_RATIO, target_type=pl.INDEX)
                    compress_state_flat[state_row : state_row + 1, 0:OUT_DIM] = cmp4_kv_proj_pad[
                        token : token + 1, 0:OUT_DIM
                    ]
                    compress_state_flat[state_row : state_row + 1, OUT_DIM:COMPRESS_STATE_DIM] = pl.add(
                        cmp4_score_proj_pad[token : token + 1, 0:OUT_DIM], ape[ape_row : ape_row + 1, 0:OUT_DIM]
                    )

    normed_kv = pl.create_tensor([BS_PAD, HEAD_DIM], dtype=pl.FP32)
    norm_w_2d = pl.reshape(norm_w, [1, HEAD_DIM])
    with pl.spmd(
        rms_blocks,
        name_hint="rmsnorm_rope_cache_write",
        deps=[pool_tid, late_write_dep],
        allow_early_resolve=True,
    ) as cache_write_tid:
        rms_blk = pl.tile.get_block_idx()
        b0 = rms_blk * RMS_PAD_TILE
        rms_blk_rows = pl.min(RMS_PAD_TILE, bs - b0)
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
            gamma = norm_w_2d[:, k0 : k0 + HEAD_TILE]
            normed_chunk = pl.col_expand_mul(pl.row_expand_mul(kv_norm_chunk, inv_rms), gamma)
            normed_kv[b0 : b0 + RMS_PAD_TILE, k0 : k0 + HEAD_TILE] = normed_chunk

        kv_rope_norm = pooled_kv[b0 : b0 + RMS_PAD_TILE, NOPE_HEAD_DIM:HEAD_DIM]
        gamma_rope = norm_w_2d[:, NOPE_HEAD_DIM:HEAD_DIM]
        # Interleaved RMSNorm and inverse-RoPE rotation.
        rope_normed = pl.col_expand_mul(pl.row_expand_mul(kv_rope_norm, inv_rms), gamma_rope)
        rope_ones = pl.full([RMS_PAD_TILE, ROPE_HEAD_DIM], dtype=pl.FP32, value=1.0)
        rope_index = pl.arange(0, [1, ROPE_HEAD_DIM], dtype=pl.INT32)
        rope_index_f = pl.cast(rope_index, target_type=pl.FP32)
        rope_col = pl.col_expand_mul(rope_ones, rope_index_f)
        rope_dup_f = pl.cast(pl.cast(pl.mul(rope_col, 0.5), target_type=pl.INT32, mode="trunc"), target_type=pl.FP32)
        rope_lane = pl.sub(rope_col, pl.mul(rope_dup_f, 2.0))  # j%2
        rope_swap_idx = pl.cast(pl.sub(pl.add(rope_col, 1.0), pl.mul(rope_lane, 2.0)), target_type=pl.INT32)  # j^1
        swapped = pl.gather(rope_normed, dim=-1, index=rope_swap_idx)
        rope_rot = pl.add(pl.mul(rope_normed, cos_b), pl.mul(swapped, sin_b))
        normed_kv[b0 : b0 + RMS_PAD_TILE, NOPE_HEAD_DIM:HEAD_DIM] = rope_rot

        for inner in pl.range(rms_blk_rows):
            token = b0 + inner
            cache_row_i64 = pl.read(cmp_slot_mapping, [token])
            if cache_row_i64 >= 0:
                cache_row = pl.cast(cache_row_i64, pl.INDEX)
                kv_row_fp32 = normed_kv[token : token + 1, 0:HEAD_DIM]
                kv_flat[token : token + 1, :] = kv_row_fp32
                cmp_kv_cache_flat[cache_row : cache_row + 1, :] = pl.cast(kv_row_fp32, target_type=pl.BF16, mode="rint")

    return cache_write_tid


@pl.jit.inline
def compressor_ratio4(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    kv: pl.Tensor[[T_DYN, HEAD_DIM], pl.FP32],
    compress_state: pl.Tensor[[COMPRESS_STATE_BLOCK_NUM_DYN, COMPRESS_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM], pl.FP32],
    wkv: pl.Tensor[[OUT_DIM, D], pl.BF16],
    wgate: pl.Tensor[[OUT_DIM, D], pl.BF16],
    ape: pl.Tensor[[COMPRESS_RATIO, OUT_DIM], pl.FP32],
    norm_w: pl.Tensor[[HEAD_DIM], pl.FP32],
    cos: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.FP32],
    sin: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.FP32],
    cmp_kv_cache: pl.Tensor[[CMP_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    position_ids: pl.Tensor[[T_DYN], pl.INT64],
    cmp_slot_mapping: pl.Tensor[[T_DYN], pl.INT64],
    state_slot_mapping: pl.Tensor[[T_DYN, 2], pl.INT32],
    late_dep: pl.Scalar[pl.TASK_ID],
) -> tuple[pl.Tensor, pl.Scalar[pl.TASK_ID], pl.Scalar[pl.TASK_ID]]:
    pooled_kv = pl.create_tensor([BS_PAD, HEAD_DIM], dtype=pl.FP32)
    kv_proj_pad = pl.create_tensor([BS_PAD, OUT_DIM], dtype=pl.FP32)
    score_proj_pad = pl.create_tensor([BS_PAD, OUT_DIM], dtype=pl.FP32)
    pool_tid, kv_score_tid = compressor_ratio4_pool(
        x,
        compress_state,
        wkv,
        wgate,
        ape,
        position_ids,
        pooled_kv,
        kv_proj_pad,
        score_proj_pad,
        late_dep,
    )
    cache_write_tid = compressor_ratio4_cache_write(
        kv,
        pooled_kv,
        norm_w,
        cos,
        sin,
        cmp_kv_cache,
        cmp_slot_mapping,
        compress_state,
        ape,
        kv_proj_pad,
        score_proj_pad,
        position_ids,
        state_slot_mapping,
        pool_tid,
        pool_tid,
    )
    return kv, cache_write_tid, kv_score_tid
