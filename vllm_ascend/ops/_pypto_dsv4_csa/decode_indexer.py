# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 Indexer (decode). Mirrors model.py Indexer (line 380-433);
golden is a port of forward's decode branch (prefill `start_pos == 0` path is omitted).
The inner Compressor is invoked via golden_compressor (placeholder)."""

import pypto.language as pl

from .config import (
    BLOCK_SIZE,
    C4A_COMPRESSOR_BLOCK_SIZE,
    CSA_INNER_STATE_PHYSICAL_BLOCKS,
    DECODE_BATCH,
    DECODE_SEQ,
    FP32_NEG_INF,
    INT8_AMAX_EPS,
    INT8_SCALE_MAX,
    TP,
)
from .config import (
    FLASH as M,
)
from .decode_indexer_compressor import indexer_compressor

# model config
B = DECODE_BATCH // TP
S = DECODE_SEQ
T = B * S
D = M.hidden_size
Q_LORA = M.q_lora_rank
ROPE_HEAD_DIM = M.qk_rope_head_dim
IDX_N_HEADS = M.index_n_heads
IDX_HEAD_DIM = M.index_head_dim
IDX_NOPE_HEAD_DIM = M.index_nope_head_dim
WEIGHTS_SCALE = M.index_weights_scale
MAX_SEQ_LEN = M.max_position_embeddings
OFFSET = M.sliding_window

# kernel-local
COMPRESS_RATIO = 4  # the indexer only runs on ratio-4 layers
CMP_WRITES_PER_REQUEST = S // COMPRESS_RATIO
assert CMP_WRITES_PER_REQUEST == 2
IDX_TOPK = M.index_topk
INNER_OVERLAP = COMPRESS_RATIO == 4
INNER_COFF = 1 + int(INNER_OVERLAP)
INNER_HEAD_DIM = IDX_HEAD_DIM
INNER_OUT_DIM = INNER_COFF * INNER_HEAD_DIM
INNER_STATE_BLOCK_SIZE = C4A_COMPRESSOR_BLOCK_SIZE
INNER_STATE_PHYSICAL_BLOCKS = CSA_INNER_STATE_PHYSICAL_BLOCKS
INNER_STATE_MAX_BLOCKS = (MAX_SEQ_LEN + INNER_STATE_BLOCK_SIZE - 1) // INNER_STATE_BLOCK_SIZE
INNER_STATE_BLOCK_NUM = INNER_STATE_PHYSICAL_BLOCKS
INNER_STATE_DIM = 2 * INNER_OUT_DIM

IDX_KV_LEN = MAX_SEQ_LEN // COMPRESS_RATIO
SCORE_LEN = IDX_KV_LEN

# tiling
CACHE_TILE = min(64, BLOCK_SIZE)
assert BLOCK_SIZE % CACHE_TILE == 0, "CACHE_TILE must not cross a paged idx_kv_cache block"
# REDUCE_TILE tiles the paged C8 cache one fused matmul+reduce step at a time.
# A tile is one contiguous cache slice, so it caps at the page size.
REDUCE_TILE = min(128, BLOCK_SIZE)
assert BLOCK_SIZE % REDUCE_TILE == 0, "REDUCE_TILE must not cross a paged idx_kv_cache block"
# Page-interleaved lanes overlap the paged C8 matmul/reduce work without changing
# disjoint output ownership.  The lane count is tuned together with the token
# group below; every lane owns disjoint cache pages.
REDUCE_NSPLIT = 5
# Tokens from one request read the same paged index K cache.  Pack consecutive
# token queries into the matmul N axis so one page load feeds the whole group and
# the C8 cube sees a wider N dimension than separate N=64 operations.
SCORE_TOKEN_GROUP = 8
assert S % SCORE_TOKEN_GROUP == 0
SCORE_AIV_LANES = 2
SCORE_LANE_CACHE = REDUCE_TILE // SCORE_AIV_LANES
SCORE_LANE_TOKENS = SCORE_TOKEN_GROUP
SCORE_LANE_WIDTH = SCORE_LANE_TOKENS * IDX_N_HEADS
SCORE_REORDER_SIZE = SCORE_LANE_CACHE * SCORE_LANE_TOKENS
SCORE_REDUCE_CHUNK_ROWS = 64
SCORE_REDUCE_CHUNKS = SCORE_REORDER_SIZE // SCORE_REDUCE_CHUNK_ROWS
assert REDUCE_TILE % SCORE_AIV_LANES == 0
assert SCORE_REORDER_SIZE % SCORE_REDUCE_CHUNK_ROWS == 0
assert SCORE_REDUCE_CHUNKS == 2
Q_TILE = 256
# Q_OUT_TILE is the per-task N granularity (sets idx_qr_proj task count); MM_N_TILE
# is the Mat-safe cube N-tile. Q_OUT_TILE fans Q_OUT_TILE // MM_N_TILE cube ops per
# task so task count halves without growing the [Q_TILE, MM_N_TILE] L1 wq load.
Q_OUT_TILE = 1024
T_PAD = ((T + 16 - 1) // 16) * 16  # static upper bound on the token axis
# Matmul M at the 16-row cube floor: a tile taller than the dynamic source is not expressible.
MM_ROW_TILE = 16
# INT32 Acc is MM_ROW_TILE * MM_N_TILE * 4B and must stay under the 128KiB L0C wall.
MM_N_TILE = min(512, (128 * 1024) // (MM_ROW_TILE * 4))
QR_OUTPUT_TILE_COUNT = IDX_N_HEADS * IDX_HEAD_DIM // Q_OUT_TILE
assert Q_OUT_TILE % MM_N_TILE == 0
HEAD_DIM_TILE = 32
D_TILE = 512
# weights_proj splits K, not N: a [D_TILE, IDX_N_HEADS] row block reads contiguous GM,
# while an N slice would take 32B out of every 128B row. Each task writes its own
# partial row block, summed by a separate reduce scope. Partials are laid out
# [K slice][T_PAD rows] so the reduce adds whole T_PAD-row blocks.
# WEIGHTS_K_SLICE // D_TILE == 2, so the inner loop is a pl.range: a degenerate
# 2-iteration pl.pipeline(stage=2) miscompiles over matmul.
WEIGHTS_OK = 4
WEIGHTS_K_SLICE = D // WEIGHTS_OK
assert WEIGHTS_K_SLICE % D_TILE == 0
QH_QUANT_TILE = 64
QH_MM_TILE = 64
QH_HEAD_DIM_TILE = 64
assert QH_MM_TILE == QH_QUANT_TILE
assert IDX_HEAD_DIM % QH_HEAD_DIM_TILE == 0
assert (S * IDX_N_HEADS) % QH_QUANT_TILE == 0
HADAMARD_SCALE = IDX_HEAD_DIM**-0.5
# qr_rope SPMD tile == row block: one ROPE_ROW_TILE-row block per SPMD tile.
ROPE_ROW_TILE = 32
# Fuse dequant and RoPE in two bounded vector tiles per block.  A 16-head
# [1, 2048] flat tile stays comfortably inside A2/A3 Vec memory while the
# outer 32-head block keeps the former qr_rope blockDim.
QR_FUSED_HEAD_TILE = 16
QR_FUSED_WIDTH = QR_FUSED_HEAD_TILE * IDX_HEAD_DIM
assert ROPE_ROW_TILE % QR_FUSED_HEAD_TILE == 0
assert QR_FUSED_WIDTH == 2 * Q_OUT_TILE
TOPK_HALF_LEN = SCORE_LEN // 2
TOPK_SHORT_TAIL = 32
TOPK_SHORT_LIMIT = TOPK_HALF_LEN + TOPK_SHORT_TAIL
TOPK_HALF_PAIR_OFFSET = 2 * TOPK_HALF_LEN
TOPK_PAIR_WIDTH = 2 * IDX_TOPK
assert SCORE_LEN == 2 * TOPK_HALF_LEN, "decode indexer topk expects an even score length"
assert TOPK_HALF_LEN == 2048, "decode indexer 4096-value topk uses two 2048-value halves"
assert TOPK_HALF_LEN % 32 == 0
assert TOPK_SHORT_TAIL == 32
assert IDX_TOPK <= TOPK_HALF_LEN, "per-half candidate list must cover the final topk width"


@pl.jit.inline
def indexer(
    x: pl.Tensor,
    qr: pl.Tensor,
    qr_scale: pl.Tensor,
    wq_b: pl.Tensor[[Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32],
    weights_proj: pl.Tensor[[D, IDX_N_HEADS], pl.BF16],
    # Native full-width interleaved cos and sign-folded sin, built once by the caller:
    #   cos[j] = cos_full[j];  sin[j] = sin_full[j] * sign[j], sign = [-1,+1,...]
    token_cos: pl.Tensor[[T, ROPE_HEAD_DIM], pl.FP32],
    token_sin: pl.Tensor[[T, ROPE_HEAD_DIM], pl.FP32],
    compress_cos: pl.Tensor[[B * CMP_WRITES_PER_REQUEST, ROPE_HEAD_DIM], pl.FP32],
    compress_sin: pl.Tensor[[B * CMP_WRITES_PER_REQUEST, ROPE_HEAD_DIM], pl.FP32],
    hadamard: pl.Tensor[[IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16],  # shared by q rotation and inner Compressor
    # The caller shares qkv_proj_rope's [tokens, 64] table.  ``tokens`` varies
    # across B4/B8/B12/B16 and the current specializer cannot infer metadata
    # when a sliced Tensor is forwarded to an inline parameter, so accept the
    # owning Tensor here and take the exact static 32-row view at the use site.
    rope_swap_idx_t: pl.Tensor,
    inner_kv: pl.Tensor,
    inner_compress_state: pl.Tensor,
    inner_compress_state_block_table: pl.Tensor,
    inner_wkv: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16],
    inner_wgate: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16],
    inner_ape: pl.Tensor[[COMPRESS_RATIO, INNER_OUT_DIM], pl.FP32],
    inner_norm_w: pl.Tensor[[INNER_HEAD_DIM], pl.BF16],
    # Canonical physical-storage aliases.  The explicit page strides below
    # preserve A3's interleaved K/scale page without a contiguous mirror.
    idx_kv_cache: pl.InOut[pl.Tensor],
    idx_kv_scale: pl.InOut[pl.Tensor],
    idx_block_table: pl.Tensor,
    score: pl.Tensor,
    topk_idxs: pl.Tensor,
    position_ids: pl.Tensor,
    kv_seq_lens: pl.Tensor,
    offset: pl.Scalar[pl.INT32],
    inner_state_page_stride: pl.Scalar[pl.INDEX],
    indexer_k_page_stride: pl.Scalar[pl.INDEX],
    indexer_scale_page_stride: pl.Scalar[pl.INDEX],
    need_index_score: pl.Tensor,
    index_score_gate_dep: pl.Scalar[pl.TASK_ID],
    late_dep: pl.Scalar[pl.TASK_ID],
):
    b_dim = pl.tensor.dim(idx_block_table, 0)
    t_dim = pl.tensor.dim(x, 0)
    t_heads = t_dim * IDX_N_HEADS
    idx_table_width = pl.tensor.dim(idx_block_table, 1)
    idx_table_len = b_dim * idx_table_width
    row_blocks = (t_dim + MM_ROW_TILE - 1) // MM_ROW_TILE
    score_units = (t_dim // SCORE_TOKEN_GROUP) * REDUCE_NSPLIT
    qr_acc_pad = pl.create_tensor([T_PAD, IDX_N_HEADS * IDX_HEAD_DIM], dtype=pl.INT32)
    with pl.spmd(
        QR_OUTPUT_TILE_COUNT * row_blocks,
        name_hint="idx_qr_proj_matmul",
        deps=[index_score_gate_dep],
        predicate=(need_index_score[0] > 0),
        allow_early_resolve=True,
    ) as _qr_proj_tid:
        qr_unit = pl.tile.get_block_idx()
        qr_rb = qr_unit // QR_OUTPUT_TILE_COUNT  # row block outermost
        output_tile = qr_unit - qr_rb * QR_OUTPUT_TILE_COUNT
        qr_r0 = qr_rb * MM_ROW_TILE
        qr_rows = pl.min(MM_ROW_TILE, t_dim - qr_r0)
        o_base = output_tile * Q_OUT_TILE
        for ns in pl.range(0, Q_OUT_TILE, MM_N_TILE):
            qr_acc = pl.create_tensor([MM_ROW_TILE, MM_N_TILE], dtype=pl.INT32)
            for kb in pl.pipeline(0, Q_LORA // Q_TILE, stage=2):
                q0 = kb * Q_TILE
                qr_tile = pl.slice(qr, [MM_ROW_TILE, Q_TILE], [qr_r0, q0], valid_shape=[qr_rows, Q_TILE])
                wq_tile = wq_b[q0 : q0 + Q_TILE, o_base + ns : o_base + ns + MM_N_TILE]
                if q0 == 0:
                    qr_acc = pl.matmul(qr_tile, wq_tile, out_dtype=pl.INT32)
                else:
                    qr_acc = pl.matmul_acc(qr_acc, qr_tile, wq_tile)
            qr_acc_pad[qr_r0 : qr_r0 + MM_ROW_TILE, o_base + ns : o_base + ns + MM_N_TILE] = qr_acc
    # BF16 q for the Hadamard matmul: nope half rounded from the FP32 dequant, rope
    # half rounded to BF16, rotated in FP32, then rounded again.  Dequant and
    # RoPE share one child so the former full [T, H*D] FP32 GM handoff and its
    # intervening task boundary disappear.
    qr_bf16 = pl.create_tensor([t_heads, IDX_HEAD_DIM], dtype=pl.BF16)
    # SPMD still owns one 32-head block.  Each block processes two 16-head
    # smaller tiles sequentially so the compiler can reuse bounded Vec storage.
    # Each token owns two outer blocks and selects its own RoPE row; cos/sin
    # arrive interleave-duplicated and sign-folded from the caller.
    #   out[j] = x[j]*cos_il[j] + x[j^1]*sin_il_signed[j]
    with pl.spmd(
        t_heads // ROPE_ROW_TILE,
        name_hint="idx_qr_dequant_rope",
        deps=[index_score_gate_dep],
        predicate=(need_index_score[0] > 0),
        allow_early_resolve=True,
    ) as _qr_dequant_rope_tid:
        idx = pl.tile.get_block_idx()
        o0 = idx * ROPE_ROW_TILE
        token_idx = o0 // IDX_N_HEADS
        cos_row = token_cos[token_idx : token_idx + 1, 0:ROPE_HEAD_DIM]
        sin_row = token_sin[token_idx : token_idx + 1, 0:ROPE_HEAD_DIM]
        rope_swap_idx = rope_swap_idx_t[0:QR_FUSED_HEAD_TILE, 0:ROPE_HEAD_DIM]
        qr_scale_value = pl.read(qr_scale, [token_idx, 0])
        head_in_token0 = o0 - token_idx * IDX_N_HEADS
        for fused_half in pl.range(ROPE_ROW_TILE // QR_FUSED_HEAD_TILE):
            head_row0 = o0 + fused_half * QR_FUSED_HEAD_TILE
            output_base = (head_in_token0 + fused_half * QR_FUSED_HEAD_TILE) * IDX_HEAD_DIM
            acc_fp32 = pl.cast(
                qr_acc_pad[token_idx : token_idx + 1, output_base : output_base + QR_FUSED_WIDTH],
                target_type=pl.FP32,
                mode="none",
            )
            wq_scale = pl.reshape(
                wq_b_scale[output_base : output_base + QR_FUSED_WIDTH],
                [1, QR_FUSED_WIDTH],
            )
            qr_dequant_flat = pl.col_expand_mul(
                pl.mul(acc_fp32, qr_scale_value),
                wq_scale,
            )
            qr_dequant = pl.reshape(qr_dequant_flat, [QR_FUSED_HEAD_TILE, IDX_HEAD_DIM])
            qr_nope_slice = qr_dequant[:, 0:IDX_NOPE_HEAD_DIM]
            qr_rope_bf16 = pl.cast(
                qr_dequant[:, IDX_NOPE_HEAD_DIM:IDX_HEAD_DIM],
                target_type=pl.BF16,
                mode="rint",
            )
            qr_rope_slice = pl.cast(qr_rope_bf16, target_type=pl.FP32)
            qr_swapped = pl.gather(qr_rope_slice, dim=-1, index=rope_swap_idx)
            rope_rot = pl.add(
                pl.col_expand_mul(qr_rope_slice, cos_row),
                pl.col_expand_mul(qr_swapped, sin_row),
            )
            qr_vec = pl.concat(
                pl.cast(qr_nope_slice, target_type=pl.BF16, mode="rint"),
                pl.cast(rope_rot, target_type=pl.BF16, mode="rint"),
            )
            qr_bf16[head_row0 : head_row0 + QR_FUSED_HEAD_TILE, :] = qr_vec

    qr_hadamard_i8 = pl.create_tensor([t_heads, IDX_HEAD_DIM], dtype=pl.INT8)
    qr_hadamard_scale_dq = pl.create_tensor([t_heads, 1], dtype=pl.FP32)
    with pl.spmd(
        t_heads // QH_QUANT_TILE,
        name_hint="qr_hadamard_quant_mixed",
        deps=[index_score_gate_dep],
        predicate=(need_index_score[0] > 0),
        allow_early_resolve=True,
        optimizations=[pl.cross_core_slot(slot_num=1)],
    ) as _qh_quant_tid:
        idx = pl.tile.get_block_idx()
        o0 = idx * QH_QUANT_TILE
        qh_acc = pl.matmul(qr_bf16[o0 : o0 + QH_MM_TILE, :], hadamard, out_dtype=pl.FP32)
        # Preserve native rounding exactly, but materialize it once for the
        # whole 128-wide row instead of reloading and recomputing two halves.
        qh_linear_bf16 = pl.cast(qh_acc, target_type=pl.BF16, mode="rint")
        qh_scaled_native = pl.mul(
            pl.cast(qh_linear_bf16, target_type=pl.FP32),
            HADAMARD_SCALE,
        )
        qh_bf16 = pl.cast(qh_scaled_native, target_type=pl.BF16, mode="rint")
        qh_f32 = pl.cast(qh_bf16, target_type=pl.FP32)
        # A2/A3 row_max lowers a 64-column vector tile to one scalar per row.
        # Reduce the two 64-column halves and merge their maxima; reducing the
        # full 128 columns in one call retains a blocked 64-wide result layout.
        qh_amax = pl.full([1, QH_QUANT_TILE], dtype=pl.FP32, value=INT8_AMAX_EPS)
        for h0 in pl.range(0, IDX_HEAD_DIM, QH_HEAD_DIM_TILE):
            qh_a_f32 = qh_f32[:, h0 : h0 + QH_HEAD_DIM_TILE]
            qh_a_abs = pl.maximum(qh_a_f32, pl.neg(qh_a_f32))
            qh_a_max = pl.reshape(pl.row_max(qh_a_abs), [1, QH_QUANT_TILE])
            qh_amax = pl.maximum(qh_amax, qh_a_max)
        qh_scale_quant_row = pl.div(pl.full([1, QH_QUANT_TILE], dtype=pl.FP32, value=INT8_SCALE_MAX), qh_amax)
        qh_scale_dq = pl.reshape(pl.recip(qh_scale_quant_row), [QH_QUANT_TILE, 1])
        qr_hadamard_scale_dq[o0 : o0 + QH_QUANT_TILE, :] = qh_scale_dq
        qh_scale_quant = pl.reshape(qh_scale_quant_row, [QH_QUANT_TILE, 1])
        for h1 in pl.range(0, IDX_HEAD_DIM, QH_HEAD_DIM_TILE):
            qh_q_f32 = qh_f32[:, h1 : h1 + QH_HEAD_DIM_TILE]
            qh_q_scaled = pl.row_expand_mul(qh_q_f32, qh_scale_quant)
            qh_q_i32 = pl.cast(qh_q_scaled, target_type=pl.INT32, mode="rint")
            qh_q_half = pl.cast(qh_q_i32, target_type=pl.FP16, mode="round")
            qh_i8 = pl.cast(qh_q_half, target_type=pl.INT8, mode="trunc")
            qr_hadamard_i8[
                o0 : o0 + QH_QUANT_TILE,
                h1 : h1 + QH_HEAD_DIM_TILE,
            ] = qh_i8

    x_flat = x
    weights = pl.create_tensor([T_PAD, IDX_N_HEADS], dtype=pl.FP32)
    weights_partial = pl.create_tensor([WEIGHTS_OK * T_PAD, IDX_N_HEADS], dtype=pl.FP32)
    score_reorder_idx_t = pl.create_tensor([1, SCORE_REORDER_SIZE], dtype=pl.INT32)
    # Deferred behind the caller's rms_norm dummy barrier: qkv's qr_proj_matmul is the
    # critical path and must win the cores when rms_norm retires.
    with pl.spmd(
        WEIGHTS_OK * row_blocks,
        name_hint="weights_proj",
        deps=[late_dep, index_score_gate_dep],
        predicate=(need_index_score[0] > 0),
    ) as _weights_tid:
        w_unit = pl.tile.get_block_idx()
        w_rb = w_unit // WEIGHTS_OK  # row block outermost
        kb = w_unit - w_rb * WEIGHTS_OK
        w_r0 = w_rb * MM_ROW_TILE
        w_rows = pl.min(MM_ROW_TILE, t_dim - w_r0)
        k_base = kb * WEIGHTS_K_SLICE
        weights_acc = pl.create_tensor([MM_ROW_TILE, IDX_N_HEADS], dtype=pl.FP32)
        for db in pl.range(WEIGHTS_K_SLICE // D_TILE):
            d0 = k_base + db * D_TILE
            x_tile = pl.slice(x_flat, [MM_ROW_TILE, D_TILE], [w_r0, d0], valid_shape=[w_rows, D_TILE])
            weights_proj_tile = weights_proj[d0 : d0 + D_TILE, :]
            if db == 0:
                weights_acc = pl.matmul(x_tile, weights_proj_tile, out_dtype=pl.FP32)
            else:
                weights_acc = pl.matmul_acc(weights_acc, x_tile, weights_proj_tile)
        weights_partial[kb * T_PAD + w_r0 : kb * T_PAD + w_r0 + MM_ROW_TILE, :] = weights_acc

    with pl.spmd(
        1,
        name_hint="weights_proj_reduce",
        deps=[_weights_tid, index_score_gate_dep],
        predicate=(need_index_score[0] > 0),
        allow_early_resolve=True,
    ) as _weights_reduce_tid:
        _weights_reduce_block = pl.tile.get_block_idx()
        # Tile shapes must remain static even though only ``t_dim`` rows are
        # subsequently consumed by score.  The unwritten tail is never read by
        # a real token, so keep the historical T_PAD reduction extent here.
        w_sum = weights_partial[0:T_PAD, :]
        for kb in pl.unroll(1, WEIGHTS_OK):
            w_sum = pl.add(w_sum, weights_partial[kb * T_PAD : (kb + 1) * T_PAD, :])
        weights_row = _weights_reduce_block * T_PAD
        weights[
            weights_row : weights_row + T_PAD,
            0:IDX_N_HEADS,
        ] = pl.mul(w_sum, WEIGHTS_SCALE)

        # This fixed gather table used to require its own one-block child.
        # Generate it in the already-required score-gated weight reduction so
        # steady state loses one launch without changing the public L1 ABI.
        reorder_col_f = pl.cast(
            pl.arange(0, [1, SCORE_REORDER_SIZE], dtype=pl.INT32),
            target_type=pl.FP32,
        )
        reorder_token_i32 = pl.cast(
            pl.mul(reorder_col_f, 1.0 / SCORE_LANE_CACHE),
            target_type=pl.INT32,
            mode="trunc",
        )
        reorder_token_f = pl.cast(reorder_token_i32, target_type=pl.FP32)
        reorder_cache_f = pl.sub(reorder_col_f, pl.mul(reorder_token_f, 16.0))
        score_reorder_idx_t[0:1, 0:SCORE_REORDER_SIZE] = pl.cast(
            pl.add(pl.mul(reorder_cache_f, 8.0), reorder_token_f),
            target_type=pl.INT32,
            mode="trunc",
        )

    indexer_compressor(
        x,
        inner_kv,
        inner_compress_state,
        inner_compress_state_block_table,
        inner_wkv,
        inner_wgate,
        inner_ape,
        inner_norm_w,
        compress_cos,
        compress_sin,
        hadamard,
        idx_kv_cache,
        idx_kv_scale,
        idx_block_table,
        position_ids,
        kv_seq_lens,
        inner_state_page_stride,
        indexer_k_page_stride,
        indexer_scale_page_stride,
        late_dep,
    )

    idx_block_table_flat = pl.reshape(idx_block_table, [idx_table_len])
    qh_scale_flat = pl.reshape(qr_hadamard_scale_dq, [1, t_heads])
    weights_flat = pl.reshape(weights, [1, T_PAD * IDX_N_HEADS])
    score_flat = score

    # No score_init: the loop writes the valid region; the tail is never read (topk re-masks).
    # Fused mixed cube+vector: the per-page C8 matmul feeds the relu/weight/head-sum reduce
    # on chip, so cube page i+1 pipelines against vector page i (no score_acc_gm handoff).
    # Eight packed tokens make the cube->vector value 64 KiB.  An explicit
    # UP_DOWN AIV split below assigns 16 cache rows to each sibling.  Each lane
    # reduces all eight token groups with two 64-row reductions that reuse one
    # 16-KiB scratch tile.  Keeping that scratch below 32 KiB lets the page
    # pipeline retain two vector stages; a fixed gather then restores
    # token-major output order.
    with pl.spmd(
        score_units,
        name_hint="score",
        deps=[_qh_quant_tid, _weights_reduce_tid, index_score_gate_dep],
        predicate=(need_index_score[0] > 0),
        allow_early_resolve=True,
        optimizations=[pl.cross_core_slot(slot_num=1)],
    ) as _score_tid:
        unit = pl.tile.get_block_idx()
        token_group = unit // REDUCE_NSPLIT
        split = unit - token_group * REDUCE_NSPLIT
        groups_per_request = S // SCORE_TOKEN_GROUP
        b = token_group // groups_per_request
        group_in_request = token_group - b * groups_per_request
        s0 = group_in_request * SCORE_TOKEN_GROUP
        compressed_len = pl.read(kv_seq_lens, [b]) // COMPRESS_RATIO
        # Positions are reconstructed as consecutive rows by the public L1
        # entry.  The last token therefore has the largest visible prefix and
        # determines how many cache pages the whole group must visit.  Per-token
        # top-k masks its own shorter suffix later.
        pos_last = pl.read(position_ids, [b * S + s0 + SCORE_TOKEN_GROUP - 1])
        visible_len_group = pl.min(
            pl.min(compressed_len, (pos_last + 1) // COMPRESS_RATIO),
            SCORE_LEN,
        )
        cblk_group = (visible_len_group + REDUCE_TILE - 1) // REDUCE_TILE
        tb = b * S
        qb = (b * S + s0) * IDX_N_HEADS
        qr_group = pl.load(
            qr_hadamard_i8,
            [qb, 0],
            [SCORE_TOKEN_GROUP * IDX_N_HEADS, IDX_HEAD_DIM],
            target_memory=pl.MemorySpace.Mat,
        )
        qr_group_t = pl.tile.transpose_view(qr_group)
        qr_group_right = pl.move(
            qr_group_t,
            target_memory=pl.MemorySpace.Right,
        )
        # These three tiles are lane-invariant.  In manual split mode, GM loads
        # are deliberately legal outside the explicit AIV region and are
        # replicated onto both vector lanes by mixed-kernel outlining.  Hoisting
        # them also avoids reloading the same coefficients for every cache page.
        qh_scale_group = pl.load(
            qh_scale_flat,
            [0, qb],
            [1, SCORE_LANE_WIDTH],
        )
        weights_group = pl.load(
            weights_flat,
            [0, (tb + s0) * IDX_N_HEADS],
            [1, SCORE_LANE_WIDTH],
        )
        reorder_idx_tile = pl.load(
            score_reorder_idx_t,
            [0, 0],
            [1, SCORE_REORDER_SIZE],
        )
        lane_iters = (cblk_group - split + REDUCE_NSPLIT - 1) // REDUCE_NSPLIT
        for cb_local in pl.pipeline(0, lane_iters, stage=2):
            cb = split + cb_local * REDUCE_NSPLIT
            cache0 = cb * REDUCE_TILE
            idx_blk_id = pl.cast(
                pl.read(idx_block_table_flat, [b * idx_table_width + cache0 // BLOCK_SIZE]),
                pl.INDEX,
            )
            cache_intra = cache0 % BLOCK_SIZE
            cache_base = idx_blk_id * indexer_k_page_stride + cache_intra * IDX_HEAD_DIM
            kv_i8_flat = pl.load(
                idx_kv_cache,
                [0, cache_base],
                [1, REDUCE_TILE * IDX_HEAD_DIM],
                target_memory=pl.MemorySpace.Mat,
            )
            kv_i8_mat = pl.reshape(kv_i8_flat, [REDUCE_TILE, IDX_HEAD_DIM])
            score_acc_red = pl.matmul(kv_i8_mat, qr_group_right, out_dtype=pl.INT32)
            for aiv_id in pl.split_aiv(SCORE_AIV_LANES, mode=pl.SplitMode.UP_DOWN):
                score_lane = pl.aiv_shard(score_acc_red)
                scale_base = idx_blk_id * indexer_scale_page_stride + cache_intra
                kv_dq_flat = pl.load(
                    idx_kv_scale,
                    [0, scale_base + aiv_id * SCORE_LANE_CACHE],
                    [1, SCORE_LANE_CACHE],
                )
                kv_dq_row = pl.cast(kv_dq_flat, target_type=pl.FP32)

                score_lane = pl.cast(score_lane, target_type=pl.FP32, mode="none")
                score_lane = pl.col_expand_mul(score_lane, qh_scale_group)
                score_lane = pl.maximum(score_lane, 0.0)
                weighted_score_lane = pl.col_expand_mul(score_lane, weights_group)
                weighted_score_heads = pl.reshape(
                    weighted_score_lane,
                    [SCORE_REORDER_SIZE, IDX_N_HEADS],
                )
                reduce_tmp = pl.create_tile(
                    [SCORE_REDUCE_CHUNK_ROWS, IDX_N_HEADS],
                    dtype=pl.FP32,
                    target_memory=pl.MemorySpace.Vec,
                )
                score_rows_cache_major_0 = pl.reshape(
                    pl.row_sum(
                        weighted_score_heads[0:SCORE_REDUCE_CHUNK_ROWS, :],
                        reduce_tmp,
                    ),
                    [1, SCORE_REDUCE_CHUNK_ROWS],
                )
                score_rows_cache_major_1 = pl.reshape(
                    pl.row_sum(
                        weighted_score_heads[
                            SCORE_REDUCE_CHUNK_ROWS:SCORE_REORDER_SIZE,
                            :,
                        ],
                        reduce_tmp,
                    ),
                    [1, SCORE_REDUCE_CHUNK_ROWS],
                )
                score_rows_cache_major = pl.concat(
                    score_rows_cache_major_0,
                    score_rows_cache_major_1,
                )
                reorder_tmp = pl.create_tile(
                    [1, SCORE_REORDER_SIZE],
                    dtype=pl.INT32,
                    target_memory=pl.MemorySpace.Vec,
                )
                score_rows_token_major = pl.tile.gather(
                    score_rows_cache_major,
                    reorder_idx_tile,
                    reorder_tmp,
                )
                score_rows_token_major = pl.reshape(
                    score_rows_token_major,
                    [SCORE_LANE_TOKENS, SCORE_LANE_CACHE],
                )
                score_rows_token_major = pl.col_expand_mul(score_rows_token_major, kv_dq_row)
                pl.store(
                    score_rows_token_major,
                    [tb + s0, cache0 + aiv_id * SCORE_LANE_CACHE],
                    score_flat,
                )

    topk_idxs_flat = topk_idxs
    with pl.spmd(
        t_dim,
        name_hint="topk",
        deps=[_score_tid],
        allow_early_resolve=True,
    ) as _topk_tid:
        t = pl.tile.get_block_idx()
        batch_idx = t // S
        token_s = t - batch_idx * S
        cache_len_b = pl.read(kv_seq_lens, [batch_idx]) // COMPRESS_RATIO
        pos_t = pl.read(position_ids, [batch_idx * S + token_s])
        visible_len_t = pl.min(pl.min(cache_len_b, (pos_t + 1) // COMPRESS_RATIO), SCORE_LEN)
        if visible_len_t <= 0:
            topk_idxs_flat[t : t + 1, :] = pl.full([1, IDX_TOPK], dtype=pl.INT32, value=-1)
        if visible_len_t > 0:
            offset_i32 = pl.cast(offset, target_type=pl.INT32)
            if visible_len_t <= IDX_TOPK:
                # Every visible compressed position belongs to top-k.  Its
                # score order is irrelevant to the following attention: K/V
                # are gathered with the same permutation.  Emit the complete
                # logical set directly and leave the unused suffix at -1.
                topk_idxs_flat[t : t + 1, :] = pl.full([1, IDX_TOPK], dtype=pl.INT32, value=-1)
                idx_all = pl.arange(0, [1, IDX_TOPK], dtype=pl.INT32)
                idx_all_valid = pl.set_validshape(idx_all, 1, visible_len_t)
                topk_idxs_flat[t : t + 1, 0:IDX_TOPK] = pl.add(idx_all_valid, offset_i32)
            else:
                if visible_len_t <= TOPK_SHORT_LIMIT:
                    # Mature C8191 decode presents 2048 or 2049 visible
                    # compressed positions.  Materialize only the physical
                    # prefix and at most one 32-value sort block instead of
                    # loading/sorting the static 4096-value capacity.
                    prefix_valid = pl.min(visible_len_t, TOPK_HALF_LEN)
                    prefix_raw = score_flat[t : t + 1, 0:TOPK_HALF_LEN]
                    prefix = pl.fillpad(
                        pl.set_validshape(prefix_raw, 1, prefix_valid),
                        pad_value=pl.PadValue.min,
                    )
                    prefix = pl.maximum(
                        prefix,
                        pl.full([1, TOPK_HALF_LEN], dtype=pl.FP32, value=FP32_NEG_INF),
                    )
                    prefix_idx = pl.arange(0, [1, TOPK_HALF_LEN], dtype=pl.UINT32)
                    prefix_pairs = pl.sort32(prefix, prefix_idx)
                    prefix_pairs = pl.mrgsort(prefix_pairs, block_len=64)
                    prefix_pairs = pl.mrgsort(prefix_pairs, block_len=256)
                    prefix_pairs = pl.mrgsort(prefix_pairs, block_len=1024)
                    prefix_candidates = prefix_pairs[:, 0:TOPK_PAIR_WIDTH]

                    if visible_len_t <= TOPK_HALF_LEN:
                        short_idxs = pl.gather(
                            prefix_candidates,
                            mask_pattern=pl.tile.MaskPattern.P1010,
                            output_dtype=pl.INT32,
                        )
                        short_idxs = pl.set_validshape(short_idxs, 1, IDX_TOPK)
                        topk_idxs_flat[t : t + 1, 0:IDX_TOPK] = pl.add(short_idxs, offset_i32)
                    else:
                        tail_valid = pl.min(visible_len_t - TOPK_HALF_LEN, TOPK_SHORT_TAIL)
                        tail_raw = score_flat[
                            t : t + 1,
                            TOPK_HALF_LEN : TOPK_HALF_LEN + TOPK_SHORT_TAIL,
                        ]
                        tail = pl.fillpad(
                            pl.set_validshape(tail_raw, 1, tail_valid),
                            pad_value=pl.PadValue.min,
                        )
                        tail = pl.maximum(
                            tail,
                            pl.full([1, TOPK_SHORT_TAIL], dtype=pl.FP32, value=FP32_NEG_INF),
                        )
                        tail_idx = pl.arange(TOPK_HALF_LEN, [1, TOPK_SHORT_TAIL], dtype=pl.UINT32)
                        tail_pairs = pl.sort32(tail, tail_idx)
                        merged_short = pl.mrgsort(prefix_candidates, tail_pairs)
                        short_topk_pairs = merged_short[:, 0:TOPK_PAIR_WIDTH]
                        short_idxs = pl.gather(
                            short_topk_pairs,
                            mask_pattern=pl.tile.MaskPattern.P1010,
                            output_dtype=pl.INT32,
                        )
                        short_idxs = pl.set_validshape(short_idxs, 1, IDX_TOPK)
                        topk_idxs_flat[t : t + 1, 0:IDX_TOPK] = pl.add(short_idxs, offset_i32)
                else:
                    score_full_raw = score_flat[t : t + 1, 0:SCORE_LEN]
                    score_full = pl.fillpad(
                        pl.set_validshape(score_full_raw, 1, visible_len_t),
                        pad_value=pl.PadValue.min,
                    )
                    score_full = pl.maximum(
                        score_full,
                        pl.full([1, SCORE_LEN], dtype=pl.FP32, value=FP32_NEG_INF),
                    )
                    idx_init = pl.arange(0, [1, SCORE_LEN], dtype=pl.UINT32)
                    sorted_full = pl.sort32(score_full, idx_init)
                    sorted_full = pl.mrgsort(sorted_full, block_len=64)
                    sorted_full = pl.mrgsort(sorted_full, block_len=256)
                    sorted_full = pl.mrgsort(sorted_full, block_len=1024)

                    # After the 1024 merge, the 4096-score row is two sorted
                    # 2048-score runs.  Keep only each half's top-512 before
                    # the final two-way merge.
                    half0_candidates = sorted_full[:, 0:TOPK_PAIR_WIDTH]
                    half1_candidates = sorted_full[
                        :,
                        TOPK_HALF_PAIR_OFFSET : TOPK_HALF_PAIR_OFFSET + TOPK_PAIR_WIDTH,
                    ]
                    merged_candidates = pl.mrgsort(half0_candidates, half1_candidates)
                    topk_pairs = merged_candidates[:, 0:TOPK_PAIR_WIDTH]
                    topk_idxs_tile = pl.gather(
                        topk_pairs,
                        mask_pattern=pl.tile.MaskPattern.P1010,
                        output_dtype=pl.INT32,
                    )
                    topk_idxs_valid = pl.set_validshape(topk_idxs_tile, 1, IDX_TOPK)
                    topk_idxs_flat[t : t + 1, 0:IDX_TOPK] = pl.add(topk_idxs_valid, offset_i32)

    return score, topk_idxs
