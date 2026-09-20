# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Qwen3-14B decode implementation of the ``Qwen3Attention.forward`` boundary.

The framework keeps both RMSNorm/residual pairs and the complete MLP.  This
kernel receives the already normalized hidden states and replaces only the
coarse-grained native attention function: packed QKV projection, QK-norm,
RoPE, paged-KV update/read, attention and output projection.

QKV/O-projection weights retain vLLM's native ``[out, in]`` layout.  K/V cache
arguments are direct views of the framework-owned device allocation; the hot
path does not pack weights or copy cache pages.
"""

import pypto.language as pl

BATCH_PAD = 16
MAX_SEQ = 4096
NUM_HEADS = 40
NUM_KV_HEADS = 8
HEAD_DIM = 128
HIDDEN = 5120
KV_HIDDEN = NUM_KV_HEADS * HEAD_DIM
QKV_HIDDEN = HIDDEN + 2 * KV_HIDDEN
BLOCK_SIZE = 128
HALF_DIM = HEAD_DIM // 2
Q_PER_KV = NUM_HEADS // NUM_KV_HEADS
Q_HEAD_PAD = 16
K_NORM_ROW_PAD = 8
HEAD_DIM_INV = 1.0 / HEAD_DIM
ATTN_SCALE = HEAD_DIM**-0.5
RMS_EPS = 1e-6
NEG_INF = -3.0e38

BATCH_DYN = pl.dynamic("ATTENTION_LAYER_BATCH_DYN")
MAX_BLOCKS_DYN = pl.dynamic("ATTENTION_LAYER_MAX_BLOCKS_DYN")
KV_CACHE_ROWS_DYN = pl.dynamic("ATTENTION_LAYER_KV_CACHE_ROWS_DYN")

# Input padding and output materialization use five equal hidden slabs.
IO_BLOCKS = 5
IO_BLOCK_WIDTH = HIDDEN // IO_BLOCKS
IO_CHUNK = 256

# Packed QKV projection: native [out, in] weights with split-K accumulation.
QKV_TN = 256
QKV_TK = 256
QKV_N_TILE = 512
QKV_N_SUB = QKV_N_TILE // QKV_TN
QKV_K_SPLITS = 5
QKV_K_SLICE = HIDDEN // QKV_K_SPLITS
QKV_K_CHUNKS = QKV_K_SLICE // QKV_TK
Q_N_TILES = HIDDEN // QKV_N_TILE
KV_N_TILES = KV_HIDDEN // QKV_N_TILE

# Output projection topology mirrors the tuned full-model decode kernel.
OUT_N_SPLITS = 10
OUT_K_SPLITS = 5
OUT_TN = HIDDEN // OUT_N_SPLITS
OUT_K_SLICE = HIDDEN // OUT_K_SPLITS
OUT_TK = 64
OUT_K_CHUNKS = OUT_K_SLICE // OUT_TK

assert BATCH_PAD == 16
assert BLOCK_SIZE == 128
assert Q_PER_KV == 5
assert Q_HEAD_PAD >= Q_PER_KV
assert K_NORM_ROW_PAD * 4 >= 32
assert IO_BLOCK_WIDTH % IO_CHUNK == 0
assert QKV_N_TILE % QKV_TN == 0
assert QKV_K_SLICE % QKV_TK == 0
assert OUT_K_SLICE % OUT_TK == 0


@pl.jit
def qwen3_decode_attention_layer(  # noqa: PLR0913 - the attention ABI is intrinsic
    normalized_hidden: pl.Tensor[[BATCH_DYN, HIDDEN], pl.BF16],
    qkv_weight: pl.Tensor[[QKV_HIDDEN, HIDDEN], pl.BF16],
    q_norm_weight: pl.Tensor[[1, HEAD_DIM], pl.FP32],
    k_norm_weight: pl.Tensor[[1, HEAD_DIM], pl.FP32],
    positions: pl.Tensor[[BATCH_DYN], pl.INT64],
    block_table: pl.Tensor[[BATCH_DYN, MAX_BLOCKS_DYN], pl.INT32],
    slot_mapping: pl.Tensor[[BATCH_DYN], pl.INT64],
    rope_cos: pl.Tensor[[MAX_SEQ, HEAD_DIM], pl.FP32],
    rope_sin: pl.Tensor[[MAX_SEQ, HEAD_DIM], pl.FP32],
    k_cache: pl.InOut[pl.Tensor[[KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16]],
    v_cache: pl.InOut[pl.Tensor[[KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16]],
    o_proj_weight: pl.Tensor[[HIDDEN, HIDDEN], pl.BF16],
    attention_out: pl.Out[pl.Tensor[[BATCH_DYN, HIDDEN], pl.BF16]],
) -> pl.Tensor[[BATCH_DYN, HIDDEN], pl.BF16]:
    """Run decode-only self-attention for at most ``BATCH_PAD`` rows."""
    normalized_hidden.bind_dynamic(0, BATCH_DYN)
    positions.bind_dynamic(0, BATCH_DYN)
    block_table.bind_dynamic(0, BATCH_DYN)
    block_table.bind_dynamic(1, MAX_BLOCKS_DYN)
    slot_mapping.bind_dynamic(0, BATCH_DYN)
    k_cache.bind_dynamic(0, KV_CACHE_ROWS_DYN)
    v_cache.bind_dynamic(0, KV_CACHE_ROWS_DYN)
    attention_out.bind_dynamic(0, BATCH_DYN)

    batch = pl.tensor.dim(normalized_hidden, 0)
    hidden_pad = pl.create_tensor([BATCH_PAD, HIDDEN], dtype=pl.BF16)

    q_proj = pl.create_tensor([BATCH_PAD, HIDDEN], dtype=pl.FP32)
    k_proj = pl.create_tensor([BATCH_PAD, KV_HIDDEN], dtype=pl.FP32)
    v_proj = pl.create_tensor([BATCH_PAD, KV_HIDDEN], dtype=pl.FP32)
    all_q_padded = pl.create_tensor(
        [BATCH_PAD * NUM_KV_HEADS * Q_HEAD_PAD, HEAD_DIM],
        dtype=pl.BF16,
    )
    attention_pad = pl.create_tensor([BATCH_PAD, HIDDEN], dtype=pl.BF16)
    projected_attention = pl.create_tensor([BATCH_PAD, HIDDEN], dtype=pl.FP32)

    with pl.manual_scope():
        with pl.spmd(IO_BLOCKS, name_hint="pad_normalized_input", allow_early_resolve=True) as input_pad_tid:
            input_block = pl.get_block_idx()
            input_block_base = input_block * IO_BLOCK_WIDTH
            for input_chunk in pl.pipeline(IO_BLOCK_WIDTH // IO_CHUNK, stage=2):
                input_k0 = input_block_base + input_chunk * IO_CHUNK
                input_tile = pl.fillpad(
                    pl.slice(
                        normalized_hidden,
                        [BATCH_PAD, IO_CHUNK],
                        [0, input_k0],
                        valid_shape=[batch, IO_CHUNK],
                    ),
                    pad_value=pl.PadValue.zero,
                )
                hidden_pad = pl.assemble(hidden_pad, input_tile, [0, input_k0])

        with pl.at(level=pl.Level.CORE_GROUP, name_hint="q_seed", allow_early_resolve=True) as q_seed_tid:
            for q_seed_tile in pl.pipeline(Q_N_TILES, stage=2):
                q_seed_n0 = q_seed_tile * QKV_N_TILE
                q_proj = pl.assemble(
                    q_proj,
                    pl.full([BATCH_PAD, QKV_N_TILE], dtype=pl.FP32, value=0.0),
                    [0, q_seed_n0],
                )

        with pl.at(level=pl.Level.CORE_GROUP, name_hint="kv_seed", allow_early_resolve=True) as kv_seed_tid:
            for kv_seed_tile in pl.pipeline(KV_N_TILES, stage=2):
                kv_seed_n0 = kv_seed_tile * QKV_N_TILE
                kv_seed_zero = pl.full([BATCH_PAD, QKV_N_TILE], dtype=pl.FP32, value=0.0)
                k_proj = pl.assemble(k_proj, kv_seed_zero, [0, kv_seed_n0])
                v_proj = pl.assemble(v_proj, kv_seed_zero, [0, kv_seed_n0])

        with pl.spmd(
            Q_N_TILES * QKV_K_SPLITS,
            name_hint="q_proj",
            allow_early_resolve=True,
            deps=[input_pad_tid, q_seed_tid],
        ) as q_proj_tid:
            q_work = pl.get_block_idx()
            q_n_tile = q_work // QKV_K_SPLITS
            q_k_split = q_work % QKV_K_SPLITS
            q_n_region = q_n_tile * QKV_N_TILE
            q_k_base = q_k_split * QKV_K_SLICE
            for q_n_sub in pl.range(QKV_N_SUB):
                q_n0 = q_n_region + q_n_sub * QKV_TN
                q_acc = pl.matmul(
                    hidden_pad[:, q_k_base : q_k_base + QKV_TK],
                    qkv_weight[q_n0 : q_n0 + QKV_TN, q_k_base : q_k_base + QKV_TK],
                    b_trans=True,
                    out_dtype=pl.FP32,
                )
                for q_k_chunk in pl.pipeline(1, QKV_K_CHUNKS, stage=2):
                    q_k0 = q_k_base + q_k_chunk * QKV_TK
                    q_acc = pl.matmul_acc(
                        q_acc,
                        hidden_pad[:, q_k0 : q_k0 + QKV_TK],
                        qkv_weight[q_n0 : q_n0 + QKV_TN, q_k0 : q_k0 + QKV_TK],
                        b_trans=True,
                    )
                q_proj = pl.assemble(q_proj, q_acc, [0, q_n0], atomic=pl.AtomicType.Add)

        with pl.spmd(
            KV_N_TILES * QKV_K_SPLITS,
            name_hint="k_proj",
            allow_early_resolve=True,
            deps=[input_pad_tid, kv_seed_tid],
        ) as k_proj_tid:
            k_work = pl.get_block_idx()
            k_n_tile = k_work // QKV_K_SPLITS
            k_k_split = k_work % QKV_K_SPLITS
            k_n_region = k_n_tile * QKV_N_TILE
            k_k_base = k_k_split * QKV_K_SLICE
            for k_n_sub in pl.range(QKV_N_SUB):
                k_n0 = k_n_region + k_n_sub * QKV_TN
                k_weight_row = HIDDEN + k_n0
                k_acc = pl.matmul(
                    hidden_pad[:, k_k_base : k_k_base + QKV_TK],
                    qkv_weight[
                        k_weight_row : k_weight_row + QKV_TN,
                        k_k_base : k_k_base + QKV_TK,
                    ],
                    b_trans=True,
                    out_dtype=pl.FP32,
                )
                for k_k_chunk in pl.pipeline(1, QKV_K_CHUNKS, stage=2):
                    k_k0 = k_k_base + k_k_chunk * QKV_TK
                    k_acc = pl.matmul_acc(
                        k_acc,
                        hidden_pad[:, k_k0 : k_k0 + QKV_TK],
                        qkv_weight[
                            k_weight_row : k_weight_row + QKV_TN,
                            k_k0 : k_k0 + QKV_TK,
                        ],
                        b_trans=True,
                    )
                k_proj = pl.assemble(k_proj, k_acc, [0, k_n0], atomic=pl.AtomicType.Add)

        with pl.spmd(
            KV_N_TILES * QKV_K_SPLITS,
            name_hint="v_proj",
            allow_early_resolve=True,
            deps=[input_pad_tid, kv_seed_tid],
        ) as v_proj_tid:
            v_work = pl.get_block_idx()
            v_n_tile = v_work // QKV_K_SPLITS
            v_k_split = v_work % QKV_K_SPLITS
            v_n_region = v_n_tile * QKV_N_TILE
            v_k_base = v_k_split * QKV_K_SLICE
            for v_n_sub in pl.range(QKV_N_SUB):
                v_n0 = v_n_region + v_n_sub * QKV_TN
                v_weight_row = HIDDEN + KV_HIDDEN + v_n0
                v_acc = pl.matmul(
                    hidden_pad[:, v_k_base : v_k_base + QKV_TK],
                    qkv_weight[
                        v_weight_row : v_weight_row + QKV_TN,
                        v_k_base : v_k_base + QKV_TK,
                    ],
                    b_trans=True,
                    out_dtype=pl.FP32,
                )
                for v_k_chunk in pl.pipeline(1, QKV_K_CHUNKS, stage=2):
                    v_k0 = v_k_base + v_k_chunk * QKV_TK
                    v_acc = pl.matmul_acc(
                        v_acc,
                        hidden_pad[:, v_k0 : v_k0 + QKV_TK],
                        qkv_weight[
                            v_weight_row : v_weight_row + QKV_TN,
                            v_k0 : v_k0 + QKV_TK,
                        ],
                        b_trans=True,
                    )
                v_proj = pl.assemble(v_proj, v_acc, [0, v_n0], atomic=pl.AtomicType.Add)

        # The framework cache is physically [page, token, kv_head, head_dim].
        # Keep the borrowed storage and expose a token-major 2-D view so every
        # KV head is a column slice; no cache packing or device copy is needed.
        cache_token_rows = pl.tensor.dim(k_cache, 0) // NUM_KV_HEADS
        k_cache_bsnd = pl.reshape(k_cache, [cache_token_rows, KV_HIDDEN])
        v_cache_bsnd = pl.reshape(v_cache, [cache_token_rows, KV_HIDDEN])

        # Q/K RMSNorm, RoPE and the current-token cache update are pure PyPTO.
        # One SPMD work item owns one (batch, KV-head) pair and therefore writes
        # a disjoint cache column range and a disjoint padded-query tile.
        with pl.spmd(
            batch * NUM_KV_HEADS,
            name_hint="qk_norm_rope_cache",
            allow_early_resolve=True,
            deps=[q_proj_tid, k_proj_tid, v_proj_tid],
        ) as qkv_finalize_tid:
            qkv_work = pl.get_block_idx()
            qkv_batch = qkv_work // NUM_KV_HEADS
            kv_head = qkv_work % NUM_KV_HEADS
            kv_col = kv_head * HEAD_DIM
            position = pl.cast(pl.tensor.read(positions, [qkv_batch]), pl.INDEX)
            cache_row = pl.cast(pl.tensor.read(slot_mapping, [qkv_batch]), pl.INDEX)

            cos_row = pl.slice(rope_cos, [1, HEAD_DIM], [position, 0])
            sin_row = pl.slice(rope_sin, [1, HEAD_DIM], [position, 0])
            cos_lo = pl.slice(cos_row, [1, HALF_DIM], [0, 0])
            cos_hi = pl.slice(cos_row, [1, HALF_DIM], [0, HALF_DIM])
            sin_lo = pl.slice(sin_row, [1, HALF_DIM], [0, 0])
            sin_hi = pl.slice(sin_row, [1, HALF_DIM], [0, HALF_DIM])

            k_head = pl.slice(k_proj, [1, HEAD_DIM], [qkv_batch, kv_col])
            # PTO tile rows/columns must occupy at least one 32-byte vector.
            # Pad this single K head to eight FP32 rows before the reduction;
            # only row zero is consumed below, so the numerical result is
            # unchanged while all reduction temporaries remain legal tiles.
            k_block = pl.full([K_NORM_ROW_PAD, HEAD_DIM], dtype=pl.FP32, value=0.0)
            k_block = pl.assemble(k_block, k_head, [0, 0])
            k_sq = pl.row_sum(pl.mul(k_block, k_block))
            k_inv_rms = pl.recip(pl.sqrt(pl.add(pl.mul(k_sq, HEAD_DIM_INV), RMS_EPS)))
            k_normed_block = pl.col_expand_mul(
                pl.row_expand_mul(k_block, k_inv_rms),
                k_norm_weight,
            )
            k_normed = pl.slice(k_normed_block, [1, HEAD_DIM], [0, 0])
            k_lo = pl.slice(k_normed, [1, HALF_DIM], [0, 0])
            k_hi = pl.slice(k_normed, [1, HALF_DIM], [0, HALF_DIM])
            k_rot_lo = pl.sub(
                pl.col_expand_mul(k_lo, cos_lo),
                pl.col_expand_mul(k_hi, sin_lo),
            )
            k_rot_hi = pl.add(
                pl.col_expand_mul(k_hi, cos_hi),
                pl.col_expand_mul(k_lo, sin_hi),
            )
            k_cache_bsnd = pl.assemble(
                k_cache_bsnd,
                pl.cast(k_rot_lo, target_type=pl.BF16),
                [cache_row, kv_col],
            )
            k_cache_bsnd = pl.assemble(
                k_cache_bsnd,
                pl.cast(k_rot_hi, target_type=pl.BF16),
                [cache_row, kv_col + HALF_DIM],
            )
            v_cache_bsnd = pl.assemble(
                v_cache_bsnd,
                pl.cast(
                    pl.slice(v_proj, [1, HEAD_DIM], [qkv_batch, kv_col]),
                    target_type=pl.BF16,
                ),
                [cache_row, kv_col],
            )

            q_col = kv_head * Q_PER_KV * HEAD_DIM
            q_raw = pl.reshape(
                pl.slice(q_proj, [1, Q_PER_KV * HEAD_DIM], [qkv_batch, q_col]),
                [Q_PER_KV, HEAD_DIM],
            )
            q_block = pl.full([Q_HEAD_PAD, HEAD_DIM], dtype=pl.FP32, value=0.0)
            q_block = pl.assemble(q_block, q_raw, [0, 0])
            q_sq = pl.reshape(pl.row_sum(pl.mul(q_block, q_block)), [Q_HEAD_PAD, 1])
            q_inv_rms = pl.recip(pl.sqrt(pl.add(pl.mul(q_sq, HEAD_DIM_INV), RMS_EPS)))
            q_normed = pl.col_expand_mul(
                pl.row_expand_mul(q_block, q_inv_rms),
                q_norm_weight,
            )
            q_lo = pl.slice(q_normed, [Q_HEAD_PAD, HALF_DIM], [0, 0])
            q_hi = pl.slice(q_normed, [Q_HEAD_PAD, HALF_DIM], [0, HALF_DIM])
            q_rot_lo = pl.sub(
                pl.col_expand_mul(q_lo, cos_lo),
                pl.col_expand_mul(q_hi, sin_lo),
            )
            q_rot_hi = pl.add(
                pl.col_expand_mul(q_hi, cos_hi),
                pl.col_expand_mul(q_lo, sin_hi),
            )
            q_pad_row = qkv_work * Q_HEAD_PAD
            all_q_padded = pl.assemble(
                all_q_padded,
                pl.cast(q_rot_lo, target_type=pl.BF16),
                [q_pad_row, 0],
            )
            all_q_padded = pl.assemble(
                all_q_padded,
                pl.cast(q_rot_hi, target_type=pl.BF16),
                [q_pad_row, HALF_DIM],
            )

        with pl.at(level=pl.Level.CORE_GROUP, name_hint="attn_out_seed", allow_early_resolve=True) as attn_seed_tid:
            for attn_seed_tile in pl.pipeline(Q_N_TILES, stage=2):
                attn_seed_n0 = attn_seed_tile * QKV_N_TILE
                attention_pad = pl.assemble(
                    attention_pad,
                    pl.full([BATCH_PAD, QKV_N_TILE], dtype=pl.BF16, value=0.0),
                    [0, attn_seed_n0],
                )

        # Online-softmax paged attention. Each work item consumes one padded
        # group of five Q heads and streams the request's physical KV pages.
        with pl.spmd(
            batch * NUM_KV_HEADS,
            name_hint="paged_attention",
            allow_early_resolve=True,
            deps=[
                qkv_finalize_tid,
                attn_seed_tid,
            ],
        ) as attention_tid:
            attention_work = pl.get_block_idx()
            attention_batch = attention_work // NUM_KV_HEADS
            attention_kv_head = attention_work % NUM_KV_HEADS
            attention_kv_col = attention_kv_head * HEAD_DIM
            context_len = pl.cast(
                pl.tensor.read(positions, [attention_batch]) + 1,
                pl.INDEX,
            )
            context_blocks = (context_len + BLOCK_SIZE - 1) // BLOCK_SIZE
            q_padded = pl.slice(
                all_q_padded,
                [Q_HEAD_PAD, HEAD_DIM],
                [attention_work * Q_HEAD_PAD, 0],
            )
            # Keep scalar-per-head online-softmax state in ND [1, heads]
            # layout.  A freshly allocated ND [heads, 1] tile has a four-byte
            # row and is rejected by ptoas; row reductions below naturally
            # produce the DN [heads, 1] form needed for broadcasting.
            running_max_nd = pl.full([1, Q_HEAD_PAD], dtype=pl.FP32, value=NEG_INF)
            running_sum_nd = pl.full([1, Q_HEAD_PAD], dtype=pl.FP32, value=0.0)
            running_out = pl.full([Q_HEAD_PAD, HEAD_DIM], dtype=pl.FP32, value=0.0)

            for block_index in pl.range(context_blocks):
                physical_block = pl.cast(
                    pl.tensor.read(block_table, [attention_batch, block_index]),
                    pl.INDEX,
                )
                cache_block_row = physical_block * BLOCK_SIZE
                key_block = pl.slice(
                    k_cache_bsnd,
                    [BLOCK_SIZE, HEAD_DIM],
                    [cache_block_row, attention_kv_col],
                )
                value_block = pl.slice(
                    v_cache_bsnd,
                    [BLOCK_SIZE, HEAD_DIM],
                    [cache_block_row, attention_kv_col],
                )
                raw_scores = pl.matmul(
                    q_padded,
                    key_block,
                    b_trans=True,
                    out_dtype=pl.FP32,
                )
                block_start = block_index * BLOCK_SIZE
                valid_len = pl.min(BLOCK_SIZE, context_len - block_start)
                scores = pl.fillpad(
                    pl.set_validshape(
                        pl.mul(raw_scores, ATTN_SCALE),
                        Q_HEAD_PAD,
                        valid_len,
                    ),
                    pad_value=pl.PadValue.min,
                )
                current_max = pl.row_max(scores)
                exp_scores = pl.exp(pl.row_expand_sub(scores, current_max))
                exp_scores_bf16 = pl.cast(exp_scores, target_type=pl.BF16)
                current_sum = pl.row_sum(pl.cast(exp_scores_bf16, target_type=pl.FP32))
                current_out = pl.matmul(
                    exp_scores_bf16,
                    value_block,
                    out_dtype=pl.FP32,
                )

                # Reductions produce DN [rows, 1] tiles. Elementwise binary
                # operations require ND here, so transpose their logical shape
                # to [1, rows] and restore DN before row broadcasting.
                current_max_nd = pl.reshape(current_max, [1, Q_HEAD_PAD])
                current_sum_nd = pl.reshape(current_sum, [1, Q_HEAD_PAD])
                merged_max_nd = pl.maximum(running_max_nd, current_max_nd)
                old_scale_nd = pl.exp(pl.sub(running_max_nd, merged_max_nd))
                new_scale_nd = pl.exp(pl.sub(current_max_nd, merged_max_nd))
                running_sum_nd = pl.add(
                    pl.mul(old_scale_nd, running_sum_nd),
                    pl.mul(new_scale_nd, current_sum_nd),
                )
                old_scale = pl.reshape(old_scale_nd, [Q_HEAD_PAD, 1])
                new_scale = pl.reshape(new_scale_nd, [Q_HEAD_PAD, 1])
                running_out = pl.add(
                    pl.row_expand_mul(running_out, old_scale),
                    pl.row_expand_mul(current_out, new_scale),
                )
                running_max_nd = merged_max_nd

            running_sum = pl.reshape(running_sum_nd, [Q_HEAD_PAD, 1])
            context = pl.row_expand_div(running_out, running_sum)
            context_row = pl.reshape(
                pl.slice(
                    pl.cast(context, target_type=pl.BF16),
                    [Q_PER_KV, HEAD_DIM],
                    [0, 0],
                ),
                [1, Q_PER_KV * HEAD_DIM],
            )
            attention_pad = pl.assemble(
                attention_pad,
                context_row,
                [attention_batch, attention_kv_head * Q_PER_KV * HEAD_DIM],
            )

        with pl.at(level=pl.Level.CORE_GROUP, name_hint="out_proj_seed", allow_early_resolve=True) as out_seed_tid:
            for out_seed_split in pl.pipeline(OUT_N_SPLITS, stage=2):
                out_seed_n0 = out_seed_split * OUT_TN
                projected_attention = pl.assemble(
                    projected_attention,
                    pl.full([BATCH_PAD, OUT_TN], dtype=pl.FP32, value=0.0),
                    [0, out_seed_n0],
                )

        with pl.spmd(
            OUT_N_SPLITS * OUT_K_SPLITS,
            name_hint="out_proj",
            deps=[attention_tid, out_seed_tid],
        ) as out_proj_tid:
            out_work = pl.get_block_idx()
            out_n_split = out_work // OUT_K_SPLITS
            out_k_split = out_work % OUT_K_SPLITS
            out_n0 = out_n_split * OUT_TN
            out_k_base = out_k_split * OUT_K_SLICE
            out_acc = pl.matmul(
                attention_pad[:, out_k_base : out_k_base + OUT_TK],
                o_proj_weight[
                    out_n0 : out_n0 + OUT_TN,
                    out_k_base : out_k_base + OUT_TK,
                ],
                b_trans=True,
                out_dtype=pl.FP32,
            )
            for out_k_chunk in pl.pipeline(1, OUT_K_CHUNKS, stage=2):
                out_k0 = out_k_base + out_k_chunk * OUT_TK
                out_acc = pl.matmul_acc(
                    out_acc,
                    attention_pad[:, out_k0 : out_k0 + OUT_TK],
                    o_proj_weight[
                        out_n0 : out_n0 + OUT_TN,
                        out_k0 : out_k0 + OUT_TK,
                    ],
                    b_trans=True,
                )
            projected_attention = pl.assemble(
                projected_attention,
                out_acc,
                [0, out_n0],
                atomic=pl.AtomicType.Add,
            )

        with pl.spmd(
            IO_BLOCKS,
            name_hint="materialize_attention_output",
            deps=[out_proj_tid],
        ) as _output_tid:
            output_block = pl.get_block_idx()
            output_k_base = output_block * IO_BLOCK_WIDTH
            output_tile = pl.cast(
                projected_attention[:, output_k_base : output_k_base + IO_BLOCK_WIDTH],
                target_type=pl.BF16,
            )
            output_tile_valid = pl.set_validshape(
                output_tile,
                batch,
                IO_BLOCK_WIDTH,
            )
            attention_out = pl.assemble(
                attention_out,
                output_tile_valid,
                [0, output_k_base],
            )

    return attention_out


__all__ = ["qwen3_decode_attention_layer"]
