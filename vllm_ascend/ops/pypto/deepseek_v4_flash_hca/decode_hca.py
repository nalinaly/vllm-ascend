# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""TP1 HCA 整层入口；公共投影和 mHC 复用已接入的 CSA 实现。"""

import pypto.language as pl

from .hc_pre_fused import NORM_EPS, hc_pre_norm
from ..deepseek_v4_flash_dspark_perf.nz_mode import BF16_WEIGHT_LAYOUT, QUANT_WEIGHT_LAYOUT, WO_A_WEIGHT_LAYOUT
from ..deepseek_v4_flash_dspark_perf.qkv_proj_rope import qkv_proj_rope
from .decode_compressor_ratio128 import (
    B_DYN, BOUNDS_DYN, CMP_PAGES_DYN, COMPACT_ROWS_DYN, STATE_COLUMNS_DYN,
    STATE_PAGE_ELEMENTS_DYN, STATE_PAGES_DYN, T_DYN, compressor_ratio128,
)
from .decode_sparse_attn_hca import (
    ORI_BLOCK_NUM_DYN, ORI_TABLE_COLUMNS_DYN, CMP_TABLE_BLOCKS_DYN, T_PAD, sparse_attn_hca_tp1,
)
from .o_proj_hc_post import o_proj_hc_post
from .weight_warm import SINK_BF16, WARM_WORKERS, warm_kv_weights

D = 4096
H = 64
HEAD_DIM = 512
ROPE_DIM = 64
Q_LORA = 1024
HC_MULT = 4
HC_DIM = HC_MULT * D
MIX_HC = 24
O_GROUPS = 8
O_GROUP_IN = 4096
O_LORA = 1024
WIDEN_ROWS = 8
RMS_COLS = 512
WIDEN_WORKERS = 48
CACHE_WORKERS = 8


def _decode_hca_tp1_layer(
    x_hc: pl.Tensor[[T_DYN, HC_MULT, D], pl.BF16],
    hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[3], pl.FP32],
    hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32],
    attn_norm_w: pl.Tensor[[D], pl.BF16],
    wq_a: pl.Tensor[[Q_LORA, D], pl.BF16, BF16_WEIGHT_LAYOUT],
    wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8, QUANT_WEIGHT_LAYOUT],
    wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16],
    gamma_cq: pl.Tensor[[Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16],
    freqs_cos: pl.Tensor[[T_DYN, ROPE_DIM], pl.FP32],
    freqs_sin: pl.Tensor[[T_DYN, ROPE_DIM], pl.FP32],
    cmp_wkv: pl.Tensor[[HEAD_DIM, D], pl.BF16],
    cmp_wgate: pl.Tensor[[HEAD_DIM, D], pl.BF16],
    cmp_ape: pl.Tensor[[128, HEAD_DIM], pl.FP32],
    cmp_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    state: pl.InOut[pl.Tensor[[STATE_PAGES_DYN, STATE_PAGE_ELEMENTS_DYN], pl.FP32]],
    state_table: pl.Tensor[[B_DYN, STATE_COLUMNS_DYN], pl.INT32],
    state_slots: pl.Tensor[[T_DYN, 2], pl.INT32],
    positions: pl.Tensor[[T_DYN], pl.INT64],
    query_bounds: pl.Tensor[[BOUNDS_DYN], pl.INT32],
    seq_lens: pl.Tensor[[B_DYN], pl.INT32],
    cmp_cos: pl.Tensor[[COMPACT_ROWS_DYN, ROPE_DIM], pl.FP32],
    cmp_sin: pl.Tensor[[COMPACT_ROWS_DYN, ROPE_DIM], pl.FP32],
    cmp_slots: pl.Tensor[[COMPACT_ROWS_DYN, 2], pl.INT32],
    cmp_cache: pl.InOut[pl.Tensor[[CMP_PAGES_DYN, 32, 1, HEAD_DIM], pl.BF16]],
    cmp_table: pl.Tensor[[B_DYN, CMP_TABLE_BLOCKS_DYN], pl.INT32],
    ori_cache: pl.InOut[pl.Tensor[[ORI_BLOCK_NUM_DYN, 32, 1, HEAD_DIM], pl.BF16]],
    ori_table: pl.Tensor[[B_DYN, ORI_TABLE_COLUMNS_DYN], pl.INT32],
    ori_slots: pl.Tensor[[T_DYN, 2], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    wo_a: pl.Tensor[[O_GROUPS, O_GROUP_IN, O_LORA], pl.BF16, WO_A_WEIGHT_LAYOUT],
    wo_b: pl.Tensor[[O_GROUPS * O_LORA, D], pl.INT8, QUANT_WEIGHT_LAYOUT],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    x_out: pl.Out[pl.Tensor[[T_DYN, HC_MULT, D], pl.BF16]],
):
    x_hc.bind_dynamic(0, T_DYN)
    x_out.bind_dynamic(0, T_DYN)
    freqs_cos.bind_dynamic(0, T_DYN)
    freqs_sin.bind_dynamic(0, T_DYN)
    state.bind_dynamic(0, STATE_PAGES_DYN)
    state.bind_dynamic(1, STATE_PAGE_ELEMENTS_DYN)
    state_table.bind_dynamic(0, B_DYN)
    state_table.bind_dynamic(1, STATE_COLUMNS_DYN)
    state_slots.bind_dynamic(0, T_DYN)
    positions.bind_dynamic(0, T_DYN)
    query_bounds.bind_dynamic(0, BOUNDS_DYN)
    seq_lens.bind_dynamic(0, B_DYN)
    cmp_cos.bind_dynamic(0, COMPACT_ROWS_DYN)
    cmp_sin.bind_dynamic(0, COMPACT_ROWS_DYN)
    cmp_slots.bind_dynamic(0, COMPACT_ROWS_DYN)
    cmp_cache.bind_dynamic(0, CMP_PAGES_DYN)
    cmp_table.bind_dynamic(0, B_DYN)
    cmp_table.bind_dynamic(1, CMP_TABLE_BLOCKS_DYN)
    ori_cache.bind_dynamic(0, ORI_BLOCK_NUM_DYN)
    ori_table.bind_dynamic(0, B_DYN)
    ori_table.bind_dynamic(1, ORI_TABLE_COLUMNS_DYN)
    ori_slots.bind_dynamic(0, T_DYN)

    tokens = pl.tensor.dim(x_hc, 0)
    post = pl.create_tensor([tokens, HC_MULT], dtype=pl.FP32)
    comb = pl.create_tensor([tokens, HC_MULT * HC_MULT], dtype=pl.FP32)
    normalized = pl.create_tensor([tokens, D], dtype=pl.BF16)
    x32 = pl.create_tensor([tokens, HC_MULT, D], dtype=pl.FP32)
    x_flat = pl.reshape(x_hc, [tokens, HC_DIM])
    x32_flat = pl.reshape(x32, [tokens, HC_DIM])
    widen_blocks = (tokens + WIDEN_ROWS - 1) // WIDEN_ROWS
    hc_padded_rows = ((tokens + 16 - 1) // 16) * 16
    inv_rms = pl.create_tensor([hc_padded_rows, 1], dtype=pl.FP32)
    tail = pl.create_tensor([WIDEN_ROWS, HC_DIM], dtype=pl.FP32)
    # 参考 CSA d1f170ff：加宽时按原 RMS 的 512 列次序顺手求平方和，删除 RMS 对 FP32
    # 中间缓冲的再次读取；归约次序、高精度 rsqrt 不变，结果与独立 RMS 任务逐 bit 相同。
    with pl.spmd(pl.min(widen_blocks, WIDEN_WORKERS), name_hint="hca_hc_widen_rms", allow_early_resolve=True) as widen_tid:
        for block in pl.range(pl.tile.get_block_idx(), widen_blocks, pl.min(widen_blocks, WIDEN_WORKERS)):
            row = block * WIDEN_ROWS
            count = pl.min(WIDEN_ROWS, tokens - row)
            sq_sum = pl.full([1, WIDEN_ROWS], dtype=pl.FP32, value=0.0)
            for col_block in pl.pipeline(HC_DIM // RMS_COLS, stage=4):
                col = col_block * RMS_COLS
                source = pl.slice(x_flat, [WIDEN_ROWS, RMS_COLS], [row, col], valid_shape=[count, RMS_COLS])
                value = pl.cast(source, pl.FP32)
                if count == WIDEN_ROWS:
                    x32_flat[row:row + WIDEN_ROWS, col:col + RMS_COLS] = value
                    squared = pl.mul(value, value)
                    sq_sum = pl.add(sq_sum, pl.reshape(pl.row_sum(squared), [1, WIDEN_ROWS]))
                else:
                    # cast 可能丢失 valid_shape，显式恢复并清零无效行后才参与归约。
                    clean = pl.fillpad(pl.set_validshape(value, count, RMS_COLS), pad_value=pl.PadValue.zero)
                    squared_tail = pl.mul(clean, clean)
                    sq_sum = pl.add(sq_sum, pl.reshape(pl.row_sum(squared_tail), [1, WIDEN_ROWS]))
                    tail[:, col:col + RMS_COLS] = value
                    valid = pl.load(tail, [0, col], [WIDEN_ROWS, RMS_COLS], valid_shape=[count, RMS_COLS],
                                    target_memory=pl.MemorySpace.Vec)
                    pl.store(valid, [row, col], x32_flat)
            mean = pl.add(pl.mul(sq_sum, 1.0 / HC_DIM), NORM_EPS)
            inv_rms[row:row + WIDEN_ROWS, 0:1] = pl.reshape(pl.rsqrt(mean, high_precision=True), [WIDEN_ROWS, 1])
    # 冷 L2 下投影读权重受单核带宽限制；mHC pre 期间 Vector 核大多空闲，先预热 KV/compressor 权重。
    warm_sink = pl.create_tensor([WARM_WORKERS, SINK_BF16], dtype=pl.BF16)
    warm_kv_weights(wkv, cmp_wkv, cmp_wgate, warm_sink, widen_tid)
    with pl.scope():
        hc_pre_norm(
            x32, hc_attn_fn, hc_attn_scale, hc_attn_base, attn_norm_w, post, comb, normalized, False, inv_rms, x_hc,
        )

    q = pl.create_tensor([tokens, H, HEAD_DIM], dtype=pl.BF16)
    kv = pl.create_tensor([tokens, HEAD_DIM], dtype=pl.BF16)
    qr = pl.create_tensor([tokens, Q_LORA], dtype=pl.INT8)
    qr_scale = pl.create_tensor([tokens, 1], dtype=pl.FP32)
    with pl.scope():
        ready = pl.system.task_dummy(deps=[])
        q, qa_tid = qkv_proj_rope(
            normalized, wq_a, wq_b, wq_b_scale, wkv, freqs_cos, freqs_sin,
            gamma_cq, gamma_ckv, q, kv, qr, qr_scale, ready,
        )
        cache_rows = pl.tensor.dim(ori_cache, 0) * 32
        cache_flat = pl.reshape(ori_cache, [cache_rows, HEAD_DIM])
        with pl.spmd(CACHE_WORKERS, name_hint="hca_raw_cache_write", allow_early_resolve=True) as raw_tid:
            for token in pl.range(pl.tile.get_block_idx(), tokens, CACHE_WORKERS):
                page = pl.read(ori_slots, [token, 0])
                offset = pl.read(ori_slots, [token, 1])
                if page >= 0 and offset >= 0 and pl.read(seq_lens, [token // 6]) > 0:
                    cache_row = pl.cast(page, pl.INDEX) * 32 + offset
                    cache_flat[cache_row:cache_row + 1, :] = kv[token:token + 1, :]
        compressed_tid = compressor_ratio128(
            normalized, cmp_wkv, cmp_wgate, cmp_ape, cmp_norm_w, state, state_table,
            state_slots, positions, query_bounds, seq_lens, cmp_cos, cmp_sin, cmp_slots, cmp_cache, qa_tid,
        )
        packed = pl.create_tensor([O_GROUPS * T_PAD, O_GROUP_IN], dtype=pl.BF16)
        packed, heads_tid = sparse_attn_hca_tp1(
            q, ori_cache, ori_table, cmp_cache, cmp_table, positions, seq_lens, attn_sink,
            freqs_cos, freqs_sin, packed, raw_tid, compressed_tid,
        )
        with pl.scope():
            x_out = o_proj_hc_post(packed, wo_a, wo_b, wo_b_scale, x_hc, post, comb, x_out, heads_tid)
    return x_out


decode_hca_tp1_layer = pl.jit.inline(auto_scope=False)(_decode_hca_tp1_layer)
decode_hca_tp1_layer_test = pl.jit(auto_scope=False)(_decode_hca_tp1_layer)
