# SPDX-License-Identifier: Apache-2.0
"""DeepSeek V4 Flash 独立 HC_pre，完整实现位于本文件。

从本仓库 deepseek_v4_flash_csa/hc_pre.py 搬入 HC_pre 计算与尾块处理，
原始算法来自 pypto-lib 的同名实现；运行时不导入 CSA/HCA 算子或配置。
包含 BF16 加宽、RMS 统计、投影、门控、Sinkhorn 和残差混合，
不包含 mixed 之后的 input_layernorm。T 为动态 token 数，即 B×S。
"""

import pypto.language as pl

# 正式 DeepSeek V4 Flash 的 HC 配置；独立入口不依赖 CSA 配置模块。
D = 4096
HC_MULT = 4
MIX_HC = (2 + HC_MULT) * HC_MULT
HC_DIM = HC_MULT * D
HC_DIM_INV = 1.0 / HC_DIM
HC_SINKHORN_ITER = 20
HC_EPS = 1e-6
NORM_EPS = 1e-6

TOKENS = pl.dynamic("HC_PRE_TOKENS")
T_DYN = pl.dynamic("T_DYN")
HC_PAD_ROWS_DYN = pl.dynamic("HC_PAD_ROWS_DYN")

# 保留已验证的分块及任务划分；四路门控与混合在下方显式展开。
MIX_PAD = 32  # 24 列投影补齐至 Cube N 轴和 Vector 行的 32 列。
HC_PAD = 8  # 4 列有效值按 32B 对齐。
T_TILE = 8
LINEAR_T_TILE = 64
COMB_T_TILE = 8
PRE_POST_WORKERS = 16
RMS_K_TILE = 2048
LINEAR_K_TILE = 512
D_TILE = 512
D_SPMD = 4096
LINEAR_OK = 8
LINEAR_K_PER_SPLIT = HC_DIM // LINEAR_OK
LINEAR_WORKERS = 24
WIDEN_COLS = 8192
WIDEN_WORKERS = 48


@pl.jit(auto_scope=False)
def hc_pre(
    x: pl.Tensor[[TOKENS, HC_MULT, D], pl.BF16],
    hc_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_scale: pl.Tensor[[3], pl.FP32],
    hc_base: pl.Tensor[[MIX_HC], pl.FP32],
    mixed: pl.Out[pl.Tensor[[TOKENS, D], pl.BF16]],
    post: pl.Out[pl.Tensor[[TOKENS, HC_MULT], pl.FP32]],
    comb_out: pl.Out[pl.Tensor[[TOKENS, HC_MULT, HC_MULT], pl.FP32]],
):
    """一次 PTO 调用写入三份 Native 同形状输出；所有输入权重均为 device tensor。"""
    tokens = pl.tensor.dim(x, 0)
    x_flat = pl.reshape(x, [tokens, HC_DIM])
    weight_parts = pl.create_tensor([MIX_HC * 3, HC_DIM], dtype=pl.BF16)
    # FP32 权重在本次调用内拆为三段 BF16，合并为一个 N 轴矩阵投影。
    for part in pl.spmd(48, name_hint="hc_weight_triple", allow_early_resolve=True):
        wr = part // 2
        wc = (part % 2) * 8192
        w = hc_fn[wr:wr + 1, wc:wc + 8192]
        high = pl.cast(w, pl.BF16, mode="rint")
        residual = pl.sub(w, pl.cast(high, pl.FP32))
        low = pl.cast(residual, pl.BF16, mode="rint")
        last = pl.cast(pl.sub(residual, pl.cast(low, pl.FP32)), pl.BF16, mode="rint")
        weight_parts[wr:wr + 1, wc:wc + 8192] = high
        weight_parts[MIX_HC + wr:MIX_HC + wr + 1, wc:wc + 8192] = low
        weight_parts[MIX_HC * 2 + wr:MIX_HC * 2 + wr + 1, wc:wc + 8192] = last
    t_dim = tokens
    token_tiles = (tokens + T_TILE - 1) // T_TILE
    t_linear = ((tokens + LINEAR_T_TILE - 1) // LINEAR_T_TILE) * LINEAR_T_TILE
    comb = pl.reshape(comb_out, [tokens, HC_MULT * HC_MULT])
    hc_base_2d = pl.reshape(hc_base, [1, MIX_HC])
    inv_rms = pl.create_tensor([t_linear, 1], dtype=pl.FP32)
    # 每个 token 块沿完整 K 轴累加平方和，生成 RMS 倒数。
    for t in pl.spmd(token_tiles, name_hint="hc_pre_rms", allow_early_resolve=True):
        t0 = t * T_TILE
        valid_rows = pl.min(T_TILE, t_dim - t0)
        sq_sum = pl.full([1, T_TILE], dtype=pl.FP32, value=0.0)
        for kb in pl.pipeline(HC_DIM // RMS_K_TILE, stage=2):
            k0 = kb * RMS_K_TILE
            if valid_rows == T_TILE:
                x_chunk_full = pl.cast(x_flat[t0:t0 + T_TILE, k0:k0 + RMS_K_TILE], pl.FP32)
                x_sq_full = pl.mul(x_chunk_full, x_chunk_full)
                x_sq_row_full = pl.reshape(pl.row_sum(x_sq_full), [1, T_TILE])
                sq_sum = pl.add(sq_sum, x_sq_row_full)
            else:
                x_chunk_tail = pl.slice(x_flat, [T_TILE, RMS_K_TILE], [t0, k0], valid_shape=[valid_rows, RMS_K_TILE])
                x_chunk_fp32 = pl.cast(x_chunk_tail, pl.FP32)
                x_sq_tail = pl.mul(x_chunk_fp32, x_chunk_fp32)
                x_sq_row_tail = pl.reshape(pl.row_sum(x_sq_tail), [1, T_TILE])
                sq_sum = pl.add(sq_sum, x_sq_row_tail)
        sq_mean = pl.add(pl.mul(sq_sum, HC_DIM_INV), NORM_EPS)
        inv = pl.reshape(pl.rsqrt(sq_mean, high_precision=True), [T_TILE, 1])
        inv_rms[t0:t0 + T_TILE, 0:1] = inv

    mixes_partials = pl.create_tensor([LINEAR_OK * t_linear, 80], dtype=pl.FP32)
    linear_units = (t_linear // LINEAR_T_TILE) * LINEAR_OK
    linear_workers = pl.min(linear_units, LINEAR_WORKERS)
    for linear_worker in pl.spmd(linear_workers, name_hint="hc_pre_linear", allow_early_resolve=True):
        for task in pl.range(linear_worker, linear_units, linear_workers):
            t0 = (task // LINEAR_OK) * LINEAR_T_TILE
            linear_split = task % LINEAR_OK
            k_base = linear_split * LINEAR_K_PER_SPLIT
            t_rows = pl.min(LINEAR_T_TILE, t_dim - t0)
            x_first = pl.slice(x_flat, [LINEAR_T_TILE, LINEAR_K_TILE], [t0, k_base], valid_shape=[t_rows, LINEAR_K_TILE])
            w_first = pl.slice(weight_parts, [80, LINEAR_K_TILE], [0, k_base], valid_shape=[MIX_HC * 3, LINEAR_K_TILE])
            acc = pl.matmul(x_first, w_first, b_trans=True, out_dtype=pl.FP32)
            for kb in pl.pipeline(1, LINEAR_K_PER_SPLIT // LINEAR_K_TILE, stage=2):
                k0 = k_base + kb * LINEAR_K_TILE
                x_chunk = pl.slice(x_flat, [LINEAR_T_TILE, LINEAR_K_TILE], [t0, k0], valid_shape=[t_rows, LINEAR_K_TILE])
                w_chunk = pl.slice(weight_parts, [80, LINEAR_K_TILE], [0, k0], valid_shape=[MIX_HC * 3, LINEAR_K_TILE])
                acc = pl.matmul_acc(acc, x_chunk, w_chunk, b_trans=True)
            pr = linear_split * t_linear + t0
            mixes_partials[pr:pr + LINEAR_T_TILE, 0:80] = acc
    parts_2d = pl.reshape(mixes_partials, [LINEAR_OK, t_linear * 80])
    for ob in pl.spmd(token_tiles, name_hint="comb_sinkhorn", allow_early_resolve=True):
        scale2 = pl.read(hc_scale, [2])
        t0 = ob * COMB_T_TILE
        valid_rows = pl.min(COMB_T_TILE, t_dim - t0)
        c_parts = pl.load(parts_2d, [0, t0 * 80], [LINEAR_OK, T_TILE * 80], target_memory=pl.MemorySpace.Vec)
        c_wide = pl.reshape(pl.col_sum(c_parts), [T_TILE, 80])
        c_high = pl.slice(c_wide, [T_TILE, MIX_HC], [0, 0])
        c_low = pl.slice(c_wide, [T_TILE, MIX_HC], [0, MIX_HC])
        c_last = pl.slice(c_wide, [T_TILE, MIX_HC], [0, MIX_HC * 2])
        c_total = pl.add(c_high, pl.add(c_low, c_last))
        c_inv = pl.load(inv_rms, [t0, 0], [T_TILE, 1], target_memory=pl.MemorySpace.Vec)
        comb_inv = c_inv
        raw = pl.add(pl.slice(c_total, [COMB_T_TILE, 16], [0, HC_MULT * 2]), 0.0)
        base = pl.load(hc_base_2d, [0, HC_MULT * 2], [1, 16], target_memory=pl.MemorySpace.Vec)
        logits = pl.add(pl.mul(pl.row_expand_mul(raw, comb_inv), scale2), pl.col_expand(raw, base))
        logits = pl.reshape(logits, [1, COMB_T_TILE * 16])
        ids = pl.tile.arange(0, [1, COMB_T_TILE * 16], dtype=pl.INT32)
        tmp = pl.create_tile([1, COMB_T_TILE * 16], dtype=pl.INT32)
        idx_float = pl.cast(ids, pl.FP32)
        ri1_low = pl.cast(pl.cast(pl.mul(idx_float, 1.0), pl.INT32, mode="floor"), pl.FP32)
        ri1_high = pl.cast(pl.cast(pl.mul(idx_float, 0.5), pl.INT32, mode="floor"), pl.FP32)
        ri1_bit = pl.sub(ri1_low, pl.mul(ri1_high, 2.0))
        ri1 = pl.cast(pl.add(pl.sub(idx_float, pl.mul(ri1_bit, 2.0)), 1.0), pl.INT32)
        ri2_low = pl.cast(pl.cast(pl.mul(idx_float, 0.5), pl.INT32, mode="floor"), pl.FP32)
        ri2_high = pl.cast(pl.cast(pl.mul(idx_float, 0.25), pl.INT32, mode="floor"), pl.FP32)
        ri2_bit = pl.sub(ri2_low, pl.mul(ri2_high, 2.0))
        ri2 = pl.cast(pl.add(pl.sub(idx_float, pl.mul(ri2_bit, 4.0)), 2.0), pl.INT32)
        ci1_low = pl.cast(pl.cast(pl.mul(idx_float, 0.25), pl.INT32, mode="floor"), pl.FP32)
        ci1_high = pl.cast(pl.cast(pl.mul(idx_float, 0.125), pl.INT32, mode="floor"), pl.FP32)
        ci1_bit = pl.sub(ci1_low, pl.mul(ci1_high, 2.0))
        ci1 = pl.cast(pl.add(pl.sub(idx_float, pl.mul(ci1_bit, 8.0)), 4.0), pl.INT32)
        ci2_low = pl.cast(pl.cast(pl.mul(idx_float, 0.125), pl.INT32, mode="floor"), pl.FP32)
        ci2_high = pl.cast(pl.cast(pl.mul(idx_float, 0.0625), pl.INT32, mode="floor"), pl.FP32)
        ci2_bit = pl.sub(ci2_low, pl.mul(ci2_high, 2.0))
        ci2 = pl.cast(pl.add(pl.sub(idx_float, pl.mul(ci2_bit, 16.0)), 8.0), pl.INT32)
        max2 = pl.maximum(logits, pl.tile.gather(logits, ri1, tmp))
        max4 = pl.maximum(max2, pl.tile.gather(max2, ri2, tmp))
        exps = pl.exp(pl.sub(logits, max4))
        sum2 = pl.add(exps, pl.tile.gather(exps, ri1, tmp))
        sum4 = pl.add(sum2, pl.tile.gather(sum2, ri2, tmp))
        probs = pl.add(pl.div(exps, sum4), HC_EPS)
        col2 = pl.add(probs, pl.tile.gather(probs, ci1, tmp))
        col4 = pl.add(pl.add(col2, pl.tile.gather(col2, ci2, tmp)), HC_EPS)
        cur = pl.div(probs, col4)
        for iteration in pl.range(HC_SINKHORN_ITER - 1):
            row2 = pl.add(cur, pl.tile.gather(cur, ri1, tmp))
            row4 = pl.add(pl.add(row2, pl.tile.gather(row2, ri2, tmp)), HC_EPS)
            norm = pl.div(cur, row4)
            col2 = pl.add(norm, pl.tile.gather(norm, ci1, tmp))
            col4 = pl.add(pl.add(col2, pl.tile.gather(col2, ci2, tmp)), HC_EPS)
            cur = pl.div(norm, col4)
        valid = pl.set_validshape(pl.reshape(cur, [COMB_T_TILE, 16]), valid_rows, 16)
        pl.store(valid, [t0, 0], comb)

    # 每个 token 块按 D/D_SPMD 划分任务，对四路残差加权求和。
    for blk in pl.spmd(token_tiles * (D // D_SPMD), name_hint="pre_post_mix", allow_early_resolve=True):
        t0 = (blk // (D // D_SPMD)) * T_TILE
        d_base = (blk % (D // D_SPMD)) * D_SPMD
        valid_rows = pl.min(T_TILE, t_dim - t0)
        scale0 = pl.read(hc_scale, [0])
        scale1 = pl.read(hc_scale, [1])
        m_parts = pl.load(parts_2d, [0, t0 * 80], [LINEAR_OK, T_TILE * 80], target_memory=pl.MemorySpace.Vec)
        m_wide = pl.reshape(pl.col_sum(m_parts), [T_TILE, 80])
        m_high = pl.slice(m_wide, [T_TILE, MIX_HC], [0, 0])
        m_low = pl.slice(m_wide, [T_TILE, MIX_HC], [0, MIX_HC])
        m_last = pl.slice(m_wide, [T_TILE, MIX_HC], [0, MIX_HC * 2])
        m_total = pl.add(m_high, pl.add(m_low, m_last))
        m_inv = pl.load(inv_rms, [t0, 0], [T_TILE, 1], target_memory=pl.MemorySpace.Vec)
        inv_col = m_inv
        pre_base = pl.load(hc_base_2d, [0, 0], [1, HC_PAD], target_memory=pl.MemorySpace.Vec)
        pre_raw = pl.add(pl.slice(m_total, [T_TILE, HC_PAD], [0, 0]), 0.0)
        pre_scaled = pl.mul(pl.row_expand_mul(pre_raw, inv_col), scale0)
        pre_logits = pl.add(pre_scaled, pl.col_expand(pre_scaled, pre_base))
        pre_sig = pl.recip(pl.add(pl.exp(pl.neg(pre_logits)), 1.0))
        pre_val = pl.add(pre_sig, HC_EPS)

        post_inv = m_inv
        post_base_tile = pl.load(hc_base_2d, [0, HC_MULT], [1, HC_PAD], target_memory=pl.MemorySpace.Vec)
        m_total_t = pl.transpose(m_total, axis1=0, axis2=1)
        post_rows = pl.add(pl.slice(m_total_t, [HC_PAD, T_TILE], [HC_MULT, 0]), 0.0)
        post_raw_tile = pl.transpose(post_rows, axis1=0, axis2=1)
        post_scaled_tile = pl.mul(pl.row_expand_mul(post_raw_tile, post_inv), scale1)
        post_logits_tile = pl.add(post_scaled_tile, pl.col_expand(post_scaled_tile, post_base_tile))
        post_sig_tile = pl.recip(pl.add(pl.exp(pl.neg(post_logits_tile)), 1.0))
        post_result = pl.set_validshape(pl.mul(post_sig_tile, 2.0), valid_rows, HC_MULT)
        pl.store(post_result, [t0, 0], post)
        pre_tile_t = pl.transpose(pre_val, axis1=0, axis2=1)
        pre0 = pl.reshape(pre_tile_t[0:1, 0:T_TILE], [T_TILE, 1])
        pre1 = pl.reshape(pre_tile_t[1:2, 0:T_TILE], [T_TILE, 1])
        pre2 = pl.reshape(pre_tile_t[2:3, 0:T_TILE], [T_TILE, 1])
        pre3 = pl.reshape(pre_tile_t[3:4, 0:T_TILE], [T_TILE, 1])
        for db in pl.pipeline(D_SPMD // D_TILE, stage=2):
            d0 = d_base + db * D_TILE
            x0_bf16 = pl.load(x_flat, [t0, 0 * D + d0], [T_TILE, D_TILE], valid_shape=[valid_rows, D_TILE], target_memory=pl.MemorySpace.Vec)
            x0 = pl.cast(x0_bf16, pl.FP32)
            x1_bf16 = pl.load(x_flat, [t0, 1 * D + d0], [T_TILE, D_TILE], valid_shape=[valid_rows, D_TILE], target_memory=pl.MemorySpace.Vec)
            x1 = pl.cast(x1_bf16, pl.FP32)
            x2_bf16 = pl.load(x_flat, [t0, 2 * D + d0], [T_TILE, D_TILE], valid_shape=[valid_rows, D_TILE], target_memory=pl.MemorySpace.Vec)
            x2 = pl.cast(x2_bf16, pl.FP32)
            x3_bf16 = pl.load(x_flat, [t0, 3 * D + d0], [T_TILE, D_TILE], valid_shape=[valid_rows, D_TILE], target_memory=pl.MemorySpace.Vec)
            x3 = pl.cast(x3_bf16, pl.FP32)
            y0 = pl.row_expand_mul(x0, pre0)
            y1 = pl.row_expand_mul(x1, pre1)
            y2 = pl.row_expand_mul(x2, pre2)
            y3 = pl.row_expand_mul(x3, pre3)
            y_tile = pl.add(pl.add(y0, y1), pl.add(y2, y3))
            y_bf16 = pl.cast(y_tile, target_type=pl.BF16, mode="rint")
            y_valid = pl.set_validshape(y_bf16, valid_rows, D_TILE)
            pl.store(y_valid, [t0, d0], mixed)
    return mixed, post, comb_out
