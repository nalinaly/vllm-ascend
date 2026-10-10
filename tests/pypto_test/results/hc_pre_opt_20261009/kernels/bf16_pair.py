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
LINEAR_T_TILE = 16
COMB_T_TILE = 8
PRE_POST_WORKERS = 16
RMS_K_TILE = 512
LINEAR_K_TILE = 512
D_TILE = 512
D_SPMD = 2048
LINEAR_OK = 2
LINEAR_K_PER_SPLIT = HC_DIM // LINEAR_OK
LINEAR_WORKERS = 24
WIDEN_COLS = 8192
WIDEN_WORKERS = 48


@pl.jit.inline
def hc_pre_gates_from_rms(
    x: pl.Tensor[[T_DYN, HC_MULT, D], pl.BF16],
    hc_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_scale: pl.Tensor[[3], pl.FP32],
    hc_base: pl.Tensor[[MIX_HC], pl.FP32],
    pre_val_store: pl.Tensor[[HC_PAD_ROWS_DYN, HC_PAD], pl.FP32],
    post: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    comb: pl.Tensor[[T_DYN, HC_MULT * HC_MULT], pl.FP32],
    row_recip: pl.Scalar[pl.BOOL],
):
    """根据本文件生成的 RMS 统计计算投影、门控和 Sinkhorn。"""
    t_dim = pl.tensor.dim(x, 0)
    token_tiles = (t_dim + T_TILE - 1) // T_TILE
    t_linear = ((t_dim + LINEAR_T_TILE - 1) // LINEAR_T_TILE) * LINEAR_T_TILE  # token 行数向上补齐至 16 行 Cube 块
    x_flat = pl.reshape(x, [t_dim, HC_DIM])
    hc_base_2d = pl.reshape(hc_base, [1, MIX_HC])  # 供 comb_sinkhorn 按组读取偏置

    # 权重保持 FP32 入参，在设备内分成 BF16 高低两部分，避免完整残差加宽中间缓冲。
    weight_hi = pl.create_tensor([MIX_HC, HC_DIM], dtype=pl.BF16)
    weight_lo = pl.create_tensor([MIX_HC, HC_DIM], dtype=pl.BF16)
    for part in pl.spmd(48, name_hint="hc_weight_split", allow_early_resolve=True):
        wr = part // 2
        wc = (part % 2) * 8192
        w = hc_fn[wr:wr + 1, wc:wc + 8192]
        high = pl.cast(w, pl.BF16, mode="rint")
        low = pl.cast(pl.sub(w, pl.cast(high, pl.FP32)), pl.BF16, mode="rint")
        weight_hi[wr:wr + 1, wc:wc + 8192] = high
        weight_lo[wr:wr + 1, wc:wc + 8192] = low

    mixes_partials_hi = pl.create_tensor([LINEAR_OK * t_linear, MIX_PAD], dtype=pl.FP32)
    mixes_partials_lo = pl.create_tensor([LINEAR_OK * t_linear, MIX_PAD], dtype=pl.FP32)
    gram_partials = pl.create_tensor([LINEAR_OK * t_linear, LINEAR_T_TILE], dtype=pl.FP32)
    inv_rms = pl.create_tensor([t_linear, 1], dtype=pl.FP32)
    linear_units = (t_linear // LINEAR_T_TILE) * LINEAR_OK
    linear_workers = pl.min(linear_units, LINEAR_WORKERS)
    for worker in pl.spmd(linear_workers, name_hint="hc_pre_linear", allow_early_resolve=True):
        for task in pl.range(worker, linear_units, linear_workers):
            t0 = (task // LINEAR_OK) * LINEAR_T_TILE
            split = task % LINEAR_OK
            rows = pl.min(LINEAR_T_TILE, t_dim - t0)
            hi_acc = pl.create_tensor([LINEAR_T_TILE, MIX_PAD], dtype=pl.FP32)
            lo_acc = pl.create_tensor([LINEAR_T_TILE, MIX_PAD], dtype=pl.FP32)
            gram_acc = pl.create_tensor([LINEAR_T_TILE, LINEAR_T_TILE], dtype=pl.FP32)
            for kb in pl.pipeline(LINEAR_K_PER_SPLIT // LINEAR_K_TILE, stage=2):
                k0 = split * LINEAR_K_PER_SPLIT + kb * LINEAR_K_TILE
                xc = pl.slice(x_flat, [LINEAR_T_TILE, LINEAR_K_TILE], [t0, k0], valid_shape=[rows, LINEAR_K_TILE])
                wh = pl.slice(weight_hi, [MIX_PAD, LINEAR_K_TILE], [0, k0], valid_shape=[MIX_HC, LINEAR_K_TILE])
                wl = pl.slice(weight_lo, [MIX_PAD, LINEAR_K_TILE], [0, k0], valid_shape=[MIX_HC, LINEAR_K_TILE])
                hi_acc = pl.matmul_acc(hi_acc, xc, wh, b_trans=True, init_cond=(kb == 0))
                lo_acc = pl.matmul_acc(lo_acc, xc, wl, b_trans=True, init_cond=(kb == 0))
                gram_acc = pl.matmul_acc(gram_acc, xc, xc, b_trans=True, init_cond=(kb == 0))
            pr = split * t_linear + t0
            mixes_partials_hi[pr:pr + LINEAR_T_TILE, 0:MIX_PAD] = hi_acc
            mixes_partials_lo[pr:pr + LINEAR_T_TILE, 0:MIX_PAD] = lo_acc
            gram_partials[pr:pr + LINEAR_T_TILE, 0:LINEAR_T_TILE] = gram_acc

    mixes_raw = pl.create_tensor([t_linear, MIX_PAD], dtype=pl.FP32)
    for block in pl.spmd(t_linear // LINEAR_T_TILE, name_hint="hc_pre_linear_reduce", allow_early_resolve=True):
        t0r = block * LINEAR_T_TILE
        high_total = pl.load(mixes_partials_hi, [t0r, 0], [LINEAR_T_TILE, MIX_PAD], target_memory=pl.MemorySpace.Vec)
        low_total = pl.load(mixes_partials_lo, [t0r, 0], [LINEAR_T_TILE, MIX_PAD], target_memory=pl.MemorySpace.Vec)
        gram_total = pl.load(gram_partials, [t0r, 0], [LINEAR_T_TILE, LINEAR_T_TILE], target_memory=pl.MemorySpace.Vec)
        for split_r in pl.range(1, LINEAR_OK):
            prr = split_r * t_linear + t0r
            high_total = pl.add(high_total, pl.load(mixes_partials_hi, [prr, 0], [LINEAR_T_TILE, MIX_PAD], target_memory=pl.MemorySpace.Vec))
            low_total = pl.add(low_total, pl.load(mixes_partials_lo, [prr, 0], [LINEAR_T_TILE, MIX_PAD], target_memory=pl.MemorySpace.Vec))
            gram_total = pl.add(gram_total, pl.load(gram_partials, [prr, 0], [LINEAR_T_TILE, LINEAR_T_TILE], target_memory=pl.MemorySpace.Vec))
        pl.store(pl.add(high_total, low_total), [t0r, 0], mixes_raw)
        indices = pl.mul(pl.tile.arange(0, [1, LINEAR_T_TILE], dtype=pl.INT32), LINEAR_T_TILE + 1)
        gather_tmp = pl.create_tile([1, LINEAR_T_TILE], dtype=pl.INT32)
        sq_sum = pl.tile.gather(gram_total, indices, gather_tmp)
        rms = pl.recip(pl.sqrt(pl.add(pl.mul(sq_sum, HC_DIM_INV), NORM_EPS)))
        pl.store(pl.reshape(rms, [LINEAR_T_TILE, 1]), [t0r, 0], inv_rms)

    # pre 门写入混合所需的中间缓冲，post 门写入输出；comb 门由下方 Sinkhorn 任务处理。
    # 两门均按 8 列计算，输出仅 4 列有效，以满足 PTOAS 对 Vector tile 的 32B 对齐要求。
    # 最后一个不足 8 行的 token 块使用固定大小的暂存缓冲，避免越界写回。
    post_tail_store = pl.create_tensor([T_TILE, HC_PAD], dtype=pl.FP32)
    for ob_worker in pl.spmd(pl.min(token_tiles, PRE_POST_WORKERS), name_hint="split_pre_post", allow_early_resolve=True):
        scale0 = pl.read(hc_scale, [0])
        scale1 = pl.read(hc_scale, [1])
        for ob in pl.range(ob_worker, token_tiles, pl.min(token_tiles, PRE_POST_WORKERS)):
            t0 = ob * T_TILE
            valid_rows = pl.min(T_TILE, t_dim - t0)
            inv_col = inv_rms[t0:t0 + T_TILE, 0:1]

            pre_base = pl.reshape(hc_base[0:HC_PAD], [1, HC_PAD])
            pre_scaled = pl.mul(pl.row_expand_mul(mixes_raw[t0:t0 + T_TILE, 0:HC_PAD], inv_col), scale0)
            pre_logits = pl.add(pre_scaled, pl.col_expand(pre_scaled, pre_base))
            pre_sig = pl.recip(pl.add(pl.exp(pl.neg(pre_logits)), 1.0))
            pre_val = pl.add(pre_sig, HC_EPS)
            pre_val_store[t0:t0 + T_TILE, 0:HC_PAD] = pre_val

            post_base = pl.reshape(hc_base[HC_MULT:HC_MULT + HC_PAD], [1, HC_PAD])
            post_scaled = pl.mul(pl.row_expand_mul(mixes_raw[t0:t0 + T_TILE, HC_MULT:HC_MULT + HC_PAD], inv_col), scale1)
            post_logits = pl.add(post_scaled, pl.col_expand(post_scaled, post_base))
            post_sig = pl.recip(pl.add(pl.exp(pl.neg(post_logits)), 1.0))
            post_pad = pl.mul(post_sig, 2.0)
            if valid_rows == T_TILE:
                post[t0:t0 + T_TILE, 0:HC_MULT] = pl.slice(post_pad, [T_TILE, HC_PAD], [0, 0], valid_shape=[T_TILE, HC_MULT])
            else:
                post_tail_store[0:T_TILE, 0:HC_PAD] = post_pad
                post_tile = pl.load(post_tail_store, [0, 0], [T_TILE, HC_PAD], valid_shape=[valid_rows, HC_MULT], target_memory=pl.MemorySpace.Vec)
                pl.store(post_tile, [t0, 0], post)

    # comb 门读取投影的第 8/12/16/20 列，各行先做 softmax，
    # 再从列归一化开始完成 20 次 Sinkhorn，写入 comb。
    comb_tail_store = pl.create_tensor([COMB_T_TILE, HC_PAD * HC_MULT], dtype=pl.FP32)
    for ob in pl.spmd(token_tiles, name_hint="comb_sinkhorn", allow_early_resolve=True):
        scale2 = pl.read(hc_scale, [2])
        t0 = ob * COMB_T_TILE
        valid_rows = pl.min(COMB_T_TILE, t_dim - t0)
        inv_col_t = pl.load(inv_rms, [t0, 0], [COMB_T_TILE, 1], valid_shape=[valid_rows, 1], target_memory=pl.MemorySpace.Vec)
        comb_off = HC_MULT * 2
        mix_g0 = pl.load(mixes_raw, [t0, comb_off + 0 * HC_MULT], [COMB_T_TILE, HC_PAD], valid_shape=[valid_rows, HC_MULT], target_memory=pl.MemorySpace.Vec)
        mix_g1 = pl.load(mixes_raw, [t0, comb_off + 1 * HC_MULT], [COMB_T_TILE, HC_PAD], valid_shape=[valid_rows, HC_MULT], target_memory=pl.MemorySpace.Vec)
        mix_g2 = pl.load(mixes_raw, [t0, comb_off + 2 * HC_MULT], [COMB_T_TILE, HC_PAD], valid_shape=[valid_rows, HC_MULT], target_memory=pl.MemorySpace.Vec)
        mix_g3 = pl.load(mixes_raw, [t0, comb_off + 3 * HC_MULT], [COMB_T_TILE, HC_PAD], valid_shape=[valid_rows, HC_MULT], target_memory=pl.MemorySpace.Vec)
        cb0 = pl.load(hc_base_2d, [0, comb_off + 0 * HC_MULT], [1, HC_PAD], valid_shape=[1, HC_MULT], target_memory=pl.MemorySpace.Vec)
        cb1 = pl.load(hc_base_2d, [0, comb_off + 1 * HC_MULT], [1, HC_PAD], valid_shape=[1, HC_MULT], target_memory=pl.MemorySpace.Vec)
        cb2 = pl.load(hc_base_2d, [0, comb_off + 2 * HC_MULT], [1, HC_PAD], valid_shape=[1, HC_MULT], target_memory=pl.MemorySpace.Vec)
        cb3 = pl.load(hc_base_2d, [0, comb_off + 3 * HC_MULT], [1, HC_PAD], valid_shape=[1, HC_MULT], target_memory=pl.MemorySpace.Vec)
        row0 = pl.add(pl.mul(pl.row_expand_mul(mix_g0, inv_col_t), scale2), pl.col_expand(mix_g0, cb0))
        row1 = pl.add(pl.mul(pl.row_expand_mul(mix_g1, inv_col_t), scale2), pl.col_expand(mix_g1, cb1))
        row2 = pl.add(pl.mul(pl.row_expand_mul(mix_g2, inv_col_t), scale2), pl.col_expand(mix_g2, cb2))
        row3 = pl.add(pl.mul(pl.row_expand_mul(mix_g3, inv_col_t), scale2), pl.col_expand(mix_g3, cb3))
        # 四路按 HC 行合并到一个连续 tile，行归一化一次完成；列归约使用 UB 子视图。
        p01 = pl.concat(pl.transpose(row0, axis1=0, axis2=1), pl.transpose(row1, axis1=0, axis2=1))
        p23 = pl.concat(pl.transpose(row2, axis1=0, axis2=1), pl.transpose(row3, axis1=0, axis2=1))
        packed = pl.transpose(pl.concat(p01, p23), axis1=0, axis2=1)
        packed_valid = pl.set_validshape(packed, 4 * COMB_T_TILE, HC_MULT)
        logits = pl.fillpad(packed_valid, pad_value=pl.PadValue.min)
        tmp = pl.create_tile([4 * COMB_T_TILE, HC_PAD], dtype=pl.FP32)
        maxima = pl.row_max(logits, tmp)
        exponentials = pl.exp(pl.row_expand_sub(logits, maxima))
        row_sums = pl.row_sum(exponentials, tmp)
        probs = pl.add(pl.row_expand_div(exponentials, row_sums), HC_EPS)
        zeros = pl.fillpad(pl.set_validshape(probs, 4 * COMB_T_TILE, HC_MULT), pad_value=pl.PadValue.zero)
        cur = pl.set_validshape(zeros, 4 * COMB_T_TILE, HC_PAD)
        for iteration in pl.range(HC_SINKHORN_ITER):
            if iteration > 0:
                sums = pl.add(pl.row_sum(cur, tmp), HC_EPS)
                normalized = pl.row_expand_div(cur, sums)
            else:
                normalized = cur
            n0 = pl.slice(normalized, [COMB_T_TILE, HC_PAD], [0, 0])
            n1 = pl.slice(normalized, [COMB_T_TILE, HC_PAD], [COMB_T_TILE, 0])
            n2 = pl.slice(normalized, [COMB_T_TILE, HC_PAD], [2 * COMB_T_TILE, 0])
            n3 = pl.slice(normalized, [COMB_T_TILE, HC_PAD], [3 * COMB_T_TILE, 0])
            col_sums = pl.add(pl.add(pl.add(n0, n1), pl.add(n2, n3)), HC_EPS)
            normalized_flat = pl.reshape(normalized, [4, COMB_T_TILE * HC_PAD])
            col_sums_flat = pl.reshape(col_sums, [1, COMB_T_TILE * HC_PAD])
            cur = pl.reshape(pl.col_expand_div(normalized_flat, col_sums_flat), [4 * COMB_T_TILE, HC_PAD])
        o0 = pl.slice(cur, [COMB_T_TILE, HC_PAD], [0, 0], valid_shape=[valid_rows, HC_MULT])
        o1 = pl.slice(cur, [COMB_T_TILE, HC_PAD], [COMB_T_TILE, 0], valid_shape=[valid_rows, HC_MULT])
        o2 = pl.slice(cur, [COMB_T_TILE, HC_PAD], [2 * COMB_T_TILE, 0], valid_shape=[valid_rows, HC_MULT])
        o3 = pl.slice(cur, [COMB_T_TILE, HC_PAD], [3 * COMB_T_TILE, 0], valid_shape=[valid_rows, HC_MULT])
        pl.store(o0, [t0, 0], comb)
        pl.store(o1, [t0, HC_MULT], comb)
        pl.store(o2, [t0, HC_MULT * 2], comb)
        pl.store(o3, [t0, HC_MULT * 3], comb)

    return pre_val_store


@pl.jit.inline
def _hc_pre_fp32(
    x: pl.Tensor[[T_DYN, HC_MULT, D], pl.BF16],
    hc_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_scale: pl.Tensor[[3], pl.FP32],
    hc_base: pl.Tensor[[MIX_HC], pl.FP32],
    x_mixed: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
    post: pl.Out[pl.Tensor[[T_DYN, HC_MULT], pl.FP32]],
    comb: pl.Out[pl.Tensor[[T_DYN, HC_MULT * HC_MULT], pl.FP32]],
):
    """计算 HC 门控并将四路残差混合为 BF16 输出。"""
    x.bind_dynamic(0, T_DYN)
    x_mixed.bind_dynamic(0, T_DYN)
    post.bind_dynamic(0, T_DYN)
    comb.bind_dynamic(0, T_DYN)
    t_dim = pl.tensor.dim(x, 0)
    token_tiles = (t_dim + T_TILE - 1) // T_TILE
    t_linear = ((t_dim + LINEAR_T_TILE - 1) // LINEAR_T_TILE) * LINEAR_T_TILE
    pre_val_store = pl.create_tensor([t_linear, HC_PAD], dtype=pl.FP32)
    hc_pre_gates_from_rms(x, hc_fn, hc_scale, hc_base, pre_val_store, post, comb, False)
    x_flat = pl.reshape(x, [t_dim, HC_DIM])

    # 每个 token 块按 D/D_SPMD 划分任务，对四路残差加权求和。
    x_mixed_tail_store = pl.create_tensor([T_TILE, D], dtype=pl.BF16)
    for blk in pl.spmd(token_tiles * (D // D_SPMD), name_hint="mix_x", allow_early_resolve=True):
        t0 = (blk // (D // D_SPMD)) * T_TILE
        d_base = (blk % (D // D_SPMD)) * D_SPMD
        valid_rows = pl.min(T_TILE, t_dim - t0)
        pre_tile_t = pl.transpose(pre_val_store[t0:t0 + T_TILE, 0:HC_PAD], axis1=0, axis2=1)
        pre0 = pl.reshape(pre_tile_t[0:1, 0:T_TILE], [T_TILE, 1])
        pre1 = pl.reshape(pre_tile_t[1:2, 0:T_TILE], [T_TILE, 1])
        pre2 = pl.reshape(pre_tile_t[2:3, 0:T_TILE], [T_TILE, 1])
        pre3 = pl.reshape(pre_tile_t[3:4, 0:T_TILE], [T_TILE, 1])
        for db in pl.pipeline(D_SPMD // D_TILE, stage=2):
            d0 = d_base + db * D_TILE
            x0 = pl.slice(x_flat, [T_TILE, D_TILE], [t0, 0 * D + d0], valid_shape=[valid_rows, D_TILE])
            x1 = pl.slice(x_flat, [T_TILE, D_TILE], [t0, 1 * D + d0], valid_shape=[valid_rows, D_TILE])
            x2 = pl.slice(x_flat, [T_TILE, D_TILE], [t0, 2 * D + d0], valid_shape=[valid_rows, D_TILE])
            x3 = pl.slice(x_flat, [T_TILE, D_TILE], [t0, 3 * D + d0], valid_shape=[valid_rows, D_TILE])
            x0_fp32 = pl.fillpad(pl.set_validshape(pl.cast(x0, pl.FP32), valid_rows, D_TILE), pad_value=pl.PadValue.zero)
            y0 = pl.row_expand_mul(x0_fp32, pre0)
            x1_fp32 = pl.fillpad(pl.set_validshape(pl.cast(x1, pl.FP32), valid_rows, D_TILE), pad_value=pl.PadValue.zero)
            y1 = pl.row_expand_mul(x1_fp32, pre1)
            x2_fp32 = pl.fillpad(pl.set_validshape(pl.cast(x2, pl.FP32), valid_rows, D_TILE), pad_value=pl.PadValue.zero)
            y2 = pl.row_expand_mul(x2_fp32, pre2)
            x3_fp32 = pl.fillpad(pl.set_validshape(pl.cast(x3, pl.FP32), valid_rows, D_TILE), pad_value=pl.PadValue.zero)
            y3 = pl.row_expand_mul(x3_fp32, pre3)
            y_tile = pl.add(pl.add(y0, y1), pl.add(y2, y3))
            y_bf16 = pl.cast(y_tile, target_type=pl.BF16, mode="rint")
            if valid_rows == T_TILE:
                x_mixed[t0:t0 + T_TILE, d0:d0 + D_TILE] = y_bf16
            else:
                x_mixed_tail_store[0:T_TILE, d0:d0 + D_TILE] = y_bf16
                y_out = pl.load(x_mixed_tail_store, [0, d0], [T_TILE, D_TILE], valid_shape=[valid_rows, D_TILE], target_memory=pl.MemorySpace.Vec)
                pl.store(y_out, [t0, d0], x_mixed)
    return x_mixed


@pl.jit(auto_scope=False)
def hc_pre(
    x: pl.Tensor[[TOKENS, HC_MULT, D], pl.BF16],
    hc_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_scale: pl.Tensor[[3], pl.FP32],
    hc_base: pl.Tensor[[MIX_HC], pl.FP32],
    mixed: pl.Out[pl.Tensor[[TOKENS, D], pl.BF16]],
    post: pl.Out[pl.Tensor[[TOKENS, HC_MULT], pl.FP32]],
    comb: pl.Out[pl.Tensor[[TOKENS, HC_MULT, HC_MULT], pl.FP32]],
):
    """一次 PTO 调用写入三份 Native 同形状输出；所有输入权重均为 device tensor。"""
    tokens = pl.tensor.dim(x, 0)
    comb_flat = pl.reshape(comb, [tokens, HC_MULT * HC_MULT])
    mixed = _hc_pre_fp32(x, hc_fn, hc_scale, hc_base, mixed, post, comb_flat)
    return mixed, post, comb
