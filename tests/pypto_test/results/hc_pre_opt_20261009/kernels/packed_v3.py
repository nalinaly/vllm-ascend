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
LINEAR_K_TILE = 256
D_TILE = 512
D_SPMD = 4096
LINEAR_OK = 4
LINEAR_K_PER_SPLIT = HC_DIM // LINEAR_OK
LINEAR_WORKERS = 24
WIDEN_COLS = 8192
WIDEN_WORKERS = 48


@pl.jit.inline
def hc_pre_gates_from_rms(
    x: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    hc_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_scale: pl.Tensor[[3], pl.FP32],
    hc_base: pl.Tensor[[MIX_HC], pl.FP32],
    pre_val_store: pl.Tensor[[HC_PAD_ROWS_DYN, HC_PAD], pl.FP32],
    post: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    comb: pl.Tensor[[T_DYN, HC_MULT * HC_MULT], pl.FP32],
    row_recip: pl.Scalar[pl.BOOL],
    inv_rms: pl.Tensor[[HC_PAD_ROWS_DYN, 1], pl.FP32],
):
    """根据本文件生成的 RMS 统计计算投影、门控和 Sinkhorn。"""
    t_dim = pl.tensor.dim(x, 0)
    token_tiles = (t_dim + T_TILE - 1) // T_TILE
    t_linear = ((t_dim + LINEAR_T_TILE - 1) // LINEAR_T_TILE) * LINEAR_T_TILE  # token 行数向上补齐至 16 行 Cube 块
    x_flat = pl.reshape(x, [t_dim, HC_DIM])
    hc_base_2d = pl.reshape(hc_base, [1, MIX_HC])  # 供 comb_sinkhorn 按组读取偏置

    # 投影沿 K 分成四段，各自写出部分和；valid_shape 将尾块无效行补零，无需展开输入。
    mixes_partials = pl.create_tensor([LINEAR_OK * t_linear, MIX_PAD], dtype=pl.FP32)
    linear_units = (t_linear // LINEAR_T_TILE) * LINEAR_OK
    linear_workers = pl.min(linear_units, LINEAR_WORKERS)
    for linear_worker in pl.spmd(linear_workers, name_hint="hc_pre_linear", allow_early_resolve=True):
        # HC权重沿用Native的ND视图及默认缓存策略；缓存策略的调整需单独实测。
        for task in pl.range(linear_worker, linear_units, linear_workers):
            t0 = (task // LINEAR_OK) * LINEAR_T_TILE
            linear_split = task % LINEAR_OK
            k_base = linear_split * LINEAR_K_PER_SPLIT
            t_rows = pl.min(LINEAR_T_TILE, t_dim - t0)  # 尾块无效行由 valid_shape 补零
            acc = pl.create_tensor([LINEAR_T_TILE, MIX_PAD], dtype=pl.FP32)
            for kb in pl.pipeline(0, LINEAR_K_PER_SPLIT // LINEAR_K_TILE, stage=2):
                k0 = k_base + kb * LINEAR_K_TILE
                x_linear_chunk = pl.slice(x_flat, [LINEAR_T_TILE, LINEAR_K_TILE], [t0, k0], valid_shape=[t_rows, LINEAR_K_TILE])
                w_chunk = pl.slice(hc_fn, [MIX_PAD, LINEAR_K_TILE], [0, k0], valid_shape=[MIX_HC, LINEAR_K_TILE])
                acc = pl.matmul_acc(acc, x_linear_chunk, w_chunk, b_trans=True, init_cond=(kb == 0))
            partial_row0 = linear_split * t_linear + t0
            mixes_partials[partial_row0 : partial_row0 + LINEAR_T_TILE, 0:MIX_PAD] = acc

    # 按 K 段递增次序合并部分和，保留原有浮点累加顺序。
    mixes_raw = pl.create_tensor([t_linear, MIX_PAD], dtype=pl.FP32)
    for linear_block in pl.spmd(t_linear // LINEAR_T_TILE, name_hint="hc_pre_linear_reduce", allow_early_resolve=True):
        linear_t0 = linear_block * LINEAR_T_TILE
        mixes_total = mixes_partials[linear_t0 : linear_t0 + LINEAR_T_TILE, 0:MIX_PAD]
        for linear_split in pl.range(1, LINEAR_OK):
            partial_t0 = linear_split * t_linear + linear_t0
            partial_tile = mixes_partials[partial_t0 : partial_t0 + LINEAR_T_TILE, 0:MIX_PAD]
            mixes_total = pl.add(mixes_total, partial_tile)
        mixes_raw[linear_t0 : linear_t0 + LINEAR_T_TILE, 0:MIX_PAD] = mixes_total

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

    # 将每个 4×4 矩阵打包为 16 个连续元素；通过行列置换并行处理四行，减少标量化归约。
    for ob in pl.spmd(token_tiles, name_hint="comb_sinkhorn", allow_early_resolve=True):
        scale2 = pl.read(hc_scale, [2])
        t0 = ob * COMB_T_TILE
        valid_rows = pl.min(COMB_T_TILE, t_dim - t0)
        comb_inv = pl.load(inv_rms, [t0, 0], [COMB_T_TILE, 1], target_memory=pl.MemorySpace.Vec)
        raw = pl.load(mixes_raw, [t0, HC_MULT * 2], [COMB_T_TILE, 16], target_memory=pl.MemorySpace.Vec)
        base = pl.load(hc_base_2d, [0, HC_MULT * 2], [1, 16], target_memory=pl.MemorySpace.Vec)
        logits = pl.add(pl.mul(pl.row_expand_mul(raw, comb_inv), scale2), pl.col_expand(raw, base))
        ids = pl.reshape(pl.tile.arange(0, [1, COMB_T_TILE * 16], dtype=pl.INT32), [COMB_T_TILE, 16])
        tmp = pl.create_tile([COMB_T_TILE, 16], dtype=pl.INT32)
        ids16 = pl.cast(ids, pl.INT16)
        xor_tmp = pl.create_tile([COMB_T_TILE, 16], dtype=pl.INT16)
        ri1 = pl.cast(pl.xors(ids16, 1, xor_tmp), pl.INT32)
        ri2 = pl.cast(pl.xors(ids16, 2, xor_tmp), pl.INT32)
        ci1 = pl.cast(pl.xors(ids16, 4, xor_tmp), pl.INT32)
        ci2 = pl.cast(pl.xors(ids16, 8, xor_tmp), pl.INT32)
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
        valid = pl.set_validshape(cur, valid_rows, 16)
        pl.store(valid, [t0, 0], comb)

    return pre_val_store


@pl.jit.inline
def hc_pre_gates(
    x: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    hc_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_scale: pl.Tensor[[3], pl.FP32],
    hc_base: pl.Tensor[[MIX_HC], pl.FP32],
    pre_val_store: pl.Tensor[[T_DYN, HC_PAD], pl.FP32],
    post: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    comb: pl.Tensor[[T_DYN, HC_MULT * HC_MULT], pl.FP32],
    row_recip: pl.Scalar[pl.BOOL],
):
    """保留原独立RMS任务及其归约次序。"""
    t_dim = pl.tensor.dim(x, 0)
    token_tiles = (t_dim + T_TILE - 1) // T_TILE
    t_linear = ((t_dim + LINEAR_T_TILE - 1) // LINEAR_T_TILE) * LINEAR_T_TILE
    x_flat = pl.reshape(x, [t_dim, HC_DIM])
    inv_rms = pl.create_tensor([t_linear, 1], dtype=pl.FP32)

    # 每个 token 块沿完整 K 轴累加平方和，生成 RMS 倒数。
    for t in pl.spmd(token_tiles, name_hint="hc_pre_rms", allow_early_resolve=True):
        t0 = t * T_TILE
        valid_rows = pl.min(T_TILE, t_dim - t0)
        sq_sum = pl.full([1, T_TILE], dtype=pl.FP32, value=0.0)
        for kb in pl.pipeline(HC_DIM // RMS_K_TILE, stage=4):
            k0 = kb * RMS_K_TILE
            if valid_rows == T_TILE:
                x_chunk_full = x_flat[t0:t0 + T_TILE, k0:k0 + RMS_K_TILE]
                x_sq_full = pl.mul(x_chunk_full, x_chunk_full)
                x_sq_row_full = pl.reshape(pl.row_sum(x_sq_full), [1, T_TILE])
                sq_sum = pl.add(sq_sum, x_sq_row_full)
            else:
                x_chunk_tail = pl.slice(x_flat, [T_TILE, RMS_K_TILE], [t0, k0], valid_shape=[valid_rows, RMS_K_TILE])
                x_sq_tail = pl.mul(x_chunk_tail, x_chunk_tail)
                x_sq_row_tail = pl.reshape(pl.row_sum(x_sq_tail), [1, T_TILE])
                sq_sum = pl.add(sq_sum, x_sq_row_tail)
        sq_mean = pl.add(pl.mul(sq_sum, HC_DIM_INV), NORM_EPS)
        inv = pl.reshape(pl.rsqrt(sq_mean, high_precision=True), [T_TILE, 1])
        inv_rms[t0:t0 + T_TILE, 0:1] = inv

    return hc_pre_gates_from_rms(
        x, hc_fn, hc_scale, hc_base, pre_val_store, post, comb, row_recip, inv_rms,
    )


@pl.jit.inline
def _hc_pre_fp32(
    x: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
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
    hc_pre_gates(x, hc_fn, hc_scale, hc_base, pre_val_store, post, comb, False)
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
            y0 = pl.row_expand_mul(x0, pre0)
            y1 = pl.row_expand_mul(x1, pre1)
            y2 = pl.row_expand_mul(x2, pre2)
            y3 = pl.row_expand_mul(x3, pre3)
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
    x_fp32 = pl.create_tensor([tokens, HC_MULT, D], dtype=pl.FP32)
    x_flat = pl.reshape(x, [tokens, HC_DIM])
    x_fp32_flat = pl.reshape(x_fp32, [tokens, HC_DIM])
    # 单行分块避免 BF16 cast 后尾行 valid_shape 丢失，转换包含在算子计时内。
    for worker in pl.spmd(pl.min(tokens, WIDEN_WORKERS), name_hint="hc_pre_widen", allow_early_resolve=True):
        for row in pl.range(worker, tokens, pl.min(tokens, WIDEN_WORKERS)):
            for column in pl.pipeline(0, HC_DIM, WIDEN_COLS, stage=2):
                source = x_flat[row : row + 1, column : column + WIDEN_COLS]
                x_fp32_flat[row : row + 1, column : column + WIDEN_COLS] = pl.cast(source, pl.FP32)
    comb_flat = pl.reshape(comb, [tokens, HC_MULT * HC_MULT])
    mixed = _hc_pre_fp32(x_fp32, hc_fn, hc_scale, hc_base, mixed, post, comb_flat)
    return mixed, post, comb
