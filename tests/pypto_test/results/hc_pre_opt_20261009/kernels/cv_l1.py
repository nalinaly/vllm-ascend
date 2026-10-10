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
    t_dim = tokens
    token_tiles = (tokens + T_TILE - 1) // T_TILE
    t_linear = ((tokens + LINEAR_T_TILE - 1) // LINEAR_T_TILE) * LINEAR_T_TILE
    comb = pl.reshape(comb_out, [tokens, HC_MULT * HC_MULT])
    hc_base_2d = pl.reshape(hc_base, [1, MIX_HC])
    inv_rms = pl.create_tensor([t_linear, 1], dtype=pl.FP32)
    # 投影沿 K 分成四段，各自写出部分和；valid_shape 将尾块无效行补零，无需展开输入。
    mixes_partials = pl.create_tensor([LINEAR_OK * t_linear, MIX_PAD], dtype=pl.FP32)
    rms_partials = pl.create_tensor([LINEAR_OK * t_linear, 1], dtype=pl.FP32)
    inv_rms = pl.create_tensor([t_linear, 1], dtype=pl.FP32)
    linear_units = (t_linear // LINEAR_T_TILE) * LINEAR_OK
    linear_workers = pl.min(linear_units, LINEAR_WORKERS)
    for linear_worker in pl.spmd(linear_workers, name_hint="hc_pre_linear", allow_early_resolve=True):
        # HC权重沿用Native的ND视图及默认缓存策略；缓存策略的调整需单独实测。
        for task in pl.range(linear_worker, linear_units, linear_workers):
            t0 = (task // LINEAR_OK) * LINEAR_T_TILE
            linear_split = task % LINEAR_OK
            k_base = linear_split * LINEAR_K_PER_SPLIT
            t_rows = pl.min(LINEAR_T_TILE, t_dim - t0)  # 尾块无效行由 valid_shape 补零
            x_l1 = pl.create_l1([LINEAR_T_TILE, LINEAR_K_PER_SPLIT], pl.FP32)
            sq_sum = pl.full([1, LINEAR_T_TILE], dtype=pl.FP32, value=0.0)
            for vk in pl.range(LINEAR_K_PER_SPLIT // 512):
                col = vk * 512
                source = pl.slice(x_flat, [LINEAR_T_TILE, 512], [t0, k_base + col], valid_shape=[t_rows, 512])
                fp32 = pl.cast(source, pl.FP32)
                clean = pl.fillpad(pl.set_validshape(fp32, t_rows, 512), pad_value=pl.PadValue.zero)
                padded = pl.set_validshape(clean, LINEAR_T_TILE, 512)
                x_l1 = pl.assemble(x_l1, padded, [0, col])
                sq_row = pl.reshape(pl.row_sum(pl.mul(padded, padded)), [1, LINEAR_T_TILE])
                sq_sum = pl.add(sq_sum, sq_row)
            acc = pl.create_tensor([LINEAR_T_TILE, MIX_PAD], dtype=pl.FP32)
            for kb in pl.pipeline(0, LINEAR_K_PER_SPLIT // LINEAR_K_TILE, stage=2):
                col2 = kb * LINEAR_K_TILE
                x_chunk = pl.slice(x_l1, [LINEAR_T_TILE, LINEAR_K_TILE], [0, col2])
                w_chunk = pl.slice(hc_fn, [MIX_PAD, LINEAR_K_TILE], [0, k_base + col2], valid_shape=[MIX_HC, LINEAR_K_TILE])
                acc = pl.matmul_acc(acc, x_chunk, w_chunk, b_trans=True, init_cond=(kb == 0))
            partial_row0 = linear_split * t_linear + t0
            mixes_partials[partial_row0 : partial_row0 + LINEAR_T_TILE, 0:MIX_PAD] = acc
            rms_partials[partial_row0 : partial_row0 + LINEAR_T_TILE, 0:1] = pl.reshape(sq_sum, [LINEAR_T_TILE, 1])

    # 按 K 段递增次序合并部分和，保留原有浮点累加顺序。
    mixes_raw = pl.create_tensor([t_linear, MIX_PAD], dtype=pl.FP32)
    for linear_block in pl.spmd(t_linear // LINEAR_T_TILE, name_hint="hc_pre_linear_reduce", allow_early_resolve=True):
        linear_t0 = linear_block * LINEAR_T_TILE
        mixes_total = mixes_partials[linear_t0 : linear_t0 + LINEAR_T_TILE, 0:MIX_PAD]
        sq_total = rms_partials[linear_t0 : linear_t0 + LINEAR_T_TILE, 0:1]
        for linear_split in pl.range(1, LINEAR_OK):
            partial_t0 = linear_split * t_linear + linear_t0
            partial_tile = mixes_partials[partial_t0 : partial_t0 + LINEAR_T_TILE, 0:MIX_PAD]
            mixes_total = pl.add(mixes_total, partial_tile)
            sq_total = pl.add(sq_total, rms_partials[partial_t0 : partial_t0 + LINEAR_T_TILE, 0:1])
        mixes_raw[linear_t0 : linear_t0 + LINEAR_T_TILE, 0:MIX_PAD] = mixes_total
        inv_rms[linear_t0 : linear_t0 + LINEAR_T_TILE, 0:1] = pl.rsqrt(pl.add(pl.mul(sq_total, HC_DIM_INV), NORM_EPS), high_precision=True)

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

    # 每个 token 块按 D/D_SPMD 划分任务，对四路残差加权求和。
    x_mixed_tail_store = pl.create_tensor([T_TILE, D], dtype=pl.BF16)
    for blk in pl.spmd(token_tiles * (D // D_SPMD), name_hint="pre_post_mix", allow_early_resolve=True):
        t0 = (blk // (D // D_SPMD)) * T_TILE
        d_base = (blk % (D // D_SPMD)) * D_SPMD
        valid_rows = pl.min(T_TILE, t_dim - t0)
        scale0 = pl.read(hc_scale, [0])
        scale1 = pl.read(hc_scale, [1])
        inv_col = inv_rms[t0:t0 + T_TILE, 0:1]
        pre_base = pl.reshape(hc_base[0:HC_PAD], [1, HC_PAD])
        pre_scaled = pl.mul(pl.row_expand_mul(mixes_raw[t0:t0 + T_TILE, 0:HC_PAD], inv_col), scale0)
        pre_logits = pl.add(pre_scaled, pl.col_expand(pre_scaled, pre_base))
        pre_sig = pl.recip(pl.add(pl.exp(pl.neg(pre_logits)), 1.0))
        pre_val = pl.add(pre_sig, HC_EPS)

        post_inv = pl.load(inv_rms, [t0, 0], [T_TILE, 1], target_memory=pl.MemorySpace.Vec)
        post_base_tile = pl.load(hc_base_2d, [0, HC_MULT], [1, HC_PAD], target_memory=pl.MemorySpace.Vec)
        post_raw_tile = pl.load(mixes_raw, [t0, HC_MULT], [T_TILE, HC_PAD], target_memory=pl.MemorySpace.Vec)
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
            x0_bf16 = pl.slice(x_flat, [T_TILE, D_TILE], [t0, 0 * D + d0], valid_shape=[valid_rows, D_TILE])
            x0 = pl.cast(x0_bf16, pl.FP32)
            x1_bf16 = pl.slice(x_flat, [T_TILE, D_TILE], [t0, 1 * D + d0], valid_shape=[valid_rows, D_TILE])
            x1 = pl.cast(x1_bf16, pl.FP32)
            x2_bf16 = pl.slice(x_flat, [T_TILE, D_TILE], [t0, 2 * D + d0], valid_shape=[valid_rows, D_TILE])
            x2 = pl.cast(x2_bf16, pl.FP32)
            x3_bf16 = pl.slice(x_flat, [T_TILE, D_TILE], [t0, 3 * D + d0], valid_shape=[valid_rows, D_TILE])
            x3 = pl.cast(x3_bf16, pl.FP32)
            y0 = pl.row_expand_mul(x0, pre0)
            y1 = pl.row_expand_mul(x1, pre1)
            y2 = pl.row_expand_mul(x2, pre2)
            y3 = pl.row_expand_mul(x3, pre3)
            y_tile = pl.add(pl.add(y0, y1), pl.add(y2, y3))
            y_bf16 = pl.cast(y_tile, target_type=pl.BF16, mode="rint")
            if valid_rows == T_TILE:
                mixed[t0:t0 + T_TILE, d0:d0 + D_TILE] = y_bf16
            else:
                x_mixed_tail_store[0:T_TILE, d0:d0 + D_TILE] = y_bf16
                y_out = pl.load(x_mixed_tail_store, [0, d0], [T_TILE, D_TILE], valid_shape=[valid_rows, D_TILE], target_memory=pl.MemorySpace.Vec)
                pl.store(y_out, [t0, d0], mixed)
    return mixed, post, comb_out
