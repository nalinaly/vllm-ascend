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
RMS_K_TILE = 512
LINEAR_K_TILE = 256
D_TILE = 512
D_SPMD = 4096
LINEAR_OK = 16
LINEAR_K_PER_SPLIT = 1024
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
    mixes_partials = pl.create_tensor([LINEAR_OK * t_linear, MIX_PAD], dtype=pl.FP32)
    rms_partials = pl.create_tensor([LINEAR_OK * t_linear, 1], dtype=pl.FP32)
    linear_units = (t_linear // LINEAR_T_TILE) * LINEAR_OK
    linear_workers = pl.min(linear_units, LINEAR_WORKERS)
    for worker in pl.spmd(linear_workers, name_hint="hc_cv_block", allow_early_resolve=True, optimizations=[pl.cross_core_slot(slot_num=1)]):
        for task in pl.range(worker, linear_units, linear_workers):
            t0 = (task // LINEAR_OK) * LINEAR_T_TILE
            split = task % LINEAR_OK
            kbase = split * LINEAR_K_PER_SPLIT
            kcols = pl.min(LINEAR_K_PER_SPLIT, HC_DIM - kbase)
            partial_row = split * t_linear + t0
            for lane in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):
                lane_row = t0 + lane * (LINEAR_T_TILE // 2)
                lane_rows = pl.max(0, pl.min(LINEAR_T_TILE // 2, t_dim - lane_row))
                whole = pl.full([LINEAR_T_TILE // 2, LINEAR_K_PER_SPLIT], dtype=pl.FP32, value=0.0)
                # 两行转换暂存与 128 KiB 完整半块同时驻留，显式填充尾行后写入。
                zero_chunk = pl.full([2, LINEAR_K_PER_SPLIT], dtype=pl.FP32, value=0.0)
                for subrow in pl.range(0, LINEAR_T_TILE // 2, 2):
                    subrows = pl.max(0, pl.min(2, lane_rows - subrow))
                    source = pl.slice(x_flat, [2, LINEAR_K_PER_SPLIT], [lane_row + subrow, kbase], valid_shape=[subrows, LINEAR_K_PER_SPLIT])
                    fp32 = pl.cast(source, pl.FP32)
                    clean = pl.fillpad(pl.set_validshape(fp32, subrows, LINEAR_K_PER_SPLIT), pad_value=pl.PadValue.zero)
                    clean_full = pl.set_validshape(clean, 2, LINEAR_K_PER_SPLIT)
                    padded = pl.add(zero_chunk, clean_full)
                    whole = pl.assemble(whole, padded, [subrow, 0])
                gathered = pl.aic_gather(whole)
                # 交给 Cube 后再分块累加平方和，降低大块 FP32 和临时平方值同时存活的 UB 需求。
                local_sum = pl.full([1, LINEAR_T_TILE // 2], dtype=pl.FP32, value=0.0)
                for rk in pl.range(LINEAR_K_PER_SPLIT // 512):
                    rcol = rk * 512
                    rvalid = pl.max(0, pl.min(512, HC_DIM - kbase - rcol))
                    xr = pl.slice(x_flat, [LINEAR_T_TILE // 2, 512], [lane_row, kbase + rcol], valid_shape=[lane_rows, rvalid])
                    xr32 = pl.cast(xr, pl.FP32)
                    xr_clean = pl.fillpad(pl.set_validshape(xr32, lane_rows, rvalid), pad_value=pl.PadValue.zero)
                    xr_full = pl.set_validshape(xr_clean, LINEAR_T_TILE // 2, 512)
                    rms_sums = pl.reshape(pl.row_sum(pl.mul(xr_full, xr_full)), [1, LINEAR_T_TILE // 2])
                    local_sum = pl.add(local_sum, rms_sums)
                rp = partial_row + lane * (LINEAR_T_TILE // 2)
                rms_partials[rp:rp + LINEAR_T_TILE // 2, 0:1] = pl.reshape(local_sum, [LINEAR_T_TILE // 2, 1])
            weight = pl.slice(hc_fn, [MIX_PAD, LINEAR_K_PER_SPLIT], [0, kbase], valid_shape=[MIX_HC, kcols], pad_value=pl.PadValue.zero)
            product = pl.matmul(gathered, weight, b_trans=True)
            mixes_partials[partial_row:partial_row + LINEAR_T_TILE, 0:MIX_PAD] = product

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
        comb_inv = pl.load(inv_rms, [t0, 0], [COMB_T_TILE, 1], target_memory=pl.MemorySpace.Vec)
        raw = pl.load(mixes_raw, [t0, HC_MULT * 2], [COMB_T_TILE, 16], target_memory=pl.MemorySpace.Vec)
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
