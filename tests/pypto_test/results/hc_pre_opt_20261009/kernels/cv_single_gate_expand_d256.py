# SPDX-License-Identifier: Apache-2.0
"""DeepSeek V4 Flash 独立 HC_pre，完整实现位于本文件。

从本仓库 CSA 版本完整搬入后，在本文件优化分块与任务划分。
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

# M32、K1536 分段投影；两个 AIV 分别转换半块，一次交给 Cube。
MIX_PAD = 32  # 24 列投影补齐至 Cube N 轴和 Vector 行的 32 列。
HC_PAD = 8  # 4 列有效值按 32B 对齐。
T_TILE = 8
LINEAR_T_TILE = 32
COMB_T_TILE = 8
D_TILE = 256
D_SPMD = 4096
LINEAR_OK = 11
LINEAR_K_PER_SPLIT = 1536
LINEAR_WORKERS = 24


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
    hc_scale_2d = pl.reshape(hc_scale, [1, 3])
    mixes_partials = pl.create_tensor([LINEAR_OK * t_linear, MIX_PAD], dtype=pl.FP32)
    rms_partials = pl.create_tensor([LINEAR_OK * t_linear, 1], dtype=pl.FP32)
    linear_units = (t_linear // LINEAR_T_TILE) * LINEAR_OK
    clusters = pl.system.available_cluster_count()
    linear_workers = clusters
    parts_2d = pl.reshape(mixes_partials, [LINEAR_OK, t_linear * MIX_PAD])
    rms_2d = pl.reshape(rms_partials, [LINEAR_OK, t_linear])
    for worker in pl.spmd(clusters, name_hint="hc_single_spmd", sync_start=True, optimizations=[pl.cross_core_slot(slot_num=1)]):
        for task in pl.range(worker, linear_units, linear_workers):
            t0 = (task // LINEAR_OK) * LINEAR_T_TILE
            split = task % LINEAR_OK
            kbase = split * LINEAR_K_PER_SPLIT
            kcols = pl.min(LINEAR_K_PER_SPLIT, HC_DIM - kbase)
            partial_row = split * t_linear + t0
            for lane in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):
                lane_row = t0 + lane * (LINEAR_T_TILE // 2)
                lane_rows = pl.max(0, pl.min(LINEAR_T_TILE // 2, t_dim - lane_row))
                source = pl.slice(x_flat, [LINEAR_T_TILE // 2, LINEAR_K_PER_SPLIT], [lane_row, kbase], valid_shape=[lane_rows, kcols])
                fp32 = pl.cast(source, pl.FP32)
                clean = pl.fillpad(pl.set_validshape(fp32, lane_rows, kcols), pad_value=pl.PadValue.zero)
                whole = pl.set_validshape(clean, LINEAR_T_TILE // 2, LINEAR_K_PER_SPLIT)
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

        # 发布全部 GM 部分和；所有 AIC/AIV 无条件参与同一硬件屏障。
        pl.system.cacheinvalid()
        pl.system.fence()
        pl.system.syncall(core_type=pl.KernelType.MIX)
        pl.system.cacheinvalid()
        for tail_lane in pl.split_aiv(2, mode=pl.SplitMode.NONE):
            if tail_lane == 0:
                for ob in pl.range(worker, token_tiles, clusters):
                    scale2 = pl.read(hc_scale, [2])
                    t0 = ob * COMB_T_TILE
                    valid_rows = pl.min(COMB_T_TILE, t_dim - t0)
                    c_parts = pl.load(parts_2d, [0, t0 * MIX_PAD], [LINEAR_OK, T_TILE * MIX_PAD], target_memory=pl.MemorySpace.Vec)
                    c_total = pl.reshape(pl.col_sum(c_parts), [T_TILE, MIX_PAD])
                    c_sq = pl.load(rms_2d, [0, t0], [LINEAR_OK, T_TILE], target_memory=pl.MemorySpace.Vec)
                    c_rms_tmp = pl.create_tile([1, T_TILE], dtype=pl.FP32)
                    c_inv = pl.reshape(pl.tile.rsqrt(pl.add(pl.mul(pl.col_sum(c_sq), HC_DIM_INV), NORM_EPS), c_rms_tmp), [T_TILE, 1])
                    comb_inv = c_inv
                    raw = pl.add(pl.slice(c_total, [COMB_T_TILE, 16], [0, HC_MULT * 2]), 0.0)
                    base = pl.load(hc_base_2d, [0, HC_MULT * 2], [1, 16], target_memory=pl.MemorySpace.Vec)
                    logits = pl.add(pl.mul(pl.row_expand_mul(raw, comb_inv), scale2), pl.col_expand(raw, base))
                    # 置换索引只在每个 token 的 16 个矩阵元素内交换；展平减少 gather 的逐行开销。
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

            else:
                for blk in pl.range(worker, token_tiles * (D // D_SPMD), clusters):
                    t0 = (blk // (D // D_SPMD)) * T_TILE
                    d_base = (blk % (D // D_SPMD)) * D_SPMD
                    valid_rows = pl.min(T_TILE, t_dim - t0)
                    m_parts = pl.load(parts_2d, [0, t0 * MIX_PAD], [LINEAR_OK, T_TILE * MIX_PAD], target_memory=pl.MemorySpace.Vec)
                    m_total = pl.reshape(pl.col_sum(m_parts), [T_TILE, MIX_PAD])
                    m_sq = pl.load(rms_2d, [0, t0], [LINEAR_OK, T_TILE], target_memory=pl.MemorySpace.Vec)
                    m_rms_tmp = pl.create_tile([1, T_TILE], dtype=pl.FP32)
                    m_inv = pl.reshape(pl.tile.rsqrt(pl.add(pl.mul(pl.col_sum(m_sq), HC_DIM_INV), NORM_EPS), m_rms_tmp), [T_TILE, 1])
                    # pre/post 共用八列归一化、偏置和 sigmoid，保留各自 scale。
                    gate_base = pl.load(hc_base_2d, [0, 0], [1, HC_PAD], target_memory=pl.MemorySpace.Vec)
                    gate_scales = pl.load(hc_scale_2d, [0, 0], [1, HC_PAD], valid_shape=[1, 3], target_memory=pl.MemorySpace.Vec)
                    gate_ids = pl.cast(pl.tile.arange(0, [1, HC_PAD], dtype=pl.INT32), pl.FP32)
                    gate_scale_ids = pl.cast(pl.mul(gate_ids, 0.25), pl.INT32, mode="floor")
                    gate_tmp = pl.create_tile([1, HC_PAD], dtype=pl.INT32)
                    gate_scale_vec = pl.tile.gather(gate_scales, gate_scale_ids, gate_tmp)
                    gate_raw = pl.add(pl.slice(m_total, [T_TILE, HC_PAD], [0, 0]), 0.0)
                    gate_norm = pl.row_expand_mul(gate_raw, m_inv)
                    gate_scaled = pl.col_expand_mul(gate_norm, gate_scale_vec)
                    gate_logits = pl.add(gate_scaled, pl.col_expand(gate_scaled, gate_base))
                    gate_sigmoid = pl.recip(pl.add(pl.exp(pl.neg(gate_logits)), 1.0))
                    gate_transposed = pl.transpose(gate_sigmoid, axis1=0, axis2=1)
                    post_half = pl.add(pl.slice(gate_transposed, [HC_MULT, T_TILE], [HC_MULT, 0]), 0.0)
                    post_padded = pl.fillpad_expand(post_half, [HC_PAD, T_TILE], pad_value=pl.PadValue.zero)
                    post_plain = pl.add(pl.tile.full([HC_PAD, T_TILE], dtype=pl.FP32, value=0.0), post_padded)
                    post_sigmoid = pl.transpose(post_plain, axis1=0, axis2=1)
                    post_result = pl.set_validshape(pl.mul(post_sigmoid, 2.0), valid_rows, HC_MULT)
                    pl.store(post_result, [t0, 0], post)
                    # 一次展开每个 token 的系数，D 循环内直接逐元素相乘。
                    coeff_ids = pl.tile.arange(0, [1, T_TILE * D_TILE], dtype=pl.INT32)
                    coeff_rows = pl.cast(pl.mul(pl.cast(coeff_ids, pl.FP32), 1.0 / D_TILE), pl.INT32, mode="floor")
                    coeff_tmp = pl.create_tile([1, T_TILE * D_TILE], dtype=pl.INT32)
                    pre0_row = pl.add(pl.slice(gate_transposed, [1, T_TILE], [0, 0]), HC_EPS)
                    pre0 = pl.reshape(pl.tile.gather(pre0_row, coeff_rows, coeff_tmp), [T_TILE, D_TILE])
                    pre1_row = pl.add(pl.slice(gate_transposed, [1, T_TILE], [1, 0]), HC_EPS)
                    pre1 = pl.reshape(pl.tile.gather(pre1_row, coeff_rows, coeff_tmp), [T_TILE, D_TILE])
                    pre2_row = pl.add(pl.slice(gate_transposed, [1, T_TILE], [2, 0]), HC_EPS)
                    pre2 = pl.reshape(pl.tile.gather(pre2_row, coeff_rows, coeff_tmp), [T_TILE, D_TILE])
                    pre3_row = pl.add(pl.slice(gate_transposed, [1, T_TILE], [3, 0]), HC_EPS)
                    pre3 = pl.reshape(pl.tile.gather(pre3_row, coeff_rows, coeff_tmp), [T_TILE, D_TILE])
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
                        y0 = pl.mul(x0, pre0)
                        y1 = pl.mul(x1, pre1)
                        y2 = pl.mul(x2, pre2)
                        y3 = pl.mul(x3, pre3)
                        y_tile = pl.add(pl.add(y0, y1), pl.add(y2, y3))
                        y_bf16 = pl.cast(y_tile, target_type=pl.BF16, mode="rint")
                        y_valid = pl.set_validshape(y_bf16, valid_rows, D_TILE)
                        pl.store(y_valid, [t0, d0], mixed)
    return mixed, post, comb_out
