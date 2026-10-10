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
LINEAR_T_TILE = 64
COMB_T_TILE = 8
D_TILE = 512
D_SPMD = 4096
LINEAR_OK = 8
LINEAR_K_PER_SPLIT = 2048
LINEAR_K_TILE = 512
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
    post_flat = pl.reshape(post, [1, tokens * HC_MULT])
    hc_base_2d = pl.reshape(hc_base, [1, MIX_HC])
    hc_scale_2d = pl.reshape(hc_scale, [1, 3])
    weight_parts = pl.create_tensor([MIX_HC * 3, HC_DIM], dtype=pl.BF16)
    mixes_partials = pl.create_tensor([LINEAR_OK * t_linear, 80], dtype=pl.FP32)
    rms_partials = pl.create_tensor([LINEAR_OK * t_linear, LINEAR_T_TILE], dtype=pl.FP32)
    linear_units = (t_linear // LINEAR_T_TILE) * LINEAR_OK
    clusters = pl.system.available_cluster_count()
    linear_workers = clusters
    parts_2d = pl.reshape(mixes_partials, [LINEAR_OK, t_linear * 80])
    rms_2d = pl.reshape(rms_partials, [LINEAR_OK, t_linear * LINEAR_T_TILE])
    for worker in pl.spmd(clusters, name_hint="hc_single_spmd", sync_start=True):
        # AIV 在本次调用内拆分完整 FP32 权重，同时 AIC 计算残差平方和。
        for lane in pl.split_aiv(2, mode=pl.SplitMode.NONE):
            for wr in pl.range(worker, MIX_HC, clusters):
                wc = lane * 8192
                w = hc_fn[wr:wr + 1, wc:wc + 8192]
                high = pl.cast(w, pl.BF16, mode="rint")
                residual = pl.sub(w, pl.cast(high, pl.FP32))
                low = pl.cast(residual, pl.BF16, mode="rint")
                last = pl.cast(pl.sub(residual, pl.cast(low, pl.FP32)), pl.BF16, mode="rint")
                weight_parts[wr:wr + 1, wc:wc + 8192] = high
                weight_parts[MIX_HC + wr:MIX_HC + wr + 1, wc:wc + 8192] = low
                weight_parts[MIX_HC * 2 + wr:MIX_HC * 2 + wr + 1, wc:wc + 8192] = last
        for task in pl.range(worker, linear_units, clusters):
            t0 = (task // LINEAR_OK) * LINEAR_T_TILE
            kbase = (task % LINEAR_OK) * LINEAR_K_PER_SPLIT
            partial_row = (task % LINEAR_OK) * t_linear + t0
            gram_rows = pl.slice(x_flat, [LINEAR_T_TILE, LINEAR_K_PER_SPLIT], [t0, kbase], valid_shape=[pl.min(LINEAR_T_TILE, t_dim - t0), LINEAR_K_PER_SPLIT], pad_value=pl.PadValue.zero)
            gram_value = pl.matmul(gram_rows, gram_rows, b_trans=True, out_dtype=pl.FP32)
            rms_partials[partial_row:partial_row + LINEAR_T_TILE, 0:LINEAR_T_TILE] = gram_value
        pl.system.cacheinvalid()
        pl.system.fence()
        pl.system.syncall(core_type=pl.KernelType.MIX)
        pl.system.cacheinvalid()
        # BF16 三段并成 72 列投影，完整转换和两道屏障都处于 incore 计时范围。
        for task in pl.range(worker, linear_units, clusters):
            t0 = (task // LINEAR_OK) * LINEAR_T_TILE
            kbase = (task % LINEAR_OK) * LINEAR_K_PER_SPLIT
            partial_row = (task % LINEAR_OK) * t_linear + t0
            t_rows = pl.min(LINEAR_T_TILE, t_dim - t0)
            x_first = pl.slice(x_flat, [LINEAR_T_TILE, LINEAR_K_TILE], [t0, kbase], valid_shape=[t_rows, LINEAR_K_TILE])
            w_first = pl.slice(weight_parts, [80, LINEAR_K_TILE], [0, kbase], valid_shape=[MIX_HC * 3, LINEAR_K_TILE], pad_value=pl.PadValue.zero)
            acc = pl.matmul(x_first, w_first, b_trans=True, out_dtype=pl.FP32)
            for kb in pl.pipeline(1, LINEAR_K_PER_SPLIT // LINEAR_K_TILE, stage=2):
                k0 = kbase + kb * LINEAR_K_TILE
                x_chunk = pl.slice(x_flat, [LINEAR_T_TILE, LINEAR_K_TILE], [t0, k0], valid_shape=[t_rows, LINEAR_K_TILE])
                w_chunk = pl.slice(weight_parts, [80, LINEAR_K_TILE], [0, k0], valid_shape=[MIX_HC * 3, LINEAR_K_TILE], pad_value=pl.PadValue.zero)
                acc = pl.matmul_acc(acc, x_chunk, w_chunk, b_trans=True)
            mixes_partials[partial_row:partial_row + LINEAR_T_TILE, 0:80] = acc

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
                    c_parts = pl.load(parts_2d, [0, t0 * 80], [LINEAR_OK, T_TILE * 80], target_memory=pl.MemorySpace.Vec)
                    c_wide = pl.reshape(pl.col_sum(c_parts), [T_TILE, 80])
                    c_high = pl.slice(c_wide, [T_TILE, MIX_HC], [0, 0])
                    c_low = pl.slice(c_wide, [T_TILE, MIX_HC], [0, MIX_HC])
                    c_last = pl.slice(c_wide, [T_TILE, MIX_HC], [0, MIX_HC * 2])
                    c_total = pl.add(c_high, pl.add(c_low, c_last))
                    c_gram = pl.load(rms_2d, [0, t0 * LINEAR_T_TILE], [LINEAR_OK, T_TILE * LINEAR_T_TILE], target_memory=pl.MemorySpace.Vec)
                    c_gram_sum = pl.col_sum(c_gram)
                    c_ids = pl.tile.arange(0, [1, T_TILE], dtype=pl.INT32)
                    c_diag = pl.add(pl.mul(c_ids, LINEAR_T_TILE + 1), pl.cast(t0 % LINEAR_T_TILE, pl.INT32))
                    c_tmp = pl.create_tile([1, T_TILE], dtype=pl.INT32)
                    c_sq_sum = pl.tile.gather(c_gram_sum, c_diag, c_tmp)
                    c_rms_tmp = pl.create_tile([1, T_TILE], dtype=pl.FP32)
                    c_inv = pl.reshape(pl.tile.rsqrt(pl.add(pl.mul(c_sq_sum, HC_DIM_INV), NORM_EPS), c_rms_tmp), [T_TILE, 1])
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
                    m_parts = pl.load(parts_2d, [0, t0 * 80], [LINEAR_OK, T_TILE * 80], target_memory=pl.MemorySpace.Vec)
                    m_wide = pl.reshape(pl.col_sum(m_parts), [T_TILE, 80])
                    m_high = pl.slice(m_wide, [T_TILE, MIX_HC], [0, 0])
                    m_low = pl.slice(m_wide, [T_TILE, MIX_HC], [0, MIX_HC])
                    m_last = pl.slice(m_wide, [T_TILE, MIX_HC], [0, MIX_HC * 2])
                    m_total = pl.add(m_high, pl.add(m_low, m_last))
                    m_gram = pl.load(rms_2d, [0, t0 * LINEAR_T_TILE], [LINEAR_OK, T_TILE * LINEAR_T_TILE], target_memory=pl.MemorySpace.Vec)
                    m_gram_sum = pl.col_sum(m_gram)
                    m_ids = pl.tile.arange(0, [1, T_TILE], dtype=pl.INT32)
                    m_diag = pl.add(pl.mul(m_ids, LINEAR_T_TILE + 1), pl.cast(t0 % LINEAR_T_TILE, pl.INT32))
                    m_tmp = pl.create_tile([1, T_TILE], dtype=pl.INT32)
                    m_sq_sum = pl.tile.gather(m_gram_sum, m_diag, m_tmp)
                    m_rms_tmp = pl.create_tile([1, T_TILE], dtype=pl.FP32)
                    m_inv = pl.reshape(pl.tile.rsqrt(pl.add(pl.mul(m_sq_sum, HC_DIM_INV), NORM_EPS), m_rms_tmp), [T_TILE, 1])
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
                    # 展平后一次提取每行后四列，连续写回 post，避免 16B 行转置。
                    post_ids_float = pl.cast(pl.tile.arange(0, [1, T_TILE * HC_MULT], dtype=pl.INT32), pl.FP32)
                    post_rows = pl.cast(pl.cast(pl.mul(post_ids_float, 1.0 / HC_MULT), pl.INT32, mode="floor"), pl.FP32)
                    post_ids = pl.cast(pl.add(pl.add(post_ids_float, pl.mul(post_rows, HC_MULT)), HC_MULT), pl.INT32)
                    post_tmp = pl.create_tile([1, T_TILE * HC_MULT], dtype=pl.INT32)
                    post_values = pl.tile.gather(pl.reshape(gate_sigmoid, [1, T_TILE * HC_PAD]), post_ids, post_tmp)
                    post_result = pl.set_validshape(pl.mul(post_values, 2.0), 1, valid_rows * HC_MULT)
                    pl.store(post_result, [0, t0 * HC_MULT], post_flat)
                    pre0 = pl.reshape(pl.add(pl.slice(gate_transposed, [1, T_TILE], [0, 0]), HC_EPS), [T_TILE, 1])
                    pre1 = pl.reshape(pl.add(pl.slice(gate_transposed, [1, T_TILE], [1, 0]), HC_EPS), [T_TILE, 1])
                    pre2 = pl.reshape(pl.add(pl.slice(gate_transposed, [1, T_TILE], [2, 0]), HC_EPS), [T_TILE, 1])
                    pre3 = pl.reshape(pl.add(pl.slice(gate_transposed, [1, T_TILE], [3, 0]), HC_EPS), [T_TILE, 1])
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
