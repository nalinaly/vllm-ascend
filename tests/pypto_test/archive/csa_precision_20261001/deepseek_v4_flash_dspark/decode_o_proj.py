# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""CSA integration dependency adapted from pypto-lib 205255b4/decode_o_proj.py."""

import pypto.language as pl

# 共用数值中性的MatMul实现：WO-A保留K256次序，WO-B保留INT32精确累加。
from ..deepseek_v4_flash_dspark_perf.decode_o_proj import (
    PROJ_A_LARGE_N_TILE,
    PROJ_B_MEDIUM_T_TILE,
    PROJ_B_SMALL_T_TILE,
    proj_a_mm,
    proj_b_mm,
)
from .config import (
    DECODE_TOKENS,
    INT8_AMAX_EPS,
    INT8_SCALE_MAX,
)
from .config import (
    FLASH as M,
)
from .config import (
    TP as TP_SIZE,
)
from .hc_post import HC_MULT, hc_post_block
from .nz_mode import (
    QUANT_WEIGHT_LAYOUT,
    WO_A_WEIGHT_LAYOUT,
)

D = M.hidden_size

H = M.num_attention_heads

HEAD_DIM = M.head_dim

O_LORA = M.o_lora_rank

O_GROUPS = M.o_groups

HEADS_PER_GROUP = H // O_GROUPS

O_GROUP_IN = HEADS_PER_GROUP * HEAD_DIM

LOCAL_T = DECODE_TOKENS // TP_SIZE

LOCAL_O_GROUPS = O_GROUPS // TP_SIZE

LOCAL_O_WIDTH = LOCAL_O_GROUPS * O_LORA

T_DYN = pl.dynamic("T_DYN")

TOKEN_TILE = 16

COMM_ROW_TILE = 8

ATTENTION_PUBLISH_WORKERS = 48

O_RS_REDUCE_WORKERS = 48  # 128 row blocks; 8 left 40 of 48 AIV idle

O_RS_PUBLISH_WORKERS = 32

O_RS_DEQUANT_WORKERS = 12  # per owner; 4 owners -> one AIV wave

O_RS_PUT_T_TILE = 8

O_RS_D_TILE = 4096

LOCAL_T_PAD = (LOCAL_T + TOKEN_TILE - 1) // TOKEN_TILE * TOKEN_TILE

T_PAD = LOCAL_T_PAD

GROUP_T_PAD = TP_SIZE * LOCAL_T_PAD

ATTENTION_WINDOW_ROWS = LOCAL_O_GROUPS * GROUP_T_PAD

O_WINDOW_ROWS = TP_SIZE * LOCAL_T_PAD

A_K_TILE = 256

PROJ_A_MM_N_TILE = 128

PROJ_A_ROW_TILE = 128  # proj_a token block; one block covers T_PAD, 8 tasks/group

# proj_b 的 K 分块 256->512、N 分块 256->128：b_trans 下连续字节沿 k，K 翻倍把
# ND 的连续段从 256B 拉到 512B，L0C 也从满格的 128KiB 降到 64KiB 留出双缓冲。
# 同样是 INT8xINT8->INT32，整数累加精确，分块不影响与 Native 的一致性。
B_K_TILE = 512

PROJ_B_MM_T_TILE = 128

PROJ_B_MM_N_TILE = 128

PROJ_B_ACT_N_TILE = D

QUANT_TOKEN_TILE = 8
QUANT_TASK_T_TILE = 32  # 每个量化任务负责的 token 跨度，沿用上游 e68e091 的调优值

PROJ_B_D_TILE = 512  # proj_b_mm D chunk per task; coarser starves the 24 AIC cores

PROJ_B_ACT_T_TILE = 1

PROJ_B_ACT_TASK_T_TILE = 4  # proj_b_act token block per task

O_A_T_TILE = 128

O_A_K_TILE = 256

O_A_N_TILE = 128

QUANT_T_TILE = 8

O_A_QUANT_WORKERS = 6  # per owner-group; 4 x 2 x 6 -> one AIV wave

O_B_T_TILE = 128

O_B_K_TILE = 256

O_B_N_TILE = 256

O_B_D_TILE = 512

ACT_T_TILE = 16

ACT_N_TILE = 512

FIXTURE_LOCAL_T = max(1, LOCAL_T - 1)

FIXTURE_OUTPUT_SENTINEL = -7.0

if DECODE_TOKENS % TP_SIZE != 0:
    raise ValueError(f"decode tokens {DECODE_TOKENS} must be divisible by TP size {TP_SIZE}")

if O_GROUPS % TP_SIZE != 0:
    raise ValueError(f"output groups {O_GROUPS} must be divisible by TP size {TP_SIZE}")

if O_GROUP_IN % O_A_K_TILE != 0:
    raise ValueError(f"O-A input {O_GROUP_IN} must be divisible by K tile {O_A_K_TILE}")

if O_LORA % O_A_N_TILE != 0:
    raise ValueError(f"O-A output {O_LORA} must be divisible by N tile {O_A_N_TILE}")

if O_LORA % O_B_K_TILE != 0:
    raise ValueError(f"O-B group width {O_LORA} must be divisible by K tile {O_B_K_TILE}")

if D % O_B_D_TILE != 0 or O_B_D_TILE % O_B_N_TILE != 0:
    raise ValueError("O-B output tiles must divide the hidden dimension")

if D % ACT_N_TILE != 0:
    raise ValueError(f"O-B activation tile {ACT_N_TILE} must divide hidden size {D}")

if D % O_RS_D_TILE != 0:
    raise ValueError(f"O-B ReduceScatter tile {O_RS_D_TILE} must divide hidden size {D}")

if O_RS_REDUCE_WORKERS % TP_SIZE != 0:
    raise ValueError(f"TP size {TP_SIZE} must divide O-B ReduceScatter workers {O_RS_REDUCE_WORKERS}")

if O_RS_PUBLISH_WORKERS % TP_SIZE != 0:
    raise ValueError(f"TP size {TP_SIZE} must divide O-B publish workers {O_RS_PUBLISH_WORKERS}")

if GROUP_T_PAD % O_B_T_TILE != 0:
    raise ValueError(f"O-B token tile {O_B_T_TILE} must divide token capacity {GROUP_T_PAD}")

if T_PAD % PROJ_B_MM_T_TILE != 0:
    raise ValueError(f"proj_b_mm token tile {PROJ_B_MM_T_TILE} must divide token capacity {T_PAD}")


@pl.jit.inline
def _decode_o_proj_tp1_tiled(
    o_packed: pl.Tensor[[O_GROUPS * T_PAD, O_GROUP_IN], pl.BF16],
    wo_a: pl.Tensor[[O_GROUPS, O_GROUP_IN, O_LORA], pl.BF16, WO_A_WEIGHT_LAYOUT],
    wo_b: pl.Tensor[[O_GROUPS * O_LORA, D], pl.INT8, QUANT_WEIGHT_LAYOUT],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    residual: pl.Tensor[[T_DYN, HC_MULT, D], pl.BF16],
    post: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    comb: pl.Tensor[[T_DYN, HC_MULT * HC_MULT], pl.FP32],
    x_out: pl.Out[pl.Tensor[[T_DYN, HC_MULT, D], pl.BF16]],
    heads_dep: pl.Scalar[pl.TASK_ID],
    ROW_TILE: pl.constexpr,
    A_COL_TILE: pl.constexpr,
    POST_SYNC: pl.constexpr,
):
    """Project local-token, full-group attention heads into BF16 hidden rows."""
    t_dim = pl.tensor.dim(x_out, 0)
    residual_flat = pl.reshape(residual, [t_dim, HC_MULT * D])
    y_flat = pl.reshape(x_out, [t_dim, HC_MULT * D])
    act_row_step = PROJ_B_ACT_TASK_T_TILE
    if POST_SYNC:
        act_row_step = 2
    act_t_blks = (t_dim + act_row_step - 1) // act_row_step
    proj_a_rows = (t_dim + PROJ_A_ROW_TILE - 1) // PROJ_A_ROW_TILE
    proj_b_t_rows = (t_dim + ROW_TILE - 1) // ROW_TILE
    proj_b_padded_rows = proj_b_t_rows * ROW_TILE

    # Native WO-A materializes BF16 across all eight groups, then WO-B
    # quantizes the concatenated 8192 columns with one scale per token.
    # Keep the reference's grouped matmul tiles and INT32 group partials.
    o_r_pad = pl.create_tensor([T_PAD, O_GROUPS * O_LORA], dtype=pl.FP32)
    o_r_i8_pad = pl.create_tensor([T_PAD, O_GROUPS * O_LORA], dtype=pl.INT8)
    act_scale_dq = pl.create_tensor([1, T_PAD], dtype=pl.FP32)
    act_scale_q = pl.create_tensor([1, T_PAD], dtype=pl.FP32)
    # Per-group INT32 partials: proj_b_mm writes group g's contribution to output
    # channel n at partials[:, g*D + n]. No atomic-add -> no zero-seed.
    partials = pl.create_tensor([T_PAD, O_GROUPS * D], dtype=pl.INT32)
    proj_a_tids = pl.array.create(O_GROUPS, pl.TASK_ID)
    proj_b_tids = pl.array.create(O_GROUPS, pl.TASK_ID)

    with pl.manual_scope():
        for g in pl.parallel(O_GROUPS):
            row_base_o = g * T_PAD
            out_col_g = g * O_LORA

            o_r_pad, pa_tid = proj_a_mm(
                o_packed, wo_a, o_r_pad, g, row_base_o, out_col_g,
                t_dim, proj_a_rows, heads_dep, A_COL_TILE, False,
            )

            proj_a_tids[g] = pa_tid

    # 逐 token 块的标度彼此独立，原先整段在一个 CORE_GROUP 任务里串行遍历全部 token，
    # 泳道实测 count=1、Exec 82.50us、Tail OH 50.56us。改成按 token 块分的 SPMD。
    #
    # 注意这里只改调度，不合并进 quant。上游 e68e091 把标度算进了 quant 并按 group
    # 各算各的 amax（act_scale_dq[g:g+1]），而我们的契约是每 token 跨全部 O_GROUPS
    # 取 amax（act_scale_dq[0:1]）并多做一次 BF16 round-trip，那是为了对齐 Native A3
    # 的 dynamic_quant。照搬上游会改变量化语义，破坏与 Native 的逐 token 一致。
    scale_blocks = (t_dim + QUANT_TASK_T_TILE - 1) // QUANT_TASK_T_TILE
    with pl.spmd(
        scale_blocks,
        name_hint="oproj_token_scale",
        deps=[proj_a_tids[i] for i in range(O_GROUPS)],
        allow_early_resolve=True,
    ) as scale_tid:
        scale_start = pl.tile.get_block_idx() * QUANT_TASK_T_TILE
        for qt in pl.pipeline(scale_start, pl.min(scale_start + QUANT_TASK_T_TILE, t_dim),
                              QUANT_TOKEN_TILE, stage=2):
            token_amax = pl.full([1, QUANT_TOKEN_TILE], dtype=pl.FP32, value=INT8_AMAX_EPS)
            for scale_group in pl.range(O_GROUPS):
                scale_col = scale_group * O_LORA
                projected = o_r_pad[qt : qt + QUANT_TOKEN_TILE, scale_col : scale_col + O_LORA]
                projected = pl.cast(pl.cast(projected, pl.BF16, mode="rint"), pl.FP32)
                group_amax = pl.reshape(pl.row_max(pl.abs(projected)), [1, QUANT_TOKEN_TILE])
                token_amax = pl.maximum(token_amax, group_amax)
            act_scale_dq[0:1, qt : qt + QUANT_TOKEN_TILE] = pl.mul(token_amax, 1.0 / INT8_SCALE_MAX)
            # Native A3 dynamic_quant computes these independently: 127/amax
            # for quantization, amax*(1/127) for dequantization. Reciprocating
            # the rounded dequant scale changes INT8 half-way boundaries.
            quant_numerator = pl.full([1, QUANT_TOKEN_TILE], dtype=pl.FP32, value=INT8_SCALE_MAX)
            act_scale_q[0:1, qt : qt + QUANT_TOKEN_TILE] = pl.div(quant_numerator, token_amax)

    with pl.manual_scope():
        for g in pl.parallel(O_GROUPS):
            col_g = g * O_LORA
            # 每个 token 块独立量化，仍读取跨全部 O 组计算的 Native 统一标度。
            with pl.spmd(
                scale_blocks, name_hint="quant", deps=[scale_tid], allow_early_resolve=True
            ) as q_tid:
                quant_start = pl.tile.get_block_idx() * QUANT_TASK_T_TILE
                for qt in pl.pipeline(
                    quant_start, pl.min(quant_start + QUANT_TASK_T_TILE, t_dim), QUANT_TOKEN_TILE, stage=2
                ):
                    token_multiplier = act_scale_q[0:1, qt : qt + QUANT_TOKEN_TILE]
                    g_sq_col = pl.reshape(token_multiplier, [QUANT_TOKEN_TILE, 1])
                    oc_q = o_r_pad[qt : qt + QUANT_TOKEN_TILE, col_g : col_g + O_LORA]
                    oc_q = pl.cast(pl.cast(oc_q, pl.BF16, mode="rint"), pl.FP32)
                    oq_scaled = pl.row_expand_mul(oc_q, g_sq_col)
                    oq_i32 = pl.cast(oq_scaled, target_type=pl.INT32, mode="rint")
                    oq_half = pl.cast(oq_i32, target_type=pl.FP16, mode="round")
                    oq_i8 = pl.cast(oq_half, target_type=pl.INT8, mode="trunc")
                    o_r_i8_pad[qt : qt + QUANT_TOKEN_TILE, col_g : col_g + O_LORA] = oq_i8
                # 尾部补零仅由最后一个 token 块负责，避免多 worker 重复写。
                if quant_start + QUANT_TASK_T_TILE >= t_dim:
                    for zt in pl.range(t_dim, proj_b_padded_rows, QUANT_TOKEN_TILE):
                        zero_half = pl.full([QUANT_TOKEN_TILE, O_LORA], dtype=pl.FP16, value=0.0)
                        zero_i8 = pl.cast(zero_half, target_type=pl.INT8, mode="trunc")
                        zero_rows = pl.min(QUANT_TOKEN_TILE, proj_b_padded_rows - zt)
                        o_r_i8_pad = pl.assemble(o_r_i8_pad, pl.set_validshape(zero_i8, zero_rows, O_LORA), [zt, col_g])

            partials, pb_tid = proj_b_mm(
                o_r_i8_pad, wo_b, partials, g, col_g, proj_b_t_rows, q_tid, ROW_TILE,
            )
            proj_b_tids[g] = pb_tid

    # Sum INT32 group partials before dequantizing by the common token scale,
    # matching Native's single quantized WO-B matmul.
    with pl.spmd(
        act_t_blks * (D // PROJ_B_ACT_N_TILE),
        name_hint="proj_b_act_hc_post",
        sync_start=POST_SYNC,
        deps=[proj_b_tids[i] for i in range(O_GROUPS)],
        allow_early_resolve=True,
    ) as _act_tid:
        act_idx = pl.tile.get_block_idx()
        tblk = act_idx // (D // PROJ_B_ACT_N_TILE)  # token block outermost
        nreg = act_idx - tblk * (D // PROJ_B_ACT_N_TILE)
        ob_n0 = nreg * PROJ_B_ACT_N_TILE
        t0 = tblk * act_row_step
        wb_scale = wo_b_scale[ob_n0 : ob_n0 + PROJ_B_ACT_N_TILE]
        wb_scale_chunk = pl.reshape(wb_scale, [1, PROJ_B_ACT_N_TILE])
        for b_tb in pl.range(t0, pl.min(t0 + act_row_step, t_dim), PROJ_B_ACT_T_TILE):
            acc_i32 = pl.full([PROJ_B_ACT_T_TILE, PROJ_B_ACT_N_TILE], dtype=pl.INT32, value=0)
            for act_g in pl.pipeline(O_GROUPS, stage=2):
                p_col0 = act_g * D + ob_n0
                p_g = partials[b_tb : b_tb + PROJ_B_ACT_T_TILE, p_col0 : p_col0 + PROJ_B_ACT_N_TILE]
                acc_i32 = pl.add(acc_i32, p_g)
            output_token_scale = pl.read(act_scale_dq, [0, b_tb])
            # Native WO-B dequantizes the INT32 result by channel first,
            # then applies the token scale before its BF16 cast.
            acc = pl.col_expand_mul(pl.cast(acc_i32, pl.FP32), wb_scale_chunk)
            out_t = pl.mul(acc, output_token_scale)
            out_bf16 = pl.create_tensor([PROJ_B_ACT_T_TILE, PROJ_B_ACT_N_TILE], dtype=pl.BF16)
            out_bf16 = pl.cast(out_t, target_type=pl.BF16, mode="rint")
            output_rows = pl.min(PROJ_B_ACT_T_TILE, t_dim - b_tb)
            y_flat = hc_post_block(
                out_bf16, residual_flat, post, comb, y_flat, b_tb, ob_n0,
                output_rows, PROJ_B_ACT_T_TILE, PROJ_B_ACT_N_TILE,
            )

    return x_out


@pl.jit.inline
def decode_o_proj_tp1(
    o_packed: pl.Tensor[[O_GROUPS * T_PAD, O_GROUP_IN], pl.BF16],
    wo_a: pl.Tensor[[O_GROUPS, O_GROUP_IN, O_LORA], pl.BF16, WO_A_WEIGHT_LAYOUT],
    wo_b: pl.Tensor[[O_GROUPS * O_LORA, D], pl.INT8, QUANT_WEIGHT_LAYOUT],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    residual: pl.Tensor[[T_DYN, HC_MULT, D], pl.BF16],
    post: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    comb: pl.Tensor[[T_DYN, HC_MULT * HC_MULT], pl.FP32],
    x_out: pl.Out[pl.Tensor[[T_DYN, HC_MULT, D], pl.BF16]],
    heads_dep: pl.Scalar[pl.TASK_ID],
):
    """Use upstream row/column tiles while retaining Native weight storage."""
    t_dim = pl.tensor.dim(x_out, 0)
    if t_dim <= PROJ_B_SMALL_T_TILE:
        x_out = _decode_o_proj_tp1_tiled(
            o_packed, wo_a, wo_b, wo_b_scale, residual, post, comb, x_out, heads_dep,
            PROJ_B_SMALL_T_TILE, PROJ_A_MM_N_TILE, False,
        )
    elif t_dim == PROJ_B_MEDIUM_T_TILE:
        # T=96: 24x4 -> 48x2 real tokens. Keep the 1xD tile, stage=2
        # group dequantization, residual reuse and rounding unchanged.
        # T=144 regressed with 48x3 + sync, so do not extend this by threshold.
        x_out = _decode_o_proj_tp1_tiled(
            o_packed, wo_a, wo_b, wo_b_scale, residual, post, comb, x_out, heads_dep,
            PROJ_B_MEDIUM_T_TILE, PROJ_A_MM_N_TILE, True,
        )
    elif t_dim <= PROJ_B_MEDIUM_T_TILE:
        x_out = _decode_o_proj_tp1_tiled(
            o_packed, wo_a, wo_b, wo_b_scale, residual, post, comb, x_out, heads_dep,
            PROJ_B_MEDIUM_T_TILE, PROJ_A_MM_N_TILE, False,
        )
    else:
        x_out = _decode_o_proj_tp1_tiled(
            o_packed, wo_a, wo_b, wo_b_scale, residual, post, comb, x_out, heads_dep,
            PROJ_B_MM_T_TILE, PROJ_A_LARGE_N_TILE, False,
        )
    return x_out
