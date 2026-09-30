# SPDX-License-Identifier: Apache-2.0
"""复用公共 O 投影主体，将反量化与 mHC post 放在同一个 Vector 任务内。"""

import pypto.language as pl

from ..deepseek_v4_flash_dspark_perf.decode_o_proj import (
    O_GROUP_IN,
    O_GROUPS,
    O_LORA,
    PROJ_A_LARGE_N_TILE,
    PROJ_A_MM_N_TILE,
    PROJ_B_MEDIUM_T_TILE,
    PROJ_B_MM_T_TILE,
    PROJ_B_SMALL_T_TILE,
    T_PAD,
    D,
    _decode_o_proj_tp1_parts,
)
from ..deepseek_v4_flash_dspark_perf.nz_mode import QUANT_WEIGHT_LAYOUT, WO_A_WEIGHT_LAYOUT

T_DYN = pl.dynamic("HCA_O_POST_TOKENS")
HC_MULT = 4
HC_DIM = HC_MULT * D
TOKEN_TILE = 8
# B16/S6 按 16 行分工，覆盖 48 个 Vector 核，避免收尾只使用半数核。
TASK_TOKENS = 16
COL_TILE = 512


@pl.jit.inline
def _o_proj_hc_post_tiled(
    packed: pl.Tensor[[O_GROUPS * T_PAD, O_GROUP_IN], pl.BF16],
    wo_a: pl.Tensor[[O_GROUPS, O_GROUP_IN, O_LORA], pl.BF16, WO_A_WEIGHT_LAYOUT],
    wo_b: pl.Tensor[[O_GROUPS * O_LORA, D], pl.INT8, QUANT_WEIGHT_LAYOUT],
    wo_scale: pl.Tensor[[D], pl.FP32],
    residual: pl.Tensor[[T_DYN, HC_MULT, D], pl.BF16],
    post: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    comb: pl.Tensor[[T_DYN, HC_MULT * HC_MULT], pl.FP32],
    output: pl.Tensor[[T_DYN, HC_MULT, D], pl.BF16],
    heads_dep: pl.Scalar[pl.TASK_ID],
    MM_ROWS: pl.constexpr,
    MM_COLS: pl.constexpr,
):
    tokens = pl.tensor.dim(residual, 0)
    partials, scales, ready = _decode_o_proj_tp1_parts(
        packed, wo_a, wo_b, tokens, heads_dep, MM_ROWS, MM_COLS, PIPELINE_OA=True,
    )
    residual_flat = pl.reshape(residual, [tokens, HC_DIM])
    output_flat = pl.reshape(output, [tokens, HC_DIM])
    blocks = (tokens + TASK_TOKENS - 1) // TASK_TOKENS
    with pl.spmd(
        blocks * (D // COL_TILE), name_hint="hca_oproj_hc_post",
        deps=[ready], allow_early_resolve=True,
    ):
        worker = pl.tile.get_block_idx()
        row0 = worker // (D // COL_TILE) * TASK_TOKENS
        col0 = worker % (D // COL_TILE) * COL_TILE
        weight_scale = pl.reshape(pl.load(wo_scale, [col0], [COL_TILE]), [1, COL_TILE])
        for row in pl.range(row0, pl.min(row0 + TASK_TOKENS, tokens), TOKEN_TILE):
            valid = pl.min(TOKEN_TILE, tokens - row)
            acc = pl.tile.full([TOKEN_TILE, COL_TILE], dtype=pl.FP32, value=0.0)
            for group in pl.pipeline(O_GROUPS, stage=2):
                part = pl.load(partials, [row, group * D + col0], [TOKEN_TILE, COL_TILE])
                scale = pl.reshape(pl.load(scales, [group, row], [1, TOKEN_TILE]), [TOKEN_TILE, 1])
                acc = pl.add(acc, pl.row_expand_mul(pl.cast(part, pl.FP32), scale))
            attention = pl.col_expand_mul(acc, weight_scale)
            # 保留原先 attention 输出落 BF16、mHC post 再转 FP32 的舍入点。
            attention = pl.cast(pl.cast(attention, pl.BF16, mode="rint"), pl.FP32)
            # Native 门控是行主序；整块读取后收集各列，生成独立的连续权重向量。
            post_rows = pl.load(post, [row, 0], [TOKEN_TILE, 8], valid_shape=[valid, HC_MULT])
            comb_rows = pl.load(comb, [row, 0], [TOKEN_TILE, HC_MULT * HC_MULT],
                                valid_shape=[valid, HC_MULT * HC_MULT])
            # 四个输入HC分量只加载、转FP32一次，供四个输出顺序复用。
            residual_0 = pl.cast(pl.load(
                residual_flat, [row, 0 * D + col0], [TOKEN_TILE, COL_TILE],
                valid_shape=[valid, COL_TILE],
            ), pl.FP32)
            residual_1 = pl.cast(pl.load(
                residual_flat, [row, 1 * D + col0], [TOKEN_TILE, COL_TILE],
                valid_shape=[valid, COL_TILE],
            ), pl.FP32)
            residual_2 = pl.cast(pl.load(
                residual_flat, [row, 2 * D + col0], [TOKEN_TILE, COL_TILE],
                valid_shape=[valid, COL_TILE],
            ), pl.FP32)
            residual_3 = pl.cast(pl.load(
                residual_flat, [row, 3 * D + col0], [TOKEN_TILE, COL_TILE],
                valid_shape=[valid, COL_TILE],
            ), pl.FP32)
            post_columns = pl.transpose(post_rows, axis1=0, axis2=1)
            comb_columns = pl.transpose(comb_rows, axis1=0, axis2=1)
            for out_h in pl.unroll(HC_MULT):
                post_weight = pl.reshape(
                    pl.tile.slice(post_columns, [1, TOKEN_TILE], [out_h, 0]), [TOKEN_TILE, 1],
                )
                value = pl.row_expand_mul(attention, post_weight)
                coefficient_0 = pl.reshape(
                    pl.tile.slice(comb_columns, [1, TOKEN_TILE], [0 * HC_MULT + out_h, 0]),
                    [TOKEN_TILE, 1],
                )
                value = pl.add(value, pl.row_expand_mul(residual_0, coefficient_0))
                coefficient_1 = pl.reshape(
                    pl.tile.slice(comb_columns, [1, TOKEN_TILE], [1 * HC_MULT + out_h, 0]),
                    [TOKEN_TILE, 1],
                )
                value = pl.add(value, pl.row_expand_mul(residual_1, coefficient_1))
                coefficient_2 = pl.reshape(
                    pl.tile.slice(comb_columns, [1, TOKEN_TILE], [2 * HC_MULT + out_h, 0]),
                    [TOKEN_TILE, 1],
                )
                value = pl.add(value, pl.row_expand_mul(residual_2, coefficient_2))
                coefficient_3 = pl.reshape(
                    pl.tile.slice(comb_columns, [1, TOKEN_TILE], [3 * HC_MULT + out_h, 0]),
                    [TOKEN_TILE, 1],
                )
                value = pl.add(value, pl.row_expand_mul(residual_3, coefficient_3))
                result = pl.cast(value, pl.BF16, mode="rint")
                pl.store(pl.set_validshape(result, valid, COL_TILE), [row, out_h * D + col0], output_flat)
    return output


@pl.jit.inline
def o_proj_hc_post(
    packed: pl.Tensor[[O_GROUPS * T_PAD, O_GROUP_IN], pl.BF16],
    wo_a: pl.Tensor[[O_GROUPS, O_GROUP_IN, O_LORA], pl.BF16, WO_A_WEIGHT_LAYOUT],
    wo_b: pl.Tensor[[O_GROUPS * O_LORA, D], pl.INT8, QUANT_WEIGHT_LAYOUT],
    wo_scale: pl.Tensor[[D], pl.FP32],
    residual: pl.Tensor[[T_DYN, HC_MULT, D], pl.BF16],
    post: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    comb: pl.Tensor[[T_DYN, HC_MULT * HC_MULT], pl.FP32],
    output: pl.Tensor[[T_DYN, HC_MULT, D], pl.BF16],
    heads_dep: pl.Scalar[pl.TASK_ID],
):
    tokens = pl.tensor.dim(residual, 0)
    if tokens <= PROJ_B_SMALL_T_TILE:
        output = _o_proj_hc_post_tiled(
            packed, wo_a, wo_b, wo_scale, residual, post, comb, output, heads_dep,
            PROJ_B_SMALL_T_TILE, PROJ_A_MM_N_TILE,
        )
    elif tokens <= PROJ_B_MEDIUM_T_TILE:
        output = _o_proj_hc_post_tiled(
            packed, wo_a, wo_b, wo_scale, residual, post, comb, output, heads_dep,
            PROJ_B_MEDIUM_T_TILE, PROJ_A_MM_N_TILE,
        )
    else:
        output = _o_proj_hc_post_tiled(
            packed, wo_a, wo_b, wo_scale, residual, post, comb, output, heads_dep,
            PROJ_B_MM_T_TILE, PROJ_A_LARGE_N_TILE,
        )
    return output
