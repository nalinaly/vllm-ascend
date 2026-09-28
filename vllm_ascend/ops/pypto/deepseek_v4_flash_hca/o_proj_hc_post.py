# SPDX-License-Identifier: Apache-2.0
"""复用公共 O 投影主体，将反量化与 mHC post 放在同一个 Vector 任务内。"""

import pypto.language as pl

from ..deepseek_v4_flash_dspark_perf.decode_o_proj import (
    D, O_GROUPS, O_GROUP_IN, O_LORA, T_PAD, _decode_o_proj_tp1_parts,
    PROJ_A_LARGE_N_TILE, PROJ_A_MM_N_TILE, PROJ_B_SMALL_T_TILE,
    PROJ_B_MEDIUM_T_TILE, PROJ_B_MM_T_TILE,
)
from ..deepseek_v4_flash_dspark_perf.nz_mode import BF16_WEIGHT_LAYOUT, QUANT_WEIGHT_LAYOUT, WO_A_WEIGHT_LAYOUT

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
        packed, wo_a, wo_b, tokens, heads_dep, MM_ROWS, MM_COLS,
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
        row_ids = pl.cast(pl.tile.arange(0, [1, TOKEN_TILE], dtype=pl.INT32), pl.FP32)
        gather_tmp = pl.create_tile([1, TOKEN_TILE], dtype=pl.INT32)
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
            for out_h in pl.unroll(HC_MULT):
                post_index = pl.cast(pl.add(pl.mul(row_ids, 8.0), pl.cast(pl.cast(out_h, pl.INT32), pl.FP32)), pl.INT32)
                post_weight = pl.reshape(pl.tile.gather(post_rows, post_index, gather_tmp), [TOKEN_TILE, 1])
                value = pl.row_expand_mul(attention, post_weight)
                # 顺序仍为 post*x，随后按输入 HC 0、1、2、3 逐项相加。
                for in_h in pl.pipeline(HC_MULT, stage=4):
                    comb_index = pl.cast(pl.add(
                        pl.mul(row_ids, 16.0), pl.cast(pl.cast(in_h * HC_MULT + out_h, pl.INT32), pl.FP32),
                    ), pl.INT32)
                    coefficient = pl.reshape(pl.tile.gather(comb_rows, comb_index, gather_tmp), [TOKEN_TILE, 1])
                    source = pl.load(
                        residual_flat, [row, in_h * D + col0], [TOKEN_TILE, COL_TILE],
                        valid_shape=[valid, COL_TILE],
                    )
                    value = pl.add(value, pl.row_expand_mul(pl.cast(source, pl.FP32), coefficient))
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
