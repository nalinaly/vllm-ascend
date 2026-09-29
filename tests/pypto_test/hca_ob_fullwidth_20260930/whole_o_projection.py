# SPDX-License-Identifier: Apache-2.0
"""实验：Native整token量化，比较分组整数归约与完整K乘法。"""

import pypto.language as pl

from ..deepseek_v4_flash_dspark_perf.config import INT8_AMAX_EPS, INT8_SCALE_MAX
from ..deepseek_v4_flash_dspark_perf.decode_o_proj import (
    O_GROUP_IN,
    O_GROUPS,
    O_LORA,
    PROJ_A_ROW_TILE,
    T_PAD,
    D,
    proj_a_mm,
    proj_b_mm,
)
from ..deepseek_v4_flash_dspark_perf.nz_mode import QUANT_WEIGHT_LAYOUT, WO_A_WEIGHT_LAYOUT

FULL_WIDTH = O_GROUPS * O_LORA
FULL_M = 32
FULL_N = 256
FULL_K = 512
QUANT_WORKERS = 48


@pl.jit.inline
def whole_quant(
    projected: pl.Tensor[[T_PAD, FULL_WIDTH], pl.FP32],
    quantized: pl.Tensor[[T_PAD, FULL_WIDTH], pl.INT8],
    scales: pl.Tensor[[1, T_PAD], pl.FP32],
    tokens: pl.Scalar[pl.INDEX],
    padded_rows: pl.Scalar[pl.INDEX],
    ready: pl.Array[O_GROUPS, pl.TASK_ID],
):
    with pl.spmd(
        pl.min(QUANT_WORKERS, padded_rows), name_hint="hca_o_whole_quant",
        deps=[ready[i] for i in range(O_GROUPS)], allow_early_resolve=True,
    ) as quant_ready:
        worker = pl.tile.get_block_idx()
        for row in pl.range(worker, padded_rows, QUANT_WORKERS):
            if row < tokens:
                value = pl.load(projected, [row, 0], [1, FULL_WIDTH])
                # Native O-A先落BF16，然后对拼接的8192列动态量化。
                value = pl.cast(pl.cast(value, pl.BF16, mode="rint"), pl.FP32)
                lanes = pl.reshape(pl.abs(value), [FULL_WIDTH // 8, 8])
                maxima = pl.col_max(lanes)
                # 用8个有效副本满足列向归约32B对齐；只读取一份标度。
                broadcast = pl.col_expand_add(pl.tile.full([8, 8], dtype=pl.FP32, value=0.0), maxima)
                maximum8 = pl.row_max(broadcast, pl.create_tile([8, 8], dtype=pl.FP32))
                amax = pl.reshape(pl.maximum(maximum8, INT8_AMAX_EPS), [1, 8])
                multiplier8 = pl.div(pl.tile.full([1, 8], dtype=pl.FP32, value=INT8_SCALE_MAX), amax)
                multiplier = pl.tile.read(multiplier8, [0, 0])
                dequant8 = pl.mul(amax, 1.0 / INT8_SCALE_MAX)
                scaled = pl.mul(value, multiplier)
                integer = pl.cast(scaled, pl.INT32, mode="rint")
                half = pl.cast(integer, pl.FP16, mode="round")
                encoded = pl.cast(half, pl.INT8, mode="trunc")
                pl.store(encoded, [row, 0], quantized)
                pl.store(pl.set_validshape(dequant8, 1, 1), [0, row], scales)
            else:
                encoded_zero = pl.cast(pl.tile.full([1, FULL_WIDTH], dtype=pl.FP16, value=0.0), pl.INT8, mode="trunc")
                pl.store(encoded_zero, [row, 0], quantized)
    return quantized, scales, quant_ready


@pl.jit.inline
def project_and_quant(
    packed: pl.Tensor[[O_GROUPS * T_PAD, O_GROUP_IN], pl.BF16],
    wo_a: pl.Tensor[[O_GROUPS, O_GROUP_IN, O_LORA], pl.BF16, WO_A_WEIGHT_LAYOUT],
    tokens: pl.Scalar[pl.INDEX],
    heads_dep: pl.Scalar[pl.TASK_ID],
    ROW_TILE: pl.constexpr,
    A_COL_TILE: pl.constexpr,
):
    projected = pl.create_tensor([T_PAD, FULL_WIDTH], dtype=pl.FP32)
    quantized = pl.create_tensor([T_PAD, FULL_WIDTH], dtype=pl.INT8)
    scales = pl.create_tensor([1, T_PAD], dtype=pl.FP32)
    ready = pl.array.create(O_GROUPS, pl.TASK_ID)
    pa_rows = (tokens + PROJ_A_ROW_TILE - 1) // PROJ_A_ROW_TILE
    padded = (tokens + ROW_TILE - 1) // ROW_TILE * ROW_TILE
    with pl.manual_scope():
        for group in pl.parallel(O_GROUPS):
            projected, a_ready = proj_a_mm(
                packed, wo_a, projected, group, group * T_PAD, group * O_LORA,
                tokens, pa_rows, heads_dep, A_COL_TILE,
            )
            ready[group] = a_ready
    quantized, scales, quant_ready = whole_quant(projected, quantized, scales, tokens, padded, ready)
    return quantized, scales, quant_ready


@pl.jit.inline
def o_parts_grouped(
    packed: pl.Tensor[[O_GROUPS * T_PAD, O_GROUP_IN], pl.BF16],
    wo_a: pl.Tensor[[O_GROUPS, O_GROUP_IN, O_LORA], pl.BF16, WO_A_WEIGHT_LAYOUT],
    wo_b: pl.Tensor[[FULL_WIDTH, D], pl.INT8, QUANT_WEIGHT_LAYOUT],
    tokens: pl.Scalar[pl.INDEX],
    heads_dep: pl.Scalar[pl.TASK_ID],
    ROW_TILE: pl.constexpr,
    A_COL_TILE: pl.constexpr,
):
    quantized, scales, quant_ready = project_and_quant(packed, wo_a, tokens, heads_dep, ROW_TILE, A_COL_TILE)
    partials = pl.create_tensor([T_PAD, O_GROUPS * D], dtype=pl.INT32)
    ready = pl.array.create(O_GROUPS, pl.TASK_ID)
    rows = (tokens + ROW_TILE - 1) // ROW_TILE
    with pl.manual_scope():
        for group in pl.parallel(O_GROUPS):
            column_base = group * O_LORA
            partials, group_ready = proj_b_mm(
                quantized, wo_b, partials, group, column_base, rows, quant_ready, ROW_TILE,
            )
            ready[group] = group_ready
    all_ready = pl.system.task_dummy(deps=[ready[i] for i in range(O_GROUPS)])
    return partials, scales, all_ready


@pl.jit.inline
def o_parts_full(
    packed: pl.Tensor[[O_GROUPS * T_PAD, O_GROUP_IN], pl.BF16],
    wo_a: pl.Tensor[[O_GROUPS, O_GROUP_IN, O_LORA], pl.BF16, WO_A_WEIGHT_LAYOUT],
    wo_b: pl.Tensor[[FULL_WIDTH, D], pl.INT8, QUANT_WEIGHT_LAYOUT],
    tokens: pl.Scalar[pl.INDEX],
    heads_dep: pl.Scalar[pl.TASK_ID],
    ROW_TILE: pl.constexpr,
    A_COL_TILE: pl.constexpr,
):
    quantized, scales, quant_ready = project_and_quant(packed, wo_a, tokens, heads_dep, ROW_TILE, A_COL_TILE)
    result = pl.create_tensor([T_PAD, D], dtype=pl.INT32)
    row_tiles = (tokens + FULL_M - 1) // FULL_M
    with pl.spmd(
        row_tiles * (D // FULL_N), name_hint="hca_o_full_k",
        deps=[quant_ready], allow_early_resolve=True,
    ) as b_ready:
        worker = pl.tile.get_block_idx()
        row = worker // (D // FULL_N) * FULL_M
        column = (worker % (D // FULL_N)) * FULL_N
        acc = pl.create_tensor([FULL_M, FULL_N], dtype=pl.INT32)
        for kb in pl.pipeline(FULL_WIDTH // FULL_K, stage=2):
            kk = kb * FULL_K
            left = quantized[row : row + FULL_M, kk : kk + FULL_K]
            right = wo_b[kk : kk + FULL_K, column : column + FULL_N]
            acc = pl.matmul_acc(acc, left, right, init_cond=(kb == 0))
        result[row : row + FULL_M, column : column + FULL_N] = acc
    return result, scales, b_ready
