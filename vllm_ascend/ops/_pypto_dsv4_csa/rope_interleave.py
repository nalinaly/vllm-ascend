# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Prepare the native DeepSeek-V4 interleaved RoPE cache for AICore users.

``ComplexExpRotaryEmbedding`` already materializes the full 64-column cache by
``repeat_interleave(2)``.  Therefore, per rope column ``j``, the A3
interleaved rotation needs:

    cos_il[j]     = cos_full[j]
    sin_signed[j] = sin_full[j] * sign[j],  sign = [-1, +1, -1, +1, ...]

so the forward rotation is ``out[j] = x[j]*cos_il[j] + x[j^1]*sin_signed[j]`` and the
inverse (conjugate) rotation is the same expression with ``pl.sub``.

Do not select the first 32 columns and duplicate them again.  That turns the
already-interleaved cache into four-column frequency groups and disagrees with
the native ``inplace_partial_rotary_mul`` and ``compressor`` operators.

Folding the sign into sin here rather than at each consumer is exact: multiplying by
+/-1 only flips the sign bit, so ``(x*sign)*sin`` and ``x*(sin*sign)`` are bit-identical.
"""

import pypto.language as pl

from .config import FLASH as M

# model config
ROPE_HEAD_DIM = M.qk_rope_head_dim
# tiling
B_TILE = 4  # rows per preparation block; runtime B is a multiple of 4


@pl.jit.inline
def rope_interleave(
    # Variable axes are inferred from the positive-static caller.  Declaring
    # them with ``pl.dynamic`` here would propagate DynDim back into the L1
    # entry and make its artifact metadata non-executable.
    cos_full: pl.Tensor,
    sin_full: pl.Tensor,
    cos_il: pl.Tensor,
    sin_signed: pl.Tensor,
):
    """Copy full-width cos and fold the interleaved sign into full-width sin."""
    b_dim = pl.tensor.dim(cos_full, 0)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="rope_interleave", allow_early_resolve=True):
        il_ones = pl.full([B_TILE, ROPE_HEAD_DIM], dtype=pl.FP32, value=1.0)
        il_col = pl.col_expand_mul(
            il_ones, pl.cast(pl.arange(0, [1, ROPE_HEAD_DIM], dtype=pl.INT32), target_type=pl.FP32)
        )
        il_dup_f = pl.cast(pl.cast(pl.mul(il_col, 0.5), target_type=pl.INT32, mode="trunc"), target_type=pl.FP32)
        il_lane = pl.sub(il_col, pl.mul(il_dup_f, 2.0))  # j%2
        il_sign = pl.sub(pl.mul(il_lane, 2.0), 1.0)  # [-1,+1,...]
        for il_blk in pl.range(b_dim // B_TILE):
            il_b0 = il_blk * B_TILE
            cos_il[il_b0 : il_b0 + B_TILE, 0:ROPE_HEAD_DIM] = cos_full[il_b0 : il_b0 + B_TILE, 0:ROPE_HEAD_DIM]
            sin_signed[il_b0 : il_b0 + B_TILE, 0:ROPE_HEAD_DIM] = pl.mul(
                sin_full[il_b0 : il_b0 + B_TILE, 0:ROPE_HEAD_DIM], il_sign
            )
