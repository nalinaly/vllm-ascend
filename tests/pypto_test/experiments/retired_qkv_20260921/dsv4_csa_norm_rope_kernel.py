"""Q/K normalization and interleaved FP32 RoPE for staged CSA comparison."""

import pypto.language as pl

TOKENS = pl.dynamic("CSA_NORM_ROPE_TOKENS")
HEADS = pl.dynamic("CSA_NORM_ROPE_HEADS")
HEAD_DIM = 512
ROPE_DIM = 64
NOPE_DIM = HEAD_DIM - ROPE_DIM
HEAD_TILE = 8
RMS_EPSILON = 1e-6


@pl.jit
def norm_rope(
    projected: pl.Tensor[[TOKENS, HEADS, HEAD_DIM], pl.BF16],
    gamma: pl.Tensor[[HEAD_DIM], pl.BF16],
    cos: pl.Tensor[[TOKENS, ROPE_DIM], pl.FP32],
    sin: pl.Tensor[[TOKENS, ROPE_DIM], pl.FP32],
    output: pl.Out[pl.Tensor[[TOKENS, HEADS, HEAD_DIM], pl.BF16]],
):
    projected.bind_dynamic(0, TOKENS)
    projected.bind_dynamic(1, HEADS)
    cos.bind_dynamic(0, TOKENS)
    sin.bind_dynamic(0, TOKENS)
    output.bind_dynamic(0, TOKENS)
    output.bind_dynamic(1, HEADS)
    tokens = pl.tensor.dim(projected, 0)
    heads = pl.tensor.dim(projected, 1)
    rows = tokens * heads
    flat_input = pl.reshape(projected, [rows, HEAD_DIM])
    flat_output = pl.reshape(output, [rows, HEAD_DIM])
    for token in pl.spmd(tokens, name_hint="csa_norm_rope"):
        g = pl.reshape(pl.cast(gamma[:], pl.FP32), [1, HEAD_DIM])
        cos_even = pl.gather(cos[token : token + 1, :], mask_pattern=pl.tile.MaskPattern.P0101)
        sin_even = pl.gather(sin[token : token + 1, :], mask_pattern=pl.tile.MaskPattern.P0101)
        for head in pl.range(0, heads, HEAD_TILE):
            row = token * heads + head
            valid_heads = pl.min(HEAD_TILE, heads - head)
            x = pl.cast(
                pl.slice(flat_input, [HEAD_TILE, HEAD_DIM], [row, 0], valid_shape=[valid_heads, HEAD_DIM]),
                pl.FP32,
            )
            variance = pl.mul(pl.reshape(pl.row_sum(pl.mul(x, x)), [1, HEAD_TILE]), 1.0 / HEAD_DIM)
            inv_rms = pl.reshape(pl.rsqrt(pl.add(variance, RMS_EPSILON), high_precision=True), [HEAD_TILE, 1])
            normed = pl.col_expand_mul(pl.row_expand_mul(x, inv_rms), g)
            normalized = pl.cast(normed, pl.BF16, mode="rint")
            flat_output = pl.assemble(flat_output, normalized[:, :NOPE_DIM], [row, 0])
            rope = pl.cast(normalized[:, NOPE_DIM:HEAD_DIM], pl.FP32)
            even = pl.gather(rope, mask_pattern=pl.tile.MaskPattern.P0101)
            odd = pl.gather(rope, mask_pattern=pl.tile.MaskPattern.P1010)
            rotated_even = pl.sub(pl.col_expand_mul(even, cos_even), pl.col_expand_mul(odd, sin_even))
            rotated_odd = pl.add(pl.col_expand_mul(even, sin_even), pl.col_expand_mul(odd, cos_even))
            rotated = pl.full([HEAD_TILE, ROPE_DIM], dtype=pl.FP32, value=0.0)
            rotated = pl.set_validshape(rotated, valid_heads, ROPE_DIM)
            rotated = pl.tensor.scatter(rotated_even, mask_pattern=pl.tile.MaskPattern.P0101, dst=rotated)
            rotated = pl.tensor.scatter(rotated_odd, mask_pattern=pl.tile.MaskPattern.P1010, dst=rotated)
            rounded = pl.cast(rotated, pl.BF16, mode="rint")
            flat_output = pl.assemble(flat_output, rounded, [row, NOPE_DIM])
    return output
# Retired experiment; the CSA integration uses the pypto-lib TP1 reference.
