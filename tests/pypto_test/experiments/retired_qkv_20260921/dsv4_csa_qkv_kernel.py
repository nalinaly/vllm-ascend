"""ND projection primitives for staged CSA comparison; not a serving adapter."""

import pypto.language as pl

TOKENS = pl.dynamic("CSA_QKV_TOKENS")
INPUT_WIDTH = pl.dynamic("CSA_QKV_INPUT_WIDTH")
OUTPUT_WIDTH = pl.dynamic("CSA_QKV_OUTPUT_WIDTH")
MATMUL_ROWS = 16
MATMUL_COLS = 128
MATMUL_REDUCTION = 256
NORM_ROWS = 8
Q_LORA = 1024
RMS_EPSILON = 1e-6


@pl.jit
def bf16_projection(
    hidden: pl.Tensor[[TOKENS, INPUT_WIDTH], pl.BF16],
    weight: pl.Tensor[[INPUT_WIDTH, OUTPUT_WIDTH], pl.BF16],
    output: pl.Out[pl.Tensor[[TOKENS, OUTPUT_WIDTH], pl.BF16]],
):
    hidden.bind_dynamic(0, TOKENS)
    hidden.bind_dynamic(1, INPUT_WIDTH)
    weight.bind_dynamic(0, INPUT_WIDTH)
    weight.bind_dynamic(1, OUTPUT_WIDTH)
    output.bind_dynamic(0, TOKENS)
    output.bind_dynamic(1, OUTPUT_WIDTH)
    tokens = pl.tensor.dim(hidden, 0)
    reduction = pl.tensor.dim(hidden, 1)
    columns = pl.tensor.dim(weight, 1)
    for col_block in pl.spmd(columns // MATMUL_COLS, name_hint="csa_bf16_projection"):
        col = col_block * MATMUL_COLS
        for row in pl.range(0, tokens, MATMUL_ROWS):
            valid_rows = pl.min(MATMUL_ROWS, tokens - row)
            acc = pl.create_tensor([MATMUL_ROWS, MATMUL_COLS], dtype=pl.FP32)
            for start in pl.pipeline(0, reduction, MATMUL_REDUCTION, stage=2):
                x = pl.slice(
                    hidden,
                    [MATMUL_ROWS, MATMUL_REDUCTION],
                    [row, start],
                    valid_shape=[valid_rows, MATMUL_REDUCTION],
                )
                w = weight[start : start + MATMUL_REDUCTION, col : col + MATMUL_COLS]
                acc = pl.matmul_acc(acc, x, w, init_cond=(start == 0))
            value = pl.cast(acc, pl.BF16, mode="rint")
            value = pl.set_validshape(value, valid_rows, MATMUL_COLS)
            output = pl.assemble(output, value, [row, col])
    return output


@pl.jit
def q_rms_quant(
    projected: pl.Tensor[[TOKENS, Q_LORA], pl.BF16],
    gamma: pl.Tensor[[Q_LORA], pl.BF16],
    quantized: pl.Out[pl.Tensor[[TOKENS, Q_LORA], pl.INT8]],
    scale: pl.Out[pl.Tensor[[TOKENS, 1], pl.FP32]],
):
    projected.bind_dynamic(0, TOKENS)
    quantized.bind_dynamic(0, TOKENS)
    scale.bind_dynamic(0, TOKENS)
    tokens = pl.tensor.dim(projected, 0)
    for block in pl.spmd((tokens + NORM_ROWS - 1) // NORM_ROWS, name_hint="csa_q_rms_quant"):
        row = block * NORM_ROWS
        valid_rows = pl.min(NORM_ROWS, tokens - row)
        x = pl.cast(pl.slice(projected, [NORM_ROWS, Q_LORA], [row, 0], valid_shape=[valid_rows, Q_LORA]), pl.FP32)
        g = pl.reshape(pl.cast(gamma[:], pl.FP32), [1, Q_LORA])
        variance = pl.mul(pl.reshape(pl.row_sum(pl.mul(x, x)), [1, NORM_ROWS]), 1.0 / Q_LORA)
        inv_rms = pl.recip(pl.sqrt(pl.add(variance, RMS_EPSILON)))
        inv_rms = pl.reshape(inv_rms, [NORM_ROWS, 1])
        normed = pl.col_expand_mul(pl.row_expand_mul(x, inv_rms), g)
        amax = pl.reshape(pl.row_max(pl.abs(normed)), [1, NORM_ROWS])
        numerator = pl.full([1, NORM_ROWS], dtype=pl.FP32, value=127.0)
        numerator = pl.set_validshape(numerator, 1, valid_rows)
        quant_scale = pl.div(numerator, amax)
        dq_scale = pl.reshape(pl.recip(quant_scale), [NORM_ROWS, 1])
        quant_scale = pl.reshape(quant_scale, [NORM_ROWS, 1])
        rounded = pl.cast(pl.row_expand_mul(normed, quant_scale), pl.INT32, mode="rint")
        half = pl.cast(rounded, pl.FP16, mode="round")
        result = pl.cast(half, pl.INT8, mode="trunc")
        result = pl.set_validshape(result, valid_rows, Q_LORA)
        dq_scale = pl.set_validshape(dq_scale, valid_rows, 1)
        quantized = pl.assemble(quantized, result, [row, 0])
        scale = pl.assemble(scale, dq_scale, [row, 0])
    return quantized, scale


@pl.jit
def w8a8_projection(
    hidden: pl.Tensor[[TOKENS, INPUT_WIDTH], pl.INT8],
    weight: pl.Tensor[[INPUT_WIDTH, OUTPUT_WIDTH], pl.INT8],
    token_scale: pl.Tensor[[TOKENS, 1], pl.FP32],
    weight_scale: pl.Tensor[[OUTPUT_WIDTH], pl.BF16],
    output: pl.Out[pl.Tensor[[TOKENS, OUTPUT_WIDTH], pl.BF16]],
):
    hidden.bind_dynamic(0, TOKENS)
    hidden.bind_dynamic(1, INPUT_WIDTH)
    weight.bind_dynamic(0, INPUT_WIDTH)
    weight.bind_dynamic(1, OUTPUT_WIDTH)
    token_scale.bind_dynamic(0, TOKENS)
    weight_scale.bind_dynamic(0, OUTPUT_WIDTH)
    output.bind_dynamic(0, TOKENS)
    output.bind_dynamic(1, OUTPUT_WIDTH)
    tokens = pl.tensor.dim(hidden, 0)
    reduction = pl.tensor.dim(hidden, 1)
    columns = pl.tensor.dim(weight, 1)
    for col_block in pl.spmd(columns // MATMUL_COLS, name_hint="csa_w8a8_projection"):
        col = col_block * MATMUL_COLS
        for row in pl.range(0, tokens, MATMUL_ROWS):
            valid_rows = pl.min(MATMUL_ROWS, tokens - row)
            acc = pl.create_tensor([MATMUL_ROWS, MATMUL_COLS], dtype=pl.INT32)
            for start in pl.pipeline(0, reduction, MATMUL_REDUCTION, stage=2):
                x = pl.slice(
                    hidden,
                    [MATMUL_ROWS, MATMUL_REDUCTION],
                    [row, start],
                    valid_shape=[valid_rows, MATMUL_REDUCTION],
                )
                w = weight[start : start + MATMUL_REDUCTION, col : col + MATMUL_COLS]
                acc = pl.matmul_acc(acc, x, w, init_cond=(start == 0))
            row_scale = pl.slice(token_scale, [MATMUL_ROWS, 1], [row, 0], valid_shape=[valid_rows, 1])
            col_scale = pl.reshape(pl.cast(weight_scale[col : col + MATMUL_COLS], pl.FP32), [1, MATMUL_COLS])
            scaled = pl.col_expand_mul(pl.row_expand_mul(pl.cast(acc, pl.FP32), row_scale), col_scale)
            value = pl.cast(scaled, pl.BF16, mode="rint")
            value = pl.set_validshape(value, valid_rows, MATMUL_COLS)
            output = pl.assemble(output, value, [row, col])
    return output
# Retired experiment; the CSA integration uses the pypto-lib TP1 reference.
