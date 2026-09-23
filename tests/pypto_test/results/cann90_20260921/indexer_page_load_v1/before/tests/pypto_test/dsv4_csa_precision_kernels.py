"""Diagnostic boundaries using the integration's existing computation functions."""

import pypto.language as pl

from vllm_ascend.ops.pypto.deepseek_v4_flash_dspark.decode_indexer import (
    IDX_HEAD_DIM,
    IDX_N_HEADS,
    ROPE_HEAD_DIM,
    T_PAD,
    TP1_WEIGHTS_WORKERS,
    NATIVE_QLI_WEIGHT_ROWS,
    indexer_head_coefficients,
    indexer_qr_hadamard_mm,
    indexer_qr_rope,
    indexer_weights_project,
)
from vllm_ascend.ops.pypto.deepseek_v4_flash_dspark.qkv_proj_rope import (
    H,
    HEAD_DIM,
    ROPE_DIM,
    Q_LORA,
    QPROJ_T_PAD,
    T_DYN,
    T_TILE,
    D,
    q_proj_qa,
    q_proj_qr,
    q_proj_qr_normalize,
    q_proj_qr_rms,
    qkv_proj_rope,
)
from vllm_ascend.ops.pypto.deepseek_v4_flash_dspark.decode_sparse_attn_csa import (
    B_DYN,
    BLOCK_SIZE,
    CMP_BLOCK_NUM_DYN,
    COMPRESSED_TABLE_COLUMNS_DYN,
    IDX_TOPK,
    ORI_BLOCK_NUM_DYN,
    ORIGINAL_TABLE_COLUMNS_DYN,
    WIN,
    sparse_attn_csa_tp1,
)
from vllm_ascend.ops.pypto.deepseek_v4_flash_dspark.decode_o_proj import (
    O_GROUPS,
    O_GROUP_IN,
    O_LORA,
    T_DYN as O_PROJ_T_DYN,
    T_PAD as O_PROJ_T_PAD,
    decode_o_proj_tp1,
)


@pl.jit(auto_scope=False)
def diagnose_qa(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.BF16],
    qa: pl.Out[pl.Tensor[[QPROJ_T_PAD, Q_LORA], pl.FP32]],
):
    x.bind_dynamic(0, T_DYN)
    q_proj_qa(x, wq_a, qa, 0, pl.tensor.dim(x, 0))
    return qa


@pl.jit(auto_scope=False)
def diagnose_qr(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.BF16],
    gamma: pl.Tensor[[Q_LORA], pl.BF16],
    qr: pl.Out[pl.Tensor[[T_DYN, Q_LORA], pl.INT8]],
    qr_scale: pl.Out[pl.Tensor[[T_DYN, 1], pl.FP32]],
):
    x.bind_dynamic(0, T_DYN)
    qr.bind_dynamic(0, T_DYN)
    qr_scale.bind_dynamic(0, T_DYN)
    qr_pad = pl.create_tensor([QPROJ_T_PAD, Q_LORA], dtype=pl.INT8)
    scale_pad = pl.create_tensor([QPROJ_T_PAD, 1], dtype=pl.FP32)
    q_proj_qr(x, wq_a, gamma, qr, qr_scale, qr_pad, scale_pad)
    return qr, qr_scale


@pl.jit(auto_scope=False)
def diagnose_qr_normalize(
    qa: pl.Tensor[[T_DYN, Q_LORA], pl.FP32],
    gamma: pl.Tensor[[Q_LORA], pl.BF16],
    qr: pl.Out[pl.Tensor[[T_DYN, Q_LORA], pl.INT8]],
    qr_scale: pl.Out[pl.Tensor[[T_DYN, 1], pl.FP32]],
):
    qa.bind_dynamic(0, T_DYN)
    qr.bind_dynamic(0, T_DYN)
    qr_scale.bind_dynamic(0, T_DYN)
    qr_pad = pl.create_tensor([QPROJ_T_PAD, Q_LORA], dtype=pl.INT8)
    scale_pad = pl.create_tensor([QPROJ_T_PAD, 1], dtype=pl.FP32)
    q_proj_qr_normalize(qa, gamma, qr, qr_scale, qr_pad, scale_pad, 0, pl.tensor.dim(qa, 0))
    return qr, qr_scale


@pl.jit(auto_scope=False)
def diagnose_qr_rms(
    qa: pl.Tensor[[T_DYN, Q_LORA], pl.FP32],
    square_sum: pl.Out[pl.Tensor[[T_DYN, 1], pl.FP32]],
    inverse_rms: pl.Out[pl.Tensor[[T_DYN, 1], pl.FP32]],
    rms: pl.Out[pl.Tensor[[T_DYN, 1], pl.FP32]],
):
    qa.bind_dynamic(0, T_DYN)
    square_sum.bind_dynamic(0, T_DYN)
    inverse_rms.bind_dynamic(0, T_DYN)
    rms.bind_dynamic(0, T_DYN)
    for block in pl.spmd(pl.tensor.dim(qa, 0) // T_TILE):
        row = block * T_TILE
        sums, inverse, root = q_proj_qr_rms(qa, row)
        square_sum[row : row + T_TILE, :] = pl.reshape(sums, [T_TILE, 1])
        inverse_rms[row : row + T_TILE, :] = pl.reshape(inverse, [T_TILE, 1])
        rms[row : row + T_TILE, :] = pl.reshape(root, [T_TILE, 1])
    return square_sum, inverse_rms, rms


@pl.jit(auto_scope=False)
def diagnose_indexer_query(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    qr: pl.Tensor[[T_DYN, Q_LORA], pl.INT8],
    qr_scale: pl.Tensor[[T_DYN, 1], pl.FP32],
    wq_b: pl.Tensor[[Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32],
    cos: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.FP32],
    sin: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.FP32],
    hadamard: pl.Tensor[[IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16],
    qr_bf16: pl.Out[pl.Tensor[[T_PAD * IDX_N_HEADS, IDX_HEAD_DIM], pl.BF16]],
    query: pl.Out[pl.Tensor[[T_PAD * IDX_N_HEADS, IDX_HEAD_DIM], pl.INT8]],
    scale: pl.Out[pl.Tensor[[T_PAD * IDX_N_HEADS, 1], pl.FP32]],
):
    x.bind_dynamic(0, T_DYN)
    qr.bind_dynamic(0, T_DYN)
    qr_scale.bind_dynamic(0, T_DYN)
    cos.bind_dynamic(0, T_DYN)
    sin.bind_dynamic(0, T_DYN)
    projection = indexer_qr_rope(x, qr, qr_scale, wq_b, wq_b_scale, cos, sin, qr_bf16)
    indexer_qr_hadamard_mm(x, qr_bf16, hadamard, query, scale, projection)
    return qr_bf16, query, scale


@pl.jit(auto_scope=False)
def diagnose_indexer_weights(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    weights_proj: pl.Tensor[[D, IDX_N_HEADS], pl.BF16],
    position_ids: pl.Tensor[[T_DYN], pl.INT64],
    query_scale: pl.Tensor[[T_PAD * IDX_N_HEADS, 1], pl.FP32],
    published_weights: pl.Out[pl.Tensor[[T_PAD, IDX_N_HEADS], pl.FP32]],
    published_coefficients: pl.Out[pl.Tensor[[T_PAD, IDX_N_HEADS], pl.FP16]],
):
    x.bind_dynamic(0, T_DYN)
    position_ids.bind_dynamic(0, T_DYN)
    gate = pl.system.task_dummy(deps=[])
    weights, weights_tid = indexer_weights_project(x, weights_proj, gate, TP1_WEIGHTS_WORKERS)
    coefficients, coefficients_tid = indexer_head_coefficients(query_scale, weights, position_ids, gate, weights_tid)
    with pl.spmd(pl.tensor.dim(x, 0), deps=[coefficients_tid]):
        row = pl.tile.get_block_idx()
        published_weights[row : row + 1, :] = weights[row : row + 1, :]
        coefficient_row = row * NATIVE_QLI_WEIGHT_ROWS
        published_coefficients[row : row + 1, :] = coefficients[coefficient_row : coefficient_row + 1, :]
    return published_weights, published_coefficients


@pl.jit(auto_scope=False)
def diagnose_o_projection(
    packed: pl.Tensor[[O_GROUPS * O_PROJ_T_PAD, O_GROUP_IN], pl.BF16],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    output: pl.Out[pl.Tensor[[O_PROJ_T_DYN, D], pl.BF16]],
):
    output.bind_dynamic(0, O_PROJ_T_DYN)
    gate = pl.system.task_dummy(deps=[])
    decode_o_proj_tp1(packed, wo_a, wo_b, wo_b_scale, output, gate)
    return output


@pl.jit(auto_scope=False)
def diagnose_main_query(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16],
    cos: pl.Tensor[[T_DYN, ROPE_DIM], pl.FP32],
    sin: pl.Tensor[[T_DYN, ROPE_DIM], pl.FP32],
    gamma_cq: pl.Tensor[[Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16],
    query: pl.Out[pl.Tensor[[T_DYN, H, HEAD_DIM], pl.BF16]],
    qr: pl.Out[pl.Tensor[[T_DYN, Q_LORA], pl.INT8]],
    qr_scale: pl.Out[pl.Tensor[[T_DYN, 1], pl.FP32]],
):
    x.bind_dynamic(0, T_DYN)
    cos.bind_dynamic(0, T_DYN)
    sin.bind_dynamic(0, T_DYN)
    query.bind_dynamic(0, T_DYN)
    qr.bind_dynamic(0, T_DYN)
    qr_scale.bind_dynamic(0, T_DYN)
    kv = pl.create_tensor([pl.tensor.dim(x, 0), HEAD_DIM], dtype=pl.BF16)
    gate = pl.system.task_dummy(deps=[])
    qkv_proj_rope(x, wq_a, wq_b, wq_b_scale, wkv, cos, sin, gamma_cq, gamma_ckv,
                  query, kv, qr, qr_scale, gate)
    return query, qr, qr_scale


@pl.jit(auto_scope=False)
def diagnose_sparse_attention(
    query: pl.Tensor[[T_DYN, H, HEAD_DIM], pl.BF16],
    ori_kv: pl.InOut[pl.Tensor[[ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    ori_table: pl.Tensor[[B_DYN, ORIGINAL_TABLE_COLUMNS_DYN], pl.INT32],
    cmp_kv: pl.Tensor[[CMP_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    cmp_table: pl.Tensor[[B_DYN, COMPRESSED_TABLE_COLUMNS_DYN], pl.INT32],
    topk: pl.Tensor[[T_DYN, IDX_TOPK], pl.INT32],
    positions: pl.Tensor[[T_DYN, 1], pl.INT64],
    sink: pl.Tensor[[H], pl.FP32],
    cos: pl.Tensor[[T_DYN, ROPE_DIM], pl.FP32],
    sin: pl.Tensor[[T_DYN, ROPE_DIM], pl.FP32],
    packed: pl.Out[pl.Tensor[[O_GROUPS * O_PROJ_T_PAD, O_GROUP_IN], pl.BF16]],
):
    query.bind_dynamic(0, T_DYN)
    ori_kv.bind_dynamic(0, ORI_BLOCK_NUM_DYN)
    ori_table.bind_dynamic(0, B_DYN)
    ori_table.bind_dynamic(1, ORIGINAL_TABLE_COLUMNS_DYN)
    cmp_kv.bind_dynamic(0, CMP_BLOCK_NUM_DYN)
    cmp_table.bind_dynamic(0, B_DYN)
    cmp_table.bind_dynamic(1, COMPRESSED_TABLE_COLUMNS_DYN)
    topk.bind_dynamic(0, T_DYN)
    positions.bind_dynamic(0, T_DYN)
    cos.bind_dynamic(0, T_DYN)
    sin.bind_dynamic(0, T_DYN)
    sparse_attn_csa_tp1(query, ori_kv, ori_table, cmp_kv, cmp_table, topk,
                        positions, sink, cos, sin, packed)
    return packed
