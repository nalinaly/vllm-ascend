# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""CPU-only mathematical references for the DeepSeek V4 decode CSA core.

This module deliberately has no PyPTO, torch_npu, or native DSA dependency.  It
models the public tensor contract rather than the accelerator tiling, so small
parameterized shapes can exercise the same dtype transitions and page-strided
state updates in host-only unit tests.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

DEFAULT_DECODE_SEQUENCE_LENGTH = 8
COMPRESS_RATIO = 4
COMPRESS_STATE_BLOCK_SIZE = 2
INT8_MAX = 127.0
INT8_AMAX_EPSILON = 1e-4


@dataclass(frozen=True, slots=True)
class QKVReferenceResult:
    """Outputs shared by the attention and indexer branches."""

    positions: torch.Tensor
    query: torch.Tensor
    key_value: torch.Tensor
    quantized_hidden_states: torch.Tensor
    hidden_dequant_scale: torch.Tensor
    quantized_query_lora: torch.Tensor
    query_lora_scale: torch.Tensor


@dataclass(frozen=True, slots=True)
class Ratio4CompressorResult:
    """Observable values and mutations produced by the main compressor."""

    positions: torch.Tensor
    projected_values: torch.Tensor
    projected_scores: torch.Tensor
    pooled_values: torch.Tensor
    normalized_values: torch.Tensor
    boundary_offsets: torch.Tensor
    state_rows_written: torch.Tensor
    cache_rows_written: torch.Tensor


@dataclass(frozen=True, slots=True)
class Ratio4IndexerCompressorResult:
    """Observable values and mutations produced by the indexer compressor."""

    compressor: Ratio4CompressorResult
    normalized_values: torch.Tensor
    transformed_values: torch.Tensor
    quantized_values: torch.Tensor
    dequant_scales: torch.Tensor


@dataclass(frozen=True, slots=True)
class IndexerReferenceResult:
    """Indexer scores/top-k plus the inner-compressor mutation record."""

    positions: torch.Tensor
    quantized_queries: torch.Tensor
    query_dequant_scales: torch.Tensor
    head_weights: torch.Tensor
    scores: torch.Tensor
    topk_indices: torch.Tensor
    inner_compressor: Ratio4IndexerCompressorResult


@dataclass(frozen=True, slots=True)
class SparseAttentionReferenceResult:
    """Sparse attention heads and grouped output-projection values."""

    positions: torch.Tensor
    attention_heads: torch.Tensor
    projected_groups: torch.Tensor
    output: torch.Tensor


def restore_decode_positions(
    start_positions: torch.Tensor,
    *,
    sequence_length: int = DEFAULT_DECODE_SEQUENCE_LENGTH,
) -> torch.Tensor:
    """Expand one absolute start position per request to token-major positions."""
    if not isinstance(start_positions, torch.Tensor):
        raise TypeError("start_positions must be a torch.Tensor")
    _require_cpu_tensor(start_positions, "start_positions")
    if start_positions.ndim != 1:
        raise ValueError(f"start_positions must have shape [batch], got {tuple(start_positions.shape)}")
    if start_positions.dtype not in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }:
        raise TypeError(f"start_positions must use an integer dtype, got {start_positions.dtype}")
    if sequence_length <= 0:
        raise ValueError(f"sequence_length must be positive, got {sequence_length}")
    if torch.any(start_positions < 0):
        raise ValueError("start_positions must be non-negative")

    offsets = torch.arange(sequence_length, dtype=torch.int64)
    return start_positions.to(torch.int64).view(-1, 1) + offsets.view(1, -1)


def qkv_projection_rope_reference(
    hidden_states: torch.Tensor,
    *,
    wq_a: torch.Tensor,
    wq_a_scale: torch.Tensor,
    wq_b: torch.Tensor,
    wq_b_scale: torch.Tensor,
    wkv: torch.Tensor,
    wkv_scale: torch.Tensor,
    gamma_cq: torch.Tensor,
    gamma_ckv: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
    start_positions: torch.Tensor,
    sequence_length: int = DEFAULT_DECODE_SEQUENCE_LENGTH,
    epsilon: float = 1e-6,
) -> QKVReferenceResult:
    """Reference W8A8 Q/KV projection, RMS normalization, and RoPE.

    The BF16 hidden activation is dynamically quantized once per token and the
    resulting INT8 rows are shared by ``wq_a`` and ``wkv``.  Both INT8 weight
    products accumulate in INT32, then apply the hidden-row and weight-column
    dequant scales.  The result is explicitly rounded to BF16, matching native
    ``npu_quant_matmul(..., output_dtype=BF16)``, before the existing FP32
    RMSNorm arithmetic.  ``wq_b`` remains an INT8 per-output-scaled projection.
    """
    named_tensors = {
        "hidden_states": hidden_states,
        "wq_a": wq_a,
        "wq_a_scale": wq_a_scale,
        "wq_b": wq_b,
        "wq_b_scale": wq_b_scale,
        "wkv": wkv,
        "wkv_scale": wkv_scale,
        "gamma_cq": gamma_cq,
        "gamma_ckv": gamma_ckv,
        "freqs_cos": freqs_cos,
        "freqs_sin": freqs_sin,
    }
    _require_cpu_tensors(named_tensors)
    positions = restore_decode_positions(start_positions, sequence_length=sequence_length)
    flat_positions = positions.reshape(-1)

    if hidden_states.ndim != 2:
        raise ValueError(f"hidden_states must have shape [tokens, hidden], got {tuple(hidden_states.shape)}")
    if hidden_states.dtype != torch.bfloat16:
        raise TypeError(f"hidden_states must use torch.bfloat16, got {hidden_states.dtype}")
    tokens, hidden_size = hidden_states.shape
    if tokens != flat_positions.numel():
        raise ValueError(f"hidden_states has {tokens} tokens but positions describe {flat_positions.numel()}")
    if wq_a.ndim != 2 or tuple(wq_a.shape) != (hidden_size, gamma_cq.numel()):
        raise ValueError("wq_a must have shape [hidden_size, query_lora_rank]")
    query_lora_rank = gamma_cq.numel()
    if wq_a.dtype != torch.int8:
        raise TypeError(f"wq_a must use torch.int8, got {wq_a.dtype}")
    if tuple(wq_a_scale.shape) != (query_lora_rank,):
        raise ValueError("wq_a_scale must have one FP32 value per query LoRA column")
    head_dim = gamma_ckv.numel()
    if head_dim <= 0:
        raise ValueError("gamma_ckv must be non-empty")
    if wkv.ndim != 2 or tuple(wkv.shape) != (hidden_size, head_dim):
        raise ValueError("wkv must have shape [hidden_size, head_dim]")
    if wkv.dtype != torch.int8:
        raise TypeError(f"wkv must use torch.int8, got {wkv.dtype}")
    if tuple(wkv_scale.shape) != (head_dim,):
        raise ValueError("wkv_scale must have one FP32 value per KV output column")
    if wq_b.ndim != 2 or wq_b.shape[0] != query_lora_rank or wq_b.shape[1] % head_dim:
        raise ValueError("wq_b must have shape [query_lora_rank, num_heads * head_dim]")
    if wq_b.dtype != torch.int8:
        raise TypeError(f"wq_b must use torch.int8, got {wq_b.dtype}")
    projected_query_size = wq_b.shape[1]
    num_heads = projected_query_size // head_dim
    if tuple(wq_b_scale.shape) != (projected_query_size,):
        raise ValueError("wq_b_scale must have one FP32 value per projected query column")
    for name, scale in (
        ("wq_a_scale", wq_a_scale),
        ("wq_b_scale", wq_b_scale),
        ("wkv_scale", wkv_scale),
    ):
        if scale.dtype != torch.float32:
            raise TypeError(f"{name} must use torch.float32, got {scale.dtype}")
    _validate_rope_tables(freqs_cos, freqs_sin, flat_positions, head_dim)

    quantized_hidden, hidden_dequant_scale = _quantize_int8_per_row(hidden_states)

    query_lora_accumulator = quantized_hidden.to(torch.int32) @ wq_a.to(torch.int32)
    query_lora_dequantized = (
        query_lora_accumulator.float() * hidden_dequant_scale * wq_a_scale.float().view(1, query_lora_rank)
    )
    query_lora_bf16 = query_lora_dequantized.to(torch.bfloat16)
    query_lora_normalized = _rms_normalize(query_lora_bf16, gamma_cq, epsilon)
    quantized_query_lora, query_lora_scale = _quantize_int8_per_row(query_lora_normalized)

    query_accumulator = quantized_query_lora.to(torch.int32) @ wq_b.to(torch.int32)
    query_dequantized = query_accumulator.float() * query_lora_scale * wq_b_scale.float().view(1, projected_query_size)
    query_heads = query_dequantized.view(tokens, num_heads, head_dim)
    query_heads = _rms_normalize(query_heads, None, epsilon)

    token_cos, token_sin = _gather_interleaved_rope(freqs_cos, freqs_sin, flat_positions, head_dim)
    rope_dim = token_cos.shape[-1]
    nope_dim = head_dim - rope_dim
    query_nope = query_heads[..., :nope_dim]
    query_rope = _apply_interleaved_rope(
        query_heads[..., nope_dim:],
        token_cos[:, None, :],
        token_sin[:, None, :],
    )
    query = torch.cat((query_nope, query_rope), dim=-1).to(torch.bfloat16)

    key_value_accumulator = quantized_hidden.to(torch.int32) @ wkv.to(torch.int32)
    key_value_dequantized = key_value_accumulator.float() * hidden_dequant_scale * wkv_scale.float().view(1, head_dim)
    key_value_bf16 = key_value_dequantized.to(torch.bfloat16)
    key_value_normalized = _rms_normalize(key_value_bf16, gamma_ckv, epsilon)
    kv_nope = key_value_normalized[:, :nope_dim]
    kv_rope = _apply_interleaved_rope(
        key_value_normalized[:, nope_dim:],
        token_cos,
        token_sin,
    )
    key_value = torch.cat((kv_nope, kv_rope), dim=-1).to(torch.bfloat16)

    return QKVReferenceResult(
        positions=positions,
        query=query,
        key_value=key_value,
        quantized_hidden_states=quantized_hidden,
        hidden_dequant_scale=hidden_dequant_scale,
        quantized_query_lora=quantized_query_lora,
        query_lora_scale=query_lora_scale,
    )


def ratio4_compressor_reference(
    hidden_states: torch.Tensor,
    *,
    state_cache: torch.Tensor,
    state_block_table: torch.Tensor,
    state_slot_mapping: torch.Tensor,
    value_weight: torch.Tensor,
    gate_weight: torch.Tensor,
    positional_bias: torch.Tensor,
    norm_weight: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
    compressed_cache: torch.Tensor,
    compressed_slot_mapping: torch.Tensor,
    start_positions: torch.Tensor,
    sequence_length: int = DEFAULT_DECODE_SEQUENCE_LENGTH,
    epsilon: float = 1e-6,
) -> Ratio4CompressorResult:
    """Run the overlapping ratio-4 main compressor and mutate public caches."""
    if compressed_cache.dtype != torch.bfloat16:
        raise TypeError(f"compressed_cache must use torch.bfloat16, got {compressed_cache.dtype}")
    pool = _project_scatter_and_pool(
        hidden_states,
        state_cache=state_cache,
        state_block_table=state_block_table,
        state_slot_mapping=state_slot_mapping,
        value_weight=value_weight,
        gate_weight=gate_weight,
        positional_bias=positional_bias,
        start_positions=start_positions,
        sequence_length=sequence_length,
    )
    head_dim = norm_weight.numel()
    compression_positions = _compression_positions(
        pool.positions,
        pool.boundary_offsets,
    )
    _validate_rope_tables(freqs_cos, freqs_sin, compression_positions, head_dim)
    cos, sin = _gather_interleaved_rope(
        freqs_cos,
        freqs_sin,
        compression_positions,
        head_dim,
    )
    normalized = _normalize_and_rotate(pool.pooled_values, norm_weight, cos, sin, epsilon)

    cache_rows = _write_boundary_cache(
        normalized.to(torch.bfloat16),
        compressed_cache,
        compressed_slot_mapping,
        pool.boundary_offsets,
        sequence_length,
    )
    return Ratio4CompressorResult(
        positions=pool.positions,
        projected_values=pool.projected_values,
        projected_scores=pool.projected_scores,
        pooled_values=pool.pooled_values,
        normalized_values=normalized,
        boundary_offsets=pool.boundary_offsets,
        state_rows_written=pool.state_rows_written,
        cache_rows_written=cache_rows,
    )


def ratio4_indexer_compressor_reference(
    hidden_states: torch.Tensor,
    *,
    state_cache: torch.Tensor,
    state_block_table: torch.Tensor,
    state_slot_mapping: torch.Tensor,
    value_weight: torch.Tensor,
    gate_weight: torch.Tensor,
    positional_bias: torch.Tensor,
    norm_weight: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
    hadamard: torch.Tensor,
    indexer_cache: torch.Tensor,
    indexer_scale: torch.Tensor,
    indexer_slot_mapping: torch.Tensor,
    start_positions: torch.Tensor,
    sequence_length: int = DEFAULT_DECODE_SEQUENCE_LENGTH,
    epsilon: float = 1e-6,
) -> Ratio4IndexerCompressorResult:
    """Run the ratio-4 indexer compressor and mutate padded INT8/FP16 caches."""
    pool = _project_scatter_and_pool(
        hidden_states,
        state_cache=state_cache,
        state_block_table=state_block_table,
        state_slot_mapping=state_slot_mapping,
        value_weight=value_weight,
        gate_weight=gate_weight,
        positional_bias=positional_bias,
        start_positions=start_positions,
        sequence_length=sequence_length,
    )
    head_dim = norm_weight.numel()
    if tuple(hadamard.shape) != (head_dim, head_dim):
        raise ValueError(f"hadamard must have shape {(head_dim, head_dim)}, got {tuple(hadamard.shape)}")
    _require_cpu_tensor(hadamard, "hadamard")
    compression_positions = _compression_positions(
        pool.positions,
        pool.boundary_offsets,
    )
    _validate_rope_tables(freqs_cos, freqs_sin, compression_positions, head_dim)

    cos, sin = _gather_interleaved_rope(
        freqs_cos,
        freqs_sin,
        compression_positions,
        head_dim,
    )
    normalized = _normalize_and_rotate(pool.pooled_values, norm_weight, cos, sin, epsilon).to(torch.bfloat16)
    transformed = normalized.float() @ hadamard.float()
    quantization_input = transformed.to(torch.bfloat16).float()
    quantized, scales = _quantize_int8_per_row(quantization_input)
    cache_rows = _write_indexer_boundary_cache(
        quantized,
        scales,
        indexer_cache,
        indexer_scale,
        indexer_slot_mapping,
        pool.boundary_offsets,
        sequence_length,
    )

    compressor = Ratio4CompressorResult(
        positions=pool.positions,
        projected_values=pool.projected_values,
        projected_scores=pool.projected_scores,
        pooled_values=pool.pooled_values,
        normalized_values=normalized.float(),
        boundary_offsets=pool.boundary_offsets,
        state_rows_written=pool.state_rows_written,
        cache_rows_written=cache_rows,
    )
    return Ratio4IndexerCompressorResult(
        compressor=compressor,
        normalized_values=normalized,
        transformed_values=transformed,
        quantized_values=quantized,
        dequant_scales=scales,
    )


def indexer_reference(
    hidden_states: torch.Tensor,
    *,
    quantized_query_lora: torch.Tensor,
    query_lora_scale: torch.Tensor,
    query_weight: torch.Tensor,
    query_weight_scale: torch.Tensor,
    head_weight_projection: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
    hadamard: torch.Tensor,
    inner_state_cache: torch.Tensor,
    inner_state_block_table: torch.Tensor,
    inner_state_slot_mapping: torch.Tensor,
    inner_value_weight: torch.Tensor,
    inner_gate_weight: torch.Tensor,
    inner_positional_bias: torch.Tensor,
    inner_norm_weight: torch.Tensor,
    indexer_cache: torch.Tensor,
    indexer_scale: torch.Tensor,
    indexer_block_table: torch.Tensor,
    indexer_slot_mapping: torch.Tensor,
    sequence_lengths: torch.Tensor,
    start_positions: torch.Tensor,
    topk_count: int,
    index_offset: int = 0,
    sequence_length: int = DEFAULT_DECODE_SEQUENCE_LENGTH,
    epsilon: float = 1e-6,
) -> IndexerReferenceResult:
    """Run the decode indexer, including its cache-producing inner compressor."""
    named_tensors = {
        "hidden_states": hidden_states,
        "quantized_query_lora": quantized_query_lora,
        "query_lora_scale": query_lora_scale,
        "query_weight": query_weight,
        "query_weight_scale": query_weight_scale,
        "head_weight_projection": head_weight_projection,
        "hadamard": hadamard,
        "indexer_cache": indexer_cache,
        "indexer_scale": indexer_scale,
        "indexer_block_table": indexer_block_table,
        "sequence_lengths": sequence_lengths,
    }
    _require_cpu_tensors(named_tensors)
    positions = restore_decode_positions(start_positions, sequence_length=sequence_length)
    batch, seq = positions.shape
    tokens, hidden_size = hidden_states.shape
    if tokens != batch * seq:
        raise ValueError(f"hidden_states has {tokens} tokens but positions describe {batch * seq}")
    if quantized_query_lora.dtype != torch.int8 or quantized_query_lora.ndim != 2:
        raise TypeError("quantized_query_lora must be a rank-2 INT8 tensor")
    query_lora_rank = quantized_query_lora.shape[1]
    if tuple(query_lora_scale.shape) != (tokens, 1):
        raise ValueError("query_lora_scale must have shape [tokens, 1]")
    if query_weight.ndim != 2 or query_weight.shape[0] != query_lora_rank:
        raise ValueError("query_weight must have shape [query_lora_rank, num_heads * head_dim]")
    if query_weight.dtype != torch.int8:
        raise TypeError(f"query_weight must use torch.int8, got {query_weight.dtype}")
    head_dim = hadamard.shape[0]
    if head_dim <= 0 or tuple(hadamard.shape) != (head_dim, head_dim):
        raise ValueError("hadamard must be a non-empty square matrix")
    if query_weight.shape[1] % head_dim:
        raise ValueError("query_weight output width must be divisible by the indexer head dimension")
    num_heads = query_weight.shape[1] // head_dim
    if tuple(query_weight_scale.shape) != (num_heads * head_dim,):
        raise ValueError("query_weight_scale must have one value per query output column")
    if tuple(head_weight_projection.shape) != (hidden_size, num_heads):
        raise ValueError("head_weight_projection must have shape [hidden_size, num_heads]")
    if tuple(sequence_lengths.shape) != (batch,):
        raise ValueError("sequence_lengths must have shape [batch]")
    if indexer_block_table.ndim != 2 or indexer_block_table.shape[0] != batch:
        raise ValueError("indexer_block_table must have shape [batch, logical_blocks]")
    if topk_count <= 0:
        raise ValueError(f"topk_count must be positive, got {topk_count}")

    query_accumulator = quantized_query_lora.to(torch.int32) @ query_weight.to(torch.int32)
    query_dequantized = (
        query_accumulator.float() * query_lora_scale.float() * query_weight_scale.float().view(1, num_heads * head_dim)
    ).view(tokens, num_heads, head_dim)

    request_cos, request_sin = _gather_interleaved_rope(
        freqs_cos,
        freqs_sin,
        positions[:, 0],
        head_dim,
    )
    token_cos = request_cos.repeat_interleave(seq, dim=0)[:, None, :]
    token_sin = request_sin.repeat_interleave(seq, dim=0)[:, None, :]
    rope_dim = token_cos.shape[-1]
    nope_dim = head_dim - rope_dim
    rotated_queries = torch.cat(
        (
            query_dequantized[..., :nope_dim],
            _apply_interleaved_rope(query_dequantized[..., nope_dim:], token_cos, token_sin),
        ),
        dim=-1,
    ).to(torch.bfloat16)
    transformed_queries = rotated_queries.float() @ hadamard.float()
    flat_queries = transformed_queries.reshape(tokens * num_heads, head_dim)
    quantized_queries, query_scales = _quantize_int8_per_row(flat_queries)
    quantized_queries = quantized_queries.view(tokens, num_heads, head_dim)
    query_scales = query_scales.view(tokens, num_heads)

    head_weight_scale = head_dim**-0.5 * num_heads**-0.5
    head_weights = (hidden_states.float() @ head_weight_projection.float()) * head_weight_scale

    inner_result = ratio4_indexer_compressor_reference(
        hidden_states,
        state_cache=inner_state_cache,
        state_block_table=inner_state_block_table,
        state_slot_mapping=inner_state_slot_mapping,
        value_weight=inner_value_weight,
        gate_weight=inner_gate_weight,
        positional_bias=inner_positional_bias,
        norm_weight=inner_norm_weight,
        freqs_cos=freqs_cos,
        freqs_sin=freqs_sin,
        hadamard=hadamard,
        indexer_cache=indexer_cache,
        indexer_scale=indexer_scale,
        indexer_slot_mapping=indexer_slot_mapping,
        start_positions=start_positions,
        sequence_length=sequence_length,
        epsilon=epsilon,
    )

    page_size = indexer_cache.shape[1]
    score_length = freqs_cos.shape[0] // COMPRESS_RATIO
    if score_length <= 0:
        raise ValueError("RoPE table must cover at least one compressed position")
    if (score_length + page_size - 1) // page_size > indexer_block_table.shape[1]:
        raise ValueError("indexer_block_table is too narrow for the score domain")
    if topk_count > score_length:
        raise ValueError(f"topk_count {topk_count} exceeds score domain {score_length}")

    negative_finite = torch.finfo(torch.float32).min
    scores = torch.full((tokens, score_length), negative_finite, dtype=torch.float32)
    topk_indices = torch.full((tokens, score_length), -1, dtype=torch.int32)
    for token in range(tokens):
        request = token // seq
        token_position = int(positions.reshape(-1)[token])
        visible = min(
            int(sequence_lengths[request]) // COMPRESS_RATIO,
            (token_position + 1) // COMPRESS_RATIO,
            score_length,
        )
        if visible <= 0:
            continue
        keys = torch.empty((visible, head_dim), dtype=torch.int32)
        key_scales = torch.empty((visible, 1), dtype=torch.float32)
        for compressed_position in range(visible):
            physical_block = int(indexer_block_table[request, compressed_position // page_size])
            intra = compressed_position % page_size
            if physical_block < 0 or physical_block >= indexer_cache.shape[0]:
                raise IndexError(f"indexer block table resolved to invalid block {physical_block}")
            keys[compressed_position].copy_(indexer_cache[physical_block, intra, 0].to(torch.int32))
            key_scales[compressed_position, 0] = indexer_scale[physical_block, intra, 0, 0].float()

        dot_products = keys @ quantized_queries[token].to(torch.int32).transpose(0, 1)
        dequantized = dot_products.float() * query_scales[token].view(1, num_heads)
        weighted = torch.relu(dequantized) * head_weights[token].view(1, num_heads)
        token_scores = weighted.sum(dim=-1, keepdim=True) * key_scales
        scores[token, :visible] = token_scores[:, 0]

        valid_topk = min(topk_count, visible)
        selected = torch.topk(token_scores[:, 0], valid_topk, largest=True, sorted=True).indices
        topk_indices[token, :valid_topk] = selected.to(torch.int32) + int(index_offset)

    return IndexerReferenceResult(
        positions=positions,
        quantized_queries=quantized_queries,
        query_dequant_scales=query_scales,
        head_weights=head_weights,
        scores=scores,
        topk_indices=topk_indices,
        inner_compressor=inner_result,
    )


def sparse_attention_csa_reference(
    query: torch.Tensor,
    *,
    swa_cache: torch.Tensor,
    window_swa_indices: torch.Tensor,
    compressed_cache: torch.Tensor,
    compressed_block_table: torch.Tensor,
    compressed_topk_indices: torch.Tensor,
    attention_sink: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
    output_lora_weight: torch.Tensor,
    output_weight: torch.Tensor,
    start_positions: torch.Tensor,
    sequence_length: int = DEFAULT_DECODE_SEQUENCE_LENGTH,
    attention_tile: int = 128,
) -> SparseAttentionReferenceResult:
    """Run sparse attention and the checkpoint's all-BF16 output projection.

    ``wo_a`` accumulates in FP32 and explicitly rounds to BF16 before ``wo_b``.
    Each BF16 ``wo_b`` group matmul produces an FP32 partial; the group sum is
    rounded once into the public BF16 output.
    """
    named_tensors = {
        "query": query,
        "swa_cache": swa_cache,
        "window_swa_indices": window_swa_indices,
        "compressed_cache": compressed_cache,
        "compressed_block_table": compressed_block_table,
        "compressed_topk_indices": compressed_topk_indices,
        "attention_sink": attention_sink,
        "freqs_cos": freqs_cos,
        "freqs_sin": freqs_sin,
        "output_lora_weight": output_lora_weight,
        "output_weight": output_weight,
    }
    _require_cpu_tensors(named_tensors)
    positions = restore_decode_positions(start_positions, sequence_length=sequence_length)
    batch, seq = positions.shape
    if query.ndim != 3:
        raise ValueError("query must have shape [tokens, num_heads, head_dim]")
    if query.dtype != torch.bfloat16:
        raise TypeError(f"query must use torch.bfloat16, got {query.dtype}")
    tokens, num_heads, head_dim = query.shape
    if tokens != batch * seq:
        raise ValueError(f"query has {tokens} tokens but positions describe {batch * seq}")
    if tuple(attention_sink.shape) != (num_heads,):
        raise ValueError("attention_sink must contain one value per query head")
    if window_swa_indices.ndim != 2 or window_swa_indices.shape[0] != tokens:
        raise ValueError("window_swa_indices must have shape [tokens, window]")
    if compressed_topk_indices.ndim != 2 or compressed_topk_indices.shape[0] != tokens:
        raise ValueError("compressed_topk_indices must have shape [tokens, topk_domain]")
    for name, cache in (("swa_cache", swa_cache), ("compressed_cache", compressed_cache)):
        if cache.ndim != 4 or cache.shape[2:] != (1, head_dim):
            raise ValueError(f"{name} must have shape [blocks, page_size, 1, head_dim]")
        if cache.dtype != torch.bfloat16:
            raise TypeError(f"{name} must use torch.bfloat16, got {cache.dtype}")
    if compressed_block_table.ndim != 2 or compressed_block_table.shape[0] != batch:
        raise ValueError("compressed_block_table must have shape [batch, logical_blocks]")
    if attention_tile <= 0:
        raise ValueError(f"attention_tile must be positive, got {attention_tile}")

    num_groups, output_lora_rank, group_input = output_lora_weight.shape
    if num_heads % num_groups:
        raise ValueError("num_heads must be divisible by output projection groups")
    heads_per_group = num_heads // num_groups
    if group_input != heads_per_group * head_dim:
        raise ValueError("output_lora_weight group input does not match packed attention heads")
    hidden_size = output_weight.shape[0]
    if tuple(output_weight.shape) != (hidden_size, num_groups * output_lora_rank):
        raise ValueError("output_weight must have shape [hidden_size, groups * output_lora_rank]")
    if output_lora_weight.dtype != torch.bfloat16:
        raise TypeError(f"output_lora_weight must use torch.bfloat16, got {output_lora_weight.dtype}")
    if output_weight.dtype != torch.bfloat16:
        raise TypeError(f"output_weight must use torch.bfloat16, got {output_weight.dtype}")

    token_cos, token_sin = _gather_interleaved_rope(
        freqs_cos,
        freqs_sin,
        positions.reshape(-1),
        head_dim,
    )
    rope_dim = token_cos.shape[-1]
    nope_dim = head_dim - rope_dim
    softmax_scale = head_dim**-0.5
    attention_heads_fp32 = torch.empty((tokens, num_heads, head_dim), dtype=torch.float32)

    for token in range(tokens):
        request = token // seq
        token_position = int(positions.reshape(-1)[token])
        selected_rows: list[torch.Tensor] = []
        for raw_row_tensor in window_swa_indices[token]:
            raw_row = int(raw_row_tensor)
            if raw_row < 0:
                continue
            block, intra = divmod(raw_row, swa_cache.shape[1])
            if block >= swa_cache.shape[0]:
                raise IndexError(f"SWA cache row {raw_row} exceeds physical storage")
            selected_rows.append(swa_cache[block, intra, 0])

        compressed_limit = (token_position + 1) // COMPRESS_RATIO
        for compressed_index_tensor in compressed_topk_indices[token]:
            compressed_index = int(compressed_index_tensor)
            if compressed_index < 0 or compressed_index >= compressed_limit:
                continue
            logical_block, intra = divmod(compressed_index, compressed_cache.shape[1])
            if logical_block >= compressed_block_table.shape[1]:
                raise IndexError(f"compressed position {compressed_index} exceeds block table")
            physical_block = int(compressed_block_table[request, logical_block])
            if physical_block < 0 or physical_block >= compressed_cache.shape[0]:
                raise IndexError(f"compressed block table resolved to invalid block {physical_block}")
            selected_rows.append(compressed_cache[physical_block, intra, 0])

        if not selected_rows:
            raise ValueError(f"token {token} has no visible sparse-attention cache rows")
        keys = torch.stack(selected_rows).to(torch.bfloat16)
        running_max = torch.full((num_heads, 1), torch.finfo(torch.float32).min)
        running_denominator = torch.zeros((num_heads, 1), dtype=torch.float32)
        running_numerator = torch.zeros((num_heads, head_dim), dtype=torch.float32)
        for start in range(0, keys.shape[0], attention_tile):
            key_tile = keys[start : start + attention_tile]
            block_scores = query[token].float() @ key_tile.float().transpose(0, 1)
            block_scores *= softmax_scale
            block_max = block_scores.amax(dim=-1, keepdim=True)
            block_exp = torch.exp(block_scores - block_max)
            block_denominator = block_exp.sum(dim=-1, keepdim=True)
            block_numerator = block_exp.to(torch.bfloat16).float() @ key_tile.float()
            merged_max = torch.maximum(running_max, block_max)
            old_scale = torch.exp(running_max - merged_max)
            new_scale = torch.exp(block_max - merged_max)
            running_denominator = running_denominator * old_scale + block_denominator * new_scale
            running_numerator = running_numerator * old_scale + block_numerator * new_scale
            running_max = merged_max

        sink_denominator = torch.exp(attention_sink.float().view(num_heads, 1) - running_max)
        attention_heads_fp32[token] = running_numerator / (running_denominator + sink_denominator)

    attention_nope = attention_heads_fp32[..., :nope_dim].to(torch.bfloat16)
    attention_rope = _apply_interleaved_rope(
        attention_heads_fp32[..., nope_dim:],
        token_cos[:, None, :],
        -token_sin[:, None, :],
    ).to(torch.bfloat16)
    attention_heads = torch.cat((attention_nope, attention_rope), dim=-1)

    projected_groups = torch.empty((tokens, num_groups, output_lora_rank), dtype=torch.bfloat16)
    output_accumulator = torch.zeros((tokens, hidden_size), dtype=torch.float32)
    for group in range(num_groups):
        first_head = group * heads_per_group
        packed = attention_heads[:, first_head : first_head + heads_per_group].reshape(tokens, group_input)
        projected = packed.float() @ output_lora_weight[group].float().transpose(0, 1)
        projected_bf16 = projected.to(torch.bfloat16)
        weight_start = group * output_lora_rank
        weight_slice = output_weight[:, weight_start : weight_start + output_lora_rank]
        partial = projected_bf16.float() @ weight_slice.float().transpose(0, 1)
        output_accumulator += partial
        projected_groups[:, group].copy_(projected_bf16)

    output = output_accumulator.to(torch.bfloat16)
    return SparseAttentionReferenceResult(
        positions=positions,
        attention_heads=attention_heads,
        projected_groups=projected_groups,
        output=output,
    )


@dataclass(frozen=True, slots=True)
class _PoolResult:
    positions: torch.Tensor
    projected_values: torch.Tensor
    projected_scores: torch.Tensor
    pooled_values: torch.Tensor
    boundary_offsets: torch.Tensor
    state_rows_written: torch.Tensor


def _project_scatter_and_pool(
    hidden_states: torch.Tensor,
    *,
    state_cache: torch.Tensor,
    state_block_table: torch.Tensor,
    state_slot_mapping: torch.Tensor,
    value_weight: torch.Tensor,
    gate_weight: torch.Tensor,
    positional_bias: torch.Tensor,
    start_positions: torch.Tensor,
    sequence_length: int,
) -> _PoolResult:
    """Project tokens, update paged state, then pool completed ratio-4 windows."""
    named_tensors = {
        "hidden_states": hidden_states,
        "state_cache": state_cache,
        "state_block_table": state_block_table,
        "state_slot_mapping": state_slot_mapping,
        "value_weight": value_weight,
        "gate_weight": gate_weight,
        "positional_bias": positional_bias,
    }
    _require_cpu_tensors(named_tensors)
    positions = restore_decode_positions(start_positions, sequence_length=sequence_length)
    batch, seq = positions.shape
    tokens, hidden_size = hidden_states.shape
    if tokens != batch * seq:
        raise ValueError(f"hidden_states has {tokens} tokens but positions describe {batch * seq}")
    if value_weight.ndim != 2 or gate_weight.shape != value_weight.shape:
        raise ValueError("value_weight and gate_weight must share shape [2 * head_dim, hidden_size]")
    output_dim = value_weight.shape[0]
    if value_weight.shape[1] != hidden_size or output_dim % 2:
        raise ValueError("compressor weights must have shape [2 * head_dim, hidden_size]")
    head_dim = output_dim // 2
    expected_state_shape = (state_cache.shape[0], COMPRESS_STATE_BLOCK_SIZE, 1, 4 * head_dim)
    if tuple(state_cache.shape) != expected_state_shape:
        raise ValueError(f"state_cache must have shape {expected_state_shape}, got {tuple(state_cache.shape)}")
    if state_cache.dtype != torch.float32:
        raise TypeError(f"state_cache must use torch.float32, got {state_cache.dtype}")
    if state_block_table.ndim != 2 or state_block_table.shape[0] != batch:
        raise ValueError("state_block_table must have shape [batch, logical_blocks]")
    if tuple(positional_bias.shape) != (COMPRESS_RATIO, output_dim):
        raise ValueError(f"positional_bias must have shape {(COMPRESS_RATIO, output_dim)}")
    if state_slot_mapping.numel() != tokens:
        raise ValueError(f"state_slot_mapping must contain {tokens} rows")

    values = hidden_states.float() @ value_weight.float().transpose(0, 1)
    scores = hidden_states.float() @ gate_weight.float().transpose(0, 1)
    scores = scores + positional_bias.float()[positions.reshape(-1).remainder(COMPRESS_RATIO)]

    state_rows_written = torch.full((tokens,), -1, dtype=torch.int64)
    for token_index in range(tokens):
        physical_row = int(state_slot_mapping.reshape(-1)[token_index])
        if physical_row < 0:
            continue
        state_row = _state_row_from_physical_row(state_cache, physical_row)
        state_row[:output_dim].copy_(values[token_index])
        state_row[output_dim:].copy_(scores[token_index])
        state_rows_written[token_index] = physical_row

    writes_per_request = (seq + COMPRESS_RATIO - 1) // COMPRESS_RATIO
    pooled = torch.zeros((batch * writes_per_request, head_dim), dtype=torch.float32)
    boundary_offsets = torch.full((batch, writes_per_request), -1, dtype=torch.int64)
    for request in range(batch):
        first_position = int(positions[request, 0])
        first_offset = COMPRESS_RATIO - 1 - first_position % COMPRESS_RATIO
        for write_index in range(writes_per_request):
            offset = first_offset + write_index * COMPRESS_RATIO
            if offset >= seq:
                continue
            boundary_offsets[request, write_index] = offset
            write_position = first_position + offset
            current_window_start = write_position + 1 - COMPRESS_RATIO

            last_row = _state_row_for_position(
                state_cache,
                state_block_table,
                request,
                write_position,
            )
            running_max = last_row[output_dim + head_dim :].clone()
            running_denominator = torch.ones_like(running_max)
            running_numerator = last_row[head_dim:output_dim].clone()

            previous_window_start = current_window_start - COMPRESS_RATIO
            if write_position >= 2 * COMPRESS_RATIO - 1:
                for absolute_position in range(previous_window_start, current_window_start):
                    row = _state_row_for_position(
                        state_cache,
                        state_block_table,
                        request,
                        absolute_position,
                    )
                    running_max, running_denominator, running_numerator = _online_softmax_add(
                        running_max,
                        running_denominator,
                        running_numerator,
                        row[output_dim : output_dim + head_dim],
                        row[:head_dim],
                    )

            for absolute_position in range(current_window_start, write_position):
                row = _state_row_for_position(
                    state_cache,
                    state_block_table,
                    request,
                    absolute_position,
                )
                running_max, running_denominator, running_numerator = _online_softmax_add(
                    running_max,
                    running_denominator,
                    running_numerator,
                    row[output_dim + head_dim :],
                    row[head_dim:output_dim],
                )
            pool_row = request * writes_per_request + write_index
            pooled[pool_row].copy_(
                running_numerator / running_denominator,
            )

    return _PoolResult(
        positions=positions,
        projected_values=values,
        projected_scores=scores,
        pooled_values=pooled,
        boundary_offsets=boundary_offsets,
        state_rows_written=state_rows_written,
    )


def _require_cpu_tensor(tensor: torch.Tensor, name: str) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.device.type != "cpu":
        raise ValueError(f"{name} must be a CPU tensor, got {tensor.device}")


def _require_cpu_tensors(named_tensors: dict[str, torch.Tensor]) -> None:
    for name, tensor in named_tensors.items():
        _require_cpu_tensor(tensor, name)


def _validate_rope_tables(
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
    positions: torch.Tensor,
    head_dim: int,
) -> None:
    _require_cpu_tensors({"freqs_cos": freqs_cos, "freqs_sin": freqs_sin})
    if freqs_cos.ndim != 2 or freqs_sin.shape != freqs_cos.shape:
        raise ValueError("freqs_cos and freqs_sin must share shape [max_sequence, rope_dim]")
    rope_dim = freqs_cos.shape[1]
    if rope_dim <= 0 or rope_dim % 2 or rope_dim > head_dim:
        raise ValueError("RoPE width must be positive, even, and no larger than head_dim")
    if positions.numel() and int(positions.max()) >= freqs_cos.shape[0]:
        raise ValueError("position exceeds the supplied RoPE table")


def _gather_interleaved_rope(
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
    positions: torch.Tensor,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather the native full-width, already-interleaved RoPE cache.

    Production stores each frequency as adjacent columns
    ``[c0, c0, c1, c1, ...]``.  No width expansion belongs here; only the
    even-negative/odd-positive sign convention is applied to the sine rows.
    """
    _validate_rope_tables(freqs_cos, freqs_sin, positions, head_dim)
    rope_dim = freqs_cos.shape[1]
    token_cos = freqs_cos.float()[positions.to(torch.int64)]
    token_sin = freqs_sin.float()[positions.to(torch.int64)]
    signs = torch.ones(rope_dim, dtype=torch.float32)
    signs[0::2] = -1.0
    return token_cos, token_sin * signs


def _apply_interleaved_rope(values: torch.Tensor, cos: torch.Tensor, signed_sin: torch.Tensor) -> torch.Tensor:
    if values.shape[-1] != cos.shape[-1] or cos.shape != signed_sin.shape:
        raise ValueError("RoPE values, cos, and sin must have the same final width")
    swap = torch.arange(values.shape[-1], dtype=torch.int64) ^ 1
    return values * cos + values[..., swap] * signed_sin


def _rms_normalize(values: torch.Tensor, weight: torch.Tensor | None, epsilon: float) -> torch.Tensor:
    values_fp32 = values.float()
    inverse_rms = torch.rsqrt(values_fp32.square().mean(dim=-1, keepdim=True) + epsilon)
    normalized = values_fp32 * inverse_rms
    if weight is not None:
        normalized = normalized * weight.float()
    return normalized


def _quantize_int8_per_row(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    values_fp32 = values.float()
    amax = values_fp32.abs().amax(dim=-1, keepdim=True).clamp_min(INT8_AMAX_EPSILON)
    dequant_scale = amax / INT8_MAX
    quantized = torch.round(values_fp32 / dequant_scale).to(torch.int32).to(torch.float16).to(torch.int8)
    return quantized, dequant_scale


def _state_row_from_physical_row(state_cache: torch.Tensor, physical_row: int) -> torch.Tensor:
    block = physical_row // COMPRESS_STATE_BLOCK_SIZE
    intra = physical_row % COMPRESS_STATE_BLOCK_SIZE
    if block < 0 or block >= state_cache.shape[0]:
        raise IndexError(f"physical state row {physical_row} is outside {state_cache.shape[0]} blocks")
    return state_cache[block, intra, 0]


def _state_row_for_position(
    state_cache: torch.Tensor,
    state_block_table: torch.Tensor,
    request: int,
    absolute_position: int,
) -> torch.Tensor:
    if absolute_position < 0:
        raise IndexError(f"state position must be non-negative, got {absolute_position}")
    logical_block = absolute_position // COMPRESS_STATE_BLOCK_SIZE
    if logical_block >= state_block_table.shape[1]:
        raise IndexError(f"state position {absolute_position} exceeds the block table")
    physical_block = int(state_block_table[request, logical_block])
    if physical_block < 0 or physical_block >= state_cache.shape[0]:
        raise IndexError(f"state block table resolved to invalid physical block {physical_block}")
    return state_cache[physical_block, absolute_position % COMPRESS_STATE_BLOCK_SIZE, 0]


def _online_softmax_add(
    running_max: torch.Tensor,
    denominator: torch.Tensor,
    numerator: torch.Tensor,
    score: torch.Tensor,
    value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    next_max = torch.maximum(running_max, score)
    previous_weight = torch.exp(running_max - next_max)
    new_weight = torch.exp(score - next_max)
    return (
        next_max,
        denominator * previous_weight + new_weight,
        numerator * previous_weight + value * new_weight,
    )


def _normalize_and_rotate(
    pooled: torch.Tensor,
    norm_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    if pooled.shape[-1] != norm_weight.numel():
        raise ValueError("norm_weight width must match the compressor head dimension")
    normalized = _rms_normalize(pooled, norm_weight, epsilon)
    rope_dim = cos.shape[-1]
    nope_dim = normalized.shape[-1] - rope_dim
    return torch.cat(
        (
            normalized[:, :nope_dim],
            _apply_interleaved_rope(normalized[:, nope_dim:], cos, sin),
        ),
        dim=-1,
    )


def _compression_positions(
    positions: torch.Tensor,
    boundary_offsets: torch.Tensor,
) -> torch.Tensor:
    if boundary_offsets.ndim != 2 or boundary_offsets.shape[0] != positions.shape[0]:
        raise ValueError("boundary_offsets must have shape [batch, writes_per_request]")
    if torch.any(boundary_offsets < 0):
        raise ValueError("reference requires every ratio-4 boundary slot to be active")
    boundary_positions = torch.gather(
        positions,
        1,
        boundary_offsets,
    )
    return (boundary_positions + 1 - COMPRESS_RATIO).reshape(-1)


def _write_boundary_cache(
    values: torch.Tensor,
    cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    boundary_offsets: torch.Tensor,
    sequence_length: int,
) -> torch.Tensor:
    _require_cpu_tensors({"cache": cache, "slot_mapping": slot_mapping})
    if cache.ndim != 4 or cache.shape[2] != 1 or cache.shape[3] != values.shape[1]:
        raise ValueError("cache must have shape [blocks, page_size, 1, head_dim]")
    if boundary_offsets.ndim != 2:
        raise ValueError("boundary_offsets must have shape [batch, writes_per_request]")
    batch, writes_per_request = boundary_offsets.shape
    if values.shape[0] != batch * writes_per_request:
        raise ValueError("values rows must match flattened boundary_offsets")
    expected_slots = batch * sequence_length
    if slot_mapping.numel() != expected_slots:
        raise ValueError(f"slot_mapping must contain {expected_slots} rows")
    rows = torch.full((batch, writes_per_request), -1, dtype=torch.int64)
    flat_slots = slot_mapping.reshape(-1)
    for request in range(batch):
        for write_index in range(writes_per_request):
            offset = int(boundary_offsets[request, write_index])
            if offset < 0:
                continue
            row = int(flat_slots[request * sequence_length + offset])
            if row < 0:
                continue
            block, intra = divmod(row, cache.shape[1])
            if block >= cache.shape[0]:
                raise IndexError(f"cache row {row} exceeds {cache.shape[0]} pages")
            value_row = request * writes_per_request + write_index
            cache[block, intra, 0].copy_(values[value_row].to(cache.dtype))
            rows[request, write_index] = row
    return rows


def _write_indexer_boundary_cache(
    quantized: torch.Tensor,
    scales: torch.Tensor,
    cache: torch.Tensor,
    scale_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    boundary_offsets: torch.Tensor,
    sequence_length: int,
) -> torch.Tensor:
    if cache.dtype != torch.int8:
        raise TypeError(f"indexer cache must use torch.int8, got {cache.dtype}")
    if scale_cache.dtype != torch.float16:
        raise TypeError(f"indexer scale cache must use torch.float16, got {scale_cache.dtype}")
    if cache.shape[:3] != scale_cache.shape[:3] or scale_cache.shape[3] != 1:
        raise ValueError("indexer K and scale caches must share [blocks, page_size, 1]")
    rows = _write_boundary_cache(
        quantized,
        cache,
        slot_mapping,
        boundary_offsets,
        sequence_length,
    )
    flat_slots = slot_mapping.reshape(-1)
    batch, writes_per_request = boundary_offsets.shape
    for request in range(batch):
        for write_index in range(writes_per_request):
            offset = int(boundary_offsets[request, write_index])
            if offset < 0:
                continue
            row = int(flat_slots[request * sequence_length + offset])
            if row < 0:
                continue
            block, intra = divmod(row, scale_cache.shape[1])
            value_row = request * writes_per_request + write_index
            scale_cache[block, intra, 0, 0] = scales[value_row, 0].to(torch.float16)
    return rows


__all__ = [
    "DEFAULT_DECODE_SEQUENCE_LENGTH",
    "IndexerReferenceResult",
    "QKVReferenceResult",
    "Ratio4CompressorResult",
    "Ratio4IndexerCompressorResult",
    "SparseAttentionReferenceResult",
    "indexer_reference",
    "qkv_projection_rope_reference",
    "ratio4_compressor_reference",
    "ratio4_indexer_compressor_reference",
    "restore_decode_positions",
    "sparse_attention_csa_reference",
]
