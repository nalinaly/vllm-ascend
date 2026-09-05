# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Host-only tests for the parameterized decode CSA mathematical reference."""

from __future__ import annotations

import math

import torch

from tests.pypto_dsv4_decode_csa.reference import (
    _gather_interleaved_rope,
    indexer_reference,
    qkv_projection_rope_reference,
    ratio4_compressor_reference,
    ratio4_indexer_compressor_reference,
    restore_decode_positions,
    sparse_attention_csa_reference,
)

SEQUENCE_LENGTH = 8


def _pattern(shape: tuple[int, ...], *, scale: float, shift: int = 0, dtype: torch.dtype) -> torch.Tensor:
    values = torch.arange(math.prod(shape), dtype=torch.float32).reshape(shape)
    values = ((values + shift).remainder(19) - 9) * scale
    return values.to(dtype)


def _rope_tables(max_position: int, rope_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    positions = torch.arange(max_position, dtype=torch.float32).view(-1, 1)
    frequencies = (torch.arange(rope_dim, dtype=torch.int64) // 2 + 1).to(torch.float32).view(1, -1)
    angles = positions * frequencies * 0.071
    # Native full caches are already interleaved across their full width.
    return (
        torch.cos(angles).to(torch.bfloat16),
        torch.sin(angles).to(torch.bfloat16),
    )


def _strided_state(
    *,
    blocks: int,
    head_dim: int,
    padding: int,
    canary: float,
) -> tuple[torch.Tensor, torch.Tensor, set[int]]:
    width = 4 * head_dim
    page_stride = 2 * width + padding
    span = (blocks - 1) * page_stride + 2 * width
    storage = torch.full((span,), canary, dtype=torch.float32)
    view = torch.as_strided(
        storage,
        size=(blocks, 2, 1, width),
        stride=(page_stride, width, width, 1),
    )
    initial = _pattern(tuple(view.shape), scale=0.013, shift=5, dtype=torch.float32)
    view.copy_(initial)
    logical_offsets = {
        block * page_stride + row * width + column
        for block in range(blocks)
        for row in range(2)
        for column in range(width)
    }
    return storage, view, logical_offsets


def _strided_indexer_caches(
    *,
    blocks: int,
    page_size: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, set[int], set[int]]:
    key_page_stride = page_size * head_dim + 3
    key_span = (blocks - 1) * key_page_stride + page_size * head_dim
    key_storage = torch.full((key_span,), -99, dtype=torch.int8)
    key_cache = torch.as_strided(
        key_storage,
        size=(blocks, page_size, 1, head_dim),
        stride=(key_page_stride, head_dim, head_dim, 1),
    )
    key_offsets = {
        block * key_page_stride + row * head_dim + column
        for block in range(blocks)
        for row in range(page_size)
        for column in range(head_dim)
    }

    scale_page_stride = page_size + 2
    scale_span = (blocks - 1) * scale_page_stride + page_size
    scale_storage = torch.full((scale_span,), -7.0, dtype=torch.float16)
    scale_cache = torch.as_strided(
        scale_storage,
        size=(blocks, page_size, 1, 1),
        stride=(scale_page_stride, 1, 1, 1),
    )
    scale_offsets = {block * scale_page_stride + row for block in range(blocks) for row in range(page_size)}
    return key_storage, key_cache, scale_storage, scale_cache, key_offsets, scale_offsets


def _block_table(*, batch: int, logical_blocks: int, blocks_per_request: int) -> torch.Tensor:
    logical = torch.arange(logical_blocks, dtype=torch.int64).remainder(blocks_per_request)
    bases = torch.arange(batch, dtype=torch.int64).view(-1, 1) * blocks_per_request
    return (bases + logical.view(1, -1)).to(torch.int32)


def _state_slots(positions: torch.Tensor, block_table: torch.Tensor) -> torch.Tensor:
    rows = torch.empty_like(positions, dtype=torch.int64)
    for request in range(positions.shape[0]):
        logical_blocks = positions[request] // 2
        physical_blocks = block_table[request, logical_blocks]
        rows[request] = physical_blocks.to(torch.int64) * 2 + positions[request].remainder(2)
    return rows.reshape(-1)


def _compressed_slots(
    positions: torch.Tensor,
    *,
    page_size: int,
    blocks_per_request: int,
) -> torch.Tensor:
    rows = torch.empty_like(positions, dtype=torch.int64)
    for request in range(positions.shape[0]):
        compressed_positions = positions[request] // 4
        physical_blocks = request * blocks_per_request + compressed_positions // page_size
        rows[request] = physical_blocks * page_size + compressed_positions.remainder(page_size)
    return rows.reshape(-1)


def _assert_padding_is_unchanged(
    storage: torch.Tensor,
    logical_offsets: set[int],
    canary: float,
) -> None:
    padding_offsets = sorted(set(range(storage.numel())) - logical_offsets)
    assert padding_offsets
    expected = torch.full((len(padding_offsets),), canary, dtype=storage.dtype)
    torch.testing.assert_close(storage[padding_offsets], expected, rtol=0, atol=0)


def test_restore_decode_positions_recovers_all_eight_tokens_per_request() -> None:
    positions = restore_decode_positions(torch.tensor([0, 5, 127], dtype=torch.int32))

    assert positions.dtype == torch.int64
    assert positions.tolist() == [
        list(range(0, 8)),
        list(range(5, 13)),
        list(range(127, 135)),
    ]


def test_rope_gather_consumes_full_interleaved_cache_without_second_expansion() -> None:
    freqs_cos = torch.tensor(
        [[2.0, 2.0, 3.0, 3.0], [5.0, 5.0, 7.0, 7.0]],
        dtype=torch.bfloat16,
    )
    freqs_sin = torch.tensor(
        [[11.0, 11.0, 13.0, 13.0], [17.0, 17.0, 19.0, 19.0]],
        dtype=torch.bfloat16,
    )

    token_cos, signed_sin = _gather_interleaved_rope(
        freqs_cos,
        freqs_sin,
        torch.tensor([1, 0], dtype=torch.int64),
        head_dim=4,
    )

    expected_cos = torch.tensor(
        [[5.0, 5.0, 7.0, 7.0], [2.0, 2.0, 3.0, 3.0]],
    )
    expected_sin = torch.tensor(
        [[-17.0, 17.0, -19.0, 19.0], [-11.0, 11.0, -13.0, 13.0]],
    )
    torch.testing.assert_close(token_cos, expected_cos, rtol=0, atol=0)
    torch.testing.assert_close(signed_sin, expected_sin, rtol=0, atol=0)

    legacy_second_expansion = freqs_cos.float()[torch.tensor([1, 0]), :2].repeat_interleave(2, dim=-1)
    assert not torch.equal(token_cos, legacy_second_expansion)


def test_qkv_projection_and_rope_produces_nonzero_bf16_results() -> None:
    batch = 2
    hidden_size = 6
    query_lora_rank = 4
    num_heads = 2
    head_dim = 8
    rope_dim = 4
    tokens = batch * SEQUENCE_LENGTH
    starts = torch.tensor([1, 7], dtype=torch.int32)
    hidden = _pattern((tokens, hidden_size), scale=0.07, shift=2, dtype=torch.bfloat16)
    wq_a = _pattern((hidden_size, query_lora_rank), scale=1.0, shift=3, dtype=torch.int8)
    wq_a_scale = torch.linspace(0.012, 0.028, query_lora_rank)
    wkv = _pattern((hidden_size, head_dim), scale=1.0, shift=7, dtype=torch.int8)
    wkv_scale = torch.linspace(0.009, 0.024, head_dim)
    gamma_cq = torch.linspace(0.7, 1.2, query_lora_rank).to(torch.bfloat16)
    gamma_ckv = torch.linspace(0.6, 1.3, head_dim).to(torch.bfloat16)
    freqs_cos, freqs_sin = _rope_tables(32, rope_dim)

    result = qkv_projection_rope_reference(
        hidden,
        wq_a=wq_a,
        wq_a_scale=wq_a_scale,
        wq_b=_pattern(
            (query_lora_rank, num_heads * head_dim),
            scale=1.0,
            shift=4,
            dtype=torch.int8,
        ),
        wq_b_scale=torch.linspace(0.01, 0.03, num_heads * head_dim),
        wkv=wkv,
        wkv_scale=wkv_scale,
        gamma_cq=gamma_cq,
        gamma_ckv=gamma_ckv,
        freqs_cos=freqs_cos,
        freqs_sin=freqs_sin,
        start_positions=starts,
    )

    assert result.positions.tolist() == [list(range(1, 9)), list(range(7, 15))]
    assert result.query.shape == (tokens, num_heads, head_dim)
    assert result.key_value.shape == (tokens, head_dim)
    assert result.query.dtype == torch.bfloat16
    assert result.key_value.dtype == torch.bfloat16
    assert result.quantized_hidden_states.dtype == torch.int8
    assert result.hidden_dequant_scale.dtype == torch.float32
    assert result.quantized_query_lora.dtype == torch.int8
    assert result.query_lora_scale.dtype == torch.float32
    assert torch.count_nonzero(result.query) > 0
    assert torch.count_nonzero(result.key_value) > 0
    assert torch.count_nonzero(result.quantized_hidden_states) > 0
    assert torch.count_nonzero(result.quantized_query_lora) > 0
    assert torch.all(result.query_lora_scale > 0)
    assert torch.all(result.hidden_dequant_scale > 0)
    assert int(result.quantized_hidden_states.abs().max()) <= 127
    assert int(result.quantized_query_lora.abs().max()) <= 127

    hidden_amax = hidden.float().abs().amax(dim=-1, keepdim=True).clamp_min(1e-4)
    expected_hidden_scale = hidden_amax / 127.0
    expected_hidden_i8 = (
        torch.round(hidden.float() / expected_hidden_scale).to(torch.int32).to(torch.float16).to(torch.int8)
    )
    torch.testing.assert_close(result.hidden_dequant_scale, expected_hidden_scale, rtol=0, atol=0)
    torch.testing.assert_close(result.quantized_hidden_states, expected_hidden_i8, rtol=0, atol=0)

    # The same hidden quantization feeds both INT32 matmuls.  In particular,
    # wq_a dequantization rounds to BF16 before RMSNorm and query-LoRA quant.
    query_lora_acc = expected_hidden_i8.to(torch.int32) @ wq_a.to(torch.int32)
    query_lora_bf16 = (query_lora_acc.float() * expected_hidden_scale * wq_a_scale.view(1, query_lora_rank)).to(
        torch.bfloat16
    )
    inverse_rms = torch.rsqrt(query_lora_bf16.float().square().mean(dim=-1, keepdim=True) + 1e-6)
    normalized_query_lora = query_lora_bf16.float() * inverse_rms * gamma_cq.float().view(1, query_lora_rank)
    normalized_amax = normalized_query_lora.abs().amax(dim=-1, keepdim=True).clamp_min(1e-4)
    expected_query_lora_scale = normalized_amax / 127.0
    expected_query_lora_i8 = (
        torch.round(normalized_query_lora / expected_query_lora_scale).to(torch.int32).to(torch.float16).to(torch.int8)
    )
    torch.testing.assert_close(result.query_lora_scale, expected_query_lora_scale, rtol=0, atol=0)
    torch.testing.assert_close(result.quantized_query_lora, expected_query_lora_i8, rtol=0, atol=0)

    key_value_acc = expected_hidden_i8.to(torch.int32) @ wkv.to(torch.int32)
    key_value_bf16 = (key_value_acc.float() * expected_hidden_scale * wkv_scale.view(1, head_dim)).to(torch.bfloat16)
    key_value_inverse_rms = torch.rsqrt(key_value_bf16.float().square().mean(dim=-1, keepdim=True) + 1e-6)
    normalized_key_value = key_value_bf16.float() * key_value_inverse_rms * gamma_ckv.float().view(1, head_dim)
    flat_positions = result.positions.reshape(-1)
    cos = freqs_cos.float()[flat_positions]
    signed_sin = freqs_sin.float()[flat_positions]
    signed_sin[:, 0::2].neg_()
    swap = torch.arange(rope_dim, dtype=torch.int64) ^ 1
    nope_dim = head_dim - rope_dim
    expected_key_value = torch.cat(
        (
            normalized_key_value[:, :nope_dim],
            normalized_key_value[:, nope_dim:] * cos + normalized_key_value[:, nope_dim:][:, swap] * signed_sin,
        ),
        dim=-1,
    ).to(torch.bfloat16)
    torch.testing.assert_close(result.key_value, expected_key_value, rtol=0, atol=0)

    # Consecutive tokens share weights but use different RoPE rows, so their
    # rotated regions must not collapse to the same value.
    assert not torch.equal(result.key_value[0, -rope_dim:], result.key_value[1, -rope_dim:])


def test_main_ratio4_compressor_writes_rows_without_touching_page_padding() -> None:
    batch = 2
    hidden_size = 6
    head_dim = 4
    rope_dim = 2
    state_blocks = 20
    state_blocks_per_request = state_blocks // batch
    starts = torch.tensor([0, 5], dtype=torch.int32)
    positions = restore_decode_positions(starts)
    tokens = positions.numel()
    state_storage, state, state_offsets = _strided_state(
        blocks=state_blocks,
        head_dim=head_dim,
        padding=5,
        canary=-7777.0,
    )
    state_table = _block_table(
        batch=batch,
        logical_blocks=16,
        blocks_per_request=state_blocks_per_request,
    )
    state_slots = _state_slots(positions, state_table)

    cache_page_size = 4
    cache_blocks_per_request = 2
    compressed_cache = torch.full(
        (batch * cache_blocks_per_request, cache_page_size, 1, head_dim),
        -31.0,
        dtype=torch.bfloat16,
    )
    compressed_slots = _compressed_slots(
        positions,
        page_size=cache_page_size,
        blocks_per_request=cache_blocks_per_request,
    )
    hidden = _pattern((tokens, hidden_size), scale=0.04, shift=1, dtype=torch.bfloat16)
    value_weight = _pattern((2 * head_dim, hidden_size), scale=0.05, shift=3, dtype=torch.bfloat16)
    gate_weight = _pattern((2 * head_dim, hidden_size), scale=0.03, shift=8, dtype=torch.bfloat16)
    positional_bias = _pattern((4, 2 * head_dim), scale=0.02, shift=2, dtype=torch.float32)
    freqs_cos, freqs_sin = _rope_tables(32, rope_dim)

    result = ratio4_compressor_reference(
        hidden,
        state_cache=state,
        state_block_table=state_table,
        state_slot_mapping=state_slots,
        value_weight=value_weight,
        gate_weight=gate_weight,
        positional_bias=positional_bias,
        norm_weight=torch.linspace(0.75, 1.25, head_dim).to(torch.bfloat16),
        freqs_cos=freqs_cos,
        freqs_sin=freqs_sin,
        compressed_cache=compressed_cache,
        compressed_slot_mapping=compressed_slots,
        start_positions=starts,
    )

    assert result.boundary_offsets.tolist() == [[3, 7], [2, 6]]
    assert result.state_rows_written.tolist() == state_slots.tolist()
    expected_cache_rows = [
        [int(compressed_slots[3]), int(compressed_slots[7])],
        [
            int(compressed_slots[SEQUENCE_LENGTH + 2]),
            int(compressed_slots[SEQUENCE_LENGTH + 6]),
        ],
    ]
    assert result.cache_rows_written.tolist() == expected_cache_rows
    assert torch.count_nonzero(result.pooled_values) > 0
    assert torch.isfinite(result.normalized_values).all()

    expected_values = hidden.float() @ value_weight.float().transpose(0, 1)
    expected_scores = hidden.float() @ gate_weight.float().transpose(0, 1)
    expected_scores += positional_bias[positions.reshape(-1).remainder(4)]
    first_state_row = state[int(state_slots[0]) // 2, int(state_slots[0]) % 2, 0]
    torch.testing.assert_close(first_state_row[: 2 * head_dim], expected_values[0], rtol=0, atol=0)
    torch.testing.assert_close(first_state_row[2 * head_dim :], expected_scores[0], rtol=0, atol=0)

    for request, request_rows in enumerate(expected_cache_rows):
        for write_index, row in enumerate(request_rows):
            block, intra = divmod(row, cache_page_size)
            value_row = request * 2 + write_index
            torch.testing.assert_close(
                compressed_cache[block, intra, 0],
                result.normalized_values[value_row].to(torch.bfloat16),
                rtol=0,
                atol=0,
            )
    assert int((compressed_cache != torch.tensor(-31.0, dtype=torch.bfloat16)).any(dim=-1).sum()) == batch * 2
    _assert_padding_is_unchanged(state_storage, state_offsets, -7777.0)


def test_inner_ratio4_compressor_preserves_key_and_fp16_scale_page_padding() -> None:
    batch = 2
    hidden_size = 6
    head_dim = 4
    rope_dim = 2
    state_blocks = 20
    starts = torch.tensor([0, 5], dtype=torch.int64)
    positions = restore_decode_positions(starts)
    tokens = positions.numel()
    state_storage, state, state_offsets = _strided_state(
        blocks=state_blocks,
        head_dim=head_dim,
        padding=7,
        canary=-8888.0,
    )
    state_table = _block_table(
        batch=batch,
        logical_blocks=16,
        blocks_per_request=state_blocks // batch,
    )
    state_slots = _state_slots(positions, state_table)

    page_size = 4
    blocks_per_request = 2
    key_storage, key_cache, scale_storage, scale_cache, key_offsets, scale_offsets = _strided_indexer_caches(
        blocks=batch * blocks_per_request,
        page_size=page_size,
        head_dim=head_dim,
    )
    indexer_slots = _compressed_slots(
        positions,
        page_size=page_size,
        blocks_per_request=blocks_per_request,
    )
    freqs_cos, freqs_sin = _rope_tables(32, rope_dim)

    result = ratio4_indexer_compressor_reference(
        _pattern((tokens, hidden_size), scale=0.045, shift=4, dtype=torch.bfloat16),
        state_cache=state,
        state_block_table=state_table,
        state_slot_mapping=state_slots,
        value_weight=_pattern((2 * head_dim, hidden_size), scale=0.06, shift=2, dtype=torch.bfloat16),
        gate_weight=_pattern((2 * head_dim, hidden_size), scale=0.025, shift=6, dtype=torch.bfloat16),
        positional_bias=_pattern((4, 2 * head_dim), scale=0.015, shift=9, dtype=torch.float32),
        norm_weight=torch.linspace(0.8, 1.1, head_dim).to(torch.bfloat16),
        freqs_cos=freqs_cos,
        freqs_sin=freqs_sin,
        hadamard=torch.eye(head_dim, dtype=torch.bfloat16),
        indexer_cache=key_cache,
        indexer_scale=scale_cache,
        indexer_slot_mapping=indexer_slots,
        start_positions=starts,
    )

    expected_rows = [
        [int(indexer_slots[3]), int(indexer_slots[7])],
        [
            int(indexer_slots[SEQUENCE_LENGTH + 2]),
            int(indexer_slots[SEQUENCE_LENGTH + 6]),
        ],
    ]
    assert result.compressor.cache_rows_written.tolist() == expected_rows
    assert result.normalized_values.dtype == torch.bfloat16
    assert result.quantized_values.dtype == torch.int8
    assert result.dequant_scales.dtype == torch.float32
    assert torch.count_nonzero(result.transformed_values) > 0
    assert torch.all(result.dequant_scales > 0)

    for request, request_rows in enumerate(expected_rows):
        for write_index, row in enumerate(request_rows):
            block, intra = divmod(row, page_size)
            value_row = request * 2 + write_index
            torch.testing.assert_close(
                key_cache[block, intra, 0],
                result.quantized_values[value_row],
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                scale_cache[block, intra, 0, 0],
                result.dequant_scales[value_row, 0].to(torch.float16),
                rtol=0,
                atol=0,
            )
    _assert_padding_is_unchanged(state_storage, state_offsets, -8888.0)
    _assert_padding_is_unchanged(key_storage, key_offsets, -99)
    _assert_padding_is_unchanged(scale_storage, scale_offsets, -7.0)


def test_indexer_and_sparse_attention_form_a_nonzero_host_only_pipeline() -> None:
    batch = 2
    hidden_size = 6
    query_lora_rank = 4
    num_heads = 2
    head_dim = 8
    rope_dim = 4
    tokens = batch * SEQUENCE_LENGTH
    starts = torch.tensor([0, 5], dtype=torch.int32)
    positions = restore_decode_positions(starts)
    hidden = _pattern((tokens, hidden_size), scale=0.035, shift=3, dtype=torch.bfloat16)
    freqs_cos, freqs_sin = _rope_tables(32, rope_dim)
    qkv = qkv_projection_rope_reference(
        hidden,
        wq_a=_pattern((hidden_size, query_lora_rank), scale=1.0, shift=1, dtype=torch.int8),
        wq_a_scale=torch.linspace(0.009, 0.021, query_lora_rank),
        wq_b=_pattern(
            (query_lora_rank, num_heads * head_dim),
            scale=1.0,
            shift=4,
            dtype=torch.int8,
        ),
        wq_b_scale=torch.linspace(0.01, 0.025, num_heads * head_dim),
        wkv=_pattern((hidden_size, head_dim), scale=1.0, shift=8, dtype=torch.int8),
        wkv_scale=torch.linspace(0.007, 0.019, head_dim),
        gamma_cq=torch.linspace(0.8, 1.15, query_lora_rank).to(torch.bfloat16),
        gamma_ckv=torch.linspace(0.7, 1.25, head_dim).to(torch.bfloat16),
        freqs_cos=freqs_cos,
        freqs_sin=freqs_sin,
        start_positions=starts,
    )

    state_blocks = 20
    _, inner_state, _ = _strided_state(
        blocks=state_blocks,
        head_dim=head_dim,
        padding=5,
        canary=-9999.0,
    )
    inner_state_table = _block_table(
        batch=batch,
        logical_blocks=16,
        blocks_per_request=state_blocks // batch,
    )
    inner_state_slots = _state_slots(positions, inner_state_table)

    page_size = 4
    blocks_per_request = 2
    _, indexer_cache, _, indexer_scale, _, _ = _strided_indexer_caches(
        blocks=batch * blocks_per_request,
        page_size=page_size,
        head_dim=head_dim,
    )
    indexer_cache.copy_(_pattern(tuple(indexer_cache.shape), scale=1.0, shift=5, dtype=torch.int8))
    indexer_scale.fill_(0.03125)
    indexer_slots = _compressed_slots(
        positions,
        page_size=page_size,
        blocks_per_request=blocks_per_request,
    )
    indexer_block_table = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)

    indexer = indexer_reference(
        hidden,
        quantized_query_lora=qkv.quantized_query_lora,
        query_lora_scale=qkv.query_lora_scale,
        query_weight=_pattern(
            (query_lora_rank, num_heads * head_dim),
            scale=1.0,
            shift=7,
            dtype=torch.int8,
        ),
        query_weight_scale=torch.linspace(0.008, 0.018, num_heads * head_dim),
        head_weight_projection=_pattern((hidden_size, num_heads), scale=0.04, shift=3, dtype=torch.bfloat16),
        freqs_cos=freqs_cos,
        freqs_sin=freqs_sin,
        hadamard=torch.eye(head_dim, dtype=torch.bfloat16),
        inner_state_cache=inner_state,
        inner_state_block_table=inner_state_table,
        inner_state_slot_mapping=inner_state_slots,
        inner_value_weight=_pattern((2 * head_dim, hidden_size), scale=0.04, shift=2, dtype=torch.bfloat16),
        inner_gate_weight=_pattern((2 * head_dim, hidden_size), scale=0.025, shift=9, dtype=torch.bfloat16),
        inner_positional_bias=_pattern((4, 2 * head_dim), scale=0.013, shift=6, dtype=torch.float32),
        inner_norm_weight=torch.linspace(0.75, 1.2, head_dim).to(torch.bfloat16),
        indexer_cache=indexer_cache,
        indexer_scale=indexer_scale,
        indexer_block_table=indexer_block_table,
        indexer_slot_mapping=indexer_slots,
        sequence_lengths=positions[:, -1].to(torch.int32) + 1,
        start_positions=starts,
        topk_count=3,
    )

    assert indexer.scores.shape == (tokens, 8)
    assert indexer.topk_indices.shape == (tokens, 8)
    assert indexer.quantized_queries.dtype == torch.int8
    assert torch.count_nonzero(indexer.quantized_queries) > 0
    assert torch.count_nonzero(indexer.head_weights) > 0
    assert indexer.topk_indices[0].eq(-1).all()
    assert indexer.topk_indices[3, 0] == 0
    assert indexer.topk_indices[3, 1:].eq(-1).all()
    assert torch.isfinite(indexer.scores[3, 0])

    swa_cache = _pattern(
        (batch * blocks_per_request, page_size, 1, head_dim),
        scale=0.04,
        shift=2,
        dtype=torch.bfloat16,
    )
    compressed_cache = _pattern(
        (batch * blocks_per_request, page_size, 1, head_dim),
        scale=0.03,
        shift=11,
        dtype=torch.bfloat16,
    )
    window_indices = torch.full((tokens, 3), -1, dtype=torch.int32)
    for request in range(batch):
        request_base = request * blocks_per_request * page_size
        for token_offset in range(SEQUENCE_LENGTH):
            token = request * SEQUENCE_LENGTH + token_offset
            visible_start = max(0, token_offset - 2)
            visible = torch.arange(visible_start, token_offset + 1, dtype=torch.int32)
            window_indices[token, : visible.numel()] = request_base + visible

    num_groups = 2
    output_lora_rank = 3
    output_lora_weight = _pattern(
        (num_groups, output_lora_rank, head_dim),
        scale=0.045,
        shift=3,
        dtype=torch.bfloat16,
    )
    output_weight = _pattern(
        (hidden_size, num_groups * output_lora_rank),
        scale=0.028,
        shift=6,
        dtype=torch.bfloat16,
    )
    sparse = sparse_attention_csa_reference(
        qkv.query,
        swa_cache=swa_cache,
        window_swa_indices=window_indices,
        compressed_cache=compressed_cache,
        compressed_block_table=indexer_block_table,
        compressed_topk_indices=indexer.topk_indices,
        attention_sink=torch.tensor([0.2, -0.1]),
        freqs_cos=freqs_cos,
        freqs_sin=freqs_sin,
        output_lora_weight=output_lora_weight,
        output_weight=output_weight,
        start_positions=starts,
        attention_tile=2,
    )

    assert sparse.attention_heads.shape == (tokens, num_heads, head_dim)
    assert sparse.output.shape == (tokens, hidden_size)
    assert sparse.attention_heads.dtype == torch.bfloat16
    assert sparse.projected_groups.dtype == torch.bfloat16
    assert sparse.output.dtype == torch.bfloat16
    assert torch.count_nonzero(sparse.attention_heads) > 0
    assert torch.count_nonzero(sparse.output) > 0

    # wo_a accumulates in FP32 but rounds to BF16 before the BF16 wo_b matmul;
    # each group contributes an FP32 partial and the final sum rounds once.
    expected_output = torch.zeros((tokens, hidden_size), dtype=torch.float32)
    for group in range(num_groups):
        first_head = group * (num_heads // num_groups)
        packed = sparse.attention_heads[:, first_head : first_head + 1].reshape(tokens, head_dim)
        projected = (packed.float() @ output_lora_weight[group].float().transpose(0, 1)).to(torch.bfloat16)
        torch.testing.assert_close(sparse.projected_groups[:, group], projected, rtol=0, atol=0)
        weight_start = group * output_lora_rank
        weight_slice = output_weight[:, weight_start : weight_start + output_lora_rank]
        expected_output += projected.float() @ weight_slice.float().transpose(0, 1)
    torch.testing.assert_close(sparse.output, expected_output.to(torch.bfloat16), rtol=0, atol=0)
