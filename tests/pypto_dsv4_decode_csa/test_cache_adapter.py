# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Host-only tests for the concrete A3 six-cache adapter binding."""

from __future__ import annotations

import math

import pytest
import torch

from vllm_ascend.ops._pypto_dsv4_csa.adapter import (
    DecodeCSACacheContractError,
    prepare_decode_csa_caches,
)
from vllm_ascend.ops._pypto_dsv4_csa.config import BLOCK_SIZE, FLASH
from vllm_ascend.ops._pypto_dsv4_csa.contract import DecodeCSAProgramSpec


def _strided_tensor(
    shape: tuple[int, ...],
    strides: tuple[int, ...],
    dtype: torch.dtype,
    *,
    storage_offset: int = 3,
) -> tuple[torch.Tensor, torch.Tensor]:
    span = 1 + sum((extent - 1) * stride for extent, stride in zip(shape, strides, strict=True))
    storage = torch.empty(storage_offset + span + 5, dtype=dtype)
    view = torch.as_strided(
        storage,
        size=shape,
        stride=strides,
        storage_offset=storage_offset,
    )
    return storage, view


def _small_spec() -> DecodeCSAProgramSpec:
    return DecodeCSAProgramSpec(
        batch=4,
        swa_blocks=2,
        compressed_blocks=3,
        main_state_blocks=3,
        inner_state_blocks=4,
        indexer_blocks=5,
    )


def _valid_cache_tuple(
    spec: DecodeCSAProgramSpec,
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    physical = spec.physical_layout
    owners: list[torch.Tensor] = []

    compressed_kv = torch.empty(
        (spec.compressed_blocks, BLOCK_SIZE, 1, FLASH.head_dim),
        dtype=torch.bfloat16,
    )
    swa_kv = torch.empty(
        (spec.swa_blocks, BLOCK_SIZE, 1, FLASH.head_dim),
        dtype=torch.bfloat16,
    )
    owners.extend((compressed_kv, swa_kv))

    for shape, strides, dtype in (
        (
            (spec.main_state_blocks, 2, 1, 4 * FLASH.head_dim),
            (
                physical.main_state_strides[0],
                physical.main_state_strides[1],
                physical.main_state_strides[1],
                1,
            ),
            torch.float32,
        ),
        (
            (spec.inner_state_blocks, 2, 1, 4 * FLASH.index_head_dim),
            (
                physical.inner_state_strides[0],
                physical.inner_state_strides[1],
                physical.inner_state_strides[1],
                1,
            ),
            torch.float32,
        ),
        (
            (spec.indexer_blocks, BLOCK_SIZE, 1, FLASH.index_head_dim),
            physical.indexer_k_strides,
            torch.int8,
        ),
        (
            (spec.indexer_blocks, BLOCK_SIZE, 1, 1),
            physical.indexer_scale_strides,
            torch.float16,
        ),
    ):
        storage, view = _strided_tensor(shape, strides, dtype)
        owners.append(storage)
        owners.append(view)

    caches = (
        compressed_kv,
        swa_kv,
        owners[3],
        owners[5],
        owners[7],
        owners[9],
    )
    return caches, tuple(owners)


def test_prepares_exact_six_cache_mapping_without_copy() -> None:
    spec = _small_spec()
    caches, owners = _valid_cache_tuple(spec)

    prepared = prepare_decode_csa_caches(caches, spec)
    arguments = prepared.for_launch()

    assert owners  # Keep raw storages explicit for the lifetime of this test.
    assert arguments["cmp_kv"] is caches[0]
    assert arguments["kv_cache"] is caches[1]
    assert arguments["compress_state"] is prepared.main_compressor.launch_tensor
    assert arguments["inner_compress_state"] is prepared.inner_compressor.launch_tensor
    assert arguments["idx_kv_cache"] is prepared.indexer_k.launch_tensor
    assert arguments["idx_kv_scale"] is prepared.indexer_scale.launch_tensor
    assert all(actual is expected for actual, expected in zip(prepared.source_caches, caches, strict=True))
    assert prepared.main_compressor.source.shape == (
        spec.main_state_blocks,
        2,
        4 * FLASH.head_dim,
    )
    assert prepared.inner_compressor.source.shape == (
        spec.inner_state_blocks,
        2,
        4 * FLASH.index_head_dim,
    )

    for source, alias in (
        (caches[2], arguments["compress_state"]),
        (caches[3], arguments["inner_compress_state"]),
        (caches[4], arguments["idx_kv_cache"]),
        (caches[5], arguments["idx_kv_scale"]),
    ):
        assert isinstance(alias, torch.Tensor)
        assert alias.data_ptr() == source.data_ptr()
        assert alias.untyped_storage()._cdata == source.untyped_storage()._cdata

    assert arguments["compress_state"].shape == (1, spec.main_state_span)
    assert arguments["inner_compress_state"].shape == (1, spec.inner_state_span)
    assert arguments["idx_kv_cache"].shape == (1, spec.indexer_k_span)
    assert arguments["idx_kv_scale"].shape == (1, spec.indexer_scale_span)


def test_launch_mapping_is_stable_immutable_and_contains_scalar_abi() -> None:
    spec = _small_spec()
    caches, _ = _valid_cache_tuple(spec)
    prepared = prepare_decode_csa_caches(caches, spec)

    first = prepared.for_launch()
    assert prepared.for_launch() is first
    assert first["main_state_page_stride"] == 8192
    assert first["inner_state_page_stride"] == 1040
    assert first["indexer_k_page_stride"] == 4160
    assert first["indexer_scale_page_stride"] == 2080
    with pytest.raises(TypeError):
        first["main_state_page_stride"] = 1  # type: ignore[index]


def test_for_launch_does_not_recreate_tensor_metadata(monkeypatch) -> None:
    spec = _small_spec()
    caches, _ = _valid_cache_tuple(spec)
    prepared = prepare_decode_csa_caches(caches, spec)

    def fail_if_recreated(*args, **kwargs):
        raise AssertionError("launch must not create a new as_strided tensor")

    monkeypatch.setattr(torch, "as_strided", fail_if_recreated)
    for _ in range(8):
        assert prepared.for_launch() is prepared.launch_arguments


def test_same_program_spec_accepts_different_cache_addresses() -> None:
    spec = _small_spec()
    first_caches, _ = _valid_cache_tuple(spec)
    second_caches, _ = _valid_cache_tuple(spec)

    first = prepare_decode_csa_caches(first_caches, spec)
    second = prepare_decode_csa_caches(second_caches, spec)

    assert first.spec.key == second.spec.key
    assert first.for_launch()["compress_state"].data_ptr() != second.for_launch()["compress_state"].data_ptr()
    assert first.for_launch()["idx_kv_cache"].data_ptr() != second.for_launch()["idx_kv_cache"].data_ptr()


@pytest.mark.parametrize("bad_value", (None, object(), torch.tensor(1.0)))
def test_rejects_missing_or_non_tensor_cache_entries(bad_value: object) -> None:
    spec = _small_spec()
    caches, _ = _valid_cache_tuple(spec)
    invalid = list(caches)
    invalid[2] = bad_value

    with pytest.raises(DecodeCSACacheContractError, match="main_compressor"):
        prepare_decode_csa_caches(invalid, spec)


@pytest.mark.parametrize("length", (0, 5, 7))
def test_rejects_wrong_cache_tuple_length(length: int) -> None:
    with pytest.raises(DecodeCSACacheContractError, match="exactly 6"):
        prepare_decode_csa_caches([object()] * length, _small_spec())


def test_rejects_contiguous_main_state_that_loses_page_stride() -> None:
    spec = _small_spec()
    caches, _ = _valid_cache_tuple(spec)
    invalid = list(caches)
    invalid[2] = torch.empty((spec.main_state_blocks, 2, 1, 4 * FLASH.head_dim), dtype=torch.float32)

    assert invalid[2].stride(0) == 4096
    with pytest.raises(DecodeCSACacheContractError, match="expected strides"):
        prepare_decode_csa_caches(invalid, spec)


def test_rejects_fp32_indexer_scale() -> None:
    spec = _small_spec()
    caches, _ = _valid_cache_tuple(spec)
    invalid = list(caches)
    _, invalid[5] = _strided_tensor(
        (spec.indexer_blocks, BLOCK_SIZE, 1, 1),
        spec.physical_layout.indexer_scale_strides,
        torch.float32,
    )

    with pytest.raises(DecodeCSACacheContractError, match="expected dtype=torch.float16"):
        prepare_decode_csa_caches(invalid, spec)


def test_rejects_wrong_block_capacity_even_when_layout_is_valid() -> None:
    spec = _small_spec()
    caches, _ = _valid_cache_tuple(spec)
    invalid = list(caches)
    _, invalid[4] = _strided_tensor(
        (spec.indexer_blocks + 1, BLOCK_SIZE, 1, FLASH.index_head_dim),
        spec.physical_layout.indexer_k_strides,
        torch.int8,
    )

    with pytest.raises(DecodeCSACacheContractError, match="expected shape"):
        prepare_decode_csa_caches(invalid, spec)


def test_fixture_uses_expected_physical_padding() -> None:
    spec = _small_spec()
    caches, _ = _valid_cache_tuple(spec)
    prepared = prepare_decode_csa_caches(caches, spec)

    expected_main_padding = spec.main_state_span - math.prod(caches[2].shape)
    expected_inner_padding = spec.inner_state_span - math.prod(caches[3].shape)
    assert prepared.main_compressor.layout.padding_elements == expected_main_padding
    assert prepared.inner_compressor.layout.padding_elements == expected_inner_padding
    assert prepared.indexer_k.layout.padding_elements == (spec.indexer_k_span - math.prod(caches[4].shape))
    assert prepared.indexer_scale.layout.padding_elements == (spec.indexer_scale_span - math.prod(caches[5].shape))
