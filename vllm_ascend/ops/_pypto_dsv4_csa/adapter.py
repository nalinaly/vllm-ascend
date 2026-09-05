# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""A3 cache bindings for the DeepSeek V4 decode CSA PyPTO L1 entry.

This module intentionally has no PyPTO dependency.  It validates the concrete
vLLM cache tuple and prepares capture-stable, zero-copy tensor aliases before a
program is warmed up or captured.  The launch path only reads the immutable
mapping retained by :class:`PreparedDecodeCSACaches`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TypeAlias

import torch

from .config import BLOCK_SIZE, FLASH
from .contract import DecodeCSAProgramSpec
from .physical_storage import (
    PreparedPhysicalStorageAlias,
    prepare_physical_storage_alias,
)


class DecodeCSACacheContractError(ValueError):
    """Raised when a vLLM cache tuple is incompatible with a static program."""


LaunchValue: TypeAlias = torch.Tensor | int


@dataclass(frozen=True, slots=True)
class PreparedDecodeCSACaches:
    """Long-lived owner of one validated A3 cache binding.

    Four page-strided sources need canonical physical-storage aliases because
    an outlined AICore child does not consume runtime ``ChipTensor`` strides.
    The two packed BF16 caches are retained directly.  ``launch_arguments`` is
    built exactly once, outside capture, and includes the four scalar ABI slots
    that remain in a compiled L1 callable even though their literal values are
    folded into child-kernel address expressions.
    """

    spec: DecodeCSAProgramSpec
    source_caches: tuple[torch.Tensor, ...]
    compressed_kv: torch.Tensor
    swa_kv: torch.Tensor
    main_compressor: PreparedPhysicalStorageAlias
    inner_compressor: PreparedPhysicalStorageAlias
    indexer_k: PreparedPhysicalStorageAlias
    indexer_scale: PreparedPhysicalStorageAlias
    launch_arguments: Mapping[str, LaunchValue]

    def for_launch(self) -> Mapping[str, LaunchValue]:
        """Return the same immutable name-to-argument mapping on every call."""
        return self.launch_arguments


def _require_tensor(
    value: object,
    *,
    name: str,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise DecodeCSACacheContractError(f"{name}: expected torch.Tensor, got {type(value).__name__}")
    return value


def _require_tensor_contract(
    tensor: torch.Tensor,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    strides: tuple[int, ...],
) -> None:
    actual_shape = tuple(int(extent) for extent in tensor.shape)
    actual_strides = tuple(int(stride) for stride in tensor.stride())
    if actual_shape != shape:
        raise DecodeCSACacheContractError(f"{name}: expected shape={shape}, got {actual_shape}")
    if tensor.dtype != dtype:
        raise DecodeCSACacheContractError(f"{name}: expected dtype={dtype}, got {tensor.dtype}")
    if actual_strides != strides:
        raise DecodeCSACacheContractError(
            f"{name}: expected strides={strides}, got {actual_strides}; "
            "the A3 L1 entry does not accept arbitrary runtime strides"
        )


def _require_same_device(named_tensors: Mapping[str, torch.Tensor]) -> None:
    first_name, first_tensor = next(iter(named_tensors.items()))
    expected = first_tensor.device
    mismatches = {name: tensor.device for name, tensor in named_tensors.items() if tensor.device != expected}
    if mismatches:
        raise DecodeCSACacheContractError(
            f"all CSA caches must share one device; {first_name} is on {expected}, mismatches={mismatches}"
        )


def prepare_decode_csa_caches(
    cache_tuple: tuple[object, ...] | list[object],
    spec: DecodeCSAProgramSpec,
) -> PreparedDecodeCSACaches:
    """Validate and bind the six-cache A3 ratio-4 tuple without copying data.

    The accepted tuple order is the public vLLM DSA order:

    ``(compressed KV, SWA KV, main state, inner state, indexer K, scale)``.

    This function may create Torch *metadata* for four aliases, so callers must
    invoke it during adapter prepare/warmup, never from an L1 launch or inside
    ACLGraph capture.  It performs no device allocation and no data copy.
    """
    if not isinstance(cache_tuple, (tuple, list)):
        raise DecodeCSACacheContractError("cache_tuple must be a tuple or list in vLLM DSA cache order")
    if len(cache_tuple) != 6:
        raise DecodeCSACacheContractError(
            f"ratio-4 decode CSA requires exactly 6 cache tensors, got {len(cache_tuple)}"
        )

    names = (
        "compressed_kv",
        "swa_kv",
        "main_compressor",
        "inner_compressor",
        "indexer_k",
        "indexer_scale",
    )
    tensors = {name: _require_tensor(value, name=name) for name, value in zip(names, cache_tuple, strict=True)}
    _require_same_device(tensors)

    head_dim = FLASH.head_dim
    index_head_dim = FLASH.index_head_dim
    physical = spec.physical_layout
    packed_cache_strides = (BLOCK_SIZE * head_dim, head_dim, head_dim, 1)

    _require_tensor_contract(
        tensors["compressed_kv"],
        name="compressed_kv",
        shape=(spec.compressed_blocks, BLOCK_SIZE, 1, head_dim),
        dtype=torch.bfloat16,
        strides=packed_cache_strides,
    )
    _require_tensor_contract(
        tensors["swa_kv"],
        name="swa_kv",
        shape=(spec.swa_blocks, BLOCK_SIZE, 1, head_dim),
        dtype=torch.bfloat16,
        strides=packed_cache_strides,
    )
    main_raw_strides = (
        physical.main_state_strides[0],
        physical.main_state_strides[1],
        physical.main_state_strides[1],
        1,
    )
    inner_raw_strides = (
        physical.inner_state_strides[0],
        physical.inner_state_strides[1],
        physical.inner_state_strides[1],
        1,
    )
    _require_tensor_contract(
        tensors["main_compressor"],
        name="main_compressor",
        shape=(spec.main_state_blocks, 2, 1, 4 * head_dim),
        dtype=torch.float32,
        strides=main_raw_strides,
    )
    _require_tensor_contract(
        tensors["inner_compressor"],
        name="inner_compressor",
        shape=(spec.inner_state_blocks, 2, 1, 4 * index_head_dim),
        dtype=torch.float32,
        strides=inner_raw_strides,
    )
    _require_tensor_contract(
        tensors["indexer_k"],
        name="indexer_k",
        shape=(spec.indexer_blocks, BLOCK_SIZE, 1, index_head_dim),
        dtype=torch.int8,
        strides=physical.indexer_k_strides,
    )
    _require_tensor_contract(
        tensors["indexer_scale"],
        name="indexer_scale",
        shape=(spec.indexer_blocks, BLOCK_SIZE, 1, 1),
        dtype=torch.float16,
        strides=physical.indexer_scale_strides,
    )

    # Native dsa_v1 passes these two state caches to the compressor only after
    # ``squeeze(-2)``.  Mirror that public-tuple adaptation once during
    # prepare, then retain the resulting 3-D views through the alias owners.
    main_state_view = tensors["main_compressor"].squeeze(-2)
    inner_state_view = tensors["inner_compressor"].squeeze(-2)
    if tuple(main_state_view.stride()) != physical.main_state_strides:
        raise DecodeCSACacheContractError("main_compressor: squeeze(-2) did not produce the static state layout")
    if tuple(inner_state_view.stride()) != physical.inner_state_strides:
        raise DecodeCSACacheContractError("inner_compressor: squeeze(-2) did not produce the static state layout")
    main = prepare_physical_storage_alias(main_state_view, name="main_compressor")
    inner = prepare_physical_storage_alias(inner_state_view, name="inner_compressor")
    indexer_k = prepare_physical_storage_alias(tensors["indexer_k"], name="indexer_k")
    indexer_scale = prepare_physical_storage_alias(tensors["indexer_scale"], name="indexer_scale")

    expected_spans = {
        "main_compressor": (main.layout.physical_span, spec.main_state_span),
        "inner_compressor": (inner.layout.physical_span, spec.inner_state_span),
        "indexer_k": (indexer_k.layout.physical_span, spec.indexer_k_span),
        "indexer_scale": (indexer_scale.layout.physical_span, spec.indexer_scale_span),
    }
    mismatched_spans = {
        name: {"actual": actual, "expected": expected}
        for name, (actual, expected) in expected_spans.items()
        if actual != expected
    }
    if mismatched_spans:
        raise DecodeCSACacheContractError(f"physical alias spans do not match the static program: {mismatched_spans}")

    launch_arguments: Mapping[str, LaunchValue] = MappingProxyType(
        {
            "cmp_kv": tensors["compressed_kv"],
            "kv_cache": tensors["swa_kv"],
            "compress_state": main.for_launch(),
            "inner_compress_state": inner.for_launch(),
            "idx_kv_cache": indexer_k.for_launch(),
            "idx_kv_scale": indexer_scale.for_launch(),
            "main_state_page_stride": physical.main_state_strides[0],
            "inner_state_page_stride": physical.inner_state_strides[0],
            "indexer_k_page_stride": physical.indexer_k_strides[0],
            "indexer_scale_page_stride": physical.indexer_scale_strides[0],
        }
    )
    return PreparedDecodeCSACaches(
        spec=spec,
        source_caches=tuple(tensors[name] for name in names),
        compressed_kv=tensors["compressed_kv"],
        swa_kv=tensors["swa_kv"],
        main_compressor=main,
        inner_compressor=inner,
        indexer_k=indexer_k,
        indexer_scale=indexer_scale,
        launch_arguments=launch_arguments,
    )


__all__ = [
    "DecodeCSACacheContractError",
    "PreparedDecodeCSACaches",
    "prepare_decode_csa_caches",
]
