# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Host-side contract for static DeepSeek V4 decode CSA specializations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

from .config import (
    BLOCK_SIZE,
    CSA_INNER_STATE_PHYSICAL_BLOCKS,
    CSA_STATE_PHYSICAL_BLOCKS,
    DECODE_BATCH,
    DECODE_SEQ,
    FLASH,
    IDX_CACHE_BLOCK_NUM,
    KV_CMP_BLOCK_NUM,
    KV_CMP_MAX_BLOCKS,
    KV_ORI_BLOCK_NUM,
    KV_ORI_MAX_BLOCKS,
    TP,
)

SUPPORTED_BATCH_BUCKETS = (4, 8, 12, 16)

_COMPRESS_OUTPUT_FACTOR = 2
_MAIN_STATE_DIM = 2 * _COMPRESS_OUTPUT_FACTOR * FLASH.head_dim
_INNER_STATE_DIM = 2 * _COMPRESS_OUTPUT_FACTOR * FLASH.index_head_dim


@dataclass(frozen=True, slots=True)
class DecodeCSAPhysicalLayout:
    """Static physical layout of the four page-strided A3 cache views.

    The first stride of each tuple is a physical page stride, measured in
    elements of that tensor's dtype.  Dimensions inside a page must remain
    canonical and contiguous.  This is the exact contract consumed by the
    flat-storage L1 entry; arbitrary Torch strides are deliberately rejected
    before capture rather than being silently interpreted as packed tensors by
    an outlined AICore child.
    """

    main_state_strides: tuple[int, int, int] = (8192, _MAIN_STATE_DIM, 1)
    inner_state_strides: tuple[int, int, int] = (1040, _INNER_STATE_DIM, 1)
    indexer_k_strides: tuple[int, int, int, int] = (
        4160,
        FLASH.index_head_dim,
        FLASH.index_head_dim,
        1,
    )
    indexer_scale_strides: tuple[int, int, int, int] = (2080, 1, 1, 1)
    version: int = 1

    def __post_init__(self) -> None:
        if self.version != 1:
            raise ValueError(f"unsupported CSA physical layout version: {self.version}")

        expected_tails = {
            "main_state_strides": (_MAIN_STATE_DIM, 1),
            "inner_state_strides": (_INNER_STATE_DIM, 1),
            "indexer_k_strides": (FLASH.index_head_dim, FLASH.index_head_dim, 1),
            "indexer_scale_strides": (1, 1, 1),
        }
        for name, expected_tail in expected_tails.items():
            strides = getattr(self, name)
            if any(isinstance(stride, bool) or not isinstance(stride, int) or stride <= 0 for stride in strides):
                raise ValueError(f"{name} must contain only positive integer strides, got {strides}")
            if strides[1:] != expected_tail:
                raise ValueError(
                    f"{name} must be contiguous within each physical page: "
                    f"expected tail {expected_tail}, got {strides[1:]}"
                )

        minimum_page_strides = {
            "main_state_strides": 2 * _MAIN_STATE_DIM,
            "inner_state_strides": 2 * _INNER_STATE_DIM,
            "indexer_k_strides": BLOCK_SIZE * FLASH.index_head_dim,
            "indexer_scale_strides": BLOCK_SIZE,
        }
        for name, minimum in minimum_page_strides.items():
            page_stride = getattr(self, name)[0]
            if page_stride < minimum:
                raise ValueError(
                    f"{name} page stride must cover one logical page: expected >= {minimum}, got {page_stride}"
                )

        # A3 stores indexer K (INT8) and scale (FP16) in the same physical
        # 4160-byte page.  Their element strides must describe one page size.
        if self.indexer_k_strides[0] != 2 * self.indexer_scale_strides[0]:
            raise ValueError(
                "indexer K/scale page strides describe different byte sizes: "
                f"k={self.indexer_k_strides[0]}B, "
                f"scale={2 * self.indexer_scale_strides[0]}B"
            )

    @staticmethod
    def _physical_span(blocks: int, page_stride: int, logical_page_elements: int) -> int:
        return (blocks - 1) * page_stride + logical_page_elements

    def main_state_span(self, blocks: int) -> int:
        return self._physical_span(blocks, self.main_state_strides[0], 2 * _MAIN_STATE_DIM)

    def inner_state_span(self, blocks: int) -> int:
        return self._physical_span(blocks, self.inner_state_strides[0], 2 * _INNER_STATE_DIM)

    def indexer_k_span(self, blocks: int) -> int:
        return self._physical_span(
            blocks,
            self.indexer_k_strides[0],
            BLOCK_SIZE * FLASH.index_head_dim,
        )

    def indexer_scale_span(self, blocks: int) -> int:
        return self._physical_span(blocks, self.indexer_scale_strides[0], BLOCK_SIZE)

    @property
    def key(self) -> tuple[object, ...]:
        return (
            self.version,
            self.main_state_strides,
            self.inner_state_strides,
            self.indexer_k_strides,
            self.indexer_scale_strides,
        )


@dataclass(frozen=True, slots=True)
class DecodeCSAProgramSpec:
    """Compile-time shape identity for one PyPTO L1 program.

    PyPTO L1 v1 accepts only positive static tensor extents.  A separate
    callable is therefore compiled for every batch bucket; addresses may vary
    between eager calls without changing this identity.
    """

    batch: int
    seq: int = DECODE_SEQ
    swa_blocks: int = KV_ORI_BLOCK_NUM
    compressed_blocks: int = KV_CMP_BLOCK_NUM
    main_state_blocks: int = CSA_STATE_PHYSICAL_BLOCKS
    inner_state_blocks: int = CSA_INNER_STATE_PHYSICAL_BLOCKS
    indexer_blocks: int = IDX_CACHE_BLOCK_NUM
    # vLLM owns these block-table tensors.  Their row widths are determined by
    # the model/cache configuration, not by the number of physical pages that
    # happened to be allocated.  They therefore belong to the static callable
    # identity and are validated independently from the five capacities above.
    #
    # Ratio-4 compressed/indexer tables cover one entry per four model tokens;
    # SWA covers one per model token; compressor state uses two-token pages.
    swa_table_width: int = KV_ORI_MAX_BLOCKS
    compressed_table_width: int = KV_CMP_MAX_BLOCKS
    main_state_table_width: int = FLASH.max_position_embeddings // 2
    inner_state_table_width: int = FLASH.max_position_embeddings // 2
    indexer_table_width: int = KV_CMP_MAX_BLOCKS
    # The outlined AICore child does not consume runtime ChipTensor strides.
    # Page strides therefore participate in the static program identity and
    # are also used to size the canonical physical-storage aliases.
    physical_layout: DecodeCSAPhysicalLayout = DecodeCSAPhysicalLayout()

    def __post_init__(self) -> None:
        max_batch = DECODE_BATCH // TP
        if self.batch not in SUPPORTED_BATCH_BUCKETS or self.batch > max_batch:
            raise ValueError(f"batch must be one of {SUPPORTED_BATCH_BUCKETS} and <= {max_batch}, got {self.batch}")
        if self.seq != DECODE_SEQ:
            raise ValueError(f"decode CSA is specialized for seq={DECODE_SEQ}, got {self.seq}")
        block_counts = {
            "swa_blocks": self.swa_blocks,
            "compressed_blocks": self.compressed_blocks,
            "main_state_blocks": self.main_state_blocks,
            "inner_state_blocks": self.inner_state_blocks,
            "indexer_blocks": self.indexer_blocks,
        }
        invalid = {name: value for name, value in block_counts.items() if value <= 0}
        if invalid:
            raise ValueError(f"cache block counts must be positive, got {invalid}")

        table_widths = {
            "swa_table_width": self.swa_table_width,
            "compressed_table_width": self.compressed_table_width,
            "main_state_table_width": self.main_state_table_width,
            "inner_state_table_width": self.inner_state_table_width,
            "indexer_table_width": self.indexer_table_width,
        }
        invalid_widths = {
            name: value
            for name, value in table_widths.items()
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0
        }
        if invalid_widths:
            raise ValueError(f"block-table widths must be positive integers, got {invalid_widths}")

    @property
    def tokens(self) -> int:
        return self.batch * self.seq

    @property
    def main_state_span(self) -> int:
        return self.physical_layout.main_state_span(self.main_state_blocks)

    @property
    def inner_state_span(self) -> int:
        return self.physical_layout.inner_state_span(self.inner_state_blocks)

    @property
    def indexer_k_span(self) -> int:
        return self.physical_layout.indexer_k_span(self.indexer_blocks)

    @property
    def indexer_scale_span(self) -> int:
        return self.physical_layout.indexer_scale_span(self.indexer_blocks)

    @property
    def key(self) -> tuple[object, ...]:
        return (
            self.batch,
            self.seq,
            self.swa_blocks,
            self.compressed_blocks,
            self.main_state_blocks,
            self.inner_state_blocks,
            self.indexer_blocks,
            self.swa_table_width,
            self.compressed_table_width,
            self.main_state_table_width,
            self.inner_state_table_width,
            self.indexer_table_width,
            self.physical_layout.key,
        )


class MutableCSAState(NamedTuple):
    """The six state families mutated by one A3 ratio-4 CSA invocation."""

    compressed_kv: object
    swa_kv: object
    main_compressor: object
    inner_compressor: object
    indexer_k: object
    indexer_scale: object


MUTABLE_ARGUMENT_NAMES = frozenset(
    {
        "compress_state",
        "inner_compress_state",
        "kv_cache",
        "cmp_kv",
        "idx_kv_cache",
        "idx_kv_scale",
    }
)

OUTPUT_ARGUMENT_NAME = "attn_out"
