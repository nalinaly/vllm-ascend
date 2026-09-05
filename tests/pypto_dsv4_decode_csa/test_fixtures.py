# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Host tests for deterministic physical-address fixture metadata."""

from __future__ import annotations

import pytest
import torch

from tests.pypto_dsv4_decode_csa.fixtures import (
    _native_swa_slot_mapping,
    _paged_slot_mapping,
    _physical_block_table,
    _position_rows,
    _state_metadata,
    _window_metadata,
)
from vllm_ascend.ops._pypto_dsv4_csa.config import BLOCK_SIZE, FLASH


def test_physical_block_table_partitions_requests_and_wraps_locally() -> None:
    table = _physical_block_table(
        batch=2,
        width=6,
        blocks_per_request=3,
        physical_blocks=6,
    )

    torch.testing.assert_close(
        table,
        torch.tensor(
            [[0, 1, 2, 0, 1, 2], [3, 4, 5, 3, 4, 5]],
            dtype=torch.int32,
        ),
    )
    assert set(table[0].tolist()).isdisjoint(table[1].tolist())


def test_physical_block_table_rejects_overcommitted_partition() -> None:
    with pytest.raises(ValueError, match="exceeds physical cache capacity"):
        _physical_block_table(
            batch=3,
            width=4,
            blocks_per_request=2,
            physical_blocks=5,
        )


def test_physical_block_table_can_reserve_block_zero_sentinel() -> None:
    table = _physical_block_table(
        batch=2,
        width=6,
        blocks_per_request=3,
        physical_blocks=7,
        first_physical_block=1,
    )

    torch.testing.assert_close(
        table,
        torch.tensor(
            [[1, 2, 3, 1, 2, 3], [4, 5, 6, 4, 5, 6]],
            dtype=torch.int32,
        ),
    )
    assert not torch.any(table == 0)


def test_paged_and_state_slot_mappings_use_physical_block_ids() -> None:
    positions = torch.tensor([[0, 31, 32, 39], [4, 7, 8, 11]], dtype=torch.int32)
    page_table = torch.tensor(
        [[7, 9, 11], [13, 15, 17]],
        dtype=torch.int32,
    )
    state_table = torch.arange(64, dtype=torch.int32).reshape(2, 32)

    paged = _paged_slot_mapping(
        positions=positions,
        block_table=page_table,
        position_divisor=1,
    )
    torch.testing.assert_close(
        paged,
        torch.tensor(
            [
                7 * BLOCK_SIZE,
                7 * BLOCK_SIZE + 31,
                9 * BLOCK_SIZE,
                9 * BLOCK_SIZE + 7,
                13 * BLOCK_SIZE + 4,
                13 * BLOCK_SIZE + 7,
                13 * BLOCK_SIZE + 8,
                13 * BLOCK_SIZE + 11,
            ],
            dtype=torch.int64,
        ),
    )

    state = _state_metadata(
        positions=positions,
        block_table=state_table,
        state_block_size=2,
    )
    expected = []
    for request in range(2):
        for position in positions[request].tolist():
            physical = int(state_table[request, position // 2])
            expected.append(physical * 2 + position % 2)
    torch.testing.assert_close(state, torch.tensor(expected, dtype=torch.int64))

    native = _native_swa_slot_mapping(
        positions=positions,
        block_table=page_table,
    )
    assert native.dtype == torch.int32
    assert tuple(native.shape) == (positions.numel(), 2)
    torch.testing.assert_close(
        native,
        torch.tensor(
            [
                [7, 0],
                [7, 31],
                [9, 0],
                [9, 7],
                [13, 4],
                [13, 7],
                [13, 8],
                [13, 11],
            ],
            dtype=torch.int32,
        ),
    )


def test_window_metadata_has_valid_prefix_and_negative_one_tail() -> None:
    positions = torch.tensor([[0, 2], [31, 33]], dtype=torch.int32)
    table = torch.stack(
        (
            torch.arange(512, dtype=torch.int32),
            torch.arange(512, dtype=torch.int32) + 512,
        )
    )

    indices, lengths = _window_metadata(
        positions=positions,
        ori_block_table=table,
    )

    assert tuple(indices.shape) == (4, FLASH.sliding_window)
    torch.testing.assert_close(lengths, torch.tensor([1, 3, 32, 34], dtype=torch.int32))
    assert indices[0, 0] == 0
    assert indices[1, :3].tolist() == [0, 1, 2]
    assert indices[2, :32].tolist() == [512 * BLOCK_SIZE + i for i in range(32)]
    assert indices[3, :34].tolist() == [512 * BLOCK_SIZE + i for i in range(34)]
    for row, length in zip(indices, lengths, strict=True):
        assert torch.all(row[int(length) :] == -1)


def test_position_rows_are_reconstructible_from_per_request_starts() -> None:
    starts = (0, 1, 3, 127)
    positions = _position_rows(starts, seq=8)

    assert tuple(positions.shape) == (4, 8)
    assert tuple(positions[:, 0].tolist()) == starts
    assert positions[3].tolist() == list(range(127, 135))
