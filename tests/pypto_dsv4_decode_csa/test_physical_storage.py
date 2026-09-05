# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Host-only tests for page-strided CSA physical-storage aliases."""

from __future__ import annotations

import gc
import weakref

import pytest
import torch

from vllm_ascend.ops._pypto_dsv4_csa.physical_storage import (
    PhysicalStorageLayoutError,
    prepare_physical_storage_alias,
)


def _make_page_strided_fixture() -> tuple[torch.Tensor, torch.Tensor, int]:
    canary = -7777
    storage = torch.full((48,), canary, dtype=torch.int32)
    # Each logical page contains two rows of three elements.  The physical
    # layout leaves one element between rows and five elements between pages.
    view = torch.as_strided(
        storage,
        size=(3, 2, 3),
        stride=(12, 4, 1),
        storage_offset=2,
    )
    return storage, view, canary


def test_alias_is_zero_copy_and_preserves_storage_offset() -> None:
    storage, view, _ = _make_page_strided_fixture()

    prepared = prepare_physical_storage_alias(view, name="main_compressor")
    alias = prepared.for_launch()

    assert alias.shape == (1, 31)
    assert alias.stride() == (31, 1)
    assert alias.data_ptr() == view.data_ptr()
    assert alias.untyped_storage().data_ptr() == view.untyped_storage().data_ptr()
    assert alias.untyped_storage()._cdata == view.untyped_storage()._cdata
    assert alias.untyped_storage()._cdata == storage.untyped_storage()._cdata
    assert alias.storage_offset() == view.storage_offset() == 2
    assert prepared.layout.storage_offset == 2
    assert prepared.layout.end_storage_offset_exclusive == 33
    assert prepared.layout.storage_numel == storage.numel()


def test_physical_span_has_exact_boundaries() -> None:
    _, view, _ = _make_page_strided_fixture()
    prepared = prepare_physical_storage_alias(view, name="main_compressor")
    layout = prepared.layout
    alias = prepared.for_launch()

    assert layout.shape == (3, 2, 3)
    assert layout.strides == (12, 4, 1)
    assert layout.logical_numel == 18
    assert layout.physical_span == 31
    assert layout.padding_elements == 13
    assert layout.physical_offset((0, 0, 0)) == 0
    assert layout.physical_offset((2, 1, 2)) == layout.physical_span - 1
    assert alias[0, layout.physical_span - 1].data_ptr() == (
        view.data_ptr() + (layout.physical_span - 1) * view.element_size()
    )
    with pytest.raises(IndexError):
        _ = alias[0, layout.physical_span]


def test_physical_alias_retains_page_padding_canaries() -> None:
    storage, view, canary = _make_page_strided_fixture()
    prepared = prepare_physical_storage_alias(view, name="main_compressor")
    alias = prepared.for_launch()

    logical_offsets: set[int] = set()
    for block in range(view.shape[0]):
        for row in range(view.shape[1]):
            for column in range(view.shape[2]):
                offset = prepared.layout.physical_offset((block, row, column))
                logical_offsets.add(offset)
                view[block, row, column] = 100 * block + 10 * row + column

    for offset in range(prepared.layout.physical_span):
        if offset in logical_offsets:
            continue
        assert alias[0, offset].item() == canary

    # A block>0/intra>0 address must use the physical page and row strides.
    block_two_offset = prepared.layout.physical_offset((2, 1, 2))
    assert block_two_offset == 30
    assert alias[0, block_two_offset].item() == 212
    alias[0, block_two_offset] = 912
    assert view[2, 1, 2].item() == 912

    # Neither logical write path touched padding before, between, or after pages.
    assert storage[view.storage_offset() + 3].item() == canary
    assert storage[view.storage_offset() + 7].item() == canary
    assert storage[view.storage_offset() + 11].item() == canary


@pytest.mark.parametrize(
    "make_invalid, error_match",
    (
        (
            lambda: torch.arange(3, dtype=torch.float32).expand(2, 3),
            "zero or negative strides",
        ),
        (
            lambda: torch.as_strided(
                torch.arange(8, dtype=torch.float32),
                size=(2, 3),
                stride=(1, 1),
            ),
            "non-overlapping row-major page layout",
        ),
        (
            lambda: torch.arange(6, dtype=torch.float32).reshape(2, 3).transpose(0, 1),
            "non-overlapping row-major page layout",
        ),
        (
            lambda: torch.empty((0, 3), dtype=torch.float32),
            "every logical extent must be positive",
        ),
    ),
)
def test_rejects_invalid_page_layouts(make_invalid, error_match: str) -> None:
    with pytest.raises(PhysicalStorageLayoutError, match=error_match):
        prepare_physical_storage_alias(make_invalid(), name="cache")


def test_rejects_non_tensor_scalar_and_meta_inputs() -> None:
    with pytest.raises(TypeError, match="expected torch.Tensor"):
        prepare_physical_storage_alias(object(), name="cache")  # type: ignore[arg-type]
    with pytest.raises(PhysicalStorageLayoutError, match="scalar tensors"):
        prepare_physical_storage_alias(torch.tensor(1.0), name="cache")
    with pytest.raises(PhysicalStorageLayoutError, match="meta tensors"):
        prepare_physical_storage_alias(torch.empty((2, 3), device="meta"), name="cache")


def test_physical_offset_rejects_invalid_indices() -> None:
    _, view, _ = _make_page_strided_fixture()
    layout = prepare_physical_storage_alias(view, name="cache").layout

    with pytest.raises(IndexError, match="expected 3 indices"):
        layout.physical_offset((0, 0))
    with pytest.raises(IndexError, match="outside axis 0"):
        layout.physical_offset((3, 0, 0))
    with pytest.raises(IndexError, match="outside axis 1"):
        layout.physical_offset((0, -1, 0))
    with pytest.raises(TypeError, match="axis 2 must be int"):
        layout.physical_offset((0, 0, 1.0))  # type: ignore[arg-type]


def test_prepared_owner_is_stable_and_keeps_source_alive_before_capture(monkeypatch) -> None:
    source = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    source_reference = weakref.ref(source)
    prepared = prepare_physical_storage_alias(source, name="cache")
    launch_tensor = prepared.for_launch()

    del source
    gc.collect()
    assert source_reference() is prepared.source

    def fail_if_recreated(*args, **kwargs):
        raise AssertionError("for_launch must not rebuild tensor metadata")

    monkeypatch.setattr(torch, "as_strided", fail_if_recreated)
    for _ in range(8):
        assert prepared.for_launch() is launch_tensor
