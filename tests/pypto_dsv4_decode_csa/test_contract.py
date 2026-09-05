# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Host-only tests for the static DeepSeek V4 decode CSA contract."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from vllm_ascend.ops._pypto_dsv4_csa.config import DECODE_SEQ
from vllm_ascend.ops._pypto_dsv4_csa.contract import (
    DecodeCSAPhysicalLayout,
    DecodeCSAProgramSpec,
)


@pytest.mark.parametrize("batch", (4, 8, 12, 16))
def test_supported_batch_buckets(batch: int) -> None:
    spec = DecodeCSAProgramSpec(batch=batch)

    assert spec.batch == batch
    assert spec.seq == DECODE_SEQ
    assert spec.tokens == batch * DECODE_SEQ


@pytest.mark.parametrize("batch", (-4, 0, 1, 5, 20))
def test_rejects_unsupported_batch(batch: int) -> None:
    with pytest.raises(ValueError, match="batch must be one of"):
        DecodeCSAProgramSpec(batch=batch)


@pytest.mark.parametrize("seq", (0, DECODE_SEQ - 1, DECODE_SEQ + 1))
def test_rejects_non_decode_sequence_length(seq: int) -> None:
    with pytest.raises(ValueError, match="decode CSA is specialized for seq"):
        DecodeCSAProgramSpec(batch=4, seq=seq)


@pytest.mark.parametrize(
    "field",
    (
        "swa_blocks",
        "compressed_blocks",
        "main_state_blocks",
        "inner_state_blocks",
        "indexer_blocks",
    ),
)
@pytest.mark.parametrize("invalid_count", (0, -1))
def test_rejects_non_positive_physical_block_count(
    field: str,
    invalid_count: int,
) -> None:
    with pytest.raises(ValueError, match="cache block counts must be positive") as error:
        DecodeCSAProgramSpec(batch=4, **{field: invalid_count})

    assert field in str(error.value)


@pytest.mark.parametrize(
    "field",
    (
        "swa_table_width",
        "compressed_table_width",
        "main_state_table_width",
        "inner_state_table_width",
        "indexer_table_width",
    ),
)
@pytest.mark.parametrize("invalid_width", (False, 0, -1, 1.5))
def test_rejects_invalid_block_table_width(
    field: str,
    invalid_width: object,
) -> None:
    with pytest.raises(ValueError, match="block-table widths") as error:
        DecodeCSAProgramSpec(batch=4, **{field: invalid_width})

    assert field in str(error.value)


@pytest.mark.parametrize(
    ("field", "strides", "error"),
    (
        ("main_state_strides", (8192, 2048, 0), "positive integer strides"),
        ("main_state_strides", (8192, 1024, 1), "contiguous within each physical page"),
        ("main_state_strides", (4095, 2048, 1), "cover one logical page"),
        ("inner_state_strides", (1040, 256, 1), "contiguous within each physical page"),
        ("indexer_k_strides", (4160, 64, 128, 1), "contiguous within each physical page"),
        ("indexer_scale_strides", (2080, 2, 1, 1), "contiguous within each physical page"),
    ),
)
def test_rejects_invalid_physical_layout(
    field: str,
    strides: tuple[int, ...],
    error: str,
) -> None:
    with pytest.raises(ValueError, match=error):
        DecodeCSAPhysicalLayout(**{field: strides})


def test_rejects_mismatched_indexer_page_byte_sizes() -> None:
    with pytest.raises(ValueError, match="different byte sizes"):
        DecodeCSAPhysicalLayout(indexer_scale_strides=(2048, 1, 1, 1))


def test_key_contains_every_physical_capacity_and_layout_stride() -> None:
    layout = DecodeCSAPhysicalLayout(
        main_state_strides=(9000, 2048, 1),
        inner_state_strides=(1200, 512, 1),
        indexer_k_strides=(5000, 128, 128, 1),
        indexer_scale_strides=(2500, 1, 1, 1),
    )
    spec = DecodeCSAProgramSpec(
        batch=12,
        seq=DECODE_SEQ,
        swa_blocks=101,
        compressed_blocks=103,
        main_state_blocks=107,
        inner_state_blocks=109,
        indexer_blocks=113,
        swa_table_width=127,
        compressed_table_width=131,
        main_state_table_width=137,
        inner_state_table_width=139,
        indexer_table_width=149,
        physical_layout=layout,
    )

    assert spec.key == (
        12,
        DECODE_SEQ,
        101,
        103,
        107,
        109,
        113,
        127,
        131,
        137,
        139,
        149,
        layout.key,
    )


def test_a3_physical_spans_include_inter_page_padding_but_not_tail_padding() -> None:
    spec = DecodeCSAProgramSpec(
        batch=4,
        main_state_blocks=3,
        inner_state_blocks=3,
        indexer_blocks=3,
    )

    assert spec.physical_layout.main_state_strides == (8192, 2048, 1)
    assert spec.physical_layout.inner_state_strides == (1040, 512, 1)
    assert spec.physical_layout.indexer_k_strides == (4160, 128, 128, 1)
    assert spec.physical_layout.indexer_scale_strides == (2080, 1, 1, 1)
    assert spec.main_state_span == 2 * 8192 + 2 * 2048
    assert spec.inner_state_span == 2 * 1040 + 2 * 512
    assert spec.indexer_k_span == 2 * 4160 + 32 * 128
    assert spec.indexer_scale_span == 2 * 2080 + 32


@pytest.mark.parametrize(
    "module_name",
    (
        "vllm_ascend.ops._pypto_dsv4_csa",
        "vllm_ascend.ops._pypto_dsv4_csa.contract",
    ),
)
def test_import_does_not_eagerly_import_pypto(module_name: str) -> None:
    repository_root = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = os.pathsep.join(part for part in (str(repository_root), existing_pythonpath) if part)
    probe = r"""
import importlib
import importlib.abc
import sys


def is_pypto_module(name):
    return name == "pypto" or name.startswith("pypto.")


if any(is_pypto_module(name) for name in sys.modules):
    raise AssertionError("pypto was loaded before the import probe")


class RejectPyPTOImport(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if is_pypto_module(fullname):
            raise AssertionError(f"{sys.argv[1]} eagerly imported {fullname}")
        return None


sys.meta_path.insert(0, RejectPyPTOImport())
importlib.import_module(sys.argv[1])
if any(is_pypto_module(name) for name in sys.modules):
    raise AssertionError(f"{sys.argv[1]} left pypto loaded")
"""

    result = subprocess.run(
        [sys.executable, "-c", probe, module_name],
        cwd=repository_root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, (
        f"isolated import probe failed for {module_name}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
