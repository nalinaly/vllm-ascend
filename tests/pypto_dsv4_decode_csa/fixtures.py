# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Deterministic real-shape fixtures for the decode CSA L1 entry.

The zero fixture is deliberately not presented as a numerical golden.  It is
the first A3 execution probe: all public tensors have production dimensions,
the six mutable states use the concrete vLLM A3 layouts, and every index is a
valid physical address.  Zero weights make the expected output exactly zero,
which isolates ABI/scheduling failures before the non-zero reference is used.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING

import torch

from vllm_ascend.ops._pypto_dsv4_csa import (
    DecodeCSAProgramSpec,
    PreparedDecodeCSACaches,
    prepare_decode_csa_caches,
)
from vllm_ascend.ops._pypto_dsv4_csa.config import BLOCK_SIZE, FLASH

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


_CACHE_ARGUMENTS = frozenset(
    {
        "compress_state",
        "inner_compress_state",
        "kv_cache",
        "cmp_kv",
        "idx_kv_cache",
        "idx_kv_scale",
    }
)

_PYPTO_TO_TORCH_DTYPE = {
    "bfloat16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
    "int8": torch.int8,
    "int32": torch.int32,
    "int64": torch.int64,
}


@dataclass(frozen=True, slots=True)
class DecodeCSAZeroFixture:
    """Strong owner for one capture-stable zero-workload binding."""

    spec: DecodeCSAProgramSpec
    program: object
    arguments: Mapping[str, torch.Tensor | int]
    prepared_caches: PreparedDecodeCSACaches
    raw_cache_tuple: tuple[torch.Tensor, ...]
    storage_owners: tuple[torch.Tensor, ...]
    start_positions: tuple[int, ...]

    @property
    def output(self) -> torch.Tensor:
        value = self.arguments["attn_out"]
        assert isinstance(value, torch.Tensor)
        return value


def _make_strided_tensor(
    *,
    shape: tuple[int, ...],
    strides: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    span = 1 + sum((extent - 1) * stride for extent, stride in zip(shape, strides, strict=True))
    storage = torch.zeros(span, dtype=dtype, device=device)
    view = torch.as_strided(storage, size=shape, stride=strides)
    return storage, view


def _physical_block_table(
    *,
    batch: int,
    width: int,
    blocks_per_request: int,
    physical_blocks: int,
    first_physical_block: int = 0,
) -> torch.Tensor:
    if first_physical_block < 0 or first_physical_block >= physical_blocks:
        raise ValueError(f"first physical block must be within [0, {physical_blocks}), got {first_physical_block}")
    if first_physical_block + batch * blocks_per_request > physical_blocks:
        raise ValueError(
            "fixture partition exceeds physical cache capacity: "
            f"batch={batch}, per_request={blocks_per_request}, "
            f"physical={physical_blocks}, first={first_physical_block}"
        )
    logical = torch.arange(width, dtype=torch.int64).remainder(blocks_per_request)
    bases = first_physical_block + torch.arange(batch, dtype=torch.int64).view(batch, 1) * blocks_per_request
    return (bases + logical.view(1, width)).to(torch.int32)


def _position_rows(
    starts: Sequence[int],
    *,
    seq: int,
) -> torch.Tensor:
    """Build the full Host metadata implied by request start positions.

    Only the first column crosses the production L1 ABI.  The full rows are
    retained in this fixture to derive all slot and window metadata from the
    exact same consecutive-position contract as the device kernel.
    """
    start_tensor = torch.tensor(tuple(int(value) for value in starts), dtype=torch.int32)
    return start_tensor.view(-1, 1) + torch.arange(seq, dtype=torch.int32).view(1, seq)


def _state_metadata(
    *,
    positions: torch.Tensor,
    block_table: torch.Tensor,
    state_block_size: int,
) -> torch.Tensor:
    batch, seq = positions.shape
    rows = torch.empty((batch, seq), dtype=torch.int64)
    for request in range(batch):
        logical_blocks = positions[request].to(torch.int64) // state_block_size
        physical = block_table[request, logical_blocks]
        rows[request] = physical.to(torch.int64) * state_block_size + positions[request].to(torch.int64).remainder(
            state_block_size
        )
    return rows.reshape(-1)


def _paged_slot_mapping(
    *,
    positions: torch.Tensor,
    block_table: torch.Tensor,
    position_divisor: int,
) -> torch.Tensor:
    batch, seq = positions.shape
    rows = torch.empty((batch, seq), dtype=torch.int64)
    logical_positions = positions.to(torch.int64) // position_divisor
    for request in range(batch):
        logical_blocks = logical_positions[request] // BLOCK_SIZE
        physical = block_table[request, logical_blocks]
        rows[request] = physical.to(torch.int64) * BLOCK_SIZE + logical_positions[request].remainder(BLOCK_SIZE)
    return rows.reshape(-1)


def _native_swa_slot_mapping(
    *,
    positions: torch.Tensor,
    block_table: torch.Tensor,
) -> torch.Tensor:
    """Return the A2/A3 native ``[physical_block, block_offset]`` ABI."""
    batch, seq = positions.shape
    mapping = torch.empty((batch, seq, 2), dtype=torch.int32)
    for request in range(batch):
        logical_blocks = positions[request].to(torch.int64) // BLOCK_SIZE
        mapping[request, :, 0] = block_table[request, logical_blocks]
        mapping[request, :, 1] = positions[request].remainder(BLOCK_SIZE)
    return mapping.reshape(batch * seq, 2)


def _window_metadata(
    *,
    positions: torch.Tensor,
    ori_block_table: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, seq = positions.shape
    window = FLASH.sliding_window
    indices = torch.full((batch * seq, window), -1, dtype=torch.int32)
    lengths = torch.empty(batch * seq, dtype=torch.int32)
    for request in range(batch):
        for token in range(seq):
            absolute = int(positions[request, token])
            visible = min(absolute + 1, window)
            first = absolute - visible + 1
            for offset, history_position in enumerate(range(first, absolute + 1)):
                logical_block = history_position // BLOCK_SIZE
                physical_block = int(ori_block_table[request, logical_block])
                indices[request * seq + token, offset] = physical_block * BLOCK_SIZE + history_position % BLOCK_SIZE
            lengths[request * seq + token] = visible
    return indices, lengths


def build_zero_fixture(
    spec: DecodeCSAProgramSpec,
    *,
    device: torch.device | str,
    runtime: str = "tensormap_and_ringbuffer",
    start_positions: Sequence[int] | None = None,
) -> DecodeCSAZeroFixture:
    """Allocate one production-shape, capture-stable, zero-output fixture."""
    from vllm_ascend.ops._pypto_dsv4_csa import make_decode_csa_l1_program

    target = torch.device(device)
    if start_positions is None:
        canonical = (0, 1, 3, 4, 7, 8, 31, 32, 63, 64, 127, 128, 255, 256, 511, 512)
        starts = canonical[: spec.batch]
    else:
        starts = tuple(int(value) for value in start_positions)
    if len(starts) != spec.batch:
        raise ValueError(f"expected {spec.batch} start positions, got {len(starts)}")
    if min(starts) < 0 or max(starts) + spec.seq > FLASH.max_position_embeddings:
        raise ValueError(f"start positions out of range: {starts}")

    program = make_decode_csa_l1_program(spec, runtime)
    arguments: dict[str, torch.Tensor | int] = {}
    for name, parameter in inspect.signature(program._func).parameters.items():
        annotation = parameter.annotation
        shape = getattr(annotation, "shape", None)
        if shape is None or name in _CACHE_ARGUMENTS:
            continue
        dtype_name = str(annotation.dtype)
        try:
            dtype = _PYPTO_TO_TORCH_DTYPE[dtype_name]
        except KeyError as error:
            raise TypeError(f"unsupported fixture dtype {dtype_name!r} for {name}") from error
        arguments[name] = torch.zeros(tuple(shape), dtype=dtype, device=target)

    physical = spec.physical_layout
    compressed_kv = torch.zeros(
        (spec.compressed_blocks, BLOCK_SIZE, 1, FLASH.head_dim),
        dtype=torch.bfloat16,
        device=target,
    )
    swa_kv = torch.zeros(
        (spec.swa_blocks, BLOCK_SIZE, 1, FLASH.head_dim),
        dtype=torch.bfloat16,
        device=target,
    )
    main_storage, main_state = _make_strided_tensor(
        shape=(spec.main_state_blocks, 2, 1, 4 * FLASH.head_dim),
        strides=(
            physical.main_state_strides[0],
            physical.main_state_strides[1],
            physical.main_state_strides[1],
            1,
        ),
        dtype=torch.float32,
        device=target,
    )
    inner_storage, inner_state = _make_strided_tensor(
        shape=(spec.inner_state_blocks, 2, 1, 4 * FLASH.index_head_dim),
        strides=(
            physical.inner_state_strides[0],
            physical.inner_state_strides[1],
            physical.inner_state_strides[1],
            1,
        ),
        dtype=torch.float32,
        device=target,
    )
    indexer_k_storage, indexer_k = _make_strided_tensor(
        shape=(spec.indexer_blocks, BLOCK_SIZE, 1, FLASH.index_head_dim),
        strides=physical.indexer_k_strides,
        dtype=torch.int8,
        device=target,
    )
    indexer_scale_storage, indexer_scale = _make_strided_tensor(
        shape=(spec.indexer_blocks, BLOCK_SIZE, 1, 1),
        strides=physical.indexer_scale_strides,
        dtype=torch.float16,
        device=target,
    )
    indexer_scale.fill_(1.0)
    raw_caches = (
        compressed_kv,
        swa_kv,
        main_state,
        inner_state,
        indexer_k,
        indexer_scale,
    )
    prepared = prepare_decode_csa_caches(raw_caches, spec)
    arguments.update(prepared.for_launch())

    positions = _position_rows(starts, seq=spec.seq)

    # Give each request a disjoint physical region.  Table entries outside the
    # active prefix wrap within that region, remaining valid even if a zero-score
    # top-k lane is selected during this ABI smoke.
    ori_per_request = spec.swa_blocks // spec.batch
    cmp_per_request = spec.compressed_blocks // spec.batch
    idx_per_request = spec.indexer_blocks // spec.batch
    # CANN compressor treats physical state block 0 as an invalid sentinel and
    # skips its writes.  Production allocators likewise start these two state
    # namespaces at block 1; the synthetic native/PyPTO comparison must not
    # hand request 0 a sentinel block as if it were usable storage.
    state_per_request = (spec.main_state_blocks - 1) // spec.batch
    inner_state_per_request = (spec.inner_state_blocks - 1) // spec.batch
    if (
        min(
            ori_per_request,
            cmp_per_request,
            idx_per_request,
            state_per_request,
            inner_state_per_request,
        )
        <= 0
    ):
        raise ValueError("fixture requires at least one physical block per request")

    main_table = _physical_block_table(
        batch=spec.batch,
        width=arguments["compress_state_block_table"].shape[1],
        blocks_per_request=state_per_request,
        physical_blocks=spec.main_state_blocks,
        first_physical_block=1,
    )
    inner_table = _physical_block_table(
        batch=spec.batch,
        width=arguments["inner_compress_state_block_table"].shape[1],
        blocks_per_request=inner_state_per_request,
        physical_blocks=spec.inner_state_blocks,
        first_physical_block=1,
    )
    cmp_table = _physical_block_table(
        batch=spec.batch,
        width=arguments["cmp_block_table"].shape[1],
        blocks_per_request=cmp_per_request,
        physical_blocks=spec.compressed_blocks,
    )
    idx_table = _physical_block_table(
        batch=spec.batch,
        width=arguments["idx_block_table"].shape[1],
        blocks_per_request=idx_per_request,
        physical_blocks=spec.indexer_blocks,
    )
    ori_table = _physical_block_table(
        batch=spec.batch,
        width=arguments["swa_block_table"].shape[1],
        blocks_per_request=ori_per_request,
        physical_blocks=spec.swa_blocks,
    )

    host_values = {
        "compress_state_block_table": main_table,
        "inner_compress_state_block_table": inner_table,
        "cmp_block_table": cmp_table,
        "idx_block_table": idx_table,
        "swa_block_table": ori_table,
        "start_positions": torch.tensor(starts, dtype=torch.int32),
        "kv_seq_lens": positions[:, -1] + 1,
    }
    for name, value in host_values.items():
        target_tensor = arguments[name]
        assert isinstance(target_tensor, torch.Tensor)
        target_tensor.copy_(value.to(dtype=target_tensor.dtype, device=target))

    freqs_cos = arguments["freqs_cos"]
    attn_sink = arguments["attn_sink"]
    assert isinstance(freqs_cos, torch.Tensor)
    assert isinstance(attn_sink, torch.Tensor)
    freqs_cos.fill_(1.0)
    attn_sink.fill_(4.0)

    return DecodeCSAZeroFixture(
        spec=spec,
        program=program,
        arguments=MappingProxyType(arguments),
        prepared_caches=prepared,
        raw_cache_tuple=raw_caches,
        storage_owners=(
            main_storage,
            inner_storage,
            indexer_k_storage,
            indexer_scale_storage,
        ),
        start_positions=starts,
    )


__all__ = ["DecodeCSAZeroFixture", "build_zero_fixture"]
