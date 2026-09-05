# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Production-shaped metadata materialization for stateful decode traces.

``trace.py`` deliberately knows nothing about torch or the production DSA ABI.
This module is the boundary between those two worlds:

* :class:`DecodeStepMetadataPayload` is a deeply immutable Host description of
  one ``DecodeStep`` in full static-bucket shape;
* :class:`StagedDecodeStepMetadataPayload` owns the corresponding canonical
  INT32 tensors on a caller-selected device;
* :class:`DecodeStepMetadataBufferOwner` owns the fixed tensor addresses used
  by one captured static-bucket graph and only updates their contents in place.

The replay signal for an active request is the already-existing device
``kv_seq_lens[row] > 0`` predicate.  ``num_reqs_actual`` remains Host metadata
for eager/capture construction and is never treated as a replay-time scalar.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

import torch

from tests.pypto_dsv4_decode_csa.trace import (
    CACHE_FAMILY_NAMES,
    COMPRESSOR_ATTENTION,
    INDEXER_CACHE,
    INNER_COMPRESSOR_STATE,
    INVALID_REQUEST_ID,
    MAIN_COMPRESSOR_STATE,
    SWA_CACHE,
    DecodeStep,
    DecodeTrace,
)
from vllm_ascend.ops._pypto_dsv4_csa import DecodeCSAProgramSpec

INVALID_BLOCK_ID = 0


class DecodeStepMetadataError(ValueError):
    """Raised when a trace step cannot be represented by a production ABI."""


@dataclass(frozen=True, slots=True)
class _FamilyContract:
    family: str
    argument_name: str
    table_width_attribute: str
    physical_blocks_attribute: str
    default_block_offset: int


_FAMILY_CONTRACTS = (
    _FamilyContract(
        COMPRESSOR_ATTENTION,
        "cmp_block_table",
        "compressed_table_width",
        "compressed_blocks",
        0,
    ),
    _FamilyContract(
        MAIN_COMPRESSOR_STATE,
        "compress_state_block_table",
        "main_state_table_width",
        "main_state_blocks",
        1,
    ),
    _FamilyContract(
        INNER_COMPRESSOR_STATE,
        "inner_compress_state_block_table",
        "inner_state_table_width",
        "inner_state_blocks",
        1,
    ),
    _FamilyContract(
        INDEXER_CACHE,
        "idx_block_table",
        "indexer_table_width",
        "indexer_blocks",
        0,
    ),
    _FamilyContract(
        SWA_CACHE,
        "swa_block_table",
        "swa_table_width",
        "swa_blocks",
        0,
    ),
)
_CONTRACT_BY_FAMILY = {contract.family: contract for contract in _FAMILY_CONTRACTS}
_CONTRACT_BY_ARGUMENT = {contract.argument_name: contract for contract in _FAMILY_CONTRACTS}

TABLE_ARGUMENT_NAMES = tuple(contract.argument_name for contract in _FAMILY_CONTRACTS)
DEFAULT_PHYSICAL_BLOCK_OFFSETS = MappingProxyType(
    {contract.family: contract.default_block_offset for contract in _FAMILY_CONTRACTS}
)


def _normalize_block_offsets(offsets: Mapping[str, int] | None) -> tuple[int, ...]:
    source = DEFAULT_PHYSICAL_BLOCK_OFFSETS if offsets is None else offsets
    if set(source) != set(CACHE_FAMILY_NAMES):
        missing = sorted(set(CACHE_FAMILY_NAMES) - set(source))
        extra = sorted(set(source) - set(CACHE_FAMILY_NAMES))
        raise DecodeStepMetadataError(
            f"physical block offsets must name all five families; missing={missing}, extra={extra}"
        )
    normalized: list[int] = []
    for family in CACHE_FAMILY_NAMES:
        offset = source[family]
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise DecodeStepMetadataError(f"{family}: physical block offset must be a non-negative integer")
        normalized.append(offset)
    return tuple(normalized)


@dataclass(frozen=True, slots=True)
class ProductionBlockTable:
    """One immutable full-B production block table."""

    family: str
    argument_name: str
    width: int
    rows: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        contract = _CONTRACT_BY_FAMILY.get(self.family)
        if contract is None:
            raise DecodeStepMetadataError(f"unknown cache family {self.family!r}")
        if self.argument_name != contract.argument_name:
            raise DecodeStepMetadataError(
                f"{self.family}: expected argument {contract.argument_name!r}, got {self.argument_name!r}"
            )
        if isinstance(self.width, bool) or not isinstance(self.width, int) or self.width <= 0:
            raise DecodeStepMetadataError(f"{self.argument_name}: width must be positive")
        for row_index, row in enumerate(self.rows):
            if len(row) != self.width:
                raise DecodeStepMetadataError(
                    f"{self.argument_name}[{row_index}]: expected width={self.width}, got {len(row)}"
                )
            if any(isinstance(block, bool) or not isinstance(block, int) or block < 0 for block in row):
                raise DecodeStepMetadataError(
                    f"{self.argument_name}[{row_index}]: block IDs must be non-negative integers"
                )


@dataclass(frozen=True, slots=True)
class DecodeStepMetadataPayload:
    """Deeply immutable full-B Host payload for one trace step.

    Table rows are in production ABI order.  Active requests occupy
    ``[0:A)``; rows ``[A:B)`` are explicit zero padding.  The five cache
    namespaces remain independent even when their integer block IDs happen to
    be equal.
    """

    source_step: DecodeStep
    spec: DecodeCSAProgramSpec
    request_ids: tuple[int, ...]
    query_start_loc: tuple[int, ...]
    start_positions: tuple[int, ...]
    kv_seq_lens: tuple[int, ...]
    tables: tuple[ProductionBlockTable, ...]
    physical_block_offsets: tuple[int, ...]

    def __post_init__(self) -> None:
        step = self.source_step
        if self.spec.batch != step.bucket_size:
            raise DecodeStepMetadataError(
                f"step {step.step_id}: B{step.bucket_size} cannot use a B{self.spec.batch} specialization"
            )
        if self.spec.seq != len(step.requests[0].token_positions):
            raise DecodeStepMetadataError(
                f"step {step.step_id}: trace S={len(step.requests[0].token_positions)} "
                f"does not match specialization S={self.spec.seq}"
            )
        if len(self.physical_block_offsets) != len(CACHE_FAMILY_NAMES):
            raise DecodeStepMetadataError("physical block offsets are not in canonical five-family order")

        bucket = self.spec.batch
        actual = step.num_reqs_actual
        expected_ids = tuple(request.request_id for request in step.active_requests) + (INVALID_REQUEST_ID,) * (
            bucket - actual
        )
        if self.request_ids != expected_ids:
            raise DecodeStepMetadataError(
                f"step {step.step_id}: request rows are not the compacted active-prefix order"
            )
        expected_query = tuple(range(0, (bucket + 1) * self.spec.seq, self.spec.seq))
        if self.query_start_loc != expected_query:
            raise DecodeStepMetadataError(
                f"step {step.step_id}: query_start_loc must retain full-B uniform S={self.spec.seq} rows"
            )
        expected_starts = tuple(request.sequence_length_before for request in step.active_requests) + (0,) * (
            bucket - actual
        )
        expected_lengths = tuple(request.sequence_length_after for request in step.active_requests) + (0,) * (
            bucket - actual
        )
        if self.start_positions != expected_starts:
            raise DecodeStepMetadataError(f"step {step.step_id}: start_positions do not match the trace")
        if self.kv_seq_lens != expected_lengths:
            raise DecodeStepMetadataError(f"step {step.step_id}: kv_seq_lens do not match the trace")

        expected_table_families = tuple(contract.family for contract in _FAMILY_CONTRACTS)
        if tuple(table.family for table in self.tables) != expected_table_families:
            raise DecodeStepMetadataError("block tables are not in canonical production family order")
        for family_index, (table, contract) in enumerate(zip(self.tables, _FAMILY_CONTRACTS, strict=True)):
            expected_width = int(getattr(self.spec, contract.table_width_attribute))
            capacity = int(getattr(self.spec, contract.physical_blocks_attribute))
            offset = self.physical_block_offsets[family_index]
            if table.width != expected_width:
                raise DecodeStepMetadataError(
                    f"{table.argument_name}: expected static width={expected_width}, got {table.width}"
                )
            if len(table.rows) != bucket:
                raise DecodeStepMetadataError(f"{table.argument_name}: expected B={bucket} rows, got {len(table.rows)}")
            for row_index, request in enumerate(step.active_requests):
                abstract_blocks = request.ownership_for(contract.family).logical_block_table
                concrete_blocks = tuple(block + offset for block in abstract_blocks)
                if len(concrete_blocks) > expected_width:
                    raise DecodeStepMetadataError(
                        f"step {step.step_id}/{table.argument_name}[{row_index}]: "
                        f"requires {len(concrete_blocks)} entries but static width is {expected_width}"
                    )
                if any(block >= capacity for block in concrete_blocks):
                    largest = max(concrete_blocks, default=-1)
                    raise DecodeStepMetadataError(
                        f"step {step.step_id}/{table.argument_name}[{row_index}]: "
                        f"physical block {largest} exceeds capacity={capacity}"
                    )
                expected_row = concrete_blocks + (INVALID_BLOCK_ID,) * (expected_width - len(concrete_blocks))
                if table.rows[row_index] != expected_row:
                    raise DecodeStepMetadataError(
                        f"step {step.step_id}/{table.argument_name}[{row_index}] does not match ownership"
                    )
            if any(
                any(block != INVALID_BLOCK_ID for block in table.rows[row_index]) for row_index in range(actual, bucket)
            ):
                raise DecodeStepMetadataError(
                    f"step {step.step_id}/{table.argument_name}: padded rows must be all zero"
                )

    @property
    def step_id(self) -> int:
        return self.source_step.step_id

    @property
    def bucket_size(self) -> int:
        return self.source_step.bucket_size

    @property
    def actual(self) -> int:
        return self.source_step.num_reqs_actual

    @property
    def num_reqs_actual(self) -> int:
        return self.actual

    @property
    def actual_tokens(self) -> int:
        return self.actual * self.spec.seq

    @property
    def block_tables(self) -> Mapping[str, tuple[tuple[int, ...], ...]]:
        return MappingProxyType({table.argument_name: table.rows for table in self.tables})

    def table_for_family(self, family: str) -> ProductionBlockTable:
        try:
            index = CACHE_FAMILY_NAMES.index(family)
        except ValueError as error:
            raise DecodeStepMetadataError(f"unknown cache family {family!r}") from error
        return self.tables[index]


class DecodeStepMetadataMaterializer:
    """Pure ``DecodeStep`` -> immutable production metadata transform."""

    def __init__(
        self,
        spec: DecodeCSAProgramSpec,
        *,
        physical_block_offsets: Mapping[str, int] | None = None,
    ) -> None:
        self._spec = spec
        self._offsets = _normalize_block_offsets(physical_block_offsets)

    @property
    def spec(self) -> DecodeCSAProgramSpec:
        return self._spec

    @property
    def physical_block_offsets(self) -> Mapping[str, int]:
        return MappingProxyType(dict(zip(CACHE_FAMILY_NAMES, self._offsets, strict=True)))

    def materialize(self, step: DecodeStep) -> DecodeStepMetadataPayload:
        if not isinstance(step, DecodeStep):
            raise TypeError(f"expected DecodeStep, got {type(step).__name__}")
        if step.bucket_size != self._spec.batch:
            raise DecodeStepMetadataError(
                f"step {step.step_id}: B{step.bucket_size} cannot use a B{self._spec.batch} specialization"
            )

        tables: list[ProductionBlockTable] = []
        for family_index, contract in enumerate(_FAMILY_CONTRACTS):
            width = int(getattr(self._spec, contract.table_width_attribute))
            capacity = int(getattr(self._spec, contract.physical_blocks_attribute))
            offset = self._offsets[family_index]
            rows: list[tuple[int, ...]] = []
            for request in step.active_requests:
                concrete = tuple(block + offset for block in request.ownership_for(contract.family).logical_block_table)
                if len(concrete) > width:
                    raise DecodeStepMetadataError(
                        f"step {step.step_id}/{contract.argument_name}: request {request.request_id} "
                        f"requires {len(concrete)} entries but static width is {width}"
                    )
                if any(block >= capacity for block in concrete):
                    largest = max(concrete, default=-1)
                    raise DecodeStepMetadataError(
                        f"step {step.step_id}/{contract.argument_name}: request {request.request_id} "
                        f"uses physical block {largest} with capacity={capacity}"
                    )
                rows.append(concrete + (INVALID_BLOCK_ID,) * (width - len(concrete)))
            rows.extend((INVALID_BLOCK_ID,) * width for _ in range(self._spec.batch - step.num_reqs_actual))
            tables.append(
                ProductionBlockTable(
                    family=contract.family,
                    argument_name=contract.argument_name,
                    width=width,
                    rows=tuple(rows),
                )
            )

        request_ids = tuple(request.request_id for request in step.active_requests) + (INVALID_REQUEST_ID,) * (
            self._spec.batch - step.num_reqs_actual
        )
        starts = tuple(request.sequence_length_before for request in step.active_requests) + (0,) * (
            self._spec.batch - step.num_reqs_actual
        )
        lengths = tuple(request.sequence_length_after for request in step.active_requests) + (0,) * (
            self._spec.batch - step.num_reqs_actual
        )
        query_start_loc = tuple(range(0, (self._spec.batch + 1) * self._spec.seq, self._spec.seq))
        return DecodeStepMetadataPayload(
            source_step=step,
            spec=self._spec,
            request_ids=request_ids,
            query_start_loc=query_start_loc,
            start_positions=starts,
            kv_seq_lens=lengths,
            tables=tuple(tables),
            physical_block_offsets=self._offsets,
        )


@dataclass(frozen=True, slots=True)
class FamilyMetadataRequirement:
    """Minimum one-family static shape/capacity required by a trace."""

    family: str
    argument_name: str
    table_width: int
    physical_blocks: int
    physical_block_offset: int


@dataclass(frozen=True, slots=True)
class DecodeTraceMetadataRequirements:
    """Static metadata requirements derived without allocating tensors."""

    families: tuple[FamilyMetadataRequirement, ...]

    @property
    def program_spec_overrides(self) -> Mapping[str, int]:
        values: dict[str, int] = {}
        for requirement in self.families:
            contract = _CONTRACT_BY_FAMILY[requirement.family]
            values[contract.table_width_attribute] = requirement.table_width
            values[contract.physical_blocks_attribute] = requirement.physical_blocks
        return MappingProxyType(values)


def derive_trace_metadata_requirements(
    trace: DecodeTrace,
    *,
    physical_block_offsets: Mapping[str, int] | None = None,
) -> DecodeTraceMetadataRequirements:
    """Return the smallest static widths/capacities that can hold ``trace``."""
    if not isinstance(trace, DecodeTrace):
        raise TypeError(f"expected DecodeTrace, got {type(trace).__name__}")
    offsets = _normalize_block_offsets(physical_block_offsets)
    requirements: list[FamilyMetadataRequirement] = []
    for family_index, contract in enumerate(_FAMILY_CONTRACTS):
        width = 1
        largest = INVALID_BLOCK_ID
        offset = offsets[family_index]
        for step in trace.steps:
            for request in step.active_requests:
                blocks = request.ownership_for(contract.family).logical_block_table
                width = max(width, len(blocks))
                if blocks:
                    largest = max(largest, max(blocks) + offset)
        requirements.append(
            FamilyMetadataRequirement(
                family=contract.family,
                argument_name=contract.argument_name,
                table_width=width,
                physical_blocks=largest + 1,
                physical_block_offset=offset,
            )
        )
    return DecodeTraceMetadataRequirements(tuple(requirements))


def materialize_decode_trace(
    trace: DecodeTrace,
    specs_by_bucket: Mapping[int, DecodeCSAProgramSpec],
    *,
    physical_block_offsets: Mapping[str, int] | None = None,
) -> tuple[DecodeStepMetadataPayload, ...]:
    """Materialize a mixed-bucket trace using one static spec per bucket."""
    if not isinstance(trace, DecodeTrace):
        raise TypeError(f"expected DecodeTrace, got {type(trace).__name__}")
    required_buckets = set(trace.bucket_sequence)
    missing = sorted(required_buckets - set(specs_by_bucket))
    if missing:
        raise DecodeStepMetadataError(f"missing static program specs for buckets {missing}")
    materializers: dict[int, DecodeStepMetadataMaterializer] = {}
    for bucket in required_buckets:
        spec = specs_by_bucket[bucket]
        if spec.batch != bucket:
            raise DecodeStepMetadataError(f"specs_by_bucket[{bucket}] contains a B{spec.batch} specialization")
        materializers[bucket] = DecodeStepMetadataMaterializer(
            spec,
            physical_block_offsets=physical_block_offsets,
        )
    return tuple(materializers[step.bucket_size].materialize(step) for step in trace.steps)


def _require_staged_tensor(
    tensor: object,
    *,
    name: str,
    shape: tuple[int, ...],
    device: torch.device,
) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise DecodeStepMetadataError(f"{name}: expected torch.Tensor")
    if tuple(tensor.shape) != shape:
        raise DecodeStepMetadataError(f"{name}: expected shape={shape}, got {tuple(tensor.shape)}")
    if tensor.dtype != torch.int32:
        raise DecodeStepMetadataError(f"{name}: expected torch.int32, got {tensor.dtype}")
    if tensor.device != device:
        raise DecodeStepMetadataError(f"{name}: expected device={device}, got {tensor.device}")
    if not tensor.is_contiguous():
        raise DecodeStepMetadataError(f"{name}: expected canonical contiguous storage")
    return tensor


@dataclass(frozen=True, slots=True)
class StagedDecodeStepMetadataPayload:
    """Canonical tensor staging for one immutable Host payload."""

    host_payload: DecodeStepMetadataPayload
    query_start_loc: torch.Tensor
    block_tables: Mapping[str, torch.Tensor]
    start_positions: torch.Tensor
    kv_seq_lens: torch.Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.host_payload, DecodeStepMetadataPayload):
            raise TypeError("host_payload must be DecodeStepMetadataPayload")
        normalized_tables = dict(self.block_tables)
        if tuple(normalized_tables) != TABLE_ARGUMENT_NAMES:
            raise DecodeStepMetadataError(
                f"staged block tables must use canonical order {TABLE_ARGUMENT_NAMES}, got {tuple(normalized_tables)}"
            )
        object.__setattr__(self, "block_tables", MappingProxyType(normalized_tables))

        device = self.start_positions.device
        bucket = self.spec.batch
        _require_staged_tensor(
            self.query_start_loc,
            name="query_start_loc",
            shape=(bucket + 1,),
            device=device,
        )
        _require_staged_tensor(
            self.start_positions,
            name="start_positions",
            shape=(bucket,),
            device=device,
        )
        _require_staged_tensor(
            self.kv_seq_lens,
            name="kv_seq_lens",
            shape=(bucket,),
            device=device,
        )
        for table in self.host_payload.tables:
            _require_staged_tensor(
                self.block_tables[table.argument_name],
                name=table.argument_name,
                shape=(bucket, table.width),
                device=device,
            )

    @classmethod
    def from_host(
        cls,
        payload: DecodeStepMetadataPayload,
        *,
        device: torch.device | str = "cpu",
    ) -> StagedDecodeStepMetadataPayload:
        """Allocate one staging set outside capture/replay."""
        if not isinstance(payload, DecodeStepMetadataPayload):
            raise TypeError(f"expected DecodeStepMetadataPayload, got {type(payload).__name__}")
        target = torch.device(device)
        tables = {
            table.argument_name: torch.tensor(table.rows, dtype=torch.int32, device=target) for table in payload.tables
        }
        return cls(
            host_payload=payload,
            query_start_loc=torch.tensor(payload.query_start_loc, dtype=torch.int32, device=target),
            block_tables=MappingProxyType(tables),
            start_positions=torch.tensor(payload.start_positions, dtype=torch.int32, device=target),
            kv_seq_lens=torch.tensor(payload.kv_seq_lens, dtype=torch.int32, device=target),
        )

    @property
    def spec(self) -> DecodeCSAProgramSpec:
        return self.host_payload.spec

    @property
    def step_id(self) -> int:
        return self.host_payload.step_id

    @property
    def actual(self) -> int:
        return self.host_payload.actual

    @property
    def num_reqs_actual(self) -> int:
        return self.actual

    @property
    def actual_tokens(self) -> int:
        return self.host_payload.actual_tokens

    @property
    def device(self) -> torch.device:
        return self.start_positions.device

    def clone_buffers(self) -> StagedDecodeStepMetadataPayload:
        """Allocate an independent fixed-address set outside capture."""
        return StagedDecodeStepMetadataPayload(
            host_payload=self.host_payload,
            query_start_loc=self.query_start_loc.clone(),
            block_tables=MappingProxyType({name: tensor.clone() for name, tensor in self.block_tables.items()}),
            start_positions=self.start_positions.clone(),
            kv_seq_lens=self.kv_seq_lens.clone(),
        )

    def launch_arguments(self) -> Mapping[str, torch.Tensor]:
        return MappingProxyType(
            {
                **self.block_tables,
                "start_positions": self.start_positions,
                "kv_seq_lens": self.kv_seq_lens,
            }
        )


@dataclass(slots=True)
class DecodeStepMetadataBufferOwner:
    """Fixed-address device metadata owner for one static-B graph.

    ``capture_actual`` is intentionally immutable: it describes the Host
    metadata object used to build the captured node.  ``current_actual`` is a
    diagnostic mirror of the payload most recently copied into device buffers;
    device kernels determine activity from ``kv_seq_lens`` and never consume
    either Python integer during replay.

    Applied staging sources are strongly retained because copies may be
    asynchronous.  The caller may clear them with
    :meth:`release_staging_owners_after_sync` only after an external stream or
    device synchronization proves all preceding copies complete.
    """

    bound_payload: StagedDecodeStepMetadataPayload
    capture_step_id: int
    capture_actual: int
    current_step_id: int
    current_actual: int
    current_request_ids: tuple[int, ...]
    _pending_staging_sources: list[StagedDecodeStepMetadataPayload] = field(
        default_factory=list,
        repr=False,
    )

    @classmethod
    def create(
        cls,
        payload: StagedDecodeStepMetadataPayload,
    ) -> DecodeStepMetadataBufferOwner:
        if not isinstance(payload, StagedDecodeStepMetadataPayload):
            raise TypeError(f"expected StagedDecodeStepMetadataPayload, got {type(payload).__name__}")
        bound = payload.clone_buffers()
        return cls(
            bound_payload=bound,
            capture_step_id=payload.step_id,
            capture_actual=payload.actual,
            current_step_id=payload.step_id,
            current_actual=payload.actual,
            current_request_ids=payload.host_payload.request_ids,
            _pending_staging_sources=[payload],
        )

    @property
    def pending_staging_owner_count(self) -> int:
        return len(self._pending_staging_sources)

    def device_metadata_addresses(self) -> Mapping[str, int]:
        tensors = {
            **self.bound_payload.block_tables,
            "start_positions": self.bound_payload.start_positions,
            "kv_seq_lens": self.bound_payload.kv_seq_lens,
            "query_start_loc": self.bound_payload.query_start_loc,
        }
        return MappingProxyType({name: int(tensor.data_ptr()) for name, tensor in tensors.items()})

    def apply_staged_payload(self, payload: StagedDecodeStepMetadataPayload) -> None:
        """Enqueue an allocation-free in-place metadata update.

        This method neither synchronizes nor allocates device tensors.  All
        validation precedes the first ``copy_`` so a rejected specialization,
        device, shape or query-row contract leaves the owner untouched.
        """
        if not isinstance(payload, StagedDecodeStepMetadataPayload):
            raise TypeError(f"expected StagedDecodeStepMetadataPayload, got {type(payload).__name__}")
        if payload.spec.key != self.bound_payload.spec.key:
            raise DecodeStepMetadataError("payload uses a different static B/cache/layout specialization")
        if payload.device != self.bound_payload.device:
            raise DecodeStepMetadataError(
                f"payload device mismatch: expected {self.bound_payload.device}, got {payload.device}"
            )
        if payload.host_payload.query_start_loc != self.bound_payload.host_payload.query_start_loc:
            raise DecodeStepMetadataError("query_start_loc changed inside one static-B owner")

        # Revalidate tensor metadata without reading a device value.  Users can
        # resize a Tensor even though the frozen staging wrapper is immutable.
        bucket = self.bound_payload.spec.batch
        _require_staged_tensor(
            payload.start_positions,
            name="start_positions",
            shape=(bucket,),
            device=self.bound_payload.device,
        )
        _require_staged_tensor(
            payload.kv_seq_lens,
            name="kv_seq_lens",
            shape=(bucket,),
            device=self.bound_payload.device,
        )
        for table in payload.host_payload.tables:
            _require_staged_tensor(
                payload.block_tables[table.argument_name],
                name=table.argument_name,
                shape=(bucket, table.width),
                device=self.bound_payload.device,
            )

        before = dict(self.device_metadata_addresses())
        for argument_name in TABLE_ARGUMENT_NAMES:
            self.bound_payload.block_tables[argument_name].copy_(payload.block_tables[argument_name])
        self.bound_payload.start_positions.copy_(payload.start_positions)
        self.bound_payload.kv_seq_lens.copy_(payload.kv_seq_lens)
        after = dict(self.device_metadata_addresses())
        if after != before:
            raise AssertionError("in-place metadata update changed a captured tensor address")

        self.current_step_id = payload.step_id
        self.current_actual = payload.actual
        self.current_request_ids = payload.host_payload.request_ids
        self._pending_staging_sources.append(payload)

    def release_staging_owners_after_sync(self) -> None:
        """Release copy sources after the caller has externally synchronized."""
        self._pending_staging_sources.clear()


__all__ = [
    "DEFAULT_PHYSICAL_BLOCK_OFFSETS",
    "DecodeStepMetadataBufferOwner",
    "DecodeStepMetadataError",
    "DecodeStepMetadataMaterializer",
    "DecodeStepMetadataPayload",
    "DecodeTraceMetadataRequirements",
    "FamilyMetadataRequirement",
    "INVALID_BLOCK_ID",
    "ProductionBlockTable",
    "StagedDecodeStepMetadataPayload",
    "TABLE_ARGUMENT_NAMES",
    "derive_trace_metadata_requirements",
    "materialize_decode_trace",
]
