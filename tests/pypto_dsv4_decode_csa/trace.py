# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Framework-neutral stateful decode traces for DeepSeek-V4 CSA tests.

The objects emitted by DecodeTraceBuilder are immutable Host records. They
describe scheduler-visible request state and five independent cache block
namespaces without importing torch, vLLM, PyPTO, or an NPU runtime. Device
metadata materialization is deliberately a later phase.
"""

from __future__ import annotations

import heapq
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

DECODE_SEQUENCE_LENGTH = 8
COMPRESS_RATIO = 4
BATCH_BUCKETS = (4, 8, 12, 16)
INVALID_REQUEST_ID = -1
INVALID_SEQUENCE_LENGTH = -1
INVALID_POSITION = -1

COMPRESSOR_ATTENTION = "compressor_attention"
MAIN_COMPRESSOR_STATE = "main_compressor_state"
INNER_COMPRESSOR_STATE = "inner_compressor_state"
INDEXER_CACHE = "indexer_cache"
SWA_CACHE = "swa_cache"


class DecodeTraceError(ValueError):
    """Base class for an invalid trace or lifecycle operation."""


class CacheBlockCapacityError(DecodeTraceError):
    """Raised when a cache family has too few free physical blocks."""


class CacheBlockStateError(DecodeTraceError):
    """Raised for duplicate request IDs, double-free, or invalid ownership."""


@dataclass(frozen=True, slots=True)
class CacheFamilySpec:
    """Paging rule for one independently allocated DSA metadata family."""

    name: str
    block_size: int
    position_divisor: int = 1

    def __post_init__(self) -> None:
        if not self.name:
            raise DecodeTraceError("cache family name must not be empty")
        if self.block_size <= 0:
            raise DecodeTraceError(f"{self.name}: block_size must be positive")
        if self.position_divisor <= 0:
            raise DecodeTraceError(f"{self.name}: position_divisor must be positive")

    def logical_row_count(self, sequence_length: int) -> int:
        """Return materialized rows after sequence_length source tokens.

        Ratio-4 cache rows exist only after a complete four-token group. State
        and SWA families retain one logical row per source token.
        """
        if sequence_length < 0:
            raise DecodeTraceError(f"sequence_length must be non-negative, got {sequence_length}")
        return sequence_length // self.position_divisor

    def required_block_count(self, sequence_length: int) -> int:
        rows = self.logical_row_count(sequence_length)
        return (rows + self.block_size - 1) // self.block_size


CACHE_FAMILY_SPECS = (
    CacheFamilySpec(COMPRESSOR_ATTENTION, block_size=32, position_divisor=COMPRESS_RATIO),
    CacheFamilySpec(MAIN_COMPRESSOR_STATE, block_size=2),
    CacheFamilySpec(INNER_COMPRESSOR_STATE, block_size=2),
    CacheFamilySpec(INDEXER_CACHE, block_size=32, position_divisor=COMPRESS_RATIO),
    CacheFamilySpec(SWA_CACHE, block_size=32),
)
CACHE_FAMILY_NAMES = tuple(spec.name for spec in CACHE_FAMILY_SPECS)
_CACHE_FAMILY_BY_NAME = {spec.name: spec for spec in CACHE_FAMILY_SPECS}


def required_block_counts(sequence_length: int) -> Mapping[str, int]:
    """Return an immutable five-family block requirement."""
    return MappingProxyType({spec.name: spec.required_block_count(sequence_length) for spec in CACHE_FAMILY_SPECS})


def select_batch_bucket(num_reqs_actual: int) -> int:
    """Select the smallest static L1 bucket that contains the active batch."""
    if isinstance(num_reqs_actual, bool) or not isinstance(num_reqs_actual, int):
        raise DecodeTraceError("num_reqs_actual must be an integer")
    if num_reqs_actual <= 0:
        raise DecodeTraceError("a decode step must contain at least one active request")
    for bucket in BATCH_BUCKETS:
        if num_reqs_actual <= bucket:
            return bucket
    raise DecodeTraceError(
        f"{num_reqs_actual} active requests exceed the largest supported B{BATCH_BUCKETS[-1]} bucket"
    )


@dataclass(frozen=True, slots=True)
class FamilyBlockOwnership:
    """One request's logical-block-to-physical-block table for a family."""

    family: str
    logical_block_table: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.family not in _CACHE_FAMILY_BY_NAME:
            raise DecodeTraceError(f"unknown cache family {self.family!r}")
        if any(
            isinstance(block, bool) or not isinstance(block, int) or block < 0 for block in self.logical_block_table
        ):
            raise DecodeTraceError(f"{self.family}: physical block IDs must be non-negative integers")
        if len(set(self.logical_block_table)) != len(self.logical_block_table):
            raise DecodeTraceError(f"{self.family}: one request cannot own the same physical block twice")

    @property
    def physical_blocks(self) -> tuple[int, ...]:
        return self.logical_block_table


def _empty_ownership() -> tuple[FamilyBlockOwnership, ...]:
    return tuple(FamilyBlockOwnership(family, ()) for family in CACHE_FAMILY_NAMES)


@dataclass(frozen=True, slots=True)
class RequestStep:
    """Immutable active or explicitly invalid padded request row."""

    row_index: int
    request_id: int
    sequence_length_before: int
    token_positions: tuple[int, ...]
    block_ownership: tuple[FamilyBlockOwnership, ...]
    active: bool

    def __post_init__(self) -> None:
        if isinstance(self.row_index, bool) or not isinstance(self.row_index, int) or self.row_index < 0:
            raise DecodeTraceError("row_index must be a non-negative integer")
        families = tuple(owner.family for owner in self.block_ownership)
        if families != CACHE_FAMILY_NAMES:
            raise DecodeTraceError(f"row {self.row_index}: ownership must list all five families in canonical order")
        if len(self.token_positions) != DECODE_SEQUENCE_LENGTH:
            raise DecodeTraceError(
                f"row {self.row_index}: expected S={DECODE_SEQUENCE_LENGTH} positions, got {len(self.token_positions)}"
            )

        if not self.active:
            if self.request_id != INVALID_REQUEST_ID:
                raise DecodeTraceError("padded rows must use the invalid request sentinel")
            if self.sequence_length_before != INVALID_SEQUENCE_LENGTH:
                raise DecodeTraceError("padded rows must use the invalid sequence-length sentinel")
            if any(position != INVALID_POSITION for position in self.token_positions):
                raise DecodeTraceError("padded rows must contain only invalid token positions")
            if any(owner.logical_block_table for owner in self.block_ownership):
                raise DecodeTraceError("padded rows cannot own cache blocks")
            return

        if isinstance(self.request_id, bool) or not isinstance(self.request_id, int) or self.request_id < 0:
            raise DecodeTraceError("active request IDs must be non-negative integers")
        if self.sequence_length_before < 0:
            raise DecodeTraceError("active sequence lengths must be non-negative")
        expected_positions = tuple(
            range(self.sequence_length_before, self.sequence_length_before + DECODE_SEQUENCE_LENGTH)
        )
        if self.token_positions != expected_positions:
            raise DecodeTraceError(
                f"request {self.request_id}: positions must be consecutive S={DECODE_SEQUENCE_LENGTH} rows"
            )
        expected_counts = required_block_counts(self.sequence_length_after)
        for owner in self.block_ownership:
            expected = expected_counts[owner.family]
            if len(owner.logical_block_table) != expected:
                raise DecodeTraceError(
                    f"request {self.request_id}/{owner.family}: expected {expected} blocks, "
                    f"got {len(owner.logical_block_table)}"
                )

    @property
    def sequence_length_after(self) -> int:
        if not self.active:
            return INVALID_SEQUENCE_LENGTH
        return self.sequence_length_before + DECODE_SEQUENCE_LENGTH

    def ownership_for(self, family: str) -> FamilyBlockOwnership:
        if family not in _CACHE_FAMILY_BY_NAME:
            raise DecodeTraceError(f"unknown cache family {family!r}")
        return self.block_ownership[CACHE_FAMILY_NAMES.index(family)]

    @classmethod
    def padded(cls, row_index: int) -> RequestStep:
        return cls(
            row_index=row_index,
            request_id=INVALID_REQUEST_ID,
            sequence_length_before=INVALID_SEQUENCE_LENGTH,
            token_positions=(INVALID_POSITION,) * DECODE_SEQUENCE_LENGTH,
            block_ownership=_empty_ownership(),
            active=False,
        )


@dataclass(frozen=True, slots=True)
class DecodeStep:
    """One compacted active batch plus explicit invalid padding rows."""

    step_id: int
    bucket_size: int
    num_reqs_actual: int
    requests: tuple[RequestStep, ...]
    retired_request_ids: tuple[int, ...]
    admitted_request_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if isinstance(self.step_id, bool) or not isinstance(self.step_id, int) or self.step_id < 0:
            raise DecodeTraceError("step_id must be a non-negative integer")
        if self.bucket_size != select_batch_bucket(self.num_reqs_actual):
            raise DecodeTraceError(
                f"step {self.step_id}: B{self.bucket_size} is not the minimal bucket for "
                f"{self.num_reqs_actual} requests"
            )
        if len(self.requests) != self.bucket_size:
            raise DecodeTraceError(f"step {self.step_id}: expected {self.bucket_size} rows, got {len(self.requests)}")
        if any(request.row_index != row for row, request in enumerate(self.requests)):
            raise DecodeTraceError(f"step {self.step_id}: request row_index values are not canonical")
        active = self.requests[: self.num_reqs_actual]
        padding = self.requests[self.num_reqs_actual :]
        if any(not request.active for request in active) or any(request.active for request in padding):
            raise DecodeTraceError(f"step {self.step_id}: active rows must precede padded rows")
        active_ids = tuple(request.request_id for request in active)
        if len(set(active_ids)) != len(active_ids):
            raise DecodeTraceError(f"step {self.step_id}: active request IDs must be unique")
        if len(set(self.admitted_request_ids)) != len(self.admitted_request_ids):
            raise DecodeTraceError(f"step {self.step_id}: admitted request IDs must be unique")
        if len(set(self.retired_request_ids)) != len(self.retired_request_ids):
            raise DecodeTraceError(f"step {self.step_id}: retired request IDs must be unique")
        if not set(self.admitted_request_ids).issubset(active_ids):
            raise DecodeTraceError(f"step {self.step_id}: every admitted request must be active")
        if set(self.retired_request_ids) & set(active_ids):
            raise DecodeTraceError(f"step {self.step_id}: retired requests cannot remain active")
        if set(self.retired_request_ids) & set(self.admitted_request_ids):
            raise DecodeTraceError(f"step {self.step_id}: a request cannot retire and readmit in one step")

        for family in CACHE_FAMILY_NAMES:
            physical_blocks = [
                block for request in active for block in request.ownership_for(family).logical_block_table
            ]
            if len(physical_blocks) != len(set(physical_blocks)):
                raise DecodeTraceError(f"step {self.step_id}/{family}: physical block ownership overlaps")

    @property
    def active_requests(self) -> tuple[RequestStep, ...]:
        return self.requests[: self.num_reqs_actual]

    @property
    def padded_requests(self) -> tuple[RequestStep, ...]:
        return self.requests[self.num_reqs_actual :]


@dataclass(frozen=True, slots=True)
class DecodeTrace:
    """A fully generated, transition-validated decode lifecycle."""

    seed: int
    steps: tuple[DecodeStep, ...]

    def __post_init__(self) -> None:
        if not self.steps:
            raise DecodeTraceError("DecodeTrace must contain at least one step")
        previous: dict[int, RequestStep] = {}
        seen: set[int] = set()
        for expected_step_id, step in enumerate(self.steps):
            if step.step_id != expected_step_id:
                raise DecodeTraceError("DecodeTrace step IDs must be contiguous and zero-based")
            current = {request.request_id: request for request in step.active_requests}
            retired = set(step.retired_request_ids)
            admitted = set(step.admitted_request_ids)
            if not retired.issubset(previous):
                raise DecodeTraceError(f"step {step.step_id}: retired request was not active")
            if admitted & seen:
                raise DecodeTraceError(f"step {step.step_id}: request IDs cannot be reused after retirement")
            expected_ids = (set(previous) - retired) | admitted
            if set(current) != expected_ids:
                raise DecodeTraceError(f"step {step.step_id}: request lifecycle transition is inconsistent")

            for request_id in set(previous) - retired:
                old = previous[request_id]
                new = current[request_id]
                if new.sequence_length_before != old.sequence_length_after:
                    raise DecodeTraceError(
                        f"step {step.step_id}: request {request_id} did not advance by S={DECODE_SEQUENCE_LENGTH}"
                    )
                for family in CACHE_FAMILY_NAMES:
                    old_table = old.ownership_for(family).logical_block_table
                    new_table = new.ownership_for(family).logical_block_table
                    if new_table[: len(old_table)] != old_table:
                        raise DecodeTraceError(
                            f"step {step.step_id}: request {request_id}/{family} remapped a live block"
                        )
            seen.update(admitted)
            previous = current

    @property
    def bucket_sequence(self) -> tuple[int, ...]:
        return tuple(step.bucket_size for step in self.steps)


class CacheBlockAllocator:
    """Deterministic lowest-free-block allocator for five cache namespaces."""

    def __init__(self, capacities: Mapping[str, int]) -> None:
        if set(capacities) != set(CACHE_FAMILY_NAMES):
            missing = sorted(set(CACHE_FAMILY_NAMES) - set(capacities))
            extra = sorted(set(capacities) - set(CACHE_FAMILY_NAMES))
            raise DecodeTraceError(
                f"capacities must name exactly five cache families; missing={missing}, extra={extra}"
            )
        normalized: dict[str, int] = {}
        for family in CACHE_FAMILY_NAMES:
            capacity = capacities[family]
            if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 0:
                raise DecodeTraceError(f"{family}: capacity must be a non-negative integer")
            normalized[family] = capacity
        self._capacities = normalized
        self._free = {family: list(range(normalized[family])) for family in CACHE_FAMILY_NAMES}
        self._owned: dict[int, dict[str, tuple[int, ...]]] = {}
        self._seen_request_ids: set[int] = set()

    @property
    def capacities(self) -> Mapping[str, int]:
        return MappingProxyType(dict(self._capacities))

    @property
    def active_request_ids(self) -> tuple[int, ...]:
        return tuple(self._owned)

    def clone(self) -> CacheBlockAllocator:
        duplicate = CacheBlockAllocator(self._capacities)
        duplicate._free = {family: list(blocks) for family, blocks in self._free.items()}
        duplicate._owned = {
            request_id: {family: tuple(blocks) for family, blocks in ownership.items()}
            for request_id, ownership in self._owned.items()
        }
        duplicate._seen_request_ids = set(self._seen_request_ids)
        return duplicate

    def admit(self, request_id: int, block_counts: Mapping[str, int]) -> tuple[FamilyBlockOwnership, ...]:
        self._validate_request_id(request_id)
        if request_id in self._seen_request_ids:
            raise CacheBlockStateError(f"request {request_id} was already admitted")
        counts = self._normalize_block_counts(block_counts)
        self._preflight_growth({}, counts)
        ownership = {family: self._allocate(family, counts[family]) for family in CACHE_FAMILY_NAMES}
        self._owned[request_id] = ownership
        self._seen_request_ids.add(request_id)
        self.assert_consistent()
        return self.snapshot(request_id)

    def ensure(
        self,
        request_id: int,
        block_counts: Mapping[str, int],
    ) -> tuple[FamilyBlockOwnership, ...]:
        if request_id not in self._owned:
            raise CacheBlockStateError(f"request {request_id} is not active")
        counts = self._normalize_block_counts(block_counts)
        current = self._owned[request_id]
        for family in CACHE_FAMILY_NAMES:
            if counts[family] < len(current[family]):
                raise CacheBlockStateError(
                    f"request {request_id}/{family}: live ownership cannot shrink before retirement"
                )
        self._preflight_growth(current, counts)
        for family in CACHE_FAMILY_NAMES:
            additional = counts[family] - len(current[family])
            if additional:
                current[family] = current[family] + self._allocate(family, additional)
        self.assert_consistent()
        return self.snapshot(request_id)

    def release(self, request_id: int) -> tuple[FamilyBlockOwnership, ...]:
        if request_id not in self._owned:
            raise CacheBlockStateError(f"request {request_id} is not active; double-free is forbidden")
        released = self.snapshot(request_id)
        ownership = self._owned.pop(request_id)
        for family in CACHE_FAMILY_NAMES:
            for block in ownership[family]:
                heapq.heappush(self._free[family], block)
        self.assert_consistent()
        return released

    def snapshot(self, request_id: int) -> tuple[FamilyBlockOwnership, ...]:
        if request_id not in self._owned:
            raise CacheBlockStateError(f"request {request_id} is not active")
        return tuple(
            FamilyBlockOwnership(family, tuple(self._owned[request_id][family])) for family in CACHE_FAMILY_NAMES
        )

    def free_blocks(self, family: str) -> tuple[int, ...]:
        self._require_family(family)
        return tuple(sorted(self._free[family]))

    def owners(self, family: str) -> Mapping[int, int]:
        self._require_family(family)
        result: dict[int, int] = {}
        for request_id, ownership in self._owned.items():
            for block in ownership[family]:
                result[block] = request_id
        return MappingProxyType(result)

    def assert_consistent(self) -> None:
        for family in CACHE_FAMILY_NAMES:
            free = self._free[family]
            if len(free) != len(set(free)):
                raise CacheBlockStateError(f"{family}: free list contains duplicate blocks")
            owned = [block for request in self._owned.values() for block in request[family]]
            if len(owned) != len(set(owned)):
                raise CacheBlockStateError(f"{family}: one physical block has multiple owners")
            all_blocks = free + owned
            capacity = self._capacities[family]
            if len(all_blocks) != capacity or set(all_blocks) != set(range(capacity)):
                raise CacheBlockStateError(f"{family}: free and owned blocks do not partition capacity")

    def _preflight_growth(
        self,
        current: Mapping[str, tuple[int, ...]],
        desired: Mapping[str, int],
    ) -> None:
        for family in CACHE_FAMILY_NAMES:
            additional = desired[family] - len(current.get(family, ()))
            if additional > len(self._free[family]):
                raise CacheBlockCapacityError(
                    f"{family}: requested {additional} additional blocks with only {len(self._free[family])} free"
                )

    def _allocate(self, family: str, count: int) -> tuple[int, ...]:
        return tuple(heapq.heappop(self._free[family]) for _ in range(count))

    @staticmethod
    def _validate_request_id(request_id: int) -> None:
        if isinstance(request_id, bool) or not isinstance(request_id, int) or request_id < 0:
            raise CacheBlockStateError("request IDs must be non-negative integers")

    @staticmethod
    def _normalize_block_counts(block_counts: Mapping[str, int]) -> dict[str, int]:
        if set(block_counts) != set(CACHE_FAMILY_NAMES):
            raise DecodeTraceError("block_counts must name exactly the five cache families")
        normalized: dict[str, int] = {}
        for family in CACHE_FAMILY_NAMES:
            count = block_counts[family]
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise DecodeTraceError(f"{family}: block count must be a non-negative integer")
            normalized[family] = count
        return normalized

    @staticmethod
    def _require_family(family: str) -> None:
        if family not in _CACHE_FAMILY_BY_NAME:
            raise DecodeTraceError(f"unknown cache family {family!r}")


@dataclass(frozen=True, slots=True)
class RequestAdmission:
    request_id: int
    sequence_length_before: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.request_id, bool) or not isinstance(self.request_id, int) or self.request_id < 0:
            raise DecodeTraceError("admitted request IDs must be non-negative integers")
        if (
            isinstance(self.sequence_length_before, bool)
            or not isinstance(self.sequence_length_before, int)
            or self.sequence_length_before < 0
        ):
            raise DecodeTraceError("admitted sequence lengths must be non-negative integers")


class DecodeTraceBuilder:
    """Transactional Host builder whose published step records are immutable."""

    def __init__(self, capacities: Mapping[str, int], *, seed: int = 0) -> None:
        self._allocator = CacheBlockAllocator(capacities)
        self._active_sequence_lengths: dict[int, int] = {}
        self._steps: list[DecodeStep] = []
        self._seed = seed

    @property
    def active_request_ids(self) -> tuple[int, ...]:
        return tuple(self._active_sequence_lengths)

    def append_step(
        self,
        *,
        admit: Sequence[RequestAdmission] = (),
        retire: Sequence[int] = (),
    ) -> DecodeStep:
        admissions = tuple(admit)
        retired = tuple(retire)
        admitted_ids = tuple(admission.request_id for admission in admissions)
        if len(set(admitted_ids)) != len(admitted_ids):
            raise CacheBlockStateError("one step cannot admit the same request twice")
        if len(set(retired)) != len(retired):
            raise CacheBlockStateError("one step cannot retire the same request twice")
        if set(admitted_ids) & set(retired):
            raise CacheBlockStateError("one step cannot retire and readmit the same request")

        allocator = self._allocator.clone()
        active = dict(self._active_sequence_lengths)
        for request_id in retired:
            if request_id not in active:
                raise CacheBlockStateError(f"request {request_id} is not active; double-retire is forbidden")
            allocator.release(request_id)
            del active[request_id]

        for admission in admissions:
            if admission.request_id in active:
                raise CacheBlockStateError(f"request {admission.request_id} is already active")
            active[admission.request_id] = admission.sequence_length_before

        bucket = select_batch_bucket(len(active))

        # Allocate new requests first, after retire, so released low block IDs
        # are deterministically reusable by the replacement request.
        for admission in admissions:
            after = admission.sequence_length_before + DECODE_SEQUENCE_LENGTH
            allocator.admit(admission.request_id, required_block_counts(after))
        admitted_set = set(admitted_ids)
        for request_id, sequence_length_before in active.items():
            if request_id in admitted_set:
                continue
            allocator.ensure(
                request_id,
                required_block_counts(sequence_length_before + DECODE_SEQUENCE_LENGTH),
            )

        rows: list[RequestStep] = []
        next_lengths: dict[int, int] = {}
        for row_index, (request_id, sequence_length_before) in enumerate(active.items()):
            rows.append(
                RequestStep(
                    row_index=row_index,
                    request_id=request_id,
                    sequence_length_before=sequence_length_before,
                    token_positions=tuple(
                        range(sequence_length_before, sequence_length_before + DECODE_SEQUENCE_LENGTH)
                    ),
                    block_ownership=allocator.snapshot(request_id),
                    active=True,
                )
            )
            next_lengths[request_id] = sequence_length_before + DECODE_SEQUENCE_LENGTH
        rows.extend(RequestStep.padded(row) for row in range(len(rows), bucket))
        step = DecodeStep(
            step_id=len(self._steps),
            bucket_size=bucket,
            num_reqs_actual=len(active),
            requests=tuple(rows),
            retired_request_ids=retired,
            admitted_request_ids=admitted_ids,
        )

        self._allocator = allocator
        self._active_sequence_lengths = next_lengths
        self._steps.append(step)
        return step

    def build(self) -> DecodeTrace:
        return DecodeTrace(seed=self._seed, steps=tuple(self._steps))


def default_trace_capacities() -> Mapping[str, int]:
    """Generous integer-only capacities for a 128-step B16 Host trace."""
    return MappingProxyType(
        {
            COMPRESSOR_ATTENTION: 512,
            MAIN_COMPRESSOR_STATE: 4_096,
            INNER_COMPRESSOR_STATE: 4_096,
            INDEXER_CACHE: 512,
            SWA_CACHE: 512,
        }
    )


def build_deterministic_churn_trace(
    num_steps: int,
    *,
    seed: int = 0,
    capacities: Mapping[str, int] | None = None,
    admission_prefixes: Sequence[int] | None = None,
) -> DecodeTrace:
    """Build a reproducible bucket-switching request-churn trace.

    The first four steps select B4/B8/B12/B16 while deliberately keeping
    ``num_reqs_actual < bucket_size``. Later shrink/grow transitions retire
    requests and force newly admitted IDs to consume returned low blocks.
    The default Host-only admission prefix lengths cover ratio-4 and page
    boundaries while remaining cheap because the model stores only integer
    ownership records.  Device runners that initialize newly admitted cache to
    an empty state must pass ``admission_prefixes=(0,)``; a non-zero prefix is
    only semantically valid when its prefill history has actually been loaded.
    """
    if isinstance(num_steps, bool) or not isinstance(num_steps, int) or num_steps <= 0:
        raise DecodeTraceError("num_steps must be a positive integer")
    rng = random.Random(seed)
    builder = DecodeTraceBuilder(capacities or default_trace_capacities(), seed=seed)
    # Boundary counts such as 4/8/12/16 only prove bucket switching; they do
    # not exercise graph padding.  These targets retain the same bucket cycle
    # while making every generated step a real A<B partial bucket.
    target_cycle = (3, 7, 11, 15, 9, 6, 2, 5)
    prefixes = (0, 1, 3, 4, 24, 31, 32, 120, 127, 128) if admission_prefixes is None else tuple(admission_prefixes)
    if not prefixes:
        raise DecodeTraceError("admission_prefixes must not be empty")
    if any(isinstance(prefix, bool) or not isinstance(prefix, int) or prefix < 0 for prefix in prefixes):
        raise DecodeTraceError("admission_prefixes must contain only non-negative integers")
    next_request_id = 0

    for step_id in range(num_steps):
        target = target_cycle[step_id % len(target_cycle)]
        current = builder.active_request_ids

        # A monotonically growing first bucket cycle only proves static-B
        # selection; it does not exercise request retirement, row compaction or
        # same-step block reuse.  Once the initial three requests exist, replace
        # at least one live request on every step.  Retire from the middle before
        # wrapping toward row 0 so a surviving request moves to a lower compacted
        # row.  DecodeTraceBuilder releases before admitting, therefore the first
        # replacement deterministically consumes the returned lowest-free blocks.
        shrink_count = max(0, len(current) - target)
        replacement_count = 1 if current else 0
        retire_count = shrink_count + replacement_count
        if retire_count:
            retirement_order = current[1:] + current[:1]
            retired = retirement_order[:retire_count]
        else:
            retired = ()
        remaining = len(current) - retire_count
        admissions = tuple(
            RequestAdmission(
                request_id=next_request_id + offset,
                sequence_length_before=rng.choice(prefixes),
            )
            for offset in range(target - remaining)
        )
        next_request_id += len(admissions)
        builder.append_step(admit=admissions, retire=retired)

    return builder.build()


__all__ = [
    "BATCH_BUCKETS",
    "CACHE_FAMILY_NAMES",
    "CACHE_FAMILY_SPECS",
    "COMPRESSOR_ATTENTION",
    "COMPRESS_RATIO",
    "CacheBlockAllocator",
    "CacheBlockCapacityError",
    "CacheBlockStateError",
    "CacheFamilySpec",
    "DECODE_SEQUENCE_LENGTH",
    "DecodeStep",
    "DecodeTrace",
    "DecodeTraceBuilder",
    "DecodeTraceError",
    "FamilyBlockOwnership",
    "INDEXER_CACHE",
    "INNER_COMPRESSOR_STATE",
    "INVALID_POSITION",
    "INVALID_REQUEST_ID",
    "INVALID_SEQUENCE_LENGTH",
    "MAIN_COMPRESSOR_STATE",
    "RequestAdmission",
    "RequestStep",
    "SWA_CACHE",
    "build_deterministic_churn_trace",
    "default_trace_capacities",
    "required_block_counts",
    "select_batch_bucket",
]
