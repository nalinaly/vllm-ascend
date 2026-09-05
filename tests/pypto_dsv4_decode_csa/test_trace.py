# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Host-only lifecycle tests for the stateful DeepSeek-V4 CSA trace."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from tests.pypto_dsv4_decode_csa.trace import (
    BATCH_BUCKETS,
    CACHE_FAMILY_NAMES,
    COMPRESSOR_ATTENTION,
    INDEXER_CACHE,
    INNER_COMPRESSOR_STATE,
    INVALID_POSITION,
    INVALID_REQUEST_ID,
    INVALID_SEQUENCE_LENGTH,
    MAIN_COMPRESSOR_STATE,
    SWA_CACHE,
    CacheBlockAllocator,
    CacheBlockCapacityError,
    CacheBlockStateError,
    DecodeTraceBuilder,
    DecodeTraceError,
    FamilyBlockOwnership,
    RequestAdmission,
    RequestStep,
    build_deterministic_churn_trace,
    required_block_counts,
    select_batch_bucket,
)


def _capacities(value: int) -> dict[str, int]:
    return {family: value for family in CACHE_FAMILY_NAMES}


def _ownership_by_family(request: RequestStep) -> dict[str, tuple[int, ...]]:
    return {family: request.ownership_for(family).logical_block_table for family in CACHE_FAMILY_NAMES}


@pytest.mark.parametrize(
    ("actual", "expected"),
    (
        (1, 4),
        (4, 4),
        (5, 8),
        (8, 8),
        (9, 12),
        (12, 12),
        (13, 16),
        (16, 16),
    ),
)
def test_bucket_selection_uses_the_smallest_static_specialization(
    actual: int,
    expected: int,
) -> None:
    assert select_batch_bucket(actual) == expected


@pytest.mark.parametrize("actual", (0, -1, 17, True, 1.5))
def test_bucket_selection_rejects_an_unrepresentable_batch(actual: object) -> None:
    with pytest.raises(DecodeTraceError):
        select_batch_bucket(actual)  # type: ignore[arg-type]


def test_ratio4_and_five_family_block_rules_are_explicit() -> None:
    at_three = required_block_counts(3)
    assert at_three == {
        COMPRESSOR_ATTENTION: 0,
        MAIN_COMPRESSOR_STATE: 2,
        INNER_COMPRESSOR_STATE: 2,
        INDEXER_CACHE: 0,
        SWA_CACHE: 1,
    }

    at_four = required_block_counts(4)
    assert at_four[COMPRESSOR_ATTENTION] == 1
    assert at_four[INDEXER_CACHE] == 1
    assert at_four[MAIN_COMPRESSOR_STATE] == 2
    assert at_four[INNER_COMPRESSOR_STATE] == 2

    at_128 = required_block_counts(128)
    assert at_128 == {
        COMPRESSOR_ATTENTION: 1,
        MAIN_COMPRESSOR_STATE: 64,
        INNER_COMPRESSOR_STATE: 64,
        INDEXER_CACHE: 1,
        SWA_CACHE: 4,
    }
    with pytest.raises(TypeError):
        at_128[SWA_CACHE] = 99  # type: ignore[index]


def test_allocator_is_deterministic_and_reuses_the_lowest_released_blocks() -> None:
    counts = {family: 1 for family in CACHE_FAMILY_NAMES}

    def run() -> tuple[tuple[FamilyBlockOwnership, ...], tuple[FamilyBlockOwnership, ...]]:
        allocator = CacheBlockAllocator(_capacities(3))
        first = allocator.admit(10, counts)
        allocator.admit(11, counts)
        released = allocator.release(10)
        replacement = allocator.admit(12, counts)
        assert replacement == released
        allocator.assert_consistent()
        return first, replacement

    assert run() == run()


def test_allocator_rejects_double_free_repeated_request_and_capacity_exhaustion() -> None:
    counts = {family: 1 for family in CACHE_FAMILY_NAMES}
    allocator = CacheBlockAllocator(_capacities(1))
    allocator.admit(1, counts)

    with pytest.raises(CacheBlockStateError, match="already admitted"):
        allocator.admit(1, counts)
    with pytest.raises(CacheBlockCapacityError):
        allocator.admit(2, counts)
    assert allocator.active_request_ids == (1,)
    allocator.assert_consistent()

    allocator.release(1)
    with pytest.raises(CacheBlockStateError, match="double-free"):
        allocator.release(1)
    with pytest.raises(CacheBlockStateError, match="already admitted"):
        allocator.admit(1, counts)
    with pytest.raises(CacheBlockStateError, match="not active"):
        allocator.ensure(99, counts)


def test_builder_failure_is_transactional_when_capacity_is_insufficient() -> None:
    capacities = _capacities(8)
    capacities[MAIN_COMPRESSOR_STATE] = 3
    builder = DecodeTraceBuilder(capacities)

    with pytest.raises(CacheBlockCapacityError):
        builder.append_step(admit=(RequestAdmission(0),))
    assert builder.active_request_ids == ()
    with pytest.raises(DecodeTraceError, match="at least one step"):
        builder.build()


def test_actual_batch_and_padded_rows_are_separate_and_padding_owns_nothing() -> None:
    builder = DecodeTraceBuilder(_capacities(256))
    step = builder.append_step(admit=tuple(RequestAdmission(request_id) for request_id in range(5)))

    assert step.bucket_size == 8
    assert step.num_reqs_actual == 5
    assert len(step.active_requests) == 5
    assert len(step.padded_requests) == 3
    for row in step.padded_requests:
        assert not row.active
        assert row.request_id == INVALID_REQUEST_ID
        assert row.sequence_length_before == INVALID_SEQUENCE_LENGTH
        assert row.token_positions == (INVALID_POSITION,) * 8
        assert all(not owner.logical_block_table for owner in row.block_ownership)

    first = step.active_requests[0]
    assert first.token_positions == tuple(range(8))
    assert {family: len(blocks) for family, blocks in _ownership_by_family(first).items()} == {
        COMPRESSOR_ATTENTION: 1,
        MAIN_COMPRESSOR_STATE: 4,
        INNER_COMPRESSOR_STATE: 4,
        INDEXER_CACHE: 1,
        SWA_CACHE: 1,
    }


def test_padded_row_cannot_be_constructed_with_block_ownership() -> None:
    invalid_ownership = tuple(
        FamilyBlockOwnership(family, (0,) if family == SWA_CACHE else ()) for family in CACHE_FAMILY_NAMES
    )
    with pytest.raises(DecodeTraceError, match="padded rows cannot own"):
        RequestStep(
            row_index=0,
            request_id=INVALID_REQUEST_ID,
            sequence_length_before=INVALID_SEQUENCE_LENGTH,
            token_positions=(INVALID_POSITION,) * 8,
            block_ownership=invalid_ownership,
            active=False,
        )


def test_retire_then_admit_reuses_all_five_released_block_tables() -> None:
    builder = DecodeTraceBuilder(_capacities(128))
    first = builder.append_step(admit=tuple(RequestAdmission(request_id) for request_id in range(4)))
    retired_ownership = _ownership_by_family(first.active_requests[0])

    second = builder.append_step(
        retire=(0,),
        admit=(RequestAdmission(4),),
    )
    replacement = next(request for request in second.active_requests if request.request_id == 4)
    assert _ownership_by_family(replacement) == retired_ownership
    assert second.bucket_size == 4
    assert second.retired_request_ids == (0,)
    assert second.admitted_request_ids == (4,)
    builder.build()


def test_trace_records_are_frozen_after_publication() -> None:
    trace = build_deterministic_churn_trace(4, seed=7)
    with pytest.raises(FrozenInstanceError):
        trace.steps[0].bucket_size = 8  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        trace.steps[0].requests[0].active = False  # type: ignore[misc]


def test_seeded_trace_is_reproducible_and_exercises_all_buckets() -> None:
    first = build_deterministic_churn_trace(32, seed=2026)
    second = build_deterministic_churn_trace(32, seed=2026)
    different = build_deterministic_churn_trace(32, seed=2027)

    assert first == second
    assert first.steps != different.steps
    assert first.bucket_sequence[:4] == BATCH_BUCKETS
    assert set(first.bucket_sequence) == set(BATCH_BUCKETS)
    assert all(step.num_reqs_actual < step.bucket_size for step in first.steps)


def test_first_bucket_cycle_already_retires_compacts_replaces_and_reuses() -> None:
    trace = build_deterministic_churn_trace(4, seed=73)

    assert trace.bucket_sequence == BATCH_BUCKETS
    assert not trace.steps[0].retired_request_ids
    assert all(step.retired_request_ids for step in trace.steps[1:])
    assert all(step.admitted_request_ids for step in trace.steps[1:])

    previous_rows = {request.request_id: request.row_index for request in trace.steps[0].active_requests}
    compacted_survivors = []
    reused_by_family: dict[str, bool] = {family: False for family in CACHE_FAMILY_NAMES}
    for step in trace.steps[1:]:
        current_rows = {request.request_id: request.row_index for request in step.active_requests}
        compacted_survivors.extend(
            request_id
            for request_id in set(previous_rows) & set(current_rows)
            if current_rows[request_id] < previous_rows[request_id]
        )

        previous_step = trace.steps[step.step_id - 1]
        retired = {
            request.request_id: request
            for request in previous_step.active_requests
            if request.request_id in step.retired_request_ids
        }
        admitted = {
            request.request_id: request
            for request in step.active_requests
            if request.request_id in step.admitted_request_ids
        }
        for family in CACHE_FAMILY_NAMES:
            released = {
                block for request in retired.values() for block in request.ownership_for(family).logical_block_table
            }
            acquired = {
                block for request in admitted.values() for block in request.ownership_for(family).logical_block_table
            }
            reused_by_family[family] |= bool(released & acquired)
        previous_rows = current_rows

    assert compacted_survivors
    assert all(reused_by_family.values())


def test_device_empty_admission_mode_never_invents_unloaded_prefix_state() -> None:
    trace = build_deterministic_churn_trace(32, seed=73, admission_prefixes=(0,))

    for step in trace.steps:
        admitted = {
            request.request_id: request
            for request in step.active_requests
            if request.request_id in step.admitted_request_ids
        }
        assert admitted
        assert all(request.sequence_length_before == 0 for request in admitted.values())

    with pytest.raises(DecodeTraceError, match="must not be empty"):
        build_deterministic_churn_trace(4, admission_prefixes=())
    with pytest.raises(DecodeTraceError, match="non-negative integers"):
        build_deterministic_churn_trace(4, admission_prefixes=(0, -1))


@pytest.mark.parametrize("num_steps", (4, 32, 128))
def test_long_trace_invariants_hold_for_requested_step_counts(num_steps: int) -> None:
    trace = build_deterministic_churn_trace(num_steps, seed=73)
    assert len(trace.steps) == num_steps

    block_users: dict[str, dict[int, set[int]]] = {family: {} for family in CACHE_FAMILY_NAMES}
    for step in trace.steps:
        assert step.bucket_size == select_batch_bucket(step.num_reqs_actual)
        assert len(step.requests) == step.bucket_size
        for request in step.active_requests:
            assert len(request.token_positions) == 8
            assert request.token_positions == tuple(
                range(request.sequence_length_before, request.sequence_length_after)
            )
            expected = required_block_counts(request.sequence_length_after)
            for family in CACHE_FAMILY_NAMES:
                blocks = request.ownership_for(family).logical_block_table
                assert len(blocks) == expected[family]
                for block in blocks:
                    block_users[family].setdefault(block, set()).add(request.request_id)
        for padded in step.padded_requests:
            assert padded.token_positions == (INVALID_POSITION,) * 8
            assert all(not padded.ownership_for(family).logical_block_table for family in CACHE_FAMILY_NAMES)
        for family in CACHE_FAMILY_NAMES:
            live_blocks = [
                block for request in step.active_requests for block in request.ownership_for(family).logical_block_table
            ]
            assert len(live_blocks) == len(set(live_blocks))

    if num_steps >= 4:
        for family in CACHE_FAMILY_NAMES:
            assert any(len(request_ids) > 1 for request_ids in block_users[family].values())
