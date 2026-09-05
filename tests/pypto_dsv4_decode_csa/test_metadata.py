# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Host-only contracts for stateful trace metadata materialization."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
import torch

from tests.pypto_dsv4_decode_csa.a3_partial_bucket_smoke import (
    build_partial_bucket_native_metadata,
)
from tests.pypto_dsv4_decode_csa.metadata import (
    DEFAULT_PHYSICAL_BLOCK_OFFSETS,
    TABLE_ARGUMENT_NAMES,
    DecodeStepMetadataBufferOwner,
    DecodeStepMetadataError,
    DecodeStepMetadataMaterializer,
    StagedDecodeStepMetadataPayload,
    derive_trace_metadata_requirements,
    materialize_decode_trace,
)
from tests.pypto_dsv4_decode_csa.trace import (
    CACHE_FAMILY_NAMES,
    COMPRESSOR_ATTENTION,
    INDEXER_CACHE,
    INNER_COMPRESSOR_STATE,
    MAIN_COMPRESSOR_STATE,
    SWA_CACHE,
    DecodeTrace,
    DecodeTraceBuilder,
    RequestAdmission,
    build_deterministic_churn_trace,
)
from vllm_ascend.ops._pypto_dsv4_csa import DecodeCSAProgramSpec
from vllm_ascend.ops._pypto_dsv4_csa.native_metadata import (
    bind_decode_csa_native_metadata,
)


def _capacities(value: int) -> dict[str, int]:
    return {family: value for family in CACHE_FAMILY_NAMES}


def _specs_for_trace(trace: DecodeTrace) -> dict[int, DecodeCSAProgramSpec]:
    requirements = derive_trace_metadata_requirements(trace)
    overrides = dict(requirements.program_spec_overrides)
    return {bucket: DecodeCSAProgramSpec(batch=bucket, **overrides) for bucket in set(trace.bucket_sequence)}


def _one_step_trace(*admissions: RequestAdmission) -> DecodeTrace:
    builder = DecodeTraceBuilder(_capacities(512), seed=17)
    builder.append_step(admit=admissions)
    return builder.build()


def _materialize_one(trace: DecodeTrace):
    step = trace.steps[0]
    return DecodeStepMetadataMaterializer(_specs_for_trace(trace)[step.bucket_size]).materialize(step)


def test_materializer_builds_full_bucket_production_metadata_with_explicit_A_and_B() -> None:
    trace = _one_step_trace(
        RequestAdmission(10, sequence_length_before=31),
        RequestAdmission(11, sequence_length_before=4),
        RequestAdmission(12, sequence_length_before=127),
        RequestAdmission(13, sequence_length_before=0),
        RequestAdmission(14, sequence_length_before=32),
    )
    payload = _materialize_one(trace)

    assert payload.actual == 5
    assert payload.bucket_size == 8
    assert payload.request_ids == (10, 11, 12, 13, 14, -1, -1, -1)
    assert payload.query_start_loc == tuple(range(0, 9 * 8, 8))
    assert payload.start_positions == (31, 4, 127, 0, 32, 0, 0, 0)
    assert payload.kv_seq_lens == (39, 12, 135, 8, 40, 0, 0, 0)
    assert tuple(payload.block_tables) == TABLE_ARGUMENT_NAMES

    for table in payload.tables:
        assert len(table.rows) == 8
        assert all(len(row) == table.width for row in table.rows)
        assert all(not any(row) for row in table.rows[5:])


def test_default_translation_reserves_state_block_zero_but_preserves_other_namespaces() -> None:
    payload = _materialize_one(_one_step_trace(RequestAdmission(0)))

    assert DEFAULT_PHYSICAL_BLOCK_OFFSETS == {
        COMPRESSOR_ATTENTION: 0,
        MAIN_COMPRESSOR_STATE: 1,
        INNER_COMPRESSOR_STATE: 1,
        INDEXER_CACHE: 0,
        SWA_CACHE: 0,
    }
    assert payload.table_for_family(COMPRESSOR_ATTENTION).rows[0][0] == 0
    assert payload.table_for_family(MAIN_COMPRESSOR_STATE).rows[0][0] == 1
    assert payload.table_for_family(INNER_COMPRESSOR_STATE).rows[0][0] == 1
    assert payload.table_for_family(INDEXER_CACHE).rows[0][0] == 0
    assert payload.table_for_family(SWA_CACHE).rows[0][0] == 0


def test_custom_nonzero_offsets_make_block_zero_a_canary_in_all_families() -> None:
    trace = _one_step_trace(RequestAdmission(0))
    spec = _specs_for_trace(trace)[4]
    offsets = {family: 1 for family in CACHE_FAMILY_NAMES}
    requirements = derive_trace_metadata_requirements(trace, physical_block_offsets=offsets)
    spec = DecodeCSAProgramSpec(batch=4, **dict(requirements.program_spec_overrides))

    payload = DecodeStepMetadataMaterializer(
        spec,
        physical_block_offsets=offsets,
    ).materialize(trace.steps[0])

    assert all(payload.table_for_family(family).rows[0][0] == 1 for family in CACHE_FAMILY_NAMES)
    assert all(not any(table.rows[1]) for table in payload.tables)


def test_host_payload_is_deeply_immutable_and_retains_the_source_step() -> None:
    payload = _materialize_one(_one_step_trace(RequestAdmission(7)))

    with pytest.raises(FrozenInstanceError):
        payload.request_ids = (99, 99, 99, 99)  # type: ignore[misc]
    with pytest.raises(TypeError):
        payload.block_tables["cmp_block_table"][0][0] = 99  # type: ignore[index]
    assert payload.source_step.active_requests[0].request_id == 7


def test_staging_is_canonical_int32_and_directly_compatible_with_production_binder() -> None:
    payload = _materialize_one(_one_step_trace(RequestAdmission(1), RequestAdmission(2), RequestAdmission(3)))
    staged = StagedDecodeStepMetadataPayload.from_host(payload)
    rope_cos = torch.zeros((128, 64), dtype=torch.float32)
    rope_sin = torch.zeros((128, 64), dtype=torch.float32)
    hadamard = torch.eye(128, dtype=torch.bfloat16)
    metadata = build_partial_bucket_native_metadata(
        staged,  # production-shape duck type, not a parallel metadata ABI
        full_rope_cos=rope_cos,
        full_rope_sin=rope_sin,
        hadamard=hadamard,
    )
    prepared = bind_decode_csa_native_metadata(metadata, payload.spec, device=torch.device("cpu"))

    assert staged.device == torch.device("cpu")
    assert staged.actual == 3
    assert staged.actual_tokens == 24
    assert staged.query_start_loc.dtype == torch.int32
    assert staged.query_start_loc.is_contiguous()
    for name in TABLE_ARGUMENT_NAMES:
        assert staged.block_tables[name].dtype == torch.int32
        assert staged.block_tables[name].is_contiguous()
        assert prepared.for_launch()[name] is staged.block_tables[name]
    assert prepared.for_launch()["start_positions"] is staged.start_positions
    assert prepared.for_launch()["kv_seq_lens"] is staged.kv_seq_lens


def test_stable_owner_updates_five_tables_and_positions_without_changing_addresses() -> None:
    builder = DecodeTraceBuilder(_capacities(512), seed=5)
    first = builder.append_step(admit=(RequestAdmission(0),))
    second = builder.append_step(admit=(RequestAdmission(1), RequestAdmission(2)))
    trace = builder.build()
    spec = _specs_for_trace(trace)[4]
    materializer = DecodeStepMetadataMaterializer(spec)
    first_staged = StagedDecodeStepMetadataPayload.from_host(materializer.materialize(first))
    second_staged = StagedDecodeStepMetadataPayload.from_host(materializer.materialize(second))
    owner = DecodeStepMetadataBufferOwner.create(first_staged)
    addresses = dict(owner.device_metadata_addresses())
    query_before = owner.bound_payload.query_start_loc.clone()

    owner.apply_staged_payload(second_staged)

    assert dict(owner.device_metadata_addresses()) == addresses
    assert owner.capture_step_id == 0
    assert owner.capture_actual == 1
    assert owner.current_step_id == 1
    assert owner.current_actual == 3
    assert owner.current_request_ids == (0, 1, 2, -1)
    assert owner.pending_staging_owner_count == 2
    torch.testing.assert_close(owner.bound_payload.query_start_loc, query_before)
    torch.testing.assert_close(owner.bound_payload.start_positions, second_staged.start_positions)
    torch.testing.assert_close(owner.bound_payload.kv_seq_lens, second_staged.kv_seq_lens)
    for name in TABLE_ARGUMENT_NAMES:
        torch.testing.assert_close(owner.bound_payload.block_tables[name], second_staged.block_tables[name])

    owner.release_staging_owners_after_sync()
    assert owner.pending_staging_owner_count == 0


def test_owner_rejects_another_bucket_before_mutating_any_buffer() -> None:
    small_trace = _one_step_trace(RequestAdmission(0))
    large_trace = _one_step_trace(*(RequestAdmission(index) for index in range(5)))
    small = StagedDecodeStepMetadataPayload.from_host(_materialize_one(small_trace))
    large = StagedDecodeStepMetadataPayload.from_host(_materialize_one(large_trace))
    owner = DecodeStepMetadataBufferOwner.create(small)
    before = {name: tensor.clone() for name, tensor in owner.bound_payload.launch_arguments().items()}

    with pytest.raises(DecodeStepMetadataError, match="different static B"):
        owner.apply_staged_payload(large)

    for name, expected in before.items():
        torch.testing.assert_close(owner.bound_payload.launch_arguments()[name], expected)
    assert owner.current_step_id == small.step_id
    assert owner.pending_staging_owner_count == 1


def test_owner_validates_resized_staging_before_copying_any_field() -> None:
    builder = DecodeTraceBuilder(_capacities(512))
    first = builder.append_step(admit=(RequestAdmission(0),))
    second = builder.append_step(admit=(RequestAdmission(1),))
    trace = builder.build()
    materializer = DecodeStepMetadataMaterializer(_specs_for_trace(trace)[4])
    owner = DecodeStepMetadataBufferOwner.create(
        StagedDecodeStepMetadataPayload.from_host(materializer.materialize(first))
    )
    invalid = StagedDecodeStepMetadataPayload.from_host(materializer.materialize(second))
    before = owner.bound_payload.start_positions.clone()
    invalid.start_positions.resize_(5)

    with pytest.raises(DecodeStepMetadataError, match="expected shape"):
        owner.apply_staged_payload(invalid)

    torch.testing.assert_close(owner.bound_payload.start_positions, before)
    assert owner.current_step_id == first.step_id


def test_lifecycle_materialization_preserves_advance_compacts_rows_and_reuses_blocks() -> None:
    builder = DecodeTraceBuilder(_capacities(512), seed=11)
    first = builder.append_step(admit=tuple(RequestAdmission(index) for index in range(4)))
    builder.append_step(retire=(0,), admit=(RequestAdmission(4),))
    builder.append_step(retire=(1, 2), admit=(RequestAdmission(5),))
    trace = builder.build()
    spec = _specs_for_trace(trace)[4]
    payloads = tuple(DecodeStepMetadataMaterializer(spec).materialize(step) for step in trace.steps)

    assert payloads[0].request_ids == (0, 1, 2, 3)
    assert payloads[1].request_ids == (1, 2, 3, 4)
    assert payloads[2].request_ids == (3, 4, 5, -1)
    assert payloads[1].start_positions == (8, 8, 8, 0)
    assert payloads[2].start_positions == (16, 8, 0, 0)
    assert payloads[2].kv_seq_lens == (24, 16, 8, 0)

    # Request 4 consumes request 0's released lowest-free blocks.  Request 1
    # compacts from row 1 to row 0 while retaining and extending its old prefix.
    for family in CACHE_FAMILY_NAMES:
        first_table = payloads[0].table_for_family(family)
        second_table = payloads[1].table_for_family(family)
        first_count = len(first.active_requests[0].ownership_for(family).logical_block_table)
        survivor_count = len(first.active_requests[1].ownership_for(family).logical_block_table)
        assert second_table.rows[3][:first_count] == first_table.rows[0][:first_count]
        assert second_table.rows[0][:survivor_count] == first_table.rows[1][:survivor_count]
        assert not any(payloads[2].table_for_family(family).rows[3])


@pytest.mark.parametrize("steps", (4, 32, 128))
def test_4_32_128_step_traces_materialize_without_runtime_or_NPU(steps: int) -> None:
    trace = build_deterministic_churn_trace(steps, seed=73)
    requirements = derive_trace_metadata_requirements(trace)
    payloads = materialize_decode_trace(trace, _specs_for_trace(trace))

    assert len(payloads) == steps
    assert tuple(payload.step_id for payload in payloads) == tuple(range(steps))
    assert tuple(payload.bucket_size for payload in payloads) == trace.bucket_sequence
    assert all(payload.actual <= payload.bucket_size for payload in payloads)
    assert all(requirement.table_width >= 1 for requirement in requirements.families)
    assert all(requirement.physical_blocks >= 1 for requirement in requirements.families)
    for payload in payloads:
        assert all(length > 0 for length in payload.kv_seq_lens[: payload.actual])
        assert all(length == 0 for length in payload.kv_seq_lens[payload.actual :])
        assert all(not any(row) for table in payload.tables for row in table.rows[payload.actual :])


def test_requirements_map_every_family_to_the_matching_program_spec_field() -> None:
    trace = build_deterministic_churn_trace(32, seed=2026)
    requirements = derive_trace_metadata_requirements(trace)
    overrides = requirements.program_spec_overrides

    assert set(overrides) == {
        "compressed_table_width",
        "main_state_table_width",
        "inner_state_table_width",
        "indexer_table_width",
        "swa_table_width",
        "compressed_blocks",
        "main_state_blocks",
        "inner_state_blocks",
        "indexer_blocks",
        "swa_blocks",
    }
    spec = DecodeCSAProgramSpec(batch=16, **dict(overrides))
    assert spec.main_state_blocks > spec.compressed_blocks
    assert spec.inner_state_blocks > spec.indexer_blocks


def test_materializer_fails_fast_on_width_capacity_bucket_and_offset_contracts() -> None:
    trace = _one_step_trace(RequestAdmission(0, sequence_length_before=8))
    step = trace.steps[0]
    requirements = derive_trace_metadata_requirements(trace)
    valid = dict(requirements.program_spec_overrides)

    with pytest.raises(DecodeStepMetadataError, match="cannot use a B8"):
        DecodeStepMetadataMaterializer(DecodeCSAProgramSpec(batch=8, **valid)).materialize(step)

    too_narrow = dict(valid)
    too_narrow["main_state_table_width"] = 1
    with pytest.raises(DecodeStepMetadataError, match="static width is 1"):
        DecodeStepMetadataMaterializer(DecodeCSAProgramSpec(batch=4, **too_narrow)).materialize(step)

    too_small = dict(valid)
    too_small["main_state_blocks"] = 1
    with pytest.raises(DecodeStepMetadataError, match="capacity=1"):
        DecodeStepMetadataMaterializer(DecodeCSAProgramSpec(batch=4, **too_small)).materialize(step)

    with pytest.raises(DecodeStepMetadataError, match="must name all five families"):
        DecodeStepMetadataMaterializer(
            DecodeCSAProgramSpec(batch=4, **valid),
            physical_block_offsets={SWA_CACHE: 0},
        )


def test_mixed_bucket_trace_requires_one_correct_spec_per_used_bucket() -> None:
    trace = build_deterministic_churn_trace(4, seed=9)
    specs = _specs_for_trace(trace)
    specs.pop(12)

    with pytest.raises(DecodeStepMetadataError, match=r"missing static program specs.*12"):
        materialize_decode_trace(trace, specs)
