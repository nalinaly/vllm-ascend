# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Host contracts for the real stateful request-churn A3 runner."""

from __future__ import annotations

import inspect

import pytest
import torch

from tests.pypto_dsv4_decode_csa.a3_stateful_churn_compare import (
    ALL_FAMILY_BLOCK_OFFSETS,
    A3StatefulChurnResult,
    A3StatefulStaticExtentOverrides,
    StateRowDelta,
    _compare_step_states,
    apply_state_deltas,
    build_a3_stateful_churn_diagnostic_plan,
    build_a3_stateful_churn_plan,
    build_parser,
    compute_state_deltas,
    expected_mutated_rows,
    initialize_acquired_blocks,
    run_a3_stateful_churn_compare,
)
from tests.pypto_dsv4_decode_csa.native_fixture import STATE_NAMES
from tests.pypto_dsv4_decode_csa.trace import (
    CACHE_FAMILY_NAMES,
    COMPRESSOR_ATTENTION,
    INDEXER_CACHE,
    INNER_COMPRESSOR_STATE,
    MAIN_COMPRESSOR_STATE,
    SWA_CACHE,
)


@pytest.mark.parametrize("steps", (4, 32, 128))
def test_plan_materializes_real_partial_bucket_churn_for_supported_lengths(steps: int) -> None:
    plan = build_a3_stateful_churn_plan(steps=steps, seed=73)

    assert plan.spec_trace is plan.trace or plan.spec_trace == plan.trace
    assert plan.spec_steps == steps
    assert plan.execute_steps == steps
    assert plan.native_precondition_steps == steps
    assert plan.retained_stage_steps == steps
    assert len(plan.trace.steps) == steps
    assert len(plan.payloads) == steps
    assert len(plan.initializations) == steps
    assert plan.native_precondition_payloads == plan.payloads
    assert plan.native_precondition_initializations == plan.initializations
    assert plan.retained_stage_payloads == plan.payloads
    assert plan.buckets == (4, 8, 12, 16)
    assert tuple(plan.physical_block_offsets) == CACHE_FAMILY_NAMES
    assert all(offset == 1 for offset in plan.physical_block_offsets.values())
    assert all(step.num_reqs_actual < step.bucket_size for step in plan.trace.steps)
    assert tuple(payload.step_id for payload in plan.payloads) == tuple(range(steps))
    family_keys = {spec.key[1:] for spec in plan.specs_by_bucket.values()}
    assert len(family_keys) == 1
    assert plan.reuse_event_count > 0


@pytest.mark.parametrize("execute_steps", (1, 4))
def test_128_step_static_spec_can_execute_only_an_exact_trace_prefix(execute_steps: int) -> None:
    isolated = build_a3_stateful_churn_plan(
        spec_steps=128,
        execute_steps=execute_steps,
        seed=20260903,
    )
    full = build_a3_stateful_churn_plan(steps=128, seed=20260903)

    assert isolated.spec_steps == 128
    assert isolated.execute_steps == execute_steps
    assert isolated.native_precondition_steps == execute_steps
    assert isolated.retained_stage_steps == execute_steps
    assert isolated.trace.steps == isolated.spec_trace.steps[:execute_steps]
    assert isolated.trace.steps == full.trace.steps[:execute_steps]
    assert isolated.payloads == full.payloads[:execute_steps]
    assert isolated.initializations == full.initializations[:execute_steps]
    assert isolated.specs_by_bucket == full.specs_by_bucket
    assert isolated.buckets == (4, 8, 12, 16)
    assert tuple(payload.spec.batch for payload in isolated.warmup_payloads) == isolated.buckets

    spec = isolated.specs_by_bucket[4]
    assert (
        spec.compressed_blocks,
        spec.swa_blocks,
        spec.main_state_blocks,
        spec.inner_state_blocks,
        spec.indexer_blocks,
    ) == (23, 46, 605, 605, 23)
    assert (
        spec.compressed_table_width,
        spec.swa_table_width,
        spec.main_state_table_width,
        spec.inner_state_table_width,
        spec.indexer_table_width,
    ) == (8, 32, 512, 512, 8)


@pytest.mark.parametrize(
    ("native_precondition_steps", "retained_stage_steps"),
    ((1, 1), (1, 128), (128, 1), (128, 128)),
)
def test_128_step_isolation_axes_are_independent_exact_prefixes(
    native_precondition_steps: int,
    retained_stage_steps: int,
) -> None:
    plan = build_a3_stateful_churn_plan(
        spec_steps=128,
        execute_steps=1,
        native_precondition_steps=native_precondition_steps,
        retained_stage_steps=retained_stage_steps,
        seed=20260903,
    )
    full = build_a3_stateful_churn_plan(steps=128, seed=20260903)

    assert plan.spec_steps == 128
    assert plan.execute_steps == 1
    assert plan.native_precondition_steps == native_precondition_steps
    assert plan.retained_stage_steps == retained_stage_steps
    assert plan.payloads == full.payloads[:1]
    assert plan.initializations == full.initializations[:1]
    assert plan.native_precondition_payloads == full.payloads[:native_precondition_steps]
    assert plan.native_precondition_initializations == full.initializations[:native_precondition_steps]
    assert plan.retained_stage_payloads == full.payloads[:retained_stage_steps]
    assert plan.specs_by_bucket == full.specs_by_bucket
    assert plan.trace.steps == full.spec_trace.steps[:1]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"steps": 4, "spec_steps": 128, "execute_steps": 1}, "cannot be combined"),
        ({"steps": 4, "native_precondition_steps": 4}, "cannot be combined"),
        ({"spec_steps": 128}, "provided together"),
        ({"execute_steps": 1}, "provided together"),
        ({"native_precondition_steps": 1}, "provided together"),
        ({"retained_stage_steps": 1}, "provided together"),
        ({"spec_steps": 4, "execute_steps": 32}, "cannot exceed"),
        ({"spec_steps": 32, "execute_steps": 2}, "execute_steps must be one of"),
        (
            {"spec_steps": 32, "execute_steps": 4, "native_precondition_steps": 1},
            "cannot be smaller",
        ),
        (
            {"spec_steps": 32, "execute_steps": 1, "native_precondition_steps": 128},
            "cannot exceed",
        ),
        (
            {"spec_steps": 32, "execute_steps": 1, "retained_stage_steps": 128},
            "cannot exceed",
        ),
        (
            {"spec_steps": 32, "execute_steps": 1, "native_precondition_steps": 2},
            "native_precondition_steps must be one of",
        ),
        (
            {"spec_steps": 32, "execute_steps": 1, "retained_stage_steps": 2},
            "retained_stage_steps must be one of",
        ),
    ),
)
def test_plan_rejects_ambiguous_or_unreviewed_static_execute_combinations(
    kwargs: dict[str, int],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        build_a3_stateful_churn_plan(seed=73, **kwargs)


def test_diagnostic_static_extent_overrides_are_separate_and_rematerialized() -> None:
    baseline_32 = build_a3_stateful_churn_plan(steps=32, seed=20260903)
    baseline_spec = baseline_32.specs_by_bucket[4]
    extents = A3StatefulStaticExtentOverrides(
        **{
            name: getattr(baseline_spec, name)
            for name in (
                "swa_blocks",
                "compressed_blocks",
                "main_state_blocks",
                "inner_state_blocks",
                "indexer_blocks",
                "swa_table_width",
                "compressed_table_width",
                "main_state_table_width",
                "inner_state_table_width",
                "indexer_table_width",
            )
        }
    )

    diagnostic = build_a3_stateful_churn_diagnostic_plan(
        spec_steps=128,
        execute_steps=1,
        seed=20260903,
        static_extent_overrides=extents,
    )

    assert diagnostic.spec_steps == 128
    assert diagnostic.execute_steps == 1
    for spec in diagnostic.specs_by_bucket.values():
        assert spec.key[1:-1] == baseline_spec.key[1:-1]
    assert diagnostic.payloads[0].spec is diagnostic.specs_by_bucket[4]
    assert tuple(payload.spec.batch for payload in diagnostic.warmup_payloads) == diagnostic.buckets


def test_diagnostic_static_extent_override_rejects_invalid_or_undersized_values() -> None:
    with pytest.raises(ValueError, match="positive integers"):
        A3StatefulStaticExtentOverrides(compressed_blocks=0)
    with pytest.raises(ValueError, match="capacity=1"):
        build_a3_stateful_churn_diagnostic_plan(
            spec_steps=128,
            execute_steps=1,
            seed=20260903,
            static_extent_overrides=A3StatefulStaticExtentOverrides(compressed_blocks=1),
        )


def test_result_reports_static_and_executed_lengths_without_claiming_full_spec_trace() -> None:
    result = A3StatefulChurnResult(
        runtime="tensormap_and_ringbuffer",
        device=0,
        trace_seed=20260903,
        spec_steps=128,
        requested_steps=1,
        native_precondition_steps=128,
        retained_stage_steps=1,
        pack_weights_before_native=True,
        buckets=(4, 8, 12, 16),
        reuse_event_count=0,
        pypto_sacrificial_bucket_warmups_reset=True,
        steps=(),
        native_final_page_padding_mismatches={},
        pypto_final_page_padding_mismatches={},
    )

    serialized = result.to_dict()
    assert serialized["spec_steps"] == 128
    assert serialized["execute_steps"] == 1
    assert serialized["requested_steps"] == 1
    assert serialized["native_precondition_steps"] == 128
    assert serialized["retained_stage_steps"] == 1
    assert serialized["pack_weights_before_native"] is True
    assert not result.close


def test_four_step_a3_plan_contains_real_replacement_reuse_and_row_compaction() -> None:
    plan = build_a3_stateful_churn_plan(steps=4, seed=73)

    assert not plan.initializations[0].reuse_events
    assert all(initialization.reuse_events for initialization in plan.initializations[1:])
    assert all(step.retired_request_ids for step in plan.trace.steps[1:])
    assert all(step.admitted_request_ids for step in plan.trace.steps[1:])
    assert all(
        request.sequence_length_before == 0
        for step in plan.trace.steps
        for request in step.active_requests
        if request.request_id in step.admitted_request_ids
    )

    previous_rows = {request.request_id: request.row_index for request in plan.trace.steps[0].active_requests}
    compacted = False
    for step in plan.trace.steps[1:]:
        current_rows = {request.request_id: request.row_index for request in step.active_requests}
        compacted |= any(
            current_rows[request_id] < previous_rows[request_id]
            for request_id in set(previous_rows) & set(current_rows)
        )
        previous_rows = current_rows
    assert compacted

    reused_families = {
        event.family for initialization in plan.initializations[1:] for event in initialization.reuse_events
    }
    assert reused_families == set(CACHE_FAMILY_NAMES)


def test_reuse_events_have_a_matching_clear_before_the_admitted_launch() -> None:
    plan = build_a3_stateful_churn_plan(steps=32, seed=73)

    assert plan.reuse_event_count > 0
    for initialization in plan.initializations:
        for event in initialization.reuse_events:
            assert event.acquired_step == initialization.step_id
            assert event.retired_step <= event.acquired_step
            assert event.physical_block in initialization.physical_blocks[event.family]
            assert event.physical_block == event.abstract_block + 1
    assert all(
        block > 0
        for initialization in plan.initializations
        for blocks in initialization.physical_blocks.values()
        for block in blocks
    )


def test_survivor_growth_reuse_is_initialized_not_only_new_admissions() -> None:
    plan = build_a3_stateful_churn_plan(steps=128, seed=73)
    survivor_growth = [
        event
        for initialization in plan.initializations
        for event in initialization.reuse_events
        if event.acquiring_request_id not in initialization.admitted_request_ids
    ]

    assert survivor_growth
    for event in survivor_growth:
        initialization = plan.initializations[event.acquired_step]
        assert event.physical_block in initialization.physical_blocks[event.family]


def test_initialization_plan_exactly_matches_every_ownership_growth_delta() -> None:
    plan = build_a3_stateful_churn_plan(steps=32, seed=73)
    previous = {}
    for step, initialization in zip(plan.trace.steps, plan.initializations, strict=True):
        current = {request.request_id: request for request in step.active_requests}
        for family in CACHE_FAMILY_NAMES:
            expected = []
            for request_id, request in current.items():
                blocks = request.ownership_for(family).logical_block_table
                previous_blocks = (
                    previous[request_id].ownership_for(family).logical_block_table if request_id in previous else ()
                )
                expected.extend(block + 1 for block in blocks[len(previous_blocks) :])
            assert tuple(expected) == initialization.physical_blocks[family]
        previous = current


def _cpu_state(blocks: int = 256) -> tuple[torch.Tensor, ...]:
    return (
        torch.full((blocks, 32, 1, 2), 11.0, dtype=torch.bfloat16),
        torch.full((blocks, 32, 1, 2), 13.0, dtype=torch.bfloat16),
        torch.full((blocks, 2, 1, 4), 17.0, dtype=torch.float32),
        torch.full((blocks, 2, 1, 4), 19.0, dtype=torch.float32),
        torch.full((blocks, 32, 1, 2), 23, dtype=torch.int8),
        torch.full((blocks, 32, 1, 1), 29.0, dtype=torch.float16),
    )


def test_admission_initialization_preserves_canary_and_uses_scale_identity() -> None:
    plan = build_a3_stateful_churn_plan(steps=4, seed=73)
    initialization = plan.initializations[0]
    maximum = max(block for blocks in initialization.physical_blocks.values() for block in blocks)
    state = _cpu_state(maximum + 2)
    canaries = tuple(tensor[0].clone() for tensor in state)

    initialize_acquired_blocks(state, initialization)

    for tensor, canary in zip(state, canaries, strict=True):
        torch.testing.assert_close(tensor[0], canary)
    state_by_name = dict(zip(STATE_NAMES, state, strict=True))
    family_states = {
        COMPRESSOR_ATTENTION: ("compressed_kv",),
        SWA_CACHE: ("swa_kv",),
        MAIN_COMPRESSOR_STATE: ("compressor_state",),
        INNER_COMPRESSOR_STATE: ("indexer_compressor_state",),
        INDEXER_CACHE: ("indexer_k", "indexer_scale"),
    }
    for family, blocks in initialization.physical_blocks.items():
        for name in family_states[family]:
            for block in blocks:
                if name == "indexer_scale":
                    assert torch.all(state_by_name[name][block] == 1)
                else:
                    assert torch.count_nonzero(state_by_name[name][block]) == 0


def test_expected_rows_include_every_clear_block_and_exclude_canary_block() -> None:
    plan = build_a3_stateful_churn_plan(steps=4, seed=73)
    payload = plan.payloads[0]
    initialization = plan.initializations[0]

    legal = expected_mutated_rows(payload, initialization)

    rows_per_block = {
        COMPRESSOR_ATTENTION: 32,
        SWA_CACHE: 32,
        MAIN_COMPRESSOR_STATE: 2,
        INNER_COMPRESSOR_STATE: 2,
        INDEXER_CACHE: 32,
    }
    state_names = {
        COMPRESSOR_ATTENTION: ("compressed_kv",),
        SWA_CACHE: ("swa_kv",),
        MAIN_COMPRESSOR_STATE: ("compressor_state",),
        INNER_COMPRESSOR_STATE: ("indexer_compressor_state",),
        INDEXER_CACHE: ("indexer_k", "indexer_scale"),
    }
    for family, blocks in initialization.physical_blocks.items():
        width = rows_per_block[family]
        for name in state_names[family]:
            legal_set = set(legal[name])
            for block in blocks:
                assert set(range(block * width, (block + 1) * width)).issubset(legal_set)
            assert all(row >= width for row in legal_set)


def _snapshot() -> dict[str, torch.Tensor]:
    return {
        "compressed_kv": torch.zeros((2, 2, 1, 3), dtype=torch.bfloat16),
        "swa_kv": torch.zeros((2, 2, 1, 3), dtype=torch.bfloat16),
        "compressor_state": torch.zeros((2, 2, 1, 4), dtype=torch.float32),
        "indexer_compressor_state": torch.zeros((2, 2, 1, 4), dtype=torch.float32),
        "indexer_k": torch.zeros((2, 2, 1, 3), dtype=torch.int8),
        "indexer_scale": torch.ones((2, 2, 1, 1), dtype=torch.float16),
    }


def test_row_delta_round_trip_reconstructs_all_six_state_families() -> None:
    previous = _snapshot()
    current = {name: tensor.clone() for name, tensor in previous.items()}
    for index, name in enumerate(STATE_NAMES):
        current[name].reshape(-1, current[name].shape[-1])[index % 4, 0] += 1

    deltas = compute_state_deltas(previous, current)
    reconstructed = {name: tensor.clone() for name, tensor in previous.items()}
    apply_state_deltas(reconstructed, deltas)

    assert tuple(deltas) == STATE_NAMES
    for name in STATE_NAMES:
        assert isinstance(deltas[name], StateRowDelta)
        assert len(deltas[name].row_indices) == 1
        torch.testing.assert_close(reconstructed[name], current[name])


def test_delta_rejects_wrong_snapshot_owner() -> None:
    snapshot = _snapshot()
    current = {name: tensor.clone() for name, tensor in snapshot.items()}
    deltas = dict(compute_state_deltas(snapshot, current))
    wrong = deltas["compressed_kv"]
    deltas["compressed_kv"] = StateRowDelta(
        shape=(99,),
        dtype=wrong.dtype,
        row_indices=wrong.row_indices,
        row_values=wrong.row_values,
    )

    with pytest.raises(ValueError, match="does not belong"):
        apply_state_deltas(snapshot, deltas)


def test_step_comparator_gates_local_quantized_density_but_retains_cumulative_report() -> None:
    initial = _snapshot()
    initial["indexer_k"] = torch.zeros((1024, 1, 1, 128), dtype=torch.int8)
    initial["indexer_scale"] = torch.zeros((1024, 1, 1, 1), dtype=torch.float16)
    native = {name: tensor.clone() for name, tensor in initial.items()}
    native["indexer_k"][:, 0, 0, 0] = 3
    native["indexer_scale"].fill_(0.019317626953125)
    pypto = {name: tensor.clone() for name, tensor in native.items()}
    pypto["indexer_k"][1023, 0, 0, 54] += 1
    pypto["indexer_k"][1023, 0, 0, 55] += 1

    native_previous = {name: tensor.clone() for name, tensor in native.items()}
    pypto_previous = {name: tensor.clone() for name, tensor in pypto.items()}
    for previous in (native_previous, pypto_previous):
        previous["indexer_k"][1023].zero_()
        previous["indexer_scale"][1023].zero_()
    native_deltas = compute_state_deltas(native_previous, native)
    pypto_deltas = compute_state_deltas(pypto_previous, pypto)
    legal_rows = {name: ((1023,) if name in {"indexer_k", "indexer_scale"} else ()) for name in STATE_NAMES}

    _, step_local, cumulative = _compare_step_states(
        native,
        pypto,
        initial,
        native_deltas,
        pypto_deltas,
        legal_rows,
        atol=0.1,
        rtol=0.1,
    )

    assert step_local.compared_rows == 1
    assert step_local.raw_mismatch_fraction == pytest.approx(2 / 128)
    assert step_local.allowed_raw_mismatch_elements == 1
    assert not step_local.acceptable
    assert cumulative.compared_rows == 1024
    assert cumulative.raw_mismatch_fraction == pytest.approx(2 / (1024 * 128))
    assert cumulative.acceptable


def test_cli_requires_runtime_and_limits_trace_length() -> None:
    parser = build_parser()
    args = parser.parse_args(["--runtime", "host_build_graph", "--steps", "128"])

    assert args.runtime == "host_build_graph"
    assert args.steps == 128
    assert args.device == 0
    with pytest.raises(SystemExit):
        parser.parse_args(["--runtime", "tensormap_and_ringbuffer", "--steps", "5"])

    isolated = parser.parse_args(
        [
            "--runtime",
            "tensormap_and_ringbuffer",
            "--spec-steps",
            "128",
            "--execute-steps",
            "1",
        ]
    )
    assert isolated.steps is None
    assert isolated.spec_steps == 128
    assert isolated.execute_steps == 1
    assert isolated.native_precondition_steps is None
    assert isolated.retained_stage_steps is None
    assert isolated.pack_weights_before_native is False

    matrix = parser.parse_args(
        [
            "--runtime",
            "tensormap_and_ringbuffer",
            "--spec-steps",
            "128",
            "--execute-steps",
            "1",
            "--native-precondition-steps",
            "128",
            "--retained-stage-steps",
            "1",
            "--pack-weights-before-native",
        ]
    )
    assert matrix.native_precondition_steps == 128
    assert matrix.retained_stage_steps == 1
    assert matrix.pack_weights_before_native is True


def test_npu_and_distributed_imports_remain_inside_runner() -> None:
    source = inspect.getsource(run_a3_stateful_churn_compare)

    assert "import torch_npu" in source
    assert "_initialize_tp1" in source
    assert source.index("if pack_weights_before_native:") < source.index("for payload, initialization in zip(")
    assert "prepared_weights=prepared_weights" in source
    assert "plan.native_precondition_payloads" in source
    assert "plan.retained_stage_payloads" in source
    assert {family: 1 for family in CACHE_FAMILY_NAMES} == ALL_FAMILY_BLOCK_OFFSETS


def test_runner_rejects_invalid_isolation_axes_before_importing_npu_runtime() -> None:
    with pytest.raises(ValueError, match="cannot be smaller"):
        run_a3_stateful_churn_compare(
            runtime="tensormap_and_ringbuffer",
            spec_steps=128,
            execute_steps=4,
            native_precondition_steps=1,
            retained_stage_steps=1,
        )
    with pytest.raises(TypeError, match="must be a bool"):
        run_a3_stateful_churn_compare(
            runtime="tensormap_and_ringbuffer",
            steps=4,
            pack_weights_before_native=1,  # type: ignore[arg-type]
        )


def test_plan_rejects_unreviewed_trace_lengths() -> None:
    with pytest.raises(ValueError, match="spec_steps must be one of"):
        build_a3_stateful_churn_plan(steps=5, seed=73)
