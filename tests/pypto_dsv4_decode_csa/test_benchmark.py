# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Host-only tests for benchmark scheduling, statistics, and serialization."""

from __future__ import annotations

import csv
import io
import json
import math

import pytest

from tests.pypto_dsv4_decode_csa.benchmark import (
    BENCHMARK_SCHEMA_NAME,
    BENCHMARK_SCHEMA_VERSION,
    TIMING_METRICS,
    BenchmarkAccumulator,
    BenchmarkBackend,
    BenchmarkError,
    BenchmarkKey,
    BenchmarkMode,
    DuplicateSampleError,
    MeasurementStatus,
    MissingSampleError,
    SamplingOrder,
    SamplingPhase,
    SamplingPolicy,
    TimingMetric,
    TimingStatistics,
    WorkloadKey,
    build_benchmark_schedule,
    required_metrics,
)


def _workload(
    *,
    bucket: int = 4,
    actual: int = 4,
    positions: tuple[int, ...] = (128,),
    layers: int = 1,
) -> WorkloadKey:
    return WorkloadKey(
        batch_bucket=bucket,
        decode_sequence_length=8,
        context_positions=positions,
        num_layers=layers,
        num_reqs_actual=actual,
        topology="single_op" if layers == 1 else "PPPP",
        trace_seed=73,
    )


def _key(
    backend: BenchmarkBackend,
    *,
    mode: BenchmarkMode = BenchmarkMode.EAGER,
    suffix: str = "B4-S8",
) -> BenchmarkKey:
    return BenchmarkKey(
        backend=backend,
        mode=mode,
        workload=_workload(),
        specialization_key=suffix,
    )


def _durations(invocation) -> dict[TimingMetric, int]:
    base = 100 if invocation.phase is SamplingPhase.SAMPLE else 1_000_000
    value = base + invocation.case_iteration
    durations = {
        TimingMetric.HOST_ENQUEUE: value,
        TimingMetric.METADATA_UPDATE: value + 10,
        TimingMetric.DEVICE_SPAN: value + 20,
    }
    if invocation.key.mode is BenchmarkMode.ACLGRAPH:
        durations[TimingMetric.GRAPH_REPLAY] = value + 30
    return durations


def _complete(schedule, *, metadata: dict[str, str] | None = None):
    accumulator = BenchmarkAccumulator(schedule)
    for invocation in schedule.all_invocations:
        accumulator.record(invocation, _durations(invocation))
    return accumulator.finalize(metadata=metadata)


@pytest.mark.parametrize(
    "kwargs",
    (
        {"batch_bucket": 3},
        {"decode_sequence_length": 0},
        {"context_positions": ()},
        {"context_positions": (-1,)},
        {"num_layers": 3},
        {"num_layers": True},
        {"num_reqs_actual": 0},
        {"num_reqs_actual": 5},
        {"context_positions": (1, 2)},
        {"topology": ""},
        {"topology": 1},
        {"trace_seed": True},
    ),
)
def test_workload_key_rejects_invalid_or_ambiguous_dimensions(kwargs: dict[str, object]) -> None:
    values: dict[str, object] = {
        "batch_bucket": 4,
        "decode_sequence_length": 8,
        "context_positions": (128,),
        "num_layers": 1,
        "num_reqs_actual": 4,
        "topology": "single_op",
        "trace_seed": 1,
    }
    values.update(kwargs)
    with pytest.raises(BenchmarkError):
        WorkloadKey(**values)  # type: ignore[arg-type]


def test_backend_mode_workload_key_has_a_stable_serializable_identity() -> None:
    key = _key(BenchmarkBackend.PYPTO_TRB, mode=BenchmarkMode.ACLGRAPH)

    assert key.backend is BenchmarkBackend.PYPTO_TRB
    assert key.mode is BenchmarkMode.ACLGRAPH
    assert "pypto_trb/aclgraph/B4-S8/B4S8" in key.stable_id
    assert key.to_dict()["workload"] == key.workload.to_dict()
    with pytest.raises(BenchmarkError, match="specialization_key"):
        BenchmarkKey(
            backend=BenchmarkBackend.PYPTO_TRB,
            mode=BenchmarkMode.EAGER,
            workload=_workload(),
            specialization_key="",
        )


@pytest.mark.parametrize(
    "policy",
    (
        {"warmup_iterations": 2, "sample_iterations": 100},
        {"warmup_iterations": 51, "sample_iterations": 100},
        {"warmup_iterations": 20, "sample_iterations": 9},
        {"warmup_iterations": 20, "sample_iterations": 1_001},
    ),
)
def test_sampling_policy_enforces_the_design_ranges(policy: dict[str, int]) -> None:
    with pytest.raises(BenchmarkError):
        SamplingPolicy(**policy)


def test_abba_order_is_balanced_and_warmup_is_separate_from_sampling() -> None:
    native = _key(BenchmarkBackend.NATIVE_PRODUCTION)
    pypto = _key(BenchmarkBackend.PYPTO_TRB)
    policy = SamplingPolicy(warmup_iterations=20, sample_iterations=100)
    schedule = build_benchmark_schedule(
        (native, pypto),
        policy=policy,
        order=SamplingOrder.ABBA,
    )

    assert [item.key for item in schedule.warmup_invocations[:8]] == [
        native,
        pypto,
        pypto,
        native,
        native,
        pypto,
        pypto,
        native,
    ]
    assert all(item.phase is SamplingPhase.WARMUP for item in schedule.warmup_invocations)
    assert all(item.phase is SamplingPhase.SAMPLE for item in schedule.sample_invocations)
    assert len(schedule.warmup_invocations) == 40
    assert len(schedule.sample_invocations) == 200
    assert {key: sum(item.key == key for item in schedule.sample_invocations) for key in schedule.keys} == {
        native: 100,
        pypto: 100,
    }


def test_rotation_changes_leading_case_and_balances_three_backends() -> None:
    keys = (
        _key(BenchmarkBackend.NATIVE_PRODUCTION),
        _key(BenchmarkBackend.PYPTO_TRB),
        _key(BenchmarkBackend.PYPTO_HBG),
    )
    schedule = build_benchmark_schedule(keys, order=SamplingOrder.ROTATE)

    assert [item.key for item in schedule.warmup_invocations[:9]] == [
        keys[0],
        keys[1],
        keys[2],
        keys[1],
        keys[2],
        keys[0],
        keys[2],
        keys[0],
        keys[1],
    ]
    assert all(sum(item.key == key for item in schedule.sample_invocations) == 100 for key in keys)
    with pytest.raises(BenchmarkError, match="exactly two"):
        build_benchmark_schedule(keys, order=SamplingOrder.ABBA)


def test_statistics_report_required_percentiles_and_population_std() -> None:
    stats = TimingStatistics.from_samples((1, 2, 3, 4, 5))

    assert stats.count == 5
    assert stats.minimum_ns == 1
    assert stats.mean_ns == 3
    assert stats.std_ns == pytest.approx(math.sqrt(2))
    assert stats.p50_ns == 3
    assert stats.p90_ns == pytest.approx(4.6)
    assert stats.p99_ns == pytest.approx(4.96)
    with pytest.raises(MissingSampleError):
        TimingStatistics.from_samples(())
    with pytest.raises(BenchmarkError):
        TimingStatistics.from_samples((1, 0, 2))


def test_accumulator_fails_fast_for_missing_extra_invalid_and_duplicate_samples() -> None:
    key = _key(BenchmarkBackend.NATIVE_PRODUCTION)
    schedule = build_benchmark_schedule((key,))
    invocation = schedule.warmup_invocations[0]

    with pytest.raises(MissingSampleError, match="scheduled invocations"):
        BenchmarkAccumulator(schedule).finalize()

    missing_metric = BenchmarkAccumulator(schedule)
    with pytest.raises(MissingSampleError, match="missing"):
        missing_metric.record(
            invocation,
            {
                TimingMetric.HOST_ENQUEUE: 1,
                TimingMetric.METADATA_UPDATE: 2,
            },
        )

    extra_metric = BenchmarkAccumulator(schedule)
    with pytest.raises(MissingSampleError, match="extra"):
        extra_metric.record(
            invocation,
            {
                **_durations(invocation),
                TimingMetric.GRAPH_REPLAY: 4,
            },
        )

    invalid_duration = BenchmarkAccumulator(schedule)
    bad = _durations(invocation)
    bad[TimingMetric.HOST_ENQUEUE] = 0
    with pytest.raises(BenchmarkError, match="positive integer"):
        invalid_duration.record(invocation, bad)

    duplicate = BenchmarkAccumulator(schedule)
    duplicate.record(invocation, _durations(invocation))
    with pytest.raises(DuplicateSampleError):
        duplicate.record(invocation, _durations(invocation))


def test_accumulator_rejects_reordered_and_sample_before_warmup_observations() -> None:
    keys = (
        _key(BenchmarkBackend.NATIVE_PRODUCTION),
        _key(BenchmarkBackend.PYPTO_TRB),
    )
    schedule = build_benchmark_schedule(keys, order=SamplingOrder.ABBA)

    reordered = BenchmarkAccumulator(schedule)
    with pytest.raises(BenchmarkError, match="out of schedule order"):
        reordered.record(schedule.warmup_invocations[1], _durations(schedule.warmup_invocations[1]))

    sample_first = BenchmarkAccumulator(schedule)
    with pytest.raises(BenchmarkError, match="out of schedule order"):
        sample_first.record(schedule.sample_invocations[0], _durations(schedule.sample_invocations[0]))

    reverse = BenchmarkAccumulator(schedule)
    with pytest.raises(BenchmarkError, match="out of schedule order"):
        first = schedule.all_invocations[-1]
        reverse.record(first, _durations(first))


def test_eager_result_keeps_warmup_out_of_stats_and_marks_graph_replay_na() -> None:
    key = _key(BenchmarkBackend.PYPTO_TRB)
    schedule = build_benchmark_schedule((key,))
    report = _complete(schedule)
    case = report.cases[0]

    host = case.measurement_for(TimingMetric.HOST_ENQUEUE)
    graph = case.measurement_for(TimingMetric.GRAPH_REPLAY)
    assert len(host.warmup_samples_ns) == 20
    assert len(host.samples_ns) == 100
    assert min(host.warmup_samples_ns) >= 1_000_000
    assert host.statistics is not None
    assert host.statistics.minimum_ns == 100
    assert host.statistics.p50_ns == pytest.approx(149.5)
    assert graph.status is MeasurementStatus.NOT_APPLICABLE
    assert graph.statistics is None
    assert graph.reason is not None

    combined = case.host_call_including_external_update
    assert combined.minimum_ns == 210
    assert combined.p50_ns == pytest.approx(309)


def test_aclgraph_requires_and_reports_all_four_timing_categories() -> None:
    key = _key(
        BenchmarkBackend.PYPTO_HBG,
        mode=BenchmarkMode.ACLGRAPH,
    )
    assert required_metrics(key.mode) == frozenset(TIMING_METRICS)
    schedule = build_benchmark_schedule((key,))
    first = schedule.warmup_invocations[0]

    incomplete = BenchmarkAccumulator(schedule)
    durations = _durations(first)
    del durations[TimingMetric.GRAPH_REPLAY]
    with pytest.raises(MissingSampleError, match="graph_replay"):
        incomplete.record(first, durations)

    report = _complete(schedule)
    assert all(measurement.status is MeasurementStatus.MEASURED for measurement in report.cases[0].measurements)
    replay = report.cases[0].measurement_for(TimingMetric.GRAPH_REPLAY)
    assert replay.statistics is not None
    assert replay.statistics.count == 100


def test_json_and_csv_are_versioned_and_standard_library_serializable() -> None:
    keys = (
        _key(BenchmarkBackend.NATIVE_PRODUCTION),
        _key(BenchmarkBackend.PYPTO_TRB),
    )
    report = _complete(
        build_benchmark_schedule(keys, order=SamplingOrder.ABBA),
        metadata={
            "device": "A3:0",
            "vllm_ascend_commit": "deadbeef",
        },
    )

    payload = json.loads(report.to_json())
    assert payload["schema_name"] == BENCHMARK_SCHEMA_NAME
    assert payload["schema_version"] == BENCHMARK_SCHEMA_VERSION
    assert payload["sampling"] == {
        "order": "abba",
        "sample_iterations": 100,
        "warmup_iterations": 20,
    }
    assert payload["metadata"]["device"] == "A3:0"
    assert len(payload["cases"]) == 2
    assert len(payload["cases"][0]["measurements"]) == 4
    assert len(payload["cases"][0]["measurements"][0]["warmup_samples_ns"]) == 20
    assert len(payload["cases"][0]["measurements"][0]["samples_ns"]) == 100
    assert "host_enqueue_callback" in payload["cases"][0]["derived"]
    assert "host_call_including_external_update" in payload["cases"][0]["derived"]
    assert "host_enqueue_excluding_metadata" not in payload["cases"][0]["derived"]

    rows = list(csv.DictReader(io.StringIO(report.to_csv())))
    assert len(rows) == 10
    assert {row["schema_name"] for row in rows} == {BENCHMARK_SCHEMA_NAME}
    assert {row["schema_version"] for row in rows} == {BENCHMARK_SCHEMA_VERSION}
    assert {row["backend"] for row in rows} == {
        BenchmarkBackend.NATIVE_PRODUCTION.value,
        BenchmarkBackend.PYPTO_TRB.value,
    }
    assert {row["metric"] for row in rows if row["backend"] == BenchmarkBackend.PYPTO_TRB.value} == {
        *(metric.value for metric in TIMING_METRICS),
        "host_call_including_external_update",
    }
    graph_rows = [row for row in rows if row["metric"] == TimingMetric.GRAPH_REPLAY.value]
    assert all(row["status"] == MeasurementStatus.NOT_APPLICABLE.value for row in graph_rows)


def test_schedule_and_report_reject_duplicate_case_keys() -> None:
    key = _key(BenchmarkBackend.PYPTO_TRB)
    with pytest.raises(BenchmarkError, match="unique"):
        build_benchmark_schedule((key, key))

    different_workload = BenchmarkKey(
        backend=BenchmarkBackend.PYPTO_HBG,
        mode=BenchmarkMode.EAGER,
        workload=_workload(bucket=8, actual=8),
        specialization_key="B8-S8",
    )
    with pytest.raises(BenchmarkError, match="same exact workload"):
        build_benchmark_schedule((key, different_workload), order=SamplingOrder.ABBA)
