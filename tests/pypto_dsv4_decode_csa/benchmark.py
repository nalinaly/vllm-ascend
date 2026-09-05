# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Pure-Host benchmark scheduling, sampling, and result serialization.

The module owns measurement semantics but never reads a clock or device. A
future eager/ACLGraph runner executes the generated schedule and records all
durations in nanoseconds, keeping warmup and steady-state samples disjoint.
"""

from __future__ import annotations

import csv
import io
import json
import math
import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum

BENCHMARK_SCHEMA_NAME = "dsv4_csa.performance_benchmark"
BENCHMARK_SCHEMA_VERSION = "1.0.0"


class BenchmarkError(ValueError):
    """Base class for an invalid benchmark definition or result."""


class MissingSampleError(BenchmarkError):
    """Raised when finalization finds an unrecorded scheduled sample."""


class DuplicateSampleError(BenchmarkError):
    """Raised when one scheduled invocation is recorded more than once."""


class BenchmarkBackend(str, Enum):
    NATIVE_PRODUCTION = "native_production"
    NATIVE_SERIAL = "native_serial"
    PYPTO_TRB = "pypto_trb"
    PYPTO_HBG = "pypto_hbg"
    MIXED = "mixed"


class BenchmarkMode(str, Enum):
    EAGER = "eager"
    ACLGRAPH = "aclgraph"


class TimingMetric(str, Enum):
    """Independent timing views; they are not generally additive.

    ``HOST_ENQUEUE`` covers the complete runner callback through Host return,
    including implementation-internal metadata/binding/address patch work.
    ``METADATA_UPDATE`` is only an optional external update performed before
    that callback.
    """

    HOST_ENQUEUE = "host_enqueue"
    METADATA_UPDATE = "metadata_update"
    DEVICE_SPAN = "device_span"
    GRAPH_REPLAY = "graph_replay"


TIMING_METRICS = tuple(TimingMetric)


class MeasurementStatus(str, Enum):
    MEASURED = "measured"
    NOT_APPLICABLE = "not_applicable"


class SamplingPhase(str, Enum):
    WARMUP = "warmup"
    SAMPLE = "sample"


class SamplingOrder(str, Enum):
    ABBA = "abba"
    ROTATE = "rotate"


@dataclass(frozen=True, slots=True)
class WorkloadKey:
    """Hashable workload identity shared by all compared backends."""

    batch_bucket: int
    decode_sequence_length: int
    context_positions: tuple[int, ...]
    num_layers: int
    num_reqs_actual: int
    topology: str = "single_op"
    trace_seed: int | None = None

    def __post_init__(self) -> None:
        try:
            positions = tuple(self.context_positions)
        except TypeError as error:
            raise BenchmarkError("context_positions must be an iterable of integers") from error
        object.__setattr__(self, "context_positions", positions)
        if (
            isinstance(self.batch_bucket, bool)
            or not isinstance(self.batch_bucket, int)
            or self.batch_bucket not in {4, 8, 12, 16}
        ):
            raise BenchmarkError("batch_bucket must be one of B4/B8/B12/B16")
        if (
            isinstance(self.decode_sequence_length, bool)
            or not isinstance(self.decode_sequence_length, int)
            or self.decode_sequence_length <= 0
        ):
            raise BenchmarkError("decode_sequence_length must be a positive integer")
        if not self.context_positions or any(
            isinstance(position, bool) or not isinstance(position, int) or position < 0
            for position in self.context_positions
        ):
            raise BenchmarkError("context_positions must contain non-negative integers")
        if (
            isinstance(self.num_layers, bool)
            or not isinstance(self.num_layers, int)
            or self.num_layers not in {1, 2, 4}
        ):
            raise BenchmarkError("num_layers must be one of 1, 2, or 4")
        if (
            isinstance(self.num_reqs_actual, bool)
            or not isinstance(self.num_reqs_actual, int)
            or not 1 <= self.num_reqs_actual <= self.batch_bucket
        ):
            raise BenchmarkError("num_reqs_actual must be within the selected batch bucket")
        if len(self.context_positions) not in {1, self.num_reqs_actual}:
            raise BenchmarkError(
                "context_positions must contain one shared position or one position per actual request"
            )
        if not isinstance(self.topology, str) or not self.topology:
            raise BenchmarkError("topology must not be empty")
        if self.trace_seed is not None and (isinstance(self.trace_seed, bool) or not isinstance(self.trace_seed, int)):
            raise BenchmarkError("trace_seed must be an integer or None")

    def to_dict(self) -> dict[str, object]:
        return {
            "batch_bucket": self.batch_bucket,
            "decode_sequence_length": self.decode_sequence_length,
            "context_positions": list(self.context_positions),
            "num_layers": self.num_layers,
            "num_reqs_actual": self.num_reqs_actual,
            "topology": self.topology,
            "trace_seed": self.trace_seed,
        }


@dataclass(frozen=True, slots=True)
class BenchmarkKey:
    """Backend/mode/workload key for one apples-to-apples result row."""

    backend: BenchmarkBackend
    mode: BenchmarkMode
    workload: WorkloadKey
    specialization_key: str

    def __post_init__(self) -> None:
        try:
            backend = BenchmarkBackend(self.backend)
            mode = BenchmarkMode(self.mode)
        except ValueError as error:
            raise BenchmarkError("unsupported backend or execution mode") from error
        object.__setattr__(self, "backend", backend)
        object.__setattr__(self, "mode", mode)
        if not isinstance(self.specialization_key, str) or not self.specialization_key:
            raise BenchmarkError("specialization_key must not be empty")

    @property
    def stable_id(self) -> str:
        positions = "-".join(str(position) for position in self.workload.context_positions)
        workload = self.workload
        return (
            f"{self.backend.value}/{self.mode.value}/{self.specialization_key}/"
            f"B{workload.batch_bucket}S{workload.decode_sequence_length}/"
            f"C{positions}/L{workload.num_layers}/A{workload.num_reqs_actual}/"
            f"{workload.topology}/seed={workload.trace_seed}"
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "stable_id": self.stable_id,
            "backend": self.backend.value,
            "mode": self.mode.value,
            "specialization_key": self.specialization_key,
            "workload": self.workload.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class SamplingPolicy:
    """Steady-state policy from the design: 20-50 warmups, 100-1000 samples."""

    warmup_iterations: int = 20
    sample_iterations: int = 100

    def __post_init__(self) -> None:
        if (
            isinstance(self.warmup_iterations, bool)
            or not isinstance(self.warmup_iterations, int)
            or not 20 <= self.warmup_iterations <= 50
        ):
            raise BenchmarkError("warmup_iterations must be within [20, 50]")
        if (
            isinstance(self.sample_iterations, bool)
            or not isinstance(self.sample_iterations, int)
            or not 100 <= self.sample_iterations <= 1_000
        ):
            raise BenchmarkError("sample_iterations must be within [100, 1000]")

    def to_dict(self) -> dict[str, int]:
        return {
            "warmup_iterations": self.warmup_iterations,
            "sample_iterations": self.sample_iterations,
        }


@dataclass(frozen=True, slots=True)
class ScheduledInvocation:
    phase: SamplingPhase
    sequence_ordinal: int
    key: BenchmarkKey
    case_iteration: int

    def __post_init__(self) -> None:
        if self.sequence_ordinal < 0 or self.case_iteration < 0:
            raise BenchmarkError("scheduled invocation indices must be non-negative")


@dataclass(frozen=True, slots=True)
class BenchmarkSchedule:
    """Separate warmup and measurement schedules with balanced backend order."""

    keys: tuple[BenchmarkKey, ...]
    policy: SamplingPolicy
    order: SamplingOrder
    warmup_invocations: tuple[ScheduledInvocation, ...]
    sample_invocations: tuple[ScheduledInvocation, ...]

    def __post_init__(self) -> None:
        if not self.keys:
            raise BenchmarkError("benchmark schedule requires at least one case")
        if len(set(self.keys)) != len(self.keys):
            raise BenchmarkError("benchmark schedule keys must be unique")
        expected = (
            (SamplingPhase.WARMUP, self.warmup_invocations, self.policy.warmup_iterations),
            (SamplingPhase.SAMPLE, self.sample_invocations, self.policy.sample_iterations),
        )
        for phase, invocations, target in expected:
            if any(invocation.phase is not phase for invocation in invocations):
                raise BenchmarkError(f"{phase.value} schedule contains an invocation from another phase")
            if tuple(invocation.sequence_ordinal for invocation in invocations) != tuple(range(len(invocations))):
                raise BenchmarkError(f"{phase.value} schedule ordinals must be contiguous")
            counts = Counter(invocation.key for invocation in invocations)
            if counts != Counter({key: target for key in self.keys}):
                raise BenchmarkError(f"{phase.value} schedule is not balanced across cases")
            for key in self.keys:
                case_indices = sorted(invocation.case_iteration for invocation in invocations if invocation.key == key)
                if case_indices != list(range(target)):
                    raise BenchmarkError(f"{phase.value}/{key.stable_id}: case indices are not contiguous")

    @property
    def all_invocations(self) -> tuple[ScheduledInvocation, ...]:
        return self.warmup_invocations + self.sample_invocations


def build_benchmark_schedule(
    keys: Sequence[BenchmarkKey],
    *,
    policy: SamplingPolicy | None = None,
    order: SamplingOrder = SamplingOrder.ROTATE,
) -> BenchmarkSchedule:
    """Generate ABBA for two cases or balanced rotations for arbitrary cases."""
    normalized_keys = tuple(keys)
    normalized_policy = policy or SamplingPolicy()
    try:
        normalized_order = SamplingOrder(order)
    except ValueError as error:
        raise BenchmarkError(f"unsupported sampling order {order!r}") from error
    if not normalized_keys:
        raise BenchmarkError("at least one benchmark key is required")
    if len(set(normalized_keys)) != len(normalized_keys):
        raise BenchmarkError("benchmark keys must be unique")
    if len({key.workload for key in normalized_keys}) != 1:
        raise BenchmarkError("one alternating schedule must compare the same exact workload")
    if normalized_order is SamplingOrder.ABBA and len(normalized_keys) != 2:
        raise BenchmarkError("ABBA order requires exactly two benchmark keys")

    return BenchmarkSchedule(
        keys=normalized_keys,
        policy=normalized_policy,
        order=normalized_order,
        warmup_invocations=_build_phase_schedule(
            normalized_keys,
            phase=SamplingPhase.WARMUP,
            target_per_case=normalized_policy.warmup_iterations,
            order=normalized_order,
        ),
        sample_invocations=_build_phase_schedule(
            normalized_keys,
            phase=SamplingPhase.SAMPLE,
            target_per_case=normalized_policy.sample_iterations,
            order=normalized_order,
        ),
    )


def _build_phase_schedule(
    keys: tuple[BenchmarkKey, ...],
    *,
    phase: SamplingPhase,
    target_per_case: int,
    order: SamplingOrder,
) -> tuple[ScheduledInvocation, ...]:
    counts = {key: 0 for key in keys}
    result: list[ScheduledInvocation] = []
    round_index = 0
    while any(count < target_per_case for count in counts.values()):
        if order is SamplingOrder.ABBA:
            candidates = (keys[0], keys[1], keys[1], keys[0])
        else:
            offset = round_index % len(keys)
            candidates = keys[offset:] + keys[:offset]
        for key in candidates:
            if counts[key] >= target_per_case:
                continue
            result.append(
                ScheduledInvocation(
                    phase=phase,
                    sequence_ordinal=len(result),
                    key=key,
                    case_iteration=counts[key],
                )
            )
            counts[key] += 1
            if all(count == target_per_case for count in counts.values()):
                break
        round_index += 1
    return tuple(result)


@dataclass(frozen=True, slots=True)
class TimingStatistics:
    """Population statistics over steady-state samples, in nanoseconds."""

    count: int
    minimum_ns: float
    mean_ns: float
    std_ns: float
    p50_ns: float
    p90_ns: float
    p99_ns: float

    @classmethod
    def from_samples(cls, samples_ns: Sequence[int]) -> TimingStatistics:
        samples = tuple(_validate_duration(value) for value in samples_ns)
        if not samples:
            raise MissingSampleError("cannot summarize an empty sample series")
        sorted_samples = tuple(sorted(samples))
        return cls(
            count=len(samples),
            minimum_ns=float(sorted_samples[0]),
            mean_ns=float(statistics.fmean(samples)),
            std_ns=float(statistics.pstdev(samples)),
            p50_ns=_percentile(sorted_samples, 0.50),
            p90_ns=_percentile(sorted_samples, 0.90),
            p99_ns=_percentile(sorted_samples, 0.99),
        )

    def to_dict(self) -> dict[str, int | float]:
        return {
            "count": self.count,
            "min_ns": self.minimum_ns,
            "mean_ns": self.mean_ns,
            "std_ns": self.std_ns,
            "p50_ns": self.p50_ns,
            "p90_ns": self.p90_ns,
            "p99_ns": self.p99_ns,
        }


def _percentile(sorted_samples: tuple[int, ...], quantile: float) -> float:
    if not sorted_samples:
        raise MissingSampleError("cannot calculate a percentile without samples")
    rank = (len(sorted_samples) - 1) * quantile
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return float(sorted_samples[lower])
    fraction = rank - lower
    return float(sorted_samples[lower] + (sorted_samples[upper] - sorted_samples[lower]) * fraction)


def _validate_duration(duration_ns: object) -> int:
    if isinstance(duration_ns, bool) or not isinstance(duration_ns, int) or duration_ns <= 0:
        raise BenchmarkError(f"duration must be a positive integer number of nanoseconds, got {duration_ns!r}")
    return duration_ns


@dataclass(frozen=True, slots=True)
class MetricMeasurement:
    """Raw warmup/samples plus steady-state summary for one timing category."""

    metric: TimingMetric
    status: MeasurementStatus
    warmup_samples_ns: tuple[int, ...]
    samples_ns: tuple[int, ...]
    statistics: TimingStatistics | None
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.status is MeasurementStatus.MEASURED:
            if not self.warmup_samples_ns or not self.samples_ns:
                raise MissingSampleError(f"{self.metric.value}: measured series is incomplete")
            if self.reason is not None:
                raise BenchmarkError(f"{self.metric.value}: measured series cannot carry an N/A reason")
            if self.statistics is None or self.statistics.count != len(self.samples_ns):
                raise BenchmarkError(f"{self.metric.value}: statistics do not match raw samples")
            tuple(_validate_duration(value) for value in self.warmup_samples_ns)
            tuple(_validate_duration(value) for value in self.samples_ns)
            return
        if self.warmup_samples_ns or self.samples_ns or self.statistics is not None:
            raise BenchmarkError(f"{self.metric.value}: N/A series cannot contain timing samples")
        if not self.reason:
            raise BenchmarkError(f"{self.metric.value}: N/A series requires an explicit reason")

    @classmethod
    def measured(
        cls,
        metric: TimingMetric,
        *,
        warmup_samples_ns: Sequence[int],
        samples_ns: Sequence[int],
    ) -> MetricMeasurement:
        warmup = tuple(_validate_duration(value) for value in warmup_samples_ns)
        samples = tuple(_validate_duration(value) for value in samples_ns)
        return cls(
            metric=metric,
            status=MeasurementStatus.MEASURED,
            warmup_samples_ns=warmup,
            samples_ns=samples,
            statistics=TimingStatistics.from_samples(samples),
        )

    @classmethod
    def not_applicable(cls, metric: TimingMetric, *, reason: str) -> MetricMeasurement:
        return cls(
            metric=metric,
            status=MeasurementStatus.NOT_APPLICABLE,
            warmup_samples_ns=(),
            samples_ns=(),
            statistics=None,
            reason=reason,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "metric": self.metric.value,
            "status": self.status.value,
            "reason": self.reason,
            "warmup_samples_ns": list(self.warmup_samples_ns),
            "samples_ns": list(self.samples_ns),
            "statistics": self.statistics.to_dict() if self.statistics is not None else None,
        }


@dataclass(frozen=True, slots=True)
class BenchmarkCaseResult:
    key: BenchmarkKey
    measurements: tuple[MetricMeasurement, ...]

    def __post_init__(self) -> None:
        metrics = tuple(measurement.metric for measurement in self.measurements)
        if metrics != TIMING_METRICS:
            raise BenchmarkError("case result must list all four timing metrics in canonical order")
        required = required_metrics(self.key.mode)
        for measurement in self.measurements:
            should_measure = measurement.metric in required
            if should_measure != (measurement.status is MeasurementStatus.MEASURED):
                raise BenchmarkError(
                    f"{self.key.stable_id}/{measurement.metric.value}: applicability does not match mode"
                )

    def measurement_for(self, metric: TimingMetric) -> MetricMeasurement:
        return self.measurements[TIMING_METRICS.index(metric)]

    @property
    def host_call_including_external_update(self) -> TimingStatistics:
        enqueue = self.measurement_for(TimingMetric.HOST_ENQUEUE)
        metadata = self.measurement_for(TimingMetric.METADATA_UPDATE)
        if enqueue.status is not MeasurementStatus.MEASURED or metadata.status is not MeasurementStatus.MEASURED:
            raise MissingSampleError("host call plus external metadata update requires both measured series")
        if len(enqueue.samples_ns) != len(metadata.samples_ns):
            raise MissingSampleError("host enqueue callback and external update series must have equal sample counts")
        combined = tuple(
            enqueue_ns + metadata_ns
            for enqueue_ns, metadata_ns in zip(enqueue.samples_ns, metadata.samples_ns, strict=True)
        )
        return TimingStatistics.from_samples(combined)

    def to_dict(self) -> dict[str, object]:
        return {
            "key": self.key.to_dict(),
            "measurements": [measurement.to_dict() for measurement in self.measurements],
            "derived": {
                "host_enqueue_callback": self.measurement_for(TimingMetric.HOST_ENQUEUE).statistics.to_dict(),
                "host_call_including_external_update": self.host_call_including_external_update.to_dict(),
            },
        }


def required_metrics(mode: BenchmarkMode) -> frozenset[TimingMetric]:
    common = {
        TimingMetric.HOST_ENQUEUE,
        TimingMetric.METADATA_UPDATE,
        TimingMetric.DEVICE_SPAN,
    }
    if mode is BenchmarkMode.ACLGRAPH:
        common.add(TimingMetric.GRAPH_REPLAY)
    return frozenset(common)


class BenchmarkAccumulator:
    """Strict recorder that rejects missing, duplicate, or wrong metric samples."""

    def __init__(self, schedule: BenchmarkSchedule) -> None:
        self._schedule = schedule
        self._expected_sequence = schedule.all_invocations
        self._expected = frozenset(self._expected_sequence)
        self._recorded: dict[ScheduledInvocation, dict[TimingMetric, int]] = {}

    @property
    def schedule(self) -> BenchmarkSchedule:
        return self._schedule

    def record(
        self,
        invocation: ScheduledInvocation,
        durations_ns: Mapping[TimingMetric | str, int],
    ) -> None:
        if invocation not in self._expected:
            raise BenchmarkError("invocation does not belong to this benchmark schedule")
        if invocation in self._recorded:
            raise DuplicateSampleError(
                f"{invocation.phase.value}/{invocation.key.stable_id}/{invocation.case_iteration} was already recorded"
            )
        expected_invocation = self._expected_sequence[len(self._recorded)]
        if invocation != expected_invocation:
            raise BenchmarkError(
                "benchmark invocation was recorded out of schedule order; "
                f"expected={expected_invocation.phase.value}/{expected_invocation.sequence_ordinal}/"
                f"{expected_invocation.key.stable_id}, got={invocation.phase.value}/"
                f"{invocation.sequence_ordinal}/{invocation.key.stable_id}"
            )
        normalized: dict[TimingMetric, int] = {}
        for metric, value in durations_ns.items():
            try:
                normalized_metric = TimingMetric(metric)
            except ValueError as error:
                raise BenchmarkError(f"unknown timing metric {metric!r}") from error
            normalized[normalized_metric] = _validate_duration(value)
        expected_metrics = required_metrics(invocation.key.mode)
        if set(normalized) != set(expected_metrics):
            missing = sorted(metric.value for metric in expected_metrics - set(normalized))
            extra = sorted(metric.value for metric in set(normalized) - expected_metrics)
            raise MissingSampleError(
                f"{invocation.key.stable_id}: timing metric mismatch; missing={missing}, extra={extra}"
            )
        self._recorded[invocation] = normalized

    def finalize(
        self,
        *,
        metadata: Mapping[str, str] | None = None,
    ) -> BenchmarkReport:
        missing = self._expected - set(self._recorded)
        if missing:
            first = min(
                missing,
                key=lambda invocation: (
                    invocation.phase.value,
                    invocation.sequence_ordinal,
                    invocation.key.stable_id,
                ),
            )
            raise MissingSampleError(
                f"{len(missing)} scheduled invocations are missing; first="
                f"{first.phase.value}/{first.key.stable_id}/{first.case_iteration}"
            )

        cases: list[BenchmarkCaseResult] = []
        for key in self._schedule.keys:
            measurements: list[MetricMeasurement] = []
            applicable = required_metrics(key.mode)
            for metric in TIMING_METRICS:
                if metric not in applicable:
                    measurements.append(
                        MetricMeasurement.not_applicable(
                            metric,
                            reason=f"{metric.value} is not applicable to {key.mode.value}",
                        )
                    )
                    continue
                warmup = self._series_for(key, SamplingPhase.WARMUP, metric)
                samples = self._series_for(key, SamplingPhase.SAMPLE, metric)
                measurements.append(
                    MetricMeasurement.measured(
                        metric,
                        warmup_samples_ns=warmup,
                        samples_ns=samples,
                    )
                )
            cases.append(BenchmarkCaseResult(key=key, measurements=tuple(measurements)))
        return BenchmarkReport(
            policy=self._schedule.policy,
            order=self._schedule.order,
            cases=tuple(cases),
            metadata=tuple(sorted((metadata or {}).items())),
        )

    def _series_for(
        self,
        key: BenchmarkKey,
        phase: SamplingPhase,
        metric: TimingMetric,
    ) -> tuple[int, ...]:
        invocations = (
            self._schedule.warmup_invocations if phase is SamplingPhase.WARMUP else self._schedule.sample_invocations
        )
        selected = sorted(
            (invocation for invocation in invocations if invocation.key == key),
            key=lambda invocation: invocation.case_iteration,
        )
        return tuple(self._recorded[invocation][metric] for invocation in selected)


@dataclass(frozen=True, slots=True)
class BenchmarkReport:
    """Versioned JSON/CSV-serializable benchmark artifact."""

    policy: SamplingPolicy
    order: SamplingOrder
    cases: tuple[BenchmarkCaseResult, ...]
    metadata: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not self.cases:
            raise BenchmarkError("benchmark report must contain at least one case")
        keys = tuple(case.key for case in self.cases)
        if len(set(keys)) != len(keys):
            raise BenchmarkError("benchmark report contains duplicate case keys")
        metadata_keys = tuple(key for key, _value in self.metadata)
        if len(set(metadata_keys)) != len(metadata_keys):
            raise BenchmarkError("benchmark metadata keys must be unique")
        if any(not isinstance(key, str) or not key or not isinstance(value, str) for key, value in self.metadata):
            raise BenchmarkError("benchmark metadata must contain non-empty string pairs")

    @property
    def schema_version(self) -> str:
        return BENCHMARK_SCHEMA_VERSION

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_name": BENCHMARK_SCHEMA_NAME,
            "schema_version": self.schema_version,
            "sampling": {
                "order": self.order.value,
                **self.policy.to_dict(),
            },
            "metadata": dict(self.metadata),
            "cases": [case.to_dict() for case in self.cases],
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, allow_nan=False)

    def to_csv_rows(self) -> tuple[dict[str, object], ...]:
        rows: list[dict[str, object]] = []
        metadata_json = json.dumps(dict(self.metadata), sort_keys=True)
        for case in self.cases:
            workload = case.key.workload
            base: dict[str, object] = {
                "schema_name": BENCHMARK_SCHEMA_NAME,
                "schema_version": self.schema_version,
                "case_id": case.key.stable_id,
                "backend": case.key.backend.value,
                "mode": case.key.mode.value,
                "specialization_key": case.key.specialization_key,
                "batch_bucket": workload.batch_bucket,
                "decode_sequence_length": workload.decode_sequence_length,
                "context_positions": "|".join(str(value) for value in workload.context_positions),
                "num_layers": workload.num_layers,
                "num_reqs_actual": workload.num_reqs_actual,
                "topology": workload.topology,
                "trace_seed": "" if workload.trace_seed is None else workload.trace_seed,
                "sampling_order": self.order.value,
                "warmup_iterations": self.policy.warmup_iterations,
                "sample_iterations": self.policy.sample_iterations,
                "metadata_json": metadata_json,
            }
            for measurement in case.measurements:
                rows.append(
                    _csv_measurement_row(
                        base,
                        metric=measurement.metric.value,
                        status=measurement.status,
                        reason=measurement.reason,
                        statistics=measurement.statistics,
                    )
                )
            rows.append(
                _csv_measurement_row(
                    base,
                    metric="host_call_including_external_update",
                    status=MeasurementStatus.MEASURED,
                    reason=None,
                    statistics=case.host_call_including_external_update,
                )
            )
        return tuple(rows)

    def to_csv(self) -> str:
        rows = self.to_csv_rows()
        output = io.StringIO(newline="")
        writer = csv.DictWriter(output, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        return output.getvalue()


def _csv_measurement_row(
    base: Mapping[str, object],
    *,
    metric: str,
    status: MeasurementStatus,
    reason: str | None,
    statistics: TimingStatistics | None,
) -> dict[str, object]:
    row = dict(base)
    row.update(
        {
            "metric": metric,
            "status": status.value,
            "reason": reason or "",
            "count": statistics.count if statistics else 0,
            "min_ns": statistics.minimum_ns if statistics else "",
            "mean_ns": statistics.mean_ns if statistics else "",
            "std_ns": statistics.std_ns if statistics else "",
            "p50_ns": statistics.p50_ns if statistics else "",
            "p90_ns": statistics.p90_ns if statistics else "",
            "p99_ns": statistics.p99_ns if statistics else "",
        }
    )
    return row


__all__ = [
    "BENCHMARK_SCHEMA_NAME",
    "BENCHMARK_SCHEMA_VERSION",
    "TIMING_METRICS",
    "BenchmarkAccumulator",
    "BenchmarkBackend",
    "BenchmarkCaseResult",
    "BenchmarkError",
    "BenchmarkKey",
    "BenchmarkMode",
    "BenchmarkReport",
    "BenchmarkSchedule",
    "DuplicateSampleError",
    "MeasurementStatus",
    "MetricMeasurement",
    "MissingSampleError",
    "SamplingOrder",
    "SamplingPhase",
    "SamplingPolicy",
    "ScheduledInvocation",
    "TimingMetric",
    "TimingStatistics",
    "WorkloadKey",
    "build_benchmark_schedule",
    "required_metrics",
]
