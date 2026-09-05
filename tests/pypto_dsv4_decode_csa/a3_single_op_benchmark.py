# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Executable A3 B4/S8 single-op benchmark for native DSA and PyPTO.

One invocation of this module owns exactly one PyPTO runtime.  Run TRB and HBG
in different fresh processes.  The device runner constructs two equal real
ratio-4 layers and two disjoint cache worlds in the same process, then executes
the existing Host-only benchmark schema with same-stream ABBA ordering.

The scheduling/timing engine is deliberately device-agnostic.  Host unit tests
exercise it with fake events; NPU imports and allocations happen only inside
``run_a3_single_op_benchmark``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Protocol

from tests.pypto_dsv4_decode_csa.benchmark import (
    BENCHMARK_SCHEMA_NAME,
    BENCHMARK_SCHEMA_VERSION,
    BenchmarkAccumulator,
    BenchmarkBackend,
    BenchmarkError,
    BenchmarkKey,
    BenchmarkMode,
    BenchmarkReport,
    BenchmarkSchedule,
    SamplingOrder,
    SamplingPolicy,
    ScheduledInvocation,
    TimingMetric,
    WorkloadKey,
    build_benchmark_schedule,
)

TRB_RUNTIME = "tensormap_and_ringbuffer"
HBG_RUNTIME = "host_build_graph"
SUPPORTED_RUNTIMES = (TRB_RUNTIME, HBG_RUNTIME)
SUPPORTED_DEVICE = 0
SUPPORTED_BATCH = 4
SUPPORTED_SEQ = 8
DEFAULT_PROFILE_ROUNDS = 4
MAX_PROFILE_ROUNDS = 16
TASK_QUEUE_ENV = "TASK_QUEUE_ENABLE"
_TASK_QUEUE_ENABLE_BY_MODE = {
    BenchmarkMode.EAGER: "2",
    BenchmarkMode.ACLGRAPH: "1",
}
STEADY_STATE_EXCLUDED_SETUP = (
    "program compile",
    "PTOAS/codegen",
    "binary registration, materialization, and loading",
    "runtime/context/owner creation, prepare, and callable setup",
    "weight packing and preparation",
    "ordinary eager warmup",
    "ACLGraph warmup",
    "ACLGraph capture and graph construction",
    "first invocation/replay at the final sampled addresses",
    "one-time structure/binding-cache population",
    "timing-event pool and runtime-handle initialization",
    "benchmark warmup iterations",
    "correctness golden generation, execution, comparison, and validation-only copies",
)
STEADY_STATE_INCLUDED_REPEATED_PATH = (
    "per-call validation required by the production path",
    "tensor-address and scalar patching required by the production path",
    "taskQueue enqueue and dequeue",
    "AICPU/AICore device scheduling",
    "kernel execution",
    "repeated binding-cache replacement or cache miss caused by the measured workload",
)

# PyPTO binaries and runtime-owned objects are process pinned.  Cleanup does
# not make switching TRB/HBG in one process valid, so remember the first claim
# and fail closed even if a caller catches an earlier exception.
_PROCESS_RUNTIME: str | None = None


class NativeProductionUnsupported(BenchmarkError):
    """The runner cannot prove the requested production-native baseline."""


class TimingEvent(Protocol):
    """Small common surface implemented by torch.npu.Event and Host fakes."""

    def record(self) -> None: ...

    def elapsed_time(self, end_event: TimingEvent) -> float: ...


class TimedCase(Protocol):
    """One backend participating in a common benchmark schedule."""

    key: BenchmarkKey

    def prepare_invocation(self) -> None:
        """Enqueue the per-call fixed-address input update, without syncing."""

    def enqueue(self) -> None:
        """Enqueue exactly one eager op or one graph replay, without syncing."""


@dataclass(frozen=True, slots=True)
class A3SingleOpBenchmarkConfig:
    """Validated Phase-6 benchmark configuration."""

    runtime: str
    mode: BenchmarkMode | str
    device: int = SUPPORTED_DEVICE
    batch: int = SUPPORTED_BATCH
    warmup_iterations: int = 20
    sample_iterations: int = 100
    enqueue_batch_size: int = 20
    start_position: int = 0
    atol: float = 0.1
    rtol: float = 0.1
    seed: int = 20260902
    master_port: int = 29731
    profile_directory: str | None = None
    profile_rounds: int = DEFAULT_PROFILE_ROUNDS

    def __post_init__(self) -> None:
        if self.runtime not in SUPPORTED_RUNTIMES:
            raise BenchmarkError(f"runtime must be one of {SUPPORTED_RUNTIMES}, got {self.runtime!r}")
        try:
            mode = BenchmarkMode(self.mode)
        except ValueError as error:
            raise BenchmarkError(f"unsupported benchmark mode {self.mode!r}") from error
        object.__setattr__(self, "mode", mode)
        if self.device != SUPPORTED_DEVICE:
            raise BenchmarkError("the Phase-6 A3 runner is intentionally pinned to device 0")
        if self.batch != SUPPORTED_BATCH:
            raise BenchmarkError("the Phase-6 apples-to-apples workload is fixed to B4/S8")
        # SamplingPolicy owns the normative ranges and error messages.
        SamplingPolicy(
            warmup_iterations=self.warmup_iterations,
            sample_iterations=self.sample_iterations,
        )
        if (
            isinstance(self.enqueue_batch_size, bool)
            or not isinstance(self.enqueue_batch_size, int)
            or not 1 <= self.enqueue_batch_size <= 100
        ):
            raise BenchmarkError("enqueue_batch_size must be within [1, 100]")
        if isinstance(self.start_position, bool) or not isinstance(self.start_position, int) or self.start_position < 0:
            raise BenchmarkError("start_position must be a non-negative integer")
        if not math.isfinite(self.atol) or self.atol < 0 or not math.isfinite(self.rtol) or self.rtol < 0:
            raise BenchmarkError("atol and rtol must be finite and non-negative")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise BenchmarkError("seed must be an integer")
        if (
            isinstance(self.master_port, bool)
            or not isinstance(self.master_port, int)
            or not 1 <= self.master_port <= 65_535
        ):
            raise BenchmarkError("master_port must be within [1, 65535]")
        if self.profile_directory is not None:
            if not isinstance(self.profile_directory, str) or not self.profile_directory.strip():
                raise BenchmarkError("profile_directory must be a non-empty path when profiling is enabled")
        if (
            isinstance(self.profile_rounds, bool)
            or not isinstance(self.profile_rounds, int)
            or not 1 <= self.profile_rounds <= MAX_PROFILE_ROUNDS
        ):
            raise BenchmarkError(f"profile_rounds must be within [1, {MAX_PROFILE_ROUNDS}]")

    @property
    def policy(self) -> SamplingPolicy:
        return SamplingPolicy(
            warmup_iterations=self.warmup_iterations,
            sample_iterations=self.sample_iterations,
        )


@dataclass(frozen=True, slots=True)
class CorrectnessGateResult:
    """Correctness evidence produced before any benchmark timing starts."""

    output: Mapping[str, Any]
    states: Mapping[str, Mapping[str, Any]]
    indexer_quantized: Mapping[str, Any]
    close: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "output": dict(self.output),
            "states": {name: dict(value) for name, value in self.states.items()},
            "indexer_quantized": dict(self.indexer_quantized),
            "close": self.close,
        }


@dataclass(frozen=True, slots=True)
class ProfileArtifacts:
    """Validated files produced by one optional diagnostic profile run."""

    run_directory: str
    kernel_details_csv: tuple[str, ...]
    trace_view_json: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_directory": self.run_directory,
            "kernel_details_csv": list(self.kernel_details_csv),
            "trace_view_json": list(self.trace_view_json),
        }


@dataclass(slots=True)
class _DeviceCase:
    key: BenchmarkKey
    hidden_states: Any
    source_hidden_states: Any
    callback: Callable[[], None]

    def prepare_invocation(self) -> None:
        # B4/S8 uses stable metadata and an already-resident input produced by
        # the preceding graph/op.  There is no external per-call update.  The
        # schema retains this separately measured no-op boundary so it never
        # gets confused with the real binding/argument work inside enqueue().
        return

    def enqueue(self) -> None:
        self.callback()


def _required_task_queue_enable(mode: BenchmarkMode | str) -> str:
    try:
        normalized_mode = BenchmarkMode(mode)
    except ValueError as error:
        raise BenchmarkError(f"unsupported benchmark mode {mode!r}") from error
    return _TASK_QUEUE_ENABLE_BY_MODE[normalized_mode]


def _validate_task_queue_environment(mode: BenchmarkMode | str) -> str:
    """Fail before torch_npu import when the process queue mode is ambiguous."""
    normalized_mode = BenchmarkMode(mode)
    expected = _required_task_queue_enable(normalized_mode)
    actual = os.environ.get(TASK_QUEUE_ENV)
    if actual != expected:
        raise BenchmarkError(
            f"{normalized_mode.value} benchmark requires {TASK_QUEUE_ENV}={expected} "
            f"in the process environment before importing torch/torch_npu; got {actual!r}; "
            "relaunch a fresh process"
        )
    return actual


def _build_measurement_metadata(
    config: A3SingleOpBenchmarkConfig,
    correctness: CorrectnessGateResult,
    *,
    task_queue_enable: str,
    pid: int | None = None,
) -> dict[str, str]:
    """Describe the exact steady-state boundary without changing timed work."""
    process_id = os.getpid() if pid is None else pid
    required_task_queue_enable = _required_task_queue_enable(config.mode)
    if task_queue_enable != required_task_queue_enable:
        raise BenchmarkError(
            f"measurement metadata expected {TASK_QUEUE_ENV}={required_task_queue_enable}, got {task_queue_enable!r}"
        )
    graph_patch_semantics = (
        "ACLGraph replays the already-captured fixed B4/S8 tensor/scalar snapshot; "
        "there is no per-replay Python metadata/address/scalar patch to move outside timing"
        if config.mode is BenchmarkMode.ACLGRAPH
        else (
            "each eager custom-op call performs metadata filtering/extraction, tensor/scalar validation, "
            "runtime argument construction and address patch before its taskQueue enqueue; all remain inside "
            "host_enqueue"
        )
    )
    task_queue_contract = (
        "TASK_QUEUE_ENABLE=1 as required by torch_npu 2.12 ACLGraph capture; steady replay is graph.replay()"
        if config.mode is BenchmarkMode.ACLGRAPH
        else "TASK_QUEUE_ENABLE=2; the full custom-op call uses RunOpApiV2 producer/consumer taskQueue"
    )
    task_queue_coverage = (
        "host_enqueue is exactly graph.replay; device_span covers dequeue/launch of captured nodes plus device "
        "execution; the post-tail-event caller synchronization only collects/quiesces completed event work and "
        "is not reported as a latency metric"
        if config.mode is BenchmarkMode.ACLGRAPH
        else (
            "host_enqueue includes the full custom-op and producer-side RunOpApiV2 enqueue; same-queue "
            "start/op/end ordering makes device_span include consumer dequeue/launch gaps plus device execution; "
            "the current runner does not time the final caller-stream quiesce, so these eager measurements are "
            "attribution diagnostics rather than a formal steady-state acceptance metric"
        )
    )
    measurement_contract = {
        "goal": "steady_state_only",
        "sample_phase": "sample",
        "compile_in_acceptance": False,
        "warmup_in_acceptance": False,
        "pre_sample_quiesced": True,
        "excluded_setup": list(STEADY_STATE_EXCLUDED_SETUP),
        "included_repeated_path": list(STEADY_STATE_INCLUDED_REPEATED_PATH),
        "captured_binding": "fixed" if config.mode is BenchmarkMode.ACLGRAPH else "per_call",
        "primary_metric": (
            "device_span"
            if config.mode is BenchmarkMode.ACLGRAPH
            else "unavailable: steady_enqueue_batch_critical_path_not_implemented"
        ),
        "formal_acceptance_supported": config.mode is BenchmarkMode.ACLGRAPH,
        "host_enqueue_and_device_span_are_non_additive": True,
        "first_final_binding_primed": True,
        "event_handles_primed_by_benchmark_warmup": True,
        "correctness_gate_completed_before_sampling": correctness.close,
    }
    return {
        "runtime": config.runtime,
        "fresh_process_runtime": "true",
        "pid": str(process_id),
        "device": "A3:0",
        "batch": "4",
        "sequence_length": "8",
        "native_path": "production vllm::dsa_forward with default multistream overlap enabled",
        "native_multistream_dsv4_dsa_overlap": "true",
        "native_device_span_coverage": (
            "caller-stream tail event is complete: every production DSA auxiliary-stream branch "
            "joins back to the caller stream before dsa_forward returns"
        ),
        "state_worlds": "independent equal weights/input; disjoint mutable caches/output",
        "ordering": "same-card same-stream ABBA",
        "pre_timing_correctness_gate": json.dumps(correctness.to_dict(), sort_keys=True, allow_nan=False),
        "steady_state_statistics": (
            "p50/p90/p99 are computed only from SamplingPhase.SAMPLE; cold setup and the separately retained "
            "SamplingPhase.WARMUP series never contribute"
        ),
        "timing_excludes": ", ".join(STEADY_STATE_EXCLUDED_SETUP),
        "timing_includes_repeated_path": ", ".join(STEADY_STATE_INCLUDED_REPEATED_PATH),
        "measurement_contract": json.dumps(measurement_contract, sort_keys=True),
        "cold_path_boundary": (
            "compile/PTOAS/codegen, binary registration/materialization/loading, runtime/context/owner/callable "
            "creation and prepare, weight packing/preparation, ordinary eager/ACLGraph/benchmark warmup, optional "
            "ACLGraph capture/build, correctness generation/execution/comparison/validation-only copies, the first "
            "invocation/replay at final sampled addresses, one-time structure/binding-cache population, and "
            "timing-event/runtime-handle initialization all complete and caller-quiesce before SamplingPhase.SAMPLE"
        ),
        "metadata_update_semantics": (
            "separate static no-op boundary: B4/S8 external metadata does not change; wrapper metadata binding "
            "and tensor/scalar/address argument patch work is not precomputed and remains inside host_enqueue"
        ),
        "per_call_patch_coverage": graph_patch_semantics,
        "external_input_update": (
            "no backend-specific update is required in this steady workload: input is already device-resident; "
            "only backend-independent, byte-identical correctness/phase-reset fixture copies are outside timing"
        ),
        "host_enqueue_semantics": (
            "producer-side full custom-op call through taskQueue in eager; exact graph.replay enqueue in aclgraph"
        ),
        "task_queue_enable": task_queue_enable,
        "task_queue_contract": task_queue_contract,
        "task_queue_validated_before_torch_npu_import": "true",
        "task_queue_coverage": task_queue_coverage,
        "device_span_semantics": (
            "pre-created same-stream NPU timing events surround each full op/replay; host_enqueue and device_span "
            "are overlapping views and must not be added"
        ),
        "graph_replay_semantics": (
            "N/A in eager; in aclgraph aliases host_enqueue because the enqueue is exactly graph.replay"
        ),
        "caller_quiesce": f"stream synchronize after at most {config.enqueue_batch_size} enqueues",
    }


def _claim_process_runtime(runtime: str) -> None:
    global _PROCESS_RUNTIME
    if runtime not in SUPPORTED_RUNTIMES:
        raise BenchmarkError(f"unsupported runtime {runtime!r}")
    if _PROCESS_RUNTIME is None:
        _PROCESS_RUNTIME = runtime
        return
    if runtime != _PROCESS_RUNTIME:
        raise BenchmarkError(
            f"TRB and HBG must run in different fresh processes; this process already claimed {_PROCESS_RUNTIME!r}"
        )


def _require_native_production(layer: object) -> None:
    """Prove the native baseline retained production multistream overlap."""
    impl = getattr(getattr(layer, "dsa_attn", None), "impl", None)
    if impl is None:
        raise NativeProductionUnsupported("unsupported: real AscendDSAImpl is unavailable")
    overlap = getattr(impl, "multistream_dsv4_dsa_overlap", None)
    if overlap is not True:
        raise NativeProductionUnsupported(
            "unsupported: native baseline did not retain production multistream_dsv4_dsa_overlap=True; "
            "the runner will not relabel a serial diagnostic as native_production"
        )


def _positive_host_duration(start_ns: int, end_ns: int) -> int:
    if (
        isinstance(start_ns, bool)
        or isinstance(end_ns, bool)
        or not isinstance(start_ns, int)
        or not isinstance(end_ns, int)
    ):
        raise BenchmarkError("clock_ns must return integer nanoseconds")
    if end_ns < start_ns:
        raise BenchmarkError("clock_ns moved backwards")
    # Extremely fast Host fakes and coarse platform clocks can return zero.
    # The schema requires a positive duration; one nanosecond is the smallest
    # representable censored value and is recorded rather than dropping data.
    return max(1, end_ns - start_ns)


def _device_elapsed_ns(start_event: TimingEvent, end_event: TimingEvent) -> int:
    elapsed_ms = float(start_event.elapsed_time(end_event))
    if not math.isfinite(elapsed_ms) or elapsed_ms < 0:
        raise BenchmarkError(f"device event returned invalid elapsed time {elapsed_ms!r} ms")
    return max(1, round(elapsed_ms * 1_000_000.0))


def execute_profile_abba_rounds(
    native_case: TimedCase,
    pypto_case: TimedCase,
    *,
    profiler: Any,
    record_function: Callable[[str], Any],
    quiesce: Callable[[], None],
    rounds: int,
) -> None:
    """Profile a small, diagnostic-only steady-state ABBA sequence.

    This helper deliberately has no accumulator and cannot contribute samples
    to :func:`execute_benchmark_schedule`.  A profiler step covers one complete
    ``native, PyPTO, PyPTO, native`` round.  Synchronizing before ``step`` keeps
    each round's NPU work within the corresponding profiler step while never
    putting a synchronization inside either operator callback.
    """
    if isinstance(rounds, bool) or not isinstance(rounds, int) or not 1 <= rounds <= MAX_PROFILE_ROUNDS:
        raise BenchmarkError(f"profile rounds must be within [1, {MAX_PROFILE_ROUNDS}]")
    if native_case.key.backend is not BenchmarkBackend.NATIVE_PRODUCTION:
        raise BenchmarkError("profile native case must use the native_production backend")
    if pypto_case.key.backend not in {BenchmarkBackend.PYPTO_TRB, BenchmarkBackend.PYPTO_HBG}:
        raise BenchmarkError("profile PyPTO case must use a PyPTO backend")

    abba = (
        ("native", native_case),
        ("pypto", pypto_case),
        ("pypto", pypto_case),
        ("native", native_case),
    )
    with profiler as active_profiler:
        for _round_index in range(rounds):
            for backend_name, case in abba:
                case.prepare_invocation()
                with record_function(f"csa_steady_state.{backend_name}.enqueue"):
                    case.enqueue()
            quiesce()
            active_profiler.step()


def _create_torch_npu_profiler(torch_npu: Any, run_directory: Path, *, rounds: int) -> Any:
    """Construct the profiler using the repository's proven Qwen settings."""
    profiler_api = torch_npu.profiler
    experimental_config = profiler_api._ExperimentalConfig(
        profiler_level=profiler_api.ProfilerLevel.Level1,
        aic_metrics=profiler_api.AiCMetrics.PipeUtilization,
    )
    return profiler_api.profile(
        activities=[
            profiler_api.ProfilerActivity.NPU,
            profiler_api.ProfilerActivity.CPU,
        ],
        with_stack=True,
        record_shapes=False,
        profile_memory=False,
        experimental_config=experimental_config,
        schedule=profiler_api.schedule(wait=0, warmup=0, active=rounds, repeat=1, skip_first=0),
        on_trace_ready=profiler_api.tensorboard_trace_handler(str(run_directory)),
    )


def _new_profile_run_directory(root: str, *, runtime: str, mode: BenchmarkMode) -> Path:
    """Create a fresh child so stale profiler files can never satisfy validation."""
    profile_root = Path(root).expanduser()
    profile_root.mkdir(parents=True, exist_ok=True)
    run_directory = profile_root / f"decode_csa_{runtime}_{mode.value}_{os.getpid()}_{time.time_ns()}"
    run_directory.mkdir(exist_ok=False)
    return run_directory


def _require_profile_artifacts(run_directory: Path) -> ProfileArtifacts:
    """Fail closed unless torch_npu produced complete, parseable diagnostics.

    CANN's export subprocess can report success even when the downstream
    timeline/relation/view parsers fail.  File size alone is therefore not a
    validity check: a failed TraceView parser can leave a non-empty but
    truncated JSON prefix behind.
    """
    issues: list[str] = []
    kernel_paths = tuple(sorted(run_directory.rglob("kernel_details.csv")))
    trace_paths = tuple(sorted(run_directory.rglob("trace_view.json")))

    if not kernel_paths:
        issues.append("kernel_details.csv is missing")
    for path in kernel_paths:
        if not path.is_file() or path.stat().st_size == 0:
            issues.append(f"kernel_details.csv is empty: {path}")
            continue
        try:
            with path.open(newline="", encoding="utf-8-sig") as handle:
                rows = csv.DictReader(handle)
                if rows.fieldnames is None or "Name" not in rows.fieldnames:
                    issues.append(f"kernel_details.csv has no Name column: {path}")
                elif next(rows, None) is None:
                    issues.append(f"kernel_details.csv has no kernel rows: {path}")
        except (OSError, UnicodeError, csv.Error) as error:
            issues.append(f"kernel_details.csv is not parseable: {path}: {error}")

    required_markers = {
        "csa_steady_state.native.enqueue",
        "csa_steady_state.pypto.enqueue",
    }
    if not trace_paths:
        issues.append("trace_view.json is missing")
    for path in trace_paths:
        if not path.is_file() or path.stat().st_size == 0:
            issues.append(f"trace_view.json is empty: {path}")
            continue
        try:
            with path.open(encoding="utf-8") as handle:
                trace_events = json.load(handle)
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            issues.append(f"trace_view.json is not valid JSON: {path}: {error}")
            continue
        if not isinstance(trace_events, list) or not trace_events:
            issues.append(f"trace_view.json is not a non-empty event list: {path}")
            continue
        event_names = {
            event.get("name")
            for event in trace_events
            if isinstance(event, dict) and isinstance(event.get("name"), str)
        }
        missing_markers = sorted(required_markers - event_names)
        if missing_markers:
            issues.append(f"trace_view.json is missing CPU enqueue markers {missing_markers}: {path}")

    if issues:
        analyzed_outputs = ", ".join(
            f"{path.relative_to(run_directory)}={path.stat().st_size}B"
            for path in sorted(run_directory.rglob("*"))
            if path.is_file() and path.name in {"analyse.done", "kernel_details.csv", "trace_view.json"}
        )
        parser_logs = ", ".join(
            str(path.relative_to(run_directory)) for path in sorted(run_directory.rglob("*Parser.log"))
        )
        raise BenchmarkError(
            f"torch_npu profiler analysis failed closed under {run_directory}: "
            + "; ".join(issues)
            + f"; analyzed_outputs=[{analyzed_outputs or 'none'}]; parser_logs=[{parser_logs or 'none'}]"
        )

    kernel_details = tuple(str(path.resolve()) for path in kernel_paths)
    trace_views = tuple(str(path.resolve()) for path in trace_paths)
    return ProfileArtifacts(
        run_directory=str(run_directory.resolve()),
        kernel_details_csv=kernel_details,
        trace_view_json=trace_views,
    )


def _run_optional_steady_state_profile(
    config: A3SingleOpBenchmarkConfig,
    native_case: TimedCase,
    pypto_case: TimedCase,
    *,
    torch: Any,
    torch_npu: Any,
    quiesce: Callable[[], None],
) -> ProfileArtifacts | None:
    """Run the profiler only when an explicit output directory was supplied."""
    if config.profile_directory is None:
        return None
    run_directory = _new_profile_run_directory(
        config.profile_directory,
        runtime=config.runtime,
        mode=config.mode,
    )
    profiler = _create_torch_npu_profiler(torch_npu, run_directory, rounds=config.profile_rounds)
    execute_profile_abba_rounds(
        native_case,
        pypto_case,
        profiler=profiler,
        record_function=torch.profiler.record_function,
        quiesce=quiesce,
        rounds=config.profile_rounds,
    )
    return _require_profile_artifacts(run_directory)


def execute_benchmark_schedule(
    schedule: BenchmarkSchedule,
    cases: Mapping[BenchmarkKey, TimedCase],
    *,
    event_factory: Callable[[], TimingEvent],
    quiesce: Callable[[], None],
    between_phases: Callable[[], None],
    enqueue_batch_size: int,
    clock_ns: Callable[[], int] = time.perf_counter_ns,
    metadata: Mapping[str, str] | None = None,
) -> BenchmarkReport:
    """Execute a schedule without putting synchronization inside an op call.

    A bounded pool of timing events is created before timing and reused only
    after the caller quiesces a submitted batch.  DEVICE_SPAN is measured by
    events around the op/replay.  HOST_ENQUEUE measures the complete
    ``case.enqueue`` callback through Host return, so implementation-internal
    metadata/binding/address patch work stays included; METADATA_UPDATE is the
    separate external update boundary (a static no-op for B4/S8).  In ACLGraph
    mode GRAPH_REPLAY intentionally aliases the same raw Host duration as
    HOST_ENQUEUE because the measured callback is exactly ``graph.replay()``.
    """
    if isinstance(enqueue_batch_size, bool) or not isinstance(enqueue_batch_size, int) or enqueue_batch_size <= 0:
        raise BenchmarkError("enqueue_batch_size must be a positive integer")
    if set(cases) != set(schedule.keys):
        missing = sorted(key.stable_id for key in set(schedule.keys) - set(cases))
        extra = sorted(key.stable_id for key in set(cases) - set(schedule.keys))
        raise BenchmarkError(f"timed cases do not match schedule; missing={missing}, extra={extra}")
    if any(case.key != key for key, case in cases.items()):
        raise BenchmarkError("timed case mapping key does not match case.key")

    # torch.npu.Event is lazy at the C level, but all Python event owners are
    # allocated before the first timed call.  The pool is bounded and reused
    # only after elapsed times have been consumed.
    # Never allocate more reusable pairs than the warmup phase can touch.  NPU
    # Event owns a Python object immediately but initializes its runtime handle
    # lazily on first record; this cap guarantees every pair is initialized in
    # warmup before it can contribute a steady-state sample.
    event_pool_size = min(enqueue_batch_size, len(schedule.warmup_invocations))
    event_pairs = tuple((event_factory(), event_factory()) for _ in range(event_pool_size))
    accumulator = BenchmarkAccumulator(schedule)

    def execute_phase(invocations: Sequence[ScheduledInvocation]) -> None:
        for chunk_start in range(0, len(invocations), event_pool_size):
            chunk = invocations[chunk_start : chunk_start + event_pool_size]
            pending: list[tuple[ScheduledInvocation, int, int, TimingEvent, TimingEvent]] = []
            for invocation, (start_event, end_event) in zip(chunk, event_pairs, strict=False):
                case = cases[invocation.key]
                metadata_start = clock_ns()
                case.prepare_invocation()
                metadata_ns = _positive_host_duration(metadata_start, clock_ns())

                start_event.record()
                enqueue_start = clock_ns()
                case.enqueue()
                host_enqueue_ns = _positive_host_duration(enqueue_start, clock_ns())
                end_event.record()
                pending.append((invocation, metadata_ns, host_enqueue_ns, start_event, end_event))

            # This is the sole timed-loop synchronization boundary.  It is a
            # caller action after a batch of enqueues, never part of an op.
            quiesce()
            for invocation, metadata_ns, host_enqueue_ns, start_event, end_event in pending:
                durations = {
                    TimingMetric.HOST_ENQUEUE: host_enqueue_ns,
                    TimingMetric.METADATA_UPDATE: metadata_ns,
                    TimingMetric.DEVICE_SPAN: _device_elapsed_ns(start_event, end_event),
                }
                if invocation.key.mode is BenchmarkMode.ACLGRAPH:
                    durations[TimingMetric.GRAPH_REPLAY] = host_enqueue_ns
                accumulator.record(invocation, durations)

    execute_phase(schedule.warmup_invocations)
    between_phases()
    execute_phase(schedule.sample_invocations)
    return accumulator.finalize(metadata=metadata)


def _benchmark_keys(config: A3SingleOpBenchmarkConfig) -> tuple[BenchmarkKey, BenchmarkKey]:
    workload = WorkloadKey(
        batch_bucket=SUPPORTED_BATCH,
        decode_sequence_length=SUPPORTED_SEQ,
        context_positions=(config.start_position,),
        num_layers=1,
        num_reqs_actual=SUPPORTED_BATCH,
        topology="single_op_ratio4_decode_csa",
        trace_seed=config.seed,
    )
    native = BenchmarkKey(
        backend=BenchmarkBackend.NATIVE_PRODUCTION,
        mode=config.mode,
        workload=workload,
        specialization_key="A3-TP1-B4-S8-ratio4",
    )
    pypto = BenchmarkKey(
        backend=(BenchmarkBackend.PYPTO_TRB if config.runtime == TRB_RUNTIME else BenchmarkBackend.PYPTO_HBG),
        mode=config.mode,
        workload=workload,
        specialization_key="A3-TP1-B4-S8-ratio4",
    )
    return native, pypto


def _restore_world(world: Any, pristine: Sequence[Any]) -> None:
    if len(world.raw_cache_tuple) != len(pristine):
        raise BenchmarkError("pristine snapshot does not cover all six cache families")
    for cache, initial in zip(world.raw_cache_tuple, pristine, strict=True):
        cache.copy_(initial)


def _initialize_case_inputs(*cases: _DeviceCase) -> None:
    """Fill every fixed-address input before eager execution or graph replay."""
    for case in cases:
        case.hidden_states.copy_(case.source_hidden_states)


def _compare_current_results(
    *,
    native_world: Any,
    pypto_world: Any,
    initial_state: Mapping[str, Any],
    native_output: Any,
    pypto_output: Any,
    atol: float,
    rtol: float,
    phase: str,
) -> CorrectnessGateResult:
    """Compare current output and all six state families without launching work."""
    from tests.pypto_dsv4_decode_csa.a3_native_compare import (
        compare_state_snapshots,
        compare_tensors,
    )
    from tests.pypto_dsv4_decode_csa.indexer_quantized_comparison import compare_indexer_quantized

    native_output_host = native_output.detach().cpu()
    pypto_output_host = pypto_output.detach().cpu()
    output = compare_tensors(native_output_host, pypto_output_host, atol=atol, rtol=rtol)
    native_state = native_world.snapshot()
    pypto_state = pypto_world.snapshot()
    states = compare_state_snapshots(
        native_state,
        pypto_state,
        initial_state,
        atol=atol,
        rtol=rtol,
    )
    indexer = compare_indexer_quantized(
        native_state["indexer_k"],
        pypto_state["indexer_k"],
        initial_state["indexer_k"],
        native_state["indexer_scale"],
        pypto_state["indexer_scale"],
        initial_state["indexer_scale"],
    )
    non_quantized_close = all(
        value.close for name, value in states.items() if name not in {"indexer_k", "indexer_scale"}
    )
    finite_nonzero = bool(
        output.finite and float(native_output_host.abs().max()) > 0.0 and float(pypto_output_host.abs().max()) > 0.0
    )
    close = bool(output.close and non_quantized_close and indexer.acceptable and finite_nonzero)
    result = CorrectnessGateResult(
        output=asdict(output),
        states={name: asdict(value) for name, value in states.items()},
        indexer_quantized=indexer.to_dict(),
        close=close,
    )
    if not result.close:
        raise AssertionError(
            f"correctness gate failed {phase}:\n"
            + json.dumps(result.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)
        )
    return result


def _correctness_gate(
    *,
    native_case: _DeviceCase,
    pypto_case: _DeviceCase,
    native_world: Any,
    pypto_world: Any,
    initial_state: Mapping[str, Any],
    native_pristine: Sequence[Any],
    pypto_pristine: Sequence[Any],
    native_output: Any,
    pypto_output: Any,
    quiesce: Callable[[], None],
    atol: float,
    rtol: float,
) -> CorrectnessGateResult:
    """Run one golden comparison before creating benchmark timing events."""
    _restore_world(native_world, native_pristine)
    _restore_world(pypto_world, pypto_pristine)
    _initialize_case_inputs(native_case, pypto_case)
    quiesce()
    native_case.enqueue()
    pypto_case.enqueue()
    quiesce()
    return _compare_current_results(
        native_world=native_world,
        pypto_world=pypto_world,
        initial_state=initial_state,
        native_output=native_output,
        pypto_output=pypto_output,
        atol=atol,
        rtol=rtol,
        phase="before benchmark timing",
    )


def run_a3_single_op_benchmark(config: A3SingleOpBenchmarkConfig) -> BenchmarkReport:
    """Run production-native and PyPTO in one fresh process on A3 device 0."""
    if not isinstance(config, A3SingleOpBenchmarkConfig):
        raise TypeError("config must be A3SingleOpBenchmarkConfig")
    task_queue_enable = _validate_task_queue_environment(config.mode)
    _claim_process_runtime(config.runtime)

    # Keep all heavyweight imports behind the executable entry point so Host
    # CLI/scheduler/result tests never initialize an NPU runtime.
    import torch
    import torch_npu
    from vllm.config import set_current_vllm_config

    import vllm_ascend.ops.dsa  # noqa: F401 -- register vllm::dsa_forward
    from tests.pypto_dsv4_decode_csa.a3_native_compare import (
        _custom_op_call,
        _destroy_tp1,
        _forward_context,
        _initialize_tp1,
    )
    from tests.pypto_dsv4_decode_csa.native_fixture import (
        bind_state_world,
        build_real_native_metadata,
        build_real_ratio4_attention,
        build_synthetic_vllm_config,
        build_twin_decode_csa_state_worlds,
    )
    from vllm_ascend.ascend_config import init_ascend_config
    from vllm_ascend.ops._pypto_dsv4_csa import (
        DecodeCSADeviceOwnerRegistry,
        DecodeCSAProgramSpec,
        install_pypto_dsv4_decode_csa,
        uninstall_pypto_dsv4_decode_csa,
    )
    from vllm_ascend.ops._pypto_dsv4_csa.config import FLASH
    from vllm_ascend.utils import register_ascend_customop

    native_prefix = "model.layers.2.self_attn"
    pypto_prefix = "model.layers.4.self_attn"
    config_owner = build_synthetic_vllm_config(
        batch=SUPPORTED_BATCH,
        prefix=(native_prefix, pypto_prefix),
    )
    registry = DecodeCSADeviceOwnerRegistry()
    owner = None
    native_graph = None
    pypto_graph = None
    pypto_wrapper = None
    tp_initialized = False
    target = torch.device("npu", config.device)
    stream = None
    try:
        with set_current_vllm_config(config_owner.vllm_config):
            _initialize_tp1(device=config.device, master_port=config.master_port)
            tp_initialized = True
            register_ascend_customop(config_owner.vllm_config)

            # Two real production layers permit true same-process ABBA without
            # installing/removing a private dispatch hook in the timed path.
            # Identical seeds/topologies produce equivalent read-only weights;
            # layer prefixes and all mutable state/input/output addresses are
            # disjoint.
            additional_config = config_owner.vllm_config.additional_config
            if not isinstance(additional_config, dict):
                raise NativeProductionUnsupported("unsupported: additional_config is not mutable")
            # The fixture defaults to overlap=false for functional comparisons.
            # Remove that override before constructing the formal baseline so
            # AscendConfig applies its production default (True).  This never
            # mutates the constructed native implementation.
            additional_config.pop("multistream_dsv4_dsa_overlap", None)
            additional_config["refresh"] = True
            init_ascend_config(config_owner.vllm_config)
            native_attention = build_real_ratio4_attention(
                config_owner,
                device=target,
                prefix=native_prefix,
                seed=config.seed,
            )
            native_wrapper = native_attention.dsa_attn
            _require_native_production(native_wrapper)

            # PyPTO is one L1 op and intentionally has no production-native
            # auxiliary stream.  Build only its independent comparison layer
            # under overlap=false; the native object above remains untouched.
            additional_config["multistream_dsv4_dsa_overlap"] = False
            init_ascend_config(config_owner.vllm_config)
            pypto_attention = build_real_ratio4_attention(
                config_owner,
                device=target,
                prefix=pypto_prefix,
                seed=config.seed,
            )
            pypto_wrapper = pypto_attention.dsa_attn
            # Restore the process-level config to its production default for
            # execution; both already-constructed impls retain their intended
            # per-layer value (native=True, PyPTO comparison layer=False).
            additional_config.pop("multistream_dsv4_dsa_overlap", None)
            init_ascend_config(config_owner.vllm_config)
            additional_config.pop("refresh", None)

            spec = DecodeCSAProgramSpec(batch=SUPPORTED_BATCH)
            if spec.seq != SUPPORTED_SEQ:
                raise BenchmarkError(f"runner expected S8, program reported S{spec.seq}")
            native_world, pypto_world = build_twin_decode_csa_state_worlds(
                spec,
                device=target,
                start_positions=(config.start_position,) * SUPPORTED_BATCH,
            )
            initial_state = native_world.snapshot()
            native_pristine = tuple(tensor.detach().clone() for tensor in native_world.raw_cache_tuple)
            pypto_pristine = tuple(tensor.detach().clone() for tensor in pypto_world.raw_cache_tuple)

            bind_state_world(native_wrapper, native_world)
            bind_state_world(pypto_wrapper, pypto_world)
            native_metadata = build_real_native_metadata(
                native_wrapper,
                native_world,
                config_owner.vllm_config,
            )
            pypto_metadata = build_real_native_metadata(
                pypto_wrapper,
                pypto_world,
                config_owner.vllm_config,
            )
            native_context = _forward_context(native_wrapper, native_metadata)
            pypto_context = _forward_context(pypto_wrapper, pypto_metadata)

            # Weight packing, program compilation and prepare all complete here,
            # before any schema warmup/sample duration is collected.
            owner = install_pypto_dsv4_decode_csa(
                pypto_wrapper,
                kv_cache=pypto_world.raw_cache_tuple,
                attn_metadata=pypto_metadata.metadata,
                device=config.device,
                runtime=config.runtime,
                batch_buckets=(SUPPORTED_BATCH,),
                registry=registry,
                uniform_query_rows_contract=True,
            )
            grad_packed = tuple(name for name, tensor in owner.weights.launch_arguments.items() if tensor.requires_grad)
            if grad_packed:
                raise AssertionError(f"packed inference tensors retained autograd: {grad_packed}")

            host_generator = torch.Generator(device="cpu")
            host_generator.manual_seed(config.seed + 1)
            source_hidden = (
                torch.randn(
                    (spec.tokens, FLASH.hidden_size),
                    dtype=torch.bfloat16,
                    generator=host_generator,
                )
                .mul_(0.125)
                .to(target)
            )
            native_warm_hidden = source_hidden.clone()
            pypto_warm_hidden = source_hidden.clone()
            native_warm_output = torch.empty_like(source_hidden)
            pypto_warm_output = torch.empty_like(source_hidden)
            native_hidden = torch.empty_like(source_hidden)
            pypto_hidden = torch.empty_like(source_hidden)
            native_output = torch.empty_like(source_hidden)
            pypto_output = torch.empty_like(source_hidden)
            if not all(
                left.data_ptr() != right.data_ptr()
                for left, right in zip(
                    (native_hidden, native_output, *native_world.raw_cache_tuple),
                    (pypto_hidden, pypto_output, *pypto_world.raw_cache_tuple),
                    strict=True,
                )
            ):
                raise AssertionError("native and PyPTO benchmark state worlds must have disjoint addresses")

            native_warm_call = lambda: _custom_op_call(
                native_wrapper,
                native_context,
                native_warm_hidden,
                native_warm_output,
            )
            pypto_warm_call = lambda: _custom_op_call(
                pypto_wrapper,
                pypto_context,
                pypto_warm_hidden,
                pypto_warm_output,
            )
            native_direct_call = lambda: _custom_op_call(
                native_wrapper,
                native_context,
                native_hidden,
                native_output,
            )
            pypto_direct_call = lambda: _custom_op_call(
                pypto_wrapper,
                pypto_context,
                pypto_hidden,
                pypto_output,
            )

            stream = torch_npu.npu.Stream(device=config.device)
            with torch_npu.npu.stream(stream):
                native_warm_call()
            stream.synchronize()
            _restore_world(native_world, native_pristine)
            torch_npu.npu.synchronize(config.device)

            # The first PyPTO call is a sacrificial compile/warmup launch.  Its
            # external quiescence is part of setup and explicitly communicated
            # to the owner before ordinary L1 launches/capture.
            with torch_npu.npu.stream(stream):
                pypto_warm_call()
            stream.synchronize()
            owner.device_owner.mark_warmups_quiesced()
            _restore_world(pypto_world, pypto_pristine)
            torch_npu.npu.synchronize(config.device)

            if config.mode is BenchmarkMode.ACLGRAPH:
                native_graph = torch_npu.npu.NPUGraph()
                pypto_graph = torch_npu.npu.NPUGraph()
                native_hidden.copy_(source_hidden)
                pypto_hidden.copy_(source_hidden)
                torch_npu.npu.synchronize(config.device)

                native_context.capturing = True
                try:
                    with torch_npu.npu.graph(native_graph, stream=stream):
                        native_direct_call()
                finally:
                    native_context.capturing = False
                stream.synchronize()
                _restore_world(native_world, native_pristine)
                torch_npu.npu.synchronize(config.device)

                pypto_context.capturing = True
                try:
                    with torch_npu.npu.graph(pypto_graph, stream=stream):
                        pypto_direct_call()
                finally:
                    pypto_context.capturing = False
                stream.synchronize()
                _restore_world(pypto_world, pypto_pristine)
                torch_npu.npu.synchronize(config.device)

                native_callback = native_graph.replay
                pypto_callback = pypto_graph.replay
            else:
                native_callback = native_direct_call
                pypto_callback = pypto_direct_call

            native_key, pypto_key = _benchmark_keys(config)
            native_case = _DeviceCase(native_key, native_hidden, source_hidden, native_callback)
            pypto_case = _DeviceCase(pypto_key, pypto_hidden, source_hidden, pypto_callback)

            def caller_quiesce() -> None:
                assert stream is not None
                stream.synchronize()

            with torch_npu.npu.stream(stream):
                correctness = _correctness_gate(
                    native_case=native_case,
                    pypto_case=pypto_case,
                    native_world=native_world,
                    pypto_world=pypto_world,
                    initial_state=initial_state,
                    native_pristine=native_pristine,
                    pypto_pristine=pypto_pristine,
                    native_output=native_output,
                    pypto_output=pypto_output,
                    quiesce=caller_quiesce,
                    atol=config.atol,
                    rtol=config.rtol,
                )

            schedule = build_benchmark_schedule(
                (native_key, pypto_key),
                policy=config.policy,
                order=SamplingOrder.ABBA,
            )

            def reset_between_phases() -> None:
                # Warmup state evolution is never allowed to leak into the
                # steady-state sample series.  Copies and this sync are outside
                # both timing phases.
                _restore_world(native_world, native_pristine)
                _restore_world(pypto_world, pypto_pristine)
                _initialize_case_inputs(native_case, pypto_case)
                caller_quiesce()

            metadata = _build_measurement_metadata(
                config,
                correctness,
                task_queue_enable=task_queue_enable,
            )
            with torch_npu.npu.stream(stream):
                report = execute_benchmark_schedule(
                    schedule,
                    {native_key: native_case, pypto_key: pypto_case},
                    event_factory=lambda: torch_npu.npu.Event(enable_timing=True),
                    quiesce=caller_quiesce,
                    between_phases=reset_between_phases,
                    enqueue_batch_size=config.enqueue_batch_size,
                    metadata=metadata,
                )
            # ABBA gives each state world exactly the same number of stateful
            # launches.  Validate the final output and all six cache families
            # after the complete 100-1000 sample trace; a one-step pre-gate is
            # not allowed to hide steady-state drift.
            post_timing_correctness = _compare_current_results(
                native_world=native_world,
                pypto_world=pypto_world,
                initial_state=initial_state,
                native_output=native_output,
                pypto_output=pypto_output,
                atol=config.atol,
                rtol=config.rtol,
                phase="after the complete steady-state sample trace",
            )
            metadata["post_timing_correctness_gate"] = json.dumps(
                post_timing_correctness.to_dict(),
                sort_keys=True,
                allow_nan=False,
            )

            # The optional profiler is intentionally later than every cold
            # boundary, ordinary benchmark warmup, formal sample, and formal
            # post-sample correctness gate.  It therefore cannot contribute to
            # or perturb the report's winner/loser samples.  Restore equal
            # worlds before profiling, then validate its diagnostic ABBA work
            # independently after the profiler has stopped.
            if config.profile_directory is not None:
                _restore_world(native_world, native_pristine)
                _restore_world(pypto_world, pypto_pristine)
                _initialize_case_inputs(native_case, pypto_case)
                caller_quiesce()
                with torch_npu.npu.stream(stream):
                    profile_artifacts = _run_optional_steady_state_profile(
                        config,
                        native_case,
                        pypto_case,
                        torch=torch,
                        torch_npu=torch_npu,
                        quiesce=caller_quiesce,
                    )
                assert profile_artifacts is not None
                profile_correctness = _compare_current_results(
                    native_world=native_world,
                    pypto_world=pypto_world,
                    initial_state=initial_state,
                    native_output=native_output,
                    pypto_output=pypto_output,
                    atol=config.atol,
                    rtol=config.rtol,
                    phase="after the diagnostic-only steady-state profile",
                )
                metadata["diagnostic_profile_formal_samples"] = "false"
                metadata["diagnostic_profile_scope"] = (
                    f"{config.profile_rounds} post-benchmark ABBA rounds; profiler activity and results are "
                    "excluded from every formal latency percentile"
                )
                metadata["diagnostic_profile_artifacts"] = json.dumps(
                    profile_artifacts.to_dict(),
                    sort_keys=True,
                    allow_nan=False,
                )
                metadata["diagnostic_profile_correctness_gate"] = json.dumps(
                    profile_correctness.to_dict(),
                    sort_keys=True,
                    allow_nan=False,
                )
            return replace(report, metadata=tuple(sorted(metadata.items())))
    finally:
        try:
            if tp_initialized:
                torch_npu.npu.synchronize(config.device)
                for graph in (pypto_graph, native_graph):
                    if graph is not None:
                        graph.reset()
                if owner is not None:
                    uninstall_pypto_dsv4_decode_csa(owner.layer)
                    owner.device_owner.backend.close()
        finally:
            config_owner.close()
            if tp_initialized:
                _destroy_tp1()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", choices=SUPPORTED_RUNTIMES, required=True)
    parser.add_argument("--mode", choices=tuple(mode.value for mode in BenchmarkMode), required=True)
    parser.add_argument("--device", type=int, choices=(SUPPORTED_DEVICE,), default=SUPPORTED_DEVICE)
    parser.add_argument("--batch", type=int, choices=(SUPPORTED_BATCH,), default=SUPPORTED_BATCH)
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--enqueue-batch-size", type=int, default=20)
    parser.add_argument("--start-position", type=int, default=0)
    parser.add_argument("--atol", type=float, default=0.1)
    parser.add_argument("--rtol", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--master-port", type=int, default=29731)
    parser.add_argument(
        "--profile-directory",
        "--profile-dir",
        help=(
            "enable a diagnostic-only torch_npu Level1 CPU+NPU profile under this directory; "
            "the profiled ABBA rounds never contribute formal benchmark samples"
        ),
    )
    parser.add_argument("--profile-rounds", type=int, default=DEFAULT_PROFILE_ROUNDS)
    return parser


def _config_from_args(args: argparse.Namespace) -> A3SingleOpBenchmarkConfig:
    return A3SingleOpBenchmarkConfig(
        runtime=args.runtime,
        mode=args.mode,
        device=args.device,
        batch=args.batch,
        warmup_iterations=args.warmups,
        sample_iterations=args.samples,
        enqueue_batch_size=args.enqueue_batch_size,
        start_position=args.start_position,
        atol=args.atol,
        rtol=args.rtol,
        seed=args.seed,
        master_port=args.master_port,
        profile_directory=args.profile_directory,
        profile_rounds=args.profile_rounds,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = _config_from_args(args)
    try:
        report = run_a3_single_op_benchmark(config)
    except NativeProductionUnsupported as error:
        print(
            json.dumps(
                {
                    "schema_name": BENCHMARK_SCHEMA_NAME,
                    "schema_version": BENCHMARK_SCHEMA_VERSION,
                    "status": "unsupported",
                    "reason": str(error),
                    "runtime": config.runtime,
                    "mode": config.mode.value,
                    "device": config.device,
                    "batch": config.batch,
                    "sequence_length": SUPPORTED_SEQ,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2
    print(report.to_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "A3SingleOpBenchmarkConfig",
    "CorrectnessGateResult",
    "DEFAULT_PROFILE_ROUNDS",
    "HBG_RUNTIME",
    "MAX_PROFILE_ROUNDS",
    "NativeProductionUnsupported",
    "ProfileArtifacts",
    "SUPPORTED_RUNTIMES",
    "TASK_QUEUE_ENV",
    "TRB_RUNTIME",
    "build_parser",
    "execute_benchmark_schedule",
    "execute_profile_abba_rounds",
    "main",
    "run_a3_single_op_benchmark",
]
