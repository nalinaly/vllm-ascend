# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Host-only tests for the executable A3 single-op benchmark runner."""

from __future__ import annotations

import inspect
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.pypto_dsv4_decode_csa import a3_single_op_benchmark as runner
from tests.pypto_dsv4_decode_csa.benchmark import (
    BENCHMARK_SCHEMA_NAME,
    BENCHMARK_SCHEMA_VERSION,
    BenchmarkBackend,
    BenchmarkError,
    BenchmarkKey,
    BenchmarkMode,
    SamplingOrder,
    SamplingPolicy,
    TimingMetric,
    WorkloadKey,
    build_benchmark_schedule,
)


class _Clock:
    def __init__(self) -> None:
        self.value = 0

    def __call__(self) -> int:
        self.value += 10
        return self.value


class _FakeEvent:
    def __init__(self, log: list[str], ordinal: int) -> None:
        self.log = log
        self.ordinal = ordinal

    def record(self) -> None:
        self.log.append(f"event:{self.ordinal}")

    def elapsed_time(self, end_event: _FakeEvent) -> float:
        assert isinstance(end_event, _FakeEvent)
        return 0.25


class _FakeCase:
    def __init__(self, key: BenchmarkKey, name: str, log: list[str]) -> None:
        self.key = key
        self.name = name
        self.log = log

    def prepare_invocation(self) -> None:
        self.log.append(f"prepare:{self.name}")

    def enqueue(self) -> None:
        self.log.append(f"enqueue:{self.name}")


class _FakeProfiler:
    def __init__(self, log: list[str]) -> None:
        self.log = log

    def __enter__(self) -> _FakeProfiler:
        self.log.append("profiler:enter")
        return self

    def __exit__(self, *_args: object) -> None:
        self.log.append("profiler:exit")

    def step(self) -> None:
        self.log.append("profiler:step")


class _FakeRecordFunction:
    def __init__(self, name: str, log: list[str]) -> None:
        self.name = name
        self.log = log

    def __enter__(self) -> None:
        self.log.append(f"record:enter:{self.name}")

    def __exit__(self, *_args: object) -> None:
        self.log.append(f"record:exit:{self.name}")


def _keys(mode: BenchmarkMode) -> tuple[BenchmarkKey, BenchmarkKey]:
    workload = WorkloadKey(
        batch_bucket=4,
        decode_sequence_length=8,
        context_positions=(0,),
        num_layers=1,
        num_reqs_actual=4,
        topology="single_op_ratio4_decode_csa",
        trace_seed=73,
    )
    return (
        BenchmarkKey(
            backend=BenchmarkBackend.NATIVE_PRODUCTION,
            mode=mode,
            workload=workload,
            specialization_key="A3-TP1-B4-S8-ratio4",
        ),
        BenchmarkKey(
            backend=BenchmarkBackend.PYPTO_TRB,
            mode=mode,
            workload=workload,
            specialization_key="A3-TP1-B4-S8-ratio4",
        ),
    )


def _execute_fake(mode: BenchmarkMode, *, enqueue_batch_size: int = 7):
    keys = _keys(mode)
    schedule = build_benchmark_schedule(
        keys,
        policy=SamplingPolicy(warmup_iterations=20, sample_iterations=100),
        order=SamplingOrder.ABBA,
    )
    log: list[str] = []
    events: list[_FakeEvent] = []

    def event_factory() -> _FakeEvent:
        event = _FakeEvent(log, len(events))
        events.append(event)
        return event

    def quiesce() -> None:
        log.append("quiesce")

    def between_phases() -> None:
        log.append("between")

    cases = {
        keys[0]: _FakeCase(keys[0], "A", log),
        keys[1]: _FakeCase(keys[1], "B", log),
    }
    report = runner.execute_benchmark_schedule(
        schedule,
        cases,
        event_factory=event_factory,
        quiesce=quiesce,
        between_phases=between_phases,
        enqueue_batch_size=enqueue_batch_size,
        clock_ns=_Clock(),
        metadata={"source": "host-fake"},
    )
    return report, log, events


def test_config_fixes_a3_workload_and_enforces_phase6_sampling_ranges() -> None:
    config = runner.A3SingleOpBenchmarkConfig(
        runtime=runner.TRB_RUNTIME,
        mode="aclgraph",
    )

    assert config.mode is BenchmarkMode.ACLGRAPH
    assert config.device == 0
    assert config.batch == 4
    assert config.policy == SamplingPolicy(warmup_iterations=20, sample_iterations=100)
    assert config.profile_directory is None
    assert config.profile_rounds == runner.DEFAULT_PROFILE_ROUNDS
    native_key, _pypto_key = runner._benchmark_keys(config)
    assert native_key.backend is BenchmarkBackend.NATIVE_PRODUCTION

    invalid = (
        {"runtime": "bad", "mode": "eager"},
        {"runtime": runner.TRB_RUNTIME, "mode": "bad"},
        {"runtime": runner.TRB_RUNTIME, "mode": "eager", "device": 1},
        {"runtime": runner.TRB_RUNTIME, "mode": "eager", "batch": 8},
        {"runtime": runner.TRB_RUNTIME, "mode": "eager", "warmup_iterations": 19},
        {"runtime": runner.TRB_RUNTIME, "mode": "eager", "sample_iterations": 99},
        {"runtime": runner.TRB_RUNTIME, "mode": "eager", "enqueue_batch_size": 0},
        {"runtime": runner.TRB_RUNTIME, "mode": "eager", "profile_directory": ""},
        {"runtime": runner.TRB_RUNTIME, "mode": "eager", "profile_directory": "   "},
        {"runtime": runner.TRB_RUNTIME, "mode": "eager", "profile_rounds": 0},
        {
            "runtime": runner.TRB_RUNTIME,
            "mode": "eager",
            "profile_rounds": runner.MAX_PROFILE_ROUNDS + 1,
        },
    )
    for kwargs in invalid:
        with pytest.raises(BenchmarkError):
            runner.A3SingleOpBenchmarkConfig(**kwargs)


@pytest.mark.parametrize(
    ("mode", "expected"),
    (
        (BenchmarkMode.EAGER, "2"),
        (BenchmarkMode.ACLGRAPH, "1"),
    ),
)
def test_task_queue_mode_is_explicit_and_validated_before_device_imports(
    mode: BenchmarkMode,
    expected: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(runner.TASK_QUEUE_ENV, raising=False)
    with pytest.raises(BenchmarkError, match=rf"{mode.value}.*TASK_QUEUE_ENABLE={expected}.*got None"):
        runner._validate_task_queue_environment(mode)

    wrong = "1" if expected == "2" else "2"
    monkeypatch.setenv(runner.TASK_QUEUE_ENV, wrong)
    with pytest.raises(BenchmarkError, match=rf"before importing torch/torch_npu.*got '{wrong}'.*fresh process"):
        runner._validate_task_queue_environment(mode)

    monkeypatch.setenv(runner.TASK_QUEUE_ENV, expected)
    assert runner._validate_task_queue_environment(mode) == expected

    source = inspect.getsource(runner.run_a3_single_op_benchmark)
    validate = source.index("_validate_task_queue_environment(config.mode)")
    claim = source.index("_claim_process_runtime(config.runtime)")
    import_torch = source.index("\n    import torch\n")
    import_torch_npu = source.index("\n    import torch_npu\n")
    assert validate < claim < import_torch < import_torch_npu


def test_executor_obeys_abba_order_and_keeps_phase_boundary_explicit() -> None:
    report, log, _events = _execute_fake(BenchmarkMode.EAGER)

    enqueues = [entry for entry in log if entry.startswith("enqueue:")]
    assert enqueues[:8] == [
        "enqueue:A",
        "enqueue:B",
        "enqueue:B",
        "enqueue:A",
        "enqueue:A",
        "enqueue:B",
        "enqueue:B",
        "enqueue:A",
    ]
    assert enqueues.index("enqueue:A", 40) == 40
    between_index = log.index("between")
    assert sum(entry.startswith("enqueue:") for entry in log[:between_index]) == 40
    assert sum(entry.startswith("enqueue:") for entry in log[between_index + 1 :]) == 200
    assert report.order is SamplingOrder.ABBA
    assert dict(report.metadata)["source"] == "host-fake"


def test_executor_precreates_bounded_event_pool_and_quiesces_after_batches() -> None:
    _report, log, events = _execute_fake(BenchmarkMode.EAGER, enqueue_batch_size=7)

    # Seven reusable pairs, not one pair per 240 scheduled invocation.
    assert len(events) == 14
    first_prepare = next(index for index, entry in enumerate(log) if entry.startswith("prepare:"))
    # Event construction itself does not log; every event owner nevertheless
    # exists before the first prepare and is reused only after quiescence.
    assert first_prepare == 0
    expected_quiesces = math.ceil(40 / 7) + math.ceil(200 / 7)
    assert log.count("quiesce") == expected_quiesces

    # A requested batch larger than the complete warmup is capped so every
    # lazy runtime event handle is initialized before steady-state sampling.
    _report, large_log, large_events = _execute_fake(BenchmarkMode.EAGER, enqueue_batch_size=100)
    assert len(large_events) == 80
    assert large_log.count("quiesce") == 1 + math.ceil(200 / 40)


def test_diagnostic_profiler_executes_small_abba_rounds_with_cpu_markers_and_steps() -> None:
    native_key, pypto_key = _keys(BenchmarkMode.ACLGRAPH)
    log: list[str] = []
    native = _FakeCase(native_key, "native", log)
    pypto = _FakeCase(pypto_key, "pypto", log)

    runner.execute_profile_abba_rounds(
        native,
        pypto,
        profiler=_FakeProfiler(log),
        record_function=lambda name: _FakeRecordFunction(name, log),
        quiesce=lambda: log.append("quiesce"),
        rounds=2,
    )

    assert log[0] == "profiler:enter"
    assert log[-1] == "profiler:exit"
    assert [entry for entry in log if entry.startswith("enqueue:")] == [
        "enqueue:native",
        "enqueue:pypto",
        "enqueue:pypto",
        "enqueue:native",
    ] * 2
    assert [entry for entry in log if entry.startswith("record:enter:")] == [
        "record:enter:csa_steady_state.native.enqueue",
        "record:enter:csa_steady_state.pypto.enqueue",
        "record:enter:csa_steady_state.pypto.enqueue",
        "record:enter:csa_steady_state.native.enqueue",
    ] * 2
    assert log.count("profiler:step") == 2
    assert log.count("quiesce") == 2
    for step_index in (log.index("profiler:step"), log.index("profiler:step", log.index("profiler:step") + 1)):
        assert log[step_index - 1] == "quiesce"


def test_diagnostic_profiler_rejects_non_profile_backends_and_unbounded_rounds() -> None:
    native_key, pypto_key = _keys(BenchmarkMode.EAGER)
    log: list[str] = []
    native = _FakeCase(native_key, "native", log)
    pypto = _FakeCase(pypto_key, "pypto", log)
    bad_native = _FakeCase(pypto_key, "bad", log)
    bad_pypto = _FakeCase(native_key, "bad", log)

    kwargs = {
        "profiler": _FakeProfiler(log),
        "record_function": lambda name: _FakeRecordFunction(name, log),
        "quiesce": lambda: None,
    }
    with pytest.raises(BenchmarkError, match="profile rounds"):
        runner.execute_profile_abba_rounds(native, pypto, rounds=runner.MAX_PROFILE_ROUNDS + 1, **kwargs)
    with pytest.raises(BenchmarkError, match="native_production"):
        runner.execute_profile_abba_rounds(bad_native, pypto, rounds=1, **kwargs)
    with pytest.raises(BenchmarkError, match="PyPTO backend"):
        runner.execute_profile_abba_rounds(native, bad_pypto, rounds=1, **kwargs)


def test_torch_npu_profiler_configuration_is_cpu_npu_level1_pipe_utilization(tmp_path: Path) -> None:
    calls: dict[str, object] = {}

    def experimental_config(**kwargs: object) -> str:
        calls["experimental_config"] = kwargs
        return "experimental"

    def schedule(**kwargs: object) -> str:
        calls["schedule"] = kwargs
        return "schedule"

    def trace_handler(*args: object, **kwargs: object) -> str:
        calls["trace_handler"] = (args, kwargs)
        return "handler"

    profile_owner = object()

    def profile(**kwargs: object) -> object:
        calls["profile"] = kwargs
        return profile_owner

    profiler_api = SimpleNamespace(
        _ExperimentalConfig=experimental_config,
        ProfilerLevel=SimpleNamespace(Level1="level1"),
        AiCMetrics=SimpleNamespace(PipeUtilization="pipe"),
        ProfilerActivity=SimpleNamespace(CPU="cpu", NPU="npu"),
        schedule=schedule,
        tensorboard_trace_handler=trace_handler,
        profile=profile,
    )

    result = runner._create_torch_npu_profiler(
        SimpleNamespace(profiler=profiler_api),
        tmp_path,
        rounds=3,
    )

    assert result is profile_owner
    assert calls["experimental_config"] == {
        "profiler_level": "level1",
        "aic_metrics": "pipe",
    }
    assert calls["schedule"] == {"wait": 0, "warmup": 0, "active": 3, "repeat": 1, "skip_first": 0}
    assert calls["trace_handler"] == ((str(tmp_path),), {})
    assert calls["profile"] == {
        "activities": ["npu", "cpu"],
        "with_stack": True,
        "record_shapes": False,
        "profile_memory": False,
        "experimental_config": "experimental",
        "schedule": "schedule",
        "on_trace_ready": "handler",
    }


def test_profile_artifact_validation_requires_nonempty_kernel_csv_and_trace_json(tmp_path: Path) -> None:
    run_directory = tmp_path / "fresh-run"
    output_directory = run_directory / "worker" / "ASCEND_PROFILER_OUTPUT"
    output_directory.mkdir(parents=True)
    kernel_csv = output_directory / "kernel_details.csv"
    trace_json = output_directory / "trace_view.json"

    kernel_csv.write_text("Name,Duration\naicore_kernel_0,1\n")
    with pytest.raises(BenchmarkError, match="trace_view.json"):
        runner._require_profile_artifacts(run_directory)

    trace_json.write_text(
        json.dumps(
            [
                {"name": "csa_steady_state.native.enqueue"},
                {"name": "csa_steady_state.pypto.enqueue"},
                {"name": "ProfilerStep#0"},
            ]
        )
    )
    artifacts = runner._require_profile_artifacts(run_directory)
    payload = artifacts.to_dict()
    assert payload["run_directory"] == str(run_directory.resolve())
    assert payload["kernel_details_csv"] == [str(kernel_csv.resolve())]
    assert payload["trace_view_json"] == [str(trace_json.resolve())]

    kernel_csv.write_text("")
    with pytest.raises(BenchmarkError, match="kernel_details.csv"):
        runner._require_profile_artifacts(run_directory)


def test_profile_artifact_validation_rejects_truncated_trace_and_reports_parser_evidence(tmp_path: Path) -> None:
    run_directory = tmp_path / "failed-run"
    output_directory = run_directory / "worker" / "ASCEND_PROFILER_OUTPUT"
    log_directory = run_directory / "worker" / "logs"
    output_directory.mkdir(parents=True)
    log_directory.mkdir()
    (output_directory / "analyse.done").write_text("")
    (output_directory / "trace_view.json").write_text('[{"name":"csa_steady_state.native.enqueue"}')
    (log_directory / "profiler_1_CANNExportParser.log").write_text("export succeeded")

    with pytest.raises(BenchmarkError) as error_info:
        runner._require_profile_artifacts(run_directory)

    message = str(error_info.value)
    assert "kernel_details.csv is missing" in message
    assert "trace_view.json is not valid JSON" in message
    assert "trace_view.json=" in message
    assert "analyse.done=0B" in message
    assert "CANNExportParser.log" in message


def test_optional_profiler_is_a_noop_without_an_explicit_directory() -> None:
    config = runner.A3SingleOpBenchmarkConfig(runtime=runner.TRB_RUNTIME, mode="eager")
    native_key, pypto_key = _keys(BenchmarkMode.EAGER)

    result = runner._run_optional_steady_state_profile(
        config,
        _FakeCase(native_key, "native", []),
        _FakeCase(pypto_key, "pypto", []),
        torch=None,
        torch_npu=None,
        quiesce=lambda: pytest.fail("disabled profiler must not quiesce"),
    )

    assert result is None


def test_fixed_inputs_are_initialized_for_both_eager_cases_before_timing() -> None:
    copied: list[tuple[str, str]] = []

    class Tensor:
        def __init__(self, name: str) -> None:
            self.name = name

        def copy_(self, source: Tensor) -> None:
            copied.append((self.name, source.name))

    keys = _keys(BenchmarkMode.EAGER)
    native = runner._DeviceCase(keys[0], Tensor("native-hidden"), Tensor("source"), lambda: None)
    pypto = runner._DeviceCase(keys[1], Tensor("pypto-hidden"), Tensor("source"), lambda: None)

    runner._initialize_case_inputs(native, pypto)

    assert copied == [("native-hidden", "source"), ("pypto-hidden", "source")]


def test_eager_result_has_host_and_device_percentiles_and_graph_is_na() -> None:
    report, _log, _events = _execute_fake(BenchmarkMode.EAGER)

    assert len(report.cases) == 2
    for case in report.cases:
        host = case.measurement_for(TimingMetric.HOST_ENQUEUE)
        metadata = case.measurement_for(TimingMetric.METADATA_UPDATE)
        device = case.measurement_for(TimingMetric.DEVICE_SPAN)
        graph = case.measurement_for(TimingMetric.GRAPH_REPLAY)
        assert host.statistics is not None
        assert host.statistics.count == 100
        assert host.statistics.p50_ns == 10
        assert host.statistics.p90_ns == 10
        assert host.statistics.p99_ns == 10
        assert metadata.statistics is not None
        assert metadata.statistics.p50_ns == 10
        assert device.statistics is not None
        assert device.statistics.p50_ns == 250_000
        assert graph.statistics is None
        assert graph.reason is not None


def test_aclgraph_reports_graph_replay_as_the_exact_host_replay_enqueue() -> None:
    report, _log, _events = _execute_fake(BenchmarkMode.ACLGRAPH)

    for case in report.cases:
        host = case.measurement_for(TimingMetric.HOST_ENQUEUE)
        replay = case.measurement_for(TimingMetric.GRAPH_REPLAY)
        assert replay.samples_ns == host.samples_ns
        assert replay.warmup_samples_ns == host.warmup_samples_ns
        assert replay.statistics is not None
        assert replay.statistics.p50_ns == 10


@pytest.mark.parametrize(
    ("mode", "task_queue_enable"),
    (
        (BenchmarkMode.EAGER, "2"),
        (BenchmarkMode.ACLGRAPH, "1"),
    ),
)
def test_measurement_metadata_makes_the_complete_steady_state_boundary_auditable(
    mode: BenchmarkMode,
    task_queue_enable: str,
) -> None:
    config = runner.A3SingleOpBenchmarkConfig(
        runtime=runner.TRB_RUNTIME,
        mode=mode,
    )
    correctness = runner.CorrectnessGateResult(
        output={"close": True},
        states={"swa": {"close": True}},
        indexer_quantized={"acceptable": True},
        close=True,
    )

    metadata = runner._build_measurement_metadata(
        config,
        correctness,
        task_queue_enable=task_queue_enable,
        pid=1234,
    )

    assert metadata["pid"] == "1234"
    assert metadata["ordering"] == "same-card same-stream ABBA"
    assert metadata["native_multistream_dsv4_dsa_overlap"] == "true"
    for excluded in runner.STEADY_STATE_EXCLUDED_SETUP:
        assert excluded in metadata["timing_excludes"]
    for included in runner.STEADY_STATE_INCLUDED_REPEATED_PATH:
        assert included in metadata["timing_includes_repeated_path"]
    measurement_contract = json.loads(metadata["measurement_contract"])
    assert measurement_contract["goal"] == "steady_state_only"
    assert measurement_contract["sample_phase"] == "sample"
    assert measurement_contract["compile_in_acceptance"] is False
    assert measurement_contract["warmup_in_acceptance"] is False
    assert measurement_contract["pre_sample_quiesced"] is True
    assert measurement_contract["excluded_setup"] == list(runner.STEADY_STATE_EXCLUDED_SETUP)
    assert measurement_contract["included_repeated_path"] == list(runner.STEADY_STATE_INCLUDED_REPEATED_PATH)
    assert measurement_contract["first_final_binding_primed"] is True
    assert measurement_contract["event_handles_primed_by_benchmark_warmup"] is True
    assert measurement_contract["correctness_gate_completed_before_sampling"] is True
    assert "SamplingPhase.SAMPLE" in metadata["steady_state_statistics"]
    assert "p50/p90/p99" in metadata["steady_state_statistics"]
    assert metadata["task_queue_enable"] == task_queue_enable
    assert metadata["task_queue_validated_before_torch_npu_import"] == "true"
    assert "must not be added" in metadata["device_span_semantics"]
    if mode is BenchmarkMode.EAGER:
        assert measurement_contract["captured_binding"] == "per_call"
        assert measurement_contract["formal_acceptance_supported"] is False
        assert measurement_contract["primary_metric"] == (
            "unavailable: steady_enqueue_batch_critical_path_not_implemented"
        )
        assert "TASK_QUEUE_ENABLE=2" in metadata["task_queue_contract"]
        assert "RunOpApiV2 enqueue" in metadata["task_queue_coverage"]
        assert "consumer dequeue/launch" in metadata["task_queue_coverage"]
        assert "attribution diagnostics" in metadata["task_queue_coverage"]
        assert "metadata filtering/extraction" in metadata["per_call_patch_coverage"]
        assert "address patch" in metadata["per_call_patch_coverage"]
        assert "inside host_enqueue" in metadata["per_call_patch_coverage"]
    else:
        assert measurement_contract["captured_binding"] == "fixed"
        assert measurement_contract["primary_metric"] == "device_span"
        assert measurement_contract["formal_acceptance_supported"] is True
        assert "TASK_QUEUE_ENABLE=1" in metadata["task_queue_contract"]
        assert "graph.replay" in metadata["task_queue_coverage"]
        assert "RunOpApiV2" not in metadata["task_queue_coverage"]
        assert "fixed B4/S8 tensor/scalar snapshot" in metadata["per_call_patch_coverage"]
        assert "no per-replay Python" in metadata["per_call_patch_coverage"]


def test_runner_primes_every_cold_boundary_before_constructing_the_sample_schedule() -> None:
    source = inspect.getsource(runner.run_a3_single_op_benchmark)

    install = source.index("owner = install_pypto_dsv4_decode_csa(")
    native_warmup = source.index("\n                native_warm_call()\n")
    pypto_warmup = source.index("\n                pypto_warm_call()\n")
    capture = source.index("with torch_npu.npu.graph(native_graph, stream=stream):")
    graph_replay_binding = source.index("native_callback = native_graph.replay")
    first_steady_address_invocation = source.index("correctness = _correctness_gate(")
    schedule = source.index("schedule = build_benchmark_schedule(")
    timed_execution = source.index("report = execute_benchmark_schedule(")
    post_timing_correctness = source.index("post_timing_correctness = _compare_current_results(")
    diagnostic_profile = source.index("profile_artifacts = _run_optional_steady_state_profile(")

    assert (
        install
        < native_warmup
        < pypto_warmup
        < capture
        < graph_replay_binding
        < first_steady_address_invocation
        < schedule
        < timed_execution
        < post_timing_correctness
        < diagnostic_profile
    )
    assert source.index("native_case = _DeviceCase(") < first_steady_address_invocation
    assert source.index("pypto_case = _DeviceCase(") < first_steady_address_invocation


def test_diagnostic_profiler_cannot_feed_the_formal_accumulator() -> None:
    formal_source = inspect.getsource(runner.execute_benchmark_schedule)
    diagnostic_source = inspect.getsource(runner.execute_profile_abba_rounds)

    assert "BenchmarkAccumulator" in formal_source
    assert "accumulator.record" in formal_source
    assert "BenchmarkAccumulator" not in diagnostic_source
    assert "accumulator.record" not in diagnostic_source
    assert "profiler.step" in diagnostic_source


def test_executor_rejects_case_mismatch_invalid_clock_and_event_duration() -> None:
    keys = _keys(BenchmarkMode.EAGER)
    schedule = build_benchmark_schedule(keys, order=SamplingOrder.ABBA)
    case = _FakeCase(keys[0], "A", [])
    with pytest.raises(BenchmarkError, match="do not match"):
        runner.execute_benchmark_schedule(
            schedule,
            {keys[0]: case},
            event_factory=lambda: _FakeEvent([], 0),
            quiesce=lambda: None,
            between_phases=lambda: None,
            enqueue_batch_size=1,
        )

    class BackwardsClock:
        def __init__(self) -> None:
            self.value = 2

        def __call__(self) -> int:
            self.value -= 1
            return self.value

    cases = {key: _FakeCase(key, str(index), []) for index, key in enumerate(keys)}
    with pytest.raises(BenchmarkError, match="backwards"):
        runner.execute_benchmark_schedule(
            schedule,
            cases,
            event_factory=lambda: _FakeEvent([], 0),
            quiesce=lambda: None,
            between_phases=lambda: None,
            enqueue_batch_size=1,
            clock_ns=BackwardsClock(),
        )

    class BadEvent(_FakeEvent):
        def elapsed_time(self, end_event: _FakeEvent) -> float:
            return float("nan")

    with pytest.raises(BenchmarkError, match="invalid elapsed"):
        runner.execute_benchmark_schedule(
            schedule,
            cases,
            event_factory=lambda: BadEvent([], 0),
            quiesce=lambda: None,
            between_phases=lambda: None,
            enqueue_batch_size=1,
            clock_ns=_Clock(),
        )


def test_native_production_check_requires_overlap_and_never_mutates_impl() -> None:
    production_impl = SimpleNamespace(multistream_dsv4_dsa_overlap=True)
    production = SimpleNamespace(dsa_attn=SimpleNamespace(impl=production_impl))
    runner._require_native_production(production)
    assert production_impl.multistream_dsv4_dsa_overlap is True

    for value in (False, None):
        impl = SimpleNamespace(multistream_dsv4_dsa_overlap=value)
        layer = SimpleNamespace(dsa_attn=SimpleNamespace(impl=impl))
        with pytest.raises(runner.NativeProductionUnsupported, match="unsupported"):
            runner._require_native_production(layer)
        assert impl.multistream_dsv4_dsa_overlap is value

    with pytest.raises(runner.NativeProductionUnsupported, match="unsupported"):
        runner._require_native_production(SimpleNamespace())


def test_process_runtime_claim_rejects_trb_hbg_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner, "_PROCESS_RUNTIME", None)

    runner._claim_process_runtime(runner.TRB_RUNTIME)
    runner._claim_process_runtime(runner.TRB_RUNTIME)
    with pytest.raises(BenchmarkError, match="fresh processes"):
        runner._claim_process_runtime(runner.HBG_RUNTIME)


@pytest.mark.parametrize("runtime", (runner.TRB_RUNTIME, runner.HBG_RUNTIME))
@pytest.mark.parametrize("mode", tuple(mode.value for mode in BenchmarkMode))
def test_cli_accepts_every_device0_b4_s8_runtime_mode_pair(runtime: str, mode: str) -> None:
    parser = runner.build_parser()

    args = parser.parse_args(["--runtime", runtime, "--mode", mode])
    config = runner._config_from_args(args)

    assert config.runtime == runtime
    assert config.mode.value == mode
    assert config.device == 0
    assert config.batch == 4
    assert config.profile_directory is None
    assert config.profile_rounds == runner.DEFAULT_PROFILE_ROUNDS
    assert runner.SUPPORTED_SEQ == 8


def test_cli_enables_profiler_only_with_an_explicit_directory() -> None:
    parser = runner.build_parser()

    disabled = runner._config_from_args(parser.parse_args(["--runtime", runner.TRB_RUNTIME, "--mode", "aclgraph"]))
    enabled = runner._config_from_args(
        parser.parse_args(
            [
                "--runtime",
                runner.HBG_RUNTIME,
                "--mode",
                "aclgraph",
                "--profile-dir",
                "/tmp/decode-csa-profile",
                "--profile-rounds",
                "7",
            ]
        )
    )

    assert disabled.profile_directory is None
    assert enabled.profile_directory == "/tmp/decode-csa-profile"
    assert enabled.profile_rounds == 7


def test_cli_requires_one_runtime_and_mode_and_rejects_device1_and_non_b4() -> None:
    parser = runner.build_parser()

    args = parser.parse_args(["--runtime", runner.HBG_RUNTIME, "--mode", "aclgraph"])
    config = runner._config_from_args(args)
    assert config.runtime == runner.HBG_RUNTIME
    assert config.mode is BenchmarkMode.ACLGRAPH

    with pytest.raises(SystemExit):
        parser.parse_args(["--mode", "eager"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--runtime", runner.TRB_RUNTIME])
    with pytest.raises(SystemExit):
        parser.parse_args(["--runtime", runner.TRB_RUNTIME, "--mode", "eager", "--device", "1"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--runtime", runner.TRB_RUNTIME, "--mode", "eager", "--batch", "8"])


def test_cli_prints_existing_benchmark_schema_without_writing_a_result_file(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    report, _log, _events = _execute_fake(BenchmarkMode.EAGER)
    observed = []

    def fake_run(config: runner.A3SingleOpBenchmarkConfig):
        observed.append(config)
        return report

    monkeypatch.setattr(runner, "run_a3_single_op_benchmark", fake_run)
    status = runner.main(
        [
            "--runtime",
            runner.TRB_RUNTIME,
            "--mode",
            "eager",
            "--warmups",
            "20",
            "--samples",
            "100",
        ]
    )

    assert status == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_name"] == BENCHMARK_SCHEMA_NAME
    assert payload["schema_version"] == BENCHMARK_SCHEMA_VERSION
    assert payload["sampling"]["order"] == "abba"
    assert len(payload["cases"]) == 2
    assert observed[0].runtime == runner.TRB_RUNTIME
    assert observed[0].device == 0
    assert observed[0].profile_directory is None


def test_cli_marks_unprovable_native_production_as_unsupported(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def unsupported(_config: runner.A3SingleOpBenchmarkConfig):
        raise runner.NativeProductionUnsupported("unsupported: production overlap unavailable")

    monkeypatch.setattr(runner, "run_a3_single_op_benchmark", unsupported)
    status = runner.main(["--runtime", runner.TRB_RUNTIME, "--mode", "eager"])

    assert status == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "batch": 4,
        "device": 0,
        "mode": "eager",
        "reason": "unsupported: production overlap unavailable",
        "runtime": runner.TRB_RUNTIME,
        "schema_name": BENCHMARK_SCHEMA_NAME,
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "sequence_length": 8,
        "status": "unsupported",
    }
