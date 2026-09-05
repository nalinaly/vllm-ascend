# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Pure Host tests for the production decode CSA PyPTO L1 owner."""

from __future__ import annotations

import gc
import subprocess
import sys
import threading
import weakref
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from vllm_ascend.ops._pypto_dsv4_csa.backend import (
    EXPECTED_ABI_PARAMETER_COUNT,
    EXPECTED_OUTPUT_INDICES,
    EXPECTED_PARAMETER_NAMES,
    EXPECTED_TENSOR_COUNT,
    HBG_RUNTIME,
    TRB_RUNTIME,
    DeviceRuntimeOwnerRegistry,
    PyPTOBackendContractError,
    PyPTOBackendDependencies,
    PyPTOBackendState,
    PyPTODSABackend,
    TensorBindingMetadata,
    _default_create_context,
)
from vllm_ascend.ops._pypto_dsv4_csa.contract import (
    MUTABLE_ARGUMENT_NAMES,
    OUTPUT_ARGUMENT_NAME,
    DecodeCSAProgramSpec,
)


@dataclass(frozen=True)
class FakeParamInfo:
    name: str
    shape: tuple[int, ...] | None
    dtype: str
    direction: object


class FakeCompiled:
    def __init__(
        self,
        *,
        runtime: str,
        param_infos: list[FakeParamInfo] | None = None,
        param_names: tuple[str, ...] = EXPECTED_PARAMETER_NAMES,
        output_indices: tuple[int, ...] = EXPECTED_OUTPUT_INDICES,
        platform: str = "a2a3",
        enable_sdma: bool = False,
    ) -> None:
        self.param_names = list(param_names)
        self.output_indices = list(output_indices)
        self.runtime_name = runtime
        self.platform = platform
        self.runtime_config = {"enable_sdma": enable_sdma}
        self._param_infos = param_infos or _valid_param_infos()

    def _get_metadata(self):
        return self._param_infos, list(self.output_indices), []


class FakeTensor:
    def __init__(
        self,
        name: str,
        *,
        shape: tuple[int, ...] = (1,),
        dtype: str = "fp32",
        strides: tuple[int, ...] = (1,),
        device: int = 0,
    ) -> None:
        self.name = name
        self.metadata = TensorBindingMetadata(
            shape=shape,
            dtype=dtype,
            strides=strides,
            device_type="npu",
            device_index=device,
        )

    def contiguous(self):
        raise AssertionError("backend launch must not materialize a contiguous tensor")

    def copy_(self, _value):
        raise AssertionError("backend launch must not copy tensor data")


class FakePreparedCaches:
    def __init__(
        self,
        spec: DecodeCSAProgramSpec,
        launch_arguments: dict[str, object],
    ) -> None:
        self.spec = spec
        self._launch_arguments = launch_arguments

    def for_launch(self) -> dict[str, object]:
        return self._launch_arguments


class FakeOperator:
    def __init__(self) -> None:
        self.warmup_calls: list[tuple[tuple[object, ...], object]] = []
        self.launch_calls: list[tuple[tuple[object, ...], object]] = []
        self.fail_warmup = False
        self.fail_launch = False
        self.wrong_return = False

    def warmup(self, *args: object, out: object) -> object:
        self.warmup_calls.append((args, out))
        if self.fail_warmup:
            raise RuntimeError("warmup enqueue failed")
        return object() if self.wrong_return else out

    def __call__(self, *args: object, out: object) -> object:
        self.launch_calls.append((args, out))
        if self.fail_launch:
            raise RuntimeError("launch enqueue failed")
        return object() if self.wrong_return else out


class FakeContext:
    def __init__(
        self,
        programs: tuple[object, ...],
        *,
        fail_prepare: bool = False,
        close_failures: int = 0,
    ) -> None:
        self.programs = programs
        self.operators = {id(program): FakeOperator() for program in programs}
        self.prepare_calls = 0
        self.close_calls = 0
        self.fail_prepare = fail_prepare
        self.close_failures = close_failures

    def operator(self, program: object) -> FakeOperator:
        return self.operators[id(program)]

    def prepare(self) -> None:
        self.prepare_calls += 1
        if self.fail_prepare:
            raise RuntimeError("prepare failed")

    def close(self) -> None:
        self.close_calls += 1
        if self.close_failures:
            self.close_failures -= 1
            raise RuntimeError("close failed")


class FakeEnvironment:
    def __init__(
        self,
        *,
        current_device: int = 0,
        owner_registry: DeviceRuntimeOwnerRegistry | None = None,
        compiled_factory=None,
        fail_prepare: bool = False,
        close_failures: int = 0,
    ) -> None:
        self.current_device = current_device
        self.owner_registry = owner_registry or DeviceRuntimeOwnerRegistry()
        self.compiled_factory = compiled_factory
        self.fail_prepare = fail_prepare
        self.close_failures = close_failures
        self.make_calls: list[tuple[DecodeCSAProgramSpec, str]] = []
        self.compile_calls: list[tuple[object, int, str]] = []
        self.context_calls: list[tuple[tuple[object, ...], int, bool]] = []
        self.contexts: list[FakeContext] = []
        self.allow_tensor_inspection = True
        self.allow_compile = True

    def make_program(self, spec: DecodeCSAProgramSpec, runtime: str) -> object:
        self.make_calls.append((spec, runtime))
        return SimpleNamespace(spec=spec, runtime=runtime)

    def compile_program(self, program: object, device: int, runtime: str) -> FakeCompiled:
        if not self.allow_compile:
            raise AssertionError("launch must not compile")
        self.compile_calls.append((program, device, runtime))
        if self.compiled_factory is not None:
            return self.compiled_factory(runtime)
        return FakeCompiled(runtime=runtime)

    def create_context(
        self,
        programs: tuple[object, ...],
        device: int,
        use_task_queue: bool,
    ) -> FakeContext:
        values = tuple(programs)
        self.context_calls.append((values, device, use_task_queue))
        context = FakeContext(
            values,
            fail_prepare=self.fail_prepare,
            close_failures=self.close_failures,
        )
        self.contexts.append(context)
        return context

    def inspect_tensor(self, value: object) -> TensorBindingMetadata:
        if not self.allow_tensor_inspection:
            raise AssertionError("launch must not inspect or rebuild tensor metadata")
        if not isinstance(value, FakeTensor):
            raise TypeError(f"expected FakeTensor, got {type(value).__name__}")
        return value.metadata

    def dependencies(self) -> PyPTOBackendDependencies:
        return PyPTOBackendDependencies(
            make_program=self.make_program,
            compile_program=self.compile_program,
            create_context=self.create_context,
            current_device=lambda: self.current_device,
            inspect_tensor=self.inspect_tensor,
            owner_registry=self.owner_registry,
        )


def _valid_param_infos() -> list[FakeParamInfo]:
    infos: list[FakeParamInfo] = []
    for index, name in enumerate(EXPECTED_PARAMETER_NAMES):
        if index >= EXPECTED_TENSOR_COUNT:
            shape = None
            dtype = "index"
        else:
            shape = (1,)
            dtype = "fp32"
        direction = "Out" if name == OUTPUT_ARGUMENT_NAME else "InOut" if name in MUTABLE_ARGUMENT_NAMES else "In"
        infos.append(
            FakeParamInfo(
                name=name,
                shape=shape,
                dtype=dtype,
                direction=SimpleNamespace(name=direction),
            )
        )
    return infos


def _binding_inputs(
    spec: DecodeCSAProgramSpec,
    *,
    address_tag: str = "a",
    device: int = 0,
) -> tuple[dict[str, object], FakePreparedCaches]:
    arguments: dict[str, object] = {}
    scalar_values = {
        "main_state_page_stride": spec.physical_layout.main_state_strides[0],
        "inner_state_page_stride": spec.physical_layout.inner_state_strides[0],
        "indexer_k_page_stride": spec.physical_layout.indexer_k_strides[0],
        "indexer_scale_page_stride": spec.physical_layout.indexer_scale_strides[0],
    }
    for index, name in enumerate(EXPECTED_PARAMETER_NAMES):
        if index < EXPECTED_TENSOR_COUNT:
            arguments[name] = FakeTensor(f"{name}-{address_tag}", device=device)
        else:
            arguments[name] = scalar_values[name]
    prepared_mapping = {
        name: arguments[name] for name in (*tuple(MUTABLE_ARGUMENT_NAMES), *EXPECTED_PARAMETER_NAMES[-4:])
    }
    return arguments, FakePreparedCaches(spec, prepared_mapping)


def _ready_backend(
    *,
    runtime: str = "tensormap_and_ringbuffer",
    environment: FakeEnvironment | None = None,
) -> tuple[
    PyPTODSABackend,
    FakeEnvironment,
    DecodeCSAProgramSpec,
    object,
]:
    env = environment or FakeEnvironment()
    spec = DecodeCSAProgramSpec(batch=4)
    backend = PyPTODSABackend(
        device=0,
        runtime=runtime,
        dependencies=env.dependencies(),
    )
    compiled = backend.compile(spec)
    return backend, env, spec, compiled


def test_importing_backend_does_not_import_pypto() -> None:
    repository = Path(__file__).resolve().parents[2]
    code = """
import sys
before = set(sys.modules)
import vllm_ascend.ops._pypto_dsv4_csa.backend
new = set(sys.modules) - before
assert not any(name == 'pypto' or name.startswith('pypto.') for name in new), sorted(new)
"""
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=repository,
        check=True,
    )


def test_default_context_uses_compact_hbg_ring_without_changing_trb(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    class FakeL1Config:
        def __init__(self, **kwargs: object) -> None:
            self.ring_task_window = kwargs.get("ring_task_window")
            self.use_task_queue = kwargs.get("use_task_queue")

    def fake_pypto_init(**kwargs: object) -> object:
        calls.append(kwargs)
        return kwargs

    fake_pypto = ModuleType("pypto")
    fake_pypto.__path__ = []  # type: ignore[attr-defined]
    fake_runtime = ModuleType("pypto.runtime")
    fake_runtime.__path__ = []  # type: ignore[attr-defined]
    fake_l1 = ModuleType("pypto.runtime.l1")
    fake_l1.L1Config = FakeL1Config  # type: ignore[attr-defined]
    fake_l1.pypto_init = fake_pypto_init  # type: ignore[attr-defined]
    fake_pypto.runtime = fake_runtime  # type: ignore[attr-defined]
    fake_runtime.l1 = fake_l1  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pypto", fake_pypto)
    monkeypatch.setitem(sys.modules, "pypto.runtime", fake_runtime)
    monkeypatch.setitem(sys.modules, "pypto.runtime.l1", fake_l1)
    hbg = _default_create_context(
        (SimpleNamespace(runtime_name=HBG_RUNTIME),),
        0,
        True,
    )
    trb = _default_create_context(
        (SimpleNamespace(runtime_name=TRB_RUNTIME),),
        0,
        True,
    )

    assert hbg is calls[0]
    assert calls[0]["config"].ring_task_window == 128
    assert calls[0]["config"].use_task_queue is True
    assert trb is calls[1]
    assert calls[1]["config"].ring_task_window is None


def test_compile_is_annotation_only_idempotent_and_registry_is_static() -> None:
    backend, env, spec, compiled = _ready_backend()

    assert backend.compile(spec) is compiled
    assert "swa_slot_mapping" not in EXPECTED_PARAMETER_NAMES
    assert EXPECTED_PARAMETER_NAMES.index("swa_block_table") == 33
    assert EXPECTED_PARAMETER_NAMES.index("start_positions") == 34
    assert EXPECTED_PARAMETER_NAMES.index("kv_seq_lens") == 35
    assert EXPECTED_PARAMETER_NAMES.index(OUTPUT_ARGUMENT_NAME) == 39
    assert EXPECTED_OUTPUT_INDICES == (39,)
    assert EXPECTED_TENSOR_COUNT == 40
    assert EXPECTED_ABI_PARAMETER_COUNT == 44
    assert len(env.make_calls) == 1
    # The injected ABI deliberately has no sample-argument slot.  This proves
    # the backend calls compile(program, device, runtime), not compile(*tensors).
    assert env.compile_calls == [(env.compile_calls[0][0], 0, backend.runtime)]
    assert backend.registered_specs == (spec,)
    assert backend.state is PyPTOBackendState.COMPILED


@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        ("parameter_order", "parameter order"),
        ("metadata_count", "metadata requires 44"),
        ("output", "attn_out"),
        ("dynamic_shape", "positive static"),
        ("direction", "requires direction"),
        ("scalar_dtype", "must use index dtype"),
        ("runtime", "runtime mismatch"),
        ("platform", "only a2a3"),
        ("sdma", "SDMA"),
    ),
)
def test_compile_fails_fast_on_invalid_44_slot_metadata(
    mutation: str,
    match: str,
) -> None:
    def factory(runtime: str) -> FakeCompiled:
        infos = _valid_param_infos()
        names = EXPECTED_PARAMETER_NAMES
        outputs = EXPECTED_OUTPUT_INDICES
        platform = "a2a3"
        enable_sdma = False
        artifact_runtime = runtime
        if mutation == "parameter_order":
            names = (*EXPECTED_PARAMETER_NAMES[:-2], EXPECTED_PARAMETER_NAMES[-1])
        elif mutation == "metadata_count":
            infos = infos[:-1]
        elif mutation == "output":
            outputs = (EXPECTED_OUTPUT_INDICES[0] - 1,)
        elif mutation == "dynamic_shape":
            infos[0] = FakeParamInfo("hidden_states", (-1,), "fp32", SimpleNamespace(name="In"))
        elif mutation == "direction":
            index = EXPECTED_PARAMETER_NAMES.index("compress_state")
            infos[index] = FakeParamInfo("compress_state", (1,), "fp32", SimpleNamespace(name="In"))
        elif mutation == "scalar_dtype":
            infos[EXPECTED_TENSOR_COUNT] = FakeParamInfo(
                "main_state_page_stride",
                None,
                "int32",
                SimpleNamespace(name="In"),
            )
        elif mutation == "runtime":
            artifact_runtime = "host_build_graph"
        elif mutation == "platform":
            platform = "a5"
        elif mutation == "sdma":
            enable_sdma = True
        return FakeCompiled(
            runtime=artifact_runtime,
            param_infos=infos,
            param_names=names,
            output_indices=outputs,
            platform=platform,
            enable_sdma=enable_sdma,
        )

    env = FakeEnvironment(compiled_factory=factory)
    backend = PyPTODSABackend(device=0, dependencies=env.dependencies())

    with pytest.raises(PyPTOBackendContractError, match=match):
        backend.compile(DecodeCSAProgramSpec(batch=4))
    assert backend.state is PyPTOBackendState.CREATED


def test_prepare_uses_default_task_queue_and_seals_registry() -> None:
    env = FakeEnvironment()
    backend = PyPTODSABackend(device=0, dependencies=env.dependencies())
    first = DecodeCSAProgramSpec(batch=4)
    second = DecodeCSAProgramSpec(batch=8)
    first_compiled = backend.compile(first)
    second_compiled = backend.compile(second)

    backend.prepare()
    backend.prepare()

    assert env.context_calls == [((first_compiled, second_compiled), 0, True)]
    assert env.contexts[0].prepare_calls == 1
    assert backend.state is PyPTOBackendState.PREPARED
    with pytest.raises(RuntimeError, match="sealed"):
        backend.compile(DecodeCSAProgramSpec(batch=12))


def test_bind_orders_by_compiled_metadata_and_strongly_retains_owners() -> None:
    backend, _, spec, _ = _ready_backend()
    arguments, prepared = _binding_inputs(spec)
    external_owner = object()

    binding = backend.bind(
        spec,
        arguments,
        prepared_caches=prepared,
        owners=(external_owner,),
    )

    assert binding.ordered_arguments == tuple(arguments[name] for name in EXPECTED_PARAMETER_NAMES)
    assert len(binding.input_arguments) == len(EXPECTED_PARAMETER_NAMES) - 1
    assert binding.output_argument is arguments[OUTPUT_ARGUMENT_NAME]
    assert binding.prepared_caches is prepared
    assert binding.strong_owners == (arguments, prepared, external_owner)
    assert backend._bindings[0] is binding


def test_transient_eager_binding_does_not_pin_fresh_output_until_close() -> None:
    backend, _, spec, _ = _ready_backend()
    arguments, prepared = _binding_inputs(spec)

    binding = backend.bind(
        spec,
        arguments,
        prepared_caches=prepared,
        retain=False,
    )

    assert binding.output_argument is arguments[OUTPUT_ARGUMENT_NAME]
    assert backend._bindings == []


def test_bind_rejects_missing_extra_conflicting_cache_and_wrong_spec() -> None:
    backend, _, spec, _ = _ready_backend()
    arguments, prepared = _binding_inputs(spec)

    missing = dict(arguments)
    del missing["wq_a"]
    with pytest.raises(PyPTOBackendContractError, match="missing"):
        backend.bind(spec, missing, prepared_caches=prepared)

    extra = dict(arguments)
    extra["unexpected"] = object()
    with pytest.raises(PyPTOBackendContractError, match="extras"):
        backend.bind(spec, extra, prepared_caches=prepared)

    conflict = dict(arguments)
    conflict["compress_state"] = FakeTensor("different")
    with pytest.raises(PyPTOBackendContractError, match="conflicts"):
        backend.bind(spec, conflict, prepared_caches=prepared)

    other_spec = DecodeCSAProgramSpec(batch=8)
    wrong_prepared = FakePreparedCaches(other_spec, prepared.for_launch())
    with pytest.raises(PyPTOBackendContractError, match="spec does not match"):
        backend.bind(spec, arguments, prepared_caches=wrong_prepared)

    bad_scalar_arguments, bad_scalar_prepared = _binding_inputs(spec)
    bad_scalar_arguments["main_state_page_stride"] = 4096
    bad_scalar_prepared._launch_arguments["main_state_page_stride"] = 4096
    with pytest.raises(PyPTOBackendContractError, match="static physical layout"):
        backend.bind(spec, bad_scalar_arguments, prepared_caches=bad_scalar_prepared)


def test_bind_allows_address_change_but_rejects_layout_or_device_change() -> None:
    backend, env, spec, _ = _ready_backend()
    first_args, first_prepared = _binding_inputs(spec, address_tag="first")
    second_args, second_prepared = _binding_inputs(spec, address_tag="second")

    first = backend.bind(spec, first_args, prepared_caches=first_prepared)
    second = backend.bind(spec, second_args, prepared_caches=second_prepared)
    assert first.ordered_arguments[0] is not second.ordered_arguments[0]

    bad_layout_args, bad_layout_prepared = _binding_inputs(spec, address_tag="stride")
    bad_layout_args["hidden_states"] = FakeTensor("hidden-stride", strides=(2,))
    with pytest.raises(PyPTOBackendContractError, match="canonical contiguous strides"):
        backend.bind(spec, bad_layout_args, prepared_caches=bad_layout_prepared)

    other_device_args, other_device_prepared = _binding_inputs(spec, device=1)
    with pytest.raises(PyPTOBackendContractError, match="must be on npu:0"):
        backend.bind(spec, other_device_args, prepared_caches=other_device_prepared)
    assert env.current_device == 0


def test_first_binding_rejects_noncanonical_tensor_instead_of_pinning_bad_layout() -> None:
    backend, _, spec, _ = _ready_backend()
    arguments, prepared = _binding_inputs(spec)
    arguments["hidden_states"] = FakeTensor("hidden-stride", strides=(2,))

    with pytest.raises(PyPTOBackendContractError, match="canonical contiguous strides"):
        backend.bind(spec, arguments, prepared_caches=prepared)

    assert backend._record_for(spec).bound_tensor_layout is None


def test_explicit_prepare_warmup_launch_uses_preordered_snapshot() -> None:
    backend, env, spec, compiled = _ready_backend()
    arguments, prepared = _binding_inputs(spec)
    binding = backend.bind(spec, arguments, prepared_caches=prepared)

    with pytest.raises(RuntimeError, match="prepare"):
        backend.launch(binding)
    backend.prepare()
    with pytest.raises(RuntimeError, match="not warmed"):
        backend.launch(binding)

    assert backend.warmup(binding) is binding.output_argument
    assert backend.launch(binding) is binding.output_argument
    operator = env.contexts[0].operators[id(compiled)]
    assert operator.warmup_calls == [(binding.input_arguments, binding.output_argument)]
    assert operator.launch_calls == [(binding.input_arguments, binding.output_argument)]
    assert backend.state is PyPTOBackendState.WARMED


def test_launch_performs_no_compile_rebind_copy_or_sync_work() -> None:
    backend, env, spec, _ = _ready_backend()
    arguments, prepared = _binding_inputs(spec)
    binding = backend.bind(spec, arguments, prepared_caches=prepared)
    backend.prepare()
    backend.warmup(binding)

    env.allow_compile = False
    env.allow_tensor_inspection = False
    assert backend.launch(binding) is binding.output_argument


def test_backend_retains_binding_after_caller_drops_local_references() -> None:
    backend, _, spec, _ = _ready_backend()
    arguments, prepared = _binding_inputs(spec)
    output_ref = weakref.ref(arguments[OUTPUT_ARGUMENT_NAME])
    binding = backend.bind(spec, arguments, prepared_caches=prepared)

    del arguments, prepared, binding
    gc.collect()

    assert output_ref() is not None
    assert len(backend._bindings) == 1


def test_wrong_thread_and_wrong_current_device_fail_before_enqueue() -> None:
    backend, env, spec, compiled = _ready_backend()
    arguments, prepared = _binding_inputs(spec)
    binding = backend.bind(spec, arguments, prepared_caches=prepared)
    backend.prepare()
    backend.warmup(binding)
    operator = env.contexts[0].operators[id(compiled)]

    errors: list[BaseException] = []

    def invoke_from_other_thread() -> None:
        try:
            backend.launch(binding)
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=invoke_from_other_thread)
    thread.start()
    thread.join()
    assert len(errors) == 1
    assert "thread-affine" in str(errors[0])

    env.current_device = 1
    with pytest.raises(RuntimeError, match="current device is 1"):
        backend.launch(binding)
    assert len(operator.launch_calls) == 0


def test_prepare_failure_retains_owner_for_explicit_close() -> None:
    env = FakeEnvironment(fail_prepare=True)
    backend, _, _, _ = _ready_backend(environment=env)

    with pytest.raises(RuntimeError, match="prepare failed"):
        backend.prepare()

    assert backend.state is PyPTOBackendState.POISONED
    assert len(env.contexts) == 1
    backend.close()
    assert env.contexts[0].close_calls == 1
    assert backend.state is PyPTOBackendState.CLOSED


def test_close_is_idempotent_and_failure_keeps_retryable_owner_and_bindings() -> None:
    env = FakeEnvironment(close_failures=1)
    backend, _, spec, _ = _ready_backend(environment=env)
    arguments, prepared = _binding_inputs(spec)
    binding = backend.bind(spec, arguments, prepared_caches=prepared)
    backend.prepare()

    with pytest.raises(RuntimeError, match="close failed"):
        backend.close()
    assert backend.state is PyPTOBackendState.POISONED
    assert backend._bindings == [binding]

    backend.close()
    backend.close()
    assert env.contexts[0].close_calls == 2
    assert backend.closed
    assert backend._bindings == []


def test_context_init_cleanup_owner_is_retained_for_close_retry() -> None:
    cleanup_context = FakeContext(())

    class InitializationFailure(RuntimeError):
        def __init__(self) -> None:
            super().__init__("init retained owner")
            self.cleanup_context = cleanup_context

    env = FakeEnvironment()

    def fail_context(_programs, _device, _use_task_queue):
        raise InitializationFailure

    dependencies = PyPTOBackendDependencies(
        make_program=env.make_program,
        compile_program=env.compile_program,
        create_context=fail_context,
        current_device=lambda: env.current_device,
        inspect_tensor=env.inspect_tensor,
        owner_registry=env.owner_registry,
    )
    backend = PyPTODSABackend(device=0, dependencies=dependencies)
    backend.compile(DecodeCSAProgramSpec(batch=4))

    with pytest.raises(InitializationFailure):
        backend.prepare()
    assert backend.state is PyPTOBackendState.POISONED
    backend.close()
    assert cleanup_context.close_calls == 1


def test_runtime_claim_is_process_pinned_and_trb_hbg_cannot_mix() -> None:
    registry = DeviceRuntimeOwnerRegistry()
    trb_env = FakeEnvironment(owner_registry=registry)
    hbg_env = FakeEnvironment(owner_registry=registry)
    trb, _, _, _ = _ready_backend(environment=trb_env)
    hbg, _, _, _ = _ready_backend(runtime="host_build_graph", environment=hbg_env)

    trb.prepare()
    trb.close()

    with pytest.raises(RuntimeError, match="TRB/HBG"):
        hbg.prepare()
    assert hbg_env.context_calls == []


def test_operator_enqueue_failure_poisoning_prevents_owner_replacement() -> None:
    backend, env, spec, compiled = _ready_backend()
    arguments, prepared = _binding_inputs(spec)
    binding = backend.bind(spec, arguments, prepared_caches=prepared)
    backend.prepare()
    operator = env.contexts[0].operators[id(compiled)]
    operator.fail_warmup = True

    with pytest.raises(RuntimeError, match="warmup enqueue failed"):
        backend.warmup(binding)
    assert backend.state is PyPTOBackendState.POISONED
    with pytest.raises(RuntimeError, match="only close"):
        backend.launch(binding)
    assert len(env.context_calls) == 1
