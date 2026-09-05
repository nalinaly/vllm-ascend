# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Pure Host contract tests for production decode-CSA dispatch integration."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

import vllm_ascend.ops._pypto_dsv4_csa.dispatch as dispatch_module
from vllm_ascend.ops._pypto_dsv4_csa.contract import DecodeCSAProgramSpec
from vllm_ascend.ops._pypto_dsv4_csa.dispatch import (
    DecodeCSADeviceOwner,
    DecodeCSADeviceOwnerRegistry,
    DecodeCSADispatchError,
    DecodeCSALayerOwner,
    install_pypto_dsv4_decode_csa,
    uninstall_pypto_dsv4_decode_csa,
)


class FakeBackend:
    """Small recorder that deliberately performs no Torch or device work."""

    def __init__(self, *, device: int, runtime: str, dependencies: object) -> None:
        self.device = device
        self.runtime = runtime
        self.dependencies = dependencies
        self.events: list[tuple[object, ...]] = []
        self.compiled_specs: list[DecodeCSAProgramSpec] = []
        self.prepare_calls = 0
        self.bind_calls: list[dict[str, object]] = []
        self.warmup_calls: list[object] = []
        self.launch_calls: list[object] = []
        self.close_calls = 0

    def compile(self, spec: DecodeCSAProgramSpec) -> object:
        self.events.append(("compile", spec.batch))
        self.compiled_specs.append(spec)
        return SimpleNamespace(spec=spec)

    def prepare(self) -> None:
        self.events.append(("prepare",))
        self.prepare_calls += 1

    def bind(
        self,
        spec: DecodeCSAProgramSpec,
        arguments: dict[str, object],
        *,
        prepared_caches: object,
        owners: tuple[object, ...],
        retain: bool,
    ) -> object:
        record = {
            "spec": spec,
            "arguments": arguments,
            "prepared_caches": prepared_caches,
            "owners": owners,
            "retain": retain,
        }
        self.bind_calls.append(record)
        return record

    def warmup(self, binding: object) -> object:
        self.warmup_calls.append(binding)
        return "warmup-result"

    def launch(self, binding: object) -> object:
        self.launch_calls.append(binding)
        return "launch-result"

    def close(self) -> None:
        self.close_calls += 1


class FakeBackendFactory:
    def __init__(self) -> None:
        self.instances: list[FakeBackend] = []

    def __call__(self, *, device: int, runtime: str, dependencies: object) -> FakeBackend:
        backend = FakeBackend(device=device, runtime=runtime, dependencies=dependencies)
        self.instances.append(backend)
        return backend


@dataclass(frozen=True)
class FakeWeights:
    arguments: dict[str, object]

    def for_launch(self) -> dict[str, object]:
        return self.arguments


@dataclass(frozen=True)
class FakeNativeMetadata:
    arguments: dict[str, object]

    def for_launch(self) -> dict[str, object]:
        return self.arguments


@dataclass(frozen=True)
class FakePreparedCaches:
    arguments: dict[str, object]

    def for_launch(self) -> dict[str, object]:
        return self.arguments


def _spec(*, batch: int = 4, swa_blocks: int = 8) -> DecodeCSAProgramSpec:
    return DecodeCSAProgramSpec(
        batch=batch,
        swa_blocks=swa_blocks,
        compressed_blocks=8,
        main_state_blocks=8,
        inner_state_blocks=8,
        indexer_blocks=8,
        swa_table_width=8,
        compressed_table_width=8,
        main_state_table_width=8,
        inner_state_table_width=8,
        indexer_table_width=8,
    )


def _cache_tuple() -> tuple[torch.Tensor, ...]:
    return tuple(torch.empty(index + 1) for index in range(6))


def _layer_owner(
    monkeypatch: pytest.MonkeyPatch,
    *,
    backend: FakeBackend | None = None,
    caches: tuple[torch.Tensor, ...] | None = None,
) -> tuple[DecodeCSALayerOwner, FakeBackend, tuple[torch.Tensor, ...]]:
    spec = _spec()
    target_backend = backend or FakeBackend(device=0, runtime="trb", dependencies=None)
    device_owner = DecodeCSADeviceOwner(
        device=0,
        runtime="trb",
        backend=target_backend,  # type: ignore[arg-type]
        specs={spec.batch: spec},
        family_key=spec.key[1:],
    )
    cache_tuple = caches or _cache_tuple()
    prepared_caches = FakePreparedCaches({"cache_argument": cache_tuple[0]})
    owner = DecodeCSALayerOwner(
        layer=SimpleNamespace(),
        device_owner=device_owner,
        weights=FakeWeights({"weight": "layer-weight"}),  # type: ignore[arg-type]
        prepared_caches={spec.batch: prepared_caches},  # type: ignore[dict-item]
        cache_key=dispatch_module._cache_storage_key(cache_tuple),
    )
    monkeypatch.setattr(DecodeCSALayerOwner, "_require_io", lambda _self, _hidden, _output: spec.batch)
    monkeypatch.setattr(
        dispatch_module,
        "bind_decode_csa_native_metadata",
        lambda metadata, actual_spec, *, device: FakeNativeMetadata(
            {
                "metadata": metadata,
                "metadata_spec": actual_spec,
                "metadata_device": device,
            }
        ),
    )
    return owner, target_backend, cache_tuple


def _invoke(
    owner: DecodeCSALayerOwner,
    caches: tuple[torch.Tensor, ...],
    *,
    capturing: bool,
) -> object:
    hidden = torch.empty((32, 4), dtype=torch.bfloat16)
    output = torch.empty_like(hidden)
    return owner(
        forward_context=SimpleNamespace(capturing=capturing),
        capturing=capturing,
        hidden_states=hidden,
        need_gather_q_kv=False,
        output=output,
        kv_cache=caches,
        attn_metadata=("native-metadata",),
    )


def test_registry_compiles_every_bucket_then_prepares_once_and_shares_owner() -> None:
    factory = FakeBackendFactory()
    registry = DecodeCSADeviceOwnerRegistry(backend_factory=factory)  # type: ignore[arg-type]
    dependencies = object()

    first = registry.get_or_create(
        device=0,
        runtime="trb",
        base_spec=_spec(),
        batch_buckets=(4, 8, 4, 12),
        backend_dependencies=dependencies,  # type: ignore[arg-type]
    )
    second = registry.get_or_create(
        device=0,
        runtime="trb",
        base_spec=_spec(),
        batch_buckets=(12, 8),
    )

    assert first is second
    assert len(factory.instances) == 1
    backend = factory.instances[0]
    assert backend.dependencies is dependencies
    assert [spec.batch for spec in backend.compiled_specs] == [4, 8, 12]
    assert backend.events == [("compile", 4), ("compile", 8), ("compile", 12), ("prepare",)]
    assert backend.prepare_calls == 1
    assert tuple(first.specs) == (4, 8, 12)


def test_prepared_registry_rejects_late_bucket_and_incompatible_family() -> None:
    factory = FakeBackendFactory()
    registry = DecodeCSADeviceOwnerRegistry(backend_factory=factory)  # type: ignore[arg-type]
    registry.get_or_create(device=0, runtime="trb", base_spec=_spec(), batch_buckets=(4, 8))

    with pytest.raises(DecodeCSADispatchError, match="cannot add batch buckets"):
        registry.get_or_create(device=0, runtime="trb", base_spec=_spec(), batch_buckets=(4, 12))
    with pytest.raises(DecodeCSADispatchError, match="different decode CSA cache/layout family"):
        registry.get_or_create(
            device=0,
            runtime="trb",
            base_spec=_spec(swa_blocks=9),
            batch_buckets=(4, 8),
        )

    assert len(factory.instances) == 1
    assert factory.instances[0].prepare_calls == 1


def test_registry_retains_cleanup_only_owner_after_prepare_failure() -> None:
    class FailingPrepareBackend(FakeBackend):
        def prepare(self) -> None:
            self.events.append(("prepare-failed",))
            raise RuntimeError("partial context initialization failed")

    class FailingFactory:
        def __init__(self) -> None:
            self.backend: FailingPrepareBackend | None = None

        def __call__(self, *, device: int, runtime: str, dependencies: object) -> FailingPrepareBackend:
            self.backend = FailingPrepareBackend(device=device, runtime=runtime, dependencies=dependencies)
            return self.backend

    factory = FailingFactory()
    registry = DecodeCSADeviceOwnerRegistry(backend_factory=factory)  # type: ignore[arg-type]

    with pytest.raises(RuntimeError, match="partial context initialization failed") as raised:
        registry.get_or_create(device=0, runtime="trb", base_spec=_spec(), batch_buckets=(4,))

    retained = registry.get(device=0, runtime="trb")
    assert retained is not None
    assert retained.cleanup_only
    assert raised.value.decode_csa_cleanup_owner is retained  # type: ignore[attr-defined]
    with pytest.raises(DecodeCSADispatchError, match="cleanup-only"):
        registry.get_or_create(device=0, runtime="trb", base_spec=_spec(), batch_buckets=(4,))

    retained.close()
    assert factory.backend is not None
    assert factory.backend.close_calls == 1


def test_first_eager_call_warms_with_transient_binding_then_launches(monkeypatch: pytest.MonkeyPatch) -> None:
    owner, backend, caches = _layer_owner(monkeypatch)

    assert _invoke(owner, caches, capturing=False) == "warmup-result"
    assert len(backend.bind_calls) == 1
    assert backend.bind_calls[0]["retain"] is False
    assert backend.bind_calls[0]["prepared_caches"] is owner.prepared_caches[4]
    assert backend.bind_calls[0]["arguments"]["weight"] == "layer-weight"  # type: ignore[index]
    assert len(backend.warmup_calls) == 1
    assert backend.launch_calls == []

    assert _invoke(owner, caches, capturing=False) == "launch-result"
    assert [call["retain"] for call in backend.bind_calls] == [False, False]
    assert len(backend.warmup_calls) == 1
    assert len(backend.launch_calls) == 1


def test_capture_requires_eager_warmup_and_explicit_quiescence(monkeypatch: pytest.MonkeyPatch) -> None:
    owner, backend, caches = _layer_owner(monkeypatch)

    with pytest.raises(DecodeCSADispatchError, match="ordinary-eager warmup"):
        _invoke(owner, caches, capturing=True)
    assert backend.bind_calls == []

    _invoke(owner, caches, capturing=False)
    with pytest.raises(DecodeCSADispatchError, match="externally quiesced"):
        _invoke(owner, caches, capturing=True)
    assert len(backend.bind_calls) == 1

    owner.device_owner.mark_warmups_quiesced()
    assert _invoke(owner, caches, capturing=True) == "launch-result"
    assert [call["retain"] for call in backend.bind_calls] == [False, True]
    assert len(backend.launch_calls) == 1


def test_capture_gate_runs_before_hot_metadata_binding(monkeypatch: pytest.MonkeyPatch) -> None:
    owner, backend, caches = _layer_owner(monkeypatch)

    monkeypatch.setattr(
        dispatch_module,
        "bind_decode_csa_native_metadata",
        lambda *_args, **_kwargs: pytest.fail("unwarmed capture must not enter metadata binding"),
    )

    with pytest.raises(DecodeCSADispatchError, match="ordinary-eager warmup"):
        _invoke(owner, caches, capturing=True)
    assert backend.bind_calls == []


def test_output_storage_alias_fails_before_backend_binding(monkeypatch: pytest.MonkeyPatch) -> None:
    owner, backend, caches = _layer_owner(monkeypatch)
    hidden = torch.empty((32, 4), dtype=torch.bfloat16)

    with pytest.raises(DecodeCSADispatchError, match="must not alias.*hidden_states"):
        owner(
            forward_context=SimpleNamespace(capturing=False),
            capturing=False,
            hidden_states=hidden,
            need_gather_q_kv=False,
            output=hidden.view_as(hidden),
            kv_cache=caches,
            attn_metadata=("native-metadata",),
        )

    assert backend.bind_calls == []


def test_warmup_state_is_shared_by_batch_but_bindings_remain_layer_specific(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = FakeBackend(device=0, runtime="trb", dependencies=None)
    first, _, first_caches = _layer_owner(monkeypatch, backend=backend)
    second, _, second_caches = _layer_owner(monkeypatch, backend=backend)
    second.device_owner = first.device_owner
    second.weights = FakeWeights({"weight": "second-layer"})  # type: ignore[assignment]

    _invoke(first, first_caches, capturing=False)
    first.device_owner.mark_warmups_quiesced()
    _invoke(second, second_caches, capturing=True)

    assert len(backend.warmup_calls) == 1
    assert len(backend.launch_calls) == 1
    assert backend.bind_calls[0]["arguments"]["weight"] == "layer-weight"  # type: ignore[index]
    assert backend.bind_calls[1]["arguments"]["weight"] == "second-layer"  # type: ignore[index]
    assert backend.bind_calls[1]["retain"] is True


def test_storage_change_and_flash_comm_fail_before_binding(monkeypatch: pytest.MonkeyPatch) -> None:
    owner, backend, caches = _layer_owner(monkeypatch)
    changed = list(caches)
    changed[3] = torch.empty_like(changed[3])

    with pytest.raises(DecodeCSADispatchError, match="cache storage changed"):
        _invoke(owner, tuple(changed), capturing=False)
    with pytest.raises(DecodeCSADispatchError, match="FlashComm"):
        owner(
            forward_context=SimpleNamespace(capturing=False),
            capturing=False,
            hidden_states=torch.empty((32, 4), dtype=torch.bfloat16),
            need_gather_q_kv=True,
            output=torch.empty((32, 4), dtype=torch.bfloat16),
            kv_cache=caches,
            attn_metadata=(),
        )
    assert backend.bind_calls == []


def test_install_attaches_two_layers_to_one_prepared_device_owner(monkeypatch: pytest.MonkeyPatch) -> None:
    spec = _spec()
    caches = _cache_tuple()
    metadata = (object(),)
    prepared_aliases: list[tuple[int, tuple[torch.Tensor, ...]]] = []
    factory = FakeBackendFactory()
    registry = DecodeCSADeviceOwnerRegistry(backend_factory=factory)  # type: ignore[arg-type]
    weights = FakeWeights({"weight": "packed"})

    monkeypatch.setattr(dispatch_module, "derive_decode_csa_program_spec", lambda *_args: spec)
    monkeypatch.setattr(dispatch_module, "validate_decode_csa_uniform_query_rows", lambda *_args: None)
    monkeypatch.setattr(
        dispatch_module,
        "bind_decode_csa_native_metadata",
        lambda *_args, **_kwargs: SimpleNamespace(
            full_rope_cos=object(),
            full_rope_sin=object(),
            hadamard=object(),
        ),
    )

    def prepare_cache_aliases(
        actual_caches: tuple[torch.Tensor, ...],
        actual_spec: DecodeCSAProgramSpec,
    ) -> object:
        prepared_aliases.append((actual_spec.batch, actual_caches))
        return f"aliases-b{actual_spec.batch}"

    monkeypatch.setattr(dispatch_module, "prepare_decode_csa_caches", prepare_cache_aliases)
    first_layer = SimpleNamespace()
    second_layer = SimpleNamespace()

    first = install_pypto_dsv4_decode_csa(
        first_layer,
        kv_cache=caches,
        attn_metadata=metadata,
        device=0,
        runtime="trb",
        batch_buckets=(4, 8),
        registry=registry,
        prepared_weights=weights,  # type: ignore[arg-type]
        uniform_query_rows_contract=True,
    )
    second = install_pypto_dsv4_decode_csa(
        second_layer,
        kv_cache=caches,
        attn_metadata=metadata,
        device=0,
        runtime="trb",
        batch_buckets=(4, 8),
        registry=registry,
        prepared_weights=weights,  # type: ignore[arg-type]
        uniform_query_rows_contract=True,
    )

    assert first.device_owner is second.device_owner
    assert first_layer._pypto_dsv4_csa_dispatch is first
    assert second_layer._pypto_dsv4_csa_dispatch is second
    assert len(factory.instances) == 1
    assert factory.instances[0].events == [("compile", 4), ("compile", 8), ("prepare",)]
    assert prepared_aliases == [(4, caches), (8, caches), (4, caches), (8, caches)]

    uninstall_pypto_dsv4_decode_csa(first_layer)
    assert not hasattr(first_layer, "_pypto_dsv4_csa_dispatch")
    assert registry.get(device=0, runtime="trb") is first.device_owner


def test_custom_op_hook_uses_built_cache_and_sorted_metadata_without_native_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vllm_ascend.ops.dsa as dsa_module

    calls: list[dict[str, object]] = []
    native_calls: list[tuple[object, ...]] = []
    caches = ("six", "cache", "objects", "are", "built", "first")
    layer = SimpleNamespace(
        prefix="model.layers.2.self_attn",
        dsa_attn=SimpleNamespace(
            layer_name="native-layer",
            impl=SimpleNamespace(forward=lambda *args: native_calls.append(args)),
        ),
    )

    def installed_dispatch(**kwargs: object) -> None:
        calls.append(kwargs)

    layer._pypto_dsv4_csa_dispatch = installed_dispatch
    metadata_a = object()
    metadata_b = object()
    context = SimpleNamespace(
        no_compile_layers={"layer-name": layer},
        attn_metadata={
            "model.layers.2.self_attn.zz": metadata_b,
            "unrelated": object(),
            "model.layers.2.self_attn.aa": metadata_a,
        },
        capturing=True,
    )
    monkeypatch.setattr(dsa_module, "get_forward_context", lambda: context)
    monkeypatch.setattr(dsa_module, "_build_kv_cache", lambda actual_layer, actual_context: caches)
    hidden = torch.empty((32, 4))
    output = torch.empty_like(hidden)

    dsa_module.dsa_forward(hidden, False, output, "layer-name")

    assert native_calls == []
    assert calls == [
        {
            "forward_context": context,
            "capturing": True,
            "hidden_states": hidden,
            "need_gather_q_kv": False,
            "output": output,
            "kv_cache": caches,
            "attn_metadata": [metadata_a, metadata_b],
        }
    ]

    def fail_after_possible_cache_writes(**_kwargs: object) -> None:
        raise RuntimeError("enqueue failed after mutable state may have changed")

    layer._pypto_dsv4_csa_dispatch = fail_after_possible_cache_writes
    with pytest.raises(RuntimeError, match="enqueue failed"):
        dsa_module.dsa_forward(hidden, False, output, "layer-name")
    assert native_calls == []


def test_custom_op_hook_honors_v2_extra_context_capture_signal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vllm_ascend.ops.dsa as dsa_module

    calls: list[dict[str, object]] = []
    caches = ("six", "cache", "objects", "are", "built", "first")
    layer = SimpleNamespace(
        prefix="model.layers.2.self_attn",
        dsa_attn=SimpleNamespace(
            layer_name="native-layer",
            impl=SimpleNamespace(forward=lambda *_args: pytest.fail("native fallback is forbidden")),
        ),
        _pypto_dsv4_csa_dispatch=lambda **kwargs: calls.append(kwargs),
    )
    metadata = object()
    # v2 full-graph capture does not set ForwardContext.capturing.  Its
    # wrapper-level extra context is the authoritative capture signal.
    context = SimpleNamespace(
        no_compile_layers={"layer-name": layer},
        attn_metadata={"model.layers.2.self_attn.decode": metadata},
        capturing=False,
    )
    monkeypatch.setattr(dsa_module, "get_forward_context", lambda: context)
    monkeypatch.setattr(dsa_module, "_EXTRA_CTX", SimpleNamespace(capturing=True))
    monkeypatch.setattr(dsa_module, "_build_kv_cache", lambda *_args: caches)
    hidden = torch.empty((32, 4))
    output = torch.empty_like(hidden)

    dsa_module.dsa_forward(hidden, False, output, "layer-name")

    assert len(calls) == 1
    assert calls[0]["forward_context"] is context
    assert calls[0]["capturing"] is True
    assert calls[0]["kv_cache"] is caches
    assert calls[0]["attn_metadata"] == [metadata]


def test_custom_op_profiling_and_uninstalled_paths_remain_native(monkeypatch: pytest.MonkeyPatch) -> None:
    import vllm_ascend.ops.dsa as dsa_module

    native_calls: list[tuple[object, ...]] = []
    layer = SimpleNamespace(
        prefix="layer",
        dsa_attn=SimpleNamespace(
            layer_name="native-layer",
            impl=SimpleNamespace(forward=lambda *args: native_calls.append(args)),
        ),
    )
    hidden = torch.empty((32, 4))
    output = torch.empty_like(hidden)

    profiling_context = SimpleNamespace(no_compile_layers={"name": layer}, attn_metadata=None)
    monkeypatch.setattr(dsa_module, "get_forward_context", lambda: profiling_context)
    dsa_module.dsa_forward(hidden, False, output, "name")
    assert native_calls == [("native-layer", hidden, None, None, False, output)]

    native_calls.clear()
    metadata = object()
    eager_context = SimpleNamespace(no_compile_layers={"name": layer}, attn_metadata={"layer.x": metadata})
    monkeypatch.setattr(dsa_module, "get_forward_context", lambda: eager_context)
    monkeypatch.setattr(dsa_module, "_build_kv_cache", lambda *_args: (1, 2, 3, 4, 5, 6))
    dsa_module.dsa_forward(hidden, True, output, "name")
    assert native_calls == [("native-layer", hidden, (1, 2, 3, 4, 5, 6), [metadata], True, output)]
