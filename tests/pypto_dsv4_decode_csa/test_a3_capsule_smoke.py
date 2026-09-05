# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Host contracts for the production custom-op A3 Capsule runner."""

from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

import tests.pypto_dsv4_decode_csa.a3_capsule_smoke as capsule_smoke_module
from tests.pypto_dsv4_decode_csa.a3_capsule_smoke import (
    A3CapsuleLayerAddressEvidence,
    A3CapsuleReplayObservation,
    A3CapsuleSmokeConfig,
    A3CapsuleSmokeResult,
    build_a3_capsule_execution_plan,
    build_parser,
    run_a3_capsule_smoke,
    validate_a3_capsule_resource_independence,
)
from tests.pypto_dsv4_decode_csa.capsule import CapsuleTopology
from vllm_ascend.ops._pypto_dsv4_csa.dispatch import DecodeCSADeviceOwner


@pytest.mark.parametrize("num_layers", (1, 2, 4))
@pytest.mark.parametrize(
    "topology",
    (
        CapsuleTopology.PPPP,
        CapsuleTopology.PP_SAME,
        CapsuleTopology.PP_DISTINCT,
    ),
)
def test_execution_plan_supports_intended_depths_and_pypto_topologies(
    num_layers: int,
    topology: CapsuleTopology,
) -> None:
    config = A3CapsuleSmokeConfig(
        runtime="tensormap_and_ringbuffer",
        num_layers=num_layers,
        topology=topology,
    )
    plan = build_a3_capsule_execution_plan(config)

    assert len(plan.layers) == num_layers
    assert [layer.layer_index for layer in plan.layers] == list(range(num_layers))
    assert len({layer.layer_name for layer in plan.layers}) == num_layers
    assert len({layer.weight_owner_token for layer in plan.layers}) == num_layers
    assert len({layer.cache_owner_token for layer in plan.layers}) == num_layers
    assert all(layer.spec.batch == 4 for layer in plan.layers)
    assert plan.layers[-1].bridge_scale is None
    assert plan.layers[-1].bridge_bias is None
    assert all(layer.bridge_scale is not None for layer in plan.layers[:-1])


@pytest.mark.parametrize(
    ("topology", "expected_groups"),
    (
        (CapsuleTopology.PPPP, (0, 1, 2, 3)),
        (CapsuleTopology.PP_SAME, (0, 0, 1, 2)),
        (CapsuleTopology.PP_DISTINCT, (0, 1, 2, 3)),
    ),
)
def test_callable_groups_map_bijectively_to_real_static_specs(
    topology: CapsuleTopology,
    expected_groups: tuple[int, ...],
) -> None:
    plan = build_a3_capsule_execution_plan(
        A3CapsuleSmokeConfig(
            runtime="host_build_graph",
            topology=topology,
        )
    )
    groups = tuple(layer.callable_group for layer in plan.layers)
    keys = tuple(layer.spec.key for layer in plan.layers)

    assert groups == expected_groups
    assert plan.compiled_callable_count == len(set(expected_groups))
    for left in range(4):
        for right in range(4):
            assert (keys[left] == keys[right]) is (groups[left] == groups[right])


def test_pp_same_shares_callable_only_and_keeps_layer_owners_independent() -> None:
    plan = build_a3_capsule_execution_plan(
        A3CapsuleSmokeConfig(
            runtime="tensormap_and_ringbuffer",
            topology=CapsuleTopology.PP_SAME,
        )
    )

    assert plan.layers[0].callable_group == plan.layers[1].callable_group
    assert plan.layers[0].spec.key == plan.layers[1].spec.key
    assert plan.layers[0].weight_owner_token != plan.layers[1].weight_owner_token
    assert plan.layers[0].cache_owner_token != plan.layers[1].cache_owner_token
    assert plan.layers[0].layer_name != plan.layers[1].layer_name


def test_bridge_parameters_are_deterministic_and_seeded() -> None:
    first = build_a3_capsule_execution_plan(A3CapsuleSmokeConfig(runtime="tensormap_and_ringbuffer", seed=41))
    second = build_a3_capsule_execution_plan(A3CapsuleSmokeConfig(runtime="tensormap_and_ringbuffer", seed=41))
    changed = build_a3_capsule_execution_plan(A3CapsuleSmokeConfig(runtime="tensormap_and_ringbuffer", seed=42))

    assert first.layers == second.layers
    assert tuple(layer.bridge_bias for layer in first.layers) != tuple(layer.bridge_bias for layer in changed.layers)
    assert tuple(layer.bridge_scale for layer in first.layers) == tuple(layer.bridge_scale for layer in changed.layers)


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    (
        ({"runtime": "invalid"}, ValueError, "unsupported runtime"),
        ({"runtime": "tensormap_and_ringbuffer", "device": True}, ValueError, "device"),
        ({"runtime": "tensormap_and_ringbuffer", "num_layers": 3}, ValueError, "num_layers"),
        (
            {"runtime": "tensormap_and_ringbuffer", "topology": CapsuleTopology.NNNN},
            ValueError,
            "does not support",
        ),
        (
            {"runtime": "tensormap_and_ringbuffer", "topology": CapsuleTopology.NPNP},
            ValueError,
            "does not support",
        ),
        ({"runtime": "tensormap_and_ringbuffer", "replay_values": ()}, ValueError, "must not be empty"),
        (
            {"runtime": "tensormap_and_ringbuffer", "replay_values": (float("inf"),)},
            ValueError,
            "finite",
        ),
    ),
)
def test_config_rejects_unsupported_runner_plans_before_npu_import(
    kwargs: dict[str, object],
    error: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error, match=match):
        A3CapsuleSmokeConfig(**kwargs)  # type: ignore[arg-type]


def _independent_evidence(plan) -> tuple[A3CapsuleLayerAddressEvidence, ...]:
    result = []
    for layer in plan.layers:
        base = 10_000 * (layer.layer_index + 1)
        result.append(
            A3CapsuleLayerAddressEvidence(
                layer_index=layer.layer_index,
                layer_name=layer.layer_name,
                callable_group=layer.callable_group,
                fixture_owner_id=base + 1,
                weight_owner_id=base + 2,
                cache_owner_id=base + 3,
                layer_owner_id=base + 4,
                device_owner_id=50_000 + layer.callable_group,
                backend_owner_id=701,
                context_owner_id=702,
                weight_addresses=(base + 10, base + 11),
                metadata_addresses=(base + 12, base + 13),
                cache_addresses=(base + 20, base + 21, base + 22),
                eager_input_address=base + 30,
                eager_output_address=base + 31,
                capture_input_address=base + 40,
                capture_output_address=base + 41,
            )
        )
    return tuple(result)


@pytest.mark.parametrize("topology", (CapsuleTopology.PPPP, CapsuleTopology.PP_SAME, CapsuleTopology.PP_DISTINCT))
def test_resource_audit_proves_layer_and_address_independence(topology: CapsuleTopology) -> None:
    plan = build_a3_capsule_execution_plan(A3CapsuleSmokeConfig(runtime="tensormap_and_ringbuffer", topology=topology))
    evidence = _independent_evidence(plan)

    validate_a3_capsule_resource_independence(plan, evidence)

    assert len({row.fixture_owner_id for row in evidence}) == 4
    assert len({row.weight_owner_id for row in evidence}) == 4
    assert len({row.cache_owner_id for row in evidence}) == 4
    assert len({row.layer_owner_id for row in evidence}) == 4
    assert len({row.device_owner_id for row in evidence}) == plan.compiled_callable_count
    assert len({row.backend_owner_id for row in evidence}) == 1
    assert len({row.context_owner_id for row in evidence}) == 1
    if topology is CapsuleTopology.PP_SAME:
        assert evidence[0].callable_group == evidence[1].callable_group
        assert evidence[0].weight_addresses != evidence[1].weight_addresses
        assert evidence[0].cache_addresses != evidence[1].cache_addresses
        assert evidence[0].capture_input_address != evidence[1].capture_input_address


def test_resource_audit_rejects_cross_layer_storage_and_graph_address_aliases() -> None:
    plan = build_a3_capsule_execution_plan(
        A3CapsuleSmokeConfig(runtime="tensormap_and_ringbuffer", topology=CapsuleTopology.PP_DISTINCT)
    )
    evidence = list(_independent_evidence(plan))
    evidence[1] = replace(evidence[1], weight_addresses=evidence[0].weight_addresses)
    with pytest.raises(ValueError, match="alias live tensor storage"):
        validate_a3_capsule_resource_independence(plan, evidence)

    evidence = list(_independent_evidence(plan))
    evidence[1] = replace(evidence[1], capture_input_address=evidence[0].capture_input_address)
    with pytest.raises(ValueError, match="capture_input_address"):
        validate_a3_capsule_resource_independence(plan, evidence)

    evidence = list(_independent_evidence(plan))
    evidence[0] = replace(evidence[0], capture_output_address=evidence[0].eager_output_address)
    with pytest.raises(ValueError, match="eager/capture outputs alias"):
        validate_a3_capsule_resource_independence(plan, evidence)


def test_resource_audit_rejects_split_backend_context_and_cross_category_alias() -> None:
    plan = build_a3_capsule_execution_plan(
        A3CapsuleSmokeConfig(runtime="tensormap_and_ringbuffer", topology=CapsuleTopology.PP_SAME)
    )

    evidence = list(_independent_evidence(plan))
    evidence[1] = replace(evidence[1], device_owner_id=999)
    with pytest.raises(ValueError, match="device-owner identity"):
        validate_a3_capsule_resource_independence(plan, evidence)

    evidence = list(_independent_evidence(plan))
    evidence[1] = replace(evidence[1], backend_owner_id=999)
    with pytest.raises(ValueError, match="share one backend_owner_id"):
        validate_a3_capsule_resource_independence(plan, evidence)

    evidence = list(_independent_evidence(plan))
    evidence[1] = replace(evidence[1], context_owner_id=999)
    with pytest.raises(ValueError, match="share one context_owner_id"):
        validate_a3_capsule_resource_independence(plan, evidence)

    evidence = list(_independent_evidence(plan))
    evidence[1] = replace(evidence[1], metadata_addresses=(evidence[0].cache_addresses[0],))
    with pytest.raises(ValueError, match="alias live tensor storage"):
        validate_a3_capsule_resource_independence(plan, evidence)


def test_runtime_audit_checks_prepared_callable_and_context_identity() -> None:
    plan = build_a3_capsule_execution_plan(
        A3CapsuleSmokeConfig(runtime="host_build_graph", topology=CapsuleTopology.PP_SAME)
    )
    context = object()
    operators = {group: object() for group in {layer.callable_group for layer in plan.layers}}
    backend = SimpleNamespace(
        _context=context,
        _records={layer.spec.key: SimpleNamespace(operator=operators[layer.callable_group]) for layer in plan.layers},
    )
    facades = {group: SimpleNamespace(backend=backend) for group in {layer.callable_group for layer in plan.layers}}
    runtime_layers = tuple(
        SimpleNamespace(owner=SimpleNamespace(device_owner=facades[layer.callable_group])) for layer in plan.layers
    )

    capsule_smoke_module._validate_runtime_layer_sharing(plan, runtime_layers, backend)

    split_backend = SimpleNamespace(_context=object(), _records=backend._records)
    broken = list(runtime_layers)
    broken[1] = SimpleNamespace(owner=SimpleNamespace(device_owner=SimpleNamespace(backend=split_backend)))
    with pytest.raises(AssertionError, match="share the requested PyPTO backend"):
        capsule_smoke_module._validate_runtime_layer_sharing(plan, broken, backend)


def test_pp_same_shared_facade_warms_one_callable_once() -> None:
    calls: list[str] = []
    fake_backend = SimpleNamespace(
        warmup=lambda _binding: calls.append("warmup") or object(),
        launch=lambda _binding: calls.append("launch") or object(),
    )
    shared = DecodeCSADeviceOwner(
        device=0,
        runtime="tensormap_and_ringbuffer",
        backend=fake_backend,  # type: ignore[arg-type]
        specs={4: object()},  # type: ignore[dict-item]
        family_key=("shared",),
    )

    # This is the exact dispatch choice made by two PP_same layer owners that
    # share one callable-group facade in run_a3_capsule_smoke().
    for _ in range(2):
        if shared.is_warmed(4):
            shared.launch(object())
        else:
            shared.warmup(object(), batch=4)
    shared.mark_warmups_quiesced()
    shared.require_capture_ready(4)

    assert calls == ["warmup", "launch"]


def test_result_schema_is_json_ready_and_reports_one_graph() -> None:
    plan = build_a3_capsule_execution_plan(
        A3CapsuleSmokeConfig(runtime="host_build_graph", num_layers=2, topology=CapsuleTopology.PP_SAME)
    )
    observation = A3CapsuleReplayObservation(
        input_value=0.25,
        eager_graph_max_abs_error=0.0,
        dsa_output_max_abs=0.0,
    )
    result = A3CapsuleSmokeResult(
        runtime="host_build_graph",
        device=0,
        batch=4,
        num_layers=2,
        topology="PP_same",
        graph_capture_count=1,
        compiled_callable_count=1,
        retained_capture_bindings=2,
        replay_observations=(observation,),
        layer_evidence=_independent_evidence(plan),
    )

    payload = result.to_dict()

    assert payload["graph_capture_count"] == 1
    assert payload["compiled_callable_count"] == 1
    assert payload["retained_capture_bindings"] == 2
    assert payload["max_replay_error"] == 0.0
    assert len(payload["layer_evidence"]) == 2
    json.dumps(payload)


def test_runner_type_guard_fails_before_lazy_npu_import() -> None:
    with pytest.raises(TypeError, match="A3CapsuleSmokeConfig"):
        run_a3_capsule_smoke(object())  # type: ignore[arg-type]


def test_cli_exposes_fresh_process_runtime_topology_and_depth_contract() -> None:
    args = build_parser().parse_args(
        [
            "--runtime",
            "host_build_graph",
            "--device",
            "0",
            "--batch",
            "8",
            "--num-layers",
            "2",
            "--topology",
            "PP_same",
            "--replay-values",
            "0.5",
            "-1.0",
        ]
    )

    assert args.runtime == "host_build_graph"
    assert args.device == 0
    assert args.batch == 8
    assert args.num_layers == 2
    assert args.topology == CapsuleTopology.PP_SAME.value
    assert args.replay_values == [0.5, -1.0]
