# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Host-only structural tests for the multi-layer decode CSA Capsule."""

from __future__ import annotations

import pytest

from tests.pypto_dsv4_decode_csa.capsule import (
    BackendKind,
    BridgeKind,
    CallableIdentity,
    CallStage,
    CapsuleConfig,
    CapsuleError,
    CapsuleTopology,
    CSALayerCapsule,
    HCPreResult,
    HostActivation,
    HostRecordingExecutor,
    MutableCacheOwner,
    WeightOwner,
    build_capsule,
    topology_backends,
)


@pytest.mark.parametrize("num_layers", (1, 2, 4))
def test_capsule_supports_only_the_intended_layer_depths(num_layers: int) -> None:
    runner = build_capsule(CapsuleConfig(num_layers=num_layers))

    assert len(runner.layers) == num_layers
    assert len(runner.bridges) == num_layers - 1
    assert [layer.layer_index for layer in runner.layers] == list(range(num_layers))
    assert len({layer.layer_name for layer in runner.layers}) == num_layers


@pytest.mark.parametrize("num_layers", (0, 3, 8))
def test_capsule_rejects_unplanned_layer_depths(num_layers: int) -> None:
    with pytest.raises(CapsuleError, match="num_layers"):
        CapsuleConfig(num_layers=num_layers)


@pytest.mark.parametrize(
    ("topology", "expected"),
    (
        (CapsuleTopology.NNNN, "NNNN"),
        (CapsuleTopology.PPPP, "PPPP"),
        (CapsuleTopology.NPNP, "NPNP"),
        (CapsuleTopology.PNPN, "PNPN"),
        (CapsuleTopology.PP_SAME, "PPPP"),
        (CapsuleTopology.PP_DISTINCT, "PPPP"),
    ),
)
def test_four_layer_topology_routes_match_the_design(
    topology: CapsuleTopology,
    expected: str,
) -> None:
    route = topology_backends(topology, 4)
    encoded = "".join("N" if backend is BackendKind.NATIVE else "P" for backend in route)
    assert encoded == expected


@pytest.mark.parametrize(
    "topology",
    tuple(CapsuleTopology),
)
def test_every_topology_has_independent_layer_weight_and_cache_owners(
    topology: CapsuleTopology,
) -> None:
    runner = build_capsule(CapsuleConfig(topology=topology))

    weight_ids = [layer.weight_owner.owner_id for layer in runner.layers]
    cache_ids = [layer.cache_owner.owner_id for layer in runner.layers]
    weight_addresses = [layer.weight_owner.address_token for layer in runner.layers]
    cache_addresses = [layer.cache_owner.address_token for layer in runner.layers]
    assert len(set(weight_ids)) == 4
    assert len(set(cache_ids)) == 4
    assert len(set(weight_addresses)) == 4
    assert len(set(cache_addresses)) == 4
    assert all(layer.weight_owner.layer_name == layer.layer_name for layer in runner.layers)
    assert all(layer.cache_owner.layer_name == layer.layer_name for layer in runner.layers)


def test_recorded_stage_order_matches_the_layer_shell_and_bridge_boundary() -> None:
    runner = build_capsule(
        CapsuleConfig(
            num_layers=2,
            topology=CapsuleTopology.NPNP,
            bridge_kind=BridgeKind.IDENTITY_RESIDUAL,
        )
    )
    executor = HostRecordingExecutor()
    result = runner.run(executor.make_input("decode"), executor)

    assert [record.ordinal for record in result.call_records] == list(range(11))
    assert [record.stage for record in result.call_records] == [
        CallStage.RESIDUAL_CLONE,
        CallStage.HC_PRE,
        CallStage.RMS_NORM,
        CallStage.DSA_FORWARD,
        CallStage.HC_POST,
        CallStage.IDENTITY_RESIDUAL_BRIDGE,
        CallStage.RESIDUAL_CLONE,
        CallStage.HC_PRE,
        CallStage.RMS_NORM,
        CallStage.DSA_FORWARD,
        CallStage.HC_POST,
    ]
    assert [record.backend for record in result.csa_calls] == [
        BackendKind.NATIVE,
        BackendKind.PYPTO,
    ]
    assert [record.layer_index for record in result.csa_calls] == [0, 1]
    assert len(result.layer_outputs) == 2


def test_hc_pre_result_exposes_hidden_post_and_comb_as_distinct_values() -> None:
    layer = build_capsule(CapsuleConfig(num_layers=1)).layers[0]
    executor = HostRecordingExecutor()
    value = executor.make_input("hc-input")
    residual = executor.residual_clone(layer, value)

    result = executor.hc_pre(layer, value, residual)

    assert isinstance(result, HCPreResult)
    assert isinstance(result.hidden, HostActivation)
    assert isinstance(result.post, HostActivation)
    assert isinstance(result.comb, HostActivation)
    addresses = tuple(executor.address_token(item) for item in (result.hidden, result.post, result.comb))
    assert len(set(addresses)) == 3
    assert result.hidden.lineage[-1].endswith("hc_pre:hidden")
    assert result.post.lineage[-1].endswith("hc_pre:post")
    assert result.comb.lineage[-1].endswith("hc_pre:comb")


def test_hc_post_consumes_the_exact_post_and_comb_produced_by_hc_pre() -> None:
    runner = build_capsule(CapsuleConfig(num_layers=1))
    executor = HostRecordingExecutor()

    records = runner.run(executor.make_input("hc-flow"), executor).call_records
    residual_record = records[0]
    hc_pre_record = records[1]
    rms_norm_record = records[2]
    dsa_record = records[3]
    hc_post_record = records[4]

    assert hc_pre_record.stage is CallStage.HC_PRE
    assert hc_post_record.stage is CallStage.HC_POST
    assert hc_pre_record.output_address == rms_norm_record.input_address
    assert dsa_record.output_address == hc_post_record.input_address
    assert hc_pre_record.auxiliary_addresses == hc_post_record.auxiliary_addresses
    assert hc_pre_record.auxiliary_addresses[0] == residual_record.output_address
    _, post_address, comb_address = hc_pre_record.auxiliary_addresses
    assert post_address != comb_address
    assert post_address not in {hc_pre_record.input_address, hc_pre_record.output_address}
    assert comb_address not in {hc_pre_record.input_address, hc_pre_record.output_address}


def test_runner_rejects_executor_that_hides_hc_post_and_comb_state() -> None:
    class InvalidHCExecutor(HostRecordingExecutor):
        def hc_pre(
            self,
            layer: CSALayerCapsule,
            value: object,
            residual: object,
        ) -> object:
            return super().hc_pre(layer, value, residual).hidden

    runner = build_capsule(CapsuleConfig(num_layers=1))
    executor = InvalidHCExecutor()

    with pytest.raises(TypeError, match="must return HCPreResult"):
        runner.run(executor.make_input("invalid-hc"), executor)


@pytest.mark.parametrize(
    ("topology", "expected"),
    (
        (CapsuleTopology.NNNN, (BackendKind.NATIVE,) * 4),
        (CapsuleTopology.PPPP, (BackendKind.PYPTO,) * 4),
        (
            CapsuleTopology.NPNP,
            (
                BackendKind.NATIVE,
                BackendKind.PYPTO,
                BackendKind.NATIVE,
                BackendKind.PYPTO,
            ),
        ),
        (
            CapsuleTopology.PNPN,
            (
                BackendKind.PYPTO,
                BackendKind.NATIVE,
                BackendKind.PYPTO,
                BackendKind.NATIVE,
            ),
        ),
    ),
)
def test_call_records_prove_backend_route(
    topology: CapsuleTopology,
    expected: tuple[BackendKind, ...],
) -> None:
    runner = build_capsule(CapsuleConfig(topology=topology))
    executor = HostRecordingExecutor()
    result = runner.run(executor.make_input(), executor)

    assert tuple(record.backend for record in result.csa_calls) == expected


def test_pp_same_reuses_callable_but_binds_distinct_layer_addresses() -> None:
    runner = build_capsule(CapsuleConfig(topology=CapsuleTopology.PP_SAME))
    assert runner.layers[0].callable_identity is runner.layers[1].callable_identity
    assert runner.layers[2].callable_identity is not runner.layers[0].callable_identity
    assert runner.layers[3].callable_identity is not runner.layers[0].callable_identity

    executor = HostRecordingExecutor()
    result = runner.run(executor.make_input(), executor)
    first, second, third, fourth = result.csa_calls
    assert first.callable_id == second.callable_id
    assert first.logical_function_id == second.logical_function_id
    assert first.argument_addresses != second.argument_addresses
    assert first.weight_owner_id != second.weight_owner_id
    assert first.cache_owner_id != second.cache_owner_id
    assert len({third.callable_id, fourth.callable_id, first.callable_id}) == 3


def test_pp_distinct_has_one_logical_callable_identity_per_layer() -> None:
    runner = build_capsule(CapsuleConfig(topology=CapsuleTopology.PP_DISTINCT))
    executor = HostRecordingExecutor()
    calls = runner.run(executor.make_input(), executor).csa_calls

    assert len({call.callable_id for call in calls}) == 4
    assert len({call.logical_function_id for call in calls}) == 4
    assert len({id(layer.callable_identity) for layer in runner.layers}) == 4


def test_mutable_cache_audit_trails_remain_layer_local_across_repeated_runs() -> None:
    runner = build_capsule(CapsuleConfig(topology=CapsuleTopology.PPPP))
    executor = HostRecordingExecutor()
    runner.run(executor.make_input("first"), executor)
    runner.run(executor.make_input("second"), executor)

    for layer in runner.layers:
        assert layer.cache_owner.mutation_count == 2
        assert [mutation.owner_local_invocation for mutation in layer.cache_owner.mutations] == [0, 1]
        assert all(mutation.layer_name == layer.layer_name for mutation in layer.cache_owner.mutations)
        assert all(
            mutation.callable_id == layer.callable_identity.callable_id for mutation in layer.cache_owner.mutations
        )
    assert len({id(layer.cache_owner) for layer in runner.layers}) == 4
    mutation_inputs = {mutation.input_address for layer in runner.layers for mutation in layer.cache_owner.mutations}
    assert len(mutation_inputs) == 8


@pytest.mark.parametrize(
    ("kind", "expected_stage", "expected_rank"),
    (
        (
            BridgeKind.IDENTITY_RESIDUAL,
            CallStage.IDENTITY_RESIDUAL_BRIDGE,
            0,
        ),
        (
            BridgeKind.LOW_RANK_FFN,
            CallStage.LOW_RANK_FFN_BRIDGE,
            24,
        ),
    ),
)
def test_bridges_are_deterministic_symbolic_specs_not_fake_numerics(
    kind: BridgeKind,
    expected_stage: CallStage,
    expected_rank: int,
) -> None:
    config = CapsuleConfig(
        num_layers=4,
        bridge_kind=kind,
        seed=37,
        low_rank_ffn_rank=24,
    )
    first_runner = build_capsule(config)
    second_runner = build_capsule(config)
    assert first_runner.bridges == second_runner.bridges
    assert all(bridge.low_rank == expected_rank for bridge in first_runner.bridges)

    first_executor = HostRecordingExecutor()
    second_executor = HostRecordingExecutor()
    first = first_runner.run(first_executor.make_input("same"), first_executor)
    second = second_runner.run(second_executor.make_input("same"), second_executor)
    assert first.call_records == second.call_records
    assert isinstance(first.final_value, HostActivation)
    assert isinstance(second.final_value, HostActivation)
    assert first.final_value.lineage == second.final_value.lineage
    assert not hasattr(first.final_value, "numeric_value")
    assert [record.stage for record in first.bridge_calls] == [expected_stage] * 3


def test_injected_owners_and_callable_payloads_are_preserved_for_future_npu_runner() -> None:
    config = CapsuleConfig(num_layers=2, topology=CapsuleTopology.PP_DISTINCT)
    names = tuple(f"{config.layer_name_prefix}.{index}.dsa" for index in range(2))
    weight_payloads = (object(), object())
    cache_payloads = (object(), object())
    callable_payloads = (object(), object())
    weights = tuple(
        WeightOwner(
            owner_id=f"injected-weight-{index}",
            address_token=10_000 + index,
            layer_name=name,
            payload=weight_payloads[index],
        )
        for index, name in enumerate(names)
    )
    caches = tuple(
        MutableCacheOwner(
            owner_id=f"injected-cache-{index}",
            address_token=20_000 + index,
            layer_name=name,
            payload=cache_payloads[index],
        )
        for index, name in enumerate(names)
    )
    callables = tuple(
        CallableIdentity(
            callable_id=f"injected-callable-{index}",
            logical_function_id=index,
            backend=BackendKind.PYPTO,
            payload=callable_payloads[index],
        )
        for index in range(2)
    )

    runner = build_capsule(
        config,
        weight_owners=weights,
        cache_owners=caches,
        callable_identities=callables,
    )
    for index, layer in enumerate(runner.layers):
        assert layer.weight_owner.payload is weight_payloads[index]
        assert layer.cache_owner.payload is cache_payloads[index]
        assert layer.callable_identity.payload is callable_payloads[index]


def test_owner_aliasing_and_topology_identity_mismatch_are_rejected() -> None:
    config = CapsuleConfig(num_layers=2, topology=CapsuleTopology.PP_DISTINCT)
    names = tuple(f"{config.layer_name_prefix}.{index}.dsa" for index in range(2))
    weights = tuple(
        WeightOwner(
            owner_id=f"weight-{index}",
            address_token="same-address",
            layer_name=name,
        )
        for index, name in enumerate(names)
    )
    with pytest.raises(CapsuleError, match="weight addresses"):
        build_capsule(config, weight_owners=weights)

    shared = CallableIdentity(
        callable_id="unexpected-shared",
        logical_function_id=0,
        backend=BackendKind.PYPTO,
    )
    with pytest.raises(CapsuleError, match="PP_distinct"):
        build_capsule(config, callable_identities=(shared, shared))

    aliased_function_ids = tuple(
        CallableIdentity(
            callable_id=f"distinct-name-{index}",
            logical_function_id=0,
            backend=BackendKind.PYPTO,
        )
        for index in range(2)
    )
    with pytest.raises(CapsuleError, match="logical function ID"):
        build_capsule(config, callable_identities=aliased_function_ids)
