# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Host contracts for nonzero NNNN-vs-candidate Capsules."""

from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from tests.pypto_dsv4_decode_csa.a3_native_capsule_compare import (
    A3NativeCapsuleCompareConfig,
    A3NativeCapsuleCompareResult,
    A3NativeCapsuleHCLayerEvidence,
    A3NativeCapsuleLayerComparison,
    A3NativeCapsuleLayerEvidence,
    CapsuleLocalOracleTrace,
    CapsuleShellKind,
    HCFidelityLayerOps,
    _apply_capsule_cleanup_outcome,
    _cleanup_a3_native_capsule_compare,
    build_a3_native_capsule_compare_plan,
    build_parser,
    compare_native_capsule_traces,
    execute_capsule_with_local_oracles,
    execute_deterministic_capsule_chain,
    execute_hc_fidelity_capsule_chain,
    execute_hc_fidelity_capsule_with_local_oracles,
    run_a3_native_capsule_compare,
    validate_native_capsule_resource_independence,
)
from tests.pypto_dsv4_decode_csa.a3_native_compare import TensorComparison
from tests.pypto_dsv4_decode_csa.capsule import BackendKind, CapsuleTopology, topology_backends
from tests.pypto_dsv4_decode_csa.indexer_quantized_comparison import compare_indexer_quantized
from tests.pypto_dsv4_decode_csa.native_fixture import (
    STATE_NAMES,
    ratio4_quant_description,
    ratio4_quant_description_for_prefixes,
)


def _config(**overrides: object) -> A3NativeCapsuleCompareConfig:
    values: dict[str, object] = {"runtime": "tensormap_and_ringbuffer"}
    values.update(overrides)
    return A3NativeCapsuleCompareConfig(**values)  # type: ignore[arg-type]


def _comparison(*, close: bool = True) -> TensorComparison:
    return TensorComparison(
        active_elements=1,
        total_elements=1,
        max_abs_error=0.0 if close else 1.0,
        mean_abs_error=0.0 if close else 1.0,
        max_relative_error=0.0 if close else 1.0,
        reference_max_abs=1.0,
        candidate_max_abs=1.0 if close else 2.0,
        finite=True,
        close=close,
    )


def _evidence(
    topology: CapsuleTopology = CapsuleTopology.PPPP,
    *,
    num_layers: int = 4,
) -> tuple[A3NativeCapsuleLayerEvidence, ...]:
    rows = []
    for index, (backend, model_layer_index) in enumerate(
        zip(topology_backends(topology, num_layers), (2, 4, 6, 8)[:num_layers], strict=True)
    ):
        base = 100_000 * (index + 1)
        is_pypto = backend is BackendKind.PYPTO
        rows.append(
            A3NativeCapsuleLayerEvidence(
                layer_index=index,
                layer_name=f"model.layers.{model_layer_index}.self_attn",
                candidate_backend=backend,
                attention_owner_id=base + 1,
                wrapper_owner_id=base + 2,
                prepared_weight_owner_id=base + 3 if is_pypto else None,
                pypto_device_owner_id=7001 if is_pypto else None,
                pypto_backend_owner_id=7002 if is_pypto else None,
                parameter_addresses=(base + 10, base + 11),
                packed_weight_addresses=(base + 20, base + 21) if is_pypto else (),
                shared_static_argument_addresses=(8001, 8002, 8003) if is_pypto else (),
                native_cache_addresses=tuple(base + 100 + offset for offset in range(6)),
                local_oracle_cache_addresses=tuple(base + 200 + offset for offset in range(6)),
                candidate_cache_addresses=tuple(base + 300 + offset for offset in range(6)),
            )
        )
    return tuple(rows)


def _states(*, changed: float = 1.0, num_layers: int = 4) -> tuple[dict[str, torch.Tensor], ...]:
    result = []
    for layer_index in range(num_layers):
        values = {}
        for name_index, name in enumerate(STATE_NAMES):
            value = changed if name_index == layer_index % len(STATE_NAMES) else 0.0
            dtype = torch.int8 if name == "indexer_k" else torch.float16 if name == "indexer_scale" else torch.float32
            values[name] = torch.tensor([value], dtype=dtype)
        result.append(values)
    return tuple(result)


def _hc_evidence(*, num_layers: int = 4, tokens: int = 32) -> tuple[A3NativeCapsuleHCLayerEvidence, ...]:
    rows = []
    for layer_index in range(num_layers):
        base = 10_000 * (layer_index + 1)
        rows.append(
            A3NativeCapsuleHCLayerEvidence(
                layer_index=layer_index,
                layer_name=f"model.layers.{2 + layer_index * 2}.self_attn",
                hc_pre_operator="torch.ops._C_ascend.npu_hc_pre_v2",
                rms_norm_operator="AscendRMSNorm",
                hc_post_operator="torch.ops._C_ascend.npu_hc_post",
                input_shape=(tokens, 4, 4096),
                hidden_shape=(tokens, 4096),
                normalized_shape=(tokens, 4096),
                post_shape=(tokens, 4),
                comb_shape=(tokens, 4, 4),
                output_shape=(tokens, 4, 4096),
                input_dtype="torch.bfloat16",
                input_address=base + 1,
                residual_address=base + 2,
                hidden_address=base + 3,
                normalized_address=base + 4,
                post_address=base + 5,
                comb_address=base + 6,
                output_address=base + 7,
                rms_norm_owner_id=base + 8,
                rms_norm_weight_address=base + 9,
                rms_norm_weight_requires_grad=False,
                hc_parameter_addresses=(base + 10, base + 11, base + 12),
                local_oracle_same_normalized_input=True,
                local_oracle_shared_residual_post_comb=True,
            )
        )
    return tuple(rows)


class _RecordingHCShell:
    def __init__(self) -> None:
        self.hc_pre_inputs: list[torch.Tensor] = []
        self.norm_inputs: list[torch.Tensor] = []
        self.hc_post_arguments: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []

    @property
    def ops(self) -> HCFidelityLayerOps:
        return HCFidelityLayerOps(
            hc_pre=self.hc_pre,
            rms_norm=self.rms_norm,
            hc_post=self.hc_post,
        )

    def hc_pre(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.hc_pre_inputs.append(value)
        tokens = value.shape[0]
        return (
            value[:, 0, :].clone(),
            torch.zeros((tokens, 4), dtype=torch.float32, device=value.device),
            torch.zeros((tokens, 4, 4), dtype=torch.float32, device=value.device),
        )

    def rms_norm(self, value: torch.Tensor) -> torch.Tensor:
        self.norm_inputs.append(value)
        return value.clone()

    def hc_post(
        self,
        value: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
    ) -> torch.Tensor:
        self.hc_post_arguments.append((value, residual, post, comb))
        return residual + value.unsqueeze(1)


def _indexer_quantized():
    initial_k = torch.zeros((1, 1), dtype=torch.int8)
    k = torch.ones_like(initial_k)
    initial_scale = torch.zeros((1, 1), dtype=torch.float16)
    scale = torch.ones_like(initial_scale)
    return compare_indexer_quantized(k, k.clone(), initial_k, scale, scale.clone(), initial_scale)


def _good_fidelity_result(*, num_layers: int = 1) -> A3NativeCapsuleCompareResult:
    layers = tuple(
        A3NativeCapsuleLayerComparison(
            layer_index=index,
            layer_name=f"model.layers.{2 + index * 2}.self_attn",
            dsa_output=_comparison(),
            layer_output=_comparison(),
            states={name: _comparison() for name in STATE_NAMES},
            indexer_quantized=_indexer_quantized(),
        )
        for index in range(num_layers)
    )
    return A3NativeCapsuleCompareResult(
        runtime="tensormap_and_ringbuffer",
        device=0,
        batch=4,
        num_layers=num_layers,
        native_topology="NNNN",
        candidate_topology="PPPP",
        shell_kind=CapsuleShellKind.FIDELITY.value,
        local_oracle_same_candidate_inputs=True,
        candidate_sacrificial_warmup_reset=True,
        local_atol=0.1,
        local_rtol=0.1,
        chain_atol=0.1 * num_layers**0.5,
        chain_rtol=0.1,
        layers=layers,
        final_output=_comparison(),
        resource_evidence=_evidence(num_layers=num_layers),
        real_hc_evidence=_hc_evidence(num_layers=num_layers),
    )


@pytest.mark.parametrize("num_layers", (1, 2, 4))
def test_plan_supports_independent_one_two_and_four_layer_routes(num_layers: int) -> None:
    plan = build_a3_native_capsule_compare_plan(_config(seed=41, num_layers=num_layers))

    assert plan.native_route == (BackendKind.NATIVE,) * num_layers
    assert plan.candidate_route == (BackendKind.PYPTO,) * num_layers
    assert tuple(layer.layer_index for layer in plan.layers) == tuple(range(num_layers))
    assert tuple(layer.model_layer_index for layer in plan.layers) == (2, 4, 6, 8)[:num_layers]
    assert tuple(layer.layer_name for layer in plan.layers) == tuple(
        f"model.layers.{model_layer_index}.self_attn" for model_layer_index in (2, 4, 6, 8)[:num_layers]
    )
    assert len({layer.layer_name for layer in plan.layers}) == num_layers
    assert len({layer.weight_seed for layer in plan.layers}) == num_layers
    assert len({layer.native_cache_owner_token for layer in plan.layers}) == num_layers
    assert len({layer.local_oracle_cache_owner_token for layer in plan.layers}) == num_layers
    assert len({layer.candidate_cache_owner_token for layer in plan.layers}) == num_layers
    assert tuple(layer.candidate_backend for layer in plan.layers) == plan.candidate_route
    assert all(layer.bridge_scale is not None for layer in plan.layers[:-1])
    assert all(layer.bridge_bias is not None for layer in plan.layers[:-1])
    assert plan.layers[-1].bridge_scale is None
    assert plan.layers[-1].bridge_bias is None
    assert plan.to_dict()["native_topology"] == "NNNN"
    assert plan.to_dict()["candidate_topology"] == "PPPP"
    assert plan.to_dict()["shell_kind"] == "fidelity"
    assert plan.to_dict()["chain_atol"] == pytest.approx(0.1 * num_layers**0.5)


@pytest.mark.parametrize(
    ("topology", "route"),
    (
        (CapsuleTopology.PPPP, (BackendKind.PYPTO,) * 4),
        (
            CapsuleTopology.NPNP,
            (BackendKind.NATIVE, BackendKind.PYPTO, BackendKind.NATIVE, BackendKind.PYPTO),
        ),
        (
            CapsuleTopology.PNPN,
            (BackendKind.PYPTO, BackendKind.NATIVE, BackendKind.PYPTO, BackendKind.NATIVE),
        ),
    ),
)
def test_candidate_topologies_route_each_layer_to_an_independent_state_world(
    topology: CapsuleTopology,
    route: tuple[BackendKind, ...],
) -> None:
    plan = build_a3_native_capsule_compare_plan(_config(candidate_topology=topology))

    assert plan.native_route == (BackendKind.NATIVE,) * 4
    assert plan.candidate_route == route
    assert tuple(layer.candidate_backend for layer in plan.layers) == route
    assert len({layer.native_cache_owner_token for layer in plan.layers}) == 4
    assert len({layer.local_oracle_cache_owner_token for layer in plan.layers}) == 4
    assert len({layer.candidate_cache_owner_token for layer in plan.layers}) == 4
    assert not (
        {layer.native_cache_owner_token for layer in plan.layers}
        | {layer.local_oracle_cache_owner_token for layer in plan.layers}
    ) & {layer.candidate_cache_owner_token for layer in plan.layers}
    assert not (
        {layer.native_cache_owner_token for layer in plan.layers}
        & {layer.candidate_cache_owner_token for layer in plan.layers}
    )


@pytest.mark.parametrize("num_layers", (1, 2, 4))
@pytest.mark.parametrize(
    "topology",
    (CapsuleTopology.PPPP, CapsuleTopology.NPNP, CapsuleTopology.PNPN),
)
def test_every_eager_candidate_topology_is_plannable_at_each_supported_depth(
    num_layers: int,
    topology: CapsuleTopology,
) -> None:
    plan = build_a3_native_capsule_compare_plan(_config(num_layers=num_layers, candidate_topology=topology))

    assert plan.candidate_route == topology_backends(topology, num_layers)
    assert len(plan.layers) == num_layers
    assert tuple(layer.candidate_backend for layer in plan.layers) == plan.candidate_route


def test_deterministic_diagnostic_shell_remains_explicitly_selectable() -> None:
    config = _config(shell_kind=CapsuleShellKind.DETERMINISTIC_DIAGNOSTIC)
    plan = build_a3_native_capsule_compare_plan(config)

    assert config.shell_kind is CapsuleShellKind.DETERMINISTIC_DIAGNOSTIC
    assert plan.to_dict()["shell_kind"] == "deterministic_diagnostic"


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"runtime": "invalid"}, "runtime"),
        ({"device": True}, "device"),
        ({"batch": 8}, "B4"),
        ({"num_layers": True}, "num_layers"),
        ({"num_layers": 3}, "num_layers"),
        ({"candidate_topology": CapsuleTopology.NNNN}, "candidate_topology"),
        ({"start_position": -1}, "RoPE"),
        ({"seed": True}, "seed"),
        ({"atol": -1.0}, "atol"),
        ({"rtol": float("nan")}, "rtol"),
        ({"chain_atol": -1.0}, "chain_atol"),
        ({"chain_rtol": float("nan")}, "chain_rtol"),
        ({"layer_name_prefix": ""}, "prefix"),
        ({"shell_kind": "fake"}, "shell_kind"),
    ),
)
def test_config_fails_before_device_bootstrap(overrides: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _config(**overrides)


def test_deterministic_chain_preserves_order_and_uses_exact_bridge_parameters() -> None:
    plan = build_a3_native_capsule_compare_plan(_config(seed=17))
    observed_inputs: list[torch.Tensor] = []

    def layer_call(index: int):
        def call(value: torch.Tensor) -> torch.Tensor:
            observed_inputs.append(value.clone())
            return torch.full_like(value, (index + 1) / 8.0)

        return call

    initial = torch.full((2, 3), 0.25, dtype=torch.float32)
    trace = execute_deterministic_capsule_chain(
        plan,
        initial,
        tuple(layer_call(index) for index in range(4)),
    )

    assert len(trace.dsa_outputs) == 4
    assert len(trace.layer_outputs) == 4
    assert len(trace.bridge_outputs) == 3
    assert torch.equal(observed_inputs[0], initial)
    for index in range(3):
        expected = trace.layer_outputs[index] * plan.layers[index].bridge_scale + plan.layers[index].bridge_bias
        assert torch.equal(trace.bridge_outputs[index], expected)
        assert torch.equal(observed_inputs[index + 1], expected)
    assert torch.equal(trace.final_output, trace.layer_outputs[-1])
    assert float(trace.final_output.abs().max()) > 0.0


def test_deterministic_chain_rejects_call_count_and_bad_output_contract() -> None:
    plan = build_a3_native_capsule_compare_plan(_config())
    initial = torch.zeros((2, 3))

    with pytest.raises(ValueError, match="4 layer calls"):
        execute_deterministic_capsule_chain(plan, initial, (lambda value: value,))

    calls = [lambda value: value.clone() for _ in range(4)]
    calls[2] = lambda value: torch.zeros((1,), dtype=value.dtype)
    with pytest.raises(ValueError, match="match its input"):
        execute_deterministic_capsule_chain(plan, initial, calls)


@pytest.mark.parametrize("num_layers", (1, 2, 4))
def test_fidelity_chain_enforces_real_hc_tensor_contract_for_every_depth(num_layers: int) -> None:
    plan = build_a3_native_capsule_compare_plan(_config(num_layers=num_layers))
    shells = tuple(_RecordingHCShell() for _ in range(num_layers))
    dsa_inputs: list[torch.Tensor] = []

    def dsa_call(value: torch.Tensor) -> torch.Tensor:
        dsa_inputs.append(value)
        return torch.full_like(value, 0.125)

    initial = torch.zeros((2, 4, 4096), dtype=torch.bfloat16)
    trace = execute_hc_fidelity_capsule_chain(
        plan,
        initial,
        (dsa_call,) * num_layers,
        tuple(shell.ops for shell in shells),
    )

    assert len(trace.hc_layers) == len(trace.dsa_outputs) == len(trace.layer_outputs) == num_layers
    assert len(trace.bridge_outputs) == num_layers - 1
    assert trace.final_output.shape == initial.shape
    assert all(value.shape == (2, 4096) for value in dsa_inputs)
    assert all(value.dtype is torch.bfloat16 for value in dsa_inputs)
    assert all(hc.input_value.shape == (2, 4, 4096) for hc in trace.hc_layers)
    assert all(hc.post.shape == (2, 4) for hc in trace.hc_layers)
    assert all(hc.comb.shape == (2, 4, 4) for hc in trace.hc_layers)


def test_fidelity_local_oracle_shares_exact_normalized_and_hc_auxiliary_objects() -> None:
    plan = build_a3_native_capsule_compare_plan(_config(num_layers=2, seed=19))
    shells = (_RecordingHCShell(), _RecordingHCShell())
    oracle_inputs: list[torch.Tensor] = []
    candidate_inputs: list[torch.Tensor] = []

    def oracle_call(value: torch.Tensor) -> torch.Tensor:
        oracle_inputs.append(value)
        return torch.full_like(value, 0.0625)

    def candidate_call(value: torch.Tensor) -> torch.Tensor:
        candidate_inputs.append(value)
        return torch.full_like(value, 0.125)

    local, candidate = execute_hc_fidelity_capsule_with_local_oracles(
        plan,
        torch.zeros((2, 4, 4096), dtype=torch.bfloat16),
        (oracle_call, oracle_call),
        (candidate_call, candidate_call),
        tuple(shell.ops for shell in shells),
    )

    assert local.hc_layers is candidate.hc_layers
    assert len(local.hc_layers) == 2
    for index, (oracle_input, candidate_input, hc_trace, shell) in enumerate(
        zip(oracle_inputs, candidate_inputs, local.hc_layers, shells, strict=True)
    ):
        assert oracle_input is candidate_input is hc_trace.normalized
        assert len(shell.hc_post_arguments) == 2
        oracle_post_args, candidate_post_args = shell.hc_post_arguments
        assert oracle_post_args[0] is local.dsa_outputs[index]
        assert candidate_post_args[0] is candidate.dsa_outputs[index]
        assert oracle_post_args[1] is candidate_post_args[1] is hc_trace.residual
        assert oracle_post_args[2] is candidate_post_args[2] is hc_trace.post
        assert oracle_post_args[3] is candidate_post_args[3] is hc_trace.comb
    assert not torch.equal(local.layer_outputs[0], candidate.layer_outputs[0])
    assert candidate.final_output.shape == (2, 4, 4096)


def test_fidelity_shell_rejects_flat_input_and_malformed_hc_pre_outputs() -> None:
    plan = build_a3_native_capsule_compare_plan(_config(num_layers=1))
    shell = _RecordingHCShell()
    call = lambda value: value.clone()
    with pytest.raises(ValueError, match="fidelity input"):
        execute_hc_fidelity_capsule_chain(
            plan,
            torch.zeros((2, 4096), dtype=torch.bfloat16),
            (call,),
            (shell.ops,),
        )

    broken = HCFidelityLayerOps(
        hc_pre=lambda value: (value[:, 0, :],),  # type: ignore[arg-type,return-value]
        rms_norm=call,
        hc_post=lambda value, residual, post, comb: residual,
    )
    with pytest.raises(TypeError, match="exactly"):
        execute_hc_fidelity_capsule_chain(
            plan,
            torch.zeros((2, 4, 4096), dtype=torch.bfloat16),
            (call,),
            (broken,),
        )


def test_local_oracle_consumes_each_actual_candidate_input_without_forming_its_own_chain() -> None:
    plan = build_a3_native_capsule_compare_plan(_config(seed=19))
    oracle_inputs: list[torch.Tensor] = []
    candidate_inputs: list[torch.Tensor] = []

    def oracle_call(index: int):
        def call(value: torch.Tensor) -> torch.Tensor:
            oracle_inputs.append(value.clone())
            return torch.full_like(value, (index + 1) / 32.0)

        return call

    def candidate_call(index: int):
        def call(value: torch.Tensor) -> torch.Tensor:
            candidate_inputs.append(value.clone())
            return torch.full_like(value, (index + 1) / 16.0)

        return call

    local, candidate = execute_capsule_with_local_oracles(
        plan,
        torch.full((2, 3), 0.25),
        tuple(oracle_call(index) for index in range(4)),
        tuple(candidate_call(index) for index in range(4)),
    )

    assert len(local.dsa_outputs) == len(local.layer_outputs) == 4
    assert len(candidate.dsa_outputs) == len(candidate.layer_outputs) == 4
    assert all(torch.equal(oracle, actual) for oracle, actual in zip(oracle_inputs, candidate_inputs, strict=True))
    assert torch.equal(local.layer_outputs[0], oracle_inputs[0] + local.dsa_outputs[0])
    assert torch.equal(candidate.layer_outputs[0], candidate_inputs[0] + candidate.dsa_outputs[0])
    assert not torch.equal(oracle_inputs[1], local.layer_outputs[0])
    expected_next = candidate.layer_outputs[0] * plan.layers[0].bridge_scale + plan.layers[0].bridge_bias
    assert torch.equal(candidate_inputs[1], expected_next)


def test_local_oracle_runner_rejects_mismatched_call_contracts() -> None:
    plan = build_a3_native_capsule_compare_plan(_config())
    calls = tuple(lambda value: value.clone() for _ in range(4))
    with pytest.raises(ValueError, match="local-oracle and candidate"):
        execute_capsule_with_local_oracles(plan, torch.zeros((2, 3)), calls[:3], calls)

    broken = list(calls)
    broken[1] = lambda value: torch.zeros((1,), dtype=value.dtype)
    with pytest.raises(ValueError, match="candidate CSA output"):
        execute_capsule_with_local_oracles(plan, torch.zeros((2, 3)), calls, broken)


def test_four_unique_prefixes_receive_the_complete_real_quant_description() -> None:
    prefixes = tuple(f"capsule.layer.{index}" for index in range(4))
    combined = ratio4_quant_description_for_prefixes(prefixes)

    assert combined["model_quant_type"] == "W8A8_DYNAMIC"
    for prefix in prefixes:
        single = ratio4_quant_description(prefix)
        for name, method in single.items():
            assert combined[name] == method
    assert len(combined) == 1 + 11 * 4
    with pytest.raises(ValueError, match="unique"):
        ratio4_quant_description_for_prefixes((prefixes[0], prefixes[0]))


@pytest.mark.parametrize(
    "topology",
    (CapsuleTopology.PPPP, CapsuleTopology.NPNP, CapsuleTopology.PNPN),
)
def test_resource_evidence_requires_independent_weights_and_twelve_cache_worlds(
    topology: CapsuleTopology,
) -> None:
    plan = build_a3_native_capsule_compare_plan(_config(candidate_topology=topology))
    rows = _evidence(topology)

    validate_native_capsule_resource_independence(plan, rows)

    assert len({row.attention_owner_id for row in rows}) == 4
    pypto_rows = tuple(row for row in rows if row.candidate_backend is BackendKind.PYPTO)
    assert len({row.prepared_weight_owner_id for row in pypto_rows}) == len(pypto_rows)
    assert len({row.pypto_device_owner_id for row in pypto_rows}) == 1
    assert len({address for row in pypto_rows for address in row.packed_weight_addresses}) == sum(
        len(row.packed_weight_addresses) for row in pypto_rows
    )
    assert len({row.shared_static_argument_addresses for row in pypto_rows}) <= 1
    assert len({address for row in rows for address in row.native_cache_addresses}) == 24
    assert len({address for row in rows for address in row.local_oracle_cache_addresses}) == 24
    assert len({address for row in rows for address in row.candidate_cache_addresses}) == 24
    assert (
        len(
            {
                address
                for row in rows
                for addresses in (
                    row.native_cache_addresses,
                    row.local_oracle_cache_addresses,
                    row.candidate_cache_addresses,
                )
                for address in addresses
            }
        )
        == 72
    )


def test_resource_evidence_rejects_parameter_cache_and_runtime_owner_aliases() -> None:
    plan = build_a3_native_capsule_compare_plan(_config())

    rows = list(_evidence())
    rows[1] = replace(rows[1], parameter_addresses=(rows[0].parameter_addresses[0],))
    with pytest.raises(ValueError, match="parameter storage"):
        validate_native_capsule_resource_independence(plan, rows)

    rows = list(_evidence())
    rows[0] = replace(rows[0], parameter_addresses=(rows[0].parameter_addresses[0],) * 2)
    with pytest.raises(ValueError, match="aliases its own real parameter"):
        validate_native_capsule_resource_independence(plan, rows)

    rows = list(_evidence())
    rows[1] = replace(rows[1], prepared_weight_owner_id=rows[0].attention_owner_id)
    with pytest.raises(ValueError, match="owners must be independent"):
        validate_native_capsule_resource_independence(plan, rows)

    rows = list(_evidence())
    rows[2] = replace(rows[2], candidate_cache_addresses=rows[0].native_cache_addresses)
    with pytest.raises(ValueError, match="cache storage aliases"):
        validate_native_capsule_resource_independence(plan, rows)

    rows = list(_evidence())
    rows[2] = replace(rows[2], local_oracle_cache_addresses=rows[0].candidate_cache_addresses)
    with pytest.raises(ValueError, match="cache storage aliases"):
        validate_native_capsule_resource_independence(plan, rows)

    rows = list(_evidence())
    rows[3] = replace(rows[3], pypto_backend_owner_id=9999)
    with pytest.raises(ValueError, match="share one prepared pypto_backend_owner_id"):
        validate_native_capsule_resource_independence(plan, rows)

    rows = list(_evidence())
    rows[2] = replace(
        rows[2],
        packed_weight_addresses=(rows[0].packed_weight_addresses[0],),
    )
    with pytest.raises(ValueError, match="alias packed weight storage"):
        validate_native_capsule_resource_independence(plan, rows)

    rows = list(_evidence())
    rows[1] = replace(rows[1], shared_static_argument_addresses=(9001, 9002, 9003))
    with pytest.raises(ValueError, match="share the same immutable"):
        validate_native_capsule_resource_independence(plan, rows)


def test_trace_comparison_reports_every_layer_output_and_six_state_families() -> None:
    plan = build_a3_native_capsule_compare_plan(_config())
    initial = torch.full((2, 3), 0.25)
    calls = tuple((lambda value, index=index: value * 0.0 + (index + 1) / 8.0) for index in range(4))
    native_trace = execute_deterministic_capsule_chain(plan, initial.clone(), calls)
    local_oracle_trace = CapsuleLocalOracleTrace(
        dsa_outputs=native_trace.dsa_outputs,
        layer_outputs=native_trace.layer_outputs,
    )
    pypto_trace = execute_deterministic_capsule_chain(plan, initial.clone(), calls)
    initial_states = _states(changed=0.0)
    native_states = _states(changed=1.0)
    pypto_states = _states(changed=1.0)

    comparisons = compare_native_capsule_traces(
        plan,
        local_oracle_trace,
        pypto_trace,
        native_states,
        pypto_states,
        initial_states,
    )

    assert len(comparisons) == 4
    assert all(comparison.close for comparison in comparisons)
    assert all(tuple(comparison.states) == STATE_NAMES for comparison in comparisons)
    assert all(comparison.dsa_output.close for comparison in comparisons)
    assert all(comparison.layer_output.close for comparison in comparisons)

    broken_states = list(pypto_states)
    broken_states[2] = dict(broken_states[2])
    broken_states[2][STATE_NAMES[2]] = torch.tensor([9.0])
    broken = compare_native_capsule_traces(
        plan,
        local_oracle_trace,
        pypto_trace,
        native_states,
        broken_states,
        initial_states,
    )
    assert broken[0].close
    assert broken[1].close
    assert not broken[2].close
    assert broken[3].close


def test_result_schema_preserves_bridge_boundary_and_layer_failures() -> None:
    good_layer = A3NativeCapsuleLayerComparison(
        layer_index=0,
        layer_name="layer.0",
        dsa_output=_comparison(),
        layer_output=_comparison(),
        states={name: _comparison() for name in STATE_NAMES},
        indexer_quantized=_indexer_quantized(),
    )
    bad_layer = replace(good_layer, layer_index=1, layer_name="layer.1", dsa_output=_comparison(close=False))
    result = A3NativeCapsuleCompareResult(
        runtime="host_build_graph",
        device=0,
        batch=4,
        num_layers=4,
        native_topology="NNNN",
        candidate_topology="PPPP",
        shell_kind=CapsuleShellKind.DETERMINISTIC_DIAGNOSTIC.value,
        local_oracle_same_candidate_inputs=True,
        candidate_sacrificial_warmup_reset=True,
        local_atol=0.1,
        local_rtol=0.1,
        chain_atol=0.2,
        chain_rtol=0.1,
        layers=(good_layer, bad_layer, replace(good_layer, layer_index=2), replace(good_layer, layer_index=3)),
        final_output=_comparison(),
        resource_evidence=_evidence(),
        real_hc_evidence=(),
    )

    payload = result.to_dict()

    assert not result.close
    assert payload["deterministic_bridge_only"] is True
    assert payload["shell_kind"] == "deterministic_diagnostic"
    assert payload["real_hc_complete"] is False
    assert payload["shell_evidence_complete"] is True
    assert payload["real_hc_evidence"] == ()
    assert payload["local_oracle_same_candidate_inputs"] is True
    assert payload["local_atol"] == 0.1
    assert payload["chain_atol"] == 0.2
    assert payload["native_topology"] == "NNNN"
    assert payload["candidate_topology"] == "PPPP"
    assert payload["layers"][1]["close"] is False  # type: ignore[index]
    json.dumps(payload)


def test_fidelity_result_requires_and_reports_complete_real_hc_evidence() -> None:
    good_layer = A3NativeCapsuleLayerComparison(
        layer_index=0,
        layer_name="model.layers.2.self_attn",
        dsa_output=_comparison(),
        layer_output=_comparison(),
        states={name: _comparison() for name in STATE_NAMES},
        indexer_quantized=_indexer_quantized(),
    )
    evidence = _hc_evidence(num_layers=1)
    result = A3NativeCapsuleCompareResult(
        runtime="tensormap_and_ringbuffer",
        device=0,
        batch=4,
        num_layers=1,
        native_topology="NNNN",
        candidate_topology="PPPP",
        shell_kind=CapsuleShellKind.FIDELITY.value,
        local_oracle_same_candidate_inputs=True,
        candidate_sacrificial_warmup_reset=True,
        local_atol=0.1,
        local_rtol=0.1,
        chain_atol=0.1,
        chain_rtol=0.1,
        layers=(good_layer,),
        final_output=_comparison(),
        resource_evidence=_evidence(num_layers=1),
        real_hc_evidence=evidence,
    )

    payload = result.to_dict()

    assert result.close
    assert result.real_hc_complete
    assert not result.deterministic_bridge_only
    assert payload["shell_kind"] == "fidelity"
    assert payload["topology_evidence_complete"] is True
    assert payload["resource_evidence_complete"] is True
    assert payload["real_hc_complete"] is True
    assert payload["shell_evidence_complete"] is True
    assert payload["execution_contract_complete"] is True
    assert payload["real_hc_evidence"][0]["hc_pre_operator"].endswith("npu_hc_pre_v2")  # type: ignore[index]
    assert payload["real_hc_evidence"][0]["rms_norm_operator"] == "AscendRMSNorm"  # type: ignore[index]
    assert payload["real_hc_evidence"][0]["hc_post_operator"].endswith("npu_hc_post")  # type: ignore[index]

    incomplete = replace(result, real_hc_evidence=())
    assert not incomplete.real_hc_complete
    assert not incomplete.close


def test_result_close_rejects_missing_or_inconsistent_contract_evidence() -> None:
    result = _good_fidelity_result(num_layers=2)

    assert result.close
    assert result.topology_evidence_complete
    assert result.resource_evidence_complete
    assert result.execution_contract_complete

    missing_resources = replace(result, resource_evidence=())
    assert not missing_resources.topology_evidence_complete
    assert not missing_resources.resource_evidence_complete
    assert not missing_resources.close

    wrong_native_topology = replace(result, native_topology="PPPP")
    assert not wrong_native_topology.topology_evidence_complete
    assert not wrong_native_topology.close

    wrong_candidate_topology = replace(result, candidate_topology="NNNN")
    assert not wrong_candidate_topology.topology_evidence_complete
    assert not wrong_candidate_topology.close

    assert not replace(result, local_oracle_same_candidate_inputs=False).close
    assert not replace(result, candidate_sacrificial_warmup_reset=False).close
    assert not replace(result, local_oracle_same_candidate_inputs=1).close  # type: ignore[arg-type]
    assert not replace(result, candidate_sacrificial_warmup_reset=1).close  # type: ignore[arg-type]

    one_layer_result = _good_fidelity_result(num_layers=1)
    assert not replace(one_layer_result, num_layers=True).topology_evidence_complete  # type: ignore[arg-type]
    assert not replace(one_layer_result, num_layers=True).close  # type: ignore[arg-type]

    mismatched_resources = list(result.resource_evidence)
    mismatched_resources[1] = replace(mismatched_resources[1], layer_name="wrong.layer")
    mismatched_result = replace(result, resource_evidence=tuple(mismatched_resources))
    assert mismatched_result.topology_evidence_complete
    assert not mismatched_result.resource_evidence_complete
    assert not mismatched_result.close

    mismatched_hc = list(result.real_hc_evidence)
    mismatched_hc[1] = replace(mismatched_hc[1], layer_index=0)
    mismatched_hc_result = replace(result, real_hc_evidence=tuple(mismatched_hc))
    assert not mismatched_hc_result.real_hc_complete
    assert not mismatched_hc_result.close

    incomplete_states = dict(result.layers[0].states)
    incomplete_states.pop(STATE_NAMES[-1])
    bad_layers = (replace(result.layers[0], states=incomplete_states), *result.layers[1:])
    assert not replace(result, layers=bad_layers).execution_contract_complete
    assert not replace(result, layers=bad_layers).close


class _FakeCleanupOwner:
    def __init__(self, *, close_error: BaseException | None = None) -> None:
        self.close_error = close_error
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1
        if self.close_error is not None:
            raise self.close_error


def _fake_cleanup_layer(owner: _FakeCleanupOwner):
    return SimpleNamespace(
        plan=SimpleNamespace(layer_index=0),
        wrapper=object(),
        owner=SimpleNamespace(device_owner=owner),
    )


def test_cleanup_closes_only_after_quiescence_and_destroys_partial_tp_init() -> None:
    owner = _FakeCleanupOwner()
    layer = _fake_cleanup_layer(owner)
    events: list[str] = []
    outcome = _cleanup_a3_native_capsule_compare(
        tp_initialized=True,
        tp_initialization_attempted=True,
        layers=(layer,),  # type: ignore[arg-type]
        fallback_owner=None,
        get_owner=lambda: owner,
        synchronize=lambda: events.append("sync"),
        uninstall=lambda wrapper: events.append("uninstall"),
        close_config=lambda: events.append("config"),
        destroy_tp1=lambda: events.append("destroy"),
    )

    assert outcome.error is None
    assert outcome.owner_closed
    assert outcome.retry_owner is None
    assert owner.close_calls == 1
    assert events == ["sync", "uninstall", "config", "destroy"]

    partial_events: list[str] = []
    partial = _cleanup_a3_native_capsule_compare(
        tp_initialized=False,
        tp_initialization_attempted=True,
        layers=(),
        fallback_owner=None,
        get_owner=lambda: None,
        synchronize=lambda: partial_events.append("unexpected-sync"),
        uninstall=lambda wrapper: partial_events.append("unexpected-uninstall"),
        close_config=lambda: partial_events.append("config"),
        destroy_tp1=lambda: partial_events.append("destroy"),
    )
    assert partial.error is None
    assert partial_events == ["config", "destroy"]


def test_cleanup_sync_failure_retains_owner_and_never_attempts_unsafe_close() -> None:
    owner = _FakeCleanupOwner()
    layer = _fake_cleanup_layer(owner)
    sync_error = RuntimeError("device did not quiesce")
    events: list[str] = []

    def fail_sync() -> None:
        events.append("sync")
        raise sync_error

    outcome = _cleanup_a3_native_capsule_compare(
        tp_initialized=True,
        tp_initialization_attempted=True,
        layers=(layer,),  # type: ignore[arg-type]
        fallback_owner=None,
        get_owner=lambda: owner,
        synchronize=fail_sync,
        uninstall=lambda wrapper: events.append("unexpected-uninstall"),
        close_config=lambda: events.append("config"),
        destroy_tp1=lambda: events.append("destroy"),
    )

    assert outcome.error is sync_error
    assert outcome.retry_owner is owner
    assert owner.close_calls == 0
    assert events == ["sync", "config", "destroy"]
    with pytest.raises(RuntimeError, match="did not quiesce") as raised:
        _apply_capsule_cleanup_outcome(None, outcome)
    assert raised.value.decode_csa_cleanup_owner is owner  # type: ignore[attr-defined]


def test_cleanup_uninstall_failure_retains_owner_and_skips_close() -> None:
    owner = _FakeCleanupOwner()
    layer = _fake_cleanup_layer(owner)
    uninstall_error = RuntimeError("dispatch hook remains attached")
    events: list[str] = []

    def fail_uninstall(_wrapper: object) -> None:
        events.append("uninstall")
        raise uninstall_error

    outcome = _cleanup_a3_native_capsule_compare(
        tp_initialized=True,
        tp_initialization_attempted=True,
        layers=(layer,),  # type: ignore[arg-type]
        fallback_owner=None,
        get_owner=lambda: owner,
        synchronize=lambda: events.append("sync"),
        uninstall=fail_uninstall,
        close_config=lambda: events.append("config"),
        destroy_tp1=lambda: events.append("destroy"),
    )

    assert outcome.error is uninstall_error
    assert outcome.retry_owner is owner
    assert owner.close_calls == 0
    assert events == ["sync", "uninstall", "config", "destroy"]


def test_cleanup_close_failure_keeps_primary_error_and_retryable_owner() -> None:
    close_error = RuntimeError("retry close")
    owner = _FakeCleanupOwner(close_error=close_error)
    layer = _fake_cleanup_layer(owner)
    primary_error = RuntimeError("prepare failed")
    primary_error.decode_csa_cleanup_owner = owner  # type: ignore[attr-defined]
    events: list[str] = []
    outcome = _cleanup_a3_native_capsule_compare(
        tp_initialized=True,
        tp_initialization_attempted=True,
        layers=(layer,),  # type: ignore[arg-type]
        fallback_owner=primary_error.decode_csa_cleanup_owner,  # type: ignore[attr-defined]
        get_owner=lambda: None,
        synchronize=lambda: events.append("sync"),
        uninstall=lambda wrapper: events.append("uninstall"),
        close_config=lambda: events.append("config"),
        destroy_tp1=lambda: events.append("destroy"),
    )

    assert outcome.error is close_error
    assert outcome.retry_owner is owner
    assert owner.close_calls == 1
    assert events == ["sync", "uninstall", "config", "destroy"]
    _apply_capsule_cleanup_outcome(primary_error, outcome)
    assert primary_error.decode_csa_cleanup_owner is owner  # type: ignore[attr-defined]
    assert primary_error.decode_csa_cleanup_error is close_error  # type: ignore[attr-defined]


def test_parser_and_runner_guard_are_host_safe() -> None:
    args = build_parser().parse_args(["--runtime", "tensormap_and_ringbuffer"])

    assert args.device == 0
    assert args.batch == 4
    assert args.num_layers == 4
    assert args.candidate_topology == "PPPP"
    assert args.shell_kind == "fidelity"
    assert args.start_position == 0
    assert args.chain_atol is None
    assert args.chain_rtol == 0.1
    assert args.master_port == 29671
    with pytest.raises(TypeError, match="A3NativeCapsuleCompareConfig"):
        run_a3_native_capsule_compare(object())  # type: ignore[arg-type]
