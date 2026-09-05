# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Host contracts for the synthetic native/PyPTO CSA comparison fixture."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from tests.pypto_dsv4_decode_csa.a3_native_compare import (
    SUPPORTED_RUNTIMES,
    A3NativeCompareResult,
    TensorComparison,
    _forward_context,
    build_parser,
    compare_state_snapshots,
    compare_tensors,
    diagnose_state_snapshots,
    diagnose_state_transitions,
    expected_state_row_positions,
    expected_state_row_positions_for_worlds,
)
from tests.pypto_dsv4_decode_csa.indexer_quantized_comparison import compare_indexer_quantized
from tests.pypto_dsv4_decode_csa.native_fixture import (
    DEFAULT_LAYER_PREFIX,
    STATE_NAMES,
    _common_metadata,
    build_synthetic_vllm_config,
    build_twin_decode_csa_state_worlds,
    metadata_family_names,
    ratio4_quant_description,
    retarget_decode_csa_state_world,
)
from vllm_ascend.ops._pypto_dsv4_csa import DecodeCSAProgramSpec


def test_ratio4_quant_description_selects_four_real_w8_and_seven_float_linears() -> None:
    description = ratio4_quant_description()

    assert description["model_quant_type"] == "W8A8_DYNAMIC"
    values = tuple(value for name, value in description.items() if name != "model_quant_type")
    assert values.count("W8A8_DYNAMIC") == 4
    assert values.count("FLOAT") == 7
    assert description[f"{DEFAULT_LAYER_PREFIX}.indexer.wq_b.weight"] == "W8A8_DYNAMIC"
    assert description[f"{DEFAULT_LAYER_PREFIX}.indexer.weights_proj.weight"] == "FLOAT"


def test_metadata_names_are_the_native_filter_order() -> None:
    names = metadata_family_names()

    assert names == tuple(sorted(names))
    assert names == (
        f"{DEFAULT_LAYER_PREFIX}.attn",
        f"{DEFAULT_LAYER_PREFIX}.compressor.state_cache",
        f"{DEFAULT_LAYER_PREFIX}.indexer.compressor.state_cache",
        f"{DEFAULT_LAYER_PREFIX}.indexer.k_cache",
        f"{DEFAULT_LAYER_PREFIX}.swa_cache",
    )


def test_b4_twin_worlds_have_exact_a3_shapes_strides_tables_and_no_aliases() -> None:
    spec = DecodeCSAProgramSpec(batch=4)
    left, right = build_twin_decode_csa_state_worlds(spec, device="cpu")

    assert tuple(tensor.shape for tensor in left.raw_cache_tuple) == (
        torch.Size((256, 32, 1, 512)),
        torch.Size((512, 32, 1, 512)),
        torch.Size((260, 2, 1, 2048)),
        torch.Size((260, 2, 1, 512)),
        torch.Size((256, 32, 1, 128)),
        torch.Size((256, 32, 1, 1)),
    )
    assert tuple(left.raw_cache_tuple[2].stride()) == (8192, 2048, 2048, 1)
    assert tuple(left.raw_cache_tuple[3].stride()) == (1040, 512, 512, 1)
    assert tuple(left.raw_cache_tuple[4].stride()) == (4160, 128, 128, 1)
    assert tuple(left.raw_cache_tuple[5].stride()) == (2080, 1, 1, 1)
    assert {name: tuple(table.shape) for name, table in left.block_tables.items()} == {
        "compressed": (4, 128),
        "main_state": (4, 8192),
        "inner_state": (4, 8192),
        "indexer": (4, 128),
        "swa": (4, 512),
    }
    assert tuple(left.swa_slot_mapping.shape) == (32, 2)
    assert tuple(left.query_start_loc.tolist()) == (0, 8, 16, 24, 32)
    assert tuple(left.seq_lens.tolist()) == (8, 8, 8, 8)
    for left_tensor, right_tensor in zip(left.raw_cache_tuple, right.raw_cache_tuple, strict=True):
        assert left_tensor.data_ptr() != right_tensor.data_ptr()
        torch.testing.assert_close(left_tensor, right_tensor)


def test_real_synthetic_vllm_config_uses_flash_ratio4_tp1_and_modelslim() -> None:
    owner = build_synthetic_vllm_config(batch=4)
    try:
        config = owner.vllm_config
        assert type(config).__name__ == "VllmConfig"
        assert type(config.model_config).__name__ == "ModelConfig"
        assert type(owner.quant_config).__name__ == "AscendModelSlimConfig"
        assert config.model_config.hf_config.model_type == "deepseek_v4"
        assert config.model_config.hf_config.compress_ratios[2] == 4
        assert config.parallel_config.tensor_parallel_size == 1
        assert config.cache_config.block_size == 32
        assert config.additional_config["multistream_dsv4_dsa_overlap"] is False
    finally:
        owner.close()


def test_common_metadata_keeps_is_prefilling_with_cpu_query_lengths() -> None:
    world, _ = build_twin_decode_csa_state_worlds(DecodeCSAProgramSpec(batch=4), device="cpu")

    common = _common_metadata(world, family="compressed", block_table=world.block_tables["compressed"])

    assert common.query_start_loc_cpu.device.type == "cpu"
    assert common.is_prefilling.device.type == "cpu"


def test_active_union_comparison_ignores_equal_untouched_tail_and_detects_integer_error() -> None:
    initial = torch.zeros(8, dtype=torch.float32)
    reference = initial.clone()
    candidate = initial.clone()
    reference[2] = 2.0
    candidate[2] = 2.01

    close = compare_tensors(reference, candidate, initial=initial, atol=0.02, rtol=0.0)
    assert close.active_elements == 1
    assert close.total_elements == 8
    assert close.close

    initial_i8 = torch.zeros(8, dtype=torch.int8)
    reference_i8 = initial_i8.clone()
    candidate_i8 = initial_i8.clone()
    reference_i8[3] = 1
    candidate_i8[3] = 2
    mismatch = compare_tensors(reference_i8, candidate_i8, initial=initial_i8, atol=99.0, rtol=99.0)
    assert mismatch.active_elements == 1
    assert not mismatch.close


def test_six_state_comparison_preserves_family_names() -> None:
    initial = {name: torch.zeros(2) for name in STATE_NAMES}
    reference = {name: value.clone() for name, value in initial.items()}
    candidate = {name: value.clone() for name, value in initial.items()}
    for value in reference.values():
        value[0] = 1.0
    for value in candidate.values():
        value[0] = 1.0

    result = compare_state_snapshots(reference, candidate, initial, atol=0.0, rtol=0.0)

    assert tuple(result) == STATE_NAMES
    assert all(comparison.active_elements == 1 and comparison.close for comparison in result.values())


def test_state_diagnostics_split_write_sets_segments_and_indexer_dequant() -> None:
    initial = {
        "compressed_kv": torch.zeros(2, 512),
        "swa_kv": torch.zeros(2, 512),
        "compressor_state": torch.zeros(2, 4),
        "indexer_compressor_state": torch.zeros(2, 4),
        "indexer_k": torch.zeros(2, 128, dtype=torch.int8),
        "indexer_scale": torch.ones(2, 1, dtype=torch.float16),
    }
    reference = {name: value.clone() for name, value in initial.items()}
    candidate = {name: value.clone() for name, value in initial.items()}
    for name in STATE_NAMES:
        reference[name][0, 0] = 2
        candidate[name][0, 0] = 1
    reference["indexer_scale"][0, 0] = 0.1
    candidate["indexer_scale"][0, 0] = 0.25

    diagnostics = diagnose_state_snapshots(reference, candidate, initial)

    assert tuple(diagnostics) == STATE_NAMES
    assert diagnostics["swa_kv"]["write_intersection_rows"] == 1
    assert diagnostics["swa_kv"]["reference_only_rows"] == 0
    assert diagnostics["swa_kv"]["segments"]["nope_0_448"]["max_abs_error"] == 1.0
    assert diagnostics["swa_kv"]["segments"]["rope_448_512"]["max_abs_error"] == 0.0
    assert diagnostics["indexer_k"]["dequantized_int8_x_scale"]["max_abs_error"] > 0.0


def test_state_transition_diagnostics_separate_target_and_outside_rows() -> None:
    world, _ = build_twin_decode_csa_state_worlds(DecodeCSAProgramSpec(batch=4), device="cpu")
    initial = world.snapshot()
    current = {name: tensor.clone() for name, tensor in initial.items()}
    current["swa_kv"].reshape(-1, 512)[0, 0] = 1
    current["swa_kv"].reshape(-1, 512)[100, 0] = 2

    transition = diagnose_state_transitions(initial, (current,), (world,))[0]["swa_kv"]

    assert transition["changed_rows"] == 2
    assert transition["changed_outside_target_rows"] == 1
    assert transition["outside_target_row_indices"] == (100,)


def test_expected_state_rows_reverse_to_fixture_token_positions() -> None:
    world, _ = build_twin_decode_csa_state_worlds(DecodeCSAProgramSpec(batch=4), device="cpu")

    positions = expected_state_row_positions(world)

    assert tuple(item["absolute_position"] for item in positions["compressed_kv"][0]) == (0, 1, 2, 3)
    assert tuple(item["absolute_position"] for item in positions["swa_kv"][0]) == (0,)


def test_retarget_world_reuses_cache_storage_but_builds_next_step_metadata() -> None:
    spec = DecodeCSAProgramSpec(batch=4)
    world, _ = build_twin_decode_csa_state_worlds(spec, device="cpu")

    next_world = retarget_decode_csa_state_world(
        world,
        start_positions=(8,) * spec.batch,
    )

    assert tuple(next_world.positions[0].tolist()) == tuple(range(8, 16))
    assert tuple(next_world.seq_lens.tolist()) == (16,) * spec.batch
    assert tuple(next_world.swa_slot_mapping[: spec.seq, 0].tolist()) == (0,) * spec.seq
    assert tuple(next_world.swa_slot_mapping[: spec.seq, 1].tolist()) == tuple(range(8, 16))
    assert tuple(next_world.linear_slot_mappings["swa"][: spec.seq].tolist()) == tuple(range(8, 16))
    assert tuple(world.positions[0].tolist()) == tuple(range(8))
    for original, retargeted in zip(
        world.raw_cache_tuple,
        next_world.raw_cache_tuple,
        strict=True,
    ):
        assert original.data_ptr() == retargeted.data_ptr()
    for name in world.block_tables:
        assert world.block_tables[name].data_ptr() == next_world.block_tables[name].data_ptr()


def test_multi_step_expected_rows_preserve_decode_step_provenance() -> None:
    spec = DecodeCSAProgramSpec(batch=4)
    first, _ = build_twin_decode_csa_state_worlds(spec, device="cpu")
    second = retarget_decode_csa_state_world(first, start_positions=(8,) * spec.batch)

    positions = expected_state_row_positions_for_worlds((first, second))

    assert positions["swa_kv"][0] == (
        {
            "request": 0,
            "token_offset": 0,
            "absolute_position": 0,
            "decode_step": 0,
        },
    )
    assert positions["swa_kv"][8] == (
        {
            "request": 0,
            "token_offset": 0,
            "absolute_position": 8,
            "decode_step": 1,
        },
    )


def test_fresh_process_cli_requires_an_explicit_runtime() -> None:
    parser = build_parser()
    for runtime in SUPPORTED_RUNTIMES:
        args = parser.parse_args(["--runtime", runtime])
        assert args.runtime == runtime
        assert args.device == 0
        assert args.batch == 4
        assert args.start_position == 0
        assert args.correctness_steps == 1
        assert args.sync_between_correctness_steps is False
        assert args.diagnose_step_states is False


def test_forward_context_carries_production_num_tokens_proxy_value() -> None:
    wrapper = SimpleNamespace(prefix=DEFAULT_LAYER_PREFIX)
    metadata = SimpleNamespace(
        by_name={"family": object()},
        metadata=(SimpleNamespace(num_actual_tokens=32),),
    )

    context = _forward_context(wrapper, metadata)

    assert context.num_tokens == 32


def test_result_json_payload_handles_immutable_state_mapping() -> None:
    comparison = TensorComparison(1, 1, 0.0, 0.0, 0.0, 1.0, 1.0, True, True)
    indexer_k = torch.ones((1, 128), dtype=torch.int8)
    initial_k = torch.zeros_like(indexer_k)
    indexer_scale = torch.ones((1, 1), dtype=torch.float16)
    initial_scale = torch.zeros_like(indexer_scale)
    indexer_quantized = compare_indexer_quantized(
        indexer_k,
        indexer_k.clone(),
        initial_k,
        indexer_scale,
        indexer_scale.clone(),
        initial_scale,
    )
    result = A3NativeCompareResult(
        runtime="tensormap_and_ringbuffer",
        device=0,
        batch=4,
        seq=8,
        start_position=0,
        correctness_steps=1,
        sync_between_correctness_steps=False,
        diagnose_step_states=False,
        pypto_sacrificial_warmup_reset=True,
        rope_dtype="fp32-production",
        native_ms=2.0,
        pypto_ms=1.0,
        speedup=2.0,
        output=comparison,
        states={name: comparison for name in STATE_NAMES},
        indexer_quantized=indexer_quantized,
        native_hot_operators=(),
    )

    payload = result.to_dict()

    assert payload["close"] is True
    assert tuple(payload["states"]) == STATE_NAMES
    assert payload["indexer_quantized"]["raw_exact"] is True


def test_result_keeps_raw_int8_failure_visible_but_uses_independent_joint_contract() -> None:
    close = TensorComparison(1, 1, 0.0, 0.0, 0.0, 1.0, 1.0, True, True)
    raw_int8_failure = TensorComparison(16384, 16384, 1.0, 1 / 16384, 1.0, 127.0, 127.0, True, False)
    initial_k = torch.zeros((128, 128), dtype=torch.int8)
    reference_k = torch.full_like(initial_k, 3)
    candidate_k = reference_k.clone()
    candidate_k[69, 54] += 1
    initial_scale = torch.zeros((128, 1), dtype=torch.float16)
    scale = torch.full_like(initial_scale, 0.019317626953125)
    indexer_quantized = compare_indexer_quantized(
        reference_k,
        candidate_k,
        initial_k,
        scale,
        scale.clone(),
        initial_scale,
    )
    states = {name: close for name in STATE_NAMES}
    states["indexer_k"] = raw_int8_failure
    result = A3NativeCompareResult(
        runtime="tensormap_and_ringbuffer",
        device=0,
        batch=4,
        seq=8,
        start_position=0,
        correctness_steps=128,
        sync_between_correctness_steps=False,
        diagnose_step_states=False,
        pypto_sacrificial_warmup_reset=True,
        rope_dtype="fp32-production",
        native_ms=2.0,
        pypto_ms=1.0,
        speedup=2.0,
        output=close,
        states=states,
        indexer_quantized=indexer_quantized,
        native_hot_operators=(),
    )

    payload = result.to_dict()

    assert payload["states"]["indexer_k"]["close"] is False
    assert payload["indexer_quantized"]["raw_exact"] is False
    assert payload["indexer_quantized"]["acceptable"] is True
    assert payload["close"] is True
