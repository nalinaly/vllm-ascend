# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Host contracts for the nonzero native/PyPTO ACLGraph runner."""

from __future__ import annotations

import inspect

import pytest
import torch

from tests.pypto_dsv4_decode_csa.a3_native_aclgraph_compare import (
    A3NativeACLGraphCompareResult,
    ACLGraphReplayComparison,
    _validate_run_arguments,
    build_parser,
    run_a3_native_aclgraph_compare,
)
from tests.pypto_dsv4_decode_csa.a3_native_compare import TensorComparison
from tests.pypto_dsv4_decode_csa.indexer_quantized_comparison import compare_indexer_quantized


def _comparison(*, close: bool) -> TensorComparison:
    return TensorComparison(
        active_elements=1,
        total_elements=1,
        max_abs_error=0.0,
        mean_abs_error=0.0,
        max_relative_error=0.0,
        reference_max_abs=1.0,
        candidate_max_abs=1.0,
        finite=True,
        close=close,
    )


def _exact_quantized_comparison():
    initial_k = torch.zeros((1, 4), dtype=torch.int8)
    initial_scale = torch.ones((1, 1), dtype=torch.float16)
    written_k = torch.tensor([[1, 2, 3, 4]], dtype=torch.int8)
    written_scale = torch.tensor([[0.25]], dtype=torch.float16)
    return compare_indexer_quantized(
        written_k,
        written_k,
        initial_k,
        written_scale,
        written_scale,
        initial_scale,
    )


def test_result_requires_output_and_every_state_on_every_replay() -> None:
    good = ACLGraphReplayComparison(
        input_scale=0.5,
        output=_comparison(close=True),
        states={"swa_kv": _comparison(close=True)},
        indexer_quantized=_exact_quantized_comparison(),
    )
    bad = ACLGraphReplayComparison(
        input_scale=-1.0,
        output=_comparison(close=True),
        states={"swa_kv": _comparison(close=False)},
        indexer_quantized=_exact_quantized_comparison(),
    )
    result = A3NativeACLGraphCompareResult(
        runtime="tensormap_and_ringbuffer",
        device=0,
        batch=4,
        seq=8,
        start_position=0,
        replay_count=2,
        native_graph_ms=2.0,
        pypto_graph_ms=1.0,
        speedup=2.0,
        native_and_pypto_addresses_disjoint=True,
        native_capture_addresses_differ_from_warmup=True,
        pypto_capture_addresses_differ_from_warmup=True,
        pypto_capture_bindings_retained=1,
        replays=(good, bad),
    )

    assert not result.close
    assert result.to_dict()["replays"][0]["close"] is True
    assert result.to_dict()["replays"][1]["close"] is False


def test_replay_acceptance_uses_joint_indexer_quantization_contract() -> None:
    initial_k = torch.zeros((1, 512), dtype=torch.int8)
    reference_k = torch.full((1, 512), 10, dtype=torch.int8)
    candidate_k = reference_k.clone()
    candidate_k[0, 7] = 11
    initial_scale = torch.ones((1, 1), dtype=torch.float16)
    scale = torch.tensor([[0.025]], dtype=torch.float16)
    quantized = compare_indexer_quantized(
        reference_k,
        candidate_k,
        initial_k,
        scale,
        scale,
        initial_scale,
    )
    assert not quantized.raw_exact
    assert quantized.acceptable

    replay = ACLGraphReplayComparison(
        input_scale=1.0,
        output=_comparison(close=True),
        states={
            "indexer_k": _comparison(close=False),
            "indexer_scale": _comparison(close=True),
            "swa_kv": _comparison(close=True),
        },
        indexer_quantized=quantized,
    )
    assert replay.close
    assert replay.to_dict()["indexer_quantized"]["raw_mismatch_elements"] == 1


def test_parser_exposes_fresh_process_graph_matrix() -> None:
    args = build_parser().parse_args(["--runtime", "tensormap_and_ringbuffer"])

    assert args.device == 0
    assert args.batch == 4
    assert args.start_position == 0
    assert tuple(args.replay_values) == (0.5, -1.0, 1.75)
    assert args.warmups == 3
    assert args.iterations == 10


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"runtime": "invalid"}, "runtime"),
        ({"batch": 8}, "B4"),
        ({"warmups": -1}, "warmups"),
        ({"iterations": 0}, "iterations"),
        ({"replay_values": ()}, "replay value"),
        ({"replay_values": (float("nan"),)}, "finite"),
    ),
)
def test_argument_validation_fails_before_device_bootstrap(overrides: dict[str, object], message: str) -> None:
    arguments: dict[str, object] = {
        "runtime": "tensormap_and_ringbuffer",
        "batch": 4,
        "warmups": 0,
        "iterations": 1,
        "replay_values": (1.0,),
    }
    arguments.update(overrides)

    with pytest.raises(ValueError, match=message):
        _validate_run_arguments(**arguments)


def test_capture_regions_are_single_custom_ops_and_graphs_die_before_runtime() -> None:
    source = inspect.getsource(run_a3_native_aclgraph_compare)

    assert source.count("torch_npu.npu.NPUGraph()") == 2
    assert source.count("with torch_npu.npu.graph(") == 2
    assert source.count("_custom_op_call(wrapper, native_context, native_hidden, native_output)") == 1
    assert source.count("_custom_op_call(wrapper, pypto_context, pypto_hidden, pypto_output)") == 1
    assert "torch.add(" not in source
    assert "torch.mul(" not in source
    assert source.index("graph.reset()") < source.index("owner.device_owner.backend.close()")
    assert "native_and_pypto_addresses_disjoint" in source
    assert "native_capture_addresses_differ_from_warmup" in source
    assert "pypto_capture_addresses_differ_from_warmup" in source
    assert "native_hidden.copy_(base_hidden)" in source
    assert "pypto_hidden.copy_(base_hidden)" in source
    assert "_restore_state_world(native_world, native_pristine)" in source
    assert "_restore_state_world(pypto_world, pypto_pristine)" in source
