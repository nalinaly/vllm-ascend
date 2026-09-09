# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Hash paired CSA fixtures outside timing, then run the unchanged benchmark.

This diagnostic records logical tensor bytes, dtype, shape and strides. It
does not modify weights, caches, callbacks, graph capture or sample scheduling.
Its timing samples are not part of the three baseline reproduction processes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from unittest.mock import patch


def tensor_fingerprint(tensor):
    import torch

    logical = tensor.detach().cpu().contiguous()
    return {
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "dtype": str(tensor.dtype),
        "logical_sha256": hashlib.sha256(logical.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest(),
        "nonzero_elements": int(torch.count_nonzero(logical)),
        "elements": logical.numel(),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--master-port", type=int, default=29853)
    args = parser.parse_args(argv)
    from . import a3_single_op_benchmark as benchmark

    # Validate the queue setting before importing the fixture (which imports torch).
    benchmark._validate_task_queue_environment("aclgraph")
    from . import native_fixture

    config = benchmark.A3SingleOpBenchmarkConfig(
        runtime="tensormap_and_ringbuffer",
        mode="aclgraph",
        device=0,
        batch=4,
        start_position=8191,
        seed=20260902,
        master_port=args.master_port,
        warmup_iterations=20,
        sample_iterations=100,
        enqueue_batch_size=20,
        atol=0.001,
        rtol=0.01,
    )
    layer_factory = native_fixture.build_real_ratio4_attention
    world_factory = native_fixture.build_twin_decode_csa_state_worlds
    case_factory = benchmark._DeviceCase
    weights, worlds, inputs = [], [], []

    def audited_layer(*positional, **keyword):
        layer = layer_factory(*positional, **keyword)
        tensors = dict(layer.named_parameters())
        tensors.update(dict(layer.named_buffers()))
        weights.append({name: tensor_fingerprint(tensor) for name, tensor in tensors.items()})
        if len(weights) == 2:
            assert weights[0] == weights[1], "paired production weights/buffers are not identical"
        return layer

    def audited_worlds(*positional, **keyword):
        pair = world_factory(*positional, **keyword)
        for world in pair:
            values = dict(zip(native_fixture.STATE_NAMES, world.raw_cache_tuple, strict=True))
            values.update({f"block_table.{key}": value for key, value in world.block_tables.items()})
            values.update({f"linear_slot.{key}": value for key, value in world.linear_slot_mappings.items()})
            values.update(
                {
                    key: getattr(world, key)
                    for key in ("positions", "start_positions", "seq_lens", "query_start_loc", "swa_slot_mapping")
                }
            )
            worlds.append({name: tensor_fingerprint(tensor) for name, tensor in values.items()})
        assert worlds[0] == worlds[1], "paired initial caches/metadata are not identical"
        assert all(
            a.data_ptr() != b.data_ptr() for a, b in zip(pair[0].raw_cache_tuple, pair[1].raw_cache_tuple, strict=True)
        )
        return pair

    def audited_case(*positional, **keyword):
        case = case_factory(*positional, **keyword)
        hidden = tensor_fingerprint(case.hidden_states)
        source = tensor_fingerprint(case.source_hidden_states)
        assert hidden == source, "captured hidden input differs from resident source input"
        inputs.append(hidden)
        if len(inputs) == 2:
            assert inputs[0] == inputs[1], "paired hidden inputs are not identical"
        return case

    with (
        patch.object(native_fixture, "build_real_ratio4_attention", audited_layer),
        patch.object(native_fixture, "build_twin_decode_csa_state_worlds", audited_worlds),
        patch.object(benchmark, "_DeviceCase", audited_case),
    ):
        report = benchmark.run_a3_single_op_benchmark(config)
    assert len(weights) == len(worlds) == len(inputs) == 2
    result = {
        "status": "ok",
        "all_logical_bytes_equal": True,
        "checks_outside_timing": True,
        "diagnostic_only_not_formal_baseline": True,
        "weights_and_buffers": weights[0],
        "initial_caches_and_metadata": worlds[0],
        "hidden_input": inputs[0],
        "mutable_cache_addresses_disjoint": True,
        "correctness": {
            key: json.loads(value)
            for key, value in report.metadata
            if key in ("pre_timing_correctness_gate", "post_timing_correctness_gate")
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "status": "ok",
                "weight_buffer_tensors": len(weights[0]),
                "cache_metadata_tensors": len(worlds[0]),
                "output": str(args.output),
            }
        )
    )


if __name__ == "__main__":
    main()
