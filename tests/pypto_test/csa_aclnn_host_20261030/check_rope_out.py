# SPDX-License-Identifier: Apache-2.0
"""单卡核对标准index_select(out=)的RoPE缓冲、草稿代际与replay。"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops import rope_dsv4 as rope


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.npu.set_device(args.device)
    torch.manual_seed(31)
    state = rope.RopeGlobalState()
    rope._ROPE_STATE = state
    key, layer = "case", "layer"
    state.full_rope_cache[key] = tuple(torch.randn((131136, 1, 1, 64), device="npu") for _ in range(2))
    state.registry_summary[key] = {"default"}
    state.layer_info[layer] = (key, ["default"])
    state.runtime_buffer[key] = {"default": tuple(torch.full((128, 1, 1, 64), -91., device="npu") for _ in range(2))}
    state.spec_runtime_buffer[key] = {"default": tuple(
        [torch.full((128, 1, 1, 64), -91., device="npu") for _ in range(2)] for _ in range(2)
    )}
    cases = [list(range(131040, 131136)), [0] * 6, [0, 1, 31, 127, 131071, 131072] + [0] * 27]
    checks = []
    for values in cases:
        positions = torch.tensor(values, dtype=torch.int64, device="npu")
        for draft in (None, 1, 2):
            outputs = rope.get_cos_and_sin_dsa(positions, True, draft_index=draft)
            expected = [value[layer].clone() for value in outputs]
            original_ptrs = [value[layer].data_ptr() for value in outputs]
            other_buffers = ([] if draft is None else
                             [value[2 - draft] for value in state.spec_runtime_buffer[key]["default"]])
            other_values = [value.clone() for value in other_buffers]
            result = rope.get_cos_and_sin_dsa(positions, True, draft_index=draft, write_cache_directly=True)
            assert all(torch.equal(value[layer], reference) for value, reference in zip(result, expected))
            assert [value[layer].data_ptr() for value in result] == original_ptrs
            assert all(torch.equal(value, reference) for value, reference in zip(other_buffers, other_values))
            buffers = (state.runtime_buffer[key]["default"] if draft is None else
                       tuple(value[draft - 1] for value in state.spec_runtime_buffer[key]["default"]))
            assert all(torch.all(value[96:] == -91).item() for value in buffers)
            checks.append({"tokens": len(values), "draft": draft, "exact": True, "stable_address": True})

    positions = torch.tensor(cases[0], dtype=torch.int64, device="npu")
    for _ in range(3):
        rope.get_cos_and_sin_dsa(positions, True, write_cache_directly=True)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        actual = rope.get_cos_and_sin_dsa(positions, True, write_cache_directly=True)
    for values in (cases[0], list(range(96))):
        positions.copy_(torch.tensor(values, dtype=torch.int64, device="npu"))
        expected = [full[positions] for full in state.full_rope_cache[key]]
        graph.replay()
        assert all(torch.equal(value[layer], reference) for value, reference in zip(actual, expected))

    timings = []
    for direct in (False, True, False):
        for _ in range(3):
            rope.get_cos_and_sin_dsa(positions, True, write_cache_directly=direct)
        torch.npu.synchronize()
        start = time.perf_counter()
        for _ in range(40):
            rope.get_cos_and_sin_dsa(positions, True, write_cache_directly=direct)
        submitted = time.perf_counter()
        torch.npu.synchronize()
        timings.append({"direct": direct, "host_mean_us": (submitted - start) * 1e6 / 40,
                        "drained_mean_us": (time.perf_counter() - start) * 1e6 / 40})
    report = {"device": args.device, "checks": checks, "replays_exact": 2, "timings": timings,
              "scope": "单个RoPE配置的功能与下发诊断；不代表整模型target性能。"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
