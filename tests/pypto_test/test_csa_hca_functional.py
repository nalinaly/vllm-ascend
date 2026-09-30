# SPDX-License-Identifier: Apache-2.0
"""功能闸门拒绝缺 rank、缺层、未重放及不完整请求，不能用部分结果冒充通过。"""

import json

import pytest

from csa_hca_merge_20260930.functional import validate


@pytest.mark.parametrize("missing", [None, "rank", "hca", "replay", "cache", "tokens"])
def test_functional_requires_complete_execution(tmp_path, missing):
    plan = {"prefill": {"dp": 1}, "decode": {"speculative_tokens": 5},
            "cases": [{"key": "h131072_v0", "p_dp_rank": 0}]}
    layers = [f"model.layers.{i}.self_attn.attn" for i in range(2, 43)]
    for rank in range(2):
        selection = {layer: {"pto_tokens12": 1} for layer in layers}
        observation = {"target_layer_names": layers, "capture_time_selection": selection,
                       "model_forward_counts": {"FULL_tokens12": 4}}
        case = {"key": "h131072_v0", "submitted": 2, "output_token_ids": [[7, 8, 9], [7, 8, 9]],
                "csa_observation": [observation],
                "spec_decode": {"sufficient": True, "counters": {
                    "num_drafts": 2, "num_draft_tokens": 10, "num_accepted_tokens": 5},
                    "num_accepted_tokens_per_pos": [2, 1, 1, 1, 0]}}
        if rank == 1:
            if missing == "rank":
                continue
            if missing == "hca":
                del selection["model.layers.3.self_attn.attn"]
            if missing == "replay":
                observation["model_forward_counts"] = {"NONE_tokens12": 4}
            if missing == "tokens":
                case["output_token_ids"][0].pop()
        payload = {"role": "decode", "backend": "pto", "rank": rank, "batch": 2,
                   "pto_attention": "both", "decode_dp": 2, "cases": [case]}
        (tmp_path / f"rank{rank}.json").write_text(json.dumps(payload))
        count = 1 if rank == 1 and missing == "cache" else 2
        (tmp_path / f"rank{rank}.log").write_text(
            f"OFFLINE_CACHE_LOADED dp={rank} key=h131072_v0\n" * count)
    result = validate(tmp_path, plan, batch=2, tokens=3, ranks=2)
    assert result["status"] == ("PASS" if missing is None else "FAIL")
    assert len(result["ranks"]) == (2 if missing is None else 1)
