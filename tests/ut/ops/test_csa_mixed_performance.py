# SPDX-License-Identifier: Apache-2.0
"""混合性能场景保留逐请求KV key；每轮等齐只发生在生成结束后。"""

import json
from types import SimpleNamespace as NS
from unittest.mock import Mock

from offline_pd.run import generate_round


def test_warmup_and_measurement_use_same_distinct_requests(monkeypatch, tmp_path):
    cases = [{"key": f"request_{i}", "tokens": f"tokens_{i}.json"} for i in range(2)]
    for index, case in enumerate(cases):
        (tmp_path / case["tokens"]).write_text(json.dumps([11 + index, 21 + index]))
    calls = []

    def generate(llm, prompts, params):
        calls.append((prompts, params))
        return [NS(outputs=[NS(token_ids=[99])]) for _ in prompts]

    peers = Mock()
    monkeypatch.setattr("offline_pd.batch.generate_aligned_batch", generate)
    monkeypatch.setattr("low_acceptance_20261001.mixed.wait_for_generation_peers", peers)
    args = NS(
        bank=tmp_path,
        output=tmp_path,
        rank=0,
        rank_batch=None,
        batch=2,
        command="performance",
        mixed_performance_cases=cases,
    )
    for limit in (96, 192):
        generate_round(None, args, cases[0], limit)
    assert calls[0][0] == calls[1][0] == [{"prompt_token_ids": [11, 21]}, {"prompt_token_ids": [12, 22]}]
    for _, params in calls:
        assert [p.extra_args["kv_transfer_params"]["offline_key"] for p in params] == [c["key"] for c in cases]
    assert [p.max_tokens for p in calls[0][1]] == [96, 96]
    assert [p.max_tokens for p in calls[1][1]] == [192, 192]
    assert [c.args[2] for c in peers.call_args_list] == ["performance_0", "performance_1"]
