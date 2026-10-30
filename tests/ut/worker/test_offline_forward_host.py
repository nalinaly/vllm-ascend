# SPDX-License-Identifier: Apache-2.0
"""计时钩子保留原通信调用、异常及draft路径，不添加同步。"""

import pytest
import torch.distributed as dist

from offline_pd.forward_host import ForwardHostDiagnostics


@pytest.mark.parametrize("draft,raises", [(False, False), (True, False), (False, True)])
def test_dp_observation_preserves_calls_and_restores_hooks(monkeypatch, draft, raises):
    calls = []

    def original(payload, **kwargs):
        calls.append(payload)
        if raises:
            raise RuntimeError("original failure")
        return "unchanged"

    class Runner:
        def _sync_metadata_across_dp(self, payload, is_draft_model=False):
            return dist.all_reduce(payload)

    monkeypatch.setattr(dist, "all_reduce", original)
    runner, entry = Runner(), {"index": 0}
    diagnostic = ForwardHostDiagnostics()
    diagnostic.attach_runner(runner, lambda: entry)
    try:
        if raises:
            with pytest.raises(RuntimeError, match="original failure"):
                runner._sync_metadata_across_dp("same tensor", is_draft_model=draft)
        else:
            assert runner._sync_metadata_across_dp("same tensor", is_draft_model=draft) == "unchanged"
        assert calls == ["same tensor"]
        assert diagnostic.dp_sync_entry is None
        if draft:
            assert "host" not in entry
        else:
            labels = ["dp_sync_begin", "dp_all_reduce_begin", "dp_all_reduce_end", "dp_sync_end"]
            timestamps = [entry["host"][key]["monotonic_ns"] for key in labels]
            assert timestamps == sorted(timestamps)
    finally:
        diagnostic.finish([entry])
    assert dist.all_reduce is original
    assert "_sync_metadata_across_dp" not in vars(runner)
