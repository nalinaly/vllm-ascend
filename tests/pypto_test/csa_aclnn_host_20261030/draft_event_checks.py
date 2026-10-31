# SPDX-License-Identifier: Apache-2.0
"""私有候选的Event生命周期边界：主图、prefill、非eager draft不复用。"""

from contextlib import contextmanager
from types import SimpleNamespace as NS
from unittest.mock import MagicMock, Mock

import pytest
import torch

import vllm_ascend.ops  # noqa: F401
from vllm_ascend.attention import dsa_v1 as dsa


@pytest.mark.parametrize("draft,eager,prefill,reused", [
    (True, True, False, True), (False, True, False, False),
    (True, True, True, False), (True, False, False, False),
])
def test_reuse_only_eager_draft_prolog(monkeypatch, draft, eager, prefill, reused):
    created = Mock(side_effect=object)
    records, waits = [], []

    class Stream:
        def record_event(self, event=None):
            event = created() if event is None else event
            records.append(event)
            return event

        def wait_event(self, event):
            waits.append(event)

    main, aux = Stream(), Stream()
    current = [main]

    @contextmanager
    def switch(stream, enabled):
        previous = current[0]
        current[0] = stream
        try:
            yield
        finally:
            current[0] = previous

    monkeypatch.setattr(torch.npu, "Event", created)
    monkeypatch.setattr(torch.npu, "current_stream", lambda: current[0])
    monkeypatch.setattr(dsa, "dsv4_dsa_overlap_stream", lambda: aux)
    monkeypatch.setattr(dsa, "npu_stream_switch", switch)
    monkeypatch.setattr(dsa, "get_forward_context", lambda: NS(is_draft_model=draft))
    monkeypatch.setattr(dsa, "_is_w8a8_dynamic", lambda _: False)
    monkeypatch.setattr(torch.ops._C_ascend, "inplace_partial_rotary_mul", Mock(), raising=False)
    monkeypatch.setattr(dsa.DeviceOperator, "dsa_kv_compress_scatter", Mock())
    monkeypatch.setattr(dsa.DeviceOperator, "apply_dsa_q_rms", Mock(return_value=MagicMock()))
    impl = dsa.AscendDSAImpl.__new__(dsa.AscendDSAImpl)
    impl.vllm_config = NS(speculative_config=NS(enforce_eager=eager))
    impl._eager_draft_prolog_events = None
    for name in ("wq_b", "cv_wq_a", "cv_wq_b", "cv_wkv", "q_norm", "kv_norm", "q_norm_without_weight"):
        setattr(impl, name, MagicMock())
    for name in ("cv_wq_a", "cv_wq_b", "cv_wkv"):
        getattr(impl, name).quantize.return_value = (MagicMock(), MagicMock())
    impl.rope_head_dim, impl.nope_head_dim, impl.head_dim = 2, 2, 4
    impl.n_local_heads, impl.eps = 2, 1e-6
    for _ in range(2):
        impl._mla_prolog_multistream(MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(), prefill)
    assert created.call_count == (6 if reused else 12)
    assert len(records) == len(waits) == 12
    assert all(a is b for a, b in zip(records, waits))
    assert all((a is b) == reused for a, b in zip(records[:6], records[6:]))
