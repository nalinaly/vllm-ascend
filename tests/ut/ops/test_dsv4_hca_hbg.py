# SPDX-License-Identifier: Apache-2.0
"""HCA HBG 的独立 ABI 和加载期 Host 权重快照契约；无需 NPU。"""

import inspect
from types import SimpleNamespace

import pytest
import torch


def test_hca_hbg_keeps_tensor_abi_and_requires_host_scalars(monkeypatch):
    from vllm_ascend.ops.pypto.deepseek_v4_flash_hca.decode_hca import _decode_hca_tp1_layer_hbg
    from vllm_ascend.ops.pypto.deepseek_v4_flash_hca.native_adapter import HCAOperators

    monkeypatch.setenv("PTO_CSA_RUNTIME", "tensormap_and_ringbuffer")
    legacy = HCAOperators.register()
    monkeypatch.setenv("PTO_CSA_RUNTIME", "host_build_graph")
    hbg = HCAOperators.register()
    assert not legacy.is_hbg and hbg.is_hbg
    legacy_args, hbg_args = legacy.attention._schema.arguments, hbg.attention._schema.arguments
    assert len(legacy_args) == 36 and len(hbg_args) == 39
    assert all(str(arg.type) == "Tensor" for arg in legacy_args)
    assert [arg.name for arg in hbg_args[:36]] == [arg.name for arg in legacy_args]
    for index, arg in enumerate(hbg_args[36:]):
        assert arg.name == f"host_hc_scale{index}" and str(arg.type) == "float"
        assert not arg.has_default_value()
        assert inspect.signature(_decode_hca_tp1_layer_hbg).parameters[arg.name].default is inspect.Parameter.empty


def test_hca_host_scales_are_frozen_at_weight_preparation(monkeypatch):
    from vllm_ascend.ops.pypto.deepseek_v4_flash_hca import native_adapter

    source = torch.tensor([0.125, -2.0, 3.25], dtype=torch.float32)
    monkeypatch.setattr(native_adapter, "_prepare_weights", lambda *args, **kwargs: {"hc_attn_scale": source})
    attention = SimpleNamespace(compress_ratio=128)
    legacy = native_adapter.prepare_weights(attention, None)
    hbg = native_adapter.prepare_weights(attention, None, host_scalars=True)
    assert set(legacy) == {"hc_attn_scale"}
    assert [hbg[f"host_hc_scale{i}"] for i in range(3)] == [0.125, -2.0, 3.25]
    assert hbg["hc_attn_scale"] is source
    source.fill_(float("nan"))
    assert hbg["host_hc_scale0"] == 0.125
    with pytest.raises(ValueError, match="有限值"):
        native_adapter.prepare_weights(attention, None, host_scalars=True)
