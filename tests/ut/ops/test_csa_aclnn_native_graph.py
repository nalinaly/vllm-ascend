# SPDX-License-Identifier: Apache-2.0
"""后置ACLNN沿Native图生命周期，禁止稳态重算CSA合同或按长度回退。"""

from types import SimpleNamespace as NS

import pytest
import torch

from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd_tnd.service import CSAServiceRuntime


def test_device_route_does_not_read_length_or_scan_descriptors():
    class Decode:
        def __getattr__(self, name):
            raise AssertionError(f"后置ACLNN不应在Host读取decode.{name}")

    runtime = CSAServiceRuntime.__new__(CSAServiceRuntime)
    runtime.native_graph_lifecycle = True
    runtime.prefixes = {"compressed": "c4"}
    runtime.graph_signature = lambda *a: pytest.fail("不应生成CSA签名")
    context = NS(
        attn_metadata={"c4": NS(num_prefills=0, decode=Decode())},
        capturing=True,
        additional_kwargs={},
        is_draft_model=False,
    )
    # 不能通过hidden/positions的shape扫描决定长度或图可用性。
    assert runtime.eligible(context, None, None)
    runtime.record_graph_route(context, None, None)
    assert context.additional_kwargs == {}


def test_padded_capacity_captures_real_operator_instead_of_host_zero_branch():
    runtime = CSAServiceRuntime.__new__(CSAServiceRuntime)
    runtime.native_graph_lifecycle = True
    # 有容量、全padding仍需要捕获ACLNN，以便同图下一次处理真实Q/L。
    hidden = torch.empty((96, 4, 4096), dtype=torch.bfloat16)
    assert runtime.handle_empty_call(NS(), hidden, None, None) is False


def test_native_prefill_routing_is_preserved():
    runtime = CSAServiceRuntime.__new__(CSAServiceRuntime)
    runtime.native_graph_lifecycle = True
    runtime.prefixes = {"compressed": "c4"}
    context = NS(attn_metadata={"c4": NS(num_prefills=1, decode=NS())}, is_draft_model=False)
    assert not runtime.eligible(context, None, None)


def test_model_preparation_keeps_config_replace_compatible(monkeypatch):
    from dataclasses import dataclass, field

    from vllm.config.utils import replace

    from vllm_ascend.models.pypto_deepseek_v4 import prepare_csa_model
    from vllm_ascend.ops.pypto import variant
    from vllm_ascend.ops.pypto.deepseek_v4_flash_csa import nz_mode
    from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd_tnd import backend, service

    @dataclass
    class Config:
        additional_config: dict = field(default_factory=dict)
        scheduler_config: object = field(default_factory=lambda: NS(max_num_seqs=16))

    config = Config()
    monkeypatch.setattr("vllm.config.get_current_vllm_config", lambda: config)
    monkeypatch.setattr("vllm_ascend.ascend_config.get_ascend_config", lambda: NS(weight_nz_mode=2))
    monkeypatch.setattr(variant, "selected_variant", lambda: "spmd_tnd")
    monkeypatch.setattr(variant, "selected_csa_backend", lambda: "aclnn")
    monkeypatch.setattr(variant, "csa_runtime", lambda: "tensormap_and_ringbuffer")
    monkeypatch.setattr(variant, "variant_package", lambda: service.__package__)
    monkeypatch.setattr(backend, "validate_backend_installation", lambda: None)
    monkeypatch.setattr(backend, "uses_device_policy", lambda: True)
    monkeypatch.setattr(nz_mode, "root_weight_layouts", lambda _: {})
    monkeypatch.setattr(service, "CSAServiceRuntime", lambda *args: NS(native_graph_lifecycle=True))
    layer = NS(self_attn=NS(compress_ratio=4, dsa_attn=NS()))
    prepare_csa_model(NS(model=NS(layers=[layer])))
    # 与DSpark相同的复制函数：旧版挂载私有字段时，这一步会报Field not found。
    copied = replace(config)
    assert copied.additional_config["_pto_csa_device_policy"] is True
    assert "_pto_csa_device_policy" not in vars(config)
