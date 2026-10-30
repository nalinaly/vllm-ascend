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
