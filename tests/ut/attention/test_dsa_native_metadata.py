# SPDX-License-Identifier: Apache-2.0
"""PTO按需跳过Native分核计划；实际builder、回退和图重放的CPU回归。"""

from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import torch

import vllm_ascend.ops  # noqa: F401 先注册算子，避免DeviceOperator循环导入。
from vllm_ascend.attention import dsa_v1 as dsa


def make_builder(monkeypatch, ratio=4, defer=True, cache=None, device="cpu"):
    """两条真实请求S1/S6和一条图padding请求，固定输出缓冲可跨步复用。"""
    builder = dsa.AscendDSAMetadataBuilder.__new__(dsa.AscendDSAMetadataBuilder)
    query = torch.tensor([0, 1, 7, 12], dtype=torch.int32, device=device)
    lengths = torch.tensor([131073, 131078, 0], dtype=torch.int32, device=device)
    rope = torch.zeros((12, 64), dtype=torch.float32, device=device)
    builder.decode_ratio_to_sas_metadata = {
        "query_start_loc": query,
        "query_start_loc_cpu": query.cpu(),
        "input_positions": torch.arange(12, device=device),
        "cos": rope,
        "sin": rope,
        "max_seq_lens": 131078,
        "seq_lens_list": [131073, 131078, 0],
        "max_seqlen_kv": 131078,
        "max_seqlen_q": 6,
        "start_pos_decode": lengths - query.diff(),
    } if cache is None else cache
    builder.num_decodes, builder.num_decode_tokens = 3, 7
    builder.num_prefills = 0
    builder.compressor_ratio = ratio
    builder.defer_native_csa_metadata = defer
    builder.block_size = 32
    builder.seq_lens = lengths
    builder.start_pos_decode = torch.zeros(3, dtype=torch.int32, device=device)
    builder.block_table = torch.zeros((3, 2), dtype=torch.int32, device=device)
    builder.slot_mapping = torch.zeros((12, 2), dtype=torch.int32, device=device)
    builder.seqused_q = torch.empty(0, dtype=torch.int32, device=device)
    builder._zero_i32 = torch.zeros(1, dtype=torch.int32, device=device)
    builder.cu_seqlens_ori_kv = torch.zeros(4, dtype=torch.int32, device=device)
    builder.cu_seqlens_cmp_kv = torch.empty(0, dtype=torch.int32, device=device)
    builder.decode_sas_metadata = torch.full((1024,), -91, dtype=torch.int32, device=device)
    builder.decode_qli_metadata = torch.full((1024,), -92, dtype=torch.int32, device=device)
    builder.model_config = NS(
        get_head_size=lambda: 512,
        hf_config=NS(num_attention_heads=64, index_topk=512, index_n_heads=64, index_head_dim=128, sliding_window=128),
    )
    builder._num_compressor_metadata_rows = lambda *args: 6
    monkeypatch.setattr(dsa, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(dsa, "get_full_cos_and_sin_dsa", lambda *args: (rope, rope))
    return builder


def build(builder):
    return builder.build_decode_metadata(0, NS(max_query_len=6), 2)


@pytest.fixture
def native_ops(monkeypatch):
    def sas(**kwargs):
        result = torch.full((1024,), kwargs["cmp_ratio"], dtype=torch.int32)
        result[:3] = kwargs["seqused_kv"]
        return result

    def qli(**kwargs):
        result = torch.zeros(1024, dtype=torch.int32)
        result[:3] = kwargs["actual_seq_lengths_key"]
        return result

    sas_op, qli_op = Mock(side_effect=sas), Mock(side_effect=qli)
    monkeypatch.setattr(dsa.DeviceOperator, "get_dsa_sparse_attn_metadata_op", lambda: sas_op)
    monkeypatch.setattr(dsa.DeviceOperator, "get_dsa_sparse_attn_metadata_kwargs", lambda *_: {})
    monkeypatch.setattr(dsa.DeviceOperator, "get_dsa_decode_cu_seqlens_ori_kv", lambda *args: args[-1])
    monkeypatch.setattr(dsa.DeviceOperator, "get_dsa_decode_cu_seqlens_cmp_kv", lambda value: value)
    monkeypatch.setattr(dsa.DeviceOperator, "pad_dsa_decode_slot_mapping", lambda slots, *args: slots)
    monkeypatch.setattr(torch.ops._C_ascend, "npu_vllm_quant_lightning_indexer_metadata", qli_op, raising=False)
    return sas_op, qli_op


@pytest.mark.parametrize("ratio", [0, 4, 128])
@pytest.mark.parametrize("defer", [False, True])
def test_builder_skips_only_unused_plans(monkeypatch, native_ops, ratio, defer):
    builder = make_builder(monkeypatch, ratio, defer)
    result = build(builder)
    sas, qli = native_ops
    assert sas.call_count == int(not (defer and ratio == 4))
    assert qli.call_count == int(not defer)
    if defer and ratio == 4:
        assert result.sas_metadata is None
        assert (builder.decode_sas_metadata == -91).all()  # 连目标buffer复制也没有发生。
    if defer:
        assert result.qli_metadata is None
        assert (builder.decode_qli_metadata == -92).all()
    assert result.query_start_loc.shape == (4,) and result.seq_lens.shape == (3,)
    if ratio > 1:
        assert result.full_compress_cos is not None and result.full_compress_sin is not None


def test_fallback_shares_production_but_fills_each_builders_buffer(monkeypatch, native_ops):
    first = make_builder(monkeypatch)
    second = make_builder(monkeypatch, cache=first.decode_ratio_to_sas_metadata)
    one, two = build(first), build(second)
    one.ensure_native_metadata()
    two.ensure_native_metadata()
    two.ensure_native_metadata()
    assert [op.call_count for op in native_ops] == [1, 1]
    assert one.sas_metadata is first.decode_sas_metadata
    assert two.sas_metadata is second.decode_sas_metadata
    assert one.qli_metadata is first.decode_qli_metadata
    assert two.qli_metadata is second.decode_qli_metadata
    assert torch.equal(one.sas_metadata, two.sas_metadata)
    assert torch.equal(one.qli_metadata, two.qli_metadata)


def test_unused_c4_plans_do_not_prepare_host_kwargs_or_lengths(monkeypatch, native_ops):
    def unexpected(*args):
        pytest.fail("PTO不消费QLI/C4 SAS，不应为它们准备Host kwargs或cu-seqlens")

    for name in (
        "get_dsa_sparse_attn_metadata_op", "get_dsa_sparse_attn_metadata_kwargs",
        "get_dsa_decode_cu_seqlens_ori_kv", "get_dsa_decode_cu_seqlens_cmp_kv",
    ):
        monkeypatch.setattr(dsa.DeviceOperator, name, unexpected)
    result = build(make_builder(monkeypatch, ratio=4, defer=True))
    assert result.sas_metadata is None and result.qli_metadata is None


@pytest.mark.parametrize("ratio,defer", [(4, True), (4, False), (0, True), (128, True)])
def test_c4_decode_does_not_format_unconsumed_slots(monkeypatch, native_ops, ratio, defer):
    builder = make_builder(monkeypatch, ratio=ratio, defer=defer)
    builder.slot_mapping.fill_(-91)
    cache = builder.decode_ratio_to_sas_metadata
    common_cache = {
        "num_decodes": 3, "num_prefills": 0, "num_decode_tokens": 7, "num_prefill_tokens": 0,
        "input_positions": cache["input_positions"], "cos": cache["cos"], "sin": cache["sin"],
        "seq_lens": builder.seq_lens, "query_lens": cache["query_start_loc_cpu"].diff(),
    }
    common = NS(num_reqs=3, num_actual_tokens=7, num_input_tokens=12, graph_pad_size=12,
                query_start_loc=cache["query_start_loc"], slot_mapping=torch.arange(12),
                block_table_tensor=builder.block_table, attn_state=None, max_query_len=6)
    builder.metadata_cls = dsa.AscendDSAMetadata
    format_slots = Mock(side_effect=lambda slots, block: torch.stack((slots // block, slots % block), dim=-1))
    monkeypatch.setattr(dsa.DeviceOperator, "format_dsa_slot_mapping", format_slots)
    result = builder.build(0, common, num_reqs_actual=2, block_size=32,
                           prefill_ratio_to_sas_metadata={}, decode_ratio_to_sas_metadata=cache,
                           common_ratio_to_sas_metadata=common_cache, defer_native_csa_metadata=defer)
    assert format_slots.call_count == int(not (defer and ratio == 4))
    if defer and ratio == 4:
        assert result.decode.slot_mapping is None
        assert (builder.slot_mapping == -91).all()
        # Real Native fallback consumes the lazy plans and CompressorMetadata,
        # rather than the unused uncompressed slot buffer.
        result.decode.ensure_native_metadata()
        assert [op.call_count for op in native_ops] == [1, 1]
    elif ratio <= 1:
        slots = torch.arange(7)
        assert torch.equal(result.decode.slot_mapping[:7], torch.stack((slots // 32, slots % 32), dim=-1))


def test_pto_decode_reuses_common_rope_for_real_rows_and_padding(monkeypatch, native_ops):
    builder = make_builder(monkeypatch, cache={})
    query = torch.tensor([0, 1, 7, 12], dtype=torch.int32)
    positions = torch.arange(12)
    cos = torch.arange(12 * 64, dtype=torch.float32).view(12, 64)
    sin = -cos
    builder.common_ratio_to_sas_metadata = {"input_positions": positions, "cos": cos, "sin": sin}
    common = NS(query_start_loc=query, query_start_loc_cpu=query, positions=positions,
                _seq_lens_cpu=builder.seq_lens, max_query_len=6)
    monkeypatch.setattr(dsa, "get_cos_and_sin_dsa", lambda *a, **kw: pytest.fail("不能重复gather已准备的RoPE"))
    result = builder.build_decode_metadata(0, common, 2)
    assert torch.equal(result.cos, cos[:7]) and torch.equal(result.sin, sin[:7])
    assert result.cos.data_ptr() == cos.data_ptr()
    assert builder.decode_ratio_to_sas_metadata["cos"].shape[0] == 12
    assert torch.equal(result.input_positions, positions[:7])
    assert result.start_pos[-1] == 0  # padding request仍屏蔽。


def test_native_fallback_prepares_and_records_real_consumers(monkeypatch, native_ops):
    from vllm_ascend.ops import dsv4_csa

    builder = make_builder(monkeypatch)
    decoded = build(builder)
    context = NS(
        attn_metadata={"c4": NS(decode=decoded), "indexer": NS(decode=decoded)},
        additional_kwargs={}, capturing=True,
    )
    hidden = torch.zeros((1, 4, 4096), dtype=torch.bfloat16)

    def hc_pre(value, *args):
        assert decoded.native_metadata_factory is None
        assert [op.call_count for op in native_ops] == [1, 1]
        return value, None, None

    layer = NS(
        hc_pre=hc_pre, hc_attn_fn=None, hc_attn_scale=None, hc_attn_base=None,
        input_layernorm=lambda value: value,
        self_attn=lambda **kwargs: kwargs["hidden_states"],
        hc_post=lambda value, *args: value,
    )
    wrapper = NS(
        _pto_csa_layer=(layer,), dsa_attn=NS(layer_name="c4"), indexer=NS(k_cache=NS(prefix="indexer")),
    )
    monkeypatch.setattr(dsv4_csa, "get_forward_context", lambda: context)
    dsv4_csa._native_attention_half(wrapper, hidden, torch.zeros(1, dtype=torch.int64), torch.empty_like(hidden))
    assert context.additional_kwargs["pto_csa_native_metadata_prefixes"] == {"c4", "indexer"}


@pytest.mark.parametrize("native_graph", [False, True])
def test_graph_replay_prepares_current_step_only_for_native(monkeypatch, native_ops, native_graph):
    from vllm.config import CUDAGraphMode

    from vllm_ascend.compilation import acl_graph

    builder = make_builder(monkeypatch)
    # 图捕获时固定的目标地址；下一步真实长度改变，但地址不变。
    captured = build(builder)
    captured.ensure_native_metadata()
    old_value = captured.sas_metadata.clone()
    builder.decode_ratio_to_sas_metadata = dict(builder.decode_ratio_to_sas_metadata)
    builder.decode_ratio_to_sas_metadata.pop("c4")
    builder.decode_ratio_to_sas_metadata.pop("qli")
    builder.seq_lens.add_(6)
    current = build(builder)
    context = NS(
        attn_metadata={"c4": NS(decode=current)}, additional_kwargs={},
        batch_descriptor="b3", cudagraph_runtime_mode=CUDAGraphMode.FULL,
    )
    events = []

    def replay():
        events.append("replay")
        if native_graph:
            assert torch.equal(captured.sas_metadata[:3], builder.seq_lens)
            assert torch.equal(captured.qli_metadata[:3], builder.seq_lens)
        else:
            assert torch.equal(captured.sas_metadata, old_value)

    entry = acl_graph.ACLGraphEntry(
        batch_descriptor="b3", aclgraph=NS(replay=replay), output="output",
        csa_native_metadata_prefixes=("c4",) if native_graph else (),
    )
    wrapper = acl_graph.ACLGraphWrapper.__new__(acl_graph.ACLGraphWrapper)
    wrapper.runtime_mode = CUDAGraphMode.FULL
    wrapper.concrete_aclgraph_entries = {"b3": entry}
    wrapper.is_debugging_mode = False
    wrapper.enable_enpu = True
    wrapper.use_eagle = False
    monkeypatch.setattr(acl_graph, "get_forward_context", lambda: context)
    monkeypatch.setattr(acl_graph, "_EXTRA_CTX", NS(is_draft_model=False))
    assert wrapper() == "output" and events == ["replay"]
    assert [op.call_count for op in native_ops] == [1 + int(native_graph)] * 2
    assert (current.native_metadata_factory is None) == native_graph
