# SPDX-License-Identifier: Apache-2.0
"""通过private bootstrap显式交给pytest；不在公共路径自动发现的候选回归。"""

from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import torch

import vllm_ascend.ops  # noqa: F401
from vllm_ascend.attention.dsa_v1 import AscendDSAMetadataBuilder
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.worker.block_table import BlockTable
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner


def table(cp=1, hybrid=False):
    result = BlockTable.__new__(BlockTable)
    result.dcp_world_size = cp
    result.use_hybrid_blocks = hybrid
    result.max_num_batched_tokens = 12
    result.dsa_slot_mappings = ()
    return result


def builder(ratio=1):
    result = AscendDSAMetadataBuilder.__new__(AscendDSAMetadataBuilder)
    result.slot_mapping = torch.full((12, 2), -91, dtype=torch.int32)
    result.slot_mapping_precomputed = False
    result.compressor_ratio = ratio
    return result


def test_registration_preserves_two_independent_buffers_and_compressed_path():
    first, second, compressed = builder(), builder(), builder(4)
    first_buffer, second_buffer = first.slot_mapping, second.slot_mapping
    tables = [table(), table()]
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.use_compress = True
    runner.parallel_config = NS(enable_dbo=False)
    runner.input_batch = NS(block_table=tables)
    runner.attn_groups = [
        [NS(get_metadata_builder=lambda _: first), NS(get_metadata_builder=lambda _: second)],
        [NS(get_metadata_builder=lambda _: compressed)],
    ]
    runner._bind_dsa_slot_mappings()
    assert first.slot_mapping is first_buffer and second.slot_mapping is second_buffer
    assert first_buffer.data_ptr() != second_buffer.data_ptr()
    assert tables[0].dsa_slot_mappings[0] is first_buffer
    assert tables[0].dsa_slot_mappings[1] is second_buffer
    assert first.slot_mapping_precomputed and second.slot_mapping_precomputed
    assert not compressed.slot_mapping_precomputed and tables[1].dsa_slot_mappings == ()


@pytest.mark.parametrize("cp,hybrid", [(2, False), (1, True)])
def test_unsupported_block_paths_keep_upstream(cp, hybrid):
    value = table(cp, hybrid)
    assert not value.bind_dsa_slot_mappings((builder().slot_mapping,))
    assert value.dsa_slot_mappings == ()


@pytest.mark.parametrize("shape", [(12,), (11, 2)])
def test_incompatible_destination_keeps_upstream(shape):
    value = table()
    assert not value.bind_dsa_slot_mappings((torch.empty(shape, dtype=torch.int32),))
    assert value.dsa_slot_mappings == ()


@pytest.mark.parametrize("precomputed", [False, True])
def test_builder_consumes_prepared_slots_without_reformatting(monkeypatch, precomputed):
    value = builder()
    raw = torch.arange(-1, 11, dtype=torch.int32)
    expected = DeviceOperator.format_dsa_slot_mapping(raw, 32)
    query = torch.tensor([0, 1, 7, 12], dtype=torch.int32)
    rope = torch.zeros((12, 64))
    cache = {"num_decodes": 3, "num_prefills": 0, "num_decode_tokens": 7, "num_prefill_tokens": 0,
             "input_positions": torch.arange(12), "cos": rope, "sin": rope,
             "seq_lens": torch.tensor([131073, 131078, 0]), "query_lens": query.diff()}
    common = NS(num_reqs=3, num_actual_tokens=7, num_input_tokens=12, graph_pad_size=12,
                query_start_loc=query, slot_mapping=raw.clone(), block_table_tensor=torch.zeros((3, 2)),
                attn_state=None)
    value.model_config = NS(get_head_size=lambda: 512)
    value.metadata_cls = lambda **fields: NS(**fields)
    value.get_block_table_size = lambda *args: 3
    value.build_decode_metadata = lambda *args: NS(slot_mapping=value.slot_mapping[:7])
    if precomputed:
        value.slot_mapping.copy_(expected)
        common.slot_mapping.fill_(999)  # Must consume the prepared output, not reread the flat input.
    formatter = Mock(wraps=DeviceOperator.format_dsa_slot_mapping)
    monkeypatch.setattr(DeviceOperator, "format_dsa_slot_mapping", formatter)
    result = value.build(0, common, num_reqs_actual=2, block_size=32,
                         prefill_ratio_to_sas_metadata={}, decode_ratio_to_sas_metadata={},
                         common_ratio_to_sas_metadata=cache, slot_mapping_precomputed=precomputed)
    assert formatter.call_count == int(not precomputed)
    assert torch.equal(result.decode.slot_mapping, expected[:7])
    assert torch.equal(value.slot_mapping, expected)
