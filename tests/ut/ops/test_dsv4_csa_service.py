# SPDX-License-Identifier: Apache-2.0
"""CPU checks for real service selection, including the runner's graph gate."""

from types import SimpleNamespace as NS
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from vllm.config import CUDAGraphMode
from vllm.forward_context import BatchDescriptor

from vllm_ascend.ops.pypto.deepseek_v4_flash_dspark.service import CSAServiceRuntime
from vllm_ascend.ops.pypto.deepseek_v4_flash_dspark.service_config import MODEL_ARCHITECTURE


@pytest.mark.parametrize(
    "tokens,reqs,bucket,initial,architecture,expected",
    [
        (6, 1, 6, True, MODEL_ARCHITECTURE, CUDAGraphMode.FULL),
        (5, 1, 6, True, MODEL_ARCHITECTURE, CUDAGraphMode.NONE),
        (24, 4, 24, True, MODEL_ARCHITECTURE, CUDAGraphMode.FULL),
        (18, 3, 24, True, MODEL_ARCHITECTURE, CUDAGraphMode.NONE),
        (23, 4, 24, True, MODEL_ARCHITECTURE, CUDAGraphMode.NONE),
        (24, 4, 24, False, MODEL_ARCHITECTURE, CUDAGraphMode.NONE),
        (18, 3, 30, True, MODEL_ARCHITECTURE, CUDAGraphMode.NONE),
        (18, 3, 18, True, MODEL_ARCHITECTURE, CUDAGraphMode.FULL),
        (30, 5, 30, True, MODEL_ARCHITECTURE, CUDAGraphMode.FULL),
        (18, 3, 24, True, "DeepseekV4ForCausalLM", CUDAGraphMode.FULL),
    ],
)
def test_real_runner_graph_dispatch(tokens, reqs, bucket, initial, architecture, expected, monkeypatch):
    from vllm_ascend.worker import model_runner_v1 as module

    monkeypatch.setattr(module, "enable_sp", lambda _: False)
    config = NS(model_config=NS(hf_config=NS(architectures=[architecture])),
                parallel_config=NS(data_parallel_size=1),
                observability_config=NS(cudagraph_metrics=False))
    runner = NS(
        _pad_for_sequence_parallelism=lambda n: n,
        input_batch=NS(num_computed_tokens_cpu=np.full(reqs, int(initial)), lora_id_to_lora_request={}),
        speculative_config=NS(method="dspark"), uniform_decode_query_len=6, model_config=NS(is_encoder_decoder=False),
        vllm_config=config,
        cudagraph_dispatcher=NS(dispatch=Mock(return_value=(CUDAGraphMode.FULL, BatchDescriptor(bucket)))),
    )
    mode, descriptor, *_ = module.NPUModelRunner._determine_batch_execution_and_padding(
        runner, tokens, reqs, np.full(reqs, 6), 6, False,
    )
    assert mode == expected
    assert descriptor.num_tokens == (tokens if expected == CUDAGraphMode.NONE else bucket)


def metadata_context():
    runtime = CSAServiceRuntime.__new__(CSAServiceRuntime)
    runtime.batch_capacity = 40
    runtime.prefixes = {name: name for name in ("swa", "compressed", "state", "indexer", "indexer_state")}
    items = {}
    for prefix in runtime.prefixes:
        req = NS(seq_lens=torch.empty(4, dtype=torch.int32), query_start_loc=torch.empty(5, dtype=torch.int32),
                 num_reqs_actual=4, max_seqlen_q=6, ori_win_right=0, ori_win_left=127,
                 dspark_swa_indices=None)
        items[prefix] = NS(num_prefills=0, num_actual_tokens=24, num_decodes=4, decode=req)
    return runtime, NS(attn_metadata=items, is_draft_model=False)


@pytest.mark.parametrize("case", ["supported", "prefill", "padding", "nonuniform", "dummy", "draft", "window"])
def test_service_gate_uses_host_metadata(case):
    runtime, context = metadata_context()
    if case == "prefill":
        context.attn_metadata["compressed"].num_prefills = 1
    elif case == "padding":
        context.attn_metadata["compressed"].num_actual_tokens = 18
    elif case == "nonuniform":
        context.attn_metadata["compressed"].decode.max_seqlen_q = 7
    elif case == "dummy":
        context.attn_metadata["compressed"].decode.num_reqs_actual = 3
    elif case == "draft":
        context.is_draft_model = True
    elif case == "window":
        context.attn_metadata["swa"].decode.ori_win_left = 255
    assert runtime.eligible(context, torch.empty((24, 4096), dtype=torch.bfloat16),
                            torch.empty(24, dtype=torch.int64)) == (case == "supported")


def test_release_metadata_and_compact_buffers_bind_without_copy():
    """Use release dataclasses and Native storage views across the actual ABI."""
    from vllm_ascend.attention.dsa_v1 import AscendDSADecodeMetadata, AscendDSAMetadata
    from vllm_ascend.ops.pypto.deepseek_v4_flash_dspark.decode_csa import decode_csa_tp1_attention_test
    from vllm_ascend.ops.pypto.deepseek_v4_flash_dspark.native_adapter import NativeCSACall

    tokens, batch, pages = 6, 1, 4
    hidden = torch.empty((tokens, 4096), dtype=torch.bfloat16)
    positions = torch.arange(tokens, dtype=torch.int64)
    bounds = torch.tensor([0, tokens], dtype=torch.int32)
    lengths = torch.tensor([tokens], dtype=torch.int32)
    layer_name = "model.layers.2.attn"
    cos = torch.ones((tokens, 64), dtype=torch.float32)
    sin = torch.zeros_like(cos)
    groups, compact = {}, {}
    for name in ("swa", "compressed", "state", "indexer", "indexer_state"):
        table = torch.zeros((batch, 8), dtype=torch.int32)
        slots = None if name in ("compressed", "indexer") else torch.zeros((tokens, 2), dtype=torch.int32)
        req = AscendDSADecodeMetadata(
            input_positions=positions, block_table=table, seq_lens=lengths,
            max_seqlen_kv=tokens, max_seqlen_q=tokens, seq_lens_list=[tokens], max_seq_lens=tokens,
            slot_mapping=slots, block_size=2 if "state" in name else 32,
            query_start_loc=bounds, query_start_loc_cpu=bounds, num_reqs_actual=batch,
            cos={layer_name: cos}, sin={layer_name: sin},
        )
        metadata = AscendDSAMetadata(
            num_actual_tokens=tokens, slot_mapping=slots, query_start_loc=bounds,
            seq_lens=lengths, block_tables=table, sin=sin, cos=cos,
            num_decodes=batch, num_decode_tokens=tokens, num_prefills=0, decode=req,
        )
        if name == "indexer":
            storage = torch.empty((pages, 4160), dtype=torch.int8)
            key = storage.as_strided((pages, 32, 1, 128), (4160, 128, 128, 1))
            scale = storage.view(torch.float16).as_strided((pages, 32, 1, 1), (2080, 1, 1, 1), 2048)
            views = (key, scale)
        elif "state" in name:
            width = 512 if name == "indexer_state" else 2048
            storage = torch.empty((pages, 1040 if width == 512 else 8192), dtype=torch.float32)
            views = (storage.as_strided((pages, 2, 1, width), (storage.shape[1], width, width, 1)),)
        else:
            views = (torch.empty((pages, 32, 1, 512), dtype=torch.bfloat16),)
        groups[name] = (metadata, views)
        if name in ("compressed", "indexer"):
            compact[name] = (cos[:2], sin[:2], torch.zeros((2, 2), dtype=torch.int32))

    runtime = CSAServiceRuntime.__new__(CSAServiceRuntime)
    runtime.batch_capacity = 64
    runtime.prefixes = {name: name for name in groups}
    context = NS(attn_metadata={name: value[0] for name, value in groups.items()}, is_draft_model=False)
    assert runtime.eligible(context, hidden, positions)
    # Weight values are irrelevant to CPU descriptor binding; no kernel runs.
    weights = dict.fromkeys(decode_csa_tp1_attention_test.param_names, torch.empty(0))
    submit = Mock()
    call = NativeCSACall(NS(attention=submit), weights, hidden, positions, groups,
                         layer_name=layer_name, compact_metadata=compact)
    assert call.args["position_ids"] is positions
    assert call.args["ori_slot_mapping"] is groups["swa"][0].decode.slot_mapping
    assert call.args["cmp_slot_mapping"] is compact["compressed"][2]
    assert call.args["idx_slot_mapping"] is compact["indexer"][2]
    assert call.args["cmp_freqs_cos"].data_ptr() == compact["compressed"][0].data_ptr()
    assert call.args["freqs_cos"].data_ptr() == cos.data_ptr()
    for key, group in (("compress_state", "state"), ("inner_compress_state", "indexer_state"),
                       ("idx_kv_cache", "indexer")):
        assert call.args[key].data_ptr() == groups[group][1][0].data_ptr()
    assert call() is call.args["attn_out"]
    submit.assert_called_once()
