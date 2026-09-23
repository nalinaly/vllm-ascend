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
        use_dcp=False, uniform_decode_query_len=6, model_config=NS(is_encoder_decoder=False),
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
                 num_actual_reqs=4, ori_win_right=0, ori_win_left=127,
                 vision_swa_indices=None, dspark_swa_indices=None)
        items[prefix] = NS(num_prefills=0, num_actual_tokens=24, max_query_len=6, num_decodes=4, req_metadata=req)
    return runtime, NS(attn_metadata=items, is_draft_model=False)


@pytest.mark.parametrize("case", ["supported", "prefill", "padding", "nonuniform", "dummy", "draft", "window"])
def test_service_gate_uses_host_metadata(case):
    runtime, context = metadata_context()
    if case == "prefill":
        context.attn_metadata["compressed"].num_prefills = 1
    elif case == "padding":
        context.attn_metadata["compressed"].num_actual_tokens = 18
    elif case == "nonuniform":
        context.attn_metadata["compressed"].max_query_len = 7
    elif case == "dummy":
        context.attn_metadata["compressed"].req_metadata.num_actual_reqs = 3
    elif case == "draft":
        context.is_draft_model = True
    elif case == "window":
        context.attn_metadata["swa"].req_metadata.ori_win_left = 255
    assert runtime.eligible(context, torch.empty((24, 4096), dtype=torch.bfloat16),
                            torch.empty(24, dtype=torch.int64)) == (case == "supported")
