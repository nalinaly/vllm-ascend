# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_ascend.models import pypto_qwen3_attention as pypto_attention
from vllm_ascend.ops import qwen3_decode_attention as attention_op

PYPTO_DFX_DIR_ENV = "VLLM_ASCEND_QWEN3_PYPTO_DFX_DIR"


def _cache() -> torch.Tensor:
    return torch.zeros(
        (
            2,
            1,
            pypto_attention.BLOCK_SIZE,
            pypto_attention.NUM_KV_HEADS,
            pypto_attention.HEAD_DIM,
        ),
        dtype=torch.bfloat16,
    )


def _attention(kernel):
    attention = pypto_attention.Qwen3Attention.__new__(pypto_attention.Qwen3Attention)
    nn.Module.__init__(attention)
    attention.attn = SimpleNamespace(layer_name="model.layers.0.self_attn.attn")
    attention.qkv_proj = nn.Module()
    attention.qkv_proj.weight = nn.Parameter(torch.ones(1), requires_grad=False)
    attention.o_proj = nn.Module()
    attention.o_proj.weight = nn.Parameter(torch.ones(1), requires_grad=False)
    attention._pypto_kernel = kernel
    attention._pypto_q_norm = torch.ones((1, pypto_attention.HEAD_DIM))
    attention._pypto_k_norm = torch.ones((1, pypto_attention.HEAD_DIM))
    attention._pypto_rope_cos = torch.ones((pypto_attention.MAX_SEQ_LEN, pypto_attention.HEAD_DIM))
    attention._pypto_rope_sin = torch.zeros_like(attention._pypto_rope_cos)
    return attention


def _meta_attention_args(batch: int = 2) -> tuple[torch.Tensor, ...]:
    device = torch.device("meta")
    return (
        torch.empty((batch, attention_op.HIDDEN_SIZE), dtype=torch.bfloat16, device=device),
        torch.empty(
            (attention_op.QKV_HIDDEN_SIZE, attention_op.HIDDEN_SIZE),
            dtype=torch.bfloat16,
            device=device,
        ),
        torch.empty((1, attention_op.HEAD_DIM), dtype=torch.float32, device=device),
        torch.empty((1, attention_op.HEAD_DIM), dtype=torch.float32, device=device),
        torch.empty((batch,), dtype=torch.int64, device=device),
        torch.empty((batch, 32), dtype=torch.int32, device=device),
        torch.empty((batch,), dtype=torch.int64, device=device),
        torch.empty(
            (attention_op.MAX_SEQ_LEN, attention_op.HEAD_DIM),
            dtype=torch.float32,
            device=device,
        ),
        torch.empty(
            (attention_op.MAX_SEQ_LEN, attention_op.HEAD_DIM),
            dtype=torch.float32,
            device=device,
        ),
        torch.empty((1024, attention_op.HEAD_DIM), dtype=torch.bfloat16, device=device),
        torch.empty((1024, attention_op.HEAD_DIM), dtype=torch.bfloat16, device=device),
        torch.empty(
            (attention_op.HIDDEN_SIZE, attention_op.HIDDEN_SIZE),
            dtype=torch.bfloat16,
            device=device,
        ),
    )


def test_backend_neutral_meta_infers_output_without_pypto():
    args = _meta_attention_args()

    output = attention_op.qwen3_decode_attention(*args)

    assert output.device.type == "meta"
    assert output.dtype == args[0].dtype
    assert output.shape == args[0].shape


def test_model_adapter_does_not_replace_model_or_decoder_forward():
    adapter = pypto_attention.PyptoAttentionQwen3ForCausalLM

    assert "forward" not in adapter.__dict__
    assert "load_weights" not in adapter.__dict__


def test_dfx_environment_variable_selects_output_directory(monkeypatch, tmp_path):
    monkeypatch.delenv(PYPTO_DFX_DIR_ENV, raising=False)
    assert pypto_attention._read_pypto_dfx_dir() is None

    monkeypatch.setenv(PYPTO_DFX_DIR_ENV, str(tmp_path / "dfx"))
    assert pypto_attention._read_pypto_dfx_dir() == (tmp_path / "dfx").resolve()


def test_dfx_rejects_acl_graph(tmp_path):
    vllm_config = SimpleNamespace(
        compilation_config=SimpleNamespace(
            mode=pypto_attention.CompilationMode.NONE,
            cudagraph_mode=pypto_attention.CUDAGraphMode.FULL_DECODE_ONLY,
        )
    )

    with pytest.raises(ValueError, match="ACL Graph to be disabled"):
        pypto_attention._validate_pypto_dfx_graph_mode(vllm_config, tmp_path)

    vllm_config.compilation_config.cudagraph_mode = pypto_attention.CUDAGraphMode.NONE
    pypto_attention._validate_pypto_dfx_graph_mode(vllm_config, tmp_path)

    vllm_config.compilation_config.mode = pypto_attention.CompilationMode.VLLM_COMPILE
    with pytest.raises(ValueError, match="torch.compile and ACL Graph"):
        pypto_attention._validate_pypto_dfx_graph_mode(vllm_config, tmp_path)


def test_dfx_brackets_only_the_first_selected_layer_decode(monkeypatch):
    events = []

    def kernel(*args):
        events.append("kernel")
        return args[-1].fill_(5)

    attention = _attention(kernel)
    dfx_runtime = SimpleNamespace(
        begin_dfx=lambda: events.append("begin"),
        end_dfx=lambda: events.append("end"),
    )
    pypto_attention._capture_next_kernel_call(attention, kernel, dfx_runtime)
    cache = _cache()
    batch = 1
    metadata = SimpleNamespace(
        attn_state=SimpleNamespace(name="DecodeOnly"),
        num_actual_tokens=batch,
        block_tables=torch.zeros((batch, 4), dtype=torch.int32),
        slot_mapping=None,
    )
    monkeypatch.setattr(
        pypto_attention,
        "get_attention_context",
        lambda _layer_name: (metadata, object(), cache, torch.zeros(batch, dtype=torch.int64)),
    )
    for _ in range(2):
        output = pypto_attention._try_pypto_decode_attention(
            attention,
            torch.zeros(batch, dtype=torch.int64),
            torch.zeros((batch, 8), dtype=torch.bfloat16),
        )
        torch.testing.assert_close(output, torch.full_like(output, 5))

    assert events == ["begin", "kernel", "end", "kernel"]
    assert attention._pypto_kernel is kernel


def test_kv_cache_views_reuse_framework_storage():
    cache = _cache()

    key, value = pypto_attention._kv_cache_views(cache)

    assert key.shape == (
        pypto_attention.BLOCK_SIZE * pypto_attention.NUM_KV_HEADS,
        pypto_attention.HEAD_DIM,
    )
    assert key.data_ptr() == cache[0].data_ptr()
    assert value.data_ptr() == cache[1].data_ptr()


def test_kv_cache_views_reject_unexpected_layout():
    bad_cache = torch.zeros((2, 1, 64, 8, 128), dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="cache tail"):
        pypto_attention._kv_cache_views(bad_cache)


def test_decode_attention_passes_direct_cache_views(monkeypatch):
    calls = []

    def kernel(*args):
        calls.append(args)
        return args[-1].fill_(3)

    attention = _attention(kernel)
    cache = _cache()
    batch = 2
    block_tables = torch.zeros((batch, 4), dtype=torch.int32)
    slots = torch.tensor([0, 128], dtype=torch.int64)
    metadata = SimpleNamespace(
        attn_state=SimpleNamespace(name="DecodeOnly"),
        num_actual_tokens=batch,
        block_tables=block_tables,
        slot_mapping=None,
    )
    monkeypatch.setattr(
        pypto_attention,
        "get_attention_context",
        lambda _layer_name: (metadata, object(), cache, slots),
    )
    hidden = torch.zeros((batch, 8), dtype=torch.bfloat16)
    positions = torch.tensor([3, 7], dtype=torch.int64)

    output = pypto_attention._try_pypto_decode_attention(attention, positions, hidden)

    assert calls
    assert output is not None
    torch.testing.assert_close(output, torch.full_like(hidden, 3))
    assert calls[0][9].data_ptr() == cache[0].data_ptr()
    assert calls[0][10].data_ptr() == cache[1].data_ptr()
    assert calls[0][5].data_ptr() == block_tables.data_ptr()
    assert calls[0][6].data_ptr() == slots.data_ptr()


def test_decode_attention_uses_backend_neutral_op_while_compiling(monkeypatch):
    direct_calls = []
    compiled_calls = []

    attention = _attention(lambda *args: direct_calls.append(args))
    cache = _cache()
    batch = 2
    metadata = SimpleNamespace(
        attn_state=SimpleNamespace(name="DecodeOnly"),
        num_actual_tokens=batch,
        block_tables=torch.zeros((batch, 4), dtype=torch.int32),
        slot_mapping=None,
    )
    monkeypatch.setattr(
        pypto_attention,
        "get_attention_context",
        lambda _layer_name: (metadata, object(), cache, torch.zeros(batch, dtype=torch.int64)),
    )
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)

    def compiled_op(*args):
        compiled_calls.append(args)
        return torch.full_like(args[0], 4)

    monkeypatch.setattr(pypto_attention, "qwen3_decode_attention", compiled_op)
    hidden = torch.zeros((batch, 8), dtype=torch.bfloat16)

    output = pypto_attention._try_pypto_decode_attention(
        attention,
        torch.zeros(batch, dtype=torch.int64),
        hidden,
    )

    assert not direct_calls
    assert compiled_calls
    torch.testing.assert_close(output, torch.full_like(hidden, 4))


@pytest.mark.parametrize(
    ("state", "actual_tokens"),
    [("ChunkedPrefill", 2), ("DecodeOnly", 1)],
)
def test_decode_attention_falls_back_for_non_decode_or_padded_batch(
    monkeypatch,
    state,
    actual_tokens,
):
    attention = _attention(lambda *args: pytest.fail("PyPTO kernel must not run"))
    metadata = SimpleNamespace(
        attn_state=SimpleNamespace(name=state),
        num_actual_tokens=actual_tokens,
        block_tables=torch.zeros((2, 4), dtype=torch.int32),
        slot_mapping=None,
    )
    monkeypatch.setattr(
        pypto_attention,
        "get_attention_context",
        lambda _layer_name: (metadata, object(), _cache(), torch.zeros(2, dtype=torch.int64)),
    )

    output = pypto_attention._try_pypto_decode_attention(
        attention,
        torch.zeros(2, dtype=torch.int64),
        torch.zeros((2, 8), dtype=torch.bfloat16),
    )

    assert output is None
