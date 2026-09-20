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

"""Backend-neutral graph contract for Qwen3-14B decode attention.

The Torch Library operator defined here represents the model-level mega-kernel
boundary.  Its Fake implementation, tensor contract and mutation declaration
are intentionally independent of PyPTO, AscendC or any future implementation.
A process selects one concrete implementation after model weights are loaded.
"""

from collections.abc import Callable
from threading import Lock
from typing import Any

import torch
from vllm.utils.torch_utils import direct_register_custom_op

OP_NAME = "qwen3_14b_decode_attention"

BATCH_PAD = 16
BLOCK_SIZE = 128
MAX_SEQ_LEN = 4096
NUM_HEADS = 40
NUM_KV_HEADS = 8
HEAD_DIM = 128
HIDDEN_SIZE = NUM_HEADS * HEAD_DIM
KV_HIDDEN_SIZE = NUM_KV_HEADS * HEAD_DIM
QKV_HIDDEN_SIZE = HIDDEN_SIZE + 2 * KV_HIDDEN_SIZE

AttentionImpl = Callable[..., Any]

_impl_lock = Lock()
_attention_impl: AttentionImpl | None = None
_attention_impl_name: str | None = None


def bind_qwen3_decode_attention_impl(name: str, impl: AttentionImpl) -> None:
    """Bind one runtime implementation without changing the graph contract."""
    if not name:
        raise ValueError("Qwen3 decode attention implementation name must not be empty")
    if not callable(impl):
        raise TypeError(f"Qwen3 decode attention implementation must be callable, got {type(impl)!r}")

    global _attention_impl, _attention_impl_name
    with _impl_lock:
        if _attention_impl is not None and (_attention_impl is not impl or _attention_impl_name != name):
            raise RuntimeError(
                "Qwen3 decode attention already has an active implementation: "
                f"{_attention_impl_name!r}; cannot replace it with {name!r}"
            )
        _attention_impl = impl
        _attention_impl_name = name


def _check_dtype(name: str, tensor: torch.Tensor, dtype: torch.dtype) -> None:
    if tensor.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}, got {tensor.dtype}")


def _check_rank(name: str, tensor: torch.Tensor, rank: int) -> None:
    torch._check(tensor.ndim == rank, lambda: f"{name} must be rank {rank}")


def _check_qwen3_decode_attention_contract(  # noqa: PLR0913
    normalized_hidden: torch.Tensor,
    qkv_weight: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    positions: torch.Tensor,
    block_table: torch.Tensor,
    slot_mapping: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    o_proj_weight: torch.Tensor,
) -> None:
    """Validate only the model ABI; backend-specific layout checks stay local."""
    tensors = {
        "normalized_hidden": (normalized_hidden, torch.bfloat16, 2),
        "qkv_weight": (qkv_weight, torch.bfloat16, 2),
        "q_norm_weight": (q_norm_weight, torch.float32, 2),
        "k_norm_weight": (k_norm_weight, torch.float32, 2),
        "positions": (positions, torch.int64, 1),
        "block_table": (block_table, torch.int32, 2),
        "slot_mapping": (slot_mapping, torch.int64, 1),
        "rope_cos": (rope_cos, torch.float32, 2),
        "rope_sin": (rope_sin, torch.float32, 2),
        "k_cache": (k_cache, torch.bfloat16, 2),
        "v_cache": (v_cache, torch.bfloat16, 2),
        "o_proj_weight": (o_proj_weight, torch.bfloat16, 2),
    }
    for name, (tensor, dtype, rank) in tensors.items():
        _check_dtype(name, tensor, dtype)
        _check_rank(name, tensor, rank)

    batch = normalized_hidden.shape[0]
    torch._check(batch <= BATCH_PAD, lambda: f"batch must be <= {BATCH_PAD}")
    torch._check(normalized_hidden.shape[1] == HIDDEN_SIZE, lambda: "normalized_hidden width is invalid")
    torch._check(qkv_weight.shape[0] == QKV_HIDDEN_SIZE, lambda: "qkv_weight output width is invalid")
    torch._check(qkv_weight.shape[1] == HIDDEN_SIZE, lambda: "qkv_weight input width is invalid")
    torch._check(q_norm_weight.shape[0] == 1, lambda: "q_norm_weight leading dimension is invalid")
    torch._check(q_norm_weight.shape[1] == HEAD_DIM, lambda: "q_norm_weight width is invalid")
    torch._check(k_norm_weight.shape[0] == 1, lambda: "k_norm_weight leading dimension is invalid")
    torch._check(k_norm_weight.shape[1] == HEAD_DIM, lambda: "k_norm_weight width is invalid")
    torch._check(positions.shape[0] == batch, lambda: "positions batch does not match hidden states")
    torch._check(block_table.shape[0] == batch, lambda: "block_table batch does not match hidden states")
    torch._check(slot_mapping.shape[0] == batch, lambda: "slot_mapping batch does not match hidden states")
    torch._check(rope_cos.shape[0] >= MAX_SEQ_LEN, lambda: "rope_cos sequence dimension is too short")
    torch._check(rope_cos.shape[1] == HEAD_DIM, lambda: "rope_cos width is invalid")
    torch._check(rope_sin.shape[0] == rope_cos.shape[0], lambda: "RoPE table lengths do not match")
    torch._check(rope_sin.shape[1] == HEAD_DIM, lambda: "rope_sin width is invalid")
    torch._check(k_cache.shape[0] == v_cache.shape[0], lambda: "K/V cache row counts do not match")
    torch._check(k_cache.shape[1] == HEAD_DIM, lambda: "K cache width is invalid")
    torch._check(v_cache.shape[1] == HEAD_DIM, lambda: "V cache width is invalid")
    torch._check(
        k_cache.shape[0] % (BLOCK_SIZE * NUM_KV_HEADS) == 0,
        lambda: "K/V cache rows do not contain complete pages",
    )
    torch._check(o_proj_weight.shape[0] == HIDDEN_SIZE, lambda: "o_proj_weight output width is invalid")
    torch._check(o_proj_weight.shape[1] == HIDDEN_SIZE, lambda: "o_proj_weight input width is invalid")


def _qwen3_decode_attention_impl(  # noqa: PLR0913
    normalized_hidden: torch.Tensor,
    qkv_weight: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    positions: torch.Tensor,
    block_table: torch.Tensor,
    slot_mapping: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    o_proj_weight: torch.Tensor,
) -> torch.Tensor:
    _check_qwen3_decode_attention_contract(
        normalized_hidden,
        qkv_weight,
        q_norm_weight,
        k_norm_weight,
        positions,
        block_table,
        slot_mapping,
        rope_cos,
        rope_sin,
        k_cache,
        v_cache,
        o_proj_weight,
    )
    if _attention_impl is None:
        raise RuntimeError("No Qwen3 decode attention implementation has been bound")
    attention_out = torch.empty_like(normalized_hidden)
    _attention_impl(
        normalized_hidden,
        qkv_weight,
        q_norm_weight,
        k_norm_weight,
        positions,
        block_table,
        slot_mapping,
        rope_cos,
        rope_sin,
        k_cache,
        v_cache,
        o_proj_weight,
        attention_out,
    )
    return attention_out


def _qwen3_decode_attention_fake(  # noqa: PLR0913
    normalized_hidden: torch.Tensor,
    qkv_weight: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    positions: torch.Tensor,
    block_table: torch.Tensor,
    slot_mapping: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    o_proj_weight: torch.Tensor,
) -> torch.Tensor:
    _check_qwen3_decode_attention_contract(
        normalized_hidden,
        qkv_weight,
        q_norm_weight,
        k_norm_weight,
        positions,
        block_table,
        slot_mapping,
        rope_cos,
        rope_sin,
        k_cache,
        v_cache,
        o_proj_weight,
    )
    return torch.empty_like(normalized_hidden)


def qwen3_decode_attention(  # noqa: PLR0913
    normalized_hidden: torch.Tensor,
    qkv_weight: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    positions: torch.Tensor,
    block_table: torch.Tensor,
    slot_mapping: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    o_proj_weight: torch.Tensor,
) -> torch.Tensor:
    """Run the active backend through the backend-neutral Torch Library node."""
    return torch.ops.vllm.qwen3_14b_decode_attention(
        normalized_hidden,
        qkv_weight,
        q_norm_weight,
        k_norm_weight,
        positions,
        block_table,
        slot_mapping,
        rope_cos,
        rope_sin,
        k_cache,
        v_cache,
        o_proj_weight,
    )


direct_register_custom_op(
    op_name=OP_NAME,
    op_func=_qwen3_decode_attention_impl,
    mutates_args=["k_cache", "v_cache"],
    fake_impl=_qwen3_decode_attention_fake,
    dispatch_key="PrivateUse1",
)


__all__ = [
    "bind_qwen3_decode_attention_impl",
    "qwen3_decode_attention",
]
