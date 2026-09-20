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

"""Minimal Qwen3-14B attention replacement backed by PyPTO.

The native ``Qwen3ForCausalLM`` model, decoder layers, normalization, MLP and
weight-loading paths remain unchanged. This module only replaces each
``Qwen3Attention.forward`` method. Prefill calls the native method directly;
eligible decode-only batches call the PyPTO attention program.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from types import MethodType
from typing import Any

import torch
from vllm.config import CompilationMode, CUDAGraphMode, VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.logger import logger
from vllm.model_executor.layers.attention.attention import get_attention_context
from vllm.model_executor.models.qwen3 import Qwen3Attention, Qwen3ForCausalLM

from vllm_ascend import envs
from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ops.qwen3_decode_attention import (
    bind_qwen3_decode_attention_impl,
    qwen3_decode_attention,
)

BATCH_PAD = 16
BLOCK_SIZE = 128
MAX_SEQ_LEN = 4096
NUM_LAYERS = 40
NUM_HEADS = 40
NUM_KV_HEADS = 8
HEAD_DIM = 128
HIDDEN_SIZE = 5120
KV_HIDDEN_SIZE = NUM_KV_HEADS * HEAD_DIM
QKV_HIDDEN_SIZE = HIDDEN_SIZE + 2 * KV_HIDDEN_SIZE
ROPE_THETA = 1_000_000.0


def _read_pypto_dfx_dir() -> Path | None:
    output_dir = envs.VLLM_ASCEND_QWEN3_PYPTO_DFX_DIR
    return Path(output_dir).expanduser().resolve() if output_dir else None


def _validate_pypto_dfx_graph_mode(vllm_config: VllmConfig, dfx_dir: Path | None) -> None:
    if dfx_dir is None:
        return
    compilation_config = vllm_config.compilation_config
    if compilation_config.mode != CompilationMode.NONE or compilation_config.cudagraph_mode != CUDAGraphMode.NONE:
        raise ValueError(
            "PyPTO DFX requires torch.compile and ACL Graph to be disabled because "
            "pypto.torch.begin_dfx()/end_dfx() synchronize the current stream; "
            "set compilation_config.mode=0 and compilation_config.cudagraph_mode='NONE'"
        )


def _load_attention_kernel() -> Any:
    module = importlib.import_module("vllm_ascend.ops.pypto.qwen3_14b.decode_attention")
    return module.qwen3_decode_attention_layer


def _build_rope_tables(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    half_dim = HEAD_DIM // 2
    inv_freq = 1.0 / (ROPE_THETA ** (torch.arange(half_dim, dtype=torch.float32) / float(half_dim)))
    positions = torch.arange(MAX_SEQ_LEN, dtype=torch.float32)
    frequencies = torch.outer(positions, inv_freq)
    angles = torch.cat((frequencies, frequencies), dim=-1)
    return (
        angles.cos().contiguous().to(device=device),
        angles.sin().contiguous().to(device=device),
    )


def _kv_cache_views(kv_cache: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """Return direct ND ``[pages * block * heads, dim]`` K/V cache views."""
    if isinstance(kv_cache, (tuple, list)):
        if len(kv_cache) != 2:
            raise ValueError(f"expected a K/V pair, got {len(kv_cache)} cache tensors")
        key_cache, value_cache = kv_cache
    elif isinstance(kv_cache, torch.Tensor) and kv_cache.ndim == 5 and kv_cache.shape[0] == 2:
        key_cache, value_cache = kv_cache[0], kv_cache[1]
    else:
        shape = tuple(kv_cache.shape) if isinstance(kv_cache, torch.Tensor) else None
        raise ValueError(f"unsupported vLLM KV cache type={type(kv_cache)!r}, shape={shape}")

    expected_tail = (BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM)
    for name, cache in (("key", key_cache), ("value", value_cache)):
        if not isinstance(cache, torch.Tensor) or cache.ndim != 4:
            raise ValueError(f"{name} cache must be a rank-4 tensor")
        if tuple(cache.shape[1:]) != expected_tail:
            raise ValueError(f"{name} cache tail must be {expected_tail}, got {tuple(cache.shape[1:])}")
        if not cache.is_contiguous():
            raise ValueError(f"{name} cache must use contiguous ND storage")
    return key_cache.view(-1, HEAD_DIM), value_cache.view(-1, HEAD_DIM)


def _prepare_attention(
    attention: Qwen3Attention,
    kernel: Any,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
) -> None:
    qkv_weight = attention.qkv_proj.weight
    o_proj_weight = attention.o_proj.weight
    if tuple(qkv_weight.shape) != (QKV_HIDDEN_SIZE, HIDDEN_SIZE):
        raise ValueError(f"unexpected packed QKV shape: {tuple(qkv_weight.shape)}")
    if tuple(o_proj_weight.shape) != (HIDDEN_SIZE, HIDDEN_SIZE):
        raise ValueError(f"unexpected O projection shape: {tuple(o_proj_weight.shape)}")
    if qkv_weight.dtype != torch.bfloat16 or o_proj_weight.dtype != torch.bfloat16:
        raise TypeError("PyPTO Qwen3 attention requires BF16 QKV and O projection weights")
    if not qkv_weight.is_contiguous() or not o_proj_weight.is_contiguous():
        raise ValueError("PyPTO Qwen3 attention requires contiguous ND weights; set additional_config.weight_nz_mode=0")
    if getattr(attention.qkv_proj, "bias", None) is not None:
        raise ValueError("Qwen3 PyPTO attention does not support QKV bias")

    attention._pypto_q_norm = attention.q_norm.weight.detach().float().view(1, HEAD_DIM).contiguous()
    attention._pypto_k_norm = attention.k_norm.weight.detach().float().view(1, HEAD_DIM).contiguous()
    attention._pypto_kernel = kernel
    attention._pypto_rope_cos = rope_cos
    attention._pypto_rope_sin = rope_sin


def _try_pypto_decode_attention(
    attention: Qwen3Attention,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
) -> torch.Tensor | None:
    kernel = getattr(attention, "_pypto_kernel", None)
    if kernel is None:
        return None
    metadata, _, kv_cache, layer_slot_mapping = get_attention_context(attention.attn.layer_name)
    if metadata is None or getattr(metadata.attn_state, "name", None) != "DecodeOnly":
        return None
    if hidden_states.ndim != 2 or positions.ndim != 1:
        return None
    batch = hidden_states.shape[0]
    if batch > BATCH_PAD or positions.shape[0] != batch:
        return None
    if int(metadata.num_actual_tokens) != batch:
        # Native attention owns padded -1 slots until the PyPTO ABI gains an
        # explicit padding mask.
        return None

    block_table = metadata.block_tables
    slot_mapping = layer_slot_mapping
    if slot_mapping is None:
        slot_mapping = metadata.slot_mapping
    if block_table is None or slot_mapping is None:
        return None

    key_cache, value_cache = _kv_cache_views(kv_cache)
    block_table = block_table[:batch]
    slot_mapping = slot_mapping[:batch]
    if not block_table.is_contiguous() or not slot_mapping.is_contiguous():
        raise ValueError("block table and slot mapping must be contiguous")

    args = (
        hidden_states,
        attention.qkv_proj.weight,
        attention._pypto_q_norm,
        attention._pypto_k_norm,
        positions,
        block_table,
        slot_mapping,
        attention._pypto_rope_cos,
        attention._pypto_rope_sin,
        key_cache,
        value_cache,
        attention.o_proj.weight,
    )
    if torch.compiler.is_compiling():
        return qwen3_decode_attention(*args)
    output = torch.empty_like(hidden_states)
    return kernel(*args, output)


def _pypto_attention_forward(
    attention: Qwen3Attention,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    output = _try_pypto_decode_attention(attention, positions, hidden_states)
    if output is not None:
        return output
    return Qwen3Attention.forward(attention, positions, hidden_states)


def _replace_attention_forward(attention: Qwen3Attention) -> None:
    """Replace only this native attention instance's Python forward method."""
    attention._pypto_kernel = None
    attention._pypto_q_norm = None
    attention._pypto_k_norm = None
    attention._pypto_rope_cos = None
    attention._pypto_rope_sin = None
    attention.forward = MethodType(_pypto_attention_forward, attention)


def _capture_next_kernel_call(
    attention: Qwen3Attention,
    kernel: Any,
    dfx_runtime: Any,
) -> None:
    """Bracket one direct kernel call, then restore the zero-overhead path."""

    def capture_once(*args: Any) -> Any:
        dfx_runtime.begin_dfx()
        try:
            return kernel(*args)
        finally:
            try:
                dfx_runtime.end_dfx()
            finally:
                attention._pypto_kernel = kernel

    attention._pypto_kernel = capture_once


def _iter_attentions(model: Qwen3ForCausalLM) -> list[Qwen3Attention]:
    attentions: list[Qwen3Attention] = []
    for layer in model.model.layers:
        attention = getattr(layer, "self_attn", None)
        if not isinstance(attention, Qwen3Attention):
            raise TypeError(f"unexpected Qwen3 attention type: {type(attention)!r}")
        attentions.append(attention)
    return attentions


class PyptoAttentionQwen3ForCausalLM(Qwen3ForCausalLM):
    """Native TP1 Qwen3-14B with only decode attention replaced."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        dfx_dir = _read_pypto_dfx_dir()
        self._validate_configuration(vllm_config, dfx_dir)
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        attentions = _iter_attentions(self)
        for attention in attentions:
            _replace_attention_forward(attention)
        self._pypto_dfx_dir = dfx_dir

    @staticmethod
    def _validate_configuration(
        vllm_config: VllmConfig,
        dfx_dir: Path | None = None,
    ) -> None:
        _validate_pypto_dfx_graph_mode(vllm_config, dfx_dir)
        config = vllm_config.model_config.hf_config.get_text_config()
        expected = {
            "hidden_size": HIDDEN_SIZE,
            "num_hidden_layers": NUM_LAYERS,
            "num_attention_heads": NUM_HEADS,
            "num_key_value_heads": NUM_KV_HEADS,
            "head_dim": HEAD_DIM,
        }
        mismatches = {
            name: (getattr(config, name, None), value)
            for name, value in expected.items()
            if getattr(config, name, None) != value
        }
        if mismatches:
            raise ValueError(f"PyPTO attention path only supports Qwen3-14B: {mismatches}")
        if vllm_config.quant_config is not None:
            raise ValueError("PyPTO Qwen3 attention currently supports unquantized BF16 only")
        if get_tensor_model_parallel_world_size() != 1:
            raise ValueError("PyPTO Qwen3 attention currently supports tensor_parallel_size=1 only")
        if vllm_config.parallel_config.pipeline_parallel_size != 1:
            raise ValueError("PyPTO Qwen3 attention currently supports pipeline_parallel_size=1 only")
        if vllm_config.cache_config.block_size != BLOCK_SIZE:
            raise ValueError(f"PyPTO Qwen3 attention requires block_size={BLOCK_SIZE}")
        if vllm_config.model_config.max_model_len > MAX_SEQ_LEN:
            raise ValueError(f"PyPTO Qwen3 attention requires max_model_len <= {MAX_SEQ_LEN}")
        if vllm_config.scheduler_config.max_num_seqs > BATCH_PAD:
            raise ValueError(f"PyPTO Qwen3 attention requires max_num_seqs <= {BATCH_PAD}")
        if vllm_config.model_config.dtype != torch.bfloat16:
            raise TypeError("PyPTO Qwen3 attention requires model dtype bfloat16")
        ascend_config = get_ascend_config()
        if ascend_config.weight_nz_mode != 0:
            raise ValueError("PyPTO Qwen3 attention requires additional_config.weight_nz_mode=0")
        if ascend_config.enable_kv_nz:
            raise ValueError("PyPTO Qwen3 attention requires additional_config.enable_kv_nz=false")
        if vllm_config.speculative_config is not None:
            raise ValueError("PyPTO Qwen3 attention does not yet support speculative decoding")

    def process_weights_after_loading(self) -> None:
        if not torch.npu.is_available():
            raise RuntimeError("PyPTO Qwen3 attention requires an available NPU")
        import pypto  # noqa: PLC0415
        import pypto.torch  # noqa: PLC0415

        kernel = _load_attention_kernel()
        init_kwargs: dict[str, Any] = {
            "device": torch.npu.current_device(),
            "platform": "a2a3",
            "runtime": "tensormap_and_ringbuffer",
        }
        if self._pypto_dfx_dir is not None:
            init_kwargs.update(
                enable_chip_swimlane=4,
                enable_dep_gen=True,
                output_dir=self._pypto_dfx_dir,
            )
        pypto.torch.init(**init_kwargs)
        bind_qwen3_decode_attention_impl("pypto", kernel)
        device = next(self.parameters()).device
        rope_cos, rope_sin = _build_rope_tables(device)
        attentions = _iter_attentions(self)
        for attention in attentions:
            _prepare_attention(attention, kernel, rope_cos, rope_sin)
        self._warm_pypto_kernel(kernel, attentions[0], rope_cos, rope_sin)
        if self._pypto_dfx_dir is not None:
            _capture_next_kernel_call(attentions[0], kernel, pypto.torch)
        logger.info(
            "Initialized pto_eager Qwen3 attention for %d layers from %s",
            len(attentions),
            Path(pypto.__file__).resolve(),
        )
        if self._pypto_dfx_dir is not None:
            logger.info(
                "PyPTO DFX will capture layer 0 on its first decode call: %s",
                self._pypto_dfx_dir,
            )

    @staticmethod
    @torch.inference_mode()
    def _warm_pypto_kernel(
        kernel: Any,
        attention: Qwen3Attention,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
    ) -> None:
        """Build the dynamic PyPTO specialization before ACL Graph capture."""
        device = attention.qkv_proj.weight.device
        hidden = torch.zeros((BATCH_PAD, HIDDEN_SIZE), dtype=torch.bfloat16, device=device)
        positions = torch.zeros(BATCH_PAD, dtype=torch.int64, device=device)
        block_table = torch.arange(BATCH_PAD, dtype=torch.int32, device=device).view(BATCH_PAD, 1)
        slot_mapping = torch.arange(BATCH_PAD, dtype=torch.int64, device=device) * BLOCK_SIZE
        cache_shape = (BATCH_PAD, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM)
        key_cache = torch.zeros(cache_shape, dtype=torch.bfloat16, device=device).view(-1, HEAD_DIM)
        value_cache = torch.zeros(cache_shape, dtype=torch.bfloat16, device=device).view(-1, HEAD_DIM)
        output = torch.empty_like(hidden)
        kernel(
            hidden,
            attention.qkv_proj.weight,
            attention._pypto_q_norm,
            attention._pypto_k_norm,
            positions,
            block_table,
            slot_mapping,
            rope_cos,
            rope_sin,
            key_cache,
            value_cache,
            attention.o_proj.weight,
            output,
        )
        torch.npu.synchronize()


__all__ = ["PyptoAttentionQwen3ForCausalLM"]
