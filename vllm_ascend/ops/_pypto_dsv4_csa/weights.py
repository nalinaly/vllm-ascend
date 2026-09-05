# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Cold-path weight packing for the DeepSeek V4 Flash decode CSA entry."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

import torch

from .config import FLASH


class DecodeCSAWeightContractError(ValueError):
    """Raised when a live vLLM layer cannot satisfy the current kernel ABI."""


@dataclass(frozen=True, slots=True)
class PreparedDecodeCSAWeights:
    """Layer-specific packed tensors retained for the process-pinned L1 owner."""

    launch_arguments: Mapping[str, torch.Tensor]
    source_owner: object

    def for_launch(self) -> Mapping[str, torch.Tensor]:
        return self.launch_arguments


ND_FORMAT = 2


def _default_format_nd(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.device.type != "npu":
        return tensor
    torch_npu = importlib.import_module("torch_npu")
    return torch_npu.npu_format_cast(tensor, ND_FORMAT)


def _require_tensor(value: object, *, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise DecodeCSAWeightContractError(f"{name}: expected torch.Tensor, got {type(value).__name__}")
    return value


def _weight(module: object, *, name: str) -> torch.Tensor:
    return _require_tensor(getattr(module, "weight", None), name=f"{name}.weight")


def _require_no_bias(module: object, *, name: str) -> None:
    if getattr(module, "bias", None) is not None:
        raise DecodeCSAWeightContractError(f"{name}: bias is unsupported by the 44-parameter CSA ABI")


def _require_no_norm_bias(module: object, *, name: str) -> None:
    if getattr(module, "bias", None) is not None:
        raise DecodeCSAWeightContractError(
            f"{name}: RMSNorm bias/anti-method m4 is unsupported by the production CSA ABI"
        )


def _require_w8_dynamic_symmetric(module: object, *, name: str) -> None:
    quant_method = getattr(module, "quant_method", None)
    inner = getattr(quant_method, "quant_method", None)
    if type(inner).__name__ != "AscendW8A8DynamicLinearMethod":
        raise DecodeCSAWeightContractError(
            f"{name}: expected AscendW8A8DynamicLinearMethod, got "
            f"{type(inner).__name__ if inner is not None else type(quant_method).__name__}"
        )
    offset = _require_tensor(getattr(module, "weight_offset", None), name=f"{name}.weight_offset")
    # This is a cold-path check.  The kernel has no zero-point argument, so a
    # non-zero offset must fail rather than silently change quant semantics.
    if int(torch.count_nonzero(offset).item()) != 0:
        raise DecodeCSAWeightContractError(f"{name}: only symmetric W8 weights with zero offset are supported")


def _pack_matrix(
    tensor: torch.Tensor,
    *,
    name: str,
    shape: tuple[int, int],
    dtype: torch.dtype,
    transpose_if_reversed: bool,
    format_nd: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    if tensor.dtype != dtype:
        raise DecodeCSAWeightContractError(f"{name}: expected dtype={dtype}, got {tensor.dtype}")
    packed = format_nd(tensor)
    actual = tuple(int(extent) for extent in packed.shape)
    if actual == shape:
        pass
    elif transpose_if_reversed and actual == (shape[1], shape[0]):
        packed = packed.transpose(0, 1)
    else:
        raise DecodeCSAWeightContractError(f"{name}: expected shape={shape}, got {actual}")
    # Model parameters commonly keep requires_grad=True even inside
    # inference_mode.  L1 launch tensors must not carry autograd state, and
    # contiguous() is a no-op for an already-canonical Parameter, so detach is
    # an explicit part of the cold packed-weight contract.
    packed = packed.detach().contiguous()
    if tuple(packed.shape) != shape or not packed.is_contiguous():
        raise DecodeCSAWeightContractError(f"{name}: failed to materialize canonical ND shape={shape}")
    return packed


def _pack_exact(
    tensor: torch.Tensor,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    format_nd: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    if tensor.dtype != dtype:
        raise DecodeCSAWeightContractError(f"{name}: expected dtype={dtype}, got {tensor.dtype}")
    packed = format_nd(tensor)
    if tuple(int(extent) for extent in packed.shape) != shape:
        raise DecodeCSAWeightContractError(f"{name}: expected shape={shape}, got {tuple(packed.shape)}")
    return packed.detach().contiguous()


def _pack_proj_b_tile_major(
    tensor: torch.Tensor,
    *,
    groups: int,
    o_lora: int,
    n_tile: int,
    k_tile: int,
) -> torch.Tensor:
    """Reorder ``[D, groups * K]`` for contiguous proj-b cube tile loads.

    The public 44-slot ABI deliberately keeps the original two-dimensional
    shape.  Only its physical element order changes from ``[D, G, K]`` to
    ``[G, D/N, K/BK, N, BK]``.  The kernel views the same storage as rows of
    ``BK`` elements and consumes one contiguous ``[N, BK]`` tile at a time.
    This is a cold-path transform; the returned tensor is pinned by the
    prepared-weight owner for every subsequent eager launch/graph replay.
    """
    if tensor.ndim != 2:
        raise DecodeCSAWeightContractError(f"wo_b: tile-major packing expects rank 2, got shape={tuple(tensor.shape)}")
    d, grouped_k = (int(extent) for extent in tensor.shape)
    if grouped_k != groups * o_lora:
        raise DecodeCSAWeightContractError(
            f"wo_b: tile-major packing expected second dimension {groups * o_lora}, got {grouped_k}"
        )
    if d % n_tile != 0 or o_lora % k_tile != 0:
        raise DecodeCSAWeightContractError(
            f"wo_b: tile-major packing requires exact tiles, got D={d}, K={o_lora}, N-tile={n_tile}, K-tile={k_tile}"
        )
    return (
        tensor.reshape(d // n_tile, n_tile, groups, o_lora // k_tile, k_tile)
        .permute(2, 0, 3, 1, 4)
        .contiguous()
        .reshape(d, grouped_k)
        .detach()
    )


def _scale(module: object, *, name: str, elements: int) -> torch.Tensor:
    value = getattr(module, "weight_scale_fp32", None)
    if value is None:
        value = getattr(module, "weight_scale", None)
    tensor = _require_tensor(value, name=f"{name}.weight_scale_fp32")
    if tensor.dtype != torch.float32:
        raise DecodeCSAWeightContractError(
            f"{name}.weight_scale_fp32: expected dtype=torch.float32, got {tensor.dtype}"
        )
    if tensor.numel() != elements:
        raise DecodeCSAWeightContractError(
            f"{name}.weight_scale_fp32: expected {elements} values, got {tensor.numel()}"
        )
    return tensor.detach().reshape(elements).contiguous()


def _pack_rope(tensor: torch.Tensor, *, name: str) -> torch.Tensor:
    if tensor.device.type not in {"cpu", "npu"}:
        raise DecodeCSAWeightContractError(f"{name}: unsupported device {tensor.device}")
    if tensor.dtype != torch.float32:
        raise DecodeCSAWeightContractError(f"{name}: expected dtype=torch.float32, got {tensor.dtype}")
    if tensor.shape[-1] != FLASH.qk_rope_head_dim:
        raise DecodeCSAWeightContractError(
            f"{name}: expected last dim={FLASH.qk_rope_head_dim}, got shape={tuple(tensor.shape)}"
        )
    rows = tensor.reshape(-1, FLASH.qk_rope_head_dim)
    if rows.shape[0] < FLASH.max_position_embeddings:
        raise DecodeCSAWeightContractError(
            f"{name}: expected at least {FLASH.max_position_embeddings} rows, got {rows.shape[0]}"
        )
    # ComplexExpRotaryEmbedding owns a full-width FP32 cache.  Preserve that
    # native arithmetic boundary (and normally its storage) instead of making
    # a BF16 mirror that can move dynamic-quantized indexer values by one bin.
    return rows[: FLASH.max_position_embeddings].detach().contiguous()


def _validate_layer_topology(layer: object) -> None:
    expected = {
        "dim": FLASH.hidden_size,
        "n_heads": FLASH.num_attention_heads,
        "n_local_heads": FLASH.num_attention_heads,
        "q_lora_rank": FLASH.q_lora_rank,
        "o_lora_rank": FLASH.o_lora_rank,
        "head_dim": FLASH.head_dim,
        "rope_head_dim": FLASH.qk_rope_head_dim,
        "n_groups": FLASH.o_groups,
        "n_local_groups": FLASH.o_groups,
        "window_size": FLASH.sliding_window,
        "compress_ratio": 4,
    }
    mismatches = {
        name: {"expected": value, "actual": getattr(layer, name, None)}
        for name, value in expected.items()
        if getattr(layer, name, None) != value
    }
    if mismatches:
        raise DecodeCSAWeightContractError(
            f"the current kernel is a TP1/local-H64/local-G8 Flash specialization; layer topology mismatch={mismatches}"
        )
    if bool(getattr(layer, "skip_topk", False)):
        raise DecodeCSAWeightContractError("skip_topk/index-cache reuse is unsupported")
    impl = getattr(getattr(layer, "dsa_attn", None), "impl", None)
    if impl is not None and type(impl).__name__ != "AscendDSAImpl":
        raise DecodeCSAWeightContractError(
            f"only the non-CP AscendDSAImpl metadata protocol is supported, got {type(impl).__name__}"
        )
    if impl is not None and bool(getattr(impl, "multistream_dsv4_dsa_overlap", False)):
        raise DecodeCSAWeightContractError("native DSA multistream overlap must be disabled for the single L1 operator")
    if getattr(layer, "indexer", None) is None or getattr(layer, "compressor", None) is None:
        raise DecodeCSAWeightContractError("ratio-4 layer must own both compressor and indexer modules")
    model_config = getattr(getattr(impl, "vllm_config", None), "model_config", None)
    max_model_len = getattr(model_config, "max_model_len", None)
    if max_model_len is not None and int(max_model_len) > FLASH.max_position_embeddings:
        raise DecodeCSAWeightContractError(
            f"model max length {max_model_len} exceeds the current PyPTO limit {FLASH.max_position_embeddings}"
        )


def pack_decode_csa_weights(
    layer: object,
    *,
    full_rope_cos: torch.Tensor,
    full_rope_sin: torch.Tensor,
    hadamard: torch.Tensor,
    format_nd: Callable[[torch.Tensor], torch.Tensor] = _default_format_nd,
) -> PreparedDecodeCSAWeights:
    """Pack all layer-static tensors exactly once, outside ACLGraph capture.

    This ABI matches Eco-Tech/DeepSeek-V4-Flash-0731-w8a8: wq_a, wq_b, wkv
    and indexer.wq_b are symmetric dynamic-W8 linears; wo_a, wo_b and both
    compressors remain BF16. Unsupported layouts are rejected rather than
    silently dequantized or requantized into a numerically different op.
    """
    _validate_layer_topology(layer)
    indexer = layer.indexer
    inner = indexer.compressor
    compressor = layer.compressor

    linear_modules = {
        "wq_a": layer.wq_a,
        "wq_b": layer.wq_b,
        "wkv": layer.wkv,
        "cmp_wkv": compressor.wkv,
        "cmp_wgate": compressor.wgate,
        "idx_wq_b": indexer.wq_b,
        "weights_proj": indexer.weights_proj,
        "inner_wkv": inner.wkv,
        "inner_wgate": inner.wgate,
        "wo_a": layer.wo_a,
        "wo_b": layer.wo_b,
    }
    for name, module in linear_modules.items():
        _require_no_bias(module, name=name)
    for name, module in (
        ("q_norm", layer.q_norm),
        ("kv_norm", layer.kv_norm),
        ("compressor.norm", compressor.norm),
        ("indexer.compressor.norm", inner.norm),
    ):
        _require_no_norm_bias(module, name=name)
    for name, module in (
        ("wq_a", layer.wq_a),
        ("wq_b", layer.wq_b),
        ("wkv", layer.wkv),
        ("indexer.wq_b", indexer.wq_b),
    ):
        _require_w8_dynamic_symmetric(module, name=name)

    d = FLASH.hidden_size
    h = FLASH.num_attention_heads
    hd = FLASH.head_dim
    ql = FLASH.q_lora_rank
    idx_h = FLASH.index_n_heads
    idx_d = FLASH.index_head_dim
    groups = FLASH.o_groups
    ol = FLASH.o_lora_rank
    group_in = h * hd // groups

    arguments: dict[str, torch.Tensor] = {
        "wq_a": _pack_matrix(
            _weight(layer.wq_a, name="wq_a"),
            name="wq_a",
            shape=(d, ql),
            dtype=torch.int8,
            transpose_if_reversed=True,
            format_nd=format_nd,
        ),
        "wq_a_scale": _scale(layer.wq_a, name="wq_a", elements=ql),
        "wq_b": _pack_matrix(
            _weight(layer.wq_b, name="wq_b"),
            name="wq_b",
            shape=(ql, h * hd),
            dtype=torch.int8,
            transpose_if_reversed=True,
            format_nd=format_nd,
        ),
        "wq_b_scale": _scale(layer.wq_b, name="wq_b", elements=h * hd),
        "wkv": _pack_matrix(
            _weight(layer.wkv, name="wkv"),
            name="wkv",
            shape=(d, hd),
            dtype=torch.int8,
            transpose_if_reversed=True,
            format_nd=format_nd,
        ),
        "wkv_scale": _scale(layer.wkv, name="wkv", elements=hd),
        "gamma_cq": _pack_exact(
            _weight(layer.q_norm, name="q_norm"),
            name="gamma_cq",
            shape=(ql,),
            dtype=torch.bfloat16,
            format_nd=format_nd,
        ),
        "gamma_ckv": _pack_exact(
            _weight(layer.kv_norm, name="kv_norm"),
            name="gamma_ckv",
            shape=(hd,),
            dtype=torch.bfloat16,
            format_nd=format_nd,
        ),
        "freqs_cos": _pack_rope(full_rope_cos, name="full_rope_cos"),
        "freqs_sin": _pack_rope(full_rope_sin, name="full_rope_sin"),
        "cmp_wkv": _pack_matrix(
            _weight(compressor.wkv, name="compressor.wkv"),
            name="cmp_wkv",
            shape=(2 * hd, d),
            dtype=torch.bfloat16,
            transpose_if_reversed=False,
            format_nd=format_nd,
        ),
        "cmp_wgate": _pack_matrix(
            _weight(compressor.wgate, name="compressor.wgate"),
            name="cmp_wgate",
            shape=(2 * hd, d),
            dtype=torch.bfloat16,
            transpose_if_reversed=False,
            format_nd=format_nd,
        ),
        "cmp_ape": _pack_exact(
            _require_tensor(compressor.ape, name="compressor.ape"),
            name="cmp_ape",
            shape=(4, 2 * hd),
            dtype=torch.float32,
            format_nd=format_nd,
        ),
        "cmp_norm_w": _pack_exact(
            _weight(compressor.norm, name="compressor.norm"),
            name="cmp_norm_w",
            shape=(hd,),
            dtype=torch.bfloat16,
            format_nd=format_nd,
        ),
        "idx_wq_b": _pack_matrix(
            _weight(indexer.wq_b, name="indexer.wq_b"),
            name="idx_wq_b",
            shape=(ql, idx_h * idx_d),
            dtype=torch.int8,
            transpose_if_reversed=True,
            format_nd=format_nd,
        ),
        "idx_wq_b_scale": _scale(indexer.wq_b, name="indexer.wq_b", elements=idx_h * idx_d),
        "weights_proj": _pack_matrix(
            _weight(indexer.weights_proj, name="indexer.weights_proj"),
            name="weights_proj",
            shape=(d, idx_h),
            dtype=torch.bfloat16,
            transpose_if_reversed=True,
            format_nd=format_nd,
        ),
        "inner_wkv": _pack_matrix(
            _weight(inner.wkv, name="indexer.compressor.wkv"),
            name="inner_wkv",
            shape=(2 * idx_d, d),
            dtype=torch.bfloat16,
            transpose_if_reversed=False,
            format_nd=format_nd,
        ),
        "inner_wgate": _pack_matrix(
            _weight(inner.wgate, name="indexer.compressor.wgate"),
            name="inner_wgate",
            shape=(2 * idx_d, d),
            dtype=torch.bfloat16,
            transpose_if_reversed=False,
            format_nd=format_nd,
        ),
        "inner_ape": _pack_exact(
            _require_tensor(inner.ape, name="indexer.compressor.ape"),
            name="inner_ape",
            shape=(4, 2 * idx_d),
            dtype=torch.float32,
            format_nd=format_nd,
        ),
        "inner_norm_w": _pack_exact(
            _weight(inner.norm, name="indexer.compressor.norm"),
            name="inner_norm_w",
            shape=(idx_d,),
            dtype=torch.bfloat16,
            format_nd=format_nd,
        ),
        "attn_sink": _pack_exact(
            _require_tensor(layer.attn_sink, name="attn_sink"),
            name="attn_sink",
            shape=(h,),
            dtype=torch.float32,
            format_nd=format_nd,
        ),
    }

    hadamard_source = _require_tensor(hadamard, name="hadamard")
    if tuple(hadamard_source.shape) != (idx_d, idx_d):
        raise DecodeCSAWeightContractError(
            f"hadamard: expected shape={(idx_d, idx_d)}, got {tuple(hadamard_source.shape)}"
        )
    # Preserve the native rotate_activation boundary: BF16 F.linear with the
    # unnormalised +/-1 matrix, followed by a separate 1/sqrt(D) multiply.
    # Folding the scale into BF16 weights changes the cube accumulation and
    # can move npu_dynamic_quant output by one INT8 bin.
    arguments["hadamard_idx"] = _pack_exact(
        hadamard_source,
        name="hadamard",
        shape=(idx_d, idx_d),
        dtype=torch.bfloat16,
        format_nd=format_nd,
    )

    wo_a_source = _weight(layer.wo_a, name="wo_a")
    if wo_a_source.dtype != torch.bfloat16:
        raise DecodeCSAWeightContractError(f"wo_a: expected dtype=torch.bfloat16, got {wo_a_source.dtype}")
    wo_a_nd = format_nd(wo_a_source)
    if tuple(wo_a_nd.shape) == (groups, group_in, ol):
        wo_a_nd = wo_a_nd.transpose(1, 2)
    elif tuple(wo_a_nd.shape) == (groups * ol, group_in):
        wo_a_nd = wo_a_nd.reshape(groups, ol, group_in)
    elif tuple(wo_a_nd.shape) != (groups, ol, group_in):
        raise DecodeCSAWeightContractError(
            "wo_a: expected loaded [groups,group_in,o_lora] or kernel "
            f"[groups,o_lora,group_in], got {tuple(wo_a_nd.shape)}"
        )
    arguments["wo_a"] = wo_a_nd.detach().contiguous()
    wo_b_nd = _pack_matrix(
        _weight(layer.wo_b, name="wo_b"),
        name="wo_b",
        shape=(d, groups * ol),
        dtype=torch.bfloat16,
        transpose_if_reversed=False,
        format_nd=format_nd,
    )
    arguments["wo_b"] = (
        format_nd(
            _pack_proj_b_tile_major(
                wo_b_nd,
                groups=groups,
                o_lora=ol,
                n_tile=256,
                k_tile=256,
            )
        )
        .detach()
        .contiguous()
    )

    devices = {tensor.device for tensor in arguments.values()}
    if len(devices) != 1:
        raise DecodeCSAWeightContractError(f"all packed weights must share one device, got {devices}")
    return PreparedDecodeCSAWeights(
        launch_arguments=MappingProxyType(arguments),
        source_owner=layer,
    )


__all__ = [
    "DecodeCSAWeightContractError",
    "PreparedDecodeCSAWeights",
    "pack_decode_csa_weights",
]
