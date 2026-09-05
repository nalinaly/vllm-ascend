# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Pure Host tests for the checkpoint-compatible decode-CSA weight ABI."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from vllm_ascend.ops._pypto_dsv4_csa.backend import (
    EXPECTED_ABI_PARAMETER_COUNT,
    EXPECTED_PARAMETER_NAMES,
    EXPECTED_SCALAR_COUNT,
    EXPECTED_TENSOR_COUNT,
)
from vllm_ascend.ops._pypto_dsv4_csa.config import FLASH
from vllm_ascend.ops._pypto_dsv4_csa.weights import (
    DecodeCSAWeightContractError,
    _pack_proj_b_tile_major,
    pack_decode_csa_weights,
)


class AscendW8A8DynamicLinearMethod:
    """Name-compatible stand-in for the nested vLLM-Ascend quant method."""


class NonSymmetricLinearMethod:
    """A deliberately unsupported checkpoint quant method."""


class AscendDSAImpl:
    """Name-compatible stand-in for the non-CP native DSA implementation."""

    def __init__(self, *, max_model_len: int = FLASH.max_position_embeddings) -> None:
        self.multistream_dsv4_dsa_overlap = False
        self.vllm_config = SimpleNamespace(model_config=SimpleNamespace(max_model_len=max_model_len))


class AscendDSACPImpl(AscendDSAImpl):
    """Name-compatible stand-in for the unsupported DSA-CP implementation."""


def _fake_tensor(mode: FakeTensorMode, shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    with mode:
        tensor = torch.empty(shape, dtype=dtype, device="cpu")
        if dtype.is_floating_point:
            tensor.requires_grad_(True)
        return tensor


def _linear(
    mode: FakeTensorMode,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    *,
    quantized: bool = False,
    scale_elements: int | None = None,
) -> SimpleNamespace:
    module = SimpleNamespace(
        weight=_fake_tensor(mode, shape, dtype),
        bias=None,
    )
    if quantized:
        assert scale_elements is not None
        module.quant_method = SimpleNamespace(quant_method=AscendW8A8DynamicLinearMethod())
        # The live checkpoint commonly carries column scales with a singleton
        # dimension.  Packing must flatten that storage to the kernel ABI.
        module.weight_scale_fp32 = _fake_tensor(mode, (scale_elements, 1), torch.float32)
        module.weight_offset = torch.zeros((scale_elements, 1), dtype=torch.int8)
    return module


def _norm(mode: FakeTensorMode, elements: int) -> SimpleNamespace:
    return SimpleNamespace(
        weight=_fake_tensor(mode, (elements,), torch.bfloat16),
        bias=None,
    )


def _make_layer(
    *,
    max_model_len: int = FLASH.max_position_embeddings,
    impl_type: type[AscendDSAImpl] = AscendDSAImpl,
) -> SimpleNamespace:
    mode = FakeTensorMode()
    d = FLASH.hidden_size
    h = FLASH.num_attention_heads
    hd = FLASH.head_dim
    ql = FLASH.q_lora_rank
    idx_h = FLASH.index_n_heads
    idx_d = FLASH.index_head_dim
    groups = FLASH.o_groups
    ol = FLASH.o_lora_rank
    group_in = h * hd // groups

    inner = SimpleNamespace(
        wkv=_linear(mode, (2 * idx_d, d), torch.bfloat16),
        wgate=_linear(mode, (2 * idx_d, d), torch.bfloat16),
        ape=_fake_tensor(mode, (4, 2 * idx_d), torch.float32),
        norm=_norm(mode, idx_d),
    )
    indexer = SimpleNamespace(
        compressor=inner,
        wq_b=_linear(
            mode,
            (idx_h * idx_d, ql),
            torch.int8,
            quantized=True,
            scale_elements=idx_h * idx_d,
        ),
        weights_proj=_linear(mode, (idx_h, d), torch.bfloat16),
    )
    compressor = SimpleNamespace(
        wkv=_linear(mode, (2 * hd, d), torch.bfloat16),
        wgate=_linear(mode, (2 * hd, d), torch.bfloat16),
        ape=_fake_tensor(mode, (4, 2 * hd), torch.float32),
        norm=_norm(mode, hd),
    )
    return SimpleNamespace(
        dim=d,
        n_heads=h,
        n_local_heads=h,
        q_lora_rank=ql,
        o_lora_rank=ol,
        head_dim=hd,
        rope_head_dim=FLASH.qk_rope_head_dim,
        n_groups=groups,
        n_local_groups=groups,
        window_size=FLASH.sliding_window,
        compress_ratio=4,
        skip_topk=False,
        dsa_attn=SimpleNamespace(impl=impl_type(max_model_len=max_model_len)),
        indexer=indexer,
        compressor=compressor,
        wq_a=_linear(mode, (ql, d), torch.int8, quantized=True, scale_elements=ql),
        wq_b=_linear(mode, (h * hd, ql), torch.int8, quantized=True, scale_elements=h * hd),
        wkv=_linear(mode, (hd, d), torch.int8, quantized=True, scale_elements=hd),
        q_norm=_norm(mode, ql),
        kv_norm=_norm(mode, hd),
        attn_sink=_fake_tensor(mode, (h,), torch.float32),
        # This is the real loaded rank-3 layout.  The kernel consumes the last
        # two axes transposed as [groups, o_lora, group_input].
        wo_a=_linear(mode, (groups, group_in, ol), torch.bfloat16),
        wo_b=_linear(mode, (d, groups * ol), torch.bfloat16),
        _fake_mode=mode,
    )


@pytest.fixture(scope="module")
def rope_and_hadamard() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rows = FLASH.max_position_embeddings + 3
    row_values = torch.arange(rows, dtype=torch.float32).reshape(rows, 1)
    rope_cos = row_values.expand(rows, FLASH.qk_rope_head_dim)
    rope_sin = (row_values + 0.5).expand(rows, FLASH.qk_rope_head_dim)
    hadamard = torch.ones((FLASH.index_head_dim, FLASH.index_head_dim), dtype=torch.bfloat16)
    hadamard[1::2] = -1.0
    return rope_cos, rope_sin, hadamard


def _pack(
    layer: object,
    rope_and_hadamard: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> Any:
    rope_cos, rope_sin, hadamard = rope_and_hadamard
    return pack_decode_csa_weights(
        layer,
        full_rope_cos=rope_cos,
        full_rope_sin=rope_sin,
        hadamard=hadamard,
        format_nd=lambda tensor: tensor,
    )


def _resolve(root: object, path: str) -> object:
    value = root
    for component in path.split("."):
        value = getattr(value, component)
    return value


def test_proj_b_tile_major_pack_preserves_every_logical_weight() -> None:
    groups = 2
    d = 8
    o_lora = 8
    n_tile = 4
    k_tile = 2
    source = torch.arange(d * groups * o_lora, dtype=torch.float32).reshape(d, groups * o_lora)

    packed = _pack_proj_b_tile_major(
        source,
        groups=groups,
        o_lora=o_lora,
        n_tile=n_tile,
        k_tile=k_tile,
    )
    tile_rows = packed.reshape(-1, k_tile)

    assert tuple(packed.shape) == tuple(source.shape)
    assert packed.is_contiguous()
    for group in range(groups):
        for n0 in range(0, d, n_tile):
            for k0 in range(0, o_lora, k_tile):
                row0 = ((group * (d // n_tile) + n0 // n_tile) * (o_lora // k_tile) + k0 // k_tile) * n_tile
                actual = tile_rows[row0 : row0 + n_tile, :]
                expected = source[n0 : n0 + n_tile, group * o_lora + k0 : group * o_lora + k0 + k_tile]
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_pack_real_44_abi_keys_dtypes_shapes_and_static_transforms(
    rope_and_hadamard: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    layer = _make_layer()
    prepared = _pack(layer, rope_and_hadamard)
    arguments = prepared.for_launch()

    d = FLASH.hidden_size
    h = FLASH.num_attention_heads
    hd = FLASH.head_dim
    ql = FLASH.q_lora_rank
    idx_h = FLASH.index_n_heads
    idx_d = FLASH.index_head_dim
    groups = FLASH.o_groups
    ol = FLASH.o_lora_rank
    group_in = h * hd // groups
    expected = {
        "wq_a": ((d, ql), torch.int8),
        "wq_a_scale": ((ql,), torch.float32),
        "wq_b": ((ql, h * hd), torch.int8),
        "wq_b_scale": ((h * hd,), torch.float32),
        "wkv": ((d, hd), torch.int8),
        "wkv_scale": ((hd,), torch.float32),
        "gamma_cq": ((ql,), torch.bfloat16),
        "gamma_ckv": ((hd,), torch.bfloat16),
        "freqs_cos": ((FLASH.max_position_embeddings, FLASH.qk_rope_head_dim), torch.float32),
        "freqs_sin": ((FLASH.max_position_embeddings, FLASH.qk_rope_head_dim), torch.float32),
        "cmp_wkv": ((2 * hd, d), torch.bfloat16),
        "cmp_wgate": ((2 * hd, d), torch.bfloat16),
        "cmp_ape": ((4, 2 * hd), torch.float32),
        "cmp_norm_w": ((hd,), torch.bfloat16),
        "idx_wq_b": ((ql, idx_h * idx_d), torch.int8),
        "idx_wq_b_scale": ((idx_h * idx_d,), torch.float32),
        "weights_proj": ((d, idx_h), torch.bfloat16),
        "inner_wkv": ((2 * idx_d, d), torch.bfloat16),
        "inner_wgate": ((2 * idx_d, d), torch.bfloat16),
        "inner_ape": ((4, 2 * idx_d), torch.float32),
        "inner_norm_w": ((idx_d,), torch.bfloat16),
        "attn_sink": ((h,), torch.float32),
        "hadamard_idx": ((idx_d, idx_d), torch.bfloat16),
        "wo_a": ((groups, ol, group_in), torch.bfloat16),
        "wo_b": ((d, groups * ol), torch.bfloat16),
    }

    assert EXPECTED_ABI_PARAMETER_COUNT == 44
    assert EXPECTED_TENSOR_COUNT == 40
    assert EXPECTED_SCALAR_COUNT == 4
    assert tuple(arguments) == tuple(expected)
    assert set(arguments).issubset(EXPECTED_PARAMETER_NAMES[:EXPECTED_TENSOR_COUNT])
    assert prepared.source_owner is layer
    for name, (shape, dtype) in expected.items():
        assert tuple(arguments[name].shape) == shape, name
        assert arguments[name].dtype == dtype, name
        assert arguments[name].device.type == "cpu", name
        assert arguments[name].is_contiguous(), name
        assert not arguments[name].requires_grad, name

    # Rank-3 wo_a could only reach this canonical shape through the required
    # transpose(1, 2), and all four dynamic-W8 scales have been flattened.
    assert tuple(layer.wo_a.weight.shape) == (groups, group_in, ol)
    assert tuple(arguments["wo_a"].shape) == (groups, ol, group_in)

    rope_cos, rope_sin, hadamard = rope_and_hadamard
    del rope_cos, rope_sin
    assert arguments["freqs_cos"].shape[0] == FLASH.max_position_embeddings
    assert arguments["freqs_cos"][-1, 0] == torch.tensor(
        FLASH.max_position_embeddings - 1,
        dtype=torch.float32,
    )
    assert arguments["freqs_sin"][-1, 0] == torch.tensor(
        FLASH.max_position_embeddings - 0.5,
        dtype=torch.float32,
    )
    torch.testing.assert_close(arguments["hadamard_idx"], hadamard, rtol=0, atol=0)

    with pytest.raises(TypeError):
        arguments["unexpected"] = torch.empty(())  # type: ignore[index]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("n_local_heads", FLASH.num_attention_heads // 4),
        ("n_local_groups", FLASH.o_groups // 4),
    ],
)
def test_pack_rejects_tensor_parallel_local_topology(
    field: str,
    value: int,
    rope_and_hadamard: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    layer = _make_layer()
    setattr(layer, field, value)

    with pytest.raises(DecodeCSAWeightContractError, match="TP1/local-H64/local-G8"):
        _pack(layer, rope_and_hadamard)


def test_pack_rejects_dsa_context_parallel(
    rope_and_hadamard: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    layer = _make_layer(impl_type=AscendDSACPImpl)

    with pytest.raises(DecodeCSAWeightContractError, match="non-CP AscendDSAImpl"):
        _pack(layer, rope_and_hadamard)


def test_pack_rejects_model_length_above_current_rope_capacity(
    rope_and_hadamard: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    layer = _make_layer(max_model_len=FLASH.max_position_embeddings + 1)

    with pytest.raises(DecodeCSAWeightContractError, match="exceeds the current PyPTO limit"):
        _pack(layer, rope_and_hadamard)


@pytest.mark.parametrize("path", ["wq_a", "wq_b", "wkv", "indexer.wq_b"])
def test_pack_rejects_wrong_dynamic_w8_method(
    path: str,
    rope_and_hadamard: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    layer = _make_layer()
    module = _resolve(layer, path)
    module.quant_method = SimpleNamespace(quant_method=NonSymmetricLinearMethod())

    with pytest.raises(DecodeCSAWeightContractError, match="AscendW8A8DynamicLinearMethod"):
        _pack(layer, rope_and_hadamard)


@pytest.mark.parametrize("path", ["wq_a", "wq_b", "wkv", "indexer.wq_b"])
def test_pack_rejects_nonzero_dynamic_w8_offset(
    path: str,
    rope_and_hadamard: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    layer = _make_layer()
    module = _resolve(layer, path)
    module.weight_offset = torch.ones((1,), dtype=torch.int8)

    with pytest.raises(DecodeCSAWeightContractError, match="zero offset"):
        _pack(layer, rope_and_hadamard)


@pytest.mark.parametrize("path", ["wq_a", "wq_b", "wkv", "indexer.wq_b"])
@pytest.mark.parametrize("broken_field", ["weight", "weight_scale_fp32"])
def test_pack_rejects_wrong_dynamic_w8_tensor_dtype(
    path: str,
    broken_field: str,
    rope_and_hadamard: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    layer = _make_layer()
    module = _resolve(layer, path)
    original = getattr(module, broken_field)
    wrong_dtype = torch.bfloat16 if broken_field == "weight" else torch.float16
    setattr(
        module,
        broken_field,
        _fake_tensor(layer._fake_mode, tuple(original.shape), wrong_dtype),
    )

    with pytest.raises(DecodeCSAWeightContractError, match="expected dtype"):
        _pack(layer, rope_and_hadamard)


@pytest.mark.parametrize(
    "path",
    ["q_norm", "kv_norm", "compressor.norm", "indexer.compressor.norm"],
)
def test_pack_rejects_norm_bias_or_anti_method_m4(
    path: str,
    rope_and_hadamard: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    layer = _make_layer()
    module = _resolve(layer, path)
    module.bias = torch.ones((1,), dtype=torch.bfloat16)

    with pytest.raises(DecodeCSAWeightContractError, match="RMSNorm bias/anti-method m4"):
        _pack(layer, rope_and_hadamard)


def test_pack_rejects_non_bf16_wo_b(
    rope_and_hadamard: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    layer = _make_layer()
    mode = layer._fake_mode
    layer.wo_b.weight = _fake_tensor(
        mode,
        (FLASH.hidden_size, FLASH.o_groups * FLASH.o_lora_rank),
        torch.float32,
    )

    with pytest.raises(DecodeCSAWeightContractError, match=r"wo_b: expected dtype=torch\.bfloat16"):
        _pack(layer, rope_and_hadamard)


def test_pack_rejects_non_fp32_native_rope_cache(
    rope_and_hadamard: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    layer = _make_layer()
    rope_cos, rope_sin, hadamard = rope_and_hadamard

    with pytest.raises(DecodeCSAWeightContractError, match=r"full_rope_cos: expected dtype=torch\.float32"):
        pack_decode_csa_weights(
            layer,
            full_rope_cos=rope_cos.to(torch.bfloat16),
            full_rope_sin=rope_sin,
            hadamard=hadamard,
            format_nd=lambda tensor: tensor,
        )
