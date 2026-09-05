# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.ops._pypto_dsv4_csa.native_metadata import (
    DecodeCSANativeMetadataError,
    bind_decode_csa_native_metadata,
    derive_decode_csa_program_spec,
    validate_decode_csa_uniform_query_rows,
)


def _strided(
    shape: tuple[int, ...],
    strides: tuple[int, ...],
    dtype: torch.dtype,
) -> torch.Tensor:
    span = 1 + sum((extent - 1) * stride for extent, stride in zip(shape, strides, strict=True))
    return torch.as_strided(torch.zeros(span, dtype=dtype), shape, strides)


def _caches() -> tuple[torch.Tensor, ...]:
    return (
        torch.zeros((8, 32, 1, 512), dtype=torch.bfloat16),
        torch.zeros((12, 32, 1, 512), dtype=torch.bfloat16),
        _strided((10, 2, 1, 2048), (8192, 2048, 2048, 1), torch.float32),
        _strided((11, 2, 1, 512), (1040, 512, 512, 1), torch.float32),
        _strided((9, 32, 1, 128), (4160, 128, 128, 1), torch.int8),
        _strided((9, 32, 1, 1), (2080, 1, 1, 1), torch.float16),
    )


def _metadata(
    *,
    actual: int = 4,
    query_start: torch.Tensor | None = None,
) -> list[object]:
    batch = 4
    bucket_tokens = batch * 8
    actual_tokens = actual * 8
    query = torch.arange(0, bucket_tokens + 1, 8, dtype=torch.int32) if query_start is None else query_start
    widths = (7, 13, 15, 9, 17)
    block_sizes = (32, 2, 2, 32, 32)
    result = []
    for index, (width, block_size) in enumerate(zip(widths, block_sizes, strict=True)):
        request = SimpleNamespace(
            block_table=torch.zeros((batch, width), dtype=torch.int32),
            seq_lens=torch.tensor([16] * actual + [0] * (batch - actual), dtype=torch.int32),
            slot_mapping=torch.zeros((actual_tokens, 2), dtype=torch.int32) if index == 4 else None,
            block_size=block_size,
            query_start_loc=query,
            start_pos=torch.arange(batch, dtype=torch.int32) if index == 0 else None,
            num_reqs_actual=actual,
            full_compress_cos=torch.ones((64, 1, 1, 64)),
            full_compress_sin=torch.zeros((64, 1, 1, 64)),
        )
        result.append(
            SimpleNamespace(
                num_actual_tokens=actual_tokens,
                num_decodes=batch,
                num_decode_tokens=actual_tokens,
                num_prefills=0,
                req_metadata=request,
                hadamard=torch.eye(128) if index == 3 else None,
            )
        )
    return result


def test_native_metadata_derives_real_capacity_width_and_stride_family() -> None:
    caches = _caches()
    metadata = _metadata()

    spec = derive_decode_csa_program_spec(caches, metadata)
    prepared = bind_decode_csa_native_metadata(metadata, spec, device=torch.device("cpu"))
    validate_decode_csa_uniform_query_rows(metadata, spec)

    assert spec.batch == 4
    assert spec.compressed_blocks == 8
    assert spec.swa_blocks == 12
    assert spec.main_state_blocks == 10
    assert spec.inner_state_blocks == 11
    assert spec.indexer_blocks == 9
    assert (
        spec.compressed_table_width,
        spec.main_state_table_width,
        spec.inner_state_table_width,
        spec.indexer_table_width,
        spec.swa_table_width,
    ) == (7, 13, 15, 9, 17)
    assert spec.physical_layout.main_state_strides == (8192, 2048, 1)
    assert spec.physical_layout.inner_state_strides == (1040, 512, 1)
    assert prepared.for_launch()["cmp_block_table"] is metadata[0].req_metadata.block_table
    assert prepared.for_launch()["compress_state_block_table"] is metadata[1].req_metadata.block_table
    assert prepared.for_launch()["inner_compress_state_block_table"] is metadata[2].req_metadata.block_table
    assert prepared.for_launch()["idx_block_table"] is metadata[3].req_metadata.block_table
    assert prepared.for_launch()["swa_block_table"] is metadata[4].req_metadata.block_table
    assert "swa_slot_mapping" not in prepared.for_launch()


def test_native_metadata_accepts_partial_bucket_and_ignores_swa_slot_mapping() -> None:
    metadata = _metadata(actual=3)
    spec = derive_decode_csa_program_spec(_caches(), metadata)
    ignored_slot_mapping = object()
    metadata[4].req_metadata.slot_mapping = ignored_slot_mapping

    prepared = bind_decode_csa_native_metadata(metadata, spec, device=torch.device("cpu"))
    validate_decode_csa_uniform_query_rows(metadata, spec)

    assert spec.batch == 4
    assert all(item.req_metadata.num_reqs_actual == 3 for item in metadata)
    assert all(item.num_actual_tokens == 24 for item in metadata)
    assert all(item.num_decode_tokens == 24 for item in metadata)
    assert all(item.num_decodes == 4 for item in metadata)
    assert "swa_slot_mapping" not in prepared.for_launch()
    assert ignored_slot_mapping is metadata[4].req_metadata.slot_mapping


def test_uniform_query_rows_need_a_cold_value_proof() -> None:
    # Shape and total-token counts alone do not prove every request owns S=8.
    metadata = _metadata(query_start=torch.tensor([0, 7, 16, 23, 32], dtype=torch.int32))
    spec = derive_decode_csa_program_spec(_caches(), metadata)

    bind_decode_csa_native_metadata(metadata, spec, device=torch.device("cpu"))
    with pytest.raises(DecodeCSANativeMetadataError, match="uniform S=8"):
        validate_decode_csa_uniform_query_rows(metadata, spec)


def test_native_metadata_rejects_inconsistent_actual_request_count() -> None:
    metadata = _metadata(actual=3)
    spec = derive_decode_csa_program_spec(_caches(), metadata)
    metadata[2].req_metadata.num_reqs_actual = 2
    with pytest.raises(DecodeCSANativeMetadataError, match=r"inner_compressor_state\.num_reqs_actual: expected 3"):
        bind_decode_csa_native_metadata(metadata, spec, device=torch.device("cpu"))


@pytest.mark.parametrize("actual", (0, 5, True, None))
def test_native_metadata_rejects_invalid_actual_request_count(actual: object) -> None:
    metadata = _metadata()
    spec = derive_decode_csa_program_spec(_caches(), metadata)
    for item in metadata:
        item.req_metadata.num_reqs_actual = actual
    with pytest.raises(DecodeCSANativeMetadataError, match=r"1 <= A <= 4"):
        bind_decode_csa_native_metadata(metadata, spec, device=torch.device("cpu"))


@pytest.mark.parametrize(
    ("field", "bad_value", "match"),
    (
        ("num_actual_tokens", 32, r"num_actual_tokens: expected 24"),
        ("num_decode_tokens", 32, r"num_decode_tokens: expected 24"),
        ("num_decodes", 3, r"num_decodes: expected 4"),
        ("num_prefills", 1, r"num_prefills: expected 0"),
    ),
)
def test_native_metadata_rejects_partial_bucket_counter_mismatch(
    field: str,
    bad_value: int,
    match: str,
) -> None:
    metadata = _metadata(actual=3)
    spec = derive_decode_csa_program_spec(_caches(), metadata)
    setattr(metadata[1], field, bad_value)

    with pytest.raises(DecodeCSANativeMetadataError, match=match):
        bind_decode_csa_native_metadata(metadata, spec, device=torch.device("cpu"))


class _NoDeviceValueReadTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, source: torch.Tensor) -> "_NoDeviceValueReadTensor":
        return torch.Tensor._make_subclass(cls, source, require_grad=False)

    def item(self, *args: object, **kwargs: object) -> object:
        raise AssertionError("hot metadata binding must not call Tensor.item()")

    def cpu(self, *args: object, **kwargs: object) -> object:
        raise AssertionError("hot metadata binding must not call Tensor.cpu()")

    def to(self, *args: object, **kwargs: object) -> object:
        raise AssertionError("hot metadata binding must not call Tensor.to()")

    def tolist(self, *args: object, **kwargs: object) -> object:
        raise AssertionError("hot metadata binding must not call Tensor.tolist()")

    def numpy(self, *args: object, **kwargs: object) -> object:
        raise AssertionError("hot metadata binding must not call Tensor.numpy()")


def test_hot_metadata_binding_never_reads_device_tensor_values() -> None:
    metadata = _metadata(actual=3)
    spec = derive_decode_csa_program_spec(_caches(), metadata)
    for item in metadata:
        request = item.req_metadata
        request.block_table = _NoDeviceValueReadTensor(request.block_table)
        request.query_start_loc = _NoDeviceValueReadTensor(request.query_start_loc)
    first = metadata[0].req_metadata
    first.seq_lens = _NoDeviceValueReadTensor(first.seq_lens)
    first.start_pos = _NoDeviceValueReadTensor(first.start_pos)

    prepared = bind_decode_csa_native_metadata(metadata, spec, device=torch.device("cpu"))

    assert prepared.for_launch()["kv_seq_lens"] is first.seq_lens


def test_native_metadata_rejects_noncanonical_table() -> None:
    metadata = _metadata()
    spec = derive_decode_csa_program_spec(_caches(), metadata)
    metadata[0].req_metadata.block_table = torch.zeros((4, 14), dtype=torch.int32)[:, ::2]
    with pytest.raises(DecodeCSANativeMetadataError, match="canonical strides"):
        bind_decode_csa_native_metadata(metadata, spec, device=torch.device("cpu"))
