# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Zero-copy adapter from vLLM DSA metadata to the PyPTO decode ABI.

The vLLM custom op already receives all device-resident paging metadata needed
by the ratio-4 decode kernel.  This module deliberately keeps those tensors in
their native representation: it validates and names existing views, but never
allocates, copies, casts, or reads a device value back to the host.  That makes
``bind_decode_csa_native_metadata`` safe for the address-snapshot part of an
ACLGraph capture after the static specialization has been prepared and warmed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import torch

from .config import BLOCK_SIZE, DECODE_SEQ
from .contract import DecodeCSAPhysicalLayout, DecodeCSAProgramSpec


class DecodeCSANativeMetadataError(ValueError):
    """Raised when production DSA metadata cannot represent the static ABI."""


@dataclass(frozen=True, slots=True)
class PreparedDecodeCSANativeMetadata:
    """One validated, allocation-free view of the five DSA metadata families."""

    spec: DecodeCSAProgramSpec
    source_metadata: tuple[object, ...]
    launch_arguments: Mapping[str, torch.Tensor]
    full_rope_cos: torch.Tensor
    full_rope_sin: torch.Tensor
    hadamard: torch.Tensor

    def for_launch(self) -> Mapping[str, torch.Tensor]:
        return self.launch_arguments


_METADATA_FAMILY_NAMES = (
    "compressor_attention",
    "main_compressor_state",
    "inner_compressor_state",
    "indexer_cache",
    "swa_cache",
)
_EXPECTED_BLOCK_SIZES = (BLOCK_SIZE, 2, 2, BLOCK_SIZE, BLOCK_SIZE)


def _require_tensor(value: object, *, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise DecodeCSANativeMetadataError(f"{name}: expected torch.Tensor, got {type(value).__name__}")
    return value


def _canonical_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    strides = [1] * len(shape)
    running = 1
    for axis in range(len(shape) - 1, -1, -1):
        strides[axis] = running
        running *= shape[axis]
    return tuple(strides)


def _require_tensor_layout(
    value: object,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    tensor = _require_tensor(value, name=name)
    actual_shape = tuple(int(extent) for extent in tensor.shape)
    if actual_shape != shape:
        raise DecodeCSANativeMetadataError(f"{name}: expected shape={shape}, got {actual_shape}")
    if tensor.dtype != dtype:
        raise DecodeCSANativeMetadataError(f"{name}: expected dtype={dtype}, got {tensor.dtype}")
    if tensor.device != device:
        raise DecodeCSANativeMetadataError(f"{name}: expected device={device}, got {tensor.device}")
    expected_strides = _canonical_strides(shape)
    actual_strides = tuple(int(stride) for stride in tensor.stride())
    if actual_strides != expected_strides:
        raise DecodeCSANativeMetadataError(
            f"{name}: expected canonical strides={expected_strides}, got {actual_strides}"
        )
    return tensor


def _require_req_metadata(metadata: object, *, family: str) -> object:
    request = getattr(metadata, "req_metadata", None)
    if request is None:
        raise DecodeCSANativeMetadataError(f"{family}: req_metadata is required")
    return request


def _require_int(value: object, *, name: str, expected: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value != expected:
        raise DecodeCSANativeMetadataError(f"{name}: expected {expected}, got {value!r}")


def _require_actual_requests(value: object, *, name: str, bucket: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DecodeCSANativeMetadataError(
            f"{name}: expected an integer A satisfying 1 <= A <= {bucket}, got {value!r}"
        )
    if not 1 <= value <= bucket:
        raise DecodeCSANativeMetadataError(f"{name}: expected 1 <= A <= {bucket}, got {value}")
    return value


def _physical_layout_from_caches(cache_tuple: Sequence[object]) -> DecodeCSAPhysicalLayout:
    if len(cache_tuple) != 6:
        raise DecodeCSANativeMetadataError(f"ratio-4 A2/A3 CSA requires six caches, got {len(cache_tuple)}")
    names = (
        "compressed_kv",
        "swa_kv",
        "main_compressor",
        "inner_compressor",
        "indexer_k",
        "indexer_scale",
    )
    tensors = tuple(_require_tensor(value, name=name) for name, value in zip(names, cache_tuple, strict=True))
    main, inner, indexer_k, indexer_scale = tensors[2:]
    if main.ndim != 4 or inner.ndim != 4 or indexer_k.ndim != 4 or indexer_scale.ndim != 4:
        raise DecodeCSANativeMetadataError("all four page-strided A3 caches must be rank 4")
    return DecodeCSAPhysicalLayout(
        main_state_strides=(int(main.stride(0)), int(main.stride(1)), int(main.stride(3))),
        inner_state_strides=(int(inner.stride(0)), int(inner.stride(1)), int(inner.stride(3))),
        indexer_k_strides=tuple(int(stride) for stride in indexer_k.stride()),
        indexer_scale_strides=tuple(int(stride) for stride in indexer_scale.stride()),
    )


def derive_decode_csa_program_spec(
    cache_tuple: Sequence[object],
    attn_metadata: Sequence[object],
) -> DecodeCSAProgramSpec:
    """Derive the static specialization from production cache/table storage."""
    if not isinstance(attn_metadata, (tuple, list)) or len(attn_metadata) != 5:
        count = len(attn_metadata) if isinstance(attn_metadata, (tuple, list)) else type(attn_metadata).__name__
        raise DecodeCSANativeMetadataError(f"ratio-4 decode requires five sorted metadata families, got {count}")
    requests = tuple(
        _require_req_metadata(metadata, family=family)
        for metadata, family in zip(attn_metadata, _METADATA_FAMILY_NAMES, strict=True)
    )
    tables = tuple(
        _require_tensor(getattr(request, "block_table", None), name=f"{family}.block_table")
        for request, family in zip(requests, _METADATA_FAMILY_NAMES, strict=True)
    )
    if any(table.ndim != 2 for table in tables):
        shapes = tuple(tuple(table.shape) for table in tables)
        raise DecodeCSANativeMetadataError(f"all five block tables must be rank 2, got {shapes}")
    batch = int(tables[0].shape[0])
    if any(int(table.shape[0]) != batch for table in tables):
        raise DecodeCSANativeMetadataError("all five block tables must use the same exact batch bucket")

    if not isinstance(cache_tuple, (tuple, list)) or len(cache_tuple) != 6:
        count = len(cache_tuple) if isinstance(cache_tuple, (tuple, list)) else type(cache_tuple).__name__
        raise DecodeCSANativeMetadataError(f"ratio-4 A2/A3 CSA requires six caches, got {count}")
    caches = tuple(_require_tensor(value, name=f"cache[{index}]") for index, value in enumerate(cache_tuple))
    if any(cache.ndim < 1 for cache in caches):
        raise DecodeCSANativeMetadataError("all six caches must expose a physical-block dimension")
    if int(caches[4].shape[0]) != int(caches[5].shape[0]):
        raise DecodeCSANativeMetadataError("indexer K and scale caches must have the same physical block count")

    return DecodeCSAProgramSpec(
        batch=batch,
        seq=DECODE_SEQ,
        compressed_blocks=int(caches[0].shape[0]),
        swa_blocks=int(caches[1].shape[0]),
        main_state_blocks=int(caches[2].shape[0]),
        inner_state_blocks=int(caches[3].shape[0]),
        indexer_blocks=int(caches[4].shape[0]),
        compressed_table_width=int(tables[0].shape[1]),
        main_state_table_width=int(tables[1].shape[1]),
        inner_state_table_width=int(tables[2].shape[1]),
        indexer_table_width=int(tables[3].shape[1]),
        swa_table_width=int(tables[4].shape[1]),
        physical_layout=_physical_layout_from_caches(cache_tuple),
    )


def bind_decode_csa_native_metadata(
    attn_metadata: Sequence[object],
    spec: DecodeCSAProgramSpec,
    *,
    device: torch.device,
) -> PreparedDecodeCSANativeMetadata:
    """Validate an exact pure-decode S=8 batch and name its existing tensors.

    No device value is inspected.  In particular, ``query_start_loc`` contents
    are owned by the vLLM builder; this adapter validates its capture-stable
    shape but does not call ``item()`` or ``cpu()`` in the launch path.
    """
    if not isinstance(attn_metadata, (tuple, list)) or len(attn_metadata) != 5:
        count = len(attn_metadata) if isinstance(attn_metadata, (tuple, list)) else type(attn_metadata).__name__
        raise DecodeCSANativeMetadataError(f"ratio-4 decode requires five sorted metadata families, got {count}")
    metadata_tuple = tuple(attn_metadata)
    requests = tuple(
        _require_req_metadata(metadata, family=family)
        for metadata, family in zip(metadata_tuple, _METADATA_FAMILY_NAMES, strict=True)
    )
    actual_requests = _require_actual_requests(
        getattr(requests[0], "num_reqs_actual", None),
        name=f"{_METADATA_FAMILY_NAMES[0]}.num_reqs_actual",
        bucket=spec.batch,
    )
    actual_tokens = actual_requests * spec.seq

    for metadata, request, family, block_size in zip(
        metadata_tuple,
        requests,
        _METADATA_FAMILY_NAMES,
        _EXPECTED_BLOCK_SIZES,
        strict=True,
    ):
        _require_int(
            getattr(metadata, "num_actual_tokens", None),
            name=f"{family}.num_actual_tokens",
            expected=actual_tokens,
        )
        _require_int(getattr(metadata, "num_decodes", None), name=f"{family}.num_decodes", expected=spec.batch)
        _require_int(
            getattr(metadata, "num_decode_tokens", None),
            name=f"{family}.num_decode_tokens",
            expected=actual_tokens,
        )
        _require_int(getattr(metadata, "num_prefills", None), name=f"{family}.num_prefills", expected=0)
        _require_int(
            getattr(request, "num_reqs_actual", None),
            name=f"{family}.num_reqs_actual",
            expected=actual_requests,
        )
        _require_int(getattr(request, "block_size", None), name=f"{family}.block_size", expected=block_size)
        _require_tensor_layout(
            getattr(request, "query_start_loc", None),
            name=f"{family}.query_start_loc",
            shape=(spec.batch + 1,),
            dtype=torch.int32,
            device=device,
        )

    table_contracts = (
        ("cmp_block_table", requests[0], spec.compressed_table_width),
        ("compress_state_block_table", requests[1], spec.main_state_table_width),
        ("inner_compress_state_block_table", requests[2], spec.inner_state_table_width),
        ("idx_block_table", requests[3], spec.indexer_table_width),
        ("swa_block_table", requests[4], spec.swa_table_width),
    )
    launch: dict[str, torch.Tensor] = {}
    for argument_name, request, width in table_contracts:
        launch[argument_name] = _require_tensor_layout(
            getattr(request, "block_table", None),
            name=argument_name,
            shape=(spec.batch, width),
            dtype=torch.int32,
            device=device,
        )

    launch["start_positions"] = _require_tensor_layout(
        getattr(requests[0], "start_pos", None),
        name="start_positions",
        shape=(spec.batch,),
        dtype=torch.int32,
        device=device,
    )
    launch["kv_seq_lens"] = _require_tensor_layout(
        getattr(requests[0], "seq_lens", None),
        name="kv_seq_lens",
        shape=(spec.batch,),
        dtype=torch.int32,
        device=device,
    )

    full_rope_cos = _require_tensor(getattr(requests[0], "full_compress_cos", None), name="full_compress_cos")
    full_rope_sin = _require_tensor(getattr(requests[0], "full_compress_sin", None), name="full_compress_sin")
    hadamard = _require_tensor(getattr(metadata_tuple[3], "hadamard", None), name="indexer_cache.hadamard")
    for name, tensor in (
        ("full_compress_cos", full_rope_cos),
        ("full_compress_sin", full_rope_sin),
        ("indexer_cache.hadamard", hadamard),
    ):
        if tensor.device != device:
            raise DecodeCSANativeMetadataError(f"{name}: expected device={device}, got {tensor.device}")

    return PreparedDecodeCSANativeMetadata(
        spec=spec,
        source_metadata=metadata_tuple,
        launch_arguments=MappingProxyType(launch),
        full_rope_cos=full_rope_cos,
        full_rope_sin=full_rope_sin,
        hadamard=hadamard,
    )


def validate_decode_csa_uniform_query_rows(
    attn_metadata: Sequence[object],
    spec: DecodeCSAProgramSpec,
) -> None:
    """Cold-path proof that every request owns exactly the static S=8 rows.

    The 44-slot production ABI reconstructs token-to-request mapping from the
    fixed S=8 contract and intentionally does not pass query_start_loc.  This
    function may synchronize while copying metadata to CPU and therefore must
    only be called during explicit installation, never from custom-op launch.
    The caller must additionally guarantee that the same builder buffers retain
    this uniform-query invariant for subsequent steps.
    """
    if not isinstance(attn_metadata, (tuple, list)) or len(attn_metadata) != 5:
        raise DecodeCSANativeMetadataError("uniform query validation requires five metadata families")
    expected = tuple(range(0, (spec.batch + 1) * spec.seq, spec.seq))
    for metadata, family in zip(attn_metadata, _METADATA_FAMILY_NAMES, strict=True):
        request = _require_req_metadata(metadata, family=family)
        query_start_loc = _require_tensor(
            getattr(request, "query_start_loc", None),
            name=f"{family}.query_start_loc",
        )
        actual = tuple(int(value) for value in query_start_loc.detach().to(device="cpu").tolist())
        if actual != expected:
            raise DecodeCSANativeMetadataError(
                f"{family}.query_start_loc must prove uniform S={spec.seq} rows: expected={expected}, got={actual}"
            )


__all__ = [
    "DecodeCSANativeMetadataError",
    "PreparedDecodeCSANativeMetadata",
    "bind_decode_csa_native_metadata",
    "derive_decode_csa_program_spec",
    "validate_decode_csa_uniform_query_rows",
]
