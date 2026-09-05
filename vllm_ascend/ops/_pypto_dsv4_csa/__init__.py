# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Private PyPTO implementation of the DeepSeek V4 decode CSA operator.

The package is deliberately not imported by :mod:`vllm_ascend.ops`.  Importing
the kernel pulls in the optional PyPTO compiler, while the contract and adapter
remain usable by CPU-only unit tests.
"""

from .adapter import (
    DecodeCSACacheContractError,
    PreparedDecodeCSACaches,
    prepare_decode_csa_caches,
)
from .contract import DecodeCSAPhysicalLayout, DecodeCSAProgramSpec, MutableCSAState
from .dispatch import (
    DecodeCSADeviceOwnerRegistry,
    DecodeCSADispatchError,
    DecodeCSALayerOwner,
    install_pypto_dsv4_decode_csa,
    mark_pypto_dsv4_decode_csa_warmups_quiesced,
    uninstall_pypto_dsv4_decode_csa,
)
from .native_metadata import (
    DecodeCSANativeMetadataError,
    PreparedDecodeCSANativeMetadata,
    bind_decode_csa_native_metadata,
    derive_decode_csa_program_spec,
    validate_decode_csa_uniform_query_rows,
)
from .weights import (
    DecodeCSAWeightContractError,
    PreparedDecodeCSAWeights,
    pack_decode_csa_weights,
)


def make_decode_csa_l1_program(
    spec: DecodeCSAProgramSpec,
    runtime: str = "tensormap_and_ringbuffer",
):
    """Lazily import PyPTO and construct one static L1 specialization."""
    from .kernel import make_decode_csa_l1_program as make_program

    return make_program(spec, runtime)


__all__ = [
    "DecodeCSAProgramSpec",
    "DecodeCSAPhysicalLayout",
    "DecodeCSACacheContractError",
    "DecodeCSADispatchError",
    "DecodeCSADeviceOwnerRegistry",
    "DecodeCSALayerOwner",
    "DecodeCSANativeMetadataError",
    "DecodeCSAWeightContractError",
    "MutableCSAState",
    "PreparedDecodeCSACaches",
    "PreparedDecodeCSANativeMetadata",
    "PreparedDecodeCSAWeights",
    "bind_decode_csa_native_metadata",
    "derive_decode_csa_program_spec",
    "install_pypto_dsv4_decode_csa",
    "make_decode_csa_l1_program",
    "mark_pypto_dsv4_decode_csa_warmups_quiesced",
    "pack_decode_csa_weights",
    "prepare_decode_csa_caches",
    "uninstall_pypto_dsv4_decode_csa",
    "validate_decode_csa_uniform_query_rows",
]
