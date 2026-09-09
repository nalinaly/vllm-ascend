# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Host checks that the standalone CSA case stays on the production contract."""

from __future__ import annotations

from tests.pypto_dsv4_decode_csa.standalone_csa_perf_case import (
    DEFAULT_START_POSITION,
    SPECIALIZATION,
    actual_workload_spec,
)
from vllm_ascend.ops._pypto_dsv4_csa.config import DECODE_SEQ, FLASH
from vllm_ascend.ops._pypto_dsv4_csa.contract import DecodeCSAProgramSpec


def test_actual_workload_spec_matches_production_contract() -> None:
    spec = DecodeCSAProgramSpec(batch=4)
    shapes = actual_workload_spec(batch=4)
    assert shapes["specialization"] == SPECIALIZATION
    assert shapes["batch"] == 4
    assert shapes["seq"] == DECODE_SEQ == spec.seq == 8
    assert shapes["tokens"] == 32
    assert shapes["public_tensors"]["hidden_states"] == [32, FLASH.hidden_size]
    assert shapes["public_tensors"]["attn_out"] == [32, FLASH.hidden_size]
    assert shapes["cache_blocks"]["swa"] == spec.swa_blocks
    assert shapes["cache_blocks"]["compressed"] == spec.compressed_blocks
    assert shapes["physical_spans"]["main_state"] == spec.main_state_span
    assert shapes["defaults"]["start_position"] == DEFAULT_START_POSITION
    assert shapes["program_key"] == list(spec.key)
