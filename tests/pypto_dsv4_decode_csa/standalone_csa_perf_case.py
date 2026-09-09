# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Standalone real-shape decode CSA case: native DSA vs PyPTO L1.

The shapes are taken from the vLLM-Ascend production contract
``DecodeCSAProgramSpec(batch=4)`` (A3, TP1, ratio-4, ``S = DECODE_SEQ = 8``).
This module does not invent a second kernel.  Timing and correctness still go
through ``a3_single_op_benchmark`` so later optimization diffs stay comparable
to the 2026-09-05 reproduction.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tests.pypto_dsv4_decode_csa.a3_single_op_benchmark import (
    HBG_RUNTIME,
    SUPPORTED_BATCH,
    SUPPORTED_SEQ,
    TRB_RUNTIME,
    A3SingleOpBenchmarkConfig,
    NativeProductionUnsupported,
    run_a3_single_op_benchmark,
)
from tests.pypto_dsv4_decode_csa.benchmark import BENCHMARK_SCHEMA_NAME, BENCHMARK_SCHEMA_VERSION, BenchmarkMode
from vllm_ascend.ops._pypto_dsv4_csa.config import (
    BLOCK_SIZE,
    DECODE_SEQ,
    FLASH,
    IDX_CACHE_BLOCK_NUM,
    KV_CMP_BLOCK_NUM,
    KV_ORI_BLOCK_NUM,
)
from vllm_ascend.ops._pypto_dsv4_csa.contract import DecodeCSAProgramSpec

SPECIALIZATION = "A3-TP1-B4-S8-ratio4"
DEFAULT_START_POSITION = 8191
DEFAULT_RESULT_DIR = Path(__file__).resolve().parent / "results" / "20260907_standalone_csa_perf"


def actual_workload_spec(*, batch: int = SUPPORTED_BATCH) -> dict[str, Any]:
    """Return the production CSA shapes used by the vLLM-Ascend adapter."""
    spec = DecodeCSAProgramSpec(batch=batch)
    tokens = spec.tokens
    layout = spec.physical_layout
    return {
        "specialization": SPECIALIZATION,
        "native_backend": "vllm.dsa_forward / DSAAttention (library CSA)",
        "pypto_backend": "vllm_ascend.ops._pypto_dsv4_csa decode_csa_core L1",
        "model": FLASH.name,
        "compress_ratio": 4,
        "tensor_parallel": 1,
        "batch": spec.batch,
        "seq": spec.seq,
        "tokens": tokens,
        "hidden_size": FLASH.hidden_size,
        "num_attention_heads": FLASH.num_attention_heads,
        "head_dim": FLASH.head_dim,
        "qk_rope_head_dim": FLASH.qk_rope_head_dim,
        "q_lora_rank": FLASH.q_lora_rank,
        "o_lora_rank": FLASH.o_lora_rank,
        "o_groups": FLASH.o_groups,
        "index_n_heads": FLASH.index_n_heads,
        "index_head_dim": FLASH.index_head_dim,
        "index_topk": FLASH.index_topk,
        "block_size": BLOCK_SIZE,
        "public_tensors": {
            "hidden_states": [tokens, FLASH.hidden_size],
            "attn_out": [tokens, FLASH.hidden_size],
            "dtype": "bfloat16",
        },
        "cache_blocks": {
            "swa": spec.swa_blocks,
            "compressed": spec.compressed_blocks,
            "main_state": spec.main_state_blocks,
            "inner_state": spec.inner_state_blocks,
            "indexer": spec.indexer_blocks,
        },
        "block_table_widths": {
            "swa": spec.swa_table_width,
            "compressed": spec.compressed_table_width,
            "main_state": spec.main_state_table_width,
            "inner_state": spec.inner_state_table_width,
            "indexer": spec.indexer_table_width,
        },
        "physical_spans": {
            "main_state": spec.main_state_span,
            "inner_state": spec.inner_state_span,
            "indexer_k": spec.indexer_k_span,
            "indexer_scale": spec.indexer_scale_span,
        },
        "physical_page_strides": {
            "main_state": layout.main_state_strides,
            "inner_state": layout.inner_state_strides,
            "indexer_k": layout.indexer_k_strides,
            "indexer_scale": layout.indexer_scale_strides,
        },
        "defaults": {
            "start_position": DEFAULT_START_POSITION,
            "swa_block_budget": KV_ORI_BLOCK_NUM,
            "compressed_block_budget": KV_CMP_BLOCK_NUM,
            "indexer_block_budget": IDX_CACHE_BLOCK_NUM,
        },
        "program_key": list(spec.key),
    }


def _assert_supported_shape(spec: Mapping[str, Any]) -> None:
    if spec["batch"] != SUPPORTED_BATCH or spec["seq"] != SUPPORTED_SEQ:
        raise ValueError(
            f"standalone case is pinned to B{SUPPORTED_BATCH}/S{SUPPORTED_SEQ}, got B{spec['batch']}/S{spec['seq']}"
        )
    if spec["seq"] != DECODE_SEQ:
        raise ValueError(f"contract seq {spec['seq']} != DECODE_SEQ {DECODE_SEQ}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--print-shapes", action="store_true", help="print the production shape table and exit")
    parser.add_argument("--runtime", choices=(TRB_RUNTIME, HBG_RUNTIME), default=TRB_RUNTIME)
    parser.add_argument(
        "--mode",
        choices=tuple(mode.value for mode in BenchmarkMode),
        default=BenchmarkMode.ACLGRAPH.value,
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batch", type=int, default=SUPPORTED_BATCH)
    parser.add_argument("--start-position", type=int, default=DEFAULT_START_POSITION)
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--enqueue-batch-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--master-port", type=int, default=29741)
    parser.add_argument(
        "--result-json",
        type=Path,
        default=DEFAULT_RESULT_DIR / "run.json",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    shapes = actual_workload_spec(batch=args.batch)
    _assert_supported_shape(shapes)
    if args.print_shapes:
        print(json.dumps(shapes, indent=2, sort_keys=True))
        return 0

    config = A3SingleOpBenchmarkConfig(
        runtime=args.runtime,
        mode=args.mode,
        device=args.device,
        batch=args.batch,
        warmup_iterations=args.warmups,
        sample_iterations=args.samples,
        enqueue_batch_size=args.enqueue_batch_size,
        start_position=args.start_position,
        seed=args.seed,
        master_port=args.master_port,
    )
    envelope: dict[str, Any] = {
        "schema_name": BENCHMARK_SCHEMA_NAME,
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "case": "standalone_csa_perf_case",
        "shapes": shapes,
        "runtime": config.runtime,
        "mode": config.mode.value,
        "start_position": config.start_position,
    }
    try:
        report = run_a3_single_op_benchmark(config)
    except NativeProductionUnsupported as error:
        envelope["status"] = "unsupported"
        envelope["reason"] = str(error)
        print(json.dumps(envelope, indent=2, sort_keys=True))
        return 2

    payload = json.loads(report.to_json())
    envelope["status"] = "ok"
    envelope["benchmark"] = payload
    text = json.dumps(envelope, indent=2, sort_keys=True)
    print(text)
    if args.result_json is not None:
        args.result_json.parent.mkdir(parents=True, exist_ok=True)
        args.result_json.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
