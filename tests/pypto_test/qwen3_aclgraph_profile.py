#!/usr/bin/env python3
"""Capture comparable Qwen3-14B Native/PyPTO full-decode ACL Graph traces.

Run this script once per variant in a fresh process.  If ``--decode-steps`` is
three, four output tokens are requested because the first token is sampled by
prefill and the following three model forwards are decode graph replays.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

import torch
from pto_eager_env import activate_pto_eager

PYPTO_DFX_DIR_ENV = "VLLM_ASCEND_QWEN3_PYPTO_DFX_DIR"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("native", "pypto"), required=True)
    parser.add_argument("--model", default="/mnt/workspace/inductor/models/Qwen3-14B")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--decode-steps", type=int, default=3)
    return parser.parse_args()


def _generate(llm: Any, prompt: str, sampling_params: Any, output_tokens: int):
    outputs = llm.generate([prompt], sampling_params, use_tqdm=False)
    assert len(outputs) == 1
    result = outputs[0].outputs[0]
    if len(result.token_ids) != output_tokens:
        raise RuntimeError(f"expected {output_tokens} output tokens, got {result.token_ids}")
    return result


def _prepare_output_dir(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_dir = output_dir / "torch_npu_trace"
    if trace_dir.exists() and any(trace_dir.iterdir()):
        raise FileExistsError(f"refusing to mix a new profile with existing files in {trace_dir}")
    trace_dir.mkdir(parents=True, exist_ok=True)
    return trace_dir


def main() -> None:
    args = parse_args()
    if args.variant == "pypto":
        activate_pto_eager()
    from vllm import LLM, SamplingParams  # noqa: PLC0415

    if args.decode_steps <= 0:
        raise ValueError("--decode-steps must be positive")
    if os.environ.get(PYPTO_DFX_DIR_ENV):
        raise RuntimeError(
            f"unset {PYPTO_DFX_DIR_ENV}: begin_dfx/end_dfx synchronization must not be mixed with ACL Graph profiling"
        )

    args.output_dir = args.output_dir.expanduser().resolve()
    trace_dir = _prepare_output_dir(args.output_dir)
    output_tokens = args.decode_steps + 1

    llm_kwargs = {
        "model": args.model,
        "tensor_parallel_size": 1,
        "dtype": "bfloat16",
        "block_size": 128,
        "max_model_len": 256,
        "max_num_seqs": 1,
        "max_num_batched_tokens": 256,
        "gpu_memory_utilization": 0.75,
        "enable_prefix_caching": False,
        "additional_config": {
            "weight_nz_mode": 0,
            "enable_kv_nz": False,
        },
        # The decode routine is one full-model graph. mode=NONE is required by
        # the attention-only Python dispatch so capture sees the PyPTO branch.
        "compilation_config": {
            "mode": 0,
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": [1],
        },
        "profiler_config": {
            "profiler": "torch",
            "torch_profiler_dir": str(trace_dir),
            "torch_profiler_with_stack": False,
            "torch_profiler_record_shapes": False,
            "torch_profiler_with_memory": False,
            "capture_torch_profiler": False,
        },
        "trust_remote_code": False,
        "disable_log_stats": True,
        "seed": 0,
    }
    if args.variant == "pypto":
        llm_kwargs["hf_overrides"] = {"architectures": ["PyptoAttentionQwen3ForCausalLM"]}

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=output_tokens,
        min_tokens=output_tokens,
        ignore_eos=True,
        seed=0,
    )
    llm = LLM(**llm_kwargs)

    warmup = _generate(llm, args.prompt, sampling_params, output_tokens)
    torch.npu.synchronize()

    llm.start_profile(profile_prefix=f"qwen3_14b_{args.variant}_full_decode")
    started_at = time.perf_counter()
    measured = _generate(llm, args.prompt, sampling_params, output_tokens)
    torch.npu.synchronize()
    elapsed_s = time.perf_counter() - started_at
    llm.stop_profile()

    traces = list(trace_dir.rglob("trace_view.json"))
    if len(traces) != 1:
        raise RuntimeError(f"expected one trace_view.json below {trace_dir}, got {traces}")
    easy_trace = args.output_dir / f"{args.variant}_profiling.json"
    shutil.copy2(traces[0], easy_trace)

    metadata = {
        "variant": args.variant,
        "model": args.model,
        "prompt": args.prompt,
        "tensor_parallel_size": 1,
        "device": 0,
        "dtype": "bfloat16",
        "weight_layout": "ND",
        "kv_cache_layout": "ND",
        "compilation_mode": "NONE",
        "cudagraph_mode": "FULL_DECODE_ONLY",
        "cudagraph_capture_sizes": [1],
        "output_tokens": output_tokens,
        "decode_aclgraph_replays": args.decode_steps,
        "profiler_with_stack": False,
        "profiler_record_shapes": False,
        "profiler_with_memory": False,
        "warmup_token_ids": list(warmup.token_ids),
        "profiled_token_ids": list(measured.token_ids),
        "profiled_text": measured.text,
        "profiled_generate_elapsed_s": elapsed_s,
        "trace_json": str(easy_trace),
    }
    metadata_path = args.output_dir / "run_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
