#!/usr/bin/env python3
"""Capture one PyPTO attention layer while Qwen3-14B runs end to end.

The environment variable enables a one-shot DFX window around layer 0's first
real decode call. ACL Graph is deliberately disabled because begin_dfx/end_dfx
synchronize the current torch_npu stream and its host task queue.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from pto_eager_env import activate_pto_eager

PYPTO_DFX_DIR_ENV = "VLLM_ASCEND_QWEN3_PYPTO_DFX_DIR"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/mnt/workspace/inductor/models/Qwen3-14B")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt", default="The capital of France is")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    activate_pto_eager()
    from vllm import LLM, SamplingParams  # noqa: PLC0415

    pto_eager_root = Path(os.environ.get("PTO_EAGER_ROOT", "")).resolve()
    converter_python = pto_eager_root / ".venv" / "bin" / "python"
    if not converter_python.is_file():
        raise RuntimeError("source pto_eager/env.sh before running this script")

    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty DFX directory {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ[PYPTO_DFX_DIR_ENV] = str(output_dir)
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    llm = LLM(
        model=args.model,
        hf_overrides={"architectures": ["PyptoAttentionQwen3ForCausalLM"]},
        tensor_parallel_size=1,
        dtype="bfloat16",
        block_size=128,
        max_model_len=256,
        max_num_seqs=1,
        max_num_batched_tokens=256,
        gpu_memory_utilization=0.75,
        enable_prefix_caching=False,
        enforce_eager=True,
        additional_config={"weight_nz_mode": 0, "enable_kv_nz": False},
        compilation_config={"mode": 0, "cudagraph_mode": "NONE"},
        trust_remote_code=False,
        disable_log_stats=True,
        seed=0,
    )
    # Two output tokens mean one prefill sample followed by one decode forward.
    outputs = llm.generate(
        [args.prompt],
        SamplingParams(
            temperature=0.0,
            max_tokens=2,
            min_tokens=2,
            ignore_eos=True,
            seed=0,
        ),
        use_tqdm=False,
    )
    result = outputs[0].outputs[0]
    if len(result.token_ids) != 2:
        raise RuntimeError(f"expected two output tokens, got {result.token_ids}")

    records = output_dir / "chip_swimlane_records.json"
    deps = output_dir / "deps.json"
    if not records.is_file() or not deps.is_file():
        raise RuntimeError(f"PyPTO DFX did not produce both {records} and {deps}")
    raw = json.loads(records.read_text())
    boundaries = raw.get("metadata", {}).get("run_boundaries", [])
    if len(boundaries) != 1:
        raise RuntimeError(f"expected exactly one captured layer launch, got {len(boundaries)}")
    aicore_task_count = len(raw.get("aicore_tasks", []))
    if aicore_task_count == 0:
        raise RuntimeError("the captured layer launch contains no AICore tasks")

    merged = output_dir / "merged_swimlane.json"
    subprocess.run(
        [
            str(converter_python),
            "-m",
            "simpler_setup.tools.swimlane_converter",
            str(records),
            "-o",
            str(merged),
        ],
        check=True,
    )
    trace_events = json.loads(merged.read_text()).get("traceEvents", [])
    device_task_count = sum(event.get("ph") == "X" and event.get("cat") == "event" for event in trace_events)
    if device_task_count < aicore_task_count:
        raise RuntimeError(
            f"converted swimlane lost device tasks: raw={aicore_task_count}, converted={device_task_count}"
        )
    summary = {
        "model": args.model,
        "prompt": args.prompt,
        "token_ids": list(result.token_ids),
        "text": result.text,
        "tensor_parallel_size": 1,
        "device": 0,
        "layout": "ND",
        "cudagraph_mode": "NONE",
        "captured_transformer_layer": 0,
        "captured_pypto_launches": len(boundaries),
        "aicore_task_count": aicore_task_count,
        "converted_device_slices": device_task_count,
        "chip_swimlane_records": str(records),
        "deps_json": str(deps),
        "merged_swimlane": str(merged),
    }
    (output_dir / "capture_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
