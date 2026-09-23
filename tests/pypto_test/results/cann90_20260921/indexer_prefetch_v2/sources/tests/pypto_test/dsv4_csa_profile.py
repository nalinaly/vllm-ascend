"""Capture a single reference-weight CSA layer as ACL Graph or PTO DFX.

Follow the Qwen profiling workflow: fresh Native/PTO processes, a separate eager
DFX process, disabled stack/shape/memory capture, and Perfetto-compatible exports.
This reuses the real Native layer and the production PTO CSA without rewriting
their computation. It is a fixed-input single-layer profile, not a model service.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

from dsv4_csa_env import activate, load_native_extension, write_json
from dsv4_csa_full_compare import compare_tensor


def fingerprint(tensor):
    import torch

    raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy()
    return hashlib.sha256(memoryview(raw)).hexdigest()


def export_swimlane(output_dir):
    records = output_dir / "chip_swimlane_records.json"
    deps = output_dir / "deps.json"
    raw = json.loads(records.read_text())
    boundaries = raw.get("metadata", {}).get("run_boundaries", [])
    assert len(boundaries) == 1, boundaries
    assert raw.get("metadata", {}).get("dropped_run_boundaries", 0) == 0
    task_count = len(raw.get("aicore_tasks", []))
    assert task_count > 0 and deps.is_file()
    merged = output_dir / "merged_swimlane.json"
    subprocess.run(
        [sys.executable, "-m", "simpler_setup.tools.swimlane_converter", str(records), "-o", str(merged)],
        check=True,
    )
    events = json.loads(merged.read_text())["traceEvents"]
    device_slices = sum(event.get("ph") == "X" and event.get("cat") == "event" for event in events)
    assert device_slices >= task_count, (device_slices, task_count)
    return {
        "captured_pypto_launches": len(boundaries),
        "aicore_task_count": task_count,
        "converted_device_slices": device_slices,
        "chip_swimlane_records": str(records),
        "deps_json": str(deps),
        "merged_swimlane": str(merged),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("native", "pypto"), required=True)
    parser.add_argument("--swimlane", action="store_true")
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=40)
    parser.add_argument("--history", type=int, default=131073)
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--replays", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.swimlane and args.variant != "pypto":
        parser.error("--swimlane requires --variant pypto")
    if args.replays < 1:
        parser.error("--replays must be positive")
    if (args.checkpoint / "quant_model_description.json").exists():
        raise ValueError("The reference loader must not load the target ModelSlim checkpoint")
    repo = activate()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "run_metadata.json"
    if report_path.exists():
        raise FileExistsError(f"Refusing to overwrite an earlier run: {report_path}")
    report = {
        "status": "FAIL",
        "scope": "single_reference_weight_CSA_layer_fixed_inputs_not_full_model_or_target_checkpoint",
        "variant": args.variant,
        "checkpoint": str(args.checkpoint.resolve()),
        "layer": "model.layers.2.attn",
        "batch": args.batch,
        "query_tokens_per_request": 6,
        "history": args.history,
        "seed": args.seed,
        "device": args.device,
        "tp": 1,
        "layout": "ND",
        "hidden_dtype": "bfloat16",
        "cudagraph_mode": "NONE" if args.swimlane else "single_full_CSA_NPUGraph",
        "decode_aclgraph_replays": 0 if args.swimlane else args.replays,
        "dfx_level": 4 if args.swimlane else 0,
        "dependency_collection": args.swimlane,
        "profiling_scope": "one_eager_CSA_call" if args.swimlane else "Native_metadata_submit_and_full_CSA_graph_replay",
        "profiler_with_stack": False,
        "profiler_record_shapes": False,
        "profiler_with_memory": False,
        "profiler_level": None if args.swimlane else "Level1",
    }
    write_json(report_path, report)
    try:
        import torch
        import torch_npu
        from dsv4_csa_native_fixture import make_attention, make_config, native_session
        from dsv4_csa_native_forward import make_numerical_fixture
        from dsv4_csa_native_layout import load_layer_weights
        from vllm.forward_context import BatchDescriptor

        torch.set_num_threads(2)
        load_native_extension(repo)
        import vllm_ascend.ops  # noqa: F401
        from vllm_ascend.ascend_forward_context import set_ascend_forward_context

        config = make_config(args.checkpoint, full_decode_graph=not args.swimlane)
        device = torch.device(f"npu:{args.device}")
        with native_session(config, args.device), torch.inference_mode():
            attention = make_attention(config, device)
            weights, methods = load_layer_weights(attention, args.checkpoint)
            fixture = make_numerical_fixture(config, device, attention, args.batch, args.history, args.seed)
            report.update(weights=weights, quant_methods=methods)
            report["input_hashes"] = {
                "hidden": fingerprint(fixture["hidden"]),
                "positions": fingerprint(fixture["positions"]),
                **{f"allocation.{name}": fingerprint(group["allocation"]) for name, group in fixture["groups"].items()},
                **{f"table.{name}": fingerprint(group["common"].block_table_tensor) for name, group in fixture["groups"].items()},
            }
            report["cache_layouts"] = {name: group["layout"] for name, group in fixture["groups"].items()}
            executor = fixture["executor"]
            descriptor = BatchDescriptor(num_tokens=fixture["actual"], num_reqs=args.batch, uniform=True)
            if args.variant == "pypto":
                import pypto.torch
                from vllm_ascend.ops.pypto.deepseek_v4_flash_dspark.native_adapter import (
                    CSAOperators,
                    NativeCSACall,
                    prepare_weights,
                )

                kwargs = dict(device=args.device, platform="a2a3", runtime="tensormap_and_ringbuffer")
                if args.swimlane:
                    kwargs.update(enable_chip_swimlane=4, enable_dep_gen=True, output_dir=output_dir)
                pypto.torch.init(**kwargs)
                bindings = {
                    name: (fixture["metadata"][group["prefix"]], group["views"])
                    for name, group in fixture["groups"].items()
                }
                call = NativeCSACall(
                    CSAOperators.register(), prepare_weights(attention, bindings["indexer"][0].hadamard),
                    fixture["hidden"], fixture["positions"], bindings,
                    layer_name=fixture["groups"]["compressed"]["prefix"],
                )

                def invoke():
                    for task in fixture["tasks"]:
                        executor.wait(task.stage, task.group_id)
                    return call()
            else:
                def invoke():
                    with set_ascend_forward_context(
                        fixture["metadata"], config, num_tokens=fixture["actual"],
                        num_actual_tokens=fixture["actual"], device_metadata_executor=executor,
                    ):
                        return attention(fixture["positions"], fixture["hidden"], None)

            eager_output = None
            for _ in range(2):
                executor.submit(fixture["tasks"], batch_descriptor=descriptor)
                output = invoke()
                executor.release()
                torch.npu.synchronize()
                if eager_output is None:
                    eager_output = output.cpu()
                else:
                    report["fixed_input_repeat_exact"] = compare_tensor(output, eager_output, 0, 0)
                    assert report["fixed_input_repeat_exact"]["status"] == "PASS"
            report["eager_warmup_calls"] = 2
            print(f"{args.variant}: warmup complete", flush=True)

            if args.swimlane:
                executor.submit(fixture["tasks"], batch_descriptor=descriptor)
                for task in fixture["tasks"]:
                    executor.wait(task.stage, task.group_id)
                pypto.torch.begin_dfx()
                try:
                    output = call()
                finally:
                    pypto.torch.end_dfx()
                executor.release()
                torch.npu.synchronize()
                report.update(export_swimlane(output_dir))
                report["diagnostic_timing_only"] = True
            else:
                graph = torch_npu.npu.NPUGraph()
                executor.submit(fixture["tasks"], batch_descriptor=descriptor)
                with torch_npu.npu.graph(graph):
                    output = invoke()
                executor.release()
                torch.npu.synchronize()
                executor.submit(fixture["tasks"], batch_descriptor=descriptor)
                graph.replay()
                executor.release()
                torch.npu.synchronize()
                report["graph_warmup_replays"] = 1
                trace_dir = output_dir / "torch_npu_trace"
                with torch_npu.profiler.profile(
                    activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
                    record_shapes=False, profile_memory=False, with_stack=False,
                    experimental_config=torch_npu.profiler._ExperimentalConfig(
                        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
                    ),
                    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(trace_dir)),
                ):
                    for replay in range(args.replays):
                        with torch.profiler.record_function(f"csa.profile.replay.{replay}"):
                            executor.submit(fixture["tasks"], batch_descriptor=descriptor)
                            graph.replay()
                            executor.release()
                            torch.npu.synchronize()
                traces = list(trace_dir.rglob("trace_view.json"))
                assert len(traces) == 1, traces
                easy_trace = output_dir / f"{args.variant}_profiling.json"
                shutil.copy2(traces[0], easy_trace)
                report["trace_json"] = str(easy_trace)
                del graph

            report["profiled_vs_eager_output"] = compare_tensor(output, eager_output, 0, 0)
            assert report["profiled_vs_eager_output"]["status"] == "PASS"
            torch.save({"output": output.cpu()}, output_dir / "output.pt")
            report["native_external_metadata_events"] = True
            report["status"] = "PASS"
            print(f"{args.variant}: capture PASS", flush=True)
    except BaseException:
        report["error"] = traceback.format_exc()
        raise
    finally:
        write_json(report_path, report)


if __name__ == "__main__":
    main()
