"""Full CSA eager/ACL Graph A-B-A replay with Native metadata and cache storage.

This exercises one captured graph with changed positions, lengths and page-table
contents. Each variant starts from the same cache snapshot; continuous accepted
token trajectories are a separate P3 check.
"""

from __future__ import annotations

import argparse
import traceback
from pathlib import Path

from dsv4_csa_env import activate, load_native_extension, write_json
from dsv4_csa_full_compare import check_untouched, compare_tensor


def pointer_signature(call, fixture):
    """Include both captured kernel arguments and Native producer outputs."""
    values = {f"core.{name}": value for name, value in call.args.items()}
    values["positions"] = fixture["positions"]
    for name, group in fixture["groups"].items():
        request = fixture["metadata"][group["prefix"]].req_metadata
        for field in ("query_start_loc", "seq_lens", "block_table", "slot_mapping"):
            values[f"{name}.{field}"] = getattr(request, field)
        if name in ("compressed", "indexer"):
            for index, value in enumerate(request.compressor_metadata):
                values[f"{name}.compressor.{index}"] = value
    prefix = fixture["groups"]["compressed"]["prefix"]
    request = fixture["metadata"][prefix].req_metadata
    values["rope.cos"] = request.cos[prefix]
    values["rope.sin"] = request.sin[prefix]
    return {
        name: None if value is None else (value.data_ptr(), tuple(value.shape), tuple(value.stride()))
        for name, value in values.items()
    }


def restore(fixture, snapshots):
    for name, group in fixture["groups"].items():
        group["allocation"].copy_(snapshots[name])
        group["owner"].kv_cache = group["views"]


def tolerance(dtype):
    import torch

    if dtype == torch.int8:
        return 0, 0
    if dtype == torch.float32:
        return 1e-3, 1e-3
    if dtype == torch.float16:
        return 1e-5, 1e-3
    return 1e-2, 1e-2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--history", type=int, default=131071)
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    repo = activate()
    report = {
        "status": "FAIL",
        "scope": "reference_full_CSA_graph_A_B_A_not_continuous_trace_or_target_checkpoint_acceptance",
        "batch": args.batch,
        "history": args.history,
        "seed": args.seed,
        "checkpoint": str(args.checkpoint),
        "cases": [],
        "thresholds_frozen_before_execution": {
            "native_output_bf16": [1e-2, 1e-2],
            "native_cache_bf16": [1e-2, 1e-2],
            "native_state_fp32": [1e-3, 1e-3],
            "native_scale_fp16": [1e-5, 1e-3],
            "native_indexer_int8_and_topk": [0, 0],
            "graph_vs_eager_all_tensors": [0, 0],
            "untouched_bytes": [0, 0],
        },
    }
    write_json(args.output_dir / "full_replay.json", report)
    try:
        import pypto.torch
        import torch
        import torch_npu
        from dsv4_csa_native_fixture import make_attention, make_config, native_session
        from dsv4_csa_native_forward import execute_native, make_numerical_fixture, update_numerical_metadata
        from dsv4_csa_native_layout import load_layer_weights
        from vllm.forward_context import BatchDescriptor

        load_native_extension(repo)
        import vllm_ascend.ops  # noqa: F401
        from vllm_ascend.ops.pypto.deepseek_v4_flash_dspark.native_adapter import (
            CSAOperators,
            NativeCSACall,
            prepare_weights,
        )

        if (args.checkpoint / "quant_model_description.json").exists():
            raise ValueError("Reference loader is forbidden for the target ModelSlim checkpoint")
        config = make_config(args.checkpoint, full_decode_graph=True)
        device = torch.device(f"npu:{args.device}")
        with native_session(config, args.device), torch.inference_mode():
            pypto.torch.init(device=args.device, platform="a2a3", runtime="tensormap_and_ringbuffer")
            attention = make_attention(config, device)
            records, methods = load_layer_weights(attention, args.checkpoint)
            report.update(weights=records, quant_methods=methods)
            native = make_numerical_fixture(config, device, attention, args.batch, args.history, args.seed)
            pto = make_numerical_fixture(config, device, attention, args.batch, args.history, args.seed)
            snapshots = {name: group["allocation"].cpu() for name, group in native["groups"].items()}
            for name, group in pto["groups"].items():
                torch.testing.assert_close(group["allocation"].cpu(), snapshots[name], atol=0, rtol=0)
            bindings = {
                name: (pto["metadata"][group["prefix"]], group["views"]) for name, group in pto["groups"].items()
            }
            call = NativeCSACall(
                CSAOperators.register(),
                prepare_weights(attention, bindings["indexer"][0].hadamard),
                pto["hidden"],
                pto["positions"],
                bindings,
                layer_name=pto["groups"]["compressed"]["prefix"],
            )
            descriptor = BatchDescriptor(num_tokens=pto["actual"], num_reqs=args.batch, uniform=True)
            executor = pto["executor"]

            def invoke():
                for task in pto["tasks"]:
                    executor.wait(task.stage, task.group_id)
                return call()

            print("Warming full CSA and Native external metadata events", flush=True)
            executor.submit(pto["tasks"], batch_descriptor=descriptor)
            invoke()
            executor.release()
            torch.npu.synchronize()
            signature = pointer_signature(call, pto)
            executor.submit(pto["tasks"], batch_descriptor=descriptor)
            graph = torch_npu.npu.NPUGraph()
            with torch_npu.npu.graph(graph):
                invoke()
            executor.release()
            torch.npu.synchronize()

            captured = {}

            def capture_topk(module, inputs, output):
                captured["topk"] = output.detach().clone()

            hook = attention.indexer.register_forward_hook(capture_topk)
            try:
                for iteration, variant in enumerate((0, 1, 0)):
                    starts = torch.full((args.batch,), args.history + 2 * variant, dtype=torch.int32)
                    restore(native, snapshots)
                    update_numerical_metadata(native, starts, swap_tables=iteration > 0)
                    expected = execute_native(config, attention, native)
                    entry = {"variant": variant, "history": int(starts[0]), "checks": {}}
                    checks = entry["checks"]
                    checks.update(
                        {f"native_guard.{name}": item for name, item in check_untouched(native, snapshots).items()}
                    )

                    restore(pto, snapshots)
                    update_numerical_metadata(pto, starts, swap_tables=iteration > 0)
                    if pointer_signature(call, pto) != signature:
                        raise AssertionError("Native metadata addresses changed after graph capture")
                    executor.submit(pto["tasks"], batch_descriptor=descriptor)
                    eager = invoke()
                    executor.release()
                    torch.npu.synchronize()
                    checks["eager_vs_native_output"] = compare_tensor(eager, expected, 1e-2, 1e-2)
                    checks["eager_vs_native_topk"] = compare_tensor(
                        call.args["idx_topk"], captured["topk"].view(args.batch * 6, 512), 0, 0
                    )
                    eager_output = eager.cpu()
                    eager_topk = call.args["idx_topk"].cpu()
                    eager_cache = {name: group["allocation"].cpu() for name, group in pto["groups"].items()}
                    checks.update(
                        {f"eager_guard.{name}": item for name, item in check_untouched(pto, snapshots).items()}
                    )

                    restore(pto, snapshots)
                    executor.submit(pto["tasks"], batch_descriptor=descriptor)
                    graph.replay()
                    executor.release()
                    torch.npu.synchronize()
                    checks["graph_vs_eager_output"] = compare_tensor(call.args["attn_out"], eager_output, 0, 0)
                    checks["graph_vs_eager_topk"] = compare_tensor(call.args["idx_topk"], eager_topk, 0, 0)
                    for name, group in pto["groups"].items():
                        checks[f"graph_vs_eager_allocation.{name}"] = compare_tensor(
                            group["allocation"], eager_cache[name], 0, 0
                        )
                        for index, view in enumerate(group["views"]):
                            checks[f"native_cache.{name}.{index}"] = compare_tensor(
                                view, native["groups"][name]["views"][index], *tolerance(view.dtype)
                            )
                    checks.update(
                        {f"graph_guard.{name}": item for name, item in check_untouched(pto, snapshots).items()}
                    )
                    entry["status"] = "PASS" if all(item["status"] == "PASS" for item in checks.values()) else "FAIL"
                    report["cases"].append(entry)
                    write_json(args.output_dir / "full_replay.json", report)
                    print(f"Full CSA replay variant {variant}: {entry['status']}", flush=True)
            finally:
                hook.remove()
                del graph
            report["fixed_addresses"] = True
            report["native_external_metadata_events"] = True
            report["status"] = "PASS" if all(case["status"] == "PASS" for case in report["cases"]) else "FAIL"
            if report["status"] != "PASS":
                raise AssertionError("Full CSA graph comparison failed; inspect full_replay.json")
    except Exception:
        report["error"] = traceback.format_exc()
        raise
    finally:
        write_json(args.output_dir / "full_replay.json", report)


if __name__ == "__main__":
    main()
