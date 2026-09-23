"""Reference full CSA accepted-token trajectories, first eager then ACL Graph.

Cache/state survive between steps. Both phases start from the same initial
allocation contents and use Native acceptance correction and metadata producers.
This covers G01/G02 trajectories, not batch changes or request lifecycle cases.
"""

from __future__ import annotations

import argparse
import hashlib
import traceback
from contextlib import nullcontext
from pathlib import Path

from dsv4_csa_env import activate, load_native_extension, write_json
from dsv4_csa_full_compare import check_untouched, compare_tensor
from dsv4_csa_full_replay import pointer_signature, restore, tolerance


def digest(tensor):
    import torch

    raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy()
    return hashlib.sha256(memoryview(raw)).hexdigest()


def accepted_counts(torch, trajectory, step, batch):
    if trajectory == "mixed":
        pattern = (1, 6, 2, 5, 5)
        counts = [pattern[(step + row) % len(pattern)] for row in range(batch)]
    elif trajectory == "all1":
        counts = [1] * batch
    elif trajectory == "all6":
        counts = [6] * batch
    else:
        counts = [1 if step % 25 < 20 else 6] * batch
    return torch.tensor(counts, dtype=torch.int32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--history", type=int, default=131071)
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--trajectory", choices=("mixed", "all1", "all6", "reject_then_accept"), default="mixed")
    parser.add_argument("--diagnostic-step", type=int, default=-1)
    parser.add_argument("--query-diagnostic-step", type=int, default=-1)
    parser.add_argument("--output-diagnostic-step", type=int, default=-1)
    parser.add_argument("--stop-after-step", type=int, default=-1,
                        help="Stop after this eager step for diagnosis; keep --steps fixture capacity unchanged")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    if args.stop_after_step >= args.steps or args.stop_after_step < -1:
        parser.error("--stop-after-step must be -1 or a step within --steps")
    repo = activate()
    report = {
        "status": "FAIL",
        "scope": "reference_full_CSA_continuous_trajectory_not_target_checkpoint_or_all_P3_cases",
        "batch": args.batch,
        "history": args.history,
        "seed": args.seed,
        "steps": args.steps,
        "fixture_capacity": args.history + 6 * args.steps + 12,
        "trajectory": args.trajectory,
        "checkpoint": str(args.checkpoint),
        "thresholds_frozen_before_execution": {
            "output_bf16": [1e-2, 1e-2],
            "cache_bf16": [1e-2, 1e-2],
            "state_fp32": [1e-3, 1e-3],
            "scale_fp16": [1e-5, 1e-3],
            "indexer_int8_and_topk": [0, 0],
            "graph_vs_eager_and_untouched_bytes": [0, 0],
        },
        "phases": {"eager": [], "graph": []},
    }
    write_json(args.output_dir / "continuous.json", report)
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
        from vllm_ascend.spec_decode.utils import (
            correct_optimistic_seq_lens_cpu,
            update_num_computed_tokens_for_batch_change,
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
            capacity = args.history + 6 * args.steps + 12
            native = make_numerical_fixture(
                config, device, attention, args.batch, args.history, args.seed, max_history=capacity
            )
            pto = make_numerical_fixture(
                config, device, attention, args.batch, args.history, args.seed, max_history=capacity
            )
            initial = {name: group["allocation"].cpu() for name, group in native["groups"].items()}
            for name, group in pto["groups"].items():
                torch.testing.assert_close(group["allocation"].cpu(), initial[name], atol=0, rtol=0)
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
            diagnostic_capture = {}
            diagnostic_enabled = False
            quantize_key = attention.indexer.ops.quantize_key_and_update_cache

            def capture_compressor(module, inputs, output):
                if diagnostic_enabled:
                    diagnostic_capture["compressor_output"] = output[0].detach().clone()
                    diagnostic_capture["compressor_slots"] = output[1].detach().clone()

            def capture_quantization(key, *arguments):
                if diagnostic_enabled:
                    diagnostic_capture["quant_input"] = key.detach().clone()
                result = quantize_key(key, *arguments)
                if diagnostic_enabled:
                    diagnostic_capture["quant_output"] = result[0].detach().clone()
                    diagnostic_capture["quant_scale"] = result[1].detach().clone()
                return result

            compressor_hook = None
            if args.diagnostic_step >= 0:
                compressor_hook = attention.indexer.compressor.register_forward_hook(capture_compressor)
                attention.indexer.ops.quantize_key_and_update_cache = capture_quantization
            query_boundary = None
            if args.query_diagnostic_step >= 0:
                from dsv4_csa_query_boundary import QueryBoundary

                query_boundary = QueryBoundary(attention)
            output_boundary = None
            if args.output_diagnostic_step >= 0:
                from dsv4_csa_output_boundary import OutputBoundary

                output_boundary = OutputBoundary(attention)
            eager_fingerprints = []
            advances = []
            try:
                for phase in ("eager", "graph"):
                    restore(native, initial)
                    restore(pto, initial)
                    previous_snapshots = {"native": initial, "pto": initial}
                    starts = torch.full((args.batch,), args.history, dtype=torch.int32)
                    computed = starts.to(device)
                    accepted = torch.ones(args.batch, dtype=torch.int32, device=device)
                    previous_rows = torch.arange(args.batch, dtype=torch.int64)
                    previous_rows_device = previous_rows.to(device)
                    drafts = torch.full((args.batch,), 5, dtype=torch.int32)
                    drafts_device = drafts.to(device)
                    for step in range(args.steps):
                        # Test inputs are deterministic host data, independent of device results.
                        generator = torch.Generator().manual_seed(args.seed + step)
                        hidden = torch.randn((args.batch * 6, 4096), generator=generator, dtype=torch.bfloat16)
                        for fixture in (native, pto):
                            fixture["hidden"].copy_(hidden)
                            update_numerical_metadata(fixture, starts, starts_device=computed)
                        if pointer_signature(call, pto) != signature:
                            raise AssertionError("Captured addresses changed during continuous replay")
                        for group in native["groups"].values():
                            group["owner"].kv_cache = group["views"]
                        diagnostic_enabled = phase == "eager" and step == args.diagnostic_step
                        if query_boundary is not None:
                            query_boundary.enabled = phase == "eager" and step == args.query_diagnostic_step
                        if output_boundary is not None:
                            output_boundary.enabled = phase == "eager" and step == args.output_diagnostic_step
                        if diagnostic_enabled:
                            from vllm_ascend.ops.pypto.deepseek_v4_flash_dspark.native_storage import physical_pages

                            state_before = {
                                "native": physical_pages(native["groups"]["indexer_state"]["views"][0]).clone(),
                                "pto": call.state_storage["indexer_state"].clone(),
                            }
                        expected = execute_native(config, attention, native)
                        if diagnostic_enabled:
                            group = native["groups"]["indexer_state"]
                            state_slots = native["metadata"][group["prefix"]].req_metadata.slot_mapping.long()
                            diagnostic_capture["native_state_tokens"] = group["views"][0][
                                state_slots[:, 0], state_slots[:, 1], 0, :
                            ].clone()
                        profile_last = phase == "graph" and step == args.steps - 1
                        profiler = (
                            torch_npu.profiler.profile(
                                activities=[
                                    torch_npu.profiler.ProfilerActivity.CPU,
                                    torch_npu.profiler.ProfilerActivity.NPU,
                                ],
                                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                                    str(args.output_dir / "graph_profile")
                                ),
                            )
                            if profile_last
                            else nullcontext()
                        )
                        with profiler:
                            executor.submit(pto["tasks"], batch_descriptor=descriptor)
                            if phase == "eager":
                                invoke()
                            else:
                                graph.replay()
                            executor.release()
                            # Assertion/profiler boundary; no sync inside the captured invocation.
                            torch.npu.synchronize()
                        checks = {
                            "output": compare_tensor(call.args["attn_out"], expected, 1e-2, 1e-2),
                            "topk": compare_tensor(
                                call.args["idx_topk"], captured["topk"].view(args.batch * 6, 512), 0, 0
                            ),
                        }
                        for name, group in pto["groups"].items():
                            for index, view in enumerate(group["views"]):
                                checks[f"cache.{name}.{index}"] = compare_tensor(
                                    view, native["groups"][name]["views"][index], *tolerance(view.dtype)
                                )
                        for label, fixture in (("native", native), ("pto", pto)):
                            checks.update(
                                {
                                    f"{label}_guard.{name}": result
                                    for name, result in check_untouched(fixture, previous_snapshots[label]).items()
                                }
                            )
                            previous_snapshots[label] = {
                                name: group["allocation"].cpu() for name, group in fixture["groups"].items()
                            }
                        fingerprint = {
                            "output": digest(call.args["attn_out"]),
                            "topk": digest(call.args["idx_topk"]),
                            "allocations": {name: digest(value) for name, value in previous_snapshots["pto"].items()},
                        }
                        if phase == "eager":
                            eager_fingerprints.append(fingerprint)
                        else:
                            checks["graph_vs_eager_all_bytes"] = {
                                "status": "PASS" if fingerprint == eager_fingerprints[step] else "FAIL",
                                "method": "SHA256 of complete contiguous output, Top-K and physical allocations",
                            }
                        counts = accepted_counts(torch, args.trajectory, step, args.batch)
                        optimistic = starts + 6
                        corrected = optimistic.clone()
                        correct_optimistic_seq_lens_cpu(
                            corrected.numpy(), previous_rows.numpy(), drafts.numpy(), counts.numpy(), args.batch
                        )
                        update_num_computed_tokens_for_batch_change(
                            computed,
                            accepted,
                            previous_rows_device,
                            counts.to(device),
                            drafts_device,
                            optimistic.to(device),
                        )
                        next_starts = starts + counts
                        torch.testing.assert_close(corrected, next_starts, rtol=0, atol=0)
                        torch.testing.assert_close(computed.cpu(), next_starts, rtol=0, atol=0)
                        torch.testing.assert_close(accepted.cpu(), counts, rtol=0, atol=0)
                        entry = {
                            "step": step,
                            "starts": starts.tolist(),
                            "accepted": counts.tolist(),
                            "checks": checks,
                            "pto_fingerprint": fingerprint,
                            "status": "PASS" if all(value["status"] == "PASS" for value in checks.values()) else "FAIL",
                        }
                        report["phases"][phase].append(entry)
                        if diagnostic_enabled:
                            from dsv4_csa_compressor_boundary import observe_compressor

                            report["compressor_diagnostics"] = observe_compressor(
                                call, state_before, diagnostic_capture, args.output_dir
                            )
                        if query_boundary is not None and query_boundary.enabled:
                            report["query_diagnostics"] = query_boundary.observe(
                                call, captured["topk"], args.output_dir
                            )
                        if output_boundary is not None and output_boundary.enabled:
                            report["output_diagnostics"] = output_boundary.observe(
                                call, expected, args.output_dir, native=native
                            )
                        write_json(args.output_dir / "continuous.json", report)
                        print(f"{phase} step {step}: {entry['status']}", flush=True)
                        if entry["status"] != "PASS":
                            torch.save(
                                {
                                    "hidden": pto["hidden"].cpu(),
                                    "positions": pto["positions"].cpu(),
                                    "native_output": expected.cpu(),
                                    "pto_output": call.args["attn_out"].cpu(),
                                    "native_topk": captured["topk"].cpu(),
                                    "pto_topk": call.args["idx_topk"].cpu(),
                                    "pto_scores": call.args["idx_topk_scores"].cpu(),
                                },
                                args.output_dir / "failure.pt",
                            )
                            raise AssertionError(f"Continuous {phase} failed at step {step}")
                        if phase == "eager" and step == args.stop_after_step:
                            report["status"] = "DIAGNOSTIC_COMPLETE"
                            report["stopped_after_eager_step"] = step
                            report["hundred_steps_completed"] = False
                            return
                        starts = next_starts
                        if phase == "eager":
                            advances.extend(counts.tolist())
                report["mean_advance"] = sum(advances) / len(advances)
                if args.trajectory == "mixed" and args.steps % 5 == 0:
                    assert report["mean_advance"] == 3.8
                report["fixed_addresses"] = True
                report["native_acceptance_correction"] = True
                report["native_external_metadata_events"] = True
                report["cache_state_preserved_between_steps"] = True
                report["hundred_steps_completed"] = args.steps >= 100
                report["graph_profile_directory"] = str(args.output_dir / "graph_profile")
                report["status"] = "PASS"
            finally:
                hook.remove()
                if compressor_hook is not None:
                    compressor_hook.remove()
                    attention.indexer.ops.quantize_key_and_update_cache = quantize_key
                if query_boundary is not None:
                    query_boundary.close()
                if output_boundary is not None:
                    output_boundary.close()
                del graph
    except Exception:
        report["error"] = traceback.format_exc()
        raise
    finally:
        write_json(args.output_dir / "continuous.json", report)


if __name__ == "__main__":
    main()
