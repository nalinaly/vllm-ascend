"""Execute only the native real-weight CSA layer with full 128K candidate history."""

from __future__ import annotations

import argparse
import sys
import traceback
from contextlib import ExitStack, contextmanager, nullcontext
from pathlib import Path

from dsv4_csa_env import activate, load_native_extension, write_json


def make_numerical_fixture(config, device, attention, batch, history, seed, *, max_history=None):
    import torch
    from dsv4_csa_native_fixture import make_cache_groups
    from dsv4_csa_native_layout import allocate_native_cache

    from vllm_ascend.attention.attention_v1 import AscendAttentionState
    from vllm_ascend.attention.utils import AscendCommonAttentionMetadata
    from vllm_ascend.worker.block_table import BlockTable
    from vllm_ascend.worker.device_metadata import DeviceMetadataExecutor

    torch.manual_seed(seed)
    torch.npu.manual_seed(seed)
    query = 6
    actual = batch * query
    bounds_cpu = torch.arange(batch + 1, dtype=torch.int32) * query
    starts_cpu = torch.full((batch,), history, dtype=torch.int32)
    lengths_cpu = starts_cpu + query
    bounds, lengths = bounds_cpu.to(device), lengths_cpu.to(device)
    positions = (starts_cpu[:, None] + torch.arange(query)).flatten().to(device=device, dtype=torch.int64)
    hidden = torch.randn((actual, 4096), device=device, dtype=torch.bfloat16)
    groups = make_cache_groups(config, device, attention)
    metadata = {}
    common_cache = {}
    tasks = []
    for kind, group in groups.items():
        spec = group["spec"]
        full_history = kind in ("compressed", "indexer")
        capacity_history = history if max_history is None else max_history
        if capacity_history < history:
            raise ValueError("History capacity cannot be smaller than the initial history")
        columns = (capacity_history + query + spec.block_size - 1) // spec.block_size + 2
        per_request = columns if full_history else 9
        pages = batch * per_request + 1
        layout = allocate_native_cache(group, pages, device)
        group["layout"] = layout
        for index, view in enumerate(group["views"]):
            if view.dtype == torch.int8:
                view.copy_(torch.randint(-127, 128, view.shape, device=device, dtype=torch.int8))
            elif kind == "indexer" and index == 1:
                view.fill_(0.01)
            else:
                view.copy_(torch.randn(view.shape, dtype=view.dtype, device=device) * 0.5)
        group["owner"].kv_cache = group["views"]
        table = BlockTable(
            block_size=spec.block_size,
            max_num_reqs=40,
            max_num_blocks_per_req=columns,
            max_num_batched_tokens=400,
            pin_memory=True,
            device=device,
            num_speculative_tokens=5,
        )
        for row in range(batch):
            table.add_row(
                [1 + row * per_request + (per_request - 1 - col) % per_request for col in range(columns)], row
            )
        table.commit_block_table(batch)
        table.compute_slot_mapping(batch, bounds, positions)
        common = AscendCommonAttentionMetadata(
            query_start_loc=bounds,
            query_start_loc_cpu=bounds_cpu,
            seq_lens=lengths,
            _seq_lens_cpu=lengths_cpu + 5,
            seq_lens_cpu=None,
            num_reqs=batch,
            num_actual_tokens=actual,
            num_input_tokens=actual,
            max_query_len=query,
            max_seq_len=history + query + 5,
            block_table_tensor=table.block_table.gpu[:batch],
            slot_mapping=table.slot_mapping.gpu[:actual],
            positions=positions,
            attn_state=AscendAttentionState.SpecDecoding,
            causal=True,
        )
        builder = group["builder"]
        metadata[group["prefix"]] = builder.build(
            0, common, common_ratio_to_sas_metadata=common_cache, num_actual_reqs=batch
        )
        tasks.extend(builder.take_device_metadata_tasks())
        group["table"] = table
        group["common"] = common
    return {
        "groups": groups,
        "metadata": metadata,
        "tasks": tasks,
        "executor": DeviceMetadataExecutor(),
        "positions": positions,
        "hidden": hidden,
        "actual": actual,
    }


def update_numerical_metadata(fixture, starts_cpu, *, starts_device=None, swap_tables=False):
    """Rebuild Native metadata while retaining the fixture's device addresses."""
    import torch

    groups = fixture["groups"]
    sample = groups["swa"]["common"]
    query = sample.max_query_len
    batch = sample.num_reqs
    if starts_cpu.shape != (batch,) or starts_cpu.dtype != torch.int32:
        raise ValueError("Expected one INT32 history length per request")
    lengths = sample.seq_lens
    if starts_device is None:
        lengths.copy_(starts_cpu + query)
        fixture["positions"].copy_((starts_cpu[:, None] + torch.arange(query)).flatten())
    else:
        lengths.copy_(starts_device + query)
        fixture["positions"].copy_(
            (starts_device[:, None] + torch.arange(query, device=starts_device.device)).flatten()
        )
    tasks, shared = [], {}
    for group in groups.values():
        common, table, builder = group["common"], group["table"], group["builder"]
        if swap_tables:
            table.swap_row(0, batch - 1)
        table.commit_block_table(batch)
        table.compute_slot_mapping(batch, common.query_start_loc, fixture["positions"])
        common._seq_lens_cpu = starts_cpu + query + 5
        common.max_seq_len = int(common._seq_lens_cpu.max())
        fixture["metadata"][group["prefix"]] = builder.build(
            0, common, common_ratio_to_sas_metadata=shared, num_actual_reqs=batch
        )
        tasks.extend(builder.take_device_metadata_tasks())
    fixture["tasks"] = tasks


def execute_native(config, attention, fixture):
    import torch

    from vllm_ascend.ascend_forward_context import set_ascend_forward_context

    executor = fixture["executor"]
    executor.submit(fixture["tasks"])
    with (
        torch.inference_mode(),
        set_ascend_forward_context(
            fixture["metadata"],
            config,
            num_tokens=fixture["actual"],
            num_actual_tokens=fixture["actual"],
            device_metadata_executor=executor,
        ),
    ):
        output = attention(fixture["positions"], fixture["hidden"], None)
    executor.release()
    torch.npu.synchronize()
    return output


@contextmanager
def trace_native_stages(attention, output_dir, report):
    """Record synchronized diagnostics; never use this path for timing."""
    from functools import wraps
    from unittest.mock import patch

    import torch

    stages = []
    report["diagnostic_scope"] = "synchronized_stage_observation_not_performance_or_stream_correctness"

    def describe(value):
        if isinstance(value, torch.Tensor):
            data = value.detach().cpu()
            result = {"shape": list(data.shape), "dtype": str(data.dtype), "stride": list(value.stride())}
            if data.numel():
                finite = torch.isfinite(data)
                result["nonfinite"] = int((~finite).sum())
                valid = data[finite].float()
                if valid.numel():
                    result.update(min=float(valid.min()), max=float(valid.max()))
            return result
        if isinstance(value, (tuple, list)):
            return [describe(item) for item in value]
        if isinstance(value, dict):
            return {key: describe(item) for key, item in value.items()}
        return type(value).__name__

    def record(name, value):
        stages.append({"stage": name, "output": describe(value)})
        write_json(output_dir / "native_stage_trace.json", {"scope": report["diagnostic_scope"], "stages": stages})

    def observe(name, original):
        @wraps(original)
        def wrapped(*args, **kwargs):
            # The prolog receives populated RoPE rows after native metadata waits.
            if name.startswith("_mla_prolog"):
                record(name + ".hidden_cos_sin", args[:3])
            output = original(*args, **kwargs)
            record(name, output)
            return output

        return wrapped

    with ExitStack() as stack:
        for name, module in attention.named_modules():
            if name in (
                "wq_a",
                "wkv",
                "kv_norm",
                "q_norm",
                "q_norm_without_weight",
                "wo_b",
                "indexer",
                "compressor",
                "indexer.compressor",
            ):
                handle = module.register_forward_hook(lambda mod, inputs, output, name=name: record(name, output))
                stack.callback(handle.remove)
        impl = attention.dsa_attn.dsa_attn.impl
        for name in (
            "_mla_prolog_single_stream",
            "_mla_prolog_multistream",
            "_maybe_update_compressed_caches_and_select_topk",
            "_forward_attention",
            "_forward_o_proj",
        ):
            stack.enter_context(patch.object(impl, name, observe(name, getattr(impl, name))))
        yield
    report["diagnostic_stages"] = len(stages)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--history", type=int, default=131072)
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cann92-abi", action="store_true")
    parser.add_argument("--diagnostics", action="store_true", help="Synchronize and record native stage outputs")
    args = parser.parse_args()
    repo = activate()
    report = {
        "case_id": "P2_NATIVE_FORWARD_PREFLIGHT",
        "status": "FAIL",
        "python": sys.executable,
        "batch": args.batch,
        "history": args.history,
        "seed": args.seed,
    }
    try:
        import torch
        import torch_npu  # noqa: F401
        from dsv4_csa_native_fixture import make_attention, make_config, native_session
        from dsv4_csa_native_layout import load_layer_weights

        report["extension"] = str(load_native_extension(repo, cann92_abi=args.cann92_abi))
        import vllm_ascend.ops  # noqa: F401

        config = make_config(args.checkpoint)
        device = torch.device(f"npu:{args.device}")
        with native_session(config, args.device):
            attention = make_attention(config, device)
            records, methods = load_layer_weights(attention, args.checkpoint)
            report.update(weights=records, quant_methods=methods)
            fixture = make_numerical_fixture(config, device, attention, args.batch, args.history, args.seed)
            report["groups"] = {kind: group["layout"] for kind, group in fixture["groups"].items()}
            tracing = trace_native_stages(attention, args.output_dir, report) if args.diagnostics else nullcontext()
            with tracing:
                output = execute_native(config, attention, fixture)
            cpu_output = output.cpu()
            assert torch.isfinite(cpu_output).all(), "nonfinite native output"
            report.update(
                status="PASS",
                output_shape=list(output.shape),
                output_dtype=str(output.dtype),
                output_abs_max=float(cpu_output.float().abs().max()),
                peak_npu_allocated_bytes=torch.npu.max_memory_allocated(),
                scope="native_forward_execution_only_no_PTO_numerical_comparison",
            )
            print("native CSA attention.forward with real weights/full history: PASS", flush=True)
    except Exception:
        report["error"] = traceback.format_exc()
        raise
    finally:
        write_json(args.output_dir / f"native_forward_B{args.batch}_L{args.history}_device{args.device}.json", report)


if __name__ == "__main__":
    main()
