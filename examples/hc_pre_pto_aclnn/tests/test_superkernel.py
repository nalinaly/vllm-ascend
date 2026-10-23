# SPDX-License-Identifier: Apache-2.0
"""Check and time a T96 Add -> ACLNN HC_pre -> Mul chain using npugraph_ex.

Run each mode in a separate process with the same private OPP package.
SuperKernel membership is audited from sk_meta in addition to this numerical test.
"""

import argparse
import json
import os
import statistics
import time
import traceback
from pathlib import Path

import torch
import torch_npu
from check_outputs import reference

TOKENS = 96
OUTPUT_NAMES = ("mixed", "post", "comb")


class HcPreChain(torch.nn.Module):
    def __init__(self, mark_scope, neighbours):
        super().__init__()
        self.mark_scope = mark_scope
        self.neighbours = neighbours

    def forward(self, x, weight, scale, base, addend, multiplier):
        if self.mark_scope:
            torch.npu.super_kernel_scope_begin("hcpre_add_mul")
        shifted = torch.ops.bytenn_pto_ops.ascendc_add(x, addend) if self.neighbours == "ascendc" else torch.add(x, addend)
        mixed, post, comb = torch.ops.bytenn_pto_ops.hc_pre_pto_bnn(shifted, weight, scale, base)
        scaled = (
            torch.ops.bytenn_pto_ops.ascendc_mul(mixed, multiplier)
            if self.neighbours == "ascendc"
            else torch.mul(mixed, multiplier)
        )
        if self.mark_scope:
            torch.npu.super_kernel_scope_end("hcpre_add_mul")
        return scaled, post, comb


def make_inputs(seed):
    generator = torch.Generator().manual_seed(seed)
    return (
        torch.randn((TOKENS, 4, 4096), dtype=torch.bfloat16, generator=generator),
        torch.randn((24, 16384), generator=generator) * 0.01,
        torch.tensor([0.8, 1.0, 1.2]),
        torch.randn(24, generator=generator) * 0.1,
        torch.randn((TOKENS, 4, 4096), dtype=torch.bfloat16, generator=generator) * 0.125,
        torch.randn((TOKENS, 4096), dtype=torch.bfloat16, generator=generator) * 0.125 + 0.875,
    )


def cpu_reference(inputs):
    x, weight, scale, base, addend, multiplier = inputs
    mixed, post, comb = reference(torch.add(x, addend), weight, scale, base)
    return torch.mul(mixed.to(torch.bfloat16), multiplier), post, comb


def compare(outputs, expected):
    report = {}
    for name, value, ref in zip(OUTPUT_NAMES, outputs, expected):
        actual = value.detach().cpu().float()
        ref = ref.detach().cpu().float()
        atol, rtol = (0.01, 0.01) if name == "mixed" else (1e-5, 1e-4)
        report[name] = {
            "shape": list(actual.shape),
            "finite": bool(torch.isfinite(actual).all()),
            "passed": bool(torch.allclose(actual, ref, atol=atol, rtol=rtol)),
            "max_abs": float((actual - ref).abs().max()),
            "atol": atol,
            "rtol": rtol,
        }
    return report


def audit_fusion(early_start=0):
    """Fail when compilation or fusion silently falls back to ordinary tasks."""
    models = list((Path.cwd() / "sk_meta" / str(os.getpid())).glob("model_*"))
    assert len(models) == 1, f"Expected one SK model, found: {models}"
    model = models[0]
    fused = (model / "sk_fused_nodes.log").read_text()
    members = [
        line.split("funcName:static_kernel_", 1)[1].split("_", 1)[0]
        for line in fused.splitlines()
        if "funcName:static_kernel_" in line
    ]
    assert members == ["HcPreAdd", "HcPrePtoBnn", "HcPreMul"], fused
    assert fused.count("SK Function:") == 1 and "Node Count: 3" in fused, fused
    updated = json.loads((model / "sk_mdl_updated.json").read_text())
    expected_entry = "sk_entry_mix12_early_start" if early_start else "sk_entry_mix12"
    kernels = [task for task in updated if task.get("args", {}).get("Task Type", "").startswith("KERNEL")]
    assert len(kernels) == 1 and kernels[0]["name"] == expected_entry, updated
    queue = json.loads((model / "sk_task_queue.json").read_text())
    assert queue["scopeCount"] == 1, queue
    queues = queue["scopes"][0]["taskQueues"]
    assert queues["aic"]["funcCnt"] == 1 and queues["aiv"]["funcCnt"] == 3, queues
    log = (model / "super_kernel.log").read_text()
    capabilities = [
        line for line in log.splitlines() if "[InitKernelResolvedFuncs]" in line and "bindMap size=" in line
    ]
    assert len(capabilities) == len(members), capabilities
    for line in capabilities:
        assert f"earlyStartWaitFlag={early_start}," in line, line
        assert f"earlyStartSetFlag={early_start}," in line, line
    entries = [line for line in log.splitlines() if "[GenEntryInfo]" in line and "enableEarlyStart=" in line]
    assert entries and all(f"enableEarlyStart={early_start}," in line for line in entries), entries
    states = [line for line in log.splitlines() if "[LogEarlyStartFinalState]" in line]
    assert len(states) == (len(members) if early_start else 0), states
    configs = [int(line.split("funcConfig=", 1)[1].split(",", 1)[0], 0) for line in states]
    assert all(configs), states
    return {
        "passed": True,
        "members": members,
        "device_task": kernels[0]["name"],
        "early_start": early_start,
        "early_start_capabilities": capabilities,
        "early_start_entries": entries,
        "early_start_func_configs": configs,
        "early_start_states": states,
        "evidence": str(model),
    }


def guarded_inputs(host):
    tensors, guards = [], []
    for value in host:
        storage = torch.full((value.numel() + 128,), 37, dtype=value.dtype, device="npu")
        tensor = storage[64:-64].view(value.shape)
        tensor.copy_(value)
        tensors.append(tensor)
        guards.append(storage)
    return tuple(tensors), guards


def benchmark(run, inputs, samples, calls):
    for _ in range(10):
        run(*inputs)
    torch.npu.synchronize()
    durations = []
    for _ in range(samples):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        for _ in range(calls):
            run(*inputs)
        end.record()
        end.synchronize()
        durations.append(start.elapsed_time(end) * 1000 / calls)
    return {
        "metric": "device event window per compiled-chain call, includes input handling and launch gaps",
        "samples": samples,
        "calls_per_sample": calls,
        "min_us": min(durations),
        "mean_us": statistics.mean(durations),
        "max_us": max(durations),
        "samples_us": durations,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("eager", "graph", "superkernel"), required=True)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--calls", type=int, default=8)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--neighbours", choices=("ascendc", "torch"), default="ascendc")
    parser.add_argument("--early-start", type=int, choices=(0, 1), default=0)
    args = parser.parse_args()
    if args.early_start and args.mode != "superkernel":
        parser.error("--early-start=1 requires --mode=superkernel")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.npu.set_device(0)
    torch.ops.load_library(str(args.binding.resolve()))
    # Addresses and storage offsets stay fixed; replay changes values in place.
    # Cloning views would change their offsets after static-kernel specialization.
    options = {"static_kernel_compile": True, "super_kernel_optimize": args.mode == "superkernel", "clone_input": False}
    if args.mode == "superkernel":
        options["super_kernel_optimize_options"] = {"early_start": args.early_start}
    model = HcPreChain(mark_scope=args.mode == "superkernel", neighbours=args.neighbours).eval()
    run = (
        model
        if args.mode == "eager"
        else torch.compile(model, backend="npugraph_ex", options=options, dynamic=False, fullgraph=True)
    )
    report = {
        "mode": args.mode,
        "tokens": TOKENS,
        "neighbours": args.neighbours,
        "chain": "Add(x, addend) -> torch.ops.bytenn_pto_ops.hc_pre_pto_bnn -> Mul(mixed, multiplier)",
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "device": str(torch.npu.get_device_properties(0)),
        "options": options if args.mode != "eager" else {},
        "binding": str(args.binding.resolve()),
        "validation": [],
    }
    try:
        host = make_inputs(20261015)
        inputs, guards = guarded_inputs(host)
        with torch.inference_mode():
            for replay, seed in enumerate((20261015, 20261016)):
                if replay:
                    host = make_inputs(seed)
                    for device_input, host_input in zip(inputs, host):
                        device_input.copy_(host_input)
                expected = cpu_reference(host)
                begin = time.perf_counter()
                outputs = run(*inputs)
                torch.npu.synchronize()
                elapsed = time.perf_counter() - begin
                checked = compare(outputs, expected)
                inputs_preserved = all(torch.equal(t.cpu(), h) for t, h in zip(inputs, host))
                guards_preserved = all(bool((torch.cat((g[:64], g[-64:])).cpu() == 37).all()) for g in guards)
                record = {
                    "replay": replay,
                    "seed": seed,
                    "outputs": checked,
                    "inputs_preserved": inputs_preserved,
                    "guards_preserved": guards_preserved,
                    "compile_or_call_seconds": elapsed,
                }
                report["validation"].append(record)
                (args.output / "progress.json").write_text(json.dumps(report, indent=2) + "\n")
                assert all(v["passed"] and v["finite"] for v in checked.values()), record
                assert inputs_preserved, record
                assert guards_preserved, record
                torch.save(
                    {name: t.cpu() for name, t in zip(OUTPUT_NAMES, outputs)}, args.output / f"replay-{replay}.pt"
                )
            if args.mode == "superkernel":
                report["fusion"] = audit_fusion(args.early_start)
            report["timing"] = benchmark(run, inputs, args.samples, args.calls)
            if args.profile:
                with torch_npu.profiler.profile(
                    activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
                    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(args.output / "profile")),
                ) as prof:
                    for _ in range(args.samples):
                        for _ in range(args.calls):
                            run(*inputs)
                        torch.npu.synchronize()
                        prof.step()
        report["numerical_passed"] = True
    except Exception:
        report["error"] = traceback.format_exc()
        raise
    finally:
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
