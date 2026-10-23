# SPDX-License-Identifier: Apache-2.0
"""T96 eager/graph correctness and optional Native device profiling."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch_npu

from check_outputs import reference


def compare(actual, expected, require_pass=True):
    report = {}
    for name, a, b in zip(("mixed", "post", "comb"), actual, expected):
        a, b = a.detach().cpu().float(), b.detach().cpu().float().reshape(a.shape)
        atol, rtol = (0.01, 0.01) if name == "mixed" else (1e-5, 1e-4)
        ok = torch.allclose(a, b, atol=atol, rtol=rtol)
        report[name] = {"passed": bool(ok), "max_abs": (a-b).abs().max().item()}
        if not ok and require_pass:
            raise AssertionError(report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--native-library", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.npu.set_device(0)
    torch.ops.load_library(str(args.binding.resolve()))
    specs = [("x", (96,4,4096), True), ("weight", (24,16384), False),
             ("scale", (3,), False), ("base", (24,), False)]
    host = []
    for name, shape, bf16 in specs:
        t = torch.from_numpy(np.fromfile(args.input/f"{name}.bin", dtype=np.uint16 if bf16 else np.float32))
        host.append((t.view(torch.bfloat16) if bf16 else t).reshape(shape))
    inputs = [t.npu() for t in host]
    pto = lambda: torch.ops.bytenn_pto_ops.hc_pre_pto_bnn(*inputs)
    report = {"case": "B16/S6/T96", "eager": compare(pto(), reference(*host))}
    invalid = []
    for bad_x in [inputs[0][:, :, :4095], inputs[0].float(), inputs[0].transpose(1,2).contiguous().transpose(1,2)]:
        try:
            torch.ops.bytenn_pto_ops.hc_pre_pto_bnn(bad_x, *inputs[1:])
        except RuntimeError:
            invalid.append("rejected")
        else:
            raise AssertionError("Invalid input accepted")
    report["invalid_shape_dtype_stride"] = invalid
    runners = {"pto": pto}
    if args.native_library:
        torch.ops.load_library(str(args.native_library.resolve()))
        def native():
            return torch.ops.custom.npu_hc_pre(inputs[0].view(16,6,4,4096), *inputs[1:],
                hc_mult=4, hc_sinkhorn_iters=20, norm_eps=1e-6, hc_eps=1e-6)
        runners["native"] = native
        report["native_vs_cpu"] = compare(native(), [v.reshape(s) for v,s in zip(reference(*host), [(16,6,4096),(16,6,4),(16,6,4,4)])], require_pass=False)
    graphs, outputs = {}, {}
    for name, run in runners.items():
        for _ in range(5):
            run()
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            outputs[name] = run()
        graph.replay()
        torch.npu.synchronize()
        graphs[name] = graph
        report[name+"_graph"] = compare(outputs[name], reference(*host), require_pass=(name == "pto"))
    host[0].mul_(-0.5)
    inputs[0].copy_(host[0])
    for name, graph in graphs.items():
        graph.replay()
        torch.npu.synchronize()
        report[name+"_updated_input"] = compare(outputs[name], reference(*host), require_pass=(name == "pto"))
    # Restore identical original input before timing/profile.
    host[0].mul_(-2)
    inputs[0].copy_(host[0])
    report["graph_replay_us"] = {}
    for name, graph in graphs.items():
        for _ in range(10):
            graph.replay()
        torch.npu.synchronize()
        samples = []
        for _ in range(10):
            start, end = [torch.npu.Event(enable_timing=True) for _ in range(2)]
            start.record(); graph.replay(); end.record(); end.synchronize()
            samples.append(start.elapsed_time(end)*1000)
        report["graph_replay_us"][name] = {"median": float(np.median(samples)), "samples": samples}
        with torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.CPU,
                                                    torch_npu.profiler.ProfilerActivity.NPU]) as prof:
            for _ in range(3):
                graph.replay()
                torch.npu.synchronize()
                prof.step()
        prof.export_chrome_trace(str(args.output/f"{name}.trace.json"))
    (args.output/"report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
