# SPDX-License-Identifier: Apache-2.0
"""测量 PyPTO kernel 模式单次图重放的固定开销与按任务/块增长的调度开销。

与 HCA 单层计时同一方式：torch custom op 捕获进 NPUGraph，图外 NPU Event 计时。
kernel 只做极小的拷贝，用来分离运行时启动、编排与调度本身的代价，不代表任何算子。
"""

import argparse
import json
import math
import os
import statistics
from pathlib import Path

import pypto.language as pl

ROWS = 8
COLS = 64


def _floor_t1_b1_c(
    x: pl.Tensor[[ROWS, COLS], pl.FP32],
    out: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
):
    scratch = pl.create_tensor([1 * ROWS, COLS], dtype=pl.FP32)
    previous = pl.system.task_dummy(deps=[])
    for task in pl.range(1):
        with pl.spmd(1, name_hint="floor_task", deps=[previous]) as tid:
            row = (task * 1 + pl.tile.get_block_idx()) * ROWS
            scratch[row:row + ROWS, :] = pl.add(x[0:ROWS, :], 1.0)
        previous = tid
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="floor_out", deps=[previous]):
        out[0:ROWS, :] = scratch[0:ROWS, :]
    return out


floor_t1_b1_c = pl.jit(auto_scope=False)(_floor_t1_b1_c)

def _floor_t8_b1_c(
    x: pl.Tensor[[ROWS, COLS], pl.FP32],
    out: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
):
    scratch = pl.create_tensor([8 * ROWS, COLS], dtype=pl.FP32)
    previous = pl.system.task_dummy(deps=[])
    for task in pl.range(8):
        with pl.spmd(1, name_hint="floor_task", deps=[previous]) as tid:
            row = (task * 1 + pl.tile.get_block_idx()) * ROWS
            scratch[row:row + ROWS, :] = pl.add(x[0:ROWS, :], 1.0)
        previous = tid
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="floor_out", deps=[previous]):
        out[0:ROWS, :] = scratch[0:ROWS, :]
    return out


floor_t8_b1_c = pl.jit(auto_scope=False)(_floor_t8_b1_c)

def _floor_t32_b1_c(
    x: pl.Tensor[[ROWS, COLS], pl.FP32],
    out: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
):
    scratch = pl.create_tensor([32 * ROWS, COLS], dtype=pl.FP32)
    previous = pl.system.task_dummy(deps=[])
    for task in pl.range(32):
        with pl.spmd(1, name_hint="floor_task", deps=[previous]) as tid:
            row = (task * 1 + pl.tile.get_block_idx()) * ROWS
            scratch[row:row + ROWS, :] = pl.add(x[0:ROWS, :], 1.0)
        previous = tid
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="floor_out", deps=[previous]):
        out[0:ROWS, :] = scratch[0:ROWS, :]
    return out


floor_t32_b1_c = pl.jit(auto_scope=False)(_floor_t32_b1_c)

def _floor_t8_b24_c(
    x: pl.Tensor[[ROWS, COLS], pl.FP32],
    out: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
):
    scratch = pl.create_tensor([192 * ROWS, COLS], dtype=pl.FP32)
    previous = pl.system.task_dummy(deps=[])
    for task in pl.range(8):
        with pl.spmd(24, name_hint="floor_task", deps=[previous]) as tid:
            row = (task * 24 + pl.tile.get_block_idx()) * ROWS
            scratch[row:row + ROWS, :] = pl.add(x[0:ROWS, :], 1.0)
        previous = tid
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="floor_out", deps=[previous]):
        out[0:ROWS, :] = scratch[0:ROWS, :]
    return out


floor_t8_b24_c = pl.jit(auto_scope=False)(_floor_t8_b24_c)

def _floor_t32_b24_c(
    x: pl.Tensor[[ROWS, COLS], pl.FP32],
    out: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
):
    scratch = pl.create_tensor([768 * ROWS, COLS], dtype=pl.FP32)
    previous = pl.system.task_dummy(deps=[])
    for task in pl.range(32):
        with pl.spmd(24, name_hint="floor_task", deps=[previous]) as tid:
            row = (task * 24 + pl.tile.get_block_idx()) * ROWS
            scratch[row:row + ROWS, :] = pl.add(x[0:ROWS, :], 1.0)
        previous = tid
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="floor_out", deps=[previous]):
        out[0:ROWS, :] = scratch[0:ROWS, :]
    return out


floor_t32_b24_c = pl.jit(auto_scope=False)(_floor_t32_b24_c)

def _floor_t75_b8_c(
    x: pl.Tensor[[ROWS, COLS], pl.FP32],
    out: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
):
    scratch = pl.create_tensor([600 * ROWS, COLS], dtype=pl.FP32)
    previous = pl.system.task_dummy(deps=[])
    for task in pl.range(75):
        with pl.spmd(8, name_hint="floor_task", deps=[previous]) as tid:
            row = (task * 8 + pl.tile.get_block_idx()) * ROWS
            scratch[row:row + ROWS, :] = pl.add(x[0:ROWS, :], 1.0)
        previous = tid
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="floor_out", deps=[previous]):
        out[0:ROWS, :] = scratch[0:ROWS, :]
    return out


floor_t75_b8_c = pl.jit(auto_scope=False)(_floor_t75_b8_c)

def _floor_t8_b48_p(
    x: pl.Tensor[[ROWS, COLS], pl.FP32],
    out: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
):
    scratch = pl.create_tensor([384 * ROWS, COLS], dtype=pl.FP32)
    previous = pl.system.task_dummy(deps=[])
    for task in pl.range(8):
        with pl.spmd(48, name_hint="floor_task", deps=[]) as tid:
            row = (task * 48 + pl.tile.get_block_idx()) * ROWS
            scratch[row:row + ROWS, :] = pl.add(x[0:ROWS, :], 1.0)
        previous = tid
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="floor_out", deps=[previous]):
        out[0:ROWS, :] = scratch[0:ROWS, :]
    return out


floor_t8_b48_p = pl.jit(auto_scope=False)(_floor_t8_b48_p)

def _floor_t75_b8_p(
    x: pl.Tensor[[ROWS, COLS], pl.FP32],
    out: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
):
    scratch = pl.create_tensor([600 * ROWS, COLS], dtype=pl.FP32)
    previous = pl.system.task_dummy(deps=[])
    for task in pl.range(75):
        with pl.spmd(8, name_hint="floor_task", deps=[]) as tid:
            row = (task * 8 + pl.tile.get_block_idx()) * ROWS
            scratch[row:row + ROWS, :] = pl.add(x[0:ROWS, :], 1.0)
        previous = tid
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="floor_out", deps=[previous]):
        out[0:ROWS, :] = scratch[0:ROWS, :]
    return out


floor_t75_b8_p = pl.jit(auto_scope=False)(_floor_t75_b8_p)

KERNELS = [
    (1, 1, True, floor_t1_b1_c),
    (8, 1, True, floor_t8_b1_c),
    (32, 1, True, floor_t32_b1_c),
    (8, 24, True, floor_t8_b24_c),
    (32, 24, True, floor_t32_b24_c),
    (75, 8, True, floor_t75_b8_c),
    (8, 48, False, floor_t8_b48_p),
    (75, 8, False, floor_t75_b8_p),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--ring-task-window", type=int, default=0, help="0 表示沿用运行时默认")
    parser.add_argument("--ring-heap-mb", type=int, default=0, help="0 表示沿用运行时默认")
    parser.add_argument("--profile", action="store_true", help="每个用例额外采一次 profiler，查看 AICPU/AICore 核起止")
    args = parser.parse_args()
    if not os.environ.get("TASK_DEVICE"):
        raise RuntimeError("NPU 验证必须通过 task-submit 提交")
    args.output.mkdir(parents=True, exist_ok=True)
    import pypto.torch
    import torch
    import torch_npu  # noqa: F401

    torch.npu.set_device(0)
    ring = {}
    if args.ring_task_window:
        ring["ring_task_window"] = args.ring_task_window
    if args.ring_heap_mb:
        ring["ring_heap"] = args.ring_heap_mb * 1024 * 1024
    pypto.torch.init(device=0, platform="a2a3", runtime="tensormap_and_ringbuffer", **ring)
    x = torch.zeros(ROWS, COLS, dtype=torch.float32, device="npu")
    out = torch.empty_like(x)
    result = {"scope": "PyPTO kernel 模式图重放开销探针；与算子数值无关", "ring": ring, "cases": []}
    for tasks, blocks, chained, kernel in KERNELS:
        op = pypto.torch.register(kernel, f"floor_probe::t{tasks}_b{blocks}_{int(chained)}")
        op(x, out)
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            op(x, out)
        start, end = (torch.npu.Event(enable_timing=True) for _ in range(2))
        samples = []
        for i in range(args.warmup + args.iters):
            start.record()
            graph.replay()
            end.record()
            torch.npu.synchronize()
            if i >= args.warmup:
                samples.append(start.elapsed_time(end) * 1000)
        if args.profile:
            import torch_npu

            torch.npu.synchronize()
            with torch_npu.profiler.profile(
                activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
                schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=3, repeat=1),
                experimental_config=torch_npu.profiler._ExperimentalConfig(
                    profiler_level=torch_npu.profiler.ProfilerLevel.Level1),
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                    str(args.output / "profile" / f"t{tasks}_b{blocks}_{int(chained)}")),
            ) as trace:
                for _ in range(3):
                    start.record()
                    graph.replay()
                    end.record()
                    torch.npu.synchronize()
                    trace.step()
        ordered = sorted(samples)
        row = {"tasks": tasks, "blocks_per_task": blocks, "chained": chained,
               "total_blocks": tasks * blocks + 1, "us_p50": statistics.median(samples),
               "us_p95": ordered[math.ceil(0.95 * len(ordered)) - 1], "us_min": ordered[0],
               "correct": bool(torch.equal(out.cpu(), torch.ones(ROWS, COLS)))}
        result["cases"].append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    (args.output / "report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
