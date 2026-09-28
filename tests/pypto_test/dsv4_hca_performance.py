# SPDX-License-Identifier: Apache-2.0
"""同卡、同初态交替测量 Native/HCA 服务入口的图重放设备区间。"""

import math
import statistics

from dsv4_csa_validation import compare_tensor


def measure_graph_pair(fixture, output, runs, references, collect, *, iters, warmup, profile_dir=None):
    import torch
    from dsv4_csa_single_layer import guard_checks

    initial = {
        name: group["initial"].to(group["allocation"].device)
        for name, group in fixture["groups"].items()
    }

    def reset():
        for name, group in fixture["groups"].items():
            group["allocation"].copy_(initial[name])
        output.fill_(float("nan"))

    graphs, events = {}, {}
    for name, run in runs.items():
        reset()
        run()
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            run()
        graphs[name] = graph
        events[name] = tuple(torch.npu.Event(enable_timing=True) for _ in range(2))
        for event in events[name]:
            event.record()
    torch.npu.synchronize()

    samples = {name: [] for name in runs}
    timestamps = {name: [] for name in runs}
    order = tuple(runs)
    for iteration in range(warmup + iters):
        # 轮流使用 A/B 和 B/A 顺序，恢复初态的设备拷贝位于开始事件之前。
        for name in (order if iteration % 2 == 0 else order[::-1]):
            reset()
            start, end = events[name]
            # 当前 torch_npu 的图内事件不会随 replay 更新时间，事件必须在图外。
            start.record()
            graphs[name].replay()
            end.record()
            torch.npu.synchronize()
            stamp = start.recorded_time()
            if timestamps[name] and stamp <= timestamps[name][-1]:
                raise ValueError(f"{name} 图外事件未更新时间戳")
            timestamps[name].append(stamp)
            elapsed = start.elapsed_time(end) * 1000
            if not math.isfinite(elapsed) or elapsed <= 0:
                raise ValueError(f"{name} 设备计时无效：{elapsed}")
            if iteration >= warmup:
                samples[name].append(elapsed)

    result = {
        "scope": "mHC pre + input norm + HCA attention + mHC post；包含本层 compact metadata 生成",
        "method": "图外 NPU Event；相同初态交替 A/B、B/A；无 profiler；恢复与编译不计时",
        "iters": iters, "warmup": warmup, "device": torch.npu.get_device_name(0),
        "status": "MEASURED",
    }
    for name in runs:
        # 独立重放复核完整输出及 allocation，验证不进入计时区间。
        reset()
        graphs[name].replay()
        torch.npu.synchronize()
        state = collect()
        checks = {key: compare_tensor(value, references[name][key], 0, 0) for key, value in state.items()}
        guards = guard_checks(fixture)
        ordered = sorted(samples[name])
        result[name] = {
            "samples_us": samples[name], "start_timestamps_raw": timestamps[name][warmup:],
            "us_mean": statistics.mean(samples[name]), "us_p50": statistics.median(samples[name]),
            "us_p95": ordered[math.ceil(0.95 * iters) - 1], "us_min": ordered[0], "us_max": ordered[-1],
            "eager_comparison": checks, "guards": guards, "exact_comparison_required": name == "pto",
        }
        if any(value["status"] != "PASS" for value in guards.values()):
            raise ValueError(f"{name} 计时图的保护区或 metadata 检查失败")
        if any(value.get("nonfinite") != 0 for value in checks.values()):
            raise ValueError(f"{name} 计时图出现非有限值或不完整输出")
        if name == "pto" and any(value["status"] != "PASS" for value in checks.values()):
            raise ValueError("PTO 固定归约计时图与 eager 输出或状态不一致")
    result["speedup_p50"] = result["native"]["us_p50"] / result["pto"]["us_p50"]
    result["pto_latency_change_pct"] = (result["pto"]["us_p50"] / result["native"]["us_p50"] - 1) * 100
    if profile_dir is not None:
        import torch_npu

        for name, graph in graphs.items():
            reset()
            torch.npu.synchronize()
            with torch_npu.profiler.profile(
                activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
                schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=1),
                record_shapes=False, profile_memory=False, with_stack=False, with_modules=False,
                experimental_config=torch_npu.profiler._ExperimentalConfig(
                    profiler_level=torch_npu.profiler.ProfilerLevel.Level1),
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(profile_dir / name)),
            ) as trace:
                graph.replay()
                torch.npu.synchronize()
                trace.step()
        result["profile_directory"] = str(profile_dir)
    return result


def capture_swimlane(fixture, call, directory, cold_l2=False, before=None):
    """每窗口一次真实根调用，采图重放核内信息；不作为正式性能数据。

    cold_l2 为真时，每个窗口前写一块 512 MiB 缓冲冲刷 L2，模拟正式计时里
    Native/PTO 交替重放造成的权重冷读取；缓冲与被测张量无关，不改变数值。
    before 不为空时，每个窗口前先执行一次 Native（其后恢复初态），复现正式计时里
    “Native 刚跑完再跑 PTO”的 L2 状态。
    """
    import pypto.torch
    import torch
    from dsv4_csa_single_card_bench import _export_swimlane
    from dsv4_csa_single_layer import restore

    restore(fixture)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        call()
    for _ in range(5):
        restore(fixture)
        graph.replay()
    torch.npu.synchronize()
    flush = torch.empty(128 * 1024 * 1024, dtype=torch.float32, device="npu") if cold_l2 else None
    windows = []
    for window in range(2):
        if before is not None:
            restore(fixture)
            before()
        restore(fixture)
        if flush is not None:
            flush.fill_(float(window))
        torch.npu.synchronize()
        pypto.torch.begin_dfx()
        try:
            graph.replay()
        finally:
            pypto.torch.end_dfx()
        target = directory if window == 0 else directory / f"window_{window}"
        exported = _export_swimlane(target, kernel_pattern="_jit__decode_hca_tp1_layer_*/kernel_config.py")
        if not exported["exported"]:
            raise RuntimeError(f"HCA 泳道导出失败：{exported}")
        windows.append({**exported, "window": window, "cold_l2": cold_l2, "after_native": before is not None,
                        "scope": "单卡 HCA 整层根图重放；复用 compact metadata"})
    return windows
