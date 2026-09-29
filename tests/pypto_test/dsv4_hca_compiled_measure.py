# SPDX-License-Identifier: Apache-2.0
"""对已由 npugraph_ex 编译的整层半边做设备计时。

移植自 CSA 会话的 coefficients_seven_experiment/native_measure.py，两处按 HCA 调整：

1. 去掉 Top-K 校验。HCA 没有 indexer，一次调用的可观测结果只有输出 x_out
   与三组（compressed / swa / state）cache 的各个 view。
2. **不再自己嵌套一层外部 NPUGraph**。npugraph_ex 自己拥有 capture/replay，
   外面再套一层 `torch.npu.graph` 会把它的重放包进另一张图，量到的不是同一个东西。
   计时区间仍然是图外 NPU Event，初态恢复与输出毒化都在区间之外。
"""

import math
import statistics

from dsv4_csa_single_layer import guard_checks
from dsv4_csa_validation import compare_tensor


def collect_state(fixture, output):
    """HCA 的状态快照：输出，加三组 allocation 的每个 view。"""
    state = {"x_out": output.detach().cpu()}
    for name, group in fixture["groups"].items():
        for index, view in enumerate(group["views"]):
            state[f"{name}.{index}"] = view.detach().cpu()
    return state


def measure_graph_interval(fixture, run, output, reference, *, iters, warmup, require_exact, profile_dir=None):
    """图外事件包住一次整层重放；初态恢复与输出毒化均在计时区间外。"""
    import torch

    initial = {
        name: group["initial"].to(group["allocation"].device)
        for name, group in fixture["groups"].items()
    }

    def reset():
        for name, group in fixture["groups"].items():
            group["allocation"].copy_(initial[name])
        output.fill_(float("nan"))

    start, end = (torch.npu.Event(enable_timing=True) for _ in range(2))
    # 当前 torch_npu 图内 Event 不随重放更新时间戳，必须在图外显式 record。
    start.record()
    end.record()
    reset()
    torch.npu.synchronize()
    samples, timestamps = [], []
    for iteration in range(warmup + iters):
        reset()
        start.record()
        run()
        end.record()
        torch.npu.synchronize()
        stamp = start.recorded_time()
        if timestamps and stamp <= timestamps[-1]:
            raise ValueError("图计时的开始事件时间戳未更新，不能采纳重复的旧时间")
        timestamps.append(stamp)
        if iteration >= warmup:
            elapsed = start.elapsed_time(end) * 1000
            if not math.isfinite(elapsed) or elapsed <= 0:
                raise ValueError(f"图外设备事件没有产生有效时间戳：{elapsed}")
            samples.append(elapsed)
    state = collect_state(fixture, output)
    checks = {name: compare_tensor(value, reference[name], 0, 0) for name, value in state.items()}
    guards = guard_checks(fixture)
    if any(check["status"] != "PASS" for check in guards.values()):
        raise ValueError("计时重放改写了 metadata 或 slot 外存储")
    if any(check.get("nonfinite") != 0 for check in checks.values()):
        raise ValueError("计时重放出现非有限值或不完整输出")
    if require_exact and any(check["status"] != "PASS" for check in checks.values()):
        raise ValueError("固定归约的计时重放与同初态 eager 不一致")
    profile = None
    if profile_dir is not None:
        import torch_npu

        reset()
        torch.npu.synchronize()
        with torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
            schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=1),
            record_shapes=False, profile_memory=False, with_stack=False, with_modules=False,
            experimental_config=torch_npu.profiler._ExperimentalConfig(
                profiler_level=torch_npu.profiler.ProfilerLevel.Level1),
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(profile_dir)),
        ) as trace:
            start.record()
            run()
            end.record()
            torch.npu.synchronize()
            trace.step()
        profile = {"directory": str(profile_dir), "event_envelope_us": start.elapsed_time(end) * 1000,
                   "scope": "独立一次重放，核对设备区间和热点；不混入无 profiler 的采样"}
    ordered = sorted(samples)
    return {
        "samples_us": samples,
        # recorded_time 的原始计数只用于检查时间戳确实更新；耗时单位由 elapsed_time 给出。
        "start_timestamps_raw": timestamps[warmup:],
        "us_min": ordered[0], "us_p50": statistics.median(samples),
        "us_p95": ordered[math.ceil(0.95 * len(ordered)) - 1], "us_max": ordered[-1],
        "us_mean": statistics.mean(samples),
        "eager_comparison": checks, "exact_comparison_required": require_exact,
        "guards": guards, "profile": profile,
    }
