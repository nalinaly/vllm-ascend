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


def device_interval(profile_dir, replays=1):
    """从 profiler 的 kernel_details.csv 算出每次重放的**纯设备**耗时，返回中位数。

    验收口径只算纯 device 耗时：图外 NPU Event 的区间里含每次迭代的主机派发
    （`acl_graph_replay`、`TorchDynamo Cache Lookup`、`npu_fx_compiler inference` 等），
    那些在生产里被 FULL_DECODE_ONLY 的整步图重放摊掉，不该计入。

    返回的 `span_us` 是第一个 kernel 开始到最后一个 kernel 结束，**这是可跨侧比较的量**。
    `busy_us` 只作诊断：PTO 侧的 AICPU 调度器 kernel 与 AICore kernel 并发，
    忙时相加没有意义。
    """
    import csv
    import glob
    import os

    hits = glob.glob(os.path.join(str(profile_dir), "*", "ASCEND_PROFILER_OUTPUT", "kernel_details.csv"))
    if not hits:
        return {"available": False, "reason": "未找到 kernel_details.csv"}
    rows = list(csv.DictReader(open(hits[0])))
    if not rows:
        return {"available": False, "reason": "kernel_details.csv 为空"}
    rows.sort(key=lambda r: float(r["Start Time(us)"]))
    start = [float(r["Start Time(us)"]) for r in rows]
    dur = [float(r["Duration(us)"]) for r in rows]
    # 多次重放连在一起采：重放之间有主机派发（约 250 μs），是全局最大的几个间隔，
    # 用它们切段。单次重放的 span 方差很大（实测同一配置两次相差 12 μs，
    # 本不该受影响的档位摆了 44 μs），必须取多次的中位数才能分辨 20 μs 量级的改动。
    gaps = sorted(((start[i + 1] - (start[i] + dur[i]), i) for i in range(len(rows) - 1)),
                  reverse=True)[:max(0, replays - 1)]
    cuts = sorted(i for _, i in gaps)
    spans, bounds, lo = [], [], 0
    for hi in cuts + [len(rows) - 1]:
        seg = range(lo, hi + 1)
        spans.append(max(start[i] + dur[i] for i in seg) - min(start[i] for i in seg))
        bounds.append((lo, hi))
        lo = hi + 1
    # 开了 aic_metrics 之后 csv 会多出 pipe 占比 / L2 命中率之类的列。这些列按 kernel
    # 逐行给出，聚合时按耗时加权平均才有意义（长 kernel 的占比更能代表整体）。
    base = {"Name", "Type", "Accelerator Core", "Start Time(us)", "Duration(us)", "Wait Time(us)",
            "Block Dim", "Mix Block Dim", "HF32 Eligible", "Input Shapes", "Input Data Types",
            "Input Formats", "Output Shapes", "Output Data Types", "Output Formats", "Context ID",
            "aic_*", "Step Id"}
    extra = [k for k in rows[0] if k and k not in base and k not in ("Model ID", "Task ID", "Stream ID")]
    by_type = {}
    for r in rows:
        slot = by_type.setdefault(r["Name"], {"count": 0, "total_us": 0.0, "metrics": {}})
        slot["count"] += 1
        d = float(r["Duration(us)"])
        slot["total_us"] += d
        for k in extra:
            try:
                value = float(r[k])
            except (TypeError, ValueError):
                continue
            acc = slot["metrics"].setdefault(k, [0.0, 0.0])
            acc[0] += value * d
            acc[1] += d
    for slot in by_type.values():
        slot["metrics"] = {k: round(v[0] / v[1], 6) for k, v in slot["metrics"].items() if v[1] > 0}
        slot["count"] /= max(1, len(spans))
        slot["total_us"] /= max(1, len(spans))
    return {
        "available": True,
        "kernels": len(rows),
        "replays": len(spans),
        "kernels_per_replay": [hi - lo + 1 for lo, hi in bounds],
        "span_us": statistics.median(spans),
        "span_us_all": spans,
        "span_us_min": min(spans),
        "span_us_stdev": (statistics.stdev(spans) if len(spans) > 1 else 0.0),
        "busy_us": sum(dur) / max(1, len(spans)),
        "metric_columns": extra,
        "top": sorted(({"name": k, **v} for k, v in by_type.items()),
                      key=lambda x: -x["total_us"])[:12],
        "note": "span_us 为纯设备跨度，可跨侧比较；busy_us 仅诊断（PTO 的 AICPU/AICore 并发）",
    }


def collect_state(fixture, output):
    """HCA 的状态快照：输出，加三组 allocation 的每个 view。"""
    state = {"x_out": output.detach().cpu()}
    for name, group in fixture["groups"].items():
        for index, view in enumerate(group["views"]):
            state[f"{name}.{index}"] = view.detach().cpu()
    return state


def measure_graph_interval(fixture, run, output, reference, *, iters, warmup, require_exact,
                           profile_dir=None, aic_metrics=None, profile_replays=1):
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
                profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
                # aic_metrics 选 PipeUtilization 时，kernel_details.csv 会多出
                # mte1/mte2/mte3/cube/vec/scalar 的 ratio 列，可判定每个 kernel
                # 到底受哪条 pipe 限制；选 L2Cache 时给 L2 命中率。
                **({"aic_metrics": aic_metrics} if aic_metrics is not None else {})),
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(profile_dir)),
        ) as trace:
            start.record()
            for _ in range(profile_replays):
                run()
                # 每次重放之间同步，主机派发会留下一个明显的间隔，供切段用。
                torch.npu.synchronize()
            end.record()
            torch.npu.synchronize()
            trace.step()
        profile = {"directory": str(profile_dir), "replays": profile_replays,
                   "event_envelope_us": start.elapsed_time(end) * 1000,
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
