# SPDX-License-Identifier: Apache-2.0
"""CPU诊断：16rank小张量Gloo协调在迟到输入下的完成时刻偏差，不测模型性能。"""

import argparse
import datetime
import json
import os
import statistics
import time
from pathlib import Path


def worker(rank, output, address, rounds):
    import torch
    import torch.distributed as dist

    # 使用已观测到的A3服务CPU分区，仅影响本诊断进程，不修改机器配置。
    os.sched_setaffinity(0, range(rank * 20 + 2, rank * 20 + 18))
    torch.set_num_threads(10)
    dist.init_process_group("gloo", init_method=address, rank=rank, world_size=16,
                            timeout=datetime.timedelta(seconds=60))
    results = []
    for iteration in range(rounds):
        for mode in ("blocking", "poll") if iteration % 2 == 0 else ("poll", "blocking"):
            dist.barrier()  # 仅CPU诊断的起点，不进入模型或候选实现。
            # 一半场景无迟到，一半模拟已观测到的rank8晚约60ms进入协调。
            delay = 0.060 if iteration % 2 and rank == 8 else 0.0
            if delay:
                time.sleep(delay)
            packed = torch.zeros((2, 16), dtype=torch.int32)
            packed[0, rank] = 96
            packed[1, rank] = 2
            begin, cpu = time.monotonic_ns(), time.thread_time_ns()
            if mode == "blocking":
                dist.all_reduce(packed)
            else:
                work = dist.all_reduce(packed, async_op=True)
                deadline = time.monotonic() + 65
                while not work.is_completed():
                    if time.monotonic() > deadline:
                        raise TimeoutError("Gloo诊断未完成")
                work.wait()  # 保留异步错误传播；不把is_completed当成功。
            finish, cpu_end = time.monotonic_ns(), time.thread_time_ns()
            if not bool((packed[0] == 96).all() and (packed[1] == 2).all()):
                raise AssertionError("Gloo结果错误")
            results.append({"iteration": iteration, "mode": mode, "late_rank": iteration % 2 == 1,
                            "begin_ns": begin, "end_ns": finish, "cpu_ms": (cpu_end - cpu) / 1e6})
    dist.destroy_process_group()
    (Path(output) / f"rank{rank}.json").write_text(json.dumps(results, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--address", default="tcp://192.168.0.106:30997")
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ["GLOO_SOCKET_IFNAME"] = "enp23s0f3"
    import torch.multiprocessing as mp

    mp.spawn(worker, args=(str(args.output), args.address, args.rounds), nprocs=16, join=True)
    rows = [json.loads((args.output / f"rank{rank}.json").read_text()) for rank in range(16)]
    summary = {}
    for mode in ("blocking", "poll"):
        for late in (False, True):
            spreads, cpu = [], []
            for iteration in range(args.rounds):
                selected = [row for rank in rows for row in rank
                            if row["iteration"] == iteration and row["mode"] == mode and row["late_rank"] == late]
                if not selected:
                    continue
                spreads.append((max(x["end_ns"] for x in selected) - min(x["end_ns"] for x in selected)) / 1e6)
                cpu.extend(x["cpu_ms"] for x in selected)
            summary[f"{mode}_late{int(late)}"] = {
                "return_spread_ms": {"min": min(spreads), "mean": statistics.mean(spreads), "max": max(spreads)},
                "thread_cpu_mean_ms": statistics.mean(cpu), "spread_samples_ms": spreads,
            }
    report = {"scope": "纯CPU Gloo诊断，不能外推模型收益；poll是诊断候选，没有接入生产。", "results": summary}
    (args.output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
