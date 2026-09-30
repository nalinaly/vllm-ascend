# SPDX-License-Identifier: Apache-2.0
"""按真实 launch 边界绑定 CSA/HCA 的依赖图，导出同轮联合泳道。"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from dsv4_csa_env import activate, write_json

    activate()
    from simpler_setup.tools import swimlane_converter as conv

    report = json.loads((args.input / "report.json").read_text())
    if report["status"] != "PASS":
        raise ValueError("只能导出完成联合状态检查的采集")
    records = Path(report["joint_dfx_directory"]) / "chip_swimlane_records.json"
    data = conv.read_perf_data(records)
    boundaries = data.get("run_boundaries", [])
    if len(boundaries) != 6:
        raise ValueError(f"三步 CSA→HCA 应有六个 launch，实际 {len(boundaries)}")
    events, ids, metadata_keys, launches = [], {}, set(), []
    tables = {}
    for op in ("csa", "hca"):
        configs = list((args.input / "build").glob(f"_decode_{op}_tp1_layer_*/kernel_config.py"))
        if len(configs) != 1:
            raise ValueError(f"{op} 的编译配置不是唯一：{configs}")
        deps = args.input / f"{op}_deps.json"
        tables[op] = (
            conv.load_kernel_config(configs[0]),
            conv.load_deps_kernel_map(deps),
            conv.load_deps_block_map(deps),
            conv.load_deps_json(deps),
        )
        if any(value is None for value in tables[op]):
            raise ValueError(f"{op} 的依赖或名称信息不完整")
    for index, boundary in enumerate(boundaries):
        epoch = boundary["launch_epoch"]
        op = "csa" if index % 2 == 0 else "hca"
        names, kernel_map, blocks, edges = tables[op]
        tasks = [t for t in data["tasks"] if t.get("launch_epoch") == epoch]
        if not tasks:
            raise ValueError(f"{op} launch {epoch} 无任务")
        unknown = {t["task_id"] for t in tasks if t["task_id"] not in kernel_map}
        if unknown:
            raise ValueError(f"{op} launch {epoch} 与依赖图不符：{sorted(unknown)[:5]}")
        expected = Counter()
        for task_id, kernel_ids in kernel_map.items():
            expected[task_id, "aic"] = blocks[task_id] * int(kernel_ids[0] >= 0)
            expected[task_id, "aiv"] = blocks[task_id] * sum(k >= 0 for k in kernel_ids[1:])
        actual = Counter((t["task_id"], t["core_type"]) for t in tasks)
        if actual != expected:
            raise ValueError(f"{op} launch {epoch} 的每任务 AIC/AIV 记录数与 block_num 不符")

        def phases(key, launch_epoch=epoch):
            return [[r for r in rows if r.get("launch_epoch") == launch_epoch] for rows in data.get(key, [])]

        trace = conv.generate_chrome_trace_json(
            tasks,
            None,
            names,
            scheduler_phases=phases("aicpu_scheduler_phases"),
            scheduler_streams=data.get("scheduler_streams"),
            orchestrator_phases=phases("aicpu_orchestrator_phases"),
            core_to_thread=data.get("core_to_thread"),
            orchestrator_name=op.upper(),
            orchestrator_source="host",
            timeline_metadata=data.get("timeline_metadata"),
            deps_edges=edges,
            deps_kernel_map=kernel_map,
            deps_block_map=blocks,
            aicpu_lifecycle_records=[
                r for r in data.get("aicpu_lifecycle_records", []) if r.get("launch_epoch") == epoch
            ],
        )
        conv._remap_trace_ids(trace["traceEvents"], epoch, ids)
        for event in trace["traceEvents"]:
            if event.get("ph") == "M":
                key = json.dumps(event, sort_keys=True)
                if key in metadata_keys:
                    continue
                metadata_keys.add(key)
            else:
                event.setdefault("args", {}).update(operator=op.upper(), launch_epoch=epoch, joint_step=index // 2 + 1)
            events.append(event)
        first = min(t["start_time_us"] for t in tasks)
        last = max(t["end_time_us"] for t in tasks)
        launch = {
            "operator": op.upper(),
            "step": index // 2 + 1,
            "epoch": epoch,
            "scheduler_init_boundary_us": boundary["start_time_us"],
            "first_incore_us": first,
            "last_incore_us": last,
            "incore_envelope_us": last - first,
            "recorded_start_to_first_incore_us": first - boundary["start_time_us"],
            "subtask_records": len(tasks),
        }
        launches.append(launch)
        events.append(
            {
                "ph": "X",
                "pid": 7,
                "tid": 0,
                "cat": "kernel_launch",
                "name": f"Step {index // 2 + 1} {op.upper()} scheduler+incore",
                "ts": boundary["start_time_us"],
                "dur": last - boundary["start_time_us"],
                "args": {
                    **launch,
                    "start_basis": "scheduler init; excludes packet validation/restore",
                    "end_basis": "last_incore_record; excludes exit tail",
                },
            }
        )
    events.append(
        {"ph": "M", "pid": 7, "tid": 0, "name": "process_name", "args": {"name": "CSA to HCA joint launches"}}
    )
    write_json(
        args.input / "joint_swimlane.json",
        {
            "traceEvents": events,
            "metadata": {
                **(data.get("timeline_metadata") or {}),
                "run_boundaries": boundaries,
                "scope": "same DFX window as three-step PyTorch profile; per-launch dependency maps",
            },
        },
    )
    write_json(args.input / "launch_summary.json", launches)
    print(json.dumps(launches, indent=2))


if __name__ == "__main__":
    main()
