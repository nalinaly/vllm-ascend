# SPDX-License-Identifier: Apache-2.0
"""用重叠核区间和长 AICPU 包络验证 HC_pre 性能统计边界。"""

import json

import pytest
from dsv4_hc_pre_timing import interval_metrics, summarize_native, summarize_swimlane


def test_parallel_tasks_exclude_setup_and_aicpu_envelope(tmp_path):
    path = tmp_path / "trace.json"
    events = [
        {"ph": "M", "name": "process_name", "pid": 4, "args": {"name": "Worker View"}},
        {"ph": "M", "name": "process_name", "pid": 2, "args": {"name": "AICPU Scheduler"}},
        {"ph": "X", "pid": 2, "ts": 0, "dur": 10000, "name": "dispatch"},
    ]
    # 两个重叠任务 [12,20]、[15,25]，空闲 5us，再执行 [30,34]。
    for epoch, offset in ((7, 0), (8, 100)):
        for index, (receive, setup, kernel) in enumerate(((10, 2, 8), (14, 1, 10), (29, 1, 4))):
            events.append({"ph": "X", "pid": 4, "name": f"task{index}_spmd(t{index})",
                           "ts": receive + offset, "dur": setup + kernel,
                           "args": {"launch_epoch": epoch, "local_setup_us": setup,
                                    "kernel-duration-us": kernel}})
    path.write_text(json.dumps({"traceEvents": events}))
    result = summarize_swimlane(path, 2)
    assert result["span_p50_us"] == 22
    assert result["active_union_p50_us"] == 17
    assert result["no_incore_gap_p50_us"] == 5
    assert [row["launch_epoch"] for row in result["samples"]] == [7, 8]
    assert result["samples"][0]["first_start_us"] == 12
    assert result["samples"][0]["last_end_us"] == 34
    with pytest.raises(ValueError, match="需要 3 次"):
        summarize_swimlane(path, 3)
    events[-1]["args"].pop("launch_epoch")
    path.write_text(json.dumps({"traceEvents": events}))
    with pytest.raises(ValueError, match="launch_epoch"):
        summarize_swimlane(path, 2)


@pytest.mark.parametrize("intervals", [[], [(1, 1)], [(2, 1)], [(float("nan"), 2)]])
def test_invalid_intervals_fail(intervals):
    with pytest.raises(ValueError):
        interval_metrics(intervals)


def test_native_uses_device_hcpre_only(tmp_path):
    path = tmp_path / "native.json"
    path.write_text(json.dumps({"traceEvents": [
        {"name": "ProfilerStep#0", "ph": "X", "ts": 0, "dur": 1000},
        {"name": "HcPre", "ph": "X", "ts": 10, "dur": 8},
        {"name": "HcPre", "ph": "X", "ts": 30, "dur": 12},
        {"name": "HcPre", "ph": "X", "ts": 50, "dur": 10},
    ]}))
    result = summarize_native(path, 3)
    assert result["samples_us"] == [8, 12, 10]
    assert result["p50_us"] == 10
