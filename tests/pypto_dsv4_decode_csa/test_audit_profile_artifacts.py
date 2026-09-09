# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Host-only tests of timestamp preservation and profiling evidence checks."""

import copy
from decimal import Decimal

import pytest

from .audit_profile_artifacts import percentile, split_device_traces, union_duration


@pytest.mark.parametrize("values,q,expected", [([1, 2, 3, 4], 0.5, 2.5), ([1, 100], 0.99, 99.01), ([7], 0.9, 7)])
def test_percentile_keeps_outliers(values, q, expected):
    assert percentile(values, q) == pytest.approx(expected)


@pytest.mark.parametrize("intervals,expected", [([], 0), ([(0, 4), (2, 3), (3, 7)], 7), ([(2, 4), (6, 9)], 5)])
def test_union_preserves_gaps(intervals, expected):
    assert union_duration([(Decimal(a), Decimal(b)) for a, b in intervals]) == expected


def trace_fixture():
    origin = Decimal("1788832570591299.960")
    result = [{"ph": "M", "pid": 4, "name": "process_name", "args": {"name": "Ascend Hardware"}}]
    for backend, model, offset in (("native", 49, 0), ("pypto", 48, 20)):
        result.append({"ph": "X", "pid": 1, "name": f"csa_steady_state.{backend}.enqueue", "ts": "0", "dur": 1})
        for name, start, duration in (
            ("MODEL_EXECUTE", 0, 1),
            ("MODEL_WAIT_COMPLETE", 1, 9),
            ("aicore_kernel_0" if backend == "pypto" else "MatMul", 2, 7),
        ):
            result.append(
                {
                    "ph": "X",
                    "pid": 4,
                    "tid": model,
                    "name": name,
                    "ts": str(origin + offset + start),
                    "dur": duration,
                    "args": {"Model Id": model},
                }
            )
    return result


def test_split_uses_device_model_not_host_time():
    events = trace_fixture()
    before = copy.deepcopy(events)
    split = split_device_traces(events, expected_replays=1)
    assert split["native"]["model_id"] == 49
    assert split["pypto"]["model_id"] == 48
    assert split["native"]["windows"][0][1] - split["native"]["windows"][0][0] == Decimal(10)
    assert events == before
    assert all(event in events for group in split.values() for event in group["trace"])


def test_split_rejects_missing_replay():
    with pytest.raises(AssertionError):
        split_device_traces(trace_fixture(), expected_replays=2)


def test_split_rejects_duplicate_device_record():
    events = trace_fixture()
    events.append(copy.deepcopy(events[-1]))
    with pytest.raises(AssertionError):
        split_device_traces(events, expected_replays=1)


def test_split_rejects_truncated_device_window():
    events = trace_fixture()
    events[-1]["dur"] = 20
    with pytest.raises(AssertionError):
        split_device_traces(events, expected_replays=1)
