# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Host contracts for the stateful multi-bucket ACLGraph runner."""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass

import pytest

from tests.pypto_dsv4_decode_csa.a3_stateful_aclgraph_compare import (
    GRAPH_MODEL,
    A3StatefulACLGraphResult,
    BucketGraphReplayPlan,
    StatefulACLGraphBucketEvidence,
    _validate_run_arguments,
    build_a3_stateful_aclgraph_plan,
    build_parser,
    run_a3_stateful_aclgraph_compare,
)
from tests.pypto_dsv4_decode_csa.metadata import (
    DecodeStepMetadataBufferOwner,
    StagedDecodeStepMetadataPayload,
)
from tests.pypto_dsv4_decode_csa.trace import BATCH_BUCKETS


@pytest.mark.parametrize("steps", (4, 32, 128))
def test_plan_routes_every_churn_step_to_exactly_one_static_graph(steps: int) -> None:
    plan = build_a3_stateful_aclgraph_plan(steps=steps, seed=73)

    assert plan.buckets == tuple(BATCH_BUCKETS) == (4, 8, 12, 16)
    assert plan.graph_capture_count == 4
    assert len(plan.replay_bucket_order) == steps
    assert plan.replay_bucket_order == tuple(payload.spec.batch for payload in plan.churn.payloads)
    routed = [step_id for graph in plan.bucket_graphs for step_id in graph.replay_step_ids]
    assert sorted(routed) == list(range(steps))
    assert len(routed) == len(set(routed))
    for graph in plan.bucket_graphs:
        assert graph.capture_step_id == graph.replay_step_ids[0]
        assert graph.capture_actual == graph.replay_actuals[0]
        assert graph.metadata_update_count == len(graph.replay_step_ids)
        assert all(actual < graph.bucket for actual in graph.replay_actuals)


def test_default_trace_reuses_each_graph_with_multiple_metadata_updates() -> None:
    plan = build_a3_stateful_aclgraph_plan(steps=32, seed=73)

    assert all(graph.metadata_update_count > 1 for graph in plan.bucket_graphs)
    assert len(set(plan.replay_bucket_order)) == 4
    assert any(
        left != right for left, right in zip(plan.replay_bucket_order, plan.replay_bucket_order[1:], strict=False)
    )
    serialized = plan.to_dict()
    assert serialized["graph_model"] == GRAPH_MODEL
    assert serialized["graph_capture_count"] == 4
    assert json.loads(json.dumps(serialized))["requested_steps"] == 32


def test_cpu_metadata_owners_keep_one_address_family_per_bucket_across_trace() -> None:
    plan = build_a3_stateful_aclgraph_plan(steps=32, seed=73)
    staged = tuple(StagedDecodeStepMetadataPayload.from_host(payload) for payload in plan.churn.payloads)
    owners = {
        graph.bucket: DecodeStepMetadataBufferOwner.create(staged[graph.capture_step_id])
        for graph in plan.bucket_graphs
    }
    addresses = {bucket: dict(owner.device_metadata_addresses()) for bucket, owner in owners.items()}

    for payload in staged:
        owner = owners[payload.spec.batch]
        owner.apply_staged_payload(payload)
        assert dict(owner.device_metadata_addresses()) == addresses[payload.spec.batch]
        assert owner.current_step_id == payload.step_id
        assert owner.current_actual == payload.actual

    for graph in plan.bucket_graphs:
        owner = owners[graph.bucket]
        assert owner.pending_staging_owner_count == graph.metadata_update_count + 1
        owner.release_staging_owners_after_sync()
        assert owner.pending_staging_owner_count == 0


@pytest.mark.parametrize(
    "kwargs, message",
    (
        ({"runtime": "bad"}, "runtime"),
        ({"device": -1}, "device"),
        ({"device": True}, "device"),
        ({"steps": 5}, "4, 32, or 128"),
        ({"atol": -0.1}, "non-negative"),
        ({"rtol": -0.1}, "non-negative"),
    ),
)
def test_run_arguments_fail_before_device_bootstrap(kwargs: dict[str, object], message: str) -> None:
    arguments: dict[str, object] = {
        "runtime": "tensormap_and_ringbuffer",
        "device": 0,
        "steps": 32,
        "atol": 0.1,
        "rtol": 0.1,
    }
    arguments.update(kwargs)

    with pytest.raises(ValueError, match=message):
        _validate_run_arguments(**arguments)


def test_bucket_plan_rejects_dynamic_or_misrouted_contracts() -> None:
    with pytest.raises(ValueError, match="partial-bucket"):
        BucketGraphReplayPlan(
            bucket=4,
            capture_step_id=0,
            capture_actual=4,
            replay_step_ids=(0,),
            replay_actuals=(4,),
        )
    with pytest.raises(ValueError, match="first trace"):
        BucketGraphReplayPlan(
            bucket=4,
            capture_step_id=1,
            capture_actual=2,
            replay_step_ids=(0, 1),
            replay_actuals=(2, 3),
        )


def _bucket_evidence(bucket: int, step_id: int) -> StatefulACLGraphBucketEvidence:
    addresses = {"block_table": 1000 + bucket, "kv_seq_lens": 2000 + bucket}
    return StatefulACLGraphBucketEvidence(
        bucket=bucket,
        capture_step_id=step_id,
        capture_actual=bucket - 1,
        replay_step_ids=(step_id,),
        replay_actuals=(bucket - 1,),
        metadata_update_count=1,
        metadata_addresses_before=addresses,
        metadata_addresses_after=addresses,
        hidden_address=3000 + bucket,
        output_address=4000 + bucket,
        io_addresses_stable=True,
        warmup_capture_addresses_differ=True,
        pending_staging_owners_after_sync=0,
    )


@dataclass(frozen=True)
class _FakeStep:
    bucket: int
    reuse_event_count: int
    close: bool = True

    @property
    def actual(self) -> int:
        return self.bucket - 1

    def to_dict(self) -> dict[str, object]:
        return {
            "bucket": self.bucket,
            "actual": self.actual,
            "reuse_event_count": self.reuse_event_count,
            "close": self.close,
        }


def test_result_serializes_four_graph_shared_runtime_evidence() -> None:
    buckets = tuple(_bucket_evidence(bucket, index) for index, bucket in enumerate(BATCH_BUCKETS))
    steps = tuple(
        _FakeStep(bucket, reuse_event_count)
        for bucket, reuse_event_count in zip(BATCH_BUCKETS, (0, 1, 2, 2), strict=True)
    )
    result = A3StatefulACLGraphResult(
        runtime="tensormap_and_ringbuffer",
        device=0,
        trace_seed=73,
        requested_steps=4,
        graph_model=GRAPH_MODEL,
        runtime_owner_count=1,
        graph_capture_count=4,
        retained_capture_bindings=4,
        reuse_event_count=5,
        partial_bucket_step_count=4,
        replay_bucket_order=tuple(BATCH_BUCKETS),
        buckets=buckets,
        steps=steps,  # type: ignore[arg-type]
        native_final_page_padding_mismatches={"main": 0},
        pypto_final_page_padding_mismatches={"main": 0},
    )

    payload = result.to_dict()
    assert result.close
    assert payload["graph_model"] == "one_graph_per_static_bucket"
    assert payload["runtime_owner_count"] == 1
    assert payload["graph_capture_count"] == 4
    assert payload["reuse_event_count"] == 5
    assert payload["partial_bucket_step_count"] == 4
    assert all(item["metadata_addresses_stable"] for item in payload["buckets"])
    assert json.loads(json.dumps(payload))["close"] is True


def test_bucket_evidence_rejects_address_change_or_unsynchronized_source_owner() -> None:
    evidence = _bucket_evidence(4, 0)
    changed = StatefulACLGraphBucketEvidence(
        bucket=evidence.bucket,
        capture_step_id=evidence.capture_step_id,
        capture_actual=evidence.capture_actual,
        replay_step_ids=evidence.replay_step_ids,
        replay_actuals=evidence.replay_actuals,
        metadata_update_count=evidence.metadata_update_count,
        metadata_addresses_before=evidence.metadata_addresses_before,
        metadata_addresses_after={"block_table": 999, "kv_seq_lens": 2004},
        hidden_address=evidence.hidden_address,
        output_address=evidence.output_address,
        io_addresses_stable=True,
        warmup_capture_addresses_differ=True,
        pending_staging_owners_after_sync=1,
    )

    assert not changed.metadata_addresses_stable
    assert not changed.close


def test_cli_requires_one_runtime_and_defaults_to_repeated_four_bucket_trace() -> None:
    parser = build_parser()
    args = parser.parse_args(["--runtime", "host_build_graph"])

    assert args.runtime == "host_build_graph"
    assert args.device == 0
    assert args.steps == 32
    assert args.master_port == 29743
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_npu_runner_uses_one_graph_per_bucket_and_external_serial_replay() -> None:
    source = inspect.getsource(run_a3_stateful_aclgraph_compare)
    capture_region = source[
        source.index("with torch_npu.npu.graph(graph, stream=capture_stream):") : source.index(
            "capture_stream.synchronize()"
        )
    ]
    replay_region = source[
        source.index("# This is the complete replay enqueue region.") : source.index(
            "bucket_owner.metadata_owner.release_staging_owners_after_sync()"
        )
    ]

    assert "import torch_npu" in source
    assert source.count("installed_owner = install_pypto_dsv4_decode_csa(") == 1
    assert "batch_buckets=plan.buckets" in source
    assert source.count("torch_npu.npu.NPUGraph()") == 1
    assert "for bucket in plan.buckets:" in source
    assert capture_region.count("_custom_op_call(") == 1
    assert "synchronize" not in capture_region
    assert "zeros_like" not in capture_region
    assert "empty_like" not in capture_region
    assert "full_like" not in capture_region
    assert replay_region.index("initialize_acquired_blocks(") < replay_region.index("apply_staged_payload(")
    assert replay_region.index("apply_staged_payload(") < replay_region.index("hidden.copy_(hidden)")
    assert replay_region.index("hidden.copy_(hidden)") < replay_region.index("graph.replay()")
    assert replay_region.index("graph.replay()") < replay_region.index("stream.synchronize()")
    assert "torch.zeros" not in replay_region
    assert "torch.empty" not in replay_region
    assert source.index("bucket_owner.graph.reset()") < source.index("backend.close()")
