# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Fresh-process stateful multi-bucket ACLGraph runner for A3 decode CSA.

The four supported batch buckets are four different static PyPTO
specializations.  They cannot honestly be represented by one dynamic graph.
This runner therefore captures exactly one fixed-address graph for each of
``B4/B8/B12/B16`` while sharing one process-pinned PyPTO runtime owner and one
six-cache state world.  Decode steps select a graph by bucket and replay the
graphs strictly serially.

All graph-visible input, output, and metadata storage is allocated before
capture.  Every trace payload is also staged before capture.  A replay step
only enqueues block initialization, device-to-device copies into fixed graph
buffers, and ``graph.replay()`` on the bucket stream.  Synchronization is an
external caller boundary after replay; neither capture nor the L1 launch body
allocates or synchronizes.

Invoke this module once per runtime in a fresh process.  TRB and HBG must never
be selected sequentially in the same process because binaries, callable
handles, and captured bindings are intentionally process-pinned.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import torch

from tests.pypto_dsv4_decode_csa.a3_native_compare import (
    SUPPORTED_RUNTIMES,
    _custom_op_call,
    _destroy_tp1,
    _forward_context,
    _initialize_tp1,
    compare_tensors,
)
from tests.pypto_dsv4_decode_csa.a3_partial_bucket_smoke import (
    count_nonzero_padded_output,
    inspect_partial_bucket_canaries,
    seed_partial_bucket_canaries,
)
from tests.pypto_dsv4_decode_csa.a3_stateful_churn_compare import (
    A3StatefulChurnPlan,
    A3StatefulChurnStepResult,
    _block0_mismatches,
    _build_real_trace_metadata,
    _compare_step_states,
    _host_hidden_for_step,
    _minimal_trace_metadata,
    _NativeStepObservation,
    _restore_logical_state,
    _trace_world_from_staged,
    apply_state_deltas,
    build_a3_stateful_churn_plan,
    compute_state_deltas,
    expected_mutated_rows,
    initialize_acquired_blocks,
)
from tests.pypto_dsv4_decode_csa.metadata import (
    DecodeStepMetadataBufferOwner,
    StagedDecodeStepMetadataPayload,
)
from tests.pypto_dsv4_decode_csa.native_fixture import STATE_NAMES
from tests.pypto_dsv4_decode_csa.trace import BATCH_BUCKETS

GRAPH_MODEL = "one_graph_per_static_bucket"
_OUTPUT_CANARY = 37.0


@dataclass(frozen=True, slots=True)
class BucketGraphReplayPlan:
    """Immutable trace routing for one static-B captured graph."""

    bucket: int
    capture_step_id: int
    capture_actual: int
    replay_step_ids: tuple[int, ...]
    replay_actuals: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.bucket not in BATCH_BUCKETS:
            raise ValueError(f"unsupported graph bucket B{self.bucket}")
        if len(self.replay_step_ids) != len(self.replay_actuals):
            raise ValueError("replay step IDs and actual-request counts must have equal length")
        if not self.replay_step_ids:
            raise ValueError(f"B{self.bucket} graph must own at least one replay step")
        if self.capture_step_id != self.replay_step_ids[0]:
            raise ValueError("capture must use the first trace payload assigned to the bucket")
        if self.capture_actual != self.replay_actuals[0]:
            raise ValueError("capture actual must match its trace payload")
        if any(not 1 <= actual < self.bucket for actual in self.replay_actuals):
            raise ValueError(f"B{self.bucket} stateful graph requires partial-bucket 1 <= A < B payloads")

    @property
    def metadata_update_count(self) -> int:
        return len(self.replay_step_ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "bucket": self.bucket,
            "capture_step_id": self.capture_step_id,
            "capture_actual": self.capture_actual,
            "replay_step_ids": self.replay_step_ids,
            "replay_actuals": self.replay_actuals,
            "metadata_update_count": self.metadata_update_count,
        }


@dataclass(frozen=True, slots=True)
class A3StatefulACLGraphPlan:
    """Host-only plan joining lifecycle churn to four static graph owners."""

    churn: A3StatefulChurnPlan
    bucket_graphs: tuple[BucketGraphReplayPlan, ...]

    def __post_init__(self) -> None:
        buckets = tuple(item.bucket for item in self.bucket_graphs)
        if buckets != tuple(BATCH_BUCKETS):
            raise ValueError(f"graph plan must contain B4/B8/B12/B16 exactly once, got {buckets}")
        routed = sorted(step_id for item in self.bucket_graphs for step_id in item.replay_step_ids)
        expected = list(range(len(self.churn.payloads)))
        if routed != expected:
            raise ValueError("each trace step must be routed to exactly one static bucket graph")
        for item in self.bucket_graphs:
            for step_id, actual in zip(item.replay_step_ids, item.replay_actuals, strict=True):
                payload = self.churn.payloads[step_id]
                if payload.spec.batch != item.bucket or payload.actual != actual:
                    raise ValueError(f"step {step_id} was routed to the wrong graph specialization")

    @property
    def buckets(self) -> tuple[int, ...]:
        return tuple(item.bucket for item in self.bucket_graphs)

    @property
    def replay_bucket_order(self) -> tuple[int, ...]:
        return tuple(payload.spec.batch for payload in self.churn.payloads)

    @property
    def graph_capture_count(self) -> int:
        return len(self.bucket_graphs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "graph_model": GRAPH_MODEL,
            "requested_steps": len(self.churn.payloads),
            "buckets": self.buckets,
            "graph_capture_count": self.graph_capture_count,
            "replay_bucket_order": self.replay_bucket_order,
            "bucket_graphs": tuple(item.to_dict() for item in self.bucket_graphs),
        }


def build_a3_stateful_aclgraph_plan(*, steps: int, seed: int) -> A3StatefulACLGraphPlan:
    """Build a complete one-static-graph-per-bucket routing plan on Host."""
    churn = build_a3_stateful_churn_plan(steps=steps, seed=seed)
    bucket_graphs = []
    for bucket in BATCH_BUCKETS:
        assigned = tuple(payload for payload in churn.payloads if payload.spec.batch == bucket)
        if not assigned:
            raise ValueError(f"stateful trace did not exercise required graph bucket B{bucket}")
        bucket_graphs.append(
            BucketGraphReplayPlan(
                bucket=bucket,
                capture_step_id=assigned[0].step_id,
                capture_actual=assigned[0].actual,
                replay_step_ids=tuple(payload.step_id for payload in assigned),
                replay_actuals=tuple(payload.actual for payload in assigned),
            )
        )
    return A3StatefulACLGraphPlan(churn=churn, bucket_graphs=tuple(bucket_graphs))


@dataclass(frozen=True, slots=True)
class StatefulACLGraphBucketEvidence:
    """Address and replay evidence for one captured static specialization."""

    bucket: int
    capture_step_id: int
    capture_actual: int
    replay_step_ids: tuple[int, ...]
    replay_actuals: tuple[int, ...]
    metadata_update_count: int
    metadata_addresses_before: Mapping[str, int]
    metadata_addresses_after: Mapping[str, int]
    hidden_address: int
    output_address: int
    io_addresses_stable: bool
    warmup_capture_addresses_differ: bool
    pending_staging_owners_after_sync: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "metadata_addresses_before",
            MappingProxyType(dict(self.metadata_addresses_before)),
        )
        object.__setattr__(
            self,
            "metadata_addresses_after",
            MappingProxyType(dict(self.metadata_addresses_after)),
        )

    @property
    def metadata_addresses_stable(self) -> bool:
        return dict(self.metadata_addresses_before) == dict(self.metadata_addresses_after)

    @property
    def close(self) -> bool:
        return bool(
            self.bucket in BATCH_BUCKETS
            and bool(self.replay_step_ids)
            and len(self.replay_step_ids) == len(self.replay_actuals)
            and self.capture_step_id == self.replay_step_ids[0]
            and self.capture_actual == self.replay_actuals[0]
            and self.metadata_update_count == len(self.replay_step_ids)
            and all(1 <= actual < self.bucket for actual in self.replay_actuals)
            and self.metadata_addresses_stable
            and self.io_addresses_stable
            and self.warmup_capture_addresses_differ
            and self.pending_staging_owners_after_sync == 0
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "bucket": self.bucket,
            "capture_step_id": self.capture_step_id,
            "capture_actual": self.capture_actual,
            "replay_step_ids": self.replay_step_ids,
            "replay_actuals": self.replay_actuals,
            "metadata_update_count": self.metadata_update_count,
            "metadata_addresses_before": dict(self.metadata_addresses_before),
            "metadata_addresses_after": dict(self.metadata_addresses_after),
            "metadata_addresses_stable": self.metadata_addresses_stable,
            "hidden_address": self.hidden_address,
            "output_address": self.output_address,
            "io_addresses_stable": self.io_addresses_stable,
            "warmup_capture_addresses_differ": self.warmup_capture_addresses_differ,
            "pending_staging_owners_after_sync": self.pending_staging_owners_after_sync,
            "close": self.close,
        }


@dataclass(frozen=True, slots=True)
class A3StatefulACLGraphResult:
    """Inspectable correctness result for a shared-runtime four-graph trace."""

    runtime: str
    device: int
    trace_seed: int
    requested_steps: int
    graph_model: str
    runtime_owner_count: int
    graph_capture_count: int
    retained_capture_bindings: int
    reuse_event_count: int
    partial_bucket_step_count: int
    replay_bucket_order: tuple[int, ...]
    buckets: tuple[StatefulACLGraphBucketEvidence, ...]
    steps: tuple[A3StatefulChurnStepResult, ...]
    native_final_page_padding_mismatches: Mapping[str, int]
    pypto_final_page_padding_mismatches: Mapping[str, int]

    @property
    def close(self) -> bool:
        return bool(
            self.graph_model == GRAPH_MODEL
            and self.runtime_owner_count == 1
            and self.graph_capture_count == len(BATCH_BUCKETS)
            and self.retained_capture_bindings == len(BATCH_BUCKETS)
            and self.reuse_event_count > 0
            and self.partial_bucket_step_count == self.requested_steps
            and tuple(item.bucket for item in self.buckets) == tuple(BATCH_BUCKETS)
            and all(item.close for item in self.buckets)
            and len(self.steps) == self.requested_steps
            and tuple(step.bucket for step in self.steps) == self.replay_bucket_order
            and self.reuse_event_count == sum(step.reuse_event_count for step in self.steps)
            and self.partial_bucket_step_count == sum(step.actual < step.bucket for step in self.steps)
            and all(step.close for step in self.steps)
            and not any(self.native_final_page_padding_mismatches.values())
            and not any(self.pypto_final_page_padding_mismatches.values())
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "runtime": self.runtime,
            "device": self.device,
            "trace_seed": self.trace_seed,
            "requested_steps": self.requested_steps,
            "graph_model": self.graph_model,
            "runtime_owner_count": self.runtime_owner_count,
            "graph_capture_count": self.graph_capture_count,
            "retained_capture_bindings": self.retained_capture_bindings,
            "reuse_event_count": self.reuse_event_count,
            "partial_bucket_step_count": self.partial_bucket_step_count,
            "replay_bucket_order": self.replay_bucket_order,
            "buckets": tuple(item.to_dict() for item in self.buckets),
            "steps": tuple(step.to_dict() for step in self.steps),
            "native_final_page_padding_mismatches": dict(self.native_final_page_padding_mismatches),
            "pypto_final_page_padding_mismatches": dict(self.pypto_final_page_padding_mismatches),
            "close": self.close,
        }


@dataclass(slots=True)
class _CapturedBucketOwner:
    plan: BucketGraphReplayPlan
    metadata_owner: DecodeStepMetadataBufferOwner
    metadata: tuple[Any, ...]
    context: Any
    graph: Any
    stream: Any
    hidden: torch.Tensor
    output: torch.Tensor
    warmup_hidden_address: int
    warmup_output_address: int
    initial_metadata_addresses: Mapping[str, int]
    initial_hidden_address: int
    initial_output_address: int
    replay_step_ids: list[int]
    replay_actuals: list[int]
    io_addresses_stable: bool = True

    def evidence(self) -> StatefulACLGraphBucketEvidence:
        return StatefulACLGraphBucketEvidence(
            bucket=self.plan.bucket,
            capture_step_id=self.plan.capture_step_id,
            capture_actual=self.plan.capture_actual,
            replay_step_ids=tuple(self.replay_step_ids),
            replay_actuals=tuple(self.replay_actuals),
            metadata_update_count=len(self.replay_step_ids),
            metadata_addresses_before=self.initial_metadata_addresses,
            metadata_addresses_after=self.metadata_owner.device_metadata_addresses(),
            hidden_address=self.initial_hidden_address,
            output_address=self.initial_output_address,
            io_addresses_stable=self.io_addresses_stable,
            warmup_capture_addresses_differ=(
                self.warmup_hidden_address != self.initial_hidden_address
                and self.warmup_output_address != self.initial_output_address
            ),
            pending_staging_owners_after_sync=self.metadata_owner.pending_staging_owner_count,
        )


def _validate_run_arguments(
    *,
    runtime: str,
    device: int,
    steps: int,
    atol: float,
    rtol: float,
) -> None:
    if runtime not in SUPPORTED_RUNTIMES:
        raise ValueError(f"runtime must be one of {SUPPORTED_RUNTIMES}, got {runtime!r}")
    if isinstance(device, bool) or not isinstance(device, int) or device < 0:
        raise ValueError("device must be a non-negative integer")
    if steps not in {4, 32, 128}:
        raise ValueError("stateful ACLGraph runner supports exactly 4, 32, or 128 steps")
    if atol < 0.0 or rtol < 0.0:
        raise ValueError("atol and rtol must be non-negative")


def run_a3_stateful_aclgraph_compare(
    *,
    runtime: str,
    device: int = 0,
    steps: int = 32,
    trace_seed: int = 20260903,
    weight_seed: int = 20260902,
    atol: float = 0.1,
    rtol: float = 0.1,
    master_port: int = 29743,
) -> A3StatefulACLGraphResult:
    """Run a nonzero native oracle, then the same trace through four graphs."""
    _validate_run_arguments(
        runtime=runtime,
        device=device,
        steps=steps,
        atol=atol,
        rtol=rtol,
    )
    plan = build_a3_stateful_aclgraph_plan(steps=steps, seed=trace_seed)

    import torch_npu
    from vllm.config import set_current_vllm_config

    import vllm_ascend.ops.dsa  # noqa: F401 -- register production custom op
    from tests.pypto_dsv4_decode_csa.native_fixture import (
        DEFAULT_LAYER_PREFIX,
        bind_state_world,
        build_real_ratio4_attention,
        build_synthetic_vllm_config,
        build_twin_decode_csa_state_worlds,
    )
    from vllm_ascend.ops._pypto_dsv4_csa import (
        DecodeCSADeviceOwnerRegistry,
        install_pypto_dsv4_decode_csa,
        uninstall_pypto_dsv4_decode_csa,
    )
    from vllm_ascend.utils import register_ascend_customop

    config_owner = build_synthetic_vllm_config(batch=max(BATCH_BUCKETS))
    registry = DecodeCSADeviceOwnerRegistry()
    installed_owner = None
    captured: dict[int, _CapturedBucketOwner] = {}
    tp_initialized = False
    target = torch.device(f"npu:{device}")
    try:
        with set_current_vllm_config(config_owner.vllm_config):
            _initialize_tp1(device=device, master_port=master_port)
            tp_initialized = True
            register_ascend_customop(config_owner.vllm_config)
            attention = build_real_ratio4_attention(
                config_owner,
                device=target,
                prefix=DEFAULT_LAYER_PREFIX,
                seed=weight_seed,
            )
            wrapper = attention.dsa_attn
            base_spec = plan.churn.specs_by_bucket[max(plan.buckets)]
            native_base, pypto_base = build_twin_decode_csa_state_worlds(base_spec, device=target)
            seed_partial_bucket_canaries(native_base.raw_cache_tuple, native_base.storage_owners)
            seed_partial_bucket_canaries(pypto_base.raw_cache_tuple, pypto_base.storage_owners)
            torch_npu.npu.synchronize(device)
            native_initial = native_base.snapshot()
            pypto_initial = pypto_base.snapshot()
            for name in STATE_NAMES:
                if not torch.equal(native_initial[name], pypto_initial[name]):
                    raise AssertionError(f"independent initial state differs for {name}")
            native_block0 = tuple(tensor[0].detach().cpu().clone() for tensor in native_base.raw_cache_tuple)
            pypto_block0 = tuple(tensor[0].detach().cpu().clone() for tensor in pypto_base.raw_cache_tuple)

            # Every replay source is allocated and copied to the device before
            # graph capture.  The replay path below is device-to-device only.
            staged_payloads = tuple(
                StagedDecodeStepMetadataPayload.from_host(payload, device=target) for payload in plan.churn.payloads
            )
            staged_hidden = tuple(
                _host_hidden_for_step(payload, seed=trace_seed).to(target) for payload in plan.churn.payloads
            )

            # Native executes the same lifecycle first and stores compact row
            # deltas, avoiding 128 complete Host snapshots of all six caches.
            bind_state_world(wrapper, native_base)
            native_previous = native_initial
            native_observations = []
            for payload, staged, hidden, initialization in zip(
                plan.churn.payloads,
                staged_payloads,
                staged_hidden,
                plan.churn.initializations,
                strict=True,
            ):
                initialize_acquired_blocks(native_base.raw_cache_tuple, initialization)
                world = _trace_world_from_staged(native_base, staged)
                metadata = _build_real_trace_metadata(
                    wrapper,
                    world,
                    payload,
                    config_owner.vllm_config,
                )
                context = _forward_context(wrapper, metadata)
                context.num_tokens = payload.spec.tokens
                output = torch.zeros_like(hidden)
                _custom_op_call(wrapper, context, hidden, output)
                torch_npu.npu.synchronize(device)
                current = native_base.snapshot()
                native_observations.append(
                    _NativeStepObservation(
                        active_output=output[: payload.actual_tokens].detach().cpu().clone(),
                        padded_output_nonzero=count_nonzero_padded_output(
                            output,
                            actual=payload.actual,
                            seq=payload.spec.seq,
                        ),
                        deltas=compute_state_deltas(native_previous, current),
                        block0_mismatches=_block0_mismatches(native_base.raw_cache_tuple, native_block0),
                    )
                )
                native_previous = current
            native_canaries = inspect_partial_bucket_canaries(
                native_base.raw_cache_tuple,
                native_base.storage_owners,
            )

            # One real metadata bundle provides installation-time static
            # weights/RoPE/Hadamard ownership.  Replay metadata lives in the
            # separate fixed-address owner for each bucket below.
            bind_state_world(wrapper, pypto_base)
            install_world = _trace_world_from_staged(pypto_base, staged_payloads[0])
            install_metadata = _build_real_trace_metadata(
                wrapper,
                install_world,
                plan.churn.payloads[0],
                config_owner.vllm_config,
            )
            installed_owner = install_pypto_dsv4_decode_csa(
                wrapper,
                kv_cache=pypto_base.raw_cache_tuple,
                attn_metadata=install_metadata.metadata,
                device=device,
                runtime=runtime,
                batch_buckets=plan.buckets,
                registry=registry,
                uniform_query_rows_contract=True,
            )
            install_request = install_metadata.metadata[0].req_metadata
            full_rope_cos = install_request.full_compress_cos
            full_rope_sin = install_request.full_compress_sin
            hadamard = install_metadata.metadata[3].hadamard
            if not all(isinstance(value, torch.Tensor) for value in (full_rope_cos, full_rope_sin, hadamard)):
                raise RuntimeError("real DSA metadata did not expose full RoPE and Hadamard tensors")

            # Allocate one immutable graph-visible buffer family per static B.
            # Warmups use distinct input/output addresses and happen before
            # mark_warmups_quiesced().
            warmup_io: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
            bucket_setup: dict[
                int,
                tuple[BucketGraphReplayPlan, DecodeStepMetadataBufferOwner, tuple[Any, ...], Any],
            ] = {}
            for graph_plan in plan.bucket_graphs:
                staged = staged_payloads[graph_plan.capture_step_id]
                metadata_owner = DecodeStepMetadataBufferOwner.create(staged)
                metadata = _minimal_trace_metadata(
                    wrapper,
                    metadata_owner.bound_payload,
                    full_rope_cos=full_rope_cos,
                    full_rope_sin=full_rope_sin,
                    hadamard=hadamard,
                )
                context = _forward_context(wrapper, metadata)
                context.capturing = False
                warmup_hidden = staged_hidden[graph_plan.capture_step_id].clone()
                warmup_output = torch.zeros_like(warmup_hidden)
                _custom_op_call(wrapper, context, warmup_hidden, warmup_output)
                warmup_io[graph_plan.bucket] = (warmup_hidden, warmup_output)
                bucket_setup[graph_plan.bucket] = (graph_plan, metadata_owner, metadata, context)
            torch_npu.npu.synchronize(device)
            installed_owner.device_owner.mark_warmups_quiesced()
            _restore_logical_state(pypto_base.raw_cache_tuple, pypto_initial)
            torch_npu.npu.synchronize(device)

            # Capture exactly one custom-op node per bucket.  A graph capture
            # executes its node, so restore the sacrificial cache state after
            # every capture before constructing the next graph.
            for bucket in plan.buckets:
                graph_plan, metadata_owner, metadata, context = bucket_setup[bucket]
                capture_hidden = torch.empty_like(staged_hidden[graph_plan.capture_step_id])
                capture_output = torch.full_like(capture_hidden, _OUTPUT_CANARY)
                capture_hidden.copy_(staged_hidden[graph_plan.capture_step_id])
                # The fixed capture buffers were populated on the caller's
                # default stream.  Establish quiescence before handing their
                # addresses to a distinct capture stream; this is outside the
                # graph and outside the L1 launch body.
                torch_npu.npu.synchronize(device)
                capture_stream = torch_npu.npu.Stream(device=device)
                graph = torch_npu.npu.NPUGraph()
                warmup_hidden, warmup_output = warmup_io[bucket]
                bucket_owner = _CapturedBucketOwner(
                    plan=graph_plan,
                    metadata_owner=metadata_owner,
                    metadata=metadata,
                    context=context,
                    graph=graph,
                    stream=capture_stream,
                    hidden=capture_hidden,
                    output=capture_output,
                    warmup_hidden_address=int(warmup_hidden.data_ptr()),
                    warmup_output_address=int(warmup_output.data_ptr()),
                    initial_metadata_addresses=metadata_owner.device_metadata_addresses(),
                    initial_hidden_address=int(capture_hidden.data_ptr()),
                    initial_output_address=int(capture_output.data_ptr()),
                    replay_step_ids=[],
                    replay_actuals=[],
                )
                # Register the cleanup owner before capture so even a partial
                # capture failure cannot lose the graph-visible allocations.
                captured[bucket] = bucket_owner
                context.capturing = True
                with torch_npu.npu.graph(graph, stream=capture_stream):
                    _custom_op_call(wrapper, context, capture_hidden, capture_output)
                capture_stream.synchronize()
                metadata_owner.release_staging_owners_after_sync()
                _restore_logical_state(pypto_base.raw_cache_tuple, pypto_initial)
                torch_npu.npu.synchronize(device)

            retained_bindings = len(installed_owner.device_owner.backend._bindings)
            if retained_bindings != len(BATCH_BUCKETS):
                raise AssertionError(f"expected one retained binding per static graph, got {retained_bindings}")
            capture_canaries = inspect_partial_bucket_canaries(
                pypto_base.raw_cache_tuple,
                pypto_base.storage_owners,
            )
            if not capture_canaries.clean:
                raise AssertionError(
                    f"sacrificial warmup/capture corrupted reserved canaries: {capture_canaries.to_dict()}"
                )

            native_reconstructed = {name: value.clone() for name, value in native_initial.items()}
            pypto_previous = pypto_initial
            step_results = []
            for payload, staged, hidden, initialization, native in zip(
                plan.churn.payloads,
                staged_payloads,
                staged_hidden,
                plan.churn.initializations,
                native_observations,
                strict=True,
            ):
                bucket_owner = captured[payload.spec.batch]
                # This is the complete replay enqueue region.  Sources and
                # destinations already exist on device; all operations share
                # the graph stream and therefore need no pre-replay sync.
                with torch_npu.npu.stream(bucket_owner.stream):
                    initialize_acquired_blocks(pypto_base.raw_cache_tuple, initialization)
                    bucket_owner.metadata_owner.apply_staged_payload(staged)
                    bucket_owner.hidden.copy_(hidden)
                    bucket_owner.output.fill_(_OUTPUT_CANARY)
                    bucket_owner.graph.replay()
                bucket_owner.stream.synchronize()
                bucket_owner.metadata_owner.release_staging_owners_after_sync()

                bucket_owner.replay_step_ids.append(payload.step_id)
                bucket_owner.replay_actuals.append(payload.actual)
                bucket_owner.io_addresses_stable &= bool(
                    bucket_owner.hidden.data_ptr() == bucket_owner.initial_hidden_address
                    and bucket_owner.output.data_ptr() == bucket_owner.initial_output_address
                )
                if dict(bucket_owner.metadata_owner.device_metadata_addresses()) != dict(
                    bucket_owner.initial_metadata_addresses
                ):
                    raise AssertionError(f"B{payload.spec.batch} metadata address changed across replay")

                pypto_current = pypto_base.snapshot()
                pypto_deltas = compute_state_deltas(pypto_previous, pypto_current)
                apply_state_deltas(native_reconstructed, native.deltas)
                legal_rows = expected_mutated_rows(payload, initialization)
                state_results, indexer_quantized, indexer_quantized_cumulative = _compare_step_states(
                    native_reconstructed,
                    pypto_current,
                    native_initial,
                    native.deltas,
                    pypto_deltas,
                    legal_rows,
                    atol=atol,
                    rtol=rtol,
                )
                active_output = bucket_owner.output[: payload.actual_tokens].detach().cpu()
                output_comparison = compare_tensors(
                    native.active_output,
                    active_output,
                    atol=atol,
                    rtol=rtol,
                )
                step_result = A3StatefulChurnStepResult(
                    step_id=payload.step_id,
                    bucket=payload.spec.batch,
                    actual=payload.actual,
                    request_ids=payload.request_ids,
                    admitted_request_ids=payload.source_step.admitted_request_ids,
                    retired_request_ids=payload.source_step.retired_request_ids,
                    reuse_event_count=len(initialization.reuse_events),
                    output=output_comparison,
                    states=state_results,
                    indexer_quantized=indexer_quantized,
                    indexer_quantized_cumulative=indexer_quantized_cumulative,
                    native_padded_output_nonzero=native.padded_output_nonzero,
                    pypto_padded_output_nonzero=count_nonzero_padded_output(
                        bucket_owner.output,
                        actual=payload.actual,
                        seq=payload.spec.seq,
                    ),
                    native_block0_mismatches=native.block0_mismatches,
                    pypto_block0_mismatches=_block0_mismatches(
                        pypto_base.raw_cache_tuple,
                        pypto_block0,
                    ),
                )
                step_results.append(step_result)
                if not step_result.close:
                    raise AssertionError(json.dumps(step_result.to_dict(), ensure_ascii=False, indent=2))
                pypto_previous = pypto_current

            bucket_evidence = tuple(captured[bucket].evidence() for bucket in plan.buckets)
            for graph_plan, evidence in zip(plan.bucket_graphs, bucket_evidence, strict=True):
                if evidence.replay_step_ids != graph_plan.replay_step_ids:
                    raise AssertionError(f"B{graph_plan.bucket} graph replay routing diverged from Host plan")
                if not evidence.close:
                    raise AssertionError(json.dumps(evidence.to_dict(), ensure_ascii=False, indent=2))
            pypto_canaries = inspect_partial_bucket_canaries(
                pypto_base.raw_cache_tuple,
                pypto_base.storage_owners,
            )
            result = A3StatefulACLGraphResult(
                runtime=runtime,
                device=device,
                trace_seed=trace_seed,
                requested_steps=steps,
                graph_model=GRAPH_MODEL,
                runtime_owner_count=1,
                graph_capture_count=len(captured),
                retained_capture_bindings=retained_bindings,
                reuse_event_count=plan.churn.reuse_event_count,
                partial_bucket_step_count=sum(payload.actual < payload.spec.batch for payload in plan.churn.payloads),
                replay_bucket_order=plan.replay_bucket_order,
                buckets=bucket_evidence,
                steps=tuple(step_results),
                native_final_page_padding_mismatches=native_canaries.page_padding_mismatches,
                pypto_final_page_padding_mismatches=pypto_canaries.page_padding_mismatches,
            )
            if not result.close:
                raise AssertionError(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
            return result
    finally:
        try:
            if tp_initialized:
                torch_npu.npu.synchronize(device)
                for bucket_owner in captured.values():
                    bucket_owner.graph.reset()
                if installed_owner is not None:
                    uninstall_pypto_dsv4_decode_csa(installed_owner.layer)
                    installed_owner.device_owner.backend.close()
        finally:
            config_owner.close()
            if tp_initialized:
                _destroy_tp1()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", choices=SUPPORTED_RUNTIMES, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--steps", type=int, choices=(4, 32, 128), default=32)
    parser.add_argument("--trace-seed", type=int, default=20260903)
    parser.add_argument("--weight-seed", type=int, default=20260902)
    parser.add_argument("--atol", type=float, default=0.1)
    parser.add_argument("--rtol", type=float, default=0.1)
    parser.add_argument("--master-port", type=int, default=29743)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_a3_stateful_aclgraph_compare(
        runtime=args.runtime,
        device=args.device,
        steps=args.steps,
        trace_seed=args.trace_seed,
        weight_seed=args.weight_seed,
        atol=args.atol,
        rtol=args.rtol,
        master_port=args.master_port,
    )
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "GRAPH_MODEL",
    "A3StatefulACLGraphPlan",
    "A3StatefulACLGraphResult",
    "BucketGraphReplayPlan",
    "StatefulACLGraphBucketEvidence",
    "build_a3_stateful_aclgraph_plan",
    "build_parser",
    "main",
    "run_a3_stateful_aclgraph_compare",
]
