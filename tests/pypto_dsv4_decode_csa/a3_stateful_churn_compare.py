# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Fresh-process A3 native-vs-PyPTO stateful request-churn runner.

Unlike the fixed-request advance probes, this runner consumes the immutable
``DecodeTrace`` lifecycle: requests are admitted, advanced, retired, compacted
to an active prefix, and their physical blocks are reused.  Every block newly
acquired by an admission or survivor-growth transition is semantically
initialized before its first decode launch.  Reuse is therefore explicit
test-scheduler work rather than an accidental dependency on stale contents.

Imports that initialize distributed or NPU state are confined to
``run_a3_stateful_churn_compare``.  The plan, lifecycle, delta and report
contracts remain Host-testable.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Any

import torch

from tests.pypto_dsv4_decode_csa.a3_native_compare import (
    SUPPORTED_RUNTIMES,
    TensorComparison,
    _custom_op_call,
    _destroy_tp1,
    _forward_context,
    _initialize_tp1,
    compare_state_snapshots,
    compare_tensors,
)
from tests.pypto_dsv4_decode_csa.a3_partial_bucket_smoke import (
    build_partial_bucket_native_metadata,
    count_nonzero_padded_output,
    inspect_partial_bucket_canaries,
    seed_partial_bucket_canaries,
)
from tests.pypto_dsv4_decode_csa.indexer_quantized_comparison import (
    IndexerQuantizedComparison,
    compare_indexer_quantized,
)
from tests.pypto_dsv4_decode_csa.metadata import (
    DecodeStepMetadataMaterializer,
    DecodeStepMetadataPayload,
    StagedDecodeStepMetadataPayload,
    derive_trace_metadata_requirements,
    materialize_decode_trace,
)
from tests.pypto_dsv4_decode_csa.native_fixture import STATE_NAMES
from tests.pypto_dsv4_decode_csa.trace import (
    BATCH_BUCKETS,
    CACHE_FAMILY_NAMES,
    COMPRESSOR_ATTENTION,
    INDEXER_CACHE,
    INNER_COMPRESSOR_STATE,
    MAIN_COMPRESSOR_STATE,
    SWA_CACHE,
    DecodeStep,
    DecodeTrace,
    build_deterministic_churn_trace,
)
from vllm_ascend.ops._pypto_dsv4_csa import DecodeCSAProgramSpec
from vllm_ascend.ops._pypto_dsv4_csa.config import BLOCK_SIZE, FLASH

ALL_FAMILY_BLOCK_OFFSETS = MappingProxyType({family: 1 for family in CACHE_FAMILY_NAMES})
SUPPORTED_SPEC_STEPS = (4, 32, 128)
SUPPORTED_EXECUTE_STEPS = (1, 4, 32, 128)
SUPPORTED_NATIVE_PRECONDITION_STEPS = (1, 4, 32, 128)
SUPPORTED_RETAINED_STAGE_STEPS = (1, 4, 32, 128)
_STATIC_EXTENT_NAMES = (
    "swa_blocks",
    "compressed_blocks",
    "main_state_blocks",
    "inner_state_blocks",
    "indexer_blocks",
    "swa_table_width",
    "compressed_table_width",
    "main_state_table_width",
    "inner_state_table_width",
    "indexer_table_width",
)
_STATE_TO_TRACE_FAMILY = MappingProxyType(
    {
        "compressed_kv": COMPRESSOR_ATTENTION,
        "swa_kv": SWA_CACHE,
        "compressor_state": MAIN_COMPRESSOR_STATE,
        "indexer_compressor_state": INNER_COMPRESSOR_STATE,
        "indexer_k": INDEXER_CACHE,
        "indexer_scale": INDEXER_CACHE,
    }
)
_TRACE_FAMILY_TO_STATE_NAMES = MappingProxyType(
    {
        COMPRESSOR_ATTENTION: ("compressed_kv",),
        SWA_CACHE: ("swa_kv",),
        MAIN_COMPRESSOR_STATE: ("compressor_state",),
        INNER_COMPRESSOR_STATE: ("indexer_compressor_state",),
        INDEXER_CACHE: ("indexer_k", "indexer_scale"),
    }
)
_ROWS_PER_BLOCK = MappingProxyType(
    {
        COMPRESSOR_ATTENTION: BLOCK_SIZE,
        SWA_CACHE: BLOCK_SIZE,
        MAIN_COMPRESSOR_STATE: 2,
        INNER_COMPRESSOR_STATE: 2,
        INDEXER_CACHE: BLOCK_SIZE,
    }
)


class A3StatefulChurnCapabilityError(RuntimeError):
    """A fail-fast boundary when the installed native stack lacks a required API."""


@dataclass(frozen=True, slots=True)
class BlockReuseEvent:
    family: str
    abstract_block: int
    physical_block: int
    retired_request_id: int
    retired_step: int
    acquiring_request_id: int
    acquired_step: int


@dataclass(frozen=True, slots=True)
class StepBlockInitialization:
    step_id: int
    admitted_request_ids: tuple[int, ...]
    physical_blocks: Mapping[str, tuple[int, ...]]
    reuse_events: tuple[BlockReuseEvent, ...]

    def __post_init__(self) -> None:
        if tuple(self.physical_blocks) != CACHE_FAMILY_NAMES:
            raise ValueError("physical_blocks must preserve the five-family canonical order")
        if any(0 in blocks for blocks in self.physical_blocks.values()):
            raise ValueError("physical block 0 is reserved as a canary and must never be initialized")


@dataclass(frozen=True, slots=True)
class A3StatefulChurnPlan:
    spec_trace: DecodeTrace
    trace: DecodeTrace
    specs_by_bucket: Mapping[int, DecodeCSAProgramSpec]
    warmup_payloads: tuple[DecodeStepMetadataPayload, ...]
    payloads: tuple[DecodeStepMetadataPayload, ...]
    initializations: tuple[StepBlockInitialization, ...]
    native_precondition_payloads: tuple[DecodeStepMetadataPayload, ...]
    native_precondition_initializations: tuple[StepBlockInitialization, ...]
    retained_stage_payloads: tuple[DecodeStepMetadataPayload, ...]
    physical_block_offsets: Mapping[str, int]

    def __post_init__(self) -> None:
        if self.trace.steps != self.spec_trace.steps[: len(self.trace.steps)]:
            raise ValueError("execution trace must be an exact prefix of the static-spec trace")
        if len(self.payloads) != len(self.trace.steps):
            raise ValueError("execution payload count must equal execution trace length")
        if len(self.initializations) != len(self.trace.steps):
            raise ValueError("initialization count must equal execution trace length")
        if len(self.native_precondition_payloads) < len(self.payloads):
            raise ValueError("native precondition must cover every candidate execution step")
        if len(self.native_precondition_payloads) != len(self.native_precondition_initializations):
            raise ValueError("native payload and initialization counts must match")
        if not self.retained_stage_payloads:
            raise ValueError("at least one staged metadata payload owner must be retained")
        for name, selected in (
            ("native precondition", self.native_precondition_payloads),
            ("retained stage", self.retained_stage_payloads),
        ):
            selected_steps = tuple(payload.source_step for payload in selected)
            if selected_steps != self.spec_trace.steps[: len(selected)]:
                raise ValueError(f"{name} payloads must be an exact prefix of the static-spec trace")
        if tuple(payload.spec.batch for payload in self.warmup_payloads) != tuple(sorted(self.specs_by_bucket)):
            raise ValueError("warmup payloads must cover each static batch specialization exactly once")

    @property
    def spec_steps(self) -> int:
        return len(self.spec_trace.steps)

    @property
    def execute_steps(self) -> int:
        return len(self.trace.steps)

    @property
    def native_precondition_steps(self) -> int:
        return len(self.native_precondition_payloads)

    @property
    def retained_stage_steps(self) -> int:
        return len(self.retained_stage_payloads)

    @property
    def buckets(self) -> tuple[int, ...]:
        return tuple(sorted(self.specs_by_bucket))

    @property
    def reuse_event_count(self) -> int:
        return sum(len(item.reuse_events) for item in self.initializations)


@dataclass(frozen=True, slots=True)
class A3StatefulStaticExtentOverrides:
    """Diagnostic-only replacements for independently bisecting static extents.

    Production plans always use the minimum extents derived from ``spec_trace``.
    This value object is accepted only by
    :func:`build_a3_stateful_churn_diagnostic_plan`; it deliberately does not
    enter the normal runner API.
    """

    swa_blocks: int | None = None
    compressed_blocks: int | None = None
    main_state_blocks: int | None = None
    inner_state_blocks: int | None = None
    indexer_blocks: int | None = None
    swa_table_width: int | None = None
    compressed_table_width: int | None = None
    main_state_table_width: int | None = None
    inner_state_table_width: int | None = None
    indexer_table_width: int | None = None

    def __post_init__(self) -> None:
        invalid = {
            name: value
            for name in _STATIC_EXTENT_NAMES
            if (value := getattr(self, name)) is not None
            and (isinstance(value, bool) or not isinstance(value, int) or value <= 0)
        }
        if invalid:
            raise ValueError(f"static extent overrides must be positive integers or None, got {invalid}")

    def apply(self, derived: Mapping[str, int]) -> dict[str, int]:
        result = dict(derived)
        for name in _STATIC_EXTENT_NAMES:
            value = getattr(self, name)
            if value is not None:
                result[name] = value
        return result


def _ownership_by_request(step: DecodeStep) -> Mapping[int, object]:
    return MappingProxyType({request.request_id: request for request in step.active_requests})


def _build_initialization_plan(trace: DecodeTrace) -> tuple[StepBlockInitialization, ...]:
    previous: Mapping[int, object] = MappingProxyType({})
    free_history: dict[str, dict[int, tuple[int, int]]] = {family: {} for family in CACHE_FAMILY_NAMES}
    result = []
    for step in trace.steps:
        current = _ownership_by_request(step)
        for request_id in step.retired_request_ids:
            retired = previous[request_id]
            for family in CACHE_FAMILY_NAMES:
                for block in retired.ownership_for(family).logical_block_table:
                    free_history[family][block] = (request_id, step.step_id)

        blocks_by_family: dict[str, tuple[int, ...]] = {}
        reuse_events = []
        for family in CACHE_FAMILY_NAMES:
            acquired_by_request: dict[int, tuple[int, ...]] = {}
            for request_id, request in current.items():
                current_blocks = request.ownership_for(family).logical_block_table
                previous_blocks = (
                    previous[request_id].ownership_for(family).logical_block_table if request_id in previous else ()
                )
                if current_blocks[: len(previous_blocks)] != previous_blocks:
                    raise AssertionError(f"step {step.step_id}/{family}: live ownership prefix unexpectedly changed")
                acquired_by_request[request_id] = current_blocks[len(previous_blocks) :]
            abstract_blocks = tuple(block for blocks in acquired_by_request.values() for block in blocks)
            concrete_blocks = tuple(block + ALL_FAMILY_BLOCK_OFFSETS[family] for block in abstract_blocks)
            blocks_by_family[family] = concrete_blocks
            for request_id, acquired_blocks in acquired_by_request.items():
                for block in acquired_blocks:
                    retired_owner = free_history[family].pop(block, None)
                    if retired_owner is not None:
                        reuse_events.append(
                            BlockReuseEvent(
                                family=family,
                                abstract_block=block,
                                physical_block=block + ALL_FAMILY_BLOCK_OFFSETS[family],
                                retired_request_id=retired_owner[0],
                                retired_step=retired_owner[1],
                                acquiring_request_id=request_id,
                                acquired_step=step.step_id,
                            )
                        )
        result.append(
            StepBlockInitialization(
                step_id=step.step_id,
                admitted_request_ids=step.admitted_request_ids,
                physical_blocks=MappingProxyType(blocks_by_family),
                reuse_events=tuple(reuse_events),
            )
        )
        previous = current
    return tuple(result)


def _resolve_trace_lengths(
    *,
    steps: int | None,
    spec_steps: int | None,
    execute_steps: int | None,
    native_precondition_steps: int | None,
    retained_stage_steps: int | None,
    default_steps: int | None,
) -> tuple[int, int, int, int]:
    """Resolve legacy one-knob or four independent trace-prefix lengths."""
    explicit = (spec_steps, execute_steps, native_precondition_steps, retained_stage_steps)
    if steps is not None and any(value is not None for value in explicit):
        raise ValueError(
            "steps cannot be combined with spec_steps, execute_steps, "
            "native_precondition_steps, or retained_stage_steps"
        )
    if steps is not None:
        spec_steps = execute_steps = native_precondition_steps = retained_stage_steps = steps
    elif all(value is None for value in explicit):
        if default_steps is None:
            raise ValueError("provide steps, or provide both spec_steps and execute_steps")
        spec_steps = execute_steps = native_precondition_steps = retained_stage_steps = default_steps
    elif spec_steps is None or execute_steps is None:
        raise ValueError("spec_steps and execute_steps must be provided together")
    else:
        if native_precondition_steps is None:
            native_precondition_steps = execute_steps
        if retained_stage_steps is None:
            retained_stage_steps = execute_steps

    assert spec_steps is not None
    assert execute_steps is not None
    assert native_precondition_steps is not None
    assert retained_stage_steps is not None
    if spec_steps not in SUPPORTED_SPEC_STEPS:
        raise ValueError(f"spec_steps must be one of {SUPPORTED_SPEC_STEPS}, got {spec_steps}")
    if execute_steps not in SUPPORTED_EXECUTE_STEPS:
        raise ValueError(f"execute_steps must be one of {SUPPORTED_EXECUTE_STEPS}, got {execute_steps}")
    if native_precondition_steps not in SUPPORTED_NATIVE_PRECONDITION_STEPS:
        raise ValueError(
            "native_precondition_steps must be one of "
            f"{SUPPORTED_NATIVE_PRECONDITION_STEPS}, got {native_precondition_steps}"
        )
    if retained_stage_steps not in SUPPORTED_RETAINED_STAGE_STEPS:
        raise ValueError(
            f"retained_stage_steps must be one of {SUPPORTED_RETAINED_STAGE_STEPS}, got {retained_stage_steps}"
        )
    if execute_steps > spec_steps:
        raise ValueError(f"execute_steps={execute_steps} cannot exceed spec_steps={spec_steps}")
    if native_precondition_steps < execute_steps:
        raise ValueError(
            f"native_precondition_steps={native_precondition_steps} cannot be smaller than "
            f"execute_steps={execute_steps} because every candidate step needs a native oracle"
        )
    if native_precondition_steps > spec_steps:
        raise ValueError(f"native_precondition_steps={native_precondition_steps} cannot exceed spec_steps={spec_steps}")
    if retained_stage_steps > spec_steps:
        raise ValueError(f"retained_stage_steps={retained_stage_steps} cannot exceed spec_steps={spec_steps}")
    return spec_steps, execute_steps, native_precondition_steps, retained_stage_steps


def _first_warmup_payloads(
    spec_trace: DecodeTrace,
    specs: Mapping[int, DecodeCSAProgramSpec],
    execution_payloads: tuple[DecodeStepMetadataPayload, ...],
) -> tuple[DecodeStepMetadataPayload, ...]:
    """Materialize only the first payload for every bucket in the spec trace."""
    execution_by_step = {payload.step_id: payload for payload in execution_payloads}
    first_steps: dict[int, DecodeStep] = {}
    for step in spec_trace.steps:
        first_steps.setdefault(step.bucket_size, step)

    result = []
    for bucket in sorted(specs):
        step = first_steps[bucket]
        execution_payload = execution_by_step.get(step.step_id)
        if execution_payload is not None:
            result.append(execution_payload)
            continue
        result.append(
            DecodeStepMetadataMaterializer(
                specs[bucket],
                physical_block_offsets=ALL_FAMILY_BLOCK_OFFSETS,
            ).materialize(step)
        )
    return tuple(result)


def _build_a3_stateful_churn_plan(
    *,
    spec_steps: int,
    execute_steps: int,
    native_precondition_steps: int,
    retained_stage_steps: int,
    seed: int,
    static_extent_overrides: A3StatefulStaticExtentOverrides | None,
) -> A3StatefulChurnPlan:
    """Build a plan after trace-length validation has selected all four axes."""
    # This runner initializes every newly admitted block to the empty-request
    # state.  Therefore admissions must start at position zero; non-zero decode
    # prefixes require real prefill-generated cache content and belong in a
    # separate preloaded-state boundary runner.
    spec_trace = build_deterministic_churn_trace(spec_steps, seed=seed, admission_prefixes=(0,))
    trace = (
        spec_trace
        if execute_steps == spec_steps
        else DecodeTrace(seed=spec_trace.seed, steps=spec_trace.steps[:execute_steps])
    )
    requirements = derive_trace_metadata_requirements(
        spec_trace,
        physical_block_offsets=ALL_FAMILY_BLOCK_OFFSETS,
    )
    overrides = dict(requirements.program_spec_overrides)
    if static_extent_overrides is not None:
        overrides = static_extent_overrides.apply(overrides)
    specs = {
        bucket: DecodeCSAProgramSpec(batch=bucket, **overrides) for bucket in sorted(set(spec_trace.bucket_sequence))
    }
    materialized_steps = max(execute_steps, native_precondition_steps, retained_stage_steps)
    materialized_trace = (
        spec_trace
        if materialized_steps == spec_steps
        else DecodeTrace(seed=spec_trace.seed, steps=spec_trace.steps[:materialized_steps])
    )
    materialized_payloads = materialize_decode_trace(
        materialized_trace,
        specs,
        physical_block_offsets=ALL_FAMILY_BLOCK_OFFSETS,
    )
    initialization_trace_steps = max(execute_steps, native_precondition_steps)
    initialization_trace = (
        spec_trace
        if initialization_trace_steps == spec_steps
        else DecodeTrace(seed=spec_trace.seed, steps=spec_trace.steps[:initialization_trace_steps])
    )
    materialized_initializations = _build_initialization_plan(initialization_trace)
    payloads = materialized_payloads[:execute_steps]
    return A3StatefulChurnPlan(
        spec_trace=spec_trace,
        trace=trace,
        specs_by_bucket=MappingProxyType(specs),
        warmup_payloads=_first_warmup_payloads(spec_trace, specs, materialized_payloads),
        payloads=payloads,
        initializations=materialized_initializations[:execute_steps],
        native_precondition_payloads=materialized_payloads[:native_precondition_steps],
        native_precondition_initializations=materialized_initializations[:native_precondition_steps],
        retained_stage_payloads=materialized_payloads[:retained_stage_steps],
        physical_block_offsets=ALL_FAMILY_BLOCK_OFFSETS,
    )


def build_a3_stateful_churn_plan(
    *,
    seed: int,
    steps: int | None = None,
    spec_steps: int | None = None,
    execute_steps: int | None = None,
    native_precondition_steps: int | None = None,
    retained_stage_steps: int | None = None,
) -> A3StatefulChurnPlan:
    """Build immutable Host artifacts without opening an NPU.

    ``steps=N`` preserves the original contract and means both static extents
    and execution cover N trace steps.  In the explicit diagnostic form,
    ``spec_steps`` derives capacities/table widths, ``execute_steps`` selects
    the candidate/oracle prefix, ``native_precondition_steps`` may continue
    native-only execution beyond that prefix, and ``retained_stage_steps``
    controls how many staged trace metadata owners remain strongly held.
    """
    resolved = _resolve_trace_lengths(
        steps=steps,
        spec_steps=spec_steps,
        execute_steps=execute_steps,
        native_precondition_steps=native_precondition_steps,
        retained_stage_steps=retained_stage_steps,
        default_steps=None,
    )
    return _build_a3_stateful_churn_plan(
        spec_steps=resolved[0],
        execute_steps=resolved[1],
        native_precondition_steps=resolved[2],
        retained_stage_steps=resolved[3],
        seed=seed,
        static_extent_overrides=None,
    )


def build_a3_stateful_churn_diagnostic_plan(
    *,
    spec_steps: int,
    execute_steps: int,
    native_precondition_steps: int | None = None,
    retained_stage_steps: int | None = None,
    seed: int,
    static_extent_overrides: A3StatefulStaticExtentOverrides,
) -> A3StatefulChurnPlan:
    """Build a Host-only static-extent bisection plan.

    Overrides replace only named static extents after deriving the baseline
    from the full ``spec_steps`` trace.  Every selected execution and warmup
    payload is rematerialized against the resulting specialization, so an
    undersized diagnostic extent fails before any NPU is opened.
    """
    if not isinstance(static_extent_overrides, A3StatefulStaticExtentOverrides):
        raise TypeError("static_extent_overrides must be A3StatefulStaticExtentOverrides")
    resolved = _resolve_trace_lengths(
        steps=None,
        spec_steps=spec_steps,
        execute_steps=execute_steps,
        native_precondition_steps=native_precondition_steps,
        retained_stage_steps=retained_stage_steps,
        default_steps=None,
    )
    return _build_a3_stateful_churn_plan(
        spec_steps=resolved[0],
        execute_steps=resolved[1],
        native_precondition_steps=resolved[2],
        retained_stage_steps=resolved[3],
        seed=seed,
        static_extent_overrides=static_extent_overrides,
    )


def _state_tensor_by_name(raw_cache_tuple: Sequence[torch.Tensor]) -> Mapping[str, torch.Tensor]:
    if len(raw_cache_tuple) != len(STATE_NAMES):
        raise ValueError("state world must contain exactly six cache tensors")
    return MappingProxyType(dict(zip(STATE_NAMES, raw_cache_tuple, strict=True)))


def initialize_acquired_blocks(
    raw_cache_tuple: Sequence[torch.Tensor],
    initialization: StepBlockInitialization,
) -> None:
    """Initialize admission and survivor-growth blocks to empty-prefix state."""
    named = _state_tensor_by_name(raw_cache_tuple)
    for family, blocks in initialization.physical_blocks.items():
        for state_name in _TRACE_FAMILY_TO_STATE_NAMES[family]:
            tensor = named[state_name]
            for block in blocks:
                if block <= 0 or block >= tensor.shape[0]:
                    raise ValueError(
                        f"step {initialization.step_id}/{state_name}: physical block {block} "
                        f"outside [1, {tensor.shape[0]})"
                    )
                if state_name == "indexer_scale":
                    tensor[block].fill_(1.0)
                else:
                    tensor[block].zero_()


def expected_mutated_rows(
    payload: DecodeStepMetadataPayload,
    initialization: StepBlockInitialization,
) -> Mapping[str, tuple[int, ...]]:
    """Return the legal row set for lifecycle clears plus this decode launch."""
    if payload.step_id != initialization.step_id:
        raise ValueError("payload and initialization step IDs differ")
    target: dict[str, set[int]] = {name: set() for name in STATE_NAMES}
    for family, blocks in initialization.physical_blocks.items():
        rows_per_block = _ROWS_PER_BLOCK[family]
        for state_name in _TRACE_FAMILY_TO_STATE_NAMES[family]:
            for block in blocks:
                target[state_name].update(range(block * rows_per_block, (block + 1) * rows_per_block))

    step = payload.source_step
    for request in step.active_requests:
        for position in request.token_positions:
            mappings = (
                (COMPRESSOR_ATTENTION, position // 4, "compressed_kv"),
                (SWA_CACHE, position, "swa_kv"),
                (MAIN_COMPRESSOR_STATE, position, "compressor_state"),
                (INNER_COMPRESSOR_STATE, position, "indexer_compressor_state"),
                (INDEXER_CACHE, position // 4, "indexer_k"),
                (INDEXER_CACHE, position // 4, "indexer_scale"),
            )
            for family, logical_position, state_name in mappings:
                rows_per_block = _ROWS_PER_BLOCK[family]
                logical_block = logical_position // rows_per_block
                block_table = request.ownership_for(family).logical_block_table
                physical_block = (
                    block_table[logical_block] + payload.physical_block_offsets[CACHE_FAMILY_NAMES.index(family)]
                )
                target[state_name].add(physical_block * rows_per_block + logical_position % rows_per_block)
    return MappingProxyType({name: tuple(sorted(rows)) for name, rows in target.items()})


@dataclass(frozen=True, slots=True)
class StateRowDelta:
    shape: tuple[int, ...]
    dtype: torch.dtype
    row_indices: tuple[int, ...]
    row_values: torch.Tensor


def compute_state_deltas(
    previous: Mapping[str, torch.Tensor],
    current: Mapping[str, torch.Tensor],
) -> Mapping[str, StateRowDelta]:
    if tuple(previous) != STATE_NAMES or tuple(current) != STATE_NAMES:
        raise ValueError(f"snapshots must preserve state order {STATE_NAMES}")
    result = {}
    for name in STATE_NAMES:
        before = previous[name].detach().cpu()
        after = current[name].detach().cpu()
        if before.shape != after.shape or before.dtype != after.dtype:
            raise ValueError(f"{name}: snapshot shape/dtype changed")
        before_rows = before.reshape(-1, before.shape[-1])
        after_rows = after.reshape(-1, after.shape[-1])
        changed = after_rows.ne(before_rows).any(dim=1)
        indices = tuple(int(value) for value in torch.nonzero(changed, as_tuple=False).reshape(-1).tolist())
        result[name] = StateRowDelta(
            shape=tuple(after.shape),
            dtype=after.dtype,
            row_indices=indices,
            row_values=after_rows[changed].clone(),
        )
    return MappingProxyType(result)


def apply_state_deltas(
    snapshot: Mapping[str, torch.Tensor],
    deltas: Mapping[str, StateRowDelta],
) -> None:
    if tuple(snapshot) != STATE_NAMES or tuple(deltas) != STATE_NAMES:
        raise ValueError(f"snapshot/delta mappings must preserve state order {STATE_NAMES}")
    for name in STATE_NAMES:
        tensor = snapshot[name]
        delta = deltas[name]
        if tuple(tensor.shape) != delta.shape or tensor.dtype != delta.dtype:
            raise ValueError(f"{name}: delta does not belong to this snapshot")
        if delta.row_indices:
            tensor.reshape(-1, tensor.shape[-1])[list(delta.row_indices)] = delta.row_values


@dataclass(frozen=True, slots=True)
class StatefulFamilyComparison:
    values: TensorComparison
    native_changed_rows: tuple[int, ...]
    pypto_changed_rows: tuple[int, ...]
    legal_rows: tuple[int, ...]
    native_unexpected_rows: tuple[int, ...]
    pypto_unexpected_rows: tuple[int, ...]

    @property
    def close(self) -> bool:
        return (
            self.values.close
            and self.native_changed_rows == self.pypto_changed_rows
            and not self.native_unexpected_rows
            and not self.pypto_unexpected_rows
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "values": asdict(self.values),
            "native_changed_rows": self.native_changed_rows,
            "pypto_changed_rows": self.pypto_changed_rows,
            "legal_rows": self.legal_rows,
            "native_unexpected_rows": self.native_unexpected_rows,
            "pypto_unexpected_rows": self.pypto_unexpected_rows,
            "close": self.close,
        }


@dataclass(frozen=True, slots=True)
class A3StatefulChurnStepResult:
    step_id: int
    bucket: int
    actual: int
    request_ids: tuple[int, ...]
    admitted_request_ids: tuple[int, ...]
    retired_request_ids: tuple[int, ...]
    reuse_event_count: int
    output: TensorComparison
    states: Mapping[str, StatefulFamilyComparison]
    indexer_quantized: IndexerQuantizedComparison
    indexer_quantized_cumulative: IndexerQuantizedComparison
    native_padded_output_nonzero: int
    pypto_padded_output_nonzero: int
    native_block0_mismatches: Mapping[str, int]
    pypto_block0_mismatches: Mapping[str, int]

    @property
    def close(self) -> bool:
        non_quantized = all(
            comparison.close for name, comparison in self.states.items() if name not in {"indexer_k", "indexer_scale"}
        )
        quantized_write_sets = all(
            self.states[name].native_changed_rows == self.states[name].pypto_changed_rows
            and not self.states[name].native_unexpected_rows
            and not self.states[name].pypto_unexpected_rows
            for name in ("indexer_k", "indexer_scale")
        )
        return bool(
            self.output.close
            and non_quantized
            and quantized_write_sets
            and self.indexer_quantized.acceptable
            and self.pypto_padded_output_nonzero == 0
            and not any(self.native_block0_mismatches.values())
            and not any(self.pypto_block0_mismatches.values())
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "bucket": self.bucket,
            "actual": self.actual,
            "request_ids": self.request_ids,
            "admitted_request_ids": self.admitted_request_ids,
            "retired_request_ids": self.retired_request_ids,
            "reuse_event_count": self.reuse_event_count,
            "output": asdict(self.output),
            "states": {name: value.to_dict() for name, value in self.states.items()},
            "indexer_quantized": self.indexer_quantized.to_dict(),
            "indexer_quantized_cumulative": self.indexer_quantized_cumulative.to_dict(),
            "native_padded_output_nonzero": self.native_padded_output_nonzero,
            "pypto_padded_output_nonzero": self.pypto_padded_output_nonzero,
            "native_block0_mismatches": dict(self.native_block0_mismatches),
            "pypto_block0_mismatches": dict(self.pypto_block0_mismatches),
            "close": self.close,
        }


@dataclass(frozen=True, slots=True)
class A3StatefulChurnResult:
    runtime: str
    device: int
    trace_seed: int
    spec_steps: int
    requested_steps: int
    native_precondition_steps: int
    retained_stage_steps: int
    pack_weights_before_native: bool
    buckets: tuple[int, ...]
    reuse_event_count: int
    pypto_sacrificial_bucket_warmups_reset: bool
    steps: tuple[A3StatefulChurnStepResult, ...]
    native_final_page_padding_mismatches: Mapping[str, int]
    pypto_final_page_padding_mismatches: Mapping[str, int]

    @property
    def close(self) -> bool:
        return bool(
            len(self.steps) == self.requested_steps
            and all(step.close for step in self.steps)
            and not any(self.native_final_page_padding_mismatches.values())
            and not any(self.pypto_final_page_padding_mismatches.values())
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "runtime": self.runtime,
            "device": self.device,
            "trace_seed": self.trace_seed,
            "spec_steps": self.spec_steps,
            "execute_steps": self.requested_steps,
            "requested_steps": self.requested_steps,
            "native_precondition_steps": self.native_precondition_steps,
            "retained_stage_steps": self.retained_stage_steps,
            "pack_weights_before_native": self.pack_weights_before_native,
            "buckets": self.buckets,
            "reuse_event_count": self.reuse_event_count,
            "pypto_sacrificial_bucket_warmups_reset": self.pypto_sacrificial_bucket_warmups_reset,
            "steps": tuple(step.to_dict() for step in self.steps),
            "native_final_page_padding_mismatches": dict(self.native_final_page_padding_mismatches),
            "pypto_final_page_padding_mismatches": dict(self.pypto_final_page_padding_mismatches),
            "close": self.close,
        }


@dataclass(frozen=True, slots=True)
class _TraceMetadataBundle:
    metadata: tuple[Any, ...]
    by_name: Mapping[str, Any]
    owners: tuple[Any, ...]


@dataclass(frozen=True, slots=True)
class _NativeStepObservation:
    active_output: torch.Tensor
    padded_output_nonzero: int
    deltas: Mapping[str, StateRowDelta]
    block0_mismatches: Mapping[str, int]


def _weight_pack_inputs(metadata: _TraceMetadataBundle) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract and validate the three metadata-owned static packing inputs."""
    request = metadata.metadata[0].req_metadata
    values = (
        request.full_compress_cos,
        request.full_compress_sin,
        metadata.metadata[3].hadamard,
    )
    if not all(isinstance(value, torch.Tensor) for value in values):
        raise A3StatefulChurnCapabilityError("real DSA metadata did not expose full RoPE and Hadamard tensors")
    return values


def _block0_mismatches(
    raw_cache_tuple: Sequence[torch.Tensor],
    expected: Sequence[torch.Tensor],
) -> Mapping[str, int]:
    if len(raw_cache_tuple) != len(STATE_NAMES) or len(expected) != len(STATE_NAMES):
        raise ValueError("block-0 canary snapshots must cover all six state families")
    return MappingProxyType(
        {
            name: int(torch.count_nonzero(tensor[0].detach().cpu() != canary))
            for name, tensor, canary in zip(STATE_NAMES, raw_cache_tuple, expected, strict=True)
        }
    )


def _restore_logical_state(
    raw_cache_tuple: Sequence[torch.Tensor],
    pristine: Mapping[str, torch.Tensor],
) -> None:
    for name, tensor in zip(STATE_NAMES, raw_cache_tuple, strict=True):
        tensor.copy_(pristine[name])


def _host_hidden_for_step(payload: DecodeStepMetadataPayload, *, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + payload.step_id * 104729)
    hidden = torch.zeros((payload.spec.tokens, FLASH.hidden_size), dtype=torch.bfloat16)
    active_tokens = payload.actual_tokens
    hidden[:active_tokens].copy_(
        torch.randn(
            (active_tokens, FLASH.hidden_size),
            dtype=torch.bfloat16,
            generator=generator,
        ).mul_(0.125)
    )
    return hidden


def _trace_world_from_staged(base_world: Any, staged: StagedDecodeStepMetadataPayload) -> Any:
    """View one persistent cache world through the current trace metadata."""
    from tests.pypto_dsv4_decode_csa.fixtures import (
        _native_swa_slot_mapping,
        _paged_slot_mapping,
        _position_rows,
        _state_metadata,
    )
    from tests.pypto_dsv4_decode_csa.native_fixture import DecodeCSAStateWorld

    spec = staged.spec
    starts = staged.host_payload.start_positions
    positions_cpu = _position_rows(starts, seq=spec.seq)
    host_tables = {
        "compressed": staged.host_payload.block_tables["cmp_block_table"],
        "main_state": staged.host_payload.block_tables["compress_state_block_table"],
        "inner_state": staged.host_payload.block_tables["inner_compress_state_block_table"],
        "indexer": staged.host_payload.block_tables["idx_block_table"],
        "swa": staged.host_payload.block_tables["swa_block_table"],
    }
    host_tables = {name: torch.tensor(rows, dtype=torch.int32) for name, rows in host_tables.items()}
    native_swa_slots = _native_swa_slot_mapping(
        positions=positions_cpu,
        block_table=host_tables["swa"],
    ).to(staged.device)
    linear_slots = {
        "compressed": _paged_slot_mapping(
            positions=positions_cpu,
            block_table=host_tables["compressed"],
            position_divisor=4,
        ).to(dtype=torch.int32, device=staged.device),
        "main_state": _state_metadata(
            positions=positions_cpu,
            block_table=host_tables["main_state"],
            state_block_size=2,
        ).to(dtype=torch.int32, device=staged.device),
        "inner_state": _state_metadata(
            positions=positions_cpu,
            block_table=host_tables["inner_state"],
            state_block_size=2,
        ).to(dtype=torch.int32, device=staged.device),
        "indexer": _paged_slot_mapping(
            positions=positions_cpu,
            block_table=host_tables["indexer"],
            position_divisor=4,
        ).to(dtype=torch.int32, device=staged.device),
        "swa": (native_swa_slots[:, 0] * BLOCK_SIZE + native_swa_slots[:, 1]).contiguous(),
    }
    device_tables = {
        "compressed": staged.block_tables["cmp_block_table"],
        "main_state": staged.block_tables["compress_state_block_table"],
        "inner_state": staged.block_tables["inner_compress_state_block_table"],
        "indexer": staged.block_tables["idx_block_table"],
        "swa": staged.block_tables["swa_block_table"],
    }
    return DecodeCSAStateWorld(
        spec=spec,
        raw_cache_tuple=base_world.raw_cache_tuple,
        storage_owners=base_world.storage_owners,
        block_tables=MappingProxyType(device_tables),
        positions=positions_cpu.to(staged.device),
        start_positions=staged.start_positions,
        seq_lens=staged.kv_seq_lens,
        query_start_loc=staged.query_start_loc,
        swa_slot_mapping=native_swa_slots,
        linear_slot_mappings=MappingProxyType(linear_slots),
    )


def _build_real_trace_metadata(
    wrapper: Any,
    world: Any,
    payload: DecodeStepMetadataPayload,
    vllm_config: Any,
) -> _TraceMetadataBundle:
    """Build the full native SAS/QLI metadata for an A<B trace step."""
    try:
        from vllm_ascend.attention.attention_v1 import AscendAttentionState
        from vllm_ascend.attention.dsa_v1 import AscendDSAMetadataBuilder
        from vllm_ascend.attention.utils import AscendCommonAttentionMetadata
    except (ImportError, AttributeError) as error:
        raise A3StatefulChurnCapabilityError(
            "installed vLLM-Ascend lacks the production DSA metadata builder required by the churn runner"
        ) from error

    from tests.pypto_dsv4_decode_csa.native_fixture import metadata_family_names

    names = metadata_family_names(wrapper.prefix)
    cache_modules = (
        wrapper.dsa_attn,
        wrapper.compressor.state_cache,
        wrapper.indexer.compressor.state_cache,
        wrapper.indexer.k_cache,
        wrapper.swa_cache_layer,
    )
    table_families = ("compressed", "main_state", "inner_state", "indexer", "swa")
    shared_metadata: dict[str, Any] = {}
    metadata = []
    owners = []
    starts_cpu = world.start_positions.detach().cpu()
    query_cpu = world.query_start_loc.detach().cpu()
    seq_lens_cpu = world.seq_lens.detach().cpu()
    for name, cache_module, family in zip(names, cache_modules, table_families, strict=True):
        slot_mapping = world.linear_slot_mappings["swa"] if family == "swa" else world.linear_slot_mappings[family]
        common = AscendCommonAttentionMetadata(
            query_start_loc=world.query_start_loc,
            query_start_loc_cpu=query_cpu,
            seq_lens=world.seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            _seq_lens_cpu=seq_lens_cpu,
            _num_computed_tokens_cpu=starts_cpu,
            num_computed_tokens_cpu=starts_cpu,
            num_reqs=world.spec.batch,
            num_actual_tokens=payload.actual_tokens,
            max_query_len=world.spec.seq,
            max_seq_len=max(payload.kv_seq_lens),
            block_table_tensor=world.block_tables[family],
            slot_mapping=slot_mapping,
            causal=True,
            positions=world.positions.reshape(-1),
            positions_cpu=world.positions.detach().cpu().reshape(-1),
            is_prefilling=torch.zeros(world.spec.batch, dtype=torch.bool, device="cpu"),
            decode_token_per_req=world.spec.seq,
            actual_seq_lengths_q=[world.spec.seq] * world.spec.batch,
            attn_state=AscendAttentionState.DecodeOnly,
            graph_pad_size=-1,
            num_input_tokens=world.spec.tokens,
        )
        cache_spec = cache_module.get_kv_cache_spec(vllm_config)
        builder = AscendDSAMetadataBuilder(cache_spec, [name], vllm_config, world.positions.device)
        builder.decode_threshold = world.spec.seq
        built = builder.build(
            common_prefix_len=0,
            common_attn_metadata=common,
            block_size=cache_spec.block_size,
            common_ratio_to_sas_metadata=shared_metadata,
            num_reqs_actual=payload.actual,
        )
        metadata.append(built)
        owners.extend((builder, common))
    metadata_tuple = tuple(metadata)
    return _TraceMetadataBundle(
        metadata=metadata_tuple,
        by_name=MappingProxyType(dict(zip(names, metadata_tuple, strict=True))),
        owners=tuple(owners),
    )


def _minimal_trace_metadata(
    wrapper: Any,
    staged: StagedDecodeStepMetadataPayload,
    *,
    full_rope_cos: torch.Tensor,
    full_rope_sin: torch.Tensor,
    hadamard: torch.Tensor,
) -> _TraceMetadataBundle:
    from tests.pypto_dsv4_decode_csa.native_fixture import metadata_family_names

    metadata = build_partial_bucket_native_metadata(
        staged,
        full_rope_cos=full_rope_cos,
        full_rope_sin=full_rope_sin,
        hadamard=hadamard,
    )
    names = metadata_family_names(wrapper.prefix)
    return _TraceMetadataBundle(
        metadata=metadata,
        by_name=MappingProxyType(dict(zip(names, metadata, strict=True))),
        owners=(staged, full_rope_cos, full_rope_sin, hadamard),
    )


def _compare_step_states(
    native_state: Mapping[str, torch.Tensor],
    pypto_state: Mapping[str, torch.Tensor],
    initial_state: Mapping[str, torch.Tensor],
    native_deltas: Mapping[str, StateRowDelta],
    pypto_deltas: Mapping[str, StateRowDelta],
    legal_rows: Mapping[str, tuple[int, ...]],
    *,
    atol: float,
    rtol: float,
) -> tuple[
    Mapping[str, StatefulFamilyComparison],
    IndexerQuantizedComparison,
    IndexerQuantizedComparison,
]:
    values = compare_state_snapshots(native_state, pypto_state, initial_state, atol=atol, rtol=rtol)
    result = {}
    for name in STATE_NAMES:
        legal = set(legal_rows[name])
        native_rows = native_deltas[name].row_indices
        pypto_rows = pypto_deltas[name].row_indices
        result[name] = StatefulFamilyComparison(
            values=values[name],
            native_changed_rows=native_rows,
            pypto_changed_rows=pypto_rows,
            legal_rows=legal_rows[name],
            native_unexpected_rows=tuple(sorted(set(native_rows) - legal)),
            pypto_unexpected_rows=tuple(sorted(set(pypto_rows) - legal)),
        )
    native_quantized_rows = tuple(
        sorted(set(native_deltas["indexer_k"].row_indices) | set(native_deltas["indexer_scale"].row_indices))
    )
    pypto_quantized_rows = tuple(
        sorted(set(pypto_deltas["indexer_k"].row_indices) | set(pypto_deltas["indexer_scale"].row_indices))
    )
    indexer_quantized = compare_indexer_quantized(
        native_state["indexer_k"],
        pypto_state["indexer_k"],
        initial_state["indexer_k"],
        native_state["indexer_scale"],
        pypto_state["indexer_scale"],
        initial_state["indexer_scale"],
        reference_write_row_indices=native_quantized_rows,
        candidate_write_row_indices=pypto_quantized_rows,
    )
    indexer_quantized_cumulative = compare_indexer_quantized(
        native_state["indexer_k"],
        pypto_state["indexer_k"],
        initial_state["indexer_k"],
        native_state["indexer_scale"],
        pypto_state["indexer_scale"],
        initial_state["indexer_scale"],
    )
    return MappingProxyType(result), indexer_quantized, indexer_quantized_cumulative


def run_a3_stateful_churn_compare(
    *,
    runtime: str,
    device: int = 0,
    steps: int | None = None,
    spec_steps: int | None = None,
    execute_steps: int | None = None,
    native_precondition_steps: int | None = None,
    retained_stage_steps: int | None = None,
    pack_weights_before_native: bool = False,
    trace_seed: int = 20260903,
    weight_seed: int = 20260902,
    atol: float = 0.1,
    rtol: float = 0.1,
    master_port: int = 29731,
) -> A3StatefulChurnResult:
    """Execute native first and PyPTO second against equal independent worlds."""
    if runtime not in SUPPORTED_RUNTIMES:
        raise ValueError(f"runtime must be one of {SUPPORTED_RUNTIMES}, got {runtime!r}")
    if isinstance(device, bool) or not isinstance(device, int) or device < 0:
        raise ValueError("device must be a non-negative integer")
    if atol < 0.0 or rtol < 0.0:
        raise ValueError("atol and rtol must be non-negative")
    if not isinstance(pack_weights_before_native, bool):
        raise TypeError("pack_weights_before_native must be a bool")
    resolved = _resolve_trace_lengths(
        steps=steps,
        spec_steps=spec_steps,
        execute_steps=execute_steps,
        native_precondition_steps=native_precondition_steps,
        retained_stage_steps=retained_stage_steps,
        default_steps=32,
    )
    plan = _build_a3_stateful_churn_plan(
        spec_steps=resolved[0],
        execute_steps=resolved[1],
        native_precondition_steps=resolved[2],
        retained_stage_steps=resolved[3],
        seed=trace_seed,
        static_extent_overrides=None,
    )

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
        pack_decode_csa_weights,
        uninstall_pypto_dsv4_decode_csa,
    )
    from vllm_ascend.utils import register_ascend_customop

    config_owner = build_synthetic_vllm_config(batch=max(BATCH_BUCKETS))
    registry = DecodeCSADeviceOwnerRegistry()
    owner = None
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
            base_spec = plan.specs_by_bucket[max(plan.buckets)]
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

            retained_staged_payloads = tuple(
                StagedDecodeStepMetadataPayload.from_host(payload, device=target)
                for payload in plan.retained_stage_payloads
            )
            retained_staged_by_step = {
                payload.step_id: staged
                for payload, staged in zip(
                    plan.retained_stage_payloads,
                    retained_staged_payloads,
                    strict=True,
                )
            }
            warmup_staged_payloads = tuple(
                retained_staged_by_step.get(payload.step_id)
                or StagedDecodeStepMetadataPayload.from_host(payload, device=target)
                for payload in plan.warmup_payloads
            )

            # Native runs to completion before PyPTO is installed.  Each step
            # stores oracle data only for the candidate prefix.  It may then
            # continue as a pure device/process precondition without falsely
            # extending the candidate result.  Trace metadata beyond the
            # retained prefix is staged one step at a time and released after
            # the explicit sync boundary.
            bind_state_world(wrapper, native_base)
            prepared_weights = None
            if pack_weights_before_native:
                prepack_payload = plan.native_precondition_payloads[0]
                prepack_staged = retained_staged_by_step[prepack_payload.step_id]
                prepack_world = _trace_world_from_staged(native_base, prepack_staged)
                prepack_metadata = _build_real_trace_metadata(
                    wrapper,
                    prepack_world,
                    prepack_payload,
                    config_owner.vllm_config,
                )
                full_rope_cos, full_rope_sin, hadamard = _weight_pack_inputs(prepack_metadata)
                prepared_weights = pack_decode_csa_weights(
                    wrapper,
                    full_rope_cos=full_rope_cos,
                    full_rope_sin=full_rope_sin,
                    hadamard=hadamard,
                )
                torch_npu.npu.synchronize(device)
                del prepack_metadata, prepack_world

            native_previous = native_initial
            native_observations = []
            for payload, initialization in zip(
                plan.native_precondition_payloads,
                plan.native_precondition_initializations,
                strict=True,
            ):
                staged = retained_staged_by_step.get(payload.step_id)
                transient_stage = staged is None
                if staged is None:
                    staged = StagedDecodeStepMetadataPayload.from_host(payload, device=target)
                initialize_acquired_blocks(native_base.raw_cache_tuple, initialization)
                world = _trace_world_from_staged(native_base, staged)
                metadata = _build_real_trace_metadata(
                    wrapper,
                    world,
                    payload,
                    config_owner.vllm_config,
                )
                context = _forward_context(wrapper, metadata)
                # Native o_proj sizes its temporary/output assignment from the
                # padded scheduler extent, while metadata.num_actual_tokens
                # limits attention/cache mutation to A*S. Native padded output
                # values are diagnostic only; PyPTO is held to exact zero.
                context.num_tokens = payload.spec.tokens
                hidden = _host_hidden_for_step(payload, seed=trace_seed).to(target)
                output = torch.zeros_like(hidden)
                _custom_op_call(wrapper, context, hidden, output)
                torch_npu.npu.synchronize(device)
                if payload.step_id < plan.execute_steps:
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
                if transient_stage:
                    del context, metadata, world, staged
            native_canaries = inspect_partial_bucket_canaries(
                native_base.raw_cache_tuple,
                native_base.storage_owners,
            )

            # Build one full real metadata bundle for installation-time RoPE,
            # Hadamard and uniform-query validation.  Subsequent PyPTO calls use
            # the exact staged production tensors without rebuilding native
            # scheduler metadata.
            bind_state_world(wrapper, pypto_base)
            install_staged = retained_staged_by_step[plan.payloads[0].step_id]
            install_world = _trace_world_from_staged(pypto_base, install_staged)
            install_metadata = _build_real_trace_metadata(
                wrapper,
                install_world,
                plan.payloads[0],
                config_owner.vllm_config,
            )
            owner = install_pypto_dsv4_decode_csa(
                wrapper,
                kv_cache=pypto_base.raw_cache_tuple,
                attn_metadata=install_metadata.metadata,
                device=device,
                runtime=runtime,
                batch_buckets=plan.buckets,
                registry=registry,
                prepared_weights=prepared_weights,
                uniform_query_rows_contract=True,
            )
            full_rope_cos, full_rope_sin, hadamard = _weight_pack_inputs(install_metadata)

            for payload, staged in zip(plan.warmup_payloads, warmup_staged_payloads, strict=True):
                metadata = _minimal_trace_metadata(
                    wrapper,
                    staged,
                    full_rope_cos=full_rope_cos,
                    full_rope_sin=full_rope_sin,
                    hadamard=hadamard,
                )
                context = _forward_context(wrapper, metadata)
                hidden = _host_hidden_for_step(payload, seed=trace_seed).to(target)
                output = torch.zeros_like(hidden)
                _custom_op_call(wrapper, context, hidden, output)
            torch_npu.npu.synchronize(device)
            _restore_logical_state(pypto_base.raw_cache_tuple, pypto_initial)
            torch_npu.npu.synchronize(device)
            warmup_canaries = inspect_partial_bucket_canaries(
                pypto_base.raw_cache_tuple,
                pypto_base.storage_owners,
            )
            if not warmup_canaries.clean:
                raise AssertionError(
                    f"sacrificial bucket warmup corrupted reserved cache canaries: {warmup_canaries.to_dict()}"
                )

            native_reconstructed = {name: value.clone() for name, value in native_initial.items()}
            pypto_previous = pypto_initial
            step_results = []
            for payload, initialization, native in zip(
                plan.payloads,
                plan.initializations,
                native_observations,
                strict=True,
            ):
                staged = retained_staged_by_step.get(payload.step_id)
                transient_stage = staged is None
                if staged is None:
                    staged = StagedDecodeStepMetadataPayload.from_host(payload, device=target)
                initialize_acquired_blocks(pypto_base.raw_cache_tuple, initialization)
                metadata = _minimal_trace_metadata(
                    wrapper,
                    staged,
                    full_rope_cos=full_rope_cos,
                    full_rope_sin=full_rope_sin,
                    hadamard=hadamard,
                )
                context = _forward_context(wrapper, metadata)
                hidden = _host_hidden_for_step(payload, seed=trace_seed).to(target)
                output = torch.zeros_like(hidden)
                _custom_op_call(wrapper, context, hidden, output)
                torch_npu.npu.synchronize(device)
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
                active_output = output[: payload.actual_tokens].detach().cpu()
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
                        output,
                        actual=payload.actual,
                        seq=payload.spec.seq,
                    ),
                    native_block0_mismatches=native.block0_mismatches,
                    pypto_block0_mismatches=_block0_mismatches(pypto_base.raw_cache_tuple, pypto_block0),
                )
                step_results.append(step_result)
                if not step_result.close:
                    raise AssertionError(json.dumps(step_result.to_dict(), ensure_ascii=False, indent=2))
                pypto_previous = pypto_current
                if transient_stage:
                    del context, metadata, staged

            pypto_canaries = inspect_partial_bucket_canaries(
                pypto_base.raw_cache_tuple,
                pypto_base.storage_owners,
            )
            result = A3StatefulChurnResult(
                runtime=runtime,
                device=device,
                trace_seed=trace_seed,
                spec_steps=plan.spec_steps,
                requested_steps=plan.execute_steps,
                native_precondition_steps=plan.native_precondition_steps,
                retained_stage_steps=plan.retained_stage_steps,
                pack_weights_before_native=pack_weights_before_native,
                buckets=plan.buckets,
                reuse_event_count=plan.reuse_event_count,
                pypto_sacrificial_bucket_warmups_reset=True,
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
                if owner is not None:
                    uninstall_pypto_dsv4_decode_csa(owner.layer)
                    owner.device_owner.backend.close()
        finally:
            config_owner.close()
            if tp_initialized:
                _destroy_tp1()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", choices=SUPPORTED_RUNTIMES, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--steps",
        type=int,
        choices=SUPPORTED_SPEC_STEPS,
        default=None,
        help="legacy shorthand: derive static specs and execute the same number of steps",
    )
    parser.add_argument(
        "--spec-steps",
        type=int,
        choices=SUPPORTED_SPEC_STEPS,
        default=None,
        help="trace length used to derive static cache capacities and table widths",
    )
    parser.add_argument(
        "--execute-steps",
        type=int,
        choices=SUPPORTED_EXECUTE_STEPS,
        default=None,
        help="exact candidate prefix of the spec trace to execute and compare",
    )
    parser.add_argument(
        "--native-precondition-steps",
        type=int,
        choices=SUPPORTED_NATIVE_PRECONDITION_STEPS,
        default=None,
        help="native prefix to run before PyPTO; must cover execute-steps",
    )
    parser.add_argument(
        "--retained-stage-steps",
        type=int,
        choices=SUPPORTED_RETAINED_STAGE_STEPS,
        default=None,
        help="trace metadata prefix staged and strongly retained across both phases",
    )
    parser.add_argument(
        "--pack-weights-before-native",
        action="store_true",
        help="diagnostic: pack PyPTO weights before, rather than after, the native precondition phase",
    )
    parser.add_argument("--trace-seed", type=int, default=20260903)
    parser.add_argument("--weight-seed", type=int, default=20260902)
    parser.add_argument("--atol", type=float, default=0.1)
    parser.add_argument("--rtol", type=float, default=0.1)
    parser.add_argument("--master-port", type=int, default=29731)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_a3_stateful_churn_compare(
        runtime=args.runtime,
        device=args.device,
        steps=args.steps,
        spec_steps=args.spec_steps,
        execute_steps=args.execute_steps,
        native_precondition_steps=args.native_precondition_steps,
        retained_stage_steps=args.retained_stage_steps,
        pack_weights_before_native=args.pack_weights_before_native,
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
    "ALL_FAMILY_BLOCK_OFFSETS",
    "A3StatefulChurnCapabilityError",
    "A3StatefulChurnPlan",
    "A3StatefulChurnResult",
    "A3StatefulChurnStepResult",
    "A3StatefulStaticExtentOverrides",
    "BlockReuseEvent",
    "SUPPORTED_EXECUTE_STEPS",
    "SUPPORTED_NATIVE_PRECONDITION_STEPS",
    "SUPPORTED_RETAINED_STAGE_STEPS",
    "SUPPORTED_SPEC_STEPS",
    "StateRowDelta",
    "StatefulFamilyComparison",
    "StepBlockInitialization",
    "apply_state_deltas",
    "build_a3_stateful_churn_diagnostic_plan",
    "build_a3_stateful_churn_plan",
    "build_parser",
    "compute_state_deltas",
    "expected_mutated_rows",
    "initialize_acquired_blocks",
    "main",
    "run_a3_stateful_churn_compare",
]
