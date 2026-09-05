# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""A3 partial-bucket smoke through the production DSA custom-op boundary.

The public L1 callable remains specialized for a static batch bucket ``B``.
Only ``A`` request rows are active on a particular decode step.  Capture binds
one stable set of full-``B`` device metadata buffers; replay changes ``A`` by
copying a different payload into those buffers without calling Python dispatch
or capturing another graph.

Everything except :func:`run_a3_partial_bucket_smoke` is Host-testable.  The
runner imports ``torch_npu`` and the production custom op lazily so importing
this module never opens an NPU context.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from types import MappingProxyType, SimpleNamespace

import torch

from vllm_ascend.ops._pypto_dsv4_csa import DecodeCSAProgramSpec
from vllm_ascend.ops._pypto_dsv4_csa.config import BLOCK_SIZE, FLASH

_TABLE_ARGUMENTS = (
    "cmp_block_table",
    "compress_state_block_table",
    "inner_compress_state_block_table",
    "idx_block_table",
    "swa_block_table",
)
_BLOCK_SIZES = (BLOCK_SIZE, 2, 2, BLOCK_SIZE, BLOCK_SIZE)
_CACHE_FAMILY_NAMES = (
    "compressed_kv",
    "swa_kv",
    "main_compressor_state",
    "inner_compressor_state",
    "indexer_k",
    "indexer_scale",
)
_CACHE_CANARIES = (11.0, 13.0, 17.0, 19.0, 23, 29.0)
_BOUNDARY_START_POSITIONS = (
    31,
    32,
    63,
    64,
    95,
    96,
    127,
    128,
    159,
    160,
    191,
    192,
    223,
    224,
    255,
    256,
)
_OUTPUT_CANARY = 37.0


def _validate_actual(*, actual: int, bucket: int) -> None:
    if isinstance(actual, bool) or not isinstance(actual, int):
        raise TypeError(f"actual request count must be an integer, got {actual!r}")
    if not 1 <= actual <= bucket:
        raise ValueError(f"actual request count must satisfy 1 <= A <= B={bucket}, got {actual}")


def _table_contracts(spec: DecodeCSAProgramSpec) -> tuple[tuple[str, int, int], ...]:
    return (
        ("cmp_block_table", spec.compressed_table_width, spec.compressed_blocks),
        ("compress_state_block_table", spec.main_state_table_width, spec.main_state_blocks),
        ("inner_compress_state_block_table", spec.inner_state_table_width, spec.inner_state_blocks),
        ("idx_block_table", spec.indexer_table_width, spec.indexer_blocks),
        ("swa_block_table", spec.swa_table_width, spec.swa_blocks),
    )


def _block_zero_safe_table(
    *,
    bucket: int,
    actual: int,
    width: int,
    physical_blocks: int,
) -> torch.Tensor:
    """Partition non-zero physical blocks and leave all padded rows at zero."""
    blocks_per_request = (physical_blocks - 1) // bucket
    if blocks_per_request <= 0:
        raise ValueError(
            "partial-bucket canary requires one non-zero block partition per "
            f"bucket row, got B={bucket}, physical_blocks={physical_blocks}"
        )
    table = torch.zeros((bucket, width), dtype=torch.int32)
    logical = torch.arange(width, dtype=torch.int64).remainder(blocks_per_request)
    for request_index in range(actual):
        base = 1 + request_index * blocks_per_request
        table[request_index].copy_((logical + base).to(torch.int32))
    return table


@dataclass(frozen=True, slots=True)
class PartialBucketPayload:
    """One immutable description of full-B metadata contents for active A."""

    spec: DecodeCSAProgramSpec
    actual: int
    query_start_loc: torch.Tensor
    block_tables: Mapping[str, torch.Tensor]
    start_positions: torch.Tensor
    kv_seq_lens: torch.Tensor

    @property
    def actual_tokens(self) -> int:
        return self.actual * self.spec.seq

    @property
    def device(self) -> torch.device:
        return self.start_positions.device


def build_partial_bucket_payload(
    spec: DecodeCSAProgramSpec,
    *,
    actual: int,
    device: torch.device | str = "cpu",
    start_positions: Sequence[int] | None = None,
) -> PartialBucketPayload:
    """Build deterministic B-shaped metadata with an explicit invalid tail.

    Active table rows only reference non-zero physical blocks.  Every padded
    table row, start position and sequence length is zero, which makes block 0
    a canary for a missing device-side active-row guard.
    """
    _validate_actual(actual=actual, bucket=spec.batch)
    target = torch.device(device)
    if start_positions is None:
        starts = _BOUNDARY_START_POSITIONS[: spec.batch]
    else:
        starts = tuple(int(value) for value in start_positions)
    if len(starts) != spec.batch:
        raise ValueError(f"expected B={spec.batch} start positions, got {len(starts)}")
    if min(starts) < 0 or max(starts) + spec.seq > FLASH.max_position_embeddings:
        raise ValueError(f"start positions out of range: {starts}")

    tables = {
        name: _block_zero_safe_table(
            bucket=spec.batch,
            actual=actual,
            width=width,
            physical_blocks=physical_blocks,
        ).to(target)
        for name, width, physical_blocks in _table_contracts(spec)
    }
    active_starts = torch.tensor(starts[:actual], dtype=torch.int32)
    padded_rows = spec.batch - actual
    full_starts = torch.cat((active_starts, torch.zeros(padded_rows, dtype=torch.int32)))
    active_seq_lens = active_starts + spec.seq
    full_seq_lens = torch.cat((active_seq_lens, torch.zeros(padded_rows, dtype=torch.int32)))
    query_start_loc = torch.arange(
        0,
        spec.tokens + 1,
        spec.seq,
        dtype=torch.int32,
    )
    return PartialBucketPayload(
        spec=spec,
        actual=actual,
        query_start_loc=query_start_loc.to(target),
        block_tables=MappingProxyType(tables),
        start_positions=full_starts.to(target),
        kv_seq_lens=full_seq_lens.to(target),
    )


def build_partial_bucket_native_metadata(
    payload: PartialBucketPayload,
    *,
    full_rope_cos: torch.Tensor,
    full_rope_sin: torch.Tensor,
    hadamard: torch.Tensor,
) -> tuple[object, ...]:
    """Expose one payload in the five-family production metadata shape."""
    families: list[object] = []
    for index, (table_name, block_size) in enumerate(zip(_TABLE_ARGUMENTS, _BLOCK_SIZES, strict=True)):
        request = SimpleNamespace(
            block_table=payload.block_tables[table_name],
            seq_lens=payload.kv_seq_lens,
            slot_mapping=None,
            block_size=block_size,
            query_start_loc=payload.query_start_loc,
            start_pos=payload.start_positions if index == 0 else None,
            num_reqs_actual=payload.actual,
            full_compress_cos=full_rope_cos if index == 0 else None,
            full_compress_sin=full_rope_sin if index == 0 else None,
        )
        families.append(
            SimpleNamespace(
                num_actual_tokens=payload.actual_tokens,
                num_decodes=payload.spec.batch,
                num_decode_tokens=payload.actual_tokens,
                num_prefills=0,
                req_metadata=request,
                hadamard=hadamard if index == 3 else None,
            )
        )
    return tuple(families)


def _clone_payload_buffers(payload: PartialBucketPayload) -> PartialBucketPayload:
    return PartialBucketPayload(
        spec=payload.spec,
        actual=payload.actual,
        query_start_loc=payload.query_start_loc.clone(),
        block_tables=MappingProxyType({name: tensor.clone() for name, tensor in payload.block_tables.items()}),
        start_positions=payload.start_positions.clone(),
        kv_seq_lens=payload.kv_seq_lens.clone(),
    )


@dataclass(slots=True)
class PartialBucketMetadataOwner:
    """Own the stable metadata addresses captured by one static-B graph.

    Python counters in ``metadata`` describe the capture call only.  Replay
    never invokes the binder again; :meth:`apply_device_payload` updates the
    device tables/start/sequence buffers in place and deliberately leaves those
    Host counters untouched.
    """

    bound_payload: PartialBucketPayload
    metadata: tuple[object, ...]
    source_owners: tuple[object, ...]

    @classmethod
    def create(
        cls,
        payload: PartialBucketPayload,
        *,
        full_rope_cos: torch.Tensor,
        full_rope_sin: torch.Tensor,
        hadamard: torch.Tensor,
    ) -> PartialBucketMetadataOwner:
        bound = _clone_payload_buffers(payload)
        metadata = build_partial_bucket_native_metadata(
            bound,
            full_rope_cos=full_rope_cos,
            full_rope_sin=full_rope_sin,
            hadamard=hadamard,
        )
        return cls(
            bound_payload=bound,
            metadata=metadata,
            source_owners=(payload, full_rope_cos, full_rope_sin, hadamard),
        )

    @property
    def capture_actual(self) -> int:
        return self.bound_payload.actual

    def device_metadata_addresses(self) -> Mapping[str, int]:
        tensors = {
            **self.bound_payload.block_tables,
            "start_positions": self.bound_payload.start_positions,
            "kv_seq_lens": self.bound_payload.kv_seq_lens,
            "query_start_loc": self.bound_payload.query_start_loc,
        }
        return MappingProxyType({name: int(tensor.data_ptr()) for name, tensor in tensors.items()})

    def apply_device_payload(self, payload: PartialBucketPayload) -> None:
        """Copy replay metadata without changing any captured tensor address."""
        if payload.spec.key != self.bound_payload.spec.key:
            raise ValueError("replay payload uses a different static B/cache/layout specialization")
        if payload.device != self.bound_payload.device:
            raise ValueError(
                f"replay payload device mismatch: expected {self.bound_payload.device}, got {payload.device}"
            )
        before = dict(self.device_metadata_addresses())
        for name in _TABLE_ARGUMENTS:
            self.bound_payload.block_tables[name].copy_(payload.block_tables[name])
        self.bound_payload.start_positions.copy_(payload.start_positions)
        self.bound_payload.kv_seq_lens.copy_(payload.kv_seq_lens)
        if dict(self.device_metadata_addresses()) != before:
            raise AssertionError("in-place metadata update unexpectedly changed a captured address")


def resolve_swa_table_position_mapping(
    *,
    positions: torch.Tensor,
    block_table: torch.Tensor,
) -> torch.Tensor:
    """Cold Host proof of the kernel's ``table + position`` SWA address rule."""
    if positions.device.type != "cpu" or block_table.device.type != "cpu":
        raise ValueError("SWA address proof is Host-only; pass CPU tensors")
    if positions.ndim != 2 or block_table.ndim != 2 or positions.shape[0] != block_table.shape[0]:
        raise ValueError(
            "positions and block_table must be rank-2 with equal request rows, "
            f"got {tuple(positions.shape)} and {tuple(block_table.shape)}"
        )
    logical_blocks = torch.div(positions.to(torch.int64), BLOCK_SIZE, rounding_mode="floor")
    if torch.any(logical_blocks < 0) or torch.any(logical_blocks >= block_table.shape[1]):
        raise ValueError("position addresses a logical block outside the SWA table")
    physical_blocks = torch.gather(block_table.to(torch.int64), 1, logical_blocks)
    offsets = positions.to(torch.int64).remainder(BLOCK_SIZE)
    return torch.stack((physical_blocks, offsets), dim=-1).to(torch.int32).reshape(-1, 2)


@dataclass(frozen=True, slots=True)
class PartialBucketCanaryReport:
    block0_mismatches: Mapping[str, int]
    page_padding_mismatches: Mapping[str, int]

    @property
    def clean(self) -> bool:
        return not any(self.block0_mismatches.values()) and not any(self.page_padding_mismatches.values())

    def to_dict(self) -> dict[str, object]:
        return {
            "block0_mismatches": dict(self.block0_mismatches),
            "page_padding_mismatches": dict(self.page_padding_mismatches),
            "clean": self.clean,
        }


def _validate_cache_owners(
    raw_cache_tuple: Sequence[torch.Tensor],
    storage_owners: Sequence[torch.Tensor],
) -> None:
    if len(raw_cache_tuple) != 6:
        raise ValueError(f"expected six cache/state tensors, got {len(raw_cache_tuple)}")
    if len(storage_owners) != 4:
        raise ValueError(f"expected four page-strided storage owners, got {len(storage_owners)}")
    if any(tensor.shape[0] < 1 for tensor in raw_cache_tuple):
        raise ValueError("all cache/state families need a physical block 0 canary")


def seed_partial_bucket_canaries(
    raw_cache_tuple: Sequence[torch.Tensor],
    storage_owners: Sequence[torch.Tensor],
) -> None:
    """Seed six block-0 canaries and four physical-page padding canaries."""
    _validate_cache_owners(raw_cache_tuple, storage_owners)
    for tensor, canary in zip(raw_cache_tuple[:2], _CACHE_CANARIES[:2], strict=True):
        tensor.zero_()
        tensor[0].fill_(canary)
    for index, (tensor, storage, canary) in enumerate(
        zip(raw_cache_tuple[2:], storage_owners, _CACHE_CANARIES[2:], strict=True),
        start=2,
    ):
        storage.fill_(canary)
        if index == 5:
            tensor[1:].fill_(1.0)
        else:
            tensor[1:].zero_()


def inspect_partial_bucket_canaries(
    raw_cache_tuple: Sequence[torch.Tensor],
    storage_owners: Sequence[torch.Tensor],
) -> PartialBucketCanaryReport:
    """Inspect canaries after the caller has externally synchronized."""
    _validate_cache_owners(raw_cache_tuple, storage_owners)
    block0_mismatches: dict[str, int] = {}
    for name, tensor, canary in zip(_CACHE_FAMILY_NAMES, raw_cache_tuple, _CACHE_CANARIES, strict=True):
        block0 = tensor[0].detach().to(device="cpu")
        block0_mismatches[name] = int(torch.count_nonzero(block0 != canary))

    page_padding_mismatches: dict[str, int] = {}
    for name, tensor, storage, canary in zip(
        _CACHE_FAMILY_NAMES[2:],
        raw_cache_tuple[2:],
        storage_owners,
        _CACHE_CANARIES[2:],
        strict=True,
    ):
        page_stride = int(tensor.stride(0))
        logical_page_elements = int(tensor[0].numel())
        storage_offset = int(tensor.storage_offset())
        host_storage = storage.detach().to(device="cpu").reshape(-1)
        mismatches = 0
        for block in range(int(tensor.shape[0]) - 1):
            padding_start = storage_offset + block * page_stride + logical_page_elements
            padding_end = storage_offset + (block + 1) * page_stride
            padding = host_storage[padding_start:padding_end]
            mismatches += int(torch.count_nonzero(padding != canary))
        page_padding_mismatches[name] = mismatches
    return PartialBucketCanaryReport(
        block0_mismatches=MappingProxyType(block0_mismatches),
        page_padding_mismatches=MappingProxyType(page_padding_mismatches),
    )


def count_nonzero_padded_output(
    output: torch.Tensor,
    *,
    actual: int,
    seq: int,
) -> int:
    """Return the exact-zero violation count in rows ``[A*S:B*S)``."""
    if output.ndim < 1 or output.shape[0] % seq:
        raise ValueError(f"output leading extent must be divisible by S={seq}, got {tuple(output.shape)}")
    bucket = int(output.shape[0]) // seq
    _validate_actual(actual=actual, bucket=bucket)
    padded = output[actual * seq :].detach().to(device="cpu")
    return int(torch.count_nonzero(padded))


@dataclass(frozen=True, slots=True)
class A3PartialBucketReplayObservation:
    actual_requests: int
    output_max_abs: float
    padded_output_nonzero: int


@dataclass(frozen=True, slots=True)
class A3PartialBucketSmokeResult:
    runtime: str
    device: int
    bucket: int
    capture_actual: int
    graph_capture_count: int
    replay_observations: tuple[A3PartialBucketReplayObservation, ...]
    metadata_addresses_stable: bool
    eager_padded_output_nonzero: int
    block0_stage_mismatches: Mapping[str, Mapping[str, int]]
    block0_mismatches: Mapping[str, int]
    page_padding_mismatches: Mapping[str, int]

    def to_dict(self) -> dict[str, object]:
        return {
            "runtime": self.runtime,
            "device": self.device,
            "bucket": self.bucket,
            "capture_actual": self.capture_actual,
            "graph_capture_count": self.graph_capture_count,
            "replay_observations": tuple(asdict(observation) for observation in self.replay_observations),
            "metadata_addresses_stable": self.metadata_addresses_stable,
            "eager_padded_output_nonzero": self.eager_padded_output_nonzero,
            "block0_stage_mismatches": {
                stage: dict(mismatches) for stage, mismatches in self.block0_stage_mismatches.items()
            },
            "block0_mismatches": dict(self.block0_mismatches),
            "page_padding_mismatches": dict(self.page_padding_mismatches),
        }


def run_a3_partial_bucket_smoke(
    *,
    runtime: str,
    device: int = 0,
    bucket: int = 4,
    capture_actual: int = 2,
    replay_actuals: tuple[int, ...] = (1, 3, 2, 4, 1),
) -> A3PartialBucketSmokeResult:
    """Capture production custom-op once, then replay several A values in B."""
    if runtime not in {"tensormap_and_ringbuffer", "host_build_graph"}:
        raise ValueError(f"unsupported runtime: {runtime!r}")
    _validate_actual(actual=capture_actual, bucket=bucket)
    if capture_actual == bucket:
        raise ValueError("capture_actual must exercise A < B")
    if not replay_actuals:
        raise ValueError("replay_actuals must not be empty")
    for actual in replay_actuals:
        _validate_actual(actual=actual, bucket=bucket)
    if all(actual == capture_actual for actual in replay_actuals):
        raise ValueError("replay sequence must switch A after capture")

    import torch_npu
    from vllm.forward_context import ForwardContext, override_forward_context

    import vllm_ascend.ops.dsa  # noqa: F401 -- registers torch.ops.vllm.dsa_forward
    from tests.pypto_dsv4_decode_csa.a3_dispatch_smoke import (
        _custom_op_layer,
        _metadata_dict,
        _prepared_weights,
    )
    from tests.pypto_dsv4_decode_csa.fixtures import build_zero_fixture
    from vllm_ascend.ops._pypto_dsv4_csa import (
        bind_decode_csa_native_metadata,
        install_pypto_dsv4_decode_csa,
    )
    from vllm_ascend.ops._pypto_dsv4_csa.dispatch import DecodeCSADeviceOwnerRegistry

    torch_npu.npu.set_device(device)
    target = torch.device(f"npu:{device}")
    spec = DecodeCSAProgramSpec(batch=bucket)
    fixture = build_zero_fixture(
        spec,
        device=target,
        runtime=runtime,
        start_positions=_BOUNDARY_START_POSITIONS[:bucket],
    )
    seed_partial_bucket_canaries(fixture.raw_cache_tuple, fixture.storage_owners)
    block0_stages: dict[str, Mapping[str, int]] = {}

    def record_block0_stage(stage: str) -> None:
        report = inspect_partial_bucket_canaries(fixture.raw_cache_tuple, fixture.storage_owners)
        block0_stages[stage] = MappingProxyType(dict(report.block0_mismatches))

    record_block0_stage("seeded")
    fixture.arguments["cmp_ape"].fill_(0.25)
    fixture.arguments["inner_ape"].fill_(0.5)

    required_actuals = tuple(dict.fromkeys((capture_actual, *replay_actuals)))
    payloads = {
        actual: build_partial_bucket_payload(
            spec,
            actual=actual,
            device=target,
            start_positions=_BOUNDARY_START_POSITIONS[:bucket],
        )
        for actual in required_actuals
    }
    for actual, payload in payloads.items():
        for name, table in payload.block_tables.items():
            host_table = table.detach().to(device="cpu")
            if torch.any(host_table[:actual] <= 0) or torch.any(host_table[actual:] != 0):
                raise AssertionError(f"A={actual} {name} does not preserve nonzero-active/zero-padded rows")
    metadata_owner = PartialBucketMetadataOwner.create(
        payloads[capture_actual],
        full_rope_cos=fixture.arguments["freqs_cos"],
        full_rope_sin=fixture.arguments["freqs_sin"],
        hadamard=fixture.arguments["hadamard_idx"],
    )
    bound_native = bind_decode_csa_native_metadata(
        metadata_owner.metadata,
        spec,
        device=target,
    )
    expected_bound_tensors = {
        **metadata_owner.bound_payload.block_tables,
        "start_positions": metadata_owner.bound_payload.start_positions,
        "kv_seq_lens": metadata_owner.bound_payload.kv_seq_lens,
    }
    for name, expected in expected_bound_tensors.items():
        actual = bound_native.launch_arguments[name]
        if actual is not expected or actual.data_ptr() != expected.data_ptr():
            raise AssertionError(f"binder did not retain metadata_owner tensor {name!r}")
    initial_addresses = dict(metadata_owner.device_metadata_addresses())
    prefix = "pypto.partial_bucket.layers.0.self_attn"
    layer = _custom_op_layer(fixture, prefix=prefix)
    registry = DecodeCSADeviceOwnerRegistry()
    installed_owner = install_pypto_dsv4_decode_csa(
        layer,
        kv_cache=fixture.raw_cache_tuple,
        attn_metadata=metadata_owner.metadata,
        device=device,
        runtime=runtime,
        batch_buckets=(bucket,),
        registry=registry,
        prepared_weights=_prepared_weights(fixture, metadata_owner.metadata),
        uniform_query_rows_contract=True,
    )
    record_block0_stage("installed")
    context = ForwardContext(
        no_compile_layers={prefix: layer},
        attn_metadata=_metadata_dict(prefix, metadata_owner.metadata),
        slot_mapping={},
    )
    context.capturing = False

    graph = None
    replay_observations: list[A3PartialBucketReplayObservation] = []
    try:
        fixture.output.fill_(_OUTPUT_CANARY)
        with override_forward_context(context):
            torch.ops.vllm.dsa_forward(
                fixture.arguments["hidden_states"],
                False,
                fixture.output,
                prefix,
            )
        torch_npu.npu.synchronize(device)
        record_block0_stage("eager")
        eager_padded = count_nonzero_padded_output(
            fixture.output,
            actual=capture_actual,
            seq=spec.seq,
        )
        if eager_padded:
            raise AssertionError(f"eager padded output was not explicitly zeroed: {eager_padded}")
        if torch.count_nonzero(fixture.output.detach().to(device="cpu")):
            raise AssertionError("zero fixture produced a non-zero eager output")
        installed_owner.device_owner.mark_warmups_quiesced()

        capture_hidden = torch.empty_like(fixture.arguments["hidden_states"])
        capture_output = torch.full_like(fixture.output, _OUTPUT_CANARY)
        source = torch.zeros_like(capture_hidden)
        bias = torch.ones_like(capture_hidden)
        capture_stream = torch_npu.npu.Stream(device=device)
        graph = torch_npu.npu.NPUGraph()
        context.capturing = True
        with override_forward_context(context), torch_npu.npu.graph(graph, stream=capture_stream):
            torch.add(source, bias, out=capture_hidden)
            torch.ops.vllm.dsa_forward(capture_hidden, False, capture_output, prefix)
        capture_stream.synchronize()
        record_block0_stage("capture")

        for step, actual in enumerate(replay_actuals):
            with torch_npu.npu.stream(capture_stream):
                metadata_owner.apply_device_payload(payloads[actual])
                source.fill_(float(step - 2))
                capture_output.fill_(_OUTPUT_CANARY)
                graph.replay()
            capture_stream.synchronize()
            record_block0_stage(f"replay_{step}_a{actual}")
            host_output = capture_output.detach().float().to(device="cpu")
            if not torch.isfinite(host_output).all():
                raise AssertionError(f"A={actual} replay output contains NaN or Inf")
            padded_nonzero = count_nonzero_padded_output(
                host_output,
                actual=actual,
                seq=spec.seq,
            )
            output_max_abs = float(host_output.abs().max())
            if padded_nonzero:
                raise AssertionError(f"A={actual} padded output was not explicitly zeroed: {padded_nonzero}")
            if output_max_abs != 0.0:
                raise AssertionError(f"A={actual} zero fixture produced non-zero output: {output_max_abs}")
            replay_observations.append(
                A3PartialBucketReplayObservation(
                    actual_requests=actual,
                    output_max_abs=output_max_abs,
                    padded_output_nonzero=padded_nonzero,
                )
            )

        metadata_addresses_stable = dict(metadata_owner.device_metadata_addresses()) == initial_addresses
        if not metadata_addresses_stable:
            raise AssertionError("device metadata address changed across partial-bucket replays")
        canaries = inspect_partial_bucket_canaries(fixture.raw_cache_tuple, fixture.storage_owners)
        if not canaries.clean:
            stages_snapshot = {stage: dict(values) for stage, values in block0_stages.items()}
            raise AssertionError(
                f"partial-bucket cache canary violation: {canaries.to_dict()}, block0_stages={stages_snapshot}"
            )
    finally:
        torch_npu.npu.synchronize(device)
        if graph is not None:
            graph.reset()
        installed_owner.device_owner.backend.close()

    return A3PartialBucketSmokeResult(
        runtime=runtime,
        device=device,
        bucket=bucket,
        capture_actual=capture_actual,
        graph_capture_count=1,
        replay_observations=tuple(replay_observations),
        metadata_addresses_stable=metadata_addresses_stable,
        eager_padded_output_nonzero=eager_padded,
        block0_stage_mismatches=MappingProxyType(block0_stages),
        block0_mismatches=canaries.block0_mismatches,
        page_padding_mismatches=canaries.page_padding_mismatches,
    )


__all__ = [
    "A3PartialBucketReplayObservation",
    "A3PartialBucketSmokeResult",
    "PartialBucketCanaryReport",
    "PartialBucketMetadataOwner",
    "PartialBucketPayload",
    "build_partial_bucket_native_metadata",
    "build_partial_bucket_payload",
    "count_nonzero_padded_output",
    "inspect_partial_bucket_canaries",
    "resolve_swa_table_position_mapping",
    "run_a3_partial_bucket_smoke",
    "seed_partial_bucket_canaries",
]
