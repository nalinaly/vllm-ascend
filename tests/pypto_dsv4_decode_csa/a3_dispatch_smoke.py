# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""A3 smoke through the production ``torch.ops.vllm.dsa_forward`` boundary.

Unlike :mod:`a3_smoke`, this harness does not invoke the JIT program directly.
It constructs the six-cache object graph consumed by ``ops/dsa.py``, installs
the production PyPTO dispatch owner, and calls the unchanged vLLM custom-op
schema in eager and ACLGraph modes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from types import MappingProxyType, SimpleNamespace

import torch

from tests.pypto_dsv4_decode_csa.fixtures import build_zero_fixture
from vllm_ascend.ops._pypto_dsv4_csa import (
    DecodeCSAProgramSpec,
    PreparedDecodeCSAWeights,
    install_pypto_dsv4_decode_csa,
)
from vllm_ascend.ops._pypto_dsv4_csa.dispatch import (
    DecodeCSADeviceOwnerRegistry,
)


@dataclass(frozen=True, slots=True)
class A3DispatchSmokeResult:
    runtime: str
    device: int
    batch: int
    eager_max_abs: float
    replay_count: int
    replay_max_error: float
    eager_and_capture_addresses_differ: bool
    eager_bindings_retained: int
    capture_bindings_retained: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class A3MultiBucketDispatchSmokeResult:
    runtime: str
    device: int
    buckets: tuple[int, ...]
    compiled_bucket_count: int
    replay_order: tuple[int, ...]
    replay_max_error: float
    retained_capture_bindings: int
    survivor_replay_after_destroy: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _native_metadata(fixture, *, batch: int | None = None) -> tuple[object, ...]:
    """Build the five sorted metadata families from fixture-owned tensors."""
    spec = fixture.spec
    active_batch = spec.batch if batch is None else int(batch)
    if active_batch <= 0 or active_batch > spec.batch:
        raise ValueError(f"metadata batch must be in [1,{spec.batch}], got {active_batch}")
    tokens = active_batch * spec.seq
    args = fixture.arguments
    target = fixture.output.device
    query_start_loc = torch.arange(
        0,
        tokens + 1,
        spec.seq,
        dtype=torch.int32,
        device=target,
    )
    table_names = (
        "cmp_block_table",
        "compress_state_block_table",
        "inner_compress_state_block_table",
        "idx_block_table",
        "swa_block_table",
    )
    block_sizes = (32, 2, 2, 32, 32)
    families: list[object] = []
    for index, (table_name, block_size) in enumerate(zip(table_names, block_sizes, strict=True)):
        request = SimpleNamespace(
            block_table=args[table_name][:active_batch],
            seq_lens=args["kv_seq_lens"][:active_batch],
            # The production builder exposes a dynamic [actual_tokens, 2]
            # SWA view.  PyPTO deliberately does not bind it: its fixed L1 ABI
            # reconstructs writes from the static block table and positions.
            slot_mapping=None,
            block_size=block_size,
            query_start_loc=query_start_loc,
            start_pos=args["start_positions"][:active_batch] if index == 0 else None,
            num_reqs_actual=active_batch,
            full_compress_cos=args["freqs_cos"] if index == 0 else None,
            full_compress_sin=args["freqs_sin"] if index == 0 else None,
        )
        families.append(
            SimpleNamespace(
                num_actual_tokens=tokens,
                num_decodes=active_batch,
                num_decode_tokens=tokens,
                num_prefills=0,
                req_metadata=request,
                hadamard=args["hadamard_idx"] if index == 3 else None,
            )
        )
    return tuple(families)


def _prepared_weights(fixture, metadata: tuple[object, ...]) -> PreparedDecodeCSAWeights:
    cache_names = set(fixture.prepared_caches.for_launch())
    metadata_names = {
        "cmp_block_table",
        "compress_state_block_table",
        "inner_compress_state_block_table",
        "idx_block_table",
        "swa_block_table",
        "start_positions",
        "kv_seq_lens",
    }
    dynamic_names = cache_names | metadata_names | {"hidden_states", "attn_out"}
    weights = {name: value for name, value in fixture.arguments.items() if name not in dynamic_names}
    # Keep metadata alive as well: it owns the cold query_start_loc proof tensor.
    return PreparedDecodeCSAWeights(
        launch_arguments=MappingProxyType(weights),
        source_owner=(fixture, metadata),
    )


def _custom_op_layer(fixture, *, prefix: str) -> object:
    raw = fixture.raw_cache_tuple
    return SimpleNamespace(
        prefix=prefix,
        compress_ratio=4,
        swa_cache_layer=SimpleNamespace(kv_cache=raw[1]),
        compressor=SimpleNamespace(state_cache=SimpleNamespace(kv_cache=raw[2])),
        indexer=SimpleNamespace(
            compressor=SimpleNamespace(state_cache=SimpleNamespace(kv_cache=raw[3])),
            k_cache=SimpleNamespace(kv_cache=(raw[4], raw[5])),
        ),
        dsa_attn=SimpleNamespace(
            layer_name=f"{prefix}.attn",
            kv_cache=raw[0],
            impl=SimpleNamespace(
                forward=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                    AssertionError("installed PyPTO custom-op unexpectedly fell back to native")
                )
            ),
        ),
    )


def _metadata_dict(prefix: str, metadata: tuple[object, ...]) -> dict[str, object]:
    return {f"{prefix}.{index}": value for index, value in enumerate(metadata)}


def run_a3_dispatch_smoke(
    *,
    runtime: str,
    device: int = 0,
    batch: int = 4,
    replay_values: tuple[float, ...] = (0.0, 2.0, -3.0, 7.5),
) -> A3DispatchSmokeResult:
    """Run the production custom-op path in one fresh process/runtime owner."""
    import torch_npu
    from vllm.forward_context import ForwardContext, override_forward_context

    import vllm_ascend.ops.dsa  # noqa: F401 -- registers torch.ops.vllm.dsa_forward

    if runtime not in {"tensormap_and_ringbuffer", "host_build_graph"}:
        raise ValueError(f"unsupported runtime: {runtime!r}")
    torch_npu.npu.set_device(device)
    target = torch.device(f"npu:{device}")
    fixture = build_zero_fixture(
        DecodeCSAProgramSpec(batch=batch),
        device=target,
        runtime=runtime,
    )
    metadata = _native_metadata(fixture)
    prefix = "pypto.synthetic.layers.0.self_attn"
    layer = _custom_op_layer(fixture, prefix=prefix)
    registry = DecodeCSADeviceOwnerRegistry()
    owner = install_pypto_dsv4_decode_csa(
        layer,
        kv_cache=fixture.raw_cache_tuple,
        attn_metadata=metadata,
        device=device,
        runtime=runtime,
        batch_buckets=(batch,),
        registry=registry,
        prepared_weights=_prepared_weights(fixture, metadata),
        uniform_query_rows_contract=True,
    )

    context = ForwardContext(
        no_compile_layers={prefix: layer},
        attn_metadata=_metadata_dict(prefix, metadata),
        slot_mapping={},
    )
    context.capturing = False
    graph = None
    eager_retained = -1
    capture_retained = -1
    replay_error = 0.0
    try:
        with override_forward_context(context):
            torch.ops.vllm.dsa_forward(
                fixture.arguments["hidden_states"],
                False,
                fixture.output,
                prefix,
            )
        torch_npu.npu.synchronize(device)
        eager = fixture.output.float().cpu()
        if not torch.isfinite(eager).all():
            raise AssertionError("production-dispatch eager output contains NaN or Inf")
        eager_max_abs = float(eager.abs().max())
        if eager_max_abs != 0.0:
            raise AssertionError(f"zero fixture produced non-zero eager output: {eager_max_abs}")
        eager_retained = len(owner.device_owner.backend._bindings)
        if eager_retained != 0:
            raise AssertionError(f"ordinary eager binding leaked into capture owners: {eager_retained}")

        # The caller owns the synchronization above; this marker performs no sync.
        owner.device_owner.mark_warmups_quiesced()
        capture_hidden = torch.empty_like(fixture.arguments["hidden_states"])
        capture_output = torch.empty_like(fixture.output)
        source = torch.zeros_like(capture_hidden)
        bias = torch.ones_like(capture_hidden)
        final = torch.empty_like(capture_output)
        address_change = (
            capture_hidden.data_ptr() != fixture.arguments["hidden_states"].data_ptr()
            and capture_output.data_ptr() != fixture.output.data_ptr()
        )
        if not address_change:
            raise AssertionError("capture must bind a distinct hidden/output address snapshot")

        capture_stream = torch_npu.npu.Stream(device=device)
        graph = torch_npu.npu.NPUGraph()
        context.capturing = True
        with override_forward_context(context), torch_npu.npu.graph(graph, stream=capture_stream):
            torch.add(source, bias, out=capture_hidden)
            torch.ops.vllm.dsa_forward(capture_hidden, False, capture_output, prefix)
            torch.add(capture_output, 1.0, out=final)
        capture_retained = len(owner.device_owner.backend._bindings)
        if capture_retained != 1:
            raise AssertionError(f"capture must retain exactly one address snapshot, got {capture_retained}")

        for value in replay_values:
            with torch_npu.npu.stream(capture_stream):
                source.fill_(value)
                graph.replay()
            capture_stream.synchronize()
            host = final.float().cpu()
            if not torch.isfinite(host).all():
                raise AssertionError(f"production-dispatch replay {value} contains NaN or Inf")
            replay_error = max(replay_error, float((host - 1.0).abs().max()))
        if replay_error != 0.0:
            raise AssertionError(f"production-dispatch ACLGraph replay max error: {replay_error}")
    finally:
        torch_npu.npu.synchronize(device)
        if graph is not None:
            graph.reset()
        owner.device_owner.backend.close()

    return A3DispatchSmokeResult(
        runtime=runtime,
        device=device,
        batch=batch,
        eager_max_abs=eager_max_abs,
        replay_count=len(replay_values),
        replay_max_error=replay_error,
        eager_and_capture_addresses_differ=address_change,
        eager_bindings_retained=eager_retained,
        capture_bindings_retained=capture_retained,
    )


def run_a3_multibucket_dispatch_smoke(
    *,
    runtime: str,
    device: int = 0,
    buckets: tuple[int, ...] = (4, 8, 12, 16),
) -> A3MultiBucketDispatchSmokeResult:
    """Keep four graph buckets alive and replay them serially in one context."""
    import torch_npu
    from vllm.forward_context import ForwardContext, override_forward_context

    import vllm_ascend.ops.dsa  # noqa: F401 -- registers torch.ops.vllm.dsa_forward

    if runtime not in {"tensormap_and_ringbuffer", "host_build_graph"}:
        raise ValueError(f"unsupported runtime: {runtime!r}")
    if tuple(sorted(set(buckets))) != tuple(buckets) or any(batch not in {4, 8, 12, 16} for batch in buckets):
        raise ValueError(f"buckets must be an ordered subset of (4,8,12,16), got {buckets}")

    torch_npu.npu.set_device(device)
    target = torch.device(f"npu:{device}")
    max_batch = max(buckets)
    fixture = build_zero_fixture(
        DecodeCSAProgramSpec(batch=max_batch),
        device=target,
        runtime=runtime,
    )
    metadata_by_batch = {batch: _native_metadata(fixture, batch=batch) for batch in buckets}
    prefix = "pypto.synthetic.layers.0.self_attn"
    layer = _custom_op_layer(fixture, prefix=prefix)
    registry = DecodeCSADeviceOwnerRegistry()
    owner = install_pypto_dsv4_decode_csa(
        layer,
        kv_cache=fixture.raw_cache_tuple,
        attn_metadata=metadata_by_batch[max_batch],
        device=device,
        runtime=runtime,
        batch_buckets=buckets,
        registry=registry,
        prepared_weights=_prepared_weights(fixture, metadata_by_batch[max_batch]),
        uniform_query_rows_contract=True,
    )

    contexts: dict[int, object] = {}
    eager_io: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    graphs: dict[int, object] = {}
    streams: dict[int, object] = {}
    finals: dict[int, torch.Tensor] = {}
    sources: dict[int, torch.Tensor] = {}
    replay_order = (4, 16, 8, 12, 16, 4, 12, 8)
    replay_order = tuple(batch for batch in replay_order if batch in buckets)
    destroyed_batch = buckets[1] if len(buckets) > 1 else buckets[0]
    replay_error = 0.0
    survivor_replay = False
    try:
        # Warm every static callable/address family before any graph capture.
        for batch in buckets:
            hidden = torch.zeros((batch * 8, 4096), dtype=torch.bfloat16, device=target)
            output = torch.empty_like(hidden)
            context = ForwardContext(
                no_compile_layers={prefix: layer},
                attn_metadata=_metadata_dict(prefix, metadata_by_batch[batch]),
                slot_mapping={},
            )
            context.capturing = False
            contexts[batch] = context
            eager_io[batch] = (hidden, output)
            with override_forward_context(context):
                torch.ops.vllm.dsa_forward(hidden, False, output, prefix)
        torch_npu.npu.synchronize(device)
        owner.device_owner.mark_warmups_quiesced()

        # Each bucket owns distinct graph buffers and remains alive concurrently.
        for batch in buckets:
            hidden = torch.empty_like(eager_io[batch][0])
            output = torch.empty_like(hidden)
            source = torch.zeros_like(hidden)
            bias = torch.ones_like(hidden)
            final = torch.empty_like(output)
            stream = torch_npu.npu.Stream(device=device)
            graph = torch_npu.npu.NPUGraph()
            contexts[batch].capturing = True
            with override_forward_context(contexts[batch]), torch_npu.npu.graph(graph, stream=stream):
                torch.add(source, bias, out=hidden)
                torch.ops.vllm.dsa_forward(hidden, False, output, prefix)
                torch.add(output, 1.0, out=final)
            graphs[batch] = graph
            streams[batch] = stream
            finals[batch] = final
            sources[batch] = source

        for step, batch in enumerate(replay_order):
            stream = streams[batch]
            with torch_npu.npu.stream(stream):
                sources[batch].fill_(float(step - 3))
                graphs[batch].replay()
            stream.synchronize()
            host = finals[batch].float().cpu()
            replay_error = max(replay_error, float((host - 1.0).abs().max()))
        if replay_error != 0.0:
            raise AssertionError(f"multi-bucket ACLGraph replay max error: {replay_error}")

        graphs[destroyed_batch].reset()
        del graphs[destroyed_batch]
        survivor = next(batch for batch in reversed(buckets) if batch != destroyed_batch)
        streams[survivor].synchronize()
        graphs[survivor].replay()
        streams[survivor].synchronize()
        survivor_error = float((finals[survivor].float().cpu() - 1.0).abs().max())
        survivor_replay = survivor_error == 0.0
        if not survivor_replay:
            raise AssertionError(f"destroying B{destroyed_batch} broke surviving B{survivor}: {survivor_error}")

        retained = len(owner.device_owner.backend._bindings)
        if retained != len(buckets):
            raise AssertionError(f"expected one retained snapshot per graph bucket, got {retained}")
    finally:
        torch_npu.npu.synchronize(device)
        for graph in graphs.values():
            graph.reset()
        owner.device_owner.backend.close()

    return A3MultiBucketDispatchSmokeResult(
        runtime=runtime,
        device=device,
        buckets=buckets,
        compiled_bucket_count=len(owner.device_owner.backend.registered_specs),
        replay_order=replay_order,
        replay_max_error=replay_error,
        retained_capture_bindings=retained,
        survivor_replay_after_destroy=survivor_replay,
    )


__all__ = [
    "A3DispatchSmokeResult",
    "A3MultiBucketDispatchSmokeResult",
    "run_a3_dispatch_smoke",
    "run_a3_multibucket_dispatch_smoke",
]
