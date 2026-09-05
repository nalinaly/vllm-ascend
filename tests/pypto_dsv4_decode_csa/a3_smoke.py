# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Reusable one-card A3 smoke for the real-shape decode CSA L1 entry."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from tests.pypto_dsv4_decode_csa.fixtures import build_zero_fixture
from vllm_ascend.ops._pypto_dsv4_csa import DecodeCSAProgramSpec


@dataclass(frozen=True, slots=True)
class A3SmokeResult:
    runtime: str
    device: int
    batch: int
    eager_max_abs: float
    output_is_preallocated: bool
    replay_count: int
    replay_max_error: float
    logical_nonzero: dict[str, int]
    padding_mismatches: dict[str, int]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _apply_padding_canaries(fixture) -> None:
    canaries = (123.0, 77.0, 91, 55.0)
    for storage, logical, canary in zip(
        fixture.storage_owners,
        fixture.raw_cache_tuple[2:],
        canaries,
        strict=True,
    ):
        storage.fill_(canary)
        logical.zero_()
    fixture.raw_cache_tuple[5].fill_(1.0)

    # Zero projections keep attn_out exactly zero, while non-zero positional
    # embeddings force real main/inner state writes for address validation.
    fixture.arguments["cmp_ape"].fill_(0.25)
    fixture.arguments["inner_ape"].fill_(0.5)


def _padding_mismatches(fixture) -> dict[str, int]:
    checks = (
        ("main", fixture.storage_owners[0], fixture.spec.main_state_blocks, 8192, 4096, 123.0),
        ("inner", fixture.storage_owners[1], fixture.spec.inner_state_blocks, 1040, 1024, 77.0),
        ("indexer_k", fixture.storage_owners[2], fixture.spec.indexer_blocks, 4160, 4096, 91),
        ("indexer_scale", fixture.storage_owners[3], fixture.spec.indexer_blocks, 2080, 32, 55.0),
    )
    mismatches: dict[str, int] = {}
    for name, storage, blocks, page_stride, logical_page, canary in checks:
        flat = storage.cpu().reshape(-1)
        bad = 0
        for block in range(blocks - 1):
            padding = flat[block * page_stride + logical_page : (block + 1) * page_stride]
            bad += int(torch.count_nonzero(padding != canary))
        mismatches[name] = bad
    return mismatches


def run_a3_zero_smoke(
    *,
    runtime: str,
    device: int = 0,
    batch: int = 4,
    replay_values: tuple[float, ...] = (0.0, 2.0, -3.0, 7.5),
    artifact_dir: str | Path | None = None,
) -> A3SmokeResult:
    """Run eager plus mixed Torch/PyPTO ACLGraph and return inspectable metrics."""
    import pypto
    import torch_npu
    from pypto.runtime import RunConfig

    if runtime not in {"tensormap_and_ringbuffer", "host_build_graph"}:
        raise ValueError(f"unsupported runtime: {runtime!r}")

    torch_npu.npu.set_device(device)
    target = torch.device(f"npu:{device}")
    fixture = build_zero_fixture(
        DecodeCSAProgramSpec(batch=batch),
        device=target,
        runtime=runtime,
    )
    _apply_padding_canaries(fixture)
    config = RunConfig(
        platform="a2a3",
        device_id=device,
        runtime=runtime,
        save_kernels=artifact_dir is not None,
        save_kernels_dir=str(artifact_dir) if artifact_dir is not None else None,
    )
    graph = None
    eager_max_abs = float("nan")
    output_is_preallocated = False
    replay_max_error = 0.0
    try:
        torch_npu.npu.synchronize(device)
        returned = fixture.program(**fixture.arguments, config=config)
        torch_npu.npu.synchronize(device)
        output_is_preallocated = returned is fixture.output
        eager = fixture.output.float().cpu()
        if not torch.isfinite(eager).all():
            raise AssertionError("CSA eager output contains NaN or Inf")
        eager_max_abs = float(eager.abs().max())
        if eager_max_abs != 0.0:
            raise AssertionError(f"zero fixture produced non-zero output: {eager_max_abs}")

        source = torch.zeros_like(fixture.arguments["hidden_states"])
        bias = torch.ones_like(source)
        final = torch.empty_like(fixture.output)
        capture_stream = torch_npu.npu.Stream(device=device)
        graph = torch_npu.npu.NPUGraph()
        with torch_npu.npu.graph(graph, stream=capture_stream):
            torch.add(source, bias, out=fixture.arguments["hidden_states"])
            captured = fixture.program(**fixture.arguments, config=config)
            torch.add(captured, 1.0, out=final)
        if captured is not fixture.output:
            raise AssertionError("captured L1 call did not return the preallocated output")

        for value in replay_values:
            with torch_npu.npu.stream(capture_stream):
                source.fill_(value)
                graph.replay()
            capture_stream.synchronize()
            host = final.float().cpu()
            if not torch.isfinite(host).all():
                raise AssertionError(f"ACLGraph replay {value} contains NaN or Inf")
            replay_max_error = max(
                replay_max_error,
                float((host - 1.0).abs().max()),
            )
        if replay_max_error != 0.0:
            raise AssertionError(f"ACLGraph replay max error: {replay_max_error}")

        logical_nonzero = {
            name: int(torch.count_nonzero(tensor.cpu()))
            for name, tensor in zip(
                ("main", "inner", "indexer_k", "indexer_scale"),
                fixture.raw_cache_tuple[2:],
                strict=True,
            )
        }
        if logical_nonzero["main"] == 0 or logical_nonzero["inner"] == 0:
            raise AssertionError(f"state write probe did not mutate state: {logical_nonzero}")
        padding = _padding_mismatches(fixture)
        if any(padding.values()):
            raise AssertionError(f"page padding was modified: {padding}")
    finally:
        torch_npu.npu.synchronize(device)
        if graph is not None:
            graph.reset()
        pypto.l1.shutdown(device=device)

    return A3SmokeResult(
        runtime=runtime,
        device=device,
        batch=batch,
        eager_max_abs=eager_max_abs,
        output_is_preallocated=output_is_preallocated,
        replay_count=len(replay_values),
        replay_max_error=replay_max_error,
        logical_nonzero=logical_nonzero,
        padding_mismatches=padding,
    )


__all__ = ["A3SmokeResult", "run_a3_zero_smoke"]
