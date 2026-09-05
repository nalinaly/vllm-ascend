# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Host contracts for the production partial-bucket A3 smoke."""

from __future__ import annotations

import json
from types import MappingProxyType

import pytest
import torch

from tests.pypto_dsv4_decode_csa.a3_partial_bucket_smoke import (
    A3PartialBucketReplayObservation,
    A3PartialBucketSmokeResult,
    PartialBucketMetadataOwner,
    build_partial_bucket_native_metadata,
    build_partial_bucket_payload,
    count_nonzero_padded_output,
    inspect_partial_bucket_canaries,
    resolve_swa_table_position_mapping,
    run_a3_partial_bucket_smoke,
    seed_partial_bucket_canaries,
)
from tests.pypto_dsv4_decode_csa.fixtures import _native_swa_slot_mapping
from vllm_ascend.ops._pypto_dsv4_csa import DecodeCSAProgramSpec
from vllm_ascend.ops._pypto_dsv4_csa.native_metadata import bind_decode_csa_native_metadata


def _static_owners() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.zeros((128, 64), dtype=torch.bfloat16),
        torch.zeros((128, 64), dtype=torch.bfloat16),
        torch.eye(128, dtype=torch.bfloat16),
    )


@pytest.mark.parametrize(("bucket", "actual"), ((4, 1), (4, 3), (8, 5)))
def test_partial_payload_uses_full_bucket_shapes_and_actual_counters(
    bucket: int,
    actual: int,
) -> None:
    spec = DecodeCSAProgramSpec(batch=bucket)
    payload = build_partial_bucket_payload(spec, actual=actual)
    rope_cos, rope_sin, hadamard = _static_owners()
    metadata = build_partial_bucket_native_metadata(
        payload,
        full_rope_cos=rope_cos,
        full_rope_sin=rope_sin,
        hadamard=hadamard,
    )
    prepared = bind_decode_csa_native_metadata(metadata, spec, device=torch.device("cpu"))

    assert tuple(payload.query_start_loc.tolist()) == tuple(range(0, (bucket + 1) * 8, 8))
    assert tuple(payload.start_positions.shape) == (bucket,)
    assert tuple(payload.kv_seq_lens.shape) == (bucket,)
    assert torch.all(payload.start_positions[actual:] == 0)
    assert torch.all(payload.kv_seq_lens[actual:] == 0)
    for table in payload.block_tables.values():
        assert table.shape[0] == bucket
        assert torch.all(table[:actual] > 0)
        assert torch.all(table[actual:] == 0)

    assert len(metadata) == 5
    for family in metadata:
        assert family.num_actual_tokens == actual * 8
        assert family.num_decode_tokens == actual * 8
        assert family.num_decodes == bucket
        assert family.num_prefills == 0
        assert family.req_metadata.num_reqs_actual == actual
        assert family.req_metadata.query_start_loc.shape == (bucket + 1,)
        assert family.req_metadata.block_table.shape[0] == bucket
        assert family.req_metadata.seq_lens.shape == (bucket,)
        assert family.req_metadata.slot_mapping is None
    assert prepared.for_launch()["start_positions"] is payload.start_positions
    assert prepared.for_launch()["kv_seq_lens"] is payload.kv_seq_lens
    assert "swa_slot_mapping" not in prepared.for_launch()


def test_active_rows_have_disjoint_nonzero_partitions_and_padding_owns_no_block() -> None:
    payload = build_partial_bucket_payload(DecodeCSAProgramSpec(batch=4), actual=3)

    for table in payload.block_tables.values():
        active_sets = tuple(set(int(value) for value in row.tolist()) for row in table[:3])
        assert all(0 not in values for values in active_sets)
        assert active_sets[0].isdisjoint(active_sets[1])
        assert active_sets[0].isdisjoint(active_sets[2])
        assert active_sets[1].isdisjoint(active_sets[2])
        assert torch.count_nonzero(table[3]) == 0


def test_capture_owner_switches_A_in_place_without_mutating_host_counters() -> None:
    spec = DecodeCSAProgramSpec(batch=4)
    capture = build_partial_bucket_payload(spec, actual=1)
    replay = build_partial_bucket_payload(spec, actual=3)
    rope_cos, rope_sin, hadamard = _static_owners()
    owner = PartialBucketMetadataOwner.create(
        capture,
        full_rope_cos=rope_cos,
        full_rope_sin=rope_sin,
        hadamard=hadamard,
    )
    addresses = dict(owner.device_metadata_addresses())
    query_start_before = owner.bound_payload.query_start_loc.clone()

    owner.apply_device_payload(replay)

    assert dict(owner.device_metadata_addresses()) == addresses
    assert owner.capture_actual == 1
    assert all(family.req_metadata.num_reqs_actual == 1 for family in owner.metadata)
    assert all(family.num_actual_tokens == 8 for family in owner.metadata)
    torch.testing.assert_close(owner.bound_payload.query_start_loc, query_start_before)
    torch.testing.assert_close(owner.bound_payload.start_positions, replay.start_positions)
    torch.testing.assert_close(owner.bound_payload.kv_seq_lens, replay.kv_seq_lens)
    for name, table in replay.block_tables.items():
        torch.testing.assert_close(owner.bound_payload.block_tables[name], table)


def test_capture_owner_rejects_a_different_static_bucket() -> None:
    capture = build_partial_bucket_payload(DecodeCSAProgramSpec(batch=4), actual=1)
    replay = build_partial_bucket_payload(DecodeCSAProgramSpec(batch=8), actual=1)
    rope_cos, rope_sin, hadamard = _static_owners()
    owner = PartialBucketMetadataOwner.create(
        capture,
        full_rope_cos=rope_cos,
        full_rope_sin=rope_sin,
        hadamard=hadamard,
    )

    with pytest.raises(ValueError, match="different static B"):
        owner.apply_device_payload(replay)


@pytest.mark.parametrize("actual", (0, 5, True))
def test_partial_payload_rejects_invalid_actual_count(actual: object) -> None:
    error = TypeError if isinstance(actual, bool) else ValueError
    with pytest.raises(error, match="actual request count"):
        build_partial_bucket_payload(DecodeCSAProgramSpec(batch=4), actual=actual)  # type: ignore[arg-type]


def test_swa_table_position_formula_matches_native_mapping_at_page_boundaries() -> None:
    positions = torch.tensor([[31, 32, 63, 64]], dtype=torch.int32)
    block_table = torch.tensor([[7, 11, 13]], dtype=torch.int32)

    resolved = resolve_swa_table_position_mapping(
        positions=positions,
        block_table=block_table,
    )
    native = _native_swa_slot_mapping(
        positions=positions,
        block_table=block_table,
    )

    expected = torch.tensor(
        [[7, 31], [11, 0], [11, 31], [13, 0]],
        dtype=torch.int32,
    )
    torch.testing.assert_close(resolved, expected)
    torch.testing.assert_close(resolved, native)
    linear = resolved[:, 0].to(torch.int64) * 32 + resolved[:, 1]
    torch.testing.assert_close(
        linear,
        torch.tensor([7 * 32 + 31, 11 * 32, 11 * 32 + 31, 13 * 32], dtype=torch.int64),
    )


def _make_page_strided(
    *,
    blocks: int,
    tail_shape: tuple[int, ...],
    page_stride: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    logical_page = 1
    for extent in tail_shape:
        logical_page *= extent
    span = (blocks - 1) * page_stride + logical_page
    storage = torch.zeros(span, dtype=dtype)
    tail_strides: list[int] = []
    running = 1
    for extent in reversed(tail_shape):
        tail_strides.append(running)
        running *= extent
    view = torch.as_strided(
        storage,
        size=(blocks, *tail_shape),
        stride=(page_stride, *reversed(tail_strides)),
    )
    return storage, view


def _small_cache_world() -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    main_storage, main = _make_page_strided(
        blocks=3,
        tail_shape=(2, 1, 2),
        page_stride=7,
        dtype=torch.float32,
    )
    inner_storage, inner = _make_page_strided(
        blocks=3,
        tail_shape=(2, 1, 2),
        page_stride=8,
        dtype=torch.float32,
    )
    indexer_k_storage, indexer_k = _make_page_strided(
        blocks=3,
        tail_shape=(2, 1, 2),
        page_stride=6,
        dtype=torch.int8,
    )
    indexer_scale_storage, indexer_scale = _make_page_strided(
        blocks=3,
        tail_shape=(2, 1, 1),
        page_stride=5,
        dtype=torch.float16,
    )
    raw = (
        torch.zeros((3, 2, 1, 2), dtype=torch.bfloat16),
        torch.zeros((3, 2, 1, 2), dtype=torch.bfloat16),
        main,
        inner,
        indexer_k,
        indexer_scale,
    )
    storages = (main_storage, inner_storage, indexer_k_storage, indexer_scale_storage)
    return raw, storages


def test_six_block0_and_four_page_padding_canaries_are_independent() -> None:
    raw, storages = _small_cache_world()
    seed_partial_bucket_canaries(raw, storages)

    clean = inspect_partial_bucket_canaries(raw, storages)
    assert clean.clean
    assert tuple(clean.block0_mismatches) == (
        "compressed_kv",
        "swa_kv",
        "main_compressor_state",
        "inner_compressor_state",
        "indexer_k",
        "indexer_scale",
    )
    assert all(value == 0 for value in clean.block0_mismatches.values())
    assert all(value == 0 for value in clean.page_padding_mismatches.values())

    for tensor in raw:
        tensor[0].reshape(-1)[0] = 0
    for tensor, storage in zip(raw[2:], storages, strict=True):
        first_padding = int(tensor.storage_offset()) + int(tensor[0].numel())
        storage[first_padding] = 0

    damaged = inspect_partial_bucket_canaries(raw, storages)
    assert not damaged.clean
    assert all(value == 1 for value in damaged.block0_mismatches.values())
    assert all(value == 1 for value in damaged.page_padding_mismatches.values())


def test_padded_output_contract_requires_exact_zero() -> None:
    output = torch.zeros((4 * 8, 3), dtype=torch.bfloat16)
    output[: 3 * 8].fill_(1.0)

    assert count_nonzero_padded_output(output, actual=3, seq=8) == 0
    output[3 * 8, 1] = 2.0
    assert count_nonzero_padded_output(output, actual=3, seq=8) == 1


@pytest.mark.parametrize(
    ("kwargs", "match"),
    (
        ({"runtime": "invalid"}, "unsupported runtime"),
        ({"runtime": "tensormap_and_ringbuffer", "capture_actual": 4}, "A < B"),
        ({"runtime": "tensormap_and_ringbuffer", "replay_actuals": ()}, "must not be empty"),
        (
            {
                "runtime": "tensormap_and_ringbuffer",
                "capture_actual": 2,
                "replay_actuals": (2, 2),
            },
            "must switch A",
        ),
    ),
)
def test_a3_runner_rejects_invalid_plan_before_importing_npu(
    kwargs: dict[str, object],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        run_a3_partial_bucket_smoke(**kwargs)  # type: ignore[arg-type]


def test_partial_bucket_result_is_json_ready() -> None:
    observation = A3PartialBucketReplayObservation(
        actual_requests=3,
        output_max_abs=0.0,
        padded_output_nonzero=0,
    )
    result = A3PartialBucketSmokeResult(
        runtime="tensormap_and_ringbuffer",
        device=0,
        bucket=4,
        capture_actual=1,
        graph_capture_count=1,
        replay_observations=(observation,),
        metadata_addresses_stable=True,
        eager_padded_output_nonzero=0,
        block0_stage_mismatches=MappingProxyType(
            {
                "seeded": MappingProxyType({"swa_kv": 0}),
                "replay_0_a1": MappingProxyType({"swa_kv": 0}),
            }
        ),
        block0_mismatches=MappingProxyType({"swa_kv": 0}),
        page_padding_mismatches=MappingProxyType({"indexer_k": 0}),
    )

    payload = result.to_dict()

    assert payload["graph_capture_count"] == 1
    assert payload["replay_observations"][0]["actual_requests"] == 3
    assert payload["block0_mismatches"] == {"swa_kv": 0}
    assert payload["block0_stage_mismatches"] == {
        "seeded": {"swa_kv": 0},
        "replay_0_a1": {"swa_kv": 0},
    }
    json.dumps(payload)
