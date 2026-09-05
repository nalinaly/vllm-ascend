# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Host contracts for evidence-oriented indexer quantization comparison."""

from __future__ import annotations

import json
from types import MappingProxyType

import pytest
import torch

from tests.pypto_dsv4_decode_csa.indexer_quantized_comparison import (
    IndexerQuantizationPolicy,
    compare_indexer_quantized,
    compare_indexer_quantized_long_chain,
)


def _states(*, rows: int = 128, width: int = 128, scale: float = 0.019317626953125):
    initial_k = torch.zeros((rows, width), dtype=torch.int8)
    reference_k = initial_k.clone()
    candidate_k = initial_k.clone()
    # Mark every row written so one mismatch is measured against a realistic
    # active cache surface, rather than diluted by untouched physical storage.
    reference_k[:, 0] = 3
    candidate_k[:, 0] = 3
    initial_scale = torch.zeros((rows, 1), dtype=torch.float16)
    reference_scale = torch.full_like(initial_scale, scale)
    candidate_scale = reference_scale.clone()
    return reference_k, candidate_k, initial_k, reference_scale, candidate_scale, initial_scale


def _compare(states, **kwargs):
    reference_k, candidate_k, initial_k, reference_scale, candidate_scale, initial_scale = states
    return compare_indexer_quantized(
        reference_k,
        candidate_k,
        initial_k,
        reference_scale,
        candidate_scale,
        initial_scale,
        **kwargs,
    )


def test_bit_exact_pair_reports_raw_exact_without_using_fallback() -> None:
    result = _compare(_states())

    assert result.raw_exact
    assert result.scale_bit_exact
    assert result.raw_mismatch_elements == 0
    assert result.dequantized_max_abs_error == 0.0
    assert not result.one_bin_fallback_eligible
    assert result.acceptable


def test_rare_one_bin_difference_keeps_raw_failure_but_joint_contract_accepts() -> None:
    states = _states()
    states[1][69, 54] = states[0][69, 54] + 1
    provenance = MappingProxyType(
        {
            69: (
                MappingProxyType(
                    {
                        "request": 1,
                        "token_offset": 4,
                        "absolute_position": 740,
                        "decode_step": 92,
                    }
                ),
            )
        }
    )

    result = _compare(states, row_positions=provenance)
    payload = result.to_dict()

    assert not result.raw_exact
    assert result.raw_mismatch_elements == 1
    assert result.raw_mismatch_fraction == pytest.approx(1 / (128 * 128))
    assert result.raw_max_bin_error == 1
    assert result.scale_bit_exact
    assert result.scale_mismatch_elements == 0
    assert result.dequantized_nonzero_error_elements == 1
    assert result.dequantized_max_abs_error == pytest.approx(float(states[3][69, 0]))
    assert result.dequantized_max_scale_quanta == pytest.approx(1.0)
    assert result.dequantized_over_policy_elements == 0
    assert result.one_bin_fallback_eligible
    assert result.acceptable
    assert result.first_raw_mismatch is not None
    assert result.first_raw_mismatch.first_mapped_decode_step == 92
    assert result.first_raw_mismatch.physical_block == 2
    assert result.first_raw_mismatch.block_offset == 5
    # MappingProxy provenance must still be a durable JSON report.
    assert json.loads(json.dumps(payload))["first_raw_mismatch"]["expected_token_positions"][0]["decode_step"] == 92


def test_128_step_observed_density_is_reported_exactly_not_rounded_away() -> None:
    states = _states(rows=1024)
    for row, column in ((69, 54), (311, 27), (700, 91), (1023, 127)):
        states[1][row, column] = states[0][row, column] + 1

    result = _compare(states)

    assert result.compared_k_elements == 131072
    assert result.raw_mismatch_elements == 4
    assert result.raw_mismatch_fraction == pytest.approx(4 / 131072)
    assert result.raw_max_bin_error == 1
    assert not result.raw_exact
    assert result.scale_bit_exact
    assert result.dequantized_nonzero_error_elements == 4
    assert result.dequantized_max_abs_error == pytest.approx(float(states[3][0, 0]))
    assert result.one_bin_fallback_eligible
    assert result.acceptable


def test_explicit_step_write_rows_prevent_historical_density_dilution() -> None:
    states = _states(rows=1024)
    states[1][1023, 54] = states[0][1023, 54] + 1
    states[1][1023, 55] = states[0][1023, 55] + 1

    cumulative = _compare(states)
    step_local = _compare(
        states,
        reference_write_row_indices=(1023,),
        candidate_write_row_indices=(1023,),
    )

    assert cumulative.raw_mismatch_fraction == pytest.approx(2 / (1024 * 128))
    assert cumulative.acceptable
    assert step_local.reference_written_rows == 1
    assert step_local.candidate_written_rows == 1
    assert step_local.compared_rows == 1
    assert step_local.raw_mismatch_fraction == pytest.approx(2 / 128)
    assert step_local.allowed_raw_mismatch_elements == 1
    assert not step_local.one_bin_fallback_eligible
    assert not step_local.acceptable


def test_one_discrete_boundary_element_is_allowed_for_a_small_step() -> None:
    states = _states()
    states[1][127, 54] = states[0][127, 54] + 1

    result = _compare(
        states,
        reference_write_row_indices=(127,),
        candidate_write_row_indices=(127,),
    )

    assert result.compared_k_elements == 128
    assert result.raw_mismatch_elements == 1
    assert result.allowed_raw_mismatch_elements == 1
    assert result.raw_mismatch_fraction == pytest.approx(1 / 128)
    assert result.one_bin_fallback_eligible
    assert result.acceptable


def test_explicit_step_write_rows_fail_closed_on_different_or_invalid_sets() -> None:
    states = _states()

    different = _compare(
        states,
        reference_write_row_indices=(12,),
        candidate_write_row_indices=(),
    )
    assert not different.write_sets_equal
    assert not different.acceptable

    with pytest.raises(ValueError, match="provided together"):
        _compare(states, reference_write_row_indices=(12,))
    with pytest.raises(ValueError, match="duplicate"):
        _compare(
            states,
            reference_write_row_indices=(12, 12),
            candidate_write_row_indices=(12,),
        )
    with pytest.raises(ValueError, match=r"outside \[0, 128\)"):
        _compare(
            states,
            reference_write_row_indices=(128,),
            candidate_write_row_indices=(128,),
        )


@pytest.mark.parametrize(
    ("mutate", "policy"),
    (
        (lambda states: states[1].__setitem__((69, 54), states[0][69, 54] + 2), IndexerQuantizationPolicy()),
        (
            lambda states: states[1].__setitem__((slice(0, 2), slice(1, 128)), 4),
            IndexerQuantizationPolicy(),
        ),
        (
            lambda states: states[1].__setitem__((69, 54), states[0][69, 54] + 1),
            IndexerQuantizationPolicy(max_dequant_error_scale_quanta=0.5),
        ),
    ),
    ids=("two-bin", "dense-drift", "over-dequant-budget"),
)
def test_fallback_rejects_non_tie_like_raw_drift(mutate, policy) -> None:
    states = _states()
    mutate(states)

    result = _compare(states, policy=policy)

    assert not result.raw_exact
    assert not result.one_bin_fallback_eligible
    assert not result.acceptable


def test_scale_difference_is_reported_and_rejected_independently() -> None:
    states = _states()
    states[1][69, 54] = states[0][69, 54] + 1
    states[4][69, 0] = states[3][69, 0] * 1.125

    result = _compare(states)

    assert not result.raw_exact
    assert not result.scale_bit_exact
    assert result.scale_mismatch_elements == 1
    assert result.scale_max_abs_error > 0.0
    assert not result.acceptable


def test_write_set_difference_cannot_pass_even_when_active_values_match() -> None:
    states = _states()
    # Candidate row 12 remains pristine while the reference wrote it.
    states[1][12].zero_()
    states[4][12].zero_()

    result = _compare(states)

    assert not result.write_sets_equal
    assert not result.acceptable


def test_long_chain_reports_first_observed_step_and_peak_without_hiding_raw_drift() -> None:
    states = _states()
    reference_k, candidate_k, initial_k, reference_scale, candidate_scale, initial_scale = states
    reference_snapshots = []
    candidate_snapshots = []
    positions = []
    starts = []
    for step in range(93):
        candidate_step = candidate_k.clone()
        if step == 92:
            candidate_step[69, 54] = reference_k[69, 54] + 1
        reference_snapshots.append({"indexer_k": reference_k.clone(), "indexer_scale": reference_scale.clone()})
        candidate_snapshots.append({"indexer_k": candidate_step, "indexer_scale": candidate_scale.clone()})
        positions.append(
            {
                69: (
                    {
                        "request": 1,
                        "token_offset": 4,
                        "absolute_position": step * 8 + 4,
                        "decode_step": step,
                    },
                )
            }
        )
        starts.append(step * 8)

    report = compare_indexer_quantized_long_chain(
        reference_snapshots,
        candidate_snapshots,
        {"indexer_k": initial_k, "indexer_scale": initial_scale},
        row_positions_by_step=positions,
        start_positions=starts,
    )
    payload = report.to_dict()

    assert report.steps_evaluated == 93
    assert report.raw_mismatch_steps == 1
    assert report.unacceptable_steps == 0
    assert report.first_observed_divergence_step == 92
    assert report.first_observed_start_position == 736
    assert report.first_observed_mismatch is not None
    assert report.first_observed_mismatch.first_mapped_decode_step == 92
    assert report.peak_raw_mismatch_fraction == pytest.approx(1 / (128 * 128))
    assert report.peak_dequantized_max_abs_error == pytest.approx(float(reference_scale[69, 0]))
    assert report.all_steps_acceptable
    assert not report.final.raw_exact
    assert report.final.acceptable
    json.dumps(payload)


def test_policy_validation_fails_closed() -> None:
    with pytest.raises(ValueError, match="mismatch_fraction"):
        IndexerQuantizationPolicy(max_raw_mismatch_fraction=1.1)
    with pytest.raises(ValueError, match="scale_quanta"):
        IndexerQuantizationPolicy(max_dequant_error_scale_quanta=float("nan"))
