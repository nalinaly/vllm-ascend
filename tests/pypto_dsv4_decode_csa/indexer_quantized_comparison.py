# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Evidence-oriented comparison for an indexer INT8-K/FP16-scale pair.

Raw INT8 equality is deliberately retained as a first-class metric.  A rare
one-bin difference may only pass the *separate* joint-quantization contract
when the write set and scale storage agree, the mismatch density is bounded,
and every dequantized error stays within one scale quantum.  Consequently this
module cannot turn an arbitrary integer mismatch green through the caller's
global floating-point ``atol``/``rtol``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import torch


@dataclass(frozen=True, slots=True)
class IndexerQuantizationPolicy:
    """Narrow fallback policy for a quantization-boundary disagreement.

    ``max_raw_mismatch_elements_floor`` handles the discrete small-sample
    boundary: a step writing fewer than ``1 / max_raw_mismatch_fraction`` K
    elements could otherwise never admit even one independently bounded
    one-bin rounding disagreement.  It is an absolute count floor, not a
    second density budget; all scale, bin-distance and dequantized-error gates
    still apply, and two mismatches fail the default one-row step contract.
    """

    max_raw_bin_error: int = 1
    max_raw_mismatch_fraction: float = 1.0e-4
    max_raw_mismatch_elements_floor: int = 1
    max_dequant_error_scale_quanta: float = 1.0
    dequant_absolute_slack: float = 1.0e-7
    require_scale_bit_exact: bool = True

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_raw_bin_error, bool)
            or not isinstance(self.max_raw_bin_error, int)
            or self.max_raw_bin_error < 0
        ):
            raise ValueError("max_raw_bin_error must be a non-negative integer")
        if not math.isfinite(self.max_raw_mismatch_fraction) or not 0.0 <= self.max_raw_mismatch_fraction <= 1.0:
            raise ValueError("max_raw_mismatch_fraction must be finite and within [0, 1]")
        if (
            isinstance(self.max_raw_mismatch_elements_floor, bool)
            or not isinstance(self.max_raw_mismatch_elements_floor, int)
            or self.max_raw_mismatch_elements_floor < 0
        ):
            raise ValueError("max_raw_mismatch_elements_floor must be a non-negative integer")
        if not math.isfinite(self.max_dequant_error_scale_quanta) or self.max_dequant_error_scale_quanta < 0.0:
            raise ValueError("max_dequant_error_scale_quanta must be finite and non-negative")
        if not math.isfinite(self.dequant_absolute_slack) or self.dequant_absolute_slack < 0.0:
            raise ValueError("dequant_absolute_slack must be finite and non-negative")


DEFAULT_INDEXER_QUANTIZATION_POLICY = IndexerQuantizationPolicy()


@dataclass(frozen=True, slots=True)
class IndexerFirstMismatch:
    physical_row: int
    column: int
    physical_block: int
    block_offset: int
    reference_raw: int
    candidate_raw: int
    reference_scale: float
    candidate_scale: float
    reference_dequantized: float
    candidate_dequantized: float
    dequantized_abs_error: float
    expected_token_positions: tuple[Mapping[str, int], ...]
    first_mapped_decode_step: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "physical_row": self.physical_row,
            "column": self.column,
            "physical_block": self.physical_block,
            "block_offset": self.block_offset,
            "reference_raw": self.reference_raw,
            "candidate_raw": self.candidate_raw,
            "reference_scale": self.reference_scale,
            "candidate_scale": self.candidate_scale,
            "reference_dequantized": self.reference_dequantized,
            "candidate_dequantized": self.candidate_dequantized,
            "dequantized_abs_error": self.dequantized_abs_error,
            "expected_token_positions": tuple(dict(item) for item in self.expected_token_positions),
            "first_mapped_decode_step": self.first_mapped_decode_step,
        }


@dataclass(frozen=True, slots=True)
class IndexerQuantizedComparison:
    """Joint report; ``raw_exact`` and ``acceptable`` are intentionally distinct."""

    reference_written_rows: int
    candidate_written_rows: int
    compared_rows: int
    compared_k_elements: int
    write_sets_equal: bool
    raw_exact: bool
    raw_mismatch_elements: int
    raw_mismatch_fraction: float
    allowed_raw_mismatch_elements: int
    raw_max_bin_error: int
    scale_bit_exact: bool
    scale_mismatch_elements: int
    scale_mismatch_fraction: float
    scale_max_abs_error: float
    dequantized_nonzero_error_elements: int
    dequantized_nonzero_error_fraction: float
    dequantized_max_abs_error: float
    dequantized_mean_abs_error: float
    dequantized_max_scale_quanta: float
    dequantized_over_policy_elements: int
    dequantized_over_policy_fraction: float
    finite: bool
    one_bin_fallback_eligible: bool
    acceptable: bool
    first_raw_mismatch: IndexerFirstMismatch | None
    policy: IndexerQuantizationPolicy

    def to_dict(self) -> dict[str, Any]:
        return {
            "reference_written_rows": self.reference_written_rows,
            "candidate_written_rows": self.candidate_written_rows,
            "compared_rows": self.compared_rows,
            "compared_k_elements": self.compared_k_elements,
            "write_sets_equal": self.write_sets_equal,
            "raw_exact": self.raw_exact,
            "raw_mismatch_elements": self.raw_mismatch_elements,
            "raw_mismatch_fraction": self.raw_mismatch_fraction,
            "allowed_raw_mismatch_elements": self.allowed_raw_mismatch_elements,
            "raw_max_bin_error": self.raw_max_bin_error,
            "scale_bit_exact": self.scale_bit_exact,
            "scale_mismatch_elements": self.scale_mismatch_elements,
            "scale_mismatch_fraction": self.scale_mismatch_fraction,
            "scale_max_abs_error": self.scale_max_abs_error,
            "dequantized_nonzero_error_elements": self.dequantized_nonzero_error_elements,
            "dequantized_nonzero_error_fraction": self.dequantized_nonzero_error_fraction,
            "dequantized_max_abs_error": self.dequantized_max_abs_error,
            "dequantized_mean_abs_error": self.dequantized_mean_abs_error,
            "dequantized_max_scale_quanta": self.dequantized_max_scale_quanta,
            "dequantized_over_policy_elements": self.dequantized_over_policy_elements,
            "dequantized_over_policy_fraction": self.dequantized_over_policy_fraction,
            "finite": self.finite,
            "one_bin_fallback_eligible": self.one_bin_fallback_eligible,
            "acceptable": self.acceptable,
            "first_raw_mismatch": (self.first_raw_mismatch.to_dict() if self.first_raw_mismatch is not None else None),
            "policy": asdict(self.policy),
        }


@dataclass(frozen=True, slots=True)
class IndexerQuantizedLongChainReport:
    """Temporal evidence from explicitly synchronized diagnostic snapshots."""

    steps_evaluated: int
    raw_mismatch_steps: int
    unacceptable_steps: int
    first_observed_divergence_step: int | None
    first_observed_start_position: int | None
    first_observed_mismatch: IndexerFirstMismatch | None
    peak_raw_mismatch_fraction: float
    peak_dequantized_max_abs_error: float
    all_steps_acceptable: bool
    final: IndexerQuantizedComparison

    def to_dict(self) -> dict[str, Any]:
        return {
            "steps_evaluated": self.steps_evaluated,
            "raw_mismatch_steps": self.raw_mismatch_steps,
            "unacceptable_steps": self.unacceptable_steps,
            "first_observed_divergence_step": self.first_observed_divergence_step,
            "first_observed_start_position": self.first_observed_start_position,
            "first_observed_mismatch": (
                self.first_observed_mismatch.to_dict() if self.first_observed_mismatch is not None else None
            ),
            "peak_raw_mismatch_fraction": self.peak_raw_mismatch_fraction,
            "peak_dequantized_max_abs_error": self.peak_dequantized_max_abs_error,
            "all_steps_acceptable": self.all_steps_acceptable,
            "final": self.final.to_dict(),
        }


def _rows(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 0:
        raise ValueError("quantized state tensors must have at least one dimension")
    return tensor.detach().cpu().reshape(-1, tensor.shape[-1])


def _explicit_write_mask(
    row_indices: Sequence[int],
    *,
    row_count: int,
    name: str,
) -> torch.Tensor:
    """Materialize one caller-proven step-local write set."""
    normalized = []
    for value in row_indices:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must contain only integer row indices")
        if value < 0 or value >= row_count:
            raise ValueError(f"{name} row {value} is outside [0, {row_count})")
        normalized.append(value)
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{name} must not contain duplicate row indices")
    mask = torch.zeros(row_count, dtype=torch.bool)
    if normalized:
        mask[normalized] = True
    return mask


def _validate_inputs(
    reference_k: torch.Tensor,
    candidate_k: torch.Tensor,
    initial_k: torch.Tensor,
    reference_scale: torch.Tensor,
    candidate_scale: torch.Tensor,
    initial_scale: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    if reference_k.dtype != torch.int8 or candidate_k.dtype != torch.int8 or initial_k.dtype != torch.int8:
        raise ValueError("reference, candidate and initial indexer K must all use torch.int8")
    if not reference_scale.is_floating_point():
        raise ValueError("indexer scale must use a floating-point dtype")
    if candidate_scale.dtype != reference_scale.dtype or initial_scale.dtype != reference_scale.dtype:
        raise ValueError("reference, candidate and initial scale must share dtype")
    if reference_k.shape != candidate_k.shape or reference_k.shape != initial_k.shape:
        raise ValueError("reference, candidate and initial indexer K must share shape")
    if reference_scale.shape != candidate_scale.shape or reference_scale.shape != initial_scale.shape:
        raise ValueError("reference, candidate and initial scale must share shape")
    tensors = tuple(
        _rows(tensor)
        for tensor in (
            reference_k,
            candidate_k,
            initial_k,
            reference_scale,
            candidate_scale,
            initial_scale,
        )
    )
    k_rows = tensors[:3]
    scale_rows = tensors[3:]
    if any(tensor.shape != k_rows[0].shape for tensor in k_rows[1:]):
        raise ValueError("flattened indexer K row layouts must agree")
    if any(tensor.shape != scale_rows[0].shape for tensor in scale_rows[1:]):
        raise ValueError("flattened indexer scale row layouts must agree")
    if k_rows[0].shape[0] != scale_rows[0].shape[0] or scale_rows[0].shape[1] != 1:
        raise ValueError("indexer K requires exactly one scale scalar per physical row")
    return tensors


def compare_indexer_quantized(
    reference_k: torch.Tensor,
    candidate_k: torch.Tensor,
    initial_k: torch.Tensor,
    reference_scale: torch.Tensor,
    candidate_scale: torch.Tensor,
    initial_scale: torch.Tensor,
    *,
    row_positions: Mapping[int, tuple[Mapping[str, int], ...]] | None = None,
    policy: IndexerQuantizationPolicy = DEFAULT_INDEXER_QUANTIZATION_POLICY,
    reference_write_row_indices: Sequence[int] | None = None,
    candidate_write_row_indices: Sequence[int] | None = None,
) -> IndexerQuantizedComparison:
    """Compare K and scale as one quantized object without hiding raw drift.

    By default, written rows are inferred cumulatively against ``initial_*``.
    Stateful callers may instead pass both explicit write-row sequences.  Those
    rows must come from the current invocation's independently observed deltas;
    using them prevents a dense local error from being diluted by a large
    historical cache surface.  Supplying only one side fails closed.
    """
    if not isinstance(policy, IndexerQuantizationPolicy):
        raise TypeError("policy must be IndexerQuantizationPolicy")
    ref_k, cand_k, init_k, ref_scale, cand_scale, init_scale = _validate_inputs(
        reference_k,
        candidate_k,
        initial_k,
        reference_scale,
        candidate_scale,
        initial_scale,
    )

    if (reference_write_row_indices is None) != (candidate_write_row_indices is None):
        raise ValueError("reference and candidate explicit write-row indices must be provided together")
    if reference_write_row_indices is None:
        ref_written = ref_k.ne(init_k).any(dim=1) | ref_scale.ne(init_scale).any(dim=1)
        cand_written = cand_k.ne(init_k).any(dim=1) | cand_scale.ne(init_scale).any(dim=1)
    else:
        assert candidate_write_row_indices is not None
        ref_written = _explicit_write_mask(
            reference_write_row_indices,
            row_count=ref_k.shape[0],
            name="reference_write_row_indices",
        )
        cand_written = _explicit_write_mask(
            candidate_write_row_indices,
            row_count=cand_k.shape[0],
            name="candidate_write_row_indices",
        )
    compared_row_mask = ref_written | cand_written
    compared_row_indices = torch.nonzero(compared_row_mask, as_tuple=False).reshape(-1)
    ref_active_k = ref_k[compared_row_mask]
    cand_active_k = cand_k[compared_row_mask]
    ref_active_scale = ref_scale[compared_row_mask]
    cand_active_scale = cand_scale[compared_row_mask]

    raw_error = (ref_active_k.to(torch.int16) - cand_active_k.to(torch.int16)).abs()
    raw_diff = raw_error.ne(0)
    raw_mismatch_elements = int(raw_diff.sum().item())
    compared_k_elements = raw_error.numel()
    raw_mismatch_fraction = raw_mismatch_elements / compared_k_elements if compared_k_elements else 0.0
    allowed_raw_mismatch_elements = max(
        policy.max_raw_mismatch_elements_floor,
        math.floor(compared_k_elements * policy.max_raw_mismatch_fraction),
    )
    raw_max_bin_error = int(raw_error.max().item()) if compared_k_elements else 0

    scale_error = (ref_active_scale.to(torch.float64) - cand_active_scale.to(torch.float64)).abs()
    scale_diff = scale_error.ne(0)
    scale_mismatch_elements = int(scale_diff.sum().item())
    scale_elements = scale_error.numel()
    scale_mismatch_fraction = scale_mismatch_elements / scale_elements if scale_elements else 0.0
    scale_max_abs_error = float(scale_error.max().item()) if scale_elements else 0.0

    ref_dequant = ref_active_k.to(torch.float64) * ref_active_scale.to(torch.float64)
    cand_dequant = cand_active_k.to(torch.float64) * cand_active_scale.to(torch.float64)
    dequant_error = (ref_dequant - cand_dequant).abs()
    dequant_nonzero = dequant_error.ne(0)
    dequant_nonzero_elements = int(dequant_nonzero.sum().item())
    dequant_elements = dequant_error.numel()
    dequant_nonzero_fraction = dequant_nonzero_elements / dequant_elements if dequant_elements else 0.0
    dequant_max_abs_error = float(dequant_error.max().item()) if dequant_elements else 0.0
    dequant_mean_abs_error = float(dequant_error.mean().item()) if dequant_elements else 0.0

    scale_quantum = torch.maximum(ref_active_scale.abs(), cand_active_scale.abs()).to(torch.float64)
    allowed_error = scale_quantum * policy.max_dequant_error_scale_quanta + policy.dequant_absolute_slack
    over_policy = dequant_error.gt(allowed_error)
    over_policy_elements = int(over_policy.sum().item())
    over_policy_fraction = over_policy_elements / dequant_elements if dequant_elements else 0.0
    if dequant_elements:
        positive_quantum = scale_quantum.gt(0).expand_as(dequant_error)
        scale_quanta = torch.zeros_like(dequant_error)
        scale_quanta[positive_quantum] = (
            dequant_error[positive_quantum] / scale_quantum.expand_as(dequant_error)[positive_quantum]
        )
        scale_quanta[~positive_quantum & dequant_error.gt(policy.dequant_absolute_slack)] = torch.inf
        dequant_max_scale_quanta = float(scale_quanta.max().item())
    else:
        dequant_max_scale_quanta = 0.0

    first_mismatch = None
    if raw_mismatch_elements:
        active_row, column = (int(value) for value in torch.nonzero(raw_diff, as_tuple=False)[0].tolist())
        physical_row = int(compared_row_indices[active_row].item())
        ref_raw = int(ref_active_k[active_row, column].item())
        cand_raw = int(cand_active_k[active_row, column].item())
        ref_scale_value = float(ref_active_scale[active_row, 0].item())
        cand_scale_value = float(cand_active_scale[active_row, 0].item())
        locations = tuple(row_positions.get(physical_row, ())) if row_positions is not None else ()
        mapped_steps = tuple(int(item["decode_step"]) for item in locations if "decode_step" in item)
        first_mismatch = IndexerFirstMismatch(
            physical_row=physical_row,
            column=column,
            physical_block=physical_row // 32,
            block_offset=physical_row % 32,
            reference_raw=ref_raw,
            candidate_raw=cand_raw,
            reference_scale=ref_scale_value,
            candidate_scale=cand_scale_value,
            reference_dequantized=ref_raw * ref_scale_value,
            candidate_dequantized=cand_raw * cand_scale_value,
            dequantized_abs_error=abs(ref_raw * ref_scale_value - cand_raw * cand_scale_value),
            expected_token_positions=locations,
            first_mapped_decode_step=min(mapped_steps) if mapped_steps else None,
        )

    finite = bool(
        torch.isfinite(ref_active_scale).all()
        and torch.isfinite(cand_active_scale).all()
        and torch.isfinite(ref_dequant).all()
        and torch.isfinite(cand_dequant).all()
    )
    write_sets_equal = bool(torch.equal(ref_written, cand_written))
    scale_bit_exact = scale_mismatch_elements == 0
    raw_exact = raw_mismatch_elements == 0
    scale_contract = scale_bit_exact if policy.require_scale_bit_exact else finite
    one_bin_fallback_eligible = bool(
        not raw_exact
        and write_sets_equal
        and scale_contract
        and raw_max_bin_error <= policy.max_raw_bin_error
        and raw_mismatch_elements <= allowed_raw_mismatch_elements
        and over_policy_elements == 0
        and finite
    )
    acceptable = bool(write_sets_equal and scale_contract and finite and (raw_exact or one_bin_fallback_eligible))
    return IndexerQuantizedComparison(
        reference_written_rows=int(ref_written.sum().item()),
        candidate_written_rows=int(cand_written.sum().item()),
        compared_rows=int(compared_row_mask.sum().item()),
        compared_k_elements=compared_k_elements,
        write_sets_equal=write_sets_equal,
        raw_exact=raw_exact,
        raw_mismatch_elements=raw_mismatch_elements,
        raw_mismatch_fraction=raw_mismatch_fraction,
        allowed_raw_mismatch_elements=allowed_raw_mismatch_elements,
        raw_max_bin_error=raw_max_bin_error,
        scale_bit_exact=scale_bit_exact,
        scale_mismatch_elements=scale_mismatch_elements,
        scale_mismatch_fraction=scale_mismatch_fraction,
        scale_max_abs_error=scale_max_abs_error,
        dequantized_nonzero_error_elements=dequant_nonzero_elements,
        dequantized_nonzero_error_fraction=dequant_nonzero_fraction,
        dequantized_max_abs_error=dequant_max_abs_error,
        dequantized_mean_abs_error=dequant_mean_abs_error,
        dequantized_max_scale_quanta=dequant_max_scale_quanta,
        dequantized_over_policy_elements=over_policy_elements,
        dequantized_over_policy_fraction=over_policy_fraction,
        finite=finite,
        one_bin_fallback_eligible=one_bin_fallback_eligible,
        acceptable=acceptable,
        first_raw_mismatch=first_mismatch,
        policy=policy,
    )


def compare_indexer_quantized_long_chain(
    reference_snapshots: Sequence[Mapping[str, torch.Tensor]],
    candidate_snapshots: Sequence[Mapping[str, torch.Tensor]],
    initial: Mapping[str, torch.Tensor],
    *,
    row_positions_by_step: Sequence[Mapping[int, tuple[Mapping[str, int], ...]]] | None = None,
    start_positions: Sequence[int] | None = None,
    policy: IndexerQuantizationPolicy = DEFAULT_INDEXER_QUANTIZATION_POLICY,
) -> IndexerQuantizedLongChainReport:
    """Summarize the first and peak disagreement across diagnostic snapshots.

    Snapshot collection itself is intentionally outside this Host helper.  An
    A3 runner must label it diagnostic because taking per-step Host snapshots
    introduces synchronization and therefore is not evidence for the normal
    asynchronous launch path.
    """
    if not reference_snapshots or len(reference_snapshots) != len(candidate_snapshots):
        raise ValueError("reference and candidate snapshot sequences must have equal non-zero length")
    steps = len(reference_snapshots)
    if row_positions_by_step is not None and len(row_positions_by_step) != steps:
        raise ValueError("row_positions_by_step must match the snapshot count")
    if start_positions is not None and len(start_positions) != steps:
        raise ValueError("start_positions must match the snapshot count")
    required = {"indexer_k", "indexer_scale"}
    if not required.issubset(initial):
        raise ValueError("initial snapshot must contain indexer_k and indexer_scale")

    comparisons = []
    for step, (reference, candidate) in enumerate(zip(reference_snapshots, candidate_snapshots, strict=True)):
        if not required.issubset(reference) or not required.issubset(candidate):
            raise ValueError(f"step {step} snapshots must contain indexer_k and indexer_scale")
        comparisons.append(
            compare_indexer_quantized(
                reference["indexer_k"],
                candidate["indexer_k"],
                initial["indexer_k"],
                reference["indexer_scale"],
                candidate["indexer_scale"],
                initial["indexer_scale"],
                row_positions=(row_positions_by_step[step] if row_positions_by_step is not None else None),
                policy=policy,
            )
        )
    first_step = next((index for index, item in enumerate(comparisons) if not item.raw_exact), None)
    return IndexerQuantizedLongChainReport(
        steps_evaluated=steps,
        raw_mismatch_steps=sum(not item.raw_exact for item in comparisons),
        unacceptable_steps=sum(not item.acceptable for item in comparisons),
        first_observed_divergence_step=first_step,
        first_observed_start_position=(
            int(start_positions[first_step]) if first_step is not None and start_positions is not None else None
        ),
        first_observed_mismatch=(comparisons[first_step].first_raw_mismatch if first_step is not None else None),
        peak_raw_mismatch_fraction=max(item.raw_mismatch_fraction for item in comparisons),
        peak_dequantized_max_abs_error=max(item.dequantized_max_abs_error for item in comparisons),
        all_steps_acceptable=all(item.acceptable for item in comparisons),
        final=comparisons[-1],
    )


__all__ = [
    "DEFAULT_INDEXER_QUANTIZATION_POLICY",
    "IndexerFirstMismatch",
    "IndexerQuantizationPolicy",
    "IndexerQuantizedComparison",
    "IndexerQuantizedLongChainReport",
    "compare_indexer_quantized",
    "compare_indexer_quantized_long_chain",
]
