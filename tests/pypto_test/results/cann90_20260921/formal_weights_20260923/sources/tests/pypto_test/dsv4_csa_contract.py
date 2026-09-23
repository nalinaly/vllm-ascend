"""Explicit contract checks for the standalone uniform-query CSA consumer."""

from __future__ import annotations


def validate_decode_contract(metadata, group, bounds_cpu, positions):
    """Check host-known layout/dispatch facts, without reading device contents."""
    import torch

    from vllm_ascend.attention.attention_v1 import AscendAttentionState

    request = metadata.req_metadata
    if metadata.num_prefills or metadata.attn_state != AscendAttentionState.SpecDecoding:
        raise ValueError("CSA probe supports target speculative decode only; prefill must use native")
    if bounds_cpu.device.type != "cpu":
        raise ValueError("dispatch bounds must be the scheduler's CPU tensor")
    if bounds_cpu[0] != 0 or not bool(((bounds_cpu[1:] - bounds_cpu[:-1]) == 6).all()):
        raise ValueError("CSA consumer requires uniform query length 6")
    batch = bounds_cpu.numel() - 1
    actual = int(bounds_cpu[-1])
    if request.cache_group_key != group["prefix"]:
        raise ValueError("metadata/cache group mismatch")
    if request.query_start_loc.shape != (batch + 1,) or request.seq_lens.shape != (batch,):
        raise ValueError("cross-parameter request dimensions disagree")
    if metadata.num_actual_tokens != actual or positions.numel() < actual:
        raise ValueError("token capacity is smaller than the actual query")
    if request.query_start_loc.dtype != torch.int32 or request.seq_lens.dtype != torch.int32:
        raise ValueError("query boundaries and actual lengths must be INT32")
    if positions.dtype != torch.int64 or not positions.is_contiguous():
        raise ValueError("native positions must be contiguous INT64")
    if request.block_table.shape[0] != batch or request.block_table.dtype != torch.int32:
        raise ValueError("block table request dimensions or dtype disagree")
    if request.storage_block_size != group["page_size"]:
        raise ValueError("logical and physical block units are mixed")
    compact = group["builder"].compressor_ratio == 4
    slots = request.compressor_metadata[2] if compact else request.slot_mapping
    if slots.ndim != 2 or slots.shape[1] != 2 or slots.dtype != torch.int32:
        raise ValueError("A3 slots must be INT32 block/offset pairs")
    minimum_rows = min(actual, actual // 4 + batch) if compact else actual
    if slots.shape[0] < minimum_rows:
        raise ValueError("slot storage capacity is too small")
    if any(
        tensor.device != positions.device
        for tensor in (request.query_start_loc, request.seq_lens, request.block_table, slots)
    ):
        raise ValueError("metadata tensors are not on the consumer device")
    view = group["views"][0]
    last_byte = view.storage_offset() * view.element_size()
    last_byte += (
        sum((size - 1) * stride for size, stride in zip(view.shape, view.stride(), strict=True)) * view.element_size()
    )
    if last_byte + view.element_size() > group["allocation"].untyped_storage().nbytes():
        raise ValueError("cache view exceeds the backing storage")


def validate_allocator_snapshot(group, batch, upper_bound):
    """Diagnostic preflight on the native CPU allocator mirror, never D2H.

    This scans page IDs only in the validation fixture. It is not a proposed
    production per-forward CPU scan and cannot detect arbitrary device corruption.
    """
    if upper_bound > group["table_stride"] * group["spec"].block_size:
        raise ValueError("logical page exceeds the block-table capacity")
    table_cpu = group["table"].block_table.cpu[:batch]
    if table_cpu.device.type != "cpu":
        raise ValueError("allocator snapshot must already reside on CPU")
    if bool(((table_cpu < 0) | (table_cpu >= group["pages"])).any()):
        raise ValueError("physical page exceeds the cache allocation")


def run_rejection_cases(metadata, group, bounds_cpu, positions):
    """Inject unsupported inputs and allocator corruption before any PTO call."""
    from dataclasses import replace

    import torch

    from vllm_ascend.attention.attention_v1 import AscendAttentionState

    cases = []

    def rejected(name, call):
        try:
            call()
        except ValueError as error:
            cases.append({"case": name, "status": "PASS", "rejected_before_pto_launch": True, "reason": str(error)})
        else:
            raise AssertionError(f"negative input was accepted: {name}")

    ragged = bounds_cpu.clone()
    ragged[1] -= 1
    rejected("M11_ragged_query", lambda: validate_decode_contract(metadata, group, ragged, positions))
    prefill = replace(metadata, num_prefills=1, attn_state=AscendAttentionState.ChunkedPrefill)
    rejected("M11_mixed_prefill", lambda: validate_decode_contract(prefill, group, bounds_cpu, positions))
    wrong = replace(metadata, req_metadata=replace(metadata.req_metadata, cache_group_key="wrong.cache.group"))
    rejected("M12_wrong_group", lambda: validate_decode_contract(wrong, group, bounds_cpu, positions))
    wrong_shape = replace(
        metadata, req_metadata=replace(metadata.req_metadata, seq_lens=metadata.req_metadata.seq_lens[:-1])
    )
    rejected(
        "cross_parameter_dynamic_axis", lambda: validate_decode_contract(wrong_shape, group, bounds_cpu, positions)
    )
    wrong_dtype = positions.to(torch.int32)
    rejected("position_dtype", lambda: validate_decode_contract(metadata, group, bounds_cpu, wrong_dtype))
    rejected("token_capacity", lambda: validate_decode_contract(metadata, group, bounds_cpu, positions[:1]))
    upper = group["table_stride"] * group["spec"].block_size + 1
    rejected("M12_logical_page_oob", lambda: validate_allocator_snapshot(group, len(bounds_cpu) - 1, upper))
    table_cpu = group["table"].block_table.cpu
    saved = table_cpu[0, 0].clone()
    try:
        table_cpu[0, 0] = group["pages"]
        rejected("M12_physical_page_oob", lambda: validate_allocator_snapshot(group, len(bounds_cpu) - 1, 1))
    finally:
        table_cpu[0, 0] = saved
    return cases
