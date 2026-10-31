# SPDX-License-Identifier: Apache-2.0
"""Emit the existing flat slot and DSA page/offset in the same kernel.

This entry is for unsplit physical blocks and CP=1. Other block-table paths
keep the upstream implementation. The flat output and Native padding values
are preserved; consumers do not need a cache-layout conversion.
"""

from vllm.triton_utils import tl, triton


@triton.jit
def compute_dsa_slot_mapping_kernel(
    num_tokens,
    max_num_tokens,
    query_start_loc_ptr,
    positions_ptr,
    block_table_ptr,
    block_table_stride,
    slot_mapping_ptr,
    dsa_slot_mapping_ptr,
    secondary_slot_mapping_ptr,
    CACHE_BLOCK_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    WRITE_SECOND: tl.constexpr,
):
    request = tl.program_id(0)
    if request == tl.num_programs(0) - 1:
        for start in range(num_tokens, max_num_tokens, BLOCK_SIZE):
            offsets = start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < max_num_tokens
            tl.store(slot_mapping_ptr + offsets, -1, mask=mask)
            # Native format_dsa_slot_mapping(-1, K) is [-1, K-1].
            tl.store(dsa_slot_mapping_ptr + 2 * offsets, -1, mask=mask)
            tl.store(dsa_slot_mapping_ptr + 2 * offsets + 1, CACHE_BLOCK_SIZE - 1, mask=mask)
            if WRITE_SECOND:
                tl.store(secondary_slot_mapping_ptr + 2 * offsets, -1, mask=mask)
                tl.store(secondary_slot_mapping_ptr + 2 * offsets + 1, CACHE_BLOCK_SIZE - 1, mask=mask)
        return

    begin = tl.load(query_start_loc_ptr + request).to(tl.int64)
    end = tl.load(query_start_loc_ptr + request + 1).to(tl.int64)
    for start in range(begin, end, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < end
        positions = tl.load(positions_ptr + offsets, mask=mask, other=0)
        columns = positions // CACHE_BLOCK_SIZE
        pages = tl.load(
            block_table_ptr + request * block_table_stride + columns, mask=mask, other=0
        ).to(tl.int64)
        within_page = positions - columns * CACHE_BLOCK_SIZE
        tl.store(slot_mapping_ptr + offsets, pages * CACHE_BLOCK_SIZE + within_page, mask=mask)
        tl.store(dsa_slot_mapping_ptr + 2 * offsets, pages, mask=mask)
        tl.store(dsa_slot_mapping_ptr + 2 * offsets + 1, within_page, mask=mask)
        if WRITE_SECOND:
            tl.store(secondary_slot_mapping_ptr + 2 * offsets, pages, mask=mask)
            tl.store(secondary_slot_mapping_ptr + 2 * offsets + 1, within_page, mask=mask)
