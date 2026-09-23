# SPDX-License-Identifier: Apache-2.0
"""Device adapters for native metadata and bounded compressor state windows.

These kernels move metadata/storage only. Attention math stays in the pinned
TP1 computation chain. No request content is inspected by the host adapter.
"""

import pypto.language as pl

REQUESTS = pl.dynamic("CSA_NATIVE_REQUESTS")
BOUNDS = pl.dynamic("CSA_NATIVE_BOUNDS")
TOKENS = pl.dynamic("CSA_NATIVE_TOKENS")
COMPACT_ROWS = pl.dynamic("CSA_NATIVE_COMPACT_ROWS")
TABLE_COLUMNS = pl.dynamic("CSA_NATIVE_TABLE_COLUMNS")
PAGES = pl.dynamic("CSA_NATIVE_PAGES")
PAGE_ELEMENTS = pl.dynamic("CSA_NATIVE_PAGE_ELEMENTS")
STATE_WIDTH = pl.dynamic("CSA_NATIVE_STATE_WIDTH")
SCRATCH_PAGES = pl.dynamic("CSA_NATIVE_SCRATCH_PAGES")
WINDOW = 128
ROPE = 64
PAGE_TOKENS = 32
STATE_PAGE_TOKENS = 2
STATE_HISTORY = 8
STATE_ROWS = 14
STATE_PAGES = STATE_ROWS // STATE_PAGE_TOKENS
STATE_COPY_TILE = 512


@pl.jit(auto_scope=False)
def prepare_token_metadata(
    bounds: pl.Tensor[[BOUNDS], pl.INT32],
    lengths: pl.Tensor[[REQUESTS], pl.INT32],
    positions: pl.Tensor[[TOKENS], pl.INT64],
    swa_table: pl.Tensor[[REQUESTS, TABLE_COLUMNS], pl.INT32],
    swa_slots: pl.Tensor[[TOKENS, 2], pl.INT32],
    state_slots: pl.Tensor[[TOKENS, 2], pl.INT32],
    inner_state_slots: pl.Tensor[[TOKENS, 2], pl.INT32],
    positions_i32: pl.Out[pl.Tensor[[TOKENS], pl.INT32]],
    original_slots: pl.Out[pl.Tensor[[TOKENS], pl.INT64]],
    ring_slots: pl.Out[pl.Tensor[[TOKENS], pl.INT64]],
    inner_ring_slots: pl.Out[pl.Tensor[[TOKENS], pl.INT64]],
    window_indices: pl.Out[pl.Tensor[[TOKENS, WINDOW], pl.INT32]],
    window_lengths: pl.Out[pl.Tensor[[TOKENS], pl.INT32]],
    ring_table: pl.Out[pl.Tensor[[REQUESTS, STATE_PAGES], pl.INT32]],
):
    bounds.bind_dynamic(0, BOUNDS)
    lengths.bind_dynamic(0, REQUESTS)
    positions.bind_dynamic(0, TOKENS)
    swa_table.bind_dynamic(1, TABLE_COLUMNS)
    # One owner for the small scalar arrays prevents cache-line sharing.
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="csa_native_token_metadata"):
        for request in pl.range(pl.tensor.dim(lengths, 0)):
            begin = pl.read(bounds, [request])
            end = pl.read(bounds, [request + 1])
            for page in pl.unroll(STATE_PAGES):
                pl.write(ring_table, [request, page], pl.cast(request * STATE_PAGES + page, pl.INT32))
            for token in pl.range(begin, end):
                position = pl.read(positions, [token])
                pl.write(positions_i32, [token], pl.cast(position, pl.INT32))
                original_page = pl.read(swa_slots, [token, 0])
                original_offset = pl.read(swa_slots, [token, 1])
                pl.write(original_slots, [token], pl.cast(-1, pl.INT64))
                if original_page >= 0 and original_offset >= 0:
                    pl.write(original_slots, [token], pl.cast(original_page, pl.INT64) * PAGE_TOKENS + original_offset)
                ring_row = request * STATE_ROWS + position % STATE_ROWS
                pl.write(ring_slots, [token], pl.cast(-1, pl.INT64))
                pl.write(inner_ring_slots, [token], pl.cast(-1, pl.INT64))
                if pl.read(state_slots, [token, 0]) >= 0 and pl.read(state_slots, [token, 1]) >= 0:
                    pl.write(ring_slots, [token], pl.cast(ring_row, pl.INT64))
                if pl.read(inner_state_slots, [token, 0]) >= 0 and pl.read(inner_state_slots, [token, 1]) >= 0:
                    pl.write(inner_ring_slots, [token], pl.cast(ring_row, pl.INT64))
                window_length = pl.min(position + 1, WINDOW)
                window_start = position - window_length + 1
                pl.write(window_lengths, [token], pl.cast(window_length, pl.INT32))
                for column in pl.range(WINDOW):
                    slot = pl.cast(-1, pl.INT32)
                    if column < window_length:
                        absolute = window_start + column
                        physical_page = pl.read(swa_table, [request, absolute // PAGE_TOKENS])
                        if physical_page >= 0:
                            slot = pl.cast(physical_page * PAGE_TOKENS + absolute % PAGE_TOKENS, pl.INT32)
                    pl.write(window_indices, [token, column], slot)
    return positions_i32, original_slots, ring_slots, inner_ring_slots, window_indices, window_lengths, ring_table


@pl.jit(auto_scope=False)
def prepare_rope(
    native_cos: pl.Tensor[[TOKENS, ROPE], pl.FP32],
    native_sin: pl.Tensor[[TOKENS, ROPE], pl.FP32],
    half_cos: pl.Out[pl.Tensor[[TOKENS, ROPE], pl.FP32]],
    half_sin: pl.Out[pl.Tensor[[TOKENS, ROPE], pl.FP32]],
):
    native_cos.bind_dynamic(0, TOKENS)
    with pl.spmd(pl.tensor.dim(native_cos, 0), name_hint="csa_native_rope"):
        token = pl.tile.get_block_idx()
        columns = pl.cast(pl.arange(0, [1, ROPE], dtype=pl.INT32), pl.FP32)
        half = pl.cast(pl.cast(pl.mul(columns, 1.0 / (ROPE // 2)), pl.INT32, mode="trunc"), pl.FP32)
        source_columns = pl.cast(pl.mul(pl.sub(columns, pl.mul(half, ROPE // 2)), 2.0), pl.INT32)
        half_cos[token : token + 1, :] = pl.gather(native_cos[token : token + 1, :], -1, source_columns)
        half_sin[token : token + 1, :] = pl.gather(native_sin[token : token + 1, :], -1, source_columns)
    return half_cos, half_sin


@pl.jit(auto_scope=False)
def prepare_compressed_metadata(
    bounds: pl.Tensor[[BOUNDS], pl.INT32],
    lengths: pl.Tensor[[REQUESTS], pl.INT32],
    positions: pl.Tensor[[TOKENS], pl.INT64],
    compact_slots: pl.Tensor[[COMPACT_ROWS, 2], pl.INT32],
    compact_cos: pl.Tensor[[COMPACT_ROWS, ROPE], pl.FP32],
    compact_sin: pl.Tensor[[COMPACT_ROWS, ROPE], pl.FP32],
    expanded_slots: pl.Out[pl.Tensor[[TOKENS], pl.INT64]],
    half_cos: pl.Out[pl.Tensor[[TOKENS, ROPE], pl.FP32]],
    half_sin: pl.Out[pl.Tensor[[TOKENS, ROPE], pl.FP32]],
):
    bounds.bind_dynamic(0, BOUNDS)
    lengths.bind_dynamic(0, REQUESTS)
    positions.bind_dynamic(0, TOKENS)
    compact_slots.bind_dynamic(0, COMPACT_ROWS)
    compact_indices = pl.create_tensor([pl.tensor.dim(positions, 0)], dtype=pl.INT32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="csa_native_compact_metadata") as slots_ready:
        prefix = pl.cast(0, pl.INDEX)
        for request in pl.range(pl.tensor.dim(lengths, 0)):
            begin = pl.read(bounds, [request])
            end = pl.read(bounds, [request + 1])
            length = pl.read(lengths, [request])
            start = length - (end - begin)
            for token in pl.range(begin, end):
                pl.write(expanded_slots, [token], pl.cast(-1, pl.INT64))
                pl.write(compact_indices, [token], pl.cast(-1, pl.INT32))
                position = pl.read(positions, [token])
                if (position + 1) % 4 == 0:
                    compact = prefix + (position + 1) // 4 - start // 4 - 1
                    pl.write(compact_indices, [token], pl.cast(compact, pl.INT32))
                    page = pl.read(compact_slots, [compact, 0])
                    offset = pl.read(compact_slots, [compact, 1])
                    if page >= 0 and offset >= 0:
                        pl.write(expanded_slots, [token], pl.cast(page, pl.INT64) * PAGE_TOKENS + offset)
            prefix = prefix + length // 4 - start // 4
    with pl.spmd(pl.tensor.dim(positions, 0), name_hint="csa_expand_compact_rope", deps=[slots_ready]):
        rope_token = pl.tile.get_block_idx()
        columns = pl.cast(pl.arange(0, [1, ROPE], dtype=pl.INT32), pl.FP32)
        half = pl.cast(pl.cast(pl.mul(columns, 1.0 / (ROPE // 2)), pl.INT32, mode="trunc"), pl.FP32)
        source_columns = pl.cast(pl.mul(pl.sub(columns, pl.mul(half, ROPE // 2)), 2.0), pl.INT32)
        compact_row = pl.read(compact_indices, [rope_token])
        cosine = pl.full([1, ROPE], dtype=pl.FP32, value=1.0)
        sine = pl.full([1, ROPE], dtype=pl.FP32, value=0.0)
        if compact_row >= 0:
            cosine = pl.gather(compact_cos[compact_row : compact_row + 1, :], -1, source_columns)
            sine = pl.gather(compact_sin[compact_row : compact_row + 1, :], -1, source_columns)
        half_cos[rope_token : rope_token + 1, :] = cosine
        half_sin[rope_token : rope_token + 1, :] = sine
    return expanded_slots, half_cos, half_sin


@pl.jit(auto_scope=False)
def gather_state_window(
    native_state: pl.Tensor[[PAGES, PAGE_ELEMENTS], pl.FP32],
    table: pl.Tensor[[REQUESTS, TABLE_COLUMNS], pl.INT32],
    bounds: pl.Tensor[[BOUNDS], pl.INT32],
    lengths: pl.Tensor[[REQUESTS], pl.INT32],
    scratch: pl.Out[pl.Tensor[[SCRATCH_PAGES, STATE_PAGE_TOKENS, STATE_WIDTH], pl.FP32]],
):
    native_state.bind_dynamic(0, PAGES)
    native_state.bind_dynamic(1, PAGE_ELEMENTS)
    table.bind_dynamic(0, REQUESTS)
    table.bind_dynamic(1, TABLE_COLUMNS)
    bounds.bind_dynamic(0, BOUNDS)
    scratch.bind_dynamic(0, SCRATCH_PAGES)
    scratch.bind_dynamic(2, STATE_WIDTH)
    width = pl.tensor.dim(scratch, 2)
    flat = pl.reshape(scratch, [pl.tensor.dim(scratch, 0) * STATE_PAGE_TOKENS, width])
    with pl.spmd(pl.tensor.dim(lengths, 0), name_hint="csa_gather_state_window"):
        request = pl.tile.get_block_idx()
        begin = pl.read(bounds, [request])
        end = pl.read(bounds, [request + 1])
        first_position = pl.read(lengths, [request]) - (end - begin)
        for relative in pl.range(STATE_ROWS):
            absolute = first_position - STATE_HISTORY + relative
            ring = (absolute % STATE_ROWS + STATE_ROWS) % STATE_ROWS
            destination = request * STATE_ROWS + ring
            for column in pl.range(0, width, STATE_COPY_TILE):
                values = pl.full([1, STATE_COPY_TILE], dtype=pl.FP32, value=0.0)
                if relative < STATE_HISTORY and absolute >= 0:
                    page = pl.read(table, [request, absolute // STATE_PAGE_TOKENS])
                    if page >= 0:
                        offset = (absolute % STATE_PAGE_TOKENS) * width + column
                        values = native_state[page : page + 1, offset : offset + STATE_COPY_TILE]
                flat[destination : destination + 1, column : column + STATE_COPY_TILE] = values
    return scratch


@pl.jit(auto_scope=False)
def commit_state_window(
    scratch: pl.Tensor[[SCRATCH_PAGES, STATE_PAGE_TOKENS, STATE_WIDTH], pl.FP32],
    native_slots: pl.Tensor[[TOKENS, 2], pl.INT32],
    ring_slots: pl.Tensor[[TOKENS], pl.INT64],
    bounds: pl.Tensor[[BOUNDS], pl.INT32],
    lengths: pl.Tensor[[REQUESTS], pl.INT32],
    native_state: pl.InOut[pl.Tensor[[PAGES, PAGE_ELEMENTS], pl.FP32]],
):
    scratch.bind_dynamic(0, SCRATCH_PAGES)
    scratch.bind_dynamic(2, STATE_WIDTH)
    native_slots.bind_dynamic(0, TOKENS)
    bounds.bind_dynamic(0, BOUNDS)
    lengths.bind_dynamic(0, REQUESTS)
    native_state.bind_dynamic(0, PAGES)
    native_state.bind_dynamic(1, PAGE_ELEMENTS)
    width = pl.tensor.dim(scratch, 2)
    flat = pl.reshape(scratch, [pl.tensor.dim(scratch, 0) * STATE_PAGE_TOKENS, width])
    with pl.spmd(pl.tensor.dim(lengths, 0), name_hint="csa_commit_state_window"):
        request = pl.tile.get_block_idx()
        begin = pl.read(bounds, [request])
        end = pl.read(bounds, [request + 1])
        for token in pl.range(begin, end):
            page = pl.read(native_slots, [token, 0])
            offset = pl.read(native_slots, [token, 1])
            ring_row = pl.read(ring_slots, [token])
            if page >= 0 and offset >= 0 and ring_row >= 0:
                for column in pl.range(0, width, STATE_COPY_TILE):
                    start = offset * width + column
                    native_state[page : page + 1, start : start + STATE_COPY_TILE] = flat[
                        ring_row : ring_row + 1, column : column + STATE_COPY_TILE
                    ]
    return native_state
