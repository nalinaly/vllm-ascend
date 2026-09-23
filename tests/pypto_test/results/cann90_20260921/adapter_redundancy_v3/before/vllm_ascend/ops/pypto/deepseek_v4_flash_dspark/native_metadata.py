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
def prepare_compressed_metadata(
    bounds: pl.Tensor[[BOUNDS], pl.INT32],
    lengths: pl.Tensor[[REQUESTS], pl.INT32],
    positions: pl.Tensor[[TOKENS], pl.INT64],
    compact_slots: pl.Tensor[[COMPACT_ROWS, 2], pl.INT32],
    compact_cos: pl.Tensor[[COMPACT_ROWS, ROPE], pl.FP32],
    compact_sin: pl.Tensor[[COMPACT_ROWS, ROPE], pl.FP32],
    expanded_slots: pl.Out[pl.Tensor[[TOKENS], pl.INT64]],
    expanded_cos: pl.Out[pl.Tensor[[TOKENS, ROPE], pl.FP32]],
    expanded_sin: pl.Out[pl.Tensor[[TOKENS, ROPE], pl.FP32]],
):
    """Expand compact rows while preserving Native interleaved frequency columns."""
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
        compact_row = pl.read(compact_indices, [rope_token])
        cosine = pl.full([1, ROPE], dtype=pl.FP32, value=1.0)
        sine = pl.full([1, ROPE], dtype=pl.FP32, value=0.0)
        if compact_row >= 0:
            cosine = compact_cos[compact_row : compact_row + 1, :]
            sine = compact_sin[compact_row : compact_row + 1, :]
        expanded_cos[rope_token : rope_token + 1, :] = cosine
        expanded_sin[rope_token : rope_token + 1, :] = sine
    return expanded_slots, expanded_cos, expanded_sin


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
    positions: pl.Tensor[[TOKENS], pl.INT64],
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
            position = pl.read(positions, [token])
            ring_row = request * STATE_ROWS + pl.cast(position % STATE_ROWS, pl.INDEX)
            if page >= 0 and offset >= 0 and ring_row >= 0:
                for column in pl.range(0, width, STATE_COPY_TILE):
                    start = offset * width + column
                    native_state[page : page + 1, start : start + STATE_COPY_TILE] = flat[
                        ring_row : ring_row + 1, column : column + STATE_COPY_TILE
                    ]
    return native_state
