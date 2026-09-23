"""Device-side metadata transport probe, with explicit physical storage strides."""

import pypto.language as pl

REQUESTS = pl.dynamic("REQUESTS")
QUERY_BOUNDARIES = pl.dynamic("QUERY_BOUNDARIES")
TOKENS = pl.dynamic("TOKENS")
SLOT_ROWS = pl.dynamic("SLOT_ROWS")
TABLE_STORAGE = pl.dynamic("TABLE_STORAGE")
CACHE_STORAGE = pl.dynamic("CACHE_STORAGE")
PROBE_COLUMNS = 16
PROBE_WORKERS = 24


@pl.jit
def metadata_probe(
    query_start_loc: pl.Tensor[[QUERY_BOUNDARIES], pl.INT32],
    seq_lens: pl.Tensor[[REQUESTS], pl.INT32],
    positions: pl.Tensor[[TOKENS], pl.INT64],
    block_table_storage: pl.Tensor[[TABLE_STORAGE], pl.INT32],
    slots: pl.Tensor[[SLOT_ROWS, 2], pl.INT32],
    cache_storage: pl.InOut[pl.Tensor[[CACHE_STORAGE], pl.INT32]],
    table_row_stride: pl.Scalar[pl.INT64],
    table_storage_offset: pl.Scalar[pl.INT64],
    page_size: pl.Scalar[pl.INT64],
    cache_page_stride: pl.Scalar[pl.INT64],
    cache_storage_offset: pl.Scalar[pl.INT64],
    position_divisor: pl.Scalar[pl.INT64],
    cache_token_stride: pl.Scalar[pl.INT64],
    out: pl.Out[pl.Tensor[[TOKENS, PROBE_COLUMNS], pl.INT64]],
):
    query_start_loc.bind_dynamic(0, QUERY_BOUNDARIES)
    seq_lens.bind_dynamic(0, REQUESTS)
    positions.bind_dynamic(0, TOKENS)
    block_table_storage.bind_dynamic(0, TABLE_STORAGE)
    slots.bind_dynamic(0, SLOT_ROWS)
    cache_storage.bind_dynamic(0, CACHE_STORAGE)
    out.bind_dynamic(0, TOKENS)
    token_capacity = pl.tensor.dim(positions, 0)
    request_capacity = pl.tensor.dim(seq_lens, 0)
    with pl.spmd(PROBE_WORKERS, name_hint="dsv4_csa_metadata_probe") as read_tid:
        worker = pl.tile.get_block_idx()
        for token in pl.range(worker, token_capacity, PROBE_WORKERS):
            # Every row is written, including padding. All content-dependent
            # decisions below are device scalar operations inside the task.
            for column in pl.unroll(PROBE_COLUMNS):
                pl.write(out, [token, column], pl.cast(-1, pl.INT64))
            for request in pl.range(request_capacity):
                begin = pl.read(query_start_loc, [request])
                end = pl.read(query_start_loc, [request + 1])
                if token >= begin and token < end:
                    length = pl.read(seq_lens, [request])
                    position = pl.read(positions, [token])
                    slot_block = pl.read(slots, [token, 0])
                    slot_offset = pl.read(slots, [token, 1])
                    if slot_block >= 0 and slot_offset >= 0:
                        cache_position = position // position_divisor
                        logical_page = cache_position // page_size
                        offset = cache_position % page_size
                        table_index = table_storage_offset + request * table_row_stride + logical_page
                        physical_page = pl.read(block_table_storage, [table_index])
                        cache_index = (
                            cache_storage_offset + physical_page * cache_page_stride + offset * cache_token_stride
                        )
                        sentinel = pl.read(cache_storage, [cache_index])
                        flat_slot = pl.cast(slot_block, pl.INT64) * page_size + slot_offset
                        pl.write(out, [token, 0], pl.cast(request, pl.INT64))
                        pl.write(out, [token, 1], pl.cast(begin, pl.INT64))
                        pl.write(out, [token, 2], pl.cast(end, pl.INT64))
                        pl.write(out, [token, 3], pl.cast(length, pl.INT64))
                        pl.write(out, [token, 4], position)
                        pl.write(out, [token, 5], pl.cast(length - (end - begin), pl.INT64))
                        pl.write(out, [token, 6], logical_page)
                        pl.write(out, [token, 7], pl.cast(physical_page, pl.INT64))
                        pl.write(out, [token, 8], offset)
                        pl.write(out, [token, 9], pl.cast(physical_page, pl.INT64) * page_size + offset)
                        pl.write(out, [token, 10], flat_slot)
                        pl.write(out, [token, 11], pl.cast(sentinel, pl.INT64))
                        pl.write(out, [token, 12], pl.cast(1, pl.INT64))
                        pl.write(out, [token, 13], pl.cast(end - begin, pl.INT64))
                        pl.write(out, [token, 14], pl.cast(token - begin, pl.INT64))
                        pl.write(out, [token, 15], pl.cast(length // 4, pl.INT64))
    # Probe cache entries are only four bytes: multiple token workers would
    # share a scalar-write cache line. Commit these diagnostic sentinels from
    # one task after all readers. Production KV writes must instead own whole
    # aligned rows/lines; this is deliberately not a performance kernel.
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="dsv4_csa_probe_commit", deps=[read_tid]):
        valid_tokens = pl.read(query_start_loc, [request_capacity])
        for token in pl.range(token_capacity):
            if token < valid_tokens:
                slot_block = pl.read(slots, [token, 0])
                slot_offset = pl.read(slots, [token, 1])
                if slot_block >= 0 and slot_offset >= 0:
                    cache_index = (
                        cache_storage_offset + slot_block * cache_page_stride + slot_offset * cache_token_stride
                    )
                    position = pl.read(positions, [token])
                    pl.write(cache_storage, [cache_index], pl.cast(position % 1000000, pl.INT32))
    return out


@pl.jit
def expand_compressed_slots(
    query_start_loc: pl.Tensor[[QUERY_BOUNDARIES], pl.INT32],
    seq_lens: pl.Tensor[[REQUESTS], pl.INT32],
    positions: pl.Tensor[[TOKENS], pl.INT64],
    compact_slots: pl.Tensor[[SLOT_ROWS, 2], pl.INT32],
    expanded: pl.Out[pl.Tensor[[TOKENS, 2], pl.INT32]],
):
    """Expand native compact C4 output rows without reading device data on host."""
    query_start_loc.bind_dynamic(0, QUERY_BOUNDARIES)
    seq_lens.bind_dynamic(0, REQUESTS)
    positions.bind_dynamic(0, TOKENS)
    compact_slots.bind_dynamic(0, SLOT_ROWS)
    expanded.bind_dynamic(0, TOKENS)
    tokens = pl.tensor.dim(positions, 0)
    requests = pl.tensor.dim(seq_lens, 0)
    # Diagnostic implementation owns all scalar writes in a single task.
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="csa_expand_compact_slots"):
        for token in pl.range(tokens):
            pl.write(expanded, [token, 0], pl.cast(-1, pl.INT32))
            pl.write(expanded, [token, 1], pl.cast(-1, pl.INT32))
        prefix = pl.cast(0, pl.INDEX)
        for request in pl.range(requests):
            begin = pl.read(query_start_loc, [request])
            end = pl.read(query_start_loc, [request + 1])
            length = pl.read(seq_lens, [request])
            start = length - (end - begin)
            for token in pl.range(begin, end):
                position = pl.read(positions, [token])
                if (position + 1) % 4 == 0:
                    compact = prefix + (position + 1) // 4 - start // 4 - 1
                    block = pl.read(compact_slots, [compact, 0])
                    offset = pl.read(compact_slots, [compact, 1])
                    if block >= 0:
                        pl.write(expanded, [token, 0], block)
                        pl.write(expanded, [token, 1], offset)
            prefix = prefix + length // 4 - start // 4
    return expanded
