"""Probe the A3 indexer K/scale allocation as one writable argument."""

import pypto.language as pl

PAGES = pl.dynamic("PAGES")
PAGE_STORAGE = pl.dynamic("PAGE_STORAGE")
PAGE_TOKENS = 32
KEY_DIM = 128
KEY_BYTES = PAGE_TOKENS * KEY_DIM
PAGE_BYTES = KEY_BYTES + PAGE_TOKENS * 2


@pl.jit
def shared_indexer_storage_probe(
    storage: pl.InOut[pl.Tensor[[PAGES, PAGE_STORAGE], pl.INT8]],
    observed: pl.Out[pl.Tensor[[PAGES, KEY_BYTES], pl.FP32]],
):
    storage.bind_dynamic(0, PAGES)
    storage.bind_dynamic(1, PAGE_STORAGE)
    observed.bind_dynamic(0, PAGES)
    pages = pl.tensor.dim(storage, 0)
    # One task owns a whole physical page, including K and scale. Page starts
    # and both subregions are 64-byte aligned for this A3 layout.
    for page in pl.spmd(pages, name_hint="shared_indexer_storage_probe"):
        key = pl.reshape(storage[page : page + 1, :KEY_BYTES], [PAGE_TOKENS, KEY_DIM])
        scale_bytes = storage[page : page + 1, KEY_BYTES:PAGE_BYTES]
        scale = pl.reshape(pl.reinterpret_view(scale_bytes, pl.FP16), [PAGE_TOKENS, 1])
        # A3 converts INT8 through FP16. Every INT8 value is exactly
        # representable in FP16; spell out the bridge for this compiler.
        key_fp32 = pl.cast(pl.cast(key, pl.FP16), pl.FP32)
        decoded = pl.mul(key_fp32, pl.cast(scale, pl.FP32))
        observed[page : page + 1, :] = pl.reshape(decoded, [1, KEY_BYTES])
        updated_key = pl.cast(pl.mul(key_fp32, -1.0), pl.INT8)
        updated_scale = pl.cast(pl.add(pl.cast(scale, pl.FP32), 0.5), pl.FP16)
        storage[page : page + 1, :KEY_BYTES] = pl.reshape(updated_key, [1, KEY_BYTES])
        storage[page : page + 1, KEY_BYTES:PAGE_BYTES] = pl.reshape(
            pl.reinterpret_view(updated_scale, pl.INT8), [1, PAGE_TOKENS * 2]
        )
    return observed
