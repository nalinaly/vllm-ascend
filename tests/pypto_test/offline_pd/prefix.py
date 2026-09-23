# SPDX-License-Identifier: Apache-2.0
"""CPU serialization boundaries for the formal DSV4 offline cache bank."""


def prefix_tensor(value, logical_blocks, block_size, history, compress_ratio):
    """Clear only rows beyond the computed prefix, preserving the source dump.

    Draft forward can write predictions into the unused end of the last page.
    They are not part of h(H); D recomputes them from its own continuation.
    """
    if value.device.type != "cpu" or value.shape[:2] != (len(logical_blocks), block_size):
        raise ValueError("Expected a CPU snapshot with [logical pages, rows, ...] geometry")
    if compress_ratio < 1:
        raise ValueError("Invalid cache compression ratio")
    rows = history // compress_ratio
    valid = [max(0, min(block_size, rows - block * block_size)) for block in logical_blocks]
    if any(n == 0 for n in valid):
        raise ValueError("Snapshot contains a page outside the computed prefix")
    if any(n < block_size for n in valid):
        value = value.clone()
        for index, n in enumerate(valid):
            value[index, n:].zero_()
    return value


def cache_contract(config):
    """Expected release Native cache names and compression, including DSpark."""
    expected = {}
    for layer in range(config["num_hidden_layers"]):
        prefix = f"model.layers.{layer}.self_attn"
        expected[f"{prefix}.swa_cache__0"] = 1
        ratio = config["compress_ratios"][layer]
        if ratio > 1:
            expected[f"{prefix}.attn__0"] = ratio
            expected[f"{prefix}.compressor.state_cache__0"] = 1
        if ratio == 4:
            expected[f"{prefix}.indexer.compressor.state_cache__0"] = 1
            expected[f"{prefix}.indexer.k_cache__0"] = ratio
            expected[f"{prefix}.indexer.k_cache__1"] = ratio
    draft_layers = config.get("n_mtp_layers", config.get("dspark_num_mtp_layers", 3)) or 3
    for layer in range(draft_layers):
        expected[f"mtp.{layer}.self_attn.swa_cache__0"] = 1
    return expected
