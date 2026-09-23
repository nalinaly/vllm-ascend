"""Observe the production Indexer compressor at a failing continuous step."""

import pypto.language as pl

from vllm_ascend.ops.pypto.deepseek_v4_flash_dspark.decode_indexer_compressor import (
    B_DYN,
    BS_PAD,
    COMPRESS_RATIO,
    COMPRESS_STATE_BLOCK_NUM_DYN,
    COMPRESS_STATE_BLOCK_SIZE,
    COMPRESS_STATE_DIM,
    COMPRESS_STATE_MAX_BLOCKS,
    HEAD_DIM,
    IDX_CACHE_BLOCK_NUM_DYN,
    OUT_DIM,
    ROPE_HEAD_DIM,
    T_DYN,
    D,
    indexer_compressor_pool_projected,
    indexer_compressor_project,
    indexer_compressor_write,
)
from vllm_ascend.ops.pypto.deepseek_v4_flash_dspark.layout import INDEXER_PAGE_BYTES_DYN


@pl.jit(auto_scope=False)
def diagnose_indexer_compressor(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    state: pl.InOut[pl.Tensor[[COMPRESS_STATE_BLOCK_NUM_DYN, COMPRESS_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM], pl.FP32]],
    table: pl.Tensor[[B_DYN, COMPRESS_STATE_MAX_BLOCKS], pl.INT32],
    wkv: pl.Tensor[[OUT_DIM, D], pl.BF16],
    wgate: pl.Tensor[[OUT_DIM, D], pl.BF16],
    ape: pl.Tensor[[COMPRESS_RATIO, OUT_DIM], pl.FP32],
    norm: pl.Tensor[[HEAD_DIM], pl.FP32],
    cos: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.FP32],
    sin: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.FP32],
    positions: pl.Tensor[[T_DYN], pl.INT32],
    state_slots: pl.Tensor[[T_DYN], pl.INT64],
    hadamard: pl.Tensor[[HEAD_DIM, HEAD_DIM], pl.BF16],
    cache: pl.InOut[pl.Tensor[[IDX_CACHE_BLOCK_NUM_DYN, INDEXER_PAGE_BYTES_DYN], pl.INT8]],
    slots: pl.Tensor[[T_DYN], pl.INT64],
    values: pl.Out[pl.Tensor[[BS_PAD, OUT_DIM], pl.FP32]],
    scores: pl.Out[pl.Tensor[[BS_PAD, OUT_DIM], pl.FP32]],
    pooled: pl.Out[pl.Tensor[[BS_PAD, HEAD_DIM], pl.FP32]],
    normalized: pl.Out[pl.Tensor[[BS_PAD, HEAD_DIM], pl.BF16]],
    rotated: pl.Out[pl.Tensor[[T_DYN, HEAD_DIM], pl.FP32]],
):
    x.bind_dynamic(0, T_DYN)
    state.bind_dynamic(0, COMPRESS_STATE_BLOCK_NUM_DYN)
    table.bind_dynamic(0, B_DYN)
    cache.bind_dynamic(0, IDX_CACHE_BLOCK_NUM_DYN)
    cache.bind_dynamic(1, INDEXER_PAGE_BYTES_DYN)
    cos.bind_dynamic(0, T_DYN)
    sin.bind_dynamic(0, T_DYN)
    positions.bind_dynamic(0, T_DYN)
    state_slots.bind_dynamic(0, T_DYN)
    slots.bind_dynamic(0, T_DYN)
    rotated.bind_dynamic(0, T_DYN)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="compressor_diagnostic_seed") as seed:
        rotated[0:1, :] = pl.full([1, HEAD_DIM], dtype=pl.FP32, value=0.0)
    projected = indexer_compressor_project(x, wkv, wgate, values, scores, seed, seed)
    _projected, rms = indexer_compressor_pool_projected(
        values, scores, state, table, ape, norm, cos, sin, positions, state_slots, pooled, normalized, projected, seed
    )
    indexer_compressor_write(rotated, normalized, hadamard, cache, slots, positions, rms, seed)
    return state, cache, values, scores, pooled, normalized, rotated


def observe_compressor(call, state_before, captured, output_dir):
    """Run existing production helpers with each side's pre-step state."""
    import torch
    from dsv4_csa_env import write_json
    from dsv4_csa_full_compare import compare_tensor

    a = call.args
    device = a["x_normed_t"].device
    # The full CSA caller converts half-split RoPE coefficients to interleaved
    # signed coefficients before invoking this production helper.
    sign = (torch.arange(ROPE_HEAD_DIM, device=device) % 2 * 2 - 1).float()
    cosine = a["inner_freqs_cos"]
    sine = a["inner_freqs_sin"] * sign
    positions = a["position_ids"].cpu()
    tokens = ((positions + 1) % 4 == 0).nonzero().flatten()
    compact = []
    for token in tokens.tolist():
        request, offset = divmod(token, 6)
        first = 3 - int(positions[request * 6]) % 4
        compact.append(request * 2 + (offset - first) // 4)
    native_slots = captured["compressor_slots"].cpu()
    token_slots = a["idx_slot_mapping"].cpu()[tokens]
    native_rows = []
    for slot in token_slots.tolist():
        matches = ((native_slots[:, 0] * 32 + native_slots[:, 1]) == slot).nonzero().flatten()
        if matches.numel() != 1:
            raise AssertionError(f"Expected one Native compressor row for slot {slot}")
        native_rows.append(int(matches[0]))
    dump = {name: value.detach().cpu() for name, value in captured.items()}
    dump.update(hidden=a["x_normed_t"].cpu(), positions=positions, tokens=tokens, compact_rows=compact)
    for name in ("inner_ape", "inner_norm_w", "inner_wkv", "inner_wgate", "hadamard_idx"):
        dump[name] = a[name].cpu()
    dump.update(cos=cosine.cpu(), sin=sine.cpu())
    checks = {}
    for label, physical_state in state_before.items():
        state = torch.empty_like(a["inner_compress_state"])
        req = call.req["indexer_state"]
        call.ops.gather_state(physical_state, call.tables["indexer_state"], req.query_start_loc, req.seq_lens, state)
        dump[f"{label}.state_before"] = state.cpu()
        cache = a["idx_kv_cache"].clone()
        values = torch.empty((BS_PAD, OUT_DIM), dtype=torch.float32, device=device)
        scores = torch.empty_like(values)
        pooled = torch.empty((BS_PAD, HEAD_DIM), dtype=torch.float32, device=device)
        normalized = torch.empty((BS_PAD, HEAD_DIM), dtype=torch.bfloat16, device=device)
        rotated = torch.empty((positions.numel(), HEAD_DIM), dtype=torch.float32, device=device)
        diagnose_indexer_compressor(
            a["x_normed_t"],
            state,
            a["inner_compress_state_block_table"],
            a["inner_wkv"],
            a["inner_wgate"],
            a["inner_ape"],
            a["inner_norm_w"],
            cosine,
            sine,
            a["position_ids"],
            a["inner_state_slot_mapping"],
            a["hadamard_idx"],
            cache,
            a["idx_slot_mapping"],
            values,
            scores,
            pooled,
            normalized,
            rotated,
        )
        torch.npu.synchronize()
        checks[label] = {
            "normalized": compare_tensor(normalized.cpu()[compact], dump["compressor_output"][native_rows], 0, 0),
            "rotated": compare_tensor(rotated.cpu()[tokens], dump["quant_input"][native_rows].float(), 0, 0),
            "cache_vs_production": compare_tensor(cache, a["idx_kv_cache"], 0, 0),
            "value_projection": compare_tensor(
                values[: positions.numel()], dump["native_state_tokens"][:, :OUT_DIM], 0, 0
            ),
            "score_projection_with_ape": compare_tensor(
                scores[: positions.numel()].cpu() + a["inner_ape"].cpu()[positions % 4],
                dump["native_state_tokens"][:, OUT_DIM:],
                0,
                0,
            ),
        }
        dump.update(
            {
                f"{label}.values": values[: positions.numel()].cpu(),
                f"{label}.scores": scores[: positions.numel()].cpu(),
                f"{label}.pooled": pooled.cpu()[tokens],
                f"{label}.normalized": normalized.cpu()[compact],
                f"{label}.rotated": rotated.cpu()[tokens],
                f"{label}.state_after": state.cpu(),
            }
        )
    dump["native_rows"] = native_rows
    torch.save(dump, output_dir / "compressor_boundary.pt")
    write_json(output_dir / "compressor_boundary.json", checks)
    return checks
