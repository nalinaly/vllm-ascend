# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Static PyPTO L1 programs for DeepSeek V4 Flash decode CSA on A2/A3."""

from functools import cache

import pypto.language as pl

from .config import (
    BLOCK_SIZE,
    DECODE_BATCH,
    DECODE_SEQ,
    TP,
)
from .config import (
    FLASH as M,
)
from .contract import DecodeCSAProgramSpec
from .decode_compressor_ratio4 import compressor_ratio4
from .decode_indexer import indexer
from .decode_sparse_attn_csa import sparse_attn_csa
from .qkv_proj_rope import qkv_proj_rope
from .rope_interleave import rope_interleave

EPS = M.rms_norm_eps
D = M.hidden_size
H = M.num_attention_heads
HEAD_DIM = M.head_dim
ROPE_HEAD_DIM = M.qk_rope_head_dim
HALF_ROPE = ROPE_HEAD_DIM // 2
Q_LORA = M.q_lora_rank
MAX_SEQ_LEN = M.max_position_embeddings
IDX_N_HEADS = M.index_n_heads
IDX_HEAD_DIM = M.index_head_dim
INDEXER_SCORE_LEN = MAX_SEQ_LEN // 4
INDEXER_TOPK = M.index_topk
O_LORA = M.o_lora_rank
O_GROUPS = M.o_groups
O_GROUP_IN = H * HEAD_DIM // O_GROUPS
COMPRESS_RATIO = 4
COMPRESS_OVERLAP = True
COMPRESS_OUTPUT_FACTOR = 1 + int(COMPRESS_OVERLAP)
MAIN_OUT_DIM = COMPRESS_OUTPUT_FACTOR * HEAD_DIM
INNER_OUT_DIM = COMPRESS_OUTPUT_FACTOR * IDX_HEAD_DIM
CSA_WB_TOKEN_TILE = 8
CMP_WRITES_PER_REQUEST = DECODE_SEQ // COMPRESS_RATIO
assert CMP_WRITES_PER_REQUEST == 2
B_MAX = DECODE_BATCH // TP
T_MAX = B_MAX * DECODE_SEQ
CMP_ROWS_MAX = B_MAX * CMP_WRITES_PER_REQUEST


@cache
def make_decode_csa_l1_program(
    spec: DecodeCSAProgramSpec,
    runtime: str = "tensormap_and_ringbuffer",
):
    """Return one cached, positive-static-shape L1 program for ``spec``."""
    if runtime not in {"tensormap_and_ringbuffer", "host_build_graph"}:
        raise ValueError(f"unsupported PyPTO L1 runtime: {runtime!r}")
    batch = spec.batch
    tokens = spec.tokens
    swa_blocks = spec.swa_blocks
    compressed_blocks = spec.compressed_blocks
    main_state_span = spec.main_state_span
    inner_state_span = spec.inner_state_span
    indexer_k_span = spec.indexer_k_span
    indexer_scale_span = spec.indexer_scale_span
    swa_table_width = spec.swa_table_width
    compressed_table_width = spec.compressed_table_width
    main_state_table_width = spec.main_state_table_width
    inner_state_table_width = spec.inner_state_table_width
    indexer_table_width = spec.indexer_table_width

    @pl.jit(execution="l1", runtime=runtime)
    def decode_csa_core(
        hidden_states: pl.Tensor[[tokens, D], pl.BF16],
        wq_a: pl.Tensor[[D, Q_LORA], pl.INT8],
        wq_a_scale: pl.Tensor[[Q_LORA], pl.FP32],
        wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8],
        wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32],
        wkv: pl.Tensor[[D, HEAD_DIM], pl.INT8],
        wkv_scale: pl.Tensor[[HEAD_DIM], pl.FP32],
        gamma_cq: pl.Tensor[[Q_LORA], pl.BF16],
        gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16],
        freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.FP32],
        freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.FP32],
        cmp_wkv: pl.Tensor[[MAIN_OUT_DIM, D], pl.BF16],
        cmp_wgate: pl.Tensor[[MAIN_OUT_DIM, D], pl.BF16],
        cmp_ape: pl.Tensor[[COMPRESS_RATIO, MAIN_OUT_DIM], pl.FP32],
        cmp_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
        compress_state: pl.InOut[pl.Tensor[[1, main_state_span], pl.FP32]],
        compress_state_block_table: pl.Tensor[[batch, main_state_table_width], pl.INT32],
        idx_wq_b: pl.Tensor[[Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8],
        idx_wq_b_scale: pl.Tensor[[IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32],
        weights_proj: pl.Tensor[[D, IDX_N_HEADS], pl.BF16],
        hadamard_idx: pl.Tensor[[IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16],
        inner_wkv: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16],
        inner_wgate: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16],
        inner_ape: pl.Tensor[[COMPRESS_RATIO, INNER_OUT_DIM], pl.FP32],
        inner_norm_w: pl.Tensor[[IDX_HEAD_DIM], pl.BF16],
        inner_compress_state: pl.InOut[pl.Tensor[[1, inner_state_span], pl.FP32]],
        inner_compress_state_block_table: pl.Tensor[[batch, inner_state_table_width], pl.INT32],
        kv_cache: pl.InOut[pl.Tensor[[swa_blocks, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
        cmp_kv: pl.InOut[pl.Tensor[[compressed_blocks, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
        cmp_block_table: pl.Tensor[[batch, compressed_table_width], pl.INT32],
        idx_kv_cache: pl.InOut[pl.Tensor[[1, indexer_k_span], pl.INT8]],
        idx_kv_scale: pl.InOut[pl.Tensor[[1, indexer_scale_span], pl.FP16]],
        idx_block_table: pl.Tensor[[batch, indexer_table_width], pl.INT32],
        swa_block_table: pl.Tensor[[batch, swa_table_width], pl.INT32],
        start_positions: pl.Tensor[[batch], pl.INT32],
        kv_seq_lens: pl.Tensor[[batch], pl.INT32],
        attn_sink: pl.Tensor[[H], pl.FP32],
        wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
        wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.BF16],
        attn_out: pl.Out[pl.Tensor[[tokens, D], pl.BF16]],
        main_state_page_stride: pl.Scalar[pl.INDEX] = pl.RUNTIME,
        inner_state_page_stride: pl.Scalar[pl.INDEX] = pl.RUNTIME,
        indexer_k_page_stride: pl.Scalar[pl.INDEX] = pl.RUNTIME,
        indexer_scale_page_stride: pl.Scalar[pl.INDEX] = pl.RUNTIME,
    ):
        # Inline primitives infer variable-axis metadata from this concrete
        # caller.  Keeping DynDim annotations in a child would propagate them
        # back to this public entry and produce ``-1`` L1 metadata, which the
        # runtime correctly rejects.  These two dimensions therefore come
        # from the entry's already-static tensor metadata.
        tokens = pl.tensor.dim(hidden_states, 0)
        batch = pl.tensor.dim(compress_state_block_table, 0)
        seq = DECODE_SEQ
        swa_blocks = pl.tensor.dim(kv_cache, 0)
        writeback_blocks = tokens // CSA_WB_TOKEN_TILE

        # ``torch.ops.vllm.dsa_forward`` does not expose the model's original
        # positions tensor.  The stable DSA request metadata does expose the
        # first position of each S-token target-verify row, so reconstruct the
        # consecutive token positions inside the single L1 operator.  This
        # keeps capture/replay free of a Python-side metadata allocation/copy.
        position_ids = pl.create_tensor([tokens], dtype=pl.INT32)
        rope_cos_t = pl.create_tensor([tokens, ROPE_HEAD_DIM], dtype=pl.FP32)
        rope_sin_t = pl.create_tensor([tokens, ROPE_HEAD_DIM], dtype=pl.FP32)
        token_cos_full = pl.create_tensor([tokens, ROPE_HEAD_DIM], dtype=pl.FP32)
        token_sin_full = pl.create_tensor([tokens, ROPE_HEAD_DIM], dtype=pl.FP32)
        need_index_score = pl.create_tensor([1], dtype=pl.INT32)
        token_cos_il = pl.create_tensor([T_MAX, ROPE_HEAD_DIM], dtype=pl.FP32)
        token_sin_signed = pl.create_tensor([T_MAX, ROPE_HEAD_DIM], dtype=pl.FP32)
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="csa_rope_step") as index_score_gate_tid:
            # Score/rank is needed only when at least one row has more visible
            # compressed positions than top-k can retain.  Produce the gate in
            # this already-required metadata task so the fast path adds no
            # standalone child launch.
            pl.write(need_index_score, [0], pl.cast(0, pl.INT32))
            for batch_index in pl.range(batch):
                first_token = batch_index * seq
                first_position = pl.read(start_positions, [batch_index])
                compressed_cache_len = pl.read(kv_seq_lens, [batch_index]) // COMPRESS_RATIO
                compressed_last_visible = (first_position + seq) // COMPRESS_RATIO
                if compressed_cache_len > INDEXER_TOPK:
                    if compressed_last_visible > INDEXER_TOPK:
                        pl.write(need_index_score, [0], pl.cast(1, pl.INT32))
                for token_offset in pl.range(seq):
                    token_index = first_token + token_offset
                    token_position = first_position + token_offset
                    pl.write(
                        position_ids,
                        [token_index],
                        pl.cast(token_position, pl.INT32),
                    )
                    position = pl.cast(token_position, pl.INDEX)
                    cos_row = freqs_cos[position : position + 1, 0:ROPE_HEAD_DIM]
                    sin_row = freqs_sin[position : position + 1, 0:ROPE_HEAD_DIM]
                    rope_cos_t[token_index : token_index + 1, 0:ROPE_HEAD_DIM] = cos_row
                    rope_sin_t[token_index : token_index + 1, 0:ROPE_HEAD_DIM] = sin_row
                    token_cos_full[
                        token_index : token_index + 1,
                        0:ROPE_HEAD_DIM,
                    ] = cos_row[:, 0:ROPE_HEAD_DIM]
                    token_sin_full[
                        token_index : token_index + 1,
                        0:ROPE_HEAD_DIM,
                    ] = sin_row[:, 0:ROPE_HEAD_DIM]

        rope_interleave(
            token_cos_full,
            token_sin_full,
            token_cos_il,
            token_sin_signed,
        )

        cmp_rows = batch * CMP_WRITES_PER_REQUEST
        cmp_cos = pl.create_tensor([cmp_rows, ROPE_HEAD_DIM], dtype=pl.FP32)
        cmp_sin = pl.create_tensor([cmp_rows, ROPE_HEAD_DIM], dtype=pl.FP32)
        cmp_cos_il = pl.create_tensor([CMP_ROWS_MAX, ROPE_HEAD_DIM], dtype=pl.FP32)
        cmp_sin_signed = pl.create_tensor([CMP_ROWS_MAX, ROPE_HEAD_DIM], dtype=pl.FP32)
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="csa_cmp_rope"):
            for batch_index in pl.range(batch):
                first_position = pl.read(start_positions, [batch_index])
                first_boundary = COMPRESS_RATIO - 1 - (first_position % COMPRESS_RATIO)
                for write_index in pl.range(CMP_WRITES_PER_REQUEST):
                    cmp_row = batch_index * CMP_WRITES_PER_REQUEST + write_index
                    boundary_position = first_position + first_boundary + write_index * COMPRESS_RATIO
                    cmp_position = pl.cast(
                        boundary_position + 1 - COMPRESS_RATIO,
                        pl.INDEX,
                    )
                    cmp_cos[cmp_row : cmp_row + 1, 0:ROPE_HEAD_DIM] = freqs_cos[
                        cmp_position : cmp_position + 1, 0:ROPE_HEAD_DIM
                    ]
                    cmp_sin[cmp_row : cmp_row + 1, 0:ROPE_HEAD_DIM] = freqs_sin[
                        cmp_position : cmp_position + 1, 0:ROPE_HEAD_DIM
                    ]

        rope_interleave(cmp_cos, cmp_sin, cmp_cos_il, cmp_sin_signed)

        late_dep = pl.system.task_dummy(deps=[])
        q = pl.create_tensor([tokens, H, HEAD_DIM], dtype=pl.BF16)
        kv = pl.create_tensor([tokens, HEAD_DIM], dtype=pl.BF16)
        qr = pl.create_tensor([tokens, Q_LORA], dtype=pl.INT8)
        qr_scale = pl.create_tensor([tokens, 1], dtype=pl.FP32)
        # Share qkv_proj_rope's already-required j^1 table with the indexer.
        # Declaring the tensor at this caller keeps metadata static when it is
        # forwarded into the second inline primitive.
        rope_swap_idx_t = pl.create_tensor([tokens, ROPE_HEAD_DIM], dtype=pl.INT32)
        qkv_proj_rope(
            hidden_states,
            wq_a,
            wq_a_scale,
            wq_b,
            wq_b_scale,
            wkv,
            wkv_scale,
            rope_cos_t,
            rope_sin_t,
            gamma_cq,
            gamma_ckv,
            q,
            kv,
            qr,
            qr_scale,
            rope_swap_idx_t,
            late_dep,
        )

        kv_cache_flat = pl.reshape(kv_cache, [swa_blocks * BLOCK_SIZE, HEAD_DIM])
        with pl.spmd(writeback_blocks, name_hint="csa_cache_writeback") as writeback_dep:
            writeback_block = pl.tile.get_block_idx()
            token_base = writeback_block * CSA_WB_TOKEN_TILE
            for token_offset in pl.range(CSA_WB_TOKEN_TILE):
                token_index = token_base + token_offset
                request_index = token_index // seq
                # Graph padding keeps the public bucket shape static while
                # updating this device buffer on every replay.  Resolve the
                # SWA address from the page table only for an active request;
                # padded table rows are zero and must never alias block 0.
                if pl.read(kv_seq_lens, [request_index]) > 0:
                    token_position = pl.read(position_ids, [token_index])
                    write_block_i32 = pl.read(
                        swa_block_table,
                        [request_index, token_position // BLOCK_SIZE],
                    )
                    write_block = pl.cast(write_block_i32, pl.INDEX)
                    write_offset = pl.cast(token_position % BLOCK_SIZE, pl.INDEX)
                    write_row = write_block * BLOCK_SIZE + write_offset
                    kv_cache_flat[write_row : write_row + 1, 0:HEAD_DIM] = kv[token_index : token_index + 1, 0:HEAD_DIM]

        cmp_out = pl.create_tensor([tokens, HEAD_DIM], dtype=pl.FP32)
        compressor_ratio4(
            hidden_states,
            cmp_out,
            compress_state,
            compress_state_block_table,
            cmp_wkv,
            cmp_wgate,
            cmp_ape,
            cmp_norm_w,
            cmp_cos_il,
            cmp_sin_signed,
            cmp_kv,
            cmp_block_table,
            position_ids,
            kv_seq_lens,
            main_state_page_stride,
            late_dep,
        )

        idx_kv_unused = pl.create_tensor([tokens, IDX_HEAD_DIM], dtype=pl.FP32)
        idx_score_unused = pl.create_tensor([tokens, INDEXER_SCORE_LEN], dtype=pl.FP32)
        # Downstream sparse attention consumes exactly ``index_topk`` entries.
        # Keeping the historical score-width allocation made every short-context
        # top-k task initialise 4096 indices even though only 512 can escape.
        idx_topk_full = pl.create_tensor([tokens, INDEXER_TOPK], dtype=pl.INT32)
        indexer(
            hidden_states,
            qr,
            qr_scale,
            idx_wq_b,
            idx_wq_b_scale,
            weights_proj,
            token_cos_il,
            token_sin_signed,
            cmp_cos_il,
            cmp_sin_signed,
            hadamard_idx,
            rope_swap_idx_t,
            idx_kv_unused,
            inner_compress_state,
            inner_compress_state_block_table,
            inner_wkv,
            inner_wgate,
            inner_ape,
            inner_norm_w,
            idx_kv_cache,
            idx_kv_scale,
            idx_block_table,
            idx_score_unused,
            idx_topk_full,
            position_ids,
            kv_seq_lens,
            0,
            inner_state_page_stride,
            indexer_k_page_stride,
            indexer_scale_page_stride,
            need_index_score,
            index_score_gate_tid,
            late_dep,
        )

        position_ids_t1 = pl.reshape(position_ids, [tokens, 1])
        # Keep the grouped attention scratch at this program's exact static
        # bucket extent.  Allocating it in the public entry makes its metadata
        # explicit while forwarding through the nested inline helpers.
        o_packed_heads = pl.create_tensor([O_GROUPS * tokens, O_GROUP_IN], dtype=pl.BF16)
        sparse_attn_csa(
            q,
            kv_cache,
            swa_block_table,
            cmp_kv,
            cmp_block_table,
            idx_topk_full,
            position_ids_t1,
            kv_seq_lens,
            attn_sink,
            rope_cos_t,
            rope_sin_t,
            wo_a,
            wo_b,
            o_packed_heads,
            attn_out,
            writeback_dep,
        )
        return attn_out

    return decode_csa_core
