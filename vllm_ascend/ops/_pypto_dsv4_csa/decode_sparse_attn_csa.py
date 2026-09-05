# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 CSA sparse attention with grouped output projection (decode).

Ratio-4 compressed cache plus the sliding window, with the indexer top-k
masking folded in. The SWA and HCA variants live in sibling modules.
"""

import pypto.language as pl

from .config import (
    BLOCK_SIZE,
    DECODE_BATCH,
    DECODE_SEQ,
    KV_CMP_BLOCK_NUM,
    KV_CMP_MAX_BLOCKS,
    KV_ORI_BLOCK_NUM,
    KV_ORI_MAX_BLOCKS,
    TP,
)
from .config import (
    FLASH as M,
)

# Dynamic shape variables.

# model config
B = DECODE_BATCH // TP
S = DECODE_SEQ
D = M.hidden_size
H = M.num_attention_heads
HEAD_DIM = M.head_dim
ROPE_DIM = M.qk_rope_head_dim
HALF_ROPE = ROPE_DIM // 2
NOPE_DIM = M.nope_head_dim
WIN = M.sliding_window
MAX_SEQ_LEN = M.max_position_embeddings
IDX_TOPK = M.index_topk
CMP_TOPK = IDX_TOPK
SOFTMAX_SCALE = M.softmax_scale
O_LORA = M.o_lora_rank
O_GROUPS = M.o_groups
HEADS_PER_GROUP = H // O_GROUPS
O_GROUP_IN = HEADS_PER_GROUP * HEAD_DIM
COMPRESS_RATIO = 4
COMPRESS_RATIO_INV = 1.0 / COMPRESS_RATIO
INDEXER_SCORE_LEN = MAX_SEQ_LEN // 4
CSA_CMP_GE_BIAS = 1.0  # raw + 1, folded for the ge clamp
NEG_INF = -1.0e20

# paged KV cache
ORI_MAX_BLOCKS = KV_ORI_MAX_BLOCKS
ORI_BLOCK_NUM = KV_ORI_BLOCK_NUM
CMP_MAX_BLOCKS = KV_CMP_MAX_BLOCKS
CMP_BLOCK_NUM = KV_CMP_BLOCK_NUM

# tiling
H_TILE = 16
MERGE_HEAD_TILES = H // H_TILE
GROUPS_PER_MERGE_HEAD_TILE = H_TILE // HEADS_PER_GROUP
assert H % H_TILE == 0
assert H_TILE % HEADS_PER_GROUP == 0
assert MERGE_HEAD_TILES * GROUPS_PER_MERGE_HEAD_TILE == O_GROUPS
QK_M_TILE = 32  # two head tiles share each gathered KV tile in one QK/PV pair
ATTN_K_TILE = 128
A_K_TILE = 256  # proj_a cube K frag
PROJ_A_MM_N_TILE = 128  # proj_a cube N frag
ROPE_CS_T_TILE = 8  # rope cos/sin row block; T is a multiple of 8 by the batch contract
PROJ_A_ROW_TILE = 32  # B4/S8 fills one exact cube-M block; halve proj-a blocks/weight scans
PA_N_FRAGS = O_LORA // PROJ_A_MM_N_TILE
B_K_TILE = 256  # proj_b_mm cube K frag
# proj_b_mm cube N frag; Acc = MM_T_TILE*N*4 = 128KB sits exactly on the a2a3 L0C wall.
PROJ_B_MM_N_TILE = 256
PROJ_B_ACT_N_TILE = 512  # proj_b reduction/output vector N frag
PROJ_B_ACT_N_REGS = D // PROJ_B_ACT_N_TILE
PROJ_B_D_TILE = 512  # proj_b_mm D chunk per task; its N frags loop inside the task
PROJ_B_ACT_T_TILE = 8  # proj_b_act inner token tile for the O_GROUPS-way FP32 accumulate
PROJ_B_ACT_TASK_T_TILE = 8  # proj_b_act token block per task
TOPK = WIN + CMP_TOPK
# Floor to 2: a single sparse-K block miscompiles in pypto (S-stride cross-token
# output mixup); a 2-block build with an all-invalid 2nd block is bit-exact.
SPARSE_BLOCKS = max(2, (TOPK + ATTN_K_TILE - 1) // ATTN_K_TILE)
PADDED_TOPK = SPARSE_BLOCKS * ATTN_K_TILE
assert PADDED_TOPK == TOPK, "the mature qk-plan fast path requires an exact sparse-block cover"
assert COMPRESS_RATIO * IDX_TOPK >= WIN, "a full compressed top-k must imply a full sliding window"
# Page-contiguous runs one sliding-window K tile spans. WIN, not the K tile size,
# caps how many window rows a tile can hold; BLOCK_SIZE only sets where the cuts
# fall, being where physical contiguity breaks. So: those rows plus a worst-case
# BLOCK_SIZE - 1 head offset, rounded up to pages -- 2 whenever WIN <= BLOCK_SIZE,
# whatever ATTN_K_TILE is, and it grows on its own if either outgrows a page.
SWA_TILE_WIN_ROWS = min(ATTN_K_TILE, WIN)
SWA_RUNS = (SWA_TILE_WIN_ROWS + 2 * (BLOCK_SIZE - 1)) // BLOCK_SIZE
# Token tile for the slot / bias vector work; the whole-T form would put
# [T, IDX_TOPK] FP32 tiles well past the Vec limit.
BIAS_T_TILE = 8


@pl.jit.inline
def sparse_attn_csa_heads(
    q: pl.Tensor,
    ori_kv: pl.Tensor,
    swa_block_table: pl.Tensor,
    cmp_kv: pl.Tensor,
    cmp_block_table: pl.Tensor,
    idx_topk: pl.Tensor,
    position_ids: pl.Tensor,
    kv_seq_lens: pl.Tensor,
    attn_sink: pl.Tensor[[H], pl.FP32],
    freqs_cos: pl.Tensor,
    freqs_sin: pl.Tensor,
    o_packed_heads: pl.Tensor,
    writeback_dep: pl.Scalar[pl.TASK_ID],
) -> tuple[
    pl.Tensor,
    pl.Scalar[pl.TASK_ID],
    pl.Scalar[pl.TASK_ID],
    pl.Scalar[pl.TASK_ID],
    pl.Scalar[pl.TASK_ID],
]:
    """Write CSA heads as ``[group, T_PAD, O_GROUP_IN]`` slabs.

    Only the first runtime ``t_dim`` rows in each group are valid. The
    returned four scalar task IDs each cover one disjoint head tile and its two
    packed output groups.
    """
    # Compressed index contract: -1 invalid, [0, ...) compressed KV slots.
    ori_block_num = pl.tensor.dim(ori_kv, 0)
    # The public L1 entry is separately compiled for every supported bucket.
    # Read its already-static token extent through ``shape`` so the JIT
    # specializer folds it into this inline child.  Using the module-wide B16
    # ceiling here used to make B4/B8/B12 allocate and compute 128 rows on every
    # replay even though their true extents are 32/64/96.
    t_dim = q.shape[0]
    t_heads = t_dim * H
    t_blk = t_dim * (H // H_TILE) * SPARSE_BLOCKS * H_TILE
    qk_items = t_dim * SPARSE_BLOCKS
    rope_cs_blocks = t_dim // ROPE_CS_T_TILE
    ori_kv_flat = pl.reshape(ori_kv, [ori_block_num * BLOCK_SIZE, HEAD_DIM])

    # qk_plan compacts the live (token, sparse-block) work items into qk_order[].
    # Block 0 is retained for every token: active rows compute the window and a
    # padded row uses its fallback to seed finite zeros for merge_norm. Empty
    # optional compressed blocks are omitted entirely, so AIC lanes do not walk
    # a long tail of dispatch items that immediately fail valid_block_mask.
    sparse_bias = pl.create_tensor([t_dim, PADDED_TOPK], dtype=pl.FP32)
    cmp_sparse_indices = pl.create_tensor([t_dim, CMP_TOPK], dtype=pl.INT32)
    valid_block_mask = pl.create_tensor([t_dim, SPARSE_BLOCKS], dtype=pl.INT32)
    qk_order = pl.create_tensor([qk_items], dtype=pl.INT32)
    # Slot 0 is the compacted work-item count.  Slot 1 carries the plan's
    # all-full predicate across the task boundary so mature decode consumers
    # can skip hundreds of redundant valid_block_mask GM reads.  Keeping both
    # values in this private plan-state tensor changes neither the public ABI
    # nor ACLGraph bindings.
    qk_wcur = pl.create_tensor([2], dtype=pl.INT32)
    with pl.at(
        level=pl.Level.CORE_GROUP, name_hint="csa_slots_build_valid_qk_plan", allow_early_resolve=True
    ) as qk_plan_tid:
        # indexer's private offset is fixed to zero by decode_csa_core.  Its
        # top-k output is therefore a valid prefix of length min(visible, 512):
        # visible<=512 writes [0..visible), while visible>512 writes 512 valid
        # selected indices.  If every token has 512 visible positions, all five
        # sparse blocks are full and the expensive generic revalidation/bias
        # construction below is equivalent to direct copy + zero-fill.
        # Keep the branch predicate in the executing AIV's scalar SSA.  A GM
        # scratch flag is not a sound same-kernel rendezvous: this task first
        # writes the flag while scanning the tokens and immediately reads it
        # to select a branch, so cache visibility can make a boundary token
        # (511 visible rows) incorrectly take the all-full path.  Reducing the
        # minimum visible length locally also removes one workspace object and
        # all of its per-token scalar stores.
        full_cache_len = pl.read(kv_seq_lens, [0]) // COMPRESS_RATIO
        full_pos = pl.read(position_ids, [0, 0])
        full_visible_min = pl.min(
            pl.min(full_cache_len, (full_pos + 1) // COMPRESS_RATIO),
            INDEXER_SCORE_LEN,
        )
        for full_t in pl.unroll(1, t_dim):
            full_b = full_t // S
            full_cache_len = pl.read(kv_seq_lens, [full_b]) // COMPRESS_RATIO
            full_pos = pl.read(position_ids, [full_t, 0])
            full_visible = pl.min(
                pl.min(full_cache_len, (full_pos + 1) // COMPRESS_RATIO),
                INDEXER_SCORE_LEN,
            )
            full_visible_min = pl.min(full_visible_min, full_visible)

        if full_visible_min >= IDX_TOPK:
            for bias_t0 in pl.range(0, t_dim, BIAS_T_TILE):
                # Use an explicit side-effecting store in both runtime
                # branches.  Treating the destination as a functional tensor
                # assignment makes the current frontend form a tensor-valued
                # phi; that phi can be bound incorrectly to the branch predicate
                # scratch when the following qk_pv task is outlined.
                full_indices = pl.load(
                    idx_topk,
                    [bias_t0, 0],
                    [BIAS_T_TILE, IDX_TOPK],
                )
                cmp_sparse_indices = pl.store(full_indices, [bias_t0, 0], cmp_sparse_indices)
                # qk_pv and merge_norm consume qk_wcur[1] before consulting
                # sparse_bias/valid_block_mask.  In all-full mode the bias is
                # algebraic zero and every block is live, so materializing
                # either table would be dead GM traffic.

            # All optional blocks are present, so construct the exact same
            # cost-aware order directly: all compressed blocks token-major,
            # then every SWA seed block.  This removes 160 mask reads and
            # cursor read-modify-writes from the mature path.
            for full_plan_t in pl.unroll(t_dim):
                for full_plan_sb in pl.unroll(1, SPARSE_BLOCKS):
                    full_order_slot = full_plan_t * (SPARSE_BLOCKS - 1) + full_plan_sb - 1
                    pl.write(
                        qk_order,
                        [full_order_slot],
                        pl.cast(full_plan_t * SPARSE_BLOCKS + full_plan_sb, pl.INT32),
                    )
            for full_plan_t in pl.unroll(t_dim):
                pl.write(
                    qk_order,
                    [t_dim * (SPARSE_BLOCKS - 1) + full_plan_t],
                    pl.cast(full_plan_t * SPARSE_BLOCKS, pl.INT32),
                )
            pl.write(qk_wcur, [0], pl.cast(qk_items, pl.INT32))
            pl.write(qk_wcur, [1], pl.cast(1, pl.INT32))

        if full_visible_min < IDX_TOPK:
            # Generic path: vectorized masked copy over a token tile, keeping
            # raw iff 0 <= raw < floor((pos + 1) / COMPRESS_RATIO), as
            # out = mask*(raw + 1) - 1.
            for bias_t0 in pl.range(0, t_dim, BIAS_T_TILE):
                c_raw = pl.cast(idx_topk[bias_t0 : bias_t0 + BIAS_T_TILE, 0:IDX_TOPK], target_type=pl.FP32)
                c_pos = pl.cast(position_ids[bias_t0 : bias_t0 + BIAS_T_TILE, 0:1], target_type=pl.FP32)
                c_pos_scaled = pl.mul(pl.add(c_pos, 1.0), COMPRESS_RATIO_INV)
                c_pos_i32 = pl.cast(c_pos_scaled, target_type=pl.INT32, mode="trunc")
                c_pos_q = pl.cast(c_pos_i32, target_type=pl.FP32)
                # Broadcast the per-token bound over IDX_TOPK cols.
                c_upper_b = pl.row_expand_mul(pl.full([BIAS_T_TILE, IDX_TOPK], dtype=pl.FP32, value=1.0), c_pos_q)
                c_ge = pl.minimum(pl.maximum(pl.add(c_raw, CSA_CMP_GE_BIAS), 0.0), 1.0)
                c_lt = pl.minimum(pl.maximum(pl.sub(c_upper_b, c_raw), 0.0), 1.0)
                c_mask = pl.mul(c_ge, c_lt)
                c_out = pl.sub(pl.mul(c_mask, pl.add(c_raw, 1.0)), 1.0)
                c_out_i32 = pl.cast(c_out, target_type=pl.INT32)
                cmp_sparse_indices = pl.assemble(cmp_sparse_indices, c_out_i32, [bias_t0, 0])
                # Block 0 is live only for a real request.  Uniform graph padding
                # leaves the corresponding SWA table row zeroed, so marking it
                # live would read physical block 0 during replay.
                for c_t0 in pl.range(BIAS_T_TILE):
                    c_token = bias_t0 + c_t0
                    if pl.read(kv_seq_lens, [c_token // S]) > 0:
                        pl.write(valid_block_mask, [c_token, 0], pl.cast(1, pl.INT32))
                    else:
                        pl.write(valid_block_mask, [c_token, 0], pl.cast(0, pl.INT32))
                for c_sb in pl.range(1, SPARSE_BLOCKS):
                    c_s0 = c_sb * ATTN_K_TILE - WIN
                    c_blk_valid = pl.row_max(c_mask[:, c_s0 : c_s0 + ATTN_K_TILE])
                    for c_dt in pl.range(BIAS_T_TILE):
                        c_valid = pl.cast(pl.read(c_blk_valid, [c_dt, 0]), target_type=pl.INT32)
                        pl.write(valid_block_mask, [bias_t0 + c_dt, c_sb], c_valid)

                # Additive softmax bias (0 valid / NEG_INF invalid) that qk_pv adds onto the
                # scaled scores, so invalid lanes exp to ~0 with no per-block mask multiply.
                # The production wrapper already owns the SWA block table but has no
                # per-token [T, WIN] address tensor.  Build the causal validity mask
                # from absolute positions; physical addresses are resolved from the
                # table only when qk_pv gathers a page-contiguous run below.
                v_win_len = pl.minimum(pl.add(c_pos, 1.0), WIN * 1.0)
                v_upper = pl.row_expand_mul(
                    pl.full([BIAS_T_TILE, WIN], dtype=pl.FP32, value=1.0),
                    v_win_len,
                )
                v_cols = pl.col_expand_mul(
                    pl.full([BIAS_T_TILE, WIN], dtype=pl.FP32, value=1.0),
                    pl.cast(
                        pl.arange(0, [1, WIN], dtype=pl.INT32),
                        target_type=pl.FP32,
                    ),
                )
                v_win_valid = pl.minimum(
                    pl.maximum(pl.sub(v_upper, v_cols), 0.0),
                    1.0,
                )
                v_win_bias = pl.mul(pl.sub(v_win_valid, 1.0), -NEG_INF)
                sparse_bias = pl.assemble(sparse_bias, v_win_bias, [bias_t0, 0])
                c_sparse_bias = pl.mul(
                    pl.minimum(c_out, 0.0),
                    -NEG_INF,
                )
                sparse_bias = pl.assemble(sparse_bias, c_sparse_bias, [bias_t0, WIN])

            pl.write(qk_wcur, [0], pl.cast(0, pl.INT32))
            pl.write(qk_wcur, [1], pl.cast(0, pl.INT32))
            # Generic path keeps every seed block plus non-empty optional
            # blocks. Write random-gather compressed work first and append the
            # contiguous SWA seeds afterwards. qk_pv consumes this queue with
            # a blockDim stride; token-major [seed, compressed...] ordering
            # would pin each AIC lane to one sparse-block kind.
            for plan_t in pl.unroll(t_dim):
                for plan_sb in pl.unroll(1, SPARSE_BLOCKS):
                    if plan_t < t_dim:
                        if pl.read(valid_block_mask, [plan_t, plan_sb]) > 0:
                            plan_w = pl.read(qk_wcur, [0])
                            pl.write(
                                qk_order,
                                [plan_w],
                                pl.cast(plan_t * SPARSE_BLOCKS + plan_sb, pl.INT32),
                            )
                            pl.write(qk_wcur, [0], pl.cast(plan_w + 1, pl.INT32))
            for plan_t in pl.unroll(t_dim):
                if plan_t < t_dim:
                    plan_w = pl.read(qk_wcur, [0])
                    pl.write(qk_order, [plan_w], pl.cast(plan_t * SPARSE_BLOCKS, pl.INT32))
                    pl.write(qk_wcur, [0], pl.cast(plan_w + 1, pl.INT32))

    # One lane per core. Each lane walks its planned items and gathers the
    # window/compressed KV rows into one L1 matmul operand; invalid lanes gather a
    # finite row and are zeroed by the NEG_INF softmax bias.
    cmp_block_num = pl.tensor.dim(cmp_kv, 0)
    cmp_kv_flat = pl.reshape(cmp_kv, [cmp_block_num * BLOCK_SIZE, HEAD_DIM])
    q_flat = pl.reshape(q, [t_heads, HEAD_DIM])
    sparse_blk_mi = pl.create_tensor([t_blk, 1], dtype=pl.FP32)
    sparse_blk_li = pl.create_tensor([t_blk, 1], dtype=pl.FP32)
    sparse_blk_oi = pl.create_tensor([t_blk, HEAD_DIM], dtype=pl.FP32)

    with pl.spmd(
        pl.system.available_cluster_count(),
        name_hint="qk_pv",
        deps=[qk_plan_tid, writeback_dep],
        allow_early_resolve=True,
    ) as _qk_tid:
        qk_core = pl.tile.get_block_idx()
        # The usable AIC count is a runtime property (for example, this A3 SKU
        # exposes 20 rather than the target family's maximum 24).  Match the
        # launch width to that count and derive the lane stride from blockDim so
        # no oversubscribed second wave becomes the critical path.
        qk_core_count = pl.tile.get_block_num()
        qk_valid_items = pl.read(qk_wcur, [0])
        qk_all_full = pl.read(qk_wcur, [1])
        qk_lane_iters = (qk_valid_items - qk_core + qk_core_count - 1) // qk_core_count
        for qk_it in pl.range(qk_lane_iters):
            qk_flat = qk_core + qk_it * qk_core_count
            qk_item = pl.cast(pl.read(qk_order, [qk_flat]), pl.INDEX)
            qk_t = qk_item // SPARSE_BLOCKS
            qk_sb = qk_item - qk_t * SPARSE_BLOCKS
            qk_b = qk_t // S
            qk_token_base = qk_t * (H // H_TILE) * SPARSE_BLOCKS * H_TILE
            qk_s0 = qk_sb * ATTN_K_TILE
            qk_bias_row = pl.full([1, ATTN_K_TILE], dtype=pl.FP32, value=0.0)
            if qk_all_full < 1:
                qk_bias_row = sparse_bias[qk_t : qk_t + 1, qk_s0 : qk_s0 + ATTN_K_TILE]
            # qk_order contains only valid work in both modes.  The generic
            # path still consults the mask to preserve its padded-row seed
            # fallback; the mature all-full path avoids one scalar GM read per
            # item while retaining the same body and stores.
            qk_block_valid = qk_all_full
            if qk_all_full < 1:
                qk_block_valid = pl.read(valid_block_mask, [qk_t, qk_sb])
            if qk_block_valid > 0:
                qk_kv = pl.create_l1([ATTN_K_TILE, HEAD_DIM], pl.BF16)
                # Sliding-window rows of this tile: all ATTN_K_TILE of them at
                # WIN == ATTN_K_TILE, none for a compressed tile.
                qk_win_rows = pl.min(pl.max(WIN - qk_s0, 0), ATTN_K_TILE)
                if qk_win_rows > 0:
                    # The window is consecutive absolute positions and paged KV keeps one
                    # page's positions in consecutive rows, so these rows are SWA_RUNS
                    # page-contiguous runs -- one multi-row gather each (row count carried
                    # by valid_shape) instead of a single-row DMA per row. Visible length
                    # and start mirror the metadata producers
                    # (decode_metadata.build_swa_metadata / utils.swa_indices_and_lens).
                    qk_pos = pl.cast(pl.read(position_ids, [qk_t, 0]), pl.INDEX)
                    qk_win_len = pl.min(qk_pos + 1, WIN)
                    qk_win_start = qk_pos - qk_win_len + 1
                    qk_run_rows = pl.min(pl.max(qk_win_len - qk_s0, 0), qk_win_rows)
                    # qk_head is how far into its page this tile's first window row sits,
                    # so run i holds the rows landing in the i-th page the tile touches:
                    # [i * BLOCK_SIZE - qk_head, (i + 1) * BLOCK_SIZE - qk_head) clipped to
                    # [0, qk_run_rows). Run 0 is the short one, every later run is page
                    # aligned, and runs past the end clip empty -- no carried cursor.
                    qk_head = (qk_win_start + qk_s0) % BLOCK_SIZE
                    for qk_run in pl.unroll(SWA_RUNS):
                        qk_run_lo = pl.max(qk_run * BLOCK_SIZE - qk_head, 0)
                        qk_run_hi = pl.min((qk_run + 1) * BLOCK_SIZE - qk_head, qk_run_rows)
                        if qk_run_hi > qk_run_lo:
                            qk_run_pos = qk_win_start + qk_s0 + qk_run_lo
                            qk_run_block_i32 = pl.read(
                                swa_block_table,
                                [qk_b, qk_run_pos // BLOCK_SIZE],
                            )
                            # Exact B/S production gating guarantees every visible
                            # logical block is mapped.  Keep row zero as a defensive
                            # finite fallback so malformed metadata cannot create an
                            # out-of-range DMA before the wrapper rejects it.
                            qk_run_block = pl.cast(
                                pl.max(qk_run_block_i32, 0),
                                pl.INDEX,
                            )
                            qk_run_src = qk_run_block * BLOCK_SIZE + qk_run_pos % BLOCK_SIZE
                            qk_kv = pl.gather_row(
                                qk_kv,
                                ori_kv_flat,
                                [qk_run_lo, 0],
                                [qk_run_src, 0],
                                [ATTN_K_TILE, HEAD_DIM],
                                valid_shape=[qk_run_hi - qk_run_lo, HEAD_DIM],
                            )
                    qk_tail_n = qk_win_rows - qk_run_rows
                    if qk_tail_n > 0:
                        # Slots past the visible window still need finite data so their
                        # NEG_INF-biased lanes exp to ~0 instead of reading stale L1.
                        qk_kv = pl.gather_row(
                            qk_kv,
                            ori_kv_flat,
                            [qk_run_rows, 0],
                            [0, 0],
                            [ATTN_K_TILE, HEAD_DIM],
                            valid_shape=[qk_tail_n, HEAD_DIM],
                        )
                # Compressed rows stay per-row: the indexer top-k slots are
                # scattered, while every invalid lane receives finite fallback
                # data and is suppressed by its NEG_INF sparse bias below.
                for qk_r in pl.range(qk_win_rows, ATTN_K_TILE):
                    qk_cmp_k = qk_s0 + qk_r - WIN
                    if qk_cmp_k < CMP_TOPK:
                        qk_ridx = pl.read(cmp_sparse_indices, [qk_t, qk_cmp_k])
                        if qk_ridx >= 0:
                            qk_slot = qk_ridx
                            qk_cblk = pl.cast(
                                pl.read(cmp_block_table, [qk_b, qk_slot // BLOCK_SIZE]),
                                pl.INDEX,
                            )
                            qk_csrc = qk_cblk * BLOCK_SIZE + qk_slot % BLOCK_SIZE
                            qk_kv = pl.gather_row(
                                qk_kv,
                                cmp_kv_flat,
                                [qk_r, 0],
                                [qk_csrc, 0],
                                [1, HEAD_DIM],
                            )
                        else:
                            qk_kv = pl.gather_row(
                                qk_kv,
                                ori_kv_flat,
                                [qk_r, 0],
                                [0, 0],
                                [1, HEAD_DIM],
                            )
                    else:
                        qk_kv = pl.gather_row(
                            qk_kv,
                            ori_kv_flat,
                            [qk_r, 0],
                            [0, 0],
                            [1, HEAD_DIM],
                        )

                # Cube-batch two head tiles per QK/PV matmul so the shared KV
                # tile is extracted L1->L0 once instead of once per head-tile. The
                # [QK_M_TILE, ...] softmax result is sliced back into H_TILE-row
                # stores at the SAME offsets as the per-head-tile path
                # (qk_h_idx == qk_hb * (QK_M_TILE // H_TILE) + qk_sub), so the
                # sparse_blk_* layout and merge_norm are bit-identical.
                for qk_hb in pl.pipeline(H // QK_M_TILE, stage=2):
                    qk_h0 = qk_hb * QK_M_TILE
                    qk_head_row = qk_t * H + qk_h0
                    qk_q_tile = q_flat[qk_head_row : qk_head_row + QK_M_TILE, 0:HEAD_DIM]
                    qk_raw = pl.matmul(qk_q_tile, qk_kv, b_trans=True, out_dtype=pl.FP32)
                    qk_scaled = pl.mul(qk_raw, SOFTMAX_SCALE)
                    # col_expand_add also materializes the row-major Vec
                    # layout required by row_max.  The mature path feeds its
                    # local zero row; only generic mode reads bias from GM.
                    qk_scores = pl.col_expand_add(qk_scaled, qk_bias_row)
                    qk_mi = pl.row_max(qk_scores)
                    # Invalid lanes (NEG_INF bias, zero kv rows) exp to ~0; all-invalid
                    # blocks die in the merge alpha/beta -- no mask multiply needed.
                    qk_exp = pl.exp(pl.row_expand_sub(qk_scores, qk_mi))
                    qk_li = pl.row_sum(qk_exp)
                    qk_exp_bf16 = pl.cast(qk_exp, target_type=pl.BF16, mode="rint")
                    qk_oi = pl.matmul(qk_exp_bf16, qk_kv, out_dtype=pl.FP32)
                    for qk_sub in pl.unroll(QK_M_TILE // H_TILE):
                        qk_h_idx = qk_hb * (QK_M_TILE // H_TILE) + qk_sub
                        qk_r0 = qk_sub * H_TILE
                        qk_blk_base = qk_token_base + qk_h_idx * SPARSE_BLOCKS * H_TILE
                        qk_row = qk_blk_base + qk_sb * H_TILE
                        sparse_blk_mi[qk_row : qk_row + H_TILE, 0:1] = qk_mi[qk_r0 : qk_r0 + H_TILE, 0:1]
                        sparse_blk_li[qk_row : qk_row + H_TILE, 0:1] = qk_li[qk_r0 : qk_r0 + H_TILE, 0:1]
                        sparse_blk_oi[qk_row : qk_row + H_TILE, 0:HEAD_DIM] = qk_oi[qk_r0 : qk_r0 + H_TILE, 0:HEAD_DIM]
            else:
                # Block 0 seeds merge_norm.  It is valid for every active
                # request; only a graph-padded request reaches this fallback.
                # Later invalid blocks have no contribution and are not
                # materialized in GM because merge_norm skips their reads.
                if qk_sb == 0:
                    qk_oi_zero = pl.full([H_TILE, HEAD_DIM], dtype=pl.FP32, value=0.0)
                    for qk_h_idx in pl.range(H // H_TILE):
                        qk_blk_base = qk_token_base + qk_h_idx * SPARSE_BLOCKS * H_TILE
                        qk_row = qk_blk_base
                        for qk_hr in pl.range(H_TILE):
                            pl.write(sparse_blk_mi, [qk_row + qk_hr, 0], -3.0e38)
                            pl.write(sparse_blk_li, [qk_row + qk_hr, 0], 0.0)
                        sparse_blk_oi[qk_row : qk_row + H_TILE, 0:HEAD_DIM] = qk_oi_zero

    # Head-invariant interleaved cos and sign-folded sin, built once per token.
    # The conjugate (inverse) rotation is out[j] = x[j]*cos_il[j] + x[j^1]*sign[j]*sin_il[j].
    rope_cos_il = pl.create_tensor([t_dim, ROPE_DIM], dtype=pl.FP32)
    rope_sin_signed = pl.create_tensor([t_dim, ROPE_DIM], dtype=pl.FP32)
    # j^1 lane-swap index for merge_norm's rotation gather. Shaped [H_TILE, ROPE_DIM]
    # because gather's index must match its source rows.
    rope_swap_idx = pl.create_tensor([H_TILE, ROPE_DIM], dtype=pl.INT32)
    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="rope_cs",
        allow_early_resolve=True,
    ) as rope_cs_tid:
        sw_ones = pl.full([H_TILE, ROPE_DIM], dtype=pl.FP32, value=1.0)
        sw_idx_f = pl.cast(pl.arange(0, [1, ROPE_DIM], dtype=pl.INT32), target_type=pl.FP32)
        sw_col = pl.col_expand_mul(sw_ones, sw_idx_f)
        sw_dup_i32 = pl.cast(pl.mul(sw_col, 0.5), target_type=pl.INT32, mode="trunc")
        sw_dup_f = pl.cast(sw_dup_i32, target_type=pl.FP32)
        sw_lane = pl.sub(sw_col, pl.mul(sw_dup_f, 2.0))  # j%2
        sw_swap_f = pl.sub(pl.add(sw_col, 1.0), pl.mul(sw_lane, 2.0))  # j^1
        rope_swap_idx[0:H_TILE, 0:ROPE_DIM] = pl.cast(sw_swap_f, target_type=pl.INT32)

        cs_ones = pl.full([ROPE_CS_T_TILE, ROPE_DIM], dtype=pl.FP32, value=1.0)
        cs_idx_f = pl.cast(pl.arange(0, [1, ROPE_DIM], dtype=pl.INT32), target_type=pl.FP32)
        cs_col = pl.col_expand_mul(cs_ones, cs_idx_f)
        cs_dup_i32 = pl.cast(pl.mul(cs_col, 0.5), target_type=pl.INT32, mode="trunc")
        cs_dup_f = pl.cast(cs_dup_i32, target_type=pl.FP32)
        cs_lane = pl.sub(cs_col, pl.mul(cs_dup_f, 2.0))  # j%2
        cs_sign = pl.neg(pl.sub(pl.mul(cs_lane, 2.0), 1.0))  # [+1,-1,...] (conjugate)
        for cs_rb in pl.range(rope_cs_blocks):
            cs_t0 = cs_rb * ROPE_CS_T_TILE
            cs_cos = freqs_cos[cs_t0 : cs_t0 + ROPE_CS_T_TILE, 0:ROPE_DIM]
            cs_sin = freqs_sin[cs_t0 : cs_t0 + ROPE_CS_T_TILE, 0:ROPE_DIM]
            rope_cos_il[cs_t0 : cs_t0 + ROPE_CS_T_TILE, 0:ROPE_DIM] = cs_cos
            rope_sin_signed[cs_t0 : cs_t0 + ROPE_CS_T_TILE, 0:ROPE_DIM] = pl.mul(cs_sin, cs_sign)

    # Online-softmax merge across sparse-K tiles, sink-norm, then fused inverse
    # RoPE.  Keep one independent task per head tile instead of one 128-block
    # task spanning all four tiles.  Every tile owns exactly two output groups,
    # so its TaskId lets those groups start proj_a on AIC while the other AIV
    # merge tasks are still running.  The per-(token, head-tile) arithmetic and
    # GM offsets remain identical to the monolithic implementation.
    merge_tids = pl.array.create(MERGE_HEAD_TILES, pl.TASK_ID)
    with pl.manual_scope():
        for m_h_idx in pl.parallel(MERGE_HEAD_TILES):
            with pl.spmd(
                t_dim,
                name_hint="merge_norm",
                deps=[_qk_tid, rope_cs_tid],
            ) as merge_tid:
                m_t = pl.tile.get_block_idx()
                m_idx = m_t * MERGE_HEAD_TILES + m_h_idx
                m_h0 = m_h_idx * H_TILE
                m_blk_base = m_idx * SPARSE_BLOCKS * H_TILE
                m_mi = sparse_blk_mi[m_blk_base : m_blk_base + H_TILE, 0:1]
                m_li = sparse_blk_li[m_blk_base : m_blk_base + H_TILE, 0:1]
                m_oi = sparse_blk_oi[m_blk_base : m_blk_base + H_TILE, 0:HEAD_DIM]
                m_all_full = pl.read(qk_wcur, [1])

                # Statically unroll the optional compressed blocks. Keeping
                # them out of a runtime loop lets tile SSA remain explicit and
                # performs no GM access for an absent block.
                for m_sb_offset in pl.unroll(SPARSE_BLOCKS - 1):
                    m_sb = m_sb_offset + 1
                    m_block_valid = m_all_full
                    if m_all_full < 1:
                        m_block_valid = pl.read(valid_block_mask, [m_t, m_sb])
                    if m_block_valid > 0:
                        m_row = m_blk_base + m_sb * H_TILE
                        m_cur_mi = sparse_blk_mi[m_row : m_row + H_TILE, 0:1]
                        m_cur_li = sparse_blk_li[m_row : m_row + H_TILE, 0:1]
                        m_cur_oi = sparse_blk_oi[m_row : m_row + H_TILE, 0:HEAD_DIM]
                        m_mi_new = pl.maximum(m_mi, m_cur_mi)
                        m_alpha = pl.exp(pl.sub(m_mi, m_mi_new))
                        m_beta = pl.exp(pl.sub(m_cur_mi, m_mi_new))
                        m_li = pl.add(pl.mul(m_alpha, m_li), pl.mul(m_beta, m_cur_li))
                        m_oi = pl.add(
                            pl.row_expand_mul(m_oi, m_alpha),
                            pl.row_expand_mul(m_cur_oi, m_beta),
                        )
                        m_mi = m_mi_new

                n_sink_bias = pl.reshape(attn_sink[m_h0 : m_h0 + H_TILE], [H_TILE, 1])
                n_sink_tile = pl.add(pl.sub(m_mi, m_mi), n_sink_bias)
                n_denom = pl.add(m_li, pl.exp(pl.sub(n_sink_tile, m_mi)))
                n_full = pl.row_expand_div(m_oi, n_denom)[0:H_TILE, 0:HEAD_DIM]
                n_bf16 = pl.cast(n_full, target_type=pl.BF16, mode="rint")

                # Inverse RoPE on this head-tile's fp32 rope segment. cos_il /
                # sign*sin are head-invariant for token m_t; rope_swap_idx
                # pairs the interleaved real/imag lanes.
                m_rope = n_full[0:H_TILE, NOPE_DIM:HEAD_DIM]
                m_cos_il = rope_cos_il[m_t : m_t + 1, 0:ROPE_DIM]
                m_sin_signed = rope_sin_signed[m_t : m_t + 1, 0:ROPE_DIM]
                m_swapped = pl.gather(
                    m_rope,
                    dim=-1,
                    index=rope_swap_idx[0:H_TILE, 0:ROPE_DIM],
                )
                m_rot = pl.add(
                    pl.col_expand_mul(m_rope, m_cos_il),
                    pl.col_expand_mul(m_swapped, m_sin_signed),
                )
                n_rope_bf16 = pl.cast(m_rot, target_type=pl.BF16, mode="rint")
                n_full_bf16 = pl.concat(n_bf16[:, :NOPE_DIM], n_rope_bf16)

                for n_hi in pl.unroll(H_TILE):
                    n_pack_row = ((m_h0 + n_hi) // HEADS_PER_GROUP) * t_dim + m_t
                    n_col = ((m_h0 + n_hi) % HEADS_PER_GROUP) * HEAD_DIM
                    o_packed_heads[
                        n_pack_row : n_pack_row + 1,
                        n_col : n_col + HEAD_DIM,
                    ] = n_full_bf16[n_hi : n_hi + 1, :]
            merge_tids[m_h_idx] = merge_tid

    # Array values are orchestration-local and may not escape an IR function;
    # return the four legal scalar TaskIds and reconstruct the local carrier in
    # the projection helper after JIT inlining.
    return (
        o_packed_heads,
        merge_tids[0],
        merge_tids[1],
        merge_tids[2],
        merge_tids[3],
    )


@pl.jit.inline
def sparse_attn_csa_local_o_proj(
    o_packed: pl.Tensor,
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.BF16],
    attn_out: pl.Tensor,
    kv_seq_lens: pl.Tensor,
    heads_dep_0: pl.Scalar[pl.TASK_ID],
    heads_dep_1: pl.Scalar[pl.TASK_ID],
    heads_dep_2: pl.Scalar[pl.TASK_ID],
    heads_dep_3: pl.Scalar[pl.TASK_ID],
):
    """Project local-token, full-group CSA heads into BF16 hidden rows."""
    # ``attn_out`` has a positive static L1 shape.  Folding that extent keeps
    # every output-projection scratch and cube M tile exact for this bucket.
    t_dim = attn_out.shape[0]
    act_t_blks = t_dim // PROJ_B_ACT_TASK_T_TILE
    proj_a_rows = (t_dim + PROJ_A_ROW_TILE - 1) // PROJ_A_ROW_TILE

    # The production Flash-0731 checkpoint keeps wo_a/wo_b in BF16.  Preserve
    # native semantics: proj_a accumulates in FP32 and rounds to BF16, then the
    # BF16 wo_b matmul accumulates each group's contribution in FP32.  The old
    # pypto-lib fixture path dynamically quantized proj_a and consumed INT8
    # wo_b; that is a different model ABI and cannot be used for native-vs-PyPTO
    # precision comparison.
    # Keep every group's token rows contiguous.  A token-major
    # [t_dim, groups*rank] scratch makes PA stores, the BF16 cast, and PB
    # activation loads stride by groups*rank even though each consumer owns a
    # single group.  This internal layout changes no public ABI or byte count.
    o_r_pad = pl.create_tensor([O_GROUPS * t_dim, O_LORA], dtype=pl.FP32)
    o_r_bf16_pad = pl.create_tensor([O_GROUPS * t_dim, O_LORA], dtype=pl.BF16)
    # Per-group FP32 partials preserve the deterministic group accumulation
    # order used by the final AIV stage.  Each proj_b task owns a disjoint
    # [token, group*D + channel] region, so no zero seed or atomic update is
    # required.
    partials = pl.create_tensor([t_dim, O_GROUPS * D], dtype=pl.FP32)
    # ``pack_decode_csa_weights`` preserves wo_b's public [D, G*K] shape but
    # stores it physically as [G, D/N, K/BK, N, BK].  Viewing the flat storage
    # as BK-wide rows makes every cube [N, BK] load contiguous in GM.
    wo_b_tile_rows = pl.reshape(
        wo_b,
        [
            O_GROUPS * (D // PROJ_B_MM_N_TILE) * (O_LORA // B_K_TILE) * PROJ_B_MM_N_TILE,
            B_K_TILE,
        ],
    )
    proj_b_tids = pl.array.create(O_GROUPS, pl.TASK_ID)
    heads_deps = pl.array.create(MERGE_HEAD_TILES, pl.TASK_ID)
    heads_deps[0] = heads_dep_0
    heads_deps[1] = heads_dep_1
    heads_deps[2] = heads_dep_2
    heads_deps[3] = heads_dep_3

    with pl.manual_scope():
        # A3 dispatch is consistently LIFO-like for these sibling tasks while
        # merge tiles themselves become ready in the order 3 -> 2 -> 1 -> 0.
        # Submit merge pairs in the opposite order so the scheduler's newest
        # entries correspond to the late-ready low pairs.  Keep the two groups
        # inside each pair ordered, and keep every mathematical dependency and
        # final group-reduction order unchanged.
        for g_order in pl.parallel(O_GROUPS):
            pair_order = g_order // GROUPS_PER_MERGE_HEAD_TILE
            group_in_pair = g_order - pair_order * GROUPS_PER_MERGE_HEAD_TILE
            g = (MERGE_HEAD_TILES - 1 - pair_order) * GROUPS_PER_MERGE_HEAD_TILE + group_in_pair
            row_base_o = g * t_dim

            with pl.spmd(
                proj_a_rows * PA_N_FRAGS,
                name_hint="proj_a_mm",
                deps=[heads_deps[g // GROUPS_PER_MERGE_HEAD_TILE]],
                allow_early_resolve=True,
            ) as pa_tid:
                pa_unit = pl.tile.get_block_idx()
                pa_rb = pa_unit // PA_N_FRAGS  # row block outermost
                nf = pa_unit - pa_rb * PA_N_FRAGS
                pa_r0 = pa_rb * PROJ_A_ROW_TILE
                pa_rows = pl.min(PROJ_A_ROW_TILE, t_dim - pa_r0)
                pa_src0 = row_base_o + pa_r0
                n0 = nf * PROJ_A_MM_N_TILE
                xa0_chunk = pl.slice(
                    o_packed, [PROJ_A_ROW_TILE, A_K_TILE], [pa_src0, 0], valid_shape=[pa_rows, A_K_TILE]
                )
                wa0_chunk = wo_a[g : g + 1, n0 : n0 + PROJ_A_MM_N_TILE, 0:A_K_TILE]
                acc_a = pl.matmul(xa0_chunk, wa0_chunk, b_trans=True, out_dtype=pl.FP32)
                for kb in pl.pipeline(1, O_GROUP_IN // A_K_TILE, stage=2):
                    k0 = kb * A_K_TILE
                    xa_k_chunk = pl.slice(
                        o_packed, [PROJ_A_ROW_TILE, A_K_TILE], [pa_src0, k0], valid_shape=[pa_rows, A_K_TILE]
                    )
                    wa_k_chunk = wo_a[g : g + 1, n0 : n0 + PROJ_A_MM_N_TILE, k0 : k0 + A_K_TILE]
                    acc_a = pl.matmul_acc(acc_a, xa_k_chunk, wa_k_chunk, b_trans=True)
                # acc_a is 3D (wo_a keeps its group axis), which subscript-write cannot express.
                o_r_pad = pl.assemble(o_r_pad, acc_a, [pa_src0, n0])

            with pl.at(
                level=pl.Level.CORE_GROUP,
                name_hint="proj_a_bf16",
                deps=[pa_tid],
                allow_early_resolve=True,
            ) as cast_tid:
                for qt in pl.pipeline(0, t_dim, PROJ_B_ACT_T_TILE, stage=2):
                    cast_r0 = row_base_o + qt
                    proj_a_fp32 = o_r_pad[cast_r0 : cast_r0 + PROJ_B_ACT_T_TILE, 0:O_LORA]
                    o_r_bf16_pad[cast_r0 : cast_r0 + PROJ_B_ACT_T_TILE, 0:O_LORA] = pl.cast(
                        proj_a_fp32, target_type=pl.BF16, mode="rint"
                    )
            with pl.spmd(
                D // PROJ_B_D_TILE,
                name_hint="proj_b_mm",
                deps=[cast_tid],
                allow_early_resolve=True,
            ) as pb_tid:
                dc = pl.tile.get_block_idx()
                d0 = dc * PROJ_B_D_TILE
                for nf in pl.range(PROJ_B_D_TILE // PROJ_B_MM_N_TILE):
                    n0 = d0 + nf * PROJ_B_MM_N_TILE
                    acc_b = pl.create_tensor([t_dim, PROJ_B_MM_N_TILE], dtype=pl.FP32)
                    for kb in pl.pipeline(0, O_LORA // B_K_TILE, stage=2):
                        local_k0 = kb * B_K_TILE
                        weight_row0 = (
                            (g * (D // PROJ_B_MM_N_TILE) + n0 // PROJ_B_MM_N_TILE) * (O_LORA // B_K_TILE) + kb
                        ) * PROJ_B_MM_N_TILE
                        if kb == 0:
                            b_act = o_r_bf16_pad[row_base_o : row_base_o + t_dim, 0:B_K_TILE]
                            b_weight = wo_b_tile_rows[
                                weight_row0 : weight_row0 + PROJ_B_MM_N_TILE,
                                0:B_K_TILE,
                            ]
                            acc_b = pl.matmul(b_act, b_weight, b_trans=True, out_dtype=pl.FP32)
                        else:
                            b_act = o_r_bf16_pad[
                                row_base_o : row_base_o + t_dim,
                                local_k0 : local_k0 + B_K_TILE,
                            ]
                            b_weight = wo_b_tile_rows[
                                weight_row0 : weight_row0 + PROJ_B_MM_N_TILE,
                                0:B_K_TILE,
                            ]
                            acc_b = pl.matmul_acc(acc_b, b_act, b_weight, b_trans=True)
                    partials = pl.assemble(
                        partials,
                        acc_b,
                        [0, g * D + n0],
                    )
            proj_b_tids[g] = pb_tid

    # Reduce the eight disjoint FP32 group partials in a deterministic order,
    # then round once into the caller's BF16 output.  Explicit deps bridge
    # manual_scope into the consolidated output writer.
    with pl.spmd(
        act_t_blks * PROJ_B_ACT_N_REGS,
        name_hint="proj_b_act",
        deps=[proj_b_tids[i] for i in range(O_GROUPS)],
        allow_early_resolve=True,
    ) as _act_tid:
        act_idx = pl.tile.get_block_idx()
        tblk = act_idx // PROJ_B_ACT_N_REGS  # token block outermost
        nreg = act_idx - tblk * PROJ_B_ACT_N_REGS
        ob_n0 = nreg * PROJ_B_ACT_N_TILE
        t0 = tblk * PROJ_B_ACT_TASK_T_TILE
        for b_tb in pl.range(t0, t0 + PROJ_B_ACT_TASK_T_TILE, PROJ_B_ACT_T_TILE):
            acc = pl.full(
                [PROJ_B_ACT_T_TILE, PROJ_B_ACT_N_TILE],
                dtype=pl.FP32,
                value=0.0,
            )
            for act_g in pl.pipeline(O_GROUPS, stage=2):
                p_col0 = act_g * D + ob_n0
                p_g = partials[
                    b_tb : b_tb + PROJ_B_ACT_T_TILE,
                    p_col0 : p_col0 + PROJ_B_ACT_N_TILE,
                ]
                acc = pl.add(acc, p_g)
            request_index = b_tb // S
            if pl.read(kv_seq_lens, [request_index]) > 0:
                out_bf16 = pl.cast(acc, target_type=pl.BF16, mode="rint")
            else:
                # Padded output is deterministic and cannot retain data from
                # an earlier replay of the same graph buffer.
                out_bf16 = pl.full(
                    [PROJ_B_ACT_T_TILE, PROJ_B_ACT_N_TILE],
                    dtype=pl.BF16,
                    value=0.0,
                )
            attn_out[b_tb : b_tb + PROJ_B_ACT_T_TILE, ob_n0 : ob_n0 + PROJ_B_ACT_N_TILE] = out_bf16

    return attn_out


@pl.jit.inline
def sparse_attn_csa(
    q: pl.Tensor,
    ori_kv: pl.Tensor,
    swa_block_table: pl.Tensor,
    cmp_kv: pl.Tensor,
    cmp_block_table: pl.Tensor,
    idx_topk: pl.Tensor,
    position_ids: pl.Tensor,
    kv_seq_lens: pl.Tensor,
    attn_sink: pl.Tensor[[H], pl.FP32],
    freqs_cos: pl.Tensor,
    freqs_sin: pl.Tensor,
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.BF16],
    o_packed_heads: pl.Tensor,
    attn_out: pl.Tensor,
    writeback_dep: pl.Scalar[pl.TASK_ID],
):
    """Compute CSA sparse attention and the grouped output projection."""
    (
        o_packed_heads,
        heads_dep_0,
        heads_dep_1,
        heads_dep_2,
        heads_dep_3,
    ) = sparse_attn_csa_heads(
        q,
        ori_kv,
        swa_block_table,
        cmp_kv,
        cmp_block_table,
        idx_topk,
        position_ids,
        kv_seq_lens,
        attn_sink,
        freqs_cos,
        freqs_sin,
        o_packed_heads,
        writeback_dep,
    )
    attn_out = sparse_attn_csa_local_o_proj(
        o_packed_heads,
        wo_a,
        wo_b,
        attn_out,
        kv_seq_lens,
        heads_dep_0,
        heads_dep_1,
        heads_dep_2,
        heads_dep_3,
    )
    return attn_out
