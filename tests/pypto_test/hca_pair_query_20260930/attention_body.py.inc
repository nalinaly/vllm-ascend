    # Two adjacent S6 queries belong to the same request. Only compressed KV
    # is shared; each raw window retains its own start, mask and QK/PV.
    pair_rows = 2 * H
    pair_slots = 2
    pair_transfer_rows = NUM_QK_CORES * pair_slots * pair_rows
    scores = pl.create_tensor([pair_transfer_rows, ATTN_K_TILE], dtype=pl.FP32)
    probs = pl.create_tensor([pair_transfer_rows, ATTN_K_TILE], dtype=pl.BF16)
    values = pl.create_tensor([pair_transfer_rows, HEAD_DIM], dtype=pl.FP32)
    accum = pl.create_tensor([NUM_QK_CORES * pair_rows, HEAD_DIM], dtype=pl.FP32)
    ffts = pl.create_tensor([256], dtype=pl.INT64)
    sink_col = pl.reshape(attn_sink, [H, 1])
    with pl.spmd(
        NUM_QK_CORES, name_hint="hca_unified_attention",
        deps=[raw_gather_tid, raw_valid_tid, cmp_gather_tid, rope_cs_tid], allow_early_resolve=True,
    ) as attention_tid:
        worker = pl.tile.get_block_idx()
        pl.system.set_ffts(ffts)
        kv_l1 = pl.create_tile([2 * ATTN_K_TILE, HEAD_DIM], dtype=pl.BF16, target_memory=pl.MemorySpace.Mat)
        for token0 in pl.range(worker * 2, t_dim, NUM_QK_CORES * 2):
            request = token0 // S
            length = pl.max(pl.read(kv_seq_lens, [request]), 0)
            position_last = pl.max(pl.read(position_ids, [token0 + 1]), -1)
            pair_cmp_rows = pl.cast(pl.min(HCA_MAX_COMPRESSED_ROWS, pl.min(
                (position_last + 1) // COMPRESS_RATIO, length // COMPRESS_RATIO,
            )), pl.INDEX)
            cmp_blocks = pl.min(cmp_work_count, (pair_cmp_rows + ATTN_K_TILE - 1) // ATTN_K_TILE)
            work_count = cmp_blocks + 2
            first_length = pl.min(WIN, pl.max(pl.min(pl.read(position_ids, [request * S]) + 1, length), 0))
            pair_query = pl.load(q_flat, [token0 * H, 0], [2 * H, HEAD_DIM], target_memory=pl.MemorySpace.Mat)
            # A two-slot ring leaves L1 capacity for M128 queries. Drain only
            # once per pair; QK for the next block overlaps the preceding PV.
            for tick in pl.range(work_count + 1):
                if tick < work_count:
                    l1_row = (tick % 2) * ATTN_K_TILE
                    row = (worker * 2 + tick % 2) * (2 * H)
                    if tick < cmp_blocks:
                        cmp_row = (request * cmp_work_count + tick) * ATTN_K_TILE
                        kv_l1 = pl.gather_row(kv_l1, cmp_work_kv, [l1_row, 0], [cmp_row, 0],
                                             [ATTN_K_TILE, HEAD_DIM])
                        cmp_key_t = pl.tile.slice(pl.tile.transpose_view(kv_l1), [HEAD_DIM, ATTN_K_TILE], [0, l1_row])
                        cmp_qk = pl.matmul(pair_query, cmp_key_t, out_dtype=pl.FP32)
                        pl.store(cmp_qk, [row, 0], scores)
                    else:
                        raw_lane = tick - cmp_blocks
                        raw_drop = pl.max(first_length + (token0 + raw_lane) % S - WIN, 0)
                        kv_l1 = pl.gather_row(kv_l1, raw_kv, [l1_row, 0],
                                             [request * REQUEST_KV_ROWS + raw_drop, 0], [ATTN_K_TILE, HEAD_DIM])
                        raw_key_t = pl.tile.slice(pl.tile.transpose_view(kv_l1), [HEAD_DIM, ATTN_K_TILE], [0, l1_row])
                        raw_query = pl.tile.slice(pair_query, [H, HEAD_DIM], [raw_lane * H, 0])
                        raw_qk = pl.matmul(raw_query, raw_key_t, out_dtype=pl.FP32)
                        pl.store(raw_qk, [row + raw_lane * H, 0], scores)
                    pl.system.sync_set(0, pipe=pl.PipeType.FIX, ffts_mode=2, core_type=pl.KernelType.AIC)
                if tick >= 1:
                    pv_tick = tick - 1
                    pv_row = (worker * 2 + pv_tick % 2) * (2 * H)
                    pl.system.sync_wait(1, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIC)
                    pv_l1_row = (pv_tick % 2) * ATTN_K_TILE
                    if pv_tick < cmp_blocks:
                        cmp_probability = pl.load(probs, [pv_row, 0], [2 * H, ATTN_K_TILE],
                                                  target_memory=pl.MemorySpace.Mat)
                        cmp_left = pl.tile.move(cmp_probability, target_memory=pl.MemorySpace.Left)
                        cmp_right = pl.tile.extract(kv_l1, pv_l1_row, 0, [ATTN_K_TILE, PV_N_TILE],
                                                    target_memory=pl.MemorySpace.Right)
                        cmp_previous = pl.tile.matmul(cmp_left, cmp_right)
                        for cmp_col in pl.unroll(1, HEAD_DIM // PV_N_TILE):
                            cmp_right_next = pl.tile.extract(kv_l1, pv_l1_row, cmp_col * PV_N_TILE,
                                                            [ATTN_K_TILE, PV_N_TILE], target_memory=pl.MemorySpace.Right)
                            cmp_current = pl.tile.matmul(cmp_left, cmp_right_next)
                            pl.store(cmp_previous, [pv_row, (cmp_col - 1) * PV_N_TILE], values)
                            cmp_previous = cmp_current
                        pl.store(cmp_previous, [pv_row, HEAD_DIM - PV_N_TILE], values)
                    else:
                        pv_raw_row = pv_row + (pv_tick - cmp_blocks) * H
                        raw_probability = pl.load(probs, [pv_raw_row, 0], [H, ATTN_K_TILE],
                                                  target_memory=pl.MemorySpace.Mat)
                        raw_left = pl.tile.move(raw_probability, target_memory=pl.MemorySpace.Left)
                        raw_right = pl.tile.extract(kv_l1, pv_l1_row, 0, [ATTN_K_TILE, PV_N_TILE],
                                                    target_memory=pl.MemorySpace.Right)
                        raw_previous = pl.tile.matmul(raw_left, raw_right)
                        for raw_col in pl.unroll(1, HEAD_DIM // PV_N_TILE):
                            raw_right_next = pl.tile.extract(kv_l1, pv_l1_row, raw_col * PV_N_TILE,
                                                            [ATTN_K_TILE, PV_N_TILE], target_memory=pl.MemorySpace.Right)
                            raw_current = pl.tile.matmul(raw_left, raw_right_next)
                            pl.store(raw_previous, [pv_raw_row, (raw_col - 1) * PV_N_TILE], values)
                            raw_previous = raw_current
                        pl.store(raw_previous, [pv_raw_row, HEAD_DIM - PV_N_TILE], values)
                    pl.system.sync_set(2, pipe=pl.PipeType.FIX, ffts_mode=2, core_type=pl.KernelType.AIC)

        for lane in pl.split_aiv(2, mode=pl.SplitMode.NONE):
            pl.system.set_ffts(ffts)
            reduce_tmp = pl.create_tile([H, ATTN_K_TILE], dtype=pl.FP32)
            for vec_token0 in pl.range(worker * 2, t_dim, NUM_QK_CORES * 2):
                vec_token = vec_token0 + lane
                request = vec_token0 // S
                length = pl.max(pl.read(kv_seq_lens, [request]), 0)
                position_last = pl.max(pl.read(position_ids, [vec_token0 + 1]), -1)
                pair_cmp_rows = pl.cast(pl.min(HCA_MAX_COMPRESSED_ROWS, pl.min(
                    (position_last + 1) // COMPRESS_RATIO, length // COMPRESS_RATIO,
                )), pl.INDEX)
                cmp_blocks = pl.min(cmp_work_count, (pair_cmp_rows + ATTN_K_TILE - 1) // ATTN_K_TILE)
                position = pl.max(pl.read(position_ids, [vec_token]), -1)
                cmp_rows = pl.cast(pl.min(HCA_MAX_COMPRESSED_ROWS, pl.min(
                    (position + 1) // COMPRESS_RATIO, length // COMPRESS_RATIO,
                )), pl.INDEX)
                query_blocks = pl.min(cmp_work_count, (cmp_rows + ATTN_K_TILE - 1) // ATTN_K_TILE)
                seed_m = pl.load(sink_col, [0, 0], [H, 1])
                seed_l = pl.mul(seed_m, 0.0)
                seed_max = pl.tile.full([2, H], dtype=pl.FP32, value=0.0)
                seed_sum = pl.tile.full([2, H], dtype=pl.FP32, value=0.0)
                acc_row = (worker * 2 + lane) * H
                for vec_tick, (m_iter, l_iter, has_iter, max_ring, sum_ring) in pl.range(
                    cmp_blocks + 3,
                    init_values=(seed_m, seed_l, pl.cast(0, pl.INDEX), seed_max, seed_sum),
                ):
                    if vec_tick < cmp_blocks + 2:
                        vec_row = (worker * 2 + vec_tick % 2) * (2 * H) + lane * H
                        pl.system.sync_wait(0, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIV)
                        if vec_tick < query_blocks or vec_tick == cmp_blocks + lane:
                            score = pl.load(scores, [vec_row, 0], [H, ATTN_K_TILE])
                            if vec_tick >= cmp_blocks:
                                raw_mask = pl.load(raw_valid, [vec_token, 0], [1, ATTN_K_TILE])
                                raw_bias = pl.mul(pl.sub(raw_mask, 1.0), -NEG_INF)
                                raw_scaled = pl.col_expand_add(pl.mul(score, SOFTMAX_SCALE), raw_bias)
                                raw_m = pl.row_max(raw_scaled, reduce_tmp)
                                raw_exp = pl.col_expand_mul(pl.exp(pl.row_expand_sub(raw_scaled, raw_m)), raw_mask)
                                maximum, exponent = pl.yield_(raw_m, raw_exp)
                            else:
                                cmp_mask = pl.load(cmp_work_valid, [request * cmp_work_count + vec_tick, 0],
                                                   [1, ATTN_K_TILE])
                                cmp_bias = pl.mul(pl.sub(cmp_mask, 1.0), -NEG_INF)
                                cmp_scaled = pl.col_expand_add(pl.mul(score, SOFTMAX_SCALE), cmp_bias)
                                valid_rows = pl.min(ATTN_K_TILE, cmp_rows - vec_tick * ATTN_K_TILE)
                                cmp_shaped = pl.set_validshape(cmp_scaled, H, valid_rows)
                                cmp_masked = pl.fillpad(cmp_shaped, pad_value=pl.PadValue.min)
                                cmp_m = pl.row_max(cmp_masked, reduce_tmp)
                                cmp_exp = pl.col_expand_mul(pl.exp(pl.row_expand_sub(cmp_masked, cmp_m)), cmp_mask)
                                maximum, exponent = pl.yield_(cmp_m, cmp_exp)
                            total = pl.row_sum(exponent, reduce_tmp)
                            probability = pl.cast(exponent, pl.BF16, mode="rint")
                            pl.store(probability, [vec_row, 0], probs)
                            next_max = pl.tile.assemble(max_ring, pl.reshape(maximum, [1, H]), [vec_tick % 2, 0])
                            next_sum = pl.tile.assemble(sum_ring, pl.reshape(total, [1, H]), [vec_tick % 2, 0])
                            ready_max, ready_sum = pl.yield_(next_max, next_sum)
                        else:
                            zero_probability = pl.tile.full([H, ATTN_K_TILE], dtype=pl.BF16, value=0.0)
                            pl.store(zero_probability, [vec_row, 0], probs)
                            ready_max, ready_sum = pl.yield_(max_ring, sum_ring)
                        pl.system.sync_set(1, pipe=pl.PipeType.MTE3, ffts_mode=2, core_type=pl.KernelType.AIV)
                        updated_max, updated_sum = pl.yield_(ready_max, ready_sum)
                    else:
                        updated_max, updated_sum = pl.yield_(max_ring, sum_ring)
                    if vec_tick >= 1:
                        out_tick = vec_tick - 1
                        out_row = (worker * 2 + out_tick % 2) * (2 * H) + lane * H
                        pl.system.sync_wait(2, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIV)
                        if out_tick < query_blocks or out_tick == cmp_blocks + lane:
                            out_m = pl.reshape(pl.tile.extract(updated_max, out_tick % 2, 0, [1, H],
                                                              target_memory=pl.MemorySpace.Vec), [H, 1])
                            out_l = pl.reshape(pl.tile.extract(updated_sum, out_tick % 2, 0, [1, H],
                                                              target_memory=pl.MemorySpace.Vec), [H, 1])
                            next_m = pl.maximum(m_iter, out_m)
                            alpha = pl.exp(pl.sub(m_iter, next_m))
                            beta = pl.exp(pl.sub(out_m, next_m))
                            next_l = pl.add(pl.mul(alpha, l_iter), pl.mul(beta, out_l))
                            for out_col in pl.range(HEAD_DIM // PV_N_TILE):
                                if has_iter == 0:
                                    zero_previous = pl.tile.full([H, PV_N_TILE], dtype=pl.FP32, value=0.0)
                                    previous_out = pl.yield_(zero_previous)
                                else:
                                    loaded_previous = pl.load(accum, [acc_row, out_col * PV_N_TILE], [H, PV_N_TILE])
                                    previous_out = pl.yield_(loaded_previous)
                                pv_part = pl.load(values, [out_row, out_col * PV_N_TILE], [H, PV_N_TILE])
                                next_part = pl.add(pl.row_expand_mul(previous_out, alpha), pl.row_expand_mul(pv_part, beta))
                                pl.store(next_part, [acc_row, out_col * PV_N_TILE], accum)
                            if out_tick == cmp_blocks + lane:
                                final_sink = pl.load(sink_col, [0, 0], [H, 1])
                                sink_value = pl.add(pl.sub(next_m, next_m), final_sink)
                                denominator = pl.add(next_l, pl.exp(pl.sub(sink_value, next_m)))
                                cosine = pl.load(rope_cos_il, [vec_token, 0], [1, ROPE_DIM])
                                sine = pl.load(rope_sin_signed, [vec_token, 0], [1, ROPE_DIM])
                                swap_ids = pl.cast(pl.tile.arange(0, [1, ROPE_DIM], dtype=pl.INT32), pl.FP32)
                                swap_pair = pl.cast(pl.cast(pl.mul(swap_ids, 0.5), pl.INT32, mode="trunc"), pl.FP32)
                                swap_lane = pl.sub(swap_ids, pl.mul(swap_pair, 2.0))
                                swap = pl.sub(pl.add(swap_ids, 1.0), pl.mul(swap_lane, 2.0))
                                swap_grid = pl.col_expand_add(pl.tile.full([H_TILE, ROPE_DIM], dtype=pl.FP32, value=0.0),
                                                              pl.add(swap, NOPE_DIM))
                                row_ids = pl.cast(pl.tile.arange(0, [1, H_TILE], dtype=pl.INT32), pl.FP32)
                                swap_index = pl.cast(pl.row_expand_add(swap_grid, pl.reshape(pl.mul(row_ids, HEAD_DIM),
                                                                                            [H_TILE, 1])), pl.INT32)
                                gather_tmp = pl.create_tile([H_TILE, ROPE_DIM], dtype=pl.INT32)
                                for head_part in pl.unroll(H // H_TILE):
                                    head = head_part * H_TILE
                                    merged = pl.load(accum, [acc_row + head, 0], [H_TILE, HEAD_DIM])
                                    normalized = pl.row_expand_div(merged, pl.tile.slice(denominator, [H_TILE, 1], [head, 0]))
                                    bf16 = pl.cast(normalized, pl.BF16, mode="rint")
                                    rope = normalized[0:H_TILE, NOPE_DIM:HEAD_DIM]
                                    swapped = pl.tile.gather(normalized, swap_index, gather_tmp)
                                    rotated = pl.add(pl.col_expand_mul(rope, cosine), pl.col_expand_mul(swapped, sine))
                                    rope_bf16 = pl.cast(rotated, pl.BF16, mode="rint")
                                    full = pl.concat(bf16[0:H_TILE, 0:NOPE_DIM], rope_bf16)
                                    groups = pl.reshape(full, [PUBLISH_GROUPS, O_GROUP_IN])
                                    packed_row = (head // HEADS_PER_GROUP) * T_PAD + vec_token
                                    pl.store(groups[0:1, :], [packed_row, 0], o_packed_heads)
                                    pl.store(groups[1:2, :], [packed_row + T_PAD, 0], o_packed_heads)
                            active_m, active_l, active_has = pl.yield_(next_m, next_l, pl.cast(1, pl.INDEX))
                        else:
                            active_m, active_l, active_has = pl.yield_(m_iter, l_iter, has_iter)
                        m_after, l_after, has_after = pl.yield_(active_m, active_l, active_has)
                    else:
                        m_after, l_after, has_after = pl.yield_(m_iter, l_iter, has_iter)
                    m_done, l_done, has_done, max_done, sum_done = pl.yield_(
                        m_after, l_after, has_after, updated_max, updated_sum,
                    )
    return o_packed_heads, attention_tid
