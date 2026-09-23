# pypto.program: _jit_diagnose_sparse_attention
import pypto.language as pl

B_DYN = pl.dynamic("B_DYN")
CMP_BLOCK_NUM_DYN = pl.dynamic("CMP_BLOCK_NUM_DYN")
COMPRESSED_TABLE_COLUMNS_DYN = pl.dynamic("COMPRESSED_TABLE_COLUMNS_DYN")
ORI_BLOCK_NUM_DYN = pl.dynamic("ORI_BLOCK_NUM_DYN")
T_DYN = pl.dynamic("T_DYN")
t_dim_inline216__ssa_v0 = pl.dynamic("t_dim_inline216__ssa_v0")


@pl.program
class _jit_diagnose_sparse_attention:
    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def csa_slots_build_valid_qk_plan(
        cmp_sparse_indices_inline233__ssa_v0: pl.Out[pl.Tensor[[t_dim_inline216__ssa_v0, 512], pl.INT32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)]],
        sparse_bias_inline257__ssa_v0: pl.Out[pl.Tensor[[t_dim_inline216__ssa_v0, 1024], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)]],
        t_dim_inline216__ssa_v0: pl.Scalar[pl.INDEX],
        topk__ssa_v0: pl.Tensor[[T_DYN, 512], pl.INT32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)],
        positions__ssa_v0: pl.Tensor[[T_DYN, 1], pl.INT32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)],
        window_indices__ssa_v0: pl.Tensor[[T_DYN, 128], pl.INT32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 0)],
        valid_block_mask_inline205__ssa_v0: pl.Out[pl.Tensor[[t_dim_inline216__ssa_v0, 16], pl.INT32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 0)]],
    ) -> tuple[pl.Tensor[[t_dim_inline216__ssa_v0, 512], pl.INT32], pl.Tensor[[t_dim_inline216__ssa_v0, 1024], pl.FP32]]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        mem_vec_6: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
        mem_vec_14: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
        mem_vec_16: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
        mem_vec_31: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 4096)
        mem_vec_33: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32)
        plan_worker_inline231__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        for bias_t0_inline329__idx_v0, (cmp_sparse_indices_inline233__iter_v1, sparse_bias_inline257__iter_v1) in pl.range(
            plan_worker_inline231__ssa_v0 * 8, t_dim_inline216__ssa_v0, 128, init_values=(cmp_sparse_indices_inline233__ssa_v0, sparse_bias_inline257__ssa_v0)
        ):
            t__tile: pl.Tile[[8, 512], pl.INT32, pl.MemRef(mem_vec_6, pl.const(36896, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.load(
                topk__ssa_v0, [bias_t0_inline329__idx_v0, 0], [8, 512], [8, 512], target_memory=pl.Mem.Vec
            )
            c_raw_inline246__tile: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_6, pl.const(36896, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.cast(t__tile, target_type=pl.FP32, mode="round")
            t__tile_1: pl.Tile[[8, 1], pl.INT32, pl.MemRef(mem_vec_14, pl.const(0, pl.INT64), 32), pl.Mem.Vec] = pl.tile.load(
                positions__ssa_v0, [bias_t0_inline329__idx_v0, 0], [8, 1], [8, 1], target_memory=pl.Mem.Vec
            )
            c_pos_inline236__rm_a0_tmp_v0: pl.Tile[[1, 8], pl.INT32, pl.MemRef(mem_vec_14, pl.const(0, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(t__tile_1, [1, 8])
            c_pos_inline236__row_major_tmp_v1: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_14, pl.const(0, pl.INT64), 32), pl.Mem.Vec] = pl.tile.cast(
                c_pos_inline236__rm_a0_tmp_v0, target_type=pl.FP32, mode="round"
            )
            c_pos_inline236__tile: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_14, pl.const(0, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(c_pos_inline236__row_major_tmp_v1, [8, 1])
            t__rm_a0_tmp_v2: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_14, pl.const(0, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(c_pos_inline236__tile, [1, 8])
            t__row_major_tmp_v3: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_14, pl.const(0, pl.INT64), 32), pl.Mem.Vec] = pl.tile.adds(t__rm_a0_tmp_v2, 1.0)
            t__tile_2: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_14, pl.const(0, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(t__row_major_tmp_v3, [8, 1])
            c_pos_scaled_inline266__rm_a0_tmp_v4: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_14, pl.const(0, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(t__tile_2, [1, 8])
            c_pos_scaled_inline266__row_major_tmp_v5: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_14, pl.const(0, pl.INT64), 32), pl.Mem.Vec] = pl.tile.muls(c_pos_scaled_inline266__rm_a0_tmp_v4, 0.25)
            c_pos_scaled_inline266__tile: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_14, pl.const(0, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(c_pos_scaled_inline266__row_major_tmp_v5, [8, 1])
            c_pos_i32_inline228__rm_a0_tmp_v6: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_14, pl.const(0, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(c_pos_scaled_inline266__tile, [1, 8])
            c_pos_i32_inline228__row_major_tmp_v7: pl.Tile[[1, 8], pl.INT32, pl.MemRef(mem_vec_14, pl.const(0, pl.INT64), 32), pl.Mem.Vec] = pl.tile.cast(
                c_pos_i32_inline228__rm_a0_tmp_v6, target_type=pl.INT32, mode="trunc"
            )
            c_pos_i32_inline228__tile: pl.Tile[[8, 1], pl.INT32, pl.MemRef(mem_vec_14, pl.const(0, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(c_pos_i32_inline228__row_major_tmp_v7, [8, 1])
            c_pos_q_inline242__rm_a0_tmp_v8: pl.Tile[[1, 8], pl.INT32, pl.MemRef(mem_vec_14, pl.const(0, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(c_pos_i32_inline228__tile, [1, 8])
            c_pos_q_inline242__row_major_tmp_v9: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_16, pl.const(16384, pl.INT64), 32), pl.Mem.Vec] = pl.tile.cast(
                c_pos_q_inline242__rm_a0_tmp_v8, target_type=pl.FP32, mode="round"
            )
            c_pos_q_inline242__tile: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_16, pl.const(16384, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(c_pos_q_inline242__row_major_tmp_v9, [8, 1])
            t__tile_3: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_14, pl.const(0, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.full([8, 512], dtype=pl.FP32, value=1.0)
            c_upper_b_inline224__tile: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_14, pl.const(0, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.row_expand_mul(t__tile_3, c_pos_q_inline242__tile)
            t__tile_4: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_16, pl.const(16384, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.adds(c_raw_inline246__tile, 1.0)
            t__tile_5: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_16, pl.const(16384, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.maximums(t__tile_4, 0.0)
            c_ge_inline256__tile: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_16, pl.const(16384, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.minimums(t__tile_5, 1.0)
            t__tile_6: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_14, pl.const(0, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.sub(c_upper_b_inline224__tile, c_raw_inline246__tile)
            t__tile_7: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_14, pl.const(0, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.maximums(t__tile_6, 0.0)
            c_lt_inline258__tile: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_14, pl.const(0, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.minimums(t__tile_7, 1.0)
            c_mask_inline267__tile: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_14, pl.const(0, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.mul(c_ge_inline256__tile, c_lt_inline258__tile)
            t__tile_8: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_6, pl.const(36896, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.adds(c_raw_inline246__tile, 1.0)
            t__tile_9: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_6, pl.const(36896, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.mul(c_mask_inline267__tile, t__tile_8)
            c_out_inline243__tile: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_6, pl.const(36896, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.subs(t__tile_9, 1.0)
            t__tile_10: pl.Tile[[8, 512], pl.INT32, pl.MemRef(mem_vec_16, pl.const(16384, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.cast(c_out_inline243__tile, target_type=pl.INT32, mode="round")
            cmp_sparse_indices_inline233__tile: pl.Tensor[[t_dim_inline216__ssa_v0, 512], pl.INT32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                t__tile_10, [bias_t0_inline329__idx_v0, 0], cmp_sparse_indices_inline233__iter_v1
            )
            t__tile_11: pl.Tile[[8, 128], pl.INT32, pl.MemRef(mem_vec_16, pl.const(16384, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.load(
                window_indices__ssa_v0, [bias_t0_inline329__idx_v0, 0], [8, 128], [8, 128], target_memory=pl.Mem.Vec
            )
            v_win_f_inline245__tile: pl.Tile[[8, 128], pl.FP32, pl.MemRef(mem_vec_16, pl.const(16384, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(t__tile_11, target_type=pl.FP32, mode="round")
            t__tile_12: pl.Tile[[8, 128], pl.FP32, pl.MemRef(mem_vec_16, pl.const(16384, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.adds(v_win_f_inline245__tile, 1.0)
            t__tile_13: pl.Tile[[8, 128], pl.FP32, pl.MemRef(mem_vec_16, pl.const(16384, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.maximums(t__tile_12, 0.0)
            v_win_valid_inline213__tile: pl.Tile[[8, 128], pl.FP32, pl.MemRef(mem_vec_31, pl.const(32768, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.minimums(t__tile_13, 1.0)
            tmp_tile: pl.Tile[[8, 128], pl.FP32, pl.MemRef(mem_vec_16, pl.const(16384, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.create([8, 128], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            raw_block_valid_inline263__tile: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_33, pl.const(36864, pl.INT64), 32), pl.Mem.Vec] = pl.tile.row_max(v_win_valid_inline213__tile, tmp_tile)
            for c_t0_inline268__idx_v0 in pl.range(8):
                t__tile_14: pl.Scalar[pl.FP32] = pl.tile.read(raw_block_valid_inline263__tile, [c_t0_inline268__idx_v0, 0])
                c_valid_inline269__ssa_v0: pl.Scalar[pl.INT32] = pl.cast(t__tile_14, pl.INT32)
                pl.tensor.write(valid_block_mask_inline205__ssa_v0, [bias_t0_inline329__idx_v0 + c_t0_inline268__idx_v0, 0], c_valid_inline269__ssa_v0)
            c_s0_inline237__ssa_v0: pl.Scalar[pl.INDEX] = 0
            t__tile_15: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_14, pl.const(0, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.slice(c_mask_inline267__tile, [8, 512], [0, c_s0_inline237__ssa_v0])
            tmp_tile_1: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_16, pl.const(16384, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.create([8, 512], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            c_blk_valid_inline273__tile: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_33, pl.const(36864, pl.INT64), 32), pl.Mem.Vec] = pl.tile.row_max(t__tile_15, tmp_tile_1)
            for c_dt_inline214__idx_v0 in pl.range(8):
                t__tile_16: pl.Scalar[pl.FP32] = pl.tile.read(c_blk_valid_inline273__tile, [c_dt_inline214__idx_v0, 0])
                c_valid_inline269__ssa_v1: pl.Scalar[pl.INT32] = pl.cast(t__tile_16, pl.INT32)
                pl.tensor.write(valid_block_mask_inline205__ssa_v0, [bias_t0_inline329__idx_v0 + c_dt_inline214__idx_v0, 1], c_valid_inline269__ssa_v1)
            t__tile_17: pl.Tile[[8, 128], pl.FP32, pl.MemRef(mem_vec_14, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.subs(v_win_valid_inline213__tile, 1.0)
            t__tile_18: pl.Tile[[8, 128], pl.FP32, pl.MemRef(mem_vec_14, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.muls(t__tile_17, 1e20)
            sparse_bias_inline257__tile: pl.Tensor[[t_dim_inline216__ssa_v0, 1024], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                t__tile_18, [bias_t0_inline329__idx_v0, 0], sparse_bias_inline257__iter_v1
            )
            t__tile_19: pl.Tile[[8, 384], pl.FP32, pl.MemRef(mem_vec_14, pl.const(0, pl.INT64), 12288), pl.Mem.Vec] = pl.tile.full([8, 384], dtype=pl.FP32, value=-1e20)
            sparse_bias_inline257__tile_1: pl.Tensor[[t_dim_inline216__ssa_v0, 1024], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                t__tile_19, [bias_t0_inline329__idx_v0, 128], sparse_bias_inline257__tile
            )
            t__tile_20: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_6, pl.const(36896, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.minimums(c_out_inline243__tile, 0.0)
            t__tile_21: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_6, pl.const(36896, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.muls(t__tile_20, 1e20)
            sparse_bias_inline257__tile_2: pl.Tensor[[t_dim_inline216__ssa_v0, 1024], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                t__tile_21, [bias_t0_inline329__idx_v0, 512], sparse_bias_inline257__tile_1
            )
            cmp_sparse_indices_inline233__rv_v2, sparse_bias_inline257__rv_v2 = pl.yield_(cmp_sparse_indices_inline233__tile, sparse_bias_inline257__tile_2)
        return cmp_sparse_indices_inline233__ssa_v0, sparse_bias_inline257__ssa_v0

    @pl.function(type=pl.FunctionType.Spmd)
    def csa_slots_build_valid_qk_plan_spmd(
        self,
        cmp_sparse_indices_inline233__ssa_v0: pl.Out[pl.Tensor[[t_dim_inline216__ssa_v0, 512], pl.INT32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)]],
        sparse_bias_inline257__ssa_v0: pl.Out[pl.Tensor[[t_dim_inline216__ssa_v0, 1024], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)]],
        t_dim_inline216__ssa_v0: pl.Scalar[pl.INDEX],
        topk__ssa_v0: pl.Tensor[[T_DYN, 512], pl.INT32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)],
        positions__ssa_v0: pl.Tensor[[T_DYN, 1], pl.INT32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)],
        window_indices__ssa_v0: pl.Tensor[[T_DYN, 128], pl.INT32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 0)],
        valid_block_mask_inline205__ssa_v0: pl.Out[pl.Tensor[[t_dim_inline216__ssa_v0, 16], pl.INT32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 0)]],
    ) -> tuple[pl.Tensor[[t_dim_inline216__ssa_v0, 512], pl.INT32], pl.Tensor[[t_dim_inline216__ssa_v0, 1024], pl.FP32]]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        ret__tmp_v0: pl.Tuple[pl.Tensor[[t_dim_inline216__ssa_v0, 512], pl.INT32], pl.Tensor[[t_dim_inline216__ssa_v0, 1024], pl.FP32]] = self.csa_slots_build_valid_qk_plan(
            cmp_sparse_indices_inline233__ssa_v0,
            sparse_bias_inline257__ssa_v0,
            t_dim_inline216__ssa_v0,
            topk__ssa_v0,
            positions__ssa_v0,
            window_indices__ssa_v0,
            valid_block_mask_inline205__ssa_v0,
            attrs={"arg_directions": [pl.adir.output_existing, pl.adir.output_existing, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.output_existing]},
        )
        cmp_sparse_indices_inline233__rv_v2: pl.Tensor[[t_dim_inline216__ssa_v0, 512], pl.INT32, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 0)] = ret__tmp_v0[0]
        sparse_bias_inline257__rv_v2: pl.Tensor[[t_dim_inline216__ssa_v0, 1024], pl.FP32, pl.MemRef("mem_ddr_7", pl.const(0, pl.INT64), 0)] = ret__tmp_v0[1]
        return cmp_sparse_indices_inline233__ssa_v0, sparse_bias_inline257__ssa_v0

    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def kv_touch(
        ori_kv_flat_inline239__ssa_v0: pl.InOut[pl.Tensor[[ori_block_num_inline321__ssa_v0 * pl.const(32, pl.INDEX), 512], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)]],
    ) -> pl.Tensor[[ori_block_num_inline321__ssa_v0 * pl.const(32, pl.INDEX), 512], pl.BF16]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        mem_vec_1: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 1024)
        t__tile: pl.Tile[[1, 512], pl.BF16, pl.MemRef(mem_vec_1, pl.const(0, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.load(
            ori_kv_flat_inline239__ssa_v0, [0, 0], [1, 512], [1, 512], target_memory=pl.Mem.Vec
        )
        ori_kv_flat_inline239__tile: pl.Tensor[[ori_block_num_inline321__ssa_v0 * pl.const(32, pl.INDEX), 512], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)] = pl.tile.store(
            t__tile, [0, 0], ori_kv_flat_inline239__ssa_v0
        )
        return ori_kv_flat_inline239__ssa_v0

    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def merge_norm(
        rope_swap_idx_inline162__ssa_v0: pl.Tensor[[16, 64], pl.INT32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 4096)],
        t_dim_inline155__ssa_v0: pl.Scalar[pl.INDEX],
        attn_li_inline163__ssa_v0: pl.Tensor[[t_heads_inline281__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)],
        attn_oi_inline159__ssa_v0: pl.Tensor[[t_heads_inline281__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)],
        rope_cos_il_inline170__ssa_v0: pl.Tensor[[384, 64], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 98304)],
        rope_sin_signed_inline165__ssa_v0: pl.Tensor[[384, 64], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 98304)],
        packed__ssa_v0: pl.Out[pl.Tensor[[3072, 4096], pl.BF16, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 25165824)]],
    ) -> pl.Tensor[[3072, 4096], pl.BF16]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        mem_vec_14: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 4096)
        mem_vec_15: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 4096)
        mem_vec_17: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32768)
        mem_vec_19: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
        mem_vec_21: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        mem_vec_22: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        mem_vec_23: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 4096)
        mem_vec_24: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 4096)
        m_worker_inline171__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        m_swap_inline152__ssa_v0: pl.Tile[[16, 64], pl.INT32, pl.MemRef(mem_vec_17, pl.const(8192, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.load(
            rope_swap_idx_inline162__ssa_v0, [0, 0], [16, 64], [16, 64], target_memory=pl.Mem.Vec
        )
        m_swap_f_inline168__ssa_v0: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_17, pl.const(8192, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(
            m_swap_inline152__ssa_v0, target_type=pl.FP32, mode="round"
        )
        m_swap_source_inline153__ssa_v0: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_17, pl.const(8192, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.adds(m_swap_f_inline168__ssa_v0, 448.0)
        m_row_ids_inline160__ci_tmp_v0: pl.Tile[[1, 192], pl.FP32, pl.MemRef(mem_vec_19, pl.const(40960, pl.INT64), 768), pl.Mem.Vec] = pl.tile.create(
            [1, 192], dtype=pl.FP32, target_memory=pl.Mem.Vec
        )
        m_row_ids_inline160__ssa_v0: pl.Tile[[1, 16], pl.INT32, pl.MemRef(mem_vec_14, pl.const(0, pl.INT64), 64), pl.Mem.Vec] = pl.tile.ci(
            pl.const(0, pl.INT32), [1, 16], tmp=m_row_ids_inline160__ci_tmp_v0, dtype=pl.INT32, descending=False
        )
        m_row_ids_f_inline174__ssa_v0: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_19, pl.const(40960, pl.INT64), 64), pl.Mem.Vec] = pl.tile.cast(
            m_row_ids_inline160__ssa_v0, target_type=pl.FP32, mode="round"
        )
        m_row_offsets_inline177__ssa_v0: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_19, pl.const(40960, pl.INT64), 64), pl.Mem.Vec] = pl.tile.muls(m_row_ids_f_inline174__ssa_v0, 512.0)
        m_row_offsets_col_inline178__ssa_v0: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_19, pl.const(40960, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(m_row_offsets_inline177__ssa_v0, [16, 1])
        m_swap_flat_inline179__ssa_v0: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_17, pl.const(8192, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.row_expand_add(
            m_swap_source_inline153__ssa_v0, m_row_offsets_col_inline178__ssa_v0
        )
        m_swap_idx_inline181__ssa_v0: pl.Tile[[16, 64], pl.INT32, pl.MemRef(mem_vec_14, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(
            m_swap_flat_inline179__ssa_v0, target_type=pl.INT32, mode="round"
        )
        m_gather_tmp_inline182__ssa_v0: pl.Tile[[16, 64], pl.INT32, pl.MemRef(mem_vec_15, pl.const(4096, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.create(
            [16, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec
        )
        for m_idx_inline183__idx_v0 in pl.range(m_worker_inline171__ssa_v0, t_dim_inline155__ssa_v0 * 4, 48):
            m_t_inline184__ssa_v0: pl.Scalar[pl.INDEX] = m_idx_inline183__idx_v0 // 4
            m_h_idx_inline186__ssa_v0: pl.Scalar[pl.INDEX] = m_idx_inline183__idx_v0 - m_t_inline184__ssa_v0 * 4
            m_h0_inline185__ssa_v0: pl.Scalar[pl.INDEX] = m_h_idx_inline186__ssa_v0 * 16
            m_row_inline161__ssa_v0: pl.Scalar[pl.INDEX] = m_idx_inline183__idx_v0 * 16
            m_li_inline187__ssa_v0: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_19, pl.const(40960, pl.INT64), 64), pl.Mem.Vec] = pl.tile.load(
                attn_li_inline163__ssa_v0, [m_row_inline161__ssa_v0, 0], [16, 1], [16, 1], target_memory=pl.Mem.Vec
            )
            m_oi_inline188__ssa_v0: pl.Tile[[16, 512], pl.FP32, pl.MemRef(mem_vec_17, pl.const(8192, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.load(
                attn_oi_inline159__ssa_v0, [m_row_inline161__ssa_v0, 0], [16, 512], [16, 512], target_memory=pl.Mem.Vec
            )
            n_full_inline150__ssa_v0: pl.Tile[[16, 512], pl.FP32, pl.MemRef(mem_vec_17, pl.const(8192, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.row_expand_div(
                m_oi_inline188__ssa_v0, m_li_inline187__ssa_v0
            )
            n_bf16_inline169__ssa_v0: pl.Tile[[16, 512], pl.BF16, pl.MemRef(mem_vec_19, pl.const(40960, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.cast(
                n_full_inline150__ssa_v0, target_type=pl.BF16, mode="rint"
            )
            n_rounded_inline154__ssa_v0: pl.Tile[[16, 512], pl.FP32, pl.MemRef(mem_vec_17, pl.const(8192, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.cast(
                n_bf16_inline169__ssa_v0, target_type=pl.FP32, mode="round"
            )
            m_cos_il_inline149__ssa_v0: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_21, pl.const(57344, pl.INT64), 256), pl.Mem.Vec] = pl.tile.load(
                rope_cos_il_inline170__ssa_v0, [m_t_inline184__ssa_v0, 0], [1, 64], [1, 64], target_memory=pl.Mem.Vec
            )
            m_sin_signed_inline164__ssa_v0: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_22, pl.const(57600, pl.INT64), 256), pl.Mem.Vec] = pl.tile.load(
                rope_sin_signed_inline165__ssa_v0, [m_t_inline184__ssa_v0, 0], [1, 64], [1, 64], target_memory=pl.Mem.Vec
            )
            m_swapped_inline148__ssa_v0: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_23, pl.const(57856, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.gather(
                n_rounded_inline154__ssa_v0, m_swap_idx_inline181__ssa_v0, m_gather_tmp_inline182__ssa_v0
            )
            m_rope_inline151__ssa_v0_textract: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_24, pl.const(61952, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.extract(
                n_rounded_inline154__ssa_v0, 0, 448, [16, 64], target_memory=pl.Mem.Vec
            )
            t__tmp_v48: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_17, pl.const(8192, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.col_expand_mul(
                m_rope_inline151__ssa_v0_textract, m_cos_il_inline149__ssa_v0
            )
            t__tmp_v49: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_23, pl.const(57856, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.col_expand_mul(
                m_swapped_inline148__ssa_v0, m_sin_signed_inline164__ssa_v0
            )
            m_rot_inline147__ssa_v0: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_17, pl.const(8192, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.add(t__tmp_v48, t__tmp_v49)
            n_rope_bf16_inline176__ssa_v0: pl.Tile[[16, 64], pl.BF16, pl.MemRef(mem_vec_23, pl.const(57856, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(
                m_rot_inline147__ssa_v0, target_type=pl.BF16, mode="rint"
            )
            t__tmp_v50: pl.Tile[[16, 448], pl.BF16, pl.MemRef(mem_vec_19, pl.const(40960, pl.INT64), 16256), pl.Mem.Vec] = pl.tile.slice(n_bf16_inline169__ssa_v0, [16, 448], [0, 0])
            n_full_bf16_inline146__ssa_v0: pl.Tile[[16, 512], pl.BF16, pl.MemRef(mem_vec_17, pl.const(8192, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.concat(t__tmp_v50, n_rope_bf16_inline176__ssa_v0)
            n_group_bf16_inline175__ssa_v0: pl.Tile[[2, 4096], pl.BF16, pl.MemRef(mem_vec_17, pl.const(8192, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.reshape(n_full_bf16_inline146__ssa_v0, [2, 4096])
            n_pack_first_inline156__ssa_v0: pl.Tile[[1, 4096], pl.BF16, pl.MemRef(mem_vec_17, pl.const(8192, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.slice(
                n_group_bf16_inline175__ssa_v0, [1, 4096], [0, 0]
            )
            n_pack_second_inline166__ssa_v0: pl.Tile[[1, 4096], pl.BF16, pl.MemRef(mem_vec_17, pl.const(16384, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.slice(
                n_group_bf16_inline175__ssa_v0, [1, 4096], [1, 0]
            )
            n_pack_row_inline158__ssa_v0: pl.Scalar[pl.INDEX] = m_h0_inline185__ssa_v0 // 8 * 384 + m_t_inline184__ssa_v0
            n_pack_row_second_inline145__ssa_v0: pl.Scalar[pl.INDEX] = n_pack_row_inline158__ssa_v0 + 384
            packed__store: pl.Tensor[[3072, 4096], pl.BF16, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 25165824)] = pl.tile.store(
                n_pack_first_inline156__ssa_v0, [n_pack_row_inline158__ssa_v0, 0], packed__ssa_v0
            )
            packed__store_v0: pl.Tensor[[3072, 4096], pl.BF16, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 25165824)] = pl.tile.store(
                n_pack_second_inline166__ssa_v0, [n_pack_row_second_inline145__ssa_v0, 0], packed__ssa_v0
            )
        return packed__ssa_v0

    @pl.function(type=pl.FunctionType.Spmd)
    def merge_norm_spmd(
        self,
        rope_swap_idx_inline162__ssa_v0: pl.Tensor[[16, 64], pl.INT32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 4096)],
        t_dim_inline155__ssa_v0: pl.Scalar[pl.INDEX],
        attn_li_inline163__ssa_v0: pl.Tensor[[t_heads_inline281__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)],
        attn_oi_inline159__ssa_v0: pl.Tensor[[t_heads_inline281__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)],
        rope_cos_il_inline170__ssa_v0: pl.Tensor[[384, 64], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 98304)],
        rope_sin_signed_inline165__ssa_v0: pl.Tensor[[384, 64], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 98304)],
        packed__ssa_v0: pl.Out[pl.Tensor[[3072, 4096], pl.BF16, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 25165824)]],
    ) -> pl.Tensor[[3072, 4096], pl.BF16]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        packed__ssa_v1: pl.Tensor[[3072, 4096], pl.BF16, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 25165824)] = self.merge_norm(
            rope_swap_idx_inline162__ssa_v0,
            t_dim_inline155__ssa_v0,
            attn_li_inline163__ssa_v0,
            attn_oi_inline159__ssa_v0,
            rope_cos_il_inline170__ssa_v0,
            rope_sin_signed_inline165__ssa_v0,
            packed__ssa_v0,
            attrs={"arg_directions": [pl.adir.input, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.output_existing]},
        )
        return packed__ssa_v0

    @pl.function(type=pl.FunctionType.AIC, level=pl.Level.AIC, role=pl.Role.SubWorker)
    def qk_pv_aic(
        ffts_workspace_inline255__ssa_v0: pl.Tensor[[256], pl.INT64, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 2048)],
        t_dim_inline216__ssa_v0: pl.Scalar[pl.INDEX],
        q_flat_inline248__ssa_v0: pl.Tensor[[t_heads_inline281__ssa_v0, 512], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)],
        valid_block_mask_inline205__ssa_v0: pl.Tensor[[t_dim_inline216__ssa_v0, 16], pl.INT32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)],
        kv_transfer_inline201__ssa_v0: pl.InOut[pl.Tensor[[12288, 512], pl.BF16, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 12582912)]],
        score_transfer_inline221__ssa_v0: pl.InOut[pl.Tensor[[1536, 512], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 3145728)]],
        probability_transfer_inline323__ssa_v0: pl.InOut[pl.Tensor[[1536, 512], pl.BF16, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 1572864)]],
        pv_transfer_inline200__ssa_v0: pl.InOut[pl.Tensor[[1536, 512], pl.FP32, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 3145728)]],
        attn_sink_col_inline234__ssa_v0: pl.Tensor[[64, 1], pl.FP32, pl.MemRef("mem_ddr_7", pl.const(0, pl.INT64), 256)],
        positions__ssa_v0: pl.Tensor[[T_DYN, 1], pl.INT32, pl.MemRef("mem_ddr_8", pl.const(0, pl.INT64), 0)],
        window_indices__ssa_v0: pl.Tensor[[T_DYN, 128], pl.INT32, pl.MemRef("mem_ddr_9", pl.const(0, pl.INT64), 0)],
        ori_kv_flat_inline239__ssa_v1: pl.Tensor[[ori_block_num_inline321__ssa_v0 * pl.const(32, pl.INDEX), 512], pl.BF16, pl.MemRef("mem_ddr_10", pl.const(0, pl.INT64), 0)],
        cmp_sparse_indices_inline233__rv_v2: pl.Tensor[[t_dim_inline216__ssa_v0, 512], pl.INT32, pl.MemRef("mem_ddr_11", pl.const(0, pl.INT64), 0)],
        cmp_table__ssa_v0: pl.Tensor[[B_DYN, COMPRESSED_TABLE_COLUMNS_DYN], pl.INT32, pl.MemRef("mem_ddr_12", pl.const(0, pl.INT64), 0)],
        cmp_kv_flat_inline219__ssa_v0: pl.Tensor[[cmp_block_num_inline265__ssa_v0 * pl.const(32, pl.INDEX), 512], pl.BF16, pl.MemRef("mem_ddr_13", pl.const(0, pl.INT64), 0)],
        sparse_bias_inline257__rv_v2: pl.Tensor[[t_dim_inline216__ssa_v0, 1024], pl.FP32, pl.MemRef("mem_ddr_14", pl.const(0, pl.INT64), 0)],
        mi_transfer_inline251__ssa_v0: pl.InOut[pl.Tensor[[1536, 1], pl.FP32, pl.MemRef("mem_ddr_15", pl.const(0, pl.INT64), 6144)]],
        li_transfer_inline271__ssa_v0: pl.InOut[pl.Tensor[[1536, 1], pl.FP32, pl.MemRef("mem_ddr_16", pl.const(0, pl.INT64), 6144)]],
        alpha_transfer_inline202__ssa_v0: pl.InOut[pl.Tensor[[1536, 1], pl.FP32, pl.MemRef("mem_ddr_17", pl.const(0, pl.INT64), 6144)]],
        attn_mi_inline203__ssa_v0: pl.Out[pl.Tensor[[t_heads_inline281__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_18", pl.const(0, pl.INT64), 0)]],
        attn_li_inline284__ssa_v0: pl.Out[pl.Tensor[[t_heads_inline281__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_19", pl.const(0, pl.INT64), 0)]],
        attn_oi_inline204__ssa_v0: pl.Out[pl.Tensor[[t_heads_inline281__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_20", pl.const(0, pl.INT64), 0)]],
    ):
        pl.func_attr({"mx_tensor_views_blocked": True, "split_aiv": True})
        mem_mat_21: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 65536)
        mem_mat_22: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 131072)
        mem_left_24: pl.Ptr = pl.tile.alloc(pl.Mem.Left, 16384)
        mem_right_25: pl.Ptr = pl.tile.alloc(pl.Mem.Right, 32768)
        mem_left_26: pl.Ptr = pl.tile.alloc(pl.Mem.Left, 16384)
        mem_right_27: pl.Ptr = pl.tile.alloc(pl.Mem.Right, 32768)
        mem_acc_29: pl.Ptr = pl.tile.alloc(pl.Mem.Acc, 131072)
        mem_mat_30: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 16384)
        qk_core_inline197__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        qk_kv_base_inline272__ssa_v0: pl.Scalar[pl.INDEX] = qk_core_inline197__ssa_v0 * 512
        qk_head_base_inline240__ssa_v0: pl.Scalar[pl.INDEX] = qk_core_inline197__ssa_v0 * 64
        pl.system.set_ffts(ffts_workspace_inline255__ssa_v0)
        for qk_t_inline326__idx_v0 in pl.range(qk_core_inline197__ssa_v0, t_dim_inline216__ssa_v0, 24):
            qk_q_inline227__ssa_v0: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_mat_21, pl.const(0, pl.INT64), 65536), pl.Mem.Mat] = pl.tile.load(
                q_flat_inline248__ssa_v0, [qk_t_inline326__idx_v0 * 64, 0], [64, 512], [64, 512], target_memory=pl.Mem.Mat
            )
            for qk_sb_inline249__idx_v0 in pl.range(2):
                t__tile: pl.Scalar[pl.INT32] = pl.tensor.read(valid_block_mask_inline205__ssa_v0, [qk_t_inline326__idx_v0, qk_sb_inline249__idx_v0])
                if 0 < pl.cast(t__tile, pl.INDEX):
                    pl.system.sync_wait(0, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIC)
                    for qk_part_inline253__idx_v0 in pl.range(4):
                        qk_col_inline291__ssa_v0: pl.Scalar[pl.INDEX] = qk_part_inline253__idx_v0 * 128
                        qk_kv_inline264__ssa_v0: pl.Tile[[128, 512], pl.BF16, pl.MemRef(mem_mat_22, pl.const(65536, pl.INT64), 131072), pl.Mem.Mat] = pl.tile.load(
                            kv_transfer_inline201__ssa_v0, [qk_kv_base_inline272__ssa_v0 + qk_col_inline291__ssa_v0, 0], [128, 512], [128, 512], target_memory=pl.Mem.Mat
                        )
                        t__tmp_v24: pl.Tile[
                            [512, 128], pl.BF16, pl.MemRef(mem_mat_22, pl.const(65536, pl.INT64), 131072), pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)
                        ] = pl.tile.transpose_view(qk_kv_inline264__ssa_v0)
                        qk_scores_inline244__ssa_v0_l0_init: pl.Tile[[64, 128], pl.FP32, pl.MemRef(mem_acc_29, pl.const(0, pl.INT64), 32768), pl.Mem.Acc] = pl.tile.create(
                            [64, 128], dtype=pl.FP32, target_memory=pl.Mem.Acc
                        )
                        for qk_scores_inline244__ssa_v0_l0_ko, (qk_scores_inline244__ssa_v0_l0_c,) in pl.range(0, 512, 256, init_values=(qk_scores_inline244__ssa_v0_l0_init,)):
                            qk_scores_inline244__ssa_v0_l0_a: pl.Tile[
                                [64, 128], pl.BF16, pl.MemRef(mem_left_24, pl.const(0, pl.INT64), 16384), pl.Mem.Left, pl.TileView(blayout=pl.TileLayout.row_major)
                            ] = pl.tile.extract(qk_q_inline227__ssa_v0, 0, qk_scores_inline244__ssa_v0_l0_ko, [64, 128], target_memory=pl.Mem.Left)
                            qk_scores_inline244__ssa_v0_l0_b: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_25, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                                t__tmp_v24, qk_scores_inline244__ssa_v0_l0_ko, 0, [128, 128], target_memory=pl.Mem.Right
                            )
                            qk_scores_inline244__ssa_v0_l0_a_1: pl.Tile[
                                [64, 128], pl.BF16, pl.MemRef(mem_left_26, pl.const(16384, pl.INT64), 16384), pl.Mem.Left, pl.TileView(blayout=pl.TileLayout.row_major)
                            ] = pl.tile.extract(qk_q_inline227__ssa_v0, 0, qk_scores_inline244__ssa_v0_l0_ko + 128, [64, 128], target_memory=pl.Mem.Left)
                            qk_scores_inline244__ssa_v0_l0_b_1: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_27, pl.const(32768, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                                t__tmp_v24, qk_scores_inline244__ssa_v0_l0_ko + 128, 0, [128, 128], target_memory=pl.Mem.Right
                            )
                            qk_scores_inline244__ssa_v0_l0_c_acc: pl.Tile[[64, 128], pl.FP32, pl.MemRef(mem_acc_29, pl.const(0, pl.INT64), 32768), pl.Mem.Acc] = pl.tile.matmul_acc(
                                qk_scores_inline244__ssa_v0_l0_c, qk_scores_inline244__ssa_v0_l0_a, qk_scores_inline244__ssa_v0_l0_b, qk_scores_inline244__ssa_v0_l0_ko == 0
                            )
                            qk_scores_inline244__ssa_v0_l0_c_acc_1: pl.Tile[[64, 128], pl.FP32, pl.MemRef(mem_acc_29, pl.const(0, pl.INT64), 32768), pl.Mem.Acc] = pl.tile.matmul_acc(
                                qk_scores_inline244__ssa_v0_l0_c_acc, qk_scores_inline244__ssa_v0_l0_a_1, qk_scores_inline244__ssa_v0_l0_b_1, qk_scores_inline244__ssa_v0_l0_ko == -128
                            )
                            qk_scores_inline244__ssa_v0: pl.Tile[[64, 128], pl.FP32, pl.MemRef(mem_acc_29, pl.const(0, pl.INT64), 32768), pl.Mem.Acc] = pl.yield_(
                                qk_scores_inline244__ssa_v0_l0_c_acc_1
                            )
                        score_transfer_inline221__store: pl.Tensor[[1536, 512], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 3145728)] = pl.tile.store(
                            qk_scores_inline244__ssa_v0, [qk_head_base_inline240__ssa_v0, qk_col_inline291__ssa_v0], score_transfer_inline221__ssa_v0
                        )
                    pl.system.sync_set(1, pipe=pl.PipeType.FIX, ffts_mode=2, core_type=pl.KernelType.AIC)
                    pl.system.sync_wait(2, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIC)
                    pv_acc_inline274__ssa_v0: pl.Tile[[64, 512], pl.FP32, pl.MemRef(mem_acc_29, pl.const(0, pl.INT64), 131072), pl.Mem.Acc] = pl.tile.create(
                        [64, 512], dtype=pl.FP32, target_memory=pl.Mem.Acc
                    )
                    for pv_part_inline275__idx_v0, (pv_acc_inline274__iter_v1,) in pl.range(4, init_values=(pv_acc_inline274__ssa_v0,)):
                        pv_col_inline276__ssa_v0: pl.Scalar[pl.INDEX] = pv_part_inline275__idx_v0 * 128
                        pv_probability_inline199__ssa_v0: pl.Tile[[64, 128], pl.BF16, pl.MemRef(mem_mat_30, pl.const(196608, pl.INT64), 16384), pl.Mem.Mat] = pl.tile.load(
                            probability_transfer_inline323__ssa_v0, [qk_head_base_inline240__ssa_v0, pv_col_inline276__ssa_v0], [64, 128], [64, 128], target_memory=pl.Mem.Mat
                        )
                        pv_kv_inline277__ssa_v0: pl.Tile[[128, 512], pl.BF16, pl.MemRef(mem_mat_22, pl.const(65536, pl.INT64), 131072), pl.Mem.Mat] = pl.tile.load(
                            kv_transfer_inline201__ssa_v0, [qk_kv_base_inline272__ssa_v0 + pv_col_inline276__ssa_v0, 0], [128, 512], [128, 512], target_memory=pl.Mem.Mat
                        )
                        for pv_acc_inline274__ssa_v3_l0_ko, (pv_acc_inline274__ssa_v3_l0_c,) in pl.range(0, 128, 64, init_values=(pv_acc_inline274__iter_v1,)):
                            pv_acc_inline274__ssa_v3_l0_a: pl.Tile[
                                [64, 32], pl.BF16, pl.MemRef(mem_left_24, pl.const(0, pl.INT64), 4096), pl.Mem.Left, pl.TileView(blayout=pl.TileLayout.row_major)
                            ] = pl.tile.extract(pv_probability_inline199__ssa_v0, 0, pv_acc_inline274__ssa_v3_l0_ko, [64, 32], target_memory=pl.Mem.Left)
                            pv_acc_inline274__ssa_v3_l0_b: pl.Tile[[32, 512], pl.BF16, pl.MemRef(mem_right_25, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                                pv_kv_inline277__ssa_v0, pv_acc_inline274__ssa_v3_l0_ko, 0, [32, 512], target_memory=pl.Mem.Right
                            )
                            pv_acc_inline274__ssa_v3_l0_a_1: pl.Tile[
                                [64, 32], pl.BF16, pl.MemRef(mem_left_26, pl.const(16384, pl.INT64), 4096), pl.Mem.Left, pl.TileView(blayout=pl.TileLayout.row_major)
                            ] = pl.tile.extract(pv_probability_inline199__ssa_v0, 0, pv_acc_inline274__ssa_v3_l0_ko + 32, [64, 32], target_memory=pl.Mem.Left)
                            pv_acc_inline274__ssa_v3_l0_b_1: pl.Tile[[32, 512], pl.BF16, pl.MemRef(mem_right_27, pl.const(32768, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                                pv_kv_inline277__ssa_v0, pv_acc_inline274__ssa_v3_l0_ko + 32, 0, [32, 512], target_memory=pl.Mem.Right
                            )
                            pv_acc_inline274__ssa_v3_l0_c_acc: pl.Tile[[64, 512], pl.FP32, pl.MemRef(mem_acc_29, pl.const(0, pl.INT64), 131072), pl.Mem.Acc] = pl.tile.matmul_acc(
                                pv_acc_inline274__ssa_v3_l0_c, pv_acc_inline274__ssa_v3_l0_a, pv_acc_inline274__ssa_v3_l0_b, pv_part_inline275__idx_v0 == 0 and pv_acc_inline274__ssa_v3_l0_ko == 0
                            )
                            pv_acc_inline274__ssa_v3_l0_c_acc_1: pl.Tile[[64, 512], pl.FP32, pl.MemRef(mem_acc_29, pl.const(0, pl.INT64), 131072), pl.Mem.Acc] = pl.tile.matmul_acc(
                                pv_acc_inline274__ssa_v3_l0_c_acc,
                                pv_acc_inline274__ssa_v3_l0_a_1,
                                pv_acc_inline274__ssa_v3_l0_b_1,
                                pv_part_inline275__idx_v0 == 0 and pv_acc_inline274__ssa_v3_l0_ko == -32,
                            )
                            pv_acc_inline274__ssa_v3: pl.Tile[[64, 512], pl.FP32, pl.MemRef(mem_acc_29, pl.const(0, pl.INT64), 131072), pl.Mem.Acc] = pl.yield_(pv_acc_inline274__ssa_v3_l0_c_acc_1)
                        pv_acc_inline274__rv_v2: pl.Tile[[64, 512], pl.FP32, pl.MemRef(mem_acc_29, pl.const(0, pl.INT64), 131072), pl.Mem.Acc] = pl.yield_(pv_acc_inline274__ssa_v3)
                    pv_transfer_inline200__store: pl.Tensor[[1536, 512], pl.FP32, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 3145728)] = pl.tile.store(
                        pv_acc_inline274__rv_v2, [qk_head_base_inline240__ssa_v0, 0], pv_transfer_inline200__ssa_v0
                    )
                    pl.system.sync_set(3, pipe=pl.PipeType.FIX, ffts_mode=2, core_type=pl.KernelType.AIC)
            pl.system.set_ffts(ffts_workspace_inline255__ssa_v0)

    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def qk_pv_aiv(
        ffts_workspace_inline255__ssa_v0: pl.Tensor[[256], pl.INT64, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 2048)],
        t_dim_inline216__ssa_v0: pl.Scalar[pl.INDEX],
        q_flat_inline248__ssa_v0: pl.Tensor[[t_heads_inline281__ssa_v0, 512], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)],
        valid_block_mask_inline205__ssa_v0: pl.Tensor[[t_dim_inline216__ssa_v0, 16], pl.INT32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)],
        kv_transfer_inline201__ssa_v0: pl.InOut[pl.Tensor[[12288, 512], pl.BF16, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 12582912)]],
        score_transfer_inline221__ssa_v0: pl.InOut[pl.Tensor[[1536, 512], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 3145728)]],
        probability_transfer_inline323__ssa_v0: pl.InOut[pl.Tensor[[1536, 512], pl.BF16, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 1572864)]],
        pv_transfer_inline200__ssa_v0: pl.InOut[pl.Tensor[[1536, 512], pl.FP32, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 3145728)]],
        attn_sink_col_inline234__ssa_v0: pl.Tensor[[64, 1], pl.FP32, pl.MemRef("mem_ddr_7", pl.const(0, pl.INT64), 256)],
        positions__ssa_v0: pl.Tensor[[T_DYN, 1], pl.INT32, pl.MemRef("mem_ddr_8", pl.const(0, pl.INT64), 0)],
        window_indices__ssa_v0: pl.Tensor[[T_DYN, 128], pl.INT32, pl.MemRef("mem_ddr_9", pl.const(0, pl.INT64), 0)],
        ori_kv_flat_inline239__ssa_v1: pl.Tensor[[ori_block_num_inline321__ssa_v0 * pl.const(32, pl.INDEX), 512], pl.BF16, pl.MemRef("mem_ddr_10", pl.const(0, pl.INT64), 0)],
        cmp_sparse_indices_inline233__rv_v2: pl.Tensor[[t_dim_inline216__ssa_v0, 512], pl.INT32, pl.MemRef("mem_ddr_11", pl.const(0, pl.INT64), 0)],
        cmp_table__ssa_v0: pl.Tensor[[B_DYN, COMPRESSED_TABLE_COLUMNS_DYN], pl.INT32, pl.MemRef("mem_ddr_12", pl.const(0, pl.INT64), 0)],
        cmp_kv_flat_inline219__ssa_v0: pl.Tensor[[cmp_block_num_inline265__ssa_v0 * pl.const(32, pl.INDEX), 512], pl.BF16, pl.MemRef("mem_ddr_13", pl.const(0, pl.INT64), 0)],
        sparse_bias_inline257__rv_v2: pl.Tensor[[t_dim_inline216__ssa_v0, 1024], pl.FP32, pl.MemRef("mem_ddr_14", pl.const(0, pl.INT64), 0)],
        mi_transfer_inline251__ssa_v0: pl.InOut[pl.Tensor[[1536, 1], pl.FP32, pl.MemRef("mem_ddr_15", pl.const(0, pl.INT64), 6144)]],
        li_transfer_inline271__ssa_v0: pl.InOut[pl.Tensor[[1536, 1], pl.FP32, pl.MemRef("mem_ddr_16", pl.const(0, pl.INT64), 6144)]],
        alpha_transfer_inline202__ssa_v0: pl.InOut[pl.Tensor[[1536, 1], pl.FP32, pl.MemRef("mem_ddr_17", pl.const(0, pl.INT64), 6144)]],
        attn_mi_inline203__ssa_v0: pl.Out[pl.Tensor[[t_heads_inline281__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_18", pl.const(0, pl.INT64), 0)]],
        attn_li_inline284__ssa_v0: pl.Out[pl.Tensor[[t_heads_inline281__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_19", pl.const(0, pl.INT64), 0)]],
        attn_oi_inline204__ssa_v0: pl.Out[pl.Tensor[[t_heads_inline281__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_20", pl.const(0, pl.INT64), 0)]],
    ) -> tuple[
        pl.Tensor[[12288, 512], pl.BF16],
        pl.Tensor[[1536, 512], pl.FP32],
        pl.Tensor[[1536, 512], pl.BF16],
        pl.Tensor[[1536, 512], pl.FP32],
        pl.Tensor[[1536, 1], pl.FP32],
        pl.Tensor[[1536, 1], pl.FP32],
        pl.Tensor[[1536, 1], pl.FP32],
        pl.Tensor[[t_heads_inline281__ssa_v0, 1], pl.FP32],
        pl.Tensor[[t_heads_inline281__ssa_v0, 1], pl.FP32],
        pl.Tensor[[t_heads_inline281__ssa_v0, 512], pl.FP32],
    ]:
        pl.func_attr({"mx_tensor_views_blocked": True, "split_aiv": True, "dual_aiv_dispatch": True})
        mem_vec_21: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
        mem_vec_22: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 128)
        mem_vec_23: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 128)
        mem_vec_24: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32768)
        mem_vec_25: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32768)
        mem_vec_26: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 65536)
        mem_vec_51: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32)
        mem_vec_53: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32)
        mem_vec_54: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32)
        mem_vec_59: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 128)
        mem_vec_61: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32768)
        qk_core_inline197__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        qk_kv_base_inline272__ssa_v0: pl.Scalar[pl.INDEX] = qk_core_inline197__ssa_v0 * 512
        qk_head_base_inline240__ssa_v0: pl.Scalar[pl.INDEX] = qk_core_inline197__ssa_v0 * 64
        pl.system.set_ffts(ffts_workspace_inline255__ssa_v0)
        for qk_t_inline326__idx_v0 in pl.range(qk_core_inline197__ssa_v0, t_dim_inline216__ssa_v0, 24):
            qk_b_inline211__ssa_v0: pl.Scalar[pl.INDEX] = qk_t_inline326__idx_v0 // 6
            qk_aiv_inline262__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_subblock_idx()
            pl.system.set_ffts(ffts_workspace_inline255__ssa_v0)
            qk_lane_head_inline278__ssa_v0: pl.Scalar[pl.INDEX] = qk_aiv_inline262__ssa_v0 * 32
            qk_lane_kv_inline280__ssa_v0: pl.Scalar[pl.INDEX] = qk_aiv_inline262__ssa_v0 * 64
            qk_reduce_tmp_inline285__ssa_v0: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_21, pl.const(0, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.create(
                [8, 512], dtype=pl.FP32, target_memory=pl.Mem.Vec
            )
            running_m_inline250__ssa_v0: pl.Tile[[32, 1], pl.FP32, pl.MemRef(mem_vec_22, pl.const(16384, pl.INT64), 128), pl.Mem.Vec] = pl.tile.load(
                attn_sink_col_inline234__ssa_v0, [qk_lane_head_inline278__ssa_v0, 0], [32, 1], [32, 1], target_memory=pl.Mem.Vec
            )
            running_l_inline324__ssa_v0: pl.Tile[[32, 1], pl.FP32, pl.MemRef(mem_vec_23, pl.const(16512, pl.INT64), 128), pl.Mem.Vec, pl.TileView(blayout=pl.TileLayout.row_major)] = pl.tile.full(
                [32, 1], dtype=pl.FP32, value=1.0
            )
            running_left_inline286__ssa_v0: pl.Tile[[32, 256], pl.FP32, pl.MemRef(mem_vec_24, pl.const(16640, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.full([32, 256], dtype=pl.FP32, value=0.0)
            running_right_inline289__ssa_v0: pl.Tile[[32, 256], pl.FP32, pl.MemRef(mem_vec_25, pl.const(49408, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.full([32, 256], dtype=pl.FP32, value=0.0)
            for qk_sb_inline207__idx_v0, (m_iter, l_iter, left_iter, right_iter) in pl.range(
                2, init_values=(running_m_inline250__ssa_v0, running_l_inline324__ssa_v0, running_left_inline286__ssa_v0, running_right_inline289__ssa_v0)
            ):
                t__tile: pl.Scalar[pl.INT32] = pl.tensor.read(valid_block_mask_inline205__ssa_v0, [qk_t_inline326__idx_v0, qk_sb_inline207__idx_v0])
                if 0 < pl.cast(t__tile, pl.INDEX):
                    for qk_part_inline301__idx_v0 in pl.range(4):
                        qk_part_row_inline303__ssa_v0: pl.Scalar[pl.INDEX] = qk_part_inline301__idx_v0 * 128 + qk_lane_kv_inline280__ssa_v0
                        qk_kv_half_inline287__ssa_v0: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_26, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.tile.full(
                            [64, 512], dtype=pl.BF16, value=0.0
                        )
                        if qk_sb_inline207__idx_v0 == 0:
                            t__tile_1: pl.Scalar[pl.INT32] = pl.tensor.read(positions__ssa_v0, [qk_t_inline326__idx_v0, 0])
                            qk_pos_inline279__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(t__tile_1, pl.INDEX)
                            qk_win_len_inline304__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(qk_pos_inline279__ssa_v0 + 1, 128)
                            qk_win_start_inline212__ssa_v0: pl.Scalar[pl.INDEX] = qk_pos_inline279__ssa_v0 - qk_win_len_inline304__ssa_v0 + 1
                            qk_head_inline225__ssa_v0: pl.Scalar[pl.INDEX] = qk_win_start_inline212__ssa_v0 % 32
                            qk_rows_inline215__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(pl.max(qk_win_len_inline304__ssa_v0 - qk_part_row_inline303__ssa_v0, 0), 64)
                            qk_lo_inline254__ssa_v0: pl.Scalar[pl.INDEX] = 0
                            qk_hi_inline307__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(32 - qk_head_inline225__ssa_v0 - qk_part_row_inline303__ssa_v0, qk_rows_inline215__ssa_v0)
                            if qk_lo_inline254__ssa_v0 < qk_hi_inline307__ssa_v0:
                                qk_raw_row_inline309__tile: pl.Scalar[pl.INT32] = pl.tensor.read(
                                    window_indices__ssa_v0, [qk_t_inline326__idx_v0, qk_part_row_inline303__ssa_v0 + qk_lo_inline254__ssa_v0]
                                )
                                if 0 <= pl.cast(qk_raw_row_inline309__tile, pl.INDEX):
                                    qk_kv_half_inline287__ssa_v1: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_26, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.tile.gather_row(
                                        qk_kv_half_inline287__ssa_v0,
                                        ori_kv_flat_inline239__ssa_v1,
                                        [qk_lo_inline254__ssa_v0, 0],
                                        [qk_raw_row_inline309__tile, 0],
                                        [64, 512],
                                        valid_shape=[qk_hi_inline307__ssa_v0 - qk_lo_inline254__ssa_v0, 512],
                                        transpose=False,
                                    )
                                    qk_kv_half_inline287__phi_v2: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_26, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(
                                        qk_kv_half_inline287__ssa_v1
                                    )
                                else:
                                    qk_kv_half_inline287__phi_v2: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_26, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(
                                        qk_kv_half_inline287__ssa_v0
                                    )
                                qk_kv_half_inline287__phi_v3: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_26, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(qk_kv_half_inline287__phi_v2)
                            else:
                                qk_kv_half_inline287__phi_v3: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_26, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(qk_kv_half_inline287__ssa_v0)
                            qk_lo_inline254__ssa_v1: pl.Scalar[pl.INDEX] = pl.max(32 - qk_head_inline225__ssa_v0 - qk_part_row_inline303__ssa_v0, 0)
                            qk_hi_inline307__ssa_v1: pl.Scalar[pl.INDEX] = pl.min(64 - qk_head_inline225__ssa_v0 - qk_part_row_inline303__ssa_v0, qk_rows_inline215__ssa_v0)
                            if qk_lo_inline254__ssa_v1 < qk_hi_inline307__ssa_v1:
                                qk_raw_row_inline309__tile_1: pl.Scalar[pl.INT32] = pl.tensor.read(
                                    window_indices__ssa_v0, [qk_t_inline326__idx_v0, qk_part_row_inline303__ssa_v0 + qk_lo_inline254__ssa_v1]
                                )
                                if 0 <= pl.cast(qk_raw_row_inline309__tile_1, pl.INDEX):
                                    qk_kv_half_inline287__ssa_v4: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_26, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.tile.gather_row(
                                        qk_kv_half_inline287__phi_v3,
                                        ori_kv_flat_inline239__ssa_v1,
                                        [qk_lo_inline254__ssa_v1, 0],
                                        [qk_raw_row_inline309__tile_1, 0],
                                        [64, 512],
                                        valid_shape=[qk_hi_inline307__ssa_v1 - qk_lo_inline254__ssa_v1, 512],
                                        transpose=False,
                                    )
                                    qk_kv_half_inline287__phi_v5: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_26, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(
                                        qk_kv_half_inline287__ssa_v4
                                    )
                                else:
                                    qk_kv_half_inline287__phi_v5: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_26, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(
                                        qk_kv_half_inline287__phi_v3
                                    )
                                qk_kv_half_inline287__phi_v6: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_26, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(qk_kv_half_inline287__phi_v5)
                            else:
                                qk_kv_half_inline287__phi_v6: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_26, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(qk_kv_half_inline287__phi_v3)
                            qk_lo_inline254__ssa_v2: pl.Scalar[pl.INDEX] = pl.max(64 - qk_head_inline225__ssa_v0 - qk_part_row_inline303__ssa_v0, 0)
                            qk_hi_inline307__ssa_v2: pl.Scalar[pl.INDEX] = pl.min(96 - qk_head_inline225__ssa_v0 - qk_part_row_inline303__ssa_v0, qk_rows_inline215__ssa_v0)
                            if qk_lo_inline254__ssa_v2 < qk_hi_inline307__ssa_v2:
                                qk_raw_row_inline309__tile_2: pl.Scalar[pl.INT32] = pl.tensor.read(
                                    window_indices__ssa_v0, [qk_t_inline326__idx_v0, qk_part_row_inline303__ssa_v0 + qk_lo_inline254__ssa_v2]
                                )
                                if 0 <= pl.cast(qk_raw_row_inline309__tile_2, pl.INDEX):
                                    qk_kv_half_inline287__ssa_v7: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_26, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.tile.gather_row(
                                        qk_kv_half_inline287__phi_v6,
                                        ori_kv_flat_inline239__ssa_v1,
                                        [qk_lo_inline254__ssa_v2, 0],
                                        [qk_raw_row_inline309__tile_2, 0],
                                        [64, 512],
                                        valid_shape=[qk_hi_inline307__ssa_v2 - qk_lo_inline254__ssa_v2, 512],
                                        transpose=False,
                                    )
                                    qk_kv_half_inline287__phi_v8: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_26, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(
                                        qk_kv_half_inline287__ssa_v7
                                    )
                                else:
                                    qk_kv_half_inline287__phi_v8: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_26, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(
                                        qk_kv_half_inline287__phi_v6
                                    )
                                qk_kv_half_inline287__phi_v9: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_26, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(qk_kv_half_inline287__phi_v8)
                            else:
                                qk_kv_half_inline287__phi_v9: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_26, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(qk_kv_half_inline287__phi_v6)
                            qk_lo_inline254__ssa_v3: pl.Scalar[pl.INDEX] = pl.max(96 - qk_head_inline225__ssa_v0 - qk_part_row_inline303__ssa_v0, 0)
                            qk_hi_inline307__ssa_v3: pl.Scalar[pl.INDEX] = pl.min(128 - qk_head_inline225__ssa_v0 - qk_part_row_inline303__ssa_v0, qk_rows_inline215__ssa_v0)
                            if qk_lo_inline254__ssa_v3 < qk_hi_inline307__ssa_v3:
                                qk_raw_row_inline309__tile_3: pl.Scalar[pl.INT32] = pl.tensor.read(
                                    window_indices__ssa_v0, [qk_t_inline326__idx_v0, qk_part_row_inline303__ssa_v0 + qk_lo_inline254__ssa_v3]
                                )
                                if 0 <= pl.cast(qk_raw_row_inline309__tile_3, pl.INDEX):
                                    qk_kv_half_inline287__ssa_v10: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_26, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.tile.gather_row(
                                        qk_kv_half_inline287__phi_v9,
                                        ori_kv_flat_inline239__ssa_v1,
                                        [qk_lo_inline254__ssa_v3, 0],
                                        [qk_raw_row_inline309__tile_3, 0],
                                        [64, 512],
                                        valid_shape=[qk_hi_inline307__ssa_v3 - qk_lo_inline254__ssa_v3, 512],
                                        transpose=False,
                                    )
                                    qk_kv_half_inline287__phi_v11: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_26, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(
                                        qk_kv_half_inline287__ssa_v10
                                    )
                                else:
                                    qk_kv_half_inline287__phi_v11: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_26, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(
                                        qk_kv_half_inline287__phi_v9
                                    )
                                qk_kv_half_inline287__phi_v12: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_26, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(
                                    qk_kv_half_inline287__phi_v11
                                )
                            else:
                                qk_kv_half_inline287__phi_v12: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_26, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(
                                    qk_kv_half_inline287__phi_v9
                                )
                            qk_lo_inline254__ssa_v4: pl.Scalar[pl.INDEX] = pl.max(128 - qk_head_inline225__ssa_v0 - qk_part_row_inline303__ssa_v0, 0)
                            qk_hi_inline307__ssa_v4: pl.Scalar[pl.INDEX] = pl.min(160 - qk_head_inline225__ssa_v0 - qk_part_row_inline303__ssa_v0, qk_rows_inline215__ssa_v0)
                            if qk_lo_inline254__ssa_v4 < qk_hi_inline307__ssa_v4:
                                qk_raw_row_inline309__tile_4: pl.Scalar[pl.INT32] = pl.tensor.read(
                                    window_indices__ssa_v0, [qk_t_inline326__idx_v0, qk_part_row_inline303__ssa_v0 + qk_lo_inline254__ssa_v4]
                                )
                                if 0 <= pl.cast(qk_raw_row_inline309__tile_4, pl.INDEX):
                                    qk_kv_half_inline287__ssa_v13: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_26, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.tile.gather_row(
                                        qk_kv_half_inline287__phi_v12,
                                        ori_kv_flat_inline239__ssa_v1,
                                        [qk_lo_inline254__ssa_v4, 0],
                                        [qk_raw_row_inline309__tile_4, 0],
                                        [64, 512],
                                        valid_shape=[qk_hi_inline307__ssa_v4 - qk_lo_inline254__ssa_v4, 512],
                                        transpose=False,
                                    )
                                    qk_kv_half_inline287__phi_v14: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_26, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(
                                        qk_kv_half_inline287__ssa_v13
                                    )
                                else:
                                    qk_kv_half_inline287__phi_v14: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_26, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(
                                        qk_kv_half_inline287__phi_v12
                                    )
                                qk_kv_half_inline287__phi_v15: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_26, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(
                                    qk_kv_half_inline287__phi_v14
                                )
                            else:
                                qk_kv_half_inline287__phi_v15: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_26, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(
                                    qk_kv_half_inline287__phi_v12
                                )
                            qk_kv_half_inline287__phi_v21: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_26, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(qk_kv_half_inline287__phi_v15)
                        else:
                            qk_kv_half_inline287__ssa_v0: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_26, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.tile.full(
                                [64, 512], dtype=pl.BF16, value=0.0
                            )
                            for qk_row_inline308__idx_v0, (qk_kv_half_inline287__iter_v16,) in pl.range(64, init_values=(qk_kv_half_inline287__ssa_v0,)):
                                qk_cmp_k_inline310__ssa_v0: pl.Scalar[pl.INDEX] = (qk_sb_inline207__idx_v0 - 1) * 512 + qk_part_row_inline303__ssa_v0 + qk_row_inline308__idx_v0
                                if qk_cmp_k_inline310__ssa_v0 < 512:
                                    qk_ridx_inline311__tile: pl.Scalar[pl.INT32] = pl.tensor.read(cmp_sparse_indices_inline233__rv_v2, [qk_t_inline326__idx_v0, qk_cmp_k_inline310__ssa_v0])
                                    if 0 <= pl.cast(qk_ridx_inline311__tile, pl.INDEX):
                                        t__tile_2: pl.Scalar[pl.INT32] = pl.tensor.read(cmp_table__ssa_v0, [qk_b_inline211__ssa_v0, pl.cast(qk_ridx_inline311__tile, pl.INDEX) // 32])
                                        qk_page_inline288__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(t__tile_2, pl.INDEX)
                                        qk_src_inline260__ssa_v0: pl.Scalar[pl.INDEX] = qk_page_inline288__ssa_v0 * 32 + pl.cast(qk_ridx_inline311__tile, pl.INDEX) % 32
                                        qk_kv_half_inline287__ssa_v18: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_26, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.tile.gather_row(
                                            qk_kv_half_inline287__iter_v16, cmp_kv_flat_inline219__ssa_v0, [qk_row_inline308__idx_v0, 0], [qk_src_inline260__ssa_v0, 0], [1, 512], transpose=False
                                        )
                                        qk_kv_half_inline287__phi_v19: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_26, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(
                                            qk_kv_half_inline287__ssa_v18
                                        )
                                    else:
                                        qk_kv_half_inline287__phi_v19: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_26, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(
                                            qk_kv_half_inline287__iter_v16
                                        )
                                    qk_kv_half_inline287__phi_v20: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_26, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(
                                        qk_kv_half_inline287__phi_v19
                                    )
                                else:
                                    qk_kv_half_inline287__phi_v20: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_26, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(
                                        qk_kv_half_inline287__iter_v16
                                    )
                                qk_kv_half_inline287__rv_v17: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_26, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(
                                    qk_kv_half_inline287__phi_v20
                                )
                            qk_kv_half_inline287__phi_v21: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_26, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(qk_kv_half_inline287__rv_v17)
                        kv_transfer_inline201__store: pl.Tensor[[12288, 512], pl.BF16, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 12582912)] = pl.tile.store(
                            qk_kv_half_inline287__phi_v21, [qk_kv_base_inline272__ssa_v0 + qk_part_row_inline303__ssa_v0, 0], kv_transfer_inline201__ssa_v0
                        )
                    pl.system.sync_set(0, pipe=pl.PipeType.MTE3, ffts_mode=2, core_type=pl.KernelType.AIV)
                    pl.system.sync_wait(1, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIV)
                    for sm_part_inline314__idx_v0 in pl.range(4):
                        sm_h_inline315__ssa_v0: pl.Scalar[pl.INDEX] = sm_part_inline314__idx_v0 * 8
                        sm_row_inline206__ssa_v0: pl.Scalar[pl.INDEX] = qk_head_base_inline240__ssa_v0 + qk_lane_head_inline278__ssa_v0 + sm_h_inline315__ssa_v0
                        sm_scores_inline232__ssa_v0: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_26, pl.const(82176, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.load(
                            score_transfer_inline221__ssa_v0, [sm_row_inline206__ssa_v0, 0], [8, 512], [8, 512], target_memory=pl.Mem.Vec
                        )
                        sm_bias_inline282__ssa_v0: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_61, pl.const(147936, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.load(
                            sparse_bias_inline257__rv_v2, [qk_t_inline326__idx_v0, qk_sb_inline207__idx_v0 * 512], [1, 512], [1, 512], target_memory=pl.Mem.Vec
                        )
                        t__tmp_v28: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_26, pl.const(82176, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.muls(sm_scores_inline232__ssa_v0, 0.044194173824159223)
                        sm_masked_inline220__ssa_v0: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_26, pl.const(82176, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.col_expand_add(
                            t__tmp_v28, sm_bias_inline282__ssa_v0
                        )
                        sm_block_max_inline313__ssa_v0: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_61, pl.const(147936, pl.INT64), 32), pl.Mem.Vec] = pl.tile.row_max(
                            sm_masked_inline220__ssa_v0, qk_reduce_tmp_inline285__ssa_v0
                        )
                        sm_old_m_inline299__ssa_v0: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_22, pl.const(16384, pl.INT64), 32), pl.Mem.Vec] = pl.tile.slice(
                            m_iter, [8, 1], [sm_h_inline315__ssa_v0, 0]
                        )
                        sm_old_l_inline332__ssa_v0: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_23, pl.const(16512, pl.INT64), 32), pl.Mem.Vec, pl.TileView(blayout=pl.TileLayout.row_major)] = (
                            pl.tile.slice(l_iter, [8, 1], [sm_h_inline315__ssa_v0, 0])
                        )
                        sm_max_inline316__rm_a0_tmp_v0: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_22, pl.const(16384, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(sm_old_m_inline299__ssa_v0, [1, 8])
                        sm_max_inline316__rm_a1_tmp_v1: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_61, pl.const(147936, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(
                            sm_block_max_inline313__ssa_v0, [1, 8]
                        )
                        sm_max_inline316__row_major_tmp_v2: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_61, pl.const(147936, pl.INT64), 32), pl.Mem.Vec] = pl.tile.maximum(
                            sm_max_inline316__rm_a0_tmp_v0, sm_max_inline316__rm_a1_tmp_v1
                        )
                        sm_max_inline316__ssa_v0: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_61, pl.const(147936, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(
                            sm_max_inline316__row_major_tmp_v2, [8, 1]
                        )
                        t__rm_a0_tmp_v3: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_22, pl.const(16384, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(sm_old_m_inline299__ssa_v0, [1, 8])
                        t__rm_a1_tmp_v4: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_61, pl.const(147936, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(sm_max_inline316__ssa_v0, [1, 8])
                        t__row_major_tmp_v5: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_59, pl.const(147808, pl.INT64), 32), pl.Mem.Vec] = pl.tile.sub(t__rm_a0_tmp_v3, t__rm_a1_tmp_v4)
                        t__tmp_v29: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_59, pl.const(147808, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(t__row_major_tmp_v5, [8, 1])
                        sm_alpha_inline294__rm_a0_tmp_v6: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_59, pl.const(147808, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(t__tmp_v29, [1, 8])
                        sm_alpha_inline294__row_major_tmp_v7: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_59, pl.const(147808, pl.INT64), 32), pl.Mem.Vec] = pl.tile.exp(
                            sm_alpha_inline294__rm_a0_tmp_v6
                        )
                        sm_alpha_inline294__ssa_v0: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_59, pl.const(147808, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(
                            sm_alpha_inline294__row_major_tmp_v7, [8, 1]
                        )
                        t__tmp_v30: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_26, pl.const(82176, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.row_expand_sub(
                            sm_masked_inline220__ssa_v0, sm_max_inline316__ssa_v0
                        )
                        sm_exp_inline327__ssa_v0: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_26, pl.const(82176, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.exp(t__tmp_v30)
                        t__rm_a1_tmp_v9: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_51, pl.const(147712, pl.INT64), 32), pl.Mem.Vec, pl.TileView(blayout=pl.TileLayout.row_major)] = pl.tile.move(
                            sm_alpha_inline294__ssa_v0, target_memory=pl.Mem.Vec, blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.none_box
                        )
                        t__tmp_v31: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_51, pl.const(147712, pl.INT64), 32), pl.Mem.Vec, pl.TileView(blayout=pl.TileLayout.row_major)] = pl.tile.mul(
                            sm_old_l_inline332__ssa_v0, t__rm_a1_tmp_v9
                        )
                        t__tmp_v32: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_53, pl.const(147744, pl.INT64), 32), pl.Mem.Vec] = pl.tile.row_sum(
                            sm_exp_inline327__ssa_v0, qk_reduce_tmp_inline285__ssa_v0
                        )
                        sm_sum_inline252__rm_a1_tmp_v11: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_54, pl.const(147776, pl.INT64), 32), pl.Mem.Vec, pl.TileView(blayout=pl.TileLayout.row_major)] = (
                            pl.tile.move(t__tmp_v32, target_memory=pl.Mem.Vec, blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.none_box)
                        )
                        sm_sum_inline252__ssa_v0: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_51, pl.const(147712, pl.INT64), 32), pl.Mem.Vec, pl.TileView(blayout=pl.TileLayout.row_major)] = (
                            pl.tile.add(t__tmp_v31, sm_sum_inline252__rm_a1_tmp_v11)
                        )
                        sm_probability_inline209__ssa_v0: pl.Tile[[8, 512], pl.BF16, pl.MemRef(mem_vec_26, pl.const(82176, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.cast(
                            sm_exp_inline327__ssa_v0, target_type=pl.BF16, mode="rint"
                        )
                        probability_transfer_inline323__store: pl.Tensor[[1536, 512], pl.BF16, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 1572864)] = pl.tile.store(
                            sm_probability_inline209__ssa_v0, [sm_row_inline206__ssa_v0, 0], probability_transfer_inline323__ssa_v0
                        )
                        mi_transfer_inline251__store: pl.Tensor[[1536, 1], pl.FP32, pl.MemRef("mem_ddr_15", pl.const(0, pl.INT64), 6144)] = pl.tile.store(
                            sm_max_inline316__ssa_v0, [sm_row_inline206__ssa_v0, 0], mi_transfer_inline251__ssa_v0
                        )
                        li_transfer_inline271__store: pl.Tensor[[1536, 1], pl.FP32, pl.MemRef("mem_ddr_16", pl.const(0, pl.INT64), 6144)] = pl.tile.store(
                            sm_sum_inline252__ssa_v0, [sm_row_inline206__ssa_v0, 0], li_transfer_inline271__ssa_v0
                        )
                        alpha_transfer_inline202__store: pl.Tensor[[1536, 1], pl.FP32, pl.MemRef("mem_ddr_17", pl.const(0, pl.INT64), 6144)] = pl.tile.store(
                            sm_alpha_inline294__ssa_v0, [sm_row_inline206__ssa_v0, 0], alpha_transfer_inline202__ssa_v0
                        )
                    pl.system.sync_set(2, pipe=pl.PipeType.MTE3, ffts_mode=2, core_type=pl.KernelType.AIV)
                    pl.system.sync_wait(3, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIV)
                    pv_row_inline318__ssa_v0: pl.Scalar[pl.INDEX] = qk_head_base_inline240__ssa_v0 + qk_lane_head_inline278__ssa_v0
                    next_m_inline283__ssa_v0: pl.Tile[[32, 1], pl.FP32, pl.MemRef(mem_vec_22, pl.const(16384, pl.INT64), 128), pl.Mem.Vec] = pl.tile.load(
                        mi_transfer_inline251__ssa_v0, [pv_row_inline318__ssa_v0, 0], [32, 1], [32, 1], target_memory=pl.Mem.Vec
                    )
                    next_l_inline290__ssa_v0: pl.Tile[[32, 1], pl.FP32, pl.MemRef(mem_vec_23, pl.const(16512, pl.INT64), 128), pl.Mem.Vec] = pl.tile.load(
                        li_transfer_inline271__ssa_v0, [pv_row_inline318__ssa_v0, 0], [32, 1], [32, 1], target_memory=pl.Mem.Vec
                    )
                    alpha_inline320__ssa_v0: pl.Tile[[32, 1], pl.FP32, pl.MemRef(mem_vec_59, pl.const(147808, pl.INT64), 128), pl.Mem.Vec] = pl.tile.load(
                        alpha_transfer_inline202__ssa_v0, [pv_row_inline318__ssa_v0, 0], [32, 1], [32, 1], target_memory=pl.Mem.Vec
                    )
                    pv_left_inline322__ssa_v0: pl.Tile[[32, 256], pl.FP32, pl.MemRef(mem_vec_26, pl.const(82176, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.load(
                        pv_transfer_inline200__ssa_v0, [pv_row_inline318__ssa_v0, 0], [32, 256], [32, 256], target_memory=pl.Mem.Vec
                    )
                    t__tmp_v33: pl.Tile[[32, 256], pl.FP32, pl.MemRef(mem_vec_61, pl.const(147936, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.row_expand_mul(left_iter, alpha_inline320__ssa_v0)
                    next_left_inline330__ssa_v0: pl.Tile[[32, 256], pl.FP32, pl.MemRef(mem_vec_24, pl.const(16640, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.add(t__tmp_v33, pv_left_inline322__ssa_v0)
                    pv_right_inline235__ssa_v0: pl.Tile[[32, 256], pl.FP32, pl.MemRef(mem_vec_26, pl.const(82176, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.load(
                        pv_transfer_inline200__ssa_v0, [pv_row_inline318__ssa_v0, 256], [32, 256], [32, 256], target_memory=pl.Mem.Vec
                    )
                    t__tmp_v34: pl.Tile[[32, 256], pl.FP32, pl.MemRef(mem_vec_61, pl.const(147936, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.row_expand_mul(right_iter, alpha_inline320__ssa_v0)
                    next_right_inline325__ssa_v0: pl.Tile[[32, 256], pl.FP32, pl.MemRef(mem_vec_25, pl.const(49408, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.add(t__tmp_v34, pv_right_inline235__ssa_v0)
                    m_after_inline226__rv_v0, l_after_inline297__rv_v0, left_after_inline298__rv_v0, right_after_inline300__rv_v0 = pl.yield_(
                        next_m_inline283__ssa_v0, next_l_inline290__ssa_v0, next_left_inline330__ssa_v0, next_right_inline325__ssa_v0
                    )
                else:
                    m_after_inline226__rv_v0, l_after_inline297__rv_v0, left_after_inline298__rv_v0, right_after_inline300__rv_v0 = pl.yield_(m_iter, l_iter, left_iter, right_iter)
                running_m_inline302, running_l_inline208, running_left_inline293, running_right_inline296 = pl.yield_(
                    m_after_inline226__rv_v0, l_after_inline297__rv_v0, left_after_inline298__rv_v0, right_after_inline300__rv_v0
                )
            qk_output_row_inline328__ssa_v0: pl.Scalar[pl.INDEX] = qk_t_inline326__idx_v0 * 64 + qk_lane_head_inline278__ssa_v0
            attn_mi_inline203__store: pl.Tensor[[t_heads_inline281__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_18", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                running_m_inline302, [qk_output_row_inline328__ssa_v0, 0], attn_mi_inline203__ssa_v0
            )
            attn_li_inline284__store: pl.Tensor[[t_heads_inline281__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_19", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                running_l_inline208, [qk_output_row_inline328__ssa_v0, 0], attn_li_inline284__ssa_v0
            )
            attn_oi_inline204__store: pl.Tensor[[t_heads_inline281__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_20", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                running_left_inline293, [qk_output_row_inline328__ssa_v0, 0], attn_oi_inline204__ssa_v0
            )
            attn_oi_inline204__store_v0: pl.Tensor[[t_heads_inline281__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_20", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                running_right_inline296, [qk_output_row_inline328__ssa_v0, 256], attn_oi_inline204__ssa_v0
            )
        return (
            kv_transfer_inline201__ssa_v0,
            score_transfer_inline221__ssa_v0,
            probability_transfer_inline323__ssa_v0,
            pv_transfer_inline200__ssa_v0,
            mi_transfer_inline251__ssa_v0,
            li_transfer_inline271__ssa_v0,
            alpha_transfer_inline202__ssa_v0,
            attn_mi_inline203__ssa_v0,
            attn_li_inline284__ssa_v0,
            attn_oi_inline204__ssa_v0,
        )

    @pl.function(type=pl.FunctionType.Group, level=pl.Level.CORE_GROUP, role=pl.Role.SubWorker)
    def qk_pv(
        ffts_workspace_inline255__ssa_v0: pl.Tensor[[256], pl.INT64, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 2048)],
        t_dim_inline216__ssa_v0: pl.Scalar[pl.INDEX],
        q_flat_inline248__ssa_v0: pl.Tensor[[t_heads_inline281__ssa_v0, 512], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)],
        valid_block_mask_inline205__ssa_v0: pl.Tensor[[t_dim_inline216__ssa_v0, 16], pl.INT32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)],
        kv_transfer_inline201__ssa_v0: pl.InOut[pl.Tensor[[12288, 512], pl.BF16, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 12582912)]],
        score_transfer_inline221__ssa_v0: pl.InOut[pl.Tensor[[1536, 512], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 3145728)]],
        probability_transfer_inline323__ssa_v0: pl.InOut[pl.Tensor[[1536, 512], pl.BF16, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 1572864)]],
        pv_transfer_inline200__ssa_v0: pl.InOut[pl.Tensor[[1536, 512], pl.FP32, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 3145728)]],
        attn_sink_col_inline234__ssa_v0: pl.Tensor[[64, 1], pl.FP32, pl.MemRef("mem_ddr_7", pl.const(0, pl.INT64), 256)],
        positions__ssa_v0: pl.Tensor[[T_DYN, 1], pl.INT32, pl.MemRef("mem_ddr_8", pl.const(0, pl.INT64), 0)],
        window_indices__ssa_v0: pl.Tensor[[T_DYN, 128], pl.INT32, pl.MemRef("mem_ddr_9", pl.const(0, pl.INT64), 0)],
        ori_kv_flat_inline239__ssa_v1: pl.Tensor[[ori_block_num_inline321__ssa_v0 * pl.const(32, pl.INDEX), 512], pl.BF16, pl.MemRef("mem_ddr_10", pl.const(0, pl.INT64), 0)],
        cmp_sparse_indices_inline233__rv_v2: pl.Tensor[[t_dim_inline216__ssa_v0, 512], pl.INT32, pl.MemRef("mem_ddr_11", pl.const(0, pl.INT64), 0)],
        cmp_table__ssa_v0: pl.Tensor[[B_DYN, COMPRESSED_TABLE_COLUMNS_DYN], pl.INT32, pl.MemRef("mem_ddr_12", pl.const(0, pl.INT64), 0)],
        cmp_kv_flat_inline219__ssa_v0: pl.Tensor[[cmp_block_num_inline265__ssa_v0 * pl.const(32, pl.INDEX), 512], pl.BF16, pl.MemRef("mem_ddr_13", pl.const(0, pl.INT64), 0)],
        sparse_bias_inline257__rv_v2: pl.Tensor[[t_dim_inline216__ssa_v0, 1024], pl.FP32, pl.MemRef("mem_ddr_14", pl.const(0, pl.INT64), 0)],
        mi_transfer_inline251__ssa_v0: pl.InOut[pl.Tensor[[1536, 1], pl.FP32, pl.MemRef("mem_ddr_15", pl.const(0, pl.INT64), 6144)]],
        li_transfer_inline271__ssa_v0: pl.InOut[pl.Tensor[[1536, 1], pl.FP32, pl.MemRef("mem_ddr_16", pl.const(0, pl.INT64), 6144)]],
        alpha_transfer_inline202__ssa_v0: pl.InOut[pl.Tensor[[1536, 1], pl.FP32, pl.MemRef("mem_ddr_17", pl.const(0, pl.INT64), 6144)]],
        attn_mi_inline203__ssa_v0: pl.Out[pl.Tensor[[t_heads_inline281__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_18", pl.const(0, pl.INT64), 0)]],
        attn_li_inline284__ssa_v0: pl.Out[pl.Tensor[[t_heads_inline281__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_19", pl.const(0, pl.INT64), 0)]],
        attn_oi_inline204__ssa_v0: pl.Out[pl.Tensor[[t_heads_inline281__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_20", pl.const(0, pl.INT64), 0)]],
    ) -> tuple[
        pl.Tensor[[12288, 512], pl.BF16],
        pl.Tensor[[1536, 512], pl.FP32],
        pl.Tensor[[1536, 512], pl.BF16],
        pl.Tensor[[1536, 512], pl.FP32],
        pl.Tensor[[1536, 1], pl.FP32],
        pl.Tensor[[1536, 1], pl.FP32],
        pl.Tensor[[1536, 1], pl.FP32],
        pl.Tensor[[t_heads_inline281__ssa_v0, 1], pl.FP32],
        pl.Tensor[[t_heads_inline281__ssa_v0, 1], pl.FP32],
        pl.Tensor[[t_heads_inline281__ssa_v0, 512], pl.FP32],
    ]:
        pl.func_attr({"mx_tensor_views_blocked": True, "split_aiv": True})
        self.qk_pv_aic(
            ffts_workspace_inline255__ssa_v0,
            t_dim_inline216__ssa_v0,
            q_flat_inline248__ssa_v0,
            valid_block_mask_inline205__ssa_v0,
            kv_transfer_inline201__ssa_v0,
            score_transfer_inline221__ssa_v0,
            probability_transfer_inline323__ssa_v0,
            pv_transfer_inline200__ssa_v0,
            attn_sink_col_inline234__ssa_v0,
            positions__ssa_v0,
            window_indices__ssa_v0,
            ori_kv_flat_inline239__ssa_v1,
            cmp_sparse_indices_inline233__rv_v2,
            cmp_table__ssa_v0,
            cmp_kv_flat_inline219__ssa_v0,
            sparse_bias_inline257__rv_v2,
            mi_transfer_inline251__ssa_v0,
            li_transfer_inline271__ssa_v0,
            alpha_transfer_inline202__ssa_v0,
            attn_mi_inline203__ssa_v0,
            attn_li_inline284__ssa_v0,
            attn_oi_inline204__ssa_v0,
            attrs={
                "arg_directions": [
                    pl.adir.input,
                    pl.adir.scalar,
                    pl.adir.input,
                    pl.adir.input,
                    pl.adir.inout,
                    pl.adir.inout,
                    pl.adir.inout,
                    pl.adir.inout,
                    pl.adir.input,
                    pl.adir.input,
                    pl.adir.input,
                    pl.adir.input,
                    pl.adir.input,
                    pl.adir.input,
                    pl.adir.input,
                    pl.adir.input,
                    pl.adir.inout,
                    pl.adir.inout,
                    pl.adir.inout,
                    pl.adir.output_existing,
                    pl.adir.output_existing,
                    pl.adir.output_existing,
                ]
            },
        )
        self.qk_pv_aiv(
            ffts_workspace_inline255__ssa_v0,
            t_dim_inline216__ssa_v0,
            q_flat_inline248__ssa_v0,
            valid_block_mask_inline205__ssa_v0,
            kv_transfer_inline201__ssa_v0,
            score_transfer_inline221__ssa_v0,
            probability_transfer_inline323__ssa_v0,
            pv_transfer_inline200__ssa_v0,
            attn_sink_col_inline234__ssa_v0,
            positions__ssa_v0,
            window_indices__ssa_v0,
            ori_kv_flat_inline239__ssa_v1,
            cmp_sparse_indices_inline233__rv_v2,
            cmp_table__ssa_v0,
            cmp_kv_flat_inline219__ssa_v0,
            sparse_bias_inline257__rv_v2,
            mi_transfer_inline251__ssa_v0,
            li_transfer_inline271__ssa_v0,
            alpha_transfer_inline202__ssa_v0,
            attn_mi_inline203__ssa_v0,
            attn_li_inline284__ssa_v0,
            attn_oi_inline204__ssa_v0,
            attrs={
                "arg_directions": [
                    pl.adir.input,
                    pl.adir.scalar,
                    pl.adir.input,
                    pl.adir.input,
                    pl.adir.inout,
                    pl.adir.inout,
                    pl.adir.inout,
                    pl.adir.inout,
                    pl.adir.input,
                    pl.adir.input,
                    pl.adir.input,
                    pl.adir.input,
                    pl.adir.input,
                    pl.adir.input,
                    pl.adir.input,
                    pl.adir.input,
                    pl.adir.inout,
                    pl.adir.inout,
                    pl.adir.inout,
                    pl.adir.inout,
                    pl.adir.inout,
                    pl.adir.inout,
                ]
            },
        )
        return (
            kv_transfer_inline201__ssa_v0,
            score_transfer_inline221__ssa_v0,
            probability_transfer_inline323__ssa_v0,
            pv_transfer_inline200__ssa_v0,
            mi_transfer_inline251__ssa_v0,
            li_transfer_inline271__ssa_v0,
            alpha_transfer_inline202__ssa_v0,
            attn_mi_inline203__ssa_v0,
            attn_li_inline284__ssa_v0,
            attn_oi_inline204__ssa_v0,
        )

    @pl.function(type=pl.FunctionType.Spmd)
    def qk_pv_spmd(
        self,
        ffts_workspace_inline255__ssa_v0: pl.Tensor[[256], pl.INT64, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 2048)],
        t_dim_inline216__ssa_v0: pl.Scalar[pl.INDEX],
        q_flat_inline248__ssa_v0: pl.Tensor[[t_heads_inline281__ssa_v0, 512], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)],
        valid_block_mask_inline205__ssa_v0: pl.Tensor[[t_dim_inline216__ssa_v0, 16], pl.INT32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)],
        kv_transfer_inline201__ssa_v0: pl.InOut[pl.Tensor[[12288, 512], pl.BF16, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 12582912)]],
        score_transfer_inline221__ssa_v0: pl.InOut[pl.Tensor[[1536, 512], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 3145728)]],
        probability_transfer_inline323__ssa_v0: pl.InOut[pl.Tensor[[1536, 512], pl.BF16, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 1572864)]],
        pv_transfer_inline200__ssa_v0: pl.InOut[pl.Tensor[[1536, 512], pl.FP32, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 3145728)]],
        attn_sink_col_inline234__ssa_v0: pl.Tensor[[64, 1], pl.FP32, pl.MemRef("mem_ddr_7", pl.const(0, pl.INT64), 256)],
        positions__ssa_v0: pl.Tensor[[T_DYN, 1], pl.INT32, pl.MemRef("mem_ddr_8", pl.const(0, pl.INT64), 0)],
        window_indices__ssa_v0: pl.Tensor[[T_DYN, 128], pl.INT32, pl.MemRef("mem_ddr_9", pl.const(0, pl.INT64), 0)],
        ori_kv_flat_inline239__ssa_v1: pl.Tensor[[ori_block_num_inline321__ssa_v0 * pl.const(32, pl.INDEX), 512], pl.BF16, pl.MemRef("mem_ddr_10", pl.const(0, pl.INT64), 0)],
        cmp_sparse_indices_inline233__rv_v2: pl.Tensor[[t_dim_inline216__ssa_v0, 512], pl.INT32, pl.MemRef("mem_ddr_11", pl.const(0, pl.INT64), 0)],
        cmp_table__ssa_v0: pl.Tensor[[B_DYN, COMPRESSED_TABLE_COLUMNS_DYN], pl.INT32, pl.MemRef("mem_ddr_12", pl.const(0, pl.INT64), 0)],
        cmp_kv_flat_inline219__ssa_v0: pl.Tensor[[cmp_block_num_inline265__ssa_v0 * pl.const(32, pl.INDEX), 512], pl.BF16, pl.MemRef("mem_ddr_13", pl.const(0, pl.INT64), 0)],
        sparse_bias_inline257__rv_v2: pl.Tensor[[t_dim_inline216__ssa_v0, 1024], pl.FP32, pl.MemRef("mem_ddr_14", pl.const(0, pl.INT64), 0)],
        mi_transfer_inline251__ssa_v0: pl.InOut[pl.Tensor[[1536, 1], pl.FP32, pl.MemRef("mem_ddr_15", pl.const(0, pl.INT64), 6144)]],
        li_transfer_inline271__ssa_v0: pl.InOut[pl.Tensor[[1536, 1], pl.FP32, pl.MemRef("mem_ddr_16", pl.const(0, pl.INT64), 6144)]],
        alpha_transfer_inline202__ssa_v0: pl.InOut[pl.Tensor[[1536, 1], pl.FP32, pl.MemRef("mem_ddr_17", pl.const(0, pl.INT64), 6144)]],
        attn_mi_inline203__ssa_v0: pl.Out[pl.Tensor[[t_heads_inline281__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_18", pl.const(0, pl.INT64), 0)]],
        attn_li_inline284__ssa_v0: pl.Out[pl.Tensor[[t_heads_inline281__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_19", pl.const(0, pl.INT64), 0)]],
        attn_oi_inline204__ssa_v0: pl.Out[pl.Tensor[[t_heads_inline281__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_20", pl.const(0, pl.INT64), 0)]],
    ) -> tuple[pl.Tensor[[t_heads_inline281__ssa_v0, 1], pl.FP32], pl.Tensor[[t_heads_inline281__ssa_v0, 1], pl.FP32], pl.Tensor[[t_heads_inline281__ssa_v0, 512], pl.FP32]]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        ret__tmp_v0: pl.Tuple[
            pl.Tensor[[12288, 512], pl.BF16],
            pl.Tensor[[1536, 512], pl.FP32],
            pl.Tensor[[1536, 512], pl.BF16],
            pl.Tensor[[1536, 512], pl.FP32],
            pl.Tensor[[1536, 1], pl.FP32],
            pl.Tensor[[1536, 1], pl.FP32],
            pl.Tensor[[1536, 1], pl.FP32],
            pl.Tensor[[t_heads_inline281__ssa_v0, 1], pl.FP32],
            pl.Tensor[[t_heads_inline281__ssa_v0, 1], pl.FP32],
            pl.Tensor[[t_heads_inline281__ssa_v0, 512], pl.FP32],
        ] = self.qk_pv(
            ffts_workspace_inline255__ssa_v0,
            t_dim_inline216__ssa_v0,
            q_flat_inline248__ssa_v0,
            valid_block_mask_inline205__ssa_v0,
            kv_transfer_inline201__ssa_v0,
            score_transfer_inline221__ssa_v0,
            probability_transfer_inline323__ssa_v0,
            pv_transfer_inline200__ssa_v0,
            attn_sink_col_inline234__ssa_v0,
            positions__ssa_v0,
            window_indices__ssa_v0,
            ori_kv_flat_inline239__ssa_v1,
            cmp_sparse_indices_inline233__rv_v2,
            cmp_table__ssa_v0,
            cmp_kv_flat_inline219__ssa_v0,
            sparse_bias_inline257__rv_v2,
            mi_transfer_inline251__ssa_v0,
            li_transfer_inline271__ssa_v0,
            alpha_transfer_inline202__ssa_v0,
            attn_mi_inline203__ssa_v0,
            attn_li_inline284__ssa_v0,
            attn_oi_inline204__ssa_v0,
            attrs={
                "arg_directions": [
                    pl.adir.input,
                    pl.adir.scalar,
                    pl.adir.input,
                    pl.adir.input,
                    pl.adir.inout,
                    pl.adir.inout,
                    pl.adir.inout,
                    pl.adir.inout,
                    pl.adir.input,
                    pl.adir.input,
                    pl.adir.input,
                    pl.adir.input,
                    pl.adir.input,
                    pl.adir.input,
                    pl.adir.input,
                    pl.adir.input,
                    pl.adir.inout,
                    pl.adir.inout,
                    pl.adir.inout,
                    pl.adir.output_existing,
                    pl.adir.output_existing,
                    pl.adir.output_existing,
                ]
            },
        )
        kv_transfer_inline201__ssa_v1: pl.Tensor[[12288, 512], pl.BF16, pl.MemRef("mem_ddr_21", pl.const(0, pl.INT64), 12582912)] = ret__tmp_v0[0]
        score_transfer_inline221__ssa_v1: pl.Tensor[[1536, 512], pl.FP32, pl.MemRef("mem_ddr_22", pl.const(0, pl.INT64), 3145728)] = ret__tmp_v0[1]
        probability_transfer_inline323__ssa_v1: pl.Tensor[[1536, 512], pl.BF16, pl.MemRef("mem_ddr_23", pl.const(0, pl.INT64), 1572864)] = ret__tmp_v0[2]
        pv_transfer_inline200__ssa_v1: pl.Tensor[[1536, 512], pl.FP32, pl.MemRef("mem_ddr_24", pl.const(0, pl.INT64), 3145728)] = ret__tmp_v0[3]
        mi_transfer_inline251__ssa_v1: pl.Tensor[[1536, 1], pl.FP32, pl.MemRef("mem_ddr_25", pl.const(0, pl.INT64), 6144)] = ret__tmp_v0[4]
        li_transfer_inline271__ssa_v1: pl.Tensor[[1536, 1], pl.FP32, pl.MemRef("mem_ddr_26", pl.const(0, pl.INT64), 6144)] = ret__tmp_v0[5]
        alpha_transfer_inline202__ssa_v1: pl.Tensor[[1536, 1], pl.FP32, pl.MemRef("mem_ddr_27", pl.const(0, pl.INT64), 6144)] = ret__tmp_v0[6]
        attn_mi_inline203__ssa_v1: pl.Tensor[[t_heads_inline281__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_28", pl.const(0, pl.INT64), 0)] = ret__tmp_v0[7]
        attn_li_inline284__ssa_v1: pl.Tensor[[t_heads_inline281__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_29", pl.const(0, pl.INT64), 0)] = ret__tmp_v0[8]
        attn_oi_inline204__ssa_v1: pl.Tensor[[t_heads_inline281__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_30", pl.const(0, pl.INT64), 0)] = ret__tmp_v0[9]
        return attn_mi_inline203__ssa_v0, attn_li_inline284__ssa_v0, attn_oi_inline204__ssa_v0

    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def rope_cs(
        rope_swap_idx_inline331__ssa_v0: pl.Out[pl.Tensor[[16, 64], pl.INT32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 4096)]],
        rope_cos_il_inline198__ssa_v0: pl.Out[pl.Tensor[[384, 64], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 98304)]],
        rope_sin_signed_inline223__ssa_v0: pl.Out[pl.Tensor[[384, 64], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 98304)]],
        rope_cs_blocks_inline218__ssa_v0: pl.Scalar[pl.INDEX],
        cos__ssa_v0: pl.Tensor[[T_DYN, 64], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)],
        sin__ssa_v0: pl.Tensor[[T_DYN, 64], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 0)],
    ) -> tuple[pl.Tensor[[16, 64], pl.INT32], pl.Tensor[[384, 64], pl.FP32], pl.Tensor[[384, 64], pl.FP32]]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        mem_vec_5: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 4096)
        mem_vec_10: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 4096)
        mem_vec_27: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 2048)
        mem_vec_33: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 1024)
        mem_vec_34: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 1024)
        mem_vec_36: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        mem_vec_37: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        sw_ones_inline319__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_5, pl.const(8704, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.full([16, 64], dtype=pl.FP32, value=1.0)
        t__ci_tmp_v0: pl.Tile[[1, 192], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 768), pl.Mem.Vec] = pl.tile.create([1, 192], dtype=pl.FP32, target_memory=pl.Mem.Vec)
        t__tile: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_27, pl.const(4096, pl.INT64), 256), pl.Mem.Vec] = pl.tile.ci(
            pl.const(0, pl.INT32), [1, 64], tmp=t__ci_tmp_v0, dtype=pl.INT32, descending=False
        )
        sw_idx_f_inline292__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 256), pl.Mem.Vec] = pl.tile.cast(t__tile, target_type=pl.FP32, mode="round")
        sw_col_inline312__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_5, pl.const(8704, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.col_expand_mul(sw_ones_inline319__tile, sw_idx_f_inline292__tile)
        t__tile_1: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.muls(sw_col_inline312__tile, 0.5)
        sw_dup_i32_inline196__tile: pl.Tile[[16, 64], pl.INT32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(t__tile_1, target_type=pl.INT32, mode="trunc")
        sw_dup_f_inline195__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(
            sw_dup_i32_inline196__tile, target_type=pl.FP32, mode="round"
        )
        t__tile_2: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.muls(sw_dup_f_inline195__tile, 2.0)
        sw_lane_inline194__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.sub(sw_col_inline312__tile, t__tile_2)
        t__tile_3: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_5, pl.const(8704, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.adds(sw_col_inline312__tile, 1.0)
        t__tile_4: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.muls(sw_lane_inline194__tile, 2.0)
        sw_swap_f_inline193__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_5, pl.const(8704, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.sub(t__tile_3, t__tile_4)
        t__tile_5: pl.Tile[[16, 64], pl.INT32, pl.MemRef(mem_vec_5, pl.const(8704, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(sw_swap_f_inline193__tile, target_type=pl.INT32, mode="round")
        rope_swap_idx_inline331__tile: pl.Tensor[[16, 64], pl.INT32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 4096)] = pl.tile.store(t__tile_5, [0, 0], rope_swap_idx_inline331__ssa_v0)
        cs_ones_inline295__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_5, pl.const(8704, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.full([8, 64], dtype=pl.FP32, value=1.0)
        t__ci_tmp_v1: pl.Tile[[1, 192], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 768), pl.Mem.Vec] = pl.tile.create([1, 192], dtype=pl.FP32, target_memory=pl.Mem.Vec)
        t__tile_6: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_27, pl.const(4096, pl.INT64), 256), pl.Mem.Vec] = pl.tile.ci(
            pl.const(0, pl.INT32), [1, 64], tmp=t__ci_tmp_v1, dtype=pl.INT32, descending=False
        )
        cs_idx_f_inline230__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 256), pl.Mem.Vec] = pl.tile.cast(t__tile_6, target_type=pl.FP32, mode="round")
        cs_col_inline261__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_5, pl.const(8704, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.col_expand_mul(cs_ones_inline295__tile, cs_idx_f_inline230__tile)
        t__tile_7: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.muls(cs_col_inline261__tile, 0.5)
        cs_dup_i32_inline229__tile: pl.Tile[[8, 64], pl.INT32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(t__tile_7, target_type=pl.INT32, mode="trunc")
        cs_dup_f_inline192__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(
            cs_dup_i32_inline229__tile, target_type=pl.FP32, mode="round"
        )
        cs_dup_idx_inline191__tile: pl.Tile[[8, 64], pl.INT32, pl.MemRef(mem_vec_27, pl.const(4096, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(
            cs_dup_f_inline192__tile, target_type=pl.INT32, mode="round"
        )
        t__tile_8: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.muls(cs_dup_f_inline192__tile, 2.0)
        cs_lane_inline190__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_5, pl.const(8704, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.sub(cs_col_inline261__tile, t__tile_8)
        t__tile_9: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_5, pl.const(8704, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.muls(cs_lane_inline190__tile, 2.0)
        t__tile_10: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_5, pl.const(8704, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.subs(t__tile_9, 1.0)
        cs_sign_inline217__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_5, pl.const(8704, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.neg(t__tile_10)
        for cs_rb_inline222__idx_v0, (rope_cos_il_inline198__iter_v1, rope_sin_signed_inline223__iter_v1) in pl.range(
            rope_cs_blocks_inline218__ssa_v0, init_values=(rope_cos_il_inline198__ssa_v0, rope_sin_signed_inline223__ssa_v0)
        ):
            cs_t0_inline317__ssa_v0: pl.Scalar[pl.INDEX] = cs_rb_inline222__idx_v0 * 8
            cs_cos_inline189__tile: pl.Tile[[8, 32], pl.FP32, pl.MemRef(mem_vec_33, pl.const(6144, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.load(
                cos__ssa_v0, [cs_t0_inline317__ssa_v0, 0], [8, 32], [8, 32], target_memory=pl.Mem.Vec
            )
            cs_sin_inline306__tile: pl.Tile[[8, 32], pl.FP32, pl.MemRef(mem_vec_34, pl.const(7168, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.load(
                sin__ssa_v0, [cs_t0_inline317__ssa_v0, 0], [8, 32], [8, 32], target_memory=pl.Mem.Vec
            )
            gather_acc_init: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.create([8, 64], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            for gather_lv, (gather_ia,) in pl.range(8, init_values=(gather_acc_init,)):
                gather_inp_row: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_33, pl.const(6144, pl.INT64), 128), pl.Mem.Vec] = pl.tile.slice(cs_cos_inline189__tile, [1, 32], [gather_lv, 0], [1, 32])
                gather_idx_row: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_27, pl.const(4096, pl.INT64), 256), pl.Mem.Vec] = pl.tile.slice(
                    cs_dup_idx_inline191__tile, [1, 64], [gather_lv, 0], [1, 64]
                )
                gather_row_tmp: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_36, pl.const(8192, pl.INT64), 256), pl.Mem.Vec] = pl.tile.create([1, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec)
                gather_row: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_37, pl.const(8448, pl.INT64), 256), pl.Mem.Vec] = pl.tile.gather(gather_inp_row, gather_idx_row, gather_row_tmp)
                gather_asmbl: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.assemble(gather_ia, gather_row, [gather_lv, 0])
                gather_rv: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 2048), pl.Mem.Vec] = pl.yield_(gather_asmbl)
            t__tile_11: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 2048), pl.Mem.Vec] = gather_rv
            rope_cos_il_inline198__tile: pl.Tensor[[384, 64], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 98304)] = pl.tile.store(
                t__tile_11, [cs_t0_inline317__ssa_v0, 0], rope_cos_il_inline198__iter_v1
            )
            gather_acc_init_1: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.create([8, 64], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            for gather_lv_1, (gather_ia_1,) in pl.range(8, init_values=(gather_acc_init_1,)):
                gather_inp_row_1: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_34, pl.const(7168, pl.INT64), 128), pl.Mem.Vec] = pl.tile.slice(
                    cs_sin_inline306__tile, [1, 32], [gather_lv_1, 0], [1, 32]
                )
                gather_idx_row_1: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_27, pl.const(4096, pl.INT64), 256), pl.Mem.Vec] = pl.tile.slice(
                    cs_dup_idx_inline191__tile, [1, 64], [gather_lv_1, 0], [1, 64]
                )
                gather_row_tmp_1: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_33, pl.const(6144, pl.INT64), 256), pl.Mem.Vec] = pl.tile.create([1, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec)
                gather_row_1: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_36, pl.const(8192, pl.INT64), 256), pl.Mem.Vec] = pl.tile.gather(gather_inp_row_1, gather_idx_row_1, gather_row_tmp_1)
                gather_asmbl_1: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.assemble(gather_ia_1, gather_row_1, [gather_lv_1, 0])
                gather_rv_1: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 2048), pl.Mem.Vec] = pl.yield_(gather_asmbl_1)
            cs_sin_il_inline241__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 2048), pl.Mem.Vec] = gather_rv_1
            t__tile_12: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.mul(cs_sin_il_inline241__tile, cs_sign_inline217__tile)
            rope_sin_signed_inline223__tile: pl.Tensor[[384, 64], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 98304)] = pl.tile.store(
                t__tile_12, [cs_t0_inline317__ssa_v0, 0], rope_sin_signed_inline223__iter_v1
            )
            rope_cos_il_inline198__rv_v2, rope_sin_signed_inline223__rv_v2 = pl.yield_(rope_cos_il_inline198__tile, rope_sin_signed_inline223__tile)
        return rope_swap_idx_inline331__ssa_v0, rope_cos_il_inline198__ssa_v0, rope_sin_signed_inline223__ssa_v0

    @pl.function(type=pl.FunctionType.Orchestration, level=pl.Level.CHIP, role=pl.Role.Orchestrator, auto_scope=False)
    def diagnose_sparse_attention(
        self,
        query__ssa_v0: pl.Tensor[[T_DYN, 64, 512], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)],
        ori_kv__ssa_v0: pl.InOut[pl.Tensor[[ORI_BLOCK_NUM_DYN, 32, 1, 512], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)]],
        window_indices__ssa_v0: pl.Tensor[[T_DYN, 128], pl.INT32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)],
        cmp_kv__ssa_v0: pl.Tensor[[CMP_BLOCK_NUM_DYN, 32, 1, 512], pl.BF16, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)],
        cmp_table__ssa_v0: pl.Tensor[[B_DYN, COMPRESSED_TABLE_COLUMNS_DYN], pl.INT32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 0)],
        topk__ssa_v0: pl.Tensor[[T_DYN, 512], pl.INT32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 0)],
        positions__ssa_v0: pl.Tensor[[T_DYN, 1], pl.INT32, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 0)],
        sink__ssa_v0: pl.Tensor[[64], pl.FP32, pl.MemRef("mem_ddr_7", pl.const(0, pl.INT64), 256)],
        cos__ssa_v0: pl.Tensor[[T_DYN, 64], pl.FP32, pl.MemRef("mem_ddr_8", pl.const(0, pl.INT64), 0)],
        sin__ssa_v0: pl.Tensor[[T_DYN, 64], pl.FP32, pl.MemRef("mem_ddr_9", pl.const(0, pl.INT64), 0)],
        packed__ssa_v0: pl.Out[pl.Tensor[[3072, 4096], pl.BF16, pl.MemRef("mem_ddr_10", pl.const(0, pl.INT64), 25165824)]],
    ) -> pl.Tensor[[3072, 4096], pl.BF16]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        ori_block_num_inline321__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(ori_kv__ssa_v0, 0)
        t_dim_inline216__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(query__ssa_v0, 0)
        t_heads_inline281__ssa_v0: pl.Scalar[pl.INDEX] = t_dim_inline216__ssa_v0 * 64
        rope_cs_blocks_inline218__ssa_v0: pl.Scalar[pl.INDEX] = t_dim_inline216__ssa_v0 // 8
        ori_kv_flat_inline239__ssa_v0: pl.Tensor[[ori_block_num_inline321__ssa_v0 * pl.const(32, pl.INDEX), 512], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)] = pl.tensor.reshape(
            ori_kv__ssa_v0, [ori_block_num_inline321__ssa_v0 * 32, 512]
        )
        ret__tmp_v0: pl.Tuple[pl.Tensor[[ori_block_num_inline321__ssa_v0 * pl.const(32, pl.INDEX), 512], pl.BF16], pl.Scalar[pl.TASK_ID]] = pl.submit(
            self.kv_touch, ori_kv_flat_inline239__ssa_v0, allow_early_resolve=True, attrs={"arg_directions": [pl.adir.inout]}
        )
        ori_kv_flat_inline239__ssa_v1: pl.Tensor[[ori_block_num_inline321__ssa_v0 * pl.const(32, pl.INDEX), 512], pl.BF16, pl.MemRef("mem_ddr_11", pl.const(0, pl.INT64), 0)] = ret__tmp_v0[0]
        tid__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0[1]
        sparse_bias_inline257__ssa_v0: pl.Tensor[[t_dim_inline216__ssa_v0, 1024], pl.FP32, pl.MemRef("mem_ddr_12", pl.const(0, pl.INT64), 0)] = pl.tensor.create(
            [t_dim_inline216__ssa_v0, 1024], dtype=pl.FP32, layout=pl.TensorLayout.ND
        )
        cmp_sparse_indices_inline233__ssa_v0: pl.Tensor[[t_dim_inline216__ssa_v0, 512], pl.INT32, pl.MemRef("mem_ddr_13", pl.const(0, pl.INT64), 0)] = pl.tensor.create(
            [t_dim_inline216__ssa_v0, 512], dtype=pl.INT32, layout=pl.TensorLayout.ND
        )
        valid_block_mask_inline205__ssa_v0: pl.Tensor[[t_dim_inline216__ssa_v0, 16], pl.INT32, pl.MemRef("mem_ddr_14", pl.const(0, pl.INT64), 0)] = pl.tensor.create(
            [t_dim_inline216__ssa_v0, 16], dtype=pl.INT32, layout=pl.TensorLayout.ND
        )
        ret__tmp_v0_1: pl.Tuple[pl.Tensor[[t_dim_inline216__ssa_v0, 512], pl.INT32], pl.Tensor[[t_dim_inline216__ssa_v0, 1024], pl.FP32], pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(
            self.csa_slots_build_valid_qk_plan_spmd,
            cmp_sparse_indices_inline233__ssa_v0,
            sparse_bias_inline257__ssa_v0,
            t_dim_inline216__ssa_v0,
            topk__ssa_v0,
            positions__ssa_v0,
            window_indices__ssa_v0,
            valid_block_mask_inline205__ssa_v0,
            core_num=16,
            allow_early_resolve=True,
            attrs={"arg_directions": [pl.adir.output_existing, pl.adir.output_existing, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.output_existing]},
        )
        cmp_sparse_indices_inline233__rv_v2: pl.Tensor[[t_dim_inline216__ssa_v0, 512], pl.INT32, pl.MemRef("mem_ddr_15", pl.const(0, pl.INT64), 0)] = ret__tmp_v0_1[0]
        sparse_bias_inline257__rv_v2: pl.Tensor[[t_dim_inline216__ssa_v0, 1024], pl.FP32, pl.MemRef("mem_ddr_16", pl.const(0, pl.INT64), 0)] = ret__tmp_v0_1[1]
        qk_plan_tid_inline247__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_1[2]
        cmp_block_num_inline265__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(cmp_kv__ssa_v0, 0)
        cmp_kv_flat_inline219__ssa_v0: pl.Tensor[[cmp_block_num_inline265__ssa_v0 * pl.const(32, pl.INDEX), 512], pl.BF16, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)] = pl.tensor.reshape(
            cmp_kv__ssa_v0, [cmp_block_num_inline265__ssa_v0 * 32, 512]
        )
        q_flat_inline248__ssa_v0: pl.Tensor[[t_heads_inline281__ssa_v0, 512], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)] = pl.tensor.reshape(
            query__ssa_v0, [t_heads_inline281__ssa_v0, 512]
        )
        attn_sink_col_inline234__ssa_v0: pl.Tensor[[64, 1], pl.FP32, pl.MemRef("mem_ddr_7", pl.const(0, pl.INT64), 256)] = pl.tensor.reshape(sink__ssa_v0, [64, 1])
        attn_mi_inline203__ssa_v0: pl.Tensor[[t_heads_inline281__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_17", pl.const(0, pl.INT64), 0)] = pl.tensor.create(
            [t_heads_inline281__ssa_v0, 1], dtype=pl.FP32, layout=pl.TensorLayout.ND
        )
        attn_li_inline284__ssa_v0: pl.Tensor[[t_heads_inline281__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_18", pl.const(0, pl.INT64), 0)] = pl.tensor.create(
            [t_heads_inline281__ssa_v0, 1], dtype=pl.FP32, layout=pl.TensorLayout.ND
        )
        attn_oi_inline204__ssa_v0: pl.Tensor[[t_heads_inline281__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_19", pl.const(0, pl.INT64), 0)] = pl.tensor.create(
            [t_heads_inline281__ssa_v0, 512], dtype=pl.FP32, layout=pl.TensorLayout.ND
        )
        kv_transfer_inline201__ssa_v0: pl.Tensor[[12288, 512], pl.BF16, pl.MemRef("mem_ddr_20", pl.const(0, pl.INT64), 12582912)] = pl.tensor.create(
            [12288, 512], dtype=pl.BF16, layout=pl.TensorLayout.ND
        )
        score_transfer_inline221__ssa_v0: pl.Tensor[[1536, 512], pl.FP32, pl.MemRef("mem_ddr_21", pl.const(0, pl.INT64), 3145728)] = pl.tensor.create(
            [1536, 512], dtype=pl.FP32, layout=pl.TensorLayout.ND
        )
        probability_transfer_inline323__ssa_v0: pl.Tensor[[1536, 512], pl.BF16, pl.MemRef("mem_ddr_22", pl.const(0, pl.INT64), 1572864)] = pl.tensor.create(
            [1536, 512], dtype=pl.BF16, layout=pl.TensorLayout.ND
        )
        pv_transfer_inline200__ssa_v0: pl.Tensor[[1536, 512], pl.FP32, pl.MemRef("mem_ddr_23", pl.const(0, pl.INT64), 3145728)] = pl.tensor.create(
            [1536, 512], dtype=pl.FP32, layout=pl.TensorLayout.ND
        )
        mi_transfer_inline251__ssa_v0: pl.Tensor[[1536, 1], pl.FP32, pl.MemRef("mem_ddr_24", pl.const(0, pl.INT64), 6144)] = pl.tensor.create([1536, 1], dtype=pl.FP32, layout=pl.TensorLayout.ND)
        li_transfer_inline271__ssa_v0: pl.Tensor[[1536, 1], pl.FP32, pl.MemRef("mem_ddr_25", pl.const(0, pl.INT64), 6144)] = pl.tensor.create([1536, 1], dtype=pl.FP32, layout=pl.TensorLayout.ND)
        alpha_transfer_inline202__ssa_v0: pl.Tensor[[1536, 1], pl.FP32, pl.MemRef("mem_ddr_26", pl.const(0, pl.INT64), 6144)] = pl.tensor.create([1536, 1], dtype=pl.FP32, layout=pl.TensorLayout.ND)
        ffts_workspace_inline255__ssa_v0: pl.Tensor[[256], pl.INT64, pl.MemRef("mem_ddr_27", pl.const(0, pl.INT64), 2048)] = pl.tensor.create([256], dtype=pl.INT64, layout=pl.TensorLayout.ND)
        ret__tmp_v0_2: pl.Tuple[
            pl.Tensor[[t_heads_inline281__ssa_v0, 1], pl.FP32], pl.Tensor[[t_heads_inline281__ssa_v0, 1], pl.FP32], pl.Tensor[[t_heads_inline281__ssa_v0, 512], pl.FP32], pl.Scalar[pl.TASK_ID]
        ] = pl.spmd_submit(
            self.qk_pv_spmd,
            ffts_workspace_inline255__ssa_v0,
            t_dim_inline216__ssa_v0,
            q_flat_inline248__ssa_v0,
            valid_block_mask_inline205__ssa_v0,
            kv_transfer_inline201__ssa_v0,
            score_transfer_inline221__ssa_v0,
            probability_transfer_inline323__ssa_v0,
            pv_transfer_inline200__ssa_v0,
            attn_sink_col_inline234__ssa_v0,
            positions__ssa_v0,
            window_indices__ssa_v0,
            ori_kv_flat_inline239__ssa_v1,
            cmp_sparse_indices_inline233__rv_v2,
            cmp_table__ssa_v0,
            cmp_kv_flat_inline219__ssa_v0,
            sparse_bias_inline257__rv_v2,
            mi_transfer_inline251__ssa_v0,
            li_transfer_inline271__ssa_v0,
            alpha_transfer_inline202__ssa_v0,
            attn_mi_inline203__ssa_v0,
            attn_li_inline284__ssa_v0,
            attn_oi_inline204__ssa_v0,
            deps=[qk_plan_tid_inline247__ssa_v0],
            core_num=24,
            allow_early_resolve=True,
            attrs={
                "arg_directions": [
                    pl.adir.input,
                    pl.adir.scalar,
                    pl.adir.input,
                    pl.adir.input,
                    pl.adir.inout,
                    pl.adir.inout,
                    pl.adir.inout,
                    pl.adir.inout,
                    pl.adir.input,
                    pl.adir.input,
                    pl.adir.input,
                    pl.adir.input,
                    pl.adir.input,
                    pl.adir.input,
                    pl.adir.input,
                    pl.adir.input,
                    pl.adir.inout,
                    pl.adir.inout,
                    pl.adir.inout,
                    pl.adir.output_existing,
                    pl.adir.output_existing,
                    pl.adir.output_existing,
                ]
            },
        )
        attn_mi_inline203__ssa_v1: pl.Tensor[[t_heads_inline281__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_28", pl.const(0, pl.INT64), 0)] = ret__tmp_v0_2[0]
        attn_li_inline284__ssa_v1: pl.Tensor[[t_heads_inline281__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_29", pl.const(0, pl.INT64), 0)] = ret__tmp_v0_2[1]
        attn_oi_inline204__ssa_v1: pl.Tensor[[t_heads_inline281__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_30", pl.const(0, pl.INT64), 0)] = ret__tmp_v0_2[2]
        qk_tid_inline238__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_2[3]
        rope_cos_il_inline198__ssa_v0: pl.Tensor[[384, 64], pl.FP32, pl.MemRef("mem_ddr_31", pl.const(0, pl.INT64), 98304)] = pl.tensor.create([384, 64], dtype=pl.FP32, layout=pl.TensorLayout.ND)
        rope_sin_signed_inline223__ssa_v0: pl.Tensor[[384, 64], pl.FP32, pl.MemRef("mem_ddr_32", pl.const(0, pl.INT64), 98304)] = pl.tensor.create([384, 64], dtype=pl.FP32, layout=pl.TensorLayout.ND)
        rope_swap_idx_inline331__ssa_v0: pl.Tensor[[16, 64], pl.INT32, pl.MemRef("mem_ddr_33", pl.const(0, pl.INT64), 4096)] = pl.tensor.create([16, 64], dtype=pl.INT32, layout=pl.TensorLayout.ND)
        ret__tmp_v0_3: pl.Tuple[pl.Tensor[[16, 64], pl.INT32], pl.Tensor[[384, 64], pl.FP32], pl.Tensor[[384, 64], pl.FP32], pl.Scalar[pl.TASK_ID]] = pl.submit(
            self.rope_cs,
            rope_swap_idx_inline331__ssa_v0,
            rope_cos_il_inline198__ssa_v0,
            rope_sin_signed_inline223__ssa_v0,
            rope_cs_blocks_inline218__ssa_v0,
            cos__ssa_v0,
            sin__ssa_v0,
            allow_early_resolve=True,
            attrs={"arg_directions": [pl.adir.output_existing, pl.adir.output_existing, pl.adir.output_existing, pl.adir.scalar, pl.adir.input, pl.adir.input]},
        )
        rope_swap_idx_inline331__ssa_v1: pl.Tensor[[16, 64], pl.INT32, pl.MemRef("mem_ddr_34", pl.const(0, pl.INT64), 4096)] = ret__tmp_v0_3[0]
        rope_cos_il_inline198__rv_v2: pl.Tensor[[384, 64], pl.FP32, pl.MemRef("mem_ddr_35", pl.const(0, pl.INT64), 98304)] = ret__tmp_v0_3[1]
        rope_sin_signed_inline223__rv_v2: pl.Tensor[[384, 64], pl.FP32, pl.MemRef("mem_ddr_36", pl.const(0, pl.INT64), 98304)] = ret__tmp_v0_3[2]
        rope_tid_inline333__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_3[3]
        attn_mi_inline157__ssa_v0: pl.Tensor[[t_heads_inline281__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_37", pl.const(0, pl.INT64), 0)] = attn_mi_inline203__ssa_v1
        attn_li_inline163__ssa_v0: pl.Tensor[[t_heads_inline281__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_38", pl.const(0, pl.INT64), 0)] = attn_li_inline284__ssa_v1
        attn_oi_inline159__ssa_v0: pl.Tensor[[t_heads_inline281__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_39", pl.const(0, pl.INT64), 0)] = attn_oi_inline204__ssa_v1
        rope_cos_il_inline170__ssa_v0: pl.Tensor[[384, 64], pl.FP32, pl.MemRef("mem_ddr_40", pl.const(0, pl.INT64), 98304)] = rope_cos_il_inline198__rv_v2
        rope_sin_signed_inline165__ssa_v0: pl.Tensor[[384, 64], pl.FP32, pl.MemRef("mem_ddr_41", pl.const(0, pl.INT64), 98304)] = rope_sin_signed_inline223__rv_v2
        rope_swap_idx_inline162__ssa_v0: pl.Tensor[[16, 64], pl.INT32, pl.MemRef("mem_ddr_42", pl.const(0, pl.INT64), 4096)] = rope_swap_idx_inline331__ssa_v1
        qk_tid_inline172__ssa_v0: pl.Scalar[pl.TASK_ID] = qk_tid_inline238__ssa_v0
        rope_tid_inline173__ssa_v0: pl.Scalar[pl.TASK_ID] = rope_tid_inline333__ssa_v0
        t_dim_inline155__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(query__ssa_v0, 0)
        ret__tmp_v0_4: pl.Tuple[pl.Tensor[[3072, 4096], pl.BF16], pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(
            self.merge_norm_spmd,
            rope_swap_idx_inline162__ssa_v0,
            t_dim_inline155__ssa_v0,
            attn_li_inline163__ssa_v0,
            attn_oi_inline159__ssa_v0,
            rope_cos_il_inline170__ssa_v0,
            rope_sin_signed_inline165__ssa_v0,
            packed__ssa_v0,
            deps=[qk_tid_inline172__ssa_v0, rope_tid_inline173__ssa_v0],
            core_num=48,
            attrs={"arg_directions": [pl.adir.input, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.output_existing]},
        )
        packed__ssa_v1: pl.Tensor[[3072, 4096], pl.BF16, pl.MemRef("mem_ddr_43", pl.const(0, pl.INT64), 25165824)] = ret__tmp_v0_4[0]
        merge_tid_inline180__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_4[1]
        return packed__ssa_v1