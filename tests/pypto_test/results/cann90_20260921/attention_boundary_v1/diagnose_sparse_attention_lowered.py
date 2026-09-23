# pypto.program: _jit_diagnose_sparse_attention
import pypto.language as pl

B_DYN = pl.dynamic("B_DYN")
CMP_BLOCK_NUM_DYN = pl.dynamic("CMP_BLOCK_NUM_DYN")
COMPRESSED_TABLE_COLUMNS_DYN = pl.dynamic("COMPRESSED_TABLE_COLUMNS_DYN")
ORI_BLOCK_NUM_DYN = pl.dynamic("ORI_BLOCK_NUM_DYN")
T_DYN = pl.dynamic("T_DYN")
t_dim_inline1418__ssa_v0 = pl.dynamic("t_dim_inline1418__ssa_v0")


@pl.program
class _jit_diagnose_sparse_attention:
    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def csa_slots_build_valid_qk_plan(
        cmp_sparse_indices_inline1373__ssa_v0: pl.Out[pl.Tensor[[t_dim_inline1418__ssa_v0, 512], pl.INT32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)]],
        sparse_bias_inline1413__ssa_v0: pl.Out[pl.Tensor[[t_dim_inline1418__ssa_v0, 640], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)]],
        t_dim_inline1418__ssa_v0: pl.Scalar[pl.INDEX],
        topk__ssa_v0: pl.Tensor[[T_DYN, 512], pl.INT32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)],
        positions__ssa_v0: pl.Tensor[[T_DYN, 1], pl.INT32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)],
        window_indices__ssa_v0: pl.Tensor[[T_DYN, 128], pl.INT32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 0)],
        valid_block_mask_inline1376__ssa_v0: pl.Out[pl.Tensor[[t_dim_inline1418__ssa_v0, 16], pl.INT32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 0)]],
    ) -> tuple[pl.Tensor[[t_dim_inline1418__ssa_v0, 512], pl.INT32], pl.Tensor[[t_dim_inline1418__ssa_v0, 640], pl.FP32]]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        mem_vec_6: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
        mem_vec_14: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
        mem_vec_16: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
        mem_vec_32: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 4096)
        mem_vec_33: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32)
        plan_worker_inline1417__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        for bias_t0_inline1428__idx_v0, (cmp_sparse_indices_inline1373__iter_v1, sparse_bias_inline1413__iter_v1) in pl.range(
            plan_worker_inline1417__ssa_v0 * 8, t_dim_inline1418__ssa_v0, 128, init_values=(cmp_sparse_indices_inline1373__ssa_v0, sparse_bias_inline1413__ssa_v0)
        ):
            t__tile: pl.Tile[[8, 512], pl.INT32, pl.MemRef(mem_vec_6, pl.const(36896, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.load(
                topk__ssa_v0, [bias_t0_inline1428__idx_v0, 0], [8, 512], [8, 512], target_memory=pl.Mem.Vec
            )
            c_raw_inline1405__tile: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_6, pl.const(36896, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.cast(t__tile, target_type=pl.FP32, mode="round")
            t__tile_1: pl.Tile[[8, 1], pl.INT32, pl.MemRef(mem_vec_14, pl.const(0, pl.INT64), 32), pl.Mem.Vec] = pl.tile.load(
                positions__ssa_v0, [bias_t0_inline1428__idx_v0, 0], [8, 1], [8, 1], target_memory=pl.Mem.Vec
            )
            c_pos_inline1404__rm_a0_tmp_v0: pl.Tile[[1, 8], pl.INT32, pl.MemRef(mem_vec_14, pl.const(0, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(t__tile_1, [1, 8])
            c_pos_inline1404__row_major_tmp_v1: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_14, pl.const(0, pl.INT64), 32), pl.Mem.Vec] = pl.tile.cast(
                c_pos_inline1404__rm_a0_tmp_v0, target_type=pl.FP32, mode="round"
            )
            c_pos_inline1404__tile: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_14, pl.const(0, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(c_pos_inline1404__row_major_tmp_v1, [8, 1])
            t__rm_a0_tmp_v2: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_14, pl.const(0, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(c_pos_inline1404__tile, [1, 8])
            t__row_major_tmp_v3: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_14, pl.const(0, pl.INT64), 32), pl.Mem.Vec] = pl.tile.adds(t__rm_a0_tmp_v2, 1.0)
            t__tile_2: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_14, pl.const(0, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(t__row_major_tmp_v3, [8, 1])
            c_pos_scaled_inline1370__rm_a0_tmp_v4: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_14, pl.const(0, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(t__tile_2, [1, 8])
            c_pos_scaled_inline1370__row_major_tmp_v5: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_14, pl.const(0, pl.INT64), 32), pl.Mem.Vec] = pl.tile.muls(
                c_pos_scaled_inline1370__rm_a0_tmp_v4, 0.25
            )
            c_pos_scaled_inline1370__tile: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_14, pl.const(0, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(c_pos_scaled_inline1370__row_major_tmp_v5, [8, 1])
            c_pos_i32_inline1426__rm_a0_tmp_v6: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_14, pl.const(0, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(c_pos_scaled_inline1370__tile, [1, 8])
            c_pos_i32_inline1426__row_major_tmp_v7: pl.Tile[[1, 8], pl.INT32, pl.MemRef(mem_vec_14, pl.const(0, pl.INT64), 32), pl.Mem.Vec] = pl.tile.cast(
                c_pos_i32_inline1426__rm_a0_tmp_v6, target_type=pl.INT32, mode="trunc"
            )
            c_pos_i32_inline1426__tile: pl.Tile[[8, 1], pl.INT32, pl.MemRef(mem_vec_14, pl.const(0, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(c_pos_i32_inline1426__row_major_tmp_v7, [8, 1])
            c_pos_q_inline1388__rm_a0_tmp_v8: pl.Tile[[1, 8], pl.INT32, pl.MemRef(mem_vec_14, pl.const(0, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(c_pos_i32_inline1426__tile, [1, 8])
            c_pos_q_inline1388__row_major_tmp_v9: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_16, pl.const(16384, pl.INT64), 32), pl.Mem.Vec] = pl.tile.cast(
                c_pos_q_inline1388__rm_a0_tmp_v8, target_type=pl.FP32, mode="round"
            )
            c_pos_q_inline1388__tile: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_16, pl.const(16384, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(c_pos_q_inline1388__row_major_tmp_v9, [8, 1])
            t__tile_3: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_14, pl.const(0, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.full([8, 512], dtype=pl.FP32, value=1.0)
            c_upper_b_inline1407__tile: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_14, pl.const(0, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.row_expand_mul(t__tile_3, c_pos_q_inline1388__tile)
            t__tile_4: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_16, pl.const(16384, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.adds(c_raw_inline1405__tile, 1.0)
            t__tile_5: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_16, pl.const(16384, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.maximums(t__tile_4, 0.0)
            c_ge_inline1454__tile: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_16, pl.const(16384, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.minimums(t__tile_5, 1.0)
            t__tile_6: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_14, pl.const(0, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.sub(c_upper_b_inline1407__tile, c_raw_inline1405__tile)
            t__tile_7: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_14, pl.const(0, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.maximums(t__tile_6, 0.0)
            c_lt_inline1438__tile: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_14, pl.const(0, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.minimums(t__tile_7, 1.0)
            c_mask_inline1439__tile: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_14, pl.const(0, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.mul(c_ge_inline1454__tile, c_lt_inline1438__tile)
            t__tile_8: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_6, pl.const(36896, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.adds(c_raw_inline1405__tile, 1.0)
            t__tile_9: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_6, pl.const(36896, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.mul(c_mask_inline1439__tile, t__tile_8)
            c_out_inline1446__tile: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_6, pl.const(36896, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.subs(t__tile_9, 1.0)
            t__tile_10: pl.Tile[[8, 512], pl.INT32, pl.MemRef(mem_vec_16, pl.const(16384, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.cast(c_out_inline1446__tile, target_type=pl.INT32, mode="round")
            cmp_sparse_indices_inline1373__tile: pl.Tensor[[t_dim_inline1418__ssa_v0, 512], pl.INT32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                t__tile_10, [bias_t0_inline1428__idx_v0, 0], cmp_sparse_indices_inline1373__iter_v1
            )
            t__tile_11: pl.Tile[[8, 128], pl.INT32, pl.MemRef(mem_vec_16, pl.const(16384, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.load(
                window_indices__ssa_v0, [bias_t0_inline1428__idx_v0, 0], [8, 128], [8, 128], target_memory=pl.Mem.Vec
            )
            v_win_f_inline1470__tile: pl.Tile[[8, 128], pl.FP32, pl.MemRef(mem_vec_16, pl.const(16384, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(t__tile_11, target_type=pl.FP32, mode="round")
            t__tile_12: pl.Tile[[8, 128], pl.FP32, pl.MemRef(mem_vec_16, pl.const(16384, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.adds(v_win_f_inline1470__tile, 1.0)
            t__tile_13: pl.Tile[[8, 128], pl.FP32, pl.MemRef(mem_vec_16, pl.const(16384, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.maximums(t__tile_12, 0.0)
            v_win_valid_inline1411__tile: pl.Tile[[8, 128], pl.FP32, pl.MemRef(mem_vec_16, pl.const(16384, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.minimums(t__tile_13, 1.0)
            tmp_tile: pl.Tile[[8, 128], pl.FP32, pl.MemRef(mem_vec_32, pl.const(32768, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.create([8, 128], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            raw_block_valid_inline1406__tile: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_33, pl.const(36864, pl.INT64), 32), pl.Mem.Vec] = pl.tile.row_max(v_win_valid_inline1411__tile, tmp_tile)
            for c_t0_inline1452__idx_v0 in pl.range(8):
                t__tile_14: pl.Scalar[pl.FP32] = pl.tile.read(raw_block_valid_inline1406__tile, [c_t0_inline1452__idx_v0, 0])
                c_valid_inline1422__ssa_v0: pl.Scalar[pl.INT32] = pl.cast(t__tile_14, pl.INT32)
                pl.tensor.write(valid_block_mask_inline1376__ssa_v0, [bias_t0_inline1428__idx_v0 + c_t0_inline1452__idx_v0, 0], c_valid_inline1422__ssa_v0)
            for c_sb_inline1450__idx_v0 in pl.range(1, 5):
                c_s0_inline1427__ssa_v0: pl.Scalar[pl.INDEX] = (c_sb_inline1450__idx_v0 - 1) * 128
                t__tile_15: pl.Tile[[8, 128], pl.FP32, pl.MemRef(mem_vec_14, pl.const(0, pl.INT64), 14848), pl.Mem.Vec] = pl.tile.slice(c_mask_inline1439__tile, [8, 128], [0, c_s0_inline1427__ssa_v0])
                tmp_tile_1: pl.Tile[[8, 128], pl.FP32, pl.MemRef(mem_vec_32, pl.const(32768, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.create([8, 128], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                c_blk_valid_inline1433__tile: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_33, pl.const(36864, pl.INT64), 32), pl.Mem.Vec] = pl.tile.row_max(t__tile_15, tmp_tile_1)
                for c_dt_inline1384__idx_v0 in pl.range(8):
                    t__tile_16: pl.Scalar[pl.FP32] = pl.tile.read(c_blk_valid_inline1433__tile, [c_dt_inline1384__idx_v0, 0])
                    c_valid_inline1422__ssa_v1: pl.Scalar[pl.INT32] = pl.cast(t__tile_16, pl.INT32)
                    pl.tensor.write(valid_block_mask_inline1376__ssa_v0, [bias_t0_inline1428__idx_v0 + c_dt_inline1384__idx_v0, c_sb_inline1450__idx_v0], c_valid_inline1422__ssa_v1)
            t__tile_17: pl.Tile[[8, 128], pl.FP32, pl.MemRef(mem_vec_14, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.subs(v_win_valid_inline1411__tile, 1.0)
            t__tile_18: pl.Tile[[8, 128], pl.FP32, pl.MemRef(mem_vec_14, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.muls(t__tile_17, 1e20)
            sparse_bias_inline1413__tile: pl.Tensor[[t_dim_inline1418__ssa_v0, 640], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                t__tile_18, [bias_t0_inline1428__idx_v0, 0], sparse_bias_inline1413__iter_v1
            )
            t__tile_19: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_6, pl.const(36896, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.minimums(c_out_inline1446__tile, 0.0)
            t__tile_20: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_6, pl.const(36896, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.muls(t__tile_19, 1e20)
            sparse_bias_inline1413__tile_1: pl.Tensor[[t_dim_inline1418__ssa_v0, 640], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                t__tile_20, [bias_t0_inline1428__idx_v0, 128], sparse_bias_inline1413__tile
            )
            cmp_sparse_indices_inline1373__rv_v2, sparse_bias_inline1413__rv_v2 = pl.yield_(cmp_sparse_indices_inline1373__tile, sparse_bias_inline1413__tile_1)
        return cmp_sparse_indices_inline1373__ssa_v0, sparse_bias_inline1413__ssa_v0

    @pl.function(type=pl.FunctionType.Spmd)
    def csa_slots_build_valid_qk_plan_spmd(
        self,
        cmp_sparse_indices_inline1373__ssa_v0: pl.Out[pl.Tensor[[t_dim_inline1418__ssa_v0, 512], pl.INT32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)]],
        sparse_bias_inline1413__ssa_v0: pl.Out[pl.Tensor[[t_dim_inline1418__ssa_v0, 640], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)]],
        t_dim_inline1418__ssa_v0: pl.Scalar[pl.INDEX],
        topk__ssa_v0: pl.Tensor[[T_DYN, 512], pl.INT32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)],
        positions__ssa_v0: pl.Tensor[[T_DYN, 1], pl.INT32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)],
        window_indices__ssa_v0: pl.Tensor[[T_DYN, 128], pl.INT32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 0)],
        valid_block_mask_inline1376__ssa_v0: pl.Out[pl.Tensor[[t_dim_inline1418__ssa_v0, 16], pl.INT32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 0)]],
    ) -> tuple[pl.Tensor[[t_dim_inline1418__ssa_v0, 512], pl.INT32], pl.Tensor[[t_dim_inline1418__ssa_v0, 640], pl.FP32]]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        ret__tmp_v0: pl.Tuple[pl.Tensor[[t_dim_inline1418__ssa_v0, 512], pl.INT32], pl.Tensor[[t_dim_inline1418__ssa_v0, 640], pl.FP32]] = self.csa_slots_build_valid_qk_plan(
            cmp_sparse_indices_inline1373__ssa_v0,
            sparse_bias_inline1413__ssa_v0,
            t_dim_inline1418__ssa_v0,
            topk__ssa_v0,
            positions__ssa_v0,
            window_indices__ssa_v0,
            valid_block_mask_inline1376__ssa_v0,
            attrs={"arg_directions": [pl.adir.output_existing, pl.adir.output_existing, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.output_existing]},
        )
        cmp_sparse_indices_inline1373__rv_v2: pl.Tensor[[t_dim_inline1418__ssa_v0, 512], pl.INT32, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 0)] = ret__tmp_v0[0]
        sparse_bias_inline1413__rv_v2: pl.Tensor[[t_dim_inline1418__ssa_v0, 640], pl.FP32, pl.MemRef("mem_ddr_7", pl.const(0, pl.INT64), 0)] = ret__tmp_v0[1]
        return cmp_sparse_indices_inline1373__ssa_v0, sparse_bias_inline1413__ssa_v0

    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def kv_touch(
        ori_kv_flat_inline1480__ssa_v0: pl.InOut[pl.Tensor[[ori_block_num_inline1408__ssa_v0 * pl.const(32, pl.INDEX), 512], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)]],
    ) -> pl.Tensor[[ori_block_num_inline1408__ssa_v0 * pl.const(32, pl.INDEX), 512], pl.BF16]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        mem_vec_1: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 1024)
        t__tile: pl.Tile[[1, 512], pl.BF16, pl.MemRef(mem_vec_1, pl.const(0, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.load(
            ori_kv_flat_inline1480__ssa_v0, [0, 0], [1, 512], [1, 512], target_memory=pl.Mem.Vec
        )
        ori_kv_flat_inline1480__tile: pl.Tensor[[ori_block_num_inline1408__ssa_v0 * pl.const(32, pl.INDEX), 512], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)] = pl.tile.store(
            t__tile, [0, 0], ori_kv_flat_inline1480__ssa_v0
        )
        return ori_kv_flat_inline1480__ssa_v0

    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def merge_norm(
        rope_swap_idx_inline1327__ssa_v0: pl.Tensor[[16, 64], pl.INT32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 4096)],
        t_dim_inline1326__ssa_v0: pl.Scalar[pl.INDEX],
        attn_mi_inline1336__ssa_v0: pl.Tensor[[t_heads_inline1477__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)],
        attn_li_inline1331__ssa_v0: pl.Tensor[[t_heads_inline1477__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)],
        attn_oi_inline1328__ssa_v0: pl.Tensor[[t_heads_inline1477__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)],
        merge_sink_inline1332__ssa_v0: pl.Tensor[[64, 1], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 256)],
        rope_cos_il_inline1335__ssa_v0: pl.Tensor[[384, 64], pl.FP32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 98304)],
        rope_sin_signed_inline1345__ssa_v0: pl.Tensor[[384, 64], pl.FP32, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 98304)],
        packed__ssa_v0: pl.Out[pl.Tensor[[3072, 4096], pl.BF16, pl.MemRef("mem_ddr_7", pl.const(0, pl.INT64), 25165824)]],
    ) -> pl.Tensor[[3072, 4096], pl.BF16]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        mem_vec_16: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 4096)
        mem_vec_17: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 4096)
        mem_vec_20: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32768)
        mem_vec_28: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
        mem_vec_30: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        mem_vec_31: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        mem_vec_32: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 4096)
        mem_vec_33: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 4096)
        m_worker_inline1349__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        m_swap_inline1339__ssa_v0: pl.Tile[[16, 64], pl.INT32, pl.MemRef(mem_vec_20, pl.const(8192, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.load(
            rope_swap_idx_inline1327__ssa_v0, [0, 0], [16, 64], [16, 64], target_memory=pl.Mem.Vec
        )
        m_swap_f_inline1343__ssa_v0: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_20, pl.const(8192, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(
            m_swap_inline1339__ssa_v0, target_type=pl.FP32, mode="round"
        )
        m_swap_source_inline1334__ssa_v0: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_20, pl.const(8192, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.adds(m_swap_f_inline1343__ssa_v0, 448.0)
        m_row_ids_inline1333__ci_tmp_v0: pl.Tile[[1, 192], pl.FP32, pl.MemRef(mem_vec_28, pl.const(40960, pl.INT64), 768), pl.Mem.Vec] = pl.tile.create(
            [1, 192], dtype=pl.FP32, target_memory=pl.Mem.Vec
        )
        m_row_ids_inline1333__ssa_v0: pl.Tile[[1, 16], pl.INT32, pl.MemRef(mem_vec_16, pl.const(0, pl.INT64), 64), pl.Mem.Vec] = pl.tile.ci(
            pl.const(0, pl.INT32), [1, 16], tmp=m_row_ids_inline1333__ci_tmp_v0, dtype=pl.INT32, descending=False
        )
        m_row_ids_f_inline1342__ssa_v0: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_28, pl.const(40960, pl.INT64), 64), pl.Mem.Vec] = pl.tile.cast(
            m_row_ids_inline1333__ssa_v0, target_type=pl.FP32, mode="round"
        )
        m_row_offsets_inline1348__ssa_v0: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_28, pl.const(40960, pl.INT64), 64), pl.Mem.Vec] = pl.tile.muls(m_row_ids_f_inline1342__ssa_v0, 512.0)
        m_row_offsets_col_inline1353__ssa_v0: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_28, pl.const(40960, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(m_row_offsets_inline1348__ssa_v0, [16, 1])
        m_swap_flat_inline1347__ssa_v0: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_20, pl.const(8192, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.row_expand_add(
            m_swap_source_inline1334__ssa_v0, m_row_offsets_col_inline1353__ssa_v0
        )
        m_swap_idx_inline1350__ssa_v0: pl.Tile[[16, 64], pl.INT32, pl.MemRef(mem_vec_16, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(
            m_swap_flat_inline1347__ssa_v0, target_type=pl.INT32, mode="round"
        )
        m_gather_tmp_inline1324__ssa_v0: pl.Tile[[16, 64], pl.INT32, pl.MemRef(mem_vec_17, pl.const(4096, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.create(
            [16, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec
        )
        for m_idx_inline1344__idx_v0 in pl.range(m_worker_inline1349__ssa_v0, t_dim_inline1326__ssa_v0 * 4, 48):
            m_t_inline1321__ssa_v0: pl.Scalar[pl.INDEX] = m_idx_inline1344__idx_v0 // 4
            m_h_idx_inline1354__ssa_v0: pl.Scalar[pl.INDEX] = m_idx_inline1344__idx_v0 - m_t_inline1321__ssa_v0 * 4
            m_h0_inline1340__ssa_v0: pl.Scalar[pl.INDEX] = m_h_idx_inline1354__ssa_v0 * 16
            m_row_inline1330__ssa_v0: pl.Scalar[pl.INDEX] = m_idx_inline1344__idx_v0 * 16
            m_mi_inline1323__ssa_v0: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_28, pl.const(40960, pl.INT64), 64), pl.Mem.Vec] = pl.tile.load(
                attn_mi_inline1336__ssa_v0, [m_row_inline1330__ssa_v0, 0], [16, 1], [16, 1], target_memory=pl.Mem.Vec
            )
            m_li_inline1341__ssa_v0: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_32, pl.const(57856, pl.INT64), 64), pl.Mem.Vec] = pl.tile.load(
                attn_li_inline1331__ssa_v0, [m_row_inline1330__ssa_v0, 0], [16, 1], [16, 1], target_memory=pl.Mem.Vec
            )
            m_oi_inline1320__ssa_v0: pl.Tile[[16, 512], pl.FP32, pl.MemRef(mem_vec_20, pl.const(8192, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.load(
                attn_oi_inline1328__ssa_v0, [m_row_inline1330__ssa_v0, 0], [16, 512], [16, 512], target_memory=pl.Mem.Vec
            )
            n_sink_bias_inline1319__ssa_v0: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_33, pl.const(61952, pl.INT64), 64), pl.Mem.Vec] = pl.tile.load(
                merge_sink_inline1332__ssa_v0, [m_h0_inline1340__ssa_v0, 0], [16, 1], [16, 1], target_memory=pl.Mem.Vec
            )
            t__rm_a0_tmp_v0: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_28, pl.const(40960, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(m_mi_inline1323__ssa_v0, [1, 16])
            t__rm_a1_tmp_v1: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_28, pl.const(40960, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(m_mi_inline1323__ssa_v0, [1, 16])
            t__row_major_tmp_v2: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_30, pl.const(57344, pl.INT64), 64), pl.Mem.Vec] = pl.tile.sub(t__rm_a0_tmp_v0, t__rm_a1_tmp_v1)
            t__tmp_v52: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_30, pl.const(57344, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(t__row_major_tmp_v2, [16, 1])
            n_sink_tile_inline1316__rm_a0_tmp_v3: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_30, pl.const(57344, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(t__tmp_v52, [1, 16])
            n_sink_tile_inline1316__rm_a1_tmp_v4: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_33, pl.const(61952, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(n_sink_bias_inline1319__ssa_v0, [1, 16])
            n_sink_tile_inline1316__row_major_tmp_v5: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_33, pl.const(61952, pl.INT64), 64), pl.Mem.Vec] = pl.tile.add(
                n_sink_tile_inline1316__rm_a0_tmp_v3, n_sink_tile_inline1316__rm_a1_tmp_v4
            )
            n_sink_tile_inline1316__ssa_v0: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_33, pl.const(61952, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(
                n_sink_tile_inline1316__row_major_tmp_v5, [16, 1]
            )
            t__rm_a0_tmp_v6: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_33, pl.const(61952, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(n_sink_tile_inline1316__ssa_v0, [1, 16])
            t__rm_a1_tmp_v7: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_28, pl.const(40960, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(m_mi_inline1323__ssa_v0, [1, 16])
            t__row_major_tmp_v8: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_28, pl.const(40960, pl.INT64), 64), pl.Mem.Vec] = pl.tile.sub(t__rm_a0_tmp_v6, t__rm_a1_tmp_v7)
            t__tmp_v53: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_28, pl.const(40960, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(t__row_major_tmp_v8, [16, 1])
            t__rm_a0_tmp_v9: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_28, pl.const(40960, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(t__tmp_v53, [1, 16])
            t__row_major_tmp_v10: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_28, pl.const(40960, pl.INT64), 64), pl.Mem.Vec] = pl.tile.exp(t__rm_a0_tmp_v9)
            t__tmp_v54: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_28, pl.const(40960, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(t__row_major_tmp_v10, [16, 1])
            n_denom_inline1338__rm_a0_tmp_v11: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_32, pl.const(57856, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(m_li_inline1341__ssa_v0, [1, 16])
            n_denom_inline1338__rm_a1_tmp_v12: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_28, pl.const(40960, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(t__tmp_v54, [1, 16])
            n_denom_inline1338__row_major_tmp_v13: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_28, pl.const(40960, pl.INT64), 64), pl.Mem.Vec] = pl.tile.add(
                n_denom_inline1338__rm_a0_tmp_v11, n_denom_inline1338__rm_a1_tmp_v12
            )
            n_denom_inline1338__ssa_v0: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_28, pl.const(40960, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(n_denom_inline1338__row_major_tmp_v13, [16, 1])
            n_full_inline1315__ssa_v0: pl.Tile[[16, 512], pl.FP32, pl.MemRef(mem_vec_20, pl.const(8192, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.row_expand_div(
                m_oi_inline1320__ssa_v0, n_denom_inline1338__ssa_v0
            )
            n_bf16_inline1314__ssa_v0: pl.Tile[[16, 512], pl.BF16, pl.MemRef(mem_vec_28, pl.const(40960, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.cast(
                n_full_inline1315__ssa_v0, target_type=pl.BF16, mode="rint"
            )
            n_rounded_inline1313__ssa_v0: pl.Tile[[16, 512], pl.FP32, pl.MemRef(mem_vec_20, pl.const(8192, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.cast(
                n_bf16_inline1314__ssa_v0, target_type=pl.FP32, mode="round"
            )
            m_cos_il_inline1310__ssa_v0: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_30, pl.const(57344, pl.INT64), 256), pl.Mem.Vec] = pl.tile.load(
                rope_cos_il_inline1335__ssa_v0, [m_t_inline1321__ssa_v0, 0], [1, 64], [1, 64], target_memory=pl.Mem.Vec
            )
            m_sin_signed_inline1309__ssa_v0: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_31, pl.const(57600, pl.INT64), 256), pl.Mem.Vec] = pl.tile.load(
                rope_sin_signed_inline1345__ssa_v0, [m_t_inline1321__ssa_v0, 0], [1, 64], [1, 64], target_memory=pl.Mem.Vec
            )
            m_swapped_inline1318__ssa_v0: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_32, pl.const(57856, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.gather(
                n_rounded_inline1313__ssa_v0, m_swap_idx_inline1350__ssa_v0, m_gather_tmp_inline1324__ssa_v0
            )
            m_rope_inline1312__ssa_v0_textract: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_33, pl.const(61952, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.extract(
                n_rounded_inline1313__ssa_v0, 0, 448, [16, 64], target_memory=pl.Mem.Vec
            )
            t__tmp_v55: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_20, pl.const(8192, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.col_expand_mul(
                m_rope_inline1312__ssa_v0_textract, m_cos_il_inline1310__ssa_v0
            )
            t__tmp_v56: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_32, pl.const(57856, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.col_expand_mul(
                m_swapped_inline1318__ssa_v0, m_sin_signed_inline1309__ssa_v0
            )
            m_rot_inline1308__ssa_v0: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_20, pl.const(8192, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.add(t__tmp_v55, t__tmp_v56)
            n_rope_bf16_inline1311__ssa_v0: pl.Tile[[16, 64], pl.BF16, pl.MemRef(mem_vec_32, pl.const(57856, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(
                m_rot_inline1308__ssa_v0, target_type=pl.BF16, mode="rint"
            )
            t__tmp_v57: pl.Tile[[16, 448], pl.BF16, pl.MemRef(mem_vec_28, pl.const(40960, pl.INT64), 16256), pl.Mem.Vec] = pl.tile.slice(n_bf16_inline1314__ssa_v0, [16, 448], [0, 0])
            n_full_bf16_inline1307__ssa_v0: pl.Tile[[16, 512], pl.BF16, pl.MemRef(mem_vec_20, pl.const(8192, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.concat(t__tmp_v57, n_rope_bf16_inline1311__ssa_v0)
            n_group_bf16_inline1352__ssa_v0: pl.Tile[[2, 4096], pl.BF16, pl.MemRef(mem_vec_20, pl.const(8192, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.reshape(
                n_full_bf16_inline1307__ssa_v0, [2, 4096]
            )
            n_pack_first_inline1351__ssa_v0: pl.Tile[[1, 4096], pl.BF16, pl.MemRef(mem_vec_20, pl.const(8192, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.slice(
                n_group_bf16_inline1352__ssa_v0, [1, 4096], [0, 0]
            )
            n_pack_second_inline1306__ssa_v0: pl.Tile[[1, 4096], pl.BF16, pl.MemRef(mem_vec_20, pl.const(16384, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.slice(
                n_group_bf16_inline1352__ssa_v0, [1, 4096], [1, 0]
            )
            n_pack_row_inline1317__ssa_v0: pl.Scalar[pl.INDEX] = m_h0_inline1340__ssa_v0 // 8 * 384 + m_t_inline1321__ssa_v0
            n_pack_row_second_inline1322__ssa_v0: pl.Scalar[pl.INDEX] = n_pack_row_inline1317__ssa_v0 + 384
            packed__store: pl.Tensor[[3072, 4096], pl.BF16, pl.MemRef("mem_ddr_7", pl.const(0, pl.INT64), 25165824)] = pl.tile.store(
                n_pack_first_inline1351__ssa_v0, [n_pack_row_inline1317__ssa_v0, 0], packed__ssa_v0
            )
            packed__store_v0: pl.Tensor[[3072, 4096], pl.BF16, pl.MemRef("mem_ddr_7", pl.const(0, pl.INT64), 25165824)] = pl.tile.store(
                n_pack_second_inline1306__ssa_v0, [n_pack_row_second_inline1322__ssa_v0, 0], packed__ssa_v0
            )
        return packed__ssa_v0

    @pl.function(type=pl.FunctionType.Spmd)
    def merge_norm_spmd(
        self,
        rope_swap_idx_inline1327__ssa_v0: pl.Tensor[[16, 64], pl.INT32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 4096)],
        t_dim_inline1326__ssa_v0: pl.Scalar[pl.INDEX],
        attn_mi_inline1336__ssa_v0: pl.Tensor[[t_heads_inline1477__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)],
        attn_li_inline1331__ssa_v0: pl.Tensor[[t_heads_inline1477__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)],
        attn_oi_inline1328__ssa_v0: pl.Tensor[[t_heads_inline1477__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)],
        merge_sink_inline1332__ssa_v0: pl.Tensor[[64, 1], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 256)],
        rope_cos_il_inline1335__ssa_v0: pl.Tensor[[384, 64], pl.FP32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 98304)],
        rope_sin_signed_inline1345__ssa_v0: pl.Tensor[[384, 64], pl.FP32, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 98304)],
        packed__ssa_v0: pl.Out[pl.Tensor[[3072, 4096], pl.BF16, pl.MemRef("mem_ddr_7", pl.const(0, pl.INT64), 25165824)]],
    ) -> pl.Tensor[[3072, 4096], pl.BF16]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        packed__ssa_v1: pl.Tensor[[3072, 4096], pl.BF16, pl.MemRef("mem_ddr_8", pl.const(0, pl.INT64), 25165824)] = self.merge_norm(
            rope_swap_idx_inline1327__ssa_v0,
            t_dim_inline1326__ssa_v0,
            attn_mi_inline1336__ssa_v0,
            attn_li_inline1331__ssa_v0,
            attn_oi_inline1328__ssa_v0,
            merge_sink_inline1332__ssa_v0,
            rope_cos_il_inline1335__ssa_v0,
            rope_sin_signed_inline1345__ssa_v0,
            packed__ssa_v0,
            attrs={"arg_directions": [pl.adir.input, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.output_existing]},
        )
        return packed__ssa_v0

    @pl.function(type=pl.FunctionType.AIC, level=pl.Level.AIC, role=pl.Role.SubWorker)
    def qk_pv_aic(
        ffts_workspace_inline1401__ssa_v0: pl.Tensor[[256], pl.INT64, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 2048)],
        t_dim_inline1418__ssa_v0: pl.Scalar[pl.INDEX],
        q_flat_inline1409__ssa_v0: pl.Tensor[[t_heads_inline1477__ssa_v0, 512], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)],
        valid_block_mask_inline1376__ssa_v0: pl.Tensor[[t_dim_inline1418__ssa_v0, 16], pl.INT32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)],
        kv_transfer_inline1392__ssa_v0: pl.InOut[pl.Tensor[[9216, 512], pl.BF16, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 9437184)]],
        score_transfer_inline1382__ssa_v0: pl.InOut[pl.Tensor[[4608, 128], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 2359296)]],
        probability_transfer_inline1394__ssa_v0: pl.InOut[pl.Tensor[[4608, 128], pl.BF16, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 1179648)]],
        pv_transfer_inline1381__ssa_v0: pl.InOut[pl.Tensor[[4608, 512], pl.FP32, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 9437184)]],
        attn_sink_col_inline1396__ssa_v0: pl.Tensor[[64, 1], pl.FP32, pl.MemRef("mem_ddr_7", pl.const(0, pl.INT64), 256)],
        positions__ssa_v0: pl.Tensor[[T_DYN, 1], pl.INT32, pl.MemRef("mem_ddr_8", pl.const(0, pl.INT64), 0)],
        window_indices__ssa_v0: pl.Tensor[[T_DYN, 128], pl.INT32, pl.MemRef("mem_ddr_9", pl.const(0, pl.INT64), 0)],
        ori_kv_flat_inline1480__ssa_v1: pl.Tensor[[ori_block_num_inline1408__ssa_v0 * pl.const(32, pl.INDEX), 512], pl.BF16, pl.MemRef("mem_ddr_10", pl.const(0, pl.INT64), 0)],
        cmp_sparse_indices_inline1373__rv_v2: pl.Tensor[[t_dim_inline1418__ssa_v0, 512], pl.INT32, pl.MemRef("mem_ddr_11", pl.const(0, pl.INT64), 0)],
        cmp_table__ssa_v0: pl.Tensor[[B_DYN, COMPRESSED_TABLE_COLUMNS_DYN], pl.INT32, pl.MemRef("mem_ddr_12", pl.const(0, pl.INT64), 0)],
        cmp_kv_flat_inline1399__ssa_v0: pl.Tensor[[cmp_block_num_inline1456__ssa_v0 * pl.const(32, pl.INDEX), 512], pl.BF16, pl.MemRef("mem_ddr_13", pl.const(0, pl.INT64), 0)],
        sparse_bias_inline1413__rv_v2: pl.Tensor[[t_dim_inline1418__ssa_v0, 640], pl.FP32, pl.MemRef("mem_ddr_14", pl.const(0, pl.INT64), 0)],
        mi_transfer_inline1379__ssa_v0: pl.InOut[pl.Tensor[[4608, 1], pl.FP32, pl.MemRef("mem_ddr_15", pl.const(0, pl.INT64), 18432)]],
        li_transfer_inline1432__ssa_v0: pl.InOut[pl.Tensor[[4608, 1], pl.FP32, pl.MemRef("mem_ddr_16", pl.const(0, pl.INT64), 18432)]],
        attn_mi_inline1435__ssa_v0: pl.Out[pl.Tensor[[t_heads_inline1477__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_17", pl.const(0, pl.INT64), 0)]],
        attn_li_inline1483__ssa_v0: pl.Out[pl.Tensor[[t_heads_inline1477__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_18", pl.const(0, pl.INT64), 0)]],
        attn_oi_inline1393__ssa_v0: pl.Out[pl.Tensor[[t_heads_inline1477__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_19", pl.const(0, pl.INT64), 0)]],
    ):
        pl.func_attr({"mx_tensor_views_blocked": True, "split_aiv": True})
        mem_mat_20: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 65536)
        mem_mat_21: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 393216)
        mem_left_23: pl.Ptr = pl.tile.alloc(pl.Mem.Left, 16384)
        mem_right_24: pl.Ptr = pl.tile.alloc(pl.Mem.Right, 32768)
        mem_left_25: pl.Ptr = pl.tile.alloc(pl.Mem.Left, 16384)
        mem_right_26: pl.Ptr = pl.tile.alloc(pl.Mem.Right, 32768)
        mem_mat_30: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 16384)
        mem_acc_31: pl.Ptr = pl.tile.alloc(pl.Mem.Acc, 131072)
        qk_core_inline1375__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        pl.system.set_ffts(ffts_workspace_inline1401__ssa_v0)
        for qk_t_inline1421__idx_v0 in pl.range(qk_core_inline1375__ssa_v0, t_dim_inline1418__ssa_v0, 24):
            qk_q_inline1372__ssa_v0: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_mat_20, pl.const(0, pl.INT64), 65536), pl.Mem.Mat] = pl.tile.load(
                q_flat_inline1409__ssa_v0, [qk_t_inline1421__idx_v0 * 64, 0], [64, 512], [64, 512], target_memory=pl.Mem.Mat
            )
            qk_l1_inline1448__ssa_v0: pl.Tile[[384, 512], pl.BF16, pl.MemRef(mem_mat_21, pl.const(65536, pl.INT64), 393216), pl.Mem.Mat] = pl.tile.create(
                [384, 512], dtype=pl.BF16, target_memory=pl.Mem.Mat
            )
            for qk_tick_inline1430__idx_v0, (qk_l1_inline1448__iter_v1,) in pl.range(7, init_values=(qk_l1_inline1448__ssa_v0,)):
                if qk_tick_inline1430__idx_v0 < 5:
                    qk_sb_inline1443__ssa_v0: pl.Scalar[pl.INDEX] = qk_tick_inline1430__idx_v0
                    t__tile: pl.Scalar[pl.INT32] = pl.tensor.read(valid_block_mask_inline1376__ssa_v0, [qk_t_inline1421__idx_v0, qk_sb_inline1443__ssa_v0])
                    if 0 < pl.cast(t__tile, pl.INDEX):
                        qk_slot_inline1369__ssa_v0: pl.Scalar[pl.INDEX] = qk_core_inline1375__ssa_v0 * 3 + qk_sb_inline1443__ssa_v0 % 3
                        qk_kv_row_inline1380__ssa_v0: pl.Scalar[pl.INDEX] = qk_slot_inline1369__ssa_v0 * 128
                        qk_transfer_row_inline1440__ssa_v0: pl.Scalar[pl.INDEX] = qk_slot_inline1369__ssa_v0 * 64
                        pl.system.sync_wait(0, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIC)
                        qk_l1_row_inline1475__ssa_v0: pl.Scalar[pl.INDEX] = qk_sb_inline1443__ssa_v0 % 3 * 128
                        qk_l1_inline1448__ssa_v3: pl.Tile[[384, 512], pl.BF16, pl.MemRef(mem_mat_21, pl.const(65536, pl.INT64), 393216), pl.Mem.Mat] = pl.tile.gather_row(
                            qk_l1_inline1448__iter_v1, kv_transfer_inline1392__ssa_v0, [qk_l1_row_inline1475__ssa_v0, 0], [qk_kv_row_inline1380__ssa_v0, 0], [128, 512], transpose=False
                        )
                        qk_l1_t_inline1455__ssa_v0: pl.Tile[
                            [512, 384], pl.BF16, pl.MemRef(mem_mat_21, pl.const(65536, pl.INT64), 393216), pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)
                        ] = pl.tile.transpose_view(qk_l1_inline1448__ssa_v3)
                        qk_scores_inline1487__ssa_v0_l0_init: pl.Tile[[64, 128], pl.FP32, pl.MemRef(mem_acc_31, pl.const(0, pl.INT64), 32768), pl.Mem.Acc] = pl.tile.create(
                            [64, 128], dtype=pl.FP32, target_memory=pl.Mem.Acc
                        )
                        for qk_scores_inline1487__ssa_v0_l0_ko, (qk_scores_inline1487__ssa_v0_l0_c,) in pl.range(0, 512, 256, init_values=(qk_scores_inline1487__ssa_v0_l0_init,)):
                            qk_scores_inline1487__ssa_v0_l0_a: pl.Tile[
                                [64, 128], pl.BF16, pl.MemRef(mem_left_23, pl.const(0, pl.INT64), 16384), pl.Mem.Left, pl.TileView(blayout=pl.TileLayout.row_major)
                            ] = pl.tile.extract(qk_q_inline1372__ssa_v0, 0, qk_scores_inline1487__ssa_v0_l0_ko, [64, 128], target_memory=pl.Mem.Left)
                            qk_scores_inline1487__ssa_v0_l0_b: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_24, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                                qk_l1_t_inline1455__ssa_v0, qk_scores_inline1487__ssa_v0_l0_ko, qk_l1_row_inline1475__ssa_v0, [128, 128], target_memory=pl.Mem.Right
                            )
                            qk_scores_inline1487__ssa_v0_l0_a_1: pl.Tile[
                                [64, 128], pl.BF16, pl.MemRef(mem_left_25, pl.const(16384, pl.INT64), 16384), pl.Mem.Left, pl.TileView(blayout=pl.TileLayout.row_major)
                            ] = pl.tile.extract(qk_q_inline1372__ssa_v0, 0, qk_scores_inline1487__ssa_v0_l0_ko + 128, [64, 128], target_memory=pl.Mem.Left)
                            qk_scores_inline1487__ssa_v0_l0_b_1: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_26, pl.const(32768, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                                qk_l1_t_inline1455__ssa_v0, qk_scores_inline1487__ssa_v0_l0_ko + 128, qk_l1_row_inline1475__ssa_v0, [128, 128], target_memory=pl.Mem.Right
                            )
                            qk_scores_inline1487__ssa_v0_l0_c_acc: pl.Tile[[64, 128], pl.FP32, pl.MemRef(mem_acc_31, pl.const(0, pl.INT64), 32768), pl.Mem.Acc] = pl.tile.matmul_acc(
                                qk_scores_inline1487__ssa_v0_l0_c, qk_scores_inline1487__ssa_v0_l0_a, qk_scores_inline1487__ssa_v0_l0_b, qk_scores_inline1487__ssa_v0_l0_ko == 0
                            )
                            qk_scores_inline1487__ssa_v0_l0_c_acc_1: pl.Tile[[64, 128], pl.FP32, pl.MemRef(mem_acc_31, pl.const(0, pl.INT64), 32768), pl.Mem.Acc] = pl.tile.matmul_acc(
                                qk_scores_inline1487__ssa_v0_l0_c_acc, qk_scores_inline1487__ssa_v0_l0_a_1, qk_scores_inline1487__ssa_v0_l0_b_1, qk_scores_inline1487__ssa_v0_l0_ko == -128
                            )
                            qk_scores_inline1487__ssa_v0: pl.Tile[[64, 128], pl.FP32, pl.MemRef(mem_acc_31, pl.const(0, pl.INT64), 32768), pl.Mem.Acc] = pl.yield_(
                                qk_scores_inline1487__ssa_v0_l0_c_acc_1
                            )
                        score_transfer_inline1382__store: pl.Tensor[[4608, 128], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 2359296)] = pl.tile.store(
                            qk_scores_inline1487__ssa_v0, [qk_transfer_row_inline1440__ssa_v0, 0], score_transfer_inline1382__ssa_v0
                        )
                        pl.system.sync_set(1, pipe=pl.PipeType.FIX, ffts_mode=2, core_type=pl.KernelType.AIC)
                        qk_l1_inline1448__phi_v4: pl.Tile[[384, 512], pl.BF16, pl.MemRef(mem_mat_21, pl.const(65536, pl.INT64), 393216), pl.Mem.Mat] = pl.yield_(qk_l1_inline1448__ssa_v3)
                    else:
                        qk_l1_inline1448__phi_v4: pl.Tile[[384, 512], pl.BF16, pl.MemRef(mem_mat_21, pl.const(65536, pl.INT64), 393216), pl.Mem.Mat] = pl.yield_(qk_l1_inline1448__iter_v1)
                    qk_l1_inline1448__phi_v5: pl.Tile[[384, 512], pl.BF16, pl.MemRef(mem_mat_21, pl.const(65536, pl.INT64), 393216), pl.Mem.Mat] = pl.yield_(qk_l1_inline1448__phi_v4)
                else:
                    qk_l1_inline1448__phi_v5: pl.Tile[[384, 512], pl.BF16, pl.MemRef(mem_mat_21, pl.const(65536, pl.INT64), 393216), pl.Mem.Mat] = pl.yield_(qk_l1_inline1448__iter_v1)
                if 2 <= qk_tick_inline1430__idx_v0:
                    pv_sb_inline1459__ssa_v0: pl.Scalar[pl.INDEX] = qk_tick_inline1430__idx_v0 - 2
                    t__tile_1: pl.Scalar[pl.INT32] = pl.tensor.read(valid_block_mask_inline1376__ssa_v0, [qk_t_inline1421__idx_v0, pv_sb_inline1459__ssa_v0])
                    if 0 < pl.cast(t__tile_1, pl.INDEX):
                        pv_slot_inline1460__ssa_v0: pl.Scalar[pl.INDEX] = qk_core_inline1375__ssa_v0 * 3 + pv_sb_inline1459__ssa_v0 % 3
                        pv_transfer_row_inline1461__ssa_v0: pl.Scalar[pl.INDEX] = pv_slot_inline1460__ssa_v0 * 64
                        pl.system.sync_wait(2, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIC)
                        pv_probability_inline1499__ssa_v0: pl.Tile[[64, 128], pl.BF16, pl.MemRef(mem_mat_30, pl.const(458752, pl.INT64), 16384), pl.Mem.Mat] = pl.tile.load(
                            probability_transfer_inline1394__ssa_v0, [pv_transfer_row_inline1461__ssa_v0, 0], [64, 128], [64, 128], target_memory=pl.Mem.Mat
                        )
                        pv_l1_row_inline1425__ssa_v0: pl.Scalar[pl.INDEX] = pv_sb_inline1459__ssa_v0 % 3 * 128
                        pv_output_inline1403__ssa_v0_l0_init: pl.Tile[[64, 512], pl.FP32, pl.MemRef(mem_acc_31, pl.const(0, pl.INT64), 131072), pl.Mem.Acc] = pl.tile.create(
                            [64, 512], dtype=pl.FP32, target_memory=pl.Mem.Acc
                        )
                        for pv_output_inline1403__ssa_v0_l0_ko, (pv_output_inline1403__ssa_v0_l0_c,) in pl.range(0, 128, 64, init_values=(pv_output_inline1403__ssa_v0_l0_init,)):
                            pv_output_inline1403__ssa_v0_l0_a: pl.Tile[
                                [64, 32], pl.BF16, pl.MemRef(mem_left_23, pl.const(0, pl.INT64), 4096), pl.Mem.Left, pl.TileView(blayout=pl.TileLayout.row_major)
                            ] = pl.tile.extract(pv_probability_inline1499__ssa_v0, 0, pv_output_inline1403__ssa_v0_l0_ko, [64, 32], target_memory=pl.Mem.Left)
                            pv_output_inline1403__ssa_v0_l0_b: pl.Tile[[32, 512], pl.BF16, pl.MemRef(mem_right_24, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                                qk_l1_inline1448__phi_v5, pv_output_inline1403__ssa_v0_l0_ko + pv_l1_row_inline1425__ssa_v0, 0, [32, 512], target_memory=pl.Mem.Right
                            )
                            pv_output_inline1403__ssa_v0_l0_a_1: pl.Tile[
                                [64, 32], pl.BF16, pl.MemRef(mem_left_25, pl.const(16384, pl.INT64), 4096), pl.Mem.Left, pl.TileView(blayout=pl.TileLayout.row_major)
                            ] = pl.tile.extract(pv_probability_inline1499__ssa_v0, 0, pv_output_inline1403__ssa_v0_l0_ko + 32, [64, 32], target_memory=pl.Mem.Left)
                            pv_output_inline1403__ssa_v0_l0_b_1: pl.Tile[[32, 512], pl.BF16, pl.MemRef(mem_right_26, pl.const(32768, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                                qk_l1_inline1448__phi_v5, pv_output_inline1403__ssa_v0_l0_ko + pv_l1_row_inline1425__ssa_v0 + 32, 0, [32, 512], target_memory=pl.Mem.Right
                            )
                            pv_output_inline1403__ssa_v0_l0_c_acc: pl.Tile[[64, 512], pl.FP32, pl.MemRef(mem_acc_31, pl.const(0, pl.INT64), 131072), pl.Mem.Acc] = pl.tile.matmul_acc(
                                pv_output_inline1403__ssa_v0_l0_c, pv_output_inline1403__ssa_v0_l0_a, pv_output_inline1403__ssa_v0_l0_b, pv_output_inline1403__ssa_v0_l0_ko == 0
                            )
                            pv_output_inline1403__ssa_v0_l0_c_acc_1: pl.Tile[[64, 512], pl.FP32, pl.MemRef(mem_acc_31, pl.const(0, pl.INT64), 131072), pl.Mem.Acc] = pl.tile.matmul_acc(
                                pv_output_inline1403__ssa_v0_l0_c_acc, pv_output_inline1403__ssa_v0_l0_a_1, pv_output_inline1403__ssa_v0_l0_b_1, pv_output_inline1403__ssa_v0_l0_ko == -32
                            )
                            pv_output_inline1403__ssa_v0: pl.Tile[[64, 512], pl.FP32, pl.MemRef(mem_acc_31, pl.const(0, pl.INT64), 131072), pl.Mem.Acc] = pl.yield_(
                                pv_output_inline1403__ssa_v0_l0_c_acc_1
                            )
                        pv_transfer_inline1381__store: pl.Tensor[[4608, 512], pl.FP32, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 9437184)] = pl.tile.store(
                            pv_output_inline1403__ssa_v0, [pv_transfer_row_inline1461__ssa_v0, 0], pv_transfer_inline1381__ssa_v0
                        )
                        pl.system.sync_set(3, pipe=pl.PipeType.FIX, ffts_mode=2, core_type=pl.KernelType.AIC)
                qk_l1_inline1448__rv_v2: pl.Tile[[384, 512], pl.BF16, pl.MemRef(mem_mat_21, pl.const(65536, pl.INT64), 393216), pl.Mem.Mat] = pl.yield_(qk_l1_inline1448__phi_v5)
            pl.system.set_ffts(ffts_workspace_inline1401__ssa_v0)

    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def qk_pv_aiv(
        ffts_workspace_inline1401__ssa_v0: pl.Tensor[[256], pl.INT64, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 2048)],
        t_dim_inline1418__ssa_v0: pl.Scalar[pl.INDEX],
        q_flat_inline1409__ssa_v0: pl.Tensor[[t_heads_inline1477__ssa_v0, 512], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)],
        valid_block_mask_inline1376__ssa_v0: pl.Tensor[[t_dim_inline1418__ssa_v0, 16], pl.INT32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)],
        kv_transfer_inline1392__ssa_v0: pl.InOut[pl.Tensor[[9216, 512], pl.BF16, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 9437184)]],
        score_transfer_inline1382__ssa_v0: pl.InOut[pl.Tensor[[4608, 128], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 2359296)]],
        probability_transfer_inline1394__ssa_v0: pl.InOut[pl.Tensor[[4608, 128], pl.BF16, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 1179648)]],
        pv_transfer_inline1381__ssa_v0: pl.InOut[pl.Tensor[[4608, 512], pl.FP32, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 9437184)]],
        attn_sink_col_inline1396__ssa_v0: pl.Tensor[[64, 1], pl.FP32, pl.MemRef("mem_ddr_7", pl.const(0, pl.INT64), 256)],
        positions__ssa_v0: pl.Tensor[[T_DYN, 1], pl.INT32, pl.MemRef("mem_ddr_8", pl.const(0, pl.INT64), 0)],
        window_indices__ssa_v0: pl.Tensor[[T_DYN, 128], pl.INT32, pl.MemRef("mem_ddr_9", pl.const(0, pl.INT64), 0)],
        ori_kv_flat_inline1480__ssa_v1: pl.Tensor[[ori_block_num_inline1408__ssa_v0 * pl.const(32, pl.INDEX), 512], pl.BF16, pl.MemRef("mem_ddr_10", pl.const(0, pl.INT64), 0)],
        cmp_sparse_indices_inline1373__rv_v2: pl.Tensor[[t_dim_inline1418__ssa_v0, 512], pl.INT32, pl.MemRef("mem_ddr_11", pl.const(0, pl.INT64), 0)],
        cmp_table__ssa_v0: pl.Tensor[[B_DYN, COMPRESSED_TABLE_COLUMNS_DYN], pl.INT32, pl.MemRef("mem_ddr_12", pl.const(0, pl.INT64), 0)],
        cmp_kv_flat_inline1399__ssa_v0: pl.Tensor[[cmp_block_num_inline1456__ssa_v0 * pl.const(32, pl.INDEX), 512], pl.BF16, pl.MemRef("mem_ddr_13", pl.const(0, pl.INT64), 0)],
        sparse_bias_inline1413__rv_v2: pl.Tensor[[t_dim_inline1418__ssa_v0, 640], pl.FP32, pl.MemRef("mem_ddr_14", pl.const(0, pl.INT64), 0)],
        mi_transfer_inline1379__ssa_v0: pl.InOut[pl.Tensor[[4608, 1], pl.FP32, pl.MemRef("mem_ddr_15", pl.const(0, pl.INT64), 18432)]],
        li_transfer_inline1432__ssa_v0: pl.InOut[pl.Tensor[[4608, 1], pl.FP32, pl.MemRef("mem_ddr_16", pl.const(0, pl.INT64), 18432)]],
        attn_mi_inline1435__ssa_v0: pl.Out[pl.Tensor[[t_heads_inline1477__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_17", pl.const(0, pl.INT64), 0)]],
        attn_li_inline1483__ssa_v0: pl.Out[pl.Tensor[[t_heads_inline1477__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_18", pl.const(0, pl.INT64), 0)]],
        attn_oi_inline1393__ssa_v0: pl.Out[pl.Tensor[[t_heads_inline1477__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_19", pl.const(0, pl.INT64), 0)]],
    ) -> tuple[
        pl.Tensor[[9216, 512], pl.BF16],
        pl.Tensor[[4608, 128], pl.FP32],
        pl.Tensor[[4608, 128], pl.BF16],
        pl.Tensor[[4608, 512], pl.FP32],
        pl.Tensor[[4608, 1], pl.FP32],
        pl.Tensor[[4608, 1], pl.FP32],
        pl.Tensor[[t_heads_inline1477__ssa_v0, 1], pl.FP32],
        pl.Tensor[[t_heads_inline1477__ssa_v0, 1], pl.FP32],
        pl.Tensor[[t_heads_inline1477__ssa_v0, 512], pl.FP32],
    ]:
        pl.func_attr({"mx_tensor_views_blocked": True, "split_aiv": True, "dual_aiv_dispatch": True})
        mem_vec_20: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
        mem_vec_21: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 128)
        mem_vec_22: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 128)
        mem_vec_23: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32768)
        mem_vec_24: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32768)
        mem_vec_25: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 65536)
        mem_vec_47: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 128)
        mem_vec_52: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 128)
        mem_vec_55: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 128)
        mem_vec_58: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 128)
        mem_vec_60: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32768)
        qk_core_inline1375__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        pl.system.set_ffts(ffts_workspace_inline1401__ssa_v0)
        for qk_t_inline1421__idx_v0 in pl.range(qk_core_inline1375__ssa_v0, t_dim_inline1418__ssa_v0, 24):
            qk_b_inline1374__ssa_v0: pl.Scalar[pl.INDEX] = qk_t_inline1421__idx_v0 // 6
            qk_aiv_inline1464__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_subblock_idx()
            pl.system.set_ffts(ffts_workspace_inline1401__ssa_v0)
            qk_lane_head_inline1395__ssa_v0: pl.Scalar[pl.INDEX] = qk_aiv_inline1464__ssa_v0 * 32
            qk_lane_kv_inline1465__ssa_v0: pl.Scalar[pl.INDEX] = qk_aiv_inline1464__ssa_v0 * 64
            qk_reduce_tmp_inline1469__ssa_v0: pl.Tile[[32, 128], pl.FP32, pl.MemRef(mem_vec_20, pl.const(0, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.create(
                [32, 128], dtype=pl.FP32, target_memory=pl.Mem.Vec
            )
            running_m_inline1436__ssa_v0: pl.Tile[[32, 1], pl.FP32, pl.MemRef(mem_vec_21, pl.const(16384, pl.INT64), 128), pl.Mem.Vec] = pl.tile.load(
                attn_sink_col_inline1396__ssa_v0, [qk_lane_head_inline1395__ssa_v0, 0], [32, 1], [32, 1], target_memory=pl.Mem.Vec
            )
            running_l_inline1387__rm_a0_tmp_v0: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_21, pl.const(16384, pl.INT64), 128), pl.Mem.Vec] = pl.tile.reshape(running_m_inline1436__ssa_v0, [1, 32])
            running_l_inline1387__row_major_tmp_v1: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_22, pl.const(16512, pl.INT64), 128), pl.Mem.Vec] = pl.tile.muls(running_l_inline1387__rm_a0_tmp_v0, 0.0)
            running_l_inline1387__ssa_v0: pl.Tile[[32, 1], pl.FP32, pl.MemRef(mem_vec_22, pl.const(16512, pl.INT64), 128), pl.Mem.Vec] = pl.tile.reshape(
                running_l_inline1387__row_major_tmp_v1, [32, 1]
            )
            running_left_inline1471__ssa_v0: pl.Tile[[32, 256], pl.FP32, pl.MemRef(mem_vec_23, pl.const(16640, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.full([32, 256], dtype=pl.FP32, value=0.0)
            running_right_inline1390__ssa_v0: pl.Tile[[32, 256], pl.FP32, pl.MemRef(mem_vec_24, pl.const(49408, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.full([32, 256], dtype=pl.FP32, value=0.0)
            for qk_tick_inline1412__idx_v0, (m_iter, l_iter, left_iter, right_iter) in pl.range(
                8, init_values=(running_m_inline1436__ssa_v0, running_l_inline1387__ssa_v0, running_left_inline1471__ssa_v0, running_right_inline1390__ssa_v0)
            ):
                if qk_tick_inline1412__idx_v0 < 5:
                    qk_sb_inline1443__ssa_v1: pl.Scalar[pl.INDEX] = qk_tick_inline1412__idx_v0
                    t__tile: pl.Scalar[pl.INT32] = pl.tensor.read(valid_block_mask_inline1376__ssa_v0, [qk_t_inline1421__idx_v0, qk_sb_inline1443__ssa_v1])
                    if 0 < pl.cast(t__tile, pl.INDEX):
                        qk_slot_inline1369__ssa_v1: pl.Scalar[pl.INDEX] = qk_core_inline1375__ssa_v0 * 3 + qk_sb_inline1443__ssa_v1 % 3
                        qk_kv_row_inline1380__ssa_v1: pl.Scalar[pl.INDEX] = qk_slot_inline1369__ssa_v1 * 128
                        qk_s0_inline1474__ssa_v0: pl.Scalar[pl.INDEX] = qk_sb_inline1443__ssa_v1 * 128
                        qk_kv_half_inline1416__ssa_v0: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.tile.full(
                            [64, 512], dtype=pl.BF16, value=0.0
                        )
                        if qk_s0_inline1474__ssa_v0 < 128:
                            t__tile_1: pl.Scalar[pl.INT32] = pl.tensor.read(positions__ssa_v0, [qk_t_inline1421__idx_v0, 0])
                            qk_pos_inline1476__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(t__tile_1, pl.INDEX)
                            qk_win_len_inline1466__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(qk_pos_inline1476__ssa_v0 + 1, 128)
                            qk_win_start_inline1479__ssa_v0: pl.Scalar[pl.INDEX] = qk_pos_inline1476__ssa_v0 - qk_win_len_inline1466__ssa_v0 + 1
                            qk_head_inline1391__ssa_v0: pl.Scalar[pl.INDEX] = (qk_win_start_inline1479__ssa_v0 + qk_s0_inline1474__ssa_v0) % 32
                            qk_rows_inline1481__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(pl.max(qk_win_len_inline1466__ssa_v0 - qk_s0_inline1474__ssa_v0 - qk_lane_kv_inline1465__ssa_v0, 0), 64)
                            qk_lo_inline1491__ssa_v0: pl.Scalar[pl.INDEX] = 0
                            qk_hi_inline1414__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(32 - qk_head_inline1391__ssa_v0 - qk_lane_kv_inline1465__ssa_v0, qk_rows_inline1481__ssa_v0)
                            if qk_lo_inline1491__ssa_v0 < qk_hi_inline1414__ssa_v0:
                                qk_raw_row_inline1442__tile: pl.Scalar[pl.INT32] = pl.tensor.read(
                                    window_indices__ssa_v0, [qk_t_inline1421__idx_v0, qk_s0_inline1474__ssa_v0 + qk_lane_kv_inline1465__ssa_v0 + qk_lo_inline1491__ssa_v0]
                                )
                                if 0 <= pl.cast(qk_raw_row_inline1442__tile, pl.INDEX):
                                    qk_kv_half_inline1416__ssa_v1: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.tile.gather_row(
                                        qk_kv_half_inline1416__ssa_v0,
                                        ori_kv_flat_inline1480__ssa_v1,
                                        [qk_lo_inline1491__ssa_v0, 0],
                                        [qk_raw_row_inline1442__tile, 0],
                                        [64, 512],
                                        valid_shape=[qk_hi_inline1414__ssa_v0 - qk_lo_inline1491__ssa_v0, 512],
                                        transpose=False,
                                    )
                                    qk_kv_half_inline1416__phi_v2: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(
                                        qk_kv_half_inline1416__ssa_v1
                                    )
                                else:
                                    qk_kv_half_inline1416__phi_v2: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(
                                        qk_kv_half_inline1416__ssa_v0
                                    )
                                qk_kv_half_inline1416__phi_v3: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(
                                    qk_kv_half_inline1416__phi_v2
                                )
                            else:
                                qk_kv_half_inline1416__phi_v3: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(
                                    qk_kv_half_inline1416__ssa_v0
                                )
                            qk_lo_inline1491__ssa_v1: pl.Scalar[pl.INDEX] = pl.max(32 - qk_head_inline1391__ssa_v0 - qk_lane_kv_inline1465__ssa_v0, 0)
                            qk_hi_inline1414__ssa_v1: pl.Scalar[pl.INDEX] = pl.min(64 - qk_head_inline1391__ssa_v0 - qk_lane_kv_inline1465__ssa_v0, qk_rows_inline1481__ssa_v0)
                            if qk_lo_inline1491__ssa_v1 < qk_hi_inline1414__ssa_v1:
                                qk_raw_row_inline1442__tile_1: pl.Scalar[pl.INT32] = pl.tensor.read(
                                    window_indices__ssa_v0, [qk_t_inline1421__idx_v0, qk_s0_inline1474__ssa_v0 + qk_lane_kv_inline1465__ssa_v0 + qk_lo_inline1491__ssa_v1]
                                )
                                if 0 <= pl.cast(qk_raw_row_inline1442__tile_1, pl.INDEX):
                                    qk_kv_half_inline1416__ssa_v4: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.tile.gather_row(
                                        qk_kv_half_inline1416__phi_v3,
                                        ori_kv_flat_inline1480__ssa_v1,
                                        [qk_lo_inline1491__ssa_v1, 0],
                                        [qk_raw_row_inline1442__tile_1, 0],
                                        [64, 512],
                                        valid_shape=[qk_hi_inline1414__ssa_v1 - qk_lo_inline1491__ssa_v1, 512],
                                        transpose=False,
                                    )
                                    qk_kv_half_inline1416__phi_v5: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(
                                        qk_kv_half_inline1416__ssa_v4
                                    )
                                else:
                                    qk_kv_half_inline1416__phi_v5: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(
                                        qk_kv_half_inline1416__phi_v3
                                    )
                                qk_kv_half_inline1416__phi_v6: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(
                                    qk_kv_half_inline1416__phi_v5
                                )
                            else:
                                qk_kv_half_inline1416__phi_v6: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(
                                    qk_kv_half_inline1416__phi_v3
                                )
                            qk_lo_inline1491__ssa_v2: pl.Scalar[pl.INDEX] = pl.max(64 - qk_head_inline1391__ssa_v0 - qk_lane_kv_inline1465__ssa_v0, 0)
                            qk_hi_inline1414__ssa_v2: pl.Scalar[pl.INDEX] = pl.min(96 - qk_head_inline1391__ssa_v0 - qk_lane_kv_inline1465__ssa_v0, qk_rows_inline1481__ssa_v0)
                            if qk_lo_inline1491__ssa_v2 < qk_hi_inline1414__ssa_v2:
                                qk_raw_row_inline1442__tile_2: pl.Scalar[pl.INT32] = pl.tensor.read(
                                    window_indices__ssa_v0, [qk_t_inline1421__idx_v0, qk_s0_inline1474__ssa_v0 + qk_lane_kv_inline1465__ssa_v0 + qk_lo_inline1491__ssa_v2]
                                )
                                if 0 <= pl.cast(qk_raw_row_inline1442__tile_2, pl.INDEX):
                                    qk_kv_half_inline1416__ssa_v7: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.tile.gather_row(
                                        qk_kv_half_inline1416__phi_v6,
                                        ori_kv_flat_inline1480__ssa_v1,
                                        [qk_lo_inline1491__ssa_v2, 0],
                                        [qk_raw_row_inline1442__tile_2, 0],
                                        [64, 512],
                                        valid_shape=[qk_hi_inline1414__ssa_v2 - qk_lo_inline1491__ssa_v2, 512],
                                        transpose=False,
                                    )
                                    qk_kv_half_inline1416__phi_v8: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(
                                        qk_kv_half_inline1416__ssa_v7
                                    )
                                else:
                                    qk_kv_half_inline1416__phi_v8: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(
                                        qk_kv_half_inline1416__phi_v6
                                    )
                                qk_kv_half_inline1416__phi_v9: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(
                                    qk_kv_half_inline1416__phi_v8
                                )
                            else:
                                qk_kv_half_inline1416__phi_v9: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(
                                    qk_kv_half_inline1416__phi_v6
                                )
                            qk_lo_inline1491__ssa_v3: pl.Scalar[pl.INDEX] = pl.max(96 - qk_head_inline1391__ssa_v0 - qk_lane_kv_inline1465__ssa_v0, 0)
                            qk_hi_inline1414__ssa_v3: pl.Scalar[pl.INDEX] = pl.min(128 - qk_head_inline1391__ssa_v0 - qk_lane_kv_inline1465__ssa_v0, qk_rows_inline1481__ssa_v0)
                            if qk_lo_inline1491__ssa_v3 < qk_hi_inline1414__ssa_v3:
                                qk_raw_row_inline1442__tile_3: pl.Scalar[pl.INT32] = pl.tensor.read(
                                    window_indices__ssa_v0, [qk_t_inline1421__idx_v0, qk_s0_inline1474__ssa_v0 + qk_lane_kv_inline1465__ssa_v0 + qk_lo_inline1491__ssa_v3]
                                )
                                if 0 <= pl.cast(qk_raw_row_inline1442__tile_3, pl.INDEX):
                                    qk_kv_half_inline1416__ssa_v10: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.tile.gather_row(
                                        qk_kv_half_inline1416__phi_v9,
                                        ori_kv_flat_inline1480__ssa_v1,
                                        [qk_lo_inline1491__ssa_v3, 0],
                                        [qk_raw_row_inline1442__tile_3, 0],
                                        [64, 512],
                                        valid_shape=[qk_hi_inline1414__ssa_v3 - qk_lo_inline1491__ssa_v3, 512],
                                        transpose=False,
                                    )
                                    qk_kv_half_inline1416__phi_v11: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(
                                        qk_kv_half_inline1416__ssa_v10
                                    )
                                else:
                                    qk_kv_half_inline1416__phi_v11: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(
                                        qk_kv_half_inline1416__phi_v9
                                    )
                                qk_kv_half_inline1416__phi_v12: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(
                                    qk_kv_half_inline1416__phi_v11
                                )
                            else:
                                qk_kv_half_inline1416__phi_v12: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(
                                    qk_kv_half_inline1416__phi_v9
                                )
                            qk_lo_inline1491__ssa_v4: pl.Scalar[pl.INDEX] = pl.max(128 - qk_head_inline1391__ssa_v0 - qk_lane_kv_inline1465__ssa_v0, 0)
                            qk_hi_inline1414__ssa_v4: pl.Scalar[pl.INDEX] = pl.min(160 - qk_head_inline1391__ssa_v0 - qk_lane_kv_inline1465__ssa_v0, qk_rows_inline1481__ssa_v0)
                            if qk_lo_inline1491__ssa_v4 < qk_hi_inline1414__ssa_v4:
                                qk_raw_row_inline1442__tile_4: pl.Scalar[pl.INT32] = pl.tensor.read(
                                    window_indices__ssa_v0, [qk_t_inline1421__idx_v0, qk_s0_inline1474__ssa_v0 + qk_lane_kv_inline1465__ssa_v0 + qk_lo_inline1491__ssa_v4]
                                )
                                if 0 <= pl.cast(qk_raw_row_inline1442__tile_4, pl.INDEX):
                                    qk_kv_half_inline1416__ssa_v13: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.tile.gather_row(
                                        qk_kv_half_inline1416__phi_v12,
                                        ori_kv_flat_inline1480__ssa_v1,
                                        [qk_lo_inline1491__ssa_v4, 0],
                                        [qk_raw_row_inline1442__tile_4, 0],
                                        [64, 512],
                                        valid_shape=[qk_hi_inline1414__ssa_v4 - qk_lo_inline1491__ssa_v4, 512],
                                        transpose=False,
                                    )
                                    qk_kv_half_inline1416__phi_v14: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(
                                        qk_kv_half_inline1416__ssa_v13
                                    )
                                else:
                                    qk_kv_half_inline1416__phi_v14: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(
                                        qk_kv_half_inline1416__phi_v12
                                    )
                                qk_kv_half_inline1416__phi_v15: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(
                                    qk_kv_half_inline1416__phi_v14
                                )
                            else:
                                qk_kv_half_inline1416__phi_v15: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(
                                    qk_kv_half_inline1416__phi_v12
                                )
                            qk_kv_half_inline1416__phi_v21: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(qk_kv_half_inline1416__phi_v15)
                        else:
                            qk_kv_half_inline1416__ssa_v0: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.tile.full(
                                [64, 512], dtype=pl.BF16, value=0.0
                            )
                            for qk_row_inline1485__idx_v0, (qk_kv_half_inline1416__iter_v16,) in pl.range(64, init_values=(qk_kv_half_inline1416__ssa_v0,)):
                                qk_cmp_k_inline1486__ssa_v0: pl.Scalar[pl.INDEX] = qk_s0_inline1474__ssa_v0 + qk_lane_kv_inline1465__ssa_v0 + qk_row_inline1485__idx_v0 - 128
                                if qk_cmp_k_inline1486__ssa_v0 < 512:
                                    qk_ridx_inline1444__tile: pl.Scalar[pl.INT32] = pl.tensor.read(cmp_sparse_indices_inline1373__rv_v2, [qk_t_inline1421__idx_v0, qk_cmp_k_inline1486__ssa_v0])
                                    if 0 <= pl.cast(qk_ridx_inline1444__tile, pl.INDEX):
                                        t__tile_2: pl.Scalar[pl.INT32] = pl.tensor.read(cmp_table__ssa_v0, [qk_b_inline1374__ssa_v0, pl.cast(qk_ridx_inline1444__tile, pl.INDEX) // 32])
                                        qk_page_inline1467__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(t__tile_2, pl.INDEX)
                                        qk_src_inline1489__ssa_v0: pl.Scalar[pl.INDEX] = qk_page_inline1467__ssa_v0 * 32 + pl.cast(qk_ridx_inline1444__tile, pl.INDEX) % 32
                                        qk_kv_half_inline1416__ssa_v18: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.tile.gather_row(
                                            qk_kv_half_inline1416__iter_v16, cmp_kv_flat_inline1399__ssa_v0, [qk_row_inline1485__idx_v0, 0], [qk_src_inline1489__ssa_v0, 0], [1, 512], transpose=False
                                        )
                                        qk_kv_half_inline1416__phi_v19: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(
                                            qk_kv_half_inline1416__ssa_v18
                                        )
                                    else:
                                        qk_kv_half_inline1416__phi_v19: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(
                                            qk_kv_half_inline1416__iter_v16
                                        )
                                    qk_kv_half_inline1416__phi_v20: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(
                                        qk_kv_half_inline1416__phi_v19
                                    )
                                else:
                                    qk_kv_half_inline1416__phi_v20: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(
                                        qk_kv_half_inline1416__iter_v16
                                    )
                                qk_kv_half_inline1416__rv_v17: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(
                                    qk_kv_half_inline1416__phi_v20
                                )
                            qk_kv_half_inline1416__phi_v21: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(qk_kv_half_inline1416__rv_v17)
                        kv_transfer_inline1392__store: pl.Tensor[[9216, 512], pl.BF16, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 9437184)] = pl.tile.store(
                            qk_kv_half_inline1416__phi_v21, [qk_kv_row_inline1380__ssa_v1 + qk_lane_kv_inline1465__ssa_v0, 0], kv_transfer_inline1392__ssa_v0
                        )
                if 0 < qk_tick_inline1412__idx_v0 and qk_tick_inline1412__idx_v0 <= 5:
                    softmax_sb_inline1490__ssa_v0: pl.Scalar[pl.INDEX] = qk_tick_inline1412__idx_v0 - 1
                    t__tile_3: pl.Scalar[pl.INT32] = pl.tensor.read(valid_block_mask_inline1376__ssa_v0, [qk_t_inline1421__idx_v0, softmax_sb_inline1490__ssa_v0])
                    if 0 < pl.cast(t__tile_3, pl.INDEX):
                        qk_slot_inline1369__ssa_v2: pl.Scalar[pl.INDEX] = qk_core_inline1375__ssa_v0 * 3 + softmax_sb_inline1490__ssa_v0 % 3
                        qk_transfer_row_inline1440__ssa_v2: pl.Scalar[pl.INDEX] = qk_slot_inline1369__ssa_v2 * 64
                        qk_s0_v1_inline1468__ssa_v0: pl.Scalar[pl.INDEX] = softmax_sb_inline1490__ssa_v0 * 128
                        pl.system.sync_wait(1, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIV)
                        qk_scores_half_inline1492__ssa_v0: pl.Tile[[32, 128], pl.FP32, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.load(
                            score_transfer_inline1382__ssa_v0, [qk_transfer_row_inline1440__ssa_v2 + qk_lane_head_inline1395__ssa_v0, 0], [32, 128], [32, 128], target_memory=pl.Mem.Vec
                        )
                        qk_bias_inline1451__ssa_v0: pl.Tile[[1, 128], pl.FP32, pl.MemRef(mem_vec_60, pl.const(148224, pl.INT64), 512), pl.Mem.Vec] = pl.tile.load(
                            sparse_bias_inline1413__rv_v2, [qk_t_inline1421__idx_v0, qk_s0_v1_inline1468__ssa_v0], [1, 128], [1, 128], target_memory=pl.Mem.Vec
                        )
                        qk_scaled_inline1493__ssa_v0: pl.Tile[[32, 128], pl.FP32, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.muls(
                            qk_scores_half_inline1492__ssa_v0, 0.044194173824159223
                        )
                        qk_masked_inline1495__ssa_v0: pl.Tile[[32, 128], pl.FP32, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.col_expand_add(
                            qk_scaled_inline1493__ssa_v0, qk_bias_inline1451__ssa_v0
                        )
                        qk_mi_inline1494__ssa_v0: pl.Tile[[32, 1], pl.FP32, pl.MemRef(mem_vec_60, pl.const(148224, pl.INT64), 128), pl.Mem.Vec] = pl.tile.row_max(
                            qk_masked_inline1495__ssa_v0, qk_reduce_tmp_inline1469__ssa_v0
                        )
                        t__tmp_v28: pl.Tile[[32, 128], pl.FP32, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.row_expand_sub(
                            qk_masked_inline1495__ssa_v0, qk_mi_inline1494__ssa_v0
                        )
                        qk_exp_inline1496__ssa_v0: pl.Tile[[32, 128], pl.FP32, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.exp(t__tmp_v28)
                        qk_li_inline1500__ssa_v0: pl.Tile[[32, 1], pl.FP32, pl.MemRef(mem_vec_47, pl.const(147712, pl.INT64), 128), pl.Mem.Vec] = pl.tile.row_sum(
                            qk_exp_inline1496__ssa_v0, qk_reduce_tmp_inline1469__ssa_v0
                        )
                        qk_probability_inline1498__ssa_v0: pl.Tile[[32, 128], pl.BF16, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.cast(
                            qk_exp_inline1496__ssa_v0, target_type=pl.BF16, mode="rint"
                        )
                        probability_transfer_inline1394__store: pl.Tensor[[4608, 128], pl.BF16, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 1179648)] = pl.tile.store(
                            qk_probability_inline1498__ssa_v0, [qk_transfer_row_inline1440__ssa_v2 + qk_lane_head_inline1395__ssa_v0, 0], probability_transfer_inline1394__ssa_v0
                        )
                        mi_transfer_inline1379__store: pl.Tensor[[4608, 1], pl.FP32, pl.MemRef("mem_ddr_15", pl.const(0, pl.INT64), 18432)] = pl.tile.store(
                            qk_mi_inline1494__ssa_v0, [qk_transfer_row_inline1440__ssa_v2 + qk_lane_head_inline1395__ssa_v0, 0], mi_transfer_inline1379__ssa_v0
                        )
                        li_transfer_inline1432__store: pl.Tensor[[4608, 1], pl.FP32, pl.MemRef("mem_ddr_16", pl.const(0, pl.INT64), 18432)] = pl.tile.store(
                            qk_li_inline1500__ssa_v0, [qk_transfer_row_inline1440__ssa_v2 + qk_lane_head_inline1395__ssa_v0, 0], li_transfer_inline1432__ssa_v0
                        )
                        pl.system.sync_set(2, pipe=pl.PipeType.MTE3, ffts_mode=2, core_type=pl.KernelType.AIV)
                if qk_tick_inline1412__idx_v0 < 5:
                    t__tile_4: pl.Scalar[pl.INT32] = pl.tensor.read(valid_block_mask_inline1376__ssa_v0, [qk_t_inline1421__idx_v0, qk_tick_inline1412__idx_v0])
                    if 0 < pl.cast(t__tile_4, pl.INDEX):
                        pl.system.sync_set(0, pipe=pl.PipeType.MTE3, ffts_mode=2, core_type=pl.KernelType.AIV)
                if 3 <= qk_tick_inline1412__idx_v0:
                    pv_sb_inline1459__ssa_v1: pl.Scalar[pl.INDEX] = qk_tick_inline1412__idx_v0 - 2 - 1
                    t__tile_5: pl.Scalar[pl.INT32] = pl.tensor.read(valid_block_mask_inline1376__ssa_v0, [qk_t_inline1421__idx_v0, pv_sb_inline1459__ssa_v1])
                    if 0 < pl.cast(t__tile_5, pl.INDEX):
                        pv_slot_inline1460__ssa_v1: pl.Scalar[pl.INDEX] = qk_core_inline1375__ssa_v0 * 3 + pv_sb_inline1459__ssa_v1 % 3
                        pv_transfer_row_inline1461__ssa_v1: pl.Scalar[pl.INDEX] = pv_slot_inline1460__ssa_v1 * 64
                        pl.system.sync_wait(3, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIV)
                        pv_m_inline1419__ssa_v0: pl.Tile[[32, 1], pl.FP32, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 128), pl.Mem.Vec] = pl.tile.load(
                            mi_transfer_inline1379__ssa_v0, [pv_transfer_row_inline1461__ssa_v1 + qk_lane_head_inline1395__ssa_v0, 0], [32, 1], [32, 1], target_memory=pl.Mem.Vec
                        )
                        pv_l_inline1503__ssa_v0: pl.Tile[[32, 1], pl.FP32, pl.MemRef(mem_vec_60, pl.const(148224, pl.INT64), 128), pl.Mem.Vec] = pl.tile.load(
                            li_transfer_inline1432__ssa_v0, [pv_transfer_row_inline1461__ssa_v1 + qk_lane_head_inline1395__ssa_v0, 0], [32, 1], [32, 1], target_memory=pl.Mem.Vec
                        )
                        next_m_inline1371__rm_a0_tmp_v2: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_21, pl.const(16384, pl.INT64), 128), pl.Mem.Vec] = pl.tile.reshape(m_iter, [1, 32])
                        next_m_inline1371__rm_a1_tmp_v3: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 128), pl.Mem.Vec] = pl.tile.reshape(
                            pv_m_inline1419__ssa_v0, [1, 32]
                        )
                        next_m_inline1371__row_major_tmp_v4: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_47, pl.const(147712, pl.INT64), 128), pl.Mem.Vec] = pl.tile.maximum(
                            next_m_inline1371__rm_a0_tmp_v2, next_m_inline1371__rm_a1_tmp_v3
                        )
                        next_m_inline1371__ssa_v0: pl.Tile[[32, 1], pl.FP32, pl.MemRef(mem_vec_47, pl.const(147712, pl.INT64), 128), pl.Mem.Vec] = pl.tile.reshape(
                            next_m_inline1371__row_major_tmp_v4, [32, 1]
                        )
                        t__rm_a0_tmp_v5: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_21, pl.const(16384, pl.INT64), 128), pl.Mem.Vec] = pl.tile.reshape(m_iter, [1, 32])
                        t__rm_a1_tmp_v6: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_47, pl.const(147712, pl.INT64), 128), pl.Mem.Vec] = pl.tile.reshape(next_m_inline1371__ssa_v0, [1, 32])
                        t__row_major_tmp_v7: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_52, pl.const(147840, pl.INT64), 128), pl.Mem.Vec] = pl.tile.sub(t__rm_a0_tmp_v5, t__rm_a1_tmp_v6)
                        t__tmp_v31: pl.Tile[[32, 1], pl.FP32, pl.MemRef(mem_vec_52, pl.const(147840, pl.INT64), 128), pl.Mem.Vec] = pl.tile.reshape(t__row_major_tmp_v7, [32, 1])
                        alpha_inline1505__rm_a0_tmp_v8: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_52, pl.const(147840, pl.INT64), 128), pl.Mem.Vec] = pl.tile.reshape(t__tmp_v31, [1, 32])
                        alpha_inline1505__row_major_tmp_v9: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_52, pl.const(147840, pl.INT64), 128), pl.Mem.Vec] = pl.tile.exp(alpha_inline1505__rm_a0_tmp_v8)
                        alpha_inline1505__ssa_v0: pl.Tile[[32, 1], pl.FP32, pl.MemRef(mem_vec_52, pl.const(147840, pl.INT64), 128), pl.Mem.Vec] = pl.tile.reshape(
                            alpha_inline1505__row_major_tmp_v9, [32, 1]
                        )
                        t__rm_a0_tmp_v10: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 128), pl.Mem.Vec] = pl.tile.reshape(pv_m_inline1419__ssa_v0, [1, 32])
                        t__rm_a1_tmp_v11: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_47, pl.const(147712, pl.INT64), 128), pl.Mem.Vec] = pl.tile.reshape(next_m_inline1371__ssa_v0, [1, 32])
                        t__row_major_tmp_v12: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 128), pl.Mem.Vec] = pl.tile.sub(t__rm_a0_tmp_v10, t__rm_a1_tmp_v11)
                        t__tmp_v32: pl.Tile[[32, 1], pl.FP32, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 128), pl.Mem.Vec] = pl.tile.reshape(t__row_major_tmp_v12, [32, 1])
                        beta_inline1484__rm_a0_tmp_v13: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 128), pl.Mem.Vec] = pl.tile.reshape(t__tmp_v32, [1, 32])
                        beta_inline1484__row_major_tmp_v14: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_55, pl.const(147968, pl.INT64), 128), pl.Mem.Vec] = pl.tile.exp(beta_inline1484__rm_a0_tmp_v13)
                        beta_inline1484__ssa_v0: pl.Tile[[32, 1], pl.FP32, pl.MemRef(mem_vec_55, pl.const(147968, pl.INT64), 128), pl.Mem.Vec] = pl.tile.reshape(
                            beta_inline1484__row_major_tmp_v14, [32, 1]
                        )
                        t__rm_a0_tmp_v15: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_52, pl.const(147840, pl.INT64), 128), pl.Mem.Vec] = pl.tile.reshape(alpha_inline1505__ssa_v0, [1, 32])
                        t__rm_a1_tmp_v16: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_22, pl.const(16512, pl.INT64), 128), pl.Mem.Vec] = pl.tile.reshape(l_iter, [1, 32])
                        t__row_major_tmp_v17: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 128), pl.Mem.Vec] = pl.tile.mul(t__rm_a0_tmp_v15, t__rm_a1_tmp_v16)
                        t__tmp_v33: pl.Tile[[32, 1], pl.FP32, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 128), pl.Mem.Vec] = pl.tile.reshape(t__row_major_tmp_v17, [32, 1])
                        t__rm_a0_tmp_v18: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_55, pl.const(147968, pl.INT64), 128), pl.Mem.Vec] = pl.tile.reshape(beta_inline1484__ssa_v0, [1, 32])
                        t__rm_a1_tmp_v19: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_60, pl.const(148224, pl.INT64), 128), pl.Mem.Vec] = pl.tile.reshape(pv_l_inline1503__ssa_v0, [1, 32])
                        t__row_major_tmp_v20: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_60, pl.const(148224, pl.INT64), 128), pl.Mem.Vec] = pl.tile.mul(t__rm_a0_tmp_v18, t__rm_a1_tmp_v19)
                        t__tmp_v34: pl.Tile[[32, 1], pl.FP32, pl.MemRef(mem_vec_60, pl.const(148224, pl.INT64), 128), pl.Mem.Vec] = pl.tile.reshape(t__row_major_tmp_v20, [32, 1])
                        next_l_inline1488__rm_a0_tmp_v21: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 128), pl.Mem.Vec] = pl.tile.reshape(t__tmp_v33, [1, 32])
                        next_l_inline1488__rm_a1_tmp_v22: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_60, pl.const(148224, pl.INT64), 128), pl.Mem.Vec] = pl.tile.reshape(t__tmp_v34, [1, 32])
                        next_l_inline1488__row_major_tmp_v23: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_58, pl.const(148096, pl.INT64), 128), pl.Mem.Vec] = pl.tile.add(
                            next_l_inline1488__rm_a0_tmp_v21, next_l_inline1488__rm_a1_tmp_v22
                        )
                        next_l_inline1488__ssa_v0: pl.Tile[[32, 1], pl.FP32, pl.MemRef(mem_vec_58, pl.const(148096, pl.INT64), 128), pl.Mem.Vec] = pl.tile.reshape(
                            next_l_inline1488__row_major_tmp_v23, [32, 1]
                        )
                        pv_left_inline1410__ssa_v0: pl.Tile[[32, 256], pl.FP32, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.load(
                            pv_transfer_inline1381__ssa_v0, [pv_transfer_row_inline1461__ssa_v1 + qk_lane_head_inline1395__ssa_v0, 0], [32, 256], [32, 256], target_memory=pl.Mem.Vec
                        )
                        t__tmp_v35: pl.Tile[[32, 256], pl.FP32, pl.MemRef(mem_vec_60, pl.const(148224, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.row_expand_mul(left_iter, alpha_inline1505__ssa_v0)
                        t__tmp_v36: pl.Tile[[32, 256], pl.FP32, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.row_expand_mul(
                            pv_left_inline1410__ssa_v0, beta_inline1484__ssa_v0
                        )
                        next_left_inline1506__ssa_v0: pl.Tile[[32, 256], pl.FP32, pl.MemRef(mem_vec_23, pl.const(16640, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.add(t__tmp_v35, t__tmp_v36)
                        pv_right_inline1453__ssa_v0: pl.Tile[[32, 256], pl.FP32, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.load(
                            pv_transfer_inline1381__ssa_v0, [pv_transfer_row_inline1461__ssa_v1 + qk_lane_head_inline1395__ssa_v0, 256], [32, 256], [32, 256], target_memory=pl.Mem.Vec
                        )
                        t__tmp_v37: pl.Tile[[32, 256], pl.FP32, pl.MemRef(mem_vec_60, pl.const(148224, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.row_expand_mul(right_iter, alpha_inline1505__ssa_v0)
                        t__tmp_v38: pl.Tile[[32, 256], pl.FP32, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.row_expand_mul(
                            pv_right_inline1453__ssa_v0, beta_inline1484__ssa_v0
                        )
                        next_right_inline1478__ssa_v0: pl.Tile[[32, 256], pl.FP32, pl.MemRef(mem_vec_24, pl.const(49408, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.add(t__tmp_v37, t__tmp_v38)
                        m_valid_inline1482__rv_v0, l_valid_inline1502__rv_v0, left_valid_inline1398__rv_v0, right_valid_inline1415__rv_v0 = pl.yield_(
                            next_m_inline1371__ssa_v0, next_l_inline1488__ssa_v0, next_left_inline1506__ssa_v0, next_right_inline1478__ssa_v0
                        )
                    else:
                        m_iter_mv: pl.Tile[[32, 1], pl.FP32, pl.MemRef(mem_vec_47, pl.const(147712, pl.INT64), 128), pl.Mem.Vec] = pl.tile.move(m_iter, target_memory=pl.Mem.Vec)
                        l_iter_mv: pl.Tile[[32, 1], pl.FP32, pl.MemRef(mem_vec_58, pl.const(148096, pl.INT64), 128), pl.Mem.Vec] = pl.tile.move(l_iter, target_memory=pl.Mem.Vec)
                        m_valid_inline1482__rv_v0, l_valid_inline1502__rv_v0, left_valid_inline1398__rv_v0, right_valid_inline1415__rv_v0 = pl.yield_(m_iter_mv, l_iter_mv, left_iter, right_iter)
                    m_after_inline1445__rv_v0, l_after_inline1501__rv_v0, left_after_inline1504__rv_v0, right_after_inline1429__rv_v0 = pl.yield_(
                        m_valid_inline1482__rv_v0, l_valid_inline1502__rv_v0, left_valid_inline1398__rv_v0, right_valid_inline1415__rv_v0
                    )
                else:
                    m_iter_mv_1: pl.Tile[[32, 1], pl.FP32, pl.MemRef(mem_vec_47, pl.const(147712, pl.INT64), 128), pl.Mem.Vec] = pl.tile.move(m_iter, target_memory=pl.Mem.Vec)
                    l_iter_mv_1: pl.Tile[[32, 1], pl.FP32, pl.MemRef(mem_vec_58, pl.const(148096, pl.INT64), 128), pl.Mem.Vec] = pl.tile.move(l_iter, target_memory=pl.Mem.Vec)
                    m_after_inline1445__rv_v0, l_after_inline1501__rv_v0, left_after_inline1504__rv_v0, right_after_inline1429__rv_v0 = pl.yield_(m_iter_mv_1, l_iter_mv_1, left_iter, right_iter)
                l_after_inline1501__rv_v0_mv: pl.Tile[[32, 1], pl.FP32, pl.MemRef(mem_vec_22, pl.const(16512, pl.INT64), 128), pl.Mem.Vec] = pl.tile.move(
                    l_after_inline1501__rv_v0, target_memory=pl.Mem.Vec
                )
                m_after_inline1445__rv_v0_mv: pl.Tile[[32, 1], pl.FP32, pl.MemRef(mem_vec_21, pl.const(16384, pl.INT64), 128), pl.Mem.Vec] = pl.tile.move(
                    m_after_inline1445__rv_v0, target_memory=pl.Mem.Vec
                )
                running_m_inline1472, running_l_inline1473, running_left_inline1497, running_right_inline1389 = pl.yield_(
                    m_after_inline1445__rv_v0_mv, l_after_inline1501__rv_v0_mv, left_after_inline1504__rv_v0, right_after_inline1429__rv_v0
                )
            qk_output_row_inline1449__ssa_v0: pl.Scalar[pl.INDEX] = qk_t_inline1421__idx_v0 * 64 + qk_lane_head_inline1395__ssa_v0
            attn_mi_inline1435__store: pl.Tensor[[t_heads_inline1477__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_17", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                running_m_inline1472, [qk_output_row_inline1449__ssa_v0, 0], attn_mi_inline1435__ssa_v0
            )
            attn_li_inline1483__store: pl.Tensor[[t_heads_inline1477__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_18", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                running_l_inline1473, [qk_output_row_inline1449__ssa_v0, 0], attn_li_inline1483__ssa_v0
            )
            attn_oi_inline1393__store: pl.Tensor[[t_heads_inline1477__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_19", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                running_left_inline1497, [qk_output_row_inline1449__ssa_v0, 0], attn_oi_inline1393__ssa_v0
            )
            attn_oi_inline1393__store_v0: pl.Tensor[[t_heads_inline1477__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_19", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                running_right_inline1389, [qk_output_row_inline1449__ssa_v0, 256], attn_oi_inline1393__ssa_v0
            )
        return (
            kv_transfer_inline1392__ssa_v0,
            score_transfer_inline1382__ssa_v0,
            probability_transfer_inline1394__ssa_v0,
            pv_transfer_inline1381__ssa_v0,
            mi_transfer_inline1379__ssa_v0,
            li_transfer_inline1432__ssa_v0,
            attn_mi_inline1435__ssa_v0,
            attn_li_inline1483__ssa_v0,
            attn_oi_inline1393__ssa_v0,
        )

    @pl.function(type=pl.FunctionType.Group, level=pl.Level.CORE_GROUP, role=pl.Role.SubWorker)
    def qk_pv(
        ffts_workspace_inline1401__ssa_v0: pl.Tensor[[256], pl.INT64, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 2048)],
        t_dim_inline1418__ssa_v0: pl.Scalar[pl.INDEX],
        q_flat_inline1409__ssa_v0: pl.Tensor[[t_heads_inline1477__ssa_v0, 512], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)],
        valid_block_mask_inline1376__ssa_v0: pl.Tensor[[t_dim_inline1418__ssa_v0, 16], pl.INT32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)],
        kv_transfer_inline1392__ssa_v0: pl.InOut[pl.Tensor[[9216, 512], pl.BF16, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 9437184)]],
        score_transfer_inline1382__ssa_v0: pl.InOut[pl.Tensor[[4608, 128], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 2359296)]],
        probability_transfer_inline1394__ssa_v0: pl.InOut[pl.Tensor[[4608, 128], pl.BF16, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 1179648)]],
        pv_transfer_inline1381__ssa_v0: pl.InOut[pl.Tensor[[4608, 512], pl.FP32, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 9437184)]],
        attn_sink_col_inline1396__ssa_v0: pl.Tensor[[64, 1], pl.FP32, pl.MemRef("mem_ddr_7", pl.const(0, pl.INT64), 256)],
        positions__ssa_v0: pl.Tensor[[T_DYN, 1], pl.INT32, pl.MemRef("mem_ddr_8", pl.const(0, pl.INT64), 0)],
        window_indices__ssa_v0: pl.Tensor[[T_DYN, 128], pl.INT32, pl.MemRef("mem_ddr_9", pl.const(0, pl.INT64), 0)],
        ori_kv_flat_inline1480__ssa_v1: pl.Tensor[[ori_block_num_inline1408__ssa_v0 * pl.const(32, pl.INDEX), 512], pl.BF16, pl.MemRef("mem_ddr_10", pl.const(0, pl.INT64), 0)],
        cmp_sparse_indices_inline1373__rv_v2: pl.Tensor[[t_dim_inline1418__ssa_v0, 512], pl.INT32, pl.MemRef("mem_ddr_11", pl.const(0, pl.INT64), 0)],
        cmp_table__ssa_v0: pl.Tensor[[B_DYN, COMPRESSED_TABLE_COLUMNS_DYN], pl.INT32, pl.MemRef("mem_ddr_12", pl.const(0, pl.INT64), 0)],
        cmp_kv_flat_inline1399__ssa_v0: pl.Tensor[[cmp_block_num_inline1456__ssa_v0 * pl.const(32, pl.INDEX), 512], pl.BF16, pl.MemRef("mem_ddr_13", pl.const(0, pl.INT64), 0)],
        sparse_bias_inline1413__rv_v2: pl.Tensor[[t_dim_inline1418__ssa_v0, 640], pl.FP32, pl.MemRef("mem_ddr_14", pl.const(0, pl.INT64), 0)],
        mi_transfer_inline1379__ssa_v0: pl.InOut[pl.Tensor[[4608, 1], pl.FP32, pl.MemRef("mem_ddr_15", pl.const(0, pl.INT64), 18432)]],
        li_transfer_inline1432__ssa_v0: pl.InOut[pl.Tensor[[4608, 1], pl.FP32, pl.MemRef("mem_ddr_16", pl.const(0, pl.INT64), 18432)]],
        attn_mi_inline1435__ssa_v0: pl.Out[pl.Tensor[[t_heads_inline1477__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_17", pl.const(0, pl.INT64), 0)]],
        attn_li_inline1483__ssa_v0: pl.Out[pl.Tensor[[t_heads_inline1477__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_18", pl.const(0, pl.INT64), 0)]],
        attn_oi_inline1393__ssa_v0: pl.Out[pl.Tensor[[t_heads_inline1477__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_19", pl.const(0, pl.INT64), 0)]],
    ) -> tuple[
        pl.Tensor[[9216, 512], pl.BF16],
        pl.Tensor[[4608, 128], pl.FP32],
        pl.Tensor[[4608, 128], pl.BF16],
        pl.Tensor[[4608, 512], pl.FP32],
        pl.Tensor[[4608, 1], pl.FP32],
        pl.Tensor[[4608, 1], pl.FP32],
        pl.Tensor[[t_heads_inline1477__ssa_v0, 1], pl.FP32],
        pl.Tensor[[t_heads_inline1477__ssa_v0, 1], pl.FP32],
        pl.Tensor[[t_heads_inline1477__ssa_v0, 512], pl.FP32],
    ]:
        pl.func_attr({"mx_tensor_views_blocked": True, "split_aiv": True})
        self.qk_pv_aic(
            ffts_workspace_inline1401__ssa_v0,
            t_dim_inline1418__ssa_v0,
            q_flat_inline1409__ssa_v0,
            valid_block_mask_inline1376__ssa_v0,
            kv_transfer_inline1392__ssa_v0,
            score_transfer_inline1382__ssa_v0,
            probability_transfer_inline1394__ssa_v0,
            pv_transfer_inline1381__ssa_v0,
            attn_sink_col_inline1396__ssa_v0,
            positions__ssa_v0,
            window_indices__ssa_v0,
            ori_kv_flat_inline1480__ssa_v1,
            cmp_sparse_indices_inline1373__rv_v2,
            cmp_table__ssa_v0,
            cmp_kv_flat_inline1399__ssa_v0,
            sparse_bias_inline1413__rv_v2,
            mi_transfer_inline1379__ssa_v0,
            li_transfer_inline1432__ssa_v0,
            attn_mi_inline1435__ssa_v0,
            attn_li_inline1483__ssa_v0,
            attn_oi_inline1393__ssa_v0,
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
                    pl.adir.output_existing,
                    pl.adir.output_existing,
                    pl.adir.output_existing,
                ]
            },
        )
        self.qk_pv_aiv(
            ffts_workspace_inline1401__ssa_v0,
            t_dim_inline1418__ssa_v0,
            q_flat_inline1409__ssa_v0,
            valid_block_mask_inline1376__ssa_v0,
            kv_transfer_inline1392__ssa_v0,
            score_transfer_inline1382__ssa_v0,
            probability_transfer_inline1394__ssa_v0,
            pv_transfer_inline1381__ssa_v0,
            attn_sink_col_inline1396__ssa_v0,
            positions__ssa_v0,
            window_indices__ssa_v0,
            ori_kv_flat_inline1480__ssa_v1,
            cmp_sparse_indices_inline1373__rv_v2,
            cmp_table__ssa_v0,
            cmp_kv_flat_inline1399__ssa_v0,
            sparse_bias_inline1413__rv_v2,
            mi_transfer_inline1379__ssa_v0,
            li_transfer_inline1432__ssa_v0,
            attn_mi_inline1435__ssa_v0,
            attn_li_inline1483__ssa_v0,
            attn_oi_inline1393__ssa_v0,
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
                ]
            },
        )
        return (
            kv_transfer_inline1392__ssa_v0,
            score_transfer_inline1382__ssa_v0,
            probability_transfer_inline1394__ssa_v0,
            pv_transfer_inline1381__ssa_v0,
            mi_transfer_inline1379__ssa_v0,
            li_transfer_inline1432__ssa_v0,
            attn_mi_inline1435__ssa_v0,
            attn_li_inline1483__ssa_v0,
            attn_oi_inline1393__ssa_v0,
        )

    @pl.function(type=pl.FunctionType.Spmd)
    def qk_pv_spmd(
        self,
        ffts_workspace_inline1401__ssa_v0: pl.Tensor[[256], pl.INT64, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 2048)],
        t_dim_inline1418__ssa_v0: pl.Scalar[pl.INDEX],
        q_flat_inline1409__ssa_v0: pl.Tensor[[t_heads_inline1477__ssa_v0, 512], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)],
        valid_block_mask_inline1376__ssa_v0: pl.Tensor[[t_dim_inline1418__ssa_v0, 16], pl.INT32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)],
        kv_transfer_inline1392__ssa_v0: pl.InOut[pl.Tensor[[9216, 512], pl.BF16, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 9437184)]],
        score_transfer_inline1382__ssa_v0: pl.InOut[pl.Tensor[[4608, 128], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 2359296)]],
        probability_transfer_inline1394__ssa_v0: pl.InOut[pl.Tensor[[4608, 128], pl.BF16, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 1179648)]],
        pv_transfer_inline1381__ssa_v0: pl.InOut[pl.Tensor[[4608, 512], pl.FP32, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 9437184)]],
        attn_sink_col_inline1396__ssa_v0: pl.Tensor[[64, 1], pl.FP32, pl.MemRef("mem_ddr_7", pl.const(0, pl.INT64), 256)],
        positions__ssa_v0: pl.Tensor[[T_DYN, 1], pl.INT32, pl.MemRef("mem_ddr_8", pl.const(0, pl.INT64), 0)],
        window_indices__ssa_v0: pl.Tensor[[T_DYN, 128], pl.INT32, pl.MemRef("mem_ddr_9", pl.const(0, pl.INT64), 0)],
        ori_kv_flat_inline1480__ssa_v1: pl.Tensor[[ori_block_num_inline1408__ssa_v0 * pl.const(32, pl.INDEX), 512], pl.BF16, pl.MemRef("mem_ddr_10", pl.const(0, pl.INT64), 0)],
        cmp_sparse_indices_inline1373__rv_v2: pl.Tensor[[t_dim_inline1418__ssa_v0, 512], pl.INT32, pl.MemRef("mem_ddr_11", pl.const(0, pl.INT64), 0)],
        cmp_table__ssa_v0: pl.Tensor[[B_DYN, COMPRESSED_TABLE_COLUMNS_DYN], pl.INT32, pl.MemRef("mem_ddr_12", pl.const(0, pl.INT64), 0)],
        cmp_kv_flat_inline1399__ssa_v0: pl.Tensor[[cmp_block_num_inline1456__ssa_v0 * pl.const(32, pl.INDEX), 512], pl.BF16, pl.MemRef("mem_ddr_13", pl.const(0, pl.INT64), 0)],
        sparse_bias_inline1413__rv_v2: pl.Tensor[[t_dim_inline1418__ssa_v0, 640], pl.FP32, pl.MemRef("mem_ddr_14", pl.const(0, pl.INT64), 0)],
        mi_transfer_inline1379__ssa_v0: pl.InOut[pl.Tensor[[4608, 1], pl.FP32, pl.MemRef("mem_ddr_15", pl.const(0, pl.INT64), 18432)]],
        li_transfer_inline1432__ssa_v0: pl.InOut[pl.Tensor[[4608, 1], pl.FP32, pl.MemRef("mem_ddr_16", pl.const(0, pl.INT64), 18432)]],
        attn_mi_inline1435__ssa_v0: pl.Out[pl.Tensor[[t_heads_inline1477__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_17", pl.const(0, pl.INT64), 0)]],
        attn_li_inline1483__ssa_v0: pl.Out[pl.Tensor[[t_heads_inline1477__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_18", pl.const(0, pl.INT64), 0)]],
        attn_oi_inline1393__ssa_v0: pl.Out[pl.Tensor[[t_heads_inline1477__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_19", pl.const(0, pl.INT64), 0)]],
    ) -> tuple[pl.Tensor[[t_heads_inline1477__ssa_v0, 1], pl.FP32], pl.Tensor[[t_heads_inline1477__ssa_v0, 1], pl.FP32], pl.Tensor[[t_heads_inline1477__ssa_v0, 512], pl.FP32]]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        ret__tmp_v0: pl.Tuple[
            pl.Tensor[[9216, 512], pl.BF16],
            pl.Tensor[[4608, 128], pl.FP32],
            pl.Tensor[[4608, 128], pl.BF16],
            pl.Tensor[[4608, 512], pl.FP32],
            pl.Tensor[[4608, 1], pl.FP32],
            pl.Tensor[[4608, 1], pl.FP32],
            pl.Tensor[[t_heads_inline1477__ssa_v0, 1], pl.FP32],
            pl.Tensor[[t_heads_inline1477__ssa_v0, 1], pl.FP32],
            pl.Tensor[[t_heads_inline1477__ssa_v0, 512], pl.FP32],
        ] = self.qk_pv(
            ffts_workspace_inline1401__ssa_v0,
            t_dim_inline1418__ssa_v0,
            q_flat_inline1409__ssa_v0,
            valid_block_mask_inline1376__ssa_v0,
            kv_transfer_inline1392__ssa_v0,
            score_transfer_inline1382__ssa_v0,
            probability_transfer_inline1394__ssa_v0,
            pv_transfer_inline1381__ssa_v0,
            attn_sink_col_inline1396__ssa_v0,
            positions__ssa_v0,
            window_indices__ssa_v0,
            ori_kv_flat_inline1480__ssa_v1,
            cmp_sparse_indices_inline1373__rv_v2,
            cmp_table__ssa_v0,
            cmp_kv_flat_inline1399__ssa_v0,
            sparse_bias_inline1413__rv_v2,
            mi_transfer_inline1379__ssa_v0,
            li_transfer_inline1432__ssa_v0,
            attn_mi_inline1435__ssa_v0,
            attn_li_inline1483__ssa_v0,
            attn_oi_inline1393__ssa_v0,
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
                    pl.adir.output_existing,
                    pl.adir.output_existing,
                    pl.adir.output_existing,
                ]
            },
        )
        kv_transfer_inline1392__ssa_v1: pl.Tensor[[9216, 512], pl.BF16, pl.MemRef("mem_ddr_20", pl.const(0, pl.INT64), 9437184)] = ret__tmp_v0[0]
        score_transfer_inline1382__ssa_v1: pl.Tensor[[4608, 128], pl.FP32, pl.MemRef("mem_ddr_21", pl.const(0, pl.INT64), 2359296)] = ret__tmp_v0[1]
        probability_transfer_inline1394__ssa_v1: pl.Tensor[[4608, 128], pl.BF16, pl.MemRef("mem_ddr_22", pl.const(0, pl.INT64), 1179648)] = ret__tmp_v0[2]
        pv_transfer_inline1381__ssa_v1: pl.Tensor[[4608, 512], pl.FP32, pl.MemRef("mem_ddr_23", pl.const(0, pl.INT64), 9437184)] = ret__tmp_v0[3]
        mi_transfer_inline1379__ssa_v1: pl.Tensor[[4608, 1], pl.FP32, pl.MemRef("mem_ddr_24", pl.const(0, pl.INT64), 18432)] = ret__tmp_v0[4]
        li_transfer_inline1432__ssa_v1: pl.Tensor[[4608, 1], pl.FP32, pl.MemRef("mem_ddr_25", pl.const(0, pl.INT64), 18432)] = ret__tmp_v0[5]
        attn_mi_inline1435__ssa_v1: pl.Tensor[[t_heads_inline1477__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_26", pl.const(0, pl.INT64), 0)] = ret__tmp_v0[6]
        attn_li_inline1483__ssa_v1: pl.Tensor[[t_heads_inline1477__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_27", pl.const(0, pl.INT64), 0)] = ret__tmp_v0[7]
        attn_oi_inline1393__ssa_v1: pl.Tensor[[t_heads_inline1477__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_28", pl.const(0, pl.INT64), 0)] = ret__tmp_v0[8]
        return attn_mi_inline1435__ssa_v0, attn_li_inline1483__ssa_v0, attn_oi_inline1393__ssa_v0

    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def rope_cs(
        rope_swap_idx_inline1437__ssa_v0: pl.Out[pl.Tensor[[16, 64], pl.INT32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 4096)]],
        rope_cos_il_inline1368__ssa_v0: pl.Out[pl.Tensor[[384, 64], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 98304)]],
        rope_sin_signed_inline1367__ssa_v0: pl.Out[pl.Tensor[[384, 64], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 98304)]],
        rope_cs_blocks_inline1397__ssa_v0: pl.Scalar[pl.INDEX],
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
        sw_ones_inline1365__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_5, pl.const(8704, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.full([16, 64], dtype=pl.FP32, value=1.0)
        t__ci_tmp_v0: pl.Tile[[1, 192], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 768), pl.Mem.Vec] = pl.tile.create([1, 192], dtype=pl.FP32, target_memory=pl.Mem.Vec)
        t__tile: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_27, pl.const(4096, pl.INT64), 256), pl.Mem.Vec] = pl.tile.ci(
            pl.const(0, pl.INT32), [1, 64], tmp=t__ci_tmp_v0, dtype=pl.INT32, descending=False
        )
        sw_idx_f_inline1364__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 256), pl.Mem.Vec] = pl.tile.cast(t__tile, target_type=pl.FP32, mode="round")
        sw_col_inline1363__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_5, pl.const(8704, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.col_expand_mul(
            sw_ones_inline1365__tile, sw_idx_f_inline1364__tile
        )
        t__tile_1: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.muls(sw_col_inline1363__tile, 0.5)
        sw_dup_i32_inline1362__tile: pl.Tile[[16, 64], pl.INT32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(t__tile_1, target_type=pl.INT32, mode="trunc")
        sw_dup_f_inline1424__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(
            sw_dup_i32_inline1362__tile, target_type=pl.FP32, mode="round"
        )
        t__tile_2: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.muls(sw_dup_f_inline1424__tile, 2.0)
        sw_lane_inline1361__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.sub(sw_col_inline1363__tile, t__tile_2)
        t__tile_3: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_5, pl.const(8704, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.adds(sw_col_inline1363__tile, 1.0)
        t__tile_4: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.muls(sw_lane_inline1361__tile, 2.0)
        sw_swap_f_inline1447__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_5, pl.const(8704, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.sub(t__tile_3, t__tile_4)
        t__tile_5: pl.Tile[[16, 64], pl.INT32, pl.MemRef(mem_vec_5, pl.const(8704, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(sw_swap_f_inline1447__tile, target_type=pl.INT32, mode="round")
        rope_swap_idx_inline1437__tile: pl.Tensor[[16, 64], pl.INT32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 4096)] = pl.tile.store(t__tile_5, [0, 0], rope_swap_idx_inline1437__ssa_v0)
        cs_ones_inline1360__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_5, pl.const(8704, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.full([8, 64], dtype=pl.FP32, value=1.0)
        t__ci_tmp_v1: pl.Tile[[1, 192], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 768), pl.Mem.Vec] = pl.tile.create([1, 192], dtype=pl.FP32, target_memory=pl.Mem.Vec)
        t__tile_6: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_27, pl.const(4096, pl.INT64), 256), pl.Mem.Vec] = pl.tile.ci(
            pl.const(0, pl.INT32), [1, 64], tmp=t__ci_tmp_v1, dtype=pl.INT32, descending=False
        )
        cs_idx_f_inline1431__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 256), pl.Mem.Vec] = pl.tile.cast(t__tile_6, target_type=pl.FP32, mode="round")
        cs_col_inline1359__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_5, pl.const(8704, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.col_expand_mul(
            cs_ones_inline1360__tile, cs_idx_f_inline1431__tile
        )
        t__tile_7: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.muls(cs_col_inline1359__tile, 0.5)
        cs_dup_i32_inline1358__tile: pl.Tile[[8, 64], pl.INT32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(t__tile_7, target_type=pl.INT32, mode="trunc")
        cs_dup_f_inline1458__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(
            cs_dup_i32_inline1358__tile, target_type=pl.FP32, mode="round"
        )
        cs_dup_idx_inline1441__tile: pl.Tile[[8, 64], pl.INT32, pl.MemRef(mem_vec_27, pl.const(4096, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(
            cs_dup_f_inline1458__tile, target_type=pl.INT32, mode="round"
        )
        t__tile_8: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.muls(cs_dup_f_inline1458__tile, 2.0)
        cs_lane_inline1463__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_5, pl.const(8704, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.sub(cs_col_inline1359__tile, t__tile_8)
        t__tile_9: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_5, pl.const(8704, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.muls(cs_lane_inline1463__tile, 2.0)
        t__tile_10: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_5, pl.const(8704, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.subs(t__tile_9, 1.0)
        cs_sign_inline1434__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_5, pl.const(8704, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.neg(t__tile_10)
        for cs_rb_inline1357__idx_v0, (rope_cos_il_inline1368__iter_v1, rope_sin_signed_inline1367__iter_v1) in pl.range(
            rope_cs_blocks_inline1397__ssa_v0, init_values=(rope_cos_il_inline1368__ssa_v0, rope_sin_signed_inline1367__ssa_v0)
        ):
            cs_t0_inline1383__ssa_v0: pl.Scalar[pl.INDEX] = cs_rb_inline1357__idx_v0 * 8
            cs_cos_inline1377__tile: pl.Tile[[8, 32], pl.FP32, pl.MemRef(mem_vec_33, pl.const(6144, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.load(
                cos__ssa_v0, [cs_t0_inline1383__ssa_v0, 0], [8, 32], [8, 32], target_memory=pl.Mem.Vec
            )
            cs_sin_inline1356__tile: pl.Tile[[8, 32], pl.FP32, pl.MemRef(mem_vec_34, pl.const(7168, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.load(
                sin__ssa_v0, [cs_t0_inline1383__ssa_v0, 0], [8, 32], [8, 32], target_memory=pl.Mem.Vec
            )
            gather_acc_init: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.create([8, 64], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            for gather_lv, (gather_ia,) in pl.range(8, init_values=(gather_acc_init,)):
                gather_inp_row: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_33, pl.const(6144, pl.INT64), 128), pl.Mem.Vec] = pl.tile.slice(cs_cos_inline1377__tile, [1, 32], [gather_lv, 0], [1, 32])
                gather_idx_row: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_27, pl.const(4096, pl.INT64), 256), pl.Mem.Vec] = pl.tile.slice(
                    cs_dup_idx_inline1441__tile, [1, 64], [gather_lv, 0], [1, 64]
                )
                gather_row_tmp: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_36, pl.const(8192, pl.INT64), 256), pl.Mem.Vec] = pl.tile.create([1, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec)
                gather_row: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_37, pl.const(8448, pl.INT64), 256), pl.Mem.Vec] = pl.tile.gather(gather_inp_row, gather_idx_row, gather_row_tmp)
                gather_asmbl: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.assemble(gather_ia, gather_row, [gather_lv, 0])
                gather_rv: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 2048), pl.Mem.Vec] = pl.yield_(gather_asmbl)
            t__tile_11: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 2048), pl.Mem.Vec] = gather_rv
            rope_cos_il_inline1368__tile: pl.Tensor[[384, 64], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 98304)] = pl.tile.store(
                t__tile_11, [cs_t0_inline1383__ssa_v0, 0], rope_cos_il_inline1368__iter_v1
            )
            gather_acc_init_1: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.create([8, 64], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            for gather_lv_1, (gather_ia_1,) in pl.range(8, init_values=(gather_acc_init_1,)):
                gather_inp_row_1: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_34, pl.const(7168, pl.INT64), 128), pl.Mem.Vec] = pl.tile.slice(
                    cs_sin_inline1356__tile, [1, 32], [gather_lv_1, 0], [1, 32]
                )
                gather_idx_row_1: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_27, pl.const(4096, pl.INT64), 256), pl.Mem.Vec] = pl.tile.slice(
                    cs_dup_idx_inline1441__tile, [1, 64], [gather_lv_1, 0], [1, 64]
                )
                gather_row_tmp_1: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_33, pl.const(6144, pl.INT64), 256), pl.Mem.Vec] = pl.tile.create([1, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec)
                gather_row_1: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_36, pl.const(8192, pl.INT64), 256), pl.Mem.Vec] = pl.tile.gather(gather_inp_row_1, gather_idx_row_1, gather_row_tmp_1)
                gather_asmbl_1: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.assemble(gather_ia_1, gather_row_1, [gather_lv_1, 0])
                gather_rv_1: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 2048), pl.Mem.Vec] = pl.yield_(gather_asmbl_1)
            cs_sin_il_inline1355__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 2048), pl.Mem.Vec] = gather_rv_1
            t__tile_12: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.mul(cs_sin_il_inline1355__tile, cs_sign_inline1434__tile)
            rope_sin_signed_inline1367__tile: pl.Tensor[[384, 64], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 98304)] = pl.tile.store(
                t__tile_12, [cs_t0_inline1383__ssa_v0, 0], rope_sin_signed_inline1367__iter_v1
            )
            rope_cos_il_inline1368__rv_v2, rope_sin_signed_inline1367__rv_v2 = pl.yield_(rope_cos_il_inline1368__tile, rope_sin_signed_inline1367__tile)
        return rope_swap_idx_inline1437__ssa_v0, rope_cos_il_inline1368__ssa_v0, rope_sin_signed_inline1367__ssa_v0

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
        ori_block_num_inline1408__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(ori_kv__ssa_v0, 0)
        t_dim_inline1418__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(query__ssa_v0, 0)
        t_heads_inline1477__ssa_v0: pl.Scalar[pl.INDEX] = t_dim_inline1418__ssa_v0 * 64
        rope_cs_blocks_inline1397__ssa_v0: pl.Scalar[pl.INDEX] = t_dim_inline1418__ssa_v0 // 8
        ori_kv_flat_inline1480__ssa_v0: pl.Tensor[[ori_block_num_inline1408__ssa_v0 * pl.const(32, pl.INDEX), 512], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)] = pl.tensor.reshape(
            ori_kv__ssa_v0, [ori_block_num_inline1408__ssa_v0 * 32, 512]
        )
        ret__tmp_v0: pl.Tuple[pl.Tensor[[ori_block_num_inline1408__ssa_v0 * pl.const(32, pl.INDEX), 512], pl.BF16], pl.Scalar[pl.TASK_ID]] = pl.submit(
            self.kv_touch, ori_kv_flat_inline1480__ssa_v0, allow_early_resolve=True, attrs={"arg_directions": [pl.adir.inout]}
        )
        ori_kv_flat_inline1480__ssa_v1: pl.Tensor[[ori_block_num_inline1408__ssa_v0 * pl.const(32, pl.INDEX), 512], pl.BF16, pl.MemRef("mem_ddr_11", pl.const(0, pl.INT64), 0)] = ret__tmp_v0[0]
        tid__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0[1]
        sparse_bias_inline1413__ssa_v0: pl.Tensor[[t_dim_inline1418__ssa_v0, 640], pl.FP32, pl.MemRef("mem_ddr_12", pl.const(0, pl.INT64), 0)] = pl.tensor.create(
            [t_dim_inline1418__ssa_v0, 640], dtype=pl.FP32, layout=pl.TensorLayout.ND
        )
        cmp_sparse_indices_inline1373__ssa_v0: pl.Tensor[[t_dim_inline1418__ssa_v0, 512], pl.INT32, pl.MemRef("mem_ddr_13", pl.const(0, pl.INT64), 0)] = pl.tensor.create(
            [t_dim_inline1418__ssa_v0, 512], dtype=pl.INT32, layout=pl.TensorLayout.ND
        )
        valid_block_mask_inline1376__ssa_v0: pl.Tensor[[t_dim_inline1418__ssa_v0, 16], pl.INT32, pl.MemRef("mem_ddr_14", pl.const(0, pl.INT64), 0)] = pl.tensor.create(
            [t_dim_inline1418__ssa_v0, 16], dtype=pl.INT32, layout=pl.TensorLayout.ND
        )
        ret__tmp_v0_1: pl.Tuple[pl.Tensor[[t_dim_inline1418__ssa_v0, 512], pl.INT32], pl.Tensor[[t_dim_inline1418__ssa_v0, 640], pl.FP32], pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(
            self.csa_slots_build_valid_qk_plan_spmd,
            cmp_sparse_indices_inline1373__ssa_v0,
            sparse_bias_inline1413__ssa_v0,
            t_dim_inline1418__ssa_v0,
            topk__ssa_v0,
            positions__ssa_v0,
            window_indices__ssa_v0,
            valid_block_mask_inline1376__ssa_v0,
            core_num=16,
            allow_early_resolve=True,
            attrs={"arg_directions": [pl.adir.output_existing, pl.adir.output_existing, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.output_existing]},
        )
        cmp_sparse_indices_inline1373__rv_v2: pl.Tensor[[t_dim_inline1418__ssa_v0, 512], pl.INT32, pl.MemRef("mem_ddr_15", pl.const(0, pl.INT64), 0)] = ret__tmp_v0_1[0]
        sparse_bias_inline1413__rv_v2: pl.Tensor[[t_dim_inline1418__ssa_v0, 640], pl.FP32, pl.MemRef("mem_ddr_16", pl.const(0, pl.INT64), 0)] = ret__tmp_v0_1[1]
        qk_plan_tid_inline1378__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_1[2]
        cmp_block_num_inline1456__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(cmp_kv__ssa_v0, 0)
        cmp_kv_flat_inline1399__ssa_v0: pl.Tensor[[cmp_block_num_inline1456__ssa_v0 * pl.const(32, pl.INDEX), 512], pl.BF16, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)] = pl.tensor.reshape(
            cmp_kv__ssa_v0, [cmp_block_num_inline1456__ssa_v0 * 32, 512]
        )
        q_flat_inline1409__ssa_v0: pl.Tensor[[t_heads_inline1477__ssa_v0, 512], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)] = pl.tensor.reshape(
            query__ssa_v0, [t_heads_inline1477__ssa_v0, 512]
        )
        attn_sink_col_inline1396__ssa_v0: pl.Tensor[[64, 1], pl.FP32, pl.MemRef("mem_ddr_7", pl.const(0, pl.INT64), 256)] = pl.tensor.reshape(sink__ssa_v0, [64, 1])
        attn_mi_inline1435__ssa_v0: pl.Tensor[[t_heads_inline1477__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_17", pl.const(0, pl.INT64), 0)] = pl.tensor.create(
            [t_heads_inline1477__ssa_v0, 1], dtype=pl.FP32, layout=pl.TensorLayout.ND
        )
        attn_li_inline1483__ssa_v0: pl.Tensor[[t_heads_inline1477__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_18", pl.const(0, pl.INT64), 0)] = pl.tensor.create(
            [t_heads_inline1477__ssa_v0, 1], dtype=pl.FP32, layout=pl.TensorLayout.ND
        )
        attn_oi_inline1393__ssa_v0: pl.Tensor[[t_heads_inline1477__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_19", pl.const(0, pl.INT64), 0)] = pl.tensor.create(
            [t_heads_inline1477__ssa_v0, 512], dtype=pl.FP32, layout=pl.TensorLayout.ND
        )
        kv_transfer_inline1392__ssa_v0: pl.Tensor[[9216, 512], pl.BF16, pl.MemRef("mem_ddr_20", pl.const(0, pl.INT64), 9437184)] = pl.tensor.create(
            [9216, 512], dtype=pl.BF16, layout=pl.TensorLayout.ND
        )
        score_transfer_inline1382__ssa_v0: pl.Tensor[[4608, 128], pl.FP32, pl.MemRef("mem_ddr_21", pl.const(0, pl.INT64), 2359296)] = pl.tensor.create(
            [4608, 128], dtype=pl.FP32, layout=pl.TensorLayout.ND
        )
        probability_transfer_inline1394__ssa_v0: pl.Tensor[[4608, 128], pl.BF16, pl.MemRef("mem_ddr_22", pl.const(0, pl.INT64), 1179648)] = pl.tensor.create(
            [4608, 128], dtype=pl.BF16, layout=pl.TensorLayout.ND
        )
        pv_transfer_inline1381__ssa_v0: pl.Tensor[[4608, 512], pl.FP32, pl.MemRef("mem_ddr_23", pl.const(0, pl.INT64), 9437184)] = pl.tensor.create(
            [4608, 512], dtype=pl.FP32, layout=pl.TensorLayout.ND
        )
        mi_transfer_inline1379__ssa_v0: pl.Tensor[[4608, 1], pl.FP32, pl.MemRef("mem_ddr_24", pl.const(0, pl.INT64), 18432)] = pl.tensor.create([4608, 1], dtype=pl.FP32, layout=pl.TensorLayout.ND)
        li_transfer_inline1432__ssa_v0: pl.Tensor[[4608, 1], pl.FP32, pl.MemRef("mem_ddr_25", pl.const(0, pl.INT64), 18432)] = pl.tensor.create([4608, 1], dtype=pl.FP32, layout=pl.TensorLayout.ND)
        ffts_workspace_inline1401__ssa_v0: pl.Tensor[[256], pl.INT64, pl.MemRef("mem_ddr_26", pl.const(0, pl.INT64), 2048)] = pl.tensor.create([256], dtype=pl.INT64, layout=pl.TensorLayout.ND)
        ret__tmp_v0_2: pl.Tuple[
            pl.Tensor[[t_heads_inline1477__ssa_v0, 1], pl.FP32], pl.Tensor[[t_heads_inline1477__ssa_v0, 1], pl.FP32], pl.Tensor[[t_heads_inline1477__ssa_v0, 512], pl.FP32], pl.Scalar[pl.TASK_ID]
        ] = pl.spmd_submit(
            self.qk_pv_spmd,
            ffts_workspace_inline1401__ssa_v0,
            t_dim_inline1418__ssa_v0,
            q_flat_inline1409__ssa_v0,
            valid_block_mask_inline1376__ssa_v0,
            kv_transfer_inline1392__ssa_v0,
            score_transfer_inline1382__ssa_v0,
            probability_transfer_inline1394__ssa_v0,
            pv_transfer_inline1381__ssa_v0,
            attn_sink_col_inline1396__ssa_v0,
            positions__ssa_v0,
            window_indices__ssa_v0,
            ori_kv_flat_inline1480__ssa_v1,
            cmp_sparse_indices_inline1373__rv_v2,
            cmp_table__ssa_v0,
            cmp_kv_flat_inline1399__ssa_v0,
            sparse_bias_inline1413__rv_v2,
            mi_transfer_inline1379__ssa_v0,
            li_transfer_inline1432__ssa_v0,
            attn_mi_inline1435__ssa_v0,
            attn_li_inline1483__ssa_v0,
            attn_oi_inline1393__ssa_v0,
            deps=[qk_plan_tid_inline1378__ssa_v0],
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
                    pl.adir.output_existing,
                    pl.adir.output_existing,
                    pl.adir.output_existing,
                ]
            },
        )
        attn_mi_inline1435__ssa_v1: pl.Tensor[[t_heads_inline1477__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_27", pl.const(0, pl.INT64), 0)] = ret__tmp_v0_2[0]
        attn_li_inline1483__ssa_v1: pl.Tensor[[t_heads_inline1477__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_28", pl.const(0, pl.INT64), 0)] = ret__tmp_v0_2[1]
        attn_oi_inline1393__ssa_v1: pl.Tensor[[t_heads_inline1477__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_29", pl.const(0, pl.INT64), 0)] = ret__tmp_v0_2[2]
        qk_tid_inline1386__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_2[3]
        rope_cos_il_inline1368__ssa_v0: pl.Tensor[[384, 64], pl.FP32, pl.MemRef("mem_ddr_30", pl.const(0, pl.INT64), 98304)] = pl.tensor.create([384, 64], dtype=pl.FP32, layout=pl.TensorLayout.ND)
        rope_sin_signed_inline1367__ssa_v0: pl.Tensor[[384, 64], pl.FP32, pl.MemRef("mem_ddr_31", pl.const(0, pl.INT64), 98304)] = pl.tensor.create([384, 64], dtype=pl.FP32, layout=pl.TensorLayout.ND)
        rope_swap_idx_inline1437__ssa_v0: pl.Tensor[[16, 64], pl.INT32, pl.MemRef("mem_ddr_32", pl.const(0, pl.INT64), 4096)] = pl.tensor.create([16, 64], dtype=pl.INT32, layout=pl.TensorLayout.ND)
        ret__tmp_v0_3: pl.Tuple[pl.Tensor[[16, 64], pl.INT32], pl.Tensor[[384, 64], pl.FP32], pl.Tensor[[384, 64], pl.FP32], pl.Scalar[pl.TASK_ID]] = pl.submit(
            self.rope_cs,
            rope_swap_idx_inline1437__ssa_v0,
            rope_cos_il_inline1368__ssa_v0,
            rope_sin_signed_inline1367__ssa_v0,
            rope_cs_blocks_inline1397__ssa_v0,
            cos__ssa_v0,
            sin__ssa_v0,
            allow_early_resolve=True,
            attrs={"arg_directions": [pl.adir.output_existing, pl.adir.output_existing, pl.adir.output_existing, pl.adir.scalar, pl.adir.input, pl.adir.input]},
        )
        rope_swap_idx_inline1437__ssa_v1: pl.Tensor[[16, 64], pl.INT32, pl.MemRef("mem_ddr_33", pl.const(0, pl.INT64), 4096)] = ret__tmp_v0_3[0]
        rope_cos_il_inline1368__rv_v2: pl.Tensor[[384, 64], pl.FP32, pl.MemRef("mem_ddr_34", pl.const(0, pl.INT64), 98304)] = ret__tmp_v0_3[1]
        rope_sin_signed_inline1367__rv_v2: pl.Tensor[[384, 64], pl.FP32, pl.MemRef("mem_ddr_35", pl.const(0, pl.INT64), 98304)] = ret__tmp_v0_3[2]
        rope_tid_inline1366__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_3[3]
        attn_mi_inline1336__ssa_v0: pl.Tensor[[t_heads_inline1477__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_36", pl.const(0, pl.INT64), 0)] = attn_mi_inline1435__ssa_v1
        attn_li_inline1331__ssa_v0: pl.Tensor[[t_heads_inline1477__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_37", pl.const(0, pl.INT64), 0)] = attn_li_inline1483__ssa_v1
        attn_oi_inline1328__ssa_v0: pl.Tensor[[t_heads_inline1477__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_38", pl.const(0, pl.INT64), 0)] = attn_oi_inline1393__ssa_v1
        rope_cos_il_inline1335__ssa_v0: pl.Tensor[[384, 64], pl.FP32, pl.MemRef("mem_ddr_39", pl.const(0, pl.INT64), 98304)] = rope_cos_il_inline1368__rv_v2
        rope_sin_signed_inline1345__ssa_v0: pl.Tensor[[384, 64], pl.FP32, pl.MemRef("mem_ddr_40", pl.const(0, pl.INT64), 98304)] = rope_sin_signed_inline1367__rv_v2
        rope_swap_idx_inline1327__ssa_v0: pl.Tensor[[16, 64], pl.INT32, pl.MemRef("mem_ddr_41", pl.const(0, pl.INT64), 4096)] = rope_swap_idx_inline1437__ssa_v1
        qk_tid_inline1329__ssa_v0: pl.Scalar[pl.TASK_ID] = qk_tid_inline1386__ssa_v0
        rope_tid_inline1346__ssa_v0: pl.Scalar[pl.TASK_ID] = rope_tid_inline1366__ssa_v0
        t_dim_inline1326__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(query__ssa_v0, 0)
        merge_sink_inline1332__ssa_v0: pl.Tensor[[64, 1], pl.FP32, pl.MemRef("mem_ddr_7", pl.const(0, pl.INT64), 256)] = pl.tensor.reshape(sink__ssa_v0, [64, 1])
        ret__tmp_v0_4: pl.Tuple[pl.Tensor[[3072, 4096], pl.BF16], pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(
            self.merge_norm_spmd,
            rope_swap_idx_inline1327__ssa_v0,
            t_dim_inline1326__ssa_v0,
            attn_mi_inline1336__ssa_v0,
            attn_li_inline1331__ssa_v0,
            attn_oi_inline1328__ssa_v0,
            merge_sink_inline1332__ssa_v0,
            rope_cos_il_inline1335__ssa_v0,
            rope_sin_signed_inline1345__ssa_v0,
            packed__ssa_v0,
            deps=[qk_tid_inline1329__ssa_v0, rope_tid_inline1346__ssa_v0],
            core_num=48,
            attrs={"arg_directions": [pl.adir.input, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.output_existing]},
        )
        packed__ssa_v1: pl.Tensor[[3072, 4096], pl.BF16, pl.MemRef("mem_ddr_42", pl.const(0, pl.INT64), 25165824)] = ret__tmp_v0_4[0]
        merge_tid_inline1325__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_4[1]
        return packed__ssa_v1