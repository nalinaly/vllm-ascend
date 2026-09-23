# pypto.program: _jit_diagnose_o_projection
import pypto.language as pl

O_PROJ_T_DYN = pl.dynamic("O_PROJ_T_DYN")


@pl.program
class _jit_diagnose_o_projection:
    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def oproj_token_scale(
        act_scale_dq_inline51__ssa_v0: pl.Out[pl.Tensor[[1, 384], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 1536)]],
        t_dim_inline42__ssa_v0: pl.Scalar[pl.INDEX],
        o_r_pad_inline49__rv_v2: pl.Tensor[[384, 8192], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 12582912)],
    ) -> pl.Tensor[[1, 384], pl.FP32]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        mem_vec_2: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32)
        mem_vec_3: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32768)
        mem_vec_7: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32768)
        mem_vec_8: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32)
        mem_vec_12: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32)
        mem_vec_13: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32768)
        mem_vec_17: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32768)
        mem_vec_18: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32)
        unroll_main_end: pl.Scalar[pl.INDEX] = (t_dim_inline42__ssa_v0 + 7) // 16 * 16
        for qt_inline81__idx_v0, (act_scale_dq_inline51__iter_v1,) in pl.range(0, unroll_main_end, 16, init_values=(act_scale_dq_inline51__ssa_v0,)):
            token_amax_inline83__tile: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_2, pl.const(65600, pl.INT64), 32), pl.Mem.Vec] = pl.tile.full([1, 8], dtype=pl.FP32, value=0.0001)
            for scale_group_inline77__idx_v0, (token_amax_inline83__iter_v1,) in pl.range(8, init_values=(token_amax_inline83__tile,)):
                scale_col_inline85__ssa_v0: pl.Scalar[pl.INDEX] = scale_group_inline77__idx_v0 * 1024
                projected_inline34__tile: pl.Tile[[8, 1024], pl.FP32, pl.MemRef(mem_vec_3, pl.const(65632, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.load(
                    o_r_pad_inline49__rv_v2, [qt_inline81__idx_v0, scale_col_inline85__ssa_v0], [8, 1024], [8, 1024], target_memory=pl.Mem.Vec
                )
                t__tile: pl.Tile[[8, 1024], pl.BF16, pl.MemRef(mem_vec_7, pl.const(98400, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.cast(projected_inline34__tile, target_type=pl.BF16, mode="rint")
                projected_v1_inline86__tile: pl.Tile[[8, 1024], pl.FP32, pl.MemRef(mem_vec_3, pl.const(65632, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.cast(t__tile, target_type=pl.FP32, mode="round")
                t__tile_1: pl.Tile[[8, 1024], pl.FP32, pl.MemRef(mem_vec_3, pl.const(65632, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.abs(projected_v1_inline86__tile)
                tmp_tile: pl.Tile[[8, 1024], pl.FP32, pl.MemRef(mem_vec_7, pl.const(98400, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.create([8, 1024], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                t__tile_2: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_8, pl.const(131168, pl.INT64), 32), pl.Mem.Vec] = pl.tile.row_max(t__tile_1, tmp_tile)
                group_amax_inline47__tile: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_8, pl.const(131168, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(t__tile_2, [1, 8])
                token_amax_inline83__tile_1: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_2, pl.const(65600, pl.INT64), 32), pl.Mem.Vec] = pl.tile.maximum(
                    token_amax_inline83__iter_v1, group_amax_inline47__tile
                )
                token_amax_inline83__rv_v2: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_2, pl.const(65600, pl.INT64), 32), pl.Mem.Vec] = pl.yield_(token_amax_inline83__tile_1)
            t__tile_3: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_3, pl.const(65632, pl.INT64), 32), pl.Mem.Vec] = pl.tile.muls(token_amax_inline83__rv_v2, 0.007874015748031496)
            act_scale_dq_inline51__tile: pl.Tensor[[1, 384], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 1536)] = pl.tile.store(
                t__tile_3, [0, qt_inline81__idx_v0], act_scale_dq_inline51__iter_v1
            )
            token_amax_inline83__tile_2: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_12, pl.const(0, pl.INT64), 32), pl.Mem.Vec] = pl.tile.full([1, 8], dtype=pl.FP32, value=0.0001)
            for scale_group_inline77__idx_v0_1, (token_amax_inline83__iter_v1_1,) in pl.range(8, init_values=(token_amax_inline83__tile_2,)):
                scale_col_inline85__ssa_v0_1: pl.Scalar[pl.INDEX] = scale_group_inline77__idx_v0_1 * 1024
                projected_inline34__tile_1: pl.Tile[[8, 1024], pl.FP32, pl.MemRef(mem_vec_13, pl.const(32, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.load(
                    o_r_pad_inline49__rv_v2, [qt_inline81__idx_v0 + 8, scale_col_inline85__ssa_v0_1], [8, 1024], [8, 1024], target_memory=pl.Mem.Vec
                )
                t__tile_4: pl.Tile[[8, 1024], pl.BF16, pl.MemRef(mem_vec_17, pl.const(32800, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.cast(projected_inline34__tile_1, target_type=pl.BF16, mode="rint")
                projected_v1_inline86__tile_1: pl.Tile[[8, 1024], pl.FP32, pl.MemRef(mem_vec_13, pl.const(32, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.cast(
                    t__tile_4, target_type=pl.FP32, mode="round"
                )
                t__tile_5: pl.Tile[[8, 1024], pl.FP32, pl.MemRef(mem_vec_13, pl.const(32, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.abs(projected_v1_inline86__tile_1)
                tmp_tile_1: pl.Tile[[8, 1024], pl.FP32, pl.MemRef(mem_vec_17, pl.const(32800, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.create([8, 1024], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                t__tile_6: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_18, pl.const(65568, pl.INT64), 32), pl.Mem.Vec] = pl.tile.row_max(t__tile_5, tmp_tile_1)
                group_amax_inline47__tile_1: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_18, pl.const(65568, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(t__tile_6, [1, 8])
                token_amax_inline83__tile_3: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_12, pl.const(0, pl.INT64), 32), pl.Mem.Vec] = pl.tile.maximum(
                    token_amax_inline83__iter_v1_1, group_amax_inline47__tile_1
                )
                token_amax_inline83__rv_v2_1: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_12, pl.const(0, pl.INT64), 32), pl.Mem.Vec] = pl.yield_(token_amax_inline83__tile_3)
            t__tile_7: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_13, pl.const(32, pl.INT64), 32), pl.Mem.Vec] = pl.tile.muls(token_amax_inline83__rv_v2_1, 0.007874015748031496)
            act_scale_dq_inline51__tile_1: pl.Tensor[[1, 384], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 1536)] = pl.tile.store(
                t__tile_7, [0, qt_inline81__idx_v0 + 8], act_scale_dq_inline51__tile
            )
            act_scale_dq_inline51__rv_v2_main: pl.Tensor[[1, 384], pl.FP32, pl.MemRef("mem_ddr_22", pl.const(0, pl.INT64), 1536)] = pl.yield_(act_scale_dq_inline51__tile_1)
        unroll_rem: pl.Scalar[pl.INDEX] = (t_dim_inline42__ssa_v0 + 7) // 8 - (t_dim_inline42__ssa_v0 + 7) // 16 * 2
        if unroll_rem == 1:
            token_amax_inline83__tile_4: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_13, pl.const(32, pl.INT64), 32), pl.Mem.Vec] = pl.tile.full([1, 8], dtype=pl.FP32, value=0.0001)
            for scale_group_inline77__idx_v0_2, (token_amax_inline83__iter_v1_2,) in pl.range(8, init_values=(token_amax_inline83__tile_4,)):
                scale_col_inline85__ssa_v0_2: pl.Scalar[pl.INDEX] = scale_group_inline77__idx_v0_2 * 1024
                projected_inline34__tile_2: pl.Tile[[8, 1024], pl.FP32, pl.MemRef(mem_vec_3, pl.const(65632, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.load(
                    o_r_pad_inline49__rv_v2, [unroll_main_end, scale_col_inline85__ssa_v0_2], [8, 1024], [8, 1024], target_memory=pl.Mem.Vec
                )
                t__tile_8: pl.Tile[[8, 1024], pl.BF16, pl.MemRef(mem_vec_7, pl.const(98400, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.cast(projected_inline34__tile_2, target_type=pl.BF16, mode="rint")
                projected_v1_inline86__tile_2: pl.Tile[[8, 1024], pl.FP32, pl.MemRef(mem_vec_3, pl.const(65632, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.cast(
                    t__tile_8, target_type=pl.FP32, mode="round"
                )
                t__tile_9: pl.Tile[[8, 1024], pl.FP32, pl.MemRef(mem_vec_3, pl.const(65632, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.abs(projected_v1_inline86__tile_2)
                tmp_tile_2: pl.Tile[[8, 1024], pl.FP32, pl.MemRef(mem_vec_7, pl.const(98400, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.create([8, 1024], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                t__tile_10: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_17, pl.const(32800, pl.INT64), 32), pl.Mem.Vec] = pl.tile.row_max(t__tile_9, tmp_tile_2)
                group_amax_inline47__tile_2: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_17, pl.const(32800, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(t__tile_10, [1, 8])
                token_amax_inline83__tile_5: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_13, pl.const(32, pl.INT64), 32), pl.Mem.Vec] = pl.tile.maximum(
                    token_amax_inline83__iter_v1_2, group_amax_inline47__tile_2
                )
                token_amax_inline83__rv_v2_2: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_13, pl.const(32, pl.INT64), 32), pl.Mem.Vec] = pl.yield_(token_amax_inline83__tile_5)
            t__tile_11: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_3, pl.const(65632, pl.INT64), 32), pl.Mem.Vec] = pl.tile.muls(token_amax_inline83__rv_v2_2, 0.007874015748031496)
            act_scale_dq_inline51__tile_2: pl.Tensor[[1, 384], pl.FP32, pl.MemRef("mem_ddr_22", pl.const(0, pl.INT64), 1536)] = pl.tile.store(
                t__tile_11, [0, unroll_main_end], act_scale_dq_inline51__rv_v2_main
            )
        return act_scale_dq_inline51__ssa_v0

    @pl.function(type=pl.FunctionType.AIC, level=pl.Level.AIC, role=pl.Role.SubWorker)
    def proj_a_mm(
        t_dim_inline42__ssa_v0: pl.Scalar[pl.INDEX],
        row_base_o_inline63__ssa_v0: pl.Scalar[pl.INDEX],
        packed__ssa_v0: pl.Tensor[[3072, 4096], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 25165824)],
        wo_a__ssa_v0: pl.Tensor[[8, 1024, 4096], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 67108864)],
        g_inline61__idx_v0: pl.Scalar[pl.INDEX],
        o_r_pad_inline49__iter_v1: pl.Out[pl.Tensor[[384, 8192], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 12582912)]],
        out_col_g_inline90__ssa_v0: pl.Scalar[pl.INDEX],
    ) -> pl.Tensor[[384, 8192], pl.FP32]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        mem_mat_3: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 65536)
        mem_mat_4: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 65536)
        mem_acc_5: pl.Ptr = pl.tile.alloc(pl.Mem.Acc, 65536)
        mem_left_6: pl.Ptr = pl.tile.alloc(pl.Mem.Left, 32768)
        mem_right_7: pl.Ptr = pl.tile.alloc(pl.Mem.Right, 32768)
        mem_left_8: pl.Ptr = pl.tile.alloc(pl.Mem.Left, 32768)
        mem_right_9: pl.Ptr = pl.tile.alloc(pl.Mem.Right, 32768)
        mem_mat_13: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 65536)
        mem_mat_14: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 65536)
        pa_unit_inline56__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        pa_rb_inline41__ssa_v0: pl.Scalar[pl.INDEX] = pa_unit_inline56__ssa_v0 // 8
        nf_inline35__ssa_v0: pl.Scalar[pl.INDEX] = pa_unit_inline56__ssa_v0 - pa_rb_inline41__ssa_v0 * 8
        pa_r0_inline59__ssa_v0: pl.Scalar[pl.INDEX] = pa_rb_inline41__ssa_v0 * 128
        pa_rows_inline67__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(t_dim_inline42__ssa_v0 - pa_r0_inline59__ssa_v0, 128)
        pa_src0_inline29__ssa_v0: pl.Scalar[pl.INDEX] = row_base_o_inline63__ssa_v0 + pa_r0_inline59__ssa_v0
        n0_inline28__ssa_v0: pl.Scalar[pl.INDEX] = nf_inline35__ssa_v0 * 128
        xa_first_inline27__tile: pl.Tile[[128, 256], pl.BF16, pl.MemRef(mem_mat_3, pl.const(131072, pl.INT64), 65536), pl.Mem.Mat, pl.TileView(valid_shape=[pa_rows_inline67__ssa_v0, 256])] = (
            pl.tile.load(packed__ssa_v0, [pa_src0_inline29__ssa_v0, 0], [128, 256], [pa_rows_inline67__ssa_v0, 256], target_memory=pl.Mem.Mat)
        )
        wa_first_inline31__tile_view2d: pl.Tensor[[8192, 4096], pl.BF16, pl.TensorView(stride=[4096, 1], layout=pl.TensorLayout.ND), pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 67108864)] = (
            pl.tensor.view(wo_a__ssa_v0, [8192, 4096])
        )
        wa_first_inline31__tile: pl.Tile[[128, 256], pl.BF16, pl.MemRef(mem_mat_4, pl.const(196608, pl.INT64), 65536), pl.Mem.Mat] = pl.tile.load(
            wa_first_inline31__tile_view2d, [g_inline61__idx_v0 * 1024 + n0_inline28__ssa_v0, 0], [128, 256], [128, 256], target_memory=pl.Mem.Mat
        )
        wa_first_inline31__tile_t: pl.Tile[
            [256, 128], pl.BF16, pl.MemRef(mem_mat_4, pl.const(196608, pl.INT64), 65536), pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)
        ] = pl.tile.transpose_view(wa_first_inline31__tile)
        acc_a_inline39__tile_l0_init_storage: pl.Tile[[128, 128], pl.FP32, pl.MemRef(mem_acc_5, pl.const(0, pl.INT64), 65536), pl.Mem.Acc, pl.TileView(compact=pl.CompactMode.normal)] = pl.tile.create(
            [128, 128], dtype=pl.FP32, target_memory=pl.Mem.Acc, compact=True
        )
        acc_a_inline39__tile_l0_init: pl.Tile[
            [128, 128], pl.FP32, pl.MemRef(mem_acc_5, pl.const(0, pl.INT64), 65536), pl.Mem.Acc, pl.TileView(valid_shape=[pa_rows_inline67__ssa_v0, 128], compact=pl.CompactMode.normal)
        ] = pl.tile.set_validshape(acc_a_inline39__tile_l0_init_storage, pa_rows_inline67__ssa_v0, 128)
        acc_a_inline39__tile_l0_a: pl.Tile[
            [128, 128],
            pl.BF16,
            pl.MemRef(mem_left_6, pl.const(0, pl.INT64), 32768),
            pl.Mem.Left,
            pl.TileView(valid_shape=[pa_rows_inline67__ssa_v0, 128], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal),
        ] = pl.tile.extract(xa_first_inline27__tile, 0, 0, [128, 128], target_memory=pl.Mem.Left)
        acc_a_inline39__tile_l0_b: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_7, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
            wa_first_inline31__tile_t, 0, 0, [128, 128], target_memory=pl.Mem.Right
        )
        acc_a_inline39__tile_l0_a_1: pl.Tile[
            [128, 128],
            pl.BF16,
            pl.MemRef(mem_left_8, pl.const(32768, pl.INT64), 32768),
            pl.Mem.Left,
            pl.TileView(valid_shape=[pa_rows_inline67__ssa_v0, 128], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal),
        ] = pl.tile.extract(xa_first_inline27__tile, 0, 128, [128, 128], target_memory=pl.Mem.Left)
        acc_a_inline39__tile_l0_b_1: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_9, pl.const(32768, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
            wa_first_inline31__tile_t, 128, 0, [128, 128], target_memory=pl.Mem.Right
        )
        acc_a_inline39__tile_l0_c_acc: pl.Tile[
            [128, 128], pl.FP32, pl.MemRef(mem_acc_5, pl.const(0, pl.INT64), 65536), pl.Mem.Acc, pl.TileView(valid_shape=[pa_rows_inline67__ssa_v0, 128], compact=pl.CompactMode.normal)
        ] = pl.tile.matmul_acc(acc_a_inline39__tile_l0_init, acc_a_inline39__tile_l0_a, acc_a_inline39__tile_l0_b, True)
        acc_a_inline39__tile_l0_c_acc_1: pl.Tile[
            [128, 128], pl.FP32, pl.MemRef(mem_acc_5, pl.const(0, pl.INT64), 65536), pl.Mem.Acc, pl.TileView(valid_shape=[pa_rows_inline67__ssa_v0, 128], compact=pl.CompactMode.normal)
        ] = pl.tile.matmul_acc(acc_a_inline39__tile_l0_c_acc, acc_a_inline39__tile_l0_a_1, acc_a_inline39__tile_l0_b_1, False)
        for kb_inline91__idx_v0, (acc_a_inline39__iter_v1,) in pl.range(1, 15, 2, init_values=(acc_a_inline39__tile_l0_c_acc_1,)):
            k0_inline32__ssa_v0: pl.Scalar[pl.INDEX] = kb_inline91__idx_v0 * 256
            k0_inline32__ssa_v0_1: pl.Scalar[pl.INDEX] = kb_inline91__idx_v0 * 256 + 256
            xa_k_chunk_inline54__tile: pl.Tile[[128, 256], pl.BF16, pl.MemRef(mem_mat_3, pl.const(131072, pl.INT64), 65536), pl.Mem.Mat, pl.TileView(valid_shape=[pa_rows_inline67__ssa_v0, 256])] = (
                pl.tile.load(packed__ssa_v0, [pa_src0_inline29__ssa_v0, k0_inline32__ssa_v0], [128, 256], [pa_rows_inline67__ssa_v0, 256], target_memory=pl.Mem.Mat)
            )
            xa_k_chunk_inline54__tile_1: pl.Tile[[128, 256], pl.BF16, pl.MemRef(mem_mat_4, pl.const(196608, pl.INT64), 65536), pl.Mem.Mat, pl.TileView(valid_shape=[pa_rows_inline67__ssa_v0, 256])] = (
                pl.tile.load(packed__ssa_v0, [pa_src0_inline29__ssa_v0, k0_inline32__ssa_v0_1], [128, 256], [pa_rows_inline67__ssa_v0, 256], target_memory=pl.Mem.Mat)
            )
            wa_k_chunk_inline76__tile_view2d: pl.Tensor[[8192, 4096], pl.BF16, pl.TensorView(stride=[4096, 1], layout=pl.TensorLayout.ND), pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 67108864)] = (
                pl.tensor.view(wo_a__ssa_v0, [8192, 4096])
            )
            wa_k_chunk_inline76__tile: pl.Tile[[128, 256], pl.BF16, pl.MemRef(mem_mat_13, pl.const(0, pl.INT64), 65536), pl.Mem.Mat] = pl.tile.load(
                wa_k_chunk_inline76__tile_view2d, [g_inline61__idx_v0 * 1024 + n0_inline28__ssa_v0, k0_inline32__ssa_v0], [128, 256], [128, 256], target_memory=pl.Mem.Mat
            )
            wa_k_chunk_inline76__tile_view2d_1: pl.Tensor[
                [8192, 4096], pl.BF16, pl.TensorView(stride=[4096, 1], layout=pl.TensorLayout.ND), pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 67108864)
            ] = pl.tensor.view(wo_a__ssa_v0, [8192, 4096])
            wa_k_chunk_inline76__tile_1: pl.Tile[[128, 256], pl.BF16, pl.MemRef(mem_mat_14, pl.const(65536, pl.INT64), 65536), pl.Mem.Mat] = pl.tile.load(
                wa_k_chunk_inline76__tile_view2d_1, [g_inline61__idx_v0 * 1024 + n0_inline28__ssa_v0, k0_inline32__ssa_v0_1], [128, 256], [128, 256], target_memory=pl.Mem.Mat
            )
            wa_k_chunk_inline76__tile_t: pl.Tile[
                [256, 128], pl.BF16, pl.MemRef(mem_mat_13, pl.const(0, pl.INT64), 65536), pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)
            ] = pl.tile.transpose_view(wa_k_chunk_inline76__tile)
            acc_a_inline39__iter_v1_l0_a: pl.Tile[
                [128, 128],
                pl.BF16,
                pl.MemRef(mem_left_6, pl.const(0, pl.INT64), 32768),
                pl.Mem.Left,
                pl.TileView(valid_shape=[pa_rows_inline67__ssa_v0, 128], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal),
            ] = pl.tile.extract(xa_k_chunk_inline54__tile, 0, 0, [128, 128], target_memory=pl.Mem.Left)
            acc_a_inline39__iter_v1_l0_b: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_7, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                wa_k_chunk_inline76__tile_t, 0, 0, [128, 128], target_memory=pl.Mem.Right
            )
            acc_a_inline39__iter_v1_l0_a_1: pl.Tile[
                [128, 128],
                pl.BF16,
                pl.MemRef(mem_left_8, pl.const(32768, pl.INT64), 32768),
                pl.Mem.Left,
                pl.TileView(valid_shape=[pa_rows_inline67__ssa_v0, 128], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal),
            ] = pl.tile.extract(xa_k_chunk_inline54__tile, 0, 128, [128, 128], target_memory=pl.Mem.Left)
            acc_a_inline39__iter_v1_l0_b_1: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_9, pl.const(32768, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                wa_k_chunk_inline76__tile_t, 128, 0, [128, 128], target_memory=pl.Mem.Right
            )
            acc_a_inline39__iter_v1_l0_c_acc: pl.Tile[
                [128, 128], pl.FP32, pl.MemRef(mem_acc_5, pl.const(0, pl.INT64), 65536), pl.Mem.Acc, pl.TileView(valid_shape=[pa_rows_inline67__ssa_v0, 128], compact=pl.CompactMode.normal)
            ] = pl.tile.matmul_acc(acc_a_inline39__iter_v1, acc_a_inline39__iter_v1_l0_a, acc_a_inline39__iter_v1_l0_b)
            acc_a_inline39__iter_v1_l0_c_acc_1: pl.Tile[
                [128, 128], pl.FP32, pl.MemRef(mem_acc_5, pl.const(0, pl.INT64), 65536), pl.Mem.Acc, pl.TileView(valid_shape=[pa_rows_inline67__ssa_v0, 128], compact=pl.CompactMode.normal)
            ] = pl.tile.matmul_acc(acc_a_inline39__iter_v1_l0_c_acc, acc_a_inline39__iter_v1_l0_a_1, acc_a_inline39__iter_v1_l0_b_1)
            wa_k_chunk_inline76__tile_t_1: pl.Tile[
                [256, 128], pl.BF16, pl.MemRef(mem_mat_14, pl.const(65536, pl.INT64), 65536), pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)
            ] = pl.tile.transpose_view(wa_k_chunk_inline76__tile_1)
            acc_a_inline39__iter_v1_l0_a_2: pl.Tile[
                [128, 128],
                pl.BF16,
                pl.MemRef(mem_left_6, pl.const(0, pl.INT64), 32768),
                pl.Mem.Left,
                pl.TileView(valid_shape=[pa_rows_inline67__ssa_v0, 128], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal),
            ] = pl.tile.extract(xa_k_chunk_inline54__tile_1, 0, 0, [128, 128], target_memory=pl.Mem.Left)
            acc_a_inline39__iter_v1_l0_b_2: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_7, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                wa_k_chunk_inline76__tile_t_1, 0, 0, [128, 128], target_memory=pl.Mem.Right
            )
            acc_a_inline39__iter_v1_l0_a_3: pl.Tile[
                [128, 128],
                pl.BF16,
                pl.MemRef(mem_left_8, pl.const(32768, pl.INT64), 32768),
                pl.Mem.Left,
                pl.TileView(valid_shape=[pa_rows_inline67__ssa_v0, 128], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal),
            ] = pl.tile.extract(xa_k_chunk_inline54__tile_1, 0, 128, [128, 128], target_memory=pl.Mem.Left)
            acc_a_inline39__iter_v1_l0_b_3: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_9, pl.const(32768, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                wa_k_chunk_inline76__tile_t_1, 128, 0, [128, 128], target_memory=pl.Mem.Right
            )
            acc_a_inline39__iter_v1_l0_c_acc_2: pl.Tile[
                [128, 128], pl.FP32, pl.MemRef(mem_acc_5, pl.const(0, pl.INT64), 65536), pl.Mem.Acc, pl.TileView(valid_shape=[pa_rows_inline67__ssa_v0, 128], compact=pl.CompactMode.normal)
            ] = pl.tile.matmul_acc(acc_a_inline39__iter_v1_l0_c_acc_1, acc_a_inline39__iter_v1_l0_a_2, acc_a_inline39__iter_v1_l0_b_2)
            acc_a_inline39__iter_v1_l0_c_acc_3: pl.Tile[
                [128, 128], pl.FP32, pl.MemRef(mem_acc_5, pl.const(0, pl.INT64), 65536), pl.Mem.Acc, pl.TileView(valid_shape=[pa_rows_inline67__ssa_v0, 128], compact=pl.CompactMode.normal)
            ] = pl.tile.matmul_acc(acc_a_inline39__iter_v1_l0_c_acc_2, acc_a_inline39__iter_v1_l0_a_3, acc_a_inline39__iter_v1_l0_b_3)
            acc_a_inline39__rv_v2_main: pl.Tile[
                [128, 128], pl.FP32, pl.MemRef(mem_acc_5, pl.const(0, pl.INT64), 65536), pl.Mem.Acc, pl.TileView(valid_shape=[pa_rows_inline67__ssa_v0, 128], compact=pl.CompactMode.normal)
            ] = pl.yield_(acc_a_inline39__iter_v1_l0_c_acc_3)
        xa_k_chunk_inline54__tile_2: pl.Tile[[128, 256], pl.BF16, pl.MemRef(mem_mat_3, pl.const(131072, pl.INT64), 65536), pl.Mem.Mat, pl.TileView(valid_shape=[pa_rows_inline67__ssa_v0, 256])] = (
            pl.tile.load(packed__ssa_v0, [pa_src0_inline29__ssa_v0, 3840], [128, 256], [pa_rows_inline67__ssa_v0, 256], target_memory=pl.Mem.Mat)
        )
        wa_k_chunk_inline76__tile_view2d_2: pl.Tensor[[8192, 4096], pl.BF16, pl.TensorView(stride=[4096, 1], layout=pl.TensorLayout.ND), pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 67108864)] = (
            pl.tensor.view(wo_a__ssa_v0, [8192, 4096])
        )
        wa_k_chunk_inline76__tile_2: pl.Tile[[128, 256], pl.BF16, pl.MemRef(mem_mat_4, pl.const(196608, pl.INT64), 65536), pl.Mem.Mat] = pl.tile.load(
            wa_k_chunk_inline76__tile_view2d_2, [g_inline61__idx_v0 * 1024 + n0_inline28__ssa_v0, 3840], [128, 256], [128, 256], target_memory=pl.Mem.Mat
        )
        wa_k_chunk_inline76__tile_t_2: pl.Tile[
            [256, 128], pl.BF16, pl.MemRef(mem_mat_4, pl.const(196608, pl.INT64), 65536), pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)
        ] = pl.tile.transpose_view(wa_k_chunk_inline76__tile_2)
        acc_a_inline39__iter_v1_l0_a_4: pl.Tile[
            [128, 128],
            pl.BF16,
            pl.MemRef(mem_left_6, pl.const(0, pl.INT64), 32768),
            pl.Mem.Left,
            pl.TileView(valid_shape=[pa_rows_inline67__ssa_v0, 128], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal),
        ] = pl.tile.extract(xa_k_chunk_inline54__tile_2, 0, 0, [128, 128], target_memory=pl.Mem.Left)
        acc_a_inline39__iter_v1_l0_b_4: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_7, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
            wa_k_chunk_inline76__tile_t_2, 0, 0, [128, 128], target_memory=pl.Mem.Right
        )
        acc_a_inline39__iter_v1_l0_a_5: pl.Tile[
            [128, 128],
            pl.BF16,
            pl.MemRef(mem_left_8, pl.const(32768, pl.INT64), 32768),
            pl.Mem.Left,
            pl.TileView(valid_shape=[pa_rows_inline67__ssa_v0, 128], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal),
        ] = pl.tile.extract(xa_k_chunk_inline54__tile_2, 0, 128, [128, 128], target_memory=pl.Mem.Left)
        acc_a_inline39__iter_v1_l0_b_5: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_9, pl.const(32768, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
            wa_k_chunk_inline76__tile_t_2, 128, 0, [128, 128], target_memory=pl.Mem.Right
        )
        acc_a_inline39__iter_v1_l0_c_acc_4: pl.Tile[
            [128, 128], pl.FP32, pl.MemRef(mem_acc_5, pl.const(0, pl.INT64), 65536), pl.Mem.Acc, pl.TileView(valid_shape=[pa_rows_inline67__ssa_v0, 128], compact=pl.CompactMode.normal)
        ] = pl.tile.matmul_acc(acc_a_inline39__rv_v2_main, acc_a_inline39__iter_v1_l0_a_4, acc_a_inline39__iter_v1_l0_b_4)
        acc_a_inline39__iter_v1_l0_c_acc_5: pl.Tile[
            [128, 128], pl.FP32, pl.MemRef(mem_acc_5, pl.const(0, pl.INT64), 65536), pl.Mem.Acc, pl.TileView(valid_shape=[pa_rows_inline67__ssa_v0, 128], compact=pl.CompactMode.normal)
        ] = pl.tile.matmul_acc(acc_a_inline39__iter_v1_l0_c_acc_4, acc_a_inline39__iter_v1_l0_a_5, acc_a_inline39__iter_v1_l0_b_5)
        acc_a_inline39__rv_v2: pl.Tile[
            [128, 128], pl.FP32, pl.MemRef(mem_acc_5, pl.const(0, pl.INT64), 65536), pl.Mem.Acc, pl.TileView(valid_shape=[pa_rows_inline67__ssa_v0, 128], compact=pl.CompactMode.normal)
        ] = acc_a_inline39__iter_v1_l0_c_acc_5
        o_r_pad_inline49__tile: pl.Tensor[[384, 8192], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 12582912)] = pl.tile.store(
            acc_a_inline39__rv_v2, [pa_r0_inline59__ssa_v0, out_col_g_inline90__ssa_v0 + n0_inline28__ssa_v0], o_r_pad_inline49__iter_v1
        )
        return o_r_pad_inline49__iter_v1

    @pl.function(type=pl.FunctionType.Spmd)
    def proj_a_mm_spmd(
        self,
        t_dim_inline42__ssa_v0: pl.Scalar[pl.INDEX],
        row_base_o_inline63__ssa_v0: pl.Scalar[pl.INDEX],
        packed__ssa_v0: pl.Tensor[[3072, 4096], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 25165824)],
        wo_a__ssa_v0: pl.Tensor[[8, 1024, 4096], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 67108864)],
        g_inline61__idx_v0: pl.Scalar[pl.INDEX],
        o_r_pad_inline49__iter_v1: pl.Out[pl.Tensor[[384, 8192], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 12582912)]],
        out_col_g_inline90__ssa_v0: pl.Scalar[pl.INDEX],
    ) -> pl.Tensor[[384, 8192], pl.FP32]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        o_r_pad_inline49__ssa_v3: pl.Tensor[[384, 8192], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 12582912)] = self.proj_a_mm(
            t_dim_inline42__ssa_v0,
            row_base_o_inline63__ssa_v0,
            packed__ssa_v0,
            wo_a__ssa_v0,
            g_inline61__idx_v0,
            o_r_pad_inline49__iter_v1,
            out_col_g_inline90__ssa_v0,
            attrs={"arg_directions": [pl.adir.scalar, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.scalar, pl.adir.output_existing, pl.adir.scalar]},
        )
        return o_r_pad_inline49__iter_v1

    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def proj_b_act(
        wo_b_scale__ssa_v0: pl.Tensor[[4096], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 16384)],
        output__ssa_v0: pl.Out[pl.Tensor[[O_PROJ_T_DYN, 4096], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)]],
        t_dim_inline42__ssa_v0: pl.Scalar[pl.INDEX],
        partials_inline65__rv_v2: pl.Tensor[[384, 32768], pl.INT32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 50331648)],
        act_scale_dq_inline51__rv_v2: pl.Tensor[[1, 384], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 1536)],
    ) -> pl.Tensor[[O_PROJ_T_DYN, 4096], pl.BF16]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        mem_vec_4: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 2048)
        mem_vec_5: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
        mem_vec_6: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
        mem_vec_7: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
        act_idx_inline7__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        tblk_inline6__ssa_v0: pl.Scalar[pl.INDEX] = act_idx_inline7__ssa_v0 // 8
        nreg_inline5__ssa_v0: pl.Scalar[pl.INDEX] = act_idx_inline7__ssa_v0 - tblk_inline6__ssa_v0 * 8
        ob_n0_inline69__ssa_v0: pl.Scalar[pl.INDEX] = nreg_inline5__ssa_v0 * 512
        t0_inline18__ssa_v1: pl.Scalar[pl.INDEX] = tblk_inline6__ssa_v0 * 32
        wb_scale_inline12__tile: pl.Tile[[512], pl.FP32, pl.MemRef(mem_vec_4, pl.const(0, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.load(
            wo_b_scale__ssa_v0, [ob_n0_inline69__ssa_v0], [512], [512], target_memory=pl.Mem.Vec
        )
        wb_scale_chunk_inline4__tile: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_4, pl.const(0, pl.INT64), 2048), pl.Mem.Vec] = wb_scale_inline12__tile
        for b_tb_inline3__idx_v0, (output__iter_v1,) in pl.range(t0_inline18__ssa_v1, pl.min(t0_inline18__ssa_v1 + 32, t_dim_inline42__ssa_v0), 8, init_values=(output__ssa_v0,)):
            acc_i32_inline2__tile: pl.Tile[[8, 512], pl.INT32, pl.MemRef(mem_vec_5, pl.const(2048, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.full([8, 512], dtype=pl.INT32, value=0)
            for act_g_inline46__idx_v0, (acc_i32_inline2__iter_v1,) in pl.range(0, 8, 2, init_values=(acc_i32_inline2__tile,)):
                p_col0_inline55__ssa_v0: pl.Scalar[pl.INDEX] = act_g_inline46__idx_v0 * 4096 + ob_n0_inline69__ssa_v0
                p_col0_inline55__ssa_v0_1: pl.Scalar[pl.INDEX] = act_g_inline46__idx_v0 * 4096 + ob_n0_inline69__ssa_v0 + 4096
                p_g_inline21__tile: pl.Tile[[8, 512], pl.INT32, pl.MemRef(mem_vec_6, pl.const(18432, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.load(
                    partials_inline65__rv_v2, [b_tb_inline3__idx_v0, p_col0_inline55__ssa_v0], [8, 512], [8, 512], target_memory=pl.Mem.Vec
                )
                p_g_inline21__tile_1: pl.Tile[[8, 512], pl.INT32, pl.MemRef(mem_vec_7, pl.const(34816, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.load(
                    partials_inline65__rv_v2, [b_tb_inline3__idx_v0, p_col0_inline55__ssa_v0_1], [8, 512], [8, 512], target_memory=pl.Mem.Vec
                )
                acc_i32_inline2__tile_1: pl.Tile[[8, 512], pl.INT32, pl.MemRef(mem_vec_6, pl.const(18432, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.add(acc_i32_inline2__iter_v1, p_g_inline21__tile)
                acc_i32_inline2__tile_2: pl.Tile[[8, 512], pl.INT32, pl.MemRef(mem_vec_5, pl.const(2048, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.add(acc_i32_inline2__tile_1, p_g_inline21__tile_1)
                acc_i32_inline2__rv_v2: pl.Tile[[8, 512], pl.INT32, pl.MemRef(mem_vec_5, pl.const(2048, pl.INT64), 16384), pl.Mem.Vec] = pl.yield_(acc_i32_inline2__tile_2)
            t__tile: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_6, pl.const(18432, pl.INT64), 32), pl.Mem.Vec] = pl.tile.load(
                act_scale_dq_inline51__rv_v2, [0, b_tb_inline3__idx_v0], [1, 8], [1, 8], target_memory=pl.Mem.Vec
            )
            output_token_scale_inline74__tile: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_6, pl.const(18432, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(t__tile, [8, 1])
            t__tile_1: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_5, pl.const(2048, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.cast(acc_i32_inline2__rv_v2, target_type=pl.FP32, mode="round")
            acc_inline1__tile: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_5, pl.const(2048, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.row_expand_mul(t__tile_1, output_token_scale_inline74__tile)
            out_t_inline0__tile: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_5, pl.const(2048, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.col_expand_mul(acc_inline1__tile, wb_scale_chunk_inline4__tile)
            out_bf16_inline24__tile: pl.Tile[[8, 512], pl.BF16, pl.MemRef(mem_vec_5, pl.const(2048, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.cast(out_t_inline0__tile, target_type=pl.BF16, mode="rint")
            output__tile: pl.Tensor[[O_PROJ_T_DYN, 4096], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                out_bf16_inline24__tile, [b_tb_inline3__idx_v0, ob_n0_inline69__ssa_v0], output__iter_v1
            )
            output__rv_v2: pl.Tensor[[O_PROJ_T_DYN, 4096], pl.BF16, pl.MemRef("mem_ddr_16", pl.const(0, pl.INT64), 0)] = pl.yield_(output__tile)
        return output__ssa_v0

    @pl.function(type=pl.FunctionType.Spmd)
    def proj_b_act_spmd(
        self,
        wo_b_scale__ssa_v0: pl.Tensor[[4096], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 16384)],
        output__ssa_v0: pl.Out[pl.Tensor[[O_PROJ_T_DYN, 4096], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)]],
        t_dim_inline42__ssa_v0: pl.Scalar[pl.INDEX],
        partials_inline65__rv_v2: pl.Tensor[[384, 32768], pl.INT32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 50331648)],
        act_scale_dq_inline51__rv_v2: pl.Tensor[[1, 384], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 1536)],
    ) -> pl.Tensor[[O_PROJ_T_DYN, 4096], pl.BF16]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        output__rv_v2: pl.Tensor[[O_PROJ_T_DYN, 4096], pl.BF16, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 0)] = self.proj_b_act(
            wo_b_scale__ssa_v0,
            output__ssa_v0,
            t_dim_inline42__ssa_v0,
            partials_inline65__rv_v2,
            act_scale_dq_inline51__rv_v2,
            attrs={"arg_directions": [pl.adir.input, pl.adir.output_existing, pl.adir.scalar, pl.adir.input, pl.adir.input]},
        )
        return output__ssa_v0

    @pl.function(type=pl.FunctionType.AIC, level=pl.Level.AIC, role=pl.Role.SubWorker)
    def proj_b_mm(
        partials_inline65__iter_v1: pl.Out[pl.Tensor[[384, 32768], pl.INT32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 50331648)]],
        col_g_inline87__ssa_v0: pl.Scalar[pl.INDEX],
        o_r_i8_pad_inline45__rv_v7: pl.Tensor[[384, 8192], pl.INT8, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 3145728)],
        wo_b__ssa_v0: pl.Tensor[[4096, 8192], pl.INT8, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 33554432)],
        g_inline92__idx_v0: pl.Scalar[pl.INDEX],
    ) -> pl.Tensor[[384, 32768], pl.INT32]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        mem_acc_3: pl.Ptr = pl.tile.alloc(pl.Mem.Acc, 131072)
        mem_mat_4: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 32768)
        mem_mat_5: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 65536)
        mem_mat_6: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 32768)
        mem_mat_7: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 65536)
        mem_left_8: pl.Ptr = pl.tile.alloc(pl.Mem.Left, 32768)
        mem_right_9: pl.Ptr = pl.tile.alloc(pl.Mem.Right, 65536)
        mem_left_10: pl.Ptr = pl.tile.alloc(pl.Mem.Left, 32768)
        pb_unit_inline22__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        tb_inline20__ssa_v0: pl.Scalar[pl.INDEX] = pb_unit_inline22__ssa_v0 // 8
        dc_inline19__ssa_v0: pl.Scalar[pl.INDEX] = pb_unit_inline22__ssa_v0 - tb_inline20__ssa_v0 * 8
        t0_inline18__ssa_v0: pl.Scalar[pl.INDEX] = tb_inline20__ssa_v0 * 128
        d0_inline17__ssa_v0: pl.Scalar[pl.INDEX] = dc_inline19__ssa_v0 * 512
        for nf_inline84__idx_v0, (partials_inline65__iter_v3,) in pl.range(2, init_values=(partials_inline65__iter_v1,)):
            n0_inline28__ssa_v1: pl.Scalar[pl.INDEX] = d0_inline17__ssa_v0 + nf_inline84__idx_v0 * 256
            acc_b_inline16__tile: pl.Tile[[128, 256], pl.INT32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 131072), pl.Mem.Acc] = pl.tile.create([128, 256], dtype=pl.INT32, target_memory=pl.Mem.Acc)
            for kb_inline15__idx_v0, (acc_b_inline16__iter_v1,) in pl.range(0, 4, 2, init_values=(acc_b_inline16__tile,)):
                k0_inline32__ssa_v1: pl.Scalar[pl.INDEX] = col_g_inline87__ssa_v0 + kb_inline15__idx_v0 * 256
                k0_inline32__ssa_v1_1: pl.Scalar[pl.INDEX] = col_g_inline87__ssa_v0 + (kb_inline15__idx_v0 * 256 + 256)
                b_act_inline14__tile: pl.Tile[[128, 256], pl.INT8, pl.MemRef(mem_mat_4, pl.const(0, pl.INT64), 32768), pl.Mem.Mat] = pl.tile.load(
                    o_r_i8_pad_inline45__rv_v7, [t0_inline18__ssa_v0, k0_inline32__ssa_v1], [128, 256], [128, 256], target_memory=pl.Mem.Mat
                )
                b_weight_inline13__tile: pl.Tile[[256, 256], pl.INT8, pl.MemRef(mem_mat_5, pl.const(32768, pl.INT64), 65536), pl.Mem.Mat] = pl.tile.load(
                    wo_b__ssa_v0, [n0_inline28__ssa_v1, k0_inline32__ssa_v1], [256, 256], [256, 256], target_memory=pl.Mem.Mat
                )
                b_act_inline14__tile_1: pl.Tile[[128, 256], pl.INT8, pl.MemRef(mem_mat_6, pl.const(98304, pl.INT64), 32768), pl.Mem.Mat] = pl.tile.load(
                    o_r_i8_pad_inline45__rv_v7, [t0_inline18__ssa_v0, k0_inline32__ssa_v1_1], [128, 256], [128, 256], target_memory=pl.Mem.Mat
                )
                b_weight_inline13__tile_1: pl.Tile[[256, 256], pl.INT8, pl.MemRef(mem_mat_7, pl.const(131072, pl.INT64), 65536), pl.Mem.Mat] = pl.tile.load(
                    wo_b__ssa_v0, [n0_inline28__ssa_v1, k0_inline32__ssa_v1_1], [256, 256], [256, 256], target_memory=pl.Mem.Mat
                )
                b_weight_inline13__tile_t: pl.Tile[
                    [256, 256], pl.INT8, pl.MemRef(mem_mat_5, pl.const(32768, pl.INT64), 65536), pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)
                ] = pl.tile.transpose_view(b_weight_inline13__tile)
                b_act_inline14__tile_Left: pl.Tile[[128, 256], pl.INT8, pl.MemRef(mem_left_8, pl.const(32768, pl.INT64), 32768), pl.Mem.Left] = pl.tile.move(
                    b_act_inline14__tile, target_memory=pl.Mem.Left
                )
                b_weight_inline13__tile_t_Right: pl.Tile[[256, 256], pl.INT8, pl.MemRef(mem_right_9, pl.const(0, pl.INT64), 65536), pl.Mem.Right] = pl.tile.move(
                    b_weight_inline13__tile_t, target_memory=pl.Mem.Right
                )
                acc_b_inline16__tile_1: pl.Tile[[128, 256], pl.INT32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 131072), pl.Mem.Acc] = pl.tile.matmul_acc(
                    acc_b_inline16__iter_v1, b_act_inline14__tile_Left, b_weight_inline13__tile_t_Right, kb_inline15__idx_v0 == 0
                )
                b_weight_inline13__tile_t_1: pl.Tile[
                    [256, 256], pl.INT8, pl.MemRef(mem_mat_7, pl.const(131072, pl.INT64), 65536), pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)
                ] = pl.tile.transpose_view(b_weight_inline13__tile_1)
                b_act_inline14__tile_Left_1: pl.Tile[[128, 256], pl.INT8, pl.MemRef(mem_left_10, pl.const(0, pl.INT64), 32768), pl.Mem.Left] = pl.tile.move(
                    b_act_inline14__tile_1, target_memory=pl.Mem.Left
                )
                b_weight_inline13__tile_t_Right_1: pl.Tile[[256, 256], pl.INT8, pl.MemRef(mem_right_9, pl.const(0, pl.INT64), 65536), pl.Mem.Right] = pl.tile.move(
                    b_weight_inline13__tile_t_1, target_memory=pl.Mem.Right
                )
                acc_b_inline16__tile_2: pl.Tile[[128, 256], pl.INT32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 131072), pl.Mem.Acc] = pl.tile.matmul_acc(
                    acc_b_inline16__tile_1, b_act_inline14__tile_Left_1, b_weight_inline13__tile_t_Right_1, kb_inline15__idx_v0 == -1
                )
                acc_b_inline16__rv_v2: pl.Tile[[128, 256], pl.INT32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 131072), pl.Mem.Acc] = pl.yield_(acc_b_inline16__tile_2)
            partials_inline65__tile: pl.Tensor[[384, 32768], pl.INT32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 50331648)] = pl.tile.store(
                acc_b_inline16__rv_v2, [t0_inline18__ssa_v0, g_inline92__idx_v0 * 4096 + n0_inline28__ssa_v1], partials_inline65__iter_v3
            )
            partials_inline65__rv_v4: pl.Tensor[[384, 32768], pl.INT32, pl.MemRef("mem_ddr_13", pl.const(0, pl.INT64), 50331648)] = pl.yield_(partials_inline65__tile)
        return partials_inline65__iter_v1

    @pl.function(type=pl.FunctionType.Spmd)
    def proj_b_mm_spmd(
        self,
        partials_inline65__iter_v1: pl.Out[pl.Tensor[[384, 32768], pl.INT32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 50331648)]],
        col_g_inline87__ssa_v0: pl.Scalar[pl.INDEX],
        o_r_i8_pad_inline45__rv_v7: pl.Tensor[[384, 8192], pl.INT8, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 3145728)],
        wo_b__ssa_v0: pl.Tensor[[4096, 8192], pl.INT8, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 33554432)],
        g_inline92__idx_v0: pl.Scalar[pl.INDEX],
    ) -> pl.Tensor[[384, 32768], pl.INT32]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        partials_inline65__rv_v4: pl.Tensor[[384, 32768], pl.INT32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 50331648)] = self.proj_b_mm(
            partials_inline65__iter_v1,
            col_g_inline87__ssa_v0,
            o_r_i8_pad_inline45__rv_v7,
            wo_b__ssa_v0,
            g_inline92__idx_v0,
            attrs={"arg_directions": [pl.adir.output_existing, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.scalar]},
        )
        return partials_inline65__iter_v1

    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def quant(
        o_r_i8_pad_inline45__iter_v1: pl.Out[pl.Tensor[[384, 8192], pl.INT8, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 3145728)]],
        t_dim_inline42__ssa_v0: pl.Scalar[pl.INDEX],
        act_scale_dq_inline51__rv_v2: pl.Tensor[[1, 384], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 1536)],
        o_r_pad_inline49__rv_v2: pl.Tensor[[384, 8192], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 12582912)],
        col_g_inline87__ssa_v0: pl.Scalar[pl.INDEX],
        proj_b_padded_rows_inline36__ssa_v0: pl.Scalar[pl.INDEX],
    ) -> pl.Tensor[[384, 8192], pl.INT8]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        mem_vec_4: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32768)
        mem_vec_6: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32768)
        mem_vec_7: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32)
        mem_vec_8: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
        mem_vec_14: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32)
        mem_vec_15: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
        unroll_main_end: pl.Scalar[pl.INDEX] = (t_dim_inline42__ssa_v0 + 7) // 16 * 16
        for qt_inline88__idx_v0, (o_r_i8_pad_inline45__iter_v3,) in pl.range(0, unroll_main_end, 16, init_values=(o_r_i8_pad_inline45__iter_v1,)):
            token_scale_inline66__tile: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_8, pl.const(81984, pl.INT64), 32), pl.Mem.Vec] = pl.tile.load(
                act_scale_dq_inline51__rv_v2, [0, qt_inline88__idx_v0], [1, 8], [1, 8], target_memory=pl.Mem.Vec
            )
            oc_q_inline40__tile: pl.Tile[[8, 1024], pl.FP32, pl.MemRef(mem_vec_4, pl.const(16416, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.load(
                o_r_pad_inline49__rv_v2, [qt_inline88__idx_v0, col_g_inline87__ssa_v0], [8, 1024], [8, 1024], target_memory=pl.Mem.Vec
            )
            token_scale_inline66__tile_1: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_15, pl.const(32, pl.INT64), 32), pl.Mem.Vec] = pl.tile.load(
                act_scale_dq_inline51__rv_v2, [0, qt_inline88__idx_v0 + 8], [1, 8], [1, 8], target_memory=pl.Mem.Vec
            )
            oc_q_inline40__tile_1: pl.Tile[[8, 1024], pl.FP32, pl.MemRef(mem_vec_6, pl.const(49184, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.load(
                o_r_pad_inline49__rv_v2, [qt_inline88__idx_v0 + 8, col_g_inline87__ssa_v0], [8, 1024], [8, 1024], target_memory=pl.Mem.Vec
            )
            t__tile: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_7, pl.const(81952, pl.INT64), 32), pl.Mem.Vec] = pl.tile.recip(token_scale_inline66__tile)
            g_sq_col_inline78__tile: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_7, pl.const(81952, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(t__tile, [8, 1])
            t__tile_1: pl.Tile[[8, 1024], pl.BF16, pl.MemRef(mem_vec_8, pl.const(81984, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.cast(oc_q_inline40__tile, target_type=pl.BF16, mode="rint")
            oc_q_v1_inline38__tile: pl.Tile[[8, 1024], pl.FP32, pl.MemRef(mem_vec_4, pl.const(16416, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.cast(t__tile_1, target_type=pl.FP32, mode="round")
            oq_scaled_inline52__tile: pl.Tile[[8, 1024], pl.FP32, pl.MemRef(mem_vec_4, pl.const(16416, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.row_expand_mul(
                oc_q_v1_inline38__tile, g_sq_col_inline78__tile
            )
            oq_i32_inline60__tile: pl.Tile[[8, 1024], pl.INT32, pl.MemRef(mem_vec_4, pl.const(16416, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.cast(
                oq_scaled_inline52__tile, target_type=pl.INT32, mode="rint"
            )
            oq_half_inline89__tile: pl.Tile[[8, 1024], pl.FP16, pl.MemRef(mem_vec_4, pl.const(16416, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.cast(
                oq_i32_inline60__tile, target_type=pl.FP16, mode="round"
            )
            oq_i8_inline95__tile: pl.Tile[[8, 1024], pl.INT8, pl.MemRef(mem_vec_4, pl.const(16416, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.cast(
                oq_half_inline89__tile, target_type=pl.INT8, mode="trunc"
            )
            o_r_i8_pad_inline45__tile: pl.Tensor[[384, 8192], pl.INT8, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 3145728)] = pl.tile.store(
                oq_i8_inline95__tile, [qt_inline88__idx_v0, col_g_inline87__ssa_v0], o_r_i8_pad_inline45__iter_v3
            )
            t__tile_2: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_14, pl.const(0, pl.INT64), 32), pl.Mem.Vec] = pl.tile.recip(token_scale_inline66__tile_1)
            g_sq_col_inline78__tile_1: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_14, pl.const(0, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(t__tile_2, [8, 1])
            t__tile_3: pl.Tile[[8, 1024], pl.BF16, pl.MemRef(mem_vec_15, pl.const(32, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.cast(oc_q_inline40__tile_1, target_type=pl.BF16, mode="rint")
            oc_q_v1_inline38__tile_1: pl.Tile[[8, 1024], pl.FP32, pl.MemRef(mem_vec_6, pl.const(49184, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.cast(t__tile_3, target_type=pl.FP32, mode="round")
            oq_scaled_inline52__tile_1: pl.Tile[[8, 1024], pl.FP32, pl.MemRef(mem_vec_6, pl.const(49184, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.row_expand_mul(
                oc_q_v1_inline38__tile_1, g_sq_col_inline78__tile_1
            )
            oq_i32_inline60__tile_1: pl.Tile[[8, 1024], pl.INT32, pl.MemRef(mem_vec_6, pl.const(49184, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.cast(
                oq_scaled_inline52__tile_1, target_type=pl.INT32, mode="rint"
            )
            oq_half_inline89__tile_1: pl.Tile[[8, 1024], pl.FP16, pl.MemRef(mem_vec_6, pl.const(49184, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.cast(
                oq_i32_inline60__tile_1, target_type=pl.FP16, mode="round"
            )
            oq_i8_inline95__tile_1: pl.Tile[[8, 1024], pl.INT8, pl.MemRef(mem_vec_6, pl.const(49184, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.cast(
                oq_half_inline89__tile_1, target_type=pl.INT8, mode="trunc"
            )
            o_r_i8_pad_inline45__tile_1: pl.Tensor[[384, 8192], pl.INT8, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 3145728)] = pl.tile.store(
                oq_i8_inline95__tile_1, [qt_inline88__idx_v0 + 8, col_g_inline87__ssa_v0], o_r_i8_pad_inline45__tile
            )
            o_r_i8_pad_inline45__rv_v4_main: pl.Tensor[[384, 8192], pl.INT8, pl.MemRef("mem_ddr_21", pl.const(0, pl.INT64), 3145728)] = pl.yield_(o_r_i8_pad_inline45__tile_1)
        unroll_rem: pl.Scalar[pl.INDEX] = (t_dim_inline42__ssa_v0 + 7) // 8 - (t_dim_inline42__ssa_v0 + 7) // 16 * 2
        if unroll_rem == 1:
            token_scale_inline66__tile_2: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_4, pl.const(16416, pl.INT64), 32), pl.Mem.Vec] = pl.tile.load(
                act_scale_dq_inline51__rv_v2, [0, unroll_main_end], [1, 8], [1, 8], target_memory=pl.Mem.Vec
            )
            t__tile_4: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_8, pl.const(81984, pl.INT64), 32), pl.Mem.Vec] = pl.tile.recip(token_scale_inline66__tile_2)
            g_sq_col_inline78__tile_2: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_8, pl.const(81984, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(t__tile_4, [8, 1])
            oc_q_inline40__tile_2: pl.Tile[[8, 1024], pl.FP32, pl.MemRef(mem_vec_4, pl.const(16416, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.load(
                o_r_pad_inline49__rv_v2, [unroll_main_end, col_g_inline87__ssa_v0], [8, 1024], [8, 1024], target_memory=pl.Mem.Vec
            )
            t__tile_5: pl.Tile[[8, 1024], pl.BF16, pl.MemRef(mem_vec_6, pl.const(49184, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.cast(oc_q_inline40__tile_2, target_type=pl.BF16, mode="rint")
            oc_q_v1_inline38__tile_2: pl.Tile[[8, 1024], pl.FP32, pl.MemRef(mem_vec_4, pl.const(16416, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.cast(t__tile_5, target_type=pl.FP32, mode="round")
            oq_scaled_inline52__tile_2: pl.Tile[[8, 1024], pl.FP32, pl.MemRef(mem_vec_4, pl.const(16416, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.row_expand_mul(
                oc_q_v1_inline38__tile_2, g_sq_col_inline78__tile_2
            )
            oq_i32_inline60__tile_2: pl.Tile[[8, 1024], pl.INT32, pl.MemRef(mem_vec_4, pl.const(16416, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.cast(
                oq_scaled_inline52__tile_2, target_type=pl.INT32, mode="rint"
            )
            oq_half_inline89__tile_2: pl.Tile[[8, 1024], pl.FP16, pl.MemRef(mem_vec_4, pl.const(16416, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.cast(
                oq_i32_inline60__tile_2, target_type=pl.FP16, mode="round"
            )
            oq_i8_inline95__tile_2: pl.Tile[[8, 1024], pl.INT8, pl.MemRef(mem_vec_4, pl.const(16416, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.cast(
                oq_half_inline89__tile_2, target_type=pl.INT8, mode="trunc"
            )
            o_r_i8_pad_inline45__tile_2: pl.Tensor[[384, 8192], pl.INT8, pl.MemRef("mem_ddr_21", pl.const(0, pl.INT64), 3145728)] = pl.tile.store(
                oq_i8_inline95__tile_2, [unroll_main_end, col_g_inline87__ssa_v0], o_r_i8_pad_inline45__rv_v4_main
            )
            o_r_i8_pad_inline45__rv_v4: pl.Tensor[[384, 8192], pl.INT8, pl.MemRef("mem_ddr_31", pl.const(0, pl.INT64), 3145728)] = pl.yield_(o_r_i8_pad_inline45__tile_2)
        else:
            o_r_i8_pad_inline45__rv_v4: pl.Tensor[[384, 8192], pl.INT8, pl.MemRef("mem_ddr_31", pl.const(0, pl.INT64), 3145728)] = pl.yield_(o_r_i8_pad_inline45__rv_v4_main)
        for zt_inline93__idx_v0, (o_r_i8_pad_inline45__iter_v6,) in pl.range(t_dim_inline42__ssa_v0, proj_b_padded_rows_inline36__ssa_v0, 8, init_values=(o_r_i8_pad_inline45__rv_v4,)):
            zero_half_inline25__tile: pl.Tile[[8, 1024], pl.FP16, pl.MemRef(mem_vec_4, pl.const(16416, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.full([8, 1024], dtype=pl.FP16, value=0.0)
            t__tile_6: pl.Tile[[8, 1024], pl.INT8, pl.MemRef(mem_vec_4, pl.const(16416, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.cast(zero_half_inline25__tile, target_type=pl.INT8, mode="trunc")
            o_r_i8_pad_inline45__tile_3: pl.Tensor[[384, 8192], pl.INT8, pl.MemRef("mem_ddr_31", pl.const(0, pl.INT64), 3145728)] = pl.tile.store(
                t__tile_6, [zt_inline93__idx_v0, col_g_inline87__ssa_v0], o_r_i8_pad_inline45__iter_v6
            )
            o_r_i8_pad_inline45__rv_v7: pl.Tensor[[384, 8192], pl.INT8, pl.MemRef("mem_ddr_34", pl.const(0, pl.INT64), 3145728)] = pl.yield_(o_r_i8_pad_inline45__tile_3)
        return o_r_i8_pad_inline45__iter_v1

    @pl.function(type=pl.FunctionType.Orchestration, level=pl.Level.CHIP, role=pl.Role.Orchestrator, auto_scope=False)
    def diagnose_o_projection(
        self,
        packed__ssa_v0: pl.Tensor[[3072, 4096], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 25165824)],
        wo_a__ssa_v0: pl.Tensor[[8, 1024, 4096], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 67108864)],
        wo_b__ssa_v0: pl.Tensor[[4096, 8192], pl.INT8, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 33554432)],
        wo_b_scale__ssa_v0: pl.Tensor[[4096], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 16384)],
        output__ssa_v0: pl.Out[pl.Tensor[[O_PROJ_T_DYN, 4096], pl.BF16, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 0)]],
    ) -> pl.Tensor[[O_PROJ_T_DYN, 4096], pl.BF16]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        gate__ssa_v0: pl.Scalar[pl.TASK_ID] = pl.system.task_dummy(deps=[])
        t_dim_inline42__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(output__ssa_v0, 0)
        act_t_blks_inline57__ssa_v0: pl.Scalar[pl.INDEX] = (t_dim_inline42__ssa_v0 + 31) // 32
        proj_a_rows_inline50__ssa_v0: pl.Scalar[pl.INDEX] = (t_dim_inline42__ssa_v0 + 127) // 128
        proj_b_t_rows_inline43__ssa_v0: pl.Scalar[pl.INDEX] = (t_dim_inline42__ssa_v0 + 127) // 128
        proj_b_padded_rows_inline36__ssa_v0: pl.Scalar[pl.INDEX] = proj_b_t_rows_inline43__ssa_v0 * 128
        o_r_pad_inline49__ssa_v0: pl.Tensor[[384, 8192], pl.FP32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 12582912)] = pl.tensor.create([384, 8192], dtype=pl.FP32, layout=pl.TensorLayout.ND)
        o_r_i8_pad_inline45__ssa_v0: pl.Tensor[[384, 8192], pl.INT8, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 3145728)] = pl.tensor.create([384, 8192], dtype=pl.INT8, layout=pl.TensorLayout.ND)
        act_scale_dq_inline51__ssa_v0: pl.Tensor[[1, 384], pl.FP32, pl.MemRef("mem_ddr_7", pl.const(0, pl.INT64), 1536)] = pl.tensor.create([1, 384], dtype=pl.FP32, layout=pl.TensorLayout.ND)
        partials_inline65__ssa_v0: pl.Tensor[[384, 32768], pl.INT32, pl.MemRef("mem_ddr_8", pl.const(0, pl.INT64), 50331648)] = pl.tensor.create(
            [384, 32768], dtype=pl.INT32, layout=pl.TensorLayout.ND
        )
        proj_a_tids_inline53__ssa_v0: pl.Array[8, pl.TASK_ID] = pl.array.create(8, dtype=pl.TASK_ID)
        proj_b_tids_inline58__ssa_v0: pl.Array[8, pl.TASK_ID] = pl.array.create(8, dtype=pl.TASK_ID)
        with pl.scope(mode=pl.ScopeMode.MANUAL):
            for g_inline61__idx_v0, (o_r_pad_inline49__iter_v1, proj_a_tids_inline53__iter_v1) in pl.parallel(
                8, init_values=(o_r_pad_inline49__ssa_v0, proj_a_tids_inline53__ssa_v0), attrs={"iter_arg_rebind_0": False, "iter_arg_rebind_1": True}
            ):
                row_base_o_inline63__ssa_v0: pl.Scalar[pl.INDEX] = g_inline61__idx_v0 * 384
                out_col_g_inline90__ssa_v0: pl.Scalar[pl.INDEX] = g_inline61__idx_v0 * 1024
                ret__tmp_v0: pl.Tuple[pl.Tensor[[384, 8192], pl.FP32], pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(
                    self.proj_a_mm_spmd,
                    t_dim_inline42__ssa_v0,
                    row_base_o_inline63__ssa_v0,
                    packed__ssa_v0,
                    wo_a__ssa_v0,
                    g_inline61__idx_v0,
                    o_r_pad_inline49__iter_v1,
                    out_col_g_inline90__ssa_v0,
                    deps=[gate__ssa_v0],
                    core_num=proj_a_rows_inline50__ssa_v0 * 8,
                    allow_early_resolve=True,
                    attrs={"arg_directions": [pl.adir.scalar, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.scalar, pl.adir.output_existing, pl.adir.scalar]},
                )
                o_r_pad_inline49__ssa_v3: pl.Tensor[[384, 8192], pl.FP32, pl.MemRef("mem_ddr_9", pl.const(0, pl.INT64), 12582912)] = ret__tmp_v0[0]
                pa_tid_inline37__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0[1]
                proj_a_tids_inline53__ssa_v3: pl.Array[8, pl.TASK_ID] = pl.array.update_element(proj_a_tids_inline53__iter_v1, g_inline61__idx_v0, pa_tid_inline37__ssa_v0)
                o_r_pad_inline49__rv_v2, proj_a_tids_inline53__rv_v2 = pl.yield_(o_r_pad_inline49__ssa_v3, proj_a_tids_inline53__ssa_v3)
        _submit_deps_buf_inline68__ssa_v0: pl.Array[8, pl.TASK_ID] = pl.array.create(8, dtype=pl.TASK_ID)
        t__tmp_v0: pl.Scalar[pl.TASK_ID] = pl.array.get_element(proj_a_tids_inline53__rv_v2, 0)
        _submit_deps_buf_inline70__ssa_v0: pl.Array[8, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline68__ssa_v0, 0, t__tmp_v0)
        t__tmp_v1: pl.Scalar[pl.TASK_ID] = pl.array.get_element(proj_a_tids_inline53__rv_v2, 1)
        _submit_deps_buf_inline33__ssa_v0: pl.Array[8, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline70__ssa_v0, 1, t__tmp_v1)
        t__tmp_v2: pl.Scalar[pl.TASK_ID] = pl.array.get_element(proj_a_tids_inline53__rv_v2, 2)
        _submit_deps_buf_inline71__ssa_v0: pl.Array[8, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline33__ssa_v0, 2, t__tmp_v2)
        t__tmp_v3: pl.Scalar[pl.TASK_ID] = pl.array.get_element(proj_a_tids_inline53__rv_v2, 3)
        _submit_deps_buf_inline72__ssa_v0: pl.Array[8, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline71__ssa_v0, 3, t__tmp_v3)
        t__tmp_v4: pl.Scalar[pl.TASK_ID] = pl.array.get_element(proj_a_tids_inline53__rv_v2, 4)
        _submit_deps_buf_inline82__ssa_v0: pl.Array[8, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline72__ssa_v0, 4, t__tmp_v4)
        t__tmp_v5: pl.Scalar[pl.TASK_ID] = pl.array.get_element(proj_a_tids_inline53__rv_v2, 5)
        _submit_deps_buf_inline80__ssa_v0: pl.Array[8, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline82__ssa_v0, 5, t__tmp_v5)
        t__tmp_v6: pl.Scalar[pl.TASK_ID] = pl.array.get_element(proj_a_tids_inline53__rv_v2, 6)
        _submit_deps_buf_inline73__ssa_v0: pl.Array[8, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline80__ssa_v0, 6, t__tmp_v6)
        t__tmp_v7: pl.Scalar[pl.TASK_ID] = pl.array.get_element(proj_a_tids_inline53__rv_v2, 7)
        _submit_deps_buf_inline79__ssa_v0: pl.Array[8, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline73__ssa_v0, 7, t__tmp_v7)
        ret__tmp_v0_1: pl.Tuple[pl.Tensor[[1, 384], pl.FP32], pl.Scalar[pl.TASK_ID]] = pl.submit(
            self.oproj_token_scale,
            act_scale_dq_inline51__ssa_v0,
            t_dim_inline42__ssa_v0,
            o_r_pad_inline49__rv_v2,
            deps=[_submit_deps_buf_inline79__ssa_v0],
            allow_early_resolve=True,
            attrs={"arg_directions": [pl.adir.output_existing, pl.adir.scalar, pl.adir.input]},
        )
        act_scale_dq_inline51__rv_v2: pl.Tensor[[1, 384], pl.FP32, pl.MemRef("mem_ddr_11", pl.const(0, pl.INT64), 1536)] = ret__tmp_v0_1[0]
        scale_tid_inline26__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_1[1]
        with pl.scope(mode=pl.ScopeMode.MANUAL):
            for g_inline92__idx_v0, (o_r_i8_pad_inline45__iter_v1, partials_inline65__iter_v1, proj_b_tids_inline58__iter_v1) in pl.parallel(
                8,
                init_values=(o_r_i8_pad_inline45__ssa_v0, partials_inline65__ssa_v0, proj_b_tids_inline58__ssa_v0),
                attrs={"iter_arg_rebind_0": False, "iter_arg_rebind_1": False, "iter_arg_rebind_2": True},
            ):
                col_g_inline87__ssa_v0: pl.Scalar[pl.INDEX] = g_inline92__idx_v0 * 1024
                ret__tmp_v0_2: pl.Tuple[pl.Tensor[[384, 8192], pl.INT8], pl.Scalar[pl.TASK_ID]] = pl.submit(
                    self.quant,
                    o_r_i8_pad_inline45__iter_v1,
                    t_dim_inline42__ssa_v0,
                    act_scale_dq_inline51__rv_v2,
                    o_r_pad_inline49__rv_v2,
                    col_g_inline87__ssa_v0,
                    proj_b_padded_rows_inline36__ssa_v0,
                    deps=[scale_tid_inline26__ssa_v0],
                    allow_early_resolve=True,
                    attrs={"arg_directions": [pl.adir.output_existing, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.scalar, pl.adir.scalar]},
                )
                o_r_i8_pad_inline45__rv_v7: pl.Tensor[[384, 8192], pl.INT8, pl.MemRef("mem_ddr_12", pl.const(0, pl.INT64), 3145728)] = ret__tmp_v0_2[0]
                q_tid_inline48__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_2[1]
                ret__tmp_v0_3: pl.Tuple[pl.Tensor[[384, 32768], pl.INT32], pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(
                    self.proj_b_mm_spmd,
                    partials_inline65__iter_v1,
                    col_g_inline87__ssa_v0,
                    o_r_i8_pad_inline45__rv_v7,
                    wo_b__ssa_v0,
                    g_inline92__idx_v0,
                    deps=[q_tid_inline48__ssa_v0],
                    core_num=proj_b_t_rows_inline43__ssa_v0 * 8,
                    allow_early_resolve=True,
                    attrs={"arg_directions": [pl.adir.output_existing, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.scalar]},
                )
                partials_inline65__rv_v4: pl.Tensor[[384, 32768], pl.INT32, pl.MemRef("mem_ddr_13", pl.const(0, pl.INT64), 50331648)] = ret__tmp_v0_3[0]
                pb_tid_inline23__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_3[1]
                proj_b_tids_inline58__ssa_v3: pl.Array[8, pl.TASK_ID] = pl.array.update_element(proj_b_tids_inline58__iter_v1, g_inline92__idx_v0, pb_tid_inline23__ssa_v0)
                o_r_i8_pad_inline45__rv_v2, partials_inline65__rv_v2, proj_b_tids_inline58__rv_v2 = pl.yield_(o_r_i8_pad_inline45__rv_v7, partials_inline65__rv_v4, proj_b_tids_inline58__ssa_v3)
        _submit_deps_buf_inline94__ssa_v0: pl.Array[8, pl.TASK_ID] = pl.array.create(8, dtype=pl.TASK_ID)
        t__tmp_v15: pl.Scalar[pl.TASK_ID] = pl.array.get_element(proj_b_tids_inline58__rv_v2, 0)
        _submit_deps_buf_inline75__ssa_v0: pl.Array[8, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline94__ssa_v0, 0, t__tmp_v15)
        t__tmp_v16: pl.Scalar[pl.TASK_ID] = pl.array.get_element(proj_b_tids_inline58__rv_v2, 1)
        _submit_deps_buf_inline44__ssa_v0: pl.Array[8, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline75__ssa_v0, 1, t__tmp_v16)
        t__tmp_v17: pl.Scalar[pl.TASK_ID] = pl.array.get_element(proj_b_tids_inline58__rv_v2, 2)
        _submit_deps_buf_inline11__ssa_v0: pl.Array[8, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline44__ssa_v0, 2, t__tmp_v17)
        t__tmp_v18: pl.Scalar[pl.TASK_ID] = pl.array.get_element(proj_b_tids_inline58__rv_v2, 3)
        _submit_deps_buf_inline62__ssa_v0: pl.Array[8, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline11__ssa_v0, 3, t__tmp_v18)
        t__tmp_v19: pl.Scalar[pl.TASK_ID] = pl.array.get_element(proj_b_tids_inline58__rv_v2, 4)
        _submit_deps_buf_inline10__ssa_v0: pl.Array[8, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline62__ssa_v0, 4, t__tmp_v19)
        t__tmp_v20: pl.Scalar[pl.TASK_ID] = pl.array.get_element(proj_b_tids_inline58__rv_v2, 5)
        _submit_deps_buf_inline9__ssa_v0: pl.Array[8, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline10__ssa_v0, 5, t__tmp_v20)
        t__tmp_v21: pl.Scalar[pl.TASK_ID] = pl.array.get_element(proj_b_tids_inline58__rv_v2, 6)
        _submit_deps_buf_inline30__ssa_v0: pl.Array[8, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline9__ssa_v0, 6, t__tmp_v21)
        t__tmp_v22: pl.Scalar[pl.TASK_ID] = pl.array.get_element(proj_b_tids_inline58__rv_v2, 7)
        _submit_deps_buf_inline8__ssa_v0: pl.Array[8, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline30__ssa_v0, 7, t__tmp_v22)
        ret__tmp_v0_4: pl.Tuple[pl.Tensor[[O_PROJ_T_DYN, 4096], pl.BF16], pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(
            self.proj_b_act_spmd,
            wo_b_scale__ssa_v0,
            output__ssa_v0,
            t_dim_inline42__ssa_v0,
            partials_inline65__rv_v2,
            act_scale_dq_inline51__rv_v2,
            deps=[_submit_deps_buf_inline8__ssa_v0],
            core_num=act_t_blks_inline57__ssa_v0 * 8,
            allow_early_resolve=True,
            attrs={"arg_directions": [pl.adir.input, pl.adir.output_existing, pl.adir.scalar, pl.adir.input, pl.adir.input]},
        )
        output__rv_v2: pl.Tensor[[O_PROJ_T_DYN, 4096], pl.BF16, pl.MemRef("mem_ddr_16", pl.const(0, pl.INT64), 0)] = ret__tmp_v0_4[0]
        _act_tid_inline64__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_4[1]
        return output__rv_v2