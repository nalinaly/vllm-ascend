# pypto.program: _jit_diagnose_indexer_weights
import pypto.language as pl

T_DYN = pl.dynamic("T_DYN")


@pl.program
class _jit_diagnose_indexer_weights:
    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def diagnose_indexer_weights_incore_3(
        weights__ssa_v0: pl.Tensor[[384, 64], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 98304)],
        published_weights__ssa_v0: pl.Out[pl.Tensor[[384, 64], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 98304)]],
        coefficients__ssa_v0: pl.Tensor[[6144, 64], pl.FP16, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 786432)],
        published_coefficients__ssa_v0: pl.Out[pl.Tensor[[384, 64], pl.FP16, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 49152)]],
    ) -> tuple[pl.Tensor[[384, 64], pl.FP32], pl.Tensor[[384, 64], pl.FP16]]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        mem_vec_4: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        row__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        t__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_4, pl.const(0, pl.INT64), 256), pl.Mem.Vec] = pl.tile.load(weights__ssa_v0, [row__ssa_v0, 0], [1, 64], [1, 64], target_memory=pl.Mem.Vec)
        published_weights__tile: pl.Tensor[[384, 64], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 98304)] = pl.tile.store(t__tile, [row__ssa_v0, 0], published_weights__ssa_v0)
        coefficient_row__ssa_v0: pl.Scalar[pl.INDEX] = row__ssa_v0 * 16
        t__tile_1: pl.Tile[[1, 64], pl.FP16, pl.MemRef(mem_vec_4, pl.const(0, pl.INT64), 128), pl.Mem.Vec] = pl.tile.load(
            coefficients__ssa_v0, [coefficient_row__ssa_v0, 0], [1, 64], [1, 64], target_memory=pl.Mem.Vec
        )
        published_coefficients__tile: pl.Tensor[[384, 64], pl.FP16, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 49152)] = pl.tile.store(t__tile_1, [row__ssa_v0, 0], published_coefficients__ssa_v0)
        return published_weights__ssa_v0, published_coefficients__ssa_v0

    @pl.function(type=pl.FunctionType.Spmd)
    def diagnose_indexer_weights_spmd_3(
        self,
        weights__ssa_v0: pl.Tensor[[384, 64], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 98304)],
        published_weights__ssa_v0: pl.Out[pl.Tensor[[384, 64], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 98304)]],
        coefficients__ssa_v0: pl.Tensor[[6144, 64], pl.FP16, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 786432)],
        published_coefficients__ssa_v0: pl.Out[pl.Tensor[[384, 64], pl.FP16, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 49152)]],
    ) -> tuple[pl.Tensor[[384, 64], pl.FP32], pl.Tensor[[384, 64], pl.FP16]]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        ret__tmp_v0: pl.Tuple[pl.Tensor[[384, 64], pl.FP32], pl.Tensor[[384, 64], pl.FP16]] = self.diagnose_indexer_weights_incore_3(
            weights__ssa_v0,
            published_weights__ssa_v0,
            coefficients__ssa_v0,
            published_coefficients__ssa_v0,
            attrs={"arg_directions": [pl.adir.input, pl.adir.output_existing, pl.adir.input, pl.adir.output_existing]},
        )
        published_weights__ssa_v1: pl.Tensor[[384, 64], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 98304)] = ret__tmp_v0[0]
        published_coefficients__ssa_v1: pl.Tensor[[384, 64], pl.FP16, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 49152)] = ret__tmp_v0[1]
        return published_weights__ssa_v0, published_coefficients__ssa_v0

    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def indexer_head_coefficients(
        position_ids__ssa_v0: pl.Tensor[[T_DYN], pl.INT32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)],
        coefficients_inline34__ssa_v0: pl.Out[pl.Tensor[[6144, 64], pl.FP16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 786432)]],
        query_scale__ssa_v0: pl.Tensor[[24576, 1], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 98304)],
        weights__ssa_v0: pl.Tensor[[384, 64], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 98304)],
    ) -> pl.Tensor[[6144, 64], pl.FP16]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        mem_vec_5: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        mem_vec_7: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 128)
        mem_vec_8: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 4096)
        coefficient_worker_inline30__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        coefficient_count_inline28__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(position_ids__ssa_v0, 0)
        for coefficient_query_inline29__idx_v0, (coefficients_inline34__iter_v1,) in pl.range(
            coefficient_worker_inline30__ssa_v0, coefficient_count_inline28__ssa_v0, 48, init_values=(coefficients_inline34__ssa_v0,)
        ):
            coefficient_head_begin_inline27__ssa_v0: pl.Scalar[pl.INDEX] = coefficient_query_inline29__idx_v0 * 64
            t__tile: pl.Tile[[64, 1], pl.FP32, pl.MemRef(mem_vec_8, pl.const(384, pl.INT64), 256), pl.Mem.Vec] = pl.tile.load(
                query_scale__ssa_v0, [coefficient_head_begin_inline27__ssa_v0, 0], [64, 1], [64, 1], target_memory=pl.Mem.Vec
            )
            query_scale_inline26__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_8, pl.const(384, pl.INT64), 256), pl.Mem.Vec] = pl.tile.reshape(t__tile, [1, 64])
            query_weight_inline33__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_5, pl.const(0, pl.INT64), 256), pl.Mem.Vec] = pl.tile.load(
                weights__ssa_v0, [coefficient_query_inline29__idx_v0, 0], [1, 64], [1, 64], target_memory=pl.Mem.Vec
            )
            t__tile_1: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_8, pl.const(384, pl.INT64), 256), pl.Mem.Vec] = pl.tile.mul(query_scale_inline26__tile, query_weight_inline33__tile)
            head_coefficient_inline25__tile: pl.Tile[[1, 64], pl.FP16, pl.MemRef(mem_vec_7, pl.const(256, pl.INT64), 128), pl.Mem.Vec] = pl.tile.cast(t__tile_1, target_type=pl.FP16, mode="rint")
            t__tile_2: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_8, pl.const(384, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.full([16, 64], dtype=pl.FP32, value=1.0)
            t__tile_3: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_5, pl.const(0, pl.INT64), 256), pl.Mem.Vec] = pl.tile.cast(head_coefficient_inline25__tile, target_type=pl.FP32, mode="round")
            coefficient_rows_inline32__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_8, pl.const(384, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.col_expand_mul(t__tile_2, t__tile_3)
            coefficient_row_inline24__ssa_v0: pl.Scalar[pl.INDEX] = coefficient_query_inline29__idx_v0 * 16
            t__tile_4: pl.Tile[[16, 64], pl.FP16, pl.MemRef(mem_vec_8, pl.const(384, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(coefficient_rows_inline32__tile, target_type=pl.FP16, mode="round")
            coefficients_inline34__tile: pl.Tensor[[6144, 64], pl.FP16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 786432)] = pl.tile.store(
                t__tile_4, [coefficient_row_inline24__ssa_v0, 0], coefficients_inline34__iter_v1
            )
            coefficients_inline34__rv_v2: pl.Tensor[[6144, 64], pl.FP16, pl.MemRef("mem_ddr_12", pl.const(0, pl.INT64), 786432)] = pl.yield_(coefficients_inline34__tile)
        return coefficients_inline34__ssa_v0

    @pl.function(type=pl.FunctionType.Spmd)
    def indexer_head_coefficients_spmd(
        self,
        position_ids__ssa_v0: pl.Tensor[[T_DYN], pl.INT32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)],
        coefficients_inline34__ssa_v0: pl.Out[pl.Tensor[[6144, 64], pl.FP16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 786432)]],
        query_scale__ssa_v0: pl.Tensor[[24576, 1], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 98304)],
        weights__ssa_v0: pl.Tensor[[384, 64], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 98304)],
    ) -> pl.Tensor[[6144, 64], pl.FP16]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        coefficients_inline34__rv_v2: pl.Tensor[[6144, 64], pl.FP16, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 786432)] = self.indexer_head_coefficients(
            position_ids__ssa_v0, coefficients_inline34__ssa_v0, query_scale__ssa_v0, weights__ssa_v0, attrs={"arg_directions": [pl.adir.input, pl.adir.output_existing, pl.adir.input, pl.adir.input]}
        )
        return coefficients_inline34__ssa_v0

    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def weights_proj_reduce(
        weights_partial_inline17__rv_v2: pl.Tensor[[384, 64], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 98304)],
        weights_inline14__ssa_v0: pl.Out[pl.Tensor[[384, 64], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 98304)]],
    ) -> pl.Tensor[[384, 64], pl.FP32]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        mem_vec_2: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 4096)
        mem_vec_3: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 2048)
        w_rb_inline16__ssa_v1: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        w_r0_inline13__ssa_v1: pl.Scalar[pl.INDEX] = w_rb_inline16__ssa_v1 * 16
        w_sum_inline2__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_2, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.load(
            weights_partial_inline17__rv_v2, [w_r0_inline13__ssa_v1, 0], [16, 64], [16, 64], target_memory=pl.Mem.Vec
        )
        t__tile: pl.Tile[[16, 64], pl.BF16, pl.MemRef(mem_vec_3, pl.const(4096, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(w_sum_inline2__tile, target_type=pl.BF16, mode="rint")
        w_sum_v1_inline21__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_2, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(t__tile, target_type=pl.FP32, mode="round")
        t__tile_1: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_2, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.muls(w_sum_v1_inline21__tile, 0.011048543456039806)
        w_scaled_inline0__tile: pl.Tile[[16, 64], pl.BF16, pl.MemRef(mem_vec_3, pl.const(4096, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(t__tile_1, target_type=pl.BF16, mode="rint")
        t__cast_fp32_tmp_v0: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_2, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(w_scaled_inline0__tile, target_type=pl.FP32, mode="round")
        t__tile_2: pl.Tile[[16, 64], pl.FP16, pl.MemRef(mem_vec_3, pl.const(4096, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(t__cast_fp32_tmp_v0, target_type=pl.FP16, mode="round")
        t__tile_3: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_2, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(t__tile_2, target_type=pl.FP32, mode="round")
        weights_inline14__tile: pl.Tensor[[384, 64], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 98304)] = pl.tile.store(t__tile_3, [w_r0_inline13__ssa_v1, 0], weights_inline14__ssa_v0)
        return weights_inline14__ssa_v0

    @pl.function(type=pl.FunctionType.Spmd)
    def weights_proj_reduce_spmd(
        self,
        weights_partial_inline17__rv_v2: pl.Tensor[[384, 64], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 98304)],
        weights_inline14__ssa_v0: pl.Out[pl.Tensor[[384, 64], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 98304)]],
    ) -> pl.Tensor[[384, 64], pl.FP32]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        weights_inline14__ssa_v1: pl.Tensor[[384, 64], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 98304)] = self.weights_proj_reduce(
            weights_partial_inline17__rv_v2, weights_inline14__ssa_v0, attrs={"arg_directions": [pl.adir.input, pl.adir.output_existing]}
        )
        return weights_inline14__ssa_v0

    @pl.function(type=pl.FunctionType.AIC, level=pl.Level.AIC, role=pl.Role.SubWorker)
    def weights_proj(
        weights_partial_inline17__ssa_v0: pl.Out[pl.Tensor[[384, 64], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 98304)]],
        row_blocks_inline11__ssa_v0: pl.Scalar[pl.INDEX],
        bs_inline8__ssa_v0: pl.Scalar[pl.INDEX],
        x_flat_inline12__ssa_v0: pl.Tensor[[T_DYN, 4096], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)],
        weights_proj__ssa_v0: pl.Tensor[[4096, 64], pl.BF16, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 524288)],
    ) -> pl.Tensor[[384, 64], pl.FP32]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        mem_acc_3: pl.Ptr = pl.tile.alloc(pl.Mem.Acc, 4096)
        mem_mat_4: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 16384)
        mem_mat_5: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 65536)
        mem_left_6: pl.Ptr = pl.tile.alloc(pl.Mem.Left, 8192)
        mem_right_7: pl.Ptr = pl.tile.alloc(pl.Mem.Right, 32768)
        mem_left_8: pl.Ptr = pl.tile.alloc(pl.Mem.Left, 8192)
        mem_right_9: pl.Ptr = pl.tile.alloc(pl.Mem.Right, 32768)
        w_worker_inline9__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        for w_unit_inline15__idx_v0, (weights_partial_inline17__iter_v1,) in pl.range(w_worker_inline9__ssa_v0, row_blocks_inline11__ssa_v0, 8, init_values=(weights_partial_inline17__ssa_v0,)):
            w_rb_inline16__ssa_v0: pl.Scalar[pl.INDEX] = w_unit_inline15__idx_v0
            kb_inline18__ssa_v0: pl.Scalar[pl.INDEX] = w_unit_inline15__idx_v0 - w_rb_inline16__ssa_v0
            w_r0_inline13__ssa_v0: pl.Scalar[pl.INDEX] = w_rb_inline16__ssa_v0 * 16
            w_rows_inline20__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(bs_inline8__ssa_v0 - w_r0_inline13__ssa_v0, 16)
            k_base_inline23__ssa_v0: pl.Scalar[pl.INDEX] = kb_inline18__ssa_v0 * 4096
            weights_acc_inline7__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 4096), pl.Mem.Acc] = pl.tile.create([16, 64], dtype=pl.FP32, target_memory=pl.Mem.Acc)
            for db_inline5__idx_v0, (weights_acc_inline7__iter_v1,) in pl.range(8, init_values=(weights_acc_inline7__tile,)):
                d0_inline4__ssa_v0: pl.Scalar[pl.INDEX] = k_base_inline23__ssa_v0 + db_inline5__idx_v0 * 512
                x_tile_inline22__tile: pl.Tile[[16, 512], pl.BF16, pl.MemRef(mem_mat_4, pl.const(0, pl.INT64), 16384), pl.Mem.Mat, pl.TileView(valid_shape=[w_rows_inline20__ssa_v0, 512])] = (
                    pl.tile.load(x_flat_inline12__ssa_v0, [w_r0_inline13__ssa_v0, d0_inline4__ssa_v0], [16, 512], [w_rows_inline20__ssa_v0, 512], target_memory=pl.Mem.Mat)
                )
                weights_proj_tile_inline3__tile: pl.Tile[[512, 64], pl.BF16, pl.MemRef(mem_mat_5, pl.const(16384, pl.INT64), 65536), pl.Mem.Mat] = pl.tile.load(
                    weights_proj__ssa_v0, [d0_inline4__ssa_v0, 0], [512, 64], [512, 64], target_memory=pl.Mem.Mat
                )
                weights_acc_inline7__tile_l0_a: pl.Tile[
                    [16, 256],
                    pl.BF16,
                    pl.MemRef(mem_left_6, pl.const(0, pl.INT64), 8192),
                    pl.Mem.Left,
                    pl.TileView(valid_shape=[w_rows_inline20__ssa_v0, 256], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal),
                ] = pl.tile.extract(x_tile_inline22__tile, 0, 0, [16, 256], target_memory=pl.Mem.Left)
                weights_acc_inline7__tile_l0_b: pl.Tile[[256, 64], pl.BF16, pl.MemRef(mem_right_7, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                    weights_proj_tile_inline3__tile, 0, 0, [256, 64], target_memory=pl.Mem.Right
                )
                weights_acc_inline7__tile_l0_a_1: pl.Tile[
                    [16, 256],
                    pl.BF16,
                    pl.MemRef(mem_left_8, pl.const(8192, pl.INT64), 8192),
                    pl.Mem.Left,
                    pl.TileView(valid_shape=[w_rows_inline20__ssa_v0, 256], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal),
                ] = pl.tile.extract(x_tile_inline22__tile, 0, 256, [16, 256], target_memory=pl.Mem.Left)
                weights_acc_inline7__tile_l0_b_1: pl.Tile[[256, 64], pl.BF16, pl.MemRef(mem_right_9, pl.const(32768, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                    weights_proj_tile_inline3__tile, 256, 0, [256, 64], target_memory=pl.Mem.Right
                )
                weights_acc_inline7__tile_l0_c_acc: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 4096), pl.Mem.Acc] = pl.tile.matmul_acc(
                    weights_acc_inline7__iter_v1, weights_acc_inline7__tile_l0_a, weights_acc_inline7__tile_l0_b, db_inline5__idx_v0 == 0
                )
                weights_acc_inline7__tile_l0_c_acc_1: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 4096), pl.Mem.Acc] = pl.tile.matmul_acc(
                    weights_acc_inline7__tile_l0_c_acc, weights_acc_inline7__tile_l0_a_1, weights_acc_inline7__tile_l0_b_1, False
                )
                weights_acc_inline7__rv_v2: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 4096), pl.Mem.Acc] = pl.yield_(weights_acc_inline7__tile_l0_c_acc_1)
            weights_partial_inline17__tile: pl.Tensor[[384, 64], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 98304)] = pl.tile.store(
                weights_acc_inline7__rv_v2, [kb_inline18__ssa_v0 * 384 + w_r0_inline13__ssa_v0, 0], weights_partial_inline17__iter_v1
            )
            weights_partial_inline17__rv_v2: pl.Tensor[[384, 64], pl.FP32, pl.MemRef("mem_ddr_12", pl.const(0, pl.INT64), 98304)] = pl.yield_(weights_partial_inline17__tile)
        return weights_partial_inline17__ssa_v0

    @pl.function(type=pl.FunctionType.Spmd)
    def weights_proj_spmd(
        self,
        weights_partial_inline17__ssa_v0: pl.Out[pl.Tensor[[384, 64], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 98304)]],
        row_blocks_inline11__ssa_v0: pl.Scalar[pl.INDEX],
        bs_inline8__ssa_v0: pl.Scalar[pl.INDEX],
        x_flat_inline12__ssa_v0: pl.Tensor[[T_DYN, 4096], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)],
        weights_proj__ssa_v0: pl.Tensor[[4096, 64], pl.BF16, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 524288)],
    ) -> pl.Tensor[[384, 64], pl.FP32]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        weights_partial_inline17__rv_v2: pl.Tensor[[384, 64], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 98304)] = self.weights_proj(
            weights_partial_inline17__ssa_v0,
            row_blocks_inline11__ssa_v0,
            bs_inline8__ssa_v0,
            x_flat_inline12__ssa_v0,
            weights_proj__ssa_v0,
            attrs={"arg_directions": [pl.adir.output_existing, pl.adir.scalar, pl.adir.scalar, pl.adir.input, pl.adir.input]},
        )
        return weights_partial_inline17__ssa_v0

    @pl.function(type=pl.FunctionType.Orchestration, level=pl.Level.CHIP, role=pl.Role.Orchestrator, auto_scope=False)
    def diagnose_indexer_weights(
        self,
        x__ssa_v0: pl.Tensor[[T_DYN, 4096], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)],
        weights_proj__ssa_v0: pl.Tensor[[4096, 64], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 524288)],
        position_ids__ssa_v0: pl.Tensor[[T_DYN], pl.INT32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)],
        query_scale__ssa_v0: pl.Tensor[[24576, 1], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 98304)],
        published_weights__ssa_v0: pl.Out[pl.Tensor[[384, 64], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 98304)]],
        published_coefficients__ssa_v0: pl.Out[pl.Tensor[[384, 64], pl.FP16, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 49152)]],
    ) -> tuple[pl.Tensor[[384, 64], pl.FP32], pl.Tensor[[384, 64], pl.FP16]]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        gate__ssa_v0: pl.Scalar[pl.TASK_ID] = pl.system.task_dummy(deps=[])
        bs_inline8__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(x__ssa_v0, 0)
        row_blocks_inline11__ssa_v0: pl.Scalar[pl.INDEX] = (bs_inline8__ssa_v0 + 15) // 16
        x_flat_inline12__ssa_v0: pl.Tensor[[T_DYN, 4096], pl.BF16, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 0)] = x__ssa_v0
        weights_inline14__ssa_v0: pl.Tensor[[384, 64], pl.FP32, pl.MemRef("mem_ddr_7", pl.const(0, pl.INT64), 98304)] = pl.tensor.create([384, 64], dtype=pl.FP32, layout=pl.TensorLayout.ND)
        weights_partial_inline17__ssa_v0: pl.Tensor[[384, 64], pl.FP32, pl.MemRef("mem_ddr_8", pl.const(0, pl.INT64), 98304)] = pl.tensor.create([384, 64], dtype=pl.FP32, layout=pl.TensorLayout.ND)
        ret__tmp_v0: pl.Tuple[pl.Tensor[[384, 64], pl.FP32], pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(
            self.weights_proj_spmd,
            weights_partial_inline17__ssa_v0,
            row_blocks_inline11__ssa_v0,
            bs_inline8__ssa_v0,
            x_flat_inline12__ssa_v0,
            weights_proj__ssa_v0,
            deps=[gate__ssa_v0],
            core_num=8,
            allow_early_resolve=True,
            attrs={"arg_directions": [pl.adir.output_existing, pl.adir.scalar, pl.adir.scalar, pl.adir.input, pl.adir.input]},
        )
        weights_partial_inline17__rv_v2: pl.Tensor[[384, 64], pl.FP32, pl.MemRef("mem_ddr_9", pl.const(0, pl.INT64), 98304)] = ret__tmp_v0[0]
        _weights_tid_inline19__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0[1]
        ret__tmp_v0_1: pl.Tuple[pl.Tensor[[384, 64], pl.FP32], pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(
            self.weights_proj_reduce_spmd,
            weights_partial_inline17__rv_v2,
            weights_inline14__ssa_v0,
            core_num=row_blocks_inline11__ssa_v0,
            allow_early_resolve=True,
            attrs={"arg_directions": [pl.adir.input, pl.adir.output_existing]},
        )
        weights_inline14__ssa_v1: pl.Tensor[[384, 64], pl.FP32, pl.MemRef("mem_ddr_10", pl.const(0, pl.INT64), 98304)] = ret__tmp_v0_1[0]
        weights_tid_inline6__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_1[1]
        weights__ssa_v0: pl.Tensor[[384, 64], pl.FP32, pl.MemRef("mem_ddr_11", pl.const(0, pl.INT64), 98304)] = weights_inline14__ssa_v1
        weights_tid__ssa_v0: pl.Scalar[pl.TASK_ID] = weights_tid_inline6__ssa_v0
        coefficients_inline34__ssa_v0: pl.Tensor[[6144, 64], pl.FP16, pl.MemRef("mem_ddr_12", pl.const(0, pl.INT64), 786432)] = pl.tensor.create([6144, 64], dtype=pl.FP16, layout=pl.TensorLayout.ND)
        ret__tmp_v0_2: pl.Tuple[pl.Tensor[[6144, 64], pl.FP16], pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(
            self.indexer_head_coefficients_spmd,
            position_ids__ssa_v0,
            coefficients_inline34__ssa_v0,
            query_scale__ssa_v0,
            weights__ssa_v0,
            deps=[gate__ssa_v0, weights_tid__ssa_v0],
            core_num=48,
            allow_early_resolve=True,
            attrs={"arg_directions": [pl.adir.input, pl.adir.output_existing, pl.adir.input, pl.adir.input]},
        )
        coefficients_inline34__rv_v2: pl.Tensor[[6144, 64], pl.FP16, pl.MemRef("mem_ddr_13", pl.const(0, pl.INT64), 786432)] = ret__tmp_v0_2[0]
        coefficients_tid_inline31__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_2[1]
        coefficients__ssa_v0: pl.Tensor[[6144, 64], pl.FP16, pl.MemRef("mem_ddr_14", pl.const(0, pl.INT64), 786432)] = coefficients_inline34__rv_v2
        coefficients_tid__ssa_v0: pl.Scalar[pl.TASK_ID] = coefficients_tid_inline31__ssa_v0
        ret__tmp_v0_3: pl.Tuple[pl.Tensor[[384, 64], pl.FP32], pl.Tensor[[384, 64], pl.FP16], pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(
            self.diagnose_indexer_weights_spmd_3,
            weights__ssa_v0,
            published_weights__ssa_v0,
            coefficients__ssa_v0,
            published_coefficients__ssa_v0,
            deps=[coefficients_tid__ssa_v0],
            core_num=T_DYN,
            attrs={"arg_directions": [pl.adir.input, pl.adir.output_existing, pl.adir.input, pl.adir.output_existing]},
        )
        published_weights__ssa_v1: pl.Tensor[[384, 64], pl.FP32, pl.MemRef("mem_ddr_15", pl.const(0, pl.INT64), 98304)] = ret__tmp_v0_3[0]
        published_coefficients__ssa_v1: pl.Tensor[[384, 64], pl.FP16, pl.MemRef("mem_ddr_16", pl.const(0, pl.INT64), 49152)] = ret__tmp_v0_3[1]
        tid__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_3[2]
        return published_weights__ssa_v1, published_coefficients__ssa_v1