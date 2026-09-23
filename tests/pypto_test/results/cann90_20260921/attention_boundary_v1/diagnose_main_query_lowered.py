# pypto.program: _jit_diagnose_main_query
import pypto.language as pl

T_DYN = pl.dynamic("T_DYN")


@pl.program
class _jit_diagnose_main_query:
    @pl.function(type=pl.FunctionType.AIC, level=pl.Level.AIC, role=pl.Role.SubWorker)
    def kv_proj_matmul(
        kv_m_groups_inline875__ssa_v0: pl.Scalar[pl.INDEX],
        kv_fp32_inline882__rv_v2: pl.InOut[pl.Tensor[[t_matmul_inline907__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)]],
        kv_full_rows_inline898__ssa_v0: pl.Scalar[pl.INDEX],
        tile_base_inline876__idx_v0: pl.Scalar[pl.INDEX],
        x_view_inline872__ssa_v0: pl.Tensor[[t_dim_inline868__ssa_v0, 4096], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)],
        wkv__ssa_v0: pl.Tensor[[4096, 512], pl.BF16, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 4194304)],
        t_matmul_inline907__ssa_v0: pl.Scalar[pl.INDEX],
        tile_rows_inline888__ssa_v0: pl.Scalar[pl.INDEX],
    ) -> pl.Tensor[[t_matmul_inline907__ssa_v0, 512], pl.FP32]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        mem_acc_3: pl.Ptr = pl.tile.alloc(pl.Mem.Acc, 32768)
        mem_mat_4: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 32768)
        mem_mat_5: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 65536)
        mem_mat_6: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 32768)
        mem_mat_7: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 65536)
        mem_left_8: pl.Ptr = pl.tile.alloc(pl.Mem.Left, 16384)
        mem_right_9: pl.Ptr = pl.tile.alloc(pl.Mem.Right, 32768)
        mem_left_10: pl.Ptr = pl.tile.alloc(pl.Mem.Left, 16384)
        mem_right_11: pl.Ptr = pl.tile.alloc(pl.Mem.Right, 32768)
        mem_left_13: pl.Ptr = pl.tile.alloc(pl.Mem.Left, 16384)
        mem_left_15: pl.Ptr = pl.tile.alloc(pl.Mem.Left, 16384)
        kbg_inline893__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        kv_col0_inline897__ssa_v0: pl.Scalar[pl.INDEX] = kbg_inline893__ssa_v0 // (kv_m_groups_inline875__ssa_v0 * 2) * 128
        kv_k_base_inline866__ssa_v0: pl.Scalar[pl.INDEX] = kbg_inline893__ssa_v0 // kv_m_groups_inline875__ssa_v0 % 2 * 2048
        kv_m_group_inline855__ssa_v0: pl.Scalar[pl.INDEX] = kbg_inline893__ssa_v0 % kv_m_groups_inline875__ssa_v0
        for dense_t0_inline902__idx_v0, (kv_fp32_inline882__iter_v6,) in pl.range(
            kv_m_group_inline855__ssa_v0 * 64, kv_full_rows_inline898__ssa_v0, kv_m_groups_inline875__ssa_v0 * 64, init_values=(kv_fp32_inline882__rv_v2,)
        ):
            dense_x0_inline892__ssa_v0: pl.Scalar[pl.INDEX] = tile_base_inline876__idx_v0 + dense_t0_inline902__idx_v0
            dense_acc_inline862__tile: pl.Tile[[64, 128], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 32768), pl.Mem.Acc] = pl.tile.create([64, 128], dtype=pl.FP32, target_memory=pl.Mem.Acc)
            for dense_k_inline870__idx_v0, (dense_acc_inline862__iter_v1,) in pl.range(0, 8, 2, init_values=(dense_acc_inline862__tile,)):
                dense_d0_inline885__ssa_v0: pl.Scalar[pl.INDEX] = kv_k_base_inline866__ssa_v0 + dense_k_inline870__idx_v0 * 256
                dense_d0_inline885__ssa_v0_1: pl.Scalar[pl.INDEX] = kv_k_base_inline866__ssa_v0 + (dense_k_inline870__idx_v0 * 256 + 256)
                dense_x_inline864__tile: pl.Tile[[64, 256], pl.BF16, pl.MemRef(mem_mat_4, pl.const(0, pl.INT64), 32768), pl.Mem.Mat] = pl.tile.load(
                    x_view_inline872__ssa_v0, [dense_x0_inline892__ssa_v0, dense_d0_inline885__ssa_v0], [64, 256], [64, 256], target_memory=pl.Mem.Mat
                )
                dense_w_inline879__tile: pl.Tile[[256, 128], pl.BF16, pl.MemRef(mem_mat_5, pl.const(32768, pl.INT64), 65536), pl.Mem.Mat] = pl.tile.load(
                    wkv__ssa_v0, [dense_d0_inline885__ssa_v0, kv_col0_inline897__ssa_v0], [256, 128], [256, 128], target_memory=pl.Mem.Mat
                )
                dense_x_inline864__tile_1: pl.Tile[[64, 256], pl.BF16, pl.MemRef(mem_mat_6, pl.const(98304, pl.INT64), 32768), pl.Mem.Mat] = pl.tile.load(
                    x_view_inline872__ssa_v0, [dense_x0_inline892__ssa_v0, dense_d0_inline885__ssa_v0_1], [64, 256], [64, 256], target_memory=pl.Mem.Mat
                )
                dense_w_inline879__tile_1: pl.Tile[[256, 128], pl.BF16, pl.MemRef(mem_mat_7, pl.const(131072, pl.INT64), 65536), pl.Mem.Mat] = pl.tile.load(
                    wkv__ssa_v0, [dense_d0_inline885__ssa_v0_1, kv_col0_inline897__ssa_v0], [256, 128], [256, 128], target_memory=pl.Mem.Mat
                )
                dense_acc_inline862__tile_l0_a: pl.Tile[[64, 128], pl.BF16, pl.MemRef(mem_left_8, pl.const(49152, pl.INT64), 16384), pl.Mem.Left, pl.TileView(blayout=pl.TileLayout.row_major)] = (
                    pl.tile.extract(dense_x_inline864__tile, 0, 0, [64, 128], target_memory=pl.Mem.Left)
                )
                dense_acc_inline862__tile_l0_b: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_9, pl.const(32768, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                    dense_w_inline879__tile, 0, 0, [128, 128], target_memory=pl.Mem.Right
                )
                dense_acc_inline862__tile_l0_a_1: pl.Tile[[64, 128], pl.BF16, pl.MemRef(mem_left_10, pl.const(0, pl.INT64), 16384), pl.Mem.Left, pl.TileView(blayout=pl.TileLayout.row_major)] = (
                    pl.tile.extract(dense_x_inline864__tile, 0, 128, [64, 128], target_memory=pl.Mem.Left)
                )
                dense_acc_inline862__tile_l0_b_1: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_11, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                    dense_w_inline879__tile, 128, 0, [128, 128], target_memory=pl.Mem.Right
                )
                dense_acc_inline862__tile_l0_c_acc: pl.Tile[[64, 128], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 32768), pl.Mem.Acc] = pl.tile.matmul_acc(
                    dense_acc_inline862__iter_v1, dense_acc_inline862__tile_l0_a, dense_acc_inline862__tile_l0_b, dense_k_inline870__idx_v0 == 0
                )
                dense_acc_inline862__tile_l0_c_acc_1: pl.Tile[[64, 128], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 32768), pl.Mem.Acc] = pl.tile.matmul_acc(
                    dense_acc_inline862__tile_l0_c_acc, dense_acc_inline862__tile_l0_a_1, dense_acc_inline862__tile_l0_b_1, False
                )
                dense_acc_inline862__tile_l0_a_2: pl.Tile[[64, 128], pl.BF16, pl.MemRef(mem_left_13, pl.const(16384, pl.INT64), 16384), pl.Mem.Left, pl.TileView(blayout=pl.TileLayout.row_major)] = (
                    pl.tile.extract(dense_x_inline864__tile_1, 0, 0, [64, 128], target_memory=pl.Mem.Left)
                )
                dense_acc_inline862__tile_l0_b_2: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_9, pl.const(32768, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                    dense_w_inline879__tile_1, 0, 0, [128, 128], target_memory=pl.Mem.Right
                )
                dense_acc_inline862__tile_l0_a_3: pl.Tile[[64, 128], pl.BF16, pl.MemRef(mem_left_15, pl.const(32768, pl.INT64), 16384), pl.Mem.Left, pl.TileView(blayout=pl.TileLayout.row_major)] = (
                    pl.tile.extract(dense_x_inline864__tile_1, 0, 128, [64, 128], target_memory=pl.Mem.Left)
                )
                dense_acc_inline862__tile_l0_b_3: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_11, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                    dense_w_inline879__tile_1, 128, 0, [128, 128], target_memory=pl.Mem.Right
                )
                dense_acc_inline862__tile_l0_c_acc_2: pl.Tile[[64, 128], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 32768), pl.Mem.Acc] = pl.tile.matmul_acc(
                    dense_acc_inline862__tile_l0_c_acc_1, dense_acc_inline862__tile_l0_a_2, dense_acc_inline862__tile_l0_b_2, dense_k_inline870__idx_v0 == -1
                )
                dense_acc_inline862__tile_l0_c_acc_3: pl.Tile[[64, 128], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 32768), pl.Mem.Acc] = pl.tile.matmul_acc(
                    dense_acc_inline862__tile_l0_c_acc_2, dense_acc_inline862__tile_l0_a_3, dense_acc_inline862__tile_l0_b_3, False
                )
                dense_acc_inline862__rv_v2: pl.Tile[[64, 128], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 32768), pl.Mem.Acc] = pl.yield_(dense_acc_inline862__tile_l0_c_acc_3)
            kv_fp32_inline882__tile: pl.Tensor[[t_matmul_inline907__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                dense_acc_inline862__rv_v2, [dense_t0_inline902__idx_v0, kv_col0_inline897__ssa_v0], kv_fp32_inline882__iter_v6, atomic=pl.AtomicType.Add
            )
            kv_fp32_inline882__rv_v7: pl.Tensor[[t_matmul_inline907__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_19", pl.const(0, pl.INT64), 0)] = pl.yield_(kv_fp32_inline882__tile)
        for t0_inline861__idx_v0, (kv_fp32_inline882__iter_v9,) in pl.range(
            kv_full_rows_inline898__ssa_v0 + kv_m_group_inline855__ssa_v0 * 16, t_matmul_inline907__ssa_v0, kv_m_groups_inline875__ssa_v0 * 16, init_values=(kv_fp32_inline882__rv_v7,)
        ):
            kv_acc_inline859__tile: pl.Tile[[16, 128], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 8192), pl.Mem.Acc] = pl.tile.create([16, 128], dtype=pl.FP32, target_memory=pl.Mem.Acc)
            for db_inline891__idx_v0, (kv_acc_inline859__iter_v1,) in pl.range(0, 8, 2, init_values=(kv_acc_inline859__tile,)):
                d0_inline858__ssa_v0: pl.Scalar[pl.INDEX] = kv_k_base_inline866__ssa_v0 + db_inline891__idx_v0 * 256
                kv_rows_inline857__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(tile_rows_inline888__ssa_v0 - t0_inline861__idx_v0, 16)
                x_t0_inline852__ssa_v0: pl.Scalar[pl.INDEX] = tile_base_inline876__idx_v0 + t0_inline861__idx_v0
                d0_inline858__ssa_v0_1: pl.Scalar[pl.INDEX] = kv_k_base_inline866__ssa_v0 + (db_inline891__idx_v0 * 256 + 256)
                kv_rows_inline857__ssa_v0_1: pl.Scalar[pl.INDEX] = pl.min(tile_rows_inline888__ssa_v0 - t0_inline861__idx_v0, 16)
                x_t0_inline852__ssa_v0_1: pl.Scalar[pl.INDEX] = tile_base_inline876__idx_v0 + t0_inline861__idx_v0
                kv_x_chunk_bf16_inline896__tile: pl.Tile[
                    [16, 256], pl.BF16, pl.MemRef(mem_mat_4, pl.const(0, pl.INT64), 8192), pl.Mem.Mat, pl.TileView(valid_shape=[kv_rows_inline857__ssa_v0, 256])
                ] = pl.tile.load(x_view_inline872__ssa_v0, [x_t0_inline852__ssa_v0, d0_inline858__ssa_v0], [16, 256], [kv_rows_inline857__ssa_v0, 256], target_memory=pl.Mem.Mat)
                wkv_chunk_inline889__tile: pl.Tile[[256, 128], pl.BF16, pl.MemRef(mem_mat_5, pl.const(32768, pl.INT64), 65536), pl.Mem.Mat] = pl.tile.load(
                    wkv__ssa_v0, [d0_inline858__ssa_v0, kv_col0_inline897__ssa_v0], [256, 128], [256, 128], target_memory=pl.Mem.Mat
                )
                kv_x_chunk_bf16_inline896__tile_1: pl.Tile[
                    [16, 256], pl.BF16, pl.MemRef(mem_mat_6, pl.const(98304, pl.INT64), 8192), pl.Mem.Mat, pl.TileView(valid_shape=[kv_rows_inline857__ssa_v0_1, 256])
                ] = pl.tile.load(x_view_inline872__ssa_v0, [x_t0_inline852__ssa_v0_1, d0_inline858__ssa_v0_1], [16, 256], [kv_rows_inline857__ssa_v0_1, 256], target_memory=pl.Mem.Mat)
                wkv_chunk_inline889__tile_1: pl.Tile[[256, 128], pl.BF16, pl.MemRef(mem_mat_7, pl.const(131072, pl.INT64), 65536), pl.Mem.Mat] = pl.tile.load(
                    wkv__ssa_v0, [d0_inline858__ssa_v0_1, kv_col0_inline897__ssa_v0], [256, 128], [256, 128], target_memory=pl.Mem.Mat
                )
                kv_acc_inline859__tile_l0_a: pl.Tile[
                    [16, 128],
                    pl.BF16,
                    pl.MemRef(mem_left_8, pl.const(49152, pl.INT64), 4096),
                    pl.Mem.Left,
                    pl.TileView(valid_shape=[kv_rows_inline857__ssa_v0, 128], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal),
                ] = pl.tile.extract(kv_x_chunk_bf16_inline896__tile, 0, 0, [16, 128], target_memory=pl.Mem.Left)
                kv_acc_inline859__tile_l0_b: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_9, pl.const(32768, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                    wkv_chunk_inline889__tile, 0, 0, [128, 128], target_memory=pl.Mem.Right
                )
                kv_acc_inline859__tile_l0_a_1: pl.Tile[
                    [16, 128],
                    pl.BF16,
                    pl.MemRef(mem_left_10, pl.const(0, pl.INT64), 4096),
                    pl.Mem.Left,
                    pl.TileView(valid_shape=[kv_rows_inline857__ssa_v0, 128], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal),
                ] = pl.tile.extract(kv_x_chunk_bf16_inline896__tile, 0, 128, [16, 128], target_memory=pl.Mem.Left)
                kv_acc_inline859__tile_l0_b_1: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_11, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                    wkv_chunk_inline889__tile, 128, 0, [128, 128], target_memory=pl.Mem.Right
                )
                kv_acc_inline859__tile_l0_c_acc: pl.Tile[[16, 128], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 8192), pl.Mem.Acc] = pl.tile.matmul_acc(
                    kv_acc_inline859__iter_v1, kv_acc_inline859__tile_l0_a, kv_acc_inline859__tile_l0_b, db_inline891__idx_v0 == 0
                )
                kv_acc_inline859__tile_l0_c_acc_1: pl.Tile[[16, 128], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 8192), pl.Mem.Acc] = pl.tile.matmul_acc(
                    kv_acc_inline859__tile_l0_c_acc, kv_acc_inline859__tile_l0_a_1, kv_acc_inline859__tile_l0_b_1, False
                )
                kv_acc_inline859__tile_l0_a_2: pl.Tile[
                    [16, 128],
                    pl.BF16,
                    pl.MemRef(mem_left_13, pl.const(16384, pl.INT64), 4096),
                    pl.Mem.Left,
                    pl.TileView(valid_shape=[kv_rows_inline857__ssa_v0_1, 128], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal),
                ] = pl.tile.extract(kv_x_chunk_bf16_inline896__tile_1, 0, 0, [16, 128], target_memory=pl.Mem.Left)
                kv_acc_inline859__tile_l0_b_2: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_9, pl.const(32768, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                    wkv_chunk_inline889__tile_1, 0, 0, [128, 128], target_memory=pl.Mem.Right
                )
                kv_acc_inline859__tile_l0_a_3: pl.Tile[
                    [16, 128],
                    pl.BF16,
                    pl.MemRef(mem_left_15, pl.const(32768, pl.INT64), 4096),
                    pl.Mem.Left,
                    pl.TileView(valid_shape=[kv_rows_inline857__ssa_v0_1, 128], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal),
                ] = pl.tile.extract(kv_x_chunk_bf16_inline896__tile_1, 0, 128, [16, 128], target_memory=pl.Mem.Left)
                kv_acc_inline859__tile_l0_b_3: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_11, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                    wkv_chunk_inline889__tile_1, 128, 0, [128, 128], target_memory=pl.Mem.Right
                )
                kv_acc_inline859__tile_l0_c_acc_2: pl.Tile[[16, 128], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 8192), pl.Mem.Acc] = pl.tile.matmul_acc(
                    kv_acc_inline859__tile_l0_c_acc_1, kv_acc_inline859__tile_l0_a_2, kv_acc_inline859__tile_l0_b_2, db_inline891__idx_v0 == -1
                )
                kv_acc_inline859__tile_l0_c_acc_3: pl.Tile[[16, 128], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 8192), pl.Mem.Acc] = pl.tile.matmul_acc(
                    kv_acc_inline859__tile_l0_c_acc_2, kv_acc_inline859__tile_l0_a_3, kv_acc_inline859__tile_l0_b_3, False
                )
                kv_acc_inline859__rv_v2: pl.Tile[[16, 128], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 8192), pl.Mem.Acc] = pl.yield_(kv_acc_inline859__tile_l0_c_acc_3)
            kv_fp32_inline882__tile_1: pl.Tensor[[t_matmul_inline907__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_19", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                kv_acc_inline859__rv_v2, [t0_inline861__idx_v0, kv_col0_inline897__ssa_v0], kv_fp32_inline882__iter_v9, atomic=pl.AtomicType.Add
            )
            kv_fp32_inline882__rv_v10: pl.Tensor[[t_matmul_inline907__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_36", pl.const(0, pl.INT64), 0)] = pl.yield_(kv_fp32_inline882__tile_1)
        return kv_fp32_inline882__rv_v2

    @pl.function(type=pl.FunctionType.Spmd)
    def kv_proj_matmul_spmd(
        self,
        kv_m_groups_inline875__ssa_v0: pl.Scalar[pl.INDEX],
        kv_fp32_inline882__rv_v2: pl.InOut[pl.Tensor[[t_matmul_inline907__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)]],
        kv_full_rows_inline898__ssa_v0: pl.Scalar[pl.INDEX],
        tile_base_inline876__idx_v0: pl.Scalar[pl.INDEX],
        x_view_inline872__ssa_v0: pl.Tensor[[t_dim_inline868__ssa_v0, 4096], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)],
        wkv__ssa_v0: pl.Tensor[[4096, 512], pl.BF16, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 4194304)],
        t_matmul_inline907__ssa_v0: pl.Scalar[pl.INDEX],
        tile_rows_inline888__ssa_v0: pl.Scalar[pl.INDEX],
    ) -> pl.Tensor[[t_matmul_inline907__ssa_v0, 512], pl.FP32]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        kv_fp32_inline882__rv_v10: pl.Tensor[[t_matmul_inline907__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)] = self.kv_proj_matmul(
            kv_m_groups_inline875__ssa_v0,
            kv_fp32_inline882__rv_v2,
            kv_full_rows_inline898__ssa_v0,
            tile_base_inline876__idx_v0,
            x_view_inline872__ssa_v0,
            wkv__ssa_v0,
            t_matmul_inline907__ssa_v0,
            tile_rows_inline888__ssa_v0,
            attrs={"arg_directions": [pl.adir.scalar, pl.adir.inout, pl.adir.scalar, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.scalar, pl.adir.scalar]},
        )
        return kv_fp32_inline882__rv_v2

    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def kv_proj_seed(
        kv_fp32_inline882__ssa_v0: pl.Out[pl.Tensor[[t_matmul_inline907__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)]], t_matmul_inline907__ssa_v0: pl.Scalar[pl.INDEX]
    ) -> pl.Tensor[[t_matmul_inline907__ssa_v0, 512], pl.FP32]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        mem_vec_1: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 8192)
        for kts0_inline881__idx_v0, (kv_fp32_inline882__iter_v1,) in pl.range(0, t_matmul_inline907__ssa_v0, 16, init_values=(kv_fp32_inline882__ssa_v0,)):
            for kvseed0_inline890__idx_v0, (kv_fp32_inline882__iter_v3,) in pl.range(0, 512, 128, init_values=(kv_fp32_inline882__iter_v1,)):
                kv_seed_inline878__tile: pl.Tile[[16, 128], pl.FP32, pl.MemRef(mem_vec_1, pl.const(0, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.full([16, 128], dtype=pl.FP32, value=0.0)
                kv_fp32_inline882__tile: pl.Tensor[[t_matmul_inline907__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                    kv_seed_inline878__tile, [kts0_inline881__idx_v0, kvseed0_inline890__idx_v0], kv_fp32_inline882__iter_v3
                )
                kv_fp32_inline882__rv_v4: pl.Tensor[[t_matmul_inline907__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)] = pl.yield_(kv_fp32_inline882__tile)
            kv_fp32_inline882__rv_v2: pl.Tensor[[t_matmul_inline907__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)] = pl.yield_(kv_fp32_inline882__rv_v4)
        return kv_fp32_inline882__ssa_v0

    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def kv_rms_norm_rope(
        tile_rows_inline888__ssa_v0: pl.Scalar[pl.INDEX],
        tile_base_inline876__idx_v0: pl.Scalar[pl.INDEX],
        kv_fp32_inline882__rv_v10: pl.Tensor[[t_matmul_inline907__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)],
        kv_view_inline900__ssa_v0: pl.InOut[pl.Tensor[[t_dim_inline868__ssa_v0, 512], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)]],
        gamma_ckv__ssa_v0: pl.Tensor[[512], pl.BF16, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 1024)],
        q_rope_cos_il_inline361__ssa_v0: pl.Tensor[[t_dim_inline362__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)],
        q_rope_sin_signed_inline360__ssa_v0: pl.Tensor[[t_dim_inline362__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 0)],
        q_rope_swap_idx_inline359__ssa_v0: pl.Tensor[[t_dim_inline362__ssa_v0, 64], pl.INT32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 0)],
    ) -> pl.Tensor[[t_dim_inline868__ssa_v0, 512], pl.BF16]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        mem_vec_11: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 8192)
        mem_vec_12: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
        mem_vec_17: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 8192)
        mem_vec_18: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
        mem_vec_64: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 8192)
        mem_vec_145: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 8192)
        mem_vec_146: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 8192)
        tg_idx_inline910__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        tg_inline911__ssa_v0: pl.Scalar[pl.INDEX] = tg_idx_inline910__ssa_v0 * 32
        valid_rows_inline923__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(tile_rows_inline888__ssa_v0 - tg_inline911__ssa_v0, 32)
        out_tg_inline905__ssa_v0: pl.Scalar[pl.INDEX] = tile_base_inline876__idx_v0 + tg_inline911__ssa_v0
        if valid_rows_inline923__ssa_v0 == 32:
            kv_sq_sum_inline887__tile: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_64, pl.const(65536, pl.INT64), 128), pl.Mem.Vec] = pl.tile.full([1, 32], dtype=pl.FP32, value=0.0)
            for kv_sq_col0_inline915__idx_v0, (kv_sq_sum_inline887__iter_v1,) in pl.range(0, 512, 128, init_values=(kv_sq_sum_inline887__tile,)):
                kv_chunk_inline895__tile: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(8192, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.load(
                    kv_fp32_inline882__rv_v10, [tg_inline911__ssa_v0, kv_sq_col0_inline915__idx_v0], [32, 64], [32, 64], target_memory=pl.Mem.Vec
                )
                kv_chunk_inline895__tile_1: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_18, pl.const(49152, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.load(
                    kv_fp32_inline882__rv_v10, [tg_inline911__ssa_v0, kv_sq_col0_inline915__idx_v0 + 64], [32, 64], [32, 64], target_memory=pl.Mem.Vec
                )
                t__tile: pl.Tile[[32, 64], pl.BF16, pl.MemRef(mem_vec_11, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(kv_chunk_inline895__tile, target_type=pl.BF16, mode="rint")
                kv_chunk_v1_inline873__tile: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(8192, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.cast(t__tile, target_type=pl.FP32, mode="round")
                kv_sq_inline901__tile: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_11, pl.const(0, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.mul(
                    kv_chunk_v1_inline873__tile, kv_chunk_v1_inline873__tile
                )
                tmp_tile: pl.Tile[[32, 128], pl.FP32, pl.MemRef(mem_vec_12, pl.const(8192, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.create([32, 128], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                t__tile_1: pl.Tile[[32, 1], pl.FP32, pl.MemRef(mem_vec_145, pl.const(24576, pl.INT64), 128), pl.Mem.Vec] = pl.tile.row_sum(kv_sq_inline901__tile, tmp_tile)
                kv_row_sum_inline894__tile: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_145, pl.const(24576, pl.INT64), 128), pl.Mem.Vec] = pl.tile.reshape(t__tile_1, [1, 32])
                kv_sq_sum_inline887__tile_1: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_12, pl.const(8192, pl.INT64), 128), pl.Mem.Vec] = pl.tile.add(
                    kv_sq_sum_inline887__iter_v1, kv_row_sum_inline894__tile
                )
                t__tile_2: pl.Tile[[32, 64], pl.BF16, pl.MemRef(mem_vec_17, pl.const(40960, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(kv_chunk_inline895__tile_1, target_type=pl.BF16, mode="rint")
                kv_chunk_v1_inline873__tile_1: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_18, pl.const(49152, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.cast(
                    t__tile_2, target_type=pl.FP32, mode="round"
                )
                kv_sq_inline901__tile_1: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_17, pl.const(40960, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.mul(
                    kv_chunk_v1_inline873__tile_1, kv_chunk_v1_inline873__tile_1
                )
                tmp_tile_1: pl.Tile[[32, 128], pl.FP32, pl.MemRef(mem_vec_18, pl.const(49152, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.create([32, 128], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                t__tile_3: pl.Tile[[32, 1], pl.FP32, pl.MemRef(mem_vec_146, pl.const(32768, pl.INT64), 128), pl.Mem.Vec] = pl.tile.row_sum(kv_sq_inline901__tile_1, tmp_tile_1)
                kv_row_sum_inline894__tile_1: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_146, pl.const(32768, pl.INT64), 128), pl.Mem.Vec] = pl.tile.reshape(t__tile_3, [1, 32])
                kv_sq_sum_inline887__tile_2: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_64, pl.const(65536, pl.INT64), 128), pl.Mem.Vec] = pl.tile.add(
                    kv_sq_sum_inline887__tile_1, kv_row_sum_inline894__tile_1
                )
                kv_sq_sum_inline887__rv_v2: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_64, pl.const(65536, pl.INT64), 128), pl.Mem.Vec] = pl.yield_(kv_sq_sum_inline887__tile_2)
            t__tile_4: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_12, pl.const(8192, pl.INT64), 128), pl.Mem.Vec] = pl.tile.muls(kv_sq_sum_inline887__rv_v2, 0.001953125)
            t__tile_5: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_12, pl.const(8192, pl.INT64), 128), pl.Mem.Vec] = pl.tile.adds(t__tile_4, 9.9999999999999995e-07)
            rsqrt_tmp: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_18, pl.const(49152, pl.INT64), 128), pl.Mem.Vec] = pl.tile.create([1, 32], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            kv_inv_rms_inline877__tile: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_64, pl.const(65536, pl.INT64), 128), pl.Mem.Vec] = pl.tile.rsqrt(t__tile_5, rsqrt_tmp)
            kv_inv_rms_t_inline917__tile: pl.Tile[[32, 1], pl.FP32, pl.MemRef(mem_vec_64, pl.const(65536, pl.INT64), 128), pl.Mem.Vec] = pl.tile.reshape(kv_inv_rms_inline877__tile, [32, 1])
            for n0_inline899__idx_v0, (kv_view_inline900__iter_v1,) in pl.range(0, 384, 128, init_values=(kv_view_inline900__ssa_v0,)):
                kv_chunk_inline895__tile_2: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(8192, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.load(
                    kv_fp32_inline882__rv_v10, [tg_inline911__ssa_v0, n0_inline899__idx_v0], [32, 64], [32, 64], target_memory=pl.Mem.Vec
                )
                t__tile_6: pl.Tile[[64], pl.BF16, pl.MemRef(mem_vec_145, pl.const(24576, pl.INT64), 128), pl.Mem.Vec] = pl.tile.load(
                    gamma_ckv__ssa_v0, [n0_inline899__idx_v0], [64], [64], target_memory=pl.Mem.Vec
                )
                kv_chunk_inline895__tile_3: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_18, pl.const(49152, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.load(
                    kv_fp32_inline882__rv_v10, [tg_inline911__ssa_v0, n0_inline899__idx_v0 + 64], [32, 64], [32, 64], target_memory=pl.Mem.Vec
                )
                t__tile_7: pl.Tile[[64], pl.BF16, pl.MemRef(mem_vec_146, pl.const(32768, pl.INT64), 128), pl.Mem.Vec] = pl.tile.load(
                    gamma_ckv__ssa_v0, [n0_inline899__idx_v0 + 64], [64], [64], target_memory=pl.Mem.Vec
                )
                t__tile_8: pl.Tile[[32, 64], pl.BF16, pl.MemRef(mem_vec_11, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(kv_chunk_inline895__tile_2, target_type=pl.BF16, mode="rint")
                kv_chunk_v2_inline853__tile: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(8192, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.cast(t__tile_8, target_type=pl.FP32, mode="round")
                gamma_kv_cast_inline919__tile: pl.Tile[[64], pl.FP32, pl.MemRef(mem_vec_11, pl.const(0, pl.INT64), 256), pl.Mem.Vec] = pl.tile.cast(t__tile_6, target_type=pl.FP32, mode="round")
                gamma_kv_chunk_inline856__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_11, pl.const(0, pl.INT64), 256), pl.Mem.Vec] = gamma_kv_cast_inline919__tile
                t__tile_9: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(8192, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.row_expand_mul(
                    kv_chunk_v2_inline853__tile, kv_inv_rms_t_inline917__tile
                )
                kv_normed_inline908__tile: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(8192, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.col_expand_mul(
                    t__tile_9, gamma_kv_chunk_inline856__tile
                )
                kv_normed_bf16_inline921__tile: pl.Tile[[32, 64], pl.BF16, pl.MemRef(mem_vec_12, pl.const(8192, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(
                    kv_normed_inline908__tile, target_type=pl.BF16, mode="rint"
                )
                kv_view_inline900__tile: pl.Tensor[[t_dim_inline868__ssa_v0, 512], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                    kv_normed_bf16_inline921__tile, [out_tg_inline905__ssa_v0, n0_inline899__idx_v0], kv_view_inline900__iter_v1
                )
                t__tile_10: pl.Tile[[32, 64], pl.BF16, pl.MemRef(mem_vec_17, pl.const(40960, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(kv_chunk_inline895__tile_3, target_type=pl.BF16, mode="rint")
                kv_chunk_v2_inline853__tile_1: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_18, pl.const(49152, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.cast(
                    t__tile_10, target_type=pl.FP32, mode="round"
                )
                gamma_kv_cast_inline919__tile_1: pl.Tile[[64], pl.FP32, pl.MemRef(mem_vec_17, pl.const(40960, pl.INT64), 256), pl.Mem.Vec] = pl.tile.cast(t__tile_7, target_type=pl.FP32, mode="round")
                gamma_kv_chunk_inline856__tile_1: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_17, pl.const(40960, pl.INT64), 256), pl.Mem.Vec] = gamma_kv_cast_inline919__tile_1
                t__tile_11: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_18, pl.const(49152, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.row_expand_mul(
                    kv_chunk_v2_inline853__tile_1, kv_inv_rms_t_inline917__tile
                )
                kv_normed_inline908__tile_1: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_18, pl.const(49152, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.col_expand_mul(
                    t__tile_11, gamma_kv_chunk_inline856__tile_1
                )
                kv_normed_bf16_inline921__tile_1: pl.Tile[[32, 64], pl.BF16, pl.MemRef(mem_vec_18, pl.const(49152, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(
                    kv_normed_inline908__tile_1, target_type=pl.BF16, mode="rint"
                )
                kv_view_inline900__tile_1: pl.Tensor[[t_dim_inline868__ssa_v0, 512], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                    kv_normed_bf16_inline921__tile_1, [out_tg_inline905__ssa_v0, n0_inline899__idx_v0 + 64], kv_view_inline900__tile
                )
                kv_view_inline900__rv_v2_main: pl.Tensor[[t_dim_inline868__ssa_v0, 512], pl.BF16, pl.MemRef("mem_ddr_42", pl.const(0, pl.INT64), 0)] = pl.yield_(kv_view_inline900__tile_1)
            kv_chunk_inline895__tile_4: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(8192, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.load(
                kv_fp32_inline882__rv_v10, [tg_inline911__ssa_v0, 384], [32, 64], [32, 64], target_memory=pl.Mem.Vec
            )
            t__tile_12: pl.Tile[[32, 64], pl.BF16, pl.MemRef(mem_vec_18, pl.const(49152, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(kv_chunk_inline895__tile_4, target_type=pl.BF16, mode="rint")
            kv_chunk_v2_inline853__tile_2: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(8192, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.cast(t__tile_12, target_type=pl.FP32, mode="round")
            t__tile_13: pl.Tile[[64], pl.BF16, pl.MemRef(mem_vec_11, pl.const(0, pl.INT64), 128), pl.Mem.Vec] = pl.tile.load(gamma_ckv__ssa_v0, [384], [64], [64], target_memory=pl.Mem.Vec)
            gamma_kv_cast_inline919__tile_2: pl.Tile[[64], pl.FP32, pl.MemRef(mem_vec_18, pl.const(49152, pl.INT64), 256), pl.Mem.Vec] = pl.tile.cast(t__tile_13, target_type=pl.FP32, mode="round")
            gamma_kv_chunk_inline856__tile_2: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_18, pl.const(49152, pl.INT64), 256), pl.Mem.Vec] = gamma_kv_cast_inline919__tile_2
            t__tile_14: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(8192, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.row_expand_mul(
                kv_chunk_v2_inline853__tile_2, kv_inv_rms_t_inline917__tile
            )
            kv_normed_inline908__tile_2: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(8192, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.col_expand_mul(
                t__tile_14, gamma_kv_chunk_inline856__tile_2
            )
            kv_normed_bf16_inline921__tile_2: pl.Tile[[32, 64], pl.BF16, pl.MemRef(mem_vec_12, pl.const(8192, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(
                kv_normed_inline908__tile_2, target_type=pl.BF16, mode="rint"
            )
            kv_view_inline900__tile_2: pl.Tensor[[t_dim_inline868__ssa_v0, 512], pl.BF16, pl.MemRef("mem_ddr_42", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                kv_normed_bf16_inline921__tile_2, [out_tg_inline905__ssa_v0, 384], kv_view_inline900__rv_v2_main
            )
            kv_view_inline900__rv_v2: pl.Tensor[[t_dim_inline868__ssa_v0, 512], pl.BF16, pl.MemRef("mem_ddr_51", pl.const(0, pl.INT64), 0)] = kv_view_inline900__tile_2
            t__tile_15: pl.Tile[[64], pl.BF16, pl.MemRef(mem_vec_12, pl.const(8192, pl.INT64), 128), pl.Mem.Vec] = pl.tile.load(gamma_ckv__ssa_v0, [448], [64], [64], target_memory=pl.Mem.Vec)
            gamma_rope_cast_inline912__tile: pl.Tile[[64], pl.FP32, pl.MemRef(mem_vec_11, pl.const(0, pl.INT64), 256), pl.Mem.Vec] = pl.tile.cast(t__tile_15, target_type=pl.FP32, mode="round")
            gamma_rope_inline918__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_11, pl.const(0, pl.INT64), 256), pl.Mem.Vec] = gamma_rope_cast_inline912__tile
            kv_rope_chunk_inline867__tile: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(8192, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.load(
                kv_fp32_inline882__rv_v10, [tg_inline911__ssa_v0, 448], [32, 64], [32, 64], target_memory=pl.Mem.Vec
            )
            t__tile_16: pl.Tile[[32, 64], pl.BF16, pl.MemRef(mem_vec_18, pl.const(49152, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(kv_rope_chunk_inline867__tile, target_type=pl.BF16, mode="rint")
            kv_rope_chunk_v1_inline924__tile: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(8192, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.cast(
                t__tile_16, target_type=pl.FP32, mode="round"
            )
            t__tile_17: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(8192, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.row_expand_mul(
                kv_rope_chunk_v1_inline924__tile, kv_inv_rms_t_inline917__tile
            )
            kv_rope_norm_chunk_inline854__tile: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(8192, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.col_expand_mul(
                t__tile_17, gamma_rope_inline918__tile
            )
            t__tile_18: pl.Tile[[32, 64], pl.BF16, pl.MemRef(mem_vec_18, pl.const(49152, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(
                kv_rope_norm_chunk_inline854__tile, target_type=pl.BF16, mode="rint"
            )
            kv_rope_norm_chunk_v1_inline884__tile: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(8192, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.cast(
                t__tile_18, target_type=pl.FP32, mode="round"
            )
            kv_cos_il_full_inline926__tile: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_18, pl.const(49152, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.load(
                q_rope_cos_il_inline361__ssa_v0, [out_tg_inline905__ssa_v0, 0], [32, 64], [32, 64], target_memory=pl.Mem.Vec
            )
            kv_sin_signed_full_inline916__tile: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_11, pl.const(0, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.load(
                q_rope_sin_signed_inline360__ssa_v0, [out_tg_inline905__ssa_v0, 0], [32, 64], [32, 64], target_memory=pl.Mem.Vec
            )
            kv_swap_idx_full_inline920__tile: pl.Tile[[32, 64], pl.INT32, pl.MemRef(mem_vec_17, pl.const(40960, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.load(
                q_rope_swap_idx_inline359__ssa_v0, [out_tg_inline905__ssa_v0, 0], [32, 64], [32, 64], target_memory=pl.Mem.Vec
            )
            gather_acc_init: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_64, pl.const(65536, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.create([32, 64], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            for gather_lv, (gather_ia,) in pl.range(32, init_values=(gather_acc_init,)):
                gather_inp_row: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(8192, pl.INT64), 256), pl.Mem.Vec] = pl.tile.slice(
                    kv_rope_norm_chunk_v1_inline884__tile, [1, 64], [gather_lv, 0], [1, 64]
                )
                gather_idx_row: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_17, pl.const(40960, pl.INT64), 256), pl.Mem.Vec] = pl.tile.slice(
                    kv_swap_idx_full_inline920__tile, [1, 64], [gather_lv, 0], [1, 64]
                )
                gather_row_tmp: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_145, pl.const(24576, pl.INT64), 256), pl.Mem.Vec] = pl.tile.create([1, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec)
                gather_row: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_146, pl.const(32768, pl.INT64), 256), pl.Mem.Vec] = pl.tile.gather(gather_inp_row, gather_idx_row, gather_row_tmp)
                gather_asmbl: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_64, pl.const(65536, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.assemble(gather_ia, gather_row, [gather_lv, 0])
                gather_rv: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_64, pl.const(65536, pl.INT64), 8192), pl.Mem.Vec] = pl.yield_(gather_asmbl)
            kv_swapped_full_inline851__tile: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_64, pl.const(65536, pl.INT64), 8192), pl.Mem.Vec] = gather_rv
            t__tile_19: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(8192, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.mul(
                kv_rope_norm_chunk_v1_inline884__tile, kv_cos_il_full_inline926__tile
            )
            t__tile_20: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_18, pl.const(49152, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.mul(
                kv_swapped_full_inline851__tile, kv_sin_signed_full_inline916__tile
            )
            kv_rope_rot_full_inline850__tile: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(8192, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.add(t__tile_19, t__tile_20)
            kv_rope_i16_full_inline849__tile: pl.Tile[[32, 64], pl.BF16, pl.MemRef(mem_vec_12, pl.const(8192, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(
                kv_rope_rot_full_inline850__tile, target_type=pl.BF16, mode="rint"
            )
            kv_view_inline900__tile_3: pl.Tensor[[t_dim_inline868__ssa_v0, 512], pl.BF16, pl.MemRef("mem_ddr_51", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                kv_rope_i16_full_inline849__tile, [out_tg_inline905__ssa_v0, 448], kv_view_inline900__rv_v2
            )
        else:
            kv_reduce_tmp_inline848__ssa_v0: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(8192, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.create(
                [32, 64], dtype=pl.FP32, target_memory=pl.Mem.Vec
            )
            kv_sq_sum_tail_inline847__ssa_v0: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_145, pl.const(24576, pl.INT64), 128), pl.Mem.Vec] = pl.tile.full([1, 32], dtype=pl.FP32, value=0.0)
            for kv_sq_col0_tail_inline914__idx_v0, (kv_sq_sum_tail_inline847__iter_v1,) in pl.range(0, 512, 128, init_values=(kv_sq_sum_tail_inline847__ssa_v0,)):
                kv_chunk_tail_inline865__ssa_v0: pl.Tile[
                    [32, 64], pl.FP32, pl.MemRef(mem_vec_18, pl.const(49152, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline923__ssa_v0, 64])
                ] = pl.tile.load(kv_fp32_inline882__rv_v10, [tg_inline911__ssa_v0, kv_sq_col0_tail_inline914__idx_v0], [32, 64], [valid_rows_inline923__ssa_v0, 64], target_memory=pl.Mem.Vec)
                kv_chunk_tail_inline865__ssa_v0_1: pl.Tile[
                    [32, 64], pl.FP32, pl.MemRef(mem_vec_11, pl.const(0, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline923__ssa_v0, 64])
                ] = pl.tile.load(kv_fp32_inline882__rv_v10, [tg_inline911__ssa_v0, kv_sq_col0_tail_inline914__idx_v0 + 64], [32, 64], [valid_rows_inline923__ssa_v0, 64], target_memory=pl.Mem.Vec)
                t__tmp_v78: pl.Tile[[32, 64], pl.BF16, pl.MemRef(mem_vec_17, pl.const(40960, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline923__ssa_v0, 64])] = pl.tile.cast(
                    kv_chunk_tail_inline865__ssa_v0, target_type=pl.BF16, mode="rint"
                )
                kv_chunk_tail_v1_inline846__ssa_v0: pl.Tile[
                    [32, 64], pl.FP32, pl.MemRef(mem_vec_18, pl.const(49152, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline923__ssa_v0, 64])
                ] = pl.tile.cast(t__tmp_v78, target_type=pl.FP32, mode="round")
                kv_sq_tail_inline845__ssa_v0: pl.Tile[
                    [32, 64], pl.FP32, pl.MemRef(mem_vec_18, pl.const(49152, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline923__ssa_v0, 64])
                ] = pl.tile.mul(kv_chunk_tail_v1_inline846__ssa_v0, kv_chunk_tail_v1_inline846__ssa_v0)
                t__tmp_v79: pl.Tile[[32, 1], pl.FP32, pl.MemRef(mem_vec_17, pl.const(40960, pl.INT64), 128), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline923__ssa_v0, 1])] = pl.tile.row_sum(
                    kv_sq_tail_inline845__ssa_v0, kv_reduce_tmp_inline848__ssa_v0
                )
                kv_row_sum_tail_inline843__ssa_v0: pl.Tile[
                    [1, 32], pl.FP32, pl.MemRef(mem_vec_17, pl.const(40960, pl.INT64), 128), pl.Mem.Vec, pl.TileView(valid_shape=[1, valid_rows_inline923__ssa_v0])
                ] = pl.tile.reshape(t__tmp_v79, [1, 32])
                kv_sq_sum_tail_inline847__ssa_v3: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_18, pl.const(49152, pl.INT64), 128), pl.Mem.Vec] = pl.tile.add(
                    kv_sq_sum_tail_inline847__iter_v1, kv_row_sum_tail_inline843__ssa_v0
                )
                t__tmp_v78_1: pl.Tile[[32, 64], pl.BF16, pl.MemRef(mem_vec_64, pl.const(65536, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline923__ssa_v0, 64])] = (
                    pl.tile.cast(kv_chunk_tail_inline865__ssa_v0_1, target_type=pl.BF16, mode="rint")
                )
                kv_chunk_tail_v1_inline846__ssa_v0_1: pl.Tile[
                    [32, 64], pl.FP32, pl.MemRef(mem_vec_11, pl.const(0, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline923__ssa_v0, 64])
                ] = pl.tile.cast(t__tmp_v78_1, target_type=pl.FP32, mode="round")
                kv_sq_tail_inline845__ssa_v0_1: pl.Tile[
                    [32, 64], pl.FP32, pl.MemRef(mem_vec_11, pl.const(0, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline923__ssa_v0, 64])
                ] = pl.tile.mul(kv_chunk_tail_v1_inline846__ssa_v0_1, kv_chunk_tail_v1_inline846__ssa_v0_1)
                t__tmp_v79_1: pl.Tile[[32, 1], pl.FP32, pl.MemRef(mem_vec_64, pl.const(65536, pl.INT64), 128), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline923__ssa_v0, 1])] = (
                    pl.tile.row_sum(kv_sq_tail_inline845__ssa_v0_1, kv_reduce_tmp_inline848__ssa_v0)
                )
                kv_row_sum_tail_inline843__ssa_v0_1: pl.Tile[
                    [1, 32], pl.FP32, pl.MemRef(mem_vec_64, pl.const(65536, pl.INT64), 128), pl.Mem.Vec, pl.TileView(valid_shape=[1, valid_rows_inline923__ssa_v0])
                ] = pl.tile.reshape(t__tmp_v79_1, [1, 32])
                kv_sq_sum_tail_inline847__ssa_v3_1: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_145, pl.const(24576, pl.INT64), 128), pl.Mem.Vec] = pl.tile.add(
                    kv_sq_sum_tail_inline847__ssa_v3, kv_row_sum_tail_inline843__ssa_v0_1
                )
                kv_sq_sum_tail_inline847__rv_v2: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_145, pl.const(24576, pl.INT64), 128), pl.Mem.Vec] = pl.yield_(kv_sq_sum_tail_inline847__ssa_v3_1)
            t__tmp_v80: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_12, pl.const(8192, pl.INT64), 128), pl.Mem.Vec] = pl.tile.muls(kv_sq_sum_tail_inline847__rv_v2, 0.001953125)
            t__tmp_v81: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_12, pl.const(8192, pl.INT64), 128), pl.Mem.Vec] = pl.tile.adds(t__tmp_v80, 9.9999999999999995e-07)
            t__tmp_v82: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_12, pl.const(8192, pl.INT64), 128), pl.Mem.Vec] = pl.tile.sqrt(t__tmp_v81)
            kv_inv_rms_tail_inline842__ssa_v0: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_64, pl.const(65536, pl.INT64), 128), pl.Mem.Vec] = pl.tile.recip(t__tmp_v82)
            kv_inv_rms_t_tail_inline871__ssa_v0: pl.Tile[[32, 1], pl.FP32, pl.MemRef(mem_vec_64, pl.const(65536, pl.INT64), 128), pl.Mem.Vec] = pl.tile.reshape(
                kv_inv_rms_tail_inline842__ssa_v0, [32, 1]
            )
            for n0_tail_inline906__idx_v0 in pl.range(0, 384, 128):
                kv_chunk_tail_inline865__ssa_v1: pl.Tile[
                    [32, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(8192, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline923__ssa_v0, 64])
                ] = pl.tile.load(kv_fp32_inline882__rv_v10, [tg_inline911__ssa_v0, n0_tail_inline906__idx_v0], [32, 64], [valid_rows_inline923__ssa_v0, 64], target_memory=pl.Mem.Vec)
                gamma_kv_input_tail_inline840__ssa_v0: pl.Tile[[64], pl.BF16, pl.MemRef(mem_vec_145, pl.const(24576, pl.INT64), 128), pl.Mem.Vec] = pl.tile.load(
                    gamma_ckv__ssa_v0, [n0_tail_inline906__idx_v0], [64], [64], target_memory=pl.Mem.Vec
                )
                kv_chunk_tail_inline865__ssa_v1_1: pl.Tile[
                    [32, 64], pl.FP32, pl.MemRef(mem_vec_18, pl.const(49152, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline923__ssa_v0, 64])
                ] = pl.tile.load(kv_fp32_inline882__rv_v10, [tg_inline911__ssa_v0, n0_tail_inline906__idx_v0 + 64], [32, 64], [valid_rows_inline923__ssa_v0, 64], target_memory=pl.Mem.Vec)
                gamma_kv_input_tail_inline840__ssa_v0_1: pl.Tile[[64], pl.BF16, pl.MemRef(mem_vec_146, pl.const(32768, pl.INT64), 128), pl.Mem.Vec] = pl.tile.load(
                    gamma_ckv__ssa_v0, [n0_tail_inline906__idx_v0 + 64], [64], [64], target_memory=pl.Mem.Vec
                )
                t__tmp_v83: pl.Tile[[32, 64], pl.BF16, pl.MemRef(mem_vec_11, pl.const(0, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline923__ssa_v0, 64])] = pl.tile.cast(
                    kv_chunk_tail_inline865__ssa_v1, target_type=pl.BF16, mode="rint"
                )
                kv_chunk_tail_v2_inline841__ssa_v0: pl.Tile[
                    [32, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(8192, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline923__ssa_v0, 64])
                ] = pl.tile.cast(t__tmp_v83, target_type=pl.FP32, mode="round")
                gamma_kv_cast_tail_inline839__ssa_v0: pl.Tile[[64], pl.FP32, pl.MemRef(mem_vec_11, pl.const(0, pl.INT64), 256), pl.Mem.Vec] = pl.tile.cast(
                    gamma_kv_input_tail_inline840__ssa_v0, target_type=pl.FP32, mode="round"
                )
                gamma_kv_chunk_tail_inline869__ssa_v0: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_11, pl.const(0, pl.INT64), 256), pl.Mem.Vec] = gamma_kv_cast_tail_inline839__ssa_v0
                t__tmp_v84: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(8192, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline923__ssa_v0, 64])] = (
                    pl.tile.row_expand_mul(kv_chunk_tail_v2_inline841__ssa_v0, kv_inv_rms_t_tail_inline871__ssa_v0)
                )
                kv_normed_tail_inline863__ssa_v0: pl.Tile[
                    [32, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(8192, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline923__ssa_v0, 64])
                ] = pl.tile.col_expand_mul(t__tmp_v84, gamma_kv_chunk_tail_inline869__ssa_v0)
                kv_normed_bf16_tail_inline838__ssa_v0: pl.Tile[
                    [32, 64], pl.BF16, pl.MemRef(mem_vec_12, pl.const(8192, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline923__ssa_v0, 64])
                ] = pl.tile.cast(kv_normed_tail_inline863__ssa_v0, target_type=pl.BF16, mode="rint")
                kv_normed_valid_inline886__ssa_v0: pl.Tile[
                    [32, 64], pl.BF16, pl.MemRef(mem_vec_12, pl.const(8192, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline923__ssa_v0, 64])
                ] = pl.tile.set_validshape(kv_normed_bf16_tail_inline838__ssa_v0, valid_rows_inline923__ssa_v0, 64)
                kv_view_inline900__store: pl.Tensor[[t_dim_inline868__ssa_v0, 512], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                    kv_normed_valid_inline886__ssa_v0, [out_tg_inline905__ssa_v0, n0_tail_inline906__idx_v0], kv_view_inline900__ssa_v0
                )
                t__tmp_v83_1: pl.Tile[[32, 64], pl.BF16, pl.MemRef(mem_vec_17, pl.const(40960, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline923__ssa_v0, 64])] = (
                    pl.tile.cast(kv_chunk_tail_inline865__ssa_v1_1, target_type=pl.BF16, mode="rint")
                )
                kv_chunk_tail_v2_inline841__ssa_v0_1: pl.Tile[
                    [32, 64], pl.FP32, pl.MemRef(mem_vec_18, pl.const(49152, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline923__ssa_v0, 64])
                ] = pl.tile.cast(t__tmp_v83_1, target_type=pl.FP32, mode="round")
                gamma_kv_cast_tail_inline839__ssa_v0_1: pl.Tile[[64], pl.FP32, pl.MemRef(mem_vec_17, pl.const(40960, pl.INT64), 256), pl.Mem.Vec] = pl.tile.cast(
                    gamma_kv_input_tail_inline840__ssa_v0_1, target_type=pl.FP32, mode="round"
                )
                gamma_kv_chunk_tail_inline869__ssa_v0_1: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_17, pl.const(40960, pl.INT64), 256), pl.Mem.Vec] = gamma_kv_cast_tail_inline839__ssa_v0_1
                t__tmp_v84_1: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_18, pl.const(49152, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline923__ssa_v0, 64])] = (
                    pl.tile.row_expand_mul(kv_chunk_tail_v2_inline841__ssa_v0_1, kv_inv_rms_t_tail_inline871__ssa_v0)
                )
                kv_normed_tail_inline863__ssa_v0_1: pl.Tile[
                    [32, 64], pl.FP32, pl.MemRef(mem_vec_18, pl.const(49152, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline923__ssa_v0, 64])
                ] = pl.tile.col_expand_mul(t__tmp_v84_1, gamma_kv_chunk_tail_inline869__ssa_v0_1)
                kv_normed_bf16_tail_inline838__ssa_v0_1: pl.Tile[
                    [32, 64], pl.BF16, pl.MemRef(mem_vec_18, pl.const(49152, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline923__ssa_v0, 64])
                ] = pl.tile.cast(kv_normed_tail_inline863__ssa_v0_1, target_type=pl.BF16, mode="rint")
                kv_normed_valid_inline886__ssa_v0_1: pl.Tile[
                    [32, 64], pl.BF16, pl.MemRef(mem_vec_18, pl.const(49152, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline923__ssa_v0, 64])
                ] = pl.tile.set_validshape(kv_normed_bf16_tail_inline838__ssa_v0_1, valid_rows_inline923__ssa_v0, 64)
                kv_view_inline900__store_1: pl.Tensor[[t_dim_inline868__ssa_v0, 512], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                    kv_normed_valid_inline886__ssa_v0_1, [out_tg_inline905__ssa_v0, n0_tail_inline906__idx_v0 + 64], kv_view_inline900__ssa_v0
                )
            kv_chunk_tail_inline865__ssa_v1_2: pl.Tile[
                [32, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(8192, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline923__ssa_v0, 64])
            ] = pl.tile.load(kv_fp32_inline882__rv_v10, [tg_inline911__ssa_v0, 384], [32, 64], [valid_rows_inline923__ssa_v0, 64], target_memory=pl.Mem.Vec)
            t__tmp_v83_2: pl.Tile[[32, 64], pl.BF16, pl.MemRef(mem_vec_18, pl.const(49152, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline923__ssa_v0, 64])] = pl.tile.cast(
                kv_chunk_tail_inline865__ssa_v1_2, target_type=pl.BF16, mode="rint"
            )
            kv_chunk_tail_v2_inline841__ssa_v0_2: pl.Tile[
                [32, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(8192, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline923__ssa_v0, 64])
            ] = pl.tile.cast(t__tmp_v83_2, target_type=pl.FP32, mode="round")
            gamma_kv_input_tail_inline840__ssa_v0_2: pl.Tile[[64], pl.BF16, pl.MemRef(mem_vec_11, pl.const(0, pl.INT64), 128), pl.Mem.Vec] = pl.tile.load(
                gamma_ckv__ssa_v0, [384], [64], [64], target_memory=pl.Mem.Vec
            )
            gamma_kv_cast_tail_inline839__ssa_v0_2: pl.Tile[[64], pl.FP32, pl.MemRef(mem_vec_18, pl.const(49152, pl.INT64), 256), pl.Mem.Vec] = pl.tile.cast(
                gamma_kv_input_tail_inline840__ssa_v0_2, target_type=pl.FP32, mode="round"
            )
            gamma_kv_chunk_tail_inline869__ssa_v0_2: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_18, pl.const(49152, pl.INT64), 256), pl.Mem.Vec] = gamma_kv_cast_tail_inline839__ssa_v0_2
            t__tmp_v84_2: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(8192, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline923__ssa_v0, 64])] = (
                pl.tile.row_expand_mul(kv_chunk_tail_v2_inline841__ssa_v0_2, kv_inv_rms_t_tail_inline871__ssa_v0)
            )
            kv_normed_tail_inline863__ssa_v0_2: pl.Tile[
                [32, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(8192, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline923__ssa_v0, 64])
            ] = pl.tile.col_expand_mul(t__tmp_v84_2, gamma_kv_chunk_tail_inline869__ssa_v0_2)
            kv_normed_bf16_tail_inline838__ssa_v0_2: pl.Tile[
                [32, 64], pl.BF16, pl.MemRef(mem_vec_12, pl.const(8192, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline923__ssa_v0, 64])
            ] = pl.tile.cast(kv_normed_tail_inline863__ssa_v0_2, target_type=pl.BF16, mode="rint")
            kv_normed_valid_inline886__ssa_v0_2: pl.Tile[
                [32, 64], pl.BF16, pl.MemRef(mem_vec_12, pl.const(8192, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline923__ssa_v0, 64])
            ] = pl.tile.set_validshape(kv_normed_bf16_tail_inline838__ssa_v0_2, valid_rows_inline923__ssa_v0, 64)
            kv_view_inline900__store_2: pl.Tensor[[t_dim_inline868__ssa_v0, 512], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                kv_normed_valid_inline886__ssa_v0_2, [out_tg_inline905__ssa_v0, 384], kv_view_inline900__ssa_v0
            )
            gamma_rope_input_tail_inline874__ssa_v0: pl.Tile[[64], pl.BF16, pl.MemRef(mem_vec_12, pl.const(8192, pl.INT64), 128), pl.Mem.Vec] = pl.tile.load(
                gamma_ckv__ssa_v0, [448], [64], [64], target_memory=pl.Mem.Vec
            )
            gamma_rope_cast_tail_inline913__ssa_v0: pl.Tile[[64], pl.FP32, pl.MemRef(mem_vec_11, pl.const(0, pl.INT64), 256), pl.Mem.Vec] = pl.tile.cast(
                gamma_rope_input_tail_inline874__ssa_v0, target_type=pl.FP32, mode="round"
            )
            gamma_rope_tail_inline837__ssa_v0: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_11, pl.const(0, pl.INT64), 256), pl.Mem.Vec] = gamma_rope_cast_tail_inline913__ssa_v0
            kv_rope_chunk_tail_inline836__ssa_v0: pl.Tile[
                [32, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(8192, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline923__ssa_v0, 64])
            ] = pl.tile.load(kv_fp32_inline882__rv_v10, [tg_inline911__ssa_v0, 448], [32, 64], [valid_rows_inline923__ssa_v0, 64], target_memory=pl.Mem.Vec)
            t__tmp_v85: pl.Tile[[32, 64], pl.BF16, pl.MemRef(mem_vec_18, pl.const(49152, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline923__ssa_v0, 64])] = pl.tile.cast(
                kv_rope_chunk_tail_inline836__ssa_v0, target_type=pl.BF16, mode="rint"
            )
            kv_rope_chunk_tail_v1_inline835__ssa_v0: pl.Tile[
                [32, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(8192, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline923__ssa_v0, 64])
            ] = pl.tile.cast(t__tmp_v85, target_type=pl.FP32, mode="round")
            t__tmp_v86: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(8192, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline923__ssa_v0, 64])] = (
                pl.tile.row_expand_mul(kv_rope_chunk_tail_v1_inline835__ssa_v0, kv_inv_rms_t_tail_inline871__ssa_v0)
            )
            kv_rope_norm_tail_inline834__ssa_v0: pl.Tile[
                [32, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(8192, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline923__ssa_v0, 64])
            ] = pl.tile.col_expand_mul(t__tmp_v86, gamma_rope_tail_inline837__ssa_v0)
            t__tmp_v87: pl.Tile[[32, 64], pl.BF16, pl.MemRef(mem_vec_18, pl.const(49152, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline923__ssa_v0, 64])] = pl.tile.cast(
                kv_rope_norm_tail_inline834__ssa_v0, target_type=pl.BF16, mode="rint"
            )
            kv_rope_norm_tail_v1_inline922__ssa_v0: pl.Tile[
                [32, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(8192, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline923__ssa_v0, 64])
            ] = pl.tile.cast(t__tmp_v87, target_type=pl.FP32, mode="round")
            kv_cos_il_tail_inline833__ssa_v0: pl.Tile[
                [32, 64], pl.FP32, pl.MemRef(mem_vec_18, pl.const(49152, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline923__ssa_v0, 64])
            ] = pl.tile.load(q_rope_cos_il_inline361__ssa_v0, [out_tg_inline905__ssa_v0, 0], [32, 64], [valid_rows_inline923__ssa_v0, 64], target_memory=pl.Mem.Vec)
            kv_sin_signed_tail_inline832__ssa_v0: pl.Tile[
                [32, 64], pl.FP32, pl.MemRef(mem_vec_11, pl.const(0, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline923__ssa_v0, 64])
            ] = pl.tile.load(q_rope_sin_signed_inline360__ssa_v0, [out_tg_inline905__ssa_v0, 0], [32, 64], [valid_rows_inline923__ssa_v0, 64], target_memory=pl.Mem.Vec)
            t__tmp_v88: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_17, pl.const(40960, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.full([32, 64], dtype=pl.FP32, value=1.0)
            t__ci_tmp_v0: pl.Tile[[1, 192], pl.FP32, pl.MemRef(mem_vec_64, pl.const(65536, pl.INT64), 768), pl.Mem.Vec] = pl.tile.create([1, 192], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            t__tmp_v89: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_145, pl.const(24576, pl.INT64), 256), pl.Mem.Vec] = pl.tile.ci(
                pl.const(0, pl.INT32), [1, 64], tmp=t__ci_tmp_v0, dtype=pl.INT32, descending=False
            )
            t__tmp_v90: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_64, pl.const(65536, pl.INT64), 256), pl.Mem.Vec] = pl.tile.cast(t__tmp_v89, target_type=pl.FP32, mode="round")
            kv_col_inline860__ssa_v0: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_17, pl.const(40960, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.col_expand_mul(t__tmp_v88, t__tmp_v90)
            t__tmp_v91: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_64, pl.const(65536, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.muls(kv_col_inline860__ssa_v0, 0.5)
            t__tmp_v92: pl.Tile[[32, 64], pl.INT32, pl.MemRef(mem_vec_64, pl.const(65536, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.cast(t__tmp_v91, target_type=pl.INT32, mode="trunc")
            kv_dup_f_inline831__ssa_v0: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_64, pl.const(65536, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.cast(t__tmp_v92, target_type=pl.FP32, mode="round")
            t__tmp_v93: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_64, pl.const(65536, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.muls(kv_dup_f_inline831__ssa_v0, 2.0)
            kv_lane_inline830__ssa_v0: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_64, pl.const(65536, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.sub(kv_col_inline860__ssa_v0, t__tmp_v93)
            t__tmp_v94: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_17, pl.const(40960, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.adds(kv_col_inline860__ssa_v0, 1.0)
            t__tmp_v95: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_64, pl.const(65536, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.muls(kv_lane_inline830__ssa_v0, 2.0)
            kv_swap_f_inline829__ssa_v0: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_17, pl.const(40960, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.sub(t__tmp_v94, t__tmp_v95)
            t__ci_tmp_v1: pl.Tile[[1, 192], pl.FP32, pl.MemRef(mem_vec_64, pl.const(65536, pl.INT64), 768), pl.Mem.Vec] = pl.tile.create([1, 192], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            t__tmp_v96: pl.Tile[[1, 32], pl.INT32, pl.MemRef(mem_vec_145, pl.const(24576, pl.INT64), 128), pl.Mem.Vec] = pl.tile.ci(
                pl.const(0, pl.INT32), [1, 32], tmp=t__ci_tmp_v1, dtype=pl.INT32, descending=False
            )
            t__tmp_v97: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_64, pl.const(65536, pl.INT64), 128), pl.Mem.Vec] = pl.tile.cast(t__tmp_v96, target_type=pl.FP32, mode="round")
            kv_row_seed_inline828__ssa_v0: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_145, pl.const(24576, pl.INT64), 128), pl.Mem.Vec] = pl.tile.muls(t__tmp_v97, 64.0)
            t__tmp_v98: pl.Tile[[64, 32], pl.FP32, pl.MemRef(mem_vec_64, pl.const(65536, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.full([64, 32], dtype=pl.FP32, value=1.0)
            kv_row_grid_inline827__ssa_v0: pl.Tile[[64, 32], pl.FP32, pl.MemRef(mem_vec_64, pl.const(65536, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.col_expand_mul(
                t__tmp_v98, kv_row_seed_inline828__ssa_v0
            )
            transpose_tmp: pl.Tile[[64, 32], pl.FP32, pl.MemRef(mem_vec_145, pl.const(24576, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.create([64, 32], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            kv_row_offset_inline909__ssa_v0: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_146, pl.const(32768, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.transpose(
                kv_row_grid_inline827__ssa_v0, 0, 1, transpose_tmp
            )
            t__tmp_v99: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_17, pl.const(40960, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.add(kv_swap_f_inline829__ssa_v0, kv_row_offset_inline909__ssa_v0)
            kv_swap_idx_tail_inline826__ssa_v0: pl.Tile[[32, 64], pl.INT32, pl.MemRef(mem_vec_17, pl.const(40960, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.cast(
                t__tmp_v99, target_type=pl.INT32, mode="round"
            )
            kv_gather_tmp_inline825__ssa_v0: pl.Tile[[32, 64], pl.INT32, pl.MemRef(mem_vec_64, pl.const(65536, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.create(
                [32, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec
            )
            kv_swapped_tail_inline880__ssa_v0: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_145, pl.const(24576, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.gather(
                kv_rope_norm_tail_v1_inline922__ssa_v0, kv_swap_idx_tail_inline826__ssa_v0, kv_gather_tmp_inline825__ssa_v0
            )
            t__tmp_v100: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(8192, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline923__ssa_v0, 64])] = pl.tile.mul(
                kv_rope_norm_tail_v1_inline922__ssa_v0, kv_cos_il_tail_inline833__ssa_v0
            )
            t__tmp_v101: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_18, pl.const(49152, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.mul(
                kv_swapped_tail_inline880__ssa_v0, kv_sin_signed_tail_inline832__ssa_v0
            )
            kv_rope_rot_tail_inline844__ssa_v0: pl.Tile[
                [32, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(8192, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline923__ssa_v0, 64])
            ] = pl.tile.add(t__tmp_v100, t__tmp_v101)
            kv_rope_i16_tail_inline925__ssa_v0: pl.Tile[
                [32, 64], pl.BF16, pl.MemRef(mem_vec_12, pl.const(8192, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline923__ssa_v0, 64])
            ] = pl.tile.cast(kv_rope_rot_tail_inline844__ssa_v0, target_type=pl.BF16, mode="rint")
            kv_rope_valid_inline904__ssa_v0: pl.Tile[
                [32, 64], pl.BF16, pl.MemRef(mem_vec_12, pl.const(8192, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline923__ssa_v0, 64])
            ] = pl.tile.set_validshape(kv_rope_i16_tail_inline925__ssa_v0, valid_rows_inline923__ssa_v0, 64)
            kv_view_inline900__store_v0: pl.Tensor[[t_dim_inline868__ssa_v0, 512], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                kv_rope_valid_inline904__ssa_v0, [out_tg_inline905__ssa_v0, 448], kv_view_inline900__ssa_v0
            )
        return kv_view_inline900__ssa_v0

    @pl.function(type=pl.FunctionType.Spmd)
    def kv_rms_norm_rope_spmd(
        self,
        tile_rows_inline888__ssa_v0: pl.Scalar[pl.INDEX],
        tile_base_inline876__idx_v0: pl.Scalar[pl.INDEX],
        kv_fp32_inline882__rv_v10: pl.Tensor[[t_matmul_inline907__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)],
        kv_view_inline900__ssa_v0: pl.InOut[pl.Tensor[[t_dim_inline868__ssa_v0, 512], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)]],
        gamma_ckv__ssa_v0: pl.Tensor[[512], pl.BF16, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 1024)],
        q_rope_cos_il_inline361__ssa_v0: pl.Tensor[[t_dim_inline362__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)],
        q_rope_sin_signed_inline360__ssa_v0: pl.Tensor[[t_dim_inline362__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 0)],
        q_rope_swap_idx_inline359__ssa_v0: pl.Tensor[[t_dim_inline362__ssa_v0, 64], pl.INT32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 0)],
    ):
        pl.func_attr({"mx_tensor_views_blocked": True})
        kv_view_inline900__ssa_v1: pl.Tensor[[t_dim_inline868__ssa_v0, 512], pl.BF16, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 0)] = self.kv_rms_norm_rope(
            tile_rows_inline888__ssa_v0,
            tile_base_inline876__idx_v0,
            kv_fp32_inline882__rv_v10,
            kv_view_inline900__ssa_v0,
            gamma_ckv__ssa_v0,
            q_rope_cos_il_inline361__ssa_v0,
            q_rope_sin_signed_inline360__ssa_v0,
            q_rope_swap_idx_inline359__ssa_v0,
            attrs={"arg_directions": [pl.adir.scalar, pl.adir.scalar, pl.adir.input, pl.adir.inout, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input]},
        )

    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def q_rope_prepare(
        rope_cos_il_view_inline794__ssa_v0: pl.Out[pl.Tensor[[t_dim_inline799__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)]],
        rope_sin_signed_view_inline783__ssa_v0: pl.Out[pl.Tensor[[t_dim_inline799__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)]],
        rope_swap_idx_view_inline780__ssa_v0: pl.Out[pl.Tensor[[t_dim_inline799__ssa_v0, 64], pl.INT32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)]],
        token_tiles_inline779__ssa_v0: pl.Scalar[pl.INDEX],
        t_dim_inline799__ssa_v0: pl.Scalar[pl.INDEX],
        rope_cos_view_inline795__ssa_v0: pl.Tensor[[t_dim_inline799__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)],
        rope_sin_view_inline793__ssa_v0: pl.Tensor[[t_dim_inline799__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 0)],
    ):
        pl.func_attr({"mx_tensor_views_blocked": True})
        mem_vec_5: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 2048)
        mem_vec_10: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 2048)
        mem_vec_13: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 2048)
        mem_vec_17: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 2048)
        mem_vec_23: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 2048)
        mem_vec_24: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 2048)
        mem_vec_40: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 2048)
        mem_vec_43: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 2048)
        mem_vec_46: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 2048)
        mem_vec_54: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 2048)
        mem_vec_55: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 2048)
        qrp_worker_inline792__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        qrp_ones_inline800__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_5, pl.const(16384, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.full([8, 64], dtype=pl.FP32, value=1.0)
        qrp_idx_i32_inline778__ci_tmp_v0: pl.Tile[[1, 192], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 768), pl.Mem.Vec] = pl.tile.create([1, 192], dtype=pl.FP32, target_memory=pl.Mem.Vec)
        qrp_idx_i32_inline778__tile: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_13, pl.const(2048, pl.INT64), 256), pl.Mem.Vec] = pl.tile.ci(
            pl.const(0, pl.INT32), [1, 64], tmp=qrp_idx_i32_inline778__ci_tmp_v0, dtype=pl.INT32, descending=False
        )
        qrp_idx_fp32_inline788__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 256), pl.Mem.Vec] = pl.tile.cast(
            qrp_idx_i32_inline778__tile, target_type=pl.FP32, mode="round"
        )
        qrp_col_inline777__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_5, pl.const(16384, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.col_expand_mul(
            qrp_ones_inline800__tile, qrp_idx_fp32_inline788__tile
        )
        qrp_half_inline784__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.muls(qrp_col_inline777__tile, 0.5)
        qrp_dup_i32_inline785__tile: pl.Tile[[8, 64], pl.INT32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(
            qrp_half_inline784__tile, target_type=pl.INT32, mode="trunc"
        )
        qrp_dup_f_inline786__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(
            qrp_dup_i32_inline785__tile, target_type=pl.FP32, mode="round"
        )
        qrp_dup_idx_inline801__tile: pl.Tile[[8, 64], pl.INT32, pl.MemRef(mem_vec_13, pl.const(2048, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(
            qrp_dup_f_inline786__tile, target_type=pl.INT32, mode="round"
        )
        t__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.muls(qrp_dup_f_inline786__tile, 2.0)
        qrp_lane_inline782__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.sub(qrp_col_inline777__tile, t__tile)
        qrp_next_col_inline797__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_5, pl.const(16384, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.adds(qrp_col_inline777__tile, 1.0)
        qrp_lane_offset_inline804__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_17, pl.const(4096, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.muls(qrp_lane_inline782__tile, 2.0)
        qrp_swap_f_inline805__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_5, pl.const(16384, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.sub(
            qrp_next_col_inline797__tile, qrp_lane_offset_inline804__tile
        )
        qrp_swap_idx_inline808__tile: pl.Tile[[8, 64], pl.INT32, pl.MemRef(mem_vec_5, pl.const(16384, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(
            qrp_swap_f_inline805__tile, target_type=pl.INT32, mode="round"
        )
        t__tile_1: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.muls(qrp_lane_inline782__tile, 2.0)
        qrp_sign_inline803__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.subs(t__tile_1, 1.0)
        for qrp_idx_inline796__idx_v0, (rope_cos_il_view_inline794__iter_v1, rope_sin_signed_view_inline783__iter_v1, rope_swap_idx_view_inline780__iter_v1) in pl.range(
            qrp_worker_inline792__ssa_v0,
            token_tiles_inline779__ssa_v0,
            pl.min(token_tiles_inline779__ssa_v0, 48),
            init_values=(rope_cos_il_view_inline794__ssa_v0, rope_sin_signed_view_inline783__ssa_v0, rope_swap_idx_view_inline780__ssa_v0),
        ):
            qrp_t0_inline806__ssa_v0: pl.Scalar[pl.INDEX] = qrp_idx_inline796__idx_v0 * 8
            qrp_valid_rows_inline789__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(t_dim_inline799__ssa_v0 - qrp_t0_inline806__ssa_v0, 8)
            if qrp_valid_rows_inline789__ssa_v0 == 8:
                qrp_cos_rows_full_inline807__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_17, pl.const(4096, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.load(
                    rope_cos_view_inline795__ssa_v0, [qrp_t0_inline806__ssa_v0, 0], [8, 64], [8, 64], target_memory=pl.Mem.Vec
                )
                qrp_sin_rows_full_inline798__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_23, pl.const(6144, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.load(
                    rope_sin_view_inline793__ssa_v0, [qrp_t0_inline806__ssa_v0, 0], [8, 64], [8, 64], target_memory=pl.Mem.Vec
                )
                qrp_cos_full_inline809__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_17, pl.const(4096, pl.INT64), 2048), pl.Mem.Vec] = qrp_cos_rows_full_inline807__tile
                qrp_sin_full_inline787__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_23, pl.const(6144, pl.INT64), 2048), pl.Mem.Vec] = qrp_sin_rows_full_inline798__tile
                gather_acc_init: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_24, pl.const(8192, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.create([8, 64], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                for gather_lv, (gather_ia,) in pl.range(8, init_values=(gather_acc_init,)):
                    gather_inp_row: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_17, pl.const(4096, pl.INT64), 256), pl.Mem.Vec] = pl.tile.slice(
                        qrp_cos_full_inline809__ssa_v0, [1, 64], [gather_lv, 0], [1, 64]
                    )
                    gather_idx_row: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_13, pl.const(2048, pl.INT64), 256), pl.Mem.Vec] = pl.tile.slice(
                        qrp_dup_idx_inline801__tile, [1, 64], [gather_lv, 0], [1, 64]
                    )
                    gather_row_tmp: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_40, pl.const(10240, pl.INT64), 256), pl.Mem.Vec] = pl.tile.create([1, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec)
                    gather_row: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_43, pl.const(12288, pl.INT64), 256), pl.Mem.Vec] = pl.tile.gather(gather_inp_row, gather_idx_row, gather_row_tmp)
                    gather_asmbl: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_24, pl.const(8192, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.assemble(gather_ia, gather_row, [gather_lv, 0])
                    gather_rv: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_24, pl.const(8192, pl.INT64), 2048), pl.Mem.Vec] = pl.yield_(gather_asmbl)
                qrp_cos_il_full_inline776__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_24, pl.const(8192, pl.INT64), 2048), pl.Mem.Vec] = gather_rv
                gather_acc_init_1: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_17, pl.const(4096, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.create([8, 64], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                for gather_lv_1, (gather_ia_1,) in pl.range(8, init_values=(gather_acc_init_1,)):
                    gather_inp_row_1: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_23, pl.const(6144, pl.INT64), 256), pl.Mem.Vec] = pl.tile.slice(
                        qrp_sin_full_inline787__ssa_v0, [1, 64], [gather_lv_1, 0], [1, 64]
                    )
                    gather_idx_row_1: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_13, pl.const(2048, pl.INT64), 256), pl.Mem.Vec] = pl.tile.slice(
                        qrp_dup_idx_inline801__tile, [1, 64], [gather_lv_1, 0], [1, 64]
                    )
                    gather_row_tmp_1: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_40, pl.const(10240, pl.INT64), 256), pl.Mem.Vec] = pl.tile.create([1, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec)
                    gather_row_1: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_43, pl.const(12288, pl.INT64), 256), pl.Mem.Vec] = pl.tile.gather(gather_inp_row_1, gather_idx_row_1, gather_row_tmp_1)
                    gather_asmbl_1: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_17, pl.const(4096, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.assemble(gather_ia_1, gather_row_1, [gather_lv_1, 0])
                    gather_rv_1: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_17, pl.const(4096, pl.INT64), 2048), pl.Mem.Vec] = pl.yield_(gather_asmbl_1)
                qrp_sin_il_full_inline773__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_17, pl.const(4096, pl.INT64), 2048), pl.Mem.Vec] = gather_rv_1
                qrp_sin_signed_full_inline771__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_17, pl.const(4096, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.mul(
                    qrp_sin_il_full_inline773__tile, qrp_sign_inline803__tile
                )
                rope_cos_il_view_inline794__tile: pl.Tensor[[t_dim_inline799__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                    qrp_cos_il_full_inline776__tile, [qrp_t0_inline806__ssa_v0, 0], rope_cos_il_view_inline794__iter_v1
                )
                rope_sin_signed_view_inline783__tile: pl.Tensor[[t_dim_inline799__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                    qrp_sin_signed_full_inline771__tile, [qrp_t0_inline806__ssa_v0, 0], rope_sin_signed_view_inline783__iter_v1
                )
                rope_swap_idx_view_inline780__tile: pl.Tensor[[t_dim_inline799__ssa_v0, 64], pl.INT32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                    qrp_swap_idx_inline808__tile, [qrp_t0_inline806__ssa_v0, 0], rope_swap_idx_view_inline780__iter_v1
                )
                rope_cos_il_view_inline794__phi_v4, rope_sin_signed_view_inline783__phi_v4, rope_swap_idx_view_inline780__phi_v4 = pl.yield_(
                    rope_cos_il_view_inline794__tile, rope_sin_signed_view_inline783__tile, rope_swap_idx_view_inline780__tile
                )
            else:
                qrp_cos_rows_tail_inline775__ssa_v0: pl.Tile[
                    [8, 64], pl.FP32, pl.MemRef(mem_vec_17, pl.const(4096, pl.INT64), 2048), pl.Mem.Vec, pl.TileView(valid_shape=[qrp_valid_rows_inline789__ssa_v0, 64])
                ] = pl.tile.load(rope_cos_view_inline795__ssa_v0, [qrp_t0_inline806__ssa_v0, 0], [8, 64], [qrp_valid_rows_inline789__ssa_v0, 64], target_memory=pl.Mem.Vec)
                qrp_sin_rows_tail_inline770__ssa_v0: pl.Tile[
                    [8, 64], pl.FP32, pl.MemRef(mem_vec_23, pl.const(6144, pl.INT64), 2048), pl.Mem.Vec, pl.TileView(valid_shape=[qrp_valid_rows_inline789__ssa_v0, 64])
                ] = pl.tile.load(rope_sin_view_inline793__ssa_v0, [qrp_t0_inline806__ssa_v0, 0], [8, 64], [qrp_valid_rows_inline789__ssa_v0, 64], target_memory=pl.Mem.Vec)
                t__tmp_v2: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_24, pl.const(8192, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.full([8, 64], dtype=pl.FP32, value=1.0)
                t__ci_tmp_v1: pl.Tile[[1, 192], pl.FP32, pl.MemRef(mem_vec_40, pl.const(10240, pl.INT64), 768), pl.Mem.Vec] = pl.tile.create([1, 192], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                t__tmp_v3: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_43, pl.const(12288, pl.INT64), 256), pl.Mem.Vec] = pl.tile.ci(
                    pl.const(0, pl.INT32), [1, 64], tmp=t__ci_tmp_v1, dtype=pl.INT32, descending=False
                )
                t__tmp_v4: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_40, pl.const(10240, pl.INT64), 256), pl.Mem.Vec] = pl.tile.cast(t__tmp_v3, target_type=pl.FP32, mode="round")
                qrp_tail_col_inline791__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_24, pl.const(8192, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.col_expand_mul(t__tmp_v2, t__tmp_v4)
                t__tmp_v5: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_40, pl.const(10240, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.muls(qrp_tail_col_inline791__ssa_v0, 0.5)
                t__tmp_v6: pl.Tile[[8, 64], pl.INT32, pl.MemRef(mem_vec_40, pl.const(10240, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(t__tmp_v5, target_type=pl.INT32, mode="trunc")
                qrp_tail_dup_f_inline774__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_40, pl.const(10240, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(
                    t__tmp_v6, target_type=pl.FP32, mode="round"
                )
                t__tmp_v7: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_43, pl.const(12288, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.muls(qrp_tail_dup_f_inline774__ssa_v0, 2.0)
                qrp_tail_lane_inline781__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_43, pl.const(12288, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.sub(qrp_tail_col_inline791__ssa_v0, t__tmp_v7)
                t__tmp_v8: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_24, pl.const(8192, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.adds(qrp_tail_col_inline791__ssa_v0, 1.0)
                t__tmp_v9: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_46, pl.const(14336, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.muls(qrp_tail_lane_inline781__ssa_v0, 2.0)
                qrp_tail_swap_f_inline802__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_24, pl.const(8192, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.sub(t__tmp_v8, t__tmp_v9)
                t__ci_tmp_v2: pl.Tile[[1, 192], pl.FP32, pl.MemRef(mem_vec_46, pl.const(14336, pl.INT64), 768), pl.Mem.Vec] = pl.tile.create([1, 192], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                t__tmp_v10: pl.Tile[[1, 8], pl.INT32, pl.MemRef(mem_vec_54, pl.const(18432, pl.INT64), 32), pl.Mem.Vec] = pl.tile.ci(
                    pl.const(0, pl.INT32), [1, 8], tmp=t__ci_tmp_v2, dtype=pl.INT32, descending=False
                )
                t__tmp_v11: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_46, pl.const(14336, pl.INT64), 32), pl.Mem.Vec] = pl.tile.cast(t__tmp_v10, target_type=pl.FP32, mode="round")
                qrp_row_seed_inline769__ssa_v0: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_54, pl.const(18432, pl.INT64), 32), pl.Mem.Vec] = pl.tile.muls(t__tmp_v11, 64.0)
                t__tmp_v12: pl.Tile[[64, 8], pl.FP32, pl.MemRef(mem_vec_46, pl.const(14336, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.full([64, 8], dtype=pl.FP32, value=1.0)
                qrp_row_grid_inline790__ssa_v0: pl.Tile[[64, 8], pl.FP32, pl.MemRef(mem_vec_46, pl.const(14336, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.col_expand_mul(
                    t__tmp_v12, qrp_row_seed_inline769__ssa_v0
                )
                transpose_tmp: pl.Tile[[64, 8], pl.FP32, pl.MemRef(mem_vec_54, pl.const(18432, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.create([64, 8], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                qrp_row_offset_inline767__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_55, pl.const(20480, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.transpose(
                    qrp_row_grid_inline790__ssa_v0, 0, 1, transpose_tmp
                )
                t__tmp_v13: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_40, pl.const(10240, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.add(
                    qrp_tail_dup_f_inline774__ssa_v0, qrp_row_offset_inline767__ssa_v0
                )
                qrp_dup_idx_tail_inline766__ssa_v0: pl.Tile[[8, 64], pl.INT32, pl.MemRef(mem_vec_40, pl.const(10240, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(
                    t__tmp_v13, target_type=pl.INT32, mode="round"
                )
                qrp_gather_tmp_inline764__ssa_v0: pl.Tile[[8, 64], pl.INT32, pl.MemRef(mem_vec_46, pl.const(14336, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.create(
                    [8, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec
                )
                qrp_cos_il_tail_inline763__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_54, pl.const(18432, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.gather(
                    qrp_cos_rows_tail_inline775__ssa_v0, qrp_dup_idx_tail_inline766__ssa_v0, qrp_gather_tmp_inline764__ssa_v0
                )
                qrp_sin_il_tail_inline772__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_17, pl.const(4096, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.gather(
                    qrp_sin_rows_tail_inline770__ssa_v0, qrp_dup_idx_tail_inline766__ssa_v0, qrp_gather_tmp_inline764__ssa_v0
                )
                t__tmp_v14: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_23, pl.const(6144, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.muls(qrp_tail_lane_inline781__ssa_v0, 2.0)
                qrp_tail_sign_inline768__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_23, pl.const(6144, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.subs(t__tmp_v14, 1.0)
                qrp_sin_signed_tail_inline765__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_17, pl.const(4096, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.mul(
                    qrp_sin_il_tail_inline772__ssa_v0, qrp_tail_sign_inline768__ssa_v0
                )
                t__tmp_v15: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_54, pl.const(18432, pl.INT64), 2048), pl.Mem.Vec, pl.TileView(valid_shape=[qrp_valid_rows_inline789__ssa_v0, 64])] = (
                    pl.tile.set_validshape(qrp_cos_il_tail_inline763__ssa_v0, qrp_valid_rows_inline789__ssa_v0, 64)
                )
                pl.tile.store(t__tmp_v15, [qrp_t0_inline806__ssa_v0, 0], rope_cos_il_view_inline794__iter_v1)
                t__tmp_v16: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_17, pl.const(4096, pl.INT64), 2048), pl.Mem.Vec, pl.TileView(valid_shape=[qrp_valid_rows_inline789__ssa_v0, 64])] = (
                    pl.tile.set_validshape(qrp_sin_signed_tail_inline765__ssa_v0, qrp_valid_rows_inline789__ssa_v0, 64)
                )
                pl.tile.store(t__tmp_v16, [qrp_t0_inline806__ssa_v0, 0], rope_sin_signed_view_inline783__iter_v1)
                t__tmp_v17: pl.Tile[[8, 64], pl.INT32, pl.MemRef(mem_vec_17, pl.const(4096, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(
                    qrp_tail_swap_f_inline802__ssa_v0, target_type=pl.INT32, mode="round"
                )
                t__tmp_v18: pl.Tile[[8, 64], pl.INT32, pl.MemRef(mem_vec_17, pl.const(4096, pl.INT64), 2048), pl.Mem.Vec, pl.TileView(valid_shape=[qrp_valid_rows_inline789__ssa_v0, 64])] = (
                    pl.tile.set_validshape(t__tmp_v17, qrp_valid_rows_inline789__ssa_v0, 64)
                )
                pl.tile.store(t__tmp_v18, [qrp_t0_inline806__ssa_v0, 0], rope_swap_idx_view_inline780__iter_v1)
                rope_cos_il_view_inline794__phi_v4, rope_sin_signed_view_inline783__phi_v4, rope_swap_idx_view_inline780__phi_v4 = pl.yield_(
                    rope_cos_il_view_inline794__iter_v1, rope_sin_signed_view_inline783__iter_v1, rope_swap_idx_view_inline780__iter_v1
                )
            rope_cos_il_view_inline794__rv_v2, rope_sin_signed_view_inline783__rv_v2, rope_swap_idx_view_inline780__rv_v2 = pl.yield_(
                rope_cos_il_view_inline794__phi_v4, rope_sin_signed_view_inline783__phi_v4, rope_swap_idx_view_inline780__phi_v4
            )
        return

    @pl.function(type=pl.FunctionType.Spmd)
    def q_rope_prepare_spmd(
        self,
        rope_cos_il_view_inline794__ssa_v0: pl.Out[pl.Tensor[[t_dim_inline799__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)]],
        rope_sin_signed_view_inline783__ssa_v0: pl.Out[pl.Tensor[[t_dim_inline799__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)]],
        rope_swap_idx_view_inline780__ssa_v0: pl.Out[pl.Tensor[[t_dim_inline799__ssa_v0, 64], pl.INT32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)]],
        token_tiles_inline779__ssa_v0: pl.Scalar[pl.INDEX],
        t_dim_inline799__ssa_v0: pl.Scalar[pl.INDEX],
        rope_cos_view_inline795__ssa_v0: pl.Tensor[[t_dim_inline799__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)],
        rope_sin_view_inline793__ssa_v0: pl.Tensor[[t_dim_inline799__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 0)],
    ):
        pl.func_attr({"mx_tensor_views_blocked": True})
        self.q_rope_prepare(
            rope_cos_il_view_inline794__ssa_v0,
            rope_sin_signed_view_inline783__ssa_v0,
            rope_swap_idx_view_inline780__ssa_v0,
            token_tiles_inline779__ssa_v0,
            t_dim_inline799__ssa_v0,
            rope_cos_view_inline795__ssa_v0,
            rope_sin_view_inline793__ssa_v0,
            attrs={"arg_directions": [pl.adir.output_existing, pl.adir.output_existing, pl.adir.output_existing, pl.adir.scalar, pl.adir.scalar, pl.adir.input, pl.adir.input]},
        )

    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def qproj_dequant_rms_nope_rope(
        q_flat_inline1121__ssa_v0: pl.Out[pl.Tensor[[t_dim_inline1134__ssa_v0, 32768], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)]],
        tile_rows_inline228_inline821__ssa_v0: pl.Scalar[pl.INDEX],
        tile_base_inline229_inline820__idx_v0: pl.Scalar[pl.INDEX],
        qr_scale_pad_store_inline813__rv_v2: pl.Tensor[[512, 1], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 2048)],
        q_rope_cos_il_inline361__ssa_v0: pl.Tensor[[t_dim_inline362__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)],
        q_rope_sin_signed_inline360__ssa_v0: pl.Tensor[[t_dim_inline362__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)],
        q_rope_swap_idx_inline359__ssa_v0: pl.Tensor[[t_dim_inline362__ssa_v0, 64], pl.INT32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 0)],
        q_proj_i32_inline230_inline824__ssa_v9: pl.Tensor[[qproj_t_matmul_inline227_inline822__ssa_v0, 32768], pl.INT32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 0)],
        wq_b_scale__ssa_v0: pl.Tensor[[32768], pl.FP32, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 131072)],
    ):
        pl.func_attr({"mx_tensor_views_blocked": True})
        mem_vec_7: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32)
        mem_vec_8: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 2048)
        mem_vec_9: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 2048)
        mem_vec_10: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 2048)
        mem_vec_11: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
        mem_vec_13: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
        mem_vec_20: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
        mem_vec_21: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
        mem_vec_33: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        mem_vec_34: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        mem_vec_45: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
        mem_vec_46: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
        mem_vec_58: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        mem_vec_59: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        dq_worker_inline1103__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        for dq_work_inline1106__idx_v0, (q_flat_inline1121__iter_v1,) in pl.range(
            dq_worker_inline1103__ssa_v0, (tile_rows_inline228_inline821__ssa_v0 + 7) // 8 * 16, 48, init_values=(q_flat_inline1121__ssa_v0,)
        ):
            hg_inline1114__ssa_v0: pl.Scalar[pl.INDEX] = dq_work_inline1106__idx_v0 % 16 * 4
            tg_inline1095__ssa_v0: pl.Scalar[pl.INDEX] = dq_work_inline1106__idx_v0 // 16 * 8
            out_tg_inline1101__ssa_v0: pl.Scalar[pl.INDEX] = tile_base_inline229_inline820__idx_v0 + tg_inline1095__ssa_v0
            if tg_inline1095__ssa_v0 + 8 <= tile_rows_inline228_inline821__ssa_v0:
                qr_scale_dq_t_inline1090__tile: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_7, pl.const(101376, pl.INT64), 32), pl.Mem.Vec] = pl.tile.load(
                    qr_scale_pad_store_inline813__rv_v2, [tg_inline1095__ssa_v0, 0], [8, 1], [8, 1], target_memory=pl.Mem.Vec
                )
                q_cos_il_inline1115__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_8, pl.const(101408, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.load(
                    q_rope_cos_il_inline361__ssa_v0, [out_tg_inline1101__ssa_v0, 0], [8, 64], [8, 64], target_memory=pl.Mem.Vec
                )
                q_sin_signed_inline1102__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_9, pl.const(103456, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.load(
                    q_rope_sin_signed_inline360__ssa_v0, [out_tg_inline1101__ssa_v0, 0], [8, 64], [8, 64], target_memory=pl.Mem.Vec
                )
                q_swap_idx_inline1125__tile: pl.Tile[[8, 64], pl.INT32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.load(
                    q_rope_swap_idx_inline359__ssa_v0, [out_tg_inline1101__ssa_v0, 0], [8, 64], [8, 64], target_memory=pl.Mem.Vec
                )
                for h_inner_inline1127__idx_v0, (q_flat_inline1121__iter_v3,) in pl.range(0, 4, 2, init_values=(q_flat_inline1121__iter_v1,)):
                    h_inline1130__ssa_v0: pl.Scalar[pl.INDEX] = hg_inline1114__ssa_v0 + h_inner_inline1127__idx_v0
                    h0_inline1124__ssa_v0: pl.Scalar[pl.INDEX] = h_inline1130__ssa_v0 * 512
                    h_inline1130__ssa_v0_1: pl.Scalar[pl.INDEX] = hg_inline1114__ssa_v0 + (h_inner_inline1127__idx_v0 + 1)
                    h0_inline1124__ssa_v0_1: pl.Scalar[pl.INDEX] = h_inline1130__ssa_v0_1 * 512
                    q_head_acc_inline1100__tile: pl.Tile[[8, 512], pl.INT32, pl.MemRef(mem_vec_11, pl.const(2048, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.load(
                        q_proj_i32_inline230_inline824__ssa_v9, [tg_inline1095__ssa_v0, h0_inline1124__ssa_v0], [8, 512], [8, 512], target_memory=pl.Mem.Vec
                    )
                    t__tile: pl.Tile[[512], pl.FP32, pl.MemRef(mem_vec_20, pl.const(34816, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.load(
                        wq_b_scale__ssa_v0, [h0_inline1124__ssa_v0], [512], [512], target_memory=pl.Mem.Vec
                    )
                    q_head_acc_inline1100__tile_1: pl.Tile[[8, 512], pl.INT32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.load(
                        q_proj_i32_inline230_inline824__ssa_v9, [tg_inline1095__ssa_v0, h0_inline1124__ssa_v0_1], [8, 512], [8, 512], target_memory=pl.Mem.Vec
                    )
                    t__tile_1: pl.Tile[[512], pl.FP32, pl.MemRef(mem_vec_45, pl.const(68096, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.load(
                        wq_b_scale__ssa_v0, [h0_inline1124__ssa_v0_1], [512], [512], target_memory=pl.Mem.Vec
                    )
                    q_head_scale_inline1138__tile: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_20, pl.const(34816, pl.INT64), 2048), pl.Mem.Vec] = t__tile
                    q_head_acc_fp32_inline1096__tile: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_11, pl.const(2048, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.cast(
                        q_head_acc_inline1100__tile, target_type=pl.FP32, mode="none"
                    )
                    q_head_row_scaled_inline1093__tile: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_11, pl.const(2048, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.row_expand_mul(
                        q_head_acc_fp32_inline1096__tile, qr_scale_dq_t_inline1090__tile
                    )
                    q_head_dq_inline1105__tile: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_11, pl.const(2048, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.col_expand_mul(
                        q_head_row_scaled_inline1093__tile, q_head_scale_inline1138__tile
                    )
                    t__tile_2: pl.Tile[[8, 512], pl.BF16, pl.MemRef(mem_vec_20, pl.const(34816, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.cast(
                        q_head_dq_inline1105__tile, target_type=pl.BF16, mode="rint"
                    )
                    q_head_dq_v1_inline1088__tile: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_11, pl.const(2048, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.cast(
                        t__tile_2, target_type=pl.FP32, mode="round"
                    )
                    q_head_sq_inline1117__tile: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_20, pl.const(34816, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.mul(
                        q_head_dq_v1_inline1088__tile, q_head_dq_v1_inline1088__tile
                    )
                    tmp_tile: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_21, pl.const(51200, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.create([8, 512], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                    q_head_sq_row_inline1129__tile: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_33, pl.const(67584, pl.INT64), 32), pl.Mem.Vec] = pl.tile.row_sum(q_head_sq_inline1117__tile, tmp_tile)
                    q_head_sq_sum_inline1136__tile: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_33, pl.const(67584, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(q_head_sq_row_inline1129__tile, [1, 8])
                    q_head_sq_mean_inline1094__tile: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_20, pl.const(34816, pl.INT64), 32), pl.Mem.Vec] = pl.tile.muls(
                        q_head_sq_sum_inline1136__tile, 0.001953125
                    )
                    q_head_var_inline1110__tile: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_20, pl.const(34816, pl.INT64), 32), pl.Mem.Vec] = pl.tile.adds(
                        q_head_sq_mean_inline1094__tile, 9.9999999999999995e-07
                    )
                    rsqrt_tmp: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_21, pl.const(51200, pl.INT64), 32), pl.Mem.Vec] = pl.tile.create([1, 8], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                    q_head_inv_rms_inline1111__tile: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_33, pl.const(67584, pl.INT64), 32), pl.Mem.Vec] = pl.tile.rsqrt(q_head_var_inline1110__tile, rsqrt_tmp)
                    q_head_inv_rms_t_inline1098__tile: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_33, pl.const(67584, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(
                        q_head_inv_rms_inline1111__tile, [8, 1]
                    )
                    t__tile_3: pl.Tile[[8, 448], pl.FP32, pl.MemRef(mem_vec_11, pl.const(2048, pl.INT64), 16128), pl.Mem.Vec] = pl.tile.slice(q_head_dq_v1_inline1088__tile, [8, 448], [0, 0])
                    q_nope_normed_inline1089__tile: pl.Tile[[8, 448], pl.FP32, pl.MemRef(mem_vec_20, pl.const(34816, pl.INT64), 14336), pl.Mem.Vec] = pl.tile.row_expand_mul(
                        t__tile_3, q_head_inv_rms_t_inline1098__tile
                    )
                    q_nope_bf16_inline1119__tile: pl.Tile[[8, 448], pl.BF16, pl.MemRef(mem_vec_20, pl.const(34816, pl.INT64), 7168), pl.Mem.Vec] = pl.tile.cast(
                        q_nope_normed_inline1089__tile, target_type=pl.BF16, mode="rint"
                    )
                    q_rope_chunk_raw_inline1132__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_11, pl.const(3840, pl.INT64), 14592), pl.Mem.Vec] = pl.tile.slice(
                        q_head_dq_v1_inline1088__tile, [8, 64], [0, 448]
                    )
                    q_rope_chunk_inline1118__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_11, pl.const(2048, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.row_expand_mul(
                        q_rope_chunk_raw_inline1132__tile, q_head_inv_rms_t_inline1098__tile
                    )
                    t__tile_4: pl.Tile[[8, 64], pl.BF16, pl.MemRef(mem_vec_21, pl.const(51200, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.cast(
                        q_rope_chunk_inline1118__tile, target_type=pl.BF16, mode="rint"
                    )
                    q_rope_chunk_v1_inline1133__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_11, pl.const(2048, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(
                        t__tile_4, target_type=pl.FP32, mode="round"
                    )
                    gather_acc_init: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_21, pl.const(51200, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.create([8, 64], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                    for gather_lv, (gather_ia,) in pl.range(8, init_values=(gather_acc_init,)):
                        gather_inp_row: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_11, pl.const(2048, pl.INT64), 256), pl.Mem.Vec] = pl.tile.slice(
                            q_rope_chunk_v1_inline1133__tile, [1, 64], [gather_lv, 0], [1, 64]
                        )
                        gather_idx_row: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 256), pl.Mem.Vec] = pl.tile.slice(
                            q_swap_idx_inline1125__tile, [1, 64], [gather_lv, 0], [1, 64]
                        )
                        gather_row_tmp: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_33, pl.const(67584, pl.INT64), 256), pl.Mem.Vec] = pl.tile.create(
                            [1, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec
                        )
                        gather_row: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_34, pl.const(67840, pl.INT64), 256), pl.Mem.Vec] = pl.tile.gather(gather_inp_row, gather_idx_row, gather_row_tmp)
                        gather_asmbl: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_21, pl.const(51200, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.assemble(gather_ia, gather_row, [gather_lv, 0])
                        gather_rv: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_21, pl.const(51200, pl.INT64), 2048), pl.Mem.Vec] = pl.yield_(gather_asmbl)
                    q_rope_swapped_inline1135__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_21, pl.const(51200, pl.INT64), 2048), pl.Mem.Vec] = gather_rv
                    q_rope_base_inline1137__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_11, pl.const(2048, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.mul(
                        q_rope_chunk_v1_inline1133__tile, q_cos_il_inline1115__tile
                    )
                    q_rope_delta_inline1150__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_21, pl.const(51200, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.mul(
                        q_rope_swapped_inline1135__tile, q_sin_signed_inline1102__tile
                    )
                    q_rope_rot_inline1131__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_11, pl.const(2048, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.add(
                        q_rope_base_inline1137__tile, q_rope_delta_inline1150__tile
                    )
                    q_rope_bf16_inline1128__tile: pl.Tile[[8, 64], pl.BF16, pl.MemRef(mem_vec_11, pl.const(2048, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.cast(
                        q_rope_rot_inline1131__tile, target_type=pl.BF16, mode="rint"
                    )
                    q_flat_inline1121__tile: pl.Tensor[[t_dim_inline1134__ssa_v0, 32768], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                        q_nope_bf16_inline1119__tile, [out_tg_inline1101__ssa_v0, h0_inline1124__ssa_v0], q_flat_inline1121__iter_v3
                    )
                    q_flat_inline1121__tile_1: pl.Tensor[[t_dim_inline1134__ssa_v0, 32768], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                        q_rope_bf16_inline1128__tile, [out_tg_inline1101__ssa_v0, h0_inline1124__ssa_v0 + 448], q_flat_inline1121__tile
                    )
                    q_head_scale_inline1138__tile_1: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_45, pl.const(68096, pl.INT64), 2048), pl.Mem.Vec] = t__tile_1
                    q_head_acc_fp32_inline1096__tile_1: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.cast(
                        q_head_acc_inline1100__tile_1, target_type=pl.FP32, mode="none"
                    )
                    q_head_row_scaled_inline1093__tile_1: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.row_expand_mul(
                        q_head_acc_fp32_inline1096__tile_1, qr_scale_dq_t_inline1090__tile
                    )
                    q_head_dq_inline1105__tile_1: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.col_expand_mul(
                        q_head_row_scaled_inline1093__tile_1, q_head_scale_inline1138__tile_1
                    )
                    t__tile_5: pl.Tile[[8, 512], pl.BF16, pl.MemRef(mem_vec_45, pl.const(68096, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.cast(
                        q_head_dq_inline1105__tile_1, target_type=pl.BF16, mode="rint"
                    )
                    q_head_dq_v1_inline1088__tile_1: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.cast(
                        t__tile_5, target_type=pl.FP32, mode="round"
                    )
                    q_head_sq_inline1117__tile_1: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_45, pl.const(68096, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.mul(
                        q_head_dq_v1_inline1088__tile_1, q_head_dq_v1_inline1088__tile_1
                    )
                    tmp_tile_1: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_46, pl.const(84480, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.create([8, 512], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                    q_head_sq_row_inline1129__tile_1: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_58, pl.const(100864, pl.INT64), 32), pl.Mem.Vec] = pl.tile.row_sum(
                        q_head_sq_inline1117__tile_1, tmp_tile_1
                    )
                    q_head_sq_sum_inline1136__tile_1: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_58, pl.const(100864, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(
                        q_head_sq_row_inline1129__tile_1, [1, 8]
                    )
                    q_head_sq_mean_inline1094__tile_1: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_45, pl.const(68096, pl.INT64), 32), pl.Mem.Vec] = pl.tile.muls(
                        q_head_sq_sum_inline1136__tile_1, 0.001953125
                    )
                    q_head_var_inline1110__tile_1: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_45, pl.const(68096, pl.INT64), 32), pl.Mem.Vec] = pl.tile.adds(
                        q_head_sq_mean_inline1094__tile_1, 9.9999999999999995e-07
                    )
                    rsqrt_tmp_1: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_46, pl.const(84480, pl.INT64), 32), pl.Mem.Vec] = pl.tile.create([1, 8], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                    q_head_inv_rms_inline1111__tile_1: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_58, pl.const(100864, pl.INT64), 32), pl.Mem.Vec] = pl.tile.rsqrt(
                        q_head_var_inline1110__tile_1, rsqrt_tmp_1
                    )
                    q_head_inv_rms_t_inline1098__tile_1: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_58, pl.const(100864, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(
                        q_head_inv_rms_inline1111__tile_1, [8, 1]
                    )
                    t__tile_6: pl.Tile[[8, 448], pl.FP32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 16128), pl.Mem.Vec] = pl.tile.slice(q_head_dq_v1_inline1088__tile_1, [8, 448], [0, 0])
                    q_nope_normed_inline1089__tile_1: pl.Tile[[8, 448], pl.FP32, pl.MemRef(mem_vec_45, pl.const(68096, pl.INT64), 14336), pl.Mem.Vec] = pl.tile.row_expand_mul(
                        t__tile_6, q_head_inv_rms_t_inline1098__tile_1
                    )
                    q_nope_bf16_inline1119__tile_1: pl.Tile[[8, 448], pl.BF16, pl.MemRef(mem_vec_45, pl.const(68096, pl.INT64), 7168), pl.Mem.Vec] = pl.tile.cast(
                        q_nope_normed_inline1089__tile_1, target_type=pl.BF16, mode="rint"
                    )
                    q_rope_chunk_raw_inline1132__tile_1: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(20224, pl.INT64), 14592), pl.Mem.Vec] = pl.tile.slice(
                        q_head_dq_v1_inline1088__tile_1, [8, 64], [0, 448]
                    )
                    q_rope_chunk_inline1118__tile_1: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.row_expand_mul(
                        q_rope_chunk_raw_inline1132__tile_1, q_head_inv_rms_t_inline1098__tile_1
                    )
                    t__tile_7: pl.Tile[[8, 64], pl.BF16, pl.MemRef(mem_vec_46, pl.const(84480, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.cast(
                        q_rope_chunk_inline1118__tile_1, target_type=pl.BF16, mode="rint"
                    )
                    q_rope_chunk_v1_inline1133__tile_1: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(
                        t__tile_7, target_type=pl.FP32, mode="round"
                    )
                    gather_acc_init_1: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_46, pl.const(84480, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.create([8, 64], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                    for gather_lv_1, (gather_ia_1,) in pl.range(8, init_values=(gather_acc_init_1,)):
                        gather_inp_row_1: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 256), pl.Mem.Vec] = pl.tile.slice(
                            q_rope_chunk_v1_inline1133__tile_1, [1, 64], [gather_lv_1, 0], [1, 64]
                        )
                        gather_idx_row_1: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 256), pl.Mem.Vec] = pl.tile.slice(
                            q_swap_idx_inline1125__tile, [1, 64], [gather_lv_1, 0], [1, 64]
                        )
                        gather_row_tmp_1: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_58, pl.const(100864, pl.INT64), 256), pl.Mem.Vec] = pl.tile.create(
                            [1, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec
                        )
                        gather_row_1: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_59, pl.const(101120, pl.INT64), 256), pl.Mem.Vec] = pl.tile.gather(
                            gather_inp_row_1, gather_idx_row_1, gather_row_tmp_1
                        )
                        gather_asmbl_1: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_46, pl.const(84480, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.assemble(gather_ia_1, gather_row_1, [gather_lv_1, 0])
                        gather_rv_1: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_46, pl.const(84480, pl.INT64), 2048), pl.Mem.Vec] = pl.yield_(gather_asmbl_1)
                    q_rope_swapped_inline1135__tile_1: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_46, pl.const(84480, pl.INT64), 2048), pl.Mem.Vec] = gather_rv_1
                    q_rope_base_inline1137__tile_1: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.mul(
                        q_rope_chunk_v1_inline1133__tile_1, q_cos_il_inline1115__tile
                    )
                    q_rope_delta_inline1150__tile_1: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_46, pl.const(84480, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.mul(
                        q_rope_swapped_inline1135__tile_1, q_sin_signed_inline1102__tile
                    )
                    q_rope_rot_inline1131__tile_1: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.add(
                        q_rope_base_inline1137__tile_1, q_rope_delta_inline1150__tile_1
                    )
                    q_rope_bf16_inline1128__tile_1: pl.Tile[[8, 64], pl.BF16, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.cast(
                        q_rope_rot_inline1131__tile_1, target_type=pl.BF16, mode="rint"
                    )
                    q_flat_inline1121__tile_2: pl.Tensor[[t_dim_inline1134__ssa_v0, 32768], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                        q_nope_bf16_inline1119__tile_1, [out_tg_inline1101__ssa_v0, h0_inline1124__ssa_v0_1], q_flat_inline1121__tile_1
                    )
                    q_flat_inline1121__tile_3: pl.Tensor[[t_dim_inline1134__ssa_v0, 32768], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                        q_rope_bf16_inline1128__tile_1, [out_tg_inline1101__ssa_v0, h0_inline1124__ssa_v0_1 + 448], q_flat_inline1121__tile_2
                    )
                    q_flat_inline1121__rv_v4: pl.Tensor[[t_dim_inline1134__ssa_v0, 32768], pl.BF16, pl.MemRef("mem_ddr_65", pl.const(0, pl.INT64), 0)] = pl.yield_(q_flat_inline1121__tile_3)
                q_flat_inline1121__phi_v7: pl.Tensor[[t_dim_inline1134__ssa_v0, 32768], pl.BF16, pl.MemRef("mem_ddr_117", pl.const(0, pl.INT64), 0)] = pl.yield_(q_flat_inline1121__rv_v4)
            else:
                valid_tail_rows_inline1107__ssa_v0: pl.Scalar[pl.INDEX] = tile_rows_inline228_inline821__ssa_v0 - tg_inline1095__ssa_v0
                qr_scale_dq_tail_inline1145__ssa_v0: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_9, pl.const(103456, pl.INT64), 32), pl.Mem.Vec] = pl.tile.load(
                    qr_scale_pad_store_inline813__rv_v2, [tg_inline1095__ssa_v0, 0], [8, 1], [8, 1], target_memory=pl.Mem.Vec
                )
                q_cos_il_tail_inline1123__ssa_v0: pl.Tile[
                    [8, 64], pl.FP32, pl.MemRef(mem_vec_21, pl.const(51200, pl.INT64), 2048), pl.Mem.Vec, pl.TileView(valid_shape=[valid_tail_rows_inline1107__ssa_v0, 64])
                ] = pl.tile.load(q_rope_cos_il_inline361__ssa_v0, [out_tg_inline1101__ssa_v0, 0], [8, 64], [valid_tail_rows_inline1107__ssa_v0, 64], target_memory=pl.Mem.Vec)
                q_sin_signed_tail_inline1097__ssa_v0: pl.Tile[
                    [8, 64], pl.FP32, pl.MemRef(mem_vec_45, pl.const(68096, pl.INT64), 2048), pl.Mem.Vec, pl.TileView(valid_shape=[valid_tail_rows_inline1107__ssa_v0, 64])
                ] = pl.tile.load(q_rope_sin_signed_inline360__ssa_v0, [out_tg_inline1101__ssa_v0, 0], [8, 64], [valid_tail_rows_inline1107__ssa_v0, 64], target_memory=pl.Mem.Vec)
                t__tmp_v47: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_11, pl.const(2048, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.full([8, 64], dtype=pl.FP32, value=1.0)
                t__ci_tmp_v0: pl.Tile[[1, 192], pl.FP32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 768), pl.Mem.Vec] = pl.tile.create([1, 192], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                t__tmp_v48: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_20, pl.const(34816, pl.INT64), 256), pl.Mem.Vec] = pl.tile.ci(
                    pl.const(0, pl.INT32), [1, 64], tmp=t__ci_tmp_v0, dtype=pl.INT32, descending=False
                )
                t__tmp_v49: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 256), pl.Mem.Vec] = pl.tile.cast(t__tmp_v48, target_type=pl.FP32, mode="round")
                q_col_inline1112__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_11, pl.const(2048, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.col_expand_mul(t__tmp_v47, t__tmp_v49)
                t__tmp_v50: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.muls(q_col_inline1112__ssa_v0, 0.5)
                t__tmp_v51: pl.Tile[[8, 64], pl.INT32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(t__tmp_v50, target_type=pl.INT32, mode="trunc")
                q_dup_f_inline1139__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(t__tmp_v51, target_type=pl.FP32, mode="round")
                t__tmp_v52: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.muls(q_dup_f_inline1139__ssa_v0, 2.0)
                q_lane_inline1108__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.sub(q_col_inline1112__ssa_v0, t__tmp_v52)
                t__tmp_v53: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_11, pl.const(2048, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.adds(q_col_inline1112__ssa_v0, 1.0)
                t__tmp_v54: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.muls(q_lane_inline1108__ssa_v0, 2.0)
                q_swap_f_inline1141__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_11, pl.const(2048, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.sub(t__tmp_v53, t__tmp_v54)
                t__ci_tmp_v1: pl.Tile[[1, 192], pl.FP32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 768), pl.Mem.Vec] = pl.tile.create([1, 192], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                t__tmp_v55: pl.Tile[[1, 8], pl.INT32, pl.MemRef(mem_vec_20, pl.const(34816, pl.INT64), 32), pl.Mem.Vec] = pl.tile.ci(
                    pl.const(0, pl.INT32), [1, 8], tmp=t__ci_tmp_v1, dtype=pl.INT32, descending=False
                )
                t__tmp_v56: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 32), pl.Mem.Vec] = pl.tile.cast(t__tmp_v55, target_type=pl.FP32, mode="round")
                q_row_seed_inline1092__ssa_v0: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_20, pl.const(34816, pl.INT64), 32), pl.Mem.Vec] = pl.tile.muls(t__tmp_v56, 64.0)
                t__tmp_v57: pl.Tile[[64, 8], pl.FP32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.full([64, 8], dtype=pl.FP32, value=1.0)
                q_row_grid_inline1142__ssa_v0: pl.Tile[[64, 8], pl.FP32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.col_expand_mul(
                    t__tmp_v57, q_row_seed_inline1092__ssa_v0
                )
                transpose_tmp: pl.Tile[[64, 8], pl.FP32, pl.MemRef(mem_vec_20, pl.const(34816, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.create([64, 8], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                q_row_offset_inline1104__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_46, pl.const(84480, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.transpose(
                    q_row_grid_inline1142__ssa_v0, 0, 1, transpose_tmp
                )
                t__tmp_v58: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_11, pl.const(2048, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.add(q_swap_f_inline1141__ssa_v0, q_row_offset_inline1104__ssa_v0)
                q_swap_idx_tail_inline1091__ssa_v0: pl.Tile[[8, 64], pl.INT32, pl.MemRef(mem_vec_46, pl.const(84480, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(
                    t__tmp_v58, target_type=pl.INT32, mode="round"
                )
                q_head_reduce_tmp_inline1116__ssa_v0: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_11, pl.const(2048, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.create(
                    [8, 512], dtype=pl.FP32, target_memory=pl.Mem.Vec
                )
                q_gather_tmp_inline1143__ssa_v0: pl.Tile[[8, 64], pl.INT32, pl.MemRef(mem_vec_8, pl.const(101408, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.create(
                    [8, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec
                )
                for h_inner_tail_inline1144__idx_v0 in pl.range(4):
                    h_tail_inline1147__ssa_v0: pl.Scalar[pl.INDEX] = hg_inline1114__ssa_v0 + h_inner_tail_inline1144__idx_v0
                    h0_tail_inline1122__ssa_v0: pl.Scalar[pl.INDEX] = h_tail_inline1147__ssa_v0 * 512
                    q_head_acc_tail_inline1151__ssa_v0: pl.Tile[[8, 512], pl.INT32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.load(
                        q_proj_i32_inline230_inline824__ssa_v9, [tg_inline1095__ssa_v0, h0_tail_inline1122__ssa_v0], [8, 512], [8, 512], target_memory=pl.Mem.Vec
                    )
                    q_head_scale_input_tail_inline1120__ssa_v0: pl.Tile[[512], pl.FP32, pl.MemRef(mem_vec_20, pl.const(34816, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.load(
                        wq_b_scale__ssa_v0, [h0_tail_inline1122__ssa_v0], [512], [512], target_memory=pl.Mem.Vec
                    )
                    q_head_scale_tail_inline1152__ssa_v0: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_20, pl.const(34816, pl.INT64), 2048), pl.Mem.Vec] = q_head_scale_input_tail_inline1120__ssa_v0
                    q_head_acc_fp32_tail_inline1153__ssa_v0: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.cast(
                        q_head_acc_tail_inline1151__ssa_v0, target_type=pl.FP32, mode="none"
                    )
                    q_head_row_scaled_tail_inline1146__ssa_v0: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.row_expand_mul(
                        q_head_acc_fp32_tail_inline1153__ssa_v0, qr_scale_dq_tail_inline1145__ssa_v0
                    )
                    q_head_dq_tail_inline1087__ssa_v0: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.col_expand_mul(
                        q_head_row_scaled_tail_inline1146__ssa_v0, q_head_scale_tail_inline1152__ssa_v0
                    )
                    t__tmp_v59: pl.Tile[[8, 512], pl.BF16, pl.MemRef(mem_vec_20, pl.const(34816, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.cast(
                        q_head_dq_tail_inline1087__ssa_v0, target_type=pl.BF16, mode="rint"
                    )
                    q_head_dq_tail_v1_inline1086__ssa_v0: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.cast(
                        t__tmp_v59, target_type=pl.FP32, mode="round"
                    )
                    q_head_sq_tail_inline1085__ssa_v0: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_20, pl.const(34816, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.mul(
                        q_head_dq_tail_v1_inline1086__ssa_v0, q_head_dq_tail_v1_inline1086__ssa_v0
                    )
                    q_head_sq_sum_tail_inline1126__ssa_v0: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 32), pl.Mem.Vec] = pl.tile.row_sum(
                        q_head_sq_tail_inline1085__ssa_v0, q_head_reduce_tmp_inline1116__ssa_v0
                    )
                    t__rm_a0_tmp_v0: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(q_head_sq_sum_tail_inline1126__ssa_v0, [1, 8])
                    t__row_major_tmp_v1: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_20, pl.const(34816, pl.INT64), 32), pl.Mem.Vec] = pl.tile.muls(t__rm_a0_tmp_v0, 0.001953125)
                    t__tmp_v60: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_20, pl.const(34816, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(t__row_major_tmp_v1, [8, 1])
                    t__rm_a0_tmp_v2: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_20, pl.const(34816, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(t__tmp_v60, [1, 8])
                    t__row_major_tmp_v3: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_20, pl.const(34816, pl.INT64), 32), pl.Mem.Vec] = pl.tile.adds(t__rm_a0_tmp_v2, 9.9999999999999995e-07)
                    t__tmp_v61: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_20, pl.const(34816, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(t__row_major_tmp_v3, [8, 1])
                    t__rm_a0_tmp_v4: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_20, pl.const(34816, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(t__tmp_v61, [1, 8])
                    t__row_major_tmp_v5: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_20, pl.const(34816, pl.INT64), 32), pl.Mem.Vec] = pl.tile.sqrt(t__rm_a0_tmp_v4)
                    t__tmp_v62: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_20, pl.const(34816, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(t__row_major_tmp_v5, [8, 1])
                    q_head_inv_rms_tail_inline1140__rm_a0_tmp_v6: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_20, pl.const(34816, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(t__tmp_v62, [1, 8])
                    q_head_inv_rms_tail_inline1140__row_major_tmp_v7: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 32), pl.Mem.Vec] = pl.tile.recip(
                        q_head_inv_rms_tail_inline1140__rm_a0_tmp_v6
                    )
                    q_head_inv_rms_tail_inline1140__ssa_v0: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(
                        q_head_inv_rms_tail_inline1140__row_major_tmp_v7, [8, 1]
                    )
                    t__tmp_v63: pl.Tile[[8, 448], pl.FP32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 16128), pl.Mem.Vec] = pl.tile.slice(q_head_dq_tail_v1_inline1086__ssa_v0, [8, 448], [0, 0])
                    q_nope_normed_tail_inline1149__ssa_v0: pl.Tile[[8, 448], pl.FP32, pl.MemRef(mem_vec_20, pl.const(34816, pl.INT64), 14336), pl.Mem.Vec] = pl.tile.row_expand_mul(
                        t__tmp_v63, q_head_inv_rms_tail_inline1140__ssa_v0
                    )
                    q_nope_bf16_tail_inline1083__ssa_v0: pl.Tile[[8, 448], pl.BF16, pl.MemRef(mem_vec_20, pl.const(34816, pl.INT64), 7168), pl.Mem.Vec] = pl.tile.cast(
                        q_nope_normed_tail_inline1149__ssa_v0, target_type=pl.BF16, mode="rint"
                    )
                    q_nope_valid_inline1148__ssa_v0: pl.Tile[
                        [8, 448], pl.BF16, pl.MemRef(mem_vec_20, pl.const(34816, pl.INT64), 7168), pl.Mem.Vec, pl.TileView(valid_shape=[valid_tail_rows_inline1107__ssa_v0, 448])
                    ] = pl.tile.set_validshape(q_nope_bf16_tail_inline1083__ssa_v0, valid_tail_rows_inline1107__ssa_v0, 448)
                    pl.tile.store(q_nope_valid_inline1148__ssa_v0, [out_tg_inline1101__ssa_v0, h0_tail_inline1122__ssa_v0], q_flat_inline1121__iter_v1)
                    q_rope_chunk_raw_tail_inline1099__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(20224, pl.INT64), 14592), pl.Mem.Vec] = pl.tile.slice(
                        q_head_dq_tail_v1_inline1086__ssa_v0, [8, 64], [0, 448]
                    )
                    q_rope_chunk_tail_inline1082__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.row_expand_mul(
                        q_rope_chunk_raw_tail_inline1099__ssa_v0, q_head_inv_rms_tail_inline1140__ssa_v0
                    )
                    t__tmp_v64: pl.Tile[[8, 64], pl.BF16, pl.MemRef(mem_vec_20, pl.const(34816, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.cast(
                        q_rope_chunk_tail_inline1082__ssa_v0, target_type=pl.BF16, mode="rint"
                    )
                    q_rope_chunk_tail_v1_inline1081__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(
                        t__tmp_v64, target_type=pl.FP32, mode="round"
                    )
                    q_rope_swapped_tail_inline1084__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_20, pl.const(34816, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.gather(
                        q_rope_chunk_tail_v1_inline1081__ssa_v0, q_swap_idx_tail_inline1091__ssa_v0, q_gather_tmp_inline1143__ssa_v0
                    )
                    q_rope_base_tail_inline1109__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.mul(
                        q_rope_chunk_tail_v1_inline1081__ssa_v0, q_cos_il_tail_inline1123__ssa_v0
                    )
                    q_rope_delta_tail_inline1080__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_20, pl.const(34816, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.mul(
                        q_rope_swapped_tail_inline1084__ssa_v0, q_sin_signed_tail_inline1097__ssa_v0
                    )
                    q_rope_rot_tail_inline1079__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.add(
                        q_rope_base_tail_inline1109__ssa_v0, q_rope_delta_tail_inline1080__ssa_v0
                    )
                    q_rope_bf16_tail_inline1078__ssa_v0: pl.Tile[[8, 64], pl.BF16, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.cast(
                        q_rope_rot_tail_inline1079__ssa_v0, target_type=pl.BF16, mode="rint"
                    )
                    q_rope_valid_inline1113__ssa_v0: pl.Tile[
                        [8, 64], pl.BF16, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 1024), pl.Mem.Vec, pl.TileView(valid_shape=[valid_tail_rows_inline1107__ssa_v0, 64])
                    ] = pl.tile.set_validshape(q_rope_bf16_tail_inline1078__ssa_v0, valid_tail_rows_inline1107__ssa_v0, 64)
                    pl.tile.store(q_rope_valid_inline1113__ssa_v0, [out_tg_inline1101__ssa_v0, h0_tail_inline1122__ssa_v0 + 448], q_flat_inline1121__iter_v1)
                q_flat_inline1121__phi_v7: pl.Tensor[[t_dim_inline1134__ssa_v0, 32768], pl.BF16, pl.MemRef("mem_ddr_117", pl.const(0, pl.INT64), 0)] = pl.yield_(q_flat_inline1121__iter_v1)
            q_flat_inline1121__rv_v2: pl.Tensor[[t_dim_inline1134__ssa_v0, 32768], pl.BF16, pl.MemRef("mem_ddr_118", pl.const(0, pl.INT64), 0)] = pl.yield_(q_flat_inline1121__phi_v7)
        return

    @pl.function(type=pl.FunctionType.Spmd)
    def qproj_dequant_rms_nope_rope_spmd(
        self,
        q_flat_inline1121__ssa_v0: pl.Out[pl.Tensor[[t_dim_inline1134__ssa_v0, 32768], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)]],
        tile_rows_inline228_inline821__ssa_v0: pl.Scalar[pl.INDEX],
        tile_base_inline229_inline820__idx_v0: pl.Scalar[pl.INDEX],
        qr_scale_pad_store_inline813__rv_v2: pl.Tensor[[512, 1], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 2048)],
        q_rope_cos_il_inline361__ssa_v0: pl.Tensor[[t_dim_inline362__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)],
        q_rope_sin_signed_inline360__ssa_v0: pl.Tensor[[t_dim_inline362__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)],
        q_rope_swap_idx_inline359__ssa_v0: pl.Tensor[[t_dim_inline362__ssa_v0, 64], pl.INT32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 0)],
        q_proj_i32_inline230_inline824__ssa_v9: pl.Tensor[[qproj_t_matmul_inline227_inline822__ssa_v0, 32768], pl.INT32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 0)],
        wq_b_scale__ssa_v0: pl.Tensor[[32768], pl.FP32, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 131072)],
    ):
        pl.func_attr({"mx_tensor_views_blocked": True})
        self.qproj_dequant_rms_nope_rope(
            q_flat_inline1121__ssa_v0,
            tile_rows_inline228_inline821__ssa_v0,
            tile_base_inline229_inline820__idx_v0,
            qr_scale_pad_store_inline813__rv_v2,
            q_rope_cos_il_inline361__ssa_v0,
            q_rope_sin_signed_inline360__ssa_v0,
            q_rope_swap_idx_inline359__ssa_v0,
            q_proj_i32_inline230_inline824__ssa_v9,
            wq_b_scale__ssa_v0,
            attrs={"arg_directions": [pl.adir.output_existing, pl.adir.scalar, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input]},
        )

    @pl.function(type=pl.FunctionType.AIC, level=pl.Level.AIC, role=pl.Role.SubWorker)
    def qproj_matmul(
        q_proj_i32_inline230_inline824__ssa_v0: pl.Out[pl.Tensor[[qproj_t_matmul_inline227_inline822__ssa_v0, 32768], pl.INT32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)]],
        qproj_full_rows_inline1076__ssa_v0: pl.Scalar[pl.INDEX],
        qr_i8_matmul_inline811__rv_v2: pl.Tensor[[512, 1024], pl.INT8, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 524288)],
        wq_b__ssa_v0: pl.Tensor[[1024, 32768], pl.INT8, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 33554432)],
        qproj_t_matmul_inline1065__ssa_v0: pl.Scalar[pl.INDEX],
        tile_rows_inline228_inline821__ssa_v0: pl.Scalar[pl.INDEX],
    ) -> pl.Tensor[[qproj_t_matmul_inline227_inline822__ssa_v0, 32768], pl.INT32]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        mem_acc_3: pl.Ptr = pl.tile.alloc(pl.Mem.Acc, 131072)
        mem_mat_4: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 16384)
        mem_mat_5: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 131072)
        mem_mat_6: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 16384)
        mem_mat_7: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 131072)
        mem_left_8: pl.Ptr = pl.tile.alloc(pl.Mem.Left, 4096)
        mem_right_9: pl.Ptr = pl.tile.alloc(pl.Mem.Right, 32768)
        mem_left_10: pl.Ptr = pl.tile.alloc(pl.Mem.Left, 4096)
        mem_right_11: pl.Ptr = pl.tile.alloc(pl.Mem.Right, 32768)
        mem_left_13: pl.Ptr = pl.tile.alloc(pl.Mem.Left, 4096)
        mem_left_15: pl.Ptr = pl.tile.alloc(pl.Mem.Left, 4096)
        qproj_worker_inline1070__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        for qproj_n_idx_inline1072__idx_v0, (q_proj_i32_inline230_inline824__iter_v1,) in pl.range(qproj_worker_inline1070__ssa_v0, 64, 24, init_values=(q_proj_i32_inline230_inline824__ssa_v0,)):
            w_col0_inline1068__ssa_v0: pl.Scalar[pl.INDEX] = qproj_n_idx_inline1072__idx_v0 * 512
            for t0_inline1073__idx_v0, (q_proj_i32_inline230_inline824__iter_v3,) in pl.range(0, qproj_full_rows_inline1076__ssa_v0, 64, init_values=(q_proj_i32_inline230_inline824__iter_v1,)):
                col_acc_inline1074__tile: pl.Tile[[64, 512], pl.INT32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 131072), pl.Mem.Acc] = pl.tile.create(
                    [64, 512], dtype=pl.INT32, target_memory=pl.Mem.Acc
                )
                for qr_proj_col0_inline1075__idx_v0, (col_acc_inline1074__iter_v1,) in pl.range(0, 1024, 512, init_values=(col_acc_inline1074__tile,)):
                    qr_i8_chunk_inline1067__tile: pl.Tile[[64, 256], pl.INT8, pl.MemRef(mem_mat_4, pl.const(0, pl.INT64), 16384), pl.Mem.Mat] = pl.tile.load(
                        qr_i8_matmul_inline811__rv_v2, [t0_inline1073__idx_v0, qr_proj_col0_inline1075__idx_v0], [64, 256], [64, 256], target_memory=pl.Mem.Mat
                    )
                    wq_chunk_inline1066__tile: pl.Tile[[256, 512], pl.INT8, pl.MemRef(mem_mat_5, pl.const(16384, pl.INT64), 131072), pl.Mem.Mat] = pl.tile.load(
                        wq_b__ssa_v0, [qr_proj_col0_inline1075__idx_v0, w_col0_inline1068__ssa_v0], [256, 512], [256, 512], target_memory=pl.Mem.Mat
                    )
                    qr_i8_chunk_inline1067__tile_1: pl.Tile[[64, 256], pl.INT8, pl.MemRef(mem_mat_6, pl.const(147456, pl.INT64), 16384), pl.Mem.Mat] = pl.tile.load(
                        qr_i8_matmul_inline811__rv_v2, [t0_inline1073__idx_v0, qr_proj_col0_inline1075__idx_v0 + 256], [64, 256], [64, 256], target_memory=pl.Mem.Mat
                    )
                    wq_chunk_inline1066__tile_1: pl.Tile[[256, 512], pl.INT8, pl.MemRef(mem_mat_7, pl.const(163840, pl.INT64), 131072), pl.Mem.Mat] = pl.tile.load(
                        wq_b__ssa_v0, [qr_proj_col0_inline1075__idx_v0 + 256, w_col0_inline1068__ssa_v0], [256, 512], [256, 512], target_memory=pl.Mem.Mat
                    )
                    for col_acc_inline1074__tile_l0_ko, (col_acc_inline1074__tile_l0_c,) in pl.range(0, 256, 128, init_values=(col_acc_inline1074__iter_v1,)):
                        col_acc_inline1074__tile_l0_a: pl.Tile[[64, 64], pl.INT8, pl.MemRef(mem_left_8, pl.const(12288, pl.INT64), 4096), pl.Mem.Left, pl.TileView(blayout=pl.TileLayout.row_major)] = (
                            pl.tile.extract(qr_i8_chunk_inline1067__tile, 0, col_acc_inline1074__tile_l0_ko, [64, 64], target_memory=pl.Mem.Left)
                        )
                        col_acc_inline1074__tile_l0_b: pl.Tile[[64, 512], pl.INT8, pl.MemRef(mem_right_9, pl.const(32768, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                            wq_chunk_inline1066__tile, col_acc_inline1074__tile_l0_ko, 0, [64, 512], target_memory=pl.Mem.Right
                        )
                        col_acc_inline1074__tile_l0_a_1: pl.Tile[[64, 64], pl.INT8, pl.MemRef(mem_left_10, pl.const(0, pl.INT64), 4096), pl.Mem.Left, pl.TileView(blayout=pl.TileLayout.row_major)] = (
                            pl.tile.extract(qr_i8_chunk_inline1067__tile, 0, col_acc_inline1074__tile_l0_ko + 64, [64, 64], target_memory=pl.Mem.Left)
                        )
                        col_acc_inline1074__tile_l0_b_1: pl.Tile[[64, 512], pl.INT8, pl.MemRef(mem_right_11, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                            wq_chunk_inline1066__tile, col_acc_inline1074__tile_l0_ko + 64, 0, [64, 512], target_memory=pl.Mem.Right
                        )
                        col_acc_inline1074__tile_l0_c_acc: pl.Tile[[64, 512], pl.INT32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 131072), pl.Mem.Acc] = pl.tile.matmul_acc(
                            col_acc_inline1074__tile_l0_c, col_acc_inline1074__tile_l0_a, col_acc_inline1074__tile_l0_b, qr_proj_col0_inline1075__idx_v0 == 0 and col_acc_inline1074__tile_l0_ko == 0
                        )
                        col_acc_inline1074__tile_l0_c_acc_1: pl.Tile[[64, 512], pl.INT32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 131072), pl.Mem.Acc] = pl.tile.matmul_acc(
                            col_acc_inline1074__tile_l0_c_acc,
                            col_acc_inline1074__tile_l0_a_1,
                            col_acc_inline1074__tile_l0_b_1,
                            qr_proj_col0_inline1075__idx_v0 == 0 and col_acc_inline1074__tile_l0_ko == -64,
                        )
                        col_acc_inline1074__tile_1: pl.Tile[[64, 512], pl.INT32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 131072), pl.Mem.Acc] = pl.yield_(col_acc_inline1074__tile_l0_c_acc_1)
                    for col_acc_inline1074__tile_l0_ko_1, (col_acc_inline1074__tile_l0_c_1,) in pl.range(0, 256, 128, init_values=(col_acc_inline1074__tile_1,)):
                        col_acc_inline1074__tile_l0_a_2: pl.Tile[
                            [64, 64], pl.INT8, pl.MemRef(mem_left_13, pl.const(4096, pl.INT64), 4096), pl.Mem.Left, pl.TileView(blayout=pl.TileLayout.row_major)
                        ] = pl.tile.extract(qr_i8_chunk_inline1067__tile_1, 0, col_acc_inline1074__tile_l0_ko_1, [64, 64], target_memory=pl.Mem.Left)
                        col_acc_inline1074__tile_l0_b_2: pl.Tile[[64, 512], pl.INT8, pl.MemRef(mem_right_9, pl.const(32768, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                            wq_chunk_inline1066__tile_1, col_acc_inline1074__tile_l0_ko_1, 0, [64, 512], target_memory=pl.Mem.Right
                        )
                        col_acc_inline1074__tile_l0_a_3: pl.Tile[
                            [64, 64], pl.INT8, pl.MemRef(mem_left_15, pl.const(8192, pl.INT64), 4096), pl.Mem.Left, pl.TileView(blayout=pl.TileLayout.row_major)
                        ] = pl.tile.extract(qr_i8_chunk_inline1067__tile_1, 0, col_acc_inline1074__tile_l0_ko_1 + 64, [64, 64], target_memory=pl.Mem.Left)
                        col_acc_inline1074__tile_l0_b_3: pl.Tile[[64, 512], pl.INT8, pl.MemRef(mem_right_11, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                            wq_chunk_inline1066__tile_1, col_acc_inline1074__tile_l0_ko_1 + 64, 0, [64, 512], target_memory=pl.Mem.Right
                        )
                        col_acc_inline1074__tile_l0_c_acc_2: pl.Tile[[64, 512], pl.INT32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 131072), pl.Mem.Acc] = pl.tile.matmul_acc(
                            col_acc_inline1074__tile_l0_c_1,
                            col_acc_inline1074__tile_l0_a_2,
                            col_acc_inline1074__tile_l0_b_2,
                            qr_proj_col0_inline1075__idx_v0 == -256 and col_acc_inline1074__tile_l0_ko_1 == 0,
                        )
                        col_acc_inline1074__tile_l0_c_acc_3: pl.Tile[[64, 512], pl.INT32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 131072), pl.Mem.Acc] = pl.tile.matmul_acc(
                            col_acc_inline1074__tile_l0_c_acc_2,
                            col_acc_inline1074__tile_l0_a_3,
                            col_acc_inline1074__tile_l0_b_3,
                            qr_proj_col0_inline1075__idx_v0 == -256 and col_acc_inline1074__tile_l0_ko_1 == -64,
                        )
                        col_acc_inline1074__tile_2: pl.Tile[[64, 512], pl.INT32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 131072), pl.Mem.Acc] = pl.yield_(col_acc_inline1074__tile_l0_c_acc_3)
                    col_acc_inline1074__rv_v2: pl.Tile[[64, 512], pl.INT32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 131072), pl.Mem.Acc] = pl.yield_(col_acc_inline1074__tile_2)
                q_proj_i32_inline230_inline824__tile: pl.Tensor[[qproj_t_matmul_inline227_inline822__ssa_v0, 32768], pl.INT32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                    col_acc_inline1074__rv_v2, [t0_inline1073__idx_v0, w_col0_inline1068__ssa_v0], q_proj_i32_inline230_inline824__iter_v3
                )
                q_proj_i32_inline230_inline824__rv_v4: pl.Tensor[[qproj_t_matmul_inline227_inline822__ssa_v0, 32768], pl.INT32, pl.MemRef("mem_ddr_19", pl.const(0, pl.INT64), 0)] = pl.yield_(
                    q_proj_i32_inline230_inline824__tile
                )
            tail_w_col0_inline1077__ssa_v0: pl.Scalar[pl.INDEX] = w_col0_inline1068__ssa_v0
            for tail_t0_inline1064__idx_v0, (q_proj_i32_inline230_inline824__iter_v6,) in pl.range(
                qproj_full_rows_inline1076__ssa_v0, qproj_t_matmul_inline1065__ssa_v0, 16, init_values=(q_proj_i32_inline230_inline824__rv_v4,)
            ):
                qproj_tail_rows_inline1063__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(tile_rows_inline228_inline821__ssa_v0 - tail_t0_inline1064__idx_v0, 16)
                tail_acc_inline1071__tile: pl.Tile[[16, 512], pl.INT32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 65536), pl.Mem.Acc] = pl.tile.create(
                    [16, 512], dtype=pl.INT32, target_memory=pl.Mem.Acc
                )
                for tail_qr_col0_inline1062__idx_v0, (tail_acc_inline1071__iter_v1,) in pl.range(0, 1024, 512, init_values=(tail_acc_inline1071__tile,)):
                    qr_i8_tail_inline1061__tile: pl.Tile[
                        [16, 256], pl.INT8, pl.MemRef(mem_mat_4, pl.const(0, pl.INT64), 4096), pl.Mem.Mat, pl.TileView(valid_shape=[qproj_tail_rows_inline1063__ssa_v0, 256])
                    ] = pl.tile.load(
                        qr_i8_matmul_inline811__rv_v2, [tail_t0_inline1064__idx_v0, tail_qr_col0_inline1062__idx_v0], [16, 256], [qproj_tail_rows_inline1063__ssa_v0, 256], target_memory=pl.Mem.Mat
                    )
                    wq_tail_inline1060__tile: pl.Tile[[256, 512], pl.INT8, pl.MemRef(mem_mat_5, pl.const(16384, pl.INT64), 131072), pl.Mem.Mat] = pl.tile.load(
                        wq_b__ssa_v0, [tail_qr_col0_inline1062__idx_v0, tail_w_col0_inline1077__ssa_v0], [256, 512], [256, 512], target_memory=pl.Mem.Mat
                    )
                    qr_i8_tail_inline1061__tile_1: pl.Tile[
                        [16, 256], pl.INT8, pl.MemRef(mem_mat_6, pl.const(147456, pl.INT64), 4096), pl.Mem.Mat, pl.TileView(valid_shape=[qproj_tail_rows_inline1063__ssa_v0, 256])
                    ] = pl.tile.load(
                        qr_i8_matmul_inline811__rv_v2,
                        [tail_t0_inline1064__idx_v0, tail_qr_col0_inline1062__idx_v0 + 256],
                        [16, 256],
                        [qproj_tail_rows_inline1063__ssa_v0, 256],
                        target_memory=pl.Mem.Mat,
                    )
                    wq_tail_inline1060__tile_1: pl.Tile[[256, 512], pl.INT8, pl.MemRef(mem_mat_7, pl.const(163840, pl.INT64), 131072), pl.Mem.Mat] = pl.tile.load(
                        wq_b__ssa_v0, [tail_qr_col0_inline1062__idx_v0 + 256, tail_w_col0_inline1077__ssa_v0], [256, 512], [256, 512], target_memory=pl.Mem.Mat
                    )
                    for tail_acc_inline1071__tile_l0_ko, (tail_acc_inline1071__tile_l0_c,) in pl.range(0, 256, 128, init_values=(tail_acc_inline1071__iter_v1,)):
                        tail_acc_inline1071__tile_l0_a: pl.Tile[
                            [16, 64],
                            pl.INT8,
                            pl.MemRef(mem_left_8, pl.const(12288, pl.INT64), 1024),
                            pl.Mem.Left,
                            pl.TileView(valid_shape=[qproj_tail_rows_inline1063__ssa_v0, 64], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal),
                        ] = pl.tile.extract(qr_i8_tail_inline1061__tile, 0, tail_acc_inline1071__tile_l0_ko, [16, 64], target_memory=pl.Mem.Left)
                        tail_acc_inline1071__tile_l0_b: pl.Tile[[64, 512], pl.INT8, pl.MemRef(mem_right_9, pl.const(32768, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                            wq_tail_inline1060__tile, tail_acc_inline1071__tile_l0_ko, 0, [64, 512], target_memory=pl.Mem.Right
                        )
                        tail_acc_inline1071__tile_l0_a_1: pl.Tile[
                            [16, 64],
                            pl.INT8,
                            pl.MemRef(mem_left_10, pl.const(0, pl.INT64), 1024),
                            pl.Mem.Left,
                            pl.TileView(valid_shape=[qproj_tail_rows_inline1063__ssa_v0, 64], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal),
                        ] = pl.tile.extract(qr_i8_tail_inline1061__tile, 0, tail_acc_inline1071__tile_l0_ko + 64, [16, 64], target_memory=pl.Mem.Left)
                        tail_acc_inline1071__tile_l0_b_1: pl.Tile[[64, 512], pl.INT8, pl.MemRef(mem_right_11, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                            wq_tail_inline1060__tile, tail_acc_inline1071__tile_l0_ko + 64, 0, [64, 512], target_memory=pl.Mem.Right
                        )
                        tail_acc_inline1071__tile_l0_c_acc: pl.Tile[[16, 512], pl.INT32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 65536), pl.Mem.Acc] = pl.tile.matmul_acc(
                            tail_acc_inline1071__tile_l0_c,
                            tail_acc_inline1071__tile_l0_a,
                            tail_acc_inline1071__tile_l0_b,
                            tail_qr_col0_inline1062__idx_v0 == 0 and tail_acc_inline1071__tile_l0_ko == 0,
                        )
                        tail_acc_inline1071__tile_l0_c_acc_1: pl.Tile[[16, 512], pl.INT32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 65536), pl.Mem.Acc] = pl.tile.matmul_acc(
                            tail_acc_inline1071__tile_l0_c_acc,
                            tail_acc_inline1071__tile_l0_a_1,
                            tail_acc_inline1071__tile_l0_b_1,
                            tail_qr_col0_inline1062__idx_v0 == 0 and tail_acc_inline1071__tile_l0_ko == -64,
                        )
                        tail_acc_inline1071__tile_1: pl.Tile[[16, 512], pl.INT32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 65536), pl.Mem.Acc] = pl.yield_(tail_acc_inline1071__tile_l0_c_acc_1)
                    for tail_acc_inline1071__tile_l0_ko_1, (tail_acc_inline1071__tile_l0_c_1,) in pl.range(0, 256, 128, init_values=(tail_acc_inline1071__tile_1,)):
                        tail_acc_inline1071__tile_l0_a_2: pl.Tile[
                            [16, 64],
                            pl.INT8,
                            pl.MemRef(mem_left_13, pl.const(4096, pl.INT64), 1024),
                            pl.Mem.Left,
                            pl.TileView(valid_shape=[qproj_tail_rows_inline1063__ssa_v0, 64], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal),
                        ] = pl.tile.extract(qr_i8_tail_inline1061__tile_1, 0, tail_acc_inline1071__tile_l0_ko_1, [16, 64], target_memory=pl.Mem.Left)
                        tail_acc_inline1071__tile_l0_b_2: pl.Tile[[64, 512], pl.INT8, pl.MemRef(mem_right_9, pl.const(32768, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                            wq_tail_inline1060__tile_1, tail_acc_inline1071__tile_l0_ko_1, 0, [64, 512], target_memory=pl.Mem.Right
                        )
                        tail_acc_inline1071__tile_l0_a_3: pl.Tile[
                            [16, 64],
                            pl.INT8,
                            pl.MemRef(mem_left_15, pl.const(8192, pl.INT64), 1024),
                            pl.Mem.Left,
                            pl.TileView(valid_shape=[qproj_tail_rows_inline1063__ssa_v0, 64], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal),
                        ] = pl.tile.extract(qr_i8_tail_inline1061__tile_1, 0, tail_acc_inline1071__tile_l0_ko_1 + 64, [16, 64], target_memory=pl.Mem.Left)
                        tail_acc_inline1071__tile_l0_b_3: pl.Tile[[64, 512], pl.INT8, pl.MemRef(mem_right_11, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                            wq_tail_inline1060__tile_1, tail_acc_inline1071__tile_l0_ko_1 + 64, 0, [64, 512], target_memory=pl.Mem.Right
                        )
                        tail_acc_inline1071__tile_l0_c_acc_2: pl.Tile[[16, 512], pl.INT32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 65536), pl.Mem.Acc] = pl.tile.matmul_acc(
                            tail_acc_inline1071__tile_l0_c_1,
                            tail_acc_inline1071__tile_l0_a_2,
                            tail_acc_inline1071__tile_l0_b_2,
                            tail_qr_col0_inline1062__idx_v0 == -256 and tail_acc_inline1071__tile_l0_ko_1 == 0,
                        )
                        tail_acc_inline1071__tile_l0_c_acc_3: pl.Tile[[16, 512], pl.INT32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 65536), pl.Mem.Acc] = pl.tile.matmul_acc(
                            tail_acc_inline1071__tile_l0_c_acc_2,
                            tail_acc_inline1071__tile_l0_a_3,
                            tail_acc_inline1071__tile_l0_b_3,
                            tail_qr_col0_inline1062__idx_v0 == -256 and tail_acc_inline1071__tile_l0_ko_1 == -64,
                        )
                        tail_acc_inline1071__tile_2: pl.Tile[[16, 512], pl.INT32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 65536), pl.Mem.Acc] = pl.yield_(tail_acc_inline1071__tile_l0_c_acc_3)
                    tail_acc_inline1071__rv_v2: pl.Tile[[16, 512], pl.INT32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 65536), pl.Mem.Acc] = pl.yield_(tail_acc_inline1071__tile_2)
                q_proj_i32_inline230_inline824__tile_1: pl.Tensor[[qproj_t_matmul_inline227_inline822__ssa_v0, 32768], pl.INT32, pl.MemRef("mem_ddr_19", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                    tail_acc_inline1071__rv_v2, [tail_t0_inline1064__idx_v0, tail_w_col0_inline1077__ssa_v0], q_proj_i32_inline230_inline824__iter_v6
                )
                q_proj_i32_inline230_inline824__rv_v7: pl.Tensor[[qproj_t_matmul_inline227_inline822__ssa_v0, 32768], pl.INT32, pl.MemRef("mem_ddr_36", pl.const(0, pl.INT64), 0)] = pl.yield_(
                    q_proj_i32_inline230_inline824__tile_1
                )
            q_proj_i32_inline230_inline824__rv_v2: pl.Tensor[[qproj_t_matmul_inline227_inline822__ssa_v0, 32768], pl.INT32, pl.MemRef("mem_ddr_37", pl.const(0, pl.INT64), 0)] = pl.yield_(
                q_proj_i32_inline230_inline824__rv_v7
            )
        return q_proj_i32_inline230_inline824__ssa_v0

    @pl.function(type=pl.FunctionType.Spmd)
    def qproj_matmul_spmd(
        self,
        q_proj_i32_inline230_inline824__ssa_v0: pl.Out[pl.Tensor[[qproj_t_matmul_inline227_inline822__ssa_v0, 32768], pl.INT32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)]],
        qproj_full_rows_inline1076__ssa_v0: pl.Scalar[pl.INDEX],
        qr_i8_matmul_inline811__rv_v2: pl.Tensor[[512, 1024], pl.INT8, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 524288)],
        wq_b__ssa_v0: pl.Tensor[[1024, 32768], pl.INT8, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 33554432)],
        qproj_t_matmul_inline1065__ssa_v0: pl.Scalar[pl.INDEX],
        tile_rows_inline228_inline821__ssa_v0: pl.Scalar[pl.INDEX],
    ) -> pl.Tensor[[qproj_t_matmul_inline227_inline822__ssa_v0, 32768], pl.INT32]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        q_proj_i32_inline230_inline824__rv_v2: pl.Tensor[[qproj_t_matmul_inline227_inline822__ssa_v0, 32768], pl.INT32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)] = self.qproj_matmul(
            q_proj_i32_inline230_inline824__ssa_v0,
            qproj_full_rows_inline1076__ssa_v0,
            qr_i8_matmul_inline811__rv_v2,
            wq_b__ssa_v0,
            qproj_t_matmul_inline1065__ssa_v0,
            tile_rows_inline228_inline821__ssa_v0,
            attrs={"arg_directions": [pl.adir.output_existing, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.scalar, pl.adir.scalar]},
        )
        return q_proj_i32_inline230_inline824__ssa_v0

    @pl.function(type=pl.FunctionType.AIC, level=pl.Level.AIC, role=pl.Role.SubWorker)
    def qr_proj_matmul(
        qr_fp32_inline220_inline818__rv_v2: pl.InOut[pl.Tensor[[qr_t_matmul_inline221_inline817__ssa_v0, 1024], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)]],
        qr_full_rows_inline970__ssa_v0: pl.Scalar[pl.INDEX],
        tile_base_inline223_inline823__idx_v0: pl.Scalar[pl.INDEX],
        tile_rows_inline222_inline816__ssa_v0: pl.Scalar[pl.INDEX],
        x_view_inline973__ssa_v0: pl.Tensor[[qa_tokens_inline968__ssa_v0, 4096], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)],
        wq_a__ssa_v0: pl.Tensor[[4096, 1024], pl.BF16, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 8388608)],
        qr_t_matmul_inline971__ssa_v0: pl.Scalar[pl.INDEX],
    ) -> pl.Tensor[[qr_t_matmul_inline221_inline817__ssa_v0, 1024], pl.FP32]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        mem_acc_3: pl.Ptr = pl.tile.alloc(pl.Mem.Acc, 8192)
        mem_mat_4: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 32768)
        mem_mat_5: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 16384)
        mem_mat_6: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 32768)
        mem_mat_7: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 16384)
        mem_left_8: pl.Ptr = pl.tile.alloc(pl.Mem.Left, 32768)
        mem_right_9: pl.Ptr = pl.tile.alloc(pl.Mem.Right, 16384)
        mem_left_10: pl.Ptr = pl.tile.alloc(pl.Mem.Left, 32768)
        mem_right_11: pl.Ptr = pl.tile.alloc(pl.Mem.Right, 16384)
        qbg_idx_inline966__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        q_a_col0_inline961__ssa_v0: pl.Scalar[pl.INDEX] = qbg_idx_inline966__ssa_v0 * 32
        qr_native_group_inline963__ssa_v0: pl.Scalar[pl.INDEX] = q_a_col0_inline961__ssa_v0 // 96
        for dense_t0_inline969__idx_v0, (qr_fp32_inline220_inline818__iter_v6,) in pl.range(0, qr_full_rows_inline970__ssa_v0, 64, init_values=(qr_fp32_inline220_inline818__rv_v2,)):
            dense_x0_inline964__ssa_v0: pl.Scalar[pl.INDEX] = tile_base_inline223_inline823__idx_v0 + dense_t0_inline969__idx_v0
            dense_acc_inline972__tile: pl.Tile[[64, 32], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 8192), pl.Mem.Acc] = pl.tile.create([64, 32], dtype=pl.FP32, target_memory=pl.Mem.Acc)
            for dense_k_inline967__idx_v0, (dense_acc_inline972__iter_v1,) in pl.range(0, 16, 2, init_values=(dense_acc_inline972__tile,)):
                dense_k_order_inline974__ssa_v0: pl.Scalar[pl.INDEX] = dense_k_inline967__idx_v0
                dense_k_order_inline974__ssa_v0_1: pl.Scalar[pl.INDEX] = dense_k_inline967__idx_v0 + 1
                if (
                    96 <= tile_rows_inline222_inline816__ssa_v0
                    and tile_rows_inline222_inline816__ssa_v0 <= 240
                    and tile_rows_inline222_inline816__ssa_v0 % 48 == 0
                    and qr_native_group_inline963__ssa_v0 < 9
                ):
                    dense_k_direction_inline980__ssa_v0: pl.Scalar[pl.INDEX] = dense_k_inline967__idx_v0
                    if qr_native_group_inline963__ssa_v0 % 2 == 1:
                        dense_k_direction_inline980__ssa_v1: pl.Scalar[pl.INDEX] = 15 - dense_k_inline967__idx_v0
                        dense_k_direction_inline980__phi_v2: pl.Scalar[pl.INDEX] = pl.yield_(dense_k_direction_inline980__ssa_v1)
                    else:
                        dense_k_direction_inline980__phi_v2: pl.Scalar[pl.INDEX] = pl.yield_(dense_k_direction_inline980__ssa_v0)
                    dense_k_order_inline974__ssa_v1: pl.Scalar[pl.INDEX] = (qr_native_group_inline963__ssa_v0 // 2 * 3 + dense_k_direction_inline980__phi_v2) % 16
                    dense_k_order_inline974__phi_v2: pl.Scalar[pl.INDEX] = pl.yield_(dense_k_order_inline974__ssa_v1)
                else:
                    dense_k_order_inline974__phi_v2: pl.Scalar[pl.INDEX] = pl.yield_(dense_k_order_inline974__ssa_v0)
                dense_d0_inline975__ssa_v0: pl.Scalar[pl.INDEX] = dense_k_order_inline974__phi_v2 * 256
                dense_x_inline976__tile: pl.Tile[[64, 256], pl.BF16, pl.MemRef(mem_mat_4, pl.const(0, pl.INT64), 32768), pl.Mem.Mat] = pl.tile.load(
                    x_view_inline973__ssa_v0, [dense_x0_inline964__ssa_v0, dense_d0_inline975__ssa_v0], [64, 256], [64, 256], target_memory=pl.Mem.Mat
                )
                dense_w_inline985__tile: pl.Tile[[256, 32], pl.BF16, pl.MemRef(mem_mat_5, pl.const(32768, pl.INT64), 16384), pl.Mem.Mat] = pl.tile.load(
                    wq_a__ssa_v0, [dense_d0_inline975__ssa_v0, q_a_col0_inline961__ssa_v0], [256, 32], [256, 32], target_memory=pl.Mem.Mat
                )
                if (
                    96 <= tile_rows_inline222_inline816__ssa_v0
                    and tile_rows_inline222_inline816__ssa_v0 <= 240
                    and tile_rows_inline222_inline816__ssa_v0 % 48 == 0
                    and qr_native_group_inline963__ssa_v0 < 9
                ):
                    dense_k_direction_inline980__ssa_v0_1: pl.Scalar[pl.INDEX] = dense_k_inline967__idx_v0 + 1
                    if qr_native_group_inline963__ssa_v0 % 2 == 1:
                        dense_k_direction_inline980__ssa_v1_1: pl.Scalar[pl.INDEX] = 14 - dense_k_inline967__idx_v0
                        dense_k_direction_inline980__phi_v2_1: pl.Scalar[pl.INDEX] = pl.yield_(dense_k_direction_inline980__ssa_v1_1)
                    else:
                        dense_k_direction_inline980__phi_v2_1: pl.Scalar[pl.INDEX] = pl.yield_(dense_k_direction_inline980__ssa_v0_1)
                    dense_k_order_inline974__ssa_v1_1: pl.Scalar[pl.INDEX] = (qr_native_group_inline963__ssa_v0 // 2 * 3 + dense_k_direction_inline980__phi_v2_1) % 16
                    dense_k_order_inline974__phi_v2_1: pl.Scalar[pl.INDEX] = pl.yield_(dense_k_order_inline974__ssa_v1_1)
                else:
                    dense_k_order_inline974__phi_v2_1: pl.Scalar[pl.INDEX] = pl.yield_(dense_k_order_inline974__ssa_v0_1)
                dense_d0_inline975__ssa_v0_1: pl.Scalar[pl.INDEX] = dense_k_order_inline974__phi_v2_1 * 256
                dense_x_inline976__tile_1: pl.Tile[[64, 256], pl.BF16, pl.MemRef(mem_mat_6, pl.const(49152, pl.INT64), 32768), pl.Mem.Mat] = pl.tile.load(
                    x_view_inline973__ssa_v0, [dense_x0_inline964__ssa_v0, dense_d0_inline975__ssa_v0_1], [64, 256], [64, 256], target_memory=pl.Mem.Mat
                )
                dense_w_inline985__tile_1: pl.Tile[[256, 32], pl.BF16, pl.MemRef(mem_mat_7, pl.const(81920, pl.INT64), 16384), pl.Mem.Mat] = pl.tile.load(
                    wq_a__ssa_v0, [dense_d0_inline975__ssa_v0_1, q_a_col0_inline961__ssa_v0], [256, 32], [256, 32], target_memory=pl.Mem.Mat
                )
                dense_x_inline976__tile_Left: pl.Tile[[64, 256], pl.BF16, pl.MemRef(mem_left_8, pl.const(32768, pl.INT64), 32768), pl.Mem.Left] = pl.tile.move(
                    dense_x_inline976__tile, target_memory=pl.Mem.Left
                )
                dense_w_inline985__tile_Right: pl.Tile[[256, 32], pl.BF16, pl.MemRef(mem_right_9, pl.const(16384, pl.INT64), 16384), pl.Mem.Right] = pl.tile.move(
                    dense_w_inline985__tile, target_memory=pl.Mem.Right
                )
                dense_acc_inline972__tile_1: pl.Tile[[64, 32], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 8192), pl.Mem.Acc] = pl.tile.matmul_acc(
                    dense_acc_inline972__iter_v1, dense_x_inline976__tile_Left, dense_w_inline985__tile_Right, dense_k_inline967__idx_v0 == 0
                )
                dense_x_inline976__tile_Left_1: pl.Tile[[64, 256], pl.BF16, pl.MemRef(mem_left_10, pl.const(0, pl.INT64), 32768), pl.Mem.Left] = pl.tile.move(
                    dense_x_inline976__tile_1, target_memory=pl.Mem.Left
                )
                dense_w_inline985__tile_Right_1: pl.Tile[[256, 32], pl.BF16, pl.MemRef(mem_right_11, pl.const(0, pl.INT64), 16384), pl.Mem.Right] = pl.tile.move(
                    dense_w_inline985__tile_1, target_memory=pl.Mem.Right
                )
                dense_acc_inline972__tile_2: pl.Tile[[64, 32], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 8192), pl.Mem.Acc] = pl.tile.matmul_acc(
                    dense_acc_inline972__tile_1, dense_x_inline976__tile_Left_1, dense_w_inline985__tile_Right_1, dense_k_inline967__idx_v0 == -1
                )
                dense_acc_inline972__rv_v2: pl.Tile[[64, 32], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 8192), pl.Mem.Acc] = pl.yield_(dense_acc_inline972__tile_2)
            qr_fp32_inline220_inline818__tile: pl.Tensor[[qr_t_matmul_inline221_inline817__ssa_v0, 1024], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                dense_acc_inline972__rv_v2, [dense_t0_inline969__idx_v0, q_a_col0_inline961__ssa_v0], qr_fp32_inline220_inline818__iter_v6, atomic=pl.AtomicType.Add
            )
            qr_fp32_inline220_inline818__rv_v7: pl.Tensor[[qr_t_matmul_inline221_inline817__ssa_v0, 1024], pl.FP32, pl.MemRef("mem_ddr_13", pl.const(0, pl.INT64), 0)] = pl.yield_(
                qr_fp32_inline220_inline818__tile
            )
        for t0_inline981__idx_v0, (qr_fp32_inline220_inline818__iter_v9,) in pl.range(
            qr_full_rows_inline970__ssa_v0, qr_t_matmul_inline971__ssa_v0, 16, init_values=(qr_fp32_inline220_inline818__rv_v7,)
        ):
            q_acc_inline977__tile: pl.Tile[[16, 32], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 2048), pl.Mem.Acc] = pl.tile.create([16, 32], dtype=pl.FP32, target_memory=pl.Mem.Acc)
            for db_inline983__idx_v0, (q_acc_inline977__iter_v1,) in pl.range(0, 16, 2, init_values=(q_acc_inline977__tile,)):
                qr_k_order_inline960__ssa_v0: pl.Scalar[pl.INDEX] = db_inline983__idx_v0
                qr_rows_inline989__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(tile_rows_inline222_inline816__ssa_v0 - t0_inline981__idx_v0, 16)
                x_t0_inline987__ssa_v0: pl.Scalar[pl.INDEX] = tile_base_inline223_inline823__idx_v0 + t0_inline981__idx_v0
                qr_k_order_inline960__ssa_v0_1: pl.Scalar[pl.INDEX] = db_inline983__idx_v0 + 1
                qr_rows_inline989__ssa_v0_1: pl.Scalar[pl.INDEX] = pl.min(tile_rows_inline222_inline816__ssa_v0 - t0_inline981__idx_v0, 16)
                x_t0_inline987__ssa_v0_1: pl.Scalar[pl.INDEX] = tile_base_inline223_inline823__idx_v0 + t0_inline981__idx_v0
                if (
                    96 <= tile_rows_inline222_inline816__ssa_v0
                    and tile_rows_inline222_inline816__ssa_v0 <= 240
                    and tile_rows_inline222_inline816__ssa_v0 % 48 == 0
                    and qr_native_group_inline963__ssa_v0 < 9
                ):
                    qr_k_direction_inline986__ssa_v0: pl.Scalar[pl.INDEX] = db_inline983__idx_v0
                    if qr_native_group_inline963__ssa_v0 % 2 == 1:
                        qr_k_direction_inline986__ssa_v1: pl.Scalar[pl.INDEX] = 15 - db_inline983__idx_v0
                        qr_k_direction_inline986__phi_v2: pl.Scalar[pl.INDEX] = pl.yield_(qr_k_direction_inline986__ssa_v1)
                    else:
                        qr_k_direction_inline986__phi_v2: pl.Scalar[pl.INDEX] = pl.yield_(qr_k_direction_inline986__ssa_v0)
                    qr_k_order_inline960__ssa_v1: pl.Scalar[pl.INDEX] = (qr_native_group_inline963__ssa_v0 // 2 * 3 + qr_k_direction_inline986__phi_v2) % 16
                    qr_k_order_inline960__phi_v2: pl.Scalar[pl.INDEX] = pl.yield_(qr_k_order_inline960__ssa_v1)
                else:
                    qr_k_order_inline960__phi_v2: pl.Scalar[pl.INDEX] = pl.yield_(qr_k_order_inline960__ssa_v0)
                qr_d0_inline988__ssa_v0: pl.Scalar[pl.INDEX] = qr_k_order_inline960__phi_v2 * 256
                q_x_chunk_bf16_inline965__tile: pl.Tile[
                    [16, 256], pl.BF16, pl.MemRef(mem_mat_5, pl.const(32768, pl.INT64), 8192), pl.Mem.Mat, pl.TileView(valid_shape=[qr_rows_inline989__ssa_v0, 256])
                ] = pl.tile.load(x_view_inline973__ssa_v0, [x_t0_inline987__ssa_v0, qr_d0_inline988__ssa_v0], [16, 256], [qr_rows_inline989__ssa_v0, 256], target_memory=pl.Mem.Mat)
                w_chunk_inline982__tile: pl.Tile[[256, 32], pl.BF16, pl.MemRef(mem_mat_4, pl.const(0, pl.INT64), 16384), pl.Mem.Mat] = pl.tile.load(
                    wq_a__ssa_v0, [qr_d0_inline988__ssa_v0, q_a_col0_inline961__ssa_v0], [256, 32], [256, 32], target_memory=pl.Mem.Mat
                )
                if (
                    96 <= tile_rows_inline222_inline816__ssa_v0
                    and tile_rows_inline222_inline816__ssa_v0 <= 240
                    and tile_rows_inline222_inline816__ssa_v0 % 48 == 0
                    and qr_native_group_inline963__ssa_v0 < 9
                ):
                    qr_k_direction_inline986__ssa_v0_1: pl.Scalar[pl.INDEX] = db_inline983__idx_v0 + 1
                    if qr_native_group_inline963__ssa_v0 % 2 == 1:
                        qr_k_direction_inline986__ssa_v1_1: pl.Scalar[pl.INDEX] = 14 - db_inline983__idx_v0
                        qr_k_direction_inline986__phi_v2_1: pl.Scalar[pl.INDEX] = pl.yield_(qr_k_direction_inline986__ssa_v1_1)
                    else:
                        qr_k_direction_inline986__phi_v2_1: pl.Scalar[pl.INDEX] = pl.yield_(qr_k_direction_inline986__ssa_v0_1)
                    qr_k_order_inline960__ssa_v1_1: pl.Scalar[pl.INDEX] = (qr_native_group_inline963__ssa_v0 // 2 * 3 + qr_k_direction_inline986__phi_v2_1) % 16
                    qr_k_order_inline960__phi_v2_1: pl.Scalar[pl.INDEX] = pl.yield_(qr_k_order_inline960__ssa_v1_1)
                else:
                    qr_k_order_inline960__phi_v2_1: pl.Scalar[pl.INDEX] = pl.yield_(qr_k_order_inline960__ssa_v0_1)
                qr_d0_inline988__ssa_v0_1: pl.Scalar[pl.INDEX] = qr_k_order_inline960__phi_v2_1 * 256
                q_x_chunk_bf16_inline965__tile_1: pl.Tile[
                    [16, 256], pl.BF16, pl.MemRef(mem_mat_7, pl.const(81920, pl.INT64), 8192), pl.Mem.Mat, pl.TileView(valid_shape=[qr_rows_inline989__ssa_v0_1, 256])
                ] = pl.tile.load(x_view_inline973__ssa_v0, [x_t0_inline987__ssa_v0_1, qr_d0_inline988__ssa_v0_1], [16, 256], [qr_rows_inline989__ssa_v0_1, 256], target_memory=pl.Mem.Mat)
                w_chunk_inline982__tile_1: pl.Tile[[256, 32], pl.BF16, pl.MemRef(mem_mat_6, pl.const(49152, pl.INT64), 16384), pl.Mem.Mat] = pl.tile.load(
                    wq_a__ssa_v0, [qr_d0_inline988__ssa_v0_1, q_a_col0_inline961__ssa_v0], [256, 32], [256, 32], target_memory=pl.Mem.Mat
                )
                q_x_chunk_bf16_inline965__tile_Left: pl.Tile[
                    [16, 256], pl.BF16, pl.MemRef(mem_left_8, pl.const(32768, pl.INT64), 8192), pl.Mem.Left, pl.TileView(valid_shape=[qr_rows_inline989__ssa_v0, 256])
                ] = pl.tile.move(q_x_chunk_bf16_inline965__tile, target_memory=pl.Mem.Left)
                w_chunk_inline982__tile_Right: pl.Tile[[256, 32], pl.BF16, pl.MemRef(mem_right_9, pl.const(16384, pl.INT64), 16384), pl.Mem.Right] = pl.tile.move(
                    w_chunk_inline982__tile, target_memory=pl.Mem.Right
                )
                q_acc_inline977__tile_1: pl.Tile[[16, 32], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 2048), pl.Mem.Acc] = pl.tile.matmul_acc(
                    q_acc_inline977__iter_v1, q_x_chunk_bf16_inline965__tile_Left, w_chunk_inline982__tile_Right, db_inline983__idx_v0 == 0
                )
                q_x_chunk_bf16_inline965__tile_Left_1: pl.Tile[
                    [16, 256], pl.BF16, pl.MemRef(mem_left_10, pl.const(0, pl.INT64), 8192), pl.Mem.Left, pl.TileView(valid_shape=[qr_rows_inline989__ssa_v0_1, 256])
                ] = pl.tile.move(q_x_chunk_bf16_inline965__tile_1, target_memory=pl.Mem.Left)
                w_chunk_inline982__tile_Right_1: pl.Tile[[256, 32], pl.BF16, pl.MemRef(mem_right_11, pl.const(0, pl.INT64), 16384), pl.Mem.Right] = pl.tile.move(
                    w_chunk_inline982__tile_1, target_memory=pl.Mem.Right
                )
                q_acc_inline977__tile_2: pl.Tile[[16, 32], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 2048), pl.Mem.Acc] = pl.tile.matmul_acc(
                    q_acc_inline977__tile_1, q_x_chunk_bf16_inline965__tile_Left_1, w_chunk_inline982__tile_Right_1, db_inline983__idx_v0 == -1
                )
                q_acc_inline977__rv_v2: pl.Tile[[16, 32], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 2048), pl.Mem.Acc] = pl.yield_(q_acc_inline977__tile_2)
            qr_fp32_inline220_inline818__tile_1: pl.Tensor[[qr_t_matmul_inline221_inline817__ssa_v0, 1024], pl.FP32, pl.MemRef("mem_ddr_13", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                q_acc_inline977__rv_v2, [t0_inline981__idx_v0, q_a_col0_inline961__ssa_v0], qr_fp32_inline220_inline818__iter_v9, atomic=pl.AtomicType.Add
            )
            qr_fp32_inline220_inline818__rv_v10: pl.Tensor[[qr_t_matmul_inline221_inline817__ssa_v0, 1024], pl.FP32, pl.MemRef("mem_ddr_24", pl.const(0, pl.INT64), 0)] = pl.yield_(
                qr_fp32_inline220_inline818__tile_1
            )
        return qr_fp32_inline220_inline818__rv_v2

    @pl.function(type=pl.FunctionType.Spmd)
    def qr_proj_matmul_spmd(
        self,
        qr_fp32_inline220_inline818__rv_v2: pl.InOut[pl.Tensor[[qr_t_matmul_inline221_inline817__ssa_v0, 1024], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)]],
        qr_full_rows_inline970__ssa_v0: pl.Scalar[pl.INDEX],
        tile_base_inline223_inline823__idx_v0: pl.Scalar[pl.INDEX],
        tile_rows_inline222_inline816__ssa_v0: pl.Scalar[pl.INDEX],
        x_view_inline973__ssa_v0: pl.Tensor[[qa_tokens_inline968__ssa_v0, 4096], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)],
        wq_a__ssa_v0: pl.Tensor[[4096, 1024], pl.BF16, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 8388608)],
        qr_t_matmul_inline971__ssa_v0: pl.Scalar[pl.INDEX],
    ) -> pl.Tensor[[qr_t_matmul_inline221_inline817__ssa_v0, 1024], pl.FP32]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        qr_fp32_inline220_inline818__rv_v10: pl.Tensor[[qr_t_matmul_inline221_inline817__ssa_v0, 1024], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)] = self.qr_proj_matmul(
            qr_fp32_inline220_inline818__rv_v2,
            qr_full_rows_inline970__ssa_v0,
            tile_base_inline223_inline823__idx_v0,
            tile_rows_inline222_inline816__ssa_v0,
            x_view_inline973__ssa_v0,
            wq_a__ssa_v0,
            qr_t_matmul_inline971__ssa_v0,
            attrs={"arg_directions": [pl.adir.inout, pl.adir.scalar, pl.adir.scalar, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.scalar]},
        )
        return qr_fp32_inline220_inline818__rv_v2

    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def qr_proj_seed(
        qr_fp32_inline220_inline818__ssa_v0: pl.Out[pl.Tensor[[qr_t_matmul_inline221_inline817__ssa_v0, 1024], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)]],
        qr_t_matmul_inline971__ssa_v0: pl.Scalar[pl.INDEX],
    ) -> pl.Tensor[[qr_t_matmul_inline221_inline817__ssa_v0, 1024], pl.FP32]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        mem_vec_1: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 2048)
        for ts0_inline979__idx_v0, (qr_fp32_inline220_inline818__iter_v1,) in pl.range(0, qr_t_matmul_inline971__ssa_v0, 16, init_values=(qr_fp32_inline220_inline818__ssa_v0,)):
            for nseed0_inline962__idx_v0, (qr_fp32_inline220_inline818__iter_v3,) in pl.range(0, 1024, 32, init_values=(qr_fp32_inline220_inline818__iter_v1,)):
                qr_seed_inline984__tile: pl.Tile[[16, 32], pl.FP32, pl.MemRef(mem_vec_1, pl.const(0, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.full([16, 32], dtype=pl.FP32, value=0.0)
                qr_fp32_inline220_inline818__tile: pl.Tensor[[qr_t_matmul_inline221_inline817__ssa_v0, 1024], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                    qr_seed_inline984__tile, [ts0_inline979__idx_v0, nseed0_inline962__idx_v0], qr_fp32_inline220_inline818__iter_v3
                )
                qr_fp32_inline220_inline818__rv_v4: pl.Tensor[[qr_t_matmul_inline221_inline817__ssa_v0, 1024], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)] = pl.yield_(
                    qr_fp32_inline220_inline818__tile
                )
            qr_fp32_inline220_inline818__rv_v2: pl.Tensor[[qr_t_matmul_inline221_inline817__ssa_v0, 1024], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)] = pl.yield_(
                qr_fp32_inline220_inline818__rv_v4
            )
        return qr_fp32_inline220_inline818__ssa_v0

    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def qr_rms_norm_quant(
        tile_rows_inline222_inline816__ssa_v0: pl.Scalar[pl.INDEX],
        tile_base_inline223_inline823__idx_v0: pl.Scalar[pl.INDEX],
        qr_fp32_inline220_inline818__rv_v10: pl.Tensor[[qr_t_matmul_inline221_inline817__ssa_v0, 1024], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)],
        gamma_cq__ssa_v0: pl.Tensor[[1024], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 2048)],
        qr_scale_pad_store_inline813__iter_v1: pl.InOut[pl.Tensor[[512, 1], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 2048)]],
        qr_scale_view_inline1040__ssa_v0: pl.Out[pl.Tensor[[t_dim_inline1002__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)]],
        qr_i8_matmul_inline811__iter_v1: pl.InOut[pl.Tensor[[512, 1024], pl.INT8, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 524288)]],
        qr_view_inline1007__ssa_v0: pl.Out[pl.Tensor[[t_dim_inline1002__ssa_v0, 1024], pl.INT8, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 0)]],
    ) -> tuple[pl.Tensor[[512, 1], pl.FP32], pl.Tensor[[512, 1024], pl.INT8], pl.Tensor[[t_dim_inline1002__ssa_v0, 1], pl.FP32]]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        mem_vec_6: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32768)
        mem_vec_7: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
        mem_vec_20: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32)
        mem_vec_21: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32)
        mem_vec_23: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 512)
        mem_vec_25: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 512)
        mem_vec_32: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 8192)
        mem_vec_41: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 8192)
        mem_vec_45: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32)
        tg_idx_inline1032__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        tg_inline1046__ssa_v0: pl.Scalar[pl.INDEX] = tg_idx_inline1032__ssa_v0 * 8
        valid_rows_inline1037__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(tile_rows_inline222_inline816__ssa_v0 - tg_inline1046__ssa_v0, 8)
        out_tg_inline1029__ssa_v0: pl.Scalar[pl.INDEX] = tile_base_inline223_inline823__idx_v0 + tg_inline1046__ssa_v0
        t__tile: pl.Tile[[8, 1024], pl.FP32, pl.MemRef(mem_vec_6, pl.const(17504, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.load(
            qr_fp32_inline220_inline818__rv_v10, [tg_inline1046__ssa_v0, 0], [8, 1024], [8, 1024], target_memory=pl.Mem.Vec
        )
        t__tile_1: pl.Tile[[8, 1024], pl.BF16, pl.MemRef(mem_vec_7, pl.const(50272, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.cast(t__tile, target_type=pl.BF16, mode="rint")
        qr_rms_full_inline249_inline1013__tile: pl.Tile[[8, 1024], pl.FP32, pl.MemRef(mem_vec_6, pl.const(17504, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.cast(
            t__tile_1, target_type=pl.FP32, mode="round"
        )
        qr_square_full_inline237_inline1034__tile: pl.Tile[[8, 1024], pl.FP32, pl.MemRef(mem_vec_6, pl.const(17504, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.mul(
            qr_rms_full_inline249_inline1013__tile, qr_rms_full_inline249_inline1013__tile
        )
        t__tile_2: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_6, pl.const(17504, pl.INT64), 30720), pl.Mem.Vec] = pl.tile.slice(qr_square_full_inline237_inline1034__tile, [8, 512], [0, 0])
        t__tile_3: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_6, pl.const(19552, pl.INT64), 30720), pl.Mem.Vec] = pl.tile.slice(qr_square_full_inline237_inline1034__tile, [8, 512], [0, 512])
        qr_square_half_inline244_inline1033__tile: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_6, pl.const(17504, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.add(t__tile_2, t__tile_3)
        t__tile_4: pl.Tile[[8, 256], pl.FP32, pl.MemRef(mem_vec_6, pl.const(17504, pl.INT64), 15360), pl.Mem.Vec] = pl.tile.slice(qr_square_half_inline244_inline1033__tile, [8, 256], [0, 0])
        t__tile_5: pl.Tile[[8, 256], pl.FP32, pl.MemRef(mem_vec_6, pl.const(18528, pl.INT64), 15360), pl.Mem.Vec] = pl.tile.slice(qr_square_half_inline244_inline1033__tile, [8, 256], [0, 256])
        qr_square_quarter_inline245_inline1020__tile: pl.Tile[[8, 256], pl.FP32, pl.MemRef(mem_vec_6, pl.const(17504, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.add(t__tile_4, t__tile_5)
        t__tile_6: pl.Tile[[8, 128], pl.FP32, pl.MemRef(mem_vec_6, pl.const(17504, pl.INT64), 7680), pl.Mem.Vec] = pl.tile.slice(qr_square_quarter_inline245_inline1020__tile, [8, 128], [0, 0])
        t__tile_7: pl.Tile[[8, 128], pl.FP32, pl.MemRef(mem_vec_6, pl.const(18016, pl.INT64), 7680), pl.Mem.Vec] = pl.tile.slice(qr_square_quarter_inline245_inline1020__tile, [8, 128], [0, 128])
        qr_square_eighth_inline240_inline1024__tile: pl.Tile[[8, 128], pl.FP32, pl.MemRef(mem_vec_6, pl.const(17504, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.add(t__tile_6, t__tile_7)
        t__tile_8: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_6, pl.const(17504, pl.INT64), 3840), pl.Mem.Vec] = pl.tile.slice(qr_square_eighth_inline240_inline1024__tile, [8, 64], [0, 0])
        t__tile_9: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_6, pl.const(17760, pl.INT64), 3840), pl.Mem.Vec] = pl.tile.slice(qr_square_eighth_inline240_inline1024__tile, [8, 64], [0, 64])
        qr_square_sixteenth_inline239_inline1012__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_7, pl.const(50272, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.add(t__tile_8, t__tile_9)
        tmp_tile: pl.Tile[[8, 128], pl.FP32, pl.MemRef(mem_vec_6, pl.const(17504, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.create([8, 128], dtype=pl.FP32, target_memory=pl.Mem.Vec)
        t__tile_10: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_32, pl.const(1088, pl.INT64), 32), pl.Mem.Vec] = pl.tile.row_sum(qr_square_sixteenth_inline239_inline1012__tile, tmp_tile)
        qr_sq_sum_inline243_inline1051__tile: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_32, pl.const(1088, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(t__tile_10, [1, 8])
        t__tile_11: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_6, pl.const(17504, pl.INT64), 32), pl.Mem.Vec] = pl.tile.muls(qr_sq_sum_inline243_inline1051__tile, 0.0009765625)
        qr_variance_inline247_inline1005__tile: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_6, pl.const(17504, pl.INT64), 32), pl.Mem.Vec] = pl.tile.adds(t__tile_11, 9.9999999999999995e-07)
        qr_rms_approx_inline238_inline1006__tile: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_7, pl.const(50272, pl.INT64), 32), pl.Mem.Vec] = pl.tile.sqrt(qr_variance_inline247_inline1005__tile)
        qr_variance_bits_inline254_inline1011__tile: pl.Tile[[1, 8], pl.INT32, pl.MemRef(mem_vec_6, pl.const(17504, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reinterpret_view(
            qr_variance_inline247_inline1005__tile, dtype=pl.INT32
        )
        qr_approx_bits_inline241_inline1028__tile: pl.Tile[[1, 8], pl.INT32, pl.MemRef(mem_vec_7, pl.const(50272, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reinterpret_view(
            qr_rms_approx_inline238_inline1006__tile, dtype=pl.INT32
        )
        qr_rounded_bits_inline250_inline1025__tile: pl.Tile[[1, 8], pl.INT32, pl.MemRef(mem_vec_41, pl.const(9280, pl.INT64), 32), pl.Mem.Vec] = pl.tile.full([1, 8], dtype=pl.INT32, value=0)
        for qr_sqrt_row_inline236_inline1004__idx_v0 in pl.range(8):
            t__tile_12: pl.Scalar[pl.INT32] = pl.tile.read(qr_variance_bits_inline254_inline1011__tile, [0, qr_sqrt_row_inline236_inline1004__idx_v0])
            qr_x_bits_inline262_inline1001__ssa_v0: pl.Scalar[pl.INT64] = pl.cast(t__tile_12, pl.INT64)
            t__tile_13: pl.Scalar[pl.INT32] = pl.tile.read(qr_approx_bits_inline241_inline1028__tile, [0, qr_sqrt_row_inline236_inline1004__idx_v0])
            qr_y_bits_inline252_inline1000__ssa_v0: pl.Scalar[pl.INT64] = pl.cast(t__tile_13, pl.INT64)
            if 8388608 <= qr_x_bits_inline262_inline1001__ssa_v0 and qr_x_bits_inline262_inline1001__ssa_v0 < 2139095040:
                qr_x_exp_inline251_inline998__ssa_v0: pl.Scalar[pl.INT64] = qr_x_bits_inline262_inline1001__ssa_v0 // 8388608
                qr_x_sig_inline242_inline1015__ssa_v0: pl.Scalar[pl.INT64] = qr_x_bits_inline262_inline1001__ssa_v0 % 8388608 + 8388608
                for _qr_sqrt_step_inline255_inline1018__idx_v0, (qr_y_bits_inline252_inline1000__iter_v1,) in pl.range(2, init_values=(qr_y_bits_inline252_inline1000__ssa_v0,)):
                    qr_y_exp_inline257_inline1003__ssa_v0: pl.Scalar[pl.INT64] = qr_y_bits_inline252_inline1000__iter_v1 // 8388608
                    qr_y_sig_inline259_inline1010__ssa_v0: pl.Scalar[pl.INT64] = qr_y_bits_inline252_inline1000__iter_v1 % 8388608 + 8388608
                    qr_upper_mid_inline253_inline1021__ssa_v0: pl.Scalar[pl.INDEX] = qr_y_sig_inline259_inline1010__ssa_v0 * 2 + 1
                    qr_upper_square_inline260_inline1026__ssa_v0: pl.Scalar[pl.INDEX] = qr_upper_mid_inline253_inline1021__ssa_v0 * qr_upper_mid_inline253_inline1021__ssa_v0
                    qr_x_upper_inline248_inline1008__ssa_v0: pl.Scalar[pl.INT64] = (
                        qr_x_sig_inline242_inline1015__ssa_v0 << qr_x_exp_inline251_inline998__ssa_v0 - qr_y_exp_inline257_inline1003__ssa_v0 * 2 + 152
                    )
                    qr_previous_bits_inline263_inline1022__ssa_v0: pl.Scalar[pl.INT64] = qr_y_bits_inline252_inline1000__iter_v1 - 1
                    qr_previous_exp_inline264_inline1048__ssa_v0: pl.Scalar[pl.INT64] = qr_previous_bits_inline263_inline1022__ssa_v0 // 8388608
                    qr_previous_sig_inline258_inline1035__ssa_v0: pl.Scalar[pl.INT64] = qr_previous_bits_inline263_inline1022__ssa_v0 % 8388608 + 8388608
                    qr_lower_mid_inline261_inline999__ssa_v0: pl.Scalar[pl.INDEX] = qr_previous_sig_inline258_inline1035__ssa_v0 * 2 + 1
                    qr_lower_square_inline246_inline1030__ssa_v0: pl.Scalar[pl.INDEX] = qr_lower_mid_inline261_inline999__ssa_v0 * qr_lower_mid_inline261_inline999__ssa_v0
                    qr_x_lower_inline256_inline1042__ssa_v0: pl.Scalar[pl.INT64] = (
                        qr_x_sig_inline242_inline1015__ssa_v0 << qr_x_exp_inline251_inline998__ssa_v0 - qr_previous_exp_inline264_inline1048__ssa_v0 * 2 + 152
                    )
                    if (
                        qr_upper_square_inline260_inline1026__ssa_v0 < qr_x_upper_inline248_inline1008__ssa_v0
                        or qr_x_upper_inline248_inline1008__ssa_v0 == qr_upper_square_inline260_inline1026__ssa_v0
                        and qr_y_bits_inline252_inline1000__iter_v1 % 2 == 1
                    ):
                        qr_y_bits_inline252_inline1000__ssa_v3: pl.Scalar[pl.INT64] = qr_y_bits_inline252_inline1000__iter_v1 + 1
                        qr_y_bits_inline252_inline1000__phi_v6: pl.Scalar[pl.INT64] = pl.yield_(qr_y_bits_inline252_inline1000__ssa_v3)
                    else:
                        if (
                            qr_x_lower_inline256_inline1042__ssa_v0 < qr_lower_square_inline246_inline1030__ssa_v0
                            or qr_x_lower_inline256_inline1042__ssa_v0 == qr_lower_square_inline246_inline1030__ssa_v0
                            and qr_y_bits_inline252_inline1000__iter_v1 % 2 == 1
                        ):
                            qr_y_bits_inline252_inline1000__ssa_v4: pl.Scalar[pl.INT64] = qr_y_bits_inline252_inline1000__iter_v1 - 1
                            qr_y_bits_inline252_inline1000__phi_v5: pl.Scalar[pl.INT64] = pl.yield_(qr_y_bits_inline252_inline1000__ssa_v4)
                        else:
                            qr_y_bits_inline252_inline1000__phi_v5: pl.Scalar[pl.INT64] = pl.yield_(qr_y_bits_inline252_inline1000__iter_v1)
                        qr_y_bits_inline252_inline1000__phi_v6: pl.Scalar[pl.INT64] = pl.yield_(qr_y_bits_inline252_inline1000__phi_v5)
                    qr_y_bits_inline252_inline1000__rv_v2: pl.Scalar[pl.INT64] = pl.yield_(qr_y_bits_inline252_inline1000__phi_v6)
                pl.tile.write(qr_rounded_bits_inline250_inline1025__tile, [0, qr_sqrt_row_inline236_inline1004__idx_v0], pl.cast(qr_y_bits_inline252_inline1000__rv_v2, pl.INT32))
            else:
                pl.tile.write(qr_rounded_bits_inline250_inline1025__tile, [0, qr_sqrt_row_inline236_inline1004__idx_v0], pl.cast(qr_y_bits_inline252_inline1000__ssa_v0, pl.INT32))
        qr_rms_inline235_inline1038__tile: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_41, pl.const(9280, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reinterpret_view(
            qr_rounded_bits_inline250_inline1025__tile, dtype=pl.FP32
        )
        qr_inv_rms_inline234_inline1052__tile: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_20, pl.const(0, pl.INT64), 32), pl.Mem.Vec] = pl.tile.full([1, 8], dtype=pl.FP32, value=0.0)
        for qr_rms_row_inline233_inline1041__idx_v0 in pl.range(8):
            qr_rms_value_inline232_inline1043__tile: pl.Scalar[pl.FP32] = pl.tile.read(qr_rms_inline235_inline1038__tile, [0, qr_rms_row_inline233_inline1041__idx_v0])
            pl.tile.write(qr_inv_rms_inline234_inline1052__tile, [0, qr_rms_row_inline233_inline1041__idx_v0], 1.0 / qr_rms_value_inline232_inline1043__tile)
        _qr_sq_sum_inline1044__ssa_v0: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_32, pl.const(1088, pl.INT64), 32), pl.Mem.Vec] = qr_sq_sum_inline243_inline1051__tile
        qr_inv_rms_inline1047__ssa_v0: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_20, pl.const(0, pl.INT64), 32), pl.Mem.Vec] = qr_inv_rms_inline234_inline1052__tile
        _qr_rms_inline1049__ssa_v0: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_41, pl.const(9280, pl.INT64), 32), pl.Mem.Vec] = qr_rms_inline235_inline1038__tile
        qr_inv_rms_t_inline1050__tile: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_20, pl.const(0, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(qr_inv_rms_inline1047__ssa_v0, [8, 1])
        qr_tile_amax_inline1009__tile: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_21, pl.const(32, pl.INT64), 32), pl.Mem.Vec] = pl.tile.full([1, 8], dtype=pl.FP32, value=0.0001)
        for qr_max_col0_inline1031__idx_v0, (qr_tile_amax_inline1009__iter_v1,) in pl.range(0, 1024, 512, init_values=(qr_tile_amax_inline1009__tile,)):
            t__tile_14: pl.Tile[[8, 256], pl.FP32, pl.MemRef(mem_vec_6, pl.const(17504, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.load(
                qr_fp32_inline220_inline818__rv_v10, [tg_inline1046__ssa_v0, qr_max_col0_inline1031__idx_v0], [8, 256], [8, 256], target_memory=pl.Mem.Vec
            )
            t__tile_15: pl.Tile[[256], pl.BF16, pl.MemRef(mem_vec_23, pl.const(64, pl.INT64), 512), pl.Mem.Vec] = pl.tile.load(
                gamma_cq__ssa_v0, [qr_max_col0_inline1031__idx_v0], [256], [256], target_memory=pl.Mem.Vec
            )
            t__tile_16: pl.Tile[[8, 256], pl.FP32, pl.MemRef(mem_vec_7, pl.const(50272, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.load(
                qr_fp32_inline220_inline818__rv_v10, [tg_inline1046__ssa_v0, qr_max_col0_inline1031__idx_v0 + 256], [8, 256], [8, 256], target_memory=pl.Mem.Vec
            )
            t__tile_17: pl.Tile[[256], pl.BF16, pl.MemRef(mem_vec_25, pl.const(576, pl.INT64), 512), pl.Mem.Vec] = pl.tile.load(
                gamma_cq__ssa_v0, [qr_max_col0_inline1031__idx_v0 + 256], [256], [256], target_memory=pl.Mem.Vec
            )
            t__tile_18: pl.Tile[[8, 256], pl.BF16, pl.MemRef(mem_vec_32, pl.const(1088, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(t__tile_14, target_type=pl.BF16, mode="rint")
            qr_max_chunk_inline1039__tile: pl.Tile[[8, 256], pl.FP32, pl.MemRef(mem_vec_6, pl.const(17504, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.cast(t__tile_18, target_type=pl.FP32, mode="round")
            gamma_max_cast_inline1036__tile: pl.Tile[[256], pl.FP32, pl.MemRef(mem_vec_32, pl.const(1088, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.cast(t__tile_15, target_type=pl.FP32, mode="round")
            gamma_max_chunk_inline1053__tile: pl.Tile[[1, 256], pl.FP32, pl.MemRef(mem_vec_32, pl.const(1088, pl.INT64), 1024), pl.Mem.Vec] = gamma_max_cast_inline1036__tile
            t__tile_19: pl.Tile[[8, 256], pl.FP32, pl.MemRef(mem_vec_6, pl.const(17504, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.row_expand_mul(
                qr_max_chunk_inline1039__tile, qr_inv_rms_t_inline1050__tile
            )
            qr_normalized_inline1054__tile: pl.Tile[[8, 256], pl.FP32, pl.MemRef(mem_vec_6, pl.const(17504, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.col_expand_mul(
                t__tile_19, gamma_max_chunk_inline1053__tile
            )
            t__tile_20: pl.Tile[[8, 256], pl.FP32, pl.MemRef(mem_vec_6, pl.const(17504, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.abs(qr_normalized_inline1054__tile)
            tmp_tile_1: pl.Tile[[8, 256], pl.FP32, pl.MemRef(mem_vec_32, pl.const(1088, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.create([8, 256], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            t__tile_21: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_23, pl.const(64, pl.INT64), 32), pl.Mem.Vec] = pl.tile.row_max(t__tile_20, tmp_tile_1)
            qr_normalized_max_inline1055__tile: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_23, pl.const(64, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(t__tile_21, [1, 8])
            qr_tile_amax_inline1009__tile_1: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_6, pl.const(17504, pl.INT64), 32), pl.Mem.Vec] = pl.tile.maximum(
                qr_tile_amax_inline1009__iter_v1, qr_normalized_max_inline1055__tile
            )
            t__tile_22: pl.Tile[[8, 256], pl.BF16, pl.MemRef(mem_vec_41, pl.const(9280, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(t__tile_16, target_type=pl.BF16, mode="rint")
            qr_max_chunk_inline1039__tile_1: pl.Tile[[8, 256], pl.FP32, pl.MemRef(mem_vec_7, pl.const(50272, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.cast(t__tile_22, target_type=pl.FP32, mode="round")
            gamma_max_cast_inline1036__tile_1: pl.Tile[[256], pl.FP32, pl.MemRef(mem_vec_41, pl.const(9280, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.cast(t__tile_17, target_type=pl.FP32, mode="round")
            gamma_max_chunk_inline1053__tile_1: pl.Tile[[1, 256], pl.FP32, pl.MemRef(mem_vec_41, pl.const(9280, pl.INT64), 1024), pl.Mem.Vec] = gamma_max_cast_inline1036__tile_1
            t__tile_23: pl.Tile[[8, 256], pl.FP32, pl.MemRef(mem_vec_7, pl.const(50272, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.row_expand_mul(
                qr_max_chunk_inline1039__tile_1, qr_inv_rms_t_inline1050__tile
            )
            qr_normalized_inline1054__tile_1: pl.Tile[[8, 256], pl.FP32, pl.MemRef(mem_vec_7, pl.const(50272, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.col_expand_mul(
                t__tile_23, gamma_max_chunk_inline1053__tile_1
            )
            t__tile_24: pl.Tile[[8, 256], pl.FP32, pl.MemRef(mem_vec_7, pl.const(50272, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.abs(qr_normalized_inline1054__tile_1)
            tmp_tile_2: pl.Tile[[8, 256], pl.FP32, pl.MemRef(mem_vec_41, pl.const(9280, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.create([8, 256], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            t__tile_25: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_25, pl.const(576, pl.INT64), 32), pl.Mem.Vec] = pl.tile.row_max(t__tile_24, tmp_tile_2)
            qr_normalized_max_inline1055__tile_1: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_25, pl.const(576, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(t__tile_25, [1, 8])
            qr_tile_amax_inline1009__tile_2: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_21, pl.const(32, pl.INT64), 32), pl.Mem.Vec] = pl.tile.maximum(
                qr_tile_amax_inline1009__tile_1, qr_normalized_max_inline1055__tile_1
            )
            qr_tile_amax_inline1009__rv_v2: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_21, pl.const(32, pl.INT64), 32), pl.Mem.Vec] = pl.yield_(qr_tile_amax_inline1009__tile_2)
        qr_scale_quant_row_inline1057__tile: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_45, pl.const(17472, pl.INT64), 32), pl.Mem.Vec] = pl.tile.full([1, 8], dtype=pl.FP32, value=0.0)
        qr_scale_dq_row_inline1059__tile: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_6, pl.const(17504, pl.INT64), 32), pl.Mem.Vec] = pl.tile.full([1, 8], dtype=pl.FP32, value=0.0)
        for qr_scale_row_inline1045__idx_v0 in pl.range(8):
            qr_amax_value_inline1027__tile: pl.Scalar[pl.FP32] = pl.tile.read(qr_tile_amax_inline1009__rv_v2, [0, qr_scale_row_inline1045__idx_v0])
            qr_quant_value_inline1014__ssa_v0: pl.Scalar[pl.FP32] = 127.0 / qr_amax_value_inline1027__tile
            pl.tile.write(qr_scale_quant_row_inline1057__tile, [0, qr_scale_row_inline1045__idx_v0], qr_quant_value_inline1014__ssa_v0)
            pl.tile.write(qr_scale_dq_row_inline1059__tile, [0, qr_scale_row_inline1045__idx_v0], 1.0 / qr_quant_value_inline1014__ssa_v0)
        qr_scale_quant_t_inline1056__tile: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_45, pl.const(17472, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(qr_scale_quant_row_inline1057__tile, [8, 1])
        qr_tile_scale_dq_inline1058__tile: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_6, pl.const(17504, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(qr_scale_dq_row_inline1059__tile, [8, 1])
        qr_scale_pad_store_inline813__tile: pl.Tensor[[512, 1], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 2048)] = pl.tile.store(
            qr_tile_scale_dq_inline1058__tile, [tg_inline1046__ssa_v0, 0], qr_scale_pad_store_inline813__iter_v1
        )
        if valid_rows_inline1037__ssa_v0 == 8:
            qr_scale_view_inline1040__tile: pl.Tensor[[t_dim_inline1002__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                qr_tile_scale_dq_inline1058__tile, [out_tg_inline1029__ssa_v0, 0], qr_scale_view_inline1040__ssa_v0
            )
        else:
            qr_scale_tail_inline997__ssa_v0: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_6, pl.const(17504, pl.INT64), 32), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1037__ssa_v0, 1])] = (
                pl.tile.load(qr_scale_pad_store_inline813__tile, [tg_inline1046__ssa_v0, 0], [8, 1], [valid_rows_inline1037__ssa_v0, 1], target_memory=pl.Mem.Vec)
            )
            qr_scale_view_inline1040__store: pl.Tensor[[t_dim_inline1002__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                qr_scale_tail_inline997__ssa_v0, [out_tg_inline1029__ssa_v0, 0], qr_scale_view_inline1040__ssa_v0
            )
        for qa_inline996__idx_v0, (qr_i8_matmul_inline811__iter_v3, qr_view_inline1007__iter_v1) in pl.range(0, 1024, 512, init_values=(qr_i8_matmul_inline811__iter_v1, qr_view_inline1007__ssa_v0)):
            t__tile_26: pl.Tile[[8, 256], pl.FP32, pl.MemRef(mem_vec_6, pl.const(17504, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.load(
                qr_fp32_inline220_inline818__rv_v10, [tg_inline1046__ssa_v0, qa_inline996__idx_v0], [8, 256], [8, 256], target_memory=pl.Mem.Vec
            )
            t__tile_27: pl.Tile[[256], pl.BF16, pl.MemRef(mem_vec_23, pl.const(64, pl.INT64), 512), pl.Mem.Vec] = pl.tile.load(
                gamma_cq__ssa_v0, [qa_inline996__idx_v0], [256], [256], target_memory=pl.Mem.Vec
            )
            t__tile_28: pl.Tile[[8, 256], pl.FP32, pl.MemRef(mem_vec_7, pl.const(50272, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.load(
                qr_fp32_inline220_inline818__rv_v10, [tg_inline1046__ssa_v0, qa_inline996__idx_v0 + 256], [8, 256], [8, 256], target_memory=pl.Mem.Vec
            )
            t__tile_29: pl.Tile[[256], pl.BF16, pl.MemRef(mem_vec_25, pl.const(576, pl.INT64), 512), pl.Mem.Vec] = pl.tile.load(
                gamma_cq__ssa_v0, [qa_inline996__idx_v0 + 256], [256], [256], target_memory=pl.Mem.Vec
            )
            t__tile_30: pl.Tile[[8, 256], pl.BF16, pl.MemRef(mem_vec_32, pl.const(1088, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(t__tile_26, target_type=pl.BF16, mode="rint")
            qr_chunk_inline1019__tile: pl.Tile[[8, 256], pl.FP32, pl.MemRef(mem_vec_6, pl.const(17504, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.cast(t__tile_30, target_type=pl.FP32, mode="round")
            gamma_q_cast_inline995__tile: pl.Tile[[256], pl.FP32, pl.MemRef(mem_vec_32, pl.const(1088, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.cast(t__tile_27, target_type=pl.FP32, mode="round")
            gamma_q_chunk_inline994__tile: pl.Tile[[1, 256], pl.FP32, pl.MemRef(mem_vec_32, pl.const(1088, pl.INT64), 1024), pl.Mem.Vec] = gamma_q_cast_inline995__tile
            t__tile_31: pl.Tile[[8, 256], pl.FP32, pl.MemRef(mem_vec_6, pl.const(17504, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.row_expand_mul(qr_chunk_inline1019__tile, qr_inv_rms_t_inline1050__tile)
            qr_q_normed_inline1017__tile: pl.Tile[[8, 256], pl.FP32, pl.MemRef(mem_vec_6, pl.const(17504, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.col_expand_mul(
                t__tile_31, gamma_q_chunk_inline994__tile
            )
            qr_q_scaled_inline993__tile: pl.Tile[[8, 256], pl.FP32, pl.MemRef(mem_vec_6, pl.const(17504, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.row_expand_mul(
                qr_q_normed_inline1017__tile, qr_scale_quant_t_inline1056__tile
            )
            qr_q_i32_inline991__tile: pl.Tile[[8, 256], pl.INT32, pl.MemRef(mem_vec_6, pl.const(17504, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.cast(
                qr_q_scaled_inline993__tile, target_type=pl.INT32, mode="rint"
            )
            qr_q_half_inline992__tile: pl.Tile[[8, 256], pl.FP16, pl.MemRef(mem_vec_6, pl.const(17504, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(
                qr_q_i32_inline991__tile, target_type=pl.FP16, mode="round"
            )
            qr_q_i8_inline990__tile: pl.Tile[[8, 256], pl.INT8, pl.MemRef(mem_vec_6, pl.const(17504, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(
                qr_q_half_inline992__tile, target_type=pl.INT8, mode="trunc"
            )
            qr_i8_matmul_inline811__tile: pl.Tensor[[512, 1024], pl.INT8, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 524288)] = pl.tile.store(
                qr_q_i8_inline990__tile, [tg_inline1046__ssa_v0, qa_inline996__idx_v0], qr_i8_matmul_inline811__iter_v3
            )
            if valid_rows_inline1037__ssa_v0 == 8:
                qr_view_inline1007__tile: pl.Tensor[[t_dim_inline1002__ssa_v0, 1024], pl.INT8, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                    qr_q_i8_inline990__tile, [out_tg_inline1029__ssa_v0, qa_inline996__idx_v0], qr_view_inline1007__iter_v1
                )
                qr_view_inline1007__phi_v4: pl.Tensor[[t_dim_inline1002__ssa_v0, 1024], pl.INT8, pl.MemRef("mem_ddr_62", pl.const(0, pl.INT64), 0)] = pl.yield_(qr_view_inline1007__tile)
            else:
                qr_q_tail_inline1016__ssa_v0: pl.Tile[
                    [8, 256], pl.INT8, pl.MemRef(mem_vec_6, pl.const(17504, pl.INT64), 2048), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1037__ssa_v0, 256])
                ] = pl.tile.load(qr_i8_matmul_inline811__tile, [tg_inline1046__ssa_v0, qa_inline996__idx_v0], [8, 256], [valid_rows_inline1037__ssa_v0, 256], target_memory=pl.Mem.Vec)
                pl.tile.store(qr_q_tail_inline1016__ssa_v0, [out_tg_inline1029__ssa_v0, qa_inline996__idx_v0], qr_view_inline1007__iter_v1)
                qr_view_inline1007__phi_v4: pl.Tensor[[t_dim_inline1002__ssa_v0, 1024], pl.INT8, pl.MemRef("mem_ddr_62", pl.const(0, pl.INT64), 0)] = pl.yield_(qr_view_inline1007__iter_v1)
            t__tile_32: pl.Tile[[8, 256], pl.BF16, pl.MemRef(mem_vec_41, pl.const(9280, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(t__tile_28, target_type=pl.BF16, mode="rint")
            qr_chunk_inline1019__tile_1: pl.Tile[[8, 256], pl.FP32, pl.MemRef(mem_vec_7, pl.const(50272, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.cast(t__tile_32, target_type=pl.FP32, mode="round")
            gamma_q_cast_inline995__tile_1: pl.Tile[[256], pl.FP32, pl.MemRef(mem_vec_41, pl.const(9280, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.cast(t__tile_29, target_type=pl.FP32, mode="round")
            gamma_q_chunk_inline994__tile_1: pl.Tile[[1, 256], pl.FP32, pl.MemRef(mem_vec_41, pl.const(9280, pl.INT64), 1024), pl.Mem.Vec] = gamma_q_cast_inline995__tile_1
            t__tile_33: pl.Tile[[8, 256], pl.FP32, pl.MemRef(mem_vec_7, pl.const(50272, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.row_expand_mul(
                qr_chunk_inline1019__tile_1, qr_inv_rms_t_inline1050__tile
            )
            qr_q_normed_inline1017__tile_1: pl.Tile[[8, 256], pl.FP32, pl.MemRef(mem_vec_7, pl.const(50272, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.col_expand_mul(
                t__tile_33, gamma_q_chunk_inline994__tile_1
            )
            qr_q_scaled_inline993__tile_1: pl.Tile[[8, 256], pl.FP32, pl.MemRef(mem_vec_7, pl.const(50272, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.row_expand_mul(
                qr_q_normed_inline1017__tile_1, qr_scale_quant_t_inline1056__tile
            )
            qr_q_i32_inline991__tile_1: pl.Tile[[8, 256], pl.INT32, pl.MemRef(mem_vec_7, pl.const(50272, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.cast(
                qr_q_scaled_inline993__tile_1, target_type=pl.INT32, mode="rint"
            )
            qr_q_half_inline992__tile_1: pl.Tile[[8, 256], pl.FP16, pl.MemRef(mem_vec_7, pl.const(50272, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(
                qr_q_i32_inline991__tile_1, target_type=pl.FP16, mode="round"
            )
            qr_q_i8_inline990__tile_1: pl.Tile[[8, 256], pl.INT8, pl.MemRef(mem_vec_7, pl.const(50272, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(
                qr_q_half_inline992__tile_1, target_type=pl.INT8, mode="trunc"
            )
            qr_i8_matmul_inline811__tile_1: pl.Tensor[[512, 1024], pl.INT8, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 524288)] = pl.tile.store(
                qr_q_i8_inline990__tile_1, [tg_inline1046__ssa_v0, qa_inline996__idx_v0 + 256], qr_i8_matmul_inline811__tile
            )
            if valid_rows_inline1037__ssa_v0 == 8:
                qr_view_inline1007__tile_1: pl.Tensor[[t_dim_inline1002__ssa_v0, 1024], pl.INT8, pl.MemRef("mem_ddr_62", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                    qr_q_i8_inline990__tile_1, [out_tg_inline1029__ssa_v0, qa_inline996__idx_v0 + 256], qr_view_inline1007__phi_v4
                )
                qr_view_inline1007__phi_v4_1: pl.Tensor[[t_dim_inline1002__ssa_v0, 1024], pl.INT8, pl.MemRef("mem_ddr_73", pl.const(0, pl.INT64), 0)] = pl.yield_(qr_view_inline1007__tile_1)
            else:
                qr_q_tail_inline1016__ssa_v0_1: pl.Tile[
                    [8, 256], pl.INT8, pl.MemRef(mem_vec_7, pl.const(50272, pl.INT64), 2048), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1037__ssa_v0, 256])
                ] = pl.tile.load(qr_i8_matmul_inline811__tile_1, [tg_inline1046__ssa_v0, qa_inline996__idx_v0 + 256], [8, 256], [valid_rows_inline1037__ssa_v0, 256], target_memory=pl.Mem.Vec)
                pl.tile.store(qr_q_tail_inline1016__ssa_v0_1, [out_tg_inline1029__ssa_v0, qa_inline996__idx_v0 + 256], qr_view_inline1007__phi_v4)
                qr_view_inline1007__phi_v4_1: pl.Tensor[[t_dim_inline1002__ssa_v0, 1024], pl.INT8, pl.MemRef("mem_ddr_73", pl.const(0, pl.INT64), 0)] = pl.yield_(qr_view_inline1007__phi_v4)
            qr_i8_matmul_inline811__rv_v4, qr_view_inline1007__rv_v2 = pl.yield_(qr_i8_matmul_inline811__tile_1, qr_view_inline1007__phi_v4_1)
        return qr_scale_pad_store_inline813__iter_v1, qr_i8_matmul_inline811__iter_v1, qr_scale_view_inline1040__ssa_v0

    @pl.function(type=pl.FunctionType.Spmd)
    def qr_rms_norm_quant_spmd(
        self,
        tile_rows_inline222_inline816__ssa_v0: pl.Scalar[pl.INDEX],
        tile_base_inline223_inline823__idx_v0: pl.Scalar[pl.INDEX],
        qr_fp32_inline220_inline818__rv_v10: pl.Tensor[[qr_t_matmul_inline221_inline817__ssa_v0, 1024], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)],
        gamma_cq__ssa_v0: pl.Tensor[[1024], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 2048)],
        qr_scale_pad_store_inline813__iter_v1: pl.InOut[pl.Tensor[[512, 1], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 2048)]],
        qr_scale_view_inline1040__ssa_v0: pl.Out[pl.Tensor[[t_dim_inline1002__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)]],
        qr_i8_matmul_inline811__iter_v1: pl.InOut[pl.Tensor[[512, 1024], pl.INT8, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 524288)]],
        qr_view_inline1007__ssa_v0: pl.Out[pl.Tensor[[t_dim_inline1002__ssa_v0, 1024], pl.INT8, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 0)]],
    ) -> tuple[pl.Tensor[[512, 1], pl.FP32], pl.Tensor[[512, 1024], pl.INT8]]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        ret__tmp_v0: pl.Tuple[pl.Tensor[[512, 1], pl.FP32], pl.Tensor[[512, 1024], pl.INT8], pl.Tensor[[t_dim_inline1002__ssa_v0, 1], pl.FP32]] = self.qr_rms_norm_quant(
            tile_rows_inline222_inline816__ssa_v0,
            tile_base_inline223_inline823__idx_v0,
            qr_fp32_inline220_inline818__rv_v10,
            gamma_cq__ssa_v0,
            qr_scale_pad_store_inline813__iter_v1,
            qr_scale_view_inline1040__ssa_v0,
            qr_i8_matmul_inline811__iter_v1,
            qr_view_inline1007__ssa_v0,
            attrs={"arg_directions": [pl.adir.scalar, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.inout, pl.adir.output_existing, pl.adir.inout, pl.adir.output_existing]},
        )
        qr_scale_pad_store_inline813__ssa_v3: pl.Tensor[[512, 1], pl.FP32, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 2048)] = ret__tmp_v0[0]
        qr_i8_matmul_inline811__rv_v4: pl.Tensor[[512, 1024], pl.INT8, pl.MemRef("mem_ddr_7", pl.const(0, pl.INT64), 524288)] = ret__tmp_v0[1]
        qr_scale_view_inline1040__ssa_v2: pl.Tensor[[t_dim_inline1002__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_8", pl.const(0, pl.INT64), 0)] = ret__tmp_v0[2]
        return qr_scale_pad_store_inline813__iter_v1, qr_i8_matmul_inline811__iter_v1

    @pl.function(type=pl.FunctionType.Orchestration, level=pl.Level.CHIP, role=pl.Role.Orchestrator, auto_scope=False)
    def diagnose_main_query(
        self,
        x__ssa_v0: pl.Tensor[[T_DYN, 4096], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)],
        wq_a__ssa_v0: pl.Tensor[[4096, 1024], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 8388608)],
        wq_b__ssa_v0: pl.Tensor[[1024, 32768], pl.INT8, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 33554432)],
        wq_b_scale__ssa_v0: pl.Tensor[[32768], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 131072)],
        wkv__ssa_v0: pl.Tensor[[4096, 512], pl.BF16, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 4194304)],
        cos__ssa_v0: pl.Tensor[[T_DYN, 64], pl.FP32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 0)],
        sin__ssa_v0: pl.Tensor[[T_DYN, 64], pl.FP32, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 0)],
        gamma_cq__ssa_v0: pl.Tensor[[1024], pl.BF16, pl.MemRef("mem_ddr_7", pl.const(0, pl.INT64), 2048)],
        gamma_ckv__ssa_v0: pl.Tensor[[512], pl.BF16, pl.MemRef("mem_ddr_8", pl.const(0, pl.INT64), 1024)],
        query__ssa_v0: pl.Out[pl.Tensor[[T_DYN, 64, 512], pl.BF16, pl.MemRef("mem_ddr_9", pl.const(0, pl.INT64), 0)]],
        qr__ssa_v0: pl.Out[pl.Tensor[[T_DYN, 1024], pl.INT8, pl.MemRef("mem_ddr_10", pl.const(0, pl.INT64), 0)]],
        qr_scale__ssa_v0: pl.Out[pl.Tensor[[T_DYN, 1], pl.FP32, pl.MemRef("mem_ddr_11", pl.const(0, pl.INT64), 0)]],
    ) -> tuple[pl.Tensor[[T_DYN, 64, 512], pl.BF16], pl.Tensor[[T_DYN, 1024], pl.INT8], pl.Tensor[[T_DYN, 1], pl.FP32]]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        kv__ssa_v0: pl.Tensor[[T_DYN, 512], pl.BF16, pl.MemRef("mem_ddr_12", pl.const(0, pl.INT64), 0)] = pl.tensor.create([T_DYN, 512], dtype=pl.BF16, layout=pl.TensorLayout.ND)
        gate__ssa_v0: pl.Scalar[pl.TASK_ID] = pl.system.task_dummy(deps=[])
        t_dim_inline362__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(x__ssa_v0, 0)
        q_rope_cos_il_inline361__ssa_v0: pl.Tensor[[t_dim_inline362__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_13", pl.const(0, pl.INT64), 0)] = pl.tensor.create(
            [t_dim_inline362__ssa_v0, 64], dtype=pl.FP32, layout=pl.TensorLayout.ND
        )
        q_rope_sin_signed_inline360__ssa_v0: pl.Tensor[[t_dim_inline362__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_14", pl.const(0, pl.INT64), 0)] = pl.tensor.create(
            [t_dim_inline362__ssa_v0, 64], dtype=pl.FP32, layout=pl.TensorLayout.ND
        )
        q_rope_swap_idx_inline359__ssa_v0: pl.Tensor[[t_dim_inline362__ssa_v0, 64], pl.INT32, pl.MemRef("mem_ddr_15", pl.const(0, pl.INT64), 0)] = pl.tensor.create(
            [t_dim_inline362__ssa_v0, 64], dtype=pl.INT32, layout=pl.TensorLayout.ND
        )
        t_dim_inline799__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(cos__ssa_v0, 0)
        rope_cos_view_inline795__ssa_v0: pl.Tensor[[t_dim_inline799__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 0)] = pl.tensor.reshape(
            cos__ssa_v0, [t_dim_inline799__ssa_v0, 64]
        )
        rope_sin_view_inline793__ssa_v0: pl.Tensor[[t_dim_inline799__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 0)] = pl.tensor.reshape(
            sin__ssa_v0, [t_dim_inline799__ssa_v0, 64]
        )
        rope_cos_il_view_inline794__ssa_v0: pl.Tensor[[t_dim_inline799__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_13", pl.const(0, pl.INT64), 0)] = pl.tensor.reshape(
            q_rope_cos_il_inline361__ssa_v0, [t_dim_inline799__ssa_v0, 64]
        )
        rope_sin_signed_view_inline783__ssa_v0: pl.Tensor[[t_dim_inline799__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_14", pl.const(0, pl.INT64), 0)] = pl.tensor.reshape(
            q_rope_sin_signed_inline360__ssa_v0, [t_dim_inline799__ssa_v0, 64]
        )
        rope_swap_idx_view_inline780__ssa_v0: pl.Tensor[[t_dim_inline799__ssa_v0, 64], pl.INT32, pl.MemRef("mem_ddr_15", pl.const(0, pl.INT64), 0)] = pl.tensor.reshape(
            q_rope_swap_idx_inline359__ssa_v0, [t_dim_inline799__ssa_v0, 64]
        )
        token_tiles_inline779__ssa_v0: pl.Scalar[pl.INDEX] = (t_dim_inline799__ssa_v0 + 7) // 8
        ret__tmp_v0: pl.Tuple[pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(
            self.q_rope_prepare_spmd,
            rope_cos_il_view_inline794__ssa_v0,
            rope_sin_signed_view_inline783__ssa_v0,
            rope_swap_idx_view_inline780__ssa_v0,
            token_tiles_inline779__ssa_v0,
            t_dim_inline799__ssa_v0,
            rope_cos_view_inline795__ssa_v0,
            rope_sin_view_inline793__ssa_v0,
            core_num=pl.min(token_tiles_inline779__ssa_v0, 48),
            allow_early_resolve=True,
            attrs={"arg_directions": [pl.adir.output_existing, pl.adir.output_existing, pl.adir.output_existing, pl.adir.scalar, pl.adir.scalar, pl.adir.input, pl.adir.input]},
        )
        tid__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0[0]
        qr_i8_matmul_inline811__ssa_v0: pl.Tensor[[512, 1024], pl.INT8, pl.MemRef("mem_ddr_16", pl.const(0, pl.INT64), 524288)] = pl.tensor.create(
            [512, 1024], dtype=pl.INT8, layout=pl.TensorLayout.ND
        )
        qr_scale_pad_store_inline813__ssa_v0: pl.Tensor[[512, 1], pl.FP32, pl.MemRef("mem_ddr_17", pl.const(0, pl.INT64), 2048)] = pl.tensor.create(
            [512, 1], dtype=pl.FP32, layout=pl.TensorLayout.ND, manual_dep=True
        )
        t_dim_inline224_inline814__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(x__ssa_v0, 0)
        for tile_base_inline223_inline823__idx_v0, (qr_i8_matmul_inline811__iter_v1, qr_scale_pad_store_inline813__iter_v1) in pl.range(
            0,
            t_dim_inline224_inline814__ssa_v0,
            512,
            init_values=(qr_i8_matmul_inline811__ssa_v0, qr_scale_pad_store_inline813__ssa_v0),
            attrs={"iter_arg_rebind_0": False, "iter_arg_rebind_1": False},
        ):
            tile_rows_inline222_inline816__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(t_dim_inline224_inline814__ssa_v0 - tile_base_inline223_inline823__idx_v0, 512)
            with pl.scope():
                qr_t_matmul_inline221_inline817__ssa_v0: pl.Scalar[pl.INDEX] = (tile_rows_inline222_inline816__ssa_v0 + 15) // 16 * 16
                qr_fp32_inline220_inline818__ssa_v0: pl.Tensor[[qr_t_matmul_inline221_inline817__ssa_v0, 1024], pl.FP32, pl.MemRef("mem_ddr_18", pl.const(0, pl.INT64), 0)] = pl.tensor.create(
                    [qr_t_matmul_inline221_inline817__ssa_v0, 1024], dtype=pl.FP32, layout=pl.TensorLayout.ND
                )
                qa_tokens_inline968__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(x__ssa_v0, 0)
                x_view_inline973__ssa_v0: pl.Tensor[[qa_tokens_inline968__ssa_v0, 4096], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)] = pl.tensor.reshape(
                    x__ssa_v0, [qa_tokens_inline968__ssa_v0, 4096]
                )
                qr_t_matmul_inline971__ssa_v0: pl.Scalar[pl.INDEX] = (tile_rows_inline222_inline816__ssa_v0 + 15) // 16 * 16
                qr_full_rows_inline970__ssa_v0: pl.Scalar[pl.INDEX] = tile_rows_inline222_inline816__ssa_v0 // 64 * 64
                qr_fp32_inline220_inline818__rv_v2: pl.Tensor[[qr_t_matmul_inline221_inline817__ssa_v0, 1024], pl.FP32, pl.MemRef("mem_ddr_19", pl.const(0, pl.INT64), 0)] = self.qr_proj_seed(
                    qr_fp32_inline220_inline818__ssa_v0, qr_t_matmul_inline971__ssa_v0, attrs={"arg_directions": [pl.adir.inout, pl.adir.scalar]}
                )
                ret__tmp_v0_1: pl.Tuple[pl.Tensor[[qr_t_matmul_inline221_inline817__ssa_v0, 1024], pl.FP32], pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(
                    self.qr_proj_matmul_spmd,
                    qr_fp32_inline220_inline818__rv_v2,
                    qr_full_rows_inline970__ssa_v0,
                    tile_base_inline223_inline823__idx_v0,
                    tile_rows_inline222_inline816__ssa_v0,
                    x_view_inline973__ssa_v0,
                    wq_a__ssa_v0,
                    qr_t_matmul_inline971__ssa_v0,
                    core_num=32,
                    allow_early_resolve=True,
                    attrs={"arg_directions": [pl.adir.inout, pl.adir.scalar, pl.adir.scalar, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.scalar]},
                )
                qr_fp32_inline220_inline818__rv_v10: pl.Tensor[[qr_t_matmul_inline221_inline817__ssa_v0, 1024], pl.FP32, pl.MemRef("mem_ddr_20", pl.const(0, pl.INT64), 0)] = ret__tmp_v0_1[0]
                tid__ssa_v1: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_1[1]
                t_dim_inline1002__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(qr__ssa_v0, 0)
                qr_view_inline1007__ssa_v0: pl.Tensor[[t_dim_inline1002__ssa_v0, 1024], pl.INT8, pl.MemRef("mem_ddr_10", pl.const(0, pl.INT64), 0)] = pl.tensor.reshape(
                    qr__ssa_v0, [t_dim_inline1002__ssa_v0, 1024]
                )
                qr_scale_view_inline1040__ssa_v0: pl.Tensor[[t_dim_inline1002__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_11", pl.const(0, pl.INT64), 0)] = pl.tensor.reshape(
                    qr_scale__ssa_v0, [t_dim_inline1002__ssa_v0, 1]
                )
                qr_token_tiles_inline1023__ssa_v0: pl.Scalar[pl.INDEX] = (tile_rows_inline222_inline816__ssa_v0 + 7) // 8
                ret__tmp_v0_2: pl.Tuple[pl.Tensor[[512, 1], pl.FP32], pl.Tensor[[512, 1024], pl.INT8], pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(
                    self.qr_rms_norm_quant_spmd,
                    tile_rows_inline222_inline816__ssa_v0,
                    tile_base_inline223_inline823__idx_v0,
                    qr_fp32_inline220_inline818__rv_v10,
                    gamma_cq__ssa_v0,
                    qr_scale_pad_store_inline813__iter_v1,
                    qr_scale_view_inline1040__ssa_v0,
                    qr_i8_matmul_inline811__iter_v1,
                    qr_view_inline1007__ssa_v0,
                    core_num=qr_token_tiles_inline1023__ssa_v0,
                    allow_early_resolve=True,
                    attrs={"arg_directions": [pl.adir.scalar, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.inout, pl.adir.inout, pl.adir.inout, pl.adir.inout]},
                )
                qr_scale_pad_store_inline813__ssa_v3: pl.Tensor[[512, 1], pl.FP32, pl.MemRef("mem_ddr_21", pl.const(0, pl.INT64), 2048)] = ret__tmp_v0_2[0]
                qr_i8_matmul_inline811__rv_v4: pl.Tensor[[512, 1024], pl.INT8, pl.MemRef("mem_ddr_22", pl.const(0, pl.INT64), 524288)] = ret__tmp_v0_2[1]
                tid__ssa_v2: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_2[2]
            qr_i8_matmul_inline811__rv_v2, qr_scale_pad_store_inline813__rv_v2 = pl.yield_(qr_i8_matmul_inline811__rv_v4, qr_scale_pad_store_inline813__ssa_v3)
        q_seq_dep_inline819__ssa_v0: pl.Scalar[pl.TASK_ID] = pl.system.task_dummy(deps=[])
        t_dim_inline231_inline815__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(x__ssa_v0, 0)
        for tile_base_inline229_inline820__idx_v0 in pl.range(0, t_dim_inline231_inline815__ssa_v0, 512):
            tile_rows_inline228_inline821__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(t_dim_inline231_inline815__ssa_v0 - tile_base_inline229_inline820__idx_v0, 512)
            with pl.scope():
                qproj_t_matmul_inline227_inline822__ssa_v0: pl.Scalar[pl.INDEX] = (tile_rows_inline228_inline821__ssa_v0 + 15) // 16 * 16
                q_proj_i32_inline230_inline824__ssa_v0: pl.Tensor[[qproj_t_matmul_inline227_inline822__ssa_v0, 32768], pl.INT32, pl.MemRef("mem_ddr_25", pl.const(0, pl.INT64), 0)] = pl.tensor.create(
                    [qproj_t_matmul_inline227_inline822__ssa_v0, 32768], dtype=pl.INT32, layout=pl.TensorLayout.ND
                )
                qproj_t_matmul_inline1065__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(q_proj_i32_inline230_inline824__ssa_v0, 0)
                qproj_full_rows_inline1076__ssa_v0: pl.Scalar[pl.INDEX] = tile_rows_inline228_inline821__ssa_v0 // 64 * 64
                ret__tmp_v0_3: pl.Tuple[pl.Tensor[[qproj_t_matmul_inline227_inline822__ssa_v0, 32768], pl.INT32], pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(
                    self.qproj_matmul_spmd,
                    q_proj_i32_inline230_inline824__ssa_v0,
                    qproj_full_rows_inline1076__ssa_v0,
                    qr_i8_matmul_inline811__rv_v2,
                    wq_b__ssa_v0,
                    qproj_t_matmul_inline1065__ssa_v0,
                    tile_rows_inline228_inline821__ssa_v0,
                    deps=[q_seq_dep_inline819__ssa_v0],
                    core_num=24,
                    attrs={"arg_directions": [pl.adir.inout, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.scalar, pl.adir.scalar]},
                )
                q_proj_i32_inline230_inline824__rv_v2: pl.Tensor[[qproj_t_matmul_inline227_inline822__ssa_v0, 32768], pl.INT32, pl.MemRef("mem_ddr_26", pl.const(0, pl.INT64), 0)] = ret__tmp_v0_3[0]
                qproj_tid_inline1069__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_3[1]
                q_proj_i32_inline230_inline824__ssa_v9: pl.Tensor[[qproj_t_matmul_inline227_inline822__ssa_v0, 32768], pl.INT32, pl.MemRef("mem_ddr_27", pl.const(0, pl.INT64), 0)] = (
                    q_proj_i32_inline230_inline824__rv_v2
                )
                t_dim_inline1134__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(query__ssa_v0, 0)
                q_flat_inline1121__ssa_v0: pl.Tensor[[t_dim_inline1134__ssa_v0, 32768], pl.BF16, pl.MemRef("mem_ddr_9", pl.const(0, pl.INT64), 0)] = pl.tensor.reshape(
                    query__ssa_v0, [t_dim_inline1134__ssa_v0, 32768]
                )
                ret__tmp_v0_4: pl.Tuple[pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(
                    self.qproj_dequant_rms_nope_rope_spmd,
                    q_flat_inline1121__ssa_v0,
                    tile_rows_inline228_inline821__ssa_v0,
                    tile_base_inline229_inline820__idx_v0,
                    qr_scale_pad_store_inline813__rv_v2,
                    q_rope_cos_il_inline361__ssa_v0,
                    q_rope_sin_signed_inline360__ssa_v0,
                    q_rope_swap_idx_inline359__ssa_v0,
                    q_proj_i32_inline230_inline824__ssa_v9,
                    wq_b_scale__ssa_v0,
                    core_num=48,
                    allow_early_resolve=True,
                    attrs={"arg_directions": [pl.adir.inout, pl.adir.scalar, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input]},
                )
                tid__ssa_v3: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_4[0]
        t_dim_inline868__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(x__ssa_v0, 0)
        for tile_base_inline876__idx_v0 in pl.range(0, t_dim_inline868__ssa_v0, 512):
            tile_rows_inline888__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(t_dim_inline868__ssa_v0 - tile_base_inline876__idx_v0, 512)
            with pl.scope():
                x_view_inline872__ssa_v0: pl.Tensor[[t_dim_inline868__ssa_v0, 4096], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)] = pl.tensor.reshape(
                    x__ssa_v0, [t_dim_inline868__ssa_v0, 4096]
                )
                t_matmul_inline907__ssa_v0: pl.Scalar[pl.INDEX] = (tile_rows_inline888__ssa_v0 + 15) // 16 * 16
                kv_full_rows_inline898__ssa_v0: pl.Scalar[pl.INDEX] = tile_rows_inline888__ssa_v0 // 64 * 64
                kv_m_groups_inline875__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(pl.max(tile_rows_inline888__ssa_v0 // 128, 1), 3)
                kv_fp32_inline882__ssa_v0: pl.Tensor[[t_matmul_inline907__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_28", pl.const(0, pl.INT64), 0)] = pl.tensor.create(
                    [t_matmul_inline907__ssa_v0, 512], dtype=pl.FP32, layout=pl.TensorLayout.ND
                )
                kv_fp32_inline882__rv_v2: pl.Tensor[[t_matmul_inline907__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_29", pl.const(0, pl.INT64), 0)] = self.kv_proj_seed(
                    kv_fp32_inline882__ssa_v0, t_matmul_inline907__ssa_v0, attrs={"arg_directions": [pl.adir.inout, pl.adir.scalar]}
                )
                ret__tmp_v0_5: pl.Tuple[pl.Tensor[[t_matmul_inline907__ssa_v0, 512], pl.FP32], pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(
                    self.kv_proj_matmul_spmd,
                    kv_m_groups_inline875__ssa_v0,
                    kv_fp32_inline882__rv_v2,
                    kv_full_rows_inline898__ssa_v0,
                    tile_base_inline876__idx_v0,
                    x_view_inline872__ssa_v0,
                    wkv__ssa_v0,
                    t_matmul_inline907__ssa_v0,
                    tile_rows_inline888__ssa_v0,
                    deps=[gate__ssa_v0],
                    core_num=kv_m_groups_inline875__ssa_v0 * 8,
                    attrs={"arg_directions": [pl.adir.scalar, pl.adir.inout, pl.adir.scalar, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.scalar, pl.adir.scalar]},
                )
                kv_fp32_inline882__rv_v10: pl.Tensor[[t_matmul_inline907__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_30", pl.const(0, pl.INT64), 0)] = ret__tmp_v0_5[0]
                _kv_tid_inline883__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_5[1]
                kv_view_inline900__ssa_v0: pl.Tensor[[t_dim_inline868__ssa_v0, 512], pl.BF16, pl.MemRef("mem_ddr_12", pl.const(0, pl.INT64), 0)] = pl.tensor.reshape(
                    kv__ssa_v0, [t_dim_inline868__ssa_v0, 512]
                )
                kv_token_tiles_inline903__ssa_v0: pl.Scalar[pl.INDEX] = (tile_rows_inline888__ssa_v0 + 31) // 32
                self.kv_rms_norm_rope_spmd(
                    tile_rows_inline888__ssa_v0,
                    tile_base_inline876__idx_v0,
                    kv_fp32_inline882__rv_v10,
                    kv_view_inline900__ssa_v0,
                    gamma_ckv__ssa_v0,
                    q_rope_cos_il_inline361__ssa_v0,
                    q_rope_sin_signed_inline360__ssa_v0,
                    q_rope_swap_idx_inline359__ssa_v0,
                    attrs={
                        "arg_directions": [pl.adir.scalar, pl.adir.scalar, pl.adir.input, pl.adir.inout, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input],
                        "core_num": kv_token_tiles_inline903__ssa_v0,
                        "sync_start": True,
                    },
                )
        return query__ssa_v0, qr__ssa_v0, qr_scale__ssa_v0