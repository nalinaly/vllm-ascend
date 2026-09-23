# pypto.program: _jit_diagnose_indexer_compressor
import pypto.language as pl

B_DYN = pl.dynamic("B_DYN")
COMPRESS_STATE_BLOCK_NUM_DYN = pl.dynamic("COMPRESS_STATE_BLOCK_NUM_DYN")
IDX_CACHE_BLOCK_NUM_DYN = pl.dynamic("IDX_CACHE_BLOCK_NUM_DYN")
INDEXER_PAGE_BYTES_DYN = pl.dynamic("INDEXER_PAGE_BYTES_DYN")
T_DYN = pl.dynamic("T_DYN")


@pl.program
class _jit_diagnose_indexer_compressor:
    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def compress_state_commit(
        compress_state_flat_inline39__ssa_v0: pl.Out[pl.Tensor[[compress_state_rows_inline44__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)]],
        b_dim_inline51__ssa_v0: pl.Scalar[pl.INDEX],
        commit_workers_inline81__ssa_v0: pl.Scalar[pl.INDEX],
        s_dim_inline54__ssa_v0: pl.Scalar[pl.INDEX],
        state_slots__ssa_v0: pl.Tensor[[T_DYN], pl.INT64, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)],
        positions__ssa_v0: pl.Tensor[[T_DYN], pl.INT32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)],
        values__rv_v2: pl.Tensor[[384, 256], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 393216)],
        scores__rv_v2: pl.Tensor[[384, 256], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 393216)],
        ape__ssa_v0: pl.Tensor[[4, 256], pl.FP32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 4096)],
    ):
        pl.func_attr({"mx_tensor_views_blocked": True})
        mem_vec_6: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 1024)
        mem_vec_8: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 1024)
        commit_worker_inline84__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        for c_idx_inline88__idx_v0, (compress_state_flat_inline39__iter_v1,) in pl.range(
            commit_worker_inline84__ssa_v0, b_dim_inline51__ssa_v0, commit_workers_inline81__ssa_v0, init_values=(compress_state_flat_inline39__ssa_v0,)
        ):
            for s_idx_inline82__idx_v0, (compress_state_flat_inline39__iter_v3,) in pl.range(s_dim_inline54__ssa_v0, init_values=(compress_state_flat_inline39__iter_v1,)):
                token_inline69__ssa_v1: pl.Scalar[pl.INDEX] = c_idx_inline88__idx_v0 * s_dim_inline54__ssa_v0 + s_idx_inline82__idx_v0
                state_row_i64_inline75__tile: pl.Scalar[pl.INT64] = pl.tensor.read(state_slots__ssa_v0, [token_inline69__ssa_v1])
                if 0 <= state_row_i64_inline75__tile:
                    state_row_inline87__ssa_v1: pl.Scalar[pl.INDEX] = pl.cast(state_row_i64_inline75__tile, pl.INDEX)
                    token_pos_inline45__tile: pl.Scalar[pl.INT32] = pl.tensor.read(positions__ssa_v0, [token_inline69__ssa_v1])
                    ape_row_inline53__ssa_v1: pl.Scalar[pl.INDEX] = pl.cast(pl.cast(token_pos_inline45__tile, pl.INDEX) % 4, pl.INDEX)
                    t__tile: pl.Tile[[1, 256], pl.FP32, pl.MemRef(mem_vec_6, pl.const(0, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.load(
                        values__rv_v2, [token_inline69__ssa_v1, 0], [1, 256], [1, 256], target_memory=pl.Mem.Vec
                    )
                    compress_state_flat_inline39__tile: pl.Tensor[[compress_state_rows_inline44__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                        t__tile, [state_row_inline87__ssa_v1, 0], compress_state_flat_inline39__iter_v3
                    )
                    t__tile_1: pl.Tile[[1, 256], pl.FP32, pl.MemRef(mem_vec_6, pl.const(0, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.load(
                        scores__rv_v2, [token_inline69__ssa_v1, 0], [1, 256], [1, 256], target_memory=pl.Mem.Vec
                    )
                    t__tile_2: pl.Tile[[1, 256], pl.FP32, pl.MemRef(mem_vec_8, pl.const(1024, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.load(
                        ape__ssa_v0, [ape_row_inline53__ssa_v1, 0], [1, 256], [1, 256], target_memory=pl.Mem.Vec
                    )
                    t__tile_3: pl.Tile[[1, 256], pl.FP32, pl.MemRef(mem_vec_6, pl.const(0, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.add(t__tile_1, t__tile_2)
                    compress_state_flat_inline39__tile_1: pl.Tensor[[compress_state_rows_inline44__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                        t__tile_3, [state_row_inline87__ssa_v1, 256], compress_state_flat_inline39__tile
                    )
                    compress_state_flat_inline39__phi_v7: pl.Tensor[[compress_state_rows_inline44__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_10", pl.const(0, pl.INT64), 0)] = pl.yield_(
                        compress_state_flat_inline39__tile_1
                    )
                else:
                    compress_state_flat_inline39__phi_v7: pl.Tensor[[compress_state_rows_inline44__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_10", pl.const(0, pl.INT64), 0)] = pl.yield_(
                        compress_state_flat_inline39__iter_v3
                    )
                compress_state_flat_inline39__rv_v4: pl.Tensor[[compress_state_rows_inline44__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_11", pl.const(0, pl.INT64), 0)] = pl.yield_(
                    compress_state_flat_inline39__phi_v7
                )
            compress_state_flat_inline39__rv_v2: pl.Tensor[[compress_state_rows_inline44__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_12", pl.const(0, pl.INT64), 0)] = pl.yield_(
                compress_state_flat_inline39__rv_v4
            )
        return

    @pl.function(type=pl.FunctionType.Spmd)
    def compress_state_commit_spmd(
        self,
        compress_state_flat_inline39__ssa_v0: pl.Out[pl.Tensor[[compress_state_rows_inline44__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)]],
        b_dim_inline51__ssa_v0: pl.Scalar[pl.INDEX],
        commit_workers_inline81__ssa_v0: pl.Scalar[pl.INDEX],
        s_dim_inline54__ssa_v0: pl.Scalar[pl.INDEX],
        state_slots__ssa_v0: pl.Tensor[[T_DYN], pl.INT64, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)],
        positions__ssa_v0: pl.Tensor[[T_DYN], pl.INT32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)],
        values__rv_v2: pl.Tensor[[384, 256], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 393216)],
        scores__rv_v2: pl.Tensor[[384, 256], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 393216)],
        ape__ssa_v0: pl.Tensor[[4, 256], pl.FP32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 4096)],
    ):
        pl.func_attr({"mx_tensor_views_blocked": True})
        self.compress_state_commit(
            compress_state_flat_inline39__ssa_v0,
            b_dim_inline51__ssa_v0,
            commit_workers_inline81__ssa_v0,
            s_dim_inline54__ssa_v0,
            state_slots__ssa_v0,
            positions__ssa_v0,
            values__rv_v2,
            scores__rv_v2,
            ape__ssa_v0,
            attrs={"arg_directions": [pl.adir.output_existing, pl.adir.scalar, pl.adir.scalar, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input]},
        )

    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def compressor_diagnostic_seed(rotated__ssa_v0: pl.Out[pl.Tensor[[T_DYN, 128], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)]]) -> pl.Tensor[[T_DYN, 128], pl.FP32]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        mem_vec_1: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 512)
        t__tile: pl.Tile[[1, 128], pl.FP32, pl.MemRef(mem_vec_1, pl.const(0, pl.INT64), 512), pl.Mem.Vec] = pl.tile.full([1, 128], dtype=pl.FP32, value=0.0)
        rotated__tile: pl.Tensor[[T_DYN, 128], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)] = pl.tile.store(t__tile, [0, 0], rotated__ssa_v0)
        return rotated__ssa_v0

    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def idx_kv_scale_commit(
        compact_rows_inline135__ssa_v0: pl.Scalar[pl.INDEX],
        positions__ssa_v0: pl.Tensor[[T_DYN], pl.INT32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)],
        slots__ssa_v0: pl.Tensor[[T_DYN], pl.INT64, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)],
        cache__rv_v2: pl.InOut[pl.Tensor[[IDX_CACHE_BLOCK_NUM_DYN, INDEXER_PAGE_BYTES_DYN], pl.INT8, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)]],
        idx_kv_scale_values_inline125__ssa_v1: pl.Tensor[[384, 1], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 1536)],
    ) -> pl.Tensor[[IDX_CACHE_BLOCK_NUM_DYN, INDEXER_PAGE_BYTES_DYN], pl.INT8]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        mem_vec_4: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 64)
        for compact_token_inline104__idx_v0 in pl.range(compact_rows_inline135__ssa_v0):
            request_inline107__ssa_v1: pl.Scalar[pl.INDEX] = compact_token_inline104__idx_v0 // 2
            first_pos_inline115__tile: pl.Scalar[pl.INT32] = pl.tensor.read(positions__ssa_v0, [request_inline107__ssa_v1 * 6])
            local_token_inline106__ssa_v1: pl.Scalar[pl.INDEX] = compact_token_inline104__idx_v0 % 2 * 4 - pl.cast(first_pos_inline115__tile, pl.INDEX) % 4 + 3
            if local_token_inline106__ssa_v1 < 6:
                token_v1_inline103__ssa_v0: pl.Scalar[pl.INDEX] = request_inline107__ssa_v1 * 6 + local_token_inline106__ssa_v1
                cache_row_i64_v1_inline141__tile: pl.Scalar[pl.INT64] = pl.tensor.read(slots__ssa_v0, [token_v1_inline103__ssa_v0])
                if 0 <= cache_row_i64_v1_inline141__tile:
                    cache_row_v1_inline118__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(cache_row_i64_v1_inline141__tile, pl.INDEX)
                    scale_page_inline133__ssa_v0: pl.Scalar[pl.INDEX] = cache_row_v1_inline118__ssa_v0 // 32
                    scale_bytes_inline102__ssa_v0: pl.Tile[[1, 64], pl.INT8, pl.MemRef(mem_vec_4, pl.const(0, pl.INT64), 64), pl.Mem.Vec] = pl.tile.load(
                        cache__rv_v2, [scale_page_inline133__ssa_v0, 4096], [1, 64], [1, 64], target_memory=pl.Mem.Vec
                    )
                    scale_half_inline109__ssa_v0: pl.Tile[[1, 32], pl.FP16, pl.MemRef(mem_vec_4, pl.const(0, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reinterpret_view(
                        scale_bytes_inline102__ssa_v0, dtype=pl.FP16
                    )
                    t__tile: pl.Scalar[pl.FP32] = pl.tensor.read(idx_kv_scale_values_inline125__ssa_v1, [compact_token_inline104__idx_v0, 0])
                    pl.tile.write(scale_half_inline109__ssa_v0, [0, cache_row_v1_inline118__ssa_v0 % 32], pl.cast(t__tile, pl.FP16))
                    updated_bytes_inline101__ssa_v0: pl.Tile[[1, 64], pl.INT8, pl.MemRef(mem_vec_4, pl.const(0, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reinterpret_view(
                        scale_half_inline109__ssa_v0, dtype=pl.INT8
                    )
                    cache__store: pl.Tensor[[IDX_CACHE_BLOCK_NUM_DYN, INDEXER_PAGE_BYTES_DYN], pl.INT8, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                        updated_bytes_inline101__ssa_v0, [scale_page_inline133__ssa_v0, 4096], cache__rv_v2
                    )
        return cache__rv_v2

    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def indexer_boundary_init(normalized__ssa_v0: pl.Out[pl.Tensor[[384, 128], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 98304)]]) -> pl.Tensor[[384, 128], pl.BF16]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        mem_vec_1: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 512)
        init_request_inline91__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        compact_begin_inline85__ssa_v0: pl.Scalar[pl.INDEX] = init_request_inline91__ssa_v0 * 2
        t__tile: pl.Tile[[2, 128], pl.BF16, pl.MemRef(mem_vec_1, pl.const(0, pl.INT64), 512), pl.Mem.Vec] = pl.tile.full([2, 128], dtype=pl.BF16, value=0.0)
        normalized__tile: pl.Tensor[[384, 128], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 98304)] = pl.tile.store(t__tile, [compact_begin_inline85__ssa_v0, 0], normalized__ssa_v0)
        return normalized__ssa_v0

    @pl.function(type=pl.FunctionType.Spmd)
    def indexer_boundary_init_spmd(self, normalized__ssa_v0: pl.Out[pl.Tensor[[384, 128], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 98304)]]) -> pl.Tensor[[384, 128], pl.BF16]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        normalized__ssa_v1: pl.Tensor[[384, 128], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 98304)] = self.indexer_boundary_init(
            normalized__ssa_v0, attrs={"arg_directions": [pl.adir.output_existing]}
        )
        return normalized__ssa_v0

    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def kv_and_cache_write(
        compact_rows_inline135__ssa_v0: pl.Scalar[pl.INDEX],
        kv_final_inline120__rv_v2: pl.Tensor[[384, 128], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 196608)],
        idx_kv_scale_values_inline125__ssa_v0: pl.Out[pl.Tensor[[384, 1], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 1536)]],
        cache__ssa_v0: pl.Out[pl.Tensor[[IDX_CACHE_BLOCK_NUM_DYN, INDEXER_PAGE_BYTES_DYN], pl.INT8, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)]],
        kv_flat_inline128__ssa_v0: pl.Out[pl.Tensor[[T_DYN, 128], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)]],
        positions__ssa_v0: pl.Tensor[[T_DYN], pl.INT32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 0)],
        slots__ssa_v0: pl.Tensor[[T_DYN], pl.INT64, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 0)],
    ) -> tuple[pl.Tensor[[384, 1], pl.FP32], pl.Tensor[[IDX_CACHE_BLOCK_NUM_DYN, INDEXER_PAGE_BYTES_DYN], pl.INT8]]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        mem_vec_6: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 8192)
        mem_vec_12: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 8192)
        mem_vec_13: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 8192)
        mem_vec_14: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 64)
        wr_blk_inline119__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        wr_b0_inline140__ssa_v0: pl.Scalar[pl.INDEX] = wr_blk_inline119__ssa_v0 * 16
        wr_blk_rows_inline136__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(compact_rows_inline135__ssa_v0 - wr_b0_inline140__ssa_v0, 16)
        t__tile: pl.Tile[[16, 128], pl.FP32, pl.MemRef(mem_vec_6, pl.const(16448, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.load(
            kv_final_inline120__rv_v2, [wr_b0_inline140__ssa_v0, 0], [16, 128], [16, 128], target_memory=pl.Mem.Vec
        )
        t__tile_1: pl.Tile[[16, 128], pl.BF16, pl.MemRef(mem_vec_12, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(t__tile, target_type=pl.BF16, mode="rint")
        kv_blk_f32_inline142__tile: pl.Tile[[16, 128], pl.FP32, pl.MemRef(mem_vec_6, pl.const(16448, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.cast(t__tile_1, target_type=pl.FP32, mode="round")
        t__tile_2: pl.Tile[[16, 128], pl.FP32, pl.MemRef(mem_vec_6, pl.const(16448, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.muls(kv_blk_f32_inline142__tile, 0.088388347648318447)
        t__tile_3: pl.Tile[[16, 128], pl.BF16, pl.MemRef(mem_vec_12, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(t__tile_2, target_type=pl.BF16, mode="rint")
        kv_blk_f32_v1_inline114__tile: pl.Tile[[16, 128], pl.FP32, pl.MemRef(mem_vec_6, pl.const(16448, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.cast(t__tile_3, target_type=pl.FP32, mode="round")
        t__tile_4: pl.Tile[[16, 128], pl.FP32, pl.MemRef(mem_vec_12, pl.const(0, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.abs(kv_blk_f32_v1_inline114__tile)
        tmp_tile: pl.Tile[[16, 128], pl.FP32, pl.MemRef(mem_vec_13, pl.const(8192, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.create([16, 128], dtype=pl.FP32, target_memory=pl.Mem.Vec)
        t__tile_5: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_14, pl.const(16384, pl.INT64), 64), pl.Mem.Vec] = pl.tile.row_max(t__tile_4, tmp_tile)
        kv_amax_inline139__tile: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_14, pl.const(16384, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(t__tile_5, [1, 16])
        t__tile_6: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_12, pl.const(0, pl.INT64), 64), pl.Mem.Vec] = pl.tile.full([1, 16], dtype=pl.FP32, value=0.0001)
        kv_amax_v1_inline145__tile: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_12, pl.const(0, pl.INT64), 64), pl.Mem.Vec] = pl.tile.maximum(kv_amax_inline139__tile, t__tile_6)
        t__tile_7: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_13, pl.const(8192, pl.INT64), 64), pl.Mem.Vec] = pl.tile.full([1, 16], dtype=pl.FP32, value=127.0)
        kv_scale_q_row_inline143__tile: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_13, pl.const(8192, pl.INT64), 64), pl.Mem.Vec] = pl.tile.div(t__tile_7, kv_amax_v1_inline145__tile)
        t__tile_8: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_12, pl.const(0, pl.INT64), 64), pl.Mem.Vec] = pl.tile.recip(kv_scale_q_row_inline143__tile)
        kv_scale_dq_col_inline129__tile: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_12, pl.const(0, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(t__tile_8, [16, 1])
        kv_scale_q_col_inline117__tile: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_13, pl.const(8192, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(kv_scale_q_row_inline143__tile, [16, 1])
        idx_kv_scale_values_inline125__tile: pl.Tensor[[384, 1], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 1536)] = pl.tile.store(
            kv_scale_dq_col_inline129__tile, [wr_b0_inline140__ssa_v0, 0], idx_kv_scale_values_inline125__ssa_v0
        )
        kv_scaled_inline122__tile: pl.Tile[[16, 128], pl.FP32, pl.MemRef(mem_vec_12, pl.const(0, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.row_expand_mul(
            kv_blk_f32_v1_inline114__tile, kv_scale_q_col_inline117__tile
        )
        kv_i32_inline146__tile: pl.Tile[[16, 128], pl.INT32, pl.MemRef(mem_vec_12, pl.const(0, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.cast(
            kv_scaled_inline122__tile, target_type=pl.INT32, mode="rint"
        )
        kv_half_inline134__tile: pl.Tile[[16, 128], pl.FP16, pl.MemRef(mem_vec_12, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(kv_i32_inline146__tile, target_type=pl.FP16, mode="round")
        kv_i8_blk_inline148__tile: pl.Tile[[16, 128], pl.INT8, pl.MemRef(mem_vec_12, pl.const(0, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(
            kv_half_inline134__tile, target_type=pl.INT8, mode="trunc"
        )
        for inner_inline111__idx_v0, (cache__iter_v1, kv_flat_inline128__iter_v1) in pl.range(wr_blk_rows_inline136__ssa_v0, init_values=(cache__ssa_v0, kv_flat_inline128__ssa_v0)):
            compact_token_inline108__ssa_v0: pl.Scalar[pl.INDEX] = wr_b0_inline140__ssa_v0 + inner_inline111__idx_v0
            request_inline107__ssa_v0: pl.Scalar[pl.INDEX] = compact_token_inline108__ssa_v0 // 2
            first_pos_inline115__tile: pl.Scalar[pl.INT32] = pl.tensor.read(positions__ssa_v0, [request_inline107__ssa_v0 * 6])
            local_token_inline106__ssa_v0: pl.Scalar[pl.INDEX] = compact_token_inline108__ssa_v0 % 2 * 4 - pl.cast(first_pos_inline115__tile, pl.INDEX) % 4 + 3
            if local_token_inline106__ssa_v0 < 6:
                token_inline110__ssa_v0: pl.Scalar[pl.INDEX] = request_inline107__ssa_v0 * 6 + local_token_inline106__ssa_v0
                cache_row_i64_inline144__tile: pl.Scalar[pl.INT64] = pl.tensor.read(slots__ssa_v0, [token_inline110__ssa_v0])
                if 0 <= cache_row_i64_inline144__tile:
                    cache_row_inline123__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(cache_row_i64_inline144__tile, pl.INDEX)
                    t__tile_9: pl.Tile[[1, 128], pl.FP32, pl.MemRef(mem_vec_6, pl.const(16448, pl.INT64), 512), pl.Mem.Vec] = pl.tile.slice(
                        kv_blk_f32_v1_inline114__tile, [1, 128], [inner_inline111__idx_v0, 0]
                    )
                    kv_flat_inline128__tile: pl.Tensor[[T_DYN, 128], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                        t__tile_9, [token_inline110__ssa_v0, 0], kv_flat_inline128__iter_v1
                    )
                    cache_page_inline105__ssa_v0: pl.Scalar[pl.INDEX] = cache_row_inline123__ssa_v0 // 32
                    key_begin_inline137__ssa_v0: pl.Scalar[pl.INDEX] = cache_row_inline123__ssa_v0 % 32 * 128
                    t__tile_10: pl.Tile[[1, 128], pl.INT8, pl.MemRef(mem_vec_12, pl.const(0, pl.INT64), 128), pl.Mem.Vec] = pl.tile.slice(
                        kv_i8_blk_inline148__tile, [1, 128], [inner_inline111__idx_v0, 0]
                    )
                    cache__tile: pl.Tensor[[IDX_CACHE_BLOCK_NUM_DYN, INDEXER_PAGE_BYTES_DYN], pl.INT8, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                        t__tile_10, [cache_page_inline105__ssa_v0, key_begin_inline137__ssa_v0], cache__iter_v1
                    )
                    cache__phi_v4, kv_flat_inline128__phi_v4 = pl.yield_(cache__tile, kv_flat_inline128__tile)
                else:
                    cache__phi_v4, kv_flat_inline128__phi_v4 = pl.yield_(cache__iter_v1, kv_flat_inline128__iter_v1)
                cache__phi_v5, kv_flat_inline128__phi_v5 = pl.yield_(cache__phi_v4, kv_flat_inline128__phi_v4)
            else:
                cache__phi_v5, kv_flat_inline128__phi_v5 = pl.yield_(cache__iter_v1, kv_flat_inline128__iter_v1)
            cache__rv_v2, kv_flat_inline128__rv_v2 = pl.yield_(cache__phi_v5, kv_flat_inline128__phi_v5)
        return idx_kv_scale_values_inline125__ssa_v0, cache__ssa_v0

    @pl.function(type=pl.FunctionType.Spmd)
    def kv_and_cache_write_spmd(
        self,
        compact_rows_inline135__ssa_v0: pl.Scalar[pl.INDEX],
        kv_final_inline120__rv_v2: pl.Tensor[[384, 128], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 196608)],
        idx_kv_scale_values_inline125__ssa_v0: pl.Out[pl.Tensor[[384, 1], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 1536)]],
        cache__ssa_v0: pl.Out[pl.Tensor[[IDX_CACHE_BLOCK_NUM_DYN, INDEXER_PAGE_BYTES_DYN], pl.INT8, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)]],
        kv_flat_inline128__ssa_v0: pl.Out[pl.Tensor[[T_DYN, 128], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)]],
        positions__ssa_v0: pl.Tensor[[T_DYN], pl.INT32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 0)],
        slots__ssa_v0: pl.Tensor[[T_DYN], pl.INT64, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 0)],
    ) -> tuple[pl.Tensor[[384, 1], pl.FP32], pl.Tensor[[IDX_CACHE_BLOCK_NUM_DYN, INDEXER_PAGE_BYTES_DYN], pl.INT8]]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        ret__tmp_v0: pl.Tuple[pl.Tensor[[384, 1], pl.FP32], pl.Tensor[[IDX_CACHE_BLOCK_NUM_DYN, INDEXER_PAGE_BYTES_DYN], pl.INT8]] = self.kv_and_cache_write(
            compact_rows_inline135__ssa_v0,
            kv_final_inline120__rv_v2,
            idx_kv_scale_values_inline125__ssa_v0,
            cache__ssa_v0,
            kv_flat_inline128__ssa_v0,
            positions__ssa_v0,
            slots__ssa_v0,
            attrs={"arg_directions": [pl.adir.scalar, pl.adir.input, pl.adir.output_existing, pl.adir.output_existing, pl.adir.output_existing, pl.adir.input, pl.adir.input]},
        )
        idx_kv_scale_values_inline125__ssa_v1: pl.Tensor[[384, 1], pl.FP32, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 1536)] = ret__tmp_v0[0]
        cache__rv_v2: pl.Tensor[[IDX_CACHE_BLOCK_NUM_DYN, INDEXER_PAGE_BYTES_DYN], pl.INT8, pl.MemRef("mem_ddr_7", pl.const(0, pl.INT64), 0)] = ret__tmp_v0[1]
        return idx_kv_scale_values_inline125__ssa_v0, cache__ssa_v0

    @pl.function(type=pl.FunctionType.AIC, level=pl.Level.AIC, role=pl.Role.SubWorker)
    def kv_hadamard(
        kv_final_inline120__ssa_v0: pl.Out[pl.Tensor[[384, 128], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 196608)]],
        hadamard__ssa_v0: pl.Tensor[[128, 128], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 32768)],
        rms_blocks_inline130__ssa_v0: pl.Scalar[pl.INDEX],
        compact_rows_inline135__ssa_v0: pl.Scalar[pl.INDEX],
        normalized__rv_v3: pl.Tensor[[384, 128], pl.BF16, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 98304)],
    ) -> pl.Tensor[[384, 128], pl.FP32]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        mem_mat_3: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 16384)
        mem_mat_4: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 4096)
        mem_left_5: pl.Ptr = pl.tile.alloc(pl.Mem.Left, 4096)
        mem_right_6: pl.Ptr = pl.tile.alloc(pl.Mem.Right, 16384)
        mem_acc_7: pl.Ptr = pl.tile.alloc(pl.Mem.Acc, 4096)
        for o0_inline121__idx_v0, (kv_final_inline120__iter_v1,) in pl.range(0, 128, 64, init_values=(kv_final_inline120__ssa_v0,)):
            hadamard_tile_inline126__tile: pl.Tile[[128, 64], pl.BF16, pl.MemRef(mem_mat_3, pl.const(0, pl.INT64), 16384), pl.Mem.Mat] = pl.tile.load(
                hadamard__ssa_v0, [0, o0_inline121__idx_v0], [128, 64], [128, 64], target_memory=pl.Mem.Mat
            )
            for had_blk_inline124__idx_v0, (kv_final_inline120__iter_v3,) in pl.range(rms_blocks_inline130__ssa_v0, init_values=(kv_final_inline120__iter_v1,)):
                had_b0_inline147__ssa_v0: pl.Scalar[pl.INDEX] = had_blk_inline124__idx_v0 * 16
                had_rows_inline127__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(compact_rows_inline135__ssa_v0 - had_b0_inline147__ssa_v0, 16)
                kv_proj_tile_inline116__tile: pl.Tile[
                    [16, 128], pl.BF16, pl.MemRef(mem_mat_4, pl.const(16384, pl.INT64), 4096), pl.Mem.Mat, pl.TileView(valid_shape=[had_rows_inline127__ssa_v0, 128])
                ] = pl.tile.load(normalized__rv_v3, [had_b0_inline147__ssa_v0, 0], [16, 128], [had_rows_inline127__ssa_v0, 128], target_memory=pl.Mem.Mat)
                kv_proj_tile_inline116__tile_Left: pl.Tile[
                    [16, 128], pl.BF16, pl.MemRef(mem_left_5, pl.const(0, pl.INT64), 4096), pl.Mem.Left, pl.TileView(valid_shape=[had_rows_inline127__ssa_v0, 128])
                ] = pl.tile.move(kv_proj_tile_inline116__tile, target_memory=pl.Mem.Left)
                hadamard_tile_inline126__tile_Right: pl.Tile[[128, 64], pl.BF16, pl.MemRef(mem_right_6, pl.const(0, pl.INT64), 16384), pl.Mem.Right] = pl.tile.move(
                    hadamard_tile_inline126__tile, target_memory=pl.Mem.Right
                )
                kv_hadamard_acc_inline112__tile: pl.Tile[
                    [16, 64], pl.FP32, pl.MemRef(mem_acc_7, pl.const(0, pl.INT64), 4096), pl.Mem.Acc, pl.TileView(valid_shape=[had_rows_inline127__ssa_v0, 64], compact=pl.CompactMode.normal)
                ] = pl.tile.matmul(kv_proj_tile_inline116__tile_Left, hadamard_tile_inline126__tile_Right)
                kv_final_inline120__tile: pl.Tensor[[384, 128], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 196608)] = pl.tile.store(
                    kv_hadamard_acc_inline112__tile, [had_b0_inline147__ssa_v0, o0_inline121__idx_v0], kv_final_inline120__iter_v3
                )
                kv_final_inline120__rv_v4: pl.Tensor[[384, 128], pl.FP32, pl.MemRef("mem_ddr_8", pl.const(0, pl.INT64), 196608)] = pl.yield_(kv_final_inline120__tile)
            kv_final_inline120__rv_v2: pl.Tensor[[384, 128], pl.FP32, pl.MemRef("mem_ddr_9", pl.const(0, pl.INT64), 196608)] = pl.yield_(kv_final_inline120__rv_v4)
        return kv_final_inline120__ssa_v0

    @pl.function(type=pl.FunctionType.AIC, level=pl.Level.AIC, role=pl.Role.SubWorker)
    def kv_score_proj(
        scores__ssa_v0: pl.Out[pl.Tensor[[384, 256], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 393216)]],
        values__ssa_v0: pl.Out[pl.Tensor[[384, 256], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 393216)]],
        t_matmul_inline2__ssa_v0: pl.Scalar[pl.INDEX],
        bs_inline5__ssa_v0: pl.Scalar[pl.INDEX],
        x_flat_inline3__ssa_v0: pl.Tensor[[T_DYN, 4096], pl.BF16, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)],
        wkv__ssa_v0: pl.Tensor[[256, 4096], pl.BF16, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 2097152)],
        wgate__ssa_v0: pl.Tensor[[256, 4096], pl.BF16, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 2097152)],
    ) -> tuple[pl.Tensor[[384, 256], pl.FP32], pl.Tensor[[384, 256], pl.FP32]]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        mem_acc_5: pl.Ptr = pl.tile.alloc(pl.Mem.Acc, 2048)
        mem_acc_6: pl.Ptr = pl.tile.alloc(pl.Mem.Acc, 2048)
        mem_mat_7: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 16384)
        mem_mat_8: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 32768)
        mem_mat_9: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 32768)
        mem_mat_10: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 16384)
        mem_mat_11: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 32768)
        mem_mat_12: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 32768)
        mem_left_13: pl.Ptr = pl.tile.alloc(pl.Mem.Left, 16384)
        mem_right_14: pl.Ptr = pl.tile.alloc(pl.Mem.Right, 32768)
        mem_left_16: pl.Ptr = pl.tile.alloc(pl.Mem.Left, 16384)
        mem_right_17: pl.Ptr = pl.tile.alloc(pl.Mem.Right, 32768)
        kv_worker_inline14__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        for idx_inline4__idx_v0, (scores__iter_v1, values__iter_v1) in pl.range(kv_worker_inline14__ssa_v0, t_matmul_inline2__ssa_v0 // 2, 24, init_values=(scores__ssa_v0, values__ssa_v0)):
            global_row0_inline9__ssa_v0: pl.Scalar[pl.INDEX] = idx_inline4__idx_v0 // 8 * 16
            o0_inline11__ssa_v0: pl.Scalar[pl.INDEX] = idx_inline4__idx_v0 % 8 * 32
            kv_acc_inline13__tile: pl.Tile[[16, 32], pl.FP32, pl.MemRef(mem_acc_5, pl.const(0, pl.INT64), 2048), pl.Mem.Acc] = pl.tile.create([16, 32], dtype=pl.FP32, target_memory=pl.Mem.Acc)
            score_acc_inline7__tile: pl.Tile[[16, 32], pl.FP32, pl.MemRef(mem_acc_6, pl.const(2048, pl.INT64), 2048), pl.Mem.Acc] = pl.tile.create([16, 32], dtype=pl.FP32, target_memory=pl.Mem.Acc)
            for kb_inline8__idx_v0, (kv_acc_inline13__iter_v1, score_acc_inline7__iter_v1) in pl.range(0, 8, 2, init_values=(kv_acc_inline13__tile, score_acc_inline7__tile)):
                k0_inline15__ssa_v0: pl.Scalar[pl.INDEX] = kb_inline8__idx_v0 * 512
                x_rows_inline12__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(bs_inline5__ssa_v0 - global_row0_inline9__ssa_v0, 16)
                k0_inline15__ssa_v0_1: pl.Scalar[pl.INDEX] = kb_inline8__idx_v0 * 512 + 512
                x_rows_inline12__ssa_v0_1: pl.Scalar[pl.INDEX] = pl.min(bs_inline5__ssa_v0 - global_row0_inline9__ssa_v0, 16)
                x_tile_inline10__tile: pl.Tile[[16, 512], pl.BF16, pl.MemRef(mem_mat_7, pl.const(81920, pl.INT64), 16384), pl.Mem.Mat, pl.TileView(valid_shape=[x_rows_inline12__ssa_v0, 512])] = (
                    pl.tile.load(x_flat_inline3__ssa_v0, [global_row0_inline9__ssa_v0, k0_inline15__ssa_v0], [16, 512], [x_rows_inline12__ssa_v0, 512], target_memory=pl.Mem.Mat)
                )
                wkv_tile_inline1__tile: pl.Tile[[32, 512], pl.BF16, pl.MemRef(mem_mat_8, pl.const(98304, pl.INT64), 32768), pl.Mem.Mat] = pl.tile.load(
                    wkv__ssa_v0, [o0_inline11__ssa_v0, k0_inline15__ssa_v0], [32, 512], [32, 512], target_memory=pl.Mem.Mat
                )
                wgate_tile_inline0__tile: pl.Tile[[32, 512], pl.BF16, pl.MemRef(mem_mat_9, pl.const(131072, pl.INT64), 32768), pl.Mem.Mat] = pl.tile.load(
                    wgate__ssa_v0, [o0_inline11__ssa_v0, k0_inline15__ssa_v0], [32, 512], [32, 512], target_memory=pl.Mem.Mat
                )
                x_tile_inline10__tile_1: pl.Tile[[16, 512], pl.BF16, pl.MemRef(mem_mat_10, pl.const(0, pl.INT64), 16384), pl.Mem.Mat, pl.TileView(valid_shape=[x_rows_inline12__ssa_v0_1, 512])] = (
                    pl.tile.load(x_flat_inline3__ssa_v0, [global_row0_inline9__ssa_v0, k0_inline15__ssa_v0_1], [16, 512], [x_rows_inline12__ssa_v0_1, 512], target_memory=pl.Mem.Mat)
                )
                wkv_tile_inline1__tile_1: pl.Tile[[32, 512], pl.BF16, pl.MemRef(mem_mat_11, pl.const(16384, pl.INT64), 32768), pl.Mem.Mat] = pl.tile.load(
                    wkv__ssa_v0, [o0_inline11__ssa_v0, k0_inline15__ssa_v0_1], [32, 512], [32, 512], target_memory=pl.Mem.Mat
                )
                wgate_tile_inline0__tile_1: pl.Tile[[32, 512], pl.BF16, pl.MemRef(mem_mat_12, pl.const(49152, pl.INT64), 32768), pl.Mem.Mat] = pl.tile.load(
                    wgate__ssa_v0, [o0_inline11__ssa_v0, k0_inline15__ssa_v0_1], [32, 512], [32, 512], target_memory=pl.Mem.Mat
                )
                wkv_tile_inline1__tile_t: pl.Tile[
                    [512, 32], pl.BF16, pl.MemRef(mem_mat_8, pl.const(98304, pl.INT64), 32768), pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)
                ] = pl.tile.transpose_view(wkv_tile_inline1__tile)
                x_tile_inline10__tile_Left: pl.Tile[[16, 512], pl.BF16, pl.MemRef(mem_left_13, pl.const(0, pl.INT64), 16384), pl.Mem.Left, pl.TileView(valid_shape=[x_rows_inline12__ssa_v0, 512])] = (
                    pl.tile.move(x_tile_inline10__tile, target_memory=pl.Mem.Left)
                )
                wkv_tile_inline1__tile_t_Right: pl.Tile[[512, 32], pl.BF16, pl.MemRef(mem_right_14, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.move(
                    wkv_tile_inline1__tile_t, target_memory=pl.Mem.Right
                )
                kv_acc_inline13__tile_1: pl.Tile[[16, 32], pl.FP32, pl.MemRef(mem_acc_5, pl.const(0, pl.INT64), 2048), pl.Mem.Acc] = pl.tile.matmul_acc(
                    kv_acc_inline13__iter_v1, x_tile_inline10__tile_Left, wkv_tile_inline1__tile_t_Right, k0_inline15__ssa_v0 == 0
                )
                wgate_tile_inline0__tile_t: pl.Tile[
                    [512, 32], pl.BF16, pl.MemRef(mem_mat_9, pl.const(131072, pl.INT64), 32768), pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)
                ] = pl.tile.transpose_view(wgate_tile_inline0__tile)
                wgate_tile_inline0__tile_t_Right: pl.Tile[[512, 32], pl.BF16, pl.MemRef(mem_right_14, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.move(
                    wgate_tile_inline0__tile_t, target_memory=pl.Mem.Right
                )
                score_acc_inline7__tile_1: pl.Tile[[16, 32], pl.FP32, pl.MemRef(mem_acc_6, pl.const(2048, pl.INT64), 2048), pl.Mem.Acc] = pl.tile.matmul_acc(
                    score_acc_inline7__iter_v1, x_tile_inline10__tile_Left, wgate_tile_inline0__tile_t_Right, k0_inline15__ssa_v0 == 0
                )
                wkv_tile_inline1__tile_t_1: pl.Tile[
                    [512, 32], pl.BF16, pl.MemRef(mem_mat_11, pl.const(16384, pl.INT64), 32768), pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)
                ] = pl.tile.transpose_view(wkv_tile_inline1__tile_1)
                x_tile_inline10__tile_Left_1: pl.Tile[
                    [16, 512], pl.BF16, pl.MemRef(mem_left_16, pl.const(16384, pl.INT64), 16384), pl.Mem.Left, pl.TileView(valid_shape=[x_rows_inline12__ssa_v0_1, 512])
                ] = pl.tile.move(x_tile_inline10__tile_1, target_memory=pl.Mem.Left)
                wkv_tile_inline1__tile_t_Right_1: pl.Tile[[512, 32], pl.BF16, pl.MemRef(mem_right_17, pl.const(32768, pl.INT64), 32768), pl.Mem.Right] = pl.tile.move(
                    wkv_tile_inline1__tile_t_1, target_memory=pl.Mem.Right
                )
                kv_acc_inline13__tile_2: pl.Tile[[16, 32], pl.FP32, pl.MemRef(mem_acc_5, pl.const(0, pl.INT64), 2048), pl.Mem.Acc] = pl.tile.matmul_acc(
                    kv_acc_inline13__tile_1, x_tile_inline10__tile_Left_1, wkv_tile_inline1__tile_t_Right_1, k0_inline15__ssa_v0_1 == 0
                )
                wgate_tile_inline0__tile_t_1: pl.Tile[
                    [512, 32], pl.BF16, pl.MemRef(mem_mat_12, pl.const(49152, pl.INT64), 32768), pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)
                ] = pl.tile.transpose_view(wgate_tile_inline0__tile_1)
                wgate_tile_inline0__tile_t_Right_1: pl.Tile[[512, 32], pl.BF16, pl.MemRef(mem_right_17, pl.const(32768, pl.INT64), 32768), pl.Mem.Right] = pl.tile.move(
                    wgate_tile_inline0__tile_t_1, target_memory=pl.Mem.Right
                )
                score_acc_inline7__tile_2: pl.Tile[[16, 32], pl.FP32, pl.MemRef(mem_acc_6, pl.const(2048, pl.INT64), 2048), pl.Mem.Acc] = pl.tile.matmul_acc(
                    score_acc_inline7__tile_1, x_tile_inline10__tile_Left_1, wgate_tile_inline0__tile_t_Right_1, k0_inline15__ssa_v0_1 == 0
                )
                kv_acc_inline13__rv_v2, score_acc_inline7__rv_v2 = pl.yield_(kv_acc_inline13__tile_2, score_acc_inline7__tile_2)
            values__tile: pl.Tensor[[384, 256], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 393216)] = pl.tile.store(
                kv_acc_inline13__rv_v2, [global_row0_inline9__ssa_v0, o0_inline11__ssa_v0], values__iter_v1
            )
            scores__tile: pl.Tensor[[384, 256], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 393216)] = pl.tile.store(
                score_acc_inline7__rv_v2, [global_row0_inline9__ssa_v0, o0_inline11__ssa_v0], scores__iter_v1
            )
            scores__rv_v2, values__rv_v2 = pl.yield_(scores__tile, values__tile)
        return scores__ssa_v0, values__ssa_v0

    @pl.function(type=pl.FunctionType.Spmd)
    def kv_score_proj_spmd(
        self,
        scores__ssa_v0: pl.Out[pl.Tensor[[384, 256], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 393216)]],
        values__ssa_v0: pl.Out[pl.Tensor[[384, 256], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 393216)]],
        t_matmul_inline2__ssa_v0: pl.Scalar[pl.INDEX],
        bs_inline5__ssa_v0: pl.Scalar[pl.INDEX],
        x_flat_inline3__ssa_v0: pl.Tensor[[T_DYN, 4096], pl.BF16, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)],
        wkv__ssa_v0: pl.Tensor[[256, 4096], pl.BF16, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 2097152)],
        wgate__ssa_v0: pl.Tensor[[256, 4096], pl.BF16, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 2097152)],
    ) -> tuple[pl.Tensor[[384, 256], pl.FP32], pl.Tensor[[384, 256], pl.FP32]]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        ret__tmp_v0: pl.Tuple[pl.Tensor[[384, 256], pl.FP32], pl.Tensor[[384, 256], pl.FP32]] = self.kv_score_proj(
            scores__ssa_v0,
            values__ssa_v0,
            t_matmul_inline2__ssa_v0,
            bs_inline5__ssa_v0,
            x_flat_inline3__ssa_v0,
            wkv__ssa_v0,
            wgate__ssa_v0,
            attrs={"arg_directions": [pl.adir.output_existing, pl.adir.output_existing, pl.adir.scalar, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.input]},
        )
        scores__rv_v2: pl.Tensor[[384, 256], pl.FP32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 393216)] = ret__tmp_v0[0]
        values__rv_v2: pl.Tensor[[384, 256], pl.FP32, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 393216)] = ret__tmp_v0[1]
        return scores__ssa_v0, values__ssa_v0

    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def rmsnorm_rope(
        normalized__ssa_v1: pl.Out[pl.Tensor[[384, 128], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 98304)]],
        rms_blocks_inline60__ssa_v0: pl.Scalar[pl.INDEX],
        rms_workers_inline93__ssa_v0: pl.Scalar[pl.INDEX],
        bs_inline52__ssa_v0: pl.Scalar[pl.INDEX],
        cos__ssa_v0: pl.Tensor[[T_DYN, 64], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)],
        sin__ssa_v0: pl.Tensor[[T_DYN, 64], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)],
        pooled_kv_inline67__rv_v2: pl.Tensor[[384, 128], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 196608)],
        norm_w_2d_inline68__ssa_v0: pl.Tensor[[1, 128], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 512)],
        positions__ssa_v0: pl.Tensor[[T_DYN], pl.INT32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 0)],
    ) -> pl.Tensor[[384, 128], pl.BF16]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        mem_vec_19: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 4096)
        mem_vec_20: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 4096)
        mem_vec_21: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 4096)
        mem_vec_25: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 4096)
        mem_vec_26: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 8192)
        mem_vec_27: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 64)
        mem_vec_29: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 4096)
        mem_vec_30: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 8192)
        mem_vec_31: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 64)
        mem_vec_49: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        rms_worker_inline92__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        rope_ones_inline98__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_26, pl.const(16384, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.full([16, 64], dtype=pl.FP32, value=1.0)
        rope_index_inline41__ci_tmp_v0: pl.Tile[[1, 192], pl.FP32, pl.MemRef(mem_vec_30, pl.const(28736, pl.INT64), 768), pl.Mem.Vec] = pl.tile.create(
            [1, 192], dtype=pl.FP32, target_memory=pl.Mem.Vec
        )
        rope_index_inline41__tile: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_19, pl.const(0, pl.INT64), 256), pl.Mem.Vec] = pl.tile.ci(
            pl.const(0, pl.INT32), [1, 64], tmp=rope_index_inline41__ci_tmp_v0, dtype=pl.INT32, descending=False
        )
        rope_index_f_inline100__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_30, pl.const(28736, pl.INT64), 256), pl.Mem.Vec] = pl.tile.cast(
            rope_index_inline41__tile, target_type=pl.FP32, mode="round"
        )
        rope_col_inline37__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_26, pl.const(16384, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.col_expand_mul(
            rope_ones_inline98__tile, rope_index_f_inline100__tile
        )
        t__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_30, pl.const(28736, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.muls(rope_col_inline37__tile, 0.5)
        t__tile_1: pl.Tile[[16, 64], pl.INT32, pl.MemRef(mem_vec_30, pl.const(28736, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(t__tile, target_type=pl.INT32, mode="trunc")
        rope_dup_f_inline83__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_30, pl.const(28736, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(t__tile_1, target_type=pl.FP32, mode="round")
        t__tile_2: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_30, pl.const(28736, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.muls(rope_dup_f_inline83__tile, 2.0)
        rope_lane_inline73__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_30, pl.const(28736, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.sub(rope_col_inline37__tile, t__tile_2)
        t__tile_3: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_26, pl.const(16384, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.adds(rope_col_inline37__tile, 1.0)
        t__tile_4: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_30, pl.const(28736, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.muls(rope_lane_inline73__tile, 2.0)
        t__tile_5: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_26, pl.const(16384, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.sub(t__tile_3, t__tile_4)
        rope_swap_idx_inline89__tile: pl.Tile[[16, 64], pl.INT32, pl.MemRef(mem_vec_19, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(t__tile_5, target_type=pl.INT32, mode="round")
        for rms_blk_inline94__idx_v0, (normalized__iter_v2,) in pl.range(rms_worker_inline92__ssa_v0, rms_blocks_inline60__ssa_v0, rms_workers_inline93__ssa_v0, init_values=(normalized__ssa_v1,)):
            b0_inline35__ssa_v0: pl.Scalar[pl.INDEX] = rms_blk_inline94__idx_v0 * 16
            rms_blk_rows_inline34__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(bs_inline52__ssa_v0 - b0_inline35__ssa_v0, 16)
            cos_b_inline33__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_20, pl.const(4096, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(valid_shape=[rms_blk_rows_inline34__ssa_v0, 64])] = (
                pl.tile.load(cos__ssa_v0, [b0_inline35__ssa_v0, 0], [16, 64], [rms_blk_rows_inline34__ssa_v0, 64], target_memory=pl.Mem.Vec)
            )
            sin_b_inline32__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_21, pl.const(8192, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(valid_shape=[rms_blk_rows_inline34__ssa_v0, 64])] = (
                pl.tile.load(sin__ssa_v0, [b0_inline35__ssa_v0, 0], [16, 64], [rms_blk_rows_inline34__ssa_v0, 64], target_memory=pl.Mem.Vec)
            )
            partial_sq_inline50__tile: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_49, pl.const(36992, pl.INT64), 64), pl.Mem.Vec] = pl.tile.full([1, 16], dtype=pl.FP32, value=0.0)
            kv_rms_chunk_inline30__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_26, pl.const(16384, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.load(
                pooled_kv_inline67__rv_v2, [b0_inline35__ssa_v0, 0], [16, 64], [16, 64], target_memory=pl.Mem.Vec
            )
            kv_rms_chunk_inline30__tile_1: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_30, pl.const(28736, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.load(
                pooled_kv_inline67__rv_v2, [b0_inline35__ssa_v0, 64], [16, 64], [16, 64], target_memory=pl.Mem.Vec
            )
            kv_rms_sq_inline28__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_25, pl.const(12288, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.mul(
                kv_rms_chunk_inline30__tile, kv_rms_chunk_inline30__tile
            )
            tmp_tile: pl.Tile[[16, 128], pl.FP32, pl.MemRef(mem_vec_26, pl.const(16384, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.create([16, 128], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            t__tile_6: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_27, pl.const(24576, pl.INT64), 64), pl.Mem.Vec] = pl.tile.row_sum(kv_rms_sq_inline28__tile, tmp_tile)
            kv_rms_rowsum_inline27__tile: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_27, pl.const(24576, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(t__tile_6, [1, 16])
            partial_sq_inline50__tile_1: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_26, pl.const(16384, pl.INT64), 64), pl.Mem.Vec] = pl.tile.add(
                partial_sq_inline50__tile, kv_rms_rowsum_inline27__tile
            )
            kv_rms_sq_inline28__tile_1: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_29, pl.const(24640, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.mul(
                kv_rms_chunk_inline30__tile_1, kv_rms_chunk_inline30__tile_1
            )
            tmp_tile_1: pl.Tile[[16, 128], pl.FP32, pl.MemRef(mem_vec_30, pl.const(28736, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.create([16, 128], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            t__tile_7: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_31, pl.const(36928, pl.INT64), 64), pl.Mem.Vec] = pl.tile.row_sum(kv_rms_sq_inline28__tile_1, tmp_tile_1)
            kv_rms_rowsum_inline27__tile_1: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_31, pl.const(36928, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(t__tile_7, [1, 16])
            partial_sq_inline50__tile_2: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_49, pl.const(36992, pl.INT64), 64), pl.Mem.Vec] = pl.tile.add(
                partial_sq_inline50__tile_1, kv_rms_rowsum_inline27__tile_1
            )
            t__tile_8: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_26, pl.const(16384, pl.INT64), 64), pl.Mem.Vec] = pl.tile.muls(partial_sq_inline50__tile_2, 0.0078125)
            t__tile_9: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_26, pl.const(16384, pl.INT64), 64), pl.Mem.Vec] = pl.tile.adds(t__tile_8, 9.9999999999999995e-07)
            variance_inline26__tile: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_26, pl.const(16384, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(t__tile_9, [16, 1])
            t__rm_a0_tmp_v0: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_26, pl.const(16384, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(variance_inline26__tile, [1, 16])
            t__row_major_tmp_v1: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_26, pl.const(16384, pl.INT64), 64), pl.Mem.Vec] = pl.tile.sqrt(t__rm_a0_tmp_v0)
            t__tile_10: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_26, pl.const(16384, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(t__row_major_tmp_v1, [16, 1])
            inv_rms_inline25__rm_a0_tmp_v2: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_26, pl.const(16384, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(t__tile_10, [1, 16])
            inv_rms_inline25__row_major_tmp_v3: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_29, pl.const(24640, pl.INT64), 64), pl.Mem.Vec] = pl.tile.recip(inv_rms_inline25__rm_a0_tmp_v2)
            inv_rms_inline25__tile: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_29, pl.const(24640, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(inv_rms_inline25__row_major_tmp_v3, [16, 1])
            kv_norm_chunk_inline24__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_26, pl.const(16384, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.load(
                pooled_kv_inline67__rv_v2, [b0_inline35__ssa_v0, 0], [16, 64], [16, 64], target_memory=pl.Mem.Vec
            )
            gamma_inline70__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_30, pl.const(28736, pl.INT64), 256), pl.Mem.Vec] = pl.tile.load(
                norm_w_2d_inline68__ssa_v0, [0, 0], [1, 64], [1, 64], target_memory=pl.Mem.Vec
            )
            t__tile_11: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_26, pl.const(16384, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.row_expand_mul(kv_norm_chunk_inline24__tile, inv_rms_inline25__tile)
            normed_chunk_inline55__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_26, pl.const(16384, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.col_expand_mul(t__tile_11, gamma_inline70__tile)
            normed_nope_inline72__tile: pl.Tile[[16, 64], pl.BF16, pl.MemRef(mem_vec_25, pl.const(12288, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(
                normed_chunk_inline55__tile, target_type=pl.BF16, mode="rint"
            )
            kv_rope_norm_inline23__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_26, pl.const(16384, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.load(
                pooled_kv_inline67__rv_v2, [b0_inline35__ssa_v0, 64], [16, 64], [16, 64], target_memory=pl.Mem.Vec
            )
            gamma_rope_inline43__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_30, pl.const(28736, pl.INT64), 256), pl.Mem.Vec] = pl.tile.load(
                norm_w_2d_inline68__ssa_v0, [0, 64], [1, 64], [1, 64], target_memory=pl.Mem.Vec
            )
            t__tile_12: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_26, pl.const(16384, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.row_expand_mul(kv_rope_norm_inline23__tile, inv_rms_inline25__tile)
            rope_normed_inline22__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_26, pl.const(16384, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.col_expand_mul(t__tile_12, gamma_rope_inline43__tile)
            gather_acc_init: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_30, pl.const(28736, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.create([16, 64], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            for gather_lv, (gather_ia,) in pl.range(16, init_values=(gather_acc_init,)):
                gather_inp_row: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_26, pl.const(16384, pl.INT64), 256), pl.Mem.Vec] = pl.tile.slice(
                    rope_normed_inline22__tile, [1, 64], [gather_lv, 0], [1, 64]
                )
                gather_idx_row: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_19, pl.const(0, pl.INT64), 256), pl.Mem.Vec] = pl.tile.slice(
                    rope_swap_idx_inline89__tile, [1, 64], [gather_lv, 0], [1, 64]
                )
                gather_row_tmp: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_29, pl.const(24640, pl.INT64), 256), pl.Mem.Vec] = pl.tile.create([1, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec)
                gather_row: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_49, pl.const(36992, pl.INT64), 256), pl.Mem.Vec] = pl.tile.gather(gather_inp_row, gather_idx_row, gather_row_tmp)
                gather_asmbl: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_30, pl.const(28736, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.assemble(gather_ia, gather_row, [gather_lv, 0])
                gather_rv: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_30, pl.const(28736, pl.INT64), 4096), pl.Mem.Vec] = pl.yield_(gather_asmbl)
            swapped_inline21__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_30, pl.const(28736, pl.INT64), 4096), pl.Mem.Vec] = gather_rv
            t__tile_13: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_26, pl.const(16384, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.mul(rope_normed_inline22__tile, cos_b_inline33__tile)
            t__tile_14: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_30, pl.const(28736, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.mul(swapped_inline21__tile, sin_b_inline32__tile)
            rope_rot_inline20__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_26, pl.const(16384, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.add(t__tile_13, t__tile_14)
            normed_rope_inline19__tile: pl.Tile[[16, 64], pl.BF16, pl.MemRef(mem_vec_26, pl.const(16384, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(
                rope_rot_inline20__tile, target_type=pl.BF16, mode="rint"
            )
            for inner_inline18__idx_v0, (normalized__iter_v4,) in pl.range(rms_blk_rows_inline34__ssa_v0, init_values=(normalized__iter_v2,)):
                token_inline69__ssa_v2: pl.Scalar[pl.INDEX] = b0_inline35__ssa_v0 + inner_inline18__idx_v0
                token_pos_inline45__tile: pl.Scalar[pl.INT32] = pl.tensor.read(positions__ssa_v0, [token_inline69__ssa_v2])
                if (pl.cast(token_pos_inline45__tile, pl.INDEX) + 1) % 4 == 0:
                    request_inline17__ssa_v0: pl.Scalar[pl.INDEX] = token_inline69__ssa_v2 // 6
                    first_pos_inline29__tile: pl.Scalar[pl.INT32] = pl.tensor.read(positions__ssa_v0, [request_inline17__ssa_v0 * 6])
                    first_boundary_inline16__ssa_v0: pl.Scalar[pl.INDEX] = 3 - pl.cast(first_pos_inline29__tile, pl.INDEX) % 4
                    compact_token_inline79__ssa_v0: pl.Scalar[pl.INDEX] = request_inline17__ssa_v0 * 2 + (token_inline69__ssa_v2 % 6 - first_boundary_inline16__ssa_v0) // 4
                    t__tile_15: pl.Tile[[1, 64], pl.BF16, pl.MemRef(mem_vec_25, pl.const(12288, pl.INT64), 128), pl.Mem.Vec] = pl.tile.slice(
                        normed_nope_inline72__tile, [1, 64], [inner_inline18__idx_v0, 0]
                    )
                    normalized__tile: pl.Tensor[[384, 128], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 98304)] = pl.tile.store(
                        t__tile_15, [compact_token_inline79__ssa_v0, 0], normalized__iter_v4
                    )
                    t__tile_16: pl.Tile[[1, 64], pl.BF16, pl.MemRef(mem_vec_26, pl.const(16384, pl.INT64), 128), pl.Mem.Vec] = pl.tile.slice(
                        normed_rope_inline19__tile, [1, 64], [inner_inline18__idx_v0, 0]
                    )
                    normalized__tile_1: pl.Tensor[[384, 128], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 98304)] = pl.tile.store(
                        t__tile_16, [compact_token_inline79__ssa_v0, 64], normalized__tile
                    )
                    normalized__phi_v8: pl.Tensor[[384, 128], pl.BF16, pl.MemRef("mem_ddr_55", pl.const(0, pl.INT64), 98304)] = pl.yield_(normalized__tile_1)
                else:
                    normalized__phi_v8: pl.Tensor[[384, 128], pl.BF16, pl.MemRef("mem_ddr_55", pl.const(0, pl.INT64), 98304)] = pl.yield_(normalized__iter_v4)
                normalized__rv_v5: pl.Tensor[[384, 128], pl.BF16, pl.MemRef("mem_ddr_56", pl.const(0, pl.INT64), 98304)] = pl.yield_(normalized__phi_v8)
            normalized__rv_v3: pl.Tensor[[384, 128], pl.BF16, pl.MemRef("mem_ddr_57", pl.const(0, pl.INT64), 98304)] = pl.yield_(normalized__rv_v5)
        return normalized__ssa_v1

    @pl.function(type=pl.FunctionType.Spmd)
    def rmsnorm_rope_spmd(
        self,
        normalized__ssa_v1: pl.Out[pl.Tensor[[384, 128], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 98304)]],
        rms_blocks_inline60__ssa_v0: pl.Scalar[pl.INDEX],
        rms_workers_inline93__ssa_v0: pl.Scalar[pl.INDEX],
        bs_inline52__ssa_v0: pl.Scalar[pl.INDEX],
        cos__ssa_v0: pl.Tensor[[T_DYN, 64], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)],
        sin__ssa_v0: pl.Tensor[[T_DYN, 64], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)],
        pooled_kv_inline67__rv_v2: pl.Tensor[[384, 128], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 196608)],
        norm_w_2d_inline68__ssa_v0: pl.Tensor[[1, 128], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 512)],
        positions__ssa_v0: pl.Tensor[[T_DYN], pl.INT32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 0)],
    ) -> pl.Tensor[[384, 128], pl.BF16]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        normalized__rv_v3: pl.Tensor[[384, 128], pl.BF16, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 98304)] = self.rmsnorm_rope(
            normalized__ssa_v1,
            rms_blocks_inline60__ssa_v0,
            rms_workers_inline93__ssa_v0,
            bs_inline52__ssa_v0,
            cos__ssa_v0,
            sin__ssa_v0,
            pooled_kv_inline67__rv_v2,
            norm_w_2d_inline68__ssa_v0,
            positions__ssa_v0,
            attrs={"arg_directions": [pl.adir.output_existing, pl.adir.scalar, pl.adir.scalar, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input]},
        )
        return normalized__ssa_v1

    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def scatter_softmax_pool(
        pooled_kv_inline67__ssa_v0: pl.Out[pl.Tensor[[384, 128], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 196608)]],
        b_dim_inline51__ssa_v0: pl.Scalar[pl.INDEX],
        pool_workers_inline59__ssa_v0: pl.Scalar[pl.INDEX],
        positions__ssa_v0: pl.Tensor[[T_DYN], pl.INT32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)],
        s_dim_inline54__ssa_v0: pl.Scalar[pl.INDEX],
        scores__rv_v2: pl.Tensor[[384, 256], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 393216)],
        ape__ssa_v0: pl.Tensor[[4, 256], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 4096)],
        values__rv_v2: pl.Tensor[[384, 256], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 393216)],
        table__ssa_v0: pl.Tensor[[B_DYN, 7], pl.INT32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 0)],
        compress_state_flat_inline39__ssa_v0: pl.Tensor[[compress_state_rows_inline44__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 0)],
    ) -> pl.Tensor[[384, 128], pl.FP32]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        mem_vec_7: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 512)
        mem_vec_9: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        mem_vec_13: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        mem_vec_14: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        mem_vec_15: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        mem_vec_16: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        mem_vec_17: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        mem_vec_24: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        pool_worker_inline95__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        for c_idx_inline63__idx_v0, (pooled_kv_inline67__iter_v1,) in pl.range(
            pool_worker_inline95__ssa_v0, b_dim_inline51__ssa_v0, pool_workers_inline59__ssa_v0, init_values=(pooled_kv_inline67__ssa_v0,)
        ):
            first_pos_b_inline62__tile: pl.Scalar[pl.INT32] = pl.tensor.read(positions__ssa_v0, [c_idx_inline63__idx_v0 * s_dim_inline54__ssa_v0])
            for s_idx_inline46__idx_v0, (pooled_kv_inline67__iter_v3,) in pl.range(s_dim_inline54__ssa_v0, init_values=(pooled_kv_inline67__iter_v1,)):
                token_inline69__ssa_v0: pl.Scalar[pl.INDEX] = c_idx_inline63__idx_v0 * s_dim_inline54__ssa_v0 + s_idx_inline46__idx_v0
                token_pos_inline45__tile: pl.Scalar[pl.INT32] = pl.tensor.read(positions__ssa_v0, [token_inline69__ssa_v0])
                t__tile: pl.Tile[[1, 128], pl.FP32, pl.MemRef(mem_vec_7, pl.const(1536, pl.INT64), 512), pl.Mem.Vec] = pl.tile.full([1, 128], dtype=pl.FP32, value=0.0)
                pooled_kv_inline67__tile: pl.Tensor[[384, 128], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 196608)] = pl.tile.store(
                    t__tile, [token_inline69__ssa_v0, 0], pooled_kv_inline67__iter_v3
                )
                if (pl.cast(token_pos_inline45__tile, pl.INDEX) + 1) % 4 == 0:
                    window_start_inline65__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(token_pos_inline45__tile, pl.INDEX) - 8 + 1
                    for h0_inline47__idx_v0, (pooled_kv_inline67__iter_v6,) in pl.range(0, 128, 64, init_values=(pooled_kv_inline67__tile,)):
                        last_ape_row_inline97__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(pl.cast(token_pos_inline45__tile, pl.INDEX) % 4, pl.INDEX)
                        t__tile_1: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_7, pl.const(1536, pl.INT64), 256), pl.Mem.Vec] = pl.tile.load(
                            scores__rv_v2, [token_inline69__ssa_v0, h0_inline47__idx_v0 + 128], [1, 64], [1, 64], target_memory=pl.Mem.Vec
                        )
                        t__tile_2: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_9, pl.const(2048, pl.INT64), 256), pl.Mem.Vec] = pl.tile.load(
                            ape__ssa_v0, [last_ape_row_inline97__ssa_v0, h0_inline47__idx_v0 + 128], [1, 64], [1, 64], target_memory=pl.Mem.Vec
                        )
                        mi_inline42__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_7, pl.const(1536, pl.INT64), 256), pl.Mem.Vec] = pl.tile.add(t__tile_1, t__tile_2)
                        t__tile_3: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_9, pl.const(2048, pl.INT64), 256), pl.Mem.Vec] = pl.tile.sub(mi_inline42__tile, mi_inline42__tile)
                        li_inline48__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_9, pl.const(2048, pl.INT64), 256), pl.Mem.Vec] = pl.tile.exp(t__tile_3)
                        oi_inline40__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(0, pl.INT64), 256), pl.Mem.Vec] = pl.tile.load(
                            values__rv_v2, [token_inline69__ssa_v0, h0_inline47__idx_v0 + 128], [1, 64], [1, 64], target_memory=pl.Mem.Vec
                        )
                        for state_idx_inline61__idx_v0, (li_inline48__iter_v1, mi_inline42__iter_v1, oi_inline40__iter_v1) in pl.range(
                            7, init_values=(li_inline48__tile, mi_inline42__tile, oi_inline40__tile)
                        ):
                            logical_pos_inline56__ssa_v0: pl.Scalar[pl.INDEX] = window_start_inline65__ssa_v0 + state_idx_inline61__idx_v0
                            value_inline58__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_14, pl.const(256, pl.INT64), 256), pl.Mem.Vec] = pl.tile.full([1, 64], dtype=pl.FP32, value=0.0)
                            score_inline38__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_15, pl.const(512, pl.INT64), 256), pl.Mem.Vec] = pl.tile.full(
                                [1, 64], dtype=pl.FP32, value=-3.4028234663852886e38
                            )
                            state_half_inline36__ssa_v0: pl.Scalar[pl.INDEX] = 0
                            if 4 <= state_idx_inline61__idx_v0:
                                state_half_inline36__ssa_v1: pl.Scalar[pl.INDEX] = 128
                                state_half_inline36__phi_v2: pl.Scalar[pl.INDEX] = pl.yield_(state_half_inline36__ssa_v1)
                            else:
                                state_half_inline36__phi_v2: pl.Scalar[pl.INDEX] = pl.yield_(state_half_inline36__ssa_v0)
                            if 0 <= logical_pos_inline56__ssa_v0 and logical_pos_inline56__ssa_v0 < pl.cast(first_pos_b_inline62__tile, pl.INDEX):
                                ring_row_inline49__ssa_v0: pl.Scalar[pl.INDEX] = logical_pos_inline56__ssa_v0 % 14
                                state_page_off_inline71__ssa_v0: pl.Scalar[pl.INDEX] = ring_row_inline49__ssa_v0 // 2
                                state_blk_id_i32_inline64__tile: pl.Scalar[pl.INT32] = pl.tensor.read(table__ssa_v0, [c_idx_inline63__idx_v0, state_page_off_inline71__ssa_v0])
                                if 0 <= pl.cast(state_blk_id_i32_inline64__tile, pl.INDEX):
                                    state_blk_id_inline74__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(state_blk_id_i32_inline64__tile, pl.INDEX)
                                    state_intra_row_inline76__ssa_v0: pl.Scalar[pl.INDEX] = ring_row_inline49__ssa_v0 % 2
                                    state_row_inline87__ssa_v0: pl.Scalar[pl.INDEX] = state_blk_id_inline74__ssa_v0 * 2 + state_intra_row_inline76__ssa_v0
                                    value_inline58__tile_1: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_16, pl.const(768, pl.INT64), 256), pl.Mem.Vec] = pl.tile.load(
                                        compress_state_flat_inline39__ssa_v0,
                                        [state_row_inline87__ssa_v0, state_half_inline36__phi_v2 + h0_inline47__idx_v0],
                                        [1, 64],
                                        [1, 64],
                                        target_memory=pl.Mem.Vec,
                                    )
                                    score_inline38__tile_1: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_17, pl.const(1024, pl.INT64), 256), pl.Mem.Vec] = pl.tile.load(
                                        compress_state_flat_inline39__ssa_v0,
                                        [state_row_inline87__ssa_v0, state_half_inline36__phi_v2 + h0_inline47__idx_v0 + 256],
                                        [1, 64],
                                        [1, 64],
                                        target_memory=pl.Mem.Vec,
                                    )
                                    score_inline38__phi_v2, value_inline58__phi_v2 = pl.yield_(score_inline38__tile_1, value_inline58__tile_1)
                                else:
                                    score_inline38__tile_mv: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_17, pl.const(1024, pl.INT64), 256), pl.Mem.Vec] = pl.tile.move(
                                        score_inline38__tile, target_memory=pl.Mem.Vec
                                    )
                                    value_inline58__tile_mv: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_16, pl.const(768, pl.INT64), 256), pl.Mem.Vec] = pl.tile.move(
                                        value_inline58__tile, target_memory=pl.Mem.Vec
                                    )
                                    score_inline38__phi_v2, value_inline58__phi_v2 = pl.yield_(score_inline38__tile_mv, value_inline58__tile_mv)
                                score_inline38__phi_v3, value_inline58__phi_v3 = pl.yield_(score_inline38__phi_v2, value_inline58__phi_v2)
                            else:
                                score_inline38__tile_mv_1: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_17, pl.const(1024, pl.INT64), 256), pl.Mem.Vec] = pl.tile.move(
                                    score_inline38__tile, target_memory=pl.Mem.Vec
                                )
                                value_inline58__tile_mv_1: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_16, pl.const(768, pl.INT64), 256), pl.Mem.Vec] = pl.tile.move(
                                    value_inline58__tile, target_memory=pl.Mem.Vec
                                )
                                score_inline38__phi_v3, value_inline58__phi_v3 = pl.yield_(score_inline38__tile_mv_1, value_inline58__tile_mv_1)
                            if pl.cast(first_pos_b_inline62__tile, pl.INDEX) <= logical_pos_inline56__ssa_v0:
                                if logical_pos_inline56__ssa_v0 <= pl.cast(token_pos_inline45__tile, pl.INDEX):
                                    overlay_token_inline57__ssa_v0: pl.Scalar[pl.INDEX] = (
                                        c_idx_inline63__idx_v0 * s_dim_inline54__ssa_v0 + logical_pos_inline56__ssa_v0 - pl.cast(first_pos_b_inline62__tile, pl.INDEX)
                                    )
                                    ape_row_inline53__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(logical_pos_inline56__ssa_v0 % 4, pl.INDEX)
                                    value_inline58__tile_2: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_14, pl.const(256, pl.INT64), 256), pl.Mem.Vec] = pl.tile.load(
                                        values__rv_v2, [overlay_token_inline57__ssa_v0, state_half_inline36__phi_v2 + h0_inline47__idx_v0], [1, 64], [1, 64], target_memory=pl.Mem.Vec
                                    )
                                    t__tile_4: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_15, pl.const(512, pl.INT64), 256), pl.Mem.Vec] = pl.tile.load(
                                        scores__rv_v2, [overlay_token_inline57__ssa_v0, state_half_inline36__phi_v2 + h0_inline47__idx_v0], [1, 64], [1, 64], target_memory=pl.Mem.Vec
                                    )
                                    t__tile_5: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_24, pl.const(1280, pl.INT64), 256), pl.Mem.Vec] = pl.tile.load(
                                        ape__ssa_v0, [ape_row_inline53__ssa_v0, state_half_inline36__phi_v2 + h0_inline47__idx_v0], [1, 64], [1, 64], target_memory=pl.Mem.Vec
                                    )
                                    score_inline38__tile_2: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_15, pl.const(512, pl.INT64), 256), pl.Mem.Vec] = pl.tile.add(t__tile_4, t__tile_5)
                                    score_inline38__phi_v5, value_inline58__phi_v5 = pl.yield_(score_inline38__tile_2, value_inline58__tile_2)
                                else:
                                    score_inline38__phi_v3_mv: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_15, pl.const(512, pl.INT64), 256), pl.Mem.Vec] = pl.tile.move(
                                        score_inline38__phi_v3, target_memory=pl.Mem.Vec
                                    )
                                    value_inline58__phi_v3_mv: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_14, pl.const(256, pl.INT64), 256), pl.Mem.Vec] = pl.tile.move(
                                        value_inline58__phi_v3, target_memory=pl.Mem.Vec
                                    )
                                    score_inline38__phi_v5, value_inline58__phi_v5 = pl.yield_(score_inline38__phi_v3_mv, value_inline58__phi_v3_mv)
                                score_inline38__phi_v6, value_inline58__phi_v6 = pl.yield_(score_inline38__phi_v5, value_inline58__phi_v5)
                            else:
                                score_inline38__phi_v3_mv_1: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_15, pl.const(512, pl.INT64), 256), pl.Mem.Vec] = pl.tile.move(
                                    score_inline38__phi_v3, target_memory=pl.Mem.Vec
                                )
                                value_inline58__phi_v3_mv_1: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_14, pl.const(256, pl.INT64), 256), pl.Mem.Vec] = pl.tile.move(
                                    value_inline58__phi_v3, target_memory=pl.Mem.Vec
                                )
                                score_inline38__phi_v6, value_inline58__phi_v6 = pl.yield_(score_inline38__phi_v3_mv_1, value_inline58__phi_v3_mv_1)
                            mi_next_inline77__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_16, pl.const(768, pl.INT64), 256), pl.Mem.Vec] = pl.tile.maximum(
                                mi_inline42__iter_v1, score_inline38__phi_v6
                            )
                            t__tile_6: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_17, pl.const(1024, pl.INT64), 256), pl.Mem.Vec] = pl.tile.sub(mi_inline42__iter_v1, mi_next_inline77__tile)
                            alpha_inline78__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_17, pl.const(1024, pl.INT64), 256), pl.Mem.Vec] = pl.tile.exp(t__tile_6)
                            t__tile_7: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_15, pl.const(512, pl.INT64), 256), pl.Mem.Vec] = pl.tile.sub(score_inline38__phi_v6, mi_next_inline77__tile)
                            beta_inline80__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_15, pl.const(512, pl.INT64), 256), pl.Mem.Vec] = pl.tile.exp(t__tile_7)
                            t__tile_8: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_24, pl.const(1280, pl.INT64), 256), pl.Mem.Vec] = pl.tile.mul(alpha_inline78__tile, li_inline48__iter_v1)
                            li_inline48__tile_1: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_9, pl.const(2048, pl.INT64), 256), pl.Mem.Vec] = pl.tile.add(t__tile_8, beta_inline80__tile)
                            t__tile_9: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_17, pl.const(1024, pl.INT64), 256), pl.Mem.Vec] = pl.tile.mul(oi_inline40__iter_v1, alpha_inline78__tile)
                            t__tile_10: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_14, pl.const(256, pl.INT64), 256), pl.Mem.Vec] = pl.tile.mul(value_inline58__phi_v6, beta_inline80__tile)
                            oi_inline40__tile_1: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(0, pl.INT64), 256), pl.Mem.Vec] = pl.tile.add(t__tile_9, t__tile_10)
                            mi_inline42__ssa_v3: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_16, pl.const(768, pl.INT64), 256), pl.Mem.Vec] = mi_next_inline77__tile
                            mi_inline42__ssa_v3_mv: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_7, pl.const(1536, pl.INT64), 256), pl.Mem.Vec] = pl.tile.move(
                                mi_inline42__ssa_v3, target_memory=pl.Mem.Vec
                            )
                            li_inline48__rv_v2, mi_inline42__rv_v2, oi_inline40__rv_v2 = pl.yield_(li_inline48__tile_1, mi_inline42__ssa_v3_mv, oi_inline40__tile_1)
                        t__tile_11: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_7, pl.const(1536, pl.INT64), 256), pl.Mem.Vec] = pl.tile.div(oi_inline40__rv_v2, li_inline48__rv_v2)
                        pooled_kv_inline67__tile_1: pl.Tensor[[384, 128], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 196608)] = pl.tile.store(
                            t__tile_11, [token_inline69__ssa_v0, h0_inline47__idx_v0], pooled_kv_inline67__iter_v6
                        )
                        pooled_kv_inline67__rv_v7: pl.Tensor[[384, 128], pl.FP32, pl.MemRef("mem_ddr_44", pl.const(0, pl.INT64), 196608)] = pl.yield_(pooled_kv_inline67__tile_1)
                    pooled_kv_inline67__phi_v9: pl.Tensor[[384, 128], pl.FP32, pl.MemRef("mem_ddr_45", pl.const(0, pl.INT64), 196608)] = pl.yield_(pooled_kv_inline67__rv_v7)
                else:
                    pooled_kv_inline67__phi_v9: pl.Tensor[[384, 128], pl.FP32, pl.MemRef("mem_ddr_45", pl.const(0, pl.INT64), 196608)] = pl.yield_(pooled_kv_inline67__tile)
                pooled_kv_inline67__rv_v4: pl.Tensor[[384, 128], pl.FP32, pl.MemRef("mem_ddr_46", pl.const(0, pl.INT64), 196608)] = pl.yield_(pooled_kv_inline67__phi_v9)
            pooled_kv_inline67__rv_v2: pl.Tensor[[384, 128], pl.FP32, pl.MemRef("mem_ddr_47", pl.const(0, pl.INT64), 196608)] = pl.yield_(pooled_kv_inline67__rv_v4)
        return pooled_kv_inline67__ssa_v0

    @pl.function(type=pl.FunctionType.Spmd)
    def scatter_softmax_pool_spmd(
        self,
        pooled_kv_inline67__ssa_v0: pl.Out[pl.Tensor[[384, 128], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 196608)]],
        b_dim_inline51__ssa_v0: pl.Scalar[pl.INDEX],
        pool_workers_inline59__ssa_v0: pl.Scalar[pl.INDEX],
        positions__ssa_v0: pl.Tensor[[T_DYN], pl.INT32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)],
        s_dim_inline54__ssa_v0: pl.Scalar[pl.INDEX],
        scores__rv_v2: pl.Tensor[[384, 256], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 393216)],
        ape__ssa_v0: pl.Tensor[[4, 256], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 4096)],
        values__rv_v2: pl.Tensor[[384, 256], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 393216)],
        table__ssa_v0: pl.Tensor[[B_DYN, 7], pl.INT32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 0)],
        compress_state_flat_inline39__ssa_v0: pl.Tensor[[compress_state_rows_inline44__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 0)],
    ) -> pl.Tensor[[384, 128], pl.FP32]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        pooled_kv_inline67__rv_v2: pl.Tensor[[384, 128], pl.FP32, pl.MemRef("mem_ddr_7", pl.const(0, pl.INT64), 196608)] = self.scatter_softmax_pool(
            pooled_kv_inline67__ssa_v0,
            b_dim_inline51__ssa_v0,
            pool_workers_inline59__ssa_v0,
            positions__ssa_v0,
            s_dim_inline54__ssa_v0,
            scores__rv_v2,
            ape__ssa_v0,
            values__rv_v2,
            table__ssa_v0,
            compress_state_flat_inline39__ssa_v0,
            attrs={
                "arg_directions": [pl.adir.output_existing, pl.adir.scalar, pl.adir.scalar, pl.adir.input, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input]
            },
        )
        return pooled_kv_inline67__ssa_v0

    @pl.function(type=pl.FunctionType.Orchestration, level=pl.Level.CHIP, role=pl.Role.Orchestrator, auto_scope=False)
    def diagnose_indexer_compressor(
        self,
        x__ssa_v0: pl.Tensor[[T_DYN, 4096], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)],
        state__ssa_v0: pl.InOut[pl.Tensor[[COMPRESS_STATE_BLOCK_NUM_DYN, 2, 512], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)]],
        table__ssa_v0: pl.Tensor[[B_DYN, 7], pl.INT32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)],
        wkv__ssa_v0: pl.Tensor[[256, 4096], pl.BF16, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 2097152)],
        wgate__ssa_v0: pl.Tensor[[256, 4096], pl.BF16, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 2097152)],
        ape__ssa_v0: pl.Tensor[[4, 256], pl.FP32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 4096)],
        norm__ssa_v0: pl.Tensor[[128], pl.FP32, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 512)],
        cos__ssa_v0: pl.Tensor[[T_DYN, 64], pl.FP32, pl.MemRef("mem_ddr_7", pl.const(0, pl.INT64), 0)],
        sin__ssa_v0: pl.Tensor[[T_DYN, 64], pl.FP32, pl.MemRef("mem_ddr_8", pl.const(0, pl.INT64), 0)],
        positions__ssa_v0: pl.Tensor[[T_DYN], pl.INT32, pl.MemRef("mem_ddr_9", pl.const(0, pl.INT64), 0)],
        state_slots__ssa_v0: pl.Tensor[[T_DYN], pl.INT64, pl.MemRef("mem_ddr_10", pl.const(0, pl.INT64), 0)],
        hadamard__ssa_v0: pl.Tensor[[128, 128], pl.BF16, pl.MemRef("mem_ddr_11", pl.const(0, pl.INT64), 32768)],
        cache__ssa_v0: pl.InOut[pl.Tensor[[IDX_CACHE_BLOCK_NUM_DYN, INDEXER_PAGE_BYTES_DYN], pl.INT8, pl.MemRef("mem_ddr_12", pl.const(0, pl.INT64), 0)]],
        slots__ssa_v0: pl.Tensor[[T_DYN], pl.INT64, pl.MemRef("mem_ddr_13", pl.const(0, pl.INT64), 0)],
        values__ssa_v0: pl.Out[pl.Tensor[[384, 256], pl.FP32, pl.MemRef("mem_ddr_14", pl.const(0, pl.INT64), 393216)]],
        scores__ssa_v0: pl.Out[pl.Tensor[[384, 256], pl.FP32, pl.MemRef("mem_ddr_15", pl.const(0, pl.INT64), 393216)]],
        normalized__ssa_v0: pl.Out[pl.Tensor[[384, 128], pl.BF16, pl.MemRef("mem_ddr_16", pl.const(0, pl.INT64), 98304)]],
        rotated__ssa_v0: pl.Out[pl.Tensor[[T_DYN, 128], pl.FP32, pl.MemRef("mem_ddr_17", pl.const(0, pl.INT64), 0)]],
    ) -> tuple[
        pl.Tensor[[COMPRESS_STATE_BLOCK_NUM_DYN, 2, 512], pl.FP32],
        pl.Tensor[[IDX_CACHE_BLOCK_NUM_DYN, INDEXER_PAGE_BYTES_DYN], pl.INT8],
        pl.Tensor[[384, 256], pl.FP32],
        pl.Tensor[[384, 256], pl.FP32],
        pl.Tensor[[384, 128], pl.BF16],
        pl.Tensor[[T_DYN, 128], pl.FP32],
    ]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        ret__tmp_v0: pl.Tuple[pl.Tensor[[T_DYN, 128], pl.FP32], pl.Scalar[pl.TASK_ID]] = pl.submit(
            self.compressor_diagnostic_seed, rotated__ssa_v0, attrs={"arg_directions": [pl.adir.output_existing]}
        )
        rotated__ssa_v1: pl.Tensor[[T_DYN, 128], pl.FP32, pl.MemRef("mem_ddr_18", pl.const(0, pl.INT64), 0)] = ret__tmp_v0[0]
        seed__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0[1]
        bs_inline5__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(x__ssa_v0, 0)
        t_matmul_inline2__ssa_v0: pl.Scalar[pl.INDEX] = (bs_inline5__ssa_v0 + 15) // 16 * 16
        x_flat_inline3__ssa_v0: pl.Tensor[[T_DYN, 4096], pl.BF16, pl.MemRef("mem_ddr_19", pl.const(0, pl.INT64), 0)] = x__ssa_v0
        ret__tmp_v0_1: pl.Tuple[pl.Tensor[[384, 256], pl.FP32], pl.Tensor[[384, 256], pl.FP32], pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(
            self.kv_score_proj_spmd,
            scores__ssa_v0,
            values__ssa_v0,
            t_matmul_inline2__ssa_v0,
            bs_inline5__ssa_v0,
            x_flat_inline3__ssa_v0,
            wkv__ssa_v0,
            wgate__ssa_v0,
            deps=[seed__ssa_v0, seed__ssa_v0],
            core_num=24,
            attrs={"arg_directions": [pl.adir.output_existing, pl.adir.output_existing, pl.adir.scalar, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.input]},
        )
        scores__rv_v2: pl.Tensor[[384, 256], pl.FP32, pl.MemRef("mem_ddr_20", pl.const(0, pl.INT64), 393216)] = ret__tmp_v0_1[0]
        values__rv_v2: pl.Tensor[[384, 256], pl.FP32, pl.MemRef("mem_ddr_21", pl.const(0, pl.INT64), 393216)] = ret__tmp_v0_1[1]
        _kv_score_tid_inline6__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_1[2]
        projected__ssa_v0: pl.Scalar[pl.TASK_ID] = _kv_score_tid_inline6__ssa_v0
        b_dim_inline51__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(table__ssa_v0, 0)
        bs_inline52__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(positions__ssa_v0, 0)
        s_dim_inline54__ssa_v0: pl.Scalar[pl.INDEX] = bs_inline52__ssa_v0 // b_dim_inline51__ssa_v0
        rms_blocks_inline60__ssa_v0: pl.Scalar[pl.INDEX] = (bs_inline52__ssa_v0 + 15) // 16
        compress_state_block_num_inline66__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(state__ssa_v0, 0)
        compress_state_rows_inline44__ssa_v0: pl.Scalar[pl.INDEX] = compress_state_block_num_inline66__ssa_v0 * 2
        compress_state_flat_inline39__ssa_v0: pl.Tensor[[compress_state_rows_inline44__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)] = pl.tensor.reshape(
            state__ssa_v0, [compress_state_rows_inline44__ssa_v0, 512]
        )
        _kv_score_tid_inline99__ssa_v0: pl.Scalar[pl.TASK_ID] = projected__ssa_v0
        pooled_kv_inline67__ssa_v0: pl.Tensor[[384, 128], pl.FP32, pl.MemRef("mem_ddr_22", pl.const(0, pl.INT64), 196608)] = pl.tensor.create([384, 128], dtype=pl.FP32, layout=pl.TensorLayout.ND)
        pool_workers_inline59__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(b_dim_inline51__ssa_v0, 48)
        ret__tmp_v0_2: pl.Tuple[pl.Tensor[[384, 128], pl.FP32], pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(
            self.scatter_softmax_pool_spmd,
            pooled_kv_inline67__ssa_v0,
            b_dim_inline51__ssa_v0,
            pool_workers_inline59__ssa_v0,
            positions__ssa_v0,
            s_dim_inline54__ssa_v0,
            scores__rv_v2,
            ape__ssa_v0,
            values__rv_v2,
            table__ssa_v0,
            compress_state_flat_inline39__ssa_v0,
            deps=[_kv_score_tid_inline99__ssa_v0],
            core_num=pool_workers_inline59__ssa_v0,
            attrs={
                "arg_directions": [pl.adir.output_existing, pl.adir.scalar, pl.adir.scalar, pl.adir.input, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input]
            },
        )
        pooled_kv_inline67__rv_v2: pl.Tensor[[384, 128], pl.FP32, pl.MemRef("mem_ddr_23", pl.const(0, pl.INT64), 196608)] = ret__tmp_v0_2[0]
        pool_tid_inline86__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_2[1]
        commit_workers_inline81__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(b_dim_inline51__ssa_v0, 48)
        ret__tmp_v0_3: pl.Tuple[pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(
            self.compress_state_commit_spmd,
            compress_state_flat_inline39__ssa_v0,
            b_dim_inline51__ssa_v0,
            commit_workers_inline81__ssa_v0,
            s_dim_inline54__ssa_v0,
            state_slots__ssa_v0,
            positions__ssa_v0,
            values__rv_v2,
            scores__rv_v2,
            ape__ssa_v0,
            deps=[pool_tid_inline86__ssa_v0],
            core_num=commit_workers_inline81__ssa_v0,
            attrs={"arg_directions": [pl.adir.output_existing, pl.adir.scalar, pl.adir.scalar, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input]},
        )
        tid__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_3[0]
        norm_w_2d_inline68__ssa_v0: pl.Tensor[[1, 128], pl.FP32, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 512)] = pl.tensor.reshape(norm__ssa_v0, [1, 128])
        ret__tmp_v0_4: pl.Tuple[pl.Tensor[[384, 128], pl.BF16], pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(
            self.indexer_boundary_init_spmd, normalized__ssa_v0, deps=[pool_tid_inline86__ssa_v0], core_num=b_dim_inline51__ssa_v0, attrs={"arg_directions": [pl.adir.output_existing]}
        )
        normalized__ssa_v1: pl.Tensor[[384, 128], pl.BF16, pl.MemRef("mem_ddr_24", pl.const(0, pl.INT64), 98304)] = ret__tmp_v0_4[0]
        boundary_init_tid_inline90__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_4[1]
        rms_workers_inline93__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(rms_blocks_inline60__ssa_v0, 2)
        ret__tmp_v0_5: pl.Tuple[pl.Tensor[[384, 128], pl.BF16], pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(
            self.rmsnorm_rope_spmd,
            normalized__ssa_v1,
            rms_blocks_inline60__ssa_v0,
            rms_workers_inline93__ssa_v0,
            bs_inline52__ssa_v0,
            cos__ssa_v0,
            sin__ssa_v0,
            pooled_kv_inline67__rv_v2,
            norm_w_2d_inline68__ssa_v0,
            positions__ssa_v0,
            deps=[pool_tid_inline86__ssa_v0, boundary_init_tid_inline90__ssa_v0],
            core_num=rms_workers_inline93__ssa_v0,
            attrs={"arg_directions": [pl.adir.inout, pl.adir.scalar, pl.adir.scalar, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input]},
        )
        normalized__rv_v3: pl.Tensor[[384, 128], pl.BF16, pl.MemRef("mem_ddr_25", pl.const(0, pl.INT64), 98304)] = ret__tmp_v0_5[0]
        rms_tid_inline96__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_5[1]
        rms__ssa_v0: pl.Scalar[pl.TASK_ID] = rms_tid_inline96__ssa_v0
        bs_inline132__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(positions__ssa_v0, 0)
        compact_rows_inline135__ssa_v0: pl.Scalar[pl.INDEX] = bs_inline132__ssa_v0 // 6 * 2
        rms_blocks_inline130__ssa_v0: pl.Scalar[pl.INDEX] = (compact_rows_inline135__ssa_v0 + 15) // 16
        kv_flat_inline128__ssa_v0: pl.Tensor[[T_DYN, 128], pl.FP32, pl.MemRef("mem_ddr_26", pl.const(0, pl.INT64), 0)] = rotated__ssa_v1
        idx_kv_scale_values_inline125__ssa_v0: pl.Tensor[[384, 1], pl.FP32, pl.MemRef("mem_ddr_27", pl.const(0, pl.INT64), 1536)] = pl.tensor.create([384, 1], dtype=pl.FP32, layout=pl.TensorLayout.ND)
        kv_final_inline120__ssa_v0: pl.Tensor[[384, 128], pl.FP32, pl.MemRef("mem_ddr_28", pl.const(0, pl.INT64), 196608)] = pl.tensor.create([384, 128], dtype=pl.FP32, layout=pl.TensorLayout.ND)
        ret__tmp_v0_6: pl.Tuple[pl.Tensor[[384, 128], pl.FP32], pl.Scalar[pl.TASK_ID]] = pl.submit(
            self.kv_hadamard,
            kv_final_inline120__ssa_v0,
            hadamard__ssa_v0,
            rms_blocks_inline130__ssa_v0,
            compact_rows_inline135__ssa_v0,
            normalized__rv_v3,
            deps=[rms__ssa_v0, seed__ssa_v0],
            attrs={"arg_directions": [pl.adir.output_existing, pl.adir.input, pl.adir.scalar, pl.adir.scalar, pl.adir.input]},
        )
        kv_final_inline120__rv_v2: pl.Tensor[[384, 128], pl.FP32, pl.MemRef("mem_ddr_29", pl.const(0, pl.INT64), 196608)] = ret__tmp_v0_6[0]
        hadamard_tid_inline131__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_6[1]
        ret__tmp_v0_7: pl.Tuple[pl.Tensor[[384, 1], pl.FP32], pl.Tensor[[IDX_CACHE_BLOCK_NUM_DYN, INDEXER_PAGE_BYTES_DYN], pl.INT8], pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(
            self.kv_and_cache_write_spmd,
            compact_rows_inline135__ssa_v0,
            kv_final_inline120__rv_v2,
            idx_kv_scale_values_inline125__ssa_v0,
            cache__ssa_v0,
            kv_flat_inline128__ssa_v0,
            positions__ssa_v0,
            slots__ssa_v0,
            deps=[hadamard_tid_inline131__ssa_v0],
            core_num=rms_blocks_inline130__ssa_v0,
            allow_early_resolve=True,
            attrs={"arg_directions": [pl.adir.scalar, pl.adir.input, pl.adir.output_existing, pl.adir.inout, pl.adir.inout, pl.adir.input, pl.adir.input]},
        )
        idx_kv_scale_values_inline125__ssa_v1: pl.Tensor[[384, 1], pl.FP32, pl.MemRef("mem_ddr_30", pl.const(0, pl.INT64), 1536)] = ret__tmp_v0_7[0]
        cache__rv_v2: pl.Tensor[[IDX_CACHE_BLOCK_NUM_DYN, INDEXER_PAGE_BYTES_DYN], pl.INT8, pl.MemRef("mem_ddr_31", pl.const(0, pl.INT64), 0)] = ret__tmp_v0_7[1]
        _write_tid_inline138__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_7[2]
        ret__tmp_v0_8: pl.Tuple[pl.Tensor[[IDX_CACHE_BLOCK_NUM_DYN, INDEXER_PAGE_BYTES_DYN], pl.INT8], pl.Scalar[pl.TASK_ID]] = pl.submit(
            self.idx_kv_scale_commit,
            compact_rows_inline135__ssa_v0,
            positions__ssa_v0,
            slots__ssa_v0,
            cache__rv_v2,
            idx_kv_scale_values_inline125__ssa_v1,
            deps=[_write_tid_inline138__ssa_v0],
            allow_early_resolve=True,
            attrs={"arg_directions": [pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.inout, pl.adir.input]},
        )
        cache__rv_v3: pl.Tensor[[IDX_CACHE_BLOCK_NUM_DYN, INDEXER_PAGE_BYTES_DYN], pl.INT8, pl.MemRef("mem_ddr_32", pl.const(0, pl.INT64), 0)] = ret__tmp_v0_8[0]
        scale_commit_tid_inline113__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_8[1]
        return state__ssa_v0, cache__rv_v3, values__rv_v2, scores__rv_v2, normalized__rv_v3, rotated__ssa_v1