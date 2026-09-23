# pypto.program: _jit_norm_rope
import pypto.language as pl

HEADS = pl.dynamic("HEADS")
TOKENS = pl.dynamic("TOKENS")


@pl.program
class _jit_norm_rope:
    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def csa_norm_rope(
        gamma__ssa_v0: pl.Tensor[[512], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 1024)],
        cos__ssa_v0: pl.Tensor[[TOKENS, 64], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)],
        sin__ssa_v0: pl.Tensor[[TOKENS, 64], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)],
        flat_output__ssa_v0: pl.Out[pl.Tensor[[rows__ssa_v0, 512], pl.BF16, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)]],
        HEADS: pl.Scalar[pl.INDEX],
        flat_input__ssa_v0: pl.Tensor[[rows__ssa_v0, 512], pl.BF16, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 0)],
    ):
        pl.func_attr({"mx_tensor_views_blocked": True})
        mem_vec_6: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 2048)
        mem_vec_8: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 128)
        mem_vec_10: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 128)
        mem_vec_12: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
        mem_vec_13: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
        mem_vec_14: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
        mem_vec_27: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 1024)
        mem_vec_31: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 1024)
        token__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        t__tile: pl.Tile[[512], pl.BF16, pl.MemRef(mem_vec_12, pl.const(128, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.load(gamma__ssa_v0, [0], [512], [512], target_memory=pl.Mem.Vec)
        t__tile_1: pl.Tile[[512], pl.FP32, pl.MemRef(mem_vec_6, pl.const(51328, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(t__tile, target_type=pl.FP32, mode="round")
        g__tile: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_6, pl.const(51328, pl.INT64), 2048), pl.Mem.Vec] = t__tile_1
        t__tile_2: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(128, pl.INT64), 256), pl.Mem.Vec] = pl.tile.load(
            cos__ssa_v0, [token__ssa_v0, 0], [1, 64], [1, 64], target_memory=pl.Mem.Vec
        )
        cos_even__tile: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_8, pl.const(53376, pl.INT64), 128), pl.Mem.Vec] = pl.tile.gather_mask(t__tile_2, mask_pattern=1)
        t__tile_3: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(128, pl.INT64), 256), pl.Mem.Vec] = pl.tile.load(
            sin__ssa_v0, [token__ssa_v0, 0], [1, 64], [1, 64], target_memory=pl.Mem.Vec
        )
        sin_even__tile: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 128), pl.Mem.Vec] = pl.tile.gather_mask(t__tile_3, mask_pattern=1)
        for head__idx_v0, (flat_output__iter_v1,) in pl.range(0, HEADS, 8, init_values=(flat_output__ssa_v0,)):
            row__ssa_v0: pl.Scalar[pl.INDEX] = token__ssa_v0 * HEADS + head__idx_v0
            valid_heads__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(HEADS - head__idx_v0, 8)
            t__tile_4: pl.Tile[[8, 512], pl.BF16, pl.MemRef(mem_vec_13, pl.const(16512, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[valid_heads__ssa_v0, 512])] = pl.tile.load(
                flat_input__ssa_v0, [row__ssa_v0, 0], [8, 512], [valid_heads__ssa_v0, 512], target_memory=pl.Mem.Vec
            )
            x__tile: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_12, pl.const(128, pl.INT64), 16384), pl.Mem.Vec, pl.TileView(valid_shape=[valid_heads__ssa_v0, 512])] = pl.tile.cast(
                t__tile_4, target_type=pl.FP32, mode="round"
            )
            t__tile_5: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_13, pl.const(16512, pl.INT64), 16384), pl.Mem.Vec, pl.TileView(valid_shape=[valid_heads__ssa_v0, 512])] = pl.tile.mul(
                x__tile, x__tile
            )
            tmp_tile: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_14, pl.const(32896, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.create([8, 512], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            t__tile_6: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_27, pl.const(49280, pl.INT64), 32), pl.Mem.Vec, pl.TileView(valid_shape=[valid_heads__ssa_v0, 1])] = pl.tile.row_sum(
                t__tile_5, tmp_tile
            )
            t__tile_7: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_27, pl.const(49280, pl.INT64), 32), pl.Mem.Vec, pl.TileView(valid_shape=[1, valid_heads__ssa_v0])] = pl.tile.reshape(
                t__tile_6, [1, 8]
            )
            variance__tile: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_13, pl.const(16512, pl.INT64), 32), pl.Mem.Vec, pl.TileView(valid_shape=[1, valid_heads__ssa_v0])] = pl.tile.muls(
                t__tile_7, 0.001953125
            )
            t__tile_8: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_13, pl.const(16512, pl.INT64), 32), pl.Mem.Vec, pl.TileView(valid_shape=[1, valid_heads__ssa_v0])] = pl.tile.adds(
                variance__tile, 9.9999999999999995e-07
            )
            rsqrt_tmp: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_14, pl.const(32896, pl.INT64), 32), pl.Mem.Vec] = pl.tile.create([1, 8], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            t__tile_9: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_27, pl.const(49280, pl.INT64), 32), pl.Mem.Vec, pl.TileView(valid_shape=[1, valid_heads__ssa_v0])] = pl.tile.rsqrt(
                t__tile_8, rsqrt_tmp
            )
            inv_rms__tile: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_27, pl.const(49280, pl.INT64), 32), pl.Mem.Vec, pl.TileView(valid_shape=[valid_heads__ssa_v0, 1])] = pl.tile.reshape(
                t__tile_9, [8, 1]
            )
            t__tile_10: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_12, pl.const(128, pl.INT64), 16384), pl.Mem.Vec, pl.TileView(valid_shape=[valid_heads__ssa_v0, 512])] = pl.tile.row_expand_mul(
                x__tile, inv_rms__tile
            )
            normed__tile: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_12, pl.const(128, pl.INT64), 16384), pl.Mem.Vec, pl.TileView(valid_shape=[valid_heads__ssa_v0, 512])] = pl.tile.col_expand_mul(
                t__tile_10, g__tile
            )
            normalized__tile: pl.Tile[[8, 512], pl.BF16, pl.MemRef(mem_vec_12, pl.const(128, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[valid_heads__ssa_v0, 512])] = pl.tile.cast(
                normed__tile, target_type=pl.BF16, mode="rint"
            )
            t__tile_11: pl.Tile[[8, 448], pl.BF16, pl.MemRef(mem_vec_12, pl.const(128, pl.INT64), 8064), pl.Mem.Vec, pl.TileView(valid_shape=[valid_heads__ssa_v0, 448])] = pl.tile.slice(
                normalized__tile, [8, 448], [0, 0]
            )
            flat_output__tile: pl.Tensor[[rows__ssa_v0, 512], pl.BF16, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)] = pl.tile.store(t__tile_11, [row__ssa_v0, 0], flat_output__iter_v1)
            t__tile_12: pl.Tile[[8, 64], pl.BF16, pl.MemRef(mem_vec_12, pl.const(1024, pl.INT64), 7296), pl.Mem.Vec, pl.TileView(valid_shape=[valid_heads__ssa_v0, 64])] = pl.tile.slice(
                normalized__tile, [8, 64], [0, 448]
            )
            rope__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(16512, pl.INT64), 2048), pl.Mem.Vec, pl.TileView(valid_shape=[valid_heads__ssa_v0, 64])] = pl.tile.cast(
                t__tile_12, target_type=pl.FP32, mode="round"
            )
            even__tile: pl.Tile[[8, 32], pl.FP32, pl.MemRef(mem_vec_12, pl.const(128, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.gather_mask(rope__tile, mask_pattern=1)
            odd__tile: pl.Tile[[8, 32], pl.FP32, pl.MemRef(mem_vec_14, pl.const(32896, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.gather_mask(rope__tile, mask_pattern=2)
            t__tile_13: pl.Tile[[8, 32], pl.FP32, pl.MemRef(mem_vec_13, pl.const(16512, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.col_expand_mul(even__tile, cos_even__tile)
            t__tile_14: pl.Tile[[8, 32], pl.FP32, pl.MemRef(mem_vec_27, pl.const(49280, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.col_expand_mul(odd__tile, sin_even__tile)
            rotated_even__tile: pl.Tile[[8, 32], pl.FP32, pl.MemRef(mem_vec_27, pl.const(49280, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.sub(t__tile_13, t__tile_14)
            t__tile_15: pl.Tile[[8, 32], pl.FP32, pl.MemRef(mem_vec_12, pl.const(128, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.col_expand_mul(even__tile, sin_even__tile)
            t__tile_16: pl.Tile[[8, 32], pl.FP32, pl.MemRef(mem_vec_13, pl.const(16512, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.col_expand_mul(odd__tile, cos_even__tile)
            rotated_odd__tile: pl.Tile[[8, 32], pl.FP32, pl.MemRef(mem_vec_31, pl.const(50304, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.add(t__tile_15, t__tile_16)
            rotated__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(128, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.full([8, 64], dtype=pl.FP32, value=0.0)
            rotated_v1__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(128, pl.INT64), 2048), pl.Mem.Vec, pl.TileView(valid_shape=[valid_heads__ssa_v0, 64])] = pl.tile.set_validshape(
                rotated__tile, valid_heads__ssa_v0, 64
            )
            scatter_mask_values_zero: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(16512, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.full([8, 64], dtype=pl.FP32, value=0.0)
            scatter_mask_values: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(16512, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.scatter_mask(
                scatter_mask_values_zero, rotated_even__tile, mask_pattern=1
            )
            scatter_mask_mask_zero: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_14, pl.const(32896, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.full([8, 64], dtype=pl.FP32, value=0.0)
            scatter_mask_ones: pl.Tile[[8, 32], pl.FP32, pl.MemRef(mem_vec_27, pl.const(49280, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.full([8, 32], dtype=pl.FP32, value=1.0)
            scatter_mask_mask: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_14, pl.const(32896, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.scatter_mask(
                scatter_mask_mask_zero, scatter_mask_ones, mask_pattern=1
            )
            scatter_mask_pred: pl.Tile[[8, 32], pl.UINT8, pl.MemRef(mem_vec_14, pl.const(32896, pl.INT64), 256), pl.Mem.Vec, pl.TileView(valid_shape=[8, 8])] = pl.tile.cmps(
                scatter_mask_mask, 0.0, cmp_type=1
            )
            scatter_mask_sel_tmp: pl.Tile[[1, 16], pl.UINT32, pl.MemRef(mem_vec_27, pl.const(49280, pl.INT64), 64), pl.Mem.Vec] = pl.tile.create([1, 16], dtype=pl.UINT32, target_memory=pl.Mem.Vec)
            rotated_v2__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(128, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.sel(
                scatter_mask_pred, scatter_mask_values, rotated_v1__tile, scatter_mask_sel_tmp
            )
            scatter_mask_values_zero_1: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(16512, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.full([8, 64], dtype=pl.FP32, value=0.0)
            scatter_mask_values_1: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(16512, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.scatter_mask(
                scatter_mask_values_zero_1, rotated_odd__tile, mask_pattern=2
            )
            scatter_mask_mask_zero_1: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_14, pl.const(32896, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.full([8, 64], dtype=pl.FP32, value=0.0)
            scatter_mask_ones_1: pl.Tile[[8, 32], pl.FP32, pl.MemRef(mem_vec_27, pl.const(49280, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.full([8, 32], dtype=pl.FP32, value=1.0)
            scatter_mask_mask_1: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_14, pl.const(32896, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.scatter_mask(
                scatter_mask_mask_zero_1, scatter_mask_ones_1, mask_pattern=2
            )
            scatter_mask_pred_1: pl.Tile[[8, 32], pl.UINT8, pl.MemRef(mem_vec_14, pl.const(32896, pl.INT64), 256), pl.Mem.Vec, pl.TileView(valid_shape=[8, 8])] = pl.tile.cmps(
                scatter_mask_mask_1, 0.0, cmp_type=1
            )
            scatter_mask_sel_tmp_1: pl.Tile[[1, 16], pl.UINT32, pl.MemRef(mem_vec_27, pl.const(49280, pl.INT64), 64), pl.Mem.Vec] = pl.tile.create([1, 16], dtype=pl.UINT32, target_memory=pl.Mem.Vec)
            rotated_v3__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(128, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.sel(
                scatter_mask_pred_1, scatter_mask_values_1, rotated_v2__tile, scatter_mask_sel_tmp_1
            )
            rounded__tile: pl.Tile[[8, 64], pl.BF16, pl.MemRef(mem_vec_12, pl.const(128, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.cast(rotated_v3__tile, target_type=pl.BF16, mode="rint")
            flat_output__tile_1: pl.Tensor[[rows__ssa_v0, 512], pl.BF16, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)] = pl.tile.store(rounded__tile, [row__ssa_v0, 448], flat_output__tile)
            flat_output__rv_v2: pl.Tensor[[rows__ssa_v0, 512], pl.BF16, pl.MemRef("mem_ddr_46", pl.const(0, pl.INT64), 0)] = pl.yield_(flat_output__tile_1)
        return

    @pl.function(type=pl.FunctionType.Spmd)
    def csa_norm_rope_spmd(
        self,
        gamma__ssa_v0: pl.Tensor[[512], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 1024)],
        cos__ssa_v0: pl.Tensor[[TOKENS, 64], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)],
        sin__ssa_v0: pl.Tensor[[TOKENS, 64], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)],
        flat_output__ssa_v0: pl.Out[pl.Tensor[[rows__ssa_v0, 512], pl.BF16, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)]],
        HEADS: pl.Scalar[pl.INDEX],
        flat_input__ssa_v0: pl.Tensor[[rows__ssa_v0, 512], pl.BF16, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 0)],
    ):
        pl.func_attr({"mx_tensor_views_blocked": True})
        self.csa_norm_rope(
            gamma__ssa_v0,
            cos__ssa_v0,
            sin__ssa_v0,
            flat_output__ssa_v0,
            HEADS,
            flat_input__ssa_v0,
            attrs={"arg_directions": [pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.output_existing, pl.adir.scalar, pl.adir.input]},
        )

    @pl.function(type=pl.FunctionType.Orchestration, level=pl.Level.CHIP, role=pl.Role.Orchestrator, auto_scope=False)
    def norm_rope(
        self,
        projected__ssa_v0: pl.Tensor[[TOKENS, HEADS, 512], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)],
        gamma__ssa_v0: pl.Tensor[[512], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 1024)],
        cos__ssa_v0: pl.Tensor[[TOKENS, 64], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)],
        sin__ssa_v0: pl.Tensor[[TOKENS, 64], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)],
        output__ssa_v0: pl.Out[pl.Tensor[[TOKENS, HEADS, 512], pl.BF16, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 0)]],
    ) -> pl.Tensor[[TOKENS, HEADS, 512], pl.BF16]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        with pl.scope():
            rows__ssa_v0: pl.Scalar[pl.INDEX] = TOKENS * HEADS
            flat_input__ssa_v0: pl.Tensor[[rows__ssa_v0, 512], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)] = pl.tensor.reshape(projected__ssa_v0, [rows__ssa_v0, 512])
            flat_output__ssa_v0: pl.Tensor[[rows__ssa_v0, 512], pl.BF16, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 0)] = pl.tensor.reshape(output__ssa_v0, [rows__ssa_v0, 512])
            self.csa_norm_rope_spmd(
                gamma__ssa_v0,
                cos__ssa_v0,
                sin__ssa_v0,
                flat_output__ssa_v0,
                HEADS,
                flat_input__ssa_v0,
                attrs={"arg_directions": [pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.output_existing, pl.adir.scalar, pl.adir.input], "core_num": TOKENS},
            )
            return output__ssa_v0