# pypto.program: _jit_vector_division
import pypto.language as pl


@pl.program
class _jit_vector_division:
    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def vector_division_incore_0(
        numerator__ssa_v0: pl.Tensor[[16, 256], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 16384)],
        denominator__ssa_v0: pl.Tensor[[16, 256], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 16384)],
        normal__ssa_v0: pl.Out[pl.Tensor[[16, 256], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 16384)]],
        precise__ssa_v0: pl.Out[pl.Tensor[[16, 256], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 16384)]],
    ) -> tuple[pl.Tensor[[16, 256], pl.FP32], pl.Tensor[[16, 256], pl.FP32]]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        mem_vec_4: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 1024)
        mem_vec_5: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 1024)
        mem_vec_6: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 1024)
        row__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        left__tile: pl.Tile[[1, 256], pl.FP32, pl.MemRef(mem_vec_4, pl.const(0, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.load(
            numerator__ssa_v0, [row__ssa_v0, 0], [1, 256], [1, 256], target_memory=pl.Mem.Vec
        )
        right__tile: pl.Tile[[1, 256], pl.FP32, pl.MemRef(mem_vec_5, pl.const(1024, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.load(
            denominator__ssa_v0, [row__ssa_v0, 0], [1, 256], [1, 256], target_memory=pl.Mem.Vec
        )
        t__tile: pl.Tile[[1, 256], pl.FP32, pl.MemRef(mem_vec_6, pl.const(2048, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.div(left__tile, right__tile)
        normal__tile: pl.Tensor[[16, 256], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 16384)] = pl.tile.store(t__tile, [row__ssa_v0, 0], normal__ssa_v0)
        t__tile_1: pl.Tile[[1, 256], pl.FP32, pl.MemRef(mem_vec_4, pl.const(0, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.div(left__tile, right__tile, high_precision=True)
        precise__tile: pl.Tensor[[16, 256], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 16384)] = pl.tile.store(t__tile_1, [row__ssa_v0, 0], precise__ssa_v0)
        return normal__ssa_v0, precise__ssa_v0

    @pl.function(type=pl.FunctionType.Spmd)
    def vector_division_spmd_0(
        self,
        numerator__ssa_v0: pl.Tensor[[16, 256], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 16384)],
        denominator__ssa_v0: pl.Tensor[[16, 256], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 16384)],
        normal__ssa_v0: pl.Out[pl.Tensor[[16, 256], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 16384)]],
        precise__ssa_v0: pl.Out[pl.Tensor[[16, 256], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 16384)]],
    ) -> tuple[pl.Tensor[[16, 256], pl.FP32], pl.Tensor[[16, 256], pl.FP32]]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        ret__tmp_v0: pl.Tuple[pl.Tensor[[16, 256], pl.FP32], pl.Tensor[[16, 256], pl.FP32]] = self.vector_division_incore_0(
            numerator__ssa_v0, denominator__ssa_v0, normal__ssa_v0, precise__ssa_v0, attrs={"arg_directions": [pl.adir.input, pl.adir.input, pl.adir.output_existing, pl.adir.output_existing]}
        )
        normal__ssa_v1: pl.Tensor[[16, 256], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 16384)] = ret__tmp_v0[0]
        precise__ssa_v1: pl.Tensor[[16, 256], pl.FP32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 16384)] = ret__tmp_v0[1]
        return normal__ssa_v0, precise__ssa_v0

    @pl.function(type=pl.FunctionType.Orchestration, level=pl.Level.CHIP, role=pl.Role.Orchestrator, auto_scope=False)
    def vector_division(
        self,
        numerator__ssa_v0: pl.Tensor[[16, 256], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 16384)],
        denominator__ssa_v0: pl.Tensor[[16, 256], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 16384)],
        normal__ssa_v0: pl.Out[pl.Tensor[[16, 256], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 16384)]],
        precise__ssa_v0: pl.Out[pl.Tensor[[16, 256], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 16384)]],
    ) -> tuple[pl.Tensor[[16, 256], pl.FP32], pl.Tensor[[16, 256], pl.FP32]]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        ret__tmp_v0: pl.Tuple[pl.Tensor[[16, 256], pl.FP32], pl.Tensor[[16, 256], pl.FP32]] = self.vector_division_spmd_0(
            numerator__ssa_v0,
            denominator__ssa_v0,
            normal__ssa_v0,
            precise__ssa_v0,
            attrs={"arg_directions": [pl.adir.input, pl.adir.input, pl.adir.output_existing, pl.adir.output_existing], "core_num": 16},
        )
        normal__ssa_v1: pl.Tensor[[16, 256], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 16384)] = ret__tmp_v0[0]
        precise__ssa_v1: pl.Tensor[[16, 256], pl.FP32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 16384)] = ret__tmp_v0[1]
        return normal__ssa_v1, precise__ssa_v1