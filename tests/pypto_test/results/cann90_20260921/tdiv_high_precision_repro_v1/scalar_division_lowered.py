# pypto.program: _jit_scalar_division
import pypto.language as pl


@pl.program
class _jit_scalar_division:
    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def scalar_division_incore_0(
        numerator__ssa_v0: pl.Tensor[[16, 256], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 16384)],
        denominator__ssa_v0: pl.Tensor[[16, 256], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 16384)],
        result__ssa_v0: pl.Out[pl.Tensor[[16, 256], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 16384)]],
    ) -> pl.Tensor[[16, 256], pl.FP32]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        mem_vec_3: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 1024)
        row__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        values__tile: pl.Tile[[1, 256], pl.FP32, pl.MemRef(mem_vec_3, pl.const(0, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.full([1, 256], dtype=pl.FP32, value=0.0)
        for column__idx_v0 in pl.range(256):
            left__tile: pl.Scalar[pl.FP32] = pl.tensor.read(numerator__ssa_v0, [row__ssa_v0, column__idx_v0])
            right__tile: pl.Scalar[pl.FP32] = pl.tensor.read(denominator__ssa_v0, [row__ssa_v0, column__idx_v0])
            pl.tile.write(values__tile, [0, column__idx_v0], left__tile / right__tile)
        result__tile: pl.Tensor[[16, 256], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 16384)] = pl.tile.store(values__tile, [row__ssa_v0, 0], result__ssa_v0)
        return result__ssa_v0

    @pl.function(type=pl.FunctionType.Spmd)
    def scalar_division_spmd_0(
        self,
        numerator__ssa_v0: pl.Tensor[[16, 256], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 16384)],
        denominator__ssa_v0: pl.Tensor[[16, 256], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 16384)],
        result__ssa_v0: pl.Out[pl.Tensor[[16, 256], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 16384)]],
    ) -> pl.Tensor[[16, 256], pl.FP32]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        result__ssa_v1: pl.Tensor[[16, 256], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 16384)] = self.scalar_division_incore_0(
            numerator__ssa_v0, denominator__ssa_v0, result__ssa_v0, attrs={"arg_directions": [pl.adir.input, pl.adir.input, pl.adir.output_existing]}
        )
        return result__ssa_v0

    @pl.function(type=pl.FunctionType.Orchestration, level=pl.Level.CHIP, role=pl.Role.Orchestrator, auto_scope=False)
    def scalar_division(
        self,
        numerator__ssa_v0: pl.Tensor[[16, 256], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 16384)],
        denominator__ssa_v0: pl.Tensor[[16, 256], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 16384)],
        result__ssa_v0: pl.Out[pl.Tensor[[16, 256], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 16384)]],
    ) -> pl.Tensor[[16, 256], pl.FP32]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        result__ssa_v1: pl.Tensor[[16, 256], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 16384)] = self.scalar_division_spmd_0(
            numerator__ssa_v0, denominator__ssa_v0, result__ssa_v0, attrs={"arg_directions": [pl.adir.input, pl.adir.input, pl.adir.output_existing], "core_num": 16}
        )
        return result__ssa_v1