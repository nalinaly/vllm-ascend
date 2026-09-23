# pypto.program: _jit_prepare_rope
import pypto.language as pl

CSA_NATIVE_TOKENS = pl.dynamic("CSA_NATIVE_TOKENS")
TOKENS = pl.dynamic("TOKENS")


@pl.program
class _jit_prepare_rope:
    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def csa_native_rope(
        native_cos__ssa_v0: pl.Tensor[[TOKENS, 64], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)],
        half_cos__ssa_v0: pl.Out[pl.Tensor[[CSA_NATIVE_TOKENS, 64], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)]],
        native_sin__ssa_v0: pl.Tensor[[CSA_NATIVE_TOKENS, 64], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)],
        half_sin__ssa_v0: pl.Out[pl.Tensor[[CSA_NATIVE_TOKENS, 64], pl.BF16, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)]],
    ):
        pl.func_attr({"mx_tensor_views_blocked": True})
        mem_vec_4: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 768)
        mem_vec_5: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        mem_vec_15: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        mem_vec_16: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        mem_vec_17: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        token__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        t__ci_tmp_v0: pl.Tile[[1, 192], pl.FP32, pl.MemRef(mem_vec_4, pl.const(768, pl.INT64), 768), pl.Mem.Vec] = pl.tile.create([1, 192], dtype=pl.FP32, target_memory=pl.Mem.Vec)
        t__tile: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_5, pl.const(1536, pl.INT64), 256), pl.Mem.Vec] = pl.tile.ci(
            pl.const(0, pl.INT32), [1, 64], tmp=t__ci_tmp_v0, dtype=pl.INT32, descending=False
        )
        columns__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_4, pl.const(768, pl.INT64), 256), pl.Mem.Vec] = pl.tile.cast(t__tile, target_type=pl.FP32, mode="round")
        t__tile_1: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_5, pl.const(1536, pl.INT64), 256), pl.Mem.Vec] = pl.tile.muls(columns__tile, 0.03125)
        t__tile_2: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_5, pl.const(1536, pl.INT64), 256), pl.Mem.Vec] = pl.tile.cast(t__tile_1, target_type=pl.INT32, mode="trunc")
        half__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_5, pl.const(1536, pl.INT64), 256), pl.Mem.Vec] = pl.tile.cast(t__tile_2, target_type=pl.FP32, mode="round")
        t__tile_3: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_5, pl.const(1536, pl.INT64), 256), pl.Mem.Vec] = pl.tile.muls(half__tile, 32.0)
        t__tile_4: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_4, pl.const(768, pl.INT64), 256), pl.Mem.Vec] = pl.tile.sub(columns__tile, t__tile_3)
        t__tile_5: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_4, pl.const(768, pl.INT64), 256), pl.Mem.Vec] = pl.tile.muls(t__tile_4, 2.0)
        source_columns__tile: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_4, pl.const(768, pl.INT64), 256), pl.Mem.Vec] = pl.tile.cast(t__tile_5, target_type=pl.INT32, mode="round")
        t__tile_6: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_5, pl.const(1536, pl.INT64), 256), pl.Mem.Vec] = pl.tile.load(
            native_cos__ssa_v0, [token__ssa_v0, 0], [1, 64], [1, 64], target_memory=pl.Mem.Vec
        )
        gather_acc_init: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_15, pl.const(0, pl.INT64), 256), pl.Mem.Vec] = pl.tile.create([1, 64], dtype=pl.FP32, target_memory=pl.Mem.Vec)
        gather_inp_row: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_5, pl.const(1536, pl.INT64), 256), pl.Mem.Vec] = pl.tile.slice(t__tile_6, [1, 64], [0, 0], [1, 64])
        gather_idx_row: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_4, pl.const(768, pl.INT64), 256), pl.Mem.Vec] = pl.tile.slice(source_columns__tile, [1, 64], [0, 0], [1, 64])
        gather_row_tmp: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_16, pl.const(256, pl.INT64), 256), pl.Mem.Vec] = pl.tile.create([1, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec)
        gather_row: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_17, pl.const(512, pl.INT64), 256), pl.Mem.Vec] = pl.tile.gather(gather_inp_row, gather_idx_row, gather_row_tmp)
        gather_asmbl: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_15, pl.const(0, pl.INT64), 256), pl.Mem.Vec] = pl.tile.assemble(gather_acc_init, gather_row, [0, 0])
        t__tile_7: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_15, pl.const(0, pl.INT64), 256), pl.Mem.Vec] = gather_asmbl
        t__tile_8: pl.Tile[[1, 64], pl.BF16, pl.MemRef(mem_vec_5, pl.const(1536, pl.INT64), 128), pl.Mem.Vec] = pl.tile.cast(t__tile_7, target_type=pl.BF16, mode="round")
        half_cos__tile: pl.Tensor[[CSA_NATIVE_TOKENS, 64], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)] = pl.tile.store(t__tile_8, [token__ssa_v0, 0], half_cos__ssa_v0)
        t__tile_9: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_5, pl.const(1536, pl.INT64), 256), pl.Mem.Vec] = pl.tile.load(
            native_sin__ssa_v0, [token__ssa_v0, 0], [1, 64], [1, 64], target_memory=pl.Mem.Vec
        )
        gather_acc_init_1: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_15, pl.const(0, pl.INT64), 256), pl.Mem.Vec] = pl.tile.create([1, 64], dtype=pl.FP32, target_memory=pl.Mem.Vec)
        gather_inp_row_1: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_5, pl.const(1536, pl.INT64), 256), pl.Mem.Vec] = pl.tile.slice(t__tile_9, [1, 64], [0, 0], [1, 64])
        gather_idx_row_1: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_4, pl.const(768, pl.INT64), 256), pl.Mem.Vec] = pl.tile.slice(source_columns__tile, [1, 64], [0, 0], [1, 64])
        gather_row_tmp_1: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_16, pl.const(256, pl.INT64), 256), pl.Mem.Vec] = pl.tile.create([1, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec)
        gather_row_1: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_17, pl.const(512, pl.INT64), 256), pl.Mem.Vec] = pl.tile.gather(gather_inp_row_1, gather_idx_row_1, gather_row_tmp_1)
        gather_asmbl_1: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_15, pl.const(0, pl.INT64), 256), pl.Mem.Vec] = pl.tile.assemble(gather_acc_init_1, gather_row_1, [0, 0])
        t__tile_10: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_15, pl.const(0, pl.INT64), 256), pl.Mem.Vec] = gather_asmbl_1
        t__tile_11: pl.Tile[[1, 64], pl.BF16, pl.MemRef(mem_vec_4, pl.const(768, pl.INT64), 128), pl.Mem.Vec] = pl.tile.cast(t__tile_10, target_type=pl.BF16, mode="round")
        half_sin__tile: pl.Tensor[[CSA_NATIVE_TOKENS, 64], pl.BF16, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)] = pl.tile.store(t__tile_11, [token__ssa_v0, 0], half_sin__ssa_v0)
        return

    @pl.function(type=pl.FunctionType.Spmd)
    def csa_native_rope_spmd(
        self,
        native_cos__ssa_v0: pl.Tensor[[TOKENS, 64], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)],
        half_cos__ssa_v0: pl.Out[pl.Tensor[[CSA_NATIVE_TOKENS, 64], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)]],
        native_sin__ssa_v0: pl.Tensor[[CSA_NATIVE_TOKENS, 64], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)],
        half_sin__ssa_v0: pl.Out[pl.Tensor[[CSA_NATIVE_TOKENS, 64], pl.BF16, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)]],
    ):
        pl.func_attr({"mx_tensor_views_blocked": True})
        self.csa_native_rope(
            native_cos__ssa_v0, half_cos__ssa_v0, native_sin__ssa_v0, half_sin__ssa_v0, attrs={"arg_directions": [pl.adir.input, pl.adir.output_existing, pl.adir.input, pl.adir.output_existing]}
        )

    @pl.function(type=pl.FunctionType.Orchestration, level=pl.Level.CHIP, role=pl.Role.Orchestrator, auto_scope=False)
    def prepare_rope(
        self,
        native_cos__ssa_v0: pl.Tensor[[TOKENS, 64], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)],
        native_sin__ssa_v0: pl.Tensor[[CSA_NATIVE_TOKENS, 64], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)],
        half_cos__ssa_v0: pl.Out[pl.Tensor[[CSA_NATIVE_TOKENS, 64], pl.BF16, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)]],
        half_sin__ssa_v0: pl.Out[pl.Tensor[[CSA_NATIVE_TOKENS, 64], pl.BF16, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)]],
    ) -> pl.Tensor[[CSA_NATIVE_TOKENS, 64], pl.BF16]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        self.csa_native_rope_spmd(
            native_cos__ssa_v0,
            half_cos__ssa_v0,
            native_sin__ssa_v0,
            half_sin__ssa_v0,
            attrs={"arg_directions": [pl.adir.input, pl.adir.output_existing, pl.adir.input, pl.adir.output_existing], "core_num": TOKENS},
        )