# pypto.program: _jit_prepare_compressed_metadata
import pypto.language as pl

BOUNDS = pl.dynamic("BOUNDS")
COMPACT_ROWS = pl.dynamic("COMPACT_ROWS")
CSA_NATIVE_COMPACT_ROWS = pl.dynamic("CSA_NATIVE_COMPACT_ROWS")
CSA_NATIVE_TOKENS = pl.dynamic("CSA_NATIVE_TOKENS")
REQUESTS = pl.dynamic("REQUESTS")
TOKENS = pl.dynamic("TOKENS")


@pl.program
class _jit_prepare_compressed_metadata:
    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def csa_expand_compact_rope(
        compact_indices__ssa_v0: pl.Tensor[[TOKENS], pl.INT32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)],
        compact_cos__ssa_v0: pl.Tensor[[CSA_NATIVE_COMPACT_ROWS, 64], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)],
        compact_sin__ssa_v0: pl.Tensor[[CSA_NATIVE_COMPACT_ROWS, 64], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)],
        half_cos__ssa_v0: pl.Out[pl.Tensor[[CSA_NATIVE_TOKENS, 64], pl.BF16, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)]],
        half_sin__ssa_v0: pl.Out[pl.Tensor[[CSA_NATIVE_TOKENS, 64], pl.BF16, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 0)]],
    ):
        pl.func_attr({"mx_tensor_views_blocked": True})
        mem_vec_5: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 768)
        mem_vec_6: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        mem_vec_15: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 128)
        mem_vec_16: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 128)
        mem_vec_18: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        mem_vec_19: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        mem_vec_20: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        mem_vec_22: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 128)
        rope_token__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        t__ci_tmp_v0: pl.Tile[[1, 192], pl.FP32, pl.MemRef(mem_vec_5, pl.const(1152, pl.INT64), 768), pl.Mem.Vec] = pl.tile.create([1, 192], dtype=pl.FP32, target_memory=pl.Mem.Vec)
        t__tile: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_6, pl.const(1920, pl.INT64), 256), pl.Mem.Vec] = pl.tile.ci(
            pl.const(0, pl.INT32), [1, 64], tmp=t__ci_tmp_v0, dtype=pl.INT32, descending=False
        )
        columns__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_5, pl.const(1152, pl.INT64), 256), pl.Mem.Vec] = pl.tile.cast(t__tile, target_type=pl.FP32, mode="round")
        t__tile_1: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_6, pl.const(1920, pl.INT64), 256), pl.Mem.Vec] = pl.tile.muls(columns__tile, 0.03125)
        t__tile_2: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_6, pl.const(1920, pl.INT64), 256), pl.Mem.Vec] = pl.tile.cast(t__tile_1, target_type=pl.INT32, mode="trunc")
        half__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_6, pl.const(1920, pl.INT64), 256), pl.Mem.Vec] = pl.tile.cast(t__tile_2, target_type=pl.FP32, mode="round")
        t__tile_3: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_6, pl.const(1920, pl.INT64), 256), pl.Mem.Vec] = pl.tile.muls(half__tile, 32.0)
        t__tile_4: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_5, pl.const(1152, pl.INT64), 256), pl.Mem.Vec] = pl.tile.sub(columns__tile, t__tile_3)
        t__tile_5: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_5, pl.const(1152, pl.INT64), 256), pl.Mem.Vec] = pl.tile.muls(t__tile_4, 2.0)
        source_columns__tile: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_5, pl.const(1152, pl.INT64), 256), pl.Mem.Vec] = pl.tile.cast(t__tile_5, target_type=pl.INT32, mode="round")
        compact_row__tile: pl.Scalar[pl.INT32] = pl.tensor.read(compact_indices__ssa_v0, [rope_token__ssa_v0])
        cosine__tile: pl.Tile[[1, 64], pl.BF16, pl.MemRef(mem_vec_15, pl.const(0, pl.INT64), 128), pl.Mem.Vec] = pl.tile.full([1, 64], dtype=pl.BF16, value=1.0)
        sine__tile: pl.Tile[[1, 64], pl.BF16, pl.MemRef(mem_vec_16, pl.const(128, pl.INT64), 128), pl.Mem.Vec] = pl.tile.full([1, 64], dtype=pl.BF16, value=0.0)
        if 0 <= pl.cast(compact_row__tile, pl.INDEX):
            t__tile_6: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_6, pl.const(1920, pl.INT64), 256), pl.Mem.Vec] = pl.tile.load(
                compact_cos__ssa_v0, [compact_row__tile, 0], [1, 64], [1, 64], target_memory=pl.Mem.Vec
            )
            gather_acc_init: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_18, pl.const(256, pl.INT64), 256), pl.Mem.Vec] = pl.tile.create([1, 64], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            gather_inp_row: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_6, pl.const(1920, pl.INT64), 256), pl.Mem.Vec] = pl.tile.slice(t__tile_6, [1, 64], [0, 0], [1, 64])
            gather_idx_row: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_5, pl.const(1152, pl.INT64), 256), pl.Mem.Vec] = pl.tile.slice(source_columns__tile, [1, 64], [0, 0], [1, 64])
            gather_row_tmp: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_19, pl.const(512, pl.INT64), 256), pl.Mem.Vec] = pl.tile.create([1, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec)
            gather_row: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_20, pl.const(768, pl.INT64), 256), pl.Mem.Vec] = pl.tile.gather(gather_inp_row, gather_idx_row, gather_row_tmp)
            gather_asmbl: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_18, pl.const(256, pl.INT64), 256), pl.Mem.Vec] = pl.tile.assemble(gather_acc_init, gather_row, [0, 0])
            t__tile_7: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_18, pl.const(256, pl.INT64), 256), pl.Mem.Vec] = gather_asmbl
            cosine__tile_1: pl.Tile[[1, 64], pl.BF16, pl.MemRef(mem_vec_22, pl.const(1024, pl.INT64), 128), pl.Mem.Vec] = pl.tile.cast(t__tile_7, target_type=pl.BF16, mode="round")
            t__tile_8: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_6, pl.const(1920, pl.INT64), 256), pl.Mem.Vec] = pl.tile.load(
                compact_sin__ssa_v0, [compact_row__tile, 0], [1, 64], [1, 64], target_memory=pl.Mem.Vec
            )
            gather_acc_init_1: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_18, pl.const(256, pl.INT64), 256), pl.Mem.Vec] = pl.tile.create([1, 64], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            gather_inp_row_1: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_6, pl.const(1920, pl.INT64), 256), pl.Mem.Vec] = pl.tile.slice(t__tile_8, [1, 64], [0, 0], [1, 64])
            gather_idx_row_1: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_5, pl.const(1152, pl.INT64), 256), pl.Mem.Vec] = pl.tile.slice(source_columns__tile, [1, 64], [0, 0], [1, 64])
            gather_row_tmp_1: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_19, pl.const(512, pl.INT64), 256), pl.Mem.Vec] = pl.tile.create([1, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec)
            gather_row_1: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_20, pl.const(768, pl.INT64), 256), pl.Mem.Vec] = pl.tile.gather(gather_inp_row_1, gather_idx_row_1, gather_row_tmp_1)
            gather_asmbl_1: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_18, pl.const(256, pl.INT64), 256), pl.Mem.Vec] = pl.tile.assemble(gather_acc_init_1, gather_row_1, [0, 0])
            t__tile_9: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_18, pl.const(256, pl.INT64), 256), pl.Mem.Vec] = gather_asmbl_1
            sine__tile_1: pl.Tile[[1, 64], pl.BF16, pl.MemRef(mem_vec_5, pl.const(1152, pl.INT64), 128), pl.Mem.Vec] = pl.tile.cast(t__tile_9, target_type=pl.BF16, mode="round")
            cosine__phi_v2, sine__phi_v2 = pl.yield_(cosine__tile_1, sine__tile_1)
        else:
            cosine__tile_mv: pl.Tile[[1, 64], pl.BF16, pl.MemRef(mem_vec_22, pl.const(1024, pl.INT64), 128), pl.Mem.Vec] = pl.tile.move(cosine__tile, target_memory=pl.Mem.Vec)
            sine__tile_mv: pl.Tile[[1, 64], pl.BF16, pl.MemRef(mem_vec_5, pl.const(1152, pl.INT64), 128), pl.Mem.Vec] = pl.tile.move(sine__tile, target_memory=pl.Mem.Vec)
            cosine__phi_v2, sine__phi_v2 = pl.yield_(cosine__tile_mv, sine__tile_mv)
        half_cos__tile: pl.Tensor[[CSA_NATIVE_TOKENS, 64], pl.BF16, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)] = pl.tile.store(cosine__phi_v2, [rope_token__ssa_v0, 0], half_cos__ssa_v0)
        half_sin__tile: pl.Tensor[[CSA_NATIVE_TOKENS, 64], pl.BF16, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 0)] = pl.tile.store(sine__phi_v2, [rope_token__ssa_v0, 0], half_sin__ssa_v0)
        return

    @pl.function(type=pl.FunctionType.Spmd)
    def csa_expand_compact_rope_spmd(
        self,
        compact_indices__ssa_v0: pl.Tensor[[TOKENS], pl.INT32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)],
        compact_cos__ssa_v0: pl.Tensor[[CSA_NATIVE_COMPACT_ROWS, 64], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)],
        compact_sin__ssa_v0: pl.Tensor[[CSA_NATIVE_COMPACT_ROWS, 64], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)],
        half_cos__ssa_v0: pl.Out[pl.Tensor[[CSA_NATIVE_TOKENS, 64], pl.BF16, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)]],
        half_sin__ssa_v0: pl.Out[pl.Tensor[[CSA_NATIVE_TOKENS, 64], pl.BF16, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 0)]],
    ):
        pl.func_attr({"mx_tensor_views_blocked": True})
        self.csa_expand_compact_rope(
            compact_indices__ssa_v0,
            compact_cos__ssa_v0,
            compact_sin__ssa_v0,
            half_cos__ssa_v0,
            half_sin__ssa_v0,
            attrs={"arg_directions": [pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.output_existing, pl.adir.output_existing]},
        )

    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def csa_native_compact_metadata(
        REQUESTS: pl.Scalar[pl.INDEX],
        bounds__ssa_v0: pl.Tensor[[BOUNDS], pl.INT32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)],
        lengths__ssa_v0: pl.Tensor[[REQUESTS], pl.INT32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)],
        expanded_slots__ssa_v0: pl.Out[pl.Tensor[[CSA_NATIVE_TOKENS], pl.INT64, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)]],
        compact_indices__ssa_v0: pl.Out[pl.Tensor[[TOKENS], pl.INT32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)]],
        positions__ssa_v0: pl.Tensor[[TOKENS], pl.INT64, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 0)],
        compact_slots__ssa_v0: pl.Tensor[[COMPACT_ROWS, 2], pl.INT32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 0)],
    ):
        pl.func_attr({"mx_tensor_views_blocked": True})
        prefix__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(0, pl.INDEX)
        for request__idx_v0, (prefix__iter_v1,) in pl.range(REQUESTS, init_values=(prefix__ssa_v0,)):
            begin__tile: pl.Scalar[pl.INT32] = pl.tensor.read(bounds__ssa_v0, [request__idx_v0])
            end__tile: pl.Scalar[pl.INT32] = pl.tensor.read(bounds__ssa_v0, [request__idx_v0 + 1])
            length__tile: pl.Scalar[pl.INT32] = pl.tensor.read(lengths__ssa_v0, [request__idx_v0])
            start__ssa_v0: pl.Scalar[pl.INT32] = length__tile + begin__tile - end__tile
            for token__idx_v0 in pl.range(begin__tile, end__tile):
                pl.tensor.write(expanded_slots__ssa_v0, [token__idx_v0], pl.cast(-1, pl.INT64))
                pl.tensor.write(compact_indices__ssa_v0, [token__idx_v0], pl.cast(-1, pl.INT32))
                position__tile: pl.Scalar[pl.INT64] = pl.tensor.read(positions__ssa_v0, [token__idx_v0])
                if (position__tile + 1) % 4 == 0:
                    compact__ssa_v0: pl.Scalar[pl.INDEX] = prefix__iter_v1 + (position__tile + 1) // 4 - pl.cast(start__ssa_v0, pl.INDEX) // 4 - 1
                    pl.tensor.write(compact_indices__ssa_v0, [token__idx_v0], pl.cast(compact__ssa_v0, pl.INT32))
                    page__tile: pl.Scalar[pl.INT32] = pl.tensor.read(compact_slots__ssa_v0, [compact__ssa_v0, 0])
                    offset__tile: pl.Scalar[pl.INT32] = pl.tensor.read(compact_slots__ssa_v0, [compact__ssa_v0, 1])
                    if 0 <= pl.cast(page__tile, pl.INDEX) and 0 <= pl.cast(offset__tile, pl.INDEX):
                        pl.tensor.write(expanded_slots__ssa_v0, [token__idx_v0], pl.cast(page__tile, pl.INT64) * 32 + pl.cast(offset__tile, pl.INT64))
            prefix__ssa_v3: pl.Scalar[pl.INDEX] = prefix__iter_v1 + pl.cast(length__tile, pl.INDEX) // 4 - pl.cast(start__ssa_v0, pl.INDEX) // 4
            prefix__rv_v2: pl.Scalar[pl.INDEX] = pl.yield_(prefix__ssa_v3)
        return

    @pl.function(type=pl.FunctionType.Orchestration, level=pl.Level.CHIP, role=pl.Role.Orchestrator, auto_scope=False)
    def prepare_compressed_metadata(
        self,
        bounds__ssa_v0: pl.Tensor[[BOUNDS], pl.INT32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)],
        lengths__ssa_v0: pl.Tensor[[REQUESTS], pl.INT32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)],
        positions__ssa_v0: pl.Tensor[[TOKENS], pl.INT64, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)],
        compact_slots__ssa_v0: pl.Tensor[[COMPACT_ROWS, 2], pl.INT32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)],
        compact_cos__ssa_v0: pl.Tensor[[CSA_NATIVE_COMPACT_ROWS, 64], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 0)],
        compact_sin__ssa_v0: pl.Tensor[[CSA_NATIVE_COMPACT_ROWS, 64], pl.FP32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 0)],
        expanded_slots__ssa_v0: pl.Out[pl.Tensor[[CSA_NATIVE_TOKENS], pl.INT64, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 0)]],
        half_cos__ssa_v0: pl.Out[pl.Tensor[[CSA_NATIVE_TOKENS, 64], pl.BF16, pl.MemRef("mem_ddr_7", pl.const(0, pl.INT64), 0)]],
        half_sin__ssa_v0: pl.Out[pl.Tensor[[CSA_NATIVE_TOKENS, 64], pl.BF16, pl.MemRef("mem_ddr_8", pl.const(0, pl.INT64), 0)]],
    ) -> pl.Tensor[[CSA_NATIVE_TOKENS], pl.INT64]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        compact_indices__ssa_v0: pl.Tensor[[TOKENS], pl.INT32, pl.MemRef("mem_ddr_9", pl.const(0, pl.INT64), 0)] = pl.tensor.create([TOKENS], dtype=pl.INT32, layout=pl.TensorLayout.ND)
        ret__tmp_v0: pl.Tuple[pl.Scalar[pl.TASK_ID]] = pl.submit(
            self.csa_native_compact_metadata,
            REQUESTS,
            bounds__ssa_v0,
            lengths__ssa_v0,
            expanded_slots__ssa_v0,
            compact_indices__ssa_v0,
            positions__ssa_v0,
            compact_slots__ssa_v0,
            attrs={"arg_directions": [pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.output_existing, pl.adir.output_existing, pl.adir.input, pl.adir.input]},
        )
        slots_ready__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0[0]
        ret__tmp_v0_1: pl.Tuple[pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(
            self.csa_expand_compact_rope_spmd,
            compact_indices__ssa_v0,
            compact_cos__ssa_v0,
            compact_sin__ssa_v0,
            half_cos__ssa_v0,
            half_sin__ssa_v0,
            deps=[slots_ready__ssa_v0],
            core_num=TOKENS,
            attrs={"arg_directions": [pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.output_existing, pl.adir.output_existing]},
        )
        tid__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_1[0]