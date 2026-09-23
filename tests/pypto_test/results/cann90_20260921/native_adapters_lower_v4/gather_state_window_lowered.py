# pypto.program: _jit_gather_state_window
import pypto.language as pl

BOUNDS = pl.dynamic("BOUNDS")
CSA_NATIVE_REQUESTS = pl.dynamic("CSA_NATIVE_REQUESTS")
PAGES = pl.dynamic("PAGES")
PAGE_ELEMENTS = pl.dynamic("PAGE_ELEMENTS")
REQUESTS = pl.dynamic("REQUESTS")
SCRATCH_PAGES = pl.dynamic("SCRATCH_PAGES")
STATE_WIDTH = pl.dynamic("STATE_WIDTH")
TABLE_COLUMNS = pl.dynamic("TABLE_COLUMNS")


@pl.program
class _jit_gather_state_window:
    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def csa_gather_state_window(
        bounds__ssa_v0: pl.Tensor[[BOUNDS], pl.INT32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)],
        lengths__ssa_v0: pl.Tensor[[CSA_NATIVE_REQUESTS], pl.INT32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)],
        flat__ssa_v0: pl.Out[pl.Tensor[[SCRATCH_PAGES * pl.const(2, pl.INDEX), STATE_WIDTH], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)]],
        STATE_WIDTH: pl.Scalar[pl.INDEX],
        table__ssa_v0: pl.Tensor[[REQUESTS, TABLE_COLUMNS], pl.INT32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)],
        native_state__ssa_v0: pl.Tensor[[PAGES, PAGE_ELEMENTS], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 0)],
    ):
        pl.func_attr({"mx_tensor_views_blocked": True})
        mem_vec_5: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 2048)
        mem_vec_6: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 2048)
        request__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        begin__tile: pl.Scalar[pl.INT32] = pl.tensor.read(bounds__ssa_v0, [request__ssa_v0])
        end__tile: pl.Scalar[pl.INT32] = pl.tensor.read(bounds__ssa_v0, [request__ssa_v0 + 1])
        t__tile: pl.Scalar[pl.INT32] = pl.tensor.read(lengths__ssa_v0, [request__ssa_v0])
        first_position__ssa_v0: pl.Scalar[pl.INT32] = t__tile + begin__tile - end__tile
        for relative__idx_v0, (flat__iter_v1,) in pl.range(14, init_values=(flat__ssa_v0,)):
            absolute__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(first_position__ssa_v0, pl.INDEX) - 8 + relative__idx_v0
            ring__ssa_v0: pl.Scalar[pl.INDEX] = absolute__ssa_v0 % 14
            destination__ssa_v0: pl.Scalar[pl.INDEX] = request__ssa_v0 * 14 + ring__ssa_v0
            for column__idx_v0, (flat__iter_v3,) in pl.range(0, STATE_WIDTH, 512, init_values=(flat__iter_v1,)):
                values__tile: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_5, pl.const(0, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.full([1, 512], dtype=pl.FP32, value=0.0)
                if relative__idx_v0 < 8 and 0 <= absolute__ssa_v0:
                    page__tile: pl.Scalar[pl.INT32] = pl.tensor.read(table__ssa_v0, [request__ssa_v0, absolute__ssa_v0 // 2])
                    if 0 <= pl.cast(page__tile, pl.INDEX):
                        offset__ssa_v0: pl.Scalar[pl.INDEX] = absolute__ssa_v0 % 2 * STATE_WIDTH + column__idx_v0
                        values__tile_1: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_6, pl.const(2048, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.load(
                            native_state__ssa_v0, [page__tile, offset__ssa_v0], [1, 512], [1, 512], target_memory=pl.Mem.Vec
                        )
                        values__phi_v2: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_6, pl.const(2048, pl.INT64), 2048), pl.Mem.Vec] = pl.yield_(values__tile_1)
                    else:
                        values__tile_mv: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_6, pl.const(2048, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.move(values__tile, target_memory=pl.Mem.Vec)
                        values__phi_v2: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_6, pl.const(2048, pl.INT64), 2048), pl.Mem.Vec] = pl.yield_(values__tile_mv)
                    values__phi_v3: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_6, pl.const(2048, pl.INT64), 2048), pl.Mem.Vec] = pl.yield_(values__phi_v2)
                else:
                    values__tile_mv_1: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_6, pl.const(2048, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.move(values__tile, target_memory=pl.Mem.Vec)
                    values__phi_v3: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_6, pl.const(2048, pl.INT64), 2048), pl.Mem.Vec] = pl.yield_(values__tile_mv_1)
                flat__tile: pl.Tensor[[SCRATCH_PAGES * pl.const(2, pl.INDEX), STATE_WIDTH], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                    values__phi_v3, [destination__ssa_v0, column__idx_v0], flat__iter_v3
                )
                flat__rv_v4: pl.Tensor[[SCRATCH_PAGES * pl.const(2, pl.INDEX), STATE_WIDTH], pl.FP32, pl.MemRef("mem_ddr_9", pl.const(0, pl.INT64), 0)] = pl.yield_(flat__tile)
            flat__rv_v2: pl.Tensor[[SCRATCH_PAGES * pl.const(2, pl.INDEX), STATE_WIDTH], pl.FP32, pl.MemRef("mem_ddr_10", pl.const(0, pl.INT64), 0)] = pl.yield_(flat__rv_v4)
        return

    @pl.function(type=pl.FunctionType.Spmd)
    def csa_gather_state_window_spmd(
        self,
        bounds__ssa_v0: pl.Tensor[[BOUNDS], pl.INT32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)],
        lengths__ssa_v0: pl.Tensor[[CSA_NATIVE_REQUESTS], pl.INT32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)],
        flat__ssa_v0: pl.Out[pl.Tensor[[SCRATCH_PAGES * pl.const(2, pl.INDEX), STATE_WIDTH], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)]],
        STATE_WIDTH: pl.Scalar[pl.INDEX],
        table__ssa_v0: pl.Tensor[[REQUESTS, TABLE_COLUMNS], pl.INT32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)],
        native_state__ssa_v0: pl.Tensor[[PAGES, PAGE_ELEMENTS], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 0)],
    ):
        pl.func_attr({"mx_tensor_views_blocked": True})
        self.csa_gather_state_window(
            bounds__ssa_v0,
            lengths__ssa_v0,
            flat__ssa_v0,
            STATE_WIDTH,
            table__ssa_v0,
            native_state__ssa_v0,
            attrs={"arg_directions": [pl.adir.input, pl.adir.input, pl.adir.output_existing, pl.adir.scalar, pl.adir.input, pl.adir.input]},
        )

    @pl.function(type=pl.FunctionType.Orchestration, level=pl.Level.CHIP, role=pl.Role.Orchestrator, auto_scope=False)
    def gather_state_window(
        self,
        native_state__ssa_v0: pl.Tensor[[PAGES, PAGE_ELEMENTS], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)],
        table__ssa_v0: pl.Tensor[[REQUESTS, TABLE_COLUMNS], pl.INT32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)],
        bounds__ssa_v0: pl.Tensor[[BOUNDS], pl.INT32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)],
        lengths__ssa_v0: pl.Tensor[[CSA_NATIVE_REQUESTS], pl.INT32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)],
        scratch__ssa_v0: pl.Out[pl.Tensor[[SCRATCH_PAGES, 2, STATE_WIDTH], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 0)]],
    ) -> pl.Tensor[[SCRATCH_PAGES, 2, STATE_WIDTH], pl.FP32]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        flat__ssa_v0: pl.Tensor[[SCRATCH_PAGES * pl.const(2, pl.INDEX), STATE_WIDTH], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 0)] = pl.tensor.reshape(
            scratch__ssa_v0, [SCRATCH_PAGES * 2, STATE_WIDTH]
        )
        self.csa_gather_state_window_spmd(
            bounds__ssa_v0,
            lengths__ssa_v0,
            flat__ssa_v0,
            STATE_WIDTH,
            table__ssa_v0,
            native_state__ssa_v0,
            attrs={"arg_directions": [pl.adir.input, pl.adir.input, pl.adir.output_existing, pl.adir.scalar, pl.adir.input, pl.adir.input], "core_num": CSA_NATIVE_REQUESTS},
        )