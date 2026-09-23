# pypto.program: _jit_commit_state_window
import pypto.language as pl

BOUNDS = pl.dynamic("BOUNDS")
CSA_NATIVE_TOKENS = pl.dynamic("CSA_NATIVE_TOKENS")
PAGES = pl.dynamic("PAGES")
PAGE_ELEMENTS = pl.dynamic("PAGE_ELEMENTS")
REQUESTS = pl.dynamic("REQUESTS")
SCRATCH_PAGES = pl.dynamic("SCRATCH_PAGES")
STATE_WIDTH = pl.dynamic("STATE_WIDTH")
TOKENS = pl.dynamic("TOKENS")


@pl.program
class _jit_commit_state_window:
    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def csa_commit_state_window(
        bounds__ssa_v0: pl.Tensor[[BOUNDS], pl.INT32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)],
        native_state__ssa_v0: pl.Out[pl.Tensor[[PAGES, PAGE_ELEMENTS], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)]],
        native_slots__ssa_v0: pl.Tensor[[TOKENS, 2], pl.INT32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)],
        ring_slots__ssa_v0: pl.Tensor[[CSA_NATIVE_TOKENS], pl.INT64, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)],
        STATE_WIDTH: pl.Scalar[pl.INDEX],
        flat__ssa_v0: pl.Tensor[[SCRATCH_PAGES * pl.const(2, pl.INDEX), STATE_WIDTH], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 0)],
    ):
        pl.func_attr({"mx_tensor_views_blocked": True})
        mem_vec_5: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 2048)
        request__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        begin__tile: pl.Scalar[pl.INT32] = pl.tensor.read(bounds__ssa_v0, [request__ssa_v0])
        end__tile: pl.Scalar[pl.INT32] = pl.tensor.read(bounds__ssa_v0, [request__ssa_v0 + 1])
        for token__idx_v0, (native_state__iter_v1,) in pl.range(begin__tile, end__tile, init_values=(native_state__ssa_v0,)):
            page__tile: pl.Scalar[pl.INT32] = pl.tensor.read(native_slots__ssa_v0, [token__idx_v0, 0])
            offset__tile: pl.Scalar[pl.INT32] = pl.tensor.read(native_slots__ssa_v0, [token__idx_v0, 1])
            ring_row__tile: pl.Scalar[pl.INT64] = pl.tensor.read(ring_slots__ssa_v0, [token__idx_v0])
            if 0 <= pl.cast(page__tile, pl.INDEX) and 0 <= pl.cast(offset__tile, pl.INDEX) and 0 <= ring_row__tile:
                for column__idx_v0, (native_state__iter_v3,) in pl.range(0, STATE_WIDTH, 512, init_values=(native_state__iter_v1,)):
                    start__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(offset__tile, pl.INDEX) * STATE_WIDTH + column__idx_v0
                    t__tile: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_5, pl.const(0, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.load(
                        flat__ssa_v0, [ring_row__tile, column__idx_v0], [1, 512], [1, 512], target_memory=pl.Mem.Vec
                    )
                    native_state__tile: pl.Tensor[[PAGES, PAGE_ELEMENTS], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                        t__tile, [page__tile, start__ssa_v0], native_state__iter_v3
                    )
                    native_state__rv_v4: pl.Tensor[[PAGES, PAGE_ELEMENTS], pl.FP32, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 0)] = pl.yield_(native_state__tile)
                native_state__phi_v6: pl.Tensor[[PAGES, PAGE_ELEMENTS], pl.FP32, pl.MemRef("mem_ddr_7", pl.const(0, pl.INT64), 0)] = pl.yield_(native_state__rv_v4)
            else:
                native_state__phi_v6: pl.Tensor[[PAGES, PAGE_ELEMENTS], pl.FP32, pl.MemRef("mem_ddr_7", pl.const(0, pl.INT64), 0)] = pl.yield_(native_state__iter_v1)
            native_state__rv_v2: pl.Tensor[[PAGES, PAGE_ELEMENTS], pl.FP32, pl.MemRef("mem_ddr_8", pl.const(0, pl.INT64), 0)] = pl.yield_(native_state__phi_v6)
        return

    @pl.function(type=pl.FunctionType.Spmd)
    def csa_commit_state_window_spmd(
        self,
        bounds__ssa_v0: pl.Tensor[[BOUNDS], pl.INT32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)],
        native_state__ssa_v0: pl.Out[pl.Tensor[[PAGES, PAGE_ELEMENTS], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)]],
        native_slots__ssa_v0: pl.Tensor[[TOKENS, 2], pl.INT32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)],
        ring_slots__ssa_v0: pl.Tensor[[CSA_NATIVE_TOKENS], pl.INT64, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)],
        STATE_WIDTH: pl.Scalar[pl.INDEX],
        flat__ssa_v0: pl.Tensor[[SCRATCH_PAGES * pl.const(2, pl.INDEX), STATE_WIDTH], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 0)],
    ):
        pl.func_attr({"mx_tensor_views_blocked": True})
        self.csa_commit_state_window(
            bounds__ssa_v0,
            native_state__ssa_v0,
            native_slots__ssa_v0,
            ring_slots__ssa_v0,
            STATE_WIDTH,
            flat__ssa_v0,
            attrs={"arg_directions": [pl.adir.input, pl.adir.output_existing, pl.adir.input, pl.adir.input, pl.adir.scalar, pl.adir.input]},
        )

    @pl.function(type=pl.FunctionType.Orchestration, level=pl.Level.CHIP, role=pl.Role.Orchestrator, auto_scope=False)
    def commit_state_window(
        self,
        scratch__ssa_v0: pl.Tensor[[SCRATCH_PAGES, 2, STATE_WIDTH], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)],
        native_slots__ssa_v0: pl.Tensor[[TOKENS, 2], pl.INT32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)],
        ring_slots__ssa_v0: pl.Tensor[[CSA_NATIVE_TOKENS], pl.INT64, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)],
        bounds__ssa_v0: pl.Tensor[[BOUNDS], pl.INT32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)],
        lengths__ssa_v0: pl.Tensor[[REQUESTS], pl.INT32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 0)],
        native_state__ssa_v0: pl.InOut[pl.Tensor[[PAGES, PAGE_ELEMENTS], pl.FP32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 0)]],
    ):
        pl.func_attr({"mx_tensor_views_blocked": True})
        flat__ssa_v0: pl.Tensor[[SCRATCH_PAGES * pl.const(2, pl.INDEX), STATE_WIDTH], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)] = pl.tensor.reshape(
            scratch__ssa_v0, [SCRATCH_PAGES * 2, STATE_WIDTH]
        )
        self.csa_commit_state_window_spmd(
            bounds__ssa_v0,
            native_state__ssa_v0,
            native_slots__ssa_v0,
            ring_slots__ssa_v0,
            STATE_WIDTH,
            flat__ssa_v0,
            attrs={"arg_directions": [pl.adir.input, pl.adir.inout, pl.adir.input, pl.adir.input, pl.adir.scalar, pl.adir.input], "core_num": REQUESTS},
        )