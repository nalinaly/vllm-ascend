# pypto.program: _jit_prepare_token_metadata
import pypto.language as pl

BOUNDS = pl.dynamic("BOUNDS")
CSA_NATIVE_REQUESTS = pl.dynamic("CSA_NATIVE_REQUESTS")
CSA_NATIVE_TOKENS = pl.dynamic("CSA_NATIVE_TOKENS")
REQUESTS = pl.dynamic("REQUESTS")
TABLE_COLUMNS = pl.dynamic("TABLE_COLUMNS")
TOKENS = pl.dynamic("TOKENS")


@pl.program
class _jit_prepare_token_metadata:
    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def csa_native_token_metadata(
        REQUESTS: pl.Scalar[pl.INDEX],
        bounds__ssa_v0: pl.Tensor[[BOUNDS], pl.INT32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)],
        ring_table__ssa_v0: pl.Out[pl.Tensor[[CSA_NATIVE_REQUESTS, 7], pl.INT32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)]],
        positions__ssa_v0: pl.Tensor[[TOKENS], pl.INT64, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)],
        positions_i32__ssa_v0: pl.Out[pl.Tensor[[CSA_NATIVE_TOKENS], pl.INT32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)]],
        swa_slots__ssa_v0: pl.Tensor[[CSA_NATIVE_TOKENS, 2], pl.INT32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 0)],
        original_slots__ssa_v0: pl.Out[pl.Tensor[[CSA_NATIVE_TOKENS], pl.INT64, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 0)]],
        ring_slots__ssa_v0: pl.Out[pl.Tensor[[CSA_NATIVE_TOKENS], pl.INT64, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 0)]],
        inner_ring_slots__ssa_v0: pl.Out[pl.Tensor[[CSA_NATIVE_TOKENS], pl.INT64, pl.MemRef("mem_ddr_7", pl.const(0, pl.INT64), 0)]],
        state_slots__ssa_v0: pl.Tensor[[CSA_NATIVE_TOKENS, 2], pl.INT32, pl.MemRef("mem_ddr_8", pl.const(0, pl.INT64), 0)],
        inner_state_slots__ssa_v0: pl.Tensor[[CSA_NATIVE_TOKENS, 2], pl.INT32, pl.MemRef("mem_ddr_9", pl.const(0, pl.INT64), 0)],
        window_lengths__ssa_v0: pl.Out[pl.Tensor[[CSA_NATIVE_TOKENS], pl.INT32, pl.MemRef("mem_ddr_10", pl.const(0, pl.INT64), 0)]],
        swa_table__ssa_v0: pl.Tensor[[CSA_NATIVE_REQUESTS, TABLE_COLUMNS], pl.INT32, pl.MemRef("mem_ddr_11", pl.const(0, pl.INT64), 0)],
        window_indices__ssa_v0: pl.Out[pl.Tensor[[CSA_NATIVE_TOKENS, 128], pl.INT32, pl.MemRef("mem_ddr_12", pl.const(0, pl.INT64), 0)]],
    ):
        pl.func_attr({"mx_tensor_views_blocked": True})
        for request__idx_v0 in pl.range(REQUESTS):
            begin__tile: pl.Scalar[pl.INT32] = pl.tensor.read(bounds__ssa_v0, [request__idx_v0])
            end__tile: pl.Scalar[pl.INT32] = pl.tensor.read(bounds__ssa_v0, [request__idx_v0 + 1])
            pl.tensor.write(ring_table__ssa_v0, [request__idx_v0, 0], pl.cast(request__idx_v0 * 7, pl.INT32))
            pl.tensor.write(ring_table__ssa_v0, [request__idx_v0, 1], pl.cast(request__idx_v0 * 7 + 1, pl.INT32))
            pl.tensor.write(ring_table__ssa_v0, [request__idx_v0, 2], pl.cast(request__idx_v0 * 7 + 2, pl.INT32))
            pl.tensor.write(ring_table__ssa_v0, [request__idx_v0, 3], pl.cast(request__idx_v0 * 7 + 3, pl.INT32))
            pl.tensor.write(ring_table__ssa_v0, [request__idx_v0, 4], pl.cast(request__idx_v0 * 7 + 4, pl.INT32))
            pl.tensor.write(ring_table__ssa_v0, [request__idx_v0, 5], pl.cast(request__idx_v0 * 7 + 5, pl.INT32))
            pl.tensor.write(ring_table__ssa_v0, [request__idx_v0, 6], pl.cast(request__idx_v0 * 7 + 6, pl.INT32))
            for token__idx_v0 in pl.range(begin__tile, end__tile):
                position__tile: pl.Scalar[pl.INT64] = pl.tensor.read(positions__ssa_v0, [token__idx_v0])
                pl.tensor.write(positions_i32__ssa_v0, [token__idx_v0], pl.cast(position__tile, pl.INT32))
                original_page__tile: pl.Scalar[pl.INT32] = pl.tensor.read(swa_slots__ssa_v0, [token__idx_v0, 0])
                original_offset__tile: pl.Scalar[pl.INT32] = pl.tensor.read(swa_slots__ssa_v0, [token__idx_v0, 1])
                pl.tensor.write(original_slots__ssa_v0, [token__idx_v0], pl.cast(-1, pl.INT64))
                if 0 <= pl.cast(original_page__tile, pl.INDEX) and 0 <= pl.cast(original_offset__tile, pl.INDEX):
                    pl.tensor.write(original_slots__ssa_v0, [token__idx_v0], pl.cast(original_page__tile, pl.INT64) * 32 + pl.cast(original_offset__tile, pl.INT64))
                ring_row__ssa_v0: pl.Scalar[pl.INDEX] = request__idx_v0 * 14 + position__tile % 14
                pl.tensor.write(ring_slots__ssa_v0, [token__idx_v0], pl.cast(-1, pl.INT64))
                pl.tensor.write(inner_ring_slots__ssa_v0, [token__idx_v0], pl.cast(-1, pl.INT64))
                t__tile: pl.Scalar[pl.INT32] = pl.tensor.read(state_slots__ssa_v0, [token__idx_v0, 0])
                t__tile_1: pl.Scalar[pl.INT32] = pl.tensor.read(state_slots__ssa_v0, [token__idx_v0, 1])
                if 0 <= pl.cast(t__tile, pl.INDEX) and 0 <= pl.cast(t__tile_1, pl.INDEX):
                    pl.tensor.write(ring_slots__ssa_v0, [token__idx_v0], pl.cast(ring_row__ssa_v0, pl.INT64))
                t__tile_2: pl.Scalar[pl.INT32] = pl.tensor.read(inner_state_slots__ssa_v0, [token__idx_v0, 0])
                t__tile_3: pl.Scalar[pl.INT32] = pl.tensor.read(inner_state_slots__ssa_v0, [token__idx_v0, 1])
                if 0 <= pl.cast(t__tile_2, pl.INDEX) and 0 <= pl.cast(t__tile_3, pl.INDEX):
                    pl.tensor.write(inner_ring_slots__ssa_v0, [token__idx_v0], pl.cast(ring_row__ssa_v0, pl.INT64))
                window_length__ssa_v0: pl.Scalar[pl.INT64] = pl.min(position__tile + 1, 128)
                window_start__ssa_v0: pl.Scalar[pl.INT64] = position__tile - window_length__ssa_v0 + 1
                pl.tensor.write(window_lengths__ssa_v0, [token__idx_v0], pl.cast(window_length__ssa_v0, pl.INT32))
                for column__idx_v0 in pl.range(128):
                    slot__ssa_v0: pl.Scalar[pl.INT32] = pl.cast(-1, pl.INT32)
                    if column__idx_v0 < window_length__ssa_v0:
                        absolute__ssa_v0: pl.Scalar[pl.INT64] = window_start__ssa_v0 + column__idx_v0
                        physical_page__tile: pl.Scalar[pl.INT32] = pl.tensor.read(swa_table__ssa_v0, [request__idx_v0, absolute__ssa_v0 // 32])
                        if 0 <= pl.cast(physical_page__tile, pl.INDEX):
                            slot__ssa_v1: pl.Scalar[pl.INT32] = pl.cast(pl.cast(physical_page__tile, pl.INDEX) * 32 + absolute__ssa_v0 % 32, pl.INT32)
                            slot__phi_v2: pl.Scalar[pl.INT32] = pl.yield_(slot__ssa_v1)
                        else:
                            slot__phi_v2: pl.Scalar[pl.INT32] = pl.yield_(slot__ssa_v0)
                        slot__phi_v3: pl.Scalar[pl.INT32] = pl.yield_(slot__phi_v2)
                    else:
                        slot__phi_v3: pl.Scalar[pl.INT32] = pl.yield_(slot__ssa_v0)
                    pl.tensor.write(window_indices__ssa_v0, [token__idx_v0, column__idx_v0], slot__phi_v3)
        return

    @pl.function(type=pl.FunctionType.Orchestration, level=pl.Level.CHIP, role=pl.Role.Orchestrator, auto_scope=False)
    def prepare_token_metadata(
        self,
        bounds__ssa_v0: pl.Tensor[[BOUNDS], pl.INT32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)],
        lengths__ssa_v0: pl.Tensor[[REQUESTS], pl.INT32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)],
        positions__ssa_v0: pl.Tensor[[TOKENS], pl.INT64, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)],
        swa_table__ssa_v0: pl.Tensor[[CSA_NATIVE_REQUESTS, TABLE_COLUMNS], pl.INT32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)],
        swa_slots__ssa_v0: pl.Tensor[[CSA_NATIVE_TOKENS, 2], pl.INT32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 0)],
        state_slots__ssa_v0: pl.Tensor[[CSA_NATIVE_TOKENS, 2], pl.INT32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 0)],
        inner_state_slots__ssa_v0: pl.Tensor[[CSA_NATIVE_TOKENS, 2], pl.INT32, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 0)],
        positions_i32__ssa_v0: pl.Out[pl.Tensor[[CSA_NATIVE_TOKENS], pl.INT32, pl.MemRef("mem_ddr_7", pl.const(0, pl.INT64), 0)]],
        original_slots__ssa_v0: pl.Out[pl.Tensor[[CSA_NATIVE_TOKENS], pl.INT64, pl.MemRef("mem_ddr_8", pl.const(0, pl.INT64), 0)]],
        ring_slots__ssa_v0: pl.Out[pl.Tensor[[CSA_NATIVE_TOKENS], pl.INT64, pl.MemRef("mem_ddr_9", pl.const(0, pl.INT64), 0)]],
        inner_ring_slots__ssa_v0: pl.Out[pl.Tensor[[CSA_NATIVE_TOKENS], pl.INT64, pl.MemRef("mem_ddr_10", pl.const(0, pl.INT64), 0)]],
        window_indices__ssa_v0: pl.Out[pl.Tensor[[CSA_NATIVE_TOKENS, 128], pl.INT32, pl.MemRef("mem_ddr_11", pl.const(0, pl.INT64), 0)]],
        window_lengths__ssa_v0: pl.Out[pl.Tensor[[CSA_NATIVE_TOKENS], pl.INT32, pl.MemRef("mem_ddr_12", pl.const(0, pl.INT64), 0)]],
        ring_table__ssa_v0: pl.Out[pl.Tensor[[CSA_NATIVE_REQUESTS, 7], pl.INT32, pl.MemRef("mem_ddr_13", pl.const(0, pl.INT64), 0)]],
    ) -> pl.Tensor[[CSA_NATIVE_TOKENS], pl.INT32]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        with pl.scope():
            self.csa_native_token_metadata(
                REQUESTS,
                bounds__ssa_v0,
                ring_table__ssa_v0,
                positions__ssa_v0,
                positions_i32__ssa_v0,
                swa_slots__ssa_v0,
                original_slots__ssa_v0,
                ring_slots__ssa_v0,
                inner_ring_slots__ssa_v0,
                state_slots__ssa_v0,
                inner_state_slots__ssa_v0,
                window_lengths__ssa_v0,
                swa_table__ssa_v0,
                window_indices__ssa_v0,
                attrs={
                    "arg_directions": [
                        pl.adir.scalar,
                        pl.adir.input,
                        pl.adir.output_existing,
                        pl.adir.input,
                        pl.adir.output_existing,
                        pl.adir.input,
                        pl.adir.output_existing,
                        pl.adir.output_existing,
                        pl.adir.output_existing,
                        pl.adir.input,
                        pl.adir.input,
                        pl.adir.output_existing,
                        pl.adir.input,
                        pl.adir.output_existing,
                    ]
                },
            )