# pypto.program: _jit__decode_csa_tp1_attention
import pypto.language as pl

B_DYN = pl.dynamic("B_DYN")
CMP_BLOCK_NUM_DYN = pl.dynamic("CMP_BLOCK_NUM_DYN")
COMPRESSED_ROWS_DYN = pl.dynamic("COMPRESSED_ROWS_DYN")
COMPRESSED_TABLE_COLUMNS_DYN = pl.dynamic("COMPRESSED_TABLE_COLUMNS_DYN")
IDX_CACHE_BLOCK_NUM_DYN = pl.dynamic("IDX_CACHE_BLOCK_NUM_DYN")
INDEXER_PAGE_BYTES_DYN = pl.dynamic("INDEXER_PAGE_BYTES_DYN")
INDEXER_ROWS_DYN = pl.dynamic("INDEXER_ROWS_DYN")
INDEXER_TABLE_COLUMNS_DYN = pl.dynamic("INDEXER_TABLE_COLUMNS_DYN")
INNER_STATE_BLOCK_NUM_DYN = pl.dynamic("INNER_STATE_BLOCK_NUM_DYN")
INNER_STATE_PAGE_ELEMENTS_DYN = pl.dynamic("INNER_STATE_PAGE_ELEMENTS_DYN")
INNER_STATE_TABLE_COLUMNS_DYN = pl.dynamic("INNER_STATE_TABLE_COLUMNS_DYN")
MAIN_STATE_BLOCK_NUM_DYN = pl.dynamic("MAIN_STATE_BLOCK_NUM_DYN")
ORIGINAL_TABLE_COLUMNS_DYN = pl.dynamic("ORIGINAL_TABLE_COLUMNS_DYN")
ORI_BLOCK_NUM_DYN = pl.dynamic("ORI_BLOCK_NUM_DYN")
QUERY_BOUNDS_DYN = pl.dynamic("QUERY_BOUNDS_DYN")
STATE_PAGE_ELEMENTS_DYN = pl.dynamic("STATE_PAGE_ELEMENTS_DYN")
STATE_TABLE_COLUMNS_DYN = pl.dynamic("STATE_TABLE_COLUMNS_DYN")
T_DYN = pl.dynamic("T_DYN")
t_dim_inline2491__ssa_v0 = pl.dynamic("t_dim_inline2491__ssa_v0")


@pl.program
class _jit__decode_csa_tp1_attention:
    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def indexer_topk_query_merge(
        position_ids__ssa_v0: pl.Tensor[[T_DYN], pl.INT64, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)],
        kv_seq_lens__ssa_v0: pl.Tensor[[B_DYN], pl.INT32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)],
        pair_arena__ssa_v0: pl.InOut[pl.Tensor[[24576, 1024], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 100663296)]],
        topk_scores__ssa_v0: pl.Out[pl.Tensor[[T_DYN, 512], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)]],
        topk_indices__ssa_v0: pl.Out[pl.Tensor[[T_DYN, 512], pl.INT32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 0)]],
    ):
        pl.func_attr({"mx_tensor_views_blocked": True})
        mem_vec_5: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 4096)
        mem_vec_6: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 4096)
        mem_vec_7: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 8192)
        mem_vec_8: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 8192)
        worker__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        query_count__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(position_ids__ssa_v0, 0)
        for query__idx_v0 in pl.range(worker__ssa_v0, query_count__ssa_v0, 48):
            batch_idx_inline537__ssa_v0: pl.Scalar[pl.INDEX] = query__idx_v0 // 6
            position_inline538__tile: pl.Scalar[pl.INT64] = pl.tensor.read(position_ids__ssa_v0, [query__idx_v0])
            t__tile: pl.Scalar[pl.INT32] = pl.tensor.read(kv_seq_lens__ssa_v0, [batch_idx_inline537__ssa_v0])
            cache_len_inline544__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(t__tile, pl.INDEX) // 4
            cache_bound_inline540__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(cache_len_inline544__ssa_v0, (position_inline538__tile + 1) // 4)
            visible_count_inline541__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(cache_bound_inline540__ssa_v0, 262144)
            if 0 < visible_count_inline541__ssa_v0:
                leaf_count_inline542__ssa_v0: pl.Scalar[pl.INDEX] = (visible_count_inline541__ssa_v0 + 8191) // 8192
                half_count_inline543__ssa_v0: pl.Scalar[pl.INDEX] = leaf_count_inline542__ssa_v0 * 2
                arena_base_inline545__ssa_v0: pl.Scalar[pl.INDEX] = query__idx_v0 * 64
                for child_inline539__idx_v0 in pl.range(1, half_count_inline543__ssa_v0):
                    left_inline2568__ssa_v0: pl.Tile[[1, 1024], pl.FP32, pl.MemRef(mem_vec_5, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.load(
                        pair_arena__ssa_v0, [arena_base_inline545__ssa_v0, 0], [1, 1024], [1, 1024], target_memory=pl.Mem.Vec
                    )
                    right_inline2567__ssa_v0: pl.Tile[[1, 1024], pl.FP32, pl.MemRef(mem_vec_6, pl.const(4096, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.load(
                        pair_arena__ssa_v0, [arena_base_inline545__ssa_v0 + child_inline539__idx_v0, 0], [1, 1024], [1, 1024], target_memory=pl.Mem.Vec
                    )
                    merge_tmp_inline2566__ssa_v0: pl.Tile[[1, 2048], pl.FP32, pl.MemRef(mem_vec_7, pl.const(8192, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.create(
                        [1, 2048], dtype=pl.FP32, target_memory=pl.Mem.Vec
                    )
                    merged_all_inline2565__ssa_v0: pl.Tile[[1, 2048], pl.FP32, pl.MemRef(mem_vec_8, pl.const(16384, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.mrgsort_format2(
                        right_inline2567__ssa_v0, left_inline2568__ssa_v0, merge_tmp_inline2566__ssa_v0, exhausted=False
                    )
                    merged_inline2564__ssa_v0: pl.Tile[[1, 1024], pl.FP32, pl.MemRef(mem_vec_8, pl.const(16384, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.slice(
                        merged_all_inline2565__ssa_v0, [1, 1024], [0, 0]
                    )
                    pl.tile.store(merged_inline2564__ssa_v0, [arena_base_inline545__ssa_v0, 0], pair_arena__ssa_v0)
                root_slot_inline546__ssa_v0: pl.Scalar[pl.INDEX] = arena_base_inline545__ssa_v0
                root_pairs_inline548__ssa_v0: pl.Tile[[1, 1024], pl.FP32, pl.MemRef(mem_vec_7, pl.const(8192, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.load(
                    pair_arena__ssa_v0, [root_slot_inline546__ssa_v0, 0], [1, 1024], [1, 1024], target_memory=pl.Mem.Vec
                )
                root_scores_inline549__ssa_v0: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_8, pl.const(16384, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.gather_mask(
                    root_pairs_inline548__ssa_v0, mask_pattern=1, output_dtype=pl.FP32
                )
                pl.tile.store(root_scores_inline549__ssa_v0, [query__idx_v0, 0], topk_scores__ssa_v0)
                root_indices_inline550__ssa_v0: pl.Tile[[1, 512], pl.INT32, pl.MemRef(mem_vec_8, pl.const(16384, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.gather_mask(
                    root_pairs_inline548__ssa_v0, mask_pattern=2, output_dtype=pl.INT32
                )
                if 512 <= visible_count_inline541__ssa_v0:
                    pl.tile.store(root_indices_inline550__ssa_v0, [query__idx_v0, 0], topk_indices__ssa_v0)
                else:
                    output_indices_inline547__ssa_v0: pl.Tile[[1, 512], pl.INT32, pl.MemRef(mem_vec_7, pl.const(8192, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.full([1, 512], dtype=pl.INT32, value=-1)
                    for lane_inline536__idx_v0 in pl.range(visible_count_inline541__ssa_v0):
                        t__tmp_v1: pl.Scalar[pl.INT32] = pl.tile.read(root_indices_inline550__ssa_v0, [0, lane_inline536__idx_v0])
                        pl.tile.write(output_indices_inline547__ssa_v0, [0, lane_inline536__idx_v0], t__tmp_v1)
                    pl.tile.store(output_indices_inline547__ssa_v0, [query__idx_v0, 0], topk_indices__ssa_v0)
            else:
                t__tmp_v2: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_7, pl.const(8192, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.full([1, 512], dtype=pl.FP32, value=-3.4028234663852886e38)
                pl.tile.store(t__tmp_v2, [query__idx_v0, 0], topk_scores__ssa_v0)
                t__tmp_v3: pl.Tile[[1, 512], pl.INT32, pl.MemRef(mem_vec_7, pl.const(8192, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.full([1, 512], dtype=pl.INT32, value=-1)
                pl.tile.store(t__tmp_v3, [query__idx_v0, 0], topk_indices__ssa_v0)
        return

    @pl.function(type=pl.FunctionType.Spmd)
    def _decode_csa_tp1_attention_indexer_topk_query_merge(
        self,
        position_ids__ssa_v0: pl.Tensor[[T_DYN], pl.INT64, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)],
        kv_seq_lens__ssa_v0: pl.Tensor[[B_DYN], pl.INT32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)],
        pair_arena_inline1064_inline2363__ssa_v1: pl.InOut[pl.Tensor[[24576, 1024], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 100663296)]],
        idx_topk_scores__ssa_v0: pl.Out[pl.Tensor[[T_DYN, 512], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)]],
        idx_topk__ssa_v0: pl.Out[pl.Tensor[[T_DYN, 512], pl.INT32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 0)]],
    ):
        pl.func_attr({"mx_tensor_views_blocked": True})
        self.indexer_topk_query_merge(
            position_ids__ssa_v0,
            kv_seq_lens__ssa_v0,
            pair_arena_inline1064_inline2363__ssa_v1,
            idx_topk_scores__ssa_v0,
            idx_topk__ssa_v0,
            attrs={"arg_directions": [pl.adir.input, pl.adir.input, pl.adir.inout, pl.adir.output_existing, pl.adir.output_existing]},
        )

    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def indexer_topk_single_leaf_publish(
        position_ids__ssa_v0: pl.Tensor[[T_DYN], pl.INT64, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)],
        kv_seq_lens__ssa_v0: pl.Tensor[[B_DYN], pl.INT32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)],
        score_arena__ssa_v0: pl.Tensor[[384, 8192], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 12582912)],
        topk_scores__ssa_v0: pl.Out[pl.Tensor[[T_DYN, 512], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)]],
        topk_indices__ssa_v0: pl.Out[pl.Tensor[[T_DYN, 512], pl.INT32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 0)]],
    ):
        pl.func_attr({"mx_tensor_views_blocked": True})
        mem_vec_34: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32768)
        mem_vec_36: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 65536)
        mem_vec_37: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 65536)
        worker__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        query_count__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(position_ids__ssa_v0, 0)
        for query__idx_v0 in pl.range(worker__ssa_v0, query_count__ssa_v0, 48):
            position__tile: pl.Scalar[pl.INT64] = pl.tensor.read(position_ids__ssa_v0, [query__idx_v0])
            t__tile: pl.Scalar[pl.INT32] = pl.tensor.read(kv_seq_lens__ssa_v0, [query__idx_v0 // 6])
            cache_len__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(t__tile, pl.INDEX) // 4
            visible_count__ssa_v0: pl.Scalar[pl.INDEX] = pl.max(pl.min(cache_len__ssa_v0, (position__tile + 1) // 4), 0)
            if 0 < visible_count__ssa_v0:
                # Sort one populated leaf and directly publish its Top-512 pairs.
                if visible_count__ssa_v0 <= 2048:
                    short_indices_inline980__ci_tmp_v0: pl.Tile[[1, 192], pl.FP32, pl.MemRef(mem_vec_36, pl.const(32768, pl.INT64), 768), pl.Mem.Vec] = pl.tile.create(
                        [1, 192], dtype=pl.FP32, target_memory=pl.Mem.Vec
                    )
                    short_indices_inline980__ssa_v0: pl.Tile[[1, 2048], pl.INT32, pl.MemRef(mem_vec_37, pl.const(98304, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.ci(
                        pl.const(0, pl.INT32), [1, 2048], tmp=short_indices_inline980__ci_tmp_v0, dtype=pl.INT32, descending=False
                    )
                    short_raw_inline982__ssa_v0: pl.Tile[
                        [1, 2048], pl.FP32, pl.MemRef(mem_vec_36, pl.const(32768, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[1, visible_count__ssa_v0])
                    ] = pl.tile.load(score_arena__ssa_v0, [query__idx_v0, 0], [1, 2048], [1, visible_count__ssa_v0], target_memory=pl.Mem.Vec)
                    short_scores_inline989__ssa_v0: pl.Tile[[1, 2048], pl.FP32, pl.MemRef(mem_vec_36, pl.const(32768, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.fillpad(
                        short_raw_inline982__ssa_v0, pad_value=pl.PadValue.min
                    )
                    short_scores_v1_inline987__ssa_v0: pl.Tile[[1, 2048], pl.FP32, pl.MemRef(mem_vec_34, pl.const(0, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.maximums(
                        short_scores_inline989__ssa_v0, -3.4028234663852886e38
                    )
                    t__tmp_v1: pl.Tile[[1, 2048], pl.UINT32, pl.MemRef(mem_vec_37, pl.const(98304, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.reinterpret_view(
                        short_indices_inline980__ssa_v0, dtype=pl.UINT32
                    )
                    short_pairs_inline981__ssa_v0: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_36, pl.const(32768, pl.INT64), 16384), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.sort32(
                        short_scores_v1_inline987__ssa_v0, t__tmp_v1
                    )
                    short_pairs_v1_inline976__ssa_v0: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_37, pl.const(98304, pl.INT64), 16384), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = (
                        pl.tile.mrgsort_format1(short_pairs_inline981__ssa_v0, pl.const(64, pl.INT32))
                    )
                    short_pairs_v2_inline988__ssa_v0: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_36, pl.const(32768, pl.INT64), 16384), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = (
                        pl.tile.mrgsort_format1(short_pairs_v1_inline976__ssa_v0, pl.const(256, pl.INT32))
                    )
                    short_pairs_v3_inline990__ssa_v0: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_37, pl.const(98304, pl.INT64), 16384), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = (
                        pl.tile.mrgsort_format1(short_pairs_v2_inline988__ssa_v0, pl.const(1024, pl.INT32))
                    )
                    short_top_inline1001__ssa_v0: pl.Tile[[1, 1024], pl.FP32, pl.MemRef(mem_vec_37, pl.const(98304, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(
                        short_pairs_v3_inline990__ssa_v0, [1, 1024], [0, 0]
                    )
                    short_values_inline978__ssa_v0: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_36, pl.const(32768, pl.INT64), 2048), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = (
                        pl.tile.gather_mask(short_top_inline1001__ssa_v0, mask_pattern=1, output_dtype=pl.FP32)
                    )
                    short_selected_inline975__ssa_v0: pl.Tile[[1, 512], pl.INT32, pl.MemRef(mem_vec_34, pl.const(0, pl.INT64), 2048), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = (
                        pl.tile.gather_mask(short_top_inline1001__ssa_v0, mask_pattern=2, output_dtype=pl.INT32)
                    )
                    pl.tile.store(short_values_inline978__ssa_v0, [query__idx_v0, 0], topk_scores__ssa_v0)
                    if 512 <= visible_count__ssa_v0:
                        pl.tile.store(short_selected_inline975__ssa_v0, [query__idx_v0, 0], topk_indices__ssa_v0)
                    else:
                        short_output_inline968__ssa_v0: pl.Tile[[1, 512], pl.INT32, pl.MemRef(mem_vec_36, pl.const(32768, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.full(
                            [1, 512], dtype=pl.INT32, value=-1
                        )
                        for lane_inline979__idx_v0 in pl.range(visible_count__ssa_v0):
                            t__tmp_v2: pl.Scalar[pl.INT32] = pl.tile.read(short_selected_inline975__ssa_v0, [0, lane_inline979__idx_v0])
                            pl.tile.write(short_output_inline968__ssa_v0, [0, lane_inline979__idx_v0], t__tmp_v2)
                        pl.tile.store(short_output_inline968__ssa_v0, [query__idx_v0, 0], topk_indices__ssa_v0)
                else:
                    if visible_count__ssa_v0 <= 4096:
                        medium_indices_inline991__ci_tmp_v1: pl.Tile[[1, 192], pl.FP32, pl.MemRef(mem_vec_36, pl.const(32768, pl.INT64), 768), pl.Mem.Vec] = pl.tile.create(
                            [1, 192], dtype=pl.FP32, target_memory=pl.Mem.Vec
                        )
                        medium_indices_inline991__ssa_v0: pl.Tile[[1, 4096], pl.INT32, pl.MemRef(mem_vec_37, pl.const(98304, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.ci(
                            pl.const(0, pl.INT32), [1, 4096], tmp=medium_indices_inline991__ci_tmp_v1, dtype=pl.INT32, descending=False
                        )
                        medium_raw_inline983__ssa_v0: pl.Tile[
                            [1, 4096], pl.FP32, pl.MemRef(mem_vec_36, pl.const(32768, pl.INT64), 16384), pl.Mem.Vec, pl.TileView(valid_shape=[1, visible_count__ssa_v0])
                        ] = pl.tile.load(score_arena__ssa_v0, [query__idx_v0, 0], [1, 4096], [1, visible_count__ssa_v0], target_memory=pl.Mem.Vec)
                        medium_scores_inline998__ssa_v0: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_36, pl.const(32768, pl.INT64), 16384), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = (
                            pl.tile.fillpad(medium_raw_inline983__ssa_v0, pad_value=pl.PadValue.min)
                        )
                        medium_scores_v1_inline974__ssa_v0: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_34, pl.const(0, pl.INT64), 16384), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = (
                            pl.tile.maximums(medium_scores_inline998__ssa_v0, -3.4028234663852886e38)
                        )
                        t__tmp_v3: pl.Tile[[1, 4096], pl.UINT32, pl.MemRef(mem_vec_37, pl.const(98304, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.reinterpret_view(
                            medium_indices_inline991__ssa_v0, dtype=pl.UINT32
                        )
                        medium_pairs_inline970__ssa_v0: pl.Tile[[1, 8192], pl.FP32, pl.MemRef(mem_vec_36, pl.const(32768, pl.INT64), 32768), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = (
                            pl.tile.sort32(medium_scores_v1_inline974__ssa_v0, t__tmp_v3)
                        )
                        medium_pairs_v1_inline992__ssa_v0: pl.Tile[[1, 8192], pl.FP32, pl.MemRef(mem_vec_37, pl.const(98304, pl.INT64), 32768), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = (
                            pl.tile.mrgsort_format1(medium_pairs_inline970__ssa_v0, pl.const(64, pl.INT32))
                        )
                        medium_pairs_v2_inline994__ssa_v0: pl.Tile[[1, 8192], pl.FP32, pl.MemRef(mem_vec_36, pl.const(32768, pl.INT64), 32768), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = (
                            pl.tile.mrgsort_format1(medium_pairs_v1_inline992__ssa_v0, pl.const(256, pl.INT32))
                        )
                        medium_pairs_v3_inline984__ssa_v0: pl.Tile[[1, 8192], pl.FP32, pl.MemRef(mem_vec_37, pl.const(98304, pl.INT64), 32768), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = (
                            pl.tile.mrgsort_format1(medium_pairs_v2_inline994__ssa_v0, pl.const(1024, pl.INT32))
                        )
                        medium_left_inline995__ssa_v0: pl.Tile[[1, 1024], pl.FP32, pl.MemRef(mem_vec_37, pl.const(98304, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = (
                            pl.tile.slice(medium_pairs_v3_inline984__ssa_v0, [1, 1024], [0, 0])
                        )
                        medium_right_inline996__ssa_v0: pl.Tile[[1, 1024], pl.FP32, pl.MemRef(mem_vec_37, pl.const(114688, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = (
                            pl.tile.slice(medium_pairs_v3_inline984__ssa_v0, [1, 1024], [0, 4096])
                        )
                        medium_tmp_inline977__ssa_v0: pl.Tile[[1, 2048], pl.FP32, pl.MemRef(mem_vec_36, pl.const(32768, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.create(
                            [1, 2048], dtype=pl.FP32, target_memory=pl.Mem.Vec
                        )
                        medium_merged_inline973__ssa_v0: pl.Tile[[1, 2048], pl.FP32, pl.MemRef(mem_vec_34, pl.const(0, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = (
                            pl.tile.mrgsort_format2(medium_right_inline996__ssa_v0, medium_left_inline995__ssa_v0, medium_tmp_inline977__ssa_v0, exhausted=False)
                        )
                        medium_top_inline997__ssa_v0: pl.Tile[[1, 1024], pl.FP32, pl.MemRef(mem_vec_34, pl.const(0, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(
                            medium_merged_inline973__ssa_v0, [1, 1024], [0, 0]
                        )
                        medium_values_inline999__ssa_v0: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_36, pl.const(32768, pl.INT64), 2048), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = (
                            pl.tile.gather_mask(medium_top_inline997__ssa_v0, mask_pattern=1, output_dtype=pl.FP32)
                        )
                        medium_selected_inline1000__ssa_v0: pl.Tile[[1, 512], pl.INT32, pl.MemRef(mem_vec_37, pl.const(98304, pl.INT64), 2048), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = (
                            pl.tile.gather_mask(medium_top_inline997__ssa_v0, mask_pattern=2, output_dtype=pl.INT32)
                        )
                        pl.tile.store(medium_values_inline999__ssa_v0, [query__idx_v0, 0], topk_scores__ssa_v0)
                        pl.tile.store(medium_selected_inline1000__ssa_v0, [query__idx_v0, 0], topk_indices__ssa_v0)
                    else:
                        full_indices_inline1002__ci_tmp_v2: pl.Tile[[1, 192], pl.FP32, pl.MemRef(mem_vec_36, pl.const(32768, pl.INT64), 768), pl.Mem.Vec] = pl.tile.create(
                            [1, 192], dtype=pl.FP32, target_memory=pl.Mem.Vec
                        )
                        full_indices_inline1002__ssa_v0: pl.Tile[[1, 8192], pl.INT32, pl.MemRef(mem_vec_37, pl.const(98304, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.ci(
                            pl.const(0, pl.INT32), [1, 8192], tmp=full_indices_inline1002__ci_tmp_v2, dtype=pl.INT32, descending=False
                        )
                        full_raw_inline967__ssa_v0: pl.Tile[
                            [1, 8192], pl.FP32, pl.MemRef(mem_vec_36, pl.const(32768, pl.INT64), 32768), pl.Mem.Vec, pl.TileView(valid_shape=[1, visible_count__ssa_v0])
                        ] = pl.tile.load(score_arena__ssa_v0, [query__idx_v0, 0], [1, 8192], [1, visible_count__ssa_v0], target_memory=pl.Mem.Vec)
                        full_scores_inline965__ssa_v0: pl.Tile[[1, 8192], pl.FP32, pl.MemRef(mem_vec_36, pl.const(32768, pl.INT64), 32768), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = (
                            pl.tile.fillpad(full_raw_inline967__ssa_v0, pad_value=pl.PadValue.min)
                        )
                        full_min_inline993__ssa_v0: pl.Tile[[1, 8192], pl.FP32, pl.MemRef(mem_vec_34, pl.const(0, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.full(
                            [1, 8192], dtype=pl.FP32, value=-3.4028234663852886e38
                        )
                        full_scores_v1_inline964__ssa_v0: pl.Tile[[1, 8192], pl.FP32, pl.MemRef(mem_vec_34, pl.const(0, pl.INT64), 32768), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = (
                            pl.tile.maximum(full_scores_inline965__ssa_v0, full_min_inline993__ssa_v0)
                        )
                        t__tmp_v4: pl.Tile[[1, 8192], pl.UINT32, pl.MemRef(mem_vec_37, pl.const(98304, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.reinterpret_view(
                            full_indices_inline1002__ssa_v0, dtype=pl.UINT32
                        )
                        full_pairs_inline963__ssa_v0: pl.Tile[[1, 16384], pl.FP32, pl.MemRef(mem_vec_36, pl.const(32768, pl.INT64), 65536), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = (
                            pl.tile.sort32(full_scores_v1_inline964__ssa_v0, t__tmp_v4)
                        )
                        full_pairs_v1_inline972__ssa_v0: pl.Tile[[1, 16384], pl.FP32, pl.MemRef(mem_vec_37, pl.const(98304, pl.INT64), 65536), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = (
                            pl.tile.mrgsort_format1(full_pairs_inline963__ssa_v0, pl.const(64, pl.INT32))
                        )
                        full_pairs_v2_inline966__ssa_v0: pl.Tile[[1, 16384], pl.FP32, pl.MemRef(mem_vec_36, pl.const(32768, pl.INT64), 65536), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = (
                            pl.tile.mrgsort_format1(full_pairs_v1_inline972__ssa_v0, pl.const(256, pl.INT32))
                        )
                        full_pairs_v3_inline962__ssa_v0: pl.Tile[[1, 16384], pl.FP32, pl.MemRef(mem_vec_37, pl.const(98304, pl.INT64), 65536), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = (
                            pl.tile.mrgsort_format1(full_pairs_v2_inline966__ssa_v0, pl.const(1024, pl.INT32))
                        )
                        full_first_inline971__ssa_v0: pl.Tile[[1, 1024], pl.FP32, pl.MemRef(mem_vec_37, pl.const(98304, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(
                            full_pairs_v3_inline962__ssa_v0, [1, 1024], [0, 0]
                        )
                        full_second_inline986__ssa_v0: pl.Tile[[1, 1024], pl.FP32, pl.MemRef(mem_vec_37, pl.const(114688, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = (
                            pl.tile.slice(full_pairs_v3_inline962__ssa_v0, [1, 1024], [0, 4096])
                        )
                        full_third_inline969__ssa_v0: pl.Tile[[1, 1024], pl.FP32, pl.MemRef(mem_vec_37, pl.const(131072, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = (
                            pl.tile.slice(full_pairs_v3_inline962__ssa_v0, [1, 1024], [0, 8192])
                        )
                        full_fourth_inline985__ssa_v0: pl.Tile[[1, 1024], pl.FP32, pl.MemRef(mem_vec_37, pl.const(147456, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = (
                            pl.tile.slice(full_pairs_v3_inline962__ssa_v0, [1, 1024], [0, 12288])
                        )
                        full_tmp_inline961__ssa_v0: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_36, pl.const(32768, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.create(
                            [1, 4096], dtype=pl.FP32, target_memory=pl.Mem.Vec
                        )
                        full_merged_inline960__ssa_v0: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_34, pl.const(0, pl.INT64), 16384), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = (
                            pl.tile.mrgsort_format2(
                                full_fourth_inline985__ssa_v0, full_third_inline969__ssa_v0, full_second_inline986__ssa_v0, full_first_inline971__ssa_v0, full_tmp_inline961__ssa_v0, exhausted=False
                            )
                        )
                        full_top_inline959__ssa_v0: pl.Tile[[1, 1024], pl.FP32, pl.MemRef(mem_vec_34, pl.const(0, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(
                            full_merged_inline960__ssa_v0, [1, 1024], [0, 0]
                        )
                        full_values_inline958__ssa_v0: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_36, pl.const(32768, pl.INT64), 2048), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = (
                            pl.tile.gather_mask(full_top_inline959__ssa_v0, mask_pattern=1, output_dtype=pl.FP32)
                        )
                        full_selected_inline957__ssa_v0: pl.Tile[[1, 512], pl.INT32, pl.MemRef(mem_vec_37, pl.const(98304, pl.INT64), 2048), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = (
                            pl.tile.gather_mask(full_top_inline959__ssa_v0, mask_pattern=2, output_dtype=pl.INT32)
                        )
                        pl.tile.store(full_values_inline958__ssa_v0, [query__idx_v0, 0], topk_scores__ssa_v0)
                        pl.tile.store(full_selected_inline957__ssa_v0, [query__idx_v0, 0], topk_indices__ssa_v0)
            else:
                t__tmp_v5: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_36, pl.const(32768, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.full([1, 512], dtype=pl.FP32, value=-3.4028234663852886e38)
                pl.tile.store(t__tmp_v5, [query__idx_v0, 0], topk_scores__ssa_v0)
                t__tmp_v6: pl.Tile[[1, 512], pl.INT32, pl.MemRef(mem_vec_36, pl.const(32768, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.full([1, 512], dtype=pl.INT32, value=-1)
                pl.tile.store(t__tmp_v6, [query__idx_v0, 0], topk_indices__ssa_v0)
        return

    @pl.function(type=pl.FunctionType.Spmd)
    def _decode_csa_tp1_attention_indexer_topk_single_leaf_publish(
        self,
        position_ids__ssa_v0: pl.Tensor[[T_DYN], pl.INT64, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)],
        kv_seq_lens__ssa_v0: pl.Tensor[[B_DYN], pl.INT32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)],
        score_arena_inline1071_inline2347__rv_v2: pl.Tensor[[384, 8192], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 12582912)],
        idx_topk_scores__ssa_v0: pl.Out[pl.Tensor[[T_DYN, 512], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)]],
        idx_topk__ssa_v0: pl.Out[pl.Tensor[[T_DYN, 512], pl.INT32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 0)]],
    ):
        pl.func_attr({"mx_tensor_views_blocked": True})
        self.indexer_topk_single_leaf_publish(
            position_ids__ssa_v0,
            kv_seq_lens__ssa_v0,
            score_arena_inline1071_inline2347__rv_v2,
            idx_topk_scores__ssa_v0,
            idx_topk__ssa_v0,
            attrs={"arg_directions": [pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.output_existing, pl.adir.output_existing]},
        )

    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def compress_state_commit(
        compress_state__ssa_v0: pl.Out[pl.Tensor[[MAIN_STATE_BLOCK_NUM_DYN, STATE_PAGE_ELEMENTS_DYN], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)]],
        b_dim_inline2037__ssa_v0: pl.Scalar[pl.INDEX],
        commit_workers_inline2011__ssa_v0: pl.Scalar[pl.INDEX],
        s_dim_inline2013__ssa_v0: pl.Scalar[pl.INDEX],
        state_slot_mapping__ssa_v0: pl.Tensor[[T_DYN, 2], pl.INT32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)],
        position_ids__ssa_v0: pl.Tensor[[T_DYN], pl.INT64, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)],
        cmp4_kv_proj_pad_inline2035__ssa_v0: pl.Tensor[[384, 1024], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 1572864)],
        cmp4_score_proj_pad_inline2039__ssa_v0: pl.Tensor[[384, 1024], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 1572864)],
        cmp_ape__ssa_v0: pl.Tensor[[4, 1024], pl.FP32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 16384)],
    ):
        pl.func_attr({"mx_tensor_views_blocked": True})
        mem_vec_6: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 4096)
        mem_vec_8: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 4096)
        commit_worker_inline1998__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        for c_idx_inline2000__idx_v0, (compress_state__iter_v1,) in pl.range(
            commit_worker_inline1998__ssa_v0, b_dim_inline2037__ssa_v0, commit_workers_inline2011__ssa_v0, init_values=(compress_state__ssa_v0,)
        ):
            for s_idx_inline2019__idx_v0, (compress_state__iter_v3,) in pl.range(s_dim_inline2013__ssa_v0, init_values=(compress_state__iter_v1,)):
                token_inline2010__ssa_v0: pl.Scalar[pl.INDEX] = c_idx_inline2000__idx_v0 * s_dim_inline2013__ssa_v0 + s_idx_inline2019__idx_v0
                state_page_inline2001__tile: pl.Scalar[pl.INT32] = pl.tensor.read(state_slot_mapping__ssa_v0, [token_inline2010__ssa_v0, 0])
                state_offset_inline2005__tile: pl.Scalar[pl.INT32] = pl.tensor.read(state_slot_mapping__ssa_v0, [token_inline2010__ssa_v0, 1])
                if 0 <= pl.cast(state_page_inline2001__tile, pl.INDEX) and 0 <= pl.cast(state_offset_inline2005__tile, pl.INDEX):
                    token_pos_inline2007__tile: pl.Scalar[pl.INT64] = pl.tensor.read(position_ids__ssa_v0, [token_inline2010__ssa_v0])
                    native_page_inline2022__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(state_page_inline2001__tile, pl.INDEX)
                    native_column_inline2006__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(state_offset_inline2005__tile, pl.INDEX) * 2048
                    ape_row_inline2029__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(token_pos_inline2007__tile % 4, pl.INDEX)
                    t__tile: pl.Tile[[1, 1024], pl.FP32, pl.MemRef(mem_vec_6, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.load(
                        cmp4_kv_proj_pad_inline2035__ssa_v0, [token_inline2010__ssa_v0, 0], [1, 1024], [1, 1024], target_memory=pl.Mem.Vec
                    )
                    compress_state__tile: pl.Tensor[[MAIN_STATE_BLOCK_NUM_DYN, STATE_PAGE_ELEMENTS_DYN], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                        t__tile, [native_page_inline2022__ssa_v0, native_column_inline2006__ssa_v0], compress_state__iter_v3
                    )
                    t__tile_1: pl.Tile[[1, 1024], pl.FP32, pl.MemRef(mem_vec_6, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.load(
                        cmp4_score_proj_pad_inline2039__ssa_v0, [token_inline2010__ssa_v0, 0], [1, 1024], [1, 1024], target_memory=pl.Mem.Vec
                    )
                    t__tile_2: pl.Tile[[1, 1024], pl.FP32, pl.MemRef(mem_vec_8, pl.const(4096, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.load(
                        cmp_ape__ssa_v0, [ape_row_inline2029__ssa_v0, 0], [1, 1024], [1, 1024], target_memory=pl.Mem.Vec
                    )
                    t__tile_3: pl.Tile[[1, 1024], pl.FP32, pl.MemRef(mem_vec_6, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.add(t__tile_1, t__tile_2)
                    compress_state__tile_1: pl.Tensor[[MAIN_STATE_BLOCK_NUM_DYN, STATE_PAGE_ELEMENTS_DYN], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                        t__tile_3, [native_page_inline2022__ssa_v0, native_column_inline2006__ssa_v0 + 1024], compress_state__tile
                    )
                    compress_state__phi_v7: pl.Tensor[[MAIN_STATE_BLOCK_NUM_DYN, STATE_PAGE_ELEMENTS_DYN], pl.FP32, pl.MemRef("mem_ddr_10", pl.const(0, pl.INT64), 0)] = pl.yield_(
                        compress_state__tile_1
                    )
                else:
                    compress_state__phi_v7: pl.Tensor[[MAIN_STATE_BLOCK_NUM_DYN, STATE_PAGE_ELEMENTS_DYN], pl.FP32, pl.MemRef("mem_ddr_10", pl.const(0, pl.INT64), 0)] = pl.yield_(
                        compress_state__iter_v3
                    )
                compress_state__rv_v4: pl.Tensor[[MAIN_STATE_BLOCK_NUM_DYN, STATE_PAGE_ELEMENTS_DYN], pl.FP32, pl.MemRef("mem_ddr_11", pl.const(0, pl.INT64), 0)] = pl.yield_(compress_state__phi_v7)
            compress_state__rv_v2: pl.Tensor[[MAIN_STATE_BLOCK_NUM_DYN, STATE_PAGE_ELEMENTS_DYN], pl.FP32, pl.MemRef("mem_ddr_12", pl.const(0, pl.INT64), 0)] = pl.yield_(compress_state__rv_v4)
        return

    @pl.function(type=pl.FunctionType.Spmd)
    def compress_state_commit_spmd(
        self,
        compress_state__ssa_v0: pl.Out[pl.Tensor[[MAIN_STATE_BLOCK_NUM_DYN, STATE_PAGE_ELEMENTS_DYN], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)]],
        b_dim_inline2037__ssa_v0: pl.Scalar[pl.INDEX],
        commit_workers_inline2011__ssa_v0: pl.Scalar[pl.INDEX],
        s_dim_inline2013__ssa_v0: pl.Scalar[pl.INDEX],
        state_slot_mapping__ssa_v0: pl.Tensor[[T_DYN, 2], pl.INT32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)],
        position_ids__ssa_v0: pl.Tensor[[T_DYN], pl.INT64, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)],
        cmp4_kv_proj_pad_inline2035__ssa_v0: pl.Tensor[[384, 1024], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 1572864)],
        cmp4_score_proj_pad_inline2039__ssa_v0: pl.Tensor[[384, 1024], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 1572864)],
        cmp_ape__ssa_v0: pl.Tensor[[4, 1024], pl.FP32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 16384)],
    ):
        pl.func_attr({"mx_tensor_views_blocked": True})
        self.compress_state_commit(
            compress_state__ssa_v0,
            b_dim_inline2037__ssa_v0,
            commit_workers_inline2011__ssa_v0,
            s_dim_inline2013__ssa_v0,
            state_slot_mapping__ssa_v0,
            position_ids__ssa_v0,
            cmp4_kv_proj_pad_inline2035__ssa_v0,
            cmp4_score_proj_pad_inline2039__ssa_v0,
            cmp_ape__ssa_v0,
            attrs={"arg_directions": [pl.adir.output_existing, pl.adir.scalar, pl.adir.scalar, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input]},
        )

    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def compress_state_commit_0(
        inner_compress_state__ssa_v0: pl.Out[pl.Tensor[[INNER_STATE_BLOCK_NUM_DYN, INNER_STATE_PAGE_ELEMENTS_DYN], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)]],
        b_dim_inline454_inline2092__ssa_v0: pl.Scalar[pl.INDEX],
        commit_workers_inline450_inline2077__ssa_v0: pl.Scalar[pl.INDEX],
        s_dim_inline421_inline2137__ssa_v0: pl.Scalar[pl.INDEX],
        inner_state_slot_mapping__ssa_v0: pl.Tensor[[T_DYN, 2], pl.INT32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)],
        position_ids__ssa_v0: pl.Tensor[[T_DYN], pl.INT64, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)],
        kv_proj_pad_inline2105__rv_v2: pl.Tensor[[384, 256], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 393216)],
        score_proj_pad_inline2117__rv_v2: pl.Tensor[[384, 256], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 393216)],
        inner_ape__ssa_v0: pl.Tensor[[4, 256], pl.FP32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 4096)],
    ):
        pl.func_attr({"mx_tensor_views_blocked": True})
        mem_vec_6: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 1024)
        mem_vec_8: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 1024)
        commit_worker_inline476_inline2100__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        for c_idx_inline462_inline2106__idx_v0, (inner_compress_state__iter_v1,) in pl.range(
            commit_worker_inline476_inline2100__ssa_v0, b_dim_inline454_inline2092__ssa_v0, commit_workers_inline450_inline2077__ssa_v0, init_values=(inner_compress_state__ssa_v0,)
        ):
            for s_idx_inline432_inline2075__idx_v0, (inner_compress_state__iter_v3,) in pl.range(s_dim_inline421_inline2137__ssa_v0, init_values=(inner_compress_state__iter_v1,)):
                token_inline420_inline2157__ssa_v1: pl.Scalar[pl.INDEX] = c_idx_inline462_inline2106__idx_v0 * s_dim_inline421_inline2137__ssa_v0 + s_idx_inline432_inline2075__idx_v0
                state_page_inline413_inline2081__tile: pl.Scalar[pl.INT32] = pl.tensor.read(inner_state_slot_mapping__ssa_v0, [token_inline420_inline2157__ssa_v1, 0])
                state_offset_inline465_inline2099__tile: pl.Scalar[pl.INT32] = pl.tensor.read(inner_state_slot_mapping__ssa_v0, [token_inline420_inline2157__ssa_v1, 1])
                if 0 <= pl.cast(state_page_inline413_inline2081__tile, pl.INDEX) and 0 <= pl.cast(state_offset_inline465_inline2099__tile, pl.INDEX):
                    token_pos_inline431_inline2148__tile: pl.Scalar[pl.INT64] = pl.tensor.read(position_ids__ssa_v0, [token_inline420_inline2157__ssa_v1])
                    native_page_inline464_inline2155__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(state_page_inline413_inline2081__tile, pl.INDEX)
                    native_column_inline469_inline2074__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(state_offset_inline465_inline2099__tile, pl.INDEX) * 512
                    ape_row_inline475_inline2128__ssa_v1: pl.Scalar[pl.INDEX] = pl.cast(token_pos_inline431_inline2148__tile % 4, pl.INDEX)
                    t__tile: pl.Tile[[1, 256], pl.FP32, pl.MemRef(mem_vec_6, pl.const(0, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.load(
                        kv_proj_pad_inline2105__rv_v2, [token_inline420_inline2157__ssa_v1, 0], [1, 256], [1, 256], target_memory=pl.Mem.Vec
                    )
                    inner_compress_state__tile: pl.Tensor[[INNER_STATE_BLOCK_NUM_DYN, INNER_STATE_PAGE_ELEMENTS_DYN], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                        t__tile, [native_page_inline464_inline2155__ssa_v0, native_column_inline469_inline2074__ssa_v0], inner_compress_state__iter_v3
                    )
                    t__tile_1: pl.Tile[[1, 256], pl.FP32, pl.MemRef(mem_vec_6, pl.const(0, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.load(
                        score_proj_pad_inline2117__rv_v2, [token_inline420_inline2157__ssa_v1, 0], [1, 256], [1, 256], target_memory=pl.Mem.Vec
                    )
                    t__tile_2: pl.Tile[[1, 256], pl.FP32, pl.MemRef(mem_vec_8, pl.const(1024, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.load(
                        inner_ape__ssa_v0, [ape_row_inline475_inline2128__ssa_v1, 0], [1, 256], [1, 256], target_memory=pl.Mem.Vec
                    )
                    t__tile_3: pl.Tile[[1, 256], pl.FP32, pl.MemRef(mem_vec_6, pl.const(0, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.add(t__tile_1, t__tile_2)
                    inner_compress_state__tile_1: pl.Tensor[[INNER_STATE_BLOCK_NUM_DYN, INNER_STATE_PAGE_ELEMENTS_DYN], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                        t__tile_3, [native_page_inline464_inline2155__ssa_v0, native_column_inline469_inline2074__ssa_v0 + 256], inner_compress_state__tile
                    )
                    inner_compress_state__phi_v7: pl.Tensor[[INNER_STATE_BLOCK_NUM_DYN, INNER_STATE_PAGE_ELEMENTS_DYN], pl.FP32, pl.MemRef("mem_ddr_10", pl.const(0, pl.INT64), 0)] = pl.yield_(
                        inner_compress_state__tile_1
                    )
                else:
                    inner_compress_state__phi_v7: pl.Tensor[[INNER_STATE_BLOCK_NUM_DYN, INNER_STATE_PAGE_ELEMENTS_DYN], pl.FP32, pl.MemRef("mem_ddr_10", pl.const(0, pl.INT64), 0)] = pl.yield_(
                        inner_compress_state__iter_v3
                    )
                inner_compress_state__rv_v4: pl.Tensor[[INNER_STATE_BLOCK_NUM_DYN, INNER_STATE_PAGE_ELEMENTS_DYN], pl.FP32, pl.MemRef("mem_ddr_11", pl.const(0, pl.INT64), 0)] = pl.yield_(
                    inner_compress_state__phi_v7
                )
            inner_compress_state__rv_v2: pl.Tensor[[INNER_STATE_BLOCK_NUM_DYN, INNER_STATE_PAGE_ELEMENTS_DYN], pl.FP32, pl.MemRef("mem_ddr_12", pl.const(0, pl.INT64), 0)] = pl.yield_(
                inner_compress_state__rv_v4
            )
        return

    @pl.function(type=pl.FunctionType.Spmd)
    def compress_state_commit_spmd_0(
        self,
        inner_compress_state__ssa_v0: pl.Out[pl.Tensor[[INNER_STATE_BLOCK_NUM_DYN, INNER_STATE_PAGE_ELEMENTS_DYN], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)]],
        b_dim_inline454_inline2092__ssa_v0: pl.Scalar[pl.INDEX],
        commit_workers_inline450_inline2077__ssa_v0: pl.Scalar[pl.INDEX],
        s_dim_inline421_inline2137__ssa_v0: pl.Scalar[pl.INDEX],
        inner_state_slot_mapping__ssa_v0: pl.Tensor[[T_DYN, 2], pl.INT32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)],
        position_ids__ssa_v0: pl.Tensor[[T_DYN], pl.INT64, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)],
        kv_proj_pad_inline2105__rv_v2: pl.Tensor[[384, 256], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 393216)],
        score_proj_pad_inline2117__rv_v2: pl.Tensor[[384, 256], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 393216)],
        inner_ape__ssa_v0: pl.Tensor[[4, 256], pl.FP32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 4096)],
    ):
        pl.func_attr({"mx_tensor_views_blocked": True})
        self.compress_state_commit_0(
            inner_compress_state__ssa_v0,
            b_dim_inline454_inline2092__ssa_v0,
            commit_workers_inline450_inline2077__ssa_v0,
            s_dim_inline421_inline2137__ssa_v0,
            inner_state_slot_mapping__ssa_v0,
            position_ids__ssa_v0,
            kv_proj_pad_inline2105__rv_v2,
            score_proj_pad_inline2117__rv_v2,
            inner_ape__ssa_v0,
            attrs={"arg_directions": [pl.adir.output_existing, pl.adir.scalar, pl.adir.scalar, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input]},
        )

    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def csa_cache_writeback(
        kv_cache_flat__ssa_v0: pl.Out[pl.Tensor[[ORI_BLOCK_NUM_DYN * pl.const(32, pl.INDEX), 512], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)]],
        wb_blocks__ssa_v0: pl.Scalar[pl.INDEX],
        ori_slot_mapping__ssa_v0: pl.Tensor[[T_DYN, 2], pl.INT32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)],
        kv__ssa_v0: pl.Tensor[[T_DYN, 512], pl.BF16, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)],
    ):
        pl.func_attr({"mx_tensor_views_blocked": True})
        mem_vec_3: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 1024)
        wb_worker__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        for wb_blk__idx_v0, (kv_cache_flat__iter_v1,) in pl.range(wb_worker__ssa_v0, wb_blocks__ssa_v0, 8, init_values=(kv_cache_flat__ssa_v0,)):
            wb_t0__ssa_v0: pl.Scalar[pl.INDEX] = wb_blk__idx_v0 * 8
            for write_dt__idx_v0, (kv_cache_flat__iter_v3,) in pl.range(8, init_values=(kv_cache_flat__iter_v1,)):
                write_t__ssa_v0: pl.Scalar[pl.INDEX] = wb_t0__ssa_v0 + write_dt__idx_v0
                write_page__tile: pl.Scalar[pl.INT32] = pl.tensor.read(ori_slot_mapping__ssa_v0, [write_t__ssa_v0, 0])
                write_offset__tile: pl.Scalar[pl.INT32] = pl.tensor.read(ori_slot_mapping__ssa_v0, [write_t__ssa_v0, 1])
                if 0 <= pl.cast(write_page__tile, pl.INDEX) and 0 <= pl.cast(write_offset__tile, pl.INDEX):
                    write_row__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(write_page__tile, pl.INDEX) * 32 + pl.cast(write_offset__tile, pl.INDEX)
                    t__tile: pl.Tile[[1, 512], pl.BF16, pl.MemRef(mem_vec_3, pl.const(0, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.load(
                        kv__ssa_v0, [write_t__ssa_v0, 0], [1, 512], [1, 512], target_memory=pl.Mem.Vec
                    )
                    kv_cache_flat__tile: pl.Tensor[[ORI_BLOCK_NUM_DYN * pl.const(32, pl.INDEX), 512], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                        t__tile, [write_row__ssa_v0, 0], kv_cache_flat__iter_v3
                    )
                    kv_cache_flat__phi_v6: pl.Tensor[[ORI_BLOCK_NUM_DYN * pl.const(32, pl.INDEX), 512], pl.BF16, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 0)] = pl.yield_(kv_cache_flat__tile)
                else:
                    kv_cache_flat__phi_v6: pl.Tensor[[ORI_BLOCK_NUM_DYN * pl.const(32, pl.INDEX), 512], pl.BF16, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 0)] = pl.yield_(kv_cache_flat__iter_v3)
                kv_cache_flat__rv_v4: pl.Tensor[[ORI_BLOCK_NUM_DYN * pl.const(32, pl.INDEX), 512], pl.BF16, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 0)] = pl.yield_(kv_cache_flat__phi_v6)
            kv_cache_flat__rv_v2: pl.Tensor[[ORI_BLOCK_NUM_DYN * pl.const(32, pl.INDEX), 512], pl.BF16, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 0)] = pl.yield_(kv_cache_flat__rv_v4)
        return

    @pl.function(type=pl.FunctionType.Spmd)
    def csa_cache_writeback_spmd(
        self,
        kv_cache_flat__ssa_v0: pl.Out[pl.Tensor[[ORI_BLOCK_NUM_DYN * pl.const(32, pl.INDEX), 512], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)]],
        wb_blocks__ssa_v0: pl.Scalar[pl.INDEX],
        ori_slot_mapping__ssa_v0: pl.Tensor[[T_DYN, 2], pl.INT32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)],
        kv__ssa_v0: pl.Tensor[[T_DYN, 512], pl.BF16, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)],
    ):
        pl.func_attr({"mx_tensor_views_blocked": True})
        self.csa_cache_writeback(
            kv_cache_flat__ssa_v0, wb_blocks__ssa_v0, ori_slot_mapping__ssa_v0, kv__ssa_v0, attrs={"arg_directions": [pl.adir.output_existing, pl.adir.scalar, pl.adir.input, pl.adir.input]}
        )

    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def csa_rope_sign(
        cmp_seq_lens__ssa_v0: pl.Tensor[[B_DYN], pl.INT32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)],
        cmp_query_start_loc__ssa_v0: pl.Tensor[[QUERY_BOUNDS_DYN], pl.INT32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)],
        cmp_row_offsets__ssa_v0: pl.Out[pl.Tensor[[B_DYN], pl.INT32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)]],
        kv_seq_lens__ssa_v0: pl.Tensor[[B_DYN], pl.INT32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)],
        idx_query_start_loc__ssa_v0: pl.Tensor[[QUERY_BOUNDS_DYN], pl.INT32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 0)],
        idx_row_offsets__ssa_v0: pl.Out[pl.Tensor[[B_DYN], pl.INT32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 0)]],
        idx_sin_signed__ssa_v0: pl.Out[pl.Tensor[[T_DYN, 64], pl.FP32, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 0)]],
        T_DYN: pl.Scalar[pl.INDEX],
        freqs_sin__ssa_v0: pl.Tensor[[T_DYN, 64], pl.FP32, pl.MemRef("mem_ddr_7", pl.const(0, pl.INT64), 0)],
    ) -> pl.Tensor[[T_DYN, 64], pl.FP32]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        mem_vec_8: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 1024)
        mem_vec_10: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        mem_vec_13: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 1024)
        prefix_inline189__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(0, pl.INDEX)
        t__tmp_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(cmp_seq_lens__ssa_v0, 0)
        for request_inline188__idx_v0, (prefix_inline189__iter_v1,) in pl.range(t__tmp_v0, init_values=(prefix_inline189__ssa_v0,)):
            begin_inline187__tile: pl.Scalar[pl.INT32] = pl.tensor.read(cmp_query_start_loc__ssa_v0, [request_inline188__idx_v0])
            end_inline186__tile: pl.Scalar[pl.INT32] = pl.tensor.read(cmp_query_start_loc__ssa_v0, [request_inline188__idx_v0 + 1])
            length_inline185__tile: pl.Scalar[pl.INT32] = pl.tensor.read(cmp_seq_lens__ssa_v0, [request_inline188__idx_v0])
            start_inline184__ssa_v0: pl.Scalar[pl.INT32] = length_inline185__tile + begin_inline187__tile - end_inline186__tile
            pl.tensor.write(cmp_row_offsets__ssa_v0, [request_inline188__idx_v0], pl.cast(prefix_inline189__iter_v1 - pl.cast(start_inline184__ssa_v0, pl.INDEX) // 4 - 1, pl.INT32))
            prefix_inline189__ssa_v3: pl.Scalar[pl.INDEX] = prefix_inline189__iter_v1 + pl.cast(length_inline185__tile, pl.INDEX) // 4 - pl.cast(start_inline184__ssa_v0, pl.INDEX) // 4
            prefix_inline189__rv_v2: pl.Scalar[pl.INDEX] = pl.yield_(prefix_inline189__ssa_v3)
        prefix_inline195__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(0, pl.INDEX)
        t__tmp_v1: pl.Scalar[pl.INDEX] = pl.tensor.dim(kv_seq_lens__ssa_v0, 0)
        for request_inline194__idx_v0, (prefix_inline195__iter_v1,) in pl.range(t__tmp_v1, init_values=(prefix_inline195__ssa_v0,)):
            begin_inline193__tile: pl.Scalar[pl.INT32] = pl.tensor.read(idx_query_start_loc__ssa_v0, [request_inline194__idx_v0])
            end_inline192__tile: pl.Scalar[pl.INT32] = pl.tensor.read(idx_query_start_loc__ssa_v0, [request_inline194__idx_v0 + 1])
            length_inline191__tile: pl.Scalar[pl.INT32] = pl.tensor.read(kv_seq_lens__ssa_v0, [request_inline194__idx_v0])
            start_inline190__ssa_v0: pl.Scalar[pl.INT32] = length_inline191__tile + begin_inline193__tile - end_inline192__tile
            pl.tensor.write(idx_row_offsets__ssa_v0, [request_inline194__idx_v0], pl.cast(prefix_inline195__iter_v1 - pl.cast(start_inline190__ssa_v0, pl.INDEX) // 4 - 1, pl.INT32))
            prefix_inline195__ssa_v3: pl.Scalar[pl.INDEX] = prefix_inline195__iter_v1 + pl.cast(length_inline191__tile, pl.INDEX) // 4 - pl.cast(start_inline190__ssa_v0, pl.INDEX) // 4
            prefix_inline195__rv_v2: pl.Scalar[pl.INDEX] = pl.yield_(prefix_inline195__ssa_v3)
        il_ones__tile: pl.Tile[[4, 64], pl.FP32, pl.MemRef(mem_vec_8, pl.const(1280, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.full([4, 64], dtype=pl.FP32, value=1.0)
        t__ci_tmp_v0: pl.Tile[[1, 192], pl.FP32, pl.MemRef(mem_vec_13, pl.const(256, pl.INT64), 768), pl.Mem.Vec] = pl.tile.create([1, 192], dtype=pl.FP32, target_memory=pl.Mem.Vec)
        t__tile: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 256), pl.Mem.Vec] = pl.tile.ci(
            pl.const(0, pl.INT32), [1, 64], tmp=t__ci_tmp_v0, dtype=pl.INT32, descending=False
        )
        il_lane_ids__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(256, pl.INT64), 256), pl.Mem.Vec] = pl.tile.cast(t__tile, target_type=pl.FP32, mode="round")
        il_col__tile: pl.Tile[[4, 64], pl.FP32, pl.MemRef(mem_vec_8, pl.const(1280, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.col_expand_mul(il_ones__tile, il_lane_ids__tile)
        t__tile_1: pl.Tile[[4, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(256, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.muls(il_col__tile, 0.5)
        t__tile_2: pl.Tile[[4, 64], pl.INT32, pl.MemRef(mem_vec_13, pl.const(256, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.cast(t__tile_1, target_type=pl.INT32, mode="trunc")
        il_dup_f__tile: pl.Tile[[4, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(256, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.cast(t__tile_2, target_type=pl.FP32, mode="round")
        t__tile_3: pl.Tile[[4, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(256, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.muls(il_dup_f__tile, 2.0)
        il_lane__tile: pl.Tile[[4, 64], pl.FP32, pl.MemRef(mem_vec_8, pl.const(1280, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.sub(il_col__tile, t__tile_3)
        t__tile_4: pl.Tile[[4, 64], pl.FP32, pl.MemRef(mem_vec_8, pl.const(1280, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.muls(il_lane__tile, 2.0)
        il_sign__tile: pl.Tile[[4, 64], pl.FP32, pl.MemRef(mem_vec_8, pl.const(1280, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.subs(t__tile_4, 1.0)
        for rope_t0__idx_v0, (idx_sin_signed__iter_v1,) in pl.range(0, T_DYN, 4, init_values=(idx_sin_signed__ssa_v0,)):
            t__tile_5: pl.Tile[[4, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(256, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.load(
                freqs_sin__ssa_v0, [rope_t0__idx_v0, 0], [4, 64], [4, 64], target_memory=pl.Mem.Vec
            )
            t__tile_6: pl.Tile[[4, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(256, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.mul(t__tile_5, il_sign__tile)
            idx_sin_signed__tile: pl.Tensor[[T_DYN, 64], pl.FP32, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 0)] = pl.tile.store(t__tile_6, [rope_t0__idx_v0, 0], idx_sin_signed__iter_v1)
            idx_sin_signed__rv_v2: pl.Tensor[[T_DYN, 64], pl.FP32, pl.MemRef("mem_ddr_22", pl.const(0, pl.INT64), 0)] = pl.yield_(idx_sin_signed__tile)
        return idx_sin_signed__ssa_v0

    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def csa_slots_build_valid_qk_plan(
        cmp_sparse_indices_inline2484__ssa_v0: pl.InOut[pl.Tensor[[t_dim_inline2491__ssa_v0, 512], pl.INT32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)]],
        sparse_bias_inline2455__ssa_v0: pl.Out[pl.Tensor[[t_dim_inline2491__ssa_v0, 1024], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)]],
        t_dim_inline2491__ssa_v0: pl.Scalar[pl.INDEX],
        position_ids_t1__ssa_v0: pl.Tensor[[T_DYN, 1], pl.INT64, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)],
        idx_topk__ssa_v3: pl.Tensor[[T_DYN, 512], pl.INT32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)],
        ori_block_table__ssa_v0: pl.Tensor[[B_DYN, ORIGINAL_TABLE_COLUMNS_DYN], pl.INT32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 0)],
        valid_block_mask_inline2452__ssa_v0: pl.Out[pl.Tensor[[t_dim_inline2491__ssa_v0, 16], pl.INT32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 0)]],
    ) -> tuple[pl.Tensor[[t_dim_inline2491__ssa_v0, 512], pl.INT32], pl.Tensor[[t_dim_inline2491__ssa_v0, 1024], pl.FP32]]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        mem_vec_11: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 2048)
        mem_vec_24: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 512)
        mem_vec_28: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 512)
        mem_vec_44: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
        mem_vec_49: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
        plan_worker_inline2471__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        for bias_t0_inline2463__idx_v0, (cmp_sparse_indices_inline2484__iter_v1, sparse_bias_inline2455__iter_v1) in pl.range(
            plan_worker_inline2471__ssa_v0 * 8, t_dim_inline2491__ssa_v0, 128, init_values=(cmp_sparse_indices_inline2484__ssa_v0, sparse_bias_inline2455__ssa_v0)
        ):
            for bias_dt_inline2450__idx_v0, (cmp_sparse_indices_inline2484__iter_v3, sparse_bias_inline2455__iter_v3) in pl.range(
                8, init_values=(cmp_sparse_indices_inline2484__iter_v1, sparse_bias_inline2455__iter_v1)
            ):
                bias_t_inline2448__ssa_v0: pl.Scalar[pl.INDEX] = bias_t0_inline2463__idx_v0 + bias_dt_inline2450__idx_v0
                bias_request_inline2454__ssa_v0: pl.Scalar[pl.INDEX] = bias_t_inline2448__ssa_v0 // 6
                t__tile: pl.Scalar[pl.INT64] = pl.tensor.read(position_ids_t1__ssa_v0, [bias_t_inline2448__ssa_v0, 0])
                c_position_inline2495__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(t__tile, pl.INDEX)
                t__tile_1: pl.Tile[[1, 512], pl.INT32, pl.MemRef(mem_vec_44, pl.const(3072, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.load(
                    idx_topk__ssa_v3, [bias_t_inline2448__ssa_v0, 0], [1, 512], [1, 512], target_memory=pl.Mem.Vec
                )
                c_raw_inline2477__tile: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_44, pl.const(3072, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(t__tile_1, target_type=pl.FP32, mode="round")
                c_upper_inline2458__ssa_v0: pl.Scalar[pl.FP32] = pl.cast(pl.cast((c_position_inline2495__ssa_v0 + 1) // 4, pl.INT32), pl.FP32)
                t__tile_2: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_49, pl.const(19456, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.adds(c_raw_inline2477__tile, 1.0)
                t__tile_3: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_49, pl.const(19456, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.maximums(t__tile_2, 0.0)
                c_ge_inline2489__tile: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_49, pl.const(19456, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.minimums(t__tile_3, 1.0)
                t__tile_4: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_11, pl.const(0, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.neg(c_raw_inline2477__tile)
                t__tile_5: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_11, pl.const(0, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.adds(t__tile_4, c_upper_inline2458__ssa_v0)
                t__tile_6: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_11, pl.const(0, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.maximums(t__tile_5, 0.0)
                c_lt_inline2494__tile: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_11, pl.const(0, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.minimums(t__tile_6, 1.0)
                c_mask_inline2503__tile: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_49, pl.const(19456, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.mul(c_ge_inline2489__tile, c_lt_inline2494__tile)
                t__tile_7: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_44, pl.const(3072, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.adds(c_raw_inline2477__tile, 1.0)
                t__tile_8: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_44, pl.const(3072, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.mul(c_mask_inline2503__tile, t__tile_7)
                c_out_inline2475__tile: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_44, pl.const(3072, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.subs(t__tile_8, 1.0)
                t__tile_9: pl.Tile[[1, 512], pl.INT32, pl.MemRef(mem_vec_49, pl.const(19456, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(c_out_inline2475__tile, target_type=pl.INT32, mode="round")
                cmp_sparse_indices_inline2484__tile: pl.Tensor[[t_dim_inline2491__ssa_v0, 512], pl.INT32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                    t__tile_9, [bias_t_inline2448__ssa_v0, 0], cmp_sparse_indices_inline2484__iter_v3
                )
                v_length_inline2483__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(c_position_inline2495__ssa_v0 + 1, 128)
                v_start_inline2472__ssa_v0: pl.Scalar[pl.INDEX] = c_position_inline2495__ssa_v0 - v_length_inline2483__ssa_v0 + 1
                v_head_inline2441__ssa_v0: pl.Scalar[pl.INDEX] = v_start_inline2472__ssa_v0 % 32
                t__ci_tmp_v0: pl.Tile[[1, 192], pl.FP32, pl.MemRef(mem_vec_49, pl.const(19456, pl.INT64), 768), pl.Mem.Vec] = pl.tile.create([1, 192], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                t__tile_10: pl.Tile[[1, 128], pl.INT32, pl.MemRef(mem_vec_11, pl.const(0, pl.INT64), 512), pl.Mem.Vec] = pl.tile.ci(
                    pl.const(0, pl.INT32), [1, 128], tmp=t__ci_tmp_v0, dtype=pl.INT32, descending=False
                )
                v_columns_inline2492__tile: pl.Tile[[1, 128], pl.FP32, pl.MemRef(mem_vec_49, pl.const(19456, pl.INT64), 512), pl.Mem.Vec] = pl.tile.cast(t__tile_10, target_type=pl.FP32, mode="round")
                v_valid_inline2553__tile: pl.Tile[[1, 128], pl.FP32, pl.MemRef(mem_vec_11, pl.const(0, pl.INT64), 512), pl.Mem.Vec] = pl.tile.full([1, 128], dtype=pl.FP32, value=0.0)
                v_block_valid_inline2478__ssa_v0: pl.Scalar[pl.INT32] = pl.cast(0, pl.INT32)
                for v_run_inline2498__idx_v0, (v_block_valid_inline2478__iter_v1, v_valid_inline2553__iter_v1) in pl.range(5, init_values=(v_block_valid_inline2478__ssa_v0, v_valid_inline2553__tile)):
                    v_lo_inline2479__ssa_v0: pl.Scalar[pl.INDEX] = pl.max(v_run_inline2498__idx_v0 * 32 - v_head_inline2441__ssa_v0, 0)
                    v_hi_inline2446__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(v_run_inline2498__idx_v0 * 32 - v_head_inline2441__ssa_v0 + 32, v_length_inline2483__ssa_v0)
                    if v_lo_inline2479__ssa_v0 < v_hi_inline2446__ssa_v0:
                        v_page_inline2474__tile: pl.Scalar[pl.INT32] = pl.tensor.read(
                            ori_block_table__ssa_v0, [bias_request_inline2454__ssa_v0, (v_start_inline2472__ssa_v0 + v_lo_inline2479__ssa_v0) // 32]
                        )
                        if 0 <= pl.cast(v_page_inline2474__tile, pl.INDEX):
                            v_block_valid_inline2478__ssa_v3: pl.Scalar[pl.INT32] = pl.cast(1, pl.INT32)
                            v_lo_fp32_inline2445__ssa_v0: pl.Scalar[pl.FP32] = pl.cast(pl.cast(v_lo_inline2479__ssa_v0, pl.INT32), pl.FP32)
                            v_hi_fp32_inline2547__ssa_v0: pl.Scalar[pl.FP32] = pl.cast(pl.cast(v_hi_inline2446__ssa_v0, pl.INT32), pl.FP32)
                            t__tile_11: pl.Tile[[1, 128], pl.FP32, pl.MemRef(mem_vec_24, pl.const(2048, pl.INT64), 512), pl.Mem.Vec] = pl.tile.subs(
                                v_columns_inline2492__tile, v_lo_fp32_inline2445__ssa_v0
                            )
                            t__tile_12: pl.Tile[[1, 128], pl.FP32, pl.MemRef(mem_vec_24, pl.const(2048, pl.INT64), 512), pl.Mem.Vec] = pl.tile.adds(t__tile_11, 1.0)
                            t__tile_13: pl.Tile[[1, 128], pl.FP32, pl.MemRef(mem_vec_24, pl.const(2048, pl.INT64), 512), pl.Mem.Vec] = pl.tile.maximums(t__tile_12, 0.0)
                            v_ge_inline2497__tile: pl.Tile[[1, 128], pl.FP32, pl.MemRef(mem_vec_24, pl.const(2048, pl.INT64), 512), pl.Mem.Vec] = pl.tile.minimums(t__tile_13, 1.0)
                            t__tile_14: pl.Tile[[1, 128], pl.FP32, pl.MemRef(mem_vec_28, pl.const(2560, pl.INT64), 512), pl.Mem.Vec] = pl.tile.neg(v_columns_inline2492__tile)
                            t__tile_15: pl.Tile[[1, 128], pl.FP32, pl.MemRef(mem_vec_28, pl.const(2560, pl.INT64), 512), pl.Mem.Vec] = pl.tile.adds(t__tile_14, v_hi_fp32_inline2547__ssa_v0)
                            t__tile_16: pl.Tile[[1, 128], pl.FP32, pl.MemRef(mem_vec_28, pl.const(2560, pl.INT64), 512), pl.Mem.Vec] = pl.tile.maximums(t__tile_15, 0.0)
                            v_lt_inline2443__tile: pl.Tile[[1, 128], pl.FP32, pl.MemRef(mem_vec_28, pl.const(2560, pl.INT64), 512), pl.Mem.Vec] = pl.tile.minimums(t__tile_16, 1.0)
                            t__tile_17: pl.Tile[[1, 128], pl.FP32, pl.MemRef(mem_vec_24, pl.const(2048, pl.INT64), 512), pl.Mem.Vec] = pl.tile.mul(v_ge_inline2497__tile, v_lt_inline2443__tile)
                            v_valid_inline2553__tile_1: pl.Tile[[1, 128], pl.FP32, pl.MemRef(mem_vec_11, pl.const(0, pl.INT64), 512), pl.Mem.Vec] = pl.tile.add(v_valid_inline2553__iter_v1, t__tile_17)
                            v_block_valid_inline2478__phi_v4, v_valid_inline2553__phi_v4 = pl.yield_(v_block_valid_inline2478__ssa_v3, v_valid_inline2553__tile_1)
                        else:
                            v_block_valid_inline2478__phi_v4, v_valid_inline2553__phi_v4 = pl.yield_(v_block_valid_inline2478__iter_v1, v_valid_inline2553__iter_v1)
                        v_block_valid_inline2478__phi_v5, v_valid_inline2553__phi_v5 = pl.yield_(v_block_valid_inline2478__phi_v4, v_valid_inline2553__phi_v4)
                    else:
                        v_block_valid_inline2478__phi_v5, v_valid_inline2553__phi_v5 = pl.yield_(v_block_valid_inline2478__iter_v1, v_valid_inline2553__iter_v1)
                    v_block_valid_inline2478__rv_v2, v_valid_inline2553__rv_v2 = pl.yield_(v_block_valid_inline2478__phi_v5, v_valid_inline2553__phi_v5)
                pl.tensor.write(valid_block_mask_inline2452__ssa_v0, [bias_t_inline2448__ssa_v0, 0], v_block_valid_inline2478__rv_v2)
                t__tile_18: pl.Tile[[1, 128], pl.FP32, pl.MemRef(mem_vec_49, pl.const(19456, pl.INT64), 512), pl.Mem.Vec] = pl.tile.subs(v_valid_inline2553__rv_v2, 1.0)
                t__tile_19: pl.Tile[[1, 128], pl.FP32, pl.MemRef(mem_vec_49, pl.const(19456, pl.INT64), 512), pl.Mem.Vec] = pl.tile.muls(t__tile_18, 1e20)
                sparse_bias_inline2455__tile: pl.Tensor[[t_dim_inline2491__ssa_v0, 1024], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                    t__tile_19, [bias_t_inline2448__ssa_v0, 0], sparse_bias_inline2455__iter_v3
                )
                t__tile_20: pl.Tile[[1, 384], pl.FP32, pl.MemRef(mem_vec_49, pl.const(19456, pl.INT64), 1536), pl.Mem.Vec] = pl.tile.full([1, 384], dtype=pl.FP32, value=-1e20)
                sparse_bias_inline2455__tile_1: pl.Tensor[[t_dim_inline2491__ssa_v0, 1024], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                    t__tile_20, [bias_t_inline2448__ssa_v0, 128], sparse_bias_inline2455__tile
                )
                t__tile_21: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_44, pl.const(3072, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.minimums(c_out_inline2475__tile, 0.0)
                t__tile_22: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_44, pl.const(3072, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.muls(t__tile_21, 1e20)
                sparse_bias_inline2455__tile_2: pl.Tensor[[t_dim_inline2491__ssa_v0, 1024], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                    t__tile_22, [bias_t_inline2448__ssa_v0, 512], sparse_bias_inline2455__tile_1
                )
                cmp_sparse_indices_inline2484__rv_v4, sparse_bias_inline2455__rv_v4 = pl.yield_(cmp_sparse_indices_inline2484__tile, sparse_bias_inline2455__tile_2)
            c_s0_inline2461__ssa_v0: pl.Scalar[pl.INDEX] = 0
            t__tile_23: pl.Tile[[8, 512], pl.INT32, pl.MemRef(mem_vec_44, pl.const(3072, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.load(
                cmp_sparse_indices_inline2484__rv_v4, [bias_t0_inline2463__idx_v0, c_s0_inline2461__ssa_v0], [8, 512], [8, 512], target_memory=pl.Mem.Vec
            )
            c_block_slots_inline2436__tile: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_44, pl.const(3072, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.cast(t__tile_23, target_type=pl.FP32, mode="round")
            t__tile_24: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_44, pl.const(3072, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.adds(c_block_slots_inline2436__tile, 1.0)
            t__tile_25: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_44, pl.const(3072, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.maximums(t__tile_24, 0.0)
            t__tile_26: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_44, pl.const(3072, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.minimums(t__tile_25, 1.0)
            tmp_tile: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_49, pl.const(19456, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.create([8, 512], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            c_blk_valid_inline2435__tile: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_11, pl.const(0, pl.INT64), 32), pl.Mem.Vec] = pl.tile.row_max(t__tile_26, tmp_tile)
            for c_dt_inline2466__idx_v0 in pl.range(8):
                t__tile_27: pl.Scalar[pl.FP32] = pl.tile.read(c_blk_valid_inline2435__tile, [c_dt_inline2466__idx_v0, 0])
                c_valid_inline2431__ssa_v0: pl.Scalar[pl.INT32] = pl.cast(t__tile_27, pl.INT32)
                pl.tensor.write(valid_block_mask_inline2452__ssa_v0, [bias_t0_inline2463__idx_v0 + c_dt_inline2466__idx_v0, 1], c_valid_inline2431__ssa_v0)
            cmp_sparse_indices_inline2484__rv_v2, sparse_bias_inline2455__rv_v2 = pl.yield_(cmp_sparse_indices_inline2484__rv_v4, sparse_bias_inline2455__rv_v4)
        return cmp_sparse_indices_inline2484__ssa_v0, sparse_bias_inline2455__ssa_v0

    @pl.function(type=pl.FunctionType.Spmd)
    def csa_slots_build_valid_qk_plan_spmd(
        self,
        cmp_sparse_indices_inline2484__ssa_v0: pl.InOut[pl.Tensor[[t_dim_inline2491__ssa_v0, 512], pl.INT32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)]],
        sparse_bias_inline2455__ssa_v0: pl.Out[pl.Tensor[[t_dim_inline2491__ssa_v0, 1024], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)]],
        t_dim_inline2491__ssa_v0: pl.Scalar[pl.INDEX],
        position_ids_t1__ssa_v0: pl.Tensor[[T_DYN, 1], pl.INT64, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)],
        idx_topk__ssa_v3: pl.Tensor[[T_DYN, 512], pl.INT32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)],
        ori_block_table__ssa_v0: pl.Tensor[[B_DYN, ORIGINAL_TABLE_COLUMNS_DYN], pl.INT32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 0)],
        valid_block_mask_inline2452__ssa_v0: pl.Out[pl.Tensor[[t_dim_inline2491__ssa_v0, 16], pl.INT32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 0)]],
    ) -> tuple[pl.Tensor[[t_dim_inline2491__ssa_v0, 512], pl.INT32], pl.Tensor[[t_dim_inline2491__ssa_v0, 1024], pl.FP32]]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        ret__tmp_v0: pl.Tuple[pl.Tensor[[t_dim_inline2491__ssa_v0, 512], pl.INT32], pl.Tensor[[t_dim_inline2491__ssa_v0, 1024], pl.FP32]] = self.csa_slots_build_valid_qk_plan(
            cmp_sparse_indices_inline2484__ssa_v0,
            sparse_bias_inline2455__ssa_v0,
            t_dim_inline2491__ssa_v0,
            position_ids_t1__ssa_v0,
            idx_topk__ssa_v3,
            ori_block_table__ssa_v0,
            valid_block_mask_inline2452__ssa_v0,
            attrs={"arg_directions": [pl.adir.inout, pl.adir.output_existing, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.output_existing]},
        )
        cmp_sparse_indices_inline2484__rv_v2: pl.Tensor[[t_dim_inline2491__ssa_v0, 512], pl.INT32, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 0)] = ret__tmp_v0[0]
        sparse_bias_inline2455__rv_v2: pl.Tensor[[t_dim_inline2491__ssa_v0, 1024], pl.FP32, pl.MemRef("mem_ddr_7", pl.const(0, pl.INT64), 0)] = ret__tmp_v0[1]
        return cmp_sparse_indices_inline2484__ssa_v0, sparse_bias_inline2455__ssa_v0

    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def idx_kv_scale_commit(
        compact_rows_inline2207__ssa_v0: pl.Scalar[pl.INDEX],
        position_ids__ssa_v0: pl.Tensor[[T_DYN], pl.INT64, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)],
        idx_row_offsets__ssa_v0: pl.Tensor[[B_DYN], pl.INT32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)],
        idx_slot_mapping__ssa_v0: pl.Tensor[[INDEXER_ROWS_DYN, 2], pl.INT32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)],
        idx_kv_cache__rv_v2: pl.InOut[pl.Tensor[[IDX_CACHE_BLOCK_NUM_DYN, INDEXER_PAGE_BYTES_DYN], pl.INT8, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)]],
        idx_kv_scale_values_inline2188__ssa_v1: pl.Tensor[[384, 1], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 1536)],
    ) -> pl.Tensor[[IDX_CACHE_BLOCK_NUM_DYN, INDEXER_PAGE_BYTES_DYN], pl.INT8]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        mem_vec_5: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 64)
        for compact_token_inline2195__idx_v0 in pl.range(compact_rows_inline2207__ssa_v0):
            request_inline2178__ssa_v1: pl.Scalar[pl.INDEX] = compact_token_inline2195__idx_v0 // 2
            first_pos_inline2177__tile: pl.Scalar[pl.INT64] = pl.tensor.read(position_ids__ssa_v0, [request_inline2178__ssa_v1 * 6])
            local_token_inline2176__ssa_v1: pl.Scalar[pl.INDEX] = compact_token_inline2195__idx_v0 % 2 * 4 - first_pos_inline2177__tile % 4 + 3
            if local_token_inline2176__ssa_v1 < 6:
                token_v1_inline2170__ssa_v0: pl.Scalar[pl.INDEX] = request_inline2178__ssa_v1 * 6 + local_token_inline2176__ssa_v1
                token_pos_v1_inline2214__tile: pl.Scalar[pl.INT64] = pl.tensor.read(position_ids__ssa_v0, [token_v1_inline2170__ssa_v0])
                t__tile: pl.Scalar[pl.INT32] = pl.tensor.read(idx_row_offsets__ssa_v0, [request_inline2178__ssa_v1])
                metadata_row_v1_inline2197__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(t__tile, pl.INDEX) + pl.cast((token_pos_v1_inline2214__tile + 1) // 4, pl.INDEX)
                native_page_v1_inline2169__tile: pl.Scalar[pl.INT32] = pl.tensor.read(idx_slot_mapping__ssa_v0, [metadata_row_v1_inline2197__ssa_v0, 0])
                native_offset_v1_inline2167__tile: pl.Scalar[pl.INT32] = pl.tensor.read(idx_slot_mapping__ssa_v0, [metadata_row_v1_inline2197__ssa_v0, 1])
                if 0 <= pl.cast(native_page_v1_inline2169__tile, pl.INDEX) and 0 <= pl.cast(native_offset_v1_inline2167__tile, pl.INDEX):
                    cache_row_v1_inline2201__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(native_page_v1_inline2169__tile, pl.INDEX) * 32 + pl.cast(native_offset_v1_inline2167__tile, pl.INDEX)
                    scale_page_inline2168__ssa_v0: pl.Scalar[pl.INDEX] = cache_row_v1_inline2201__ssa_v0 // 32
                    scale_bytes_inline2217__ssa_v0: pl.Tile[[1, 64], pl.INT8, pl.MemRef(mem_vec_5, pl.const(0, pl.INT64), 64), pl.Mem.Vec] = pl.tile.load(
                        idx_kv_cache__rv_v2, [scale_page_inline2168__ssa_v0, 4096], [1, 64], [1, 64], target_memory=pl.Mem.Vec
                    )
                    scale_half_inline2192__ssa_v0: pl.Tile[[1, 32], pl.FP16, pl.MemRef(mem_vec_5, pl.const(0, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reinterpret_view(
                        scale_bytes_inline2217__ssa_v0, dtype=pl.FP16
                    )
                    t__tile_1: pl.Scalar[pl.FP32] = pl.tensor.read(idx_kv_scale_values_inline2188__ssa_v1, [compact_token_inline2195__idx_v0, 0])
                    pl.tile.write(scale_half_inline2192__ssa_v0, [0, cache_row_v1_inline2201__ssa_v0 % 32], pl.cast(t__tile_1, pl.FP16))
                    updated_bytes_inline2166__ssa_v0: pl.Tile[[1, 64], pl.INT8, pl.MemRef(mem_vec_5, pl.const(0, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reinterpret_view(
                        scale_half_inline2192__ssa_v0, dtype=pl.INT8
                    )
                    idx_kv_cache__store: pl.Tensor[[IDX_CACHE_BLOCK_NUM_DYN, INDEXER_PAGE_BYTES_DYN], pl.INT8, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                        updated_bytes_inline2166__ssa_v0, [scale_page_inline2168__ssa_v0, 4096], idx_kv_cache__rv_v2
                    )
        return idx_kv_cache__rv_v2

    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def idx_qr_dequant_rope(
        qr_bf16_2d_inline914_inline2247__ssa_v0: pl.Out[pl.Tensor[[384, 8192], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 6291456)]],
        dq_rope_units_inline905_inline2241__ssa_v0: pl.Scalar[pl.INDEX],
        dq_rope_workers_inline918_inline2238__ssa_v0: pl.Scalar[pl.INDEX],
        qr_scale__ssa_v0: pl.Tensor[[T_DYN, 1], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)],
        freqs_cos__ssa_v0: pl.Tensor[[T_DYN, 64], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)],
        idx_sin_signed__rv_v2: pl.Tensor[[T_DYN, 64], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)],
        idx_wq_b_scale__ssa_v0: pl.Tensor[[8192], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 32768)],
        qr_acc_pad_inline908_inline2268__rv_v2: pl.Tensor[[384, 8192], pl.INT32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 12582912)],
    ):
        pl.func_attr({"mx_tensor_views_blocked": True})
        mem_vec_19: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 2048)
        mem_vec_20: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32)
        mem_vec_21: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 2048)
        mem_vec_22: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 2048)
        mem_vec_24: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 4096)
        mem_vec_26: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 4096)
        mem_vec_28: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 4096)
        mem_vec_34: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 1024)
        mem_vec_36: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        mem_vec_37: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        mem_vec_44: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 4096)
        mem_vec_50: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 1024)
        mem_vec_52: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        mem_vec_53: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        dq_rope_worker_inline901_inline2262__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        sw_ones_inline919_inline2291__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_24, pl.const(6176, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.full([8, 64], dtype=pl.FP32, value=1.0)
        t__ci_tmp_v0: pl.Tile[[1, 192], pl.FP32, pl.MemRef(mem_vec_26, pl.const(10272, pl.INT64), 768), pl.Mem.Vec] = pl.tile.create([1, 192], dtype=pl.FP32, target_memory=pl.Mem.Vec)
        t__tile: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_28, pl.const(14368, pl.INT64), 256), pl.Mem.Vec] = pl.tile.ci(
            pl.const(0, pl.INT32), [1, 64], tmp=t__ci_tmp_v0, dtype=pl.INT32, descending=False
        )
        sw_index_inline893_inline2248__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_26, pl.const(10272, pl.INT64), 256), pl.Mem.Vec] = pl.tile.cast(t__tile, target_type=pl.FP32, mode="round")
        sw_col_inline920_inline2239__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_24, pl.const(6176, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.col_expand_mul(
            sw_ones_inline919_inline2291__tile, sw_index_inline893_inline2248__tile
        )
        t__tile_1: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_26, pl.const(10272, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.muls(sw_col_inline920_inline2239__tile, 0.5)
        t__tile_2: pl.Tile[[8, 64], pl.INT32, pl.MemRef(mem_vec_26, pl.const(10272, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(t__tile_1, target_type=pl.INT32, mode="trunc")
        sw_dup_f_inline909_inline2235__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_26, pl.const(10272, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(t__tile_2, target_type=pl.FP32, mode="round")
        t__tile_3: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_26, pl.const(10272, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.muls(sw_dup_f_inline909_inline2235__tile, 2.0)
        sw_lane_inline924_inline2234__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_26, pl.const(10272, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.sub(sw_col_inline920_inline2239__tile, t__tile_3)
        t__tile_4: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_24, pl.const(6176, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.adds(sw_col_inline920_inline2239__tile, 1.0)
        t__tile_5: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_26, pl.const(10272, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.muls(sw_lane_inline924_inline2234__tile, 2.0)
        t__tile_6: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_24, pl.const(6176, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.sub(t__tile_4, t__tile_5)
        rope_swap_idx_inline925_inline2276__tile: pl.Tile[[8, 64], pl.INT32, pl.MemRef(mem_vec_19, pl.const(0, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(
            t__tile_6, target_type=pl.INT32, mode="round"
        )
        for dq_unit_inline915_inline2242__idx_v0, (qr_bf16_2d_inline914_inline2247__iter_v1,) in pl.range(
            dq_rope_worker_inline901_inline2262__ssa_v0,
            dq_rope_units_inline905_inline2241__ssa_v0,
            dq_rope_workers_inline918_inline2238__ssa_v0,
            init_values=(qr_bf16_2d_inline914_inline2247__ssa_v0,),
        ):
            hg_inline923_inline2281__ssa_v0: pl.Scalar[pl.INDEX] = dq_unit_inline915_inline2242__idx_v0 % 16 * 4
            dq_t0_inline895_inline2243__ssa_v0: pl.Scalar[pl.INDEX] = dq_unit_inline915_inline2242__idx_v0 // 16 * 8
            qr_scale_tile_inline889_inline2237__tile: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_20, pl.const(2048, pl.INT64), 32), pl.Mem.Vec] = pl.tile.load(
                qr_scale__ssa_v0, [dq_t0_inline895_inline2243__ssa_v0, 0], [8, 1], [8, 1], target_memory=pl.Mem.Vec
            )
            cos_tile_inline891_inline2284__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_21, pl.const(2080, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.load(
                freqs_cos__ssa_v0, [dq_t0_inline895_inline2243__ssa_v0, 0], [8, 64], [8, 64], target_memory=pl.Mem.Vec
            )
            sin_tile_inline921_inline2285__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_22, pl.const(4128, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.load(
                idx_sin_signed__rv_v2, [dq_t0_inline895_inline2243__ssa_v0, 0], [8, 64], [8, 64], target_memory=pl.Mem.Vec
            )
            for h_inner_inline888_inline2282__idx_v0, (qr_bf16_2d_inline914_inline2247__iter_v3,) in pl.range(0, 4, 2, init_values=(qr_bf16_2d_inline914_inline2247__iter_v1,)):
                h0_inline887_inline2271__ssa_v0: pl.Scalar[pl.INDEX] = (hg_inline923_inline2281__ssa_v0 + h_inner_inline888_inline2282__idx_v0) * 128
                h0_inline887_inline2271__ssa_v0_1: pl.Scalar[pl.INDEX] = (hg_inline923_inline2281__ssa_v0 + (h_inner_inline888_inline2282__idx_v0 + 1)) * 128
                t__tile_7: pl.Tile[[128], pl.FP32, pl.MemRef(mem_vec_34, pl.const(18464, pl.INT64), 512), pl.Mem.Vec] = pl.tile.load(
                    idx_wq_b_scale__ssa_v0, [h0_inline887_inline2271__ssa_v0], [128], [128], target_memory=pl.Mem.Vec
                )
                t__tile_8: pl.Tile[[8, 128], pl.INT32, pl.MemRef(mem_vec_24, pl.const(6176, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.load(
                    qr_acc_pad_inline908_inline2268__rv_v2, [dq_t0_inline895_inline2243__ssa_v0, h0_inline887_inline2271__ssa_v0], [8, 128], [8, 128], target_memory=pl.Mem.Vec
                )
                t__tile_9: pl.Tile[[128], pl.FP32, pl.MemRef(mem_vec_50, pl.const(24096, pl.INT64), 512), pl.Mem.Vec] = pl.tile.load(
                    idx_wq_b_scale__ssa_v0, [h0_inline887_inline2271__ssa_v0_1], [128], [128], target_memory=pl.Mem.Vec
                )
                t__tile_10: pl.Tile[[8, 128], pl.INT32, pl.MemRef(mem_vec_26, pl.const(10272, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.load(
                    qr_acc_pad_inline908_inline2268__rv_v2, [dq_t0_inline895_inline2243__ssa_v0, h0_inline887_inline2271__ssa_v0_1], [8, 128], [8, 128], target_memory=pl.Mem.Vec
                )
                wq_scale_inline886_inline2266__tile: pl.Tile[[1, 128], pl.FP32, pl.MemRef(mem_vec_34, pl.const(18464, pl.INT64), 512), pl.Mem.Vec] = t__tile_7
                acc_fp32_inline900_inline2286__tile: pl.Tile[[8, 128], pl.FP32, pl.MemRef(mem_vec_24, pl.const(6176, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(
                    t__tile_8, target_type=pl.FP32, mode="none"
                )
                t__tile_11: pl.Tile[[8, 128], pl.FP32, pl.MemRef(mem_vec_28, pl.const(14368, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.full([8, 128], dtype=pl.FP32, value=1.0)
                t__tile_12: pl.Tile[[8, 128], pl.FP32, pl.MemRef(mem_vec_28, pl.const(14368, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.row_expand_mul(
                    t__tile_11, qr_scale_tile_inline889_inline2237__tile
                )
                qr_dequant_scale_inline885_inline2245__tile: pl.Tile[[8, 128], pl.FP32, pl.MemRef(mem_vec_28, pl.const(14368, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.col_expand_mul(
                    t__tile_12, wq_scale_inline886_inline2266__tile
                )
                qr_dequant_inline884_inline2258__tile: pl.Tile[[8, 128], pl.FP32, pl.MemRef(mem_vec_24, pl.const(6176, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.mul(
                    acc_fp32_inline900_inline2286__tile, qr_dequant_scale_inline885_inline2245__tile
                )
                t__tile_13: pl.Tile[[8, 128], pl.BF16, pl.MemRef(mem_vec_28, pl.const(14368, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(
                    qr_dequant_inline884_inline2258__tile, target_type=pl.BF16, mode="rint"
                )
                qr_dequant_v1_inline917_inline2287__tile: pl.Tile[[8, 128], pl.FP32, pl.MemRef(mem_vec_24, pl.const(6176, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(
                    t__tile_13, target_type=pl.FP32, mode="round"
                )
                t__tile_14: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_24, pl.const(6176, pl.INT64), 3840), pl.Mem.Vec] = pl.tile.slice(qr_dequant_v1_inline917_inline2287__tile, [8, 64], [0, 0])
                qr_nope_bf16_inline883_inline2288__tile: pl.Tile[[8, 64], pl.BF16, pl.MemRef(mem_vec_34, pl.const(18464, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.cast(
                    t__tile_14, target_type=pl.BF16, mode="rint"
                )
                qr_rope_slice_inline882_inline2289__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_24, pl.const(6432, pl.INT64), 3840), pl.Mem.Vec] = pl.tile.slice(
                    qr_dequant_v1_inline917_inline2287__tile, [8, 64], [0, 64]
                )
                gather_acc_init: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_28, pl.const(14368, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.create([8, 64], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                for gather_lv, (gather_ia,) in pl.range(8, init_values=(gather_acc_init,)):
                    gather_inp_row: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_24, pl.const(6176, pl.INT64), 256), pl.Mem.Vec] = pl.tile.slice(
                        qr_rope_slice_inline882_inline2289__tile, [1, 64], [gather_lv, 0], [1, 64]
                    )
                    gather_idx_row: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_19, pl.const(0, pl.INT64), 256), pl.Mem.Vec] = pl.tile.slice(
                        rope_swap_idx_inline925_inline2276__tile, [1, 64], [gather_lv, 0], [1, 64]
                    )
                    gather_row_tmp: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_36, pl.const(19488, pl.INT64), 256), pl.Mem.Vec] = pl.tile.create([1, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec)
                    gather_row: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_37, pl.const(19744, pl.INT64), 256), pl.Mem.Vec] = pl.tile.gather(gather_inp_row, gather_idx_row, gather_row_tmp)
                    gather_asmbl: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_28, pl.const(14368, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.assemble(gather_ia, gather_row, [gather_lv, 0])
                    gather_rv: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_28, pl.const(14368, pl.INT64), 2048), pl.Mem.Vec] = pl.yield_(gather_asmbl)
                qr_swapped_inline890_inline2279__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_28, pl.const(14368, pl.INT64), 2048), pl.Mem.Vec] = gather_rv
                t__tile_15: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_24, pl.const(6176, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.mul(
                    qr_rope_slice_inline882_inline2289__tile, cos_tile_inline891_inline2284__tile
                )
                t__tile_16: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_28, pl.const(14368, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.mul(
                    qr_swapped_inline890_inline2279__tile, sin_tile_inline921_inline2285__tile
                )
                rope_rot_inline881_inline2292__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_24, pl.const(6176, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.add(t__tile_15, t__tile_16)
                rope_bf16_inline916_inline2294__tile: pl.Tile[[8, 64], pl.BF16, pl.MemRef(mem_vec_24, pl.const(6176, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.cast(
                    rope_rot_inline881_inline2292__tile, target_type=pl.BF16, mode="rint"
                )
                qr_bf16_2d_inline914_inline2247__tile: pl.Tensor[[384, 8192], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 6291456)] = pl.tile.store(
                    qr_nope_bf16_inline883_inline2288__tile, [dq_t0_inline895_inline2243__ssa_v0, h0_inline887_inline2271__ssa_v0], qr_bf16_2d_inline914_inline2247__iter_v3
                )
                qr_bf16_2d_inline914_inline2247__tile_1: pl.Tensor[[384, 8192], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 6291456)] = pl.tile.store(
                    rope_bf16_inline916_inline2294__tile, [dq_t0_inline895_inline2243__ssa_v0, h0_inline887_inline2271__ssa_v0 + 64], qr_bf16_2d_inline914_inline2247__tile
                )
                wq_scale_inline886_inline2266__tile_1: pl.Tile[[1, 128], pl.FP32, pl.MemRef(mem_vec_50, pl.const(24096, pl.INT64), 512), pl.Mem.Vec] = t__tile_9
                acc_fp32_inline900_inline2286__tile_1: pl.Tile[[8, 128], pl.FP32, pl.MemRef(mem_vec_26, pl.const(10272, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(
                    t__tile_10, target_type=pl.FP32, mode="none"
                )
                t__tile_17: pl.Tile[[8, 128], pl.FP32, pl.MemRef(mem_vec_44, pl.const(20000, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.full([8, 128], dtype=pl.FP32, value=1.0)
                t__tile_18: pl.Tile[[8, 128], pl.FP32, pl.MemRef(mem_vec_44, pl.const(20000, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.row_expand_mul(
                    t__tile_17, qr_scale_tile_inline889_inline2237__tile
                )
                qr_dequant_scale_inline885_inline2245__tile_1: pl.Tile[[8, 128], pl.FP32, pl.MemRef(mem_vec_44, pl.const(20000, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.col_expand_mul(
                    t__tile_18, wq_scale_inline886_inline2266__tile_1
                )
                qr_dequant_inline884_inline2258__tile_1: pl.Tile[[8, 128], pl.FP32, pl.MemRef(mem_vec_26, pl.const(10272, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.mul(
                    acc_fp32_inline900_inline2286__tile_1, qr_dequant_scale_inline885_inline2245__tile_1
                )
                t__tile_19: pl.Tile[[8, 128], pl.BF16, pl.MemRef(mem_vec_44, pl.const(20000, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(
                    qr_dequant_inline884_inline2258__tile_1, target_type=pl.BF16, mode="rint"
                )
                qr_dequant_v1_inline917_inline2287__tile_1: pl.Tile[[8, 128], pl.FP32, pl.MemRef(mem_vec_26, pl.const(10272, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(
                    t__tile_19, target_type=pl.FP32, mode="round"
                )
                t__tile_20: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_26, pl.const(10272, pl.INT64), 3840), pl.Mem.Vec] = pl.tile.slice(qr_dequant_v1_inline917_inline2287__tile_1, [8, 64], [0, 0])
                qr_nope_bf16_inline883_inline2288__tile_1: pl.Tile[[8, 64], pl.BF16, pl.MemRef(mem_vec_50, pl.const(24096, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.cast(
                    t__tile_20, target_type=pl.BF16, mode="rint"
                )
                qr_rope_slice_inline882_inline2289__tile_1: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_26, pl.const(10528, pl.INT64), 3840), pl.Mem.Vec] = pl.tile.slice(
                    qr_dequant_v1_inline917_inline2287__tile_1, [8, 64], [0, 64]
                )
                gather_acc_init_1: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_44, pl.const(20000, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.create([8, 64], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                for gather_lv_1, (gather_ia_1,) in pl.range(8, init_values=(gather_acc_init_1,)):
                    gather_inp_row_1: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_26, pl.const(10272, pl.INT64), 256), pl.Mem.Vec] = pl.tile.slice(
                        qr_rope_slice_inline882_inline2289__tile_1, [1, 64], [gather_lv_1, 0], [1, 64]
                    )
                    gather_idx_row_1: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_19, pl.const(0, pl.INT64), 256), pl.Mem.Vec] = pl.tile.slice(
                        rope_swap_idx_inline925_inline2276__tile, [1, 64], [gather_lv_1, 0], [1, 64]
                    )
                    gather_row_tmp_1: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_52, pl.const(25120, pl.INT64), 256), pl.Mem.Vec] = pl.tile.create([1, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec)
                    gather_row_1: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_53, pl.const(25376, pl.INT64), 256), pl.Mem.Vec] = pl.tile.gather(gather_inp_row_1, gather_idx_row_1, gather_row_tmp_1)
                    gather_asmbl_1: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_44, pl.const(20000, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.assemble(gather_ia_1, gather_row_1, [gather_lv_1, 0])
                    gather_rv_1: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_44, pl.const(20000, pl.INT64), 2048), pl.Mem.Vec] = pl.yield_(gather_asmbl_1)
                qr_swapped_inline890_inline2279__tile_1: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_44, pl.const(20000, pl.INT64), 2048), pl.Mem.Vec] = gather_rv_1
                t__tile_21: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_26, pl.const(10272, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.mul(
                    qr_rope_slice_inline882_inline2289__tile_1, cos_tile_inline891_inline2284__tile
                )
                t__tile_22: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_44, pl.const(20000, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.mul(
                    qr_swapped_inline890_inline2279__tile_1, sin_tile_inline921_inline2285__tile
                )
                rope_rot_inline881_inline2292__tile_1: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_26, pl.const(10272, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.add(t__tile_21, t__tile_22)
                rope_bf16_inline916_inline2294__tile_1: pl.Tile[[8, 64], pl.BF16, pl.MemRef(mem_vec_26, pl.const(10272, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.cast(
                    rope_rot_inline881_inline2292__tile_1, target_type=pl.BF16, mode="rint"
                )
                qr_bf16_2d_inline914_inline2247__tile_2: pl.Tensor[[384, 8192], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 6291456)] = pl.tile.store(
                    qr_nope_bf16_inline883_inline2288__tile_1, [dq_t0_inline895_inline2243__ssa_v0, h0_inline887_inline2271__ssa_v0_1], qr_bf16_2d_inline914_inline2247__tile_1
                )
                qr_bf16_2d_inline914_inline2247__tile_3: pl.Tensor[[384, 8192], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 6291456)] = pl.tile.store(
                    rope_bf16_inline916_inline2294__tile_1, [dq_t0_inline895_inline2243__ssa_v0, h0_inline887_inline2271__ssa_v0_1 + 64], qr_bf16_2d_inline914_inline2247__tile_2
                )
                qr_bf16_2d_inline914_inline2247__rv_v4: pl.Tensor[[384, 8192], pl.BF16, pl.MemRef("mem_ddr_59", pl.const(0, pl.INT64), 6291456)] = pl.yield_(qr_bf16_2d_inline914_inline2247__tile_3)
            qr_bf16_2d_inline914_inline2247__rv_v2: pl.Tensor[[384, 8192], pl.BF16, pl.MemRef("mem_ddr_60", pl.const(0, pl.INT64), 6291456)] = pl.yield_(qr_bf16_2d_inline914_inline2247__rv_v4)
        return

    @pl.function(type=pl.FunctionType.Spmd)
    def idx_qr_dequant_rope_spmd(
        self,
        qr_bf16_2d_inline914_inline2247__ssa_v0: pl.Out[pl.Tensor[[384, 8192], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 6291456)]],
        dq_rope_units_inline905_inline2241__ssa_v0: pl.Scalar[pl.INDEX],
        dq_rope_workers_inline918_inline2238__ssa_v0: pl.Scalar[pl.INDEX],
        qr_scale__ssa_v0: pl.Tensor[[T_DYN, 1], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)],
        freqs_cos__ssa_v0: pl.Tensor[[T_DYN, 64], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)],
        idx_sin_signed__rv_v2: pl.Tensor[[T_DYN, 64], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)],
        idx_wq_b_scale__ssa_v0: pl.Tensor[[8192], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 32768)],
        qr_acc_pad_inline908_inline2268__rv_v2: pl.Tensor[[384, 8192], pl.INT32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 12582912)],
    ):
        pl.func_attr({"mx_tensor_views_blocked": True})
        self.idx_qr_dequant_rope(
            qr_bf16_2d_inline914_inline2247__ssa_v0,
            dq_rope_units_inline905_inline2241__ssa_v0,
            dq_rope_workers_inline918_inline2238__ssa_v0,
            qr_scale__ssa_v0,
            freqs_cos__ssa_v0,
            idx_sin_signed__rv_v2,
            idx_wq_b_scale__ssa_v0,
            qr_acc_pad_inline908_inline2268__rv_v2,
            attrs={"arg_directions": [pl.adir.output_existing, pl.adir.scalar, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input]},
        )

    @pl.function(type=pl.FunctionType.AIC, level=pl.Level.AIC, role=pl.Role.SubWorker)
    def idx_qr_proj_matmul(
        qr_acc_pad_inline908_inline2268__ssa_v0: pl.Out[pl.Tensor[[384, 8192], pl.INT32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 12582912)]],
        row_blocks_inline907_inline2269__ssa_v0: pl.Scalar[pl.INDEX],
        bs_inline904_inline2257__ssa_v0: pl.Scalar[pl.INDEX],
        qr__ssa_v0: pl.Tensor[[T_DYN, 1024], pl.INT8, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)],
        idx_wq_b__ssa_v0: pl.Tensor[[1024, 8192], pl.INT8, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 8388608)],
    ) -> pl.Tensor[[384, 8192], pl.INT32]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        mem_acc_3: pl.Ptr = pl.tile.alloc(pl.Mem.Acc, 65536)
        mem_mat_4: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 4096)
        mem_mat_5: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 131072)
        mem_mat_6: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 4096)
        mem_mat_7: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 131072)
        mem_left_8: pl.Ptr = pl.tile.alloc(pl.Mem.Left, 1024)
        mem_right_9: pl.Ptr = pl.tile.alloc(pl.Mem.Right, 32768)
        mem_left_10: pl.Ptr = pl.tile.alloc(pl.Mem.Left, 1024)
        mem_right_11: pl.Ptr = pl.tile.alloc(pl.Mem.Right, 32768)
        mem_left_13: pl.Ptr = pl.tile.alloc(pl.Mem.Left, 1024)
        mem_left_15: pl.Ptr = pl.tile.alloc(pl.Mem.Left, 1024)
        qr_proj_worker_inline902_inline2252__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        for qr_unit_inline913_inline2275__idx_v0, (qr_acc_pad_inline908_inline2268__iter_v1,) in pl.range(
            qr_proj_worker_inline902_inline2252__ssa_v0, row_blocks_inline907_inline2269__ssa_v0 * 8, 24, init_values=(qr_acc_pad_inline908_inline2268__ssa_v0,)
        ):
            qr_rb_inline910_inline2277__ssa_v0: pl.Scalar[pl.INDEX] = qr_unit_inline913_inline2275__idx_v0 // 8
            ot_inline897_inline2261__ssa_v0: pl.Scalar[pl.INDEX] = qr_unit_inline913_inline2275__idx_v0 - qr_rb_inline910_inline2277__ssa_v0 * 8
            qr_r0_inline898_inline2280__ssa_v0: pl.Scalar[pl.INDEX] = qr_rb_inline910_inline2277__ssa_v0 * 16
            qr_rows_inline922_inline2272__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(bs_inline904_inline2257__ssa_v0 - qr_r0_inline898_inline2280__ssa_v0, 16)
            o_base_inline896_inline2263__ssa_v0: pl.Scalar[pl.INDEX] = ot_inline897_inline2261__ssa_v0 * 1024
            for ns_inline894_inline2255__idx_v0, (qr_acc_pad_inline908_inline2268__iter_v3,) in pl.range(0, 1024, 512, init_values=(qr_acc_pad_inline908_inline2268__iter_v1,)):
                qr_acc_inline892_inline2250__tile: pl.Tile[[16, 512], pl.INT32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 65536), pl.Mem.Acc] = pl.tile.create(
                    [16, 512], dtype=pl.INT32, target_memory=pl.Mem.Acc
                )
                for kb_inline911_inline2249__idx_v0, (qr_acc_inline892_inline2250__iter_v1,) in pl.range(0, 4, 2, init_values=(qr_acc_inline892_inline2250__tile,)):
                    q0_inline903_inline2246__ssa_v0: pl.Scalar[pl.INDEX] = kb_inline911_inline2249__idx_v0 * 256
                    q0_inline903_inline2246__ssa_v0_1: pl.Scalar[pl.INDEX] = kb_inline911_inline2249__idx_v0 * 256 + 256
                    qr_tile_inline899_inline2254__tile: pl.Tile[
                        [16, 256], pl.INT8, pl.MemRef(mem_mat_4, pl.const(0, pl.INT64), 4096), pl.Mem.Mat, pl.TileView(valid_shape=[qr_rows_inline922_inline2272__ssa_v0, 256])
                    ] = pl.tile.load(
                        qr__ssa_v0, [qr_r0_inline898_inline2280__ssa_v0, q0_inline903_inline2246__ssa_v0], [16, 256], [qr_rows_inline922_inline2272__ssa_v0, 256], target_memory=pl.Mem.Mat
                    )
                    wq_tile_inline912_inline2260__tile: pl.Tile[[256, 512], pl.INT8, pl.MemRef(mem_mat_5, pl.const(4096, pl.INT64), 131072), pl.Mem.Mat] = pl.tile.load(
                        idx_wq_b__ssa_v0, [q0_inline903_inline2246__ssa_v0, o_base_inline896_inline2263__ssa_v0 + ns_inline894_inline2255__idx_v0], [256, 512], [256, 512], target_memory=pl.Mem.Mat
                    )
                    qr_tile_inline899_inline2254__tile_1: pl.Tile[
                        [16, 256], pl.INT8, pl.MemRef(mem_mat_6, pl.const(135168, pl.INT64), 4096), pl.Mem.Mat, pl.TileView(valid_shape=[qr_rows_inline922_inline2272__ssa_v0, 256])
                    ] = pl.tile.load(
                        qr__ssa_v0, [qr_r0_inline898_inline2280__ssa_v0, q0_inline903_inline2246__ssa_v0_1], [16, 256], [qr_rows_inline922_inline2272__ssa_v0, 256], target_memory=pl.Mem.Mat
                    )
                    wq_tile_inline912_inline2260__tile_1: pl.Tile[[256, 512], pl.INT8, pl.MemRef(mem_mat_7, pl.const(139264, pl.INT64), 131072), pl.Mem.Mat] = pl.tile.load(
                        idx_wq_b__ssa_v0, [q0_inline903_inline2246__ssa_v0_1, o_base_inline896_inline2263__ssa_v0 + ns_inline894_inline2255__idx_v0], [256, 512], [256, 512], target_memory=pl.Mem.Mat
                    )
                    for qr_acc_inline892_inline2250__tile_l0_ko, (qr_acc_inline892_inline2250__tile_l0_c,) in pl.range(0, 256, 128, init_values=(qr_acc_inline892_inline2250__iter_v1,)):
                        qr_acc_inline892_inline2250__tile_l0_a: pl.Tile[
                            [16, 64],
                            pl.INT8,
                            pl.MemRef(mem_left_8, pl.const(3072, pl.INT64), 1024),
                            pl.Mem.Left,
                            pl.TileView(valid_shape=[qr_rows_inline922_inline2272__ssa_v0, 64], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal),
                        ] = pl.tile.extract(qr_tile_inline899_inline2254__tile, 0, qr_acc_inline892_inline2250__tile_l0_ko, [16, 64], target_memory=pl.Mem.Left)
                        qr_acc_inline892_inline2250__tile_l0_b: pl.Tile[[64, 512], pl.INT8, pl.MemRef(mem_right_9, pl.const(32768, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                            wq_tile_inline912_inline2260__tile, qr_acc_inline892_inline2250__tile_l0_ko, 0, [64, 512], target_memory=pl.Mem.Right
                        )
                        qr_acc_inline892_inline2250__tile_l0_a_1: pl.Tile[
                            [16, 64],
                            pl.INT8,
                            pl.MemRef(mem_left_10, pl.const(0, pl.INT64), 1024),
                            pl.Mem.Left,
                            pl.TileView(valid_shape=[qr_rows_inline922_inline2272__ssa_v0, 64], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal),
                        ] = pl.tile.extract(qr_tile_inline899_inline2254__tile, 0, qr_acc_inline892_inline2250__tile_l0_ko + 64, [16, 64], target_memory=pl.Mem.Left)
                        qr_acc_inline892_inline2250__tile_l0_b_1: pl.Tile[[64, 512], pl.INT8, pl.MemRef(mem_right_11, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                            wq_tile_inline912_inline2260__tile, qr_acc_inline892_inline2250__tile_l0_ko + 64, 0, [64, 512], target_memory=pl.Mem.Right
                        )
                        qr_acc_inline892_inline2250__tile_l0_c_acc: pl.Tile[[16, 512], pl.INT32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 65536), pl.Mem.Acc] = pl.tile.matmul_acc(
                            qr_acc_inline892_inline2250__tile_l0_c,
                            qr_acc_inline892_inline2250__tile_l0_a,
                            qr_acc_inline892_inline2250__tile_l0_b,
                            q0_inline903_inline2246__ssa_v0 == 0 and qr_acc_inline892_inline2250__tile_l0_ko == 0,
                        )
                        qr_acc_inline892_inline2250__tile_l0_c_acc_1: pl.Tile[[16, 512], pl.INT32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 65536), pl.Mem.Acc] = pl.tile.matmul_acc(
                            qr_acc_inline892_inline2250__tile_l0_c_acc,
                            qr_acc_inline892_inline2250__tile_l0_a_1,
                            qr_acc_inline892_inline2250__tile_l0_b_1,
                            q0_inline903_inline2246__ssa_v0 == 0 and qr_acc_inline892_inline2250__tile_l0_ko == -64,
                        )
                        qr_acc_inline892_inline2250__tile_1: pl.Tile[[16, 512], pl.INT32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 65536), pl.Mem.Acc] = pl.yield_(
                            qr_acc_inline892_inline2250__tile_l0_c_acc_1
                        )
                    for qr_acc_inline892_inline2250__tile_l0_ko_1, (qr_acc_inline892_inline2250__tile_l0_c_1,) in pl.range(0, 256, 128, init_values=(qr_acc_inline892_inline2250__tile_1,)):
                        qr_acc_inline892_inline2250__tile_l0_a_2: pl.Tile[
                            [16, 64],
                            pl.INT8,
                            pl.MemRef(mem_left_13, pl.const(1024, pl.INT64), 1024),
                            pl.Mem.Left,
                            pl.TileView(valid_shape=[qr_rows_inline922_inline2272__ssa_v0, 64], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal),
                        ] = pl.tile.extract(qr_tile_inline899_inline2254__tile_1, 0, qr_acc_inline892_inline2250__tile_l0_ko_1, [16, 64], target_memory=pl.Mem.Left)
                        qr_acc_inline892_inline2250__tile_l0_b_2: pl.Tile[[64, 512], pl.INT8, pl.MemRef(mem_right_9, pl.const(32768, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                            wq_tile_inline912_inline2260__tile_1, qr_acc_inline892_inline2250__tile_l0_ko_1, 0, [64, 512], target_memory=pl.Mem.Right
                        )
                        qr_acc_inline892_inline2250__tile_l0_a_3: pl.Tile[
                            [16, 64],
                            pl.INT8,
                            pl.MemRef(mem_left_15, pl.const(2048, pl.INT64), 1024),
                            pl.Mem.Left,
                            pl.TileView(valid_shape=[qr_rows_inline922_inline2272__ssa_v0, 64], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal),
                        ] = pl.tile.extract(qr_tile_inline899_inline2254__tile_1, 0, qr_acc_inline892_inline2250__tile_l0_ko_1 + 64, [16, 64], target_memory=pl.Mem.Left)
                        qr_acc_inline892_inline2250__tile_l0_b_3: pl.Tile[[64, 512], pl.INT8, pl.MemRef(mem_right_11, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                            wq_tile_inline912_inline2260__tile_1, qr_acc_inline892_inline2250__tile_l0_ko_1 + 64, 0, [64, 512], target_memory=pl.Mem.Right
                        )
                        qr_acc_inline892_inline2250__tile_l0_c_acc_2: pl.Tile[[16, 512], pl.INT32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 65536), pl.Mem.Acc] = pl.tile.matmul_acc(
                            qr_acc_inline892_inline2250__tile_l0_c_1,
                            qr_acc_inline892_inline2250__tile_l0_a_2,
                            qr_acc_inline892_inline2250__tile_l0_b_2,
                            q0_inline903_inline2246__ssa_v0_1 == 0 and qr_acc_inline892_inline2250__tile_l0_ko_1 == 0,
                        )
                        qr_acc_inline892_inline2250__tile_l0_c_acc_3: pl.Tile[[16, 512], pl.INT32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 65536), pl.Mem.Acc] = pl.tile.matmul_acc(
                            qr_acc_inline892_inline2250__tile_l0_c_acc_2,
                            qr_acc_inline892_inline2250__tile_l0_a_3,
                            qr_acc_inline892_inline2250__tile_l0_b_3,
                            q0_inline903_inline2246__ssa_v0_1 == 0 and qr_acc_inline892_inline2250__tile_l0_ko_1 == -64,
                        )
                        qr_acc_inline892_inline2250__tile_2: pl.Tile[[16, 512], pl.INT32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 65536), pl.Mem.Acc] = pl.yield_(
                            qr_acc_inline892_inline2250__tile_l0_c_acc_3
                        )
                    qr_acc_inline892_inline2250__rv_v2: pl.Tile[[16, 512], pl.INT32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 65536), pl.Mem.Acc] = pl.yield_(qr_acc_inline892_inline2250__tile_2)
                qr_acc_pad_inline908_inline2268__tile: pl.Tensor[[384, 8192], pl.INT32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 12582912)] = pl.tile.store(
                    qr_acc_inline892_inline2250__rv_v2,
                    [qr_r0_inline898_inline2280__ssa_v0, o_base_inline896_inline2263__ssa_v0 + ns_inline894_inline2255__idx_v0],
                    qr_acc_pad_inline908_inline2268__iter_v3,
                )
                qr_acc_pad_inline908_inline2268__rv_v4: pl.Tensor[[384, 8192], pl.INT32, pl.MemRef("mem_ddr_19", pl.const(0, pl.INT64), 12582912)] = pl.yield_(qr_acc_pad_inline908_inline2268__tile)
            qr_acc_pad_inline908_inline2268__rv_v2: pl.Tensor[[384, 8192], pl.INT32, pl.MemRef("mem_ddr_20", pl.const(0, pl.INT64), 12582912)] = pl.yield_(qr_acc_pad_inline908_inline2268__rv_v4)
        return qr_acc_pad_inline908_inline2268__ssa_v0

    @pl.function(type=pl.FunctionType.Spmd)
    def idx_qr_proj_matmul_spmd(
        self,
        qr_acc_pad_inline908_inline2268__ssa_v0: pl.Out[pl.Tensor[[384, 8192], pl.INT32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 12582912)]],
        row_blocks_inline907_inline2269__ssa_v0: pl.Scalar[pl.INDEX],
        bs_inline904_inline2257__ssa_v0: pl.Scalar[pl.INDEX],
        qr__ssa_v0: pl.Tensor[[T_DYN, 1024], pl.INT8, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)],
        idx_wq_b__ssa_v0: pl.Tensor[[1024, 8192], pl.INT8, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 8388608)],
    ) -> pl.Tensor[[384, 8192], pl.INT32]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        qr_acc_pad_inline908_inline2268__rv_v2: pl.Tensor[[384, 8192], pl.INT32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 12582912)] = self.idx_qr_proj_matmul(
            qr_acc_pad_inline908_inline2268__ssa_v0,
            row_blocks_inline907_inline2269__ssa_v0,
            bs_inline904_inline2257__ssa_v0,
            qr__ssa_v0,
            idx_wq_b__ssa_v0,
            attrs={"arg_directions": [pl.adir.output_existing, pl.adir.scalar, pl.adir.scalar, pl.adir.input, pl.adir.input]},
        )
        return qr_acc_pad_inline908_inline2268__ssa_v0

    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def indexer_boundary_init(normed_kv_inline212__ssa_v0: pl.Out[pl.Tensor[[384, 128], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 98304)]]) -> pl.Tensor[[384, 128], pl.BF16]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        mem_vec_1: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 512)
        init_request_inline443_inline2072__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        compact_begin_inline467_inline2071__ssa_v0: pl.Scalar[pl.INDEX] = init_request_inline443_inline2072__ssa_v0 * 2
        t__tile: pl.Tile[[2, 128], pl.BF16, pl.MemRef(mem_vec_1, pl.const(0, pl.INT64), 512), pl.Mem.Vec] = pl.tile.full([2, 128], dtype=pl.BF16, value=0.0)
        normed_kv_inline212__tile: pl.Tensor[[384, 128], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 98304)] = pl.tile.store(
            t__tile, [compact_begin_inline467_inline2071__ssa_v0, 0], normed_kv_inline212__ssa_v0
        )
        return normed_kv_inline212__ssa_v0

    @pl.function(type=pl.FunctionType.Spmd)
    def indexer_boundary_init_spmd(self, normed_kv_inline212__ssa_v0: pl.Out[pl.Tensor[[384, 128], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 98304)]]) -> pl.Tensor[[384, 128], pl.BF16]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        normed_kv_inline212__ssa_v1: pl.Tensor[[384, 128], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 98304)] = self.indexer_boundary_init(
            normed_kv_inline212__ssa_v0, attrs={"arg_directions": [pl.adir.output_existing]}
        )
        return normed_kv_inline212__ssa_v0

    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def indexer_head_coefficients(
        position_ids__ssa_v0: pl.Tensor[[T_DYN], pl.INT64, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)],
        coefficients_inline3198__ssa_v0: pl.Out[pl.Tensor[[6144, 64], pl.FP16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 786432)]],
        qr_hadamard_scale_dq_inline217__rv_v2: pl.Tensor[[24576, 1], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 98304)],
        weights_inline2400__ssa_v0: pl.Tensor[[384, 64], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 98304)],
    ) -> pl.Tensor[[6144, 64], pl.FP16]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        mem_vec_5: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        mem_vec_7: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 128)
        mem_vec_8: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 4096)
        coefficient_worker_inline3194__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        coefficient_count_inline3193__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(position_ids__ssa_v0, 0)
        for coefficient_query_inline3192__idx_v0, (coefficients_inline3198__iter_v1,) in pl.range(
            coefficient_worker_inline3194__ssa_v0, coefficient_count_inline3193__ssa_v0, 48, init_values=(coefficients_inline3198__ssa_v0,)
        ):
            coefficient_head_begin_inline3191__ssa_v0: pl.Scalar[pl.INDEX] = coefficient_query_inline3192__idx_v0 * 64
            t__tile: pl.Tile[[64, 1], pl.FP32, pl.MemRef(mem_vec_8, pl.const(384, pl.INT64), 256), pl.Mem.Vec] = pl.tile.load(
                qr_hadamard_scale_dq_inline217__rv_v2, [coefficient_head_begin_inline3191__ssa_v0, 0], [64, 1], [64, 1], target_memory=pl.Mem.Vec
            )
            query_scale_inline3190__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_8, pl.const(384, pl.INT64), 256), pl.Mem.Vec] = pl.tile.reshape(t__tile, [1, 64])
            query_weight_inline3197__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_5, pl.const(0, pl.INT64), 256), pl.Mem.Vec] = pl.tile.load(
                weights_inline2400__ssa_v0, [coefficient_query_inline3192__idx_v0, 0], [1, 64], [1, 64], target_memory=pl.Mem.Vec
            )
            t__tile_1: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_8, pl.const(384, pl.INT64), 256), pl.Mem.Vec] = pl.tile.mul(query_scale_inline3190__tile, query_weight_inline3197__tile)
            head_coefficient_inline3189__tile: pl.Tile[[1, 64], pl.FP16, pl.MemRef(mem_vec_7, pl.const(256, pl.INT64), 128), pl.Mem.Vec] = pl.tile.cast(t__tile_1, target_type=pl.FP16, mode="rint")
            t__tile_2: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_8, pl.const(384, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.full([16, 64], dtype=pl.FP32, value=1.0)
            t__tile_3: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_5, pl.const(0, pl.INT64), 256), pl.Mem.Vec] = pl.tile.cast(head_coefficient_inline3189__tile, target_type=pl.FP32, mode="round")
            coefficient_rows_inline3195__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_8, pl.const(384, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.col_expand_mul(t__tile_2, t__tile_3)
            coefficient_row_inline3188__ssa_v0: pl.Scalar[pl.INDEX] = coefficient_query_inline3192__idx_v0 * 16
            t__tile_4: pl.Tile[[16, 64], pl.FP16, pl.MemRef(mem_vec_8, pl.const(384, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(coefficient_rows_inline3195__tile, target_type=pl.FP16, mode="round")
            coefficients_inline3198__tile: pl.Tensor[[6144, 64], pl.FP16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 786432)] = pl.tile.store(
                t__tile_4, [coefficient_row_inline3188__ssa_v0, 0], coefficients_inline3198__iter_v1
            )
            coefficients_inline3198__rv_v2: pl.Tensor[[6144, 64], pl.FP16, pl.MemRef("mem_ddr_12", pl.const(0, pl.INT64), 786432)] = pl.yield_(coefficients_inline3198__tile)
        return coefficients_inline3198__ssa_v0

    @pl.function(type=pl.FunctionType.Spmd)
    def indexer_head_coefficients_spmd(
        self,
        position_ids__ssa_v0: pl.Tensor[[T_DYN], pl.INT64, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)],
        coefficients_inline3198__ssa_v0: pl.Out[pl.Tensor[[6144, 64], pl.FP16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 786432)]],
        qr_hadamard_scale_dq_inline217__rv_v2: pl.Tensor[[24576, 1], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 98304)],
        weights_inline2400__ssa_v0: pl.Tensor[[384, 64], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 98304)],
    ) -> pl.Tensor[[6144, 64], pl.FP16]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        coefficients_inline3198__rv_v2: pl.Tensor[[6144, 64], pl.FP16, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 786432)] = self.indexer_head_coefficients(
            position_ids__ssa_v0,
            coefficients_inline3198__ssa_v0,
            qr_hadamard_scale_dq_inline217__rv_v2,
            weights_inline2400__ssa_v0,
            attrs={"arg_directions": [pl.adir.input, pl.adir.output_existing, pl.adir.input, pl.adir.input]},
        )
        return coefficients_inline3198__ssa_v0

    @pl.function(type=pl.FunctionType.AIC, level=pl.Level.AIC, role=pl.Role.SubWorker)
    def indexer_score_topk_leaf_aic(
        position_ids__ssa_v0: pl.Tensor[[T_DYN], pl.INT64, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)],
        kv_seq_lens__ssa_v0: pl.Tensor[[B_DYN], pl.INT32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)],
        score_arena_inline1071_inline2347__ssa_v0: pl.InOut[pl.Tensor[[384, 8192], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 12582912)]],
        qr_hadamard_i8_inline221__rv_v2: pl.Tensor[[24576, 128], pl.INT8, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 3145728)],
        coefficients_inline1068_inline2344__ssa_v0: pl.Tensor[[6144, 64], pl.FP16, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 786432)],
        idx_block_table_flat_inline1052_inline2374__ssa_v0: pl.Tensor[[idx_table_len_inline1096_inline2335__ssa_v0], pl.INT32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 0)],
        table_columns_inline1060_inline2338__ssa_v0: pl.Scalar[pl.INDEX],
        idx_kv_cache__rv_v2: pl.Tensor[[IDX_CACHE_BLOCK_NUM_DYN, INDEXER_PAGE_BYTES_DYN], pl.INT8, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 0)],
        pair_arena_inline1064_inline2363__ssa_v0: pl.Out[pl.Tensor[[24576, 1024], pl.FP32, pl.MemRef("mem_ddr_7", pl.const(0, pl.INT64), 100663296)]],
        __gm_pipe_buffer: pl.Out[pl.Tensor[[1], pl.FP32, pl.MemRef("mem_ddr_8", pl.const(0, pl.INT64), 4)]],
    ):
        pl.func_attr({"slot_num": 1, "mx_tensor_views_blocked": True, "split_aiv": True})
        mem_mat_9: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 8192)
        mem_mat_10: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 2048)
        mem_acc_11: pl.Ptr = pl.tile.alloc(pl.Mem.Acc, 98304)
        mem_left_12: pl.Ptr = pl.tile.alloc(pl.Mem.Left, 4096)
        mem_right_13: pl.Ptr = pl.tile.alloc(pl.Mem.Right, 24576)
        mem_left_14: pl.Ptr = pl.tile.alloc(pl.Mem.Left, 4096)
        mem_right_15: pl.Ptr = pl.tile.alloc(pl.Mem.Right, 24576)
        indexer_score_topk_leaf_v2c_slot_buffer: pl.Scalar[pl.INT32] = pl.system.reserve_buffer(name="indexer_score_topk_leaf_v2c_slot_buffer", size=98304, base=0)
        indexer_score_topk_leaf_c2v_slot_buffer_import: pl.Scalar[pl.INT32] = pl.system.import_peer_buffer(name="indexer_score_topk_leaf_c2v_slot_buffer", peer_func="indexer_score_topk_leaf_aiv")
        pl.system.aic_initialize_pipe(indexer_score_topk_leaf_c2v_slot_buffer_import, indexer_score_topk_leaf_v2c_slot_buffer, dir_mask=3, slot_size=98304, slot_num=1)
        worker_inline1094_inline2380__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        query_count_inline1081_inline2381__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(position_ids__ssa_v0, 0)
        for batch_inline1079_inline2384__idx_v0, (max_cache_len_inline1053_inline2377__iter_v1,) in pl.range(query_count_inline1081_inline2381__ssa_v0 // 6, init_values=(0,)):
            t__tile: pl.Scalar[pl.INT32] = pl.tensor.read(kv_seq_lens__ssa_v0, [batch_inline1079_inline2384__idx_v0])
            batch_cache_len_inline1059_inline2387__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(t__tile, pl.INDEX) // 4
            max_cache_len_inline1053_inline2377__ssa_v3: pl.Scalar[pl.INDEX] = pl.max(max_cache_len_inline1053_inline2377__iter_v1, batch_cache_len_inline1059_inline2387__ssa_v0)
            max_cache_len_inline1053_inline2377__rv_v2: pl.Scalar[pl.INDEX] = pl.yield_(max_cache_len_inline1053_inline2377__ssa_v3)
        max_leaves_inline1082_inline2339__ssa_v0: pl.Scalar[pl.INDEX] = pl.max((pl.min(max_cache_len_inline1053_inline2377__rv_v2, 262144) + 8191) // 8192, 1)
        for item_inline1075_inline2388__idx_v0, (score_arena_inline1071_inline2347__iter_v1,) in pl.range(
            worker_inline1094_inline2380__ssa_v0, query_count_inline1081_inline2381__ssa_v0 * max_leaves_inline1082_inline2339__ssa_v0, 24, init_values=(score_arena_inline1071_inline2347__ssa_v0,)
        ):
            query_inline1074_inline2340__ssa_v0: pl.Scalar[pl.INDEX] = item_inline1075_inline2388__idx_v0 // max_leaves_inline1082_inline2339__ssa_v0
            leaf_inline1070_inline2390__ssa_v0: pl.Scalar[pl.INDEX] = item_inline1075_inline2388__idx_v0 % max_leaves_inline1082_inline2339__ssa_v0
            batch_idx_inline1055_inline2348__ssa_v0: pl.Scalar[pl.INDEX] = query_inline1074_inline2340__ssa_v0 // 6
            position_inline1061_inline2355__tile: pl.Scalar[pl.INT64] = pl.tensor.read(position_ids__ssa_v0, [query_inline1074_inline2340__ssa_v0])
            t__tile_1: pl.Scalar[pl.INT32] = pl.tensor.read(kv_seq_lens__ssa_v0, [batch_idx_inline1055_inline2348__ssa_v0])
            cache_len_inline1057_inline2392__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(t__tile_1, pl.INDEX) // 4
            cache_bound_inline1049_inline2385__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(cache_len_inline1057_inline2392__ssa_v0, (position_inline1061_inline2355__tile + 1) // 4)
            visible_count_inline1084_inline2342__ssa_v0: pl.Scalar[pl.INDEX] = pl.max(pl.min(cache_bound_inline1049_inline2385__ssa_v0, 262144), 0)
            logical_begin_inline1047_inline2394__ssa_v0: pl.Scalar[pl.INDEX] = leaf_inline1070_inline2390__ssa_v0 * 8192
            if logical_begin_inline1047_inline2394__ssa_v0 < visible_count_inline1084_inline2342__ssa_v0:
                valid_count_inline1046_inline2398__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(visible_count_inline1084_inline2342__ssa_v0 - logical_begin_inline1047_inline2394__ssa_v0, 8192)
                lane_span_inline1069_inline2401__ssa_v0: pl.Scalar[pl.INDEX] = pl.min((valid_count_inline1046_inline2398__ssa_v0 + 383) // 384 * 192, 4096)
                query_head_begin_inline1058_inline2382__ssa_v0: pl.Scalar[pl.INDEX] = query_inline1074_inline2340__ssa_v0 * 64
                query_vector_inline1085_inline2396__tile: pl.Tile[[64, 128], pl.INT8, pl.MemRef(mem_mat_9, pl.const(100352, pl.INT64), 8192), pl.Mem.Mat] = pl.tile.load(
                    qr_hadamard_i8_inline221__rv_v2, [query_head_begin_inline1058_inline2382__ssa_v0, 0], [64, 128], [64, 128], target_memory=pl.Mem.Mat
                )
                coefficient_begin_inline1087_inline2403__ssa_v0: pl.Scalar[pl.INDEX] = query_inline1074_inline2340__ssa_v0 * 16
                coefficients_l1_inline1092_inline2332__tile: pl.Tile[[16, 64], pl.FP16, pl.MemRef(mem_mat_10, pl.const(98304, pl.INT64), 2048), pl.Mem.Mat] = pl.tile.load(
                    coefficients_inline1068_inline2344__ssa_v0, [coefficient_begin_inline1087_inline2403__ssa_v0, 0], [16, 64], [16, 64], target_memory=pl.Mem.Mat
                )
                for score_begin_inline1090_inline2391__idx_v0, (score_arena_inline1071_inline2347__iter_v3,) in pl.range(
                    0, lane_span_inline1069_inline2401__ssa_v0, 192, init_values=(score_arena_inline1071_inline2347__iter_v1,)
                ):
                    kv_i8_inline1106_inline2321__tile: pl.Tile[[384, 128], pl.INT8, pl.Mem.Mat] = pl.tile.tpop_from_aiv(split=1)
                    kv_i8_inline1106_inline2321__tile_t: pl.Tile[[128, 384], pl.INT8, pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)] = (
                        pl.tile.transpose_view(kv_i8_inline1106_inline2321__tile)
                    )
                    score_i32_inline1107_inline2371__tile_l0_init: pl.Tile[[64, 384], pl.INT32, pl.MemRef(mem_acc_11, pl.const(0, pl.INT64), 98304), pl.Mem.Acc] = pl.tile.create(
                        [64, 384], dtype=pl.INT32, target_memory=pl.Mem.Acc
                    )
                    score_i32_inline1107_inline2371__tile_l0_a: pl.Tile[
                        [64, 64], pl.INT8, pl.MemRef(mem_left_12, pl.const(0, pl.INT64), 4096), pl.Mem.Left, pl.TileView(blayout=pl.TileLayout.row_major)
                    ] = pl.tile.extract(query_vector_inline1085_inline2396__tile, 0, 0, [64, 64], target_memory=pl.Mem.Left)
                    score_i32_inline1107_inline2371__tile_l0_b: pl.Tile[[64, 384], pl.INT8, pl.MemRef(mem_right_13, pl.const(0, pl.INT64), 24576), pl.Mem.Right] = pl.tile.extract(
                        kv_i8_inline1106_inline2321__tile_t, 0, 0, [64, 384], target_memory=pl.Mem.Right
                    )
                    score_i32_inline1107_inline2371__tile_l0_a_1: pl.Tile[
                        [64, 64], pl.INT8, pl.MemRef(mem_left_14, pl.const(4096, pl.INT64), 4096), pl.Mem.Left, pl.TileView(blayout=pl.TileLayout.row_major)
                    ] = pl.tile.extract(query_vector_inline1085_inline2396__tile, 0, 64, [64, 64], target_memory=pl.Mem.Left)
                    score_i32_inline1107_inline2371__tile_l0_b_1: pl.Tile[[64, 384], pl.INT8, pl.MemRef(mem_right_15, pl.const(24576, pl.INT64), 24576), pl.Mem.Right] = pl.tile.extract(
                        kv_i8_inline1106_inline2321__tile_t, 64, 0, [64, 384], target_memory=pl.Mem.Right
                    )
                    score_i32_inline1107_inline2371__tile_l0_c_acc: pl.Tile[[64, 384], pl.INT32, pl.MemRef(mem_acc_11, pl.const(0, pl.INT64), 98304), pl.Mem.Acc] = pl.tile.matmul_acc(
                        score_i32_inline1107_inline2371__tile_l0_init, score_i32_inline1107_inline2371__tile_l0_a, score_i32_inline1107_inline2371__tile_l0_b, True
                    )
                    score_i32_inline1107_inline2371__tile_l0_c_acc_1: pl.Tile[[64, 384], pl.INT32, pl.MemRef(mem_acc_11, pl.const(0, pl.INT64), 98304), pl.Mem.Acc] = pl.tile.matmul_acc(
                        score_i32_inline1107_inline2371__tile_l0_c_acc, score_i32_inline1107_inline2371__tile_l0_a_1, score_i32_inline1107_inline2371__tile_l0_b_1, False
                    )
                    pl.system.tfree_to_aiv(kv_i8_inline1106_inline2321__tile, split=1)
                    pl.tile.tpush_to_aiv(score_i32_inline1107_inline2371__tile_l0_c_acc_1, split=2)
                    scores_l1_inline1109_inline2316__tile: pl.Tile[[64, 384], pl.FP16, pl.Mem.Mat] = pl.tile.tpop_from_aiv(split=2)
                    weighted_scores_inline1065_inline2315__tile_l0_init: pl.Tile[[16, 384], pl.FP32, pl.MemRef(mem_acc_11, pl.const(0, pl.INT64), 24576), pl.Mem.Acc] = pl.tile.create(
                        [16, 384], dtype=pl.FP32, target_memory=pl.Mem.Acc
                    )
                    weighted_scores_inline1065_inline2315__tile_l0_a: pl.Tile[
                        [16, 32], pl.FP16, pl.MemRef(mem_left_12, pl.const(0, pl.INT64), 1024), pl.Mem.Left, pl.TileView(blayout=pl.TileLayout.row_major)
                    ] = pl.tile.extract(coefficients_l1_inline1092_inline2332__tile, 0, 0, [16, 32], target_memory=pl.Mem.Left)
                    weighted_scores_inline1065_inline2315__tile_l0_b: pl.Tile[[32, 384], pl.FP16, pl.MemRef(mem_right_13, pl.const(0, pl.INT64), 24576), pl.Mem.Right] = pl.tile.extract(
                        scores_l1_inline1109_inline2316__tile, 0, 0, [32, 384], target_memory=pl.Mem.Right
                    )
                    weighted_scores_inline1065_inline2315__tile_l0_a_1: pl.Tile[
                        [16, 32], pl.FP16, pl.MemRef(mem_left_14, pl.const(4096, pl.INT64), 1024), pl.Mem.Left, pl.TileView(blayout=pl.TileLayout.row_major)
                    ] = pl.tile.extract(coefficients_l1_inline1092_inline2332__tile, 0, 32, [16, 32], target_memory=pl.Mem.Left)
                    weighted_scores_inline1065_inline2315__tile_l0_b_1: pl.Tile[[32, 384], pl.FP16, pl.MemRef(mem_right_15, pl.const(24576, pl.INT64), 24576), pl.Mem.Right] = pl.tile.extract(
                        scores_l1_inline1109_inline2316__tile, 32, 0, [32, 384], target_memory=pl.Mem.Right
                    )
                    weighted_scores_inline1065_inline2315__tile_l0_c_acc: pl.Tile[[16, 384], pl.FP32, pl.MemRef(mem_acc_11, pl.const(0, pl.INT64), 24576), pl.Mem.Acc] = pl.tile.matmul_acc(
                        weighted_scores_inline1065_inline2315__tile_l0_init, weighted_scores_inline1065_inline2315__tile_l0_a, weighted_scores_inline1065_inline2315__tile_l0_b, True
                    )
                    weighted_scores_inline1065_inline2315__tile_l0_c_acc_1: pl.Tile[[16, 384], pl.FP32, pl.MemRef(mem_acc_11, pl.const(0, pl.INT64), 24576), pl.Mem.Acc] = pl.tile.matmul_acc(
                        weighted_scores_inline1065_inline2315__tile_l0_c_acc, weighted_scores_inline1065_inline2315__tile_l0_a_1, weighted_scores_inline1065_inline2315__tile_l0_b_1, False
                    )
                    pl.system.tfree_to_aiv(scores_l1_inline1109_inline2316__tile, split=2)
                    pl.tile.tpush_to_aiv(weighted_scores_inline1065_inline2315__tile_l0_c_acc_1, split=2)
                    score_arena_inline1071_inline2347__rv_v4: pl.Tensor[[384, 8192], pl.FP32, pl.MemRef("mem_ddr_23", pl.const(0, pl.INT64), 12582912)] = pl.yield_(
                        score_arena_inline1071_inline2347__iter_v3
                    )
                score_arena_inline1071_inline2347__phi_v7: pl.Tensor[[384, 8192], pl.FP32, pl.MemRef("mem_ddr_24", pl.const(0, pl.INT64), 12582912)] = pl.yield_(
                    score_arena_inline1071_inline2347__rv_v4
                )
            else:
                score_arena_inline1071_inline2347__phi_v7: pl.Tensor[[384, 8192], pl.FP32, pl.MemRef("mem_ddr_24", pl.const(0, pl.INT64), 12582912)] = pl.yield_(
                    score_arena_inline1071_inline2347__iter_v1
                )
            score_arena_inline1071_inline2347__rv_v2: pl.Tensor[[384, 8192], pl.FP32, pl.MemRef("mem_ddr_25", pl.const(0, pl.INT64), 12582912)] = pl.yield_(score_arena_inline1071_inline2347__phi_v7)

    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def indexer_score_topk_leaf_aiv(
        position_ids__ssa_v0: pl.Tensor[[T_DYN], pl.INT64, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)],
        kv_seq_lens__ssa_v0: pl.Tensor[[B_DYN], pl.INT32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)],
        score_arena_inline1071_inline2347__ssa_v0: pl.InOut[pl.Tensor[[384, 8192], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 12582912)]],
        qr_hadamard_i8_inline221__rv_v2: pl.Tensor[[24576, 128], pl.INT8, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 3145728)],
        coefficients_inline1068_inline2344__ssa_v0: pl.Tensor[[6144, 64], pl.FP16, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 786432)],
        idx_block_table_flat_inline1052_inline2374__ssa_v0: pl.Tensor[[idx_table_len_inline1096_inline2335__ssa_v0], pl.INT32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 0)],
        table_columns_inline1060_inline2338__ssa_v0: pl.Scalar[pl.INDEX],
        idx_kv_cache__rv_v2: pl.Tensor[[IDX_CACHE_BLOCK_NUM_DYN, INDEXER_PAGE_BYTES_DYN], pl.INT8, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 0)],
        pair_arena_inline1064_inline2363__ssa_v0: pl.Out[pl.Tensor[[24576, 1024], pl.FP32, pl.MemRef("mem_ddr_7", pl.const(0, pl.INT64), 100663296)]],
        __gm_pipe_buffer: pl.Out[pl.Tensor[[1], pl.FP32, pl.MemRef("mem_ddr_8", pl.const(0, pl.INT64), 4)]],
    ) -> tuple[pl.Tensor[[384, 8192], pl.FP32], pl.Tensor[[24576, 1024], pl.FP32]]:
        pl.func_attr({"slot_num": 1, "mx_tensor_views_blocked": True, "split_aiv": True, "dual_aiv_dispatch": True})
        mem_vec_10: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 49152)
        mem_vec_56: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32768)
        indexer_score_topk_leaf_v2c_slot_buffer_import: pl.Scalar[pl.INT32] = pl.system.import_peer_buffer(name="indexer_score_topk_leaf_v2c_slot_buffer", peer_func="indexer_score_topk_leaf_aic")
        indexer_score_topk_leaf_c2v_slot_buffer: pl.Scalar[pl.INT32] = pl.system.reserve_buffer(name="indexer_score_topk_leaf_c2v_slot_buffer", size=98304, base=0)
        pl.system.aiv_initialize_pipe(indexer_score_topk_leaf_c2v_slot_buffer, indexer_score_topk_leaf_v2c_slot_buffer_import, dir_mask=3, slot_size=98304, slot_num=1)
        worker_inline1094_inline2380__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        query_count_inline1081_inline2381__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(position_ids__ssa_v0, 0)
        for batch_inline1079_inline2384__idx_v0, (max_cache_len_inline1053_inline2377__iter_v1,) in pl.range(query_count_inline1081_inline2381__ssa_v0 // 6, init_values=(0,)):
            t__tile: pl.Scalar[pl.INT32] = pl.tensor.read(kv_seq_lens__ssa_v0, [batch_inline1079_inline2384__idx_v0])
            batch_cache_len_inline1059_inline2387__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(t__tile, pl.INDEX) // 4
            max_cache_len_inline1053_inline2377__ssa_v3: pl.Scalar[pl.INDEX] = pl.max(max_cache_len_inline1053_inline2377__iter_v1, batch_cache_len_inline1059_inline2387__ssa_v0)
            max_cache_len_inline1053_inline2377__rv_v2: pl.Scalar[pl.INDEX] = pl.yield_(max_cache_len_inline1053_inline2377__ssa_v3)
        max_leaves_inline1082_inline2339__ssa_v0: pl.Scalar[pl.INDEX] = pl.max((pl.min(max_cache_len_inline1053_inline2377__rv_v2, 262144) + 8191) // 8192, 1)
        single_leaf_inline1077_inline2354__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(max_leaves_inline1082_inline2339__ssa_v0 == 1, pl.INDEX)
        for item_inline1075_inline2388__idx_v0, (score_arena_inline1071_inline2347__iter_v1,) in pl.range(
            worker_inline1094_inline2380__ssa_v0, query_count_inline1081_inline2381__ssa_v0 * max_leaves_inline1082_inline2339__ssa_v0, 24, init_values=(score_arena_inline1071_inline2347__ssa_v0,)
        ):
            query_inline1074_inline2340__ssa_v0: pl.Scalar[pl.INDEX] = item_inline1075_inline2388__idx_v0 // max_leaves_inline1082_inline2339__ssa_v0
            leaf_inline1070_inline2390__ssa_v0: pl.Scalar[pl.INDEX] = item_inline1075_inline2388__idx_v0 % max_leaves_inline1082_inline2339__ssa_v0
            batch_idx_inline1055_inline2348__ssa_v0: pl.Scalar[pl.INDEX] = query_inline1074_inline2340__ssa_v0 // 6
            position_inline1061_inline2355__tile: pl.Scalar[pl.INT64] = pl.tensor.read(position_ids__ssa_v0, [query_inline1074_inline2340__ssa_v0])
            t__tile_1: pl.Scalar[pl.INT32] = pl.tensor.read(kv_seq_lens__ssa_v0, [batch_idx_inline1055_inline2348__ssa_v0])
            cache_len_inline1057_inline2392__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(t__tile_1, pl.INDEX) // 4
            cache_bound_inline1049_inline2385__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(cache_len_inline1057_inline2392__ssa_v0, (position_inline1061_inline2355__tile + 1) // 4)
            visible_count_inline1084_inline2342__ssa_v0: pl.Scalar[pl.INDEX] = pl.max(pl.min(cache_bound_inline1049_inline2385__ssa_v0, 262144), 0)
            logical_begin_inline1047_inline2394__ssa_v0: pl.Scalar[pl.INDEX] = leaf_inline1070_inline2390__ssa_v0 * 8192
            if logical_begin_inline1047_inline2394__ssa_v0 < visible_count_inline1084_inline2342__ssa_v0:
                valid_count_inline1046_inline2398__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(visible_count_inline1084_inline2342__ssa_v0 - logical_begin_inline1047_inline2394__ssa_v0, 8192)
                lane_span_inline1069_inline2401__ssa_v0: pl.Scalar[pl.INDEX] = pl.min((valid_count_inline1046_inline2398__ssa_v0 + 383) // 384 * 192, 4096)
                lane_stride_inline1050_inline2402__ssa_v0: pl.Scalar[pl.INDEX] = (
                    single_leaf_inline1077_inline2354__ssa_v0 * 192 + (1 - single_leaf_inline1077_inline2354__ssa_v0) * lane_span_inline1069_inline2401__ssa_v0
                )
                for score_begin_inline1090_inline2391__idx_v0, (score_arena_inline1071_inline2347__iter_v3,) in pl.range(
                    0, lane_span_inline1069_inline2401__ssa_v0, 192, init_values=(score_arena_inline1071_inline2347__iter_v1,)
                ):
                    read_begin_inline1093_inline2331__ssa_v0: pl.Scalar[pl.INDEX] = score_begin_inline1090_inline2391__idx_v0 * (single_leaf_inline1077_inline2354__ssa_v0 + 1)
                    key_aiv_inline1097_inline2330__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_subblock_idx()
                    key_bytes_inline1098_inline2328__tile: pl.Tile[[1, 24576], pl.INT8, pl.MemRef(mem_vec_10, pl.const(98304, pl.INT64), 24576), pl.Mem.Vec] = pl.tile.create(
                        [1, 24576], dtype=pl.INT8, target_memory=pl.Mem.Vec
                    )
                    page_begin_inline1086_inline2333__ssa_v0: pl.Scalar[pl.INDEX] = 0
                    lane_page_inline1100_inline2326__ssa_v0: pl.Scalar[pl.INDEX] = (
                        key_aiv_inline1097_inline2330__ssa_v0 * lane_stride_inline1050_inline2402__ssa_v0 + page_begin_inline1086_inline2333__ssa_v0
                    )
                    safe_page_begin_inline1091_inline2386__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(
                        read_begin_inline1093_inline2331__ssa_v0 + lane_page_inline1100_inline2326__ssa_v0, (valid_count_inline1046_inline2398__ssa_v0 - 1) // 32 * 32
                    )
                    logical_page_inline1103_inline2324__ssa_v0: pl.Scalar[pl.INDEX] = (logical_begin_inline1047_inline2394__ssa_v0 + safe_page_begin_inline1091_inline2386__ssa_v0) // 32
                    t__tile_2: pl.Scalar[pl.INT32] = pl.tensor.read(
                        idx_block_table_flat_inline1052_inline2374__ssa_v0,
                        [batch_idx_inline1055_inline2348__ssa_v0 * table_columns_inline1060_inline2338__ssa_v0 + logical_page_inline1103_inline2324__ssa_v0],
                    )
                    physical_block_inline1104_inline2393__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(t__tile_2, pl.INDEX)
                    key_bytes_inline1098_inline2328__tile_1: pl.Tile[[1, 24576], pl.INT8, pl.MemRef(mem_vec_10, pl.const(98304, pl.INT64), 24576), pl.Mem.Vec] = pl.tile.gather_row(
                        key_bytes_inline1098_inline2328__tile,
                        idx_kv_cache__rv_v2,
                        [0, page_begin_inline1086_inline2333__ssa_v0 * 128],
                        [physical_block_inline1104_inline2393__ssa_v0, 0],
                        [1, 4096],
                        transpose=False,
                    )
                    page_begin_inline1086_inline2333__ssa_v1: pl.Scalar[pl.INDEX] = 32
                    lane_page_inline1100_inline2326__ssa_v1: pl.Scalar[pl.INDEX] = (
                        key_aiv_inline1097_inline2330__ssa_v0 * lane_stride_inline1050_inline2402__ssa_v0 + page_begin_inline1086_inline2333__ssa_v1
                    )
                    safe_page_begin_inline1091_inline2386__ssa_v1: pl.Scalar[pl.INDEX] = pl.min(
                        read_begin_inline1093_inline2331__ssa_v0 + lane_page_inline1100_inline2326__ssa_v1, (valid_count_inline1046_inline2398__ssa_v0 - 1) // 32 * 32
                    )
                    logical_page_inline1103_inline2324__ssa_v1: pl.Scalar[pl.INDEX] = (logical_begin_inline1047_inline2394__ssa_v0 + safe_page_begin_inline1091_inline2386__ssa_v1) // 32
                    t__tile_3: pl.Scalar[pl.INT32] = pl.tensor.read(
                        idx_block_table_flat_inline1052_inline2374__ssa_v0,
                        [batch_idx_inline1055_inline2348__ssa_v0 * table_columns_inline1060_inline2338__ssa_v0 + logical_page_inline1103_inline2324__ssa_v1],
                    )
                    physical_block_inline1104_inline2393__ssa_v1: pl.Scalar[pl.INDEX] = pl.cast(t__tile_3, pl.INDEX)
                    key_bytes_inline1098_inline2328__tile_2: pl.Tile[[1, 24576], pl.INT8, pl.MemRef(mem_vec_10, pl.const(98304, pl.INT64), 24576), pl.Mem.Vec] = pl.tile.gather_row(
                        key_bytes_inline1098_inline2328__tile_1,
                        idx_kv_cache__rv_v2,
                        [0, page_begin_inline1086_inline2333__ssa_v1 * 128],
                        [physical_block_inline1104_inline2393__ssa_v1, 0],
                        [1, 4096],
                        transpose=False,
                    )
                    page_begin_inline1086_inline2333__ssa_v2: pl.Scalar[pl.INDEX] = 64
                    lane_page_inline1100_inline2326__ssa_v2: pl.Scalar[pl.INDEX] = (
                        key_aiv_inline1097_inline2330__ssa_v0 * lane_stride_inline1050_inline2402__ssa_v0 + page_begin_inline1086_inline2333__ssa_v2
                    )
                    safe_page_begin_inline1091_inline2386__ssa_v2: pl.Scalar[pl.INDEX] = pl.min(
                        read_begin_inline1093_inline2331__ssa_v0 + lane_page_inline1100_inline2326__ssa_v2, (valid_count_inline1046_inline2398__ssa_v0 - 1) // 32 * 32
                    )
                    logical_page_inline1103_inline2324__ssa_v2: pl.Scalar[pl.INDEX] = (logical_begin_inline1047_inline2394__ssa_v0 + safe_page_begin_inline1091_inline2386__ssa_v2) // 32
                    t__tile_4: pl.Scalar[pl.INT32] = pl.tensor.read(
                        idx_block_table_flat_inline1052_inline2374__ssa_v0,
                        [batch_idx_inline1055_inline2348__ssa_v0 * table_columns_inline1060_inline2338__ssa_v0 + logical_page_inline1103_inline2324__ssa_v2],
                    )
                    physical_block_inline1104_inline2393__ssa_v2: pl.Scalar[pl.INDEX] = pl.cast(t__tile_4, pl.INDEX)
                    key_bytes_inline1098_inline2328__tile_3: pl.Tile[[1, 24576], pl.INT8, pl.MemRef(mem_vec_10, pl.const(98304, pl.INT64), 24576), pl.Mem.Vec] = pl.tile.gather_row(
                        key_bytes_inline1098_inline2328__tile_2,
                        idx_kv_cache__rv_v2,
                        [0, page_begin_inline1086_inline2333__ssa_v2 * 128],
                        [physical_block_inline1104_inline2393__ssa_v2, 0],
                        [1, 4096],
                        transpose=False,
                    )
                    page_begin_inline1086_inline2333__ssa_v3: pl.Scalar[pl.INDEX] = 96
                    lane_page_inline1100_inline2326__ssa_v3: pl.Scalar[pl.INDEX] = (
                        key_aiv_inline1097_inline2330__ssa_v0 * lane_stride_inline1050_inline2402__ssa_v0 + page_begin_inline1086_inline2333__ssa_v3
                    )
                    safe_page_begin_inline1091_inline2386__ssa_v3: pl.Scalar[pl.INDEX] = pl.min(
                        read_begin_inline1093_inline2331__ssa_v0 + lane_page_inline1100_inline2326__ssa_v3, (valid_count_inline1046_inline2398__ssa_v0 - 1) // 32 * 32
                    )
                    logical_page_inline1103_inline2324__ssa_v3: pl.Scalar[pl.INDEX] = (logical_begin_inline1047_inline2394__ssa_v0 + safe_page_begin_inline1091_inline2386__ssa_v3) // 32
                    t__tile_5: pl.Scalar[pl.INT32] = pl.tensor.read(
                        idx_block_table_flat_inline1052_inline2374__ssa_v0,
                        [batch_idx_inline1055_inline2348__ssa_v0 * table_columns_inline1060_inline2338__ssa_v0 + logical_page_inline1103_inline2324__ssa_v3],
                    )
                    physical_block_inline1104_inline2393__ssa_v3: pl.Scalar[pl.INDEX] = pl.cast(t__tile_5, pl.INDEX)
                    key_bytes_inline1098_inline2328__tile_4: pl.Tile[[1, 24576], pl.INT8, pl.MemRef(mem_vec_10, pl.const(98304, pl.INT64), 24576), pl.Mem.Vec] = pl.tile.gather_row(
                        key_bytes_inline1098_inline2328__tile_3,
                        idx_kv_cache__rv_v2,
                        [0, page_begin_inline1086_inline2333__ssa_v3 * 128],
                        [physical_block_inline1104_inline2393__ssa_v3, 0],
                        [1, 4096],
                        transpose=False,
                    )
                    page_begin_inline1086_inline2333__ssa_v4: pl.Scalar[pl.INDEX] = 128
                    lane_page_inline1100_inline2326__ssa_v4: pl.Scalar[pl.INDEX] = (
                        key_aiv_inline1097_inline2330__ssa_v0 * lane_stride_inline1050_inline2402__ssa_v0 + page_begin_inline1086_inline2333__ssa_v4
                    )
                    safe_page_begin_inline1091_inline2386__ssa_v4: pl.Scalar[pl.INDEX] = pl.min(
                        read_begin_inline1093_inline2331__ssa_v0 + lane_page_inline1100_inline2326__ssa_v4, (valid_count_inline1046_inline2398__ssa_v0 - 1) // 32 * 32
                    )
                    logical_page_inline1103_inline2324__ssa_v4: pl.Scalar[pl.INDEX] = (logical_begin_inline1047_inline2394__ssa_v0 + safe_page_begin_inline1091_inline2386__ssa_v4) // 32
                    t__tile_6: pl.Scalar[pl.INT32] = pl.tensor.read(
                        idx_block_table_flat_inline1052_inline2374__ssa_v0,
                        [batch_idx_inline1055_inline2348__ssa_v0 * table_columns_inline1060_inline2338__ssa_v0 + logical_page_inline1103_inline2324__ssa_v4],
                    )
                    physical_block_inline1104_inline2393__ssa_v4: pl.Scalar[pl.INDEX] = pl.cast(t__tile_6, pl.INDEX)
                    key_bytes_inline1098_inline2328__tile_5: pl.Tile[[1, 24576], pl.INT8, pl.MemRef(mem_vec_10, pl.const(98304, pl.INT64), 24576), pl.Mem.Vec] = pl.tile.gather_row(
                        key_bytes_inline1098_inline2328__tile_4,
                        idx_kv_cache__rv_v2,
                        [0, page_begin_inline1086_inline2333__ssa_v4 * 128],
                        [physical_block_inline1104_inline2393__ssa_v4, 0],
                        [1, 4096],
                        transpose=False,
                    )
                    page_begin_inline1086_inline2333__ssa_v5: pl.Scalar[pl.INDEX] = 160
                    lane_page_inline1100_inline2326__ssa_v5: pl.Scalar[pl.INDEX] = (
                        key_aiv_inline1097_inline2330__ssa_v0 * lane_stride_inline1050_inline2402__ssa_v0 + page_begin_inline1086_inline2333__ssa_v5
                    )
                    safe_page_begin_inline1091_inline2386__ssa_v5: pl.Scalar[pl.INDEX] = pl.min(
                        read_begin_inline1093_inline2331__ssa_v0 + lane_page_inline1100_inline2326__ssa_v5, (valid_count_inline1046_inline2398__ssa_v0 - 1) // 32 * 32
                    )
                    logical_page_inline1103_inline2324__ssa_v5: pl.Scalar[pl.INDEX] = (logical_begin_inline1047_inline2394__ssa_v0 + safe_page_begin_inline1091_inline2386__ssa_v5) // 32
                    t__tile_7: pl.Scalar[pl.INT32] = pl.tensor.read(
                        idx_block_table_flat_inline1052_inline2374__ssa_v0,
                        [batch_idx_inline1055_inline2348__ssa_v0 * table_columns_inline1060_inline2338__ssa_v0 + logical_page_inline1103_inline2324__ssa_v5],
                    )
                    physical_block_inline1104_inline2393__ssa_v5: pl.Scalar[pl.INDEX] = pl.cast(t__tile_7, pl.INDEX)
                    key_bytes_inline1098_inline2328__tile_6: pl.Tile[[1, 24576], pl.INT8, pl.MemRef(mem_vec_10, pl.const(98304, pl.INT64), 24576), pl.Mem.Vec] = pl.tile.gather_row(
                        key_bytes_inline1098_inline2328__tile_5,
                        idx_kv_cache__rv_v2,
                        [0, page_begin_inline1086_inline2333__ssa_v5 * 128],
                        [physical_block_inline1104_inline2393__ssa_v5, 0],
                        [1, 4096],
                        transpose=False,
                    )
                    key_rows_inline1072_inline2323__tile: pl.Tile[[192, 128], pl.INT8, pl.MemRef(mem_vec_10, pl.const(98304, pl.INT64), 24576), pl.Mem.Vec] = pl.tile.reshape(
                        key_bytes_inline1098_inline2328__tile_6, [192, 128]
                    )
                    pl.tile.tpush_to_aic(key_rows_inline1072_inline2323__tile, split=1)
                    score_shard_inline1095_inline2318__tile: pl.Tile[[64, 192], pl.INT32, pl.Mem.Vec] = pl.tile.tpop_from_aic(split=2)
                    t__tile_8: pl.Tile[[64, 192], pl.FP32, pl.MemRef(mem_vec_10, pl.const(98304, pl.INT64), 49152), pl.Mem.Vec] = pl.tile.cast(
                        score_shard_inline1095_inline2318__tile, target_type=pl.FP32, mode="round"
                    )
                    pl.system.tfree_to_aic(score_shard_inline1095_inline2318__tile, split=2)
                    score_fp32_inline1108_inline2317__tile: pl.Tile[[64, 192], pl.FP32, pl.MemRef(mem_vec_10, pl.const(98304, pl.INT64), 49152), pl.Mem.Vec] = pl.tile.maximums(t__tile_8, 0.0)
                    t__tile_9: pl.Tile[[64, 192], pl.FP32, pl.MemRef(mem_vec_10, pl.const(98304, pl.INT64), 49152), pl.Mem.Vec] = pl.tile.muls(score_fp32_inline1108_inline2317__tile, 0.0009765625)
                    score_half_inline1062_inline2366__tile: pl.Tile[[64, 192], pl.FP16, pl.MemRef(mem_vec_10, pl.const(98304, pl.INT64), 24576), pl.Mem.Vec] = pl.tile.cast(
                        t__tile_9, target_type=pl.FP16, mode="rint"
                    )
                    pl.tile.tpush_to_aic(score_half_inline1062_inline2366__tile, split=2)
                    aiv_id_inline1078_inline2314__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_subblock_idx()
                    lane_begin_inline1063_inline2313__ssa_v0: pl.Scalar[pl.INDEX] = aiv_id_inline1078_inline2314__ssa_v0 * lane_stride_inline1050_inline2402__ssa_v0
                    lane_valid_rows_inline1056_inline2368__ssa_v0: pl.Scalar[pl.INDEX] = pl.max(
                        pl.min(valid_count_inline1046_inline2398__ssa_v0 - read_begin_inline1093_inline2331__ssa_v0 - lane_begin_inline1063_inline2313__ssa_v0, 192), 0
                    )
                    kv_scale_bytes_inline1101_inline2312__tile: pl.Tile[[1, 384], pl.INT8, pl.MemRef(mem_vec_10, pl.const(99072, pl.INT64), 384), pl.Mem.Vec] = pl.tile.create(
                        [1, 384], dtype=pl.INT8, target_memory=pl.Mem.Vec
                    )
                    scale_page_begin_inline1105_inline2350__ssa_v0: pl.Scalar[pl.INDEX] = 0
                    safe_scale_begin_inline1102_inline2310__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(
                        read_begin_inline1093_inline2331__ssa_v0 + lane_begin_inline1063_inline2313__ssa_v0 + scale_page_begin_inline1105_inline2350__ssa_v0,
                        (valid_count_inline1046_inline2398__ssa_v0 - 1) // 32 * 32,
                    )
                    scale_logical_page_inline1045_inline2309__ssa_v0: pl.Scalar[pl.INDEX] = (logical_begin_inline1047_inline2394__ssa_v0 + safe_scale_begin_inline1102_inline2310__ssa_v0) // 32
                    t__tile_10: pl.Scalar[pl.INT32] = pl.tensor.read(
                        idx_block_table_flat_inline1052_inline2374__ssa_v0,
                        [batch_idx_inline1055_inline2348__ssa_v0 * table_columns_inline1060_inline2338__ssa_v0 + scale_logical_page_inline1045_inline2309__ssa_v0],
                    )
                    scale_block_inline1043_inline2308__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(t__tile_10, pl.INDEX)
                    kv_scale_bytes_inline1101_inline2312__tile_1: pl.Tile[[1, 384], pl.INT8, pl.MemRef(mem_vec_10, pl.const(99072, pl.INT64), 384), pl.Mem.Vec] = pl.tile.gather_row(
                        kv_scale_bytes_inline1101_inline2312__tile,
                        idx_kv_cache__rv_v2,
                        [0, scale_page_begin_inline1105_inline2350__ssa_v0 * 2],
                        [scale_block_inline1043_inline2308__ssa_v0, 4096],
                        [1, 64],
                        transpose=False,
                    )
                    scale_page_begin_inline1105_inline2350__ssa_v1: pl.Scalar[pl.INDEX] = 32
                    safe_scale_begin_inline1102_inline2310__ssa_v1: pl.Scalar[pl.INDEX] = pl.min(
                        read_begin_inline1093_inline2331__ssa_v0 + lane_begin_inline1063_inline2313__ssa_v0 + scale_page_begin_inline1105_inline2350__ssa_v1,
                        (valid_count_inline1046_inline2398__ssa_v0 - 1) // 32 * 32,
                    )
                    scale_logical_page_inline1045_inline2309__ssa_v1: pl.Scalar[pl.INDEX] = (logical_begin_inline1047_inline2394__ssa_v0 + safe_scale_begin_inline1102_inline2310__ssa_v1) // 32
                    t__tile_11: pl.Scalar[pl.INT32] = pl.tensor.read(
                        idx_block_table_flat_inline1052_inline2374__ssa_v0,
                        [batch_idx_inline1055_inline2348__ssa_v0 * table_columns_inline1060_inline2338__ssa_v0 + scale_logical_page_inline1045_inline2309__ssa_v1],
                    )
                    scale_block_inline1043_inline2308__ssa_v1: pl.Scalar[pl.INDEX] = pl.cast(t__tile_11, pl.INDEX)
                    kv_scale_bytes_inline1101_inline2312__tile_2: pl.Tile[[1, 384], pl.INT8, pl.MemRef(mem_vec_10, pl.const(99072, pl.INT64), 384), pl.Mem.Vec] = pl.tile.gather_row(
                        kv_scale_bytes_inline1101_inline2312__tile_1,
                        idx_kv_cache__rv_v2,
                        [0, scale_page_begin_inline1105_inline2350__ssa_v1 * 2],
                        [scale_block_inline1043_inline2308__ssa_v1, 4096],
                        [1, 64],
                        transpose=False,
                    )
                    scale_page_begin_inline1105_inline2350__ssa_v2: pl.Scalar[pl.INDEX] = 64
                    safe_scale_begin_inline1102_inline2310__ssa_v2: pl.Scalar[pl.INDEX] = pl.min(
                        read_begin_inline1093_inline2331__ssa_v0 + lane_begin_inline1063_inline2313__ssa_v0 + scale_page_begin_inline1105_inline2350__ssa_v2,
                        (valid_count_inline1046_inline2398__ssa_v0 - 1) // 32 * 32,
                    )
                    scale_logical_page_inline1045_inline2309__ssa_v2: pl.Scalar[pl.INDEX] = (logical_begin_inline1047_inline2394__ssa_v0 + safe_scale_begin_inline1102_inline2310__ssa_v2) // 32
                    t__tile_12: pl.Scalar[pl.INT32] = pl.tensor.read(
                        idx_block_table_flat_inline1052_inline2374__ssa_v0,
                        [batch_idx_inline1055_inline2348__ssa_v0 * table_columns_inline1060_inline2338__ssa_v0 + scale_logical_page_inline1045_inline2309__ssa_v2],
                    )
                    scale_block_inline1043_inline2308__ssa_v2: pl.Scalar[pl.INDEX] = pl.cast(t__tile_12, pl.INDEX)
                    kv_scale_bytes_inline1101_inline2312__tile_3: pl.Tile[[1, 384], pl.INT8, pl.MemRef(mem_vec_10, pl.const(99072, pl.INT64), 384), pl.Mem.Vec] = pl.tile.gather_row(
                        kv_scale_bytes_inline1101_inline2312__tile_2,
                        idx_kv_cache__rv_v2,
                        [0, scale_page_begin_inline1105_inline2350__ssa_v2 * 2],
                        [scale_block_inline1043_inline2308__ssa_v2, 4096],
                        [1, 64],
                        transpose=False,
                    )
                    scale_page_begin_inline1105_inline2350__ssa_v3: pl.Scalar[pl.INDEX] = 96
                    safe_scale_begin_inline1102_inline2310__ssa_v3: pl.Scalar[pl.INDEX] = pl.min(
                        read_begin_inline1093_inline2331__ssa_v0 + lane_begin_inline1063_inline2313__ssa_v0 + scale_page_begin_inline1105_inline2350__ssa_v3,
                        (valid_count_inline1046_inline2398__ssa_v0 - 1) // 32 * 32,
                    )
                    scale_logical_page_inline1045_inline2309__ssa_v3: pl.Scalar[pl.INDEX] = (logical_begin_inline1047_inline2394__ssa_v0 + safe_scale_begin_inline1102_inline2310__ssa_v3) // 32
                    t__tile_13: pl.Scalar[pl.INT32] = pl.tensor.read(
                        idx_block_table_flat_inline1052_inline2374__ssa_v0,
                        [batch_idx_inline1055_inline2348__ssa_v0 * table_columns_inline1060_inline2338__ssa_v0 + scale_logical_page_inline1045_inline2309__ssa_v3],
                    )
                    scale_block_inline1043_inline2308__ssa_v3: pl.Scalar[pl.INDEX] = pl.cast(t__tile_13, pl.INDEX)
                    kv_scale_bytes_inline1101_inline2312__tile_4: pl.Tile[[1, 384], pl.INT8, pl.MemRef(mem_vec_10, pl.const(99072, pl.INT64), 384), pl.Mem.Vec] = pl.tile.gather_row(
                        kv_scale_bytes_inline1101_inline2312__tile_3,
                        idx_kv_cache__rv_v2,
                        [0, scale_page_begin_inline1105_inline2350__ssa_v3 * 2],
                        [scale_block_inline1043_inline2308__ssa_v3, 4096],
                        [1, 64],
                        transpose=False,
                    )
                    scale_page_begin_inline1105_inline2350__ssa_v4: pl.Scalar[pl.INDEX] = 128
                    safe_scale_begin_inline1102_inline2310__ssa_v4: pl.Scalar[pl.INDEX] = pl.min(
                        read_begin_inline1093_inline2331__ssa_v0 + lane_begin_inline1063_inline2313__ssa_v0 + scale_page_begin_inline1105_inline2350__ssa_v4,
                        (valid_count_inline1046_inline2398__ssa_v0 - 1) // 32 * 32,
                    )
                    scale_logical_page_inline1045_inline2309__ssa_v4: pl.Scalar[pl.INDEX] = (logical_begin_inline1047_inline2394__ssa_v0 + safe_scale_begin_inline1102_inline2310__ssa_v4) // 32
                    t__tile_14: pl.Scalar[pl.INT32] = pl.tensor.read(
                        idx_block_table_flat_inline1052_inline2374__ssa_v0,
                        [batch_idx_inline1055_inline2348__ssa_v0 * table_columns_inline1060_inline2338__ssa_v0 + scale_logical_page_inline1045_inline2309__ssa_v4],
                    )
                    scale_block_inline1043_inline2308__ssa_v4: pl.Scalar[pl.INDEX] = pl.cast(t__tile_14, pl.INDEX)
                    kv_scale_bytes_inline1101_inline2312__tile_5: pl.Tile[[1, 384], pl.INT8, pl.MemRef(mem_vec_10, pl.const(99072, pl.INT64), 384), pl.Mem.Vec] = pl.tile.gather_row(
                        kv_scale_bytes_inline1101_inline2312__tile_4,
                        idx_kv_cache__rv_v2,
                        [0, scale_page_begin_inline1105_inline2350__ssa_v4 * 2],
                        [scale_block_inline1043_inline2308__ssa_v4, 4096],
                        [1, 64],
                        transpose=False,
                    )
                    scale_page_begin_inline1105_inline2350__ssa_v5: pl.Scalar[pl.INDEX] = 160
                    safe_scale_begin_inline1102_inline2310__ssa_v5: pl.Scalar[pl.INDEX] = pl.min(
                        read_begin_inline1093_inline2331__ssa_v0 + lane_begin_inline1063_inline2313__ssa_v0 + scale_page_begin_inline1105_inline2350__ssa_v5,
                        (valid_count_inline1046_inline2398__ssa_v0 - 1) // 32 * 32,
                    )
                    scale_logical_page_inline1045_inline2309__ssa_v5: pl.Scalar[pl.INDEX] = (logical_begin_inline1047_inline2394__ssa_v0 + safe_scale_begin_inline1102_inline2310__ssa_v5) // 32
                    t__tile_15: pl.Scalar[pl.INT32] = pl.tensor.read(
                        idx_block_table_flat_inline1052_inline2374__ssa_v0,
                        [batch_idx_inline1055_inline2348__ssa_v0 * table_columns_inline1060_inline2338__ssa_v0 + scale_logical_page_inline1045_inline2309__ssa_v5],
                    )
                    scale_block_inline1043_inline2308__ssa_v5: pl.Scalar[pl.INDEX] = pl.cast(t__tile_15, pl.INDEX)
                    kv_scale_bytes_inline1101_inline2312__tile_6: pl.Tile[[1, 384], pl.INT8, pl.MemRef(mem_vec_10, pl.const(99072, pl.INT64), 384), pl.Mem.Vec] = pl.tile.gather_row(
                        kv_scale_bytes_inline1101_inline2312__tile_5,
                        idx_kv_cache__rv_v2,
                        [0, scale_page_begin_inline1105_inline2350__ssa_v5 * 2],
                        [scale_block_inline1043_inline2308__ssa_v5, 4096],
                        [1, 64],
                        transpose=False,
                    )
                    kv_scale_inline1044_inline2399__tile: pl.Tile[[1, 192], pl.FP16, pl.MemRef(mem_vec_10, pl.const(99072, pl.INT64), 384), pl.Mem.Vec] = pl.tile.reinterpret_view(
                        kv_scale_bytes_inline1101_inline2312__tile_6, dtype=pl.FP16
                    )
                    weighted_shard_inline1076_inline2337__tile: pl.Tile[[16, 192], pl.FP32, pl.Mem.Vec] = pl.tile.tpop_from_aic(split=2)
                    score_row_inline1042_inline2325__tile: pl.Tile[[1, 192], pl.FP32, pl.Mem.Vec] = pl.tile.slice(weighted_shard_inline1076_inline2337__tile, [1, 192], [0, 0])
                    t__tile_16: pl.Tile[[1, 192], pl.FP32, pl.MemRef(mem_vec_10, pl.const(98304, pl.INT64), 768), pl.Mem.Vec] = pl.tile.cast(
                        kv_scale_inline1044_inline2399__tile, target_type=pl.FP32, mode="round"
                    )
                    score_row_v1_inline1041_inline2307__tile: pl.Tile[[1, 192], pl.FP32, pl.MemRef(mem_vec_10, pl.const(98304, pl.INT64), 768), pl.Mem.Vec] = pl.tile.mul(
                        score_row_inline1042_inline2325__tile, t__tile_16
                    )
                    pl.system.tfree_to_aic(weighted_shard_inline1076_inline2337__tile, split=2)
                    score_row_id_inline1054_inline2306__ssa_v0: pl.Scalar[pl.INDEX] = single_leaf_inline1077_inline2354__ssa_v0 * query_inline1074_inline2340__ssa_v0 + (
                        1 - single_leaf_inline1077_inline2354__ssa_v0
                    ) * (worker_inline1094_inline2380__ssa_v0 * 2 + aiv_id_inline1078_inline2314__ssa_v0)
                    score_col_inline1089_inline2305__ssa_v0: pl.Scalar[pl.INDEX] = (
                        single_leaf_inline1077_inline2354__ssa_v0 * (read_begin_inline1093_inline2331__ssa_v0 + lane_begin_inline1063_inline2313__ssa_v0)
                        + (1 - single_leaf_inline1077_inline2354__ssa_v0) * score_begin_inline1090_inline2391__idx_v0
                    )
                    if 0 < lane_valid_rows_inline1056_inline2368__ssa_v0:
                        score_valid_inline1048_inline2322__tile: pl.Tile[
                            [1, 192], pl.FP32, pl.MemRef(mem_vec_10, pl.const(98304, pl.INT64), 768), pl.Mem.Vec, pl.TileView(valid_shape=[1, lane_valid_rows_inline1056_inline2368__ssa_v0])
                        ] = pl.tile.set_validshape(score_row_v1_inline1041_inline2307__tile, 1, lane_valid_rows_inline1056_inline2368__ssa_v0)
                        score_arena_inline1071_inline2347__tile: pl.Tensor[[384, 8192], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 12582912)] = pl.tile.store(
                            score_valid_inline1048_inline2322__tile, [score_row_id_inline1054_inline2306__ssa_v0, score_col_inline1089_inline2305__ssa_v0], score_arena_inline1071_inline2347__iter_v3
                        )
                        score_arena_inline1071_inline2347__phi_v6: pl.Tensor[[384, 8192], pl.FP32, pl.MemRef("mem_ddr_17", pl.const(0, pl.INT64), 12582912)] = pl.yield_(
                            score_arena_inline1071_inline2347__tile
                        )
                    else:
                        score_arena_inline1071_inline2347__phi_v6: pl.Tensor[[384, 8192], pl.FP32, pl.MemRef("mem_ddr_17", pl.const(0, pl.INT64), 12582912)] = pl.yield_(
                            score_arena_inline1071_inline2347__iter_v3
                        )
                    score_arena_inline1071_inline2347__rv_v4: pl.Tensor[[384, 8192], pl.FP32, pl.MemRef("mem_ddr_18", pl.const(0, pl.INT64), 12582912)] = pl.yield_(
                        score_arena_inline1071_inline2347__phi_v6
                    )
                if single_leaf_inline1077_inline2354__ssa_v0 == 0:
                    sort_lane_inline1040_inline2304__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_subblock_idx()
                    half_begin_inline1039_inline2302__ssa_v0: pl.Scalar[pl.INDEX] = (
                        logical_begin_inline1047_inline2394__ssa_v0 + sort_lane_inline1040_inline2304__ssa_v0 * lane_span_inline1069_inline2401__ssa_v0
                    )
                    half_valid_inline1037_inline2356__ssa_v0: pl.Scalar[pl.INDEX] = pl.max(
                        pl.min(valid_count_inline1046_inline2398__ssa_v0 - sort_lane_inline1040_inline2304__ssa_v0 * lane_span_inline1069_inline2401__ssa_v0, lane_span_inline1069_inline2401__ssa_v0),
                        0,
                    )
                    half_slot_inline1036_inline2319__ssa_v0: pl.Scalar[pl.INDEX] = (
                        query_inline1074_inline2340__ssa_v0 * 64 + leaf_inline1070_inline2390__ssa_v0 * 2 + sort_lane_inline1040_inline2304__ssa_v0
                    )
                    if 0 < half_valid_inline1037_inline2356__ssa_v0:
                        logical_begin_i32_inline3217__ssa_v0: pl.Scalar[pl.INT32] = pl.cast(half_begin_inline1039_inline2302__ssa_v0, pl.INT32)
                        if half_valid_inline1037_inline2356__ssa_v0 <= 512:
                            t__ci_tmp_v0: pl.Tile[[1, 192], pl.FP32, pl.MemRef(mem_vec_10, pl.const(100352, pl.INT64), 768), pl.Mem.Vec] = pl.tile.create(
                                [1, 192], dtype=pl.FP32, target_memory=pl.Mem.Vec
                            )
                            t__tmp_v270: pl.Tile[[1, 512], pl.INT32, pl.MemRef(mem_vec_10, pl.const(98304, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.ci(
                                pl.const(0, pl.INT32), [1, 512], tmp=t__ci_tmp_v0, dtype=pl.INT32, descending=False
                            )
                            short_indices_inline3232__ssa_v0: pl.Tile[[1, 512], pl.INT32, pl.MemRef(mem_vec_10, pl.const(102400, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.adds(
                                t__tmp_v270, logical_begin_i32_inline3217__ssa_v0
                            )
                            short_raw_inline3210__ssa_v0: pl.Tile[
                                [1, 512], pl.FP32, pl.MemRef(mem_vec_10, pl.const(98304, pl.INT64), 2048), pl.Mem.Vec, pl.TileView(valid_shape=[1, half_valid_inline1037_inline2356__ssa_v0])
                            ] = pl.tile.load(
                                score_arena_inline1071_inline2347__rv_v4,
                                [worker_inline1094_inline2380__ssa_v0 * 2 + sort_lane_inline1040_inline2304__ssa_v0, 0],
                                [1, 512],
                                [1, half_valid_inline1037_inline2356__ssa_v0],
                                target_memory=pl.Mem.Vec,
                            )
                            short_scores_inline3222__ssa_v0: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_10, pl.const(98304, pl.INT64), 2048), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = (
                                pl.tile.fillpad(short_raw_inline3210__ssa_v0, pad_value=pl.PadValue.min)
                            )
                            short_scores_v1_inline3215__ssa_v0: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_10, pl.const(104448, pl.INT64), 2048), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = (
                                pl.tile.maximums(short_scores_inline3222__ssa_v0, -3.4028234663852886e38)
                            )
                            t__tmp_v271: pl.Tile[[1, 512], pl.UINT32, pl.MemRef(mem_vec_10, pl.const(102400, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.reinterpret_view(
                                short_indices_inline3232__ssa_v0, dtype=pl.UINT32
                            )
                            short_pairs_inline3218__ssa_v0: pl.Tile[[1, 1024], pl.FP32, pl.MemRef(mem_vec_10, pl.const(98304, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = (
                                pl.tile.sort32(short_scores_v1_inline3215__ssa_v0, t__tmp_v271)
                            )
                            short_pairs_v1_inline3212__ssa_v0: pl.Tile[[1, 1024], pl.FP32, pl.MemRef(mem_vec_10, pl.const(102400, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = (
                                pl.tile.mrgsort_format1(short_pairs_inline3218__ssa_v0, pl.const(64, pl.INT32))
                            )
                            short_pairs_v2_inline3209__ssa_v0: pl.Tile[[1, 1024], pl.FP32, pl.MemRef(mem_vec_10, pl.const(98304, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = (
                                pl.tile.mrgsort_format1(short_pairs_v1_inline3212__ssa_v0, pl.const(256, pl.INT32))
                            )
                            pair_arena_inline1064_inline2363__store: pl.Tensor[[24576, 1024], pl.FP32, pl.MemRef("mem_ddr_7", pl.const(0, pl.INT64), 100663296)] = pl.tile.store(
                                short_pairs_v2_inline3209__ssa_v0, [half_slot_inline1036_inline2319__ssa_v0, 0], pair_arena_inline1064_inline2363__ssa_v0
                            )
                        else:
                            if half_valid_inline1037_inline2356__ssa_v0 <= 1024:
                                t__ci_tmp_v1: pl.Tile[[1, 192], pl.FP32, pl.MemRef(mem_vec_10, pl.const(102400, pl.INT64), 768), pl.Mem.Vec] = pl.tile.create(
                                    [1, 192], dtype=pl.FP32, target_memory=pl.Mem.Vec
                                )
                                t__tmp_v272: pl.Tile[[1, 1024], pl.INT32, pl.MemRef(mem_vec_10, pl.const(98304, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.ci(
                                    pl.const(0, pl.INT32), [1, 1024], tmp=t__ci_tmp_v1, dtype=pl.INT32, descending=False
                                )
                                small_indices_inline3211__ssa_v0: pl.Tile[[1, 1024], pl.INT32, pl.MemRef(mem_vec_10, pl.const(106496, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.adds(
                                    t__tmp_v272, logical_begin_i32_inline3217__ssa_v0
                                )
                                small_raw_inline3220__ssa_v0: pl.Tile[
                                    [1, 1024], pl.FP32, pl.MemRef(mem_vec_10, pl.const(98304, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(valid_shape=[1, half_valid_inline1037_inline2356__ssa_v0])
                                ] = pl.tile.load(
                                    score_arena_inline1071_inline2347__rv_v4,
                                    [worker_inline1094_inline2380__ssa_v0 * 2 + sort_lane_inline1040_inline2304__ssa_v0, 0],
                                    [1, 1024],
                                    [1, half_valid_inline1037_inline2356__ssa_v0],
                                    target_memory=pl.Mem.Vec,
                                )
                                small_scores_inline3206__ssa_v0: pl.Tile[[1, 1024], pl.FP32, pl.MemRef(mem_vec_10, pl.const(98304, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = (
                                    pl.tile.fillpad(small_raw_inline3220__ssa_v0, pad_value=pl.PadValue.min)
                                )
                                small_scores_v1_inline3208__ssa_v0: pl.Tile[
                                    [1, 1024], pl.FP32, pl.MemRef(mem_vec_10, pl.const(110592, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)
                                ] = pl.tile.maximums(small_scores_inline3206__ssa_v0, -3.4028234663852886e38)
                                t__tmp_v273: pl.Tile[[1, 1024], pl.UINT32, pl.MemRef(mem_vec_10, pl.const(106496, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.reinterpret_view(
                                    small_indices_inline3211__ssa_v0, dtype=pl.UINT32
                                )
                                small_pairs_inline3221__ssa_v0: pl.Tile[[1, 2048], pl.FP32, pl.MemRef(mem_vec_10, pl.const(98304, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = (
                                    pl.tile.sort32(small_scores_v1_inline3208__ssa_v0, t__tmp_v273)
                                )
                                small_pairs_v1_inline3235__ssa_v0: pl.Tile[
                                    [1, 2048], pl.FP32, pl.MemRef(mem_vec_10, pl.const(106496, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)
                                ] = pl.tile.mrgsort_format1(small_pairs_inline3221__ssa_v0, pl.const(64, pl.INT32))
                                small_pairs_v2_inline3226__ssa_v0: pl.Tile[[1, 2048], pl.FP32, pl.MemRef(mem_vec_10, pl.const(98304, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = (
                                    pl.tile.mrgsort_format1(small_pairs_v1_inline3235__ssa_v0, pl.const(256, pl.INT32))
                                )
                                small_left_inline3227__ssa_v0: pl.Tile[[1, 1024], pl.FP32, pl.MemRef(mem_vec_10, pl.const(98304, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = (
                                    pl.tile.slice(small_pairs_v2_inline3226__ssa_v0, [1, 1024], [0, 0])
                                )
                                small_right_inline3213__ssa_v0: pl.Tile[[1, 1024], pl.FP32, pl.MemRef(mem_vec_10, pl.const(102400, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = (
                                    pl.tile.slice(small_pairs_v2_inline3226__ssa_v0, [1, 1024], [0, 1024])
                                )
                                small_tmp_inline3229__ssa_v0: pl.Tile[[1, 2048], pl.FP32, pl.MemRef(mem_vec_10, pl.const(106496, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.create(
                                    [1, 2048], dtype=pl.FP32, target_memory=pl.Mem.Vec
                                )
                                small_merged_inline3231__ssa_v0: pl.Tile[[1, 2048], pl.FP32, pl.MemRef(mem_vec_10, pl.const(114688, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = (
                                    pl.tile.mrgsort_format2(small_left_inline3227__ssa_v0, small_right_inline3213__ssa_v0, small_tmp_inline3229__ssa_v0, exhausted=False)
                                )
                                small_top_inline3233__ssa_v0: pl.Tile[[1, 1024], pl.FP32, pl.MemRef(mem_vec_10, pl.const(114688, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = (
                                    pl.tile.slice(small_merged_inline3231__ssa_v0, [1, 1024], [0, 0])
                                )
                                pair_arena_inline1064_inline2363__store_v0: pl.Tensor[[24576, 1024], pl.FP32, pl.MemRef("mem_ddr_7", pl.const(0, pl.INT64), 100663296)] = pl.tile.store(
                                    small_top_inline3233__ssa_v0, [half_slot_inline1036_inline2319__ssa_v0, 0], pair_arena_inline1064_inline2363__ssa_v0
                                )
                            else:
                                if half_valid_inline1037_inline2356__ssa_v0 <= 2048:
                                    t__ci_tmp_v2: pl.Tile[[1, 192], pl.FP32, pl.MemRef(mem_vec_10, pl.const(106496, pl.INT64), 768), pl.Mem.Vec] = pl.tile.create(
                                        [1, 192], dtype=pl.FP32, target_memory=pl.Mem.Vec
                                    )
                                    t__tmp_v274: pl.Tile[[1, 2048], pl.INT32, pl.MemRef(mem_vec_10, pl.const(98304, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.ci(
                                        pl.const(0, pl.INT32), [1, 2048], tmp=t__ci_tmp_v2, dtype=pl.INT32, descending=False
                                    )
                                    leaf_indices_inline3219__ssa_v0: pl.Tile[[1, 2048], pl.INT32, pl.MemRef(mem_vec_10, pl.const(114688, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.adds(
                                        t__tmp_v274, logical_begin_i32_inline3217__ssa_v0
                                    )
                                    leaf_scores_raw_inline3228__ssa_v0: pl.Tile[
                                        [1, 2048], pl.FP32, pl.MemRef(mem_vec_10, pl.const(98304, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[1, half_valid_inline1037_inline2356__ssa_v0])
                                    ] = pl.tile.load(
                                        score_arena_inline1071_inline2347__rv_v4,
                                        [worker_inline1094_inline2380__ssa_v0 * 2 + sort_lane_inline1040_inline2304__ssa_v0, 0],
                                        [1, 2048],
                                        [1, half_valid_inline1037_inline2356__ssa_v0],
                                        target_memory=pl.Mem.Vec,
                                    )
                                    leaf_scores_inline3234__ssa_v0: pl.Tile[
                                        [1, 2048], pl.FP32, pl.MemRef(mem_vec_10, pl.const(98304, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)
                                    ] = pl.tile.fillpad(leaf_scores_raw_inline3228__ssa_v0, pad_value=pl.PadValue.min)
                                    leaf_scores_v1_inline3230__ssa_v0: pl.Tile[
                                        [1, 2048], pl.FP32, pl.MemRef(mem_vec_10, pl.const(122880, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)
                                    ] = pl.tile.maximums(leaf_scores_inline3234__ssa_v0, -3.4028234663852886e38)
                                    t__tmp_v275: pl.Tile[[1, 2048], pl.UINT32, pl.MemRef(mem_vec_10, pl.const(114688, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.reinterpret_view(
                                        leaf_indices_inline3219__ssa_v0, dtype=pl.UINT32
                                    )
                                    pairs_inline3236__ssa_v0: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_10, pl.const(98304, pl.INT64), 16384), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = (
                                        pl.tile.sort32(leaf_scores_v1_inline3230__ssa_v0, t__tmp_v275)
                                    )
                                    pairs_v1_inline3237__ssa_v0: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_10, pl.const(114688, pl.INT64), 16384), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = (
                                        pl.tile.mrgsort_format1(pairs_inline3236__ssa_v0, pl.const(64, pl.INT32))
                                    )
                                    pairs_v2_inline3224__ssa_v0: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_10, pl.const(98304, pl.INT64), 16384), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = (
                                        pl.tile.mrgsort_format1(pairs_v1_inline3237__ssa_v0, pl.const(256, pl.INT32))
                                    )
                                    pairs_v3_inline3238__ssa_v0: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_10, pl.const(114688, pl.INT64), 16384), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = (
                                        pl.tile.mrgsort_format1(pairs_v2_inline3224__ssa_v0, pl.const(1024, pl.INT32))
                                    )
                                    t__tmp_v276: pl.Tile[[1, 1024], pl.FP32, pl.MemRef(mem_vec_10, pl.const(114688, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(
                                        pairs_v3_inline3238__ssa_v0, [1, 1024], [0, 0]
                                    )
                                    pair_arena_inline1064_inline2363__store_v1: pl.Tensor[[24576, 1024], pl.FP32, pl.MemRef("mem_ddr_7", pl.const(0, pl.INT64), 100663296)] = pl.tile.store(
                                        t__tmp_v276, [half_slot_inline1036_inline2319__ssa_v0, 0], pair_arena_inline1064_inline2363__ssa_v0
                                    )
                                else:
                                    t__ci_tmp_v3: pl.Tile[[1, 192], pl.FP32, pl.MemRef(mem_vec_10, pl.const(114688, pl.INT64), 768), pl.Mem.Vec] = pl.tile.create(
                                        [1, 192], dtype=pl.FP32, target_memory=pl.Mem.Vec
                                    )
                                    t__tmp_v277: pl.Tile[[1, 4096], pl.INT32, pl.MemRef(mem_vec_10, pl.const(98304, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.ci(
                                        pl.const(0, pl.INT32), [1, 4096], tmp=t__ci_tmp_v3, dtype=pl.INT32, descending=False
                                    )
                                    medium_indices_inline3223__ssa_v0: pl.Tile[[1, 4096], pl.INT32, pl.MemRef(mem_vec_10, pl.const(131072, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.adds(
                                        t__tmp_v277, logical_begin_i32_inline3217__ssa_v0
                                    )
                                    medium_scores_raw_inline3205__ssa_v0: pl.Tile[
                                        [1, 4096], pl.FP32, pl.MemRef(mem_vec_10, pl.const(98304, pl.INT64), 16384), pl.Mem.Vec, pl.TileView(valid_shape=[1, half_valid_inline1037_inline2356__ssa_v0])
                                    ] = pl.tile.load(
                                        score_arena_inline1071_inline2347__rv_v4,
                                        [worker_inline1094_inline2380__ssa_v0 * 2 + sort_lane_inline1040_inline2304__ssa_v0, 0],
                                        [1, 4096],
                                        [1, half_valid_inline1037_inline2356__ssa_v0],
                                        target_memory=pl.Mem.Vec,
                                    )
                                    medium_scores_inline3225__ssa_v0: pl.Tile[
                                        [1, 4096], pl.FP32, pl.MemRef(mem_vec_10, pl.const(98304, pl.INT64), 16384), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)
                                    ] = pl.tile.fillpad(medium_scores_raw_inline3205__ssa_v0, pad_value=pl.PadValue.min)
                                    medium_scores_v1_inline3204__ssa_v0: pl.Tile[
                                        [1, 4096], pl.FP32, pl.MemRef(mem_vec_56, pl.const(147456, pl.INT64), 16384), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)
                                    ] = pl.tile.maximums(medium_scores_inline3225__ssa_v0, -3.4028234663852886e38)
                                    t__tmp_v278: pl.Tile[[1, 4096], pl.UINT32, pl.MemRef(mem_vec_10, pl.const(131072, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.reinterpret_view(
                                        medium_indices_inline3223__ssa_v0, dtype=pl.UINT32
                                    )
                                    medium_pairs_inline3214__ssa_v0: pl.Tile[
                                        [1, 8192], pl.FP32, pl.MemRef(mem_vec_10, pl.const(98304, pl.INT64), 32768), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)
                                    ] = pl.tile.sort32(medium_scores_v1_inline3204__ssa_v0, t__tmp_v278)
                                    medium_pairs_v1_inline3202__ssa_v0: pl.Tile[
                                        [1, 8192], pl.FP32, pl.MemRef(mem_vec_56, pl.const(147456, pl.INT64), 32768), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)
                                    ] = pl.tile.mrgsort_format1(medium_pairs_inline3214__ssa_v0, pl.const(64, pl.INT32))
                                    medium_pairs_v2_inline3203__ssa_v0: pl.Tile[
                                        [1, 8192], pl.FP32, pl.MemRef(mem_vec_10, pl.const(98304, pl.INT64), 32768), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)
                                    ] = pl.tile.mrgsort_format1(medium_pairs_v1_inline3202__ssa_v0, pl.const(256, pl.INT32))
                                    medium_pairs_v3_inline3200__ssa_v0: pl.Tile[
                                        [1, 8192], pl.FP32, pl.MemRef(mem_vec_56, pl.const(147456, pl.INT64), 32768), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)
                                    ] = pl.tile.mrgsort_format1(medium_pairs_v2_inline3203__ssa_v0, pl.const(1024, pl.INT32))
                                    medium_left_inline3216__ssa_v0: pl.Tile[
                                        [1, 1024], pl.FP32, pl.MemRef(mem_vec_56, pl.const(147456, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)
                                    ] = pl.tile.slice(medium_pairs_v3_inline3200__ssa_v0, [1, 1024], [0, 0])
                                    medium_right_inline3207__ssa_v0: pl.Tile[
                                        [1, 1024], pl.FP32, pl.MemRef(mem_vec_56, pl.const(163840, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)
                                    ] = pl.tile.slice(medium_pairs_v3_inline3200__ssa_v0, [1, 1024], [0, 4096])
                                    medium_tmp_inline3201__ssa_v0: pl.Tile[[1, 2048], pl.FP32, pl.MemRef(mem_vec_10, pl.const(98304, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.create(
                                        [1, 2048], dtype=pl.FP32, target_memory=pl.Mem.Vec
                                    )
                                    medium_merged_inline3199__ssa_v0: pl.Tile[
                                        [1, 2048], pl.FP32, pl.MemRef(mem_vec_10, pl.const(106496, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)
                                    ] = pl.tile.mrgsort_format2(medium_right_inline3207__ssa_v0, medium_left_inline3216__ssa_v0, medium_tmp_inline3201__ssa_v0, exhausted=False)
                                    t__tmp_v279: pl.Tile[[1, 1024], pl.FP32, pl.MemRef(mem_vec_10, pl.const(106496, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(
                                        medium_merged_inline3199__ssa_v0, [1, 1024], [0, 0]
                                    )
                                    pair_arena_inline1064_inline2363__store_v2: pl.Tensor[[24576, 1024], pl.FP32, pl.MemRef("mem_ddr_7", pl.const(0, pl.INT64), 100663296)] = pl.tile.store(
                                        t__tmp_v279, [half_slot_inline1036_inline2319__ssa_v0, 0], pair_arena_inline1064_inline2363__ssa_v0
                                    )
                    else:
                        empty_pairs_inline1035_inline2358__ssa_v0: pl.Tile[[1, 1024], pl.FP32, pl.MemRef(mem_vec_10, pl.const(98304, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.full(
                            [1, 1024], dtype=pl.FP32, value=-3.4028234663852886e38
                        )
                        pair_arena_inline1064_inline2363__store_v3: pl.Tensor[[24576, 1024], pl.FP32, pl.MemRef("mem_ddr_7", pl.const(0, pl.INT64), 100663296)] = pl.tile.store(
                            empty_pairs_inline1035_inline2358__ssa_v0, [half_slot_inline1036_inline2319__ssa_v0, 0], pair_arena_inline1064_inline2363__ssa_v0
                        )
                score_arena_inline1071_inline2347__phi_v7: pl.Tensor[[384, 8192], pl.FP32, pl.MemRef("mem_ddr_62", pl.const(0, pl.INT64), 12582912)] = pl.yield_(
                    score_arena_inline1071_inline2347__rv_v4
                )
            else:
                score_arena_inline1071_inline2347__phi_v7: pl.Tensor[[384, 8192], pl.FP32, pl.MemRef("mem_ddr_62", pl.const(0, pl.INT64), 12582912)] = pl.yield_(
                    score_arena_inline1071_inline2347__iter_v1
                )
            score_arena_inline1071_inline2347__rv_v2: pl.Tensor[[384, 8192], pl.FP32, pl.MemRef("mem_ddr_63", pl.const(0, pl.INT64), 12582912)] = pl.yield_(score_arena_inline1071_inline2347__phi_v7)
        return score_arena_inline1071_inline2347__ssa_v0, pair_arena_inline1064_inline2363__ssa_v0

    @pl.function(type=pl.FunctionType.Group, level=pl.Level.CORE_GROUP, role=pl.Role.SubWorker)
    def indexer_score_topk_leaf(
        position_ids__ssa_v0: pl.Tensor[[T_DYN], pl.INT64, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)],
        kv_seq_lens__ssa_v0: pl.Tensor[[B_DYN], pl.INT32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)],
        score_arena_inline1071_inline2347__ssa_v0: pl.InOut[pl.Tensor[[384, 8192], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 12582912)]],
        qr_hadamard_i8_inline221__rv_v2: pl.Tensor[[24576, 128], pl.INT8, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 3145728)],
        coefficients_inline1068_inline2344__ssa_v0: pl.Tensor[[6144, 64], pl.FP16, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 786432)],
        idx_block_table_flat_inline1052_inline2374__ssa_v0: pl.Tensor[[idx_table_len_inline1096_inline2335__ssa_v0], pl.INT32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 0)],
        table_columns_inline1060_inline2338__ssa_v0: pl.Scalar[pl.INDEX],
        idx_kv_cache__rv_v2: pl.Tensor[[IDX_CACHE_BLOCK_NUM_DYN, INDEXER_PAGE_BYTES_DYN], pl.INT8, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 0)],
        pair_arena_inline1064_inline2363__ssa_v0: pl.Out[pl.Tensor[[24576, 1024], pl.FP32, pl.MemRef("mem_ddr_7", pl.const(0, pl.INT64), 100663296)]],
        __gm_pipe_buffer: pl.Out[pl.Tensor[[1], pl.FP32, pl.MemRef("mem_ddr_8", pl.const(0, pl.INT64), 4)]],
    ) -> tuple[pl.Tensor[[384, 8192], pl.FP32], pl.Tensor[[24576, 1024], pl.FP32]]:
        pl.func_attr({"slot_num": 1, "mx_tensor_views_blocked": True, "split_aiv": True})
        self.indexer_score_topk_leaf_aic(
            position_ids__ssa_v0,
            kv_seq_lens__ssa_v0,
            score_arena_inline1071_inline2347__ssa_v0,
            qr_hadamard_i8_inline221__rv_v2,
            coefficients_inline1068_inline2344__ssa_v0,
            idx_block_table_flat_inline1052_inline2374__ssa_v0,
            table_columns_inline1060_inline2338__ssa_v0,
            idx_kv_cache__rv_v2,
            pair_arena_inline1064_inline2363__ssa_v0,
            __gm_pipe_buffer,
            attrs={
                "arg_directions": [
                    pl.adir.input,
                    pl.adir.input,
                    pl.adir.inout,
                    pl.adir.input,
                    pl.adir.input,
                    pl.adir.input,
                    pl.adir.scalar,
                    pl.adir.input,
                    pl.adir.output_existing,
                    pl.adir.output_existing,
                ]
            },
        )
        self.indexer_score_topk_leaf_aiv(
            position_ids__ssa_v0,
            kv_seq_lens__ssa_v0,
            score_arena_inline1071_inline2347__ssa_v0,
            qr_hadamard_i8_inline221__rv_v2,
            coefficients_inline1068_inline2344__ssa_v0,
            idx_block_table_flat_inline1052_inline2374__ssa_v0,
            table_columns_inline1060_inline2338__ssa_v0,
            idx_kv_cache__rv_v2,
            pair_arena_inline1064_inline2363__ssa_v0,
            __gm_pipe_buffer,
            attrs={"arg_directions": [pl.adir.input, pl.adir.input, pl.adir.inout, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.scalar, pl.adir.input, pl.adir.inout, pl.adir.inout]},
        )
        return score_arena_inline1071_inline2347__ssa_v0, pair_arena_inline1064_inline2363__ssa_v0

    @pl.function(type=pl.FunctionType.Spmd)
    def indexer_score_topk_leaf_spmd(
        self,
        position_ids__ssa_v0: pl.Tensor[[T_DYN], pl.INT64, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)],
        kv_seq_lens__ssa_v0: pl.Tensor[[B_DYN], pl.INT32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)],
        score_arena_inline1071_inline2347__ssa_v0: pl.InOut[pl.Tensor[[384, 8192], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 12582912)]],
        qr_hadamard_i8_inline221__rv_v2: pl.Tensor[[24576, 128], pl.INT8, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 3145728)],
        coefficients_inline1068_inline2344__ssa_v0: pl.Tensor[[6144, 64], pl.FP16, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 786432)],
        idx_block_table_flat_inline1052_inline2374__ssa_v0: pl.Tensor[[idx_table_len_inline1096_inline2335__ssa_v0], pl.INT32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 0)],
        table_columns_inline1060_inline2338__ssa_v0: pl.Scalar[pl.INDEX],
        idx_kv_cache__rv_v3: pl.Tensor[[IDX_CACHE_BLOCK_NUM_DYN, INDEXER_PAGE_BYTES_DYN], pl.INT8, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 0)],
        pair_arena_inline1064_inline2363__ssa_v0: pl.Out[pl.Tensor[[24576, 1024], pl.FP32, pl.MemRef("mem_ddr_7", pl.const(0, pl.INT64), 100663296)]],
        __gm_pipe_buffer: pl.Out[pl.Tensor[[1], pl.FP32, pl.MemRef("mem_ddr_8", pl.const(0, pl.INT64), 4)]],
    ) -> tuple[pl.Tensor[[384, 8192], pl.FP32], pl.Tensor[[24576, 1024], pl.FP32]]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        ret__tmp_v0: pl.Tuple[pl.Tensor[[384, 8192], pl.FP32], pl.Tensor[[24576, 1024], pl.FP32]] = self.indexer_score_topk_leaf(
            position_ids__ssa_v0,
            kv_seq_lens__ssa_v0,
            score_arena_inline1071_inline2347__ssa_v0,
            qr_hadamard_i8_inline221__rv_v2,
            coefficients_inline1068_inline2344__ssa_v0,
            idx_block_table_flat_inline1052_inline2374__ssa_v0,
            table_columns_inline1060_inline2338__ssa_v0,
            idx_kv_cache__rv_v3,
            pair_arena_inline1064_inline2363__ssa_v0,
            __gm_pipe_buffer,
            attrs={
                "arg_directions": [
                    pl.adir.input,
                    pl.adir.input,
                    pl.adir.inout,
                    pl.adir.input,
                    pl.adir.input,
                    pl.adir.input,
                    pl.adir.scalar,
                    pl.adir.input,
                    pl.adir.output_existing,
                    pl.adir.output_existing,
                ]
            },
        )
        score_arena_inline1071_inline2347__rv_v2: pl.Tensor[[384, 8192], pl.FP32, pl.MemRef("mem_ddr_9", pl.const(0, pl.INT64), 12582912)] = ret__tmp_v0[0]
        pair_arena_inline1064_inline2363__ssa_v1: pl.Tensor[[24576, 1024], pl.FP32, pl.MemRef("mem_ddr_10", pl.const(0, pl.INT64), 100663296)] = ret__tmp_v0[1]
        return score_arena_inline1071_inline2347__ssa_v0, pair_arena_inline1064_inline2363__ssa_v0

    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def kv_and_cache_write(
        compact_rows_inline2207__ssa_v0: pl.Scalar[pl.INDEX],
        kv_final_inline2183__rv_v2: pl.Tensor[[384, 128], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 196608)],
        idx_kv_scale_values_inline2188__ssa_v0: pl.Out[pl.Tensor[[384, 1], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 1536)]],
        idx_kv_cache__ssa_v0: pl.Out[pl.Tensor[[IDX_CACHE_BLOCK_NUM_DYN, INDEXER_PAGE_BYTES_DYN], pl.INT8, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)]],
        kv_flat_inline2202__ssa_v0: pl.Out[pl.Tensor[[T_DYN, 128], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)]],
        position_ids__ssa_v0: pl.Tensor[[T_DYN], pl.INT64, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 0)],
        idx_row_offsets__ssa_v0: pl.Tensor[[B_DYN], pl.INT32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 0)],
        idx_slot_mapping__ssa_v0: pl.Tensor[[INDEXER_ROWS_DYN, 2], pl.INT32, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 0)],
    ) -> tuple[pl.Tensor[[384, 1], pl.FP32], pl.Tensor[[IDX_CACHE_BLOCK_NUM_DYN, INDEXER_PAGE_BYTES_DYN], pl.INT8]]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        mem_vec_7: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 8192)
        mem_vec_13: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 8192)
        mem_vec_14: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 8192)
        mem_vec_15: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 64)
        wr_blk_inline2208__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        wr_b0_inline2194__ssa_v0: pl.Scalar[pl.INDEX] = wr_blk_inline2208__ssa_v0 * 16
        wr_blk_rows_inline2198__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(compact_rows_inline2207__ssa_v0 - wr_b0_inline2194__ssa_v0, 16)
        t__tile: pl.Tile[[16, 128], pl.FP32, pl.MemRef(mem_vec_7, pl.const(16448, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.load(
            kv_final_inline2183__rv_v2, [wr_b0_inline2194__ssa_v0, 0], [16, 128], [16, 128], target_memory=pl.Mem.Vec
        )
        t__tile_1: pl.Tile[[16, 128], pl.BF16, pl.MemRef(mem_vec_13, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(t__tile, target_type=pl.BF16, mode="rint")
        kv_blk_f32_inline2211__tile: pl.Tile[[16, 128], pl.FP32, pl.MemRef(mem_vec_7, pl.const(16448, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.cast(t__tile_1, target_type=pl.FP32, mode="round")
        t__tile_2: pl.Tile[[16, 128], pl.FP32, pl.MemRef(mem_vec_7, pl.const(16448, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.muls(kv_blk_f32_inline2211__tile, 0.088388347648318447)
        t__tile_3: pl.Tile[[16, 128], pl.BF16, pl.MemRef(mem_vec_13, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(t__tile_2, target_type=pl.BF16, mode="rint")
        kv_blk_f32_v1_inline2213__tile: pl.Tile[[16, 128], pl.FP32, pl.MemRef(mem_vec_7, pl.const(16448, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.cast(t__tile_3, target_type=pl.FP32, mode="round")
        t__tile_4: pl.Tile[[16, 128], pl.FP32, pl.MemRef(mem_vec_13, pl.const(0, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.abs(kv_blk_f32_v1_inline2213__tile)
        tmp_tile: pl.Tile[[16, 128], pl.FP32, pl.MemRef(mem_vec_14, pl.const(8192, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.create([16, 128], dtype=pl.FP32, target_memory=pl.Mem.Vec)
        t__tile_5: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_15, pl.const(16384, pl.INT64), 64), pl.Mem.Vec] = pl.tile.row_max(t__tile_4, tmp_tile)
        kv_amax_inline2216__tile: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_15, pl.const(16384, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(t__tile_5, [1, 16])
        t__tile_6: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_13, pl.const(0, pl.INT64), 64), pl.Mem.Vec] = pl.tile.full([1, 16], dtype=pl.FP32, value=0.0001)
        kv_amax_v1_inline2200__tile: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_13, pl.const(0, pl.INT64), 64), pl.Mem.Vec] = pl.tile.maximum(kv_amax_inline2216__tile, t__tile_6)
        t__tile_7: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_14, pl.const(8192, pl.INT64), 64), pl.Mem.Vec] = pl.tile.full([1, 16], dtype=pl.FP32, value=127.0)
        kv_scale_q_row_inline2212__tile: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_14, pl.const(8192, pl.INT64), 64), pl.Mem.Vec] = pl.tile.div(t__tile_7, kv_amax_v1_inline2200__tile)
        t__tile_8: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_13, pl.const(0, pl.INT64), 64), pl.Mem.Vec] = pl.tile.recip(kv_scale_q_row_inline2212__tile)
        kv_scale_dq_col_inline2204__tile: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_13, pl.const(0, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(t__tile_8, [16, 1])
        kv_scale_q_col_inline2218__tile: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_14, pl.const(8192, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(kv_scale_q_row_inline2212__tile, [16, 1])
        idx_kv_scale_values_inline2188__tile: pl.Tensor[[384, 1], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 1536)] = pl.tile.store(
            kv_scale_dq_col_inline2204__tile, [wr_b0_inline2194__ssa_v0, 0], idx_kv_scale_values_inline2188__ssa_v0
        )
        kv_scaled_inline2209__tile: pl.Tile[[16, 128], pl.FP32, pl.MemRef(mem_vec_13, pl.const(0, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.row_expand_mul(
            kv_blk_f32_v1_inline2213__tile, kv_scale_q_col_inline2218__tile
        )
        kv_i32_inline2219__tile: pl.Tile[[16, 128], pl.INT32, pl.MemRef(mem_vec_13, pl.const(0, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.cast(
            kv_scaled_inline2209__tile, target_type=pl.INT32, mode="rint"
        )
        kv_half_inline2210__tile: pl.Tile[[16, 128], pl.FP16, pl.MemRef(mem_vec_13, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(kv_i32_inline2219__tile, target_type=pl.FP16, mode="round")
        kv_i8_blk_inline2186__tile: pl.Tile[[16, 128], pl.INT8, pl.MemRef(mem_vec_13, pl.const(0, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(
            kv_half_inline2210__tile, target_type=pl.INT8, mode="trunc"
        )
        for inner_inline2179__idx_v0, (idx_kv_cache__iter_v1, kv_flat_inline2202__iter_v1) in pl.range(wr_blk_rows_inline2198__ssa_v0, init_values=(idx_kv_cache__ssa_v0, kv_flat_inline2202__ssa_v0)):
            compact_token_inline2187__ssa_v0: pl.Scalar[pl.INDEX] = wr_b0_inline2194__ssa_v0 + inner_inline2179__idx_v0
            request_inline2178__ssa_v0: pl.Scalar[pl.INDEX] = compact_token_inline2187__ssa_v0 // 2
            first_pos_inline2177__tile: pl.Scalar[pl.INT64] = pl.tensor.read(position_ids__ssa_v0, [request_inline2178__ssa_v0 * 6])
            local_token_inline2176__ssa_v0: pl.Scalar[pl.INDEX] = compact_token_inline2187__ssa_v0 % 2 * 4 - first_pos_inline2177__tile % 4 + 3
            if local_token_inline2176__ssa_v0 < 6:
                token_inline2175__ssa_v0: pl.Scalar[pl.INDEX] = request_inline2178__ssa_v0 * 6 + local_token_inline2176__ssa_v0
                token_pos_inline2174__tile: pl.Scalar[pl.INT64] = pl.tensor.read(position_ids__ssa_v0, [token_inline2175__ssa_v0])
                t__tile_9: pl.Scalar[pl.INT32] = pl.tensor.read(idx_row_offsets__ssa_v0, [request_inline2178__ssa_v0])
                metadata_row_inline2173__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(t__tile_9, pl.INDEX) + pl.cast((token_pos_inline2174__tile + 1) // 4, pl.INDEX)
                native_page_inline2172__tile: pl.Scalar[pl.INT32] = pl.tensor.read(idx_slot_mapping__ssa_v0, [metadata_row_inline2173__ssa_v0, 0])
                native_offset_inline2171__tile: pl.Scalar[pl.INT32] = pl.tensor.read(idx_slot_mapping__ssa_v0, [metadata_row_inline2173__ssa_v0, 1])
                if 0 <= pl.cast(native_page_inline2172__tile, pl.INDEX) and 0 <= pl.cast(native_offset_inline2171__tile, pl.INDEX):
                    cache_row_inline2180__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(native_page_inline2172__tile, pl.INDEX) * 32 + pl.cast(native_offset_inline2171__tile, pl.INDEX)
                    t__tile_10: pl.Tile[[1, 128], pl.FP32, pl.MemRef(mem_vec_7, pl.const(16448, pl.INT64), 512), pl.Mem.Vec] = pl.tile.slice(
                        kv_blk_f32_v1_inline2213__tile, [1, 128], [inner_inline2179__idx_v0, 0]
                    )
                    kv_flat_inline2202__tile: pl.Tensor[[T_DYN, 128], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                        t__tile_10, [token_inline2175__ssa_v0, 0], kv_flat_inline2202__iter_v1
                    )
                    cache_page_inline2193__ssa_v0: pl.Scalar[pl.INDEX] = cache_row_inline2180__ssa_v0 // 32
                    key_begin_inline2185__ssa_v0: pl.Scalar[pl.INDEX] = cache_row_inline2180__ssa_v0 % 32 * 128
                    t__tile_11: pl.Tile[[1, 128], pl.INT8, pl.MemRef(mem_vec_13, pl.const(0, pl.INT64), 128), pl.Mem.Vec] = pl.tile.slice(
                        kv_i8_blk_inline2186__tile, [1, 128], [inner_inline2179__idx_v0, 0]
                    )
                    idx_kv_cache__tile: pl.Tensor[[IDX_CACHE_BLOCK_NUM_DYN, INDEXER_PAGE_BYTES_DYN], pl.INT8, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                        t__tile_11, [cache_page_inline2193__ssa_v0, key_begin_inline2185__ssa_v0], idx_kv_cache__iter_v1
                    )
                    idx_kv_cache__phi_v4, kv_flat_inline2202__phi_v4 = pl.yield_(idx_kv_cache__tile, kv_flat_inline2202__tile)
                else:
                    idx_kv_cache__phi_v4, kv_flat_inline2202__phi_v4 = pl.yield_(idx_kv_cache__iter_v1, kv_flat_inline2202__iter_v1)
                idx_kv_cache__phi_v5, kv_flat_inline2202__phi_v5 = pl.yield_(idx_kv_cache__phi_v4, kv_flat_inline2202__phi_v4)
            else:
                idx_kv_cache__phi_v5, kv_flat_inline2202__phi_v5 = pl.yield_(idx_kv_cache__iter_v1, kv_flat_inline2202__iter_v1)
            idx_kv_cache__rv_v2, kv_flat_inline2202__rv_v2 = pl.yield_(idx_kv_cache__phi_v5, kv_flat_inline2202__phi_v5)
        return idx_kv_scale_values_inline2188__ssa_v0, idx_kv_cache__ssa_v0

    @pl.function(type=pl.FunctionType.Spmd)
    def kv_and_cache_write_spmd(
        self,
        compact_rows_inline2207__ssa_v0: pl.Scalar[pl.INDEX],
        kv_final_inline2183__rv_v2: pl.Tensor[[384, 128], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 196608)],
        idx_kv_scale_values_inline2188__ssa_v0: pl.Out[pl.Tensor[[384, 1], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 1536)]],
        idx_kv_cache__ssa_v0: pl.Out[pl.Tensor[[IDX_CACHE_BLOCK_NUM_DYN, INDEXER_PAGE_BYTES_DYN], pl.INT8, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)]],
        kv_flat_inline2202__ssa_v0: pl.Out[pl.Tensor[[T_DYN, 128], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)]],
        position_ids__ssa_v0: pl.Tensor[[T_DYN], pl.INT64, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 0)],
        idx_row_offsets__ssa_v0: pl.Tensor[[B_DYN], pl.INT32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 0)],
        idx_slot_mapping__ssa_v0: pl.Tensor[[INDEXER_ROWS_DYN, 2], pl.INT32, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 0)],
    ) -> tuple[pl.Tensor[[384, 1], pl.FP32], pl.Tensor[[IDX_CACHE_BLOCK_NUM_DYN, INDEXER_PAGE_BYTES_DYN], pl.INT8]]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        ret__tmp_v0: pl.Tuple[pl.Tensor[[384, 1], pl.FP32], pl.Tensor[[IDX_CACHE_BLOCK_NUM_DYN, INDEXER_PAGE_BYTES_DYN], pl.INT8]] = self.kv_and_cache_write(
            compact_rows_inline2207__ssa_v0,
            kv_final_inline2183__rv_v2,
            idx_kv_scale_values_inline2188__ssa_v0,
            idx_kv_cache__ssa_v0,
            kv_flat_inline2202__ssa_v0,
            position_ids__ssa_v0,
            idx_row_offsets__ssa_v0,
            idx_slot_mapping__ssa_v0,
            attrs={"arg_directions": [pl.adir.scalar, pl.adir.input, pl.adir.output_existing, pl.adir.output_existing, pl.adir.output_existing, pl.adir.input, pl.adir.input, pl.adir.input]},
        )
        idx_kv_scale_values_inline2188__ssa_v1: pl.Tensor[[384, 1], pl.FP32, pl.MemRef("mem_ddr_7", pl.const(0, pl.INT64), 1536)] = ret__tmp_v0[0]
        idx_kv_cache__rv_v2: pl.Tensor[[IDX_CACHE_BLOCK_NUM_DYN, INDEXER_PAGE_BYTES_DYN], pl.INT8, pl.MemRef("mem_ddr_8", pl.const(0, pl.INT64), 0)] = ret__tmp_v0[1]
        return idx_kv_scale_values_inline2188__ssa_v0, idx_kv_cache__ssa_v0

    @pl.function(type=pl.FunctionType.AIC, level=pl.Level.AIC, role=pl.Role.SubWorker)
    def kv_hadamard(
        kv_final_inline2183__ssa_v0: pl.Out[pl.Tensor[[384, 128], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 196608)]],
        hadamard_idx__ssa_v0: pl.Tensor[[128, 128], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 32768)],
        rms_blocks_inline2196__ssa_v0: pl.Scalar[pl.INDEX],
        compact_rows_inline2207__ssa_v0: pl.Scalar[pl.INDEX],
        normed_kv_inline212__rv_v3: pl.Tensor[[384, 128], pl.BF16, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 98304)],
    ) -> pl.Tensor[[384, 128], pl.FP32]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        mem_mat_3: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 16384)
        mem_mat_4: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 4096)
        mem_left_5: pl.Ptr = pl.tile.alloc(pl.Mem.Left, 4096)
        mem_right_6: pl.Ptr = pl.tile.alloc(pl.Mem.Right, 16384)
        mem_acc_7: pl.Ptr = pl.tile.alloc(pl.Mem.Acc, 4096)
        for o0_inline2181__idx_v0, (kv_final_inline2183__iter_v1,) in pl.range(0, 128, 64, init_values=(kv_final_inline2183__ssa_v0,)):
            hadamard_tile_inline2184__tile: pl.Tile[[128, 64], pl.BF16, pl.MemRef(mem_mat_3, pl.const(0, pl.INT64), 16384), pl.Mem.Mat] = pl.tile.load(
                hadamard_idx__ssa_v0, [0, o0_inline2181__idx_v0], [128, 64], [128, 64], target_memory=pl.Mem.Mat
            )
            for had_blk_inline2203__idx_v0, (kv_final_inline2183__iter_v3,) in pl.range(rms_blocks_inline2196__ssa_v0, init_values=(kv_final_inline2183__iter_v1,)):
                had_b0_inline2189__ssa_v0: pl.Scalar[pl.INDEX] = had_blk_inline2203__idx_v0 * 16
                had_rows_inline2190__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(compact_rows_inline2207__ssa_v0 - had_b0_inline2189__ssa_v0, 16)
                kv_proj_tile_inline2182__tile: pl.Tile[
                    [16, 128], pl.BF16, pl.MemRef(mem_mat_4, pl.const(16384, pl.INT64), 4096), pl.Mem.Mat, pl.TileView(valid_shape=[had_rows_inline2190__ssa_v0, 128])
                ] = pl.tile.load(normed_kv_inline212__rv_v3, [had_b0_inline2189__ssa_v0, 0], [16, 128], [had_rows_inline2190__ssa_v0, 128], target_memory=pl.Mem.Mat)
                kv_proj_tile_inline2182__tile_Left: pl.Tile[
                    [16, 128], pl.BF16, pl.MemRef(mem_left_5, pl.const(0, pl.INT64), 4096), pl.Mem.Left, pl.TileView(valid_shape=[had_rows_inline2190__ssa_v0, 128])
                ] = pl.tile.move(kv_proj_tile_inline2182__tile, target_memory=pl.Mem.Left)
                hadamard_tile_inline2184__tile_Right: pl.Tile[[128, 64], pl.BF16, pl.MemRef(mem_right_6, pl.const(0, pl.INT64), 16384), pl.Mem.Right] = pl.tile.move(
                    hadamard_tile_inline2184__tile, target_memory=pl.Mem.Right
                )
                kv_hadamard_acc_inline2205__tile: pl.Tile[
                    [16, 64], pl.FP32, pl.MemRef(mem_acc_7, pl.const(0, pl.INT64), 4096), pl.Mem.Acc, pl.TileView(valid_shape=[had_rows_inline2190__ssa_v0, 64], compact=pl.CompactMode.normal)
                ] = pl.tile.matmul(kv_proj_tile_inline2182__tile_Left, hadamard_tile_inline2184__tile_Right)
                kv_final_inline2183__tile: pl.Tensor[[384, 128], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 196608)] = pl.tile.store(
                    kv_hadamard_acc_inline2205__tile, [had_b0_inline2189__ssa_v0, o0_inline2181__idx_v0], kv_final_inline2183__iter_v3
                )
                kv_final_inline2183__rv_v4: pl.Tensor[[384, 128], pl.FP32, pl.MemRef("mem_ddr_8", pl.const(0, pl.INT64), 196608)] = pl.yield_(kv_final_inline2183__tile)
            kv_final_inline2183__rv_v2: pl.Tensor[[384, 128], pl.FP32, pl.MemRef("mem_ddr_9", pl.const(0, pl.INT64), 196608)] = pl.yield_(kv_final_inline2183__rv_v4)
        return kv_final_inline2183__ssa_v0

    @pl.function(type=pl.FunctionType.AIC, level=pl.Level.AIC, role=pl.Role.SubWorker)
    def kv_proj_matmul(
        kv_m_groups_inline1894__ssa_v0: pl.Scalar[pl.INDEX],
        kv_fp32_inline1850__rv_v2: pl.InOut[pl.Tensor[[t_matmul_inline1893__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)]],
        kv_full_rows_inline1888__ssa_v0: pl.Scalar[pl.INDEX],
        tile_base_inline1887__idx_v0: pl.Scalar[pl.INDEX],
        x_view_inline1855__ssa_v0: pl.Tensor[[t_dim_inline1873__ssa_v0, 4096], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)],
        wkv__ssa_v0: pl.Tensor[[4096, 512], pl.BF16, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 4194304)],
        t_matmul_inline1893__ssa_v0: pl.Scalar[pl.INDEX],
        tile_rows_inline1877__ssa_v0: pl.Scalar[pl.INDEX],
    ) -> pl.Tensor[[t_matmul_inline1893__ssa_v0, 512], pl.FP32]:
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
        kbg_inline1917__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        kv_col0_inline1871__ssa_v0: pl.Scalar[pl.INDEX] = kbg_inline1917__ssa_v0 // (kv_m_groups_inline1894__ssa_v0 * 2) * 128
        kv_k_base_inline1901__ssa_v0: pl.Scalar[pl.INDEX] = kbg_inline1917__ssa_v0 // kv_m_groups_inline1894__ssa_v0 % 2 * 2048
        kv_m_group_inline1911__ssa_v0: pl.Scalar[pl.INDEX] = kbg_inline1917__ssa_v0 % kv_m_groups_inline1894__ssa_v0
        for dense_t0_inline1867__idx_v0, (kv_fp32_inline1850__iter_v12,) in pl.range(
            kv_m_group_inline1911__ssa_v0 * 64, kv_full_rows_inline1888__ssa_v0, kv_m_groups_inline1894__ssa_v0 * 64, init_values=(kv_fp32_inline1850__rv_v2,)
        ):
            dense_x0_inline1902__ssa_v0: pl.Scalar[pl.INDEX] = tile_base_inline1887__idx_v0 + dense_t0_inline1867__idx_v0
            dense_acc_inline1853__tile: pl.Tile[[64, 128], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 32768), pl.Mem.Acc] = pl.tile.create([64, 128], dtype=pl.FP32, target_memory=pl.Mem.Acc)
            for dense_k_inline1875__idx_v0, (dense_acc_inline1853__iter_v1,) in pl.range(0, 8, 2, init_values=(dense_acc_inline1853__tile,)):
                dense_d0_inline1906__ssa_v0: pl.Scalar[pl.INDEX] = kv_k_base_inline1901__ssa_v0 + dense_k_inline1875__idx_v0 * 256
                dense_d0_inline1906__ssa_v0_1: pl.Scalar[pl.INDEX] = kv_k_base_inline1901__ssa_v0 + (dense_k_inline1875__idx_v0 * 256 + 256)
                dense_x_inline1907__tile: pl.Tile[[64, 256], pl.BF16, pl.MemRef(mem_mat_4, pl.const(0, pl.INT64), 32768), pl.Mem.Mat] = pl.tile.load(
                    x_view_inline1855__ssa_v0, [dense_x0_inline1902__ssa_v0, dense_d0_inline1906__ssa_v0], [64, 256], [64, 256], target_memory=pl.Mem.Mat
                )
                dense_w_inline1914__tile: pl.Tile[[256, 128], pl.BF16, pl.MemRef(mem_mat_5, pl.const(32768, pl.INT64), 65536), pl.Mem.Mat] = pl.tile.load(
                    wkv__ssa_v0, [dense_d0_inline1906__ssa_v0, kv_col0_inline1871__ssa_v0], [256, 128], [256, 128], target_memory=pl.Mem.Mat
                )
                dense_x_inline1907__tile_1: pl.Tile[[64, 256], pl.BF16, pl.MemRef(mem_mat_6, pl.const(98304, pl.INT64), 32768), pl.Mem.Mat] = pl.tile.load(
                    x_view_inline1855__ssa_v0, [dense_x0_inline1902__ssa_v0, dense_d0_inline1906__ssa_v0_1], [64, 256], [64, 256], target_memory=pl.Mem.Mat
                )
                dense_w_inline1914__tile_1: pl.Tile[[256, 128], pl.BF16, pl.MemRef(mem_mat_7, pl.const(131072, pl.INT64), 65536), pl.Mem.Mat] = pl.tile.load(
                    wkv__ssa_v0, [dense_d0_inline1906__ssa_v0_1, kv_col0_inline1871__ssa_v0], [256, 128], [256, 128], target_memory=pl.Mem.Mat
                )
                dense_acc_inline1853__tile_l0_a: pl.Tile[[64, 128], pl.BF16, pl.MemRef(mem_left_8, pl.const(49152, pl.INT64), 16384), pl.Mem.Left, pl.TileView(blayout=pl.TileLayout.row_major)] = (
                    pl.tile.extract(dense_x_inline1907__tile, 0, 0, [64, 128], target_memory=pl.Mem.Left)
                )
                dense_acc_inline1853__tile_l0_b: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_9, pl.const(32768, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                    dense_w_inline1914__tile, 0, 0, [128, 128], target_memory=pl.Mem.Right
                )
                dense_acc_inline1853__tile_l0_a_1: pl.Tile[[64, 128], pl.BF16, pl.MemRef(mem_left_10, pl.const(0, pl.INT64), 16384), pl.Mem.Left, pl.TileView(blayout=pl.TileLayout.row_major)] = (
                    pl.tile.extract(dense_x_inline1907__tile, 0, 128, [64, 128], target_memory=pl.Mem.Left)
                )
                dense_acc_inline1853__tile_l0_b_1: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_11, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                    dense_w_inline1914__tile, 128, 0, [128, 128], target_memory=pl.Mem.Right
                )
                dense_acc_inline1853__tile_l0_c_acc: pl.Tile[[64, 128], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 32768), pl.Mem.Acc] = pl.tile.matmul_acc(
                    dense_acc_inline1853__iter_v1, dense_acc_inline1853__tile_l0_a, dense_acc_inline1853__tile_l0_b, dense_k_inline1875__idx_v0 == 0
                )
                dense_acc_inline1853__tile_l0_c_acc_1: pl.Tile[[64, 128], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 32768), pl.Mem.Acc] = pl.tile.matmul_acc(
                    dense_acc_inline1853__tile_l0_c_acc, dense_acc_inline1853__tile_l0_a_1, dense_acc_inline1853__tile_l0_b_1, False
                )
                dense_acc_inline1853__tile_l0_a_2: pl.Tile[[64, 128], pl.BF16, pl.MemRef(mem_left_13, pl.const(16384, pl.INT64), 16384), pl.Mem.Left, pl.TileView(blayout=pl.TileLayout.row_major)] = (
                    pl.tile.extract(dense_x_inline1907__tile_1, 0, 0, [64, 128], target_memory=pl.Mem.Left)
                )
                dense_acc_inline1853__tile_l0_b_2: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_9, pl.const(32768, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                    dense_w_inline1914__tile_1, 0, 0, [128, 128], target_memory=pl.Mem.Right
                )
                dense_acc_inline1853__tile_l0_a_3: pl.Tile[[64, 128], pl.BF16, pl.MemRef(mem_left_15, pl.const(32768, pl.INT64), 16384), pl.Mem.Left, pl.TileView(blayout=pl.TileLayout.row_major)] = (
                    pl.tile.extract(dense_x_inline1907__tile_1, 0, 128, [64, 128], target_memory=pl.Mem.Left)
                )
                dense_acc_inline1853__tile_l0_b_3: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_11, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                    dense_w_inline1914__tile_1, 128, 0, [128, 128], target_memory=pl.Mem.Right
                )
                dense_acc_inline1853__tile_l0_c_acc_2: pl.Tile[[64, 128], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 32768), pl.Mem.Acc] = pl.tile.matmul_acc(
                    dense_acc_inline1853__tile_l0_c_acc_1, dense_acc_inline1853__tile_l0_a_2, dense_acc_inline1853__tile_l0_b_2, dense_k_inline1875__idx_v0 == -1
                )
                dense_acc_inline1853__tile_l0_c_acc_3: pl.Tile[[64, 128], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 32768), pl.Mem.Acc] = pl.tile.matmul_acc(
                    dense_acc_inline1853__tile_l0_c_acc_2, dense_acc_inline1853__tile_l0_a_3, dense_acc_inline1853__tile_l0_b_3, False
                )
                dense_acc_inline1853__rv_v2: pl.Tile[[64, 128], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 32768), pl.Mem.Acc] = pl.yield_(dense_acc_inline1853__tile_l0_c_acc_3)
            kv_fp32_inline1850__tile: pl.Tensor[[t_matmul_inline1893__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                dense_acc_inline1853__rv_v2, [dense_t0_inline1867__idx_v0, kv_col0_inline1871__ssa_v0], kv_fp32_inline1850__iter_v12, atomic=pl.AtomicType.Add
            )
            kv_fp32_inline1850__rv_v13: pl.Tensor[[t_matmul_inline1893__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_19", pl.const(0, pl.INT64), 0)] = pl.yield_(kv_fp32_inline1850__tile)
        for t0_inline1862__idx_v0, (kv_fp32_inline1850__iter_v15,) in pl.range(
            kv_full_rows_inline1888__ssa_v0 + kv_m_group_inline1911__ssa_v0 * 16, t_matmul_inline1893__ssa_v0, kv_m_groups_inline1894__ssa_v0 * 16, init_values=(kv_fp32_inline1850__rv_v13,)
        ):
            kv_acc_inline1908__tile: pl.Tile[[16, 128], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 8192), pl.Mem.Acc] = pl.tile.create([16, 128], dtype=pl.FP32, target_memory=pl.Mem.Acc)
            for db_inline1851__idx_v0, (kv_acc_inline1908__iter_v1,) in pl.range(0, 8, 2, init_values=(kv_acc_inline1908__tile,)):
                d0_inline1905__ssa_v0: pl.Scalar[pl.INDEX] = kv_k_base_inline1901__ssa_v0 + db_inline1851__idx_v0 * 256
                kv_rows_inline1856__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(tile_rows_inline1877__ssa_v0 - t0_inline1862__idx_v0, 16)
                x_t0_inline1910__ssa_v0: pl.Scalar[pl.INDEX] = tile_base_inline1887__idx_v0 + t0_inline1862__idx_v0
                d0_inline1905__ssa_v0_1: pl.Scalar[pl.INDEX] = kv_k_base_inline1901__ssa_v0 + (db_inline1851__idx_v0 * 256 + 256)
                kv_rows_inline1856__ssa_v0_1: pl.Scalar[pl.INDEX] = pl.min(tile_rows_inline1877__ssa_v0 - t0_inline1862__idx_v0, 16)
                x_t0_inline1910__ssa_v0_1: pl.Scalar[pl.INDEX] = tile_base_inline1887__idx_v0 + t0_inline1862__idx_v0
                kv_x_chunk_bf16_inline1912__tile: pl.Tile[
                    [16, 256], pl.BF16, pl.MemRef(mem_mat_4, pl.const(0, pl.INT64), 8192), pl.Mem.Mat, pl.TileView(valid_shape=[kv_rows_inline1856__ssa_v0, 256])
                ] = pl.tile.load(x_view_inline1855__ssa_v0, [x_t0_inline1910__ssa_v0, d0_inline1905__ssa_v0], [16, 256], [kv_rows_inline1856__ssa_v0, 256], target_memory=pl.Mem.Mat)
                wkv_chunk_inline1913__tile: pl.Tile[[256, 128], pl.BF16, pl.MemRef(mem_mat_5, pl.const(32768, pl.INT64), 65536), pl.Mem.Mat] = pl.tile.load(
                    wkv__ssa_v0, [d0_inline1905__ssa_v0, kv_col0_inline1871__ssa_v0], [256, 128], [256, 128], target_memory=pl.Mem.Mat
                )
                kv_x_chunk_bf16_inline1912__tile_1: pl.Tile[
                    [16, 256], pl.BF16, pl.MemRef(mem_mat_6, pl.const(98304, pl.INT64), 8192), pl.Mem.Mat, pl.TileView(valid_shape=[kv_rows_inline1856__ssa_v0_1, 256])
                ] = pl.tile.load(x_view_inline1855__ssa_v0, [x_t0_inline1910__ssa_v0_1, d0_inline1905__ssa_v0_1], [16, 256], [kv_rows_inline1856__ssa_v0_1, 256], target_memory=pl.Mem.Mat)
                wkv_chunk_inline1913__tile_1: pl.Tile[[256, 128], pl.BF16, pl.MemRef(mem_mat_7, pl.const(131072, pl.INT64), 65536), pl.Mem.Mat] = pl.tile.load(
                    wkv__ssa_v0, [d0_inline1905__ssa_v0_1, kv_col0_inline1871__ssa_v0], [256, 128], [256, 128], target_memory=pl.Mem.Mat
                )
                kv_acc_inline1908__tile_l0_a: pl.Tile[
                    [16, 128],
                    pl.BF16,
                    pl.MemRef(mem_left_8, pl.const(49152, pl.INT64), 4096),
                    pl.Mem.Left,
                    pl.TileView(valid_shape=[kv_rows_inline1856__ssa_v0, 128], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal),
                ] = pl.tile.extract(kv_x_chunk_bf16_inline1912__tile, 0, 0, [16, 128], target_memory=pl.Mem.Left)
                kv_acc_inline1908__tile_l0_b: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_9, pl.const(32768, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                    wkv_chunk_inline1913__tile, 0, 0, [128, 128], target_memory=pl.Mem.Right
                )
                kv_acc_inline1908__tile_l0_a_1: pl.Tile[
                    [16, 128],
                    pl.BF16,
                    pl.MemRef(mem_left_10, pl.const(0, pl.INT64), 4096),
                    pl.Mem.Left,
                    pl.TileView(valid_shape=[kv_rows_inline1856__ssa_v0, 128], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal),
                ] = pl.tile.extract(kv_x_chunk_bf16_inline1912__tile, 0, 128, [16, 128], target_memory=pl.Mem.Left)
                kv_acc_inline1908__tile_l0_b_1: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_11, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                    wkv_chunk_inline1913__tile, 128, 0, [128, 128], target_memory=pl.Mem.Right
                )
                kv_acc_inline1908__tile_l0_c_acc: pl.Tile[[16, 128], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 8192), pl.Mem.Acc] = pl.tile.matmul_acc(
                    kv_acc_inline1908__iter_v1, kv_acc_inline1908__tile_l0_a, kv_acc_inline1908__tile_l0_b, db_inline1851__idx_v0 == 0
                )
                kv_acc_inline1908__tile_l0_c_acc_1: pl.Tile[[16, 128], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 8192), pl.Mem.Acc] = pl.tile.matmul_acc(
                    kv_acc_inline1908__tile_l0_c_acc, kv_acc_inline1908__tile_l0_a_1, kv_acc_inline1908__tile_l0_b_1, False
                )
                kv_acc_inline1908__tile_l0_a_2: pl.Tile[
                    [16, 128],
                    pl.BF16,
                    pl.MemRef(mem_left_13, pl.const(16384, pl.INT64), 4096),
                    pl.Mem.Left,
                    pl.TileView(valid_shape=[kv_rows_inline1856__ssa_v0_1, 128], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal),
                ] = pl.tile.extract(kv_x_chunk_bf16_inline1912__tile_1, 0, 0, [16, 128], target_memory=pl.Mem.Left)
                kv_acc_inline1908__tile_l0_b_2: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_9, pl.const(32768, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                    wkv_chunk_inline1913__tile_1, 0, 0, [128, 128], target_memory=pl.Mem.Right
                )
                kv_acc_inline1908__tile_l0_a_3: pl.Tile[
                    [16, 128],
                    pl.BF16,
                    pl.MemRef(mem_left_15, pl.const(32768, pl.INT64), 4096),
                    pl.Mem.Left,
                    pl.TileView(valid_shape=[kv_rows_inline1856__ssa_v0_1, 128], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal),
                ] = pl.tile.extract(kv_x_chunk_bf16_inline1912__tile_1, 0, 128, [16, 128], target_memory=pl.Mem.Left)
                kv_acc_inline1908__tile_l0_b_3: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_11, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                    wkv_chunk_inline1913__tile_1, 128, 0, [128, 128], target_memory=pl.Mem.Right
                )
                kv_acc_inline1908__tile_l0_c_acc_2: pl.Tile[[16, 128], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 8192), pl.Mem.Acc] = pl.tile.matmul_acc(
                    kv_acc_inline1908__tile_l0_c_acc_1, kv_acc_inline1908__tile_l0_a_2, kv_acc_inline1908__tile_l0_b_2, db_inline1851__idx_v0 == -1
                )
                kv_acc_inline1908__tile_l0_c_acc_3: pl.Tile[[16, 128], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 8192), pl.Mem.Acc] = pl.tile.matmul_acc(
                    kv_acc_inline1908__tile_l0_c_acc_2, kv_acc_inline1908__tile_l0_a_3, kv_acc_inline1908__tile_l0_b_3, False
                )
                kv_acc_inline1908__rv_v2: pl.Tile[[16, 128], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 8192), pl.Mem.Acc] = pl.yield_(kv_acc_inline1908__tile_l0_c_acc_3)
            kv_fp32_inline1850__tile_1: pl.Tensor[[t_matmul_inline1893__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_19", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                kv_acc_inline1908__rv_v2, [t0_inline1862__idx_v0, kv_col0_inline1871__ssa_v0], kv_fp32_inline1850__iter_v15, atomic=pl.AtomicType.Add
            )
            kv_fp32_inline1850__rv_v16: pl.Tensor[[t_matmul_inline1893__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_36", pl.const(0, pl.INT64), 0)] = pl.yield_(kv_fp32_inline1850__tile_1)
        return kv_fp32_inline1850__rv_v2

    @pl.function(type=pl.FunctionType.Spmd)
    def kv_proj_matmul_spmd(
        self,
        kv_m_groups_inline1894__ssa_v0: pl.Scalar[pl.INDEX],
        kv_fp32_inline1850__rv_v2: pl.InOut[pl.Tensor[[t_matmul_inline1893__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)]],
        kv_full_rows_inline1888__ssa_v0: pl.Scalar[pl.INDEX],
        tile_base_inline1887__idx_v0: pl.Scalar[pl.INDEX],
        x_view_inline1855__ssa_v0: pl.Tensor[[t_dim_inline1873__ssa_v0, 4096], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)],
        wkv__ssa_v0: pl.Tensor[[4096, 512], pl.BF16, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 4194304)],
        t_matmul_inline1893__ssa_v0: pl.Scalar[pl.INDEX],
        tile_rows_inline1877__ssa_v0: pl.Scalar[pl.INDEX],
    ) -> pl.Tensor[[t_matmul_inline1893__ssa_v0, 512], pl.FP32]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        kv_fp32_inline1850__rv_v16: pl.Tensor[[t_matmul_inline1893__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)] = self.kv_proj_matmul(
            kv_m_groups_inline1894__ssa_v0,
            kv_fp32_inline1850__rv_v2,
            kv_full_rows_inline1888__ssa_v0,
            tile_base_inline1887__idx_v0,
            x_view_inline1855__ssa_v0,
            wkv__ssa_v0,
            t_matmul_inline1893__ssa_v0,
            tile_rows_inline1877__ssa_v0,
            attrs={"arg_directions": [pl.adir.scalar, pl.adir.inout, pl.adir.scalar, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.scalar, pl.adir.scalar]},
        )
        return kv_fp32_inline1850__rv_v2

    @pl.function(type=pl.FunctionType.AIC, level=pl.Level.AIC, role=pl.Role.SubWorker)
    def kv_proj_native_240(
        kv_fp32_inline1850__rv_v2: pl.Out[pl.Tensor[[t_matmul_inline1893__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)]],
        x_normed_t__ssa_v0: pl.Tensor[[T_DYN, 4096], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)],
        tile_base_inline1887__idx_v0: pl.Scalar[pl.INDEX],
        wkv__ssa_v0: pl.Tensor[[4096, 512], pl.BF16, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 4194304)],
    ) -> pl.Tensor[[t_matmul_inline1893__ssa_v0, 512], pl.FP32]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        mem_acc_3: pl.Ptr = pl.tile.alloc(pl.Mem.Acc, 8192)
        mem_mat_4: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 8192)
        mem_mat_5: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 4096)
        mem_mat_6: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 8192)
        mem_mat_7: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 4096)
        mem_left_8: pl.Ptr = pl.tile.alloc(pl.Mem.Left, 8192)
        mem_right_9: pl.Ptr = pl.tile.alloc(pl.Mem.Right, 4096)
        mem_left_10: pl.Ptr = pl.tile.alloc(pl.Mem.Left, 8192)
        mem_right_11: pl.Ptr = pl.tile.alloc(pl.Mem.Right, 4096)
        t__tmp_v73: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        column_inline1256_inline1890__ssa_v0: pl.Scalar[pl.INDEX] = t__tmp_v73 * 32
        group_inline1257_inline1847__ssa_v0: pl.Scalar[pl.INDEX] = column_inline1256_inline1890__ssa_v0 // 96
        for row_inline1259_inline1868__idx_v0, (kv_fp32_inline1850__iter_v6,) in pl.range(0, 192, 64, init_values=(kv_fp32_inline1850__rv_v2,)):
            accumulator_inline1260_inline1861__tile: pl.Tile[[64, 32], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 8192), pl.Mem.Acc] = pl.tile.create(
                [64, 32], dtype=pl.FP32, target_memory=pl.Mem.Acc
            )
            for step_inline1262_inline1860__idx_v0, (accumulator_inline1260_inline1861__iter_v1,) in pl.range(0, 64, 2, init_values=(accumulator_inline1260_inline1861__tile,)):
                block_inline1258_inline1858__ssa_v0: pl.Scalar[pl.INDEX] = step_inline1262_inline1860__idx_v0 // 4
                block_inline1258_inline1858__ssa_v0_1: pl.Scalar[pl.INDEX] = (step_inline1262_inline1860__idx_v0 + 1) // 4
                if group_inline1257_inline1847__ssa_v0 < 4:
                    direction_inline1264_inline1909__ssa_v0: pl.Scalar[pl.INDEX] = block_inline1258_inline1858__ssa_v0
                    if group_inline1257_inline1847__ssa_v0 % 2 == 1:
                        direction_inline1264_inline1909__ssa_v1: pl.Scalar[pl.INDEX] = 15 - block_inline1258_inline1858__ssa_v0
                        direction_inline1264_inline1909__phi_v2: pl.Scalar[pl.INDEX] = pl.yield_(direction_inline1264_inline1909__ssa_v1)
                    else:
                        direction_inline1264_inline1909__phi_v2: pl.Scalar[pl.INDEX] = pl.yield_(direction_inline1264_inline1909__ssa_v0)
                    block_inline1258_inline1858__ssa_v1: pl.Scalar[pl.INDEX] = (group_inline1257_inline1847__ssa_v0 // 2 * 5 + direction_inline1264_inline1909__phi_v2) % 16
                    block_inline1258_inline1858__phi_v2: pl.Scalar[pl.INDEX] = pl.yield_(block_inline1258_inline1858__ssa_v1)
                else:
                    block_inline1258_inline1858__phi_v2: pl.Scalar[pl.INDEX] = pl.yield_(block_inline1258_inline1858__ssa_v0)
                offset_inline1270_inline1895__ssa_v0: pl.Scalar[pl.INDEX] = block_inline1258_inline1858__phi_v2 * 256 + step_inline1262_inline1860__idx_v0 % 4 * 64
                a_inline1263_inline1880__tile: pl.Tile[[64, 64], pl.BF16, pl.MemRef(mem_mat_4, pl.const(0, pl.INT64), 8192), pl.Mem.Mat] = pl.tile.load(
                    x_normed_t__ssa_v0, [tile_base_inline1887__idx_v0 + row_inline1259_inline1868__idx_v0, offset_inline1270_inline1895__ssa_v0], [64, 64], [64, 64], target_memory=pl.Mem.Mat
                )
                b_inline1269_inline1904__tile: pl.Tile[[64, 32], pl.BF16, pl.MemRef(mem_mat_5, pl.const(8192, pl.INT64), 4096), pl.Mem.Mat] = pl.tile.load(
                    wkv__ssa_v0, [offset_inline1270_inline1895__ssa_v0, column_inline1256_inline1890__ssa_v0], [64, 32], [64, 32], target_memory=pl.Mem.Mat
                )
                if group_inline1257_inline1847__ssa_v0 < 4:
                    direction_inline1264_inline1909__ssa_v0_1: pl.Scalar[pl.INDEX] = block_inline1258_inline1858__ssa_v0_1
                    if group_inline1257_inline1847__ssa_v0 % 2 == 1:
                        direction_inline1264_inline1909__ssa_v1_1: pl.Scalar[pl.INDEX] = 15 - block_inline1258_inline1858__ssa_v0_1
                        direction_inline1264_inline1909__phi_v2_1: pl.Scalar[pl.INDEX] = pl.yield_(direction_inline1264_inline1909__ssa_v1_1)
                    else:
                        direction_inline1264_inline1909__phi_v2_1: pl.Scalar[pl.INDEX] = pl.yield_(direction_inline1264_inline1909__ssa_v0_1)
                    block_inline1258_inline1858__ssa_v1_1: pl.Scalar[pl.INDEX] = (group_inline1257_inline1847__ssa_v0 // 2 * 5 + direction_inline1264_inline1909__phi_v2_1) % 16
                    block_inline1258_inline1858__phi_v2_1: pl.Scalar[pl.INDEX] = pl.yield_(block_inline1258_inline1858__ssa_v1_1)
                else:
                    block_inline1258_inline1858__phi_v2_1: pl.Scalar[pl.INDEX] = pl.yield_(block_inline1258_inline1858__ssa_v0_1)
                offset_inline1270_inline1895__ssa_v0_1: pl.Scalar[pl.INDEX] = block_inline1258_inline1858__phi_v2_1 * 256 + (step_inline1262_inline1860__idx_v0 + 1) % 4 * 64
                a_inline1263_inline1880__tile_1: pl.Tile[[64, 64], pl.BF16, pl.MemRef(mem_mat_6, pl.const(12288, pl.INT64), 8192), pl.Mem.Mat] = pl.tile.load(
                    x_normed_t__ssa_v0, [tile_base_inline1887__idx_v0 + row_inline1259_inline1868__idx_v0, offset_inline1270_inline1895__ssa_v0_1], [64, 64], [64, 64], target_memory=pl.Mem.Mat
                )
                b_inline1269_inline1904__tile_1: pl.Tile[[64, 32], pl.BF16, pl.MemRef(mem_mat_7, pl.const(20480, pl.INT64), 4096), pl.Mem.Mat] = pl.tile.load(
                    wkv__ssa_v0, [offset_inline1270_inline1895__ssa_v0_1, column_inline1256_inline1890__ssa_v0], [64, 32], [64, 32], target_memory=pl.Mem.Mat
                )
                a_inline1263_inline1880__tile_Left: pl.Tile[[64, 64], pl.BF16, pl.MemRef(mem_left_8, pl.const(8192, pl.INT64), 8192), pl.Mem.Left] = pl.tile.move(
                    a_inline1263_inline1880__tile, target_memory=pl.Mem.Left
                )
                b_inline1269_inline1904__tile_Right: pl.Tile[[64, 32], pl.BF16, pl.MemRef(mem_right_9, pl.const(4096, pl.INT64), 4096), pl.Mem.Right] = pl.tile.move(
                    b_inline1269_inline1904__tile, target_memory=pl.Mem.Right
                )
                accumulator_inline1260_inline1861__tile_1: pl.Tile[[64, 32], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 8192), pl.Mem.Acc] = pl.tile.matmul_acc(
                    accumulator_inline1260_inline1861__iter_v1, a_inline1263_inline1880__tile_Left, b_inline1269_inline1904__tile_Right, step_inline1262_inline1860__idx_v0 == 0
                )
                a_inline1263_inline1880__tile_Left_1: pl.Tile[[64, 64], pl.BF16, pl.MemRef(mem_left_10, pl.const(0, pl.INT64), 8192), pl.Mem.Left] = pl.tile.move(
                    a_inline1263_inline1880__tile_1, target_memory=pl.Mem.Left
                )
                b_inline1269_inline1904__tile_Right_1: pl.Tile[[64, 32], pl.BF16, pl.MemRef(mem_right_11, pl.const(0, pl.INT64), 4096), pl.Mem.Right] = pl.tile.move(
                    b_inline1269_inline1904__tile_1, target_memory=pl.Mem.Right
                )
                accumulator_inline1260_inline1861__tile_2: pl.Tile[[64, 32], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 8192), pl.Mem.Acc] = pl.tile.matmul_acc(
                    accumulator_inline1260_inline1861__tile_1, a_inline1263_inline1880__tile_Left_1, b_inline1269_inline1904__tile_Right_1, step_inline1262_inline1860__idx_v0 == -1
                )
                accumulator_inline1260_inline1861__rv_v2: pl.Tile[[64, 32], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 8192), pl.Mem.Acc] = pl.yield_(
                    accumulator_inline1260_inline1861__tile_2
                )
            kv_fp32_inline1850__tile: pl.Tensor[[t_matmul_inline1893__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                accumulator_inline1260_inline1861__rv_v2, [row_inline1259_inline1868__idx_v0, column_inline1256_inline1890__ssa_v0], kv_fp32_inline1850__iter_v6
            )
            kv_fp32_inline1850__rv_v7: pl.Tensor[[t_matmul_inline1893__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_13", pl.const(0, pl.INT64), 0)] = pl.yield_(kv_fp32_inline1850__tile)
        for row_inline1265_inline1874__idx_v0, (kv_fp32_inline1850__iter_v9,) in pl.range(192, 240, 16, init_values=(kv_fp32_inline1850__rv_v7,)):
            accumulator_tail_inline1266_inline1878__tile: pl.Tile[[16, 32], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 2048), pl.Mem.Acc] = pl.tile.create(
                [16, 32], dtype=pl.FP32, target_memory=pl.Mem.Acc
            )
            for step_tail_inline1255_inline1879__idx_v0, (accumulator_tail_inline1266_inline1878__iter_v1,) in pl.range(0, 64, 2, init_values=(accumulator_tail_inline1266_inline1878__tile,)):
                block_tail_inline1261_inline1848__ssa_v0: pl.Scalar[pl.INDEX] = step_tail_inline1255_inline1879__idx_v0 // 4
                block_tail_inline1261_inline1848__ssa_v0_1: pl.Scalar[pl.INDEX] = (step_tail_inline1255_inline1879__idx_v0 + 1) // 4
                if group_inline1257_inline1847__ssa_v0 < 4:
                    direction_tail_inline1268_inline1869__ssa_v0: pl.Scalar[pl.INDEX] = block_tail_inline1261_inline1848__ssa_v0
                    if group_inline1257_inline1847__ssa_v0 % 2 == 1:
                        direction_tail_inline1268_inline1869__ssa_v1: pl.Scalar[pl.INDEX] = 15 - block_tail_inline1261_inline1848__ssa_v0
                        direction_tail_inline1268_inline1869__phi_v2: pl.Scalar[pl.INDEX] = pl.yield_(direction_tail_inline1268_inline1869__ssa_v1)
                    else:
                        direction_tail_inline1268_inline1869__phi_v2: pl.Scalar[pl.INDEX] = pl.yield_(direction_tail_inline1268_inline1869__ssa_v0)
                    block_tail_inline1261_inline1848__ssa_v1: pl.Scalar[pl.INDEX] = (group_inline1257_inline1847__ssa_v0 // 2 * 5 + direction_tail_inline1268_inline1869__phi_v2) % 16
                    block_tail_inline1261_inline1848__phi_v2: pl.Scalar[pl.INDEX] = pl.yield_(block_tail_inline1261_inline1848__ssa_v1)
                else:
                    block_tail_inline1261_inline1848__phi_v2: pl.Scalar[pl.INDEX] = pl.yield_(block_tail_inline1261_inline1848__ssa_v0)
                offset_tail_inline1254_inline1859__ssa_v0: pl.Scalar[pl.INDEX] = block_tail_inline1261_inline1848__phi_v2 * 256 + step_tail_inline1255_inline1879__idx_v0 % 4 * 64
                a_tail_inline1253_inline1884__tile: pl.Tile[[16, 64], pl.BF16, pl.MemRef(mem_mat_5, pl.const(8192, pl.INT64), 2048), pl.Mem.Mat] = pl.tile.load(
                    x_normed_t__ssa_v0, [tile_base_inline1887__idx_v0 + row_inline1265_inline1874__idx_v0, offset_tail_inline1254_inline1859__ssa_v0], [16, 64], [16, 64], target_memory=pl.Mem.Mat
                )
                b_tail_inline1267_inline1846__tile: pl.Tile[[64, 32], pl.BF16, pl.MemRef(mem_mat_4, pl.const(0, pl.INT64), 4096), pl.Mem.Mat] = pl.tile.load(
                    wkv__ssa_v0, [offset_tail_inline1254_inline1859__ssa_v0, column_inline1256_inline1890__ssa_v0], [64, 32], [64, 32], target_memory=pl.Mem.Mat
                )
                if group_inline1257_inline1847__ssa_v0 < 4:
                    direction_tail_inline1268_inline1869__ssa_v0_1: pl.Scalar[pl.INDEX] = block_tail_inline1261_inline1848__ssa_v0_1
                    if group_inline1257_inline1847__ssa_v0 % 2 == 1:
                        direction_tail_inline1268_inline1869__ssa_v1_1: pl.Scalar[pl.INDEX] = 15 - block_tail_inline1261_inline1848__ssa_v0_1
                        direction_tail_inline1268_inline1869__phi_v2_1: pl.Scalar[pl.INDEX] = pl.yield_(direction_tail_inline1268_inline1869__ssa_v1_1)
                    else:
                        direction_tail_inline1268_inline1869__phi_v2_1: pl.Scalar[pl.INDEX] = pl.yield_(direction_tail_inline1268_inline1869__ssa_v0_1)
                    block_tail_inline1261_inline1848__ssa_v1_1: pl.Scalar[pl.INDEX] = (group_inline1257_inline1847__ssa_v0 // 2 * 5 + direction_tail_inline1268_inline1869__phi_v2_1) % 16
                    block_tail_inline1261_inline1848__phi_v2_1: pl.Scalar[pl.INDEX] = pl.yield_(block_tail_inline1261_inline1848__ssa_v1_1)
                else:
                    block_tail_inline1261_inline1848__phi_v2_1: pl.Scalar[pl.INDEX] = pl.yield_(block_tail_inline1261_inline1848__ssa_v0_1)
                offset_tail_inline1254_inline1859__ssa_v0_1: pl.Scalar[pl.INDEX] = block_tail_inline1261_inline1848__phi_v2_1 * 256 + (step_tail_inline1255_inline1879__idx_v0 + 1) % 4 * 64
                a_tail_inline1253_inline1884__tile_1: pl.Tile[[16, 64], pl.BF16, pl.MemRef(mem_mat_7, pl.const(20480, pl.INT64), 2048), pl.Mem.Mat] = pl.tile.load(
                    x_normed_t__ssa_v0, [tile_base_inline1887__idx_v0 + row_inline1265_inline1874__idx_v0, offset_tail_inline1254_inline1859__ssa_v0_1], [16, 64], [16, 64], target_memory=pl.Mem.Mat
                )
                b_tail_inline1267_inline1846__tile_1: pl.Tile[[64, 32], pl.BF16, pl.MemRef(mem_mat_6, pl.const(12288, pl.INT64), 4096), pl.Mem.Mat] = pl.tile.load(
                    wkv__ssa_v0, [offset_tail_inline1254_inline1859__ssa_v0_1, column_inline1256_inline1890__ssa_v0], [64, 32], [64, 32], target_memory=pl.Mem.Mat
                )
                a_tail_inline1253_inline1884__tile_Left: pl.Tile[[16, 64], pl.BF16, pl.MemRef(mem_left_8, pl.const(8192, pl.INT64), 2048), pl.Mem.Left] = pl.tile.move(
                    a_tail_inline1253_inline1884__tile, target_memory=pl.Mem.Left
                )
                b_tail_inline1267_inline1846__tile_Right: pl.Tile[[64, 32], pl.BF16, pl.MemRef(mem_right_9, pl.const(4096, pl.INT64), 4096), pl.Mem.Right] = pl.tile.move(
                    b_tail_inline1267_inline1846__tile, target_memory=pl.Mem.Right
                )
                accumulator_tail_inline1266_inline1878__tile_1: pl.Tile[[16, 32], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 2048), pl.Mem.Acc] = pl.tile.matmul_acc(
                    accumulator_tail_inline1266_inline1878__iter_v1, a_tail_inline1253_inline1884__tile_Left, b_tail_inline1267_inline1846__tile_Right, step_tail_inline1255_inline1879__idx_v0 == 0
                )
                a_tail_inline1253_inline1884__tile_Left_1: pl.Tile[[16, 64], pl.BF16, pl.MemRef(mem_left_10, pl.const(0, pl.INT64), 2048), pl.Mem.Left] = pl.tile.move(
                    a_tail_inline1253_inline1884__tile_1, target_memory=pl.Mem.Left
                )
                b_tail_inline1267_inline1846__tile_Right_1: pl.Tile[[64, 32], pl.BF16, pl.MemRef(mem_right_11, pl.const(0, pl.INT64), 4096), pl.Mem.Right] = pl.tile.move(
                    b_tail_inline1267_inline1846__tile_1, target_memory=pl.Mem.Right
                )
                accumulator_tail_inline1266_inline1878__tile_2: pl.Tile[[16, 32], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 2048), pl.Mem.Acc] = pl.tile.matmul_acc(
                    accumulator_tail_inline1266_inline1878__tile_1, a_tail_inline1253_inline1884__tile_Left_1, b_tail_inline1267_inline1846__tile_Right_1, step_tail_inline1255_inline1879__idx_v0 == -1
                )
                accumulator_tail_inline1266_inline1878__rv_v2: pl.Tile[[16, 32], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 2048), pl.Mem.Acc] = pl.yield_(
                    accumulator_tail_inline1266_inline1878__tile_2
                )
            kv_fp32_inline1850__tile_1: pl.Tensor[[t_matmul_inline1893__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_13", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                accumulator_tail_inline1266_inline1878__rv_v2, [row_inline1265_inline1874__idx_v0, column_inline1256_inline1890__ssa_v0], kv_fp32_inline1850__iter_v9
            )
            kv_fp32_inline1850__rv_v10: pl.Tensor[[t_matmul_inline1893__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_24", pl.const(0, pl.INT64), 0)] = pl.yield_(kv_fp32_inline1850__tile_1)
        return kv_fp32_inline1850__rv_v2

    @pl.function(type=pl.FunctionType.Spmd)
    def kv_proj_native_240_spmd(
        self,
        kv_fp32_inline1850__rv_v2: pl.Out[pl.Tensor[[t_matmul_inline1893__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)]],
        x_normed_t__ssa_v0: pl.Tensor[[T_DYN, 4096], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)],
        tile_base_inline1887__idx_v0: pl.Scalar[pl.INDEX],
        wkv__ssa_v0: pl.Tensor[[4096, 512], pl.BF16, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 4194304)],
    ) -> pl.Tensor[[t_matmul_inline1893__ssa_v0, 512], pl.FP32]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        kv_fp32_inline1850__rv_v10: pl.Tensor[[t_matmul_inline1893__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)] = self.kv_proj_native_240(
            kv_fp32_inline1850__rv_v2, x_normed_t__ssa_v0, tile_base_inline1887__idx_v0, wkv__ssa_v0, attrs={"arg_directions": [pl.adir.output_existing, pl.adir.input, pl.adir.scalar, pl.adir.input]}
        )
        return kv_fp32_inline1850__rv_v2

    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def kv_proj_seed(
        kv_fp32_inline1850__ssa_v0: pl.Out[pl.Tensor[[t_matmul_inline1893__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)]], t_matmul_inline1893__ssa_v0: pl.Scalar[pl.INDEX]
    ) -> pl.Tensor[[t_matmul_inline1893__ssa_v0, 512], pl.FP32]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        mem_vec_1: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 8192)
        for kts0_inline1891__idx_v0, (kv_fp32_inline1850__iter_v1,) in pl.range(0, t_matmul_inline1893__ssa_v0, 16, init_values=(kv_fp32_inline1850__ssa_v0,)):
            for kvseed0_inline1885__idx_v0, (kv_fp32_inline1850__iter_v3,) in pl.range(0, 512, 128, init_values=(kv_fp32_inline1850__iter_v1,)):
                kv_seed_inline1882__tile: pl.Tile[[16, 128], pl.FP32, pl.MemRef(mem_vec_1, pl.const(0, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.full([16, 128], dtype=pl.FP32, value=0.0)
                kv_fp32_inline1850__tile: pl.Tensor[[t_matmul_inline1893__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                    kv_seed_inline1882__tile, [kts0_inline1891__idx_v0, kvseed0_inline1885__idx_v0], kv_fp32_inline1850__iter_v3
                )
                kv_fp32_inline1850__rv_v4: pl.Tensor[[t_matmul_inline1893__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)] = pl.yield_(kv_fp32_inline1850__tile)
            kv_fp32_inline1850__rv_v2: pl.Tensor[[t_matmul_inline1893__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)] = pl.yield_(kv_fp32_inline1850__rv_v4)
        return kv_fp32_inline1850__ssa_v0

    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def kv_rms_norm_rope(
        tile_rows_inline1877__ssa_v0: pl.Scalar[pl.INDEX],
        tile_base_inline1887__idx_v0: pl.Scalar[pl.INDEX],
        kv_fp32_inline1850__phi_v18: pl.Tensor[[t_matmul_inline1893__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)],
        kv_view_inline1915__ssa_v0: pl.InOut[pl.Tensor[[t_dim_inline1873__ssa_v0, 512], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)]],
        gamma_ckv__ssa_v0: pl.Tensor[[512], pl.BF16, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 1024)],
        freqs_cos__ssa_v0: pl.Tensor[[T_DYN, 64], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)],
        q_rope_sin_signed_inline197__ssa_v0: pl.Tensor[[t_dim_inline198__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 0)],
        q_rope_swap_idx_inline196__ssa_v0: pl.Tensor[[t_dim_inline198__ssa_v0, 64], pl.INT32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 0)],
    ) -> pl.Tensor[[t_dim_inline1873__ssa_v0, 512], pl.BF16]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        mem_vec_6: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 8192)
        mem_vec_8: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 8192)
        mem_vec_18: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
        mem_vec_62: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 8192)
        mem_vec_63: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 8192)
        mem_vec_144: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 8192)
        mem_vec_145: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 8192)
        tg_idx_inline1918__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        tg_inline1881__ssa_v0: pl.Scalar[pl.INDEX] = tg_idx_inline1918__ssa_v0 * 32
        valid_rows_inline1919__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(tile_rows_inline1877__ssa_v0 - tg_inline1881__ssa_v0, 32)
        out_tg_inline1923__ssa_v0: pl.Scalar[pl.INDEX] = tile_base_inline1887__idx_v0 + tg_inline1881__ssa_v0
        if valid_rows_inline1919__ssa_v0 == 32:
            kv_sq_lanes_inline1899__tile: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_6, pl.const(32768, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.full([32, 64], dtype=pl.FP32, value=0.0)
            for kv_sq_col0_inline1892__idx_v0, (kv_sq_lanes_inline1899__iter_v1,) in pl.range(0, 512, 128, init_values=(kv_sq_lanes_inline1899__tile,)):
                kv_chunk_inline1924__tile: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_18, pl.const(16384, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.load(
                    kv_fp32_inline1850__phi_v18, [tg_inline1881__ssa_v0, kv_sq_col0_inline1892__idx_v0], [32, 64], [32, 64], target_memory=pl.Mem.Vec
                )
                kv_chunk_inline1924__tile_1: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_8, pl.const(57344, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.load(
                    kv_fp32_inline1850__phi_v18, [tg_inline1881__ssa_v0, kv_sq_col0_inline1892__idx_v0 + 64], [32, 64], [32, 64], target_memory=pl.Mem.Vec
                )
                t__tile: pl.Tile[[32, 64], pl.BF16, pl.MemRef(mem_vec_62, pl.const(40960, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(kv_chunk_inline1924__tile, target_type=pl.BF16, mode="rint")
                kv_chunk_v1_inline1925__tile: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_18, pl.const(16384, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.cast(t__tile, target_type=pl.FP32, mode="round")
                kv_sq_inline1843__tile: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_18, pl.const(16384, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.mul(
                    kv_chunk_v1_inline1925__tile, kv_chunk_v1_inline1925__tile
                )
                kv_sq_lanes_inline1899__tile_1: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_18, pl.const(16384, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.add(
                    kv_sq_lanes_inline1899__iter_v1, kv_sq_inline1843__tile
                )
                t__tile_1: pl.Tile[[32, 64], pl.BF16, pl.MemRef(mem_vec_63, pl.const(49152, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(kv_chunk_inline1924__tile_1, target_type=pl.BF16, mode="rint")
                kv_chunk_v1_inline1925__tile_1: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_8, pl.const(57344, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.cast(
                    t__tile_1, target_type=pl.FP32, mode="round"
                )
                kv_sq_inline1843__tile_1: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_8, pl.const(57344, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.mul(
                    kv_chunk_v1_inline1925__tile_1, kv_chunk_v1_inline1925__tile_1
                )
                kv_sq_lanes_inline1899__tile_2: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_6, pl.const(32768, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.add(
                    kv_sq_lanes_inline1899__tile_1, kv_sq_inline1843__tile_1
                )
                kv_sq_lanes_inline1899__rv_v2: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_6, pl.const(32768, pl.INT64), 8192), pl.Mem.Vec] = pl.yield_(kv_sq_lanes_inline1899__tile_2)
            tmp_tile: pl.Tile[[32, 128], pl.FP32, pl.MemRef(mem_vec_18, pl.const(16384, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.create([32, 128], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            t__tile_2: pl.Tile[[32, 1], pl.FP32, pl.MemRef(mem_vec_8, pl.const(57344, pl.INT64), 128), pl.Mem.Vec] = pl.tile.row_sum(kv_sq_lanes_inline1899__rv_v2, tmp_tile)
            kv_sq_sum_inline1842__tile: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_8, pl.const(57344, pl.INT64), 128), pl.Mem.Vec] = pl.tile.reshape(t__tile_2, [1, 32])
            t__tile_3: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_18, pl.const(16384, pl.INT64), 128), pl.Mem.Vec] = pl.tile.muls(kv_sq_sum_inline1842__tile, 0.001953125)
            t__tile_4: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_18, pl.const(16384, pl.INT64), 128), pl.Mem.Vec] = pl.tile.adds(t__tile_3, 9.9999999999999995e-07)
            kv_rms_inline1922__tile: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_18, pl.const(16384, pl.INT64), 128), pl.Mem.Vec] = pl.tile.sqrt(t__tile_4)
            t__tile_5: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_6, pl.const(32768, pl.INT64), 128), pl.Mem.Vec] = pl.tile.full([1, 32], dtype=pl.FP32, value=1.0)
            kv_inv_rms_inline1841__tile: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_63, pl.const(49152, pl.INT64), 128), pl.Mem.Vec] = pl.tile.div(t__tile_5, kv_rms_inline1922__tile)
            kv_inv_rms_t_inline1866__tile: pl.Tile[[32, 1], pl.FP32, pl.MemRef(mem_vec_63, pl.const(49152, pl.INT64), 128), pl.Mem.Vec] = pl.tile.reshape(kv_inv_rms_inline1841__tile, [32, 1])
            for n0_inline1840__idx_v0, (kv_view_inline1915__iter_v1,) in pl.range(0, 384, 128, init_values=(kv_view_inline1915__ssa_v0,)):
                kv_chunk_inline1924__tile_2: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_18, pl.const(16384, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.load(
                    kv_fp32_inline1850__phi_v18, [tg_inline1881__ssa_v0, n0_inline1840__idx_v0], [32, 64], [32, 64], target_memory=pl.Mem.Vec
                )
                t__tile_6: pl.Tile[[64], pl.BF16, pl.MemRef(mem_vec_144, pl.const(0, pl.INT64), 128), pl.Mem.Vec] = pl.tile.load(
                    gamma_ckv__ssa_v0, [n0_inline1840__idx_v0], [64], [64], target_memory=pl.Mem.Vec
                )
                kv_chunk_inline1924__tile_3: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_6, pl.const(32768, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.load(
                    kv_fp32_inline1850__phi_v18, [tg_inline1881__ssa_v0, n0_inline1840__idx_v0 + 64], [32, 64], [32, 64], target_memory=pl.Mem.Vec
                )
                t__tile_7: pl.Tile[[64], pl.BF16, pl.MemRef(mem_vec_145, pl.const(8192, pl.INT64), 128), pl.Mem.Vec] = pl.tile.load(
                    gamma_ckv__ssa_v0, [n0_inline1840__idx_v0 + 64], [64], [64], target_memory=pl.Mem.Vec
                )
                t__tile_8: pl.Tile[[32, 64], pl.BF16, pl.MemRef(mem_vec_8, pl.const(57344, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(kv_chunk_inline1924__tile_2, target_type=pl.BF16, mode="rint")
                kv_chunk_v2_inline1876__tile: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_18, pl.const(16384, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.cast(
                    t__tile_8, target_type=pl.FP32, mode="round"
                )
                gamma_kv_cast_inline1839__tile: pl.Tile[[64], pl.FP32, pl.MemRef(mem_vec_8, pl.const(57344, pl.INT64), 256), pl.Mem.Vec] = pl.tile.cast(t__tile_6, target_type=pl.FP32, mode="round")
                gamma_kv_chunk_inline1838__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_8, pl.const(57344, pl.INT64), 256), pl.Mem.Vec] = gamma_kv_cast_inline1839__tile
                t__tile_9: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_18, pl.const(16384, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.row_expand_mul(
                    kv_chunk_v2_inline1876__tile, kv_inv_rms_t_inline1866__tile
                )
                kv_normed_inline1837__tile: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_18, pl.const(16384, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.col_expand_mul(
                    t__tile_9, gamma_kv_chunk_inline1838__tile
                )
                kv_normed_bf16_inline1903__tile: pl.Tile[[32, 64], pl.BF16, pl.MemRef(mem_vec_18, pl.const(16384, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(
                    kv_normed_inline1837__tile, target_type=pl.BF16, mode="rint"
                )
                kv_view_inline1915__tile: pl.Tensor[[t_dim_inline1873__ssa_v0, 512], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                    kv_normed_bf16_inline1903__tile, [out_tg_inline1923__ssa_v0, n0_inline1840__idx_v0], kv_view_inline1915__iter_v1
                )
                t__tile_10: pl.Tile[[32, 64], pl.BF16, pl.MemRef(mem_vec_62, pl.const(40960, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(kv_chunk_inline1924__tile_3, target_type=pl.BF16, mode="rint")
                kv_chunk_v2_inline1876__tile_1: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_6, pl.const(32768, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.cast(
                    t__tile_10, target_type=pl.FP32, mode="round"
                )
                gamma_kv_cast_inline1839__tile_1: pl.Tile[[64], pl.FP32, pl.MemRef(mem_vec_62, pl.const(40960, pl.INT64), 256), pl.Mem.Vec] = pl.tile.cast(t__tile_7, target_type=pl.FP32, mode="round")
                gamma_kv_chunk_inline1838__tile_1: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_62, pl.const(40960, pl.INT64), 256), pl.Mem.Vec] = gamma_kv_cast_inline1839__tile_1
                t__tile_11: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_6, pl.const(32768, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.row_expand_mul(
                    kv_chunk_v2_inline1876__tile_1, kv_inv_rms_t_inline1866__tile
                )
                kv_normed_inline1837__tile_1: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_6, pl.const(32768, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.col_expand_mul(
                    t__tile_11, gamma_kv_chunk_inline1838__tile_1
                )
                kv_normed_bf16_inline1903__tile_1: pl.Tile[[32, 64], pl.BF16, pl.MemRef(mem_vec_6, pl.const(32768, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(
                    kv_normed_inline1837__tile_1, target_type=pl.BF16, mode="rint"
                )
                kv_view_inline1915__tile_1: pl.Tensor[[t_dim_inline1873__ssa_v0, 512], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                    kv_normed_bf16_inline1903__tile_1, [out_tg_inline1923__ssa_v0, n0_inline1840__idx_v0 + 64], kv_view_inline1915__tile
                )
                kv_view_inline1915__rv_v2_main: pl.Tensor[[t_dim_inline1873__ssa_v0, 512], pl.BF16, pl.MemRef("mem_ddr_41", pl.const(0, pl.INT64), 0)] = pl.yield_(kv_view_inline1915__tile_1)
            kv_chunk_inline1924__tile_4: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_18, pl.const(16384, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.load(
                kv_fp32_inline1850__phi_v18, [tg_inline1881__ssa_v0, 384], [32, 64], [32, 64], target_memory=pl.Mem.Vec
            )
            t__tile_12: pl.Tile[[32, 64], pl.BF16, pl.MemRef(mem_vec_6, pl.const(32768, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(kv_chunk_inline1924__tile_4, target_type=pl.BF16, mode="rint")
            kv_chunk_v2_inline1876__tile_2: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_18, pl.const(16384, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.cast(t__tile_12, target_type=pl.FP32, mode="round")
            t__tile_13: pl.Tile[[64], pl.BF16, pl.MemRef(mem_vec_8, pl.const(57344, pl.INT64), 128), pl.Mem.Vec] = pl.tile.load(gamma_ckv__ssa_v0, [384], [64], [64], target_memory=pl.Mem.Vec)
            gamma_kv_cast_inline1839__tile_2: pl.Tile[[64], pl.FP32, pl.MemRef(mem_vec_6, pl.const(32768, pl.INT64), 256), pl.Mem.Vec] = pl.tile.cast(t__tile_13, target_type=pl.FP32, mode="round")
            gamma_kv_chunk_inline1838__tile_2: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_6, pl.const(32768, pl.INT64), 256), pl.Mem.Vec] = gamma_kv_cast_inline1839__tile_2
            t__tile_14: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_18, pl.const(16384, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.row_expand_mul(
                kv_chunk_v2_inline1876__tile_2, kv_inv_rms_t_inline1866__tile
            )
            kv_normed_inline1837__tile_2: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_18, pl.const(16384, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.col_expand_mul(
                t__tile_14, gamma_kv_chunk_inline1838__tile_2
            )
            kv_normed_bf16_inline1903__tile_2: pl.Tile[[32, 64], pl.BF16, pl.MemRef(mem_vec_18, pl.const(16384, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(
                kv_normed_inline1837__tile_2, target_type=pl.BF16, mode="rint"
            )
            kv_view_inline1915__tile_2: pl.Tensor[[t_dim_inline1873__ssa_v0, 512], pl.BF16, pl.MemRef("mem_ddr_41", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                kv_normed_bf16_inline1903__tile_2, [out_tg_inline1923__ssa_v0, 384], kv_view_inline1915__rv_v2_main
            )
            kv_view_inline1915__rv_v2: pl.Tensor[[t_dim_inline1873__ssa_v0, 512], pl.BF16, pl.MemRef("mem_ddr_50", pl.const(0, pl.INT64), 0)] = kv_view_inline1915__tile_2
            t__tile_15: pl.Tile[[64], pl.BF16, pl.MemRef(mem_vec_18, pl.const(16384, pl.INT64), 128), pl.Mem.Vec] = pl.tile.load(gamma_ckv__ssa_v0, [448], [64], [64], target_memory=pl.Mem.Vec)
            gamma_rope_cast_inline1836__tile: pl.Tile[[64], pl.FP32, pl.MemRef(mem_vec_8, pl.const(57344, pl.INT64), 256), pl.Mem.Vec] = pl.tile.cast(t__tile_15, target_type=pl.FP32, mode="round")
            gamma_rope_inline1833__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_8, pl.const(57344, pl.INT64), 256), pl.Mem.Vec] = gamma_rope_cast_inline1836__tile
            kv_rope_chunk_inline1830__tile: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_18, pl.const(16384, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.load(
                kv_fp32_inline1850__phi_v18, [tg_inline1881__ssa_v0, 448], [32, 64], [32, 64], target_memory=pl.Mem.Vec
            )
            t__tile_16: pl.Tile[[32, 64], pl.BF16, pl.MemRef(mem_vec_6, pl.const(32768, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(kv_rope_chunk_inline1830__tile, target_type=pl.BF16, mode="rint")
            kv_rope_chunk_v1_inline1828__tile: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_18, pl.const(16384, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.cast(
                t__tile_16, target_type=pl.FP32, mode="round"
            )
            t__tile_17: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_18, pl.const(16384, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.row_expand_mul(
                kv_rope_chunk_v1_inline1828__tile, kv_inv_rms_t_inline1866__tile
            )
            kv_rope_norm_chunk_inline1883__tile: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_18, pl.const(16384, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.col_expand_mul(
                t__tile_17, gamma_rope_inline1833__tile
            )
            t__tile_18: pl.Tile[[32, 64], pl.BF16, pl.MemRef(mem_vec_6, pl.const(32768, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(
                kv_rope_norm_chunk_inline1883__tile, target_type=pl.BF16, mode="rint"
            )
            kv_rope_norm_chunk_v1_inline1900__tile: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_18, pl.const(16384, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.cast(
                t__tile_18, target_type=pl.FP32, mode="round"
            )
            kv_cos_il_full_inline1827__tile: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_6, pl.const(32768, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.load(
                freqs_cos__ssa_v0, [out_tg_inline1923__ssa_v0, 0], [32, 64], [32, 64], target_memory=pl.Mem.Vec
            )
            kv_sin_signed_full_inline1825__tile: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_8, pl.const(57344, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.load(
                q_rope_sin_signed_inline197__ssa_v0, [out_tg_inline1923__ssa_v0, 0], [32, 64], [32, 64], target_memory=pl.Mem.Vec
            )
            kv_swap_idx_full_inline1898__tile: pl.Tile[[32, 64], pl.INT32, pl.MemRef(mem_vec_62, pl.const(40960, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.load(
                q_rope_swap_idx_inline196__ssa_v0, [out_tg_inline1923__ssa_v0, 0], [32, 64], [32, 64], target_memory=pl.Mem.Vec
            )
            gather_acc_init: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_63, pl.const(49152, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.create([32, 64], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            for gather_lv, (gather_ia,) in pl.range(32, init_values=(gather_acc_init,)):
                gather_inp_row: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_18, pl.const(16384, pl.INT64), 256), pl.Mem.Vec] = pl.tile.slice(
                    kv_rope_norm_chunk_v1_inline1900__tile, [1, 64], [gather_lv, 0], [1, 64]
                )
                gather_idx_row: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_62, pl.const(40960, pl.INT64), 256), pl.Mem.Vec] = pl.tile.slice(
                    kv_swap_idx_full_inline1898__tile, [1, 64], [gather_lv, 0], [1, 64]
                )
                gather_row_tmp: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_144, pl.const(0, pl.INT64), 256), pl.Mem.Vec] = pl.tile.create([1, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec)
                gather_row: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_145, pl.const(8192, pl.INT64), 256), pl.Mem.Vec] = pl.tile.gather(gather_inp_row, gather_idx_row, gather_row_tmp)
                gather_asmbl: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_63, pl.const(49152, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.assemble(gather_ia, gather_row, [gather_lv, 0])
                gather_rv: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_63, pl.const(49152, pl.INT64), 8192), pl.Mem.Vec] = pl.yield_(gather_asmbl)
            kv_swapped_full_inline1824__tile: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_63, pl.const(49152, pl.INT64), 8192), pl.Mem.Vec] = gather_rv
            t__tile_19: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_18, pl.const(16384, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.mul(
                kv_rope_norm_chunk_v1_inline1900__tile, kv_cos_il_full_inline1827__tile
            )
            t__tile_20: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_6, pl.const(32768, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.mul(
                kv_swapped_full_inline1824__tile, kv_sin_signed_full_inline1825__tile
            )
            kv_rope_rot_full_inline1822__tile: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_18, pl.const(16384, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.add(t__tile_19, t__tile_20)
            kv_rope_i16_full_inline1854__tile: pl.Tile[[32, 64], pl.BF16, pl.MemRef(mem_vec_18, pl.const(16384, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(
                kv_rope_rot_full_inline1822__tile, target_type=pl.BF16, mode="rint"
            )
            kv_view_inline1915__tile_3: pl.Tensor[[t_dim_inline1873__ssa_v0, 512], pl.BF16, pl.MemRef("mem_ddr_50", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                kv_rope_i16_full_inline1854__tile, [out_tg_inline1923__ssa_v0, 448], kv_view_inline1915__rv_v2
            )
        else:
            kv_reduce_tmp_inline1821__ssa_v0: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_18, pl.const(16384, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.create(
                [32, 64], dtype=pl.FP32, target_memory=pl.Mem.Vec
            )
            kv_sq_lanes_tail_inline1820__ssa_v0: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_6, pl.const(32768, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.full([32, 64], dtype=pl.FP32, value=0.0)
            for kv_sq_col0_tail_inline1916__idx_v0, (kv_sq_lanes_tail_inline1820__iter_v1,) in pl.range(0, 512, 128, init_values=(kv_sq_lanes_tail_inline1820__ssa_v0,)):
                kv_chunk_tail_inline1857__ssa_v0: pl.Tile[
                    [32, 64], pl.FP32, pl.MemRef(mem_vec_8, pl.const(57344, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1919__ssa_v0, 64])
                ] = pl.tile.load(kv_fp32_inline1850__phi_v18, [tg_inline1881__ssa_v0, kv_sq_col0_tail_inline1916__idx_v0], [32, 64], [valid_rows_inline1919__ssa_v0, 64], target_memory=pl.Mem.Vec)
                kv_chunk_tail_inline1857__ssa_v0_1: pl.Tile[
                    [32, 64], pl.FP32, pl.MemRef(mem_vec_62, pl.const(40960, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1919__ssa_v0, 64])
                ] = pl.tile.load(kv_fp32_inline1850__phi_v18, [tg_inline1881__ssa_v0, kv_sq_col0_tail_inline1916__idx_v0 + 64], [32, 64], [valid_rows_inline1919__ssa_v0, 64], target_memory=pl.Mem.Vec)
                t__tmp_v88: pl.Tile[[32, 64], pl.BF16, pl.MemRef(mem_vec_63, pl.const(49152, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1919__ssa_v0, 64])] = pl.tile.cast(
                    kv_chunk_tail_inline1857__ssa_v0, target_type=pl.BF16, mode="rint"
                )
                kv_chunk_tail_v1_inline1845__ssa_v0: pl.Tile[
                    [32, 64], pl.FP32, pl.MemRef(mem_vec_8, pl.const(57344, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1919__ssa_v0, 64])
                ] = pl.tile.cast(t__tmp_v88, target_type=pl.FP32, mode="round")
                kv_sq_tail_inline1844__ssa_v0: pl.Tile[
                    [32, 64], pl.FP32, pl.MemRef(mem_vec_8, pl.const(57344, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1919__ssa_v0, 64])
                ] = pl.tile.mul(kv_chunk_tail_v1_inline1845__ssa_v0, kv_chunk_tail_v1_inline1845__ssa_v0)
                kv_sq_lanes_tail_inline1820__ssa_v3: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_8, pl.const(57344, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.add(
                    kv_sq_lanes_tail_inline1820__iter_v1, kv_sq_tail_inline1844__ssa_v0
                )
                t__tmp_v88_1: pl.Tile[[32, 64], pl.BF16, pl.MemRef(mem_vec_144, pl.const(0, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1919__ssa_v0, 64])] = pl.tile.cast(
                    kv_chunk_tail_inline1857__ssa_v0_1, target_type=pl.BF16, mode="rint"
                )
                kv_chunk_tail_v1_inline1845__ssa_v0_1: pl.Tile[
                    [32, 64], pl.FP32, pl.MemRef(mem_vec_62, pl.const(40960, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1919__ssa_v0, 64])
                ] = pl.tile.cast(t__tmp_v88_1, target_type=pl.FP32, mode="round")
                kv_sq_tail_inline1844__ssa_v0_1: pl.Tile[
                    [32, 64], pl.FP32, pl.MemRef(mem_vec_62, pl.const(40960, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1919__ssa_v0, 64])
                ] = pl.tile.mul(kv_chunk_tail_v1_inline1845__ssa_v0_1, kv_chunk_tail_v1_inline1845__ssa_v0_1)
                kv_sq_lanes_tail_inline1820__ssa_v3_1: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_6, pl.const(32768, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.add(
                    kv_sq_lanes_tail_inline1820__ssa_v3, kv_sq_tail_inline1844__ssa_v0_1
                )
                kv_sq_lanes_tail_inline1820__rv_v2: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_6, pl.const(32768, pl.INT64), 8192), pl.Mem.Vec] = pl.yield_(kv_sq_lanes_tail_inline1820__ssa_v3_1)
            t__tmp_v89: pl.Tile[[32, 1], pl.FP32, pl.MemRef(mem_vec_8, pl.const(57344, pl.INT64), 128), pl.Mem.Vec] = pl.tile.row_sum(
                kv_sq_lanes_tail_inline1820__rv_v2, kv_reduce_tmp_inline1821__ssa_v0
            )
            kv_sq_sum_tail_inline1819__ssa_v0: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_8, pl.const(57344, pl.INT64), 128), pl.Mem.Vec] = pl.tile.reshape(t__tmp_v89, [1, 32])
            t__tmp_v90: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_18, pl.const(16384, pl.INT64), 128), pl.Mem.Vec] = pl.tile.muls(kv_sq_sum_tail_inline1819__ssa_v0, 0.001953125)
            t__tmp_v91: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_18, pl.const(16384, pl.INT64), 128), pl.Mem.Vec] = pl.tile.adds(t__tmp_v90, 9.9999999999999995e-07)
            kv_rms_tail_inline1818__ssa_v0: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_18, pl.const(16384, pl.INT64), 128), pl.Mem.Vec] = pl.tile.sqrt(t__tmp_v91)
            t__tmp_v92: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_6, pl.const(32768, pl.INT64), 128), pl.Mem.Vec] = pl.tile.full([1, 32], dtype=pl.FP32, value=1.0)
            kv_inv_rms_tail_inline1817__ssa_v0: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_63, pl.const(49152, pl.INT64), 128), pl.Mem.Vec] = pl.tile.div(t__tmp_v92, kv_rms_tail_inline1818__ssa_v0)
            kv_inv_rms_t_tail_inline1816__ssa_v0: pl.Tile[[32, 1], pl.FP32, pl.MemRef(mem_vec_63, pl.const(49152, pl.INT64), 128), pl.Mem.Vec] = pl.tile.reshape(
                kv_inv_rms_tail_inline1817__ssa_v0, [32, 1]
            )
            for n0_tail_inline1815__idx_v0 in pl.range(0, 384, 128):
                kv_chunk_tail_inline1857__ssa_v1: pl.Tile[
                    [32, 64], pl.FP32, pl.MemRef(mem_vec_18, pl.const(16384, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1919__ssa_v0, 64])
                ] = pl.tile.load(kv_fp32_inline1850__phi_v18, [tg_inline1881__ssa_v0, n0_tail_inline1815__idx_v0], [32, 64], [valid_rows_inline1919__ssa_v0, 64], target_memory=pl.Mem.Vec)
                gamma_kv_input_tail_inline1886__ssa_v0: pl.Tile[[64], pl.BF16, pl.MemRef(mem_vec_144, pl.const(0, pl.INT64), 128), pl.Mem.Vec] = pl.tile.load(
                    gamma_ckv__ssa_v0, [n0_tail_inline1815__idx_v0], [64], [64], target_memory=pl.Mem.Vec
                )
                kv_chunk_tail_inline1857__ssa_v1_1: pl.Tile[
                    [32, 64], pl.FP32, pl.MemRef(mem_vec_6, pl.const(32768, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1919__ssa_v0, 64])
                ] = pl.tile.load(kv_fp32_inline1850__phi_v18, [tg_inline1881__ssa_v0, n0_tail_inline1815__idx_v0 + 64], [32, 64], [valid_rows_inline1919__ssa_v0, 64], target_memory=pl.Mem.Vec)
                gamma_kv_input_tail_inline1886__ssa_v0_1: pl.Tile[[64], pl.BF16, pl.MemRef(mem_vec_145, pl.const(8192, pl.INT64), 128), pl.Mem.Vec] = pl.tile.load(
                    gamma_ckv__ssa_v0, [n0_tail_inline1815__idx_v0 + 64], [64], [64], target_memory=pl.Mem.Vec
                )
                t__tmp_v93: pl.Tile[[32, 64], pl.BF16, pl.MemRef(mem_vec_8, pl.const(57344, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1919__ssa_v0, 64])] = pl.tile.cast(
                    kv_chunk_tail_inline1857__ssa_v1, target_type=pl.BF16, mode="rint"
                )
                kv_chunk_tail_v2_inline1921__ssa_v0: pl.Tile[
                    [32, 64], pl.FP32, pl.MemRef(mem_vec_18, pl.const(16384, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1919__ssa_v0, 64])
                ] = pl.tile.cast(t__tmp_v93, target_type=pl.FP32, mode="round")
                gamma_kv_cast_tail_inline1814__ssa_v0: pl.Tile[[64], pl.FP32, pl.MemRef(mem_vec_8, pl.const(57344, pl.INT64), 256), pl.Mem.Vec] = pl.tile.cast(
                    gamma_kv_input_tail_inline1886__ssa_v0, target_type=pl.FP32, mode="round"
                )
                gamma_kv_chunk_tail_inline1852__ssa_v0: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_8, pl.const(57344, pl.INT64), 256), pl.Mem.Vec] = gamma_kv_cast_tail_inline1814__ssa_v0
                t__tmp_v94: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_18, pl.const(16384, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1919__ssa_v0, 64])] = (
                    pl.tile.row_expand_mul(kv_chunk_tail_v2_inline1921__ssa_v0, kv_inv_rms_t_tail_inline1816__ssa_v0)
                )
                kv_normed_tail_inline1865__ssa_v0: pl.Tile[
                    [32, 64], pl.FP32, pl.MemRef(mem_vec_18, pl.const(16384, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1919__ssa_v0, 64])
                ] = pl.tile.col_expand_mul(t__tmp_v94, gamma_kv_chunk_tail_inline1852__ssa_v0)
                kv_normed_bf16_tail_inline1813__ssa_v0: pl.Tile[
                    [32, 64], pl.BF16, pl.MemRef(mem_vec_18, pl.const(16384, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1919__ssa_v0, 64])
                ] = pl.tile.cast(kv_normed_tail_inline1865__ssa_v0, target_type=pl.BF16, mode="rint")
                kv_normed_valid_inline1864__ssa_v0: pl.Tile[
                    [32, 64], pl.BF16, pl.MemRef(mem_vec_18, pl.const(16384, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1919__ssa_v0, 64])
                ] = pl.tile.set_validshape(kv_normed_bf16_tail_inline1813__ssa_v0, valid_rows_inline1919__ssa_v0, 64)
                kv_view_inline1915__store: pl.Tensor[[t_dim_inline1873__ssa_v0, 512], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                    kv_normed_valid_inline1864__ssa_v0, [out_tg_inline1923__ssa_v0, n0_tail_inline1815__idx_v0], kv_view_inline1915__ssa_v0
                )
                t__tmp_v93_1: pl.Tile[[32, 64], pl.BF16, pl.MemRef(mem_vec_62, pl.const(40960, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1919__ssa_v0, 64])] = (
                    pl.tile.cast(kv_chunk_tail_inline1857__ssa_v1_1, target_type=pl.BF16, mode="rint")
                )
                kv_chunk_tail_v2_inline1921__ssa_v0_1: pl.Tile[
                    [32, 64], pl.FP32, pl.MemRef(mem_vec_6, pl.const(32768, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1919__ssa_v0, 64])
                ] = pl.tile.cast(t__tmp_v93_1, target_type=pl.FP32, mode="round")
                gamma_kv_cast_tail_inline1814__ssa_v0_1: pl.Tile[[64], pl.FP32, pl.MemRef(mem_vec_62, pl.const(40960, pl.INT64), 256), pl.Mem.Vec] = pl.tile.cast(
                    gamma_kv_input_tail_inline1886__ssa_v0_1, target_type=pl.FP32, mode="round"
                )
                gamma_kv_chunk_tail_inline1852__ssa_v0_1: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_62, pl.const(40960, pl.INT64), 256), pl.Mem.Vec] = gamma_kv_cast_tail_inline1814__ssa_v0_1
                t__tmp_v94_1: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_6, pl.const(32768, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1919__ssa_v0, 64])] = (
                    pl.tile.row_expand_mul(kv_chunk_tail_v2_inline1921__ssa_v0_1, kv_inv_rms_t_tail_inline1816__ssa_v0)
                )
                kv_normed_tail_inline1865__ssa_v0_1: pl.Tile[
                    [32, 64], pl.FP32, pl.MemRef(mem_vec_6, pl.const(32768, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1919__ssa_v0, 64])
                ] = pl.tile.col_expand_mul(t__tmp_v94_1, gamma_kv_chunk_tail_inline1852__ssa_v0_1)
                kv_normed_bf16_tail_inline1813__ssa_v0_1: pl.Tile[
                    [32, 64], pl.BF16, pl.MemRef(mem_vec_6, pl.const(32768, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1919__ssa_v0, 64])
                ] = pl.tile.cast(kv_normed_tail_inline1865__ssa_v0_1, target_type=pl.BF16, mode="rint")
                kv_normed_valid_inline1864__ssa_v0_1: pl.Tile[
                    [32, 64], pl.BF16, pl.MemRef(mem_vec_6, pl.const(32768, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1919__ssa_v0, 64])
                ] = pl.tile.set_validshape(kv_normed_bf16_tail_inline1813__ssa_v0_1, valid_rows_inline1919__ssa_v0, 64)
                kv_view_inline1915__store_1: pl.Tensor[[t_dim_inline1873__ssa_v0, 512], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                    kv_normed_valid_inline1864__ssa_v0_1, [out_tg_inline1923__ssa_v0, n0_tail_inline1815__idx_v0 + 64], kv_view_inline1915__ssa_v0
                )
            kv_chunk_tail_inline1857__ssa_v1_2: pl.Tile[
                [32, 64], pl.FP32, pl.MemRef(mem_vec_18, pl.const(16384, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1919__ssa_v0, 64])
            ] = pl.tile.load(kv_fp32_inline1850__phi_v18, [tg_inline1881__ssa_v0, 384], [32, 64], [valid_rows_inline1919__ssa_v0, 64], target_memory=pl.Mem.Vec)
            t__tmp_v93_2: pl.Tile[[32, 64], pl.BF16, pl.MemRef(mem_vec_6, pl.const(32768, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1919__ssa_v0, 64])] = pl.tile.cast(
                kv_chunk_tail_inline1857__ssa_v1_2, target_type=pl.BF16, mode="rint"
            )
            kv_chunk_tail_v2_inline1921__ssa_v0_2: pl.Tile[
                [32, 64], pl.FP32, pl.MemRef(mem_vec_18, pl.const(16384, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1919__ssa_v0, 64])
            ] = pl.tile.cast(t__tmp_v93_2, target_type=pl.FP32, mode="round")
            gamma_kv_input_tail_inline1886__ssa_v0_2: pl.Tile[[64], pl.BF16, pl.MemRef(mem_vec_8, pl.const(57344, pl.INT64), 128), pl.Mem.Vec] = pl.tile.load(
                gamma_ckv__ssa_v0, [384], [64], [64], target_memory=pl.Mem.Vec
            )
            gamma_kv_cast_tail_inline1814__ssa_v0_2: pl.Tile[[64], pl.FP32, pl.MemRef(mem_vec_6, pl.const(32768, pl.INT64), 256), pl.Mem.Vec] = pl.tile.cast(
                gamma_kv_input_tail_inline1886__ssa_v0_2, target_type=pl.FP32, mode="round"
            )
            gamma_kv_chunk_tail_inline1852__ssa_v0_2: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_6, pl.const(32768, pl.INT64), 256), pl.Mem.Vec] = gamma_kv_cast_tail_inline1814__ssa_v0_2
            t__tmp_v94_2: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_18, pl.const(16384, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1919__ssa_v0, 64])] = (
                pl.tile.row_expand_mul(kv_chunk_tail_v2_inline1921__ssa_v0_2, kv_inv_rms_t_tail_inline1816__ssa_v0)
            )
            kv_normed_tail_inline1865__ssa_v0_2: pl.Tile[
                [32, 64], pl.FP32, pl.MemRef(mem_vec_18, pl.const(16384, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1919__ssa_v0, 64])
            ] = pl.tile.col_expand_mul(t__tmp_v94_2, gamma_kv_chunk_tail_inline1852__ssa_v0_2)
            kv_normed_bf16_tail_inline1813__ssa_v0_2: pl.Tile[
                [32, 64], pl.BF16, pl.MemRef(mem_vec_18, pl.const(16384, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1919__ssa_v0, 64])
            ] = pl.tile.cast(kv_normed_tail_inline1865__ssa_v0_2, target_type=pl.BF16, mode="rint")
            kv_normed_valid_inline1864__ssa_v0_2: pl.Tile[
                [32, 64], pl.BF16, pl.MemRef(mem_vec_18, pl.const(16384, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1919__ssa_v0, 64])
            ] = pl.tile.set_validshape(kv_normed_bf16_tail_inline1813__ssa_v0_2, valid_rows_inline1919__ssa_v0, 64)
            kv_view_inline1915__store_2: pl.Tensor[[t_dim_inline1873__ssa_v0, 512], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                kv_normed_valid_inline1864__ssa_v0_2, [out_tg_inline1923__ssa_v0, 384], kv_view_inline1915__ssa_v0
            )
            gamma_rope_input_tail_inline1889__ssa_v0: pl.Tile[[64], pl.BF16, pl.MemRef(mem_vec_18, pl.const(16384, pl.INT64), 128), pl.Mem.Vec] = pl.tile.load(
                gamma_ckv__ssa_v0, [448], [64], [64], target_memory=pl.Mem.Vec
            )
            gamma_rope_cast_tail_inline1812__ssa_v0: pl.Tile[[64], pl.FP32, pl.MemRef(mem_vec_8, pl.const(57344, pl.INT64), 256), pl.Mem.Vec] = pl.tile.cast(
                gamma_rope_input_tail_inline1889__ssa_v0, target_type=pl.FP32, mode="round"
            )
            gamma_rope_tail_inline1811__ssa_v0: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_8, pl.const(57344, pl.INT64), 256), pl.Mem.Vec] = gamma_rope_cast_tail_inline1812__ssa_v0
            kv_rope_chunk_tail_inline1897__ssa_v0: pl.Tile[
                [32, 64], pl.FP32, pl.MemRef(mem_vec_18, pl.const(16384, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1919__ssa_v0, 64])
            ] = pl.tile.load(kv_fp32_inline1850__phi_v18, [tg_inline1881__ssa_v0, 448], [32, 64], [valid_rows_inline1919__ssa_v0, 64], target_memory=pl.Mem.Vec)
            t__tmp_v95: pl.Tile[[32, 64], pl.BF16, pl.MemRef(mem_vec_6, pl.const(32768, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1919__ssa_v0, 64])] = pl.tile.cast(
                kv_rope_chunk_tail_inline1897__ssa_v0, target_type=pl.BF16, mode="rint"
            )
            kv_rope_chunk_tail_v1_inline1823__ssa_v0: pl.Tile[
                [32, 64], pl.FP32, pl.MemRef(mem_vec_18, pl.const(16384, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1919__ssa_v0, 64])
            ] = pl.tile.cast(t__tmp_v95, target_type=pl.FP32, mode="round")
            t__tmp_v96: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_18, pl.const(16384, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1919__ssa_v0, 64])] = (
                pl.tile.row_expand_mul(kv_rope_chunk_tail_v1_inline1823__ssa_v0, kv_inv_rms_t_tail_inline1816__ssa_v0)
            )
            kv_rope_norm_tail_inline1810__ssa_v0: pl.Tile[
                [32, 64], pl.FP32, pl.MemRef(mem_vec_18, pl.const(16384, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1919__ssa_v0, 64])
            ] = pl.tile.col_expand_mul(t__tmp_v96, gamma_rope_tail_inline1811__ssa_v0)
            t__tmp_v97: pl.Tile[[32, 64], pl.BF16, pl.MemRef(mem_vec_6, pl.const(32768, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1919__ssa_v0, 64])] = pl.tile.cast(
                kv_rope_norm_tail_inline1810__ssa_v0, target_type=pl.BF16, mode="rint"
            )
            kv_rope_norm_tail_v1_inline1835__ssa_v0: pl.Tile[
                [32, 64], pl.FP32, pl.MemRef(mem_vec_18, pl.const(16384, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1919__ssa_v0, 64])
            ] = pl.tile.cast(t__tmp_v97, target_type=pl.FP32, mode="round")
            kv_cos_il_tail_inline1870__ssa_v0: pl.Tile[
                [32, 64], pl.FP32, pl.MemRef(mem_vec_6, pl.const(32768, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1919__ssa_v0, 64])
            ] = pl.tile.load(freqs_cos__ssa_v0, [out_tg_inline1923__ssa_v0, 0], [32, 64], [valid_rows_inline1919__ssa_v0, 64], target_memory=pl.Mem.Vec)
            kv_sin_signed_tail_inline1832__ssa_v0: pl.Tile[
                [32, 64], pl.FP32, pl.MemRef(mem_vec_8, pl.const(57344, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1919__ssa_v0, 64])
            ] = pl.tile.load(q_rope_sin_signed_inline197__ssa_v0, [out_tg_inline1923__ssa_v0, 0], [32, 64], [valid_rows_inline1919__ssa_v0, 64], target_memory=pl.Mem.Vec)
            t__tmp_v98: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_62, pl.const(40960, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.full([32, 64], dtype=pl.FP32, value=1.0)
            t__ci_tmp_v0: pl.Tile[[1, 192], pl.FP32, pl.MemRef(mem_vec_63, pl.const(49152, pl.INT64), 768), pl.Mem.Vec] = pl.tile.create([1, 192], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            t__tmp_v99: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_144, pl.const(0, pl.INT64), 256), pl.Mem.Vec] = pl.tile.ci(
                pl.const(0, pl.INT32), [1, 64], tmp=t__ci_tmp_v0, dtype=pl.INT32, descending=False
            )
            t__tmp_v100: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_63, pl.const(49152, pl.INT64), 256), pl.Mem.Vec] = pl.tile.cast(t__tmp_v99, target_type=pl.FP32, mode="round")
            kv_col_inline1831__ssa_v0: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_62, pl.const(40960, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.col_expand_mul(t__tmp_v98, t__tmp_v100)
            t__tmp_v101: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_63, pl.const(49152, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.muls(kv_col_inline1831__ssa_v0, 0.5)
            t__tmp_v102: pl.Tile[[32, 64], pl.INT32, pl.MemRef(mem_vec_63, pl.const(49152, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.cast(t__tmp_v101, target_type=pl.INT32, mode="trunc")
            kv_dup_f_inline1834__ssa_v0: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_63, pl.const(49152, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.cast(t__tmp_v102, target_type=pl.FP32, mode="round")
            t__tmp_v103: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_63, pl.const(49152, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.muls(kv_dup_f_inline1834__ssa_v0, 2.0)
            kv_lane_inline1809__ssa_v0: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_63, pl.const(49152, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.sub(kv_col_inline1831__ssa_v0, t__tmp_v103)
            t__tmp_v104: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_62, pl.const(40960, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.adds(kv_col_inline1831__ssa_v0, 1.0)
            t__tmp_v105: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_63, pl.const(49152, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.muls(kv_lane_inline1809__ssa_v0, 2.0)
            kv_swap_f_inline1826__ssa_v0: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_62, pl.const(40960, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.sub(t__tmp_v104, t__tmp_v105)
            t__ci_tmp_v1: pl.Tile[[1, 192], pl.FP32, pl.MemRef(mem_vec_63, pl.const(49152, pl.INT64), 768), pl.Mem.Vec] = pl.tile.create([1, 192], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            t__tmp_v106: pl.Tile[[1, 32], pl.INT32, pl.MemRef(mem_vec_144, pl.const(0, pl.INT64), 128), pl.Mem.Vec] = pl.tile.ci(
                pl.const(0, pl.INT32), [1, 32], tmp=t__ci_tmp_v1, dtype=pl.INT32, descending=False
            )
            t__tmp_v107: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_63, pl.const(49152, pl.INT64), 128), pl.Mem.Vec] = pl.tile.cast(t__tmp_v106, target_type=pl.FP32, mode="round")
            kv_row_seed_inline1920__ssa_v0: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_144, pl.const(0, pl.INT64), 128), pl.Mem.Vec] = pl.tile.muls(t__tmp_v107, 64.0)
            t__tmp_v108: pl.Tile[[64, 32], pl.FP32, pl.MemRef(mem_vec_63, pl.const(49152, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.full([64, 32], dtype=pl.FP32, value=1.0)
            kv_row_grid_inline1808__ssa_v0: pl.Tile[[64, 32], pl.FP32, pl.MemRef(mem_vec_63, pl.const(49152, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.col_expand_mul(
                t__tmp_v108, kv_row_seed_inline1920__ssa_v0
            )
            transpose_tmp: pl.Tile[[64, 32], pl.FP32, pl.MemRef(mem_vec_144, pl.const(0, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.create([64, 32], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            kv_row_offset_inline1849__ssa_v0: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_145, pl.const(8192, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.transpose(
                kv_row_grid_inline1808__ssa_v0, 0, 1, transpose_tmp
            )
            t__tmp_v109: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_62, pl.const(40960, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.add(kv_swap_f_inline1826__ssa_v0, kv_row_offset_inline1849__ssa_v0)
            kv_swap_idx_tail_inline1807__ssa_v0: pl.Tile[[32, 64], pl.INT32, pl.MemRef(mem_vec_62, pl.const(40960, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.cast(
                t__tmp_v109, target_type=pl.INT32, mode="round"
            )
            kv_gather_tmp_inline1829__ssa_v0: pl.Tile[[32, 64], pl.INT32, pl.MemRef(mem_vec_63, pl.const(49152, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.create(
                [32, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec
            )
            kv_swapped_tail_inline1806__ssa_v0: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_144, pl.const(0, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.gather(
                kv_rope_norm_tail_v1_inline1835__ssa_v0, kv_swap_idx_tail_inline1807__ssa_v0, kv_gather_tmp_inline1829__ssa_v0
            )
            t__tmp_v110: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_18, pl.const(16384, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1919__ssa_v0, 64])] = pl.tile.mul(
                kv_rope_norm_tail_v1_inline1835__ssa_v0, kv_cos_il_tail_inline1870__ssa_v0
            )
            t__tmp_v111: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_6, pl.const(32768, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.mul(
                kv_swapped_tail_inline1806__ssa_v0, kv_sin_signed_tail_inline1832__ssa_v0
            )
            kv_rope_rot_tail_inline1805__ssa_v0: pl.Tile[
                [32, 64], pl.FP32, pl.MemRef(mem_vec_18, pl.const(16384, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1919__ssa_v0, 64])
            ] = pl.tile.add(t__tmp_v110, t__tmp_v111)
            kv_rope_i16_tail_inline1804__ssa_v0: pl.Tile[
                [32, 64], pl.BF16, pl.MemRef(mem_vec_18, pl.const(16384, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1919__ssa_v0, 64])
            ] = pl.tile.cast(kv_rope_rot_tail_inline1805__ssa_v0, target_type=pl.BF16, mode="rint")
            kv_rope_valid_inline1863__ssa_v0: pl.Tile[
                [32, 64], pl.BF16, pl.MemRef(mem_vec_18, pl.const(16384, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1919__ssa_v0, 64])
            ] = pl.tile.set_validshape(kv_rope_i16_tail_inline1804__ssa_v0, valid_rows_inline1919__ssa_v0, 64)
            kv_view_inline1915__store_v0: pl.Tensor[[t_dim_inline1873__ssa_v0, 512], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                kv_rope_valid_inline1863__ssa_v0, [out_tg_inline1923__ssa_v0, 448], kv_view_inline1915__ssa_v0
            )
        return kv_view_inline1915__ssa_v0

    @pl.function(type=pl.FunctionType.Spmd)
    def kv_rms_norm_rope_spmd(
        self,
        tile_rows_inline1877__ssa_v0: pl.Scalar[pl.INDEX],
        tile_base_inline1887__idx_v0: pl.Scalar[pl.INDEX],
        kv_fp32_inline1850__phi_v18: pl.Tensor[[t_matmul_inline1893__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)],
        kv_view_inline1915__ssa_v0: pl.InOut[pl.Tensor[[t_dim_inline1873__ssa_v0, 512], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)]],
        gamma_ckv__ssa_v0: pl.Tensor[[512], pl.BF16, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 1024)],
        freqs_cos__ssa_v0: pl.Tensor[[T_DYN, 64], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)],
        q_rope_sin_signed_inline197__ssa_v0: pl.Tensor[[t_dim_inline198__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 0)],
        q_rope_swap_idx_inline196__ssa_v0: pl.Tensor[[t_dim_inline198__ssa_v0, 64], pl.INT32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 0)],
    ):
        pl.func_attr({"mx_tensor_views_blocked": True})
        kv_view_inline1915__ssa_v1: pl.Tensor[[t_dim_inline1873__ssa_v0, 512], pl.BF16, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 0)] = self.kv_rms_norm_rope(
            tile_rows_inline1877__ssa_v0,
            tile_base_inline1887__idx_v0,
            kv_fp32_inline1850__phi_v18,
            kv_view_inline1915__ssa_v0,
            gamma_ckv__ssa_v0,
            freqs_cos__ssa_v0,
            q_rope_sin_signed_inline197__ssa_v0,
            q_rope_swap_idx_inline196__ssa_v0,
            attrs={"arg_directions": [pl.adir.scalar, pl.adir.scalar, pl.adir.input, pl.adir.inout, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input]},
        )

    @pl.function(type=pl.FunctionType.AIC, level=pl.Level.AIC, role=pl.Role.SubWorker)
    def kv_score_proj(
        cmp4_kv_proj_pad_inline485_inline1955__ssa_v0: pl.Out[pl.Tensor[[384, 1024], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 1572864)]],
        cmp4_score_proj_pad_inline488_inline1934__ssa_v0: pl.Out[pl.Tensor[[384, 1024], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 1572864)]],
        t_matmul_inline483_inline1947__ssa_v0: pl.Scalar[pl.INDEX],
        bs_inline481_inline1944__ssa_v0: pl.Scalar[pl.INDEX],
        x_flat_inline484_inline1950__ssa_v0: pl.Tensor[[T_DYN, 4096], pl.BF16, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)],
        cmp_wkv__ssa_v0: pl.Tensor[[1024, 4096], pl.BF16, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 8388608)],
        cmp_wgate__ssa_v0: pl.Tensor[[1024, 4096], pl.BF16, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 8388608)],
    ):
        pl.func_attr({"mx_tensor_views_blocked": True})
        mem_vec_5: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 8192)
        mem_acc_7: pl.Ptr = pl.tile.alloc(pl.Mem.Acc, 8192)
        mem_acc_8: pl.Ptr = pl.tile.alloc(pl.Mem.Acc, 8192)
        mem_mat_9: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 16384)
        mem_mat_10: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 8192)
        mem_mat_11: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 8192)
        mem_mat_12: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 16384)
        mem_mat_13: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 8192)
        mem_mat_14: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 8192)
        mem_left_15: pl.Ptr = pl.tile.alloc(pl.Mem.Left, 16384)
        mem_right_16: pl.Ptr = pl.tile.alloc(pl.Mem.Right, 8192)
        mem_left_25: pl.Ptr = pl.tile.alloc(pl.Mem.Left, 16384)
        mem_right_26: pl.Ptr = pl.tile.alloc(pl.Mem.Right, 8192)
        kv_worker_inline491_inline1945__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        for idx_inline490_inline1952__idx_v0, (cmp4_kv_proj_pad_inline485_inline1955__iter_v1, cmp4_score_proj_pad_inline488_inline1934__iter_v1) in pl.range(
            kv_worker_inline491_inline1945__ssa_v0,
            t_matmul_inline483_inline1947__ssa_v0 // 2,
            24,
            init_values=(cmp4_kv_proj_pad_inline485_inline1955__ssa_v0, cmp4_score_proj_pad_inline488_inline1934__ssa_v0),
        ):
            global_row0_inline493_inline1949__ssa_v0: pl.Scalar[pl.INDEX] = idx_inline490_inline1952__idx_v0 // 32 * 64
            o0_inline482_inline1937__ssa_v0: pl.Scalar[pl.INDEX] = idx_inline490_inline1952__idx_v0 % 32 * 32
            x_rows_inline495_inline1954__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(bs_inline481_inline1944__ssa_v0 - global_row0_inline493_inline1949__ssa_v0, 64)
            kv_acc_inline487_inline1962__tile: pl.Tile[[64, 32], pl.FP32, pl.MemRef(mem_vec_5, pl.const(0, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.create(
                [64, 32], dtype=pl.FP32, target_memory=pl.Mem.Vec
            )
            score_acc_inline494_inline1972__tile: pl.Tile[[64, 32], pl.FP32, pl.MemRef(mem_vec_5, pl.const(0, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.create(
                [64, 32], dtype=pl.FP32, target_memory=pl.Mem.Vec
            )
            kv_acc_inline487_inline1962__tile_narrowed_storage: pl.Tile[
                [64, 32], pl.FP32, pl.MemRef(mem_acc_7, pl.const(0, pl.INT64), 8192), pl.Mem.Acc, pl.TileView(compact=pl.CompactMode.normal)
            ] = pl.tile.create([64, 32], dtype=pl.FP32, target_memory=pl.Mem.Acc, compact=True)
            kv_acc_inline487_inline1962__tile_narrowed: pl.Tile[
                [64, 32], pl.FP32, pl.MemRef(mem_acc_7, pl.const(0, pl.INT64), 8192), pl.Mem.Acc, pl.TileView(valid_shape=[x_rows_inline495_inline1954__ssa_v0, 32], compact=pl.CompactMode.normal)
            ] = pl.tile.set_validshape(kv_acc_inline487_inline1962__tile_narrowed_storage, x_rows_inline495_inline1954__ssa_v0, 32)
            score_acc_inline494_inline1972__tile_narrowed_storage: pl.Tile[
                [64, 32], pl.FP32, pl.MemRef(mem_acc_8, pl.const(8192, pl.INT64), 8192), pl.Mem.Acc, pl.TileView(compact=pl.CompactMode.normal)
            ] = pl.tile.create([64, 32], dtype=pl.FP32, target_memory=pl.Mem.Acc, compact=True)
            score_acc_inline494_inline1972__tile_narrowed: pl.Tile[
                [64, 32], pl.FP32, pl.MemRef(mem_acc_8, pl.const(8192, pl.INT64), 8192), pl.Mem.Acc, pl.TileView(valid_shape=[x_rows_inline495_inline1954__ssa_v0, 32], compact=pl.CompactMode.normal)
            ] = pl.tile.set_validshape(score_acc_inline494_inline1972__tile_narrowed_storage, x_rows_inline495_inline1954__ssa_v0, 32)
            for kb_inline480_inline1982__idx_v0, (kv_acc_inline487_inline1962__iter_v1, score_acc_inline494_inline1972__iter_v1) in pl.range(
                0, 32, 2, init_values=(kv_acc_inline487_inline1962__tile_narrowed, score_acc_inline494_inline1972__tile_narrowed)
            ):
                k0_inline479_inline1941__ssa_v0: pl.Scalar[pl.INDEX] = (kb_inline480_inline1982__idx_v0 * 128 + o0_inline482_inline1937__ssa_v0 % 512 // 32 * 256) % 4096
                k0_inline479_inline1941__ssa_v0_1: pl.Scalar[pl.INDEX] = (kb_inline480_inline1982__idx_v0 * 128 + o0_inline482_inline1937__ssa_v0 % 512 // 32 * 256 + 128) % 4096
                x_tile_inline492_inline1936__tile: pl.Tile[
                    [64, 128], pl.BF16, pl.MemRef(mem_mat_9, pl.const(49152, pl.INT64), 16384), pl.Mem.Mat, pl.TileView(valid_shape=[x_rows_inline495_inline1954__ssa_v0, 128])
                ] = pl.tile.load(
                    x_flat_inline484_inline1950__ssa_v0,
                    [global_row0_inline493_inline1949__ssa_v0, k0_inline479_inline1941__ssa_v0],
                    [64, 128],
                    [x_rows_inline495_inline1954__ssa_v0, 128],
                    target_memory=pl.Mem.Mat,
                )
                wkv_tile_inline478_inline1946__tile: pl.Tile[[32, 128], pl.BF16, pl.MemRef(mem_mat_10, pl.const(0, pl.INT64), 8192), pl.Mem.Mat] = pl.tile.load(
                    cmp_wkv__ssa_v0, [o0_inline482_inline1937__ssa_v0, k0_inline479_inline1941__ssa_v0], [32, 128], [32, 128], target_memory=pl.Mem.Mat
                )
                wgate_tile_inline486_inline1943__tile: pl.Tile[[32, 128], pl.BF16, pl.MemRef(mem_mat_11, pl.const(8192, pl.INT64), 8192), pl.Mem.Mat] = pl.tile.load(
                    cmp_wgate__ssa_v0, [o0_inline482_inline1937__ssa_v0, k0_inline479_inline1941__ssa_v0], [32, 128], [32, 128], target_memory=pl.Mem.Mat
                )
                x_tile_inline492_inline1936__tile_1: pl.Tile[
                    [64, 128], pl.BF16, pl.MemRef(mem_mat_12, pl.const(16384, pl.INT64), 16384), pl.Mem.Mat, pl.TileView(valid_shape=[x_rows_inline495_inline1954__ssa_v0, 128])
                ] = pl.tile.load(
                    x_flat_inline484_inline1950__ssa_v0,
                    [global_row0_inline493_inline1949__ssa_v0, k0_inline479_inline1941__ssa_v0_1],
                    [64, 128],
                    [x_rows_inline495_inline1954__ssa_v0, 128],
                    target_memory=pl.Mem.Mat,
                )
                wkv_tile_inline478_inline1946__tile_1: pl.Tile[[32, 128], pl.BF16, pl.MemRef(mem_mat_13, pl.const(32768, pl.INT64), 8192), pl.Mem.Mat] = pl.tile.load(
                    cmp_wkv__ssa_v0, [o0_inline482_inline1937__ssa_v0, k0_inline479_inline1941__ssa_v0_1], [32, 128], [32, 128], target_memory=pl.Mem.Mat
                )
                wgate_tile_inline486_inline1943__tile_1: pl.Tile[[32, 128], pl.BF16, pl.MemRef(mem_mat_14, pl.const(40960, pl.INT64), 8192), pl.Mem.Mat] = pl.tile.load(
                    cmp_wgate__ssa_v0, [o0_inline482_inline1937__ssa_v0, k0_inline479_inline1941__ssa_v0_1], [32, 128], [32, 128], target_memory=pl.Mem.Mat
                )
                if kb_inline480_inline1982__idx_v0 == 0:
                    wkv_tile_inline478_inline1946__tile_t: pl.Tile[
                        [128, 32], pl.BF16, pl.MemRef(mem_mat_10, pl.const(0, pl.INT64), 8192), pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)
                    ] = pl.tile.transpose_view(wkv_tile_inline478_inline1946__tile)
                    x_tile_inline492_inline1936__tile_Left: pl.Tile[
                        [64, 128], pl.BF16, pl.MemRef(mem_left_15, pl.const(0, pl.INT64), 16384), pl.Mem.Left, pl.TileView(valid_shape=[x_rows_inline495_inline1954__ssa_v0, 128])
                    ] = pl.tile.move(x_tile_inline492_inline1936__tile, target_memory=pl.Mem.Left)
                    wkv_tile_inline478_inline1946__tile_t_Right: pl.Tile[[128, 32], pl.BF16, pl.MemRef(mem_right_16, pl.const(0, pl.INT64), 8192), pl.Mem.Right] = pl.tile.move(
                        wkv_tile_inline478_inline1946__tile_t, target_memory=pl.Mem.Right
                    )
                    kv_acc_inline487_inline1962__tile_1: pl.Tile[
                        [64, 32],
                        pl.FP32,
                        pl.MemRef(mem_acc_7, pl.const(0, pl.INT64), 8192),
                        pl.Mem.Acc,
                        pl.TileView(valid_shape=[x_rows_inline495_inline1954__ssa_v0, 32], compact=pl.CompactMode.normal),
                    ] = pl.tile.matmul(x_tile_inline492_inline1936__tile_Left, wkv_tile_inline478_inline1946__tile_t_Right)
                    wgate_tile_inline486_inline1943__tile_t: pl.Tile[
                        [128, 32], pl.BF16, pl.MemRef(mem_mat_11, pl.const(8192, pl.INT64), 8192), pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)
                    ] = pl.tile.transpose_view(wgate_tile_inline486_inline1943__tile)
                    wgate_tile_inline486_inline1943__tile_t_Right: pl.Tile[[128, 32], pl.BF16, pl.MemRef(mem_right_16, pl.const(0, pl.INT64), 8192), pl.Mem.Right] = pl.tile.move(
                        wgate_tile_inline486_inline1943__tile_t, target_memory=pl.Mem.Right
                    )
                    score_acc_inline494_inline1972__tile_1: pl.Tile[
                        [64, 32],
                        pl.FP32,
                        pl.MemRef(mem_acc_8, pl.const(8192, pl.INT64), 8192),
                        pl.Mem.Acc,
                        pl.TileView(valid_shape=[x_rows_inline495_inline1954__ssa_v0, 32], compact=pl.CompactMode.normal),
                    ] = pl.tile.matmul(x_tile_inline492_inline1936__tile_Left, wgate_tile_inline486_inline1943__tile_t_Right)
                    kv_acc_inline487_inline1962__phi_v5, score_acc_inline494_inline1972__phi_v5 = pl.yield_(kv_acc_inline487_inline1962__tile_1, score_acc_inline494_inline1972__tile_1)
                else:
                    wkv_tile_inline478_inline1946__tile_t_1: pl.Tile[
                        [128, 32], pl.BF16, pl.MemRef(mem_mat_10, pl.const(0, pl.INT64), 8192), pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)
                    ] = pl.tile.transpose_view(wkv_tile_inline478_inline1946__tile)
                    x_tile_inline492_inline1936__tile_Left_1: pl.Tile[
                        [64, 128], pl.BF16, pl.MemRef(mem_left_15, pl.const(0, pl.INT64), 16384), pl.Mem.Left, pl.TileView(valid_shape=[x_rows_inline495_inline1954__ssa_v0, 128])
                    ] = pl.tile.move(x_tile_inline492_inline1936__tile, target_memory=pl.Mem.Left)
                    wkv_tile_inline478_inline1946__tile_t_Right_1: pl.Tile[[128, 32], pl.BF16, pl.MemRef(mem_right_16, pl.const(0, pl.INT64), 8192), pl.Mem.Right] = pl.tile.move(
                        wkv_tile_inline478_inline1946__tile_t_1, target_memory=pl.Mem.Right
                    )
                    kv_acc_inline487_inline1962__tile_2: pl.Tile[
                        [64, 32],
                        pl.FP32,
                        pl.MemRef(mem_acc_7, pl.const(0, pl.INT64), 8192),
                        pl.Mem.Acc,
                        pl.TileView(valid_shape=[x_rows_inline495_inline1954__ssa_v0, 32], compact=pl.CompactMode.normal),
                    ] = pl.tile.matmul_acc(kv_acc_inline487_inline1962__iter_v1, x_tile_inline492_inline1936__tile_Left_1, wkv_tile_inline478_inline1946__tile_t_Right_1)
                    wgate_tile_inline486_inline1943__tile_t_1: pl.Tile[
                        [128, 32], pl.BF16, pl.MemRef(mem_mat_11, pl.const(8192, pl.INT64), 8192), pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)
                    ] = pl.tile.transpose_view(wgate_tile_inline486_inline1943__tile)
                    wgate_tile_inline486_inline1943__tile_t_Right_1: pl.Tile[[128, 32], pl.BF16, pl.MemRef(mem_right_16, pl.const(0, pl.INT64), 8192), pl.Mem.Right] = pl.tile.move(
                        wgate_tile_inline486_inline1943__tile_t_1, target_memory=pl.Mem.Right
                    )
                    score_acc_inline494_inline1972__tile_2: pl.Tile[
                        [64, 32],
                        pl.FP32,
                        pl.MemRef(mem_acc_8, pl.const(8192, pl.INT64), 8192),
                        pl.Mem.Acc,
                        pl.TileView(valid_shape=[x_rows_inline495_inline1954__ssa_v0, 32], compact=pl.CompactMode.normal),
                    ] = pl.tile.matmul_acc(score_acc_inline494_inline1972__iter_v1, x_tile_inline492_inline1936__tile_Left_1, wgate_tile_inline486_inline1943__tile_t_Right_1)
                    kv_acc_inline487_inline1962__phi_v5, score_acc_inline494_inline1972__phi_v5 = pl.yield_(kv_acc_inline487_inline1962__tile_2, score_acc_inline494_inline1972__tile_2)
                wkv_tile_inline478_inline1946__tile_t_2: pl.Tile[
                    [128, 32], pl.BF16, pl.MemRef(mem_mat_13, pl.const(32768, pl.INT64), 8192), pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)
                ] = pl.tile.transpose_view(wkv_tile_inline478_inline1946__tile_1)
                x_tile_inline492_inline1936__tile_Left_2: pl.Tile[
                    [64, 128], pl.BF16, pl.MemRef(mem_left_25, pl.const(16384, pl.INT64), 16384), pl.Mem.Left, pl.TileView(valid_shape=[x_rows_inline495_inline1954__ssa_v0, 128])
                ] = pl.tile.move(x_tile_inline492_inline1936__tile_1, target_memory=pl.Mem.Left)
                wkv_tile_inline478_inline1946__tile_t_Right_2: pl.Tile[[128, 32], pl.BF16, pl.MemRef(mem_right_26, pl.const(8192, pl.INT64), 8192), pl.Mem.Right] = pl.tile.move(
                    wkv_tile_inline478_inline1946__tile_t_2, target_memory=pl.Mem.Right
                )
                kv_acc_inline487_inline1962__tile_3: pl.Tile[
                    [64, 32], pl.FP32, pl.MemRef(mem_acc_7, pl.const(0, pl.INT64), 8192), pl.Mem.Acc, pl.TileView(valid_shape=[x_rows_inline495_inline1954__ssa_v0, 32], compact=pl.CompactMode.normal)
                ] = pl.tile.matmul_acc(kv_acc_inline487_inline1962__phi_v5, x_tile_inline492_inline1936__tile_Left_2, wkv_tile_inline478_inline1946__tile_t_Right_2)
                wgate_tile_inline486_inline1943__tile_t_2: pl.Tile[
                    [128, 32], pl.BF16, pl.MemRef(mem_mat_14, pl.const(40960, pl.INT64), 8192), pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)
                ] = pl.tile.transpose_view(wgate_tile_inline486_inline1943__tile_1)
                wgate_tile_inline486_inline1943__tile_t_Right_2: pl.Tile[[128, 32], pl.BF16, pl.MemRef(mem_right_26, pl.const(8192, pl.INT64), 8192), pl.Mem.Right] = pl.tile.move(
                    wgate_tile_inline486_inline1943__tile_t_2, target_memory=pl.Mem.Right
                )
                score_acc_inline494_inline1972__tile_3: pl.Tile[
                    [64, 32],
                    pl.FP32,
                    pl.MemRef(mem_acc_8, pl.const(8192, pl.INT64), 8192),
                    pl.Mem.Acc,
                    pl.TileView(valid_shape=[x_rows_inline495_inline1954__ssa_v0, 32], compact=pl.CompactMode.normal),
                ] = pl.tile.matmul_acc(score_acc_inline494_inline1972__phi_v5, x_tile_inline492_inline1936__tile_Left_2, wgate_tile_inline486_inline1943__tile_t_Right_2)
                kv_acc_inline487_inline1962__rv_v2, score_acc_inline494_inline1972__rv_v2 = pl.yield_(kv_acc_inline487_inline1962__tile_3, score_acc_inline494_inline1972__tile_3)
            cmp4_kv_proj_pad_inline485_inline1955__tile: pl.Tensor[[384, 1024], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 1572864)] = pl.tile.store(
                kv_acc_inline487_inline1962__rv_v2, [global_row0_inline493_inline1949__ssa_v0, o0_inline482_inline1937__ssa_v0], cmp4_kv_proj_pad_inline485_inline1955__iter_v1
            )
            cmp4_score_proj_pad_inline488_inline1934__tile: pl.Tensor[[384, 1024], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 1572864)] = pl.tile.store(
                score_acc_inline494_inline1972__rv_v2, [global_row0_inline493_inline1949__ssa_v0, o0_inline482_inline1937__ssa_v0], cmp4_score_proj_pad_inline488_inline1934__iter_v1
            )
            cmp4_kv_proj_pad_inline485_inline1955__rv_v2, cmp4_score_proj_pad_inline488_inline1934__rv_v2 = pl.yield_(
                cmp4_kv_proj_pad_inline485_inline1955__tile, cmp4_score_proj_pad_inline488_inline1934__tile
            )
        return

    @pl.function(type=pl.FunctionType.Spmd)
    def kv_score_proj_spmd(
        self,
        cmp4_kv_proj_pad_inline485_inline1955__ssa_v0: pl.Out[pl.Tensor[[384, 1024], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 1572864)]],
        cmp4_score_proj_pad_inline488_inline1934__ssa_v0: pl.Out[pl.Tensor[[384, 1024], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 1572864)]],
        t_matmul_inline483_inline1947__ssa_v0: pl.Scalar[pl.INDEX],
        bs_inline481_inline1944__ssa_v0: pl.Scalar[pl.INDEX],
        x_flat_inline484_inline1950__ssa_v0: pl.Tensor[[T_DYN, 4096], pl.BF16, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)],
        cmp_wkv__ssa_v0: pl.Tensor[[1024, 4096], pl.BF16, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 8388608)],
        cmp_wgate__ssa_v0: pl.Tensor[[1024, 4096], pl.BF16, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 8388608)],
    ):
        pl.func_attr({"mx_tensor_views_blocked": True})
        self.kv_score_proj(
            cmp4_kv_proj_pad_inline485_inline1955__ssa_v0,
            cmp4_score_proj_pad_inline488_inline1934__ssa_v0,
            t_matmul_inline483_inline1947__ssa_v0,
            bs_inline481_inline1944__ssa_v0,
            x_flat_inline484_inline1950__ssa_v0,
            cmp_wkv__ssa_v0,
            cmp_wgate__ssa_v0,
            attrs={"arg_directions": [pl.adir.output_existing, pl.adir.output_existing, pl.adir.scalar, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.input]},
        )

    @pl.function(type=pl.FunctionType.AIC, level=pl.Level.AIC, role=pl.Role.SubWorker)
    def kv_score_proj_0(
        kv_proj_pad_inline2105__ssa_v0: pl.Out[pl.Tensor[[384, 256], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 393216)]],
        score_proj_pad_inline2117__ssa_v0: pl.Out[pl.Tensor[[384, 256], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 393216)]],
        t_matmul_inline374_inline2127__ssa_v0: pl.Scalar[pl.INDEX],
        bs_inline373_inline2096__ssa_v0: pl.Scalar[pl.INDEX],
        x_flat_inline377_inline2152__ssa_v0: pl.Tensor[[T_DYN, 4096], pl.BF16, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)],
        inner_wkv__ssa_v0: pl.Tensor[[256, 4096], pl.BF16, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 2097152)],
        inner_wgate__ssa_v0: pl.Tensor[[256, 4096], pl.BF16, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 2097152)],
    ) -> tuple[pl.Tensor[[384, 256], pl.FP32], pl.Tensor[[384, 256], pl.FP32]]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        mem_acc_5: pl.Ptr = pl.tile.alloc(pl.Mem.Acc, 1024)
        mem_acc_6: pl.Ptr = pl.tile.alloc(pl.Mem.Acc, 1024)
        mem_mat_7: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 8192)
        mem_mat_8: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 8192)
        mem_mat_9: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 8192)
        mem_mat_10: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 8192)
        mem_mat_11: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 8192)
        mem_mat_12: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 8192)
        mem_left_13: pl.Ptr = pl.tile.alloc(pl.Mem.Left, 8192)
        mem_right_14: pl.Ptr = pl.tile.alloc(pl.Mem.Right, 8192)
        mem_left_16: pl.Ptr = pl.tile.alloc(pl.Mem.Left, 8192)
        mem_right_17: pl.Ptr = pl.tile.alloc(pl.Mem.Right, 8192)
        kv_worker_inline379_inline2125__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        for idx_inline381_inline2131__idx_v0, (kv_proj_pad_inline2105__iter_v1, score_proj_pad_inline2117__iter_v1) in pl.range(
            kv_worker_inline379_inline2125__ssa_v0, t_matmul_inline374_inline2127__ssa_v0, 24, init_values=(kv_proj_pad_inline2105__ssa_v0, score_proj_pad_inline2117__ssa_v0)
        ):
            global_row0_inline382_inline2097__ssa_v0: pl.Scalar[pl.INDEX] = idx_inline381_inline2131__idx_v0 // 16 * 16
            o0_inline384_inline2107__ssa_v0: pl.Scalar[pl.INDEX] = idx_inline381_inline2131__idx_v0 % 16 * 16
            kv_acc_inline375_inline2110__tile: pl.Tile[[16, 16], pl.FP32, pl.MemRef(mem_acc_5, pl.const(0, pl.INT64), 1024), pl.Mem.Acc] = pl.tile.create(
                [16, 16], dtype=pl.FP32, target_memory=pl.Mem.Acc
            )
            score_acc_inline376_inline2144__tile: pl.Tile[[16, 16], pl.FP32, pl.MemRef(mem_acc_6, pl.const(1024, pl.INT64), 1024), pl.Mem.Acc] = pl.tile.create(
                [16, 16], dtype=pl.FP32, target_memory=pl.Mem.Acc
            )
            for kb_inline385_inline2104__idx_v0, (kv_acc_inline375_inline2110__iter_v1, score_acc_inline376_inline2144__iter_v1) in pl.range(
                0, 16, 2, init_values=(kv_acc_inline375_inline2110__tile, score_acc_inline376_inline2144__tile)
            ):
                k0_inline378_inline2135__ssa_v0: pl.Scalar[pl.INDEX] = (kb_inline385_inline2104__idx_v0 + o0_inline384_inline2107__ssa_v0 % 128 // 16) * 256 % 4096
                x_rows_inline380_inline2140__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(bs_inline373_inline2096__ssa_v0 - global_row0_inline382_inline2097__ssa_v0, 16)
                k0_inline378_inline2135__ssa_v0_1: pl.Scalar[pl.INDEX] = ((kb_inline385_inline2104__idx_v0 + o0_inline384_inline2107__ssa_v0 % 128 // 16) * 256 + 256) % 4096
                x_rows_inline380_inline2140__ssa_v0_1: pl.Scalar[pl.INDEX] = pl.min(bs_inline373_inline2096__ssa_v0 - global_row0_inline382_inline2097__ssa_v0, 16)
                x_tile_inline372_inline2132__tile: pl.Tile[
                    [16, 256], pl.BF16, pl.MemRef(mem_mat_7, pl.const(24576, pl.INT64), 8192), pl.Mem.Mat, pl.TileView(valid_shape=[x_rows_inline380_inline2140__ssa_v0, 256])
                ] = pl.tile.load(
                    x_flat_inline377_inline2152__ssa_v0,
                    [global_row0_inline382_inline2097__ssa_v0, k0_inline378_inline2135__ssa_v0],
                    [16, 256],
                    [x_rows_inline380_inline2140__ssa_v0, 256],
                    target_memory=pl.Mem.Mat,
                )
                wkv_tile_inline371_inline2103__tile: pl.Tile[[16, 256], pl.BF16, pl.MemRef(mem_mat_8, pl.const(32768, pl.INT64), 8192), pl.Mem.Mat] = pl.tile.load(
                    inner_wkv__ssa_v0, [o0_inline384_inline2107__ssa_v0, k0_inline378_inline2135__ssa_v0], [16, 256], [16, 256], target_memory=pl.Mem.Mat
                )
                wgate_tile_inline370_inline2114__tile: pl.Tile[[16, 256], pl.BF16, pl.MemRef(mem_mat_9, pl.const(40960, pl.INT64), 8192), pl.Mem.Mat] = pl.tile.load(
                    inner_wgate__ssa_v0, [o0_inline384_inline2107__ssa_v0, k0_inline378_inline2135__ssa_v0], [16, 256], [16, 256], target_memory=pl.Mem.Mat
                )
                x_tile_inline372_inline2132__tile_1: pl.Tile[
                    [16, 256], pl.BF16, pl.MemRef(mem_mat_10, pl.const(0, pl.INT64), 8192), pl.Mem.Mat, pl.TileView(valid_shape=[x_rows_inline380_inline2140__ssa_v0_1, 256])
                ] = pl.tile.load(
                    x_flat_inline377_inline2152__ssa_v0,
                    [global_row0_inline382_inline2097__ssa_v0, k0_inline378_inline2135__ssa_v0_1],
                    [16, 256],
                    [x_rows_inline380_inline2140__ssa_v0_1, 256],
                    target_memory=pl.Mem.Mat,
                )
                wkv_tile_inline371_inline2103__tile_1: pl.Tile[[16, 256], pl.BF16, pl.MemRef(mem_mat_11, pl.const(8192, pl.INT64), 8192), pl.Mem.Mat] = pl.tile.load(
                    inner_wkv__ssa_v0, [o0_inline384_inline2107__ssa_v0, k0_inline378_inline2135__ssa_v0_1], [16, 256], [16, 256], target_memory=pl.Mem.Mat
                )
                wgate_tile_inline370_inline2114__tile_1: pl.Tile[[16, 256], pl.BF16, pl.MemRef(mem_mat_12, pl.const(16384, pl.INT64), 8192), pl.Mem.Mat] = pl.tile.load(
                    inner_wgate__ssa_v0, [o0_inline384_inline2107__ssa_v0, k0_inline378_inline2135__ssa_v0_1], [16, 256], [16, 256], target_memory=pl.Mem.Mat
                )
                wkv_tile_inline371_inline2103__tile_t: pl.Tile[
                    [256, 16], pl.BF16, pl.MemRef(mem_mat_8, pl.const(32768, pl.INT64), 8192), pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)
                ] = pl.tile.transpose_view(wkv_tile_inline371_inline2103__tile)
                x_tile_inline372_inline2132__tile_Left: pl.Tile[
                    [16, 256], pl.BF16, pl.MemRef(mem_left_13, pl.const(0, pl.INT64), 8192), pl.Mem.Left, pl.TileView(valid_shape=[x_rows_inline380_inline2140__ssa_v0, 256])
                ] = pl.tile.move(x_tile_inline372_inline2132__tile, target_memory=pl.Mem.Left)
                wkv_tile_inline371_inline2103__tile_t_Right: pl.Tile[[256, 16], pl.BF16, pl.MemRef(mem_right_14, pl.const(0, pl.INT64), 8192), pl.Mem.Right] = pl.tile.move(
                    wkv_tile_inline371_inline2103__tile_t, target_memory=pl.Mem.Right
                )
                kv_acc_inline375_inline2110__tile_1: pl.Tile[[16, 16], pl.FP32, pl.MemRef(mem_acc_5, pl.const(0, pl.INT64), 1024), pl.Mem.Acc] = pl.tile.matmul_acc(
                    kv_acc_inline375_inline2110__iter_v1, x_tile_inline372_inline2132__tile_Left, wkv_tile_inline371_inline2103__tile_t_Right, kb_inline385_inline2104__idx_v0 == 0
                )
                wgate_tile_inline370_inline2114__tile_t: pl.Tile[
                    [256, 16], pl.BF16, pl.MemRef(mem_mat_9, pl.const(40960, pl.INT64), 8192), pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)
                ] = pl.tile.transpose_view(wgate_tile_inline370_inline2114__tile)
                wgate_tile_inline370_inline2114__tile_t_Right: pl.Tile[[256, 16], pl.BF16, pl.MemRef(mem_right_14, pl.const(0, pl.INT64), 8192), pl.Mem.Right] = pl.tile.move(
                    wgate_tile_inline370_inline2114__tile_t, target_memory=pl.Mem.Right
                )
                score_acc_inline376_inline2144__tile_1: pl.Tile[[16, 16], pl.FP32, pl.MemRef(mem_acc_6, pl.const(1024, pl.INT64), 1024), pl.Mem.Acc] = pl.tile.matmul_acc(
                    score_acc_inline376_inline2144__iter_v1, x_tile_inline372_inline2132__tile_Left, wgate_tile_inline370_inline2114__tile_t_Right, kb_inline385_inline2104__idx_v0 == 0
                )
                wkv_tile_inline371_inline2103__tile_t_1: pl.Tile[
                    [256, 16], pl.BF16, pl.MemRef(mem_mat_11, pl.const(8192, pl.INT64), 8192), pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)
                ] = pl.tile.transpose_view(wkv_tile_inline371_inline2103__tile_1)
                x_tile_inline372_inline2132__tile_Left_1: pl.Tile[
                    [16, 256], pl.BF16, pl.MemRef(mem_left_16, pl.const(8192, pl.INT64), 8192), pl.Mem.Left, pl.TileView(valid_shape=[x_rows_inline380_inline2140__ssa_v0_1, 256])
                ] = pl.tile.move(x_tile_inline372_inline2132__tile_1, target_memory=pl.Mem.Left)
                wkv_tile_inline371_inline2103__tile_t_Right_1: pl.Tile[[256, 16], pl.BF16, pl.MemRef(mem_right_17, pl.const(8192, pl.INT64), 8192), pl.Mem.Right] = pl.tile.move(
                    wkv_tile_inline371_inline2103__tile_t_1, target_memory=pl.Mem.Right
                )
                kv_acc_inline375_inline2110__tile_2: pl.Tile[[16, 16], pl.FP32, pl.MemRef(mem_acc_5, pl.const(0, pl.INT64), 1024), pl.Mem.Acc] = pl.tile.matmul_acc(
                    kv_acc_inline375_inline2110__tile_1, x_tile_inline372_inline2132__tile_Left_1, wkv_tile_inline371_inline2103__tile_t_Right_1, kb_inline385_inline2104__idx_v0 == -1
                )
                wgate_tile_inline370_inline2114__tile_t_1: pl.Tile[
                    [256, 16], pl.BF16, pl.MemRef(mem_mat_12, pl.const(16384, pl.INT64), 8192), pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)
                ] = pl.tile.transpose_view(wgate_tile_inline370_inline2114__tile_1)
                wgate_tile_inline370_inline2114__tile_t_Right_1: pl.Tile[[256, 16], pl.BF16, pl.MemRef(mem_right_17, pl.const(8192, pl.INT64), 8192), pl.Mem.Right] = pl.tile.move(
                    wgate_tile_inline370_inline2114__tile_t_1, target_memory=pl.Mem.Right
                )
                score_acc_inline376_inline2144__tile_2: pl.Tile[[16, 16], pl.FP32, pl.MemRef(mem_acc_6, pl.const(1024, pl.INT64), 1024), pl.Mem.Acc] = pl.tile.matmul_acc(
                    score_acc_inline376_inline2144__tile_1, x_tile_inline372_inline2132__tile_Left_1, wgate_tile_inline370_inline2114__tile_t_Right_1, kb_inline385_inline2104__idx_v0 == -1
                )
                kv_acc_inline375_inline2110__rv_v2, score_acc_inline376_inline2144__rv_v2 = pl.yield_(kv_acc_inline375_inline2110__tile_2, score_acc_inline376_inline2144__tile_2)
            kv_proj_pad_inline2105__tile: pl.Tensor[[384, 256], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 393216)] = pl.tile.store(
                kv_acc_inline375_inline2110__rv_v2, [global_row0_inline382_inline2097__ssa_v0, o0_inline384_inline2107__ssa_v0], kv_proj_pad_inline2105__iter_v1
            )
            score_proj_pad_inline2117__tile: pl.Tensor[[384, 256], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 393216)] = pl.tile.store(
                score_acc_inline376_inline2144__rv_v2, [global_row0_inline382_inline2097__ssa_v0, o0_inline384_inline2107__ssa_v0], score_proj_pad_inline2117__iter_v1
            )
            kv_proj_pad_inline2105__rv_v2, score_proj_pad_inline2117__rv_v2 = pl.yield_(kv_proj_pad_inline2105__tile, score_proj_pad_inline2117__tile)
        return kv_proj_pad_inline2105__ssa_v0, score_proj_pad_inline2117__ssa_v0

    @pl.function(type=pl.FunctionType.Spmd)
    def kv_score_proj_spmd_0(
        self,
        kv_proj_pad_inline2105__ssa_v0: pl.Out[pl.Tensor[[384, 256], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 393216)]],
        score_proj_pad_inline2117__ssa_v0: pl.Out[pl.Tensor[[384, 256], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 393216)]],
        t_matmul_inline374_inline2127__ssa_v0: pl.Scalar[pl.INDEX],
        bs_inline373_inline2096__ssa_v0: pl.Scalar[pl.INDEX],
        x_flat_inline377_inline2152__ssa_v0: pl.Tensor[[T_DYN, 4096], pl.BF16, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)],
        inner_wkv__ssa_v0: pl.Tensor[[256, 4096], pl.BF16, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 2097152)],
        inner_wgate__ssa_v0: pl.Tensor[[256, 4096], pl.BF16, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 2097152)],
    ) -> tuple[pl.Tensor[[384, 256], pl.FP32], pl.Tensor[[384, 256], pl.FP32]]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        ret__tmp_v0: pl.Tuple[pl.Tensor[[384, 256], pl.FP32], pl.Tensor[[384, 256], pl.FP32]] = self.kv_score_proj_0(
            kv_proj_pad_inline2105__ssa_v0,
            score_proj_pad_inline2117__ssa_v0,
            t_matmul_inline374_inline2127__ssa_v0,
            bs_inline373_inline2096__ssa_v0,
            x_flat_inline377_inline2152__ssa_v0,
            inner_wkv__ssa_v0,
            inner_wgate__ssa_v0,
            attrs={"arg_directions": [pl.adir.output_existing, pl.adir.output_existing, pl.adir.scalar, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.input]},
        )
        kv_proj_pad_inline2105__rv_v2: pl.Tensor[[384, 256], pl.FP32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 393216)] = ret__tmp_v0[0]
        score_proj_pad_inline2117__rv_v2: pl.Tensor[[384, 256], pl.FP32, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 393216)] = ret__tmp_v0[1]
        return kv_proj_pad_inline2105__ssa_v0, score_proj_pad_inline2117__ssa_v0

    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def kv_touch(
        ori_kv_flat_inline2493__ssa_v0: pl.InOut[pl.Tensor[[ori_block_num_inline2459__ssa_v0 * pl.const(32, pl.INDEX), 512], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)]],
    ) -> pl.Tensor[[ori_block_num_inline2459__ssa_v0 * pl.const(32, pl.INDEX), 512], pl.BF16]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        mem_vec_1: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 1024)
        t__tile: pl.Tile[[1, 512], pl.BF16, pl.MemRef(mem_vec_1, pl.const(0, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.load(
            ori_kv_flat_inline2493__ssa_v0, [0, 0], [1, 512], [1, 512], target_memory=pl.Mem.Vec
        )
        ori_kv_flat_inline2493__tile: pl.Tensor[[ori_block_num_inline2459__ssa_v0 * pl.const(32, pl.INDEX), 512], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)] = pl.tile.store(
            t__tile, [0, 0], ori_kv_flat_inline2493__ssa_v0
        )
        return ori_kv_flat_inline2493__ssa_v0

    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def merge_norm(
        rope_swap_idx_inline244__ssa_v0: pl.Tensor[[16, 64], pl.INT32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 4096)],
        t_dim_inline235__ssa_v0: pl.Scalar[pl.INDEX],
        attn_li_inline243__ssa_v0: pl.Tensor[[t_heads_inline2469__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)],
        attn_oi_inline260__ssa_v0: pl.Tensor[[t_heads_inline2469__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)],
        rope_cos_il_inline238__ssa_v0: pl.Tensor[[T_DYN, 64], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)],
        rope_sin_signed_inline242__ssa_v0: pl.Tensor[[384, 64], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 98304)],
        o_packed_heads__ssa_v0: pl.Out[pl.Tensor[[3072, 4096], pl.BF16, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 25165824)]],
    ) -> pl.Tensor[[3072, 4096], pl.BF16]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        mem_vec_14: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 4096)
        mem_vec_15: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 4096)
        mem_vec_17: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32768)
        mem_vec_19: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
        mem_vec_21: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        mem_vec_22: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        mem_vec_23: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 4096)
        mem_vec_24: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 4096)
        m_worker_inline240__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        m_swap_inline241__ssa_v0: pl.Tile[[16, 64], pl.INT32, pl.MemRef(mem_vec_17, pl.const(8192, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.load(
            rope_swap_idx_inline244__ssa_v0, [0, 0], [16, 64], [16, 64], target_memory=pl.Mem.Vec
        )
        m_swap_f_inline245__ssa_v0: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_17, pl.const(8192, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(
            m_swap_inline241__ssa_v0, target_type=pl.FP32, mode="round"
        )
        m_swap_source_inline239__ssa_v0: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_17, pl.const(8192, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.adds(m_swap_f_inline245__ssa_v0, 448.0)
        m_row_ids_inline252__ci_tmp_v0: pl.Tile[[1, 192], pl.FP32, pl.MemRef(mem_vec_19, pl.const(40960, pl.INT64), 768), pl.Mem.Vec] = pl.tile.create(
            [1, 192], dtype=pl.FP32, target_memory=pl.Mem.Vec
        )
        m_row_ids_inline252__ssa_v0: pl.Tile[[1, 16], pl.INT32, pl.MemRef(mem_vec_14, pl.const(0, pl.INT64), 64), pl.Mem.Vec] = pl.tile.ci(
            pl.const(0, pl.INT32), [1, 16], tmp=m_row_ids_inline252__ci_tmp_v0, dtype=pl.INT32, descending=False
        )
        m_row_ids_f_inline234__ssa_v0: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_19, pl.const(40960, pl.INT64), 64), pl.Mem.Vec] = pl.tile.cast(
            m_row_ids_inline252__ssa_v0, target_type=pl.FP32, mode="round"
        )
        m_row_offsets_inline246__ssa_v0: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_19, pl.const(40960, pl.INT64), 64), pl.Mem.Vec] = pl.tile.muls(m_row_ids_f_inline234__ssa_v0, 512.0)
        m_row_offsets_col_inline247__ssa_v0: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_19, pl.const(40960, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(m_row_offsets_inline246__ssa_v0, [16, 1])
        m_swap_flat_inline250__ssa_v0: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_17, pl.const(8192, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.row_expand_add(
            m_swap_source_inline239__ssa_v0, m_row_offsets_col_inline247__ssa_v0
        )
        m_swap_idx_inline253__ssa_v0: pl.Tile[[16, 64], pl.INT32, pl.MemRef(mem_vec_14, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(
            m_swap_flat_inline250__ssa_v0, target_type=pl.INT32, mode="round"
        )
        m_gather_tmp_inline256__ssa_v0: pl.Tile[[16, 64], pl.INT32, pl.MemRef(mem_vec_15, pl.const(4096, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.create(
            [16, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec
        )
        for m_idx_inline259__idx_v0 in pl.range(m_worker_inline240__ssa_v0, t_dim_inline235__ssa_v0 * 4, 48):
            m_t_inline261__ssa_v0: pl.Scalar[pl.INDEX] = m_idx_inline259__idx_v0 // 4
            m_h_idx_inline263__ssa_v0: pl.Scalar[pl.INDEX] = m_idx_inline259__idx_v0 - m_t_inline261__ssa_v0 * 4
            m_h0_inline265__ssa_v0: pl.Scalar[pl.INDEX] = m_h_idx_inline263__ssa_v0 * 16
            m_row_inline249__ssa_v0: pl.Scalar[pl.INDEX] = m_idx_inline259__idx_v0 * 16
            m_li_inline264__ssa_v0: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_19, pl.const(40960, pl.INT64), 64), pl.Mem.Vec] = pl.tile.load(
                attn_li_inline243__ssa_v0, [m_row_inline249__ssa_v0, 0], [16, 1], [16, 1], target_memory=pl.Mem.Vec
            )
            m_oi_inline254__ssa_v0: pl.Tile[[16, 512], pl.FP32, pl.MemRef(mem_vec_17, pl.const(8192, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.load(
                attn_oi_inline260__ssa_v0, [m_row_inline249__ssa_v0, 0], [16, 512], [16, 512], target_memory=pl.Mem.Vec
            )
            n_full_inline232__ssa_v0: pl.Tile[[16, 512], pl.FP32, pl.MemRef(mem_vec_17, pl.const(8192, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.row_expand_div(
                m_oi_inline254__ssa_v0, m_li_inline264__ssa_v0
            )
            n_bf16_inline231__ssa_v0: pl.Tile[[16, 512], pl.BF16, pl.MemRef(mem_vec_19, pl.const(40960, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.cast(
                n_full_inline232__ssa_v0, target_type=pl.BF16, mode="rint"
            )
            n_rounded_inline255__ssa_v0: pl.Tile[[16, 512], pl.FP32, pl.MemRef(mem_vec_17, pl.const(8192, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.cast(
                n_bf16_inline231__ssa_v0, target_type=pl.FP32, mode="round"
            )
            m_cos_il_inline248__ssa_v0: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_21, pl.const(57344, pl.INT64), 256), pl.Mem.Vec] = pl.tile.load(
                rope_cos_il_inline238__ssa_v0, [m_t_inline261__ssa_v0, 0], [1, 64], [1, 64], target_memory=pl.Mem.Vec
            )
            m_sin_signed_inline228__ssa_v0: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_22, pl.const(57600, pl.INT64), 256), pl.Mem.Vec] = pl.tile.load(
                rope_sin_signed_inline242__ssa_v0, [m_t_inline261__ssa_v0, 0], [1, 64], [1, 64], target_memory=pl.Mem.Vec
            )
            m_swapped_inline227__ssa_v0: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_23, pl.const(57856, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.gather(
                n_rounded_inline255__ssa_v0, m_swap_idx_inline253__ssa_v0, m_gather_tmp_inline256__ssa_v0
            )
            m_rope_inline229__ssa_v0_textract: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_24, pl.const(61952, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.extract(
                n_rounded_inline255__ssa_v0, 0, 448, [16, 64], target_memory=pl.Mem.Vec
            )
            t__tmp_v335: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_17, pl.const(8192, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.col_expand_mul(
                m_rope_inline229__ssa_v0_textract, m_cos_il_inline248__ssa_v0
            )
            t__tmp_v336: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_23, pl.const(57856, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.col_expand_mul(
                m_swapped_inline227__ssa_v0, m_sin_signed_inline228__ssa_v0
            )
            m_rot_inline226__ssa_v0: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_17, pl.const(8192, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.add(t__tmp_v335, t__tmp_v336)
            n_rope_bf16_inline251__ssa_v0: pl.Tile[[16, 64], pl.BF16, pl.MemRef(mem_vec_23, pl.const(57856, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(
                m_rot_inline226__ssa_v0, target_type=pl.BF16, mode="rint"
            )
            t__tmp_v337: pl.Tile[[16, 448], pl.BF16, pl.MemRef(mem_vec_19, pl.const(40960, pl.INT64), 16256), pl.Mem.Vec] = pl.tile.slice(n_bf16_inline231__ssa_v0, [16, 448], [0, 0])
            n_full_bf16_inline230__ssa_v0: pl.Tile[[16, 512], pl.BF16, pl.MemRef(mem_vec_17, pl.const(8192, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.concat(t__tmp_v337, n_rope_bf16_inline251__ssa_v0)
            n_group_bf16_inline225__ssa_v0: pl.Tile[[2, 4096], pl.BF16, pl.MemRef(mem_vec_17, pl.const(8192, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.reshape(n_full_bf16_inline230__ssa_v0, [2, 4096])
            n_pack_first_inline224__ssa_v0: pl.Tile[[1, 4096], pl.BF16, pl.MemRef(mem_vec_17, pl.const(8192, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.slice(
                n_group_bf16_inline225__ssa_v0, [1, 4096], [0, 0]
            )
            n_pack_second_inline223__ssa_v0: pl.Tile[[1, 4096], pl.BF16, pl.MemRef(mem_vec_17, pl.const(16384, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.slice(
                n_group_bf16_inline225__ssa_v0, [1, 4096], [1, 0]
            )
            n_pack_row_inline257__ssa_v0: pl.Scalar[pl.INDEX] = m_h0_inline265__ssa_v0 // 8 * 384 + m_t_inline261__ssa_v0
            n_pack_row_second_inline222__ssa_v0: pl.Scalar[pl.INDEX] = n_pack_row_inline257__ssa_v0 + 384
            o_packed_heads__store: pl.Tensor[[3072, 4096], pl.BF16, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 25165824)] = pl.tile.store(
                n_pack_first_inline224__ssa_v0, [n_pack_row_inline257__ssa_v0, 0], o_packed_heads__ssa_v0
            )
            o_packed_heads__store_v0: pl.Tensor[[3072, 4096], pl.BF16, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 25165824)] = pl.tile.store(
                n_pack_second_inline223__ssa_v0, [n_pack_row_second_inline222__ssa_v0, 0], o_packed_heads__ssa_v0
            )
        return o_packed_heads__ssa_v0

    @pl.function(type=pl.FunctionType.Spmd)
    def merge_norm_spmd(
        self,
        rope_swap_idx_inline244__ssa_v0: pl.Tensor[[16, 64], pl.INT32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 4096)],
        t_dim_inline235__ssa_v0: pl.Scalar[pl.INDEX],
        attn_li_inline243__ssa_v0: pl.Tensor[[t_heads_inline2469__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)],
        attn_oi_inline260__ssa_v0: pl.Tensor[[t_heads_inline2469__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)],
        rope_cos_il_inline238__ssa_v0: pl.Tensor[[T_DYN, 64], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)],
        rope_sin_signed_inline242__ssa_v0: pl.Tensor[[384, 64], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 98304)],
        o_packed_heads__ssa_v0: pl.Out[pl.Tensor[[3072, 4096], pl.BF16, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 25165824)]],
    ) -> pl.Tensor[[3072, 4096], pl.BF16]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        o_packed_heads__ssa_v2: pl.Tensor[[3072, 4096], pl.BF16, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 25165824)] = self.merge_norm(
            rope_swap_idx_inline244__ssa_v0,
            t_dim_inline235__ssa_v0,
            attn_li_inline243__ssa_v0,
            attn_oi_inline260__ssa_v0,
            rope_cos_il_inline238__ssa_v0,
            rope_sin_signed_inline242__ssa_v0,
            o_packed_heads__ssa_v0,
            attrs={"arg_directions": [pl.adir.input, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.output_existing]},
        )
        return o_packed_heads__ssa_v0

    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def oproj_token_scale(
        act_scale_dq_inline313__ssa_v0: pl.Out[pl.Tensor[[1, 384], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 1536)]],
        act_scale_q_inline338__ssa_v0: pl.Out[pl.Tensor[[1, 384], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 1536)]],
        t_dim_inline347__ssa_v0: pl.Scalar[pl.INDEX],
        o_r_pad_inline329__rv_v2: pl.Tensor[[384, 8192], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 12582912)],
    ) -> tuple[pl.Tensor[[1, 384], pl.FP32], pl.Tensor[[1, 384], pl.FP32]]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        mem_vec_3: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32)
        mem_vec_4: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32768)
        mem_vec_8: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32768)
        mem_vec_9: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32)
        mem_vec_15: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32)
        mem_vec_16: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32768)
        mem_vec_20: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32768)
        mem_vec_21: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32)
        unroll_main_end: pl.Scalar[pl.INDEX] = (t_dim_inline347__ssa_v0 + 7) // 16 * 16
        for qt_inline352__idx_v0, (act_scale_dq_inline313__iter_v1, act_scale_q_inline338__iter_v1) in pl.range(
            0, unroll_main_end, 16, init_values=(act_scale_dq_inline313__ssa_v0, act_scale_q_inline338__ssa_v0)
        ):
            token_amax_inline353__tile: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_3, pl.const(65600, pl.INT64), 32), pl.Mem.Vec] = pl.tile.full([1, 8], dtype=pl.FP32, value=0.0001)
            for scale_group_inline330__idx_v0, (token_amax_inline353__iter_v1,) in pl.range(8, init_values=(token_amax_inline353__tile,)):
                scale_col_inline340__ssa_v0: pl.Scalar[pl.INDEX] = scale_group_inline330__idx_v0 * 1024
                projected_inline354__tile: pl.Tile[[8, 1024], pl.FP32, pl.MemRef(mem_vec_4, pl.const(65632, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.load(
                    o_r_pad_inline329__rv_v2, [qt_inline352__idx_v0, scale_col_inline340__ssa_v0], [8, 1024], [8, 1024], target_memory=pl.Mem.Vec
                )
                t__tile: pl.Tile[[8, 1024], pl.BF16, pl.MemRef(mem_vec_8, pl.const(98400, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.cast(projected_inline354__tile, target_type=pl.BF16, mode="rint")
                projected_v1_inline328__tile: pl.Tile[[8, 1024], pl.FP32, pl.MemRef(mem_vec_4, pl.const(65632, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.cast(t__tile, target_type=pl.FP32, mode="round")
                t__tile_1: pl.Tile[[8, 1024], pl.FP32, pl.MemRef(mem_vec_4, pl.const(65632, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.abs(projected_v1_inline328__tile)
                tmp_tile: pl.Tile[[8, 1024], pl.FP32, pl.MemRef(mem_vec_8, pl.const(98400, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.create([8, 1024], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                t__tile_2: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_9, pl.const(131168, pl.INT64), 32), pl.Mem.Vec] = pl.tile.row_max(t__tile_1, tmp_tile)
                group_amax_inline356__tile: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_9, pl.const(131168, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(t__tile_2, [1, 8])
                token_amax_inline353__tile_1: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_3, pl.const(65600, pl.INT64), 32), pl.Mem.Vec] = pl.tile.maximum(
                    token_amax_inline353__iter_v1, group_amax_inline356__tile
                )
                token_amax_inline353__rv_v2: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_3, pl.const(65600, pl.INT64), 32), pl.Mem.Vec] = pl.yield_(token_amax_inline353__tile_1)
            t__tile_3: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_4, pl.const(65632, pl.INT64), 32), pl.Mem.Vec] = pl.tile.muls(token_amax_inline353__rv_v2, 0.007874015748031496)
            quant_numerator_inline348__tile: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_8, pl.const(98400, pl.INT64), 32), pl.Mem.Vec] = pl.tile.full([1, 8], dtype=pl.FP32, value=127.0)
            t__tile_4: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_8, pl.const(98400, pl.INT64), 32), pl.Mem.Vec] = pl.tile.div(quant_numerator_inline348__tile, token_amax_inline353__rv_v2)
            act_scale_dq_inline313__tile: pl.Tensor[[1, 384], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 1536)] = pl.tile.store(
                t__tile_3, [0, qt_inline352__idx_v0], act_scale_dq_inline313__iter_v1
            )
            act_scale_q_inline338__tile: pl.Tensor[[1, 384], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 1536)] = pl.tile.store(
                t__tile_4, [0, qt_inline352__idx_v0], act_scale_q_inline338__iter_v1
            )
            token_amax_inline353__tile_2: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_15, pl.const(0, pl.INT64), 32), pl.Mem.Vec] = pl.tile.full([1, 8], dtype=pl.FP32, value=0.0001)
            for scale_group_inline330__idx_v0_1, (token_amax_inline353__iter_v1_1,) in pl.range(8, init_values=(token_amax_inline353__tile_2,)):
                scale_col_inline340__ssa_v0_1: pl.Scalar[pl.INDEX] = scale_group_inline330__idx_v0_1 * 1024
                projected_inline354__tile_1: pl.Tile[[8, 1024], pl.FP32, pl.MemRef(mem_vec_16, pl.const(32, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.load(
                    o_r_pad_inline329__rv_v2, [qt_inline352__idx_v0 + 8, scale_col_inline340__ssa_v0_1], [8, 1024], [8, 1024], target_memory=pl.Mem.Vec
                )
                t__tile_5: pl.Tile[[8, 1024], pl.BF16, pl.MemRef(mem_vec_20, pl.const(32800, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.cast(
                    projected_inline354__tile_1, target_type=pl.BF16, mode="rint"
                )
                projected_v1_inline328__tile_1: pl.Tile[[8, 1024], pl.FP32, pl.MemRef(mem_vec_16, pl.const(32, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.cast(
                    t__tile_5, target_type=pl.FP32, mode="round"
                )
                t__tile_6: pl.Tile[[8, 1024], pl.FP32, pl.MemRef(mem_vec_16, pl.const(32, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.abs(projected_v1_inline328__tile_1)
                tmp_tile_1: pl.Tile[[8, 1024], pl.FP32, pl.MemRef(mem_vec_20, pl.const(32800, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.create([8, 1024], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                t__tile_7: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_21, pl.const(65568, pl.INT64), 32), pl.Mem.Vec] = pl.tile.row_max(t__tile_6, tmp_tile_1)
                group_amax_inline356__tile_1: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_21, pl.const(65568, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(t__tile_7, [1, 8])
                token_amax_inline353__tile_3: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_15, pl.const(0, pl.INT64), 32), pl.Mem.Vec] = pl.tile.maximum(
                    token_amax_inline353__iter_v1_1, group_amax_inline356__tile_1
                )
                token_amax_inline353__rv_v2_1: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_15, pl.const(0, pl.INT64), 32), pl.Mem.Vec] = pl.yield_(token_amax_inline353__tile_3)
            t__tile_8: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_16, pl.const(32, pl.INT64), 32), pl.Mem.Vec] = pl.tile.muls(token_amax_inline353__rv_v2_1, 0.007874015748031496)
            quant_numerator_inline348__tile_1: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_20, pl.const(32800, pl.INT64), 32), pl.Mem.Vec] = pl.tile.full([1, 8], dtype=pl.FP32, value=127.0)
            t__tile_9: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_20, pl.const(32800, pl.INT64), 32), pl.Mem.Vec] = pl.tile.div(quant_numerator_inline348__tile_1, token_amax_inline353__rv_v2_1)
            act_scale_dq_inline313__tile_1: pl.Tensor[[1, 384], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 1536)] = pl.tile.store(
                t__tile_8, [0, qt_inline352__idx_v0 + 8], act_scale_dq_inline313__tile
            )
            act_scale_q_inline338__tile_1: pl.Tensor[[1, 384], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 1536)] = pl.tile.store(
                t__tile_9, [0, qt_inline352__idx_v0 + 8], act_scale_q_inline338__tile
            )
            act_scale_dq_inline313__rv_v2_main, act_scale_q_inline338__rv_v2_main = pl.yield_(act_scale_dq_inline313__tile_1, act_scale_q_inline338__tile_1)
        unroll_rem: pl.Scalar[pl.INDEX] = (t_dim_inline347__ssa_v0 + 7) // 8 - (t_dim_inline347__ssa_v0 + 7) // 16 * 2
        if unroll_rem == 1:
            token_amax_inline353__tile_4: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_16, pl.const(32, pl.INT64), 32), pl.Mem.Vec] = pl.tile.full([1, 8], dtype=pl.FP32, value=0.0001)
            for scale_group_inline330__idx_v0_2, (token_amax_inline353__iter_v1_2,) in pl.range(8, init_values=(token_amax_inline353__tile_4,)):
                scale_col_inline340__ssa_v0_2: pl.Scalar[pl.INDEX] = scale_group_inline330__idx_v0_2 * 1024
                projected_inline354__tile_2: pl.Tile[[8, 1024], pl.FP32, pl.MemRef(mem_vec_4, pl.const(65632, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.load(
                    o_r_pad_inline329__rv_v2, [unroll_main_end, scale_col_inline340__ssa_v0_2], [8, 1024], [8, 1024], target_memory=pl.Mem.Vec
                )
                t__tile_10: pl.Tile[[8, 1024], pl.BF16, pl.MemRef(mem_vec_8, pl.const(98400, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.cast(
                    projected_inline354__tile_2, target_type=pl.BF16, mode="rint"
                )
                projected_v1_inline328__tile_2: pl.Tile[[8, 1024], pl.FP32, pl.MemRef(mem_vec_4, pl.const(65632, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.cast(
                    t__tile_10, target_type=pl.FP32, mode="round"
                )
                t__tile_11: pl.Tile[[8, 1024], pl.FP32, pl.MemRef(mem_vec_4, pl.const(65632, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.abs(projected_v1_inline328__tile_2)
                tmp_tile_2: pl.Tile[[8, 1024], pl.FP32, pl.MemRef(mem_vec_8, pl.const(98400, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.create([8, 1024], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                t__tile_12: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_20, pl.const(32800, pl.INT64), 32), pl.Mem.Vec] = pl.tile.row_max(t__tile_11, tmp_tile_2)
                group_amax_inline356__tile_2: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_20, pl.const(32800, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(t__tile_12, [1, 8])
                token_amax_inline353__tile_5: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_16, pl.const(32, pl.INT64), 32), pl.Mem.Vec] = pl.tile.maximum(
                    token_amax_inline353__iter_v1_2, group_amax_inline356__tile_2
                )
                token_amax_inline353__rv_v2_2: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_16, pl.const(32, pl.INT64), 32), pl.Mem.Vec] = pl.yield_(token_amax_inline353__tile_5)
            t__tile_13: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_4, pl.const(65632, pl.INT64), 32), pl.Mem.Vec] = pl.tile.muls(token_amax_inline353__rv_v2_2, 0.007874015748031496)
            act_scale_dq_inline313__tile_2: pl.Tensor[[1, 384], pl.FP32, pl.MemRef("mem_ddr_27", pl.const(0, pl.INT64), 1536)] = pl.tile.store(
                t__tile_13, [0, unroll_main_end], act_scale_dq_inline313__rv_v2_main
            )
            quant_numerator_inline348__tile_2: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_4, pl.const(65632, pl.INT64), 32), pl.Mem.Vec] = pl.tile.full([1, 8], dtype=pl.FP32, value=127.0)
            t__tile_14: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_4, pl.const(65632, pl.INT64), 32), pl.Mem.Vec] = pl.tile.div(quant_numerator_inline348__tile_2, token_amax_inline353__rv_v2_2)
            act_scale_q_inline338__tile_2: pl.Tensor[[1, 384], pl.FP32, pl.MemRef("mem_ddr_28", pl.const(0, pl.INT64), 1536)] = pl.tile.store(
                t__tile_14, [0, unroll_main_end], act_scale_q_inline338__rv_v2_main
            )
        return act_scale_dq_inline313__ssa_v0, act_scale_q_inline338__ssa_v0

    @pl.function(type=pl.FunctionType.AIC, level=pl.Level.AIC, role=pl.Role.SubWorker)
    def proj_a_mm(
        t_dim_inline347__ssa_v0: pl.Scalar[pl.INDEX],
        row_base_o_inline310__ssa_v0: pl.Scalar[pl.INDEX],
        o_packed_heads__ssa_v1: pl.Tensor[[3072, 4096], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 25165824)],
        wo_a__ssa_v0: pl.Tensor[[8, 1024, 4096], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 67108864)],
        g_inline293__idx_v0: pl.Scalar[pl.INDEX],
        o_r_pad_inline329__iter_v1: pl.Out[pl.Tensor[[384, 8192], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 12582912)]],
        out_col_g_inline306__ssa_v0: pl.Scalar[pl.INDEX],
    ) -> pl.Tensor[[384, 8192], pl.FP32]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        mem_mat_3: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 65536)
        mem_mat_4: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 65536)
        mem_acc_5: pl.Ptr = pl.tile.alloc(pl.Mem.Acc, 65536)
        mem_left_6: pl.Ptr = pl.tile.alloc(pl.Mem.Left, 32768)
        mem_right_7: pl.Ptr = pl.tile.alloc(pl.Mem.Right, 32768)
        mem_left_8: pl.Ptr = pl.tile.alloc(pl.Mem.Left, 32768)
        mem_right_9: pl.Ptr = pl.tile.alloc(pl.Mem.Right, 32768)
        mem_mat_13: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 65536)
        mem_mat_14: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 65536)
        pa_unit_inline325__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        pa_rb_inline359__ssa_v0: pl.Scalar[pl.INDEX] = pa_unit_inline325__ssa_v0 // 8
        nf_inline296__ssa_v0: pl.Scalar[pl.INDEX] = pa_unit_inline325__ssa_v0 - pa_rb_inline359__ssa_v0 * 8
        pa_r0_inline301__ssa_v0: pl.Scalar[pl.INDEX] = pa_rb_inline359__ssa_v0 * 128
        pa_rows_inline298__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(t_dim_inline347__ssa_v0 - pa_r0_inline301__ssa_v0, 128)
        pa_src0_inline324__ssa_v0: pl.Scalar[pl.INDEX] = row_base_o_inline310__ssa_v0 + pa_r0_inline301__ssa_v0
        n0_inline304__ssa_v0: pl.Scalar[pl.INDEX] = nf_inline296__ssa_v0 * 128
        xa_first_inline334__tile: pl.Tile[[128, 256], pl.BF16, pl.MemRef(mem_mat_3, pl.const(131072, pl.INT64), 65536), pl.Mem.Mat, pl.TileView(valid_shape=[pa_rows_inline298__ssa_v0, 256])] = (
            pl.tile.load(o_packed_heads__ssa_v1, [pa_src0_inline324__ssa_v0, 0], [128, 256], [pa_rows_inline298__ssa_v0, 256], target_memory=pl.Mem.Mat)
        )
        wa_first_inline292__tile_view2d: pl.Tensor[[8192, 4096], pl.BF16, pl.TensorView(stride=[4096, 1], layout=pl.TensorLayout.ND), pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 67108864)] = (
            pl.tensor.view(wo_a__ssa_v0, [8192, 4096])
        )
        wa_first_inline292__tile: pl.Tile[[128, 256], pl.BF16, pl.MemRef(mem_mat_4, pl.const(196608, pl.INT64), 65536), pl.Mem.Mat] = pl.tile.load(
            wa_first_inline292__tile_view2d, [g_inline293__idx_v0 * 1024 + n0_inline304__ssa_v0, 0], [128, 256], [128, 256], target_memory=pl.Mem.Mat
        )
        wa_first_inline292__tile_t: pl.Tile[
            [256, 128], pl.BF16, pl.MemRef(mem_mat_4, pl.const(196608, pl.INT64), 65536), pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)
        ] = pl.tile.transpose_view(wa_first_inline292__tile)
        acc_a_inline316__tile_l0_init_storage: pl.Tile[[128, 128], pl.FP32, pl.MemRef(mem_acc_5, pl.const(0, pl.INT64), 65536), pl.Mem.Acc, pl.TileView(compact=pl.CompactMode.normal)] = (
            pl.tile.create([128, 128], dtype=pl.FP32, target_memory=pl.Mem.Acc, compact=True)
        )
        acc_a_inline316__tile_l0_init: pl.Tile[
            [128, 128], pl.FP32, pl.MemRef(mem_acc_5, pl.const(0, pl.INT64), 65536), pl.Mem.Acc, pl.TileView(valid_shape=[pa_rows_inline298__ssa_v0, 128], compact=pl.CompactMode.normal)
        ] = pl.tile.set_validshape(acc_a_inline316__tile_l0_init_storage, pa_rows_inline298__ssa_v0, 128)
        acc_a_inline316__tile_l0_a: pl.Tile[
            [128, 128],
            pl.BF16,
            pl.MemRef(mem_left_6, pl.const(0, pl.INT64), 32768),
            pl.Mem.Left,
            pl.TileView(valid_shape=[pa_rows_inline298__ssa_v0, 128], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal),
        ] = pl.tile.extract(xa_first_inline334__tile, 0, 0, [128, 128], target_memory=pl.Mem.Left)
        acc_a_inline316__tile_l0_b: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_7, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
            wa_first_inline292__tile_t, 0, 0, [128, 128], target_memory=pl.Mem.Right
        )
        acc_a_inline316__tile_l0_a_1: pl.Tile[
            [128, 128],
            pl.BF16,
            pl.MemRef(mem_left_8, pl.const(32768, pl.INT64), 32768),
            pl.Mem.Left,
            pl.TileView(valid_shape=[pa_rows_inline298__ssa_v0, 128], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal),
        ] = pl.tile.extract(xa_first_inline334__tile, 0, 128, [128, 128], target_memory=pl.Mem.Left)
        acc_a_inline316__tile_l0_b_1: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_9, pl.const(32768, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
            wa_first_inline292__tile_t, 128, 0, [128, 128], target_memory=pl.Mem.Right
        )
        acc_a_inline316__tile_l0_c_acc: pl.Tile[
            [128, 128], pl.FP32, pl.MemRef(mem_acc_5, pl.const(0, pl.INT64), 65536), pl.Mem.Acc, pl.TileView(valid_shape=[pa_rows_inline298__ssa_v0, 128], compact=pl.CompactMode.normal)
        ] = pl.tile.matmul_acc(acc_a_inline316__tile_l0_init, acc_a_inline316__tile_l0_a, acc_a_inline316__tile_l0_b, True)
        acc_a_inline316__tile_l0_c_acc_1: pl.Tile[
            [128, 128], pl.FP32, pl.MemRef(mem_acc_5, pl.const(0, pl.INT64), 65536), pl.Mem.Acc, pl.TileView(valid_shape=[pa_rows_inline298__ssa_v0, 128], compact=pl.CompactMode.normal)
        ] = pl.tile.matmul_acc(acc_a_inline316__tile_l0_c_acc, acc_a_inline316__tile_l0_a_1, acc_a_inline316__tile_l0_b_1, False)
        for kb_inline321__idx_v0, (acc_a_inline316__iter_v1,) in pl.range(1, 15, 2, init_values=(acc_a_inline316__tile_l0_c_acc_1,)):
            k0_inline336__ssa_v0: pl.Scalar[pl.INDEX] = kb_inline321__idx_v0 * 256
            k0_inline336__ssa_v0_1: pl.Scalar[pl.INDEX] = kb_inline321__idx_v0 * 256 + 256
            xa_k_chunk_inline291__tile: pl.Tile[[128, 256], pl.BF16, pl.MemRef(mem_mat_3, pl.const(131072, pl.INT64), 65536), pl.Mem.Mat, pl.TileView(valid_shape=[pa_rows_inline298__ssa_v0, 256])] = (
                pl.tile.load(o_packed_heads__ssa_v1, [pa_src0_inline324__ssa_v0, k0_inline336__ssa_v0], [128, 256], [pa_rows_inline298__ssa_v0, 256], target_memory=pl.Mem.Mat)
            )
            xa_k_chunk_inline291__tile_1: pl.Tile[
                [128, 256], pl.BF16, pl.MemRef(mem_mat_4, pl.const(196608, pl.INT64), 65536), pl.Mem.Mat, pl.TileView(valid_shape=[pa_rows_inline298__ssa_v0, 256])
            ] = pl.tile.load(o_packed_heads__ssa_v1, [pa_src0_inline324__ssa_v0, k0_inline336__ssa_v0_1], [128, 256], [pa_rows_inline298__ssa_v0, 256], target_memory=pl.Mem.Mat)
            wa_k_chunk_inline341__tile_view2d: pl.Tensor[[8192, 4096], pl.BF16, pl.TensorView(stride=[4096, 1], layout=pl.TensorLayout.ND), pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 67108864)] = (
                pl.tensor.view(wo_a__ssa_v0, [8192, 4096])
            )
            wa_k_chunk_inline341__tile: pl.Tile[[128, 256], pl.BF16, pl.MemRef(mem_mat_13, pl.const(0, pl.INT64), 65536), pl.Mem.Mat] = pl.tile.load(
                wa_k_chunk_inline341__tile_view2d, [g_inline293__idx_v0 * 1024 + n0_inline304__ssa_v0, k0_inline336__ssa_v0], [128, 256], [128, 256], target_memory=pl.Mem.Mat
            )
            wa_k_chunk_inline341__tile_view2d_1: pl.Tensor[
                [8192, 4096], pl.BF16, pl.TensorView(stride=[4096, 1], layout=pl.TensorLayout.ND), pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 67108864)
            ] = pl.tensor.view(wo_a__ssa_v0, [8192, 4096])
            wa_k_chunk_inline341__tile_1: pl.Tile[[128, 256], pl.BF16, pl.MemRef(mem_mat_14, pl.const(65536, pl.INT64), 65536), pl.Mem.Mat] = pl.tile.load(
                wa_k_chunk_inline341__tile_view2d_1, [g_inline293__idx_v0 * 1024 + n0_inline304__ssa_v0, k0_inline336__ssa_v0_1], [128, 256], [128, 256], target_memory=pl.Mem.Mat
            )
            wa_k_chunk_inline341__tile_t: pl.Tile[
                [256, 128], pl.BF16, pl.MemRef(mem_mat_13, pl.const(0, pl.INT64), 65536), pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)
            ] = pl.tile.transpose_view(wa_k_chunk_inline341__tile)
            acc_a_inline316__iter_v1_l0_a: pl.Tile[
                [128, 128],
                pl.BF16,
                pl.MemRef(mem_left_6, pl.const(0, pl.INT64), 32768),
                pl.Mem.Left,
                pl.TileView(valid_shape=[pa_rows_inline298__ssa_v0, 128], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal),
            ] = pl.tile.extract(xa_k_chunk_inline291__tile, 0, 0, [128, 128], target_memory=pl.Mem.Left)
            acc_a_inline316__iter_v1_l0_b: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_7, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                wa_k_chunk_inline341__tile_t, 0, 0, [128, 128], target_memory=pl.Mem.Right
            )
            acc_a_inline316__iter_v1_l0_a_1: pl.Tile[
                [128, 128],
                pl.BF16,
                pl.MemRef(mem_left_8, pl.const(32768, pl.INT64), 32768),
                pl.Mem.Left,
                pl.TileView(valid_shape=[pa_rows_inline298__ssa_v0, 128], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal),
            ] = pl.tile.extract(xa_k_chunk_inline291__tile, 0, 128, [128, 128], target_memory=pl.Mem.Left)
            acc_a_inline316__iter_v1_l0_b_1: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_9, pl.const(32768, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                wa_k_chunk_inline341__tile_t, 128, 0, [128, 128], target_memory=pl.Mem.Right
            )
            acc_a_inline316__iter_v1_l0_c_acc: pl.Tile[
                [128, 128], pl.FP32, pl.MemRef(mem_acc_5, pl.const(0, pl.INT64), 65536), pl.Mem.Acc, pl.TileView(valid_shape=[pa_rows_inline298__ssa_v0, 128], compact=pl.CompactMode.normal)
            ] = pl.tile.matmul_acc(acc_a_inline316__iter_v1, acc_a_inline316__iter_v1_l0_a, acc_a_inline316__iter_v1_l0_b)
            acc_a_inline316__iter_v1_l0_c_acc_1: pl.Tile[
                [128, 128], pl.FP32, pl.MemRef(mem_acc_5, pl.const(0, pl.INT64), 65536), pl.Mem.Acc, pl.TileView(valid_shape=[pa_rows_inline298__ssa_v0, 128], compact=pl.CompactMode.normal)
            ] = pl.tile.matmul_acc(acc_a_inline316__iter_v1_l0_c_acc, acc_a_inline316__iter_v1_l0_a_1, acc_a_inline316__iter_v1_l0_b_1)
            wa_k_chunk_inline341__tile_t_1: pl.Tile[
                [256, 128], pl.BF16, pl.MemRef(mem_mat_14, pl.const(65536, pl.INT64), 65536), pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)
            ] = pl.tile.transpose_view(wa_k_chunk_inline341__tile_1)
            acc_a_inline316__iter_v1_l0_a_2: pl.Tile[
                [128, 128],
                pl.BF16,
                pl.MemRef(mem_left_6, pl.const(0, pl.INT64), 32768),
                pl.Mem.Left,
                pl.TileView(valid_shape=[pa_rows_inline298__ssa_v0, 128], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal),
            ] = pl.tile.extract(xa_k_chunk_inline291__tile_1, 0, 0, [128, 128], target_memory=pl.Mem.Left)
            acc_a_inline316__iter_v1_l0_b_2: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_7, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                wa_k_chunk_inline341__tile_t_1, 0, 0, [128, 128], target_memory=pl.Mem.Right
            )
            acc_a_inline316__iter_v1_l0_a_3: pl.Tile[
                [128, 128],
                pl.BF16,
                pl.MemRef(mem_left_8, pl.const(32768, pl.INT64), 32768),
                pl.Mem.Left,
                pl.TileView(valid_shape=[pa_rows_inline298__ssa_v0, 128], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal),
            ] = pl.tile.extract(xa_k_chunk_inline291__tile_1, 0, 128, [128, 128], target_memory=pl.Mem.Left)
            acc_a_inline316__iter_v1_l0_b_3: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_9, pl.const(32768, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                wa_k_chunk_inline341__tile_t_1, 128, 0, [128, 128], target_memory=pl.Mem.Right
            )
            acc_a_inline316__iter_v1_l0_c_acc_2: pl.Tile[
                [128, 128], pl.FP32, pl.MemRef(mem_acc_5, pl.const(0, pl.INT64), 65536), pl.Mem.Acc, pl.TileView(valid_shape=[pa_rows_inline298__ssa_v0, 128], compact=pl.CompactMode.normal)
            ] = pl.tile.matmul_acc(acc_a_inline316__iter_v1_l0_c_acc_1, acc_a_inline316__iter_v1_l0_a_2, acc_a_inline316__iter_v1_l0_b_2)
            acc_a_inline316__iter_v1_l0_c_acc_3: pl.Tile[
                [128, 128], pl.FP32, pl.MemRef(mem_acc_5, pl.const(0, pl.INT64), 65536), pl.Mem.Acc, pl.TileView(valid_shape=[pa_rows_inline298__ssa_v0, 128], compact=pl.CompactMode.normal)
            ] = pl.tile.matmul_acc(acc_a_inline316__iter_v1_l0_c_acc_2, acc_a_inline316__iter_v1_l0_a_3, acc_a_inline316__iter_v1_l0_b_3)
            acc_a_inline316__rv_v2_main: pl.Tile[
                [128, 128], pl.FP32, pl.MemRef(mem_acc_5, pl.const(0, pl.INT64), 65536), pl.Mem.Acc, pl.TileView(valid_shape=[pa_rows_inline298__ssa_v0, 128], compact=pl.CompactMode.normal)
            ] = pl.yield_(acc_a_inline316__iter_v1_l0_c_acc_3)
        xa_k_chunk_inline291__tile_2: pl.Tile[[128, 256], pl.BF16, pl.MemRef(mem_mat_3, pl.const(131072, pl.INT64), 65536), pl.Mem.Mat, pl.TileView(valid_shape=[pa_rows_inline298__ssa_v0, 256])] = (
            pl.tile.load(o_packed_heads__ssa_v1, [pa_src0_inline324__ssa_v0, 3840], [128, 256], [pa_rows_inline298__ssa_v0, 256], target_memory=pl.Mem.Mat)
        )
        wa_k_chunk_inline341__tile_view2d_2: pl.Tensor[[8192, 4096], pl.BF16, pl.TensorView(stride=[4096, 1], layout=pl.TensorLayout.ND), pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 67108864)] = (
            pl.tensor.view(wo_a__ssa_v0, [8192, 4096])
        )
        wa_k_chunk_inline341__tile_2: pl.Tile[[128, 256], pl.BF16, pl.MemRef(mem_mat_4, pl.const(196608, pl.INT64), 65536), pl.Mem.Mat] = pl.tile.load(
            wa_k_chunk_inline341__tile_view2d_2, [g_inline293__idx_v0 * 1024 + n0_inline304__ssa_v0, 3840], [128, 256], [128, 256], target_memory=pl.Mem.Mat
        )
        wa_k_chunk_inline341__tile_t_2: pl.Tile[
            [256, 128], pl.BF16, pl.MemRef(mem_mat_4, pl.const(196608, pl.INT64), 65536), pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)
        ] = pl.tile.transpose_view(wa_k_chunk_inline341__tile_2)
        acc_a_inline316__iter_v1_l0_a_4: pl.Tile[
            [128, 128],
            pl.BF16,
            pl.MemRef(mem_left_6, pl.const(0, pl.INT64), 32768),
            pl.Mem.Left,
            pl.TileView(valid_shape=[pa_rows_inline298__ssa_v0, 128], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal),
        ] = pl.tile.extract(xa_k_chunk_inline291__tile_2, 0, 0, [128, 128], target_memory=pl.Mem.Left)
        acc_a_inline316__iter_v1_l0_b_4: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_7, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
            wa_k_chunk_inline341__tile_t_2, 0, 0, [128, 128], target_memory=pl.Mem.Right
        )
        acc_a_inline316__iter_v1_l0_a_5: pl.Tile[
            [128, 128],
            pl.BF16,
            pl.MemRef(mem_left_8, pl.const(32768, pl.INT64), 32768),
            pl.Mem.Left,
            pl.TileView(valid_shape=[pa_rows_inline298__ssa_v0, 128], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal),
        ] = pl.tile.extract(xa_k_chunk_inline291__tile_2, 0, 128, [128, 128], target_memory=pl.Mem.Left)
        acc_a_inline316__iter_v1_l0_b_5: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_9, pl.const(32768, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
            wa_k_chunk_inline341__tile_t_2, 128, 0, [128, 128], target_memory=pl.Mem.Right
        )
        acc_a_inline316__iter_v1_l0_c_acc_4: pl.Tile[
            [128, 128], pl.FP32, pl.MemRef(mem_acc_5, pl.const(0, pl.INT64), 65536), pl.Mem.Acc, pl.TileView(valid_shape=[pa_rows_inline298__ssa_v0, 128], compact=pl.CompactMode.normal)
        ] = pl.tile.matmul_acc(acc_a_inline316__rv_v2_main, acc_a_inline316__iter_v1_l0_a_4, acc_a_inline316__iter_v1_l0_b_4)
        acc_a_inline316__iter_v1_l0_c_acc_5: pl.Tile[
            [128, 128], pl.FP32, pl.MemRef(mem_acc_5, pl.const(0, pl.INT64), 65536), pl.Mem.Acc, pl.TileView(valid_shape=[pa_rows_inline298__ssa_v0, 128], compact=pl.CompactMode.normal)
        ] = pl.tile.matmul_acc(acc_a_inline316__iter_v1_l0_c_acc_4, acc_a_inline316__iter_v1_l0_a_5, acc_a_inline316__iter_v1_l0_b_5)
        acc_a_inline316__rv_v2: pl.Tile[
            [128, 128], pl.FP32, pl.MemRef(mem_acc_5, pl.const(0, pl.INT64), 65536), pl.Mem.Acc, pl.TileView(valid_shape=[pa_rows_inline298__ssa_v0, 128], compact=pl.CompactMode.normal)
        ] = acc_a_inline316__iter_v1_l0_c_acc_5
        o_r_pad_inline329__tile: pl.Tensor[[384, 8192], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 12582912)] = pl.tile.store(
            acc_a_inline316__rv_v2, [pa_r0_inline301__ssa_v0, out_col_g_inline306__ssa_v0 + n0_inline304__ssa_v0], o_r_pad_inline329__iter_v1
        )
        return o_r_pad_inline329__iter_v1

    @pl.function(type=pl.FunctionType.Spmd)
    def proj_a_mm_spmd(
        self,
        t_dim_inline347__ssa_v0: pl.Scalar[pl.INDEX],
        row_base_o_inline310__ssa_v0: pl.Scalar[pl.INDEX],
        o_packed_heads__ssa_v1: pl.Tensor[[3072, 4096], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 25165824)],
        wo_a__ssa_v0: pl.Tensor[[8, 1024, 4096], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 67108864)],
        g_inline293__idx_v0: pl.Scalar[pl.INDEX],
        o_r_pad_inline329__iter_v1: pl.Out[pl.Tensor[[384, 8192], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 12582912)]],
        out_col_g_inline306__ssa_v0: pl.Scalar[pl.INDEX],
    ) -> pl.Tensor[[384, 8192], pl.FP32]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        o_r_pad_inline329__ssa_v3: pl.Tensor[[384, 8192], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 12582912)] = self.proj_a_mm(
            t_dim_inline347__ssa_v0,
            row_base_o_inline310__ssa_v0,
            o_packed_heads__ssa_v1,
            wo_a__ssa_v0,
            g_inline293__idx_v0,
            o_r_pad_inline329__iter_v1,
            out_col_g_inline306__ssa_v0,
            attrs={"arg_directions": [pl.adir.scalar, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.scalar, pl.adir.output_existing, pl.adir.scalar]},
        )
        return o_r_pad_inline329__iter_v1

    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def proj_b_act(
        wo_b_scale__ssa_v0: pl.Tensor[[4096], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 16384)],
        attn_out__ssa_v0: pl.Out[pl.Tensor[[T_DYN, 4096], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)]],
        t_dim_inline347__ssa_v0: pl.Scalar[pl.INDEX],
        partials_inline317__rv_v2: pl.Tensor[[384, 32768], pl.INT32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 50331648)],
        act_scale_dq_inline313__rv_v2: pl.Tensor[[1, 384], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 1536)],
    ) -> pl.Tensor[[T_DYN, 4096], pl.BF16]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        mem_vec_4: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 2048)
        mem_vec_5: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
        mem_vec_6: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
        mem_vec_7: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
        act_idx_inline285__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        tblk_inline273__ssa_v0: pl.Scalar[pl.INDEX] = act_idx_inline285__ssa_v0 // 8
        nreg_inline302__ssa_v0: pl.Scalar[pl.INDEX] = act_idx_inline285__ssa_v0 - tblk_inline273__ssa_v0 * 8
        ob_n0_inline272__ssa_v0: pl.Scalar[pl.INDEX] = nreg_inline302__ssa_v0 * 512
        t0_inline339__ssa_v1: pl.Scalar[pl.INDEX] = tblk_inline273__ssa_v0 * 32
        wb_scale_inline271__tile: pl.Tile[[512], pl.FP32, pl.MemRef(mem_vec_4, pl.const(0, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.load(
            wo_b_scale__ssa_v0, [ob_n0_inline272__ssa_v0], [512], [512], target_memory=pl.Mem.Vec
        )
        wb_scale_chunk_inline319__tile: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_4, pl.const(0, pl.INT64), 2048), pl.Mem.Vec] = wb_scale_inline271__tile
        for b_tb_inline270__idx_v0, (attn_out__iter_v1,) in pl.range(t0_inline339__ssa_v1, pl.min(t0_inline339__ssa_v1 + 32, t_dim_inline347__ssa_v0), 8, init_values=(attn_out__ssa_v0,)):
            acc_i32_inline269__tile: pl.Tile[[8, 512], pl.INT32, pl.MemRef(mem_vec_5, pl.const(2048, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.full([8, 512], dtype=pl.INT32, value=0)
            for act_g_inline268__idx_v0, (acc_i32_inline269__iter_v1,) in pl.range(0, 8, 2, init_values=(acc_i32_inline269__tile,)):
                p_col0_inline308__ssa_v0: pl.Scalar[pl.INDEX] = act_g_inline268__idx_v0 * 4096 + ob_n0_inline272__ssa_v0
                p_col0_inline308__ssa_v0_1: pl.Scalar[pl.INDEX] = act_g_inline268__idx_v0 * 4096 + ob_n0_inline272__ssa_v0 + 4096
                p_g_inline300__tile: pl.Tile[[8, 512], pl.INT32, pl.MemRef(mem_vec_6, pl.const(18432, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.load(
                    partials_inline317__rv_v2, [b_tb_inline270__idx_v0, p_col0_inline308__ssa_v0], [8, 512], [8, 512], target_memory=pl.Mem.Vec
                )
                p_g_inline300__tile_1: pl.Tile[[8, 512], pl.INT32, pl.MemRef(mem_vec_7, pl.const(34816, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.load(
                    partials_inline317__rv_v2, [b_tb_inline270__idx_v0, p_col0_inline308__ssa_v0_1], [8, 512], [8, 512], target_memory=pl.Mem.Vec
                )
                acc_i32_inline269__tile_1: pl.Tile[[8, 512], pl.INT32, pl.MemRef(mem_vec_6, pl.const(18432, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.add(
                    acc_i32_inline269__iter_v1, p_g_inline300__tile
                )
                acc_i32_inline269__tile_2: pl.Tile[[8, 512], pl.INT32, pl.MemRef(mem_vec_5, pl.const(2048, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.add(
                    acc_i32_inline269__tile_1, p_g_inline300__tile_1
                )
                acc_i32_inline269__rv_v2: pl.Tile[[8, 512], pl.INT32, pl.MemRef(mem_vec_5, pl.const(2048, pl.INT64), 16384), pl.Mem.Vec] = pl.yield_(acc_i32_inline269__tile_2)
            t__tile: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_6, pl.const(18432, pl.INT64), 32), pl.Mem.Vec] = pl.tile.load(
                act_scale_dq_inline313__rv_v2, [0, b_tb_inline270__idx_v0], [1, 8], [1, 8], target_memory=pl.Mem.Vec
            )
            output_token_scale_inline267__tile: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_6, pl.const(18432, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(t__tile, [8, 1])
            t__tile_1: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_5, pl.const(2048, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.cast(acc_i32_inline269__rv_v2, target_type=pl.FP32, mode="round")
            acc_inline357__tile: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_5, pl.const(2048, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.col_expand_mul(t__tile_1, wb_scale_chunk_inline319__tile)
            out_t_inline266__tile: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_5, pl.const(2048, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.row_expand_mul(
                acc_inline357__tile, output_token_scale_inline267__tile
            )
            out_bf16_inline307__tile: pl.Tile[[8, 512], pl.BF16, pl.MemRef(mem_vec_5, pl.const(2048, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.cast(
                out_t_inline266__tile, target_type=pl.BF16, mode="rint"
            )
            attn_out__tile: pl.Tensor[[T_DYN, 4096], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                out_bf16_inline307__tile, [b_tb_inline270__idx_v0, ob_n0_inline272__ssa_v0], attn_out__iter_v1
            )
            attn_out__rv_v2: pl.Tensor[[T_DYN, 4096], pl.BF16, pl.MemRef("mem_ddr_16", pl.const(0, pl.INT64), 0)] = pl.yield_(attn_out__tile)
        return attn_out__ssa_v0

    @pl.function(type=pl.FunctionType.Spmd)
    def proj_b_act_spmd(
        self,
        wo_b_scale__ssa_v0: pl.Tensor[[4096], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 16384)],
        attn_out__ssa_v0: pl.Out[pl.Tensor[[T_DYN, 4096], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)]],
        t_dim_inline347__ssa_v0: pl.Scalar[pl.INDEX],
        partials_inline317__rv_v2: pl.Tensor[[384, 32768], pl.INT32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 50331648)],
        act_scale_dq_inline313__rv_v2: pl.Tensor[[1, 384], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 1536)],
    ) -> pl.Tensor[[T_DYN, 4096], pl.BF16]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        attn_out__rv_v2: pl.Tensor[[T_DYN, 4096], pl.BF16, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 0)] = self.proj_b_act(
            wo_b_scale__ssa_v0,
            attn_out__ssa_v0,
            t_dim_inline347__ssa_v0,
            partials_inline317__rv_v2,
            act_scale_dq_inline313__rv_v2,
            attrs={"arg_directions": [pl.adir.input, pl.adir.output_existing, pl.adir.scalar, pl.adir.input, pl.adir.input]},
        )
        return attn_out__ssa_v0

    @pl.function(type=pl.FunctionType.AIC, level=pl.Level.AIC, role=pl.Role.SubWorker)
    def proj_b_mm(
        partials_inline317__iter_v1: pl.Out[pl.Tensor[[384, 32768], pl.INT32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 50331648)]],
        col_g_inline295__ssa_v0: pl.Scalar[pl.INDEX],
        o_r_i8_pad_inline333__rv_v7: pl.Tensor[[384, 8192], pl.INT8, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 3145728)],
        wo_b__ssa_v0: pl.Tensor[[4096, 8192], pl.INT8, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 33554432)],
        g_inline360__idx_v0: pl.Scalar[pl.INDEX],
    ) -> pl.Tensor[[384, 32768], pl.INT32]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        mem_acc_3: pl.Ptr = pl.tile.alloc(pl.Mem.Acc, 131072)
        mem_mat_4: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 32768)
        mem_mat_5: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 65536)
        mem_mat_6: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 32768)
        mem_mat_7: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 65536)
        mem_left_8: pl.Ptr = pl.tile.alloc(pl.Mem.Left, 32768)
        mem_right_9: pl.Ptr = pl.tile.alloc(pl.Mem.Right, 65536)
        mem_left_10: pl.Ptr = pl.tile.alloc(pl.Mem.Left, 32768)
        pb_unit_inline286__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        tb_inline284__ssa_v0: pl.Scalar[pl.INDEX] = pb_unit_inline286__ssa_v0 // 8
        dc_inline283__ssa_v0: pl.Scalar[pl.INDEX] = pb_unit_inline286__ssa_v0 - tb_inline284__ssa_v0 * 8
        t0_inline339__ssa_v0: pl.Scalar[pl.INDEX] = tb_inline284__ssa_v0 * 128
        d0_inline337__ssa_v0: pl.Scalar[pl.INDEX] = dc_inline283__ssa_v0 * 512
        for nf_inline282__idx_v0, (partials_inline317__iter_v3,) in pl.range(2, init_values=(partials_inline317__iter_v1,)):
            n0_inline304__ssa_v1: pl.Scalar[pl.INDEX] = d0_inline337__ssa_v0 + nf_inline282__idx_v0 * 256
            acc_b_inline281__tile: pl.Tile[[128, 256], pl.INT32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 131072), pl.Mem.Acc] = pl.tile.create([128, 256], dtype=pl.INT32, target_memory=pl.Mem.Acc)
            for kb_inline280__idx_v0, (acc_b_inline281__iter_v1,) in pl.range(0, 4, 2, init_values=(acc_b_inline281__tile,)):
                k0_inline336__ssa_v1: pl.Scalar[pl.INDEX] = col_g_inline295__ssa_v0 + kb_inline280__idx_v0 * 256
                k0_inline336__ssa_v1_1: pl.Scalar[pl.INDEX] = col_g_inline295__ssa_v0 + (kb_inline280__idx_v0 * 256 + 256)
                b_act_inline279__tile: pl.Tile[[128, 256], pl.INT8, pl.MemRef(mem_mat_4, pl.const(0, pl.INT64), 32768), pl.Mem.Mat] = pl.tile.load(
                    o_r_i8_pad_inline333__rv_v7, [t0_inline339__ssa_v0, k0_inline336__ssa_v1], [128, 256], [128, 256], target_memory=pl.Mem.Mat
                )
                b_weight_inline278__tile: pl.Tile[[256, 256], pl.INT8, pl.MemRef(mem_mat_5, pl.const(32768, pl.INT64), 65536), pl.Mem.Mat] = pl.tile.load(
                    wo_b__ssa_v0, [n0_inline304__ssa_v1, k0_inline336__ssa_v1], [256, 256], [256, 256], target_memory=pl.Mem.Mat
                )
                b_act_inline279__tile_1: pl.Tile[[128, 256], pl.INT8, pl.MemRef(mem_mat_6, pl.const(98304, pl.INT64), 32768), pl.Mem.Mat] = pl.tile.load(
                    o_r_i8_pad_inline333__rv_v7, [t0_inline339__ssa_v0, k0_inline336__ssa_v1_1], [128, 256], [128, 256], target_memory=pl.Mem.Mat
                )
                b_weight_inline278__tile_1: pl.Tile[[256, 256], pl.INT8, pl.MemRef(mem_mat_7, pl.const(131072, pl.INT64), 65536), pl.Mem.Mat] = pl.tile.load(
                    wo_b__ssa_v0, [n0_inline304__ssa_v1, k0_inline336__ssa_v1_1], [256, 256], [256, 256], target_memory=pl.Mem.Mat
                )
                b_weight_inline278__tile_t: pl.Tile[
                    [256, 256], pl.INT8, pl.MemRef(mem_mat_5, pl.const(32768, pl.INT64), 65536), pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)
                ] = pl.tile.transpose_view(b_weight_inline278__tile)
                b_act_inline279__tile_Left: pl.Tile[[128, 256], pl.INT8, pl.MemRef(mem_left_8, pl.const(32768, pl.INT64), 32768), pl.Mem.Left] = pl.tile.move(
                    b_act_inline279__tile, target_memory=pl.Mem.Left
                )
                b_weight_inline278__tile_t_Right: pl.Tile[[256, 256], pl.INT8, pl.MemRef(mem_right_9, pl.const(0, pl.INT64), 65536), pl.Mem.Right] = pl.tile.move(
                    b_weight_inline278__tile_t, target_memory=pl.Mem.Right
                )
                acc_b_inline281__tile_1: pl.Tile[[128, 256], pl.INT32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 131072), pl.Mem.Acc] = pl.tile.matmul_acc(
                    acc_b_inline281__iter_v1, b_act_inline279__tile_Left, b_weight_inline278__tile_t_Right, kb_inline280__idx_v0 == 0
                )
                b_weight_inline278__tile_t_1: pl.Tile[
                    [256, 256], pl.INT8, pl.MemRef(mem_mat_7, pl.const(131072, pl.INT64), 65536), pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)
                ] = pl.tile.transpose_view(b_weight_inline278__tile_1)
                b_act_inline279__tile_Left_1: pl.Tile[[128, 256], pl.INT8, pl.MemRef(mem_left_10, pl.const(0, pl.INT64), 32768), pl.Mem.Left] = pl.tile.move(
                    b_act_inline279__tile_1, target_memory=pl.Mem.Left
                )
                b_weight_inline278__tile_t_Right_1: pl.Tile[[256, 256], pl.INT8, pl.MemRef(mem_right_9, pl.const(0, pl.INT64), 65536), pl.Mem.Right] = pl.tile.move(
                    b_weight_inline278__tile_t_1, target_memory=pl.Mem.Right
                )
                acc_b_inline281__tile_2: pl.Tile[[128, 256], pl.INT32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 131072), pl.Mem.Acc] = pl.tile.matmul_acc(
                    acc_b_inline281__tile_1, b_act_inline279__tile_Left_1, b_weight_inline278__tile_t_Right_1, kb_inline280__idx_v0 == -1
                )
                acc_b_inline281__rv_v2: pl.Tile[[128, 256], pl.INT32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 131072), pl.Mem.Acc] = pl.yield_(acc_b_inline281__tile_2)
            partials_inline317__tile: pl.Tensor[[384, 32768], pl.INT32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 50331648)] = pl.tile.store(
                acc_b_inline281__rv_v2, [t0_inline339__ssa_v0, g_inline360__idx_v0 * 4096 + n0_inline304__ssa_v1], partials_inline317__iter_v3
            )
            partials_inline317__rv_v4: pl.Tensor[[384, 32768], pl.INT32, pl.MemRef("mem_ddr_13", pl.const(0, pl.INT64), 50331648)] = pl.yield_(partials_inline317__tile)
        return partials_inline317__iter_v1

    @pl.function(type=pl.FunctionType.Spmd)
    def proj_b_mm_spmd(
        self,
        partials_inline317__iter_v1: pl.Out[pl.Tensor[[384, 32768], pl.INT32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 50331648)]],
        col_g_inline295__ssa_v0: pl.Scalar[pl.INDEX],
        o_r_i8_pad_inline333__rv_v7: pl.Tensor[[384, 8192], pl.INT8, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 3145728)],
        wo_b__ssa_v0: pl.Tensor[[4096, 8192], pl.INT8, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 33554432)],
        g_inline360__idx_v0: pl.Scalar[pl.INDEX],
    ) -> pl.Tensor[[384, 32768], pl.INT32]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        partials_inline317__rv_v4: pl.Tensor[[384, 32768], pl.INT32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 50331648)] = self.proj_b_mm(
            partials_inline317__iter_v1,
            col_g_inline295__ssa_v0,
            o_r_i8_pad_inline333__rv_v7,
            wo_b__ssa_v0,
            g_inline360__idx_v0,
            attrs={"arg_directions": [pl.adir.output_existing, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.scalar]},
        )
        return partials_inline317__iter_v1

    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def q_rope_prepare(
        rope_sin_signed_view_inline1763__ssa_v0: pl.Out[pl.Tensor[[t_dim_inline1770__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)]],
        rope_swap_idx_view_inline1774__ssa_v0: pl.Out[pl.Tensor[[t_dim_inline1770__ssa_v0, 64], pl.INT32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)]],
        token_tiles_inline1782__ssa_v0: pl.Scalar[pl.INDEX],
        t_dim_inline1770__ssa_v0: pl.Scalar[pl.INDEX],
        rope_sin_view_inline1767__ssa_v0: pl.Tensor[[t_dim_inline1770__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)],
    ):
        pl.func_attr({"mx_tensor_views_blocked": True})
        mem_vec_3: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 2048)
        mem_vec_8: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 2048)
        mem_vec_14: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 2048)
        mem_vec_22: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 2048)
        mem_vec_27: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 2048)
        mem_vec_33: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 2048)
        qrp_worker_inline1765__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        qrp_ones_inline1766__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_3, pl.const(6144, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.full([8, 64], dtype=pl.FP32, value=1.0)
        qrp_idx_i32_inline1764__ci_tmp_v0: pl.Tile[[1, 192], pl.FP32, pl.MemRef(mem_vec_8, pl.const(10240, pl.INT64), 768), pl.Mem.Vec] = pl.tile.create(
            [1, 192], dtype=pl.FP32, target_memory=pl.Mem.Vec
        )
        qrp_idx_i32_inline1764__tile: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_14, pl.const(0, pl.INT64), 256), pl.Mem.Vec] = pl.tile.ci(
            pl.const(0, pl.INT32), [1, 64], tmp=qrp_idx_i32_inline1764__ci_tmp_v0, dtype=pl.INT32, descending=False
        )
        qrp_idx_fp32_inline1772__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_8, pl.const(10240, pl.INT64), 256), pl.Mem.Vec] = pl.tile.cast(
            qrp_idx_i32_inline1764__tile, target_type=pl.FP32, mode="round"
        )
        qrp_col_inline1760__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_3, pl.const(6144, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.col_expand_mul(
            qrp_ones_inline1766__tile, qrp_idx_fp32_inline1772__tile
        )
        qrp_half_inline1759__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_8, pl.const(10240, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.muls(qrp_col_inline1760__tile, 0.5)
        qrp_dup_i32_inline1761__tile: pl.Tile[[8, 64], pl.INT32, pl.MemRef(mem_vec_8, pl.const(10240, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(
            qrp_half_inline1759__tile, target_type=pl.INT32, mode="trunc"
        )
        qrp_dup_f_inline1777__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_8, pl.const(10240, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(
            qrp_dup_i32_inline1761__tile, target_type=pl.FP32, mode="round"
        )
        t__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_8, pl.const(10240, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.muls(qrp_dup_f_inline1777__tile, 2.0)
        qrp_lane_inline1773__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_8, pl.const(10240, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.sub(qrp_col_inline1760__tile, t__tile)
        qrp_next_col_inline1785__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_3, pl.const(6144, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.adds(qrp_col_inline1760__tile, 1.0)
        qrp_lane_offset_inline1775__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_14, pl.const(0, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.muls(qrp_lane_inline1773__tile, 2.0)
        qrp_swap_f_inline1771__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_3, pl.const(6144, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.sub(
            qrp_next_col_inline1785__tile, qrp_lane_offset_inline1775__tile
        )
        qrp_swap_idx_inline1776__tile: pl.Tile[[8, 64], pl.INT32, pl.MemRef(mem_vec_3, pl.const(6144, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(
            qrp_swap_f_inline1771__tile, target_type=pl.INT32, mode="round"
        )
        t__tile_1: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_8, pl.const(10240, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.muls(qrp_lane_inline1773__tile, 2.0)
        qrp_sign_inline1780__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_8, pl.const(10240, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.subs(t__tile_1, 1.0)
        for qrp_idx_inline1769__idx_v0, (rope_sin_signed_view_inline1763__iter_v1, rope_swap_idx_view_inline1774__iter_v1) in pl.range(
            qrp_worker_inline1765__ssa_v0,
            token_tiles_inline1782__ssa_v0,
            pl.min(token_tiles_inline1782__ssa_v0, 48),
            init_values=(rope_sin_signed_view_inline1763__ssa_v0, rope_swap_idx_view_inline1774__ssa_v0),
        ):
            qrp_t0_inline1768__ssa_v0: pl.Scalar[pl.INDEX] = qrp_idx_inline1769__idx_v0 * 8
            qrp_valid_rows_inline1786__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(t_dim_inline1770__ssa_v0 - qrp_t0_inline1768__ssa_v0, 8)
            if qrp_valid_rows_inline1786__ssa_v0 == 8:
                qrp_sin_rows_full_inline1784__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_14, pl.const(0, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.load(
                    rope_sin_view_inline1767__ssa_v0, [qrp_t0_inline1768__ssa_v0, 0], [8, 64], [8, 64], target_memory=pl.Mem.Vec
                )
                qrp_sin_signed_full_inline1779__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_14, pl.const(0, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.mul(
                    qrp_sin_rows_full_inline1784__tile, qrp_sign_inline1780__tile
                )
                rope_sin_signed_view_inline1763__tile: pl.Tensor[[t_dim_inline1770__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                    qrp_sin_signed_full_inline1779__tile, [qrp_t0_inline1768__ssa_v0, 0], rope_sin_signed_view_inline1763__iter_v1
                )
                rope_swap_idx_view_inline1774__tile: pl.Tensor[[t_dim_inline1770__ssa_v0, 64], pl.INT32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                    qrp_swap_idx_inline1776__tile, [qrp_t0_inline1768__ssa_v0, 0], rope_swap_idx_view_inline1774__iter_v1
                )
                rope_sin_signed_view_inline1763__phi_v4, rope_swap_idx_view_inline1774__phi_v4 = pl.yield_(rope_sin_signed_view_inline1763__tile, rope_swap_idx_view_inline1774__tile)
            else:
                qrp_sin_rows_tail_inline1781__ssa_v0: pl.Tile[
                    [8, 64], pl.FP32, pl.MemRef(mem_vec_14, pl.const(0, pl.INT64), 2048), pl.Mem.Vec, pl.TileView(valid_shape=[qrp_valid_rows_inline1786__ssa_v0, 64])
                ] = pl.tile.load(rope_sin_view_inline1767__ssa_v0, [qrp_t0_inline1768__ssa_v0, 0], [8, 64], [qrp_valid_rows_inline1786__ssa_v0, 64], target_memory=pl.Mem.Vec)
                t__tmp_v11: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_22, pl.const(2048, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.full([8, 64], dtype=pl.FP32, value=1.0)
                t__ci_tmp_v1: pl.Tile[[1, 192], pl.FP32, pl.MemRef(mem_vec_27, pl.const(4096, pl.INT64), 768), pl.Mem.Vec] = pl.tile.create([1, 192], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                t__tmp_v12: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_33, pl.const(8192, pl.INT64), 256), pl.Mem.Vec] = pl.tile.ci(
                    pl.const(0, pl.INT32), [1, 64], tmp=t__ci_tmp_v1, dtype=pl.INT32, descending=False
                )
                t__tmp_v13: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_27, pl.const(4096, pl.INT64), 256), pl.Mem.Vec] = pl.tile.cast(t__tmp_v12, target_type=pl.FP32, mode="round")
                qrp_tail_col_inline1778__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_22, pl.const(2048, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.col_expand_mul(t__tmp_v11, t__tmp_v13)
                t__tmp_v14: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_27, pl.const(4096, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.muls(qrp_tail_col_inline1778__ssa_v0, 0.5)
                t__tmp_v15: pl.Tile[[8, 64], pl.INT32, pl.MemRef(mem_vec_27, pl.const(4096, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(t__tmp_v14, target_type=pl.INT32, mode="trunc")
                qrp_tail_dup_f_inline1787__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_27, pl.const(4096, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(
                    t__tmp_v15, target_type=pl.FP32, mode="round"
                )
                t__tmp_v16: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_27, pl.const(4096, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.muls(qrp_tail_dup_f_inline1787__ssa_v0, 2.0)
                qrp_tail_lane_inline1783__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_27, pl.const(4096, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.sub(
                    qrp_tail_col_inline1778__ssa_v0, t__tmp_v16
                )
                t__tmp_v17: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_22, pl.const(2048, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.adds(qrp_tail_col_inline1778__ssa_v0, 1.0)
                t__tmp_v18: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_33, pl.const(8192, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.muls(qrp_tail_lane_inline1783__ssa_v0, 2.0)
                qrp_tail_swap_f_inline1788__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_22, pl.const(2048, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.sub(t__tmp_v17, t__tmp_v18)
                t__tmp_v19: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_27, pl.const(4096, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.muls(qrp_tail_lane_inline1783__ssa_v0, 2.0)
                qrp_tail_sign_inline1762__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_27, pl.const(4096, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.subs(t__tmp_v19, 1.0)
                qrp_sin_signed_tail_inline1758__ssa_v0: pl.Tile[
                    [8, 64], pl.FP32, pl.MemRef(mem_vec_14, pl.const(0, pl.INT64), 2048), pl.Mem.Vec, pl.TileView(valid_shape=[qrp_valid_rows_inline1786__ssa_v0, 64])
                ] = pl.tile.mul(qrp_sin_rows_tail_inline1781__ssa_v0, qrp_tail_sign_inline1762__ssa_v0)
                t__tmp_v20: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_14, pl.const(0, pl.INT64), 2048), pl.Mem.Vec, pl.TileView(valid_shape=[qrp_valid_rows_inline1786__ssa_v0, 64])] = (
                    pl.tile.set_validshape(qrp_sin_signed_tail_inline1758__ssa_v0, qrp_valid_rows_inline1786__ssa_v0, 64)
                )
                pl.tile.store(t__tmp_v20, [qrp_t0_inline1768__ssa_v0, 0], rope_sin_signed_view_inline1763__iter_v1)
                t__tmp_v21: pl.Tile[[8, 64], pl.INT32, pl.MemRef(mem_vec_14, pl.const(0, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(
                    qrp_tail_swap_f_inline1788__ssa_v0, target_type=pl.INT32, mode="round"
                )
                t__tmp_v22: pl.Tile[[8, 64], pl.INT32, pl.MemRef(mem_vec_14, pl.const(0, pl.INT64), 2048), pl.Mem.Vec, pl.TileView(valid_shape=[qrp_valid_rows_inline1786__ssa_v0, 64])] = (
                    pl.tile.set_validshape(t__tmp_v21, qrp_valid_rows_inline1786__ssa_v0, 64)
                )
                pl.tile.store(t__tmp_v22, [qrp_t0_inline1768__ssa_v0, 0], rope_swap_idx_view_inline1774__iter_v1)
                rope_sin_signed_view_inline1763__phi_v4, rope_swap_idx_view_inline1774__phi_v4 = pl.yield_(rope_sin_signed_view_inline1763__iter_v1, rope_swap_idx_view_inline1774__iter_v1)
            rope_sin_signed_view_inline1763__rv_v2, rope_swap_idx_view_inline1774__rv_v2 = pl.yield_(rope_sin_signed_view_inline1763__phi_v4, rope_swap_idx_view_inline1774__phi_v4)
        return

    @pl.function(type=pl.FunctionType.Spmd)
    def q_rope_prepare_spmd(
        self,
        rope_sin_signed_view_inline1763__ssa_v0: pl.Out[pl.Tensor[[t_dim_inline1770__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)]],
        rope_swap_idx_view_inline1774__ssa_v0: pl.Out[pl.Tensor[[t_dim_inline1770__ssa_v0, 64], pl.INT32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)]],
        token_tiles_inline1782__ssa_v0: pl.Scalar[pl.INDEX],
        t_dim_inline1770__ssa_v0: pl.Scalar[pl.INDEX],
        rope_sin_view_inline1767__ssa_v0: pl.Tensor[[t_dim_inline1770__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)],
    ):
        pl.func_attr({"mx_tensor_views_blocked": True})
        self.q_rope_prepare(
            rope_sin_signed_view_inline1763__ssa_v0,
            rope_swap_idx_view_inline1774__ssa_v0,
            token_tiles_inline1782__ssa_v0,
            t_dim_inline1770__ssa_v0,
            rope_sin_view_inline1767__ssa_v0,
            attrs={"arg_directions": [pl.adir.output_existing, pl.adir.output_existing, pl.adir.scalar, pl.adir.scalar, pl.adir.input]},
        )

    @pl.function(type=pl.FunctionType.AIC, level=pl.Level.AIC, role=pl.Role.SubWorker)
    def qk_pv_aic(
        ffts_workspace_inline2438__ssa_v0: pl.Tensor[[256], pl.INT64, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 2048)],
        t_dim_inline2491__ssa_v0: pl.Scalar[pl.INDEX],
        q_flat_inline2480__ssa_v0: pl.Tensor[[t_heads_inline2469__ssa_v0, 512], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)],
        valid_block_mask_inline2452__ssa_v0: pl.Tensor[[t_dim_inline2491__ssa_v0, 16], pl.INT32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)],
        kv_transfer_inline2426__ssa_v0: pl.InOut[pl.Tensor[[12288, 512], pl.BF16, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 12582912)]],
        score_transfer_inline2447__ssa_v0: pl.InOut[pl.Tensor[[1536, 512], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 3145728)]],
        probability_transfer_inline2550__ssa_v0: pl.InOut[pl.Tensor[[1536, 512], pl.BF16, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 1572864)]],
        pv_transfer_inline2437__ssa_v0: pl.InOut[pl.Tensor[[1536, 512], pl.FP32, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 3145728)]],
        attn_sink_col_inline2429__ssa_v0: pl.Tensor[[64, 1], pl.FP32, pl.MemRef("mem_ddr_7", pl.const(0, pl.INT64), 256)],
        mi_transfer_inline2500__ssa_v0: pl.InOut[pl.Tensor[[1536, 1], pl.FP32, pl.MemRef("mem_ddr_8", pl.const(0, pl.INT64), 6144)]],
        li_transfer_inline2502__ssa_v0: pl.InOut[pl.Tensor[[1536, 1], pl.FP32, pl.MemRef("mem_ddr_9", pl.const(0, pl.INT64), 6144)]],
        position_ids_t1__ssa_v0: pl.Tensor[[T_DYN, 1], pl.INT64, pl.MemRef("mem_ddr_10", pl.const(0, pl.INT64), 0)],
        ori_block_table__ssa_v0: pl.Tensor[[B_DYN, ORIGINAL_TABLE_COLUMNS_DYN], pl.INT32, pl.MemRef("mem_ddr_11", pl.const(0, pl.INT64), 0)],
        ori_kv_flat_inline2493__ssa_v1: pl.Tensor[[ori_block_num_inline2459__ssa_v0 * pl.const(32, pl.INDEX), 512], pl.BF16, pl.MemRef("mem_ddr_12", pl.const(0, pl.INT64), 0)],
        cmp_sparse_indices_inline2484__rv_v2: pl.Tensor[[t_dim_inline2491__ssa_v0, 512], pl.INT32, pl.MemRef("mem_ddr_13", pl.const(0, pl.INT64), 0)],
        cmp_block_table__ssa_v0: pl.Tensor[[B_DYN, COMPRESSED_TABLE_COLUMNS_DYN], pl.INT32, pl.MemRef("mem_ddr_14", pl.const(0, pl.INT64), 0)],
        cmp_kv_flat_inline2430__ssa_v0: pl.Tensor[[cmp_block_num_inline2525__ssa_v0 * pl.const(32, pl.INDEX), 512], pl.BF16, pl.MemRef("mem_ddr_15", pl.const(0, pl.INT64), 0)],
        sparse_bias_inline2455__rv_v2: pl.Tensor[[t_dim_inline2491__ssa_v0, 1024], pl.FP32, pl.MemRef("mem_ddr_16", pl.const(0, pl.INT64), 0)],
        alpha_transfer_inline2486__ssa_v0: pl.InOut[pl.Tensor[[1536, 1], pl.FP32, pl.MemRef("mem_ddr_17", pl.const(0, pl.INT64), 6144)]],
        attn_mi_inline2490__ssa_v0: pl.Out[pl.Tensor[[t_heads_inline2469__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_18", pl.const(0, pl.INT64), 0)]],
        attn_li_inline2516__ssa_v0: pl.Out[pl.Tensor[[t_heads_inline2469__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_19", pl.const(0, pl.INT64), 0)]],
        attn_oi_inline2428__ssa_v0: pl.Out[pl.Tensor[[t_heads_inline2469__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_20", pl.const(0, pl.INT64), 0)]],
    ):
        pl.func_attr({"mx_tensor_views_blocked": True, "split_aiv": True})
        mem_mat_21: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 65536)
        mem_mat_22: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 131072)
        mem_left_24: pl.Ptr = pl.tile.alloc(pl.Mem.Left, 16384)
        mem_right_25: pl.Ptr = pl.tile.alloc(pl.Mem.Right, 32768)
        mem_left_26: pl.Ptr = pl.tile.alloc(pl.Mem.Left, 16384)
        mem_right_27: pl.Ptr = pl.tile.alloc(pl.Mem.Right, 32768)
        mem_acc_29: pl.Ptr = pl.tile.alloc(pl.Mem.Acc, 131072)
        mem_mat_30: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 16384)
        qk_core_inline2468__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        qk_kv_base_inline2504__ssa_v0: pl.Scalar[pl.INDEX] = qk_core_inline2468__ssa_v0 * 512
        qk_head_base_inline2505__ssa_v0: pl.Scalar[pl.INDEX] = qk_core_inline2468__ssa_v0 * 64
        pl.system.set_ffts(ffts_workspace_inline2438__ssa_v0)
        for qk_t_inline2506__idx_v0 in pl.range(qk_core_inline2468__ssa_v0, t_dim_inline2491__ssa_v0, 24):
            qk_q_inline2508__ssa_v0: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_mat_21, pl.const(0, pl.INT64), 65536), pl.Mem.Mat] = pl.tile.load(
                q_flat_inline2480__ssa_v0, [qk_t_inline2506__idx_v0 * 64, 0], [64, 512], [64, 512], target_memory=pl.Mem.Mat
            )
            for qk_sb_inline2509__idx_v0 in pl.range(2):
                t__tile: pl.Scalar[pl.INT32] = pl.tensor.read(valid_block_mask_inline2452__ssa_v0, [qk_t_inline2506__idx_v0, qk_sb_inline2509__idx_v0])
                if 0 < pl.cast(t__tile, pl.INDEX):
                    pl.system.sync_wait(0, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIC)
                    for qk_part_inline2512__idx_v0 in pl.range(4):
                        qk_col_inline2515__ssa_v0: pl.Scalar[pl.INDEX] = qk_part_inline2512__idx_v0 * 128
                        qk_kv_inline2517__ssa_v0: pl.Tile[[128, 512], pl.BF16, pl.MemRef(mem_mat_22, pl.const(65536, pl.INT64), 131072), pl.Mem.Mat] = pl.tile.load(
                            kv_transfer_inline2426__ssa_v0, [qk_kv_base_inline2504__ssa_v0 + qk_col_inline2515__ssa_v0, 0], [128, 512], [128, 512], target_memory=pl.Mem.Mat
                        )
                        t__tmp_v311: pl.Tile[
                            [512, 128], pl.BF16, pl.MemRef(mem_mat_22, pl.const(65536, pl.INT64), 131072), pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)
                        ] = pl.tile.transpose_view(qk_kv_inline2517__ssa_v0)
                        qk_scores_inline2453__ssa_v0_l0_init: pl.Tile[[64, 128], pl.FP32, pl.MemRef(mem_acc_29, pl.const(0, pl.INT64), 32768), pl.Mem.Acc] = pl.tile.create(
                            [64, 128], dtype=pl.FP32, target_memory=pl.Mem.Acc
                        )
                        for qk_scores_inline2453__ssa_v0_l0_ko, (qk_scores_inline2453__ssa_v0_l0_c,) in pl.range(0, 512, 256, init_values=(qk_scores_inline2453__ssa_v0_l0_init,)):
                            qk_scores_inline2453__ssa_v0_l0_a: pl.Tile[
                                [64, 128], pl.BF16, pl.MemRef(mem_left_24, pl.const(0, pl.INT64), 16384), pl.Mem.Left, pl.TileView(blayout=pl.TileLayout.row_major)
                            ] = pl.tile.extract(qk_q_inline2508__ssa_v0, 0, qk_scores_inline2453__ssa_v0_l0_ko, [64, 128], target_memory=pl.Mem.Left)
                            qk_scores_inline2453__ssa_v0_l0_b: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_25, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                                t__tmp_v311, qk_scores_inline2453__ssa_v0_l0_ko, 0, [128, 128], target_memory=pl.Mem.Right
                            )
                            qk_scores_inline2453__ssa_v0_l0_a_1: pl.Tile[
                                [64, 128], pl.BF16, pl.MemRef(mem_left_26, pl.const(16384, pl.INT64), 16384), pl.Mem.Left, pl.TileView(blayout=pl.TileLayout.row_major)
                            ] = pl.tile.extract(qk_q_inline2508__ssa_v0, 0, qk_scores_inline2453__ssa_v0_l0_ko + 128, [64, 128], target_memory=pl.Mem.Left)
                            qk_scores_inline2453__ssa_v0_l0_b_1: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_27, pl.const(32768, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                                t__tmp_v311, qk_scores_inline2453__ssa_v0_l0_ko + 128, 0, [128, 128], target_memory=pl.Mem.Right
                            )
                            qk_scores_inline2453__ssa_v0_l0_c_acc: pl.Tile[[64, 128], pl.FP32, pl.MemRef(mem_acc_29, pl.const(0, pl.INT64), 32768), pl.Mem.Acc] = pl.tile.matmul_acc(
                                qk_scores_inline2453__ssa_v0_l0_c, qk_scores_inline2453__ssa_v0_l0_a, qk_scores_inline2453__ssa_v0_l0_b, qk_scores_inline2453__ssa_v0_l0_ko == 0
                            )
                            qk_scores_inline2453__ssa_v0_l0_c_acc_1: pl.Tile[[64, 128], pl.FP32, pl.MemRef(mem_acc_29, pl.const(0, pl.INT64), 32768), pl.Mem.Acc] = pl.tile.matmul_acc(
                                qk_scores_inline2453__ssa_v0_l0_c_acc, qk_scores_inline2453__ssa_v0_l0_a_1, qk_scores_inline2453__ssa_v0_l0_b_1, qk_scores_inline2453__ssa_v0_l0_ko == -128
                            )
                            qk_scores_inline2453__ssa_v0: pl.Tile[[64, 128], pl.FP32, pl.MemRef(mem_acc_29, pl.const(0, pl.INT64), 32768), pl.Mem.Acc] = pl.yield_(
                                qk_scores_inline2453__ssa_v0_l0_c_acc_1
                            )
                        score_transfer_inline2447__store: pl.Tensor[[1536, 512], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 3145728)] = pl.tile.store(
                            qk_scores_inline2453__ssa_v0, [qk_head_base_inline2505__ssa_v0, qk_col_inline2515__ssa_v0], score_transfer_inline2447__ssa_v0
                        )
                    pl.system.sync_set(1, pipe=pl.PipeType.FIX, ffts_mode=2, core_type=pl.KernelType.AIC)
                    pl.system.sync_wait(2, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIC)
                    pv_acc_inline2519__ssa_v0: pl.Tile[[64, 512], pl.FP32, pl.MemRef(mem_acc_29, pl.const(0, pl.INT64), 131072), pl.Mem.Acc] = pl.tile.create(
                        [64, 512], dtype=pl.FP32, target_memory=pl.Mem.Acc
                    )
                    for pv_part_inline2464__idx_v0, (pv_acc_inline2519__iter_v1,) in pl.range(4, init_values=(pv_acc_inline2519__ssa_v0,)):
                        pv_col_inline2527__ssa_v0: pl.Scalar[pl.INDEX] = pv_part_inline2464__idx_v0 * 128
                        pv_probability_inline2520__ssa_v0: pl.Tile[[64, 128], pl.BF16, pl.MemRef(mem_mat_30, pl.const(196608, pl.INT64), 16384), pl.Mem.Mat] = pl.tile.load(
                            probability_transfer_inline2550__ssa_v0, [qk_head_base_inline2505__ssa_v0, pv_col_inline2527__ssa_v0], [64, 128], [64, 128], target_memory=pl.Mem.Mat
                        )
                        pv_kv_inline2482__ssa_v0: pl.Tile[[128, 512], pl.BF16, pl.MemRef(mem_mat_22, pl.const(65536, pl.INT64), 131072), pl.Mem.Mat] = pl.tile.load(
                            kv_transfer_inline2426__ssa_v0, [qk_kv_base_inline2504__ssa_v0 + pv_col_inline2527__ssa_v0, 0], [128, 512], [128, 512], target_memory=pl.Mem.Mat
                        )
                        for pv_acc_inline2519__ssa_v3_l0_ko, (pv_acc_inline2519__ssa_v3_l0_c,) in pl.range(0, 128, 64, init_values=(pv_acc_inline2519__iter_v1,)):
                            pv_acc_inline2519__ssa_v3_l0_a: pl.Tile[
                                [64, 32], pl.BF16, pl.MemRef(mem_left_24, pl.const(0, pl.INT64), 4096), pl.Mem.Left, pl.TileView(blayout=pl.TileLayout.row_major)
                            ] = pl.tile.extract(pv_probability_inline2520__ssa_v0, 0, pv_acc_inline2519__ssa_v3_l0_ko, [64, 32], target_memory=pl.Mem.Left)
                            pv_acc_inline2519__ssa_v3_l0_b: pl.Tile[[32, 512], pl.BF16, pl.MemRef(mem_right_25, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                                pv_kv_inline2482__ssa_v0, pv_acc_inline2519__ssa_v3_l0_ko, 0, [32, 512], target_memory=pl.Mem.Right
                            )
                            pv_acc_inline2519__ssa_v3_l0_a_1: pl.Tile[
                                [64, 32], pl.BF16, pl.MemRef(mem_left_26, pl.const(16384, pl.INT64), 4096), pl.Mem.Left, pl.TileView(blayout=pl.TileLayout.row_major)
                            ] = pl.tile.extract(pv_probability_inline2520__ssa_v0, 0, pv_acc_inline2519__ssa_v3_l0_ko + 32, [64, 32], target_memory=pl.Mem.Left)
                            pv_acc_inline2519__ssa_v3_l0_b_1: pl.Tile[[32, 512], pl.BF16, pl.MemRef(mem_right_27, pl.const(32768, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                                pv_kv_inline2482__ssa_v0, pv_acc_inline2519__ssa_v3_l0_ko + 32, 0, [32, 512], target_memory=pl.Mem.Right
                            )
                            pv_acc_inline2519__ssa_v3_l0_c_acc: pl.Tile[[64, 512], pl.FP32, pl.MemRef(mem_acc_29, pl.const(0, pl.INT64), 131072), pl.Mem.Acc] = pl.tile.matmul_acc(
                                pv_acc_inline2519__ssa_v3_l0_c, pv_acc_inline2519__ssa_v3_l0_a, pv_acc_inline2519__ssa_v3_l0_b, pv_part_inline2464__idx_v0 == 0 and pv_acc_inline2519__ssa_v3_l0_ko == 0
                            )
                            pv_acc_inline2519__ssa_v3_l0_c_acc_1: pl.Tile[[64, 512], pl.FP32, pl.MemRef(mem_acc_29, pl.const(0, pl.INT64), 131072), pl.Mem.Acc] = pl.tile.matmul_acc(
                                pv_acc_inline2519__ssa_v3_l0_c_acc,
                                pv_acc_inline2519__ssa_v3_l0_a_1,
                                pv_acc_inline2519__ssa_v3_l0_b_1,
                                pv_part_inline2464__idx_v0 == 0 and pv_acc_inline2519__ssa_v3_l0_ko == -32,
                            )
                            pv_acc_inline2519__ssa_v3: pl.Tile[[64, 512], pl.FP32, pl.MemRef(mem_acc_29, pl.const(0, pl.INT64), 131072), pl.Mem.Acc] = pl.yield_(pv_acc_inline2519__ssa_v3_l0_c_acc_1)
                        pv_acc_inline2519__rv_v2: pl.Tile[[64, 512], pl.FP32, pl.MemRef(mem_acc_29, pl.const(0, pl.INT64), 131072), pl.Mem.Acc] = pl.yield_(pv_acc_inline2519__ssa_v3)
                    pv_transfer_inline2437__store: pl.Tensor[[1536, 512], pl.FP32, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 3145728)] = pl.tile.store(
                        pv_acc_inline2519__rv_v2, [qk_head_base_inline2505__ssa_v0, 0], pv_transfer_inline2437__ssa_v0
                    )
                    pl.system.sync_set(3, pipe=pl.PipeType.FIX, ffts_mode=2, core_type=pl.KernelType.AIC)
            pl.system.set_ffts(ffts_workspace_inline2438__ssa_v0)

    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def qk_pv_aiv(
        ffts_workspace_inline2438__ssa_v0: pl.Tensor[[256], pl.INT64, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 2048)],
        t_dim_inline2491__ssa_v0: pl.Scalar[pl.INDEX],
        q_flat_inline2480__ssa_v0: pl.Tensor[[t_heads_inline2469__ssa_v0, 512], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)],
        valid_block_mask_inline2452__ssa_v0: pl.Tensor[[t_dim_inline2491__ssa_v0, 16], pl.INT32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)],
        kv_transfer_inline2426__ssa_v0: pl.InOut[pl.Tensor[[12288, 512], pl.BF16, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 12582912)]],
        score_transfer_inline2447__ssa_v0: pl.InOut[pl.Tensor[[1536, 512], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 3145728)]],
        probability_transfer_inline2550__ssa_v0: pl.InOut[pl.Tensor[[1536, 512], pl.BF16, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 1572864)]],
        pv_transfer_inline2437__ssa_v0: pl.InOut[pl.Tensor[[1536, 512], pl.FP32, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 3145728)]],
        attn_sink_col_inline2429__ssa_v0: pl.Tensor[[64, 1], pl.FP32, pl.MemRef("mem_ddr_7", pl.const(0, pl.INT64), 256)],
        mi_transfer_inline2500__ssa_v0: pl.InOut[pl.Tensor[[1536, 1], pl.FP32, pl.MemRef("mem_ddr_8", pl.const(0, pl.INT64), 6144)]],
        li_transfer_inline2502__ssa_v0: pl.InOut[pl.Tensor[[1536, 1], pl.FP32, pl.MemRef("mem_ddr_9", pl.const(0, pl.INT64), 6144)]],
        position_ids_t1__ssa_v0: pl.Tensor[[T_DYN, 1], pl.INT64, pl.MemRef("mem_ddr_10", pl.const(0, pl.INT64), 0)],
        ori_block_table__ssa_v0: pl.Tensor[[B_DYN, ORIGINAL_TABLE_COLUMNS_DYN], pl.INT32, pl.MemRef("mem_ddr_11", pl.const(0, pl.INT64), 0)],
        ori_kv_flat_inline2493__ssa_v1: pl.Tensor[[ori_block_num_inline2459__ssa_v0 * pl.const(32, pl.INDEX), 512], pl.BF16, pl.MemRef("mem_ddr_12", pl.const(0, pl.INT64), 0)],
        cmp_sparse_indices_inline2484__rv_v2: pl.Tensor[[t_dim_inline2491__ssa_v0, 512], pl.INT32, pl.MemRef("mem_ddr_13", pl.const(0, pl.INT64), 0)],
        cmp_block_table__ssa_v0: pl.Tensor[[B_DYN, COMPRESSED_TABLE_COLUMNS_DYN], pl.INT32, pl.MemRef("mem_ddr_14", pl.const(0, pl.INT64), 0)],
        cmp_kv_flat_inline2430__ssa_v0: pl.Tensor[[cmp_block_num_inline2525__ssa_v0 * pl.const(32, pl.INDEX), 512], pl.BF16, pl.MemRef("mem_ddr_15", pl.const(0, pl.INT64), 0)],
        sparse_bias_inline2455__rv_v2: pl.Tensor[[t_dim_inline2491__ssa_v0, 1024], pl.FP32, pl.MemRef("mem_ddr_16", pl.const(0, pl.INT64), 0)],
        alpha_transfer_inline2486__ssa_v0: pl.InOut[pl.Tensor[[1536, 1], pl.FP32, pl.MemRef("mem_ddr_17", pl.const(0, pl.INT64), 6144)]],
        attn_mi_inline2490__ssa_v0: pl.Out[pl.Tensor[[t_heads_inline2469__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_18", pl.const(0, pl.INT64), 0)]],
        attn_li_inline2516__ssa_v0: pl.Out[pl.Tensor[[t_heads_inline2469__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_19", pl.const(0, pl.INT64), 0)]],
        attn_oi_inline2428__ssa_v0: pl.Out[pl.Tensor[[t_heads_inline2469__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_20", pl.const(0, pl.INT64), 0)]],
    ) -> tuple[
        pl.Tensor[[12288, 512], pl.BF16],
        pl.Tensor[[1536, 512], pl.FP32],
        pl.Tensor[[1536, 512], pl.BF16],
        pl.Tensor[[1536, 512], pl.FP32],
        pl.Tensor[[1536, 1], pl.FP32],
        pl.Tensor[[1536, 1], pl.FP32],
        pl.Tensor[[1536, 1], pl.FP32],
        pl.Tensor[[t_heads_inline2469__ssa_v0, 1], pl.FP32],
        pl.Tensor[[t_heads_inline2469__ssa_v0, 1], pl.FP32],
        pl.Tensor[[t_heads_inline2469__ssa_v0, 512], pl.FP32],
    ]:
        pl.func_attr({"mx_tensor_views_blocked": True, "split_aiv": True, "dual_aiv_dispatch": True})
        mem_vec_21: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
        mem_vec_22: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 128)
        mem_vec_24: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 128)
        mem_vec_25: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32768)
        mem_vec_26: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32768)
        mem_vec_27: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 65536)
        mem_vec_48: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32)
        mem_vec_55: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32)
        mem_vec_60: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 128)
        mem_vec_62: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32768)
        qk_core_inline2468__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        qk_kv_base_inline2504__ssa_v0: pl.Scalar[pl.INDEX] = qk_core_inline2468__ssa_v0 * 512
        qk_head_base_inline2505__ssa_v0: pl.Scalar[pl.INDEX] = qk_core_inline2468__ssa_v0 * 64
        pl.system.set_ffts(ffts_workspace_inline2438__ssa_v0)
        for qk_t_inline2506__idx_v0 in pl.range(qk_core_inline2468__ssa_v0, t_dim_inline2491__ssa_v0, 24):
            qk_b_inline2507__ssa_v0: pl.Scalar[pl.INDEX] = qk_t_inline2506__idx_v0 // 6
            qk_aiv_inline2485__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_subblock_idx()
            pl.system.set_ffts(ffts_workspace_inline2438__ssa_v0)
            qk_lane_head_inline2529__ssa_v0: pl.Scalar[pl.INDEX] = qk_aiv_inline2485__ssa_v0 * 32
            qk_lane_kv_inline2456__ssa_v0: pl.Scalar[pl.INDEX] = qk_aiv_inline2485__ssa_v0 * 64
            qk_reduce_tmp_inline2439__ssa_v0: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_21, pl.const(0, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.create(
                [8, 512], dtype=pl.FP32, target_memory=pl.Mem.Vec
            )
            running_m_inline2522__ssa_v0: pl.Tile[[32, 1], pl.FP32, pl.MemRef(mem_vec_22, pl.const(16384, pl.INT64), 128), pl.Mem.Vec] = pl.tile.load(
                attn_sink_col_inline2429__ssa_v0, [qk_lane_head_inline2529__ssa_v0, 0], [32, 1], [32, 1], target_memory=pl.Mem.Vec
            )
            t__rm_a0_tmp_v0: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_22, pl.const(16384, pl.INT64), 128), pl.Mem.Vec] = pl.tile.reshape(running_m_inline2522__ssa_v0, [1, 32])
            t__row_major_tmp_v1: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_27, pl.const(82176, pl.INT64), 128), pl.Mem.Vec] = pl.tile.muls(t__rm_a0_tmp_v0, 0.0)
            t__tmp_v312: pl.Tile[[32, 1], pl.FP32, pl.MemRef(mem_vec_27, pl.const(82176, pl.INT64), 128), pl.Mem.Vec] = pl.tile.reshape(t__row_major_tmp_v1, [32, 1])
            running_l_inline2556__rm_a0_tmp_v2: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_27, pl.const(82176, pl.INT64), 128), pl.Mem.Vec] = pl.tile.reshape(t__tmp_v312, [1, 32])
            running_l_inline2556__row_major_tmp_v3: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_24, pl.const(16512, pl.INT64), 128), pl.Mem.Vec] = pl.tile.adds(running_l_inline2556__rm_a0_tmp_v2, 1.0)
            running_l_inline2556__ssa_v0: pl.Tile[[32, 1], pl.FP32, pl.MemRef(mem_vec_24, pl.const(16512, pl.INT64), 128), pl.Mem.Vec] = pl.tile.reshape(
                running_l_inline2556__row_major_tmp_v3, [32, 1]
            )
            running_left_inline2487__ssa_v0: pl.Tile[[32, 256], pl.FP32, pl.MemRef(mem_vec_25, pl.const(16640, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.full([32, 256], dtype=pl.FP32, value=0.0)
            running_right_inline2524__ssa_v0: pl.Tile[[32, 256], pl.FP32, pl.MemRef(mem_vec_26, pl.const(49408, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.full([32, 256], dtype=pl.FP32, value=0.0)
            mi_transfer_inline2500__store: pl.Tensor[[1536, 1], pl.FP32, pl.MemRef("mem_ddr_8", pl.const(0, pl.INT64), 6144)] = pl.tile.store(
                running_m_inline2522__ssa_v0, [qk_head_base_inline2505__ssa_v0 + qk_lane_head_inline2529__ssa_v0, 0], mi_transfer_inline2500__ssa_v0
            )
            li_transfer_inline2502__store: pl.Tensor[[1536, 1], pl.FP32, pl.MemRef("mem_ddr_9", pl.const(0, pl.INT64), 6144)] = pl.tile.store(
                running_l_inline2556__ssa_v0, [qk_head_base_inline2505__ssa_v0 + qk_lane_head_inline2529__ssa_v0, 0], li_transfer_inline2502__ssa_v0
            )
            for qk_sb_inline2511__idx_v0, (m_iter, l_iter, left_iter, right_iter) in pl.range(
                2, init_values=(running_m_inline2522__ssa_v0, running_l_inline2556__ssa_v0, running_left_inline2487__ssa_v0, running_right_inline2524__ssa_v0)
            ):
                t__tile: pl.Scalar[pl.INT32] = pl.tensor.read(valid_block_mask_inline2452__ssa_v0, [qk_t_inline2506__idx_v0, qk_sb_inline2511__idx_v0])
                if 0 < pl.cast(t__tile, pl.INDEX):
                    for qk_part_inline2440__idx_v0 in pl.range(4):
                        qk_part_row_inline2538__ssa_v0: pl.Scalar[pl.INDEX] = qk_part_inline2440__idx_v0 * 128 + qk_lane_kv_inline2456__ssa_v0
                        qk_kv_half_inline2542__ssa_v0: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_27, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.tile.full(
                            [64, 512], dtype=pl.BF16, value=0.0
                        )
                        if qk_sb_inline2511__idx_v0 == 0:
                            t__tile_1: pl.Scalar[pl.INT64] = pl.tensor.read(position_ids_t1__ssa_v0, [qk_t_inline2506__idx_v0, 0])
                            qk_pos_inline2501__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(t__tile_1, pl.INDEX)
                            qk_win_len_inline2531__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(qk_pos_inline2501__ssa_v0 + 1, 128)
                            qk_win_start_inline2554__ssa_v0: pl.Scalar[pl.INDEX] = qk_pos_inline2501__ssa_v0 - qk_win_len_inline2531__ssa_v0 + 1
                            qk_head_inline2533__ssa_v0: pl.Scalar[pl.INDEX] = qk_win_start_inline2554__ssa_v0 % 32
                            qk_rows_inline2534__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(pl.max(qk_win_len_inline2531__ssa_v0 - qk_part_row_inline2538__ssa_v0, 0), 64)
                            qk_lo_inline2536__ssa_v0: pl.Scalar[pl.INDEX] = 0
                            qk_hi_inline2460__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(32 - qk_head_inline2533__ssa_v0 - qk_part_row_inline2538__ssa_v0, qk_rows_inline2534__ssa_v0)
                            if qk_lo_inline2536__ssa_v0 < qk_hi_inline2460__ssa_v0:
                                qk_absolute_inline2537__ssa_v0: pl.Scalar[pl.INDEX] = qk_win_start_inline2554__ssa_v0 + qk_part_row_inline2538__ssa_v0 + qk_lo_inline2536__ssa_v0
                                qk_raw_page_inline2514__tile: pl.Scalar[pl.INT32] = pl.tensor.read(ori_block_table__ssa_v0, [qk_b_inline2507__ssa_v0, qk_absolute_inline2537__ssa_v0 // 32])
                                if 0 <= pl.cast(qk_raw_page_inline2514__tile, pl.INDEX):
                                    qk_raw_row_inline2539__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(qk_raw_page_inline2514__tile, pl.INDEX) * 32 + qk_absolute_inline2537__ssa_v0 % 32
                                    qk_kv_half_inline2542__ssa_v1: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_27, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.tile.gather_row(
                                        qk_kv_half_inline2542__ssa_v0,
                                        ori_kv_flat_inline2493__ssa_v1,
                                        [qk_lo_inline2536__ssa_v0, 0],
                                        [qk_raw_row_inline2539__ssa_v0, 0],
                                        [64, 512],
                                        valid_shape=[qk_hi_inline2460__ssa_v0 - qk_lo_inline2536__ssa_v0, 512],
                                        transpose=False,
                                    )
                                    qk_kv_half_inline2542__phi_v2: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_27, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(
                                        qk_kv_half_inline2542__ssa_v1
                                    )
                                else:
                                    qk_kv_half_inline2542__phi_v2: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_27, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(
                                        qk_kv_half_inline2542__ssa_v0
                                    )
                                qk_kv_half_inline2542__phi_v3: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_27, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(
                                    qk_kv_half_inline2542__phi_v2
                                )
                            else:
                                qk_kv_half_inline2542__phi_v3: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_27, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(
                                    qk_kv_half_inline2542__ssa_v0
                                )
                            qk_lo_inline2536__ssa_v1: pl.Scalar[pl.INDEX] = pl.max(32 - qk_head_inline2533__ssa_v0 - qk_part_row_inline2538__ssa_v0, 0)
                            qk_hi_inline2460__ssa_v1: pl.Scalar[pl.INDEX] = pl.min(64 - qk_head_inline2533__ssa_v0 - qk_part_row_inline2538__ssa_v0, qk_rows_inline2534__ssa_v0)
                            if qk_lo_inline2536__ssa_v1 < qk_hi_inline2460__ssa_v1:
                                qk_absolute_inline2537__ssa_v1: pl.Scalar[pl.INDEX] = qk_win_start_inline2554__ssa_v0 + qk_part_row_inline2538__ssa_v0 + qk_lo_inline2536__ssa_v1
                                qk_raw_page_inline2514__tile_1: pl.Scalar[pl.INT32] = pl.tensor.read(ori_block_table__ssa_v0, [qk_b_inline2507__ssa_v0, qk_absolute_inline2537__ssa_v1 // 32])
                                if 0 <= pl.cast(qk_raw_page_inline2514__tile_1, pl.INDEX):
                                    qk_raw_row_inline2539__ssa_v1: pl.Scalar[pl.INDEX] = pl.cast(qk_raw_page_inline2514__tile_1, pl.INDEX) * 32 + qk_absolute_inline2537__ssa_v1 % 32
                                    qk_kv_half_inline2542__ssa_v4: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_27, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.tile.gather_row(
                                        qk_kv_half_inline2542__phi_v3,
                                        ori_kv_flat_inline2493__ssa_v1,
                                        [qk_lo_inline2536__ssa_v1, 0],
                                        [qk_raw_row_inline2539__ssa_v1, 0],
                                        [64, 512],
                                        valid_shape=[qk_hi_inline2460__ssa_v1 - qk_lo_inline2536__ssa_v1, 512],
                                        transpose=False,
                                    )
                                    qk_kv_half_inline2542__phi_v5: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_27, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(
                                        qk_kv_half_inline2542__ssa_v4
                                    )
                                else:
                                    qk_kv_half_inline2542__phi_v5: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_27, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(
                                        qk_kv_half_inline2542__phi_v3
                                    )
                                qk_kv_half_inline2542__phi_v6: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_27, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(
                                    qk_kv_half_inline2542__phi_v5
                                )
                            else:
                                qk_kv_half_inline2542__phi_v6: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_27, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(
                                    qk_kv_half_inline2542__phi_v3
                                )
                            qk_lo_inline2536__ssa_v2: pl.Scalar[pl.INDEX] = pl.max(64 - qk_head_inline2533__ssa_v0 - qk_part_row_inline2538__ssa_v0, 0)
                            qk_hi_inline2460__ssa_v2: pl.Scalar[pl.INDEX] = pl.min(96 - qk_head_inline2533__ssa_v0 - qk_part_row_inline2538__ssa_v0, qk_rows_inline2534__ssa_v0)
                            if qk_lo_inline2536__ssa_v2 < qk_hi_inline2460__ssa_v2:
                                qk_absolute_inline2537__ssa_v2: pl.Scalar[pl.INDEX] = qk_win_start_inline2554__ssa_v0 + qk_part_row_inline2538__ssa_v0 + qk_lo_inline2536__ssa_v2
                                qk_raw_page_inline2514__tile_2: pl.Scalar[pl.INT32] = pl.tensor.read(ori_block_table__ssa_v0, [qk_b_inline2507__ssa_v0, qk_absolute_inline2537__ssa_v2 // 32])
                                if 0 <= pl.cast(qk_raw_page_inline2514__tile_2, pl.INDEX):
                                    qk_raw_row_inline2539__ssa_v2: pl.Scalar[pl.INDEX] = pl.cast(qk_raw_page_inline2514__tile_2, pl.INDEX) * 32 + qk_absolute_inline2537__ssa_v2 % 32
                                    qk_kv_half_inline2542__ssa_v7: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_27, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.tile.gather_row(
                                        qk_kv_half_inline2542__phi_v6,
                                        ori_kv_flat_inline2493__ssa_v1,
                                        [qk_lo_inline2536__ssa_v2, 0],
                                        [qk_raw_row_inline2539__ssa_v2, 0],
                                        [64, 512],
                                        valid_shape=[qk_hi_inline2460__ssa_v2 - qk_lo_inline2536__ssa_v2, 512],
                                        transpose=False,
                                    )
                                    qk_kv_half_inline2542__phi_v8: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_27, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(
                                        qk_kv_half_inline2542__ssa_v7
                                    )
                                else:
                                    qk_kv_half_inline2542__phi_v8: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_27, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(
                                        qk_kv_half_inline2542__phi_v6
                                    )
                                qk_kv_half_inline2542__phi_v9: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_27, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(
                                    qk_kv_half_inline2542__phi_v8
                                )
                            else:
                                qk_kv_half_inline2542__phi_v9: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_27, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(
                                    qk_kv_half_inline2542__phi_v6
                                )
                            qk_lo_inline2536__ssa_v3: pl.Scalar[pl.INDEX] = pl.max(96 - qk_head_inline2533__ssa_v0 - qk_part_row_inline2538__ssa_v0, 0)
                            qk_hi_inline2460__ssa_v3: pl.Scalar[pl.INDEX] = pl.min(128 - qk_head_inline2533__ssa_v0 - qk_part_row_inline2538__ssa_v0, qk_rows_inline2534__ssa_v0)
                            if qk_lo_inline2536__ssa_v3 < qk_hi_inline2460__ssa_v3:
                                qk_absolute_inline2537__ssa_v3: pl.Scalar[pl.INDEX] = qk_win_start_inline2554__ssa_v0 + qk_part_row_inline2538__ssa_v0 + qk_lo_inline2536__ssa_v3
                                qk_raw_page_inline2514__tile_3: pl.Scalar[pl.INT32] = pl.tensor.read(ori_block_table__ssa_v0, [qk_b_inline2507__ssa_v0, qk_absolute_inline2537__ssa_v3 // 32])
                                if 0 <= pl.cast(qk_raw_page_inline2514__tile_3, pl.INDEX):
                                    qk_raw_row_inline2539__ssa_v3: pl.Scalar[pl.INDEX] = pl.cast(qk_raw_page_inline2514__tile_3, pl.INDEX) * 32 + qk_absolute_inline2537__ssa_v3 % 32
                                    qk_kv_half_inline2542__ssa_v10: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_27, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.tile.gather_row(
                                        qk_kv_half_inline2542__phi_v9,
                                        ori_kv_flat_inline2493__ssa_v1,
                                        [qk_lo_inline2536__ssa_v3, 0],
                                        [qk_raw_row_inline2539__ssa_v3, 0],
                                        [64, 512],
                                        valid_shape=[qk_hi_inline2460__ssa_v3 - qk_lo_inline2536__ssa_v3, 512],
                                        transpose=False,
                                    )
                                    qk_kv_half_inline2542__phi_v11: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_27, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(
                                        qk_kv_half_inline2542__ssa_v10
                                    )
                                else:
                                    qk_kv_half_inline2542__phi_v11: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_27, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(
                                        qk_kv_half_inline2542__phi_v9
                                    )
                                qk_kv_half_inline2542__phi_v12: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_27, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(
                                    qk_kv_half_inline2542__phi_v11
                                )
                            else:
                                qk_kv_half_inline2542__phi_v12: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_27, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(
                                    qk_kv_half_inline2542__phi_v9
                                )
                            qk_lo_inline2536__ssa_v4: pl.Scalar[pl.INDEX] = pl.max(128 - qk_head_inline2533__ssa_v0 - qk_part_row_inline2538__ssa_v0, 0)
                            qk_hi_inline2460__ssa_v4: pl.Scalar[pl.INDEX] = pl.min(160 - qk_head_inline2533__ssa_v0 - qk_part_row_inline2538__ssa_v0, qk_rows_inline2534__ssa_v0)
                            if qk_lo_inline2536__ssa_v4 < qk_hi_inline2460__ssa_v4:
                                qk_absolute_inline2537__ssa_v4: pl.Scalar[pl.INDEX] = qk_win_start_inline2554__ssa_v0 + qk_part_row_inline2538__ssa_v0 + qk_lo_inline2536__ssa_v4
                                qk_raw_page_inline2514__tile_4: pl.Scalar[pl.INT32] = pl.tensor.read(ori_block_table__ssa_v0, [qk_b_inline2507__ssa_v0, qk_absolute_inline2537__ssa_v4 // 32])
                                if 0 <= pl.cast(qk_raw_page_inline2514__tile_4, pl.INDEX):
                                    qk_raw_row_inline2539__ssa_v4: pl.Scalar[pl.INDEX] = pl.cast(qk_raw_page_inline2514__tile_4, pl.INDEX) * 32 + qk_absolute_inline2537__ssa_v4 % 32
                                    qk_kv_half_inline2542__ssa_v13: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_27, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.tile.gather_row(
                                        qk_kv_half_inline2542__phi_v12,
                                        ori_kv_flat_inline2493__ssa_v1,
                                        [qk_lo_inline2536__ssa_v4, 0],
                                        [qk_raw_row_inline2539__ssa_v4, 0],
                                        [64, 512],
                                        valid_shape=[qk_hi_inline2460__ssa_v4 - qk_lo_inline2536__ssa_v4, 512],
                                        transpose=False,
                                    )
                                    qk_kv_half_inline2542__phi_v14: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_27, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(
                                        qk_kv_half_inline2542__ssa_v13
                                    )
                                else:
                                    qk_kv_half_inline2542__phi_v14: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_27, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(
                                        qk_kv_half_inline2542__phi_v12
                                    )
                                qk_kv_half_inline2542__phi_v15: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_27, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(
                                    qk_kv_half_inline2542__phi_v14
                                )
                            else:
                                qk_kv_half_inline2542__phi_v15: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_27, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(
                                    qk_kv_half_inline2542__phi_v12
                                )
                            qk_kv_half_inline2542__phi_v21: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_27, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(qk_kv_half_inline2542__phi_v15)
                        else:
                            qk_kv_half_inline2542__ssa_v0: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_27, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.tile.full(
                                [64, 512], dtype=pl.BF16, value=0.0
                            )
                            for qk_row_inline2543__idx_v0, (qk_kv_half_inline2542__iter_v16,) in pl.range(64, init_values=(qk_kv_half_inline2542__ssa_v0,)):
                                qk_cmp_k_inline2465__ssa_v0: pl.Scalar[pl.INDEX] = (qk_sb_inline2511__idx_v0 - 1) * 512 + qk_part_row_inline2538__ssa_v0 + qk_row_inline2543__idx_v0
                                if qk_cmp_k_inline2465__ssa_v0 < 512:
                                    qk_ridx_inline2488__tile: pl.Scalar[pl.INT32] = pl.tensor.read(cmp_sparse_indices_inline2484__rv_v2, [qk_t_inline2506__idx_v0, qk_cmp_k_inline2465__ssa_v0])
                                    if 0 <= pl.cast(qk_ridx_inline2488__tile, pl.INDEX):
                                        t__tile_2: pl.Scalar[pl.INT32] = pl.tensor.read(cmp_block_table__ssa_v0, [qk_b_inline2507__ssa_v0, pl.cast(qk_ridx_inline2488__tile, pl.INDEX) // 32])
                                        qk_page_inline2544__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(t__tile_2, pl.INDEX)
                                        qk_src_inline2546__ssa_v0: pl.Scalar[pl.INDEX] = qk_page_inline2544__ssa_v0 * 32 + pl.cast(qk_ridx_inline2488__tile, pl.INDEX) % 32
                                        qk_kv_half_inline2542__ssa_v18: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_27, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.tile.gather_row(
                                            qk_kv_half_inline2542__iter_v16, cmp_kv_flat_inline2430__ssa_v0, [qk_row_inline2543__idx_v0, 0], [qk_src_inline2546__ssa_v0, 0], [1, 512], transpose=False
                                        )
                                        qk_kv_half_inline2542__phi_v19: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_27, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(
                                            qk_kv_half_inline2542__ssa_v18
                                        )
                                    else:
                                        qk_kv_half_inline2542__phi_v19: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_27, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(
                                            qk_kv_half_inline2542__iter_v16
                                        )
                                    qk_kv_half_inline2542__phi_v20: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_27, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(
                                        qk_kv_half_inline2542__phi_v19
                                    )
                                else:
                                    qk_kv_half_inline2542__phi_v20: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_27, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(
                                        qk_kv_half_inline2542__iter_v16
                                    )
                                qk_kv_half_inline2542__rv_v17: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_27, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(
                                    qk_kv_half_inline2542__phi_v20
                                )
                            qk_kv_half_inline2542__phi_v21: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_27, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(qk_kv_half_inline2542__rv_v17)
                        kv_transfer_inline2426__store: pl.Tensor[[12288, 512], pl.BF16, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 12582912)] = pl.tile.store(
                            qk_kv_half_inline2542__phi_v21, [qk_kv_base_inline2504__ssa_v0 + qk_part_row_inline2538__ssa_v0, 0], kv_transfer_inline2426__ssa_v0
                        )
                    pl.system.sync_set(0, pipe=pl.PipeType.MTE3, ffts_mode=2, core_type=pl.KernelType.AIV)
                    pl.system.sync_wait(1, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIV)
                    for sm_part_inline2427__idx_v0 in pl.range(4):
                        sm_h_inline2548__ssa_v0: pl.Scalar[pl.INDEX] = sm_part_inline2427__idx_v0 * 8
                        sm_row_inline2549__ssa_v0: pl.Scalar[pl.INDEX] = qk_head_base_inline2505__ssa_v0 + qk_lane_head_inline2529__ssa_v0 + sm_h_inline2548__ssa_v0
                        sm_scores_inline2470__ssa_v0: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_27, pl.const(82176, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.load(
                            score_transfer_inline2447__ssa_v0, [sm_row_inline2549__ssa_v0, 0], [8, 512], [8, 512], target_memory=pl.Mem.Vec
                        )
                        sm_bias_inline2551__ssa_v0: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_62, pl.const(147904, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.load(
                            sparse_bias_inline2455__rv_v2, [qk_t_inline2506__idx_v0, qk_sb_inline2511__idx_v0 * 512], [1, 512], [1, 512], target_memory=pl.Mem.Vec
                        )
                        t__tmp_v316: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_27, pl.const(82176, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.muls(sm_scores_inline2470__ssa_v0, 0.044194173824159223)
                        sm_masked_inline2457__ssa_v0: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_27, pl.const(82176, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.col_expand_add(
                            t__tmp_v316, sm_bias_inline2551__ssa_v0
                        )
                        sm_block_max_inline2552__ssa_v0: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_62, pl.const(147904, pl.INT64), 32), pl.Mem.Vec] = pl.tile.row_max(
                            sm_masked_inline2457__ssa_v0, qk_reduce_tmp_inline2439__ssa_v0
                        )
                        sm_old_m_inline2555__ssa_v0: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_60, pl.const(147776, pl.INT64), 32), pl.Mem.Vec] = pl.tile.load(
                            mi_transfer_inline2500__ssa_v0, [sm_row_inline2549__ssa_v0, 0], [8, 1], [8, 1], target_memory=pl.Mem.Vec
                        )
                        sm_old_l_inline2557__ssa_v0: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_48, pl.const(147712, pl.INT64), 32), pl.Mem.Vec] = pl.tile.load(
                            li_transfer_inline2502__ssa_v0, [sm_row_inline2549__ssa_v0, 0], [8, 1], [8, 1], target_memory=pl.Mem.Vec
                        )
                        sm_max_inline2434__rm_a0_tmp_v4: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_60, pl.const(147776, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(
                            sm_old_m_inline2555__ssa_v0, [1, 8]
                        )
                        sm_max_inline2434__rm_a1_tmp_v5: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_62, pl.const(147904, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(
                            sm_block_max_inline2552__ssa_v0, [1, 8]
                        )
                        sm_max_inline2434__row_major_tmp_v6: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_62, pl.const(147904, pl.INT64), 32), pl.Mem.Vec] = pl.tile.maximum(
                            sm_max_inline2434__rm_a0_tmp_v4, sm_max_inline2434__rm_a1_tmp_v5
                        )
                        sm_max_inline2434__ssa_v0: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_62, pl.const(147904, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(
                            sm_max_inline2434__row_major_tmp_v6, [8, 1]
                        )
                        t__rm_a0_tmp_v7: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_60, pl.const(147776, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(sm_old_m_inline2555__ssa_v0, [1, 8])
                        t__rm_a1_tmp_v8: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_62, pl.const(147904, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(sm_max_inline2434__ssa_v0, [1, 8])
                        t__row_major_tmp_v9: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_60, pl.const(147776, pl.INT64), 32), pl.Mem.Vec] = pl.tile.sub(t__rm_a0_tmp_v7, t__rm_a1_tmp_v8)
                        t__tmp_v317: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_60, pl.const(147776, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(t__row_major_tmp_v9, [8, 1])
                        sm_alpha_inline2521__rm_a0_tmp_v10: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_60, pl.const(147776, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(t__tmp_v317, [1, 8])
                        sm_alpha_inline2521__row_major_tmp_v11: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_60, pl.const(147776, pl.INT64), 32), pl.Mem.Vec] = pl.tile.exp(
                            sm_alpha_inline2521__rm_a0_tmp_v10
                        )
                        sm_alpha_inline2521__ssa_v0: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_60, pl.const(147776, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(
                            sm_alpha_inline2521__row_major_tmp_v11, [8, 1]
                        )
                        t__tmp_v318: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_27, pl.const(82176, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.row_expand_sub(
                            sm_masked_inline2457__ssa_v0, sm_max_inline2434__ssa_v0
                        )
                        sm_exp_inline2433__ssa_v0: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_27, pl.const(82176, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.exp(t__tmp_v318)
                        t__rm_a0_tmp_v12: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_48, pl.const(147712, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(sm_old_l_inline2557__ssa_v0, [1, 8])
                        t__rm_a1_tmp_v13: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_60, pl.const(147776, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(sm_alpha_inline2521__ssa_v0, [1, 8])
                        t__row_major_tmp_v14: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_48, pl.const(147712, pl.INT64), 32), pl.Mem.Vec] = pl.tile.mul(t__rm_a0_tmp_v12, t__rm_a1_tmp_v13)
                        t__tmp_v319: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_48, pl.const(147712, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(t__row_major_tmp_v14, [8, 1])
                        t__tmp_v320: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_55, pl.const(147744, pl.INT64), 32), pl.Mem.Vec] = pl.tile.row_sum(
                            sm_exp_inline2433__ssa_v0, qk_reduce_tmp_inline2439__ssa_v0
                        )
                        sm_sum_inline2444__rm_a0_tmp_v15: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_48, pl.const(147712, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(t__tmp_v319, [1, 8])
                        sm_sum_inline2444__rm_a1_tmp_v16: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_55, pl.const(147744, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(t__tmp_v320, [1, 8])
                        sm_sum_inline2444__row_major_tmp_v17: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_48, pl.const(147712, pl.INT64), 32), pl.Mem.Vec] = pl.tile.add(
                            sm_sum_inline2444__rm_a0_tmp_v15, sm_sum_inline2444__rm_a1_tmp_v16
                        )
                        sm_sum_inline2444__ssa_v0: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_48, pl.const(147712, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(
                            sm_sum_inline2444__row_major_tmp_v17, [8, 1]
                        )
                        sm_probability_inline2422__ssa_v0: pl.Tile[[8, 512], pl.BF16, pl.MemRef(mem_vec_27, pl.const(82176, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.cast(
                            sm_exp_inline2433__ssa_v0, target_type=pl.BF16, mode="round"
                        )
                        probability_transfer_inline2550__store: pl.Tensor[[1536, 512], pl.BF16, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 1572864)] = pl.tile.store(
                            sm_probability_inline2422__ssa_v0, [sm_row_inline2549__ssa_v0, 0], probability_transfer_inline2550__ssa_v0
                        )
                        mi_transfer_inline2500__store_v0: pl.Tensor[[1536, 1], pl.FP32, pl.MemRef("mem_ddr_8", pl.const(0, pl.INT64), 6144)] = pl.tile.store(
                            sm_max_inline2434__ssa_v0, [sm_row_inline2549__ssa_v0, 0], mi_transfer_inline2500__ssa_v0
                        )
                        li_transfer_inline2502__store_v0: pl.Tensor[[1536, 1], pl.FP32, pl.MemRef("mem_ddr_9", pl.const(0, pl.INT64), 6144)] = pl.tile.store(
                            sm_sum_inline2444__ssa_v0, [sm_row_inline2549__ssa_v0, 0], li_transfer_inline2502__ssa_v0
                        )
                        alpha_transfer_inline2486__store: pl.Tensor[[1536, 1], pl.FP32, pl.MemRef("mem_ddr_17", pl.const(0, pl.INT64), 6144)] = pl.tile.store(
                            sm_alpha_inline2521__ssa_v0, [sm_row_inline2549__ssa_v0, 0], alpha_transfer_inline2486__ssa_v0
                        )
                    pl.system.sync_set(2, pipe=pl.PipeType.MTE3, ffts_mode=2, core_type=pl.KernelType.AIV)
                    pl.system.sync_wait(3, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIV)
                    pv_row_inline2424__ssa_v0: pl.Scalar[pl.INDEX] = qk_head_base_inline2505__ssa_v0 + qk_lane_head_inline2529__ssa_v0
                    next_m_inline2541__ssa_v0: pl.Tile[[32, 1], pl.FP32, pl.MemRef(mem_vec_22, pl.const(16384, pl.INT64), 128), pl.Mem.Vec] = pl.tile.load(
                        mi_transfer_inline2500__ssa_v0, [pv_row_inline2424__ssa_v0, 0], [32, 1], [32, 1], target_memory=pl.Mem.Vec
                    )
                    next_l_inline2421__ssa_v0: pl.Tile[[32, 1], pl.FP32, pl.MemRef(mem_vec_24, pl.const(16512, pl.INT64), 128), pl.Mem.Vec] = pl.tile.load(
                        li_transfer_inline2502__ssa_v0, [pv_row_inline2424__ssa_v0, 0], [32, 1], [32, 1], target_memory=pl.Mem.Vec
                    )
                    alpha_inline2481__ssa_v0: pl.Tile[[32, 1], pl.FP32, pl.MemRef(mem_vec_60, pl.const(147776, pl.INT64), 128), pl.Mem.Vec] = pl.tile.load(
                        alpha_transfer_inline2486__ssa_v0, [pv_row_inline2424__ssa_v0, 0], [32, 1], [32, 1], target_memory=pl.Mem.Vec
                    )
                    pv_left_inline2420__ssa_v0: pl.Tile[[32, 256], pl.FP32, pl.MemRef(mem_vec_27, pl.const(82176, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.load(
                        pv_transfer_inline2437__ssa_v0, [pv_row_inline2424__ssa_v0, 0], [32, 256], [32, 256], target_memory=pl.Mem.Vec
                    )
                    t__tmp_v321: pl.Tile[[32, 256], pl.FP32, pl.MemRef(mem_vec_62, pl.const(147904, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.row_expand_mul(left_iter, alpha_inline2481__ssa_v0)
                    next_left_inline2518__ssa_v0: pl.Tile[[32, 256], pl.FP32, pl.MemRef(mem_vec_25, pl.const(16640, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.add(
                        t__tmp_v321, pv_left_inline2420__ssa_v0
                    )
                    pv_right_inline2419__ssa_v0: pl.Tile[[32, 256], pl.FP32, pl.MemRef(mem_vec_27, pl.const(82176, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.load(
                        pv_transfer_inline2437__ssa_v0, [pv_row_inline2424__ssa_v0, 256], [32, 256], [32, 256], target_memory=pl.Mem.Vec
                    )
                    t__tmp_v322: pl.Tile[[32, 256], pl.FP32, pl.MemRef(mem_vec_62, pl.const(147904, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.row_expand_mul(right_iter, alpha_inline2481__ssa_v0)
                    next_right_inline2418__ssa_v0: pl.Tile[[32, 256], pl.FP32, pl.MemRef(mem_vec_26, pl.const(49408, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.add(
                        t__tmp_v322, pv_right_inline2419__ssa_v0
                    )
                    m_after_inline2528__rv_v0, l_after_inline2432__rv_v0, left_after_inline2530__rv_v0, right_after_inline2442__rv_v0 = pl.yield_(
                        next_m_inline2541__ssa_v0, next_l_inline2421__ssa_v0, next_left_inline2518__ssa_v0, next_right_inline2418__ssa_v0
                    )
                else:
                    m_after_inline2528__rv_v0, l_after_inline2432__rv_v0, left_after_inline2530__rv_v0, right_after_inline2442__rv_v0 = pl.yield_(m_iter, l_iter, left_iter, right_iter)
                running_m_inline2526, running_l_inline2532, running_left_inline2499, running_right_inline2496 = pl.yield_(
                    m_after_inline2528__rv_v0, l_after_inline2432__rv_v0, left_after_inline2530__rv_v0, right_after_inline2442__rv_v0
                )
            qk_output_row_inline2416__ssa_v0: pl.Scalar[pl.INDEX] = qk_t_inline2506__idx_v0 * 64 + qk_lane_head_inline2529__ssa_v0
            attn_mi_inline2490__store: pl.Tensor[[t_heads_inline2469__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_18", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                running_m_inline2526, [qk_output_row_inline2416__ssa_v0, 0], attn_mi_inline2490__ssa_v0
            )
            attn_li_inline2516__store: pl.Tensor[[t_heads_inline2469__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_19", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                running_l_inline2532, [qk_output_row_inline2416__ssa_v0, 0], attn_li_inline2516__ssa_v0
            )
            attn_oi_inline2428__store: pl.Tensor[[t_heads_inline2469__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_20", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                running_left_inline2499, [qk_output_row_inline2416__ssa_v0, 0], attn_oi_inline2428__ssa_v0
            )
            attn_oi_inline2428__store_v0: pl.Tensor[[t_heads_inline2469__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_20", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                running_right_inline2496, [qk_output_row_inline2416__ssa_v0, 256], attn_oi_inline2428__ssa_v0
            )
        return (
            kv_transfer_inline2426__ssa_v0,
            score_transfer_inline2447__ssa_v0,
            probability_transfer_inline2550__ssa_v0,
            pv_transfer_inline2437__ssa_v0,
            mi_transfer_inline2500__ssa_v0,
            li_transfer_inline2502__ssa_v0,
            alpha_transfer_inline2486__ssa_v0,
            attn_mi_inline2490__ssa_v0,
            attn_li_inline2516__ssa_v0,
            attn_oi_inline2428__ssa_v0,
        )

    @pl.function(type=pl.FunctionType.Group, level=pl.Level.CORE_GROUP, role=pl.Role.SubWorker)
    def qk_pv(
        ffts_workspace_inline2438__ssa_v0: pl.Tensor[[256], pl.INT64, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 2048)],
        t_dim_inline2491__ssa_v0: pl.Scalar[pl.INDEX],
        q_flat_inline2480__ssa_v0: pl.Tensor[[t_heads_inline2469__ssa_v0, 512], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)],
        valid_block_mask_inline2452__ssa_v0: pl.Tensor[[t_dim_inline2491__ssa_v0, 16], pl.INT32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)],
        kv_transfer_inline2426__ssa_v0: pl.InOut[pl.Tensor[[12288, 512], pl.BF16, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 12582912)]],
        score_transfer_inline2447__ssa_v0: pl.InOut[pl.Tensor[[1536, 512], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 3145728)]],
        probability_transfer_inline2550__ssa_v0: pl.InOut[pl.Tensor[[1536, 512], pl.BF16, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 1572864)]],
        pv_transfer_inline2437__ssa_v0: pl.InOut[pl.Tensor[[1536, 512], pl.FP32, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 3145728)]],
        attn_sink_col_inline2429__ssa_v0: pl.Tensor[[64, 1], pl.FP32, pl.MemRef("mem_ddr_7", pl.const(0, pl.INT64), 256)],
        mi_transfer_inline2500__ssa_v0: pl.InOut[pl.Tensor[[1536, 1], pl.FP32, pl.MemRef("mem_ddr_8", pl.const(0, pl.INT64), 6144)]],
        li_transfer_inline2502__ssa_v0: pl.InOut[pl.Tensor[[1536, 1], pl.FP32, pl.MemRef("mem_ddr_9", pl.const(0, pl.INT64), 6144)]],
        position_ids_t1__ssa_v0: pl.Tensor[[T_DYN, 1], pl.INT64, pl.MemRef("mem_ddr_10", pl.const(0, pl.INT64), 0)],
        ori_block_table__ssa_v0: pl.Tensor[[B_DYN, ORIGINAL_TABLE_COLUMNS_DYN], pl.INT32, pl.MemRef("mem_ddr_11", pl.const(0, pl.INT64), 0)],
        ori_kv_flat_inline2493__ssa_v1: pl.Tensor[[ori_block_num_inline2459__ssa_v0 * pl.const(32, pl.INDEX), 512], pl.BF16, pl.MemRef("mem_ddr_12", pl.const(0, pl.INT64), 0)],
        cmp_sparse_indices_inline2484__rv_v2: pl.Tensor[[t_dim_inline2491__ssa_v0, 512], pl.INT32, pl.MemRef("mem_ddr_13", pl.const(0, pl.INT64), 0)],
        cmp_block_table__ssa_v0: pl.Tensor[[B_DYN, COMPRESSED_TABLE_COLUMNS_DYN], pl.INT32, pl.MemRef("mem_ddr_14", pl.const(0, pl.INT64), 0)],
        cmp_kv_flat_inline2430__ssa_v0: pl.Tensor[[cmp_block_num_inline2525__ssa_v0 * pl.const(32, pl.INDEX), 512], pl.BF16, pl.MemRef("mem_ddr_15", pl.const(0, pl.INT64), 0)],
        sparse_bias_inline2455__rv_v2: pl.Tensor[[t_dim_inline2491__ssa_v0, 1024], pl.FP32, pl.MemRef("mem_ddr_16", pl.const(0, pl.INT64), 0)],
        alpha_transfer_inline2486__ssa_v0: pl.InOut[pl.Tensor[[1536, 1], pl.FP32, pl.MemRef("mem_ddr_17", pl.const(0, pl.INT64), 6144)]],
        attn_mi_inline2490__ssa_v0: pl.Out[pl.Tensor[[t_heads_inline2469__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_18", pl.const(0, pl.INT64), 0)]],
        attn_li_inline2516__ssa_v0: pl.Out[pl.Tensor[[t_heads_inline2469__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_19", pl.const(0, pl.INT64), 0)]],
        attn_oi_inline2428__ssa_v0: pl.Out[pl.Tensor[[t_heads_inline2469__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_20", pl.const(0, pl.INT64), 0)]],
    ) -> tuple[
        pl.Tensor[[12288, 512], pl.BF16],
        pl.Tensor[[1536, 512], pl.FP32],
        pl.Tensor[[1536, 512], pl.BF16],
        pl.Tensor[[1536, 512], pl.FP32],
        pl.Tensor[[1536, 1], pl.FP32],
        pl.Tensor[[1536, 1], pl.FP32],
        pl.Tensor[[1536, 1], pl.FP32],
        pl.Tensor[[t_heads_inline2469__ssa_v0, 1], pl.FP32],
        pl.Tensor[[t_heads_inline2469__ssa_v0, 1], pl.FP32],
        pl.Tensor[[t_heads_inline2469__ssa_v0, 512], pl.FP32],
    ]:
        pl.func_attr({"mx_tensor_views_blocked": True, "split_aiv": True})
        self.qk_pv_aic(
            ffts_workspace_inline2438__ssa_v0,
            t_dim_inline2491__ssa_v0,
            q_flat_inline2480__ssa_v0,
            valid_block_mask_inline2452__ssa_v0,
            kv_transfer_inline2426__ssa_v0,
            score_transfer_inline2447__ssa_v0,
            probability_transfer_inline2550__ssa_v0,
            pv_transfer_inline2437__ssa_v0,
            attn_sink_col_inline2429__ssa_v0,
            mi_transfer_inline2500__ssa_v0,
            li_transfer_inline2502__ssa_v0,
            position_ids_t1__ssa_v0,
            ori_block_table__ssa_v0,
            ori_kv_flat_inline2493__ssa_v1,
            cmp_sparse_indices_inline2484__rv_v2,
            cmp_block_table__ssa_v0,
            cmp_kv_flat_inline2430__ssa_v0,
            sparse_bias_inline2455__rv_v2,
            alpha_transfer_inline2486__ssa_v0,
            attn_mi_inline2490__ssa_v0,
            attn_li_inline2516__ssa_v0,
            attn_oi_inline2428__ssa_v0,
            attrs={
                "arg_directions": [
                    pl.adir.input,
                    pl.adir.scalar,
                    pl.adir.input,
                    pl.adir.input,
                    pl.adir.inout,
                    pl.adir.inout,
                    pl.adir.inout,
                    pl.adir.inout,
                    pl.adir.input,
                    pl.adir.inout,
                    pl.adir.inout,
                    pl.adir.input,
                    pl.adir.input,
                    pl.adir.input,
                    pl.adir.input,
                    pl.adir.input,
                    pl.adir.input,
                    pl.adir.input,
                    pl.adir.inout,
                    pl.adir.output_existing,
                    pl.adir.output_existing,
                    pl.adir.output_existing,
                ]
            },
        )
        self.qk_pv_aiv(
            ffts_workspace_inline2438__ssa_v0,
            t_dim_inline2491__ssa_v0,
            q_flat_inline2480__ssa_v0,
            valid_block_mask_inline2452__ssa_v0,
            kv_transfer_inline2426__ssa_v0,
            score_transfer_inline2447__ssa_v0,
            probability_transfer_inline2550__ssa_v0,
            pv_transfer_inline2437__ssa_v0,
            attn_sink_col_inline2429__ssa_v0,
            mi_transfer_inline2500__ssa_v0,
            li_transfer_inline2502__ssa_v0,
            position_ids_t1__ssa_v0,
            ori_block_table__ssa_v0,
            ori_kv_flat_inline2493__ssa_v1,
            cmp_sparse_indices_inline2484__rv_v2,
            cmp_block_table__ssa_v0,
            cmp_kv_flat_inline2430__ssa_v0,
            sparse_bias_inline2455__rv_v2,
            alpha_transfer_inline2486__ssa_v0,
            attn_mi_inline2490__ssa_v0,
            attn_li_inline2516__ssa_v0,
            attn_oi_inline2428__ssa_v0,
            attrs={
                "arg_directions": [
                    pl.adir.input,
                    pl.adir.scalar,
                    pl.adir.input,
                    pl.adir.input,
                    pl.adir.inout,
                    pl.adir.inout,
                    pl.adir.inout,
                    pl.adir.inout,
                    pl.adir.input,
                    pl.adir.inout,
                    pl.adir.inout,
                    pl.adir.input,
                    pl.adir.input,
                    pl.adir.input,
                    pl.adir.input,
                    pl.adir.input,
                    pl.adir.input,
                    pl.adir.input,
                    pl.adir.inout,
                    pl.adir.inout,
                    pl.adir.inout,
                    pl.adir.inout,
                ]
            },
        )
        return (
            kv_transfer_inline2426__ssa_v0,
            score_transfer_inline2447__ssa_v0,
            probability_transfer_inline2550__ssa_v0,
            pv_transfer_inline2437__ssa_v0,
            mi_transfer_inline2500__ssa_v0,
            li_transfer_inline2502__ssa_v0,
            alpha_transfer_inline2486__ssa_v0,
            attn_mi_inline2490__ssa_v0,
            attn_li_inline2516__ssa_v0,
            attn_oi_inline2428__ssa_v0,
        )

    @pl.function(type=pl.FunctionType.Spmd)
    def qk_pv_spmd(
        self,
        ffts_workspace_inline2438__ssa_v0: pl.Tensor[[256], pl.INT64, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 2048)],
        t_dim_inline2491__ssa_v0: pl.Scalar[pl.INDEX],
        q_flat_inline2480__ssa_v0: pl.Tensor[[t_heads_inline2469__ssa_v0, 512], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)],
        valid_block_mask_inline2452__ssa_v0: pl.Tensor[[t_dim_inline2491__ssa_v0, 16], pl.INT32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)],
        kv_transfer_inline2426__ssa_v0: pl.InOut[pl.Tensor[[12288, 512], pl.BF16, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 12582912)]],
        score_transfer_inline2447__ssa_v0: pl.InOut[pl.Tensor[[1536, 512], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 3145728)]],
        probability_transfer_inline2550__ssa_v0: pl.InOut[pl.Tensor[[1536, 512], pl.BF16, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 1572864)]],
        pv_transfer_inline2437__ssa_v0: pl.InOut[pl.Tensor[[1536, 512], pl.FP32, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 3145728)]],
        attn_sink_col_inline2429__ssa_v0: pl.Tensor[[64, 1], pl.FP32, pl.MemRef("mem_ddr_7", pl.const(0, pl.INT64), 256)],
        mi_transfer_inline2500__ssa_v0: pl.InOut[pl.Tensor[[1536, 1], pl.FP32, pl.MemRef("mem_ddr_8", pl.const(0, pl.INT64), 6144)]],
        li_transfer_inline2502__ssa_v0: pl.InOut[pl.Tensor[[1536, 1], pl.FP32, pl.MemRef("mem_ddr_9", pl.const(0, pl.INT64), 6144)]],
        position_ids_t1__ssa_v0: pl.Tensor[[T_DYN, 1], pl.INT64, pl.MemRef("mem_ddr_10", pl.const(0, pl.INT64), 0)],
        ori_block_table__ssa_v0: pl.Tensor[[B_DYN, ORIGINAL_TABLE_COLUMNS_DYN], pl.INT32, pl.MemRef("mem_ddr_11", pl.const(0, pl.INT64), 0)],
        ori_kv_flat_inline2493__ssa_v1: pl.Tensor[[ori_block_num_inline2459__ssa_v0 * pl.const(32, pl.INDEX), 512], pl.BF16, pl.MemRef("mem_ddr_12", pl.const(0, pl.INT64), 0)],
        cmp_sparse_indices_inline2484__rv_v2: pl.Tensor[[t_dim_inline2491__ssa_v0, 512], pl.INT32, pl.MemRef("mem_ddr_13", pl.const(0, pl.INT64), 0)],
        cmp_block_table__ssa_v0: pl.Tensor[[B_DYN, COMPRESSED_TABLE_COLUMNS_DYN], pl.INT32, pl.MemRef("mem_ddr_14", pl.const(0, pl.INT64), 0)],
        cmp_kv_flat_inline2430__ssa_v0: pl.Tensor[[cmp_block_num_inline2525__ssa_v0 * pl.const(32, pl.INDEX), 512], pl.BF16, pl.MemRef("mem_ddr_15", pl.const(0, pl.INT64), 0)],
        sparse_bias_inline2455__rv_v2: pl.Tensor[[t_dim_inline2491__ssa_v0, 1024], pl.FP32, pl.MemRef("mem_ddr_16", pl.const(0, pl.INT64), 0)],
        alpha_transfer_inline2486__ssa_v0: pl.InOut[pl.Tensor[[1536, 1], pl.FP32, pl.MemRef("mem_ddr_17", pl.const(0, pl.INT64), 6144)]],
        attn_mi_inline2490__ssa_v0: pl.Out[pl.Tensor[[t_heads_inline2469__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_18", pl.const(0, pl.INT64), 0)]],
        attn_li_inline2516__ssa_v0: pl.Out[pl.Tensor[[t_heads_inline2469__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_19", pl.const(0, pl.INT64), 0)]],
        attn_oi_inline2428__ssa_v0: pl.Out[pl.Tensor[[t_heads_inline2469__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_20", pl.const(0, pl.INT64), 0)]],
    ) -> tuple[pl.Tensor[[t_heads_inline2469__ssa_v0, 1], pl.FP32], pl.Tensor[[t_heads_inline2469__ssa_v0, 1], pl.FP32], pl.Tensor[[t_heads_inline2469__ssa_v0, 512], pl.FP32]]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        ret__tmp_v0: pl.Tuple[
            pl.Tensor[[12288, 512], pl.BF16],
            pl.Tensor[[1536, 512], pl.FP32],
            pl.Tensor[[1536, 512], pl.BF16],
            pl.Tensor[[1536, 512], pl.FP32],
            pl.Tensor[[1536, 1], pl.FP32],
            pl.Tensor[[1536, 1], pl.FP32],
            pl.Tensor[[1536, 1], pl.FP32],
            pl.Tensor[[t_heads_inline2469__ssa_v0, 1], pl.FP32],
            pl.Tensor[[t_heads_inline2469__ssa_v0, 1], pl.FP32],
            pl.Tensor[[t_heads_inline2469__ssa_v0, 512], pl.FP32],
        ] = self.qk_pv(
            ffts_workspace_inline2438__ssa_v0,
            t_dim_inline2491__ssa_v0,
            q_flat_inline2480__ssa_v0,
            valid_block_mask_inline2452__ssa_v0,
            kv_transfer_inline2426__ssa_v0,
            score_transfer_inline2447__ssa_v0,
            probability_transfer_inline2550__ssa_v0,
            pv_transfer_inline2437__ssa_v0,
            attn_sink_col_inline2429__ssa_v0,
            mi_transfer_inline2500__ssa_v0,
            li_transfer_inline2502__ssa_v0,
            position_ids_t1__ssa_v0,
            ori_block_table__ssa_v0,
            ori_kv_flat_inline2493__ssa_v1,
            cmp_sparse_indices_inline2484__rv_v2,
            cmp_block_table__ssa_v0,
            cmp_kv_flat_inline2430__ssa_v0,
            sparse_bias_inline2455__rv_v2,
            alpha_transfer_inline2486__ssa_v0,
            attn_mi_inline2490__ssa_v0,
            attn_li_inline2516__ssa_v0,
            attn_oi_inline2428__ssa_v0,
            attrs={
                "arg_directions": [
                    pl.adir.input,
                    pl.adir.scalar,
                    pl.adir.input,
                    pl.adir.input,
                    pl.adir.inout,
                    pl.adir.inout,
                    pl.adir.inout,
                    pl.adir.inout,
                    pl.adir.input,
                    pl.adir.inout,
                    pl.adir.inout,
                    pl.adir.input,
                    pl.adir.input,
                    pl.adir.input,
                    pl.adir.input,
                    pl.adir.input,
                    pl.adir.input,
                    pl.adir.input,
                    pl.adir.inout,
                    pl.adir.output_existing,
                    pl.adir.output_existing,
                    pl.adir.output_existing,
                ]
            },
        )
        kv_transfer_inline2426__ssa_v1: pl.Tensor[[12288, 512], pl.BF16, pl.MemRef("mem_ddr_21", pl.const(0, pl.INT64), 12582912)] = ret__tmp_v0[0]
        score_transfer_inline2447__ssa_v1: pl.Tensor[[1536, 512], pl.FP32, pl.MemRef("mem_ddr_22", pl.const(0, pl.INT64), 3145728)] = ret__tmp_v0[1]
        probability_transfer_inline2550__ssa_v1: pl.Tensor[[1536, 512], pl.BF16, pl.MemRef("mem_ddr_23", pl.const(0, pl.INT64), 1572864)] = ret__tmp_v0[2]
        pv_transfer_inline2437__ssa_v1: pl.Tensor[[1536, 512], pl.FP32, pl.MemRef("mem_ddr_24", pl.const(0, pl.INT64), 3145728)] = ret__tmp_v0[3]
        mi_transfer_inline2500__ssa_v1: pl.Tensor[[1536, 1], pl.FP32, pl.MemRef("mem_ddr_25", pl.const(0, pl.INT64), 6144)] = ret__tmp_v0[4]
        li_transfer_inline2502__ssa_v1: pl.Tensor[[1536, 1], pl.FP32, pl.MemRef("mem_ddr_26", pl.const(0, pl.INT64), 6144)] = ret__tmp_v0[5]
        alpha_transfer_inline2486__ssa_v1: pl.Tensor[[1536, 1], pl.FP32, pl.MemRef("mem_ddr_27", pl.const(0, pl.INT64), 6144)] = ret__tmp_v0[6]
        attn_mi_inline2490__ssa_v1: pl.Tensor[[t_heads_inline2469__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_28", pl.const(0, pl.INT64), 0)] = ret__tmp_v0[7]
        attn_li_inline2516__ssa_v1: pl.Tensor[[t_heads_inline2469__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_29", pl.const(0, pl.INT64), 0)] = ret__tmp_v0[8]
        attn_oi_inline2428__ssa_v1: pl.Tensor[[t_heads_inline2469__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_30", pl.const(0, pl.INT64), 0)] = ret__tmp_v0[9]
        return attn_mi_inline2490__ssa_v0, attn_li_inline2516__ssa_v0, attn_oi_inline2428__ssa_v0

    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def qproj_dequant_rms_nope_rope(
        q_flat_inline3174__ssa_v0: pl.Out[pl.Tensor[[t_dim_inline3159__ssa_v0, 32768], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)]],
        tile_rows_inline9_inline1800__ssa_v0: pl.Scalar[pl.INDEX],
        tile_base_inline10_inline1796__idx_v0: pl.Scalar[pl.INDEX],
        qr_scale_pad_store_inline1797__rv_v2: pl.Tensor[[512, 1], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 2048)],
        freqs_cos__ssa_v0: pl.Tensor[[T_DYN, 64], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)],
        q_rope_sin_signed_inline197__ssa_v0: pl.Tensor[[t_dim_inline198__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)],
        q_rope_swap_idx_inline196__ssa_v0: pl.Tensor[[t_dim_inline198__ssa_v0, 64], pl.INT32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 0)],
        q_proj_i32_inline6_inline1803__ssa_v9: pl.Tensor[[qproj_t_matmul_inline8_inline1802__ssa_v0, 32768], pl.INT32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 0)],
        wq_b_scale__ssa_v0: pl.Tensor[[32768], pl.FP32, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 131072)],
    ):
        pl.func_attr({"mx_tensor_views_blocked": True})
        mem_vec_7: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32)
        mem_vec_8: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 2048)
        mem_vec_9: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 2048)
        mem_vec_10: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 2048)
        mem_vec_11: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
        mem_vec_13: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
        mem_vec_16: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
        mem_vec_23: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
        mem_vec_35: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        mem_vec_36: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        mem_vec_43: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
        mem_vec_50: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
        mem_vec_62: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        mem_vec_63: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        dq_worker_inline3135__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        for dq_work_inline3146__idx_v0, (q_flat_inline3174__iter_v1,) in pl.range(
            dq_worker_inline3135__ssa_v0, (tile_rows_inline9_inline1800__ssa_v0 + 7) // 8 * 16, 48, init_values=(q_flat_inline3174__ssa_v0,)
        ):
            hg_inline3136__ssa_v0: pl.Scalar[pl.INDEX] = dq_work_inline3146__idx_v0 % 16 * 4
            tg_inline3150__ssa_v0: pl.Scalar[pl.INDEX] = dq_work_inline3146__idx_v0 // 16 * 8
            out_tg_inline3138__ssa_v0: pl.Scalar[pl.INDEX] = tile_base_inline10_inline1796__idx_v0 + tg_inline3150__ssa_v0
            if tg_inline3150__ssa_v0 + 8 <= tile_rows_inline9_inline1800__ssa_v0:
                qr_scale_dq_t_inline3143__tile: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_7, pl.const(101376, pl.INT64), 32), pl.Mem.Vec] = pl.tile.load(
                    qr_scale_pad_store_inline1797__rv_v2, [tg_inline3150__ssa_v0, 0], [8, 1], [8, 1], target_memory=pl.Mem.Vec
                )
                q_cos_il_inline3137__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_8, pl.const(101408, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.load(
                    freqs_cos__ssa_v0, [out_tg_inline3138__ssa_v0, 0], [8, 64], [8, 64], target_memory=pl.Mem.Vec
                )
                q_sin_signed_inline3176__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_9, pl.const(103456, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.load(
                    q_rope_sin_signed_inline197__ssa_v0, [out_tg_inline3138__ssa_v0, 0], [8, 64], [8, 64], target_memory=pl.Mem.Vec
                )
                q_swap_idx_inline3162__tile: pl.Tile[[8, 64], pl.INT32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.load(
                    q_rope_swap_idx_inline196__ssa_v0, [out_tg_inline3138__ssa_v0, 0], [8, 64], [8, 64], target_memory=pl.Mem.Vec
                )
                for h_inner_inline3121__idx_v0, (q_flat_inline3174__iter_v3,) in pl.range(0, 4, 2, init_values=(q_flat_inline3174__iter_v1,)):
                    h_inline3157__ssa_v0: pl.Scalar[pl.INDEX] = hg_inline3136__ssa_v0 + h_inner_inline3121__idx_v0
                    h0_inline3131__ssa_v0: pl.Scalar[pl.INDEX] = h_inline3157__ssa_v0 * 512
                    h_inline3157__ssa_v0_1: pl.Scalar[pl.INDEX] = hg_inline3136__ssa_v0 + (h_inner_inline3121__idx_v0 + 1)
                    h0_inline3131__ssa_v0_1: pl.Scalar[pl.INDEX] = h_inline3157__ssa_v0_1 * 512
                    q_head_acc_inline3142__tile: pl.Tile[[8, 512], pl.INT32, pl.MemRef(mem_vec_11, pl.const(2048, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.load(
                        q_proj_i32_inline6_inline1803__ssa_v9, [tg_inline3150__ssa_v0, h0_inline3131__ssa_v0], [8, 512], [8, 512], target_memory=pl.Mem.Vec
                    )
                    t__tile: pl.Tile[[512], pl.FP32, pl.MemRef(mem_vec_23, pl.const(51200, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.load(
                        wq_b_scale__ssa_v0, [h0_inline3131__ssa_v0], [512], [512], target_memory=pl.Mem.Vec
                    )
                    q_head_acc_inline3142__tile_1: pl.Tile[[8, 512], pl.INT32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.load(
                        q_proj_i32_inline6_inline1803__ssa_v9, [tg_inline3150__ssa_v0, h0_inline3131__ssa_v0_1], [8, 512], [8, 512], target_memory=pl.Mem.Vec
                    )
                    t__tile_1: pl.Tile[[512], pl.FP32, pl.MemRef(mem_vec_50, pl.const(84480, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.load(
                        wq_b_scale__ssa_v0, [h0_inline3131__ssa_v0_1], [512], [512], target_memory=pl.Mem.Vec
                    )
                    q_head_scale_inline3129__tile: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_23, pl.const(51200, pl.INT64), 2048), pl.Mem.Vec] = t__tile
                    q_head_acc_fp32_inline3126__tile: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_11, pl.const(2048, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.cast(
                        q_head_acc_inline3142__tile, target_type=pl.FP32, mode="none"
                    )
                    t__tile_2: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_16, pl.const(34816, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.full([8, 512], dtype=pl.FP32, value=1.0)
                    t__tile_3: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_16, pl.const(34816, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.row_expand_mul(t__tile_2, qr_scale_dq_t_inline3143__tile)
                    q_head_scale_combined_inline3156__tile: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_16, pl.const(34816, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.col_expand_mul(
                        t__tile_3, q_head_scale_inline3129__tile
                    )
                    q_head_dq_inline3155__tile: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_11, pl.const(2048, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.mul(
                        q_head_acc_fp32_inline3126__tile, q_head_scale_combined_inline3156__tile
                    )
                    t__tile_4: pl.Tile[[8, 512], pl.BF16, pl.MemRef(mem_vec_16, pl.const(34816, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.cast(
                        q_head_dq_inline3155__tile, target_type=pl.BF16, mode="rint"
                    )
                    q_head_dq_v1_inline3122__tile: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_11, pl.const(2048, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.cast(
                        t__tile_4, target_type=pl.FP32, mode="round"
                    )
                    q_head_sq_inline3132__tile: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_16, pl.const(34816, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.mul(
                        q_head_dq_v1_inline3122__tile, q_head_dq_v1_inline3122__tile
                    )
                    tmp_tile: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_23, pl.const(51200, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.create([8, 512], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                    q_head_sq_row_inline3119__tile: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_35, pl.const(67584, pl.INT64), 32), pl.Mem.Vec] = pl.tile.row_sum(q_head_sq_inline3132__tile, tmp_tile)
                    q_head_sq_sum_inline3125__tile: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_35, pl.const(67584, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(q_head_sq_row_inline3119__tile, [1, 8])
                    q_head_sq_mean_inline3152__tile: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_16, pl.const(34816, pl.INT64), 32), pl.Mem.Vec] = pl.tile.muls(
                        q_head_sq_sum_inline3125__tile, 0.001953125
                    )
                    q_head_var_inline3117__tile: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_16, pl.const(34816, pl.INT64), 32), pl.Mem.Vec] = pl.tile.adds(
                        q_head_sq_mean_inline3152__tile, 9.9999999999999995e-07
                    )
                    rsqrt_tmp: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_23, pl.const(51200, pl.INT64), 32), pl.Mem.Vec] = pl.tile.create([1, 8], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                    q_head_inv_rms_inline3116__tile: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_35, pl.const(67584, pl.INT64), 32), pl.Mem.Vec] = pl.tile.rsqrt(q_head_var_inline3117__tile, rsqrt_tmp)
                    q_head_inv_rms_t_inline3133__tile: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_35, pl.const(67584, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(
                        q_head_inv_rms_inline3116__tile, [8, 1]
                    )
                    t__tile_5: pl.Tile[[8, 448], pl.FP32, pl.MemRef(mem_vec_11, pl.const(2048, pl.INT64), 16128), pl.Mem.Vec] = pl.tile.slice(q_head_dq_v1_inline3122__tile, [8, 448], [0, 0])
                    q_nope_normed_inline3130__tile: pl.Tile[[8, 448], pl.FP32, pl.MemRef(mem_vec_16, pl.const(34816, pl.INT64), 14336), pl.Mem.Vec] = pl.tile.row_expand_mul(
                        t__tile_5, q_head_inv_rms_t_inline3133__tile
                    )
                    q_nope_bf16_inline3145__tile: pl.Tile[[8, 448], pl.BF16, pl.MemRef(mem_vec_16, pl.const(34816, pl.INT64), 7168), pl.Mem.Vec] = pl.tile.cast(
                        q_nope_normed_inline3130__tile, target_type=pl.BF16, mode="rint"
                    )
                    q_rope_chunk_raw_inline3164__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_11, pl.const(3840, pl.INT64), 14592), pl.Mem.Vec] = pl.tile.slice(
                        q_head_dq_v1_inline3122__tile, [8, 64], [0, 448]
                    )
                    q_rope_chunk_inline3179__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_11, pl.const(2048, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.row_expand_mul(
                        q_rope_chunk_raw_inline3164__tile, q_head_inv_rms_t_inline3133__tile
                    )
                    t__tile_6: pl.Tile[[8, 64], pl.BF16, pl.MemRef(mem_vec_23, pl.const(51200, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.cast(
                        q_rope_chunk_inline3179__tile, target_type=pl.BF16, mode="rint"
                    )
                    q_rope_chunk_v1_inline3163__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_11, pl.const(2048, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(
                        t__tile_6, target_type=pl.FP32, mode="round"
                    )
                    gather_acc_init: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_23, pl.const(51200, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.create([8, 64], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                    for gather_lv, (gather_ia,) in pl.range(8, init_values=(gather_acc_init,)):
                        gather_inp_row: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_11, pl.const(2048, pl.INT64), 256), pl.Mem.Vec] = pl.tile.slice(
                            q_rope_chunk_v1_inline3163__tile, [1, 64], [gather_lv, 0], [1, 64]
                        )
                        gather_idx_row: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 256), pl.Mem.Vec] = pl.tile.slice(
                            q_swap_idx_inline3162__tile, [1, 64], [gather_lv, 0], [1, 64]
                        )
                        gather_row_tmp: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_35, pl.const(67584, pl.INT64), 256), pl.Mem.Vec] = pl.tile.create(
                            [1, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec
                        )
                        gather_row: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_36, pl.const(67840, pl.INT64), 256), pl.Mem.Vec] = pl.tile.gather(gather_inp_row, gather_idx_row, gather_row_tmp)
                        gather_asmbl: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_23, pl.const(51200, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.assemble(gather_ia, gather_row, [gather_lv, 0])
                        gather_rv: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_23, pl.const(51200, pl.INT64), 2048), pl.Mem.Vec] = pl.yield_(gather_asmbl)
                    q_rope_swapped_inline3153__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_23, pl.const(51200, pl.INT64), 2048), pl.Mem.Vec] = gather_rv
                    q_rope_base_inline3144__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_11, pl.const(2048, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.mul(
                        q_rope_chunk_v1_inline3163__tile, q_cos_il_inline3137__tile
                    )
                    q_rope_delta_inline3167__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_23, pl.const(51200, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.mul(
                        q_rope_swapped_inline3153__tile, q_sin_signed_inline3176__tile
                    )
                    q_rope_rot_inline3123__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_11, pl.const(2048, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.add(
                        q_rope_base_inline3144__tile, q_rope_delta_inline3167__tile
                    )
                    q_rope_bf16_inline3165__tile: pl.Tile[[8, 64], pl.BF16, pl.MemRef(mem_vec_11, pl.const(2048, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.cast(
                        q_rope_rot_inline3123__tile, target_type=pl.BF16, mode="rint"
                    )
                    q_flat_inline3174__tile: pl.Tensor[[t_dim_inline3159__ssa_v0, 32768], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                        q_nope_bf16_inline3145__tile, [out_tg_inline3138__ssa_v0, h0_inline3131__ssa_v0], q_flat_inline3174__iter_v3
                    )
                    q_flat_inline3174__tile_1: pl.Tensor[[t_dim_inline3159__ssa_v0, 32768], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                        q_rope_bf16_inline3165__tile, [out_tg_inline3138__ssa_v0, h0_inline3131__ssa_v0 + 448], q_flat_inline3174__tile
                    )
                    q_head_scale_inline3129__tile_1: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_50, pl.const(84480, pl.INT64), 2048), pl.Mem.Vec] = t__tile_1
                    q_head_acc_fp32_inline3126__tile_1: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.cast(
                        q_head_acc_inline3142__tile_1, target_type=pl.FP32, mode="none"
                    )
                    t__tile_7: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_43, pl.const(68096, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.full([8, 512], dtype=pl.FP32, value=1.0)
                    t__tile_8: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_43, pl.const(68096, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.row_expand_mul(t__tile_7, qr_scale_dq_t_inline3143__tile)
                    q_head_scale_combined_inline3156__tile_1: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_43, pl.const(68096, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.col_expand_mul(
                        t__tile_8, q_head_scale_inline3129__tile_1
                    )
                    q_head_dq_inline3155__tile_1: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.mul(
                        q_head_acc_fp32_inline3126__tile_1, q_head_scale_combined_inline3156__tile_1
                    )
                    t__tile_9: pl.Tile[[8, 512], pl.BF16, pl.MemRef(mem_vec_43, pl.const(68096, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.cast(
                        q_head_dq_inline3155__tile_1, target_type=pl.BF16, mode="rint"
                    )
                    q_head_dq_v1_inline3122__tile_1: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.cast(
                        t__tile_9, target_type=pl.FP32, mode="round"
                    )
                    q_head_sq_inline3132__tile_1: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_43, pl.const(68096, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.mul(
                        q_head_dq_v1_inline3122__tile_1, q_head_dq_v1_inline3122__tile_1
                    )
                    tmp_tile_1: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_50, pl.const(84480, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.create([8, 512], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                    q_head_sq_row_inline3119__tile_1: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_62, pl.const(100864, pl.INT64), 32), pl.Mem.Vec] = pl.tile.row_sum(
                        q_head_sq_inline3132__tile_1, tmp_tile_1
                    )
                    q_head_sq_sum_inline3125__tile_1: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_62, pl.const(100864, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(
                        q_head_sq_row_inline3119__tile_1, [1, 8]
                    )
                    q_head_sq_mean_inline3152__tile_1: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_43, pl.const(68096, pl.INT64), 32), pl.Mem.Vec] = pl.tile.muls(
                        q_head_sq_sum_inline3125__tile_1, 0.001953125
                    )
                    q_head_var_inline3117__tile_1: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_43, pl.const(68096, pl.INT64), 32), pl.Mem.Vec] = pl.tile.adds(
                        q_head_sq_mean_inline3152__tile_1, 9.9999999999999995e-07
                    )
                    rsqrt_tmp_1: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_50, pl.const(84480, pl.INT64), 32), pl.Mem.Vec] = pl.tile.create([1, 8], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                    q_head_inv_rms_inline3116__tile_1: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_62, pl.const(100864, pl.INT64), 32), pl.Mem.Vec] = pl.tile.rsqrt(
                        q_head_var_inline3117__tile_1, rsqrt_tmp_1
                    )
                    q_head_inv_rms_t_inline3133__tile_1: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_62, pl.const(100864, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(
                        q_head_inv_rms_inline3116__tile_1, [8, 1]
                    )
                    t__tile_10: pl.Tile[[8, 448], pl.FP32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 16128), pl.Mem.Vec] = pl.tile.slice(q_head_dq_v1_inline3122__tile_1, [8, 448], [0, 0])
                    q_nope_normed_inline3130__tile_1: pl.Tile[[8, 448], pl.FP32, pl.MemRef(mem_vec_43, pl.const(68096, pl.INT64), 14336), pl.Mem.Vec] = pl.tile.row_expand_mul(
                        t__tile_10, q_head_inv_rms_t_inline3133__tile_1
                    )
                    q_nope_bf16_inline3145__tile_1: pl.Tile[[8, 448], pl.BF16, pl.MemRef(mem_vec_43, pl.const(68096, pl.INT64), 7168), pl.Mem.Vec] = pl.tile.cast(
                        q_nope_normed_inline3130__tile_1, target_type=pl.BF16, mode="rint"
                    )
                    q_rope_chunk_raw_inline3164__tile_1: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(20224, pl.INT64), 14592), pl.Mem.Vec] = pl.tile.slice(
                        q_head_dq_v1_inline3122__tile_1, [8, 64], [0, 448]
                    )
                    q_rope_chunk_inline3179__tile_1: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.row_expand_mul(
                        q_rope_chunk_raw_inline3164__tile_1, q_head_inv_rms_t_inline3133__tile_1
                    )
                    t__tile_11: pl.Tile[[8, 64], pl.BF16, pl.MemRef(mem_vec_50, pl.const(84480, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.cast(
                        q_rope_chunk_inline3179__tile_1, target_type=pl.BF16, mode="rint"
                    )
                    q_rope_chunk_v1_inline3163__tile_1: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(
                        t__tile_11, target_type=pl.FP32, mode="round"
                    )
                    gather_acc_init_1: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_50, pl.const(84480, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.create([8, 64], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                    for gather_lv_1, (gather_ia_1,) in pl.range(8, init_values=(gather_acc_init_1,)):
                        gather_inp_row_1: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 256), pl.Mem.Vec] = pl.tile.slice(
                            q_rope_chunk_v1_inline3163__tile_1, [1, 64], [gather_lv_1, 0], [1, 64]
                        )
                        gather_idx_row_1: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 256), pl.Mem.Vec] = pl.tile.slice(
                            q_swap_idx_inline3162__tile, [1, 64], [gather_lv_1, 0], [1, 64]
                        )
                        gather_row_tmp_1: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_62, pl.const(100864, pl.INT64), 256), pl.Mem.Vec] = pl.tile.create(
                            [1, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec
                        )
                        gather_row_1: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_63, pl.const(101120, pl.INT64), 256), pl.Mem.Vec] = pl.tile.gather(
                            gather_inp_row_1, gather_idx_row_1, gather_row_tmp_1
                        )
                        gather_asmbl_1: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_50, pl.const(84480, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.assemble(gather_ia_1, gather_row_1, [gather_lv_1, 0])
                        gather_rv_1: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_50, pl.const(84480, pl.INT64), 2048), pl.Mem.Vec] = pl.yield_(gather_asmbl_1)
                    q_rope_swapped_inline3153__tile_1: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_50, pl.const(84480, pl.INT64), 2048), pl.Mem.Vec] = gather_rv_1
                    q_rope_base_inline3144__tile_1: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.mul(
                        q_rope_chunk_v1_inline3163__tile_1, q_cos_il_inline3137__tile
                    )
                    q_rope_delta_inline3167__tile_1: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_50, pl.const(84480, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.mul(
                        q_rope_swapped_inline3153__tile_1, q_sin_signed_inline3176__tile
                    )
                    q_rope_rot_inline3123__tile_1: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.add(
                        q_rope_base_inline3144__tile_1, q_rope_delta_inline3167__tile_1
                    )
                    q_rope_bf16_inline3165__tile_1: pl.Tile[[8, 64], pl.BF16, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.cast(
                        q_rope_rot_inline3123__tile_1, target_type=pl.BF16, mode="rint"
                    )
                    q_flat_inline3174__tile_2: pl.Tensor[[t_dim_inline3159__ssa_v0, 32768], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                        q_nope_bf16_inline3145__tile_1, [out_tg_inline3138__ssa_v0, h0_inline3131__ssa_v0_1], q_flat_inline3174__tile_1
                    )
                    q_flat_inline3174__tile_3: pl.Tensor[[t_dim_inline3159__ssa_v0, 32768], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                        q_rope_bf16_inline3165__tile_1, [out_tg_inline3138__ssa_v0, h0_inline3131__ssa_v0_1 + 448], q_flat_inline3174__tile_2
                    )
                    q_flat_inline3174__rv_v4: pl.Tensor[[t_dim_inline3159__ssa_v0, 32768], pl.BF16, pl.MemRef("mem_ddr_69", pl.const(0, pl.INT64), 0)] = pl.yield_(q_flat_inline3174__tile_3)
                q_flat_inline3174__phi_v7: pl.Tensor[[t_dim_inline3159__ssa_v0, 32768], pl.BF16, pl.MemRef("mem_ddr_123", pl.const(0, pl.INT64), 0)] = pl.yield_(q_flat_inline3174__rv_v4)
            else:
                valid_tail_rows_inline3177__ssa_v0: pl.Scalar[pl.INDEX] = tile_rows_inline9_inline1800__ssa_v0 - tg_inline3150__ssa_v0
                qr_scale_dq_tail_inline3166__ssa_v0: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 32), pl.Mem.Vec] = pl.tile.load(
                    qr_scale_pad_store_inline1797__rv_v2, [tg_inline3150__ssa_v0, 0], [8, 1], [8, 1], target_memory=pl.Mem.Vec
                )
                q_cos_il_tail_inline3169__ssa_v0: pl.Tile[
                    [8, 64], pl.FP32, pl.MemRef(mem_vec_23, pl.const(51200, pl.INT64), 2048), pl.Mem.Vec, pl.TileView(valid_shape=[valid_tail_rows_inline3177__ssa_v0, 64])
                ] = pl.tile.load(freqs_cos__ssa_v0, [out_tg_inline3138__ssa_v0, 0], [8, 64], [valid_tail_rows_inline3177__ssa_v0, 64], target_memory=pl.Mem.Vec)
                q_sin_signed_tail_inline3171__ssa_v0: pl.Tile[
                    [8, 64], pl.FP32, pl.MemRef(mem_vec_43, pl.const(68096, pl.INT64), 2048), pl.Mem.Vec, pl.TileView(valid_shape=[valid_tail_rows_inline3177__ssa_v0, 64])
                ] = pl.tile.load(q_rope_sin_signed_inline197__ssa_v0, [out_tg_inline3138__ssa_v0, 0], [8, 64], [valid_tail_rows_inline3177__ssa_v0, 64], target_memory=pl.Mem.Vec)
                t__tmp_v53: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_11, pl.const(2048, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.full([8, 64], dtype=pl.FP32, value=1.0)
                t__ci_tmp_v0: pl.Tile[[1, 192], pl.FP32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 768), pl.Mem.Vec] = pl.tile.create([1, 192], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                t__tmp_v54: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_16, pl.const(34816, pl.INT64), 256), pl.Mem.Vec] = pl.tile.ci(
                    pl.const(0, pl.INT32), [1, 64], tmp=t__ci_tmp_v0, dtype=pl.INT32, descending=False
                )
                t__tmp_v55: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 256), pl.Mem.Vec] = pl.tile.cast(t__tmp_v54, target_type=pl.FP32, mode="round")
                q_col_inline3172__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_11, pl.const(2048, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.col_expand_mul(t__tmp_v53, t__tmp_v55)
                t__tmp_v56: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.muls(q_col_inline3172__ssa_v0, 0.5)
                t__tmp_v57: pl.Tile[[8, 64], pl.INT32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(t__tmp_v56, target_type=pl.INT32, mode="trunc")
                q_dup_f_inline3141__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(t__tmp_v57, target_type=pl.FP32, mode="round")
                t__tmp_v58: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.muls(q_dup_f_inline3141__ssa_v0, 2.0)
                q_lane_inline3148__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.sub(q_col_inline3172__ssa_v0, t__tmp_v58)
                t__tmp_v59: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_11, pl.const(2048, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.adds(q_col_inline3172__ssa_v0, 1.0)
                t__tmp_v60: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.muls(q_lane_inline3148__ssa_v0, 2.0)
                q_swap_f_inline3173__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_11, pl.const(2048, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.sub(t__tmp_v59, t__tmp_v60)
                t__ci_tmp_v1: pl.Tile[[1, 192], pl.FP32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 768), pl.Mem.Vec] = pl.tile.create([1, 192], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                t__tmp_v61: pl.Tile[[1, 8], pl.INT32, pl.MemRef(mem_vec_16, pl.const(34816, pl.INT64), 32), pl.Mem.Vec] = pl.tile.ci(
                    pl.const(0, pl.INT32), [1, 8], tmp=t__ci_tmp_v1, dtype=pl.INT32, descending=False
                )
                t__tmp_v62: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 32), pl.Mem.Vec] = pl.tile.cast(t__tmp_v61, target_type=pl.FP32, mode="round")
                q_row_seed_inline3127__ssa_v0: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_16, pl.const(34816, pl.INT64), 32), pl.Mem.Vec] = pl.tile.muls(t__tmp_v62, 64.0)
                t__tmp_v63: pl.Tile[[64, 8], pl.FP32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.full([64, 8], dtype=pl.FP32, value=1.0)
                q_row_grid_inline3160__ssa_v0: pl.Tile[[64, 8], pl.FP32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.col_expand_mul(
                    t__tmp_v63, q_row_seed_inline3127__ssa_v0
                )
                transpose_tmp: pl.Tile[[64, 8], pl.FP32, pl.MemRef(mem_vec_16, pl.const(34816, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.create([64, 8], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                q_row_offset_inline3149__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_50, pl.const(84480, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.transpose(
                    q_row_grid_inline3160__ssa_v0, 0, 1, transpose_tmp
                )
                t__tmp_v64: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_11, pl.const(2048, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.add(q_swap_f_inline3173__ssa_v0, q_row_offset_inline3149__ssa_v0)
                q_swap_idx_tail_inline3175__ssa_v0: pl.Tile[[8, 64], pl.INT32, pl.MemRef(mem_vec_50, pl.const(84480, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(
                    t__tmp_v64, target_type=pl.INT32, mode="round"
                )
                q_head_reduce_tmp_inline3139__ssa_v0: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_11, pl.const(2048, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.create(
                    [8, 512], dtype=pl.FP32, target_memory=pl.Mem.Vec
                )
                q_gather_tmp_inline3118__ssa_v0: pl.Tile[[8, 64], pl.INT32, pl.MemRef(mem_vec_8, pl.const(101408, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.create(
                    [8, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec
                )
                for h_inner_tail_inline3168__idx_v0 in pl.range(4):
                    h_tail_inline3154__ssa_v0: pl.Scalar[pl.INDEX] = hg_inline3136__ssa_v0 + h_inner_tail_inline3168__idx_v0
                    h0_tail_inline3161__ssa_v0: pl.Scalar[pl.INDEX] = h_tail_inline3154__ssa_v0 * 512
                    q_head_acc_tail_inline3124__ssa_v0: pl.Tile[[8, 512], pl.INT32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.load(
                        q_proj_i32_inline6_inline1803__ssa_v9, [tg_inline3150__ssa_v0, h0_tail_inline3161__ssa_v0], [8, 512], [8, 512], target_memory=pl.Mem.Vec
                    )
                    q_head_scale_input_tail_inline3147__ssa_v0: pl.Tile[[512], pl.FP32, pl.MemRef(mem_vec_9, pl.const(103456, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.load(
                        wq_b_scale__ssa_v0, [h0_tail_inline3161__ssa_v0], [512], [512], target_memory=pl.Mem.Vec
                    )
                    q_head_scale_tail_inline3178__ssa_v0: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_9, pl.const(103456, pl.INT64), 2048), pl.Mem.Vec] = q_head_scale_input_tail_inline3147__ssa_v0
                    q_head_acc_fp32_tail_inline3180__ssa_v0: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.cast(
                        q_head_acc_tail_inline3124__ssa_v0, target_type=pl.FP32, mode="none"
                    )
                    t__tmp_v65: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_16, pl.const(34816, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.full([8, 512], dtype=pl.FP32, value=1.0)
                    t__tmp_v66: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_16, pl.const(34816, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.row_expand_mul(
                        t__tmp_v65, qr_scale_dq_tail_inline3166__ssa_v0
                    )
                    q_head_scale_combined_tail_inline3181__ssa_v0: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_16, pl.const(34816, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.col_expand_mul(
                        t__tmp_v66, q_head_scale_tail_inline3178__ssa_v0
                    )
                    q_head_dq_tail_inline3115__ssa_v0: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.mul(
                        q_head_acc_fp32_tail_inline3180__ssa_v0, q_head_scale_combined_tail_inline3181__ssa_v0
                    )
                    t__tmp_v67: pl.Tile[[8, 512], pl.BF16, pl.MemRef(mem_vec_16, pl.const(34816, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.cast(
                        q_head_dq_tail_inline3115__ssa_v0, target_type=pl.BF16, mode="rint"
                    )
                    q_head_dq_tail_v1_inline3114__ssa_v0: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.cast(
                        t__tmp_v67, target_type=pl.FP32, mode="round"
                    )
                    q_head_sq_tail_inline3113__ssa_v0: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_16, pl.const(34816, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.mul(
                        q_head_dq_tail_v1_inline3114__ssa_v0, q_head_dq_tail_v1_inline3114__ssa_v0
                    )
                    q_head_sq_sum_tail_inline3111__ssa_v0: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_9, pl.const(103456, pl.INT64), 32), pl.Mem.Vec] = pl.tile.row_sum(
                        q_head_sq_tail_inline3113__ssa_v0, q_head_reduce_tmp_inline3139__ssa_v0
                    )
                    t__rm_a0_tmp_v0: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_9, pl.const(103456, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(q_head_sq_sum_tail_inline3111__ssa_v0, [1, 8])
                    t__row_major_tmp_v1: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_16, pl.const(34816, pl.INT64), 32), pl.Mem.Vec] = pl.tile.muls(t__rm_a0_tmp_v0, 0.001953125)
                    t__tmp_v68: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_16, pl.const(34816, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(t__row_major_tmp_v1, [8, 1])
                    t__rm_a0_tmp_v2: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_16, pl.const(34816, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(t__tmp_v68, [1, 8])
                    t__row_major_tmp_v3: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_16, pl.const(34816, pl.INT64), 32), pl.Mem.Vec] = pl.tile.adds(t__rm_a0_tmp_v2, 9.9999999999999995e-07)
                    t__tmp_v69: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_16, pl.const(34816, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(t__row_major_tmp_v3, [8, 1])
                    t__rm_a0_tmp_v4: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_16, pl.const(34816, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(t__tmp_v69, [1, 8])
                    t__row_major_tmp_v5: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_16, pl.const(34816, pl.INT64), 32), pl.Mem.Vec] = pl.tile.sqrt(t__rm_a0_tmp_v4)
                    t__tmp_v70: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_16, pl.const(34816, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(t__row_major_tmp_v5, [8, 1])
                    q_head_inv_rms_tail_inline3128__rm_a0_tmp_v6: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_16, pl.const(34816, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(t__tmp_v70, [1, 8])
                    q_head_inv_rms_tail_inline3128__row_major_tmp_v7: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_9, pl.const(103456, pl.INT64), 32), pl.Mem.Vec] = pl.tile.recip(
                        q_head_inv_rms_tail_inline3128__rm_a0_tmp_v6
                    )
                    q_head_inv_rms_tail_inline3128__ssa_v0: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_9, pl.const(103456, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(
                        q_head_inv_rms_tail_inline3128__row_major_tmp_v7, [8, 1]
                    )
                    t__tmp_v71: pl.Tile[[8, 448], pl.FP32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 16128), pl.Mem.Vec] = pl.tile.slice(q_head_dq_tail_v1_inline3114__ssa_v0, [8, 448], [0, 0])
                    q_nope_normed_tail_inline3151__ssa_v0: pl.Tile[[8, 448], pl.FP32, pl.MemRef(mem_vec_16, pl.const(34816, pl.INT64), 14336), pl.Mem.Vec] = pl.tile.row_expand_mul(
                        t__tmp_v71, q_head_inv_rms_tail_inline3128__ssa_v0
                    )
                    q_nope_bf16_tail_inline3109__ssa_v0: pl.Tile[[8, 448], pl.BF16, pl.MemRef(mem_vec_16, pl.const(34816, pl.INT64), 7168), pl.Mem.Vec] = pl.tile.cast(
                        q_nope_normed_tail_inline3151__ssa_v0, target_type=pl.BF16, mode="rint"
                    )
                    q_nope_valid_inline3108__ssa_v0: pl.Tile[
                        [8, 448], pl.BF16, pl.MemRef(mem_vec_16, pl.const(34816, pl.INT64), 7168), pl.Mem.Vec, pl.TileView(valid_shape=[valid_tail_rows_inline3177__ssa_v0, 448])
                    ] = pl.tile.set_validshape(q_nope_bf16_tail_inline3109__ssa_v0, valid_tail_rows_inline3177__ssa_v0, 448)
                    pl.tile.store(q_nope_valid_inline3108__ssa_v0, [out_tg_inline3138__ssa_v0, h0_tail_inline3161__ssa_v0], q_flat_inline3174__iter_v1)
                    q_rope_chunk_raw_tail_inline3134__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(20224, pl.INT64), 14592), pl.Mem.Vec] = pl.tile.slice(
                        q_head_dq_tail_v1_inline3114__ssa_v0, [8, 64], [0, 448]
                    )
                    q_rope_chunk_tail_inline3120__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.row_expand_mul(
                        q_rope_chunk_raw_tail_inline3134__ssa_v0, q_head_inv_rms_tail_inline3128__ssa_v0
                    )
                    t__tmp_v72: pl.Tile[[8, 64], pl.BF16, pl.MemRef(mem_vec_16, pl.const(34816, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.cast(
                        q_rope_chunk_tail_inline3120__ssa_v0, target_type=pl.BF16, mode="rint"
                    )
                    q_rope_chunk_tail_v1_inline3110__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(
                        t__tmp_v72, target_type=pl.FP32, mode="round"
                    )
                    q_rope_swapped_tail_inline3140__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_16, pl.const(34816, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.gather(
                        q_rope_chunk_tail_v1_inline3110__ssa_v0, q_swap_idx_tail_inline3175__ssa_v0, q_gather_tmp_inline3118__ssa_v0
                    )
                    q_rope_base_tail_inline3158__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.mul(
                        q_rope_chunk_tail_v1_inline3110__ssa_v0, q_cos_il_tail_inline3169__ssa_v0
                    )
                    q_rope_delta_tail_inline3170__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_16, pl.const(34816, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.mul(
                        q_rope_swapped_tail_inline3140__ssa_v0, q_sin_signed_tail_inline3171__ssa_v0
                    )
                    q_rope_rot_tail_inline3107__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.add(
                        q_rope_base_tail_inline3158__ssa_v0, q_rope_delta_tail_inline3170__ssa_v0
                    )
                    q_rope_bf16_tail_inline3112__ssa_v0: pl.Tile[[8, 64], pl.BF16, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.cast(
                        q_rope_rot_tail_inline3107__ssa_v0, target_type=pl.BF16, mode="rint"
                    )
                    q_rope_valid_inline3106__ssa_v0: pl.Tile[
                        [8, 64], pl.BF16, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 1024), pl.Mem.Vec, pl.TileView(valid_shape=[valid_tail_rows_inline3177__ssa_v0, 64])
                    ] = pl.tile.set_validshape(q_rope_bf16_tail_inline3112__ssa_v0, valid_tail_rows_inline3177__ssa_v0, 64)
                    pl.tile.store(q_rope_valid_inline3106__ssa_v0, [out_tg_inline3138__ssa_v0, h0_tail_inline3161__ssa_v0 + 448], q_flat_inline3174__iter_v1)
                q_flat_inline3174__phi_v7: pl.Tensor[[t_dim_inline3159__ssa_v0, 32768], pl.BF16, pl.MemRef("mem_ddr_123", pl.const(0, pl.INT64), 0)] = pl.yield_(q_flat_inline3174__iter_v1)
            q_flat_inline3174__rv_v2: pl.Tensor[[t_dim_inline3159__ssa_v0, 32768], pl.BF16, pl.MemRef("mem_ddr_124", pl.const(0, pl.INT64), 0)] = pl.yield_(q_flat_inline3174__phi_v7)
        return

    @pl.function(type=pl.FunctionType.Spmd)
    def qproj_dequant_rms_nope_rope_spmd(
        self,
        q_flat_inline3174__ssa_v0: pl.Out[pl.Tensor[[t_dim_inline3159__ssa_v0, 32768], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)]],
        tile_rows_inline9_inline1800__ssa_v0: pl.Scalar[pl.INDEX],
        tile_base_inline10_inline1796__idx_v0: pl.Scalar[pl.INDEX],
        qr_scale_pad_store_inline1797__rv_v2: pl.Tensor[[512, 1], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 2048)],
        freqs_cos__ssa_v0: pl.Tensor[[T_DYN, 64], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)],
        q_rope_sin_signed_inline197__ssa_v0: pl.Tensor[[t_dim_inline198__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)],
        q_rope_swap_idx_inline196__ssa_v0: pl.Tensor[[t_dim_inline198__ssa_v0, 64], pl.INT32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 0)],
        q_proj_i32_inline6_inline1803__ssa_v9: pl.Tensor[[qproj_t_matmul_inline8_inline1802__ssa_v0, 32768], pl.INT32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 0)],
        wq_b_scale__ssa_v0: pl.Tensor[[32768], pl.FP32, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 131072)],
    ):
        pl.func_attr({"mx_tensor_views_blocked": True})
        self.qproj_dequant_rms_nope_rope(
            q_flat_inline3174__ssa_v0,
            tile_rows_inline9_inline1800__ssa_v0,
            tile_base_inline10_inline1796__idx_v0,
            qr_scale_pad_store_inline1797__rv_v2,
            freqs_cos__ssa_v0,
            q_rope_sin_signed_inline197__ssa_v0,
            q_rope_swap_idx_inline196__ssa_v0,
            q_proj_i32_inline6_inline1803__ssa_v9,
            wq_b_scale__ssa_v0,
            attrs={"arg_directions": [pl.adir.output_existing, pl.adir.scalar, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input]},
        )

    @pl.function(type=pl.FunctionType.AIC, level=pl.Level.AIC, role=pl.Role.SubWorker)
    def qproj_matmul(
        q_proj_i32_inline6_inline1803__ssa_v0: pl.Out[pl.Tensor[[qproj_t_matmul_inline8_inline1802__ssa_v0, 32768], pl.INT32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)]],
        qproj_full_rows_inline3094__ssa_v0: pl.Scalar[pl.INDEX],
        qr_i8_matmul_inline1790__rv_v2: pl.Tensor[[512, 1024], pl.INT8, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 524288)],
        wq_b__ssa_v0: pl.Tensor[[1024, 32768], pl.INT8, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 33554432)],
        qproj_t_matmul_inline3092__ssa_v0: pl.Scalar[pl.INDEX],
        tile_rows_inline9_inline1800__ssa_v0: pl.Scalar[pl.INDEX],
    ) -> pl.Tensor[[qproj_t_matmul_inline8_inline1802__ssa_v0, 32768], pl.INT32]:
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
        qproj_worker_inline3096__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        for qproj_n_idx_inline3097__idx_v0, (q_proj_i32_inline6_inline1803__iter_v1,) in pl.range(qproj_worker_inline3096__ssa_v0, 64, 24, init_values=(q_proj_i32_inline6_inline1803__ssa_v0,)):
            w_col0_inline3098__ssa_v0: pl.Scalar[pl.INDEX] = qproj_n_idx_inline3097__idx_v0 * 512
            for t0_inline3103__idx_v0, (q_proj_i32_inline6_inline1803__iter_v3,) in pl.range(0, qproj_full_rows_inline3094__ssa_v0, 64, init_values=(q_proj_i32_inline6_inline1803__iter_v1,)):
                col_acc_inline3093__tile: pl.Tile[[64, 512], pl.INT32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 131072), pl.Mem.Acc] = pl.tile.create(
                    [64, 512], dtype=pl.INT32, target_memory=pl.Mem.Acc
                )
                for qr_proj_col0_inline3105__idx_v0, (col_acc_inline3093__iter_v1,) in pl.range(0, 1024, 512, init_values=(col_acc_inline3093__tile,)):
                    qr_i8_chunk_inline3099__tile: pl.Tile[[64, 256], pl.INT8, pl.MemRef(mem_mat_4, pl.const(0, pl.INT64), 16384), pl.Mem.Mat] = pl.tile.load(
                        qr_i8_matmul_inline1790__rv_v2, [t0_inline3103__idx_v0, qr_proj_col0_inline3105__idx_v0], [64, 256], [64, 256], target_memory=pl.Mem.Mat
                    )
                    wq_chunk_inline3100__tile: pl.Tile[[256, 512], pl.INT8, pl.MemRef(mem_mat_5, pl.const(16384, pl.INT64), 131072), pl.Mem.Mat] = pl.tile.load(
                        wq_b__ssa_v0, [qr_proj_col0_inline3105__idx_v0, w_col0_inline3098__ssa_v0], [256, 512], [256, 512], target_memory=pl.Mem.Mat
                    )
                    qr_i8_chunk_inline3099__tile_1: pl.Tile[[64, 256], pl.INT8, pl.MemRef(mem_mat_6, pl.const(147456, pl.INT64), 16384), pl.Mem.Mat] = pl.tile.load(
                        qr_i8_matmul_inline1790__rv_v2, [t0_inline3103__idx_v0, qr_proj_col0_inline3105__idx_v0 + 256], [64, 256], [64, 256], target_memory=pl.Mem.Mat
                    )
                    wq_chunk_inline3100__tile_1: pl.Tile[[256, 512], pl.INT8, pl.MemRef(mem_mat_7, pl.const(163840, pl.INT64), 131072), pl.Mem.Mat] = pl.tile.load(
                        wq_b__ssa_v0, [qr_proj_col0_inline3105__idx_v0 + 256, w_col0_inline3098__ssa_v0], [256, 512], [256, 512], target_memory=pl.Mem.Mat
                    )
                    for col_acc_inline3093__tile_l0_ko, (col_acc_inline3093__tile_l0_c,) in pl.range(0, 256, 128, init_values=(col_acc_inline3093__iter_v1,)):
                        col_acc_inline3093__tile_l0_a: pl.Tile[[64, 64], pl.INT8, pl.MemRef(mem_left_8, pl.const(12288, pl.INT64), 4096), pl.Mem.Left, pl.TileView(blayout=pl.TileLayout.row_major)] = (
                            pl.tile.extract(qr_i8_chunk_inline3099__tile, 0, col_acc_inline3093__tile_l0_ko, [64, 64], target_memory=pl.Mem.Left)
                        )
                        col_acc_inline3093__tile_l0_b: pl.Tile[[64, 512], pl.INT8, pl.MemRef(mem_right_9, pl.const(32768, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                            wq_chunk_inline3100__tile, col_acc_inline3093__tile_l0_ko, 0, [64, 512], target_memory=pl.Mem.Right
                        )
                        col_acc_inline3093__tile_l0_a_1: pl.Tile[[64, 64], pl.INT8, pl.MemRef(mem_left_10, pl.const(0, pl.INT64), 4096), pl.Mem.Left, pl.TileView(blayout=pl.TileLayout.row_major)] = (
                            pl.tile.extract(qr_i8_chunk_inline3099__tile, 0, col_acc_inline3093__tile_l0_ko + 64, [64, 64], target_memory=pl.Mem.Left)
                        )
                        col_acc_inline3093__tile_l0_b_1: pl.Tile[[64, 512], pl.INT8, pl.MemRef(mem_right_11, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                            wq_chunk_inline3100__tile, col_acc_inline3093__tile_l0_ko + 64, 0, [64, 512], target_memory=pl.Mem.Right
                        )
                        col_acc_inline3093__tile_l0_c_acc: pl.Tile[[64, 512], pl.INT32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 131072), pl.Mem.Acc] = pl.tile.matmul_acc(
                            col_acc_inline3093__tile_l0_c, col_acc_inline3093__tile_l0_a, col_acc_inline3093__tile_l0_b, qr_proj_col0_inline3105__idx_v0 == 0 and col_acc_inline3093__tile_l0_ko == 0
                        )
                        col_acc_inline3093__tile_l0_c_acc_1: pl.Tile[[64, 512], pl.INT32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 131072), pl.Mem.Acc] = pl.tile.matmul_acc(
                            col_acc_inline3093__tile_l0_c_acc,
                            col_acc_inline3093__tile_l0_a_1,
                            col_acc_inline3093__tile_l0_b_1,
                            qr_proj_col0_inline3105__idx_v0 == 0 and col_acc_inline3093__tile_l0_ko == -64,
                        )
                        col_acc_inline3093__tile_1: pl.Tile[[64, 512], pl.INT32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 131072), pl.Mem.Acc] = pl.yield_(col_acc_inline3093__tile_l0_c_acc_1)
                    for col_acc_inline3093__tile_l0_ko_1, (col_acc_inline3093__tile_l0_c_1,) in pl.range(0, 256, 128, init_values=(col_acc_inline3093__tile_1,)):
                        col_acc_inline3093__tile_l0_a_2: pl.Tile[
                            [64, 64], pl.INT8, pl.MemRef(mem_left_13, pl.const(4096, pl.INT64), 4096), pl.Mem.Left, pl.TileView(blayout=pl.TileLayout.row_major)
                        ] = pl.tile.extract(qr_i8_chunk_inline3099__tile_1, 0, col_acc_inline3093__tile_l0_ko_1, [64, 64], target_memory=pl.Mem.Left)
                        col_acc_inline3093__tile_l0_b_2: pl.Tile[[64, 512], pl.INT8, pl.MemRef(mem_right_9, pl.const(32768, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                            wq_chunk_inline3100__tile_1, col_acc_inline3093__tile_l0_ko_1, 0, [64, 512], target_memory=pl.Mem.Right
                        )
                        col_acc_inline3093__tile_l0_a_3: pl.Tile[
                            [64, 64], pl.INT8, pl.MemRef(mem_left_15, pl.const(8192, pl.INT64), 4096), pl.Mem.Left, pl.TileView(blayout=pl.TileLayout.row_major)
                        ] = pl.tile.extract(qr_i8_chunk_inline3099__tile_1, 0, col_acc_inline3093__tile_l0_ko_1 + 64, [64, 64], target_memory=pl.Mem.Left)
                        col_acc_inline3093__tile_l0_b_3: pl.Tile[[64, 512], pl.INT8, pl.MemRef(mem_right_11, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                            wq_chunk_inline3100__tile_1, col_acc_inline3093__tile_l0_ko_1 + 64, 0, [64, 512], target_memory=pl.Mem.Right
                        )
                        col_acc_inline3093__tile_l0_c_acc_2: pl.Tile[[64, 512], pl.INT32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 131072), pl.Mem.Acc] = pl.tile.matmul_acc(
                            col_acc_inline3093__tile_l0_c_1,
                            col_acc_inline3093__tile_l0_a_2,
                            col_acc_inline3093__tile_l0_b_2,
                            qr_proj_col0_inline3105__idx_v0 == -256 and col_acc_inline3093__tile_l0_ko_1 == 0,
                        )
                        col_acc_inline3093__tile_l0_c_acc_3: pl.Tile[[64, 512], pl.INT32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 131072), pl.Mem.Acc] = pl.tile.matmul_acc(
                            col_acc_inline3093__tile_l0_c_acc_2,
                            col_acc_inline3093__tile_l0_a_3,
                            col_acc_inline3093__tile_l0_b_3,
                            qr_proj_col0_inline3105__idx_v0 == -256 and col_acc_inline3093__tile_l0_ko_1 == -64,
                        )
                        col_acc_inline3093__tile_2: pl.Tile[[64, 512], pl.INT32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 131072), pl.Mem.Acc] = pl.yield_(col_acc_inline3093__tile_l0_c_acc_3)
                    col_acc_inline3093__rv_v2: pl.Tile[[64, 512], pl.INT32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 131072), pl.Mem.Acc] = pl.yield_(col_acc_inline3093__tile_2)
                q_proj_i32_inline6_inline1803__tile: pl.Tensor[[qproj_t_matmul_inline8_inline1802__ssa_v0, 32768], pl.INT32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                    col_acc_inline3093__rv_v2, [t0_inline3103__idx_v0, w_col0_inline3098__ssa_v0], q_proj_i32_inline6_inline1803__iter_v3
                )
                q_proj_i32_inline6_inline1803__rv_v4: pl.Tensor[[qproj_t_matmul_inline8_inline1802__ssa_v0, 32768], pl.INT32, pl.MemRef("mem_ddr_19", pl.const(0, pl.INT64), 0)] = pl.yield_(
                    q_proj_i32_inline6_inline1803__tile
                )
            tail_w_col0_inline3101__ssa_v0: pl.Scalar[pl.INDEX] = w_col0_inline3098__ssa_v0
            for tail_t0_inline3091__idx_v0, (q_proj_i32_inline6_inline1803__iter_v6,) in pl.range(
                qproj_full_rows_inline3094__ssa_v0, qproj_t_matmul_inline3092__ssa_v0, 16, init_values=(q_proj_i32_inline6_inline1803__rv_v4,)
            ):
                qproj_tail_rows_inline3090__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(tile_rows_inline9_inline1800__ssa_v0 - tail_t0_inline3091__idx_v0, 16)
                tail_acc_inline3102__tile: pl.Tile[[16, 512], pl.INT32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 65536), pl.Mem.Acc] = pl.tile.create(
                    [16, 512], dtype=pl.INT32, target_memory=pl.Mem.Acc
                )
                for tail_qr_col0_inline3089__idx_v0, (tail_acc_inline3102__iter_v1,) in pl.range(0, 1024, 512, init_values=(tail_acc_inline3102__tile,)):
                    qr_i8_tail_inline3088__tile: pl.Tile[
                        [16, 256], pl.INT8, pl.MemRef(mem_mat_4, pl.const(0, pl.INT64), 4096), pl.Mem.Mat, pl.TileView(valid_shape=[qproj_tail_rows_inline3090__ssa_v0, 256])
                    ] = pl.tile.load(
                        qr_i8_matmul_inline1790__rv_v2, [tail_t0_inline3091__idx_v0, tail_qr_col0_inline3089__idx_v0], [16, 256], [qproj_tail_rows_inline3090__ssa_v0, 256], target_memory=pl.Mem.Mat
                    )
                    wq_tail_inline3104__tile: pl.Tile[[256, 512], pl.INT8, pl.MemRef(mem_mat_5, pl.const(16384, pl.INT64), 131072), pl.Mem.Mat] = pl.tile.load(
                        wq_b__ssa_v0, [tail_qr_col0_inline3089__idx_v0, tail_w_col0_inline3101__ssa_v0], [256, 512], [256, 512], target_memory=pl.Mem.Mat
                    )
                    qr_i8_tail_inline3088__tile_1: pl.Tile[
                        [16, 256], pl.INT8, pl.MemRef(mem_mat_6, pl.const(147456, pl.INT64), 4096), pl.Mem.Mat, pl.TileView(valid_shape=[qproj_tail_rows_inline3090__ssa_v0, 256])
                    ] = pl.tile.load(
                        qr_i8_matmul_inline1790__rv_v2,
                        [tail_t0_inline3091__idx_v0, tail_qr_col0_inline3089__idx_v0 + 256],
                        [16, 256],
                        [qproj_tail_rows_inline3090__ssa_v0, 256],
                        target_memory=pl.Mem.Mat,
                    )
                    wq_tail_inline3104__tile_1: pl.Tile[[256, 512], pl.INT8, pl.MemRef(mem_mat_7, pl.const(163840, pl.INT64), 131072), pl.Mem.Mat] = pl.tile.load(
                        wq_b__ssa_v0, [tail_qr_col0_inline3089__idx_v0 + 256, tail_w_col0_inline3101__ssa_v0], [256, 512], [256, 512], target_memory=pl.Mem.Mat
                    )
                    for tail_acc_inline3102__tile_l0_ko, (tail_acc_inline3102__tile_l0_c,) in pl.range(0, 256, 128, init_values=(tail_acc_inline3102__iter_v1,)):
                        tail_acc_inline3102__tile_l0_a: pl.Tile[
                            [16, 64],
                            pl.INT8,
                            pl.MemRef(mem_left_8, pl.const(12288, pl.INT64), 1024),
                            pl.Mem.Left,
                            pl.TileView(valid_shape=[qproj_tail_rows_inline3090__ssa_v0, 64], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal),
                        ] = pl.tile.extract(qr_i8_tail_inline3088__tile, 0, tail_acc_inline3102__tile_l0_ko, [16, 64], target_memory=pl.Mem.Left)
                        tail_acc_inline3102__tile_l0_b: pl.Tile[[64, 512], pl.INT8, pl.MemRef(mem_right_9, pl.const(32768, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                            wq_tail_inline3104__tile, tail_acc_inline3102__tile_l0_ko, 0, [64, 512], target_memory=pl.Mem.Right
                        )
                        tail_acc_inline3102__tile_l0_a_1: pl.Tile[
                            [16, 64],
                            pl.INT8,
                            pl.MemRef(mem_left_10, pl.const(0, pl.INT64), 1024),
                            pl.Mem.Left,
                            pl.TileView(valid_shape=[qproj_tail_rows_inline3090__ssa_v0, 64], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal),
                        ] = pl.tile.extract(qr_i8_tail_inline3088__tile, 0, tail_acc_inline3102__tile_l0_ko + 64, [16, 64], target_memory=pl.Mem.Left)
                        tail_acc_inline3102__tile_l0_b_1: pl.Tile[[64, 512], pl.INT8, pl.MemRef(mem_right_11, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                            wq_tail_inline3104__tile, tail_acc_inline3102__tile_l0_ko + 64, 0, [64, 512], target_memory=pl.Mem.Right
                        )
                        tail_acc_inline3102__tile_l0_c_acc: pl.Tile[[16, 512], pl.INT32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 65536), pl.Mem.Acc] = pl.tile.matmul_acc(
                            tail_acc_inline3102__tile_l0_c,
                            tail_acc_inline3102__tile_l0_a,
                            tail_acc_inline3102__tile_l0_b,
                            tail_qr_col0_inline3089__idx_v0 == 0 and tail_acc_inline3102__tile_l0_ko == 0,
                        )
                        tail_acc_inline3102__tile_l0_c_acc_1: pl.Tile[[16, 512], pl.INT32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 65536), pl.Mem.Acc] = pl.tile.matmul_acc(
                            tail_acc_inline3102__tile_l0_c_acc,
                            tail_acc_inline3102__tile_l0_a_1,
                            tail_acc_inline3102__tile_l0_b_1,
                            tail_qr_col0_inline3089__idx_v0 == 0 and tail_acc_inline3102__tile_l0_ko == -64,
                        )
                        tail_acc_inline3102__tile_1: pl.Tile[[16, 512], pl.INT32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 65536), pl.Mem.Acc] = pl.yield_(tail_acc_inline3102__tile_l0_c_acc_1)
                    for tail_acc_inline3102__tile_l0_ko_1, (tail_acc_inline3102__tile_l0_c_1,) in pl.range(0, 256, 128, init_values=(tail_acc_inline3102__tile_1,)):
                        tail_acc_inline3102__tile_l0_a_2: pl.Tile[
                            [16, 64],
                            pl.INT8,
                            pl.MemRef(mem_left_13, pl.const(4096, pl.INT64), 1024),
                            pl.Mem.Left,
                            pl.TileView(valid_shape=[qproj_tail_rows_inline3090__ssa_v0, 64], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal),
                        ] = pl.tile.extract(qr_i8_tail_inline3088__tile_1, 0, tail_acc_inline3102__tile_l0_ko_1, [16, 64], target_memory=pl.Mem.Left)
                        tail_acc_inline3102__tile_l0_b_2: pl.Tile[[64, 512], pl.INT8, pl.MemRef(mem_right_9, pl.const(32768, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                            wq_tail_inline3104__tile_1, tail_acc_inline3102__tile_l0_ko_1, 0, [64, 512], target_memory=pl.Mem.Right
                        )
                        tail_acc_inline3102__tile_l0_a_3: pl.Tile[
                            [16, 64],
                            pl.INT8,
                            pl.MemRef(mem_left_15, pl.const(8192, pl.INT64), 1024),
                            pl.Mem.Left,
                            pl.TileView(valid_shape=[qproj_tail_rows_inline3090__ssa_v0, 64], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal),
                        ] = pl.tile.extract(qr_i8_tail_inline3088__tile_1, 0, tail_acc_inline3102__tile_l0_ko_1 + 64, [16, 64], target_memory=pl.Mem.Left)
                        tail_acc_inline3102__tile_l0_b_3: pl.Tile[[64, 512], pl.INT8, pl.MemRef(mem_right_11, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                            wq_tail_inline3104__tile_1, tail_acc_inline3102__tile_l0_ko_1 + 64, 0, [64, 512], target_memory=pl.Mem.Right
                        )
                        tail_acc_inline3102__tile_l0_c_acc_2: pl.Tile[[16, 512], pl.INT32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 65536), pl.Mem.Acc] = pl.tile.matmul_acc(
                            tail_acc_inline3102__tile_l0_c_1,
                            tail_acc_inline3102__tile_l0_a_2,
                            tail_acc_inline3102__tile_l0_b_2,
                            tail_qr_col0_inline3089__idx_v0 == -256 and tail_acc_inline3102__tile_l0_ko_1 == 0,
                        )
                        tail_acc_inline3102__tile_l0_c_acc_3: pl.Tile[[16, 512], pl.INT32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 65536), pl.Mem.Acc] = pl.tile.matmul_acc(
                            tail_acc_inline3102__tile_l0_c_acc_2,
                            tail_acc_inline3102__tile_l0_a_3,
                            tail_acc_inline3102__tile_l0_b_3,
                            tail_qr_col0_inline3089__idx_v0 == -256 and tail_acc_inline3102__tile_l0_ko_1 == -64,
                        )
                        tail_acc_inline3102__tile_2: pl.Tile[[16, 512], pl.INT32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 65536), pl.Mem.Acc] = pl.yield_(tail_acc_inline3102__tile_l0_c_acc_3)
                    tail_acc_inline3102__rv_v2: pl.Tile[[16, 512], pl.INT32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 65536), pl.Mem.Acc] = pl.yield_(tail_acc_inline3102__tile_2)
                q_proj_i32_inline6_inline1803__tile_1: pl.Tensor[[qproj_t_matmul_inline8_inline1802__ssa_v0, 32768], pl.INT32, pl.MemRef("mem_ddr_19", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                    tail_acc_inline3102__rv_v2, [tail_t0_inline3091__idx_v0, tail_w_col0_inline3101__ssa_v0], q_proj_i32_inline6_inline1803__iter_v6
                )
                q_proj_i32_inline6_inline1803__rv_v7: pl.Tensor[[qproj_t_matmul_inline8_inline1802__ssa_v0, 32768], pl.INT32, pl.MemRef("mem_ddr_36", pl.const(0, pl.INT64), 0)] = pl.yield_(
                    q_proj_i32_inline6_inline1803__tile_1
                )
            q_proj_i32_inline6_inline1803__rv_v2: pl.Tensor[[qproj_t_matmul_inline8_inline1802__ssa_v0, 32768], pl.INT32, pl.MemRef("mem_ddr_37", pl.const(0, pl.INT64), 0)] = pl.yield_(
                q_proj_i32_inline6_inline1803__rv_v7
            )
        return q_proj_i32_inline6_inline1803__ssa_v0

    @pl.function(type=pl.FunctionType.Spmd)
    def qproj_matmul_spmd(
        self,
        q_proj_i32_inline6_inline1803__ssa_v0: pl.Out[pl.Tensor[[qproj_t_matmul_inline8_inline1802__ssa_v0, 32768], pl.INT32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)]],
        qproj_full_rows_inline3094__ssa_v0: pl.Scalar[pl.INDEX],
        qr_i8_matmul_inline1790__rv_v2: pl.Tensor[[512, 1024], pl.INT8, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 524288)],
        wq_b__ssa_v0: pl.Tensor[[1024, 32768], pl.INT8, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 33554432)],
        qproj_t_matmul_inline3092__ssa_v0: pl.Scalar[pl.INDEX],
        tile_rows_inline9_inline1800__ssa_v0: pl.Scalar[pl.INDEX],
    ) -> pl.Tensor[[qproj_t_matmul_inline8_inline1802__ssa_v0, 32768], pl.INT32]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        q_proj_i32_inline6_inline1803__rv_v2: pl.Tensor[[qproj_t_matmul_inline8_inline1802__ssa_v0, 32768], pl.INT32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)] = self.qproj_matmul(
            q_proj_i32_inline6_inline1803__ssa_v0,
            qproj_full_rows_inline3094__ssa_v0,
            qr_i8_matmul_inline1790__rv_v2,
            wq_b__ssa_v0,
            qproj_t_matmul_inline3092__ssa_v0,
            tile_rows_inline9_inline1800__ssa_v0,
            attrs={"arg_directions": [pl.adir.output_existing, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.scalar, pl.adir.scalar]},
        )
        return q_proj_i32_inline6_inline1803__ssa_v0

    @pl.function(type=pl.FunctionType.AIC, level=pl.Level.AIC, role=pl.Role.SubWorker)
    def qr_hadamard_matmul(
        hadamard_idx__ssa_v0: pl.Tensor[[128, 128], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 32768)],
        qh_acc_gm_inline928_inline2298__ssa_v0: pl.Out[pl.Tensor[[bs_heads_inline935_inline2244__ssa_v0, 128], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)]],
        bs_heads_inline935_inline2244__ssa_v0: pl.Scalar[pl.INDEX],
        qr_bf16_inline2253__ssa_v0: pl.Tensor[[24576, 128], pl.BF16, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 6291456)],
    ) -> pl.Tensor[[bs_heads_inline935_inline2244__ssa_v0, 128], pl.FP32]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        mem_mat_3: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 32768)
        mem_mat_4: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 16384)
        mem_left_5: pl.Ptr = pl.tile.alloc(pl.Mem.Left, 16384)
        mem_right_6: pl.Ptr = pl.tile.alloc(pl.Mem.Right, 32768)
        mem_acc_7: pl.Ptr = pl.tile.alloc(pl.Mem.Acc, 32768)
        qh_worker_inline931_inline2293__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        qh_hadamard_inline932_inline2299__tile: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_mat_3, pl.const(0, pl.INT64), 32768), pl.Mem.Mat] = pl.tile.load(
            hadamard_idx__ssa_v0, [0, 0], [128, 128], [128, 128], target_memory=pl.Mem.Mat
        )
        for idx_inline929_inline2274__idx_v0, (qh_acc_gm_inline928_inline2298__iter_v1,) in pl.range(
            qh_worker_inline931_inline2293__ssa_v0, bs_heads_inline935_inline2244__ssa_v0 // 64, 24, init_values=(qh_acc_gm_inline928_inline2298__ssa_v0,)
        ):
            o0_inline939_inline2236__ssa_v0: pl.Scalar[pl.INDEX] = idx_inline929_inline2274__idx_v0 * 64
            t__tile: pl.Tile[[64, 128], pl.BF16, pl.MemRef(mem_mat_4, pl.const(32768, pl.INT64), 16384), pl.Mem.Mat] = pl.tile.load(
                qr_bf16_inline2253__ssa_v0, [o0_inline939_inline2236__ssa_v0, 0], [64, 128], [64, 128], target_memory=pl.Mem.Mat
            )
            t__tile_Left: pl.Tile[[64, 128], pl.BF16, pl.MemRef(mem_left_5, pl.const(0, pl.INT64), 16384), pl.Mem.Left] = pl.tile.move(t__tile, target_memory=pl.Mem.Left)
            qh_hadamard_inline932_inline2299__tile_Right: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_6, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.move(
                qh_hadamard_inline932_inline2299__tile, target_memory=pl.Mem.Right
            )
            qh_acc_inline940_inline2278__tile: pl.Tile[[64, 128], pl.FP32, pl.MemRef(mem_acc_7, pl.const(0, pl.INT64), 32768), pl.Mem.Acc] = pl.tile.matmul(
                t__tile_Left, qh_hadamard_inline932_inline2299__tile_Right
            )
            qh_acc_gm_inline928_inline2298__tile: pl.Tensor[[bs_heads_inline935_inline2244__ssa_v0, 128], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                qh_acc_inline940_inline2278__tile, [o0_inline939_inline2236__ssa_v0, 0], qh_acc_gm_inline928_inline2298__iter_v1
            )
            qh_acc_gm_inline928_inline2298__rv_v2: pl.Tensor[[bs_heads_inline935_inline2244__ssa_v0, 128], pl.FP32, pl.MemRef("mem_ddr_8", pl.const(0, pl.INT64), 0)] = pl.yield_(
                qh_acc_gm_inline928_inline2298__tile
            )
        return qh_acc_gm_inline928_inline2298__ssa_v0

    @pl.function(type=pl.FunctionType.Spmd)
    def qr_hadamard_matmul_spmd(
        self,
        hadamard_idx__ssa_v0: pl.Tensor[[128, 128], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 32768)],
        qh_acc_gm_inline928_inline2298__ssa_v0: pl.Out[pl.Tensor[[bs_heads_inline935_inline2244__ssa_v0, 128], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)]],
        bs_heads_inline935_inline2244__ssa_v0: pl.Scalar[pl.INDEX],
        qr_bf16_inline2253__ssa_v0: pl.Tensor[[24576, 128], pl.BF16, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 6291456)],
    ) -> pl.Tensor[[bs_heads_inline935_inline2244__ssa_v0, 128], pl.FP32]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        qh_acc_gm_inline928_inline2298__rv_v2: pl.Tensor[[bs_heads_inline935_inline2244__ssa_v0, 128], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)] = self.qr_hadamard_matmul(
            hadamard_idx__ssa_v0,
            qh_acc_gm_inline928_inline2298__ssa_v0,
            bs_heads_inline935_inline2244__ssa_v0,
            qr_bf16_inline2253__ssa_v0,
            attrs={"arg_directions": [pl.adir.input, pl.adir.output_existing, pl.adir.scalar, pl.adir.input]},
        )
        return qh_acc_gm_inline928_inline2298__ssa_v0

    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def qr_hadamard_quant(
        qr_hadamard_i8_inline221__ssa_v0: pl.Out[pl.Tensor[[24576, 128], pl.INT8, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 3145728)]],
        qr_hadamard_scale_dq_inline217__ssa_v0: pl.Out[pl.Tensor[[24576, 1], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 98304)]],
        bs_heads_inline935_inline2244__ssa_v0: pl.Scalar[pl.INDEX],
        qh_acc_gm_inline928_inline2298__rv_v2: pl.Tensor[[bs_heads_inline935_inline2244__ssa_v0, 128], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)],
    ) -> tuple[pl.Tensor[[24576, 128], pl.INT8], pl.Tensor[[24576, 1], pl.FP32]]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        mem_vec_3: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32768)
        mem_vec_9: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        mem_vec_12: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
        mem_vec_14: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32768)
        mem_vec_15: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        qh_quant_worker_inline934_inline2259__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        for idx_inline930_inline2297__idx_v0, (qr_hadamard_i8_inline221__iter_v1, qr_hadamard_scale_dq_inline217__iter_v1) in pl.range(
            qh_quant_worker_inline934_inline2259__ssa_v0, bs_heads_inline935_inline2244__ssa_v0 // 64, 48, init_values=(qr_hadamard_i8_inline221__ssa_v0, qr_hadamard_scale_dq_inline217__ssa_v0)
        ):
            o0_inline939_inline2236__ssa_v1: pl.Scalar[pl.INDEX] = idx_inline930_inline2297__idx_v0 * 64
            qh_full_f32_inline936_inline2265__tile: pl.Tile[[64, 128], pl.FP32, pl.MemRef(mem_vec_3, pl.const(49408, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.load(
                qh_acc_gm_inline928_inline2298__rv_v2, [o0_inline939_inline2236__ssa_v1, 0], [64, 128], [64, 128], target_memory=pl.Mem.Vec
            )
            t__tile: pl.Tile[[64, 128], pl.BF16, pl.MemRef(mem_vec_14, pl.const(16384, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.cast(
                qh_full_f32_inline936_inline2265__tile, target_type=pl.BF16, mode="rint"
            )
            qh_full_f32_v1_inline943_inline2233__tile: pl.Tile[[64, 128], pl.FP32, pl.MemRef(mem_vec_3, pl.const(49408, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.cast(
                t__tile, target_type=pl.FP32, mode="round"
            )
            t__tile_1: pl.Tile[[64, 128], pl.FP32, pl.MemRef(mem_vec_3, pl.const(49408, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.muls(qh_full_f32_v1_inline943_inline2233__tile, 0.088388347648318447)
            t__tile_2: pl.Tile[[64, 128], pl.BF16, pl.MemRef(mem_vec_14, pl.const(16384, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.cast(t__tile_1, target_type=pl.BF16, mode="rint")
            qh_full_f32_v2_inline955_inline2232__tile: pl.Tile[[64, 128], pl.FP32, pl.MemRef(mem_vec_3, pl.const(49408, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.cast(
                t__tile_2, target_type=pl.FP32, mode="round"
            )
            qh_amax_inline944_inline2231__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_9, pl.const(82176, pl.INT64), 256), pl.Mem.Vec] = pl.tile.full([1, 64], dtype=pl.FP32, value=0.0001)
            for h0_inline945_inline2230__idx_v0, (qh_amax_inline944_inline2231__iter_v1,) in pl.range(0, 128, 64, init_values=(qh_amax_inline944_inline2231__tile,)):
                qh_a_f32_inline946_inline2229__tile_textract: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_14, pl.const(16384, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.extract(
                    qh_full_f32_v2_inline955_inline2232__tile, 0, h0_inline945_inline2230__idx_v0, [64, 64], target_memory=pl.Mem.Vec
                )
                qh_a_neg_inline948_inline2273__tile: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_14, pl.const(16384, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.neg(
                    qh_a_f32_inline946_inline2229__tile_textract
                )
                qh_a_f32_inline946_inline2229__tile_textract_1: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(0, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.extract(
                    qh_full_f32_v2_inline955_inline2232__tile, 0, h0_inline945_inline2230__idx_v0, [64, 64], target_memory=pl.Mem.Vec
                )
                qh_a_abs_inline937_inline2270__tile: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(0, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.maximum(
                    qh_a_f32_inline946_inline2229__tile_textract_1, qh_a_neg_inline948_inline2273__tile
                )
                tmp_tile: pl.Tile[[64, 128], pl.FP32, pl.MemRef(mem_vec_14, pl.const(16384, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.create([64, 128], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                qh_a_max_col_inline950_inline2228__tile: pl.Tile[[64, 1], pl.FP32, pl.MemRef(mem_vec_15, pl.const(49152, pl.INT64), 256), pl.Mem.Vec] = pl.tile.row_max(
                    qh_a_abs_inline937_inline2270__tile, tmp_tile
                )
                qh_a_max_inline926_inline2227__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_15, pl.const(49152, pl.INT64), 256), pl.Mem.Vec] = pl.tile.reshape(
                    qh_a_max_col_inline950_inline2228__tile, [1, 64]
                )
                qh_amax_inline944_inline2231__tile_1: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_9, pl.const(82176, pl.INT64), 256), pl.Mem.Vec] = pl.tile.maximum(
                    qh_amax_inline944_inline2231__iter_v1, qh_a_max_inline926_inline2227__tile
                )
                qh_amax_inline944_inline2231__rv_v2: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_9, pl.const(82176, pl.INT64), 256), pl.Mem.Vec] = pl.yield_(qh_amax_inline944_inline2231__tile_1)
            qh_scale_numerator_inline952_inline2226__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_14, pl.const(16384, pl.INT64), 256), pl.Mem.Vec] = pl.tile.full(
                [1, 64], dtype=pl.FP32, value=127.0
            )
            qh_scale_quant_row_inline954_inline2225__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_14, pl.const(16384, pl.INT64), 256), pl.Mem.Vec] = pl.tile.div(
                qh_scale_numerator_inline952_inline2226__tile, qh_amax_inline944_inline2231__rv_v2
            )
            qh_scale_recip_inline947_inline2290__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(0, pl.INT64), 256), pl.Mem.Vec] = pl.tile.recip(
                qh_scale_quant_row_inline954_inline2225__tile
            )
            qh_scale_dq_inline956_inline2224__tile: pl.Tile[[64, 1], pl.FP32, pl.MemRef(mem_vec_12, pl.const(0, pl.INT64), 256), pl.Mem.Vec] = pl.tile.reshape(
                qh_scale_recip_inline947_inline2290__tile, [64, 1]
            )
            t__rm_a0_tmp_v0: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(0, pl.INT64), 256), pl.Mem.Vec] = pl.tile.reshape(qh_scale_dq_inline956_inline2224__tile, [1, 64])
            t__row_major_tmp_v1: pl.Tile[[1, 64], pl.FP16, pl.MemRef(mem_vec_9, pl.const(82176, pl.INT64), 128), pl.Mem.Vec] = pl.tile.cast(t__rm_a0_tmp_v0, target_type=pl.FP16, mode="rint")
            t__tile_3: pl.Tile[[64, 1], pl.FP16, pl.MemRef(mem_vec_9, pl.const(82176, pl.INT64), 128), pl.Mem.Vec] = pl.tile.reshape(t__row_major_tmp_v1, [64, 1])
            t__rm_a0_tmp_v2: pl.Tile[[1, 64], pl.FP16, pl.MemRef(mem_vec_9, pl.const(82176, pl.INT64), 128), pl.Mem.Vec] = pl.tile.reshape(t__tile_3, [1, 64])
            t__row_major_tmp_v3: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(0, pl.INT64), 256), pl.Mem.Vec] = pl.tile.cast(t__rm_a0_tmp_v2, target_type=pl.FP32, mode="round")
            t__tile_4: pl.Tile[[64, 1], pl.FP32, pl.MemRef(mem_vec_12, pl.const(0, pl.INT64), 256), pl.Mem.Vec] = pl.tile.reshape(t__row_major_tmp_v3, [64, 1])
            qr_hadamard_scale_dq_inline217__tile: pl.Tensor[[24576, 1], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 98304)] = pl.tile.store(
                t__tile_4, [o0_inline939_inline2236__ssa_v1, 0], qr_hadamard_scale_dq_inline217__iter_v1
            )
            qh_scale_quant_inline941_inline2223__tile: pl.Tile[[64, 1], pl.FP32, pl.MemRef(mem_vec_14, pl.const(16384, pl.INT64), 256), pl.Mem.Vec] = pl.tile.reshape(
                qh_scale_quant_row_inline954_inline2225__tile, [64, 1]
            )
            qh_q_scaled_inline953_inline2222__tile: pl.Tile[[64, 128], pl.FP32, pl.MemRef(mem_vec_3, pl.const(49408, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.row_expand_mul(
                qh_full_f32_v2_inline955_inline2232__tile, qh_scale_quant_inline941_inline2223__tile
            )
            qh_q_i32_inline942_inline2221__tile: pl.Tile[[64, 128], pl.INT32, pl.MemRef(mem_vec_3, pl.const(49408, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.cast(
                qh_q_scaled_inline953_inline2222__tile, target_type=pl.INT32, mode="rint"
            )
            qh_q_half_inline951_inline2220__tile: pl.Tile[[64, 128], pl.FP16, pl.MemRef(mem_vec_3, pl.const(49408, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.cast(
                qh_q_i32_inline942_inline2221__tile, target_type=pl.FP16, mode="round"
            )
            qh_i8_inline949_inline2295__tile: pl.Tile[[64, 128], pl.INT8, pl.MemRef(mem_vec_3, pl.const(49408, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.cast(
                qh_q_half_inline951_inline2220__tile, target_type=pl.INT8, mode="trunc"
            )
            qr_hadamard_i8_inline221__tile: pl.Tensor[[24576, 128], pl.INT8, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 3145728)] = pl.tile.store(
                qh_i8_inline949_inline2295__tile, [o0_inline939_inline2236__ssa_v1, 0], qr_hadamard_i8_inline221__iter_v1
            )
            qr_hadamard_i8_inline221__rv_v2, qr_hadamard_scale_dq_inline217__rv_v2 = pl.yield_(qr_hadamard_i8_inline221__tile, qr_hadamard_scale_dq_inline217__tile)
        return qr_hadamard_i8_inline221__ssa_v0, qr_hadamard_scale_dq_inline217__ssa_v0

    @pl.function(type=pl.FunctionType.Spmd)
    def qr_hadamard_quant_spmd(
        self,
        qr_hadamard_i8_inline221__ssa_v0: pl.Out[pl.Tensor[[24576, 128], pl.INT8, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 3145728)]],
        qr_hadamard_scale_dq_inline217__ssa_v0: pl.Out[pl.Tensor[[24576, 1], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 98304)]],
        bs_heads_inline935_inline2244__ssa_v0: pl.Scalar[pl.INDEX],
        qh_acc_gm_inline928_inline2298__rv_v2: pl.Tensor[[bs_heads_inline935_inline2244__ssa_v0, 128], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)],
    ) -> tuple[pl.Tensor[[24576, 128], pl.INT8], pl.Tensor[[24576, 1], pl.FP32]]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        ret__tmp_v0: pl.Tuple[pl.Tensor[[24576, 128], pl.INT8], pl.Tensor[[24576, 1], pl.FP32]] = self.qr_hadamard_quant(
            qr_hadamard_i8_inline221__ssa_v0,
            qr_hadamard_scale_dq_inline217__ssa_v0,
            bs_heads_inline935_inline2244__ssa_v0,
            qh_acc_gm_inline928_inline2298__rv_v2,
            attrs={"arg_directions": [pl.adir.output_existing, pl.adir.output_existing, pl.adir.scalar, pl.adir.input]},
        )
        qr_hadamard_i8_inline221__rv_v2: pl.Tensor[[24576, 128], pl.INT8, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 3145728)] = ret__tmp_v0[0]
        qr_hadamard_scale_dq_inline217__rv_v2: pl.Tensor[[24576, 1], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 98304)] = ret__tmp_v0[1]
        return qr_hadamard_i8_inline221__ssa_v0, qr_hadamard_scale_dq_inline217__ssa_v0

    @pl.function(type=pl.FunctionType.AIC, level=pl.Level.AIC, role=pl.Role.SubWorker)
    def qr_proj_matmul(
        qr_fp32_inline0_inline1798__rv_v2: pl.InOut[pl.Tensor[[qr_t_matmul_inline1_inline1795__ssa_v0, 1024], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)]],
        qr_full_rows_inline3006__ssa_v0: pl.Scalar[pl.INDEX],
        tile_base_inline3_inline1791__idx_v0: pl.Scalar[pl.INDEX],
        tile_rows_inline2_inline1793__ssa_v0: pl.Scalar[pl.INDEX],
        x_view_inline3007__ssa_v0: pl.Tensor[[qa_tokens_inline3009__ssa_v0, 4096], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)],
        wq_a__ssa_v0: pl.Tensor[[4096, 1024], pl.BF16, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 8388608)],
        qr_t_matmul_inline2997__ssa_v0: pl.Scalar[pl.INDEX],
    ) -> pl.Tensor[[qr_t_matmul_inline1_inline1795__ssa_v0, 1024], pl.FP32]:
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
        qbg_idx_inline2995__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        q_a_col0_inline3005__ssa_v0: pl.Scalar[pl.INDEX] = qbg_idx_inline2995__ssa_v0 * 32
        qr_native_group_inline2989__ssa_v0: pl.Scalar[pl.INDEX] = q_a_col0_inline3005__ssa_v0 // 96
        for dense_t0_inline3000__idx_v0, (qr_fp32_inline0_inline1798__iter_v6,) in pl.range(0, qr_full_rows_inline3006__ssa_v0, 64, init_values=(qr_fp32_inline0_inline1798__rv_v2,)):
            dense_x0_inline2998__ssa_v0: pl.Scalar[pl.INDEX] = tile_base_inline3_inline1791__idx_v0 + dense_t0_inline3000__idx_v0
            dense_acc_inline3010__tile: pl.Tile[[64, 32], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 8192), pl.Mem.Acc] = pl.tile.create([64, 32], dtype=pl.FP32, target_memory=pl.Mem.Acc)
            for dense_k_inline3011__idx_v0, (dense_acc_inline3010__iter_v1,) in pl.range(0, 16, 2, init_values=(dense_acc_inline3010__tile,)):
                dense_k_order_inline3012__ssa_v0: pl.Scalar[pl.INDEX] = dense_k_inline3011__idx_v0
                dense_k_order_inline3012__ssa_v0_1: pl.Scalar[pl.INDEX] = dense_k_inline3011__idx_v0 + 1
                if (
                    96 <= tile_rows_inline2_inline1793__ssa_v0
                    and tile_rows_inline2_inline1793__ssa_v0 <= 240
                    and tile_rows_inline2_inline1793__ssa_v0 % 48 == 0
                    and qr_native_group_inline2989__ssa_v0 < 9
                ):
                    dense_k_direction_inline3004__ssa_v0: pl.Scalar[pl.INDEX] = dense_k_inline3011__idx_v0
                    if qr_native_group_inline2989__ssa_v0 % 2 == 1:
                        dense_k_direction_inline3004__ssa_v1: pl.Scalar[pl.INDEX] = 15 - dense_k_inline3011__idx_v0
                        dense_k_direction_inline3004__phi_v2: pl.Scalar[pl.INDEX] = pl.yield_(dense_k_direction_inline3004__ssa_v1)
                    else:
                        dense_k_direction_inline3004__phi_v2: pl.Scalar[pl.INDEX] = pl.yield_(dense_k_direction_inline3004__ssa_v0)
                    dense_k_order_inline3012__ssa_v1: pl.Scalar[pl.INDEX] = (qr_native_group_inline2989__ssa_v0 // 2 * 3 + dense_k_direction_inline3004__phi_v2) % 16
                    dense_k_order_inline3012__phi_v2: pl.Scalar[pl.INDEX] = pl.yield_(dense_k_order_inline3012__ssa_v1)
                else:
                    dense_k_order_inline3012__phi_v2: pl.Scalar[pl.INDEX] = pl.yield_(dense_k_order_inline3012__ssa_v0)
                dense_d0_inline2992__ssa_v0: pl.Scalar[pl.INDEX] = dense_k_order_inline3012__phi_v2 * 256
                dense_x_inline2990__tile: pl.Tile[[64, 256], pl.BF16, pl.MemRef(mem_mat_4, pl.const(0, pl.INT64), 32768), pl.Mem.Mat] = pl.tile.load(
                    x_view_inline3007__ssa_v0, [dense_x0_inline2998__ssa_v0, dense_d0_inline2992__ssa_v0], [64, 256], [64, 256], target_memory=pl.Mem.Mat
                )
                dense_w_inline2999__tile: pl.Tile[[256, 32], pl.BF16, pl.MemRef(mem_mat_5, pl.const(32768, pl.INT64), 16384), pl.Mem.Mat] = pl.tile.load(
                    wq_a__ssa_v0, [dense_d0_inline2992__ssa_v0, q_a_col0_inline3005__ssa_v0], [256, 32], [256, 32], target_memory=pl.Mem.Mat
                )
                if (
                    96 <= tile_rows_inline2_inline1793__ssa_v0
                    and tile_rows_inline2_inline1793__ssa_v0 <= 240
                    and tile_rows_inline2_inline1793__ssa_v0 % 48 == 0
                    and qr_native_group_inline2989__ssa_v0 < 9
                ):
                    dense_k_direction_inline3004__ssa_v0_1: pl.Scalar[pl.INDEX] = dense_k_inline3011__idx_v0 + 1
                    if qr_native_group_inline2989__ssa_v0 % 2 == 1:
                        dense_k_direction_inline3004__ssa_v1_1: pl.Scalar[pl.INDEX] = 14 - dense_k_inline3011__idx_v0
                        dense_k_direction_inline3004__phi_v2_1: pl.Scalar[pl.INDEX] = pl.yield_(dense_k_direction_inline3004__ssa_v1_1)
                    else:
                        dense_k_direction_inline3004__phi_v2_1: pl.Scalar[pl.INDEX] = pl.yield_(dense_k_direction_inline3004__ssa_v0_1)
                    dense_k_order_inline3012__ssa_v1_1: pl.Scalar[pl.INDEX] = (qr_native_group_inline2989__ssa_v0 // 2 * 3 + dense_k_direction_inline3004__phi_v2_1) % 16
                    dense_k_order_inline3012__phi_v2_1: pl.Scalar[pl.INDEX] = pl.yield_(dense_k_order_inline3012__ssa_v1_1)
                else:
                    dense_k_order_inline3012__phi_v2_1: pl.Scalar[pl.INDEX] = pl.yield_(dense_k_order_inline3012__ssa_v0_1)
                dense_d0_inline2992__ssa_v0_1: pl.Scalar[pl.INDEX] = dense_k_order_inline3012__phi_v2_1 * 256
                dense_x_inline2990__tile_1: pl.Tile[[64, 256], pl.BF16, pl.MemRef(mem_mat_6, pl.const(49152, pl.INT64), 32768), pl.Mem.Mat] = pl.tile.load(
                    x_view_inline3007__ssa_v0, [dense_x0_inline2998__ssa_v0, dense_d0_inline2992__ssa_v0_1], [64, 256], [64, 256], target_memory=pl.Mem.Mat
                )
                dense_w_inline2999__tile_1: pl.Tile[[256, 32], pl.BF16, pl.MemRef(mem_mat_7, pl.const(81920, pl.INT64), 16384), pl.Mem.Mat] = pl.tile.load(
                    wq_a__ssa_v0, [dense_d0_inline2992__ssa_v0_1, q_a_col0_inline3005__ssa_v0], [256, 32], [256, 32], target_memory=pl.Mem.Mat
                )
                dense_x_inline2990__tile_Left: pl.Tile[[64, 256], pl.BF16, pl.MemRef(mem_left_8, pl.const(32768, pl.INT64), 32768), pl.Mem.Left] = pl.tile.move(
                    dense_x_inline2990__tile, target_memory=pl.Mem.Left
                )
                dense_w_inline2999__tile_Right: pl.Tile[[256, 32], pl.BF16, pl.MemRef(mem_right_9, pl.const(16384, pl.INT64), 16384), pl.Mem.Right] = pl.tile.move(
                    dense_w_inline2999__tile, target_memory=pl.Mem.Right
                )
                dense_acc_inline3010__tile_1: pl.Tile[[64, 32], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 8192), pl.Mem.Acc] = pl.tile.matmul_acc(
                    dense_acc_inline3010__iter_v1, dense_x_inline2990__tile_Left, dense_w_inline2999__tile_Right, dense_k_inline3011__idx_v0 == 0
                )
                dense_x_inline2990__tile_Left_1: pl.Tile[[64, 256], pl.BF16, pl.MemRef(mem_left_10, pl.const(0, pl.INT64), 32768), pl.Mem.Left] = pl.tile.move(
                    dense_x_inline2990__tile_1, target_memory=pl.Mem.Left
                )
                dense_w_inline2999__tile_Right_1: pl.Tile[[256, 32], pl.BF16, pl.MemRef(mem_right_11, pl.const(0, pl.INT64), 16384), pl.Mem.Right] = pl.tile.move(
                    dense_w_inline2999__tile_1, target_memory=pl.Mem.Right
                )
                dense_acc_inline3010__tile_2: pl.Tile[[64, 32], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 8192), pl.Mem.Acc] = pl.tile.matmul_acc(
                    dense_acc_inline3010__tile_1, dense_x_inline2990__tile_Left_1, dense_w_inline2999__tile_Right_1, dense_k_inline3011__idx_v0 == -1
                )
                dense_acc_inline3010__rv_v2: pl.Tile[[64, 32], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 8192), pl.Mem.Acc] = pl.yield_(dense_acc_inline3010__tile_2)
            qr_fp32_inline0_inline1798__tile: pl.Tensor[[qr_t_matmul_inline1_inline1795__ssa_v0, 1024], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                dense_acc_inline3010__rv_v2, [dense_t0_inline3000__idx_v0, q_a_col0_inline3005__ssa_v0], qr_fp32_inline0_inline1798__iter_v6, atomic=pl.AtomicType.Add
            )
            qr_fp32_inline0_inline1798__rv_v7: pl.Tensor[[qr_t_matmul_inline1_inline1795__ssa_v0, 1024], pl.FP32, pl.MemRef("mem_ddr_13", pl.const(0, pl.INT64), 0)] = pl.yield_(
                qr_fp32_inline0_inline1798__tile
            )
        for t0_inline3002__idx_v0, (qr_fp32_inline0_inline1798__iter_v9,) in pl.range(
            qr_full_rows_inline3006__ssa_v0, qr_t_matmul_inline2997__ssa_v0, 16, init_values=(qr_fp32_inline0_inline1798__rv_v7,)
        ):
            q_acc_inline3013__tile: pl.Tile[[16, 32], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 2048), pl.Mem.Acc] = pl.tile.create([16, 32], dtype=pl.FP32, target_memory=pl.Mem.Acc)
            for db_inline3008__idx_v0, (q_acc_inline3013__iter_v1,) in pl.range(0, 16, 2, init_values=(q_acc_inline3013__tile,)):
                qr_k_order_inline3016__ssa_v0: pl.Scalar[pl.INDEX] = db_inline3008__idx_v0
                qr_rows_inline3015__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(tile_rows_inline2_inline1793__ssa_v0 - t0_inline3002__idx_v0, 16)
                x_t0_inline3017__ssa_v0: pl.Scalar[pl.INDEX] = tile_base_inline3_inline1791__idx_v0 + t0_inline3002__idx_v0
                qr_k_order_inline3016__ssa_v0_1: pl.Scalar[pl.INDEX] = db_inline3008__idx_v0 + 1
                qr_rows_inline3015__ssa_v0_1: pl.Scalar[pl.INDEX] = pl.min(tile_rows_inline2_inline1793__ssa_v0 - t0_inline3002__idx_v0, 16)
                x_t0_inline3017__ssa_v0_1: pl.Scalar[pl.INDEX] = tile_base_inline3_inline1791__idx_v0 + t0_inline3002__idx_v0
                if (
                    96 <= tile_rows_inline2_inline1793__ssa_v0
                    and tile_rows_inline2_inline1793__ssa_v0 <= 240
                    and tile_rows_inline2_inline1793__ssa_v0 % 48 == 0
                    and qr_native_group_inline2989__ssa_v0 < 9
                ):
                    qr_k_direction_inline3014__ssa_v0: pl.Scalar[pl.INDEX] = db_inline3008__idx_v0
                    if qr_native_group_inline2989__ssa_v0 % 2 == 1:
                        qr_k_direction_inline3014__ssa_v1: pl.Scalar[pl.INDEX] = 15 - db_inline3008__idx_v0
                        qr_k_direction_inline3014__phi_v2: pl.Scalar[pl.INDEX] = pl.yield_(qr_k_direction_inline3014__ssa_v1)
                    else:
                        qr_k_direction_inline3014__phi_v2: pl.Scalar[pl.INDEX] = pl.yield_(qr_k_direction_inline3014__ssa_v0)
                    qr_k_order_inline3016__ssa_v1: pl.Scalar[pl.INDEX] = (qr_native_group_inline2989__ssa_v0 // 2 * 3 + qr_k_direction_inline3014__phi_v2) % 16
                    qr_k_order_inline3016__phi_v2: pl.Scalar[pl.INDEX] = pl.yield_(qr_k_order_inline3016__ssa_v1)
                else:
                    qr_k_order_inline3016__phi_v2: pl.Scalar[pl.INDEX] = pl.yield_(qr_k_order_inline3016__ssa_v0)
                qr_d0_inline2991__ssa_v0: pl.Scalar[pl.INDEX] = qr_k_order_inline3016__phi_v2 * 256
                q_x_chunk_bf16_inline2996__tile: pl.Tile[
                    [16, 256], pl.BF16, pl.MemRef(mem_mat_5, pl.const(32768, pl.INT64), 8192), pl.Mem.Mat, pl.TileView(valid_shape=[qr_rows_inline3015__ssa_v0, 256])
                ] = pl.tile.load(x_view_inline3007__ssa_v0, [x_t0_inline3017__ssa_v0, qr_d0_inline2991__ssa_v0], [16, 256], [qr_rows_inline3015__ssa_v0, 256], target_memory=pl.Mem.Mat)
                w_chunk_inline2988__tile: pl.Tile[[256, 32], pl.BF16, pl.MemRef(mem_mat_4, pl.const(0, pl.INT64), 16384), pl.Mem.Mat] = pl.tile.load(
                    wq_a__ssa_v0, [qr_d0_inline2991__ssa_v0, q_a_col0_inline3005__ssa_v0], [256, 32], [256, 32], target_memory=pl.Mem.Mat
                )
                if (
                    96 <= tile_rows_inline2_inline1793__ssa_v0
                    and tile_rows_inline2_inline1793__ssa_v0 <= 240
                    and tile_rows_inline2_inline1793__ssa_v0 % 48 == 0
                    and qr_native_group_inline2989__ssa_v0 < 9
                ):
                    qr_k_direction_inline3014__ssa_v0_1: pl.Scalar[pl.INDEX] = db_inline3008__idx_v0 + 1
                    if qr_native_group_inline2989__ssa_v0 % 2 == 1:
                        qr_k_direction_inline3014__ssa_v1_1: pl.Scalar[pl.INDEX] = 14 - db_inline3008__idx_v0
                        qr_k_direction_inline3014__phi_v2_1: pl.Scalar[pl.INDEX] = pl.yield_(qr_k_direction_inline3014__ssa_v1_1)
                    else:
                        qr_k_direction_inline3014__phi_v2_1: pl.Scalar[pl.INDEX] = pl.yield_(qr_k_direction_inline3014__ssa_v0_1)
                    qr_k_order_inline3016__ssa_v1_1: pl.Scalar[pl.INDEX] = (qr_native_group_inline2989__ssa_v0 // 2 * 3 + qr_k_direction_inline3014__phi_v2_1) % 16
                    qr_k_order_inline3016__phi_v2_1: pl.Scalar[pl.INDEX] = pl.yield_(qr_k_order_inline3016__ssa_v1_1)
                else:
                    qr_k_order_inline3016__phi_v2_1: pl.Scalar[pl.INDEX] = pl.yield_(qr_k_order_inline3016__ssa_v0_1)
                qr_d0_inline2991__ssa_v0_1: pl.Scalar[pl.INDEX] = qr_k_order_inline3016__phi_v2_1 * 256
                q_x_chunk_bf16_inline2996__tile_1: pl.Tile[
                    [16, 256], pl.BF16, pl.MemRef(mem_mat_7, pl.const(81920, pl.INT64), 8192), pl.Mem.Mat, pl.TileView(valid_shape=[qr_rows_inline3015__ssa_v0_1, 256])
                ] = pl.tile.load(x_view_inline3007__ssa_v0, [x_t0_inline3017__ssa_v0_1, qr_d0_inline2991__ssa_v0_1], [16, 256], [qr_rows_inline3015__ssa_v0_1, 256], target_memory=pl.Mem.Mat)
                w_chunk_inline2988__tile_1: pl.Tile[[256, 32], pl.BF16, pl.MemRef(mem_mat_6, pl.const(49152, pl.INT64), 16384), pl.Mem.Mat] = pl.tile.load(
                    wq_a__ssa_v0, [qr_d0_inline2991__ssa_v0_1, q_a_col0_inline3005__ssa_v0], [256, 32], [256, 32], target_memory=pl.Mem.Mat
                )
                q_x_chunk_bf16_inline2996__tile_Left: pl.Tile[
                    [16, 256], pl.BF16, pl.MemRef(mem_left_8, pl.const(32768, pl.INT64), 8192), pl.Mem.Left, pl.TileView(valid_shape=[qr_rows_inline3015__ssa_v0, 256])
                ] = pl.tile.move(q_x_chunk_bf16_inline2996__tile, target_memory=pl.Mem.Left)
                w_chunk_inline2988__tile_Right: pl.Tile[[256, 32], pl.BF16, pl.MemRef(mem_right_9, pl.const(16384, pl.INT64), 16384), pl.Mem.Right] = pl.tile.move(
                    w_chunk_inline2988__tile, target_memory=pl.Mem.Right
                )
                q_acc_inline3013__tile_1: pl.Tile[[16, 32], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 2048), pl.Mem.Acc] = pl.tile.matmul_acc(
                    q_acc_inline3013__iter_v1, q_x_chunk_bf16_inline2996__tile_Left, w_chunk_inline2988__tile_Right, db_inline3008__idx_v0 == 0
                )
                q_x_chunk_bf16_inline2996__tile_Left_1: pl.Tile[
                    [16, 256], pl.BF16, pl.MemRef(mem_left_10, pl.const(0, pl.INT64), 8192), pl.Mem.Left, pl.TileView(valid_shape=[qr_rows_inline3015__ssa_v0_1, 256])
                ] = pl.tile.move(q_x_chunk_bf16_inline2996__tile_1, target_memory=pl.Mem.Left)
                w_chunk_inline2988__tile_Right_1: pl.Tile[[256, 32], pl.BF16, pl.MemRef(mem_right_11, pl.const(0, pl.INT64), 16384), pl.Mem.Right] = pl.tile.move(
                    w_chunk_inline2988__tile_1, target_memory=pl.Mem.Right
                )
                q_acc_inline3013__tile_2: pl.Tile[[16, 32], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 2048), pl.Mem.Acc] = pl.tile.matmul_acc(
                    q_acc_inline3013__tile_1, q_x_chunk_bf16_inline2996__tile_Left_1, w_chunk_inline2988__tile_Right_1, db_inline3008__idx_v0 == -1
                )
                q_acc_inline3013__rv_v2: pl.Tile[[16, 32], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 2048), pl.Mem.Acc] = pl.yield_(q_acc_inline3013__tile_2)
            qr_fp32_inline0_inline1798__tile_1: pl.Tensor[[qr_t_matmul_inline1_inline1795__ssa_v0, 1024], pl.FP32, pl.MemRef("mem_ddr_13", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                q_acc_inline3013__rv_v2, [t0_inline3002__idx_v0, q_a_col0_inline3005__ssa_v0], qr_fp32_inline0_inline1798__iter_v9, atomic=pl.AtomicType.Add
            )
            qr_fp32_inline0_inline1798__rv_v10: pl.Tensor[[qr_t_matmul_inline1_inline1795__ssa_v0, 1024], pl.FP32, pl.MemRef("mem_ddr_24", pl.const(0, pl.INT64), 0)] = pl.yield_(
                qr_fp32_inline0_inline1798__tile_1
            )
        return qr_fp32_inline0_inline1798__rv_v2

    @pl.function(type=pl.FunctionType.Spmd)
    def qr_proj_matmul_spmd(
        self,
        qr_fp32_inline0_inline1798__rv_v2: pl.InOut[pl.Tensor[[qr_t_matmul_inline1_inline1795__ssa_v0, 1024], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)]],
        qr_full_rows_inline3006__ssa_v0: pl.Scalar[pl.INDEX],
        tile_base_inline3_inline1791__idx_v0: pl.Scalar[pl.INDEX],
        tile_rows_inline2_inline1793__ssa_v0: pl.Scalar[pl.INDEX],
        x_view_inline3007__ssa_v0: pl.Tensor[[qa_tokens_inline3009__ssa_v0, 4096], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)],
        wq_a__ssa_v0: pl.Tensor[[4096, 1024], pl.BF16, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 8388608)],
        qr_t_matmul_inline2997__ssa_v0: pl.Scalar[pl.INDEX],
    ) -> pl.Tensor[[qr_t_matmul_inline1_inline1795__ssa_v0, 1024], pl.FP32]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        qr_fp32_inline0_inline1798__rv_v10: pl.Tensor[[qr_t_matmul_inline1_inline1795__ssa_v0, 1024], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)] = self.qr_proj_matmul(
            qr_fp32_inline0_inline1798__rv_v2,
            qr_full_rows_inline3006__ssa_v0,
            tile_base_inline3_inline1791__idx_v0,
            tile_rows_inline2_inline1793__ssa_v0,
            x_view_inline3007__ssa_v0,
            wq_a__ssa_v0,
            qr_t_matmul_inline2997__ssa_v0,
            attrs={"arg_directions": [pl.adir.inout, pl.adir.scalar, pl.adir.scalar, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.scalar]},
        )
        return qr_fp32_inline0_inline1798__rv_v2

    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def qr_proj_seed(
        qr_fp32_inline0_inline1798__ssa_v0: pl.Out[pl.Tensor[[qr_t_matmul_inline1_inline1795__ssa_v0, 1024], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)]],
        qr_t_matmul_inline2997__ssa_v0: pl.Scalar[pl.INDEX],
    ) -> pl.Tensor[[qr_t_matmul_inline1_inline1795__ssa_v0, 1024], pl.FP32]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        mem_vec_1: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 2048)
        for ts0_inline3001__idx_v0, (qr_fp32_inline0_inline1798__iter_v1,) in pl.range(0, qr_t_matmul_inline2997__ssa_v0, 16, init_values=(qr_fp32_inline0_inline1798__ssa_v0,)):
            for nseed0_inline2994__idx_v0, (qr_fp32_inline0_inline1798__iter_v3,) in pl.range(0, 1024, 32, init_values=(qr_fp32_inline0_inline1798__iter_v1,)):
                qr_seed_inline2993__tile: pl.Tile[[16, 32], pl.FP32, pl.MemRef(mem_vec_1, pl.const(0, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.full([16, 32], dtype=pl.FP32, value=0.0)
                qr_fp32_inline0_inline1798__tile: pl.Tensor[[qr_t_matmul_inline1_inline1795__ssa_v0, 1024], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                    qr_seed_inline2993__tile, [ts0_inline3001__idx_v0, nseed0_inline2994__idx_v0], qr_fp32_inline0_inline1798__iter_v3
                )
                qr_fp32_inline0_inline1798__rv_v4: pl.Tensor[[qr_t_matmul_inline1_inline1795__ssa_v0, 1024], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)] = pl.yield_(
                    qr_fp32_inline0_inline1798__tile
                )
            qr_fp32_inline0_inline1798__rv_v2: pl.Tensor[[qr_t_matmul_inline1_inline1795__ssa_v0, 1024], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)] = pl.yield_(
                qr_fp32_inline0_inline1798__rv_v4
            )
        return qr_fp32_inline0_inline1798__ssa_v0

    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def qr_rms_norm_quant(
        tile_rows_inline2_inline1793__ssa_v0: pl.Scalar[pl.INDEX],
        tile_base_inline3_inline1791__idx_v0: pl.Scalar[pl.INDEX],
        qr_fp32_inline0_inline1798__rv_v10: pl.Tensor[[qr_t_matmul_inline1_inline1795__ssa_v0, 1024], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)],
        gamma_cq__ssa_v0: pl.Tensor[[1024], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 2048)],
        qr_scale_pad_store_inline1797__iter_v1: pl.InOut[pl.Tensor[[512, 1], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 2048)]],
        qr_scale_view_inline3048__ssa_v0: pl.Out[pl.Tensor[[t_dim_inline3071__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)]],
        qr_i8_matmul_inline1790__iter_v1: pl.InOut[pl.Tensor[[512, 1024], pl.INT8, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 524288)]],
        qr_view_inline3057__ssa_v0: pl.Out[pl.Tensor[[t_dim_inline3071__ssa_v0, 1024], pl.INT8, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 0)]],
    ) -> tuple[pl.Tensor[[512, 1], pl.FP32], pl.Tensor[[512, 1024], pl.INT8], pl.Tensor[[t_dim_inline3071__ssa_v0, 1], pl.FP32]]:
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
        tg_idx_inline3054__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        tg_inline3043__ssa_v0: pl.Scalar[pl.INDEX] = tg_idx_inline3054__ssa_v0 * 8
        valid_rows_inline3053__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(tile_rows_inline2_inline1793__ssa_v0 - tg_inline3043__ssa_v0, 8)
        out_tg_inline3079__ssa_v0: pl.Scalar[pl.INDEX] = tile_base_inline3_inline1791__idx_v0 + tg_inline3043__ssa_v0
        t__tile: pl.Tile[[8, 1024], pl.FP32, pl.MemRef(mem_vec_6, pl.const(17504, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.load(
            qr_fp32_inline0_inline1798__rv_v10, [tg_inline3043__ssa_v0, 0], [8, 1024], [8, 1024], target_memory=pl.Mem.Vec
        )
        t__tile_1: pl.Tile[[8, 1024], pl.BF16, pl.MemRef(mem_vec_7, pl.const(50272, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.cast(t__tile, target_type=pl.BF16, mode="rint")
        qr_rms_full_inline1297_inline3055__tile: pl.Tile[[8, 1024], pl.FP32, pl.MemRef(mem_vec_6, pl.const(17504, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.cast(
            t__tile_1, target_type=pl.FP32, mode="round"
        )
        qr_square_full_inline1285_inline3044__tile: pl.Tile[[8, 1024], pl.FP32, pl.MemRef(mem_vec_6, pl.const(17504, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.mul(
            qr_rms_full_inline1297_inline3055__tile, qr_rms_full_inline1297_inline3055__tile
        )
        t__tile_2: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_6, pl.const(17504, pl.INT64), 30720), pl.Mem.Vec] = pl.tile.slice(qr_square_full_inline1285_inline3044__tile, [8, 512], [0, 0])
        t__tile_3: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_6, pl.const(19552, pl.INT64), 30720), pl.Mem.Vec] = pl.tile.slice(qr_square_full_inline1285_inline3044__tile, [8, 512], [0, 512])
        qr_square_half_inline1279_inline3069__tile: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_6, pl.const(17504, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.add(t__tile_2, t__tile_3)
        t__tile_4: pl.Tile[[8, 256], pl.FP32, pl.MemRef(mem_vec_6, pl.const(17504, pl.INT64), 15360), pl.Mem.Vec] = pl.tile.slice(qr_square_half_inline1279_inline3069__tile, [8, 256], [0, 0])
        t__tile_5: pl.Tile[[8, 256], pl.FP32, pl.MemRef(mem_vec_6, pl.const(18528, pl.INT64), 15360), pl.Mem.Vec] = pl.tile.slice(qr_square_half_inline1279_inline3069__tile, [8, 256], [0, 256])
        qr_square_quarter_inline1283_inline3060__tile: pl.Tile[[8, 256], pl.FP32, pl.MemRef(mem_vec_6, pl.const(17504, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.add(t__tile_4, t__tile_5)
        t__tile_6: pl.Tile[[8, 128], pl.FP32, pl.MemRef(mem_vec_6, pl.const(17504, pl.INT64), 7680), pl.Mem.Vec] = pl.tile.slice(qr_square_quarter_inline1283_inline3060__tile, [8, 128], [0, 0])
        t__tile_7: pl.Tile[[8, 128], pl.FP32, pl.MemRef(mem_vec_6, pl.const(18016, pl.INT64), 7680), pl.Mem.Vec] = pl.tile.slice(qr_square_quarter_inline1283_inline3060__tile, [8, 128], [0, 128])
        qr_square_eighth_inline1284_inline3066__tile: pl.Tile[[8, 128], pl.FP32, pl.MemRef(mem_vec_6, pl.const(17504, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.add(t__tile_6, t__tile_7)
        t__tile_8: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_6, pl.const(17504, pl.INT64), 3840), pl.Mem.Vec] = pl.tile.slice(qr_square_eighth_inline1284_inline3066__tile, [8, 64], [0, 0])
        t__tile_9: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_6, pl.const(17760, pl.INT64), 3840), pl.Mem.Vec] = pl.tile.slice(qr_square_eighth_inline1284_inline3066__tile, [8, 64], [0, 64])
        qr_square_sixteenth_inline1288_inline3056__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_7, pl.const(50272, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.add(t__tile_8, t__tile_9)
        tmp_tile: pl.Tile[[8, 128], pl.FP32, pl.MemRef(mem_vec_6, pl.const(17504, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.create([8, 128], dtype=pl.FP32, target_memory=pl.Mem.Vec)
        t__tile_10: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_32, pl.const(1088, pl.INT64), 32), pl.Mem.Vec] = pl.tile.row_sum(qr_square_sixteenth_inline1288_inline3056__tile, tmp_tile)
        qr_sq_sum_inline1281_inline3028__tile: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_32, pl.const(1088, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(t__tile_10, [1, 8])
        t__tile_11: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_6, pl.const(17504, pl.INT64), 32), pl.Mem.Vec] = pl.tile.muls(qr_sq_sum_inline1281_inline3028__tile, 0.0009765625)
        qr_variance_inline1277_inline3031__tile: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_6, pl.const(17504, pl.INT64), 32), pl.Mem.Vec] = pl.tile.adds(t__tile_11, 9.9999999999999995e-07)
        qr_rms_approx_inline1278_inline3072__tile: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_7, pl.const(50272, pl.INT64), 32), pl.Mem.Vec] = pl.tile.sqrt(qr_variance_inline1277_inline3031__tile)
        qr_variance_bits_inline1274_inline3064__tile: pl.Tile[[1, 8], pl.INT32, pl.MemRef(mem_vec_6, pl.const(17504, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reinterpret_view(
            qr_variance_inline1277_inline3031__tile, dtype=pl.INT32
        )
        qr_approx_bits_inline1300_inline3039__tile: pl.Tile[[1, 8], pl.INT32, pl.MemRef(mem_vec_7, pl.const(50272, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reinterpret_view(
            qr_rms_approx_inline1278_inline3072__tile, dtype=pl.INT32
        )
        qr_rounded_bits_inline1290_inline3042__tile: pl.Tile[[1, 8], pl.INT32, pl.MemRef(mem_vec_41, pl.const(9280, pl.INT64), 32), pl.Mem.Vec] = pl.tile.full([1, 8], dtype=pl.INT32, value=0)
        for qr_sqrt_row_inline1282_inline3032__idx_v0 in pl.range(8):
            t__tile_12: pl.Scalar[pl.INT32] = pl.tile.read(qr_variance_bits_inline1274_inline3064__tile, [0, qr_sqrt_row_inline1282_inline3032__idx_v0])
            qr_x_bits_inline1287_inline3030__ssa_v0: pl.Scalar[pl.INT64] = pl.cast(t__tile_12, pl.INT64)
            t__tile_13: pl.Scalar[pl.INT32] = pl.tile.read(qr_approx_bits_inline1300_inline3039__tile, [0, qr_sqrt_row_inline1282_inline3032__idx_v0])
            qr_y_bits_inline1291_inline3027__ssa_v0: pl.Scalar[pl.INT64] = pl.cast(t__tile_13, pl.INT64)
            if 8388608 <= qr_x_bits_inline1287_inline3030__ssa_v0 and qr_x_bits_inline1287_inline3030__ssa_v0 < 2139095040:
                qr_x_exp_inline1289_inline3070__ssa_v0: pl.Scalar[pl.INT64] = qr_x_bits_inline1287_inline3030__ssa_v0 // 8388608
                qr_x_sig_inline1295_inline3035__ssa_v0: pl.Scalar[pl.INT64] = qr_x_bits_inline1287_inline3030__ssa_v0 % 8388608 + 8388608
                for _qr_sqrt_step_inline1296_inline3036__idx_v0, (qr_y_bits_inline1291_inline3027__iter_v1,) in pl.range(2, init_values=(qr_y_bits_inline1291_inline3027__ssa_v0,)):
                    qr_y_exp_inline1273_inline3034__ssa_v0: pl.Scalar[pl.INT64] = qr_y_bits_inline1291_inline3027__iter_v1 // 8388608
                    qr_y_sig_inline1286_inline3041__ssa_v0: pl.Scalar[pl.INT64] = qr_y_bits_inline1291_inline3027__iter_v1 % 8388608 + 8388608
                    qr_upper_mid_inline1298_inline3026__ssa_v0: pl.Scalar[pl.INDEX] = qr_y_sig_inline1286_inline3041__ssa_v0 * 2 + 1
                    qr_upper_square_inline1280_inline3040__ssa_v0: pl.Scalar[pl.INDEX] = qr_upper_mid_inline1298_inline3026__ssa_v0 * qr_upper_mid_inline1298_inline3026__ssa_v0
                    qr_x_upper_inline1275_inline3049__ssa_v0: pl.Scalar[pl.INT64] = (
                        qr_x_sig_inline1295_inline3035__ssa_v0 << qr_x_exp_inline1289_inline3070__ssa_v0 - qr_y_exp_inline1273_inline3034__ssa_v0 * 2 + 152
                    )
                    qr_previous_bits_inline1301_inline3029__ssa_v0: pl.Scalar[pl.INT64] = qr_y_bits_inline1291_inline3027__iter_v1 - 1
                    qr_previous_exp_inline1293_inline3061__ssa_v0: pl.Scalar[pl.INT64] = qr_previous_bits_inline1301_inline3029__ssa_v0 // 8388608
                    qr_previous_sig_inline1299_inline3073__ssa_v0: pl.Scalar[pl.INT64] = qr_previous_bits_inline1301_inline3029__ssa_v0 % 8388608 + 8388608
                    qr_lower_mid_inline1302_inline3052__ssa_v0: pl.Scalar[pl.INDEX] = qr_previous_sig_inline1299_inline3073__ssa_v0 * 2 + 1
                    qr_lower_square_inline1276_inline3047__ssa_v0: pl.Scalar[pl.INDEX] = qr_lower_mid_inline1302_inline3052__ssa_v0 * qr_lower_mid_inline1302_inline3052__ssa_v0
                    qr_x_lower_inline1303_inline3062__ssa_v0: pl.Scalar[pl.INT64] = (
                        qr_x_sig_inline1295_inline3035__ssa_v0 << qr_x_exp_inline1289_inline3070__ssa_v0 - qr_previous_exp_inline1293_inline3061__ssa_v0 * 2 + 152
                    )
                    if (
                        qr_upper_square_inline1280_inline3040__ssa_v0 < qr_x_upper_inline1275_inline3049__ssa_v0
                        or qr_x_upper_inline1275_inline3049__ssa_v0 == qr_upper_square_inline1280_inline3040__ssa_v0
                        and qr_y_bits_inline1291_inline3027__iter_v1 % 2 == 1
                    ):
                        qr_y_bits_inline1291_inline3027__ssa_v3: pl.Scalar[pl.INT64] = qr_y_bits_inline1291_inline3027__iter_v1 + 1
                        qr_y_bits_inline1291_inline3027__phi_v6: pl.Scalar[pl.INT64] = pl.yield_(qr_y_bits_inline1291_inline3027__ssa_v3)
                    else:
                        if (
                            qr_x_lower_inline1303_inline3062__ssa_v0 < qr_lower_square_inline1276_inline3047__ssa_v0
                            or qr_x_lower_inline1303_inline3062__ssa_v0 == qr_lower_square_inline1276_inline3047__ssa_v0
                            and qr_y_bits_inline1291_inline3027__iter_v1 % 2 == 1
                        ):
                            qr_y_bits_inline1291_inline3027__ssa_v4: pl.Scalar[pl.INT64] = qr_y_bits_inline1291_inline3027__iter_v1 - 1
                            qr_y_bits_inline1291_inline3027__phi_v5: pl.Scalar[pl.INT64] = pl.yield_(qr_y_bits_inline1291_inline3027__ssa_v4)
                        else:
                            qr_y_bits_inline1291_inline3027__phi_v5: pl.Scalar[pl.INT64] = pl.yield_(qr_y_bits_inline1291_inline3027__iter_v1)
                        qr_y_bits_inline1291_inline3027__phi_v6: pl.Scalar[pl.INT64] = pl.yield_(qr_y_bits_inline1291_inline3027__phi_v5)
                    qr_y_bits_inline1291_inline3027__rv_v2: pl.Scalar[pl.INT64] = pl.yield_(qr_y_bits_inline1291_inline3027__phi_v6)
                pl.tile.write(qr_rounded_bits_inline1290_inline3042__tile, [0, qr_sqrt_row_inline1282_inline3032__idx_v0], pl.cast(qr_y_bits_inline1291_inline3027__rv_v2, pl.INT32))
            else:
                pl.tile.write(qr_rounded_bits_inline1290_inline3042__tile, [0, qr_sqrt_row_inline1282_inline3032__idx_v0], pl.cast(qr_y_bits_inline1291_inline3027__ssa_v0, pl.INT32))
        qr_rms_inline1294_inline3068__tile: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_41, pl.const(9280, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reinterpret_view(
            qr_rounded_bits_inline1290_inline3042__tile, dtype=pl.FP32
        )
        qr_inv_rms_inline1292_inline3033__tile: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_20, pl.const(0, pl.INT64), 32), pl.Mem.Vec] = pl.tile.full([1, 8], dtype=pl.FP32, value=0.0)
        for qr_rms_row_inline1272_inline3074__idx_v0 in pl.range(8):
            qr_rms_value_inline1271_inline3081__tile: pl.Scalar[pl.FP32] = pl.tile.read(qr_rms_inline1294_inline3068__tile, [0, qr_rms_row_inline1272_inline3074__idx_v0])
            pl.tile.write(qr_inv_rms_inline1292_inline3033__tile, [0, qr_rms_row_inline1272_inline3074__idx_v0], 1.0 / qr_rms_value_inline1271_inline3081__tile)
        _qr_sq_sum_inline3083__ssa_v0: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_32, pl.const(1088, pl.INT64), 32), pl.Mem.Vec] = qr_sq_sum_inline1281_inline3028__tile
        qr_inv_rms_inline3075__ssa_v0: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_20, pl.const(0, pl.INT64), 32), pl.Mem.Vec] = qr_inv_rms_inline1292_inline3033__tile
        _qr_rms_inline3051__ssa_v0: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_41, pl.const(9280, pl.INT64), 32), pl.Mem.Vec] = qr_rms_inline1294_inline3068__tile
        qr_inv_rms_t_inline3059__tile: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_20, pl.const(0, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(qr_inv_rms_inline3075__ssa_v0, [8, 1])
        qr_tile_amax_inline3037__tile: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_21, pl.const(32, pl.INT64), 32), pl.Mem.Vec] = pl.tile.full([1, 8], dtype=pl.FP32, value=0.0001)
        for qr_max_col0_inline3067__idx_v0, (qr_tile_amax_inline3037__iter_v1,) in pl.range(0, 1024, 512, init_values=(qr_tile_amax_inline3037__tile,)):
            t__tile_14: pl.Tile[[8, 256], pl.FP32, pl.MemRef(mem_vec_6, pl.const(17504, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.load(
                qr_fp32_inline0_inline1798__rv_v10, [tg_inline3043__ssa_v0, qr_max_col0_inline3067__idx_v0], [8, 256], [8, 256], target_memory=pl.Mem.Vec
            )
            t__tile_15: pl.Tile[[256], pl.BF16, pl.MemRef(mem_vec_23, pl.const(64, pl.INT64), 512), pl.Mem.Vec] = pl.tile.load(
                gamma_cq__ssa_v0, [qr_max_col0_inline3067__idx_v0], [256], [256], target_memory=pl.Mem.Vec
            )
            t__tile_16: pl.Tile[[8, 256], pl.FP32, pl.MemRef(mem_vec_7, pl.const(50272, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.load(
                qr_fp32_inline0_inline1798__rv_v10, [tg_inline3043__ssa_v0, qr_max_col0_inline3067__idx_v0 + 256], [8, 256], [8, 256], target_memory=pl.Mem.Vec
            )
            t__tile_17: pl.Tile[[256], pl.BF16, pl.MemRef(mem_vec_25, pl.const(576, pl.INT64), 512), pl.Mem.Vec] = pl.tile.load(
                gamma_cq__ssa_v0, [qr_max_col0_inline3067__idx_v0 + 256], [256], [256], target_memory=pl.Mem.Vec
            )
            t__tile_18: pl.Tile[[8, 256], pl.BF16, pl.MemRef(mem_vec_32, pl.const(1088, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(t__tile_14, target_type=pl.BF16, mode="rint")
            qr_max_chunk_inline3076__tile: pl.Tile[[8, 256], pl.FP32, pl.MemRef(mem_vec_6, pl.const(17504, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.cast(t__tile_18, target_type=pl.FP32, mode="round")
            gamma_max_cast_inline3077__tile: pl.Tile[[256], pl.FP32, pl.MemRef(mem_vec_32, pl.const(1088, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.cast(t__tile_15, target_type=pl.FP32, mode="round")
            gamma_max_chunk_inline3046__tile: pl.Tile[[1, 256], pl.FP32, pl.MemRef(mem_vec_32, pl.const(1088, pl.INT64), 1024), pl.Mem.Vec] = gamma_max_cast_inline3077__tile
            t__tile_19: pl.Tile[[8, 256], pl.FP32, pl.MemRef(mem_vec_6, pl.const(17504, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.row_expand_mul(
                qr_max_chunk_inline3076__tile, qr_inv_rms_t_inline3059__tile
            )
            qr_normalized_inline3078__tile: pl.Tile[[8, 256], pl.FP32, pl.MemRef(mem_vec_6, pl.const(17504, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.col_expand_mul(
                t__tile_19, gamma_max_chunk_inline3046__tile
            )
            t__tile_20: pl.Tile[[8, 256], pl.FP32, pl.MemRef(mem_vec_6, pl.const(17504, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.abs(qr_normalized_inline3078__tile)
            tmp_tile_1: pl.Tile[[8, 256], pl.FP32, pl.MemRef(mem_vec_32, pl.const(1088, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.create([8, 256], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            t__tile_21: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_23, pl.const(64, pl.INT64), 32), pl.Mem.Vec] = pl.tile.row_max(t__tile_20, tmp_tile_1)
            qr_normalized_max_inline3080__tile: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_23, pl.const(64, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(t__tile_21, [1, 8])
            qr_tile_amax_inline3037__tile_1: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_6, pl.const(17504, pl.INT64), 32), pl.Mem.Vec] = pl.tile.maximum(
                qr_tile_amax_inline3037__iter_v1, qr_normalized_max_inline3080__tile
            )
            t__tile_22: pl.Tile[[8, 256], pl.BF16, pl.MemRef(mem_vec_41, pl.const(9280, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(t__tile_16, target_type=pl.BF16, mode="rint")
            qr_max_chunk_inline3076__tile_1: pl.Tile[[8, 256], pl.FP32, pl.MemRef(mem_vec_7, pl.const(50272, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.cast(t__tile_22, target_type=pl.FP32, mode="round")
            gamma_max_cast_inline3077__tile_1: pl.Tile[[256], pl.FP32, pl.MemRef(mem_vec_41, pl.const(9280, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.cast(t__tile_17, target_type=pl.FP32, mode="round")
            gamma_max_chunk_inline3046__tile_1: pl.Tile[[1, 256], pl.FP32, pl.MemRef(mem_vec_41, pl.const(9280, pl.INT64), 1024), pl.Mem.Vec] = gamma_max_cast_inline3077__tile_1
            t__tile_23: pl.Tile[[8, 256], pl.FP32, pl.MemRef(mem_vec_7, pl.const(50272, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.row_expand_mul(
                qr_max_chunk_inline3076__tile_1, qr_inv_rms_t_inline3059__tile
            )
            qr_normalized_inline3078__tile_1: pl.Tile[[8, 256], pl.FP32, pl.MemRef(mem_vec_7, pl.const(50272, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.col_expand_mul(
                t__tile_23, gamma_max_chunk_inline3046__tile_1
            )
            t__tile_24: pl.Tile[[8, 256], pl.FP32, pl.MemRef(mem_vec_7, pl.const(50272, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.abs(qr_normalized_inline3078__tile_1)
            tmp_tile_2: pl.Tile[[8, 256], pl.FP32, pl.MemRef(mem_vec_41, pl.const(9280, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.create([8, 256], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            t__tile_25: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_25, pl.const(576, pl.INT64), 32), pl.Mem.Vec] = pl.tile.row_max(t__tile_24, tmp_tile_2)
            qr_normalized_max_inline3080__tile_1: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_25, pl.const(576, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(t__tile_25, [1, 8])
            qr_tile_amax_inline3037__tile_2: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_21, pl.const(32, pl.INT64), 32), pl.Mem.Vec] = pl.tile.maximum(
                qr_tile_amax_inline3037__tile_1, qr_normalized_max_inline3080__tile_1
            )
            qr_tile_amax_inline3037__rv_v2: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_21, pl.const(32, pl.INT64), 32), pl.Mem.Vec] = pl.yield_(qr_tile_amax_inline3037__tile_2)
        qr_scale_quant_row_inline3063__tile: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_45, pl.const(17472, pl.INT64), 32), pl.Mem.Vec] = pl.tile.full([1, 8], dtype=pl.FP32, value=0.0)
        qr_scale_dq_row_inline3082__tile: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_6, pl.const(17504, pl.INT64), 32), pl.Mem.Vec] = pl.tile.full([1, 8], dtype=pl.FP32, value=0.0)
        for qr_scale_row_inline3084__idx_v0 in pl.range(8):
            qr_amax_value_inline3085__tile: pl.Scalar[pl.FP32] = pl.tile.read(qr_tile_amax_inline3037__rv_v2, [0, qr_scale_row_inline3084__idx_v0])
            qr_quant_value_inline3038__ssa_v0: pl.Scalar[pl.FP32] = 127.0 / qr_amax_value_inline3085__tile
            pl.tile.write(qr_scale_quant_row_inline3063__tile, [0, qr_scale_row_inline3084__idx_v0], qr_quant_value_inline3038__ssa_v0)
            pl.tile.write(qr_scale_dq_row_inline3082__tile, [0, qr_scale_row_inline3084__idx_v0], 1.0 / qr_quant_value_inline3038__ssa_v0)
        qr_scale_quant_t_inline3086__tile: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_45, pl.const(17472, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(qr_scale_quant_row_inline3063__tile, [8, 1])
        qr_tile_scale_dq_inline3087__tile: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_6, pl.const(17504, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(qr_scale_dq_row_inline3082__tile, [8, 1])
        qr_scale_pad_store_inline1797__tile: pl.Tensor[[512, 1], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 2048)] = pl.tile.store(
            qr_tile_scale_dq_inline3087__tile, [tg_inline3043__ssa_v0, 0], qr_scale_pad_store_inline1797__iter_v1
        )
        if valid_rows_inline3053__ssa_v0 == 8:
            qr_scale_view_inline3048__tile: pl.Tensor[[t_dim_inline3071__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                qr_tile_scale_dq_inline3087__tile, [out_tg_inline3079__ssa_v0, 0], qr_scale_view_inline3048__ssa_v0
            )
        else:
            qr_scale_tail_inline3065__ssa_v0: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_6, pl.const(17504, pl.INT64), 32), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline3053__ssa_v0, 1])] = (
                pl.tile.load(qr_scale_pad_store_inline1797__tile, [tg_inline3043__ssa_v0, 0], [8, 1], [valid_rows_inline3053__ssa_v0, 1], target_memory=pl.Mem.Vec)
            )
            qr_scale_view_inline3048__store: pl.Tensor[[t_dim_inline3071__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                qr_scale_tail_inline3065__ssa_v0, [out_tg_inline3079__ssa_v0, 0], qr_scale_view_inline3048__ssa_v0
            )
        for qa_inline3025__idx_v0, (qr_i8_matmul_inline1790__iter_v3, qr_view_inline3057__iter_v1) in pl.range(
            0, 1024, 512, init_values=(qr_i8_matmul_inline1790__iter_v1, qr_view_inline3057__ssa_v0)
        ):
            t__tile_26: pl.Tile[[8, 256], pl.FP32, pl.MemRef(mem_vec_6, pl.const(17504, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.load(
                qr_fp32_inline0_inline1798__rv_v10, [tg_inline3043__ssa_v0, qa_inline3025__idx_v0], [8, 256], [8, 256], target_memory=pl.Mem.Vec
            )
            t__tile_27: pl.Tile[[256], pl.BF16, pl.MemRef(mem_vec_23, pl.const(64, pl.INT64), 512), pl.Mem.Vec] = pl.tile.load(
                gamma_cq__ssa_v0, [qa_inline3025__idx_v0], [256], [256], target_memory=pl.Mem.Vec
            )
            t__tile_28: pl.Tile[[8, 256], pl.FP32, pl.MemRef(mem_vec_7, pl.const(50272, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.load(
                qr_fp32_inline0_inline1798__rv_v10, [tg_inline3043__ssa_v0, qa_inline3025__idx_v0 + 256], [8, 256], [8, 256], target_memory=pl.Mem.Vec
            )
            t__tile_29: pl.Tile[[256], pl.BF16, pl.MemRef(mem_vec_25, pl.const(576, pl.INT64), 512), pl.Mem.Vec] = pl.tile.load(
                gamma_cq__ssa_v0, [qa_inline3025__idx_v0 + 256], [256], [256], target_memory=pl.Mem.Vec
            )
            t__tile_30: pl.Tile[[8, 256], pl.BF16, pl.MemRef(mem_vec_32, pl.const(1088, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(t__tile_26, target_type=pl.BF16, mode="rint")
            qr_chunk_inline3024__tile: pl.Tile[[8, 256], pl.FP32, pl.MemRef(mem_vec_6, pl.const(17504, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.cast(t__tile_30, target_type=pl.FP32, mode="round")
            gamma_q_cast_inline3058__tile: pl.Tile[[256], pl.FP32, pl.MemRef(mem_vec_32, pl.const(1088, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.cast(t__tile_27, target_type=pl.FP32, mode="round")
            gamma_q_chunk_inline3023__tile: pl.Tile[[1, 256], pl.FP32, pl.MemRef(mem_vec_32, pl.const(1088, pl.INT64), 1024), pl.Mem.Vec] = gamma_q_cast_inline3058__tile
            t__tile_31: pl.Tile[[8, 256], pl.FP32, pl.MemRef(mem_vec_6, pl.const(17504, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.row_expand_mul(qr_chunk_inline3024__tile, qr_inv_rms_t_inline3059__tile)
            qr_q_normed_inline3022__tile: pl.Tile[[8, 256], pl.FP32, pl.MemRef(mem_vec_6, pl.const(17504, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.col_expand_mul(
                t__tile_31, gamma_q_chunk_inline3023__tile
            )
            qr_q_scaled_inline3021__tile: pl.Tile[[8, 256], pl.FP32, pl.MemRef(mem_vec_6, pl.const(17504, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.row_expand_mul(
                qr_q_normed_inline3022__tile, qr_scale_quant_t_inline3086__tile
            )
            qr_q_i32_inline3019__tile: pl.Tile[[8, 256], pl.INT32, pl.MemRef(mem_vec_6, pl.const(17504, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.cast(
                qr_q_scaled_inline3021__tile, target_type=pl.INT32, mode="rint"
            )
            qr_q_half_inline3020__tile: pl.Tile[[8, 256], pl.FP16, pl.MemRef(mem_vec_6, pl.const(17504, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(
                qr_q_i32_inline3019__tile, target_type=pl.FP16, mode="round"
            )
            qr_q_i8_inline3018__tile: pl.Tile[[8, 256], pl.INT8, pl.MemRef(mem_vec_6, pl.const(17504, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(
                qr_q_half_inline3020__tile, target_type=pl.INT8, mode="trunc"
            )
            qr_i8_matmul_inline1790__tile: pl.Tensor[[512, 1024], pl.INT8, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 524288)] = pl.tile.store(
                qr_q_i8_inline3018__tile, [tg_inline3043__ssa_v0, qa_inline3025__idx_v0], qr_i8_matmul_inline1790__iter_v3
            )
            if valid_rows_inline3053__ssa_v0 == 8:
                qr_view_inline3057__tile: pl.Tensor[[t_dim_inline3071__ssa_v0, 1024], pl.INT8, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                    qr_q_i8_inline3018__tile, [out_tg_inline3079__ssa_v0, qa_inline3025__idx_v0], qr_view_inline3057__iter_v1
                )
                qr_view_inline3057__phi_v4: pl.Tensor[[t_dim_inline3071__ssa_v0, 1024], pl.INT8, pl.MemRef("mem_ddr_62", pl.const(0, pl.INT64), 0)] = pl.yield_(qr_view_inline3057__tile)
            else:
                qr_q_tail_inline3045__ssa_v0: pl.Tile[
                    [8, 256], pl.INT8, pl.MemRef(mem_vec_6, pl.const(17504, pl.INT64), 2048), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline3053__ssa_v0, 256])
                ] = pl.tile.load(qr_i8_matmul_inline1790__tile, [tg_inline3043__ssa_v0, qa_inline3025__idx_v0], [8, 256], [valid_rows_inline3053__ssa_v0, 256], target_memory=pl.Mem.Vec)
                pl.tile.store(qr_q_tail_inline3045__ssa_v0, [out_tg_inline3079__ssa_v0, qa_inline3025__idx_v0], qr_view_inline3057__iter_v1)
                qr_view_inline3057__phi_v4: pl.Tensor[[t_dim_inline3071__ssa_v0, 1024], pl.INT8, pl.MemRef("mem_ddr_62", pl.const(0, pl.INT64), 0)] = pl.yield_(qr_view_inline3057__iter_v1)
            t__tile_32: pl.Tile[[8, 256], pl.BF16, pl.MemRef(mem_vec_41, pl.const(9280, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(t__tile_28, target_type=pl.BF16, mode="rint")
            qr_chunk_inline3024__tile_1: pl.Tile[[8, 256], pl.FP32, pl.MemRef(mem_vec_7, pl.const(50272, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.cast(t__tile_32, target_type=pl.FP32, mode="round")
            gamma_q_cast_inline3058__tile_1: pl.Tile[[256], pl.FP32, pl.MemRef(mem_vec_41, pl.const(9280, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.cast(t__tile_29, target_type=pl.FP32, mode="round")
            gamma_q_chunk_inline3023__tile_1: pl.Tile[[1, 256], pl.FP32, pl.MemRef(mem_vec_41, pl.const(9280, pl.INT64), 1024), pl.Mem.Vec] = gamma_q_cast_inline3058__tile_1
            t__tile_33: pl.Tile[[8, 256], pl.FP32, pl.MemRef(mem_vec_7, pl.const(50272, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.row_expand_mul(
                qr_chunk_inline3024__tile_1, qr_inv_rms_t_inline3059__tile
            )
            qr_q_normed_inline3022__tile_1: pl.Tile[[8, 256], pl.FP32, pl.MemRef(mem_vec_7, pl.const(50272, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.col_expand_mul(
                t__tile_33, gamma_q_chunk_inline3023__tile_1
            )
            qr_q_scaled_inline3021__tile_1: pl.Tile[[8, 256], pl.FP32, pl.MemRef(mem_vec_7, pl.const(50272, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.row_expand_mul(
                qr_q_normed_inline3022__tile_1, qr_scale_quant_t_inline3086__tile
            )
            qr_q_i32_inline3019__tile_1: pl.Tile[[8, 256], pl.INT32, pl.MemRef(mem_vec_7, pl.const(50272, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.cast(
                qr_q_scaled_inline3021__tile_1, target_type=pl.INT32, mode="rint"
            )
            qr_q_half_inline3020__tile_1: pl.Tile[[8, 256], pl.FP16, pl.MemRef(mem_vec_7, pl.const(50272, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(
                qr_q_i32_inline3019__tile_1, target_type=pl.FP16, mode="round"
            )
            qr_q_i8_inline3018__tile_1: pl.Tile[[8, 256], pl.INT8, pl.MemRef(mem_vec_7, pl.const(50272, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(
                qr_q_half_inline3020__tile_1, target_type=pl.INT8, mode="trunc"
            )
            qr_i8_matmul_inline1790__tile_1: pl.Tensor[[512, 1024], pl.INT8, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 524288)] = pl.tile.store(
                qr_q_i8_inline3018__tile_1, [tg_inline3043__ssa_v0, qa_inline3025__idx_v0 + 256], qr_i8_matmul_inline1790__tile
            )
            if valid_rows_inline3053__ssa_v0 == 8:
                qr_view_inline3057__tile_1: pl.Tensor[[t_dim_inline3071__ssa_v0, 1024], pl.INT8, pl.MemRef("mem_ddr_62", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                    qr_q_i8_inline3018__tile_1, [out_tg_inline3079__ssa_v0, qa_inline3025__idx_v0 + 256], qr_view_inline3057__phi_v4
                )
                qr_view_inline3057__phi_v4_1: pl.Tensor[[t_dim_inline3071__ssa_v0, 1024], pl.INT8, pl.MemRef("mem_ddr_73", pl.const(0, pl.INT64), 0)] = pl.yield_(qr_view_inline3057__tile_1)
            else:
                qr_q_tail_inline3045__ssa_v0_1: pl.Tile[
                    [8, 256], pl.INT8, pl.MemRef(mem_vec_7, pl.const(50272, pl.INT64), 2048), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline3053__ssa_v0, 256])
                ] = pl.tile.load(qr_i8_matmul_inline1790__tile_1, [tg_inline3043__ssa_v0, qa_inline3025__idx_v0 + 256], [8, 256], [valid_rows_inline3053__ssa_v0, 256], target_memory=pl.Mem.Vec)
                pl.tile.store(qr_q_tail_inline3045__ssa_v0_1, [out_tg_inline3079__ssa_v0, qa_inline3025__idx_v0 + 256], qr_view_inline3057__phi_v4)
                qr_view_inline3057__phi_v4_1: pl.Tensor[[t_dim_inline3071__ssa_v0, 1024], pl.INT8, pl.MemRef("mem_ddr_73", pl.const(0, pl.INT64), 0)] = pl.yield_(qr_view_inline3057__phi_v4)
            qr_i8_matmul_inline1790__rv_v4, qr_view_inline3057__rv_v2 = pl.yield_(qr_i8_matmul_inline1790__tile_1, qr_view_inline3057__phi_v4_1)
        return qr_scale_pad_store_inline1797__iter_v1, qr_i8_matmul_inline1790__iter_v1, qr_scale_view_inline3048__ssa_v0

    @pl.function(type=pl.FunctionType.Spmd)
    def qr_rms_norm_quant_spmd(
        self,
        tile_rows_inline2_inline1793__ssa_v0: pl.Scalar[pl.INDEX],
        tile_base_inline3_inline1791__idx_v0: pl.Scalar[pl.INDEX],
        qr_fp32_inline0_inline1798__rv_v10: pl.Tensor[[qr_t_matmul_inline1_inline1795__ssa_v0, 1024], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)],
        gamma_cq__ssa_v0: pl.Tensor[[1024], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 2048)],
        qr_scale_pad_store_inline1797__iter_v1: pl.InOut[pl.Tensor[[512, 1], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 2048)]],
        qr_scale_view_inline3048__ssa_v0: pl.Out[pl.Tensor[[t_dim_inline3071__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)]],
        qr_i8_matmul_inline1790__iter_v1: pl.InOut[pl.Tensor[[512, 1024], pl.INT8, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 524288)]],
        qr_view_inline3057__ssa_v0: pl.Out[pl.Tensor[[t_dim_inline3071__ssa_v0, 1024], pl.INT8, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 0)]],
    ) -> tuple[pl.Tensor[[512, 1], pl.FP32], pl.Tensor[[512, 1024], pl.INT8]]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        ret__tmp_v0: pl.Tuple[pl.Tensor[[512, 1], pl.FP32], pl.Tensor[[512, 1024], pl.INT8], pl.Tensor[[t_dim_inline3071__ssa_v0, 1], pl.FP32]] = self.qr_rms_norm_quant(
            tile_rows_inline2_inline1793__ssa_v0,
            tile_base_inline3_inline1791__idx_v0,
            qr_fp32_inline0_inline1798__rv_v10,
            gamma_cq__ssa_v0,
            qr_scale_pad_store_inline1797__iter_v1,
            qr_scale_view_inline3048__ssa_v0,
            qr_i8_matmul_inline1790__iter_v1,
            qr_view_inline3057__ssa_v0,
            attrs={"arg_directions": [pl.adir.scalar, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.inout, pl.adir.output_existing, pl.adir.inout, pl.adir.output_existing]},
        )
        qr_scale_pad_store_inline1797__ssa_v3: pl.Tensor[[512, 1], pl.FP32, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 2048)] = ret__tmp_v0[0]
        qr_i8_matmul_inline1790__rv_v4: pl.Tensor[[512, 1024], pl.INT8, pl.MemRef("mem_ddr_7", pl.const(0, pl.INT64), 524288)] = ret__tmp_v0[1]
        qr_scale_view_inline3048__ssa_v2: pl.Tensor[[t_dim_inline3071__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_8", pl.const(0, pl.INT64), 0)] = ret__tmp_v0[2]
        return qr_scale_pad_store_inline1797__iter_v1, qr_i8_matmul_inline1790__iter_v1

    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def quant(
        o_r_i8_pad_inline333__iter_v1: pl.Out[pl.Tensor[[384, 8192], pl.INT8, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 3145728)]],
        t_dim_inline347__ssa_v0: pl.Scalar[pl.INDEX],
        act_scale_q_inline338__rv_v2: pl.Tensor[[1, 384], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 1536)],
        o_r_pad_inline329__rv_v2: pl.Tensor[[384, 8192], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 12582912)],
        col_g_inline295__ssa_v0: pl.Scalar[pl.INDEX],
        proj_b_padded_rows_inline323__ssa_v0: pl.Scalar[pl.INDEX],
    ) -> pl.Tensor[[384, 8192], pl.INT8]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        mem_vec_3: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32)
        mem_vec_4: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32768)
        mem_vec_5: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32)
        mem_vec_6: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32768)
        mem_vec_7: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
        mem_vec_13: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
        unroll_main_end: pl.Scalar[pl.INDEX] = (t_dim_inline347__ssa_v0 + 7) // 16 * 16
        for qt_inline303__idx_v0, (o_r_i8_pad_inline333__iter_v3,) in pl.range(0, unroll_main_end, 16, init_values=(o_r_i8_pad_inline333__iter_v1,)):
            token_multiplier_inline322__tile: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_3, pl.const(16384, pl.INT64), 32), pl.Mem.Vec] = pl.tile.load(
                act_scale_q_inline338__rv_v2, [0, qt_inline303__idx_v0], [1, 8], [1, 8], target_memory=pl.Mem.Vec
            )
            oc_q_inline358__tile: pl.Tile[[8, 1024], pl.FP32, pl.MemRef(mem_vec_4, pl.const(16416, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.load(
                o_r_pad_inline329__rv_v2, [qt_inline303__idx_v0, col_g_inline295__ssa_v0], [8, 1024], [8, 1024], target_memory=pl.Mem.Vec
            )
            token_multiplier_inline322__tile_1: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_5, pl.const(49184, pl.INT64), 32), pl.Mem.Vec] = pl.tile.load(
                act_scale_q_inline338__rv_v2, [0, qt_inline303__idx_v0 + 8], [1, 8], [1, 8], target_memory=pl.Mem.Vec
            )
            oc_q_inline358__tile_1: pl.Tile[[8, 1024], pl.FP32, pl.MemRef(mem_vec_6, pl.const(49216, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.load(
                o_r_pad_inline329__rv_v2, [qt_inline303__idx_v0 + 8, col_g_inline295__ssa_v0], [8, 1024], [8, 1024], target_memory=pl.Mem.Vec
            )
            g_sq_col_inline305__tile: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_3, pl.const(16384, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(token_multiplier_inline322__tile, [8, 1])
            t__tile: pl.Tile[[8, 1024], pl.BF16, pl.MemRef(mem_vec_7, pl.const(81984, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.cast(oc_q_inline358__tile, target_type=pl.BF16, mode="rint")
            oc_q_v1_inline294__tile: pl.Tile[[8, 1024], pl.FP32, pl.MemRef(mem_vec_4, pl.const(16416, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.cast(t__tile, target_type=pl.FP32, mode="round")
            oq_scaled_inline361__tile: pl.Tile[[8, 1024], pl.FP32, pl.MemRef(mem_vec_4, pl.const(16416, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.row_expand_mul(
                oc_q_v1_inline294__tile, g_sq_col_inline305__tile
            )
            oq_i32_inline318__tile: pl.Tile[[8, 1024], pl.INT32, pl.MemRef(mem_vec_4, pl.const(16416, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.cast(
                oq_scaled_inline361__tile, target_type=pl.INT32, mode="rint"
            )
            oq_half_inline363__tile: pl.Tile[[8, 1024], pl.FP16, pl.MemRef(mem_vec_4, pl.const(16416, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.cast(
                oq_i32_inline318__tile, target_type=pl.FP16, mode="round"
            )
            oq_i8_inline290__tile: pl.Tile[[8, 1024], pl.INT8, pl.MemRef(mem_vec_4, pl.const(16416, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.cast(
                oq_half_inline363__tile, target_type=pl.INT8, mode="trunc"
            )
            o_r_i8_pad_inline333__tile: pl.Tensor[[384, 8192], pl.INT8, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 3145728)] = pl.tile.store(
                oq_i8_inline290__tile, [qt_inline303__idx_v0, col_g_inline295__ssa_v0], o_r_i8_pad_inline333__iter_v3
            )
            g_sq_col_inline305__tile_1: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_5, pl.const(49184, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(token_multiplier_inline322__tile_1, [8, 1])
            t__tile_1: pl.Tile[[8, 1024], pl.BF16, pl.MemRef(mem_vec_13, pl.const(0, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.cast(oc_q_inline358__tile_1, target_type=pl.BF16, mode="rint")
            oc_q_v1_inline294__tile_1: pl.Tile[[8, 1024], pl.FP32, pl.MemRef(mem_vec_6, pl.const(49216, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.cast(t__tile_1, target_type=pl.FP32, mode="round")
            oq_scaled_inline361__tile_1: pl.Tile[[8, 1024], pl.FP32, pl.MemRef(mem_vec_6, pl.const(49216, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.row_expand_mul(
                oc_q_v1_inline294__tile_1, g_sq_col_inline305__tile_1
            )
            oq_i32_inline318__tile_1: pl.Tile[[8, 1024], pl.INT32, pl.MemRef(mem_vec_6, pl.const(49216, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.cast(
                oq_scaled_inline361__tile_1, target_type=pl.INT32, mode="rint"
            )
            oq_half_inline363__tile_1: pl.Tile[[8, 1024], pl.FP16, pl.MemRef(mem_vec_6, pl.const(49216, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.cast(
                oq_i32_inline318__tile_1, target_type=pl.FP16, mode="round"
            )
            oq_i8_inline290__tile_1: pl.Tile[[8, 1024], pl.INT8, pl.MemRef(mem_vec_6, pl.const(49216, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.cast(
                oq_half_inline363__tile_1, target_type=pl.INT8, mode="trunc"
            )
            o_r_i8_pad_inline333__tile_1: pl.Tensor[[384, 8192], pl.INT8, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 3145728)] = pl.tile.store(
                oq_i8_inline290__tile_1, [qt_inline303__idx_v0 + 8, col_g_inline295__ssa_v0], o_r_i8_pad_inline333__tile
            )
            o_r_i8_pad_inline333__rv_v4_main: pl.Tensor[[384, 8192], pl.INT8, pl.MemRef("mem_ddr_19", pl.const(0, pl.INT64), 3145728)] = pl.yield_(o_r_i8_pad_inline333__tile_1)
        unroll_rem: pl.Scalar[pl.INDEX] = (t_dim_inline347__ssa_v0 + 7) // 8 - (t_dim_inline347__ssa_v0 + 7) // 16 * 2
        if unroll_rem == 1:
            token_multiplier_inline322__tile_2: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_7, pl.const(81984, pl.INT64), 32), pl.Mem.Vec] = pl.tile.load(
                act_scale_q_inline338__rv_v2, [0, unroll_main_end], [1, 8], [1, 8], target_memory=pl.Mem.Vec
            )
            g_sq_col_inline305__tile_2: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_7, pl.const(81984, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(token_multiplier_inline322__tile_2, [8, 1])
            oc_q_inline358__tile_2: pl.Tile[[8, 1024], pl.FP32, pl.MemRef(mem_vec_4, pl.const(16416, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.load(
                o_r_pad_inline329__rv_v2, [unroll_main_end, col_g_inline295__ssa_v0], [8, 1024], [8, 1024], target_memory=pl.Mem.Vec
            )
            t__tile_2: pl.Tile[[8, 1024], pl.BF16, pl.MemRef(mem_vec_6, pl.const(49216, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.cast(oc_q_inline358__tile_2, target_type=pl.BF16, mode="rint")
            oc_q_v1_inline294__tile_2: pl.Tile[[8, 1024], pl.FP32, pl.MemRef(mem_vec_4, pl.const(16416, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.cast(t__tile_2, target_type=pl.FP32, mode="round")
            oq_scaled_inline361__tile_2: pl.Tile[[8, 1024], pl.FP32, pl.MemRef(mem_vec_4, pl.const(16416, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.row_expand_mul(
                oc_q_v1_inline294__tile_2, g_sq_col_inline305__tile_2
            )
            oq_i32_inline318__tile_2: pl.Tile[[8, 1024], pl.INT32, pl.MemRef(mem_vec_4, pl.const(16416, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.cast(
                oq_scaled_inline361__tile_2, target_type=pl.INT32, mode="rint"
            )
            oq_half_inline363__tile_2: pl.Tile[[8, 1024], pl.FP16, pl.MemRef(mem_vec_4, pl.const(16416, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.cast(
                oq_i32_inline318__tile_2, target_type=pl.FP16, mode="round"
            )
            oq_i8_inline290__tile_2: pl.Tile[[8, 1024], pl.INT8, pl.MemRef(mem_vec_4, pl.const(16416, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.cast(
                oq_half_inline363__tile_2, target_type=pl.INT8, mode="trunc"
            )
            o_r_i8_pad_inline333__tile_2: pl.Tensor[[384, 8192], pl.INT8, pl.MemRef("mem_ddr_19", pl.const(0, pl.INT64), 3145728)] = pl.tile.store(
                oq_i8_inline290__tile_2, [unroll_main_end, col_g_inline295__ssa_v0], o_r_i8_pad_inline333__rv_v4_main
            )
            o_r_i8_pad_inline333__rv_v4: pl.Tensor[[384, 8192], pl.INT8, pl.MemRef("mem_ddr_28", pl.const(0, pl.INT64), 3145728)] = pl.yield_(o_r_i8_pad_inline333__tile_2)
        else:
            o_r_i8_pad_inline333__rv_v4: pl.Tensor[[384, 8192], pl.INT8, pl.MemRef("mem_ddr_28", pl.const(0, pl.INT64), 3145728)] = pl.yield_(o_r_i8_pad_inline333__rv_v4_main)
        for zt_inline289__idx_v0, (o_r_i8_pad_inline333__iter_v6,) in pl.range(t_dim_inline347__ssa_v0, proj_b_padded_rows_inline323__ssa_v0, 8, init_values=(o_r_i8_pad_inline333__rv_v4,)):
            zero_half_inline288__tile: pl.Tile[[8, 1024], pl.FP16, pl.MemRef(mem_vec_4, pl.const(16416, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.full([8, 1024], dtype=pl.FP16, value=0.0)
            t__tile_3: pl.Tile[[8, 1024], pl.INT8, pl.MemRef(mem_vec_4, pl.const(16416, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.cast(zero_half_inline288__tile, target_type=pl.INT8, mode="trunc")
            o_r_i8_pad_inline333__tile_3: pl.Tensor[[384, 8192], pl.INT8, pl.MemRef("mem_ddr_28", pl.const(0, pl.INT64), 3145728)] = pl.tile.store(
                t__tile_3, [zt_inline289__idx_v0, col_g_inline295__ssa_v0], o_r_i8_pad_inline333__iter_v6
            )
            o_r_i8_pad_inline333__rv_v7: pl.Tensor[[384, 8192], pl.INT8, pl.MemRef("mem_ddr_31", pl.const(0, pl.INT64), 3145728)] = pl.yield_(o_r_i8_pad_inline333__tile_3)
        return o_r_i8_pad_inline333__iter_v1

    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def rmsnorm_rope_cache_write(
        bs_inline2047__ssa_v0: pl.Scalar[pl.INDEX],
        position_ids__ssa_v0: pl.Tensor[[T_DYN], pl.INT64, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)],
        cmp_row_offsets__ssa_v0: pl.Tensor[[B_DYN], pl.INT32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)],
        cmp_freqs_cos__ssa_v0: pl.Tensor[[COMPRESSED_ROWS_DYN, 64], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)],
        cmp_freqs_sin__ssa_v0: pl.Tensor[[COMPRESSED_ROWS_DYN, 64], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)],
        pooled_kv_inline205__rv_v2: pl.Tensor[[384, 512], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 786432)],
        normed_kv_inline2015__ssa_v0: pl.InOut[pl.Tensor[[384, 512], pl.FP32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 786432)]],
        norm_w_2d_inline1999__ssa_v0: pl.Tensor[[1, 512], pl.FP32, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 2048)],
        cmp_kv_cache_flat_inline2016__ssa_v0: pl.Out[pl.Tensor[[cmp_block_num_inline2008__ssa_v0 * pl.const(32, pl.INDEX), 512], pl.BF16, pl.MemRef("mem_ddr_7", pl.const(0, pl.INT64), 0)]],
        kv_flat_inline2020__ssa_v0: pl.Out[pl.Tensor[[T_DYN, 512], pl.FP32, pl.MemRef("mem_ddr_8", pl.const(0, pl.INT64), 0)]],
        cmp_slot_mapping__ssa_v0: pl.Tensor[[COMPRESSED_ROWS_DYN, 2], pl.INT32, pl.MemRef("mem_ddr_9", pl.const(0, pl.INT64), 0)],
    ):
        pl.func_attr({"mx_tensor_views_blocked": True})
        mem_vec_10: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 4096)
        mem_vec_11: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 4096)
        mem_vec_16: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
        mem_vec_17: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
        mem_vec_42: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 4096)
        mem_vec_48: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 4096)
        mem_vec_52: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        mem_vec_53: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        rms_blk_inline2009__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        b0_inline1995__ssa_v0: pl.Scalar[pl.INDEX] = rms_blk_inline2009__ssa_v0 * 16
        rms_blk_rows_inline2017__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(bs_inline2047__ssa_v0 - b0_inline1995__ssa_v0, 16)
        cosine_inline369_inline2018__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.full([16, 64], dtype=pl.FP32, value=1.0)
        sine_inline368_inline1994__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_11, pl.const(4096, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.full([16, 64], dtype=pl.FP32, value=0.0)
        for row_inline367_inline2023__idx_v0, (cosine_inline369_inline2018__iter_v1, sine_inline368_inline1994__iter_v1) in pl.range(
            rms_blk_rows_inline2017__ssa_v0, init_values=(cosine_inline369_inline2018__tile, sine_inline368_inline1994__tile)
        ):
            token_inline366_inline2024__ssa_v0: pl.Scalar[pl.INDEX] = b0_inline1995__ssa_v0 + row_inline367_inline2023__idx_v0
            position_inline365_inline2025__tile: pl.Scalar[pl.INT64] = pl.tensor.read(position_ids__ssa_v0, [token_inline366_inline2024__ssa_v0])
            if (position_inline365_inline2025__tile + 1) % 4 == 0:
                t__tile: pl.Scalar[pl.INT32] = pl.tensor.read(cmp_row_offsets__ssa_v0, [token_inline366_inline2024__ssa_v0 // 6])
                compact_row_inline364_inline2026__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(t__tile, pl.INDEX) + pl.cast((position_inline365_inline2025__tile + 1) // 4, pl.INDEX)
                cosine_inline369_inline2018__tile_1: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.gather_row(
                    cosine_inline369_inline2018__iter_v1, cmp_freqs_cos__ssa_v0, [row_inline367_inline2023__idx_v0, 0], [compact_row_inline364_inline2026__ssa_v0, 0], [1, 64], transpose=False
                )
                sine_inline368_inline1994__tile_1: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_11, pl.const(4096, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.gather_row(
                    sine_inline368_inline1994__iter_v1, cmp_freqs_sin__ssa_v0, [row_inline367_inline2023__idx_v0, 0], [compact_row_inline364_inline2026__ssa_v0, 0], [1, 64], transpose=False
                )
                cosine_inline369_inline2018__phi_v4, sine_inline368_inline1994__phi_v4 = pl.yield_(cosine_inline369_inline2018__tile_1, sine_inline368_inline1994__tile_1)
            else:
                cosine_inline369_inline2018__phi_v4, sine_inline368_inline1994__phi_v4 = pl.yield_(cosine_inline369_inline2018__iter_v1, sine_inline368_inline1994__iter_v1)
            cosine_inline369_inline2018__rv_v2, sine_inline368_inline1994__rv_v2 = pl.yield_(cosine_inline369_inline2018__phi_v4, sine_inline368_inline1994__phi_v4)
        cos_b_inline2012__ssa_v0: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = cosine_inline369_inline2018__rv_v2
        sin_b_inline2027__ssa_v0: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_11, pl.const(4096, pl.INT64), 4096), pl.Mem.Vec] = sine_inline368_inline1994__rv_v2
        rms_low_inline2028__tile: pl.Tile[[16, 256], pl.FP32, pl.MemRef(mem_vec_16, pl.const(8192, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.load(
            pooled_kv_inline205__rv_v2, [b0_inline1995__ssa_v0, 0], [16, 256], [16, 256], target_memory=pl.Mem.Vec
        )
        rms_high_inline2030__tile: pl.Tile[[16, 256], pl.FP32, pl.MemRef(mem_vec_17, pl.const(24576, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.load(
            pooled_kv_inline205__rv_v2, [b0_inline1995__ssa_v0, 256], [16, 256], [16, 256], target_memory=pl.Mem.Vec
        )
        t__tile_1: pl.Tile[[16, 256], pl.FP32, pl.MemRef(mem_vec_16, pl.const(8192, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.mul(rms_low_inline2028__tile, rms_low_inline2028__tile)
        t__tile_2: pl.Tile[[16, 256], pl.FP32, pl.MemRef(mem_vec_17, pl.const(24576, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.mul(rms_high_inline2030__tile, rms_high_inline2030__tile)
        folded4_inline2044__tile: pl.Tile[[16, 256], pl.FP32, pl.MemRef(mem_vec_16, pl.const(8192, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.add(t__tile_1, t__tile_2)
        t__tile_3: pl.Tile[[16, 128], pl.FP32, pl.MemRef(mem_vec_16, pl.const(8192, pl.INT64), 15872), pl.Mem.Vec] = pl.tile.slice(folded4_inline2044__tile, [16, 128], [0, 0])
        t__tile_4: pl.Tile[[16, 128], pl.FP32, pl.MemRef(mem_vec_16, pl.const(8704, pl.INT64), 15872), pl.Mem.Vec] = pl.tile.slice(folded4_inline2044__tile, [16, 128], [0, 128])
        folded2_inline2031__tile: pl.Tile[[16, 128], pl.FP32, pl.MemRef(mem_vec_16, pl.const(8192, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.add(t__tile_3, t__tile_4)
        t__tile_5: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_16, pl.const(8192, pl.INT64), 7936), pl.Mem.Vec] = pl.tile.slice(folded2_inline2031__tile, [16, 64], [0, 0])
        t__tile_6: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_16, pl.const(8448, pl.INT64), 7936), pl.Mem.Vec] = pl.tile.slice(folded2_inline2031__tile, [16, 64], [0, 64])
        folded1_inline2032__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_17, pl.const(24576, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.add(t__tile_5, t__tile_6)
        tmp_tile: pl.Tile[[16, 128], pl.FP32, pl.MemRef(mem_vec_16, pl.const(8192, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.create([16, 128], dtype=pl.FP32, target_memory=pl.Mem.Vec)
        square_sum_inline2033__tile: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_42, pl.const(40960, pl.INT64), 64), pl.Mem.Vec] = pl.tile.row_sum(folded1_inline2032__tile, tmp_tile)
        t__rm_a0_tmp_v0: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_42, pl.const(40960, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(square_sum_inline2033__tile, [1, 16])
        t__row_major_tmp_v1: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_16, pl.const(8192, pl.INT64), 64), pl.Mem.Vec] = pl.tile.muls(t__rm_a0_tmp_v0, 0.001953125)
        t__tile_7: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_16, pl.const(8192, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(t__row_major_tmp_v1, [16, 1])
        variance_inline2034__rm_a0_tmp_v2: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_16, pl.const(8192, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(t__tile_7, [1, 16])
        variance_inline2034__row_major_tmp_v3: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_16, pl.const(8192, pl.INT64), 64), pl.Mem.Vec] = pl.tile.adds(
            variance_inline2034__rm_a0_tmp_v2, 9.9999999999999995e-07
        )
        variance_inline2034__tile: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_16, pl.const(8192, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(variance_inline2034__row_major_tmp_v3, [16, 1])
        rms_inline2050__rm_a0_tmp_v4: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_16, pl.const(8192, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(variance_inline2034__tile, [1, 16])
        rms_inline2050__row_major_tmp_v5: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_42, pl.const(40960, pl.INT64), 64), pl.Mem.Vec] = pl.tile.sqrt(rms_inline2050__rm_a0_tmp_v4)
        rms_inline2050__tile: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_42, pl.const(40960, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(rms_inline2050__row_major_tmp_v5, [16, 1])
        for k0_inline1996__idx_v0, (normed_kv_inline2015__iter_v1,) in pl.range(0, 448, 64, init_values=(normed_kv_inline2015__ssa_v0,)):
            kv_norm_chunk_inline2002__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_16, pl.const(8192, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.load(
                pooled_kv_inline205__rv_v2, [b0_inline1995__ssa_v0, k0_inline1996__idx_v0], [16, 64], [16, 64], target_memory=pl.Mem.Vec
            )
            gamma_inline2014__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_17, pl.const(24576, pl.INT64), 256), pl.Mem.Vec] = pl.tile.load(
                norm_w_2d_inline1999__ssa_v0, [0, k0_inline1996__idx_v0], [1, 64], [1, 64], target_memory=pl.Mem.Vec
            )
            t__tile_8: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_16, pl.const(8192, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.row_expand_div(kv_norm_chunk_inline2002__tile, rms_inline2050__tile)
            normed_chunk_inline2036__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_16, pl.const(8192, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.col_expand_mul(t__tile_8, gamma_inline2014__tile)
            normed_kv_inline2015__tile: pl.Tensor[[384, 512], pl.FP32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 786432)] = pl.tile.store(
                normed_chunk_inline2036__tile, [b0_inline1995__ssa_v0, k0_inline1996__idx_v0], normed_kv_inline2015__iter_v1
            )
            normed_kv_inline2015__rv_v2: pl.Tensor[[384, 512], pl.FP32, pl.MemRef("mem_ddr_32", pl.const(0, pl.INT64), 786432)] = pl.yield_(normed_kv_inline2015__tile)
        kv_rope_norm_inline2003__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_16, pl.const(8192, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.load(
            pooled_kv_inline205__rv_v2, [b0_inline1995__ssa_v0, 448], [16, 64], [16, 64], target_memory=pl.Mem.Vec
        )
        gamma_rope_inline1993__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_17, pl.const(24576, pl.INT64), 256), pl.Mem.Vec] = pl.tile.load(
            norm_w_2d_inline1999__ssa_v0, [0, 448], [1, 64], [1, 64], target_memory=pl.Mem.Vec
        )
        t__tile_9: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_16, pl.const(8192, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.row_expand_div(kv_rope_norm_inline2003__tile, rms_inline2050__tile)
        rope_normed_inline2038__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_16, pl.const(8192, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.col_expand_mul(t__tile_9, gamma_rope_inline1993__tile)
        rope_ones_inline2040__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_17, pl.const(24576, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.full([16, 64], dtype=pl.FP32, value=1.0)
        rope_index_inline2041__ci_tmp_v0: pl.Tile[[1, 192], pl.FP32, pl.MemRef(mem_vec_42, pl.const(40960, pl.INT64), 768), pl.Mem.Vec] = pl.tile.create(
            [1, 192], dtype=pl.FP32, target_memory=pl.Mem.Vec
        )
        rope_index_inline2041__tile: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_48, pl.const(45056, pl.INT64), 256), pl.Mem.Vec] = pl.tile.ci(
            pl.const(0, pl.INT32), [1, 64], tmp=rope_index_inline2041__ci_tmp_v0, dtype=pl.INT32, descending=False
        )
        rope_index_f_inline2043__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_42, pl.const(40960, pl.INT64), 256), pl.Mem.Vec] = pl.tile.cast(
            rope_index_inline2041__tile, target_type=pl.FP32, mode="round"
        )
        rope_col_inline2046__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_17, pl.const(24576, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.col_expand_mul(
            rope_ones_inline2040__tile, rope_index_f_inline2043__tile
        )
        t__tile_10: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_42, pl.const(40960, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.muls(rope_col_inline2046__tile, 0.5)
        t__tile_11: pl.Tile[[16, 64], pl.INT32, pl.MemRef(mem_vec_42, pl.const(40960, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(t__tile_10, target_type=pl.INT32, mode="trunc")
        rope_dup_f_inline2045__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_42, pl.const(40960, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(t__tile_11, target_type=pl.FP32, mode="round")
        t__tile_12: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_42, pl.const(40960, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.muls(rope_dup_f_inline2045__tile, 2.0)
        rope_lane_inline2004__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_42, pl.const(40960, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.sub(rope_col_inline2046__tile, t__tile_12)
        t__tile_13: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_17, pl.const(24576, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.adds(rope_col_inline2046__tile, 1.0)
        t__tile_14: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_48, pl.const(45056, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.muls(rope_lane_inline2004__tile, 2.0)
        t__tile_15: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_17, pl.const(24576, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.sub(t__tile_13, t__tile_14)
        rope_swap_idx_inline2048__tile: pl.Tile[[16, 64], pl.INT32, pl.MemRef(mem_vec_17, pl.const(24576, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(t__tile_15, target_type=pl.INT32, mode="round")
        gather_acc_init: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_48, pl.const(45056, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.create([16, 64], dtype=pl.FP32, target_memory=pl.Mem.Vec)
        for gather_lv, (gather_ia,) in pl.range(16, init_values=(gather_acc_init,)):
            gather_inp_row: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_16, pl.const(8192, pl.INT64), 256), pl.Mem.Vec] = pl.tile.slice(rope_normed_inline2038__tile, [1, 64], [gather_lv, 0], [1, 64])
            gather_idx_row: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_17, pl.const(24576, pl.INT64), 256), pl.Mem.Vec] = pl.tile.slice(
                rope_swap_idx_inline2048__tile, [1, 64], [gather_lv, 0], [1, 64]
            )
            gather_row_tmp: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_52, pl.const(49152, pl.INT64), 256), pl.Mem.Vec] = pl.tile.create([1, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec)
            gather_row: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_53, pl.const(49408, pl.INT64), 256), pl.Mem.Vec] = pl.tile.gather(gather_inp_row, gather_idx_row, gather_row_tmp)
            gather_asmbl: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_48, pl.const(45056, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.assemble(gather_ia, gather_row, [gather_lv, 0])
            gather_rv: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_48, pl.const(45056, pl.INT64), 4096), pl.Mem.Vec] = pl.yield_(gather_asmbl)
        swapped_inline2049__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_48, pl.const(45056, pl.INT64), 4096), pl.Mem.Vec] = gather_rv
        t__tile_16: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_17, pl.const(24576, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.muls(rope_lane_inline2004__tile, 2.0)
        t__tile_17: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_17, pl.const(24576, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.subs(t__tile_16, 1.0)
        sin_signed_inline2051__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_17, pl.const(24576, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.mul(sin_b_inline2027__ssa_v0, t__tile_17)
        t__tile_18: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_16, pl.const(8192, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.mul(rope_normed_inline2038__tile, cos_b_inline2012__ssa_v0)
        t__tile_19: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_17, pl.const(24576, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.mul(swapped_inline2049__tile, sin_signed_inline2051__tile)
        rope_rot_inline1992__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_16, pl.const(8192, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.add(t__tile_18, t__tile_19)
        normed_kv_inline2015__tile_1: pl.Tensor[[384, 512], pl.FP32, pl.MemRef("mem_ddr_32", pl.const(0, pl.INT64), 786432)] = pl.tile.store(
            rope_rot_inline1992__tile, [b0_inline1995__ssa_v0, 448], normed_kv_inline2015__rv_v2
        )
        for inner_inline1991__idx_v0, (cmp_kv_cache_flat_inline2016__iter_v1, kv_flat_inline2020__iter_v1) in pl.range(
            rms_blk_rows_inline2017__ssa_v0, init_values=(cmp_kv_cache_flat_inline2016__ssa_v0, kv_flat_inline2020__ssa_v0)
        ):
            token_inline2010__ssa_v1: pl.Scalar[pl.INDEX] = b0_inline1995__ssa_v0 + inner_inline1991__idx_v0
            token_pos_inline2007__tile: pl.Scalar[pl.INT64] = pl.tensor.read(position_ids__ssa_v0, [token_inline2010__ssa_v1])
            if (token_pos_inline2007__tile + 1) % 4 == 0:
                t__tile_20: pl.Scalar[pl.INT32] = pl.tensor.read(cmp_row_offsets__ssa_v0, [token_inline2010__ssa_v1 // 6])
                metadata_row_inline1990__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(t__tile_20, pl.INDEX) + pl.cast((token_pos_inline2007__tile + 1) // 4, pl.INDEX)
                cache_page_inline1989__tile: pl.Scalar[pl.INT32] = pl.tensor.read(cmp_slot_mapping__ssa_v0, [metadata_row_inline1990__ssa_v0, 0])
                cache_offset_inline1988__tile: pl.Scalar[pl.INT32] = pl.tensor.read(cmp_slot_mapping__ssa_v0, [metadata_row_inline1990__ssa_v0, 1])
                if 0 <= pl.cast(cache_page_inline1989__tile, pl.INDEX) and 0 <= pl.cast(cache_offset_inline1988__tile, pl.INDEX):
                    cache_row_inline2042__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(cache_page_inline1989__tile, pl.INDEX) * 32 + pl.cast(cache_offset_inline1988__tile, pl.INDEX)
                    kv_row_fp32_inline1987__tile: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_16, pl.const(8192, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.load(
                        normed_kv_inline2015__tile_1, [token_inline2010__ssa_v1, 0], [1, 512], [1, 512], target_memory=pl.Mem.Vec
                    )
                    kv_flat_inline2020__tile: pl.Tensor[[T_DYN, 512], pl.FP32, pl.MemRef("mem_ddr_8", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                        kv_row_fp32_inline1987__tile, [token_inline2010__ssa_v1, 0], kv_flat_inline2020__iter_v1
                    )
                    t__tile_21: pl.Tile[[1, 512], pl.BF16, pl.MemRef(mem_vec_16, pl.const(8192, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.cast(
                        kv_row_fp32_inline1987__tile, target_type=pl.BF16, mode="rint"
                    )
                    cmp_kv_cache_flat_inline2016__tile: pl.Tensor[[cmp_block_num_inline2008__ssa_v0 * pl.const(32, pl.INDEX), 512], pl.BF16, pl.MemRef("mem_ddr_7", pl.const(0, pl.INT64), 0)] = (
                        pl.tile.store(t__tile_21, [cache_row_inline2042__ssa_v0, 0], cmp_kv_cache_flat_inline2016__iter_v1)
                    )
                    cmp_kv_cache_flat_inline2016__phi_v4, kv_flat_inline2020__phi_v4 = pl.yield_(cmp_kv_cache_flat_inline2016__tile, kv_flat_inline2020__tile)
                else:
                    cmp_kv_cache_flat_inline2016__phi_v4, kv_flat_inline2020__phi_v4 = pl.yield_(cmp_kv_cache_flat_inline2016__iter_v1, kv_flat_inline2020__iter_v1)
                cmp_kv_cache_flat_inline2016__phi_v5, kv_flat_inline2020__phi_v5 = pl.yield_(cmp_kv_cache_flat_inline2016__phi_v4, kv_flat_inline2020__phi_v4)
            else:
                cmp_kv_cache_flat_inline2016__phi_v5, kv_flat_inline2020__phi_v5 = pl.yield_(cmp_kv_cache_flat_inline2016__iter_v1, kv_flat_inline2020__iter_v1)
            cmp_kv_cache_flat_inline2016__rv_v2, kv_flat_inline2020__rv_v2 = pl.yield_(cmp_kv_cache_flat_inline2016__phi_v5, kv_flat_inline2020__phi_v5)
        return

    @pl.function(type=pl.FunctionType.Spmd)
    def rmsnorm_rope_cache_write_spmd(
        self,
        bs_inline2047__ssa_v0: pl.Scalar[pl.INDEX],
        position_ids__ssa_v0: pl.Tensor[[T_DYN], pl.INT64, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)],
        cmp_row_offsets__ssa_v0: pl.Tensor[[B_DYN], pl.INT32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)],
        cmp_freqs_cos__ssa_v0: pl.Tensor[[COMPRESSED_ROWS_DYN, 64], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)],
        cmp_freqs_sin__ssa_v0: pl.Tensor[[COMPRESSED_ROWS_DYN, 64], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)],
        pooled_kv_inline205__rv_v2: pl.Tensor[[384, 512], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 786432)],
        normed_kv_inline2015__ssa_v0: pl.InOut[pl.Tensor[[384, 512], pl.FP32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 786432)]],
        norm_w_2d_inline1999__ssa_v0: pl.Tensor[[1, 512], pl.FP32, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 2048)],
        cmp_kv_cache_flat_inline2016__ssa_v0: pl.Out[pl.Tensor[[cmp_block_num_inline2008__ssa_v0 * pl.const(32, pl.INDEX), 512], pl.BF16, pl.MemRef("mem_ddr_7", pl.const(0, pl.INT64), 0)]],
        kv_flat_inline2020__ssa_v0: pl.Out[pl.Tensor[[T_DYN, 512], pl.FP32, pl.MemRef("mem_ddr_8", pl.const(0, pl.INT64), 0)]],
        cmp_slot_mapping__ssa_v0: pl.Tensor[[COMPRESSED_ROWS_DYN, 2], pl.INT32, pl.MemRef("mem_ddr_9", pl.const(0, pl.INT64), 0)],
    ):
        pl.func_attr({"mx_tensor_views_blocked": True})
        self.rmsnorm_rope_cache_write(
            bs_inline2047__ssa_v0,
            position_ids__ssa_v0,
            cmp_row_offsets__ssa_v0,
            cmp_freqs_cos__ssa_v0,
            cmp_freqs_sin__ssa_v0,
            pooled_kv_inline205__rv_v2,
            normed_kv_inline2015__ssa_v0,
            norm_w_2d_inline1999__ssa_v0,
            cmp_kv_cache_flat_inline2016__ssa_v0,
            kv_flat_inline2020__ssa_v0,
            cmp_slot_mapping__ssa_v0,
            attrs={
                "arg_directions": [
                    pl.adir.scalar,
                    pl.adir.input,
                    pl.adir.input,
                    pl.adir.input,
                    pl.adir.input,
                    pl.adir.input,
                    pl.adir.inout,
                    pl.adir.input,
                    pl.adir.output_existing,
                    pl.adir.output_existing,
                    pl.adir.input,
                ]
            },
        )

    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def rmsnorm_rope(
        normed_kv_inline212__ssa_v1: pl.Out[pl.Tensor[[384, 128], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 98304)]],
        rms_blocks_inline427_inline2130__ssa_v0: pl.Scalar[pl.INDEX],
        rms_workers_inline472_inline2070__ssa_v0: pl.Scalar[pl.INDEX],
        bs_inline417_inline2091__ssa_v0: pl.Scalar[pl.INDEX],
        position_ids__ssa_v0: pl.Tensor[[T_DYN], pl.INT64, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)],
        idx_row_offsets__ssa_v0: pl.Tensor[[B_DYN], pl.INT32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)],
        inner_freqs_cos__ssa_v0: pl.Tensor[[INDEXER_ROWS_DYN, 64], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)],
        inner_freqs_sin__ssa_v0: pl.Tensor[[INDEXER_ROWS_DYN, 64], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 0)],
        pooled_kv_inline2143__rv_v2: pl.Tensor[[384, 128], pl.FP32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 196608)],
        norm_w_2d_inline470_inline2095__ssa_v0: pl.Tensor[[1, 128], pl.FP32, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 512)],
    ) -> pl.Tensor[[384, 128], pl.BF16]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        mem_vec_12: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 4096)
        mem_vec_18: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 4096)
        mem_vec_21: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 4096)
        mem_vec_22: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 4096)
        mem_vec_28: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 4096)
        mem_vec_32: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 8192)
        mem_vec_41: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 2048)
        mem_vec_48: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        mem_vec_50: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 4096)
        rms_worker_inline461_inline2069__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        rope_ones_inline442_inline2087__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_32, pl.const(20480, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.full([16, 64], dtype=pl.FP32, value=1.0)
        rope_index_inline406_inline2068__ci_tmp_v0: pl.Tile[[1, 192], pl.FP32, pl.MemRef(mem_vec_12, pl.const(0, pl.INT64), 768), pl.Mem.Vec] = pl.tile.create(
            [1, 192], dtype=pl.FP32, target_memory=pl.Mem.Vec
        )
        rope_index_inline406_inline2068__tile: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_18, pl.const(4096, pl.INT64), 256), pl.Mem.Vec] = pl.tile.ci(
            pl.const(0, pl.INT32), [1, 64], tmp=rope_index_inline406_inline2068__ci_tmp_v0, dtype=pl.INT32, descending=False
        )
        rope_index_f_inline451_inline2067__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(0, pl.INT64), 256), pl.Mem.Vec] = pl.tile.cast(
            rope_index_inline406_inline2068__tile, target_type=pl.FP32, mode="round"
        )
        rope_col_inline400_inline2066__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_32, pl.const(20480, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.col_expand_mul(
            rope_ones_inline442_inline2087__tile, rope_index_f_inline451_inline2067__tile
        )
        t__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.muls(rope_col_inline400_inline2066__tile, 0.5)
        t__tile_1: pl.Tile[[16, 64], pl.INT32, pl.MemRef(mem_vec_12, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(t__tile, target_type=pl.INT32, mode="trunc")
        rope_dup_f_inline426_inline2065__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(t__tile_1, target_type=pl.FP32, mode="round")
        t__tile_2: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.muls(rope_dup_f_inline426_inline2065__tile, 2.0)
        rope_lane_inline435_inline2064__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.sub(rope_col_inline400_inline2066__tile, t__tile_2)
        t__tile_3: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_32, pl.const(20480, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.adds(rope_col_inline400_inline2066__tile, 1.0)
        t__tile_4: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_18, pl.const(4096, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.muls(rope_lane_inline435_inline2064__tile, 2.0)
        t__tile_5: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_32, pl.const(20480, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.sub(t__tile_3, t__tile_4)
        rope_swap_idx_inline399_inline2124__tile: pl.Tile[[16, 64], pl.INT32, pl.MemRef(mem_vec_18, pl.const(4096, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(
            t__tile_5, target_type=pl.INT32, mode="round"
        )
        for rms_blk_inline460_inline2063__idx_v0, (normed_kv_inline212__iter_v2,) in pl.range(
            rms_worker_inline461_inline2069__ssa_v0, rms_blocks_inline427_inline2130__ssa_v0, rms_workers_inline472_inline2070__ssa_v0, init_values=(normed_kv_inline212__ssa_v1,)
        ):
            b0_inline411_inline2062__ssa_v0: pl.Scalar[pl.INDEX] = rms_blk_inline460_inline2063__idx_v0 * 16
            rms_blk_rows_inline419_inline2139__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(bs_inline417_inline2091__ssa_v0 - b0_inline411_inline2062__ssa_v0, 16)
            cosine_inline3187__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_21, pl.const(8192, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.full([16, 64], dtype=pl.FP32, value=1.0)
            sine_inline3186__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_22, pl.const(12288, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.full([16, 64], dtype=pl.FP32, value=0.0)
            for row_inline3185__idx_v0, (cosine_inline3187__iter_v1, sine_inline3186__iter_v1) in pl.range(
                rms_blk_rows_inline419_inline2139__ssa_v0, init_values=(cosine_inline3187__tile, sine_inline3186__tile)
            ):
                token_inline3184__ssa_v0: pl.Scalar[pl.INDEX] = b0_inline411_inline2062__ssa_v0 + row_inline3185__idx_v0
                position_inline3183__tile: pl.Scalar[pl.INT64] = pl.tensor.read(position_ids__ssa_v0, [token_inline3184__ssa_v0])
                if (position_inline3183__tile + 1) % 4 == 0:
                    t__tile_6: pl.Scalar[pl.INT32] = pl.tensor.read(idx_row_offsets__ssa_v0, [token_inline3184__ssa_v0 // 6])
                    compact_row_inline3182__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(t__tile_6, pl.INDEX) + pl.cast((position_inline3183__tile + 1) // 4, pl.INDEX)
                    cosine_inline3187__tile_1: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_21, pl.const(8192, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.gather_row(
                        cosine_inline3187__iter_v1, inner_freqs_cos__ssa_v0, [row_inline3185__idx_v0, 0], [compact_row_inline3182__ssa_v0, 0], [1, 64], transpose=False
                    )
                    sine_inline3186__tile_1: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_22, pl.const(12288, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.gather_row(
                        sine_inline3186__iter_v1, inner_freqs_sin__ssa_v0, [row_inline3185__idx_v0, 0], [compact_row_inline3182__ssa_v0, 0], [1, 64], transpose=False
                    )
                    cosine_inline3187__phi_v4, sine_inline3186__phi_v4 = pl.yield_(cosine_inline3187__tile_1, sine_inline3186__tile_1)
                else:
                    cosine_inline3187__phi_v4, sine_inline3186__phi_v4 = pl.yield_(cosine_inline3187__iter_v1, sine_inline3186__iter_v1)
                cosine_inline3187__rv_v2, sine_inline3186__rv_v2 = pl.yield_(cosine_inline3187__phi_v4, sine_inline3186__phi_v4)
            cos_b_inline398_inline2134__ssa_v0: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_21, pl.const(8192, pl.INT64), 4096), pl.Mem.Vec] = cosine_inline3187__rv_v2
            sin_b_inline397_inline2151__ssa_v0: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_22, pl.const(12288, pl.INT64), 4096), pl.Mem.Vec] = sine_inline3186__rv_v2
            kv_rms_low_inline396_inline2142__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_32, pl.const(20480, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.load(
                pooled_kv_inline2143__rv_v2, [b0_inline411_inline2062__ssa_v0, 0], [16, 64], [16, 64], target_memory=pl.Mem.Vec
            )
            kv_rms_high_inline425_inline2098__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_28, pl.const(16384, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.load(
                pooled_kv_inline2143__rv_v2, [b0_inline411_inline2062__ssa_v0, 64], [16, 64], [16, 64], target_memory=pl.Mem.Vec
            )
            t__tile_7: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_32, pl.const(20480, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.mul(
                kv_rms_low_inline396_inline2142__tile, kv_rms_low_inline396_inline2142__tile
            )
            t__tile_8: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_28, pl.const(16384, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.mul(
                kv_rms_high_inline425_inline2098__tile, kv_rms_high_inline425_inline2098__tile
            )
            folded_sq_inline405_inline2122__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_28, pl.const(16384, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.add(t__tile_7, t__tile_8)
            tmp_tile: pl.Tile[[16, 128], pl.FP32, pl.MemRef(mem_vec_32, pl.const(20480, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.create([16, 128], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            square_sum_inline395_inline2109__tile: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_50, pl.const(30976, pl.INT64), 64), pl.Mem.Vec] = pl.tile.row_sum(
                folded_sq_inline405_inline2122__tile, tmp_tile
            )
            t__rm_a0_tmp_v0: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_50, pl.const(30976, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(square_sum_inline395_inline2109__tile, [1, 16])
            t__row_major_tmp_v1: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_32, pl.const(20480, pl.INT64), 64), pl.Mem.Vec] = pl.tile.muls(t__rm_a0_tmp_v0, 0.0078125)
            t__tile_9: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_32, pl.const(20480, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(t__row_major_tmp_v1, [16, 1])
            variance_inline459_inline2138__rm_a0_tmp_v2: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_32, pl.const(20480, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(t__tile_9, [1, 16])
            variance_inline459_inline2138__row_major_tmp_v3: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_32, pl.const(20480, pl.INT64), 64), pl.Mem.Vec] = pl.tile.adds(
                variance_inline459_inline2138__rm_a0_tmp_v2, 9.9999999999999995e-07
            )
            variance_inline459_inline2138__tile: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_32, pl.const(20480, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(
                variance_inline459_inline2138__row_major_tmp_v3, [16, 1]
            )
            rms_inline394_inline2126__rm_a0_tmp_v4: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_32, pl.const(20480, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(
                variance_inline459_inline2138__tile, [1, 16]
            )
            rms_inline394_inline2126__row_major_tmp_v5: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_50, pl.const(30976, pl.INT64), 64), pl.Mem.Vec] = pl.tile.sqrt(
                rms_inline394_inline2126__rm_a0_tmp_v4
            )
            rms_inline394_inline2126__tile: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_50, pl.const(30976, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(
                rms_inline394_inline2126__row_major_tmp_v5, [16, 1]
            )
            kv_norm_chunk_inline449_inline2060__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_32, pl.const(20480, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.load(
                pooled_kv_inline2143__rv_v2, [b0_inline411_inline2062__ssa_v0, 0], [16, 64], [16, 64], target_memory=pl.Mem.Vec
            )
            gamma_inline463_inline2120__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_28, pl.const(16384, pl.INT64), 256), pl.Mem.Vec] = pl.tile.load(
                norm_w_2d_inline470_inline2095__ssa_v0, [0, 0], [1, 64], [1, 64], target_memory=pl.Mem.Vec
            )
            t__tile_10: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_32, pl.const(20480, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.row_expand_div(
                kv_norm_chunk_inline449_inline2060__tile, rms_inline394_inline2126__tile
            )
            normed_chunk_inline403_inline2059__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_32, pl.const(20480, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.col_expand_mul(
                t__tile_10, gamma_inline463_inline2120__tile
            )
            normed_nope_inline393_inline2094__tile: pl.Tile[[16, 64], pl.BF16, pl.MemRef(mem_vec_41, pl.const(28672, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(
                normed_chunk_inline403_inline2059__tile, target_type=pl.BF16, mode="rint"
            )
            kv_rope_norm_inline414_inline2108__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_32, pl.const(20480, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.load(
                pooled_kv_inline2143__rv_v2, [b0_inline411_inline2062__ssa_v0, 64], [16, 64], [16, 64], target_memory=pl.Mem.Vec
            )
            gamma_rope_inline392_inline2058__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_28, pl.const(16384, pl.INT64), 256), pl.Mem.Vec] = pl.tile.load(
                norm_w_2d_inline470_inline2095__ssa_v0, [0, 64], [1, 64], [1, 64], target_memory=pl.Mem.Vec
            )
            t__tile_11: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_32, pl.const(20480, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.row_expand_div(
                kv_rope_norm_inline414_inline2108__tile, rms_inline394_inline2126__tile
            )
            rope_normed_inline391_inline2057__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_32, pl.const(20480, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.col_expand_mul(
                t__tile_11, gamma_rope_inline392_inline2058__tile
            )
            gather_acc_init: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_28, pl.const(16384, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.create([16, 64], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            for gather_lv, (gather_ia,) in pl.range(16, init_values=(gather_acc_init,)):
                gather_inp_row: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_32, pl.const(20480, pl.INT64), 256), pl.Mem.Vec] = pl.tile.slice(
                    rope_normed_inline391_inline2057__tile, [1, 64], [gather_lv, 0], [1, 64]
                )
                gather_idx_row: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_18, pl.const(4096, pl.INT64), 256), pl.Mem.Vec] = pl.tile.slice(
                    rope_swap_idx_inline399_inline2124__tile, [1, 64], [gather_lv, 0], [1, 64]
                )
                gather_row_tmp: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_50, pl.const(30976, pl.INT64), 256), pl.Mem.Vec] = pl.tile.create([1, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec)
                gather_row: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_48, pl.const(30720, pl.INT64), 256), pl.Mem.Vec] = pl.tile.gather(gather_inp_row, gather_idx_row, gather_row_tmp)
                gather_asmbl: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_28, pl.const(16384, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.assemble(gather_ia, gather_row, [gather_lv, 0])
                gather_rv: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_28, pl.const(16384, pl.INT64), 4096), pl.Mem.Vec] = pl.yield_(gather_asmbl)
            swapped_inline390_inline2115__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_28, pl.const(16384, pl.INT64), 4096), pl.Mem.Vec] = gather_rv
            t__tile_12: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_50, pl.const(30976, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.muls(rope_lane_inline435_inline2064__tile, 2.0)
            t__tile_13: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_50, pl.const(30976, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.subs(t__tile_12, 1.0)
            sin_signed_inline474_inline2056__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_22, pl.const(12288, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.mul(
                sin_b_inline397_inline2151__ssa_v0, t__tile_13
            )
            t__tile_14: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_32, pl.const(20480, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.mul(
                rope_normed_inline391_inline2057__tile, cos_b_inline398_inline2134__ssa_v0
            )
            t__tile_15: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_21, pl.const(8192, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.mul(
                swapped_inline390_inline2115__tile, sin_signed_inline474_inline2056__tile
            )
            rope_rot_inline389_inline2055__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_32, pl.const(20480, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.add(t__tile_14, t__tile_15)
            normed_rope_inline388_inline2111__tile: pl.Tile[[16, 64], pl.BF16, pl.MemRef(mem_vec_32, pl.const(20480, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(
                rope_rot_inline389_inline2055__tile, target_type=pl.BF16, mode="rint"
            )
            for inner_inline473_inline2150__idx_v0, (normed_kv_inline212__iter_v4,) in pl.range(rms_blk_rows_inline419_inline2139__ssa_v0, init_values=(normed_kv_inline212__iter_v2,)):
                token_inline420_inline2157__ssa_v2: pl.Scalar[pl.INDEX] = b0_inline411_inline2062__ssa_v0 + inner_inline473_inline2150__idx_v0
                token_pos_inline431_inline2148__tile: pl.Scalar[pl.INT64] = pl.tensor.read(position_ids__ssa_v0, [token_inline420_inline2157__ssa_v2])
                if (token_pos_inline431_inline2148__tile + 1) % 4 == 0:
                    request_inline456_inline2149__ssa_v0: pl.Scalar[pl.INDEX] = token_inline420_inline2157__ssa_v2 // 6
                    first_pos_inline409_inline2054__tile: pl.Scalar[pl.INT64] = pl.tensor.read(position_ids__ssa_v0, [request_inline456_inline2149__ssa_v0 * 6])
                    first_boundary_inline386_inline2053__ssa_v0: pl.Scalar[pl.INDEX] = 3 - first_pos_inline409_inline2054__tile % 4
                    compact_token_inline387_inline2052__ssa_v0: pl.Scalar[pl.INDEX] = (
                        request_inline456_inline2149__ssa_v0 * 2 + (token_inline420_inline2157__ssa_v2 % 6 - first_boundary_inline386_inline2053__ssa_v0) // 4
                    )
                    t__tile_16: pl.Tile[[1, 64], pl.BF16, pl.MemRef(mem_vec_41, pl.const(28672, pl.INT64), 128), pl.Mem.Vec] = pl.tile.slice(
                        normed_nope_inline393_inline2094__tile, [1, 64], [inner_inline473_inline2150__idx_v0, 0]
                    )
                    normed_kv_inline212__tile: pl.Tensor[[384, 128], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 98304)] = pl.tile.store(
                        t__tile_16, [compact_token_inline387_inline2052__ssa_v0, 0], normed_kv_inline212__iter_v4
                    )
                    t__tile_17: pl.Tile[[1, 64], pl.BF16, pl.MemRef(mem_vec_32, pl.const(20480, pl.INT64), 128), pl.Mem.Vec] = pl.tile.slice(
                        normed_rope_inline388_inline2111__tile, [1, 64], [inner_inline473_inline2150__idx_v0, 0]
                    )
                    normed_kv_inline212__tile_1: pl.Tensor[[384, 128], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 98304)] = pl.tile.store(
                        t__tile_17, [compact_token_inline387_inline2052__ssa_v0, 64], normed_kv_inline212__tile
                    )
                    normed_kv_inline212__phi_v8: pl.Tensor[[384, 128], pl.BF16, pl.MemRef("mem_ddr_57", pl.const(0, pl.INT64), 98304)] = pl.yield_(normed_kv_inline212__tile_1)
                else:
                    normed_kv_inline212__phi_v8: pl.Tensor[[384, 128], pl.BF16, pl.MemRef("mem_ddr_57", pl.const(0, pl.INT64), 98304)] = pl.yield_(normed_kv_inline212__iter_v4)
                normed_kv_inline212__rv_v5: pl.Tensor[[384, 128], pl.BF16, pl.MemRef("mem_ddr_58", pl.const(0, pl.INT64), 98304)] = pl.yield_(normed_kv_inline212__phi_v8)
            normed_kv_inline212__rv_v3: pl.Tensor[[384, 128], pl.BF16, pl.MemRef("mem_ddr_59", pl.const(0, pl.INT64), 98304)] = pl.yield_(normed_kv_inline212__rv_v5)
        return normed_kv_inline212__ssa_v1

    @pl.function(type=pl.FunctionType.Spmd)
    def rmsnorm_rope_spmd(
        self,
        normed_kv_inline212__ssa_v1: pl.Out[pl.Tensor[[384, 128], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 98304)]],
        rms_blocks_inline427_inline2130__ssa_v0: pl.Scalar[pl.INDEX],
        rms_workers_inline472_inline2070__ssa_v0: pl.Scalar[pl.INDEX],
        bs_inline417_inline2091__ssa_v0: pl.Scalar[pl.INDEX],
        position_ids__ssa_v0: pl.Tensor[[T_DYN], pl.INT64, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)],
        idx_row_offsets__ssa_v0: pl.Tensor[[B_DYN], pl.INT32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)],
        inner_freqs_cos__ssa_v0: pl.Tensor[[INDEXER_ROWS_DYN, 64], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)],
        inner_freqs_sin__ssa_v0: pl.Tensor[[INDEXER_ROWS_DYN, 64], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 0)],
        pooled_kv_inline2143__rv_v2: pl.Tensor[[384, 128], pl.FP32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 196608)],
        norm_w_2d_inline470_inline2095__ssa_v0: pl.Tensor[[1, 128], pl.FP32, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 512)],
    ) -> pl.Tensor[[384, 128], pl.BF16]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        normed_kv_inline212__rv_v3: pl.Tensor[[384, 128], pl.BF16, pl.MemRef("mem_ddr_7", pl.const(0, pl.INT64), 98304)] = self.rmsnorm_rope(
            normed_kv_inline212__ssa_v1,
            rms_blocks_inline427_inline2130__ssa_v0,
            rms_workers_inline472_inline2070__ssa_v0,
            bs_inline417_inline2091__ssa_v0,
            position_ids__ssa_v0,
            idx_row_offsets__ssa_v0,
            inner_freqs_cos__ssa_v0,
            inner_freqs_sin__ssa_v0,
            pooled_kv_inline2143__rv_v2,
            norm_w_2d_inline470_inline2095__ssa_v0,
            attrs={
                "arg_directions": [pl.adir.output_existing, pl.adir.scalar, pl.adir.scalar, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input]
            },
        )
        return normed_kv_inline212__ssa_v1

    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def rope_cs(
        rope_swap_idx_inline2414__ssa_v0: pl.Out[pl.Tensor[[16, 64], pl.INT32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 4096)]],
        rope_sin_signed_inline2415__ssa_v0: pl.Out[pl.Tensor[[384, 64], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 98304)]],
        rope_cs_blocks_inline2473__ssa_v0: pl.Scalar[pl.INDEX],
        freqs_sin__ssa_v0: pl.Tensor[[T_DYN, 64], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)],
    ) -> tuple[pl.Tensor[[16, 64], pl.INT32], pl.Tensor[[384, 64], pl.FP32]]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        mem_vec_3: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 4096)
        mem_vec_5: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        mem_vec_8: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 4096)
        sw_ones_inline2412__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_3, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.full([16, 64], dtype=pl.FP32, value=1.0)
        t__ci_tmp_v0: pl.Tile[[1, 192], pl.FP32, pl.MemRef(mem_vec_8, pl.const(4352, pl.INT64), 768), pl.Mem.Vec] = pl.tile.create([1, 192], dtype=pl.FP32, target_memory=pl.Mem.Vec)
        t__tile: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_5, pl.const(4096, pl.INT64), 256), pl.Mem.Vec] = pl.tile.ci(
            pl.const(0, pl.INT32), [1, 64], tmp=t__ci_tmp_v0, dtype=pl.INT32, descending=False
        )
        sw_idx_f_inline2411__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_8, pl.const(4352, pl.INT64), 256), pl.Mem.Vec] = pl.tile.cast(t__tile, target_type=pl.FP32, mode="round")
        sw_col_inline2451__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_3, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.col_expand_mul(sw_ones_inline2412__tile, sw_idx_f_inline2411__tile)
        t__tile_1: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_8, pl.const(4352, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.muls(sw_col_inline2451__tile, 0.5)
        sw_dup_i32_inline2510__tile: pl.Tile[[16, 64], pl.INT32, pl.MemRef(mem_vec_8, pl.const(4352, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(t__tile_1, target_type=pl.INT32, mode="trunc")
        sw_dup_f_inline2410__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_8, pl.const(4352, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(
            sw_dup_i32_inline2510__tile, target_type=pl.FP32, mode="round"
        )
        t__tile_2: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_8, pl.const(4352, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.muls(sw_dup_f_inline2410__tile, 2.0)
        sw_lane_inline2409__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_8, pl.const(4352, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.sub(sw_col_inline2451__tile, t__tile_2)
        t__tile_3: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_3, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.adds(sw_col_inline2451__tile, 1.0)
        t__tile_4: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_8, pl.const(4352, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.muls(sw_lane_inline2409__tile, 2.0)
        sw_swap_f_inline2449__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_3, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.sub(t__tile_3, t__tile_4)
        t__tile_5: pl.Tile[[16, 64], pl.INT32, pl.MemRef(mem_vec_3, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(sw_swap_f_inline2449__tile, target_type=pl.INT32, mode="round")
        rope_swap_idx_inline2414__tile: pl.Tensor[[16, 64], pl.INT32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 4096)] = pl.tile.store(t__tile_5, [0, 0], rope_swap_idx_inline2414__ssa_v0)
        cs_ones_inline2408__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_3, pl.const(0, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.full([8, 64], dtype=pl.FP32, value=1.0)
        t__ci_tmp_v1: pl.Tile[[1, 192], pl.FP32, pl.MemRef(mem_vec_8, pl.const(4352, pl.INT64), 768), pl.Mem.Vec] = pl.tile.create([1, 192], dtype=pl.FP32, target_memory=pl.Mem.Vec)
        t__tile_6: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_5, pl.const(4096, pl.INT64), 256), pl.Mem.Vec] = pl.tile.ci(
            pl.const(0, pl.INT32), [1, 64], tmp=t__ci_tmp_v1, dtype=pl.INT32, descending=False
        )
        cs_idx_f_inline2545__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_8, pl.const(4352, pl.INT64), 256), pl.Mem.Vec] = pl.tile.cast(t__tile_6, target_type=pl.FP32, mode="round")
        cs_col_inline2407__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_3, pl.const(0, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.col_expand_mul(cs_ones_inline2408__tile, cs_idx_f_inline2545__tile)
        t__tile_7: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_8, pl.const(4352, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.muls(cs_col_inline2407__tile, 0.5)
        cs_dup_i32_inline2417__tile: pl.Tile[[8, 64], pl.INT32, pl.MemRef(mem_vec_8, pl.const(4352, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(t__tile_7, target_type=pl.INT32, mode="trunc")
        cs_dup_f_inline2406__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_8, pl.const(4352, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(
            cs_dup_i32_inline2417__tile, target_type=pl.FP32, mode="round"
        )
        t__tile_8: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_8, pl.const(4352, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.muls(cs_dup_f_inline2406__tile, 2.0)
        cs_lane_inline2540__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_3, pl.const(0, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.sub(cs_col_inline2407__tile, t__tile_8)
        t__tile_9: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_3, pl.const(0, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.muls(cs_lane_inline2540__tile, 2.0)
        t__tile_10: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_3, pl.const(0, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.subs(t__tile_9, 1.0)
        cs_sign_inline2423__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_3, pl.const(0, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.neg(t__tile_10)
        for cs_rb_inline2523__idx_v0, (rope_sin_signed_inline2415__iter_v1,) in pl.range(rope_cs_blocks_inline2473__ssa_v0, init_values=(rope_sin_signed_inline2415__ssa_v0,)):
            cs_t0_inline2405__ssa_v0: pl.Scalar[pl.INDEX] = cs_rb_inline2523__idx_v0 * 8
            cs_sin_inline2404__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_8, pl.const(4352, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.load(
                freqs_sin__ssa_v0, [cs_t0_inline2405__ssa_v0, 0], [8, 64], [8, 64], target_memory=pl.Mem.Vec
            )
            t__tile_11: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_8, pl.const(4352, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.mul(cs_sin_inline2404__tile, cs_sign_inline2423__tile)
            rope_sin_signed_inline2415__tile: pl.Tensor[[384, 64], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 98304)] = pl.tile.store(
                t__tile_11, [cs_t0_inline2405__ssa_v0, 0], rope_sin_signed_inline2415__iter_v1
            )
            rope_sin_signed_inline2415__rv_v2: pl.Tensor[[384, 64], pl.FP32, pl.MemRef("mem_ddr_32", pl.const(0, pl.INT64), 98304)] = pl.yield_(rope_sin_signed_inline2415__tile)
        return rope_swap_idx_inline2414__ssa_v0, rope_sin_signed_inline2415__ssa_v0

    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def scatter_softmax_pool(
        pooled_kv_inline205__ssa_v0: pl.Out[pl.Tensor[[384, 512], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 786432)]],
        b_dim_inline518_inline1939__ssa_v0: pl.Scalar[pl.INDEX],
        pool_workers_inline512_inline1960__ssa_v0: pl.Scalar[pl.INDEX],
        position_ids__ssa_v0: pl.Tensor[[T_DYN], pl.INT64, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)],
        s_dim_inline513_inline1935__ssa_v0: pl.Scalar[pl.INDEX],
        state_block_table__ssa_v0: pl.Tensor[[B_DYN, STATE_TABLE_COLUMNS_DYN], pl.INT32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)],
        compress_state__ssa_v0: pl.Tensor[[MAIN_STATE_BLOCK_NUM_DYN, STATE_PAGE_ELEMENTS_DYN], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)],
        cmp4_kv_proj_pad_inline530_inline1957__ssa_v0: pl.Tensor[[384, 1024], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 1572864)],
        cmp4_score_proj_pad_inline510_inline1929__ssa_v0: pl.Tensor[[384, 1024], pl.FP32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 1572864)],
        cmp_ape__ssa_v0: pl.Tensor[[4, 1024], pl.FP32, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 16384)],
    ) -> tuple[pl.Tensor[[384, 512], pl.FP32], pl.Tensor[[384, 512], pl.FP32]]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        mem_vec_8: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
        mem_vec_9: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
        mem_vec_11: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 2048)
        mem_vec_12: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 2048)
        mem_vec_13: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 2048)
        mem_vec_14: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 2048)
        mem_vec_29: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 8192)
        pool_worker_inline520_inline1930__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        for c_idx_inline508_inline1938__idx_v0, (pooled_kv_inline205__iter_v1,) in pl.range(
            pool_worker_inline520_inline1930__ssa_v0, b_dim_inline518_inline1939__ssa_v0, pool_workers_inline512_inline1960__ssa_v0, init_values=(pooled_kv_inline205__ssa_v0,)
        ):
            first_pos_b_inline529_inline1965__tile: pl.Scalar[pl.INT64] = pl.tensor.read(position_ids__ssa_v0, [c_idx_inline508_inline1938__idx_v0 * s_dim_inline513_inline1935__ssa_v0])
            for s_idx_inline506_inline1967__idx_v0, (pooled_kv_inline205__iter_v3,) in pl.range(s_dim_inline513_inline1935__ssa_v0, init_values=(pooled_kv_inline205__iter_v1,)):
                token_inline503_inline1970__ssa_v0: pl.Scalar[pl.INDEX] = c_idx_inline508_inline1938__idx_v0 * s_dim_inline513_inline1935__ssa_v0 + s_idx_inline506_inline1967__idx_v0
                token_pos_inline514_inline1959__tile: pl.Scalar[pl.INT64] = pl.tensor.read(position_ids__ssa_v0, [token_inline503_inline1970__ssa_v0])
                t__tile: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_8, pl.const(16384, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.full([1, 512], dtype=pl.FP32, value=0.0)
                pooled_kv_inline205__tile: pl.Tensor[[384, 512], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 786432)] = pl.tile.store(
                    t__tile, [token_inline503_inline1970__ssa_v0, 0], pooled_kv_inline205__iter_v3
                )
                if (token_pos_inline514_inline1959__tile + 1) % 4 == 0:
                    window_start_inline523_inline1971__ssa_v0: pl.Scalar[pl.INT64] = token_pos_inline514_inline1959__tile - 8 + 1
                    window_values_inline525_inline1953__ssa_v0: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_8, pl.const(16384, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.full(
                        [8, 512], dtype=pl.FP32, value=0.0
                    )
                    window_scores_inline526_inline1969__ssa_v0: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_9, pl.const(32768, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.full(
                        [8, 512], dtype=pl.FP32, value=-3.4028234663852886e38
                    )
                    for state_idx_inline527_inline1956__idx_v0, (window_scores_inline526_inline1969__iter_v1, window_values_inline525_inline1953__iter_v1) in pl.range(
                        8, init_values=(window_scores_inline526_inline1969__ssa_v0, window_values_inline525_inline1953__ssa_v0)
                    ):
                        logical_pos_inline515_inline1975__ssa_v0: pl.Scalar[pl.INT64] = window_start_inline523_inline1971__ssa_v0 + state_idx_inline527_inline1956__idx_v0
                        value_inline533_inline1979__ssa_v0: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_29, pl.const(8192, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.full(
                            [1, 512], dtype=pl.FP32, value=0.0
                        )
                        score_inline521_inline1977__ssa_v0: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_11, pl.const(0, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.full(
                            [1, 512], dtype=pl.FP32, value=-3.4028234663852886e38
                        )
                        state_half_inline511_inline1980__ssa_v0: pl.Scalar[pl.INDEX] = 0
                        if 4 <= state_idx_inline527_inline1956__idx_v0:
                            state_half_inline511_inline1980__ssa_v1: pl.Scalar[pl.INDEX] = 512
                            state_half_inline511_inline1980__phi_v2: pl.Scalar[pl.INDEX] = pl.yield_(state_half_inline511_inline1980__ssa_v1)
                        else:
                            state_half_inline511_inline1980__phi_v2: pl.Scalar[pl.INDEX] = pl.yield_(state_half_inline511_inline1980__ssa_v0)
                        if 0 <= logical_pos_inline515_inline1975__ssa_v0 and logical_pos_inline515_inline1975__ssa_v0 < first_pos_b_inline529_inline1965__tile:
                            score_inline521_inline1977__ssa_v1: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_12, pl.const(2048, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.full(
                                [1, 512], dtype=pl.FP32, value=0.0
                            )
                            t__tile_1: pl.Scalar[pl.INT32] = pl.tensor.read(state_block_table__ssa_v0, [c_idx_inline508_inline1938__idx_v0, logical_pos_inline515_inline1975__ssa_v0 // 2])
                            history_page_inline528_inline1940__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(t__tile_1, pl.INDEX)
                            if 0 <= history_page_inline528_inline1940__ssa_v0:
                                history_column_inline531_inline1981__ssa_v0: pl.Scalar[pl.INDEX] = (
                                    pl.cast(logical_pos_inline515_inline1975__ssa_v0 % 2, pl.INDEX) * 2048 + state_half_inline511_inline1980__phi_v2
                                )
                                value_inline533_inline1979__ssa_v1: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_13, pl.const(4096, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.load(
                                    compress_state__ssa_v0, [history_page_inline528_inline1940__ssa_v0, history_column_inline531_inline1981__ssa_v0], [1, 512], [1, 512], target_memory=pl.Mem.Vec
                                )
                                score_inline521_inline1977__ssa_v2: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_14, pl.const(6144, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.load(
                                    compress_state__ssa_v0,
                                    [history_page_inline528_inline1940__ssa_v0, history_column_inline531_inline1981__ssa_v0 + 1024],
                                    [1, 512],
                                    [1, 512],
                                    target_memory=pl.Mem.Vec,
                                )
                                score_inline521_inline1977__phi_v3, value_inline533_inline1979__phi_v2 = pl.yield_(score_inline521_inline1977__ssa_v2, value_inline533_inline1979__ssa_v1)
                            else:
                                score_inline521_inline1977__ssa_v1_mv: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_14, pl.const(6144, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.move(
                                    score_inline521_inline1977__ssa_v1, target_memory=pl.Mem.Vec
                                )
                                value_inline533_inline1979__ssa_v0_mv: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_13, pl.const(4096, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.move(
                                    value_inline533_inline1979__ssa_v0, target_memory=pl.Mem.Vec
                                )
                                score_inline521_inline1977__phi_v3, value_inline533_inline1979__phi_v2 = pl.yield_(score_inline521_inline1977__ssa_v1_mv, value_inline533_inline1979__ssa_v0_mv)
                            score_inline521_inline1977__phi_v4, value_inline533_inline1979__phi_v3 = pl.yield_(score_inline521_inline1977__phi_v3, value_inline533_inline1979__phi_v2)
                        else:
                            score_inline521_inline1977__ssa_v0_mv: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_14, pl.const(6144, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.move(
                                score_inline521_inline1977__ssa_v0, target_memory=pl.Mem.Vec
                            )
                            value_inline533_inline1979__ssa_v0_mv_1: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_13, pl.const(4096, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.move(
                                value_inline533_inline1979__ssa_v0, target_memory=pl.Mem.Vec
                            )
                            score_inline521_inline1977__phi_v4, value_inline533_inline1979__phi_v3 = pl.yield_(score_inline521_inline1977__ssa_v0_mv, value_inline533_inline1979__ssa_v0_mv_1)
                        if first_pos_b_inline529_inline1965__tile <= logical_pos_inline515_inline1975__ssa_v0:
                            if logical_pos_inline515_inline1975__ssa_v0 <= token_pos_inline514_inline1959__tile:
                                overlay_token_inline504_inline1974__ssa_v0: pl.Scalar[pl.INDEX] = (
                                    c_idx_inline508_inline1938__idx_v0 * s_dim_inline513_inline1935__ssa_v0 + logical_pos_inline515_inline1975__ssa_v0 - first_pos_b_inline529_inline1965__tile
                                )
                                ape_row_inline532_inline1961__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(logical_pos_inline515_inline1975__ssa_v0 % 4, pl.INDEX)
                                value_inline533_inline1979__ssa_v4: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_29, pl.const(8192, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.load(
                                    cmp4_kv_proj_pad_inline530_inline1957__ssa_v0,
                                    [overlay_token_inline504_inline1974__ssa_v0, state_half_inline511_inline1980__phi_v2],
                                    [1, 512],
                                    [1, 512],
                                    target_memory=pl.Mem.Vec,
                                )
                                t__tmp_v115: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_11, pl.const(0, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.load(
                                    cmp4_score_proj_pad_inline510_inline1929__ssa_v0,
                                    [overlay_token_inline504_inline1974__ssa_v0, state_half_inline511_inline1980__phi_v2],
                                    [1, 512],
                                    [1, 512],
                                    target_memory=pl.Mem.Vec,
                                )
                                t__tmp_v116: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_12, pl.const(2048, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.load(
                                    cmp_ape__ssa_v0, [ape_row_inline532_inline1961__ssa_v0, state_half_inline511_inline1980__phi_v2], [1, 512], [1, 512], target_memory=pl.Mem.Vec
                                )
                                score_inline521_inline1977__ssa_v5: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_11, pl.const(0, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.add(t__tmp_v115, t__tmp_v116)
                                score_inline521_inline1977__phi_v6, value_inline533_inline1979__phi_v5 = pl.yield_(score_inline521_inline1977__ssa_v5, value_inline533_inline1979__ssa_v4)
                            else:
                                score_inline521_inline1977__phi_v4_mv: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_11, pl.const(0, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.move(
                                    score_inline521_inline1977__phi_v4, target_memory=pl.Mem.Vec
                                )
                                value_inline533_inline1979__phi_v3_mv: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_29, pl.const(8192, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.move(
                                    value_inline533_inline1979__phi_v3, target_memory=pl.Mem.Vec
                                )
                                score_inline521_inline1977__phi_v6, value_inline533_inline1979__phi_v5 = pl.yield_(score_inline521_inline1977__phi_v4_mv, value_inline533_inline1979__phi_v3_mv)
                            score_inline521_inline1977__phi_v7, value_inline533_inline1979__phi_v6 = pl.yield_(score_inline521_inline1977__phi_v6, value_inline533_inline1979__phi_v5)
                        else:
                            score_inline521_inline1977__phi_v4_mv_1: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_11, pl.const(0, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.move(
                                score_inline521_inline1977__phi_v4, target_memory=pl.Mem.Vec
                            )
                            value_inline533_inline1979__phi_v3_mv_1: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_29, pl.const(8192, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.move(
                                value_inline533_inline1979__phi_v3, target_memory=pl.Mem.Vec
                            )
                            score_inline521_inline1977__phi_v7, value_inline533_inline1979__phi_v6 = pl.yield_(score_inline521_inline1977__phi_v4_mv_1, value_inline533_inline1979__phi_v3_mv_1)
                        native_row_inline535_inline1976__ssa_v0: pl.Scalar[pl.INDEX] = state_idx_inline527_inline1956__idx_v0 % 4 * 2 + state_idx_inline527_inline1956__idx_v0 // 4
                        window_values_inline525_inline1953__ssa_v3: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_8, pl.const(16384, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.assemble(
                            window_values_inline525_inline1953__iter_v1, value_inline533_inline1979__phi_v6, [native_row_inline535_inline1976__ssa_v0, 0]
                        )
                        window_scores_inline526_inline1969__ssa_v3: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_9, pl.const(32768, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.assemble(
                            window_scores_inline526_inline1969__iter_v1, score_inline521_inline1977__phi_v7, [native_row_inline535_inline1976__ssa_v0, 0]
                        )
                        window_scores_inline526_inline1969__rv_v2, window_values_inline525_inline1953__rv_v2 = pl.yield_(
                            window_scores_inline526_inline1969__ssa_v3, window_values_inline525_inline1953__ssa_v3
                        )
                    t__tmp_v117: pl.Tile[[4, 512], pl.FP32, pl.MemRef(mem_vec_9, pl.const(32768, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.slice(
                        window_scores_inline526_inline1969__rv_v2, [4, 512], [0, 0]
                    )
                    t__tmp_v118: pl.Tile[[4, 512], pl.FP32, pl.MemRef(mem_vec_9, pl.const(40960, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.slice(
                        window_scores_inline526_inline1969__rv_v2, [4, 512], [4, 0]
                    )
                    max4_inline517_inline1983__ssa_v0: pl.Tile[[4, 512], pl.FP32, pl.MemRef(mem_vec_29, pl.const(8192, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.maximum(t__tmp_v117, t__tmp_v118)
                    t__tmp_v119: pl.Tile[[2, 512], pl.FP32, pl.MemRef(mem_vec_29, pl.const(8192, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.slice(max4_inline517_inline1983__ssa_v0, [2, 512], [0, 0])
                    t__tmp_v120: pl.Tile[[2, 512], pl.FP32, pl.MemRef(mem_vec_29, pl.const(12288, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.slice(max4_inline517_inline1983__ssa_v0, [2, 512], [2, 0])
                    max2_inline509_inline1963__ssa_v0: pl.Tile[[2, 512], pl.FP32, pl.MemRef(mem_vec_29, pl.const(8192, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.maximum(t__tmp_v119, t__tmp_v120)
                    t__tmp_v121: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_29, pl.const(8192, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.slice(max2_inline509_inline1963__ssa_v0, [1, 512], [0, 0])
                    t__tmp_v122: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_29, pl.const(10240, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.slice(max2_inline509_inline1963__ssa_v0, [1, 512], [1, 0])
                    maximum_inline502_inline1948__ssa_v0: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_29, pl.const(8192, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.maximum(t__tmp_v121, t__tmp_v122)
                    t__tmp_v123: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_9, pl.const(32768, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.col_expand_sub(
                        window_scores_inline526_inline1969__rv_v2, maximum_inline502_inline1948__ssa_v0
                    )
                    probability_inline501_inline1978__ssa_v0: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_9, pl.const(32768, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.exp(t__tmp_v123)
                    t__tmp_v124: pl.Tile[[4, 512], pl.FP32, pl.MemRef(mem_vec_9, pl.const(32768, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.slice(
                        probability_inline501_inline1978__ssa_v0, [4, 512], [0, 0]
                    )
                    t__tmp_v125: pl.Tile[[4, 512], pl.FP32, pl.MemRef(mem_vec_9, pl.const(40960, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.slice(
                        probability_inline501_inline1978__ssa_v0, [4, 512], [4, 0]
                    )
                    sum4_inline519_inline1966__ssa_v0: pl.Tile[[4, 512], pl.FP32, pl.MemRef(mem_vec_29, pl.const(8192, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.add(t__tmp_v124, t__tmp_v125)
                    t__tmp_v126: pl.Tile[[2, 512], pl.FP32, pl.MemRef(mem_vec_29, pl.const(8192, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.slice(sum4_inline519_inline1966__ssa_v0, [2, 512], [0, 0])
                    t__tmp_v127: pl.Tile[[2, 512], pl.FP32, pl.MemRef(mem_vec_29, pl.const(12288, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.slice(sum4_inline519_inline1966__ssa_v0, [2, 512], [2, 0])
                    sum2_inline500_inline1931__ssa_v0: pl.Tile[[2, 512], pl.FP32, pl.MemRef(mem_vec_29, pl.const(8192, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.add(t__tmp_v126, t__tmp_v127)
                    t__tmp_v128: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_29, pl.const(8192, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.slice(sum2_inline500_inline1931__ssa_v0, [1, 512], [0, 0])
                    t__tmp_v129: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_29, pl.const(10240, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.slice(sum2_inline500_inline1931__ssa_v0, [1, 512], [1, 0])
                    total_inline499_inline1984__ssa_v0: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_29, pl.const(8192, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.add(t__tmp_v128, t__tmp_v129)
                    probability_v1_inline507_inline1985__ssa_v0: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_9, pl.const(32768, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.col_expand_div(
                        probability_inline501_inline1978__ssa_v0, total_inline499_inline1984__ssa_v0
                    )
                    weighted_inline498_inline1986__ssa_v0: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_8, pl.const(16384, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.mul(
                        window_values_inline525_inline1953__rv_v2, probability_v1_inline507_inline1985__ssa_v0
                    )
                    t__tmp_v130: pl.Tile[[4, 512], pl.FP32, pl.MemRef(mem_vec_8, pl.const(16384, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.slice(weighted_inline498_inline1986__ssa_v0, [4, 512], [0, 0])
                    t__tmp_v131: pl.Tile[[4, 512], pl.FP32, pl.MemRef(mem_vec_8, pl.const(24576, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.slice(weighted_inline498_inline1986__ssa_v0, [4, 512], [4, 0])
                    weighted4_inline497_inline1951__ssa_v0: pl.Tile[[4, 512], pl.FP32, pl.MemRef(mem_vec_8, pl.const(16384, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.add(t__tmp_v130, t__tmp_v131)
                    t__tmp_v132: pl.Tile[[2, 512], pl.FP32, pl.MemRef(mem_vec_8, pl.const(16384, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.slice(weighted4_inline497_inline1951__ssa_v0, [2, 512], [0, 0])
                    t__tmp_v133: pl.Tile[[2, 512], pl.FP32, pl.MemRef(mem_vec_8, pl.const(20480, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.slice(weighted4_inline497_inline1951__ssa_v0, [2, 512], [2, 0])
                    weighted2_inline496_inline1968__ssa_v0: pl.Tile[[2, 512], pl.FP32, pl.MemRef(mem_vec_8, pl.const(16384, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.add(t__tmp_v132, t__tmp_v133)
                    t__tmp_v134: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_8, pl.const(16384, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.slice(weighted2_inline496_inline1968__ssa_v0, [1, 512], [0, 0])
                    t__tmp_v135: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_8, pl.const(18432, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.slice(weighted2_inline496_inline1968__ssa_v0, [1, 512], [1, 0])
                    pooled_row_inline516_inline1927__ssa_v0: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_8, pl.const(16384, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.add(t__tmp_v134, t__tmp_v135)
                    pl.tile.store(pooled_row_inline516_inline1927__ssa_v0, [token_inline503_inline1970__ssa_v0, 0], pooled_kv_inline205__tile)
                pooled_kv_inline205__rv_v4: pl.Tensor[[384, 512], pl.FP32, pl.MemRef("mem_ddr_42", pl.const(0, pl.INT64), 786432)] = pl.yield_(pooled_kv_inline205__tile)
            pooled_kv_inline205__rv_v2: pl.Tensor[[384, 512], pl.FP32, pl.MemRef("mem_ddr_43", pl.const(0, pl.INT64), 786432)] = pl.yield_(pooled_kv_inline205__rv_v4)
        return pooled_kv_inline205__ssa_v0, pooled_kv_inline205__ssa_v0

    @pl.function(type=pl.FunctionType.Spmd)
    def scatter_softmax_pool_spmd(
        self,
        pooled_kv_inline205__ssa_v0: pl.Out[pl.Tensor[[384, 512], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 786432)]],
        b_dim_inline518_inline1939__ssa_v0: pl.Scalar[pl.INDEX],
        pool_workers_inline512_inline1960__ssa_v0: pl.Scalar[pl.INDEX],
        position_ids__ssa_v0: pl.Tensor[[T_DYN], pl.INT64, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)],
        s_dim_inline513_inline1935__ssa_v0: pl.Scalar[pl.INDEX],
        state_block_table__ssa_v0: pl.Tensor[[B_DYN, STATE_TABLE_COLUMNS_DYN], pl.INT32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)],
        compress_state__ssa_v0: pl.Tensor[[MAIN_STATE_BLOCK_NUM_DYN, STATE_PAGE_ELEMENTS_DYN], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)],
        cmp4_kv_proj_pad_inline530_inline1957__ssa_v0: pl.Tensor[[384, 1024], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 1572864)],
        cmp4_score_proj_pad_inline510_inline1929__ssa_v0: pl.Tensor[[384, 1024], pl.FP32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 1572864)],
        cmp_ape__ssa_v0: pl.Tensor[[4, 1024], pl.FP32, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 16384)],
    ) -> pl.Tensor[[384, 512], pl.FP32]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        ret__tmp_v0: pl.Tuple[pl.Tensor[[384, 512], pl.FP32], pl.Tensor[[384, 512], pl.FP32]] = self.scatter_softmax_pool(
            pooled_kv_inline205__ssa_v0,
            b_dim_inline518_inline1939__ssa_v0,
            pool_workers_inline512_inline1960__ssa_v0,
            position_ids__ssa_v0,
            s_dim_inline513_inline1935__ssa_v0,
            state_block_table__ssa_v0,
            compress_state__ssa_v0,
            cmp4_kv_proj_pad_inline530_inline1957__ssa_v0,
            cmp4_score_proj_pad_inline510_inline1929__ssa_v0,
            cmp_ape__ssa_v0,
            attrs={
                "arg_directions": [pl.adir.output_existing, pl.adir.scalar, pl.adir.scalar, pl.adir.input, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input]
            },
        )
        pooled_kv_inline205__rv_v2: pl.Tensor[[384, 512], pl.FP32, pl.MemRef("mem_ddr_7", pl.const(0, pl.INT64), 786432)] = ret__tmp_v0[0]
        pooled_kv_inline205__ssa_v5: pl.Tensor[[384, 512], pl.FP32, pl.MemRef("mem_ddr_8", pl.const(0, pl.INT64), 786432)] = ret__tmp_v0[1]
        return pooled_kv_inline205__ssa_v0

    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def scatter_softmax_pool_0(
        pooled_kv_inline2143__ssa_v0: pl.Out[pl.Tensor[[384, 128], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 196608)]],
        window_scores_inline441_inline2089__ssa_v0: pl.InOut[pl.Tensor[[384, 128], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 196608)]],
        window_values_inline429_inline2121__ssa_v0: pl.InOut[pl.Tensor[[384, 128], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 196608)]],
        b_dim_inline454_inline2092__ssa_v0: pl.Scalar[pl.INDEX],
        pool_workers_inline436_inline2082__ssa_v0: pl.Scalar[pl.INDEX],
        position_ids__ssa_v0: pl.Tensor[[T_DYN], pl.INT64, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)],
        s_dim_inline421_inline2137__ssa_v0: pl.Scalar[pl.INDEX],
        inner_state_block_table__ssa_v0: pl.Tensor[[B_DYN, INNER_STATE_TABLE_COLUMNS_DYN], pl.INT32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 0)],
        inner_compress_state__ssa_v0: pl.Tensor[[INNER_STATE_BLOCK_NUM_DYN, INNER_STATE_PAGE_ELEMENTS_DYN], pl.FP32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 0)],
        kv_proj_pad_inline2105__rv_v2: pl.Tensor[[384, 256], pl.FP32, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 393216)],
        score_proj_pad_inline2117__rv_v2: pl.Tensor[[384, 256], pl.FP32, pl.MemRef("mem_ddr_7", pl.const(0, pl.INT64), 393216)],
        inner_ape__ssa_v0: pl.Tensor[[4, 256], pl.FP32, pl.MemRef("mem_ddr_8", pl.const(0, pl.INT64), 4096)],
    ) -> pl.Tensor[[384, 128], pl.FP32]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        mem_vec_12: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        mem_vec_13: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        mem_vec_14: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        mem_vec_29: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 2048)
        mem_vec_39: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 2048)
        pool_worker_inline437_inline2146__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        for c_idx_inline434_inline2147__idx_v0, (pooled_kv_inline2143__iter_v1, window_scores_inline441_inline2089__iter_v1, window_values_inline429_inline2121__iter_v1) in pl.range(
            pool_worker_inline437_inline2146__ssa_v0,
            b_dim_inline454_inline2092__ssa_v0,
            pool_workers_inline436_inline2082__ssa_v0,
            init_values=(pooled_kv_inline2143__ssa_v0, window_scores_inline441_inline2089__ssa_v0, window_values_inline429_inline2121__ssa_v0),
        ):
            first_pos_b_inline466_inline2113__tile: pl.Scalar[pl.INT64] = pl.tensor.read(position_ids__ssa_v0, [c_idx_inline434_inline2147__idx_v0 * s_dim_inline421_inline2137__ssa_v0])
            for s_idx_inline444_inline2083__idx_v0, (pooled_kv_inline2143__iter_v3, window_scores_inline441_inline2089__iter_v3, window_values_inline429_inline2121__iter_v3) in pl.range(
                s_dim_inline421_inline2137__ssa_v0, init_values=(pooled_kv_inline2143__iter_v1, window_scores_inline441_inline2089__iter_v1, window_values_inline429_inline2121__iter_v1)
            ):
                token_inline420_inline2157__ssa_v0: pl.Scalar[pl.INDEX] = c_idx_inline434_inline2147__idx_v0 * s_dim_inline421_inline2137__ssa_v0 + s_idx_inline444_inline2083__idx_v0
                token_pos_inline431_inline2148__tile: pl.Scalar[pl.INT64] = pl.tensor.read(position_ids__ssa_v0, [token_inline420_inline2157__ssa_v0])
                t__tile: pl.Tile[[1, 128], pl.FP32, pl.MemRef(mem_vec_29, pl.const(768, pl.INT64), 512), pl.Mem.Vec] = pl.tile.full([1, 128], dtype=pl.FP32, value=0.0)
                pooled_kv_inline2143__tile: pl.Tensor[[384, 128], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 196608)] = pl.tile.store(
                    t__tile, [token_inline420_inline2157__ssa_v0, 0], pooled_kv_inline2143__iter_v3
                )
                if (token_pos_inline431_inline2148__tile + 1) % 4 == 0:
                    window_start_inline416_inline2153__ssa_v0: pl.Scalar[pl.INT64] = token_pos_inline431_inline2148__tile - 8 + 1
                    for h0_inline412_inline2154__idx_v0, (pooled_kv_inline2143__iter_v6, window_scores_inline441_inline2089__iter_v5, window_values_inline429_inline2121__iter_v5) in pl.range(
                        0, 128, 64, init_values=(pooled_kv_inline2143__tile, window_scores_inline441_inline2089__iter_v3, window_values_inline429_inline2121__iter_v3)
                    ):
                        window_row_inline439_inline2118__ssa_v0: pl.Scalar[pl.INDEX] = pool_worker_inline437_inline2146__ssa_v0 * 8
                        for state_idx_inline452_inline2136__idx_v0, (window_scores_inline441_inline2089__iter_v7, window_values_inline429_inline2121__iter_v7) in pl.range(
                            8, init_values=(window_scores_inline441_inline2089__iter_v5, window_values_inline429_inline2121__iter_v5)
                        ):
                            logical_pos_inline424_inline2156__ssa_v0: pl.Scalar[pl.INT64] = window_start_inline416_inline2153__ssa_v0 + state_idx_inline452_inline2136__idx_v0
                            value_inline415_inline2162__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_29, pl.const(768, pl.INT64), 256), pl.Mem.Vec] = pl.tile.full(
                                [1, 64], dtype=pl.FP32, value=0.0
                            )
                            score_inline408_inline2145__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_39, pl.const(2816, pl.INT64), 256), pl.Mem.Vec] = pl.tile.full(
                                [1, 64], dtype=pl.FP32, value=-3.4028234663852886e38
                            )
                            state_half_inline402_inline2158__ssa_v0: pl.Scalar[pl.INDEX] = 0
                            if 4 <= state_idx_inline452_inline2136__idx_v0:
                                state_half_inline402_inline2158__ssa_v1: pl.Scalar[pl.INDEX] = 128
                                state_half_inline402_inline2158__phi_v2: pl.Scalar[pl.INDEX] = pl.yield_(state_half_inline402_inline2158__ssa_v1)
                            else:
                                state_half_inline402_inline2158__phi_v2: pl.Scalar[pl.INDEX] = pl.yield_(state_half_inline402_inline2158__ssa_v0)
                            if 0 <= logical_pos_inline424_inline2156__ssa_v0 and logical_pos_inline424_inline2156__ssa_v0 < first_pos_b_inline466_inline2113__tile:
                                score_inline408_inline2145__tile_1: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(0, pl.INT64), 256), pl.Mem.Vec] = pl.tile.full(
                                    [1, 64], dtype=pl.FP32, value=0.0
                                )
                                t__tile_1: pl.Scalar[pl.INT32] = pl.tensor.read(inner_state_block_table__ssa_v0, [c_idx_inline434_inline2147__idx_v0, logical_pos_inline424_inline2156__ssa_v0 // 2])
                                history_page_inline418_inline2085__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(t__tile_1, pl.INDEX)
                                if 0 <= history_page_inline418_inline2085__ssa_v0:
                                    history_column_inline401_inline2088__ssa_v0: pl.Scalar[pl.INDEX] = (
                                        pl.cast(logical_pos_inline424_inline2156__ssa_v0 % 2, pl.INDEX) * 512 + state_half_inline402_inline2158__phi_v2 + h0_inline412_inline2154__idx_v0
                                    )
                                    value_inline415_inline2162__tile_1: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(256, pl.INT64), 256), pl.Mem.Vec] = pl.tile.load(
                                        inner_compress_state__ssa_v0,
                                        [history_page_inline418_inline2085__ssa_v0, history_column_inline401_inline2088__ssa_v0],
                                        [1, 64],
                                        [1, 64],
                                        target_memory=pl.Mem.Vec,
                                    )
                                    score_inline408_inline2145__tile_2: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_14, pl.const(512, pl.INT64), 256), pl.Mem.Vec] = pl.tile.load(
                                        inner_compress_state__ssa_v0,
                                        [history_page_inline418_inline2085__ssa_v0, history_column_inline401_inline2088__ssa_v0 + 256],
                                        [1, 64],
                                        [1, 64],
                                        target_memory=pl.Mem.Vec,
                                    )
                                    score_inline408_inline2145__phi_v3, value_inline415_inline2162__phi_v2 = pl.yield_(score_inline408_inline2145__tile_2, value_inline415_inline2162__tile_1)
                                else:
                                    score_inline408_inline2145__tile_mv: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_14, pl.const(512, pl.INT64), 256), pl.Mem.Vec] = pl.tile.move(
                                        score_inline408_inline2145__tile_1, target_memory=pl.Mem.Vec
                                    )
                                    value_inline415_inline2162__tile_mv: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(256, pl.INT64), 256), pl.Mem.Vec] = pl.tile.move(
                                        value_inline415_inline2162__tile, target_memory=pl.Mem.Vec
                                    )
                                    score_inline408_inline2145__phi_v3, value_inline415_inline2162__phi_v2 = pl.yield_(score_inline408_inline2145__tile_mv, value_inline415_inline2162__tile_mv)
                                score_inline408_inline2145__phi_v4, value_inline415_inline2162__phi_v3 = pl.yield_(score_inline408_inline2145__phi_v3, value_inline415_inline2162__phi_v2)
                            else:
                                score_inline408_inline2145__tile_mv_1: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_14, pl.const(512, pl.INT64), 256), pl.Mem.Vec] = pl.tile.move(
                                    score_inline408_inline2145__tile, target_memory=pl.Mem.Vec
                                )
                                value_inline415_inline2162__tile_mv_1: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(256, pl.INT64), 256), pl.Mem.Vec] = pl.tile.move(
                                    value_inline415_inline2162__tile, target_memory=pl.Mem.Vec
                                )
                                score_inline408_inline2145__phi_v4, value_inline415_inline2162__phi_v3 = pl.yield_(score_inline408_inline2145__tile_mv_1, value_inline415_inline2162__tile_mv_1)
                            if first_pos_b_inline466_inline2113__tile <= logical_pos_inline424_inline2156__ssa_v0:
                                if logical_pos_inline424_inline2156__ssa_v0 <= token_pos_inline431_inline2148__tile:
                                    overlay_token_inline404_inline2159__ssa_v0: pl.Scalar[pl.INDEX] = (
                                        c_idx_inline434_inline2147__idx_v0 * s_dim_inline421_inline2137__ssa_v0 + logical_pos_inline424_inline2156__ssa_v0 - first_pos_b_inline466_inline2113__tile
                                    )
                                    ape_row_inline475_inline2128__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(logical_pos_inline424_inline2156__ssa_v0 % 4, pl.INDEX)
                                    value_inline415_inline2162__tile_2: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_29, pl.const(768, pl.INT64), 256), pl.Mem.Vec] = pl.tile.load(
                                        kv_proj_pad_inline2105__rv_v2,
                                        [overlay_token_inline404_inline2159__ssa_v0, state_half_inline402_inline2158__phi_v2 + h0_inline412_inline2154__idx_v0],
                                        [1, 64],
                                        [1, 64],
                                        target_memory=pl.Mem.Vec,
                                    )
                                    t__tile_2: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_39, pl.const(2816, pl.INT64), 256), pl.Mem.Vec] = pl.tile.load(
                                        score_proj_pad_inline2117__rv_v2,
                                        [overlay_token_inline404_inline2159__ssa_v0, state_half_inline402_inline2158__phi_v2 + h0_inline412_inline2154__idx_v0],
                                        [1, 64],
                                        [1, 64],
                                        target_memory=pl.Mem.Vec,
                                    )
                                    t__tile_3: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(0, pl.INT64), 256), pl.Mem.Vec] = pl.tile.load(
                                        inner_ape__ssa_v0,
                                        [ape_row_inline475_inline2128__ssa_v0, state_half_inline402_inline2158__phi_v2 + h0_inline412_inline2154__idx_v0],
                                        [1, 64],
                                        [1, 64],
                                        target_memory=pl.Mem.Vec,
                                    )
                                    score_inline408_inline2145__tile_3: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_39, pl.const(2816, pl.INT64), 256), pl.Mem.Vec] = pl.tile.add(t__tile_2, t__tile_3)
                                    score_inline408_inline2145__phi_v6, value_inline415_inline2162__phi_v5 = pl.yield_(score_inline408_inline2145__tile_3, value_inline415_inline2162__tile_2)
                                else:
                                    score_inline408_inline2145__phi_v4_mv: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_39, pl.const(2816, pl.INT64), 256), pl.Mem.Vec] = pl.tile.move(
                                        score_inline408_inline2145__phi_v4, target_memory=pl.Mem.Vec
                                    )
                                    value_inline415_inline2162__phi_v3_mv: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_29, pl.const(768, pl.INT64), 256), pl.Mem.Vec] = pl.tile.move(
                                        value_inline415_inline2162__phi_v3, target_memory=pl.Mem.Vec
                                    )
                                    score_inline408_inline2145__phi_v6, value_inline415_inline2162__phi_v5 = pl.yield_(score_inline408_inline2145__phi_v4_mv, value_inline415_inline2162__phi_v3_mv)
                                score_inline408_inline2145__phi_v7, value_inline415_inline2162__phi_v6 = pl.yield_(score_inline408_inline2145__phi_v6, value_inline415_inline2162__phi_v5)
                            else:
                                score_inline408_inline2145__phi_v4_mv_1: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_39, pl.const(2816, pl.INT64), 256), pl.Mem.Vec] = pl.tile.move(
                                    score_inline408_inline2145__phi_v4, target_memory=pl.Mem.Vec
                                )
                                value_inline415_inline2162__phi_v3_mv_1: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_29, pl.const(768, pl.INT64), 256), pl.Mem.Vec] = pl.tile.move(
                                    value_inline415_inline2162__phi_v3, target_memory=pl.Mem.Vec
                                )
                                score_inline408_inline2145__phi_v7, value_inline415_inline2162__phi_v6 = pl.yield_(score_inline408_inline2145__phi_v4_mv_1, value_inline415_inline2162__phi_v3_mv_1)
                            native_row_inline422_inline2119__ssa_v0: pl.Scalar[pl.INDEX] = state_idx_inline452_inline2136__idx_v0 % 4 * 2 + state_idx_inline452_inline2136__idx_v0 // 4
                            dst_row_inline433_inline2133__ssa_v0: pl.Scalar[pl.INDEX] = window_row_inline439_inline2118__ssa_v0 + native_row_inline422_inline2119__ssa_v0
                            window_values_inline429_inline2121__tile: pl.Tensor[[384, 128], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 196608)] = pl.tile.store(
                                value_inline415_inline2162__phi_v6, [dst_row_inline433_inline2133__ssa_v0, h0_inline412_inline2154__idx_v0], window_values_inline429_inline2121__iter_v7
                            )
                            window_scores_inline441_inline2089__tile: pl.Tensor[[384, 128], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 196608)] = pl.tile.store(
                                score_inline408_inline2145__phi_v7, [dst_row_inline433_inline2133__ssa_v0, h0_inline412_inline2154__idx_v0], window_scores_inline441_inline2089__iter_v7
                            )
                            window_scores_inline441_inline2089__rv_v8, window_values_inline429_inline2121__rv_v8 = pl.yield_(
                                window_scores_inline441_inline2089__tile, window_values_inline429_inline2121__tile
                            )
                        score_rows_inline445_inline2123__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_29, pl.const(768, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.load(
                            window_scores_inline441_inline2089__rv_v8, [window_row_inline439_inline2118__ssa_v0, h0_inline412_inline2154__idx_v0], [8, 64], [8, 64], target_memory=pl.Mem.Vec
                        )
                        t__tile_4: pl.Tile[[4, 64], pl.FP32, pl.MemRef(mem_vec_29, pl.const(768, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.slice(score_rows_inline445_inline2123__tile, [4, 64], [0, 0])
                        t__tile_5: pl.Tile[[4, 64], pl.FP32, pl.MemRef(mem_vec_29, pl.const(1792, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.slice(score_rows_inline445_inline2123__tile, [4, 64], [4, 0])
                        max4_inline446_inline2141__tile: pl.Tile[[4, 64], pl.FP32, pl.MemRef(mem_vec_39, pl.const(2816, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.maximum(t__tile_4, t__tile_5)
                        t__tile_6: pl.Tile[[2, 64], pl.FP32, pl.MemRef(mem_vec_39, pl.const(2816, pl.INT64), 512), pl.Mem.Vec] = pl.tile.slice(max4_inline446_inline2141__tile, [2, 64], [0, 0])
                        t__tile_7: pl.Tile[[2, 64], pl.FP32, pl.MemRef(mem_vec_39, pl.const(3328, pl.INT64), 512), pl.Mem.Vec] = pl.tile.slice(max4_inline446_inline2141__tile, [2, 64], [2, 0])
                        max2_inline447_inline2161__tile: pl.Tile[[2, 64], pl.FP32, pl.MemRef(mem_vec_39, pl.const(2816, pl.INT64), 512), pl.Mem.Vec] = pl.tile.maximum(t__tile_6, t__tile_7)
                        t__tile_8: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_39, pl.const(2816, pl.INT64), 256), pl.Mem.Vec] = pl.tile.slice(max2_inline447_inline2161__tile, [1, 64], [0, 0])
                        t__tile_9: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_39, pl.const(3072, pl.INT64), 256), pl.Mem.Vec] = pl.tile.slice(max2_inline447_inline2161__tile, [1, 64], [1, 0])
                        maximum_inline448_inline2101__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_39, pl.const(2816, pl.INT64), 256), pl.Mem.Vec] = pl.tile.maximum(t__tile_8, t__tile_9)
                        t__tile_10: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_29, pl.const(768, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.col_expand_sub(
                            score_rows_inline445_inline2123__tile, maximum_inline448_inline2101__tile
                        )
                        probability_inline453_inline2164__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_29, pl.const(768, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.exp(t__tile_10)
                        t__tile_11: pl.Tile[[4, 64], pl.FP32, pl.MemRef(mem_vec_29, pl.const(768, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.slice(probability_inline453_inline2164__tile, [4, 64], [0, 0])
                        t__tile_12: pl.Tile[[4, 64], pl.FP32, pl.MemRef(mem_vec_29, pl.const(1792, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.slice(
                            probability_inline453_inline2164__tile, [4, 64], [4, 0]
                        )
                        sum4_inline430_inline2160__tile: pl.Tile[[4, 64], pl.FP32, pl.MemRef(mem_vec_39, pl.const(2816, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.add(t__tile_11, t__tile_12)
                        t__tile_13: pl.Tile[[2, 64], pl.FP32, pl.MemRef(mem_vec_39, pl.const(2816, pl.INT64), 512), pl.Mem.Vec] = pl.tile.slice(sum4_inline430_inline2160__tile, [2, 64], [0, 0])
                        t__tile_14: pl.Tile[[2, 64], pl.FP32, pl.MemRef(mem_vec_39, pl.const(3328, pl.INT64), 512), pl.Mem.Vec] = pl.tile.slice(sum4_inline430_inline2160__tile, [2, 64], [2, 0])
                        sum2_inline407_inline2080__tile: pl.Tile[[2, 64], pl.FP32, pl.MemRef(mem_vec_39, pl.const(2816, pl.INT64), 512), pl.Mem.Vec] = pl.tile.add(t__tile_13, t__tile_14)
                        t__tile_15: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_39, pl.const(2816, pl.INT64), 256), pl.Mem.Vec] = pl.tile.slice(sum2_inline407_inline2080__tile, [1, 64], [0, 0])
                        t__tile_16: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_39, pl.const(3072, pl.INT64), 256), pl.Mem.Vec] = pl.tile.slice(sum2_inline407_inline2080__tile, [1, 64], [1, 0])
                        total_inline438_inline2165__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_39, pl.const(2816, pl.INT64), 256), pl.Mem.Vec] = pl.tile.add(t__tile_15, t__tile_16)
                        probability_v1_inline455_inline2163__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_29, pl.const(768, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.col_expand_div(
                            probability_inline453_inline2164__tile, total_inline438_inline2165__tile
                        )
                        value_rows_inline410_inline2116__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_39, pl.const(2816, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.load(
                            window_values_inline429_inline2121__rv_v8, [window_row_inline439_inline2118__ssa_v0, h0_inline412_inline2154__idx_v0], [8, 64], [8, 64], target_memory=pl.Mem.Vec
                        )
                        weighted_inline428_inline2102__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_29, pl.const(768, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.mul(
                            value_rows_inline410_inline2116__tile, probability_v1_inline455_inline2163__tile
                        )
                        t__tile_17: pl.Tile[[4, 64], pl.FP32, pl.MemRef(mem_vec_29, pl.const(768, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.slice(weighted_inline428_inline2102__tile, [4, 64], [0, 0])
                        t__tile_18: pl.Tile[[4, 64], pl.FP32, pl.MemRef(mem_vec_29, pl.const(1792, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.slice(weighted_inline428_inline2102__tile, [4, 64], [4, 0])
                        weighted4_inline457_inline2078__tile: pl.Tile[[4, 64], pl.FP32, pl.MemRef(mem_vec_29, pl.const(768, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.add(t__tile_17, t__tile_18)
                        t__tile_19: pl.Tile[[2, 64], pl.FP32, pl.MemRef(mem_vec_29, pl.const(768, pl.INT64), 512), pl.Mem.Vec] = pl.tile.slice(weighted4_inline457_inline2078__tile, [2, 64], [0, 0])
                        t__tile_20: pl.Tile[[2, 64], pl.FP32, pl.MemRef(mem_vec_29, pl.const(1280, pl.INT64), 512), pl.Mem.Vec] = pl.tile.slice(weighted4_inline457_inline2078__tile, [2, 64], [2, 0])
                        weighted2_inline458_inline2084__tile: pl.Tile[[2, 64], pl.FP32, pl.MemRef(mem_vec_29, pl.const(768, pl.INT64), 512), pl.Mem.Vec] = pl.tile.add(t__tile_19, t__tile_20)
                        t__tile_21: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_29, pl.const(768, pl.INT64), 256), pl.Mem.Vec] = pl.tile.slice(weighted2_inline458_inline2084__tile, [1, 64], [0, 0])
                        t__tile_22: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_29, pl.const(1024, pl.INT64), 256), pl.Mem.Vec] = pl.tile.slice(weighted2_inline458_inline2084__tile, [1, 64], [1, 0])
                        t__tile_23: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_29, pl.const(768, pl.INT64), 256), pl.Mem.Vec] = pl.tile.add(t__tile_21, t__tile_22)
                        pooled_kv_inline2143__tile_1: pl.Tensor[[384, 128], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 196608)] = pl.tile.store(
                            t__tile_23, [token_inline420_inline2157__ssa_v0, h0_inline412_inline2154__idx_v0], pooled_kv_inline2143__iter_v6
                        )
                        pooled_kv_inline2143__rv_v7, window_scores_inline441_inline2089__rv_v6, window_values_inline429_inline2121__rv_v6 = pl.yield_(
                            pooled_kv_inline2143__tile_1, window_scores_inline441_inline2089__rv_v8, window_values_inline429_inline2121__rv_v8
                        )
                    pooled_kv_inline2143__phi_v9, window_scores_inline441_inline2089__phi_v10, window_values_inline429_inline2121__phi_v10 = pl.yield_(
                        pooled_kv_inline2143__rv_v7, window_scores_inline441_inline2089__rv_v6, window_values_inline429_inline2121__rv_v6
                    )
                else:
                    pooled_kv_inline2143__phi_v9, window_scores_inline441_inline2089__phi_v10, window_values_inline429_inline2121__phi_v10 = pl.yield_(
                        pooled_kv_inline2143__tile, window_scores_inline441_inline2089__iter_v3, window_values_inline429_inline2121__iter_v3
                    )
                pooled_kv_inline2143__rv_v4, window_scores_inline441_inline2089__rv_v4, window_values_inline429_inline2121__rv_v4 = pl.yield_(
                    pooled_kv_inline2143__phi_v9, window_scores_inline441_inline2089__phi_v10, window_values_inline429_inline2121__phi_v10
                )
            pooled_kv_inline2143__rv_v2, window_scores_inline441_inline2089__rv_v2, window_values_inline429_inline2121__rv_v2 = pl.yield_(
                pooled_kv_inline2143__rv_v4, window_scores_inline441_inline2089__rv_v4, window_values_inline429_inline2121__rv_v4
            )
        return pooled_kv_inline2143__ssa_v0

    @pl.function(type=pl.FunctionType.Spmd)
    def scatter_softmax_pool_spmd_0(
        self,
        pooled_kv_inline2143__ssa_v0: pl.Out[pl.Tensor[[384, 128], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 196608)]],
        window_scores_inline441_inline2089__ssa_v0: pl.InOut[pl.Tensor[[384, 128], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 196608)]],
        window_values_inline429_inline2121__ssa_v0: pl.InOut[pl.Tensor[[384, 128], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 196608)]],
        b_dim_inline454_inline2092__ssa_v0: pl.Scalar[pl.INDEX],
        pool_workers_inline436_inline2082__ssa_v0: pl.Scalar[pl.INDEX],
        position_ids__ssa_v0: pl.Tensor[[T_DYN], pl.INT64, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)],
        s_dim_inline421_inline2137__ssa_v0: pl.Scalar[pl.INDEX],
        inner_state_block_table__ssa_v0: pl.Tensor[[B_DYN, INNER_STATE_TABLE_COLUMNS_DYN], pl.INT32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 0)],
        inner_compress_state__ssa_v0: pl.Tensor[[INNER_STATE_BLOCK_NUM_DYN, INNER_STATE_PAGE_ELEMENTS_DYN], pl.FP32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 0)],
        kv_proj_pad_inline2105__rv_v2: pl.Tensor[[384, 256], pl.FP32, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 393216)],
        score_proj_pad_inline2117__rv_v2: pl.Tensor[[384, 256], pl.FP32, pl.MemRef("mem_ddr_7", pl.const(0, pl.INT64), 393216)],
        inner_ape__ssa_v0: pl.Tensor[[4, 256], pl.FP32, pl.MemRef("mem_ddr_8", pl.const(0, pl.INT64), 4096)],
    ) -> pl.Tensor[[384, 128], pl.FP32]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        pooled_kv_inline2143__rv_v2: pl.Tensor[[384, 128], pl.FP32, pl.MemRef("mem_ddr_9", pl.const(0, pl.INT64), 196608)] = self.scatter_softmax_pool_0(
            pooled_kv_inline2143__ssa_v0,
            window_scores_inline441_inline2089__ssa_v0,
            window_values_inline429_inline2121__ssa_v0,
            b_dim_inline454_inline2092__ssa_v0,
            pool_workers_inline436_inline2082__ssa_v0,
            position_ids__ssa_v0,
            s_dim_inline421_inline2137__ssa_v0,
            inner_state_block_table__ssa_v0,
            inner_compress_state__ssa_v0,
            kv_proj_pad_inline2105__rv_v2,
            score_proj_pad_inline2117__rv_v2,
            inner_ape__ssa_v0,
            attrs={
                "arg_directions": [
                    pl.adir.output_existing,
                    pl.adir.inout,
                    pl.adir.inout,
                    pl.adir.scalar,
                    pl.adir.scalar,
                    pl.adir.input,
                    pl.adir.scalar,
                    pl.adir.input,
                    pl.adir.input,
                    pl.adir.input,
                    pl.adir.input,
                    pl.adir.input,
                ]
            },
        )
        return pooled_kv_inline2143__ssa_v0

    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def weights_proj_reduce(
        weights_partial_inline1024_inline2370__rv_v2: pl.Tensor[[384, 64], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 98304)],
        weights_inline1023_inline2353__ssa_v0: pl.Out[pl.Tensor[[384, 64], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 98304)]],
    ) -> pl.Tensor[[384, 64], pl.FP32]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        mem_vec_2: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 4096)
        mem_vec_3: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 2048)
        w_rb_inline1021_inline2383__ssa_v1: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        w_r0_inline1033_inline2352__ssa_v1: pl.Scalar[pl.INDEX] = w_rb_inline1021_inline2383__ssa_v1 * 16
        w_sum_inline1011_inline2336__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_2, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.load(
            weights_partial_inline1024_inline2370__rv_v2, [w_r0_inline1033_inline2352__ssa_v1, 0], [16, 64], [16, 64], target_memory=pl.Mem.Vec
        )
        t__tile: pl.Tile[[16, 64], pl.BF16, pl.MemRef(mem_vec_3, pl.const(4096, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(w_sum_inline1011_inline2336__tile, target_type=pl.BF16, mode="rint")
        w_sum_v1_inline1010_inline2373__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_2, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(t__tile, target_type=pl.FP32, mode="round")
        t__tile_1: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_2, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.muls(w_sum_v1_inline1010_inline2373__tile, 0.011048543456039806)
        w_scaled_inline1009_inline2364__tile: pl.Tile[[16, 64], pl.BF16, pl.MemRef(mem_vec_3, pl.const(4096, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(t__tile_1, target_type=pl.BF16, mode="rint")
        t__cast_fp32_tmp_v0: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_2, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(
            w_scaled_inline1009_inline2364__tile, target_type=pl.FP32, mode="round"
        )
        t__tile_2: pl.Tile[[16, 64], pl.FP16, pl.MemRef(mem_vec_3, pl.const(4096, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(t__cast_fp32_tmp_v0, target_type=pl.FP16, mode="round")
        t__tile_3: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_2, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(t__tile_2, target_type=pl.FP32, mode="round")
        weights_inline1023_inline2353__tile: pl.Tensor[[384, 64], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 98304)] = pl.tile.store(
            t__tile_3, [w_r0_inline1033_inline2352__ssa_v1, 0], weights_inline1023_inline2353__ssa_v0
        )
        return weights_inline1023_inline2353__ssa_v0

    @pl.function(type=pl.FunctionType.Spmd)
    def weights_proj_reduce_spmd(
        self,
        weights_partial_inline1024_inline2370__rv_v2: pl.Tensor[[384, 64], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 98304)],
        weights_inline1023_inline2353__ssa_v0: pl.Out[pl.Tensor[[384, 64], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 98304)]],
    ) -> pl.Tensor[[384, 64], pl.FP32]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        weights_inline1023_inline2353__ssa_v1: pl.Tensor[[384, 64], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 98304)] = self.weights_proj_reduce(
            weights_partial_inline1024_inline2370__rv_v2, weights_inline1023_inline2353__ssa_v0, attrs={"arg_directions": [pl.adir.input, pl.adir.output_existing]}
        )
        return weights_inline1023_inline2353__ssa_v0

    @pl.function(type=pl.FunctionType.AIC, level=pl.Level.AIC, role=pl.Role.SubWorker)
    def weights_proj(
        weights_partial_inline1024_inline2370__ssa_v0: pl.Out[pl.Tensor[[384, 64], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 98304)]],
        row_blocks_inline1022_inline2357__ssa_v0: pl.Scalar[pl.INDEX],
        bs_inline1017_inline2346__ssa_v0: pl.Scalar[pl.INDEX],
        x_flat_inline1018_inline2361__ssa_v0: pl.Tensor[[T_DYN, 4096], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)],
        weights_proj__ssa_v0: pl.Tensor[[4096, 64], pl.BF16, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 524288)],
    ) -> pl.Tensor[[384, 64], pl.FP32]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        mem_acc_3: pl.Ptr = pl.tile.alloc(pl.Mem.Acc, 1024)
        mem_mat_4: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 8192)
        mem_mat_5: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 8192)
        mem_left_6: pl.Ptr = pl.tile.alloc(pl.Mem.Left, 8192)
        mem_right_7: pl.Ptr = pl.tile.alloc(pl.Mem.Right, 8192)
        w_worker_inline1027_inline2362__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        for w_unit_inline1026_inline2375__idx_v0, (weights_partial_inline1024_inline2370__iter_v1,) in pl.range(
            w_worker_inline1027_inline2362__ssa_v0, row_blocks_inline1022_inline2357__ssa_v0 * 4, 8, init_values=(weights_partial_inline1024_inline2370__ssa_v0,)
        ):
            w_rb_inline1021_inline2383__ssa_v0: pl.Scalar[pl.INDEX] = w_unit_inline1026_inline2375__idx_v0 // 4
            w_ng_inline1029_inline2334__ssa_v0: pl.Scalar[pl.INDEX] = w_unit_inline1026_inline2375__idx_v0 % 4
            w_n0_inline1032_inline2360__ssa_v0: pl.Scalar[pl.INDEX] = w_ng_inline1029_inline2334__ssa_v0 * 16
            w_r0_inline1033_inline2352__ssa_v0: pl.Scalar[pl.INDEX] = w_rb_inline1021_inline2383__ssa_v0 * 16
            w_rows_inline1031_inline2376__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(bs_inline1017_inline2346__ssa_v0 - w_r0_inline1033_inline2352__ssa_v0, 16)
            weights_acc_inline1016_inline2345__tile: pl.Tile[[16, 16], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 1024), pl.Mem.Acc] = pl.tile.create(
                [16, 16], dtype=pl.FP32, target_memory=pl.Mem.Acc
            )
            for db_inline1014_inline2343__idx_v0, (weights_acc_inline1016_inline2345__iter_v1,) in pl.range(16, init_values=(weights_acc_inline1016_inline2345__tile,)):
                k_order_inline1020_inline2367__ssa_v0: pl.Scalar[pl.INDEX] = db_inline1014_inline2343__idx_v0
                if bs_inline1017_inline2346__ssa_v0 == 24:
                    k_direction_inline1028_inline2395__ssa_v0: pl.Scalar[pl.INDEX] = db_inline1014_inline2343__idx_v0 // 2
                    if w_ng_inline1029_inline2334__ssa_v0 % 2 == 1:
                        k_direction_inline1028_inline2395__ssa_v1: pl.Scalar[pl.INDEX] = 7 - db_inline1014_inline2343__idx_v0 // 2
                        k_direction_inline1028_inline2395__phi_v2: pl.Scalar[pl.INDEX] = pl.yield_(k_direction_inline1028_inline2395__ssa_v1)
                    else:
                        k_direction_inline1028_inline2395__phi_v2: pl.Scalar[pl.INDEX] = pl.yield_(k_direction_inline1028_inline2395__ssa_v0)
                    k_order_inline1020_inline2367__ssa_v1: pl.Scalar[pl.INDEX] = (
                        w_ng_inline1029_inline2334__ssa_v0 // 2 * 4 + k_direction_inline1028_inline2395__phi_v2
                    ) % 8 * 2 + db_inline1014_inline2343__idx_v0 % 2
                    k_order_inline1020_inline2367__phi_v6: pl.Scalar[pl.INDEX] = pl.yield_(k_order_inline1020_inline2367__ssa_v1)
                else:
                    if bs_inline1017_inline2346__ssa_v0 == 48:
                        k_direction_inline1028_inline2395__ssa_v3: pl.Scalar[pl.INDEX] = db_inline1014_inline2343__idx_v0
                        if w_ng_inline1029_inline2334__ssa_v0 % 2 == 1:
                            k_direction_inline1028_inline2395__ssa_v4: pl.Scalar[pl.INDEX] = 15 - db_inline1014_inline2343__idx_v0
                            k_direction_inline1028_inline2395__phi_v5: pl.Scalar[pl.INDEX] = pl.yield_(k_direction_inline1028_inline2395__ssa_v4)
                        else:
                            k_direction_inline1028_inline2395__phi_v5: pl.Scalar[pl.INDEX] = pl.yield_(k_direction_inline1028_inline2395__ssa_v3)
                        k_order_inline1020_inline2367__ssa_v2: pl.Scalar[pl.INDEX] = (w_ng_inline1029_inline2334__ssa_v0 // 2 * 8 + k_direction_inline1028_inline2395__phi_v5) % 16
                        k_order_inline1020_inline2367__phi_v5: pl.Scalar[pl.INDEX] = pl.yield_(k_order_inline1020_inline2367__ssa_v2)
                    else:
                        if bs_inline1017_inline2346__ssa_v0 == 96 or bs_inline1017_inline2346__ssa_v0 == 144 or bs_inline1017_inline2346__ssa_v0 == 192 or bs_inline1017_inline2346__ssa_v0 == 240:
                            k_direction_inline1028_inline2395__ssa_v6: pl.Scalar[pl.INDEX] = db_inline1014_inline2343__idx_v0
                            if w_rb_inline1021_inline2383__ssa_v0 % 2 == 1:
                                k_direction_inline1028_inline2395__ssa_v7: pl.Scalar[pl.INDEX] = 15 - db_inline1014_inline2343__idx_v0
                                k_direction_inline1028_inline2395__phi_v8: pl.Scalar[pl.INDEX] = pl.yield_(k_direction_inline1028_inline2395__ssa_v7)
                            else:
                                k_direction_inline1028_inline2395__phi_v8: pl.Scalar[pl.INDEX] = pl.yield_(k_direction_inline1028_inline2395__ssa_v6)
                            k_shift_inline1015_inline2341__ssa_v0: pl.Scalar[pl.INDEX] = w_rb_inline1021_inline2383__ssa_v0 // 2 * 5
                            if bs_inline1017_inline2346__ssa_v0 == 144:
                                k_shift_inline1015_inline2341__ssa_v1: pl.Scalar[pl.INDEX] = w_rb_inline1021_inline2383__ssa_v0 % 8 // 2 * 4
                                k_shift_inline1015_inline2341__phi_v6: pl.Scalar[pl.INDEX] = pl.yield_(k_shift_inline1015_inline2341__ssa_v1)
                            else:
                                if bs_inline1017_inline2346__ssa_v0 == 192:
                                    k_shift_inline1015_inline2341__ssa_v2: pl.Scalar[pl.INDEX] = w_rb_inline1021_inline2383__ssa_v0 // 2 * 2
                                    k_shift_inline1015_inline2341__phi_v5: pl.Scalar[pl.INDEX] = pl.yield_(k_shift_inline1015_inline2341__ssa_v2)
                                else:
                                    if bs_inline1017_inline2346__ssa_v0 == 240:
                                        k_shift_inline1015_inline2341__ssa_v3: pl.Scalar[pl.INDEX] = w_rb_inline1021_inline2383__ssa_v0 % 14 // 2 * 2
                                        k_shift_inline1015_inline2341__phi_v4: pl.Scalar[pl.INDEX] = pl.yield_(k_shift_inline1015_inline2341__ssa_v3)
                                    else:
                                        k_shift_inline1015_inline2341__phi_v4: pl.Scalar[pl.INDEX] = pl.yield_(k_shift_inline1015_inline2341__ssa_v0)
                                    k_shift_inline1015_inline2341__phi_v5: pl.Scalar[pl.INDEX] = pl.yield_(k_shift_inline1015_inline2341__phi_v4)
                                k_shift_inline1015_inline2341__phi_v6: pl.Scalar[pl.INDEX] = pl.yield_(k_shift_inline1015_inline2341__phi_v5)
                            k_order_inline1020_inline2367__ssa_v3: pl.Scalar[pl.INDEX] = (k_shift_inline1015_inline2341__phi_v6 + k_direction_inline1028_inline2395__phi_v8) % 16
                            k_order_inline1020_inline2367__phi_v4: pl.Scalar[pl.INDEX] = pl.yield_(k_order_inline1020_inline2367__ssa_v3)
                        else:
                            k_order_inline1020_inline2367__phi_v4: pl.Scalar[pl.INDEX] = pl.yield_(k_order_inline1020_inline2367__ssa_v0)
                        k_order_inline1020_inline2367__phi_v5: pl.Scalar[pl.INDEX] = pl.yield_(k_order_inline1020_inline2367__phi_v4)
                    k_order_inline1020_inline2367__phi_v6: pl.Scalar[pl.INDEX] = pl.yield_(k_order_inline1020_inline2367__phi_v5)
                d0_inline1019_inline2349__ssa_v0: pl.Scalar[pl.INDEX] = k_order_inline1020_inline2367__phi_v6 * 256
                x_tile_inline1030_inline2389__tile: pl.Tile[
                    [16, 256], pl.BF16, pl.MemRef(mem_mat_4, pl.const(0, pl.INT64), 8192), pl.Mem.Mat, pl.TileView(valid_shape=[w_rows_inline1031_inline2376__ssa_v0, 256])
                ] = pl.tile.load(
                    x_flat_inline1018_inline2361__ssa_v0,
                    [w_r0_inline1033_inline2352__ssa_v0, d0_inline1019_inline2349__ssa_v0],
                    [16, 256],
                    [w_rows_inline1031_inline2376__ssa_v0, 256],
                    target_memory=pl.Mem.Mat,
                )
                weights_proj_tile_inline1013_inline2351__tile: pl.Tile[[256, 16], pl.BF16, pl.MemRef(mem_mat_5, pl.const(8192, pl.INT64), 8192), pl.Mem.Mat] = pl.tile.load(
                    weights_proj__ssa_v0, [d0_inline1019_inline2349__ssa_v0, w_n0_inline1032_inline2360__ssa_v0], [256, 16], [256, 16], target_memory=pl.Mem.Mat
                )
                x_tile_inline1030_inline2389__tile_Left: pl.Tile[
                    [16, 256], pl.BF16, pl.MemRef(mem_left_6, pl.const(0, pl.INT64), 8192), pl.Mem.Left, pl.TileView(valid_shape=[w_rows_inline1031_inline2376__ssa_v0, 256])
                ] = pl.tile.move(x_tile_inline1030_inline2389__tile, target_memory=pl.Mem.Left)
                weights_proj_tile_inline1013_inline2351__tile_Right: pl.Tile[[256, 16], pl.BF16, pl.MemRef(mem_right_7, pl.const(0, pl.INT64), 8192), pl.Mem.Right] = pl.tile.move(
                    weights_proj_tile_inline1013_inline2351__tile, target_memory=pl.Mem.Right
                )
                weights_acc_inline1016_inline2345__tile_1: pl.Tile[[16, 16], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 1024), pl.Mem.Acc] = pl.tile.matmul_acc(
                    weights_acc_inline1016_inline2345__iter_v1, x_tile_inline1030_inline2389__tile_Left, weights_proj_tile_inline1013_inline2351__tile_Right, db_inline1014_inline2343__idx_v0 == 0
                )
                weights_acc_inline1016_inline2345__rv_v2: pl.Tile[[16, 16], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 1024), pl.Mem.Acc] = pl.yield_(
                    weights_acc_inline1016_inline2345__tile_1
                )
            weights_partial_inline1024_inline2370__tile: pl.Tensor[[384, 64], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 98304)] = pl.tile.store(
                weights_acc_inline1016_inline2345__rv_v2, [w_r0_inline1033_inline2352__ssa_v0, w_n0_inline1032_inline2360__ssa_v0], weights_partial_inline1024_inline2370__iter_v1
            )
            weights_partial_inline1024_inline2370__rv_v2: pl.Tensor[[384, 64], pl.FP32, pl.MemRef("mem_ddr_9", pl.const(0, pl.INT64), 98304)] = pl.yield_(weights_partial_inline1024_inline2370__tile)
        return weights_partial_inline1024_inline2370__ssa_v0

    @pl.function(type=pl.FunctionType.Spmd)
    def weights_proj_spmd(
        self,
        weights_partial_inline1024_inline2370__ssa_v0: pl.Out[pl.Tensor[[384, 64], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 98304)]],
        row_blocks_inline1022_inline2357__ssa_v0: pl.Scalar[pl.INDEX],
        bs_inline1017_inline2346__ssa_v0: pl.Scalar[pl.INDEX],
        x_flat_inline1018_inline2361__ssa_v0: pl.Tensor[[T_DYN, 4096], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)],
        weights_proj__ssa_v0: pl.Tensor[[4096, 64], pl.BF16, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 524288)],
    ) -> pl.Tensor[[384, 64], pl.FP32]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        weights_partial_inline1024_inline2370__rv_v2: pl.Tensor[[384, 64], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 98304)] = self.weights_proj(
            weights_partial_inline1024_inline2370__ssa_v0,
            row_blocks_inline1022_inline2357__ssa_v0,
            bs_inline1017_inline2346__ssa_v0,
            x_flat_inline1018_inline2361__ssa_v0,
            weights_proj__ssa_v0,
            attrs={"arg_directions": [pl.adir.output_existing, pl.adir.scalar, pl.adir.scalar, pl.adir.input, pl.adir.input]},
        )
        return weights_partial_inline1024_inline2370__ssa_v0

    @pl.function(type=pl.FunctionType.Orchestration, level=pl.Level.CHIP, role=pl.Role.Orchestrator, auto_scope=False)
    def _decode_csa_tp1_attention(
        self,
        x_normed_t__ssa_v0: pl.Tensor[[T_DYN, 4096], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)],
        wq_a__ssa_v0: pl.Tensor[[4096, 1024], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 8388608)],
        wq_b__ssa_v0: pl.Tensor[[1024, 32768], pl.INT8, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 33554432)],
        wq_b_scale__ssa_v0: pl.Tensor[[32768], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 131072)],
        wkv__ssa_v0: pl.Tensor[[4096, 512], pl.BF16, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 4194304)],
        gamma_cq__ssa_v0: pl.Tensor[[1024], pl.BF16, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 2048)],
        gamma_ckv__ssa_v0: pl.Tensor[[512], pl.BF16, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 1024)],
        freqs_cos__ssa_v0: pl.Tensor[[T_DYN, 64], pl.FP32, pl.MemRef("mem_ddr_7", pl.const(0, pl.INT64), 0)],
        freqs_sin__ssa_v0: pl.Tensor[[T_DYN, 64], pl.FP32, pl.MemRef("mem_ddr_8", pl.const(0, pl.INT64), 0)],
        cmp_freqs_cos__ssa_v0: pl.Tensor[[COMPRESSED_ROWS_DYN, 64], pl.FP32, pl.MemRef("mem_ddr_9", pl.const(0, pl.INT64), 0)],
        cmp_freqs_sin__ssa_v0: pl.Tensor[[COMPRESSED_ROWS_DYN, 64], pl.FP32, pl.MemRef("mem_ddr_10", pl.const(0, pl.INT64), 0)],
        inner_freqs_cos__ssa_v0: pl.Tensor[[INDEXER_ROWS_DYN, 64], pl.FP32, pl.MemRef("mem_ddr_11", pl.const(0, pl.INT64), 0)],
        inner_freqs_sin__ssa_v0: pl.Tensor[[INDEXER_ROWS_DYN, 64], pl.FP32, pl.MemRef("mem_ddr_12", pl.const(0, pl.INT64), 0)],
        cmp_wkv__ssa_v0: pl.Tensor[[1024, 4096], pl.BF16, pl.MemRef("mem_ddr_13", pl.const(0, pl.INT64), 8388608)],
        cmp_wgate__ssa_v0: pl.Tensor[[1024, 4096], pl.BF16, pl.MemRef("mem_ddr_14", pl.const(0, pl.INT64), 8388608)],
        cmp_ape__ssa_v0: pl.Tensor[[4, 1024], pl.FP32, pl.MemRef("mem_ddr_15", pl.const(0, pl.INT64), 16384)],
        cmp_norm_w__ssa_v0: pl.Tensor[[512], pl.FP32, pl.MemRef("mem_ddr_16", pl.const(0, pl.INT64), 2048)],
        compress_state__ssa_v0: pl.InOut[pl.Tensor[[MAIN_STATE_BLOCK_NUM_DYN, STATE_PAGE_ELEMENTS_DYN], pl.FP32, pl.MemRef("mem_ddr_17", pl.const(0, pl.INT64), 0)]],
        state_block_table__ssa_v0: pl.Tensor[[B_DYN, STATE_TABLE_COLUMNS_DYN], pl.INT32, pl.MemRef("mem_ddr_18", pl.const(0, pl.INT64), 0)],
        idx_wq_b__ssa_v0: pl.Tensor[[1024, 8192], pl.INT8, pl.MemRef("mem_ddr_19", pl.const(0, pl.INT64), 8388608)],
        idx_wq_b_scale__ssa_v0: pl.Tensor[[8192], pl.FP32, pl.MemRef("mem_ddr_20", pl.const(0, pl.INT64), 32768)],
        weights_proj__ssa_v0: pl.Tensor[[4096, 64], pl.BF16, pl.MemRef("mem_ddr_21", pl.const(0, pl.INT64), 524288)],
        hadamard_idx__ssa_v0: pl.Tensor[[128, 128], pl.BF16, pl.MemRef("mem_ddr_22", pl.const(0, pl.INT64), 32768)],
        inner_wkv__ssa_v0: pl.Tensor[[256, 4096], pl.BF16, pl.MemRef("mem_ddr_23", pl.const(0, pl.INT64), 2097152)],
        inner_wgate__ssa_v0: pl.Tensor[[256, 4096], pl.BF16, pl.MemRef("mem_ddr_24", pl.const(0, pl.INT64), 2097152)],
        inner_ape__ssa_v0: pl.Tensor[[4, 256], pl.FP32, pl.MemRef("mem_ddr_25", pl.const(0, pl.INT64), 4096)],
        inner_norm_w__ssa_v0: pl.Tensor[[128], pl.FP32, pl.MemRef("mem_ddr_26", pl.const(0, pl.INT64), 512)],
        inner_compress_state__ssa_v0: pl.InOut[pl.Tensor[[INNER_STATE_BLOCK_NUM_DYN, INNER_STATE_PAGE_ELEMENTS_DYN], pl.FP32, pl.MemRef("mem_ddr_27", pl.const(0, pl.INT64), 0)]],
        inner_state_block_table__ssa_v0: pl.Tensor[[B_DYN, INNER_STATE_TABLE_COLUMNS_DYN], pl.INT32, pl.MemRef("mem_ddr_28", pl.const(0, pl.INT64), 0)],
        kv_cache__ssa_v0: pl.InOut[pl.Tensor[[ORI_BLOCK_NUM_DYN, 32, 1, 512], pl.BF16, pl.MemRef("mem_ddr_29", pl.const(0, pl.INT64), 0)]],
        cmp_kv__ssa_v0: pl.InOut[pl.Tensor[[CMP_BLOCK_NUM_DYN, 32, 1, 512], pl.BF16, pl.MemRef("mem_ddr_30", pl.const(0, pl.INT64), 0)]],
        cmp_block_table__ssa_v0: pl.Tensor[[B_DYN, COMPRESSED_TABLE_COLUMNS_DYN], pl.INT32, pl.MemRef("mem_ddr_31", pl.const(0, pl.INT64), 0)],
        idx_kv_cache__ssa_v0: pl.InOut[pl.Tensor[[IDX_CACHE_BLOCK_NUM_DYN, INDEXER_PAGE_BYTES_DYN], pl.INT8, pl.MemRef("mem_ddr_32", pl.const(0, pl.INT64), 0)]],
        idx_block_table__ssa_v0: pl.Tensor[[B_DYN, INDEXER_TABLE_COLUMNS_DYN], pl.INT32, pl.MemRef("mem_ddr_33", pl.const(0, pl.INT64), 0)],
        ori_slot_mapping__ssa_v0: pl.Tensor[[T_DYN, 2], pl.INT32, pl.MemRef("mem_ddr_34", pl.const(0, pl.INT64), 0)],
        ori_block_table__ssa_v0: pl.Tensor[[B_DYN, ORIGINAL_TABLE_COLUMNS_DYN], pl.INT32, pl.MemRef("mem_ddr_35", pl.const(0, pl.INT64), 0)],
        cmp_slot_mapping__ssa_v0: pl.Tensor[[COMPRESSED_ROWS_DYN, 2], pl.INT32, pl.MemRef("mem_ddr_36", pl.const(0, pl.INT64), 0)],
        idx_slot_mapping__ssa_v0: pl.Tensor[[INDEXER_ROWS_DYN, 2], pl.INT32, pl.MemRef("mem_ddr_37", pl.const(0, pl.INT64), 0)],
        state_slot_mapping__ssa_v0: pl.Tensor[[T_DYN, 2], pl.INT32, pl.MemRef("mem_ddr_38", pl.const(0, pl.INT64), 0)],
        inner_state_slot_mapping__ssa_v0: pl.Tensor[[T_DYN, 2], pl.INT32, pl.MemRef("mem_ddr_39", pl.const(0, pl.INT64), 0)],
        position_ids__ssa_v0: pl.Tensor[[T_DYN], pl.INT64, pl.MemRef("mem_ddr_40", pl.const(0, pl.INT64), 0)],
        kv_seq_lens__ssa_v0: pl.Tensor[[B_DYN], pl.INT32, pl.MemRef("mem_ddr_41", pl.const(0, pl.INT64), 0)],
        cmp_query_start_loc__ssa_v0: pl.Tensor[[QUERY_BOUNDS_DYN], pl.INT32, pl.MemRef("mem_ddr_42", pl.const(0, pl.INT64), 0)],
        cmp_seq_lens__ssa_v0: pl.Tensor[[B_DYN], pl.INT32, pl.MemRef("mem_ddr_43", pl.const(0, pl.INT64), 0)],
        idx_query_start_loc__ssa_v0: pl.Tensor[[QUERY_BOUNDS_DYN], pl.INT32, pl.MemRef("mem_ddr_44", pl.const(0, pl.INT64), 0)],
        attn_sink__ssa_v0: pl.Tensor[[64], pl.FP32, pl.MemRef("mem_ddr_45", pl.const(0, pl.INT64), 256)],
        wo_a__ssa_v0: pl.Tensor[[8, 1024, 4096], pl.BF16, pl.MemRef("mem_ddr_46", pl.const(0, pl.INT64), 67108864)],
        wo_b__ssa_v0: pl.Tensor[[4096, 8192], pl.INT8, pl.MemRef("mem_ddr_47", pl.const(0, pl.INT64), 33554432)],
        wo_b_scale__ssa_v0: pl.Tensor[[4096], pl.FP32, pl.MemRef("mem_ddr_48", pl.const(0, pl.INT64), 16384)],
        idx_topk_scores__ssa_v0: pl.Out[pl.Tensor[[T_DYN, 512], pl.FP32, pl.MemRef("mem_ddr_49", pl.const(0, pl.INT64), 0)]],
        idx_topk__ssa_v0: pl.Out[pl.Tensor[[T_DYN, 512], pl.INT32, pl.MemRef("mem_ddr_50", pl.const(0, pl.INT64), 0)]],
        attn_out__ssa_v0: pl.Out[pl.Tensor[[T_DYN, 4096], pl.BF16, pl.MemRef("mem_ddr_51", pl.const(0, pl.INT64), 0)]],
    ) -> pl.Tensor[[T_DYN, 4096], pl.BF16]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        wb_blocks__ssa_v0: pl.Scalar[pl.INDEX] = T_DYN // 8
        idx_sin_signed__ssa_v0: pl.Tensor[[T_DYN, 64], pl.FP32, pl.MemRef("mem_ddr_52", pl.const(0, pl.INT64), 0)] = pl.tensor.create([T_DYN, 64], dtype=pl.FP32, layout=pl.TensorLayout.ND)
        cmp_row_offsets__ssa_v0: pl.Tensor[[B_DYN], pl.INT32, pl.MemRef("mem_ddr_53", pl.const(0, pl.INT64), 0)] = pl.tensor.create([B_DYN], dtype=pl.INT32, layout=pl.TensorLayout.ND)
        idx_row_offsets__ssa_v0: pl.Tensor[[B_DYN], pl.INT32, pl.MemRef("mem_ddr_54", pl.const(0, pl.INT64), 0)] = pl.tensor.create([B_DYN], dtype=pl.INT32, layout=pl.TensorLayout.ND)
        ret__tmp_v0: pl.Tuple[pl.Tensor[[T_DYN, 64], pl.FP32], pl.Scalar[pl.TASK_ID]] = pl.submit(
            self.csa_rope_sign,
            cmp_seq_lens__ssa_v0,
            cmp_query_start_loc__ssa_v0,
            cmp_row_offsets__ssa_v0,
            kv_seq_lens__ssa_v0,
            idx_query_start_loc__ssa_v0,
            idx_row_offsets__ssa_v0,
            idx_sin_signed__ssa_v0,
            T_DYN,
            freqs_sin__ssa_v0,
            attrs={
                "arg_directions": [pl.adir.input, pl.adir.input, pl.adir.output_existing, pl.adir.input, pl.adir.input, pl.adir.output_existing, pl.adir.output_existing, pl.adir.scalar, pl.adir.input]
            },
        )
        idx_sin_signed__rv_v2: pl.Tensor[[T_DYN, 64], pl.FP32, pl.MemRef("mem_ddr_55", pl.const(0, pl.INT64), 0)] = ret__tmp_v0[0]
        rope_tid__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0[1]
        q__ssa_v0: pl.Tensor[[T_DYN, 64, 512], pl.BF16, pl.MemRef("mem_ddr_56", pl.const(0, pl.INT64), 0)] = pl.tensor.create([T_DYN, 64, 512], dtype=pl.BF16, layout=pl.TensorLayout.ND)
        kv__ssa_v0: pl.Tensor[[T_DYN, 512], pl.BF16, pl.MemRef("mem_ddr_57", pl.const(0, pl.INT64), 0)] = pl.tensor.create([T_DYN, 512], dtype=pl.BF16, layout=pl.TensorLayout.ND)
        qr__ssa_v0: pl.Tensor[[T_DYN, 1024], pl.INT8, pl.MemRef("mem_ddr_58", pl.const(0, pl.INT64), 0)] = pl.tensor.create([T_DYN, 1024], dtype=pl.INT8, layout=pl.TensorLayout.ND)
        qr_scale__ssa_v0: pl.Tensor[[T_DYN, 1], pl.FP32, pl.MemRef("mem_ddr_59", pl.const(0, pl.INT64), 0)] = pl.tensor.create([T_DYN, 1], dtype=pl.FP32, layout=pl.TensorLayout.ND)
        position_ids_t1__ssa_v0: pl.Tensor[[T_DYN, 1], pl.INT64, pl.MemRef("mem_ddr_40", pl.const(0, pl.INT64), 0)] = pl.tensor.reshape(position_ids__ssa_v0, [T_DYN, 1])
        with pl.scope():
            late_dep__ssa_v0: pl.Scalar[pl.TASK_ID] = pl.system.task_dummy(deps=[rope_tid__ssa_v0])
            t_dim_inline198__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(x_normed_t__ssa_v0, 0)
            q_rope_sin_signed_inline197__ssa_v0: pl.Tensor[[t_dim_inline198__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_60", pl.const(0, pl.INT64), 0)] = pl.tensor.create(
                [t_dim_inline198__ssa_v0, 64], dtype=pl.FP32, layout=pl.TensorLayout.ND
            )
            q_rope_swap_idx_inline196__ssa_v0: pl.Tensor[[t_dim_inline198__ssa_v0, 64], pl.INT32, pl.MemRef("mem_ddr_61", pl.const(0, pl.INT64), 0)] = pl.tensor.create(
                [t_dim_inline198__ssa_v0, 64], dtype=pl.INT32, layout=pl.TensorLayout.ND
            )
            t_dim_inline1770__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(freqs_sin__ssa_v0, 0)
            rope_sin_view_inline1767__ssa_v0: pl.Tensor[[t_dim_inline1770__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_8", pl.const(0, pl.INT64), 0)] = pl.tensor.reshape(
                freqs_sin__ssa_v0, [t_dim_inline1770__ssa_v0, 64]
            )
            rope_sin_signed_view_inline1763__ssa_v0: pl.Tensor[[t_dim_inline1770__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_60", pl.const(0, pl.INT64), 0)] = pl.tensor.reshape(
                q_rope_sin_signed_inline197__ssa_v0, [t_dim_inline1770__ssa_v0, 64]
            )
            rope_swap_idx_view_inline1774__ssa_v0: pl.Tensor[[t_dim_inline1770__ssa_v0, 64], pl.INT32, pl.MemRef("mem_ddr_61", pl.const(0, pl.INT64), 0)] = pl.tensor.reshape(
                q_rope_swap_idx_inline196__ssa_v0, [t_dim_inline1770__ssa_v0, 64]
            )
            token_tiles_inline1782__ssa_v0: pl.Scalar[pl.INDEX] = (t_dim_inline1770__ssa_v0 + 7) // 8
            ret__tmp_v0_1: pl.Tuple[pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(
                self.q_rope_prepare_spmd,
                rope_sin_signed_view_inline1763__ssa_v0,
                rope_swap_idx_view_inline1774__ssa_v0,
                token_tiles_inline1782__ssa_v0,
                t_dim_inline1770__ssa_v0,
                rope_sin_view_inline1767__ssa_v0,
                core_num=pl.min(token_tiles_inline1782__ssa_v0, 48),
                allow_early_resolve=True,
                attrs={"arg_directions": [pl.adir.output_existing, pl.adir.output_existing, pl.adir.scalar, pl.adir.scalar, pl.adir.input]},
            )
            tid__ssa_v1: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_1[0]
            qr_i8_matmul_inline1790__ssa_v0: pl.Tensor[[512, 1024], pl.INT8, pl.MemRef("mem_ddr_62", pl.const(0, pl.INT64), 524288)] = pl.tensor.create(
                [512, 1024], dtype=pl.INT8, layout=pl.TensorLayout.ND
            )
            qr_scale_pad_store_inline1797__ssa_v0: pl.Tensor[[512, 1], pl.FP32, pl.MemRef("mem_ddr_63", pl.const(0, pl.INT64), 2048)] = pl.tensor.create(
                [512, 1], dtype=pl.FP32, layout=pl.TensorLayout.ND, manual_dep=True
            )
            t_dim_inline4_inline1792__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(x_normed_t__ssa_v0, 0)
            for tile_base_inline3_inline1791__idx_v0, (qr_i8_matmul_inline1790__iter_v1, qr_scale_pad_store_inline1797__iter_v1) in pl.range(
                0,
                t_dim_inline4_inline1792__ssa_v0,
                512,
                init_values=(qr_i8_matmul_inline1790__ssa_v0, qr_scale_pad_store_inline1797__ssa_v0),
                attrs={"iter_arg_rebind_0": False, "iter_arg_rebind_1": False},
            ):
                tile_rows_inline2_inline1793__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(t_dim_inline4_inline1792__ssa_v0 - tile_base_inline3_inline1791__idx_v0, 512)
                with pl.scope():
                    qr_t_matmul_inline1_inline1795__ssa_v0: pl.Scalar[pl.INDEX] = (tile_rows_inline2_inline1793__ssa_v0 + 15) // 16 * 16
                    qr_fp32_inline0_inline1798__ssa_v0: pl.Tensor[[qr_t_matmul_inline1_inline1795__ssa_v0, 1024], pl.FP32, pl.MemRef("mem_ddr_64", pl.const(0, pl.INT64), 0)] = pl.tensor.create(
                        [qr_t_matmul_inline1_inline1795__ssa_v0, 1024], dtype=pl.FP32, layout=pl.TensorLayout.ND
                    )
                    qa_tokens_inline3009__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(x_normed_t__ssa_v0, 0)
                    x_view_inline3007__ssa_v0: pl.Tensor[[qa_tokens_inline3009__ssa_v0, 4096], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)] = pl.tensor.reshape(
                        x_normed_t__ssa_v0, [qa_tokens_inline3009__ssa_v0, 4096]
                    )
                    qr_t_matmul_inline2997__ssa_v0: pl.Scalar[pl.INDEX] = (tile_rows_inline2_inline1793__ssa_v0 + 15) // 16 * 16
                    qr_full_rows_inline3006__ssa_v0: pl.Scalar[pl.INDEX] = tile_rows_inline2_inline1793__ssa_v0 // 64 * 64
                    qr_fp32_inline0_inline1798__rv_v2: pl.Tensor[[qr_t_matmul_inline1_inline1795__ssa_v0, 1024], pl.FP32, pl.MemRef("mem_ddr_65", pl.const(0, pl.INT64), 0)] = self.qr_proj_seed(
                        qr_fp32_inline0_inline1798__ssa_v0, qr_t_matmul_inline2997__ssa_v0, attrs={"arg_directions": [pl.adir.inout, pl.adir.scalar]}
                    )
                    ret__tmp_v0_2: pl.Tuple[pl.Tensor[[qr_t_matmul_inline1_inline1795__ssa_v0, 1024], pl.FP32], pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(
                        self.qr_proj_matmul_spmd,
                        qr_fp32_inline0_inline1798__rv_v2,
                        qr_full_rows_inline3006__ssa_v0,
                        tile_base_inline3_inline1791__idx_v0,
                        tile_rows_inline2_inline1793__ssa_v0,
                        x_view_inline3007__ssa_v0,
                        wq_a__ssa_v0,
                        qr_t_matmul_inline2997__ssa_v0,
                        core_num=32,
                        allow_early_resolve=True,
                        attrs={"arg_directions": [pl.adir.inout, pl.adir.scalar, pl.adir.scalar, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.scalar]},
                    )
                    qr_fp32_inline0_inline1798__rv_v10: pl.Tensor[[qr_t_matmul_inline1_inline1795__ssa_v0, 1024], pl.FP32, pl.MemRef("mem_ddr_66", pl.const(0, pl.INT64), 0)] = ret__tmp_v0_2[0]
                    tid__ssa_v2: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_2[1]
                    t_dim_inline3071__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(qr__ssa_v0, 0)
                    qr_view_inline3057__ssa_v0: pl.Tensor[[t_dim_inline3071__ssa_v0, 1024], pl.INT8, pl.MemRef("mem_ddr_58", pl.const(0, pl.INT64), 0)] = pl.tensor.reshape(
                        qr__ssa_v0, [t_dim_inline3071__ssa_v0, 1024]
                    )
                    qr_scale_view_inline3048__ssa_v0: pl.Tensor[[t_dim_inline3071__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_59", pl.const(0, pl.INT64), 0)] = pl.tensor.reshape(
                        qr_scale__ssa_v0, [t_dim_inline3071__ssa_v0, 1]
                    )
                    qr_token_tiles_inline3050__ssa_v0: pl.Scalar[pl.INDEX] = (tile_rows_inline2_inline1793__ssa_v0 + 7) // 8
                    ret__tmp_v0_3: pl.Tuple[pl.Tensor[[512, 1], pl.FP32], pl.Tensor[[512, 1024], pl.INT8], pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(
                        self.qr_rms_norm_quant_spmd,
                        tile_rows_inline2_inline1793__ssa_v0,
                        tile_base_inline3_inline1791__idx_v0,
                        qr_fp32_inline0_inline1798__rv_v10,
                        gamma_cq__ssa_v0,
                        qr_scale_pad_store_inline1797__iter_v1,
                        qr_scale_view_inline3048__ssa_v0,
                        qr_i8_matmul_inline1790__iter_v1,
                        qr_view_inline3057__ssa_v0,
                        core_num=qr_token_tiles_inline3050__ssa_v0,
                        allow_early_resolve=True,
                        attrs={"arg_directions": [pl.adir.scalar, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.inout, pl.adir.inout, pl.adir.inout, pl.adir.inout]},
                    )
                    qr_scale_pad_store_inline1797__ssa_v3: pl.Tensor[[512, 1], pl.FP32, pl.MemRef("mem_ddr_67", pl.const(0, pl.INT64), 2048)] = ret__tmp_v0_3[0]
                    qr_i8_matmul_inline1790__rv_v4: pl.Tensor[[512, 1024], pl.INT8, pl.MemRef("mem_ddr_68", pl.const(0, pl.INT64), 524288)] = ret__tmp_v0_3[1]
                    tid__ssa_v3: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_3[2]
                qr_i8_matmul_inline1790__rv_v2, qr_scale_pad_store_inline1797__rv_v2 = pl.yield_(qr_i8_matmul_inline1790__rv_v4, qr_scale_pad_store_inline1797__ssa_v3)
            q_seq_dep_inline1799__ssa_v0: pl.Scalar[pl.TASK_ID] = pl.system.task_dummy(deps=[])
            t_dim_inline11_inline1801__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(x_normed_t__ssa_v0, 0)
            for tile_base_inline10_inline1796__idx_v0 in pl.range(0, t_dim_inline11_inline1801__ssa_v0, 512):
                tile_rows_inline9_inline1800__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(t_dim_inline11_inline1801__ssa_v0 - tile_base_inline10_inline1796__idx_v0, 512)
                with pl.scope():
                    qproj_t_matmul_inline8_inline1802__ssa_v0: pl.Scalar[pl.INDEX] = (tile_rows_inline9_inline1800__ssa_v0 + 15) // 16 * 16
                    q_proj_i32_inline6_inline1803__ssa_v0: pl.Tensor[[qproj_t_matmul_inline8_inline1802__ssa_v0, 32768], pl.INT32, pl.MemRef("mem_ddr_71", pl.const(0, pl.INT64), 0)] = (
                        pl.tensor.create([qproj_t_matmul_inline8_inline1802__ssa_v0, 32768], dtype=pl.INT32, layout=pl.TensorLayout.ND)
                    )
                    qproj_t_matmul_inline3092__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(q_proj_i32_inline6_inline1803__ssa_v0, 0)
                    qproj_full_rows_inline3094__ssa_v0: pl.Scalar[pl.INDEX] = tile_rows_inline9_inline1800__ssa_v0 // 64 * 64
                    ret__tmp_v0_4: pl.Tuple[pl.Tensor[[qproj_t_matmul_inline8_inline1802__ssa_v0, 32768], pl.INT32], pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(
                        self.qproj_matmul_spmd,
                        q_proj_i32_inline6_inline1803__ssa_v0,
                        qproj_full_rows_inline3094__ssa_v0,
                        qr_i8_matmul_inline1790__rv_v2,
                        wq_b__ssa_v0,
                        qproj_t_matmul_inline3092__ssa_v0,
                        tile_rows_inline9_inline1800__ssa_v0,
                        deps=[q_seq_dep_inline1799__ssa_v0],
                        core_num=24,
                        attrs={"arg_directions": [pl.adir.inout, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.scalar, pl.adir.scalar]},
                    )
                    q_proj_i32_inline6_inline1803__rv_v2: pl.Tensor[[qproj_t_matmul_inline8_inline1802__ssa_v0, 32768], pl.INT32, pl.MemRef("mem_ddr_72", pl.const(0, pl.INT64), 0)] = ret__tmp_v0_4[0]
                    qproj_tid_inline3095__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_4[1]
                    q_proj_i32_inline6_inline1803__ssa_v9: pl.Tensor[[qproj_t_matmul_inline8_inline1802__ssa_v0, 32768], pl.INT32, pl.MemRef("mem_ddr_73", pl.const(0, pl.INT64), 0)] = (
                        q_proj_i32_inline6_inline1803__rv_v2
                    )
                    t_dim_inline3159__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(q__ssa_v0, 0)
                    q_flat_inline3174__ssa_v0: pl.Tensor[[t_dim_inline3159__ssa_v0, 32768], pl.BF16, pl.MemRef("mem_ddr_56", pl.const(0, pl.INT64), 0)] = pl.tensor.reshape(
                        q__ssa_v0, [t_dim_inline3159__ssa_v0, 32768]
                    )
                    ret__tmp_v0_5: pl.Tuple[pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(
                        self.qproj_dequant_rms_nope_rope_spmd,
                        q_flat_inline3174__ssa_v0,
                        tile_rows_inline9_inline1800__ssa_v0,
                        tile_base_inline10_inline1796__idx_v0,
                        qr_scale_pad_store_inline1797__rv_v2,
                        freqs_cos__ssa_v0,
                        q_rope_sin_signed_inline197__ssa_v0,
                        q_rope_swap_idx_inline196__ssa_v0,
                        q_proj_i32_inline6_inline1803__ssa_v9,
                        wq_b_scale__ssa_v0,
                        core_num=48,
                        allow_early_resolve=True,
                        attrs={"arg_directions": [pl.adir.inout, pl.adir.scalar, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input]},
                    )
                    tid__ssa_v4: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_5[0]
            t_dim_inline1873__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(x_normed_t__ssa_v0, 0)
            for tile_base_inline1887__idx_v0 in pl.range(0, t_dim_inline1873__ssa_v0, 512):
                tile_rows_inline1877__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(t_dim_inline1873__ssa_v0 - tile_base_inline1887__idx_v0, 512)
                with pl.scope():
                    x_view_inline1855__ssa_v0: pl.Tensor[[t_dim_inline1873__ssa_v0, 4096], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)] = pl.tensor.reshape(
                        x_normed_t__ssa_v0, [t_dim_inline1873__ssa_v0, 4096]
                    )
                    t_matmul_inline1893__ssa_v0: pl.Scalar[pl.INDEX] = (tile_rows_inline1877__ssa_v0 + 15) // 16 * 16
                    kv_full_rows_inline1888__ssa_v0: pl.Scalar[pl.INDEX] = tile_rows_inline1877__ssa_v0 // 64 * 64
                    kv_m_groups_inline1894__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(pl.max(tile_rows_inline1877__ssa_v0 // 128, 1), 3)
                    kv_fp32_inline1850__ssa_v0: pl.Tensor[[t_matmul_inline1893__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_74", pl.const(0, pl.INT64), 0)] = pl.tensor.create(
                        [t_matmul_inline1893__ssa_v0, 512], dtype=pl.FP32, layout=pl.TensorLayout.ND
                    )
                    kv_fp32_inline1850__rv_v2: pl.Tensor[[t_matmul_inline1893__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_75", pl.const(0, pl.INT64), 0)] = self.kv_proj_seed(
                        kv_fp32_inline1850__ssa_v0, t_matmul_inline1893__ssa_v0, attrs={"arg_directions": [pl.adir.inout, pl.adir.scalar]}
                    )
                    if tile_rows_inline1877__ssa_v0 == 240:
                        ret__tmp_v0_6: pl.Tuple[pl.Tensor[[t_matmul_inline1893__ssa_v0, 512], pl.FP32], pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(
                            self.kv_proj_native_240_spmd,
                            kv_fp32_inline1850__rv_v2,
                            x_normed_t__ssa_v0,
                            tile_base_inline1887__idx_v0,
                            wkv__ssa_v0,
                            deps=[late_dep__ssa_v0],
                            core_num=16,
                            attrs={"arg_directions": [pl.adir.inout, pl.adir.input, pl.adir.scalar, pl.adir.input]},
                        )
                        kv_fp32_inline1850__rv_v10: pl.Tensor[[t_matmul_inline1893__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_76", pl.const(0, pl.INT64), 0)] = ret__tmp_v0_6[0]
                        tid__ssa_v5: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_6[1]
                        kv_fp32_inline1850__phi_v18: pl.Tensor[[t_matmul_inline1893__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_78", pl.const(0, pl.INT64), 0)] = pl.yield_(kv_fp32_inline1850__rv_v10)
                    else:
                        ret__tmp_v0_7: pl.Tuple[pl.Tensor[[t_matmul_inline1893__ssa_v0, 512], pl.FP32], pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(
                            self.kv_proj_matmul_spmd,
                            kv_m_groups_inline1894__ssa_v0,
                            kv_fp32_inline1850__rv_v2,
                            kv_full_rows_inline1888__ssa_v0,
                            tile_base_inline1887__idx_v0,
                            x_view_inline1855__ssa_v0,
                            wkv__ssa_v0,
                            t_matmul_inline1893__ssa_v0,
                            tile_rows_inline1877__ssa_v0,
                            deps=[late_dep__ssa_v0],
                            core_num=kv_m_groups_inline1894__ssa_v0 * 8,
                            attrs={"arg_directions": [pl.adir.scalar, pl.adir.inout, pl.adir.scalar, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.scalar, pl.adir.scalar]},
                        )
                        kv_fp32_inline1850__rv_v16: pl.Tensor[[t_matmul_inline1893__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_77", pl.const(0, pl.INT64), 0)] = ret__tmp_v0_7[0]
                        _kv_tid_inline1896__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_7[1]
                        kv_fp32_inline1850__phi_v18: pl.Tensor[[t_matmul_inline1893__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_78", pl.const(0, pl.INT64), 0)] = pl.yield_(kv_fp32_inline1850__rv_v16)
                    kv_view_inline1915__ssa_v0: pl.Tensor[[t_dim_inline1873__ssa_v0, 512], pl.BF16, pl.MemRef("mem_ddr_57", pl.const(0, pl.INT64), 0)] = pl.tensor.reshape(
                        kv__ssa_v0, [t_dim_inline1873__ssa_v0, 512]
                    )
                    kv_token_tiles_inline1872__ssa_v0: pl.Scalar[pl.INDEX] = (tile_rows_inline1877__ssa_v0 + 31) // 32
                    self.kv_rms_norm_rope_spmd(
                        tile_rows_inline1877__ssa_v0,
                        tile_base_inline1887__idx_v0,
                        kv_fp32_inline1850__phi_v18,
                        kv_view_inline1915__ssa_v0,
                        gamma_ckv__ssa_v0,
                        freqs_cos__ssa_v0,
                        q_rope_sin_signed_inline197__ssa_v0,
                        q_rope_swap_idx_inline196__ssa_v0,
                        attrs={
                            "arg_directions": [pl.adir.scalar, pl.adir.scalar, pl.adir.input, pl.adir.inout, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input],
                            "core_num": kv_token_tiles_inline1872__ssa_v0,
                            "sync_start": True,
                        },
                    )
            kv_cache_flat__ssa_v0: pl.Tensor[[ORI_BLOCK_NUM_DYN * pl.const(32, pl.INDEX), 512], pl.BF16, pl.MemRef("mem_ddr_29", pl.const(0, pl.INT64), 0)] = pl.tensor.reshape(
                kv_cache__ssa_v0, [ORI_BLOCK_NUM_DYN * 32, 512]
            )
            self.csa_cache_writeback_spmd(
                kv_cache_flat__ssa_v0,
                wb_blocks__ssa_v0,
                ori_slot_mapping__ssa_v0,
                kv__ssa_v0,
                attrs={"arg_directions": [pl.adir.output_existing, pl.adir.scalar, pl.adir.input, pl.adir.input], "core_num": 8},
            )
            cmp_out__ssa_v0: pl.Tensor[[T_DYN, 512], pl.FP32, pl.MemRef("mem_ddr_79", pl.const(0, pl.INT64), 0)] = pl.tensor.create([T_DYN, 512], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            pooled_kv_inline205__ssa_v0: pl.Tensor[[384, 512], pl.FP32, pl.MemRef("mem_ddr_80", pl.const(0, pl.INT64), 786432)] = pl.tensor.create([384, 512], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            kv_proj_pad_inline203__ssa_v0: pl.Tensor[[384, 1024], pl.FP32, pl.MemRef("mem_ddr_81", pl.const(0, pl.INT64), 1572864)] = pl.tensor.create(
                [384, 1024], dtype=pl.FP32, layout=pl.TensorLayout.ND
            )
            score_proj_pad_inline201__ssa_v0: pl.Tensor[[384, 1024], pl.FP32, pl.MemRef("mem_ddr_82", pl.const(0, pl.INT64), 1572864)] = pl.tensor.create(
                [384, 1024], dtype=pl.FP32, layout=pl.TensorLayout.ND
            )
            bs_inline481_inline1944__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(x_normed_t__ssa_v0, 0)
            t_matmul_inline483_inline1947__ssa_v0: pl.Scalar[pl.INDEX] = (bs_inline481_inline1944__ssa_v0 + 63) // 64 * 64
            x_flat_inline484_inline1950__ssa_v0: pl.Tensor[[T_DYN, 4096], pl.BF16, pl.MemRef("mem_ddr_83", pl.const(0, pl.INT64), 0)] = x_normed_t__ssa_v0
            cmp4_kv_proj_pad_inline485_inline1955__ssa_v0: pl.Tensor[[384, 1024], pl.FP32, pl.MemRef("mem_ddr_84", pl.const(0, pl.INT64), 1572864)] = kv_proj_pad_inline203__ssa_v0
            cmp4_score_proj_pad_inline488_inline1934__ssa_v0: pl.Tensor[[384, 1024], pl.FP32, pl.MemRef("mem_ddr_85", pl.const(0, pl.INT64), 1572864)] = score_proj_pad_inline201__ssa_v0
            ret__tmp_v0_8: pl.Tuple[pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(
                self.kv_score_proj_spmd,
                cmp4_kv_proj_pad_inline485_inline1955__ssa_v0,
                cmp4_score_proj_pad_inline488_inline1934__ssa_v0,
                t_matmul_inline483_inline1947__ssa_v0,
                bs_inline481_inline1944__ssa_v0,
                x_flat_inline484_inline1950__ssa_v0,
                cmp_wkv__ssa_v0,
                cmp_wgate__ssa_v0,
                deps=[late_dep__ssa_v0],
                core_num=24,
                attrs={"arg_directions": [pl.adir.output_existing, pl.adir.output_existing, pl.adir.scalar, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.input]},
            )
            _kv_score_tid_inline489_inline1964__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_8[0]
            projection_tid_inline1933__ssa_v0: pl.Scalar[pl.TASK_ID] = _kv_score_tid_inline489_inline1964__ssa_v0
            b_dim_inline518_inline1939__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(state_block_table__ssa_v0, 0)
            bs_inline522_inline1932__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(position_ids__ssa_v0, 0)
            s_dim_inline513_inline1935__ssa_v0: pl.Scalar[pl.INDEX] = bs_inline522_inline1932__ssa_v0 // b_dim_inline518_inline1939__ssa_v0
            cmp4_kv_proj_pad_inline530_inline1957__ssa_v0: pl.Tensor[[384, 1024], pl.FP32, pl.MemRef("mem_ddr_86", pl.const(0, pl.INT64), 1572864)] = kv_proj_pad_inline203__ssa_v0
            cmp4_score_proj_pad_inline510_inline1929__ssa_v0: pl.Tensor[[384, 1024], pl.FP32, pl.MemRef("mem_ddr_87", pl.const(0, pl.INT64), 1572864)] = score_proj_pad_inline201__ssa_v0
            _kv_score_tid_inline534_inline1958__ssa_v0: pl.Scalar[pl.TASK_ID] = projection_tid_inline1933__ssa_v0
            pool_workers_inline512_inline1960__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(b_dim_inline518_inline1939__ssa_v0, 48)
            ret__tmp_v0_9: pl.Tuple[pl.Tensor[[384, 512], pl.FP32], pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(
                self.scatter_softmax_pool_spmd,
                pooled_kv_inline205__ssa_v0,
                b_dim_inline518_inline1939__ssa_v0,
                pool_workers_inline512_inline1960__ssa_v0,
                position_ids__ssa_v0,
                s_dim_inline513_inline1935__ssa_v0,
                state_block_table__ssa_v0,
                compress_state__ssa_v0,
                cmp4_kv_proj_pad_inline530_inline1957__ssa_v0,
                cmp4_score_proj_pad_inline510_inline1929__ssa_v0,
                cmp_ape__ssa_v0,
                deps=[_kv_score_tid_inline534_inline1958__ssa_v0],
                core_num=pool_workers_inline512_inline1960__ssa_v0,
                allow_early_resolve=True,
                attrs={
                    "arg_directions": [
                        pl.adir.output_existing,
                        pl.adir.scalar,
                        pl.adir.scalar,
                        pl.adir.input,
                        pl.adir.scalar,
                        pl.adir.input,
                        pl.adir.input,
                        pl.adir.input,
                        pl.adir.input,
                        pl.adir.input,
                    ]
                },
            )
            pooled_kv_inline205__rv_v2: pl.Tensor[[384, 512], pl.FP32, pl.MemRef("mem_ddr_88", pl.const(0, pl.INT64), 786432)] = ret__tmp_v0_9[0]
            pool_tid_inline505_inline1928__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_9[1]
            pool_tid_inline1942__ssa_v0: pl.Scalar[pl.TASK_ID] = pool_tid_inline505_inline1928__ssa_v0
            projection_ready_tid_inline1926__ssa_v0: pl.Scalar[pl.TASK_ID] = _kv_score_tid_inline534_inline1958__ssa_v0
            pool_tid_inline200__ssa_v0: pl.Scalar[pl.TASK_ID] = pool_tid_inline1942__ssa_v0
            kv_score_tid_inline204__ssa_v0: pl.Scalar[pl.TASK_ID] = projection_ready_tid_inline1926__ssa_v0
            b_dim_inline2037__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(state_block_table__ssa_v0, 0)
            bs_inline2047__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(position_ids__ssa_v0, 0)
            s_dim_inline2013__ssa_v0: pl.Scalar[pl.INDEX] = bs_inline2047__ssa_v0 // b_dim_inline2037__ssa_v0
            rms_blocks_inline2021__ssa_v0: pl.Scalar[pl.INDEX] = (bs_inline2047__ssa_v0 + 15) // 16
            cmp_block_num_inline2008__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(cmp_kv__ssa_v0, 0)
            kv_flat_inline2020__ssa_v0: pl.Tensor[[T_DYN, 512], pl.FP32, pl.MemRef("mem_ddr_89", pl.const(0, pl.INT64), 0)] = cmp_out__ssa_v0
            cmp_kv_cache_flat_inline2016__ssa_v0: pl.Tensor[[cmp_block_num_inline2008__ssa_v0 * pl.const(32, pl.INDEX), 512], pl.BF16, pl.MemRef("mem_ddr_30", pl.const(0, pl.INT64), 0)] = (
                pl.tensor.reshape(cmp_kv__ssa_v0, [cmp_block_num_inline2008__ssa_v0 * 32, 512])
            )
            cmp4_kv_proj_pad_inline2035__ssa_v0: pl.Tensor[[384, 1024], pl.FP32, pl.MemRef("mem_ddr_90", pl.const(0, pl.INT64), 1572864)] = kv_proj_pad_inline203__ssa_v0
            cmp4_score_proj_pad_inline2039__ssa_v0: pl.Tensor[[384, 1024], pl.FP32, pl.MemRef("mem_ddr_91", pl.const(0, pl.INT64), 1572864)] = score_proj_pad_inline201__ssa_v0
            commit_workers_inline2011__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(b_dim_inline2037__ssa_v0, 48)
            ret__tmp_v0_10: pl.Tuple[pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(
                self.compress_state_commit_spmd,
                compress_state__ssa_v0,
                b_dim_inline2037__ssa_v0,
                commit_workers_inline2011__ssa_v0,
                s_dim_inline2013__ssa_v0,
                state_slot_mapping__ssa_v0,
                position_ids__ssa_v0,
                cmp4_kv_proj_pad_inline2035__ssa_v0,
                cmp4_score_proj_pad_inline2039__ssa_v0,
                cmp_ape__ssa_v0,
                deps=[pool_tid_inline200__ssa_v0, pool_tid_inline200__ssa_v0],
                core_num=commit_workers_inline2011__ssa_v0,
                attrs={"arg_directions": [pl.adir.inout, pl.adir.scalar, pl.adir.scalar, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input]},
            )
            tid__ssa_v6: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_10[0]
            normed_kv_inline2015__ssa_v0: pl.Tensor[[384, 512], pl.FP32, pl.MemRef("mem_ddr_92", pl.const(0, pl.INT64), 786432)] = pl.tensor.create(
                [384, 512], dtype=pl.FP32, layout=pl.TensorLayout.ND
            )
            norm_w_2d_inline1999__ssa_v0: pl.Tensor[[1, 512], pl.FP32, pl.MemRef("mem_ddr_16", pl.const(0, pl.INT64), 2048)] = pl.tensor.reshape(cmp_norm_w__ssa_v0, [1, 512])
            ret__tmp_v0_11: pl.Tuple[pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(
                self.rmsnorm_rope_cache_write_spmd,
                bs_inline2047__ssa_v0,
                position_ids__ssa_v0,
                cmp_row_offsets__ssa_v0,
                cmp_freqs_cos__ssa_v0,
                cmp_freqs_sin__ssa_v0,
                pooled_kv_inline205__rv_v2,
                normed_kv_inline2015__ssa_v0,
                norm_w_2d_inline1999__ssa_v0,
                cmp_kv_cache_flat_inline2016__ssa_v0,
                kv_flat_inline2020__ssa_v0,
                cmp_slot_mapping__ssa_v0,
                deps=[pool_tid_inline200__ssa_v0, pool_tid_inline200__ssa_v0],
                core_num=rms_blocks_inline2021__ssa_v0,
                allow_early_resolve=True,
                attrs={
                    "arg_directions": [
                        pl.adir.scalar,
                        pl.adir.input,
                        pl.adir.input,
                        pl.adir.input,
                        pl.adir.input,
                        pl.adir.input,
                        pl.adir.inout,
                        pl.adir.input,
                        pl.adir.output_existing,
                        pl.adir.output_existing,
                        pl.adir.input,
                    ]
                },
            )
            cache_write_tid_inline1997__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_11[0]
            cmp_out__ssa_v1: pl.Tensor[[T_DYN, 512], pl.FP32, pl.MemRef("mem_ddr_93", pl.const(0, pl.INT64), 0)] = cmp_out__ssa_v0
            cmp_kv_score_tid__ssa_v0: pl.Scalar[pl.TASK_ID] = kv_score_tid_inline204__ssa_v0
            idx_kv_unused__ssa_v0: pl.Tensor[[T_DYN, 128], pl.FP32, pl.MemRef("mem_ddr_94", pl.const(0, pl.INT64), 0)] = pl.tensor.create([T_DYN, 128], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            normed_kv_inline212__ssa_v0: pl.Tensor[[384, 128], pl.BF16, pl.MemRef("mem_ddr_95", pl.const(0, pl.INT64), 98304)] = pl.tensor.create([384, 128], dtype=pl.BF16, layout=pl.TensorLayout.ND)
            kv_proj_pad_inline2105__ssa_v0: pl.Tensor[[384, 256], pl.FP32, pl.MemRef("mem_ddr_96", pl.const(0, pl.INT64), 393216)] = pl.tensor.create(
                [384, 256], dtype=pl.FP32, layout=pl.TensorLayout.ND
            )
            score_proj_pad_inline2117__ssa_v0: pl.Tensor[[384, 256], pl.FP32, pl.MemRef("mem_ddr_97", pl.const(0, pl.INT64), 393216)] = pl.tensor.create(
                [384, 256], dtype=pl.FP32, layout=pl.TensorLayout.ND
            )
            pooled_kv_inline2143__ssa_v0: pl.Tensor[[384, 128], pl.FP32, pl.MemRef("mem_ddr_98", pl.const(0, pl.INT64), 196608)] = pl.tensor.create(
                [384, 128], dtype=pl.FP32, layout=pl.TensorLayout.ND
            )
            bs_inline373_inline2096__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(x_normed_t__ssa_v0, 0)
            t_matmul_inline374_inline2127__ssa_v0: pl.Scalar[pl.INDEX] = (bs_inline373_inline2096__ssa_v0 + 15) // 16 * 16
            x_flat_inline377_inline2152__ssa_v0: pl.Tensor[[T_DYN, 4096], pl.BF16, pl.MemRef("mem_ddr_99", pl.const(0, pl.INT64), 0)] = x_normed_t__ssa_v0
            ret__tmp_v0_12: pl.Tuple[pl.Tensor[[384, 256], pl.FP32], pl.Tensor[[384, 256], pl.FP32], pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(
                self.kv_score_proj_spmd_0,
                kv_proj_pad_inline2105__ssa_v0,
                score_proj_pad_inline2117__ssa_v0,
                t_matmul_inline374_inline2127__ssa_v0,
                bs_inline373_inline2096__ssa_v0,
                x_flat_inline377_inline2152__ssa_v0,
                inner_wkv__ssa_v0,
                inner_wgate__ssa_v0,
                deps=[late_dep__ssa_v0, late_dep__ssa_v0],
                core_num=24,
                attrs={"arg_directions": [pl.adir.output_existing, pl.adir.output_existing, pl.adir.scalar, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.input]},
            )
            kv_proj_pad_inline2105__rv_v2: pl.Tensor[[384, 256], pl.FP32, pl.MemRef("mem_ddr_100", pl.const(0, pl.INT64), 393216)] = ret__tmp_v0_12[0]
            score_proj_pad_inline2117__rv_v2: pl.Tensor[[384, 256], pl.FP32, pl.MemRef("mem_ddr_101", pl.const(0, pl.INT64), 393216)] = ret__tmp_v0_12[1]
            _kv_score_tid_inline383_inline2129__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_12[2]
            projection_tid_inline2093__ssa_v0: pl.Scalar[pl.TASK_ID] = _kv_score_tid_inline383_inline2129__ssa_v0
            b_dim_inline454_inline2092__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(inner_state_block_table__ssa_v0, 0)
            bs_inline417_inline2091__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(position_ids__ssa_v0, 0)
            s_dim_inline421_inline2137__ssa_v0: pl.Scalar[pl.INDEX] = bs_inline417_inline2091__ssa_v0 // b_dim_inline454_inline2092__ssa_v0
            rms_blocks_inline427_inline2130__ssa_v0: pl.Scalar[pl.INDEX] = (bs_inline417_inline2091__ssa_v0 + 15) // 16
            _kv_score_tid_inline468_inline2090__ssa_v0: pl.Scalar[pl.TASK_ID] = projection_tid_inline2093__ssa_v0
            window_values_inline429_inline2121__ssa_v0: pl.Tensor[[384, 128], pl.FP32, pl.MemRef("mem_ddr_102", pl.const(0, pl.INT64), 196608)] = pl.tensor.create(
                [384, 128], dtype=pl.FP32, layout=pl.TensorLayout.ND
            )
            window_scores_inline441_inline2089__ssa_v0: pl.Tensor[[384, 128], pl.FP32, pl.MemRef("mem_ddr_103", pl.const(0, pl.INT64), 196608)] = pl.tensor.create(
                [384, 128], dtype=pl.FP32, layout=pl.TensorLayout.ND
            )
            pool_workers_inline436_inline2082__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(b_dim_inline454_inline2092__ssa_v0, 48)
            ret__tmp_v0_13: pl.Tuple[pl.Tensor[[384, 128], pl.FP32], pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(
                self.scatter_softmax_pool_spmd_0,
                pooled_kv_inline2143__ssa_v0,
                window_scores_inline441_inline2089__ssa_v0,
                window_values_inline429_inline2121__ssa_v0,
                b_dim_inline454_inline2092__ssa_v0,
                pool_workers_inline436_inline2082__ssa_v0,
                position_ids__ssa_v0,
                s_dim_inline421_inline2137__ssa_v0,
                inner_state_block_table__ssa_v0,
                inner_compress_state__ssa_v0,
                kv_proj_pad_inline2105__rv_v2,
                score_proj_pad_inline2117__rv_v2,
                inner_ape__ssa_v0,
                deps=[_kv_score_tid_inline468_inline2090__ssa_v0],
                core_num=pool_workers_inline436_inline2082__ssa_v0,
                attrs={
                    "arg_directions": [
                        pl.adir.output_existing,
                        pl.adir.inout,
                        pl.adir.inout,
                        pl.adir.scalar,
                        pl.adir.scalar,
                        pl.adir.input,
                        pl.adir.scalar,
                        pl.adir.input,
                        pl.adir.input,
                        pl.adir.input,
                        pl.adir.input,
                        pl.adir.input,
                    ]
                },
            )
            pooled_kv_inline2143__rv_v2: pl.Tensor[[384, 128], pl.FP32, pl.MemRef("mem_ddr_104", pl.const(0, pl.INT64), 196608)] = ret__tmp_v0_13[0]
            pool_tid_inline423_inline2079__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_13[1]
            commit_workers_inline450_inline2077__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(b_dim_inline454_inline2092__ssa_v0, 48)
            ret__tmp_v0_14: pl.Tuple[pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(
                self.compress_state_commit_spmd_0,
                inner_compress_state__ssa_v0,
                b_dim_inline454_inline2092__ssa_v0,
                commit_workers_inline450_inline2077__ssa_v0,
                s_dim_inline421_inline2137__ssa_v0,
                inner_state_slot_mapping__ssa_v0,
                position_ids__ssa_v0,
                kv_proj_pad_inline2105__rv_v2,
                score_proj_pad_inline2117__rv_v2,
                inner_ape__ssa_v0,
                deps=[pool_tid_inline423_inline2079__ssa_v0],
                core_num=commit_workers_inline450_inline2077__ssa_v0,
                attrs={"arg_directions": [pl.adir.inout, pl.adir.scalar, pl.adir.scalar, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input]},
            )
            tid__ssa_v7: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_14[0]
            norm_w_2d_inline470_inline2095__ssa_v0: pl.Tensor[[1, 128], pl.FP32, pl.MemRef("mem_ddr_26", pl.const(0, pl.INT64), 512)] = pl.tensor.reshape(inner_norm_w__ssa_v0, [1, 128])
            ret__tmp_v0_15: pl.Tuple[pl.Tensor[[384, 128], pl.BF16], pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(
                self.indexer_boundary_init_spmd,
                normed_kv_inline212__ssa_v0,
                deps=[pool_tid_inline423_inline2079__ssa_v0],
                core_num=b_dim_inline454_inline2092__ssa_v0,
                attrs={"arg_directions": [pl.adir.output_existing]},
            )
            normed_kv_inline212__ssa_v1: pl.Tensor[[384, 128], pl.BF16, pl.MemRef("mem_ddr_105", pl.const(0, pl.INT64), 98304)] = ret__tmp_v0_15[0]
            boundary_init_tid_inline471_inline2073__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_15[1]
            rms_workers_inline472_inline2070__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(rms_blocks_inline427_inline2130__ssa_v0, 2)
            ret__tmp_v0_16: pl.Tuple[pl.Tensor[[384, 128], pl.BF16], pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(
                self.rmsnorm_rope_spmd,
                normed_kv_inline212__ssa_v1,
                rms_blocks_inline427_inline2130__ssa_v0,
                rms_workers_inline472_inline2070__ssa_v0,
                bs_inline417_inline2091__ssa_v0,
                position_ids__ssa_v0,
                idx_row_offsets__ssa_v0,
                inner_freqs_cos__ssa_v0,
                inner_freqs_sin__ssa_v0,
                pooled_kv_inline2143__rv_v2,
                norm_w_2d_inline470_inline2095__ssa_v0,
                deps=[pool_tid_inline423_inline2079__ssa_v0, boundary_init_tid_inline471_inline2073__ssa_v0],
                core_num=rms_workers_inline472_inline2070__ssa_v0,
                attrs={"arg_directions": [pl.adir.inout, pl.adir.scalar, pl.adir.scalar, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input]},
            )
            normed_kv_inline212__rv_v3: pl.Tensor[[384, 128], pl.BF16, pl.MemRef("mem_ddr_106", pl.const(0, pl.INT64), 98304)] = ret__tmp_v0_16[0]
            rms_tid_inline477_inline2112__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_16[1]
            rms_tid_inline2076__ssa_v0: pl.Scalar[pl.TASK_ID] = rms_tid_inline477_inline2112__ssa_v0
            rms_tid_inline209__ssa_v0: pl.Scalar[pl.TASK_ID] = rms_tid_inline2076__ssa_v0
            bs_inline2191__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(position_ids__ssa_v0, 0)
            compact_rows_inline2207__ssa_v0: pl.Scalar[pl.INDEX] = bs_inline2191__ssa_v0 // 6 * 2
            rms_blocks_inline2196__ssa_v0: pl.Scalar[pl.INDEX] = (compact_rows_inline2207__ssa_v0 + 15) // 16
            kv_flat_inline2202__ssa_v0: pl.Tensor[[T_DYN, 128], pl.FP32, pl.MemRef("mem_ddr_107", pl.const(0, pl.INT64), 0)] = idx_kv_unused__ssa_v0
            idx_kv_scale_values_inline2188__ssa_v0: pl.Tensor[[384, 1], pl.FP32, pl.MemRef("mem_ddr_108", pl.const(0, pl.INT64), 1536)] = pl.tensor.create(
                [384, 1], dtype=pl.FP32, layout=pl.TensorLayout.ND
            )
            kv_final_inline2183__ssa_v0: pl.Tensor[[384, 128], pl.FP32, pl.MemRef("mem_ddr_109", pl.const(0, pl.INT64), 196608)] = pl.tensor.create(
                [384, 128], dtype=pl.FP32, layout=pl.TensorLayout.ND
            )
            ret__tmp_v0_17: pl.Tuple[pl.Tensor[[384, 128], pl.FP32], pl.Scalar[pl.TASK_ID]] = pl.submit(
                self.kv_hadamard,
                kv_final_inline2183__ssa_v0,
                hadamard_idx__ssa_v0,
                rms_blocks_inline2196__ssa_v0,
                compact_rows_inline2207__ssa_v0,
                normed_kv_inline212__rv_v3,
                deps=[rms_tid_inline209__ssa_v0, cmp_kv_score_tid__ssa_v0],
                attrs={"arg_directions": [pl.adir.output_existing, pl.adir.input, pl.adir.scalar, pl.adir.scalar, pl.adir.input]},
            )
            kv_final_inline2183__rv_v2: pl.Tensor[[384, 128], pl.FP32, pl.MemRef("mem_ddr_110", pl.const(0, pl.INT64), 196608)] = ret__tmp_v0_17[0]
            hadamard_tid_inline2199__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_17[1]
            ret__tmp_v0_18: pl.Tuple[pl.Tensor[[384, 1], pl.FP32], pl.Tensor[[IDX_CACHE_BLOCK_NUM_DYN, INDEXER_PAGE_BYTES_DYN], pl.INT8], pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(
                self.kv_and_cache_write_spmd,
                compact_rows_inline2207__ssa_v0,
                kv_final_inline2183__rv_v2,
                idx_kv_scale_values_inline2188__ssa_v0,
                idx_kv_cache__ssa_v0,
                kv_flat_inline2202__ssa_v0,
                position_ids__ssa_v0,
                idx_row_offsets__ssa_v0,
                idx_slot_mapping__ssa_v0,
                deps=[hadamard_tid_inline2199__ssa_v0],
                core_num=rms_blocks_inline2196__ssa_v0,
                allow_early_resolve=True,
                attrs={"arg_directions": [pl.adir.scalar, pl.adir.input, pl.adir.output_existing, pl.adir.inout, pl.adir.output_existing, pl.adir.input, pl.adir.input, pl.adir.input]},
            )
            idx_kv_scale_values_inline2188__ssa_v1: pl.Tensor[[384, 1], pl.FP32, pl.MemRef("mem_ddr_111", pl.const(0, pl.INT64), 1536)] = ret__tmp_v0_18[0]
            idx_kv_cache__rv_v2: pl.Tensor[[IDX_CACHE_BLOCK_NUM_DYN, INDEXER_PAGE_BYTES_DYN], pl.INT8, pl.MemRef("mem_ddr_112", pl.const(0, pl.INT64), 0)] = ret__tmp_v0_18[1]
            _write_tid_inline2206__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_18[2]
            ret__tmp_v0_19: pl.Tuple[pl.Tensor[[IDX_CACHE_BLOCK_NUM_DYN, INDEXER_PAGE_BYTES_DYN], pl.INT8], pl.Scalar[pl.TASK_ID]] = pl.submit(
                self.idx_kv_scale_commit,
                compact_rows_inline2207__ssa_v0,
                position_ids__ssa_v0,
                idx_row_offsets__ssa_v0,
                idx_slot_mapping__ssa_v0,
                idx_kv_cache__rv_v2,
                idx_kv_scale_values_inline2188__ssa_v1,
                deps=[_write_tid_inline2206__ssa_v0],
                allow_early_resolve=True,
                attrs={"arg_directions": [pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.inout, pl.adir.input]},
            )
            idx_kv_cache__rv_v3: pl.Tensor[[IDX_CACHE_BLOCK_NUM_DYN, INDEXER_PAGE_BYTES_DYN], pl.INT8, pl.MemRef("mem_ddr_113", pl.const(0, pl.INT64), 0)] = ret__tmp_v0_19[0]
            scale_commit_tid_inline2215__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_19[1]
            write_tid_inline206__ssa_v0: pl.Scalar[pl.TASK_ID] = scale_commit_tid_inline2215__ssa_v0
            idx_cache_write_tid__ssa_v0: pl.Scalar[pl.TASK_ID] = write_tid_inline206__ssa_v0
            with pl.scope():
                qr_hadamard_i8_inline221__ssa_v0: pl.Tensor[[24576, 128], pl.INT8, pl.MemRef("mem_ddr_114", pl.const(0, pl.INT64), 3145728)] = pl.tensor.create(
                    [24576, 128], dtype=pl.INT8, layout=pl.TensorLayout.ND
                )
                qr_hadamard_scale_dq_inline217__ssa_v0: pl.Tensor[[24576, 1], pl.FP32, pl.MemRef("mem_ddr_115", pl.const(0, pl.INT64), 98304)] = pl.tensor.create(
                    [24576, 1], dtype=pl.FP32, layout=pl.TensorLayout.ND
                )
                qr_bf16_inline2253__ssa_v0: pl.Tensor[[24576, 128], pl.BF16, pl.MemRef("mem_ddr_116", pl.const(0, pl.INT64), 6291456)] = pl.tensor.create(
                    [24576, 128], dtype=pl.BF16, layout=pl.TensorLayout.ND
                )
                bs_inline904_inline2257__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(x_normed_t__ssa_v0, 0)
                row_blocks_inline907_inline2269__ssa_v0: pl.Scalar[pl.INDEX] = (bs_inline904_inline2257__ssa_v0 + 15) // 16
                qr_acc_pad_inline908_inline2268__ssa_v0: pl.Tensor[[384, 8192], pl.INT32, pl.MemRef("mem_ddr_117", pl.const(0, pl.INT64), 12582912)] = pl.tensor.create(
                    [384, 8192], dtype=pl.INT32, layout=pl.TensorLayout.ND
                )
                ret__tmp_v0_20: pl.Tuple[pl.Tensor[[384, 8192], pl.INT32], pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(
                    self.idx_qr_proj_matmul_spmd,
                    qr_acc_pad_inline908_inline2268__ssa_v0,
                    row_blocks_inline907_inline2269__ssa_v0,
                    bs_inline904_inline2257__ssa_v0,
                    qr__ssa_v0,
                    idx_wq_b__ssa_v0,
                    core_num=24,
                    allow_early_resolve=True,
                    attrs={"arg_directions": [pl.adir.output_existing, pl.adir.scalar, pl.adir.scalar, pl.adir.input, pl.adir.input]},
                )
                qr_acc_pad_inline908_inline2268__rv_v2: pl.Tensor[[384, 8192], pl.INT32, pl.MemRef("mem_ddr_118", pl.const(0, pl.INT64), 12582912)] = ret__tmp_v0_20[0]
                idx_qr_mm_tid_inline906_inline2264__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_20[1]
                qr_bf16_2d_inline914_inline2247__ssa_v0: pl.Tensor[[384, 8192], pl.BF16, pl.MemRef("mem_ddr_116", pl.const(0, pl.INT64), 6291456)] = pl.tensor.reshape(
                    qr_bf16_inline2253__ssa_v0, [384, 8192]
                )
                dq_rope_units_inline905_inline2241__ssa_v0: pl.Scalar[pl.INDEX] = bs_inline904_inline2257__ssa_v0 // 8 * 16
                dq_rope_workers_inline918_inline2238__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(dq_rope_units_inline905_inline2241__ssa_v0, 48)
                ret__tmp_v0_21: pl.Tuple[pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(
                    self.idx_qr_dequant_rope_spmd,
                    qr_bf16_2d_inline914_inline2247__ssa_v0,
                    dq_rope_units_inline905_inline2241__ssa_v0,
                    dq_rope_workers_inline918_inline2238__ssa_v0,
                    qr_scale__ssa_v0,
                    freqs_cos__ssa_v0,
                    idx_sin_signed__rv_v2,
                    idx_wq_b_scale__ssa_v0,
                    qr_acc_pad_inline908_inline2268__rv_v2,
                    core_num=dq_rope_workers_inline918_inline2238__ssa_v0,
                    allow_early_resolve=True,
                    attrs={"arg_directions": [pl.adir.output_existing, pl.adir.scalar, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input]},
                )
                tid__ssa_v8: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_21[0]
                idx_qr_mm_tid_inline2296__ssa_v0: pl.Scalar[pl.TASK_ID] = idx_qr_mm_tid_inline906_inline2264__ssa_v0
                bs_inline938_inline2283__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(x_normed_t__ssa_v0, 0)
                bs_heads_inline935_inline2244__ssa_v0: pl.Scalar[pl.INDEX] = bs_inline938_inline2283__ssa_v0 * 64
                qh_acc_gm_inline928_inline2298__ssa_v0: pl.Tensor[[bs_heads_inline935_inline2244__ssa_v0, 128], pl.FP32, pl.MemRef("mem_ddr_119", pl.const(0, pl.INT64), 0)] = pl.tensor.create(
                    [bs_heads_inline935_inline2244__ssa_v0, 128], dtype=pl.FP32, layout=pl.TensorLayout.ND
                )
                ret__tmp_v0_22: pl.Tuple[pl.Tensor[[bs_heads_inline935_inline2244__ssa_v0, 128], pl.FP32], pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(
                    self.qr_hadamard_matmul_spmd,
                    hadamard_idx__ssa_v0,
                    qh_acc_gm_inline928_inline2298__ssa_v0,
                    bs_heads_inline935_inline2244__ssa_v0,
                    qr_bf16_inline2253__ssa_v0,
                    deps=[idx_qr_mm_tid_inline2296__ssa_v0],
                    core_num=24,
                    allow_early_resolve=True,
                    attrs={"arg_directions": [pl.adir.input, pl.adir.output_existing, pl.adir.scalar, pl.adir.input]},
                )
                qh_acc_gm_inline928_inline2298__rv_v2: pl.Tensor[[bs_heads_inline935_inline2244__ssa_v0, 128], pl.FP32, pl.MemRef("mem_ddr_120", pl.const(0, pl.INT64), 0)] = ret__tmp_v0_22[0]
                qh_mm_tid_inline933_inline2240__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_22[1]
                ret__tmp_v0_23: pl.Tuple[pl.Tensor[[24576, 128], pl.INT8], pl.Tensor[[24576, 1], pl.FP32], pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(
                    self.qr_hadamard_quant_spmd,
                    qr_hadamard_i8_inline221__ssa_v0,
                    qr_hadamard_scale_dq_inline217__ssa_v0,
                    bs_heads_inline935_inline2244__ssa_v0,
                    qh_acc_gm_inline928_inline2298__rv_v2,
                    core_num=48,
                    allow_early_resolve=True,
                    attrs={"arg_directions": [pl.adir.output_existing, pl.adir.output_existing, pl.adir.scalar, pl.adir.input]},
                )
                qr_hadamard_i8_inline221__rv_v2: pl.Tensor[[24576, 128], pl.INT8, pl.MemRef("mem_ddr_121", pl.const(0, pl.INT64), 3145728)] = ret__tmp_v0_23[0]
                qr_hadamard_scale_dq_inline217__rv_v2: pl.Tensor[[24576, 1], pl.FP32, pl.MemRef("mem_ddr_122", pl.const(0, pl.INT64), 98304)] = ret__tmp_v0_23[1]
                qh_quant_tid_inline927_inline2267__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_23[2]
                qh_quant_tid_inline2256__ssa_v0: pl.Scalar[pl.TASK_ID] = qh_quant_tid_inline927_inline2267__ssa_v0
                qh_quant_tid_inline215__ssa_v0: pl.Scalar[pl.TASK_ID] = qh_quant_tid_inline2256__ssa_v0
                weights_gate_dep_inline220__ssa_v0: pl.Scalar[pl.TASK_ID] = pl.system.task_dummy(deps=[])
                bs_inline1017_inline2346__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(x_normed_t__ssa_v0, 0)
                row_blocks_inline1022_inline2357__ssa_v0: pl.Scalar[pl.INDEX] = (bs_inline1017_inline2346__ssa_v0 + 15) // 16
                x_flat_inline1018_inline2361__ssa_v0: pl.Tensor[[T_DYN, 4096], pl.BF16, pl.MemRef("mem_ddr_123", pl.const(0, pl.INT64), 0)] = x_normed_t__ssa_v0
                weights_inline1023_inline2353__ssa_v0: pl.Tensor[[384, 64], pl.FP32, pl.MemRef("mem_ddr_124", pl.const(0, pl.INT64), 98304)] = pl.tensor.create(
                    [384, 64], dtype=pl.FP32, layout=pl.TensorLayout.ND
                )
                weights_partial_inline1024_inline2370__ssa_v0: pl.Tensor[[384, 64], pl.FP32, pl.MemRef("mem_ddr_125", pl.const(0, pl.INT64), 98304)] = pl.tensor.create(
                    [384, 64], dtype=pl.FP32, layout=pl.TensorLayout.ND
                )
                ret__tmp_v0_24: pl.Tuple[pl.Tensor[[384, 64], pl.FP32], pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(
                    self.weights_proj_spmd,
                    weights_partial_inline1024_inline2370__ssa_v0,
                    row_blocks_inline1022_inline2357__ssa_v0,
                    bs_inline1017_inline2346__ssa_v0,
                    x_flat_inline1018_inline2361__ssa_v0,
                    weights_proj__ssa_v0,
                    deps=[weights_gate_dep_inline220__ssa_v0],
                    core_num=8,
                    allow_early_resolve=True,
                    attrs={"arg_directions": [pl.adir.output_existing, pl.adir.scalar, pl.adir.scalar, pl.adir.input, pl.adir.input]},
                )
                weights_partial_inline1024_inline2370__rv_v2: pl.Tensor[[384, 64], pl.FP32, pl.MemRef("mem_ddr_126", pl.const(0, pl.INT64), 98304)] = ret__tmp_v0_24[0]
                _weights_tid_inline1025_inline2365__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_24[1]
                ret__tmp_v0_25: pl.Tuple[pl.Tensor[[384, 64], pl.FP32], pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(
                    self.weights_proj_reduce_spmd,
                    weights_partial_inline1024_inline2370__rv_v2,
                    weights_inline1023_inline2353__ssa_v0,
                    core_num=row_blocks_inline1022_inline2357__ssa_v0,
                    allow_early_resolve=True,
                    attrs={"arg_directions": [pl.adir.input, pl.adir.output_existing]},
                )
                weights_inline1023_inline2353__ssa_v1: pl.Tensor[[384, 64], pl.FP32, pl.MemRef("mem_ddr_127", pl.const(0, pl.INT64), 98304)] = ret__tmp_v0_25[0]
                weights_tid_inline1012_inline2359__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_25[1]
                weights_inline2400__ssa_v0: pl.Tensor[[384, 64], pl.FP32, pl.MemRef("mem_ddr_128", pl.const(0, pl.INT64), 98304)] = weights_inline1023_inline2353__ssa_v1
                weights_tid_inline2397__ssa_v0: pl.Scalar[pl.TASK_ID] = weights_tid_inline1012_inline2359__ssa_v0
                b_dim_inline1051_inline2372__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(idx_block_table__ssa_v0, 0)
                table_columns_inline1060_inline2338__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(idx_block_table__ssa_v0, 1)
                idx_table_len_inline1096_inline2335__ssa_v0: pl.Scalar[pl.INDEX] = b_dim_inline1051_inline2372__ssa_v0 * table_columns_inline1060_inline2338__ssa_v0
                idx_block_table_flat_inline1052_inline2374__ssa_v0: pl.Tensor[[idx_table_len_inline1096_inline2335__ssa_v0], pl.INT32, pl.MemRef("mem_ddr_33", pl.const(0, pl.INT64), 0)] = (
                    pl.tensor.reshape(idx_block_table__ssa_v0, [idx_table_len_inline1096_inline2335__ssa_v0])
                )
                pair_arena_inline1064_inline2363__ssa_v0: pl.Tensor[[24576, 1024], pl.FP32, pl.MemRef("mem_ddr_129", pl.const(0, pl.INT64), 100663296)] = pl.tensor.create(
                    [24576, 1024], dtype=pl.FP32, layout=pl.TensorLayout.ND
                )
                score_arena_inline1071_inline2347__ssa_v0: pl.Tensor[[384, 8192], pl.FP32, pl.MemRef("mem_ddr_130", pl.const(0, pl.INT64), 12582912)] = pl.tensor.create(
                    [384, 8192], dtype=pl.FP32, layout=pl.TensorLayout.ND
                )
                coefficients_inline3198__ssa_v0: pl.Tensor[[6144, 64], pl.FP16, pl.MemRef("mem_ddr_131", pl.const(0, pl.INT64), 786432)] = pl.tensor.create(
                    [6144, 64], dtype=pl.FP16, layout=pl.TensorLayout.ND
                )
                ret__tmp_v0_26: pl.Tuple[pl.Tensor[[6144, 64], pl.FP16], pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(
                    self.indexer_head_coefficients_spmd,
                    position_ids__ssa_v0,
                    coefficients_inline3198__ssa_v0,
                    qr_hadamard_scale_dq_inline217__rv_v2,
                    weights_inline2400__ssa_v0,
                    deps=[qh_quant_tid_inline215__ssa_v0, weights_tid_inline2397__ssa_v0],
                    core_num=48,
                    allow_early_resolve=True,
                    attrs={"arg_directions": [pl.adir.input, pl.adir.output_existing, pl.adir.input, pl.adir.input]},
                )
                coefficients_inline3198__rv_v2: pl.Tensor[[6144, 64], pl.FP16, pl.MemRef("mem_ddr_132", pl.const(0, pl.INT64), 786432)] = ret__tmp_v0_26[0]
                coefficients_tid_inline3196__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_26[1]
                coefficients_inline1068_inline2344__ssa_v0: pl.Tensor[[6144, 64], pl.FP16, pl.MemRef("mem_ddr_133", pl.const(0, pl.INT64), 786432)] = coefficients_inline3198__rv_v2
                coefficients_tid_inline1083_inline2379__ssa_v0: pl.Scalar[pl.TASK_ID] = coefficients_tid_inline3196__ssa_v0
                gm_pipe_buffer_0: pl.Tensor[[1], pl.FP32, pl.MemRef("mem_ddr_134", pl.const(0, pl.INT64), 4)] = pl.tensor.create([1], dtype=pl.FP32, layout=pl.TensorLayout.ND, manual_dep=True)
                ret__tmp_v0_27: pl.Tuple[pl.Tensor[[384, 8192], pl.FP32], pl.Tensor[[24576, 1024], pl.FP32], pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(
                    self.indexer_score_topk_leaf_spmd,
                    position_ids__ssa_v0,
                    kv_seq_lens__ssa_v0,
                    score_arena_inline1071_inline2347__ssa_v0,
                    qr_hadamard_i8_inline221__rv_v2,
                    coefficients_inline1068_inline2344__ssa_v0,
                    idx_block_table_flat_inline1052_inline2374__ssa_v0,
                    table_columns_inline1060_inline2338__ssa_v0,
                    idx_kv_cache__rv_v3,
                    pair_arena_inline1064_inline2363__ssa_v0,
                    gm_pipe_buffer_0,
                    deps=[coefficients_tid_inline1083_inline2379__ssa_v0, idx_cache_write_tid__ssa_v0],
                    core_num=24,
                    allow_early_resolve=True,
                    attrs={
                        "arg_directions": [
                            pl.adir.input,
                            pl.adir.input,
                            pl.adir.inout,
                            pl.adir.input,
                            pl.adir.input,
                            pl.adir.input,
                            pl.adir.scalar,
                            pl.adir.input,
                            pl.adir.output_existing,
                            pl.adir.output_existing,
                        ]
                    },
                )
                score_arena_inline1071_inline2347__rv_v2: pl.Tensor[[384, 8192], pl.FP32, pl.MemRef("mem_ddr_135", pl.const(0, pl.INT64), 12582912)] = ret__tmp_v0_27[0]
                pair_arena_inline1064_inline2363__ssa_v1: pl.Tensor[[24576, 1024], pl.FP32, pl.MemRef("mem_ddr_136", pl.const(0, pl.INT64), 100663296)] = ret__tmp_v0_27[1]
                score_tid_inline1073_inline2369__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_27[2]
                for topk_batch_inline1034_inline2301__idx_v0, (max_topk_cache_len_inline1066_inline2303__iter_v1,) in pl.range(
                    b_dim_inline1051_inline2372__ssa_v0, init_values=(0,), attrs={"iter_arg_rebind_0": True}
                ):
                    t__tmp_v280: pl.Scalar[pl.INT32] = pl.tensor.read(kv_seq_lens__ssa_v0, [topk_batch_inline1034_inline2301__idx_v0])
                    topk_cache_len_inline1038_inline2329__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(t__tmp_v280, pl.INDEX) // 4
                    max_topk_cache_len_inline1066_inline2303__ssa_v3: pl.Scalar[pl.INDEX] = pl.max(max_topk_cache_len_inline1066_inline2303__iter_v1, topk_cache_len_inline1038_inline2329__ssa_v0)
                    max_topk_cache_len_inline1066_inline2303__rv_v2: pl.Scalar[pl.INDEX] = pl.yield_(max_topk_cache_len_inline1066_inline2303__ssa_v3)
                with pl.scope():
                    if max_topk_cache_len_inline1066_inline2303__rv_v2 <= 8192:
                        ret__tmp_v0_28: pl.Tuple[pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(
                            self._decode_csa_tp1_attention_indexer_topk_single_leaf_publish,
                            position_ids__ssa_v0,
                            kv_seq_lens__ssa_v0,
                            score_arena_inline1071_inline2347__rv_v2,
                            idx_topk_scores__ssa_v0,
                            idx_topk__ssa_v0,
                            deps=[score_tid_inline1073_inline2369__ssa_v0],
                            core_num=48,
                            allow_early_resolve=True,
                            attrs={"arg_directions": [pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.output_existing, pl.adir.output_existing]},
                        )
                        tid__ssa_v9: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_28[0]
                    else:
                        ret__tmp_v0_29: pl.Tuple[pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(
                            self._decode_csa_tp1_attention_indexer_topk_query_merge,
                            position_ids__ssa_v0,
                            kv_seq_lens__ssa_v0,
                            pair_arena_inline1064_inline2363__ssa_v1,
                            idx_topk_scores__ssa_v0,
                            idx_topk__ssa_v0,
                            deps=[score_tid_inline1073_inline2369__ssa_v0],
                            core_num=48,
                            allow_early_resolve=True,
                            attrs={"arg_directions": [pl.adir.input, pl.adir.input, pl.adir.inout, pl.adir.output_existing, pl.adir.output_existing]},
                        )
                        tid__ssa_v10: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_29[0]
                idx_topk_scores__ssa_v1: pl.Tensor[[T_DYN, 512], pl.FP32, pl.MemRef("mem_ddr_137", pl.const(0, pl.INT64), 0)] = idx_topk_scores__ssa_v0
                idx_topk__ssa_v1: pl.Tensor[[T_DYN, 512], pl.INT32, pl.MemRef("mem_ddr_138", pl.const(0, pl.INT64), 0)] = idx_topk__ssa_v0
                idx_topk_scores__ssa_v2: pl.Tensor[[T_DYN, 512], pl.FP32, pl.MemRef("mem_ddr_139", pl.const(0, pl.INT64), 0)] = idx_topk_scores__ssa_v1
                idx_topk__ssa_v2: pl.Tensor[[T_DYN, 512], pl.INT32, pl.MemRef("mem_ddr_140", pl.const(0, pl.INT64), 0)] = idx_topk__ssa_v1
                idx_topk_scores__ssa_v3: pl.Tensor[[T_DYN, 512], pl.FP32, pl.MemRef("mem_ddr_141", pl.const(0, pl.INT64), 0)] = idx_topk_scores__ssa_v2
                idx_topk__ssa_v3: pl.Tensor[[T_DYN, 512], pl.INT32, pl.MemRef("mem_ddr_142", pl.const(0, pl.INT64), 0)] = idx_topk__ssa_v2
            o_packed_heads__ssa_v0: pl.Tensor[[3072, 4096], pl.BF16, pl.MemRef("mem_ddr_143", pl.const(0, pl.INT64), 25165824)] = pl.tensor.create(
                [3072, 4096], dtype=pl.BF16, layout=pl.TensorLayout.ND
            )
            ori_block_num_inline2459__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(kv_cache__ssa_v0, 0)
            t_dim_inline2491__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(q__ssa_v0, 0)
            t_heads_inline2469__ssa_v0: pl.Scalar[pl.INDEX] = t_dim_inline2491__ssa_v0 * 64
            rope_cs_blocks_inline2473__ssa_v0: pl.Scalar[pl.INDEX] = t_dim_inline2491__ssa_v0 // 8
            ori_kv_flat_inline2493__ssa_v0: pl.Tensor[[ori_block_num_inline2459__ssa_v0 * pl.const(32, pl.INDEX), 512], pl.BF16, pl.MemRef("mem_ddr_29", pl.const(0, pl.INT64), 0)] = pl.tensor.reshape(
                kv_cache__ssa_v0, [ori_block_num_inline2459__ssa_v0 * 32, 512]
            )
            ret__tmp_v0_30: pl.Tuple[pl.Tensor[[ori_block_num_inline2459__ssa_v0 * pl.const(32, pl.INDEX), 512], pl.BF16], pl.Scalar[pl.TASK_ID]] = pl.submit(
                self.kv_touch, ori_kv_flat_inline2493__ssa_v0, allow_early_resolve=True, attrs={"arg_directions": [pl.adir.inout]}
            )
            ori_kv_flat_inline2493__ssa_v1: pl.Tensor[[ori_block_num_inline2459__ssa_v0 * pl.const(32, pl.INDEX), 512], pl.BF16, pl.MemRef("mem_ddr_144", pl.const(0, pl.INT64), 0)] = ret__tmp_v0_30[0]
            tid__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_30[1]
            sparse_bias_inline2455__ssa_v0: pl.Tensor[[t_dim_inline2491__ssa_v0, 1024], pl.FP32, pl.MemRef("mem_ddr_145", pl.const(0, pl.INT64), 0)] = pl.tensor.create(
                [t_dim_inline2491__ssa_v0, 1024], dtype=pl.FP32, layout=pl.TensorLayout.ND
            )
            cmp_sparse_indices_inline2484__ssa_v0: pl.Tensor[[t_dim_inline2491__ssa_v0, 512], pl.INT32, pl.MemRef("mem_ddr_146", pl.const(0, pl.INT64), 0)] = pl.tensor.create(
                [t_dim_inline2491__ssa_v0, 512], dtype=pl.INT32, layout=pl.TensorLayout.ND
            )
            valid_block_mask_inline2452__ssa_v0: pl.Tensor[[t_dim_inline2491__ssa_v0, 16], pl.INT32, pl.MemRef("mem_ddr_147", pl.const(0, pl.INT64), 0)] = pl.tensor.create(
                [t_dim_inline2491__ssa_v0, 16], dtype=pl.INT32, layout=pl.TensorLayout.ND
            )
            ret__tmp_v0_31: pl.Tuple[pl.Tensor[[t_dim_inline2491__ssa_v0, 512], pl.INT32], pl.Tensor[[t_dim_inline2491__ssa_v0, 1024], pl.FP32], pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(
                self.csa_slots_build_valid_qk_plan_spmd,
                cmp_sparse_indices_inline2484__ssa_v0,
                sparse_bias_inline2455__ssa_v0,
                t_dim_inline2491__ssa_v0,
                position_ids_t1__ssa_v0,
                idx_topk__ssa_v3,
                ori_block_table__ssa_v0,
                valid_block_mask_inline2452__ssa_v0,
                core_num=16,
                allow_early_resolve=True,
                attrs={"arg_directions": [pl.adir.inout, pl.adir.output_existing, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.output_existing]},
            )
            cmp_sparse_indices_inline2484__rv_v2: pl.Tensor[[t_dim_inline2491__ssa_v0, 512], pl.INT32, pl.MemRef("mem_ddr_148", pl.const(0, pl.INT64), 0)] = ret__tmp_v0_31[0]
            sparse_bias_inline2455__rv_v2: pl.Tensor[[t_dim_inline2491__ssa_v0, 1024], pl.FP32, pl.MemRef("mem_ddr_149", pl.const(0, pl.INT64), 0)] = ret__tmp_v0_31[1]
            qk_plan_tid_inline2467__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_31[2]
            cmp_block_num_inline2525__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(cmp_kv__ssa_v0, 0)
            cmp_kv_flat_inline2430__ssa_v0: pl.Tensor[[cmp_block_num_inline2525__ssa_v0 * pl.const(32, pl.INDEX), 512], pl.BF16, pl.MemRef("mem_ddr_30", pl.const(0, pl.INT64), 0)] = pl.tensor.reshape(
                cmp_kv__ssa_v0, [cmp_block_num_inline2525__ssa_v0 * 32, 512]
            )
            q_flat_inline2480__ssa_v0: pl.Tensor[[t_heads_inline2469__ssa_v0, 512], pl.BF16, pl.MemRef("mem_ddr_56", pl.const(0, pl.INT64), 0)] = pl.tensor.reshape(
                q__ssa_v0, [t_heads_inline2469__ssa_v0, 512]
            )
            attn_sink_col_inline2429__ssa_v0: pl.Tensor[[64, 1], pl.FP32, pl.MemRef("mem_ddr_45", pl.const(0, pl.INT64), 256)] = pl.tensor.reshape(attn_sink__ssa_v0, [64, 1])
            attn_mi_inline2490__ssa_v0: pl.Tensor[[t_heads_inline2469__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_150", pl.const(0, pl.INT64), 0)] = pl.tensor.create(
                [t_heads_inline2469__ssa_v0, 1], dtype=pl.FP32, layout=pl.TensorLayout.ND
            )
            attn_li_inline2516__ssa_v0: pl.Tensor[[t_heads_inline2469__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_151", pl.const(0, pl.INT64), 0)] = pl.tensor.create(
                [t_heads_inline2469__ssa_v0, 1], dtype=pl.FP32, layout=pl.TensorLayout.ND
            )
            attn_oi_inline2428__ssa_v0: pl.Tensor[[t_heads_inline2469__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_152", pl.const(0, pl.INT64), 0)] = pl.tensor.create(
                [t_heads_inline2469__ssa_v0, 512], dtype=pl.FP32, layout=pl.TensorLayout.ND
            )
            kv_transfer_inline2426__ssa_v0: pl.Tensor[[12288, 512], pl.BF16, pl.MemRef("mem_ddr_153", pl.const(0, pl.INT64), 12582912)] = pl.tensor.create(
                [12288, 512], dtype=pl.BF16, layout=pl.TensorLayout.ND
            )
            score_transfer_inline2447__ssa_v0: pl.Tensor[[1536, 512], pl.FP32, pl.MemRef("mem_ddr_154", pl.const(0, pl.INT64), 3145728)] = pl.tensor.create(
                [1536, 512], dtype=pl.FP32, layout=pl.TensorLayout.ND
            )
            probability_transfer_inline2550__ssa_v0: pl.Tensor[[1536, 512], pl.BF16, pl.MemRef("mem_ddr_155", pl.const(0, pl.INT64), 1572864)] = pl.tensor.create(
                [1536, 512], dtype=pl.BF16, layout=pl.TensorLayout.ND
            )
            pv_transfer_inline2437__ssa_v0: pl.Tensor[[1536, 512], pl.FP32, pl.MemRef("mem_ddr_156", pl.const(0, pl.INT64), 3145728)] = pl.tensor.create(
                [1536, 512], dtype=pl.FP32, layout=pl.TensorLayout.ND
            )
            mi_transfer_inline2500__ssa_v0: pl.Tensor[[1536, 1], pl.FP32, pl.MemRef("mem_ddr_157", pl.const(0, pl.INT64), 6144)] = pl.tensor.create([1536, 1], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            li_transfer_inline2502__ssa_v0: pl.Tensor[[1536, 1], pl.FP32, pl.MemRef("mem_ddr_158", pl.const(0, pl.INT64), 6144)] = pl.tensor.create([1536, 1], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            alpha_transfer_inline2486__ssa_v0: pl.Tensor[[1536, 1], pl.FP32, pl.MemRef("mem_ddr_159", pl.const(0, pl.INT64), 6144)] = pl.tensor.create(
                [1536, 1], dtype=pl.FP32, layout=pl.TensorLayout.ND
            )
            ffts_workspace_inline2438__ssa_v0: pl.Tensor[[256], pl.INT64, pl.MemRef("mem_ddr_160", pl.const(0, pl.INT64), 2048)] = pl.tensor.create([256], dtype=pl.INT64, layout=pl.TensorLayout.ND)
            ret__tmp_v0_32: pl.Tuple[
                pl.Tensor[[t_heads_inline2469__ssa_v0, 1], pl.FP32], pl.Tensor[[t_heads_inline2469__ssa_v0, 1], pl.FP32], pl.Tensor[[t_heads_inline2469__ssa_v0, 512], pl.FP32], pl.Scalar[pl.TASK_ID]
            ] = pl.spmd_submit(
                self.qk_pv_spmd,
                ffts_workspace_inline2438__ssa_v0,
                t_dim_inline2491__ssa_v0,
                q_flat_inline2480__ssa_v0,
                valid_block_mask_inline2452__ssa_v0,
                kv_transfer_inline2426__ssa_v0,
                score_transfer_inline2447__ssa_v0,
                probability_transfer_inline2550__ssa_v0,
                pv_transfer_inline2437__ssa_v0,
                attn_sink_col_inline2429__ssa_v0,
                mi_transfer_inline2500__ssa_v0,
                li_transfer_inline2502__ssa_v0,
                position_ids_t1__ssa_v0,
                ori_block_table__ssa_v0,
                ori_kv_flat_inline2493__ssa_v1,
                cmp_sparse_indices_inline2484__rv_v2,
                cmp_block_table__ssa_v0,
                cmp_kv_flat_inline2430__ssa_v0,
                sparse_bias_inline2455__rv_v2,
                alpha_transfer_inline2486__ssa_v0,
                attn_mi_inline2490__ssa_v0,
                attn_li_inline2516__ssa_v0,
                attn_oi_inline2428__ssa_v0,
                deps=[qk_plan_tid_inline2467__ssa_v0],
                core_num=24,
                allow_early_resolve=True,
                attrs={
                    "arg_directions": [
                        pl.adir.input,
                        pl.adir.scalar,
                        pl.adir.input,
                        pl.adir.input,
                        pl.adir.inout,
                        pl.adir.inout,
                        pl.adir.inout,
                        pl.adir.inout,
                        pl.adir.input,
                        pl.adir.inout,
                        pl.adir.inout,
                        pl.adir.input,
                        pl.adir.input,
                        pl.adir.input,
                        pl.adir.input,
                        pl.adir.input,
                        pl.adir.input,
                        pl.adir.input,
                        pl.adir.inout,
                        pl.adir.output_existing,
                        pl.adir.output_existing,
                        pl.adir.output_existing,
                    ]
                },
            )
            attn_mi_inline2490__ssa_v1: pl.Tensor[[t_heads_inline2469__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_161", pl.const(0, pl.INT64), 0)] = ret__tmp_v0_32[0]
            attn_li_inline2516__ssa_v1: pl.Tensor[[t_heads_inline2469__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_162", pl.const(0, pl.INT64), 0)] = ret__tmp_v0_32[1]
            attn_oi_inline2428__ssa_v1: pl.Tensor[[t_heads_inline2469__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_163", pl.const(0, pl.INT64), 0)] = ret__tmp_v0_32[2]
            qk_tid_inline2476__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_32[3]
            rope_sin_signed_inline2415__ssa_v0: pl.Tensor[[384, 64], pl.FP32, pl.MemRef("mem_ddr_164", pl.const(0, pl.INT64), 98304)] = pl.tensor.create(
                [384, 64], dtype=pl.FP32, layout=pl.TensorLayout.ND
            )
            rope_swap_idx_inline2414__ssa_v0: pl.Tensor[[16, 64], pl.INT32, pl.MemRef("mem_ddr_165", pl.const(0, pl.INT64), 4096)] = pl.tensor.create(
                [16, 64], dtype=pl.INT32, layout=pl.TensorLayout.ND
            )
            ret__tmp_v0_33: pl.Tuple[pl.Tensor[[16, 64], pl.INT32], pl.Tensor[[384, 64], pl.FP32], pl.Scalar[pl.TASK_ID]] = pl.submit(
                self.rope_cs,
                rope_swap_idx_inline2414__ssa_v0,
                rope_sin_signed_inline2415__ssa_v0,
                rope_cs_blocks_inline2473__ssa_v0,
                freqs_sin__ssa_v0,
                allow_early_resolve=True,
                attrs={"arg_directions": [pl.adir.output_existing, pl.adir.output_existing, pl.adir.scalar, pl.adir.input]},
            )
            rope_swap_idx_inline2414__ssa_v1: pl.Tensor[[16, 64], pl.INT32, pl.MemRef("mem_ddr_166", pl.const(0, pl.INT64), 4096)] = ret__tmp_v0_33[0]
            rope_sin_signed_inline2415__rv_v2: pl.Tensor[[384, 64], pl.FP32, pl.MemRef("mem_ddr_167", pl.const(0, pl.INT64), 98304)] = ret__tmp_v0_33[1]
            rope_tid_inline2413__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_33[2]
            attn_mi_inline258__ssa_v0: pl.Tensor[[t_heads_inline2469__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_168", pl.const(0, pl.INT64), 0)] = attn_mi_inline2490__ssa_v1
            attn_li_inline243__ssa_v0: pl.Tensor[[t_heads_inline2469__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_169", pl.const(0, pl.INT64), 0)] = attn_li_inline2516__ssa_v1
            attn_oi_inline260__ssa_v0: pl.Tensor[[t_heads_inline2469__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_170", pl.const(0, pl.INT64), 0)] = attn_oi_inline2428__ssa_v1
            rope_cos_il_inline238__ssa_v0: pl.Tensor[[T_DYN, 64], pl.FP32, pl.MemRef("mem_ddr_171", pl.const(0, pl.INT64), 0)] = freqs_cos__ssa_v0
            rope_sin_signed_inline242__ssa_v0: pl.Tensor[[384, 64], pl.FP32, pl.MemRef("mem_ddr_172", pl.const(0, pl.INT64), 98304)] = rope_sin_signed_inline2415__rv_v2
            rope_swap_idx_inline244__ssa_v0: pl.Tensor[[16, 64], pl.INT32, pl.MemRef("mem_ddr_173", pl.const(0, pl.INT64), 4096)] = rope_swap_idx_inline2414__ssa_v1
            qk_tid_inline237__ssa_v0: pl.Scalar[pl.TASK_ID] = qk_tid_inline2476__ssa_v0
            rope_tid_inline236__ssa_v0: pl.Scalar[pl.TASK_ID] = rope_tid_inline2413__ssa_v0
            t_dim_inline235__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(q__ssa_v0, 0)
            ret__tmp_v0_34: pl.Tuple[pl.Tensor[[3072, 4096], pl.BF16], pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(
                self.merge_norm_spmd,
                rope_swap_idx_inline244__ssa_v0,
                t_dim_inline235__ssa_v0,
                attn_li_inline243__ssa_v0,
                attn_oi_inline260__ssa_v0,
                rope_cos_il_inline238__ssa_v0,
                rope_sin_signed_inline242__ssa_v0,
                o_packed_heads__ssa_v0,
                deps=[qk_tid_inline237__ssa_v0, rope_tid_inline236__ssa_v0],
                core_num=48,
                attrs={"arg_directions": [pl.adir.input, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.output_existing]},
            )
            o_packed_heads__ssa_v2: pl.Tensor[[3072, 4096], pl.BF16, pl.MemRef("mem_ddr_174", pl.const(0, pl.INT64), 25165824)] = ret__tmp_v0_34[0]
            merge_tid_inline233__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_34[1]
            o_packed_heads__ssa_v1: pl.Tensor[[3072, 4096], pl.BF16, pl.MemRef("mem_ddr_175", pl.const(0, pl.INT64), 25165824)] = o_packed_heads__ssa_v2
            heads_dep__ssa_v0: pl.Scalar[pl.TASK_ID] = merge_tid_inline233__ssa_v0
            t_dim_inline347__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(attn_out__ssa_v0, 0)
            act_t_blks_inline312__ssa_v0: pl.Scalar[pl.INDEX] = (t_dim_inline347__ssa_v0 + 31) // 32
            proj_a_rows_inline297__ssa_v0: pl.Scalar[pl.INDEX] = (t_dim_inline347__ssa_v0 + 127) // 128
            proj_b_t_rows_inline332__ssa_v0: pl.Scalar[pl.INDEX] = (t_dim_inline347__ssa_v0 + 127) // 128
            proj_b_padded_rows_inline323__ssa_v0: pl.Scalar[pl.INDEX] = proj_b_t_rows_inline332__ssa_v0 * 128
            o_r_pad_inline329__ssa_v0: pl.Tensor[[384, 8192], pl.FP32, pl.MemRef("mem_ddr_176", pl.const(0, pl.INT64), 12582912)] = pl.tensor.create(
                [384, 8192], dtype=pl.FP32, layout=pl.TensorLayout.ND
            )
            o_r_i8_pad_inline333__ssa_v0: pl.Tensor[[384, 8192], pl.INT8, pl.MemRef("mem_ddr_177", pl.const(0, pl.INT64), 3145728)] = pl.tensor.create(
                [384, 8192], dtype=pl.INT8, layout=pl.TensorLayout.ND
            )
            act_scale_dq_inline313__ssa_v0: pl.Tensor[[1, 384], pl.FP32, pl.MemRef("mem_ddr_178", pl.const(0, pl.INT64), 1536)] = pl.tensor.create([1, 384], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            act_scale_q_inline338__ssa_v0: pl.Tensor[[1, 384], pl.FP32, pl.MemRef("mem_ddr_179", pl.const(0, pl.INT64), 1536)] = pl.tensor.create([1, 384], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            partials_inline317__ssa_v0: pl.Tensor[[384, 32768], pl.INT32, pl.MemRef("mem_ddr_180", pl.const(0, pl.INT64), 50331648)] = pl.tensor.create(
                [384, 32768], dtype=pl.INT32, layout=pl.TensorLayout.ND
            )
            proj_a_tids_inline342__ssa_v0: pl.Array[8, pl.TASK_ID] = pl.array.create(8, dtype=pl.TASK_ID)
            proj_b_tids_inline351__ssa_v0: pl.Array[8, pl.TASK_ID] = pl.array.create(8, dtype=pl.TASK_ID)
            with pl.scope(mode=pl.ScopeMode.MANUAL):
                for g_inline293__idx_v0, (o_r_pad_inline329__iter_v1, proj_a_tids_inline342__iter_v1) in pl.parallel(
                    8, init_values=(o_r_pad_inline329__ssa_v0, proj_a_tids_inline342__ssa_v0), attrs={"iter_arg_rebind_0": False, "iter_arg_rebind_1": True}
                ):
                    row_base_o_inline310__ssa_v0: pl.Scalar[pl.INDEX] = g_inline293__idx_v0 * 384
                    out_col_g_inline306__ssa_v0: pl.Scalar[pl.INDEX] = g_inline293__idx_v0 * 1024
                    ret__tmp_v0_35: pl.Tuple[pl.Tensor[[384, 8192], pl.FP32], pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(
                        self.proj_a_mm_spmd,
                        t_dim_inline347__ssa_v0,
                        row_base_o_inline310__ssa_v0,
                        o_packed_heads__ssa_v1,
                        wo_a__ssa_v0,
                        g_inline293__idx_v0,
                        o_r_pad_inline329__iter_v1,
                        out_col_g_inline306__ssa_v0,
                        deps=[heads_dep__ssa_v0],
                        core_num=proj_a_rows_inline297__ssa_v0 * 8,
                        allow_early_resolve=True,
                        attrs={"arg_directions": [pl.adir.scalar, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.scalar, pl.adir.output_existing, pl.adir.scalar]},
                    )
                    o_r_pad_inline329__ssa_v3: pl.Tensor[[384, 8192], pl.FP32, pl.MemRef("mem_ddr_181", pl.const(0, pl.INT64), 12582912)] = ret__tmp_v0_35[0]
                    pa_tid_inline309__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_35[1]
                    proj_a_tids_inline342__ssa_v3: pl.Array[8, pl.TASK_ID] = pl.array.update_element(proj_a_tids_inline342__iter_v1, g_inline293__idx_v0, pa_tid_inline309__ssa_v0)
                    o_r_pad_inline329__rv_v2, proj_a_tids_inline342__rv_v2 = pl.yield_(o_r_pad_inline329__ssa_v3, proj_a_tids_inline342__ssa_v3)
            _submit_deps_buf_inline343__ssa_v0: pl.Array[8, pl.TASK_ID] = pl.array.create(8, dtype=pl.TASK_ID)
            t__tmp_v338: pl.Scalar[pl.TASK_ID] = pl.array.get_element(proj_a_tids_inline342__rv_v2, 0)
            _submit_deps_buf_inline344__ssa_v0: pl.Array[8, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline343__ssa_v0, 0, t__tmp_v338)
            t__tmp_v339: pl.Scalar[pl.TASK_ID] = pl.array.get_element(proj_a_tids_inline342__rv_v2, 1)
            _submit_deps_buf_inline345__ssa_v0: pl.Array[8, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline344__ssa_v0, 1, t__tmp_v339)
            t__tmp_v340: pl.Scalar[pl.TASK_ID] = pl.array.get_element(proj_a_tids_inline342__rv_v2, 2)
            _submit_deps_buf_inline349__ssa_v0: pl.Array[8, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline345__ssa_v0, 2, t__tmp_v340)
            t__tmp_v341: pl.Scalar[pl.TASK_ID] = pl.array.get_element(proj_a_tids_inline342__rv_v2, 3)
            _submit_deps_buf_inline346__ssa_v0: pl.Array[8, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline349__ssa_v0, 3, t__tmp_v341)
            t__tmp_v342: pl.Scalar[pl.TASK_ID] = pl.array.get_element(proj_a_tids_inline342__rv_v2, 4)
            _submit_deps_buf_inline335__ssa_v0: pl.Array[8, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline346__ssa_v0, 4, t__tmp_v342)
            t__tmp_v343: pl.Scalar[pl.TASK_ID] = pl.array.get_element(proj_a_tids_inline342__rv_v2, 5)
            _submit_deps_buf_inline326__ssa_v0: pl.Array[8, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline335__ssa_v0, 5, t__tmp_v343)
            t__tmp_v344: pl.Scalar[pl.TASK_ID] = pl.array.get_element(proj_a_tids_inline342__rv_v2, 6)
            _submit_deps_buf_inline299__ssa_v0: pl.Array[8, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline326__ssa_v0, 6, t__tmp_v344)
            t__tmp_v345: pl.Scalar[pl.TASK_ID] = pl.array.get_element(proj_a_tids_inline342__rv_v2, 7)
            _submit_deps_buf_inline314__ssa_v0: pl.Array[8, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline299__ssa_v0, 7, t__tmp_v345)
            ret__tmp_v0_36: pl.Tuple[pl.Tensor[[1, 384], pl.FP32], pl.Tensor[[1, 384], pl.FP32], pl.Scalar[pl.TASK_ID]] = pl.submit(
                self.oproj_token_scale,
                act_scale_dq_inline313__ssa_v0,
                act_scale_q_inline338__ssa_v0,
                t_dim_inline347__ssa_v0,
                o_r_pad_inline329__rv_v2,
                deps=[_submit_deps_buf_inline314__ssa_v0],
                allow_early_resolve=True,
                attrs={"arg_directions": [pl.adir.output_existing, pl.adir.output_existing, pl.adir.scalar, pl.adir.input]},
            )
            act_scale_dq_inline313__rv_v2: pl.Tensor[[1, 384], pl.FP32, pl.MemRef("mem_ddr_183", pl.const(0, pl.INT64), 1536)] = ret__tmp_v0_36[0]
            act_scale_q_inline338__rv_v2: pl.Tensor[[1, 384], pl.FP32, pl.MemRef("mem_ddr_184", pl.const(0, pl.INT64), 1536)] = ret__tmp_v0_36[1]
            scale_tid_inline350__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_36[2]
            with pl.scope(mode=pl.ScopeMode.MANUAL):
                for g_inline360__idx_v0, (o_r_i8_pad_inline333__iter_v1, partials_inline317__iter_v1, proj_b_tids_inline351__iter_v1) in pl.parallel(
                    8,
                    init_values=(o_r_i8_pad_inline333__ssa_v0, partials_inline317__ssa_v0, proj_b_tids_inline351__ssa_v0),
                    attrs={"iter_arg_rebind_0": False, "iter_arg_rebind_1": False, "iter_arg_rebind_2": True},
                ):
                    col_g_inline295__ssa_v0: pl.Scalar[pl.INDEX] = g_inline360__idx_v0 * 1024
                    ret__tmp_v0_37: pl.Tuple[pl.Tensor[[384, 8192], pl.INT8], pl.Scalar[pl.TASK_ID]] = pl.submit(
                        self.quant,
                        o_r_i8_pad_inline333__iter_v1,
                        t_dim_inline347__ssa_v0,
                        act_scale_q_inline338__rv_v2,
                        o_r_pad_inline329__rv_v2,
                        col_g_inline295__ssa_v0,
                        proj_b_padded_rows_inline323__ssa_v0,
                        deps=[scale_tid_inline350__ssa_v0],
                        allow_early_resolve=True,
                        attrs={"arg_directions": [pl.adir.output_existing, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.scalar, pl.adir.scalar]},
                    )
                    o_r_i8_pad_inline333__rv_v7: pl.Tensor[[384, 8192], pl.INT8, pl.MemRef("mem_ddr_185", pl.const(0, pl.INT64), 3145728)] = ret__tmp_v0_37[0]
                    q_tid_inline311__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_37[1]
                    ret__tmp_v0_38: pl.Tuple[pl.Tensor[[384, 32768], pl.INT32], pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(
                        self.proj_b_mm_spmd,
                        partials_inline317__iter_v1,
                        col_g_inline295__ssa_v0,
                        o_r_i8_pad_inline333__rv_v7,
                        wo_b__ssa_v0,
                        g_inline360__idx_v0,
                        deps=[q_tid_inline311__ssa_v0],
                        core_num=proj_b_t_rows_inline332__ssa_v0 * 8,
                        allow_early_resolve=True,
                        attrs={"arg_directions": [pl.adir.output_existing, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.scalar]},
                    )
                    partials_inline317__rv_v4: pl.Tensor[[384, 32768], pl.INT32, pl.MemRef("mem_ddr_186", pl.const(0, pl.INT64), 50331648)] = ret__tmp_v0_38[0]
                    pb_tid_inline287__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_38[1]
                    proj_b_tids_inline351__ssa_v3: pl.Array[8, pl.TASK_ID] = pl.array.update_element(proj_b_tids_inline351__iter_v1, g_inline360__idx_v0, pb_tid_inline287__ssa_v0)
                    o_r_i8_pad_inline333__rv_v2, partials_inline317__rv_v2, proj_b_tids_inline351__rv_v2 = pl.yield_(
                        o_r_i8_pad_inline333__rv_v7, partials_inline317__rv_v4, proj_b_tids_inline351__ssa_v3
                    )
            _submit_deps_buf_inline327__ssa_v0: pl.Array[8, pl.TASK_ID] = pl.array.create(8, dtype=pl.TASK_ID)
            t__tmp_v353: pl.Scalar[pl.TASK_ID] = pl.array.get_element(proj_b_tids_inline351__rv_v2, 0)
            _submit_deps_buf_inline277__ssa_v0: pl.Array[8, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline327__ssa_v0, 0, t__tmp_v353)
            t__tmp_v354: pl.Scalar[pl.TASK_ID] = pl.array.get_element(proj_b_tids_inline351__rv_v2, 1)
            _submit_deps_buf_inline315__ssa_v0: pl.Array[8, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline277__ssa_v0, 1, t__tmp_v354)
            t__tmp_v355: pl.Scalar[pl.TASK_ID] = pl.array.get_element(proj_b_tids_inline351__rv_v2, 2)
            _submit_deps_buf_inline320__ssa_v0: pl.Array[8, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline315__ssa_v0, 2, t__tmp_v355)
            t__tmp_v356: pl.Scalar[pl.TASK_ID] = pl.array.get_element(proj_b_tids_inline351__rv_v2, 3)
            _submit_deps_buf_inline276__ssa_v0: pl.Array[8, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline320__ssa_v0, 3, t__tmp_v356)
            t__tmp_v357: pl.Scalar[pl.TASK_ID] = pl.array.get_element(proj_b_tids_inline351__rv_v2, 4)
            _submit_deps_buf_inline331__ssa_v0: pl.Array[8, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline276__ssa_v0, 4, t__tmp_v357)
            t__tmp_v358: pl.Scalar[pl.TASK_ID] = pl.array.get_element(proj_b_tids_inline351__rv_v2, 5)
            _submit_deps_buf_inline275__ssa_v0: pl.Array[8, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline331__ssa_v0, 5, t__tmp_v358)
            t__tmp_v359: pl.Scalar[pl.TASK_ID] = pl.array.get_element(proj_b_tids_inline351__rv_v2, 6)
            _submit_deps_buf_inline274__ssa_v0: pl.Array[8, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline275__ssa_v0, 6, t__tmp_v359)
            t__tmp_v360: pl.Scalar[pl.TASK_ID] = pl.array.get_element(proj_b_tids_inline351__rv_v2, 7)
            _submit_deps_buf_inline355__ssa_v0: pl.Array[8, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline274__ssa_v0, 7, t__tmp_v360)
            ret__tmp_v0_39: pl.Tuple[pl.Tensor[[T_DYN, 4096], pl.BF16], pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(
                self.proj_b_act_spmd,
                wo_b_scale__ssa_v0,
                attn_out__ssa_v0,
                t_dim_inline347__ssa_v0,
                partials_inline317__rv_v2,
                act_scale_dq_inline313__rv_v2,
                deps=[_submit_deps_buf_inline355__ssa_v0],
                core_num=act_t_blks_inline312__ssa_v0 * 8,
                allow_early_resolve=True,
                attrs={"arg_directions": [pl.adir.input, pl.adir.output_existing, pl.adir.scalar, pl.adir.input, pl.adir.input]},
            )
            attn_out__rv_v2: pl.Tensor[[T_DYN, 4096], pl.BF16, pl.MemRef("mem_ddr_189", pl.const(0, pl.INT64), 0)] = ret__tmp_v0_39[0]
            _act_tid_inline362__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_39[1]
        return attn_out__rv_v2