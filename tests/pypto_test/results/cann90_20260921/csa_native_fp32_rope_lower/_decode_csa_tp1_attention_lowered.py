# pypto.program: _jit__decode_csa_tp1_attention
import pypto.language as pl

B_DYN = pl.dynamic("B_DYN")
CMP_BLOCK_NUM_DYN = pl.dynamic("CMP_BLOCK_NUM_DYN")
COMPRESSED_TABLE_COLUMNS_DYN = pl.dynamic("COMPRESSED_TABLE_COLUMNS_DYN")
IDX_CACHE_BLOCK_NUM_DYN = pl.dynamic("IDX_CACHE_BLOCK_NUM_DYN")
INDEXER_PAGE_BYTES_DYN = pl.dynamic("INDEXER_PAGE_BYTES_DYN")
INDEXER_TABLE_COLUMNS_DYN = pl.dynamic("INDEXER_TABLE_COLUMNS_DYN")
INNER_STATE_BLOCK_NUM_DYN = pl.dynamic("INNER_STATE_BLOCK_NUM_DYN")
MAIN_STATE_BLOCK_NUM_DYN = pl.dynamic("MAIN_STATE_BLOCK_NUM_DYN")
ORI_BLOCK_NUM_DYN = pl.dynamic("ORI_BLOCK_NUM_DYN")
T_DYN = pl.dynamic("T_DYN")
t_dim_inline2318__ssa_v0 = pl.dynamic("t_dim_inline2318__ssa_v0")


@pl.program
class _jit__decode_csa_tp1_attention:
    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def indexer_topk_query_merge(
        position_ids__ssa_v0: pl.Tensor[[T_DYN], pl.INT32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)],
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
            batch_idx_inline726__ssa_v0: pl.Scalar[pl.INDEX] = query__idx_v0 // 6
            position_inline728__tile: pl.Scalar[pl.INT32] = pl.tensor.read(position_ids__ssa_v0, [query__idx_v0])
            t__tile: pl.Scalar[pl.INT32] = pl.tensor.read(kv_seq_lens__ssa_v0, [batch_idx_inline726__ssa_v0])
            cache_len_inline721__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(t__tile, pl.INDEX) // 4
            cache_bound_inline724__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(cache_len_inline721__ssa_v0, (pl.cast(position_inline728__tile, pl.INDEX) + 1) // 4)
            visible_count_inline725__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(cache_bound_inline724__ssa_v0, 262144)
            if 0 < visible_count_inline725__ssa_v0:
                leaf_count_inline722__ssa_v0: pl.Scalar[pl.INDEX] = (visible_count_inline725__ssa_v0 + 8191) // 8192
                half_count_inline729__ssa_v0: pl.Scalar[pl.INDEX] = leaf_count_inline722__ssa_v0 * 2
                arena_base_inline723__ssa_v0: pl.Scalar[pl.INDEX] = query__idx_v0 * 64
                for child_inline730__idx_v0 in pl.range(1, half_count_inline729__ssa_v0):
                    left_inline2426__ssa_v0: pl.Tile[[1, 1024], pl.FP32, pl.MemRef(mem_vec_5, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.load(
                        pair_arena__ssa_v0, [arena_base_inline723__ssa_v0, 0], [1, 1024], [1, 1024], target_memory=pl.Mem.Vec
                    )
                    right_inline2425__ssa_v0: pl.Tile[[1, 1024], pl.FP32, pl.MemRef(mem_vec_6, pl.const(4096, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.load(
                        pair_arena__ssa_v0, [arena_base_inline723__ssa_v0 + child_inline730__idx_v0, 0], [1, 1024], [1, 1024], target_memory=pl.Mem.Vec
                    )
                    merge_tmp_inline2424__ssa_v0: pl.Tile[[1, 2048], pl.FP32, pl.MemRef(mem_vec_7, pl.const(8192, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.create(
                        [1, 2048], dtype=pl.FP32, target_memory=pl.Mem.Vec
                    )
                    merged_all_inline2423__ssa_v0: pl.Tile[[1, 2048], pl.FP32, pl.MemRef(mem_vec_8, pl.const(16384, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.mrgsort_format2(
                        left_inline2426__ssa_v0, right_inline2425__ssa_v0, merge_tmp_inline2424__ssa_v0, exhausted=False
                    )
                    merged_inline2422__ssa_v0: pl.Tile[[1, 1024], pl.FP32, pl.MemRef(mem_vec_8, pl.const(16384, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.slice(
                        merged_all_inline2423__ssa_v0, [1, 1024], [0, 0]
                    )
                    pl.tile.store(merged_inline2422__ssa_v0, [arena_base_inline723__ssa_v0, 0], pair_arena__ssa_v0)
                root_slot_inline727__ssa_v0: pl.Scalar[pl.INDEX] = arena_base_inline723__ssa_v0
                root_pairs_inline731__ssa_v0: pl.Tile[[1, 1024], pl.FP32, pl.MemRef(mem_vec_7, pl.const(8192, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.load(
                    pair_arena__ssa_v0, [root_slot_inline727__ssa_v0, 0], [1, 1024], [1, 1024], target_memory=pl.Mem.Vec
                )
                root_scores_inline732__ssa_v0: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_8, pl.const(16384, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.gather_mask(
                    root_pairs_inline731__ssa_v0, mask_pattern=1, output_dtype=pl.FP32
                )
                pl.tile.store(root_scores_inline732__ssa_v0, [query__idx_v0, 0], topk_scores__ssa_v0)
                root_indices_inline733__ssa_v0: pl.Tile[[1, 512], pl.INT32, pl.MemRef(mem_vec_8, pl.const(16384, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.gather_mask(
                    root_pairs_inline731__ssa_v0, mask_pattern=2, output_dtype=pl.INT32
                )
                if 512 <= visible_count_inline725__ssa_v0:
                    pl.tile.store(root_indices_inline733__ssa_v0, [query__idx_v0, 0], topk_indices__ssa_v0)
                else:
                    output_indices_inline720__ssa_v0: pl.Tile[[1, 512], pl.INT32, pl.MemRef(mem_vec_7, pl.const(8192, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.full([1, 512], dtype=pl.INT32, value=-1)
                    for lane_inline719__idx_v0 in pl.range(visible_count_inline725__ssa_v0):
                        t__tmp_v1: pl.Scalar[pl.INT32] = pl.tile.read(root_indices_inline733__ssa_v0, [0, lane_inline719__idx_v0])
                        pl.tile.write(output_indices_inline720__ssa_v0, [0, lane_inline719__idx_v0], t__tmp_v1)
                    pl.tile.store(output_indices_inline720__ssa_v0, [query__idx_v0, 0], topk_indices__ssa_v0)
            else:
                t__tmp_v2: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_7, pl.const(8192, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.full([1, 512], dtype=pl.FP32, value=-3.4028234663852886e38)
                pl.tile.store(t__tmp_v2, [query__idx_v0, 0], topk_scores__ssa_v0)
                t__tmp_v3: pl.Tile[[1, 512], pl.INT32, pl.MemRef(mem_vec_7, pl.const(8192, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.full([1, 512], dtype=pl.INT32, value=-1)
                pl.tile.store(t__tmp_v3, [query__idx_v0, 0], topk_indices__ssa_v0)
        return

    @pl.function(type=pl.FunctionType.Spmd)
    def _decode_csa_tp1_attention_indexer_topk_query_merge(
        self,
        position_ids__ssa_v0: pl.Tensor[[T_DYN], pl.INT32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)],
        kv_seq_lens__ssa_v0: pl.Tensor[[B_DYN], pl.INT32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)],
        pair_arena_inline1035_inline2211__ssa_v1: pl.InOut[pl.Tensor[[24576, 1024], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 100663296)]],
        idx_topk_scores__ssa_v0: pl.Out[pl.Tensor[[T_DYN, 512], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)]],
        idx_topk__ssa_v0: pl.Out[pl.Tensor[[T_DYN, 512], pl.INT32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 0)]],
    ):
        pl.func_attr({"mx_tensor_views_blocked": True})
        self.indexer_topk_query_merge(
            position_ids__ssa_v0,
            kv_seq_lens__ssa_v0,
            pair_arena_inline1035_inline2211__ssa_v1,
            idx_topk_scores__ssa_v0,
            idx_topk__ssa_v0,
            attrs={"arg_directions": [pl.adir.input, pl.adir.input, pl.adir.inout, pl.adir.output_existing, pl.adir.output_existing]},
        )

    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def indexer_topk_single_leaf_publish(
        position_ids__ssa_v0: pl.Tensor[[T_DYN], pl.INT32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)],
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
            position__tile: pl.Scalar[pl.INT32] = pl.tensor.read(position_ids__ssa_v0, [query__idx_v0])
            t__tile: pl.Scalar[pl.INT32] = pl.tensor.read(kv_seq_lens__ssa_v0, [query__idx_v0 // 6])
            cache_len__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(t__tile, pl.INDEX) // 4
            visible_count__ssa_v0: pl.Scalar[pl.INDEX] = pl.max(pl.min(cache_len__ssa_v0, (pl.cast(position__tile, pl.INDEX) + 1) // 4), 0)
            if 0 < visible_count__ssa_v0:
                # Sort one populated leaf and directly publish its Top-512 pairs.
                if visible_count__ssa_v0 <= 2048:
                    short_indices_inline990__ci_tmp_v0: pl.Tile[[1, 192], pl.FP32, pl.MemRef(mem_vec_36, pl.const(32768, pl.INT64), 768), pl.Mem.Vec] = pl.tile.create(
                        [1, 192], dtype=pl.FP32, target_memory=pl.Mem.Vec
                    )
                    short_indices_inline990__ssa_v0: pl.Tile[[1, 2048], pl.INT32, pl.MemRef(mem_vec_37, pl.const(98304, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.ci(
                        pl.const(0, pl.INT32), [1, 2048], tmp=short_indices_inline990__ci_tmp_v0, dtype=pl.INT32, descending=False
                    )
                    short_raw_inline980__ssa_v0: pl.Tile[
                        [1, 2048], pl.FP32, pl.MemRef(mem_vec_36, pl.const(32768, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[1, visible_count__ssa_v0])
                    ] = pl.tile.load(score_arena__ssa_v0, [query__idx_v0, 0], [1, 2048], [1, visible_count__ssa_v0], target_memory=pl.Mem.Vec)
                    short_scores_inline982__ssa_v0: pl.Tile[[1, 2048], pl.FP32, pl.MemRef(mem_vec_36, pl.const(32768, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.fillpad(
                        short_raw_inline980__ssa_v0, pad_value=pl.PadValue.min
                    )
                    short_scores_v1_inline999__ssa_v0: pl.Tile[[1, 2048], pl.FP32, pl.MemRef(mem_vec_34, pl.const(0, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.maximums(
                        short_scores_inline982__ssa_v0, -3.4028234663852886e38
                    )
                    t__tmp_v1: pl.Tile[[1, 2048], pl.UINT32, pl.MemRef(mem_vec_37, pl.const(98304, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.reinterpret_view(
                        short_indices_inline990__ssa_v0, dtype=pl.UINT32
                    )
                    short_pairs_inline979__ssa_v0: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_36, pl.const(32768, pl.INT64), 16384), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.sort32(
                        short_scores_v1_inline999__ssa_v0, t__tmp_v1
                    )
                    short_pairs_v1_inline988__ssa_v0: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_37, pl.const(98304, pl.INT64), 16384), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = (
                        pl.tile.mrgsort_format1(short_pairs_inline979__ssa_v0, pl.const(64, pl.INT32))
                    )
                    short_pairs_v2_inline975__ssa_v0: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_36, pl.const(32768, pl.INT64), 16384), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = (
                        pl.tile.mrgsort_format1(short_pairs_v1_inline988__ssa_v0, pl.const(256, pl.INT32))
                    )
                    short_pairs_v3_inline995__ssa_v0: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_37, pl.const(98304, pl.INT64), 16384), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = (
                        pl.tile.mrgsort_format1(short_pairs_v2_inline975__ssa_v0, pl.const(1024, pl.INT32))
                    )
                    short_top_inline974__ssa_v0: pl.Tile[[1, 1024], pl.FP32, pl.MemRef(mem_vec_37, pl.const(98304, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(
                        short_pairs_v3_inline995__ssa_v0, [1, 1024], [0, 0]
                    )
                    short_values_inline985__ssa_v0: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_36, pl.const(32768, pl.INT64), 2048), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = (
                        pl.tile.gather_mask(short_top_inline974__ssa_v0, mask_pattern=1, output_dtype=pl.FP32)
                    )
                    short_selected_inline973__ssa_v0: pl.Tile[[1, 512], pl.INT32, pl.MemRef(mem_vec_34, pl.const(0, pl.INT64), 2048), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = (
                        pl.tile.gather_mask(short_top_inline974__ssa_v0, mask_pattern=2, output_dtype=pl.INT32)
                    )
                    pl.tile.store(short_values_inline985__ssa_v0, [query__idx_v0, 0], topk_scores__ssa_v0)
                    if 512 <= visible_count__ssa_v0:
                        pl.tile.store(short_selected_inline973__ssa_v0, [query__idx_v0, 0], topk_indices__ssa_v0)
                    else:
                        short_output_inline976__ssa_v0: pl.Tile[[1, 512], pl.INT32, pl.MemRef(mem_vec_36, pl.const(32768, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.full(
                            [1, 512], dtype=pl.INT32, value=-1
                        )
                        for lane_inline989__idx_v0 in pl.range(visible_count__ssa_v0):
                            t__tmp_v2: pl.Scalar[pl.INT32] = pl.tile.read(short_selected_inline973__ssa_v0, [0, lane_inline989__idx_v0])
                            pl.tile.write(short_output_inline976__ssa_v0, [0, lane_inline989__idx_v0], t__tmp_v2)
                        pl.tile.store(short_output_inline976__ssa_v0, [query__idx_v0, 0], topk_indices__ssa_v0)
                else:
                    if visible_count__ssa_v0 <= 4096:
                        medium_indices_inline991__ci_tmp_v1: pl.Tile[[1, 192], pl.FP32, pl.MemRef(mem_vec_36, pl.const(32768, pl.INT64), 768), pl.Mem.Vec] = pl.tile.create(
                            [1, 192], dtype=pl.FP32, target_memory=pl.Mem.Vec
                        )
                        medium_indices_inline991__ssa_v0: pl.Tile[[1, 4096], pl.INT32, pl.MemRef(mem_vec_37, pl.const(98304, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.ci(
                            pl.const(0, pl.INT32), [1, 4096], tmp=medium_indices_inline991__ci_tmp_v1, dtype=pl.INT32, descending=False
                        )
                        medium_raw_inline1001__ssa_v0: pl.Tile[
                            [1, 4096], pl.FP32, pl.MemRef(mem_vec_36, pl.const(32768, pl.INT64), 16384), pl.Mem.Vec, pl.TileView(valid_shape=[1, visible_count__ssa_v0])
                        ] = pl.tile.load(score_arena__ssa_v0, [query__idx_v0, 0], [1, 4096], [1, visible_count__ssa_v0], target_memory=pl.Mem.Vec)
                        medium_scores_inline992__ssa_v0: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_36, pl.const(32768, pl.INT64), 16384), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = (
                            pl.tile.fillpad(medium_raw_inline1001__ssa_v0, pad_value=pl.PadValue.min)
                        )
                        medium_scores_v1_inline994__ssa_v0: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_34, pl.const(0, pl.INT64), 16384), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = (
                            pl.tile.maximums(medium_scores_inline992__ssa_v0, -3.4028234663852886e38)
                        )
                        t__tmp_v3: pl.Tile[[1, 4096], pl.UINT32, pl.MemRef(mem_vec_37, pl.const(98304, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.reinterpret_view(
                            medium_indices_inline991__ssa_v0, dtype=pl.UINT32
                        )
                        medium_pairs_inline996__ssa_v0: pl.Tile[[1, 8192], pl.FP32, pl.MemRef(mem_vec_36, pl.const(32768, pl.INT64), 32768), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = (
                            pl.tile.sort32(medium_scores_v1_inline994__ssa_v0, t__tmp_v3)
                        )
                        medium_pairs_v1_inline998__ssa_v0: pl.Tile[[1, 8192], pl.FP32, pl.MemRef(mem_vec_37, pl.const(98304, pl.INT64), 32768), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = (
                            pl.tile.mrgsort_format1(medium_pairs_inline996__ssa_v0, pl.const(64, pl.INT32))
                        )
                        medium_pairs_v2_inline981__ssa_v0: pl.Tile[[1, 8192], pl.FP32, pl.MemRef(mem_vec_36, pl.const(32768, pl.INT64), 32768), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = (
                            pl.tile.mrgsort_format1(medium_pairs_v1_inline998__ssa_v0, pl.const(256, pl.INT32))
                        )
                        medium_pairs_v3_inline986__ssa_v0: pl.Tile[[1, 8192], pl.FP32, pl.MemRef(mem_vec_37, pl.const(98304, pl.INT64), 32768), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = (
                            pl.tile.mrgsort_format1(medium_pairs_v2_inline981__ssa_v0, pl.const(1024, pl.INT32))
                        )
                        medium_left_inline983__ssa_v0: pl.Tile[[1, 1024], pl.FP32, pl.MemRef(mem_vec_37, pl.const(98304, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = (
                            pl.tile.slice(medium_pairs_v3_inline986__ssa_v0, [1, 1024], [0, 0])
                        )
                        medium_right_inline1000__ssa_v0: pl.Tile[[1, 1024], pl.FP32, pl.MemRef(mem_vec_37, pl.const(114688, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = (
                            pl.tile.slice(medium_pairs_v3_inline986__ssa_v0, [1, 1024], [0, 4096])
                        )
                        medium_tmp_inline997__ssa_v0: pl.Tile[[1, 2048], pl.FP32, pl.MemRef(mem_vec_36, pl.const(32768, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.create(
                            [1, 2048], dtype=pl.FP32, target_memory=pl.Mem.Vec
                        )
                        medium_merged_inline1002__ssa_v0: pl.Tile[[1, 2048], pl.FP32, pl.MemRef(mem_vec_34, pl.const(0, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = (
                            pl.tile.mrgsort_format2(medium_left_inline983__ssa_v0, medium_right_inline1000__ssa_v0, medium_tmp_inline997__ssa_v0, exhausted=False)
                        )
                        medium_top_inline977__ssa_v0: pl.Tile[[1, 1024], pl.FP32, pl.MemRef(mem_vec_34, pl.const(0, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(
                            medium_merged_inline1002__ssa_v0, [1, 1024], [0, 0]
                        )
                        medium_values_inline1003__ssa_v0: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_36, pl.const(32768, pl.INT64), 2048), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = (
                            pl.tile.gather_mask(medium_top_inline977__ssa_v0, mask_pattern=1, output_dtype=pl.FP32)
                        )
                        medium_selected_inline993__ssa_v0: pl.Tile[[1, 512], pl.INT32, pl.MemRef(mem_vec_37, pl.const(98304, pl.INT64), 2048), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = (
                            pl.tile.gather_mask(medium_top_inline977__ssa_v0, mask_pattern=2, output_dtype=pl.INT32)
                        )
                        pl.tile.store(medium_values_inline1003__ssa_v0, [query__idx_v0, 0], topk_scores__ssa_v0)
                        pl.tile.store(medium_selected_inline993__ssa_v0, [query__idx_v0, 0], topk_indices__ssa_v0)
                    else:
                        full_indices_inline978__ci_tmp_v2: pl.Tile[[1, 192], pl.FP32, pl.MemRef(mem_vec_36, pl.const(32768, pl.INT64), 768), pl.Mem.Vec] = pl.tile.create(
                            [1, 192], dtype=pl.FP32, target_memory=pl.Mem.Vec
                        )
                        full_indices_inline978__ssa_v0: pl.Tile[[1, 8192], pl.INT32, pl.MemRef(mem_vec_37, pl.const(98304, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.ci(
                            pl.const(0, pl.INT32), [1, 8192], tmp=full_indices_inline978__ci_tmp_v2, dtype=pl.INT32, descending=False
                        )
                        full_raw_inline972__ssa_v0: pl.Tile[
                            [1, 8192], pl.FP32, pl.MemRef(mem_vec_36, pl.const(32768, pl.INT64), 32768), pl.Mem.Vec, pl.TileView(valid_shape=[1, visible_count__ssa_v0])
                        ] = pl.tile.load(score_arena__ssa_v0, [query__idx_v0, 0], [1, 8192], [1, visible_count__ssa_v0], target_memory=pl.Mem.Vec)
                        full_scores_inline971__ssa_v0: pl.Tile[[1, 8192], pl.FP32, pl.MemRef(mem_vec_36, pl.const(32768, pl.INT64), 32768), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = (
                            pl.tile.fillpad(full_raw_inline972__ssa_v0, pad_value=pl.PadValue.min)
                        )
                        full_min_inline969__ssa_v0: pl.Tile[[1, 8192], pl.FP32, pl.MemRef(mem_vec_34, pl.const(0, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.full(
                            [1, 8192], dtype=pl.FP32, value=-3.4028234663852886e38
                        )
                        full_scores_v1_inline968__ssa_v0: pl.Tile[[1, 8192], pl.FP32, pl.MemRef(mem_vec_34, pl.const(0, pl.INT64), 32768), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = (
                            pl.tile.maximum(full_scores_inline971__ssa_v0, full_min_inline969__ssa_v0)
                        )
                        t__tmp_v4: pl.Tile[[1, 8192], pl.UINT32, pl.MemRef(mem_vec_37, pl.const(98304, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.reinterpret_view(
                            full_indices_inline978__ssa_v0, dtype=pl.UINT32
                        )
                        full_pairs_inline970__ssa_v0: pl.Tile[[1, 16384], pl.FP32, pl.MemRef(mem_vec_36, pl.const(32768, pl.INT64), 65536), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = (
                            pl.tile.sort32(full_scores_v1_inline968__ssa_v0, t__tmp_v4)
                        )
                        full_pairs_v1_inline966__ssa_v0: pl.Tile[[1, 16384], pl.FP32, pl.MemRef(mem_vec_37, pl.const(98304, pl.INT64), 65536), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = (
                            pl.tile.mrgsort_format1(full_pairs_inline970__ssa_v0, pl.const(64, pl.INT32))
                        )
                        full_pairs_v2_inline987__ssa_v0: pl.Tile[[1, 16384], pl.FP32, pl.MemRef(mem_vec_36, pl.const(32768, pl.INT64), 65536), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = (
                            pl.tile.mrgsort_format1(full_pairs_v1_inline966__ssa_v0, pl.const(256, pl.INT32))
                        )
                        full_pairs_v3_inline984__ssa_v0: pl.Tile[[1, 16384], pl.FP32, pl.MemRef(mem_vec_37, pl.const(98304, pl.INT64), 65536), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = (
                            pl.tile.mrgsort_format1(full_pairs_v2_inline987__ssa_v0, pl.const(1024, pl.INT32))
                        )
                        full_pairs_v4_inline965__ssa_v0: pl.Tile[[1, 16384], pl.FP32, pl.MemRef(mem_vec_36, pl.const(32768, pl.INT64), 65536), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = (
                            pl.tile.mrgsort_format1(full_pairs_v3_inline984__ssa_v0, pl.const(4096, pl.INT32))
                        )
                        full_top_inline967__ssa_v0: pl.Tile[[1, 1024], pl.FP32, pl.MemRef(mem_vec_36, pl.const(32768, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(
                            full_pairs_v4_inline965__ssa_v0, [1, 1024], [0, 0]
                        )
                        full_values_inline964__ssa_v0: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_37, pl.const(98304, pl.INT64), 2048), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = (
                            pl.tile.gather_mask(full_top_inline967__ssa_v0, mask_pattern=1, output_dtype=pl.FP32)
                        )
                        full_selected_inline963__ssa_v0: pl.Tile[[1, 512], pl.INT32, pl.MemRef(mem_vec_34, pl.const(0, pl.INT64), 2048), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = (
                            pl.tile.gather_mask(full_top_inline967__ssa_v0, mask_pattern=2, output_dtype=pl.INT32)
                        )
                        pl.tile.store(full_values_inline964__ssa_v0, [query__idx_v0, 0], topk_scores__ssa_v0)
                        pl.tile.store(full_selected_inline963__ssa_v0, [query__idx_v0, 0], topk_indices__ssa_v0)
            else:
                t__tmp_v5: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_36, pl.const(32768, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.full([1, 512], dtype=pl.FP32, value=-3.4028234663852886e38)
                pl.tile.store(t__tmp_v5, [query__idx_v0, 0], topk_scores__ssa_v0)
                t__tmp_v6: pl.Tile[[1, 512], pl.INT32, pl.MemRef(mem_vec_36, pl.const(32768, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.full([1, 512], dtype=pl.INT32, value=-1)
                pl.tile.store(t__tmp_v6, [query__idx_v0, 0], topk_indices__ssa_v0)
        return

    @pl.function(type=pl.FunctionType.Spmd)
    def _decode_csa_tp1_attention_indexer_topk_single_leaf_publish(
        self,
        position_ids__ssa_v0: pl.Tensor[[T_DYN], pl.INT32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)],
        kv_seq_lens__ssa_v0: pl.Tensor[[B_DYN], pl.INT32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)],
        score_arena_inline1043_inline2235__rv_v2: pl.Tensor[[384, 8192], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 12582912)],
        idx_topk_scores__ssa_v0: pl.Out[pl.Tensor[[T_DYN, 512], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)]],
        idx_topk__ssa_v0: pl.Out[pl.Tensor[[T_DYN, 512], pl.INT32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 0)]],
    ):
        pl.func_attr({"mx_tensor_views_blocked": True})
        self.indexer_topk_single_leaf_publish(
            position_ids__ssa_v0,
            kv_seq_lens__ssa_v0,
            score_arena_inline1043_inline2235__rv_v2,
            idx_topk_scores__ssa_v0,
            idx_topk__ssa_v0,
            attrs={"arg_directions": [pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.output_existing, pl.adir.output_existing]},
        )

    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def compress_state_commit(
        compress_state_flat_inline1925__ssa_v0: pl.Out[pl.Tensor[[compress_state_rows_inline1905__ssa_v0, 2048], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)]],
        b_dim_inline1929__ssa_v0: pl.Scalar[pl.INDEX],
        commit_workers_inline1920__ssa_v0: pl.Scalar[pl.INDEX],
        s_dim_inline1913__ssa_v0: pl.Scalar[pl.INDEX],
        state_slot_mapping__ssa_v0: pl.Tensor[[T_DYN], pl.INT64, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)],
        position_ids__ssa_v0: pl.Tensor[[T_DYN], pl.INT32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)],
        cmp4_kv_proj_pad_inline1912__ssa_v0: pl.Tensor[[384, 1024], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 1572864)],
        cmp4_score_proj_pad_inline1921__ssa_v0: pl.Tensor[[384, 1024], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 1572864)],
        cmp_ape__ssa_v0: pl.Tensor[[4, 1024], pl.FP32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 16384)],
    ):
        pl.func_attr({"mx_tensor_views_blocked": True})
        mem_vec_6: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 4096)
        mem_vec_8: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 4096)
        commit_worker_inline1926__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        for c_idx_inline1930__idx_v0, (compress_state_flat_inline1925__iter_v1,) in pl.range(
            commit_worker_inline1926__ssa_v0, b_dim_inline1929__ssa_v0, commit_workers_inline1920__ssa_v0, init_values=(compress_state_flat_inline1925__ssa_v0,)
        ):
            for s_idx_inline1918__idx_v0, (compress_state_flat_inline1925__iter_v3,) in pl.range(s_dim_inline1913__ssa_v0, init_values=(compress_state_flat_inline1925__iter_v1,)):
                token_inline1931__ssa_v0: pl.Scalar[pl.INDEX] = c_idx_inline1930__idx_v0 * s_dim_inline1913__ssa_v0 + s_idx_inline1918__idx_v0
                state_row_i64_inline1934__tile: pl.Scalar[pl.INT64] = pl.tensor.read(state_slot_mapping__ssa_v0, [token_inline1931__ssa_v0])
                if 0 <= state_row_i64_inline1934__tile:
                    state_row_inline1936__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(state_row_i64_inline1934__tile, pl.INDEX)
                    token_pos_inline1938__tile: pl.Scalar[pl.INT32] = pl.tensor.read(position_ids__ssa_v0, [token_inline1931__ssa_v0])
                    ape_row_inline1907__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(pl.cast(token_pos_inline1938__tile, pl.INDEX) % 4, pl.INDEX)
                    t__tile: pl.Tile[[1, 1024], pl.FP32, pl.MemRef(mem_vec_6, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.load(
                        cmp4_kv_proj_pad_inline1912__ssa_v0, [token_inline1931__ssa_v0, 0], [1, 1024], [1, 1024], target_memory=pl.Mem.Vec
                    )
                    compress_state_flat_inline1925__tile: pl.Tensor[[compress_state_rows_inline1905__ssa_v0, 2048], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                        t__tile, [state_row_inline1936__ssa_v0, 0], compress_state_flat_inline1925__iter_v3
                    )
                    t__tile_1: pl.Tile[[1, 1024], pl.FP32, pl.MemRef(mem_vec_6, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.load(
                        cmp4_score_proj_pad_inline1921__ssa_v0, [token_inline1931__ssa_v0, 0], [1, 1024], [1, 1024], target_memory=pl.Mem.Vec
                    )
                    t__tile_2: pl.Tile[[1, 1024], pl.FP32, pl.MemRef(mem_vec_8, pl.const(4096, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.load(
                        cmp_ape__ssa_v0, [ape_row_inline1907__ssa_v0, 0], [1, 1024], [1, 1024], target_memory=pl.Mem.Vec
                    )
                    t__tile_3: pl.Tile[[1, 1024], pl.FP32, pl.MemRef(mem_vec_6, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.add(t__tile_1, t__tile_2)
                    compress_state_flat_inline1925__tile_1: pl.Tensor[[compress_state_rows_inline1905__ssa_v0, 2048], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                        t__tile_3, [state_row_inline1936__ssa_v0, 1024], compress_state_flat_inline1925__tile
                    )
                    compress_state_flat_inline1925__phi_v7: pl.Tensor[[compress_state_rows_inline1905__ssa_v0, 2048], pl.FP32, pl.MemRef("mem_ddr_10", pl.const(0, pl.INT64), 0)] = pl.yield_(
                        compress_state_flat_inline1925__tile_1
                    )
                else:
                    compress_state_flat_inline1925__phi_v7: pl.Tensor[[compress_state_rows_inline1905__ssa_v0, 2048], pl.FP32, pl.MemRef("mem_ddr_10", pl.const(0, pl.INT64), 0)] = pl.yield_(
                        compress_state_flat_inline1925__iter_v3
                    )
                compress_state_flat_inline1925__rv_v4: pl.Tensor[[compress_state_rows_inline1905__ssa_v0, 2048], pl.FP32, pl.MemRef("mem_ddr_11", pl.const(0, pl.INT64), 0)] = pl.yield_(
                    compress_state_flat_inline1925__phi_v7
                )
            compress_state_flat_inline1925__rv_v2: pl.Tensor[[compress_state_rows_inline1905__ssa_v0, 2048], pl.FP32, pl.MemRef("mem_ddr_12", pl.const(0, pl.INT64), 0)] = pl.yield_(
                compress_state_flat_inline1925__rv_v4
            )
        return

    @pl.function(type=pl.FunctionType.Spmd)
    def compress_state_commit_spmd(
        self,
        compress_state_flat_inline1925__ssa_v0: pl.Out[pl.Tensor[[compress_state_rows_inline1905__ssa_v0, 2048], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)]],
        b_dim_inline1929__ssa_v0: pl.Scalar[pl.INDEX],
        commit_workers_inline1920__ssa_v0: pl.Scalar[pl.INDEX],
        s_dim_inline1913__ssa_v0: pl.Scalar[pl.INDEX],
        state_slot_mapping__ssa_v0: pl.Tensor[[T_DYN], pl.INT64, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)],
        position_ids__ssa_v0: pl.Tensor[[T_DYN], pl.INT32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)],
        cmp4_kv_proj_pad_inline1912__ssa_v0: pl.Tensor[[384, 1024], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 1572864)],
        cmp4_score_proj_pad_inline1921__ssa_v0: pl.Tensor[[384, 1024], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 1572864)],
        cmp_ape__ssa_v0: pl.Tensor[[4, 1024], pl.FP32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 16384)],
    ):
        pl.func_attr({"mx_tensor_views_blocked": True})
        self.compress_state_commit(
            compress_state_flat_inline1925__ssa_v0,
            b_dim_inline1929__ssa_v0,
            commit_workers_inline1920__ssa_v0,
            s_dim_inline1913__ssa_v0,
            state_slot_mapping__ssa_v0,
            position_ids__ssa_v0,
            cmp4_kv_proj_pad_inline1912__ssa_v0,
            cmp4_score_proj_pad_inline1921__ssa_v0,
            cmp_ape__ssa_v0,
            attrs={"arg_directions": [pl.adir.output_existing, pl.adir.scalar, pl.adir.scalar, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input]},
        )

    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def compress_state_commit_0(
        compress_state_flat_inline440_inline1999__ssa_v0: pl.Out[pl.Tensor[[compress_state_rows_inline425_inline2017__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)]],
        b_dim_inline445_inline1977__ssa_v0: pl.Scalar[pl.INDEX],
        commit_workers_inline462_inline2046__ssa_v0: pl.Scalar[pl.INDEX],
        s_dim_inline427_inline2038__ssa_v0: pl.Scalar[pl.INDEX],
        inner_state_slot_mapping__ssa_v0: pl.Tensor[[T_DYN], pl.INT64, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)],
        position_ids__ssa_v0: pl.Tensor[[T_DYN], pl.INT32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)],
        kv_proj_pad_inline1986__rv_v2: pl.Tensor[[384, 256], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 393216)],
        score_proj_pad_inline1993__rv_v2: pl.Tensor[[384, 256], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 393216)],
        inner_ape__ssa_v0: pl.Tensor[[4, 256], pl.FP32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 4096)],
    ):
        pl.func_attr({"mx_tensor_views_blocked": True})
        mem_vec_6: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 1024)
        mem_vec_8: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 1024)
        commit_worker_inline463_inline2003__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        for c_idx_inline456_inline1972__idx_v0, (compress_state_flat_inline440_inline1999__iter_v1,) in pl.range(
            commit_worker_inline463_inline2003__ssa_v0, b_dim_inline445_inline1977__ssa_v0, commit_workers_inline462_inline2046__ssa_v0, init_values=(compress_state_flat_inline440_inline1999__ssa_v0,)
        ):
            for s_idx_inline412_inline1984__idx_v0, (compress_state_flat_inline440_inline1999__iter_v3,) in pl.range(
                s_dim_inline427_inline2038__ssa_v0, init_values=(compress_state_flat_inline440_inline1999__iter_v1,)
            ):
                token_inline428_inline2029__ssa_v1: pl.Scalar[pl.INDEX] = c_idx_inline456_inline1972__idx_v0 * s_dim_inline427_inline2038__ssa_v0 + s_idx_inline412_inline1984__idx_v0
                state_row_i64_inline464_inline1971__tile: pl.Scalar[pl.INT64] = pl.tensor.read(inner_state_slot_mapping__ssa_v0, [token_inline428_inline2029__ssa_v1])
                if 0 <= state_row_i64_inline464_inline1971__tile:
                    state_row_inline451_inline2035__ssa_v1: pl.Scalar[pl.INDEX] = pl.cast(state_row_i64_inline464_inline1971__tile, pl.INDEX)
                    token_pos_inline437_inline2007__tile: pl.Scalar[pl.INT32] = pl.tensor.read(position_ids__ssa_v0, [token_inline428_inline2029__ssa_v1])
                    ape_row_inline454_inline2027__ssa_v1: pl.Scalar[pl.INDEX] = pl.cast(pl.cast(token_pos_inline437_inline2007__tile, pl.INDEX) % 4, pl.INDEX)
                    t__tile: pl.Tile[[1, 256], pl.FP32, pl.MemRef(mem_vec_6, pl.const(0, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.load(
                        kv_proj_pad_inline1986__rv_v2, [token_inline428_inline2029__ssa_v1, 0], [1, 256], [1, 256], target_memory=pl.Mem.Vec
                    )
                    compress_state_flat_inline440_inline1999__tile: pl.Tensor[[compress_state_rows_inline425_inline2017__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)] = (
                        pl.tile.store(t__tile, [state_row_inline451_inline2035__ssa_v1, 0], compress_state_flat_inline440_inline1999__iter_v3)
                    )
                    t__tile_1: pl.Tile[[1, 256], pl.FP32, pl.MemRef(mem_vec_6, pl.const(0, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.load(
                        score_proj_pad_inline1993__rv_v2, [token_inline428_inline2029__ssa_v1, 0], [1, 256], [1, 256], target_memory=pl.Mem.Vec
                    )
                    t__tile_2: pl.Tile[[1, 256], pl.FP32, pl.MemRef(mem_vec_8, pl.const(1024, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.load(
                        inner_ape__ssa_v0, [ape_row_inline454_inline2027__ssa_v1, 0], [1, 256], [1, 256], target_memory=pl.Mem.Vec
                    )
                    t__tile_3: pl.Tile[[1, 256], pl.FP32, pl.MemRef(mem_vec_6, pl.const(0, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.add(t__tile_1, t__tile_2)
                    compress_state_flat_inline440_inline1999__tile_1: pl.Tensor[[compress_state_rows_inline425_inline2017__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)] = (
                        pl.tile.store(t__tile_3, [state_row_inline451_inline2035__ssa_v1, 256], compress_state_flat_inline440_inline1999__tile)
                    )
                    compress_state_flat_inline440_inline1999__phi_v7: pl.Tensor[[compress_state_rows_inline425_inline2017__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_10", pl.const(0, pl.INT64), 0)] = (
                        pl.yield_(compress_state_flat_inline440_inline1999__tile_1)
                    )
                else:
                    compress_state_flat_inline440_inline1999__phi_v7: pl.Tensor[[compress_state_rows_inline425_inline2017__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_10", pl.const(0, pl.INT64), 0)] = (
                        pl.yield_(compress_state_flat_inline440_inline1999__iter_v3)
                    )
                compress_state_flat_inline440_inline1999__rv_v4: pl.Tensor[[compress_state_rows_inline425_inline2017__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_11", pl.const(0, pl.INT64), 0)] = (
                    pl.yield_(compress_state_flat_inline440_inline1999__phi_v7)
                )
            compress_state_flat_inline440_inline1999__rv_v2: pl.Tensor[[compress_state_rows_inline425_inline2017__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_12", pl.const(0, pl.INT64), 0)] = pl.yield_(
                compress_state_flat_inline440_inline1999__rv_v4
            )
        return

    @pl.function(type=pl.FunctionType.Spmd)
    def compress_state_commit_spmd_0(
        self,
        compress_state_flat_inline440_inline1999__ssa_v0: pl.Out[pl.Tensor[[compress_state_rows_inline425_inline2017__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)]],
        b_dim_inline445_inline1977__ssa_v0: pl.Scalar[pl.INDEX],
        commit_workers_inline462_inline2046__ssa_v0: pl.Scalar[pl.INDEX],
        s_dim_inline427_inline2038__ssa_v0: pl.Scalar[pl.INDEX],
        inner_state_slot_mapping__ssa_v0: pl.Tensor[[T_DYN], pl.INT64, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)],
        position_ids__ssa_v0: pl.Tensor[[T_DYN], pl.INT32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)],
        kv_proj_pad_inline1986__rv_v2: pl.Tensor[[384, 256], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 393216)],
        score_proj_pad_inline1993__rv_v2: pl.Tensor[[384, 256], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 393216)],
        inner_ape__ssa_v0: pl.Tensor[[4, 256], pl.FP32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 4096)],
    ):
        pl.func_attr({"mx_tensor_views_blocked": True})
        self.compress_state_commit_0(
            compress_state_flat_inline440_inline1999__ssa_v0,
            b_dim_inline445_inline1977__ssa_v0,
            commit_workers_inline462_inline2046__ssa_v0,
            s_dim_inline427_inline2038__ssa_v0,
            inner_state_slot_mapping__ssa_v0,
            position_ids__ssa_v0,
            kv_proj_pad_inline1986__rv_v2,
            score_proj_pad_inline1993__rv_v2,
            inner_ape__ssa_v0,
            attrs={"arg_directions": [pl.adir.output_existing, pl.adir.scalar, pl.adir.scalar, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input]},
        )

    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def csa_cache_writeback(
        kv_cache_flat__ssa_v0: pl.Out[pl.Tensor[[ORI_BLOCK_NUM_DYN * pl.const(32, pl.INDEX), 512], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)]],
        wb_blocks__ssa_v0: pl.Scalar[pl.INDEX],
        ori_slot_mapping__ssa_v0: pl.Tensor[[T_DYN], pl.INT64, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)],
        kv__ssa_v0: pl.Tensor[[T_DYN, 512], pl.BF16, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)],
    ):
        pl.func_attr({"mx_tensor_views_blocked": True})
        mem_vec_3: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 1024)
        wb_worker__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        for wb_blk__idx_v0, (kv_cache_flat__iter_v1,) in pl.range(wb_worker__ssa_v0, wb_blocks__ssa_v0, 8, init_values=(kv_cache_flat__ssa_v0,)):
            wb_t0__ssa_v0: pl.Scalar[pl.INDEX] = wb_blk__idx_v0 * 8
            for write_dt__idx_v0, (kv_cache_flat__iter_v3,) in pl.range(8, init_values=(kv_cache_flat__iter_v1,)):
                write_t__ssa_v0: pl.Scalar[pl.INDEX] = wb_t0__ssa_v0 + write_dt__idx_v0
                write_row_i64__tile: pl.Scalar[pl.INT64] = pl.tensor.read(ori_slot_mapping__ssa_v0, [write_t__ssa_v0])
                if 0 <= write_row_i64__tile:
                    write_row__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(write_row_i64__tile, pl.INDEX)
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
        ori_slot_mapping__ssa_v0: pl.Tensor[[T_DYN], pl.INT64, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)],
        kv__ssa_v0: pl.Tensor[[T_DYN, 512], pl.BF16, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)],
    ):
        pl.func_attr({"mx_tensor_views_blocked": True})
        self.csa_cache_writeback(
            kv_cache_flat__ssa_v0, wb_blocks__ssa_v0, ori_slot_mapping__ssa_v0, kv__ssa_v0, attrs={"arg_directions": [pl.adir.output_existing, pl.adir.scalar, pl.adir.input, pl.adir.input]}
        )

    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def csa_rope_interleave(
        cmp_cos_il__ssa_v0: pl.Out[pl.Tensor[[T_DYN, 64], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)]],
        cmp_sin_signed__ssa_v0: pl.Out[pl.Tensor[[T_DYN, 64], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)]],
        idx_cos_il__ssa_v0: pl.Out[pl.Tensor[[T_DYN, 64], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)]],
        idx_sin_signed__ssa_v0: pl.Out[pl.Tensor[[T_DYN, 64], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)]],
        inner_cos_il__ssa_v0: pl.Out[pl.Tensor[[T_DYN, 64], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 0)]],
        inner_sin_signed__ssa_v0: pl.Out[pl.Tensor[[T_DYN, 64], pl.FP32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 0)]],
        T_DYN: pl.Scalar[pl.INDEX],
        freqs_cos__ssa_v0: pl.Tensor[[T_DYN, 64], pl.FP32, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 0)],
        freqs_sin__ssa_v0: pl.Tensor[[T_DYN, 64], pl.FP32, pl.MemRef("mem_ddr_7", pl.const(0, pl.INT64), 0)],
        cmp_freqs_cos__ssa_v0: pl.Tensor[[T_DYN, 64], pl.FP32, pl.MemRef("mem_ddr_8", pl.const(0, pl.INT64), 0)],
        cmp_freqs_sin__ssa_v0: pl.Tensor[[T_DYN, 64], pl.FP32, pl.MemRef("mem_ddr_9", pl.const(0, pl.INT64), 0)],
        inner_freqs_cos__ssa_v0: pl.Tensor[[T_DYN, 64], pl.FP32, pl.MemRef("mem_ddr_10", pl.const(0, pl.INT64), 0)],
        inner_freqs_sin__ssa_v0: pl.Tensor[[T_DYN, 64], pl.FP32, pl.MemRef("mem_ddr_11", pl.const(0, pl.INT64), 0)],
    ) -> tuple[
        pl.Tensor[[T_DYN, 64], pl.FP32],
        pl.Tensor[[T_DYN, 64], pl.FP32],
        pl.Tensor[[T_DYN, 64], pl.FP32],
        pl.Tensor[[T_DYN, 64], pl.FP32],
        pl.Tensor[[T_DYN, 64], pl.FP32],
        pl.Tensor[[T_DYN, 64], pl.FP32],
    ]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        mem_vec_12: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 1024)
        mem_vec_17: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 1024)
        mem_vec_20: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 1024)
        mem_vec_25: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 512)
        mem_vec_27: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        mem_vec_28: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        il_ones__tile: pl.Tile[[4, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(0, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.full([4, 64], dtype=pl.FP32, value=1.0)
        t__ci_tmp_v0: pl.Tile[[1, 192], pl.FP32, pl.MemRef(mem_vec_17, pl.const(1024, pl.INT64), 768), pl.Mem.Vec] = pl.tile.create([1, 192], dtype=pl.FP32, target_memory=pl.Mem.Vec)
        t__tile: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_20, pl.const(2048, pl.INT64), 256), pl.Mem.Vec] = pl.tile.ci(
            pl.const(0, pl.INT32), [1, 64], tmp=t__ci_tmp_v0, dtype=pl.INT32, descending=False
        )
        il_lane_ids__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_17, pl.const(1024, pl.INT64), 256), pl.Mem.Vec] = pl.tile.cast(t__tile, target_type=pl.FP32, mode="round")
        il_col__tile: pl.Tile[[4, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(0, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.col_expand_mul(il_ones__tile, il_lane_ids__tile)
        t__tile_1: pl.Tile[[4, 64], pl.FP32, pl.MemRef(mem_vec_17, pl.const(1024, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.muls(il_col__tile, 0.5)
        t__tile_2: pl.Tile[[4, 64], pl.INT32, pl.MemRef(mem_vec_17, pl.const(1024, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.cast(t__tile_1, target_type=pl.INT32, mode="trunc")
        il_dup_f__tile: pl.Tile[[4, 64], pl.FP32, pl.MemRef(mem_vec_17, pl.const(1024, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.cast(t__tile_2, target_type=pl.FP32, mode="round")
        il_dup_idx__tile: pl.Tile[[4, 64], pl.INT32, pl.MemRef(mem_vec_20, pl.const(2048, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.cast(il_dup_f__tile, target_type=pl.INT32, mode="round")
        t__tile_3: pl.Tile[[4, 64], pl.FP32, pl.MemRef(mem_vec_17, pl.const(1024, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.muls(il_dup_f__tile, 2.0)
        il_lane__tile: pl.Tile[[4, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(0, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.sub(il_col__tile, t__tile_3)
        t__tile_4: pl.Tile[[4, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(0, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.muls(il_lane__tile, 2.0)
        il_sign__tile: pl.Tile[[4, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(0, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.subs(t__tile_4, 1.0)
        for rope_t0__idx_v0, (cmp_cos_il__iter_v1, cmp_sin_signed__iter_v1, idx_cos_il__iter_v1, idx_sin_signed__iter_v1, inner_cos_il__iter_v1, inner_sin_signed__iter_v1) in pl.range(
            0, T_DYN, 4, init_values=(cmp_cos_il__ssa_v0, cmp_sin_signed__ssa_v0, idx_cos_il__ssa_v0, idx_sin_signed__ssa_v0, inner_cos_il__ssa_v0, inner_sin_signed__ssa_v0)
        ):
            idx_cos_half__tile: pl.Tile[[4, 32], pl.FP32, pl.MemRef(mem_vec_25, pl.const(3072, pl.INT64), 512), pl.Mem.Vec] = pl.tile.load(
                freqs_cos__ssa_v0, [rope_t0__idx_v0, 0], [4, 32], [4, 32], target_memory=pl.Mem.Vec
            )
            gather_acc_init: pl.Tile[[4, 64], pl.FP32, pl.MemRef(mem_vec_17, pl.const(1024, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.create([4, 64], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            for gather_lv, (gather_ia,) in pl.range(4, init_values=(gather_acc_init,)):
                gather_inp_row: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_25, pl.const(3072, pl.INT64), 128), pl.Mem.Vec] = pl.tile.slice(idx_cos_half__tile, [1, 32], [gather_lv, 0], [1, 32])
                gather_idx_row: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_20, pl.const(2048, pl.INT64), 256), pl.Mem.Vec] = pl.tile.slice(il_dup_idx__tile, [1, 64], [gather_lv, 0], [1, 64])
                gather_row_tmp: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_27, pl.const(3584, pl.INT64), 256), pl.Mem.Vec] = pl.tile.create([1, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec)
                gather_row: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_28, pl.const(3840, pl.INT64), 256), pl.Mem.Vec] = pl.tile.gather(gather_inp_row, gather_idx_row, gather_row_tmp)
                gather_asmbl: pl.Tile[[4, 64], pl.FP32, pl.MemRef(mem_vec_17, pl.const(1024, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.assemble(gather_ia, gather_row, [gather_lv, 0])
                gather_rv: pl.Tile[[4, 64], pl.FP32, pl.MemRef(mem_vec_17, pl.const(1024, pl.INT64), 1024), pl.Mem.Vec] = pl.yield_(gather_asmbl)
            t__tile_5: pl.Tile[[4, 64], pl.FP32, pl.MemRef(mem_vec_17, pl.const(1024, pl.INT64), 1024), pl.Mem.Vec] = gather_rv
            idx_cos_il__tile: pl.Tensor[[T_DYN, 64], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)] = pl.tile.store(t__tile_5, [rope_t0__idx_v0, 0], idx_cos_il__iter_v1)
            idx_sin_half__tile: pl.Tile[[4, 32], pl.FP32, pl.MemRef(mem_vec_25, pl.const(3072, pl.INT64), 512), pl.Mem.Vec] = pl.tile.load(
                freqs_sin__ssa_v0, [rope_t0__idx_v0, 0], [4, 32], [4, 32], target_memory=pl.Mem.Vec
            )
            gather_acc_init_1: pl.Tile[[4, 64], pl.FP32, pl.MemRef(mem_vec_17, pl.const(1024, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.create([4, 64], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            for gather_lv_1, (gather_ia_1,) in pl.range(4, init_values=(gather_acc_init_1,)):
                gather_inp_row_1: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_25, pl.const(3072, pl.INT64), 128), pl.Mem.Vec] = pl.tile.slice(idx_sin_half__tile, [1, 32], [gather_lv_1, 0], [1, 32])
                gather_idx_row_1: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_20, pl.const(2048, pl.INT64), 256), pl.Mem.Vec] = pl.tile.slice(il_dup_idx__tile, [1, 64], [gather_lv_1, 0], [1, 64])
                gather_row_tmp_1: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_27, pl.const(3584, pl.INT64), 256), pl.Mem.Vec] = pl.tile.create([1, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec)
                gather_row_1: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_28, pl.const(3840, pl.INT64), 256), pl.Mem.Vec] = pl.tile.gather(gather_inp_row_1, gather_idx_row_1, gather_row_tmp_1)
                gather_asmbl_1: pl.Tile[[4, 64], pl.FP32, pl.MemRef(mem_vec_17, pl.const(1024, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.assemble(gather_ia_1, gather_row_1, [gather_lv_1, 0])
                gather_rv_1: pl.Tile[[4, 64], pl.FP32, pl.MemRef(mem_vec_17, pl.const(1024, pl.INT64), 1024), pl.Mem.Vec] = pl.yield_(gather_asmbl_1)
            idx_sin_il__tile: pl.Tile[[4, 64], pl.FP32, pl.MemRef(mem_vec_17, pl.const(1024, pl.INT64), 1024), pl.Mem.Vec] = gather_rv_1
            t__tile_6: pl.Tile[[4, 64], pl.FP32, pl.MemRef(mem_vec_17, pl.const(1024, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.mul(idx_sin_il__tile, il_sign__tile)
            idx_sin_signed__tile: pl.Tensor[[T_DYN, 64], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)] = pl.tile.store(t__tile_6, [rope_t0__idx_v0, 0], idx_sin_signed__iter_v1)
            cmp_cos_half__tile: pl.Tile[[4, 32], pl.FP32, pl.MemRef(mem_vec_25, pl.const(3072, pl.INT64), 512), pl.Mem.Vec] = pl.tile.load(
                cmp_freqs_cos__ssa_v0, [rope_t0__idx_v0, 0], [4, 32], [4, 32], target_memory=pl.Mem.Vec
            )
            gather_acc_init_2: pl.Tile[[4, 64], pl.FP32, pl.MemRef(mem_vec_17, pl.const(1024, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.create([4, 64], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            for gather_lv_2, (gather_ia_2,) in pl.range(4, init_values=(gather_acc_init_2,)):
                gather_inp_row_2: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_25, pl.const(3072, pl.INT64), 128), pl.Mem.Vec] = pl.tile.slice(cmp_cos_half__tile, [1, 32], [gather_lv_2, 0], [1, 32])
                gather_idx_row_2: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_20, pl.const(2048, pl.INT64), 256), pl.Mem.Vec] = pl.tile.slice(il_dup_idx__tile, [1, 64], [gather_lv_2, 0], [1, 64])
                gather_row_tmp_2: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_27, pl.const(3584, pl.INT64), 256), pl.Mem.Vec] = pl.tile.create([1, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec)
                gather_row_2: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_28, pl.const(3840, pl.INT64), 256), pl.Mem.Vec] = pl.tile.gather(gather_inp_row_2, gather_idx_row_2, gather_row_tmp_2)
                gather_asmbl_2: pl.Tile[[4, 64], pl.FP32, pl.MemRef(mem_vec_17, pl.const(1024, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.assemble(gather_ia_2, gather_row_2, [gather_lv_2, 0])
                gather_rv_2: pl.Tile[[4, 64], pl.FP32, pl.MemRef(mem_vec_17, pl.const(1024, pl.INT64), 1024), pl.Mem.Vec] = pl.yield_(gather_asmbl_2)
            t__tile_7: pl.Tile[[4, 64], pl.FP32, pl.MemRef(mem_vec_17, pl.const(1024, pl.INT64), 1024), pl.Mem.Vec] = gather_rv_2
            cmp_cos_il__tile: pl.Tensor[[T_DYN, 64], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)] = pl.tile.store(t__tile_7, [rope_t0__idx_v0, 0], cmp_cos_il__iter_v1)
            cmp_sin_half__tile: pl.Tile[[4, 32], pl.FP32, pl.MemRef(mem_vec_25, pl.const(3072, pl.INT64), 512), pl.Mem.Vec] = pl.tile.load(
                cmp_freqs_sin__ssa_v0, [rope_t0__idx_v0, 0], [4, 32], [4, 32], target_memory=pl.Mem.Vec
            )
            gather_acc_init_3: pl.Tile[[4, 64], pl.FP32, pl.MemRef(mem_vec_17, pl.const(1024, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.create([4, 64], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            for gather_lv_3, (gather_ia_3,) in pl.range(4, init_values=(gather_acc_init_3,)):
                gather_inp_row_3: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_25, pl.const(3072, pl.INT64), 128), pl.Mem.Vec] = pl.tile.slice(cmp_sin_half__tile, [1, 32], [gather_lv_3, 0], [1, 32])
                gather_idx_row_3: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_20, pl.const(2048, pl.INT64), 256), pl.Mem.Vec] = pl.tile.slice(il_dup_idx__tile, [1, 64], [gather_lv_3, 0], [1, 64])
                gather_row_tmp_3: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_27, pl.const(3584, pl.INT64), 256), pl.Mem.Vec] = pl.tile.create([1, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec)
                gather_row_3: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_28, pl.const(3840, pl.INT64), 256), pl.Mem.Vec] = pl.tile.gather(gather_inp_row_3, gather_idx_row_3, gather_row_tmp_3)
                gather_asmbl_3: pl.Tile[[4, 64], pl.FP32, pl.MemRef(mem_vec_17, pl.const(1024, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.assemble(gather_ia_3, gather_row_3, [gather_lv_3, 0])
                gather_rv_3: pl.Tile[[4, 64], pl.FP32, pl.MemRef(mem_vec_17, pl.const(1024, pl.INT64), 1024), pl.Mem.Vec] = pl.yield_(gather_asmbl_3)
            cmp_sin_il__tile: pl.Tile[[4, 64], pl.FP32, pl.MemRef(mem_vec_17, pl.const(1024, pl.INT64), 1024), pl.Mem.Vec] = gather_rv_3
            t__tile_8: pl.Tile[[4, 64], pl.FP32, pl.MemRef(mem_vec_17, pl.const(1024, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.mul(cmp_sin_il__tile, il_sign__tile)
            cmp_sin_signed__tile: pl.Tensor[[T_DYN, 64], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)] = pl.tile.store(t__tile_8, [rope_t0__idx_v0, 0], cmp_sin_signed__iter_v1)
            inner_cos_half__tile: pl.Tile[[4, 32], pl.FP32, pl.MemRef(mem_vec_25, pl.const(3072, pl.INT64), 512), pl.Mem.Vec] = pl.tile.load(
                inner_freqs_cos__ssa_v0, [rope_t0__idx_v0, 0], [4, 32], [4, 32], target_memory=pl.Mem.Vec
            )
            gather_acc_init_4: pl.Tile[[4, 64], pl.FP32, pl.MemRef(mem_vec_17, pl.const(1024, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.create([4, 64], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            for gather_lv_4, (gather_ia_4,) in pl.range(4, init_values=(gather_acc_init_4,)):
                gather_inp_row_4: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_25, pl.const(3072, pl.INT64), 128), pl.Mem.Vec] = pl.tile.slice(inner_cos_half__tile, [1, 32], [gather_lv_4, 0], [1, 32])
                gather_idx_row_4: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_20, pl.const(2048, pl.INT64), 256), pl.Mem.Vec] = pl.tile.slice(il_dup_idx__tile, [1, 64], [gather_lv_4, 0], [1, 64])
                gather_row_tmp_4: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_27, pl.const(3584, pl.INT64), 256), pl.Mem.Vec] = pl.tile.create([1, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec)
                gather_row_4: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_28, pl.const(3840, pl.INT64), 256), pl.Mem.Vec] = pl.tile.gather(gather_inp_row_4, gather_idx_row_4, gather_row_tmp_4)
                gather_asmbl_4: pl.Tile[[4, 64], pl.FP32, pl.MemRef(mem_vec_17, pl.const(1024, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.assemble(gather_ia_4, gather_row_4, [gather_lv_4, 0])
                gather_rv_4: pl.Tile[[4, 64], pl.FP32, pl.MemRef(mem_vec_17, pl.const(1024, pl.INT64), 1024), pl.Mem.Vec] = pl.yield_(gather_asmbl_4)
            t__tile_9: pl.Tile[[4, 64], pl.FP32, pl.MemRef(mem_vec_17, pl.const(1024, pl.INT64), 1024), pl.Mem.Vec] = gather_rv_4
            inner_cos_il__tile: pl.Tensor[[T_DYN, 64], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 0)] = pl.tile.store(t__tile_9, [rope_t0__idx_v0, 0], inner_cos_il__iter_v1)
            inner_sin_half__tile: pl.Tile[[4, 32], pl.FP32, pl.MemRef(mem_vec_25, pl.const(3072, pl.INT64), 512), pl.Mem.Vec] = pl.tile.load(
                inner_freqs_sin__ssa_v0, [rope_t0__idx_v0, 0], [4, 32], [4, 32], target_memory=pl.Mem.Vec
            )
            gather_acc_init_5: pl.Tile[[4, 64], pl.FP32, pl.MemRef(mem_vec_17, pl.const(1024, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.create([4, 64], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            for gather_lv_5, (gather_ia_5,) in pl.range(4, init_values=(gather_acc_init_5,)):
                gather_inp_row_5: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_25, pl.const(3072, pl.INT64), 128), pl.Mem.Vec] = pl.tile.slice(inner_sin_half__tile, [1, 32], [gather_lv_5, 0], [1, 32])
                gather_idx_row_5: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_20, pl.const(2048, pl.INT64), 256), pl.Mem.Vec] = pl.tile.slice(il_dup_idx__tile, [1, 64], [gather_lv_5, 0], [1, 64])
                gather_row_tmp_5: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_27, pl.const(3584, pl.INT64), 256), pl.Mem.Vec] = pl.tile.create([1, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec)
                gather_row_5: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_28, pl.const(3840, pl.INT64), 256), pl.Mem.Vec] = pl.tile.gather(gather_inp_row_5, gather_idx_row_5, gather_row_tmp_5)
                gather_asmbl_5: pl.Tile[[4, 64], pl.FP32, pl.MemRef(mem_vec_17, pl.const(1024, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.assemble(gather_ia_5, gather_row_5, [gather_lv_5, 0])
                gather_rv_5: pl.Tile[[4, 64], pl.FP32, pl.MemRef(mem_vec_17, pl.const(1024, pl.INT64), 1024), pl.Mem.Vec] = pl.yield_(gather_asmbl_5)
            inner_sin_il__tile: pl.Tile[[4, 64], pl.FP32, pl.MemRef(mem_vec_17, pl.const(1024, pl.INT64), 1024), pl.Mem.Vec] = gather_rv_5
            t__tile_10: pl.Tile[[4, 64], pl.FP32, pl.MemRef(mem_vec_17, pl.const(1024, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.mul(inner_sin_il__tile, il_sign__tile)
            inner_sin_signed__tile: pl.Tensor[[T_DYN, 64], pl.FP32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 0)] = pl.tile.store(t__tile_10, [rope_t0__idx_v0, 0], inner_sin_signed__iter_v1)
            cmp_cos_il__rv_v2, cmp_sin_signed__rv_v2, idx_cos_il__rv_v2, idx_sin_signed__rv_v2, inner_cos_il__rv_v2, inner_sin_signed__rv_v2 = pl.yield_(
                cmp_cos_il__tile, cmp_sin_signed__tile, idx_cos_il__tile, idx_sin_signed__tile, inner_cos_il__tile, inner_sin_signed__tile
            )
        return cmp_cos_il__ssa_v0, cmp_sin_signed__ssa_v0, idx_cos_il__ssa_v0, idx_sin_signed__ssa_v0, inner_cos_il__ssa_v0, inner_sin_signed__ssa_v0

    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def csa_slots_build_valid_qk_plan(
        cmp_sparse_indices_inline2323__ssa_v0: pl.Out[pl.Tensor[[t_dim_inline2318__ssa_v0, 512], pl.INT32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)]],
        sparse_bias_inline2369__ssa_v0: pl.Out[pl.Tensor[[t_dim_inline2318__ssa_v0, 640], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)]],
        t_dim_inline2318__ssa_v0: pl.Scalar[pl.INDEX],
        idx_topk__ssa_v3: pl.Tensor[[T_DYN, 512], pl.INT32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)],
        position_ids_t1__ssa_v0: pl.Tensor[[T_DYN, 1], pl.INT32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)],
        window_swa_indices__ssa_v0: pl.Tensor[[T_DYN, 128], pl.INT32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 0)],
        valid_block_mask_inline2310__ssa_v0: pl.Out[pl.Tensor[[t_dim_inline2318__ssa_v0, 16], pl.INT32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 0)]],
    ) -> tuple[pl.Tensor[[t_dim_inline2318__ssa_v0, 512], pl.INT32], pl.Tensor[[t_dim_inline2318__ssa_v0, 640], pl.FP32]]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        mem_vec_6: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
        mem_vec_14: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
        mem_vec_16: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
        mem_vec_32: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 4096)
        mem_vec_33: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32)
        plan_worker_inline2313__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        for bias_t0_inline2351__idx_v0, (cmp_sparse_indices_inline2323__iter_v1, sparse_bias_inline2369__iter_v1) in pl.range(
            plan_worker_inline2313__ssa_v0 * 8, t_dim_inline2318__ssa_v0, 128, init_values=(cmp_sparse_indices_inline2323__ssa_v0, sparse_bias_inline2369__ssa_v0)
        ):
            t__tile: pl.Tile[[8, 512], pl.INT32, pl.MemRef(mem_vec_6, pl.const(36896, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.load(
                idx_topk__ssa_v3, [bias_t0_inline2351__idx_v0, 0], [8, 512], [8, 512], target_memory=pl.Mem.Vec
            )
            c_raw_inline2337__tile: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_6, pl.const(36896, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.cast(t__tile, target_type=pl.FP32, mode="round")
            t__tile_1: pl.Tile[[8, 1], pl.INT32, pl.MemRef(mem_vec_14, pl.const(0, pl.INT64), 32), pl.Mem.Vec] = pl.tile.load(
                position_ids_t1__ssa_v0, [bias_t0_inline2351__idx_v0, 0], [8, 1], [8, 1], target_memory=pl.Mem.Vec
            )
            c_pos_inline2353__rm_a0_tmp_v0: pl.Tile[[1, 8], pl.INT32, pl.MemRef(mem_vec_14, pl.const(0, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(t__tile_1, [1, 8])
            c_pos_inline2353__row_major_tmp_v1: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_14, pl.const(0, pl.INT64), 32), pl.Mem.Vec] = pl.tile.cast(
                c_pos_inline2353__rm_a0_tmp_v0, target_type=pl.FP32, mode="round"
            )
            c_pos_inline2353__tile: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_14, pl.const(0, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(c_pos_inline2353__row_major_tmp_v1, [8, 1])
            t__rm_a0_tmp_v2: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_14, pl.const(0, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(c_pos_inline2353__tile, [1, 8])
            t__row_major_tmp_v3: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_14, pl.const(0, pl.INT64), 32), pl.Mem.Vec] = pl.tile.adds(t__rm_a0_tmp_v2, 1.0)
            t__tile_2: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_14, pl.const(0, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(t__row_major_tmp_v3, [8, 1])
            c_pos_scaled_inline2344__rm_a0_tmp_v4: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_14, pl.const(0, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(t__tile_2, [1, 8])
            c_pos_scaled_inline2344__row_major_tmp_v5: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_14, pl.const(0, pl.INT64), 32), pl.Mem.Vec] = pl.tile.muls(
                c_pos_scaled_inline2344__rm_a0_tmp_v4, 0.25
            )
            c_pos_scaled_inline2344__tile: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_14, pl.const(0, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(c_pos_scaled_inline2344__row_major_tmp_v5, [8, 1])
            c_pos_i32_inline2354__rm_a0_tmp_v6: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_14, pl.const(0, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(c_pos_scaled_inline2344__tile, [1, 8])
            c_pos_i32_inline2354__row_major_tmp_v7: pl.Tile[[1, 8], pl.INT32, pl.MemRef(mem_vec_14, pl.const(0, pl.INT64), 32), pl.Mem.Vec] = pl.tile.cast(
                c_pos_i32_inline2354__rm_a0_tmp_v6, target_type=pl.INT32, mode="trunc"
            )
            c_pos_i32_inline2354__tile: pl.Tile[[8, 1], pl.INT32, pl.MemRef(mem_vec_14, pl.const(0, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(c_pos_i32_inline2354__row_major_tmp_v7, [8, 1])
            c_pos_q_inline2341__rm_a0_tmp_v8: pl.Tile[[1, 8], pl.INT32, pl.MemRef(mem_vec_14, pl.const(0, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(c_pos_i32_inline2354__tile, [1, 8])
            c_pos_q_inline2341__row_major_tmp_v9: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_16, pl.const(16384, pl.INT64), 32), pl.Mem.Vec] = pl.tile.cast(
                c_pos_q_inline2341__rm_a0_tmp_v8, target_type=pl.FP32, mode="round"
            )
            c_pos_q_inline2341__tile: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_16, pl.const(16384, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(c_pos_q_inline2341__row_major_tmp_v9, [8, 1])
            t__tile_3: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_14, pl.const(0, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.full([8, 512], dtype=pl.FP32, value=1.0)
            c_upper_b_inline2307__tile: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_14, pl.const(0, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.row_expand_mul(t__tile_3, c_pos_q_inline2341__tile)
            t__tile_4: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_16, pl.const(16384, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.adds(c_raw_inline2337__tile, 1.0)
            t__tile_5: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_16, pl.const(16384, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.maximums(t__tile_4, 0.0)
            c_ge_inline2416__tile: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_16, pl.const(16384, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.minimums(t__tile_5, 1.0)
            t__tile_6: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_14, pl.const(0, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.sub(c_upper_b_inline2307__tile, c_raw_inline2337__tile)
            t__tile_7: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_14, pl.const(0, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.maximums(t__tile_6, 0.0)
            c_lt_inline2299__tile: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_14, pl.const(0, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.minimums(t__tile_7, 1.0)
            c_mask_inline2304__tile: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_14, pl.const(0, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.mul(c_ge_inline2416__tile, c_lt_inline2299__tile)
            t__tile_8: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_6, pl.const(36896, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.adds(c_raw_inline2337__tile, 1.0)
            t__tile_9: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_6, pl.const(36896, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.mul(c_mask_inline2304__tile, t__tile_8)
            c_out_inline2359__tile: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_6, pl.const(36896, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.subs(t__tile_9, 1.0)
            t__tile_10: pl.Tile[[8, 512], pl.INT32, pl.MemRef(mem_vec_16, pl.const(16384, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.cast(c_out_inline2359__tile, target_type=pl.INT32, mode="round")
            cmp_sparse_indices_inline2323__tile: pl.Tensor[[t_dim_inline2318__ssa_v0, 512], pl.INT32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                t__tile_10, [bias_t0_inline2351__idx_v0, 0], cmp_sparse_indices_inline2323__iter_v1
            )
            t__tile_11: pl.Tile[[8, 128], pl.INT32, pl.MemRef(mem_vec_16, pl.const(16384, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.load(
                window_swa_indices__ssa_v0, [bias_t0_inline2351__idx_v0, 0], [8, 128], [8, 128], target_memory=pl.Mem.Vec
            )
            v_win_f_inline2360__tile: pl.Tile[[8, 128], pl.FP32, pl.MemRef(mem_vec_16, pl.const(16384, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(t__tile_11, target_type=pl.FP32, mode="round")
            t__tile_12: pl.Tile[[8, 128], pl.FP32, pl.MemRef(mem_vec_16, pl.const(16384, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.adds(v_win_f_inline2360__tile, 1.0)
            t__tile_13: pl.Tile[[8, 128], pl.FP32, pl.MemRef(mem_vec_16, pl.const(16384, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.maximums(t__tile_12, 0.0)
            v_win_valid_inline2317__tile: pl.Tile[[8, 128], pl.FP32, pl.MemRef(mem_vec_16, pl.const(16384, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.minimums(t__tile_13, 1.0)
            tmp_tile: pl.Tile[[8, 128], pl.FP32, pl.MemRef(mem_vec_32, pl.const(32768, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.create([8, 128], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            raw_block_valid_inline2364__tile: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_33, pl.const(36864, pl.INT64), 32), pl.Mem.Vec] = pl.tile.row_max(v_win_valid_inline2317__tile, tmp_tile)
            for c_t0_inline2328__idx_v0 in pl.range(8):
                t__tile_14: pl.Scalar[pl.FP32] = pl.tile.read(raw_block_valid_inline2364__tile, [c_t0_inline2328__idx_v0, 0])
                c_valid_inline2319__ssa_v0: pl.Scalar[pl.INT32] = pl.cast(t__tile_14, pl.INT32)
                pl.tensor.write(valid_block_mask_inline2310__ssa_v0, [bias_t0_inline2351__idx_v0 + c_t0_inline2328__idx_v0, 0], c_valid_inline2319__ssa_v0)
            for c_sb_inline2320__idx_v0 in pl.range(1, 5):
                c_s0_inline2330__ssa_v0: pl.Scalar[pl.INDEX] = (c_sb_inline2320__idx_v0 - 1) * 128
                t__tile_15: pl.Tile[[8, 128], pl.FP32, pl.MemRef(mem_vec_14, pl.const(0, pl.INT64), 14848), pl.Mem.Vec] = pl.tile.slice(c_mask_inline2304__tile, [8, 128], [0, c_s0_inline2330__ssa_v0])
                tmp_tile_1: pl.Tile[[8, 128], pl.FP32, pl.MemRef(mem_vec_32, pl.const(32768, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.create([8, 128], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                c_blk_valid_inline2381__tile: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_33, pl.const(36864, pl.INT64), 32), pl.Mem.Vec] = pl.tile.row_max(t__tile_15, tmp_tile_1)
                for c_dt_inline2403__idx_v0 in pl.range(8):
                    t__tile_16: pl.Scalar[pl.FP32] = pl.tile.read(c_blk_valid_inline2381__tile, [c_dt_inline2403__idx_v0, 0])
                    c_valid_inline2319__ssa_v1: pl.Scalar[pl.INT32] = pl.cast(t__tile_16, pl.INT32)
                    pl.tensor.write(valid_block_mask_inline2310__ssa_v0, [bias_t0_inline2351__idx_v0 + c_dt_inline2403__idx_v0, c_sb_inline2320__idx_v0], c_valid_inline2319__ssa_v1)
            t__tile_17: pl.Tile[[8, 128], pl.FP32, pl.MemRef(mem_vec_14, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.subs(v_win_valid_inline2317__tile, 1.0)
            t__tile_18: pl.Tile[[8, 128], pl.FP32, pl.MemRef(mem_vec_14, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.muls(t__tile_17, 1e20)
            sparse_bias_inline2369__tile: pl.Tensor[[t_dim_inline2318__ssa_v0, 640], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                t__tile_18, [bias_t0_inline2351__idx_v0, 0], sparse_bias_inline2369__iter_v1
            )
            t__tile_19: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_6, pl.const(36896, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.minimums(c_out_inline2359__tile, 0.0)
            t__tile_20: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_6, pl.const(36896, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.muls(t__tile_19, 1e20)
            sparse_bias_inline2369__tile_1: pl.Tensor[[t_dim_inline2318__ssa_v0, 640], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                t__tile_20, [bias_t0_inline2351__idx_v0, 128], sparse_bias_inline2369__tile
            )
            cmp_sparse_indices_inline2323__rv_v2, sparse_bias_inline2369__rv_v2 = pl.yield_(cmp_sparse_indices_inline2323__tile, sparse_bias_inline2369__tile_1)
        return cmp_sparse_indices_inline2323__ssa_v0, sparse_bias_inline2369__ssa_v0

    @pl.function(type=pl.FunctionType.Spmd)
    def csa_slots_build_valid_qk_plan_spmd(
        self,
        cmp_sparse_indices_inline2323__ssa_v0: pl.Out[pl.Tensor[[t_dim_inline2318__ssa_v0, 512], pl.INT32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)]],
        sparse_bias_inline2369__ssa_v0: pl.Out[pl.Tensor[[t_dim_inline2318__ssa_v0, 640], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)]],
        t_dim_inline2318__ssa_v0: pl.Scalar[pl.INDEX],
        idx_topk__ssa_v3: pl.Tensor[[T_DYN, 512], pl.INT32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)],
        position_ids_t1__ssa_v0: pl.Tensor[[T_DYN, 1], pl.INT32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)],
        window_swa_indices__ssa_v0: pl.Tensor[[T_DYN, 128], pl.INT32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 0)],
        valid_block_mask_inline2310__ssa_v0: pl.Out[pl.Tensor[[t_dim_inline2318__ssa_v0, 16], pl.INT32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 0)]],
    ) -> tuple[pl.Tensor[[t_dim_inline2318__ssa_v0, 512], pl.INT32], pl.Tensor[[t_dim_inline2318__ssa_v0, 640], pl.FP32]]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        ret__tmp_v0: pl.Tuple[pl.Tensor[[t_dim_inline2318__ssa_v0, 512], pl.INT32], pl.Tensor[[t_dim_inline2318__ssa_v0, 640], pl.FP32]] = self.csa_slots_build_valid_qk_plan(
            cmp_sparse_indices_inline2323__ssa_v0,
            sparse_bias_inline2369__ssa_v0,
            t_dim_inline2318__ssa_v0,
            idx_topk__ssa_v3,
            position_ids_t1__ssa_v0,
            window_swa_indices__ssa_v0,
            valid_block_mask_inline2310__ssa_v0,
            attrs={"arg_directions": [pl.adir.output_existing, pl.adir.output_existing, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.output_existing]},
        )
        cmp_sparse_indices_inline2323__rv_v2: pl.Tensor[[t_dim_inline2318__ssa_v0, 512], pl.INT32, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 0)] = ret__tmp_v0[0]
        sparse_bias_inline2369__rv_v2: pl.Tensor[[t_dim_inline2318__ssa_v0, 640], pl.FP32, pl.MemRef("mem_ddr_7", pl.const(0, pl.INT64), 0)] = ret__tmp_v0[1]
        return cmp_sparse_indices_inline2323__ssa_v0, sparse_bias_inline2369__ssa_v0

    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def idx_kv_scale_commit(
        compact_rows_inline2073__ssa_v0: pl.Scalar[pl.INDEX],
        position_ids__ssa_v0: pl.Tensor[[T_DYN], pl.INT32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)],
        idx_slot_mapping__ssa_v0: pl.Tensor[[T_DYN], pl.INT64, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)],
        idx_kv_cache__rv_v2: pl.InOut[pl.Tensor[[IDX_CACHE_BLOCK_NUM_DYN, INDEXER_PAGE_BYTES_DYN], pl.INT8, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)]],
        idx_kv_scale_values_inline2059__ssa_v1: pl.Tensor[[384, 1], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 1536)],
    ) -> pl.Tensor[[IDX_CACHE_BLOCK_NUM_DYN, INDEXER_PAGE_BYTES_DYN], pl.INT8]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        mem_vec_4: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 64)
        for compact_token_inline2053__idx_v0 in pl.range(compact_rows_inline2073__ssa_v0):
            request_inline2085__ssa_v1: pl.Scalar[pl.INDEX] = compact_token_inline2053__idx_v0 // 2
            first_pos_inline2065__tile: pl.Scalar[pl.INT32] = pl.tensor.read(position_ids__ssa_v0, [request_inline2085__ssa_v1 * 6])
            local_token_inline2057__ssa_v1: pl.Scalar[pl.INDEX] = compact_token_inline2053__idx_v0 % 2 * 4 - pl.cast(first_pos_inline2065__tile, pl.INDEX) % 4 + 3
            if local_token_inline2057__ssa_v1 < 6:
                token_v1_inline2052__ssa_v0: pl.Scalar[pl.INDEX] = request_inline2085__ssa_v1 * 6 + local_token_inline2057__ssa_v1
                cache_row_i64_v1_inline2051__tile: pl.Scalar[pl.INT64] = pl.tensor.read(idx_slot_mapping__ssa_v0, [token_v1_inline2052__ssa_v0])
                if 0 <= cache_row_i64_v1_inline2051__tile:
                    cache_row_v1_inline2050__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(cache_row_i64_v1_inline2051__tile, pl.INDEX)
                    scale_page_inline2049__ssa_v0: pl.Scalar[pl.INDEX] = cache_row_v1_inline2050__ssa_v0 // 32
                    scale_bytes_inline2048__ssa_v0: pl.Tile[[1, 64], pl.INT8, pl.MemRef(mem_vec_4, pl.const(0, pl.INT64), 64), pl.Mem.Vec] = pl.tile.load(
                        idx_kv_cache__rv_v2, [scale_page_inline2049__ssa_v0, 4096], [1, 64], [1, 64], target_memory=pl.Mem.Vec
                    )
                    scale_half_inline2080__ssa_v0: pl.Tile[[1, 32], pl.FP16, pl.MemRef(mem_vec_4, pl.const(0, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reinterpret_view(
                        scale_bytes_inline2048__ssa_v0, dtype=pl.FP16
                    )
                    t__tile: pl.Scalar[pl.FP32] = pl.tensor.read(idx_kv_scale_values_inline2059__ssa_v1, [compact_token_inline2053__idx_v0, 0])
                    pl.tile.write(scale_half_inline2080__ssa_v0, [0, cache_row_v1_inline2050__ssa_v0 % 32], pl.cast(t__tile, pl.FP16))
                    updated_bytes_inline2047__ssa_v0: pl.Tile[[1, 64], pl.INT8, pl.MemRef(mem_vec_4, pl.const(0, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reinterpret_view(
                        scale_half_inline2080__ssa_v0, dtype=pl.INT8
                    )
                    idx_kv_cache__store: pl.Tensor[[IDX_CACHE_BLOCK_NUM_DYN, INDEXER_PAGE_BYTES_DYN], pl.INT8, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                        updated_bytes_inline2047__ssa_v0, [scale_page_inline2049__ssa_v0, 4096], idx_kv_cache__rv_v2
                    )
        return idx_kv_cache__rv_v2

    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def idx_qr_dequant_rope(
        qr_bf16_2d_inline928_inline2152__ssa_v0: pl.Out[pl.Tensor[[384, 8192], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 6291456)]],
        dq_rope_units_inline900_inline2142__ssa_v0: pl.Scalar[pl.INDEX],
        dq_rope_workers_inline924_inline2119__ssa_v0: pl.Scalar[pl.INDEX],
        qr_scale__ssa_v0: pl.Tensor[[T_DYN, 1], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)],
        idx_cos_il__rv_v2: pl.Tensor[[T_DYN, 64], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)],
        idx_sin_signed__rv_v2: pl.Tensor[[T_DYN, 64], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)],
        idx_wq_b_scale__ssa_v0: pl.Tensor[[8192], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 32768)],
        qr_acc_pad_inline911_inline2125__rv_v2: pl.Tensor[[384, 8192], pl.INT32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 12582912)],
    ):
        pl.func_attr({"mx_tensor_views_blocked": True})
        mem_vec_19: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 2048)
        mem_vec_20: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32)
        mem_vec_21: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 2048)
        mem_vec_22: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 2048)
        mem_vec_24: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 4096)
        mem_vec_26: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 4096)
        mem_vec_30: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 2048)
        mem_vec_32: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 1024)
        mem_vec_34: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        mem_vec_35: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        mem_vec_44: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 2048)
        mem_vec_46: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 1024)
        mem_vec_48: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        mem_vec_49: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        dq_rope_worker_inline912_inline2115__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        sw_ones_inline908_inline2131__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_24, pl.const(6176, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.full([8, 64], dtype=pl.FP32, value=1.0)
        t__ci_tmp_v0: pl.Tile[[1, 192], pl.FP32, pl.MemRef(mem_vec_26, pl.const(10272, pl.INT64), 768), pl.Mem.Vec] = pl.tile.create([1, 192], dtype=pl.FP32, target_memory=pl.Mem.Vec)
        t__tile: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_19, pl.const(0, pl.INT64), 256), pl.Mem.Vec] = pl.tile.ci(
            pl.const(0, pl.INT32), [1, 64], tmp=t__ci_tmp_v0, dtype=pl.INT32, descending=False
        )
        sw_index_inline915_inline2113__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_26, pl.const(10272, pl.INT64), 256), pl.Mem.Vec] = pl.tile.cast(t__tile, target_type=pl.FP32, mode="round")
        sw_col_inline926_inline2112__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_24, pl.const(6176, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.col_expand_mul(
            sw_ones_inline908_inline2131__tile, sw_index_inline915_inline2113__tile
        )
        t__tile_1: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_26, pl.const(10272, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.muls(sw_col_inline926_inline2112__tile, 0.5)
        t__tile_2: pl.Tile[[8, 64], pl.INT32, pl.MemRef(mem_vec_26, pl.const(10272, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(t__tile_1, target_type=pl.INT32, mode="trunc")
        sw_dup_f_inline927_inline2110__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_26, pl.const(10272, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(t__tile_2, target_type=pl.FP32, mode="round")
        t__tile_3: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_26, pl.const(10272, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.muls(sw_dup_f_inline927_inline2110__tile, 2.0)
        sw_lane_inline929_inline2130__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_26, pl.const(10272, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.sub(sw_col_inline926_inline2112__tile, t__tile_3)
        t__tile_4: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_24, pl.const(6176, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.adds(sw_col_inline926_inline2112__tile, 1.0)
        t__tile_5: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_26, pl.const(10272, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.muls(sw_lane_inline929_inline2130__tile, 2.0)
        t__tile_6: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_24, pl.const(6176, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.sub(t__tile_4, t__tile_5)
        rope_swap_idx_inline930_inline2136__tile: pl.Tile[[8, 64], pl.INT32, pl.MemRef(mem_vec_19, pl.const(0, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(
            t__tile_6, target_type=pl.INT32, mode="round"
        )
        for dq_unit_inline931_inline2140__idx_v0, (qr_bf16_2d_inline928_inline2152__iter_v1,) in pl.range(
            dq_rope_worker_inline912_inline2115__ssa_v0,
            dq_rope_units_inline900_inline2142__ssa_v0,
            dq_rope_workers_inline924_inline2119__ssa_v0,
            init_values=(qr_bf16_2d_inline928_inline2152__ssa_v0,),
        ):
            hg_inline919_inline2147__ssa_v0: pl.Scalar[pl.INDEX] = dq_unit_inline931_inline2140__idx_v0 % 16 * 4
            dq_t0_inline897_inline2151__ssa_v0: pl.Scalar[pl.INDEX] = dq_unit_inline931_inline2140__idx_v0 // 16 * 8
            qr_scale_tile_inline896_inline2154__tile: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_20, pl.const(2048, pl.INT64), 32), pl.Mem.Vec] = pl.tile.load(
                qr_scale__ssa_v0, [dq_t0_inline897_inline2151__ssa_v0, 0], [8, 1], [8, 1], target_memory=pl.Mem.Vec
            )
            cos_tile_inline895_inline2155__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_21, pl.const(2080, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.load(
                idx_cos_il__rv_v2, [dq_t0_inline897_inline2151__ssa_v0, 0], [8, 64], [8, 64], target_memory=pl.Mem.Vec
            )
            sin_tile_inline918_inline2159__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_22, pl.const(4128, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.load(
                idx_sin_signed__rv_v2, [dq_t0_inline897_inline2151__ssa_v0, 0], [8, 64], [8, 64], target_memory=pl.Mem.Vec
            )
            for h_inner_inline894_inline2157__idx_v0, (qr_bf16_2d_inline928_inline2152__iter_v3,) in pl.range(0, 4, 2, init_values=(qr_bf16_2d_inline928_inline2152__iter_v1,)):
                h0_inline916_inline2158__ssa_v0: pl.Scalar[pl.INDEX] = (hg_inline919_inline2147__ssa_v0 + h_inner_inline894_inline2157__idx_v0) * 128
                h0_inline916_inline2158__ssa_v0_1: pl.Scalar[pl.INDEX] = (hg_inline919_inline2147__ssa_v0 + (h_inner_inline894_inline2157__idx_v0 + 1)) * 128
                t__tile_7: pl.Tile[[128], pl.FP32, pl.MemRef(mem_vec_30, pl.const(14368, pl.INT64), 512), pl.Mem.Vec] = pl.tile.load(
                    idx_wq_b_scale__ssa_v0, [h0_inline916_inline2158__ssa_v0], [128], [128], target_memory=pl.Mem.Vec
                )
                t__tile_8: pl.Tile[[8, 128], pl.INT32, pl.MemRef(mem_vec_24, pl.const(6176, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.load(
                    qr_acc_pad_inline911_inline2125__rv_v2, [dq_t0_inline897_inline2151__ssa_v0, h0_inline916_inline2158__ssa_v0], [8, 128], [8, 128], target_memory=pl.Mem.Vec
                )
                t__tile_9: pl.Tile[[128], pl.FP32, pl.MemRef(mem_vec_44, pl.const(17952, pl.INT64), 512), pl.Mem.Vec] = pl.tile.load(
                    idx_wq_b_scale__ssa_v0, [h0_inline916_inline2158__ssa_v0_1], [128], [128], target_memory=pl.Mem.Vec
                )
                t__tile_10: pl.Tile[[8, 128], pl.INT32, pl.MemRef(mem_vec_26, pl.const(10272, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.load(
                    qr_acc_pad_inline911_inline2125__rv_v2, [dq_t0_inline897_inline2151__ssa_v0, h0_inline916_inline2158__ssa_v0_1], [8, 128], [8, 128], target_memory=pl.Mem.Vec
                )
                wq_scale_inline893_inline2145__tile: pl.Tile[[1, 128], pl.FP32, pl.MemRef(mem_vec_30, pl.const(14368, pl.INT64), 512), pl.Mem.Vec] = t__tile_7
                acc_fp32_inline892_inline2163__tile: pl.Tile[[8, 128], pl.FP32, pl.MemRef(mem_vec_24, pl.const(6176, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(
                    t__tile_8, target_type=pl.FP32, mode="none"
                )
                t__tile_11: pl.Tile[[8, 128], pl.FP32, pl.MemRef(mem_vec_24, pl.const(6176, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.row_expand_mul(
                    acc_fp32_inline892_inline2163__tile, qr_scale_tile_inline896_inline2154__tile
                )
                qr_dequant_inline914_inline2164__tile: pl.Tile[[8, 128], pl.FP32, pl.MemRef(mem_vec_24, pl.const(6176, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.col_expand_mul(
                    t__tile_11, wq_scale_inline893_inline2145__tile
                )
                t__tile_12: pl.Tile[[8, 128], pl.BF16, pl.MemRef(mem_vec_30, pl.const(14368, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(
                    qr_dequant_inline914_inline2164__tile, target_type=pl.BF16, mode="rint"
                )
                qr_dequant_v1_inline920_inline2167__tile: pl.Tile[[8, 128], pl.FP32, pl.MemRef(mem_vec_24, pl.const(6176, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(
                    t__tile_12, target_type=pl.FP32, mode="round"
                )
                t__tile_13: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_24, pl.const(6176, pl.INT64), 3840), pl.Mem.Vec] = pl.tile.slice(qr_dequant_v1_inline920_inline2167__tile, [8, 64], [0, 0])
                qr_nope_bf16_inline906_inline2168__tile: pl.Tile[[8, 64], pl.BF16, pl.MemRef(mem_vec_32, pl.const(16416, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.cast(
                    t__tile_13, target_type=pl.BF16, mode="rint"
                )
                qr_rope_slice_inline891_inline2160__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_24, pl.const(6432, pl.INT64), 3840), pl.Mem.Vec] = pl.tile.slice(
                    qr_dequant_v1_inline920_inline2167__tile, [8, 64], [0, 64]
                )
                gather_acc_init: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_30, pl.const(14368, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.create([8, 64], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                for gather_lv, (gather_ia,) in pl.range(8, init_values=(gather_acc_init,)):
                    gather_inp_row: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_24, pl.const(6176, pl.INT64), 256), pl.Mem.Vec] = pl.tile.slice(
                        qr_rope_slice_inline891_inline2160__tile, [1, 64], [gather_lv, 0], [1, 64]
                    )
                    gather_idx_row: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_19, pl.const(0, pl.INT64), 256), pl.Mem.Vec] = pl.tile.slice(
                        rope_swap_idx_inline930_inline2136__tile, [1, 64], [gather_lv, 0], [1, 64]
                    )
                    gather_row_tmp: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_34, pl.const(17440, pl.INT64), 256), pl.Mem.Vec] = pl.tile.create([1, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec)
                    gather_row: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_35, pl.const(17696, pl.INT64), 256), pl.Mem.Vec] = pl.tile.gather(gather_inp_row, gather_idx_row, gather_row_tmp)
                    gather_asmbl: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_30, pl.const(14368, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.assemble(gather_ia, gather_row, [gather_lv, 0])
                    gather_rv: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_30, pl.const(14368, pl.INT64), 2048), pl.Mem.Vec] = pl.yield_(gather_asmbl)
                qr_swapped_inline890_inline2169__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_30, pl.const(14368, pl.INT64), 2048), pl.Mem.Vec] = gather_rv
                t__tile_14: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_24, pl.const(6176, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.mul(
                    qr_rope_slice_inline891_inline2160__tile, cos_tile_inline895_inline2155__tile
                )
                t__tile_15: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_30, pl.const(14368, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.mul(
                    qr_swapped_inline890_inline2169__tile, sin_tile_inline918_inline2159__tile
                )
                rope_rot_inline889_inline2133__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_24, pl.const(6176, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.add(t__tile_14, t__tile_15)
                rope_bf16_inline888_inline2170__tile: pl.Tile[[8, 64], pl.BF16, pl.MemRef(mem_vec_24, pl.const(6176, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.cast(
                    rope_rot_inline889_inline2133__tile, target_type=pl.BF16, mode="rint"
                )
                qr_bf16_2d_inline928_inline2152__tile: pl.Tensor[[384, 8192], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 6291456)] = pl.tile.store(
                    qr_nope_bf16_inline906_inline2168__tile, [dq_t0_inline897_inline2151__ssa_v0, h0_inline916_inline2158__ssa_v0], qr_bf16_2d_inline928_inline2152__iter_v3
                )
                qr_bf16_2d_inline928_inline2152__tile_1: pl.Tensor[[384, 8192], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 6291456)] = pl.tile.store(
                    rope_bf16_inline888_inline2170__tile, [dq_t0_inline897_inline2151__ssa_v0, h0_inline916_inline2158__ssa_v0 + 64], qr_bf16_2d_inline928_inline2152__tile
                )
                wq_scale_inline893_inline2145__tile_1: pl.Tile[[1, 128], pl.FP32, pl.MemRef(mem_vec_44, pl.const(17952, pl.INT64), 512), pl.Mem.Vec] = t__tile_9
                acc_fp32_inline892_inline2163__tile_1: pl.Tile[[8, 128], pl.FP32, pl.MemRef(mem_vec_26, pl.const(10272, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(
                    t__tile_10, target_type=pl.FP32, mode="none"
                )
                t__tile_16: pl.Tile[[8, 128], pl.FP32, pl.MemRef(mem_vec_26, pl.const(10272, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.row_expand_mul(
                    acc_fp32_inline892_inline2163__tile_1, qr_scale_tile_inline896_inline2154__tile
                )
                qr_dequant_inline914_inline2164__tile_1: pl.Tile[[8, 128], pl.FP32, pl.MemRef(mem_vec_26, pl.const(10272, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.col_expand_mul(
                    t__tile_16, wq_scale_inline893_inline2145__tile_1
                )
                t__tile_17: pl.Tile[[8, 128], pl.BF16, pl.MemRef(mem_vec_44, pl.const(17952, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(
                    qr_dequant_inline914_inline2164__tile_1, target_type=pl.BF16, mode="rint"
                )
                qr_dequant_v1_inline920_inline2167__tile_1: pl.Tile[[8, 128], pl.FP32, pl.MemRef(mem_vec_26, pl.const(10272, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(
                    t__tile_17, target_type=pl.FP32, mode="round"
                )
                t__tile_18: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_26, pl.const(10272, pl.INT64), 3840), pl.Mem.Vec] = pl.tile.slice(qr_dequant_v1_inline920_inline2167__tile_1, [8, 64], [0, 0])
                qr_nope_bf16_inline906_inline2168__tile_1: pl.Tile[[8, 64], pl.BF16, pl.MemRef(mem_vec_46, pl.const(20000, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.cast(
                    t__tile_18, target_type=pl.BF16, mode="rint"
                )
                qr_rope_slice_inline891_inline2160__tile_1: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_26, pl.const(10528, pl.INT64), 3840), pl.Mem.Vec] = pl.tile.slice(
                    qr_dequant_v1_inline920_inline2167__tile_1, [8, 64], [0, 64]
                )
                gather_acc_init_1: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_44, pl.const(17952, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.create([8, 64], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                for gather_lv_1, (gather_ia_1,) in pl.range(8, init_values=(gather_acc_init_1,)):
                    gather_inp_row_1: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_26, pl.const(10272, pl.INT64), 256), pl.Mem.Vec] = pl.tile.slice(
                        qr_rope_slice_inline891_inline2160__tile_1, [1, 64], [gather_lv_1, 0], [1, 64]
                    )
                    gather_idx_row_1: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_19, pl.const(0, pl.INT64), 256), pl.Mem.Vec] = pl.tile.slice(
                        rope_swap_idx_inline930_inline2136__tile, [1, 64], [gather_lv_1, 0], [1, 64]
                    )
                    gather_row_tmp_1: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_48, pl.const(21024, pl.INT64), 256), pl.Mem.Vec] = pl.tile.create([1, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec)
                    gather_row_1: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_49, pl.const(21280, pl.INT64), 256), pl.Mem.Vec] = pl.tile.gather(gather_inp_row_1, gather_idx_row_1, gather_row_tmp_1)
                    gather_asmbl_1: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_44, pl.const(17952, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.assemble(gather_ia_1, gather_row_1, [gather_lv_1, 0])
                    gather_rv_1: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_44, pl.const(17952, pl.INT64), 2048), pl.Mem.Vec] = pl.yield_(gather_asmbl_1)
                qr_swapped_inline890_inline2169__tile_1: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_44, pl.const(17952, pl.INT64), 2048), pl.Mem.Vec] = gather_rv_1
                t__tile_19: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_26, pl.const(10272, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.mul(
                    qr_rope_slice_inline891_inline2160__tile_1, cos_tile_inline895_inline2155__tile
                )
                t__tile_20: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_44, pl.const(17952, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.mul(
                    qr_swapped_inline890_inline2169__tile_1, sin_tile_inline918_inline2159__tile
                )
                rope_rot_inline889_inline2133__tile_1: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_26, pl.const(10272, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.add(t__tile_19, t__tile_20)
                rope_bf16_inline888_inline2170__tile_1: pl.Tile[[8, 64], pl.BF16, pl.MemRef(mem_vec_26, pl.const(10272, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.cast(
                    rope_rot_inline889_inline2133__tile_1, target_type=pl.BF16, mode="rint"
                )
                qr_bf16_2d_inline928_inline2152__tile_2: pl.Tensor[[384, 8192], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 6291456)] = pl.tile.store(
                    qr_nope_bf16_inline906_inline2168__tile_1, [dq_t0_inline897_inline2151__ssa_v0, h0_inline916_inline2158__ssa_v0_1], qr_bf16_2d_inline928_inline2152__tile_1
                )
                qr_bf16_2d_inline928_inline2152__tile_3: pl.Tensor[[384, 8192], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 6291456)] = pl.tile.store(
                    rope_bf16_inline888_inline2170__tile_1, [dq_t0_inline897_inline2151__ssa_v0, h0_inline916_inline2158__ssa_v0_1 + 64], qr_bf16_2d_inline928_inline2152__tile_2
                )
                qr_bf16_2d_inline928_inline2152__rv_v4: pl.Tensor[[384, 8192], pl.BF16, pl.MemRef("mem_ddr_55", pl.const(0, pl.INT64), 6291456)] = pl.yield_(qr_bf16_2d_inline928_inline2152__tile_3)
            qr_bf16_2d_inline928_inline2152__rv_v2: pl.Tensor[[384, 8192], pl.BF16, pl.MemRef("mem_ddr_56", pl.const(0, pl.INT64), 6291456)] = pl.yield_(qr_bf16_2d_inline928_inline2152__rv_v4)
        return

    @pl.function(type=pl.FunctionType.Spmd)
    def idx_qr_dequant_rope_spmd(
        self,
        qr_bf16_2d_inline928_inline2152__ssa_v0: pl.Out[pl.Tensor[[384, 8192], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 6291456)]],
        dq_rope_units_inline900_inline2142__ssa_v0: pl.Scalar[pl.INDEX],
        dq_rope_workers_inline924_inline2119__ssa_v0: pl.Scalar[pl.INDEX],
        qr_scale__ssa_v0: pl.Tensor[[T_DYN, 1], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)],
        idx_cos_il__rv_v2: pl.Tensor[[T_DYN, 64], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)],
        idx_sin_signed__rv_v2: pl.Tensor[[T_DYN, 64], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)],
        idx_wq_b_scale__ssa_v0: pl.Tensor[[8192], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 32768)],
        qr_acc_pad_inline911_inline2125__rv_v2: pl.Tensor[[384, 8192], pl.INT32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 12582912)],
    ):
        pl.func_attr({"mx_tensor_views_blocked": True})
        self.idx_qr_dequant_rope(
            qr_bf16_2d_inline928_inline2152__ssa_v0,
            dq_rope_units_inline900_inline2142__ssa_v0,
            dq_rope_workers_inline924_inline2119__ssa_v0,
            qr_scale__ssa_v0,
            idx_cos_il__rv_v2,
            idx_sin_signed__rv_v2,
            idx_wq_b_scale__ssa_v0,
            qr_acc_pad_inline911_inline2125__rv_v2,
            attrs={"arg_directions": [pl.adir.output_existing, pl.adir.scalar, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input]},
        )

    @pl.function(type=pl.FunctionType.AIC, level=pl.Level.AIC, role=pl.Role.SubWorker)
    def idx_qr_proj_matmul(
        qr_acc_pad_inline911_inline2125__ssa_v0: pl.Out[pl.Tensor[[384, 8192], pl.INT32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 12582912)]],
        row_blocks_inline909_inline2135__ssa_v0: pl.Scalar[pl.INDEX],
        bs_inline913_inline2122__ssa_v0: pl.Scalar[pl.INDEX],
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
        qr_proj_worker_inline901_inline2126__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        for qr_unit_inline925_inline2137__idx_v0, (qr_acc_pad_inline911_inline2125__iter_v1,) in pl.range(
            qr_proj_worker_inline901_inline2126__ssa_v0, row_blocks_inline909_inline2135__ssa_v0 * 8, 24, init_values=(qr_acc_pad_inline911_inline2125__ssa_v0,)
        ):
            qr_rb_inline904_inline2144__ssa_v0: pl.Scalar[pl.INDEX] = qr_unit_inline925_inline2137__idx_v0 // 8
            ot_inline910_inline2141__ssa_v0: pl.Scalar[pl.INDEX] = qr_unit_inline925_inline2137__idx_v0 - qr_rb_inline904_inline2144__ssa_v0 * 8
            qr_r0_inline898_inline2132__ssa_v0: pl.Scalar[pl.INDEX] = qr_rb_inline904_inline2144__ssa_v0 * 16
            qr_rows_inline907_inline2139__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(bs_inline913_inline2122__ssa_v0 - qr_r0_inline898_inline2132__ssa_v0, 16)
            o_base_inline905_inline2146__ssa_v0: pl.Scalar[pl.INDEX] = ot_inline910_inline2141__ssa_v0 * 1024
            for ns_inline899_inline2166__idx_v0, (qr_acc_pad_inline911_inline2125__iter_v3,) in pl.range(0, 1024, 512, init_values=(qr_acc_pad_inline911_inline2125__iter_v1,)):
                qr_acc_inline922_inline2117__tile: pl.Tile[[16, 512], pl.INT32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 65536), pl.Mem.Acc] = pl.tile.create(
                    [16, 512], dtype=pl.INT32, target_memory=pl.Mem.Acc
                )
                for kb_inline917_inline2128__idx_v0, (qr_acc_inline922_inline2117__iter_v1,) in pl.range(0, 4, 2, init_values=(qr_acc_inline922_inline2117__tile,)):
                    q0_inline921_inline2124__ssa_v0: pl.Scalar[pl.INDEX] = kb_inline917_inline2128__idx_v0 * 256
                    q0_inline921_inline2124__ssa_v0_1: pl.Scalar[pl.INDEX] = kb_inline917_inline2128__idx_v0 * 256 + 256
                    qr_tile_inline923_inline2121__tile: pl.Tile[
                        [16, 256], pl.INT8, pl.MemRef(mem_mat_4, pl.const(0, pl.INT64), 4096), pl.Mem.Mat, pl.TileView(valid_shape=[qr_rows_inline907_inline2139__ssa_v0, 256])
                    ] = pl.tile.load(
                        qr__ssa_v0, [qr_r0_inline898_inline2132__ssa_v0, q0_inline921_inline2124__ssa_v0], [16, 256], [qr_rows_inline907_inline2139__ssa_v0, 256], target_memory=pl.Mem.Mat
                    )
                    wq_tile_inline902_inline2127__tile: pl.Tile[[256, 512], pl.INT8, pl.MemRef(mem_mat_5, pl.const(4096, pl.INT64), 131072), pl.Mem.Mat] = pl.tile.load(
                        idx_wq_b__ssa_v0, [q0_inline921_inline2124__ssa_v0, o_base_inline905_inline2146__ssa_v0 + ns_inline899_inline2166__idx_v0], [256, 512], [256, 512], target_memory=pl.Mem.Mat
                    )
                    qr_tile_inline923_inline2121__tile_1: pl.Tile[
                        [16, 256], pl.INT8, pl.MemRef(mem_mat_6, pl.const(135168, pl.INT64), 4096), pl.Mem.Mat, pl.TileView(valid_shape=[qr_rows_inline907_inline2139__ssa_v0, 256])
                    ] = pl.tile.load(
                        qr__ssa_v0, [qr_r0_inline898_inline2132__ssa_v0, q0_inline921_inline2124__ssa_v0_1], [16, 256], [qr_rows_inline907_inline2139__ssa_v0, 256], target_memory=pl.Mem.Mat
                    )
                    wq_tile_inline902_inline2127__tile_1: pl.Tile[[256, 512], pl.INT8, pl.MemRef(mem_mat_7, pl.const(139264, pl.INT64), 131072), pl.Mem.Mat] = pl.tile.load(
                        idx_wq_b__ssa_v0, [q0_inline921_inline2124__ssa_v0_1, o_base_inline905_inline2146__ssa_v0 + ns_inline899_inline2166__idx_v0], [256, 512], [256, 512], target_memory=pl.Mem.Mat
                    )
                    for qr_acc_inline922_inline2117__tile_l0_ko, (qr_acc_inline922_inline2117__tile_l0_c,) in pl.range(0, 256, 128, init_values=(qr_acc_inline922_inline2117__iter_v1,)):
                        qr_acc_inline922_inline2117__tile_l0_a: pl.Tile[
                            [16, 64],
                            pl.INT8,
                            pl.MemRef(mem_left_8, pl.const(3072, pl.INT64), 1024),
                            pl.Mem.Left,
                            pl.TileView(valid_shape=[qr_rows_inline907_inline2139__ssa_v0, 64], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal),
                        ] = pl.tile.extract(qr_tile_inline923_inline2121__tile, 0, qr_acc_inline922_inline2117__tile_l0_ko, [16, 64], target_memory=pl.Mem.Left)
                        qr_acc_inline922_inline2117__tile_l0_b: pl.Tile[[64, 512], pl.INT8, pl.MemRef(mem_right_9, pl.const(32768, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                            wq_tile_inline902_inline2127__tile, qr_acc_inline922_inline2117__tile_l0_ko, 0, [64, 512], target_memory=pl.Mem.Right
                        )
                        qr_acc_inline922_inline2117__tile_l0_a_1: pl.Tile[
                            [16, 64],
                            pl.INT8,
                            pl.MemRef(mem_left_10, pl.const(0, pl.INT64), 1024),
                            pl.Mem.Left,
                            pl.TileView(valid_shape=[qr_rows_inline907_inline2139__ssa_v0, 64], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal),
                        ] = pl.tile.extract(qr_tile_inline923_inline2121__tile, 0, qr_acc_inline922_inline2117__tile_l0_ko + 64, [16, 64], target_memory=pl.Mem.Left)
                        qr_acc_inline922_inline2117__tile_l0_b_1: pl.Tile[[64, 512], pl.INT8, pl.MemRef(mem_right_11, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                            wq_tile_inline902_inline2127__tile, qr_acc_inline922_inline2117__tile_l0_ko + 64, 0, [64, 512], target_memory=pl.Mem.Right
                        )
                        qr_acc_inline922_inline2117__tile_l0_c_acc: pl.Tile[[16, 512], pl.INT32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 65536), pl.Mem.Acc] = pl.tile.matmul_acc(
                            qr_acc_inline922_inline2117__tile_l0_c,
                            qr_acc_inline922_inline2117__tile_l0_a,
                            qr_acc_inline922_inline2117__tile_l0_b,
                            q0_inline921_inline2124__ssa_v0 == 0 and qr_acc_inline922_inline2117__tile_l0_ko == 0,
                        )
                        qr_acc_inline922_inline2117__tile_l0_c_acc_1: pl.Tile[[16, 512], pl.INT32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 65536), pl.Mem.Acc] = pl.tile.matmul_acc(
                            qr_acc_inline922_inline2117__tile_l0_c_acc,
                            qr_acc_inline922_inline2117__tile_l0_a_1,
                            qr_acc_inline922_inline2117__tile_l0_b_1,
                            q0_inline921_inline2124__ssa_v0 == 0 and qr_acc_inline922_inline2117__tile_l0_ko == -64,
                        )
                        qr_acc_inline922_inline2117__tile_1: pl.Tile[[16, 512], pl.INT32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 65536), pl.Mem.Acc] = pl.yield_(
                            qr_acc_inline922_inline2117__tile_l0_c_acc_1
                        )
                    for qr_acc_inline922_inline2117__tile_l0_ko_1, (qr_acc_inline922_inline2117__tile_l0_c_1,) in pl.range(0, 256, 128, init_values=(qr_acc_inline922_inline2117__tile_1,)):
                        qr_acc_inline922_inline2117__tile_l0_a_2: pl.Tile[
                            [16, 64],
                            pl.INT8,
                            pl.MemRef(mem_left_13, pl.const(1024, pl.INT64), 1024),
                            pl.Mem.Left,
                            pl.TileView(valid_shape=[qr_rows_inline907_inline2139__ssa_v0, 64], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal),
                        ] = pl.tile.extract(qr_tile_inline923_inline2121__tile_1, 0, qr_acc_inline922_inline2117__tile_l0_ko_1, [16, 64], target_memory=pl.Mem.Left)
                        qr_acc_inline922_inline2117__tile_l0_b_2: pl.Tile[[64, 512], pl.INT8, pl.MemRef(mem_right_9, pl.const(32768, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                            wq_tile_inline902_inline2127__tile_1, qr_acc_inline922_inline2117__tile_l0_ko_1, 0, [64, 512], target_memory=pl.Mem.Right
                        )
                        qr_acc_inline922_inline2117__tile_l0_a_3: pl.Tile[
                            [16, 64],
                            pl.INT8,
                            pl.MemRef(mem_left_15, pl.const(2048, pl.INT64), 1024),
                            pl.Mem.Left,
                            pl.TileView(valid_shape=[qr_rows_inline907_inline2139__ssa_v0, 64], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal),
                        ] = pl.tile.extract(qr_tile_inline923_inline2121__tile_1, 0, qr_acc_inline922_inline2117__tile_l0_ko_1 + 64, [16, 64], target_memory=pl.Mem.Left)
                        qr_acc_inline922_inline2117__tile_l0_b_3: pl.Tile[[64, 512], pl.INT8, pl.MemRef(mem_right_11, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                            wq_tile_inline902_inline2127__tile_1, qr_acc_inline922_inline2117__tile_l0_ko_1 + 64, 0, [64, 512], target_memory=pl.Mem.Right
                        )
                        qr_acc_inline922_inline2117__tile_l0_c_acc_2: pl.Tile[[16, 512], pl.INT32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 65536), pl.Mem.Acc] = pl.tile.matmul_acc(
                            qr_acc_inline922_inline2117__tile_l0_c_1,
                            qr_acc_inline922_inline2117__tile_l0_a_2,
                            qr_acc_inline922_inline2117__tile_l0_b_2,
                            q0_inline921_inline2124__ssa_v0_1 == 0 and qr_acc_inline922_inline2117__tile_l0_ko_1 == 0,
                        )
                        qr_acc_inline922_inline2117__tile_l0_c_acc_3: pl.Tile[[16, 512], pl.INT32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 65536), pl.Mem.Acc] = pl.tile.matmul_acc(
                            qr_acc_inline922_inline2117__tile_l0_c_acc_2,
                            qr_acc_inline922_inline2117__tile_l0_a_3,
                            qr_acc_inline922_inline2117__tile_l0_b_3,
                            q0_inline921_inline2124__ssa_v0_1 == 0 and qr_acc_inline922_inline2117__tile_l0_ko_1 == -64,
                        )
                        qr_acc_inline922_inline2117__tile_2: pl.Tile[[16, 512], pl.INT32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 65536), pl.Mem.Acc] = pl.yield_(
                            qr_acc_inline922_inline2117__tile_l0_c_acc_3
                        )
                    qr_acc_inline922_inline2117__rv_v2: pl.Tile[[16, 512], pl.INT32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 65536), pl.Mem.Acc] = pl.yield_(qr_acc_inline922_inline2117__tile_2)
                qr_acc_pad_inline911_inline2125__tile: pl.Tensor[[384, 8192], pl.INT32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 12582912)] = pl.tile.store(
                    qr_acc_inline922_inline2117__rv_v2,
                    [qr_r0_inline898_inline2132__ssa_v0, o_base_inline905_inline2146__ssa_v0 + ns_inline899_inline2166__idx_v0],
                    qr_acc_pad_inline911_inline2125__iter_v3,
                )
                qr_acc_pad_inline911_inline2125__rv_v4: pl.Tensor[[384, 8192], pl.INT32, pl.MemRef("mem_ddr_19", pl.const(0, pl.INT64), 12582912)] = pl.yield_(qr_acc_pad_inline911_inline2125__tile)
            qr_acc_pad_inline911_inline2125__rv_v2: pl.Tensor[[384, 8192], pl.INT32, pl.MemRef("mem_ddr_20", pl.const(0, pl.INT64), 12582912)] = pl.yield_(qr_acc_pad_inline911_inline2125__rv_v4)
        return qr_acc_pad_inline911_inline2125__ssa_v0

    @pl.function(type=pl.FunctionType.Spmd)
    def idx_qr_proj_matmul_spmd(
        self,
        qr_acc_pad_inline911_inline2125__ssa_v0: pl.Out[pl.Tensor[[384, 8192], pl.INT32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 12582912)]],
        row_blocks_inline909_inline2135__ssa_v0: pl.Scalar[pl.INDEX],
        bs_inline913_inline2122__ssa_v0: pl.Scalar[pl.INDEX],
        qr__ssa_v0: pl.Tensor[[T_DYN, 1024], pl.INT8, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)],
        idx_wq_b__ssa_v0: pl.Tensor[[1024, 8192], pl.INT8, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 8388608)],
    ) -> pl.Tensor[[384, 8192], pl.INT32]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        qr_acc_pad_inline911_inline2125__rv_v2: pl.Tensor[[384, 8192], pl.INT32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 12582912)] = self.idx_qr_proj_matmul(
            qr_acc_pad_inline911_inline2125__ssa_v0,
            row_blocks_inline909_inline2135__ssa_v0,
            bs_inline913_inline2122__ssa_v0,
            qr__ssa_v0,
            idx_wq_b__ssa_v0,
            attrs={"arg_directions": [pl.adir.output_existing, pl.adir.scalar, pl.adir.scalar, pl.adir.input, pl.adir.input]},
        )
        return qr_acc_pad_inline911_inline2125__ssa_v0

    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def indexer_boundary_init(normed_kv_inline518__ssa_v0: pl.Out[pl.Tensor[[384, 128], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 98304)]]) -> pl.Tensor[[384, 128], pl.BF16]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        mem_vec_1: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 512)
        init_request_inline406_inline2028__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        compact_begin_inline438_inline1968__ssa_v0: pl.Scalar[pl.INDEX] = init_request_inline406_inline2028__ssa_v0 * 2
        t__tile: pl.Tile[[2, 128], pl.BF16, pl.MemRef(mem_vec_1, pl.const(0, pl.INT64), 512), pl.Mem.Vec] = pl.tile.full([2, 128], dtype=pl.BF16, value=0.0)
        normed_kv_inline518__tile: pl.Tensor[[384, 128], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 98304)] = pl.tile.store(
            t__tile, [compact_begin_inline438_inline1968__ssa_v0, 0], normed_kv_inline518__ssa_v0
        )
        return normed_kv_inline518__ssa_v0

    @pl.function(type=pl.FunctionType.Spmd)
    def indexer_boundary_init_spmd(self, normed_kv_inline518__ssa_v0: pl.Out[pl.Tensor[[384, 128], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 98304)]]) -> pl.Tensor[[384, 128], pl.BF16]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        normed_kv_inline518__ssa_v1: pl.Tensor[[384, 128], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 98304)] = self.indexer_boundary_init(
            normed_kv_inline518__ssa_v0, attrs={"arg_directions": [pl.adir.output_existing]}
        )
        return normed_kv_inline518__ssa_v0

    @pl.function(type=pl.FunctionType.AIC, level=pl.Level.AIC, role=pl.Role.SubWorker)
    def indexer_score_topk_leaf_aic(
        position_ids__ssa_v0: pl.Tensor[[T_DYN], pl.INT32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)],
        kv_seq_lens__ssa_v0: pl.Tensor[[B_DYN], pl.INT32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)],
        score_arena_inline1043_inline2235__ssa_v0: pl.InOut[pl.Tensor[[384, 8192], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 12582912)]],
        qr_hadamard_i8_inline527__rv_v2: pl.Tensor[[24576, 128], pl.INT8, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 3145728)],
        qr_hadamard_scale_dq_inline525__rv_v2: pl.Tensor[[24576, 1], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 98304)],
        weights_inline2220__ssa_v1: pl.Tensor[[384, 64], pl.FP32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 98304)],
        idx_block_table_flat_inline1044_inline2217__ssa_v0: pl.Tensor[[idx_table_len_inline1025_inline2222__ssa_v0], pl.INT32, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 0)],
        table_columns_inline1029_inline2268__ssa_v0: pl.Scalar[pl.INDEX],
        idx_kv_cache__rv_v2: pl.Tensor[[IDX_CACHE_BLOCK_NUM_DYN, INDEXER_PAGE_BYTES_DYN], pl.INT8, pl.MemRef("mem_ddr_7", pl.const(0, pl.INT64), 0)],
        pair_arena_inline1035_inline2211__ssa_v0: pl.Out[pl.Tensor[[24576, 1024], pl.FP32, pl.MemRef("mem_ddr_8", pl.const(0, pl.INT64), 100663296)]],
        __gm_pipe_buffer: pl.Out[pl.Tensor[[1], pl.FP32, pl.MemRef("mem_ddr_9", pl.const(0, pl.INT64), 4)]],
    ):
        pl.func_attr({"slot_num": 1, "mx_tensor_views_blocked": True, "split_aiv": True})
        mem_mat_10: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 8192)
        mem_mat_11: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 49152)
        mem_acc_24: pl.Ptr = pl.tile.alloc(pl.Mem.Acc, 98304)
        mem_left_25: pl.Ptr = pl.tile.alloc(pl.Mem.Left, 4096)
        mem_right_26: pl.Ptr = pl.tile.alloc(pl.Mem.Right, 24576)
        mem_left_27: pl.Ptr = pl.tile.alloc(pl.Mem.Left, 4096)
        mem_right_28: pl.Ptr = pl.tile.alloc(pl.Mem.Right, 24576)
        mem_mat_30: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 49152)
        mem_left_44: pl.Ptr = pl.tile.alloc(pl.Mem.Left, 4096)
        mem_left_46: pl.Ptr = pl.tile.alloc(pl.Mem.Left, 4096)
        indexer_score_topk_leaf_c2v_slot_buffer_import: pl.Scalar[pl.INT32] = pl.system.import_peer_buffer(name="indexer_score_topk_leaf_c2v_slot_buffer", peer_func="indexer_score_topk_leaf_aiv")
        pl.system.aic_initialize_pipe(indexer_score_topk_leaf_c2v_slot_buffer_import, pl.const(0, pl.INT32), dir_mask=1, slot_size=98304, slot_num=1)
        worker_inline1037_inline2245__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        query_count_inline1038_inline2205__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(position_ids__ssa_v0, 0)
        for batch_inline1034_inline2239__idx_v0, (max_cache_len_inline1046_inline2234__iter_v1,) in pl.range(query_count_inline1038_inline2205__ssa_v0 // 6, init_values=(0,)):
            t__tile: pl.Scalar[pl.INT32] = pl.tensor.read(kv_seq_lens__ssa_v0, [batch_inline1034_inline2239__idx_v0])
            batch_cache_len_inline1026_inline2241__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(t__tile, pl.INDEX) // 4
            max_cache_len_inline1046_inline2234__ssa_v3: pl.Scalar[pl.INDEX] = pl.max(max_cache_len_inline1046_inline2234__iter_v1, batch_cache_len_inline1026_inline2241__ssa_v0)
            max_cache_len_inline1046_inline2234__rv_v2: pl.Scalar[pl.INDEX] = pl.yield_(max_cache_len_inline1046_inline2234__ssa_v3)
        max_leaves_inline1023_inline2216__ssa_v0: pl.Scalar[pl.INDEX] = pl.max((pl.min(max_cache_len_inline1046_inline2234__rv_v2, 262144) + 8191) // 8192, 1)
        single_leaf_inline1032_inline2263__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(max_leaves_inline1023_inline2216__ssa_v0 == 1, pl.INDEX)
        for item_inline1031_inline2243__idx_v0, (score_arena_inline1043_inline2235__iter_v1,) in pl.range(
            worker_inline1037_inline2245__ssa_v0, query_count_inline1038_inline2205__ssa_v0 * max_leaves_inline1023_inline2216__ssa_v0, 24, init_values=(score_arena_inline1043_inline2235__ssa_v0,)
        ):
            query_inline1030_inline2255__ssa_v0: pl.Scalar[pl.INDEX] = item_inline1031_inline2243__idx_v0 // max_leaves_inline1023_inline2216__ssa_v0
            leaf_inline1051_inline2227__ssa_v0: pl.Scalar[pl.INDEX] = item_inline1031_inline2243__idx_v0 % max_leaves_inline1023_inline2216__ssa_v0
            batch_idx_inline1058_inline2244__ssa_v0: pl.Scalar[pl.INDEX] = query_inline1030_inline2255__ssa_v0 // 6
            position_inline1016_inline2246__tile: pl.Scalar[pl.INT32] = pl.tensor.read(position_ids__ssa_v0, [query_inline1030_inline2255__ssa_v0])
            t__tile_1: pl.Scalar[pl.INT32] = pl.tensor.read(kv_seq_lens__ssa_v0, [batch_idx_inline1058_inline2244__ssa_v0])
            cache_len_inline1020_inline2232__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(t__tile_1, pl.INDEX) // 4
            cache_bound_inline1013_inline2248__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(cache_len_inline1020_inline2232__ssa_v0, (pl.cast(position_inline1016_inline2246__tile, pl.INDEX) + 1) // 4)
            visible_count_inline1019_inline2238__ssa_v0: pl.Scalar[pl.INDEX] = pl.max(pl.min(cache_bound_inline1013_inline2248__ssa_v0, 262144), 0)
            logical_begin_inline1045_inline2251__ssa_v0: pl.Scalar[pl.INDEX] = leaf_inline1051_inline2227__ssa_v0 * 8192
            if logical_begin_inline1045_inline2251__ssa_v0 < visible_count_inline1019_inline2238__ssa_v0:
                valid_count_inline1015_inline2253__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(visible_count_inline1019_inline2238__ssa_v0 - logical_begin_inline1045_inline2251__ssa_v0, 8192)
                lane_span_inline1067_inline2256__ssa_v0: pl.Scalar[pl.INDEX] = pl.min((valid_count_inline1015_inline2253__ssa_v0 + 383) // 384 * 192, 4096)
                lane_stride_inline1014_inline2207__ssa_v0: pl.Scalar[pl.INDEX] = (
                    single_leaf_inline1032_inline2263__ssa_v0 * 192 + (1 - single_leaf_inline1032_inline2263__ssa_v0) * lane_span_inline1067_inline2256__ssa_v0
                )
                query_head_begin_inline1017_inline2258__ssa_v0: pl.Scalar[pl.INDEX] = query_inline1030_inline2255__ssa_v0 * 64
                query_vector_inline1024_inline2229__tile: pl.Tile[[64, 128], pl.INT8, pl.MemRef(mem_mat_10, pl.const(0, pl.INT64), 8192), pl.Mem.Mat] = pl.tile.load(
                    qr_hadamard_i8_inline527__rv_v2, [query_head_begin_inline1017_inline2258__ssa_v0, 0], [64, 128], [64, 128], target_memory=pl.Mem.Mat
                )
                unroll_main_end: pl.Scalar[pl.INDEX] = (lane_span_inline1067_inline2256__ssa_v0 + 191) // 384 * 384
                for score_begin_inline1050_inline2269__idx_v0, (score_arena_inline1043_inline2235__iter_v3,) in pl.range(
                    0, unroll_main_end, 384, init_values=(score_arena_inline1043_inline2235__iter_v1,)
                ):
                    read_begin_inline1055_inline2200__ssa_v0: pl.Scalar[pl.INDEX] = score_begin_inline1050_inline2269__idx_v0 * (single_leaf_inline1032_inline2263__ssa_v0 + 1)
                    page_begin_inline1053_inline2194__ssa_v0: pl.Scalar[pl.INDEX] = 0
                    lane_page_inline1041_inline2193__ssa_v0: pl.Scalar[pl.INDEX] = page_begin_inline1053_inline2194__ssa_v0
                    safe_page_begin_inline1047_inline2192__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(
                        read_begin_inline1055_inline2200__ssa_v0 + lane_page_inline1041_inline2193__ssa_v0, (valid_count_inline1015_inline2253__ssa_v0 - 1) // 32 * 32
                    )
                    logical_page_inline1054_inline2247__ssa_v0: pl.Scalar[pl.INDEX] = (logical_begin_inline1045_inline2251__ssa_v0 + safe_page_begin_inline1047_inline2192__ssa_v0) // 32
                    t__tile_2: pl.Scalar[pl.INT32] = pl.tensor.read(
                        idx_block_table_flat_inline1044_inline2217__ssa_v0,
                        [batch_idx_inline1058_inline2244__ssa_v0 * table_columns_inline1029_inline2268__ssa_v0 + logical_page_inline1054_inline2247__ssa_v0],
                    )
                    physical_block_inline1056_inline2230__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(t__tile_2, pl.INDEX)
                    page_begin_inline1053_inline2194__ssa_v1: pl.Scalar[pl.INDEX] = 32
                    lane_page_inline1041_inline2193__ssa_v1: pl.Scalar[pl.INDEX] = page_begin_inline1053_inline2194__ssa_v1
                    safe_page_begin_inline1047_inline2192__ssa_v1: pl.Scalar[pl.INDEX] = pl.min(
                        read_begin_inline1055_inline2200__ssa_v0 + lane_page_inline1041_inline2193__ssa_v1, (valid_count_inline1015_inline2253__ssa_v0 - 1) // 32 * 32
                    )
                    logical_page_inline1054_inline2247__ssa_v1: pl.Scalar[pl.INDEX] = (logical_begin_inline1045_inline2251__ssa_v0 + safe_page_begin_inline1047_inline2192__ssa_v1) // 32
                    t__tile_3: pl.Scalar[pl.INT32] = pl.tensor.read(
                        idx_block_table_flat_inline1044_inline2217__ssa_v0,
                        [batch_idx_inline1058_inline2244__ssa_v0 * table_columns_inline1029_inline2268__ssa_v0 + logical_page_inline1054_inline2247__ssa_v1],
                    )
                    physical_block_inline1056_inline2230__ssa_v1: pl.Scalar[pl.INDEX] = pl.cast(t__tile_3, pl.INDEX)
                    page_begin_inline1053_inline2194__ssa_v2: pl.Scalar[pl.INDEX] = 64
                    lane_page_inline1041_inline2193__ssa_v2: pl.Scalar[pl.INDEX] = page_begin_inline1053_inline2194__ssa_v2
                    safe_page_begin_inline1047_inline2192__ssa_v2: pl.Scalar[pl.INDEX] = pl.min(
                        read_begin_inline1055_inline2200__ssa_v0 + lane_page_inline1041_inline2193__ssa_v2, (valid_count_inline1015_inline2253__ssa_v0 - 1) // 32 * 32
                    )
                    logical_page_inline1054_inline2247__ssa_v2: pl.Scalar[pl.INDEX] = (logical_begin_inline1045_inline2251__ssa_v0 + safe_page_begin_inline1047_inline2192__ssa_v2) // 32
                    t__tile_4: pl.Scalar[pl.INT32] = pl.tensor.read(
                        idx_block_table_flat_inline1044_inline2217__ssa_v0,
                        [batch_idx_inline1058_inline2244__ssa_v0 * table_columns_inline1029_inline2268__ssa_v0 + logical_page_inline1054_inline2247__ssa_v2],
                    )
                    physical_block_inline1056_inline2230__ssa_v2: pl.Scalar[pl.INDEX] = pl.cast(t__tile_4, pl.INDEX)
                    page_begin_inline1053_inline2194__ssa_v3: pl.Scalar[pl.INDEX] = 96
                    lane_page_inline1041_inline2193__ssa_v3: pl.Scalar[pl.INDEX] = page_begin_inline1053_inline2194__ssa_v3
                    safe_page_begin_inline1047_inline2192__ssa_v3: pl.Scalar[pl.INDEX] = pl.min(
                        read_begin_inline1055_inline2200__ssa_v0 + lane_page_inline1041_inline2193__ssa_v3, (valid_count_inline1015_inline2253__ssa_v0 - 1) // 32 * 32
                    )
                    logical_page_inline1054_inline2247__ssa_v3: pl.Scalar[pl.INDEX] = (logical_begin_inline1045_inline2251__ssa_v0 + safe_page_begin_inline1047_inline2192__ssa_v3) // 32
                    t__tile_5: pl.Scalar[pl.INT32] = pl.tensor.read(
                        idx_block_table_flat_inline1044_inline2217__ssa_v0,
                        [batch_idx_inline1058_inline2244__ssa_v0 * table_columns_inline1029_inline2268__ssa_v0 + logical_page_inline1054_inline2247__ssa_v3],
                    )
                    physical_block_inline1056_inline2230__ssa_v3: pl.Scalar[pl.INDEX] = pl.cast(t__tile_5, pl.INDEX)
                    page_begin_inline1053_inline2194__ssa_v4: pl.Scalar[pl.INDEX] = 128
                    lane_page_inline1041_inline2193__ssa_v4: pl.Scalar[pl.INDEX] = page_begin_inline1053_inline2194__ssa_v4
                    safe_page_begin_inline1047_inline2192__ssa_v4: pl.Scalar[pl.INDEX] = pl.min(
                        read_begin_inline1055_inline2200__ssa_v0 + lane_page_inline1041_inline2193__ssa_v4, (valid_count_inline1015_inline2253__ssa_v0 - 1) // 32 * 32
                    )
                    logical_page_inline1054_inline2247__ssa_v4: pl.Scalar[pl.INDEX] = (logical_begin_inline1045_inline2251__ssa_v0 + safe_page_begin_inline1047_inline2192__ssa_v4) // 32
                    t__tile_6: pl.Scalar[pl.INT32] = pl.tensor.read(
                        idx_block_table_flat_inline1044_inline2217__ssa_v0,
                        [batch_idx_inline1058_inline2244__ssa_v0 * table_columns_inline1029_inline2268__ssa_v0 + logical_page_inline1054_inline2247__ssa_v4],
                    )
                    physical_block_inline1056_inline2230__ssa_v4: pl.Scalar[pl.INDEX] = pl.cast(t__tile_6, pl.INDEX)
                    page_begin_inline1053_inline2194__ssa_v5: pl.Scalar[pl.INDEX] = 160
                    lane_page_inline1041_inline2193__ssa_v5: pl.Scalar[pl.INDEX] = page_begin_inline1053_inline2194__ssa_v5
                    safe_page_begin_inline1047_inline2192__ssa_v5: pl.Scalar[pl.INDEX] = pl.min(
                        read_begin_inline1055_inline2200__ssa_v0 + lane_page_inline1041_inline2193__ssa_v5, (valid_count_inline1015_inline2253__ssa_v0 - 1) // 32 * 32
                    )
                    logical_page_inline1054_inline2247__ssa_v5: pl.Scalar[pl.INDEX] = (logical_begin_inline1045_inline2251__ssa_v0 + safe_page_begin_inline1047_inline2192__ssa_v5) // 32
                    t__tile_7: pl.Scalar[pl.INT32] = pl.tensor.read(
                        idx_block_table_flat_inline1044_inline2217__ssa_v0,
                        [batch_idx_inline1058_inline2244__ssa_v0 * table_columns_inline1029_inline2268__ssa_v0 + logical_page_inline1054_inline2247__ssa_v5],
                    )
                    physical_block_inline1056_inline2230__ssa_v5: pl.Scalar[pl.INDEX] = pl.cast(t__tile_7, pl.INDEX)
                    page_begin_inline1053_inline2194__ssa_v6: pl.Scalar[pl.INDEX] = 192
                    lane_page_inline1041_inline2193__ssa_v6: pl.Scalar[pl.INDEX] = (
                        page_begin_inline1053_inline2194__ssa_v6 // 192 * lane_stride_inline1014_inline2207__ssa_v0 + page_begin_inline1053_inline2194__ssa_v6 % 192
                    )
                    safe_page_begin_inline1047_inline2192__ssa_v6: pl.Scalar[pl.INDEX] = pl.min(
                        read_begin_inline1055_inline2200__ssa_v0 + lane_page_inline1041_inline2193__ssa_v6, (valid_count_inline1015_inline2253__ssa_v0 - 1) // 32 * 32
                    )
                    logical_page_inline1054_inline2247__ssa_v6: pl.Scalar[pl.INDEX] = (logical_begin_inline1045_inline2251__ssa_v0 + safe_page_begin_inline1047_inline2192__ssa_v6) // 32
                    t__tile_8: pl.Scalar[pl.INT32] = pl.tensor.read(
                        idx_block_table_flat_inline1044_inline2217__ssa_v0,
                        [batch_idx_inline1058_inline2244__ssa_v0 * table_columns_inline1029_inline2268__ssa_v0 + logical_page_inline1054_inline2247__ssa_v6],
                    )
                    physical_block_inline1056_inline2230__ssa_v6: pl.Scalar[pl.INDEX] = pl.cast(t__tile_8, pl.INDEX)
                    page_begin_inline1053_inline2194__ssa_v7: pl.Scalar[pl.INDEX] = 224
                    lane_page_inline1041_inline2193__ssa_v7: pl.Scalar[pl.INDEX] = (
                        page_begin_inline1053_inline2194__ssa_v7 // 192 * lane_stride_inline1014_inline2207__ssa_v0 + page_begin_inline1053_inline2194__ssa_v7 % 192
                    )
                    safe_page_begin_inline1047_inline2192__ssa_v7: pl.Scalar[pl.INDEX] = pl.min(
                        read_begin_inline1055_inline2200__ssa_v0 + lane_page_inline1041_inline2193__ssa_v7, (valid_count_inline1015_inline2253__ssa_v0 - 1) // 32 * 32
                    )
                    logical_page_inline1054_inline2247__ssa_v7: pl.Scalar[pl.INDEX] = (logical_begin_inline1045_inline2251__ssa_v0 + safe_page_begin_inline1047_inline2192__ssa_v7) // 32
                    t__tile_9: pl.Scalar[pl.INT32] = pl.tensor.read(
                        idx_block_table_flat_inline1044_inline2217__ssa_v0,
                        [batch_idx_inline1058_inline2244__ssa_v0 * table_columns_inline1029_inline2268__ssa_v0 + logical_page_inline1054_inline2247__ssa_v7],
                    )
                    physical_block_inline1056_inline2230__ssa_v7: pl.Scalar[pl.INDEX] = pl.cast(t__tile_9, pl.INDEX)
                    page_begin_inline1053_inline2194__ssa_v8: pl.Scalar[pl.INDEX] = 256
                    lane_page_inline1041_inline2193__ssa_v8: pl.Scalar[pl.INDEX] = (
                        page_begin_inline1053_inline2194__ssa_v8 // 192 * lane_stride_inline1014_inline2207__ssa_v0 + page_begin_inline1053_inline2194__ssa_v8 % 192
                    )
                    safe_page_begin_inline1047_inline2192__ssa_v8: pl.Scalar[pl.INDEX] = pl.min(
                        read_begin_inline1055_inline2200__ssa_v0 + lane_page_inline1041_inline2193__ssa_v8, (valid_count_inline1015_inline2253__ssa_v0 - 1) // 32 * 32
                    )
                    logical_page_inline1054_inline2247__ssa_v8: pl.Scalar[pl.INDEX] = (logical_begin_inline1045_inline2251__ssa_v0 + safe_page_begin_inline1047_inline2192__ssa_v8) // 32
                    t__tile_10: pl.Scalar[pl.INT32] = pl.tensor.read(
                        idx_block_table_flat_inline1044_inline2217__ssa_v0,
                        [batch_idx_inline1058_inline2244__ssa_v0 * table_columns_inline1029_inline2268__ssa_v0 + logical_page_inline1054_inline2247__ssa_v8],
                    )
                    physical_block_inline1056_inline2230__ssa_v8: pl.Scalar[pl.INDEX] = pl.cast(t__tile_10, pl.INDEX)
                    page_begin_inline1053_inline2194__ssa_v9: pl.Scalar[pl.INDEX] = 288
                    lane_page_inline1041_inline2193__ssa_v9: pl.Scalar[pl.INDEX] = (
                        page_begin_inline1053_inline2194__ssa_v9 // 192 * lane_stride_inline1014_inline2207__ssa_v0 + page_begin_inline1053_inline2194__ssa_v9 % 192
                    )
                    safe_page_begin_inline1047_inline2192__ssa_v9: pl.Scalar[pl.INDEX] = pl.min(
                        read_begin_inline1055_inline2200__ssa_v0 + lane_page_inline1041_inline2193__ssa_v9, (valid_count_inline1015_inline2253__ssa_v0 - 1) // 32 * 32
                    )
                    logical_page_inline1054_inline2247__ssa_v9: pl.Scalar[pl.INDEX] = (logical_begin_inline1045_inline2251__ssa_v0 + safe_page_begin_inline1047_inline2192__ssa_v9) // 32
                    t__tile_11: pl.Scalar[pl.INT32] = pl.tensor.read(
                        idx_block_table_flat_inline1044_inline2217__ssa_v0,
                        [batch_idx_inline1058_inline2244__ssa_v0 * table_columns_inline1029_inline2268__ssa_v0 + logical_page_inline1054_inline2247__ssa_v9],
                    )
                    physical_block_inline1056_inline2230__ssa_v9: pl.Scalar[pl.INDEX] = pl.cast(t__tile_11, pl.INDEX)
                    page_begin_inline1053_inline2194__ssa_v10: pl.Scalar[pl.INDEX] = 320
                    lane_page_inline1041_inline2193__ssa_v10: pl.Scalar[pl.INDEX] = (
                        page_begin_inline1053_inline2194__ssa_v10 // 192 * lane_stride_inline1014_inline2207__ssa_v0 + page_begin_inline1053_inline2194__ssa_v10 % 192
                    )
                    safe_page_begin_inline1047_inline2192__ssa_v10: pl.Scalar[pl.INDEX] = pl.min(
                        read_begin_inline1055_inline2200__ssa_v0 + lane_page_inline1041_inline2193__ssa_v10, (valid_count_inline1015_inline2253__ssa_v0 - 1) // 32 * 32
                    )
                    logical_page_inline1054_inline2247__ssa_v10: pl.Scalar[pl.INDEX] = (logical_begin_inline1045_inline2251__ssa_v0 + safe_page_begin_inline1047_inline2192__ssa_v10) // 32
                    t__tile_12: pl.Scalar[pl.INT32] = pl.tensor.read(
                        idx_block_table_flat_inline1044_inline2217__ssa_v0,
                        [batch_idx_inline1058_inline2244__ssa_v0 * table_columns_inline1029_inline2268__ssa_v0 + logical_page_inline1054_inline2247__ssa_v10],
                    )
                    physical_block_inline1056_inline2230__ssa_v10: pl.Scalar[pl.INDEX] = pl.cast(t__tile_12, pl.INDEX)
                    page_begin_inline1053_inline2194__ssa_v11: pl.Scalar[pl.INDEX] = 352
                    lane_page_inline1041_inline2193__ssa_v11: pl.Scalar[pl.INDEX] = (
                        page_begin_inline1053_inline2194__ssa_v11 // 192 * lane_stride_inline1014_inline2207__ssa_v0 + page_begin_inline1053_inline2194__ssa_v11 % 192
                    )
                    safe_page_begin_inline1047_inline2192__ssa_v11: pl.Scalar[pl.INDEX] = pl.min(
                        read_begin_inline1055_inline2200__ssa_v0 + lane_page_inline1041_inline2193__ssa_v11, (valid_count_inline1015_inline2253__ssa_v0 - 1) // 32 * 32
                    )
                    logical_page_inline1054_inline2247__ssa_v11: pl.Scalar[pl.INDEX] = (logical_begin_inline1045_inline2251__ssa_v0 + safe_page_begin_inline1047_inline2192__ssa_v11) // 32
                    t__tile_13: pl.Scalar[pl.INT32] = pl.tensor.read(
                        idx_block_table_flat_inline1044_inline2217__ssa_v0,
                        [batch_idx_inline1058_inline2244__ssa_v0 * table_columns_inline1029_inline2268__ssa_v0 + logical_page_inline1054_inline2247__ssa_v11],
                    )
                    physical_block_inline1056_inline2230__ssa_v11: pl.Scalar[pl.INDEX] = pl.cast(t__tile_13, pl.INDEX)
                    read_begin_inline1055_inline2200__ssa_v0_1: pl.Scalar[pl.INDEX] = (score_begin_inline1050_inline2269__idx_v0 + 192) * (single_leaf_inline1032_inline2263__ssa_v0 + 1)
                    page_begin_inline1053_inline2194__ssa_v0_1: pl.Scalar[pl.INDEX] = 0
                    lane_page_inline1041_inline2193__ssa_v0_1: pl.Scalar[pl.INDEX] = page_begin_inline1053_inline2194__ssa_v0_1
                    safe_page_begin_inline1047_inline2192__ssa_v0_1: pl.Scalar[pl.INDEX] = pl.min(
                        read_begin_inline1055_inline2200__ssa_v0_1 + lane_page_inline1041_inline2193__ssa_v0_1, (valid_count_inline1015_inline2253__ssa_v0 - 1) // 32 * 32
                    )
                    logical_page_inline1054_inline2247__ssa_v0_1: pl.Scalar[pl.INDEX] = (logical_begin_inline1045_inline2251__ssa_v0 + safe_page_begin_inline1047_inline2192__ssa_v0_1) // 32
                    t__tile_14: pl.Scalar[pl.INT32] = pl.tensor.read(
                        idx_block_table_flat_inline1044_inline2217__ssa_v0,
                        [batch_idx_inline1058_inline2244__ssa_v0 * table_columns_inline1029_inline2268__ssa_v0 + logical_page_inline1054_inline2247__ssa_v0_1],
                    )
                    physical_block_inline1056_inline2230__ssa_v0_1: pl.Scalar[pl.INDEX] = pl.cast(t__tile_14, pl.INDEX)
                    page_begin_inline1053_inline2194__ssa_v1_1: pl.Scalar[pl.INDEX] = 32
                    lane_page_inline1041_inline2193__ssa_v1_1: pl.Scalar[pl.INDEX] = page_begin_inline1053_inline2194__ssa_v1_1
                    safe_page_begin_inline1047_inline2192__ssa_v1_1: pl.Scalar[pl.INDEX] = pl.min(
                        read_begin_inline1055_inline2200__ssa_v0_1 + lane_page_inline1041_inline2193__ssa_v1_1, (valid_count_inline1015_inline2253__ssa_v0 - 1) // 32 * 32
                    )
                    logical_page_inline1054_inline2247__ssa_v1_1: pl.Scalar[pl.INDEX] = (logical_begin_inline1045_inline2251__ssa_v0 + safe_page_begin_inline1047_inline2192__ssa_v1_1) // 32
                    t__tile_15: pl.Scalar[pl.INT32] = pl.tensor.read(
                        idx_block_table_flat_inline1044_inline2217__ssa_v0,
                        [batch_idx_inline1058_inline2244__ssa_v0 * table_columns_inline1029_inline2268__ssa_v0 + logical_page_inline1054_inline2247__ssa_v1_1],
                    )
                    physical_block_inline1056_inline2230__ssa_v1_1: pl.Scalar[pl.INDEX] = pl.cast(t__tile_15, pl.INDEX)
                    page_begin_inline1053_inline2194__ssa_v2_1: pl.Scalar[pl.INDEX] = 64
                    lane_page_inline1041_inline2193__ssa_v2_1: pl.Scalar[pl.INDEX] = page_begin_inline1053_inline2194__ssa_v2_1
                    safe_page_begin_inline1047_inline2192__ssa_v2_1: pl.Scalar[pl.INDEX] = pl.min(
                        read_begin_inline1055_inline2200__ssa_v0_1 + lane_page_inline1041_inline2193__ssa_v2_1, (valid_count_inline1015_inline2253__ssa_v0 - 1) // 32 * 32
                    )
                    logical_page_inline1054_inline2247__ssa_v2_1: pl.Scalar[pl.INDEX] = (logical_begin_inline1045_inline2251__ssa_v0 + safe_page_begin_inline1047_inline2192__ssa_v2_1) // 32
                    t__tile_16: pl.Scalar[pl.INT32] = pl.tensor.read(
                        idx_block_table_flat_inline1044_inline2217__ssa_v0,
                        [batch_idx_inline1058_inline2244__ssa_v0 * table_columns_inline1029_inline2268__ssa_v0 + logical_page_inline1054_inline2247__ssa_v2_1],
                    )
                    physical_block_inline1056_inline2230__ssa_v2_1: pl.Scalar[pl.INDEX] = pl.cast(t__tile_16, pl.INDEX)
                    page_begin_inline1053_inline2194__ssa_v3_1: pl.Scalar[pl.INDEX] = 96
                    lane_page_inline1041_inline2193__ssa_v3_1: pl.Scalar[pl.INDEX] = page_begin_inline1053_inline2194__ssa_v3_1
                    safe_page_begin_inline1047_inline2192__ssa_v3_1: pl.Scalar[pl.INDEX] = pl.min(
                        read_begin_inline1055_inline2200__ssa_v0_1 + lane_page_inline1041_inline2193__ssa_v3_1, (valid_count_inline1015_inline2253__ssa_v0 - 1) // 32 * 32
                    )
                    logical_page_inline1054_inline2247__ssa_v3_1: pl.Scalar[pl.INDEX] = (logical_begin_inline1045_inline2251__ssa_v0 + safe_page_begin_inline1047_inline2192__ssa_v3_1) // 32
                    t__tile_17: pl.Scalar[pl.INT32] = pl.tensor.read(
                        idx_block_table_flat_inline1044_inline2217__ssa_v0,
                        [batch_idx_inline1058_inline2244__ssa_v0 * table_columns_inline1029_inline2268__ssa_v0 + logical_page_inline1054_inline2247__ssa_v3_1],
                    )
                    physical_block_inline1056_inline2230__ssa_v3_1: pl.Scalar[pl.INDEX] = pl.cast(t__tile_17, pl.INDEX)
                    page_begin_inline1053_inline2194__ssa_v4_1: pl.Scalar[pl.INDEX] = 128
                    lane_page_inline1041_inline2193__ssa_v4_1: pl.Scalar[pl.INDEX] = page_begin_inline1053_inline2194__ssa_v4_1
                    safe_page_begin_inline1047_inline2192__ssa_v4_1: pl.Scalar[pl.INDEX] = pl.min(
                        read_begin_inline1055_inline2200__ssa_v0_1 + lane_page_inline1041_inline2193__ssa_v4_1, (valid_count_inline1015_inline2253__ssa_v0 - 1) // 32 * 32
                    )
                    logical_page_inline1054_inline2247__ssa_v4_1: pl.Scalar[pl.INDEX] = (logical_begin_inline1045_inline2251__ssa_v0 + safe_page_begin_inline1047_inline2192__ssa_v4_1) // 32
                    t__tile_18: pl.Scalar[pl.INT32] = pl.tensor.read(
                        idx_block_table_flat_inline1044_inline2217__ssa_v0,
                        [batch_idx_inline1058_inline2244__ssa_v0 * table_columns_inline1029_inline2268__ssa_v0 + logical_page_inline1054_inline2247__ssa_v4_1],
                    )
                    physical_block_inline1056_inline2230__ssa_v4_1: pl.Scalar[pl.INDEX] = pl.cast(t__tile_18, pl.INDEX)
                    page_begin_inline1053_inline2194__ssa_v5_1: pl.Scalar[pl.INDEX] = 160
                    lane_page_inline1041_inline2193__ssa_v5_1: pl.Scalar[pl.INDEX] = page_begin_inline1053_inline2194__ssa_v5_1
                    safe_page_begin_inline1047_inline2192__ssa_v5_1: pl.Scalar[pl.INDEX] = pl.min(
                        read_begin_inline1055_inline2200__ssa_v0_1 + lane_page_inline1041_inline2193__ssa_v5_1, (valid_count_inline1015_inline2253__ssa_v0 - 1) // 32 * 32
                    )
                    logical_page_inline1054_inline2247__ssa_v5_1: pl.Scalar[pl.INDEX] = (logical_begin_inline1045_inline2251__ssa_v0 + safe_page_begin_inline1047_inline2192__ssa_v5_1) // 32
                    t__tile_19: pl.Scalar[pl.INT32] = pl.tensor.read(
                        idx_block_table_flat_inline1044_inline2217__ssa_v0,
                        [batch_idx_inline1058_inline2244__ssa_v0 * table_columns_inline1029_inline2268__ssa_v0 + logical_page_inline1054_inline2247__ssa_v5_1],
                    )
                    physical_block_inline1056_inline2230__ssa_v5_1: pl.Scalar[pl.INDEX] = pl.cast(t__tile_19, pl.INDEX)
                    page_begin_inline1053_inline2194__ssa_v6_1: pl.Scalar[pl.INDEX] = 192
                    lane_page_inline1041_inline2193__ssa_v6_1: pl.Scalar[pl.INDEX] = (
                        page_begin_inline1053_inline2194__ssa_v6_1 // 192 * lane_stride_inline1014_inline2207__ssa_v0 + page_begin_inline1053_inline2194__ssa_v6_1 % 192
                    )
                    safe_page_begin_inline1047_inline2192__ssa_v6_1: pl.Scalar[pl.INDEX] = pl.min(
                        read_begin_inline1055_inline2200__ssa_v0_1 + lane_page_inline1041_inline2193__ssa_v6_1, (valid_count_inline1015_inline2253__ssa_v0 - 1) // 32 * 32
                    )
                    logical_page_inline1054_inline2247__ssa_v6_1: pl.Scalar[pl.INDEX] = (logical_begin_inline1045_inline2251__ssa_v0 + safe_page_begin_inline1047_inline2192__ssa_v6_1) // 32
                    t__tile_20: pl.Scalar[pl.INT32] = pl.tensor.read(
                        idx_block_table_flat_inline1044_inline2217__ssa_v0,
                        [batch_idx_inline1058_inline2244__ssa_v0 * table_columns_inline1029_inline2268__ssa_v0 + logical_page_inline1054_inline2247__ssa_v6_1],
                    )
                    physical_block_inline1056_inline2230__ssa_v6_1: pl.Scalar[pl.INDEX] = pl.cast(t__tile_20, pl.INDEX)
                    page_begin_inline1053_inline2194__ssa_v7_1: pl.Scalar[pl.INDEX] = 224
                    lane_page_inline1041_inline2193__ssa_v7_1: pl.Scalar[pl.INDEX] = (
                        page_begin_inline1053_inline2194__ssa_v7_1 // 192 * lane_stride_inline1014_inline2207__ssa_v0 + page_begin_inline1053_inline2194__ssa_v7_1 % 192
                    )
                    safe_page_begin_inline1047_inline2192__ssa_v7_1: pl.Scalar[pl.INDEX] = pl.min(
                        read_begin_inline1055_inline2200__ssa_v0_1 + lane_page_inline1041_inline2193__ssa_v7_1, (valid_count_inline1015_inline2253__ssa_v0 - 1) // 32 * 32
                    )
                    logical_page_inline1054_inline2247__ssa_v7_1: pl.Scalar[pl.INDEX] = (logical_begin_inline1045_inline2251__ssa_v0 + safe_page_begin_inline1047_inline2192__ssa_v7_1) // 32
                    t__tile_21: pl.Scalar[pl.INT32] = pl.tensor.read(
                        idx_block_table_flat_inline1044_inline2217__ssa_v0,
                        [batch_idx_inline1058_inline2244__ssa_v0 * table_columns_inline1029_inline2268__ssa_v0 + logical_page_inline1054_inline2247__ssa_v7_1],
                    )
                    physical_block_inline1056_inline2230__ssa_v7_1: pl.Scalar[pl.INDEX] = pl.cast(t__tile_21, pl.INDEX)
                    page_begin_inline1053_inline2194__ssa_v8_1: pl.Scalar[pl.INDEX] = 256
                    lane_page_inline1041_inline2193__ssa_v8_1: pl.Scalar[pl.INDEX] = (
                        page_begin_inline1053_inline2194__ssa_v8_1 // 192 * lane_stride_inline1014_inline2207__ssa_v0 + page_begin_inline1053_inline2194__ssa_v8_1 % 192
                    )
                    safe_page_begin_inline1047_inline2192__ssa_v8_1: pl.Scalar[pl.INDEX] = pl.min(
                        read_begin_inline1055_inline2200__ssa_v0_1 + lane_page_inline1041_inline2193__ssa_v8_1, (valid_count_inline1015_inline2253__ssa_v0 - 1) // 32 * 32
                    )
                    logical_page_inline1054_inline2247__ssa_v8_1: pl.Scalar[pl.INDEX] = (logical_begin_inline1045_inline2251__ssa_v0 + safe_page_begin_inline1047_inline2192__ssa_v8_1) // 32
                    t__tile_22: pl.Scalar[pl.INT32] = pl.tensor.read(
                        idx_block_table_flat_inline1044_inline2217__ssa_v0,
                        [batch_idx_inline1058_inline2244__ssa_v0 * table_columns_inline1029_inline2268__ssa_v0 + logical_page_inline1054_inline2247__ssa_v8_1],
                    )
                    physical_block_inline1056_inline2230__ssa_v8_1: pl.Scalar[pl.INDEX] = pl.cast(t__tile_22, pl.INDEX)
                    page_begin_inline1053_inline2194__ssa_v9_1: pl.Scalar[pl.INDEX] = 288
                    lane_page_inline1041_inline2193__ssa_v9_1: pl.Scalar[pl.INDEX] = (
                        page_begin_inline1053_inline2194__ssa_v9_1 // 192 * lane_stride_inline1014_inline2207__ssa_v0 + page_begin_inline1053_inline2194__ssa_v9_1 % 192
                    )
                    safe_page_begin_inline1047_inline2192__ssa_v9_1: pl.Scalar[pl.INDEX] = pl.min(
                        read_begin_inline1055_inline2200__ssa_v0_1 + lane_page_inline1041_inline2193__ssa_v9_1, (valid_count_inline1015_inline2253__ssa_v0 - 1) // 32 * 32
                    )
                    logical_page_inline1054_inline2247__ssa_v9_1: pl.Scalar[pl.INDEX] = (logical_begin_inline1045_inline2251__ssa_v0 + safe_page_begin_inline1047_inline2192__ssa_v9_1) // 32
                    t__tile_23: pl.Scalar[pl.INT32] = pl.tensor.read(
                        idx_block_table_flat_inline1044_inline2217__ssa_v0,
                        [batch_idx_inline1058_inline2244__ssa_v0 * table_columns_inline1029_inline2268__ssa_v0 + logical_page_inline1054_inline2247__ssa_v9_1],
                    )
                    physical_block_inline1056_inline2230__ssa_v9_1: pl.Scalar[pl.INDEX] = pl.cast(t__tile_23, pl.INDEX)
                    page_begin_inline1053_inline2194__ssa_v10_1: pl.Scalar[pl.INDEX] = 320
                    lane_page_inline1041_inline2193__ssa_v10_1: pl.Scalar[pl.INDEX] = (
                        page_begin_inline1053_inline2194__ssa_v10_1 // 192 * lane_stride_inline1014_inline2207__ssa_v0 + page_begin_inline1053_inline2194__ssa_v10_1 % 192
                    )
                    safe_page_begin_inline1047_inline2192__ssa_v10_1: pl.Scalar[pl.INDEX] = pl.min(
                        read_begin_inline1055_inline2200__ssa_v0_1 + lane_page_inline1041_inline2193__ssa_v10_1, (valid_count_inline1015_inline2253__ssa_v0 - 1) // 32 * 32
                    )
                    logical_page_inline1054_inline2247__ssa_v10_1: pl.Scalar[pl.INDEX] = (logical_begin_inline1045_inline2251__ssa_v0 + safe_page_begin_inline1047_inline2192__ssa_v10_1) // 32
                    t__tile_24: pl.Scalar[pl.INT32] = pl.tensor.read(
                        idx_block_table_flat_inline1044_inline2217__ssa_v0,
                        [batch_idx_inline1058_inline2244__ssa_v0 * table_columns_inline1029_inline2268__ssa_v0 + logical_page_inline1054_inline2247__ssa_v10_1],
                    )
                    physical_block_inline1056_inline2230__ssa_v10_1: pl.Scalar[pl.INDEX] = pl.cast(t__tile_24, pl.INDEX)
                    page_begin_inline1053_inline2194__ssa_v11_1: pl.Scalar[pl.INDEX] = 352
                    lane_page_inline1041_inline2193__ssa_v11_1: pl.Scalar[pl.INDEX] = (
                        page_begin_inline1053_inline2194__ssa_v11_1 // 192 * lane_stride_inline1014_inline2207__ssa_v0 + page_begin_inline1053_inline2194__ssa_v11_1 % 192
                    )
                    safe_page_begin_inline1047_inline2192__ssa_v11_1: pl.Scalar[pl.INDEX] = pl.min(
                        read_begin_inline1055_inline2200__ssa_v0_1 + lane_page_inline1041_inline2193__ssa_v11_1, (valid_count_inline1015_inline2253__ssa_v0 - 1) // 32 * 32
                    )
                    logical_page_inline1054_inline2247__ssa_v11_1: pl.Scalar[pl.INDEX] = (logical_begin_inline1045_inline2251__ssa_v0 + safe_page_begin_inline1047_inline2192__ssa_v11_1) // 32
                    t__tile_25: pl.Scalar[pl.INT32] = pl.tensor.read(
                        idx_block_table_flat_inline1044_inline2217__ssa_v0,
                        [batch_idx_inline1058_inline2244__ssa_v0 * table_columns_inline1029_inline2268__ssa_v0 + logical_page_inline1054_inline2247__ssa_v11_1],
                    )
                    physical_block_inline1056_inline2230__ssa_v11_1: pl.Scalar[pl.INDEX] = pl.cast(t__tile_25, pl.INDEX)
                    kv_i8_inline1052_inline2201__tile: pl.Tile[[384, 128], pl.INT8, pl.MemRef(mem_mat_11, pl.const(8192, pl.INT64), 49152), pl.Mem.Mat] = pl.tile.create(
                        [384, 128], dtype=pl.INT8, target_memory=pl.Mem.Mat, transpose=False
                    )
                    for key_row_inline1057_inline2257__idx_v0, (kv_i8_inline1052_inline2201__iter_v1,) in pl.range(32, init_values=(kv_i8_inline1052_inline2201__tile,)):
                        kv_i8_inline1052_inline2201__tile_1: pl.Tile[[384, 128], pl.INT8, pl.MemRef(mem_mat_11, pl.const(8192, pl.INT64), 49152), pl.Mem.Mat] = pl.tile.gather_row(
                            kv_i8_inline1052_inline2201__iter_v1,
                            idx_kv_cache__rv_v2,
                            [page_begin_inline1053_inline2194__ssa_v0 + key_row_inline1057_inline2257__idx_v0, 0],
                            [physical_block_inline1056_inline2230__ssa_v0, key_row_inline1057_inline2257__idx_v0 * 128],
                            [1, 128],
                            transpose=False,
                        )
                        kv_i8_inline1052_inline2201__rv_v2: pl.Tile[[384, 128], pl.INT8, pl.MemRef(mem_mat_11, pl.const(8192, pl.INT64), 49152), pl.Mem.Mat] = pl.yield_(
                            kv_i8_inline1052_inline2201__tile_1
                        )
                    for key_row_inline1057_inline2257__idx_v1, (kv_i8_inline1052_inline2201__iter_v4,) in pl.range(32, init_values=(kv_i8_inline1052_inline2201__rv_v2,)):
                        kv_i8_inline1052_inline2201__tile_2: pl.Tile[[384, 128], pl.INT8, pl.MemRef(mem_mat_11, pl.const(8192, pl.INT64), 49152), pl.Mem.Mat] = pl.tile.gather_row(
                            kv_i8_inline1052_inline2201__iter_v4,
                            idx_kv_cache__rv_v2,
                            [page_begin_inline1053_inline2194__ssa_v1 + key_row_inline1057_inline2257__idx_v1, 0],
                            [physical_block_inline1056_inline2230__ssa_v1, key_row_inline1057_inline2257__idx_v1 * 128],
                            [1, 128],
                            transpose=False,
                        )
                        kv_i8_inline1052_inline2201__rv_v5: pl.Tile[[384, 128], pl.INT8, pl.MemRef(mem_mat_11, pl.const(8192, pl.INT64), 49152), pl.Mem.Mat] = pl.yield_(
                            kv_i8_inline1052_inline2201__tile_2
                        )
                    for key_row_inline1057_inline2257__idx_v2, (kv_i8_inline1052_inline2201__iter_v7,) in pl.range(32, init_values=(kv_i8_inline1052_inline2201__rv_v5,)):
                        kv_i8_inline1052_inline2201__tile_3: pl.Tile[[384, 128], pl.INT8, pl.MemRef(mem_mat_11, pl.const(8192, pl.INT64), 49152), pl.Mem.Mat] = pl.tile.gather_row(
                            kv_i8_inline1052_inline2201__iter_v7,
                            idx_kv_cache__rv_v2,
                            [page_begin_inline1053_inline2194__ssa_v2 + key_row_inline1057_inline2257__idx_v2, 0],
                            [physical_block_inline1056_inline2230__ssa_v2, key_row_inline1057_inline2257__idx_v2 * 128],
                            [1, 128],
                            transpose=False,
                        )
                        kv_i8_inline1052_inline2201__rv_v8: pl.Tile[[384, 128], pl.INT8, pl.MemRef(mem_mat_11, pl.const(8192, pl.INT64), 49152), pl.Mem.Mat] = pl.yield_(
                            kv_i8_inline1052_inline2201__tile_3
                        )
                    for key_row_inline1057_inline2257__idx_v3, (kv_i8_inline1052_inline2201__iter_v10,) in pl.range(32, init_values=(kv_i8_inline1052_inline2201__rv_v8,)):
                        kv_i8_inline1052_inline2201__tile_4: pl.Tile[[384, 128], pl.INT8, pl.MemRef(mem_mat_11, pl.const(8192, pl.INT64), 49152), pl.Mem.Mat] = pl.tile.gather_row(
                            kv_i8_inline1052_inline2201__iter_v10,
                            idx_kv_cache__rv_v2,
                            [page_begin_inline1053_inline2194__ssa_v3 + key_row_inline1057_inline2257__idx_v3, 0],
                            [physical_block_inline1056_inline2230__ssa_v3, key_row_inline1057_inline2257__idx_v3 * 128],
                            [1, 128],
                            transpose=False,
                        )
                        kv_i8_inline1052_inline2201__rv_v11: pl.Tile[[384, 128], pl.INT8, pl.MemRef(mem_mat_11, pl.const(8192, pl.INT64), 49152), pl.Mem.Mat] = pl.yield_(
                            kv_i8_inline1052_inline2201__tile_4
                        )
                    for key_row_inline1057_inline2257__idx_v4, (kv_i8_inline1052_inline2201__iter_v13,) in pl.range(32, init_values=(kv_i8_inline1052_inline2201__rv_v11,)):
                        kv_i8_inline1052_inline2201__tile_5: pl.Tile[[384, 128], pl.INT8, pl.MemRef(mem_mat_11, pl.const(8192, pl.INT64), 49152), pl.Mem.Mat] = pl.tile.gather_row(
                            kv_i8_inline1052_inline2201__iter_v13,
                            idx_kv_cache__rv_v2,
                            [page_begin_inline1053_inline2194__ssa_v4 + key_row_inline1057_inline2257__idx_v4, 0],
                            [physical_block_inline1056_inline2230__ssa_v4, key_row_inline1057_inline2257__idx_v4 * 128],
                            [1, 128],
                            transpose=False,
                        )
                        kv_i8_inline1052_inline2201__rv_v14: pl.Tile[[384, 128], pl.INT8, pl.MemRef(mem_mat_11, pl.const(8192, pl.INT64), 49152), pl.Mem.Mat] = pl.yield_(
                            kv_i8_inline1052_inline2201__tile_5
                        )
                    for key_row_inline1057_inline2257__idx_v5, (kv_i8_inline1052_inline2201__iter_v16,) in pl.range(32, init_values=(kv_i8_inline1052_inline2201__rv_v14,)):
                        kv_i8_inline1052_inline2201__tile_6: pl.Tile[[384, 128], pl.INT8, pl.MemRef(mem_mat_11, pl.const(8192, pl.INT64), 49152), pl.Mem.Mat] = pl.tile.gather_row(
                            kv_i8_inline1052_inline2201__iter_v16,
                            idx_kv_cache__rv_v2,
                            [page_begin_inline1053_inline2194__ssa_v5 + key_row_inline1057_inline2257__idx_v5, 0],
                            [physical_block_inline1056_inline2230__ssa_v5, key_row_inline1057_inline2257__idx_v5 * 128],
                            [1, 128],
                            transpose=False,
                        )
                        kv_i8_inline1052_inline2201__rv_v17: pl.Tile[[384, 128], pl.INT8, pl.MemRef(mem_mat_11, pl.const(8192, pl.INT64), 49152), pl.Mem.Mat] = pl.yield_(
                            kv_i8_inline1052_inline2201__tile_6
                        )
                    for key_row_inline1057_inline2257__idx_v6, (kv_i8_inline1052_inline2201__iter_v19,) in pl.range(32, init_values=(kv_i8_inline1052_inline2201__rv_v17,)):
                        kv_i8_inline1052_inline2201__tile_7: pl.Tile[[384, 128], pl.INT8, pl.MemRef(mem_mat_11, pl.const(8192, pl.INT64), 49152), pl.Mem.Mat] = pl.tile.gather_row(
                            kv_i8_inline1052_inline2201__iter_v19,
                            idx_kv_cache__rv_v2,
                            [page_begin_inline1053_inline2194__ssa_v6 + key_row_inline1057_inline2257__idx_v6, 0],
                            [physical_block_inline1056_inline2230__ssa_v6, key_row_inline1057_inline2257__idx_v6 * 128],
                            [1, 128],
                            transpose=False,
                        )
                        kv_i8_inline1052_inline2201__rv_v20: pl.Tile[[384, 128], pl.INT8, pl.MemRef(mem_mat_11, pl.const(8192, pl.INT64), 49152), pl.Mem.Mat] = pl.yield_(
                            kv_i8_inline1052_inline2201__tile_7
                        )
                    for key_row_inline1057_inline2257__idx_v7, (kv_i8_inline1052_inline2201__iter_v22,) in pl.range(32, init_values=(kv_i8_inline1052_inline2201__rv_v20,)):
                        kv_i8_inline1052_inline2201__tile_8: pl.Tile[[384, 128], pl.INT8, pl.MemRef(mem_mat_11, pl.const(8192, pl.INT64), 49152), pl.Mem.Mat] = pl.tile.gather_row(
                            kv_i8_inline1052_inline2201__iter_v22,
                            idx_kv_cache__rv_v2,
                            [page_begin_inline1053_inline2194__ssa_v7 + key_row_inline1057_inline2257__idx_v7, 0],
                            [physical_block_inline1056_inline2230__ssa_v7, key_row_inline1057_inline2257__idx_v7 * 128],
                            [1, 128],
                            transpose=False,
                        )
                        kv_i8_inline1052_inline2201__rv_v23: pl.Tile[[384, 128], pl.INT8, pl.MemRef(mem_mat_11, pl.const(8192, pl.INT64), 49152), pl.Mem.Mat] = pl.yield_(
                            kv_i8_inline1052_inline2201__tile_8
                        )
                    for key_row_inline1057_inline2257__idx_v8, (kv_i8_inline1052_inline2201__iter_v25,) in pl.range(32, init_values=(kv_i8_inline1052_inline2201__rv_v23,)):
                        kv_i8_inline1052_inline2201__tile_9: pl.Tile[[384, 128], pl.INT8, pl.MemRef(mem_mat_11, pl.const(8192, pl.INT64), 49152), pl.Mem.Mat] = pl.tile.gather_row(
                            kv_i8_inline1052_inline2201__iter_v25,
                            idx_kv_cache__rv_v2,
                            [page_begin_inline1053_inline2194__ssa_v8 + key_row_inline1057_inline2257__idx_v8, 0],
                            [physical_block_inline1056_inline2230__ssa_v8, key_row_inline1057_inline2257__idx_v8 * 128],
                            [1, 128],
                            transpose=False,
                        )
                        kv_i8_inline1052_inline2201__rv_v26: pl.Tile[[384, 128], pl.INT8, pl.MemRef(mem_mat_11, pl.const(8192, pl.INT64), 49152), pl.Mem.Mat] = pl.yield_(
                            kv_i8_inline1052_inline2201__tile_9
                        )
                    for key_row_inline1057_inline2257__idx_v9, (kv_i8_inline1052_inline2201__iter_v28,) in pl.range(32, init_values=(kv_i8_inline1052_inline2201__rv_v26,)):
                        kv_i8_inline1052_inline2201__tile_10: pl.Tile[[384, 128], pl.INT8, pl.MemRef(mem_mat_11, pl.const(8192, pl.INT64), 49152), pl.Mem.Mat] = pl.tile.gather_row(
                            kv_i8_inline1052_inline2201__iter_v28,
                            idx_kv_cache__rv_v2,
                            [page_begin_inline1053_inline2194__ssa_v9 + key_row_inline1057_inline2257__idx_v9, 0],
                            [physical_block_inline1056_inline2230__ssa_v9, key_row_inline1057_inline2257__idx_v9 * 128],
                            [1, 128],
                            transpose=False,
                        )
                        kv_i8_inline1052_inline2201__rv_v29: pl.Tile[[384, 128], pl.INT8, pl.MemRef(mem_mat_11, pl.const(8192, pl.INT64), 49152), pl.Mem.Mat] = pl.yield_(
                            kv_i8_inline1052_inline2201__tile_10
                        )
                    for key_row_inline1057_inline2257__idx_v10, (kv_i8_inline1052_inline2201__iter_v31,) in pl.range(32, init_values=(kv_i8_inline1052_inline2201__rv_v29,)):
                        kv_i8_inline1052_inline2201__tile_11: pl.Tile[[384, 128], pl.INT8, pl.MemRef(mem_mat_11, pl.const(8192, pl.INT64), 49152), pl.Mem.Mat] = pl.tile.gather_row(
                            kv_i8_inline1052_inline2201__iter_v31,
                            idx_kv_cache__rv_v2,
                            [page_begin_inline1053_inline2194__ssa_v10 + key_row_inline1057_inline2257__idx_v10, 0],
                            [physical_block_inline1056_inline2230__ssa_v10, key_row_inline1057_inline2257__idx_v10 * 128],
                            [1, 128],
                            transpose=False,
                        )
                        kv_i8_inline1052_inline2201__rv_v32: pl.Tile[[384, 128], pl.INT8, pl.MemRef(mem_mat_11, pl.const(8192, pl.INT64), 49152), pl.Mem.Mat] = pl.yield_(
                            kv_i8_inline1052_inline2201__tile_11
                        )
                    for key_row_inline1057_inline2257__idx_v11, (kv_i8_inline1052_inline2201__iter_v34,) in pl.range(32, init_values=(kv_i8_inline1052_inline2201__rv_v32,)):
                        kv_i8_inline1052_inline2201__tile_12: pl.Tile[[384, 128], pl.INT8, pl.MemRef(mem_mat_11, pl.const(8192, pl.INT64), 49152), pl.Mem.Mat] = pl.tile.gather_row(
                            kv_i8_inline1052_inline2201__iter_v34,
                            idx_kv_cache__rv_v2,
                            [page_begin_inline1053_inline2194__ssa_v11 + key_row_inline1057_inline2257__idx_v11, 0],
                            [physical_block_inline1056_inline2230__ssa_v11, key_row_inline1057_inline2257__idx_v11 * 128],
                            [1, 128],
                            transpose=False,
                        )
                        kv_i8_inline1052_inline2201__rv_v35: pl.Tile[[384, 128], pl.INT8, pl.MemRef(mem_mat_11, pl.const(8192, pl.INT64), 49152), pl.Mem.Mat] = pl.yield_(
                            kv_i8_inline1052_inline2201__tile_12
                        )
                    kv_i8_inline1052_inline2201__rv_v35_t: pl.Tile[
                        [128, 384], pl.INT8, pl.MemRef(mem_mat_11, pl.const(8192, pl.INT64), 49152), pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)
                    ] = pl.tile.transpose_view(kv_i8_inline1052_inline2201__rv_v35)
                    score_i32_inline1028_inline2191__tile_l0_init: pl.Tile[[64, 384], pl.INT32, pl.MemRef(mem_acc_24, pl.const(0, pl.INT64), 98304), pl.Mem.Acc] = pl.tile.create(
                        [64, 384], dtype=pl.INT32, target_memory=pl.Mem.Acc
                    )
                    score_i32_inline1028_inline2191__tile_l0_a: pl.Tile[
                        [64, 64], pl.INT8, pl.MemRef(mem_left_25, pl.const(0, pl.INT64), 4096), pl.Mem.Left, pl.TileView(blayout=pl.TileLayout.row_major)
                    ] = pl.tile.extract(query_vector_inline1024_inline2229__tile, 0, 0, [64, 64], target_memory=pl.Mem.Left)
                    score_i32_inline1028_inline2191__tile_l0_b: pl.Tile[[64, 384], pl.INT8, pl.MemRef(mem_right_26, pl.const(0, pl.INT64), 24576), pl.Mem.Right] = pl.tile.extract(
                        kv_i8_inline1052_inline2201__rv_v35_t, 0, 0, [64, 384], target_memory=pl.Mem.Right
                    )
                    score_i32_inline1028_inline2191__tile_l0_a_1: pl.Tile[
                        [64, 64], pl.INT8, pl.MemRef(mem_left_27, pl.const(4096, pl.INT64), 4096), pl.Mem.Left, pl.TileView(blayout=pl.TileLayout.row_major)
                    ] = pl.tile.extract(query_vector_inline1024_inline2229__tile, 0, 64, [64, 64], target_memory=pl.Mem.Left)
                    score_i32_inline1028_inline2191__tile_l0_b_1: pl.Tile[[64, 384], pl.INT8, pl.MemRef(mem_right_28, pl.const(24576, pl.INT64), 24576), pl.Mem.Right] = pl.tile.extract(
                        kv_i8_inline1052_inline2201__rv_v35_t, 64, 0, [64, 384], target_memory=pl.Mem.Right
                    )
                    score_i32_inline1028_inline2191__tile_l0_c_acc: pl.Tile[[64, 384], pl.INT32, pl.MemRef(mem_acc_24, pl.const(0, pl.INT64), 98304), pl.Mem.Acc] = pl.tile.matmul_acc(
                        score_i32_inline1028_inline2191__tile_l0_init, score_i32_inline1028_inline2191__tile_l0_a, score_i32_inline1028_inline2191__tile_l0_b, True
                    )
                    score_i32_inline1028_inline2191__tile_l0_c_acc_1: pl.Tile[[64, 384], pl.INT32, pl.MemRef(mem_acc_24, pl.const(0, pl.INT64), 98304), pl.Mem.Acc] = pl.tile.matmul_acc(
                        score_i32_inline1028_inline2191__tile_l0_c_acc, score_i32_inline1028_inline2191__tile_l0_a_1, score_i32_inline1028_inline2191__tile_l0_b_1, False
                    )
                    pl.tile.tpush_to_aiv(score_i32_inline1028_inline2191__tile_l0_c_acc_1, split=2)
                    kv_i8_inline1052_inline2201__tile_13: pl.Tile[[384, 128], pl.INT8, pl.MemRef(mem_mat_30, pl.const(57344, pl.INT64), 49152), pl.Mem.Mat] = pl.tile.create(
                        [384, 128], dtype=pl.INT8, target_memory=pl.Mem.Mat, transpose=False
                    )
                    for key_row_inline1057_inline2257__idx_v0_1, (kv_i8_inline1052_inline2201__iter_v1_1,) in pl.range(32, init_values=(kv_i8_inline1052_inline2201__tile_13,)):
                        kv_i8_inline1052_inline2201__tile_14: pl.Tile[[384, 128], pl.INT8, pl.MemRef(mem_mat_30, pl.const(57344, pl.INT64), 49152), pl.Mem.Mat] = pl.tile.gather_row(
                            kv_i8_inline1052_inline2201__iter_v1_1,
                            idx_kv_cache__rv_v2,
                            [page_begin_inline1053_inline2194__ssa_v0_1 + key_row_inline1057_inline2257__idx_v0_1, 0],
                            [physical_block_inline1056_inline2230__ssa_v0_1, key_row_inline1057_inline2257__idx_v0_1 * 128],
                            [1, 128],
                            transpose=False,
                        )
                        kv_i8_inline1052_inline2201__rv_v2_1: pl.Tile[[384, 128], pl.INT8, pl.MemRef(mem_mat_30, pl.const(57344, pl.INT64), 49152), pl.Mem.Mat] = pl.yield_(
                            kv_i8_inline1052_inline2201__tile_14
                        )
                    for key_row_inline1057_inline2257__idx_v1_1, (kv_i8_inline1052_inline2201__iter_v4_1,) in pl.range(32, init_values=(kv_i8_inline1052_inline2201__rv_v2_1,)):
                        kv_i8_inline1052_inline2201__tile_15: pl.Tile[[384, 128], pl.INT8, pl.MemRef(mem_mat_30, pl.const(57344, pl.INT64), 49152), pl.Mem.Mat] = pl.tile.gather_row(
                            kv_i8_inline1052_inline2201__iter_v4_1,
                            idx_kv_cache__rv_v2,
                            [page_begin_inline1053_inline2194__ssa_v1_1 + key_row_inline1057_inline2257__idx_v1_1, 0],
                            [physical_block_inline1056_inline2230__ssa_v1_1, key_row_inline1057_inline2257__idx_v1_1 * 128],
                            [1, 128],
                            transpose=False,
                        )
                        kv_i8_inline1052_inline2201__rv_v5_1: pl.Tile[[384, 128], pl.INT8, pl.MemRef(mem_mat_30, pl.const(57344, pl.INT64), 49152), pl.Mem.Mat] = pl.yield_(
                            kv_i8_inline1052_inline2201__tile_15
                        )
                    for key_row_inline1057_inline2257__idx_v2_1, (kv_i8_inline1052_inline2201__iter_v7_1,) in pl.range(32, init_values=(kv_i8_inline1052_inline2201__rv_v5_1,)):
                        kv_i8_inline1052_inline2201__tile_16: pl.Tile[[384, 128], pl.INT8, pl.MemRef(mem_mat_30, pl.const(57344, pl.INT64), 49152), pl.Mem.Mat] = pl.tile.gather_row(
                            kv_i8_inline1052_inline2201__iter_v7_1,
                            idx_kv_cache__rv_v2,
                            [page_begin_inline1053_inline2194__ssa_v2_1 + key_row_inline1057_inline2257__idx_v2_1, 0],
                            [physical_block_inline1056_inline2230__ssa_v2_1, key_row_inline1057_inline2257__idx_v2_1 * 128],
                            [1, 128],
                            transpose=False,
                        )
                        kv_i8_inline1052_inline2201__rv_v8_1: pl.Tile[[384, 128], pl.INT8, pl.MemRef(mem_mat_30, pl.const(57344, pl.INT64), 49152), pl.Mem.Mat] = pl.yield_(
                            kv_i8_inline1052_inline2201__tile_16
                        )
                    for key_row_inline1057_inline2257__idx_v3_1, (kv_i8_inline1052_inline2201__iter_v10_1,) in pl.range(32, init_values=(kv_i8_inline1052_inline2201__rv_v8_1,)):
                        kv_i8_inline1052_inline2201__tile_17: pl.Tile[[384, 128], pl.INT8, pl.MemRef(mem_mat_30, pl.const(57344, pl.INT64), 49152), pl.Mem.Mat] = pl.tile.gather_row(
                            kv_i8_inline1052_inline2201__iter_v10_1,
                            idx_kv_cache__rv_v2,
                            [page_begin_inline1053_inline2194__ssa_v3_1 + key_row_inline1057_inline2257__idx_v3_1, 0],
                            [physical_block_inline1056_inline2230__ssa_v3_1, key_row_inline1057_inline2257__idx_v3_1 * 128],
                            [1, 128],
                            transpose=False,
                        )
                        kv_i8_inline1052_inline2201__rv_v11_1: pl.Tile[[384, 128], pl.INT8, pl.MemRef(mem_mat_30, pl.const(57344, pl.INT64), 49152), pl.Mem.Mat] = pl.yield_(
                            kv_i8_inline1052_inline2201__tile_17
                        )
                    for key_row_inline1057_inline2257__idx_v4_1, (kv_i8_inline1052_inline2201__iter_v13_1,) in pl.range(32, init_values=(kv_i8_inline1052_inline2201__rv_v11_1,)):
                        kv_i8_inline1052_inline2201__tile_18: pl.Tile[[384, 128], pl.INT8, pl.MemRef(mem_mat_30, pl.const(57344, pl.INT64), 49152), pl.Mem.Mat] = pl.tile.gather_row(
                            kv_i8_inline1052_inline2201__iter_v13_1,
                            idx_kv_cache__rv_v2,
                            [page_begin_inline1053_inline2194__ssa_v4_1 + key_row_inline1057_inline2257__idx_v4_1, 0],
                            [physical_block_inline1056_inline2230__ssa_v4_1, key_row_inline1057_inline2257__idx_v4_1 * 128],
                            [1, 128],
                            transpose=False,
                        )
                        kv_i8_inline1052_inline2201__rv_v14_1: pl.Tile[[384, 128], pl.INT8, pl.MemRef(mem_mat_30, pl.const(57344, pl.INT64), 49152), pl.Mem.Mat] = pl.yield_(
                            kv_i8_inline1052_inline2201__tile_18
                        )
                    for key_row_inline1057_inline2257__idx_v5_1, (kv_i8_inline1052_inline2201__iter_v16_1,) in pl.range(32, init_values=(kv_i8_inline1052_inline2201__rv_v14_1,)):
                        kv_i8_inline1052_inline2201__tile_19: pl.Tile[[384, 128], pl.INT8, pl.MemRef(mem_mat_30, pl.const(57344, pl.INT64), 49152), pl.Mem.Mat] = pl.tile.gather_row(
                            kv_i8_inline1052_inline2201__iter_v16_1,
                            idx_kv_cache__rv_v2,
                            [page_begin_inline1053_inline2194__ssa_v5_1 + key_row_inline1057_inline2257__idx_v5_1, 0],
                            [physical_block_inline1056_inline2230__ssa_v5_1, key_row_inline1057_inline2257__idx_v5_1 * 128],
                            [1, 128],
                            transpose=False,
                        )
                        kv_i8_inline1052_inline2201__rv_v17_1: pl.Tile[[384, 128], pl.INT8, pl.MemRef(mem_mat_30, pl.const(57344, pl.INT64), 49152), pl.Mem.Mat] = pl.yield_(
                            kv_i8_inline1052_inline2201__tile_19
                        )
                    for key_row_inline1057_inline2257__idx_v6_1, (kv_i8_inline1052_inline2201__iter_v19_1,) in pl.range(32, init_values=(kv_i8_inline1052_inline2201__rv_v17_1,)):
                        kv_i8_inline1052_inline2201__tile_20: pl.Tile[[384, 128], pl.INT8, pl.MemRef(mem_mat_30, pl.const(57344, pl.INT64), 49152), pl.Mem.Mat] = pl.tile.gather_row(
                            kv_i8_inline1052_inline2201__iter_v19_1,
                            idx_kv_cache__rv_v2,
                            [page_begin_inline1053_inline2194__ssa_v6_1 + key_row_inline1057_inline2257__idx_v6_1, 0],
                            [physical_block_inline1056_inline2230__ssa_v6_1, key_row_inline1057_inline2257__idx_v6_1 * 128],
                            [1, 128],
                            transpose=False,
                        )
                        kv_i8_inline1052_inline2201__rv_v20_1: pl.Tile[[384, 128], pl.INT8, pl.MemRef(mem_mat_30, pl.const(57344, pl.INT64), 49152), pl.Mem.Mat] = pl.yield_(
                            kv_i8_inline1052_inline2201__tile_20
                        )
                    for key_row_inline1057_inline2257__idx_v7_1, (kv_i8_inline1052_inline2201__iter_v22_1,) in pl.range(32, init_values=(kv_i8_inline1052_inline2201__rv_v20_1,)):
                        kv_i8_inline1052_inline2201__tile_21: pl.Tile[[384, 128], pl.INT8, pl.MemRef(mem_mat_30, pl.const(57344, pl.INT64), 49152), pl.Mem.Mat] = pl.tile.gather_row(
                            kv_i8_inline1052_inline2201__iter_v22_1,
                            idx_kv_cache__rv_v2,
                            [page_begin_inline1053_inline2194__ssa_v7_1 + key_row_inline1057_inline2257__idx_v7_1, 0],
                            [physical_block_inline1056_inline2230__ssa_v7_1, key_row_inline1057_inline2257__idx_v7_1 * 128],
                            [1, 128],
                            transpose=False,
                        )
                        kv_i8_inline1052_inline2201__rv_v23_1: pl.Tile[[384, 128], pl.INT8, pl.MemRef(mem_mat_30, pl.const(57344, pl.INT64), 49152), pl.Mem.Mat] = pl.yield_(
                            kv_i8_inline1052_inline2201__tile_21
                        )
                    for key_row_inline1057_inline2257__idx_v8_1, (kv_i8_inline1052_inline2201__iter_v25_1,) in pl.range(32, init_values=(kv_i8_inline1052_inline2201__rv_v23_1,)):
                        kv_i8_inline1052_inline2201__tile_22: pl.Tile[[384, 128], pl.INT8, pl.MemRef(mem_mat_30, pl.const(57344, pl.INT64), 49152), pl.Mem.Mat] = pl.tile.gather_row(
                            kv_i8_inline1052_inline2201__iter_v25_1,
                            idx_kv_cache__rv_v2,
                            [page_begin_inline1053_inline2194__ssa_v8_1 + key_row_inline1057_inline2257__idx_v8_1, 0],
                            [physical_block_inline1056_inline2230__ssa_v8_1, key_row_inline1057_inline2257__idx_v8_1 * 128],
                            [1, 128],
                            transpose=False,
                        )
                        kv_i8_inline1052_inline2201__rv_v26_1: pl.Tile[[384, 128], pl.INT8, pl.MemRef(mem_mat_30, pl.const(57344, pl.INT64), 49152), pl.Mem.Mat] = pl.yield_(
                            kv_i8_inline1052_inline2201__tile_22
                        )
                    for key_row_inline1057_inline2257__idx_v9_1, (kv_i8_inline1052_inline2201__iter_v28_1,) in pl.range(32, init_values=(kv_i8_inline1052_inline2201__rv_v26_1,)):
                        kv_i8_inline1052_inline2201__tile_23: pl.Tile[[384, 128], pl.INT8, pl.MemRef(mem_mat_30, pl.const(57344, pl.INT64), 49152), pl.Mem.Mat] = pl.tile.gather_row(
                            kv_i8_inline1052_inline2201__iter_v28_1,
                            idx_kv_cache__rv_v2,
                            [page_begin_inline1053_inline2194__ssa_v9_1 + key_row_inline1057_inline2257__idx_v9_1, 0],
                            [physical_block_inline1056_inline2230__ssa_v9_1, key_row_inline1057_inline2257__idx_v9_1 * 128],
                            [1, 128],
                            transpose=False,
                        )
                        kv_i8_inline1052_inline2201__rv_v29_1: pl.Tile[[384, 128], pl.INT8, pl.MemRef(mem_mat_30, pl.const(57344, pl.INT64), 49152), pl.Mem.Mat] = pl.yield_(
                            kv_i8_inline1052_inline2201__tile_23
                        )
                    for key_row_inline1057_inline2257__idx_v10_1, (kv_i8_inline1052_inline2201__iter_v31_1,) in pl.range(32, init_values=(kv_i8_inline1052_inline2201__rv_v29_1,)):
                        kv_i8_inline1052_inline2201__tile_24: pl.Tile[[384, 128], pl.INT8, pl.MemRef(mem_mat_30, pl.const(57344, pl.INT64), 49152), pl.Mem.Mat] = pl.tile.gather_row(
                            kv_i8_inline1052_inline2201__iter_v31_1,
                            idx_kv_cache__rv_v2,
                            [page_begin_inline1053_inline2194__ssa_v10_1 + key_row_inline1057_inline2257__idx_v10_1, 0],
                            [physical_block_inline1056_inline2230__ssa_v10_1, key_row_inline1057_inline2257__idx_v10_1 * 128],
                            [1, 128],
                            transpose=False,
                        )
                        kv_i8_inline1052_inline2201__rv_v32_1: pl.Tile[[384, 128], pl.INT8, pl.MemRef(mem_mat_30, pl.const(57344, pl.INT64), 49152), pl.Mem.Mat] = pl.yield_(
                            kv_i8_inline1052_inline2201__tile_24
                        )
                    for key_row_inline1057_inline2257__idx_v11_1, (kv_i8_inline1052_inline2201__iter_v34_1,) in pl.range(32, init_values=(kv_i8_inline1052_inline2201__rv_v32_1,)):
                        kv_i8_inline1052_inline2201__tile_25: pl.Tile[[384, 128], pl.INT8, pl.MemRef(mem_mat_30, pl.const(57344, pl.INT64), 49152), pl.Mem.Mat] = pl.tile.gather_row(
                            kv_i8_inline1052_inline2201__iter_v34_1,
                            idx_kv_cache__rv_v2,
                            [page_begin_inline1053_inline2194__ssa_v11_1 + key_row_inline1057_inline2257__idx_v11_1, 0],
                            [physical_block_inline1056_inline2230__ssa_v11_1, key_row_inline1057_inline2257__idx_v11_1 * 128],
                            [1, 128],
                            transpose=False,
                        )
                        kv_i8_inline1052_inline2201__rv_v35_1: pl.Tile[[384, 128], pl.INT8, pl.MemRef(mem_mat_30, pl.const(57344, pl.INT64), 49152), pl.Mem.Mat] = pl.yield_(
                            kv_i8_inline1052_inline2201__tile_25
                        )
                    kv_i8_inline1052_inline2201__rv_v35_t_1: pl.Tile[
                        [128, 384], pl.INT8, pl.MemRef(mem_mat_30, pl.const(57344, pl.INT64), 49152), pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)
                    ] = pl.tile.transpose_view(kv_i8_inline1052_inline2201__rv_v35_1)
                    score_i32_inline1028_inline2191__tile_l0_init_1: pl.Tile[[64, 384], pl.INT32, pl.MemRef(mem_acc_24, pl.const(0, pl.INT64), 98304), pl.Mem.Acc] = pl.tile.create(
                        [64, 384], dtype=pl.INT32, target_memory=pl.Mem.Acc
                    )
                    score_i32_inline1028_inline2191__tile_l0_a_2: pl.Tile[
                        [64, 64], pl.INT8, pl.MemRef(mem_left_44, pl.const(8192, pl.INT64), 4096), pl.Mem.Left, pl.TileView(blayout=pl.TileLayout.row_major)
                    ] = pl.tile.extract(query_vector_inline1024_inline2229__tile, 0, 0, [64, 64], target_memory=pl.Mem.Left)
                    score_i32_inline1028_inline2191__tile_l0_b_2: pl.Tile[[64, 384], pl.INT8, pl.MemRef(mem_right_26, pl.const(0, pl.INT64), 24576), pl.Mem.Right] = pl.tile.extract(
                        kv_i8_inline1052_inline2201__rv_v35_t_1, 0, 0, [64, 384], target_memory=pl.Mem.Right
                    )
                    score_i32_inline1028_inline2191__tile_l0_a_3: pl.Tile[
                        [64, 64], pl.INT8, pl.MemRef(mem_left_46, pl.const(12288, pl.INT64), 4096), pl.Mem.Left, pl.TileView(blayout=pl.TileLayout.row_major)
                    ] = pl.tile.extract(query_vector_inline1024_inline2229__tile, 0, 64, [64, 64], target_memory=pl.Mem.Left)
                    score_i32_inline1028_inline2191__tile_l0_b_3: pl.Tile[[64, 384], pl.INT8, pl.MemRef(mem_right_28, pl.const(24576, pl.INT64), 24576), pl.Mem.Right] = pl.tile.extract(
                        kv_i8_inline1052_inline2201__rv_v35_t_1, 64, 0, [64, 384], target_memory=pl.Mem.Right
                    )
                    score_i32_inline1028_inline2191__tile_l0_c_acc_2: pl.Tile[[64, 384], pl.INT32, pl.MemRef(mem_acc_24, pl.const(0, pl.INT64), 98304), pl.Mem.Acc] = pl.tile.matmul_acc(
                        score_i32_inline1028_inline2191__tile_l0_init_1, score_i32_inline1028_inline2191__tile_l0_a_2, score_i32_inline1028_inline2191__tile_l0_b_2, True
                    )
                    score_i32_inline1028_inline2191__tile_l0_c_acc_3: pl.Tile[[64, 384], pl.INT32, pl.MemRef(mem_acc_24, pl.const(0, pl.INT64), 98304), pl.Mem.Acc] = pl.tile.matmul_acc(
                        score_i32_inline1028_inline2191__tile_l0_c_acc_2, score_i32_inline1028_inline2191__tile_l0_a_3, score_i32_inline1028_inline2191__tile_l0_b_3, False
                    )
                    pl.tile.tpush_to_aiv(score_i32_inline1028_inline2191__tile_l0_c_acc_3, split=2)
                    score_arena_inline1043_inline2235__rv_v4_main: pl.Tensor[[384, 8192], pl.FP32, pl.MemRef("mem_ddr_49", pl.const(0, pl.INT64), 12582912)] = pl.yield_(
                        score_arena_inline1043_inline2235__iter_v3
                    )
                unroll_rem: pl.Scalar[pl.INDEX] = (lane_span_inline1067_inline2256__ssa_v0 + 191) // 192 - (lane_span_inline1067_inline2256__ssa_v0 + 191) // 384 * 2
                if unroll_rem == 1:
                    read_begin_inline1055_inline2200__ssa_v0_2: pl.Scalar[pl.INDEX] = unroll_main_end * (single_leaf_inline1032_inline2263__ssa_v0 + 1)
                    kv_i8_inline1052_inline2201__tile_26: pl.Tile[[384, 128], pl.INT8, pl.MemRef(mem_mat_11, pl.const(8192, pl.INT64), 49152), pl.Mem.Mat] = pl.tile.create(
                        [384, 128], dtype=pl.INT8, target_memory=pl.Mem.Mat, transpose=False
                    )
                    page_begin_inline1053_inline2194__ssa_v0_2: pl.Scalar[pl.INDEX] = 0
                    lane_page_inline1041_inline2193__ssa_v0_2: pl.Scalar[pl.INDEX] = page_begin_inline1053_inline2194__ssa_v0_2
                    safe_page_begin_inline1047_inline2192__ssa_v0_2: pl.Scalar[pl.INDEX] = pl.min(
                        read_begin_inline1055_inline2200__ssa_v0_2 + lane_page_inline1041_inline2193__ssa_v0_2, (valid_count_inline1015_inline2253__ssa_v0 - 1) // 32 * 32
                    )
                    logical_page_inline1054_inline2247__ssa_v0_2: pl.Scalar[pl.INDEX] = (logical_begin_inline1045_inline2251__ssa_v0 + safe_page_begin_inline1047_inline2192__ssa_v0_2) // 32
                    t__tile_26: pl.Scalar[pl.INT32] = pl.tensor.read(
                        idx_block_table_flat_inline1044_inline2217__ssa_v0,
                        [batch_idx_inline1058_inline2244__ssa_v0 * table_columns_inline1029_inline2268__ssa_v0 + logical_page_inline1054_inline2247__ssa_v0_2],
                    )
                    physical_block_inline1056_inline2230__ssa_v0_2: pl.Scalar[pl.INDEX] = pl.cast(t__tile_26, pl.INDEX)
                    for key_row_inline1057_inline2257__idx_v0_2, (kv_i8_inline1052_inline2201__iter_v1_2,) in pl.range(32, init_values=(kv_i8_inline1052_inline2201__tile_26,)):
                        kv_i8_inline1052_inline2201__tile_27: pl.Tile[[384, 128], pl.INT8, pl.MemRef(mem_mat_11, pl.const(8192, pl.INT64), 49152), pl.Mem.Mat] = pl.tile.gather_row(
                            kv_i8_inline1052_inline2201__iter_v1_2,
                            idx_kv_cache__rv_v2,
                            [page_begin_inline1053_inline2194__ssa_v0_2 + key_row_inline1057_inline2257__idx_v0_2, 0],
                            [physical_block_inline1056_inline2230__ssa_v0_2, key_row_inline1057_inline2257__idx_v0_2 * 128],
                            [1, 128],
                            transpose=False,
                        )
                        kv_i8_inline1052_inline2201__rv_v2_2: pl.Tile[[384, 128], pl.INT8, pl.MemRef(mem_mat_11, pl.const(8192, pl.INT64), 49152), pl.Mem.Mat] = pl.yield_(
                            kv_i8_inline1052_inline2201__tile_27
                        )
                    page_begin_inline1053_inline2194__ssa_v1_2: pl.Scalar[pl.INDEX] = 32
                    lane_page_inline1041_inline2193__ssa_v1_2: pl.Scalar[pl.INDEX] = page_begin_inline1053_inline2194__ssa_v1_2
                    safe_page_begin_inline1047_inline2192__ssa_v1_2: pl.Scalar[pl.INDEX] = pl.min(
                        read_begin_inline1055_inline2200__ssa_v0_2 + lane_page_inline1041_inline2193__ssa_v1_2, (valid_count_inline1015_inline2253__ssa_v0 - 1) // 32 * 32
                    )
                    logical_page_inline1054_inline2247__ssa_v1_2: pl.Scalar[pl.INDEX] = (logical_begin_inline1045_inline2251__ssa_v0 + safe_page_begin_inline1047_inline2192__ssa_v1_2) // 32
                    t__tile_27: pl.Scalar[pl.INT32] = pl.tensor.read(
                        idx_block_table_flat_inline1044_inline2217__ssa_v0,
                        [batch_idx_inline1058_inline2244__ssa_v0 * table_columns_inline1029_inline2268__ssa_v0 + logical_page_inline1054_inline2247__ssa_v1_2],
                    )
                    physical_block_inline1056_inline2230__ssa_v1_2: pl.Scalar[pl.INDEX] = pl.cast(t__tile_27, pl.INDEX)
                    for key_row_inline1057_inline2257__idx_v1_2, (kv_i8_inline1052_inline2201__iter_v4_2,) in pl.range(32, init_values=(kv_i8_inline1052_inline2201__rv_v2_2,)):
                        kv_i8_inline1052_inline2201__tile_28: pl.Tile[[384, 128], pl.INT8, pl.MemRef(mem_mat_11, pl.const(8192, pl.INT64), 49152), pl.Mem.Mat] = pl.tile.gather_row(
                            kv_i8_inline1052_inline2201__iter_v4_2,
                            idx_kv_cache__rv_v2,
                            [page_begin_inline1053_inline2194__ssa_v1_2 + key_row_inline1057_inline2257__idx_v1_2, 0],
                            [physical_block_inline1056_inline2230__ssa_v1_2, key_row_inline1057_inline2257__idx_v1_2 * 128],
                            [1, 128],
                            transpose=False,
                        )
                        kv_i8_inline1052_inline2201__rv_v5_2: pl.Tile[[384, 128], pl.INT8, pl.MemRef(mem_mat_11, pl.const(8192, pl.INT64), 49152), pl.Mem.Mat] = pl.yield_(
                            kv_i8_inline1052_inline2201__tile_28
                        )
                    page_begin_inline1053_inline2194__ssa_v2_2: pl.Scalar[pl.INDEX] = 64
                    lane_page_inline1041_inline2193__ssa_v2_2: pl.Scalar[pl.INDEX] = page_begin_inline1053_inline2194__ssa_v2_2
                    safe_page_begin_inline1047_inline2192__ssa_v2_2: pl.Scalar[pl.INDEX] = pl.min(
                        read_begin_inline1055_inline2200__ssa_v0_2 + lane_page_inline1041_inline2193__ssa_v2_2, (valid_count_inline1015_inline2253__ssa_v0 - 1) // 32 * 32
                    )
                    logical_page_inline1054_inline2247__ssa_v2_2: pl.Scalar[pl.INDEX] = (logical_begin_inline1045_inline2251__ssa_v0 + safe_page_begin_inline1047_inline2192__ssa_v2_2) // 32
                    t__tile_28: pl.Scalar[pl.INT32] = pl.tensor.read(
                        idx_block_table_flat_inline1044_inline2217__ssa_v0,
                        [batch_idx_inline1058_inline2244__ssa_v0 * table_columns_inline1029_inline2268__ssa_v0 + logical_page_inline1054_inline2247__ssa_v2_2],
                    )
                    physical_block_inline1056_inline2230__ssa_v2_2: pl.Scalar[pl.INDEX] = pl.cast(t__tile_28, pl.INDEX)
                    for key_row_inline1057_inline2257__idx_v2_2, (kv_i8_inline1052_inline2201__iter_v7_2,) in pl.range(32, init_values=(kv_i8_inline1052_inline2201__rv_v5_2,)):
                        kv_i8_inline1052_inline2201__tile_29: pl.Tile[[384, 128], pl.INT8, pl.MemRef(mem_mat_11, pl.const(8192, pl.INT64), 49152), pl.Mem.Mat] = pl.tile.gather_row(
                            kv_i8_inline1052_inline2201__iter_v7_2,
                            idx_kv_cache__rv_v2,
                            [page_begin_inline1053_inline2194__ssa_v2_2 + key_row_inline1057_inline2257__idx_v2_2, 0],
                            [physical_block_inline1056_inline2230__ssa_v2_2, key_row_inline1057_inline2257__idx_v2_2 * 128],
                            [1, 128],
                            transpose=False,
                        )
                        kv_i8_inline1052_inline2201__rv_v8_2: pl.Tile[[384, 128], pl.INT8, pl.MemRef(mem_mat_11, pl.const(8192, pl.INT64), 49152), pl.Mem.Mat] = pl.yield_(
                            kv_i8_inline1052_inline2201__tile_29
                        )
                    page_begin_inline1053_inline2194__ssa_v3_2: pl.Scalar[pl.INDEX] = 96
                    lane_page_inline1041_inline2193__ssa_v3_2: pl.Scalar[pl.INDEX] = page_begin_inline1053_inline2194__ssa_v3_2
                    safe_page_begin_inline1047_inline2192__ssa_v3_2: pl.Scalar[pl.INDEX] = pl.min(
                        read_begin_inline1055_inline2200__ssa_v0_2 + lane_page_inline1041_inline2193__ssa_v3_2, (valid_count_inline1015_inline2253__ssa_v0 - 1) // 32 * 32
                    )
                    logical_page_inline1054_inline2247__ssa_v3_2: pl.Scalar[pl.INDEX] = (logical_begin_inline1045_inline2251__ssa_v0 + safe_page_begin_inline1047_inline2192__ssa_v3_2) // 32
                    t__tile_29: pl.Scalar[pl.INT32] = pl.tensor.read(
                        idx_block_table_flat_inline1044_inline2217__ssa_v0,
                        [batch_idx_inline1058_inline2244__ssa_v0 * table_columns_inline1029_inline2268__ssa_v0 + logical_page_inline1054_inline2247__ssa_v3_2],
                    )
                    physical_block_inline1056_inline2230__ssa_v3_2: pl.Scalar[pl.INDEX] = pl.cast(t__tile_29, pl.INDEX)
                    for key_row_inline1057_inline2257__idx_v3_2, (kv_i8_inline1052_inline2201__iter_v10_2,) in pl.range(32, init_values=(kv_i8_inline1052_inline2201__rv_v8_2,)):
                        kv_i8_inline1052_inline2201__tile_30: pl.Tile[[384, 128], pl.INT8, pl.MemRef(mem_mat_11, pl.const(8192, pl.INT64), 49152), pl.Mem.Mat] = pl.tile.gather_row(
                            kv_i8_inline1052_inline2201__iter_v10_2,
                            idx_kv_cache__rv_v2,
                            [page_begin_inline1053_inline2194__ssa_v3_2 + key_row_inline1057_inline2257__idx_v3_2, 0],
                            [physical_block_inline1056_inline2230__ssa_v3_2, key_row_inline1057_inline2257__idx_v3_2 * 128],
                            [1, 128],
                            transpose=False,
                        )
                        kv_i8_inline1052_inline2201__rv_v11_2: pl.Tile[[384, 128], pl.INT8, pl.MemRef(mem_mat_11, pl.const(8192, pl.INT64), 49152), pl.Mem.Mat] = pl.yield_(
                            kv_i8_inline1052_inline2201__tile_30
                        )
                    page_begin_inline1053_inline2194__ssa_v4_2: pl.Scalar[pl.INDEX] = 128
                    lane_page_inline1041_inline2193__ssa_v4_2: pl.Scalar[pl.INDEX] = page_begin_inline1053_inline2194__ssa_v4_2
                    safe_page_begin_inline1047_inline2192__ssa_v4_2: pl.Scalar[pl.INDEX] = pl.min(
                        read_begin_inline1055_inline2200__ssa_v0_2 + lane_page_inline1041_inline2193__ssa_v4_2, (valid_count_inline1015_inline2253__ssa_v0 - 1) // 32 * 32
                    )
                    logical_page_inline1054_inline2247__ssa_v4_2: pl.Scalar[pl.INDEX] = (logical_begin_inline1045_inline2251__ssa_v0 + safe_page_begin_inline1047_inline2192__ssa_v4_2) // 32
                    t__tile_30: pl.Scalar[pl.INT32] = pl.tensor.read(
                        idx_block_table_flat_inline1044_inline2217__ssa_v0,
                        [batch_idx_inline1058_inline2244__ssa_v0 * table_columns_inline1029_inline2268__ssa_v0 + logical_page_inline1054_inline2247__ssa_v4_2],
                    )
                    physical_block_inline1056_inline2230__ssa_v4_2: pl.Scalar[pl.INDEX] = pl.cast(t__tile_30, pl.INDEX)
                    for key_row_inline1057_inline2257__idx_v4_2, (kv_i8_inline1052_inline2201__iter_v13_2,) in pl.range(32, init_values=(kv_i8_inline1052_inline2201__rv_v11_2,)):
                        kv_i8_inline1052_inline2201__tile_31: pl.Tile[[384, 128], pl.INT8, pl.MemRef(mem_mat_11, pl.const(8192, pl.INT64), 49152), pl.Mem.Mat] = pl.tile.gather_row(
                            kv_i8_inline1052_inline2201__iter_v13_2,
                            idx_kv_cache__rv_v2,
                            [page_begin_inline1053_inline2194__ssa_v4_2 + key_row_inline1057_inline2257__idx_v4_2, 0],
                            [physical_block_inline1056_inline2230__ssa_v4_2, key_row_inline1057_inline2257__idx_v4_2 * 128],
                            [1, 128],
                            transpose=False,
                        )
                        kv_i8_inline1052_inline2201__rv_v14_2: pl.Tile[[384, 128], pl.INT8, pl.MemRef(mem_mat_11, pl.const(8192, pl.INT64), 49152), pl.Mem.Mat] = pl.yield_(
                            kv_i8_inline1052_inline2201__tile_31
                        )
                    page_begin_inline1053_inline2194__ssa_v5_2: pl.Scalar[pl.INDEX] = 160
                    lane_page_inline1041_inline2193__ssa_v5_2: pl.Scalar[pl.INDEX] = page_begin_inline1053_inline2194__ssa_v5_2
                    safe_page_begin_inline1047_inline2192__ssa_v5_2: pl.Scalar[pl.INDEX] = pl.min(
                        read_begin_inline1055_inline2200__ssa_v0_2 + lane_page_inline1041_inline2193__ssa_v5_2, (valid_count_inline1015_inline2253__ssa_v0 - 1) // 32 * 32
                    )
                    logical_page_inline1054_inline2247__ssa_v5_2: pl.Scalar[pl.INDEX] = (logical_begin_inline1045_inline2251__ssa_v0 + safe_page_begin_inline1047_inline2192__ssa_v5_2) // 32
                    t__tile_31: pl.Scalar[pl.INT32] = pl.tensor.read(
                        idx_block_table_flat_inline1044_inline2217__ssa_v0,
                        [batch_idx_inline1058_inline2244__ssa_v0 * table_columns_inline1029_inline2268__ssa_v0 + logical_page_inline1054_inline2247__ssa_v5_2],
                    )
                    physical_block_inline1056_inline2230__ssa_v5_2: pl.Scalar[pl.INDEX] = pl.cast(t__tile_31, pl.INDEX)
                    for key_row_inline1057_inline2257__idx_v5_2, (kv_i8_inline1052_inline2201__iter_v16_2,) in pl.range(32, init_values=(kv_i8_inline1052_inline2201__rv_v14_2,)):
                        kv_i8_inline1052_inline2201__tile_32: pl.Tile[[384, 128], pl.INT8, pl.MemRef(mem_mat_11, pl.const(8192, pl.INT64), 49152), pl.Mem.Mat] = pl.tile.gather_row(
                            kv_i8_inline1052_inline2201__iter_v16_2,
                            idx_kv_cache__rv_v2,
                            [page_begin_inline1053_inline2194__ssa_v5_2 + key_row_inline1057_inline2257__idx_v5_2, 0],
                            [physical_block_inline1056_inline2230__ssa_v5_2, key_row_inline1057_inline2257__idx_v5_2 * 128],
                            [1, 128],
                            transpose=False,
                        )
                        kv_i8_inline1052_inline2201__rv_v17_2: pl.Tile[[384, 128], pl.INT8, pl.MemRef(mem_mat_11, pl.const(8192, pl.INT64), 49152), pl.Mem.Mat] = pl.yield_(
                            kv_i8_inline1052_inline2201__tile_32
                        )
                    page_begin_inline1053_inline2194__ssa_v6_2: pl.Scalar[pl.INDEX] = 192
                    lane_page_inline1041_inline2193__ssa_v6_2: pl.Scalar[pl.INDEX] = (
                        page_begin_inline1053_inline2194__ssa_v6_2 // 192 * lane_stride_inline1014_inline2207__ssa_v0 + page_begin_inline1053_inline2194__ssa_v6_2 % 192
                    )
                    safe_page_begin_inline1047_inline2192__ssa_v6_2: pl.Scalar[pl.INDEX] = pl.min(
                        read_begin_inline1055_inline2200__ssa_v0_2 + lane_page_inline1041_inline2193__ssa_v6_2, (valid_count_inline1015_inline2253__ssa_v0 - 1) // 32 * 32
                    )
                    logical_page_inline1054_inline2247__ssa_v6_2: pl.Scalar[pl.INDEX] = (logical_begin_inline1045_inline2251__ssa_v0 + safe_page_begin_inline1047_inline2192__ssa_v6_2) // 32
                    t__tile_32: pl.Scalar[pl.INT32] = pl.tensor.read(
                        idx_block_table_flat_inline1044_inline2217__ssa_v0,
                        [batch_idx_inline1058_inline2244__ssa_v0 * table_columns_inline1029_inline2268__ssa_v0 + logical_page_inline1054_inline2247__ssa_v6_2],
                    )
                    physical_block_inline1056_inline2230__ssa_v6_2: pl.Scalar[pl.INDEX] = pl.cast(t__tile_32, pl.INDEX)
                    for key_row_inline1057_inline2257__idx_v6_2, (kv_i8_inline1052_inline2201__iter_v19_2,) in pl.range(32, init_values=(kv_i8_inline1052_inline2201__rv_v17_2,)):
                        kv_i8_inline1052_inline2201__tile_33: pl.Tile[[384, 128], pl.INT8, pl.MemRef(mem_mat_11, pl.const(8192, pl.INT64), 49152), pl.Mem.Mat] = pl.tile.gather_row(
                            kv_i8_inline1052_inline2201__iter_v19_2,
                            idx_kv_cache__rv_v2,
                            [page_begin_inline1053_inline2194__ssa_v6_2 + key_row_inline1057_inline2257__idx_v6_2, 0],
                            [physical_block_inline1056_inline2230__ssa_v6_2, key_row_inline1057_inline2257__idx_v6_2 * 128],
                            [1, 128],
                            transpose=False,
                        )
                        kv_i8_inline1052_inline2201__rv_v20_2: pl.Tile[[384, 128], pl.INT8, pl.MemRef(mem_mat_11, pl.const(8192, pl.INT64), 49152), pl.Mem.Mat] = pl.yield_(
                            kv_i8_inline1052_inline2201__tile_33
                        )
                    page_begin_inline1053_inline2194__ssa_v7_2: pl.Scalar[pl.INDEX] = 224
                    lane_page_inline1041_inline2193__ssa_v7_2: pl.Scalar[pl.INDEX] = (
                        page_begin_inline1053_inline2194__ssa_v7_2 // 192 * lane_stride_inline1014_inline2207__ssa_v0 + page_begin_inline1053_inline2194__ssa_v7_2 % 192
                    )
                    safe_page_begin_inline1047_inline2192__ssa_v7_2: pl.Scalar[pl.INDEX] = pl.min(
                        read_begin_inline1055_inline2200__ssa_v0_2 + lane_page_inline1041_inline2193__ssa_v7_2, (valid_count_inline1015_inline2253__ssa_v0 - 1) // 32 * 32
                    )
                    logical_page_inline1054_inline2247__ssa_v7_2: pl.Scalar[pl.INDEX] = (logical_begin_inline1045_inline2251__ssa_v0 + safe_page_begin_inline1047_inline2192__ssa_v7_2) // 32
                    t__tile_33: pl.Scalar[pl.INT32] = pl.tensor.read(
                        idx_block_table_flat_inline1044_inline2217__ssa_v0,
                        [batch_idx_inline1058_inline2244__ssa_v0 * table_columns_inline1029_inline2268__ssa_v0 + logical_page_inline1054_inline2247__ssa_v7_2],
                    )
                    physical_block_inline1056_inline2230__ssa_v7_2: pl.Scalar[pl.INDEX] = pl.cast(t__tile_33, pl.INDEX)
                    for key_row_inline1057_inline2257__idx_v7_2, (kv_i8_inline1052_inline2201__iter_v22_2,) in pl.range(32, init_values=(kv_i8_inline1052_inline2201__rv_v20_2,)):
                        kv_i8_inline1052_inline2201__tile_34: pl.Tile[[384, 128], pl.INT8, pl.MemRef(mem_mat_11, pl.const(8192, pl.INT64), 49152), pl.Mem.Mat] = pl.tile.gather_row(
                            kv_i8_inline1052_inline2201__iter_v22_2,
                            idx_kv_cache__rv_v2,
                            [page_begin_inline1053_inline2194__ssa_v7_2 + key_row_inline1057_inline2257__idx_v7_2, 0],
                            [physical_block_inline1056_inline2230__ssa_v7_2, key_row_inline1057_inline2257__idx_v7_2 * 128],
                            [1, 128],
                            transpose=False,
                        )
                        kv_i8_inline1052_inline2201__rv_v23_2: pl.Tile[[384, 128], pl.INT8, pl.MemRef(mem_mat_11, pl.const(8192, pl.INT64), 49152), pl.Mem.Mat] = pl.yield_(
                            kv_i8_inline1052_inline2201__tile_34
                        )
                    page_begin_inline1053_inline2194__ssa_v8_2: pl.Scalar[pl.INDEX] = 256
                    lane_page_inline1041_inline2193__ssa_v8_2: pl.Scalar[pl.INDEX] = (
                        page_begin_inline1053_inline2194__ssa_v8_2 // 192 * lane_stride_inline1014_inline2207__ssa_v0 + page_begin_inline1053_inline2194__ssa_v8_2 % 192
                    )
                    safe_page_begin_inline1047_inline2192__ssa_v8_2: pl.Scalar[pl.INDEX] = pl.min(
                        read_begin_inline1055_inline2200__ssa_v0_2 + lane_page_inline1041_inline2193__ssa_v8_2, (valid_count_inline1015_inline2253__ssa_v0 - 1) // 32 * 32
                    )
                    logical_page_inline1054_inline2247__ssa_v8_2: pl.Scalar[pl.INDEX] = (logical_begin_inline1045_inline2251__ssa_v0 + safe_page_begin_inline1047_inline2192__ssa_v8_2) // 32
                    t__tile_34: pl.Scalar[pl.INT32] = pl.tensor.read(
                        idx_block_table_flat_inline1044_inline2217__ssa_v0,
                        [batch_idx_inline1058_inline2244__ssa_v0 * table_columns_inline1029_inline2268__ssa_v0 + logical_page_inline1054_inline2247__ssa_v8_2],
                    )
                    physical_block_inline1056_inline2230__ssa_v8_2: pl.Scalar[pl.INDEX] = pl.cast(t__tile_34, pl.INDEX)
                    for key_row_inline1057_inline2257__idx_v8_2, (kv_i8_inline1052_inline2201__iter_v25_2,) in pl.range(32, init_values=(kv_i8_inline1052_inline2201__rv_v23_2,)):
                        kv_i8_inline1052_inline2201__tile_35: pl.Tile[[384, 128], pl.INT8, pl.MemRef(mem_mat_11, pl.const(8192, pl.INT64), 49152), pl.Mem.Mat] = pl.tile.gather_row(
                            kv_i8_inline1052_inline2201__iter_v25_2,
                            idx_kv_cache__rv_v2,
                            [page_begin_inline1053_inline2194__ssa_v8_2 + key_row_inline1057_inline2257__idx_v8_2, 0],
                            [physical_block_inline1056_inline2230__ssa_v8_2, key_row_inline1057_inline2257__idx_v8_2 * 128],
                            [1, 128],
                            transpose=False,
                        )
                        kv_i8_inline1052_inline2201__rv_v26_2: pl.Tile[[384, 128], pl.INT8, pl.MemRef(mem_mat_11, pl.const(8192, pl.INT64), 49152), pl.Mem.Mat] = pl.yield_(
                            kv_i8_inline1052_inline2201__tile_35
                        )
                    page_begin_inline1053_inline2194__ssa_v9_2: pl.Scalar[pl.INDEX] = 288
                    lane_page_inline1041_inline2193__ssa_v9_2: pl.Scalar[pl.INDEX] = (
                        page_begin_inline1053_inline2194__ssa_v9_2 // 192 * lane_stride_inline1014_inline2207__ssa_v0 + page_begin_inline1053_inline2194__ssa_v9_2 % 192
                    )
                    safe_page_begin_inline1047_inline2192__ssa_v9_2: pl.Scalar[pl.INDEX] = pl.min(
                        read_begin_inline1055_inline2200__ssa_v0_2 + lane_page_inline1041_inline2193__ssa_v9_2, (valid_count_inline1015_inline2253__ssa_v0 - 1) // 32 * 32
                    )
                    logical_page_inline1054_inline2247__ssa_v9_2: pl.Scalar[pl.INDEX] = (logical_begin_inline1045_inline2251__ssa_v0 + safe_page_begin_inline1047_inline2192__ssa_v9_2) // 32
                    t__tile_35: pl.Scalar[pl.INT32] = pl.tensor.read(
                        idx_block_table_flat_inline1044_inline2217__ssa_v0,
                        [batch_idx_inline1058_inline2244__ssa_v0 * table_columns_inline1029_inline2268__ssa_v0 + logical_page_inline1054_inline2247__ssa_v9_2],
                    )
                    physical_block_inline1056_inline2230__ssa_v9_2: pl.Scalar[pl.INDEX] = pl.cast(t__tile_35, pl.INDEX)
                    for key_row_inline1057_inline2257__idx_v9_2, (kv_i8_inline1052_inline2201__iter_v28_2,) in pl.range(32, init_values=(kv_i8_inline1052_inline2201__rv_v26_2,)):
                        kv_i8_inline1052_inline2201__tile_36: pl.Tile[[384, 128], pl.INT8, pl.MemRef(mem_mat_11, pl.const(8192, pl.INT64), 49152), pl.Mem.Mat] = pl.tile.gather_row(
                            kv_i8_inline1052_inline2201__iter_v28_2,
                            idx_kv_cache__rv_v2,
                            [page_begin_inline1053_inline2194__ssa_v9_2 + key_row_inline1057_inline2257__idx_v9_2, 0],
                            [physical_block_inline1056_inline2230__ssa_v9_2, key_row_inline1057_inline2257__idx_v9_2 * 128],
                            [1, 128],
                            transpose=False,
                        )
                        kv_i8_inline1052_inline2201__rv_v29_2: pl.Tile[[384, 128], pl.INT8, pl.MemRef(mem_mat_11, pl.const(8192, pl.INT64), 49152), pl.Mem.Mat] = pl.yield_(
                            kv_i8_inline1052_inline2201__tile_36
                        )
                    page_begin_inline1053_inline2194__ssa_v10_2: pl.Scalar[pl.INDEX] = 320
                    lane_page_inline1041_inline2193__ssa_v10_2: pl.Scalar[pl.INDEX] = (
                        page_begin_inline1053_inline2194__ssa_v10_2 // 192 * lane_stride_inline1014_inline2207__ssa_v0 + page_begin_inline1053_inline2194__ssa_v10_2 % 192
                    )
                    safe_page_begin_inline1047_inline2192__ssa_v10_2: pl.Scalar[pl.INDEX] = pl.min(
                        read_begin_inline1055_inline2200__ssa_v0_2 + lane_page_inline1041_inline2193__ssa_v10_2, (valid_count_inline1015_inline2253__ssa_v0 - 1) // 32 * 32
                    )
                    logical_page_inline1054_inline2247__ssa_v10_2: pl.Scalar[pl.INDEX] = (logical_begin_inline1045_inline2251__ssa_v0 + safe_page_begin_inline1047_inline2192__ssa_v10_2) // 32
                    t__tile_36: pl.Scalar[pl.INT32] = pl.tensor.read(
                        idx_block_table_flat_inline1044_inline2217__ssa_v0,
                        [batch_idx_inline1058_inline2244__ssa_v0 * table_columns_inline1029_inline2268__ssa_v0 + logical_page_inline1054_inline2247__ssa_v10_2],
                    )
                    physical_block_inline1056_inline2230__ssa_v10_2: pl.Scalar[pl.INDEX] = pl.cast(t__tile_36, pl.INDEX)
                    for key_row_inline1057_inline2257__idx_v10_2, (kv_i8_inline1052_inline2201__iter_v31_2,) in pl.range(32, init_values=(kv_i8_inline1052_inline2201__rv_v29_2,)):
                        kv_i8_inline1052_inline2201__tile_37: pl.Tile[[384, 128], pl.INT8, pl.MemRef(mem_mat_11, pl.const(8192, pl.INT64), 49152), pl.Mem.Mat] = pl.tile.gather_row(
                            kv_i8_inline1052_inline2201__iter_v31_2,
                            idx_kv_cache__rv_v2,
                            [page_begin_inline1053_inline2194__ssa_v10_2 + key_row_inline1057_inline2257__idx_v10_2, 0],
                            [physical_block_inline1056_inline2230__ssa_v10_2, key_row_inline1057_inline2257__idx_v10_2 * 128],
                            [1, 128],
                            transpose=False,
                        )
                        kv_i8_inline1052_inline2201__rv_v32_2: pl.Tile[[384, 128], pl.INT8, pl.MemRef(mem_mat_11, pl.const(8192, pl.INT64), 49152), pl.Mem.Mat] = pl.yield_(
                            kv_i8_inline1052_inline2201__tile_37
                        )
                    page_begin_inline1053_inline2194__ssa_v11_2: pl.Scalar[pl.INDEX] = 352
                    lane_page_inline1041_inline2193__ssa_v11_2: pl.Scalar[pl.INDEX] = (
                        page_begin_inline1053_inline2194__ssa_v11_2 // 192 * lane_stride_inline1014_inline2207__ssa_v0 + page_begin_inline1053_inline2194__ssa_v11_2 % 192
                    )
                    safe_page_begin_inline1047_inline2192__ssa_v11_2: pl.Scalar[pl.INDEX] = pl.min(
                        read_begin_inline1055_inline2200__ssa_v0_2 + lane_page_inline1041_inline2193__ssa_v11_2, (valid_count_inline1015_inline2253__ssa_v0 - 1) // 32 * 32
                    )
                    logical_page_inline1054_inline2247__ssa_v11_2: pl.Scalar[pl.INDEX] = (logical_begin_inline1045_inline2251__ssa_v0 + safe_page_begin_inline1047_inline2192__ssa_v11_2) // 32
                    t__tile_37: pl.Scalar[pl.INT32] = pl.tensor.read(
                        idx_block_table_flat_inline1044_inline2217__ssa_v0,
                        [batch_idx_inline1058_inline2244__ssa_v0 * table_columns_inline1029_inline2268__ssa_v0 + logical_page_inline1054_inline2247__ssa_v11_2],
                    )
                    physical_block_inline1056_inline2230__ssa_v11_2: pl.Scalar[pl.INDEX] = pl.cast(t__tile_37, pl.INDEX)
                    for key_row_inline1057_inline2257__idx_v11_2, (kv_i8_inline1052_inline2201__iter_v34_2,) in pl.range(32, init_values=(kv_i8_inline1052_inline2201__rv_v32_2,)):
                        kv_i8_inline1052_inline2201__tile_38: pl.Tile[[384, 128], pl.INT8, pl.MemRef(mem_mat_11, pl.const(8192, pl.INT64), 49152), pl.Mem.Mat] = pl.tile.gather_row(
                            kv_i8_inline1052_inline2201__iter_v34_2,
                            idx_kv_cache__rv_v2,
                            [page_begin_inline1053_inline2194__ssa_v11_2 + key_row_inline1057_inline2257__idx_v11_2, 0],
                            [physical_block_inline1056_inline2230__ssa_v11_2, key_row_inline1057_inline2257__idx_v11_2 * 128],
                            [1, 128],
                            transpose=False,
                        )
                        kv_i8_inline1052_inline2201__rv_v35_2: pl.Tile[[384, 128], pl.INT8, pl.MemRef(mem_mat_11, pl.const(8192, pl.INT64), 49152), pl.Mem.Mat] = pl.yield_(
                            kv_i8_inline1052_inline2201__tile_38
                        )
                    kv_i8_inline1052_inline2201__rv_v35_t_2: pl.Tile[
                        [128, 384], pl.INT8, pl.MemRef(mem_mat_11, pl.const(8192, pl.INT64), 49152), pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)
                    ] = pl.tile.transpose_view(kv_i8_inline1052_inline2201__rv_v35_2)
                    score_i32_inline1028_inline2191__tile_l0_init_2: pl.Tile[[64, 384], pl.INT32, pl.MemRef(mem_acc_24, pl.const(0, pl.INT64), 98304), pl.Mem.Acc] = pl.tile.create(
                        [64, 384], dtype=pl.INT32, target_memory=pl.Mem.Acc
                    )
                    score_i32_inline1028_inline2191__tile_l0_a_4: pl.Tile[
                        [64, 64], pl.INT8, pl.MemRef(mem_left_25, pl.const(0, pl.INT64), 4096), pl.Mem.Left, pl.TileView(blayout=pl.TileLayout.row_major)
                    ] = pl.tile.extract(query_vector_inline1024_inline2229__tile, 0, 0, [64, 64], target_memory=pl.Mem.Left)
                    score_i32_inline1028_inline2191__tile_l0_b_4: pl.Tile[[64, 384], pl.INT8, pl.MemRef(mem_right_26, pl.const(0, pl.INT64), 24576), pl.Mem.Right] = pl.tile.extract(
                        kv_i8_inline1052_inline2201__rv_v35_t_2, 0, 0, [64, 384], target_memory=pl.Mem.Right
                    )
                    score_i32_inline1028_inline2191__tile_l0_a_5: pl.Tile[
                        [64, 64], pl.INT8, pl.MemRef(mem_left_27, pl.const(4096, pl.INT64), 4096), pl.Mem.Left, pl.TileView(blayout=pl.TileLayout.row_major)
                    ] = pl.tile.extract(query_vector_inline1024_inline2229__tile, 0, 64, [64, 64], target_memory=pl.Mem.Left)
                    score_i32_inline1028_inline2191__tile_l0_b_5: pl.Tile[[64, 384], pl.INT8, pl.MemRef(mem_right_28, pl.const(24576, pl.INT64), 24576), pl.Mem.Right] = pl.tile.extract(
                        kv_i8_inline1052_inline2201__rv_v35_t_2, 64, 0, [64, 384], target_memory=pl.Mem.Right
                    )
                    score_i32_inline1028_inline2191__tile_l0_c_acc_4: pl.Tile[[64, 384], pl.INT32, pl.MemRef(mem_acc_24, pl.const(0, pl.INT64), 98304), pl.Mem.Acc] = pl.tile.matmul_acc(
                        score_i32_inline1028_inline2191__tile_l0_init_2, score_i32_inline1028_inline2191__tile_l0_a_4, score_i32_inline1028_inline2191__tile_l0_b_4, True
                    )
                    score_i32_inline1028_inline2191__tile_l0_c_acc_5: pl.Tile[[64, 384], pl.INT32, pl.MemRef(mem_acc_24, pl.const(0, pl.INT64), 98304), pl.Mem.Acc] = pl.tile.matmul_acc(
                        score_i32_inline1028_inline2191__tile_l0_c_acc_4, score_i32_inline1028_inline2191__tile_l0_a_5, score_i32_inline1028_inline2191__tile_l0_b_5, False
                    )
                    pl.tile.tpush_to_aiv(score_i32_inline1028_inline2191__tile_l0_c_acc_5, split=2)
                    score_arena_inline1043_inline2235__rv_v4: pl.Tensor[[384, 8192], pl.FP32, pl.MemRef("mem_ddr_69", pl.const(0, pl.INT64), 12582912)] = pl.yield_(
                        score_arena_inline1043_inline2235__rv_v4_main
                    )
                else:
                    score_arena_inline1043_inline2235__rv_v4: pl.Tensor[[384, 8192], pl.FP32, pl.MemRef("mem_ddr_69", pl.const(0, pl.INT64), 12582912)] = pl.yield_(
                        score_arena_inline1043_inline2235__rv_v4_main
                    )
                score_arena_inline1043_inline2235__phi_v7: pl.Tensor[[384, 8192], pl.FP32, pl.MemRef("mem_ddr_70", pl.const(0, pl.INT64), 12582912)] = pl.yield_(
                    score_arena_inline1043_inline2235__rv_v4
                )
            else:
                score_arena_inline1043_inline2235__phi_v7: pl.Tensor[[384, 8192], pl.FP32, pl.MemRef("mem_ddr_70", pl.const(0, pl.INT64), 12582912)] = pl.yield_(
                    score_arena_inline1043_inline2235__iter_v1
                )
            score_arena_inline1043_inline2235__rv_v2: pl.Tensor[[384, 8192], pl.FP32, pl.MemRef("mem_ddr_71", pl.const(0, pl.INT64), 12582912)] = pl.yield_(score_arena_inline1043_inline2235__phi_v7)

    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def indexer_score_topk_leaf_aiv(
        position_ids__ssa_v0: pl.Tensor[[T_DYN], pl.INT32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)],
        kv_seq_lens__ssa_v0: pl.Tensor[[B_DYN], pl.INT32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)],
        score_arena_inline1043_inline2235__ssa_v0: pl.InOut[pl.Tensor[[384, 8192], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 12582912)]],
        qr_hadamard_i8_inline527__rv_v2: pl.Tensor[[24576, 128], pl.INT8, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 3145728)],
        qr_hadamard_scale_dq_inline525__rv_v2: pl.Tensor[[24576, 1], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 98304)],
        weights_inline2220__ssa_v1: pl.Tensor[[384, 64], pl.FP32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 98304)],
        idx_block_table_flat_inline1044_inline2217__ssa_v0: pl.Tensor[[idx_table_len_inline1025_inline2222__ssa_v0], pl.INT32, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 0)],
        table_columns_inline1029_inline2268__ssa_v0: pl.Scalar[pl.INDEX],
        idx_kv_cache__rv_v2: pl.Tensor[[IDX_CACHE_BLOCK_NUM_DYN, INDEXER_PAGE_BYTES_DYN], pl.INT8, pl.MemRef("mem_ddr_7", pl.const(0, pl.INT64), 0)],
        pair_arena_inline1035_inline2211__ssa_v0: pl.Out[pl.Tensor[[24576, 1024], pl.FP32, pl.MemRef("mem_ddr_8", pl.const(0, pl.INT64), 100663296)]],
        __gm_pipe_buffer: pl.Out[pl.Tensor[[1], pl.FP32, pl.MemRef("mem_ddr_9", pl.const(0, pl.INT64), 4)]],
    ) -> tuple[pl.Tensor[[384, 8192], pl.FP32], pl.Tensor[[24576, 1024], pl.FP32]]:
        pl.func_attr({"slot_num": 1, "mx_tensor_views_blocked": True, "split_aiv": True, "dual_aiv_dispatch": True})
        mem_vec_14: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 49152)
        mem_vec_76: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32768)
        indexer_score_topk_leaf_c2v_slot_buffer: pl.Scalar[pl.INT32] = pl.system.reserve_buffer(name="indexer_score_topk_leaf_c2v_slot_buffer", size=98304, base=0)
        pl.system.aiv_initialize_pipe(indexer_score_topk_leaf_c2v_slot_buffer, pl.const(0, pl.INT32), dir_mask=1, slot_size=98304, slot_num=1)
        worker_inline1037_inline2245__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        query_count_inline1038_inline2205__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(position_ids__ssa_v0, 0)
        for batch_inline1034_inline2239__idx_v0, (max_cache_len_inline1046_inline2234__iter_v1,) in pl.range(query_count_inline1038_inline2205__ssa_v0 // 6, init_values=(0,)):
            t__tile: pl.Scalar[pl.INT32] = pl.tensor.read(kv_seq_lens__ssa_v0, [batch_inline1034_inline2239__idx_v0])
            batch_cache_len_inline1026_inline2241__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(t__tile, pl.INDEX) // 4
            max_cache_len_inline1046_inline2234__ssa_v3: pl.Scalar[pl.INDEX] = pl.max(max_cache_len_inline1046_inline2234__iter_v1, batch_cache_len_inline1026_inline2241__ssa_v0)
            max_cache_len_inline1046_inline2234__rv_v2: pl.Scalar[pl.INDEX] = pl.yield_(max_cache_len_inline1046_inline2234__ssa_v3)
        max_leaves_inline1023_inline2216__ssa_v0: pl.Scalar[pl.INDEX] = pl.max((pl.min(max_cache_len_inline1046_inline2234__rv_v2, 262144) + 8191) // 8192, 1)
        single_leaf_inline1032_inline2263__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(max_leaves_inline1023_inline2216__ssa_v0 == 1, pl.INDEX)
        for item_inline1031_inline2243__idx_v0, (score_arena_inline1043_inline2235__iter_v1,) in pl.range(
            worker_inline1037_inline2245__ssa_v0, query_count_inline1038_inline2205__ssa_v0 * max_leaves_inline1023_inline2216__ssa_v0, 24, init_values=(score_arena_inline1043_inline2235__ssa_v0,)
        ):
            query_inline1030_inline2255__ssa_v0: pl.Scalar[pl.INDEX] = item_inline1031_inline2243__idx_v0 // max_leaves_inline1023_inline2216__ssa_v0
            leaf_inline1051_inline2227__ssa_v0: pl.Scalar[pl.INDEX] = item_inline1031_inline2243__idx_v0 % max_leaves_inline1023_inline2216__ssa_v0
            batch_idx_inline1058_inline2244__ssa_v0: pl.Scalar[pl.INDEX] = query_inline1030_inline2255__ssa_v0 // 6
            position_inline1016_inline2246__tile: pl.Scalar[pl.INT32] = pl.tensor.read(position_ids__ssa_v0, [query_inline1030_inline2255__ssa_v0])
            t__tile_1: pl.Scalar[pl.INT32] = pl.tensor.read(kv_seq_lens__ssa_v0, [batch_idx_inline1058_inline2244__ssa_v0])
            cache_len_inline1020_inline2232__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(t__tile_1, pl.INDEX) // 4
            cache_bound_inline1013_inline2248__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(cache_len_inline1020_inline2232__ssa_v0, (pl.cast(position_inline1016_inline2246__tile, pl.INDEX) + 1) // 4)
            visible_count_inline1019_inline2238__ssa_v0: pl.Scalar[pl.INDEX] = pl.max(pl.min(cache_bound_inline1013_inline2248__ssa_v0, 262144), 0)
            logical_begin_inline1045_inline2251__ssa_v0: pl.Scalar[pl.INDEX] = leaf_inline1051_inline2227__ssa_v0 * 8192
            if logical_begin_inline1045_inline2251__ssa_v0 < visible_count_inline1019_inline2238__ssa_v0:
                valid_count_inline1015_inline2253__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(visible_count_inline1019_inline2238__ssa_v0 - logical_begin_inline1045_inline2251__ssa_v0, 8192)
                lane_span_inline1067_inline2256__ssa_v0: pl.Scalar[pl.INDEX] = pl.min((valid_count_inline1015_inline2253__ssa_v0 + 383) // 384 * 192, 4096)
                lane_stride_inline1014_inline2207__ssa_v0: pl.Scalar[pl.INDEX] = (
                    single_leaf_inline1032_inline2263__ssa_v0 * 192 + (1 - single_leaf_inline1032_inline2263__ssa_v0) * lane_span_inline1067_inline2256__ssa_v0
                )
                query_head_begin_inline1017_inline2258__ssa_v0: pl.Scalar[pl.INDEX] = query_inline1030_inline2255__ssa_v0 * 64
                t__tile_2: pl.Tile[[64, 1], pl.FP32, pl.MemRef(mem_vec_14, pl.const(98304, pl.INT64), 256), pl.Mem.Vec] = pl.tile.load(
                    qr_hadamard_scale_dq_inline525__rv_v2, [query_head_begin_inline1017_inline2258__ssa_v0, 0], [64, 1], [64, 1], target_memory=pl.Mem.Vec
                )
                query_scale_inline1068_inline2264__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_14, pl.const(98304, pl.INT64), 256), pl.Mem.Vec] = pl.tile.reshape(t__tile_2, [1, 64])
                query_weight_inline1048_inline2221__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_14, pl.const(98560, pl.INT64), 256), pl.Mem.Vec] = pl.tile.load(
                    weights_inline2220__ssa_v1, [query_inline1030_inline2255__ssa_v0, 0], [1, 64], [1, 64], target_memory=pl.Mem.Vec
                )
                t__tile_3: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_76, pl.const(147840, pl.INT64), 256), pl.Mem.Vec] = pl.tile.mul(
                    query_scale_inline1068_inline2264__tile, query_weight_inline1048_inline2221__tile
                )
                head_coefficient_inline1049_inline2265__tile: pl.Tile[[64, 1], pl.FP32, pl.MemRef(mem_vec_76, pl.const(147840, pl.INT64), 256), pl.Mem.Vec] = pl.tile.reshape(t__tile_3, [64, 1])
                unroll_main_end: pl.Scalar[pl.INDEX] = (lane_span_inline1067_inline2256__ssa_v0 + 191) // 384 * 384
                for score_begin_inline1050_inline2269__idx_v0, (score_arena_inline1043_inline2235__iter_v3,) in pl.range(
                    0, unroll_main_end, 384, init_values=(score_arena_inline1043_inline2235__iter_v1,)
                ):
                    read_begin_inline1055_inline2200__ssa_v0: pl.Scalar[pl.INDEX] = score_begin_inline1050_inline2269__idx_v0 * (single_leaf_inline1032_inline2263__ssa_v0 + 1)
                    aiv_id_inline1059_inline2189__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_subblock_idx()
                    lane_begin_inline1022_inline2188__ssa_v0: pl.Scalar[pl.INDEX] = aiv_id_inline1059_inline2189__ssa_v0 * lane_stride_inline1014_inline2207__ssa_v0
                    lane_valid_rows_inline1070_inline2186__ssa_v0: pl.Scalar[pl.INDEX] = pl.max(
                        pl.min(valid_count_inline1015_inline2253__ssa_v0 - read_begin_inline1055_inline2200__ssa_v0 - lane_begin_inline1022_inline2188__ssa_v0, 192), 0
                    )
                    scale_page_begin_inline1042_inline2185__ssa_v0: pl.Scalar[pl.INDEX] = 0
                    safe_scale_begin_inline1060_inline2184__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(
                        read_begin_inline1055_inline2200__ssa_v0 + lane_begin_inline1022_inline2188__ssa_v0 + scale_page_begin_inline1042_inline2185__ssa_v0,
                        (valid_count_inline1015_inline2253__ssa_v0 - 1) // 32 * 32,
                    )
                    scale_logical_page_inline1062_inline2250__ssa_v0: pl.Scalar[pl.INDEX] = (logical_begin_inline1045_inline2251__ssa_v0 + safe_scale_begin_inline1060_inline2184__ssa_v0) // 32
                    t__tile_4: pl.Scalar[pl.INT32] = pl.tensor.read(
                        idx_block_table_flat_inline1044_inline2217__ssa_v0,
                        [batch_idx_inline1058_inline2244__ssa_v0 * table_columns_inline1029_inline2268__ssa_v0 + scale_logical_page_inline1062_inline2250__ssa_v0],
                    )
                    scale_block_inline1072_inline2182__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(t__tile_4, pl.INDEX)
                    scale_page_begin_inline1042_inline2185__ssa_v1: pl.Scalar[pl.INDEX] = 32
                    safe_scale_begin_inline1060_inline2184__ssa_v1: pl.Scalar[pl.INDEX] = pl.min(
                        read_begin_inline1055_inline2200__ssa_v0 + lane_begin_inline1022_inline2188__ssa_v0 + scale_page_begin_inline1042_inline2185__ssa_v1,
                        (valid_count_inline1015_inline2253__ssa_v0 - 1) // 32 * 32,
                    )
                    scale_logical_page_inline1062_inline2250__ssa_v1: pl.Scalar[pl.INDEX] = (logical_begin_inline1045_inline2251__ssa_v0 + safe_scale_begin_inline1060_inline2184__ssa_v1) // 32
                    t__tile_5: pl.Scalar[pl.INT32] = pl.tensor.read(
                        idx_block_table_flat_inline1044_inline2217__ssa_v0,
                        [batch_idx_inline1058_inline2244__ssa_v0 * table_columns_inline1029_inline2268__ssa_v0 + scale_logical_page_inline1062_inline2250__ssa_v1],
                    )
                    scale_block_inline1072_inline2182__ssa_v1: pl.Scalar[pl.INDEX] = pl.cast(t__tile_5, pl.INDEX)
                    scale_page_begin_inline1042_inline2185__ssa_v2: pl.Scalar[pl.INDEX] = 64
                    safe_scale_begin_inline1060_inline2184__ssa_v2: pl.Scalar[pl.INDEX] = pl.min(
                        read_begin_inline1055_inline2200__ssa_v0 + lane_begin_inline1022_inline2188__ssa_v0 + scale_page_begin_inline1042_inline2185__ssa_v2,
                        (valid_count_inline1015_inline2253__ssa_v0 - 1) // 32 * 32,
                    )
                    scale_logical_page_inline1062_inline2250__ssa_v2: pl.Scalar[pl.INDEX] = (logical_begin_inline1045_inline2251__ssa_v0 + safe_scale_begin_inline1060_inline2184__ssa_v2) // 32
                    t__tile_6: pl.Scalar[pl.INT32] = pl.tensor.read(
                        idx_block_table_flat_inline1044_inline2217__ssa_v0,
                        [batch_idx_inline1058_inline2244__ssa_v0 * table_columns_inline1029_inline2268__ssa_v0 + scale_logical_page_inline1062_inline2250__ssa_v2],
                    )
                    scale_block_inline1072_inline2182__ssa_v2: pl.Scalar[pl.INDEX] = pl.cast(t__tile_6, pl.INDEX)
                    scale_page_begin_inline1042_inline2185__ssa_v3: pl.Scalar[pl.INDEX] = 96
                    safe_scale_begin_inline1060_inline2184__ssa_v3: pl.Scalar[pl.INDEX] = pl.min(
                        read_begin_inline1055_inline2200__ssa_v0 + lane_begin_inline1022_inline2188__ssa_v0 + scale_page_begin_inline1042_inline2185__ssa_v3,
                        (valid_count_inline1015_inline2253__ssa_v0 - 1) // 32 * 32,
                    )
                    scale_logical_page_inline1062_inline2250__ssa_v3: pl.Scalar[pl.INDEX] = (logical_begin_inline1045_inline2251__ssa_v0 + safe_scale_begin_inline1060_inline2184__ssa_v3) // 32
                    t__tile_7: pl.Scalar[pl.INT32] = pl.tensor.read(
                        idx_block_table_flat_inline1044_inline2217__ssa_v0,
                        [batch_idx_inline1058_inline2244__ssa_v0 * table_columns_inline1029_inline2268__ssa_v0 + scale_logical_page_inline1062_inline2250__ssa_v3],
                    )
                    scale_block_inline1072_inline2182__ssa_v3: pl.Scalar[pl.INDEX] = pl.cast(t__tile_7, pl.INDEX)
                    scale_page_begin_inline1042_inline2185__ssa_v4: pl.Scalar[pl.INDEX] = 128
                    safe_scale_begin_inline1060_inline2184__ssa_v4: pl.Scalar[pl.INDEX] = pl.min(
                        read_begin_inline1055_inline2200__ssa_v0 + lane_begin_inline1022_inline2188__ssa_v0 + scale_page_begin_inline1042_inline2185__ssa_v4,
                        (valid_count_inline1015_inline2253__ssa_v0 - 1) // 32 * 32,
                    )
                    scale_logical_page_inline1062_inline2250__ssa_v4: pl.Scalar[pl.INDEX] = (logical_begin_inline1045_inline2251__ssa_v0 + safe_scale_begin_inline1060_inline2184__ssa_v4) // 32
                    t__tile_8: pl.Scalar[pl.INT32] = pl.tensor.read(
                        idx_block_table_flat_inline1044_inline2217__ssa_v0,
                        [batch_idx_inline1058_inline2244__ssa_v0 * table_columns_inline1029_inline2268__ssa_v0 + scale_logical_page_inline1062_inline2250__ssa_v4],
                    )
                    scale_block_inline1072_inline2182__ssa_v4: pl.Scalar[pl.INDEX] = pl.cast(t__tile_8, pl.INDEX)
                    scale_page_begin_inline1042_inline2185__ssa_v5: pl.Scalar[pl.INDEX] = 160
                    safe_scale_begin_inline1060_inline2184__ssa_v5: pl.Scalar[pl.INDEX] = pl.min(
                        read_begin_inline1055_inline2200__ssa_v0 + lane_begin_inline1022_inline2188__ssa_v0 + scale_page_begin_inline1042_inline2185__ssa_v5,
                        (valid_count_inline1015_inline2253__ssa_v0 - 1) // 32 * 32,
                    )
                    scale_logical_page_inline1062_inline2250__ssa_v5: pl.Scalar[pl.INDEX] = (logical_begin_inline1045_inline2251__ssa_v0 + safe_scale_begin_inline1060_inline2184__ssa_v5) // 32
                    t__tile_9: pl.Scalar[pl.INT32] = pl.tensor.read(
                        idx_block_table_flat_inline1044_inline2217__ssa_v0,
                        [batch_idx_inline1058_inline2244__ssa_v0 * table_columns_inline1029_inline2268__ssa_v0 + scale_logical_page_inline1062_inline2250__ssa_v5],
                    )
                    scale_block_inline1072_inline2182__ssa_v5: pl.Scalar[pl.INDEX] = pl.cast(t__tile_9, pl.INDEX)
                    score_row_id_inline1010_inline2242__ssa_v0: pl.Scalar[pl.INDEX] = single_leaf_inline1032_inline2263__ssa_v0 * query_inline1030_inline2255__ssa_v0 + (
                        1 - single_leaf_inline1032_inline2263__ssa_v0
                    ) * (worker_inline1037_inline2245__ssa_v0 * 2 + aiv_id_inline1059_inline2189__ssa_v0)
                    score_col_inline1009_inline2254__ssa_v0: pl.Scalar[pl.INDEX] = (
                        single_leaf_inline1032_inline2263__ssa_v0 * (read_begin_inline1055_inline2200__ssa_v0 + lane_begin_inline1022_inline2188__ssa_v0)
                        + (1 - single_leaf_inline1032_inline2263__ssa_v0) * score_begin_inline1050_inline2269__idx_v0
                    )
                    read_begin_inline1055_inline2200__ssa_v0_1: pl.Scalar[pl.INDEX] = (score_begin_inline1050_inline2269__idx_v0 + 192) * (single_leaf_inline1032_inline2263__ssa_v0 + 1)
                    aiv_id_inline1059_inline2189__ssa_v0_1: pl.Scalar[pl.INDEX] = pl.tile.get_subblock_idx()
                    lane_begin_inline1022_inline2188__ssa_v0_1: pl.Scalar[pl.INDEX] = aiv_id_inline1059_inline2189__ssa_v0_1 * lane_stride_inline1014_inline2207__ssa_v0
                    lane_valid_rows_inline1070_inline2186__ssa_v0_1: pl.Scalar[pl.INDEX] = pl.max(
                        pl.min(valid_count_inline1015_inline2253__ssa_v0 - read_begin_inline1055_inline2200__ssa_v0_1 - lane_begin_inline1022_inline2188__ssa_v0_1, 192), 0
                    )
                    scale_page_begin_inline1042_inline2185__ssa_v0_1: pl.Scalar[pl.INDEX] = 0
                    safe_scale_begin_inline1060_inline2184__ssa_v0_1: pl.Scalar[pl.INDEX] = pl.min(
                        read_begin_inline1055_inline2200__ssa_v0_1 + lane_begin_inline1022_inline2188__ssa_v0_1 + scale_page_begin_inline1042_inline2185__ssa_v0_1,
                        (valid_count_inline1015_inline2253__ssa_v0 - 1) // 32 * 32,
                    )
                    scale_logical_page_inline1062_inline2250__ssa_v0_1: pl.Scalar[pl.INDEX] = (logical_begin_inline1045_inline2251__ssa_v0 + safe_scale_begin_inline1060_inline2184__ssa_v0_1) // 32
                    t__tile_10: pl.Scalar[pl.INT32] = pl.tensor.read(
                        idx_block_table_flat_inline1044_inline2217__ssa_v0,
                        [batch_idx_inline1058_inline2244__ssa_v0 * table_columns_inline1029_inline2268__ssa_v0 + scale_logical_page_inline1062_inline2250__ssa_v0_1],
                    )
                    scale_block_inline1072_inline2182__ssa_v0_1: pl.Scalar[pl.INDEX] = pl.cast(t__tile_10, pl.INDEX)
                    scale_page_begin_inline1042_inline2185__ssa_v1_1: pl.Scalar[pl.INDEX] = 32
                    safe_scale_begin_inline1060_inline2184__ssa_v1_1: pl.Scalar[pl.INDEX] = pl.min(
                        read_begin_inline1055_inline2200__ssa_v0_1 + lane_begin_inline1022_inline2188__ssa_v0_1 + scale_page_begin_inline1042_inline2185__ssa_v1_1,
                        (valid_count_inline1015_inline2253__ssa_v0 - 1) // 32 * 32,
                    )
                    scale_logical_page_inline1062_inline2250__ssa_v1_1: pl.Scalar[pl.INDEX] = (logical_begin_inline1045_inline2251__ssa_v0 + safe_scale_begin_inline1060_inline2184__ssa_v1_1) // 32
                    t__tile_11: pl.Scalar[pl.INT32] = pl.tensor.read(
                        idx_block_table_flat_inline1044_inline2217__ssa_v0,
                        [batch_idx_inline1058_inline2244__ssa_v0 * table_columns_inline1029_inline2268__ssa_v0 + scale_logical_page_inline1062_inline2250__ssa_v1_1],
                    )
                    scale_block_inline1072_inline2182__ssa_v1_1: pl.Scalar[pl.INDEX] = pl.cast(t__tile_11, pl.INDEX)
                    scale_page_begin_inline1042_inline2185__ssa_v2_1: pl.Scalar[pl.INDEX] = 64
                    safe_scale_begin_inline1060_inline2184__ssa_v2_1: pl.Scalar[pl.INDEX] = pl.min(
                        read_begin_inline1055_inline2200__ssa_v0_1 + lane_begin_inline1022_inline2188__ssa_v0_1 + scale_page_begin_inline1042_inline2185__ssa_v2_1,
                        (valid_count_inline1015_inline2253__ssa_v0 - 1) // 32 * 32,
                    )
                    scale_logical_page_inline1062_inline2250__ssa_v2_1: pl.Scalar[pl.INDEX] = (logical_begin_inline1045_inline2251__ssa_v0 + safe_scale_begin_inline1060_inline2184__ssa_v2_1) // 32
                    t__tile_12: pl.Scalar[pl.INT32] = pl.tensor.read(
                        idx_block_table_flat_inline1044_inline2217__ssa_v0,
                        [batch_idx_inline1058_inline2244__ssa_v0 * table_columns_inline1029_inline2268__ssa_v0 + scale_logical_page_inline1062_inline2250__ssa_v2_1],
                    )
                    scale_block_inline1072_inline2182__ssa_v2_1: pl.Scalar[pl.INDEX] = pl.cast(t__tile_12, pl.INDEX)
                    scale_page_begin_inline1042_inline2185__ssa_v3_1: pl.Scalar[pl.INDEX] = 96
                    safe_scale_begin_inline1060_inline2184__ssa_v3_1: pl.Scalar[pl.INDEX] = pl.min(
                        read_begin_inline1055_inline2200__ssa_v0_1 + lane_begin_inline1022_inline2188__ssa_v0_1 + scale_page_begin_inline1042_inline2185__ssa_v3_1,
                        (valid_count_inline1015_inline2253__ssa_v0 - 1) // 32 * 32,
                    )
                    scale_logical_page_inline1062_inline2250__ssa_v3_1: pl.Scalar[pl.INDEX] = (logical_begin_inline1045_inline2251__ssa_v0 + safe_scale_begin_inline1060_inline2184__ssa_v3_1) // 32
                    t__tile_13: pl.Scalar[pl.INT32] = pl.tensor.read(
                        idx_block_table_flat_inline1044_inline2217__ssa_v0,
                        [batch_idx_inline1058_inline2244__ssa_v0 * table_columns_inline1029_inline2268__ssa_v0 + scale_logical_page_inline1062_inline2250__ssa_v3_1],
                    )
                    scale_block_inline1072_inline2182__ssa_v3_1: pl.Scalar[pl.INDEX] = pl.cast(t__tile_13, pl.INDEX)
                    scale_page_begin_inline1042_inline2185__ssa_v4_1: pl.Scalar[pl.INDEX] = 128
                    safe_scale_begin_inline1060_inline2184__ssa_v4_1: pl.Scalar[pl.INDEX] = pl.min(
                        read_begin_inline1055_inline2200__ssa_v0_1 + lane_begin_inline1022_inline2188__ssa_v0_1 + scale_page_begin_inline1042_inline2185__ssa_v4_1,
                        (valid_count_inline1015_inline2253__ssa_v0 - 1) // 32 * 32,
                    )
                    scale_logical_page_inline1062_inline2250__ssa_v4_1: pl.Scalar[pl.INDEX] = (logical_begin_inline1045_inline2251__ssa_v0 + safe_scale_begin_inline1060_inline2184__ssa_v4_1) // 32
                    t__tile_14: pl.Scalar[pl.INT32] = pl.tensor.read(
                        idx_block_table_flat_inline1044_inline2217__ssa_v0,
                        [batch_idx_inline1058_inline2244__ssa_v0 * table_columns_inline1029_inline2268__ssa_v0 + scale_logical_page_inline1062_inline2250__ssa_v4_1],
                    )
                    scale_block_inline1072_inline2182__ssa_v4_1: pl.Scalar[pl.INDEX] = pl.cast(t__tile_14, pl.INDEX)
                    scale_page_begin_inline1042_inline2185__ssa_v5_1: pl.Scalar[pl.INDEX] = 160
                    safe_scale_begin_inline1060_inline2184__ssa_v5_1: pl.Scalar[pl.INDEX] = pl.min(
                        read_begin_inline1055_inline2200__ssa_v0_1 + lane_begin_inline1022_inline2188__ssa_v0_1 + scale_page_begin_inline1042_inline2185__ssa_v5_1,
                        (valid_count_inline1015_inline2253__ssa_v0 - 1) // 32 * 32,
                    )
                    scale_logical_page_inline1062_inline2250__ssa_v5_1: pl.Scalar[pl.INDEX] = (logical_begin_inline1045_inline2251__ssa_v0 + safe_scale_begin_inline1060_inline2184__ssa_v5_1) // 32
                    t__tile_15: pl.Scalar[pl.INT32] = pl.tensor.read(
                        idx_block_table_flat_inline1044_inline2217__ssa_v0,
                        [batch_idx_inline1058_inline2244__ssa_v0 * table_columns_inline1029_inline2268__ssa_v0 + scale_logical_page_inline1062_inline2250__ssa_v5_1],
                    )
                    scale_block_inline1072_inline2182__ssa_v5_1: pl.Scalar[pl.INDEX] = pl.cast(t__tile_15, pl.INDEX)
                    score_row_id_inline1010_inline2242__ssa_v0_1: pl.Scalar[pl.INDEX] = single_leaf_inline1032_inline2263__ssa_v0 * query_inline1030_inline2255__ssa_v0 + (
                        1 - single_leaf_inline1032_inline2263__ssa_v0
                    ) * (worker_inline1037_inline2245__ssa_v0 * 2 + aiv_id_inline1059_inline2189__ssa_v0_1)
                    score_col_inline1009_inline2254__ssa_v0_1: pl.Scalar[pl.INDEX] = single_leaf_inline1032_inline2263__ssa_v0 * (
                        read_begin_inline1055_inline2200__ssa_v0_1 + lane_begin_inline1022_inline2188__ssa_v0_1
                    ) + (1 - single_leaf_inline1032_inline2263__ssa_v0) * (score_begin_inline1050_inline2269__idx_v0 + 192)
                    kv_scale_bytes_inline1064_inline2240__tile: pl.Tile[[1, 384], pl.INT8, pl.MemRef(mem_vec_76, pl.const(147456, pl.INT64), 384), pl.Mem.Vec] = pl.tile.create(
                        [1, 384], dtype=pl.INT8, target_memory=pl.Mem.Vec
                    )
                    kv_scale_bytes_inline1064_inline2240__tile_1: pl.Tile[[1, 384], pl.INT8, pl.MemRef(mem_vec_76, pl.const(147456, pl.INT64), 384), pl.Mem.Vec] = pl.tile.gather_row(
                        kv_scale_bytes_inline1064_inline2240__tile,
                        idx_kv_cache__rv_v2,
                        [0, scale_page_begin_inline1042_inline2185__ssa_v0 * 2],
                        [scale_block_inline1072_inline2182__ssa_v0, 4096],
                        [1, 64],
                        transpose=False,
                    )
                    kv_scale_bytes_inline1064_inline2240__tile_2: pl.Tile[[1, 384], pl.INT8, pl.MemRef(mem_vec_76, pl.const(147456, pl.INT64), 384), pl.Mem.Vec] = pl.tile.gather_row(
                        kv_scale_bytes_inline1064_inline2240__tile_1,
                        idx_kv_cache__rv_v2,
                        [0, scale_page_begin_inline1042_inline2185__ssa_v1 * 2],
                        [scale_block_inline1072_inline2182__ssa_v1, 4096],
                        [1, 64],
                        transpose=False,
                    )
                    kv_scale_bytes_inline1064_inline2240__tile_3: pl.Tile[[1, 384], pl.INT8, pl.MemRef(mem_vec_76, pl.const(147456, pl.INT64), 384), pl.Mem.Vec] = pl.tile.gather_row(
                        kv_scale_bytes_inline1064_inline2240__tile_2,
                        idx_kv_cache__rv_v2,
                        [0, scale_page_begin_inline1042_inline2185__ssa_v2 * 2],
                        [scale_block_inline1072_inline2182__ssa_v2, 4096],
                        [1, 64],
                        transpose=False,
                    )
                    kv_scale_bytes_inline1064_inline2240__tile_4: pl.Tile[[1, 384], pl.INT8, pl.MemRef(mem_vec_76, pl.const(147456, pl.INT64), 384), pl.Mem.Vec] = pl.tile.gather_row(
                        kv_scale_bytes_inline1064_inline2240__tile_3,
                        idx_kv_cache__rv_v2,
                        [0, scale_page_begin_inline1042_inline2185__ssa_v3 * 2],
                        [scale_block_inline1072_inline2182__ssa_v3, 4096],
                        [1, 64],
                        transpose=False,
                    )
                    kv_scale_bytes_inline1064_inline2240__tile_5: pl.Tile[[1, 384], pl.INT8, pl.MemRef(mem_vec_76, pl.const(147456, pl.INT64), 384), pl.Mem.Vec] = pl.tile.gather_row(
                        kv_scale_bytes_inline1064_inline2240__tile_4,
                        idx_kv_cache__rv_v2,
                        [0, scale_page_begin_inline1042_inline2185__ssa_v4 * 2],
                        [scale_block_inline1072_inline2182__ssa_v4, 4096],
                        [1, 64],
                        transpose=False,
                    )
                    kv_scale_bytes_inline1064_inline2240__tile_6: pl.Tile[[1, 384], pl.INT8, pl.MemRef(mem_vec_76, pl.const(147456, pl.INT64), 384), pl.Mem.Vec] = pl.tile.gather_row(
                        kv_scale_bytes_inline1064_inline2240__tile_5,
                        idx_kv_cache__rv_v2,
                        [0, scale_page_begin_inline1042_inline2185__ssa_v5 * 2],
                        [scale_block_inline1072_inline2182__ssa_v5, 4096],
                        [1, 64],
                        transpose=False,
                    )
                    kv_scale_inline1061_inline2187__tile: pl.Tile[[1, 192], pl.FP16, pl.MemRef(mem_vec_76, pl.const(147456, pl.INT64), 384), pl.Mem.Vec] = pl.tile.reinterpret_view(
                        kv_scale_bytes_inline1064_inline2240__tile_6, dtype=pl.FP16
                    )
                    score_shard_inline1066_inline2180__tile: pl.Tile[[64, 192], pl.INT32, pl.Mem.Vec] = pl.tile.tpop_from_aic(split=2)
                    score_fp32_inline1063_inline2179__tile: pl.Tile[[64, 192], pl.FP32, pl.MemRef(mem_vec_14, pl.const(98304, pl.INT64), 49152), pl.Mem.Vec] = pl.tile.cast(
                        score_shard_inline1066_inline2180__tile, target_type=pl.FP32, mode="none"
                    )
                    pl.system.tfree_to_aic(score_shard_inline1066_inline2180__tile, split=2)
                    score_fp32_v1_inline1069_inline2196__tile: pl.Tile[[64, 192], pl.FP32, pl.MemRef(mem_vec_14, pl.const(98304, pl.INT64), 49152), pl.Mem.Vec] = pl.tile.maximums(
                        score_fp32_inline1063_inline2179__tile, 0.0
                    )
                    score_fp32_v2_inline1073_inline2224__tile: pl.Tile[[64, 192], pl.FP32, pl.MemRef(mem_vec_14, pl.const(98304, pl.INT64), 49152), pl.Mem.Vec] = pl.tile.row_expand_mul(
                        score_fp32_v1_inline1069_inline2196__tile, head_coefficient_inline1049_inline2265__tile
                    )
                    score_sum_inline1074_inline2178__tile: pl.Tile[[1, 192], pl.FP32, pl.MemRef(mem_vec_14, pl.const(98304, pl.INT64), 768), pl.Mem.Vec] = pl.tile.col_sum(
                        score_fp32_v2_inline1073_inline2224__tile
                    )
                    score_row_inline1071_inline2177__tile: pl.Tile[[1, 192], pl.FP32, pl.MemRef(mem_vec_14, pl.const(98304, pl.INT64), 768), pl.Mem.Vec] = score_sum_inline1074_inline2178__tile
                    t__tile_16: pl.Tile[[1, 192], pl.FP32, pl.MemRef(mem_vec_14, pl.const(99072, pl.INT64), 768), pl.Mem.Vec] = pl.tile.cast(
                        kv_scale_inline1061_inline2187__tile, target_type=pl.FP32, mode="round"
                    )
                    score_row_v1_inline1011_inline2261__tile: pl.Tile[[1, 192], pl.FP32, pl.MemRef(mem_vec_14, pl.const(98304, pl.INT64), 768), pl.Mem.Vec] = pl.tile.mul(
                        score_row_inline1071_inline2177__tile, t__tile_16
                    )
                    if 0 < lane_valid_rows_inline1070_inline2186__ssa_v0:
                        score_valid_inline1007_inline2190__tile: pl.Tile[
                            [1, 192], pl.FP32, pl.MemRef(mem_vec_14, pl.const(98304, pl.INT64), 768), pl.Mem.Vec, pl.TileView(valid_shape=[1, lane_valid_rows_inline1070_inline2186__ssa_v0])
                        ] = pl.tile.set_validshape(score_row_v1_inline1011_inline2261__tile, 1, lane_valid_rows_inline1070_inline2186__ssa_v0)
                        score_arena_inline1043_inline2235__tile: pl.Tensor[[384, 8192], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 12582912)] = pl.tile.store(
                            score_valid_inline1007_inline2190__tile, [score_row_id_inline1010_inline2242__ssa_v0, score_col_inline1009_inline2254__ssa_v0], score_arena_inline1043_inline2235__iter_v3
                        )
                        score_arena_inline1043_inline2235__phi_v6: pl.Tensor[[384, 8192], pl.FP32, pl.MemRef("mem_ddr_20", pl.const(0, pl.INT64), 12582912)] = pl.yield_(
                            score_arena_inline1043_inline2235__tile
                        )
                    else:
                        score_arena_inline1043_inline2235__phi_v6: pl.Tensor[[384, 8192], pl.FP32, pl.MemRef("mem_ddr_20", pl.const(0, pl.INT64), 12582912)] = pl.yield_(
                            score_arena_inline1043_inline2235__iter_v3
                        )
                    kv_scale_bytes_inline1064_inline2240__tile_7: pl.Tile[[1, 384], pl.INT8, pl.MemRef(mem_vec_76, pl.const(147456, pl.INT64), 384), pl.Mem.Vec] = pl.tile.create(
                        [1, 384], dtype=pl.INT8, target_memory=pl.Mem.Vec
                    )
                    kv_scale_bytes_inline1064_inline2240__tile_8: pl.Tile[[1, 384], pl.INT8, pl.MemRef(mem_vec_76, pl.const(147456, pl.INT64), 384), pl.Mem.Vec] = pl.tile.gather_row(
                        kv_scale_bytes_inline1064_inline2240__tile_7,
                        idx_kv_cache__rv_v2,
                        [0, scale_page_begin_inline1042_inline2185__ssa_v0_1 * 2],
                        [scale_block_inline1072_inline2182__ssa_v0_1, 4096],
                        [1, 64],
                        transpose=False,
                    )
                    kv_scale_bytes_inline1064_inline2240__tile_9: pl.Tile[[1, 384], pl.INT8, pl.MemRef(mem_vec_76, pl.const(147456, pl.INT64), 384), pl.Mem.Vec] = pl.tile.gather_row(
                        kv_scale_bytes_inline1064_inline2240__tile_8,
                        idx_kv_cache__rv_v2,
                        [0, scale_page_begin_inline1042_inline2185__ssa_v1_1 * 2],
                        [scale_block_inline1072_inline2182__ssa_v1_1, 4096],
                        [1, 64],
                        transpose=False,
                    )
                    kv_scale_bytes_inline1064_inline2240__tile_10: pl.Tile[[1, 384], pl.INT8, pl.MemRef(mem_vec_76, pl.const(147456, pl.INT64), 384), pl.Mem.Vec] = pl.tile.gather_row(
                        kv_scale_bytes_inline1064_inline2240__tile_9,
                        idx_kv_cache__rv_v2,
                        [0, scale_page_begin_inline1042_inline2185__ssa_v2_1 * 2],
                        [scale_block_inline1072_inline2182__ssa_v2_1, 4096],
                        [1, 64],
                        transpose=False,
                    )
                    kv_scale_bytes_inline1064_inline2240__tile_11: pl.Tile[[1, 384], pl.INT8, pl.MemRef(mem_vec_76, pl.const(147456, pl.INT64), 384), pl.Mem.Vec] = pl.tile.gather_row(
                        kv_scale_bytes_inline1064_inline2240__tile_10,
                        idx_kv_cache__rv_v2,
                        [0, scale_page_begin_inline1042_inline2185__ssa_v3_1 * 2],
                        [scale_block_inline1072_inline2182__ssa_v3_1, 4096],
                        [1, 64],
                        transpose=False,
                    )
                    kv_scale_bytes_inline1064_inline2240__tile_12: pl.Tile[[1, 384], pl.INT8, pl.MemRef(mem_vec_76, pl.const(147456, pl.INT64), 384), pl.Mem.Vec] = pl.tile.gather_row(
                        kv_scale_bytes_inline1064_inline2240__tile_11,
                        idx_kv_cache__rv_v2,
                        [0, scale_page_begin_inline1042_inline2185__ssa_v4_1 * 2],
                        [scale_block_inline1072_inline2182__ssa_v4_1, 4096],
                        [1, 64],
                        transpose=False,
                    )
                    kv_scale_bytes_inline1064_inline2240__tile_13: pl.Tile[[1, 384], pl.INT8, pl.MemRef(mem_vec_76, pl.const(147456, pl.INT64), 384), pl.Mem.Vec] = pl.tile.gather_row(
                        kv_scale_bytes_inline1064_inline2240__tile_12,
                        idx_kv_cache__rv_v2,
                        [0, scale_page_begin_inline1042_inline2185__ssa_v5_1 * 2],
                        [scale_block_inline1072_inline2182__ssa_v5_1, 4096],
                        [1, 64],
                        transpose=False,
                    )
                    kv_scale_inline1061_inline2187__tile_1: pl.Tile[[1, 192], pl.FP16, pl.MemRef(mem_vec_76, pl.const(147456, pl.INT64), 384), pl.Mem.Vec] = pl.tile.reinterpret_view(
                        kv_scale_bytes_inline1064_inline2240__tile_13, dtype=pl.FP16
                    )
                    score_shard_inline1066_inline2180__tile_1: pl.Tile[[64, 192], pl.INT32, pl.Mem.Vec] = pl.tile.tpop_from_aic(split=2)
                    score_fp32_inline1063_inline2179__tile_1: pl.Tile[[64, 192], pl.FP32, pl.MemRef(mem_vec_14, pl.const(98304, pl.INT64), 49152), pl.Mem.Vec] = pl.tile.cast(
                        score_shard_inline1066_inline2180__tile_1, target_type=pl.FP32, mode="none"
                    )
                    pl.system.tfree_to_aic(score_shard_inline1066_inline2180__tile_1, split=2)
                    score_fp32_v1_inline1069_inline2196__tile_1: pl.Tile[[64, 192], pl.FP32, pl.MemRef(mem_vec_14, pl.const(98304, pl.INT64), 49152), pl.Mem.Vec] = pl.tile.maximums(
                        score_fp32_inline1063_inline2179__tile_1, 0.0
                    )
                    score_fp32_v2_inline1073_inline2224__tile_1: pl.Tile[[64, 192], pl.FP32, pl.MemRef(mem_vec_14, pl.const(98304, pl.INT64), 49152), pl.Mem.Vec] = pl.tile.row_expand_mul(
                        score_fp32_v1_inline1069_inline2196__tile_1, head_coefficient_inline1049_inline2265__tile
                    )
                    score_sum_inline1074_inline2178__tile_1: pl.Tile[[1, 192], pl.FP32, pl.MemRef(mem_vec_14, pl.const(98304, pl.INT64), 768), pl.Mem.Vec] = pl.tile.col_sum(
                        score_fp32_v2_inline1073_inline2224__tile_1
                    )
                    score_row_inline1071_inline2177__tile_1: pl.Tile[[1, 192], pl.FP32, pl.MemRef(mem_vec_14, pl.const(98304, pl.INT64), 768), pl.Mem.Vec] = score_sum_inline1074_inline2178__tile_1
                    t__tile_17: pl.Tile[[1, 192], pl.FP32, pl.MemRef(mem_vec_14, pl.const(99072, pl.INT64), 768), pl.Mem.Vec] = pl.tile.cast(
                        kv_scale_inline1061_inline2187__tile_1, target_type=pl.FP32, mode="round"
                    )
                    score_row_v1_inline1011_inline2261__tile_1: pl.Tile[[1, 192], pl.FP32, pl.MemRef(mem_vec_14, pl.const(98304, pl.INT64), 768), pl.Mem.Vec] = pl.tile.mul(
                        score_row_inline1071_inline2177__tile_1, t__tile_17
                    )
                    if 0 < lane_valid_rows_inline1070_inline2186__ssa_v0_1:
                        score_valid_inline1007_inline2190__tile_1: pl.Tile[
                            [1, 192], pl.FP32, pl.MemRef(mem_vec_14, pl.const(98304, pl.INT64), 768), pl.Mem.Vec, pl.TileView(valid_shape=[1, lane_valid_rows_inline1070_inline2186__ssa_v0_1])
                        ] = pl.tile.set_validshape(score_row_v1_inline1011_inline2261__tile_1, 1, lane_valid_rows_inline1070_inline2186__ssa_v0_1)
                        score_arena_inline1043_inline2235__tile_1: pl.Tensor[[384, 8192], pl.FP32, pl.MemRef("mem_ddr_20", pl.const(0, pl.INT64), 12582912)] = pl.tile.store(
                            score_valid_inline1007_inline2190__tile_1,
                            [score_row_id_inline1010_inline2242__ssa_v0_1, score_col_inline1009_inline2254__ssa_v0_1],
                            score_arena_inline1043_inline2235__phi_v6,
                        )
                        score_arena_inline1043_inline2235__phi_v6_1: pl.Tensor[[384, 8192], pl.FP32, pl.MemRef("mem_ddr_28", pl.const(0, pl.INT64), 12582912)] = pl.yield_(
                            score_arena_inline1043_inline2235__tile_1
                        )
                    else:
                        score_arena_inline1043_inline2235__phi_v6_1: pl.Tensor[[384, 8192], pl.FP32, pl.MemRef("mem_ddr_28", pl.const(0, pl.INT64), 12582912)] = pl.yield_(
                            score_arena_inline1043_inline2235__phi_v6
                        )
                    score_arena_inline1043_inline2235__rv_v4_main: pl.Tensor[[384, 8192], pl.FP32, pl.MemRef("mem_ddr_29", pl.const(0, pl.INT64), 12582912)] = pl.yield_(
                        score_arena_inline1043_inline2235__phi_v6_1
                    )
                unroll_rem: pl.Scalar[pl.INDEX] = (lane_span_inline1067_inline2256__ssa_v0 + 191) // 192 - (lane_span_inline1067_inline2256__ssa_v0 + 191) // 384 * 2
                if unroll_rem == 1:
                    read_begin_inline1055_inline2200__ssa_v0_2: pl.Scalar[pl.INDEX] = unroll_main_end * (single_leaf_inline1032_inline2263__ssa_v0 + 1)
                    aiv_id_inline1059_inline2189__ssa_v0_2: pl.Scalar[pl.INDEX] = pl.tile.get_subblock_idx()
                    lane_begin_inline1022_inline2188__ssa_v0_2: pl.Scalar[pl.INDEX] = aiv_id_inline1059_inline2189__ssa_v0_2 * lane_stride_inline1014_inline2207__ssa_v0
                    lane_valid_rows_inline1070_inline2186__ssa_v0_2: pl.Scalar[pl.INDEX] = pl.max(
                        pl.min(valid_count_inline1015_inline2253__ssa_v0 - read_begin_inline1055_inline2200__ssa_v0_2 - lane_begin_inline1022_inline2188__ssa_v0_2, 192), 0
                    )
                    kv_scale_bytes_inline1064_inline2240__tile_14: pl.Tile[[1, 384], pl.INT8, pl.MemRef(mem_vec_76, pl.const(147456, pl.INT64), 384), pl.Mem.Vec] = pl.tile.create(
                        [1, 384], dtype=pl.INT8, target_memory=pl.Mem.Vec
                    )
                    scale_page_begin_inline1042_inline2185__ssa_v0_2: pl.Scalar[pl.INDEX] = 0
                    safe_scale_begin_inline1060_inline2184__ssa_v0_2: pl.Scalar[pl.INDEX] = pl.min(
                        read_begin_inline1055_inline2200__ssa_v0_2 + lane_begin_inline1022_inline2188__ssa_v0_2 + scale_page_begin_inline1042_inline2185__ssa_v0_2,
                        (valid_count_inline1015_inline2253__ssa_v0 - 1) // 32 * 32,
                    )
                    scale_logical_page_inline1062_inline2250__ssa_v0_2: pl.Scalar[pl.INDEX] = (logical_begin_inline1045_inline2251__ssa_v0 + safe_scale_begin_inline1060_inline2184__ssa_v0_2) // 32
                    t__tile_18: pl.Scalar[pl.INT32] = pl.tensor.read(
                        idx_block_table_flat_inline1044_inline2217__ssa_v0,
                        [batch_idx_inline1058_inline2244__ssa_v0 * table_columns_inline1029_inline2268__ssa_v0 + scale_logical_page_inline1062_inline2250__ssa_v0_2],
                    )
                    scale_block_inline1072_inline2182__ssa_v0_2: pl.Scalar[pl.INDEX] = pl.cast(t__tile_18, pl.INDEX)
                    kv_scale_bytes_inline1064_inline2240__tile_15: pl.Tile[[1, 384], pl.INT8, pl.MemRef(mem_vec_76, pl.const(147456, pl.INT64), 384), pl.Mem.Vec] = pl.tile.gather_row(
                        kv_scale_bytes_inline1064_inline2240__tile_14,
                        idx_kv_cache__rv_v2,
                        [0, scale_page_begin_inline1042_inline2185__ssa_v0_2 * 2],
                        [scale_block_inline1072_inline2182__ssa_v0_2, 4096],
                        [1, 64],
                        transpose=False,
                    )
                    scale_page_begin_inline1042_inline2185__ssa_v1_2: pl.Scalar[pl.INDEX] = 32
                    safe_scale_begin_inline1060_inline2184__ssa_v1_2: pl.Scalar[pl.INDEX] = pl.min(
                        read_begin_inline1055_inline2200__ssa_v0_2 + lane_begin_inline1022_inline2188__ssa_v0_2 + scale_page_begin_inline1042_inline2185__ssa_v1_2,
                        (valid_count_inline1015_inline2253__ssa_v0 - 1) // 32 * 32,
                    )
                    scale_logical_page_inline1062_inline2250__ssa_v1_2: pl.Scalar[pl.INDEX] = (logical_begin_inline1045_inline2251__ssa_v0 + safe_scale_begin_inline1060_inline2184__ssa_v1_2) // 32
                    t__tile_19: pl.Scalar[pl.INT32] = pl.tensor.read(
                        idx_block_table_flat_inline1044_inline2217__ssa_v0,
                        [batch_idx_inline1058_inline2244__ssa_v0 * table_columns_inline1029_inline2268__ssa_v0 + scale_logical_page_inline1062_inline2250__ssa_v1_2],
                    )
                    scale_block_inline1072_inline2182__ssa_v1_2: pl.Scalar[pl.INDEX] = pl.cast(t__tile_19, pl.INDEX)
                    kv_scale_bytes_inline1064_inline2240__tile_16: pl.Tile[[1, 384], pl.INT8, pl.MemRef(mem_vec_76, pl.const(147456, pl.INT64), 384), pl.Mem.Vec] = pl.tile.gather_row(
                        kv_scale_bytes_inline1064_inline2240__tile_15,
                        idx_kv_cache__rv_v2,
                        [0, scale_page_begin_inline1042_inline2185__ssa_v1_2 * 2],
                        [scale_block_inline1072_inline2182__ssa_v1_2, 4096],
                        [1, 64],
                        transpose=False,
                    )
                    scale_page_begin_inline1042_inline2185__ssa_v2_2: pl.Scalar[pl.INDEX] = 64
                    safe_scale_begin_inline1060_inline2184__ssa_v2_2: pl.Scalar[pl.INDEX] = pl.min(
                        read_begin_inline1055_inline2200__ssa_v0_2 + lane_begin_inline1022_inline2188__ssa_v0_2 + scale_page_begin_inline1042_inline2185__ssa_v2_2,
                        (valid_count_inline1015_inline2253__ssa_v0 - 1) // 32 * 32,
                    )
                    scale_logical_page_inline1062_inline2250__ssa_v2_2: pl.Scalar[pl.INDEX] = (logical_begin_inline1045_inline2251__ssa_v0 + safe_scale_begin_inline1060_inline2184__ssa_v2_2) // 32
                    t__tile_20: pl.Scalar[pl.INT32] = pl.tensor.read(
                        idx_block_table_flat_inline1044_inline2217__ssa_v0,
                        [batch_idx_inline1058_inline2244__ssa_v0 * table_columns_inline1029_inline2268__ssa_v0 + scale_logical_page_inline1062_inline2250__ssa_v2_2],
                    )
                    scale_block_inline1072_inline2182__ssa_v2_2: pl.Scalar[pl.INDEX] = pl.cast(t__tile_20, pl.INDEX)
                    kv_scale_bytes_inline1064_inline2240__tile_17: pl.Tile[[1, 384], pl.INT8, pl.MemRef(mem_vec_76, pl.const(147456, pl.INT64), 384), pl.Mem.Vec] = pl.tile.gather_row(
                        kv_scale_bytes_inline1064_inline2240__tile_16,
                        idx_kv_cache__rv_v2,
                        [0, scale_page_begin_inline1042_inline2185__ssa_v2_2 * 2],
                        [scale_block_inline1072_inline2182__ssa_v2_2, 4096],
                        [1, 64],
                        transpose=False,
                    )
                    scale_page_begin_inline1042_inline2185__ssa_v3_2: pl.Scalar[pl.INDEX] = 96
                    safe_scale_begin_inline1060_inline2184__ssa_v3_2: pl.Scalar[pl.INDEX] = pl.min(
                        read_begin_inline1055_inline2200__ssa_v0_2 + lane_begin_inline1022_inline2188__ssa_v0_2 + scale_page_begin_inline1042_inline2185__ssa_v3_2,
                        (valid_count_inline1015_inline2253__ssa_v0 - 1) // 32 * 32,
                    )
                    scale_logical_page_inline1062_inline2250__ssa_v3_2: pl.Scalar[pl.INDEX] = (logical_begin_inline1045_inline2251__ssa_v0 + safe_scale_begin_inline1060_inline2184__ssa_v3_2) // 32
                    t__tile_21: pl.Scalar[pl.INT32] = pl.tensor.read(
                        idx_block_table_flat_inline1044_inline2217__ssa_v0,
                        [batch_idx_inline1058_inline2244__ssa_v0 * table_columns_inline1029_inline2268__ssa_v0 + scale_logical_page_inline1062_inline2250__ssa_v3_2],
                    )
                    scale_block_inline1072_inline2182__ssa_v3_2: pl.Scalar[pl.INDEX] = pl.cast(t__tile_21, pl.INDEX)
                    kv_scale_bytes_inline1064_inline2240__tile_18: pl.Tile[[1, 384], pl.INT8, pl.MemRef(mem_vec_76, pl.const(147456, pl.INT64), 384), pl.Mem.Vec] = pl.tile.gather_row(
                        kv_scale_bytes_inline1064_inline2240__tile_17,
                        idx_kv_cache__rv_v2,
                        [0, scale_page_begin_inline1042_inline2185__ssa_v3_2 * 2],
                        [scale_block_inline1072_inline2182__ssa_v3_2, 4096],
                        [1, 64],
                        transpose=False,
                    )
                    scale_page_begin_inline1042_inline2185__ssa_v4_2: pl.Scalar[pl.INDEX] = 128
                    safe_scale_begin_inline1060_inline2184__ssa_v4_2: pl.Scalar[pl.INDEX] = pl.min(
                        read_begin_inline1055_inline2200__ssa_v0_2 + lane_begin_inline1022_inline2188__ssa_v0_2 + scale_page_begin_inline1042_inline2185__ssa_v4_2,
                        (valid_count_inline1015_inline2253__ssa_v0 - 1) // 32 * 32,
                    )
                    scale_logical_page_inline1062_inline2250__ssa_v4_2: pl.Scalar[pl.INDEX] = (logical_begin_inline1045_inline2251__ssa_v0 + safe_scale_begin_inline1060_inline2184__ssa_v4_2) // 32
                    t__tile_22: pl.Scalar[pl.INT32] = pl.tensor.read(
                        idx_block_table_flat_inline1044_inline2217__ssa_v0,
                        [batch_idx_inline1058_inline2244__ssa_v0 * table_columns_inline1029_inline2268__ssa_v0 + scale_logical_page_inline1062_inline2250__ssa_v4_2],
                    )
                    scale_block_inline1072_inline2182__ssa_v4_2: pl.Scalar[pl.INDEX] = pl.cast(t__tile_22, pl.INDEX)
                    kv_scale_bytes_inline1064_inline2240__tile_19: pl.Tile[[1, 384], pl.INT8, pl.MemRef(mem_vec_76, pl.const(147456, pl.INT64), 384), pl.Mem.Vec] = pl.tile.gather_row(
                        kv_scale_bytes_inline1064_inline2240__tile_18,
                        idx_kv_cache__rv_v2,
                        [0, scale_page_begin_inline1042_inline2185__ssa_v4_2 * 2],
                        [scale_block_inline1072_inline2182__ssa_v4_2, 4096],
                        [1, 64],
                        transpose=False,
                    )
                    scale_page_begin_inline1042_inline2185__ssa_v5_2: pl.Scalar[pl.INDEX] = 160
                    safe_scale_begin_inline1060_inline2184__ssa_v5_2: pl.Scalar[pl.INDEX] = pl.min(
                        read_begin_inline1055_inline2200__ssa_v0_2 + lane_begin_inline1022_inline2188__ssa_v0_2 + scale_page_begin_inline1042_inline2185__ssa_v5_2,
                        (valid_count_inline1015_inline2253__ssa_v0 - 1) // 32 * 32,
                    )
                    scale_logical_page_inline1062_inline2250__ssa_v5_2: pl.Scalar[pl.INDEX] = (logical_begin_inline1045_inline2251__ssa_v0 + safe_scale_begin_inline1060_inline2184__ssa_v5_2) // 32
                    t__tile_23: pl.Scalar[pl.INT32] = pl.tensor.read(
                        idx_block_table_flat_inline1044_inline2217__ssa_v0,
                        [batch_idx_inline1058_inline2244__ssa_v0 * table_columns_inline1029_inline2268__ssa_v0 + scale_logical_page_inline1062_inline2250__ssa_v5_2],
                    )
                    scale_block_inline1072_inline2182__ssa_v5_2: pl.Scalar[pl.INDEX] = pl.cast(t__tile_23, pl.INDEX)
                    kv_scale_bytes_inline1064_inline2240__tile_20: pl.Tile[[1, 384], pl.INT8, pl.MemRef(mem_vec_76, pl.const(147456, pl.INT64), 384), pl.Mem.Vec] = pl.tile.gather_row(
                        kv_scale_bytes_inline1064_inline2240__tile_19,
                        idx_kv_cache__rv_v2,
                        [0, scale_page_begin_inline1042_inline2185__ssa_v5_2 * 2],
                        [scale_block_inline1072_inline2182__ssa_v5_2, 4096],
                        [1, 64],
                        transpose=False,
                    )
                    kv_scale_inline1061_inline2187__tile_2: pl.Tile[[1, 192], pl.FP16, pl.MemRef(mem_vec_76, pl.const(147456, pl.INT64), 384), pl.Mem.Vec] = pl.tile.reinterpret_view(
                        kv_scale_bytes_inline1064_inline2240__tile_20, dtype=pl.FP16
                    )
                    score_shard_inline1066_inline2180__tile_2: pl.Tile[[64, 192], pl.INT32, pl.Mem.Vec] = pl.tile.tpop_from_aic(split=2)
                    score_fp32_inline1063_inline2179__tile_2: pl.Tile[[64, 192], pl.FP32, pl.MemRef(mem_vec_14, pl.const(98304, pl.INT64), 49152), pl.Mem.Vec] = pl.tile.cast(
                        score_shard_inline1066_inline2180__tile_2, target_type=pl.FP32, mode="none"
                    )
                    pl.system.tfree_to_aic(score_shard_inline1066_inline2180__tile_2, split=2)
                    score_fp32_v1_inline1069_inline2196__tile_2: pl.Tile[[64, 192], pl.FP32, pl.MemRef(mem_vec_14, pl.const(98304, pl.INT64), 49152), pl.Mem.Vec] = pl.tile.maximums(
                        score_fp32_inline1063_inline2179__tile_2, 0.0
                    )
                    score_fp32_v2_inline1073_inline2224__tile_2: pl.Tile[[64, 192], pl.FP32, pl.MemRef(mem_vec_14, pl.const(98304, pl.INT64), 49152), pl.Mem.Vec] = pl.tile.row_expand_mul(
                        score_fp32_v1_inline1069_inline2196__tile_2, head_coefficient_inline1049_inline2265__tile
                    )
                    score_sum_inline1074_inline2178__tile_2: pl.Tile[[1, 192], pl.FP32, pl.MemRef(mem_vec_14, pl.const(98304, pl.INT64), 768), pl.Mem.Vec] = pl.tile.col_sum(
                        score_fp32_v2_inline1073_inline2224__tile_2
                    )
                    score_row_inline1071_inline2177__tile_2: pl.Tile[[1, 192], pl.FP32, pl.MemRef(mem_vec_14, pl.const(98304, pl.INT64), 768), pl.Mem.Vec] = score_sum_inline1074_inline2178__tile_2
                    t__tile_24: pl.Tile[[1, 192], pl.FP32, pl.MemRef(mem_vec_14, pl.const(99072, pl.INT64), 768), pl.Mem.Vec] = pl.tile.cast(
                        kv_scale_inline1061_inline2187__tile_2, target_type=pl.FP32, mode="round"
                    )
                    score_row_v1_inline1011_inline2261__tile_2: pl.Tile[[1, 192], pl.FP32, pl.MemRef(mem_vec_14, pl.const(98304, pl.INT64), 768), pl.Mem.Vec] = pl.tile.mul(
                        score_row_inline1071_inline2177__tile_2, t__tile_24
                    )
                    score_row_id_inline1010_inline2242__ssa_v0_2: pl.Scalar[pl.INDEX] = single_leaf_inline1032_inline2263__ssa_v0 * query_inline1030_inline2255__ssa_v0 + (
                        1 - single_leaf_inline1032_inline2263__ssa_v0
                    ) * (worker_inline1037_inline2245__ssa_v0 * 2 + aiv_id_inline1059_inline2189__ssa_v0_2)
                    score_col_inline1009_inline2254__ssa_v0_2: pl.Scalar[pl.INDEX] = (
                        single_leaf_inline1032_inline2263__ssa_v0 * (read_begin_inline1055_inline2200__ssa_v0_2 + lane_begin_inline1022_inline2188__ssa_v0_2)
                        + (1 - single_leaf_inline1032_inline2263__ssa_v0) * unroll_main_end
                    )
                    if 0 < lane_valid_rows_inline1070_inline2186__ssa_v0_2:
                        score_valid_inline1007_inline2190__tile_2: pl.Tile[
                            [1, 192], pl.FP32, pl.MemRef(mem_vec_14, pl.const(98304, pl.INT64), 768), pl.Mem.Vec, pl.TileView(valid_shape=[1, lane_valid_rows_inline1070_inline2186__ssa_v0_2])
                        ] = pl.tile.set_validshape(score_row_v1_inline1011_inline2261__tile_2, 1, lane_valid_rows_inline1070_inline2186__ssa_v0_2)
                        score_arena_inline1043_inline2235__tile_2: pl.Tensor[[384, 8192], pl.FP32, pl.MemRef("mem_ddr_29", pl.const(0, pl.INT64), 12582912)] = pl.tile.store(
                            score_valid_inline1007_inline2190__tile_2,
                            [score_row_id_inline1010_inline2242__ssa_v0_2, score_col_inline1009_inline2254__ssa_v0_2],
                            score_arena_inline1043_inline2235__rv_v4_main,
                        )
                        score_arena_inline1043_inline2235__phi_v6_2: pl.Tensor[[384, 8192], pl.FP32, pl.MemRef("mem_ddr_37", pl.const(0, pl.INT64), 12582912)] = pl.yield_(
                            score_arena_inline1043_inline2235__tile_2
                        )
                    else:
                        score_arena_inline1043_inline2235__phi_v6_2: pl.Tensor[[384, 8192], pl.FP32, pl.MemRef("mem_ddr_37", pl.const(0, pl.INT64), 12582912)] = pl.yield_(
                            score_arena_inline1043_inline2235__rv_v4_main
                        )
                    score_arena_inline1043_inline2235__rv_v4: pl.Tensor[[384, 8192], pl.FP32, pl.MemRef("mem_ddr_38", pl.const(0, pl.INT64), 12582912)] = pl.yield_(
                        score_arena_inline1043_inline2235__phi_v6_2
                    )
                else:
                    score_arena_inline1043_inline2235__rv_v4: pl.Tensor[[384, 8192], pl.FP32, pl.MemRef("mem_ddr_38", pl.const(0, pl.INT64), 12582912)] = pl.yield_(
                        score_arena_inline1043_inline2235__rv_v4_main
                    )
                if single_leaf_inline1032_inline2263__ssa_v0 == 0:
                    sort_lane_inline1006_inline2176__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_subblock_idx()
                    half_begin_inline1008_inline2267__ssa_v0: pl.Scalar[pl.INDEX] = (
                        logical_begin_inline1045_inline2251__ssa_v0 + sort_lane_inline1006_inline2176__ssa_v0 * lane_span_inline1067_inline2256__ssa_v0
                    )
                    half_valid_inline1005_inline2175__ssa_v0: pl.Scalar[pl.INDEX] = pl.max(
                        pl.min(valid_count_inline1015_inline2253__ssa_v0 - sort_lane_inline1006_inline2176__ssa_v0 * lane_span_inline1067_inline2256__ssa_v0, lane_span_inline1067_inline2256__ssa_v0),
                        0,
                    )
                    half_slot_inline1004_inline2266__ssa_v0: pl.Scalar[pl.INDEX] = (
                        query_inline1030_inline2255__ssa_v0 * 64 + leaf_inline1051_inline2227__ssa_v0 * 2 + sort_lane_inline1006_inline2176__ssa_v0
                    )
                    if 0 < half_valid_inline1005_inline2175__ssa_v0:
                        logical_begin_i32_inline2734__ssa_v0: pl.Scalar[pl.INT32] = pl.cast(half_begin_inline1008_inline2267__ssa_v0, pl.INT32)
                        if half_valid_inline1005_inline2175__ssa_v0 <= 512:
                            t__ci_tmp_v0: pl.Tile[[1, 192], pl.FP32, pl.MemRef(mem_vec_14, pl.const(100352, pl.INT64), 768), pl.Mem.Vec] = pl.tile.create(
                                [1, 192], dtype=pl.FP32, target_memory=pl.Mem.Vec
                            )
                            t__tmp_v217: pl.Tile[[1, 512], pl.INT32, pl.MemRef(mem_vec_14, pl.const(98304, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.ci(
                                pl.const(0, pl.INT32), [1, 512], tmp=t__ci_tmp_v0, dtype=pl.INT32, descending=False
                            )
                            short_indices_inline2753__ssa_v0: pl.Tile[[1, 512], pl.INT32, pl.MemRef(mem_vec_14, pl.const(102400, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.adds(
                                t__tmp_v217, logical_begin_i32_inline2734__ssa_v0
                            )
                            short_raw_inline2743__ssa_v0: pl.Tile[
                                [1, 512], pl.FP32, pl.MemRef(mem_vec_14, pl.const(98304, pl.INT64), 2048), pl.Mem.Vec, pl.TileView(valid_shape=[1, half_valid_inline1005_inline2175__ssa_v0])
                            ] = pl.tile.load(
                                score_arena_inline1043_inline2235__rv_v4,
                                [worker_inline1037_inline2245__ssa_v0 * 2 + sort_lane_inline1006_inline2176__ssa_v0, 0],
                                [1, 512],
                                [1, half_valid_inline1005_inline2175__ssa_v0],
                                target_memory=pl.Mem.Vec,
                            )
                            short_scores_inline2729__ssa_v0: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_14, pl.const(98304, pl.INT64), 2048), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = (
                                pl.tile.fillpad(short_raw_inline2743__ssa_v0, pad_value=pl.PadValue.min)
                            )
                            short_scores_v1_inline2733__ssa_v0: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_14, pl.const(104448, pl.INT64), 2048), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = (
                                pl.tile.maximums(short_scores_inline2729__ssa_v0, -3.4028234663852886e38)
                            )
                            t__tmp_v218: pl.Tile[[1, 512], pl.UINT32, pl.MemRef(mem_vec_14, pl.const(102400, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.reinterpret_view(
                                short_indices_inline2753__ssa_v0, dtype=pl.UINT32
                            )
                            short_pairs_inline2730__ssa_v0: pl.Tile[[1, 1024], pl.FP32, pl.MemRef(mem_vec_14, pl.const(98304, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = (
                                pl.tile.sort32(short_scores_v1_inline2733__ssa_v0, t__tmp_v218)
                            )
                            short_pairs_v1_inline2752__ssa_v0: pl.Tile[[1, 1024], pl.FP32, pl.MemRef(mem_vec_14, pl.const(102400, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = (
                                pl.tile.mrgsort_format1(short_pairs_inline2730__ssa_v0, pl.const(64, pl.INT32))
                            )
                            short_pairs_v2_inline2736__ssa_v0: pl.Tile[[1, 1024], pl.FP32, pl.MemRef(mem_vec_14, pl.const(98304, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = (
                                pl.tile.mrgsort_format1(short_pairs_v1_inline2752__ssa_v0, pl.const(256, pl.INT32))
                            )
                            pair_arena_inline1035_inline2211__store: pl.Tensor[[24576, 1024], pl.FP32, pl.MemRef("mem_ddr_8", pl.const(0, pl.INT64), 100663296)] = pl.tile.store(
                                short_pairs_v2_inline2736__ssa_v0, [half_slot_inline1004_inline2266__ssa_v0, 0], pair_arena_inline1035_inline2211__ssa_v0
                            )
                        else:
                            if half_valid_inline1005_inline2175__ssa_v0 <= 1024:
                                t__ci_tmp_v1: pl.Tile[[1, 192], pl.FP32, pl.MemRef(mem_vec_14, pl.const(102400, pl.INT64), 768), pl.Mem.Vec] = pl.tile.create(
                                    [1, 192], dtype=pl.FP32, target_memory=pl.Mem.Vec
                                )
                                t__tmp_v219: pl.Tile[[1, 1024], pl.INT32, pl.MemRef(mem_vec_14, pl.const(98304, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.ci(
                                    pl.const(0, pl.INT32), [1, 1024], tmp=t__ci_tmp_v1, dtype=pl.INT32, descending=False
                                )
                                small_indices_inline2726__ssa_v0: pl.Tile[[1, 1024], pl.INT32, pl.MemRef(mem_vec_14, pl.const(106496, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.adds(
                                    t__tmp_v219, logical_begin_i32_inline2734__ssa_v0
                                )
                                small_raw_inline2725__ssa_v0: pl.Tile[
                                    [1, 1024], pl.FP32, pl.MemRef(mem_vec_14, pl.const(98304, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(valid_shape=[1, half_valid_inline1005_inline2175__ssa_v0])
                                ] = pl.tile.load(
                                    score_arena_inline1043_inline2235__rv_v4,
                                    [worker_inline1037_inline2245__ssa_v0 * 2 + sort_lane_inline1006_inline2176__ssa_v0, 0],
                                    [1, 1024],
                                    [1, half_valid_inline1005_inline2175__ssa_v0],
                                    target_memory=pl.Mem.Vec,
                                )
                                small_scores_inline2727__ssa_v0: pl.Tile[[1, 1024], pl.FP32, pl.MemRef(mem_vec_14, pl.const(98304, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = (
                                    pl.tile.fillpad(small_raw_inline2725__ssa_v0, pad_value=pl.PadValue.min)
                                )
                                small_scores_v1_inline2747__ssa_v0: pl.Tile[
                                    [1, 1024], pl.FP32, pl.MemRef(mem_vec_14, pl.const(110592, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)
                                ] = pl.tile.maximums(small_scores_inline2727__ssa_v0, -3.4028234663852886e38)
                                t__tmp_v220: pl.Tile[[1, 1024], pl.UINT32, pl.MemRef(mem_vec_14, pl.const(106496, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.reinterpret_view(
                                    small_indices_inline2726__ssa_v0, dtype=pl.UINT32
                                )
                                small_pairs_inline2728__ssa_v0: pl.Tile[[1, 2048], pl.FP32, pl.MemRef(mem_vec_14, pl.const(98304, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = (
                                    pl.tile.sort32(small_scores_v1_inline2747__ssa_v0, t__tmp_v220)
                                )
                                small_pairs_v1_inline2737__ssa_v0: pl.Tile[
                                    [1, 2048], pl.FP32, pl.MemRef(mem_vec_14, pl.const(106496, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)
                                ] = pl.tile.mrgsort_format1(small_pairs_inline2728__ssa_v0, pl.const(64, pl.INT32))
                                small_pairs_v2_inline2739__ssa_v0: pl.Tile[[1, 2048], pl.FP32, pl.MemRef(mem_vec_14, pl.const(98304, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = (
                                    pl.tile.mrgsort_format1(small_pairs_v1_inline2737__ssa_v0, pl.const(256, pl.INT32))
                                )
                                small_left_inline2731__ssa_v0: pl.Tile[[1, 1024], pl.FP32, pl.MemRef(mem_vec_14, pl.const(98304, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = (
                                    pl.tile.slice(small_pairs_v2_inline2739__ssa_v0, [1, 1024], [0, 0])
                                )
                                small_right_inline2740__ssa_v0: pl.Tile[[1, 1024], pl.FP32, pl.MemRef(mem_vec_14, pl.const(102400, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = (
                                    pl.tile.slice(small_pairs_v2_inline2739__ssa_v0, [1, 1024], [0, 1024])
                                )
                                small_tmp_inline2741__ssa_v0: pl.Tile[[1, 2048], pl.FP32, pl.MemRef(mem_vec_14, pl.const(106496, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.create(
                                    [1, 2048], dtype=pl.FP32, target_memory=pl.Mem.Vec
                                )
                                small_merged_inline2742__ssa_v0: pl.Tile[[1, 2048], pl.FP32, pl.MemRef(mem_vec_14, pl.const(114688, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = (
                                    pl.tile.mrgsort_format2(small_left_inline2731__ssa_v0, small_right_inline2740__ssa_v0, small_tmp_inline2741__ssa_v0, exhausted=False)
                                )
                                small_top_inline2732__ssa_v0: pl.Tile[[1, 1024], pl.FP32, pl.MemRef(mem_vec_14, pl.const(114688, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = (
                                    pl.tile.slice(small_merged_inline2742__ssa_v0, [1, 1024], [0, 0])
                                )
                                pair_arena_inline1035_inline2211__store_v0: pl.Tensor[[24576, 1024], pl.FP32, pl.MemRef("mem_ddr_8", pl.const(0, pl.INT64), 100663296)] = pl.tile.store(
                                    small_top_inline2732__ssa_v0, [half_slot_inline1004_inline2266__ssa_v0, 0], pair_arena_inline1035_inline2211__ssa_v0
                                )
                            else:
                                if half_valid_inline1005_inline2175__ssa_v0 <= 2048:
                                    t__ci_tmp_v2: pl.Tile[[1, 192], pl.FP32, pl.MemRef(mem_vec_14, pl.const(106496, pl.INT64), 768), pl.Mem.Vec] = pl.tile.create(
                                        [1, 192], dtype=pl.FP32, target_memory=pl.Mem.Vec
                                    )
                                    t__tmp_v221: pl.Tile[[1, 2048], pl.INT32, pl.MemRef(mem_vec_14, pl.const(98304, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.ci(
                                        pl.const(0, pl.INT32), [1, 2048], tmp=t__ci_tmp_v2, dtype=pl.INT32, descending=False
                                    )
                                    leaf_indices_inline2735__ssa_v0: pl.Tile[[1, 2048], pl.INT32, pl.MemRef(mem_vec_14, pl.const(114688, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.adds(
                                        t__tmp_v221, logical_begin_i32_inline2734__ssa_v0
                                    )
                                    leaf_scores_raw_inline2744__ssa_v0: pl.Tile[
                                        [1, 2048], pl.FP32, pl.MemRef(mem_vec_14, pl.const(98304, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[1, half_valid_inline1005_inline2175__ssa_v0])
                                    ] = pl.tile.load(
                                        score_arena_inline1043_inline2235__rv_v4,
                                        [worker_inline1037_inline2245__ssa_v0 * 2 + sort_lane_inline1006_inline2176__ssa_v0, 0],
                                        [1, 2048],
                                        [1, half_valid_inline1005_inline2175__ssa_v0],
                                        target_memory=pl.Mem.Vec,
                                    )
                                    leaf_scores_inline2745__ssa_v0: pl.Tile[
                                        [1, 2048], pl.FP32, pl.MemRef(mem_vec_14, pl.const(98304, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)
                                    ] = pl.tile.fillpad(leaf_scores_raw_inline2744__ssa_v0, pad_value=pl.PadValue.min)
                                    leaf_scores_v1_inline2746__ssa_v0: pl.Tile[
                                        [1, 2048], pl.FP32, pl.MemRef(mem_vec_14, pl.const(122880, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)
                                    ] = pl.tile.maximums(leaf_scores_inline2745__ssa_v0, -3.4028234663852886e38)
                                    t__tmp_v222: pl.Tile[[1, 2048], pl.UINT32, pl.MemRef(mem_vec_14, pl.const(114688, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.reinterpret_view(
                                        leaf_indices_inline2735__ssa_v0, dtype=pl.UINT32
                                    )
                                    pairs_inline2738__ssa_v0: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_14, pl.const(98304, pl.INT64), 16384), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = (
                                        pl.tile.sort32(leaf_scores_v1_inline2746__ssa_v0, t__tmp_v222)
                                    )
                                    pairs_v1_inline2748__ssa_v0: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_14, pl.const(114688, pl.INT64), 16384), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = (
                                        pl.tile.mrgsort_format1(pairs_inline2738__ssa_v0, pl.const(64, pl.INT32))
                                    )
                                    pairs_v2_inline2750__ssa_v0: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_14, pl.const(98304, pl.INT64), 16384), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = (
                                        pl.tile.mrgsort_format1(pairs_v1_inline2748__ssa_v0, pl.const(256, pl.INT32))
                                    )
                                    pairs_v3_inline2754__ssa_v0: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_14, pl.const(114688, pl.INT64), 16384), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = (
                                        pl.tile.mrgsort_format1(pairs_v2_inline2750__ssa_v0, pl.const(1024, pl.INT32))
                                    )
                                    t__tmp_v223: pl.Tile[[1, 1024], pl.FP32, pl.MemRef(mem_vec_14, pl.const(114688, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(
                                        pairs_v3_inline2754__ssa_v0, [1, 1024], [0, 0]
                                    )
                                    pair_arena_inline1035_inline2211__store_v1: pl.Tensor[[24576, 1024], pl.FP32, pl.MemRef("mem_ddr_8", pl.const(0, pl.INT64), 100663296)] = pl.tile.store(
                                        t__tmp_v223, [half_slot_inline1004_inline2266__ssa_v0, 0], pair_arena_inline1035_inline2211__ssa_v0
                                    )
                                else:
                                    t__ci_tmp_v3: pl.Tile[[1, 192], pl.FP32, pl.MemRef(mem_vec_14, pl.const(114688, pl.INT64), 768), pl.Mem.Vec] = pl.tile.create(
                                        [1, 192], dtype=pl.FP32, target_memory=pl.Mem.Vec
                                    )
                                    t__tmp_v224: pl.Tile[[1, 4096], pl.INT32, pl.MemRef(mem_vec_14, pl.const(98304, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.ci(
                                        pl.const(0, pl.INT32), [1, 4096], tmp=t__ci_tmp_v3, dtype=pl.INT32, descending=False
                                    )
                                    medium_indices_inline2749__ssa_v0: pl.Tile[[1, 4096], pl.INT32, pl.MemRef(mem_vec_14, pl.const(131072, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.adds(
                                        t__tmp_v224, logical_begin_i32_inline2734__ssa_v0
                                    )
                                    medium_scores_raw_inline2724__ssa_v0: pl.Tile[
                                        [1, 4096], pl.FP32, pl.MemRef(mem_vec_14, pl.const(98304, pl.INT64), 16384), pl.Mem.Vec, pl.TileView(valid_shape=[1, half_valid_inline1005_inline2175__ssa_v0])
                                    ] = pl.tile.load(
                                        score_arena_inline1043_inline2235__rv_v4,
                                        [worker_inline1037_inline2245__ssa_v0 * 2 + sort_lane_inline1006_inline2176__ssa_v0, 0],
                                        [1, 4096],
                                        [1, half_valid_inline1005_inline2175__ssa_v0],
                                        target_memory=pl.Mem.Vec,
                                    )
                                    medium_scores_inline2723__ssa_v0: pl.Tile[
                                        [1, 4096], pl.FP32, pl.MemRef(mem_vec_14, pl.const(98304, pl.INT64), 16384), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)
                                    ] = pl.tile.fillpad(medium_scores_raw_inline2724__ssa_v0, pad_value=pl.PadValue.min)
                                    medium_scores_v1_inline2722__ssa_v0: pl.Tile[
                                        [1, 4096], pl.FP32, pl.MemRef(mem_vec_76, pl.const(147456, pl.INT64), 16384), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)
                                    ] = pl.tile.maximums(medium_scores_inline2723__ssa_v0, -3.4028234663852886e38)
                                    t__tmp_v225: pl.Tile[[1, 4096], pl.UINT32, pl.MemRef(mem_vec_14, pl.const(131072, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.reinterpret_view(
                                        medium_indices_inline2749__ssa_v0, dtype=pl.UINT32
                                    )
                                    medium_pairs_inline2721__ssa_v0: pl.Tile[
                                        [1, 8192], pl.FP32, pl.MemRef(mem_vec_14, pl.const(98304, pl.INT64), 32768), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)
                                    ] = pl.tile.sort32(medium_scores_v1_inline2722__ssa_v0, t__tmp_v225)
                                    medium_pairs_v1_inline2720__ssa_v0: pl.Tile[
                                        [1, 8192], pl.FP32, pl.MemRef(mem_vec_76, pl.const(147456, pl.INT64), 32768), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)
                                    ] = pl.tile.mrgsort_format1(medium_pairs_inline2721__ssa_v0, pl.const(64, pl.INT32))
                                    medium_pairs_v2_inline2718__ssa_v0: pl.Tile[
                                        [1, 8192], pl.FP32, pl.MemRef(mem_vec_14, pl.const(98304, pl.INT64), 32768), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)
                                    ] = pl.tile.mrgsort_format1(medium_pairs_v1_inline2720__ssa_v0, pl.const(256, pl.INT32))
                                    medium_pairs_v3_inline2717__ssa_v0: pl.Tile[
                                        [1, 8192], pl.FP32, pl.MemRef(mem_vec_76, pl.const(147456, pl.INT64), 32768), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)
                                    ] = pl.tile.mrgsort_format1(medium_pairs_v2_inline2718__ssa_v0, pl.const(1024, pl.INT32))
                                    medium_left_inline2751__ssa_v0: pl.Tile[
                                        [1, 1024], pl.FP32, pl.MemRef(mem_vec_76, pl.const(147456, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)
                                    ] = pl.tile.slice(medium_pairs_v3_inline2717__ssa_v0, [1, 1024], [0, 0])
                                    medium_right_inline2716__ssa_v0: pl.Tile[
                                        [1, 1024], pl.FP32, pl.MemRef(mem_vec_76, pl.const(163840, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)
                                    ] = pl.tile.slice(medium_pairs_v3_inline2717__ssa_v0, [1, 1024], [0, 4096])
                                    medium_tmp_inline2719__ssa_v0: pl.Tile[[1, 2048], pl.FP32, pl.MemRef(mem_vec_14, pl.const(98304, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.create(
                                        [1, 2048], dtype=pl.FP32, target_memory=pl.Mem.Vec
                                    )
                                    medium_merged_inline2715__ssa_v0: pl.Tile[
                                        [1, 2048], pl.FP32, pl.MemRef(mem_vec_14, pl.const(106496, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)
                                    ] = pl.tile.mrgsort_format2(medium_left_inline2751__ssa_v0, medium_right_inline2716__ssa_v0, medium_tmp_inline2719__ssa_v0, exhausted=False)
                                    t__tmp_v226: pl.Tile[[1, 1024], pl.FP32, pl.MemRef(mem_vec_14, pl.const(106496, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(
                                        medium_merged_inline2715__ssa_v0, [1, 1024], [0, 0]
                                    )
                                    pair_arena_inline1035_inline2211__store_v2: pl.Tensor[[24576, 1024], pl.FP32, pl.MemRef("mem_ddr_8", pl.const(0, pl.INT64), 100663296)] = pl.tile.store(
                                        t__tmp_v226, [half_slot_inline1004_inline2266__ssa_v0, 0], pair_arena_inline1035_inline2211__ssa_v0
                                    )
                    else:
                        empty_pairs_inline1040_inline2174__ssa_v0: pl.Tile[[1, 1024], pl.FP32, pl.MemRef(mem_vec_14, pl.const(98304, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.full(
                            [1, 1024], dtype=pl.FP32, value=-3.4028234663852886e38
                        )
                        pair_arena_inline1035_inline2211__store_v3: pl.Tensor[[24576, 1024], pl.FP32, pl.MemRef("mem_ddr_8", pl.const(0, pl.INT64), 100663296)] = pl.tile.store(
                            empty_pairs_inline1040_inline2174__ssa_v0, [half_slot_inline1004_inline2266__ssa_v0, 0], pair_arena_inline1035_inline2211__ssa_v0
                        )
                score_arena_inline1043_inline2235__phi_v7: pl.Tensor[[384, 8192], pl.FP32, pl.MemRef("mem_ddr_82", pl.const(0, pl.INT64), 12582912)] = pl.yield_(
                    score_arena_inline1043_inline2235__rv_v4
                )
            else:
                score_arena_inline1043_inline2235__phi_v7: pl.Tensor[[384, 8192], pl.FP32, pl.MemRef("mem_ddr_82", pl.const(0, pl.INT64), 12582912)] = pl.yield_(
                    score_arena_inline1043_inline2235__iter_v1
                )
            score_arena_inline1043_inline2235__rv_v2: pl.Tensor[[384, 8192], pl.FP32, pl.MemRef("mem_ddr_83", pl.const(0, pl.INT64), 12582912)] = pl.yield_(score_arena_inline1043_inline2235__phi_v7)
        return score_arena_inline1043_inline2235__ssa_v0, pair_arena_inline1035_inline2211__ssa_v0

    @pl.function(type=pl.FunctionType.Group, level=pl.Level.CORE_GROUP, role=pl.Role.SubWorker)
    def indexer_score_topk_leaf(
        position_ids__ssa_v0: pl.Tensor[[T_DYN], pl.INT32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)],
        kv_seq_lens__ssa_v0: pl.Tensor[[B_DYN], pl.INT32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)],
        score_arena_inline1043_inline2235__ssa_v0: pl.InOut[pl.Tensor[[384, 8192], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 12582912)]],
        qr_hadamard_i8_inline527__rv_v2: pl.Tensor[[24576, 128], pl.INT8, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 3145728)],
        qr_hadamard_scale_dq_inline525__rv_v2: pl.Tensor[[24576, 1], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 98304)],
        weights_inline2220__ssa_v1: pl.Tensor[[384, 64], pl.FP32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 98304)],
        idx_block_table_flat_inline1044_inline2217__ssa_v0: pl.Tensor[[idx_table_len_inline1025_inline2222__ssa_v0], pl.INT32, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 0)],
        table_columns_inline1029_inline2268__ssa_v0: pl.Scalar[pl.INDEX],
        idx_kv_cache__rv_v2: pl.Tensor[[IDX_CACHE_BLOCK_NUM_DYN, INDEXER_PAGE_BYTES_DYN], pl.INT8, pl.MemRef("mem_ddr_7", pl.const(0, pl.INT64), 0)],
        pair_arena_inline1035_inline2211__ssa_v0: pl.Out[pl.Tensor[[24576, 1024], pl.FP32, pl.MemRef("mem_ddr_8", pl.const(0, pl.INT64), 100663296)]],
        __gm_pipe_buffer: pl.Out[pl.Tensor[[1], pl.FP32, pl.MemRef("mem_ddr_9", pl.const(0, pl.INT64), 4)]],
    ) -> tuple[pl.Tensor[[384, 8192], pl.FP32], pl.Tensor[[24576, 1024], pl.FP32]]:
        pl.func_attr({"slot_num": 1, "mx_tensor_views_blocked": True, "split_aiv": True})
        self.indexer_score_topk_leaf_aic(
            position_ids__ssa_v0,
            kv_seq_lens__ssa_v0,
            score_arena_inline1043_inline2235__ssa_v0,
            qr_hadamard_i8_inline527__rv_v2,
            qr_hadamard_scale_dq_inline525__rv_v2,
            weights_inline2220__ssa_v1,
            idx_block_table_flat_inline1044_inline2217__ssa_v0,
            table_columns_inline1029_inline2268__ssa_v0,
            idx_kv_cache__rv_v2,
            pair_arena_inline1035_inline2211__ssa_v0,
            __gm_pipe_buffer,
            attrs={
                "arg_directions": [
                    pl.adir.input,
                    pl.adir.input,
                    pl.adir.inout,
                    pl.adir.input,
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
            score_arena_inline1043_inline2235__ssa_v0,
            qr_hadamard_i8_inline527__rv_v2,
            qr_hadamard_scale_dq_inline525__rv_v2,
            weights_inline2220__ssa_v1,
            idx_block_table_flat_inline1044_inline2217__ssa_v0,
            table_columns_inline1029_inline2268__ssa_v0,
            idx_kv_cache__rv_v2,
            pair_arena_inline1035_inline2211__ssa_v0,
            __gm_pipe_buffer,
            attrs={
                "arg_directions": [pl.adir.input, pl.adir.input, pl.adir.inout, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.scalar, pl.adir.input, pl.adir.inout, pl.adir.inout]
            },
        )
        return score_arena_inline1043_inline2235__ssa_v0, pair_arena_inline1035_inline2211__ssa_v0

    @pl.function(type=pl.FunctionType.Spmd)
    def indexer_score_topk_leaf_spmd(
        self,
        position_ids__ssa_v0: pl.Tensor[[T_DYN], pl.INT32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)],
        kv_seq_lens__ssa_v0: pl.Tensor[[B_DYN], pl.INT32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)],
        score_arena_inline1043_inline2235__ssa_v0: pl.InOut[pl.Tensor[[384, 8192], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 12582912)]],
        qr_hadamard_i8_inline527__rv_v2: pl.Tensor[[24576, 128], pl.INT8, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 3145728)],
        qr_hadamard_scale_dq_inline525__rv_v2: pl.Tensor[[24576, 1], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 98304)],
        weights_inline2220__ssa_v1: pl.Tensor[[384, 64], pl.FP32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 98304)],
        idx_block_table_flat_inline1044_inline2217__ssa_v0: pl.Tensor[[idx_table_len_inline1025_inline2222__ssa_v0], pl.INT32, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 0)],
        table_columns_inline1029_inline2268__ssa_v0: pl.Scalar[pl.INDEX],
        idx_kv_cache__rv_v3: pl.Tensor[[IDX_CACHE_BLOCK_NUM_DYN, INDEXER_PAGE_BYTES_DYN], pl.INT8, pl.MemRef("mem_ddr_7", pl.const(0, pl.INT64), 0)],
        pair_arena_inline1035_inline2211__ssa_v0: pl.Out[pl.Tensor[[24576, 1024], pl.FP32, pl.MemRef("mem_ddr_8", pl.const(0, pl.INT64), 100663296)]],
        __gm_pipe_buffer: pl.Out[pl.Tensor[[1], pl.FP32, pl.MemRef("mem_ddr_9", pl.const(0, pl.INT64), 4)]],
    ) -> tuple[pl.Tensor[[384, 8192], pl.FP32], pl.Tensor[[24576, 1024], pl.FP32]]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        ret__tmp_v0: pl.Tuple[pl.Tensor[[384, 8192], pl.FP32], pl.Tensor[[24576, 1024], pl.FP32]] = self.indexer_score_topk_leaf(
            position_ids__ssa_v0,
            kv_seq_lens__ssa_v0,
            score_arena_inline1043_inline2235__ssa_v0,
            qr_hadamard_i8_inline527__rv_v2,
            qr_hadamard_scale_dq_inline525__rv_v2,
            weights_inline2220__ssa_v1,
            idx_block_table_flat_inline1044_inline2217__ssa_v0,
            table_columns_inline1029_inline2268__ssa_v0,
            idx_kv_cache__rv_v3,
            pair_arena_inline1035_inline2211__ssa_v0,
            __gm_pipe_buffer,
            attrs={
                "arg_directions": [
                    pl.adir.input,
                    pl.adir.input,
                    pl.adir.inout,
                    pl.adir.input,
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
        score_arena_inline1043_inline2235__rv_v2: pl.Tensor[[384, 8192], pl.FP32, pl.MemRef("mem_ddr_10", pl.const(0, pl.INT64), 12582912)] = ret__tmp_v0[0]
        pair_arena_inline1035_inline2211__ssa_v1: pl.Tensor[[24576, 1024], pl.FP32, pl.MemRef("mem_ddr_11", pl.const(0, pl.INT64), 100663296)] = ret__tmp_v0[1]
        return score_arena_inline1043_inline2235__ssa_v0, pair_arena_inline1035_inline2211__ssa_v0

    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def kv_and_cache_write(
        compact_rows_inline2073__ssa_v0: pl.Scalar[pl.INDEX],
        kv_final_inline2068__rv_v2: pl.Tensor[[384, 128], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 196608)],
        idx_kv_scale_values_inline2059__ssa_v0: pl.Out[pl.Tensor[[384, 1], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 1536)]],
        idx_kv_cache__ssa_v0: pl.Out[pl.Tensor[[IDX_CACHE_BLOCK_NUM_DYN, INDEXER_PAGE_BYTES_DYN], pl.INT8, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)]],
        kv_flat_inline2077__ssa_v0: pl.Out[pl.Tensor[[T_DYN, 128], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)]],
        position_ids__ssa_v0: pl.Tensor[[T_DYN], pl.INT32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 0)],
        idx_slot_mapping__ssa_v0: pl.Tensor[[T_DYN], pl.INT64, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 0)],
    ) -> tuple[pl.Tensor[[384, 1], pl.FP32], pl.Tensor[[IDX_CACHE_BLOCK_NUM_DYN, INDEXER_PAGE_BYTES_DYN], pl.INT8]]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        mem_vec_6: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 8192)
        mem_vec_12: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 8192)
        mem_vec_13: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 8192)
        mem_vec_14: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 64)
        wr_blk_inline2083__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        wr_b0_inline2082__ssa_v0: pl.Scalar[pl.INDEX] = wr_blk_inline2083__ssa_v0 * 16
        wr_blk_rows_inline2089__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(compact_rows_inline2073__ssa_v0 - wr_b0_inline2082__ssa_v0, 16)
        t__tile: pl.Tile[[16, 128], pl.FP32, pl.MemRef(mem_vec_6, pl.const(16448, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.load(
            kv_final_inline2068__rv_v2, [wr_b0_inline2082__ssa_v0, 0], [16, 128], [16, 128], target_memory=pl.Mem.Vec
        )
        t__tile_1: pl.Tile[[16, 128], pl.BF16, pl.MemRef(mem_vec_12, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(t__tile, target_type=pl.BF16, mode="rint")
        kv_blk_f32_inline2086__tile: pl.Tile[[16, 128], pl.FP32, pl.MemRef(mem_vec_6, pl.const(16448, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.cast(t__tile_1, target_type=pl.FP32, mode="round")
        t__tile_2: pl.Tile[[16, 128], pl.FP32, pl.MemRef(mem_vec_6, pl.const(16448, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.muls(kv_blk_f32_inline2086__tile, 0.088388347648318447)
        t__tile_3: pl.Tile[[16, 128], pl.BF16, pl.MemRef(mem_vec_12, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(t__tile_2, target_type=pl.BF16, mode="rint")
        kv_blk_f32_v1_inline2066__tile: pl.Tile[[16, 128], pl.FP32, pl.MemRef(mem_vec_6, pl.const(16448, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.cast(t__tile_3, target_type=pl.FP32, mode="round")
        t__tile_4: pl.Tile[[16, 128], pl.FP32, pl.MemRef(mem_vec_12, pl.const(0, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.abs(kv_blk_f32_v1_inline2066__tile)
        tmp_tile: pl.Tile[[16, 128], pl.FP32, pl.MemRef(mem_vec_13, pl.const(8192, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.create([16, 128], dtype=pl.FP32, target_memory=pl.Mem.Vec)
        t__tile_5: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_14, pl.const(16384, pl.INT64), 64), pl.Mem.Vec] = pl.tile.row_max(t__tile_4, tmp_tile)
        kv_amax_inline2075__tile: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_14, pl.const(16384, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(t__tile_5, [1, 16])
        t__tile_6: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_12, pl.const(0, pl.INT64), 64), pl.Mem.Vec] = pl.tile.full([1, 16], dtype=pl.FP32, value=0.0001)
        kv_amax_v1_inline2078__tile: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_12, pl.const(0, pl.INT64), 64), pl.Mem.Vec] = pl.tile.maximum(kv_amax_inline2075__tile, t__tile_6)
        t__tile_7: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_13, pl.const(8192, pl.INT64), 64), pl.Mem.Vec] = pl.tile.full([1, 16], dtype=pl.FP32, value=127.0)
        kv_scale_q_row_inline2090__tile: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_13, pl.const(8192, pl.INT64), 64), pl.Mem.Vec] = pl.tile.div(t__tile_7, kv_amax_v1_inline2078__tile)
        t__tile_8: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_12, pl.const(0, pl.INT64), 64), pl.Mem.Vec] = pl.tile.recip(kv_scale_q_row_inline2090__tile)
        kv_scale_dq_col_inline2093__tile: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_12, pl.const(0, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(t__tile_8, [16, 1])
        kv_scale_q_col_inline2094__tile: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_13, pl.const(8192, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(kv_scale_q_row_inline2090__tile, [16, 1])
        idx_kv_scale_values_inline2059__tile: pl.Tensor[[384, 1], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 1536)] = pl.tile.store(
            kv_scale_dq_col_inline2093__tile, [wr_b0_inline2082__ssa_v0, 0], idx_kv_scale_values_inline2059__ssa_v0
        )
        kv_scaled_inline2063__tile: pl.Tile[[16, 128], pl.FP32, pl.MemRef(mem_vec_12, pl.const(0, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.row_expand_mul(
            kv_blk_f32_v1_inline2066__tile, kv_scale_q_col_inline2094__tile
        )
        kv_i32_inline2088__tile: pl.Tile[[16, 128], pl.INT32, pl.MemRef(mem_vec_12, pl.const(0, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.cast(
            kv_scaled_inline2063__tile, target_type=pl.INT32, mode="rint"
        )
        kv_half_inline2074__tile: pl.Tile[[16, 128], pl.FP16, pl.MemRef(mem_vec_12, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(kv_i32_inline2088__tile, target_type=pl.FP16, mode="round")
        kv_i8_blk_inline2070__tile: pl.Tile[[16, 128], pl.INT8, pl.MemRef(mem_vec_12, pl.const(0, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(
            kv_half_inline2074__tile, target_type=pl.INT8, mode="trunc"
        )
        for inner_inline2067__idx_v0, (idx_kv_cache__iter_v1, kv_flat_inline2077__iter_v1) in pl.range(wr_blk_rows_inline2089__ssa_v0, init_values=(idx_kv_cache__ssa_v0, kv_flat_inline2077__ssa_v0)):
            compact_token_inline2091__ssa_v0: pl.Scalar[pl.INDEX] = wr_b0_inline2082__ssa_v0 + inner_inline2067__idx_v0
            request_inline2085__ssa_v0: pl.Scalar[pl.INDEX] = compact_token_inline2091__ssa_v0 // 2
            first_pos_inline2065__tile: pl.Scalar[pl.INT32] = pl.tensor.read(position_ids__ssa_v0, [request_inline2085__ssa_v0 * 6])
            local_token_inline2057__ssa_v0: pl.Scalar[pl.INDEX] = compact_token_inline2091__ssa_v0 % 2 * 4 - pl.cast(first_pos_inline2065__tile, pl.INDEX) % 4 + 3
            if local_token_inline2057__ssa_v0 < 6:
                token_inline2071__ssa_v0: pl.Scalar[pl.INDEX] = request_inline2085__ssa_v0 * 6 + local_token_inline2057__ssa_v0
                cache_row_i64_inline2056__tile: pl.Scalar[pl.INT64] = pl.tensor.read(idx_slot_mapping__ssa_v0, [token_inline2071__ssa_v0])
                if 0 <= cache_row_i64_inline2056__tile:
                    cache_row_inline2055__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(cache_row_i64_inline2056__tile, pl.INDEX)
                    t__tile_9: pl.Tile[[1, 128], pl.FP32, pl.MemRef(mem_vec_6, pl.const(16448, pl.INT64), 512), pl.Mem.Vec] = pl.tile.slice(
                        kv_blk_f32_v1_inline2066__tile, [1, 128], [inner_inline2067__idx_v0, 0]
                    )
                    kv_flat_inline2077__tile: pl.Tensor[[T_DYN, 128], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                        t__tile_9, [token_inline2071__ssa_v0, 0], kv_flat_inline2077__iter_v1
                    )
                    cache_page_inline2062__ssa_v0: pl.Scalar[pl.INDEX] = cache_row_inline2055__ssa_v0 // 32
                    key_begin_inline2054__ssa_v0: pl.Scalar[pl.INDEX] = cache_row_inline2055__ssa_v0 % 32 * 128
                    t__tile_10: pl.Tile[[1, 128], pl.INT8, pl.MemRef(mem_vec_12, pl.const(0, pl.INT64), 128), pl.Mem.Vec] = pl.tile.slice(
                        kv_i8_blk_inline2070__tile, [1, 128], [inner_inline2067__idx_v0, 0]
                    )
                    idx_kv_cache__tile: pl.Tensor[[IDX_CACHE_BLOCK_NUM_DYN, INDEXER_PAGE_BYTES_DYN], pl.INT8, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                        t__tile_10, [cache_page_inline2062__ssa_v0, key_begin_inline2054__ssa_v0], idx_kv_cache__iter_v1
                    )
                    idx_kv_cache__phi_v4, kv_flat_inline2077__phi_v4 = pl.yield_(idx_kv_cache__tile, kv_flat_inline2077__tile)
                else:
                    idx_kv_cache__phi_v4, kv_flat_inline2077__phi_v4 = pl.yield_(idx_kv_cache__iter_v1, kv_flat_inline2077__iter_v1)
                idx_kv_cache__phi_v5, kv_flat_inline2077__phi_v5 = pl.yield_(idx_kv_cache__phi_v4, kv_flat_inline2077__phi_v4)
            else:
                idx_kv_cache__phi_v5, kv_flat_inline2077__phi_v5 = pl.yield_(idx_kv_cache__iter_v1, kv_flat_inline2077__iter_v1)
            idx_kv_cache__rv_v2, kv_flat_inline2077__rv_v2 = pl.yield_(idx_kv_cache__phi_v5, kv_flat_inline2077__phi_v5)
        return idx_kv_scale_values_inline2059__ssa_v0, idx_kv_cache__ssa_v0

    @pl.function(type=pl.FunctionType.Spmd)
    def kv_and_cache_write_spmd(
        self,
        compact_rows_inline2073__ssa_v0: pl.Scalar[pl.INDEX],
        kv_final_inline2068__rv_v2: pl.Tensor[[384, 128], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 196608)],
        idx_kv_scale_values_inline2059__ssa_v0: pl.Out[pl.Tensor[[384, 1], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 1536)]],
        idx_kv_cache__ssa_v0: pl.Out[pl.Tensor[[IDX_CACHE_BLOCK_NUM_DYN, INDEXER_PAGE_BYTES_DYN], pl.INT8, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)]],
        kv_flat_inline2077__ssa_v0: pl.Out[pl.Tensor[[T_DYN, 128], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)]],
        position_ids__ssa_v0: pl.Tensor[[T_DYN], pl.INT32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 0)],
        idx_slot_mapping__ssa_v0: pl.Tensor[[T_DYN], pl.INT64, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 0)],
    ) -> tuple[pl.Tensor[[384, 1], pl.FP32], pl.Tensor[[IDX_CACHE_BLOCK_NUM_DYN, INDEXER_PAGE_BYTES_DYN], pl.INT8]]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        ret__tmp_v0: pl.Tuple[pl.Tensor[[384, 1], pl.FP32], pl.Tensor[[IDX_CACHE_BLOCK_NUM_DYN, INDEXER_PAGE_BYTES_DYN], pl.INT8]] = self.kv_and_cache_write(
            compact_rows_inline2073__ssa_v0,
            kv_final_inline2068__rv_v2,
            idx_kv_scale_values_inline2059__ssa_v0,
            idx_kv_cache__ssa_v0,
            kv_flat_inline2077__ssa_v0,
            position_ids__ssa_v0,
            idx_slot_mapping__ssa_v0,
            attrs={"arg_directions": [pl.adir.scalar, pl.adir.input, pl.adir.output_existing, pl.adir.output_existing, pl.adir.output_existing, pl.adir.input, pl.adir.input]},
        )
        idx_kv_scale_values_inline2059__ssa_v1: pl.Tensor[[384, 1], pl.FP32, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 1536)] = ret__tmp_v0[0]
        idx_kv_cache__rv_v2: pl.Tensor[[IDX_CACHE_BLOCK_NUM_DYN, INDEXER_PAGE_BYTES_DYN], pl.INT8, pl.MemRef("mem_ddr_7", pl.const(0, pl.INT64), 0)] = ret__tmp_v0[1]
        return idx_kv_scale_values_inline2059__ssa_v0, idx_kv_cache__ssa_v0

    @pl.function(type=pl.FunctionType.AIC, level=pl.Level.AIC, role=pl.Role.SubWorker)
    def kv_hadamard(
        kv_final_inline2068__ssa_v0: pl.Out[pl.Tensor[[384, 128], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 196608)]],
        hadamard_idx__ssa_v0: pl.Tensor[[128, 128], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 32768)],
        rms_blocks_inline2072__ssa_v0: pl.Scalar[pl.INDEX],
        compact_rows_inline2073__ssa_v0: pl.Scalar[pl.INDEX],
        normed_kv_inline518__rv_v3: pl.Tensor[[384, 128], pl.BF16, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 98304)],
    ) -> pl.Tensor[[384, 128], pl.FP32]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        mem_mat_3: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 16384)
        mem_mat_4: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 4096)
        mem_left_5: pl.Ptr = pl.tile.alloc(pl.Mem.Left, 4096)
        mem_right_6: pl.Ptr = pl.tile.alloc(pl.Mem.Right, 16384)
        mem_acc_7: pl.Ptr = pl.tile.alloc(pl.Mem.Acc, 4096)
        for o0_inline2064__idx_v0, (kv_final_inline2068__iter_v1,) in pl.range(0, 128, 64, init_values=(kv_final_inline2068__ssa_v0,)):
            hadamard_tile_inline2061__tile: pl.Tile[[128, 64], pl.BF16, pl.MemRef(mem_mat_3, pl.const(0, pl.INT64), 16384), pl.Mem.Mat] = pl.tile.load(
                hadamard_idx__ssa_v0, [0, o0_inline2064__idx_v0], [128, 64], [128, 64], target_memory=pl.Mem.Mat
            )
            for had_blk_inline2092__idx_v0, (kv_final_inline2068__iter_v3,) in pl.range(rms_blocks_inline2072__ssa_v0, init_values=(kv_final_inline2068__iter_v1,)):
                had_b0_inline2058__ssa_v0: pl.Scalar[pl.INDEX] = had_blk_inline2092__idx_v0 * 16
                had_rows_inline2087__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(compact_rows_inline2073__ssa_v0 - had_b0_inline2058__ssa_v0, 16)
                kv_proj_tile_inline2060__tile: pl.Tile[
                    [16, 128], pl.BF16, pl.MemRef(mem_mat_4, pl.const(16384, pl.INT64), 4096), pl.Mem.Mat, pl.TileView(valid_shape=[had_rows_inline2087__ssa_v0, 128])
                ] = pl.tile.load(normed_kv_inline518__rv_v3, [had_b0_inline2058__ssa_v0, 0], [16, 128], [had_rows_inline2087__ssa_v0, 128], target_memory=pl.Mem.Mat)
                kv_proj_tile_inline2060__tile_Left: pl.Tile[
                    [16, 128], pl.BF16, pl.MemRef(mem_left_5, pl.const(0, pl.INT64), 4096), pl.Mem.Left, pl.TileView(valid_shape=[had_rows_inline2087__ssa_v0, 128])
                ] = pl.tile.move(kv_proj_tile_inline2060__tile, target_memory=pl.Mem.Left)
                hadamard_tile_inline2061__tile_Right: pl.Tile[[128, 64], pl.BF16, pl.MemRef(mem_right_6, pl.const(0, pl.INT64), 16384), pl.Mem.Right] = pl.tile.move(
                    hadamard_tile_inline2061__tile, target_memory=pl.Mem.Right
                )
                kv_hadamard_acc_inline2081__tile: pl.Tile[
                    [16, 64], pl.FP32, pl.MemRef(mem_acc_7, pl.const(0, pl.INT64), 4096), pl.Mem.Acc, pl.TileView(valid_shape=[had_rows_inline2087__ssa_v0, 64], compact=pl.CompactMode.normal)
                ] = pl.tile.matmul(kv_proj_tile_inline2060__tile_Left, hadamard_tile_inline2061__tile_Right)
                kv_final_inline2068__tile: pl.Tensor[[384, 128], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 196608)] = pl.tile.store(
                    kv_hadamard_acc_inline2081__tile, [had_b0_inline2058__ssa_v0, o0_inline2064__idx_v0], kv_final_inline2068__iter_v3
                )
                kv_final_inline2068__rv_v4: pl.Tensor[[384, 128], pl.FP32, pl.MemRef("mem_ddr_8", pl.const(0, pl.INT64), 196608)] = pl.yield_(kv_final_inline2068__tile)
            kv_final_inline2068__rv_v2: pl.Tensor[[384, 128], pl.FP32, pl.MemRef("mem_ddr_9", pl.const(0, pl.INT64), 196608)] = pl.yield_(kv_final_inline2068__rv_v4)
        return kv_final_inline2068__ssa_v0

    @pl.function(type=pl.FunctionType.AIC, level=pl.Level.AIC, role=pl.Role.SubWorker)
    def kv_proj_matmul(
        kv_m_groups_inline1787__ssa_v0: pl.Scalar[pl.INDEX],
        kv_fp32_inline1786__rv_v2: pl.InOut[pl.Tensor[[t_matmul_inline1778__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)]],
        kv_full_rows_inline1775__ssa_v0: pl.Scalar[pl.INDEX],
        tile_base_inline1766__idx_v0: pl.Scalar[pl.INDEX],
        x_view_inline1760__ssa_v0: pl.Tensor[[t_dim_inline1779__ssa_v0, 4096], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)],
        wkv__ssa_v0: pl.Tensor[[4096, 512], pl.BF16, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 4194304)],
        t_matmul_inline1778__ssa_v0: pl.Scalar[pl.INDEX],
        tile_rows_inline1768__ssa_v0: pl.Scalar[pl.INDEX],
    ) -> pl.Tensor[[t_matmul_inline1778__ssa_v0, 512], pl.FP32]:
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
        kbg_inline1801__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        kv_col0_inline1763__ssa_v0: pl.Scalar[pl.INDEX] = kbg_inline1801__ssa_v0 // (kv_m_groups_inline1787__ssa_v0 * 2) * 128
        kv_k_base_inline1820__ssa_v0: pl.Scalar[pl.INDEX] = kbg_inline1801__ssa_v0 // kv_m_groups_inline1787__ssa_v0 % 2 * 2048
        kv_m_group_inline1757__ssa_v0: pl.Scalar[pl.INDEX] = kbg_inline1801__ssa_v0 % kv_m_groups_inline1787__ssa_v0
        for dense_t0_inline1773__idx_v0, (kv_fp32_inline1786__iter_v6,) in pl.range(
            kv_m_group_inline1757__ssa_v0 * 64, kv_full_rows_inline1775__ssa_v0, kv_m_groups_inline1787__ssa_v0 * 64, init_values=(kv_fp32_inline1786__rv_v2,)
        ):
            dense_x0_inline1781__ssa_v0: pl.Scalar[pl.INDEX] = tile_base_inline1766__idx_v0 + dense_t0_inline1773__idx_v0
            dense_acc_inline1756__tile: pl.Tile[[64, 128], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 32768), pl.Mem.Acc] = pl.tile.create([64, 128], dtype=pl.FP32, target_memory=pl.Mem.Acc)
            for dense_k_inline1782__idx_v0, (dense_acc_inline1756__iter_v1,) in pl.range(0, 8, 2, init_values=(dense_acc_inline1756__tile,)):
                dense_d0_inline1754__ssa_v0: pl.Scalar[pl.INDEX] = kv_k_base_inline1820__ssa_v0 + dense_k_inline1782__idx_v0 * 256
                dense_d0_inline1754__ssa_v0_1: pl.Scalar[pl.INDEX] = kv_k_base_inline1820__ssa_v0 + (dense_k_inline1782__idx_v0 * 256 + 256)
                dense_x_inline1752__tile: pl.Tile[[64, 256], pl.BF16, pl.MemRef(mem_mat_4, pl.const(0, pl.INT64), 32768), pl.Mem.Mat] = pl.tile.load(
                    x_view_inline1760__ssa_v0, [dense_x0_inline1781__ssa_v0, dense_d0_inline1754__ssa_v0], [64, 256], [64, 256], target_memory=pl.Mem.Mat
                )
                dense_w_inline1794__tile: pl.Tile[[256, 128], pl.BF16, pl.MemRef(mem_mat_5, pl.const(32768, pl.INT64), 65536), pl.Mem.Mat] = pl.tile.load(
                    wkv__ssa_v0, [dense_d0_inline1754__ssa_v0, kv_col0_inline1763__ssa_v0], [256, 128], [256, 128], target_memory=pl.Mem.Mat
                )
                dense_x_inline1752__tile_1: pl.Tile[[64, 256], pl.BF16, pl.MemRef(mem_mat_6, pl.const(98304, pl.INT64), 32768), pl.Mem.Mat] = pl.tile.load(
                    x_view_inline1760__ssa_v0, [dense_x0_inline1781__ssa_v0, dense_d0_inline1754__ssa_v0_1], [64, 256], [64, 256], target_memory=pl.Mem.Mat
                )
                dense_w_inline1794__tile_1: pl.Tile[[256, 128], pl.BF16, pl.MemRef(mem_mat_7, pl.const(131072, pl.INT64), 65536), pl.Mem.Mat] = pl.tile.load(
                    wkv__ssa_v0, [dense_d0_inline1754__ssa_v0_1, kv_col0_inline1763__ssa_v0], [256, 128], [256, 128], target_memory=pl.Mem.Mat
                )
                dense_acc_inline1756__tile_l0_a: pl.Tile[[64, 128], pl.BF16, pl.MemRef(mem_left_8, pl.const(49152, pl.INT64), 16384), pl.Mem.Left, pl.TileView(blayout=pl.TileLayout.row_major)] = (
                    pl.tile.extract(dense_x_inline1752__tile, 0, 0, [64, 128], target_memory=pl.Mem.Left)
                )
                dense_acc_inline1756__tile_l0_b: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_9, pl.const(32768, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                    dense_w_inline1794__tile, 0, 0, [128, 128], target_memory=pl.Mem.Right
                )
                dense_acc_inline1756__tile_l0_a_1: pl.Tile[[64, 128], pl.BF16, pl.MemRef(mem_left_10, pl.const(0, pl.INT64), 16384), pl.Mem.Left, pl.TileView(blayout=pl.TileLayout.row_major)] = (
                    pl.tile.extract(dense_x_inline1752__tile, 0, 128, [64, 128], target_memory=pl.Mem.Left)
                )
                dense_acc_inline1756__tile_l0_b_1: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_11, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                    dense_w_inline1794__tile, 128, 0, [128, 128], target_memory=pl.Mem.Right
                )
                dense_acc_inline1756__tile_l0_c_acc: pl.Tile[[64, 128], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 32768), pl.Mem.Acc] = pl.tile.matmul_acc(
                    dense_acc_inline1756__iter_v1, dense_acc_inline1756__tile_l0_a, dense_acc_inline1756__tile_l0_b, dense_k_inline1782__idx_v0 == 0
                )
                dense_acc_inline1756__tile_l0_c_acc_1: pl.Tile[[64, 128], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 32768), pl.Mem.Acc] = pl.tile.matmul_acc(
                    dense_acc_inline1756__tile_l0_c_acc, dense_acc_inline1756__tile_l0_a_1, dense_acc_inline1756__tile_l0_b_1, False
                )
                dense_acc_inline1756__tile_l0_a_2: pl.Tile[[64, 128], pl.BF16, pl.MemRef(mem_left_13, pl.const(16384, pl.INT64), 16384), pl.Mem.Left, pl.TileView(blayout=pl.TileLayout.row_major)] = (
                    pl.tile.extract(dense_x_inline1752__tile_1, 0, 0, [64, 128], target_memory=pl.Mem.Left)
                )
                dense_acc_inline1756__tile_l0_b_2: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_9, pl.const(32768, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                    dense_w_inline1794__tile_1, 0, 0, [128, 128], target_memory=pl.Mem.Right
                )
                dense_acc_inline1756__tile_l0_a_3: pl.Tile[[64, 128], pl.BF16, pl.MemRef(mem_left_15, pl.const(32768, pl.INT64), 16384), pl.Mem.Left, pl.TileView(blayout=pl.TileLayout.row_major)] = (
                    pl.tile.extract(dense_x_inline1752__tile_1, 0, 128, [64, 128], target_memory=pl.Mem.Left)
                )
                dense_acc_inline1756__tile_l0_b_3: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_11, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                    dense_w_inline1794__tile_1, 128, 0, [128, 128], target_memory=pl.Mem.Right
                )
                dense_acc_inline1756__tile_l0_c_acc_2: pl.Tile[[64, 128], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 32768), pl.Mem.Acc] = pl.tile.matmul_acc(
                    dense_acc_inline1756__tile_l0_c_acc_1, dense_acc_inline1756__tile_l0_a_2, dense_acc_inline1756__tile_l0_b_2, dense_k_inline1782__idx_v0 == -1
                )
                dense_acc_inline1756__tile_l0_c_acc_3: pl.Tile[[64, 128], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 32768), pl.Mem.Acc] = pl.tile.matmul_acc(
                    dense_acc_inline1756__tile_l0_c_acc_2, dense_acc_inline1756__tile_l0_a_3, dense_acc_inline1756__tile_l0_b_3, False
                )
                dense_acc_inline1756__rv_v2: pl.Tile[[64, 128], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 32768), pl.Mem.Acc] = pl.yield_(dense_acc_inline1756__tile_l0_c_acc_3)
            kv_fp32_inline1786__tile: pl.Tensor[[t_matmul_inline1778__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                dense_acc_inline1756__rv_v2, [dense_t0_inline1773__idx_v0, kv_col0_inline1763__ssa_v0], kv_fp32_inline1786__iter_v6, atomic=pl.AtomicType.Add
            )
            kv_fp32_inline1786__rv_v7: pl.Tensor[[t_matmul_inline1778__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_19", pl.const(0, pl.INT64), 0)] = pl.yield_(kv_fp32_inline1786__tile)
        for t0_inline1788__idx_v0, (kv_fp32_inline1786__iter_v9,) in pl.range(
            kv_full_rows_inline1775__ssa_v0 + kv_m_group_inline1757__ssa_v0 * 16, t_matmul_inline1778__ssa_v0, kv_m_groups_inline1787__ssa_v0 * 16, init_values=(kv_fp32_inline1786__rv_v7,)
        ):
            kv_acc_inline1750__tile: pl.Tile[[16, 128], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 8192), pl.Mem.Acc] = pl.tile.create([16, 128], dtype=pl.FP32, target_memory=pl.Mem.Acc)
            for db_inline1809__idx_v0, (kv_acc_inline1750__iter_v1,) in pl.range(0, 8, 2, init_values=(kv_acc_inline1750__tile,)):
                d0_inline1822__ssa_v0: pl.Scalar[pl.INDEX] = kv_k_base_inline1820__ssa_v0 + db_inline1809__idx_v0 * 256
                kv_rows_inline1764__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(tile_rows_inline1768__ssa_v0 - t0_inline1788__idx_v0, 16)
                x_t0_inline1791__ssa_v0: pl.Scalar[pl.INDEX] = tile_base_inline1766__idx_v0 + t0_inline1788__idx_v0
                d0_inline1822__ssa_v0_1: pl.Scalar[pl.INDEX] = kv_k_base_inline1820__ssa_v0 + (db_inline1809__idx_v0 * 256 + 256)
                kv_rows_inline1764__ssa_v0_1: pl.Scalar[pl.INDEX] = pl.min(tile_rows_inline1768__ssa_v0 - t0_inline1788__idx_v0, 16)
                x_t0_inline1791__ssa_v0_1: pl.Scalar[pl.INDEX] = tile_base_inline1766__idx_v0 + t0_inline1788__idx_v0
                kv_x_chunk_bf16_inline1780__tile: pl.Tile[
                    [16, 256], pl.BF16, pl.MemRef(mem_mat_4, pl.const(0, pl.INT64), 8192), pl.Mem.Mat, pl.TileView(valid_shape=[kv_rows_inline1764__ssa_v0, 256])
                ] = pl.tile.load(x_view_inline1760__ssa_v0, [x_t0_inline1791__ssa_v0, d0_inline1822__ssa_v0], [16, 256], [kv_rows_inline1764__ssa_v0, 256], target_memory=pl.Mem.Mat)
                wkv_chunk_inline1797__tile: pl.Tile[[256, 128], pl.BF16, pl.MemRef(mem_mat_5, pl.const(32768, pl.INT64), 65536), pl.Mem.Mat] = pl.tile.load(
                    wkv__ssa_v0, [d0_inline1822__ssa_v0, kv_col0_inline1763__ssa_v0], [256, 128], [256, 128], target_memory=pl.Mem.Mat
                )
                kv_x_chunk_bf16_inline1780__tile_1: pl.Tile[
                    [16, 256], pl.BF16, pl.MemRef(mem_mat_6, pl.const(98304, pl.INT64), 8192), pl.Mem.Mat, pl.TileView(valid_shape=[kv_rows_inline1764__ssa_v0_1, 256])
                ] = pl.tile.load(x_view_inline1760__ssa_v0, [x_t0_inline1791__ssa_v0_1, d0_inline1822__ssa_v0_1], [16, 256], [kv_rows_inline1764__ssa_v0_1, 256], target_memory=pl.Mem.Mat)
                wkv_chunk_inline1797__tile_1: pl.Tile[[256, 128], pl.BF16, pl.MemRef(mem_mat_7, pl.const(131072, pl.INT64), 65536), pl.Mem.Mat] = pl.tile.load(
                    wkv__ssa_v0, [d0_inline1822__ssa_v0_1, kv_col0_inline1763__ssa_v0], [256, 128], [256, 128], target_memory=pl.Mem.Mat
                )
                kv_acc_inline1750__tile_l0_a: pl.Tile[
                    [16, 128],
                    pl.BF16,
                    pl.MemRef(mem_left_8, pl.const(49152, pl.INT64), 4096),
                    pl.Mem.Left,
                    pl.TileView(valid_shape=[kv_rows_inline1764__ssa_v0, 128], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal),
                ] = pl.tile.extract(kv_x_chunk_bf16_inline1780__tile, 0, 0, [16, 128], target_memory=pl.Mem.Left)
                kv_acc_inline1750__tile_l0_b: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_9, pl.const(32768, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                    wkv_chunk_inline1797__tile, 0, 0, [128, 128], target_memory=pl.Mem.Right
                )
                kv_acc_inline1750__tile_l0_a_1: pl.Tile[
                    [16, 128],
                    pl.BF16,
                    pl.MemRef(mem_left_10, pl.const(0, pl.INT64), 4096),
                    pl.Mem.Left,
                    pl.TileView(valid_shape=[kv_rows_inline1764__ssa_v0, 128], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal),
                ] = pl.tile.extract(kv_x_chunk_bf16_inline1780__tile, 0, 128, [16, 128], target_memory=pl.Mem.Left)
                kv_acc_inline1750__tile_l0_b_1: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_11, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                    wkv_chunk_inline1797__tile, 128, 0, [128, 128], target_memory=pl.Mem.Right
                )
                kv_acc_inline1750__tile_l0_c_acc: pl.Tile[[16, 128], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 8192), pl.Mem.Acc] = pl.tile.matmul_acc(
                    kv_acc_inline1750__iter_v1, kv_acc_inline1750__tile_l0_a, kv_acc_inline1750__tile_l0_b, db_inline1809__idx_v0 == 0
                )
                kv_acc_inline1750__tile_l0_c_acc_1: pl.Tile[[16, 128], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 8192), pl.Mem.Acc] = pl.tile.matmul_acc(
                    kv_acc_inline1750__tile_l0_c_acc, kv_acc_inline1750__tile_l0_a_1, kv_acc_inline1750__tile_l0_b_1, False
                )
                kv_acc_inline1750__tile_l0_a_2: pl.Tile[
                    [16, 128],
                    pl.BF16,
                    pl.MemRef(mem_left_13, pl.const(16384, pl.INT64), 4096),
                    pl.Mem.Left,
                    pl.TileView(valid_shape=[kv_rows_inline1764__ssa_v0_1, 128], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal),
                ] = pl.tile.extract(kv_x_chunk_bf16_inline1780__tile_1, 0, 0, [16, 128], target_memory=pl.Mem.Left)
                kv_acc_inline1750__tile_l0_b_2: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_9, pl.const(32768, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                    wkv_chunk_inline1797__tile_1, 0, 0, [128, 128], target_memory=pl.Mem.Right
                )
                kv_acc_inline1750__tile_l0_a_3: pl.Tile[
                    [16, 128],
                    pl.BF16,
                    pl.MemRef(mem_left_15, pl.const(32768, pl.INT64), 4096),
                    pl.Mem.Left,
                    pl.TileView(valid_shape=[kv_rows_inline1764__ssa_v0_1, 128], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal),
                ] = pl.tile.extract(kv_x_chunk_bf16_inline1780__tile_1, 0, 128, [16, 128], target_memory=pl.Mem.Left)
                kv_acc_inline1750__tile_l0_b_3: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_11, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                    wkv_chunk_inline1797__tile_1, 128, 0, [128, 128], target_memory=pl.Mem.Right
                )
                kv_acc_inline1750__tile_l0_c_acc_2: pl.Tile[[16, 128], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 8192), pl.Mem.Acc] = pl.tile.matmul_acc(
                    kv_acc_inline1750__tile_l0_c_acc_1, kv_acc_inline1750__tile_l0_a_2, kv_acc_inline1750__tile_l0_b_2, db_inline1809__idx_v0 == -1
                )
                kv_acc_inline1750__tile_l0_c_acc_3: pl.Tile[[16, 128], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 8192), pl.Mem.Acc] = pl.tile.matmul_acc(
                    kv_acc_inline1750__tile_l0_c_acc_2, kv_acc_inline1750__tile_l0_a_3, kv_acc_inline1750__tile_l0_b_3, False
                )
                kv_acc_inline1750__rv_v2: pl.Tile[[16, 128], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 8192), pl.Mem.Acc] = pl.yield_(kv_acc_inline1750__tile_l0_c_acc_3)
            kv_fp32_inline1786__tile_1: pl.Tensor[[t_matmul_inline1778__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_19", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                kv_acc_inline1750__rv_v2, [t0_inline1788__idx_v0, kv_col0_inline1763__ssa_v0], kv_fp32_inline1786__iter_v9, atomic=pl.AtomicType.Add
            )
            kv_fp32_inline1786__rv_v10: pl.Tensor[[t_matmul_inline1778__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_36", pl.const(0, pl.INT64), 0)] = pl.yield_(kv_fp32_inline1786__tile_1)
        return kv_fp32_inline1786__rv_v2

    @pl.function(type=pl.FunctionType.Spmd)
    def kv_proj_matmul_spmd(
        self,
        kv_m_groups_inline1787__ssa_v0: pl.Scalar[pl.INDEX],
        kv_fp32_inline1786__rv_v2: pl.InOut[pl.Tensor[[t_matmul_inline1778__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)]],
        kv_full_rows_inline1775__ssa_v0: pl.Scalar[pl.INDEX],
        tile_base_inline1766__idx_v0: pl.Scalar[pl.INDEX],
        x_view_inline1760__ssa_v0: pl.Tensor[[t_dim_inline1779__ssa_v0, 4096], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)],
        wkv__ssa_v0: pl.Tensor[[4096, 512], pl.BF16, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 4194304)],
        t_matmul_inline1778__ssa_v0: pl.Scalar[pl.INDEX],
        tile_rows_inline1768__ssa_v0: pl.Scalar[pl.INDEX],
    ) -> pl.Tensor[[t_matmul_inline1778__ssa_v0, 512], pl.FP32]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        kv_fp32_inline1786__rv_v10: pl.Tensor[[t_matmul_inline1778__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)] = self.kv_proj_matmul(
            kv_m_groups_inline1787__ssa_v0,
            kv_fp32_inline1786__rv_v2,
            kv_full_rows_inline1775__ssa_v0,
            tile_base_inline1766__idx_v0,
            x_view_inline1760__ssa_v0,
            wkv__ssa_v0,
            t_matmul_inline1778__ssa_v0,
            tile_rows_inline1768__ssa_v0,
            attrs={"arg_directions": [pl.adir.scalar, pl.adir.inout, pl.adir.scalar, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.scalar, pl.adir.scalar]},
        )
        return kv_fp32_inline1786__rv_v2

    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def kv_proj_seed(
        kv_fp32_inline1786__ssa_v0: pl.Out[pl.Tensor[[t_matmul_inline1778__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)]], t_matmul_inline1778__ssa_v0: pl.Scalar[pl.INDEX]
    ) -> pl.Tensor[[t_matmul_inline1778__ssa_v0, 512], pl.FP32]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        mem_vec_1: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 8192)
        for kts0_inline1783__idx_v0, (kv_fp32_inline1786__iter_v1,) in pl.range(0, t_matmul_inline1778__ssa_v0, 16, init_values=(kv_fp32_inline1786__ssa_v0,)):
            for kvseed0_inline1792__idx_v0, (kv_fp32_inline1786__iter_v3,) in pl.range(0, 512, 128, init_values=(kv_fp32_inline1786__iter_v1,)):
                kv_seed_inline1795__tile: pl.Tile[[16, 128], pl.FP32, pl.MemRef(mem_vec_1, pl.const(0, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.full([16, 128], dtype=pl.FP32, value=0.0)
                kv_fp32_inline1786__tile: pl.Tensor[[t_matmul_inline1778__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                    kv_seed_inline1795__tile, [kts0_inline1783__idx_v0, kvseed0_inline1792__idx_v0], kv_fp32_inline1786__iter_v3
                )
                kv_fp32_inline1786__rv_v4: pl.Tensor[[t_matmul_inline1778__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)] = pl.yield_(kv_fp32_inline1786__tile)
            kv_fp32_inline1786__rv_v2: pl.Tensor[[t_matmul_inline1778__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)] = pl.yield_(kv_fp32_inline1786__rv_v4)
        return kv_fp32_inline1786__ssa_v0

    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def kv_rms_norm_rope(
        tile_rows_inline1768__ssa_v0: pl.Scalar[pl.INDEX],
        tile_base_inline1766__idx_v0: pl.Scalar[pl.INDEX],
        kv_fp32_inline1786__rv_v10: pl.Tensor[[t_matmul_inline1778__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)],
        kv_view_inline1818__ssa_v0: pl.InOut[pl.Tensor[[t_dim_inline1779__ssa_v0, 512], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)]],
        gamma_ckv__ssa_v0: pl.Tensor[[512], pl.BF16, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 1024)],
        q_rope_cos_il_inline503__ssa_v0: pl.Tensor[[t_dim_inline504__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)],
        q_rope_sin_signed_inline502__ssa_v0: pl.Tensor[[t_dim_inline504__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 0)],
        q_rope_swap_idx_inline501__ssa_v0: pl.Tensor[[t_dim_inline504__ssa_v0, 64], pl.INT32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 0)],
    ) -> pl.Tensor[[t_dim_inline1779__ssa_v0, 512], pl.BF16]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        mem_vec_9: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 8192)
        mem_vec_10: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
        mem_vec_13: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 8192)
        mem_vec_14: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
        mem_vec_50: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 8192)
        mem_vec_117: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 8192)
        mem_vec_118: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 8192)
        tg_idx_inline1806__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        tg_inline1802__ssa_v0: pl.Scalar[pl.INDEX] = tg_idx_inline1806__ssa_v0 * 32
        valid_rows_inline1755__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(tile_rows_inline1768__ssa_v0 - tg_inline1802__ssa_v0, 32)
        out_tg_inline1789__ssa_v0: pl.Scalar[pl.INDEX] = tile_base_inline1766__idx_v0 + tg_inline1802__ssa_v0
        if valid_rows_inline1755__ssa_v0 == 32:
            kv_sq_sum_inline1814__tile: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_50, pl.const(57344, pl.INT64), 128), pl.Mem.Vec] = pl.tile.full([1, 32], dtype=pl.FP32, value=0.0)
            for kv_sq_col0_inline1804__idx_v0, (kv_sq_sum_inline1814__iter_v1,) in pl.range(0, 512, 128, init_values=(kv_sq_sum_inline1814__tile,)):
                kv_chunk_inline1769__tile: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.load(
                    kv_fp32_inline1786__rv_v10, [tg_inline1802__ssa_v0, kv_sq_col0_inline1804__idx_v0], [32, 64], [32, 64], target_memory=pl.Mem.Vec
                )
                kv_chunk_inline1769__tile_1: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_14, pl.const(40960, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.load(
                    kv_fp32_inline1786__rv_v10, [tg_inline1802__ssa_v0, kv_sq_col0_inline1804__idx_v0 + 64], [32, 64], [32, 64], target_memory=pl.Mem.Vec
                )
                kv_sq_inline1765__tile: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_9, pl.const(65536, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.mul(
                    kv_chunk_inline1769__tile, kv_chunk_inline1769__tile
                )
                tmp_tile: pl.Tile[[32, 128], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.create([32, 128], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                t__tile: pl.Tile[[32, 1], pl.FP32, pl.MemRef(mem_vec_117, pl.const(16384, pl.INT64), 128), pl.Mem.Vec] = pl.tile.row_sum(kv_sq_inline1765__tile, tmp_tile)
                kv_row_sum_inline1811__tile: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_117, pl.const(16384, pl.INT64), 128), pl.Mem.Vec] = pl.tile.reshape(t__tile, [1, 32])
                kv_sq_sum_inline1814__tile_1: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 128), pl.Mem.Vec] = pl.tile.add(
                    kv_sq_sum_inline1814__iter_v1, kv_row_sum_inline1811__tile
                )
                kv_sq_inline1765__tile_1: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(32768, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.mul(
                    kv_chunk_inline1769__tile_1, kv_chunk_inline1769__tile_1
                )
                tmp_tile_1: pl.Tile[[32, 128], pl.FP32, pl.MemRef(mem_vec_14, pl.const(40960, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.create([32, 128], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                t__tile_1: pl.Tile[[32, 1], pl.FP32, pl.MemRef(mem_vec_118, pl.const(24576, pl.INT64), 128), pl.Mem.Vec] = pl.tile.row_sum(kv_sq_inline1765__tile_1, tmp_tile_1)
                kv_row_sum_inline1811__tile_1: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_118, pl.const(24576, pl.INT64), 128), pl.Mem.Vec] = pl.tile.reshape(t__tile_1, [1, 32])
                kv_sq_sum_inline1814__tile_2: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_50, pl.const(57344, pl.INT64), 128), pl.Mem.Vec] = pl.tile.add(
                    kv_sq_sum_inline1814__tile_1, kv_row_sum_inline1811__tile_1
                )
                kv_sq_sum_inline1814__rv_v2: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_50, pl.const(57344, pl.INT64), 128), pl.Mem.Vec] = pl.yield_(kv_sq_sum_inline1814__tile_2)
            t__tile_2: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 128), pl.Mem.Vec] = pl.tile.muls(kv_sq_sum_inline1814__rv_v2, 0.001953125)
            t__tile_3: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 128), pl.Mem.Vec] = pl.tile.adds(t__tile_2, 9.9999999999999995e-07)
            rsqrt_tmp: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_14, pl.const(40960, pl.INT64), 128), pl.Mem.Vec] = pl.tile.create([1, 32], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            kv_inv_rms_inline1812__tile: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_50, pl.const(57344, pl.INT64), 128), pl.Mem.Vec] = pl.tile.rsqrt(t__tile_3, rsqrt_tmp)
            kv_inv_rms_t_inline1813__tile: pl.Tile[[32, 1], pl.FP32, pl.MemRef(mem_vec_50, pl.const(57344, pl.INT64), 128), pl.Mem.Vec] = pl.tile.reshape(kv_inv_rms_inline1812__tile, [32, 1])
            for n0_inline1816__idx_v0, (kv_view_inline1818__iter_v1,) in pl.range(0, 384, 128, init_values=(kv_view_inline1818__ssa_v0,)):
                kv_chunk_inline1769__tile_2: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.load(
                    kv_fp32_inline1786__rv_v10, [tg_inline1802__ssa_v0, n0_inline1816__idx_v0], [32, 64], [32, 64], target_memory=pl.Mem.Vec
                )
                t__tile_4: pl.Tile[[64], pl.BF16, pl.MemRef(mem_vec_117, pl.const(16384, pl.INT64), 128), pl.Mem.Vec] = pl.tile.load(
                    gamma_ckv__ssa_v0, [n0_inline1816__idx_v0], [64], [64], target_memory=pl.Mem.Vec
                )
                kv_chunk_inline1769__tile_3: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_14, pl.const(40960, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.load(
                    kv_fp32_inline1786__rv_v10, [tg_inline1802__ssa_v0, n0_inline1816__idx_v0 + 64], [32, 64], [32, 64], target_memory=pl.Mem.Vec
                )
                t__tile_5: pl.Tile[[64], pl.BF16, pl.MemRef(mem_vec_118, pl.const(24576, pl.INT64), 128), pl.Mem.Vec] = pl.tile.load(
                    gamma_ckv__ssa_v0, [n0_inline1816__idx_v0 + 64], [64], [64], target_memory=pl.Mem.Vec
                )
                gamma_kv_cast_inline1815__tile: pl.Tile[[64], pl.FP32, pl.MemRef(mem_vec_9, pl.const(65536, pl.INT64), 256), pl.Mem.Vec] = pl.tile.cast(t__tile_4, target_type=pl.FP32, mode="round")
                gamma_kv_chunk_inline1798__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_9, pl.const(65536, pl.INT64), 256), pl.Mem.Vec] = gamma_kv_cast_inline1815__tile
                t__tile_6: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.row_expand_mul(
                    kv_chunk_inline1769__tile_2, kv_inv_rms_t_inline1813__tile
                )
                kv_normed_inline1819__tile: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.col_expand_mul(
                    t__tile_6, gamma_kv_chunk_inline1798__tile
                )
                kv_normed_bf16_inline1784__tile: pl.Tile[[32, 64], pl.BF16, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(
                    kv_normed_inline1819__tile, target_type=pl.BF16, mode="rint"
                )
                kv_view_inline1818__tile: pl.Tensor[[t_dim_inline1779__ssa_v0, 512], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                    kv_normed_bf16_inline1784__tile, [out_tg_inline1789__ssa_v0, n0_inline1816__idx_v0], kv_view_inline1818__iter_v1
                )
                gamma_kv_cast_inline1815__tile_1: pl.Tile[[64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(32768, pl.INT64), 256), pl.Mem.Vec] = pl.tile.cast(t__tile_5, target_type=pl.FP32, mode="round")
                gamma_kv_chunk_inline1798__tile_1: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(32768, pl.INT64), 256), pl.Mem.Vec] = gamma_kv_cast_inline1815__tile_1
                t__tile_7: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_14, pl.const(40960, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.row_expand_mul(
                    kv_chunk_inline1769__tile_3, kv_inv_rms_t_inline1813__tile
                )
                kv_normed_inline1819__tile_1: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_14, pl.const(40960, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.col_expand_mul(
                    t__tile_7, gamma_kv_chunk_inline1798__tile_1
                )
                kv_normed_bf16_inline1784__tile_1: pl.Tile[[32, 64], pl.BF16, pl.MemRef(mem_vec_14, pl.const(40960, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(
                    kv_normed_inline1819__tile_1, target_type=pl.BF16, mode="rint"
                )
                kv_view_inline1818__tile_1: pl.Tensor[[t_dim_inline1779__ssa_v0, 512], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                    kv_normed_bf16_inline1784__tile_1, [out_tg_inline1789__ssa_v0, n0_inline1816__idx_v0 + 64], kv_view_inline1818__tile
                )
                kv_view_inline1818__rv_v2_main: pl.Tensor[[t_dim_inline1779__ssa_v0, 512], pl.BF16, pl.MemRef("mem_ddr_34", pl.const(0, pl.INT64), 0)] = pl.yield_(kv_view_inline1818__tile_1)
            kv_chunk_inline1769__tile_4: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.load(
                kv_fp32_inline1786__rv_v10, [tg_inline1802__ssa_v0, 384], [32, 64], [32, 64], target_memory=pl.Mem.Vec
            )
            t__tile_8: pl.Tile[[64], pl.BF16, pl.MemRef(mem_vec_9, pl.const(65536, pl.INT64), 128), pl.Mem.Vec] = pl.tile.load(gamma_ckv__ssa_v0, [384], [64], [64], target_memory=pl.Mem.Vec)
            gamma_kv_cast_inline1815__tile_2: pl.Tile[[64], pl.FP32, pl.MemRef(mem_vec_14, pl.const(40960, pl.INT64), 256), pl.Mem.Vec] = pl.tile.cast(t__tile_8, target_type=pl.FP32, mode="round")
            gamma_kv_chunk_inline1798__tile_2: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_14, pl.const(40960, pl.INT64), 256), pl.Mem.Vec] = gamma_kv_cast_inline1815__tile_2
            t__tile_9: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.row_expand_mul(kv_chunk_inline1769__tile_4, kv_inv_rms_t_inline1813__tile)
            kv_normed_inline1819__tile_2: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.col_expand_mul(
                t__tile_9, gamma_kv_chunk_inline1798__tile_2
            )
            kv_normed_bf16_inline1784__tile_2: pl.Tile[[32, 64], pl.BF16, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(
                kv_normed_inline1819__tile_2, target_type=pl.BF16, mode="rint"
            )
            kv_view_inline1818__tile_2: pl.Tensor[[t_dim_inline1779__ssa_v0, 512], pl.BF16, pl.MemRef("mem_ddr_34", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                kv_normed_bf16_inline1784__tile_2, [out_tg_inline1789__ssa_v0, 384], kv_view_inline1818__rv_v2_main
            )
            kv_view_inline1818__rv_v2: pl.Tensor[[t_dim_inline1779__ssa_v0, 512], pl.BF16, pl.MemRef("mem_ddr_41", pl.const(0, pl.INT64), 0)] = kv_view_inline1818__tile_2
            t__tile_10: pl.Tile[[64], pl.BF16, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 128), pl.Mem.Vec] = pl.tile.load(gamma_ckv__ssa_v0, [448], [64], [64], target_memory=pl.Mem.Vec)
            gamma_rope_cast_inline1821__tile: pl.Tile[[64], pl.FP32, pl.MemRef(mem_vec_14, pl.const(40960, pl.INT64), 256), pl.Mem.Vec] = pl.tile.cast(t__tile_10, target_type=pl.FP32, mode="round")
            gamma_rope_inline1807__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_14, pl.const(40960, pl.INT64), 256), pl.Mem.Vec] = gamma_rope_cast_inline1821__tile
            kv_rope_chunk_inline1823__tile: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.load(
                kv_fp32_inline1786__rv_v10, [tg_inline1802__ssa_v0, 448], [32, 64], [32, 64], target_memory=pl.Mem.Vec
            )
            t__tile_11: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.row_expand_mul(
                kv_rope_chunk_inline1823__tile, kv_inv_rms_t_inline1813__tile
            )
            kv_rope_norm_chunk_inline1761__tile: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.col_expand_mul(
                t__tile_11, gamma_rope_inline1807__tile
            )
            kv_cos_il_full_inline1771__tile: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_14, pl.const(40960, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.load(
                q_rope_cos_il_inline503__ssa_v0, [out_tg_inline1789__ssa_v0, 0], [32, 64], [32, 64], target_memory=pl.Mem.Vec
            )
            kv_sin_signed_full_inline1810__tile: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_9, pl.const(65536, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.load(
                q_rope_sin_signed_inline502__ssa_v0, [out_tg_inline1789__ssa_v0, 0], [32, 64], [32, 64], target_memory=pl.Mem.Vec
            )
            kv_swap_idx_full_inline1800__tile: pl.Tile[[32, 64], pl.INT32, pl.MemRef(mem_vec_13, pl.const(32768, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.load(
                q_rope_swap_idx_inline501__ssa_v0, [out_tg_inline1789__ssa_v0, 0], [32, 64], [32, 64], target_memory=pl.Mem.Vec
            )
            gather_acc_init: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_50, pl.const(57344, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.create([32, 64], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            for gather_lv, (gather_ia,) in pl.range(32, init_values=(gather_acc_init,)):
                gather_inp_row: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 256), pl.Mem.Vec] = pl.tile.slice(
                    kv_rope_norm_chunk_inline1761__tile, [1, 64], [gather_lv, 0], [1, 64]
                )
                gather_idx_row: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_13, pl.const(32768, pl.INT64), 256), pl.Mem.Vec] = pl.tile.slice(
                    kv_swap_idx_full_inline1800__tile, [1, 64], [gather_lv, 0], [1, 64]
                )
                gather_row_tmp: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_117, pl.const(16384, pl.INT64), 256), pl.Mem.Vec] = pl.tile.create([1, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec)
                gather_row: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_118, pl.const(24576, pl.INT64), 256), pl.Mem.Vec] = pl.tile.gather(gather_inp_row, gather_idx_row, gather_row_tmp)
                gather_asmbl: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_50, pl.const(57344, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.assemble(gather_ia, gather_row, [gather_lv, 0])
                gather_rv: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_50, pl.const(57344, pl.INT64), 8192), pl.Mem.Vec] = pl.yield_(gather_asmbl)
            kv_swapped_full_inline1796__tile: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_50, pl.const(57344, pl.INT64), 8192), pl.Mem.Vec] = gather_rv
            t__tile_12: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.mul(kv_rope_norm_chunk_inline1761__tile, kv_cos_il_full_inline1771__tile)
            t__tile_13: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_14, pl.const(40960, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.mul(
                kv_swapped_full_inline1796__tile, kv_sin_signed_full_inline1810__tile
            )
            kv_rope_rot_full_inline1758__tile: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.add(t__tile_12, t__tile_13)
            kv_rope_i16_full_inline1824__tile: pl.Tile[[32, 64], pl.BF16, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(
                kv_rope_rot_full_inline1758__tile, target_type=pl.BF16, mode="rint"
            )
            kv_view_inline1818__tile_3: pl.Tensor[[t_dim_inline1779__ssa_v0, 512], pl.BF16, pl.MemRef("mem_ddr_41", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                kv_rope_i16_full_inline1824__tile, [out_tg_inline1789__ssa_v0, 448], kv_view_inline1818__rv_v2
            )
        else:
            kv_reduce_tmp_inline1777__ssa_v0: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.create(
                [32, 64], dtype=pl.FP32, target_memory=pl.Mem.Vec
            )
            kv_sq_sum_tail_inline1772__ssa_v0: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_13, pl.const(32768, pl.INT64), 128), pl.Mem.Vec] = pl.tile.full([1, 32], dtype=pl.FP32, value=0.0)
            for kv_sq_col0_tail_inline1749__idx_v0, (kv_sq_sum_tail_inline1772__iter_v1,) in pl.range(0, 512, 128, init_values=(kv_sq_sum_tail_inline1772__ssa_v0,)):
                kv_chunk_tail_inline1774__ssa_v0: pl.Tile[
                    [32, 64], pl.FP32, pl.MemRef(mem_vec_14, pl.const(40960, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1755__ssa_v0, 64])
                ] = pl.tile.load(kv_fp32_inline1786__rv_v10, [tg_inline1802__ssa_v0, kv_sq_col0_tail_inline1749__idx_v0], [32, 64], [valid_rows_inline1755__ssa_v0, 64], target_memory=pl.Mem.Vec)
                kv_chunk_tail_inline1774__ssa_v0_1: pl.Tile[
                    [32, 64], pl.FP32, pl.MemRef(mem_vec_9, pl.const(65536, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1755__ssa_v0, 64])
                ] = pl.tile.load(kv_fp32_inline1786__rv_v10, [tg_inline1802__ssa_v0, kv_sq_col0_tail_inline1749__idx_v0 + 64], [32, 64], [valid_rows_inline1755__ssa_v0, 64], target_memory=pl.Mem.Vec)
                kv_sq_tail_inline1808__ssa_v0: pl.Tile[
                    [32, 64], pl.FP32, pl.MemRef(mem_vec_14, pl.const(40960, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1755__ssa_v0, 64])
                ] = pl.tile.mul(kv_chunk_tail_inline1774__ssa_v0, kv_chunk_tail_inline1774__ssa_v0)
                t__tmp_v70: pl.Tile[[32, 1], pl.FP32, pl.MemRef(mem_vec_50, pl.const(57344, pl.INT64), 128), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1755__ssa_v0, 1])] = pl.tile.row_sum(
                    kv_sq_tail_inline1808__ssa_v0, kv_reduce_tmp_inline1777__ssa_v0
                )
                kv_row_sum_tail_inline1748__ssa_v0: pl.Tile[
                    [1, 32], pl.FP32, pl.MemRef(mem_vec_50, pl.const(57344, pl.INT64), 128), pl.Mem.Vec, pl.TileView(valid_shape=[1, valid_rows_inline1755__ssa_v0])
                ] = pl.tile.reshape(t__tmp_v70, [1, 32])
                kv_sq_sum_tail_inline1772__ssa_v3: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_14, pl.const(40960, pl.INT64), 128), pl.Mem.Vec] = pl.tile.add(
                    kv_sq_sum_tail_inline1772__iter_v1, kv_row_sum_tail_inline1748__ssa_v0
                )
                kv_sq_tail_inline1808__ssa_v0_1: pl.Tile[
                    [32, 64], pl.FP32, pl.MemRef(mem_vec_9, pl.const(65536, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1755__ssa_v0, 64])
                ] = pl.tile.mul(kv_chunk_tail_inline1774__ssa_v0_1, kv_chunk_tail_inline1774__ssa_v0_1)
                t__tmp_v70_1: pl.Tile[[32, 1], pl.FP32, pl.MemRef(mem_vec_117, pl.const(16384, pl.INT64), 128), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1755__ssa_v0, 1])] = (
                    pl.tile.row_sum(kv_sq_tail_inline1808__ssa_v0_1, kv_reduce_tmp_inline1777__ssa_v0)
                )
                kv_row_sum_tail_inline1748__ssa_v0_1: pl.Tile[
                    [1, 32], pl.FP32, pl.MemRef(mem_vec_117, pl.const(16384, pl.INT64), 128), pl.Mem.Vec, pl.TileView(valid_shape=[1, valid_rows_inline1755__ssa_v0])
                ] = pl.tile.reshape(t__tmp_v70_1, [1, 32])
                kv_sq_sum_tail_inline1772__ssa_v3_1: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_13, pl.const(32768, pl.INT64), 128), pl.Mem.Vec] = pl.tile.add(
                    kv_sq_sum_tail_inline1772__ssa_v3, kv_row_sum_tail_inline1748__ssa_v0_1
                )
                kv_sq_sum_tail_inline1772__rv_v2: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_13, pl.const(32768, pl.INT64), 128), pl.Mem.Vec] = pl.yield_(kv_sq_sum_tail_inline1772__ssa_v3_1)
            t__tmp_v71: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 128), pl.Mem.Vec] = pl.tile.muls(kv_sq_sum_tail_inline1772__rv_v2, 0.001953125)
            t__tmp_v72: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 128), pl.Mem.Vec] = pl.tile.adds(t__tmp_v71, 9.9999999999999995e-07)
            t__tmp_v73: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 128), pl.Mem.Vec] = pl.tile.sqrt(t__tmp_v72)
            kv_inv_rms_tail_inline1817__ssa_v0: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_50, pl.const(57344, pl.INT64), 128), pl.Mem.Vec] = pl.tile.recip(t__tmp_v73)
            kv_inv_rms_t_tail_inline1746__ssa_v0: pl.Tile[[32, 1], pl.FP32, pl.MemRef(mem_vec_50, pl.const(57344, pl.INT64), 128), pl.Mem.Vec] = pl.tile.reshape(
                kv_inv_rms_tail_inline1817__ssa_v0, [32, 1]
            )
            for n0_tail_inline1744__idx_v0 in pl.range(0, 384, 128):
                kv_chunk_tail_inline1774__ssa_v1: pl.Tile[
                    [32, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1755__ssa_v0, 64])
                ] = pl.tile.load(kv_fp32_inline1786__rv_v10, [tg_inline1802__ssa_v0, n0_tail_inline1744__idx_v0], [32, 64], [valid_rows_inline1755__ssa_v0, 64], target_memory=pl.Mem.Vec)
                gamma_kv_input_tail_inline1776__ssa_v0: pl.Tile[[64], pl.BF16, pl.MemRef(mem_vec_117, pl.const(16384, pl.INT64), 128), pl.Mem.Vec] = pl.tile.load(
                    gamma_ckv__ssa_v0, [n0_tail_inline1744__idx_v0], [64], [64], target_memory=pl.Mem.Vec
                )
                kv_chunk_tail_inline1774__ssa_v1_1: pl.Tile[
                    [32, 64], pl.FP32, pl.MemRef(mem_vec_14, pl.const(40960, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1755__ssa_v0, 64])
                ] = pl.tile.load(kv_fp32_inline1786__rv_v10, [tg_inline1802__ssa_v0, n0_tail_inline1744__idx_v0 + 64], [32, 64], [valid_rows_inline1755__ssa_v0, 64], target_memory=pl.Mem.Vec)
                gamma_kv_input_tail_inline1776__ssa_v0_1: pl.Tile[[64], pl.BF16, pl.MemRef(mem_vec_118, pl.const(24576, pl.INT64), 128), pl.Mem.Vec] = pl.tile.load(
                    gamma_ckv__ssa_v0, [n0_tail_inline1744__idx_v0 + 64], [64], [64], target_memory=pl.Mem.Vec
                )
                gamma_kv_cast_tail_inline1790__ssa_v0: pl.Tile[[64], pl.FP32, pl.MemRef(mem_vec_9, pl.const(65536, pl.INT64), 256), pl.Mem.Vec] = pl.tile.cast(
                    gamma_kv_input_tail_inline1776__ssa_v0, target_type=pl.FP32, mode="round"
                )
                gamma_kv_chunk_tail_inline1803__ssa_v0: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_9, pl.const(65536, pl.INT64), 256), pl.Mem.Vec] = gamma_kv_cast_tail_inline1790__ssa_v0
                t__tmp_v74: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1755__ssa_v0, 64])] = (
                    pl.tile.row_expand_mul(kv_chunk_tail_inline1774__ssa_v1, kv_inv_rms_t_tail_inline1746__ssa_v0)
                )
                kv_normed_tail_inline1747__ssa_v0: pl.Tile[
                    [32, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1755__ssa_v0, 64])
                ] = pl.tile.col_expand_mul(t__tmp_v74, gamma_kv_chunk_tail_inline1803__ssa_v0)
                kv_normed_bf16_tail_inline1753__ssa_v0: pl.Tile[
                    [32, 64], pl.BF16, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1755__ssa_v0, 64])
                ] = pl.tile.cast(kv_normed_tail_inline1747__ssa_v0, target_type=pl.BF16, mode="rint")
                kv_normed_valid_inline1751__ssa_v0: pl.Tile[
                    [32, 64], pl.BF16, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1755__ssa_v0, 64])
                ] = pl.tile.set_validshape(kv_normed_bf16_tail_inline1753__ssa_v0, valid_rows_inline1755__ssa_v0, 64)
                kv_view_inline1818__store: pl.Tensor[[t_dim_inline1779__ssa_v0, 512], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                    kv_normed_valid_inline1751__ssa_v0, [out_tg_inline1789__ssa_v0, n0_tail_inline1744__idx_v0], kv_view_inline1818__ssa_v0
                )
                gamma_kv_cast_tail_inline1790__ssa_v0_1: pl.Tile[[64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(32768, pl.INT64), 256), pl.Mem.Vec] = pl.tile.cast(
                    gamma_kv_input_tail_inline1776__ssa_v0_1, target_type=pl.FP32, mode="round"
                )
                gamma_kv_chunk_tail_inline1803__ssa_v0_1: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(32768, pl.INT64), 256), pl.Mem.Vec] = gamma_kv_cast_tail_inline1790__ssa_v0_1
                t__tmp_v74_1: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_14, pl.const(40960, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1755__ssa_v0, 64])] = (
                    pl.tile.row_expand_mul(kv_chunk_tail_inline1774__ssa_v1_1, kv_inv_rms_t_tail_inline1746__ssa_v0)
                )
                kv_normed_tail_inline1747__ssa_v0_1: pl.Tile[
                    [32, 64], pl.FP32, pl.MemRef(mem_vec_14, pl.const(40960, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1755__ssa_v0, 64])
                ] = pl.tile.col_expand_mul(t__tmp_v74_1, gamma_kv_chunk_tail_inline1803__ssa_v0_1)
                kv_normed_bf16_tail_inline1753__ssa_v0_1: pl.Tile[
                    [32, 64], pl.BF16, pl.MemRef(mem_vec_14, pl.const(40960, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1755__ssa_v0, 64])
                ] = pl.tile.cast(kv_normed_tail_inline1747__ssa_v0_1, target_type=pl.BF16, mode="rint")
                kv_normed_valid_inline1751__ssa_v0_1: pl.Tile[
                    [32, 64], pl.BF16, pl.MemRef(mem_vec_14, pl.const(40960, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1755__ssa_v0, 64])
                ] = pl.tile.set_validshape(kv_normed_bf16_tail_inline1753__ssa_v0_1, valid_rows_inline1755__ssa_v0, 64)
                kv_view_inline1818__store_1: pl.Tensor[[t_dim_inline1779__ssa_v0, 512], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                    kv_normed_valid_inline1751__ssa_v0_1, [out_tg_inline1789__ssa_v0, n0_tail_inline1744__idx_v0 + 64], kv_view_inline1818__ssa_v0
                )
            kv_chunk_tail_inline1774__ssa_v1_2: pl.Tile[
                [32, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1755__ssa_v0, 64])
            ] = pl.tile.load(kv_fp32_inline1786__rv_v10, [tg_inline1802__ssa_v0, 384], [32, 64], [valid_rows_inline1755__ssa_v0, 64], target_memory=pl.Mem.Vec)
            gamma_kv_input_tail_inline1776__ssa_v0_2: pl.Tile[[64], pl.BF16, pl.MemRef(mem_vec_9, pl.const(65536, pl.INT64), 128), pl.Mem.Vec] = pl.tile.load(
                gamma_ckv__ssa_v0, [384], [64], [64], target_memory=pl.Mem.Vec
            )
            gamma_kv_cast_tail_inline1790__ssa_v0_2: pl.Tile[[64], pl.FP32, pl.MemRef(mem_vec_14, pl.const(40960, pl.INT64), 256), pl.Mem.Vec] = pl.tile.cast(
                gamma_kv_input_tail_inline1776__ssa_v0_2, target_type=pl.FP32, mode="round"
            )
            gamma_kv_chunk_tail_inline1803__ssa_v0_2: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_14, pl.const(40960, pl.INT64), 256), pl.Mem.Vec] = gamma_kv_cast_tail_inline1790__ssa_v0_2
            t__tmp_v74_2: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1755__ssa_v0, 64])] = (
                pl.tile.row_expand_mul(kv_chunk_tail_inline1774__ssa_v1_2, kv_inv_rms_t_tail_inline1746__ssa_v0)
            )
            kv_normed_tail_inline1747__ssa_v0_2: pl.Tile[
                [32, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1755__ssa_v0, 64])
            ] = pl.tile.col_expand_mul(t__tmp_v74_2, gamma_kv_chunk_tail_inline1803__ssa_v0_2)
            kv_normed_bf16_tail_inline1753__ssa_v0_2: pl.Tile[
                [32, 64], pl.BF16, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1755__ssa_v0, 64])
            ] = pl.tile.cast(kv_normed_tail_inline1747__ssa_v0_2, target_type=pl.BF16, mode="rint")
            kv_normed_valid_inline1751__ssa_v0_2: pl.Tile[
                [32, 64], pl.BF16, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1755__ssa_v0, 64])
            ] = pl.tile.set_validshape(kv_normed_bf16_tail_inline1753__ssa_v0_2, valid_rows_inline1755__ssa_v0, 64)
            kv_view_inline1818__store_2: pl.Tensor[[t_dim_inline1779__ssa_v0, 512], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                kv_normed_valid_inline1751__ssa_v0_2, [out_tg_inline1789__ssa_v0, 384], kv_view_inline1818__ssa_v0
            )
            gamma_rope_input_tail_inline1743__ssa_v0: pl.Tile[[64], pl.BF16, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 128), pl.Mem.Vec] = pl.tile.load(
                gamma_ckv__ssa_v0, [448], [64], [64], target_memory=pl.Mem.Vec
            )
            gamma_rope_cast_tail_inline1741__ssa_v0: pl.Tile[[64], pl.FP32, pl.MemRef(mem_vec_14, pl.const(40960, pl.INT64), 256), pl.Mem.Vec] = pl.tile.cast(
                gamma_rope_input_tail_inline1743__ssa_v0, target_type=pl.FP32, mode="round"
            )
            gamma_rope_tail_inline1805__ssa_v0: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_14, pl.const(40960, pl.INT64), 256), pl.Mem.Vec] = gamma_rope_cast_tail_inline1741__ssa_v0
            kv_rope_chunk_tail_inline1740__ssa_v0: pl.Tile[
                [32, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1755__ssa_v0, 64])
            ] = pl.tile.load(kv_fp32_inline1786__rv_v10, [tg_inline1802__ssa_v0, 448], [32, 64], [valid_rows_inline1755__ssa_v0, 64], target_memory=pl.Mem.Vec)
            t__tmp_v75: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1755__ssa_v0, 64])] = (
                pl.tile.row_expand_mul(kv_rope_chunk_tail_inline1740__ssa_v0, kv_inv_rms_t_tail_inline1746__ssa_v0)
            )
            kv_rope_norm_tail_inline1745__ssa_v0: pl.Tile[
                [32, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1755__ssa_v0, 64])
            ] = pl.tile.col_expand_mul(t__tmp_v75, gamma_rope_tail_inline1805__ssa_v0)
            kv_cos_il_tail_inline1739__ssa_v0: pl.Tile[
                [32, 64], pl.FP32, pl.MemRef(mem_vec_14, pl.const(40960, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1755__ssa_v0, 64])
            ] = pl.tile.load(q_rope_cos_il_inline503__ssa_v0, [out_tg_inline1789__ssa_v0, 0], [32, 64], [valid_rows_inline1755__ssa_v0, 64], target_memory=pl.Mem.Vec)
            kv_sin_signed_tail_inline1738__ssa_v0: pl.Tile[
                [32, 64], pl.FP32, pl.MemRef(mem_vec_9, pl.const(65536, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1755__ssa_v0, 64])
            ] = pl.tile.load(q_rope_sin_signed_inline502__ssa_v0, [out_tg_inline1789__ssa_v0, 0], [32, 64], [valid_rows_inline1755__ssa_v0, 64], target_memory=pl.Mem.Vec)
            t__tmp_v76: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(32768, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.full([32, 64], dtype=pl.FP32, value=1.0)
            t__ci_tmp_v0: pl.Tile[[1, 192], pl.FP32, pl.MemRef(mem_vec_50, pl.const(57344, pl.INT64), 768), pl.Mem.Vec] = pl.tile.create([1, 192], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            t__tmp_v77: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_117, pl.const(16384, pl.INT64), 256), pl.Mem.Vec] = pl.tile.ci(
                pl.const(0, pl.INT32), [1, 64], tmp=t__ci_tmp_v0, dtype=pl.INT32, descending=False
            )
            t__tmp_v78: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_50, pl.const(57344, pl.INT64), 256), pl.Mem.Vec] = pl.tile.cast(t__tmp_v77, target_type=pl.FP32, mode="round")
            kv_col_inline1737__ssa_v0: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(32768, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.col_expand_mul(t__tmp_v76, t__tmp_v78)
            t__tmp_v79: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_50, pl.const(57344, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.muls(kv_col_inline1737__ssa_v0, 0.5)
            t__tmp_v80: pl.Tile[[32, 64], pl.INT32, pl.MemRef(mem_vec_50, pl.const(57344, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.cast(t__tmp_v79, target_type=pl.INT32, mode="trunc")
            kv_dup_f_inline1767__ssa_v0: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_50, pl.const(57344, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.cast(t__tmp_v80, target_type=pl.FP32, mode="round")
            t__tmp_v81: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_50, pl.const(57344, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.muls(kv_dup_f_inline1767__ssa_v0, 2.0)
            kv_lane_inline1759__ssa_v0: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_50, pl.const(57344, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.sub(kv_col_inline1737__ssa_v0, t__tmp_v81)
            t__tmp_v82: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(32768, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.adds(kv_col_inline1737__ssa_v0, 1.0)
            t__tmp_v83: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_50, pl.const(57344, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.muls(kv_lane_inline1759__ssa_v0, 2.0)
            kv_swap_f_inline1736__ssa_v0: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(32768, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.sub(t__tmp_v82, t__tmp_v83)
            t__ci_tmp_v1: pl.Tile[[1, 192], pl.FP32, pl.MemRef(mem_vec_50, pl.const(57344, pl.INT64), 768), pl.Mem.Vec] = pl.tile.create([1, 192], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            t__tmp_v84: pl.Tile[[1, 32], pl.INT32, pl.MemRef(mem_vec_117, pl.const(16384, pl.INT64), 128), pl.Mem.Vec] = pl.tile.ci(
                pl.const(0, pl.INT32), [1, 32], tmp=t__ci_tmp_v1, dtype=pl.INT32, descending=False
            )
            t__tmp_v85: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_50, pl.const(57344, pl.INT64), 128), pl.Mem.Vec] = pl.tile.cast(t__tmp_v84, target_type=pl.FP32, mode="round")
            kv_row_seed_inline1785__ssa_v0: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_117, pl.const(16384, pl.INT64), 128), pl.Mem.Vec] = pl.tile.muls(t__tmp_v85, 64.0)
            t__tmp_v86: pl.Tile[[64, 32], pl.FP32, pl.MemRef(mem_vec_50, pl.const(57344, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.full([64, 32], dtype=pl.FP32, value=1.0)
            kv_row_grid_inline1735__ssa_v0: pl.Tile[[64, 32], pl.FP32, pl.MemRef(mem_vec_50, pl.const(57344, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.col_expand_mul(
                t__tmp_v86, kv_row_seed_inline1785__ssa_v0
            )
            transpose_tmp: pl.Tile[[64, 32], pl.FP32, pl.MemRef(mem_vec_117, pl.const(16384, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.create([64, 32], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            kv_row_offset_inline1734__ssa_v0: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_118, pl.const(24576, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.transpose(
                kv_row_grid_inline1735__ssa_v0, 0, 1, transpose_tmp
            )
            t__tmp_v87: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(32768, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.add(kv_swap_f_inline1736__ssa_v0, kv_row_offset_inline1734__ssa_v0)
            kv_swap_idx_tail_inline1742__ssa_v0: pl.Tile[[32, 64], pl.INT32, pl.MemRef(mem_vec_13, pl.const(32768, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.cast(
                t__tmp_v87, target_type=pl.INT32, mode="round"
            )
            kv_gather_tmp_inline1733__ssa_v0: pl.Tile[[32, 64], pl.INT32, pl.MemRef(mem_vec_50, pl.const(57344, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.create(
                [32, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec
            )
            kv_swapped_tail_inline1732__ssa_v0: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_117, pl.const(16384, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.gather(
                kv_rope_norm_tail_inline1745__ssa_v0, kv_swap_idx_tail_inline1742__ssa_v0, kv_gather_tmp_inline1733__ssa_v0
            )
            t__tmp_v88: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1755__ssa_v0, 64])] = pl.tile.mul(
                kv_rope_norm_tail_inline1745__ssa_v0, kv_cos_il_tail_inline1739__ssa_v0
            )
            t__tmp_v89: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_14, pl.const(40960, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.mul(
                kv_swapped_tail_inline1732__ssa_v0, kv_sin_signed_tail_inline1738__ssa_v0
            )
            kv_rope_rot_tail_inline1770__ssa_v0: pl.Tile[
                [32, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1755__ssa_v0, 64])
            ] = pl.tile.add(t__tmp_v88, t__tmp_v89)
            kv_rope_i16_tail_inline1731__ssa_v0: pl.Tile[
                [32, 64], pl.BF16, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1755__ssa_v0, 64])
            ] = pl.tile.cast(kv_rope_rot_tail_inline1770__ssa_v0, target_type=pl.BF16, mode="rint")
            kv_rope_valid_inline1793__ssa_v0: pl.Tile[
                [32, 64], pl.BF16, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1755__ssa_v0, 64])
            ] = pl.tile.set_validshape(kv_rope_i16_tail_inline1731__ssa_v0, valid_rows_inline1755__ssa_v0, 64)
            kv_view_inline1818__store_v0: pl.Tensor[[t_dim_inline1779__ssa_v0, 512], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                kv_rope_valid_inline1793__ssa_v0, [out_tg_inline1789__ssa_v0, 448], kv_view_inline1818__ssa_v0
            )
        return kv_view_inline1818__ssa_v0

    @pl.function(type=pl.FunctionType.Spmd)
    def kv_rms_norm_rope_spmd(
        self,
        tile_rows_inline1768__ssa_v0: pl.Scalar[pl.INDEX],
        tile_base_inline1766__idx_v0: pl.Scalar[pl.INDEX],
        kv_fp32_inline1786__rv_v10: pl.Tensor[[t_matmul_inline1778__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)],
        kv_view_inline1818__ssa_v0: pl.InOut[pl.Tensor[[t_dim_inline1779__ssa_v0, 512], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)]],
        gamma_ckv__ssa_v0: pl.Tensor[[512], pl.BF16, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 1024)],
        q_rope_cos_il_inline503__ssa_v0: pl.Tensor[[t_dim_inline504__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)],
        q_rope_sin_signed_inline502__ssa_v0: pl.Tensor[[t_dim_inline504__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 0)],
        q_rope_swap_idx_inline501__ssa_v0: pl.Tensor[[t_dim_inline504__ssa_v0, 64], pl.INT32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 0)],
    ):
        pl.func_attr({"mx_tensor_views_blocked": True})
        kv_view_inline1818__ssa_v1: pl.Tensor[[t_dim_inline1779__ssa_v0, 512], pl.BF16, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 0)] = self.kv_rms_norm_rope(
            tile_rows_inline1768__ssa_v0,
            tile_base_inline1766__idx_v0,
            kv_fp32_inline1786__rv_v10,
            kv_view_inline1818__ssa_v0,
            gamma_ckv__ssa_v0,
            q_rope_cos_il_inline503__ssa_v0,
            q_rope_sin_signed_inline502__ssa_v0,
            q_rope_swap_idx_inline501__ssa_v0,
            attrs={"arg_directions": [pl.adir.scalar, pl.adir.scalar, pl.adir.input, pl.adir.inout, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input]},
        )

    @pl.function(type=pl.FunctionType.AIC, level=pl.Level.AIC, role=pl.Role.SubWorker)
    def kv_score_proj(
        cmp4_kv_proj_pad_inline669_inline1882__ssa_v0: pl.Out[pl.Tensor[[384, 1024], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 1572864)]],
        cmp4_score_proj_pad_inline671_inline1867__ssa_v0: pl.Out[pl.Tensor[[384, 1024], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 1572864)]],
        t_matmul_inline667_inline1842__ssa_v0: pl.Scalar[pl.INDEX],
        bs_inline666_inline1848__ssa_v0: pl.Scalar[pl.INDEX],
        x_flat_inline668_inline1840__ssa_v0: pl.Tensor[[T_DYN, 4096], pl.BF16, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)],
        cmp_wkv__ssa_v0: pl.Tensor[[1024, 4096], pl.BF16, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 8388608)],
        cmp_wgate__ssa_v0: pl.Tensor[[1024, 4096], pl.BF16, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 8388608)],
    ):
        pl.func_attr({"mx_tensor_views_blocked": True})
        mem_vec_5: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
        mem_acc_7: pl.Ptr = pl.tile.alloc(pl.Mem.Acc, 16384)
        mem_acc_8: pl.Ptr = pl.tile.alloc(pl.Mem.Acc, 16384)
        mem_mat_9: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 65536)
        mem_mat_10: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 65536)
        mem_mat_11: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 65536)
        mem_mat_12: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 65536)
        mem_mat_13: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 65536)
        mem_mat_14: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 65536)
        mem_left_16: pl.Ptr = pl.tile.alloc(pl.Mem.Left, 32768)
        mem_right_17: pl.Ptr = pl.tile.alloc(pl.Mem.Right, 32768)
        mem_left_18: pl.Ptr = pl.tile.alloc(pl.Mem.Left, 32768)
        mem_right_19: pl.Ptr = pl.tile.alloc(pl.Mem.Right, 32768)
        kv_worker_inline678_inline1834__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        for idx_inline672_inline1857__idx_v0, (cmp4_kv_proj_pad_inline669_inline1882__iter_v1, cmp4_score_proj_pad_inline671_inline1867__iter_v1) in pl.range(
            kv_worker_inline678_inline1834__ssa_v0,
            t_matmul_inline667_inline1842__ssa_v0 // 4,
            24,
            init_values=(cmp4_kv_proj_pad_inline669_inline1882__ssa_v0, cmp4_score_proj_pad_inline671_inline1867__ssa_v0),
        ):
            global_row0_inline670_inline1853__ssa_v0: pl.Scalar[pl.INDEX] = idx_inline672_inline1857__idx_v0 // 16 * 64
            o0_inline677_inline1854__ssa_v0: pl.Scalar[pl.INDEX] = idx_inline672_inline1857__idx_v0 % 16 * 64
            x_rows_inline673_inline1860__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(bs_inline666_inline1848__ssa_v0 - global_row0_inline670_inline1853__ssa_v0, 64)
            kv_acc_inline679_inline1861__tile: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_5, pl.const(0, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.create(
                [64, 64], dtype=pl.FP32, target_memory=pl.Mem.Vec
            )
            score_acc_inline674_inline1864__tile: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_5, pl.const(0, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.create(
                [64, 64], dtype=pl.FP32, target_memory=pl.Mem.Vec
            )
            kv_acc_inline679_inline1861__tile_narrowed_storage: pl.Tile[
                [64, 64], pl.FP32, pl.MemRef(mem_acc_7, pl.const(0, pl.INT64), 16384), pl.Mem.Acc, pl.TileView(compact=pl.CompactMode.normal)
            ] = pl.tile.create([64, 64], dtype=pl.FP32, target_memory=pl.Mem.Acc, compact=True)
            kv_acc_inline679_inline1861__tile_narrowed: pl.Tile[
                [64, 64], pl.FP32, pl.MemRef(mem_acc_7, pl.const(0, pl.INT64), 16384), pl.Mem.Acc, pl.TileView(valid_shape=[x_rows_inline673_inline1860__ssa_v0, 64], compact=pl.CompactMode.normal)
            ] = pl.tile.set_validshape(kv_acc_inline679_inline1861__tile_narrowed_storage, x_rows_inline673_inline1860__ssa_v0, 64)
            score_acc_inline674_inline1864__tile_narrowed_storage: pl.Tile[
                [64, 64], pl.FP32, pl.MemRef(mem_acc_8, pl.const(16384, pl.INT64), 16384), pl.Mem.Acc, pl.TileView(compact=pl.CompactMode.normal)
            ] = pl.tile.create([64, 64], dtype=pl.FP32, target_memory=pl.Mem.Acc, compact=True)
            score_acc_inline674_inline1864__tile_narrowed: pl.Tile[
                [64, 64], pl.FP32, pl.MemRef(mem_acc_8, pl.const(16384, pl.INT64), 16384), pl.Mem.Acc, pl.TileView(valid_shape=[x_rows_inline673_inline1860__ssa_v0, 64], compact=pl.CompactMode.normal)
            ] = pl.tile.set_validshape(score_acc_inline674_inline1864__tile_narrowed_storage, x_rows_inline673_inline1860__ssa_v0, 64)
            for kb_inline665_inline1839__idx_v0, (kv_acc_inline679_inline1861__iter_v1, score_acc_inline674_inline1864__iter_v1) in pl.range(
                0, 8, 2, init_values=(kv_acc_inline679_inline1861__tile_narrowed, score_acc_inline674_inline1864__tile_narrowed)
            ):
                k0_inline664_inline1837__ssa_v0: pl.Scalar[pl.INDEX] = kb_inline665_inline1839__idx_v0 * 512
                k0_inline664_inline1837__ssa_v0_1: pl.Scalar[pl.INDEX] = kb_inline665_inline1839__idx_v0 * 512 + 512
                x_tile_inline675_inline1832__tile: pl.Tile[
                    [64, 512], pl.BF16, pl.MemRef(mem_mat_9, pl.const(327680, pl.INT64), 65536), pl.Mem.Mat, pl.TileView(valid_shape=[x_rows_inline673_inline1860__ssa_v0, 512])
                ] = pl.tile.load(
                    x_flat_inline668_inline1840__ssa_v0,
                    [global_row0_inline670_inline1853__ssa_v0, k0_inline664_inline1837__ssa_v0],
                    [64, 512],
                    [x_rows_inline673_inline1860__ssa_v0, 512],
                    target_memory=pl.Mem.Mat,
                )
                wkv_tile_inline663_inline1831__tile: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_mat_10, pl.const(0, pl.INT64), 65536), pl.Mem.Mat] = pl.tile.load(
                    cmp_wkv__ssa_v0, [o0_inline677_inline1854__ssa_v0, k0_inline664_inline1837__ssa_v0], [64, 512], [64, 512], target_memory=pl.Mem.Mat
                )
                wgate_tile_inline662_inline1859__tile: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_mat_11, pl.const(65536, pl.INT64), 65536), pl.Mem.Mat] = pl.tile.load(
                    cmp_wgate__ssa_v0, [o0_inline677_inline1854__ssa_v0, k0_inline664_inline1837__ssa_v0], [64, 512], [64, 512], target_memory=pl.Mem.Mat
                )
                x_tile_inline675_inline1832__tile_1: pl.Tile[
                    [64, 512], pl.BF16, pl.MemRef(mem_mat_12, pl.const(131072, pl.INT64), 65536), pl.Mem.Mat, pl.TileView(valid_shape=[x_rows_inline673_inline1860__ssa_v0, 512])
                ] = pl.tile.load(
                    x_flat_inline668_inline1840__ssa_v0,
                    [global_row0_inline670_inline1853__ssa_v0, k0_inline664_inline1837__ssa_v0_1],
                    [64, 512],
                    [x_rows_inline673_inline1860__ssa_v0, 512],
                    target_memory=pl.Mem.Mat,
                )
                wkv_tile_inline663_inline1831__tile_1: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_mat_13, pl.const(196608, pl.INT64), 65536), pl.Mem.Mat] = pl.tile.load(
                    cmp_wkv__ssa_v0, [o0_inline677_inline1854__ssa_v0, k0_inline664_inline1837__ssa_v0_1], [64, 512], [64, 512], target_memory=pl.Mem.Mat
                )
                wgate_tile_inline662_inline1859__tile_1: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_mat_14, pl.const(262144, pl.INT64), 65536), pl.Mem.Mat] = pl.tile.load(
                    cmp_wgate__ssa_v0, [o0_inline677_inline1854__ssa_v0, k0_inline664_inline1837__ssa_v0_1], [64, 512], [64, 512], target_memory=pl.Mem.Mat
                )
                if k0_inline664_inline1837__ssa_v0 == 0:
                    wkv_tile_inline663_inline1831__tile_t: pl.Tile[
                        [512, 64], pl.BF16, pl.MemRef(mem_mat_10, pl.const(0, pl.INT64), 65536), pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)
                    ] = pl.tile.transpose_view(wkv_tile_inline663_inline1831__tile)
                    kv_acc_inline679_inline1861__tile_l0_init_storage: pl.Tile[
                        [64, 64], pl.FP32, pl.MemRef(mem_acc_7, pl.const(0, pl.INT64), 16384), pl.Mem.Acc, pl.TileView(compact=pl.CompactMode.normal)
                    ] = pl.tile.create([64, 64], dtype=pl.FP32, target_memory=pl.Mem.Acc, compact=True)
                    kv_acc_inline679_inline1861__tile_l0_init: pl.Tile[
                        [64, 64],
                        pl.FP32,
                        pl.MemRef(mem_acc_7, pl.const(0, pl.INT64), 16384),
                        pl.Mem.Acc,
                        pl.TileView(valid_shape=[x_rows_inline673_inline1860__ssa_v0, 64], compact=pl.CompactMode.normal),
                    ] = pl.tile.set_validshape(kv_acc_inline679_inline1861__tile_l0_init_storage, x_rows_inline673_inline1860__ssa_v0, 64)
                    kv_acc_inline679_inline1861__tile_l0_a: pl.Tile[
                        [64, 256],
                        pl.BF16,
                        pl.MemRef(mem_left_16, pl.const(0, pl.INT64), 32768),
                        pl.Mem.Left,
                        pl.TileView(valid_shape=[x_rows_inline673_inline1860__ssa_v0, 256], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal),
                    ] = pl.tile.extract(x_tile_inline675_inline1832__tile, 0, 0, [64, 256], target_memory=pl.Mem.Left)
                    kv_acc_inline679_inline1861__tile_l0_b: pl.Tile[[256, 64], pl.BF16, pl.MemRef(mem_right_17, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                        wkv_tile_inline663_inline1831__tile_t, 0, 0, [256, 64], target_memory=pl.Mem.Right
                    )
                    kv_acc_inline679_inline1861__tile_l0_a_1: pl.Tile[
                        [64, 256],
                        pl.BF16,
                        pl.MemRef(mem_left_18, pl.const(32768, pl.INT64), 32768),
                        pl.Mem.Left,
                        pl.TileView(valid_shape=[x_rows_inline673_inline1860__ssa_v0, 256], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal),
                    ] = pl.tile.extract(x_tile_inline675_inline1832__tile, 0, 256, [64, 256], target_memory=pl.Mem.Left)
                    kv_acc_inline679_inline1861__tile_l0_b_1: pl.Tile[[256, 64], pl.BF16, pl.MemRef(mem_right_19, pl.const(32768, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                        wkv_tile_inline663_inline1831__tile_t, 256, 0, [256, 64], target_memory=pl.Mem.Right
                    )
                    kv_acc_inline679_inline1861__tile_l0_c_acc: pl.Tile[
                        [64, 64],
                        pl.FP32,
                        pl.MemRef(mem_acc_7, pl.const(0, pl.INT64), 16384),
                        pl.Mem.Acc,
                        pl.TileView(valid_shape=[x_rows_inline673_inline1860__ssa_v0, 64], compact=pl.CompactMode.normal),
                    ] = pl.tile.matmul_acc(kv_acc_inline679_inline1861__tile_l0_init, kv_acc_inline679_inline1861__tile_l0_a, kv_acc_inline679_inline1861__tile_l0_b, True)
                    kv_acc_inline679_inline1861__tile_l0_c_acc_1: pl.Tile[
                        [64, 64],
                        pl.FP32,
                        pl.MemRef(mem_acc_7, pl.const(0, pl.INT64), 16384),
                        pl.Mem.Acc,
                        pl.TileView(valid_shape=[x_rows_inline673_inline1860__ssa_v0, 64], compact=pl.CompactMode.normal),
                    ] = pl.tile.matmul_acc(kv_acc_inline679_inline1861__tile_l0_c_acc, kv_acc_inline679_inline1861__tile_l0_a_1, kv_acc_inline679_inline1861__tile_l0_b_1, False)
                    wgate_tile_inline662_inline1859__tile_t: pl.Tile[
                        [512, 64], pl.BF16, pl.MemRef(mem_mat_11, pl.const(65536, pl.INT64), 65536), pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)
                    ] = pl.tile.transpose_view(wgate_tile_inline662_inline1859__tile)
                    score_acc_inline674_inline1864__tile_l0_init_storage: pl.Tile[
                        [64, 64], pl.FP32, pl.MemRef(mem_acc_8, pl.const(16384, pl.INT64), 16384), pl.Mem.Acc, pl.TileView(compact=pl.CompactMode.normal)
                    ] = pl.tile.create([64, 64], dtype=pl.FP32, target_memory=pl.Mem.Acc, compact=True)
                    score_acc_inline674_inline1864__tile_l0_init: pl.Tile[
                        [64, 64],
                        pl.FP32,
                        pl.MemRef(mem_acc_8, pl.const(16384, pl.INT64), 16384),
                        pl.Mem.Acc,
                        pl.TileView(valid_shape=[x_rows_inline673_inline1860__ssa_v0, 64], compact=pl.CompactMode.normal),
                    ] = pl.tile.set_validshape(score_acc_inline674_inline1864__tile_l0_init_storage, x_rows_inline673_inline1860__ssa_v0, 64)
                    score_acc_inline674_inline1864__tile_l0_a: pl.Tile[
                        [64, 256],
                        pl.BF16,
                        pl.MemRef(mem_left_16, pl.const(0, pl.INT64), 32768),
                        pl.Mem.Left,
                        pl.TileView(valid_shape=[x_rows_inline673_inline1860__ssa_v0, 256], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal),
                    ] = pl.tile.extract(x_tile_inline675_inline1832__tile, 0, 0, [64, 256], target_memory=pl.Mem.Left)
                    score_acc_inline674_inline1864__tile_l0_b: pl.Tile[[256, 64], pl.BF16, pl.MemRef(mem_right_17, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                        wgate_tile_inline662_inline1859__tile_t, 0, 0, [256, 64], target_memory=pl.Mem.Right
                    )
                    score_acc_inline674_inline1864__tile_l0_a_1: pl.Tile[
                        [64, 256],
                        pl.BF16,
                        pl.MemRef(mem_left_18, pl.const(32768, pl.INT64), 32768),
                        pl.Mem.Left,
                        pl.TileView(valid_shape=[x_rows_inline673_inline1860__ssa_v0, 256], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal),
                    ] = pl.tile.extract(x_tile_inline675_inline1832__tile, 0, 256, [64, 256], target_memory=pl.Mem.Left)
                    score_acc_inline674_inline1864__tile_l0_b_1: pl.Tile[[256, 64], pl.BF16, pl.MemRef(mem_right_19, pl.const(32768, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                        wgate_tile_inline662_inline1859__tile_t, 256, 0, [256, 64], target_memory=pl.Mem.Right
                    )
                    score_acc_inline674_inline1864__tile_l0_c_acc: pl.Tile[
                        [64, 64],
                        pl.FP32,
                        pl.MemRef(mem_acc_8, pl.const(16384, pl.INT64), 16384),
                        pl.Mem.Acc,
                        pl.TileView(valid_shape=[x_rows_inline673_inline1860__ssa_v0, 64], compact=pl.CompactMode.normal),
                    ] = pl.tile.matmul_acc(score_acc_inline674_inline1864__tile_l0_init, score_acc_inline674_inline1864__tile_l0_a, score_acc_inline674_inline1864__tile_l0_b, True)
                    score_acc_inline674_inline1864__tile_l0_c_acc_1: pl.Tile[
                        [64, 64],
                        pl.FP32,
                        pl.MemRef(mem_acc_8, pl.const(16384, pl.INT64), 16384),
                        pl.Mem.Acc,
                        pl.TileView(valid_shape=[x_rows_inline673_inline1860__ssa_v0, 64], compact=pl.CompactMode.normal),
                    ] = pl.tile.matmul_acc(score_acc_inline674_inline1864__tile_l0_c_acc, score_acc_inline674_inline1864__tile_l0_a_1, score_acc_inline674_inline1864__tile_l0_b_1, False)
                    kv_acc_inline679_inline1861__phi_v5, score_acc_inline674_inline1864__phi_v5 = pl.yield_(
                        kv_acc_inline679_inline1861__tile_l0_c_acc_1, score_acc_inline674_inline1864__tile_l0_c_acc_1
                    )
                else:
                    wkv_tile_inline663_inline1831__tile_t_1: pl.Tile[
                        [512, 64], pl.BF16, pl.MemRef(mem_mat_10, pl.const(0, pl.INT64), 65536), pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)
                    ] = pl.tile.transpose_view(wkv_tile_inline663_inline1831__tile)
                    kv_acc_inline679_inline1861__tile_l0_a_2: pl.Tile[
                        [64, 256],
                        pl.BF16,
                        pl.MemRef(mem_left_16, pl.const(0, pl.INT64), 32768),
                        pl.Mem.Left,
                        pl.TileView(valid_shape=[x_rows_inline673_inline1860__ssa_v0, 256], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal),
                    ] = pl.tile.extract(x_tile_inline675_inline1832__tile, 0, 0, [64, 256], target_memory=pl.Mem.Left)
                    kv_acc_inline679_inline1861__tile_l0_b_2: pl.Tile[[256, 64], pl.BF16, pl.MemRef(mem_right_17, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                        wkv_tile_inline663_inline1831__tile_t_1, 0, 0, [256, 64], target_memory=pl.Mem.Right
                    )
                    kv_acc_inline679_inline1861__tile_l0_a_3: pl.Tile[
                        [64, 256],
                        pl.BF16,
                        pl.MemRef(mem_left_18, pl.const(32768, pl.INT64), 32768),
                        pl.Mem.Left,
                        pl.TileView(valid_shape=[x_rows_inline673_inline1860__ssa_v0, 256], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal),
                    ] = pl.tile.extract(x_tile_inline675_inline1832__tile, 0, 256, [64, 256], target_memory=pl.Mem.Left)
                    kv_acc_inline679_inline1861__tile_l0_b_3: pl.Tile[[256, 64], pl.BF16, pl.MemRef(mem_right_19, pl.const(32768, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                        wkv_tile_inline663_inline1831__tile_t_1, 256, 0, [256, 64], target_memory=pl.Mem.Right
                    )
                    kv_acc_inline679_inline1861__tile_l0_c_acc_2: pl.Tile[
                        [64, 64],
                        pl.FP32,
                        pl.MemRef(mem_acc_7, pl.const(0, pl.INT64), 16384),
                        pl.Mem.Acc,
                        pl.TileView(valid_shape=[x_rows_inline673_inline1860__ssa_v0, 64], compact=pl.CompactMode.normal),
                    ] = pl.tile.matmul_acc(kv_acc_inline679_inline1861__iter_v1, kv_acc_inline679_inline1861__tile_l0_a_2, kv_acc_inline679_inline1861__tile_l0_b_2)
                    kv_acc_inline679_inline1861__tile_l0_c_acc_3: pl.Tile[
                        [64, 64],
                        pl.FP32,
                        pl.MemRef(mem_acc_7, pl.const(0, pl.INT64), 16384),
                        pl.Mem.Acc,
                        pl.TileView(valid_shape=[x_rows_inline673_inline1860__ssa_v0, 64], compact=pl.CompactMode.normal),
                    ] = pl.tile.matmul_acc(kv_acc_inline679_inline1861__tile_l0_c_acc_2, kv_acc_inline679_inline1861__tile_l0_a_3, kv_acc_inline679_inline1861__tile_l0_b_3)
                    wgate_tile_inline662_inline1859__tile_t_1: pl.Tile[
                        [512, 64], pl.BF16, pl.MemRef(mem_mat_11, pl.const(65536, pl.INT64), 65536), pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)
                    ] = pl.tile.transpose_view(wgate_tile_inline662_inline1859__tile)
                    score_acc_inline674_inline1864__tile_l0_a_2: pl.Tile[
                        [64, 256],
                        pl.BF16,
                        pl.MemRef(mem_left_16, pl.const(0, pl.INT64), 32768),
                        pl.Mem.Left,
                        pl.TileView(valid_shape=[x_rows_inline673_inline1860__ssa_v0, 256], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal),
                    ] = pl.tile.extract(x_tile_inline675_inline1832__tile, 0, 0, [64, 256], target_memory=pl.Mem.Left)
                    score_acc_inline674_inline1864__tile_l0_b_2: pl.Tile[[256, 64], pl.BF16, pl.MemRef(mem_right_17, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                        wgate_tile_inline662_inline1859__tile_t_1, 0, 0, [256, 64], target_memory=pl.Mem.Right
                    )
                    score_acc_inline674_inline1864__tile_l0_a_3: pl.Tile[
                        [64, 256],
                        pl.BF16,
                        pl.MemRef(mem_left_18, pl.const(32768, pl.INT64), 32768),
                        pl.Mem.Left,
                        pl.TileView(valid_shape=[x_rows_inline673_inline1860__ssa_v0, 256], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal),
                    ] = pl.tile.extract(x_tile_inline675_inline1832__tile, 0, 256, [64, 256], target_memory=pl.Mem.Left)
                    score_acc_inline674_inline1864__tile_l0_b_3: pl.Tile[[256, 64], pl.BF16, pl.MemRef(mem_right_19, pl.const(32768, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                        wgate_tile_inline662_inline1859__tile_t_1, 256, 0, [256, 64], target_memory=pl.Mem.Right
                    )
                    score_acc_inline674_inline1864__tile_l0_c_acc_2: pl.Tile[
                        [64, 64],
                        pl.FP32,
                        pl.MemRef(mem_acc_8, pl.const(16384, pl.INT64), 16384),
                        pl.Mem.Acc,
                        pl.TileView(valid_shape=[x_rows_inline673_inline1860__ssa_v0, 64], compact=pl.CompactMode.normal),
                    ] = pl.tile.matmul_acc(score_acc_inline674_inline1864__iter_v1, score_acc_inline674_inline1864__tile_l0_a_2, score_acc_inline674_inline1864__tile_l0_b_2)
                    score_acc_inline674_inline1864__tile_l0_c_acc_3: pl.Tile[
                        [64, 64],
                        pl.FP32,
                        pl.MemRef(mem_acc_8, pl.const(16384, pl.INT64), 16384),
                        pl.Mem.Acc,
                        pl.TileView(valid_shape=[x_rows_inline673_inline1860__ssa_v0, 64], compact=pl.CompactMode.normal),
                    ] = pl.tile.matmul_acc(score_acc_inline674_inline1864__tile_l0_c_acc_2, score_acc_inline674_inline1864__tile_l0_a_3, score_acc_inline674_inline1864__tile_l0_b_3)
                    kv_acc_inline679_inline1861__phi_v5, score_acc_inline674_inline1864__phi_v5 = pl.yield_(
                        kv_acc_inline679_inline1861__tile_l0_c_acc_3, score_acc_inline674_inline1864__tile_l0_c_acc_3
                    )
                wkv_tile_inline663_inline1831__tile_t_2: pl.Tile[
                    [512, 64], pl.BF16, pl.MemRef(mem_mat_13, pl.const(196608, pl.INT64), 65536), pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)
                ] = pl.tile.transpose_view(wkv_tile_inline663_inline1831__tile_1)
                kv_acc_inline679_inline1861__tile_l0_a_4: pl.Tile[
                    [64, 256],
                    pl.BF16,
                    pl.MemRef(mem_left_16, pl.const(0, pl.INT64), 32768),
                    pl.Mem.Left,
                    pl.TileView(valid_shape=[x_rows_inline673_inline1860__ssa_v0, 256], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal),
                ] = pl.tile.extract(x_tile_inline675_inline1832__tile_1, 0, 0, [64, 256], target_memory=pl.Mem.Left)
                kv_acc_inline679_inline1861__tile_l0_b_4: pl.Tile[[256, 64], pl.BF16, pl.MemRef(mem_right_17, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                    wkv_tile_inline663_inline1831__tile_t_2, 0, 0, [256, 64], target_memory=pl.Mem.Right
                )
                kv_acc_inline679_inline1861__tile_l0_a_5: pl.Tile[
                    [64, 256],
                    pl.BF16,
                    pl.MemRef(mem_left_18, pl.const(32768, pl.INT64), 32768),
                    pl.Mem.Left,
                    pl.TileView(valid_shape=[x_rows_inline673_inline1860__ssa_v0, 256], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal),
                ] = pl.tile.extract(x_tile_inline675_inline1832__tile_1, 0, 256, [64, 256], target_memory=pl.Mem.Left)
                kv_acc_inline679_inline1861__tile_l0_b_5: pl.Tile[[256, 64], pl.BF16, pl.MemRef(mem_right_19, pl.const(32768, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                    wkv_tile_inline663_inline1831__tile_t_2, 256, 0, [256, 64], target_memory=pl.Mem.Right
                )
                kv_acc_inline679_inline1861__tile_l0_c_acc_4: pl.Tile[
                    [64, 64], pl.FP32, pl.MemRef(mem_acc_7, pl.const(0, pl.INT64), 16384), pl.Mem.Acc, pl.TileView(valid_shape=[x_rows_inline673_inline1860__ssa_v0, 64], compact=pl.CompactMode.normal)
                ] = pl.tile.matmul_acc(kv_acc_inline679_inline1861__phi_v5, kv_acc_inline679_inline1861__tile_l0_a_4, kv_acc_inline679_inline1861__tile_l0_b_4)
                kv_acc_inline679_inline1861__tile_l0_c_acc_5: pl.Tile[
                    [64, 64], pl.FP32, pl.MemRef(mem_acc_7, pl.const(0, pl.INT64), 16384), pl.Mem.Acc, pl.TileView(valid_shape=[x_rows_inline673_inline1860__ssa_v0, 64], compact=pl.CompactMode.normal)
                ] = pl.tile.matmul_acc(kv_acc_inline679_inline1861__tile_l0_c_acc_4, kv_acc_inline679_inline1861__tile_l0_a_5, kv_acc_inline679_inline1861__tile_l0_b_5)
                wgate_tile_inline662_inline1859__tile_t_2: pl.Tile[
                    [512, 64], pl.BF16, pl.MemRef(mem_mat_14, pl.const(262144, pl.INT64), 65536), pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)
                ] = pl.tile.transpose_view(wgate_tile_inline662_inline1859__tile_1)
                score_acc_inline674_inline1864__tile_l0_a_4: pl.Tile[
                    [64, 256],
                    pl.BF16,
                    pl.MemRef(mem_left_16, pl.const(0, pl.INT64), 32768),
                    pl.Mem.Left,
                    pl.TileView(valid_shape=[x_rows_inline673_inline1860__ssa_v0, 256], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal),
                ] = pl.tile.extract(x_tile_inline675_inline1832__tile_1, 0, 0, [64, 256], target_memory=pl.Mem.Left)
                score_acc_inline674_inline1864__tile_l0_b_4: pl.Tile[[256, 64], pl.BF16, pl.MemRef(mem_right_17, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                    wgate_tile_inline662_inline1859__tile_t_2, 0, 0, [256, 64], target_memory=pl.Mem.Right
                )
                score_acc_inline674_inline1864__tile_l0_a_5: pl.Tile[
                    [64, 256],
                    pl.BF16,
                    pl.MemRef(mem_left_18, pl.const(32768, pl.INT64), 32768),
                    pl.Mem.Left,
                    pl.TileView(valid_shape=[x_rows_inline673_inline1860__ssa_v0, 256], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal),
                ] = pl.tile.extract(x_tile_inline675_inline1832__tile_1, 0, 256, [64, 256], target_memory=pl.Mem.Left)
                score_acc_inline674_inline1864__tile_l0_b_5: pl.Tile[[256, 64], pl.BF16, pl.MemRef(mem_right_19, pl.const(32768, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                    wgate_tile_inline662_inline1859__tile_t_2, 256, 0, [256, 64], target_memory=pl.Mem.Right
                )
                score_acc_inline674_inline1864__tile_l0_c_acc_4: pl.Tile[
                    [64, 64],
                    pl.FP32,
                    pl.MemRef(mem_acc_8, pl.const(16384, pl.INT64), 16384),
                    pl.Mem.Acc,
                    pl.TileView(valid_shape=[x_rows_inline673_inline1860__ssa_v0, 64], compact=pl.CompactMode.normal),
                ] = pl.tile.matmul_acc(score_acc_inline674_inline1864__phi_v5, score_acc_inline674_inline1864__tile_l0_a_4, score_acc_inline674_inline1864__tile_l0_b_4)
                score_acc_inline674_inline1864__tile_l0_c_acc_5: pl.Tile[
                    [64, 64],
                    pl.FP32,
                    pl.MemRef(mem_acc_8, pl.const(16384, pl.INT64), 16384),
                    pl.Mem.Acc,
                    pl.TileView(valid_shape=[x_rows_inline673_inline1860__ssa_v0, 64], compact=pl.CompactMode.normal),
                ] = pl.tile.matmul_acc(score_acc_inline674_inline1864__tile_l0_c_acc_4, score_acc_inline674_inline1864__tile_l0_a_5, score_acc_inline674_inline1864__tile_l0_b_5)
                kv_acc_inline679_inline1861__rv_v2, score_acc_inline674_inline1864__rv_v2 = pl.yield_(kv_acc_inline679_inline1861__tile_l0_c_acc_5, score_acc_inline674_inline1864__tile_l0_c_acc_5)
            cmp4_kv_proj_pad_inline669_inline1882__tile: pl.Tensor[[384, 1024], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 1572864)] = pl.tile.store(
                kv_acc_inline679_inline1861__rv_v2, [global_row0_inline670_inline1853__ssa_v0, o0_inline677_inline1854__ssa_v0], cmp4_kv_proj_pad_inline669_inline1882__iter_v1
            )
            cmp4_score_proj_pad_inline671_inline1867__tile: pl.Tensor[[384, 1024], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 1572864)] = pl.tile.store(
                score_acc_inline674_inline1864__rv_v2, [global_row0_inline670_inline1853__ssa_v0, o0_inline677_inline1854__ssa_v0], cmp4_score_proj_pad_inline671_inline1867__iter_v1
            )
            cmp4_kv_proj_pad_inline669_inline1882__rv_v2, cmp4_score_proj_pad_inline671_inline1867__rv_v2 = pl.yield_(
                cmp4_kv_proj_pad_inline669_inline1882__tile, cmp4_score_proj_pad_inline671_inline1867__tile
            )
        return

    @pl.function(type=pl.FunctionType.Spmd)
    def kv_score_proj_spmd(
        self,
        cmp4_kv_proj_pad_inline669_inline1882__ssa_v0: pl.Out[pl.Tensor[[384, 1024], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 1572864)]],
        cmp4_score_proj_pad_inline671_inline1867__ssa_v0: pl.Out[pl.Tensor[[384, 1024], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 1572864)]],
        t_matmul_inline667_inline1842__ssa_v0: pl.Scalar[pl.INDEX],
        bs_inline666_inline1848__ssa_v0: pl.Scalar[pl.INDEX],
        x_flat_inline668_inline1840__ssa_v0: pl.Tensor[[T_DYN, 4096], pl.BF16, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)],
        cmp_wkv__ssa_v0: pl.Tensor[[1024, 4096], pl.BF16, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 8388608)],
        cmp_wgate__ssa_v0: pl.Tensor[[1024, 4096], pl.BF16, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 8388608)],
    ):
        pl.func_attr({"mx_tensor_views_blocked": True})
        self.kv_score_proj(
            cmp4_kv_proj_pad_inline669_inline1882__ssa_v0,
            cmp4_score_proj_pad_inline671_inline1867__ssa_v0,
            t_matmul_inline667_inline1842__ssa_v0,
            bs_inline666_inline1848__ssa_v0,
            x_flat_inline668_inline1840__ssa_v0,
            cmp_wkv__ssa_v0,
            cmp_wgate__ssa_v0,
            attrs={"arg_directions": [pl.adir.output_existing, pl.adir.output_existing, pl.adir.scalar, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.input]},
        )

    @pl.function(type=pl.FunctionType.AIC, level=pl.Level.AIC, role=pl.Role.SubWorker)
    def kv_score_proj_0(
        kv_proj_pad_inline1986__ssa_v0: pl.Out[pl.Tensor[[384, 256], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 393216)]],
        score_proj_pad_inline1993__ssa_v0: pl.Out[pl.Tensor[[384, 256], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 393216)]],
        t_matmul_inline370_inline2004__ssa_v0: pl.Scalar[pl.INDEX],
        bs_inline377_inline2009__ssa_v0: pl.Scalar[pl.INDEX],
        x_flat_inline372_inline1997__ssa_v0: pl.Tensor[[T_DYN, 4096], pl.BF16, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)],
        inner_wkv__ssa_v0: pl.Tensor[[256, 4096], pl.BF16, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 2097152)],
        inner_wgate__ssa_v0: pl.Tensor[[256, 4096], pl.BF16, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 2097152)],
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
        kv_worker_inline374_inline2008__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        for idx_inline371_inline2013__idx_v0, (kv_proj_pad_inline1986__iter_v1, score_proj_pad_inline1993__iter_v1) in pl.range(
            kv_worker_inline374_inline2008__ssa_v0, t_matmul_inline370_inline2004__ssa_v0 // 2, 24, init_values=(kv_proj_pad_inline1986__ssa_v0, score_proj_pad_inline1993__ssa_v0)
        ):
            global_row0_inline380_inline1995__ssa_v0: pl.Scalar[pl.INDEX] = idx_inline371_inline2013__idx_v0 // 8 * 16
            o0_inline381_inline2018__ssa_v0: pl.Scalar[pl.INDEX] = idx_inline371_inline2013__idx_v0 % 8 * 32
            kv_acc_inline375_inline2014__tile: pl.Tile[[16, 32], pl.FP32, pl.MemRef(mem_acc_5, pl.const(0, pl.INT64), 2048), pl.Mem.Acc] = pl.tile.create(
                [16, 32], dtype=pl.FP32, target_memory=pl.Mem.Acc
            )
            score_acc_inline376_inline2015__tile: pl.Tile[[16, 32], pl.FP32, pl.MemRef(mem_acc_6, pl.const(2048, pl.INT64), 2048), pl.Mem.Acc] = pl.tile.create(
                [16, 32], dtype=pl.FP32, target_memory=pl.Mem.Acc
            )
            for kb_inline382_inline2020__idx_v0, (kv_acc_inline375_inline2014__iter_v1, score_acc_inline376_inline2015__iter_v1) in pl.range(
                0, 8, 2, init_values=(kv_acc_inline375_inline2014__tile, score_acc_inline376_inline2015__tile)
            ):
                k0_inline384_inline1989__ssa_v0: pl.Scalar[pl.INDEX] = kb_inline382_inline2020__idx_v0 * 512
                x_rows_inline378_inline1988__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(bs_inline377_inline2009__ssa_v0 - global_row0_inline380_inline1995__ssa_v0, 16)
                k0_inline384_inline1989__ssa_v0_1: pl.Scalar[pl.INDEX] = kb_inline382_inline2020__idx_v0 * 512 + 512
                x_rows_inline378_inline1988__ssa_v0_1: pl.Scalar[pl.INDEX] = pl.min(bs_inline377_inline2009__ssa_v0 - global_row0_inline380_inline1995__ssa_v0, 16)
                x_tile_inline369_inline2036__tile: pl.Tile[
                    [16, 512], pl.BF16, pl.MemRef(mem_mat_7, pl.const(81920, pl.INT64), 16384), pl.Mem.Mat, pl.TileView(valid_shape=[x_rows_inline378_inline1988__ssa_v0, 512])
                ] = pl.tile.load(
                    x_flat_inline372_inline1997__ssa_v0,
                    [global_row0_inline380_inline1995__ssa_v0, k0_inline384_inline1989__ssa_v0],
                    [16, 512],
                    [x_rows_inline378_inline1988__ssa_v0, 512],
                    target_memory=pl.Mem.Mat,
                )
                wkv_tile_inline379_inline1983__tile: pl.Tile[[32, 512], pl.BF16, pl.MemRef(mem_mat_8, pl.const(98304, pl.INT64), 32768), pl.Mem.Mat] = pl.tile.load(
                    inner_wkv__ssa_v0, [o0_inline381_inline2018__ssa_v0, k0_inline384_inline1989__ssa_v0], [32, 512], [32, 512], target_memory=pl.Mem.Mat
                )
                wgate_tile_inline373_inline1981__tile: pl.Tile[[32, 512], pl.BF16, pl.MemRef(mem_mat_9, pl.const(131072, pl.INT64), 32768), pl.Mem.Mat] = pl.tile.load(
                    inner_wgate__ssa_v0, [o0_inline381_inline2018__ssa_v0, k0_inline384_inline1989__ssa_v0], [32, 512], [32, 512], target_memory=pl.Mem.Mat
                )
                x_tile_inline369_inline2036__tile_1: pl.Tile[
                    [16, 512], pl.BF16, pl.MemRef(mem_mat_10, pl.const(0, pl.INT64), 16384), pl.Mem.Mat, pl.TileView(valid_shape=[x_rows_inline378_inline1988__ssa_v0_1, 512])
                ] = pl.tile.load(
                    x_flat_inline372_inline1997__ssa_v0,
                    [global_row0_inline380_inline1995__ssa_v0, k0_inline384_inline1989__ssa_v0_1],
                    [16, 512],
                    [x_rows_inline378_inline1988__ssa_v0_1, 512],
                    target_memory=pl.Mem.Mat,
                )
                wkv_tile_inline379_inline1983__tile_1: pl.Tile[[32, 512], pl.BF16, pl.MemRef(mem_mat_11, pl.const(16384, pl.INT64), 32768), pl.Mem.Mat] = pl.tile.load(
                    inner_wkv__ssa_v0, [o0_inline381_inline2018__ssa_v0, k0_inline384_inline1989__ssa_v0_1], [32, 512], [32, 512], target_memory=pl.Mem.Mat
                )
                wgate_tile_inline373_inline1981__tile_1: pl.Tile[[32, 512], pl.BF16, pl.MemRef(mem_mat_12, pl.const(49152, pl.INT64), 32768), pl.Mem.Mat] = pl.tile.load(
                    inner_wgate__ssa_v0, [o0_inline381_inline2018__ssa_v0, k0_inline384_inline1989__ssa_v0_1], [32, 512], [32, 512], target_memory=pl.Mem.Mat
                )
                wkv_tile_inline379_inline1983__tile_t: pl.Tile[
                    [512, 32], pl.BF16, pl.MemRef(mem_mat_8, pl.const(98304, pl.INT64), 32768), pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)
                ] = pl.tile.transpose_view(wkv_tile_inline379_inline1983__tile)
                x_tile_inline369_inline2036__tile_Left: pl.Tile[
                    [16, 512], pl.BF16, pl.MemRef(mem_left_13, pl.const(0, pl.INT64), 16384), pl.Mem.Left, pl.TileView(valid_shape=[x_rows_inline378_inline1988__ssa_v0, 512])
                ] = pl.tile.move(x_tile_inline369_inline2036__tile, target_memory=pl.Mem.Left)
                wkv_tile_inline379_inline1983__tile_t_Right: pl.Tile[[512, 32], pl.BF16, pl.MemRef(mem_right_14, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.move(
                    wkv_tile_inline379_inline1983__tile_t, target_memory=pl.Mem.Right
                )
                kv_acc_inline375_inline2014__tile_1: pl.Tile[[16, 32], pl.FP32, pl.MemRef(mem_acc_5, pl.const(0, pl.INT64), 2048), pl.Mem.Acc] = pl.tile.matmul_acc(
                    kv_acc_inline375_inline2014__iter_v1, x_tile_inline369_inline2036__tile_Left, wkv_tile_inline379_inline1983__tile_t_Right, k0_inline384_inline1989__ssa_v0 == 0
                )
                wgate_tile_inline373_inline1981__tile_t: pl.Tile[
                    [512, 32], pl.BF16, pl.MemRef(mem_mat_9, pl.const(131072, pl.INT64), 32768), pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)
                ] = pl.tile.transpose_view(wgate_tile_inline373_inline1981__tile)
                wgate_tile_inline373_inline1981__tile_t_Right: pl.Tile[[512, 32], pl.BF16, pl.MemRef(mem_right_14, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.move(
                    wgate_tile_inline373_inline1981__tile_t, target_memory=pl.Mem.Right
                )
                score_acc_inline376_inline2015__tile_1: pl.Tile[[16, 32], pl.FP32, pl.MemRef(mem_acc_6, pl.const(2048, pl.INT64), 2048), pl.Mem.Acc] = pl.tile.matmul_acc(
                    score_acc_inline376_inline2015__iter_v1, x_tile_inline369_inline2036__tile_Left, wgate_tile_inline373_inline1981__tile_t_Right, k0_inline384_inline1989__ssa_v0 == 0
                )
                wkv_tile_inline379_inline1983__tile_t_1: pl.Tile[
                    [512, 32], pl.BF16, pl.MemRef(mem_mat_11, pl.const(16384, pl.INT64), 32768), pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)
                ] = pl.tile.transpose_view(wkv_tile_inline379_inline1983__tile_1)
                x_tile_inline369_inline2036__tile_Left_1: pl.Tile[
                    [16, 512], pl.BF16, pl.MemRef(mem_left_16, pl.const(16384, pl.INT64), 16384), pl.Mem.Left, pl.TileView(valid_shape=[x_rows_inline378_inline1988__ssa_v0_1, 512])
                ] = pl.tile.move(x_tile_inline369_inline2036__tile_1, target_memory=pl.Mem.Left)
                wkv_tile_inline379_inline1983__tile_t_Right_1: pl.Tile[[512, 32], pl.BF16, pl.MemRef(mem_right_17, pl.const(32768, pl.INT64), 32768), pl.Mem.Right] = pl.tile.move(
                    wkv_tile_inline379_inline1983__tile_t_1, target_memory=pl.Mem.Right
                )
                kv_acc_inline375_inline2014__tile_2: pl.Tile[[16, 32], pl.FP32, pl.MemRef(mem_acc_5, pl.const(0, pl.INT64), 2048), pl.Mem.Acc] = pl.tile.matmul_acc(
                    kv_acc_inline375_inline2014__tile_1, x_tile_inline369_inline2036__tile_Left_1, wkv_tile_inline379_inline1983__tile_t_Right_1, k0_inline384_inline1989__ssa_v0_1 == 0
                )
                wgate_tile_inline373_inline1981__tile_t_1: pl.Tile[
                    [512, 32], pl.BF16, pl.MemRef(mem_mat_12, pl.const(49152, pl.INT64), 32768), pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)
                ] = pl.tile.transpose_view(wgate_tile_inline373_inline1981__tile_1)
                wgate_tile_inline373_inline1981__tile_t_Right_1: pl.Tile[[512, 32], pl.BF16, pl.MemRef(mem_right_17, pl.const(32768, pl.INT64), 32768), pl.Mem.Right] = pl.tile.move(
                    wgate_tile_inline373_inline1981__tile_t_1, target_memory=pl.Mem.Right
                )
                score_acc_inline376_inline2015__tile_2: pl.Tile[[16, 32], pl.FP32, pl.MemRef(mem_acc_6, pl.const(2048, pl.INT64), 2048), pl.Mem.Acc] = pl.tile.matmul_acc(
                    score_acc_inline376_inline2015__tile_1, x_tile_inline369_inline2036__tile_Left_1, wgate_tile_inline373_inline1981__tile_t_Right_1, k0_inline384_inline1989__ssa_v0_1 == 0
                )
                kv_acc_inline375_inline2014__rv_v2, score_acc_inline376_inline2015__rv_v2 = pl.yield_(kv_acc_inline375_inline2014__tile_2, score_acc_inline376_inline2015__tile_2)
            kv_proj_pad_inline1986__tile: pl.Tensor[[384, 256], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 393216)] = pl.tile.store(
                kv_acc_inline375_inline2014__rv_v2, [global_row0_inline380_inline1995__ssa_v0, o0_inline381_inline2018__ssa_v0], kv_proj_pad_inline1986__iter_v1
            )
            score_proj_pad_inline1993__tile: pl.Tensor[[384, 256], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 393216)] = pl.tile.store(
                score_acc_inline376_inline2015__rv_v2, [global_row0_inline380_inline1995__ssa_v0, o0_inline381_inline2018__ssa_v0], score_proj_pad_inline1993__iter_v1
            )
            kv_proj_pad_inline1986__rv_v2, score_proj_pad_inline1993__rv_v2 = pl.yield_(kv_proj_pad_inline1986__tile, score_proj_pad_inline1993__tile)
        return kv_proj_pad_inline1986__ssa_v0, score_proj_pad_inline1993__ssa_v0

    @pl.function(type=pl.FunctionType.Spmd)
    def kv_score_proj_spmd_0(
        self,
        kv_proj_pad_inline1986__ssa_v0: pl.Out[pl.Tensor[[384, 256], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 393216)]],
        score_proj_pad_inline1993__ssa_v0: pl.Out[pl.Tensor[[384, 256], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 393216)]],
        t_matmul_inline370_inline2004__ssa_v0: pl.Scalar[pl.INDEX],
        bs_inline377_inline2009__ssa_v0: pl.Scalar[pl.INDEX],
        x_flat_inline372_inline1997__ssa_v0: pl.Tensor[[T_DYN, 4096], pl.BF16, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)],
        inner_wkv__ssa_v0: pl.Tensor[[256, 4096], pl.BF16, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 2097152)],
        inner_wgate__ssa_v0: pl.Tensor[[256, 4096], pl.BF16, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 2097152)],
    ) -> tuple[pl.Tensor[[384, 256], pl.FP32], pl.Tensor[[384, 256], pl.FP32]]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        ret__tmp_v0: pl.Tuple[pl.Tensor[[384, 256], pl.FP32], pl.Tensor[[384, 256], pl.FP32]] = self.kv_score_proj_0(
            kv_proj_pad_inline1986__ssa_v0,
            score_proj_pad_inline1993__ssa_v0,
            t_matmul_inline370_inline2004__ssa_v0,
            bs_inline377_inline2009__ssa_v0,
            x_flat_inline372_inline1997__ssa_v0,
            inner_wkv__ssa_v0,
            inner_wgate__ssa_v0,
            attrs={"arg_directions": [pl.adir.output_existing, pl.adir.output_existing, pl.adir.scalar, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.input]},
        )
        kv_proj_pad_inline1986__rv_v2: pl.Tensor[[384, 256], pl.FP32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 393216)] = ret__tmp_v0[0]
        score_proj_pad_inline1993__rv_v2: pl.Tensor[[384, 256], pl.FP32, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 393216)] = ret__tmp_v0[1]
        return kv_proj_pad_inline1986__ssa_v0, score_proj_pad_inline1993__ssa_v0

    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def kv_touch(
        ori_kv_flat_inline2377__ssa_v0: pl.InOut[pl.Tensor[[ori_block_num_inline2385__ssa_v0 * pl.const(32, pl.INDEX), 512], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)]],
    ) -> pl.Tensor[[ori_block_num_inline2385__ssa_v0 * pl.const(32, pl.INDEX), 512], pl.BF16]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        mem_vec_1: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 1024)
        t__tile: pl.Tile[[1, 512], pl.BF16, pl.MemRef(mem_vec_1, pl.const(0, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.load(
            ori_kv_flat_inline2377__ssa_v0, [0, 0], [1, 512], [1, 512], target_memory=pl.Mem.Vec
        )
        ori_kv_flat_inline2377__tile: pl.Tensor[[ori_block_num_inline2385__ssa_v0 * pl.const(32, pl.INDEX), 512], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)] = pl.tile.store(
            t__tile, [0, 0], ori_kv_flat_inline2377__ssa_v0
        )
        return ori_kv_flat_inline2377__ssa_v0

    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def merge_norm(
        rope_swap_idx_inline574__ssa_v0: pl.Tensor[[16, 64], pl.INT32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 4096)],
        t_dim_inline541__ssa_v0: pl.Scalar[pl.INDEX],
        attn_mi_inline552__ssa_v0: pl.Tensor[[t_heads_inline2333__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)],
        attn_li_inline560__ssa_v0: pl.Tensor[[t_heads_inline2333__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)],
        attn_oi_inline550__ssa_v0: pl.Tensor[[t_heads_inline2333__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)],
        merge_sink_inline570__ssa_v0: pl.Tensor[[64, 1], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 256)],
        rope_cos_il_inline544__ssa_v0: pl.Tensor[[384, 64], pl.FP32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 98304)],
        rope_sin_signed_inline543__ssa_v0: pl.Tensor[[384, 64], pl.FP32, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 98304)],
        o_packed_heads__ssa_v0: pl.Out[pl.Tensor[[3072, 4096], pl.BF16, pl.MemRef("mem_ddr_7", pl.const(0, pl.INT64), 25165824)]],
    ) -> pl.Tensor[[3072, 4096], pl.BF16]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        mem_vec_16: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 4096)
        mem_vec_17: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 4096)
        mem_vec_20: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32768)
        mem_vec_28: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
        mem_vec_29: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        mem_vec_30: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        mem_vec_31: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 4096)
        mem_vec_32: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 4096)
        m_worker_inline553__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        m_swap_inline555__ssa_v0: pl.Tile[[16, 64], pl.INT32, pl.MemRef(mem_vec_20, pl.const(8192, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.load(
            rope_swap_idx_inline574__ssa_v0, [0, 0], [16, 64], [16, 64], target_memory=pl.Mem.Vec
        )
        m_swap_f_inline545__ssa_v0: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_20, pl.const(8192, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(
            m_swap_inline555__ssa_v0, target_type=pl.FP32, mode="round"
        )
        m_swap_source_inline556__ssa_v0: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_20, pl.const(8192, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.adds(m_swap_f_inline545__ssa_v0, 448.0)
        m_row_ids_inline565__ci_tmp_v0: pl.Tile[[1, 192], pl.FP32, pl.MemRef(mem_vec_28, pl.const(40960, pl.INT64), 768), pl.Mem.Vec] = pl.tile.create(
            [1, 192], dtype=pl.FP32, target_memory=pl.Mem.Vec
        )
        m_row_ids_inline565__ssa_v0: pl.Tile[[1, 16], pl.INT32, pl.MemRef(mem_vec_16, pl.const(0, pl.INT64), 64), pl.Mem.Vec] = pl.tile.ci(
            pl.const(0, pl.INT32), [1, 16], tmp=m_row_ids_inline565__ci_tmp_v0, dtype=pl.INT32, descending=False
        )
        m_row_ids_f_inline542__ssa_v0: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_28, pl.const(40960, pl.INT64), 64), pl.Mem.Vec] = pl.tile.cast(
            m_row_ids_inline565__ssa_v0, target_type=pl.FP32, mode="round"
        )
        m_row_offsets_inline558__ssa_v0: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_28, pl.const(40960, pl.INT64), 64), pl.Mem.Vec] = pl.tile.muls(m_row_ids_f_inline542__ssa_v0, 512.0)
        m_row_offsets_col_inline561__ssa_v0: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_28, pl.const(40960, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(m_row_offsets_inline558__ssa_v0, [16, 1])
        m_swap_flat_inline564__ssa_v0: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_20, pl.const(8192, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.row_expand_add(
            m_swap_source_inline556__ssa_v0, m_row_offsets_col_inline561__ssa_v0
        )
        m_swap_idx_inline566__ssa_v0: pl.Tile[[16, 64], pl.INT32, pl.MemRef(mem_vec_16, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(
            m_swap_flat_inline564__ssa_v0, target_type=pl.INT32, mode="round"
        )
        m_gather_tmp_inline571__ssa_v0: pl.Tile[[16, 64], pl.INT32, pl.MemRef(mem_vec_17, pl.const(4096, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.create(
            [16, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec
        )
        for m_idx_inline567__idx_v0 in pl.range(m_worker_inline553__ssa_v0, t_dim_inline541__ssa_v0 * 4, 48):
            m_t_inline559__ssa_v0: pl.Scalar[pl.INDEX] = m_idx_inline567__idx_v0 // 4
            m_h_idx_inline551__ssa_v0: pl.Scalar[pl.INDEX] = m_idx_inline567__idx_v0 - m_t_inline559__ssa_v0 * 4
            m_h0_inline557__ssa_v0: pl.Scalar[pl.INDEX] = m_h_idx_inline551__ssa_v0 * 16
            m_row_inline575__ssa_v0: pl.Scalar[pl.INDEX] = m_idx_inline567__idx_v0 * 16
            m_mi_inline546__ssa_v0: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_28, pl.const(40960, pl.INT64), 64), pl.Mem.Vec] = pl.tile.load(
                attn_mi_inline552__ssa_v0, [m_row_inline575__ssa_v0, 0], [16, 1], [16, 1], target_memory=pl.Mem.Vec
            )
            m_li_inline563__ssa_v0: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_31, pl.const(57856, pl.INT64), 64), pl.Mem.Vec] = pl.tile.load(
                attn_li_inline560__ssa_v0, [m_row_inline575__ssa_v0, 0], [16, 1], [16, 1], target_memory=pl.Mem.Vec
            )
            m_oi_inline539__ssa_v0: pl.Tile[[16, 512], pl.FP32, pl.MemRef(mem_vec_20, pl.const(8192, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.load(
                attn_oi_inline550__ssa_v0, [m_row_inline575__ssa_v0, 0], [16, 512], [16, 512], target_memory=pl.Mem.Vec
            )
            n_sink_bias_inline537__ssa_v0: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_32, pl.const(61952, pl.INT64), 64), pl.Mem.Vec] = pl.tile.load(
                merge_sink_inline570__ssa_v0, [m_h0_inline557__ssa_v0, 0], [16, 1], [16, 1], target_memory=pl.Mem.Vec
            )
            t__rm_a0_tmp_v0: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_28, pl.const(40960, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(m_mi_inline546__ssa_v0, [1, 16])
            t__rm_a1_tmp_v1: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_28, pl.const(40960, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(m_mi_inline546__ssa_v0, [1, 16])
            t__row_major_tmp_v2: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_29, pl.const(57344, pl.INT64), 64), pl.Mem.Vec] = pl.tile.sub(t__rm_a0_tmp_v0, t__rm_a1_tmp_v1)
            t__tmp_v280: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_29, pl.const(57344, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(t__row_major_tmp_v2, [16, 1])
            n_sink_tile_inline536__rm_a0_tmp_v3: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_29, pl.const(57344, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(t__tmp_v280, [1, 16])
            n_sink_tile_inline536__rm_a1_tmp_v4: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_32, pl.const(61952, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(n_sink_bias_inline537__ssa_v0, [1, 16])
            n_sink_tile_inline536__row_major_tmp_v5: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_32, pl.const(61952, pl.INT64), 64), pl.Mem.Vec] = pl.tile.add(
                n_sink_tile_inline536__rm_a0_tmp_v3, n_sink_tile_inline536__rm_a1_tmp_v4
            )
            n_sink_tile_inline536__ssa_v0: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_32, pl.const(61952, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(
                n_sink_tile_inline536__row_major_tmp_v5, [16, 1]
            )
            t__rm_a0_tmp_v6: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_32, pl.const(61952, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(n_sink_tile_inline536__ssa_v0, [1, 16])
            t__rm_a1_tmp_v7: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_28, pl.const(40960, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(m_mi_inline546__ssa_v0, [1, 16])
            t__row_major_tmp_v8: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_28, pl.const(40960, pl.INT64), 64), pl.Mem.Vec] = pl.tile.sub(t__rm_a0_tmp_v6, t__rm_a1_tmp_v7)
            t__tmp_v281: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_28, pl.const(40960, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(t__row_major_tmp_v8, [16, 1])
            t__rm_a0_tmp_v9: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_28, pl.const(40960, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(t__tmp_v281, [1, 16])
            t__row_major_tmp_v10: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_28, pl.const(40960, pl.INT64), 64), pl.Mem.Vec] = pl.tile.exp(t__rm_a0_tmp_v9)
            t__tmp_v282: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_28, pl.const(40960, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(t__row_major_tmp_v10, [16, 1])
            n_denom_inline573__rm_a0_tmp_v11: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_31, pl.const(57856, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(m_li_inline563__ssa_v0, [1, 16])
            n_denom_inline573__rm_a1_tmp_v12: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_28, pl.const(40960, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(t__tmp_v282, [1, 16])
            n_denom_inline573__row_major_tmp_v13: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_28, pl.const(40960, pl.INT64), 64), pl.Mem.Vec] = pl.tile.add(
                n_denom_inline573__rm_a0_tmp_v11, n_denom_inline573__rm_a1_tmp_v12
            )
            n_denom_inline573__ssa_v0: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_28, pl.const(40960, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(n_denom_inline573__row_major_tmp_v13, [16, 1])
            n_full_inline535__ssa_v0: pl.Tile[[16, 512], pl.FP32, pl.MemRef(mem_vec_20, pl.const(8192, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.row_expand_div(
                m_oi_inline539__ssa_v0, n_denom_inline573__ssa_v0
            )
            n_bf16_inline562__ssa_v0: pl.Tile[[16, 512], pl.BF16, pl.MemRef(mem_vec_28, pl.const(40960, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.cast(
                n_full_inline535__ssa_v0, target_type=pl.BF16, mode="rint"
            )
            m_cos_il_inline533__ssa_v0: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_29, pl.const(57344, pl.INT64), 256), pl.Mem.Vec] = pl.tile.load(
                rope_cos_il_inline544__ssa_v0, [m_t_inline559__ssa_v0, 0], [1, 64], [1, 64], target_memory=pl.Mem.Vec
            )
            m_sin_signed_inline569__ssa_v0: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_30, pl.const(57600, pl.INT64), 256), pl.Mem.Vec] = pl.tile.load(
                rope_sin_signed_inline543__ssa_v0, [m_t_inline559__ssa_v0, 0], [1, 64], [1, 64], target_memory=pl.Mem.Vec
            )
            m_swapped_inline532__ssa_v0: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_31, pl.const(57856, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.gather(
                n_full_inline535__ssa_v0, m_swap_idx_inline566__ssa_v0, m_gather_tmp_inline571__ssa_v0
            )
            m_rope_inline534__ssa_v0_textract: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_32, pl.const(61952, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.extract(
                n_full_inline535__ssa_v0, 0, 448, [16, 64], target_memory=pl.Mem.Vec
            )
            t__tmp_v283: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_20, pl.const(8192, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.col_expand_mul(
                m_rope_inline534__ssa_v0_textract, m_cos_il_inline533__ssa_v0
            )
            t__tmp_v284: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_31, pl.const(57856, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.col_expand_mul(
                m_swapped_inline532__ssa_v0, m_sin_signed_inline569__ssa_v0
            )
            m_rot_inline531__ssa_v0: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_20, pl.const(8192, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.add(t__tmp_v283, t__tmp_v284)
            n_rope_bf16_inline530__ssa_v0: pl.Tile[[16, 64], pl.BF16, pl.MemRef(mem_vec_31, pl.const(57856, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(
                m_rot_inline531__ssa_v0, target_type=pl.BF16, mode="rint"
            )
            t__tmp_v285: pl.Tile[[16, 448], pl.BF16, pl.MemRef(mem_vec_28, pl.const(40960, pl.INT64), 16256), pl.Mem.Vec] = pl.tile.slice(n_bf16_inline562__ssa_v0, [16, 448], [0, 0])
            n_full_bf16_inline549__ssa_v0: pl.Tile[[16, 512], pl.BF16, pl.MemRef(mem_vec_20, pl.const(8192, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.concat(t__tmp_v285, n_rope_bf16_inline530__ssa_v0)
            n_group_bf16_inline538__ssa_v0: pl.Tile[[2, 4096], pl.BF16, pl.MemRef(mem_vec_20, pl.const(8192, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.reshape(n_full_bf16_inline549__ssa_v0, [2, 4096])
            n_pack_first_inline529__ssa_v0: pl.Tile[[1, 4096], pl.BF16, pl.MemRef(mem_vec_20, pl.const(8192, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.slice(
                n_group_bf16_inline538__ssa_v0, [1, 4096], [0, 0]
            )
            n_pack_second_inline572__ssa_v0: pl.Tile[[1, 4096], pl.BF16, pl.MemRef(mem_vec_20, pl.const(16384, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.slice(
                n_group_bf16_inline538__ssa_v0, [1, 4096], [1, 0]
            )
            n_pack_row_inline568__ssa_v0: pl.Scalar[pl.INDEX] = m_h0_inline557__ssa_v0 // 8 * 384 + m_t_inline559__ssa_v0
            n_pack_row_second_inline528__ssa_v0: pl.Scalar[pl.INDEX] = n_pack_row_inline568__ssa_v0 + 384
            o_packed_heads__store: pl.Tensor[[3072, 4096], pl.BF16, pl.MemRef("mem_ddr_7", pl.const(0, pl.INT64), 25165824)] = pl.tile.store(
                n_pack_first_inline529__ssa_v0, [n_pack_row_inline568__ssa_v0, 0], o_packed_heads__ssa_v0
            )
            o_packed_heads__store_v0: pl.Tensor[[3072, 4096], pl.BF16, pl.MemRef("mem_ddr_7", pl.const(0, pl.INT64), 25165824)] = pl.tile.store(
                n_pack_second_inline572__ssa_v0, [n_pack_row_second_inline528__ssa_v0, 0], o_packed_heads__ssa_v0
            )
        return o_packed_heads__ssa_v0

    @pl.function(type=pl.FunctionType.Spmd)
    def merge_norm_spmd(
        self,
        rope_swap_idx_inline574__ssa_v0: pl.Tensor[[16, 64], pl.INT32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 4096)],
        t_dim_inline541__ssa_v0: pl.Scalar[pl.INDEX],
        attn_mi_inline552__ssa_v0: pl.Tensor[[t_heads_inline2333__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)],
        attn_li_inline560__ssa_v0: pl.Tensor[[t_heads_inline2333__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)],
        attn_oi_inline550__ssa_v0: pl.Tensor[[t_heads_inline2333__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)],
        merge_sink_inline570__ssa_v0: pl.Tensor[[64, 1], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 256)],
        rope_cos_il_inline544__ssa_v0: pl.Tensor[[384, 64], pl.FP32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 98304)],
        rope_sin_signed_inline543__ssa_v0: pl.Tensor[[384, 64], pl.FP32, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 98304)],
        o_packed_heads__ssa_v0: pl.Out[pl.Tensor[[3072, 4096], pl.BF16, pl.MemRef("mem_ddr_7", pl.const(0, pl.INT64), 25165824)]],
    ) -> pl.Tensor[[3072, 4096], pl.BF16]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        o_packed_heads__ssa_v2: pl.Tensor[[3072, 4096], pl.BF16, pl.MemRef("mem_ddr_8", pl.const(0, pl.INT64), 25165824)] = self.merge_norm(
            rope_swap_idx_inline574__ssa_v0,
            t_dim_inline541__ssa_v0,
            attn_mi_inline552__ssa_v0,
            attn_li_inline560__ssa_v0,
            attn_oi_inline550__ssa_v0,
            merge_sink_inline570__ssa_v0,
            rope_cos_il_inline544__ssa_v0,
            rope_sin_signed_inline543__ssa_v0,
            o_packed_heads__ssa_v0,
            attrs={"arg_directions": [pl.adir.input, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.output_existing]},
        )
        return o_packed_heads__ssa_v0

    @pl.function(type=pl.FunctionType.AIC, level=pl.Level.AIC, role=pl.Role.SubWorker)
    def proj_a_mm(
        t_dim_inline607__ssa_v0: pl.Scalar[pl.INDEX],
        row_base_o_inline637__ssa_v0: pl.Scalar[pl.INDEX],
        o_packed_heads__ssa_v1: pl.Tensor[[3072, 4096], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 25165824)],
        wo_a__ssa_v0: pl.Tensor[[8, 1024, 4096], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 67108864)],
        g_inline628__idx_v0: pl.Scalar[pl.INDEX],
        o_r_pad_inline620__iter_v1: pl.Out[pl.Tensor[[384, 8192], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 12582912)]],
        out_col_g_inline613__ssa_v0: pl.Scalar[pl.INDEX],
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
        pa_unit_inline606__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        pa_rb_inline605__ssa_v0: pl.Scalar[pl.INDEX] = pa_unit_inline606__ssa_v0 // 8
        nf_inline604__ssa_v0: pl.Scalar[pl.INDEX] = pa_unit_inline606__ssa_v0 - pa_rb_inline605__ssa_v0 * 8
        pa_r0_inline602__ssa_v0: pl.Scalar[pl.INDEX] = pa_rb_inline605__ssa_v0 * 128
        pa_rows_inline624__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(t_dim_inline607__ssa_v0 - pa_r0_inline602__ssa_v0, 128)
        pa_src0_inline596__ssa_v0: pl.Scalar[pl.INDEX] = row_base_o_inline637__ssa_v0 + pa_r0_inline602__ssa_v0
        n0_inline623__ssa_v0: pl.Scalar[pl.INDEX] = nf_inline604__ssa_v0 * 128
        xa_first_inline595__tile: pl.Tile[[128, 256], pl.BF16, pl.MemRef(mem_mat_3, pl.const(131072, pl.INT64), 65536), pl.Mem.Mat, pl.TileView(valid_shape=[pa_rows_inline624__ssa_v0, 256])] = (
            pl.tile.load(o_packed_heads__ssa_v1, [pa_src0_inline596__ssa_v0, 0], [128, 256], [pa_rows_inline624__ssa_v0, 256], target_memory=pl.Mem.Mat)
        )
        wa_first_inline600__tile_view2d: pl.Tensor[[8192, 4096], pl.BF16, pl.TensorView(stride=[4096, 1], layout=pl.TensorLayout.ND), pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 67108864)] = (
            pl.tensor.view(wo_a__ssa_v0, [8192, 4096])
        )
        wa_first_inline600__tile: pl.Tile[[128, 256], pl.BF16, pl.MemRef(mem_mat_4, pl.const(196608, pl.INT64), 65536), pl.Mem.Mat] = pl.tile.load(
            wa_first_inline600__tile_view2d, [g_inline628__idx_v0 * 1024 + n0_inline623__ssa_v0, 0], [128, 256], [128, 256], target_memory=pl.Mem.Mat
        )
        wa_first_inline600__tile_t: pl.Tile[
            [256, 128], pl.BF16, pl.MemRef(mem_mat_4, pl.const(196608, pl.INT64), 65536), pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)
        ] = pl.tile.transpose_view(wa_first_inline600__tile)
        acc_a_inline638__tile_l0_init_storage: pl.Tile[[128, 128], pl.FP32, pl.MemRef(mem_acc_5, pl.const(0, pl.INT64), 65536), pl.Mem.Acc, pl.TileView(compact=pl.CompactMode.normal)] = (
            pl.tile.create([128, 128], dtype=pl.FP32, target_memory=pl.Mem.Acc, compact=True)
        )
        acc_a_inline638__tile_l0_init: pl.Tile[
            [128, 128], pl.FP32, pl.MemRef(mem_acc_5, pl.const(0, pl.INT64), 65536), pl.Mem.Acc, pl.TileView(valid_shape=[pa_rows_inline624__ssa_v0, 128], compact=pl.CompactMode.normal)
        ] = pl.tile.set_validshape(acc_a_inline638__tile_l0_init_storage, pa_rows_inline624__ssa_v0, 128)
        acc_a_inline638__tile_l0_a: pl.Tile[
            [128, 128],
            pl.BF16,
            pl.MemRef(mem_left_6, pl.const(0, pl.INT64), 32768),
            pl.Mem.Left,
            pl.TileView(valid_shape=[pa_rows_inline624__ssa_v0, 128], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal),
        ] = pl.tile.extract(xa_first_inline595__tile, 0, 0, [128, 128], target_memory=pl.Mem.Left)
        acc_a_inline638__tile_l0_b: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_7, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
            wa_first_inline600__tile_t, 0, 0, [128, 128], target_memory=pl.Mem.Right
        )
        acc_a_inline638__tile_l0_a_1: pl.Tile[
            [128, 128],
            pl.BF16,
            pl.MemRef(mem_left_8, pl.const(32768, pl.INT64), 32768),
            pl.Mem.Left,
            pl.TileView(valid_shape=[pa_rows_inline624__ssa_v0, 128], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal),
        ] = pl.tile.extract(xa_first_inline595__tile, 0, 128, [128, 128], target_memory=pl.Mem.Left)
        acc_a_inline638__tile_l0_b_1: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_9, pl.const(32768, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
            wa_first_inline600__tile_t, 128, 0, [128, 128], target_memory=pl.Mem.Right
        )
        acc_a_inline638__tile_l0_c_acc: pl.Tile[
            [128, 128], pl.FP32, pl.MemRef(mem_acc_5, pl.const(0, pl.INT64), 65536), pl.Mem.Acc, pl.TileView(valid_shape=[pa_rows_inline624__ssa_v0, 128], compact=pl.CompactMode.normal)
        ] = pl.tile.matmul_acc(acc_a_inline638__tile_l0_init, acc_a_inline638__tile_l0_a, acc_a_inline638__tile_l0_b, True)
        acc_a_inline638__tile_l0_c_acc_1: pl.Tile[
            [128, 128], pl.FP32, pl.MemRef(mem_acc_5, pl.const(0, pl.INT64), 65536), pl.Mem.Acc, pl.TileView(valid_shape=[pa_rows_inline624__ssa_v0, 128], compact=pl.CompactMode.normal)
        ] = pl.tile.matmul_acc(acc_a_inline638__tile_l0_c_acc, acc_a_inline638__tile_l0_a_1, acc_a_inline638__tile_l0_b_1, False)
        for kb_inline609__idx_v0, (acc_a_inline638__iter_v1,) in pl.range(1, 15, 2, init_values=(acc_a_inline638__tile_l0_c_acc_1,)):
            k0_inline599__ssa_v0: pl.Scalar[pl.INDEX] = kb_inline609__idx_v0 * 256
            k0_inline599__ssa_v0_1: pl.Scalar[pl.INDEX] = kb_inline609__idx_v0 * 256 + 256
            xa_k_chunk_inline591__tile: pl.Tile[[128, 256], pl.BF16, pl.MemRef(mem_mat_3, pl.const(131072, pl.INT64), 65536), pl.Mem.Mat, pl.TileView(valid_shape=[pa_rows_inline624__ssa_v0, 256])] = (
                pl.tile.load(o_packed_heads__ssa_v1, [pa_src0_inline596__ssa_v0, k0_inline599__ssa_v0], [128, 256], [pa_rows_inline624__ssa_v0, 256], target_memory=pl.Mem.Mat)
            )
            xa_k_chunk_inline591__tile_1: pl.Tile[
                [128, 256], pl.BF16, pl.MemRef(mem_mat_4, pl.const(196608, pl.INT64), 65536), pl.Mem.Mat, pl.TileView(valid_shape=[pa_rows_inline624__ssa_v0, 256])
            ] = pl.tile.load(o_packed_heads__ssa_v1, [pa_src0_inline596__ssa_v0, k0_inline599__ssa_v0_1], [128, 256], [pa_rows_inline624__ssa_v0, 256], target_memory=pl.Mem.Mat)
            wa_k_chunk_inline589__tile_view2d: pl.Tensor[[8192, 4096], pl.BF16, pl.TensorView(stride=[4096, 1], layout=pl.TensorLayout.ND), pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 67108864)] = (
                pl.tensor.view(wo_a__ssa_v0, [8192, 4096])
            )
            wa_k_chunk_inline589__tile: pl.Tile[[128, 256], pl.BF16, pl.MemRef(mem_mat_13, pl.const(0, pl.INT64), 65536), pl.Mem.Mat] = pl.tile.load(
                wa_k_chunk_inline589__tile_view2d, [g_inline628__idx_v0 * 1024 + n0_inline623__ssa_v0, k0_inline599__ssa_v0], [128, 256], [128, 256], target_memory=pl.Mem.Mat
            )
            wa_k_chunk_inline589__tile_view2d_1: pl.Tensor[
                [8192, 4096], pl.BF16, pl.TensorView(stride=[4096, 1], layout=pl.TensorLayout.ND), pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 67108864)
            ] = pl.tensor.view(wo_a__ssa_v0, [8192, 4096])
            wa_k_chunk_inline589__tile_1: pl.Tile[[128, 256], pl.BF16, pl.MemRef(mem_mat_14, pl.const(65536, pl.INT64), 65536), pl.Mem.Mat] = pl.tile.load(
                wa_k_chunk_inline589__tile_view2d_1, [g_inline628__idx_v0 * 1024 + n0_inline623__ssa_v0, k0_inline599__ssa_v0_1], [128, 256], [128, 256], target_memory=pl.Mem.Mat
            )
            wa_k_chunk_inline589__tile_t: pl.Tile[
                [256, 128], pl.BF16, pl.MemRef(mem_mat_13, pl.const(0, pl.INT64), 65536), pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)
            ] = pl.tile.transpose_view(wa_k_chunk_inline589__tile)
            acc_a_inline638__iter_v1_l0_a: pl.Tile[
                [128, 128],
                pl.BF16,
                pl.MemRef(mem_left_6, pl.const(0, pl.INT64), 32768),
                pl.Mem.Left,
                pl.TileView(valid_shape=[pa_rows_inline624__ssa_v0, 128], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal),
            ] = pl.tile.extract(xa_k_chunk_inline591__tile, 0, 0, [128, 128], target_memory=pl.Mem.Left)
            acc_a_inline638__iter_v1_l0_b: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_7, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                wa_k_chunk_inline589__tile_t, 0, 0, [128, 128], target_memory=pl.Mem.Right
            )
            acc_a_inline638__iter_v1_l0_a_1: pl.Tile[
                [128, 128],
                pl.BF16,
                pl.MemRef(mem_left_8, pl.const(32768, pl.INT64), 32768),
                pl.Mem.Left,
                pl.TileView(valid_shape=[pa_rows_inline624__ssa_v0, 128], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal),
            ] = pl.tile.extract(xa_k_chunk_inline591__tile, 0, 128, [128, 128], target_memory=pl.Mem.Left)
            acc_a_inline638__iter_v1_l0_b_1: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_9, pl.const(32768, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                wa_k_chunk_inline589__tile_t, 128, 0, [128, 128], target_memory=pl.Mem.Right
            )
            acc_a_inline638__iter_v1_l0_c_acc: pl.Tile[
                [128, 128], pl.FP32, pl.MemRef(mem_acc_5, pl.const(0, pl.INT64), 65536), pl.Mem.Acc, pl.TileView(valid_shape=[pa_rows_inline624__ssa_v0, 128], compact=pl.CompactMode.normal)
            ] = pl.tile.matmul_acc(acc_a_inline638__iter_v1, acc_a_inline638__iter_v1_l0_a, acc_a_inline638__iter_v1_l0_b)
            acc_a_inline638__iter_v1_l0_c_acc_1: pl.Tile[
                [128, 128], pl.FP32, pl.MemRef(mem_acc_5, pl.const(0, pl.INT64), 65536), pl.Mem.Acc, pl.TileView(valid_shape=[pa_rows_inline624__ssa_v0, 128], compact=pl.CompactMode.normal)
            ] = pl.tile.matmul_acc(acc_a_inline638__iter_v1_l0_c_acc, acc_a_inline638__iter_v1_l0_a_1, acc_a_inline638__iter_v1_l0_b_1)
            wa_k_chunk_inline589__tile_t_1: pl.Tile[
                [256, 128], pl.BF16, pl.MemRef(mem_mat_14, pl.const(65536, pl.INT64), 65536), pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)
            ] = pl.tile.transpose_view(wa_k_chunk_inline589__tile_1)
            acc_a_inline638__iter_v1_l0_a_2: pl.Tile[
                [128, 128],
                pl.BF16,
                pl.MemRef(mem_left_6, pl.const(0, pl.INT64), 32768),
                pl.Mem.Left,
                pl.TileView(valid_shape=[pa_rows_inline624__ssa_v0, 128], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal),
            ] = pl.tile.extract(xa_k_chunk_inline591__tile_1, 0, 0, [128, 128], target_memory=pl.Mem.Left)
            acc_a_inline638__iter_v1_l0_b_2: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_7, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                wa_k_chunk_inline589__tile_t_1, 0, 0, [128, 128], target_memory=pl.Mem.Right
            )
            acc_a_inline638__iter_v1_l0_a_3: pl.Tile[
                [128, 128],
                pl.BF16,
                pl.MemRef(mem_left_8, pl.const(32768, pl.INT64), 32768),
                pl.Mem.Left,
                pl.TileView(valid_shape=[pa_rows_inline624__ssa_v0, 128], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal),
            ] = pl.tile.extract(xa_k_chunk_inline591__tile_1, 0, 128, [128, 128], target_memory=pl.Mem.Left)
            acc_a_inline638__iter_v1_l0_b_3: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_9, pl.const(32768, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                wa_k_chunk_inline589__tile_t_1, 128, 0, [128, 128], target_memory=pl.Mem.Right
            )
            acc_a_inline638__iter_v1_l0_c_acc_2: pl.Tile[
                [128, 128], pl.FP32, pl.MemRef(mem_acc_5, pl.const(0, pl.INT64), 65536), pl.Mem.Acc, pl.TileView(valid_shape=[pa_rows_inline624__ssa_v0, 128], compact=pl.CompactMode.normal)
            ] = pl.tile.matmul_acc(acc_a_inline638__iter_v1_l0_c_acc_1, acc_a_inline638__iter_v1_l0_a_2, acc_a_inline638__iter_v1_l0_b_2)
            acc_a_inline638__iter_v1_l0_c_acc_3: pl.Tile[
                [128, 128], pl.FP32, pl.MemRef(mem_acc_5, pl.const(0, pl.INT64), 65536), pl.Mem.Acc, pl.TileView(valid_shape=[pa_rows_inline624__ssa_v0, 128], compact=pl.CompactMode.normal)
            ] = pl.tile.matmul_acc(acc_a_inline638__iter_v1_l0_c_acc_2, acc_a_inline638__iter_v1_l0_a_3, acc_a_inline638__iter_v1_l0_b_3)
            acc_a_inline638__rv_v2_main: pl.Tile[
                [128, 128], pl.FP32, pl.MemRef(mem_acc_5, pl.const(0, pl.INT64), 65536), pl.Mem.Acc, pl.TileView(valid_shape=[pa_rows_inline624__ssa_v0, 128], compact=pl.CompactMode.normal)
            ] = pl.yield_(acc_a_inline638__iter_v1_l0_c_acc_3)
        xa_k_chunk_inline591__tile_2: pl.Tile[[128, 256], pl.BF16, pl.MemRef(mem_mat_3, pl.const(131072, pl.INT64), 65536), pl.Mem.Mat, pl.TileView(valid_shape=[pa_rows_inline624__ssa_v0, 256])] = (
            pl.tile.load(o_packed_heads__ssa_v1, [pa_src0_inline596__ssa_v0, 3840], [128, 256], [pa_rows_inline624__ssa_v0, 256], target_memory=pl.Mem.Mat)
        )
        wa_k_chunk_inline589__tile_view2d_2: pl.Tensor[[8192, 4096], pl.BF16, pl.TensorView(stride=[4096, 1], layout=pl.TensorLayout.ND), pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 67108864)] = (
            pl.tensor.view(wo_a__ssa_v0, [8192, 4096])
        )
        wa_k_chunk_inline589__tile_2: pl.Tile[[128, 256], pl.BF16, pl.MemRef(mem_mat_4, pl.const(196608, pl.INT64), 65536), pl.Mem.Mat] = pl.tile.load(
            wa_k_chunk_inline589__tile_view2d_2, [g_inline628__idx_v0 * 1024 + n0_inline623__ssa_v0, 3840], [128, 256], [128, 256], target_memory=pl.Mem.Mat
        )
        wa_k_chunk_inline589__tile_t_2: pl.Tile[
            [256, 128], pl.BF16, pl.MemRef(mem_mat_4, pl.const(196608, pl.INT64), 65536), pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)
        ] = pl.tile.transpose_view(wa_k_chunk_inline589__tile_2)
        acc_a_inline638__iter_v1_l0_a_4: pl.Tile[
            [128, 128],
            pl.BF16,
            pl.MemRef(mem_left_6, pl.const(0, pl.INT64), 32768),
            pl.Mem.Left,
            pl.TileView(valid_shape=[pa_rows_inline624__ssa_v0, 128], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal),
        ] = pl.tile.extract(xa_k_chunk_inline591__tile_2, 0, 0, [128, 128], target_memory=pl.Mem.Left)
        acc_a_inline638__iter_v1_l0_b_4: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_7, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
            wa_k_chunk_inline589__tile_t_2, 0, 0, [128, 128], target_memory=pl.Mem.Right
        )
        acc_a_inline638__iter_v1_l0_a_5: pl.Tile[
            [128, 128],
            pl.BF16,
            pl.MemRef(mem_left_8, pl.const(32768, pl.INT64), 32768),
            pl.Mem.Left,
            pl.TileView(valid_shape=[pa_rows_inline624__ssa_v0, 128], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal),
        ] = pl.tile.extract(xa_k_chunk_inline591__tile_2, 0, 128, [128, 128], target_memory=pl.Mem.Left)
        acc_a_inline638__iter_v1_l0_b_5: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_9, pl.const(32768, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
            wa_k_chunk_inline589__tile_t_2, 128, 0, [128, 128], target_memory=pl.Mem.Right
        )
        acc_a_inline638__iter_v1_l0_c_acc_4: pl.Tile[
            [128, 128], pl.FP32, pl.MemRef(mem_acc_5, pl.const(0, pl.INT64), 65536), pl.Mem.Acc, pl.TileView(valid_shape=[pa_rows_inline624__ssa_v0, 128], compact=pl.CompactMode.normal)
        ] = pl.tile.matmul_acc(acc_a_inline638__rv_v2_main, acc_a_inline638__iter_v1_l0_a_4, acc_a_inline638__iter_v1_l0_b_4)
        acc_a_inline638__iter_v1_l0_c_acc_5: pl.Tile[
            [128, 128], pl.FP32, pl.MemRef(mem_acc_5, pl.const(0, pl.INT64), 65536), pl.Mem.Acc, pl.TileView(valid_shape=[pa_rows_inline624__ssa_v0, 128], compact=pl.CompactMode.normal)
        ] = pl.tile.matmul_acc(acc_a_inline638__iter_v1_l0_c_acc_4, acc_a_inline638__iter_v1_l0_a_5, acc_a_inline638__iter_v1_l0_b_5)
        acc_a_inline638__rv_v2: pl.Tile[
            [128, 128], pl.FP32, pl.MemRef(mem_acc_5, pl.const(0, pl.INT64), 65536), pl.Mem.Acc, pl.TileView(valid_shape=[pa_rows_inline624__ssa_v0, 128], compact=pl.CompactMode.normal)
        ] = acc_a_inline638__iter_v1_l0_c_acc_5
        o_r_pad_inline620__tile: pl.Tensor[[384, 8192], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 12582912)] = pl.tile.store(
            acc_a_inline638__rv_v2, [pa_r0_inline602__ssa_v0, out_col_g_inline613__ssa_v0 + n0_inline623__ssa_v0], o_r_pad_inline620__iter_v1
        )
        return o_r_pad_inline620__iter_v1

    @pl.function(type=pl.FunctionType.Spmd)
    def proj_a_mm_spmd(
        self,
        t_dim_inline607__ssa_v0: pl.Scalar[pl.INDEX],
        row_base_o_inline637__ssa_v0: pl.Scalar[pl.INDEX],
        o_packed_heads__ssa_v1: pl.Tensor[[3072, 4096], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 25165824)],
        wo_a__ssa_v0: pl.Tensor[[8, 1024, 4096], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 67108864)],
        g_inline628__idx_v0: pl.Scalar[pl.INDEX],
        o_r_pad_inline620__iter_v1: pl.Out[pl.Tensor[[384, 8192], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 12582912)]],
        out_col_g_inline613__ssa_v0: pl.Scalar[pl.INDEX],
    ) -> pl.Tensor[[384, 8192], pl.FP32]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        o_r_pad_inline620__ssa_v3: pl.Tensor[[384, 8192], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 12582912)] = self.proj_a_mm(
            t_dim_inline607__ssa_v0,
            row_base_o_inline637__ssa_v0,
            o_packed_heads__ssa_v1,
            wo_a__ssa_v0,
            g_inline628__idx_v0,
            o_r_pad_inline620__iter_v1,
            out_col_g_inline613__ssa_v0,
            attrs={"arg_directions": [pl.adir.scalar, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.scalar, pl.adir.output_existing, pl.adir.scalar]},
        )
        return o_r_pad_inline620__iter_v1

    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def proj_b_act(
        wo_b_scale__ssa_v0: pl.Tensor[[4096], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 16384)],
        attn_out__ssa_v0: pl.Out[pl.Tensor[[T_DYN, 4096], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)]],
        t_dim_inline607__ssa_v0: pl.Scalar[pl.INDEX],
        partials_inline627__rv_v2: pl.Tensor[[384, 32768], pl.INT32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 50331648)],
        act_scale_dq_inline598__rv_v2: pl.Tensor[[8, 384], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 12288)],
    ) -> pl.Tensor[[T_DYN, 4096], pl.BF16]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        mem_vec_4: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 2048)
        mem_vec_5: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
        mem_vec_6: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
        mem_vec_7: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32)
        mem_vec_8: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
        mem_vec_9: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32)
        act_idx_inline583__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        tblk_inline582__ssa_v0: pl.Scalar[pl.INDEX] = act_idx_inline583__ssa_v0 // 8
        nreg_inline581__ssa_v0: pl.Scalar[pl.INDEX] = act_idx_inline583__ssa_v0 - tblk_inline582__ssa_v0 * 8
        ob_n0_inline656__ssa_v0: pl.Scalar[pl.INDEX] = nreg_inline581__ssa_v0 * 512
        t0_inline659__ssa_v1: pl.Scalar[pl.INDEX] = tblk_inline582__ssa_v0 * 32
        wb_scale_inline618__tile: pl.Tile[[512], pl.FP32, pl.MemRef(mem_vec_4, pl.const(0, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.load(
            wo_b_scale__ssa_v0, [ob_n0_inline656__ssa_v0], [512], [512], target_memory=pl.Mem.Vec
        )
        wb_scale_chunk_inline580__tile: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_4, pl.const(0, pl.INT64), 2048), pl.Mem.Vec] = wb_scale_inline618__tile
        for b_tb_inline660__idx_v0, (attn_out__iter_v1,) in pl.range(t0_inline659__ssa_v1, pl.min(t0_inline659__ssa_v1 + 32, t_dim_inline607__ssa_v0), 8, init_values=(attn_out__ssa_v0,)):
            acc_inline579__tile: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_5, pl.const(2048, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.full([8, 512], dtype=pl.FP32, value=0.0)
            for act_g_inline653__idx_v0, (acc_inline579__iter_v1,) in pl.range(0, 8, 2, init_values=(acc_inline579__tile,)):
                p_col0_inline633__ssa_v0: pl.Scalar[pl.INDEX] = act_g_inline653__idx_v0 * 4096 + ob_n0_inline656__ssa_v0
                p_col0_inline633__ssa_v0_1: pl.Scalar[pl.INDEX] = act_g_inline653__idx_v0 * 4096 + ob_n0_inline656__ssa_v0 + 4096
                p_g_inline578__tile: pl.Tile[[8, 512], pl.INT32, pl.MemRef(mem_vec_6, pl.const(18432, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.load(
                    partials_inline627__rv_v2, [b_tb_inline660__idx_v0, p_col0_inline633__ssa_v0], [8, 512], [8, 512], target_memory=pl.Mem.Vec
                )
                g_scale_row_inline577__tile: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_7, pl.const(34816, pl.INT64), 32), pl.Mem.Vec] = pl.tile.load(
                    act_scale_dq_inline598__rv_v2, [act_g_inline653__idx_v0, b_tb_inline660__idx_v0], [1, 8], [1, 8], target_memory=pl.Mem.Vec
                )
                p_g_inline578__tile_1: pl.Tile[[8, 512], pl.INT32, pl.MemRef(mem_vec_8, pl.const(34848, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.load(
                    partials_inline627__rv_v2, [b_tb_inline660__idx_v0, p_col0_inline633__ssa_v0_1], [8, 512], [8, 512], target_memory=pl.Mem.Vec
                )
                g_scale_row_inline577__tile_1: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_9, pl.const(51232, pl.INT64), 32), pl.Mem.Vec] = pl.tile.load(
                    act_scale_dq_inline598__rv_v2, [act_g_inline653__idx_v0 + 1, b_tb_inline660__idx_v0], [1, 8], [1, 8], target_memory=pl.Mem.Vec
                )
                g_scale_inline603__tile: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_7, pl.const(34816, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(g_scale_row_inline577__tile, [8, 1])
                p_g_f32_inline576__tile: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_6, pl.const(18432, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.cast(
                    p_g_inline578__tile, target_type=pl.FP32, mode="none"
                )
                p_g_scaled_inline644__tile: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_6, pl.const(18432, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.row_expand_mul(
                    p_g_f32_inline576__tile, g_scale_inline603__tile
                )
                acc_inline579__tile_1: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_6, pl.const(18432, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.add(acc_inline579__iter_v1, p_g_scaled_inline644__tile)
                g_scale_inline603__tile_1: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_9, pl.const(51232, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(g_scale_row_inline577__tile_1, [8, 1])
                p_g_f32_inline576__tile_1: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_8, pl.const(34848, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.cast(
                    p_g_inline578__tile_1, target_type=pl.FP32, mode="none"
                )
                p_g_scaled_inline644__tile_1: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_8, pl.const(34848, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.row_expand_mul(
                    p_g_f32_inline576__tile_1, g_scale_inline603__tile_1
                )
                acc_inline579__tile_2: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_5, pl.const(2048, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.add(acc_inline579__tile_1, p_g_scaled_inline644__tile_1)
                acc_inline579__rv_v2: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_5, pl.const(2048, pl.INT64), 16384), pl.Mem.Vec] = pl.yield_(acc_inline579__tile_2)
            out_t_inline631__tile: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_5, pl.const(2048, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.col_expand_mul(
                acc_inline579__rv_v2, wb_scale_chunk_inline580__tile
            )
            out_bf16_inline626__tile: pl.Tile[[8, 512], pl.BF16, pl.MemRef(mem_vec_5, pl.const(2048, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.cast(
                out_t_inline631__tile, target_type=pl.BF16, mode="rint"
            )
            attn_out__tile: pl.Tensor[[T_DYN, 4096], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                out_bf16_inline626__tile, [b_tb_inline660__idx_v0, ob_n0_inline656__ssa_v0], attn_out__iter_v1
            )
            attn_out__rv_v2: pl.Tensor[[T_DYN, 4096], pl.BF16, pl.MemRef("mem_ddr_19", pl.const(0, pl.INT64), 0)] = pl.yield_(attn_out__tile)
        return attn_out__ssa_v0

    @pl.function(type=pl.FunctionType.Spmd)
    def proj_b_act_spmd(
        self,
        wo_b_scale__ssa_v0: pl.Tensor[[4096], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 16384)],
        attn_out__ssa_v0: pl.Out[pl.Tensor[[T_DYN, 4096], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)]],
        t_dim_inline607__ssa_v0: pl.Scalar[pl.INDEX],
        partials_inline627__rv_v2: pl.Tensor[[384, 32768], pl.INT32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 50331648)],
        act_scale_dq_inline598__rv_v2: pl.Tensor[[8, 384], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 12288)],
    ) -> pl.Tensor[[T_DYN, 4096], pl.BF16]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        attn_out__rv_v2: pl.Tensor[[T_DYN, 4096], pl.BF16, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 0)] = self.proj_b_act(
            wo_b_scale__ssa_v0,
            attn_out__ssa_v0,
            t_dim_inline607__ssa_v0,
            partials_inline627__rv_v2,
            act_scale_dq_inline598__rv_v2,
            attrs={"arg_directions": [pl.adir.input, pl.adir.output_existing, pl.adir.scalar, pl.adir.input, pl.adir.input]},
        )
        return attn_out__ssa_v0

    @pl.function(type=pl.FunctionType.AIC, level=pl.Level.AIC, role=pl.Role.SubWorker)
    def proj_b_mm(
        partials_inline627__iter_v1: pl.Out[pl.Tensor[[384, 32768], pl.INT32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 50331648)]],
        col_g_inline610__ssa_v0: pl.Scalar[pl.INDEX],
        o_r_i8_pad_inline621__rv_v7: pl.Tensor[[384, 8192], pl.INT8, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 3145728)],
        wo_b__ssa_v0: pl.Tensor[[4096, 8192], pl.INT8, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 33554432)],
        g_inline628__idx_v0: pl.Scalar[pl.INDEX],
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
        pb_unit_inline655__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        tb_inline597__ssa_v0: pl.Scalar[pl.INDEX] = pb_unit_inline655__ssa_v0 // 8
        dc_inline657__ssa_v0: pl.Scalar[pl.INDEX] = pb_unit_inline655__ssa_v0 - tb_inline597__ssa_v0 * 8
        t0_inline659__ssa_v0: pl.Scalar[pl.INDEX] = tb_inline597__ssa_v0 * 128
        d0_inline643__ssa_v0: pl.Scalar[pl.INDEX] = dc_inline657__ssa_v0 * 512
        for nf_inline661__idx_v0, (partials_inline627__iter_v3,) in pl.range(2, init_values=(partials_inline627__iter_v1,)):
            n0_inline623__ssa_v3: pl.Scalar[pl.INDEX] = d0_inline643__ssa_v0 + nf_inline661__idx_v0 * 256
            acc_b_inline616__tile: pl.Tile[[128, 256], pl.INT32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 131072), pl.Mem.Acc] = pl.tile.create([128, 256], dtype=pl.INT32, target_memory=pl.Mem.Acc)
            for kb_inline636__idx_v0, (acc_b_inline616__iter_v1,) in pl.range(0, 4, 2, init_values=(acc_b_inline616__tile,)):
                k0_inline599__ssa_v1: pl.Scalar[pl.INDEX] = col_g_inline610__ssa_v0 + kb_inline636__idx_v0 * 256
                k0_inline599__ssa_v1_1: pl.Scalar[pl.INDEX] = col_g_inline610__ssa_v0 + (kb_inline636__idx_v0 * 256 + 256)
                b_act_inline648__tile: pl.Tile[[128, 256], pl.INT8, pl.MemRef(mem_mat_4, pl.const(0, pl.INT64), 32768), pl.Mem.Mat] = pl.tile.load(
                    o_r_i8_pad_inline621__rv_v7, [t0_inline659__ssa_v0, k0_inline599__ssa_v1], [128, 256], [128, 256], target_memory=pl.Mem.Mat
                )
                b_weight_inline632__tile: pl.Tile[[256, 256], pl.INT8, pl.MemRef(mem_mat_5, pl.const(32768, pl.INT64), 65536), pl.Mem.Mat] = pl.tile.load(
                    wo_b__ssa_v0, [n0_inline623__ssa_v3, k0_inline599__ssa_v1], [256, 256], [256, 256], target_memory=pl.Mem.Mat
                )
                b_act_inline648__tile_1: pl.Tile[[128, 256], pl.INT8, pl.MemRef(mem_mat_6, pl.const(98304, pl.INT64), 32768), pl.Mem.Mat] = pl.tile.load(
                    o_r_i8_pad_inline621__rv_v7, [t0_inline659__ssa_v0, k0_inline599__ssa_v1_1], [128, 256], [128, 256], target_memory=pl.Mem.Mat
                )
                b_weight_inline632__tile_1: pl.Tile[[256, 256], pl.INT8, pl.MemRef(mem_mat_7, pl.const(131072, pl.INT64), 65536), pl.Mem.Mat] = pl.tile.load(
                    wo_b__ssa_v0, [n0_inline623__ssa_v3, k0_inline599__ssa_v1_1], [256, 256], [256, 256], target_memory=pl.Mem.Mat
                )
                b_weight_inline632__tile_t: pl.Tile[
                    [256, 256], pl.INT8, pl.MemRef(mem_mat_5, pl.const(32768, pl.INT64), 65536), pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)
                ] = pl.tile.transpose_view(b_weight_inline632__tile)
                b_act_inline648__tile_Left: pl.Tile[[128, 256], pl.INT8, pl.MemRef(mem_left_8, pl.const(32768, pl.INT64), 32768), pl.Mem.Left] = pl.tile.move(
                    b_act_inline648__tile, target_memory=pl.Mem.Left
                )
                b_weight_inline632__tile_t_Right: pl.Tile[[256, 256], pl.INT8, pl.MemRef(mem_right_9, pl.const(0, pl.INT64), 65536), pl.Mem.Right] = pl.tile.move(
                    b_weight_inline632__tile_t, target_memory=pl.Mem.Right
                )
                acc_b_inline616__tile_1: pl.Tile[[128, 256], pl.INT32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 131072), pl.Mem.Acc] = pl.tile.matmul_acc(
                    acc_b_inline616__iter_v1, b_act_inline648__tile_Left, b_weight_inline632__tile_t_Right, kb_inline636__idx_v0 == 0
                )
                b_weight_inline632__tile_t_1: pl.Tile[
                    [256, 256], pl.INT8, pl.MemRef(mem_mat_7, pl.const(131072, pl.INT64), 65536), pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)
                ] = pl.tile.transpose_view(b_weight_inline632__tile_1)
                b_act_inline648__tile_Left_1: pl.Tile[[128, 256], pl.INT8, pl.MemRef(mem_left_10, pl.const(0, pl.INT64), 32768), pl.Mem.Left] = pl.tile.move(
                    b_act_inline648__tile_1, target_memory=pl.Mem.Left
                )
                b_weight_inline632__tile_t_Right_1: pl.Tile[[256, 256], pl.INT8, pl.MemRef(mem_right_9, pl.const(0, pl.INT64), 65536), pl.Mem.Right] = pl.tile.move(
                    b_weight_inline632__tile_t_1, target_memory=pl.Mem.Right
                )
                acc_b_inline616__tile_2: pl.Tile[[128, 256], pl.INT32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 131072), pl.Mem.Acc] = pl.tile.matmul_acc(
                    acc_b_inline616__tile_1, b_act_inline648__tile_Left_1, b_weight_inline632__tile_t_Right_1, kb_inline636__idx_v0 == -1
                )
                acc_b_inline616__rv_v2: pl.Tile[[128, 256], pl.INT32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 131072), pl.Mem.Acc] = pl.yield_(acc_b_inline616__tile_2)
            partials_inline627__tile: pl.Tensor[[384, 32768], pl.INT32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 50331648)] = pl.tile.store(
                acc_b_inline616__rv_v2, [t0_inline659__ssa_v0, g_inline628__idx_v0 * 4096 + n0_inline623__ssa_v3], partials_inline627__iter_v3
            )
            partials_inline627__rv_v4: pl.Tensor[[384, 32768], pl.INT32, pl.MemRef("mem_ddr_13", pl.const(0, pl.INT64), 50331648)] = pl.yield_(partials_inline627__tile)
        return partials_inline627__iter_v1

    @pl.function(type=pl.FunctionType.Spmd)
    def proj_b_mm_spmd(
        self,
        partials_inline627__iter_v1: pl.Out[pl.Tensor[[384, 32768], pl.INT32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 50331648)]],
        col_g_inline610__ssa_v0: pl.Scalar[pl.INDEX],
        o_r_i8_pad_inline621__rv_v7: pl.Tensor[[384, 8192], pl.INT8, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 3145728)],
        wo_b__ssa_v0: pl.Tensor[[4096, 8192], pl.INT8, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 33554432)],
        g_inline628__idx_v0: pl.Scalar[pl.INDEX],
    ) -> pl.Tensor[[384, 32768], pl.INT32]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        partials_inline627__rv_v4: pl.Tensor[[384, 32768], pl.INT32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 50331648)] = self.proj_b_mm(
            partials_inline627__iter_v1,
            col_g_inline610__ssa_v0,
            o_r_i8_pad_inline621__rv_v7,
            wo_b__ssa_v0,
            g_inline628__idx_v0,
            attrs={"arg_directions": [pl.adir.output_existing, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.scalar]},
        )
        return partials_inline627__iter_v1

    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def q_rope_prepare(
        rope_cos_il_view_inline1632__ssa_v0: pl.Out[pl.Tensor[[t_dim_inline1637__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)]],
        rope_sin_signed_view_inline1646__ssa_v0: pl.Out[pl.Tensor[[t_dim_inline1637__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)]],
        rope_swap_idx_view_inline1631__ssa_v0: pl.Out[pl.Tensor[[t_dim_inline1637__ssa_v0, 64], pl.INT32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)]],
        token_tiles_inline1636__ssa_v0: pl.Scalar[pl.INDEX],
        t_dim_inline1637__ssa_v0: pl.Scalar[pl.INDEX],
        rope_cos_view_inline1642__ssa_v0: pl.Tensor[[t_dim_inline1637__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)],
        rope_sin_view_inline1634__ssa_v0: pl.Tensor[[t_dim_inline1637__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 0)],
    ):
        pl.func_attr({"mx_tensor_views_blocked": True})
        mem_vec_5: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 2048)
        mem_vec_10: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 2048)
        mem_vec_13: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 2048)
        mem_vec_17: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 2048)
        mem_vec_23: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 2048)
        mem_vec_24: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 2048)
        mem_vec_40: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 2048)
        mem_vec_43: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 2048)
        mem_vec_46: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 2048)
        mem_vec_54: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 2048)
        mem_vec_55: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 2048)
        qrp_worker_inline1629__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        qrp_ones_inline1627__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_5, pl.const(16384, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.full([8, 64], dtype=pl.FP32, value=1.0)
        qrp_idx_i32_inline1623__ci_tmp_v0: pl.Tile[[1, 192], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 768), pl.Mem.Vec] = pl.tile.create([1, 192], dtype=pl.FP32, target_memory=pl.Mem.Vec)
        qrp_idx_i32_inline1623__tile: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_13, pl.const(2048, pl.INT64), 256), pl.Mem.Vec] = pl.tile.ci(
            pl.const(0, pl.INT32), [1, 64], tmp=qrp_idx_i32_inline1623__ci_tmp_v0, dtype=pl.INT32, descending=False
        )
        qrp_idx_fp32_inline1624__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 256), pl.Mem.Vec] = pl.tile.cast(
            qrp_idx_i32_inline1623__tile, target_type=pl.FP32, mode="round"
        )
        qrp_col_inline1620__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_5, pl.const(16384, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.col_expand_mul(
            qrp_ones_inline1627__tile, qrp_idx_fp32_inline1624__tile
        )
        qrp_half_inline1633__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.muls(qrp_col_inline1620__tile, 0.5)
        qrp_dup_i32_inline1641__tile: pl.Tile[[8, 64], pl.INT32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(
            qrp_half_inline1633__tile, target_type=pl.INT32, mode="trunc"
        )
        qrp_dup_f_inline1638__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(
            qrp_dup_i32_inline1641__tile, target_type=pl.FP32, mode="round"
        )
        qrp_dup_idx_inline1643__tile: pl.Tile[[8, 64], pl.INT32, pl.MemRef(mem_vec_13, pl.const(2048, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(
            qrp_dup_f_inline1638__tile, target_type=pl.INT32, mode="round"
        )
        t__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.muls(qrp_dup_f_inline1638__tile, 2.0)
        qrp_lane_inline1651__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.sub(qrp_col_inline1620__tile, t__tile)
        qrp_next_col_inline1628__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_5, pl.const(16384, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.adds(qrp_col_inline1620__tile, 1.0)
        qrp_lane_offset_inline1621__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_17, pl.const(4096, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.muls(qrp_lane_inline1651__tile, 2.0)
        qrp_swap_f_inline1645__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_5, pl.const(16384, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.sub(
            qrp_next_col_inline1628__tile, qrp_lane_offset_inline1621__tile
        )
        qrp_swap_idx_inline1644__tile: pl.Tile[[8, 64], pl.INT32, pl.MemRef(mem_vec_5, pl.const(16384, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(
            qrp_swap_f_inline1645__tile, target_type=pl.INT32, mode="round"
        )
        t__tile_1: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.muls(qrp_lane_inline1651__tile, 2.0)
        qrp_sign_inline1647__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.subs(t__tile_1, 1.0)
        for qrp_idx_inline1625__idx_v0, (rope_cos_il_view_inline1632__iter_v1, rope_sin_signed_view_inline1646__iter_v1, rope_swap_idx_view_inline1631__iter_v1) in pl.range(
            qrp_worker_inline1629__ssa_v0,
            token_tiles_inline1636__ssa_v0,
            pl.min(token_tiles_inline1636__ssa_v0, 48),
            init_values=(rope_cos_il_view_inline1632__ssa_v0, rope_sin_signed_view_inline1646__ssa_v0, rope_swap_idx_view_inline1631__ssa_v0),
        ):
            qrp_t0_inline1652__ssa_v0: pl.Scalar[pl.INDEX] = qrp_idx_inline1625__idx_v0 * 8
            qrp_valid_rows_inline1650__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(t_dim_inline1637__ssa_v0 - qrp_t0_inline1652__ssa_v0, 8)
            if qrp_valid_rows_inline1650__ssa_v0 == 8:
                qrp_cos_rows_full_inline1626__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_17, pl.const(4096, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.load(
                    rope_cos_view_inline1642__ssa_v0, [qrp_t0_inline1652__ssa_v0, 0], [8, 64], [8, 64], target_memory=pl.Mem.Vec
                )
                qrp_sin_rows_full_inline1653__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_23, pl.const(6144, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.load(
                    rope_sin_view_inline1634__ssa_v0, [qrp_t0_inline1652__ssa_v0, 0], [8, 64], [8, 64], target_memory=pl.Mem.Vec
                )
                qrp_cos_full_inline1654__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_17, pl.const(4096, pl.INT64), 2048), pl.Mem.Vec] = qrp_cos_rows_full_inline1626__tile
                qrp_sin_full_inline1639__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_23, pl.const(6144, pl.INT64), 2048), pl.Mem.Vec] = qrp_sin_rows_full_inline1653__tile
                gather_acc_init: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_24, pl.const(8192, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.create([8, 64], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                for gather_lv, (gather_ia,) in pl.range(8, init_values=(gather_acc_init,)):
                    gather_inp_row: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_17, pl.const(4096, pl.INT64), 256), pl.Mem.Vec] = pl.tile.slice(
                        qrp_cos_full_inline1654__ssa_v0, [1, 64], [gather_lv, 0], [1, 64]
                    )
                    gather_idx_row: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_13, pl.const(2048, pl.INT64), 256), pl.Mem.Vec] = pl.tile.slice(
                        qrp_dup_idx_inline1643__tile, [1, 64], [gather_lv, 0], [1, 64]
                    )
                    gather_row_tmp: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_40, pl.const(10240, pl.INT64), 256), pl.Mem.Vec] = pl.tile.create([1, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec)
                    gather_row: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_43, pl.const(12288, pl.INT64), 256), pl.Mem.Vec] = pl.tile.gather(gather_inp_row, gather_idx_row, gather_row_tmp)
                    gather_asmbl: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_24, pl.const(8192, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.assemble(gather_ia, gather_row, [gather_lv, 0])
                    gather_rv: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_24, pl.const(8192, pl.INT64), 2048), pl.Mem.Vec] = pl.yield_(gather_asmbl)
                qrp_cos_il_full_inline1649__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_24, pl.const(8192, pl.INT64), 2048), pl.Mem.Vec] = gather_rv
                gather_acc_init_1: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_17, pl.const(4096, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.create([8, 64], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                for gather_lv_1, (gather_ia_1,) in pl.range(8, init_values=(gather_acc_init_1,)):
                    gather_inp_row_1: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_23, pl.const(6144, pl.INT64), 256), pl.Mem.Vec] = pl.tile.slice(
                        qrp_sin_full_inline1639__ssa_v0, [1, 64], [gather_lv_1, 0], [1, 64]
                    )
                    gather_idx_row_1: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_13, pl.const(2048, pl.INT64), 256), pl.Mem.Vec] = pl.tile.slice(
                        qrp_dup_idx_inline1643__tile, [1, 64], [gather_lv_1, 0], [1, 64]
                    )
                    gather_row_tmp_1: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_40, pl.const(10240, pl.INT64), 256), pl.Mem.Vec] = pl.tile.create([1, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec)
                    gather_row_1: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_43, pl.const(12288, pl.INT64), 256), pl.Mem.Vec] = pl.tile.gather(gather_inp_row_1, gather_idx_row_1, gather_row_tmp_1)
                    gather_asmbl_1: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_17, pl.const(4096, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.assemble(gather_ia_1, gather_row_1, [gather_lv_1, 0])
                    gather_rv_1: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_17, pl.const(4096, pl.INT64), 2048), pl.Mem.Vec] = pl.yield_(gather_asmbl_1)
                qrp_sin_il_full_inline1640__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_17, pl.const(4096, pl.INT64), 2048), pl.Mem.Vec] = gather_rv_1
                qrp_sin_signed_full_inline1618__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_17, pl.const(4096, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.mul(
                    qrp_sin_il_full_inline1640__tile, qrp_sign_inline1647__tile
                )
                rope_cos_il_view_inline1632__tile: pl.Tensor[[t_dim_inline1637__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                    qrp_cos_il_full_inline1649__tile, [qrp_t0_inline1652__ssa_v0, 0], rope_cos_il_view_inline1632__iter_v1
                )
                rope_sin_signed_view_inline1646__tile: pl.Tensor[[t_dim_inline1637__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                    qrp_sin_signed_full_inline1618__tile, [qrp_t0_inline1652__ssa_v0, 0], rope_sin_signed_view_inline1646__iter_v1
                )
                rope_swap_idx_view_inline1631__tile: pl.Tensor[[t_dim_inline1637__ssa_v0, 64], pl.INT32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                    qrp_swap_idx_inline1644__tile, [qrp_t0_inline1652__ssa_v0, 0], rope_swap_idx_view_inline1631__iter_v1
                )
                rope_cos_il_view_inline1632__phi_v4, rope_sin_signed_view_inline1646__phi_v4, rope_swap_idx_view_inline1631__phi_v4 = pl.yield_(
                    rope_cos_il_view_inline1632__tile, rope_sin_signed_view_inline1646__tile, rope_swap_idx_view_inline1631__tile
                )
            else:
                qrp_cos_rows_tail_inline1617__ssa_v0: pl.Tile[
                    [8, 64], pl.FP32, pl.MemRef(mem_vec_17, pl.const(4096, pl.INT64), 2048), pl.Mem.Vec, pl.TileView(valid_shape=[qrp_valid_rows_inline1650__ssa_v0, 64])
                ] = pl.tile.load(rope_cos_view_inline1642__ssa_v0, [qrp_t0_inline1652__ssa_v0, 0], [8, 64], [qrp_valid_rows_inline1650__ssa_v0, 64], target_memory=pl.Mem.Vec)
                qrp_sin_rows_tail_inline1616__ssa_v0: pl.Tile[
                    [8, 64], pl.FP32, pl.MemRef(mem_vec_23, pl.const(6144, pl.INT64), 2048), pl.Mem.Vec, pl.TileView(valid_shape=[qrp_valid_rows_inline1650__ssa_v0, 64])
                ] = pl.tile.load(rope_sin_view_inline1634__ssa_v0, [qrp_t0_inline1652__ssa_v0, 0], [8, 64], [qrp_valid_rows_inline1650__ssa_v0, 64], target_memory=pl.Mem.Vec)
                t__tmp_v13: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_24, pl.const(8192, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.full([8, 64], dtype=pl.FP32, value=1.0)
                t__ci_tmp_v1: pl.Tile[[1, 192], pl.FP32, pl.MemRef(mem_vec_40, pl.const(10240, pl.INT64), 768), pl.Mem.Vec] = pl.tile.create([1, 192], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                t__tmp_v14: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_43, pl.const(12288, pl.INT64), 256), pl.Mem.Vec] = pl.tile.ci(
                    pl.const(0, pl.INT32), [1, 64], tmp=t__ci_tmp_v1, dtype=pl.INT32, descending=False
                )
                t__tmp_v15: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_40, pl.const(10240, pl.INT64), 256), pl.Mem.Vec] = pl.tile.cast(t__tmp_v14, target_type=pl.FP32, mode="round")
                qrp_tail_col_inline1622__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_24, pl.const(8192, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.col_expand_mul(t__tmp_v13, t__tmp_v15)
                t__tmp_v16: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_40, pl.const(10240, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.muls(qrp_tail_col_inline1622__ssa_v0, 0.5)
                t__tmp_v17: pl.Tile[[8, 64], pl.INT32, pl.MemRef(mem_vec_40, pl.const(10240, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(t__tmp_v16, target_type=pl.INT32, mode="trunc")
                qrp_tail_dup_f_inline1619__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_40, pl.const(10240, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(
                    t__tmp_v17, target_type=pl.FP32, mode="round"
                )
                t__tmp_v18: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_43, pl.const(12288, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.muls(qrp_tail_dup_f_inline1619__ssa_v0, 2.0)
                qrp_tail_lane_inline1615__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_43, pl.const(12288, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.sub(
                    qrp_tail_col_inline1622__ssa_v0, t__tmp_v18
                )
                t__tmp_v19: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_24, pl.const(8192, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.adds(qrp_tail_col_inline1622__ssa_v0, 1.0)
                t__tmp_v20: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_46, pl.const(14336, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.muls(qrp_tail_lane_inline1615__ssa_v0, 2.0)
                qrp_tail_swap_f_inline1630__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_24, pl.const(8192, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.sub(t__tmp_v19, t__tmp_v20)
                t__ci_tmp_v2: pl.Tile[[1, 192], pl.FP32, pl.MemRef(mem_vec_46, pl.const(14336, pl.INT64), 768), pl.Mem.Vec] = pl.tile.create([1, 192], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                t__tmp_v21: pl.Tile[[1, 8], pl.INT32, pl.MemRef(mem_vec_54, pl.const(18432, pl.INT64), 32), pl.Mem.Vec] = pl.tile.ci(
                    pl.const(0, pl.INT32), [1, 8], tmp=t__ci_tmp_v2, dtype=pl.INT32, descending=False
                )
                t__tmp_v22: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_46, pl.const(14336, pl.INT64), 32), pl.Mem.Vec] = pl.tile.cast(t__tmp_v21, target_type=pl.FP32, mode="round")
                qrp_row_seed_inline1648__ssa_v0: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_54, pl.const(18432, pl.INT64), 32), pl.Mem.Vec] = pl.tile.muls(t__tmp_v22, 64.0)
                t__tmp_v23: pl.Tile[[64, 8], pl.FP32, pl.MemRef(mem_vec_46, pl.const(14336, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.full([64, 8], dtype=pl.FP32, value=1.0)
                qrp_row_grid_inline1614__ssa_v0: pl.Tile[[64, 8], pl.FP32, pl.MemRef(mem_vec_46, pl.const(14336, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.col_expand_mul(
                    t__tmp_v23, qrp_row_seed_inline1648__ssa_v0
                )
                transpose_tmp: pl.Tile[[64, 8], pl.FP32, pl.MemRef(mem_vec_54, pl.const(18432, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.create([64, 8], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                qrp_row_offset_inline1613__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_55, pl.const(20480, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.transpose(
                    qrp_row_grid_inline1614__ssa_v0, 0, 1, transpose_tmp
                )
                t__tmp_v24: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_40, pl.const(10240, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.add(
                    qrp_tail_dup_f_inline1619__ssa_v0, qrp_row_offset_inline1613__ssa_v0
                )
                qrp_dup_idx_tail_inline1612__ssa_v0: pl.Tile[[8, 64], pl.INT32, pl.MemRef(mem_vec_40, pl.const(10240, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(
                    t__tmp_v24, target_type=pl.INT32, mode="round"
                )
                qrp_gather_tmp_inline1611__ssa_v0: pl.Tile[[8, 64], pl.INT32, pl.MemRef(mem_vec_46, pl.const(14336, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.create(
                    [8, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec
                )
                qrp_cos_il_tail_inline1635__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_54, pl.const(18432, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.gather(
                    qrp_cos_rows_tail_inline1617__ssa_v0, qrp_dup_idx_tail_inline1612__ssa_v0, qrp_gather_tmp_inline1611__ssa_v0
                )
                qrp_sin_il_tail_inline1610__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_17, pl.const(4096, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.gather(
                    qrp_sin_rows_tail_inline1616__ssa_v0, qrp_dup_idx_tail_inline1612__ssa_v0, qrp_gather_tmp_inline1611__ssa_v0
                )
                t__tmp_v25: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_23, pl.const(6144, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.muls(qrp_tail_lane_inline1615__ssa_v0, 2.0)
                qrp_tail_sign_inline1609__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_23, pl.const(6144, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.subs(t__tmp_v25, 1.0)
                qrp_sin_signed_tail_inline1608__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_17, pl.const(4096, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.mul(
                    qrp_sin_il_tail_inline1610__ssa_v0, qrp_tail_sign_inline1609__ssa_v0
                )
                t__tmp_v26: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_54, pl.const(18432, pl.INT64), 2048), pl.Mem.Vec, pl.TileView(valid_shape=[qrp_valid_rows_inline1650__ssa_v0, 64])] = (
                    pl.tile.set_validshape(qrp_cos_il_tail_inline1635__ssa_v0, qrp_valid_rows_inline1650__ssa_v0, 64)
                )
                pl.tile.store(t__tmp_v26, [qrp_t0_inline1652__ssa_v0, 0], rope_cos_il_view_inline1632__iter_v1)
                t__tmp_v27: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_17, pl.const(4096, pl.INT64), 2048), pl.Mem.Vec, pl.TileView(valid_shape=[qrp_valid_rows_inline1650__ssa_v0, 64])] = (
                    pl.tile.set_validshape(qrp_sin_signed_tail_inline1608__ssa_v0, qrp_valid_rows_inline1650__ssa_v0, 64)
                )
                pl.tile.store(t__tmp_v27, [qrp_t0_inline1652__ssa_v0, 0], rope_sin_signed_view_inline1646__iter_v1)
                t__tmp_v28: pl.Tile[[8, 64], pl.INT32, pl.MemRef(mem_vec_17, pl.const(4096, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(
                    qrp_tail_swap_f_inline1630__ssa_v0, target_type=pl.INT32, mode="round"
                )
                t__tmp_v29: pl.Tile[[8, 64], pl.INT32, pl.MemRef(mem_vec_17, pl.const(4096, pl.INT64), 2048), pl.Mem.Vec, pl.TileView(valid_shape=[qrp_valid_rows_inline1650__ssa_v0, 64])] = (
                    pl.tile.set_validshape(t__tmp_v28, qrp_valid_rows_inline1650__ssa_v0, 64)
                )
                pl.tile.store(t__tmp_v29, [qrp_t0_inline1652__ssa_v0, 0], rope_swap_idx_view_inline1631__iter_v1)
                rope_cos_il_view_inline1632__phi_v4, rope_sin_signed_view_inline1646__phi_v4, rope_swap_idx_view_inline1631__phi_v4 = pl.yield_(
                    rope_cos_il_view_inline1632__iter_v1, rope_sin_signed_view_inline1646__iter_v1, rope_swap_idx_view_inline1631__iter_v1
                )
            rope_cos_il_view_inline1632__rv_v2, rope_sin_signed_view_inline1646__rv_v2, rope_swap_idx_view_inline1631__rv_v2 = pl.yield_(
                rope_cos_il_view_inline1632__phi_v4, rope_sin_signed_view_inline1646__phi_v4, rope_swap_idx_view_inline1631__phi_v4
            )
        return

    @pl.function(type=pl.FunctionType.Spmd)
    def q_rope_prepare_spmd(
        self,
        rope_cos_il_view_inline1632__ssa_v0: pl.Out[pl.Tensor[[t_dim_inline1637__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)]],
        rope_sin_signed_view_inline1646__ssa_v0: pl.Out[pl.Tensor[[t_dim_inline1637__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)]],
        rope_swap_idx_view_inline1631__ssa_v0: pl.Out[pl.Tensor[[t_dim_inline1637__ssa_v0, 64], pl.INT32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)]],
        token_tiles_inline1636__ssa_v0: pl.Scalar[pl.INDEX],
        t_dim_inline1637__ssa_v0: pl.Scalar[pl.INDEX],
        rope_cos_view_inline1642__ssa_v0: pl.Tensor[[t_dim_inline1637__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)],
        rope_sin_view_inline1634__ssa_v0: pl.Tensor[[t_dim_inline1637__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 0)],
    ):
        pl.func_attr({"mx_tensor_views_blocked": True})
        self.q_rope_prepare(
            rope_cos_il_view_inline1632__ssa_v0,
            rope_sin_signed_view_inline1646__ssa_v0,
            rope_swap_idx_view_inline1631__ssa_v0,
            token_tiles_inline1636__ssa_v0,
            t_dim_inline1637__ssa_v0,
            rope_cos_view_inline1642__ssa_v0,
            rope_sin_view_inline1634__ssa_v0,
            attrs={"arg_directions": [pl.adir.output_existing, pl.adir.output_existing, pl.adir.output_existing, pl.adir.scalar, pl.adir.scalar, pl.adir.input, pl.adir.input]},
        )

    @pl.function(type=pl.FunctionType.AIC, level=pl.Level.AIC, role=pl.Role.SubWorker)
    def qk_pv_aic(
        ffts_workspace_inline2289__ssa_v0: pl.Tensor[[256], pl.INT64, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 2048)],
        t_dim_inline2318__ssa_v0: pl.Scalar[pl.INDEX],
        q_flat_inline2306__ssa_v0: pl.Tensor[[t_heads_inline2333__ssa_v0, 512], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)],
        valid_block_mask_inline2310__ssa_v0: pl.Tensor[[t_dim_inline2318__ssa_v0, 16], pl.INT32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)],
        kv_transfer_inline2324__ssa_v0: pl.InOut[pl.Tensor[[9216, 512], pl.BF16, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 9437184)]],
        score_transfer_inline2293__ssa_v0: pl.InOut[pl.Tensor[[4608, 128], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 2359296)]],
        probability_transfer_inline2339__ssa_v0: pl.InOut[pl.Tensor[[4608, 128], pl.BF16, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 1179648)]],
        pv_transfer_inline2347__ssa_v0: pl.InOut[pl.Tensor[[4608, 512], pl.FP32, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 9437184)]],
        attn_sink_col_inline2332__ssa_v0: pl.Tensor[[64, 1], pl.FP32, pl.MemRef("mem_ddr_7", pl.const(0, pl.INT64), 256)],
        position_ids_t1__ssa_v0: pl.Tensor[[T_DYN, 1], pl.INT32, pl.MemRef("mem_ddr_8", pl.const(0, pl.INT64), 0)],
        window_swa_indices__ssa_v0: pl.Tensor[[T_DYN, 128], pl.INT32, pl.MemRef("mem_ddr_9", pl.const(0, pl.INT64), 0)],
        ori_kv_flat_inline2377__ssa_v1: pl.Tensor[[ori_block_num_inline2385__ssa_v0 * pl.const(32, pl.INDEX), 512], pl.BF16, pl.MemRef("mem_ddr_10", pl.const(0, pl.INT64), 0)],
        cmp_sparse_indices_inline2323__rv_v2: pl.Tensor[[t_dim_inline2318__ssa_v0, 512], pl.INT32, pl.MemRef("mem_ddr_11", pl.const(0, pl.INT64), 0)],
        cmp_block_table__ssa_v0: pl.Tensor[[B_DYN, COMPRESSED_TABLE_COLUMNS_DYN], pl.INT32, pl.MemRef("mem_ddr_12", pl.const(0, pl.INT64), 0)],
        cmp_kv_flat_inline2399__ssa_v0: pl.Tensor[[cmp_block_num_inline2362__ssa_v0 * pl.const(32, pl.INDEX), 512], pl.BF16, pl.MemRef("mem_ddr_13", pl.const(0, pl.INT64), 0)],
        sparse_bias_inline2369__rv_v2: pl.Tensor[[t_dim_inline2318__ssa_v0, 640], pl.FP32, pl.MemRef("mem_ddr_14", pl.const(0, pl.INT64), 0)],
        mi_transfer_inline2291__ssa_v0: pl.InOut[pl.Tensor[[4608, 1], pl.FP32, pl.MemRef("mem_ddr_15", pl.const(0, pl.INT64), 18432)]],
        li_transfer_inline2388__ssa_v0: pl.InOut[pl.Tensor[[4608, 1], pl.FP32, pl.MemRef("mem_ddr_16", pl.const(0, pl.INT64), 18432)]],
        attn_mi_inline2305__ssa_v0: pl.Out[pl.Tensor[[t_heads_inline2333__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_17", pl.const(0, pl.INT64), 0)]],
        attn_li_inline2396__ssa_v0: pl.Out[pl.Tensor[[t_heads_inline2333__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_18", pl.const(0, pl.INT64), 0)]],
        attn_oi_inline2303__ssa_v0: pl.Out[pl.Tensor[[t_heads_inline2333__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_19", pl.const(0, pl.INT64), 0)]],
    ):
        pl.func_attr({"mx_tensor_views_blocked": True, "split_aiv": True})
        mem_mat_20: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 65536)
        mem_mat_21: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 393216)
        mem_left_23: pl.Ptr = pl.tile.alloc(pl.Mem.Left, 16384)
        mem_right_24: pl.Ptr = pl.tile.alloc(pl.Mem.Right, 32768)
        mem_left_25: pl.Ptr = pl.tile.alloc(pl.Mem.Left, 16384)
        mem_right_26: pl.Ptr = pl.tile.alloc(pl.Mem.Right, 32768)
        mem_mat_30: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 16384)
        mem_acc_31: pl.Ptr = pl.tile.alloc(pl.Mem.Acc, 131072)
        qk_core_inline2314__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        pl.system.set_ffts(ffts_workspace_inline2289__ssa_v0)
        for qk_t_inline2286__idx_v0 in pl.range(qk_core_inline2314__ssa_v0, t_dim_inline2318__ssa_v0, 24):
            qk_q_inline2287__ssa_v0: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_mat_20, pl.const(0, pl.INT64), 65536), pl.Mem.Mat] = pl.tile.load(
                q_flat_inline2306__ssa_v0, [qk_t_inline2286__idx_v0 * 64, 0], [64, 512], [64, 512], target_memory=pl.Mem.Mat
            )
            qk_l1_inline2316__ssa_v0: pl.Tile[[384, 512], pl.BF16, pl.MemRef(mem_mat_21, pl.const(65536, pl.INT64), 393216), pl.Mem.Mat] = pl.tile.create(
                [384, 512], dtype=pl.BF16, target_memory=pl.Mem.Mat
            )
            for qk_tick_inline2301__idx_v0, (qk_l1_inline2316__iter_v1,) in pl.range(7, init_values=(qk_l1_inline2316__ssa_v0,)):
                if qk_tick_inline2301__idx_v0 < 5:
                    qk_sb_inline2298__ssa_v0: pl.Scalar[pl.INDEX] = qk_tick_inline2301__idx_v0
                    t__tile: pl.Scalar[pl.INT32] = pl.tensor.read(valid_block_mask_inline2310__ssa_v0, [qk_t_inline2286__idx_v0, qk_sb_inline2298__ssa_v0])
                    if 0 < pl.cast(t__tile, pl.INDEX):
                        qk_slot_inline2282__ssa_v0: pl.Scalar[pl.INDEX] = qk_core_inline2314__ssa_v0 * 3 + qk_sb_inline2298__ssa_v0 % 3
                        qk_kv_row_inline2395__ssa_v0: pl.Scalar[pl.INDEX] = qk_slot_inline2282__ssa_v0 * 128
                        qk_transfer_row_inline2284__ssa_v0: pl.Scalar[pl.INDEX] = qk_slot_inline2282__ssa_v0 * 64
                        pl.system.sync_wait(0, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIC)
                        qk_l1_row_inline2365__ssa_v0: pl.Scalar[pl.INDEX] = qk_sb_inline2298__ssa_v0 % 3 * 128
                        qk_l1_inline2316__ssa_v3: pl.Tile[[384, 512], pl.BF16, pl.MemRef(mem_mat_21, pl.const(65536, pl.INT64), 393216), pl.Mem.Mat] = pl.tile.gather_row(
                            qk_l1_inline2316__iter_v1, kv_transfer_inline2324__ssa_v0, [qk_l1_row_inline2365__ssa_v0, 0], [qk_kv_row_inline2395__ssa_v0, 0], [128, 512], transpose=False
                        )
                        qk_l1_t_inline2370__ssa_v0: pl.Tile[
                            [512, 384], pl.BF16, pl.MemRef(mem_mat_21, pl.const(65536, pl.INT64), 393216), pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)
                        ] = pl.tile.transpose_view(qk_l1_inline2316__ssa_v3)
                        qk_scores_inline2311__ssa_v0_l0_init: pl.Tile[[64, 128], pl.FP32, pl.MemRef(mem_acc_31, pl.const(0, pl.INT64), 32768), pl.Mem.Acc] = pl.tile.create(
                            [64, 128], dtype=pl.FP32, target_memory=pl.Mem.Acc
                        )
                        for qk_scores_inline2311__ssa_v0_l0_ko, (qk_scores_inline2311__ssa_v0_l0_c,) in pl.range(0, 512, 256, init_values=(qk_scores_inline2311__ssa_v0_l0_init,)):
                            qk_scores_inline2311__ssa_v0_l0_a: pl.Tile[
                                [64, 128], pl.BF16, pl.MemRef(mem_left_23, pl.const(0, pl.INT64), 16384), pl.Mem.Left, pl.TileView(blayout=pl.TileLayout.row_major)
                            ] = pl.tile.extract(qk_q_inline2287__ssa_v0, 0, qk_scores_inline2311__ssa_v0_l0_ko, [64, 128], target_memory=pl.Mem.Left)
                            qk_scores_inline2311__ssa_v0_l0_b: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_24, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                                qk_l1_t_inline2370__ssa_v0, qk_scores_inline2311__ssa_v0_l0_ko, qk_l1_row_inline2365__ssa_v0, [128, 128], target_memory=pl.Mem.Right
                            )
                            qk_scores_inline2311__ssa_v0_l0_a_1: pl.Tile[
                                [64, 128], pl.BF16, pl.MemRef(mem_left_25, pl.const(16384, pl.INT64), 16384), pl.Mem.Left, pl.TileView(blayout=pl.TileLayout.row_major)
                            ] = pl.tile.extract(qk_q_inline2287__ssa_v0, 0, qk_scores_inline2311__ssa_v0_l0_ko + 128, [64, 128], target_memory=pl.Mem.Left)
                            qk_scores_inline2311__ssa_v0_l0_b_1: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_26, pl.const(32768, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                                qk_l1_t_inline2370__ssa_v0, qk_scores_inline2311__ssa_v0_l0_ko + 128, qk_l1_row_inline2365__ssa_v0, [128, 128], target_memory=pl.Mem.Right
                            )
                            qk_scores_inline2311__ssa_v0_l0_c_acc: pl.Tile[[64, 128], pl.FP32, pl.MemRef(mem_acc_31, pl.const(0, pl.INT64), 32768), pl.Mem.Acc] = pl.tile.matmul_acc(
                                qk_scores_inline2311__ssa_v0_l0_c, qk_scores_inline2311__ssa_v0_l0_a, qk_scores_inline2311__ssa_v0_l0_b, qk_scores_inline2311__ssa_v0_l0_ko == 0
                            )
                            qk_scores_inline2311__ssa_v0_l0_c_acc_1: pl.Tile[[64, 128], pl.FP32, pl.MemRef(mem_acc_31, pl.const(0, pl.INT64), 32768), pl.Mem.Acc] = pl.tile.matmul_acc(
                                qk_scores_inline2311__ssa_v0_l0_c_acc, qk_scores_inline2311__ssa_v0_l0_a_1, qk_scores_inline2311__ssa_v0_l0_b_1, qk_scores_inline2311__ssa_v0_l0_ko == -128
                            )
                            qk_scores_inline2311__ssa_v0: pl.Tile[[64, 128], pl.FP32, pl.MemRef(mem_acc_31, pl.const(0, pl.INT64), 32768), pl.Mem.Acc] = pl.yield_(
                                qk_scores_inline2311__ssa_v0_l0_c_acc_1
                            )
                        score_transfer_inline2293__store: pl.Tensor[[4608, 128], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 2359296)] = pl.tile.store(
                            qk_scores_inline2311__ssa_v0, [qk_transfer_row_inline2284__ssa_v0, 0], score_transfer_inline2293__ssa_v0
                        )
                        pl.system.sync_set(1, pipe=pl.PipeType.FIX, ffts_mode=2, core_type=pl.KernelType.AIC)
                        qk_l1_inline2316__phi_v4: pl.Tile[[384, 512], pl.BF16, pl.MemRef(mem_mat_21, pl.const(65536, pl.INT64), 393216), pl.Mem.Mat] = pl.yield_(qk_l1_inline2316__ssa_v3)
                    else:
                        qk_l1_inline2316__phi_v4: pl.Tile[[384, 512], pl.BF16, pl.MemRef(mem_mat_21, pl.const(65536, pl.INT64), 393216), pl.Mem.Mat] = pl.yield_(qk_l1_inline2316__iter_v1)
                    qk_l1_inline2316__phi_v5: pl.Tile[[384, 512], pl.BF16, pl.MemRef(mem_mat_21, pl.const(65536, pl.INT64), 393216), pl.Mem.Mat] = pl.yield_(qk_l1_inline2316__phi_v4)
                else:
                    qk_l1_inline2316__phi_v5: pl.Tile[[384, 512], pl.BF16, pl.MemRef(mem_mat_21, pl.const(65536, pl.INT64), 393216), pl.Mem.Mat] = pl.yield_(qk_l1_inline2316__iter_v1)
                if 2 <= qk_tick_inline2301__idx_v0:
                    pv_sb_inline2374__ssa_v0: pl.Scalar[pl.INDEX] = qk_tick_inline2301__idx_v0 - 2
                    t__tile_1: pl.Scalar[pl.INT32] = pl.tensor.read(valid_block_mask_inline2310__ssa_v0, [qk_t_inline2286__idx_v0, pv_sb_inline2374__ssa_v0])
                    if 0 < pl.cast(t__tile_1, pl.INDEX):
                        pv_slot_inline2375__ssa_v0: pl.Scalar[pl.INDEX] = qk_core_inline2314__ssa_v0 * 3 + pv_sb_inline2374__ssa_v0 % 3
                        pv_transfer_row_inline2329__ssa_v0: pl.Scalar[pl.INDEX] = pv_slot_inline2375__ssa_v0 * 64
                        pl.system.sync_wait(2, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIC)
                        pv_probability_inline2376__ssa_v0: pl.Tile[[64, 128], pl.BF16, pl.MemRef(mem_mat_30, pl.const(458752, pl.INT64), 16384), pl.Mem.Mat] = pl.tile.load(
                            probability_transfer_inline2339__ssa_v0, [pv_transfer_row_inline2329__ssa_v0, 0], [64, 128], [64, 128], target_memory=pl.Mem.Mat
                        )
                        pv_l1_row_inline2380__ssa_v0: pl.Scalar[pl.INDEX] = pv_sb_inline2374__ssa_v0 % 3 * 128
                        pv_output_inline2290__ssa_v0_l0_init: pl.Tile[[64, 512], pl.FP32, pl.MemRef(mem_acc_31, pl.const(0, pl.INT64), 131072), pl.Mem.Acc] = pl.tile.create(
                            [64, 512], dtype=pl.FP32, target_memory=pl.Mem.Acc
                        )
                        for pv_output_inline2290__ssa_v0_l0_ko, (pv_output_inline2290__ssa_v0_l0_c,) in pl.range(0, 128, 64, init_values=(pv_output_inline2290__ssa_v0_l0_init,)):
                            pv_output_inline2290__ssa_v0_l0_a: pl.Tile[
                                [64, 32], pl.BF16, pl.MemRef(mem_left_23, pl.const(0, pl.INT64), 4096), pl.Mem.Left, pl.TileView(blayout=pl.TileLayout.row_major)
                            ] = pl.tile.extract(pv_probability_inline2376__ssa_v0, 0, pv_output_inline2290__ssa_v0_l0_ko, [64, 32], target_memory=pl.Mem.Left)
                            pv_output_inline2290__ssa_v0_l0_b: pl.Tile[[32, 512], pl.BF16, pl.MemRef(mem_right_24, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                                qk_l1_inline2316__phi_v5, pv_output_inline2290__ssa_v0_l0_ko + pv_l1_row_inline2380__ssa_v0, 0, [32, 512], target_memory=pl.Mem.Right
                            )
                            pv_output_inline2290__ssa_v0_l0_a_1: pl.Tile[
                                [64, 32], pl.BF16, pl.MemRef(mem_left_25, pl.const(16384, pl.INT64), 4096), pl.Mem.Left, pl.TileView(blayout=pl.TileLayout.row_major)
                            ] = pl.tile.extract(pv_probability_inline2376__ssa_v0, 0, pv_output_inline2290__ssa_v0_l0_ko + 32, [64, 32], target_memory=pl.Mem.Left)
                            pv_output_inline2290__ssa_v0_l0_b_1: pl.Tile[[32, 512], pl.BF16, pl.MemRef(mem_right_26, pl.const(32768, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                                qk_l1_inline2316__phi_v5, pv_output_inline2290__ssa_v0_l0_ko + pv_l1_row_inline2380__ssa_v0 + 32, 0, [32, 512], target_memory=pl.Mem.Right
                            )
                            pv_output_inline2290__ssa_v0_l0_c_acc: pl.Tile[[64, 512], pl.FP32, pl.MemRef(mem_acc_31, pl.const(0, pl.INT64), 131072), pl.Mem.Acc] = pl.tile.matmul_acc(
                                pv_output_inline2290__ssa_v0_l0_c, pv_output_inline2290__ssa_v0_l0_a, pv_output_inline2290__ssa_v0_l0_b, pv_output_inline2290__ssa_v0_l0_ko == 0
                            )
                            pv_output_inline2290__ssa_v0_l0_c_acc_1: pl.Tile[[64, 512], pl.FP32, pl.MemRef(mem_acc_31, pl.const(0, pl.INT64), 131072), pl.Mem.Acc] = pl.tile.matmul_acc(
                                pv_output_inline2290__ssa_v0_l0_c_acc, pv_output_inline2290__ssa_v0_l0_a_1, pv_output_inline2290__ssa_v0_l0_b_1, pv_output_inline2290__ssa_v0_l0_ko == -32
                            )
                            pv_output_inline2290__ssa_v0: pl.Tile[[64, 512], pl.FP32, pl.MemRef(mem_acc_31, pl.const(0, pl.INT64), 131072), pl.Mem.Acc] = pl.yield_(
                                pv_output_inline2290__ssa_v0_l0_c_acc_1
                            )
                        pv_transfer_inline2347__store: pl.Tensor[[4608, 512], pl.FP32, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 9437184)] = pl.tile.store(
                            pv_output_inline2290__ssa_v0, [pv_transfer_row_inline2329__ssa_v0, 0], pv_transfer_inline2347__ssa_v0
                        )
                        pl.system.sync_set(3, pipe=pl.PipeType.FIX, ffts_mode=2, core_type=pl.KernelType.AIC)
                qk_l1_inline2316__rv_v2: pl.Tile[[384, 512], pl.BF16, pl.MemRef(mem_mat_21, pl.const(65536, pl.INT64), 393216), pl.Mem.Mat] = pl.yield_(qk_l1_inline2316__phi_v5)
            pl.system.set_ffts(ffts_workspace_inline2289__ssa_v0)

    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def qk_pv_aiv(
        ffts_workspace_inline2289__ssa_v0: pl.Tensor[[256], pl.INT64, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 2048)],
        t_dim_inline2318__ssa_v0: pl.Scalar[pl.INDEX],
        q_flat_inline2306__ssa_v0: pl.Tensor[[t_heads_inline2333__ssa_v0, 512], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)],
        valid_block_mask_inline2310__ssa_v0: pl.Tensor[[t_dim_inline2318__ssa_v0, 16], pl.INT32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)],
        kv_transfer_inline2324__ssa_v0: pl.InOut[pl.Tensor[[9216, 512], pl.BF16, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 9437184)]],
        score_transfer_inline2293__ssa_v0: pl.InOut[pl.Tensor[[4608, 128], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 2359296)]],
        probability_transfer_inline2339__ssa_v0: pl.InOut[pl.Tensor[[4608, 128], pl.BF16, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 1179648)]],
        pv_transfer_inline2347__ssa_v0: pl.InOut[pl.Tensor[[4608, 512], pl.FP32, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 9437184)]],
        attn_sink_col_inline2332__ssa_v0: pl.Tensor[[64, 1], pl.FP32, pl.MemRef("mem_ddr_7", pl.const(0, pl.INT64), 256)],
        position_ids_t1__ssa_v0: pl.Tensor[[T_DYN, 1], pl.INT32, pl.MemRef("mem_ddr_8", pl.const(0, pl.INT64), 0)],
        window_swa_indices__ssa_v0: pl.Tensor[[T_DYN, 128], pl.INT32, pl.MemRef("mem_ddr_9", pl.const(0, pl.INT64), 0)],
        ori_kv_flat_inline2377__ssa_v1: pl.Tensor[[ori_block_num_inline2385__ssa_v0 * pl.const(32, pl.INDEX), 512], pl.BF16, pl.MemRef("mem_ddr_10", pl.const(0, pl.INT64), 0)],
        cmp_sparse_indices_inline2323__rv_v2: pl.Tensor[[t_dim_inline2318__ssa_v0, 512], pl.INT32, pl.MemRef("mem_ddr_11", pl.const(0, pl.INT64), 0)],
        cmp_block_table__ssa_v0: pl.Tensor[[B_DYN, COMPRESSED_TABLE_COLUMNS_DYN], pl.INT32, pl.MemRef("mem_ddr_12", pl.const(0, pl.INT64), 0)],
        cmp_kv_flat_inline2399__ssa_v0: pl.Tensor[[cmp_block_num_inline2362__ssa_v0 * pl.const(32, pl.INDEX), 512], pl.BF16, pl.MemRef("mem_ddr_13", pl.const(0, pl.INT64), 0)],
        sparse_bias_inline2369__rv_v2: pl.Tensor[[t_dim_inline2318__ssa_v0, 640], pl.FP32, pl.MemRef("mem_ddr_14", pl.const(0, pl.INT64), 0)],
        mi_transfer_inline2291__ssa_v0: pl.InOut[pl.Tensor[[4608, 1], pl.FP32, pl.MemRef("mem_ddr_15", pl.const(0, pl.INT64), 18432)]],
        li_transfer_inline2388__ssa_v0: pl.InOut[pl.Tensor[[4608, 1], pl.FP32, pl.MemRef("mem_ddr_16", pl.const(0, pl.INT64), 18432)]],
        attn_mi_inline2305__ssa_v0: pl.Out[pl.Tensor[[t_heads_inline2333__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_17", pl.const(0, pl.INT64), 0)]],
        attn_li_inline2396__ssa_v0: pl.Out[pl.Tensor[[t_heads_inline2333__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_18", pl.const(0, pl.INT64), 0)]],
        attn_oi_inline2303__ssa_v0: pl.Out[pl.Tensor[[t_heads_inline2333__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_19", pl.const(0, pl.INT64), 0)]],
    ) -> tuple[
        pl.Tensor[[9216, 512], pl.BF16],
        pl.Tensor[[4608, 128], pl.FP32],
        pl.Tensor[[4608, 128], pl.BF16],
        pl.Tensor[[4608, 512], pl.FP32],
        pl.Tensor[[4608, 1], pl.FP32],
        pl.Tensor[[4608, 1], pl.FP32],
        pl.Tensor[[t_heads_inline2333__ssa_v0, 1], pl.FP32],
        pl.Tensor[[t_heads_inline2333__ssa_v0, 1], pl.FP32],
        pl.Tensor[[t_heads_inline2333__ssa_v0, 512], pl.FP32],
    ]:
        pl.func_attr({"mx_tensor_views_blocked": True, "split_aiv": True, "dual_aiv_dispatch": True})
        mem_vec_20: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
        mem_vec_21: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 128)
        mem_vec_22: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 128)
        mem_vec_23: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32768)
        mem_vec_24: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32768)
        mem_vec_25: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 65536)
        mem_vec_47: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 128)
        mem_vec_52: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 128)
        mem_vec_55: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 128)
        mem_vec_58: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 128)
        mem_vec_60: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32768)
        qk_core_inline2314__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        pl.system.set_ffts(ffts_workspace_inline2289__ssa_v0)
        for qk_t_inline2286__idx_v0 in pl.range(qk_core_inline2314__ssa_v0, t_dim_inline2318__ssa_v0, 24):
            qk_b_inline2285__ssa_v0: pl.Scalar[pl.INDEX] = qk_t_inline2286__idx_v0 // 6
            qk_aiv_inline2378__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_subblock_idx()
            pl.system.set_ffts(ffts_workspace_inline2289__ssa_v0)
            qk_lane_head_inline2383__ssa_v0: pl.Scalar[pl.INDEX] = qk_aiv_inline2378__ssa_v0 * 32
            qk_lane_kv_inline2386__ssa_v0: pl.Scalar[pl.INDEX] = qk_aiv_inline2378__ssa_v0 * 64
            qk_reduce_tmp_inline2334__ssa_v0: pl.Tile[[32, 128], pl.FP32, pl.MemRef(mem_vec_20, pl.const(0, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.create(
                [32, 128], dtype=pl.FP32, target_memory=pl.Mem.Vec
            )
            running_m_inline2346__ssa_v0: pl.Tile[[32, 1], pl.FP32, pl.MemRef(mem_vec_21, pl.const(16384, pl.INT64), 128), pl.Mem.Vec] = pl.tile.load(
                attn_sink_col_inline2332__ssa_v0, [qk_lane_head_inline2383__ssa_v0, 0], [32, 1], [32, 1], target_memory=pl.Mem.Vec
            )
            running_l_inline2389__rm_a0_tmp_v0: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_21, pl.const(16384, pl.INT64), 128), pl.Mem.Vec] = pl.tile.reshape(running_m_inline2346__ssa_v0, [1, 32])
            running_l_inline2389__row_major_tmp_v1: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_22, pl.const(16512, pl.INT64), 128), pl.Mem.Vec] = pl.tile.muls(running_l_inline2389__rm_a0_tmp_v0, 0.0)
            running_l_inline2389__ssa_v0: pl.Tile[[32, 1], pl.FP32, pl.MemRef(mem_vec_22, pl.const(16512, pl.INT64), 128), pl.Mem.Vec] = pl.tile.reshape(
                running_l_inline2389__row_major_tmp_v1, [32, 1]
            )
            running_left_inline2391__ssa_v0: pl.Tile[[32, 256], pl.FP32, pl.MemRef(mem_vec_23, pl.const(16640, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.full([32, 256], dtype=pl.FP32, value=0.0)
            running_right_inline2294__ssa_v0: pl.Tile[[32, 256], pl.FP32, pl.MemRef(mem_vec_24, pl.const(49408, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.full([32, 256], dtype=pl.FP32, value=0.0)
            for qk_tick_inline2392__idx_v0, (m_iter, l_iter, left_iter, right_iter) in pl.range(
                8, init_values=(running_m_inline2346__ssa_v0, running_l_inline2389__ssa_v0, running_left_inline2391__ssa_v0, running_right_inline2294__ssa_v0)
            ):
                if qk_tick_inline2392__idx_v0 < 5:
                    qk_sb_inline2298__ssa_v1: pl.Scalar[pl.INDEX] = qk_tick_inline2392__idx_v0
                    t__tile: pl.Scalar[pl.INT32] = pl.tensor.read(valid_block_mask_inline2310__ssa_v0, [qk_t_inline2286__idx_v0, qk_sb_inline2298__ssa_v1])
                    if 0 < pl.cast(t__tile, pl.INDEX):
                        qk_slot_inline2282__ssa_v1: pl.Scalar[pl.INDEX] = qk_core_inline2314__ssa_v0 * 3 + qk_sb_inline2298__ssa_v1 % 3
                        qk_kv_row_inline2395__ssa_v1: pl.Scalar[pl.INDEX] = qk_slot_inline2282__ssa_v1 * 128
                        qk_s0_inline2398__ssa_v0: pl.Scalar[pl.INDEX] = qk_sb_inline2298__ssa_v1 * 128
                        qk_kv_half_inline2400__ssa_v0: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.tile.full(
                            [64, 512], dtype=pl.BF16, value=0.0
                        )
                        if qk_s0_inline2398__ssa_v0 < 128:
                            t__tile_1: pl.Scalar[pl.INT32] = pl.tensor.read(position_ids_t1__ssa_v0, [qk_t_inline2286__idx_v0, 0])
                            qk_pos_inline2401__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(t__tile_1, pl.INDEX)
                            qk_win_len_inline2355__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(qk_pos_inline2401__ssa_v0 + 1, 128)
                            qk_win_start_inline2404__ssa_v0: pl.Scalar[pl.INDEX] = qk_pos_inline2401__ssa_v0 - qk_win_len_inline2355__ssa_v0 + 1
                            qk_head_inline2379__ssa_v0: pl.Scalar[pl.INDEX] = (qk_win_start_inline2404__ssa_v0 + qk_s0_inline2398__ssa_v0) % 32
                            qk_rows_inline2394__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(pl.max(qk_win_len_inline2355__ssa_v0 - qk_s0_inline2398__ssa_v0 - qk_lane_kv_inline2386__ssa_v0, 0), 64)
                            qk_lo_inline2350__ssa_v0: pl.Scalar[pl.INDEX] = 0
                            qk_hi_inline2366__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(32 - qk_head_inline2379__ssa_v0 - qk_lane_kv_inline2386__ssa_v0, qk_rows_inline2394__ssa_v0)
                            if qk_lo_inline2350__ssa_v0 < qk_hi_inline2366__ssa_v0:
                                qk_raw_row_inline2340__tile: pl.Scalar[pl.INT32] = pl.tensor.read(
                                    window_swa_indices__ssa_v0, [qk_t_inline2286__idx_v0, qk_s0_inline2398__ssa_v0 + qk_lane_kv_inline2386__ssa_v0 + qk_lo_inline2350__ssa_v0]
                                )
                                if 0 <= pl.cast(qk_raw_row_inline2340__tile, pl.INDEX):
                                    qk_kv_half_inline2400__ssa_v1: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.tile.gather_row(
                                        qk_kv_half_inline2400__ssa_v0,
                                        ori_kv_flat_inline2377__ssa_v1,
                                        [qk_lo_inline2350__ssa_v0, 0],
                                        [qk_raw_row_inline2340__tile, 0],
                                        [64, 512],
                                        valid_shape=[qk_hi_inline2366__ssa_v0 - qk_lo_inline2350__ssa_v0, 512],
                                        transpose=False,
                                    )
                                    qk_kv_half_inline2400__phi_v2: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(
                                        qk_kv_half_inline2400__ssa_v1
                                    )
                                else:
                                    qk_kv_half_inline2400__phi_v2: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(
                                        qk_kv_half_inline2400__ssa_v0
                                    )
                                qk_kv_half_inline2400__phi_v3: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(
                                    qk_kv_half_inline2400__phi_v2
                                )
                            else:
                                qk_kv_half_inline2400__phi_v3: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(
                                    qk_kv_half_inline2400__ssa_v0
                                )
                            qk_lo_inline2350__ssa_v1: pl.Scalar[pl.INDEX] = pl.max(32 - qk_head_inline2379__ssa_v0 - qk_lane_kv_inline2386__ssa_v0, 0)
                            qk_hi_inline2366__ssa_v1: pl.Scalar[pl.INDEX] = pl.min(64 - qk_head_inline2379__ssa_v0 - qk_lane_kv_inline2386__ssa_v0, qk_rows_inline2394__ssa_v0)
                            if qk_lo_inline2350__ssa_v1 < qk_hi_inline2366__ssa_v1:
                                qk_raw_row_inline2340__tile_1: pl.Scalar[pl.INT32] = pl.tensor.read(
                                    window_swa_indices__ssa_v0, [qk_t_inline2286__idx_v0, qk_s0_inline2398__ssa_v0 + qk_lane_kv_inline2386__ssa_v0 + qk_lo_inline2350__ssa_v1]
                                )
                                if 0 <= pl.cast(qk_raw_row_inline2340__tile_1, pl.INDEX):
                                    qk_kv_half_inline2400__ssa_v4: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.tile.gather_row(
                                        qk_kv_half_inline2400__phi_v3,
                                        ori_kv_flat_inline2377__ssa_v1,
                                        [qk_lo_inline2350__ssa_v1, 0],
                                        [qk_raw_row_inline2340__tile_1, 0],
                                        [64, 512],
                                        valid_shape=[qk_hi_inline2366__ssa_v1 - qk_lo_inline2350__ssa_v1, 512],
                                        transpose=False,
                                    )
                                    qk_kv_half_inline2400__phi_v5: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(
                                        qk_kv_half_inline2400__ssa_v4
                                    )
                                else:
                                    qk_kv_half_inline2400__phi_v5: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(
                                        qk_kv_half_inline2400__phi_v3
                                    )
                                qk_kv_half_inline2400__phi_v6: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(
                                    qk_kv_half_inline2400__phi_v5
                                )
                            else:
                                qk_kv_half_inline2400__phi_v6: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(
                                    qk_kv_half_inline2400__phi_v3
                                )
                            qk_lo_inline2350__ssa_v2: pl.Scalar[pl.INDEX] = pl.max(64 - qk_head_inline2379__ssa_v0 - qk_lane_kv_inline2386__ssa_v0, 0)
                            qk_hi_inline2366__ssa_v2: pl.Scalar[pl.INDEX] = pl.min(96 - qk_head_inline2379__ssa_v0 - qk_lane_kv_inline2386__ssa_v0, qk_rows_inline2394__ssa_v0)
                            if qk_lo_inline2350__ssa_v2 < qk_hi_inline2366__ssa_v2:
                                qk_raw_row_inline2340__tile_2: pl.Scalar[pl.INT32] = pl.tensor.read(
                                    window_swa_indices__ssa_v0, [qk_t_inline2286__idx_v0, qk_s0_inline2398__ssa_v0 + qk_lane_kv_inline2386__ssa_v0 + qk_lo_inline2350__ssa_v2]
                                )
                                if 0 <= pl.cast(qk_raw_row_inline2340__tile_2, pl.INDEX):
                                    qk_kv_half_inline2400__ssa_v7: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.tile.gather_row(
                                        qk_kv_half_inline2400__phi_v6,
                                        ori_kv_flat_inline2377__ssa_v1,
                                        [qk_lo_inline2350__ssa_v2, 0],
                                        [qk_raw_row_inline2340__tile_2, 0],
                                        [64, 512],
                                        valid_shape=[qk_hi_inline2366__ssa_v2 - qk_lo_inline2350__ssa_v2, 512],
                                        transpose=False,
                                    )
                                    qk_kv_half_inline2400__phi_v8: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(
                                        qk_kv_half_inline2400__ssa_v7
                                    )
                                else:
                                    qk_kv_half_inline2400__phi_v8: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(
                                        qk_kv_half_inline2400__phi_v6
                                    )
                                qk_kv_half_inline2400__phi_v9: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(
                                    qk_kv_half_inline2400__phi_v8
                                )
                            else:
                                qk_kv_half_inline2400__phi_v9: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(
                                    qk_kv_half_inline2400__phi_v6
                                )
                            qk_lo_inline2350__ssa_v3: pl.Scalar[pl.INDEX] = pl.max(96 - qk_head_inline2379__ssa_v0 - qk_lane_kv_inline2386__ssa_v0, 0)
                            qk_hi_inline2366__ssa_v3: pl.Scalar[pl.INDEX] = pl.min(128 - qk_head_inline2379__ssa_v0 - qk_lane_kv_inline2386__ssa_v0, qk_rows_inline2394__ssa_v0)
                            if qk_lo_inline2350__ssa_v3 < qk_hi_inline2366__ssa_v3:
                                qk_raw_row_inline2340__tile_3: pl.Scalar[pl.INT32] = pl.tensor.read(
                                    window_swa_indices__ssa_v0, [qk_t_inline2286__idx_v0, qk_s0_inline2398__ssa_v0 + qk_lane_kv_inline2386__ssa_v0 + qk_lo_inline2350__ssa_v3]
                                )
                                if 0 <= pl.cast(qk_raw_row_inline2340__tile_3, pl.INDEX):
                                    qk_kv_half_inline2400__ssa_v10: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.tile.gather_row(
                                        qk_kv_half_inline2400__phi_v9,
                                        ori_kv_flat_inline2377__ssa_v1,
                                        [qk_lo_inline2350__ssa_v3, 0],
                                        [qk_raw_row_inline2340__tile_3, 0],
                                        [64, 512],
                                        valid_shape=[qk_hi_inline2366__ssa_v3 - qk_lo_inline2350__ssa_v3, 512],
                                        transpose=False,
                                    )
                                    qk_kv_half_inline2400__phi_v11: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(
                                        qk_kv_half_inline2400__ssa_v10
                                    )
                                else:
                                    qk_kv_half_inline2400__phi_v11: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(
                                        qk_kv_half_inline2400__phi_v9
                                    )
                                qk_kv_half_inline2400__phi_v12: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(
                                    qk_kv_half_inline2400__phi_v11
                                )
                            else:
                                qk_kv_half_inline2400__phi_v12: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(
                                    qk_kv_half_inline2400__phi_v9
                                )
                            qk_lo_inline2350__ssa_v4: pl.Scalar[pl.INDEX] = pl.max(128 - qk_head_inline2379__ssa_v0 - qk_lane_kv_inline2386__ssa_v0, 0)
                            qk_hi_inline2366__ssa_v4: pl.Scalar[pl.INDEX] = pl.min(160 - qk_head_inline2379__ssa_v0 - qk_lane_kv_inline2386__ssa_v0, qk_rows_inline2394__ssa_v0)
                            if qk_lo_inline2350__ssa_v4 < qk_hi_inline2366__ssa_v4:
                                qk_raw_row_inline2340__tile_4: pl.Scalar[pl.INT32] = pl.tensor.read(
                                    window_swa_indices__ssa_v0, [qk_t_inline2286__idx_v0, qk_s0_inline2398__ssa_v0 + qk_lane_kv_inline2386__ssa_v0 + qk_lo_inline2350__ssa_v4]
                                )
                                if 0 <= pl.cast(qk_raw_row_inline2340__tile_4, pl.INDEX):
                                    qk_kv_half_inline2400__ssa_v13: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.tile.gather_row(
                                        qk_kv_half_inline2400__phi_v12,
                                        ori_kv_flat_inline2377__ssa_v1,
                                        [qk_lo_inline2350__ssa_v4, 0],
                                        [qk_raw_row_inline2340__tile_4, 0],
                                        [64, 512],
                                        valid_shape=[qk_hi_inline2366__ssa_v4 - qk_lo_inline2350__ssa_v4, 512],
                                        transpose=False,
                                    )
                                    qk_kv_half_inline2400__phi_v14: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(
                                        qk_kv_half_inline2400__ssa_v13
                                    )
                                else:
                                    qk_kv_half_inline2400__phi_v14: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(
                                        qk_kv_half_inline2400__phi_v12
                                    )
                                qk_kv_half_inline2400__phi_v15: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(
                                    qk_kv_half_inline2400__phi_v14
                                )
                            else:
                                qk_kv_half_inline2400__phi_v15: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(
                                    qk_kv_half_inline2400__phi_v12
                                )
                            qk_kv_half_inline2400__phi_v21: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(qk_kv_half_inline2400__phi_v15)
                        else:
                            qk_kv_half_inline2400__ssa_v0: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.tile.full(
                                [64, 512], dtype=pl.BF16, value=0.0
                            )
                            for qk_row_inline2387__idx_v0, (qk_kv_half_inline2400__iter_v16,) in pl.range(64, init_values=(qk_kv_half_inline2400__ssa_v0,)):
                                qk_cmp_k_inline2406__ssa_v0: pl.Scalar[pl.INDEX] = qk_s0_inline2398__ssa_v0 + qk_lane_kv_inline2386__ssa_v0 + qk_row_inline2387__idx_v0 - 128
                                if qk_cmp_k_inline2406__ssa_v0 < 512:
                                    qk_ridx_inline2368__tile: pl.Scalar[pl.INT32] = pl.tensor.read(cmp_sparse_indices_inline2323__rv_v2, [qk_t_inline2286__idx_v0, qk_cmp_k_inline2406__ssa_v0])
                                    if 0 <= pl.cast(qk_ridx_inline2368__tile, pl.INDEX):
                                        t__tile_2: pl.Scalar[pl.INT32] = pl.tensor.read(cmp_block_table__ssa_v0, [qk_b_inline2285__ssa_v0, pl.cast(qk_ridx_inline2368__tile, pl.INDEX) // 32])
                                        qk_page_inline2352__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(t__tile_2, pl.INDEX)
                                        qk_src_inline2407__ssa_v0: pl.Scalar[pl.INDEX] = qk_page_inline2352__ssa_v0 * 32 + pl.cast(qk_ridx_inline2368__tile, pl.INDEX) % 32
                                        qk_kv_half_inline2400__ssa_v18: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.tile.gather_row(
                                            qk_kv_half_inline2400__iter_v16, cmp_kv_flat_inline2399__ssa_v0, [qk_row_inline2387__idx_v0, 0], [qk_src_inline2407__ssa_v0, 0], [1, 512], transpose=False
                                        )
                                        qk_kv_half_inline2400__phi_v19: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(
                                            qk_kv_half_inline2400__ssa_v18
                                        )
                                    else:
                                        qk_kv_half_inline2400__phi_v19: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(
                                            qk_kv_half_inline2400__iter_v16
                                        )
                                    qk_kv_half_inline2400__phi_v20: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(
                                        qk_kv_half_inline2400__phi_v19
                                    )
                                else:
                                    qk_kv_half_inline2400__phi_v20: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(
                                        qk_kv_half_inline2400__iter_v16
                                    )
                                qk_kv_half_inline2400__rv_v17: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(
                                    qk_kv_half_inline2400__phi_v20
                                )
                            qk_kv_half_inline2400__phi_v21: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(qk_kv_half_inline2400__rv_v17)
                        kv_transfer_inline2324__store: pl.Tensor[[9216, 512], pl.BF16, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 9437184)] = pl.tile.store(
                            qk_kv_half_inline2400__phi_v21, [qk_kv_row_inline2395__ssa_v1 + qk_lane_kv_inline2386__ssa_v0, 0], kv_transfer_inline2324__ssa_v0
                        )
                if 0 < qk_tick_inline2392__idx_v0 and qk_tick_inline2392__idx_v0 <= 5:
                    softmax_sb_inline2315__ssa_v0: pl.Scalar[pl.INDEX] = qk_tick_inline2392__idx_v0 - 1
                    t__tile_3: pl.Scalar[pl.INT32] = pl.tensor.read(valid_block_mask_inline2310__ssa_v0, [qk_t_inline2286__idx_v0, softmax_sb_inline2315__ssa_v0])
                    if 0 < pl.cast(t__tile_3, pl.INDEX):
                        qk_slot_inline2282__ssa_v2: pl.Scalar[pl.INDEX] = qk_core_inline2314__ssa_v0 * 3 + softmax_sb_inline2315__ssa_v0 % 3
                        qk_transfer_row_inline2284__ssa_v2: pl.Scalar[pl.INDEX] = qk_slot_inline2282__ssa_v2 * 64
                        qk_s0_v1_inline2331__ssa_v0: pl.Scalar[pl.INDEX] = softmax_sb_inline2315__ssa_v0 * 128
                        pl.system.sync_wait(1, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIV)
                        qk_scores_half_inline2345__ssa_v0: pl.Tile[[32, 128], pl.FP32, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.load(
                            score_transfer_inline2293__ssa_v0, [qk_transfer_row_inline2284__ssa_v2 + qk_lane_head_inline2383__ssa_v0, 0], [32, 128], [32, 128], target_memory=pl.Mem.Vec
                        )
                        qk_bias_inline2336__ssa_v0: pl.Tile[[1, 128], pl.FP32, pl.MemRef(mem_vec_60, pl.const(148224, pl.INT64), 512), pl.Mem.Vec] = pl.tile.load(
                            sparse_bias_inline2369__rv_v2, [qk_t_inline2286__idx_v0, qk_s0_v1_inline2331__ssa_v0], [1, 128], [1, 128], target_memory=pl.Mem.Vec
                        )
                        qk_scaled_inline2402__ssa_v0: pl.Tile[[32, 128], pl.FP32, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.muls(
                            qk_scores_half_inline2345__ssa_v0, 0.044194173824159223
                        )
                        qk_masked_inline2382__ssa_v0: pl.Tile[[32, 128], pl.FP32, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.col_expand_add(
                            qk_scaled_inline2402__ssa_v0, qk_bias_inline2336__ssa_v0
                        )
                        qk_mi_inline2357__ssa_v0: pl.Tile[[32, 1], pl.FP32, pl.MemRef(mem_vec_60, pl.const(148224, pl.INT64), 128), pl.Mem.Vec] = pl.tile.row_max(
                            qk_masked_inline2382__ssa_v0, qk_reduce_tmp_inline2334__ssa_v0
                        )
                        t__tmp_v256: pl.Tile[[32, 128], pl.FP32, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.row_expand_sub(
                            qk_masked_inline2382__ssa_v0, qk_mi_inline2357__ssa_v0
                        )
                        qk_exp_inline2358__ssa_v0: pl.Tile[[32, 128], pl.FP32, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.exp(t__tmp_v256)
                        qk_li_inline2408__ssa_v0: pl.Tile[[32, 1], pl.FP32, pl.MemRef(mem_vec_47, pl.const(147712, pl.INT64), 128), pl.Mem.Vec] = pl.tile.row_sum(
                            qk_exp_inline2358__ssa_v0, qk_reduce_tmp_inline2334__ssa_v0
                        )
                        qk_probability_inline2321__ssa_v0: pl.Tile[[32, 128], pl.BF16, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.cast(
                            qk_exp_inline2358__ssa_v0, target_type=pl.BF16, mode="rint"
                        )
                        probability_transfer_inline2339__store: pl.Tensor[[4608, 128], pl.BF16, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 1179648)] = pl.tile.store(
                            qk_probability_inline2321__ssa_v0, [qk_transfer_row_inline2284__ssa_v2 + qk_lane_head_inline2383__ssa_v0, 0], probability_transfer_inline2339__ssa_v0
                        )
                        mi_transfer_inline2291__store: pl.Tensor[[4608, 1], pl.FP32, pl.MemRef("mem_ddr_15", pl.const(0, pl.INT64), 18432)] = pl.tile.store(
                            qk_mi_inline2357__ssa_v0, [qk_transfer_row_inline2284__ssa_v2 + qk_lane_head_inline2383__ssa_v0, 0], mi_transfer_inline2291__ssa_v0
                        )
                        li_transfer_inline2388__store: pl.Tensor[[4608, 1], pl.FP32, pl.MemRef("mem_ddr_16", pl.const(0, pl.INT64), 18432)] = pl.tile.store(
                            qk_li_inline2408__ssa_v0, [qk_transfer_row_inline2284__ssa_v2 + qk_lane_head_inline2383__ssa_v0, 0], li_transfer_inline2388__ssa_v0
                        )
                        pl.system.sync_set(2, pipe=pl.PipeType.MTE3, ffts_mode=2, core_type=pl.KernelType.AIV)
                if qk_tick_inline2392__idx_v0 < 5:
                    t__tile_4: pl.Scalar[pl.INT32] = pl.tensor.read(valid_block_mask_inline2310__ssa_v0, [qk_t_inline2286__idx_v0, qk_tick_inline2392__idx_v0])
                    if 0 < pl.cast(t__tile_4, pl.INDEX):
                        pl.system.sync_set(0, pipe=pl.PipeType.MTE3, ffts_mode=2, core_type=pl.KernelType.AIV)
                if 3 <= qk_tick_inline2392__idx_v0:
                    pv_sb_inline2374__ssa_v1: pl.Scalar[pl.INDEX] = qk_tick_inline2392__idx_v0 - 2 - 1
                    t__tile_5: pl.Scalar[pl.INT32] = pl.tensor.read(valid_block_mask_inline2310__ssa_v0, [qk_t_inline2286__idx_v0, pv_sb_inline2374__ssa_v1])
                    if 0 < pl.cast(t__tile_5, pl.INDEX):
                        pv_slot_inline2375__ssa_v1: pl.Scalar[pl.INDEX] = qk_core_inline2314__ssa_v0 * 3 + pv_sb_inline2374__ssa_v1 % 3
                        pv_transfer_row_inline2329__ssa_v1: pl.Scalar[pl.INDEX] = pv_slot_inline2375__ssa_v1 * 64
                        pl.system.sync_wait(3, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIV)
                        pv_m_inline2413__ssa_v0: pl.Tile[[32, 1], pl.FP32, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 128), pl.Mem.Vec] = pl.tile.load(
                            mi_transfer_inline2291__ssa_v0, [pv_transfer_row_inline2329__ssa_v1 + qk_lane_head_inline2383__ssa_v0, 0], [32, 1], [32, 1], target_memory=pl.Mem.Vec
                        )
                        pv_l_inline2414__ssa_v0: pl.Tile[[32, 1], pl.FP32, pl.MemRef(mem_vec_60, pl.const(148224, pl.INT64), 128), pl.Mem.Vec] = pl.tile.load(
                            li_transfer_inline2388__ssa_v0, [pv_transfer_row_inline2329__ssa_v1 + qk_lane_head_inline2383__ssa_v0, 0], [32, 1], [32, 1], target_memory=pl.Mem.Vec
                        )
                        next_m_inline2417__rm_a0_tmp_v2: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_21, pl.const(16384, pl.INT64), 128), pl.Mem.Vec] = pl.tile.reshape(m_iter, [1, 32])
                        next_m_inline2417__rm_a1_tmp_v3: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 128), pl.Mem.Vec] = pl.tile.reshape(
                            pv_m_inline2413__ssa_v0, [1, 32]
                        )
                        next_m_inline2417__row_major_tmp_v4: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_47, pl.const(147712, pl.INT64), 128), pl.Mem.Vec] = pl.tile.maximum(
                            next_m_inline2417__rm_a0_tmp_v2, next_m_inline2417__rm_a1_tmp_v3
                        )
                        next_m_inline2417__ssa_v0: pl.Tile[[32, 1], pl.FP32, pl.MemRef(mem_vec_47, pl.const(147712, pl.INT64), 128), pl.Mem.Vec] = pl.tile.reshape(
                            next_m_inline2417__row_major_tmp_v4, [32, 1]
                        )
                        t__rm_a0_tmp_v5: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_21, pl.const(16384, pl.INT64), 128), pl.Mem.Vec] = pl.tile.reshape(m_iter, [1, 32])
                        t__rm_a1_tmp_v6: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_47, pl.const(147712, pl.INT64), 128), pl.Mem.Vec] = pl.tile.reshape(next_m_inline2417__ssa_v0, [1, 32])
                        t__row_major_tmp_v7: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_52, pl.const(147840, pl.INT64), 128), pl.Mem.Vec] = pl.tile.sub(t__rm_a0_tmp_v5, t__rm_a1_tmp_v6)
                        t__tmp_v259: pl.Tile[[32, 1], pl.FP32, pl.MemRef(mem_vec_52, pl.const(147840, pl.INT64), 128), pl.Mem.Vec] = pl.tile.reshape(t__row_major_tmp_v7, [32, 1])
                        alpha_inline2363__rm_a0_tmp_v8: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_52, pl.const(147840, pl.INT64), 128), pl.Mem.Vec] = pl.tile.reshape(t__tmp_v259, [1, 32])
                        alpha_inline2363__row_major_tmp_v9: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_52, pl.const(147840, pl.INT64), 128), pl.Mem.Vec] = pl.tile.exp(alpha_inline2363__rm_a0_tmp_v8)
                        alpha_inline2363__ssa_v0: pl.Tile[[32, 1], pl.FP32, pl.MemRef(mem_vec_52, pl.const(147840, pl.INT64), 128), pl.Mem.Vec] = pl.tile.reshape(
                            alpha_inline2363__row_major_tmp_v9, [32, 1]
                        )
                        t__rm_a0_tmp_v10: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 128), pl.Mem.Vec] = pl.tile.reshape(pv_m_inline2413__ssa_v0, [1, 32])
                        t__rm_a1_tmp_v11: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_47, pl.const(147712, pl.INT64), 128), pl.Mem.Vec] = pl.tile.reshape(next_m_inline2417__ssa_v0, [1, 32])
                        t__row_major_tmp_v12: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 128), pl.Mem.Vec] = pl.tile.sub(t__rm_a0_tmp_v10, t__rm_a1_tmp_v11)
                        t__tmp_v260: pl.Tile[[32, 1], pl.FP32, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 128), pl.Mem.Vec] = pl.tile.reshape(t__row_major_tmp_v12, [32, 1])
                        beta_inline2312__rm_a0_tmp_v13: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 128), pl.Mem.Vec] = pl.tile.reshape(t__tmp_v260, [1, 32])
                        beta_inline2312__row_major_tmp_v14: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_55, pl.const(147968, pl.INT64), 128), pl.Mem.Vec] = pl.tile.exp(beta_inline2312__rm_a0_tmp_v13)
                        beta_inline2312__ssa_v0: pl.Tile[[32, 1], pl.FP32, pl.MemRef(mem_vec_55, pl.const(147968, pl.INT64), 128), pl.Mem.Vec] = pl.tile.reshape(
                            beta_inline2312__row_major_tmp_v14, [32, 1]
                        )
                        t__rm_a0_tmp_v15: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_52, pl.const(147840, pl.INT64), 128), pl.Mem.Vec] = pl.tile.reshape(alpha_inline2363__ssa_v0, [1, 32])
                        t__rm_a1_tmp_v16: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_22, pl.const(16512, pl.INT64), 128), pl.Mem.Vec] = pl.tile.reshape(l_iter, [1, 32])
                        t__row_major_tmp_v17: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 128), pl.Mem.Vec] = pl.tile.mul(t__rm_a0_tmp_v15, t__rm_a1_tmp_v16)
                        t__tmp_v261: pl.Tile[[32, 1], pl.FP32, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 128), pl.Mem.Vec] = pl.tile.reshape(t__row_major_tmp_v17, [32, 1])
                        t__rm_a0_tmp_v18: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_55, pl.const(147968, pl.INT64), 128), pl.Mem.Vec] = pl.tile.reshape(beta_inline2312__ssa_v0, [1, 32])
                        t__rm_a1_tmp_v19: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_60, pl.const(148224, pl.INT64), 128), pl.Mem.Vec] = pl.tile.reshape(pv_l_inline2414__ssa_v0, [1, 32])
                        t__row_major_tmp_v20: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_60, pl.const(148224, pl.INT64), 128), pl.Mem.Vec] = pl.tile.mul(t__rm_a0_tmp_v18, t__rm_a1_tmp_v19)
                        t__tmp_v262: pl.Tile[[32, 1], pl.FP32, pl.MemRef(mem_vec_60, pl.const(148224, pl.INT64), 128), pl.Mem.Vec] = pl.tile.reshape(t__row_major_tmp_v20, [32, 1])
                        next_l_inline2418__rm_a0_tmp_v21: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 128), pl.Mem.Vec] = pl.tile.reshape(t__tmp_v261, [1, 32])
                        next_l_inline2418__rm_a1_tmp_v22: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_60, pl.const(148224, pl.INT64), 128), pl.Mem.Vec] = pl.tile.reshape(t__tmp_v262, [1, 32])
                        next_l_inline2418__row_major_tmp_v23: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_58, pl.const(148096, pl.INT64), 128), pl.Mem.Vec] = pl.tile.add(
                            next_l_inline2418__rm_a0_tmp_v21, next_l_inline2418__rm_a1_tmp_v22
                        )
                        next_l_inline2418__ssa_v0: pl.Tile[[32, 1], pl.FP32, pl.MemRef(mem_vec_58, pl.const(148096, pl.INT64), 128), pl.Mem.Vec] = pl.tile.reshape(
                            next_l_inline2418__row_major_tmp_v23, [32, 1]
                        )
                        pv_left_inline2390__ssa_v0: pl.Tile[[32, 256], pl.FP32, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.load(
                            pv_transfer_inline2347__ssa_v0, [pv_transfer_row_inline2329__ssa_v1 + qk_lane_head_inline2383__ssa_v0, 0], [32, 256], [32, 256], target_memory=pl.Mem.Vec
                        )
                        t__tmp_v263: pl.Tile[[32, 256], pl.FP32, pl.MemRef(mem_vec_60, pl.const(148224, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.row_expand_mul(left_iter, alpha_inline2363__ssa_v0)
                        t__tmp_v264: pl.Tile[[32, 256], pl.FP32, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.row_expand_mul(
                            pv_left_inline2390__ssa_v0, beta_inline2312__ssa_v0
                        )
                        next_left_inline2421__ssa_v0: pl.Tile[[32, 256], pl.FP32, pl.MemRef(mem_vec_23, pl.const(16640, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.add(t__tmp_v263, t__tmp_v264)
                        pv_right_inline2288__ssa_v0: pl.Tile[[32, 256], pl.FP32, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.load(
                            pv_transfer_inline2347__ssa_v0, [pv_transfer_row_inline2329__ssa_v1 + qk_lane_head_inline2383__ssa_v0, 256], [32, 256], [32, 256], target_memory=pl.Mem.Vec
                        )
                        t__tmp_v265: pl.Tile[[32, 256], pl.FP32, pl.MemRef(mem_vec_60, pl.const(148224, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.row_expand_mul(right_iter, alpha_inline2363__ssa_v0)
                        t__tmp_v266: pl.Tile[[32, 256], pl.FP32, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.row_expand_mul(
                            pv_right_inline2288__ssa_v0, beta_inline2312__ssa_v0
                        )
                        next_right_inline2415__ssa_v0: pl.Tile[[32, 256], pl.FP32, pl.MemRef(mem_vec_24, pl.const(49408, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.add(t__tmp_v265, t__tmp_v266)
                        m_valid_inline2373__rv_v0, l_valid_inline2411__rv_v0, left_valid_inline2412__rv_v0, right_valid_inline2338__rv_v0 = pl.yield_(
                            next_m_inline2417__ssa_v0, next_l_inline2418__ssa_v0, next_left_inline2421__ssa_v0, next_right_inline2415__ssa_v0
                        )
                    else:
                        m_iter_mv: pl.Tile[[32, 1], pl.FP32, pl.MemRef(mem_vec_47, pl.const(147712, pl.INT64), 128), pl.Mem.Vec] = pl.tile.move(m_iter, target_memory=pl.Mem.Vec)
                        l_iter_mv: pl.Tile[[32, 1], pl.FP32, pl.MemRef(mem_vec_58, pl.const(148096, pl.INT64), 128), pl.Mem.Vec] = pl.tile.move(l_iter, target_memory=pl.Mem.Vec)
                        m_valid_inline2373__rv_v0, l_valid_inline2411__rv_v0, left_valid_inline2412__rv_v0, right_valid_inline2338__rv_v0 = pl.yield_(m_iter_mv, l_iter_mv, left_iter, right_iter)
                    m_after_inline2420__rv_v0, l_after_inline2409__rv_v0, left_after_inline2342__rv_v0, right_after_inline2410__rv_v0 = pl.yield_(
                        m_valid_inline2373__rv_v0, l_valid_inline2411__rv_v0, left_valid_inline2412__rv_v0, right_valid_inline2338__rv_v0
                    )
                else:
                    m_iter_mv_1: pl.Tile[[32, 1], pl.FP32, pl.MemRef(mem_vec_47, pl.const(147712, pl.INT64), 128), pl.Mem.Vec] = pl.tile.move(m_iter, target_memory=pl.Mem.Vec)
                    l_iter_mv_1: pl.Tile[[32, 1], pl.FP32, pl.MemRef(mem_vec_58, pl.const(148096, pl.INT64), 128), pl.Mem.Vec] = pl.tile.move(l_iter, target_memory=pl.Mem.Vec)
                    m_after_inline2420__rv_v0, l_after_inline2409__rv_v0, left_after_inline2342__rv_v0, right_after_inline2410__rv_v0 = pl.yield_(m_iter_mv_1, l_iter_mv_1, left_iter, right_iter)
                l_after_inline2409__rv_v0_mv: pl.Tile[[32, 1], pl.FP32, pl.MemRef(mem_vec_22, pl.const(16512, pl.INT64), 128), pl.Mem.Vec] = pl.tile.move(
                    l_after_inline2409__rv_v0, target_memory=pl.Mem.Vec
                )
                m_after_inline2420__rv_v0_mv: pl.Tile[[32, 1], pl.FP32, pl.MemRef(mem_vec_21, pl.const(16384, pl.INT64), 128), pl.Mem.Vec] = pl.tile.move(
                    m_after_inline2420__rv_v0, target_memory=pl.Mem.Vec
                )
                running_m_inline2393, running_l_inline2397, running_left_inline2297, running_right_inline2367 = pl.yield_(
                    m_after_inline2420__rv_v0_mv, l_after_inline2409__rv_v0_mv, left_after_inline2342__rv_v0, right_after_inline2410__rv_v0
                )
            qk_output_row_inline2384__ssa_v0: pl.Scalar[pl.INDEX] = qk_t_inline2286__idx_v0 * 64 + qk_lane_head_inline2383__ssa_v0
            attn_mi_inline2305__store: pl.Tensor[[t_heads_inline2333__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_17", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                running_m_inline2393, [qk_output_row_inline2384__ssa_v0, 0], attn_mi_inline2305__ssa_v0
            )
            attn_li_inline2396__store: pl.Tensor[[t_heads_inline2333__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_18", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                running_l_inline2397, [qk_output_row_inline2384__ssa_v0, 0], attn_li_inline2396__ssa_v0
            )
            attn_oi_inline2303__store: pl.Tensor[[t_heads_inline2333__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_19", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                running_left_inline2297, [qk_output_row_inline2384__ssa_v0, 0], attn_oi_inline2303__ssa_v0
            )
            attn_oi_inline2303__store_v0: pl.Tensor[[t_heads_inline2333__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_19", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                running_right_inline2367, [qk_output_row_inline2384__ssa_v0, 256], attn_oi_inline2303__ssa_v0
            )
        return (
            kv_transfer_inline2324__ssa_v0,
            score_transfer_inline2293__ssa_v0,
            probability_transfer_inline2339__ssa_v0,
            pv_transfer_inline2347__ssa_v0,
            mi_transfer_inline2291__ssa_v0,
            li_transfer_inline2388__ssa_v0,
            attn_mi_inline2305__ssa_v0,
            attn_li_inline2396__ssa_v0,
            attn_oi_inline2303__ssa_v0,
        )

    @pl.function(type=pl.FunctionType.Group, level=pl.Level.CORE_GROUP, role=pl.Role.SubWorker)
    def qk_pv(
        ffts_workspace_inline2289__ssa_v0: pl.Tensor[[256], pl.INT64, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 2048)],
        t_dim_inline2318__ssa_v0: pl.Scalar[pl.INDEX],
        q_flat_inline2306__ssa_v0: pl.Tensor[[t_heads_inline2333__ssa_v0, 512], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)],
        valid_block_mask_inline2310__ssa_v0: pl.Tensor[[t_dim_inline2318__ssa_v0, 16], pl.INT32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)],
        kv_transfer_inline2324__ssa_v0: pl.InOut[pl.Tensor[[9216, 512], pl.BF16, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 9437184)]],
        score_transfer_inline2293__ssa_v0: pl.InOut[pl.Tensor[[4608, 128], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 2359296)]],
        probability_transfer_inline2339__ssa_v0: pl.InOut[pl.Tensor[[4608, 128], pl.BF16, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 1179648)]],
        pv_transfer_inline2347__ssa_v0: pl.InOut[pl.Tensor[[4608, 512], pl.FP32, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 9437184)]],
        attn_sink_col_inline2332__ssa_v0: pl.Tensor[[64, 1], pl.FP32, pl.MemRef("mem_ddr_7", pl.const(0, pl.INT64), 256)],
        position_ids_t1__ssa_v0: pl.Tensor[[T_DYN, 1], pl.INT32, pl.MemRef("mem_ddr_8", pl.const(0, pl.INT64), 0)],
        window_swa_indices__ssa_v0: pl.Tensor[[T_DYN, 128], pl.INT32, pl.MemRef("mem_ddr_9", pl.const(0, pl.INT64), 0)],
        ori_kv_flat_inline2377__ssa_v1: pl.Tensor[[ori_block_num_inline2385__ssa_v0 * pl.const(32, pl.INDEX), 512], pl.BF16, pl.MemRef("mem_ddr_10", pl.const(0, pl.INT64), 0)],
        cmp_sparse_indices_inline2323__rv_v2: pl.Tensor[[t_dim_inline2318__ssa_v0, 512], pl.INT32, pl.MemRef("mem_ddr_11", pl.const(0, pl.INT64), 0)],
        cmp_block_table__ssa_v0: pl.Tensor[[B_DYN, COMPRESSED_TABLE_COLUMNS_DYN], pl.INT32, pl.MemRef("mem_ddr_12", pl.const(0, pl.INT64), 0)],
        cmp_kv_flat_inline2399__ssa_v0: pl.Tensor[[cmp_block_num_inline2362__ssa_v0 * pl.const(32, pl.INDEX), 512], pl.BF16, pl.MemRef("mem_ddr_13", pl.const(0, pl.INT64), 0)],
        sparse_bias_inline2369__rv_v2: pl.Tensor[[t_dim_inline2318__ssa_v0, 640], pl.FP32, pl.MemRef("mem_ddr_14", pl.const(0, pl.INT64), 0)],
        mi_transfer_inline2291__ssa_v0: pl.InOut[pl.Tensor[[4608, 1], pl.FP32, pl.MemRef("mem_ddr_15", pl.const(0, pl.INT64), 18432)]],
        li_transfer_inline2388__ssa_v0: pl.InOut[pl.Tensor[[4608, 1], pl.FP32, pl.MemRef("mem_ddr_16", pl.const(0, pl.INT64), 18432)]],
        attn_mi_inline2305__ssa_v0: pl.Out[pl.Tensor[[t_heads_inline2333__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_17", pl.const(0, pl.INT64), 0)]],
        attn_li_inline2396__ssa_v0: pl.Out[pl.Tensor[[t_heads_inline2333__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_18", pl.const(0, pl.INT64), 0)]],
        attn_oi_inline2303__ssa_v0: pl.Out[pl.Tensor[[t_heads_inline2333__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_19", pl.const(0, pl.INT64), 0)]],
    ) -> tuple[
        pl.Tensor[[9216, 512], pl.BF16],
        pl.Tensor[[4608, 128], pl.FP32],
        pl.Tensor[[4608, 128], pl.BF16],
        pl.Tensor[[4608, 512], pl.FP32],
        pl.Tensor[[4608, 1], pl.FP32],
        pl.Tensor[[4608, 1], pl.FP32],
        pl.Tensor[[t_heads_inline2333__ssa_v0, 1], pl.FP32],
        pl.Tensor[[t_heads_inline2333__ssa_v0, 1], pl.FP32],
        pl.Tensor[[t_heads_inline2333__ssa_v0, 512], pl.FP32],
    ]:
        pl.func_attr({"mx_tensor_views_blocked": True, "split_aiv": True})
        self.qk_pv_aic(
            ffts_workspace_inline2289__ssa_v0,
            t_dim_inline2318__ssa_v0,
            q_flat_inline2306__ssa_v0,
            valid_block_mask_inline2310__ssa_v0,
            kv_transfer_inline2324__ssa_v0,
            score_transfer_inline2293__ssa_v0,
            probability_transfer_inline2339__ssa_v0,
            pv_transfer_inline2347__ssa_v0,
            attn_sink_col_inline2332__ssa_v0,
            position_ids_t1__ssa_v0,
            window_swa_indices__ssa_v0,
            ori_kv_flat_inline2377__ssa_v1,
            cmp_sparse_indices_inline2323__rv_v2,
            cmp_block_table__ssa_v0,
            cmp_kv_flat_inline2399__ssa_v0,
            sparse_bias_inline2369__rv_v2,
            mi_transfer_inline2291__ssa_v0,
            li_transfer_inline2388__ssa_v0,
            attn_mi_inline2305__ssa_v0,
            attn_li_inline2396__ssa_v0,
            attn_oi_inline2303__ssa_v0,
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
                    pl.adir.input,
                    pl.adir.input,
                    pl.adir.input,
                    pl.adir.input,
                    pl.adir.input,
                    pl.adir.input,
                    pl.adir.input,
                    pl.adir.inout,
                    pl.adir.inout,
                    pl.adir.output_existing,
                    pl.adir.output_existing,
                    pl.adir.output_existing,
                ]
            },
        )
        self.qk_pv_aiv(
            ffts_workspace_inline2289__ssa_v0,
            t_dim_inline2318__ssa_v0,
            q_flat_inline2306__ssa_v0,
            valid_block_mask_inline2310__ssa_v0,
            kv_transfer_inline2324__ssa_v0,
            score_transfer_inline2293__ssa_v0,
            probability_transfer_inline2339__ssa_v0,
            pv_transfer_inline2347__ssa_v0,
            attn_sink_col_inline2332__ssa_v0,
            position_ids_t1__ssa_v0,
            window_swa_indices__ssa_v0,
            ori_kv_flat_inline2377__ssa_v1,
            cmp_sparse_indices_inline2323__rv_v2,
            cmp_block_table__ssa_v0,
            cmp_kv_flat_inline2399__ssa_v0,
            sparse_bias_inline2369__rv_v2,
            mi_transfer_inline2291__ssa_v0,
            li_transfer_inline2388__ssa_v0,
            attn_mi_inline2305__ssa_v0,
            attn_li_inline2396__ssa_v0,
            attn_oi_inline2303__ssa_v0,
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
                    pl.adir.inout,
                ]
            },
        )
        return (
            kv_transfer_inline2324__ssa_v0,
            score_transfer_inline2293__ssa_v0,
            probability_transfer_inline2339__ssa_v0,
            pv_transfer_inline2347__ssa_v0,
            mi_transfer_inline2291__ssa_v0,
            li_transfer_inline2388__ssa_v0,
            attn_mi_inline2305__ssa_v0,
            attn_li_inline2396__ssa_v0,
            attn_oi_inline2303__ssa_v0,
        )

    @pl.function(type=pl.FunctionType.Spmd)
    def qk_pv_spmd(
        self,
        ffts_workspace_inline2289__ssa_v0: pl.Tensor[[256], pl.INT64, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 2048)],
        t_dim_inline2318__ssa_v0: pl.Scalar[pl.INDEX],
        q_flat_inline2306__ssa_v0: pl.Tensor[[t_heads_inline2333__ssa_v0, 512], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)],
        valid_block_mask_inline2310__ssa_v0: pl.Tensor[[t_dim_inline2318__ssa_v0, 16], pl.INT32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)],
        kv_transfer_inline2324__ssa_v0: pl.InOut[pl.Tensor[[9216, 512], pl.BF16, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 9437184)]],
        score_transfer_inline2293__ssa_v0: pl.InOut[pl.Tensor[[4608, 128], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 2359296)]],
        probability_transfer_inline2339__ssa_v0: pl.InOut[pl.Tensor[[4608, 128], pl.BF16, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 1179648)]],
        pv_transfer_inline2347__ssa_v0: pl.InOut[pl.Tensor[[4608, 512], pl.FP32, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 9437184)]],
        attn_sink_col_inline2332__ssa_v0: pl.Tensor[[64, 1], pl.FP32, pl.MemRef("mem_ddr_7", pl.const(0, pl.INT64), 256)],
        position_ids_t1__ssa_v0: pl.Tensor[[T_DYN, 1], pl.INT32, pl.MemRef("mem_ddr_8", pl.const(0, pl.INT64), 0)],
        window_swa_indices__ssa_v0: pl.Tensor[[T_DYN, 128], pl.INT32, pl.MemRef("mem_ddr_9", pl.const(0, pl.INT64), 0)],
        ori_kv_flat_inline2377__ssa_v1: pl.Tensor[[ori_block_num_inline2385__ssa_v0 * pl.const(32, pl.INDEX), 512], pl.BF16, pl.MemRef("mem_ddr_10", pl.const(0, pl.INT64), 0)],
        cmp_sparse_indices_inline2323__rv_v2: pl.Tensor[[t_dim_inline2318__ssa_v0, 512], pl.INT32, pl.MemRef("mem_ddr_11", pl.const(0, pl.INT64), 0)],
        cmp_block_table__ssa_v0: pl.Tensor[[B_DYN, COMPRESSED_TABLE_COLUMNS_DYN], pl.INT32, pl.MemRef("mem_ddr_12", pl.const(0, pl.INT64), 0)],
        cmp_kv_flat_inline2399__ssa_v0: pl.Tensor[[cmp_block_num_inline2362__ssa_v0 * pl.const(32, pl.INDEX), 512], pl.BF16, pl.MemRef("mem_ddr_13", pl.const(0, pl.INT64), 0)],
        sparse_bias_inline2369__rv_v2: pl.Tensor[[t_dim_inline2318__ssa_v0, 640], pl.FP32, pl.MemRef("mem_ddr_14", pl.const(0, pl.INT64), 0)],
        mi_transfer_inline2291__ssa_v0: pl.InOut[pl.Tensor[[4608, 1], pl.FP32, pl.MemRef("mem_ddr_15", pl.const(0, pl.INT64), 18432)]],
        li_transfer_inline2388__ssa_v0: pl.InOut[pl.Tensor[[4608, 1], pl.FP32, pl.MemRef("mem_ddr_16", pl.const(0, pl.INT64), 18432)]],
        attn_mi_inline2305__ssa_v0: pl.Out[pl.Tensor[[t_heads_inline2333__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_17", pl.const(0, pl.INT64), 0)]],
        attn_li_inline2396__ssa_v0: pl.Out[pl.Tensor[[t_heads_inline2333__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_18", pl.const(0, pl.INT64), 0)]],
        attn_oi_inline2303__ssa_v0: pl.Out[pl.Tensor[[t_heads_inline2333__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_19", pl.const(0, pl.INT64), 0)]],
    ) -> tuple[pl.Tensor[[t_heads_inline2333__ssa_v0, 1], pl.FP32], pl.Tensor[[t_heads_inline2333__ssa_v0, 1], pl.FP32], pl.Tensor[[t_heads_inline2333__ssa_v0, 512], pl.FP32]]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        ret__tmp_v0: pl.Tuple[
            pl.Tensor[[9216, 512], pl.BF16],
            pl.Tensor[[4608, 128], pl.FP32],
            pl.Tensor[[4608, 128], pl.BF16],
            pl.Tensor[[4608, 512], pl.FP32],
            pl.Tensor[[4608, 1], pl.FP32],
            pl.Tensor[[4608, 1], pl.FP32],
            pl.Tensor[[t_heads_inline2333__ssa_v0, 1], pl.FP32],
            pl.Tensor[[t_heads_inline2333__ssa_v0, 1], pl.FP32],
            pl.Tensor[[t_heads_inline2333__ssa_v0, 512], pl.FP32],
        ] = self.qk_pv(
            ffts_workspace_inline2289__ssa_v0,
            t_dim_inline2318__ssa_v0,
            q_flat_inline2306__ssa_v0,
            valid_block_mask_inline2310__ssa_v0,
            kv_transfer_inline2324__ssa_v0,
            score_transfer_inline2293__ssa_v0,
            probability_transfer_inline2339__ssa_v0,
            pv_transfer_inline2347__ssa_v0,
            attn_sink_col_inline2332__ssa_v0,
            position_ids_t1__ssa_v0,
            window_swa_indices__ssa_v0,
            ori_kv_flat_inline2377__ssa_v1,
            cmp_sparse_indices_inline2323__rv_v2,
            cmp_block_table__ssa_v0,
            cmp_kv_flat_inline2399__ssa_v0,
            sparse_bias_inline2369__rv_v2,
            mi_transfer_inline2291__ssa_v0,
            li_transfer_inline2388__ssa_v0,
            attn_mi_inline2305__ssa_v0,
            attn_li_inline2396__ssa_v0,
            attn_oi_inline2303__ssa_v0,
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
                    pl.adir.input,
                    pl.adir.input,
                    pl.adir.input,
                    pl.adir.input,
                    pl.adir.input,
                    pl.adir.input,
                    pl.adir.input,
                    pl.adir.inout,
                    pl.adir.inout,
                    pl.adir.output_existing,
                    pl.adir.output_existing,
                    pl.adir.output_existing,
                ]
            },
        )
        kv_transfer_inline2324__ssa_v1: pl.Tensor[[9216, 512], pl.BF16, pl.MemRef("mem_ddr_20", pl.const(0, pl.INT64), 9437184)] = ret__tmp_v0[0]
        score_transfer_inline2293__ssa_v1: pl.Tensor[[4608, 128], pl.FP32, pl.MemRef("mem_ddr_21", pl.const(0, pl.INT64), 2359296)] = ret__tmp_v0[1]
        probability_transfer_inline2339__ssa_v1: pl.Tensor[[4608, 128], pl.BF16, pl.MemRef("mem_ddr_22", pl.const(0, pl.INT64), 1179648)] = ret__tmp_v0[2]
        pv_transfer_inline2347__ssa_v1: pl.Tensor[[4608, 512], pl.FP32, pl.MemRef("mem_ddr_23", pl.const(0, pl.INT64), 9437184)] = ret__tmp_v0[3]
        mi_transfer_inline2291__ssa_v1: pl.Tensor[[4608, 1], pl.FP32, pl.MemRef("mem_ddr_24", pl.const(0, pl.INT64), 18432)] = ret__tmp_v0[4]
        li_transfer_inline2388__ssa_v1: pl.Tensor[[4608, 1], pl.FP32, pl.MemRef("mem_ddr_25", pl.const(0, pl.INT64), 18432)] = ret__tmp_v0[5]
        attn_mi_inline2305__ssa_v1: pl.Tensor[[t_heads_inline2333__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_26", pl.const(0, pl.INT64), 0)] = ret__tmp_v0[6]
        attn_li_inline2396__ssa_v1: pl.Tensor[[t_heads_inline2333__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_27", pl.const(0, pl.INT64), 0)] = ret__tmp_v0[7]
        attn_oi_inline2303__ssa_v1: pl.Tensor[[t_heads_inline2333__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_28", pl.const(0, pl.INT64), 0)] = ret__tmp_v0[8]
        return attn_mi_inline2305__ssa_v0, attn_li_inline2396__ssa_v0, attn_oi_inline2303__ssa_v0

    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def qproj_dequant_rms_nope_rope(
        q_flat_inline2682__ssa_v0: pl.Out[pl.Tensor[[t_dim_inline2655__ssa_v0, 32768], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)]],
        tile_rows_inline366_inline1656__ssa_v0: pl.Scalar[pl.INDEX],
        tile_base_inline367_inline1657__idx_v0: pl.Scalar[pl.INDEX],
        qr_scale_pad_store_inline1684__rv_v2: pl.Tensor[[512, 1], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 2048)],
        q_rope_cos_il_inline503__ssa_v0: pl.Tensor[[t_dim_inline504__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)],
        q_rope_sin_signed_inline502__ssa_v0: pl.Tensor[[t_dim_inline504__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)],
        q_rope_swap_idx_inline501__ssa_v0: pl.Tensor[[t_dim_inline504__ssa_v0, 64], pl.INT32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 0)],
        q_proj_i32_inline364_inline1723__ssa_v9: pl.Tensor[[qproj_t_matmul_inline365_inline1672__ssa_v0, 32768], pl.INT32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 0)],
        wq_b_scale__ssa_v0: pl.Tensor[[32768], pl.FP32, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 131072)],
    ):
        pl.func_attr({"mx_tensor_views_blocked": True})
        mem_vec_7: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32)
        mem_vec_8: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 2048)
        mem_vec_9: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 2048)
        mem_vec_10: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 2048)
        mem_vec_11: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
        mem_vec_13: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
        mem_vec_18: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
        mem_vec_19: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
        mem_vec_29: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        mem_vec_30: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        mem_vec_39: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
        mem_vec_40: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
        mem_vec_50: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        mem_vec_51: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        dq_worker_inline2676__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        for dq_work_inline2662__idx_v0, (q_flat_inline2682__iter_v1,) in pl.range(
            dq_worker_inline2676__ssa_v0, (tile_rows_inline366_inline1656__ssa_v0 + 7) // 8 * 16, 48, init_values=(q_flat_inline2682__ssa_v0,)
        ):
            hg_inline2671__ssa_v0: pl.Scalar[pl.INDEX] = dq_work_inline2662__idx_v0 % 16 * 4
            tg_inline2679__ssa_v0: pl.Scalar[pl.INDEX] = dq_work_inline2662__idx_v0 // 16 * 8
            out_tg_inline2674__ssa_v0: pl.Scalar[pl.INDEX] = tile_base_inline367_inline1657__idx_v0 + tg_inline2679__ssa_v0
            if tg_inline2679__ssa_v0 + 8 <= tile_rows_inline366_inline1656__ssa_v0:
                qr_scale_dq_t_inline2684__tile: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_7, pl.const(101376, pl.INT64), 32), pl.Mem.Vec] = pl.tile.load(
                    qr_scale_pad_store_inline1684__rv_v2, [tg_inline2679__ssa_v0, 0], [8, 1], [8, 1], target_memory=pl.Mem.Vec
                )
                q_cos_il_inline2713__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_8, pl.const(101408, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.load(
                    q_rope_cos_il_inline503__ssa_v0, [out_tg_inline2674__ssa_v0, 0], [8, 64], [8, 64], target_memory=pl.Mem.Vec
                )
                q_sin_signed_inline2686__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_9, pl.const(103456, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.load(
                    q_rope_sin_signed_inline502__ssa_v0, [out_tg_inline2674__ssa_v0, 0], [8, 64], [8, 64], target_memory=pl.Mem.Vec
                )
                q_swap_idx_inline2672__tile: pl.Tile[[8, 64], pl.INT32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.load(
                    q_rope_swap_idx_inline501__ssa_v0, [out_tg_inline2674__ssa_v0, 0], [8, 64], [8, 64], target_memory=pl.Mem.Vec
                )
                for h_inner_inline2711__idx_v0, (q_flat_inline2682__iter_v3,) in pl.range(0, 4, 2, init_values=(q_flat_inline2682__iter_v1,)):
                    h_inline2689__ssa_v0: pl.Scalar[pl.INDEX] = hg_inline2671__ssa_v0 + h_inner_inline2711__idx_v0
                    h0_inline2670__ssa_v0: pl.Scalar[pl.INDEX] = h_inline2689__ssa_v0 * 512
                    h_inline2689__ssa_v0_1: pl.Scalar[pl.INDEX] = hg_inline2671__ssa_v0 + (h_inner_inline2711__idx_v0 + 1)
                    h0_inline2670__ssa_v0_1: pl.Scalar[pl.INDEX] = h_inline2689__ssa_v0_1 * 512
                    q_head_acc_inline2667__tile: pl.Tile[[8, 512], pl.INT32, pl.MemRef(mem_vec_11, pl.const(2048, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.load(
                        q_proj_i32_inline364_inline1723__ssa_v9, [tg_inline2679__ssa_v0, h0_inline2670__ssa_v0], [8, 512], [8, 512], target_memory=pl.Mem.Vec
                    )
                    t__tile: pl.Tile[[512], pl.FP32, pl.MemRef(mem_vec_18, pl.const(34816, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.load(
                        wq_b_scale__ssa_v0, [h0_inline2670__ssa_v0], [512], [512], target_memory=pl.Mem.Vec
                    )
                    q_head_acc_inline2667__tile_1: pl.Tile[[8, 512], pl.INT32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.load(
                        q_proj_i32_inline364_inline1723__ssa_v9, [tg_inline2679__ssa_v0, h0_inline2670__ssa_v0_1], [8, 512], [8, 512], target_memory=pl.Mem.Vec
                    )
                    t__tile_1: pl.Tile[[512], pl.FP32, pl.MemRef(mem_vec_39, pl.const(68096, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.load(
                        wq_b_scale__ssa_v0, [h0_inline2670__ssa_v0_1], [512], [512], target_memory=pl.Mem.Vec
                    )
                    q_head_scale_inline2666__tile: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_18, pl.const(34816, pl.INT64), 2048), pl.Mem.Vec] = t__tile
                    q_head_acc_fp32_inline2668__tile: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_11, pl.const(2048, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.cast(
                        q_head_acc_inline2667__tile, target_type=pl.FP32, mode="none"
                    )
                    q_head_row_scaled_inline2654__tile: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_11, pl.const(2048, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.row_expand_mul(
                        q_head_acc_fp32_inline2668__tile, qr_scale_dq_t_inline2684__tile
                    )
                    q_head_dq_inline2664__tile: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_11, pl.const(2048, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.col_expand_mul(
                        q_head_row_scaled_inline2654__tile, q_head_scale_inline2666__tile
                    )
                    q_head_sq_inline2663__tile: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_18, pl.const(34816, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.mul(
                        q_head_dq_inline2664__tile, q_head_dq_inline2664__tile
                    )
                    tmp_tile: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_19, pl.const(51200, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.create([8, 512], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                    q_head_sq_row_inline2707__tile: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_29, pl.const(67584, pl.INT64), 32), pl.Mem.Vec] = pl.tile.row_sum(q_head_sq_inline2663__tile, tmp_tile)
                    q_head_sq_sum_inline2661__tile: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_29, pl.const(67584, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(q_head_sq_row_inline2707__tile, [1, 8])
                    q_head_sq_mean_inline2657__tile: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_18, pl.const(34816, pl.INT64), 32), pl.Mem.Vec] = pl.tile.muls(
                        q_head_sq_sum_inline2661__tile, 0.001953125
                    )
                    q_head_var_inline2687__tile: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_18, pl.const(34816, pl.INT64), 32), pl.Mem.Vec] = pl.tile.adds(
                        q_head_sq_mean_inline2657__tile, 9.9999999999999995e-07
                    )
                    rsqrt_tmp: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_19, pl.const(51200, pl.INT64), 32), pl.Mem.Vec] = pl.tile.create([1, 8], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                    q_head_inv_rms_inline2659__tile: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_29, pl.const(67584, pl.INT64), 32), pl.Mem.Vec] = pl.tile.rsqrt(q_head_var_inline2687__tile, rsqrt_tmp)
                    q_head_inv_rms_t_inline2658__tile: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_29, pl.const(67584, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(
                        q_head_inv_rms_inline2659__tile, [8, 1]
                    )
                    t__tile_2: pl.Tile[[8, 448], pl.FP32, pl.MemRef(mem_vec_11, pl.const(2048, pl.INT64), 16128), pl.Mem.Vec] = pl.tile.slice(q_head_dq_inline2664__tile, [8, 448], [0, 0])
                    q_nope_normed_inline2656__tile: pl.Tile[[8, 448], pl.FP32, pl.MemRef(mem_vec_18, pl.const(34816, pl.INT64), 14336), pl.Mem.Vec] = pl.tile.row_expand_mul(
                        t__tile_2, q_head_inv_rms_t_inline2658__tile
                    )
                    q_nope_bf16_inline2653__tile: pl.Tile[[8, 448], pl.BF16, pl.MemRef(mem_vec_18, pl.const(34816, pl.INT64), 7168), pl.Mem.Vec] = pl.tile.cast(
                        q_nope_normed_inline2656__tile, target_type=pl.BF16, mode="rint"
                    )
                    q_rope_chunk_raw_inline2660__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_11, pl.const(3840, pl.INT64), 14592), pl.Mem.Vec] = pl.tile.slice(
                        q_head_dq_inline2664__tile, [8, 64], [0, 448]
                    )
                    q_rope_chunk_inline2680__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_11, pl.const(2048, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.row_expand_mul(
                        q_rope_chunk_raw_inline2660__tile, q_head_inv_rms_t_inline2658__tile
                    )
                    gather_acc_init: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_19, pl.const(51200, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.create([8, 64], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                    for gather_lv, (gather_ia,) in pl.range(8, init_values=(gather_acc_init,)):
                        gather_inp_row: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_11, pl.const(2048, pl.INT64), 256), pl.Mem.Vec] = pl.tile.slice(
                            q_rope_chunk_inline2680__tile, [1, 64], [gather_lv, 0], [1, 64]
                        )
                        gather_idx_row: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 256), pl.Mem.Vec] = pl.tile.slice(
                            q_swap_idx_inline2672__tile, [1, 64], [gather_lv, 0], [1, 64]
                        )
                        gather_row_tmp: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_29, pl.const(67584, pl.INT64), 256), pl.Mem.Vec] = pl.tile.create(
                            [1, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec
                        )
                        gather_row: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_30, pl.const(67840, pl.INT64), 256), pl.Mem.Vec] = pl.tile.gather(gather_inp_row, gather_idx_row, gather_row_tmp)
                        gather_asmbl: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_19, pl.const(51200, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.assemble(gather_ia, gather_row, [gather_lv, 0])
                        gather_rv: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_19, pl.const(51200, pl.INT64), 2048), pl.Mem.Vec] = pl.yield_(gather_asmbl)
                    q_rope_swapped_inline2690__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_19, pl.const(51200, pl.INT64), 2048), pl.Mem.Vec] = gather_rv
                    q_rope_base_inline2703__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_11, pl.const(2048, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.mul(
                        q_rope_chunk_inline2680__tile, q_cos_il_inline2713__tile
                    )
                    q_rope_delta_inline2669__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_19, pl.const(51200, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.mul(
                        q_rope_swapped_inline2690__tile, q_sin_signed_inline2686__tile
                    )
                    q_rope_rot_inline2691__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_11, pl.const(2048, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.add(
                        q_rope_base_inline2703__tile, q_rope_delta_inline2669__tile
                    )
                    q_rope_bf16_inline2692__tile: pl.Tile[[8, 64], pl.BF16, pl.MemRef(mem_vec_11, pl.const(2048, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.cast(
                        q_rope_rot_inline2691__tile, target_type=pl.BF16, mode="rint"
                    )
                    q_flat_inline2682__tile: pl.Tensor[[t_dim_inline2655__ssa_v0, 32768], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                        q_nope_bf16_inline2653__tile, [out_tg_inline2674__ssa_v0, h0_inline2670__ssa_v0], q_flat_inline2682__iter_v3
                    )
                    q_flat_inline2682__tile_1: pl.Tensor[[t_dim_inline2655__ssa_v0, 32768], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                        q_rope_bf16_inline2692__tile, [out_tg_inline2674__ssa_v0, h0_inline2670__ssa_v0 + 448], q_flat_inline2682__tile
                    )
                    q_head_scale_inline2666__tile_1: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_39, pl.const(68096, pl.INT64), 2048), pl.Mem.Vec] = t__tile_1
                    q_head_acc_fp32_inline2668__tile_1: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.cast(
                        q_head_acc_inline2667__tile_1, target_type=pl.FP32, mode="none"
                    )
                    q_head_row_scaled_inline2654__tile_1: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.row_expand_mul(
                        q_head_acc_fp32_inline2668__tile_1, qr_scale_dq_t_inline2684__tile
                    )
                    q_head_dq_inline2664__tile_1: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.col_expand_mul(
                        q_head_row_scaled_inline2654__tile_1, q_head_scale_inline2666__tile_1
                    )
                    q_head_sq_inline2663__tile_1: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_39, pl.const(68096, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.mul(
                        q_head_dq_inline2664__tile_1, q_head_dq_inline2664__tile_1
                    )
                    tmp_tile_1: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_40, pl.const(84480, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.create([8, 512], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                    q_head_sq_row_inline2707__tile_1: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_50, pl.const(100864, pl.INT64), 32), pl.Mem.Vec] = pl.tile.row_sum(
                        q_head_sq_inline2663__tile_1, tmp_tile_1
                    )
                    q_head_sq_sum_inline2661__tile_1: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_50, pl.const(100864, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(
                        q_head_sq_row_inline2707__tile_1, [1, 8]
                    )
                    q_head_sq_mean_inline2657__tile_1: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_39, pl.const(68096, pl.INT64), 32), pl.Mem.Vec] = pl.tile.muls(
                        q_head_sq_sum_inline2661__tile_1, 0.001953125
                    )
                    q_head_var_inline2687__tile_1: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_39, pl.const(68096, pl.INT64), 32), pl.Mem.Vec] = pl.tile.adds(
                        q_head_sq_mean_inline2657__tile_1, 9.9999999999999995e-07
                    )
                    rsqrt_tmp_1: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_40, pl.const(84480, pl.INT64), 32), pl.Mem.Vec] = pl.tile.create([1, 8], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                    q_head_inv_rms_inline2659__tile_1: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_50, pl.const(100864, pl.INT64), 32), pl.Mem.Vec] = pl.tile.rsqrt(
                        q_head_var_inline2687__tile_1, rsqrt_tmp_1
                    )
                    q_head_inv_rms_t_inline2658__tile_1: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_50, pl.const(100864, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(
                        q_head_inv_rms_inline2659__tile_1, [8, 1]
                    )
                    t__tile_3: pl.Tile[[8, 448], pl.FP32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 16128), pl.Mem.Vec] = pl.tile.slice(q_head_dq_inline2664__tile_1, [8, 448], [0, 0])
                    q_nope_normed_inline2656__tile_1: pl.Tile[[8, 448], pl.FP32, pl.MemRef(mem_vec_39, pl.const(68096, pl.INT64), 14336), pl.Mem.Vec] = pl.tile.row_expand_mul(
                        t__tile_3, q_head_inv_rms_t_inline2658__tile_1
                    )
                    q_nope_bf16_inline2653__tile_1: pl.Tile[[8, 448], pl.BF16, pl.MemRef(mem_vec_39, pl.const(68096, pl.INT64), 7168), pl.Mem.Vec] = pl.tile.cast(
                        q_nope_normed_inline2656__tile_1, target_type=pl.BF16, mode="rint"
                    )
                    q_rope_chunk_raw_inline2660__tile_1: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(20224, pl.INT64), 14592), pl.Mem.Vec] = pl.tile.slice(
                        q_head_dq_inline2664__tile_1, [8, 64], [0, 448]
                    )
                    q_rope_chunk_inline2680__tile_1: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.row_expand_mul(
                        q_rope_chunk_raw_inline2660__tile_1, q_head_inv_rms_t_inline2658__tile_1
                    )
                    gather_acc_init_1: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_40, pl.const(84480, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.create([8, 64], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                    for gather_lv_1, (gather_ia_1,) in pl.range(8, init_values=(gather_acc_init_1,)):
                        gather_inp_row_1: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 256), pl.Mem.Vec] = pl.tile.slice(
                            q_rope_chunk_inline2680__tile_1, [1, 64], [gather_lv_1, 0], [1, 64]
                        )
                        gather_idx_row_1: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 256), pl.Mem.Vec] = pl.tile.slice(
                            q_swap_idx_inline2672__tile, [1, 64], [gather_lv_1, 0], [1, 64]
                        )
                        gather_row_tmp_1: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_50, pl.const(100864, pl.INT64), 256), pl.Mem.Vec] = pl.tile.create(
                            [1, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec
                        )
                        gather_row_1: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_51, pl.const(101120, pl.INT64), 256), pl.Mem.Vec] = pl.tile.gather(
                            gather_inp_row_1, gather_idx_row_1, gather_row_tmp_1
                        )
                        gather_asmbl_1: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_40, pl.const(84480, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.assemble(gather_ia_1, gather_row_1, [gather_lv_1, 0])
                        gather_rv_1: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_40, pl.const(84480, pl.INT64), 2048), pl.Mem.Vec] = pl.yield_(gather_asmbl_1)
                    q_rope_swapped_inline2690__tile_1: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_40, pl.const(84480, pl.INT64), 2048), pl.Mem.Vec] = gather_rv_1
                    q_rope_base_inline2703__tile_1: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.mul(
                        q_rope_chunk_inline2680__tile_1, q_cos_il_inline2713__tile
                    )
                    q_rope_delta_inline2669__tile_1: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_40, pl.const(84480, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.mul(
                        q_rope_swapped_inline2690__tile_1, q_sin_signed_inline2686__tile
                    )
                    q_rope_rot_inline2691__tile_1: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.add(
                        q_rope_base_inline2703__tile_1, q_rope_delta_inline2669__tile_1
                    )
                    q_rope_bf16_inline2692__tile_1: pl.Tile[[8, 64], pl.BF16, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.cast(
                        q_rope_rot_inline2691__tile_1, target_type=pl.BF16, mode="rint"
                    )
                    q_flat_inline2682__tile_2: pl.Tensor[[t_dim_inline2655__ssa_v0, 32768], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                        q_nope_bf16_inline2653__tile_1, [out_tg_inline2674__ssa_v0, h0_inline2670__ssa_v0_1], q_flat_inline2682__tile_1
                    )
                    q_flat_inline2682__tile_3: pl.Tensor[[t_dim_inline2655__ssa_v0, 32768], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                        q_rope_bf16_inline2692__tile_1, [out_tg_inline2674__ssa_v0, h0_inline2670__ssa_v0_1 + 448], q_flat_inline2682__tile_2
                    )
                    q_flat_inline2682__rv_v4: pl.Tensor[[t_dim_inline2655__ssa_v0, 32768], pl.BF16, pl.MemRef("mem_ddr_57", pl.const(0, pl.INT64), 0)] = pl.yield_(q_flat_inline2682__tile_3)
                q_flat_inline2682__phi_v7: pl.Tensor[[t_dim_inline2655__ssa_v0, 32768], pl.BF16, pl.MemRef("mem_ddr_105", pl.const(0, pl.INT64), 0)] = pl.yield_(q_flat_inline2682__rv_v4)
            else:
                valid_tail_rows_inline2693__ssa_v0: pl.Scalar[pl.INDEX] = tile_rows_inline366_inline1656__ssa_v0 - tg_inline2679__ssa_v0
                qr_scale_dq_tail_inline2688__ssa_v0: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_9, pl.const(103456, pl.INT64), 32), pl.Mem.Vec] = pl.tile.load(
                    qr_scale_pad_store_inline1684__rv_v2, [tg_inline2679__ssa_v0, 0], [8, 1], [8, 1], target_memory=pl.Mem.Vec
                )
                q_cos_il_tail_inline2695__ssa_v0: pl.Tile[
                    [8, 64], pl.FP32, pl.MemRef(mem_vec_19, pl.const(51200, pl.INT64), 2048), pl.Mem.Vec, pl.TileView(valid_shape=[valid_tail_rows_inline2693__ssa_v0, 64])
                ] = pl.tile.load(q_rope_cos_il_inline503__ssa_v0, [out_tg_inline2674__ssa_v0, 0], [8, 64], [valid_tail_rows_inline2693__ssa_v0, 64], target_memory=pl.Mem.Vec)
                q_sin_signed_tail_inline2696__ssa_v0: pl.Tile[
                    [8, 64], pl.FP32, pl.MemRef(mem_vec_39, pl.const(68096, pl.INT64), 2048), pl.Mem.Vec, pl.TileView(valid_shape=[valid_tail_rows_inline2693__ssa_v0, 64])
                ] = pl.tile.load(q_rope_sin_signed_inline502__ssa_v0, [out_tg_inline2674__ssa_v0, 0], [8, 64], [valid_tail_rows_inline2693__ssa_v0, 64], target_memory=pl.Mem.Vec)
                t__tmp_v45: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_11, pl.const(2048, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.full([8, 64], dtype=pl.FP32, value=1.0)
                t__ci_tmp_v0: pl.Tile[[1, 192], pl.FP32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 768), pl.Mem.Vec] = pl.tile.create([1, 192], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                t__tmp_v46: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_18, pl.const(34816, pl.INT64), 256), pl.Mem.Vec] = pl.tile.ci(
                    pl.const(0, pl.INT32), [1, 64], tmp=t__ci_tmp_v0, dtype=pl.INT32, descending=False
                )
                t__tmp_v47: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 256), pl.Mem.Vec] = pl.tile.cast(t__tmp_v46, target_type=pl.FP32, mode="round")
                q_col_inline2677__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_11, pl.const(2048, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.col_expand_mul(t__tmp_v45, t__tmp_v47)
                t__tmp_v48: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.muls(q_col_inline2677__ssa_v0, 0.5)
                t__tmp_v49: pl.Tile[[8, 64], pl.INT32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(t__tmp_v48, target_type=pl.INT32, mode="trunc")
                q_dup_f_inline2698__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(t__tmp_v49, target_type=pl.FP32, mode="round")
                t__tmp_v50: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.muls(q_dup_f_inline2698__ssa_v0, 2.0)
                q_lane_inline2699__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.sub(q_col_inline2677__ssa_v0, t__tmp_v50)
                t__tmp_v51: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_11, pl.const(2048, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.adds(q_col_inline2677__ssa_v0, 1.0)
                t__tmp_v52: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.muls(q_lane_inline2699__ssa_v0, 2.0)
                q_swap_f_inline2701__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_11, pl.const(2048, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.sub(t__tmp_v51, t__tmp_v52)
                t__ci_tmp_v1: pl.Tile[[1, 192], pl.FP32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 768), pl.Mem.Vec] = pl.tile.create([1, 192], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                t__tmp_v53: pl.Tile[[1, 8], pl.INT32, pl.MemRef(mem_vec_18, pl.const(34816, pl.INT64), 32), pl.Mem.Vec] = pl.tile.ci(
                    pl.const(0, pl.INT32), [1, 8], tmp=t__ci_tmp_v1, dtype=pl.INT32, descending=False
                )
                t__tmp_v54: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 32), pl.Mem.Vec] = pl.tile.cast(t__tmp_v53, target_type=pl.FP32, mode="round")
                q_row_seed_inline2702__ssa_v0: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_18, pl.const(34816, pl.INT64), 32), pl.Mem.Vec] = pl.tile.muls(t__tmp_v54, 64.0)
                t__tmp_v55: pl.Tile[[64, 8], pl.FP32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.full([64, 8], dtype=pl.FP32, value=1.0)
                q_row_grid_inline2673__ssa_v0: pl.Tile[[64, 8], pl.FP32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.col_expand_mul(
                    t__tmp_v55, q_row_seed_inline2702__ssa_v0
                )
                transpose_tmp: pl.Tile[[64, 8], pl.FP32, pl.MemRef(mem_vec_18, pl.const(34816, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.create([64, 8], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                q_row_offset_inline2700__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_40, pl.const(84480, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.transpose(
                    q_row_grid_inline2673__ssa_v0, 0, 1, transpose_tmp
                )
                t__tmp_v56: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_11, pl.const(2048, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.add(q_swap_f_inline2701__ssa_v0, q_row_offset_inline2700__ssa_v0)
                q_swap_idx_tail_inline2705__ssa_v0: pl.Tile[[8, 64], pl.INT32, pl.MemRef(mem_vec_40, pl.const(84480, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(
                    t__tmp_v56, target_type=pl.INT32, mode="round"
                )
                q_head_reduce_tmp_inline2708__ssa_v0: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_11, pl.const(2048, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.create(
                    [8, 512], dtype=pl.FP32, target_memory=pl.Mem.Vec
                )
                q_gather_tmp_inline2709__ssa_v0: pl.Tile[[8, 64], pl.INT32, pl.MemRef(mem_vec_8, pl.const(101408, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.create(
                    [8, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec
                )
                for h_inner_tail_inline2710__idx_v0 in pl.range(4):
                    h_tail_inline2697__ssa_v0: pl.Scalar[pl.INDEX] = hg_inline2671__ssa_v0 + h_inner_tail_inline2710__idx_v0
                    h0_tail_inline2665__ssa_v0: pl.Scalar[pl.INDEX] = h_tail_inline2697__ssa_v0 * 512
                    q_head_acc_tail_inline2685__ssa_v0: pl.Tile[[8, 512], pl.INT32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.load(
                        q_proj_i32_inline364_inline1723__ssa_v9, [tg_inline2679__ssa_v0, h0_tail_inline2665__ssa_v0], [8, 512], [8, 512], target_memory=pl.Mem.Vec
                    )
                    q_head_scale_input_tail_inline2712__ssa_v0: pl.Tile[[512], pl.FP32, pl.MemRef(mem_vec_18, pl.const(34816, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.load(
                        wq_b_scale__ssa_v0, [h0_tail_inline2665__ssa_v0], [512], [512], target_memory=pl.Mem.Vec
                    )
                    q_head_scale_tail_inline2704__ssa_v0: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_18, pl.const(34816, pl.INT64), 2048), pl.Mem.Vec] = q_head_scale_input_tail_inline2712__ssa_v0
                    q_head_acc_fp32_tail_inline2678__ssa_v0: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.cast(
                        q_head_acc_tail_inline2685__ssa_v0, target_type=pl.FP32, mode="none"
                    )
                    q_head_row_scaled_tail_inline2675__ssa_v0: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.row_expand_mul(
                        q_head_acc_fp32_tail_inline2678__ssa_v0, qr_scale_dq_tail_inline2688__ssa_v0
                    )
                    q_head_dq_tail_inline2714__ssa_v0: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.col_expand_mul(
                        q_head_row_scaled_tail_inline2675__ssa_v0, q_head_scale_tail_inline2704__ssa_v0
                    )
                    q_head_sq_tail_inline2706__ssa_v0: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_18, pl.const(34816, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.mul(
                        q_head_dq_tail_inline2714__ssa_v0, q_head_dq_tail_inline2714__ssa_v0
                    )
                    q_head_sq_sum_tail_inline2681__ssa_v0: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 32), pl.Mem.Vec] = pl.tile.row_sum(
                        q_head_sq_tail_inline2706__ssa_v0, q_head_reduce_tmp_inline2708__ssa_v0
                    )
                    t__rm_a0_tmp_v0: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(q_head_sq_sum_tail_inline2681__ssa_v0, [1, 8])
                    t__row_major_tmp_v1: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_18, pl.const(34816, pl.INT64), 32), pl.Mem.Vec] = pl.tile.muls(t__rm_a0_tmp_v0, 0.001953125)
                    t__tmp_v57: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_18, pl.const(34816, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(t__row_major_tmp_v1, [8, 1])
                    t__rm_a0_tmp_v2: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_18, pl.const(34816, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(t__tmp_v57, [1, 8])
                    t__row_major_tmp_v3: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_18, pl.const(34816, pl.INT64), 32), pl.Mem.Vec] = pl.tile.adds(t__rm_a0_tmp_v2, 9.9999999999999995e-07)
                    t__tmp_v58: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_18, pl.const(34816, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(t__row_major_tmp_v3, [8, 1])
                    t__rm_a0_tmp_v4: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_18, pl.const(34816, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(t__tmp_v58, [1, 8])
                    t__row_major_tmp_v5: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_18, pl.const(34816, pl.INT64), 32), pl.Mem.Vec] = pl.tile.sqrt(t__rm_a0_tmp_v4)
                    t__tmp_v59: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_18, pl.const(34816, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(t__row_major_tmp_v5, [8, 1])
                    q_head_inv_rms_tail_inline2694__rm_a0_tmp_v6: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_18, pl.const(34816, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(t__tmp_v59, [1, 8])
                    q_head_inv_rms_tail_inline2694__row_major_tmp_v7: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 32), pl.Mem.Vec] = pl.tile.recip(
                        q_head_inv_rms_tail_inline2694__rm_a0_tmp_v6
                    )
                    q_head_inv_rms_tail_inline2694__ssa_v0: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(
                        q_head_inv_rms_tail_inline2694__row_major_tmp_v7, [8, 1]
                    )
                    t__tmp_v60: pl.Tile[[8, 448], pl.FP32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 16128), pl.Mem.Vec] = pl.tile.slice(q_head_dq_tail_inline2714__ssa_v0, [8, 448], [0, 0])
                    q_nope_normed_tail_inline2652__ssa_v0: pl.Tile[[8, 448], pl.FP32, pl.MemRef(mem_vec_18, pl.const(34816, pl.INT64), 14336), pl.Mem.Vec] = pl.tile.row_expand_mul(
                        t__tmp_v60, q_head_inv_rms_tail_inline2694__ssa_v0
                    )
                    q_nope_bf16_tail_inline2683__ssa_v0: pl.Tile[[8, 448], pl.BF16, pl.MemRef(mem_vec_18, pl.const(34816, pl.INT64), 7168), pl.Mem.Vec] = pl.tile.cast(
                        q_nope_normed_tail_inline2652__ssa_v0, target_type=pl.BF16, mode="rint"
                    )
                    q_nope_valid_inline2651__ssa_v0: pl.Tile[
                        [8, 448], pl.BF16, pl.MemRef(mem_vec_18, pl.const(34816, pl.INT64), 7168), pl.Mem.Vec, pl.TileView(valid_shape=[valid_tail_rows_inline2693__ssa_v0, 448])
                    ] = pl.tile.set_validshape(q_nope_bf16_tail_inline2683__ssa_v0, valid_tail_rows_inline2693__ssa_v0, 448)
                    pl.tile.store(q_nope_valid_inline2651__ssa_v0, [out_tg_inline2674__ssa_v0, h0_tail_inline2665__ssa_v0], q_flat_inline2682__iter_v1)
                    q_rope_chunk_raw_tail_inline2650__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(20224, pl.INT64), 14592), pl.Mem.Vec] = pl.tile.slice(
                        q_head_dq_tail_inline2714__ssa_v0, [8, 64], [0, 448]
                    )
                    q_rope_chunk_tail_inline2649__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.row_expand_mul(
                        q_rope_chunk_raw_tail_inline2650__ssa_v0, q_head_inv_rms_tail_inline2694__ssa_v0
                    )
                    q_rope_swapped_tail_inline2648__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_18, pl.const(34816, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.gather(
                        q_rope_chunk_tail_inline2649__ssa_v0, q_swap_idx_tail_inline2705__ssa_v0, q_gather_tmp_inline2709__ssa_v0
                    )
                    q_rope_base_tail_inline2647__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.mul(
                        q_rope_chunk_tail_inline2649__ssa_v0, q_cos_il_tail_inline2695__ssa_v0
                    )
                    q_rope_delta_tail_inline2646__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_18, pl.const(34816, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.mul(
                        q_rope_swapped_tail_inline2648__ssa_v0, q_sin_signed_tail_inline2696__ssa_v0
                    )
                    q_rope_rot_tail_inline2645__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.add(
                        q_rope_base_tail_inline2647__ssa_v0, q_rope_delta_tail_inline2646__ssa_v0
                    )
                    q_rope_bf16_tail_inline2644__ssa_v0: pl.Tile[[8, 64], pl.BF16, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.cast(
                        q_rope_rot_tail_inline2645__ssa_v0, target_type=pl.BF16, mode="rint"
                    )
                    q_rope_valid_inline2643__ssa_v0: pl.Tile[
                        [8, 64], pl.BF16, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 1024), pl.Mem.Vec, pl.TileView(valid_shape=[valid_tail_rows_inline2693__ssa_v0, 64])
                    ] = pl.tile.set_validshape(q_rope_bf16_tail_inline2644__ssa_v0, valid_tail_rows_inline2693__ssa_v0, 64)
                    pl.tile.store(q_rope_valid_inline2643__ssa_v0, [out_tg_inline2674__ssa_v0, h0_tail_inline2665__ssa_v0 + 448], q_flat_inline2682__iter_v1)
                q_flat_inline2682__phi_v7: pl.Tensor[[t_dim_inline2655__ssa_v0, 32768], pl.BF16, pl.MemRef("mem_ddr_105", pl.const(0, pl.INT64), 0)] = pl.yield_(q_flat_inline2682__iter_v1)
            q_flat_inline2682__rv_v2: pl.Tensor[[t_dim_inline2655__ssa_v0, 32768], pl.BF16, pl.MemRef("mem_ddr_106", pl.const(0, pl.INT64), 0)] = pl.yield_(q_flat_inline2682__phi_v7)
        return

    @pl.function(type=pl.FunctionType.Spmd)
    def qproj_dequant_rms_nope_rope_spmd(
        self,
        q_flat_inline2682__ssa_v0: pl.Out[pl.Tensor[[t_dim_inline2655__ssa_v0, 32768], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)]],
        tile_rows_inline366_inline1656__ssa_v0: pl.Scalar[pl.INDEX],
        tile_base_inline367_inline1657__idx_v0: pl.Scalar[pl.INDEX],
        qr_scale_pad_store_inline1684__rv_v2: pl.Tensor[[512, 1], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 2048)],
        q_rope_cos_il_inline503__ssa_v0: pl.Tensor[[t_dim_inline504__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)],
        q_rope_sin_signed_inline502__ssa_v0: pl.Tensor[[t_dim_inline504__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)],
        q_rope_swap_idx_inline501__ssa_v0: pl.Tensor[[t_dim_inline504__ssa_v0, 64], pl.INT32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 0)],
        q_proj_i32_inline364_inline1723__ssa_v9: pl.Tensor[[qproj_t_matmul_inline365_inline1672__ssa_v0, 32768], pl.INT32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 0)],
        wq_b_scale__ssa_v0: pl.Tensor[[32768], pl.FP32, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 131072)],
    ):
        pl.func_attr({"mx_tensor_views_blocked": True})
        self.qproj_dequant_rms_nope_rope(
            q_flat_inline2682__ssa_v0,
            tile_rows_inline366_inline1656__ssa_v0,
            tile_base_inline367_inline1657__idx_v0,
            qr_scale_pad_store_inline1684__rv_v2,
            q_rope_cos_il_inline503__ssa_v0,
            q_rope_sin_signed_inline502__ssa_v0,
            q_rope_swap_idx_inline501__ssa_v0,
            q_proj_i32_inline364_inline1723__ssa_v9,
            wq_b_scale__ssa_v0,
            attrs={"arg_directions": [pl.adir.output_existing, pl.adir.scalar, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input]},
        )

    @pl.function(type=pl.FunctionType.AIC, level=pl.Level.AIC, role=pl.Role.SubWorker)
    def qproj_matmul(
        q_proj_i32_inline364_inline1723__ssa_v0: pl.Out[pl.Tensor[[qproj_t_matmul_inline365_inline1672__ssa_v0, 32768], pl.INT32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)]],
        qproj_full_rows_inline2636__ssa_v0: pl.Scalar[pl.INDEX],
        qr_i8_matmul_inline1727__rv_v2: pl.Tensor[[512, 1024], pl.INT8, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 524288)],
        wq_b__ssa_v0: pl.Tensor[[1024, 32768], pl.INT8, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 33554432)],
        qproj_t_matmul_inline2640__ssa_v0: pl.Scalar[pl.INDEX],
        tile_rows_inline366_inline1656__ssa_v0: pl.Scalar[pl.INDEX],
    ) -> pl.Tensor[[qproj_t_matmul_inline365_inline1672__ssa_v0, 32768], pl.INT32]:
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
        qproj_worker_inline2635__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        for qproj_n_idx_inline2630__idx_v0, (q_proj_i32_inline364_inline1723__iter_v1,) in pl.range(qproj_worker_inline2635__ssa_v0, 64, 24, init_values=(q_proj_i32_inline364_inline1723__ssa_v0,)):
            w_col0_inline2637__ssa_v0: pl.Scalar[pl.INDEX] = qproj_n_idx_inline2630__idx_v0 * 512
            for t0_inline2631__idx_v0, (q_proj_i32_inline364_inline1723__iter_v3,) in pl.range(0, qproj_full_rows_inline2636__ssa_v0, 64, init_values=(q_proj_i32_inline364_inline1723__iter_v1,)):
                col_acc_inline2639__tile: pl.Tile[[64, 512], pl.INT32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 131072), pl.Mem.Acc] = pl.tile.create(
                    [64, 512], dtype=pl.INT32, target_memory=pl.Mem.Acc
                )
                for qr_proj_col0_inline2634__idx_v0, (col_acc_inline2639__iter_v1,) in pl.range(0, 1024, 512, init_values=(col_acc_inline2639__tile,)):
                    qr_i8_chunk_inline2641__tile: pl.Tile[[64, 256], pl.INT8, pl.MemRef(mem_mat_4, pl.const(0, pl.INT64), 16384), pl.Mem.Mat] = pl.tile.load(
                        qr_i8_matmul_inline1727__rv_v2, [t0_inline2631__idx_v0, qr_proj_col0_inline2634__idx_v0], [64, 256], [64, 256], target_memory=pl.Mem.Mat
                    )
                    wq_chunk_inline2638__tile: pl.Tile[[256, 512], pl.INT8, pl.MemRef(mem_mat_5, pl.const(16384, pl.INT64), 131072), pl.Mem.Mat] = pl.tile.load(
                        wq_b__ssa_v0, [qr_proj_col0_inline2634__idx_v0, w_col0_inline2637__ssa_v0], [256, 512], [256, 512], target_memory=pl.Mem.Mat
                    )
                    qr_i8_chunk_inline2641__tile_1: pl.Tile[[64, 256], pl.INT8, pl.MemRef(mem_mat_6, pl.const(147456, pl.INT64), 16384), pl.Mem.Mat] = pl.tile.load(
                        qr_i8_matmul_inline1727__rv_v2, [t0_inline2631__idx_v0, qr_proj_col0_inline2634__idx_v0 + 256], [64, 256], [64, 256], target_memory=pl.Mem.Mat
                    )
                    wq_chunk_inline2638__tile_1: pl.Tile[[256, 512], pl.INT8, pl.MemRef(mem_mat_7, pl.const(163840, pl.INT64), 131072), pl.Mem.Mat] = pl.tile.load(
                        wq_b__ssa_v0, [qr_proj_col0_inline2634__idx_v0 + 256, w_col0_inline2637__ssa_v0], [256, 512], [256, 512], target_memory=pl.Mem.Mat
                    )
                    for col_acc_inline2639__tile_l0_ko, (col_acc_inline2639__tile_l0_c,) in pl.range(0, 256, 128, init_values=(col_acc_inline2639__iter_v1,)):
                        col_acc_inline2639__tile_l0_a: pl.Tile[[64, 64], pl.INT8, pl.MemRef(mem_left_8, pl.const(12288, pl.INT64), 4096), pl.Mem.Left, pl.TileView(blayout=pl.TileLayout.row_major)] = (
                            pl.tile.extract(qr_i8_chunk_inline2641__tile, 0, col_acc_inline2639__tile_l0_ko, [64, 64], target_memory=pl.Mem.Left)
                        )
                        col_acc_inline2639__tile_l0_b: pl.Tile[[64, 512], pl.INT8, pl.MemRef(mem_right_9, pl.const(32768, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                            wq_chunk_inline2638__tile, col_acc_inline2639__tile_l0_ko, 0, [64, 512], target_memory=pl.Mem.Right
                        )
                        col_acc_inline2639__tile_l0_a_1: pl.Tile[[64, 64], pl.INT8, pl.MemRef(mem_left_10, pl.const(0, pl.INT64), 4096), pl.Mem.Left, pl.TileView(blayout=pl.TileLayout.row_major)] = (
                            pl.tile.extract(qr_i8_chunk_inline2641__tile, 0, col_acc_inline2639__tile_l0_ko + 64, [64, 64], target_memory=pl.Mem.Left)
                        )
                        col_acc_inline2639__tile_l0_b_1: pl.Tile[[64, 512], pl.INT8, pl.MemRef(mem_right_11, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                            wq_chunk_inline2638__tile, col_acc_inline2639__tile_l0_ko + 64, 0, [64, 512], target_memory=pl.Mem.Right
                        )
                        col_acc_inline2639__tile_l0_c_acc: pl.Tile[[64, 512], pl.INT32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 131072), pl.Mem.Acc] = pl.tile.matmul_acc(
                            col_acc_inline2639__tile_l0_c, col_acc_inline2639__tile_l0_a, col_acc_inline2639__tile_l0_b, qr_proj_col0_inline2634__idx_v0 == 0 and col_acc_inline2639__tile_l0_ko == 0
                        )
                        col_acc_inline2639__tile_l0_c_acc_1: pl.Tile[[64, 512], pl.INT32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 131072), pl.Mem.Acc] = pl.tile.matmul_acc(
                            col_acc_inline2639__tile_l0_c_acc,
                            col_acc_inline2639__tile_l0_a_1,
                            col_acc_inline2639__tile_l0_b_1,
                            qr_proj_col0_inline2634__idx_v0 == 0 and col_acc_inline2639__tile_l0_ko == -64,
                        )
                        col_acc_inline2639__tile_1: pl.Tile[[64, 512], pl.INT32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 131072), pl.Mem.Acc] = pl.yield_(col_acc_inline2639__tile_l0_c_acc_1)
                    for col_acc_inline2639__tile_l0_ko_1, (col_acc_inline2639__tile_l0_c_1,) in pl.range(0, 256, 128, init_values=(col_acc_inline2639__tile_1,)):
                        col_acc_inline2639__tile_l0_a_2: pl.Tile[
                            [64, 64], pl.INT8, pl.MemRef(mem_left_13, pl.const(4096, pl.INT64), 4096), pl.Mem.Left, pl.TileView(blayout=pl.TileLayout.row_major)
                        ] = pl.tile.extract(qr_i8_chunk_inline2641__tile_1, 0, col_acc_inline2639__tile_l0_ko_1, [64, 64], target_memory=pl.Mem.Left)
                        col_acc_inline2639__tile_l0_b_2: pl.Tile[[64, 512], pl.INT8, pl.MemRef(mem_right_9, pl.const(32768, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                            wq_chunk_inline2638__tile_1, col_acc_inline2639__tile_l0_ko_1, 0, [64, 512], target_memory=pl.Mem.Right
                        )
                        col_acc_inline2639__tile_l0_a_3: pl.Tile[
                            [64, 64], pl.INT8, pl.MemRef(mem_left_15, pl.const(8192, pl.INT64), 4096), pl.Mem.Left, pl.TileView(blayout=pl.TileLayout.row_major)
                        ] = pl.tile.extract(qr_i8_chunk_inline2641__tile_1, 0, col_acc_inline2639__tile_l0_ko_1 + 64, [64, 64], target_memory=pl.Mem.Left)
                        col_acc_inline2639__tile_l0_b_3: pl.Tile[[64, 512], pl.INT8, pl.MemRef(mem_right_11, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                            wq_chunk_inline2638__tile_1, col_acc_inline2639__tile_l0_ko_1 + 64, 0, [64, 512], target_memory=pl.Mem.Right
                        )
                        col_acc_inline2639__tile_l0_c_acc_2: pl.Tile[[64, 512], pl.INT32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 131072), pl.Mem.Acc] = pl.tile.matmul_acc(
                            col_acc_inline2639__tile_l0_c_1,
                            col_acc_inline2639__tile_l0_a_2,
                            col_acc_inline2639__tile_l0_b_2,
                            qr_proj_col0_inline2634__idx_v0 == -256 and col_acc_inline2639__tile_l0_ko_1 == 0,
                        )
                        col_acc_inline2639__tile_l0_c_acc_3: pl.Tile[[64, 512], pl.INT32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 131072), pl.Mem.Acc] = pl.tile.matmul_acc(
                            col_acc_inline2639__tile_l0_c_acc_2,
                            col_acc_inline2639__tile_l0_a_3,
                            col_acc_inline2639__tile_l0_b_3,
                            qr_proj_col0_inline2634__idx_v0 == -256 and col_acc_inline2639__tile_l0_ko_1 == -64,
                        )
                        col_acc_inline2639__tile_2: pl.Tile[[64, 512], pl.INT32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 131072), pl.Mem.Acc] = pl.yield_(col_acc_inline2639__tile_l0_c_acc_3)
                    col_acc_inline2639__rv_v2: pl.Tile[[64, 512], pl.INT32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 131072), pl.Mem.Acc] = pl.yield_(col_acc_inline2639__tile_2)
                q_proj_i32_inline364_inline1723__tile: pl.Tensor[[qproj_t_matmul_inline365_inline1672__ssa_v0, 32768], pl.INT32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                    col_acc_inline2639__rv_v2, [t0_inline2631__idx_v0, w_col0_inline2637__ssa_v0], q_proj_i32_inline364_inline1723__iter_v3
                )
                q_proj_i32_inline364_inline1723__rv_v4: pl.Tensor[[qproj_t_matmul_inline365_inline1672__ssa_v0, 32768], pl.INT32, pl.MemRef("mem_ddr_19", pl.const(0, pl.INT64), 0)] = pl.yield_(
                    q_proj_i32_inline364_inline1723__tile
                )
            tail_w_col0_inline2642__ssa_v0: pl.Scalar[pl.INDEX] = w_col0_inline2637__ssa_v0
            for tail_t0_inline2629__idx_v0, (q_proj_i32_inline364_inline1723__iter_v6,) in pl.range(
                qproj_full_rows_inline2636__ssa_v0, qproj_t_matmul_inline2640__ssa_v0, 16, init_values=(q_proj_i32_inline364_inline1723__rv_v4,)
            ):
                qproj_tail_rows_inline2627__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(tile_rows_inline366_inline1656__ssa_v0 - tail_t0_inline2629__idx_v0, 16)
                tail_acc_inline2628__tile: pl.Tile[[16, 512], pl.INT32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 65536), pl.Mem.Acc] = pl.tile.create(
                    [16, 512], dtype=pl.INT32, target_memory=pl.Mem.Acc
                )
                for tail_qr_col0_inline2626__idx_v0, (tail_acc_inline2628__iter_v1,) in pl.range(0, 1024, 512, init_values=(tail_acc_inline2628__tile,)):
                    qr_i8_tail_inline2625__tile: pl.Tile[
                        [16, 256], pl.INT8, pl.MemRef(mem_mat_4, pl.const(0, pl.INT64), 4096), pl.Mem.Mat, pl.TileView(valid_shape=[qproj_tail_rows_inline2627__ssa_v0, 256])
                    ] = pl.tile.load(
                        qr_i8_matmul_inline1727__rv_v2, [tail_t0_inline2629__idx_v0, tail_qr_col0_inline2626__idx_v0], [16, 256], [qproj_tail_rows_inline2627__ssa_v0, 256], target_memory=pl.Mem.Mat
                    )
                    wq_tail_inline2632__tile: pl.Tile[[256, 512], pl.INT8, pl.MemRef(mem_mat_5, pl.const(16384, pl.INT64), 131072), pl.Mem.Mat] = pl.tile.load(
                        wq_b__ssa_v0, [tail_qr_col0_inline2626__idx_v0, tail_w_col0_inline2642__ssa_v0], [256, 512], [256, 512], target_memory=pl.Mem.Mat
                    )
                    qr_i8_tail_inline2625__tile_1: pl.Tile[
                        [16, 256], pl.INT8, pl.MemRef(mem_mat_6, pl.const(147456, pl.INT64), 4096), pl.Mem.Mat, pl.TileView(valid_shape=[qproj_tail_rows_inline2627__ssa_v0, 256])
                    ] = pl.tile.load(
                        qr_i8_matmul_inline1727__rv_v2,
                        [tail_t0_inline2629__idx_v0, tail_qr_col0_inline2626__idx_v0 + 256],
                        [16, 256],
                        [qproj_tail_rows_inline2627__ssa_v0, 256],
                        target_memory=pl.Mem.Mat,
                    )
                    wq_tail_inline2632__tile_1: pl.Tile[[256, 512], pl.INT8, pl.MemRef(mem_mat_7, pl.const(163840, pl.INT64), 131072), pl.Mem.Mat] = pl.tile.load(
                        wq_b__ssa_v0, [tail_qr_col0_inline2626__idx_v0 + 256, tail_w_col0_inline2642__ssa_v0], [256, 512], [256, 512], target_memory=pl.Mem.Mat
                    )
                    for tail_acc_inline2628__tile_l0_ko, (tail_acc_inline2628__tile_l0_c,) in pl.range(0, 256, 128, init_values=(tail_acc_inline2628__iter_v1,)):
                        tail_acc_inline2628__tile_l0_a: pl.Tile[
                            [16, 64],
                            pl.INT8,
                            pl.MemRef(mem_left_8, pl.const(12288, pl.INT64), 1024),
                            pl.Mem.Left,
                            pl.TileView(valid_shape=[qproj_tail_rows_inline2627__ssa_v0, 64], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal),
                        ] = pl.tile.extract(qr_i8_tail_inline2625__tile, 0, tail_acc_inline2628__tile_l0_ko, [16, 64], target_memory=pl.Mem.Left)
                        tail_acc_inline2628__tile_l0_b: pl.Tile[[64, 512], pl.INT8, pl.MemRef(mem_right_9, pl.const(32768, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                            wq_tail_inline2632__tile, tail_acc_inline2628__tile_l0_ko, 0, [64, 512], target_memory=pl.Mem.Right
                        )
                        tail_acc_inline2628__tile_l0_a_1: pl.Tile[
                            [16, 64],
                            pl.INT8,
                            pl.MemRef(mem_left_10, pl.const(0, pl.INT64), 1024),
                            pl.Mem.Left,
                            pl.TileView(valid_shape=[qproj_tail_rows_inline2627__ssa_v0, 64], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal),
                        ] = pl.tile.extract(qr_i8_tail_inline2625__tile, 0, tail_acc_inline2628__tile_l0_ko + 64, [16, 64], target_memory=pl.Mem.Left)
                        tail_acc_inline2628__tile_l0_b_1: pl.Tile[[64, 512], pl.INT8, pl.MemRef(mem_right_11, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                            wq_tail_inline2632__tile, tail_acc_inline2628__tile_l0_ko + 64, 0, [64, 512], target_memory=pl.Mem.Right
                        )
                        tail_acc_inline2628__tile_l0_c_acc: pl.Tile[[16, 512], pl.INT32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 65536), pl.Mem.Acc] = pl.tile.matmul_acc(
                            tail_acc_inline2628__tile_l0_c,
                            tail_acc_inline2628__tile_l0_a,
                            tail_acc_inline2628__tile_l0_b,
                            tail_qr_col0_inline2626__idx_v0 == 0 and tail_acc_inline2628__tile_l0_ko == 0,
                        )
                        tail_acc_inline2628__tile_l0_c_acc_1: pl.Tile[[16, 512], pl.INT32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 65536), pl.Mem.Acc] = pl.tile.matmul_acc(
                            tail_acc_inline2628__tile_l0_c_acc,
                            tail_acc_inline2628__tile_l0_a_1,
                            tail_acc_inline2628__tile_l0_b_1,
                            tail_qr_col0_inline2626__idx_v0 == 0 and tail_acc_inline2628__tile_l0_ko == -64,
                        )
                        tail_acc_inline2628__tile_1: pl.Tile[[16, 512], pl.INT32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 65536), pl.Mem.Acc] = pl.yield_(tail_acc_inline2628__tile_l0_c_acc_1)
                    for tail_acc_inline2628__tile_l0_ko_1, (tail_acc_inline2628__tile_l0_c_1,) in pl.range(0, 256, 128, init_values=(tail_acc_inline2628__tile_1,)):
                        tail_acc_inline2628__tile_l0_a_2: pl.Tile[
                            [16, 64],
                            pl.INT8,
                            pl.MemRef(mem_left_13, pl.const(4096, pl.INT64), 1024),
                            pl.Mem.Left,
                            pl.TileView(valid_shape=[qproj_tail_rows_inline2627__ssa_v0, 64], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal),
                        ] = pl.tile.extract(qr_i8_tail_inline2625__tile_1, 0, tail_acc_inline2628__tile_l0_ko_1, [16, 64], target_memory=pl.Mem.Left)
                        tail_acc_inline2628__tile_l0_b_2: pl.Tile[[64, 512], pl.INT8, pl.MemRef(mem_right_9, pl.const(32768, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                            wq_tail_inline2632__tile_1, tail_acc_inline2628__tile_l0_ko_1, 0, [64, 512], target_memory=pl.Mem.Right
                        )
                        tail_acc_inline2628__tile_l0_a_3: pl.Tile[
                            [16, 64],
                            pl.INT8,
                            pl.MemRef(mem_left_15, pl.const(8192, pl.INT64), 1024),
                            pl.Mem.Left,
                            pl.TileView(valid_shape=[qproj_tail_rows_inline2627__ssa_v0, 64], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal),
                        ] = pl.tile.extract(qr_i8_tail_inline2625__tile_1, 0, tail_acc_inline2628__tile_l0_ko_1 + 64, [16, 64], target_memory=pl.Mem.Left)
                        tail_acc_inline2628__tile_l0_b_3: pl.Tile[[64, 512], pl.INT8, pl.MemRef(mem_right_11, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                            wq_tail_inline2632__tile_1, tail_acc_inline2628__tile_l0_ko_1 + 64, 0, [64, 512], target_memory=pl.Mem.Right
                        )
                        tail_acc_inline2628__tile_l0_c_acc_2: pl.Tile[[16, 512], pl.INT32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 65536), pl.Mem.Acc] = pl.tile.matmul_acc(
                            tail_acc_inline2628__tile_l0_c_1,
                            tail_acc_inline2628__tile_l0_a_2,
                            tail_acc_inline2628__tile_l0_b_2,
                            tail_qr_col0_inline2626__idx_v0 == -256 and tail_acc_inline2628__tile_l0_ko_1 == 0,
                        )
                        tail_acc_inline2628__tile_l0_c_acc_3: pl.Tile[[16, 512], pl.INT32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 65536), pl.Mem.Acc] = pl.tile.matmul_acc(
                            tail_acc_inline2628__tile_l0_c_acc_2,
                            tail_acc_inline2628__tile_l0_a_3,
                            tail_acc_inline2628__tile_l0_b_3,
                            tail_qr_col0_inline2626__idx_v0 == -256 and tail_acc_inline2628__tile_l0_ko_1 == -64,
                        )
                        tail_acc_inline2628__tile_2: pl.Tile[[16, 512], pl.INT32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 65536), pl.Mem.Acc] = pl.yield_(tail_acc_inline2628__tile_l0_c_acc_3)
                    tail_acc_inline2628__rv_v2: pl.Tile[[16, 512], pl.INT32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 65536), pl.Mem.Acc] = pl.yield_(tail_acc_inline2628__tile_2)
                q_proj_i32_inline364_inline1723__tile_1: pl.Tensor[[qproj_t_matmul_inline365_inline1672__ssa_v0, 32768], pl.INT32, pl.MemRef("mem_ddr_19", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                    tail_acc_inline2628__rv_v2, [tail_t0_inline2629__idx_v0, tail_w_col0_inline2642__ssa_v0], q_proj_i32_inline364_inline1723__iter_v6
                )
                q_proj_i32_inline364_inline1723__rv_v7: pl.Tensor[[qproj_t_matmul_inline365_inline1672__ssa_v0, 32768], pl.INT32, pl.MemRef("mem_ddr_36", pl.const(0, pl.INT64), 0)] = pl.yield_(
                    q_proj_i32_inline364_inline1723__tile_1
                )
            q_proj_i32_inline364_inline1723__rv_v2: pl.Tensor[[qproj_t_matmul_inline365_inline1672__ssa_v0, 32768], pl.INT32, pl.MemRef("mem_ddr_37", pl.const(0, pl.INT64), 0)] = pl.yield_(
                q_proj_i32_inline364_inline1723__rv_v7
            )
        return q_proj_i32_inline364_inline1723__ssa_v0

    @pl.function(type=pl.FunctionType.Spmd)
    def qproj_matmul_spmd(
        self,
        q_proj_i32_inline364_inline1723__ssa_v0: pl.Out[pl.Tensor[[qproj_t_matmul_inline365_inline1672__ssa_v0, 32768], pl.INT32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)]],
        qproj_full_rows_inline2636__ssa_v0: pl.Scalar[pl.INDEX],
        qr_i8_matmul_inline1727__rv_v2: pl.Tensor[[512, 1024], pl.INT8, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 524288)],
        wq_b__ssa_v0: pl.Tensor[[1024, 32768], pl.INT8, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 33554432)],
        qproj_t_matmul_inline2640__ssa_v0: pl.Scalar[pl.INDEX],
        tile_rows_inline366_inline1656__ssa_v0: pl.Scalar[pl.INDEX],
    ) -> pl.Tensor[[qproj_t_matmul_inline365_inline1672__ssa_v0, 32768], pl.INT32]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        q_proj_i32_inline364_inline1723__rv_v2: pl.Tensor[[qproj_t_matmul_inline365_inline1672__ssa_v0, 32768], pl.INT32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)] = self.qproj_matmul(
            q_proj_i32_inline364_inline1723__ssa_v0,
            qproj_full_rows_inline2636__ssa_v0,
            qr_i8_matmul_inline1727__rv_v2,
            wq_b__ssa_v0,
            qproj_t_matmul_inline2640__ssa_v0,
            tile_rows_inline366_inline1656__ssa_v0,
            attrs={"arg_directions": [pl.adir.output_existing, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.scalar, pl.adir.scalar]},
        )
        return q_proj_i32_inline364_inline1723__ssa_v0

    @pl.function(type=pl.FunctionType.AIC, level=pl.Level.AIC, role=pl.Role.SubWorker)
    def qr_hadamard_matmul(
        hadamard_idx__ssa_v0: pl.Tensor[[128, 128], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 32768)],
        qh_acc_gm_inline943_inline2148__ssa_v0: pl.Out[pl.Tensor[[bs_heads_inline945_inline2111__ssa_v0, 128], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)]],
        bs_heads_inline945_inline2111__ssa_v0: pl.Scalar[pl.INDEX],
        qr_bf16_inline2161__ssa_v0: pl.Tensor[[24576, 128], pl.BF16, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 6291456)],
    ) -> pl.Tensor[[bs_heads_inline945_inline2111__ssa_v0, 128], pl.FP32]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        mem_mat_3: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 32768)
        mem_mat_4: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 16384)
        mem_left_5: pl.Ptr = pl.tile.alloc(pl.Mem.Left, 16384)
        mem_right_6: pl.Ptr = pl.tile.alloc(pl.Mem.Right, 32768)
        mem_acc_7: pl.Ptr = pl.tile.alloc(pl.Mem.Acc, 32768)
        qh_worker_inline942_inline2156__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        qh_hadamard_inline949_inline2171__tile: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_mat_3, pl.const(0, pl.INT64), 32768), pl.Mem.Mat] = pl.tile.load(
            hadamard_idx__ssa_v0, [0, 0], [128, 128], [128, 128], target_memory=pl.Mem.Mat
        )
        for idx_inline941_inline2129__idx_v0, (qh_acc_gm_inline943_inline2148__iter_v1,) in pl.range(
            qh_worker_inline942_inline2156__ssa_v0, bs_heads_inline945_inline2111__ssa_v0 // 64, 24, init_values=(qh_acc_gm_inline943_inline2148__ssa_v0,)
        ):
            o0_inline948_inline2172__ssa_v0: pl.Scalar[pl.INDEX] = idx_inline941_inline2129__idx_v0 * 64
            t__tile: pl.Tile[[64, 128], pl.BF16, pl.MemRef(mem_mat_4, pl.const(32768, pl.INT64), 16384), pl.Mem.Mat] = pl.tile.load(
                qr_bf16_inline2161__ssa_v0, [o0_inline948_inline2172__ssa_v0, 0], [64, 128], [64, 128], target_memory=pl.Mem.Mat
            )
            t__tile_Left: pl.Tile[[64, 128], pl.BF16, pl.MemRef(mem_left_5, pl.const(0, pl.INT64), 16384), pl.Mem.Left] = pl.tile.move(t__tile, target_memory=pl.Mem.Left)
            qh_hadamard_inline949_inline2171__tile_Right: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_6, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.move(
                qh_hadamard_inline949_inline2171__tile, target_memory=pl.Mem.Right
            )
            qh_acc_inline947_inline2173__tile: pl.Tile[[64, 128], pl.FP32, pl.MemRef(mem_acc_7, pl.const(0, pl.INT64), 32768), pl.Mem.Acc] = pl.tile.matmul(
                t__tile_Left, qh_hadamard_inline949_inline2171__tile_Right
            )
            qh_acc_gm_inline943_inline2148__tile: pl.Tensor[[bs_heads_inline945_inline2111__ssa_v0, 128], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                qh_acc_inline947_inline2173__tile, [o0_inline948_inline2172__ssa_v0, 0], qh_acc_gm_inline943_inline2148__iter_v1
            )
            qh_acc_gm_inline943_inline2148__rv_v2: pl.Tensor[[bs_heads_inline945_inline2111__ssa_v0, 128], pl.FP32, pl.MemRef("mem_ddr_8", pl.const(0, pl.INT64), 0)] = pl.yield_(
                qh_acc_gm_inline943_inline2148__tile
            )
        return qh_acc_gm_inline943_inline2148__ssa_v0

    @pl.function(type=pl.FunctionType.Spmd)
    def qr_hadamard_matmul_spmd(
        self,
        hadamard_idx__ssa_v0: pl.Tensor[[128, 128], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 32768)],
        qh_acc_gm_inline943_inline2148__ssa_v0: pl.Out[pl.Tensor[[bs_heads_inline945_inline2111__ssa_v0, 128], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)]],
        bs_heads_inline945_inline2111__ssa_v0: pl.Scalar[pl.INDEX],
        qr_bf16_inline2161__ssa_v0: pl.Tensor[[24576, 128], pl.BF16, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 6291456)],
    ) -> pl.Tensor[[bs_heads_inline945_inline2111__ssa_v0, 128], pl.FP32]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        qh_acc_gm_inline943_inline2148__rv_v2: pl.Tensor[[bs_heads_inline945_inline2111__ssa_v0, 128], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)] = self.qr_hadamard_matmul(
            hadamard_idx__ssa_v0,
            qh_acc_gm_inline943_inline2148__ssa_v0,
            bs_heads_inline945_inline2111__ssa_v0,
            qr_bf16_inline2161__ssa_v0,
            attrs={"arg_directions": [pl.adir.input, pl.adir.output_existing, pl.adir.scalar, pl.adir.input]},
        )
        return qh_acc_gm_inline943_inline2148__ssa_v0

    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def qr_hadamard_quant(
        qr_hadamard_i8_inline527__ssa_v0: pl.Out[pl.Tensor[[24576, 128], pl.INT8, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 3145728)]],
        qr_hadamard_scale_dq_inline525__ssa_v0: pl.Out[pl.Tensor[[24576, 1], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 98304)]],
        bs_heads_inline945_inline2111__ssa_v0: pl.Scalar[pl.INDEX],
        qh_acc_gm_inline943_inline2148__rv_v2: pl.Tensor[[bs_heads_inline945_inline2111__ssa_v0, 128], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)],
    ) -> tuple[pl.Tensor[[24576, 128], pl.INT8], pl.Tensor[[24576, 1], pl.FP32]]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        mem_vec_3: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32768)
        mem_vec_9: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        mem_vec_12: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
        mem_vec_14: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32768)
        mem_vec_15: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        qh_quant_worker_inline938_inline2165__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        for idx_inline961_inline2153__idx_v0, (qr_hadamard_i8_inline527__iter_v1, qr_hadamard_scale_dq_inline525__iter_v1) in pl.range(
            qh_quant_worker_inline938_inline2165__ssa_v0, bs_heads_inline945_inline2111__ssa_v0 // 64, 48, init_values=(qr_hadamard_i8_inline527__ssa_v0, qr_hadamard_scale_dq_inline525__ssa_v0)
        ):
            o0_inline948_inline2172__ssa_v1: pl.Scalar[pl.INDEX] = idx_inline961_inline2153__idx_v0 * 64
            qh_full_f32_inline935_inline2149__tile: pl.Tile[[64, 128], pl.FP32, pl.MemRef(mem_vec_3, pl.const(49408, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.load(
                qh_acc_gm_inline943_inline2148__rv_v2, [o0_inline948_inline2172__ssa_v1, 0], [64, 128], [64, 128], target_memory=pl.Mem.Vec
            )
            t__tile: pl.Tile[[64, 128], pl.BF16, pl.MemRef(mem_vec_14, pl.const(16384, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.cast(
                qh_full_f32_inline935_inline2149__tile, target_type=pl.BF16, mode="rint"
            )
            qh_full_f32_v1_inline951_inline2150__tile: pl.Tile[[64, 128], pl.FP32, pl.MemRef(mem_vec_3, pl.const(49408, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.cast(
                t__tile, target_type=pl.FP32, mode="round"
            )
            t__tile_1: pl.Tile[[64, 128], pl.FP32, pl.MemRef(mem_vec_3, pl.const(49408, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.muls(qh_full_f32_v1_inline951_inline2150__tile, 0.088388347648318447)
            t__tile_2: pl.Tile[[64, 128], pl.BF16, pl.MemRef(mem_vec_14, pl.const(16384, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.cast(t__tile_1, target_type=pl.BF16, mode="rint")
            qh_full_f32_v2_inline952_inline2109__tile: pl.Tile[[64, 128], pl.FP32, pl.MemRef(mem_vec_3, pl.const(49408, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.cast(
                t__tile_2, target_type=pl.FP32, mode="round"
            )
            qh_amax_inline958_inline2108__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_9, pl.const(82176, pl.INT64), 256), pl.Mem.Vec] = pl.tile.full([1, 64], dtype=pl.FP32, value=0.0001)
            for h0_inline954_inline2116__idx_v0, (qh_amax_inline958_inline2108__iter_v1,) in pl.range(0, 128, 64, init_values=(qh_amax_inline958_inline2108__tile,)):
                qh_a_f32_inline960_inline2107__tile_textract: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_14, pl.const(16384, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.extract(
                    qh_full_f32_v2_inline952_inline2109__tile, 0, h0_inline954_inline2116__idx_v0, [64, 64], target_memory=pl.Mem.Vec
                )
                qh_a_neg_inline955_inline2120__tile: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_14, pl.const(16384, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.neg(
                    qh_a_f32_inline960_inline2107__tile_textract
                )
                qh_a_f32_inline960_inline2107__tile_textract_1: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(0, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.extract(
                    qh_full_f32_v2_inline952_inline2109__tile, 0, h0_inline954_inline2116__idx_v0, [64, 64], target_memory=pl.Mem.Vec
                )
                qh_a_abs_inline944_inline2106__tile: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(0, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.maximum(
                    qh_a_f32_inline960_inline2107__tile_textract_1, qh_a_neg_inline955_inline2120__tile
                )
                tmp_tile: pl.Tile[[64, 128], pl.FP32, pl.MemRef(mem_vec_14, pl.const(16384, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.create([64, 128], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                qh_a_max_col_inline957_inline2143__tile: pl.Tile[[64, 1], pl.FP32, pl.MemRef(mem_vec_15, pl.const(49152, pl.INT64), 256), pl.Mem.Vec] = pl.tile.row_max(
                    qh_a_abs_inline944_inline2106__tile, tmp_tile
                )
                qh_a_max_inline937_inline2105__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_15, pl.const(49152, pl.INT64), 256), pl.Mem.Vec] = pl.tile.reshape(
                    qh_a_max_col_inline957_inline2143__tile, [1, 64]
                )
                qh_amax_inline958_inline2108__tile_1: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_9, pl.const(82176, pl.INT64), 256), pl.Mem.Vec] = pl.tile.maximum(
                    qh_amax_inline958_inline2108__iter_v1, qh_a_max_inline937_inline2105__tile
                )
                qh_amax_inline958_inline2108__rv_v2: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_9, pl.const(82176, pl.INT64), 256), pl.Mem.Vec] = pl.yield_(qh_amax_inline958_inline2108__tile_1)
            qh_scale_numerator_inline946_inline2104__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_14, pl.const(16384, pl.INT64), 256), pl.Mem.Vec] = pl.tile.full(
                [1, 64], dtype=pl.FP32, value=127.0
            )
            qh_scale_quant_row_inline956_inline2103__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_14, pl.const(16384, pl.INT64), 256), pl.Mem.Vec] = pl.tile.div(
                qh_scale_numerator_inline946_inline2104__tile, qh_amax_inline958_inline2108__rv_v2
            )
            qh_scale_recip_inline953_inline2123__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(0, pl.INT64), 256), pl.Mem.Vec] = pl.tile.recip(
                qh_scale_quant_row_inline956_inline2103__tile
            )
            qh_scale_dq_inline950_inline2102__tile: pl.Tile[[64, 1], pl.FP32, pl.MemRef(mem_vec_12, pl.const(0, pl.INT64), 256), pl.Mem.Vec] = pl.tile.reshape(
                qh_scale_recip_inline953_inline2123__tile, [64, 1]
            )
            t__rm_a0_tmp_v0: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(0, pl.INT64), 256), pl.Mem.Vec] = pl.tile.reshape(qh_scale_dq_inline950_inline2102__tile, [1, 64])
            t__row_major_tmp_v1: pl.Tile[[1, 64], pl.FP16, pl.MemRef(mem_vec_9, pl.const(82176, pl.INT64), 128), pl.Mem.Vec] = pl.tile.cast(t__rm_a0_tmp_v0, target_type=pl.FP16, mode="rint")
            t__tile_3: pl.Tile[[64, 1], pl.FP16, pl.MemRef(mem_vec_9, pl.const(82176, pl.INT64), 128), pl.Mem.Vec] = pl.tile.reshape(t__row_major_tmp_v1, [64, 1])
            t__rm_a0_tmp_v2: pl.Tile[[1, 64], pl.FP16, pl.MemRef(mem_vec_9, pl.const(82176, pl.INT64), 128), pl.Mem.Vec] = pl.tile.reshape(t__tile_3, [1, 64])
            t__row_major_tmp_v3: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(0, pl.INT64), 256), pl.Mem.Vec] = pl.tile.cast(t__rm_a0_tmp_v2, target_type=pl.FP32, mode="round")
            t__tile_4: pl.Tile[[64, 1], pl.FP32, pl.MemRef(mem_vec_12, pl.const(0, pl.INT64), 256), pl.Mem.Vec] = pl.tile.reshape(t__row_major_tmp_v3, [64, 1])
            qr_hadamard_scale_dq_inline525__tile: pl.Tensor[[24576, 1], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 98304)] = pl.tile.store(
                t__tile_4, [o0_inline948_inline2172__ssa_v1, 0], qr_hadamard_scale_dq_inline525__iter_v1
            )
            qh_scale_quant_inline959_inline2101__tile: pl.Tile[[64, 1], pl.FP32, pl.MemRef(mem_vec_14, pl.const(16384, pl.INT64), 256), pl.Mem.Vec] = pl.tile.reshape(
                qh_scale_quant_row_inline956_inline2103__tile, [64, 1]
            )
            qh_q_scaled_inline962_inline2100__tile: pl.Tile[[64, 128], pl.FP32, pl.MemRef(mem_vec_3, pl.const(49408, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.row_expand_mul(
                qh_full_f32_v2_inline952_inline2109__tile, qh_scale_quant_inline959_inline2101__tile
            )
            qh_q_i32_inline934_inline2099__tile: pl.Tile[[64, 128], pl.INT32, pl.MemRef(mem_vec_3, pl.const(49408, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.cast(
                qh_q_scaled_inline962_inline2100__tile, target_type=pl.INT32, mode="rint"
            )
            qh_q_half_inline933_inline2098__tile: pl.Tile[[64, 128], pl.FP16, pl.MemRef(mem_vec_3, pl.const(49408, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.cast(
                qh_q_i32_inline934_inline2099__tile, target_type=pl.FP16, mode="round"
            )
            qh_i8_inline932_inline2096__tile: pl.Tile[[64, 128], pl.INT8, pl.MemRef(mem_vec_3, pl.const(49408, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.cast(
                qh_q_half_inline933_inline2098__tile, target_type=pl.INT8, mode="trunc"
            )
            qr_hadamard_i8_inline527__tile: pl.Tensor[[24576, 128], pl.INT8, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 3145728)] = pl.tile.store(
                qh_i8_inline932_inline2096__tile, [o0_inline948_inline2172__ssa_v1, 0], qr_hadamard_i8_inline527__iter_v1
            )
            qr_hadamard_i8_inline527__rv_v2, qr_hadamard_scale_dq_inline525__rv_v2 = pl.yield_(qr_hadamard_i8_inline527__tile, qr_hadamard_scale_dq_inline525__tile)
        return qr_hadamard_i8_inline527__ssa_v0, qr_hadamard_scale_dq_inline525__ssa_v0

    @pl.function(type=pl.FunctionType.Spmd)
    def qr_hadamard_quant_spmd(
        self,
        qr_hadamard_i8_inline527__ssa_v0: pl.Out[pl.Tensor[[24576, 128], pl.INT8, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 3145728)]],
        qr_hadamard_scale_dq_inline525__ssa_v0: pl.Out[pl.Tensor[[24576, 1], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 98304)]],
        bs_heads_inline945_inline2111__ssa_v0: pl.Scalar[pl.INDEX],
        qh_acc_gm_inline943_inline2148__rv_v2: pl.Tensor[[bs_heads_inline945_inline2111__ssa_v0, 128], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)],
    ) -> tuple[pl.Tensor[[24576, 128], pl.INT8], pl.Tensor[[24576, 1], pl.FP32]]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        ret__tmp_v0: pl.Tuple[pl.Tensor[[24576, 128], pl.INT8], pl.Tensor[[24576, 1], pl.FP32]] = self.qr_hadamard_quant(
            qr_hadamard_i8_inline527__ssa_v0,
            qr_hadamard_scale_dq_inline525__ssa_v0,
            bs_heads_inline945_inline2111__ssa_v0,
            qh_acc_gm_inline943_inline2148__rv_v2,
            attrs={"arg_directions": [pl.adir.output_existing, pl.adir.output_existing, pl.adir.scalar, pl.adir.input]},
        )
        qr_hadamard_i8_inline527__rv_v2: pl.Tensor[[24576, 128], pl.INT8, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 3145728)] = ret__tmp_v0[0]
        qr_hadamard_scale_dq_inline525__rv_v2: pl.Tensor[[24576, 1], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 98304)] = ret__tmp_v0[1]
        return qr_hadamard_i8_inline527__ssa_v0, qr_hadamard_scale_dq_inline525__ssa_v0

    @pl.function(type=pl.FunctionType.AIC, level=pl.Level.AIC, role=pl.Role.SubWorker)
    def qr_proj_matmul(
        qr_fp32_inline302_inline1687__rv_v2: pl.InOut[pl.Tensor[[qr_t_matmul_inline322_inline1689__ssa_v0, 1024], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)]],
        qr_full_rows_inline325_inline1680__ssa_v0: pl.Scalar[pl.INDEX],
        tile_base_inline319_inline1715__idx_v0: pl.Scalar[pl.INDEX],
        x_view_inline316_inline1698__ssa_v0: pl.Tensor[[t_dim_inline313_inline1671__ssa_v0, 4096], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)],
        wq_a__ssa_v0: pl.Tensor[[4096, 1024], pl.BF16, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 8388608)],
        qr_t_matmul_inline322_inline1689__ssa_v0: pl.Scalar[pl.INDEX],
        tile_rows_inline340_inline1692__ssa_v0: pl.Scalar[pl.INDEX],
    ) -> pl.Tensor[[qr_t_matmul_inline322_inline1689__ssa_v0, 1024], pl.FP32]:
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
        qbg_idx_inline328_inline1681__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        q_a_col0_inline334_inline1693__ssa_v0: pl.Scalar[pl.INDEX] = qbg_idx_inline328_inline1681__ssa_v0 // 2 * 128
        qr_k_base_inline345_inline1679__ssa_v0: pl.Scalar[pl.INDEX] = qbg_idx_inline328_inline1681__ssa_v0 % 2 * 2048
        for dense_t0_inline312_inline1688__idx_v0, (qr_fp32_inline302_inline1687__iter_v6,) in pl.range(
            0, qr_full_rows_inline325_inline1680__ssa_v0, 64, init_values=(qr_fp32_inline302_inline1687__rv_v2,)
        ):
            dense_x0_inline306_inline1683__ssa_v0: pl.Scalar[pl.INDEX] = tile_base_inline319_inline1715__idx_v0 + dense_t0_inline312_inline1688__idx_v0
            dense_acc_inline314_inline1717__tile: pl.Tile[[64, 128], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 32768), pl.Mem.Acc] = pl.tile.create(
                [64, 128], dtype=pl.FP32, target_memory=pl.Mem.Acc
            )
            for dense_k_inline311_inline1721__idx_v0, (dense_acc_inline314_inline1717__iter_v1,) in pl.range(0, 8, 2, init_values=(dense_acc_inline314_inline1717__tile,)):
                dense_d0_inline323_inline1700__ssa_v0: pl.Scalar[pl.INDEX] = qr_k_base_inline345_inline1679__ssa_v0 + dense_k_inline311_inline1721__idx_v0 * 256
                dense_d0_inline323_inline1700__ssa_v0_1: pl.Scalar[pl.INDEX] = qr_k_base_inline345_inline1679__ssa_v0 + (dense_k_inline311_inline1721__idx_v0 * 256 + 256)
                dense_x_inline317_inline1695__tile: pl.Tile[[64, 256], pl.BF16, pl.MemRef(mem_mat_4, pl.const(0, pl.INT64), 32768), pl.Mem.Mat] = pl.tile.load(
                    x_view_inline316_inline1698__ssa_v0, [dense_x0_inline306_inline1683__ssa_v0, dense_d0_inline323_inline1700__ssa_v0], [64, 256], [64, 256], target_memory=pl.Mem.Mat
                )
                dense_w_inline309_inline1673__tile: pl.Tile[[256, 128], pl.BF16, pl.MemRef(mem_mat_5, pl.const(32768, pl.INT64), 65536), pl.Mem.Mat] = pl.tile.load(
                    wq_a__ssa_v0, [dense_d0_inline323_inline1700__ssa_v0, q_a_col0_inline334_inline1693__ssa_v0], [256, 128], [256, 128], target_memory=pl.Mem.Mat
                )
                dense_x_inline317_inline1695__tile_1: pl.Tile[[64, 256], pl.BF16, pl.MemRef(mem_mat_6, pl.const(98304, pl.INT64), 32768), pl.Mem.Mat] = pl.tile.load(
                    x_view_inline316_inline1698__ssa_v0, [dense_x0_inline306_inline1683__ssa_v0, dense_d0_inline323_inline1700__ssa_v0_1], [64, 256], [64, 256], target_memory=pl.Mem.Mat
                )
                dense_w_inline309_inline1673__tile_1: pl.Tile[[256, 128], pl.BF16, pl.MemRef(mem_mat_7, pl.const(131072, pl.INT64), 65536), pl.Mem.Mat] = pl.tile.load(
                    wq_a__ssa_v0, [dense_d0_inline323_inline1700__ssa_v0_1, q_a_col0_inline334_inline1693__ssa_v0], [256, 128], [256, 128], target_memory=pl.Mem.Mat
                )
                dense_acc_inline314_inline1717__tile_l0_a: pl.Tile[
                    [64, 128], pl.BF16, pl.MemRef(mem_left_8, pl.const(49152, pl.INT64), 16384), pl.Mem.Left, pl.TileView(blayout=pl.TileLayout.row_major)
                ] = pl.tile.extract(dense_x_inline317_inline1695__tile, 0, 0, [64, 128], target_memory=pl.Mem.Left)
                dense_acc_inline314_inline1717__tile_l0_b: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_9, pl.const(32768, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                    dense_w_inline309_inline1673__tile, 0, 0, [128, 128], target_memory=pl.Mem.Right
                )
                dense_acc_inline314_inline1717__tile_l0_a_1: pl.Tile[
                    [64, 128], pl.BF16, pl.MemRef(mem_left_10, pl.const(0, pl.INT64), 16384), pl.Mem.Left, pl.TileView(blayout=pl.TileLayout.row_major)
                ] = pl.tile.extract(dense_x_inline317_inline1695__tile, 0, 128, [64, 128], target_memory=pl.Mem.Left)
                dense_acc_inline314_inline1717__tile_l0_b_1: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_11, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                    dense_w_inline309_inline1673__tile, 128, 0, [128, 128], target_memory=pl.Mem.Right
                )
                dense_acc_inline314_inline1717__tile_l0_c_acc: pl.Tile[[64, 128], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 32768), pl.Mem.Acc] = pl.tile.matmul_acc(
                    dense_acc_inline314_inline1717__iter_v1, dense_acc_inline314_inline1717__tile_l0_a, dense_acc_inline314_inline1717__tile_l0_b, dense_k_inline311_inline1721__idx_v0 == 0
                )
                dense_acc_inline314_inline1717__tile_l0_c_acc_1: pl.Tile[[64, 128], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 32768), pl.Mem.Acc] = pl.tile.matmul_acc(
                    dense_acc_inline314_inline1717__tile_l0_c_acc, dense_acc_inline314_inline1717__tile_l0_a_1, dense_acc_inline314_inline1717__tile_l0_b_1, False
                )
                dense_acc_inline314_inline1717__tile_l0_a_2: pl.Tile[
                    [64, 128], pl.BF16, pl.MemRef(mem_left_13, pl.const(16384, pl.INT64), 16384), pl.Mem.Left, pl.TileView(blayout=pl.TileLayout.row_major)
                ] = pl.tile.extract(dense_x_inline317_inline1695__tile_1, 0, 0, [64, 128], target_memory=pl.Mem.Left)
                dense_acc_inline314_inline1717__tile_l0_b_2: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_9, pl.const(32768, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                    dense_w_inline309_inline1673__tile_1, 0, 0, [128, 128], target_memory=pl.Mem.Right
                )
                dense_acc_inline314_inline1717__tile_l0_a_3: pl.Tile[
                    [64, 128], pl.BF16, pl.MemRef(mem_left_15, pl.const(32768, pl.INT64), 16384), pl.Mem.Left, pl.TileView(blayout=pl.TileLayout.row_major)
                ] = pl.tile.extract(dense_x_inline317_inline1695__tile_1, 0, 128, [64, 128], target_memory=pl.Mem.Left)
                dense_acc_inline314_inline1717__tile_l0_b_3: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_11, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                    dense_w_inline309_inline1673__tile_1, 128, 0, [128, 128], target_memory=pl.Mem.Right
                )
                dense_acc_inline314_inline1717__tile_l0_c_acc_2: pl.Tile[[64, 128], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 32768), pl.Mem.Acc] = pl.tile.matmul_acc(
                    dense_acc_inline314_inline1717__tile_l0_c_acc_1,
                    dense_acc_inline314_inline1717__tile_l0_a_2,
                    dense_acc_inline314_inline1717__tile_l0_b_2,
                    dense_k_inline311_inline1721__idx_v0 == -1,
                )
                dense_acc_inline314_inline1717__tile_l0_c_acc_3: pl.Tile[[64, 128], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 32768), pl.Mem.Acc] = pl.tile.matmul_acc(
                    dense_acc_inline314_inline1717__tile_l0_c_acc_2, dense_acc_inline314_inline1717__tile_l0_a_3, dense_acc_inline314_inline1717__tile_l0_b_3, False
                )
                dense_acc_inline314_inline1717__rv_v2: pl.Tile[[64, 128], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 32768), pl.Mem.Acc] = pl.yield_(
                    dense_acc_inline314_inline1717__tile_l0_c_acc_3
                )
            qr_fp32_inline302_inline1687__tile: pl.Tensor[[qr_t_matmul_inline322_inline1689__ssa_v0, 1024], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                dense_acc_inline314_inline1717__rv_v2, [dense_t0_inline312_inline1688__idx_v0, q_a_col0_inline334_inline1693__ssa_v0], qr_fp32_inline302_inline1687__iter_v6, atomic=pl.AtomicType.Add
            )
            qr_fp32_inline302_inline1687__rv_v7: pl.Tensor[[qr_t_matmul_inline322_inline1689__ssa_v0, 1024], pl.FP32, pl.MemRef("mem_ddr_19", pl.const(0, pl.INT64), 0)] = pl.yield_(
                qr_fp32_inline302_inline1687__tile
            )
        for t0_inline308_inline1704__idx_v0, (qr_fp32_inline302_inline1687__iter_v9,) in pl.range(
            qr_full_rows_inline325_inline1680__ssa_v0, qr_t_matmul_inline322_inline1689__ssa_v0, 16, init_values=(qr_fp32_inline302_inline1687__rv_v7,)
        ):
            q_acc_inline344_inline1690__tile: pl.Tile[[16, 128], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 8192), pl.Mem.Acc] = pl.tile.create(
                [16, 128], dtype=pl.FP32, target_memory=pl.Mem.Acc
            )
            for db_inline318_inline1691__idx_v0, (q_acc_inline344_inline1690__iter_v1,) in pl.range(0, 8, 2, init_values=(q_acc_inline344_inline1690__tile,)):
                qr_d0_inline332_inline1675__ssa_v0: pl.Scalar[pl.INDEX] = qr_k_base_inline345_inline1679__ssa_v0 + db_inline318_inline1691__idx_v0 * 256
                qr_rows_inline303_inline1670__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(tile_rows_inline340_inline1692__ssa_v0 - t0_inline308_inline1704__idx_v0, 16)
                x_t0_inline310_inline1668__ssa_v0: pl.Scalar[pl.INDEX] = tile_base_inline319_inline1715__idx_v0 + t0_inline308_inline1704__idx_v0
                qr_d0_inline332_inline1675__ssa_v0_1: pl.Scalar[pl.INDEX] = qr_k_base_inline345_inline1679__ssa_v0 + (db_inline318_inline1691__idx_v0 * 256 + 256)
                qr_rows_inline303_inline1670__ssa_v0_1: pl.Scalar[pl.INDEX] = pl.min(tile_rows_inline340_inline1692__ssa_v0 - t0_inline308_inline1704__idx_v0, 16)
                x_t0_inline310_inline1668__ssa_v0_1: pl.Scalar[pl.INDEX] = tile_base_inline319_inline1715__idx_v0 + t0_inline308_inline1704__idx_v0
                q_x_chunk_bf16_inline320_inline1710__tile: pl.Tile[
                    [16, 256], pl.BF16, pl.MemRef(mem_mat_4, pl.const(0, pl.INT64), 8192), pl.Mem.Mat, pl.TileView(valid_shape=[qr_rows_inline303_inline1670__ssa_v0, 256])
                ] = pl.tile.load(
                    x_view_inline316_inline1698__ssa_v0,
                    [x_t0_inline310_inline1668__ssa_v0, qr_d0_inline332_inline1675__ssa_v0],
                    [16, 256],
                    [qr_rows_inline303_inline1670__ssa_v0, 256],
                    target_memory=pl.Mem.Mat,
                )
                w_chunk_inline301_inline1701__tile: pl.Tile[[256, 128], pl.BF16, pl.MemRef(mem_mat_5, pl.const(32768, pl.INT64), 65536), pl.Mem.Mat] = pl.tile.load(
                    wq_a__ssa_v0, [qr_d0_inline332_inline1675__ssa_v0, q_a_col0_inline334_inline1693__ssa_v0], [256, 128], [256, 128], target_memory=pl.Mem.Mat
                )
                q_x_chunk_bf16_inline320_inline1710__tile_1: pl.Tile[
                    [16, 256], pl.BF16, pl.MemRef(mem_mat_6, pl.const(98304, pl.INT64), 8192), pl.Mem.Mat, pl.TileView(valid_shape=[qr_rows_inline303_inline1670__ssa_v0_1, 256])
                ] = pl.tile.load(
                    x_view_inline316_inline1698__ssa_v0,
                    [x_t0_inline310_inline1668__ssa_v0_1, qr_d0_inline332_inline1675__ssa_v0_1],
                    [16, 256],
                    [qr_rows_inline303_inline1670__ssa_v0_1, 256],
                    target_memory=pl.Mem.Mat,
                )
                w_chunk_inline301_inline1701__tile_1: pl.Tile[[256, 128], pl.BF16, pl.MemRef(mem_mat_7, pl.const(131072, pl.INT64), 65536), pl.Mem.Mat] = pl.tile.load(
                    wq_a__ssa_v0, [qr_d0_inline332_inline1675__ssa_v0_1, q_a_col0_inline334_inline1693__ssa_v0], [256, 128], [256, 128], target_memory=pl.Mem.Mat
                )
                q_acc_inline344_inline1690__tile_l0_a: pl.Tile[
                    [16, 128],
                    pl.BF16,
                    pl.MemRef(mem_left_8, pl.const(49152, pl.INT64), 4096),
                    pl.Mem.Left,
                    pl.TileView(valid_shape=[qr_rows_inline303_inline1670__ssa_v0, 128], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal),
                ] = pl.tile.extract(q_x_chunk_bf16_inline320_inline1710__tile, 0, 0, [16, 128], target_memory=pl.Mem.Left)
                q_acc_inline344_inline1690__tile_l0_b: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_9, pl.const(32768, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                    w_chunk_inline301_inline1701__tile, 0, 0, [128, 128], target_memory=pl.Mem.Right
                )
                q_acc_inline344_inline1690__tile_l0_a_1: pl.Tile[
                    [16, 128],
                    pl.BF16,
                    pl.MemRef(mem_left_10, pl.const(0, pl.INT64), 4096),
                    pl.Mem.Left,
                    pl.TileView(valid_shape=[qr_rows_inline303_inline1670__ssa_v0, 128], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal),
                ] = pl.tile.extract(q_x_chunk_bf16_inline320_inline1710__tile, 0, 128, [16, 128], target_memory=pl.Mem.Left)
                q_acc_inline344_inline1690__tile_l0_b_1: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_11, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                    w_chunk_inline301_inline1701__tile, 128, 0, [128, 128], target_memory=pl.Mem.Right
                )
                q_acc_inline344_inline1690__tile_l0_c_acc: pl.Tile[[16, 128], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 8192), pl.Mem.Acc] = pl.tile.matmul_acc(
                    q_acc_inline344_inline1690__iter_v1, q_acc_inline344_inline1690__tile_l0_a, q_acc_inline344_inline1690__tile_l0_b, db_inline318_inline1691__idx_v0 == 0
                )
                q_acc_inline344_inline1690__tile_l0_c_acc_1: pl.Tile[[16, 128], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 8192), pl.Mem.Acc] = pl.tile.matmul_acc(
                    q_acc_inline344_inline1690__tile_l0_c_acc, q_acc_inline344_inline1690__tile_l0_a_1, q_acc_inline344_inline1690__tile_l0_b_1, False
                )
                q_acc_inline344_inline1690__tile_l0_a_2: pl.Tile[
                    [16, 128],
                    pl.BF16,
                    pl.MemRef(mem_left_13, pl.const(16384, pl.INT64), 4096),
                    pl.Mem.Left,
                    pl.TileView(valid_shape=[qr_rows_inline303_inline1670__ssa_v0_1, 128], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal),
                ] = pl.tile.extract(q_x_chunk_bf16_inline320_inline1710__tile_1, 0, 0, [16, 128], target_memory=pl.Mem.Left)
                q_acc_inline344_inline1690__tile_l0_b_2: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_9, pl.const(32768, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                    w_chunk_inline301_inline1701__tile_1, 0, 0, [128, 128], target_memory=pl.Mem.Right
                )
                q_acc_inline344_inline1690__tile_l0_a_3: pl.Tile[
                    [16, 128],
                    pl.BF16,
                    pl.MemRef(mem_left_15, pl.const(32768, pl.INT64), 4096),
                    pl.Mem.Left,
                    pl.TileView(valid_shape=[qr_rows_inline303_inline1670__ssa_v0_1, 128], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal),
                ] = pl.tile.extract(q_x_chunk_bf16_inline320_inline1710__tile_1, 0, 128, [16, 128], target_memory=pl.Mem.Left)
                q_acc_inline344_inline1690__tile_l0_b_3: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_11, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                    w_chunk_inline301_inline1701__tile_1, 128, 0, [128, 128], target_memory=pl.Mem.Right
                )
                q_acc_inline344_inline1690__tile_l0_c_acc_2: pl.Tile[[16, 128], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 8192), pl.Mem.Acc] = pl.tile.matmul_acc(
                    q_acc_inline344_inline1690__tile_l0_c_acc_1, q_acc_inline344_inline1690__tile_l0_a_2, q_acc_inline344_inline1690__tile_l0_b_2, db_inline318_inline1691__idx_v0 == -1
                )
                q_acc_inline344_inline1690__tile_l0_c_acc_3: pl.Tile[[16, 128], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 8192), pl.Mem.Acc] = pl.tile.matmul_acc(
                    q_acc_inline344_inline1690__tile_l0_c_acc_2, q_acc_inline344_inline1690__tile_l0_a_3, q_acc_inline344_inline1690__tile_l0_b_3, False
                )
                q_acc_inline344_inline1690__rv_v2: pl.Tile[[16, 128], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 8192), pl.Mem.Acc] = pl.yield_(q_acc_inline344_inline1690__tile_l0_c_acc_3)
            qr_fp32_inline302_inline1687__tile_1: pl.Tensor[[qr_t_matmul_inline322_inline1689__ssa_v0, 1024], pl.FP32, pl.MemRef("mem_ddr_19", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                q_acc_inline344_inline1690__rv_v2, [t0_inline308_inline1704__idx_v0, q_a_col0_inline334_inline1693__ssa_v0], qr_fp32_inline302_inline1687__iter_v9, atomic=pl.AtomicType.Add
            )
            qr_fp32_inline302_inline1687__rv_v10: pl.Tensor[[qr_t_matmul_inline322_inline1689__ssa_v0, 1024], pl.FP32, pl.MemRef("mem_ddr_36", pl.const(0, pl.INT64), 0)] = pl.yield_(
                qr_fp32_inline302_inline1687__tile_1
            )
        return qr_fp32_inline302_inline1687__rv_v2

    @pl.function(type=pl.FunctionType.Spmd)
    def qr_proj_matmul_spmd(
        self,
        qr_fp32_inline302_inline1687__rv_v2: pl.InOut[pl.Tensor[[qr_t_matmul_inline322_inline1689__ssa_v0, 1024], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)]],
        qr_full_rows_inline325_inline1680__ssa_v0: pl.Scalar[pl.INDEX],
        tile_base_inline319_inline1715__idx_v0: pl.Scalar[pl.INDEX],
        x_view_inline316_inline1698__ssa_v0: pl.Tensor[[t_dim_inline313_inline1671__ssa_v0, 4096], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)],
        wq_a__ssa_v0: pl.Tensor[[4096, 1024], pl.BF16, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 8388608)],
        qr_t_matmul_inline322_inline1689__ssa_v0: pl.Scalar[pl.INDEX],
        tile_rows_inline340_inline1692__ssa_v0: pl.Scalar[pl.INDEX],
    ) -> pl.Tensor[[qr_t_matmul_inline322_inline1689__ssa_v0, 1024], pl.FP32]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        qr_fp32_inline302_inline1687__rv_v10: pl.Tensor[[qr_t_matmul_inline322_inline1689__ssa_v0, 1024], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)] = self.qr_proj_matmul(
            qr_fp32_inline302_inline1687__rv_v2,
            qr_full_rows_inline325_inline1680__ssa_v0,
            tile_base_inline319_inline1715__idx_v0,
            x_view_inline316_inline1698__ssa_v0,
            wq_a__ssa_v0,
            qr_t_matmul_inline322_inline1689__ssa_v0,
            tile_rows_inline340_inline1692__ssa_v0,
            attrs={"arg_directions": [pl.adir.inout, pl.adir.scalar, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.scalar, pl.adir.scalar]},
        )
        return qr_fp32_inline302_inline1687__rv_v2

    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def qr_proj_seed(
        qr_fp32_inline302_inline1687__ssa_v0: pl.Out[pl.Tensor[[qr_t_matmul_inline322_inline1689__ssa_v0, 1024], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)]],
        qr_t_matmul_inline322_inline1689__ssa_v0: pl.Scalar[pl.INDEX],
    ) -> pl.Tensor[[qr_t_matmul_inline322_inline1689__ssa_v0, 1024], pl.FP32]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        mem_vec_1: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 8192)
        for ts0_inline354_inline1699__idx_v0, (qr_fp32_inline302_inline1687__iter_v1,) in pl.range(
            0, qr_t_matmul_inline322_inline1689__ssa_v0, 16, init_values=(qr_fp32_inline302_inline1687__ssa_v0,)
        ):
            for nseed0_inline327_inline1686__idx_v0, (qr_fp32_inline302_inline1687__iter_v3,) in pl.range(0, 1024, 128, init_values=(qr_fp32_inline302_inline1687__iter_v1,)):
                qr_seed_inline329_inline1694__tile: pl.Tile[[16, 128], pl.FP32, pl.MemRef(mem_vec_1, pl.const(0, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.full([16, 128], dtype=pl.FP32, value=0.0)
                qr_fp32_inline302_inline1687__tile: pl.Tensor[[qr_t_matmul_inline322_inline1689__ssa_v0, 1024], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                    qr_seed_inline329_inline1694__tile, [ts0_inline354_inline1699__idx_v0, nseed0_inline327_inline1686__idx_v0], qr_fp32_inline302_inline1687__iter_v3
                )
                qr_fp32_inline302_inline1687__rv_v4: pl.Tensor[[qr_t_matmul_inline322_inline1689__ssa_v0, 1024], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)] = pl.yield_(
                    qr_fp32_inline302_inline1687__tile
                )
            qr_fp32_inline302_inline1687__rv_v2: pl.Tensor[[qr_t_matmul_inline322_inline1689__ssa_v0, 1024], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)] = pl.yield_(
                qr_fp32_inline302_inline1687__rv_v4
            )
        return qr_fp32_inline302_inline1687__ssa_v0

    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def qr_rms_norm_quant(
        tile_rows_inline340_inline1692__ssa_v0: pl.Scalar[pl.INDEX],
        tile_base_inline319_inline1715__idx_v0: pl.Scalar[pl.INDEX],
        qr_fp32_inline302_inline1687__rv_v10: pl.Tensor[[qr_t_matmul_inline322_inline1689__ssa_v0, 1024], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)],
        gamma_cq__ssa_v0: pl.Tensor[[1024], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 2048)],
        qr_scale_pad_store_inline1684__iter_v1: pl.InOut[pl.Tensor[[512, 1], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 2048)]],
        qr_scale_view_inline335_inline1703__ssa_v0: pl.Out[pl.Tensor[[t_dim_inline313_inline1671__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)]],
        qr_i8_matmul_inline1727__iter_v1: pl.InOut[pl.Tensor[[512, 1024], pl.INT8, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 524288)]],
        qr_view_inline333_inline1702__ssa_v0: pl.Out[pl.Tensor[[t_dim_inline313_inline1671__ssa_v0, 1024], pl.INT8, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 0)]],
    ) -> tuple[pl.Tensor[[512, 1], pl.FP32], pl.Tensor[[512, 1024], pl.INT8], pl.Tensor[[t_dim_inline313_inline1671__ssa_v0, 1], pl.FP32]]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        mem_vec_6: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32)
        mem_vec_7: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32)
        mem_vec_8: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 8192)
        mem_vec_9: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 512)
        mem_vec_10: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 8192)
        mem_vec_11: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 512)
        mem_vec_14: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 8192)
        mem_vec_15: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 8192)
        mem_vec_16: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32)
        mem_vec_26: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 8192)
        mem_vec_27: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 8192)
        mem_vec_28: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32)
        tg_idx_inline339_inline1696__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        tg_inline330_inline1705__ssa_v0: pl.Scalar[pl.INDEX] = tg_idx_inline339_inline1696__ssa_v0 * 8
        valid_rows_inline336_inline1706__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(tile_rows_inline340_inline1692__ssa_v0 - tg_inline330_inline1705__ssa_v0, 8)
        out_tg_inline359_inline1708__ssa_v0: pl.Scalar[pl.INDEX] = tile_base_inline319_inline1715__idx_v0 + tg_inline330_inline1705__ssa_v0
        qr_sq_sum_inline349_inline1709__tile: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_6, pl.const(41536, pl.INT64), 32), pl.Mem.Vec] = pl.tile.full([1, 8], dtype=pl.FP32, value=0.0)
        qr_amax_g_inline342_inline1711__tile: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_7, pl.const(41568, pl.INT64), 32), pl.Mem.Vec] = pl.tile.full([1, 8], dtype=pl.FP32, value=0.0)
        for qr_rms_col0_inline321_inline1677__idx_v0, (qr_amax_g_inline342_inline1711__iter_v1, qr_sq_sum_inline349_inline1709__iter_v1) in pl.range(
            0, 1024, 512, init_values=(qr_amax_g_inline342_inline1711__tile, qr_sq_sum_inline349_inline1709__tile)
        ):
            t__tile: pl.Tile[[8, 256], pl.FP32, pl.MemRef(mem_vec_8, pl.const(41600, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.load(
                qr_fp32_inline302_inline1687__rv_v10, [tg_inline330_inline1705__ssa_v0, qr_rms_col0_inline321_inline1677__idx_v0], [8, 256], [8, 256], target_memory=pl.Mem.Vec
            )
            t__tile_1: pl.Tile[[256], pl.BF16, pl.MemRef(mem_vec_9, pl.const(49792, pl.INT64), 512), pl.Mem.Vec] = pl.tile.load(
                gamma_cq__ssa_v0, [qr_rms_col0_inline321_inline1677__idx_v0], [256], [256], target_memory=pl.Mem.Vec
            )
            t__tile_2: pl.Tile[[8, 256], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.load(
                qr_fp32_inline302_inline1687__rv_v10, [tg_inline330_inline1705__ssa_v0, qr_rms_col0_inline321_inline1677__idx_v0 + 256], [8, 256], [8, 256], target_memory=pl.Mem.Vec
            )
            t__tile_3: pl.Tile[[256], pl.BF16, pl.MemRef(mem_vec_11, pl.const(8192, pl.INT64), 512), pl.Mem.Vec] = pl.tile.load(
                gamma_cq__ssa_v0, [qr_rms_col0_inline321_inline1677__idx_v0 + 256], [256], [256], target_memory=pl.Mem.Vec
            )
            t__tile_4: pl.Tile[[8, 256], pl.BF16, pl.MemRef(mem_vec_14, pl.const(8704, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(t__tile, target_type=pl.BF16, mode="rint")
            qr_rms_chunk_inline346_inline1729__tile: pl.Tile[[8, 256], pl.FP32, pl.MemRef(mem_vec_8, pl.const(41600, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.cast(
                t__tile_4, target_type=pl.FP32, mode="round"
            )
            qr_rms_sq_inline348_inline1713__tile: pl.Tile[[8, 256], pl.FP32, pl.MemRef(mem_vec_14, pl.const(8704, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.mul(
                qr_rms_chunk_inline346_inline1729__tile, qr_rms_chunk_inline346_inline1729__tile
            )
            tmp_tile: pl.Tile[[8, 256], pl.FP32, pl.MemRef(mem_vec_15, pl.const(16896, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.create([8, 256], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            t__tile_5: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_16, pl.const(25088, pl.INT64), 32), pl.Mem.Vec] = pl.tile.row_sum(qr_rms_sq_inline348_inline1713__tile, tmp_tile)
            qr_rms_row_sum_inline304_inline1714__tile: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_16, pl.const(25088, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(t__tile_5, [1, 8])
            qr_sq_sum_inline349_inline1709__tile_1: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_15, pl.const(16896, pl.INT64), 32), pl.Mem.Vec] = pl.tile.add(
                qr_sq_sum_inline349_inline1709__iter_v1, qr_rms_row_sum_inline304_inline1714__tile
            )
            gamma_rms_cast_inline341_inline1716__tile: pl.Tile[[256], pl.FP32, pl.MemRef(mem_vec_14, pl.const(8704, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.cast(
                t__tile_1, target_type=pl.FP32, mode="round"
            )
            gamma_rms_chunk_inline337_inline1718__tile: pl.Tile[[1, 256], pl.FP32, pl.MemRef(mem_vec_14, pl.const(8704, pl.INT64), 1024), pl.Mem.Vec] = gamma_rms_cast_inline341_inline1716__tile
            qr_g_inline351_inline1719__tile: pl.Tile[[8, 256], pl.FP32, pl.MemRef(mem_vec_8, pl.const(41600, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.col_expand_mul(
                qr_rms_chunk_inline346_inline1729__tile, gamma_rms_chunk_inline337_inline1718__tile
            )
            qr_g_abs_inline307_inline1720__tile: pl.Tile[[8, 256], pl.FP32, pl.MemRef(mem_vec_8, pl.const(41600, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.abs(qr_g_inline351_inline1719__tile)
            tmp_tile_1: pl.Tile[[8, 256], pl.FP32, pl.MemRef(mem_vec_14, pl.const(8704, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.create([8, 256], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            t__tile_6: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_9, pl.const(49792, pl.INT64), 32), pl.Mem.Vec] = pl.tile.row_max(qr_g_abs_inline307_inline1720__tile, tmp_tile_1)
            qr_g_row_max_inline353_inline1674__tile: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_9, pl.const(49792, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(t__tile_6, [1, 8])
            qr_amax_g_inline342_inline1711__tile_1: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_8, pl.const(41600, pl.INT64), 32), pl.Mem.Vec] = pl.tile.maximum(
                qr_amax_g_inline342_inline1711__iter_v1, qr_g_row_max_inline353_inline1674__tile
            )
            t__tile_7: pl.Tile[[8, 256], pl.BF16, pl.MemRef(mem_vec_26, pl.const(25120, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(t__tile_2, target_type=pl.BF16, mode="rint")
            qr_rms_chunk_inline346_inline1729__tile_1: pl.Tile[[8, 256], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.cast(
                t__tile_7, target_type=pl.FP32, mode="round"
            )
            qr_rms_sq_inline348_inline1713__tile_1: pl.Tile[[8, 256], pl.FP32, pl.MemRef(mem_vec_26, pl.const(25120, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.mul(
                qr_rms_chunk_inline346_inline1729__tile_1, qr_rms_chunk_inline346_inline1729__tile_1
            )
            tmp_tile_2: pl.Tile[[8, 256], pl.FP32, pl.MemRef(mem_vec_27, pl.const(33312, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.create([8, 256], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            t__tile_8: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_28, pl.const(41504, pl.INT64), 32), pl.Mem.Vec] = pl.tile.row_sum(qr_rms_sq_inline348_inline1713__tile_1, tmp_tile_2)
            qr_rms_row_sum_inline304_inline1714__tile_1: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_28, pl.const(41504, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(t__tile_8, [1, 8])
            qr_sq_sum_inline349_inline1709__tile_2: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_6, pl.const(41536, pl.INT64), 32), pl.Mem.Vec] = pl.tile.add(
                qr_sq_sum_inline349_inline1709__tile_1, qr_rms_row_sum_inline304_inline1714__tile_1
            )
            gamma_rms_cast_inline341_inline1716__tile_1: pl.Tile[[256], pl.FP32, pl.MemRef(mem_vec_26, pl.const(25120, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.cast(
                t__tile_3, target_type=pl.FP32, mode="round"
            )
            gamma_rms_chunk_inline337_inline1718__tile_1: pl.Tile[[1, 256], pl.FP32, pl.MemRef(mem_vec_26, pl.const(25120, pl.INT64), 1024), pl.Mem.Vec] = gamma_rms_cast_inline341_inline1716__tile_1
            qr_g_inline351_inline1719__tile_1: pl.Tile[[8, 256], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.col_expand_mul(
                qr_rms_chunk_inline346_inline1729__tile_1, gamma_rms_chunk_inline337_inline1718__tile_1
            )
            qr_g_abs_inline307_inline1720__tile_1: pl.Tile[[8, 256], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.abs(qr_g_inline351_inline1719__tile_1)
            tmp_tile_3: pl.Tile[[8, 256], pl.FP32, pl.MemRef(mem_vec_26, pl.const(25120, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.create([8, 256], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            t__tile_9: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_27, pl.const(33312, pl.INT64), 32), pl.Mem.Vec] = pl.tile.row_max(qr_g_abs_inline307_inline1720__tile_1, tmp_tile_3)
            qr_g_row_max_inline353_inline1674__tile_1: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_27, pl.const(33312, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(t__tile_9, [1, 8])
            qr_amax_g_inline342_inline1711__tile_2: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_7, pl.const(41568, pl.INT64), 32), pl.Mem.Vec] = pl.tile.maximum(
                qr_amax_g_inline342_inline1711__tile_1, qr_g_row_max_inline353_inline1674__tile_1
            )
            qr_amax_g_inline342_inline1711__rv_v2, qr_sq_sum_inline349_inline1709__rv_v2 = pl.yield_(qr_amax_g_inline342_inline1711__tile_2, qr_sq_sum_inline349_inline1709__tile_2)
        t__tile_10: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_8, pl.const(41600, pl.INT64), 32), pl.Mem.Vec] = pl.tile.muls(qr_sq_sum_inline349_inline1709__rv_v2, 0.0009765625)
        t__tile_11: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_8, pl.const(41600, pl.INT64), 32), pl.Mem.Vec] = pl.tile.adds(t__tile_10, 9.9999999999999995e-07)
        rsqrt_tmp: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 32), pl.Mem.Vec] = pl.tile.create([1, 8], dtype=pl.FP32, target_memory=pl.Mem.Vec)
        qr_inv_rms_inline315_inline1707__tile: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_9, pl.const(49792, pl.INT64), 32), pl.Mem.Vec] = pl.tile.rsqrt(t__tile_11, rsqrt_tmp)
        qr_inv_rms_t_inline352_inline1678__tile: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_9, pl.const(49792, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(
            qr_inv_rms_inline315_inline1707__tile, [8, 1]
        )
        qr_amax_floor_inline355_inline1722__tile: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_8, pl.const(41600, pl.INT64), 32), pl.Mem.Vec] = pl.tile.full([1, 8], dtype=pl.FP32, value=0.0001)
        qr_amax_normed_inline357_inline1724__tile: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 32), pl.Mem.Vec] = pl.tile.mul(
            qr_inv_rms_inline315_inline1707__tile, qr_amax_g_inline342_inline1711__rv_v2
        )
        qr_tile_amax_inline360_inline1725__tile: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_8, pl.const(41600, pl.INT64), 32), pl.Mem.Vec] = pl.tile.maximum(
            qr_amax_floor_inline355_inline1722__tile, qr_amax_normed_inline357_inline1724__tile
        )
        t__tile_12: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 32), pl.Mem.Vec] = pl.tile.full([1, 8], dtype=pl.FP32, value=127.0)
        qr_scale_quant_row_inline356_inline1728__tile: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_11, pl.const(8192, pl.INT64), 32), pl.Mem.Vec] = pl.tile.div(
            t__tile_12, qr_tile_amax_inline360_inline1725__tile
        )
        qr_scale_quant_t_inline350_inline1726__tile: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_11, pl.const(8192, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(
            qr_scale_quant_row_inline356_inline1728__tile, [8, 1]
        )
        t__tile_13: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_8, pl.const(41600, pl.INT64), 32), pl.Mem.Vec] = pl.tile.recip(qr_scale_quant_row_inline356_inline1728__tile)
        qr_tile_scale_dq_inline326_inline1712__tile: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_8, pl.const(41600, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(t__tile_13, [8, 1])
        qr_scale_pad_store_inline1684__tile: pl.Tensor[[512, 1], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 2048)] = pl.tile.store(
            qr_tile_scale_dq_inline326_inline1712__tile, [tg_inline330_inline1705__ssa_v0, 0], qr_scale_pad_store_inline1684__iter_v1
        )
        if valid_rows_inline336_inline1706__ssa_v0 == 8:
            qr_scale_view_inline335_inline1703__tile: pl.Tensor[[t_dim_inline313_inline1671__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                qr_tile_scale_dq_inline326_inline1712__tile, [out_tg_inline359_inline1708__ssa_v0, 0], qr_scale_view_inline335_inline1703__ssa_v0
            )
        else:
            qr_scale_tail_inline343_inline1682__ssa_v0: pl.Tile[
                [8, 1], pl.FP32, pl.MemRef(mem_vec_8, pl.const(41600, pl.INT64), 32), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline336_inline1706__ssa_v0, 1])
            ] = pl.tile.load(qr_scale_pad_store_inline1684__tile, [tg_inline330_inline1705__ssa_v0, 0], [8, 1], [valid_rows_inline336_inline1706__ssa_v0, 1], target_memory=pl.Mem.Vec)
            qr_scale_view_inline335_inline1703__store: pl.Tensor[[t_dim_inline313_inline1671__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                qr_scale_tail_inline343_inline1682__ssa_v0, [out_tg_inline359_inline1708__ssa_v0, 0], qr_scale_view_inline335_inline1703__ssa_v0
            )
        for qa_inline324_inline1730__idx_v0, (qr_i8_matmul_inline1727__iter_v3, qr_view_inline333_inline1702__iter_v1) in pl.range(
            0, 1024, 512, init_values=(qr_i8_matmul_inline1727__iter_v1, qr_view_inline333_inline1702__ssa_v0)
        ):
            t__tile_14: pl.Tile[[8, 256], pl.FP32, pl.MemRef(mem_vec_8, pl.const(41600, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.load(
                qr_fp32_inline302_inline1687__rv_v10, [tg_inline330_inline1705__ssa_v0, qa_inline324_inline1730__idx_v0], [8, 256], [8, 256], target_memory=pl.Mem.Vec
            )
            t__tile_15: pl.Tile[[256], pl.BF16, pl.MemRef(mem_vec_26, pl.const(25120, pl.INT64), 512), pl.Mem.Vec] = pl.tile.load(
                gamma_cq__ssa_v0, [qa_inline324_inline1730__idx_v0], [256], [256], target_memory=pl.Mem.Vec
            )
            t__tile_16: pl.Tile[[8, 256], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.load(
                qr_fp32_inline302_inline1687__rv_v10, [tg_inline330_inline1705__ssa_v0, qa_inline324_inline1730__idx_v0 + 256], [8, 256], [8, 256], target_memory=pl.Mem.Vec
            )
            t__tile_17: pl.Tile[[256], pl.BF16, pl.MemRef(mem_vec_27, pl.const(33312, pl.INT64), 512), pl.Mem.Vec] = pl.tile.load(
                gamma_cq__ssa_v0, [qa_inline324_inline1730__idx_v0 + 256], [256], [256], target_memory=pl.Mem.Vec
            )
            t__tile_18: pl.Tile[[8, 256], pl.BF16, pl.MemRef(mem_vec_14, pl.const(8704, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(t__tile_14, target_type=pl.BF16, mode="rint")
            qr_chunk_inline361_inline1667__tile: pl.Tile[[8, 256], pl.FP32, pl.MemRef(mem_vec_8, pl.const(41600, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.cast(
                t__tile_18, target_type=pl.FP32, mode="round"
            )
            gamma_q_cast_inline358_inline1666__tile: pl.Tile[[256], pl.FP32, pl.MemRef(mem_vec_14, pl.const(8704, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.cast(
                t__tile_15, target_type=pl.FP32, mode="round"
            )
            gamma_q_chunk_inline305_inline1665__tile: pl.Tile[[1, 256], pl.FP32, pl.MemRef(mem_vec_14, pl.const(8704, pl.INT64), 1024), pl.Mem.Vec] = gamma_q_cast_inline358_inline1666__tile
            t__tile_19: pl.Tile[[8, 256], pl.FP32, pl.MemRef(mem_vec_8, pl.const(41600, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.row_expand_mul(
                qr_chunk_inline361_inline1667__tile, qr_inv_rms_t_inline352_inline1678__tile
            )
            qr_q_normed_inline300_inline1676__tile: pl.Tile[[8, 256], pl.FP32, pl.MemRef(mem_vec_8, pl.const(41600, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.col_expand_mul(
                t__tile_19, gamma_q_chunk_inline305_inline1665__tile
            )
            qr_q_scaled_inline299_inline1664__tile: pl.Tile[[8, 256], pl.FP32, pl.MemRef(mem_vec_8, pl.const(41600, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.row_expand_mul(
                qr_q_normed_inline300_inline1676__tile, qr_scale_quant_t_inline350_inline1726__tile
            )
            qr_q_i32_inline297_inline1663__tile: pl.Tile[[8, 256], pl.INT32, pl.MemRef(mem_vec_8, pl.const(41600, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.cast(
                qr_q_scaled_inline299_inline1664__tile, target_type=pl.INT32, mode="rint"
            )
            qr_q_half_inline298_inline1662__tile: pl.Tile[[8, 256], pl.FP16, pl.MemRef(mem_vec_8, pl.const(41600, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(
                qr_q_i32_inline297_inline1663__tile, target_type=pl.FP16, mode="round"
            )
            qr_q_i8_inline347_inline1661__tile: pl.Tile[[8, 256], pl.INT8, pl.MemRef(mem_vec_8, pl.const(41600, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(
                qr_q_half_inline298_inline1662__tile, target_type=pl.INT8, mode="trunc"
            )
            qr_i8_matmul_inline1727__tile: pl.Tensor[[512, 1024], pl.INT8, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 524288)] = pl.tile.store(
                qr_q_i8_inline347_inline1661__tile, [tg_inline330_inline1705__ssa_v0, qa_inline324_inline1730__idx_v0], qr_i8_matmul_inline1727__iter_v3
            )
            if valid_rows_inline336_inline1706__ssa_v0 == 8:
                qr_view_inline333_inline1702__tile: pl.Tensor[[t_dim_inline313_inline1671__ssa_v0, 1024], pl.INT8, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                    qr_q_i8_inline347_inline1661__tile, [out_tg_inline359_inline1708__ssa_v0, qa_inline324_inline1730__idx_v0], qr_view_inline333_inline1702__iter_v1
                )
                qr_view_inline333_inline1702__phi_v4: pl.Tensor[[t_dim_inline313_inline1671__ssa_v0, 1024], pl.INT8, pl.MemRef("mem_ddr_63", pl.const(0, pl.INT64), 0)] = pl.yield_(
                    qr_view_inline333_inline1702__tile
                )
            else:
                qr_q_tail_inline296_inline1660__ssa_v0: pl.Tile[
                    [8, 256], pl.INT8, pl.MemRef(mem_vec_8, pl.const(41600, pl.INT64), 2048), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline336_inline1706__ssa_v0, 256])
                ] = pl.tile.load(
                    qr_i8_matmul_inline1727__tile,
                    [tg_inline330_inline1705__ssa_v0, qa_inline324_inline1730__idx_v0],
                    [8, 256],
                    [valid_rows_inline336_inline1706__ssa_v0, 256],
                    target_memory=pl.Mem.Vec,
                )
                pl.tile.store(qr_q_tail_inline296_inline1660__ssa_v0, [out_tg_inline359_inline1708__ssa_v0, qa_inline324_inline1730__idx_v0], qr_view_inline333_inline1702__iter_v1)
                qr_view_inline333_inline1702__phi_v4: pl.Tensor[[t_dim_inline313_inline1671__ssa_v0, 1024], pl.INT8, pl.MemRef("mem_ddr_63", pl.const(0, pl.INT64), 0)] = pl.yield_(
                    qr_view_inline333_inline1702__iter_v1
                )
            t__tile_20: pl.Tile[[8, 256], pl.BF16, pl.MemRef(mem_vec_15, pl.const(16896, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(t__tile_16, target_type=pl.BF16, mode="rint")
            qr_chunk_inline361_inline1667__tile_1: pl.Tile[[8, 256], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.cast(
                t__tile_20, target_type=pl.FP32, mode="round"
            )
            gamma_q_cast_inline358_inline1666__tile_1: pl.Tile[[256], pl.FP32, pl.MemRef(mem_vec_15, pl.const(16896, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.cast(
                t__tile_17, target_type=pl.FP32, mode="round"
            )
            gamma_q_chunk_inline305_inline1665__tile_1: pl.Tile[[1, 256], pl.FP32, pl.MemRef(mem_vec_15, pl.const(16896, pl.INT64), 1024), pl.Mem.Vec] = gamma_q_cast_inline358_inline1666__tile_1
            t__tile_21: pl.Tile[[8, 256], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.row_expand_mul(
                qr_chunk_inline361_inline1667__tile_1, qr_inv_rms_t_inline352_inline1678__tile
            )
            qr_q_normed_inline300_inline1676__tile_1: pl.Tile[[8, 256], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.col_expand_mul(
                t__tile_21, gamma_q_chunk_inline305_inline1665__tile_1
            )
            qr_q_scaled_inline299_inline1664__tile_1: pl.Tile[[8, 256], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.row_expand_mul(
                qr_q_normed_inline300_inline1676__tile_1, qr_scale_quant_t_inline350_inline1726__tile
            )
            qr_q_i32_inline297_inline1663__tile_1: pl.Tile[[8, 256], pl.INT32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.cast(
                qr_q_scaled_inline299_inline1664__tile_1, target_type=pl.INT32, mode="rint"
            )
            qr_q_half_inline298_inline1662__tile_1: pl.Tile[[8, 256], pl.FP16, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(
                qr_q_i32_inline297_inline1663__tile_1, target_type=pl.FP16, mode="round"
            )
            qr_q_i8_inline347_inline1661__tile_1: pl.Tile[[8, 256], pl.INT8, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(
                qr_q_half_inline298_inline1662__tile_1, target_type=pl.INT8, mode="trunc"
            )
            qr_i8_matmul_inline1727__tile_1: pl.Tensor[[512, 1024], pl.INT8, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 524288)] = pl.tile.store(
                qr_q_i8_inline347_inline1661__tile_1, [tg_inline330_inline1705__ssa_v0, qa_inline324_inline1730__idx_v0 + 256], qr_i8_matmul_inline1727__tile
            )
            if valid_rows_inline336_inline1706__ssa_v0 == 8:
                qr_view_inline333_inline1702__tile_1: pl.Tensor[[t_dim_inline313_inline1671__ssa_v0, 1024], pl.INT8, pl.MemRef("mem_ddr_63", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                    qr_q_i8_inline347_inline1661__tile_1, [out_tg_inline359_inline1708__ssa_v0, qa_inline324_inline1730__idx_v0 + 256], qr_view_inline333_inline1702__phi_v4
                )
                qr_view_inline333_inline1702__phi_v4_1: pl.Tensor[[t_dim_inline313_inline1671__ssa_v0, 1024], pl.INT8, pl.MemRef("mem_ddr_74", pl.const(0, pl.INT64), 0)] = pl.yield_(
                    qr_view_inline333_inline1702__tile_1
                )
            else:
                qr_q_tail_inline296_inline1660__ssa_v0_1: pl.Tile[
                    [8, 256], pl.INT8, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 2048), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline336_inline1706__ssa_v0, 256])
                ] = pl.tile.load(
                    qr_i8_matmul_inline1727__tile_1,
                    [tg_inline330_inline1705__ssa_v0, qa_inline324_inline1730__idx_v0 + 256],
                    [8, 256],
                    [valid_rows_inline336_inline1706__ssa_v0, 256],
                    target_memory=pl.Mem.Vec,
                )
                pl.tile.store(qr_q_tail_inline296_inline1660__ssa_v0_1, [out_tg_inline359_inline1708__ssa_v0, qa_inline324_inline1730__idx_v0 + 256], qr_view_inline333_inline1702__phi_v4)
                qr_view_inline333_inline1702__phi_v4_1: pl.Tensor[[t_dim_inline313_inline1671__ssa_v0, 1024], pl.INT8, pl.MemRef("mem_ddr_74", pl.const(0, pl.INT64), 0)] = pl.yield_(
                    qr_view_inline333_inline1702__phi_v4
                )
            qr_i8_matmul_inline1727__rv_v4, qr_view_inline333_inline1702__rv_v2 = pl.yield_(qr_i8_matmul_inline1727__tile_1, qr_view_inline333_inline1702__phi_v4_1)
        return qr_scale_pad_store_inline1684__iter_v1, qr_i8_matmul_inline1727__iter_v1, qr_scale_view_inline335_inline1703__ssa_v0

    @pl.function(type=pl.FunctionType.Spmd)
    def qr_rms_norm_quant_spmd(
        self,
        tile_rows_inline340_inline1692__ssa_v0: pl.Scalar[pl.INDEX],
        tile_base_inline319_inline1715__idx_v0: pl.Scalar[pl.INDEX],
        qr_fp32_inline302_inline1687__rv_v10: pl.Tensor[[qr_t_matmul_inline322_inline1689__ssa_v0, 1024], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)],
        gamma_cq__ssa_v0: pl.Tensor[[1024], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 2048)],
        qr_scale_pad_store_inline1684__iter_v1: pl.InOut[pl.Tensor[[512, 1], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 2048)]],
        qr_scale_view_inline335_inline1703__ssa_v0: pl.Out[pl.Tensor[[t_dim_inline313_inline1671__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)]],
        qr_i8_matmul_inline1727__iter_v1: pl.InOut[pl.Tensor[[512, 1024], pl.INT8, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 524288)]],
        qr_view_inline333_inline1702__ssa_v0: pl.Out[pl.Tensor[[t_dim_inline313_inline1671__ssa_v0, 1024], pl.INT8, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 0)]],
    ) -> tuple[pl.Tensor[[512, 1], pl.FP32], pl.Tensor[[512, 1024], pl.INT8]]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        ret__tmp_v0: pl.Tuple[pl.Tensor[[512, 1], pl.FP32], pl.Tensor[[512, 1024], pl.INT8], pl.Tensor[[t_dim_inline313_inline1671__ssa_v0, 1], pl.FP32]] = self.qr_rms_norm_quant(
            tile_rows_inline340_inline1692__ssa_v0,
            tile_base_inline319_inline1715__idx_v0,
            qr_fp32_inline302_inline1687__rv_v10,
            gamma_cq__ssa_v0,
            qr_scale_pad_store_inline1684__iter_v1,
            qr_scale_view_inline335_inline1703__ssa_v0,
            qr_i8_matmul_inline1727__iter_v1,
            qr_view_inline333_inline1702__ssa_v0,
            attrs={"arg_directions": [pl.adir.scalar, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.inout, pl.adir.output_existing, pl.adir.inout, pl.adir.output_existing]},
        )
        qr_scale_pad_store_inline1684__ssa_v3: pl.Tensor[[512, 1], pl.FP32, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 2048)] = ret__tmp_v0[0]
        qr_i8_matmul_inline1727__rv_v4: pl.Tensor[[512, 1024], pl.INT8, pl.MemRef("mem_ddr_7", pl.const(0, pl.INT64), 524288)] = ret__tmp_v0[1]
        qr_scale_view_inline335_inline1703__ssa_v2: pl.Tensor[[t_dim_inline313_inline1671__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_8", pl.const(0, pl.INT64), 0)] = ret__tmp_v0[2]
        return qr_scale_pad_store_inline1684__iter_v1, qr_i8_matmul_inline1727__iter_v1

    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def quant(
        act_scale_dq_inline598__iter_v1: pl.Out[pl.Tensor[[8, 384], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 12288)]],
        o_r_i8_pad_inline621__iter_v1: pl.Out[pl.Tensor[[384, 8192], pl.INT8, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 3145728)]],
        t_dim_inline607__ssa_v0: pl.Scalar[pl.INDEX],
        o_r_pad_inline620__ssa_v3: pl.Tensor[[384, 8192], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 12582912)],
        col_g_inline610__ssa_v0: pl.Scalar[pl.INDEX],
        g_inline628__idx_v0: pl.Scalar[pl.INDEX],
        proj_b_padded_rows_inline614__ssa_v0: pl.Scalar[pl.INDEX],
    ) -> tuple[pl.Tensor[[8, 384], pl.FP32], pl.Tensor[[384, 8192], pl.INT8]]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        mem_vec_3: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32768)
        mem_vec_4: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32768)
        mem_vec_5: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32768)
        mem_vec_6: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32768)
        mem_vec_8: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32768)
        mem_vec_9: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32)
        unroll_main_end: pl.Scalar[pl.INDEX] = (t_dim_inline607__ssa_v0 + 7) // 16 * 16
        for qt_inline590__idx_v0, (act_scale_dq_inline598__iter_v3, o_r_i8_pad_inline621__iter_v3) in pl.range(
            0, unroll_main_end, 16, init_values=(act_scale_dq_inline598__iter_v1, o_r_i8_pad_inline621__iter_v1)
        ):
            oc_amax_inline650__tile: pl.Tile[[8, 1024], pl.FP32, pl.MemRef(mem_vec_3, pl.const(0, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.load(
                o_r_pad_inline620__ssa_v3, [qt_inline590__idx_v0, col_g_inline610__ssa_v0], [8, 1024], [8, 1024], target_memory=pl.Mem.Vec
            )
            oc_q_inline594__tile: pl.Tile[[8, 1024], pl.FP32, pl.MemRef(mem_vec_4, pl.const(32768, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.load(
                o_r_pad_inline620__ssa_v3, [qt_inline590__idx_v0, col_g_inline610__ssa_v0], [8, 1024], [8, 1024], target_memory=pl.Mem.Vec
            )
            oc_amax_inline650__tile_1: pl.Tile[[8, 1024], pl.FP32, pl.MemRef(mem_vec_5, pl.const(65536, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.load(
                o_r_pad_inline620__ssa_v3, [qt_inline590__idx_v0 + 8, col_g_inline610__ssa_v0], [8, 1024], [8, 1024], target_memory=pl.Mem.Vec
            )
            oc_q_inline594__tile_1: pl.Tile[[8, 1024], pl.FP32, pl.MemRef(mem_vec_6, pl.const(98304, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.load(
                o_r_pad_inline620__ssa_v3, [qt_inline590__idx_v0 + 8, col_g_inline610__ssa_v0], [8, 1024], [8, 1024], target_memory=pl.Mem.Vec
            )
            g_abs_inline640__tile: pl.Tile[[8, 1024], pl.FP32, pl.MemRef(mem_vec_3, pl.const(0, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.abs(oc_amax_inline650__tile)
            tmp_tile: pl.Tile[[8, 1024], pl.FP32, pl.MemRef(mem_vec_8, pl.const(131072, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.create([8, 1024], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            g_row_max_inline601__tile: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_9, pl.const(163840, pl.INT64), 32), pl.Mem.Vec] = pl.tile.row_max(g_abs_inline640__tile, tmp_tile)
            g_row_max_v1_inline619__tile: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_9, pl.const(163840, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(g_row_max_inline601__tile, [1, 8])
            g_amax_floor_inline641__tile: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_3, pl.const(0, pl.INT64), 32), pl.Mem.Vec] = pl.tile.full([1, 8], dtype=pl.FP32, value=0.0001)
            g_amax_inline642__tile: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_3, pl.const(0, pl.INT64), 32), pl.Mem.Vec] = pl.tile.maximum(g_amax_floor_inline641__tile, g_row_max_v1_inline619__tile)
            g_scale_num_inline630__tile: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_8, pl.const(131072, pl.INT64), 32), pl.Mem.Vec] = pl.tile.full([1, 8], dtype=pl.FP32, value=127.0)
            g_sq_row_inline646__tile: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_8, pl.const(131072, pl.INT64), 32), pl.Mem.Vec] = pl.tile.div(g_scale_num_inline630__tile, g_amax_inline642__tile)
            g_scale_dq_inline647__tile: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_9, pl.const(163840, pl.INT64), 32), pl.Mem.Vec] = pl.tile.muls(g_amax_inline642__tile, 0.007874015748031496)
            g_sq_col_inline612__tile: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_8, pl.const(131072, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(g_sq_row_inline646__tile, [8, 1])
            oq_scaled_inline651__tile: pl.Tile[[8, 1024], pl.FP32, pl.MemRef(mem_vec_3, pl.const(0, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.row_expand_mul(
                oc_q_inline594__tile, g_sq_col_inline612__tile
            )
            oq_i32_inline617__tile: pl.Tile[[8, 1024], pl.INT32, pl.MemRef(mem_vec_3, pl.const(0, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.cast(
                oq_scaled_inline651__tile, target_type=pl.INT32, mode="rint"
            )
            oq_half_inline593__tile: pl.Tile[[8, 1024], pl.FP16, pl.MemRef(mem_vec_3, pl.const(0, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.cast(
                oq_i32_inline617__tile, target_type=pl.FP16, mode="round"
            )
            oq_i8_inline649__tile: pl.Tile[[8, 1024], pl.INT8, pl.MemRef(mem_vec_3, pl.const(0, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.cast(oq_half_inline593__tile, target_type=pl.INT8, mode="trunc")
            act_scale_dq_inline598__tile: pl.Tensor[[8, 384], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 12288)] = pl.tile.store(
                g_scale_dq_inline647__tile, [g_inline628__idx_v0, qt_inline590__idx_v0], act_scale_dq_inline598__iter_v3
            )
            o_r_i8_pad_inline621__tile: pl.Tensor[[384, 8192], pl.INT8, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 3145728)] = pl.tile.store(
                oq_i8_inline649__tile, [qt_inline590__idx_v0, col_g_inline610__ssa_v0], o_r_i8_pad_inline621__iter_v3
            )
            g_abs_inline640__tile_1: pl.Tile[[8, 1024], pl.FP32, pl.MemRef(mem_vec_3, pl.const(0, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.abs(oc_amax_inline650__tile_1)
            tmp_tile_1: pl.Tile[[8, 1024], pl.FP32, pl.MemRef(mem_vec_4, pl.const(32768, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.create([8, 1024], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            g_row_max_inline601__tile_1: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_5, pl.const(65536, pl.INT64), 32), pl.Mem.Vec] = pl.tile.row_max(g_abs_inline640__tile_1, tmp_tile_1)
            g_row_max_v1_inline619__tile_1: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_5, pl.const(65536, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(g_row_max_inline601__tile_1, [1, 8])
            g_amax_floor_inline641__tile_1: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_3, pl.const(0, pl.INT64), 32), pl.Mem.Vec] = pl.tile.full([1, 8], dtype=pl.FP32, value=0.0001)
            g_amax_inline642__tile_1: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_3, pl.const(0, pl.INT64), 32), pl.Mem.Vec] = pl.tile.maximum(
                g_amax_floor_inline641__tile_1, g_row_max_v1_inline619__tile_1
            )
            g_scale_num_inline630__tile_1: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_4, pl.const(32768, pl.INT64), 32), pl.Mem.Vec] = pl.tile.full([1, 8], dtype=pl.FP32, value=127.0)
            g_sq_row_inline646__tile_1: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_4, pl.const(32768, pl.INT64), 32), pl.Mem.Vec] = pl.tile.div(g_scale_num_inline630__tile_1, g_amax_inline642__tile_1)
            g_scale_dq_inline647__tile_1: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_5, pl.const(65536, pl.INT64), 32), pl.Mem.Vec] = pl.tile.muls(g_amax_inline642__tile_1, 0.007874015748031496)
            g_sq_col_inline612__tile_1: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_4, pl.const(32768, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(g_sq_row_inline646__tile_1, [8, 1])
            oq_scaled_inline651__tile_1: pl.Tile[[8, 1024], pl.FP32, pl.MemRef(mem_vec_3, pl.const(0, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.row_expand_mul(
                oc_q_inline594__tile_1, g_sq_col_inline612__tile_1
            )
            oq_i32_inline617__tile_1: pl.Tile[[8, 1024], pl.INT32, pl.MemRef(mem_vec_3, pl.const(0, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.cast(
                oq_scaled_inline651__tile_1, target_type=pl.INT32, mode="rint"
            )
            oq_half_inline593__tile_1: pl.Tile[[8, 1024], pl.FP16, pl.MemRef(mem_vec_3, pl.const(0, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.cast(
                oq_i32_inline617__tile_1, target_type=pl.FP16, mode="round"
            )
            oq_i8_inline649__tile_1: pl.Tile[[8, 1024], pl.INT8, pl.MemRef(mem_vec_3, pl.const(0, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.cast(
                oq_half_inline593__tile_1, target_type=pl.INT8, mode="trunc"
            )
            act_scale_dq_inline598__tile_1: pl.Tensor[[8, 384], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 12288)] = pl.tile.store(
                g_scale_dq_inline647__tile_1, [g_inline628__idx_v0, qt_inline590__idx_v0 + 8], act_scale_dq_inline598__tile
            )
            o_r_i8_pad_inline621__tile_1: pl.Tensor[[384, 8192], pl.INT8, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 3145728)] = pl.tile.store(
                oq_i8_inline649__tile_1, [qt_inline590__idx_v0 + 8, col_g_inline610__ssa_v0], o_r_i8_pad_inline621__tile
            )
            act_scale_dq_inline598__rv_v4_main, o_r_i8_pad_inline621__rv_v4_main = pl.yield_(act_scale_dq_inline598__tile_1, o_r_i8_pad_inline621__tile_1)
        unroll_rem: pl.Scalar[pl.INDEX] = (t_dim_inline607__ssa_v0 + 7) // 8 - (t_dim_inline607__ssa_v0 + 7) // 16 * 2
        if unroll_rem == 1:
            oc_amax_inline650__tile_2: pl.Tile[[8, 1024], pl.FP32, pl.MemRef(mem_vec_3, pl.const(0, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.load(
                o_r_pad_inline620__ssa_v3, [unroll_main_end, col_g_inline610__ssa_v0], [8, 1024], [8, 1024], target_memory=pl.Mem.Vec
            )
            g_abs_inline640__tile_2: pl.Tile[[8, 1024], pl.FP32, pl.MemRef(mem_vec_3, pl.const(0, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.abs(oc_amax_inline650__tile_2)
            tmp_tile_2: pl.Tile[[8, 1024], pl.FP32, pl.MemRef(mem_vec_4, pl.const(32768, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.create([8, 1024], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            g_row_max_inline601__tile_2: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_5, pl.const(65536, pl.INT64), 32), pl.Mem.Vec] = pl.tile.row_max(g_abs_inline640__tile_2, tmp_tile_2)
            g_row_max_v1_inline619__tile_2: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_5, pl.const(65536, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(g_row_max_inline601__tile_2, [1, 8])
            g_amax_floor_inline641__tile_2: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_3, pl.const(0, pl.INT64), 32), pl.Mem.Vec] = pl.tile.full([1, 8], dtype=pl.FP32, value=0.0001)
            g_amax_inline642__tile_2: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_3, pl.const(0, pl.INT64), 32), pl.Mem.Vec] = pl.tile.maximum(
                g_amax_floor_inline641__tile_2, g_row_max_v1_inline619__tile_2
            )
            g_scale_num_inline630__tile_2: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_4, pl.const(32768, pl.INT64), 32), pl.Mem.Vec] = pl.tile.full([1, 8], dtype=pl.FP32, value=127.0)
            g_sq_row_inline646__tile_2: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_4, pl.const(32768, pl.INT64), 32), pl.Mem.Vec] = pl.tile.div(g_scale_num_inline630__tile_2, g_amax_inline642__tile_2)
            g_scale_dq_inline647__tile_2: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_3, pl.const(0, pl.INT64), 32), pl.Mem.Vec] = pl.tile.muls(g_amax_inline642__tile_2, 0.007874015748031496)
            act_scale_dq_inline598__tile_2: pl.Tensor[[8, 384], pl.FP32, pl.MemRef("mem_ddr_31", pl.const(0, pl.INT64), 12288)] = pl.tile.store(
                g_scale_dq_inline647__tile_2, [g_inline628__idx_v0, unroll_main_end], act_scale_dq_inline598__rv_v4_main
            )
            g_sq_col_inline612__tile_2: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_4, pl.const(32768, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(g_sq_row_inline646__tile_2, [8, 1])
            oc_q_inline594__tile_2: pl.Tile[[8, 1024], pl.FP32, pl.MemRef(mem_vec_3, pl.const(0, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.load(
                o_r_pad_inline620__ssa_v3, [unroll_main_end, col_g_inline610__ssa_v0], [8, 1024], [8, 1024], target_memory=pl.Mem.Vec
            )
            oq_scaled_inline651__tile_2: pl.Tile[[8, 1024], pl.FP32, pl.MemRef(mem_vec_3, pl.const(0, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.row_expand_mul(
                oc_q_inline594__tile_2, g_sq_col_inline612__tile_2
            )
            oq_i32_inline617__tile_2: pl.Tile[[8, 1024], pl.INT32, pl.MemRef(mem_vec_3, pl.const(0, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.cast(
                oq_scaled_inline651__tile_2, target_type=pl.INT32, mode="rint"
            )
            oq_half_inline593__tile_2: pl.Tile[[8, 1024], pl.FP16, pl.MemRef(mem_vec_3, pl.const(0, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.cast(
                oq_i32_inline617__tile_2, target_type=pl.FP16, mode="round"
            )
            oq_i8_inline649__tile_2: pl.Tile[[8, 1024], pl.INT8, pl.MemRef(mem_vec_3, pl.const(0, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.cast(
                oq_half_inline593__tile_2, target_type=pl.INT8, mode="trunc"
            )
            o_r_i8_pad_inline621__tile_2: pl.Tensor[[384, 8192], pl.INT8, pl.MemRef("mem_ddr_32", pl.const(0, pl.INT64), 3145728)] = pl.tile.store(
                oq_i8_inline649__tile_2, [unroll_main_end, col_g_inline610__ssa_v0], o_r_i8_pad_inline621__rv_v4_main
            )
            o_r_i8_pad_inline621__rv_v4: pl.Tensor[[384, 8192], pl.INT8, pl.MemRef("mem_ddr_48", pl.const(0, pl.INT64), 3145728)] = pl.yield_(o_r_i8_pad_inline621__tile_2)
        else:
            o_r_i8_pad_inline621__rv_v4: pl.Tensor[[384, 8192], pl.INT8, pl.MemRef("mem_ddr_48", pl.const(0, pl.INT64), 3145728)] = pl.yield_(o_r_i8_pad_inline621__rv_v4_main)
        for zt_inline608__idx_v0, (o_r_i8_pad_inline621__iter_v6,) in pl.range(t_dim_inline607__ssa_v0, proj_b_padded_rows_inline614__ssa_v0, 8, init_values=(o_r_i8_pad_inline621__rv_v4,)):
            zero_half_inline652__tile: pl.Tile[[8, 1024], pl.FP16, pl.MemRef(mem_vec_3, pl.const(0, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.full([8, 1024], dtype=pl.FP16, value=0.0)
            t__tile: pl.Tile[[8, 1024], pl.INT8, pl.MemRef(mem_vec_3, pl.const(0, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.cast(zero_half_inline652__tile, target_type=pl.INT8, mode="trunc")
            o_r_i8_pad_inline621__tile_3: pl.Tensor[[384, 8192], pl.INT8, pl.MemRef("mem_ddr_48", pl.const(0, pl.INT64), 3145728)] = pl.tile.store(
                t__tile, [zt_inline608__idx_v0, col_g_inline610__ssa_v0], o_r_i8_pad_inline621__iter_v6
            )
            o_r_i8_pad_inline621__rv_v7: pl.Tensor[[384, 8192], pl.INT8, pl.MemRef("mem_ddr_51", pl.const(0, pl.INT64), 3145728)] = pl.yield_(o_r_i8_pad_inline621__tile_3)
        return act_scale_dq_inline598__iter_v1, o_r_i8_pad_inline621__iter_v1

    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def rmsnorm_rope_cache_write(
        bs_inline1923__ssa_v0: pl.Scalar[pl.INDEX],
        cmp_cos_il__rv_v2: pl.Tensor[[T_DYN, 64], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)],
        cmp_sin_signed__rv_v2: pl.Tensor[[T_DYN, 64], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)],
        pooled_kv_inline511__rv_v2: pl.Tensor[[384, 512], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 786432)],
        normed_kv_inline1917__ssa_v0: pl.InOut[pl.Tensor[[384, 512], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 786432)]],
        norm_w_2d_inline1909__ssa_v0: pl.Tensor[[1, 512], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 2048)],
        cmp_kv_cache_flat_inline1914__ssa_v0: pl.Out[pl.Tensor[[cmp_block_num_inline1916__ssa_v0 * pl.const(32, pl.INDEX), 512], pl.BF16, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 0)]],
        kv_flat_inline1928__ssa_v0: pl.Out[pl.Tensor[[T_DYN, 512], pl.FP32, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 0)]],
        cmp_slot_mapping__ssa_v0: pl.Tensor[[T_DYN], pl.INT64, pl.MemRef("mem_ddr_7", pl.const(0, pl.INT64), 0)],
    ):
        pl.func_attr({"mx_tensor_views_blocked": True})
        mem_vec_8: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 4096)
        mem_vec_9: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 4096)
        mem_vec_12: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 4096)
        mem_vec_13: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 8192)
        mem_vec_32: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        mem_vec_35: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 4096)
        mem_vec_46: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        rms_blk_inline1927__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        b0_inline1937__ssa_v0: pl.Scalar[pl.INDEX] = rms_blk_inline1927__ssa_v0 * 16
        rms_blk_rows_inline1932__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(bs_inline1923__ssa_v0 - b0_inline1937__ssa_v0, 16)
        cos_b_inline1939__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_8, pl.const(16896, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(valid_shape=[rms_blk_rows_inline1932__ssa_v0, 64])] = (
            pl.tile.load(cmp_cos_il__rv_v2, [b0_inline1937__ssa_v0, 0], [16, 64], [rms_blk_rows_inline1932__ssa_v0, 64], target_memory=pl.Mem.Vec)
        )
        sin_b_inline1924__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_9, pl.const(20992, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(valid_shape=[rms_blk_rows_inline1932__ssa_v0, 64])] = (
            pl.tile.load(cmp_sin_signed__rv_v2, [b0_inline1937__ssa_v0, 0], [16, 64], [rms_blk_rows_inline1932__ssa_v0, 64], target_memory=pl.Mem.Vec)
        )
        partial_sq_inline1904__tile: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_35, pl.const(12544, pl.INT64), 64), pl.Mem.Vec] = pl.tile.full([1, 16], dtype=pl.FP32, value=0.0)
        for k0_inline1906__idx_v0, (partial_sq_inline1904__iter_v1,) in pl.range(0, 512, 64, init_values=(partial_sq_inline1904__tile,)):
            kv_rms_chunk_inline1902__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(4096, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.load(
                pooled_kv_inline511__rv_v2, [b0_inline1937__ssa_v0, k0_inline1906__idx_v0], [16, 64], [16, 64], target_memory=pl.Mem.Vec
            )
            kv_rms_sq_inline1901__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.mul(
                kv_rms_chunk_inline1902__tile, kv_rms_chunk_inline1902__tile
            )
            tmp_tile: pl.Tile[[16, 128], pl.FP32, pl.MemRef(mem_vec_13, pl.const(4096, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.create([16, 128], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            t__tile: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_32, pl.const(12288, pl.INT64), 64), pl.Mem.Vec] = pl.tile.row_sum(kv_rms_sq_inline1901__tile, tmp_tile)
            kv_rms_rowsum_inline1900__tile: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_32, pl.const(12288, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(t__tile, [1, 16])
            partial_sq_inline1904__tile_1: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_35, pl.const(12544, pl.INT64), 64), pl.Mem.Vec] = pl.tile.add(
                partial_sq_inline1904__iter_v1, kv_rms_rowsum_inline1900__tile
            )
            partial_sq_inline1904__rv_v2: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_35, pl.const(12544, pl.INT64), 64), pl.Mem.Vec] = pl.yield_(partial_sq_inline1904__tile_1)
        t__tile_1: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_13, pl.const(4096, pl.INT64), 64), pl.Mem.Vec] = pl.tile.muls(partial_sq_inline1904__rv_v2, 0.001953125)
        t__tile_2: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_13, pl.const(4096, pl.INT64), 64), pl.Mem.Vec] = pl.tile.adds(t__tile_1, 9.9999999999999995e-07)
        variance_inline1911__tile: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_13, pl.const(4096, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(t__tile_2, [16, 1])
        t__rm_a0_tmp_v0: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_13, pl.const(4096, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(variance_inline1911__tile, [1, 16])
        t__row_major_tmp_v1: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_13, pl.const(4096, pl.INT64), 64), pl.Mem.Vec] = pl.tile.sqrt(t__rm_a0_tmp_v0)
        t__tile_3: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_13, pl.const(4096, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(t__row_major_tmp_v1, [16, 1])
        inv_rms_inline1935__rm_a0_tmp_v2: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_13, pl.const(4096, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(t__tile_3, [1, 16])
        inv_rms_inline1935__row_major_tmp_v3: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_35, pl.const(12544, pl.INT64), 64), pl.Mem.Vec] = pl.tile.recip(inv_rms_inline1935__rm_a0_tmp_v2)
        inv_rms_inline1935__tile: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_35, pl.const(12544, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(inv_rms_inline1935__row_major_tmp_v3, [16, 1])
        for k0_inline1898__idx_v0, (normed_kv_inline1917__iter_v1,) in pl.range(0, 448, 64, init_values=(normed_kv_inline1917__ssa_v0,)):
            kv_norm_chunk_inline1897__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(4096, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.load(
                pooled_kv_inline511__rv_v2, [b0_inline1937__ssa_v0, k0_inline1898__idx_v0], [16, 64], [16, 64], target_memory=pl.Mem.Vec
            )
            gamma_inline1899__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(0, pl.INT64), 256), pl.Mem.Vec] = pl.tile.load(
                norm_w_2d_inline1909__ssa_v0, [0, k0_inline1898__idx_v0], [1, 64], [1, 64], target_memory=pl.Mem.Vec
            )
            t__tile_4: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(4096, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.row_expand_mul(kv_norm_chunk_inline1897__tile, inv_rms_inline1935__tile)
            normed_chunk_inline1896__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(4096, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.col_expand_mul(t__tile_4, gamma_inline1899__tile)
            normed_kv_inline1917__tile: pl.Tensor[[384, 512], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 786432)] = pl.tile.store(
                normed_chunk_inline1896__tile, [b0_inline1937__ssa_v0, k0_inline1898__idx_v0], normed_kv_inline1917__iter_v1
            )
            normed_kv_inline1917__rv_v2: pl.Tensor[[384, 512], pl.FP32, pl.MemRef("mem_ddr_25", pl.const(0, pl.INT64), 786432)] = pl.yield_(normed_kv_inline1917__tile)
        kv_rope_norm_inline1895__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(4096, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.load(
            pooled_kv_inline511__rv_v2, [b0_inline1937__ssa_v0, 448], [16, 64], [16, 64], target_memory=pl.Mem.Vec
        )
        gamma_rope_inline1910__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(0, pl.INT64), 256), pl.Mem.Vec] = pl.tile.load(
            norm_w_2d_inline1909__ssa_v0, [0, 448], [1, 64], [1, 64], target_memory=pl.Mem.Vec
        )
        t__tile_5: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(4096, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.row_expand_mul(kv_rope_norm_inline1895__tile, inv_rms_inline1935__tile)
        rope_normed_inline1894__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(4096, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.col_expand_mul(t__tile_5, gamma_rope_inline1910__tile)
        rope_ones_inline1908__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.full([16, 64], dtype=pl.FP32, value=1.0)
        rope_index_inline1893__ci_tmp_v0: pl.Tile[[1, 192], pl.FP32, pl.MemRef(mem_vec_35, pl.const(12544, pl.INT64), 768), pl.Mem.Vec] = pl.tile.create(
            [1, 192], dtype=pl.FP32, target_memory=pl.Mem.Vec
        )
        rope_index_inline1893__tile: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_32, pl.const(12288, pl.INT64), 256), pl.Mem.Vec] = pl.tile.ci(
            pl.const(0, pl.INT32), [1, 64], tmp=rope_index_inline1893__ci_tmp_v0, dtype=pl.INT32, descending=False
        )
        rope_index_f_inline1892__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_35, pl.const(12544, pl.INT64), 256), pl.Mem.Vec] = pl.tile.cast(
            rope_index_inline1893__tile, target_type=pl.FP32, mode="round"
        )
        rope_col_inline1890__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.col_expand_mul(
            rope_ones_inline1908__tile, rope_index_f_inline1892__tile
        )
        t__tile_6: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_35, pl.const(12544, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.muls(rope_col_inline1890__tile, 0.5)
        t__tile_7: pl.Tile[[16, 64], pl.INT32, pl.MemRef(mem_vec_35, pl.const(12544, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(t__tile_6, target_type=pl.INT32, mode="trunc")
        rope_dup_f_inline1903__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_35, pl.const(12544, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(t__tile_7, target_type=pl.FP32, mode="round")
        t__tile_8: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_35, pl.const(12544, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.muls(rope_dup_f_inline1903__tile, 2.0)
        rope_lane_inline1889__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_35, pl.const(12544, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.sub(rope_col_inline1890__tile, t__tile_8)
        t__tile_9: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.adds(rope_col_inline1890__tile, 1.0)
        t__tile_10: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_35, pl.const(12544, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.muls(rope_lane_inline1889__tile, 2.0)
        t__tile_11: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.sub(t__tile_9, t__tile_10)
        rope_swap_idx_inline1888__tile: pl.Tile[[16, 64], pl.INT32, pl.MemRef(mem_vec_12, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(t__tile_11, target_type=pl.INT32, mode="round")
        gather_acc_init: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_35, pl.const(12544, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.create([16, 64], dtype=pl.FP32, target_memory=pl.Mem.Vec)
        for gather_lv, (gather_ia,) in pl.range(16, init_values=(gather_acc_init,)):
            gather_inp_row: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(4096, pl.INT64), 256), pl.Mem.Vec] = pl.tile.slice(rope_normed_inline1894__tile, [1, 64], [gather_lv, 0], [1, 64])
            gather_idx_row: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_12, pl.const(0, pl.INT64), 256), pl.Mem.Vec] = pl.tile.slice(rope_swap_idx_inline1888__tile, [1, 64], [gather_lv, 0], [1, 64])
            gather_row_tmp: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_32, pl.const(12288, pl.INT64), 256), pl.Mem.Vec] = pl.tile.create([1, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec)
            gather_row: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_46, pl.const(16640, pl.INT64), 256), pl.Mem.Vec] = pl.tile.gather(gather_inp_row, gather_idx_row, gather_row_tmp)
            gather_asmbl: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_35, pl.const(12544, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.assemble(gather_ia, gather_row, [gather_lv, 0])
            gather_rv: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_35, pl.const(12544, pl.INT64), 4096), pl.Mem.Vec] = pl.yield_(gather_asmbl)
        swapped_inline1922__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_35, pl.const(12544, pl.INT64), 4096), pl.Mem.Vec] = gather_rv
        t__tile_12: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(4096, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.mul(rope_normed_inline1894__tile, cos_b_inline1939__tile)
        t__tile_13: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_8, pl.const(16896, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.mul(swapped_inline1922__tile, sin_b_inline1924__tile)
        rope_rot_inline1933__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(4096, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.add(t__tile_12, t__tile_13)
        normed_kv_inline1917__tile_1: pl.Tensor[[384, 512], pl.FP32, pl.MemRef("mem_ddr_25", pl.const(0, pl.INT64), 786432)] = pl.tile.store(
            rope_rot_inline1933__tile, [b0_inline1937__ssa_v0, 448], normed_kv_inline1917__rv_v2
        )
        for inner_inline1887__idx_v0, (cmp_kv_cache_flat_inline1914__iter_v1, kv_flat_inline1928__iter_v1) in pl.range(
            rms_blk_rows_inline1932__ssa_v0, init_values=(cmp_kv_cache_flat_inline1914__ssa_v0, kv_flat_inline1928__ssa_v0)
        ):
            token_inline1931__ssa_v1: pl.Scalar[pl.INDEX] = b0_inline1937__ssa_v0 + inner_inline1887__idx_v0
            cache_row_i64_inline1891__tile: pl.Scalar[pl.INT64] = pl.tensor.read(cmp_slot_mapping__ssa_v0, [token_inline1931__ssa_v1])
            if 0 <= cache_row_i64_inline1891__tile:
                cache_row_inline1886__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(cache_row_i64_inline1891__tile, pl.INDEX)
                kv_row_fp32_inline1885__tile: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_13, pl.const(4096, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.load(
                    normed_kv_inline1917__tile_1, [token_inline1931__ssa_v1, 0], [1, 512], [1, 512], target_memory=pl.Mem.Vec
                )
                kv_flat_inline1928__tile: pl.Tensor[[T_DYN, 512], pl.FP32, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                    kv_row_fp32_inline1885__tile, [token_inline1931__ssa_v1, 0], kv_flat_inline1928__iter_v1
                )
                t__tile_14: pl.Tile[[1, 512], pl.BF16, pl.MemRef(mem_vec_13, pl.const(4096, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.cast(kv_row_fp32_inline1885__tile, target_type=pl.BF16, mode="rint")
                cmp_kv_cache_flat_inline1914__tile: pl.Tensor[[cmp_block_num_inline1916__ssa_v0 * pl.const(32, pl.INDEX), 512], pl.BF16, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 0)] = (
                    pl.tile.store(t__tile_14, [cache_row_inline1886__ssa_v0, 0], cmp_kv_cache_flat_inline1914__iter_v1)
                )
                cmp_kv_cache_flat_inline1914__phi_v4, kv_flat_inline1928__phi_v4 = pl.yield_(cmp_kv_cache_flat_inline1914__tile, kv_flat_inline1928__tile)
            else:
                cmp_kv_cache_flat_inline1914__phi_v4, kv_flat_inline1928__phi_v4 = pl.yield_(cmp_kv_cache_flat_inline1914__iter_v1, kv_flat_inline1928__iter_v1)
            cmp_kv_cache_flat_inline1914__rv_v2, kv_flat_inline1928__rv_v2 = pl.yield_(cmp_kv_cache_flat_inline1914__phi_v4, kv_flat_inline1928__phi_v4)
        return

    @pl.function(type=pl.FunctionType.Spmd)
    def rmsnorm_rope_cache_write_spmd(
        self,
        bs_inline1923__ssa_v0: pl.Scalar[pl.INDEX],
        cmp_cos_il__rv_v2: pl.Tensor[[T_DYN, 64], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)],
        cmp_sin_signed__rv_v2: pl.Tensor[[T_DYN, 64], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)],
        pooled_kv_inline511__rv_v2: pl.Tensor[[384, 512], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 786432)],
        normed_kv_inline1917__ssa_v0: pl.InOut[pl.Tensor[[384, 512], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 786432)]],
        norm_w_2d_inline1909__ssa_v0: pl.Tensor[[1, 512], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 2048)],
        cmp_kv_cache_flat_inline1914__ssa_v0: pl.Out[pl.Tensor[[cmp_block_num_inline1916__ssa_v0 * pl.const(32, pl.INDEX), 512], pl.BF16, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 0)]],
        kv_flat_inline1928__ssa_v0: pl.Out[pl.Tensor[[T_DYN, 512], pl.FP32, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 0)]],
        cmp_slot_mapping__ssa_v0: pl.Tensor[[T_DYN], pl.INT64, pl.MemRef("mem_ddr_7", pl.const(0, pl.INT64), 0)],
    ):
        pl.func_attr({"mx_tensor_views_blocked": True})
        self.rmsnorm_rope_cache_write(
            bs_inline1923__ssa_v0,
            cmp_cos_il__rv_v2,
            cmp_sin_signed__rv_v2,
            pooled_kv_inline511__rv_v2,
            normed_kv_inline1917__ssa_v0,
            norm_w_2d_inline1909__ssa_v0,
            cmp_kv_cache_flat_inline1914__ssa_v0,
            kv_flat_inline1928__ssa_v0,
            cmp_slot_mapping__ssa_v0,
            attrs={"arg_directions": [pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.inout, pl.adir.input, pl.adir.output_existing, pl.adir.output_existing, pl.adir.input]},
        )

    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def rmsnorm_rope(
        normed_kv_inline518__ssa_v1: pl.Out[pl.Tensor[[384, 128], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 98304)]],
        rms_blocks_inline422_inline1996__ssa_v0: pl.Scalar[pl.INDEX],
        rms_workers_inline457_inline1967__ssa_v0: pl.Scalar[pl.INDEX],
        bs_inline423_inline1991__ssa_v0: pl.Scalar[pl.INDEX],
        inner_cos_il__rv_v2: pl.Tensor[[T_DYN, 64], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)],
        inner_sin_signed__rv_v2: pl.Tensor[[T_DYN, 64], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)],
        pooled_kv_inline461_inline2039__rv_v2: pl.Tensor[[384, 128], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 196608)],
        norm_w_2d_inline465_inline1970__ssa_v0: pl.Tensor[[1, 128], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 512)],
        position_ids__ssa_v0: pl.Tensor[[T_DYN], pl.INT32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 0)],
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
        rms_worker_inline453_inline1965__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        rope_ones_inline458_inline1964__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_26, pl.const(16384, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.full([16, 64], dtype=pl.FP32, value=1.0)
        rope_index_inline420_inline2043__ci_tmp_v0: pl.Tile[[1, 192], pl.FP32, pl.MemRef(mem_vec_30, pl.const(28736, pl.INT64), 768), pl.Mem.Vec] = pl.tile.create(
            [1, 192], dtype=pl.FP32, target_memory=pl.Mem.Vec
        )
        rope_index_inline420_inline2043__tile: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_19, pl.const(0, pl.INT64), 256), pl.Mem.Vec] = pl.tile.ci(
            pl.const(0, pl.INT32), [1, 64], tmp=rope_index_inline420_inline2043__ci_tmp_v0, dtype=pl.INT32, descending=False
        )
        rope_index_f_inline467_inline1963__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_30, pl.const(28736, pl.INT64), 256), pl.Mem.Vec] = pl.tile.cast(
            rope_index_inline420_inline2043__tile, target_type=pl.FP32, mode="round"
        )
        rope_col_inline414_inline1962__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_26, pl.const(16384, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.col_expand_mul(
            rope_ones_inline458_inline1964__tile, rope_index_f_inline467_inline1963__tile
        )
        t__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_30, pl.const(28736, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.muls(rope_col_inline414_inline1962__tile, 0.5)
        t__tile_1: pl.Tile[[16, 64], pl.INT32, pl.MemRef(mem_vec_30, pl.const(28736, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(t__tile, target_type=pl.INT32, mode="trunc")
        rope_dup_f_inline468_inline1960__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_30, pl.const(28736, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(
            t__tile_1, target_type=pl.FP32, mode="round"
        )
        t__tile_2: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_30, pl.const(28736, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.muls(rope_dup_f_inline468_inline1960__tile, 2.0)
        rope_lane_inline469_inline1958__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_30, pl.const(28736, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.sub(
            rope_col_inline414_inline1962__tile, t__tile_2
        )
        t__tile_3: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_26, pl.const(16384, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.adds(rope_col_inline414_inline1962__tile, 1.0)
        t__tile_4: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_30, pl.const(28736, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.muls(rope_lane_inline469_inline1958__tile, 2.0)
        t__tile_5: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_26, pl.const(16384, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.sub(t__tile_3, t__tile_4)
        rope_swap_idx_inline441_inline2012__tile: pl.Tile[[16, 64], pl.INT32, pl.MemRef(mem_vec_19, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(
            t__tile_5, target_type=pl.INT32, mode="round"
        )
        for rms_blk_inline452_inline1957__idx_v0, (normed_kv_inline518__iter_v2,) in pl.range(
            rms_worker_inline453_inline1965__ssa_v0, rms_blocks_inline422_inline1996__ssa_v0, rms_workers_inline457_inline1967__ssa_v0, init_values=(normed_kv_inline518__ssa_v1,)
        ):
            b0_inline404_inline1975__ssa_v0: pl.Scalar[pl.INDEX] = rms_blk_inline452_inline1957__idx_v0 * 16
            rms_blk_rows_inline403_inline2016__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(bs_inline423_inline1991__ssa_v0 - b0_inline404_inline1975__ssa_v0, 16)
            cos_b_inline402_inline1955__tile: pl.Tile[
                [16, 64], pl.FP32, pl.MemRef(mem_vec_20, pl.const(4096, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(valid_shape=[rms_blk_rows_inline403_inline2016__ssa_v0, 64])
            ] = pl.tile.load(inner_cos_il__rv_v2, [b0_inline404_inline1975__ssa_v0, 0], [16, 64], [rms_blk_rows_inline403_inline2016__ssa_v0, 64], target_memory=pl.Mem.Vec)
            sin_b_inline401_inline1990__tile: pl.Tile[
                [16, 64], pl.FP32, pl.MemRef(mem_vec_21, pl.const(8192, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(valid_shape=[rms_blk_rows_inline403_inline2016__ssa_v0, 64])
            ] = pl.tile.load(inner_sin_signed__rv_v2, [b0_inline404_inline1975__ssa_v0, 0], [16, 64], [rms_blk_rows_inline403_inline2016__ssa_v0, 64], target_memory=pl.Mem.Vec)
            partial_sq_inline400_inline1966__tile: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_49, pl.const(36992, pl.INT64), 64), pl.Mem.Vec] = pl.tile.full([1, 16], dtype=pl.FP32, value=0.0)
            kv_rms_chunk_inline399_inline1998__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_26, pl.const(16384, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.load(
                pooled_kv_inline461_inline2039__rv_v2, [b0_inline404_inline1975__ssa_v0, 0], [16, 64], [16, 64], target_memory=pl.Mem.Vec
            )
            kv_rms_chunk_inline399_inline1998__tile_1: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_30, pl.const(28736, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.load(
                pooled_kv_inline461_inline2039__rv_v2, [b0_inline404_inline1975__ssa_v0, 64], [16, 64], [16, 64], target_memory=pl.Mem.Vec
            )
            kv_rms_sq_inline436_inline1953__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_25, pl.const(12288, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.mul(
                kv_rms_chunk_inline399_inline1998__tile, kv_rms_chunk_inline399_inline1998__tile
            )
            tmp_tile: pl.Tile[[16, 128], pl.FP32, pl.MemRef(mem_vec_26, pl.const(16384, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.create([16, 128], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            t__tile_6: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_27, pl.const(24576, pl.INT64), 64), pl.Mem.Vec] = pl.tile.row_sum(kv_rms_sq_inline436_inline1953__tile, tmp_tile)
            kv_rms_rowsum_inline398_inline1952__tile: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_27, pl.const(24576, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(t__tile_6, [1, 16])
            partial_sq_inline400_inline1966__tile_1: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_26, pl.const(16384, pl.INT64), 64), pl.Mem.Vec] = pl.tile.add(
                partial_sq_inline400_inline1966__tile, kv_rms_rowsum_inline398_inline1952__tile
            )
            kv_rms_sq_inline436_inline1953__tile_1: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_29, pl.const(24640, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.mul(
                kv_rms_chunk_inline399_inline1998__tile_1, kv_rms_chunk_inline399_inline1998__tile_1
            )
            tmp_tile_1: pl.Tile[[16, 128], pl.FP32, pl.MemRef(mem_vec_30, pl.const(28736, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.create([16, 128], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            t__tile_7: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_31, pl.const(36928, pl.INT64), 64), pl.Mem.Vec] = pl.tile.row_sum(kv_rms_sq_inline436_inline1953__tile_1, tmp_tile_1)
            kv_rms_rowsum_inline398_inline1952__tile_1: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_31, pl.const(36928, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(t__tile_7, [1, 16])
            partial_sq_inline400_inline1966__tile_2: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_49, pl.const(36992, pl.INT64), 64), pl.Mem.Vec] = pl.tile.add(
                partial_sq_inline400_inline1966__tile_1, kv_rms_rowsum_inline398_inline1952__tile_1
            )
            t__tile_8: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_26, pl.const(16384, pl.INT64), 64), pl.Mem.Vec] = pl.tile.muls(partial_sq_inline400_inline1966__tile_2, 0.0078125)
            t__tile_9: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_26, pl.const(16384, pl.INT64), 64), pl.Mem.Vec] = pl.tile.adds(t__tile_8, 9.9999999999999995e-07)
            variance_inline397_inline1951__tile: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_26, pl.const(16384, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(t__tile_9, [16, 1])
            t__rm_a0_tmp_v0: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_26, pl.const(16384, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(variance_inline397_inline1951__tile, [1, 16])
            t__row_major_tmp_v1: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_26, pl.const(16384, pl.INT64), 64), pl.Mem.Vec] = pl.tile.sqrt(t__rm_a0_tmp_v0)
            t__tile_10: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_26, pl.const(16384, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(t__row_major_tmp_v1, [16, 1])
            inv_rms_inline418_inline2026__rm_a0_tmp_v2: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_26, pl.const(16384, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(t__tile_10, [1, 16])
            inv_rms_inline418_inline2026__row_major_tmp_v3: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_29, pl.const(24640, pl.INT64), 64), pl.Mem.Vec] = pl.tile.recip(
                inv_rms_inline418_inline2026__rm_a0_tmp_v2
            )
            inv_rms_inline418_inline2026__tile: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_29, pl.const(24640, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(
                inv_rms_inline418_inline2026__row_major_tmp_v3, [16, 1]
            )
            kv_norm_chunk_inline396_inline1950__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_26, pl.const(16384, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.load(
                pooled_kv_inline461_inline2039__rv_v2, [b0_inline404_inline1975__ssa_v0, 0], [16, 64], [16, 64], target_memory=pl.Mem.Vec
            )
            gamma_inline424_inline1954__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_30, pl.const(28736, pl.INT64), 256), pl.Mem.Vec] = pl.tile.load(
                norm_w_2d_inline465_inline1970__ssa_v0, [0, 0], [1, 64], [1, 64], target_memory=pl.Mem.Vec
            )
            t__tile_11: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_26, pl.const(16384, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.row_expand_mul(
                kv_norm_chunk_inline396_inline1950__tile, inv_rms_inline418_inline2026__tile
            )
            normed_chunk_inline395_inline2006__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_26, pl.const(16384, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.col_expand_mul(
                t__tile_11, gamma_inline424_inline1954__tile
            )
            normed_nope_inline394_inline1948__tile: pl.Tile[[16, 64], pl.BF16, pl.MemRef(mem_vec_25, pl.const(12288, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(
                normed_chunk_inline395_inline2006__tile, target_type=pl.BF16, mode="rint"
            )
            kv_rope_norm_inline393_inline1959__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_26, pl.const(16384, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.load(
                pooled_kv_inline461_inline2039__rv_v2, [b0_inline404_inline1975__ssa_v0, 64], [16, 64], [16, 64], target_memory=pl.Mem.Vec
            )
            gamma_rope_inline392_inline1961__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_30, pl.const(28736, pl.INT64), 256), pl.Mem.Vec] = pl.tile.load(
                norm_w_2d_inline465_inline1970__ssa_v0, [0, 64], [1, 64], [1, 64], target_memory=pl.Mem.Vec
            )
            t__tile_12: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_26, pl.const(16384, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.row_expand_mul(
                kv_rope_norm_inline393_inline1959__tile, inv_rms_inline418_inline2026__tile
            )
            rope_normed_inline411_inline1947__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_26, pl.const(16384, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.col_expand_mul(
                t__tile_12, gamma_rope_inline392_inline1961__tile
            )
            gather_acc_init: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_30, pl.const(28736, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.create([16, 64], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            for gather_lv, (gather_ia,) in pl.range(16, init_values=(gather_acc_init,)):
                gather_inp_row: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_26, pl.const(16384, pl.INT64), 256), pl.Mem.Vec] = pl.tile.slice(
                    rope_normed_inline411_inline1947__tile, [1, 64], [gather_lv, 0], [1, 64]
                )
                gather_idx_row: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_19, pl.const(0, pl.INT64), 256), pl.Mem.Vec] = pl.tile.slice(
                    rope_swap_idx_inline441_inline2012__tile, [1, 64], [gather_lv, 0], [1, 64]
                )
                gather_row_tmp: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_29, pl.const(24640, pl.INT64), 256), pl.Mem.Vec] = pl.tile.create([1, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec)
                gather_row: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_49, pl.const(36992, pl.INT64), 256), pl.Mem.Vec] = pl.tile.gather(gather_inp_row, gather_idx_row, gather_row_tmp)
                gather_asmbl: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_30, pl.const(28736, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.assemble(gather_ia, gather_row, [gather_lv, 0])
                gather_rv: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_30, pl.const(28736, pl.INT64), 4096), pl.Mem.Vec] = pl.yield_(gather_asmbl)
            swapped_inline391_inline1946__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_30, pl.const(28736, pl.INT64), 4096), pl.Mem.Vec] = gather_rv
            t__tile_13: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_26, pl.const(16384, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.mul(
                rope_normed_inline411_inline1947__tile, cos_b_inline402_inline1955__tile
            )
            t__tile_14: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_30, pl.const(28736, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.mul(
                swapped_inline391_inline1946__tile, sin_b_inline401_inline1990__tile
            )
            rope_rot_inline390_inline1945__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_26, pl.const(16384, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.add(t__tile_13, t__tile_14)
            normed_rope_inline416_inline1944__tile: pl.Tile[[16, 64], pl.BF16, pl.MemRef(mem_vec_26, pl.const(16384, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(
                rope_rot_inline390_inline1945__tile, target_type=pl.BF16, mode="rint"
            )
            for inner_inline389_inline2000__idx_v0, (normed_kv_inline518__iter_v4,) in pl.range(rms_blk_rows_inline403_inline2016__ssa_v0, init_values=(normed_kv_inline518__iter_v2,)):
                token_inline428_inline2029__ssa_v2: pl.Scalar[pl.INDEX] = b0_inline404_inline1975__ssa_v0 + inner_inline389_inline2000__idx_v0
                token_pos_inline437_inline2007__tile: pl.Scalar[pl.INT32] = pl.tensor.read(position_ids__ssa_v0, [token_inline428_inline2029__ssa_v2])
                if (pl.cast(token_pos_inline437_inline2007__tile, pl.INDEX) + 1) % 4 == 0:
                    request_inline388_inline1943__ssa_v0: pl.Scalar[pl.INDEX] = token_inline428_inline2029__ssa_v2 // 6
                    first_pos_inline387_inline1956__tile: pl.Scalar[pl.INT32] = pl.tensor.read(position_ids__ssa_v0, [request_inline388_inline1943__ssa_v0 * 6])
                    first_boundary_inline386_inline2010__ssa_v0: pl.Scalar[pl.INDEX] = 3 - pl.cast(first_pos_inline387_inline1956__tile, pl.INDEX) % 4
                    compact_token_inline385_inline1949__ssa_v0: pl.Scalar[pl.INDEX] = (
                        request_inline388_inline1943__ssa_v0 * 2 + (token_inline428_inline2029__ssa_v2 % 6 - first_boundary_inline386_inline2010__ssa_v0) // 4
                    )
                    t__tile_15: pl.Tile[[1, 64], pl.BF16, pl.MemRef(mem_vec_25, pl.const(12288, pl.INT64), 128), pl.Mem.Vec] = pl.tile.slice(
                        normed_nope_inline394_inline1948__tile, [1, 64], [inner_inline389_inline2000__idx_v0, 0]
                    )
                    normed_kv_inline518__tile: pl.Tensor[[384, 128], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 98304)] = pl.tile.store(
                        t__tile_15, [compact_token_inline385_inline1949__ssa_v0, 0], normed_kv_inline518__iter_v4
                    )
                    t__tile_16: pl.Tile[[1, 64], pl.BF16, pl.MemRef(mem_vec_26, pl.const(16384, pl.INT64), 128), pl.Mem.Vec] = pl.tile.slice(
                        normed_rope_inline416_inline1944__tile, [1, 64], [inner_inline389_inline2000__idx_v0, 0]
                    )
                    normed_kv_inline518__tile_1: pl.Tensor[[384, 128], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 98304)] = pl.tile.store(
                        t__tile_16, [compact_token_inline385_inline1949__ssa_v0, 64], normed_kv_inline518__tile
                    )
                    normed_kv_inline518__phi_v8: pl.Tensor[[384, 128], pl.BF16, pl.MemRef("mem_ddr_55", pl.const(0, pl.INT64), 98304)] = pl.yield_(normed_kv_inline518__tile_1)
                else:
                    normed_kv_inline518__phi_v8: pl.Tensor[[384, 128], pl.BF16, pl.MemRef("mem_ddr_55", pl.const(0, pl.INT64), 98304)] = pl.yield_(normed_kv_inline518__iter_v4)
                normed_kv_inline518__rv_v5: pl.Tensor[[384, 128], pl.BF16, pl.MemRef("mem_ddr_56", pl.const(0, pl.INT64), 98304)] = pl.yield_(normed_kv_inline518__phi_v8)
            normed_kv_inline518__rv_v3: pl.Tensor[[384, 128], pl.BF16, pl.MemRef("mem_ddr_57", pl.const(0, pl.INT64), 98304)] = pl.yield_(normed_kv_inline518__rv_v5)
        return normed_kv_inline518__ssa_v1

    @pl.function(type=pl.FunctionType.Spmd)
    def rmsnorm_rope_spmd(
        self,
        normed_kv_inline518__ssa_v1: pl.Out[pl.Tensor[[384, 128], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 98304)]],
        rms_blocks_inline422_inline1996__ssa_v0: pl.Scalar[pl.INDEX],
        rms_workers_inline457_inline1967__ssa_v0: pl.Scalar[pl.INDEX],
        bs_inline423_inline1991__ssa_v0: pl.Scalar[pl.INDEX],
        inner_cos_il__rv_v2: pl.Tensor[[T_DYN, 64], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)],
        inner_sin_signed__rv_v2: pl.Tensor[[T_DYN, 64], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)],
        pooled_kv_inline461_inline2039__rv_v2: pl.Tensor[[384, 128], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 196608)],
        norm_w_2d_inline465_inline1970__ssa_v0: pl.Tensor[[1, 128], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 512)],
        position_ids__ssa_v0: pl.Tensor[[T_DYN], pl.INT32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 0)],
    ) -> pl.Tensor[[384, 128], pl.BF16]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        normed_kv_inline518__rv_v3: pl.Tensor[[384, 128], pl.BF16, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 98304)] = self.rmsnorm_rope(
            normed_kv_inline518__ssa_v1,
            rms_blocks_inline422_inline1996__ssa_v0,
            rms_workers_inline457_inline1967__ssa_v0,
            bs_inline423_inline1991__ssa_v0,
            inner_cos_il__rv_v2,
            inner_sin_signed__rv_v2,
            pooled_kv_inline461_inline2039__rv_v2,
            norm_w_2d_inline465_inline1970__ssa_v0,
            position_ids__ssa_v0,
            attrs={"arg_directions": [pl.adir.output_existing, pl.adir.scalar, pl.adir.scalar, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input]},
        )
        return normed_kv_inline518__ssa_v1

    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def rope_cs(
        rope_swap_idx_inline2361__ssa_v0: pl.Out[pl.Tensor[[16, 64], pl.INT32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 4096)]],
        rope_cos_il_inline2281__ssa_v0: pl.Out[pl.Tensor[[384, 64], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 98304)]],
        rope_sin_signed_inline2280__ssa_v0: pl.Out[pl.Tensor[[384, 64], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 98304)]],
        rope_cs_blocks_inline2348__ssa_v0: pl.Scalar[pl.INDEX],
        freqs_cos__ssa_v0: pl.Tensor[[T_DYN, 64], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)],
        freqs_sin__ssa_v0: pl.Tensor[[T_DYN, 64], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 0)],
    ) -> tuple[pl.Tensor[[16, 64], pl.INT32], pl.Tensor[[384, 64], pl.FP32], pl.Tensor[[384, 64], pl.FP32]]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        mem_vec_5: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 4096)
        mem_vec_10: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 4096)
        mem_vec_27: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 2048)
        mem_vec_33: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 1024)
        mem_vec_34: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 1024)
        mem_vec_36: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        mem_vec_37: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        sw_ones_inline2283__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_5, pl.const(8704, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.full([16, 64], dtype=pl.FP32, value=1.0)
        t__ci_tmp_v0: pl.Tile[[1, 192], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 768), pl.Mem.Vec] = pl.tile.create([1, 192], dtype=pl.FP32, target_memory=pl.Mem.Vec)
        t__tile: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_27, pl.const(4096, pl.INT64), 256), pl.Mem.Vec] = pl.tile.ci(
            pl.const(0, pl.INT32), [1, 64], tmp=t__ci_tmp_v0, dtype=pl.INT32, descending=False
        )
        sw_idx_f_inline2300__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 256), pl.Mem.Vec] = pl.tile.cast(t__tile, target_type=pl.FP32, mode="round")
        sw_col_inline2278__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_5, pl.const(8704, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.col_expand_mul(
            sw_ones_inline2283__tile, sw_idx_f_inline2300__tile
        )
        t__tile_1: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.muls(sw_col_inline2278__tile, 0.5)
        sw_dup_i32_inline2372__tile: pl.Tile[[16, 64], pl.INT32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(t__tile_1, target_type=pl.INT32, mode="trunc")
        sw_dup_f_inline2277__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(
            sw_dup_i32_inline2372__tile, target_type=pl.FP32, mode="round"
        )
        t__tile_2: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.muls(sw_dup_f_inline2277__tile, 2.0)
        sw_lane_inline2275__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.sub(sw_col_inline2278__tile, t__tile_2)
        t__tile_3: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_5, pl.const(8704, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.adds(sw_col_inline2278__tile, 1.0)
        t__tile_4: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.muls(sw_lane_inline2275__tile, 2.0)
        sw_swap_f_inline2343__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_5, pl.const(8704, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.sub(t__tile_3, t__tile_4)
        t__tile_5: pl.Tile[[16, 64], pl.INT32, pl.MemRef(mem_vec_5, pl.const(8704, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(sw_swap_f_inline2343__tile, target_type=pl.INT32, mode="round")
        rope_swap_idx_inline2361__tile: pl.Tensor[[16, 64], pl.INT32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 4096)] = pl.tile.store(t__tile_5, [0, 0], rope_swap_idx_inline2361__ssa_v0)
        cs_ones_inline2327__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_5, pl.const(8704, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.full([8, 64], dtype=pl.FP32, value=1.0)
        t__ci_tmp_v1: pl.Tile[[1, 192], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 768), pl.Mem.Vec] = pl.tile.create([1, 192], dtype=pl.FP32, target_memory=pl.Mem.Vec)
        t__tile_6: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_27, pl.const(4096, pl.INT64), 256), pl.Mem.Vec] = pl.tile.ci(
            pl.const(0, pl.INT32), [1, 64], tmp=t__ci_tmp_v1, dtype=pl.INT32, descending=False
        )
        cs_idx_f_inline2419__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 256), pl.Mem.Vec] = pl.tile.cast(t__tile_6, target_type=pl.FP32, mode="round")
        cs_col_inline2274__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_5, pl.const(8704, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.col_expand_mul(
            cs_ones_inline2327__tile, cs_idx_f_inline2419__tile
        )
        t__tile_7: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.muls(cs_col_inline2274__tile, 0.5)
        cs_dup_i32_inline2273__tile: pl.Tile[[8, 64], pl.INT32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(t__tile_7, target_type=pl.INT32, mode="trunc")
        cs_dup_f_inline2272__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(
            cs_dup_i32_inline2273__tile, target_type=pl.FP32, mode="round"
        )
        cs_dup_idx_inline2356__tile: pl.Tile[[8, 64], pl.INT32, pl.MemRef(mem_vec_27, pl.const(4096, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(
            cs_dup_f_inline2272__tile, target_type=pl.INT32, mode="round"
        )
        t__tile_8: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.muls(cs_dup_f_inline2272__tile, 2.0)
        cs_lane_inline2271__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_5, pl.const(8704, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.sub(cs_col_inline2274__tile, t__tile_8)
        t__tile_9: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_5, pl.const(8704, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.muls(cs_lane_inline2271__tile, 2.0)
        t__tile_10: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_5, pl.const(8704, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.subs(t__tile_9, 1.0)
        cs_sign_inline2276__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_5, pl.const(8704, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.neg(t__tile_10)
        for cs_rb_inline2295__idx_v0, (rope_cos_il_inline2281__iter_v1, rope_sin_signed_inline2280__iter_v1) in pl.range(
            rope_cs_blocks_inline2348__ssa_v0, init_values=(rope_cos_il_inline2281__ssa_v0, rope_sin_signed_inline2280__ssa_v0)
        ):
            cs_t0_inline2270__ssa_v0: pl.Scalar[pl.INDEX] = cs_rb_inline2295__idx_v0 * 8
            cs_cos_inline2326__tile: pl.Tile[[8, 32], pl.FP32, pl.MemRef(mem_vec_33, pl.const(6144, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.load(
                freqs_cos__ssa_v0, [cs_t0_inline2270__ssa_v0, 0], [8, 32], [8, 32], target_memory=pl.Mem.Vec
            )
            cs_sin_inline2371__tile: pl.Tile[[8, 32], pl.FP32, pl.MemRef(mem_vec_34, pl.const(7168, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.load(
                freqs_sin__ssa_v0, [cs_t0_inline2270__ssa_v0, 0], [8, 32], [8, 32], target_memory=pl.Mem.Vec
            )
            gather_acc_init: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.create([8, 64], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            for gather_lv, (gather_ia,) in pl.range(8, init_values=(gather_acc_init,)):
                gather_inp_row: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_33, pl.const(6144, pl.INT64), 128), pl.Mem.Vec] = pl.tile.slice(cs_cos_inline2326__tile, [1, 32], [gather_lv, 0], [1, 32])
                gather_idx_row: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_27, pl.const(4096, pl.INT64), 256), pl.Mem.Vec] = pl.tile.slice(
                    cs_dup_idx_inline2356__tile, [1, 64], [gather_lv, 0], [1, 64]
                )
                gather_row_tmp: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_36, pl.const(8192, pl.INT64), 256), pl.Mem.Vec] = pl.tile.create([1, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec)
                gather_row: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_37, pl.const(8448, pl.INT64), 256), pl.Mem.Vec] = pl.tile.gather(gather_inp_row, gather_idx_row, gather_row_tmp)
                gather_asmbl: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.assemble(gather_ia, gather_row, [gather_lv, 0])
                gather_rv: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 2048), pl.Mem.Vec] = pl.yield_(gather_asmbl)
            t__tile_11: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 2048), pl.Mem.Vec] = gather_rv
            rope_cos_il_inline2281__tile: pl.Tensor[[384, 64], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 98304)] = pl.tile.store(
                t__tile_11, [cs_t0_inline2270__ssa_v0, 0], rope_cos_il_inline2281__iter_v1
            )
            gather_acc_init_1: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.create([8, 64], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            for gather_lv_1, (gather_ia_1,) in pl.range(8, init_values=(gather_acc_init_1,)):
                gather_inp_row_1: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_34, pl.const(7168, pl.INT64), 128), pl.Mem.Vec] = pl.tile.slice(
                    cs_sin_inline2371__tile, [1, 32], [gather_lv_1, 0], [1, 32]
                )
                gather_idx_row_1: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_27, pl.const(4096, pl.INT64), 256), pl.Mem.Vec] = pl.tile.slice(
                    cs_dup_idx_inline2356__tile, [1, 64], [gather_lv_1, 0], [1, 64]
                )
                gather_row_tmp_1: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_33, pl.const(6144, pl.INT64), 256), pl.Mem.Vec] = pl.tile.create([1, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec)
                gather_row_1: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_36, pl.const(8192, pl.INT64), 256), pl.Mem.Vec] = pl.tile.gather(gather_inp_row_1, gather_idx_row_1, gather_row_tmp_1)
                gather_asmbl_1: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.assemble(gather_ia_1, gather_row_1, [gather_lv_1, 0])
                gather_rv_1: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 2048), pl.Mem.Vec] = pl.yield_(gather_asmbl_1)
            cs_sin_il_inline2349__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 2048), pl.Mem.Vec] = gather_rv_1
            t__tile_12: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.mul(cs_sin_il_inline2349__tile, cs_sign_inline2276__tile)
            rope_sin_signed_inline2280__tile: pl.Tensor[[384, 64], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 98304)] = pl.tile.store(
                t__tile_12, [cs_t0_inline2270__ssa_v0, 0], rope_sin_signed_inline2280__iter_v1
            )
            rope_cos_il_inline2281__rv_v2, rope_sin_signed_inline2280__rv_v2 = pl.yield_(rope_cos_il_inline2281__tile, rope_sin_signed_inline2280__tile)
        return rope_swap_idx_inline2361__ssa_v0, rope_cos_il_inline2281__ssa_v0, rope_sin_signed_inline2280__ssa_v0

    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def scatter_softmax_pool(
        pooled_kv_inline511__ssa_v0: pl.Out[pl.Tensor[[384, 512], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 786432)]],
        b_dim_inline706_inline1830__ssa_v0: pl.Scalar[pl.INDEX],
        pool_workers_inline692_inline1827__ssa_v0: pl.Scalar[pl.INDEX],
        position_ids__ssa_v0: pl.Tensor[[T_DYN], pl.INT32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)],
        s_dim_inline700_inline1846__ssa_v0: pl.Scalar[pl.INDEX],
        cmp4_score_proj_pad_inline702_inline1863__ssa_v0: pl.Tensor[[384, 1024], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 1572864)],
        cmp_ape__ssa_v0: pl.Tensor[[4, 1024], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 16384)],
        cmp4_kv_proj_pad_inline708_inline1836__ssa_v0: pl.Tensor[[384, 1024], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 1572864)],
        compress_state_block_table__ssa_v0: pl.Tensor[[B_DYN, 7], pl.INT32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 0)],
        compress_state_flat_inline699_inline1828__ssa_v0: pl.Tensor[[compress_state_rows_inline693_inline1838__ssa_v0, 2048], pl.FP32, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 0)],
    ) -> pl.Tensor[[384, 512], pl.FP32]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        mem_vec_7: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 2048)
        mem_vec_9: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 2048)
        mem_vec_13: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 2048)
        mem_vec_14: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 2048)
        mem_vec_15: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 2048)
        mem_vec_16: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 2048)
        mem_vec_17: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 2048)
        mem_vec_24: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 2048)
        pool_worker_inline690_inline1850__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        for c_idx_inline698_inline1862__idx_v0, (pooled_kv_inline511__iter_v1,) in pl.range(
            pool_worker_inline690_inline1850__ssa_v0, b_dim_inline706_inline1830__ssa_v0, pool_workers_inline692_inline1827__ssa_v0, init_values=(pooled_kv_inline511__ssa_v0,)
        ):
            first_pos_b_inline709_inline1868__tile: pl.Scalar[pl.INT32] = pl.tensor.read(position_ids__ssa_v0, [c_idx_inline698_inline1862__idx_v0 * s_dim_inline700_inline1846__ssa_v0])
            for s_idx_inline710_inline1847__idx_v0, (pooled_kv_inline511__iter_v3,) in pl.range(s_dim_inline700_inline1846__ssa_v0, init_values=(pooled_kv_inline511__iter_v1,)):
                token_inline691_inline1869__ssa_v0: pl.Scalar[pl.INDEX] = c_idx_inline698_inline1862__idx_v0 * s_dim_inline700_inline1846__ssa_v0 + s_idx_inline710_inline1847__idx_v0
                token_pos_inline711_inline1871__tile: pl.Scalar[pl.INT32] = pl.tensor.read(position_ids__ssa_v0, [token_inline691_inline1869__ssa_v0])
                t__tile: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_7, pl.const(12288, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.full([1, 512], dtype=pl.FP32, value=0.0)
                pooled_kv_inline511__tile: pl.Tensor[[384, 512], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 786432)] = pl.tile.store(
                    t__tile, [token_inline691_inline1869__ssa_v0, 0], pooled_kv_inline511__iter_v3
                )
                if (pl.cast(token_pos_inline711_inline1871__tile, pl.INDEX) + 1) % 4 == 0:
                    window_start_inline697_inline1835__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(token_pos_inline711_inline1871__tile, pl.INDEX) - 8 + 1
                    last_ape_row_inline713_inline1872__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(pl.cast(token_pos_inline711_inline1871__tile, pl.INDEX) % 4, pl.INDEX)
                    t__tile_1: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_7, pl.const(12288, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.load(
                        cmp4_score_proj_pad_inline702_inline1863__ssa_v0, [token_inline691_inline1869__ssa_v0, 512], [1, 512], [1, 512], target_memory=pl.Mem.Vec
                    )
                    t__tile_2: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_9, pl.const(14336, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.load(
                        cmp_ape__ssa_v0, [last_ape_row_inline713_inline1872__ssa_v0, 512], [1, 512], [1, 512], target_memory=pl.Mem.Vec
                    )
                    mi_inline707_inline1874__tile: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_7, pl.const(12288, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.add(t__tile_1, t__tile_2)
                    t__tile_3: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_9, pl.const(14336, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.sub(mi_inline707_inline1874__tile, mi_inline707_inline1874__tile)
                    li_inline703_inline1875__tile: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_9, pl.const(14336, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.exp(t__tile_3)
                    oi_inline714_inline1876__tile: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_13, pl.const(0, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.load(
                        cmp4_kv_proj_pad_inline708_inline1836__ssa_v0, [token_inline691_inline1869__ssa_v0, 512], [1, 512], [1, 512], target_memory=pl.Mem.Vec
                    )
                    for state_idx_inline715_inline1877__idx_v0, (li_inline703_inline1875__iter_v1, mi_inline707_inline1874__iter_v1, oi_inline714_inline1876__iter_v1) in pl.range(
                        7, init_values=(li_inline703_inline1875__tile, mi_inline707_inline1874__tile, oi_inline714_inline1876__tile)
                    ):
                        logical_pos_inline694_inline1878__ssa_v0: pl.Scalar[pl.INDEX] = window_start_inline697_inline1835__ssa_v0 + state_idx_inline715_inline1877__idx_v0
                        value_inline705_inline1879__tile: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_14, pl.const(2048, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.full(
                            [1, 512], dtype=pl.FP32, value=0.0
                        )
                        score_inline717_inline1881__tile: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_15, pl.const(4096, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.full(
                            [1, 512], dtype=pl.FP32, value=-3.4028234663852886e38
                        )
                        state_half_inline718_inline1851__ssa_v0: pl.Scalar[pl.INDEX] = 0
                        if 4 <= state_idx_inline715_inline1877__idx_v0:
                            state_half_inline718_inline1851__ssa_v1: pl.Scalar[pl.INDEX] = 512
                            state_half_inline718_inline1851__phi_v2: pl.Scalar[pl.INDEX] = pl.yield_(state_half_inline718_inline1851__ssa_v1)
                        else:
                            state_half_inline718_inline1851__phi_v2: pl.Scalar[pl.INDEX] = pl.yield_(state_half_inline718_inline1851__ssa_v0)
                        if 0 <= logical_pos_inline694_inline1878__ssa_v0 and logical_pos_inline694_inline1878__ssa_v0 < pl.cast(first_pos_b_inline709_inline1868__tile, pl.INDEX):
                            ring_row_inline689_inline1845__ssa_v0: pl.Scalar[pl.INDEX] = logical_pos_inline694_inline1878__ssa_v0 % 14
                            state_page_off_inline688_inline1858__ssa_v0: pl.Scalar[pl.INDEX] = ring_row_inline689_inline1845__ssa_v0 // 2
                            state_blk_id_i32_inline687_inline1866__tile: pl.Scalar[pl.INT32] = pl.tensor.read(
                                compress_state_block_table__ssa_v0, [c_idx_inline698_inline1862__idx_v0, state_page_off_inline688_inline1858__ssa_v0]
                            )
                            if 0 <= pl.cast(state_blk_id_i32_inline687_inline1866__tile, pl.INDEX):
                                state_blk_id_inline685_inline1883__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(state_blk_id_i32_inline687_inline1866__tile, pl.INDEX)
                                state_intra_row_inline683_inline1855__ssa_v0: pl.Scalar[pl.INDEX] = ring_row_inline689_inline1845__ssa_v0 % 2
                                state_row_inline682_inline1849__ssa_v0: pl.Scalar[pl.INDEX] = state_blk_id_inline685_inline1883__ssa_v0 * 2 + state_intra_row_inline683_inline1855__ssa_v0
                                value_inline705_inline1879__tile_1: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_16, pl.const(6144, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.load(
                                    compress_state_flat_inline699_inline1828__ssa_v0,
                                    [state_row_inline682_inline1849__ssa_v0, state_half_inline718_inline1851__phi_v2],
                                    [1, 512],
                                    [1, 512],
                                    target_memory=pl.Mem.Vec,
                                )
                                score_inline717_inline1881__tile_1: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_17, pl.const(8192, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.load(
                                    compress_state_flat_inline699_inline1828__ssa_v0,
                                    [state_row_inline682_inline1849__ssa_v0, state_half_inline718_inline1851__phi_v2 + 1024],
                                    [1, 512],
                                    [1, 512],
                                    target_memory=pl.Mem.Vec,
                                )
                                score_inline717_inline1881__phi_v2, value_inline705_inline1879__phi_v2 = pl.yield_(score_inline717_inline1881__tile_1, value_inline705_inline1879__tile_1)
                            else:
                                score_inline717_inline1881__tile_mv: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_17, pl.const(8192, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.move(
                                    score_inline717_inline1881__tile, target_memory=pl.Mem.Vec
                                )
                                value_inline705_inline1879__tile_mv: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_16, pl.const(6144, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.move(
                                    value_inline705_inline1879__tile, target_memory=pl.Mem.Vec
                                )
                                score_inline717_inline1881__phi_v2, value_inline705_inline1879__phi_v2 = pl.yield_(score_inline717_inline1881__tile_mv, value_inline705_inline1879__tile_mv)
                            score_inline717_inline1881__phi_v3, value_inline705_inline1879__phi_v3 = pl.yield_(score_inline717_inline1881__phi_v2, value_inline705_inline1879__phi_v2)
                        else:
                            score_inline717_inline1881__tile_mv_1: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_17, pl.const(8192, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.move(
                                score_inline717_inline1881__tile, target_memory=pl.Mem.Vec
                            )
                            value_inline705_inline1879__tile_mv_1: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_16, pl.const(6144, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.move(
                                value_inline705_inline1879__tile, target_memory=pl.Mem.Vec
                            )
                            score_inline717_inline1881__phi_v3, value_inline705_inline1879__phi_v3 = pl.yield_(score_inline717_inline1881__tile_mv_1, value_inline705_inline1879__tile_mv_1)
                        if pl.cast(first_pos_b_inline709_inline1868__tile, pl.INDEX) <= logical_pos_inline694_inline1878__ssa_v0:
                            if logical_pos_inline694_inline1878__ssa_v0 <= pl.cast(token_pos_inline711_inline1871__tile, pl.INDEX):
                                overlay_token_inline681_inline1870__ssa_v0: pl.Scalar[pl.INDEX] = (
                                    c_idx_inline698_inline1862__idx_v0 * s_dim_inline700_inline1846__ssa_v0
                                    + logical_pos_inline694_inline1878__ssa_v0
                                    - pl.cast(first_pos_b_inline709_inline1868__tile, pl.INDEX)
                                )
                                ape_row_inline686_inline1852__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(logical_pos_inline694_inline1878__ssa_v0 % 4, pl.INDEX)
                                value_inline705_inline1879__tile_2: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_14, pl.const(2048, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.load(
                                    cmp4_kv_proj_pad_inline708_inline1836__ssa_v0,
                                    [overlay_token_inline681_inline1870__ssa_v0, state_half_inline718_inline1851__phi_v2],
                                    [1, 512],
                                    [1, 512],
                                    target_memory=pl.Mem.Vec,
                                )
                                t__tile_4: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_15, pl.const(4096, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.load(
                                    cmp4_score_proj_pad_inline702_inline1863__ssa_v0,
                                    [overlay_token_inline681_inline1870__ssa_v0, state_half_inline718_inline1851__phi_v2],
                                    [1, 512],
                                    [1, 512],
                                    target_memory=pl.Mem.Vec,
                                )
                                t__tile_5: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_24, pl.const(10240, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.load(
                                    cmp_ape__ssa_v0, [ape_row_inline686_inline1852__ssa_v0, state_half_inline718_inline1851__phi_v2], [1, 512], [1, 512], target_memory=pl.Mem.Vec
                                )
                                score_inline717_inline1881__tile_2: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_15, pl.const(4096, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.add(t__tile_4, t__tile_5)
                                score_inline717_inline1881__phi_v5, value_inline705_inline1879__phi_v5 = pl.yield_(score_inline717_inline1881__tile_2, value_inline705_inline1879__tile_2)
                            else:
                                score_inline717_inline1881__phi_v3_mv: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_15, pl.const(4096, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.move(
                                    score_inline717_inline1881__phi_v3, target_memory=pl.Mem.Vec
                                )
                                value_inline705_inline1879__phi_v3_mv: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_14, pl.const(2048, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.move(
                                    value_inline705_inline1879__phi_v3, target_memory=pl.Mem.Vec
                                )
                                score_inline717_inline1881__phi_v5, value_inline705_inline1879__phi_v5 = pl.yield_(score_inline717_inline1881__phi_v3_mv, value_inline705_inline1879__phi_v3_mv)
                            score_inline717_inline1881__phi_v6, value_inline705_inline1879__phi_v6 = pl.yield_(score_inline717_inline1881__phi_v5, value_inline705_inline1879__phi_v5)
                        else:
                            score_inline717_inline1881__phi_v3_mv_1: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_15, pl.const(4096, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.move(
                                score_inline717_inline1881__phi_v3, target_memory=pl.Mem.Vec
                            )
                            value_inline705_inline1879__phi_v3_mv_1: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_14, pl.const(2048, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.move(
                                value_inline705_inline1879__phi_v3, target_memory=pl.Mem.Vec
                            )
                            score_inline717_inline1881__phi_v6, value_inline705_inline1879__phi_v6 = pl.yield_(score_inline717_inline1881__phi_v3_mv_1, value_inline705_inline1879__phi_v3_mv_1)
                        mi_next_inline684_inline1873__tile: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_16, pl.const(6144, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.maximum(
                            mi_inline707_inline1874__iter_v1, score_inline717_inline1881__phi_v6
                        )
                        t__tile_6: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_17, pl.const(8192, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.sub(
                            mi_inline707_inline1874__iter_v1, mi_next_inline684_inline1873__tile
                        )
                        alpha_inline680_inline1880__tile: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_17, pl.const(8192, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.exp(t__tile_6)
                        t__tile_7: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_15, pl.const(4096, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.sub(
                            score_inline717_inline1881__phi_v6, mi_next_inline684_inline1873__tile
                        )
                        beta_inline704_inline1884__tile: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_15, pl.const(4096, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.exp(t__tile_7)
                        t__tile_8: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_24, pl.const(10240, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.mul(
                            alpha_inline680_inline1880__tile, li_inline703_inline1875__iter_v1
                        )
                        li_inline703_inline1875__tile_1: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_9, pl.const(14336, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.add(
                            t__tile_8, beta_inline704_inline1884__tile
                        )
                        t__tile_9: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_17, pl.const(8192, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.mul(
                            oi_inline714_inline1876__iter_v1, alpha_inline680_inline1880__tile
                        )
                        t__tile_10: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_14, pl.const(2048, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.mul(
                            value_inline705_inline1879__phi_v6, beta_inline704_inline1884__tile
                        )
                        oi_inline714_inline1876__tile_1: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_13, pl.const(0, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.add(t__tile_9, t__tile_10)
                        mi_inline707_inline1874__ssa_v3: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_16, pl.const(6144, pl.INT64), 2048), pl.Mem.Vec] = mi_next_inline684_inline1873__tile
                        mi_inline707_inline1874__ssa_v3_mv: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_7, pl.const(12288, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.move(
                            mi_inline707_inline1874__ssa_v3, target_memory=pl.Mem.Vec
                        )
                        li_inline703_inline1875__rv_v2, mi_inline707_inline1874__rv_v2, oi_inline714_inline1876__rv_v2 = pl.yield_(
                            li_inline703_inline1875__tile_1, mi_inline707_inline1874__ssa_v3_mv, oi_inline714_inline1876__tile_1
                        )
                    t__tile_11: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_7, pl.const(12288, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.div(
                        oi_inline714_inline1876__rv_v2, li_inline703_inline1875__rv_v2
                    )
                    pooled_kv_inline511__tile_1: pl.Tensor[[384, 512], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 786432)] = pl.tile.store(
                        t__tile_11, [token_inline691_inline1869__ssa_v0, 0], pooled_kv_inline511__tile
                    )
                    pooled_kv_inline511__phi_v9: pl.Tensor[[384, 512], pl.FP32, pl.MemRef("mem_ddr_44", pl.const(0, pl.INT64), 786432)] = pl.yield_(pooled_kv_inline511__tile_1)
                else:
                    pooled_kv_inline511__phi_v9: pl.Tensor[[384, 512], pl.FP32, pl.MemRef("mem_ddr_44", pl.const(0, pl.INT64), 786432)] = pl.yield_(pooled_kv_inline511__tile)
                pooled_kv_inline511__rv_v4: pl.Tensor[[384, 512], pl.FP32, pl.MemRef("mem_ddr_45", pl.const(0, pl.INT64), 786432)] = pl.yield_(pooled_kv_inline511__phi_v9)
            pooled_kv_inline511__rv_v2: pl.Tensor[[384, 512], pl.FP32, pl.MemRef("mem_ddr_46", pl.const(0, pl.INT64), 786432)] = pl.yield_(pooled_kv_inline511__rv_v4)
        return pooled_kv_inline511__ssa_v0

    @pl.function(type=pl.FunctionType.Spmd)
    def scatter_softmax_pool_spmd(
        self,
        pooled_kv_inline511__ssa_v0: pl.Out[pl.Tensor[[384, 512], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 786432)]],
        b_dim_inline706_inline1830__ssa_v0: pl.Scalar[pl.INDEX],
        pool_workers_inline692_inline1827__ssa_v0: pl.Scalar[pl.INDEX],
        position_ids__ssa_v0: pl.Tensor[[T_DYN], pl.INT32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)],
        s_dim_inline700_inline1846__ssa_v0: pl.Scalar[pl.INDEX],
        cmp4_score_proj_pad_inline702_inline1863__ssa_v0: pl.Tensor[[384, 1024], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 1572864)],
        cmp_ape__ssa_v0: pl.Tensor[[4, 1024], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 16384)],
        cmp4_kv_proj_pad_inline708_inline1836__ssa_v0: pl.Tensor[[384, 1024], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 1572864)],
        compress_state_block_table__ssa_v0: pl.Tensor[[B_DYN, 7], pl.INT32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 0)],
        compress_state_flat_inline699_inline1828__ssa_v0: pl.Tensor[[compress_state_rows_inline693_inline1838__ssa_v0, 2048], pl.FP32, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 0)],
    ) -> pl.Tensor[[384, 512], pl.FP32]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        pooled_kv_inline511__rv_v2: pl.Tensor[[384, 512], pl.FP32, pl.MemRef("mem_ddr_7", pl.const(0, pl.INT64), 786432)] = self.scatter_softmax_pool(
            pooled_kv_inline511__ssa_v0,
            b_dim_inline706_inline1830__ssa_v0,
            pool_workers_inline692_inline1827__ssa_v0,
            position_ids__ssa_v0,
            s_dim_inline700_inline1846__ssa_v0,
            cmp4_score_proj_pad_inline702_inline1863__ssa_v0,
            cmp_ape__ssa_v0,
            cmp4_kv_proj_pad_inline708_inline1836__ssa_v0,
            compress_state_block_table__ssa_v0,
            compress_state_flat_inline699_inline1828__ssa_v0,
            attrs={
                "arg_directions": [pl.adir.output_existing, pl.adir.scalar, pl.adir.scalar, pl.adir.input, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input]
            },
        )
        return pooled_kv_inline511__ssa_v0

    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def scatter_softmax_pool_0(
        pooled_kv_inline461_inline2039__ssa_v0: pl.Out[pl.Tensor[[384, 128], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 196608)]],
        b_dim_inline445_inline1977__ssa_v0: pl.Scalar[pl.INDEX],
        pool_workers_inline435_inline1973__ssa_v0: pl.Scalar[pl.INDEX],
        position_ids__ssa_v0: pl.Tensor[[T_DYN], pl.INT32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)],
        s_dim_inline427_inline2038__ssa_v0: pl.Scalar[pl.INDEX],
        score_proj_pad_inline1993__rv_v2: pl.Tensor[[384, 256], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 393216)],
        inner_ape__ssa_v0: pl.Tensor[[4, 256], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 4096)],
        kv_proj_pad_inline1986__rv_v2: pl.Tensor[[384, 256], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 393216)],
        inner_compress_state_block_table__ssa_v0: pl.Tensor[[B_DYN, 7], pl.INT32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 0)],
        compress_state_flat_inline440_inline1999__ssa_v0: pl.Tensor[[compress_state_rows_inline425_inline2017__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 0)],
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
        pool_worker_inline459_inline2021__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        for c_idx_inline431_inline2023__idx_v0, (pooled_kv_inline461_inline2039__iter_v1,) in pl.range(
            pool_worker_inline459_inline2021__ssa_v0, b_dim_inline445_inline1977__ssa_v0, pool_workers_inline435_inline1973__ssa_v0, init_values=(pooled_kv_inline461_inline2039__ssa_v0,)
        ):
            first_pos_b_inline429_inline1979__tile: pl.Scalar[pl.INT32] = pl.tensor.read(position_ids__ssa_v0, [c_idx_inline431_inline2023__idx_v0 * s_dim_inline427_inline2038__ssa_v0])
            for s_idx_inline419_inline1980__idx_v0, (pooled_kv_inline461_inline2039__iter_v3,) in pl.range(s_dim_inline427_inline2038__ssa_v0, init_values=(pooled_kv_inline461_inline2039__iter_v1,)):
                token_inline428_inline2029__ssa_v0: pl.Scalar[pl.INDEX] = c_idx_inline431_inline2023__idx_v0 * s_dim_inline427_inline2038__ssa_v0 + s_idx_inline419_inline1980__idx_v0
                token_pos_inline437_inline2007__tile: pl.Scalar[pl.INT32] = pl.tensor.read(position_ids__ssa_v0, [token_inline428_inline2029__ssa_v0])
                t__tile: pl.Tile[[1, 128], pl.FP32, pl.MemRef(mem_vec_7, pl.const(1536, pl.INT64), 512), pl.Mem.Vec] = pl.tile.full([1, 128], dtype=pl.FP32, value=0.0)
                pooled_kv_inline461_inline2039__tile: pl.Tensor[[384, 128], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 196608)] = pl.tile.store(
                    t__tile, [token_inline428_inline2029__ssa_v0, 0], pooled_kv_inline461_inline2039__iter_v3
                )
                if (pl.cast(token_pos_inline437_inline2007__tile, pl.INDEX) + 1) % 4 == 0:
                    window_start_inline415_inline1974__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(token_pos_inline437_inline2007__tile, pl.INDEX) - 8 + 1
                    for h0_inline413_inline2030__idx_v0, (pooled_kv_inline461_inline2039__iter_v6,) in pl.range(0, 128, 64, init_values=(pooled_kv_inline461_inline2039__tile,)):
                        last_ape_row_inline410_inline2032__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(pl.cast(token_pos_inline437_inline2007__tile, pl.INDEX) % 4, pl.INDEX)
                        t__tile_1: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_7, pl.const(1536, pl.INT64), 256), pl.Mem.Vec] = pl.tile.load(
                            score_proj_pad_inline1993__rv_v2, [token_inline428_inline2029__ssa_v0, h0_inline413_inline2030__idx_v0 + 128], [1, 64], [1, 64], target_memory=pl.Mem.Vec
                        )
                        t__tile_2: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_9, pl.const(2048, pl.INT64), 256), pl.Mem.Vec] = pl.tile.load(
                            inner_ape__ssa_v0, [last_ape_row_inline410_inline2032__ssa_v0, h0_inline413_inline2030__idx_v0 + 128], [1, 64], [1, 64], target_memory=pl.Mem.Vec
                        )
                        mi_inline421_inline2024__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_7, pl.const(1536, pl.INT64), 256), pl.Mem.Vec] = pl.tile.add(t__tile_1, t__tile_2)
                        t__tile_3: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_9, pl.const(2048, pl.INT64), 256), pl.Mem.Vec] = pl.tile.sub(
                            mi_inline421_inline2024__tile, mi_inline421_inline2024__tile
                        )
                        li_inline432_inline2001__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_9, pl.const(2048, pl.INT64), 256), pl.Mem.Vec] = pl.tile.exp(t__tile_3)
                        oi_inline426_inline1976__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(0, pl.INT64), 256), pl.Mem.Vec] = pl.tile.load(
                            kv_proj_pad_inline1986__rv_v2, [token_inline428_inline2029__ssa_v0, h0_inline413_inline2030__idx_v0 + 128], [1, 64], [1, 64], target_memory=pl.Mem.Vec
                        )
                        for state_idx_inline408_inline2034__idx_v0, (li_inline432_inline2001__iter_v1, mi_inline421_inline2024__iter_v1, oi_inline426_inline1976__iter_v1) in pl.range(
                            7, init_values=(li_inline432_inline2001__tile, mi_inline421_inline2024__tile, oi_inline426_inline1976__tile)
                        ):
                            logical_pos_inline447_inline2011__ssa_v0: pl.Scalar[pl.INDEX] = window_start_inline415_inline1974__ssa_v0 + state_idx_inline408_inline2034__idx_v0
                            value_inline439_inline2025__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_14, pl.const(256, pl.INT64), 256), pl.Mem.Vec] = pl.tile.full(
                                [1, 64], dtype=pl.FP32, value=0.0
                            )
                            score_inline434_inline2037__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_15, pl.const(512, pl.INT64), 256), pl.Mem.Vec] = pl.tile.full(
                                [1, 64], dtype=pl.FP32, value=-3.4028234663852886e38
                            )
                            state_half_inline405_inline2040__ssa_v0: pl.Scalar[pl.INDEX] = 0
                            if 4 <= state_idx_inline408_inline2034__idx_v0:
                                state_half_inline405_inline2040__ssa_v1: pl.Scalar[pl.INDEX] = 128
                                state_half_inline405_inline2040__phi_v2: pl.Scalar[pl.INDEX] = pl.yield_(state_half_inline405_inline2040__ssa_v1)
                            else:
                                state_half_inline405_inline2040__phi_v2: pl.Scalar[pl.INDEX] = pl.yield_(state_half_inline405_inline2040__ssa_v0)
                            if 0 <= logical_pos_inline447_inline2011__ssa_v0 and logical_pos_inline447_inline2011__ssa_v0 < pl.cast(first_pos_b_inline429_inline1979__tile, pl.INDEX):
                                ring_row_inline443_inline1987__ssa_v0: pl.Scalar[pl.INDEX] = logical_pos_inline447_inline2011__ssa_v0 % 14
                                state_page_off_inline450_inline2041__ssa_v0: pl.Scalar[pl.INDEX] = ring_row_inline443_inline1987__ssa_v0 // 2
                                state_blk_id_i32_inline442_inline2019__tile: pl.Scalar[pl.INT32] = pl.tensor.read(
                                    inner_compress_state_block_table__ssa_v0, [c_idx_inline431_inline2023__idx_v0, state_page_off_inline450_inline2041__ssa_v0]
                                )
                                if 0 <= pl.cast(state_blk_id_i32_inline442_inline2019__tile, pl.INDEX):
                                    state_blk_id_inline446_inline1994__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(state_blk_id_i32_inline442_inline2019__tile, pl.INDEX)
                                    state_intra_row_inline448_inline1992__ssa_v0: pl.Scalar[pl.INDEX] = ring_row_inline443_inline1987__ssa_v0 % 2
                                    state_row_inline451_inline2035__ssa_v0: pl.Scalar[pl.INDEX] = state_blk_id_inline446_inline1994__ssa_v0 * 2 + state_intra_row_inline448_inline1992__ssa_v0
                                    value_inline439_inline2025__tile_1: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_16, pl.const(768, pl.INT64), 256), pl.Mem.Vec] = pl.tile.load(
                                        compress_state_flat_inline440_inline1999__ssa_v0,
                                        [state_row_inline451_inline2035__ssa_v0, state_half_inline405_inline2040__phi_v2 + h0_inline413_inline2030__idx_v0],
                                        [1, 64],
                                        [1, 64],
                                        target_memory=pl.Mem.Vec,
                                    )
                                    score_inline434_inline2037__tile_1: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_17, pl.const(1024, pl.INT64), 256), pl.Mem.Vec] = pl.tile.load(
                                        compress_state_flat_inline440_inline1999__ssa_v0,
                                        [state_row_inline451_inline2035__ssa_v0, state_half_inline405_inline2040__phi_v2 + h0_inline413_inline2030__idx_v0 + 256],
                                        [1, 64],
                                        [1, 64],
                                        target_memory=pl.Mem.Vec,
                                    )
                                    score_inline434_inline2037__phi_v2, value_inline439_inline2025__phi_v2 = pl.yield_(score_inline434_inline2037__tile_1, value_inline439_inline2025__tile_1)
                                else:
                                    score_inline434_inline2037__tile_mv: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_17, pl.const(1024, pl.INT64), 256), pl.Mem.Vec] = pl.tile.move(
                                        score_inline434_inline2037__tile, target_memory=pl.Mem.Vec
                                    )
                                    value_inline439_inline2025__tile_mv: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_16, pl.const(768, pl.INT64), 256), pl.Mem.Vec] = pl.tile.move(
                                        value_inline439_inline2025__tile, target_memory=pl.Mem.Vec
                                    )
                                    score_inline434_inline2037__phi_v2, value_inline439_inline2025__phi_v2 = pl.yield_(score_inline434_inline2037__tile_mv, value_inline439_inline2025__tile_mv)
                                score_inline434_inline2037__phi_v3, value_inline439_inline2025__phi_v3 = pl.yield_(score_inline434_inline2037__phi_v2, value_inline439_inline2025__phi_v2)
                            else:
                                score_inline434_inline2037__tile_mv_1: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_17, pl.const(1024, pl.INT64), 256), pl.Mem.Vec] = pl.tile.move(
                                    score_inline434_inline2037__tile, target_memory=pl.Mem.Vec
                                )
                                value_inline439_inline2025__tile_mv_1: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_16, pl.const(768, pl.INT64), 256), pl.Mem.Vec] = pl.tile.move(
                                    value_inline439_inline2025__tile, target_memory=pl.Mem.Vec
                                )
                                score_inline434_inline2037__phi_v3, value_inline439_inline2025__phi_v3 = pl.yield_(score_inline434_inline2037__tile_mv_1, value_inline439_inline2025__tile_mv_1)
                            if pl.cast(first_pos_b_inline429_inline1979__tile, pl.INDEX) <= logical_pos_inline447_inline2011__ssa_v0:
                                if logical_pos_inline447_inline2011__ssa_v0 <= pl.cast(token_pos_inline437_inline2007__tile, pl.INDEX):
                                    overlay_token_inline444_inline2042__ssa_v0: pl.Scalar[pl.INDEX] = (
                                        c_idx_inline431_inline2023__idx_v0 * s_dim_inline427_inline2038__ssa_v0
                                        + logical_pos_inline447_inline2011__ssa_v0
                                        - pl.cast(first_pos_b_inline429_inline1979__tile, pl.INDEX)
                                    )
                                    ape_row_inline454_inline2027__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(logical_pos_inline447_inline2011__ssa_v0 % 4, pl.INDEX)
                                    value_inline439_inline2025__tile_2: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_14, pl.const(256, pl.INT64), 256), pl.Mem.Vec] = pl.tile.load(
                                        kv_proj_pad_inline1986__rv_v2,
                                        [overlay_token_inline444_inline2042__ssa_v0, state_half_inline405_inline2040__phi_v2 + h0_inline413_inline2030__idx_v0],
                                        [1, 64],
                                        [1, 64],
                                        target_memory=pl.Mem.Vec,
                                    )
                                    t__tile_4: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_15, pl.const(512, pl.INT64), 256), pl.Mem.Vec] = pl.tile.load(
                                        score_proj_pad_inline1993__rv_v2,
                                        [overlay_token_inline444_inline2042__ssa_v0, state_half_inline405_inline2040__phi_v2 + h0_inline413_inline2030__idx_v0],
                                        [1, 64],
                                        [1, 64],
                                        target_memory=pl.Mem.Vec,
                                    )
                                    t__tile_5: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_24, pl.const(1280, pl.INT64), 256), pl.Mem.Vec] = pl.tile.load(
                                        inner_ape__ssa_v0,
                                        [ape_row_inline454_inline2027__ssa_v0, state_half_inline405_inline2040__phi_v2 + h0_inline413_inline2030__idx_v0],
                                        [1, 64],
                                        [1, 64],
                                        target_memory=pl.Mem.Vec,
                                    )
                                    score_inline434_inline2037__tile_2: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_15, pl.const(512, pl.INT64), 256), pl.Mem.Vec] = pl.tile.add(t__tile_4, t__tile_5)
                                    score_inline434_inline2037__phi_v5, value_inline439_inline2025__phi_v5 = pl.yield_(score_inline434_inline2037__tile_2, value_inline439_inline2025__tile_2)
                                else:
                                    score_inline434_inline2037__phi_v3_mv: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_15, pl.const(512, pl.INT64), 256), pl.Mem.Vec] = pl.tile.move(
                                        score_inline434_inline2037__phi_v3, target_memory=pl.Mem.Vec
                                    )
                                    value_inline439_inline2025__phi_v3_mv: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_14, pl.const(256, pl.INT64), 256), pl.Mem.Vec] = pl.tile.move(
                                        value_inline439_inline2025__phi_v3, target_memory=pl.Mem.Vec
                                    )
                                    score_inline434_inline2037__phi_v5, value_inline439_inline2025__phi_v5 = pl.yield_(score_inline434_inline2037__phi_v3_mv, value_inline439_inline2025__phi_v3_mv)
                                score_inline434_inline2037__phi_v6, value_inline439_inline2025__phi_v6 = pl.yield_(score_inline434_inline2037__phi_v5, value_inline439_inline2025__phi_v5)
                            else:
                                score_inline434_inline2037__phi_v3_mv_1: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_15, pl.const(512, pl.INT64), 256), pl.Mem.Vec] = pl.tile.move(
                                    score_inline434_inline2037__phi_v3, target_memory=pl.Mem.Vec
                                )
                                value_inline439_inline2025__phi_v3_mv_1: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_14, pl.const(256, pl.INT64), 256), pl.Mem.Vec] = pl.tile.move(
                                    value_inline439_inline2025__phi_v3, target_memory=pl.Mem.Vec
                                )
                                score_inline434_inline2037__phi_v6, value_inline439_inline2025__phi_v6 = pl.yield_(score_inline434_inline2037__phi_v3_mv_1, value_inline439_inline2025__phi_v3_mv_1)
                            mi_next_inline455_inline2044__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_16, pl.const(768, pl.INT64), 256), pl.Mem.Vec] = pl.tile.maximum(
                                mi_inline421_inline2024__iter_v1, score_inline434_inline2037__phi_v6
                            )
                            t__tile_6: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_17, pl.const(1024, pl.INT64), 256), pl.Mem.Vec] = pl.tile.sub(
                                mi_inline421_inline2024__iter_v1, mi_next_inline455_inline2044__tile
                            )
                            alpha_inline433_inline1985__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_17, pl.const(1024, pl.INT64), 256), pl.Mem.Vec] = pl.tile.exp(t__tile_6)
                            t__tile_7: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_15, pl.const(512, pl.INT64), 256), pl.Mem.Vec] = pl.tile.sub(
                                score_inline434_inline2037__phi_v6, mi_next_inline455_inline2044__tile
                            )
                            beta_inline460_inline2045__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_15, pl.const(512, pl.INT64), 256), pl.Mem.Vec] = pl.tile.exp(t__tile_7)
                            t__tile_8: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_24, pl.const(1280, pl.INT64), 256), pl.Mem.Vec] = pl.tile.mul(
                                alpha_inline433_inline1985__tile, li_inline432_inline2001__iter_v1
                            )
                            li_inline432_inline2001__tile_1: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_9, pl.const(2048, pl.INT64), 256), pl.Mem.Vec] = pl.tile.add(
                                t__tile_8, beta_inline460_inline2045__tile
                            )
                            t__tile_9: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_17, pl.const(1024, pl.INT64), 256), pl.Mem.Vec] = pl.tile.mul(
                                oi_inline426_inline1976__iter_v1, alpha_inline433_inline1985__tile
                            )
                            t__tile_10: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_14, pl.const(256, pl.INT64), 256), pl.Mem.Vec] = pl.tile.mul(
                                value_inline439_inline2025__phi_v6, beta_inline460_inline2045__tile
                            )
                            oi_inline426_inline1976__tile_1: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(0, pl.INT64), 256), pl.Mem.Vec] = pl.tile.add(t__tile_9, t__tile_10)
                            mi_inline421_inline2024__ssa_v3: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_16, pl.const(768, pl.INT64), 256), pl.Mem.Vec] = mi_next_inline455_inline2044__tile
                            mi_inline421_inline2024__ssa_v3_mv: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_7, pl.const(1536, pl.INT64), 256), pl.Mem.Vec] = pl.tile.move(
                                mi_inline421_inline2024__ssa_v3, target_memory=pl.Mem.Vec
                            )
                            li_inline432_inline2001__rv_v2, mi_inline421_inline2024__rv_v2, oi_inline426_inline1976__rv_v2 = pl.yield_(
                                li_inline432_inline2001__tile_1, mi_inline421_inline2024__ssa_v3_mv, oi_inline426_inline1976__tile_1
                            )
                        t__tile_11: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_7, pl.const(1536, pl.INT64), 256), pl.Mem.Vec] = pl.tile.div(
                            oi_inline426_inline1976__rv_v2, li_inline432_inline2001__rv_v2
                        )
                        pooled_kv_inline461_inline2039__tile_1: pl.Tensor[[384, 128], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 196608)] = pl.tile.store(
                            t__tile_11, [token_inline428_inline2029__ssa_v0, h0_inline413_inline2030__idx_v0], pooled_kv_inline461_inline2039__iter_v6
                        )
                        pooled_kv_inline461_inline2039__rv_v7: pl.Tensor[[384, 128], pl.FP32, pl.MemRef("mem_ddr_44", pl.const(0, pl.INT64), 196608)] = pl.yield_(
                            pooled_kv_inline461_inline2039__tile_1
                        )
                    pooled_kv_inline461_inline2039__phi_v9: pl.Tensor[[384, 128], pl.FP32, pl.MemRef("mem_ddr_45", pl.const(0, pl.INT64), 196608)] = pl.yield_(pooled_kv_inline461_inline2039__rv_v7)
                else:
                    pooled_kv_inline461_inline2039__phi_v9: pl.Tensor[[384, 128], pl.FP32, pl.MemRef("mem_ddr_45", pl.const(0, pl.INT64), 196608)] = pl.yield_(pooled_kv_inline461_inline2039__tile)
                pooled_kv_inline461_inline2039__rv_v4: pl.Tensor[[384, 128], pl.FP32, pl.MemRef("mem_ddr_46", pl.const(0, pl.INT64), 196608)] = pl.yield_(pooled_kv_inline461_inline2039__phi_v9)
            pooled_kv_inline461_inline2039__rv_v2: pl.Tensor[[384, 128], pl.FP32, pl.MemRef("mem_ddr_47", pl.const(0, pl.INT64), 196608)] = pl.yield_(pooled_kv_inline461_inline2039__rv_v4)
        return pooled_kv_inline461_inline2039__ssa_v0

    @pl.function(type=pl.FunctionType.Spmd)
    def scatter_softmax_pool_spmd_0(
        self,
        pooled_kv_inline461_inline2039__ssa_v0: pl.Out[pl.Tensor[[384, 128], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 196608)]],
        b_dim_inline445_inline1977__ssa_v0: pl.Scalar[pl.INDEX],
        pool_workers_inline435_inline1973__ssa_v0: pl.Scalar[pl.INDEX],
        position_ids__ssa_v0: pl.Tensor[[T_DYN], pl.INT32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)],
        s_dim_inline427_inline2038__ssa_v0: pl.Scalar[pl.INDEX],
        score_proj_pad_inline1993__rv_v2: pl.Tensor[[384, 256], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 393216)],
        inner_ape__ssa_v0: pl.Tensor[[4, 256], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 4096)],
        kv_proj_pad_inline1986__rv_v2: pl.Tensor[[384, 256], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 393216)],
        inner_compress_state_block_table__ssa_v0: pl.Tensor[[B_DYN, 7], pl.INT32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 0)],
        compress_state_flat_inline440_inline1999__ssa_v0: pl.Tensor[[compress_state_rows_inline425_inline2017__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 0)],
    ) -> pl.Tensor[[384, 128], pl.FP32]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        pooled_kv_inline461_inline2039__rv_v2: pl.Tensor[[384, 128], pl.FP32, pl.MemRef("mem_ddr_7", pl.const(0, pl.INT64), 196608)] = self.scatter_softmax_pool_0(
            pooled_kv_inline461_inline2039__ssa_v0,
            b_dim_inline445_inline1977__ssa_v0,
            pool_workers_inline435_inline1973__ssa_v0,
            position_ids__ssa_v0,
            s_dim_inline427_inline2038__ssa_v0,
            score_proj_pad_inline1993__rv_v2,
            inner_ape__ssa_v0,
            kv_proj_pad_inline1986__rv_v2,
            inner_compress_state_block_table__ssa_v0,
            compress_state_flat_inline440_inline1999__ssa_v0,
            attrs={
                "arg_directions": [pl.adir.output_existing, pl.adir.scalar, pl.adir.scalar, pl.adir.input, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input]
            },
        )
        return pooled_kv_inline461_inline2039__ssa_v0

    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def weights_proj_reduce(
        weights_partial_inline2262__rv_v2: pl.Tensor[[1536, 64], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 393216)],
        weights_inline2220__ssa_v0: pl.Out[pl.Tensor[[384, 64], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 98304)]],
    ) -> pl.Tensor[[384, 64], pl.FP32]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        mem_vec_2: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 4096)
        mem_vec_3: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 4096)
        w_rb_inline2210__ssa_v1: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        w_r0_inline2218__ssa_v1: pl.Scalar[pl.INDEX] = w_rb_inline2210__ssa_v1 * 16
        w_sum_inline2203__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_2, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.load(
            weights_partial_inline2262__rv_v2, [w_r0_inline2218__ssa_v1, 0], [16, 64], [16, 64], target_memory=pl.Mem.Vec
        )
        partial_r0_inline2209__ssa_v0: pl.Scalar[pl.INDEX] = w_r0_inline2218__ssa_v1 + 384
        t__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_3, pl.const(4096, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.load(
            weights_partial_inline2262__rv_v2, [partial_r0_inline2209__ssa_v0, 0], [16, 64], [16, 64], target_memory=pl.Mem.Vec
        )
        w_sum_inline2203__tile_1: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_2, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.add(w_sum_inline2203__tile, t__tile)
        partial_r0_inline2209__ssa_v1: pl.Scalar[pl.INDEX] = w_r0_inline2218__ssa_v1 + 768
        t__tile_1: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_3, pl.const(4096, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.load(
            weights_partial_inline2262__rv_v2, [partial_r0_inline2209__ssa_v1, 0], [16, 64], [16, 64], target_memory=pl.Mem.Vec
        )
        w_sum_inline2203__tile_2: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_2, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.add(w_sum_inline2203__tile_1, t__tile_1)
        partial_r0_inline2209__ssa_v2: pl.Scalar[pl.INDEX] = w_r0_inline2218__ssa_v1 + 1152
        t__tile_2: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_3, pl.const(4096, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.load(
            weights_partial_inline2262__rv_v2, [partial_r0_inline2209__ssa_v2, 0], [16, 64], [16, 64], target_memory=pl.Mem.Vec
        )
        w_sum_inline2203__tile_3: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_2, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.add(w_sum_inline2203__tile_2, t__tile_2)
        t__tile_3: pl.Tile[[16, 64], pl.BF16, pl.MemRef(mem_vec_3, pl.const(4096, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(w_sum_inline2203__tile_3, target_type=pl.BF16, mode="rint")
        w_sum_v1_inline2223__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_2, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(t__tile_3, target_type=pl.FP32, mode="round")
        t__tile_4: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_2, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.muls(w_sum_v1_inline2223__tile, 0.011048543456039806)
        w_scaled_inline2236__tile: pl.Tile[[16, 64], pl.BF16, pl.MemRef(mem_vec_3, pl.const(4096, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(t__tile_4, target_type=pl.BF16, mode="rint")
        t__cast_fp32_tmp_v0: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_2, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(w_scaled_inline2236__tile, target_type=pl.FP32, mode="round")
        t__tile_5: pl.Tile[[16, 64], pl.FP16, pl.MemRef(mem_vec_3, pl.const(4096, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(t__cast_fp32_tmp_v0, target_type=pl.FP16, mode="round")
        t__tile_6: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_2, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(t__tile_5, target_type=pl.FP32, mode="round")
        weights_inline2220__tile: pl.Tensor[[384, 64], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 98304)] = pl.tile.store(
            t__tile_6, [w_r0_inline2218__ssa_v1, 0], weights_inline2220__ssa_v0
        )
        return weights_inline2220__ssa_v0

    @pl.function(type=pl.FunctionType.Spmd)
    def weights_proj_reduce_spmd(
        self,
        weights_partial_inline2262__rv_v2: pl.Tensor[[1536, 64], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 393216)],
        weights_inline2220__ssa_v0: pl.Out[pl.Tensor[[384, 64], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 98304)]],
    ) -> pl.Tensor[[384, 64], pl.FP32]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        weights_inline2220__ssa_v1: pl.Tensor[[384, 64], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 98304)] = self.weights_proj_reduce(
            weights_partial_inline2262__rv_v2, weights_inline2220__ssa_v0, attrs={"arg_directions": [pl.adir.input, pl.adir.output_existing]}
        )
        return weights_inline2220__ssa_v0

    @pl.function(type=pl.FunctionType.AIC, level=pl.Level.AIC, role=pl.Role.SubWorker)
    def weights_proj(
        weights_partial_inline2262__ssa_v0: pl.Out[pl.Tensor[[1536, 64], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 393216)]],
        row_blocks_inline2215__ssa_v0: pl.Scalar[pl.INDEX],
        bs_inline2214__ssa_v0: pl.Scalar[pl.INDEX],
        x_flat_inline2226__ssa_v0: pl.Tensor[[T_DYN, 4096], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)],
        weights_proj__ssa_v0: pl.Tensor[[4096, 64], pl.BF16, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 524288)],
    ) -> pl.Tensor[[1536, 64], pl.FP32]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        mem_acc_3: pl.Ptr = pl.tile.alloc(pl.Mem.Acc, 4096)
        mem_mat_4: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 16384)
        mem_mat_5: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 65536)
        mem_left_6: pl.Ptr = pl.tile.alloc(pl.Mem.Left, 8192)
        mem_right_7: pl.Ptr = pl.tile.alloc(pl.Mem.Right, 32768)
        mem_left_8: pl.Ptr = pl.tile.alloc(pl.Mem.Left, 8192)
        mem_right_9: pl.Ptr = pl.tile.alloc(pl.Mem.Right, 32768)
        w_worker_inline2233__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        for w_unit_inline2231__idx_v0, (weights_partial_inline2262__iter_v1,) in pl.range(
            w_worker_inline2233__ssa_v0, row_blocks_inline2215__ssa_v0 * 4, 8, init_values=(weights_partial_inline2262__ssa_v0,)
        ):
            w_rb_inline2210__ssa_v0: pl.Scalar[pl.INDEX] = w_unit_inline2231__idx_v0 // 4
            kb_inline2225__ssa_v0: pl.Scalar[pl.INDEX] = w_unit_inline2231__idx_v0 - w_rb_inline2210__ssa_v0 * 4
            w_r0_inline2218__ssa_v0: pl.Scalar[pl.INDEX] = w_rb_inline2210__ssa_v0 * 16
            w_rows_inline2219__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(bs_inline2214__ssa_v0 - w_r0_inline2218__ssa_v0, 16)
            k_base_inline2212__ssa_v0: pl.Scalar[pl.INDEX] = kb_inline2225__ssa_v0 * 1024
            weights_acc_inline2213__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 4096), pl.Mem.Acc] = pl.tile.create([16, 64], dtype=pl.FP32, target_memory=pl.Mem.Acc)
            for db_inline2197__idx_v0, (weights_acc_inline2213__iter_v1,) in pl.range(2, init_values=(weights_acc_inline2213__tile,)):
                d0_inline2228__ssa_v0: pl.Scalar[pl.INDEX] = k_base_inline2212__ssa_v0 + db_inline2197__idx_v0 * 512
                x_tile_inline2206__tile: pl.Tile[[16, 512], pl.BF16, pl.MemRef(mem_mat_4, pl.const(0, pl.INT64), 16384), pl.Mem.Mat, pl.TileView(valid_shape=[w_rows_inline2219__ssa_v0, 512])] = (
                    pl.tile.load(x_flat_inline2226__ssa_v0, [w_r0_inline2218__ssa_v0, d0_inline2228__ssa_v0], [16, 512], [w_rows_inline2219__ssa_v0, 512], target_memory=pl.Mem.Mat)
                )
                weights_proj_tile_inline2252__tile: pl.Tile[[512, 64], pl.BF16, pl.MemRef(mem_mat_5, pl.const(16384, pl.INT64), 65536), pl.Mem.Mat] = pl.tile.load(
                    weights_proj__ssa_v0, [d0_inline2228__ssa_v0, 0], [512, 64], [512, 64], target_memory=pl.Mem.Mat
                )
                weights_acc_inline2213__tile_l0_a: pl.Tile[
                    [16, 256],
                    pl.BF16,
                    pl.MemRef(mem_left_6, pl.const(0, pl.INT64), 8192),
                    pl.Mem.Left,
                    pl.TileView(valid_shape=[w_rows_inline2219__ssa_v0, 256], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal),
                ] = pl.tile.extract(x_tile_inline2206__tile, 0, 0, [16, 256], target_memory=pl.Mem.Left)
                weights_acc_inline2213__tile_l0_b: pl.Tile[[256, 64], pl.BF16, pl.MemRef(mem_right_7, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                    weights_proj_tile_inline2252__tile, 0, 0, [256, 64], target_memory=pl.Mem.Right
                )
                weights_acc_inline2213__tile_l0_a_1: pl.Tile[
                    [16, 256],
                    pl.BF16,
                    pl.MemRef(mem_left_8, pl.const(8192, pl.INT64), 8192),
                    pl.Mem.Left,
                    pl.TileView(valid_shape=[w_rows_inline2219__ssa_v0, 256], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal),
                ] = pl.tile.extract(x_tile_inline2206__tile, 0, 256, [16, 256], target_memory=pl.Mem.Left)
                weights_acc_inline2213__tile_l0_b_1: pl.Tile[[256, 64], pl.BF16, pl.MemRef(mem_right_9, pl.const(32768, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                    weights_proj_tile_inline2252__tile, 256, 0, [256, 64], target_memory=pl.Mem.Right
                )
                weights_acc_inline2213__tile_l0_c_acc: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 4096), pl.Mem.Acc] = pl.tile.matmul_acc(
                    weights_acc_inline2213__iter_v1, weights_acc_inline2213__tile_l0_a, weights_acc_inline2213__tile_l0_b, db_inline2197__idx_v0 == 0
                )
                weights_acc_inline2213__tile_l0_c_acc_1: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 4096), pl.Mem.Acc] = pl.tile.matmul_acc(
                    weights_acc_inline2213__tile_l0_c_acc, weights_acc_inline2213__tile_l0_a_1, weights_acc_inline2213__tile_l0_b_1, False
                )
                weights_acc_inline2213__rv_v2: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 4096), pl.Mem.Acc] = pl.yield_(weights_acc_inline2213__tile_l0_c_acc_1)
            weights_partial_inline2262__tile: pl.Tensor[[1536, 64], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 393216)] = pl.tile.store(
                weights_acc_inline2213__rv_v2, [kb_inline2225__ssa_v0 * 384 + w_r0_inline2218__ssa_v0, 0], weights_partial_inline2262__iter_v1
            )
            weights_partial_inline2262__rv_v2: pl.Tensor[[1536, 64], pl.FP32, pl.MemRef("mem_ddr_12", pl.const(0, pl.INT64), 393216)] = pl.yield_(weights_partial_inline2262__tile)
        return weights_partial_inline2262__ssa_v0

    @pl.function(type=pl.FunctionType.Spmd)
    def weights_proj_spmd(
        self,
        weights_partial_inline2262__ssa_v0: pl.Out[pl.Tensor[[1536, 64], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 393216)]],
        row_blocks_inline2215__ssa_v0: pl.Scalar[pl.INDEX],
        bs_inline2214__ssa_v0: pl.Scalar[pl.INDEX],
        x_flat_inline2226__ssa_v0: pl.Tensor[[T_DYN, 4096], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)],
        weights_proj__ssa_v0: pl.Tensor[[4096, 64], pl.BF16, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 524288)],
    ) -> pl.Tensor[[1536, 64], pl.FP32]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        weights_partial_inline2262__rv_v2: pl.Tensor[[1536, 64], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 393216)] = self.weights_proj(
            weights_partial_inline2262__ssa_v0,
            row_blocks_inline2215__ssa_v0,
            bs_inline2214__ssa_v0,
            x_flat_inline2226__ssa_v0,
            weights_proj__ssa_v0,
            attrs={"arg_directions": [pl.adir.output_existing, pl.adir.scalar, pl.adir.scalar, pl.adir.input, pl.adir.input]},
        )
        return weights_partial_inline2262__ssa_v0

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
        cmp_freqs_cos__ssa_v0: pl.Tensor[[T_DYN, 64], pl.FP32, pl.MemRef("mem_ddr_9", pl.const(0, pl.INT64), 0)],
        cmp_freqs_sin__ssa_v0: pl.Tensor[[T_DYN, 64], pl.FP32, pl.MemRef("mem_ddr_10", pl.const(0, pl.INT64), 0)],
        inner_freqs_cos__ssa_v0: pl.Tensor[[T_DYN, 64], pl.FP32, pl.MemRef("mem_ddr_11", pl.const(0, pl.INT64), 0)],
        inner_freqs_sin__ssa_v0: pl.Tensor[[T_DYN, 64], pl.FP32, pl.MemRef("mem_ddr_12", pl.const(0, pl.INT64), 0)],
        cmp_wkv__ssa_v0: pl.Tensor[[1024, 4096], pl.BF16, pl.MemRef("mem_ddr_13", pl.const(0, pl.INT64), 8388608)],
        cmp_wgate__ssa_v0: pl.Tensor[[1024, 4096], pl.BF16, pl.MemRef("mem_ddr_14", pl.const(0, pl.INT64), 8388608)],
        cmp_ape__ssa_v0: pl.Tensor[[4, 1024], pl.FP32, pl.MemRef("mem_ddr_15", pl.const(0, pl.INT64), 16384)],
        cmp_norm_w__ssa_v0: pl.Tensor[[512], pl.FP32, pl.MemRef("mem_ddr_16", pl.const(0, pl.INT64), 2048)],
        compress_state__ssa_v0: pl.InOut[pl.Tensor[[MAIN_STATE_BLOCK_NUM_DYN, 2, 2048], pl.FP32, pl.MemRef("mem_ddr_17", pl.const(0, pl.INT64), 0)]],
        compress_state_block_table__ssa_v0: pl.Tensor[[B_DYN, 7], pl.INT32, pl.MemRef("mem_ddr_18", pl.const(0, pl.INT64), 0)],
        idx_wq_b__ssa_v0: pl.Tensor[[1024, 8192], pl.INT8, pl.MemRef("mem_ddr_19", pl.const(0, pl.INT64), 8388608)],
        idx_wq_b_scale__ssa_v0: pl.Tensor[[8192], pl.FP32, pl.MemRef("mem_ddr_20", pl.const(0, pl.INT64), 32768)],
        weights_proj__ssa_v0: pl.Tensor[[4096, 64], pl.BF16, pl.MemRef("mem_ddr_21", pl.const(0, pl.INT64), 524288)],
        hadamard_idx__ssa_v0: pl.Tensor[[128, 128], pl.BF16, pl.MemRef("mem_ddr_22", pl.const(0, pl.INT64), 32768)],
        inner_wkv__ssa_v0: pl.Tensor[[256, 4096], pl.BF16, pl.MemRef("mem_ddr_23", pl.const(0, pl.INT64), 2097152)],
        inner_wgate__ssa_v0: pl.Tensor[[256, 4096], pl.BF16, pl.MemRef("mem_ddr_24", pl.const(0, pl.INT64), 2097152)],
        inner_ape__ssa_v0: pl.Tensor[[4, 256], pl.FP32, pl.MemRef("mem_ddr_25", pl.const(0, pl.INT64), 4096)],
        inner_norm_w__ssa_v0: pl.Tensor[[128], pl.FP32, pl.MemRef("mem_ddr_26", pl.const(0, pl.INT64), 512)],
        inner_compress_state__ssa_v0: pl.InOut[pl.Tensor[[INNER_STATE_BLOCK_NUM_DYN, 2, 512], pl.FP32, pl.MemRef("mem_ddr_27", pl.const(0, pl.INT64), 0)]],
        inner_compress_state_block_table__ssa_v0: pl.Tensor[[B_DYN, 7], pl.INT32, pl.MemRef("mem_ddr_28", pl.const(0, pl.INT64), 0)],
        kv_cache__ssa_v0: pl.InOut[pl.Tensor[[ORI_BLOCK_NUM_DYN, 32, 1, 512], pl.BF16, pl.MemRef("mem_ddr_29", pl.const(0, pl.INT64), 0)]],
        cmp_kv__ssa_v0: pl.InOut[pl.Tensor[[CMP_BLOCK_NUM_DYN, 32, 1, 512], pl.BF16, pl.MemRef("mem_ddr_30", pl.const(0, pl.INT64), 0)]],
        cmp_block_table__ssa_v0: pl.Tensor[[B_DYN, COMPRESSED_TABLE_COLUMNS_DYN], pl.INT32, pl.MemRef("mem_ddr_31", pl.const(0, pl.INT64), 0)],
        idx_kv_cache__ssa_v0: pl.InOut[pl.Tensor[[IDX_CACHE_BLOCK_NUM_DYN, INDEXER_PAGE_BYTES_DYN], pl.INT8, pl.MemRef("mem_ddr_32", pl.const(0, pl.INT64), 0)]],
        idx_block_table__ssa_v0: pl.Tensor[[B_DYN, INDEXER_TABLE_COLUMNS_DYN], pl.INT32, pl.MemRef("mem_ddr_33", pl.const(0, pl.INT64), 0)],
        ori_slot_mapping__ssa_v0: pl.Tensor[[T_DYN], pl.INT64, pl.MemRef("mem_ddr_34", pl.const(0, pl.INT64), 0)],
        window_swa_indices__ssa_v0: pl.Tensor[[T_DYN, 128], pl.INT32, pl.MemRef("mem_ddr_35", pl.const(0, pl.INT64), 0)],
        window_swa_lens__ssa_v0: pl.Tensor[[T_DYN], pl.INT32, pl.MemRef("mem_ddr_36", pl.const(0, pl.INT64), 0)],
        cmp_slot_mapping__ssa_v0: pl.Tensor[[T_DYN], pl.INT64, pl.MemRef("mem_ddr_37", pl.const(0, pl.INT64), 0)],
        idx_slot_mapping__ssa_v0: pl.Tensor[[T_DYN], pl.INT64, pl.MemRef("mem_ddr_38", pl.const(0, pl.INT64), 0)],
        state_slot_mapping__ssa_v0: pl.Tensor[[T_DYN], pl.INT64, pl.MemRef("mem_ddr_39", pl.const(0, pl.INT64), 0)],
        inner_state_slot_mapping__ssa_v0: pl.Tensor[[T_DYN], pl.INT64, pl.MemRef("mem_ddr_40", pl.const(0, pl.INT64), 0)],
        position_ids__ssa_v0: pl.Tensor[[T_DYN], pl.INT32, pl.MemRef("mem_ddr_41", pl.const(0, pl.INT64), 0)],
        kv_seq_lens__ssa_v0: pl.Tensor[[B_DYN], pl.INT32, pl.MemRef("mem_ddr_42", pl.const(0, pl.INT64), 0)],
        attn_sink__ssa_v0: pl.Tensor[[64], pl.FP32, pl.MemRef("mem_ddr_43", pl.const(0, pl.INT64), 256)],
        wo_a__ssa_v0: pl.Tensor[[8, 1024, 4096], pl.BF16, pl.MemRef("mem_ddr_44", pl.const(0, pl.INT64), 67108864)],
        wo_b__ssa_v0: pl.Tensor[[4096, 8192], pl.INT8, pl.MemRef("mem_ddr_45", pl.const(0, pl.INT64), 33554432)],
        wo_b_scale__ssa_v0: pl.Tensor[[4096], pl.FP32, pl.MemRef("mem_ddr_46", pl.const(0, pl.INT64), 16384)],
        idx_topk_scores__ssa_v0: pl.Out[pl.Tensor[[T_DYN, 512], pl.FP32, pl.MemRef("mem_ddr_47", pl.const(0, pl.INT64), 0)]],
        idx_topk__ssa_v0: pl.Out[pl.Tensor[[T_DYN, 512], pl.INT32, pl.MemRef("mem_ddr_48", pl.const(0, pl.INT64), 0)]],
        attn_out__ssa_v0: pl.Out[pl.Tensor[[T_DYN, 4096], pl.BF16, pl.MemRef("mem_ddr_49", pl.const(0, pl.INT64), 0)]],
    ) -> pl.Tensor[[T_DYN, 4096], pl.BF16]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        wb_blocks__ssa_v0: pl.Scalar[pl.INDEX] = T_DYN // 8
        idx_cos_il__ssa_v0: pl.Tensor[[T_DYN, 64], pl.FP32, pl.MemRef("mem_ddr_50", pl.const(0, pl.INT64), 0)] = pl.tensor.create([T_DYN, 64], dtype=pl.FP32, layout=pl.TensorLayout.ND)
        idx_sin_signed__ssa_v0: pl.Tensor[[T_DYN, 64], pl.FP32, pl.MemRef("mem_ddr_51", pl.const(0, pl.INT64), 0)] = pl.tensor.create([T_DYN, 64], dtype=pl.FP32, layout=pl.TensorLayout.ND)
        cmp_cos_il__ssa_v0: pl.Tensor[[T_DYN, 64], pl.FP32, pl.MemRef("mem_ddr_52", pl.const(0, pl.INT64), 0)] = pl.tensor.create([T_DYN, 64], dtype=pl.FP32, layout=pl.TensorLayout.ND)
        cmp_sin_signed__ssa_v0: pl.Tensor[[T_DYN, 64], pl.FP32, pl.MemRef("mem_ddr_53", pl.const(0, pl.INT64), 0)] = pl.tensor.create([T_DYN, 64], dtype=pl.FP32, layout=pl.TensorLayout.ND)
        inner_cos_il__ssa_v0: pl.Tensor[[T_DYN, 64], pl.FP32, pl.MemRef("mem_ddr_54", pl.const(0, pl.INT64), 0)] = pl.tensor.create([T_DYN, 64], dtype=pl.FP32, layout=pl.TensorLayout.ND)
        inner_sin_signed__ssa_v0: pl.Tensor[[T_DYN, 64], pl.FP32, pl.MemRef("mem_ddr_55", pl.const(0, pl.INT64), 0)] = pl.tensor.create([T_DYN, 64], dtype=pl.FP32, layout=pl.TensorLayout.ND)
        ret__tmp_v0: pl.Tuple[
            pl.Tensor[[T_DYN, 64], pl.FP32],
            pl.Tensor[[T_DYN, 64], pl.FP32],
            pl.Tensor[[T_DYN, 64], pl.FP32],
            pl.Tensor[[T_DYN, 64], pl.FP32],
            pl.Tensor[[T_DYN, 64], pl.FP32],
            pl.Tensor[[T_DYN, 64], pl.FP32],
            pl.Scalar[pl.TASK_ID],
        ] = pl.submit(
            self.csa_rope_interleave,
            cmp_cos_il__ssa_v0,
            cmp_sin_signed__ssa_v0,
            idx_cos_il__ssa_v0,
            idx_sin_signed__ssa_v0,
            inner_cos_il__ssa_v0,
            inner_sin_signed__ssa_v0,
            T_DYN,
            freqs_cos__ssa_v0,
            freqs_sin__ssa_v0,
            cmp_freqs_cos__ssa_v0,
            cmp_freqs_sin__ssa_v0,
            inner_freqs_cos__ssa_v0,
            inner_freqs_sin__ssa_v0,
            attrs={
                "arg_directions": [
                    pl.adir.output_existing,
                    pl.adir.output_existing,
                    pl.adir.output_existing,
                    pl.adir.output_existing,
                    pl.adir.output_existing,
                    pl.adir.output_existing,
                    pl.adir.scalar,
                    pl.adir.input,
                    pl.adir.input,
                    pl.adir.input,
                    pl.adir.input,
                    pl.adir.input,
                    pl.adir.input,
                ]
            },
        )
        cmp_cos_il__rv_v2: pl.Tensor[[T_DYN, 64], pl.FP32, pl.MemRef("mem_ddr_56", pl.const(0, pl.INT64), 0)] = ret__tmp_v0[0]
        cmp_sin_signed__rv_v2: pl.Tensor[[T_DYN, 64], pl.FP32, pl.MemRef("mem_ddr_57", pl.const(0, pl.INT64), 0)] = ret__tmp_v0[1]
        idx_cos_il__rv_v2: pl.Tensor[[T_DYN, 64], pl.FP32, pl.MemRef("mem_ddr_58", pl.const(0, pl.INT64), 0)] = ret__tmp_v0[2]
        idx_sin_signed__rv_v2: pl.Tensor[[T_DYN, 64], pl.FP32, pl.MemRef("mem_ddr_59", pl.const(0, pl.INT64), 0)] = ret__tmp_v0[3]
        inner_cos_il__rv_v2: pl.Tensor[[T_DYN, 64], pl.FP32, pl.MemRef("mem_ddr_60", pl.const(0, pl.INT64), 0)] = ret__tmp_v0[4]
        inner_sin_signed__rv_v2: pl.Tensor[[T_DYN, 64], pl.FP32, pl.MemRef("mem_ddr_61", pl.const(0, pl.INT64), 0)] = ret__tmp_v0[5]
        rope_tid__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0[6]
        q__ssa_v0: pl.Tensor[[T_DYN, 64, 512], pl.BF16, pl.MemRef("mem_ddr_62", pl.const(0, pl.INT64), 0)] = pl.tensor.create([T_DYN, 64, 512], dtype=pl.BF16, layout=pl.TensorLayout.ND)
        kv__ssa_v0: pl.Tensor[[T_DYN, 512], pl.BF16, pl.MemRef("mem_ddr_63", pl.const(0, pl.INT64), 0)] = pl.tensor.create([T_DYN, 512], dtype=pl.BF16, layout=pl.TensorLayout.ND)
        qr__ssa_v0: pl.Tensor[[T_DYN, 1024], pl.INT8, pl.MemRef("mem_ddr_64", pl.const(0, pl.INT64), 0)] = pl.tensor.create([T_DYN, 1024], dtype=pl.INT8, layout=pl.TensorLayout.ND)
        qr_scale__ssa_v0: pl.Tensor[[T_DYN, 1], pl.FP32, pl.MemRef("mem_ddr_65", pl.const(0, pl.INT64), 0)] = pl.tensor.create([T_DYN, 1], dtype=pl.FP32, layout=pl.TensorLayout.ND)
        position_ids_t1__ssa_v0: pl.Tensor[[T_DYN, 1], pl.INT32, pl.MemRef("mem_ddr_41", pl.const(0, pl.INT64), 0)] = pl.tensor.reshape(position_ids__ssa_v0, [T_DYN, 1])
        with pl.scope():
            late_dep__ssa_v0: pl.Scalar[pl.TASK_ID] = pl.system.task_dummy(deps=[rope_tid__ssa_v0])
            t_dim_inline504__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(x_normed_t__ssa_v0, 0)
            q_rope_cos_il_inline503__ssa_v0: pl.Tensor[[t_dim_inline504__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_66", pl.const(0, pl.INT64), 0)] = pl.tensor.create(
                [t_dim_inline504__ssa_v0, 64], dtype=pl.FP32, layout=pl.TensorLayout.ND
            )
            q_rope_sin_signed_inline502__ssa_v0: pl.Tensor[[t_dim_inline504__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_67", pl.const(0, pl.INT64), 0)] = pl.tensor.create(
                [t_dim_inline504__ssa_v0, 64], dtype=pl.FP32, layout=pl.TensorLayout.ND
            )
            q_rope_swap_idx_inline501__ssa_v0: pl.Tensor[[t_dim_inline504__ssa_v0, 64], pl.INT32, pl.MemRef("mem_ddr_68", pl.const(0, pl.INT64), 0)] = pl.tensor.create(
                [t_dim_inline504__ssa_v0, 64], dtype=pl.INT32, layout=pl.TensorLayout.ND
            )
            t_dim_inline1637__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(freqs_cos__ssa_v0, 0)
            rope_cos_view_inline1642__ssa_v0: pl.Tensor[[t_dim_inline1637__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_7", pl.const(0, pl.INT64), 0)] = pl.tensor.reshape(
                freqs_cos__ssa_v0, [t_dim_inline1637__ssa_v0, 64]
            )
            rope_sin_view_inline1634__ssa_v0: pl.Tensor[[t_dim_inline1637__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_8", pl.const(0, pl.INT64), 0)] = pl.tensor.reshape(
                freqs_sin__ssa_v0, [t_dim_inline1637__ssa_v0, 64]
            )
            rope_cos_il_view_inline1632__ssa_v0: pl.Tensor[[t_dim_inline1637__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_66", pl.const(0, pl.INT64), 0)] = pl.tensor.reshape(
                q_rope_cos_il_inline503__ssa_v0, [t_dim_inline1637__ssa_v0, 64]
            )
            rope_sin_signed_view_inline1646__ssa_v0: pl.Tensor[[t_dim_inline1637__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_67", pl.const(0, pl.INT64), 0)] = pl.tensor.reshape(
                q_rope_sin_signed_inline502__ssa_v0, [t_dim_inline1637__ssa_v0, 64]
            )
            rope_swap_idx_view_inline1631__ssa_v0: pl.Tensor[[t_dim_inline1637__ssa_v0, 64], pl.INT32, pl.MemRef("mem_ddr_68", pl.const(0, pl.INT64), 0)] = pl.tensor.reshape(
                q_rope_swap_idx_inline501__ssa_v0, [t_dim_inline1637__ssa_v0, 64]
            )
            token_tiles_inline1636__ssa_v0: pl.Scalar[pl.INDEX] = (t_dim_inline1637__ssa_v0 + 7) // 8
            ret__tmp_v0_1: pl.Tuple[pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(
                self.q_rope_prepare_spmd,
                rope_cos_il_view_inline1632__ssa_v0,
                rope_sin_signed_view_inline1646__ssa_v0,
                rope_swap_idx_view_inline1631__ssa_v0,
                token_tiles_inline1636__ssa_v0,
                t_dim_inline1637__ssa_v0,
                rope_cos_view_inline1642__ssa_v0,
                rope_sin_view_inline1634__ssa_v0,
                core_num=pl.min(token_tiles_inline1636__ssa_v0, 48),
                allow_early_resolve=True,
                attrs={"arg_directions": [pl.adir.output_existing, pl.adir.output_existing, pl.adir.output_existing, pl.adir.scalar, pl.adir.scalar, pl.adir.input, pl.adir.input]},
            )
            tid__ssa_v1: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_1[0]
            qr_i8_matmul_inline1727__ssa_v0: pl.Tensor[[512, 1024], pl.INT8, pl.MemRef("mem_ddr_69", pl.const(0, pl.INT64), 524288)] = pl.tensor.create(
                [512, 1024], dtype=pl.INT8, layout=pl.TensorLayout.ND
            )
            qr_scale_pad_store_inline1684__ssa_v0: pl.Tensor[[512, 1], pl.FP32, pl.MemRef("mem_ddr_70", pl.const(0, pl.INT64), 2048)] = pl.tensor.create(
                [512, 1], dtype=pl.FP32, layout=pl.TensorLayout.ND, manual_dep=True
            )
            t_dim_inline313_inline1671__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(x_normed_t__ssa_v0, 0)
            for tile_base_inline319_inline1715__idx_v0, (qr_i8_matmul_inline1727__iter_v1, qr_scale_pad_store_inline1684__iter_v1) in pl.range(
                0,
                t_dim_inline313_inline1671__ssa_v0,
                512,
                init_values=(qr_i8_matmul_inline1727__ssa_v0, qr_scale_pad_store_inline1684__ssa_v0),
                attrs={"iter_arg_rebind_0": False, "iter_arg_rebind_1": False},
            ):
                tile_rows_inline340_inline1692__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(t_dim_inline313_inline1671__ssa_v0 - tile_base_inline319_inline1715__idx_v0, 512)
                with pl.scope():
                    x_view_inline316_inline1698__ssa_v0: pl.Tensor[[t_dim_inline313_inline1671__ssa_v0, 4096], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)] = pl.tensor.reshape(
                        x_normed_t__ssa_v0, [t_dim_inline313_inline1671__ssa_v0, 4096]
                    )
                    qr_t_matmul_inline322_inline1689__ssa_v0: pl.Scalar[pl.INDEX] = (tile_rows_inline340_inline1692__ssa_v0 + 15) // 16 * 16
                    qr_full_rows_inline325_inline1680__ssa_v0: pl.Scalar[pl.INDEX] = tile_rows_inline340_inline1692__ssa_v0 // 64 * 64
                    qr_fp32_inline302_inline1687__ssa_v0: pl.Tensor[[qr_t_matmul_inline322_inline1689__ssa_v0, 1024], pl.FP32, pl.MemRef("mem_ddr_71", pl.const(0, pl.INT64), 0)] = pl.tensor.create(
                        [qr_t_matmul_inline322_inline1689__ssa_v0, 1024], dtype=pl.FP32, layout=pl.TensorLayout.ND
                    )
                    qr_fp32_inline302_inline1687__rv_v2: pl.Tensor[[qr_t_matmul_inline322_inline1689__ssa_v0, 1024], pl.FP32, pl.MemRef("mem_ddr_72", pl.const(0, pl.INT64), 0)] = self.qr_proj_seed(
                        qr_fp32_inline302_inline1687__ssa_v0, qr_t_matmul_inline322_inline1689__ssa_v0, attrs={"arg_directions": [pl.adir.inout, pl.adir.scalar]}
                    )
                    ret__tmp_v0_2: pl.Tuple[pl.Tensor[[qr_t_matmul_inline322_inline1689__ssa_v0, 1024], pl.FP32], pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(
                        self.qr_proj_matmul_spmd,
                        qr_fp32_inline302_inline1687__rv_v2,
                        qr_full_rows_inline325_inline1680__ssa_v0,
                        tile_base_inline319_inline1715__idx_v0,
                        x_view_inline316_inline1698__ssa_v0,
                        wq_a__ssa_v0,
                        qr_t_matmul_inline322_inline1689__ssa_v0,
                        tile_rows_inline340_inline1692__ssa_v0,
                        core_num=16,
                        allow_early_resolve=True,
                        attrs={"arg_directions": [pl.adir.inout, pl.adir.scalar, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.scalar, pl.adir.scalar]},
                    )
                    qr_fp32_inline302_inline1687__rv_v10: pl.Tensor[[qr_t_matmul_inline322_inline1689__ssa_v0, 1024], pl.FP32, pl.MemRef("mem_ddr_73", pl.const(0, pl.INT64), 0)] = ret__tmp_v0_2[0]
                    tid__ssa_v2: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_2[1]
                    qr_view_inline333_inline1702__ssa_v0: pl.Tensor[[t_dim_inline313_inline1671__ssa_v0, 1024], pl.INT8, pl.MemRef("mem_ddr_64", pl.const(0, pl.INT64), 0)] = pl.tensor.reshape(
                        qr__ssa_v0, [t_dim_inline313_inline1671__ssa_v0, 1024]
                    )
                    qr_scale_view_inline335_inline1703__ssa_v0: pl.Tensor[[t_dim_inline313_inline1671__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_65", pl.const(0, pl.INT64), 0)] = pl.tensor.reshape(
                        qr_scale__ssa_v0, [t_dim_inline313_inline1671__ssa_v0, 1]
                    )
                    qr_token_tiles_inline338_inline1697__ssa_v0: pl.Scalar[pl.INDEX] = (tile_rows_inline340_inline1692__ssa_v0 + 7) // 8
                    ret__tmp_v0_3: pl.Tuple[pl.Tensor[[512, 1], pl.FP32], pl.Tensor[[512, 1024], pl.INT8], pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(
                        self.qr_rms_norm_quant_spmd,
                        tile_rows_inline340_inline1692__ssa_v0,
                        tile_base_inline319_inline1715__idx_v0,
                        qr_fp32_inline302_inline1687__rv_v10,
                        gamma_cq__ssa_v0,
                        qr_scale_pad_store_inline1684__iter_v1,
                        qr_scale_view_inline335_inline1703__ssa_v0,
                        qr_i8_matmul_inline1727__iter_v1,
                        qr_view_inline333_inline1702__ssa_v0,
                        core_num=qr_token_tiles_inline338_inline1697__ssa_v0,
                        allow_early_resolve=True,
                        attrs={"arg_directions": [pl.adir.scalar, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.inout, pl.adir.inout, pl.adir.inout, pl.adir.inout]},
                    )
                    qr_scale_pad_store_inline1684__ssa_v3: pl.Tensor[[512, 1], pl.FP32, pl.MemRef("mem_ddr_74", pl.const(0, pl.INT64), 2048)] = ret__tmp_v0_3[0]
                    qr_i8_matmul_inline1727__rv_v4: pl.Tensor[[512, 1024], pl.INT8, pl.MemRef("mem_ddr_75", pl.const(0, pl.INT64), 524288)] = ret__tmp_v0_3[1]
                    tid__ssa_v3: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_3[2]
                qr_i8_matmul_inline1727__rv_v2, qr_scale_pad_store_inline1684__rv_v2 = pl.yield_(qr_i8_matmul_inline1727__rv_v4, qr_scale_pad_store_inline1684__ssa_v3)
            q_seq_dep_inline1659__ssa_v0: pl.Scalar[pl.TASK_ID] = pl.system.task_dummy(deps=[])
            t_dim_inline368_inline1658__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(x_normed_t__ssa_v0, 0)
            for tile_base_inline367_inline1657__idx_v0 in pl.range(0, t_dim_inline368_inline1658__ssa_v0, 512):
                tile_rows_inline366_inline1656__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(t_dim_inline368_inline1658__ssa_v0 - tile_base_inline367_inline1657__idx_v0, 512)
                with pl.scope():
                    qproj_t_matmul_inline365_inline1672__ssa_v0: pl.Scalar[pl.INDEX] = (tile_rows_inline366_inline1656__ssa_v0 + 15) // 16 * 16
                    q_proj_i32_inline364_inline1723__ssa_v0: pl.Tensor[[qproj_t_matmul_inline365_inline1672__ssa_v0, 32768], pl.INT32, pl.MemRef("mem_ddr_78", pl.const(0, pl.INT64), 0)] = (
                        pl.tensor.create([qproj_t_matmul_inline365_inline1672__ssa_v0, 32768], dtype=pl.INT32, layout=pl.TensorLayout.ND)
                    )
                    qproj_t_matmul_inline2640__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(q_proj_i32_inline364_inline1723__ssa_v0, 0)
                    qproj_full_rows_inline2636__ssa_v0: pl.Scalar[pl.INDEX] = tile_rows_inline366_inline1656__ssa_v0 // 64 * 64
                    ret__tmp_v0_4: pl.Tuple[pl.Tensor[[qproj_t_matmul_inline365_inline1672__ssa_v0, 32768], pl.INT32], pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(
                        self.qproj_matmul_spmd,
                        q_proj_i32_inline364_inline1723__ssa_v0,
                        qproj_full_rows_inline2636__ssa_v0,
                        qr_i8_matmul_inline1727__rv_v2,
                        wq_b__ssa_v0,
                        qproj_t_matmul_inline2640__ssa_v0,
                        tile_rows_inline366_inline1656__ssa_v0,
                        deps=[q_seq_dep_inline1659__ssa_v0],
                        core_num=24,
                        attrs={"arg_directions": [pl.adir.inout, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.scalar, pl.adir.scalar]},
                    )
                    q_proj_i32_inline364_inline1723__rv_v2: pl.Tensor[[qproj_t_matmul_inline365_inline1672__ssa_v0, 32768], pl.INT32, pl.MemRef("mem_ddr_79", pl.const(0, pl.INT64), 0)] = (
                        ret__tmp_v0_4[0]
                    )
                    qproj_tid_inline2633__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_4[1]
                    q_proj_i32_inline364_inline1723__ssa_v9: pl.Tensor[[qproj_t_matmul_inline365_inline1672__ssa_v0, 32768], pl.INT32, pl.MemRef("mem_ddr_80", pl.const(0, pl.INT64), 0)] = (
                        q_proj_i32_inline364_inline1723__rv_v2
                    )
                    t_dim_inline2655__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(q__ssa_v0, 0)
                    q_flat_inline2682__ssa_v0: pl.Tensor[[t_dim_inline2655__ssa_v0, 32768], pl.BF16, pl.MemRef("mem_ddr_62", pl.const(0, pl.INT64), 0)] = pl.tensor.reshape(
                        q__ssa_v0, [t_dim_inline2655__ssa_v0, 32768]
                    )
                    ret__tmp_v0_5: pl.Tuple[pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(
                        self.qproj_dequant_rms_nope_rope_spmd,
                        q_flat_inline2682__ssa_v0,
                        tile_rows_inline366_inline1656__ssa_v0,
                        tile_base_inline367_inline1657__idx_v0,
                        qr_scale_pad_store_inline1684__rv_v2,
                        q_rope_cos_il_inline503__ssa_v0,
                        q_rope_sin_signed_inline502__ssa_v0,
                        q_rope_swap_idx_inline501__ssa_v0,
                        q_proj_i32_inline364_inline1723__ssa_v9,
                        wq_b_scale__ssa_v0,
                        core_num=48,
                        allow_early_resolve=True,
                        attrs={"arg_directions": [pl.adir.inout, pl.adir.scalar, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input]},
                    )
                    tid__ssa_v4: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_5[0]
            t_dim_inline1779__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(x_normed_t__ssa_v0, 0)
            for tile_base_inline1766__idx_v0 in pl.range(0, t_dim_inline1779__ssa_v0, 512):
                tile_rows_inline1768__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(t_dim_inline1779__ssa_v0 - tile_base_inline1766__idx_v0, 512)
                with pl.scope():
                    x_view_inline1760__ssa_v0: pl.Tensor[[t_dim_inline1779__ssa_v0, 4096], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)] = pl.tensor.reshape(
                        x_normed_t__ssa_v0, [t_dim_inline1779__ssa_v0, 4096]
                    )
                    t_matmul_inline1778__ssa_v0: pl.Scalar[pl.INDEX] = (tile_rows_inline1768__ssa_v0 + 15) // 16 * 16
                    kv_full_rows_inline1775__ssa_v0: pl.Scalar[pl.INDEX] = tile_rows_inline1768__ssa_v0 // 64 * 64
                    kv_m_groups_inline1787__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(pl.max(tile_rows_inline1768__ssa_v0 // 128, 1), 3)
                    kv_fp32_inline1786__ssa_v0: pl.Tensor[[t_matmul_inline1778__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_81", pl.const(0, pl.INT64), 0)] = pl.tensor.create(
                        [t_matmul_inline1778__ssa_v0, 512], dtype=pl.FP32, layout=pl.TensorLayout.ND
                    )
                    kv_fp32_inline1786__rv_v2: pl.Tensor[[t_matmul_inline1778__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_82", pl.const(0, pl.INT64), 0)] = self.kv_proj_seed(
                        kv_fp32_inline1786__ssa_v0, t_matmul_inline1778__ssa_v0, attrs={"arg_directions": [pl.adir.inout, pl.adir.scalar]}
                    )
                    ret__tmp_v0_6: pl.Tuple[pl.Tensor[[t_matmul_inline1778__ssa_v0, 512], pl.FP32], pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(
                        self.kv_proj_matmul_spmd,
                        kv_m_groups_inline1787__ssa_v0,
                        kv_fp32_inline1786__rv_v2,
                        kv_full_rows_inline1775__ssa_v0,
                        tile_base_inline1766__idx_v0,
                        x_view_inline1760__ssa_v0,
                        wkv__ssa_v0,
                        t_matmul_inline1778__ssa_v0,
                        tile_rows_inline1768__ssa_v0,
                        deps=[late_dep__ssa_v0],
                        core_num=kv_m_groups_inline1787__ssa_v0 * 8,
                        attrs={"arg_directions": [pl.adir.scalar, pl.adir.inout, pl.adir.scalar, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.scalar, pl.adir.scalar]},
                    )
                    kv_fp32_inline1786__rv_v10: pl.Tensor[[t_matmul_inline1778__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_83", pl.const(0, pl.INT64), 0)] = ret__tmp_v0_6[0]
                    _kv_tid_inline1762__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_6[1]
                    kv_view_inline1818__ssa_v0: pl.Tensor[[t_dim_inline1779__ssa_v0, 512], pl.BF16, pl.MemRef("mem_ddr_63", pl.const(0, pl.INT64), 0)] = pl.tensor.reshape(
                        kv__ssa_v0, [t_dim_inline1779__ssa_v0, 512]
                    )
                    kv_token_tiles_inline1799__ssa_v0: pl.Scalar[pl.INDEX] = (tile_rows_inline1768__ssa_v0 + 31) // 32
                    self.kv_rms_norm_rope_spmd(
                        tile_rows_inline1768__ssa_v0,
                        tile_base_inline1766__idx_v0,
                        kv_fp32_inline1786__rv_v10,
                        kv_view_inline1818__ssa_v0,
                        gamma_ckv__ssa_v0,
                        q_rope_cos_il_inline503__ssa_v0,
                        q_rope_sin_signed_inline502__ssa_v0,
                        q_rope_swap_idx_inline501__ssa_v0,
                        attrs={
                            "arg_directions": [pl.adir.scalar, pl.adir.scalar, pl.adir.input, pl.adir.inout, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input],
                            "core_num": kv_token_tiles_inline1799__ssa_v0,
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
            cmp_out__ssa_v0: pl.Tensor[[T_DYN, 512], pl.FP32, pl.MemRef("mem_ddr_84", pl.const(0, pl.INT64), 0)] = pl.tensor.create([T_DYN, 512], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            pooled_kv_inline511__ssa_v0: pl.Tensor[[384, 512], pl.FP32, pl.MemRef("mem_ddr_85", pl.const(0, pl.INT64), 786432)] = pl.tensor.create([384, 512], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            kv_proj_pad_inline510__ssa_v0: pl.Tensor[[384, 1024], pl.FP32, pl.MemRef("mem_ddr_86", pl.const(0, pl.INT64), 1572864)] = pl.tensor.create(
                [384, 1024], dtype=pl.FP32, layout=pl.TensorLayout.ND
            )
            score_proj_pad_inline509__ssa_v0: pl.Tensor[[384, 1024], pl.FP32, pl.MemRef("mem_ddr_87", pl.const(0, pl.INT64), 1572864)] = pl.tensor.create(
                [384, 1024], dtype=pl.FP32, layout=pl.TensorLayout.ND
            )
            bs_inline666_inline1848__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(x_normed_t__ssa_v0, 0)
            t_matmul_inline667_inline1842__ssa_v0: pl.Scalar[pl.INDEX] = (bs_inline666_inline1848__ssa_v0 + 63) // 64 * 64
            x_flat_inline668_inline1840__ssa_v0: pl.Tensor[[T_DYN, 4096], pl.BF16, pl.MemRef("mem_ddr_88", pl.const(0, pl.INT64), 0)] = x_normed_t__ssa_v0
            cmp4_kv_proj_pad_inline669_inline1882__ssa_v0: pl.Tensor[[384, 1024], pl.FP32, pl.MemRef("mem_ddr_89", pl.const(0, pl.INT64), 1572864)] = kv_proj_pad_inline510__ssa_v0
            cmp4_score_proj_pad_inline671_inline1867__ssa_v0: pl.Tensor[[384, 1024], pl.FP32, pl.MemRef("mem_ddr_90", pl.const(0, pl.INT64), 1572864)] = score_proj_pad_inline509__ssa_v0
            ret__tmp_v0_7: pl.Tuple[pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(
                self.kv_score_proj_spmd,
                cmp4_kv_proj_pad_inline669_inline1882__ssa_v0,
                cmp4_score_proj_pad_inline671_inline1867__ssa_v0,
                t_matmul_inline667_inline1842__ssa_v0,
                bs_inline666_inline1848__ssa_v0,
                x_flat_inline668_inline1840__ssa_v0,
                cmp_wkv__ssa_v0,
                cmp_wgate__ssa_v0,
                deps=[late_dep__ssa_v0],
                core_num=24,
                attrs={"arg_directions": [pl.adir.output_existing, pl.adir.output_existing, pl.adir.scalar, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.input]},
            )
            _kv_score_tid_inline676_inline1841__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_7[0]
            projection_tid_inline1844__ssa_v0: pl.Scalar[pl.TASK_ID] = _kv_score_tid_inline676_inline1841__ssa_v0
            b_dim_inline706_inline1830__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(compress_state_block_table__ssa_v0, 0)
            bs_inline701_inline1829__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(position_ids__ssa_v0, 0)
            s_dim_inline700_inline1846__ssa_v0: pl.Scalar[pl.INDEX] = bs_inline701_inline1829__ssa_v0 // b_dim_inline706_inline1830__ssa_v0
            cmp4_kv_proj_pad_inline708_inline1836__ssa_v0: pl.Tensor[[384, 1024], pl.FP32, pl.MemRef("mem_ddr_91", pl.const(0, pl.INT64), 1572864)] = kv_proj_pad_inline510__ssa_v0
            cmp4_score_proj_pad_inline702_inline1863__ssa_v0: pl.Tensor[[384, 1024], pl.FP32, pl.MemRef("mem_ddr_92", pl.const(0, pl.INT64), 1572864)] = score_proj_pad_inline509__ssa_v0
            compress_state_block_num_inline695_inline1843__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(compress_state__ssa_v0, 0)
            compress_state_rows_inline693_inline1838__ssa_v0: pl.Scalar[pl.INDEX] = compress_state_block_num_inline695_inline1843__ssa_v0 * 2
            compress_state_flat_inline699_inline1828__ssa_v0: pl.Tensor[[compress_state_rows_inline693_inline1838__ssa_v0, 2048], pl.FP32, pl.MemRef("mem_ddr_17", pl.const(0, pl.INT64), 0)] = (
                pl.tensor.reshape(compress_state__ssa_v0, [compress_state_rows_inline693_inline1838__ssa_v0, 2048])
            )
            _kv_score_tid_inline716_inline1833__ssa_v0: pl.Scalar[pl.TASK_ID] = projection_tid_inline1844__ssa_v0
            pool_workers_inline692_inline1827__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(b_dim_inline706_inline1830__ssa_v0, 48)
            ret__tmp_v0_8: pl.Tuple[pl.Tensor[[384, 512], pl.FP32], pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(
                self.scatter_softmax_pool_spmd,
                pooled_kv_inline511__ssa_v0,
                b_dim_inline706_inline1830__ssa_v0,
                pool_workers_inline692_inline1827__ssa_v0,
                position_ids__ssa_v0,
                s_dim_inline700_inline1846__ssa_v0,
                cmp4_score_proj_pad_inline702_inline1863__ssa_v0,
                cmp_ape__ssa_v0,
                cmp4_kv_proj_pad_inline708_inline1836__ssa_v0,
                compress_state_block_table__ssa_v0,
                compress_state_flat_inline699_inline1828__ssa_v0,
                deps=[_kv_score_tid_inline716_inline1833__ssa_v0],
                core_num=pool_workers_inline692_inline1827__ssa_v0,
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
            pooled_kv_inline511__rv_v2: pl.Tensor[[384, 512], pl.FP32, pl.MemRef("mem_ddr_93", pl.const(0, pl.INT64), 786432)] = ret__tmp_v0_8[0]
            pool_tid_inline712_inline1865__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_8[1]
            pool_tid_inline1826__ssa_v0: pl.Scalar[pl.TASK_ID] = pool_tid_inline712_inline1865__ssa_v0
            projection_ready_tid_inline1825__ssa_v0: pl.Scalar[pl.TASK_ID] = _kv_score_tid_inline716_inline1833__ssa_v0
            pool_tid_inline507__ssa_v0: pl.Scalar[pl.TASK_ID] = pool_tid_inline1826__ssa_v0
            kv_score_tid_inline506__ssa_v0: pl.Scalar[pl.TASK_ID] = projection_ready_tid_inline1825__ssa_v0
            b_dim_inline1929__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(compress_state_block_table__ssa_v0, 0)
            bs_inline1923__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(position_ids__ssa_v0, 0)
            s_dim_inline1913__ssa_v0: pl.Scalar[pl.INDEX] = bs_inline1923__ssa_v0 // b_dim_inline1929__ssa_v0
            rms_blocks_inline1919__ssa_v0: pl.Scalar[pl.INDEX] = (bs_inline1923__ssa_v0 + 15) // 16
            cmp_block_num_inline1916__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(cmp_kv__ssa_v0, 0)
            kv_flat_inline1928__ssa_v0: pl.Tensor[[T_DYN, 512], pl.FP32, pl.MemRef("mem_ddr_94", pl.const(0, pl.INT64), 0)] = cmp_out__ssa_v0
            cmp_kv_cache_flat_inline1914__ssa_v0: pl.Tensor[[cmp_block_num_inline1916__ssa_v0 * pl.const(32, pl.INDEX), 512], pl.BF16, pl.MemRef("mem_ddr_30", pl.const(0, pl.INT64), 0)] = (
                pl.tensor.reshape(cmp_kv__ssa_v0, [cmp_block_num_inline1916__ssa_v0 * 32, 512])
            )
            compress_state_block_num_inline1915__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(compress_state__ssa_v0, 0)
            compress_state_rows_inline1905__ssa_v0: pl.Scalar[pl.INDEX] = compress_state_block_num_inline1915__ssa_v0 * 2
            compress_state_flat_inline1925__ssa_v0: pl.Tensor[[compress_state_rows_inline1905__ssa_v0, 2048], pl.FP32, pl.MemRef("mem_ddr_17", pl.const(0, pl.INT64), 0)] = pl.tensor.reshape(
                compress_state__ssa_v0, [compress_state_rows_inline1905__ssa_v0, 2048]
            )
            cmp4_kv_proj_pad_inline1912__ssa_v0: pl.Tensor[[384, 1024], pl.FP32, pl.MemRef("mem_ddr_95", pl.const(0, pl.INT64), 1572864)] = kv_proj_pad_inline510__ssa_v0
            cmp4_score_proj_pad_inline1921__ssa_v0: pl.Tensor[[384, 1024], pl.FP32, pl.MemRef("mem_ddr_96", pl.const(0, pl.INT64), 1572864)] = score_proj_pad_inline509__ssa_v0
            commit_workers_inline1920__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(b_dim_inline1929__ssa_v0, 48)
            ret__tmp_v0_9: pl.Tuple[pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(
                self.compress_state_commit_spmd,
                compress_state_flat_inline1925__ssa_v0,
                b_dim_inline1929__ssa_v0,
                commit_workers_inline1920__ssa_v0,
                s_dim_inline1913__ssa_v0,
                state_slot_mapping__ssa_v0,
                position_ids__ssa_v0,
                cmp4_kv_proj_pad_inline1912__ssa_v0,
                cmp4_score_proj_pad_inline1921__ssa_v0,
                cmp_ape__ssa_v0,
                deps=[pool_tid_inline507__ssa_v0, pool_tid_inline507__ssa_v0],
                core_num=commit_workers_inline1920__ssa_v0,
                attrs={"arg_directions": [pl.adir.output_existing, pl.adir.scalar, pl.adir.scalar, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input]},
            )
            tid__ssa_v5: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_9[0]
            normed_kv_inline1917__ssa_v0: pl.Tensor[[384, 512], pl.FP32, pl.MemRef("mem_ddr_97", pl.const(0, pl.INT64), 786432)] = pl.tensor.create(
                [384, 512], dtype=pl.FP32, layout=pl.TensorLayout.ND
            )
            norm_w_2d_inline1909__ssa_v0: pl.Tensor[[1, 512], pl.FP32, pl.MemRef("mem_ddr_16", pl.const(0, pl.INT64), 2048)] = pl.tensor.reshape(cmp_norm_w__ssa_v0, [1, 512])
            ret__tmp_v0_10: pl.Tuple[pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(
                self.rmsnorm_rope_cache_write_spmd,
                bs_inline1923__ssa_v0,
                cmp_cos_il__rv_v2,
                cmp_sin_signed__rv_v2,
                pooled_kv_inline511__rv_v2,
                normed_kv_inline1917__ssa_v0,
                norm_w_2d_inline1909__ssa_v0,
                cmp_kv_cache_flat_inline1914__ssa_v0,
                kv_flat_inline1928__ssa_v0,
                cmp_slot_mapping__ssa_v0,
                deps=[pool_tid_inline507__ssa_v0, pool_tid_inline507__ssa_v0],
                core_num=rms_blocks_inline1919__ssa_v0,
                allow_early_resolve=True,
                attrs={"arg_directions": [pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.inout, pl.adir.input, pl.adir.output_existing, pl.adir.output_existing, pl.adir.input]},
            )
            cache_write_tid_inline1940__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_10[0]
            cmp_out__ssa_v1: pl.Tensor[[T_DYN, 512], pl.FP32, pl.MemRef("mem_ddr_98", pl.const(0, pl.INT64), 0)] = cmp_out__ssa_v0
            cmp_kv_score_tid__ssa_v0: pl.Scalar[pl.TASK_ID] = kv_score_tid_inline506__ssa_v0
            idx_kv_unused__ssa_v0: pl.Tensor[[T_DYN, 128], pl.FP32, pl.MemRef("mem_ddr_99", pl.const(0, pl.INT64), 0)] = pl.tensor.create([T_DYN, 128], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            normed_kv_inline518__ssa_v0: pl.Tensor[[384, 128], pl.BF16, pl.MemRef("mem_ddr_100", pl.const(0, pl.INT64), 98304)] = pl.tensor.create([384, 128], dtype=pl.BF16, layout=pl.TensorLayout.ND)
            kv_proj_pad_inline1986__ssa_v0: pl.Tensor[[384, 256], pl.FP32, pl.MemRef("mem_ddr_101", pl.const(0, pl.INT64), 393216)] = pl.tensor.create(
                [384, 256], dtype=pl.FP32, layout=pl.TensorLayout.ND
            )
            score_proj_pad_inline1993__ssa_v0: pl.Tensor[[384, 256], pl.FP32, pl.MemRef("mem_ddr_102", pl.const(0, pl.INT64), 393216)] = pl.tensor.create(
                [384, 256], dtype=pl.FP32, layout=pl.TensorLayout.ND
            )
            bs_inline377_inline2009__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(x_normed_t__ssa_v0, 0)
            t_matmul_inline370_inline2004__ssa_v0: pl.Scalar[pl.INDEX] = (bs_inline377_inline2009__ssa_v0 + 15) // 16 * 16
            x_flat_inline372_inline1997__ssa_v0: pl.Tensor[[T_DYN, 4096], pl.BF16, pl.MemRef("mem_ddr_103", pl.const(0, pl.INT64), 0)] = x_normed_t__ssa_v0
            ret__tmp_v0_11: pl.Tuple[pl.Tensor[[384, 256], pl.FP32], pl.Tensor[[384, 256], pl.FP32], pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(
                self.kv_score_proj_spmd_0,
                kv_proj_pad_inline1986__ssa_v0,
                score_proj_pad_inline1993__ssa_v0,
                t_matmul_inline370_inline2004__ssa_v0,
                bs_inline377_inline2009__ssa_v0,
                x_flat_inline372_inline1997__ssa_v0,
                inner_wkv__ssa_v0,
                inner_wgate__ssa_v0,
                deps=[late_dep__ssa_v0, late_dep__ssa_v0],
                core_num=24,
                attrs={"arg_directions": [pl.adir.output_existing, pl.adir.output_existing, pl.adir.scalar, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.input]},
            )
            kv_proj_pad_inline1986__rv_v2: pl.Tensor[[384, 256], pl.FP32, pl.MemRef("mem_ddr_104", pl.const(0, pl.INT64), 393216)] = ret__tmp_v0_11[0]
            score_proj_pad_inline1993__rv_v2: pl.Tensor[[384, 256], pl.FP32, pl.MemRef("mem_ddr_105", pl.const(0, pl.INT64), 393216)] = ret__tmp_v0_11[1]
            _kv_score_tid_inline383_inline2005__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_11[2]
            projection_tid_inline2002__ssa_v0: pl.Scalar[pl.TASK_ID] = _kv_score_tid_inline383_inline2005__ssa_v0
            b_dim_inline445_inline1977__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(inner_compress_state_block_table__ssa_v0, 0)
            bs_inline423_inline1991__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(position_ids__ssa_v0, 0)
            s_dim_inline427_inline2038__ssa_v0: pl.Scalar[pl.INDEX] = bs_inline423_inline1991__ssa_v0 // b_dim_inline445_inline1977__ssa_v0
            rms_blocks_inline422_inline1996__ssa_v0: pl.Scalar[pl.INDEX] = (bs_inline423_inline1991__ssa_v0 + 15) // 16
            compress_state_block_num_inline417_inline1982__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(inner_compress_state__ssa_v0, 0)
            compress_state_rows_inline425_inline2017__ssa_v0: pl.Scalar[pl.INDEX] = compress_state_block_num_inline417_inline1982__ssa_v0 * 2
            compress_state_flat_inline440_inline1999__ssa_v0: pl.Tensor[[compress_state_rows_inline425_inline2017__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_27", pl.const(0, pl.INT64), 0)] = (
                pl.tensor.reshape(inner_compress_state__ssa_v0, [compress_state_rows_inline425_inline2017__ssa_v0, 512])
            )
            _kv_score_tid_inline449_inline2031__ssa_v0: pl.Scalar[pl.TASK_ID] = projection_tid_inline2002__ssa_v0
            pooled_kv_inline461_inline2039__ssa_v0: pl.Tensor[[384, 128], pl.FP32, pl.MemRef("mem_ddr_106", pl.const(0, pl.INT64), 196608)] = pl.tensor.create(
                [384, 128], dtype=pl.FP32, layout=pl.TensorLayout.ND
            )
            pool_workers_inline435_inline1973__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(b_dim_inline445_inline1977__ssa_v0, 48)
            ret__tmp_v0_12: pl.Tuple[pl.Tensor[[384, 128], pl.FP32], pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(
                self.scatter_softmax_pool_spmd_0,
                pooled_kv_inline461_inline2039__ssa_v0,
                b_dim_inline445_inline1977__ssa_v0,
                pool_workers_inline435_inline1973__ssa_v0,
                position_ids__ssa_v0,
                s_dim_inline427_inline2038__ssa_v0,
                score_proj_pad_inline1993__rv_v2,
                inner_ape__ssa_v0,
                kv_proj_pad_inline1986__rv_v2,
                inner_compress_state_block_table__ssa_v0,
                compress_state_flat_inline440_inline1999__ssa_v0,
                deps=[_kv_score_tid_inline449_inline2031__ssa_v0],
                core_num=pool_workers_inline435_inline1973__ssa_v0,
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
            pooled_kv_inline461_inline2039__rv_v2: pl.Tensor[[384, 128], pl.FP32, pl.MemRef("mem_ddr_107", pl.const(0, pl.INT64), 196608)] = ret__tmp_v0_12[0]
            pool_tid_inline430_inline2033__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_12[1]
            commit_workers_inline462_inline2046__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(b_dim_inline445_inline1977__ssa_v0, 48)
            ret__tmp_v0_13: pl.Tuple[pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(
                self.compress_state_commit_spmd_0,
                compress_state_flat_inline440_inline1999__ssa_v0,
                b_dim_inline445_inline1977__ssa_v0,
                commit_workers_inline462_inline2046__ssa_v0,
                s_dim_inline427_inline2038__ssa_v0,
                inner_state_slot_mapping__ssa_v0,
                position_ids__ssa_v0,
                kv_proj_pad_inline1986__rv_v2,
                score_proj_pad_inline1993__rv_v2,
                inner_ape__ssa_v0,
                deps=[pool_tid_inline430_inline2033__ssa_v0],
                core_num=commit_workers_inline462_inline2046__ssa_v0,
                attrs={"arg_directions": [pl.adir.output_existing, pl.adir.scalar, pl.adir.scalar, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input]},
            )
            tid__ssa_v6: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_13[0]
            norm_w_2d_inline465_inline1970__ssa_v0: pl.Tensor[[1, 128], pl.FP32, pl.MemRef("mem_ddr_26", pl.const(0, pl.INT64), 512)] = pl.tensor.reshape(inner_norm_w__ssa_v0, [1, 128])
            ret__tmp_v0_14: pl.Tuple[pl.Tensor[[384, 128], pl.BF16], pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(
                self.indexer_boundary_init_spmd,
                normed_kv_inline518__ssa_v0,
                deps=[pool_tid_inline430_inline2033__ssa_v0],
                core_num=b_dim_inline445_inline1977__ssa_v0,
                attrs={"arg_directions": [pl.adir.output_existing]},
            )
            normed_kv_inline518__ssa_v1: pl.Tensor[[384, 128], pl.BF16, pl.MemRef("mem_ddr_108", pl.const(0, pl.INT64), 98304)] = ret__tmp_v0_14[0]
            boundary_init_tid_inline466_inline1969__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_14[1]
            rms_workers_inline457_inline1967__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(rms_blocks_inline422_inline1996__ssa_v0, 2)
            ret__tmp_v0_15: pl.Tuple[pl.Tensor[[384, 128], pl.BF16], pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(
                self.rmsnorm_rope_spmd,
                normed_kv_inline518__ssa_v1,
                rms_blocks_inline422_inline1996__ssa_v0,
                rms_workers_inline457_inline1967__ssa_v0,
                bs_inline423_inline1991__ssa_v0,
                inner_cos_il__rv_v2,
                inner_sin_signed__rv_v2,
                pooled_kv_inline461_inline2039__rv_v2,
                norm_w_2d_inline465_inline1970__ssa_v0,
                position_ids__ssa_v0,
                deps=[pool_tid_inline430_inline2033__ssa_v0, boundary_init_tid_inline466_inline1969__ssa_v0],
                core_num=rms_workers_inline457_inline1967__ssa_v0,
                attrs={"arg_directions": [pl.adir.inout, pl.adir.scalar, pl.adir.scalar, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input]},
            )
            normed_kv_inline518__rv_v3: pl.Tensor[[384, 128], pl.BF16, pl.MemRef("mem_ddr_109", pl.const(0, pl.INT64), 98304)] = ret__tmp_v0_15[0]
            rms_tid_inline409_inline2022__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_15[1]
            rms_tid_inline1941__ssa_v0: pl.Scalar[pl.TASK_ID] = rms_tid_inline409_inline2022__ssa_v0
            rms_tid_inline515__ssa_v0: pl.Scalar[pl.TASK_ID] = rms_tid_inline1941__ssa_v0
            bs_inline2079__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(position_ids__ssa_v0, 0)
            compact_rows_inline2073__ssa_v0: pl.Scalar[pl.INDEX] = bs_inline2079__ssa_v0 // 6 * 2
            rms_blocks_inline2072__ssa_v0: pl.Scalar[pl.INDEX] = (compact_rows_inline2073__ssa_v0 + 15) // 16
            kv_flat_inline2077__ssa_v0: pl.Tensor[[T_DYN, 128], pl.FP32, pl.MemRef("mem_ddr_110", pl.const(0, pl.INT64), 0)] = idx_kv_unused__ssa_v0
            idx_kv_scale_values_inline2059__ssa_v0: pl.Tensor[[384, 1], pl.FP32, pl.MemRef("mem_ddr_111", pl.const(0, pl.INT64), 1536)] = pl.tensor.create(
                [384, 1], dtype=pl.FP32, layout=pl.TensorLayout.ND
            )
            kv_final_inline2068__ssa_v0: pl.Tensor[[384, 128], pl.FP32, pl.MemRef("mem_ddr_112", pl.const(0, pl.INT64), 196608)] = pl.tensor.create(
                [384, 128], dtype=pl.FP32, layout=pl.TensorLayout.ND
            )
            ret__tmp_v0_16: pl.Tuple[pl.Tensor[[384, 128], pl.FP32], pl.Scalar[pl.TASK_ID]] = pl.submit(
                self.kv_hadamard,
                kv_final_inline2068__ssa_v0,
                hadamard_idx__ssa_v0,
                rms_blocks_inline2072__ssa_v0,
                compact_rows_inline2073__ssa_v0,
                normed_kv_inline518__rv_v3,
                deps=[rms_tid_inline515__ssa_v0, cmp_kv_score_tid__ssa_v0],
                attrs={"arg_directions": [pl.adir.output_existing, pl.adir.input, pl.adir.scalar, pl.adir.scalar, pl.adir.input]},
            )
            kv_final_inline2068__rv_v2: pl.Tensor[[384, 128], pl.FP32, pl.MemRef("mem_ddr_113", pl.const(0, pl.INT64), 196608)] = ret__tmp_v0_16[0]
            hadamard_tid_inline2069__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_16[1]
            ret__tmp_v0_17: pl.Tuple[pl.Tensor[[384, 1], pl.FP32], pl.Tensor[[IDX_CACHE_BLOCK_NUM_DYN, INDEXER_PAGE_BYTES_DYN], pl.INT8], pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(
                self.kv_and_cache_write_spmd,
                compact_rows_inline2073__ssa_v0,
                kv_final_inline2068__rv_v2,
                idx_kv_scale_values_inline2059__ssa_v0,
                idx_kv_cache__ssa_v0,
                kv_flat_inline2077__ssa_v0,
                position_ids__ssa_v0,
                idx_slot_mapping__ssa_v0,
                deps=[hadamard_tid_inline2069__ssa_v0],
                core_num=rms_blocks_inline2072__ssa_v0,
                allow_early_resolve=True,
                attrs={"arg_directions": [pl.adir.scalar, pl.adir.input, pl.adir.output_existing, pl.adir.inout, pl.adir.output_existing, pl.adir.input, pl.adir.input]},
            )
            idx_kv_scale_values_inline2059__ssa_v1: pl.Tensor[[384, 1], pl.FP32, pl.MemRef("mem_ddr_114", pl.const(0, pl.INT64), 1536)] = ret__tmp_v0_17[0]
            idx_kv_cache__rv_v2: pl.Tensor[[IDX_CACHE_BLOCK_NUM_DYN, INDEXER_PAGE_BYTES_DYN], pl.INT8, pl.MemRef("mem_ddr_115", pl.const(0, pl.INT64), 0)] = ret__tmp_v0_17[1]
            _write_tid_inline2084__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_17[2]
            ret__tmp_v0_18: pl.Tuple[pl.Tensor[[IDX_CACHE_BLOCK_NUM_DYN, INDEXER_PAGE_BYTES_DYN], pl.INT8], pl.Scalar[pl.TASK_ID]] = pl.submit(
                self.idx_kv_scale_commit,
                compact_rows_inline2073__ssa_v0,
                position_ids__ssa_v0,
                idx_slot_mapping__ssa_v0,
                idx_kv_cache__rv_v2,
                idx_kv_scale_values_inline2059__ssa_v1,
                deps=[_write_tid_inline2084__ssa_v0],
                allow_early_resolve=True,
                attrs={"arg_directions": [pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.inout, pl.adir.input]},
            )
            idx_kv_cache__rv_v3: pl.Tensor[[IDX_CACHE_BLOCK_NUM_DYN, INDEXER_PAGE_BYTES_DYN], pl.INT8, pl.MemRef("mem_ddr_116", pl.const(0, pl.INT64), 0)] = ret__tmp_v0_18[0]
            scale_commit_tid_inline2076__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_18[1]
            write_tid_inline512__ssa_v0: pl.Scalar[pl.TASK_ID] = scale_commit_tid_inline2076__ssa_v0
            idx_cache_write_tid__ssa_v0: pl.Scalar[pl.TASK_ID] = write_tid_inline512__ssa_v0
            qr_hadamard_i8_inline527__ssa_v0: pl.Tensor[[24576, 128], pl.INT8, pl.MemRef("mem_ddr_117", pl.const(0, pl.INT64), 3145728)] = pl.tensor.create(
                [24576, 128], dtype=pl.INT8, layout=pl.TensorLayout.ND
            )
            qr_hadamard_scale_dq_inline525__ssa_v0: pl.Tensor[[24576, 1], pl.FP32, pl.MemRef("mem_ddr_118", pl.const(0, pl.INT64), 98304)] = pl.tensor.create(
                [24576, 1], dtype=pl.FP32, layout=pl.TensorLayout.ND
            )
            qr_bf16_inline2161__ssa_v0: pl.Tensor[[24576, 128], pl.BF16, pl.MemRef("mem_ddr_119", pl.const(0, pl.INT64), 6291456)] = pl.tensor.create(
                [24576, 128], dtype=pl.BF16, layout=pl.TensorLayout.ND
            )
            bs_inline913_inline2122__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(x_normed_t__ssa_v0, 0)
            row_blocks_inline909_inline2135__ssa_v0: pl.Scalar[pl.INDEX] = (bs_inline913_inline2122__ssa_v0 + 15) // 16
            qr_acc_pad_inline911_inline2125__ssa_v0: pl.Tensor[[384, 8192], pl.INT32, pl.MemRef("mem_ddr_120", pl.const(0, pl.INT64), 12582912)] = pl.tensor.create(
                [384, 8192], dtype=pl.INT32, layout=pl.TensorLayout.ND
            )
            ret__tmp_v0_19: pl.Tuple[pl.Tensor[[384, 8192], pl.INT32], pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(
                self.idx_qr_proj_matmul_spmd,
                qr_acc_pad_inline911_inline2125__ssa_v0,
                row_blocks_inline909_inline2135__ssa_v0,
                bs_inline913_inline2122__ssa_v0,
                qr__ssa_v0,
                idx_wq_b__ssa_v0,
                core_num=24,
                allow_early_resolve=True,
                attrs={"arg_directions": [pl.adir.output_existing, pl.adir.scalar, pl.adir.scalar, pl.adir.input, pl.adir.input]},
            )
            qr_acc_pad_inline911_inline2125__rv_v2: pl.Tensor[[384, 8192], pl.INT32, pl.MemRef("mem_ddr_121", pl.const(0, pl.INT64), 12582912)] = ret__tmp_v0_19[0]
            idx_qr_mm_tid_inline903_inline2134__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_19[1]
            qr_bf16_2d_inline928_inline2152__ssa_v0: pl.Tensor[[384, 8192], pl.BF16, pl.MemRef("mem_ddr_119", pl.const(0, pl.INT64), 6291456)] = pl.tensor.reshape(
                qr_bf16_inline2161__ssa_v0, [384, 8192]
            )
            dq_rope_units_inline900_inline2142__ssa_v0: pl.Scalar[pl.INDEX] = bs_inline913_inline2122__ssa_v0 // 8 * 16
            dq_rope_workers_inline924_inline2119__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(dq_rope_units_inline900_inline2142__ssa_v0, 48)
            ret__tmp_v0_20: pl.Tuple[pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(
                self.idx_qr_dequant_rope_spmd,
                qr_bf16_2d_inline928_inline2152__ssa_v0,
                dq_rope_units_inline900_inline2142__ssa_v0,
                dq_rope_workers_inline924_inline2119__ssa_v0,
                qr_scale__ssa_v0,
                idx_cos_il__rv_v2,
                idx_sin_signed__rv_v2,
                idx_wq_b_scale__ssa_v0,
                qr_acc_pad_inline911_inline2125__rv_v2,
                core_num=dq_rope_workers_inline924_inline2119__ssa_v0,
                allow_early_resolve=True,
                attrs={"arg_directions": [pl.adir.output_existing, pl.adir.scalar, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input]},
            )
            tid__ssa_v7: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_20[0]
            idx_qr_mm_tid_inline2138__ssa_v0: pl.Scalar[pl.TASK_ID] = idx_qr_mm_tid_inline903_inline2134__ssa_v0
            bs_inline939_inline2114__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(x_normed_t__ssa_v0, 0)
            bs_heads_inline945_inline2111__ssa_v0: pl.Scalar[pl.INDEX] = bs_inline939_inline2114__ssa_v0 * 64
            qh_acc_gm_inline943_inline2148__ssa_v0: pl.Tensor[[bs_heads_inline945_inline2111__ssa_v0, 128], pl.FP32, pl.MemRef("mem_ddr_122", pl.const(0, pl.INT64), 0)] = pl.tensor.create(
                [bs_heads_inline945_inline2111__ssa_v0, 128], dtype=pl.FP32, layout=pl.TensorLayout.ND
            )
            ret__tmp_v0_21: pl.Tuple[pl.Tensor[[bs_heads_inline945_inline2111__ssa_v0, 128], pl.FP32], pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(
                self.qr_hadamard_matmul_spmd,
                hadamard_idx__ssa_v0,
                qh_acc_gm_inline943_inline2148__ssa_v0,
                bs_heads_inline945_inline2111__ssa_v0,
                qr_bf16_inline2161__ssa_v0,
                deps=[idx_qr_mm_tid_inline2138__ssa_v0],
                core_num=24,
                allow_early_resolve=True,
                attrs={"arg_directions": [pl.adir.input, pl.adir.output_existing, pl.adir.scalar, pl.adir.input]},
            )
            qh_acc_gm_inline943_inline2148__rv_v2: pl.Tensor[[bs_heads_inline945_inline2111__ssa_v0, 128], pl.FP32, pl.MemRef("mem_ddr_123", pl.const(0, pl.INT64), 0)] = ret__tmp_v0_21[0]
            qh_mm_tid_inline940_inline2162__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_21[1]
            ret__tmp_v0_22: pl.Tuple[pl.Tensor[[24576, 128], pl.INT8], pl.Tensor[[24576, 1], pl.FP32], pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(
                self.qr_hadamard_quant_spmd,
                qr_hadamard_i8_inline527__ssa_v0,
                qr_hadamard_scale_dq_inline525__ssa_v0,
                bs_heads_inline945_inline2111__ssa_v0,
                qh_acc_gm_inline943_inline2148__rv_v2,
                core_num=48,
                allow_early_resolve=True,
                attrs={"arg_directions": [pl.adir.output_existing, pl.adir.output_existing, pl.adir.scalar, pl.adir.input]},
            )
            qr_hadamard_i8_inline527__rv_v2: pl.Tensor[[24576, 128], pl.INT8, pl.MemRef("mem_ddr_124", pl.const(0, pl.INT64), 3145728)] = ret__tmp_v0_22[0]
            qr_hadamard_scale_dq_inline525__rv_v2: pl.Tensor[[24576, 1], pl.FP32, pl.MemRef("mem_ddr_125", pl.const(0, pl.INT64), 98304)] = ret__tmp_v0_22[1]
            qh_quant_tid_inline936_inline2118__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_22[2]
            qh_quant_tid_inline2095__ssa_v0: pl.Scalar[pl.TASK_ID] = qh_quant_tid_inline936_inline2118__ssa_v0
            qh_quant_tid_inline522__ssa_v0: pl.Scalar[pl.TASK_ID] = qh_quant_tid_inline2095__ssa_v0
            weights_gate_dep_inline520__ssa_v0: pl.Scalar[pl.TASK_ID] = pl.system.task_dummy(deps=[])
            bs_inline2214__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(x_normed_t__ssa_v0, 0)
            row_blocks_inline2215__ssa_v0: pl.Scalar[pl.INDEX] = (bs_inline2214__ssa_v0 + 15) // 16
            x_flat_inline2226__ssa_v0: pl.Tensor[[T_DYN, 4096], pl.BF16, pl.MemRef("mem_ddr_126", pl.const(0, pl.INT64), 0)] = x_normed_t__ssa_v0
            weights_inline2220__ssa_v0: pl.Tensor[[384, 64], pl.FP32, pl.MemRef("mem_ddr_127", pl.const(0, pl.INT64), 98304)] = pl.tensor.create([384, 64], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            weights_partial_inline2262__ssa_v0: pl.Tensor[[1536, 64], pl.FP32, pl.MemRef("mem_ddr_128", pl.const(0, pl.INT64), 393216)] = pl.tensor.create(
                [1536, 64], dtype=pl.FP32, layout=pl.TensorLayout.ND
            )
            ret__tmp_v0_23: pl.Tuple[pl.Tensor[[1536, 64], pl.FP32], pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(
                self.weights_proj_spmd,
                weights_partial_inline2262__ssa_v0,
                row_blocks_inline2215__ssa_v0,
                bs_inline2214__ssa_v0,
                x_flat_inline2226__ssa_v0,
                weights_proj__ssa_v0,
                deps=[weights_gate_dep_inline520__ssa_v0],
                core_num=8,
                allow_early_resolve=True,
                attrs={"arg_directions": [pl.adir.output_existing, pl.adir.scalar, pl.adir.scalar, pl.adir.input, pl.adir.input]},
            )
            weights_partial_inline2262__rv_v2: pl.Tensor[[1536, 64], pl.FP32, pl.MemRef("mem_ddr_129", pl.const(0, pl.INT64), 393216)] = ret__tmp_v0_23[0]
            _weights_tid_inline2199__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_23[1]
            ret__tmp_v0_24: pl.Tuple[pl.Tensor[[384, 64], pl.FP32], pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(
                self.weights_proj_reduce_spmd,
                weights_partial_inline2262__rv_v2,
                weights_inline2220__ssa_v0,
                core_num=row_blocks_inline2215__ssa_v0,
                allow_early_resolve=True,
                attrs={"arg_directions": [pl.adir.input, pl.adir.output_existing]},
            )
            weights_inline2220__ssa_v1: pl.Tensor[[384, 64], pl.FP32, pl.MemRef("mem_ddr_130", pl.const(0, pl.INT64), 98304)] = ret__tmp_v0_24[0]
            weights_tid_inline2204__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_24[1]
            b_dim_inline1027_inline2198__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(idx_block_table__ssa_v0, 0)
            table_columns_inline1029_inline2268__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(idx_block_table__ssa_v0, 1)
            idx_table_len_inline1025_inline2222__ssa_v0: pl.Scalar[pl.INDEX] = b_dim_inline1027_inline2198__ssa_v0 * table_columns_inline1029_inline2268__ssa_v0
            idx_block_table_flat_inline1044_inline2217__ssa_v0: pl.Tensor[[idx_table_len_inline1025_inline2222__ssa_v0], pl.INT32, pl.MemRef("mem_ddr_33", pl.const(0, pl.INT64), 0)] = (
                pl.tensor.reshape(idx_block_table__ssa_v0, [idx_table_len_inline1025_inline2222__ssa_v0])
            )
            pair_arena_inline1035_inline2211__ssa_v0: pl.Tensor[[24576, 1024], pl.FP32, pl.MemRef("mem_ddr_131", pl.const(0, pl.INT64), 100663296)] = pl.tensor.create(
                [24576, 1024], dtype=pl.FP32, layout=pl.TensorLayout.ND
            )
            score_arena_inline1043_inline2235__ssa_v0: pl.Tensor[[384, 8192], pl.FP32, pl.MemRef("mem_ddr_132", pl.const(0, pl.INT64), 12582912)] = pl.tensor.create(
                [384, 8192], dtype=pl.FP32, layout=pl.TensorLayout.ND
            )
            gm_pipe_buffer_0: pl.Tensor[[1], pl.FP32, pl.MemRef("mem_ddr_133", pl.const(0, pl.INT64), 4)] = pl.tensor.create([1], dtype=pl.FP32, layout=pl.TensorLayout.ND, manual_dep=True)
            ret__tmp_v0_25: pl.Tuple[pl.Tensor[[384, 8192], pl.FP32], pl.Tensor[[24576, 1024], pl.FP32], pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(
                self.indexer_score_topk_leaf_spmd,
                position_ids__ssa_v0,
                kv_seq_lens__ssa_v0,
                score_arena_inline1043_inline2235__ssa_v0,
                qr_hadamard_i8_inline527__rv_v2,
                qr_hadamard_scale_dq_inline525__rv_v2,
                weights_inline2220__ssa_v1,
                idx_block_table_flat_inline1044_inline2217__ssa_v0,
                table_columns_inline1029_inline2268__ssa_v0,
                idx_kv_cache__rv_v3,
                pair_arena_inline1035_inline2211__ssa_v0,
                gm_pipe_buffer_0,
                deps=[qh_quant_tid_inline522__ssa_v0, weights_tid_inline2204__ssa_v0, idx_cache_write_tid__ssa_v0],
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
                        pl.adir.input,
                        pl.adir.scalar,
                        pl.adir.input,
                        pl.adir.output_existing,
                        pl.adir.output_existing,
                    ]
                },
            )
            score_arena_inline1043_inline2235__rv_v2: pl.Tensor[[384, 8192], pl.FP32, pl.MemRef("mem_ddr_134", pl.const(0, pl.INT64), 12582912)] = ret__tmp_v0_25[0]
            pair_arena_inline1035_inline2211__ssa_v1: pl.Tensor[[24576, 1024], pl.FP32, pl.MemRef("mem_ddr_135", pl.const(0, pl.INT64), 100663296)] = ret__tmp_v0_25[1]
            score_tid_inline1039_inline2237__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_25[2]
            for topk_batch_inline1033_inline2259__idx_v0, (max_topk_cache_len_inline1036_inline2181__iter_v1,) in pl.range(
                b_dim_inline1027_inline2198__ssa_v0, init_values=(0,), attrs={"iter_arg_rebind_0": True}
            ):
                t__tmp_v227: pl.Scalar[pl.INT32] = pl.tensor.read(kv_seq_lens__ssa_v0, [topk_batch_inline1033_inline2259__idx_v0])
                topk_cache_len_inline1021_inline2183__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(t__tmp_v227, pl.INDEX) // 4
                max_topk_cache_len_inline1036_inline2181__ssa_v3: pl.Scalar[pl.INDEX] = pl.max(max_topk_cache_len_inline1036_inline2181__iter_v1, topk_cache_len_inline1021_inline2183__ssa_v0)
                max_topk_cache_len_inline1036_inline2181__rv_v2: pl.Scalar[pl.INDEX] = pl.yield_(max_topk_cache_len_inline1036_inline2181__ssa_v3)
            with pl.scope():
                if max_topk_cache_len_inline1036_inline2181__rv_v2 <= 8192:
                    ret__tmp_v0_26: pl.Tuple[pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(
                        self._decode_csa_tp1_attention_indexer_topk_single_leaf_publish,
                        position_ids__ssa_v0,
                        kv_seq_lens__ssa_v0,
                        score_arena_inline1043_inline2235__rv_v2,
                        idx_topk_scores__ssa_v0,
                        idx_topk__ssa_v0,
                        deps=[score_tid_inline1039_inline2237__ssa_v0],
                        core_num=48,
                        allow_early_resolve=True,
                        attrs={"arg_directions": [pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.output_existing, pl.adir.output_existing]},
                    )
                    tid__ssa_v8: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_26[0]
                else:
                    ret__tmp_v0_27: pl.Tuple[pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(
                        self._decode_csa_tp1_attention_indexer_topk_query_merge,
                        position_ids__ssa_v0,
                        kv_seq_lens__ssa_v0,
                        pair_arena_inline1035_inline2211__ssa_v1,
                        idx_topk_scores__ssa_v0,
                        idx_topk__ssa_v0,
                        deps=[score_tid_inline1039_inline2237__ssa_v0],
                        core_num=48,
                        allow_early_resolve=True,
                        attrs={"arg_directions": [pl.adir.input, pl.adir.input, pl.adir.inout, pl.adir.output_existing, pl.adir.output_existing]},
                    )
                    tid__ssa_v9: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_27[0]
            idx_topk_scores__ssa_v1: pl.Tensor[[T_DYN, 512], pl.FP32, pl.MemRef("mem_ddr_136", pl.const(0, pl.INT64), 0)] = idx_topk_scores__ssa_v0
            idx_topk__ssa_v1: pl.Tensor[[T_DYN, 512], pl.INT32, pl.MemRef("mem_ddr_137", pl.const(0, pl.INT64), 0)] = idx_topk__ssa_v0
            idx_topk_scores__ssa_v2: pl.Tensor[[T_DYN, 512], pl.FP32, pl.MemRef("mem_ddr_138", pl.const(0, pl.INT64), 0)] = idx_topk_scores__ssa_v1
            idx_topk__ssa_v2: pl.Tensor[[T_DYN, 512], pl.INT32, pl.MemRef("mem_ddr_139", pl.const(0, pl.INT64), 0)] = idx_topk__ssa_v1
            idx_topk_scores__ssa_v3: pl.Tensor[[T_DYN, 512], pl.FP32, pl.MemRef("mem_ddr_140", pl.const(0, pl.INT64), 0)] = idx_topk_scores__ssa_v2
            idx_topk__ssa_v3: pl.Tensor[[T_DYN, 512], pl.INT32, pl.MemRef("mem_ddr_141", pl.const(0, pl.INT64), 0)] = idx_topk__ssa_v2
            o_packed_heads__ssa_v0: pl.Tensor[[3072, 4096], pl.BF16, pl.MemRef("mem_ddr_142", pl.const(0, pl.INT64), 25165824)] = pl.tensor.create(
                [3072, 4096], dtype=pl.BF16, layout=pl.TensorLayout.ND
            )
            ori_block_num_inline2385__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(kv_cache__ssa_v0, 0)
            t_dim_inline2318__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(q__ssa_v0, 0)
            t_heads_inline2333__ssa_v0: pl.Scalar[pl.INDEX] = t_dim_inline2318__ssa_v0 * 64
            rope_cs_blocks_inline2348__ssa_v0: pl.Scalar[pl.INDEX] = t_dim_inline2318__ssa_v0 // 8
            ori_kv_flat_inline2377__ssa_v0: pl.Tensor[[ori_block_num_inline2385__ssa_v0 * pl.const(32, pl.INDEX), 512], pl.BF16, pl.MemRef("mem_ddr_29", pl.const(0, pl.INT64), 0)] = pl.tensor.reshape(
                kv_cache__ssa_v0, [ori_block_num_inline2385__ssa_v0 * 32, 512]
            )
            ret__tmp_v0_28: pl.Tuple[pl.Tensor[[ori_block_num_inline2385__ssa_v0 * pl.const(32, pl.INDEX), 512], pl.BF16], pl.Scalar[pl.TASK_ID]] = pl.submit(
                self.kv_touch, ori_kv_flat_inline2377__ssa_v0, allow_early_resolve=True, attrs={"arg_directions": [pl.adir.inout]}
            )
            ori_kv_flat_inline2377__ssa_v1: pl.Tensor[[ori_block_num_inline2385__ssa_v0 * pl.const(32, pl.INDEX), 512], pl.BF16, pl.MemRef("mem_ddr_143", pl.const(0, pl.INT64), 0)] = ret__tmp_v0_28[0]
            tid__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_28[1]
            sparse_bias_inline2369__ssa_v0: pl.Tensor[[t_dim_inline2318__ssa_v0, 640], pl.FP32, pl.MemRef("mem_ddr_144", pl.const(0, pl.INT64), 0)] = pl.tensor.create(
                [t_dim_inline2318__ssa_v0, 640], dtype=pl.FP32, layout=pl.TensorLayout.ND
            )
            cmp_sparse_indices_inline2323__ssa_v0: pl.Tensor[[t_dim_inline2318__ssa_v0, 512], pl.INT32, pl.MemRef("mem_ddr_145", pl.const(0, pl.INT64), 0)] = pl.tensor.create(
                [t_dim_inline2318__ssa_v0, 512], dtype=pl.INT32, layout=pl.TensorLayout.ND
            )
            valid_block_mask_inline2310__ssa_v0: pl.Tensor[[t_dim_inline2318__ssa_v0, 16], pl.INT32, pl.MemRef("mem_ddr_146", pl.const(0, pl.INT64), 0)] = pl.tensor.create(
                [t_dim_inline2318__ssa_v0, 16], dtype=pl.INT32, layout=pl.TensorLayout.ND
            )
            ret__tmp_v0_29: pl.Tuple[pl.Tensor[[t_dim_inline2318__ssa_v0, 512], pl.INT32], pl.Tensor[[t_dim_inline2318__ssa_v0, 640], pl.FP32], pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(
                self.csa_slots_build_valid_qk_plan_spmd,
                cmp_sparse_indices_inline2323__ssa_v0,
                sparse_bias_inline2369__ssa_v0,
                t_dim_inline2318__ssa_v0,
                idx_topk__ssa_v3,
                position_ids_t1__ssa_v0,
                window_swa_indices__ssa_v0,
                valid_block_mask_inline2310__ssa_v0,
                core_num=16,
                allow_early_resolve=True,
                attrs={"arg_directions": [pl.adir.output_existing, pl.adir.output_existing, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.output_existing]},
            )
            cmp_sparse_indices_inline2323__rv_v2: pl.Tensor[[t_dim_inline2318__ssa_v0, 512], pl.INT32, pl.MemRef("mem_ddr_147", pl.const(0, pl.INT64), 0)] = ret__tmp_v0_29[0]
            sparse_bias_inline2369__rv_v2: pl.Tensor[[t_dim_inline2318__ssa_v0, 640], pl.FP32, pl.MemRef("mem_ddr_148", pl.const(0, pl.INT64), 0)] = ret__tmp_v0_29[1]
            qk_plan_tid_inline2335__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_29[2]
            cmp_block_num_inline2362__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(cmp_kv__ssa_v0, 0)
            cmp_kv_flat_inline2399__ssa_v0: pl.Tensor[[cmp_block_num_inline2362__ssa_v0 * pl.const(32, pl.INDEX), 512], pl.BF16, pl.MemRef("mem_ddr_30", pl.const(0, pl.INT64), 0)] = pl.tensor.reshape(
                cmp_kv__ssa_v0, [cmp_block_num_inline2362__ssa_v0 * 32, 512]
            )
            q_flat_inline2306__ssa_v0: pl.Tensor[[t_heads_inline2333__ssa_v0, 512], pl.BF16, pl.MemRef("mem_ddr_62", pl.const(0, pl.INT64), 0)] = pl.tensor.reshape(
                q__ssa_v0, [t_heads_inline2333__ssa_v0, 512]
            )
            attn_sink_col_inline2332__ssa_v0: pl.Tensor[[64, 1], pl.FP32, pl.MemRef("mem_ddr_43", pl.const(0, pl.INT64), 256)] = pl.tensor.reshape(attn_sink__ssa_v0, [64, 1])
            attn_mi_inline2305__ssa_v0: pl.Tensor[[t_heads_inline2333__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_149", pl.const(0, pl.INT64), 0)] = pl.tensor.create(
                [t_heads_inline2333__ssa_v0, 1], dtype=pl.FP32, layout=pl.TensorLayout.ND
            )
            attn_li_inline2396__ssa_v0: pl.Tensor[[t_heads_inline2333__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_150", pl.const(0, pl.INT64), 0)] = pl.tensor.create(
                [t_heads_inline2333__ssa_v0, 1], dtype=pl.FP32, layout=pl.TensorLayout.ND
            )
            attn_oi_inline2303__ssa_v0: pl.Tensor[[t_heads_inline2333__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_151", pl.const(0, pl.INT64), 0)] = pl.tensor.create(
                [t_heads_inline2333__ssa_v0, 512], dtype=pl.FP32, layout=pl.TensorLayout.ND
            )
            kv_transfer_inline2324__ssa_v0: pl.Tensor[[9216, 512], pl.BF16, pl.MemRef("mem_ddr_152", pl.const(0, pl.INT64), 9437184)] = pl.tensor.create(
                [9216, 512], dtype=pl.BF16, layout=pl.TensorLayout.ND
            )
            score_transfer_inline2293__ssa_v0: pl.Tensor[[4608, 128], pl.FP32, pl.MemRef("mem_ddr_153", pl.const(0, pl.INT64), 2359296)] = pl.tensor.create(
                [4608, 128], dtype=pl.FP32, layout=pl.TensorLayout.ND
            )
            probability_transfer_inline2339__ssa_v0: pl.Tensor[[4608, 128], pl.BF16, pl.MemRef("mem_ddr_154", pl.const(0, pl.INT64), 1179648)] = pl.tensor.create(
                [4608, 128], dtype=pl.BF16, layout=pl.TensorLayout.ND
            )
            pv_transfer_inline2347__ssa_v0: pl.Tensor[[4608, 512], pl.FP32, pl.MemRef("mem_ddr_155", pl.const(0, pl.INT64), 9437184)] = pl.tensor.create(
                [4608, 512], dtype=pl.FP32, layout=pl.TensorLayout.ND
            )
            mi_transfer_inline2291__ssa_v0: pl.Tensor[[4608, 1], pl.FP32, pl.MemRef("mem_ddr_156", pl.const(0, pl.INT64), 18432)] = pl.tensor.create(
                [4608, 1], dtype=pl.FP32, layout=pl.TensorLayout.ND
            )
            li_transfer_inline2388__ssa_v0: pl.Tensor[[4608, 1], pl.FP32, pl.MemRef("mem_ddr_157", pl.const(0, pl.INT64), 18432)] = pl.tensor.create(
                [4608, 1], dtype=pl.FP32, layout=pl.TensorLayout.ND
            )
            ffts_workspace_inline2289__ssa_v0: pl.Tensor[[256], pl.INT64, pl.MemRef("mem_ddr_158", pl.const(0, pl.INT64), 2048)] = pl.tensor.create([256], dtype=pl.INT64, layout=pl.TensorLayout.ND)
            ret__tmp_v0_30: pl.Tuple[
                pl.Tensor[[t_heads_inline2333__ssa_v0, 1], pl.FP32], pl.Tensor[[t_heads_inline2333__ssa_v0, 1], pl.FP32], pl.Tensor[[t_heads_inline2333__ssa_v0, 512], pl.FP32], pl.Scalar[pl.TASK_ID]
            ] = pl.spmd_submit(
                self.qk_pv_spmd,
                ffts_workspace_inline2289__ssa_v0,
                t_dim_inline2318__ssa_v0,
                q_flat_inline2306__ssa_v0,
                valid_block_mask_inline2310__ssa_v0,
                kv_transfer_inline2324__ssa_v0,
                score_transfer_inline2293__ssa_v0,
                probability_transfer_inline2339__ssa_v0,
                pv_transfer_inline2347__ssa_v0,
                attn_sink_col_inline2332__ssa_v0,
                position_ids_t1__ssa_v0,
                window_swa_indices__ssa_v0,
                ori_kv_flat_inline2377__ssa_v1,
                cmp_sparse_indices_inline2323__rv_v2,
                cmp_block_table__ssa_v0,
                cmp_kv_flat_inline2399__ssa_v0,
                sparse_bias_inline2369__rv_v2,
                mi_transfer_inline2291__ssa_v0,
                li_transfer_inline2388__ssa_v0,
                attn_mi_inline2305__ssa_v0,
                attn_li_inline2396__ssa_v0,
                attn_oi_inline2303__ssa_v0,
                deps=[qk_plan_tid_inline2335__ssa_v0],
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
                        pl.adir.input,
                        pl.adir.input,
                        pl.adir.input,
                        pl.adir.input,
                        pl.adir.input,
                        pl.adir.input,
                        pl.adir.input,
                        pl.adir.inout,
                        pl.adir.inout,
                        pl.adir.output_existing,
                        pl.adir.output_existing,
                        pl.adir.output_existing,
                    ]
                },
            )
            attn_mi_inline2305__ssa_v1: pl.Tensor[[t_heads_inline2333__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_159", pl.const(0, pl.INT64), 0)] = ret__tmp_v0_30[0]
            attn_li_inline2396__ssa_v1: pl.Tensor[[t_heads_inline2333__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_160", pl.const(0, pl.INT64), 0)] = ret__tmp_v0_30[1]
            attn_oi_inline2303__ssa_v1: pl.Tensor[[t_heads_inline2333__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_161", pl.const(0, pl.INT64), 0)] = ret__tmp_v0_30[2]
            qk_tid_inline2292__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_30[3]
            rope_cos_il_inline2281__ssa_v0: pl.Tensor[[384, 64], pl.FP32, pl.MemRef("mem_ddr_162", pl.const(0, pl.INT64), 98304)] = pl.tensor.create(
                [384, 64], dtype=pl.FP32, layout=pl.TensorLayout.ND
            )
            rope_sin_signed_inline2280__ssa_v0: pl.Tensor[[384, 64], pl.FP32, pl.MemRef("mem_ddr_163", pl.const(0, pl.INT64), 98304)] = pl.tensor.create(
                [384, 64], dtype=pl.FP32, layout=pl.TensorLayout.ND
            )
            rope_swap_idx_inline2361__ssa_v0: pl.Tensor[[16, 64], pl.INT32, pl.MemRef("mem_ddr_164", pl.const(0, pl.INT64), 4096)] = pl.tensor.create(
                [16, 64], dtype=pl.INT32, layout=pl.TensorLayout.ND
            )
            ret__tmp_v0_31: pl.Tuple[pl.Tensor[[16, 64], pl.INT32], pl.Tensor[[384, 64], pl.FP32], pl.Tensor[[384, 64], pl.FP32], pl.Scalar[pl.TASK_ID]] = pl.submit(
                self.rope_cs,
                rope_swap_idx_inline2361__ssa_v0,
                rope_cos_il_inline2281__ssa_v0,
                rope_sin_signed_inline2280__ssa_v0,
                rope_cs_blocks_inline2348__ssa_v0,
                freqs_cos__ssa_v0,
                freqs_sin__ssa_v0,
                allow_early_resolve=True,
                attrs={"arg_directions": [pl.adir.output_existing, pl.adir.output_existing, pl.adir.output_existing, pl.adir.scalar, pl.adir.input, pl.adir.input]},
            )
            rope_swap_idx_inline2361__ssa_v1: pl.Tensor[[16, 64], pl.INT32, pl.MemRef("mem_ddr_165", pl.const(0, pl.INT64), 4096)] = ret__tmp_v0_31[0]
            rope_cos_il_inline2281__rv_v2: pl.Tensor[[384, 64], pl.FP32, pl.MemRef("mem_ddr_166", pl.const(0, pl.INT64), 98304)] = ret__tmp_v0_31[1]
            rope_sin_signed_inline2280__rv_v2: pl.Tensor[[384, 64], pl.FP32, pl.MemRef("mem_ddr_167", pl.const(0, pl.INT64), 98304)] = ret__tmp_v0_31[2]
            rope_tid_inline2279__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_31[3]
            attn_mi_inline552__ssa_v0: pl.Tensor[[t_heads_inline2333__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_168", pl.const(0, pl.INT64), 0)] = attn_mi_inline2305__ssa_v1
            attn_li_inline560__ssa_v0: pl.Tensor[[t_heads_inline2333__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_169", pl.const(0, pl.INT64), 0)] = attn_li_inline2396__ssa_v1
            attn_oi_inline550__ssa_v0: pl.Tensor[[t_heads_inline2333__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_170", pl.const(0, pl.INT64), 0)] = attn_oi_inline2303__ssa_v1
            rope_cos_il_inline544__ssa_v0: pl.Tensor[[384, 64], pl.FP32, pl.MemRef("mem_ddr_171", pl.const(0, pl.INT64), 98304)] = rope_cos_il_inline2281__rv_v2
            rope_sin_signed_inline543__ssa_v0: pl.Tensor[[384, 64], pl.FP32, pl.MemRef("mem_ddr_172", pl.const(0, pl.INT64), 98304)] = rope_sin_signed_inline2280__rv_v2
            rope_swap_idx_inline574__ssa_v0: pl.Tensor[[16, 64], pl.INT32, pl.MemRef("mem_ddr_173", pl.const(0, pl.INT64), 4096)] = rope_swap_idx_inline2361__ssa_v1
            qk_tid_inline548__ssa_v0: pl.Scalar[pl.TASK_ID] = qk_tid_inline2292__ssa_v0
            rope_tid_inline547__ssa_v0: pl.Scalar[pl.TASK_ID] = rope_tid_inline2279__ssa_v0
            t_dim_inline541__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(q__ssa_v0, 0)
            merge_sink_inline570__ssa_v0: pl.Tensor[[64, 1], pl.FP32, pl.MemRef("mem_ddr_43", pl.const(0, pl.INT64), 256)] = pl.tensor.reshape(attn_sink__ssa_v0, [64, 1])
            ret__tmp_v0_32: pl.Tuple[pl.Tensor[[3072, 4096], pl.BF16], pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(
                self.merge_norm_spmd,
                rope_swap_idx_inline574__ssa_v0,
                t_dim_inline541__ssa_v0,
                attn_mi_inline552__ssa_v0,
                attn_li_inline560__ssa_v0,
                attn_oi_inline550__ssa_v0,
                merge_sink_inline570__ssa_v0,
                rope_cos_il_inline544__ssa_v0,
                rope_sin_signed_inline543__ssa_v0,
                o_packed_heads__ssa_v0,
                deps=[qk_tid_inline548__ssa_v0, rope_tid_inline547__ssa_v0],
                core_num=48,
                attrs={"arg_directions": [pl.adir.input, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.output_existing]},
            )
            o_packed_heads__ssa_v2: pl.Tensor[[3072, 4096], pl.BF16, pl.MemRef("mem_ddr_174", pl.const(0, pl.INT64), 25165824)] = ret__tmp_v0_32[0]
            merge_tid_inline540__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_32[1]
            o_packed_heads__ssa_v1: pl.Tensor[[3072, 4096], pl.BF16, pl.MemRef("mem_ddr_175", pl.const(0, pl.INT64), 25165824)] = o_packed_heads__ssa_v2
            heads_dep__ssa_v0: pl.Scalar[pl.TASK_ID] = merge_tid_inline540__ssa_v0
            t_dim_inline607__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(attn_out__ssa_v0, 0)
            act_t_blks_inline611__ssa_v0: pl.Scalar[pl.INDEX] = (t_dim_inline607__ssa_v0 + 31) // 32
            proj_a_rows_inline625__ssa_v0: pl.Scalar[pl.INDEX] = (t_dim_inline607__ssa_v0 + 127) // 128
            proj_b_t_rows_inline658__ssa_v0: pl.Scalar[pl.INDEX] = (t_dim_inline607__ssa_v0 + 127) // 128
            proj_b_padded_rows_inline614__ssa_v0: pl.Scalar[pl.INDEX] = proj_b_t_rows_inline658__ssa_v0 * 128
            o_r_pad_inline620__ssa_v0: pl.Tensor[[384, 8192], pl.FP32, pl.MemRef("mem_ddr_176", pl.const(0, pl.INT64), 12582912)] = pl.tensor.create(
                [384, 8192], dtype=pl.FP32, layout=pl.TensorLayout.ND
            )
            o_r_i8_pad_inline621__ssa_v0: pl.Tensor[[384, 8192], pl.INT8, pl.MemRef("mem_ddr_177", pl.const(0, pl.INT64), 3145728)] = pl.tensor.create(
                [384, 8192], dtype=pl.INT8, layout=pl.TensorLayout.ND
            )
            act_scale_dq_inline598__ssa_v0: pl.Tensor[[8, 384], pl.FP32, pl.MemRef("mem_ddr_178", pl.const(0, pl.INT64), 12288)] = pl.tensor.create([8, 384], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            partials_inline627__ssa_v0: pl.Tensor[[384, 32768], pl.INT32, pl.MemRef("mem_ddr_179", pl.const(0, pl.INT64), 50331648)] = pl.tensor.create(
                [384, 32768], dtype=pl.INT32, layout=pl.TensorLayout.ND
            )
            proj_b_tids_inline634__ssa_v0: pl.Array[8, pl.TASK_ID] = pl.array.create(8, dtype=pl.TASK_ID)
            with pl.scope(mode=pl.ScopeMode.MANUAL):
                for g_inline628__idx_v0, (
                    act_scale_dq_inline598__iter_v1,
                    o_r_i8_pad_inline621__iter_v1,
                    o_r_pad_inline620__iter_v1,
                    partials_inline627__iter_v1,
                    proj_b_tids_inline634__iter_v1,
                ) in pl.parallel(
                    8,
                    init_values=(act_scale_dq_inline598__ssa_v0, o_r_i8_pad_inline621__ssa_v0, o_r_pad_inline620__ssa_v0, partials_inline627__ssa_v0, proj_b_tids_inline634__ssa_v0),
                    attrs={"iter_arg_rebind_0": False, "iter_arg_rebind_1": False, "iter_arg_rebind_2": False, "iter_arg_rebind_3": False, "iter_arg_rebind_4": True},
                ):
                    row_base_o_inline637__ssa_v0: pl.Scalar[pl.INDEX] = g_inline628__idx_v0 * 384
                    out_col_g_inline613__ssa_v0: pl.Scalar[pl.INDEX] = g_inline628__idx_v0 * 1024
                    ret__tmp_v0_33: pl.Tuple[pl.Tensor[[384, 8192], pl.FP32], pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(
                        self.proj_a_mm_spmd,
                        t_dim_inline607__ssa_v0,
                        row_base_o_inline637__ssa_v0,
                        o_packed_heads__ssa_v1,
                        wo_a__ssa_v0,
                        g_inline628__idx_v0,
                        o_r_pad_inline620__iter_v1,
                        out_col_g_inline613__ssa_v0,
                        deps=[heads_dep__ssa_v0],
                        core_num=proj_a_rows_inline625__ssa_v0 * 8,
                        allow_early_resolve=True,
                        attrs={"arg_directions": [pl.adir.scalar, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.scalar, pl.adir.output_existing, pl.adir.scalar]},
                    )
                    o_r_pad_inline620__ssa_v3: pl.Tensor[[384, 8192], pl.FP32, pl.MemRef("mem_ddr_180", pl.const(0, pl.INT64), 12582912)] = ret__tmp_v0_33[0]
                    pa_tid_inline635__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_33[1]
                    col_g_inline610__ssa_v0: pl.Scalar[pl.INDEX] = g_inline628__idx_v0 * 1024
                    ret__tmp_v0_34: pl.Tuple[pl.Tensor[[8, 384], pl.FP32], pl.Tensor[[384, 8192], pl.INT8], pl.Scalar[pl.TASK_ID]] = pl.submit(
                        self.quant,
                        act_scale_dq_inline598__iter_v1,
                        o_r_i8_pad_inline621__iter_v1,
                        t_dim_inline607__ssa_v0,
                        o_r_pad_inline620__ssa_v3,
                        col_g_inline610__ssa_v0,
                        g_inline628__idx_v0,
                        proj_b_padded_rows_inline614__ssa_v0,
                        deps=[pa_tid_inline635__ssa_v0],
                        allow_early_resolve=True,
                        attrs={"arg_directions": [pl.adir.output_existing, pl.adir.output_existing, pl.adir.scalar, pl.adir.input, pl.adir.scalar, pl.adir.scalar, pl.adir.scalar]},
                    )
                    act_scale_dq_inline598__rv_v4: pl.Tensor[[8, 384], pl.FP32, pl.MemRef("mem_ddr_181", pl.const(0, pl.INT64), 12288)] = ret__tmp_v0_34[0]
                    o_r_i8_pad_inline621__rv_v7: pl.Tensor[[384, 8192], pl.INT8, pl.MemRef("mem_ddr_182", pl.const(0, pl.INT64), 3145728)] = ret__tmp_v0_34[1]
                    q_tid_inline629__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_34[2]
                    ret__tmp_v0_35: pl.Tuple[pl.Tensor[[384, 32768], pl.INT32], pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(
                        self.proj_b_mm_spmd,
                        partials_inline627__iter_v1,
                        col_g_inline610__ssa_v0,
                        o_r_i8_pad_inline621__rv_v7,
                        wo_b__ssa_v0,
                        g_inline628__idx_v0,
                        deps=[q_tid_inline629__ssa_v0],
                        core_num=proj_b_t_rows_inline658__ssa_v0 * 8,
                        allow_early_resolve=True,
                        attrs={"arg_directions": [pl.adir.output_existing, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.scalar]},
                    )
                    partials_inline627__rv_v4: pl.Tensor[[384, 32768], pl.INT32, pl.MemRef("mem_ddr_183", pl.const(0, pl.INT64), 50331648)] = ret__tmp_v0_35[0]
                    pb_tid_inline639__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_35[1]
                    proj_b_tids_inline634__ssa_v3: pl.Array[8, pl.TASK_ID] = pl.array.update_element(proj_b_tids_inline634__iter_v1, g_inline628__idx_v0, pb_tid_inline639__ssa_v0)
                    act_scale_dq_inline598__rv_v2, o_r_i8_pad_inline621__rv_v2, o_r_pad_inline620__rv_v2, partials_inline627__rv_v2, proj_b_tids_inline634__rv_v2 = pl.yield_(
                        act_scale_dq_inline598__rv_v4, o_r_i8_pad_inline621__rv_v7, o_r_pad_inline620__ssa_v3, partials_inline627__rv_v4, proj_b_tids_inline634__ssa_v3
                    )
            _submit_deps_buf_inline588__ssa_v0: pl.Array[8, pl.TASK_ID] = pl.array.create(8, dtype=pl.TASK_ID)
            t__tmp_v287: pl.Scalar[pl.TASK_ID] = pl.array.get_element(proj_b_tids_inline634__rv_v2, 0)
            _submit_deps_buf_inline587__ssa_v0: pl.Array[8, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline588__ssa_v0, 0, t__tmp_v287)
            t__tmp_v288: pl.Scalar[pl.TASK_ID] = pl.array.get_element(proj_b_tids_inline634__rv_v2, 1)
            _submit_deps_buf_inline592__ssa_v0: pl.Array[8, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline587__ssa_v0, 1, t__tmp_v288)
            t__tmp_v289: pl.Scalar[pl.TASK_ID] = pl.array.get_element(proj_b_tids_inline634__rv_v2, 2)
            _submit_deps_buf_inline586__ssa_v0: pl.Array[8, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline592__ssa_v0, 2, t__tmp_v289)
            t__tmp_v290: pl.Scalar[pl.TASK_ID] = pl.array.get_element(proj_b_tids_inline634__rv_v2, 3)
            _submit_deps_buf_inline585__ssa_v0: pl.Array[8, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline586__ssa_v0, 3, t__tmp_v290)
            t__tmp_v291: pl.Scalar[pl.TASK_ID] = pl.array.get_element(proj_b_tids_inline634__rv_v2, 4)
            _submit_deps_buf_inline645__ssa_v0: pl.Array[8, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline585__ssa_v0, 4, t__tmp_v291)
            t__tmp_v292: pl.Scalar[pl.TASK_ID] = pl.array.get_element(proj_b_tids_inline634__rv_v2, 5)
            _submit_deps_buf_inline654__ssa_v0: pl.Array[8, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline645__ssa_v0, 5, t__tmp_v292)
            t__tmp_v293: pl.Scalar[pl.TASK_ID] = pl.array.get_element(proj_b_tids_inline634__rv_v2, 6)
            _submit_deps_buf_inline584__ssa_v0: pl.Array[8, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline654__ssa_v0, 6, t__tmp_v293)
            t__tmp_v294: pl.Scalar[pl.TASK_ID] = pl.array.get_element(proj_b_tids_inline634__rv_v2, 7)
            _submit_deps_buf_inline615__ssa_v0: pl.Array[8, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline584__ssa_v0, 7, t__tmp_v294)
            ret__tmp_v0_36: pl.Tuple[pl.Tensor[[T_DYN, 4096], pl.BF16], pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(
                self.proj_b_act_spmd,
                wo_b_scale__ssa_v0,
                attn_out__ssa_v0,
                t_dim_inline607__ssa_v0,
                partials_inline627__rv_v2,
                act_scale_dq_inline598__rv_v2,
                deps=[_submit_deps_buf_inline615__ssa_v0],
                core_num=act_t_blks_inline611__ssa_v0 * 8,
                allow_early_resolve=True,
                attrs={"arg_directions": [pl.adir.input, pl.adir.output_existing, pl.adir.scalar, pl.adir.input, pl.adir.input]},
            )
            attn_out__rv_v2: pl.Tensor[[T_DYN, 4096], pl.BF16, pl.MemRef("mem_ddr_188", pl.const(0, pl.INT64), 0)] = ret__tmp_v0_36[0]
            _act_tid_inline622__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_36[1]
        return attn_out__rv_v2