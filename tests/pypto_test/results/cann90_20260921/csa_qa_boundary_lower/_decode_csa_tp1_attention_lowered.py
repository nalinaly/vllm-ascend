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
t_dim_inline2515__ssa_v0 = pl.dynamic("t_dim_inline2515__ssa_v0")


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
            batch_idx_inline774__ssa_v0: pl.Scalar[pl.INDEX] = query__idx_v0 // 6
            position_inline782__tile: pl.Scalar[pl.INT32] = pl.tensor.read(position_ids__ssa_v0, [query__idx_v0])
            t__tile: pl.Scalar[pl.INT32] = pl.tensor.read(kv_seq_lens__ssa_v0, [batch_idx_inline774__ssa_v0])
            cache_len_inline776__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(t__tile, pl.INDEX) // 4
            cache_bound_inline777__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(cache_len_inline776__ssa_v0, (pl.cast(position_inline782__tile, pl.INDEX) + 1) // 4)
            visible_count_inline779__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(cache_bound_inline777__ssa_v0, 262144)
            if 0 < visible_count_inline779__ssa_v0:
                leaf_count_inline778__ssa_v0: pl.Scalar[pl.INDEX] = (visible_count_inline779__ssa_v0 + 8191) // 8192
                half_count_inline783__ssa_v0: pl.Scalar[pl.INDEX] = leaf_count_inline778__ssa_v0 * 2
                arena_base_inline784__ssa_v0: pl.Scalar[pl.INDEX] = query__idx_v0 * 64
                for child_inline775__idx_v0 in pl.range(1, half_count_inline783__ssa_v0):
                    left_inline2602__ssa_v0: pl.Tile[[1, 1024], pl.FP32, pl.MemRef(mem_vec_5, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.load(
                        pair_arena__ssa_v0, [arena_base_inline784__ssa_v0, 0], [1, 1024], [1, 1024], target_memory=pl.Mem.Vec
                    )
                    right_inline2601__ssa_v0: pl.Tile[[1, 1024], pl.FP32, pl.MemRef(mem_vec_6, pl.const(4096, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.load(
                        pair_arena__ssa_v0, [arena_base_inline784__ssa_v0 + child_inline775__idx_v0, 0], [1, 1024], [1, 1024], target_memory=pl.Mem.Vec
                    )
                    merge_tmp_inline2600__ssa_v0: pl.Tile[[1, 2048], pl.FP32, pl.MemRef(mem_vec_7, pl.const(8192, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.create(
                        [1, 2048], dtype=pl.FP32, target_memory=pl.Mem.Vec
                    )
                    merged_all_inline2599__ssa_v0: pl.Tile[[1, 2048], pl.FP32, pl.MemRef(mem_vec_8, pl.const(16384, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.mrgsort_format2(
                        right_inline2601__ssa_v0, left_inline2602__ssa_v0, merge_tmp_inline2600__ssa_v0, exhausted=False
                    )
                    merged_inline2598__ssa_v0: pl.Tile[[1, 1024], pl.FP32, pl.MemRef(mem_vec_8, pl.const(16384, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.slice(
                        merged_all_inline2599__ssa_v0, [1, 1024], [0, 0]
                    )
                    pl.tile.store(merged_inline2598__ssa_v0, [arena_base_inline784__ssa_v0, 0], pair_arena__ssa_v0)
                root_slot_inline785__ssa_v0: pl.Scalar[pl.INDEX] = arena_base_inline784__ssa_v0
                root_pairs_inline780__ssa_v0: pl.Tile[[1, 1024], pl.FP32, pl.MemRef(mem_vec_7, pl.const(8192, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.load(
                    pair_arena__ssa_v0, [root_slot_inline785__ssa_v0, 0], [1, 1024], [1, 1024], target_memory=pl.Mem.Vec
                )
                root_scores_inline786__ssa_v0: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_8, pl.const(16384, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.gather_mask(
                    root_pairs_inline780__ssa_v0, mask_pattern=1, output_dtype=pl.FP32
                )
                pl.tile.store(root_scores_inline786__ssa_v0, [query__idx_v0, 0], topk_scores__ssa_v0)
                root_indices_inline781__ssa_v0: pl.Tile[[1, 512], pl.INT32, pl.MemRef(mem_vec_8, pl.const(16384, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.gather_mask(
                    root_pairs_inline780__ssa_v0, mask_pattern=2, output_dtype=pl.INT32
                )
                if 512 <= visible_count_inline779__ssa_v0:
                    pl.tile.store(root_indices_inline781__ssa_v0, [query__idx_v0, 0], topk_indices__ssa_v0)
                else:
                    output_indices_inline773__ssa_v0: pl.Tile[[1, 512], pl.INT32, pl.MemRef(mem_vec_7, pl.const(8192, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.full([1, 512], dtype=pl.INT32, value=-1)
                    for lane_inline772__idx_v0 in pl.range(visible_count_inline779__ssa_v0):
                        t__tmp_v1: pl.Scalar[pl.INT32] = pl.tile.read(root_indices_inline781__ssa_v0, [0, lane_inline772__idx_v0])
                        pl.tile.write(output_indices_inline773__ssa_v0, [0, lane_inline772__idx_v0], t__tmp_v1)
                    pl.tile.store(output_indices_inline773__ssa_v0, [query__idx_v0, 0], topk_indices__ssa_v0)
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
        pair_arena_inline1088_inline2392__ssa_v1: pl.InOut[pl.Tensor[[24576, 1024], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 100663296)]],
        idx_topk_scores__ssa_v0: pl.Out[pl.Tensor[[T_DYN, 512], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)]],
        idx_topk__ssa_v0: pl.Out[pl.Tensor[[T_DYN, 512], pl.INT32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 0)]],
    ):
        pl.func_attr({"mx_tensor_views_blocked": True})
        self.indexer_topk_query_merge(
            position_ids__ssa_v0,
            kv_seq_lens__ssa_v0,
            pair_arena_inline1088_inline2392__ssa_v1,
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
                    short_indices_inline1039__ci_tmp_v0: pl.Tile[[1, 192], pl.FP32, pl.MemRef(mem_vec_36, pl.const(32768, pl.INT64), 768), pl.Mem.Vec] = pl.tile.create(
                        [1, 192], dtype=pl.FP32, target_memory=pl.Mem.Vec
                    )
                    short_indices_inline1039__ssa_v0: pl.Tile[[1, 2048], pl.INT32, pl.MemRef(mem_vec_37, pl.const(98304, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.ci(
                        pl.const(0, pl.INT32), [1, 2048], tmp=short_indices_inline1039__ci_tmp_v0, dtype=pl.INT32, descending=False
                    )
                    short_raw_inline1027__ssa_v0: pl.Tile[
                        [1, 2048], pl.FP32, pl.MemRef(mem_vec_36, pl.const(32768, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[1, visible_count__ssa_v0])
                    ] = pl.tile.load(score_arena__ssa_v0, [query__idx_v0, 0], [1, 2048], [1, visible_count__ssa_v0], target_memory=pl.Mem.Vec)
                    short_scores_inline1036__ssa_v0: pl.Tile[[1, 2048], pl.FP32, pl.MemRef(mem_vec_36, pl.const(32768, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = (
                        pl.tile.fillpad(short_raw_inline1027__ssa_v0, pad_value=pl.PadValue.min)
                    )
                    short_scores_v1_inline1042__ssa_v0: pl.Tile[[1, 2048], pl.FP32, pl.MemRef(mem_vec_34, pl.const(0, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = (
                        pl.tile.maximums(short_scores_inline1036__ssa_v0, -3.4028234663852886e38)
                    )
                    t__tmp_v1: pl.Tile[[1, 2048], pl.UINT32, pl.MemRef(mem_vec_37, pl.const(98304, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.reinterpret_view(
                        short_indices_inline1039__ssa_v0, dtype=pl.UINT32
                    )
                    short_pairs_inline1034__ssa_v0: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_36, pl.const(32768, pl.INT64), 16384), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.sort32(
                        short_scores_v1_inline1042__ssa_v0, t__tmp_v1
                    )
                    short_pairs_v1_inline1031__ssa_v0: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_37, pl.const(98304, pl.INT64), 16384), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = (
                        pl.tile.mrgsort_format1(short_pairs_inline1034__ssa_v0, pl.const(64, pl.INT32))
                    )
                    short_pairs_v2_inline1052__ssa_v0: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_36, pl.const(32768, pl.INT64), 16384), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = (
                        pl.tile.mrgsort_format1(short_pairs_v1_inline1031__ssa_v0, pl.const(256, pl.INT32))
                    )
                    short_pairs_v3_inline1033__ssa_v0: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_37, pl.const(98304, pl.INT64), 16384), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = (
                        pl.tile.mrgsort_format1(short_pairs_v2_inline1052__ssa_v0, pl.const(1024, pl.INT32))
                    )
                    short_top_inline1035__ssa_v0: pl.Tile[[1, 1024], pl.FP32, pl.MemRef(mem_vec_37, pl.const(98304, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(
                        short_pairs_v3_inline1033__ssa_v0, [1, 1024], [0, 0]
                    )
                    short_values_inline1026__ssa_v0: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_36, pl.const(32768, pl.INT64), 2048), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = (
                        pl.tile.gather_mask(short_top_inline1035__ssa_v0, mask_pattern=1, output_dtype=pl.FP32)
                    )
                    short_selected_inline1032__ssa_v0: pl.Tile[[1, 512], pl.INT32, pl.MemRef(mem_vec_34, pl.const(0, pl.INT64), 2048), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = (
                        pl.tile.gather_mask(short_top_inline1035__ssa_v0, mask_pattern=2, output_dtype=pl.INT32)
                    )
                    pl.tile.store(short_values_inline1026__ssa_v0, [query__idx_v0, 0], topk_scores__ssa_v0)
                    if 512 <= visible_count__ssa_v0:
                        pl.tile.store(short_selected_inline1032__ssa_v0, [query__idx_v0, 0], topk_indices__ssa_v0)
                    else:
                        short_output_inline1038__ssa_v0: pl.Tile[[1, 512], pl.INT32, pl.MemRef(mem_vec_36, pl.const(32768, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.full(
                            [1, 512], dtype=pl.INT32, value=-1
                        )
                        for lane_inline1037__idx_v0 in pl.range(visible_count__ssa_v0):
                            t__tmp_v2: pl.Scalar[pl.INT32] = pl.tile.read(short_selected_inline1032__ssa_v0, [0, lane_inline1037__idx_v0])
                            pl.tile.write(short_output_inline1038__ssa_v0, [0, lane_inline1037__idx_v0], t__tmp_v2)
                        pl.tile.store(short_output_inline1038__ssa_v0, [query__idx_v0, 0], topk_indices__ssa_v0)
                else:
                    if visible_count__ssa_v0 <= 4096:
                        medium_indices_inline1041__ci_tmp_v1: pl.Tile[[1, 192], pl.FP32, pl.MemRef(mem_vec_36, pl.const(32768, pl.INT64), 768), pl.Mem.Vec] = pl.tile.create(
                            [1, 192], dtype=pl.FP32, target_memory=pl.Mem.Vec
                        )
                        medium_indices_inline1041__ssa_v0: pl.Tile[[1, 4096], pl.INT32, pl.MemRef(mem_vec_37, pl.const(98304, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.ci(
                            pl.const(0, pl.INT32), [1, 4096], tmp=medium_indices_inline1041__ci_tmp_v1, dtype=pl.INT32, descending=False
                        )
                        medium_raw_inline1043__ssa_v0: pl.Tile[
                            [1, 4096], pl.FP32, pl.MemRef(mem_vec_36, pl.const(32768, pl.INT64), 16384), pl.Mem.Vec, pl.TileView(valid_shape=[1, visible_count__ssa_v0])
                        ] = pl.tile.load(score_arena__ssa_v0, [query__idx_v0, 0], [1, 4096], [1, visible_count__ssa_v0], target_memory=pl.Mem.Vec)
                        medium_scores_inline1044__ssa_v0: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_36, pl.const(32768, pl.INT64), 16384), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = (
                            pl.tile.fillpad(medium_raw_inline1043__ssa_v0, pad_value=pl.PadValue.min)
                        )
                        medium_scores_v1_inline1048__ssa_v0: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_34, pl.const(0, pl.INT64), 16384), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = (
                            pl.tile.maximums(medium_scores_inline1044__ssa_v0, -3.4028234663852886e38)
                        )
                        t__tmp_v3: pl.Tile[[1, 4096], pl.UINT32, pl.MemRef(mem_vec_37, pl.const(98304, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.reinterpret_view(
                            medium_indices_inline1041__ssa_v0, dtype=pl.UINT32
                        )
                        medium_pairs_inline1045__ssa_v0: pl.Tile[[1, 8192], pl.FP32, pl.MemRef(mem_vec_36, pl.const(32768, pl.INT64), 32768), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = (
                            pl.tile.sort32(medium_scores_v1_inline1048__ssa_v0, t__tmp_v3)
                        )
                        medium_pairs_v1_inline1055__ssa_v0: pl.Tile[[1, 8192], pl.FP32, pl.MemRef(mem_vec_37, pl.const(98304, pl.INT64), 32768), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = (
                            pl.tile.mrgsort_format1(medium_pairs_inline1045__ssa_v0, pl.const(64, pl.INT32))
                        )
                        medium_pairs_v2_inline1046__ssa_v0: pl.Tile[[1, 8192], pl.FP32, pl.MemRef(mem_vec_36, pl.const(32768, pl.INT64), 32768), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = (
                            pl.tile.mrgsort_format1(medium_pairs_v1_inline1055__ssa_v0, pl.const(256, pl.INT32))
                        )
                        medium_pairs_v3_inline1047__ssa_v0: pl.Tile[[1, 8192], pl.FP32, pl.MemRef(mem_vec_37, pl.const(98304, pl.INT64), 32768), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = (
                            pl.tile.mrgsort_format1(medium_pairs_v2_inline1046__ssa_v0, pl.const(1024, pl.INT32))
                        )
                        medium_left_inline1053__ssa_v0: pl.Tile[[1, 1024], pl.FP32, pl.MemRef(mem_vec_37, pl.const(98304, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = (
                            pl.tile.slice(medium_pairs_v3_inline1047__ssa_v0, [1, 1024], [0, 0])
                        )
                        medium_right_inline1054__ssa_v0: pl.Tile[[1, 1024], pl.FP32, pl.MemRef(mem_vec_37, pl.const(114688, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = (
                            pl.tile.slice(medium_pairs_v3_inline1047__ssa_v0, [1, 1024], [0, 4096])
                        )
                        medium_tmp_inline1056__ssa_v0: pl.Tile[[1, 2048], pl.FP32, pl.MemRef(mem_vec_36, pl.const(32768, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.create(
                            [1, 2048], dtype=pl.FP32, target_memory=pl.Mem.Vec
                        )
                        medium_merged_inline1057__ssa_v0: pl.Tile[[1, 2048], pl.FP32, pl.MemRef(mem_vec_34, pl.const(0, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = (
                            pl.tile.mrgsort_format2(medium_left_inline1053__ssa_v0, medium_right_inline1054__ssa_v0, medium_tmp_inline1056__ssa_v0, exhausted=False)
                        )
                        medium_top_inline1028__ssa_v0: pl.Tile[[1, 1024], pl.FP32, pl.MemRef(mem_vec_34, pl.const(0, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(
                            medium_merged_inline1057__ssa_v0, [1, 1024], [0, 0]
                        )
                        medium_values_inline1040__ssa_v0: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_36, pl.const(32768, pl.INT64), 2048), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = (
                            pl.tile.gather_mask(medium_top_inline1028__ssa_v0, mask_pattern=1, output_dtype=pl.FP32)
                        )
                        medium_selected_inline1049__ssa_v0: pl.Tile[[1, 512], pl.INT32, pl.MemRef(mem_vec_37, pl.const(98304, pl.INT64), 2048), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = (
                            pl.tile.gather_mask(medium_top_inline1028__ssa_v0, mask_pattern=2, output_dtype=pl.INT32)
                        )
                        pl.tile.store(medium_values_inline1040__ssa_v0, [query__idx_v0, 0], topk_scores__ssa_v0)
                        pl.tile.store(medium_selected_inline1049__ssa_v0, [query__idx_v0, 0], topk_indices__ssa_v0)
                    else:
                        full_indices_inline1030__ci_tmp_v2: pl.Tile[[1, 192], pl.FP32, pl.MemRef(mem_vec_36, pl.const(32768, pl.INT64), 768), pl.Mem.Vec] = pl.tile.create(
                            [1, 192], dtype=pl.FP32, target_memory=pl.Mem.Vec
                        )
                        full_indices_inline1030__ssa_v0: pl.Tile[[1, 8192], pl.INT32, pl.MemRef(mem_vec_37, pl.const(98304, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.ci(
                            pl.const(0, pl.INT32), [1, 8192], tmp=full_indices_inline1030__ci_tmp_v2, dtype=pl.INT32, descending=False
                        )
                        full_raw_inline1025__ssa_v0: pl.Tile[
                            [1, 8192], pl.FP32, pl.MemRef(mem_vec_36, pl.const(32768, pl.INT64), 32768), pl.Mem.Vec, pl.TileView(valid_shape=[1, visible_count__ssa_v0])
                        ] = pl.tile.load(score_arena__ssa_v0, [query__idx_v0, 0], [1, 8192], [1, visible_count__ssa_v0], target_memory=pl.Mem.Vec)
                        full_scores_inline1051__ssa_v0: pl.Tile[[1, 8192], pl.FP32, pl.MemRef(mem_vec_36, pl.const(32768, pl.INT64), 32768), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = (
                            pl.tile.fillpad(full_raw_inline1025__ssa_v0, pad_value=pl.PadValue.min)
                        )
                        full_min_inline1024__ssa_v0: pl.Tile[[1, 8192], pl.FP32, pl.MemRef(mem_vec_34, pl.const(0, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.full(
                            [1, 8192], dtype=pl.FP32, value=-3.4028234663852886e38
                        )
                        full_scores_v1_inline1023__ssa_v0: pl.Tile[[1, 8192], pl.FP32, pl.MemRef(mem_vec_34, pl.const(0, pl.INT64), 32768), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = (
                            pl.tile.maximum(full_scores_inline1051__ssa_v0, full_min_inline1024__ssa_v0)
                        )
                        t__tmp_v4: pl.Tile[[1, 8192], pl.UINT32, pl.MemRef(mem_vec_37, pl.const(98304, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.reinterpret_view(
                            full_indices_inline1030__ssa_v0, dtype=pl.UINT32
                        )
                        full_pairs_inline1050__ssa_v0: pl.Tile[[1, 16384], pl.FP32, pl.MemRef(mem_vec_36, pl.const(32768, pl.INT64), 65536), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = (
                            pl.tile.sort32(full_scores_v1_inline1023__ssa_v0, t__tmp_v4)
                        )
                        full_pairs_v1_inline1029__ssa_v0: pl.Tile[[1, 16384], pl.FP32, pl.MemRef(mem_vec_37, pl.const(98304, pl.INT64), 65536), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = (
                            pl.tile.mrgsort_format1(full_pairs_inline1050__ssa_v0, pl.const(64, pl.INT32))
                        )
                        full_pairs_v2_inline1021__ssa_v0: pl.Tile[[1, 16384], pl.FP32, pl.MemRef(mem_vec_36, pl.const(32768, pl.INT64), 65536), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = (
                            pl.tile.mrgsort_format1(full_pairs_v1_inline1029__ssa_v0, pl.const(256, pl.INT32))
                        )
                        full_pairs_v3_inline1020__ssa_v0: pl.Tile[[1, 16384], pl.FP32, pl.MemRef(mem_vec_37, pl.const(98304, pl.INT64), 65536), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = (
                            pl.tile.mrgsort_format1(full_pairs_v2_inline1021__ssa_v0, pl.const(1024, pl.INT32))
                        )
                        full_pairs_v4_inline1019__ssa_v0: pl.Tile[[1, 16384], pl.FP32, pl.MemRef(mem_vec_36, pl.const(32768, pl.INT64), 65536), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = (
                            pl.tile.mrgsort_format1(full_pairs_v3_inline1020__ssa_v0, pl.const(4096, pl.INT32))
                        )
                        full_top_inline1022__ssa_v0: pl.Tile[[1, 1024], pl.FP32, pl.MemRef(mem_vec_36, pl.const(32768, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(
                            full_pairs_v4_inline1019__ssa_v0, [1, 1024], [0, 0]
                        )
                        full_values_inline1018__ssa_v0: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_37, pl.const(98304, pl.INT64), 2048), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = (
                            pl.tile.gather_mask(full_top_inline1022__ssa_v0, mask_pattern=1, output_dtype=pl.FP32)
                        )
                        full_selected_inline1017__ssa_v0: pl.Tile[[1, 512], pl.INT32, pl.MemRef(mem_vec_34, pl.const(0, pl.INT64), 2048), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = (
                            pl.tile.gather_mask(full_top_inline1022__ssa_v0, mask_pattern=2, output_dtype=pl.INT32)
                        )
                        pl.tile.store(full_values_inline1018__ssa_v0, [query__idx_v0, 0], topk_scores__ssa_v0)
                        pl.tile.store(full_selected_inline1017__ssa_v0, [query__idx_v0, 0], topk_indices__ssa_v0)
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
        score_arena_inline1090_inline2379__rv_v2: pl.Tensor[[384, 8192], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 12582912)],
        idx_topk_scores__ssa_v0: pl.Out[pl.Tensor[[T_DYN, 512], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)]],
        idx_topk__ssa_v0: pl.Out[pl.Tensor[[T_DYN, 512], pl.INT32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 0)]],
    ):
        pl.func_attr({"mx_tensor_views_blocked": True})
        self.indexer_topk_single_leaf_publish(
            position_ids__ssa_v0,
            kv_seq_lens__ssa_v0,
            score_arena_inline1090_inline2379__rv_v2,
            idx_topk_scores__ssa_v0,
            idx_topk__ssa_v0,
            attrs={"arg_directions": [pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.output_existing, pl.adir.output_existing]},
        )

    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def compress_state_commit(
        compress_state_flat_inline2090__ssa_v0: pl.Out[pl.Tensor[[compress_state_rows_inline2065__ssa_v0, 2048], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)]],
        b_dim_inline2092__ssa_v0: pl.Scalar[pl.INDEX],
        commit_workers_inline2086__ssa_v0: pl.Scalar[pl.INDEX],
        s_dim_inline2071__ssa_v0: pl.Scalar[pl.INDEX],
        state_slot_mapping__ssa_v0: pl.Tensor[[T_DYN], pl.INT64, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)],
        position_ids__ssa_v0: pl.Tensor[[T_DYN], pl.INT32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)],
        cmp4_kv_proj_pad_inline2070__ssa_v0: pl.Tensor[[384, 1024], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 1572864)],
        cmp4_score_proj_pad_inline2087__ssa_v0: pl.Tensor[[384, 1024], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 1572864)],
        cmp_ape__ssa_v0: pl.Tensor[[4, 1024], pl.FP32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 16384)],
    ):
        pl.func_attr({"mx_tensor_views_blocked": True})
        mem_vec_6: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 4096)
        mem_vec_8: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 4096)
        commit_worker_inline2076__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        for c_idx_inline2093__idx_v0, (compress_state_flat_inline2090__iter_v1,) in pl.range(
            commit_worker_inline2076__ssa_v0, b_dim_inline2092__ssa_v0, commit_workers_inline2086__ssa_v0, init_values=(compress_state_flat_inline2090__ssa_v0,)
        ):
            for s_idx_inline2081__idx_v0, (compress_state_flat_inline2090__iter_v3,) in pl.range(s_dim_inline2071__ssa_v0, init_values=(compress_state_flat_inline2090__iter_v1,)):
                token_inline2094__ssa_v0: pl.Scalar[pl.INDEX] = c_idx_inline2093__idx_v0 * s_dim_inline2071__ssa_v0 + s_idx_inline2081__idx_v0
                state_row_i64_inline2096__tile: pl.Scalar[pl.INT64] = pl.tensor.read(state_slot_mapping__ssa_v0, [token_inline2094__ssa_v0])
                if 0 <= state_row_i64_inline2096__tile:
                    state_row_inline2098__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(state_row_i64_inline2096__tile, pl.INDEX)
                    token_pos_inline2077__tile: pl.Scalar[pl.INT32] = pl.tensor.read(position_ids__ssa_v0, [token_inline2094__ssa_v0])
                    ape_row_inline2068__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(pl.cast(token_pos_inline2077__tile, pl.INDEX) % 4, pl.INDEX)
                    t__tile: pl.Tile[[1, 1024], pl.FP32, pl.MemRef(mem_vec_6, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.load(
                        cmp4_kv_proj_pad_inline2070__ssa_v0, [token_inline2094__ssa_v0, 0], [1, 1024], [1, 1024], target_memory=pl.Mem.Vec
                    )
                    compress_state_flat_inline2090__tile: pl.Tensor[[compress_state_rows_inline2065__ssa_v0, 2048], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                        t__tile, [state_row_inline2098__ssa_v0, 0], compress_state_flat_inline2090__iter_v3
                    )
                    t__tile_1: pl.Tile[[1, 1024], pl.FP32, pl.MemRef(mem_vec_6, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.load(
                        cmp4_score_proj_pad_inline2087__ssa_v0, [token_inline2094__ssa_v0, 0], [1, 1024], [1, 1024], target_memory=pl.Mem.Vec
                    )
                    t__tile_2: pl.Tile[[1, 1024], pl.FP32, pl.MemRef(mem_vec_8, pl.const(4096, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.load(
                        cmp_ape__ssa_v0, [ape_row_inline2068__ssa_v0, 0], [1, 1024], [1, 1024], target_memory=pl.Mem.Vec
                    )
                    t__tile_3: pl.Tile[[1, 1024], pl.FP32, pl.MemRef(mem_vec_6, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.add(t__tile_1, t__tile_2)
                    compress_state_flat_inline2090__tile_1: pl.Tensor[[compress_state_rows_inline2065__ssa_v0, 2048], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                        t__tile_3, [state_row_inline2098__ssa_v0, 1024], compress_state_flat_inline2090__tile
                    )
                    compress_state_flat_inline2090__phi_v7: pl.Tensor[[compress_state_rows_inline2065__ssa_v0, 2048], pl.FP32, pl.MemRef("mem_ddr_10", pl.const(0, pl.INT64), 0)] = pl.yield_(
                        compress_state_flat_inline2090__tile_1
                    )
                else:
                    compress_state_flat_inline2090__phi_v7: pl.Tensor[[compress_state_rows_inline2065__ssa_v0, 2048], pl.FP32, pl.MemRef("mem_ddr_10", pl.const(0, pl.INT64), 0)] = pl.yield_(
                        compress_state_flat_inline2090__iter_v3
                    )
                compress_state_flat_inline2090__rv_v4: pl.Tensor[[compress_state_rows_inline2065__ssa_v0, 2048], pl.FP32, pl.MemRef("mem_ddr_11", pl.const(0, pl.INT64), 0)] = pl.yield_(
                    compress_state_flat_inline2090__phi_v7
                )
            compress_state_flat_inline2090__rv_v2: pl.Tensor[[compress_state_rows_inline2065__ssa_v0, 2048], pl.FP32, pl.MemRef("mem_ddr_12", pl.const(0, pl.INT64), 0)] = pl.yield_(
                compress_state_flat_inline2090__rv_v4
            )
        return

    @pl.function(type=pl.FunctionType.Spmd)
    def compress_state_commit_spmd(
        self,
        compress_state_flat_inline2090__ssa_v0: pl.Out[pl.Tensor[[compress_state_rows_inline2065__ssa_v0, 2048], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)]],
        b_dim_inline2092__ssa_v0: pl.Scalar[pl.INDEX],
        commit_workers_inline2086__ssa_v0: pl.Scalar[pl.INDEX],
        s_dim_inline2071__ssa_v0: pl.Scalar[pl.INDEX],
        state_slot_mapping__ssa_v0: pl.Tensor[[T_DYN], pl.INT64, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)],
        position_ids__ssa_v0: pl.Tensor[[T_DYN], pl.INT32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)],
        cmp4_kv_proj_pad_inline2070__ssa_v0: pl.Tensor[[384, 1024], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 1572864)],
        cmp4_score_proj_pad_inline2087__ssa_v0: pl.Tensor[[384, 1024], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 1572864)],
        cmp_ape__ssa_v0: pl.Tensor[[4, 1024], pl.FP32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 16384)],
    ):
        pl.func_attr({"mx_tensor_views_blocked": True})
        self.compress_state_commit(
            compress_state_flat_inline2090__ssa_v0,
            b_dim_inline2092__ssa_v0,
            commit_workers_inline2086__ssa_v0,
            s_dim_inline2071__ssa_v0,
            state_slot_mapping__ssa_v0,
            position_ids__ssa_v0,
            cmp4_kv_proj_pad_inline2070__ssa_v0,
            cmp4_score_proj_pad_inline2087__ssa_v0,
            cmp_ape__ssa_v0,
            attrs={"arg_directions": [pl.adir.output_existing, pl.adir.scalar, pl.adir.scalar, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input]},
        )

    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def compress_state_commit_0(
        compress_state_flat_inline474_inline2141__ssa_v0: pl.Out[pl.Tensor[[compress_state_rows_inline470_inline2190__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)]],
        b_dim_inline459_inline2149__ssa_v0: pl.Scalar[pl.INDEX],
        commit_workers_inline493_inline2210__ssa_v0: pl.Scalar[pl.INDEX],
        s_dim_inline464_inline2145__ssa_v0: pl.Scalar[pl.INDEX],
        inner_state_slot_mapping__ssa_v0: pl.Tensor[[T_DYN], pl.INT64, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)],
        position_ids__ssa_v0: pl.Tensor[[T_DYN], pl.INT32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)],
        kv_proj_pad_inline2152__rv_v2: pl.Tensor[[384, 256], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 393216)],
        score_proj_pad_inline2156__rv_v2: pl.Tensor[[384, 256], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 393216)],
        inner_ape__ssa_v0: pl.Tensor[[4, 256], pl.FP32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 4096)],
    ):
        pl.func_attr({"mx_tensor_views_blocked": True})
        mem_vec_6: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 1024)
        mem_vec_8: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 1024)
        commit_worker_inline500_inline2137__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        for c_idx_inline494_inline2136__idx_v0, (compress_state_flat_inline474_inline2141__iter_v1,) in pl.range(
            commit_worker_inline500_inline2137__ssa_v0, b_dim_inline459_inline2149__ssa_v0, commit_workers_inline493_inline2210__ssa_v0, init_values=(compress_state_flat_inline474_inline2141__ssa_v0,)
        ):
            for s_idx_inline495_inline2134__idx_v0, (compress_state_flat_inline474_inline2141__iter_v3,) in pl.range(
                s_dim_inline464_inline2145__ssa_v0, init_values=(compress_state_flat_inline474_inline2141__iter_v1,)
            ):
                token_inline454_inline2186__ssa_v1: pl.Scalar[pl.INDEX] = c_idx_inline494_inline2136__idx_v0 * s_dim_inline464_inline2145__ssa_v0 + s_idx_inline495_inline2134__idx_v0
                state_row_i64_inline496_inline2132__tile: pl.Scalar[pl.INT64] = pl.tensor.read(inner_state_slot_mapping__ssa_v0, [token_inline454_inline2186__ssa_v1])
                if 0 <= state_row_i64_inline496_inline2132__tile:
                    state_row_inline481_inline2188__ssa_v1: pl.Scalar[pl.INDEX] = pl.cast(state_row_i64_inline496_inline2132__tile, pl.INDEX)
                    token_pos_inline478_inline2184__tile: pl.Scalar[pl.INT32] = pl.tensor.read(position_ids__ssa_v0, [token_inline454_inline2186__ssa_v1])
                    ape_row_inline491_inline2206__ssa_v1: pl.Scalar[pl.INDEX] = pl.cast(pl.cast(token_pos_inline478_inline2184__tile, pl.INDEX) % 4, pl.INDEX)
                    t__tile: pl.Tile[[1, 256], pl.FP32, pl.MemRef(mem_vec_6, pl.const(0, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.load(
                        kv_proj_pad_inline2152__rv_v2, [token_inline454_inline2186__ssa_v1, 0], [1, 256], [1, 256], target_memory=pl.Mem.Vec
                    )
                    compress_state_flat_inline474_inline2141__tile: pl.Tensor[[compress_state_rows_inline470_inline2190__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)] = (
                        pl.tile.store(t__tile, [state_row_inline481_inline2188__ssa_v1, 0], compress_state_flat_inline474_inline2141__iter_v3)
                    )
                    t__tile_1: pl.Tile[[1, 256], pl.FP32, pl.MemRef(mem_vec_6, pl.const(0, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.load(
                        score_proj_pad_inline2156__rv_v2, [token_inline454_inline2186__ssa_v1, 0], [1, 256], [1, 256], target_memory=pl.Mem.Vec
                    )
                    t__tile_2: pl.Tile[[1, 256], pl.FP32, pl.MemRef(mem_vec_8, pl.const(1024, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.load(
                        inner_ape__ssa_v0, [ape_row_inline491_inline2206__ssa_v1, 0], [1, 256], [1, 256], target_memory=pl.Mem.Vec
                    )
                    t__tile_3: pl.Tile[[1, 256], pl.FP32, pl.MemRef(mem_vec_6, pl.const(0, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.add(t__tile_1, t__tile_2)
                    compress_state_flat_inline474_inline2141__tile_1: pl.Tensor[[compress_state_rows_inline470_inline2190__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)] = (
                        pl.tile.store(t__tile_3, [state_row_inline481_inline2188__ssa_v1, 256], compress_state_flat_inline474_inline2141__tile)
                    )
                    compress_state_flat_inline474_inline2141__phi_v7: pl.Tensor[[compress_state_rows_inline470_inline2190__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_10", pl.const(0, pl.INT64), 0)] = (
                        pl.yield_(compress_state_flat_inline474_inline2141__tile_1)
                    )
                else:
                    compress_state_flat_inline474_inline2141__phi_v7: pl.Tensor[[compress_state_rows_inline470_inline2190__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_10", pl.const(0, pl.INT64), 0)] = (
                        pl.yield_(compress_state_flat_inline474_inline2141__iter_v3)
                    )
                compress_state_flat_inline474_inline2141__rv_v4: pl.Tensor[[compress_state_rows_inline470_inline2190__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_11", pl.const(0, pl.INT64), 0)] = (
                    pl.yield_(compress_state_flat_inline474_inline2141__phi_v7)
                )
            compress_state_flat_inline474_inline2141__rv_v2: pl.Tensor[[compress_state_rows_inline470_inline2190__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_12", pl.const(0, pl.INT64), 0)] = pl.yield_(
                compress_state_flat_inline474_inline2141__rv_v4
            )
        return

    @pl.function(type=pl.FunctionType.Spmd)
    def compress_state_commit_spmd_0(
        self,
        compress_state_flat_inline474_inline2141__ssa_v0: pl.Out[pl.Tensor[[compress_state_rows_inline470_inline2190__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)]],
        b_dim_inline459_inline2149__ssa_v0: pl.Scalar[pl.INDEX],
        commit_workers_inline493_inline2210__ssa_v0: pl.Scalar[pl.INDEX],
        s_dim_inline464_inline2145__ssa_v0: pl.Scalar[pl.INDEX],
        inner_state_slot_mapping__ssa_v0: pl.Tensor[[T_DYN], pl.INT64, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)],
        position_ids__ssa_v0: pl.Tensor[[T_DYN], pl.INT32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)],
        kv_proj_pad_inline2152__rv_v2: pl.Tensor[[384, 256], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 393216)],
        score_proj_pad_inline2156__rv_v2: pl.Tensor[[384, 256], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 393216)],
        inner_ape__ssa_v0: pl.Tensor[[4, 256], pl.FP32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 4096)],
    ):
        pl.func_attr({"mx_tensor_views_blocked": True})
        self.compress_state_commit_0(
            compress_state_flat_inline474_inline2141__ssa_v0,
            b_dim_inline459_inline2149__ssa_v0,
            commit_workers_inline493_inline2210__ssa_v0,
            s_dim_inline464_inline2145__ssa_v0,
            inner_state_slot_mapping__ssa_v0,
            position_ids__ssa_v0,
            kv_proj_pad_inline2152__rv_v2,
            score_proj_pad_inline2156__rv_v2,
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
        cmp_sparse_indices_inline2527__ssa_v0: pl.Out[pl.Tensor[[t_dim_inline2515__ssa_v0, 512], pl.INT32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)]],
        sparse_bias_inline2513__ssa_v0: pl.Out[pl.Tensor[[t_dim_inline2515__ssa_v0, 640], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)]],
        t_dim_inline2515__ssa_v0: pl.Scalar[pl.INDEX],
        idx_topk__ssa_v3: pl.Tensor[[T_DYN, 512], pl.INT32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)],
        position_ids_t1__ssa_v0: pl.Tensor[[T_DYN, 1], pl.INT32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)],
        window_swa_indices__ssa_v0: pl.Tensor[[T_DYN, 128], pl.INT32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 0)],
        valid_block_mask_inline2517__ssa_v0: pl.Out[pl.Tensor[[t_dim_inline2515__ssa_v0, 16], pl.INT32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 0)]],
    ) -> tuple[pl.Tensor[[t_dim_inline2515__ssa_v0, 512], pl.INT32], pl.Tensor[[t_dim_inline2515__ssa_v0, 640], pl.FP32]]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        mem_vec_6: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
        mem_vec_14: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
        mem_vec_16: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
        mem_vec_32: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 4096)
        mem_vec_33: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32)
        plan_worker_inline2507__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        for bias_t0_inline2494__idx_v0, (cmp_sparse_indices_inline2527__iter_v1, sparse_bias_inline2513__iter_v1) in pl.range(
            plan_worker_inline2507__ssa_v0 * 8, t_dim_inline2515__ssa_v0, 128, init_values=(cmp_sparse_indices_inline2527__ssa_v0, sparse_bias_inline2513__ssa_v0)
        ):
            t__tile: pl.Tile[[8, 512], pl.INT32, pl.MemRef(mem_vec_6, pl.const(36896, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.load(
                idx_topk__ssa_v3, [bias_t0_inline2494__idx_v0, 0], [8, 512], [8, 512], target_memory=pl.Mem.Vec
            )
            c_raw_inline2578__tile: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_6, pl.const(36896, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.cast(t__tile, target_type=pl.FP32, mode="round")
            t__tile_1: pl.Tile[[8, 1], pl.INT32, pl.MemRef(mem_vec_14, pl.const(0, pl.INT64), 32), pl.Mem.Vec] = pl.tile.load(
                position_ids_t1__ssa_v0, [bias_t0_inline2494__idx_v0, 0], [8, 1], [8, 1], target_memory=pl.Mem.Vec
            )
            c_pos_inline2500__rm_a0_tmp_v0: pl.Tile[[1, 8], pl.INT32, pl.MemRef(mem_vec_14, pl.const(0, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(t__tile_1, [1, 8])
            c_pos_inline2500__row_major_tmp_v1: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_14, pl.const(0, pl.INT64), 32), pl.Mem.Vec] = pl.tile.cast(
                c_pos_inline2500__rm_a0_tmp_v0, target_type=pl.FP32, mode="round"
            )
            c_pos_inline2500__tile: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_14, pl.const(0, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(c_pos_inline2500__row_major_tmp_v1, [8, 1])
            t__rm_a0_tmp_v2: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_14, pl.const(0, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(c_pos_inline2500__tile, [1, 8])
            t__row_major_tmp_v3: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_14, pl.const(0, pl.INT64), 32), pl.Mem.Vec] = pl.tile.adds(t__rm_a0_tmp_v2, 1.0)
            t__tile_2: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_14, pl.const(0, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(t__row_major_tmp_v3, [8, 1])
            c_pos_scaled_inline2529__rm_a0_tmp_v4: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_14, pl.const(0, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(t__tile_2, [1, 8])
            c_pos_scaled_inline2529__row_major_tmp_v5: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_14, pl.const(0, pl.INT64), 32), pl.Mem.Vec] = pl.tile.muls(
                c_pos_scaled_inline2529__rm_a0_tmp_v4, 0.25
            )
            c_pos_scaled_inline2529__tile: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_14, pl.const(0, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(c_pos_scaled_inline2529__row_major_tmp_v5, [8, 1])
            c_pos_i32_inline2523__rm_a0_tmp_v6: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_14, pl.const(0, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(c_pos_scaled_inline2529__tile, [1, 8])
            c_pos_i32_inline2523__row_major_tmp_v7: pl.Tile[[1, 8], pl.INT32, pl.MemRef(mem_vec_14, pl.const(0, pl.INT64), 32), pl.Mem.Vec] = pl.tile.cast(
                c_pos_i32_inline2523__rm_a0_tmp_v6, target_type=pl.INT32, mode="trunc"
            )
            c_pos_i32_inline2523__tile: pl.Tile[[8, 1], pl.INT32, pl.MemRef(mem_vec_14, pl.const(0, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(c_pos_i32_inline2523__row_major_tmp_v7, [8, 1])
            c_pos_q_inline2524__rm_a0_tmp_v8: pl.Tile[[1, 8], pl.INT32, pl.MemRef(mem_vec_14, pl.const(0, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(c_pos_i32_inline2523__tile, [1, 8])
            c_pos_q_inline2524__row_major_tmp_v9: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_16, pl.const(16384, pl.INT64), 32), pl.Mem.Vec] = pl.tile.cast(
                c_pos_q_inline2524__rm_a0_tmp_v8, target_type=pl.FP32, mode="round"
            )
            c_pos_q_inline2524__tile: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_16, pl.const(16384, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(c_pos_q_inline2524__row_major_tmp_v9, [8, 1])
            t__tile_3: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_14, pl.const(0, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.full([8, 512], dtype=pl.FP32, value=1.0)
            c_upper_b_inline2528__tile: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_14, pl.const(0, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.row_expand_mul(t__tile_3, c_pos_q_inline2524__tile)
            t__tile_4: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_16, pl.const(16384, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.adds(c_raw_inline2578__tile, 1.0)
            t__tile_5: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_16, pl.const(16384, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.maximums(t__tile_4, 0.0)
            c_ge_inline2553__tile: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_16, pl.const(16384, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.minimums(t__tile_5, 1.0)
            t__tile_6: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_14, pl.const(0, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.sub(c_upper_b_inline2528__tile, c_raw_inline2578__tile)
            t__tile_7: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_14, pl.const(0, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.maximums(t__tile_6, 0.0)
            c_lt_inline2511__tile: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_14, pl.const(0, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.minimums(t__tile_7, 1.0)
            c_mask_inline2481__tile: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_14, pl.const(0, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.mul(c_ge_inline2553__tile, c_lt_inline2511__tile)
            t__tile_8: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_6, pl.const(36896, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.adds(c_raw_inline2578__tile, 1.0)
            t__tile_9: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_6, pl.const(36896, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.mul(c_mask_inline2481__tile, t__tile_8)
            c_out_inline2544__tile: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_6, pl.const(36896, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.subs(t__tile_9, 1.0)
            t__tile_10: pl.Tile[[8, 512], pl.INT32, pl.MemRef(mem_vec_16, pl.const(16384, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.cast(c_out_inline2544__tile, target_type=pl.INT32, mode="round")
            cmp_sparse_indices_inline2527__tile: pl.Tensor[[t_dim_inline2515__ssa_v0, 512], pl.INT32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                t__tile_10, [bias_t0_inline2494__idx_v0, 0], cmp_sparse_indices_inline2527__iter_v1
            )
            t__tile_11: pl.Tile[[8, 128], pl.INT32, pl.MemRef(mem_vec_16, pl.const(16384, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.load(
                window_swa_indices__ssa_v0, [bias_t0_inline2494__idx_v0, 0], [8, 128], [8, 128], target_memory=pl.Mem.Vec
            )
            v_win_f_inline2537__tile: pl.Tile[[8, 128], pl.FP32, pl.MemRef(mem_vec_16, pl.const(16384, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(t__tile_11, target_type=pl.FP32, mode="round")
            t__tile_12: pl.Tile[[8, 128], pl.FP32, pl.MemRef(mem_vec_16, pl.const(16384, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.adds(v_win_f_inline2537__tile, 1.0)
            t__tile_13: pl.Tile[[8, 128], pl.FP32, pl.MemRef(mem_vec_16, pl.const(16384, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.maximums(t__tile_12, 0.0)
            v_win_valid_inline2547__tile: pl.Tile[[8, 128], pl.FP32, pl.MemRef(mem_vec_16, pl.const(16384, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.minimums(t__tile_13, 1.0)
            tmp_tile: pl.Tile[[8, 128], pl.FP32, pl.MemRef(mem_vec_32, pl.const(32768, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.create([8, 128], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            raw_block_valid_inline2576__tile: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_33, pl.const(36864, pl.INT64), 32), pl.Mem.Vec] = pl.tile.row_max(v_win_valid_inline2547__tile, tmp_tile)
            for c_t0_inline2591__idx_v0 in pl.range(8):
                t__tile_14: pl.Scalar[pl.FP32] = pl.tile.read(raw_block_valid_inline2576__tile, [c_t0_inline2591__idx_v0, 0])
                c_valid_inline2545__ssa_v0: pl.Scalar[pl.INT32] = pl.cast(t__tile_14, pl.INT32)
                pl.tensor.write(valid_block_mask_inline2517__ssa_v0, [bias_t0_inline2494__idx_v0 + c_t0_inline2591__idx_v0, 0], c_valid_inline2545__ssa_v0)
            for c_sb_inline2541__idx_v0 in pl.range(1, 5):
                c_s0_inline2581__ssa_v0: pl.Scalar[pl.INDEX] = (c_sb_inline2541__idx_v0 - 1) * 128
                t__tile_15: pl.Tile[[8, 128], pl.FP32, pl.MemRef(mem_vec_14, pl.const(0, pl.INT64), 14848), pl.Mem.Vec] = pl.tile.slice(c_mask_inline2481__tile, [8, 128], [0, c_s0_inline2581__ssa_v0])
                tmp_tile_1: pl.Tile[[8, 128], pl.FP32, pl.MemRef(mem_vec_32, pl.const(32768, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.create([8, 128], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                c_blk_valid_inline2496__tile: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_33, pl.const(36864, pl.INT64), 32), pl.Mem.Vec] = pl.tile.row_max(t__tile_15, tmp_tile_1)
                for c_dt_inline2498__idx_v0 in pl.range(8):
                    t__tile_16: pl.Scalar[pl.FP32] = pl.tile.read(c_blk_valid_inline2496__tile, [c_dt_inline2498__idx_v0, 0])
                    c_valid_inline2545__ssa_v1: pl.Scalar[pl.INT32] = pl.cast(t__tile_16, pl.INT32)
                    pl.tensor.write(valid_block_mask_inline2517__ssa_v0, [bias_t0_inline2494__idx_v0 + c_dt_inline2498__idx_v0, c_sb_inline2541__idx_v0], c_valid_inline2545__ssa_v1)
            t__tile_17: pl.Tile[[8, 128], pl.FP32, pl.MemRef(mem_vec_14, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.subs(v_win_valid_inline2547__tile, 1.0)
            t__tile_18: pl.Tile[[8, 128], pl.FP32, pl.MemRef(mem_vec_14, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.muls(t__tile_17, 1e20)
            sparse_bias_inline2513__tile: pl.Tensor[[t_dim_inline2515__ssa_v0, 640], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                t__tile_18, [bias_t0_inline2494__idx_v0, 0], sparse_bias_inline2513__iter_v1
            )
            t__tile_19: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_6, pl.const(36896, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.minimums(c_out_inline2544__tile, 0.0)
            t__tile_20: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_6, pl.const(36896, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.muls(t__tile_19, 1e20)
            sparse_bias_inline2513__tile_1: pl.Tensor[[t_dim_inline2515__ssa_v0, 640], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                t__tile_20, [bias_t0_inline2494__idx_v0, 128], sparse_bias_inline2513__tile
            )
            cmp_sparse_indices_inline2527__rv_v2, sparse_bias_inline2513__rv_v2 = pl.yield_(cmp_sparse_indices_inline2527__tile, sparse_bias_inline2513__tile_1)
        return cmp_sparse_indices_inline2527__ssa_v0, sparse_bias_inline2513__ssa_v0

    @pl.function(type=pl.FunctionType.Spmd)
    def csa_slots_build_valid_qk_plan_spmd(
        self,
        cmp_sparse_indices_inline2527__ssa_v0: pl.Out[pl.Tensor[[t_dim_inline2515__ssa_v0, 512], pl.INT32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)]],
        sparse_bias_inline2513__ssa_v0: pl.Out[pl.Tensor[[t_dim_inline2515__ssa_v0, 640], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)]],
        t_dim_inline2515__ssa_v0: pl.Scalar[pl.INDEX],
        idx_topk__ssa_v3: pl.Tensor[[T_DYN, 512], pl.INT32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)],
        position_ids_t1__ssa_v0: pl.Tensor[[T_DYN, 1], pl.INT32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)],
        window_swa_indices__ssa_v0: pl.Tensor[[T_DYN, 128], pl.INT32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 0)],
        valid_block_mask_inline2517__ssa_v0: pl.Out[pl.Tensor[[t_dim_inline2515__ssa_v0, 16], pl.INT32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 0)]],
    ) -> tuple[pl.Tensor[[t_dim_inline2515__ssa_v0, 512], pl.INT32], pl.Tensor[[t_dim_inline2515__ssa_v0, 640], pl.FP32]]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        ret__tmp_v0: pl.Tuple[pl.Tensor[[t_dim_inline2515__ssa_v0, 512], pl.INT32], pl.Tensor[[t_dim_inline2515__ssa_v0, 640], pl.FP32]] = self.csa_slots_build_valid_qk_plan(
            cmp_sparse_indices_inline2527__ssa_v0,
            sparse_bias_inline2513__ssa_v0,
            t_dim_inline2515__ssa_v0,
            idx_topk__ssa_v3,
            position_ids_t1__ssa_v0,
            window_swa_indices__ssa_v0,
            valid_block_mask_inline2517__ssa_v0,
            attrs={"arg_directions": [pl.adir.output_existing, pl.adir.output_existing, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.output_existing]},
        )
        cmp_sparse_indices_inline2527__rv_v2: pl.Tensor[[t_dim_inline2515__ssa_v0, 512], pl.INT32, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 0)] = ret__tmp_v0[0]
        sparse_bias_inline2513__rv_v2: pl.Tensor[[t_dim_inline2515__ssa_v0, 640], pl.FP32, pl.MemRef("mem_ddr_7", pl.const(0, pl.INT64), 0)] = ret__tmp_v0[1]
        return cmp_sparse_indices_inline2527__ssa_v0, sparse_bias_inline2513__ssa_v0

    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def idx_kv_scale_commit(
        compact_rows_inline2241__ssa_v0: pl.Scalar[pl.INDEX],
        position_ids__ssa_v0: pl.Tensor[[T_DYN], pl.INT32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)],
        idx_slot_mapping__ssa_v0: pl.Tensor[[T_DYN], pl.INT64, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)],
        idx_kv_cache__rv_v2: pl.InOut[pl.Tensor[[IDX_CACHE_BLOCK_NUM_DYN, INDEXER_PAGE_BYTES_DYN], pl.INT8, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)]],
        idx_kv_scale_values_inline2235__ssa_v1: pl.Tensor[[384, 1], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 1536)],
    ) -> pl.Tensor[[IDX_CACHE_BLOCK_NUM_DYN, INDEXER_PAGE_BYTES_DYN], pl.INT8]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        mem_vec_4: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 64)
        for compact_token_inline2233__idx_v0 in pl.range(compact_rows_inline2241__ssa_v0):
            request_inline2220__ssa_v1: pl.Scalar[pl.INDEX] = compact_token_inline2233__idx_v0 // 2
            first_pos_inline2219__tile: pl.Scalar[pl.INT32] = pl.tensor.read(position_ids__ssa_v0, [request_inline2220__ssa_v1 * 6])
            local_token_inline2217__ssa_v1: pl.Scalar[pl.INDEX] = compact_token_inline2233__idx_v0 % 2 * 4 - pl.cast(first_pos_inline2219__tile, pl.INDEX) % 4 + 3
            if local_token_inline2217__ssa_v1 < 6:
                token_v1_inline2213__ssa_v0: pl.Scalar[pl.INDEX] = request_inline2220__ssa_v1 * 6 + local_token_inline2217__ssa_v1
                cache_row_i64_v1_inline2252__tile: pl.Scalar[pl.INT64] = pl.tensor.read(idx_slot_mapping__ssa_v0, [token_v1_inline2213__ssa_v0])
                if 0 <= cache_row_i64_v1_inline2252__tile:
                    cache_row_v1_inline2225__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(cache_row_i64_v1_inline2252__tile, pl.INDEX)
                    scale_page_inline2212__ssa_v0: pl.Scalar[pl.INDEX] = cache_row_v1_inline2225__ssa_v0 // 32
                    scale_bytes_inline2211__ssa_v0: pl.Tile[[1, 64], pl.INT8, pl.MemRef(mem_vec_4, pl.const(0, pl.INT64), 64), pl.Mem.Vec] = pl.tile.load(
                        idx_kv_cache__rv_v2, [scale_page_inline2212__ssa_v0, 4096], [1, 64], [1, 64], target_memory=pl.Mem.Vec
                    )
                    scale_half_inline2232__ssa_v0: pl.Tile[[1, 32], pl.FP16, pl.MemRef(mem_vec_4, pl.const(0, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reinterpret_view(
                        scale_bytes_inline2211__ssa_v0, dtype=pl.FP16
                    )
                    t__tile: pl.Scalar[pl.FP32] = pl.tensor.read(idx_kv_scale_values_inline2235__ssa_v1, [compact_token_inline2233__idx_v0, 0])
                    pl.tile.write(scale_half_inline2232__ssa_v0, [0, cache_row_v1_inline2225__ssa_v0 % 32], pl.cast(t__tile, pl.FP16))
                    updated_bytes_inline2221__ssa_v0: pl.Tile[[1, 64], pl.INT8, pl.MemRef(mem_vec_4, pl.const(0, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reinterpret_view(
                        scale_half_inline2232__ssa_v0, dtype=pl.INT8
                    )
                    idx_kv_cache__store: pl.Tensor[[IDX_CACHE_BLOCK_NUM_DYN, INDEXER_PAGE_BYTES_DYN], pl.INT8, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                        updated_bytes_inline2221__ssa_v0, [scale_page_inline2212__ssa_v0, 4096], idx_kv_cache__rv_v2
                    )
        return idx_kv_cache__rv_v2

    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def idx_qr_dequant_rope(
        qr_bf16_2d_inline975_inline2321__ssa_v0: pl.Out[pl.Tensor[[384, 8192], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 6291456)]],
        dq_rope_units_inline957_inline2290__ssa_v0: pl.Scalar[pl.INDEX],
        dq_rope_workers_inline969_inline2273__ssa_v0: pl.Scalar[pl.INDEX],
        qr_scale__ssa_v0: pl.Tensor[[T_DYN, 1], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)],
        idx_cos_il__rv_v2: pl.Tensor[[T_DYN, 64], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)],
        idx_sin_signed__rv_v2: pl.Tensor[[T_DYN, 64], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)],
        idx_wq_b_scale__ssa_v0: pl.Tensor[[8192], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 32768)],
        qr_acc_pad_inline964_inline2287__rv_v2: pl.Tensor[[384, 8192], pl.INT32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 12582912)],
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
        dq_rope_worker_inline961_inline2297__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        sw_ones_inline977_inline2268__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_24, pl.const(6176, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.full([8, 64], dtype=pl.FP32, value=1.0)
        t__ci_tmp_v0: pl.Tile[[1, 192], pl.FP32, pl.MemRef(mem_vec_26, pl.const(10272, pl.INT64), 768), pl.Mem.Vec] = pl.tile.create([1, 192], dtype=pl.FP32, target_memory=pl.Mem.Vec)
        t__tile: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_28, pl.const(14368, pl.INT64), 256), pl.Mem.Vec] = pl.tile.ci(
            pl.const(0, pl.INT32), [1, 64], tmp=t__ci_tmp_v0, dtype=pl.INT32, descending=False
        )
        sw_index_inline953_inline2316__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_26, pl.const(10272, pl.INT64), 256), pl.Mem.Vec] = pl.tile.cast(t__tile, target_type=pl.FP32, mode="round")
        sw_col_inline979_inline2267__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_24, pl.const(6176, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.col_expand_mul(
            sw_ones_inline977_inline2268__tile, sw_index_inline953_inline2316__tile
        )
        t__tile_1: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_26, pl.const(10272, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.muls(sw_col_inline979_inline2267__tile, 0.5)
        t__tile_2: pl.Tile[[8, 64], pl.INT32, pl.MemRef(mem_vec_26, pl.const(10272, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(t__tile_1, target_type=pl.INT32, mode="trunc")
        sw_dup_f_inline983_inline2289__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_26, pl.const(10272, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(t__tile_2, target_type=pl.FP32, mode="round")
        t__tile_3: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_26, pl.const(10272, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.muls(sw_dup_f_inline983_inline2289__tile, 2.0)
        sw_lane_inline984_inline2285__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_26, pl.const(10272, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.sub(sw_col_inline979_inline2267__tile, t__tile_3)
        t__tile_4: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_24, pl.const(6176, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.adds(sw_col_inline979_inline2267__tile, 1.0)
        t__tile_5: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_26, pl.const(10272, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.muls(sw_lane_inline984_inline2285__tile, 2.0)
        t__tile_6: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_24, pl.const(6176, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.sub(t__tile_4, t__tile_5)
        rope_swap_idx_inline985_inline2305__tile: pl.Tile[[8, 64], pl.INT32, pl.MemRef(mem_vec_19, pl.const(0, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(
            t__tile_6, target_type=pl.INT32, mode="round"
        )
        for dq_unit_inline976_inline2329__idx_v0, (qr_bf16_2d_inline975_inline2321__iter_v1,) in pl.range(
            dq_rope_worker_inline961_inline2297__ssa_v0,
            dq_rope_units_inline957_inline2290__ssa_v0,
            dq_rope_workers_inline969_inline2273__ssa_v0,
            init_values=(qr_bf16_2d_inline975_inline2321__ssa_v0,),
        ):
            hg_inline982_inline2311__ssa_v0: pl.Scalar[pl.INDEX] = dq_unit_inline976_inline2329__idx_v0 % 16 * 4
            dq_t0_inline954_inline2322__ssa_v0: pl.Scalar[pl.INDEX] = dq_unit_inline976_inline2329__idx_v0 // 16 * 8
            qr_scale_tile_inline949_inline2315__tile: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_20, pl.const(2048, pl.INT64), 32), pl.Mem.Vec] = pl.tile.load(
                qr_scale__ssa_v0, [dq_t0_inline954_inline2322__ssa_v0, 0], [8, 1], [8, 1], target_memory=pl.Mem.Vec
            )
            cos_tile_inline951_inline2333__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_21, pl.const(2080, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.load(
                idx_cos_il__rv_v2, [dq_t0_inline954_inline2322__ssa_v0, 0], [8, 64], [8, 64], target_memory=pl.Mem.Vec
            )
            sin_tile_inline980_inline2319__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_22, pl.const(4128, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.load(
                idx_sin_signed__rv_v2, [dq_t0_inline954_inline2322__ssa_v0, 0], [8, 64], [8, 64], target_memory=pl.Mem.Vec
            )
            for h_inner_inline948_inline2282__idx_v0, (qr_bf16_2d_inline975_inline2321__iter_v3,) in pl.range(0, 4, 2, init_values=(qr_bf16_2d_inline975_inline2321__iter_v1,)):
                h0_inline947_inline2324__ssa_v0: pl.Scalar[pl.INDEX] = (hg_inline982_inline2311__ssa_v0 + h_inner_inline948_inline2282__idx_v0) * 128
                h0_inline947_inline2324__ssa_v0_1: pl.Scalar[pl.INDEX] = (hg_inline982_inline2311__ssa_v0 + (h_inner_inline948_inline2282__idx_v0 + 1)) * 128
                t__tile_7: pl.Tile[[128], pl.FP32, pl.MemRef(mem_vec_34, pl.const(18464, pl.INT64), 512), pl.Mem.Vec] = pl.tile.load(
                    idx_wq_b_scale__ssa_v0, [h0_inline947_inline2324__ssa_v0], [128], [128], target_memory=pl.Mem.Vec
                )
                t__tile_8: pl.Tile[[8, 128], pl.INT32, pl.MemRef(mem_vec_24, pl.const(6176, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.load(
                    qr_acc_pad_inline964_inline2287__rv_v2, [dq_t0_inline954_inline2322__ssa_v0, h0_inline947_inline2324__ssa_v0], [8, 128], [8, 128], target_memory=pl.Mem.Vec
                )
                t__tile_9: pl.Tile[[128], pl.FP32, pl.MemRef(mem_vec_50, pl.const(24096, pl.INT64), 512), pl.Mem.Vec] = pl.tile.load(
                    idx_wq_b_scale__ssa_v0, [h0_inline947_inline2324__ssa_v0_1], [128], [128], target_memory=pl.Mem.Vec
                )
                t__tile_10: pl.Tile[[8, 128], pl.INT32, pl.MemRef(mem_vec_26, pl.const(10272, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.load(
                    qr_acc_pad_inline964_inline2287__rv_v2, [dq_t0_inline954_inline2322__ssa_v0, h0_inline947_inline2324__ssa_v0_1], [8, 128], [8, 128], target_memory=pl.Mem.Vec
                )
                wq_scale_inline946_inline2331__tile: pl.Tile[[1, 128], pl.FP32, pl.MemRef(mem_vec_34, pl.const(18464, pl.INT64), 512), pl.Mem.Vec] = t__tile_7
                acc_fp32_inline960_inline2309__tile: pl.Tile[[8, 128], pl.FP32, pl.MemRef(mem_vec_24, pl.const(6176, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(
                    t__tile_8, target_type=pl.FP32, mode="none"
                )
                t__tile_11: pl.Tile[[8, 128], pl.FP32, pl.MemRef(mem_vec_28, pl.const(14368, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.full([8, 128], dtype=pl.FP32, value=1.0)
                t__tile_12: pl.Tile[[8, 128], pl.FP32, pl.MemRef(mem_vec_28, pl.const(14368, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.row_expand_mul(
                    t__tile_11, qr_scale_tile_inline949_inline2315__tile
                )
                qr_dequant_scale_inline945_inline2327__tile: pl.Tile[[8, 128], pl.FP32, pl.MemRef(mem_vec_28, pl.const(14368, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.col_expand_mul(
                    t__tile_12, wq_scale_inline946_inline2331__tile
                )
                qr_dequant_inline944_inline2299__tile: pl.Tile[[8, 128], pl.FP32, pl.MemRef(mem_vec_24, pl.const(6176, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.mul(
                    acc_fp32_inline960_inline2309__tile, qr_dequant_scale_inline945_inline2327__tile
                )
                t__tile_13: pl.Tile[[8, 128], pl.BF16, pl.MemRef(mem_vec_28, pl.const(14368, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(
                    qr_dequant_inline944_inline2299__tile, target_type=pl.BF16, mode="rint"
                )
                qr_dequant_v1_inline968_inline2328__tile: pl.Tile[[8, 128], pl.FP32, pl.MemRef(mem_vec_24, pl.const(6176, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(
                    t__tile_13, target_type=pl.FP32, mode="round"
                )
                t__tile_14: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_24, pl.const(6176, pl.INT64), 3840), pl.Mem.Vec] = pl.tile.slice(qr_dequant_v1_inline968_inline2328__tile, [8, 64], [0, 0])
                qr_nope_bf16_inline943_inline2330__tile: pl.Tile[[8, 64], pl.BF16, pl.MemRef(mem_vec_34, pl.const(18464, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.cast(
                    t__tile_14, target_type=pl.BF16, mode="rint"
                )
                qr_rope_slice_inline942_inline2291__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_24, pl.const(6432, pl.INT64), 3840), pl.Mem.Vec] = pl.tile.slice(
                    qr_dequant_v1_inline968_inline2328__tile, [8, 64], [0, 64]
                )
                gather_acc_init: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_28, pl.const(14368, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.create([8, 64], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                for gather_lv, (gather_ia,) in pl.range(8, init_values=(gather_acc_init,)):
                    gather_inp_row: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_24, pl.const(6176, pl.INT64), 256), pl.Mem.Vec] = pl.tile.slice(
                        qr_rope_slice_inline942_inline2291__tile, [1, 64], [gather_lv, 0], [1, 64]
                    )
                    gather_idx_row: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_19, pl.const(0, pl.INT64), 256), pl.Mem.Vec] = pl.tile.slice(
                        rope_swap_idx_inline985_inline2305__tile, [1, 64], [gather_lv, 0], [1, 64]
                    )
                    gather_row_tmp: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_36, pl.const(19488, pl.INT64), 256), pl.Mem.Vec] = pl.tile.create([1, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec)
                    gather_row: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_37, pl.const(19744, pl.INT64), 256), pl.Mem.Vec] = pl.tile.gather(gather_inp_row, gather_idx_row, gather_row_tmp)
                    gather_asmbl: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_28, pl.const(14368, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.assemble(gather_ia, gather_row, [gather_lv, 0])
                    gather_rv: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_28, pl.const(14368, pl.INT64), 2048), pl.Mem.Vec] = pl.yield_(gather_asmbl)
                qr_swapped_inline950_inline2320__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_28, pl.const(14368, pl.INT64), 2048), pl.Mem.Vec] = gather_rv
                t__tile_15: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_24, pl.const(6176, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.mul(
                    qr_rope_slice_inline942_inline2291__tile, cos_tile_inline951_inline2333__tile
                )
                t__tile_16: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_28, pl.const(14368, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.mul(
                    qr_swapped_inline950_inline2320__tile, sin_tile_inline980_inline2319__tile
                )
                rope_rot_inline941_inline2304__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_24, pl.const(6176, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.add(t__tile_15, t__tile_16)
                rope_bf16_inline967_inline2332__tile: pl.Tile[[8, 64], pl.BF16, pl.MemRef(mem_vec_24, pl.const(6176, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.cast(
                    rope_rot_inline941_inline2304__tile, target_type=pl.BF16, mode="rint"
                )
                qr_bf16_2d_inline975_inline2321__tile: pl.Tensor[[384, 8192], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 6291456)] = pl.tile.store(
                    qr_nope_bf16_inline943_inline2330__tile, [dq_t0_inline954_inline2322__ssa_v0, h0_inline947_inline2324__ssa_v0], qr_bf16_2d_inline975_inline2321__iter_v3
                )
                qr_bf16_2d_inline975_inline2321__tile_1: pl.Tensor[[384, 8192], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 6291456)] = pl.tile.store(
                    rope_bf16_inline967_inline2332__tile, [dq_t0_inline954_inline2322__ssa_v0, h0_inline947_inline2324__ssa_v0 + 64], qr_bf16_2d_inline975_inline2321__tile
                )
                wq_scale_inline946_inline2331__tile_1: pl.Tile[[1, 128], pl.FP32, pl.MemRef(mem_vec_50, pl.const(24096, pl.INT64), 512), pl.Mem.Vec] = t__tile_9
                acc_fp32_inline960_inline2309__tile_1: pl.Tile[[8, 128], pl.FP32, pl.MemRef(mem_vec_26, pl.const(10272, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(
                    t__tile_10, target_type=pl.FP32, mode="none"
                )
                t__tile_17: pl.Tile[[8, 128], pl.FP32, pl.MemRef(mem_vec_44, pl.const(20000, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.full([8, 128], dtype=pl.FP32, value=1.0)
                t__tile_18: pl.Tile[[8, 128], pl.FP32, pl.MemRef(mem_vec_44, pl.const(20000, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.row_expand_mul(
                    t__tile_17, qr_scale_tile_inline949_inline2315__tile
                )
                qr_dequant_scale_inline945_inline2327__tile_1: pl.Tile[[8, 128], pl.FP32, pl.MemRef(mem_vec_44, pl.const(20000, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.col_expand_mul(
                    t__tile_18, wq_scale_inline946_inline2331__tile_1
                )
                qr_dequant_inline944_inline2299__tile_1: pl.Tile[[8, 128], pl.FP32, pl.MemRef(mem_vec_26, pl.const(10272, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.mul(
                    acc_fp32_inline960_inline2309__tile_1, qr_dequant_scale_inline945_inline2327__tile_1
                )
                t__tile_19: pl.Tile[[8, 128], pl.BF16, pl.MemRef(mem_vec_44, pl.const(20000, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(
                    qr_dequant_inline944_inline2299__tile_1, target_type=pl.BF16, mode="rint"
                )
                qr_dequant_v1_inline968_inline2328__tile_1: pl.Tile[[8, 128], pl.FP32, pl.MemRef(mem_vec_26, pl.const(10272, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(
                    t__tile_19, target_type=pl.FP32, mode="round"
                )
                t__tile_20: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_26, pl.const(10272, pl.INT64), 3840), pl.Mem.Vec] = pl.tile.slice(qr_dequant_v1_inline968_inline2328__tile_1, [8, 64], [0, 0])
                qr_nope_bf16_inline943_inline2330__tile_1: pl.Tile[[8, 64], pl.BF16, pl.MemRef(mem_vec_50, pl.const(24096, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.cast(
                    t__tile_20, target_type=pl.BF16, mode="rint"
                )
                qr_rope_slice_inline942_inline2291__tile_1: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_26, pl.const(10528, pl.INT64), 3840), pl.Mem.Vec] = pl.tile.slice(
                    qr_dequant_v1_inline968_inline2328__tile_1, [8, 64], [0, 64]
                )
                gather_acc_init_1: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_44, pl.const(20000, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.create([8, 64], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                for gather_lv_1, (gather_ia_1,) in pl.range(8, init_values=(gather_acc_init_1,)):
                    gather_inp_row_1: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_26, pl.const(10272, pl.INT64), 256), pl.Mem.Vec] = pl.tile.slice(
                        qr_rope_slice_inline942_inline2291__tile_1, [1, 64], [gather_lv_1, 0], [1, 64]
                    )
                    gather_idx_row_1: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_19, pl.const(0, pl.INT64), 256), pl.Mem.Vec] = pl.tile.slice(
                        rope_swap_idx_inline985_inline2305__tile, [1, 64], [gather_lv_1, 0], [1, 64]
                    )
                    gather_row_tmp_1: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_52, pl.const(25120, pl.INT64), 256), pl.Mem.Vec] = pl.tile.create([1, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec)
                    gather_row_1: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_53, pl.const(25376, pl.INT64), 256), pl.Mem.Vec] = pl.tile.gather(gather_inp_row_1, gather_idx_row_1, gather_row_tmp_1)
                    gather_asmbl_1: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_44, pl.const(20000, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.assemble(gather_ia_1, gather_row_1, [gather_lv_1, 0])
                    gather_rv_1: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_44, pl.const(20000, pl.INT64), 2048), pl.Mem.Vec] = pl.yield_(gather_asmbl_1)
                qr_swapped_inline950_inline2320__tile_1: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_44, pl.const(20000, pl.INT64), 2048), pl.Mem.Vec] = gather_rv_1
                t__tile_21: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_26, pl.const(10272, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.mul(
                    qr_rope_slice_inline942_inline2291__tile_1, cos_tile_inline951_inline2333__tile
                )
                t__tile_22: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_44, pl.const(20000, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.mul(
                    qr_swapped_inline950_inline2320__tile_1, sin_tile_inline980_inline2319__tile
                )
                rope_rot_inline941_inline2304__tile_1: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_26, pl.const(10272, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.add(t__tile_21, t__tile_22)
                rope_bf16_inline967_inline2332__tile_1: pl.Tile[[8, 64], pl.BF16, pl.MemRef(mem_vec_26, pl.const(10272, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.cast(
                    rope_rot_inline941_inline2304__tile_1, target_type=pl.BF16, mode="rint"
                )
                qr_bf16_2d_inline975_inline2321__tile_2: pl.Tensor[[384, 8192], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 6291456)] = pl.tile.store(
                    qr_nope_bf16_inline943_inline2330__tile_1, [dq_t0_inline954_inline2322__ssa_v0, h0_inline947_inline2324__ssa_v0_1], qr_bf16_2d_inline975_inline2321__tile_1
                )
                qr_bf16_2d_inline975_inline2321__tile_3: pl.Tensor[[384, 8192], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 6291456)] = pl.tile.store(
                    rope_bf16_inline967_inline2332__tile_1, [dq_t0_inline954_inline2322__ssa_v0, h0_inline947_inline2324__ssa_v0_1 + 64], qr_bf16_2d_inline975_inline2321__tile_2
                )
                qr_bf16_2d_inline975_inline2321__rv_v4: pl.Tensor[[384, 8192], pl.BF16, pl.MemRef("mem_ddr_59", pl.const(0, pl.INT64), 6291456)] = pl.yield_(qr_bf16_2d_inline975_inline2321__tile_3)
            qr_bf16_2d_inline975_inline2321__rv_v2: pl.Tensor[[384, 8192], pl.BF16, pl.MemRef("mem_ddr_60", pl.const(0, pl.INT64), 6291456)] = pl.yield_(qr_bf16_2d_inline975_inline2321__rv_v4)
        return

    @pl.function(type=pl.FunctionType.Spmd)
    def idx_qr_dequant_rope_spmd(
        self,
        qr_bf16_2d_inline975_inline2321__ssa_v0: pl.Out[pl.Tensor[[384, 8192], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 6291456)]],
        dq_rope_units_inline957_inline2290__ssa_v0: pl.Scalar[pl.INDEX],
        dq_rope_workers_inline969_inline2273__ssa_v0: pl.Scalar[pl.INDEX],
        qr_scale__ssa_v0: pl.Tensor[[T_DYN, 1], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)],
        idx_cos_il__rv_v2: pl.Tensor[[T_DYN, 64], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)],
        idx_sin_signed__rv_v2: pl.Tensor[[T_DYN, 64], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)],
        idx_wq_b_scale__ssa_v0: pl.Tensor[[8192], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 32768)],
        qr_acc_pad_inline964_inline2287__rv_v2: pl.Tensor[[384, 8192], pl.INT32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 12582912)],
    ):
        pl.func_attr({"mx_tensor_views_blocked": True})
        self.idx_qr_dequant_rope(
            qr_bf16_2d_inline975_inline2321__ssa_v0,
            dq_rope_units_inline957_inline2290__ssa_v0,
            dq_rope_workers_inline969_inline2273__ssa_v0,
            qr_scale__ssa_v0,
            idx_cos_il__rv_v2,
            idx_sin_signed__rv_v2,
            idx_wq_b_scale__ssa_v0,
            qr_acc_pad_inline964_inline2287__rv_v2,
            attrs={"arg_directions": [pl.adir.output_existing, pl.adir.scalar, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input]},
        )

    @pl.function(type=pl.FunctionType.AIC, level=pl.Level.AIC, role=pl.Role.SubWorker)
    def idx_qr_proj_matmul(
        qr_acc_pad_inline964_inline2287__ssa_v0: pl.Out[pl.Tensor[[384, 8192], pl.INT32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 12582912)]],
        row_blocks_inline966_inline2326__ssa_v0: pl.Scalar[pl.INDEX],
        bs_inline970_inline2307__ssa_v0: pl.Scalar[pl.INDEX],
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
        qr_proj_worker_inline962_inline2277__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        for qr_unit_inline974_inline2288__idx_v0, (qr_acc_pad_inline964_inline2287__iter_v1,) in pl.range(
            qr_proj_worker_inline962_inline2277__ssa_v0, row_blocks_inline966_inline2326__ssa_v0 * 8, 24, init_values=(qr_acc_pad_inline964_inline2287__ssa_v0,)
        ):
            qr_rb_inline971_inline2300__ssa_v0: pl.Scalar[pl.INDEX] = qr_unit_inline974_inline2288__idx_v0 // 8
            ot_inline956_inline2313__ssa_v0: pl.Scalar[pl.INDEX] = qr_unit_inline974_inline2288__idx_v0 - qr_rb_inline971_inline2300__ssa_v0 * 8
            qr_r0_inline958_inline2295__ssa_v0: pl.Scalar[pl.INDEX] = qr_rb_inline971_inline2300__ssa_v0 * 16
            qr_rows_inline981_inline2301__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(bs_inline970_inline2307__ssa_v0 - qr_r0_inline958_inline2295__ssa_v0, 16)
            o_base_inline955_inline2303__ssa_v0: pl.Scalar[pl.INDEX] = ot_inline956_inline2313__ssa_v0 * 1024
            for ns_inline965_inline2310__idx_v0, (qr_acc_pad_inline964_inline2287__iter_v3,) in pl.range(0, 1024, 512, init_values=(qr_acc_pad_inline964_inline2287__iter_v1,)):
                qr_acc_inline952_inline2283__tile: pl.Tile[[16, 512], pl.INT32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 65536), pl.Mem.Acc] = pl.tile.create(
                    [16, 512], dtype=pl.INT32, target_memory=pl.Mem.Acc
                )
                for kb_inline972_inline2275__idx_v0, (qr_acc_inline952_inline2283__iter_v1,) in pl.range(0, 4, 2, init_values=(qr_acc_inline952_inline2283__tile,)):
                    q0_inline963_inline2280__ssa_v0: pl.Scalar[pl.INDEX] = kb_inline972_inline2275__idx_v0 * 256
                    q0_inline963_inline2280__ssa_v0_1: pl.Scalar[pl.INDEX] = kb_inline972_inline2275__idx_v0 * 256 + 256
                    qr_tile_inline959_inline2271__tile: pl.Tile[
                        [16, 256], pl.INT8, pl.MemRef(mem_mat_4, pl.const(0, pl.INT64), 4096), pl.Mem.Mat, pl.TileView(valid_shape=[qr_rows_inline981_inline2301__ssa_v0, 256])
                    ] = pl.tile.load(
                        qr__ssa_v0, [qr_r0_inline958_inline2295__ssa_v0, q0_inline963_inline2280__ssa_v0], [16, 256], [qr_rows_inline981_inline2301__ssa_v0, 256], target_memory=pl.Mem.Mat
                    )
                    wq_tile_inline973_inline2278__tile: pl.Tile[[256, 512], pl.INT8, pl.MemRef(mem_mat_5, pl.const(4096, pl.INT64), 131072), pl.Mem.Mat] = pl.tile.load(
                        idx_wq_b__ssa_v0, [q0_inline963_inline2280__ssa_v0, o_base_inline955_inline2303__ssa_v0 + ns_inline965_inline2310__idx_v0], [256, 512], [256, 512], target_memory=pl.Mem.Mat
                    )
                    qr_tile_inline959_inline2271__tile_1: pl.Tile[
                        [16, 256], pl.INT8, pl.MemRef(mem_mat_6, pl.const(135168, pl.INT64), 4096), pl.Mem.Mat, pl.TileView(valid_shape=[qr_rows_inline981_inline2301__ssa_v0, 256])
                    ] = pl.tile.load(
                        qr__ssa_v0, [qr_r0_inline958_inline2295__ssa_v0, q0_inline963_inline2280__ssa_v0_1], [16, 256], [qr_rows_inline981_inline2301__ssa_v0, 256], target_memory=pl.Mem.Mat
                    )
                    wq_tile_inline973_inline2278__tile_1: pl.Tile[[256, 512], pl.INT8, pl.MemRef(mem_mat_7, pl.const(139264, pl.INT64), 131072), pl.Mem.Mat] = pl.tile.load(
                        idx_wq_b__ssa_v0, [q0_inline963_inline2280__ssa_v0_1, o_base_inline955_inline2303__ssa_v0 + ns_inline965_inline2310__idx_v0], [256, 512], [256, 512], target_memory=pl.Mem.Mat
                    )
                    for qr_acc_inline952_inline2283__tile_l0_ko, (qr_acc_inline952_inline2283__tile_l0_c,) in pl.range(0, 256, 128, init_values=(qr_acc_inline952_inline2283__iter_v1,)):
                        qr_acc_inline952_inline2283__tile_l0_a: pl.Tile[
                            [16, 64],
                            pl.INT8,
                            pl.MemRef(mem_left_8, pl.const(3072, pl.INT64), 1024),
                            pl.Mem.Left,
                            pl.TileView(valid_shape=[qr_rows_inline981_inline2301__ssa_v0, 64], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal),
                        ] = pl.tile.extract(qr_tile_inline959_inline2271__tile, 0, qr_acc_inline952_inline2283__tile_l0_ko, [16, 64], target_memory=pl.Mem.Left)
                        qr_acc_inline952_inline2283__tile_l0_b: pl.Tile[[64, 512], pl.INT8, pl.MemRef(mem_right_9, pl.const(32768, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                            wq_tile_inline973_inline2278__tile, qr_acc_inline952_inline2283__tile_l0_ko, 0, [64, 512], target_memory=pl.Mem.Right
                        )
                        qr_acc_inline952_inline2283__tile_l0_a_1: pl.Tile[
                            [16, 64],
                            pl.INT8,
                            pl.MemRef(mem_left_10, pl.const(0, pl.INT64), 1024),
                            pl.Mem.Left,
                            pl.TileView(valid_shape=[qr_rows_inline981_inline2301__ssa_v0, 64], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal),
                        ] = pl.tile.extract(qr_tile_inline959_inline2271__tile, 0, qr_acc_inline952_inline2283__tile_l0_ko + 64, [16, 64], target_memory=pl.Mem.Left)
                        qr_acc_inline952_inline2283__tile_l0_b_1: pl.Tile[[64, 512], pl.INT8, pl.MemRef(mem_right_11, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                            wq_tile_inline973_inline2278__tile, qr_acc_inline952_inline2283__tile_l0_ko + 64, 0, [64, 512], target_memory=pl.Mem.Right
                        )
                        qr_acc_inline952_inline2283__tile_l0_c_acc: pl.Tile[[16, 512], pl.INT32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 65536), pl.Mem.Acc] = pl.tile.matmul_acc(
                            qr_acc_inline952_inline2283__tile_l0_c,
                            qr_acc_inline952_inline2283__tile_l0_a,
                            qr_acc_inline952_inline2283__tile_l0_b,
                            q0_inline963_inline2280__ssa_v0 == 0 and qr_acc_inline952_inline2283__tile_l0_ko == 0,
                        )
                        qr_acc_inline952_inline2283__tile_l0_c_acc_1: pl.Tile[[16, 512], pl.INT32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 65536), pl.Mem.Acc] = pl.tile.matmul_acc(
                            qr_acc_inline952_inline2283__tile_l0_c_acc,
                            qr_acc_inline952_inline2283__tile_l0_a_1,
                            qr_acc_inline952_inline2283__tile_l0_b_1,
                            q0_inline963_inline2280__ssa_v0 == 0 and qr_acc_inline952_inline2283__tile_l0_ko == -64,
                        )
                        qr_acc_inline952_inline2283__tile_1: pl.Tile[[16, 512], pl.INT32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 65536), pl.Mem.Acc] = pl.yield_(
                            qr_acc_inline952_inline2283__tile_l0_c_acc_1
                        )
                    for qr_acc_inline952_inline2283__tile_l0_ko_1, (qr_acc_inline952_inline2283__tile_l0_c_1,) in pl.range(0, 256, 128, init_values=(qr_acc_inline952_inline2283__tile_1,)):
                        qr_acc_inline952_inline2283__tile_l0_a_2: pl.Tile[
                            [16, 64],
                            pl.INT8,
                            pl.MemRef(mem_left_13, pl.const(1024, pl.INT64), 1024),
                            pl.Mem.Left,
                            pl.TileView(valid_shape=[qr_rows_inline981_inline2301__ssa_v0, 64], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal),
                        ] = pl.tile.extract(qr_tile_inline959_inline2271__tile_1, 0, qr_acc_inline952_inline2283__tile_l0_ko_1, [16, 64], target_memory=pl.Mem.Left)
                        qr_acc_inline952_inline2283__tile_l0_b_2: pl.Tile[[64, 512], pl.INT8, pl.MemRef(mem_right_9, pl.const(32768, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                            wq_tile_inline973_inline2278__tile_1, qr_acc_inline952_inline2283__tile_l0_ko_1, 0, [64, 512], target_memory=pl.Mem.Right
                        )
                        qr_acc_inline952_inline2283__tile_l0_a_3: pl.Tile[
                            [16, 64],
                            pl.INT8,
                            pl.MemRef(mem_left_15, pl.const(2048, pl.INT64), 1024),
                            pl.Mem.Left,
                            pl.TileView(valid_shape=[qr_rows_inline981_inline2301__ssa_v0, 64], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal),
                        ] = pl.tile.extract(qr_tile_inline959_inline2271__tile_1, 0, qr_acc_inline952_inline2283__tile_l0_ko_1 + 64, [16, 64], target_memory=pl.Mem.Left)
                        qr_acc_inline952_inline2283__tile_l0_b_3: pl.Tile[[64, 512], pl.INT8, pl.MemRef(mem_right_11, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                            wq_tile_inline973_inline2278__tile_1, qr_acc_inline952_inline2283__tile_l0_ko_1 + 64, 0, [64, 512], target_memory=pl.Mem.Right
                        )
                        qr_acc_inline952_inline2283__tile_l0_c_acc_2: pl.Tile[[16, 512], pl.INT32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 65536), pl.Mem.Acc] = pl.tile.matmul_acc(
                            qr_acc_inline952_inline2283__tile_l0_c_1,
                            qr_acc_inline952_inline2283__tile_l0_a_2,
                            qr_acc_inline952_inline2283__tile_l0_b_2,
                            q0_inline963_inline2280__ssa_v0_1 == 0 and qr_acc_inline952_inline2283__tile_l0_ko_1 == 0,
                        )
                        qr_acc_inline952_inline2283__tile_l0_c_acc_3: pl.Tile[[16, 512], pl.INT32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 65536), pl.Mem.Acc] = pl.tile.matmul_acc(
                            qr_acc_inline952_inline2283__tile_l0_c_acc_2,
                            qr_acc_inline952_inline2283__tile_l0_a_3,
                            qr_acc_inline952_inline2283__tile_l0_b_3,
                            q0_inline963_inline2280__ssa_v0_1 == 0 and qr_acc_inline952_inline2283__tile_l0_ko_1 == -64,
                        )
                        qr_acc_inline952_inline2283__tile_2: pl.Tile[[16, 512], pl.INT32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 65536), pl.Mem.Acc] = pl.yield_(
                            qr_acc_inline952_inline2283__tile_l0_c_acc_3
                        )
                    qr_acc_inline952_inline2283__rv_v2: pl.Tile[[16, 512], pl.INT32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 65536), pl.Mem.Acc] = pl.yield_(qr_acc_inline952_inline2283__tile_2)
                qr_acc_pad_inline964_inline2287__tile: pl.Tensor[[384, 8192], pl.INT32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 12582912)] = pl.tile.store(
                    qr_acc_inline952_inline2283__rv_v2,
                    [qr_r0_inline958_inline2295__ssa_v0, o_base_inline955_inline2303__ssa_v0 + ns_inline965_inline2310__idx_v0],
                    qr_acc_pad_inline964_inline2287__iter_v3,
                )
                qr_acc_pad_inline964_inline2287__rv_v4: pl.Tensor[[384, 8192], pl.INT32, pl.MemRef("mem_ddr_19", pl.const(0, pl.INT64), 12582912)] = pl.yield_(qr_acc_pad_inline964_inline2287__tile)
            qr_acc_pad_inline964_inline2287__rv_v2: pl.Tensor[[384, 8192], pl.INT32, pl.MemRef("mem_ddr_20", pl.const(0, pl.INT64), 12582912)] = pl.yield_(qr_acc_pad_inline964_inline2287__rv_v4)
        return qr_acc_pad_inline964_inline2287__ssa_v0

    @pl.function(type=pl.FunctionType.Spmd)
    def idx_qr_proj_matmul_spmd(
        self,
        qr_acc_pad_inline964_inline2287__ssa_v0: pl.Out[pl.Tensor[[384, 8192], pl.INT32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 12582912)]],
        row_blocks_inline966_inline2326__ssa_v0: pl.Scalar[pl.INDEX],
        bs_inline970_inline2307__ssa_v0: pl.Scalar[pl.INDEX],
        qr__ssa_v0: pl.Tensor[[T_DYN, 1024], pl.INT8, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)],
        idx_wq_b__ssa_v0: pl.Tensor[[1024, 8192], pl.INT8, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 8388608)],
    ) -> pl.Tensor[[384, 8192], pl.INT32]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        qr_acc_pad_inline964_inline2287__rv_v2: pl.Tensor[[384, 8192], pl.INT32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 12582912)] = self.idx_qr_proj_matmul(
            qr_acc_pad_inline964_inline2287__ssa_v0,
            row_blocks_inline966_inline2326__ssa_v0,
            bs_inline970_inline2307__ssa_v0,
            qr__ssa_v0,
            idx_wq_b__ssa_v0,
            attrs={"arg_directions": [pl.adir.output_existing, pl.adir.scalar, pl.adir.scalar, pl.adir.input, pl.adir.input]},
        )
        return qr_acc_pad_inline964_inline2287__ssa_v0

    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def indexer_boundary_init(normed_kv_inline560__ssa_v0: pl.Out[pl.Tensor[[384, 128], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 98304)]]) -> pl.Tensor[[384, 128], pl.BF16]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        mem_vec_1: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 512)
        init_request_inline499_inline2130__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        compact_begin_inline443_inline2129__ssa_v0: pl.Scalar[pl.INDEX] = init_request_inline499_inline2130__ssa_v0 * 2
        t__tile: pl.Tile[[2, 128], pl.BF16, pl.MemRef(mem_vec_1, pl.const(0, pl.INT64), 512), pl.Mem.Vec] = pl.tile.full([2, 128], dtype=pl.BF16, value=0.0)
        normed_kv_inline560__tile: pl.Tensor[[384, 128], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 98304)] = pl.tile.store(
            t__tile, [compact_begin_inline443_inline2129__ssa_v0, 0], normed_kv_inline560__ssa_v0
        )
        return normed_kv_inline560__ssa_v0

    @pl.function(type=pl.FunctionType.Spmd)
    def indexer_boundary_init_spmd(self, normed_kv_inline560__ssa_v0: pl.Out[pl.Tensor[[384, 128], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 98304)]]) -> pl.Tensor[[384, 128], pl.BF16]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        normed_kv_inline560__ssa_v1: pl.Tensor[[384, 128], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 98304)] = self.indexer_boundary_init(
            normed_kv_inline560__ssa_v0, attrs={"arg_directions": [pl.adir.output_existing]}
        )
        return normed_kv_inline560__ssa_v0

    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def indexer_head_coefficients(
        position_ids__ssa_v0: pl.Tensor[[T_DYN], pl.INT32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)],
        coefficients_inline1085_inline2414__ssa_v0: pl.Out[pl.Tensor[[6144, 64], pl.FP16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 786432)]],
        qr_hadamard_scale_dq_inline567__rv_v2: pl.Tensor[[24576, 1], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 98304)],
        weights_inline2424__ssa_v1: pl.Tensor[[384, 64], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 98304)],
    ) -> pl.Tensor[[6144, 64], pl.FP16]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        mem_vec_5: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        mem_vec_7: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 128)
        mem_vec_8: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 4096)
        coefficient_worker_inline1104_inline2417__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        coefficient_count_inline1102_inline2389__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(position_ids__ssa_v0, 0)
        for coefficient_query_inline1108_inline2420__idx_v0, (coefficients_inline1085_inline2414__iter_v1,) in pl.range(
            coefficient_worker_inline1104_inline2417__ssa_v0, coefficient_count_inline1102_inline2389__ssa_v0, 48, init_values=(coefficients_inline1085_inline2414__ssa_v0,)
        ):
            coefficient_head_begin_inline1109_inline2423__ssa_v0: pl.Scalar[pl.INDEX] = coefficient_query_inline1108_inline2420__idx_v0 * 64
            t__tile: pl.Tile[[64, 1], pl.FP32, pl.MemRef(mem_vec_8, pl.const(384, pl.INT64), 256), pl.Mem.Vec] = pl.tile.load(
                qr_hadamard_scale_dq_inline567__rv_v2, [coefficient_head_begin_inline1109_inline2423__ssa_v0, 0], [64, 1], [64, 1], target_memory=pl.Mem.Vec
            )
            query_scale_inline1078_inline2403__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_8, pl.const(384, pl.INT64), 256), pl.Mem.Vec] = pl.tile.reshape(t__tile, [1, 64])
            query_weight_inline1082_inline2425__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_5, pl.const(0, pl.INT64), 256), pl.Mem.Vec] = pl.tile.load(
                weights_inline2424__ssa_v1, [coefficient_query_inline1108_inline2420__idx_v0, 0], [1, 64], [1, 64], target_memory=pl.Mem.Vec
            )
            t__tile_1: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_8, pl.const(384, pl.INT64), 256), pl.Mem.Vec] = pl.tile.mul(
                query_scale_inline1078_inline2403__tile, query_weight_inline1082_inline2425__tile
            )
            head_coefficient_inline1081_inline2422__tile: pl.Tile[[1, 64], pl.FP16, pl.MemRef(mem_vec_7, pl.const(256, pl.INT64), 128), pl.Mem.Vec] = pl.tile.cast(
                t__tile_1, target_type=pl.FP16, mode="rint"
            )
            t__tile_2: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_8, pl.const(384, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.full([16, 64], dtype=pl.FP32, value=1.0)
            t__tile_3: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_5, pl.const(0, pl.INT64), 256), pl.Mem.Vec] = pl.tile.cast(
                head_coefficient_inline1081_inline2422__tile, target_type=pl.FP32, mode="round"
            )
            coefficient_rows_inline1080_inline2397__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_8, pl.const(384, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.col_expand_mul(t__tile_2, t__tile_3)
            coefficient_row_inline1079_inline2427__ssa_v0: pl.Scalar[pl.INDEX] = coefficient_query_inline1108_inline2420__idx_v0 * 16
            t__tile_4: pl.Tile[[16, 64], pl.FP16, pl.MemRef(mem_vec_8, pl.const(384, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(
                coefficient_rows_inline1080_inline2397__tile, target_type=pl.FP16, mode="round"
            )
            coefficients_inline1085_inline2414__tile: pl.Tensor[[6144, 64], pl.FP16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 786432)] = pl.tile.store(
                t__tile_4, [coefficient_row_inline1079_inline2427__ssa_v0, 0], coefficients_inline1085_inline2414__iter_v1
            )
            coefficients_inline1085_inline2414__rv_v2: pl.Tensor[[6144, 64], pl.FP16, pl.MemRef("mem_ddr_12", pl.const(0, pl.INT64), 786432)] = pl.yield_(coefficients_inline1085_inline2414__tile)
        return coefficients_inline1085_inline2414__ssa_v0

    @pl.function(type=pl.FunctionType.Spmd)
    def indexer_head_coefficients_spmd(
        self,
        position_ids__ssa_v0: pl.Tensor[[T_DYN], pl.INT32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)],
        coefficients_inline1085_inline2414__ssa_v0: pl.Out[pl.Tensor[[6144, 64], pl.FP16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 786432)]],
        qr_hadamard_scale_dq_inline567__rv_v2: pl.Tensor[[24576, 1], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 98304)],
        weights_inline2424__ssa_v1: pl.Tensor[[384, 64], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 98304)],
    ) -> pl.Tensor[[6144, 64], pl.FP16]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        coefficients_inline1085_inline2414__rv_v2: pl.Tensor[[6144, 64], pl.FP16, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 786432)] = self.indexer_head_coefficients(
            position_ids__ssa_v0,
            coefficients_inline1085_inline2414__ssa_v0,
            qr_hadamard_scale_dq_inline567__rv_v2,
            weights_inline2424__ssa_v1,
            attrs={"arg_directions": [pl.adir.input, pl.adir.output_existing, pl.adir.input, pl.adir.input]},
        )
        return coefficients_inline1085_inline2414__ssa_v0

    @pl.function(type=pl.FunctionType.AIC, level=pl.Level.AIC, role=pl.Role.SubWorker)
    def indexer_score_topk_leaf_aic(
        position_ids__ssa_v0: pl.Tensor[[T_DYN], pl.INT32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)],
        kv_seq_lens__ssa_v0: pl.Tensor[[B_DYN], pl.INT32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)],
        score_arena_inline1090_inline2379__ssa_v0: pl.InOut[pl.Tensor[[384, 8192], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 12582912)]],
        qr_hadamard_i8_inline569__rv_v2: pl.Tensor[[24576, 128], pl.INT8, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 3145728)],
        coefficients_inline1085_inline2414__rv_v2: pl.Tensor[[6144, 64], pl.FP16, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 786432)],
        idx_block_table_flat_inline1138_inline2374__ssa_v0: pl.Tensor[[idx_table_len_inline1095_inline2409__ssa_v0], pl.INT32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 0)],
        table_columns_inline1070_inline2373__ssa_v0: pl.Scalar[pl.INDEX],
        idx_kv_cache__rv_v2: pl.Tensor[[IDX_CACHE_BLOCK_NUM_DYN, INDEXER_PAGE_BYTES_DYN], pl.INT8, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 0)],
        pair_arena_inline1088_inline2392__ssa_v0: pl.Out[pl.Tensor[[24576, 1024], pl.FP32, pl.MemRef("mem_ddr_7", pl.const(0, pl.INT64), 100663296)]],
        __gm_pipe_buffer: pl.Out[pl.Tensor[[1], pl.FP32, pl.MemRef("mem_ddr_8", pl.const(0, pl.INT64), 4)]],
    ):
        pl.func_attr({"slot_num": 1, "mx_tensor_views_blocked": True, "split_aiv": True})
        mem_mat_9: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 8192)
        mem_mat_10: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 2048)
        mem_mat_11: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 49152)
        mem_acc_24: pl.Ptr = pl.tile.alloc(pl.Mem.Acc, 98304)
        mem_left_25: pl.Ptr = pl.tile.alloc(pl.Mem.Left, 4096)
        mem_right_26: pl.Ptr = pl.tile.alloc(pl.Mem.Right, 24576)
        mem_left_27: pl.Ptr = pl.tile.alloc(pl.Mem.Left, 4096)
        mem_right_28: pl.Ptr = pl.tile.alloc(pl.Mem.Right, 24576)
        indexer_score_topk_leaf_v2c_slot_buffer: pl.Scalar[pl.INT32] = pl.system.reserve_buffer(name="indexer_score_topk_leaf_v2c_slot_buffer", size=98304, base=0)
        indexer_score_topk_leaf_c2v_slot_buffer_import: pl.Scalar[pl.INT32] = pl.system.import_peer_buffer(name="indexer_score_topk_leaf_c2v_slot_buffer", peer_func="indexer_score_topk_leaf_aiv")
        pl.system.aic_initialize_pipe(indexer_score_topk_leaf_c2v_slot_buffer_import, indexer_score_topk_leaf_v2c_slot_buffer, dir_mask=3, slot_size=98304, slot_num=1)
        worker_inline1086_inline2428__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        query_count_inline1120_inline2383__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(position_ids__ssa_v0, 0)
        for batch_inline1072_inline2432__idx_v0, (max_cache_len_inline1100_inline2431__iter_v1,) in pl.range(query_count_inline1120_inline2383__ssa_v0 // 6, init_values=(0,)):
            t__tile: pl.Scalar[pl.INT32] = pl.tensor.read(kv_seq_lens__ssa_v0, [batch_inline1072_inline2432__idx_v0])
            batch_cache_len_inline1076_inline2433__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(t__tile, pl.INDEX) // 4
            max_cache_len_inline1100_inline2431__ssa_v3: pl.Scalar[pl.INDEX] = pl.max(max_cache_len_inline1100_inline2431__iter_v1, batch_cache_len_inline1076_inline2433__ssa_v0)
            max_cache_len_inline1100_inline2431__rv_v2: pl.Scalar[pl.INDEX] = pl.yield_(max_cache_len_inline1100_inline2431__ssa_v3)
        max_leaves_inline1092_inline2441__ssa_v0: pl.Scalar[pl.INDEX] = pl.max((pl.min(max_cache_len_inline1100_inline2431__rv_v2, 262144) + 8191) // 8192, 1)
        single_leaf_inline1069_inline2434__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(max_leaves_inline1092_inline2441__ssa_v0 == 1, pl.INDEX)
        for item_inline1114_inline2435__idx_v0, (score_arena_inline1090_inline2379__iter_v1,) in pl.range(
            worker_inline1086_inline2428__ssa_v0, query_count_inline1120_inline2383__ssa_v0 * max_leaves_inline1092_inline2441__ssa_v0, 24, init_values=(score_arena_inline1090_inline2379__ssa_v0,)
        ):
            query_inline1077_inline2438__ssa_v0: pl.Scalar[pl.INDEX] = item_inline1114_inline2435__idx_v0 // max_leaves_inline1092_inline2441__ssa_v0
            leaf_inline1111_inline2418__ssa_v0: pl.Scalar[pl.INDEX] = item_inline1114_inline2435__idx_v0 % max_leaves_inline1092_inline2441__ssa_v0
            batch_idx_inline1093_inline2439__ssa_v0: pl.Scalar[pl.INDEX] = query_inline1077_inline2438__ssa_v0 // 6
            position_inline1073_inline2440__tile: pl.Scalar[pl.INT32] = pl.tensor.read(position_ids__ssa_v0, [query_inline1077_inline2438__ssa_v0])
            t__tile_1: pl.Scalar[pl.INT32] = pl.tensor.read(kv_seq_lens__ssa_v0, [batch_idx_inline1093_inline2439__ssa_v0])
            cache_len_inline1132_inline2426__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(t__tile_1, pl.INDEX) // 4
            cache_bound_inline1110_inline2442__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(cache_len_inline1132_inline2426__ssa_v0, (pl.cast(position_inline1073_inline2440__tile, pl.INDEX) + 1) // 4)
            visible_count_inline1115_inline2437__ssa_v0: pl.Scalar[pl.INDEX] = pl.max(pl.min(cache_bound_inline1110_inline2442__ssa_v0, 262144), 0)
            logical_begin_inline1117_inline2395__ssa_v0: pl.Scalar[pl.INDEX] = leaf_inline1111_inline2418__ssa_v0 * 8192
            if logical_begin_inline1117_inline2395__ssa_v0 < visible_count_inline1115_inline2437__ssa_v0:
                valid_count_inline1119_inline2445__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(visible_count_inline1115_inline2437__ssa_v0 - logical_begin_inline1117_inline2395__ssa_v0, 8192)
                lane_span_inline1122_inline2372__ssa_v0: pl.Scalar[pl.INDEX] = pl.min((valid_count_inline1119_inline2445__ssa_v0 + 383) // 384 * 192, 4096)
                lane_stride_inline1103_inline2376__ssa_v0: pl.Scalar[pl.INDEX] = (
                    single_leaf_inline1069_inline2434__ssa_v0 * 192 + (1 - single_leaf_inline1069_inline2434__ssa_v0) * lane_span_inline1122_inline2372__ssa_v0
                )
                query_head_begin_inline1075_inline2398__ssa_v0: pl.Scalar[pl.INDEX] = query_inline1077_inline2438__ssa_v0 * 64
                query_vector_inline1118_inline2371__tile: pl.Tile[[64, 128], pl.INT8, pl.MemRef(mem_mat_9, pl.const(149504, pl.INT64), 8192), pl.Mem.Mat] = pl.tile.load(
                    qr_hadamard_i8_inline569__rv_v2, [query_head_begin_inline1075_inline2398__ssa_v0, 0], [64, 128], [64, 128], target_memory=pl.Mem.Mat
                )
                coefficient_begin_inline1116_inline2370__ssa_v0: pl.Scalar[pl.INDEX] = query_inline1077_inline2438__ssa_v0 * 16
                coefficients_l1_inline1105_inline2369__tile: pl.Tile[[16, 64], pl.FP16, pl.MemRef(mem_mat_10, pl.const(98304, pl.INT64), 2048), pl.Mem.Mat] = pl.tile.load(
                    coefficients_inline1085_inline2414__rv_v2, [coefficient_begin_inline1116_inline2370__ssa_v0, 0], [16, 64], [16, 64], target_memory=pl.Mem.Mat
                )
                for score_begin_inline1124_inline2367__idx_v0, (score_arena_inline1090_inline2379__iter_v3,) in pl.range(
                    0, lane_span_inline1122_inline2372__ssa_v0, 192, init_values=(score_arena_inline1090_inline2379__iter_v1,)
                ):
                    read_begin_inline1068_inline2378__ssa_v0: pl.Scalar[pl.INDEX] = score_begin_inline1124_inline2367__idx_v0 * (single_leaf_inline1069_inline2434__ssa_v0 + 1)
                    kv_i8_inline1089_inline2366__tile: pl.Tile[[384, 128], pl.INT8, pl.MemRef(mem_mat_11, pl.const(100352, pl.INT64), 49152), pl.Mem.Mat] = pl.tile.create(
                        [384, 128], dtype=pl.INT8, target_memory=pl.Mem.Mat, transpose=False
                    )
                    page_begin_inline1126_inline2365__ssa_v0: pl.Scalar[pl.INDEX] = 0
                    lane_page_inline1130_inline2364__ssa_v0: pl.Scalar[pl.INDEX] = page_begin_inline1126_inline2365__ssa_v0
                    safe_page_begin_inline1096_inline2363__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(
                        read_begin_inline1068_inline2378__ssa_v0 + lane_page_inline1130_inline2364__ssa_v0, (valid_count_inline1119_inline2445__ssa_v0 - 1) // 32 * 32
                    )
                    logical_page_inline1098_inline2362__ssa_v0: pl.Scalar[pl.INDEX] = (logical_begin_inline1117_inline2395__ssa_v0 + safe_page_begin_inline1096_inline2363__ssa_v0) // 32
                    t__tile_2: pl.Scalar[pl.INT32] = pl.tensor.read(
                        idx_block_table_flat_inline1138_inline2374__ssa_v0,
                        [batch_idx_inline1093_inline2439__ssa_v0 * table_columns_inline1070_inline2373__ssa_v0 + logical_page_inline1098_inline2362__ssa_v0],
                    )
                    physical_block_inline1107_inline2391__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(t__tile_2, pl.INDEX)
                    for key_row_inline1071_inline2361__idx_v0, (kv_i8_inline1089_inline2366__iter_v1,) in pl.range(32, init_values=(kv_i8_inline1089_inline2366__tile,)):
                        kv_i8_inline1089_inline2366__tile_1: pl.Tile[[384, 128], pl.INT8, pl.MemRef(mem_mat_11, pl.const(100352, pl.INT64), 49152), pl.Mem.Mat] = pl.tile.gather_row(
                            kv_i8_inline1089_inline2366__iter_v1,
                            idx_kv_cache__rv_v2,
                            [page_begin_inline1126_inline2365__ssa_v0 + key_row_inline1071_inline2361__idx_v0, 0],
                            [physical_block_inline1107_inline2391__ssa_v0, key_row_inline1071_inline2361__idx_v0 * 128],
                            [1, 128],
                            transpose=False,
                        )
                        kv_i8_inline1089_inline2366__rv_v2: pl.Tile[[384, 128], pl.INT8, pl.MemRef(mem_mat_11, pl.const(100352, pl.INT64), 49152), pl.Mem.Mat] = pl.yield_(
                            kv_i8_inline1089_inline2366__tile_1
                        )
                    page_begin_inline1126_inline2365__ssa_v1: pl.Scalar[pl.INDEX] = 32
                    lane_page_inline1130_inline2364__ssa_v1: pl.Scalar[pl.INDEX] = page_begin_inline1126_inline2365__ssa_v1
                    safe_page_begin_inline1096_inline2363__ssa_v1: pl.Scalar[pl.INDEX] = pl.min(
                        read_begin_inline1068_inline2378__ssa_v0 + lane_page_inline1130_inline2364__ssa_v1, (valid_count_inline1119_inline2445__ssa_v0 - 1) // 32 * 32
                    )
                    logical_page_inline1098_inline2362__ssa_v1: pl.Scalar[pl.INDEX] = (logical_begin_inline1117_inline2395__ssa_v0 + safe_page_begin_inline1096_inline2363__ssa_v1) // 32
                    t__tile_3: pl.Scalar[pl.INT32] = pl.tensor.read(
                        idx_block_table_flat_inline1138_inline2374__ssa_v0,
                        [batch_idx_inline1093_inline2439__ssa_v0 * table_columns_inline1070_inline2373__ssa_v0 + logical_page_inline1098_inline2362__ssa_v1],
                    )
                    physical_block_inline1107_inline2391__ssa_v1: pl.Scalar[pl.INDEX] = pl.cast(t__tile_3, pl.INDEX)
                    for key_row_inline1071_inline2361__idx_v1, (kv_i8_inline1089_inline2366__iter_v4,) in pl.range(32, init_values=(kv_i8_inline1089_inline2366__rv_v2,)):
                        kv_i8_inline1089_inline2366__tile_2: pl.Tile[[384, 128], pl.INT8, pl.MemRef(mem_mat_11, pl.const(100352, pl.INT64), 49152), pl.Mem.Mat] = pl.tile.gather_row(
                            kv_i8_inline1089_inline2366__iter_v4,
                            idx_kv_cache__rv_v2,
                            [page_begin_inline1126_inline2365__ssa_v1 + key_row_inline1071_inline2361__idx_v1, 0],
                            [physical_block_inline1107_inline2391__ssa_v1, key_row_inline1071_inline2361__idx_v1 * 128],
                            [1, 128],
                            transpose=False,
                        )
                        kv_i8_inline1089_inline2366__rv_v5: pl.Tile[[384, 128], pl.INT8, pl.MemRef(mem_mat_11, pl.const(100352, pl.INT64), 49152), pl.Mem.Mat] = pl.yield_(
                            kv_i8_inline1089_inline2366__tile_2
                        )
                    page_begin_inline1126_inline2365__ssa_v2: pl.Scalar[pl.INDEX] = 64
                    lane_page_inline1130_inline2364__ssa_v2: pl.Scalar[pl.INDEX] = page_begin_inline1126_inline2365__ssa_v2
                    safe_page_begin_inline1096_inline2363__ssa_v2: pl.Scalar[pl.INDEX] = pl.min(
                        read_begin_inline1068_inline2378__ssa_v0 + lane_page_inline1130_inline2364__ssa_v2, (valid_count_inline1119_inline2445__ssa_v0 - 1) // 32 * 32
                    )
                    logical_page_inline1098_inline2362__ssa_v2: pl.Scalar[pl.INDEX] = (logical_begin_inline1117_inline2395__ssa_v0 + safe_page_begin_inline1096_inline2363__ssa_v2) // 32
                    t__tile_4: pl.Scalar[pl.INT32] = pl.tensor.read(
                        idx_block_table_flat_inline1138_inline2374__ssa_v0,
                        [batch_idx_inline1093_inline2439__ssa_v0 * table_columns_inline1070_inline2373__ssa_v0 + logical_page_inline1098_inline2362__ssa_v2],
                    )
                    physical_block_inline1107_inline2391__ssa_v2: pl.Scalar[pl.INDEX] = pl.cast(t__tile_4, pl.INDEX)
                    for key_row_inline1071_inline2361__idx_v2, (kv_i8_inline1089_inline2366__iter_v7,) in pl.range(32, init_values=(kv_i8_inline1089_inline2366__rv_v5,)):
                        kv_i8_inline1089_inline2366__tile_3: pl.Tile[[384, 128], pl.INT8, pl.MemRef(mem_mat_11, pl.const(100352, pl.INT64), 49152), pl.Mem.Mat] = pl.tile.gather_row(
                            kv_i8_inline1089_inline2366__iter_v7,
                            idx_kv_cache__rv_v2,
                            [page_begin_inline1126_inline2365__ssa_v2 + key_row_inline1071_inline2361__idx_v2, 0],
                            [physical_block_inline1107_inline2391__ssa_v2, key_row_inline1071_inline2361__idx_v2 * 128],
                            [1, 128],
                            transpose=False,
                        )
                        kv_i8_inline1089_inline2366__rv_v8: pl.Tile[[384, 128], pl.INT8, pl.MemRef(mem_mat_11, pl.const(100352, pl.INT64), 49152), pl.Mem.Mat] = pl.yield_(
                            kv_i8_inline1089_inline2366__tile_3
                        )
                    page_begin_inline1126_inline2365__ssa_v3: pl.Scalar[pl.INDEX] = 96
                    lane_page_inline1130_inline2364__ssa_v3: pl.Scalar[pl.INDEX] = page_begin_inline1126_inline2365__ssa_v3
                    safe_page_begin_inline1096_inline2363__ssa_v3: pl.Scalar[pl.INDEX] = pl.min(
                        read_begin_inline1068_inline2378__ssa_v0 + lane_page_inline1130_inline2364__ssa_v3, (valid_count_inline1119_inline2445__ssa_v0 - 1) // 32 * 32
                    )
                    logical_page_inline1098_inline2362__ssa_v3: pl.Scalar[pl.INDEX] = (logical_begin_inline1117_inline2395__ssa_v0 + safe_page_begin_inline1096_inline2363__ssa_v3) // 32
                    t__tile_5: pl.Scalar[pl.INT32] = pl.tensor.read(
                        idx_block_table_flat_inline1138_inline2374__ssa_v0,
                        [batch_idx_inline1093_inline2439__ssa_v0 * table_columns_inline1070_inline2373__ssa_v0 + logical_page_inline1098_inline2362__ssa_v3],
                    )
                    physical_block_inline1107_inline2391__ssa_v3: pl.Scalar[pl.INDEX] = pl.cast(t__tile_5, pl.INDEX)
                    for key_row_inline1071_inline2361__idx_v3, (kv_i8_inline1089_inline2366__iter_v10,) in pl.range(32, init_values=(kv_i8_inline1089_inline2366__rv_v8,)):
                        kv_i8_inline1089_inline2366__tile_4: pl.Tile[[384, 128], pl.INT8, pl.MemRef(mem_mat_11, pl.const(100352, pl.INT64), 49152), pl.Mem.Mat] = pl.tile.gather_row(
                            kv_i8_inline1089_inline2366__iter_v10,
                            idx_kv_cache__rv_v2,
                            [page_begin_inline1126_inline2365__ssa_v3 + key_row_inline1071_inline2361__idx_v3, 0],
                            [physical_block_inline1107_inline2391__ssa_v3, key_row_inline1071_inline2361__idx_v3 * 128],
                            [1, 128],
                            transpose=False,
                        )
                        kv_i8_inline1089_inline2366__rv_v11: pl.Tile[[384, 128], pl.INT8, pl.MemRef(mem_mat_11, pl.const(100352, pl.INT64), 49152), pl.Mem.Mat] = pl.yield_(
                            kv_i8_inline1089_inline2366__tile_4
                        )
                    page_begin_inline1126_inline2365__ssa_v4: pl.Scalar[pl.INDEX] = 128
                    lane_page_inline1130_inline2364__ssa_v4: pl.Scalar[pl.INDEX] = page_begin_inline1126_inline2365__ssa_v4
                    safe_page_begin_inline1096_inline2363__ssa_v4: pl.Scalar[pl.INDEX] = pl.min(
                        read_begin_inline1068_inline2378__ssa_v0 + lane_page_inline1130_inline2364__ssa_v4, (valid_count_inline1119_inline2445__ssa_v0 - 1) // 32 * 32
                    )
                    logical_page_inline1098_inline2362__ssa_v4: pl.Scalar[pl.INDEX] = (logical_begin_inline1117_inline2395__ssa_v0 + safe_page_begin_inline1096_inline2363__ssa_v4) // 32
                    t__tile_6: pl.Scalar[pl.INT32] = pl.tensor.read(
                        idx_block_table_flat_inline1138_inline2374__ssa_v0,
                        [batch_idx_inline1093_inline2439__ssa_v0 * table_columns_inline1070_inline2373__ssa_v0 + logical_page_inline1098_inline2362__ssa_v4],
                    )
                    physical_block_inline1107_inline2391__ssa_v4: pl.Scalar[pl.INDEX] = pl.cast(t__tile_6, pl.INDEX)
                    for key_row_inline1071_inline2361__idx_v4, (kv_i8_inline1089_inline2366__iter_v13,) in pl.range(32, init_values=(kv_i8_inline1089_inline2366__rv_v11,)):
                        kv_i8_inline1089_inline2366__tile_5: pl.Tile[[384, 128], pl.INT8, pl.MemRef(mem_mat_11, pl.const(100352, pl.INT64), 49152), pl.Mem.Mat] = pl.tile.gather_row(
                            kv_i8_inline1089_inline2366__iter_v13,
                            idx_kv_cache__rv_v2,
                            [page_begin_inline1126_inline2365__ssa_v4 + key_row_inline1071_inline2361__idx_v4, 0],
                            [physical_block_inline1107_inline2391__ssa_v4, key_row_inline1071_inline2361__idx_v4 * 128],
                            [1, 128],
                            transpose=False,
                        )
                        kv_i8_inline1089_inline2366__rv_v14: pl.Tile[[384, 128], pl.INT8, pl.MemRef(mem_mat_11, pl.const(100352, pl.INT64), 49152), pl.Mem.Mat] = pl.yield_(
                            kv_i8_inline1089_inline2366__tile_5
                        )
                    page_begin_inline1126_inline2365__ssa_v5: pl.Scalar[pl.INDEX] = 160
                    lane_page_inline1130_inline2364__ssa_v5: pl.Scalar[pl.INDEX] = page_begin_inline1126_inline2365__ssa_v5
                    safe_page_begin_inline1096_inline2363__ssa_v5: pl.Scalar[pl.INDEX] = pl.min(
                        read_begin_inline1068_inline2378__ssa_v0 + lane_page_inline1130_inline2364__ssa_v5, (valid_count_inline1119_inline2445__ssa_v0 - 1) // 32 * 32
                    )
                    logical_page_inline1098_inline2362__ssa_v5: pl.Scalar[pl.INDEX] = (logical_begin_inline1117_inline2395__ssa_v0 + safe_page_begin_inline1096_inline2363__ssa_v5) // 32
                    t__tile_7: pl.Scalar[pl.INT32] = pl.tensor.read(
                        idx_block_table_flat_inline1138_inline2374__ssa_v0,
                        [batch_idx_inline1093_inline2439__ssa_v0 * table_columns_inline1070_inline2373__ssa_v0 + logical_page_inline1098_inline2362__ssa_v5],
                    )
                    physical_block_inline1107_inline2391__ssa_v5: pl.Scalar[pl.INDEX] = pl.cast(t__tile_7, pl.INDEX)
                    for key_row_inline1071_inline2361__idx_v5, (kv_i8_inline1089_inline2366__iter_v16,) in pl.range(32, init_values=(kv_i8_inline1089_inline2366__rv_v14,)):
                        kv_i8_inline1089_inline2366__tile_6: pl.Tile[[384, 128], pl.INT8, pl.MemRef(mem_mat_11, pl.const(100352, pl.INT64), 49152), pl.Mem.Mat] = pl.tile.gather_row(
                            kv_i8_inline1089_inline2366__iter_v16,
                            idx_kv_cache__rv_v2,
                            [page_begin_inline1126_inline2365__ssa_v5 + key_row_inline1071_inline2361__idx_v5, 0],
                            [physical_block_inline1107_inline2391__ssa_v5, key_row_inline1071_inline2361__idx_v5 * 128],
                            [1, 128],
                            transpose=False,
                        )
                        kv_i8_inline1089_inline2366__rv_v17: pl.Tile[[384, 128], pl.INT8, pl.MemRef(mem_mat_11, pl.const(100352, pl.INT64), 49152), pl.Mem.Mat] = pl.yield_(
                            kv_i8_inline1089_inline2366__tile_6
                        )
                    page_begin_inline1126_inline2365__ssa_v6: pl.Scalar[pl.INDEX] = 192
                    lane_page_inline1130_inline2364__ssa_v6: pl.Scalar[pl.INDEX] = (
                        page_begin_inline1126_inline2365__ssa_v6 // 192 * lane_stride_inline1103_inline2376__ssa_v0 + page_begin_inline1126_inline2365__ssa_v6 % 192
                    )
                    safe_page_begin_inline1096_inline2363__ssa_v6: pl.Scalar[pl.INDEX] = pl.min(
                        read_begin_inline1068_inline2378__ssa_v0 + lane_page_inline1130_inline2364__ssa_v6, (valid_count_inline1119_inline2445__ssa_v0 - 1) // 32 * 32
                    )
                    logical_page_inline1098_inline2362__ssa_v6: pl.Scalar[pl.INDEX] = (logical_begin_inline1117_inline2395__ssa_v0 + safe_page_begin_inline1096_inline2363__ssa_v6) // 32
                    t__tile_8: pl.Scalar[pl.INT32] = pl.tensor.read(
                        idx_block_table_flat_inline1138_inline2374__ssa_v0,
                        [batch_idx_inline1093_inline2439__ssa_v0 * table_columns_inline1070_inline2373__ssa_v0 + logical_page_inline1098_inline2362__ssa_v6],
                    )
                    physical_block_inline1107_inline2391__ssa_v6: pl.Scalar[pl.INDEX] = pl.cast(t__tile_8, pl.INDEX)
                    for key_row_inline1071_inline2361__idx_v6, (kv_i8_inline1089_inline2366__iter_v19,) in pl.range(32, init_values=(kv_i8_inline1089_inline2366__rv_v17,)):
                        kv_i8_inline1089_inline2366__tile_7: pl.Tile[[384, 128], pl.INT8, pl.MemRef(mem_mat_11, pl.const(100352, pl.INT64), 49152), pl.Mem.Mat] = pl.tile.gather_row(
                            kv_i8_inline1089_inline2366__iter_v19,
                            idx_kv_cache__rv_v2,
                            [page_begin_inline1126_inline2365__ssa_v6 + key_row_inline1071_inline2361__idx_v6, 0],
                            [physical_block_inline1107_inline2391__ssa_v6, key_row_inline1071_inline2361__idx_v6 * 128],
                            [1, 128],
                            transpose=False,
                        )
                        kv_i8_inline1089_inline2366__rv_v20: pl.Tile[[384, 128], pl.INT8, pl.MemRef(mem_mat_11, pl.const(100352, pl.INT64), 49152), pl.Mem.Mat] = pl.yield_(
                            kv_i8_inline1089_inline2366__tile_7
                        )
                    page_begin_inline1126_inline2365__ssa_v7: pl.Scalar[pl.INDEX] = 224
                    lane_page_inline1130_inline2364__ssa_v7: pl.Scalar[pl.INDEX] = (
                        page_begin_inline1126_inline2365__ssa_v7 // 192 * lane_stride_inline1103_inline2376__ssa_v0 + page_begin_inline1126_inline2365__ssa_v7 % 192
                    )
                    safe_page_begin_inline1096_inline2363__ssa_v7: pl.Scalar[pl.INDEX] = pl.min(
                        read_begin_inline1068_inline2378__ssa_v0 + lane_page_inline1130_inline2364__ssa_v7, (valid_count_inline1119_inline2445__ssa_v0 - 1) // 32 * 32
                    )
                    logical_page_inline1098_inline2362__ssa_v7: pl.Scalar[pl.INDEX] = (logical_begin_inline1117_inline2395__ssa_v0 + safe_page_begin_inline1096_inline2363__ssa_v7) // 32
                    t__tile_9: pl.Scalar[pl.INT32] = pl.tensor.read(
                        idx_block_table_flat_inline1138_inline2374__ssa_v0,
                        [batch_idx_inline1093_inline2439__ssa_v0 * table_columns_inline1070_inline2373__ssa_v0 + logical_page_inline1098_inline2362__ssa_v7],
                    )
                    physical_block_inline1107_inline2391__ssa_v7: pl.Scalar[pl.INDEX] = pl.cast(t__tile_9, pl.INDEX)
                    for key_row_inline1071_inline2361__idx_v7, (kv_i8_inline1089_inline2366__iter_v22,) in pl.range(32, init_values=(kv_i8_inline1089_inline2366__rv_v20,)):
                        kv_i8_inline1089_inline2366__tile_8: pl.Tile[[384, 128], pl.INT8, pl.MemRef(mem_mat_11, pl.const(100352, pl.INT64), 49152), pl.Mem.Mat] = pl.tile.gather_row(
                            kv_i8_inline1089_inline2366__iter_v22,
                            idx_kv_cache__rv_v2,
                            [page_begin_inline1126_inline2365__ssa_v7 + key_row_inline1071_inline2361__idx_v7, 0],
                            [physical_block_inline1107_inline2391__ssa_v7, key_row_inline1071_inline2361__idx_v7 * 128],
                            [1, 128],
                            transpose=False,
                        )
                        kv_i8_inline1089_inline2366__rv_v23: pl.Tile[[384, 128], pl.INT8, pl.MemRef(mem_mat_11, pl.const(100352, pl.INT64), 49152), pl.Mem.Mat] = pl.yield_(
                            kv_i8_inline1089_inline2366__tile_8
                        )
                    page_begin_inline1126_inline2365__ssa_v8: pl.Scalar[pl.INDEX] = 256
                    lane_page_inline1130_inline2364__ssa_v8: pl.Scalar[pl.INDEX] = (
                        page_begin_inline1126_inline2365__ssa_v8 // 192 * lane_stride_inline1103_inline2376__ssa_v0 + page_begin_inline1126_inline2365__ssa_v8 % 192
                    )
                    safe_page_begin_inline1096_inline2363__ssa_v8: pl.Scalar[pl.INDEX] = pl.min(
                        read_begin_inline1068_inline2378__ssa_v0 + lane_page_inline1130_inline2364__ssa_v8, (valid_count_inline1119_inline2445__ssa_v0 - 1) // 32 * 32
                    )
                    logical_page_inline1098_inline2362__ssa_v8: pl.Scalar[pl.INDEX] = (logical_begin_inline1117_inline2395__ssa_v0 + safe_page_begin_inline1096_inline2363__ssa_v8) // 32
                    t__tile_10: pl.Scalar[pl.INT32] = pl.tensor.read(
                        idx_block_table_flat_inline1138_inline2374__ssa_v0,
                        [batch_idx_inline1093_inline2439__ssa_v0 * table_columns_inline1070_inline2373__ssa_v0 + logical_page_inline1098_inline2362__ssa_v8],
                    )
                    physical_block_inline1107_inline2391__ssa_v8: pl.Scalar[pl.INDEX] = pl.cast(t__tile_10, pl.INDEX)
                    for key_row_inline1071_inline2361__idx_v8, (kv_i8_inline1089_inline2366__iter_v25,) in pl.range(32, init_values=(kv_i8_inline1089_inline2366__rv_v23,)):
                        kv_i8_inline1089_inline2366__tile_9: pl.Tile[[384, 128], pl.INT8, pl.MemRef(mem_mat_11, pl.const(100352, pl.INT64), 49152), pl.Mem.Mat] = pl.tile.gather_row(
                            kv_i8_inline1089_inline2366__iter_v25,
                            idx_kv_cache__rv_v2,
                            [page_begin_inline1126_inline2365__ssa_v8 + key_row_inline1071_inline2361__idx_v8, 0],
                            [physical_block_inline1107_inline2391__ssa_v8, key_row_inline1071_inline2361__idx_v8 * 128],
                            [1, 128],
                            transpose=False,
                        )
                        kv_i8_inline1089_inline2366__rv_v26: pl.Tile[[384, 128], pl.INT8, pl.MemRef(mem_mat_11, pl.const(100352, pl.INT64), 49152), pl.Mem.Mat] = pl.yield_(
                            kv_i8_inline1089_inline2366__tile_9
                        )
                    page_begin_inline1126_inline2365__ssa_v9: pl.Scalar[pl.INDEX] = 288
                    lane_page_inline1130_inline2364__ssa_v9: pl.Scalar[pl.INDEX] = (
                        page_begin_inline1126_inline2365__ssa_v9 // 192 * lane_stride_inline1103_inline2376__ssa_v0 + page_begin_inline1126_inline2365__ssa_v9 % 192
                    )
                    safe_page_begin_inline1096_inline2363__ssa_v9: pl.Scalar[pl.INDEX] = pl.min(
                        read_begin_inline1068_inline2378__ssa_v0 + lane_page_inline1130_inline2364__ssa_v9, (valid_count_inline1119_inline2445__ssa_v0 - 1) // 32 * 32
                    )
                    logical_page_inline1098_inline2362__ssa_v9: pl.Scalar[pl.INDEX] = (logical_begin_inline1117_inline2395__ssa_v0 + safe_page_begin_inline1096_inline2363__ssa_v9) // 32
                    t__tile_11: pl.Scalar[pl.INT32] = pl.tensor.read(
                        idx_block_table_flat_inline1138_inline2374__ssa_v0,
                        [batch_idx_inline1093_inline2439__ssa_v0 * table_columns_inline1070_inline2373__ssa_v0 + logical_page_inline1098_inline2362__ssa_v9],
                    )
                    physical_block_inline1107_inline2391__ssa_v9: pl.Scalar[pl.INDEX] = pl.cast(t__tile_11, pl.INDEX)
                    for key_row_inline1071_inline2361__idx_v9, (kv_i8_inline1089_inline2366__iter_v28,) in pl.range(32, init_values=(kv_i8_inline1089_inline2366__rv_v26,)):
                        kv_i8_inline1089_inline2366__tile_10: pl.Tile[[384, 128], pl.INT8, pl.MemRef(mem_mat_11, pl.const(100352, pl.INT64), 49152), pl.Mem.Mat] = pl.tile.gather_row(
                            kv_i8_inline1089_inline2366__iter_v28,
                            idx_kv_cache__rv_v2,
                            [page_begin_inline1126_inline2365__ssa_v9 + key_row_inline1071_inline2361__idx_v9, 0],
                            [physical_block_inline1107_inline2391__ssa_v9, key_row_inline1071_inline2361__idx_v9 * 128],
                            [1, 128],
                            transpose=False,
                        )
                        kv_i8_inline1089_inline2366__rv_v29: pl.Tile[[384, 128], pl.INT8, pl.MemRef(mem_mat_11, pl.const(100352, pl.INT64), 49152), pl.Mem.Mat] = pl.yield_(
                            kv_i8_inline1089_inline2366__tile_10
                        )
                    page_begin_inline1126_inline2365__ssa_v10: pl.Scalar[pl.INDEX] = 320
                    lane_page_inline1130_inline2364__ssa_v10: pl.Scalar[pl.INDEX] = (
                        page_begin_inline1126_inline2365__ssa_v10 // 192 * lane_stride_inline1103_inline2376__ssa_v0 + page_begin_inline1126_inline2365__ssa_v10 % 192
                    )
                    safe_page_begin_inline1096_inline2363__ssa_v10: pl.Scalar[pl.INDEX] = pl.min(
                        read_begin_inline1068_inline2378__ssa_v0 + lane_page_inline1130_inline2364__ssa_v10, (valid_count_inline1119_inline2445__ssa_v0 - 1) // 32 * 32
                    )
                    logical_page_inline1098_inline2362__ssa_v10: pl.Scalar[pl.INDEX] = (logical_begin_inline1117_inline2395__ssa_v0 + safe_page_begin_inline1096_inline2363__ssa_v10) // 32
                    t__tile_12: pl.Scalar[pl.INT32] = pl.tensor.read(
                        idx_block_table_flat_inline1138_inline2374__ssa_v0,
                        [batch_idx_inline1093_inline2439__ssa_v0 * table_columns_inline1070_inline2373__ssa_v0 + logical_page_inline1098_inline2362__ssa_v10],
                    )
                    physical_block_inline1107_inline2391__ssa_v10: pl.Scalar[pl.INDEX] = pl.cast(t__tile_12, pl.INDEX)
                    for key_row_inline1071_inline2361__idx_v10, (kv_i8_inline1089_inline2366__iter_v31,) in pl.range(32, init_values=(kv_i8_inline1089_inline2366__rv_v29,)):
                        kv_i8_inline1089_inline2366__tile_11: pl.Tile[[384, 128], pl.INT8, pl.MemRef(mem_mat_11, pl.const(100352, pl.INT64), 49152), pl.Mem.Mat] = pl.tile.gather_row(
                            kv_i8_inline1089_inline2366__iter_v31,
                            idx_kv_cache__rv_v2,
                            [page_begin_inline1126_inline2365__ssa_v10 + key_row_inline1071_inline2361__idx_v10, 0],
                            [physical_block_inline1107_inline2391__ssa_v10, key_row_inline1071_inline2361__idx_v10 * 128],
                            [1, 128],
                            transpose=False,
                        )
                        kv_i8_inline1089_inline2366__rv_v32: pl.Tile[[384, 128], pl.INT8, pl.MemRef(mem_mat_11, pl.const(100352, pl.INT64), 49152), pl.Mem.Mat] = pl.yield_(
                            kv_i8_inline1089_inline2366__tile_11
                        )
                    page_begin_inline1126_inline2365__ssa_v11: pl.Scalar[pl.INDEX] = 352
                    lane_page_inline1130_inline2364__ssa_v11: pl.Scalar[pl.INDEX] = (
                        page_begin_inline1126_inline2365__ssa_v11 // 192 * lane_stride_inline1103_inline2376__ssa_v0 + page_begin_inline1126_inline2365__ssa_v11 % 192
                    )
                    safe_page_begin_inline1096_inline2363__ssa_v11: pl.Scalar[pl.INDEX] = pl.min(
                        read_begin_inline1068_inline2378__ssa_v0 + lane_page_inline1130_inline2364__ssa_v11, (valid_count_inline1119_inline2445__ssa_v0 - 1) // 32 * 32
                    )
                    logical_page_inline1098_inline2362__ssa_v11: pl.Scalar[pl.INDEX] = (logical_begin_inline1117_inline2395__ssa_v0 + safe_page_begin_inline1096_inline2363__ssa_v11) // 32
                    t__tile_13: pl.Scalar[pl.INT32] = pl.tensor.read(
                        idx_block_table_flat_inline1138_inline2374__ssa_v0,
                        [batch_idx_inline1093_inline2439__ssa_v0 * table_columns_inline1070_inline2373__ssa_v0 + logical_page_inline1098_inline2362__ssa_v11],
                    )
                    physical_block_inline1107_inline2391__ssa_v11: pl.Scalar[pl.INDEX] = pl.cast(t__tile_13, pl.INDEX)
                    for key_row_inline1071_inline2361__idx_v11, (kv_i8_inline1089_inline2366__iter_v34,) in pl.range(32, init_values=(kv_i8_inline1089_inline2366__rv_v32,)):
                        kv_i8_inline1089_inline2366__tile_12: pl.Tile[[384, 128], pl.INT8, pl.MemRef(mem_mat_11, pl.const(100352, pl.INT64), 49152), pl.Mem.Mat] = pl.tile.gather_row(
                            kv_i8_inline1089_inline2366__iter_v34,
                            idx_kv_cache__rv_v2,
                            [page_begin_inline1126_inline2365__ssa_v11 + key_row_inline1071_inline2361__idx_v11, 0],
                            [physical_block_inline1107_inline2391__ssa_v11, key_row_inline1071_inline2361__idx_v11 * 128],
                            [1, 128],
                            transpose=False,
                        )
                        kv_i8_inline1089_inline2366__rv_v35: pl.Tile[[384, 128], pl.INT8, pl.MemRef(mem_mat_11, pl.const(100352, pl.INT64), 49152), pl.Mem.Mat] = pl.yield_(
                            kv_i8_inline1089_inline2366__tile_12
                        )
                    kv_i8_inline1089_inline2366__rv_v35_t: pl.Tile[
                        [128, 384], pl.INT8, pl.MemRef(mem_mat_11, pl.const(100352, pl.INT64), 49152), pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)
                    ] = pl.tile.transpose_view(kv_i8_inline1089_inline2366__rv_v35)
                    score_i32_inline1131_inline2381__tile_l0_init: pl.Tile[[64, 384], pl.INT32, pl.MemRef(mem_acc_24, pl.const(0, pl.INT64), 98304), pl.Mem.Acc] = pl.tile.create(
                        [64, 384], dtype=pl.INT32, target_memory=pl.Mem.Acc
                    )
                    score_i32_inline1131_inline2381__tile_l0_a: pl.Tile[
                        [64, 64], pl.INT8, pl.MemRef(mem_left_25, pl.const(0, pl.INT64), 4096), pl.Mem.Left, pl.TileView(blayout=pl.TileLayout.row_major)
                    ] = pl.tile.extract(query_vector_inline1118_inline2371__tile, 0, 0, [64, 64], target_memory=pl.Mem.Left)
                    score_i32_inline1131_inline2381__tile_l0_b: pl.Tile[[64, 384], pl.INT8, pl.MemRef(mem_right_26, pl.const(0, pl.INT64), 24576), pl.Mem.Right] = pl.tile.extract(
                        kv_i8_inline1089_inline2366__rv_v35_t, 0, 0, [64, 384], target_memory=pl.Mem.Right
                    )
                    score_i32_inline1131_inline2381__tile_l0_a_1: pl.Tile[
                        [64, 64], pl.INT8, pl.MemRef(mem_left_27, pl.const(4096, pl.INT64), 4096), pl.Mem.Left, pl.TileView(blayout=pl.TileLayout.row_major)
                    ] = pl.tile.extract(query_vector_inline1118_inline2371__tile, 0, 64, [64, 64], target_memory=pl.Mem.Left)
                    score_i32_inline1131_inline2381__tile_l0_b_1: pl.Tile[[64, 384], pl.INT8, pl.MemRef(mem_right_28, pl.const(24576, pl.INT64), 24576), pl.Mem.Right] = pl.tile.extract(
                        kv_i8_inline1089_inline2366__rv_v35_t, 64, 0, [64, 384], target_memory=pl.Mem.Right
                    )
                    score_i32_inline1131_inline2381__tile_l0_c_acc: pl.Tile[[64, 384], pl.INT32, pl.MemRef(mem_acc_24, pl.const(0, pl.INT64), 98304), pl.Mem.Acc] = pl.tile.matmul_acc(
                        score_i32_inline1131_inline2381__tile_l0_init, score_i32_inline1131_inline2381__tile_l0_a, score_i32_inline1131_inline2381__tile_l0_b, True
                    )
                    score_i32_inline1131_inline2381__tile_l0_c_acc_1: pl.Tile[[64, 384], pl.INT32, pl.MemRef(mem_acc_24, pl.const(0, pl.INT64), 98304), pl.Mem.Acc] = pl.tile.matmul_acc(
                        score_i32_inline1131_inline2381__tile_l0_c_acc, score_i32_inline1131_inline2381__tile_l0_a_1, score_i32_inline1131_inline2381__tile_l0_b_1, False
                    )
                    pl.tile.tpush_to_aiv(score_i32_inline1131_inline2381__tile_l0_c_acc_1, split=2)
                    scores_l1_inline1112_inline2358__tile: pl.Tile[[64, 384], pl.FP16, pl.Mem.Mat] = pl.tile.tpop_from_aiv(split=2)
                    weighted_scores_inline1121_inline2357__tile_l0_init: pl.Tile[[16, 384], pl.FP32, pl.MemRef(mem_acc_24, pl.const(0, pl.INT64), 24576), pl.Mem.Acc] = pl.tile.create(
                        [16, 384], dtype=pl.FP32, target_memory=pl.Mem.Acc
                    )
                    weighted_scores_inline1121_inline2357__tile_l0_a: pl.Tile[
                        [16, 32], pl.FP16, pl.MemRef(mem_left_25, pl.const(0, pl.INT64), 1024), pl.Mem.Left, pl.TileView(blayout=pl.TileLayout.row_major)
                    ] = pl.tile.extract(coefficients_l1_inline1105_inline2369__tile, 0, 0, [16, 32], target_memory=pl.Mem.Left)
                    weighted_scores_inline1121_inline2357__tile_l0_b: pl.Tile[[32, 384], pl.FP16, pl.MemRef(mem_right_26, pl.const(0, pl.INT64), 24576), pl.Mem.Right] = pl.tile.extract(
                        scores_l1_inline1112_inline2358__tile, 0, 0, [32, 384], target_memory=pl.Mem.Right
                    )
                    weighted_scores_inline1121_inline2357__tile_l0_a_1: pl.Tile[
                        [16, 32], pl.FP16, pl.MemRef(mem_left_27, pl.const(4096, pl.INT64), 1024), pl.Mem.Left, pl.TileView(blayout=pl.TileLayout.row_major)
                    ] = pl.tile.extract(coefficients_l1_inline1105_inline2369__tile, 0, 32, [16, 32], target_memory=pl.Mem.Left)
                    weighted_scores_inline1121_inline2357__tile_l0_b_1: pl.Tile[[32, 384], pl.FP16, pl.MemRef(mem_right_28, pl.const(24576, pl.INT64), 24576), pl.Mem.Right] = pl.tile.extract(
                        scores_l1_inline1112_inline2358__tile, 32, 0, [32, 384], target_memory=pl.Mem.Right
                    )
                    weighted_scores_inline1121_inline2357__tile_l0_c_acc: pl.Tile[[16, 384], pl.FP32, pl.MemRef(mem_acc_24, pl.const(0, pl.INT64), 24576), pl.Mem.Acc] = pl.tile.matmul_acc(
                        weighted_scores_inline1121_inline2357__tile_l0_init, weighted_scores_inline1121_inline2357__tile_l0_a, weighted_scores_inline1121_inline2357__tile_l0_b, True
                    )
                    weighted_scores_inline1121_inline2357__tile_l0_c_acc_1: pl.Tile[[16, 384], pl.FP32, pl.MemRef(mem_acc_24, pl.const(0, pl.INT64), 24576), pl.Mem.Acc] = pl.tile.matmul_acc(
                        weighted_scores_inline1121_inline2357__tile_l0_c_acc, weighted_scores_inline1121_inline2357__tile_l0_a_1, weighted_scores_inline1121_inline2357__tile_l0_b_1, False
                    )
                    pl.system.tfree_to_aiv(scores_l1_inline1112_inline2358__tile, split=2)
                    pl.tile.tpush_to_aiv(weighted_scores_inline1121_inline2357__tile_l0_c_acc_1, split=2)
                    score_arena_inline1090_inline2379__rv_v4: pl.Tensor[[384, 8192], pl.FP32, pl.MemRef("mem_ddr_36", pl.const(0, pl.INT64), 12582912)] = pl.yield_(
                        score_arena_inline1090_inline2379__iter_v3
                    )
                score_arena_inline1090_inline2379__phi_v7: pl.Tensor[[384, 8192], pl.FP32, pl.MemRef("mem_ddr_37", pl.const(0, pl.INT64), 12582912)] = pl.yield_(
                    score_arena_inline1090_inline2379__rv_v4
                )
            else:
                score_arena_inline1090_inline2379__phi_v7: pl.Tensor[[384, 8192], pl.FP32, pl.MemRef("mem_ddr_37", pl.const(0, pl.INT64), 12582912)] = pl.yield_(
                    score_arena_inline1090_inline2379__iter_v1
                )
            score_arena_inline1090_inline2379__rv_v2: pl.Tensor[[384, 8192], pl.FP32, pl.MemRef("mem_ddr_38", pl.const(0, pl.INT64), 12582912)] = pl.yield_(score_arena_inline1090_inline2379__phi_v7)

    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def indexer_score_topk_leaf_aiv(
        position_ids__ssa_v0: pl.Tensor[[T_DYN], pl.INT32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)],
        kv_seq_lens__ssa_v0: pl.Tensor[[B_DYN], pl.INT32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)],
        score_arena_inline1090_inline2379__ssa_v0: pl.InOut[pl.Tensor[[384, 8192], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 12582912)]],
        qr_hadamard_i8_inline569__rv_v2: pl.Tensor[[24576, 128], pl.INT8, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 3145728)],
        coefficients_inline1085_inline2414__rv_v2: pl.Tensor[[6144, 64], pl.FP16, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 786432)],
        idx_block_table_flat_inline1138_inline2374__ssa_v0: pl.Tensor[[idx_table_len_inline1095_inline2409__ssa_v0], pl.INT32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 0)],
        table_columns_inline1070_inline2373__ssa_v0: pl.Scalar[pl.INDEX],
        idx_kv_cache__rv_v2: pl.Tensor[[IDX_CACHE_BLOCK_NUM_DYN, INDEXER_PAGE_BYTES_DYN], pl.INT8, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 0)],
        pair_arena_inline1088_inline2392__ssa_v0: pl.Out[pl.Tensor[[24576, 1024], pl.FP32, pl.MemRef("mem_ddr_7", pl.const(0, pl.INT64), 100663296)]],
        __gm_pipe_buffer: pl.Out[pl.Tensor[[1], pl.FP32, pl.MemRef("mem_ddr_8", pl.const(0, pl.INT64), 4)]],
    ) -> tuple[pl.Tensor[[384, 8192], pl.FP32], pl.Tensor[[24576, 1024], pl.FP32]]:
        pl.func_attr({"slot_num": 1, "mx_tensor_views_blocked": True, "split_aiv": True, "dual_aiv_dispatch": True})
        mem_vec_9: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 49152)
        mem_vec_55: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32768)
        indexer_score_topk_leaf_v2c_slot_buffer_import: pl.Scalar[pl.INT32] = pl.system.import_peer_buffer(name="indexer_score_topk_leaf_v2c_slot_buffer", peer_func="indexer_score_topk_leaf_aic")
        indexer_score_topk_leaf_c2v_slot_buffer: pl.Scalar[pl.INT32] = pl.system.reserve_buffer(name="indexer_score_topk_leaf_c2v_slot_buffer", size=98304, base=0)
        pl.system.aiv_initialize_pipe(indexer_score_topk_leaf_c2v_slot_buffer, indexer_score_topk_leaf_v2c_slot_buffer_import, dir_mask=3, slot_size=98304, slot_num=1)
        worker_inline1086_inline2428__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        query_count_inline1120_inline2383__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(position_ids__ssa_v0, 0)
        for batch_inline1072_inline2432__idx_v0, (max_cache_len_inline1100_inline2431__iter_v1,) in pl.range(query_count_inline1120_inline2383__ssa_v0 // 6, init_values=(0,)):
            t__tile: pl.Scalar[pl.INT32] = pl.tensor.read(kv_seq_lens__ssa_v0, [batch_inline1072_inline2432__idx_v0])
            batch_cache_len_inline1076_inline2433__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(t__tile, pl.INDEX) // 4
            max_cache_len_inline1100_inline2431__ssa_v3: pl.Scalar[pl.INDEX] = pl.max(max_cache_len_inline1100_inline2431__iter_v1, batch_cache_len_inline1076_inline2433__ssa_v0)
            max_cache_len_inline1100_inline2431__rv_v2: pl.Scalar[pl.INDEX] = pl.yield_(max_cache_len_inline1100_inline2431__ssa_v3)
        max_leaves_inline1092_inline2441__ssa_v0: pl.Scalar[pl.INDEX] = pl.max((pl.min(max_cache_len_inline1100_inline2431__rv_v2, 262144) + 8191) // 8192, 1)
        single_leaf_inline1069_inline2434__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(max_leaves_inline1092_inline2441__ssa_v0 == 1, pl.INDEX)
        for item_inline1114_inline2435__idx_v0, (score_arena_inline1090_inline2379__iter_v1,) in pl.range(
            worker_inline1086_inline2428__ssa_v0, query_count_inline1120_inline2383__ssa_v0 * max_leaves_inline1092_inline2441__ssa_v0, 24, init_values=(score_arena_inline1090_inline2379__ssa_v0,)
        ):
            query_inline1077_inline2438__ssa_v0: pl.Scalar[pl.INDEX] = item_inline1114_inline2435__idx_v0 // max_leaves_inline1092_inline2441__ssa_v0
            leaf_inline1111_inline2418__ssa_v0: pl.Scalar[pl.INDEX] = item_inline1114_inline2435__idx_v0 % max_leaves_inline1092_inline2441__ssa_v0
            batch_idx_inline1093_inline2439__ssa_v0: pl.Scalar[pl.INDEX] = query_inline1077_inline2438__ssa_v0 // 6
            position_inline1073_inline2440__tile: pl.Scalar[pl.INT32] = pl.tensor.read(position_ids__ssa_v0, [query_inline1077_inline2438__ssa_v0])
            t__tile_1: pl.Scalar[pl.INT32] = pl.tensor.read(kv_seq_lens__ssa_v0, [batch_idx_inline1093_inline2439__ssa_v0])
            cache_len_inline1132_inline2426__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(t__tile_1, pl.INDEX) // 4
            cache_bound_inline1110_inline2442__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(cache_len_inline1132_inline2426__ssa_v0, (pl.cast(position_inline1073_inline2440__tile, pl.INDEX) + 1) // 4)
            visible_count_inline1115_inline2437__ssa_v0: pl.Scalar[pl.INDEX] = pl.max(pl.min(cache_bound_inline1110_inline2442__ssa_v0, 262144), 0)
            logical_begin_inline1117_inline2395__ssa_v0: pl.Scalar[pl.INDEX] = leaf_inline1111_inline2418__ssa_v0 * 8192
            if logical_begin_inline1117_inline2395__ssa_v0 < visible_count_inline1115_inline2437__ssa_v0:
                valid_count_inline1119_inline2445__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(visible_count_inline1115_inline2437__ssa_v0 - logical_begin_inline1117_inline2395__ssa_v0, 8192)
                lane_span_inline1122_inline2372__ssa_v0: pl.Scalar[pl.INDEX] = pl.min((valid_count_inline1119_inline2445__ssa_v0 + 383) // 384 * 192, 4096)
                lane_stride_inline1103_inline2376__ssa_v0: pl.Scalar[pl.INDEX] = (
                    single_leaf_inline1069_inline2434__ssa_v0 * 192 + (1 - single_leaf_inline1069_inline2434__ssa_v0) * lane_span_inline1122_inline2372__ssa_v0
                )
                for score_begin_inline1124_inline2367__idx_v0, (score_arena_inline1090_inline2379__iter_v3,) in pl.range(
                    0, lane_span_inline1122_inline2372__ssa_v0, 192, init_values=(score_arena_inline1090_inline2379__iter_v1,)
                ):
                    read_begin_inline1068_inline2378__ssa_v0: pl.Scalar[pl.INDEX] = score_begin_inline1124_inline2367__idx_v0 * (single_leaf_inline1069_inline2434__ssa_v0 + 1)
                    score_shard_inline1133_inline2429__tile: pl.Tile[[64, 192], pl.INT32, pl.Mem.Vec] = pl.tile.tpop_from_aic(split=2)
                    t__tile_2: pl.Tile[[64, 192], pl.FP32, pl.MemRef(mem_vec_9, pl.const(131072, pl.INT64), 49152), pl.Mem.Vec] = pl.tile.cast(
                        score_shard_inline1133_inline2429__tile, target_type=pl.FP32, mode="round"
                    )
                    pl.system.tfree_to_aic(score_shard_inline1133_inline2429__tile, split=2)
                    score_fp32_inline1135_inline2384__tile: pl.Tile[[64, 192], pl.FP32, pl.MemRef(mem_vec_9, pl.const(131072, pl.INT64), 49152), pl.Mem.Vec] = pl.tile.maximums(t__tile_2, 0.0)
                    t__tile_3: pl.Tile[[64, 192], pl.FP32, pl.MemRef(mem_vec_9, pl.const(131072, pl.INT64), 49152), pl.Mem.Vec] = pl.tile.muls(score_fp32_inline1135_inline2384__tile, 0.0009765625)
                    score_half_inline1136_inline2359__tile: pl.Tile[[64, 192], pl.FP16, pl.MemRef(mem_vec_9, pl.const(131072, pl.INT64), 24576), pl.Mem.Vec] = pl.tile.cast(
                        t__tile_3, target_type=pl.FP16, mode="rint"
                    )
                    pl.tile.tpush_to_aic(score_half_inline1136_inline2359__tile, split=2)
                    aiv_id_inline1139_inline2356__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_subblock_idx()
                    lane_begin_inline1067_inline2355__ssa_v0: pl.Scalar[pl.INDEX] = aiv_id_inline1139_inline2356__ssa_v0 * lane_stride_inline1103_inline2376__ssa_v0
                    lane_valid_rows_inline1097_inline2354__ssa_v0: pl.Scalar[pl.INDEX] = pl.max(
                        pl.min(valid_count_inline1119_inline2445__ssa_v0 - read_begin_inline1068_inline2378__ssa_v0 - lane_begin_inline1067_inline2355__ssa_v0, 192), 0
                    )
                    kv_scale_bytes_inline1066_inline2353__tile: pl.Tile[[1, 384], pl.INT8, pl.MemRef(mem_vec_9, pl.const(131840, pl.INT64), 384), pl.Mem.Vec] = pl.tile.create(
                        [1, 384], dtype=pl.INT8, target_memory=pl.Mem.Vec
                    )
                    scale_page_begin_inline1087_inline2351__ssa_v0: pl.Scalar[pl.INDEX] = 0
                    safe_scale_begin_inline1064_inline2368__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(
                        read_begin_inline1068_inline2378__ssa_v0 + lane_begin_inline1067_inline2355__ssa_v0 + scale_page_begin_inline1087_inline2351__ssa_v0,
                        (valid_count_inline1119_inline2445__ssa_v0 - 1) // 32 * 32,
                    )
                    scale_logical_page_inline1128_inline2350__ssa_v0: pl.Scalar[pl.INDEX] = (logical_begin_inline1117_inline2395__ssa_v0 + safe_scale_begin_inline1064_inline2368__ssa_v0) // 32
                    t__tile_4: pl.Scalar[pl.INT32] = pl.tensor.read(
                        idx_block_table_flat_inline1138_inline2374__ssa_v0,
                        [batch_idx_inline1093_inline2439__ssa_v0 * table_columns_inline1070_inline2373__ssa_v0 + scale_logical_page_inline1128_inline2350__ssa_v0],
                    )
                    scale_block_inline1063_inline2349__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(t__tile_4, pl.INDEX)
                    kv_scale_bytes_inline1066_inline2353__tile_1: pl.Tile[[1, 384], pl.INT8, pl.MemRef(mem_vec_9, pl.const(131840, pl.INT64), 384), pl.Mem.Vec] = pl.tile.gather_row(
                        kv_scale_bytes_inline1066_inline2353__tile,
                        idx_kv_cache__rv_v2,
                        [0, scale_page_begin_inline1087_inline2351__ssa_v0 * 2],
                        [scale_block_inline1063_inline2349__ssa_v0, 4096],
                        [1, 64],
                        transpose=False,
                    )
                    scale_page_begin_inline1087_inline2351__ssa_v1: pl.Scalar[pl.INDEX] = 32
                    safe_scale_begin_inline1064_inline2368__ssa_v1: pl.Scalar[pl.INDEX] = pl.min(
                        read_begin_inline1068_inline2378__ssa_v0 + lane_begin_inline1067_inline2355__ssa_v0 + scale_page_begin_inline1087_inline2351__ssa_v1,
                        (valid_count_inline1119_inline2445__ssa_v0 - 1) // 32 * 32,
                    )
                    scale_logical_page_inline1128_inline2350__ssa_v1: pl.Scalar[pl.INDEX] = (logical_begin_inline1117_inline2395__ssa_v0 + safe_scale_begin_inline1064_inline2368__ssa_v1) // 32
                    t__tile_5: pl.Scalar[pl.INT32] = pl.tensor.read(
                        idx_block_table_flat_inline1138_inline2374__ssa_v0,
                        [batch_idx_inline1093_inline2439__ssa_v0 * table_columns_inline1070_inline2373__ssa_v0 + scale_logical_page_inline1128_inline2350__ssa_v1],
                    )
                    scale_block_inline1063_inline2349__ssa_v1: pl.Scalar[pl.INDEX] = pl.cast(t__tile_5, pl.INDEX)
                    kv_scale_bytes_inline1066_inline2353__tile_2: pl.Tile[[1, 384], pl.INT8, pl.MemRef(mem_vec_9, pl.const(131840, pl.INT64), 384), pl.Mem.Vec] = pl.tile.gather_row(
                        kv_scale_bytes_inline1066_inline2353__tile_1,
                        idx_kv_cache__rv_v2,
                        [0, scale_page_begin_inline1087_inline2351__ssa_v1 * 2],
                        [scale_block_inline1063_inline2349__ssa_v1, 4096],
                        [1, 64],
                        transpose=False,
                    )
                    scale_page_begin_inline1087_inline2351__ssa_v2: pl.Scalar[pl.INDEX] = 64
                    safe_scale_begin_inline1064_inline2368__ssa_v2: pl.Scalar[pl.INDEX] = pl.min(
                        read_begin_inline1068_inline2378__ssa_v0 + lane_begin_inline1067_inline2355__ssa_v0 + scale_page_begin_inline1087_inline2351__ssa_v2,
                        (valid_count_inline1119_inline2445__ssa_v0 - 1) // 32 * 32,
                    )
                    scale_logical_page_inline1128_inline2350__ssa_v2: pl.Scalar[pl.INDEX] = (logical_begin_inline1117_inline2395__ssa_v0 + safe_scale_begin_inline1064_inline2368__ssa_v2) // 32
                    t__tile_6: pl.Scalar[pl.INT32] = pl.tensor.read(
                        idx_block_table_flat_inline1138_inline2374__ssa_v0,
                        [batch_idx_inline1093_inline2439__ssa_v0 * table_columns_inline1070_inline2373__ssa_v0 + scale_logical_page_inline1128_inline2350__ssa_v2],
                    )
                    scale_block_inline1063_inline2349__ssa_v2: pl.Scalar[pl.INDEX] = pl.cast(t__tile_6, pl.INDEX)
                    kv_scale_bytes_inline1066_inline2353__tile_3: pl.Tile[[1, 384], pl.INT8, pl.MemRef(mem_vec_9, pl.const(131840, pl.INT64), 384), pl.Mem.Vec] = pl.tile.gather_row(
                        kv_scale_bytes_inline1066_inline2353__tile_2,
                        idx_kv_cache__rv_v2,
                        [0, scale_page_begin_inline1087_inline2351__ssa_v2 * 2],
                        [scale_block_inline1063_inline2349__ssa_v2, 4096],
                        [1, 64],
                        transpose=False,
                    )
                    scale_page_begin_inline1087_inline2351__ssa_v3: pl.Scalar[pl.INDEX] = 96
                    safe_scale_begin_inline1064_inline2368__ssa_v3: pl.Scalar[pl.INDEX] = pl.min(
                        read_begin_inline1068_inline2378__ssa_v0 + lane_begin_inline1067_inline2355__ssa_v0 + scale_page_begin_inline1087_inline2351__ssa_v3,
                        (valid_count_inline1119_inline2445__ssa_v0 - 1) // 32 * 32,
                    )
                    scale_logical_page_inline1128_inline2350__ssa_v3: pl.Scalar[pl.INDEX] = (logical_begin_inline1117_inline2395__ssa_v0 + safe_scale_begin_inline1064_inline2368__ssa_v3) // 32
                    t__tile_7: pl.Scalar[pl.INT32] = pl.tensor.read(
                        idx_block_table_flat_inline1138_inline2374__ssa_v0,
                        [batch_idx_inline1093_inline2439__ssa_v0 * table_columns_inline1070_inline2373__ssa_v0 + scale_logical_page_inline1128_inline2350__ssa_v3],
                    )
                    scale_block_inline1063_inline2349__ssa_v3: pl.Scalar[pl.INDEX] = pl.cast(t__tile_7, pl.INDEX)
                    kv_scale_bytes_inline1066_inline2353__tile_4: pl.Tile[[1, 384], pl.INT8, pl.MemRef(mem_vec_9, pl.const(131840, pl.INT64), 384), pl.Mem.Vec] = pl.tile.gather_row(
                        kv_scale_bytes_inline1066_inline2353__tile_3,
                        idx_kv_cache__rv_v2,
                        [0, scale_page_begin_inline1087_inline2351__ssa_v3 * 2],
                        [scale_block_inline1063_inline2349__ssa_v3, 4096],
                        [1, 64],
                        transpose=False,
                    )
                    scale_page_begin_inline1087_inline2351__ssa_v4: pl.Scalar[pl.INDEX] = 128
                    safe_scale_begin_inline1064_inline2368__ssa_v4: pl.Scalar[pl.INDEX] = pl.min(
                        read_begin_inline1068_inline2378__ssa_v0 + lane_begin_inline1067_inline2355__ssa_v0 + scale_page_begin_inline1087_inline2351__ssa_v4,
                        (valid_count_inline1119_inline2445__ssa_v0 - 1) // 32 * 32,
                    )
                    scale_logical_page_inline1128_inline2350__ssa_v4: pl.Scalar[pl.INDEX] = (logical_begin_inline1117_inline2395__ssa_v0 + safe_scale_begin_inline1064_inline2368__ssa_v4) // 32
                    t__tile_8: pl.Scalar[pl.INT32] = pl.tensor.read(
                        idx_block_table_flat_inline1138_inline2374__ssa_v0,
                        [batch_idx_inline1093_inline2439__ssa_v0 * table_columns_inline1070_inline2373__ssa_v0 + scale_logical_page_inline1128_inline2350__ssa_v4],
                    )
                    scale_block_inline1063_inline2349__ssa_v4: pl.Scalar[pl.INDEX] = pl.cast(t__tile_8, pl.INDEX)
                    kv_scale_bytes_inline1066_inline2353__tile_5: pl.Tile[[1, 384], pl.INT8, pl.MemRef(mem_vec_9, pl.const(131840, pl.INT64), 384), pl.Mem.Vec] = pl.tile.gather_row(
                        kv_scale_bytes_inline1066_inline2353__tile_4,
                        idx_kv_cache__rv_v2,
                        [0, scale_page_begin_inline1087_inline2351__ssa_v4 * 2],
                        [scale_block_inline1063_inline2349__ssa_v4, 4096],
                        [1, 64],
                        transpose=False,
                    )
                    scale_page_begin_inline1087_inline2351__ssa_v5: pl.Scalar[pl.INDEX] = 160
                    safe_scale_begin_inline1064_inline2368__ssa_v5: pl.Scalar[pl.INDEX] = pl.min(
                        read_begin_inline1068_inline2378__ssa_v0 + lane_begin_inline1067_inline2355__ssa_v0 + scale_page_begin_inline1087_inline2351__ssa_v5,
                        (valid_count_inline1119_inline2445__ssa_v0 - 1) // 32 * 32,
                    )
                    scale_logical_page_inline1128_inline2350__ssa_v5: pl.Scalar[pl.INDEX] = (logical_begin_inline1117_inline2395__ssa_v0 + safe_scale_begin_inline1064_inline2368__ssa_v5) // 32
                    t__tile_9: pl.Scalar[pl.INT32] = pl.tensor.read(
                        idx_block_table_flat_inline1138_inline2374__ssa_v0,
                        [batch_idx_inline1093_inline2439__ssa_v0 * table_columns_inline1070_inline2373__ssa_v0 + scale_logical_page_inline1128_inline2350__ssa_v5],
                    )
                    scale_block_inline1063_inline2349__ssa_v5: pl.Scalar[pl.INDEX] = pl.cast(t__tile_9, pl.INDEX)
                    kv_scale_bytes_inline1066_inline2353__tile_6: pl.Tile[[1, 384], pl.INT8, pl.MemRef(mem_vec_9, pl.const(131840, pl.INT64), 384), pl.Mem.Vec] = pl.tile.gather_row(
                        kv_scale_bytes_inline1066_inline2353__tile_5,
                        idx_kv_cache__rv_v2,
                        [0, scale_page_begin_inline1087_inline2351__ssa_v5 * 2],
                        [scale_block_inline1063_inline2349__ssa_v5, 4096],
                        [1, 64],
                        transpose=False,
                    )
                    kv_scale_inline1127_inline2348__tile: pl.Tile[[1, 192], pl.FP16, pl.MemRef(mem_vec_9, pl.const(131840, pl.INT64), 384), pl.Mem.Vec] = pl.tile.reinterpret_view(
                        kv_scale_bytes_inline1066_inline2353__tile_6, dtype=pl.FP16
                    )
                    weighted_shard_inline1134_inline2347__tile: pl.Tile[[16, 192], pl.FP32, pl.Mem.Vec] = pl.tile.tpop_from_aic(split=2)
                    score_row_inline1062_inline2346__tile: pl.Tile[[1, 192], pl.FP32, pl.Mem.Vec] = pl.tile.slice(weighted_shard_inline1134_inline2347__tile, [1, 192], [0, 0])
                    t__tile_10: pl.Tile[[1, 192], pl.FP32, pl.MemRef(mem_vec_9, pl.const(131072, pl.INT64), 768), pl.Mem.Vec] = pl.tile.cast(
                        kv_scale_inline1127_inline2348__tile, target_type=pl.FP32, mode="round"
                    )
                    score_row_v1_inline1101_inline2394__tile: pl.Tile[[1, 192], pl.FP32, pl.MemRef(mem_vec_9, pl.const(131072, pl.INT64), 768), pl.Mem.Vec] = pl.tile.mul(
                        score_row_inline1062_inline2346__tile, t__tile_10
                    )
                    pl.system.tfree_to_aic(weighted_shard_inline1134_inline2347__tile, split=2)
                    score_row_id_inline1061_inline2407__ssa_v0: pl.Scalar[pl.INDEX] = single_leaf_inline1069_inline2434__ssa_v0 * query_inline1077_inline2438__ssa_v0 + (
                        1 - single_leaf_inline1069_inline2434__ssa_v0
                    ) * (worker_inline1086_inline2428__ssa_v0 * 2 + aiv_id_inline1139_inline2356__ssa_v0)
                    score_col_inline1123_inline2344__ssa_v0: pl.Scalar[pl.INDEX] = (
                        single_leaf_inline1069_inline2434__ssa_v0 * (read_begin_inline1068_inline2378__ssa_v0 + lane_begin_inline1067_inline2355__ssa_v0)
                        + (1 - single_leaf_inline1069_inline2434__ssa_v0) * score_begin_inline1124_inline2367__idx_v0
                    )
                    if 0 < lane_valid_rows_inline1097_inline2354__ssa_v0:
                        score_valid_inline1106_inline2343__tile: pl.Tile[
                            [1, 192], pl.FP32, pl.MemRef(mem_vec_9, pl.const(131072, pl.INT64), 768), pl.Mem.Vec, pl.TileView(valid_shape=[1, lane_valid_rows_inline1097_inline2354__ssa_v0])
                        ] = pl.tile.set_validshape(score_row_v1_inline1101_inline2394__tile, 1, lane_valid_rows_inline1097_inline2354__ssa_v0)
                        score_arena_inline1090_inline2379__tile: pl.Tensor[[384, 8192], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 12582912)] = pl.tile.store(
                            score_valid_inline1106_inline2343__tile, [score_row_id_inline1061_inline2407__ssa_v0, score_col_inline1123_inline2344__ssa_v0], score_arena_inline1090_inline2379__iter_v3
                        )
                        score_arena_inline1090_inline2379__phi_v6: pl.Tensor[[384, 8192], pl.FP32, pl.MemRef("mem_ddr_16", pl.const(0, pl.INT64), 12582912)] = pl.yield_(
                            score_arena_inline1090_inline2379__tile
                        )
                    else:
                        score_arena_inline1090_inline2379__phi_v6: pl.Tensor[[384, 8192], pl.FP32, pl.MemRef("mem_ddr_16", pl.const(0, pl.INT64), 12582912)] = pl.yield_(
                            score_arena_inline1090_inline2379__iter_v3
                        )
                    score_arena_inline1090_inline2379__rv_v4: pl.Tensor[[384, 8192], pl.FP32, pl.MemRef("mem_ddr_17", pl.const(0, pl.INT64), 12582912)] = pl.yield_(
                        score_arena_inline1090_inline2379__phi_v6
                    )
                if single_leaf_inline1069_inline2434__ssa_v0 == 0:
                    sort_lane_inline1060_inline2405__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_subblock_idx()
                    half_begin_inline1137_inline2412__ssa_v0: pl.Scalar[pl.INDEX] = (
                        logical_begin_inline1117_inline2395__ssa_v0 + sort_lane_inline1060_inline2405__ssa_v0 * lane_span_inline1122_inline2372__ssa_v0
                    )
                    half_valid_inline1084_inline2416__ssa_v0: pl.Scalar[pl.INDEX] = pl.max(
                        pl.min(valid_count_inline1119_inline2445__ssa_v0 - sort_lane_inline1060_inline2405__ssa_v0 * lane_span_inline1122_inline2372__ssa_v0, lane_span_inline1122_inline2372__ssa_v0),
                        0,
                    )
                    half_slot_inline1059_inline2342__ssa_v0: pl.Scalar[pl.INDEX] = (
                        query_inline1077_inline2438__ssa_v0 * 64 + leaf_inline1111_inline2418__ssa_v0 * 2 + sort_lane_inline1060_inline2405__ssa_v0
                    )
                    if 0 < half_valid_inline1084_inline2416__ssa_v0:
                        logical_begin_i32_inline3030__ssa_v0: pl.Scalar[pl.INT32] = pl.cast(half_begin_inline1137_inline2412__ssa_v0, pl.INT32)
                        if half_valid_inline1084_inline2416__ssa_v0 <= 512:
                            t__ci_tmp_v0: pl.Tile[[1, 192], pl.FP32, pl.MemRef(mem_vec_9, pl.const(133120, pl.INT64), 768), pl.Mem.Vec] = pl.tile.create(
                                [1, 192], dtype=pl.FP32, target_memory=pl.Mem.Vec
                            )
                            t__tmp_v243: pl.Tile[[1, 512], pl.INT32, pl.MemRef(mem_vec_9, pl.const(131072, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.ci(
                                pl.const(0, pl.INT32), [1, 512], tmp=t__ci_tmp_v0, dtype=pl.INT32, descending=False
                            )
                            short_indices_inline3048__ssa_v0: pl.Tile[[1, 512], pl.INT32, pl.MemRef(mem_vec_9, pl.const(135168, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.adds(
                                t__tmp_v243, logical_begin_i32_inline3030__ssa_v0
                            )
                            short_raw_inline3040__ssa_v0: pl.Tile[
                                [1, 512], pl.FP32, pl.MemRef(mem_vec_9, pl.const(131072, pl.INT64), 2048), pl.Mem.Vec, pl.TileView(valid_shape=[1, half_valid_inline1084_inline2416__ssa_v0])
                            ] = pl.tile.load(
                                score_arena_inline1090_inline2379__rv_v4,
                                [worker_inline1086_inline2428__ssa_v0 * 2 + sort_lane_inline1060_inline2405__ssa_v0, 0],
                                [1, 512],
                                [1, half_valid_inline1084_inline2416__ssa_v0],
                                target_memory=pl.Mem.Vec,
                            )
                            short_scores_inline3043__ssa_v0: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_9, pl.const(131072, pl.INT64), 2048), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = (
                                pl.tile.fillpad(short_raw_inline3040__ssa_v0, pad_value=pl.PadValue.min)
                            )
                            short_scores_v1_inline3036__ssa_v0: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_9, pl.const(137216, pl.INT64), 2048), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = (
                                pl.tile.maximums(short_scores_inline3043__ssa_v0, -3.4028234663852886e38)
                            )
                            t__tmp_v244: pl.Tile[[1, 512], pl.UINT32, pl.MemRef(mem_vec_9, pl.const(135168, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.reinterpret_view(
                                short_indices_inline3048__ssa_v0, dtype=pl.UINT32
                            )
                            short_pairs_inline3057__ssa_v0: pl.Tile[[1, 1024], pl.FP32, pl.MemRef(mem_vec_9, pl.const(131072, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = (
                                pl.tile.sort32(short_scores_v1_inline3036__ssa_v0, t__tmp_v244)
                            )
                            short_pairs_v1_inline3034__ssa_v0: pl.Tile[[1, 1024], pl.FP32, pl.MemRef(mem_vec_9, pl.const(135168, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = (
                                pl.tile.mrgsort_format1(short_pairs_inline3057__ssa_v0, pl.const(64, pl.INT32))
                            )
                            short_pairs_v2_inline3044__ssa_v0: pl.Tile[[1, 1024], pl.FP32, pl.MemRef(mem_vec_9, pl.const(131072, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = (
                                pl.tile.mrgsort_format1(short_pairs_v1_inline3034__ssa_v0, pl.const(256, pl.INT32))
                            )
                            pair_arena_inline1088_inline2392__store: pl.Tensor[[24576, 1024], pl.FP32, pl.MemRef("mem_ddr_7", pl.const(0, pl.INT64), 100663296)] = pl.tile.store(
                                short_pairs_v2_inline3044__ssa_v0, [half_slot_inline1059_inline2342__ssa_v0, 0], pair_arena_inline1088_inline2392__ssa_v0
                            )
                        else:
                            if half_valid_inline1084_inline2416__ssa_v0 <= 1024:
                                t__ci_tmp_v1: pl.Tile[[1, 192], pl.FP32, pl.MemRef(mem_vec_9, pl.const(135168, pl.INT64), 768), pl.Mem.Vec] = pl.tile.create(
                                    [1, 192], dtype=pl.FP32, target_memory=pl.Mem.Vec
                                )
                                t__tmp_v245: pl.Tile[[1, 1024], pl.INT32, pl.MemRef(mem_vec_9, pl.const(131072, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.ci(
                                    pl.const(0, pl.INT32), [1, 1024], tmp=t__ci_tmp_v1, dtype=pl.INT32, descending=False
                                )
                                small_indices_inline3033__ssa_v0: pl.Tile[[1, 1024], pl.INT32, pl.MemRef(mem_vec_9, pl.const(139264, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.adds(
                                    t__tmp_v245, logical_begin_i32_inline3030__ssa_v0
                                )
                                small_raw_inline3045__ssa_v0: pl.Tile[
                                    [1, 1024], pl.FP32, pl.MemRef(mem_vec_9, pl.const(131072, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(valid_shape=[1, half_valid_inline1084_inline2416__ssa_v0])
                                ] = pl.tile.load(
                                    score_arena_inline1090_inline2379__rv_v4,
                                    [worker_inline1086_inline2428__ssa_v0 * 2 + sort_lane_inline1060_inline2405__ssa_v0, 0],
                                    [1, 1024],
                                    [1, half_valid_inline1084_inline2416__ssa_v0],
                                    target_memory=pl.Mem.Vec,
                                )
                                small_scores_inline3029__ssa_v0: pl.Tile[[1, 1024], pl.FP32, pl.MemRef(mem_vec_9, pl.const(131072, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = (
                                    pl.tile.fillpad(small_raw_inline3045__ssa_v0, pad_value=pl.PadValue.min)
                                )
                                small_scores_v1_inline3038__ssa_v0: pl.Tile[
                                    [1, 1024], pl.FP32, pl.MemRef(mem_vec_9, pl.const(143360, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)
                                ] = pl.tile.maximums(small_scores_inline3029__ssa_v0, -3.4028234663852886e38)
                                t__tmp_v246: pl.Tile[[1, 1024], pl.UINT32, pl.MemRef(mem_vec_9, pl.const(139264, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.reinterpret_view(
                                    small_indices_inline3033__ssa_v0, dtype=pl.UINT32
                                )
                                small_pairs_inline3047__ssa_v0: pl.Tile[[1, 2048], pl.FP32, pl.MemRef(mem_vec_9, pl.const(131072, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = (
                                    pl.tile.sort32(small_scores_v1_inline3038__ssa_v0, t__tmp_v246)
                                )
                                small_pairs_v1_inline3049__ssa_v0: pl.Tile[[1, 2048], pl.FP32, pl.MemRef(mem_vec_9, pl.const(139264, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = (
                                    pl.tile.mrgsort_format1(small_pairs_inline3047__ssa_v0, pl.const(64, pl.INT32))
                                )
                                small_pairs_v2_inline3050__ssa_v0: pl.Tile[[1, 2048], pl.FP32, pl.MemRef(mem_vec_9, pl.const(131072, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = (
                                    pl.tile.mrgsort_format1(small_pairs_v1_inline3049__ssa_v0, pl.const(256, pl.INT32))
                                )
                                small_left_inline3051__ssa_v0: pl.Tile[[1, 1024], pl.FP32, pl.MemRef(mem_vec_9, pl.const(131072, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = (
                                    pl.tile.slice(small_pairs_v2_inline3050__ssa_v0, [1, 1024], [0, 0])
                                )
                                small_right_inline3053__ssa_v0: pl.Tile[[1, 1024], pl.FP32, pl.MemRef(mem_vec_9, pl.const(135168, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = (
                                    pl.tile.slice(small_pairs_v2_inline3050__ssa_v0, [1, 1024], [0, 1024])
                                )
                                small_tmp_inline3054__ssa_v0: pl.Tile[[1, 2048], pl.FP32, pl.MemRef(mem_vec_9, pl.const(139264, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.create(
                                    [1, 2048], dtype=pl.FP32, target_memory=pl.Mem.Vec
                                )
                                small_merged_inline3041__ssa_v0: pl.Tile[[1, 2048], pl.FP32, pl.MemRef(mem_vec_9, pl.const(147456, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = (
                                    pl.tile.mrgsort_format2(small_left_inline3051__ssa_v0, small_right_inline3053__ssa_v0, small_tmp_inline3054__ssa_v0, exhausted=False)
                                )
                                small_top_inline3059__ssa_v0: pl.Tile[[1, 1024], pl.FP32, pl.MemRef(mem_vec_9, pl.const(147456, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = (
                                    pl.tile.slice(small_merged_inline3041__ssa_v0, [1, 1024], [0, 0])
                                )
                                pair_arena_inline1088_inline2392__store_v0: pl.Tensor[[24576, 1024], pl.FP32, pl.MemRef("mem_ddr_7", pl.const(0, pl.INT64), 100663296)] = pl.tile.store(
                                    small_top_inline3059__ssa_v0, [half_slot_inline1059_inline2342__ssa_v0, 0], pair_arena_inline1088_inline2392__ssa_v0
                                )
                            else:
                                if half_valid_inline1084_inline2416__ssa_v0 <= 2048:
                                    t__ci_tmp_v2: pl.Tile[[1, 192], pl.FP32, pl.MemRef(mem_vec_9, pl.const(139264, pl.INT64), 768), pl.Mem.Vec] = pl.tile.create(
                                        [1, 192], dtype=pl.FP32, target_memory=pl.Mem.Vec
                                    )
                                    t__tmp_v247: pl.Tile[[1, 2048], pl.INT32, pl.MemRef(mem_vec_9, pl.const(131072, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.ci(
                                        pl.const(0, pl.INT32), [1, 2048], tmp=t__ci_tmp_v2, dtype=pl.INT32, descending=False
                                    )
                                    leaf_indices_inline3037__ssa_v0: pl.Tile[[1, 2048], pl.INT32, pl.MemRef(mem_vec_9, pl.const(147456, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.adds(
                                        t__tmp_v247, logical_begin_i32_inline3030__ssa_v0
                                    )
                                    leaf_scores_raw_inline3031__ssa_v0: pl.Tile[
                                        [1, 2048], pl.FP32, pl.MemRef(mem_vec_9, pl.const(131072, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[1, half_valid_inline1084_inline2416__ssa_v0])
                                    ] = pl.tile.load(
                                        score_arena_inline1090_inline2379__rv_v4,
                                        [worker_inline1086_inline2428__ssa_v0 * 2 + sort_lane_inline1060_inline2405__ssa_v0, 0],
                                        [1, 2048],
                                        [1, half_valid_inline1084_inline2416__ssa_v0],
                                        target_memory=pl.Mem.Vec,
                                    )
                                    leaf_scores_inline3052__ssa_v0: pl.Tile[
                                        [1, 2048], pl.FP32, pl.MemRef(mem_vec_9, pl.const(131072, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)
                                    ] = pl.tile.fillpad(leaf_scores_raw_inline3031__ssa_v0, pad_value=pl.PadValue.min)
                                    leaf_scores_v1_inline3046__ssa_v0: pl.Tile[
                                        [1, 2048], pl.FP32, pl.MemRef(mem_vec_9, pl.const(155648, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)
                                    ] = pl.tile.maximums(leaf_scores_inline3052__ssa_v0, -3.4028234663852886e38)
                                    t__tmp_v248: pl.Tile[[1, 2048], pl.UINT32, pl.MemRef(mem_vec_9, pl.const(147456, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.reinterpret_view(
                                        leaf_indices_inline3037__ssa_v0, dtype=pl.UINT32
                                    )
                                    pairs_inline3056__ssa_v0: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_9, pl.const(131072, pl.INT64), 16384), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = (
                                        pl.tile.sort32(leaf_scores_v1_inline3046__ssa_v0, t__tmp_v248)
                                    )
                                    pairs_v1_inline3058__ssa_v0: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_9, pl.const(147456, pl.INT64), 16384), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = (
                                        pl.tile.mrgsort_format1(pairs_inline3056__ssa_v0, pl.const(64, pl.INT32))
                                    )
                                    pairs_v2_inline3060__ssa_v0: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_9, pl.const(131072, pl.INT64), 16384), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = (
                                        pl.tile.mrgsort_format1(pairs_v1_inline3058__ssa_v0, pl.const(256, pl.INT32))
                                    )
                                    pairs_v3_inline3042__ssa_v0: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_9, pl.const(147456, pl.INT64), 16384), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = (
                                        pl.tile.mrgsort_format1(pairs_v2_inline3060__ssa_v0, pl.const(1024, pl.INT32))
                                    )
                                    t__tmp_v249: pl.Tile[[1, 1024], pl.FP32, pl.MemRef(mem_vec_9, pl.const(147456, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(
                                        pairs_v3_inline3042__ssa_v0, [1, 1024], [0, 0]
                                    )
                                    pair_arena_inline1088_inline2392__store_v1: pl.Tensor[[24576, 1024], pl.FP32, pl.MemRef("mem_ddr_7", pl.const(0, pl.INT64), 100663296)] = pl.tile.store(
                                        t__tmp_v249, [half_slot_inline1059_inline2342__ssa_v0, 0], pair_arena_inline1088_inline2392__ssa_v0
                                    )
                                else:
                                    t__ci_tmp_v3: pl.Tile[[1, 192], pl.FP32, pl.MemRef(mem_vec_9, pl.const(147456, pl.INT64), 768), pl.Mem.Vec] = pl.tile.create(
                                        [1, 192], dtype=pl.FP32, target_memory=pl.Mem.Vec
                                    )
                                    t__tmp_v250: pl.Tile[[1, 4096], pl.INT32, pl.MemRef(mem_vec_9, pl.const(131072, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.ci(
                                        pl.const(0, pl.INT32), [1, 4096], tmp=t__ci_tmp_v3, dtype=pl.INT32, descending=False
                                    )
                                    medium_indices_inline3061__ssa_v0: pl.Tile[[1, 4096], pl.INT32, pl.MemRef(mem_vec_9, pl.const(163840, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.adds(
                                        t__tmp_v250, logical_begin_i32_inline3030__ssa_v0
                                    )
                                    medium_scores_raw_inline3055__ssa_v0: pl.Tile[
                                        [1, 4096], pl.FP32, pl.MemRef(mem_vec_9, pl.const(131072, pl.INT64), 16384), pl.Mem.Vec, pl.TileView(valid_shape=[1, half_valid_inline1084_inline2416__ssa_v0])
                                    ] = pl.tile.load(
                                        score_arena_inline1090_inline2379__rv_v4,
                                        [worker_inline1086_inline2428__ssa_v0 * 2 + sort_lane_inline1060_inline2405__ssa_v0, 0],
                                        [1, 4096],
                                        [1, half_valid_inline1084_inline2416__ssa_v0],
                                        target_memory=pl.Mem.Vec,
                                    )
                                    medium_scores_inline3028__ssa_v0: pl.Tile[
                                        [1, 4096], pl.FP32, pl.MemRef(mem_vec_9, pl.const(131072, pl.INT64), 16384), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)
                                    ] = pl.tile.fillpad(medium_scores_raw_inline3055__ssa_v0, pad_value=pl.PadValue.min)
                                    medium_scores_v1_inline3039__ssa_v0: pl.Tile[
                                        [1, 4096], pl.FP32, pl.MemRef(mem_vec_55, pl.const(98304, pl.INT64), 16384), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)
                                    ] = pl.tile.maximums(medium_scores_inline3028__ssa_v0, -3.4028234663852886e38)
                                    t__tmp_v251: pl.Tile[[1, 4096], pl.UINT32, pl.MemRef(mem_vec_9, pl.const(163840, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.reinterpret_view(
                                        medium_indices_inline3061__ssa_v0, dtype=pl.UINT32
                                    )
                                    medium_pairs_inline3025__ssa_v0: pl.Tile[
                                        [1, 8192], pl.FP32, pl.MemRef(mem_vec_9, pl.const(131072, pl.INT64), 32768), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)
                                    ] = pl.tile.sort32(medium_scores_v1_inline3039__ssa_v0, t__tmp_v251)
                                    medium_pairs_v1_inline3027__ssa_v0: pl.Tile[
                                        [1, 8192], pl.FP32, pl.MemRef(mem_vec_55, pl.const(98304, pl.INT64), 32768), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)
                                    ] = pl.tile.mrgsort_format1(medium_pairs_inline3025__ssa_v0, pl.const(64, pl.INT32))
                                    medium_pairs_v2_inline3024__ssa_v0: pl.Tile[
                                        [1, 8192], pl.FP32, pl.MemRef(mem_vec_9, pl.const(131072, pl.INT64), 32768), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)
                                    ] = pl.tile.mrgsort_format1(medium_pairs_v1_inline3027__ssa_v0, pl.const(256, pl.INT32))
                                    medium_pairs_v3_inline3032__ssa_v0: pl.Tile[
                                        [1, 8192], pl.FP32, pl.MemRef(mem_vec_55, pl.const(98304, pl.INT64), 32768), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)
                                    ] = pl.tile.mrgsort_format1(medium_pairs_v2_inline3024__ssa_v0, pl.const(1024, pl.INT32))
                                    medium_left_inline3023__ssa_v0: pl.Tile[
                                        [1, 1024], pl.FP32, pl.MemRef(mem_vec_55, pl.const(98304, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)
                                    ] = pl.tile.slice(medium_pairs_v3_inline3032__ssa_v0, [1, 1024], [0, 0])
                                    medium_right_inline3022__ssa_v0: pl.Tile[
                                        [1, 1024], pl.FP32, pl.MemRef(mem_vec_55, pl.const(114688, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)
                                    ] = pl.tile.slice(medium_pairs_v3_inline3032__ssa_v0, [1, 1024], [0, 4096])
                                    medium_tmp_inline3026__ssa_v0: pl.Tile[[1, 2048], pl.FP32, pl.MemRef(mem_vec_9, pl.const(131072, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.create(
                                        [1, 2048], dtype=pl.FP32, target_memory=pl.Mem.Vec
                                    )
                                    medium_merged_inline3035__ssa_v0: pl.Tile[
                                        [1, 2048], pl.FP32, pl.MemRef(mem_vec_9, pl.const(139264, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)
                                    ] = pl.tile.mrgsort_format2(medium_left_inline3023__ssa_v0, medium_right_inline3022__ssa_v0, medium_tmp_inline3026__ssa_v0, exhausted=False)
                                    t__tmp_v252: pl.Tile[[1, 1024], pl.FP32, pl.MemRef(mem_vec_9, pl.const(139264, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(
                                        medium_merged_inline3035__ssa_v0, [1, 1024], [0, 0]
                                    )
                                    pair_arena_inline1088_inline2392__store_v2: pl.Tensor[[24576, 1024], pl.FP32, pl.MemRef("mem_ddr_7", pl.const(0, pl.INT64), 100663296)] = pl.tile.store(
                                        t__tmp_v252, [half_slot_inline1059_inline2342__ssa_v0, 0], pair_arena_inline1088_inline2392__ssa_v0
                                    )
                    else:
                        empty_pairs_inline1074_inline2341__ssa_v0: pl.Tile[[1, 1024], pl.FP32, pl.MemRef(mem_vec_9, pl.const(131072, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.full(
                            [1, 1024], dtype=pl.FP32, value=-3.4028234663852886e38
                        )
                        pair_arena_inline1088_inline2392__store_v3: pl.Tensor[[24576, 1024], pl.FP32, pl.MemRef("mem_ddr_7", pl.const(0, pl.INT64), 100663296)] = pl.tile.store(
                            empty_pairs_inline1074_inline2341__ssa_v0, [half_slot_inline1059_inline2342__ssa_v0, 0], pair_arena_inline1088_inline2392__ssa_v0
                        )
                score_arena_inline1090_inline2379__phi_v7: pl.Tensor[[384, 8192], pl.FP32, pl.MemRef("mem_ddr_61", pl.const(0, pl.INT64), 12582912)] = pl.yield_(
                    score_arena_inline1090_inline2379__rv_v4
                )
            else:
                score_arena_inline1090_inline2379__phi_v7: pl.Tensor[[384, 8192], pl.FP32, pl.MemRef("mem_ddr_61", pl.const(0, pl.INT64), 12582912)] = pl.yield_(
                    score_arena_inline1090_inline2379__iter_v1
                )
            score_arena_inline1090_inline2379__rv_v2: pl.Tensor[[384, 8192], pl.FP32, pl.MemRef("mem_ddr_62", pl.const(0, pl.INT64), 12582912)] = pl.yield_(score_arena_inline1090_inline2379__phi_v7)
        return score_arena_inline1090_inline2379__ssa_v0, pair_arena_inline1088_inline2392__ssa_v0

    @pl.function(type=pl.FunctionType.Group, level=pl.Level.CORE_GROUP, role=pl.Role.SubWorker)
    def indexer_score_topk_leaf(
        position_ids__ssa_v0: pl.Tensor[[T_DYN], pl.INT32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)],
        kv_seq_lens__ssa_v0: pl.Tensor[[B_DYN], pl.INT32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)],
        score_arena_inline1090_inline2379__ssa_v0: pl.InOut[pl.Tensor[[384, 8192], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 12582912)]],
        qr_hadamard_i8_inline569__rv_v2: pl.Tensor[[24576, 128], pl.INT8, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 3145728)],
        coefficients_inline1085_inline2414__rv_v2: pl.Tensor[[6144, 64], pl.FP16, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 786432)],
        idx_block_table_flat_inline1138_inline2374__ssa_v0: pl.Tensor[[idx_table_len_inline1095_inline2409__ssa_v0], pl.INT32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 0)],
        table_columns_inline1070_inline2373__ssa_v0: pl.Scalar[pl.INDEX],
        idx_kv_cache__rv_v2: pl.Tensor[[IDX_CACHE_BLOCK_NUM_DYN, INDEXER_PAGE_BYTES_DYN], pl.INT8, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 0)],
        pair_arena_inline1088_inline2392__ssa_v0: pl.Out[pl.Tensor[[24576, 1024], pl.FP32, pl.MemRef("mem_ddr_7", pl.const(0, pl.INT64), 100663296)]],
        __gm_pipe_buffer: pl.Out[pl.Tensor[[1], pl.FP32, pl.MemRef("mem_ddr_8", pl.const(0, pl.INT64), 4)]],
    ) -> tuple[pl.Tensor[[384, 8192], pl.FP32], pl.Tensor[[24576, 1024], pl.FP32]]:
        pl.func_attr({"slot_num": 1, "mx_tensor_views_blocked": True, "split_aiv": True})
        self.indexer_score_topk_leaf_aic(
            position_ids__ssa_v0,
            kv_seq_lens__ssa_v0,
            score_arena_inline1090_inline2379__ssa_v0,
            qr_hadamard_i8_inline569__rv_v2,
            coefficients_inline1085_inline2414__rv_v2,
            idx_block_table_flat_inline1138_inline2374__ssa_v0,
            table_columns_inline1070_inline2373__ssa_v0,
            idx_kv_cache__rv_v2,
            pair_arena_inline1088_inline2392__ssa_v0,
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
            score_arena_inline1090_inline2379__ssa_v0,
            qr_hadamard_i8_inline569__rv_v2,
            coefficients_inline1085_inline2414__rv_v2,
            idx_block_table_flat_inline1138_inline2374__ssa_v0,
            table_columns_inline1070_inline2373__ssa_v0,
            idx_kv_cache__rv_v2,
            pair_arena_inline1088_inline2392__ssa_v0,
            __gm_pipe_buffer,
            attrs={"arg_directions": [pl.adir.input, pl.adir.input, pl.adir.inout, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.scalar, pl.adir.input, pl.adir.inout, pl.adir.inout]},
        )
        return score_arena_inline1090_inline2379__ssa_v0, pair_arena_inline1088_inline2392__ssa_v0

    @pl.function(type=pl.FunctionType.Spmd)
    def indexer_score_topk_leaf_spmd(
        self,
        position_ids__ssa_v0: pl.Tensor[[T_DYN], pl.INT32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)],
        kv_seq_lens__ssa_v0: pl.Tensor[[B_DYN], pl.INT32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)],
        score_arena_inline1090_inline2379__ssa_v0: pl.InOut[pl.Tensor[[384, 8192], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 12582912)]],
        qr_hadamard_i8_inline569__rv_v2: pl.Tensor[[24576, 128], pl.INT8, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 3145728)],
        coefficients_inline1085_inline2414__rv_v2: pl.Tensor[[6144, 64], pl.FP16, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 786432)],
        idx_block_table_flat_inline1138_inline2374__ssa_v0: pl.Tensor[[idx_table_len_inline1095_inline2409__ssa_v0], pl.INT32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 0)],
        table_columns_inline1070_inline2373__ssa_v0: pl.Scalar[pl.INDEX],
        idx_kv_cache__rv_v3: pl.Tensor[[IDX_CACHE_BLOCK_NUM_DYN, INDEXER_PAGE_BYTES_DYN], pl.INT8, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 0)],
        pair_arena_inline1088_inline2392__ssa_v0: pl.Out[pl.Tensor[[24576, 1024], pl.FP32, pl.MemRef("mem_ddr_7", pl.const(0, pl.INT64), 100663296)]],
        __gm_pipe_buffer: pl.Out[pl.Tensor[[1], pl.FP32, pl.MemRef("mem_ddr_8", pl.const(0, pl.INT64), 4)]],
    ) -> tuple[pl.Tensor[[384, 8192], pl.FP32], pl.Tensor[[24576, 1024], pl.FP32]]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        ret__tmp_v0: pl.Tuple[pl.Tensor[[384, 8192], pl.FP32], pl.Tensor[[24576, 1024], pl.FP32]] = self.indexer_score_topk_leaf(
            position_ids__ssa_v0,
            kv_seq_lens__ssa_v0,
            score_arena_inline1090_inline2379__ssa_v0,
            qr_hadamard_i8_inline569__rv_v2,
            coefficients_inline1085_inline2414__rv_v2,
            idx_block_table_flat_inline1138_inline2374__ssa_v0,
            table_columns_inline1070_inline2373__ssa_v0,
            idx_kv_cache__rv_v3,
            pair_arena_inline1088_inline2392__ssa_v0,
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
        score_arena_inline1090_inline2379__rv_v2: pl.Tensor[[384, 8192], pl.FP32, pl.MemRef("mem_ddr_9", pl.const(0, pl.INT64), 12582912)] = ret__tmp_v0[0]
        pair_arena_inline1088_inline2392__ssa_v1: pl.Tensor[[24576, 1024], pl.FP32, pl.MemRef("mem_ddr_10", pl.const(0, pl.INT64), 100663296)] = ret__tmp_v0[1]
        return score_arena_inline1090_inline2379__ssa_v0, pair_arena_inline1088_inline2392__ssa_v0

    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def kv_and_cache_write(
        compact_rows_inline2241__ssa_v0: pl.Scalar[pl.INDEX],
        kv_final_inline2230__rv_v2: pl.Tensor[[384, 128], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 196608)],
        idx_kv_scale_values_inline2235__ssa_v0: pl.Out[pl.Tensor[[384, 1], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 1536)]],
        idx_kv_cache__ssa_v0: pl.Out[pl.Tensor[[IDX_CACHE_BLOCK_NUM_DYN, INDEXER_PAGE_BYTES_DYN], pl.INT8, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)]],
        kv_flat_inline2250__ssa_v0: pl.Out[pl.Tensor[[T_DYN, 128], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)]],
        position_ids__ssa_v0: pl.Tensor[[T_DYN], pl.INT32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 0)],
        idx_slot_mapping__ssa_v0: pl.Tensor[[T_DYN], pl.INT64, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 0)],
    ) -> tuple[pl.Tensor[[384, 1], pl.FP32], pl.Tensor[[IDX_CACHE_BLOCK_NUM_DYN, INDEXER_PAGE_BYTES_DYN], pl.INT8]]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        mem_vec_6: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 8192)
        mem_vec_12: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 8192)
        mem_vec_13: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 8192)
        mem_vec_14: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 64)
        wr_blk_inline2244__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        wr_b0_inline2247__ssa_v0: pl.Scalar[pl.INDEX] = wr_blk_inline2244__ssa_v0 * 16
        wr_blk_rows_inline2246__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(compact_rows_inline2241__ssa_v0 - wr_b0_inline2247__ssa_v0, 16)
        t__tile: pl.Tile[[16, 128], pl.FP32, pl.MemRef(mem_vec_6, pl.const(16448, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.load(
            kv_final_inline2230__rv_v2, [wr_b0_inline2247__ssa_v0, 0], [16, 128], [16, 128], target_memory=pl.Mem.Vec
        )
        t__tile_1: pl.Tile[[16, 128], pl.BF16, pl.MemRef(mem_vec_12, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(t__tile, target_type=pl.BF16, mode="rint")
        kv_blk_f32_inline2248__tile: pl.Tile[[16, 128], pl.FP32, pl.MemRef(mem_vec_6, pl.const(16448, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.cast(t__tile_1, target_type=pl.FP32, mode="round")
        t__tile_2: pl.Tile[[16, 128], pl.FP32, pl.MemRef(mem_vec_6, pl.const(16448, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.muls(kv_blk_f32_inline2248__tile, 0.088388347648318447)
        t__tile_3: pl.Tile[[16, 128], pl.BF16, pl.MemRef(mem_vec_12, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(t__tile_2, target_type=pl.BF16, mode="rint")
        kv_blk_f32_v1_inline2251__tile: pl.Tile[[16, 128], pl.FP32, pl.MemRef(mem_vec_6, pl.const(16448, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.cast(t__tile_3, target_type=pl.FP32, mode="round")
        t__tile_4: pl.Tile[[16, 128], pl.FP32, pl.MemRef(mem_vec_12, pl.const(0, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.abs(kv_blk_f32_v1_inline2251__tile)
        tmp_tile: pl.Tile[[16, 128], pl.FP32, pl.MemRef(mem_vec_13, pl.const(8192, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.create([16, 128], dtype=pl.FP32, target_memory=pl.Mem.Vec)
        t__tile_5: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_14, pl.const(16384, pl.INT64), 64), pl.Mem.Vec] = pl.tile.row_max(t__tile_4, tmp_tile)
        kv_amax_inline2223__tile: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_14, pl.const(16384, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(t__tile_5, [1, 16])
        t__tile_6: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_12, pl.const(0, pl.INT64), 64), pl.Mem.Vec] = pl.tile.full([1, 16], dtype=pl.FP32, value=0.0001)
        kv_amax_v1_inline2254__tile: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_12, pl.const(0, pl.INT64), 64), pl.Mem.Vec] = pl.tile.maximum(kv_amax_inline2223__tile, t__tile_6)
        t__tile_7: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_13, pl.const(8192, pl.INT64), 64), pl.Mem.Vec] = pl.tile.full([1, 16], dtype=pl.FP32, value=127.0)
        kv_scale_q_row_inline2255__tile: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_13, pl.const(8192, pl.INT64), 64), pl.Mem.Vec] = pl.tile.div(t__tile_7, kv_amax_v1_inline2254__tile)
        t__tile_8: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_12, pl.const(0, pl.INT64), 64), pl.Mem.Vec] = pl.tile.recip(kv_scale_q_row_inline2255__tile)
        kv_scale_dq_col_inline2256__tile: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_12, pl.const(0, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(t__tile_8, [16, 1])
        kv_scale_q_col_inline2257__tile: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_13, pl.const(8192, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(kv_scale_q_row_inline2255__tile, [16, 1])
        idx_kv_scale_values_inline2235__tile: pl.Tensor[[384, 1], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 1536)] = pl.tile.store(
            kv_scale_dq_col_inline2256__tile, [wr_b0_inline2247__ssa_v0, 0], idx_kv_scale_values_inline2235__ssa_v0
        )
        kv_scaled_inline2243__tile: pl.Tile[[16, 128], pl.FP32, pl.MemRef(mem_vec_12, pl.const(0, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.row_expand_mul(
            kv_blk_f32_v1_inline2251__tile, kv_scale_q_col_inline2257__tile
        )
        kv_i32_inline2253__tile: pl.Tile[[16, 128], pl.INT32, pl.MemRef(mem_vec_12, pl.const(0, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.cast(
            kv_scaled_inline2243__tile, target_type=pl.INT32, mode="rint"
        )
        kv_half_inline2242__tile: pl.Tile[[16, 128], pl.FP16, pl.MemRef(mem_vec_12, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(kv_i32_inline2253__tile, target_type=pl.FP16, mode="round")
        kv_i8_blk_inline2258__tile: pl.Tile[[16, 128], pl.INT8, pl.MemRef(mem_vec_12, pl.const(0, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(
            kv_half_inline2242__tile, target_type=pl.INT8, mode="trunc"
        )
        for inner_inline2249__idx_v0, (idx_kv_cache__iter_v1, kv_flat_inline2250__iter_v1) in pl.range(wr_blk_rows_inline2246__ssa_v0, init_values=(idx_kv_cache__ssa_v0, kv_flat_inline2250__ssa_v0)):
            compact_token_inline2240__ssa_v0: pl.Scalar[pl.INDEX] = wr_b0_inline2247__ssa_v0 + inner_inline2249__idx_v0
            request_inline2220__ssa_v0: pl.Scalar[pl.INDEX] = compact_token_inline2240__ssa_v0 // 2
            first_pos_inline2219__tile: pl.Scalar[pl.INT32] = pl.tensor.read(position_ids__ssa_v0, [request_inline2220__ssa_v0 * 6])
            local_token_inline2217__ssa_v0: pl.Scalar[pl.INDEX] = compact_token_inline2240__ssa_v0 % 2 * 4 - pl.cast(first_pos_inline2219__tile, pl.INDEX) % 4 + 3
            if local_token_inline2217__ssa_v0 < 6:
                token_inline2216__ssa_v0: pl.Scalar[pl.INDEX] = request_inline2220__ssa_v0 * 6 + local_token_inline2217__ssa_v0
                cache_row_i64_inline2215__tile: pl.Scalar[pl.INT64] = pl.tensor.read(idx_slot_mapping__ssa_v0, [token_inline2216__ssa_v0])
                if 0 <= cache_row_i64_inline2215__tile:
                    cache_row_inline2237__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(cache_row_i64_inline2215__tile, pl.INDEX)
                    t__tile_9: pl.Tile[[1, 128], pl.FP32, pl.MemRef(mem_vec_6, pl.const(16448, pl.INT64), 512), pl.Mem.Vec] = pl.tile.slice(
                        kv_blk_f32_v1_inline2251__tile, [1, 128], [inner_inline2249__idx_v0, 0]
                    )
                    kv_flat_inline2250__tile: pl.Tensor[[T_DYN, 128], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                        t__tile_9, [token_inline2216__ssa_v0, 0], kv_flat_inline2250__iter_v1
                    )
                    cache_page_inline2218__ssa_v0: pl.Scalar[pl.INDEX] = cache_row_inline2237__ssa_v0 // 32
                    key_begin_inline2234__ssa_v0: pl.Scalar[pl.INDEX] = cache_row_inline2237__ssa_v0 % 32 * 128
                    t__tile_10: pl.Tile[[1, 128], pl.INT8, pl.MemRef(mem_vec_12, pl.const(0, pl.INT64), 128), pl.Mem.Vec] = pl.tile.slice(
                        kv_i8_blk_inline2258__tile, [1, 128], [inner_inline2249__idx_v0, 0]
                    )
                    idx_kv_cache__tile: pl.Tensor[[IDX_CACHE_BLOCK_NUM_DYN, INDEXER_PAGE_BYTES_DYN], pl.INT8, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                        t__tile_10, [cache_page_inline2218__ssa_v0, key_begin_inline2234__ssa_v0], idx_kv_cache__iter_v1
                    )
                    idx_kv_cache__phi_v4, kv_flat_inline2250__phi_v4 = pl.yield_(idx_kv_cache__tile, kv_flat_inline2250__tile)
                else:
                    idx_kv_cache__phi_v4, kv_flat_inline2250__phi_v4 = pl.yield_(idx_kv_cache__iter_v1, kv_flat_inline2250__iter_v1)
                idx_kv_cache__phi_v5, kv_flat_inline2250__phi_v5 = pl.yield_(idx_kv_cache__phi_v4, kv_flat_inline2250__phi_v4)
            else:
                idx_kv_cache__phi_v5, kv_flat_inline2250__phi_v5 = pl.yield_(idx_kv_cache__iter_v1, kv_flat_inline2250__iter_v1)
            idx_kv_cache__rv_v2, kv_flat_inline2250__rv_v2 = pl.yield_(idx_kv_cache__phi_v5, kv_flat_inline2250__phi_v5)
        return idx_kv_scale_values_inline2235__ssa_v0, idx_kv_cache__ssa_v0

    @pl.function(type=pl.FunctionType.Spmd)
    def kv_and_cache_write_spmd(
        self,
        compact_rows_inline2241__ssa_v0: pl.Scalar[pl.INDEX],
        kv_final_inline2230__rv_v2: pl.Tensor[[384, 128], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 196608)],
        idx_kv_scale_values_inline2235__ssa_v0: pl.Out[pl.Tensor[[384, 1], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 1536)]],
        idx_kv_cache__ssa_v0: pl.Out[pl.Tensor[[IDX_CACHE_BLOCK_NUM_DYN, INDEXER_PAGE_BYTES_DYN], pl.INT8, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)]],
        kv_flat_inline2250__ssa_v0: pl.Out[pl.Tensor[[T_DYN, 128], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)]],
        position_ids__ssa_v0: pl.Tensor[[T_DYN], pl.INT32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 0)],
        idx_slot_mapping__ssa_v0: pl.Tensor[[T_DYN], pl.INT64, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 0)],
    ) -> tuple[pl.Tensor[[384, 1], pl.FP32], pl.Tensor[[IDX_CACHE_BLOCK_NUM_DYN, INDEXER_PAGE_BYTES_DYN], pl.INT8]]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        ret__tmp_v0: pl.Tuple[pl.Tensor[[384, 1], pl.FP32], pl.Tensor[[IDX_CACHE_BLOCK_NUM_DYN, INDEXER_PAGE_BYTES_DYN], pl.INT8]] = self.kv_and_cache_write(
            compact_rows_inline2241__ssa_v0,
            kv_final_inline2230__rv_v2,
            idx_kv_scale_values_inline2235__ssa_v0,
            idx_kv_cache__ssa_v0,
            kv_flat_inline2250__ssa_v0,
            position_ids__ssa_v0,
            idx_slot_mapping__ssa_v0,
            attrs={"arg_directions": [pl.adir.scalar, pl.adir.input, pl.adir.output_existing, pl.adir.output_existing, pl.adir.output_existing, pl.adir.input, pl.adir.input]},
        )
        idx_kv_scale_values_inline2235__ssa_v1: pl.Tensor[[384, 1], pl.FP32, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 1536)] = ret__tmp_v0[0]
        idx_kv_cache__rv_v2: pl.Tensor[[IDX_CACHE_BLOCK_NUM_DYN, INDEXER_PAGE_BYTES_DYN], pl.INT8, pl.MemRef("mem_ddr_7", pl.const(0, pl.INT64), 0)] = ret__tmp_v0[1]
        return idx_kv_scale_values_inline2235__ssa_v0, idx_kv_cache__ssa_v0

    @pl.function(type=pl.FunctionType.AIC, level=pl.Level.AIC, role=pl.Role.SubWorker)
    def kv_hadamard(
        kv_final_inline2230__ssa_v0: pl.Out[pl.Tensor[[384, 128], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 196608)]],
        hadamard_idx__ssa_v0: pl.Tensor[[128, 128], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 32768)],
        rms_blocks_inline2239__ssa_v0: pl.Scalar[pl.INDEX],
        compact_rows_inline2241__ssa_v0: pl.Scalar[pl.INDEX],
        normed_kv_inline560__rv_v3: pl.Tensor[[384, 128], pl.BF16, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 98304)],
    ) -> pl.Tensor[[384, 128], pl.FP32]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        mem_mat_3: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 16384)
        mem_mat_4: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 4096)
        mem_left_5: pl.Ptr = pl.tile.alloc(pl.Mem.Left, 4096)
        mem_right_6: pl.Ptr = pl.tile.alloc(pl.Mem.Right, 16384)
        mem_acc_7: pl.Ptr = pl.tile.alloc(pl.Mem.Acc, 4096)
        for o0_inline2224__idx_v0, (kv_final_inline2230__iter_v1,) in pl.range(0, 128, 64, init_values=(kv_final_inline2230__ssa_v0,)):
            hadamard_tile_inline2231__tile: pl.Tile[[128, 64], pl.BF16, pl.MemRef(mem_mat_3, pl.const(0, pl.INT64), 16384), pl.Mem.Mat] = pl.tile.load(
                hadamard_idx__ssa_v0, [0, o0_inline2224__idx_v0], [128, 64], [128, 64], target_memory=pl.Mem.Mat
            )
            for had_blk_inline2227__idx_v0, (kv_final_inline2230__iter_v3,) in pl.range(rms_blocks_inline2239__ssa_v0, init_values=(kv_final_inline2230__iter_v1,)):
                had_b0_inline2236__ssa_v0: pl.Scalar[pl.INDEX] = had_blk_inline2227__idx_v0 * 16
                had_rows_inline2222__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(compact_rows_inline2241__ssa_v0 - had_b0_inline2236__ssa_v0, 16)
                kv_proj_tile_inline2229__tile: pl.Tile[
                    [16, 128], pl.BF16, pl.MemRef(mem_mat_4, pl.const(16384, pl.INT64), 4096), pl.Mem.Mat, pl.TileView(valid_shape=[had_rows_inline2222__ssa_v0, 128])
                ] = pl.tile.load(normed_kv_inline560__rv_v3, [had_b0_inline2236__ssa_v0, 0], [16, 128], [had_rows_inline2222__ssa_v0, 128], target_memory=pl.Mem.Mat)
                kv_proj_tile_inline2229__tile_Left: pl.Tile[
                    [16, 128], pl.BF16, pl.MemRef(mem_left_5, pl.const(0, pl.INT64), 4096), pl.Mem.Left, pl.TileView(valid_shape=[had_rows_inline2222__ssa_v0, 128])
                ] = pl.tile.move(kv_proj_tile_inline2229__tile, target_memory=pl.Mem.Left)
                hadamard_tile_inline2231__tile_Right: pl.Tile[[128, 64], pl.BF16, pl.MemRef(mem_right_6, pl.const(0, pl.INT64), 16384), pl.Mem.Right] = pl.tile.move(
                    hadamard_tile_inline2231__tile, target_memory=pl.Mem.Right
                )
                kv_hadamard_acc_inline2245__tile: pl.Tile[
                    [16, 64], pl.FP32, pl.MemRef(mem_acc_7, pl.const(0, pl.INT64), 4096), pl.Mem.Acc, pl.TileView(valid_shape=[had_rows_inline2222__ssa_v0, 64], compact=pl.CompactMode.normal)
                ] = pl.tile.matmul(kv_proj_tile_inline2229__tile_Left, hadamard_tile_inline2231__tile_Right)
                kv_final_inline2230__tile: pl.Tensor[[384, 128], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 196608)] = pl.tile.store(
                    kv_hadamard_acc_inline2245__tile, [had_b0_inline2236__ssa_v0, o0_inline2224__idx_v0], kv_final_inline2230__iter_v3
                )
                kv_final_inline2230__rv_v4: pl.Tensor[[384, 128], pl.FP32, pl.MemRef("mem_ddr_8", pl.const(0, pl.INT64), 196608)] = pl.yield_(kv_final_inline2230__tile)
            kv_final_inline2230__rv_v2: pl.Tensor[[384, 128], pl.FP32, pl.MemRef("mem_ddr_9", pl.const(0, pl.INT64), 196608)] = pl.yield_(kv_final_inline2230__rv_v4)
        return kv_final_inline2230__ssa_v0

    @pl.function(type=pl.FunctionType.AIC, level=pl.Level.AIC, role=pl.Role.SubWorker)
    def kv_proj_matmul(
        kv_m_groups_inline1949__ssa_v0: pl.Scalar[pl.INDEX],
        kv_fp32_inline1938__rv_v2: pl.InOut[pl.Tensor[[t_matmul_inline1930__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)]],
        kv_full_rows_inline1946__ssa_v0: pl.Scalar[pl.INDEX],
        tile_base_inline1950__idx_v0: pl.Scalar[pl.INDEX],
        x_view_inline1933__ssa_v0: pl.Tensor[[t_dim_inline1931__ssa_v0, 4096], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)],
        wkv__ssa_v0: pl.Tensor[[4096, 512], pl.BF16, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 4194304)],
        t_matmul_inline1930__ssa_v0: pl.Scalar[pl.INDEX],
        tile_rows_inline1923__ssa_v0: pl.Scalar[pl.INDEX],
    ) -> pl.Tensor[[t_matmul_inline1930__ssa_v0, 512], pl.FP32]:
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
        kbg_inline1981__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        kv_col0_inline1929__ssa_v0: pl.Scalar[pl.INDEX] = kbg_inline1981__ssa_v0 // (kv_m_groups_inline1949__ssa_v0 * 2) * 128
        kv_k_base_inline1948__ssa_v0: pl.Scalar[pl.INDEX] = kbg_inline1981__ssa_v0 // kv_m_groups_inline1949__ssa_v0 % 2 * 2048
        kv_m_group_inline1927__ssa_v0: pl.Scalar[pl.INDEX] = kbg_inline1981__ssa_v0 % kv_m_groups_inline1949__ssa_v0
        for dense_t0_inline1925__idx_v0, (kv_fp32_inline1938__iter_v6,) in pl.range(
            kv_m_group_inline1927__ssa_v0 * 64, kv_full_rows_inline1946__ssa_v0, kv_m_groups_inline1949__ssa_v0 * 64, init_values=(kv_fp32_inline1938__rv_v2,)
        ):
            dense_x0_inline1941__ssa_v0: pl.Scalar[pl.INDEX] = tile_base_inline1950__idx_v0 + dense_t0_inline1925__idx_v0
            dense_acc_inline1924__tile: pl.Tile[[64, 128], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 32768), pl.Mem.Acc] = pl.tile.create([64, 128], dtype=pl.FP32, target_memory=pl.Mem.Acc)
            for dense_k_inline1919__idx_v0, (dense_acc_inline1924__iter_v1,) in pl.range(0, 8, 2, init_values=(dense_acc_inline1924__tile,)):
                dense_d0_inline1918__ssa_v0: pl.Scalar[pl.INDEX] = kv_k_base_inline1948__ssa_v0 + dense_k_inline1919__idx_v0 * 256
                dense_d0_inline1918__ssa_v0_1: pl.Scalar[pl.INDEX] = kv_k_base_inline1948__ssa_v0 + (dense_k_inline1919__idx_v0 * 256 + 256)
                dense_x_inline1916__tile: pl.Tile[[64, 256], pl.BF16, pl.MemRef(mem_mat_4, pl.const(0, pl.INT64), 32768), pl.Mem.Mat] = pl.tile.load(
                    x_view_inline1933__ssa_v0, [dense_x0_inline1941__ssa_v0, dense_d0_inline1918__ssa_v0], [64, 256], [64, 256], target_memory=pl.Mem.Mat
                )
                dense_w_inline1922__tile: pl.Tile[[256, 128], pl.BF16, pl.MemRef(mem_mat_5, pl.const(32768, pl.INT64), 65536), pl.Mem.Mat] = pl.tile.load(
                    wkv__ssa_v0, [dense_d0_inline1918__ssa_v0, kv_col0_inline1929__ssa_v0], [256, 128], [256, 128], target_memory=pl.Mem.Mat
                )
                dense_x_inline1916__tile_1: pl.Tile[[64, 256], pl.BF16, pl.MemRef(mem_mat_6, pl.const(98304, pl.INT64), 32768), pl.Mem.Mat] = pl.tile.load(
                    x_view_inline1933__ssa_v0, [dense_x0_inline1941__ssa_v0, dense_d0_inline1918__ssa_v0_1], [64, 256], [64, 256], target_memory=pl.Mem.Mat
                )
                dense_w_inline1922__tile_1: pl.Tile[[256, 128], pl.BF16, pl.MemRef(mem_mat_7, pl.const(131072, pl.INT64), 65536), pl.Mem.Mat] = pl.tile.load(
                    wkv__ssa_v0, [dense_d0_inline1918__ssa_v0_1, kv_col0_inline1929__ssa_v0], [256, 128], [256, 128], target_memory=pl.Mem.Mat
                )
                dense_acc_inline1924__tile_l0_a: pl.Tile[[64, 128], pl.BF16, pl.MemRef(mem_left_8, pl.const(49152, pl.INT64), 16384), pl.Mem.Left, pl.TileView(blayout=pl.TileLayout.row_major)] = (
                    pl.tile.extract(dense_x_inline1916__tile, 0, 0, [64, 128], target_memory=pl.Mem.Left)
                )
                dense_acc_inline1924__tile_l0_b: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_9, pl.const(32768, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                    dense_w_inline1922__tile, 0, 0, [128, 128], target_memory=pl.Mem.Right
                )
                dense_acc_inline1924__tile_l0_a_1: pl.Tile[[64, 128], pl.BF16, pl.MemRef(mem_left_10, pl.const(0, pl.INT64), 16384), pl.Mem.Left, pl.TileView(blayout=pl.TileLayout.row_major)] = (
                    pl.tile.extract(dense_x_inline1916__tile, 0, 128, [64, 128], target_memory=pl.Mem.Left)
                )
                dense_acc_inline1924__tile_l0_b_1: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_11, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                    dense_w_inline1922__tile, 128, 0, [128, 128], target_memory=pl.Mem.Right
                )
                dense_acc_inline1924__tile_l0_c_acc: pl.Tile[[64, 128], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 32768), pl.Mem.Acc] = pl.tile.matmul_acc(
                    dense_acc_inline1924__iter_v1, dense_acc_inline1924__tile_l0_a, dense_acc_inline1924__tile_l0_b, dense_k_inline1919__idx_v0 == 0
                )
                dense_acc_inline1924__tile_l0_c_acc_1: pl.Tile[[64, 128], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 32768), pl.Mem.Acc] = pl.tile.matmul_acc(
                    dense_acc_inline1924__tile_l0_c_acc, dense_acc_inline1924__tile_l0_a_1, dense_acc_inline1924__tile_l0_b_1, False
                )
                dense_acc_inline1924__tile_l0_a_2: pl.Tile[[64, 128], pl.BF16, pl.MemRef(mem_left_13, pl.const(16384, pl.INT64), 16384), pl.Mem.Left, pl.TileView(blayout=pl.TileLayout.row_major)] = (
                    pl.tile.extract(dense_x_inline1916__tile_1, 0, 0, [64, 128], target_memory=pl.Mem.Left)
                )
                dense_acc_inline1924__tile_l0_b_2: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_9, pl.const(32768, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                    dense_w_inline1922__tile_1, 0, 0, [128, 128], target_memory=pl.Mem.Right
                )
                dense_acc_inline1924__tile_l0_a_3: pl.Tile[[64, 128], pl.BF16, pl.MemRef(mem_left_15, pl.const(32768, pl.INT64), 16384), pl.Mem.Left, pl.TileView(blayout=pl.TileLayout.row_major)] = (
                    pl.tile.extract(dense_x_inline1916__tile_1, 0, 128, [64, 128], target_memory=pl.Mem.Left)
                )
                dense_acc_inline1924__tile_l0_b_3: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_11, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                    dense_w_inline1922__tile_1, 128, 0, [128, 128], target_memory=pl.Mem.Right
                )
                dense_acc_inline1924__tile_l0_c_acc_2: pl.Tile[[64, 128], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 32768), pl.Mem.Acc] = pl.tile.matmul_acc(
                    dense_acc_inline1924__tile_l0_c_acc_1, dense_acc_inline1924__tile_l0_a_2, dense_acc_inline1924__tile_l0_b_2, dense_k_inline1919__idx_v0 == -1
                )
                dense_acc_inline1924__tile_l0_c_acc_3: pl.Tile[[64, 128], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 32768), pl.Mem.Acc] = pl.tile.matmul_acc(
                    dense_acc_inline1924__tile_l0_c_acc_2, dense_acc_inline1924__tile_l0_a_3, dense_acc_inline1924__tile_l0_b_3, False
                )
                dense_acc_inline1924__rv_v2: pl.Tile[[64, 128], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 32768), pl.Mem.Acc] = pl.yield_(dense_acc_inline1924__tile_l0_c_acc_3)
            kv_fp32_inline1938__tile: pl.Tensor[[t_matmul_inline1930__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                dense_acc_inline1924__rv_v2, [dense_t0_inline1925__idx_v0, kv_col0_inline1929__ssa_v0], kv_fp32_inline1938__iter_v6, atomic=pl.AtomicType.Add
            )
            kv_fp32_inline1938__rv_v7: pl.Tensor[[t_matmul_inline1930__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_19", pl.const(0, pl.INT64), 0)] = pl.yield_(kv_fp32_inline1938__tile)
        for t0_inline1977__idx_v0, (kv_fp32_inline1938__iter_v9,) in pl.range(
            kv_full_rows_inline1946__ssa_v0 + kv_m_group_inline1927__ssa_v0 * 16, t_matmul_inline1930__ssa_v0, kv_m_groups_inline1949__ssa_v0 * 16, init_values=(kv_fp32_inline1938__rv_v7,)
        ):
            kv_acc_inline1914__tile: pl.Tile[[16, 128], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 8192), pl.Mem.Acc] = pl.tile.create([16, 128], dtype=pl.FP32, target_memory=pl.Mem.Acc)
            for db_inline1952__idx_v0, (kv_acc_inline1914__iter_v1,) in pl.range(0, 8, 2, init_values=(kv_acc_inline1914__tile,)):
                d0_inline1921__ssa_v0: pl.Scalar[pl.INDEX] = kv_k_base_inline1948__ssa_v0 + db_inline1952__idx_v0 * 256
                kv_rows_inline1928__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(tile_rows_inline1923__ssa_v0 - t0_inline1977__idx_v0, 16)
                x_t0_inline1913__ssa_v0: pl.Scalar[pl.INDEX] = tile_base_inline1950__idx_v0 + t0_inline1977__idx_v0
                d0_inline1921__ssa_v0_1: pl.Scalar[pl.INDEX] = kv_k_base_inline1948__ssa_v0 + (db_inline1952__idx_v0 * 256 + 256)
                kv_rows_inline1928__ssa_v0_1: pl.Scalar[pl.INDEX] = pl.min(tile_rows_inline1923__ssa_v0 - t0_inline1977__idx_v0, 16)
                x_t0_inline1913__ssa_v0_1: pl.Scalar[pl.INDEX] = tile_base_inline1950__idx_v0 + t0_inline1977__idx_v0
                kv_x_chunk_bf16_inline1960__tile: pl.Tile[
                    [16, 256], pl.BF16, pl.MemRef(mem_mat_4, pl.const(0, pl.INT64), 8192), pl.Mem.Mat, pl.TileView(valid_shape=[kv_rows_inline1928__ssa_v0, 256])
                ] = pl.tile.load(x_view_inline1933__ssa_v0, [x_t0_inline1913__ssa_v0, d0_inline1921__ssa_v0], [16, 256], [kv_rows_inline1928__ssa_v0, 256], target_memory=pl.Mem.Mat)
                wkv_chunk_inline1961__tile: pl.Tile[[256, 128], pl.BF16, pl.MemRef(mem_mat_5, pl.const(32768, pl.INT64), 65536), pl.Mem.Mat] = pl.tile.load(
                    wkv__ssa_v0, [d0_inline1921__ssa_v0, kv_col0_inline1929__ssa_v0], [256, 128], [256, 128], target_memory=pl.Mem.Mat
                )
                kv_x_chunk_bf16_inline1960__tile_1: pl.Tile[
                    [16, 256], pl.BF16, pl.MemRef(mem_mat_6, pl.const(98304, pl.INT64), 8192), pl.Mem.Mat, pl.TileView(valid_shape=[kv_rows_inline1928__ssa_v0_1, 256])
                ] = pl.tile.load(x_view_inline1933__ssa_v0, [x_t0_inline1913__ssa_v0_1, d0_inline1921__ssa_v0_1], [16, 256], [kv_rows_inline1928__ssa_v0_1, 256], target_memory=pl.Mem.Mat)
                wkv_chunk_inline1961__tile_1: pl.Tile[[256, 128], pl.BF16, pl.MemRef(mem_mat_7, pl.const(131072, pl.INT64), 65536), pl.Mem.Mat] = pl.tile.load(
                    wkv__ssa_v0, [d0_inline1921__ssa_v0_1, kv_col0_inline1929__ssa_v0], [256, 128], [256, 128], target_memory=pl.Mem.Mat
                )
                kv_acc_inline1914__tile_l0_a: pl.Tile[
                    [16, 128],
                    pl.BF16,
                    pl.MemRef(mem_left_8, pl.const(49152, pl.INT64), 4096),
                    pl.Mem.Left,
                    pl.TileView(valid_shape=[kv_rows_inline1928__ssa_v0, 128], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal),
                ] = pl.tile.extract(kv_x_chunk_bf16_inline1960__tile, 0, 0, [16, 128], target_memory=pl.Mem.Left)
                kv_acc_inline1914__tile_l0_b: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_9, pl.const(32768, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                    wkv_chunk_inline1961__tile, 0, 0, [128, 128], target_memory=pl.Mem.Right
                )
                kv_acc_inline1914__tile_l0_a_1: pl.Tile[
                    [16, 128],
                    pl.BF16,
                    pl.MemRef(mem_left_10, pl.const(0, pl.INT64), 4096),
                    pl.Mem.Left,
                    pl.TileView(valid_shape=[kv_rows_inline1928__ssa_v0, 128], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal),
                ] = pl.tile.extract(kv_x_chunk_bf16_inline1960__tile, 0, 128, [16, 128], target_memory=pl.Mem.Left)
                kv_acc_inline1914__tile_l0_b_1: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_11, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                    wkv_chunk_inline1961__tile, 128, 0, [128, 128], target_memory=pl.Mem.Right
                )
                kv_acc_inline1914__tile_l0_c_acc: pl.Tile[[16, 128], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 8192), pl.Mem.Acc] = pl.tile.matmul_acc(
                    kv_acc_inline1914__iter_v1, kv_acc_inline1914__tile_l0_a, kv_acc_inline1914__tile_l0_b, db_inline1952__idx_v0 == 0
                )
                kv_acc_inline1914__tile_l0_c_acc_1: pl.Tile[[16, 128], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 8192), pl.Mem.Acc] = pl.tile.matmul_acc(
                    kv_acc_inline1914__tile_l0_c_acc, kv_acc_inline1914__tile_l0_a_1, kv_acc_inline1914__tile_l0_b_1, False
                )
                kv_acc_inline1914__tile_l0_a_2: pl.Tile[
                    [16, 128],
                    pl.BF16,
                    pl.MemRef(mem_left_13, pl.const(16384, pl.INT64), 4096),
                    pl.Mem.Left,
                    pl.TileView(valid_shape=[kv_rows_inline1928__ssa_v0_1, 128], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal),
                ] = pl.tile.extract(kv_x_chunk_bf16_inline1960__tile_1, 0, 0, [16, 128], target_memory=pl.Mem.Left)
                kv_acc_inline1914__tile_l0_b_2: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_9, pl.const(32768, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                    wkv_chunk_inline1961__tile_1, 0, 0, [128, 128], target_memory=pl.Mem.Right
                )
                kv_acc_inline1914__tile_l0_a_3: pl.Tile[
                    [16, 128],
                    pl.BF16,
                    pl.MemRef(mem_left_15, pl.const(32768, pl.INT64), 4096),
                    pl.Mem.Left,
                    pl.TileView(valid_shape=[kv_rows_inline1928__ssa_v0_1, 128], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal),
                ] = pl.tile.extract(kv_x_chunk_bf16_inline1960__tile_1, 0, 128, [16, 128], target_memory=pl.Mem.Left)
                kv_acc_inline1914__tile_l0_b_3: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_11, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                    wkv_chunk_inline1961__tile_1, 128, 0, [128, 128], target_memory=pl.Mem.Right
                )
                kv_acc_inline1914__tile_l0_c_acc_2: pl.Tile[[16, 128], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 8192), pl.Mem.Acc] = pl.tile.matmul_acc(
                    kv_acc_inline1914__tile_l0_c_acc_1, kv_acc_inline1914__tile_l0_a_2, kv_acc_inline1914__tile_l0_b_2, db_inline1952__idx_v0 == -1
                )
                kv_acc_inline1914__tile_l0_c_acc_3: pl.Tile[[16, 128], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 8192), pl.Mem.Acc] = pl.tile.matmul_acc(
                    kv_acc_inline1914__tile_l0_c_acc_2, kv_acc_inline1914__tile_l0_a_3, kv_acc_inline1914__tile_l0_b_3, False
                )
                kv_acc_inline1914__rv_v2: pl.Tile[[16, 128], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 8192), pl.Mem.Acc] = pl.yield_(kv_acc_inline1914__tile_l0_c_acc_3)
            kv_fp32_inline1938__tile_1: pl.Tensor[[t_matmul_inline1930__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_19", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                kv_acc_inline1914__rv_v2, [t0_inline1977__idx_v0, kv_col0_inline1929__ssa_v0], kv_fp32_inline1938__iter_v9, atomic=pl.AtomicType.Add
            )
            kv_fp32_inline1938__rv_v10: pl.Tensor[[t_matmul_inline1930__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_36", pl.const(0, pl.INT64), 0)] = pl.yield_(kv_fp32_inline1938__tile_1)
        return kv_fp32_inline1938__rv_v2

    @pl.function(type=pl.FunctionType.Spmd)
    def kv_proj_matmul_spmd(
        self,
        kv_m_groups_inline1949__ssa_v0: pl.Scalar[pl.INDEX],
        kv_fp32_inline1938__rv_v2: pl.InOut[pl.Tensor[[t_matmul_inline1930__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)]],
        kv_full_rows_inline1946__ssa_v0: pl.Scalar[pl.INDEX],
        tile_base_inline1950__idx_v0: pl.Scalar[pl.INDEX],
        x_view_inline1933__ssa_v0: pl.Tensor[[t_dim_inline1931__ssa_v0, 4096], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)],
        wkv__ssa_v0: pl.Tensor[[4096, 512], pl.BF16, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 4194304)],
        t_matmul_inline1930__ssa_v0: pl.Scalar[pl.INDEX],
        tile_rows_inline1923__ssa_v0: pl.Scalar[pl.INDEX],
    ) -> pl.Tensor[[t_matmul_inline1930__ssa_v0, 512], pl.FP32]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        kv_fp32_inline1938__rv_v10: pl.Tensor[[t_matmul_inline1930__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)] = self.kv_proj_matmul(
            kv_m_groups_inline1949__ssa_v0,
            kv_fp32_inline1938__rv_v2,
            kv_full_rows_inline1946__ssa_v0,
            tile_base_inline1950__idx_v0,
            x_view_inline1933__ssa_v0,
            wkv__ssa_v0,
            t_matmul_inline1930__ssa_v0,
            tile_rows_inline1923__ssa_v0,
            attrs={"arg_directions": [pl.adir.scalar, pl.adir.inout, pl.adir.scalar, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.scalar, pl.adir.scalar]},
        )
        return kv_fp32_inline1938__rv_v2

    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def kv_proj_seed(
        kv_fp32_inline1938__ssa_v0: pl.Out[pl.Tensor[[t_matmul_inline1930__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)]], t_matmul_inline1930__ssa_v0: pl.Scalar[pl.INDEX]
    ) -> pl.Tensor[[t_matmul_inline1930__ssa_v0, 512], pl.FP32]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        mem_vec_1: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 8192)
        for kts0_inline1964__idx_v0, (kv_fp32_inline1938__iter_v1,) in pl.range(0, t_matmul_inline1930__ssa_v0, 16, init_values=(kv_fp32_inline1938__ssa_v0,)):
            for kvseed0_inline1955__idx_v0, (kv_fp32_inline1938__iter_v3,) in pl.range(0, 512, 128, init_values=(kv_fp32_inline1938__iter_v1,)):
                kv_seed_inline1967__tile: pl.Tile[[16, 128], pl.FP32, pl.MemRef(mem_vec_1, pl.const(0, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.full([16, 128], dtype=pl.FP32, value=0.0)
                kv_fp32_inline1938__tile: pl.Tensor[[t_matmul_inline1930__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                    kv_seed_inline1967__tile, [kts0_inline1964__idx_v0, kvseed0_inline1955__idx_v0], kv_fp32_inline1938__iter_v3
                )
                kv_fp32_inline1938__rv_v4: pl.Tensor[[t_matmul_inline1930__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)] = pl.yield_(kv_fp32_inline1938__tile)
            kv_fp32_inline1938__rv_v2: pl.Tensor[[t_matmul_inline1930__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)] = pl.yield_(kv_fp32_inline1938__rv_v4)
        return kv_fp32_inline1938__ssa_v0

    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def kv_rms_norm_rope(
        tile_rows_inline1923__ssa_v0: pl.Scalar[pl.INDEX],
        tile_base_inline1950__idx_v0: pl.Scalar[pl.INDEX],
        kv_fp32_inline1938__rv_v10: pl.Tensor[[t_matmul_inline1930__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)],
        kv_view_inline1962__ssa_v0: pl.InOut[pl.Tensor[[t_dim_inline1931__ssa_v0, 512], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)]],
        gamma_ckv__ssa_v0: pl.Tensor[[512], pl.BF16, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 1024)],
        q_rope_cos_il_inline545__ssa_v0: pl.Tensor[[t_dim_inline546__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)],
        q_rope_sin_signed_inline544__ssa_v0: pl.Tensor[[t_dim_inline546__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 0)],
        q_rope_swap_idx_inline543__ssa_v0: pl.Tensor[[t_dim_inline546__ssa_v0, 64], pl.INT32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 0)],
    ) -> pl.Tensor[[t_dim_inline1931__ssa_v0, 512], pl.BF16]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        mem_vec_11: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 8192)
        mem_vec_12: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
        mem_vec_17: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 8192)
        mem_vec_18: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
        mem_vec_64: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 8192)
        mem_vec_145: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 8192)
        mem_vec_146: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 8192)
        tg_idx_inline1943__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        tg_inline1968__ssa_v0: pl.Scalar[pl.INDEX] = tg_idx_inline1943__ssa_v0 * 32
        valid_rows_inline1934__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(tile_rows_inline1923__ssa_v0 - tg_inline1968__ssa_v0, 32)
        out_tg_inline1969__ssa_v0: pl.Scalar[pl.INDEX] = tile_base_inline1950__idx_v0 + tg_inline1968__ssa_v0
        if valid_rows_inline1934__ssa_v0 == 32:
            kv_sq_sum_inline1970__tile: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_64, pl.const(65536, pl.INT64), 128), pl.Mem.Vec] = pl.tile.full([1, 32], dtype=pl.FP32, value=0.0)
            for kv_sq_col0_inline1972__idx_v0, (kv_sq_sum_inline1970__iter_v1,) in pl.range(0, 512, 128, init_values=(kv_sq_sum_inline1970__tile,)):
                kv_chunk_inline1980__tile: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(8192, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.load(
                    kv_fp32_inline1938__rv_v10, [tg_inline1968__ssa_v0, kv_sq_col0_inline1972__idx_v0], [32, 64], [32, 64], target_memory=pl.Mem.Vec
                )
                kv_chunk_inline1980__tile_1: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_18, pl.const(49152, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.load(
                    kv_fp32_inline1938__rv_v10, [tg_inline1968__ssa_v0, kv_sq_col0_inline1972__idx_v0 + 64], [32, 64], [32, 64], target_memory=pl.Mem.Vec
                )
                t__tile: pl.Tile[[32, 64], pl.BF16, pl.MemRef(mem_vec_11, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(kv_chunk_inline1980__tile, target_type=pl.BF16, mode="rint")
                kv_chunk_v1_inline1965__tile: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(8192, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.cast(t__tile, target_type=pl.FP32, mode="round")
                kv_sq_inline1973__tile: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_11, pl.const(0, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.mul(
                    kv_chunk_v1_inline1965__tile, kv_chunk_v1_inline1965__tile
                )
                tmp_tile: pl.Tile[[32, 128], pl.FP32, pl.MemRef(mem_vec_12, pl.const(8192, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.create([32, 128], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                t__tile_1: pl.Tile[[32, 1], pl.FP32, pl.MemRef(mem_vec_145, pl.const(24576, pl.INT64), 128), pl.Mem.Vec] = pl.tile.row_sum(kv_sq_inline1973__tile, tmp_tile)
                kv_row_sum_inline1954__tile: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_145, pl.const(24576, pl.INT64), 128), pl.Mem.Vec] = pl.tile.reshape(t__tile_1, [1, 32])
                kv_sq_sum_inline1970__tile_1: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_12, pl.const(8192, pl.INT64), 128), pl.Mem.Vec] = pl.tile.add(
                    kv_sq_sum_inline1970__iter_v1, kv_row_sum_inline1954__tile
                )
                t__tile_2: pl.Tile[[32, 64], pl.BF16, pl.MemRef(mem_vec_17, pl.const(40960, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(kv_chunk_inline1980__tile_1, target_type=pl.BF16, mode="rint")
                kv_chunk_v1_inline1965__tile_1: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_18, pl.const(49152, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.cast(
                    t__tile_2, target_type=pl.FP32, mode="round"
                )
                kv_sq_inline1973__tile_1: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_17, pl.const(40960, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.mul(
                    kv_chunk_v1_inline1965__tile_1, kv_chunk_v1_inline1965__tile_1
                )
                tmp_tile_1: pl.Tile[[32, 128], pl.FP32, pl.MemRef(mem_vec_18, pl.const(49152, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.create([32, 128], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                t__tile_3: pl.Tile[[32, 1], pl.FP32, pl.MemRef(mem_vec_146, pl.const(32768, pl.INT64), 128), pl.Mem.Vec] = pl.tile.row_sum(kv_sq_inline1973__tile_1, tmp_tile_1)
                kv_row_sum_inline1954__tile_1: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_146, pl.const(32768, pl.INT64), 128), pl.Mem.Vec] = pl.tile.reshape(t__tile_3, [1, 32])
                kv_sq_sum_inline1970__tile_2: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_64, pl.const(65536, pl.INT64), 128), pl.Mem.Vec] = pl.tile.add(
                    kv_sq_sum_inline1970__tile_1, kv_row_sum_inline1954__tile_1
                )
                kv_sq_sum_inline1970__rv_v2: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_64, pl.const(65536, pl.INT64), 128), pl.Mem.Vec] = pl.yield_(kv_sq_sum_inline1970__tile_2)
            t__tile_4: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_12, pl.const(8192, pl.INT64), 128), pl.Mem.Vec] = pl.tile.muls(kv_sq_sum_inline1970__rv_v2, 0.001953125)
            t__tile_5: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_12, pl.const(8192, pl.INT64), 128), pl.Mem.Vec] = pl.tile.adds(t__tile_4, 9.9999999999999995e-07)
            rsqrt_tmp: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_18, pl.const(49152, pl.INT64), 128), pl.Mem.Vec] = pl.tile.create([1, 32], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            kv_inv_rms_inline1974__tile: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_64, pl.const(65536, pl.INT64), 128), pl.Mem.Vec] = pl.tile.rsqrt(t__tile_5, rsqrt_tmp)
            kv_inv_rms_t_inline1915__tile: pl.Tile[[32, 1], pl.FP32, pl.MemRef(mem_vec_64, pl.const(65536, pl.INT64), 128), pl.Mem.Vec] = pl.tile.reshape(kv_inv_rms_inline1974__tile, [32, 1])
            for n0_inline1956__idx_v0, (kv_view_inline1962__iter_v1,) in pl.range(0, 384, 128, init_values=(kv_view_inline1962__ssa_v0,)):
                kv_chunk_inline1980__tile_2: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(8192, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.load(
                    kv_fp32_inline1938__rv_v10, [tg_inline1968__ssa_v0, n0_inline1956__idx_v0], [32, 64], [32, 64], target_memory=pl.Mem.Vec
                )
                t__tile_6: pl.Tile[[64], pl.BF16, pl.MemRef(mem_vec_145, pl.const(24576, pl.INT64), 128), pl.Mem.Vec] = pl.tile.load(
                    gamma_ckv__ssa_v0, [n0_inline1956__idx_v0], [64], [64], target_memory=pl.Mem.Vec
                )
                kv_chunk_inline1980__tile_3: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_18, pl.const(49152, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.load(
                    kv_fp32_inline1938__rv_v10, [tg_inline1968__ssa_v0, n0_inline1956__idx_v0 + 64], [32, 64], [32, 64], target_memory=pl.Mem.Vec
                )
                t__tile_7: pl.Tile[[64], pl.BF16, pl.MemRef(mem_vec_146, pl.const(32768, pl.INT64), 128), pl.Mem.Vec] = pl.tile.load(
                    gamma_ckv__ssa_v0, [n0_inline1956__idx_v0 + 64], [64], [64], target_memory=pl.Mem.Vec
                )
                t__tile_8: pl.Tile[[32, 64], pl.BF16, pl.MemRef(mem_vec_11, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(kv_chunk_inline1980__tile_2, target_type=pl.BF16, mode="rint")
                kv_chunk_v2_inline1940__tile: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(8192, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.cast(t__tile_8, target_type=pl.FP32, mode="round")
                gamma_kv_cast_inline1976__tile: pl.Tile[[64], pl.FP32, pl.MemRef(mem_vec_11, pl.const(0, pl.INT64), 256), pl.Mem.Vec] = pl.tile.cast(t__tile_6, target_type=pl.FP32, mode="round")
                gamma_kv_chunk_inline1985__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_11, pl.const(0, pl.INT64), 256), pl.Mem.Vec] = gamma_kv_cast_inline1976__tile
                t__tile_9: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(8192, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.row_expand_mul(
                    kv_chunk_v2_inline1940__tile, kv_inv_rms_t_inline1915__tile
                )
                kv_normed_inline1920__tile: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(8192, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.col_expand_mul(
                    t__tile_9, gamma_kv_chunk_inline1985__tile
                )
                kv_normed_bf16_inline1982__tile: pl.Tile[[32, 64], pl.BF16, pl.MemRef(mem_vec_12, pl.const(8192, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(
                    kv_normed_inline1920__tile, target_type=pl.BF16, mode="rint"
                )
                kv_view_inline1962__tile: pl.Tensor[[t_dim_inline1931__ssa_v0, 512], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                    kv_normed_bf16_inline1982__tile, [out_tg_inline1969__ssa_v0, n0_inline1956__idx_v0], kv_view_inline1962__iter_v1
                )
                t__tile_10: pl.Tile[[32, 64], pl.BF16, pl.MemRef(mem_vec_17, pl.const(40960, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(kv_chunk_inline1980__tile_3, target_type=pl.BF16, mode="rint")
                kv_chunk_v2_inline1940__tile_1: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_18, pl.const(49152, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.cast(
                    t__tile_10, target_type=pl.FP32, mode="round"
                )
                gamma_kv_cast_inline1976__tile_1: pl.Tile[[64], pl.FP32, pl.MemRef(mem_vec_17, pl.const(40960, pl.INT64), 256), pl.Mem.Vec] = pl.tile.cast(t__tile_7, target_type=pl.FP32, mode="round")
                gamma_kv_chunk_inline1985__tile_1: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_17, pl.const(40960, pl.INT64), 256), pl.Mem.Vec] = gamma_kv_cast_inline1976__tile_1
                t__tile_11: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_18, pl.const(49152, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.row_expand_mul(
                    kv_chunk_v2_inline1940__tile_1, kv_inv_rms_t_inline1915__tile
                )
                kv_normed_inline1920__tile_1: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_18, pl.const(49152, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.col_expand_mul(
                    t__tile_11, gamma_kv_chunk_inline1985__tile_1
                )
                kv_normed_bf16_inline1982__tile_1: pl.Tile[[32, 64], pl.BF16, pl.MemRef(mem_vec_18, pl.const(49152, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(
                    kv_normed_inline1920__tile_1, target_type=pl.BF16, mode="rint"
                )
                kv_view_inline1962__tile_1: pl.Tensor[[t_dim_inline1931__ssa_v0, 512], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                    kv_normed_bf16_inline1982__tile_1, [out_tg_inline1969__ssa_v0, n0_inline1956__idx_v0 + 64], kv_view_inline1962__tile
                )
                kv_view_inline1962__rv_v2_main: pl.Tensor[[t_dim_inline1931__ssa_v0, 512], pl.BF16, pl.MemRef("mem_ddr_42", pl.const(0, pl.INT64), 0)] = pl.yield_(kv_view_inline1962__tile_1)
            kv_chunk_inline1980__tile_4: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(8192, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.load(
                kv_fp32_inline1938__rv_v10, [tg_inline1968__ssa_v0, 384], [32, 64], [32, 64], target_memory=pl.Mem.Vec
            )
            t__tile_12: pl.Tile[[32, 64], pl.BF16, pl.MemRef(mem_vec_18, pl.const(49152, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(kv_chunk_inline1980__tile_4, target_type=pl.BF16, mode="rint")
            kv_chunk_v2_inline1940__tile_2: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(8192, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.cast(t__tile_12, target_type=pl.FP32, mode="round")
            t__tile_13: pl.Tile[[64], pl.BF16, pl.MemRef(mem_vec_11, pl.const(0, pl.INT64), 128), pl.Mem.Vec] = pl.tile.load(gamma_ckv__ssa_v0, [384], [64], [64], target_memory=pl.Mem.Vec)
            gamma_kv_cast_inline1976__tile_2: pl.Tile[[64], pl.FP32, pl.MemRef(mem_vec_18, pl.const(49152, pl.INT64), 256), pl.Mem.Vec] = pl.tile.cast(t__tile_13, target_type=pl.FP32, mode="round")
            gamma_kv_chunk_inline1985__tile_2: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_18, pl.const(49152, pl.INT64), 256), pl.Mem.Vec] = gamma_kv_cast_inline1976__tile_2
            t__tile_14: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(8192, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.row_expand_mul(
                kv_chunk_v2_inline1940__tile_2, kv_inv_rms_t_inline1915__tile
            )
            kv_normed_inline1920__tile_2: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(8192, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.col_expand_mul(
                t__tile_14, gamma_kv_chunk_inline1985__tile_2
            )
            kv_normed_bf16_inline1982__tile_2: pl.Tile[[32, 64], pl.BF16, pl.MemRef(mem_vec_12, pl.const(8192, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(
                kv_normed_inline1920__tile_2, target_type=pl.BF16, mode="rint"
            )
            kv_view_inline1962__tile_2: pl.Tensor[[t_dim_inline1931__ssa_v0, 512], pl.BF16, pl.MemRef("mem_ddr_42", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                kv_normed_bf16_inline1982__tile_2, [out_tg_inline1969__ssa_v0, 384], kv_view_inline1962__rv_v2_main
            )
            kv_view_inline1962__rv_v2: pl.Tensor[[t_dim_inline1931__ssa_v0, 512], pl.BF16, pl.MemRef("mem_ddr_51", pl.const(0, pl.INT64), 0)] = kv_view_inline1962__tile_2
            t__tile_15: pl.Tile[[64], pl.BF16, pl.MemRef(mem_vec_12, pl.const(8192, pl.INT64), 128), pl.Mem.Vec] = pl.tile.load(gamma_ckv__ssa_v0, [448], [64], [64], target_memory=pl.Mem.Vec)
            gamma_rope_cast_inline1936__tile: pl.Tile[[64], pl.FP32, pl.MemRef(mem_vec_11, pl.const(0, pl.INT64), 256), pl.Mem.Vec] = pl.tile.cast(t__tile_15, target_type=pl.FP32, mode="round")
            gamma_rope_inline1945__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_11, pl.const(0, pl.INT64), 256), pl.Mem.Vec] = gamma_rope_cast_inline1936__tile
            kv_rope_chunk_inline1978__tile: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(8192, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.load(
                kv_fp32_inline1938__rv_v10, [tg_inline1968__ssa_v0, 448], [32, 64], [32, 64], target_memory=pl.Mem.Vec
            )
            t__tile_16: pl.Tile[[32, 64], pl.BF16, pl.MemRef(mem_vec_18, pl.const(49152, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(kv_rope_chunk_inline1978__tile, target_type=pl.BF16, mode="rint")
            kv_rope_chunk_v1_inline1983__tile: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(8192, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.cast(
                t__tile_16, target_type=pl.FP32, mode="round"
            )
            t__tile_17: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(8192, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.row_expand_mul(
                kv_rope_chunk_v1_inline1983__tile, kv_inv_rms_t_inline1915__tile
            )
            kv_rope_norm_chunk_inline1986__tile: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(8192, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.col_expand_mul(
                t__tile_17, gamma_rope_inline1945__tile
            )
            t__tile_18: pl.Tile[[32, 64], pl.BF16, pl.MemRef(mem_vec_18, pl.const(49152, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(
                kv_rope_norm_chunk_inline1986__tile, target_type=pl.BF16, mode="rint"
            )
            kv_rope_norm_chunk_v1_inline1958__tile: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(8192, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.cast(
                t__tile_18, target_type=pl.FP32, mode="round"
            )
            kv_cos_il_full_inline1987__tile: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_18, pl.const(49152, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.load(
                q_rope_cos_il_inline545__ssa_v0, [out_tg_inline1969__ssa_v0, 0], [32, 64], [32, 64], target_memory=pl.Mem.Vec
            )
            kv_sin_signed_full_inline1988__tile: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_11, pl.const(0, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.load(
                q_rope_sin_signed_inline544__ssa_v0, [out_tg_inline1969__ssa_v0, 0], [32, 64], [32, 64], target_memory=pl.Mem.Vec
            )
            kv_swap_idx_full_inline1911__tile: pl.Tile[[32, 64], pl.INT32, pl.MemRef(mem_vec_17, pl.const(40960, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.load(
                q_rope_swap_idx_inline543__ssa_v0, [out_tg_inline1969__ssa_v0, 0], [32, 64], [32, 64], target_memory=pl.Mem.Vec
            )
            gather_acc_init: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_64, pl.const(65536, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.create([32, 64], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            for gather_lv, (gather_ia,) in pl.range(32, init_values=(gather_acc_init,)):
                gather_inp_row: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(8192, pl.INT64), 256), pl.Mem.Vec] = pl.tile.slice(
                    kv_rope_norm_chunk_v1_inline1958__tile, [1, 64], [gather_lv, 0], [1, 64]
                )
                gather_idx_row: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_17, pl.const(40960, pl.INT64), 256), pl.Mem.Vec] = pl.tile.slice(
                    kv_swap_idx_full_inline1911__tile, [1, 64], [gather_lv, 0], [1, 64]
                )
                gather_row_tmp: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_145, pl.const(24576, pl.INT64), 256), pl.Mem.Vec] = pl.tile.create([1, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec)
                gather_row: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_146, pl.const(32768, pl.INT64), 256), pl.Mem.Vec] = pl.tile.gather(gather_inp_row, gather_idx_row, gather_row_tmp)
                gather_asmbl: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_64, pl.const(65536, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.assemble(gather_ia, gather_row, [gather_lv, 0])
                gather_rv: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_64, pl.const(65536, pl.INT64), 8192), pl.Mem.Vec] = pl.yield_(gather_asmbl)
            kv_swapped_full_inline1910__tile: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_64, pl.const(65536, pl.INT64), 8192), pl.Mem.Vec] = gather_rv
            t__tile_19: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(8192, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.mul(
                kv_rope_norm_chunk_v1_inline1958__tile, kv_cos_il_full_inline1987__tile
            )
            t__tile_20: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_18, pl.const(49152, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.mul(
                kv_swapped_full_inline1910__tile, kv_sin_signed_full_inline1988__tile
            )
            kv_rope_rot_full_inline1984__tile: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(8192, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.add(t__tile_19, t__tile_20)
            kv_rope_i16_full_inline1939__tile: pl.Tile[[32, 64], pl.BF16, pl.MemRef(mem_vec_12, pl.const(8192, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(
                kv_rope_rot_full_inline1984__tile, target_type=pl.BF16, mode="rint"
            )
            kv_view_inline1962__tile_3: pl.Tensor[[t_dim_inline1931__ssa_v0, 512], pl.BF16, pl.MemRef("mem_ddr_51", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                kv_rope_i16_full_inline1939__tile, [out_tg_inline1969__ssa_v0, 448], kv_view_inline1962__rv_v2
            )
        else:
            kv_reduce_tmp_inline1959__ssa_v0: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(8192, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.create(
                [32, 64], dtype=pl.FP32, target_memory=pl.Mem.Vec
            )
            kv_sq_sum_tail_inline1908__ssa_v0: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_145, pl.const(24576, pl.INT64), 128), pl.Mem.Vec] = pl.tile.full([1, 32], dtype=pl.FP32, value=0.0)
            for kv_sq_col0_tail_inline1953__idx_v0, (kv_sq_sum_tail_inline1908__iter_v1,) in pl.range(0, 512, 128, init_values=(kv_sq_sum_tail_inline1908__ssa_v0,)):
                kv_chunk_tail_inline1909__ssa_v0: pl.Tile[
                    [32, 64], pl.FP32, pl.MemRef(mem_vec_18, pl.const(49152, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1934__ssa_v0, 64])
                ] = pl.tile.load(kv_fp32_inline1938__rv_v10, [tg_inline1968__ssa_v0, kv_sq_col0_tail_inline1953__idx_v0], [32, 64], [valid_rows_inline1934__ssa_v0, 64], target_memory=pl.Mem.Vec)
                kv_chunk_tail_inline1909__ssa_v0_1: pl.Tile[
                    [32, 64], pl.FP32, pl.MemRef(mem_vec_11, pl.const(0, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1934__ssa_v0, 64])
                ] = pl.tile.load(kv_fp32_inline1938__rv_v10, [tg_inline1968__ssa_v0, kv_sq_col0_tail_inline1953__idx_v0 + 64], [32, 64], [valid_rows_inline1934__ssa_v0, 64], target_memory=pl.Mem.Vec)
                t__tmp_v89: pl.Tile[[32, 64], pl.BF16, pl.MemRef(mem_vec_17, pl.const(40960, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1934__ssa_v0, 64])] = pl.tile.cast(
                    kv_chunk_tail_inline1909__ssa_v0, target_type=pl.BF16, mode="rint"
                )
                kv_chunk_tail_v1_inline1907__ssa_v0: pl.Tile[
                    [32, 64], pl.FP32, pl.MemRef(mem_vec_18, pl.const(49152, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1934__ssa_v0, 64])
                ] = pl.tile.cast(t__tmp_v89, target_type=pl.FP32, mode="round")
                kv_sq_tail_inline1906__ssa_v0: pl.Tile[
                    [32, 64], pl.FP32, pl.MemRef(mem_vec_18, pl.const(49152, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1934__ssa_v0, 64])
                ] = pl.tile.mul(kv_chunk_tail_v1_inline1907__ssa_v0, kv_chunk_tail_v1_inline1907__ssa_v0)
                t__tmp_v90: pl.Tile[[32, 1], pl.FP32, pl.MemRef(mem_vec_17, pl.const(40960, pl.INT64), 128), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1934__ssa_v0, 1])] = pl.tile.row_sum(
                    kv_sq_tail_inline1906__ssa_v0, kv_reduce_tmp_inline1959__ssa_v0
                )
                kv_row_sum_tail_inline1905__ssa_v0: pl.Tile[
                    [1, 32], pl.FP32, pl.MemRef(mem_vec_17, pl.const(40960, pl.INT64), 128), pl.Mem.Vec, pl.TileView(valid_shape=[1, valid_rows_inline1934__ssa_v0])
                ] = pl.tile.reshape(t__tmp_v90, [1, 32])
                kv_sq_sum_tail_inline1908__ssa_v3: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_18, pl.const(49152, pl.INT64), 128), pl.Mem.Vec] = pl.tile.add(
                    kv_sq_sum_tail_inline1908__iter_v1, kv_row_sum_tail_inline1905__ssa_v0
                )
                t__tmp_v89_1: pl.Tile[[32, 64], pl.BF16, pl.MemRef(mem_vec_64, pl.const(65536, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1934__ssa_v0, 64])] = (
                    pl.tile.cast(kv_chunk_tail_inline1909__ssa_v0_1, target_type=pl.BF16, mode="rint")
                )
                kv_chunk_tail_v1_inline1907__ssa_v0_1: pl.Tile[
                    [32, 64], pl.FP32, pl.MemRef(mem_vec_11, pl.const(0, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1934__ssa_v0, 64])
                ] = pl.tile.cast(t__tmp_v89_1, target_type=pl.FP32, mode="round")
                kv_sq_tail_inline1906__ssa_v0_1: pl.Tile[
                    [32, 64], pl.FP32, pl.MemRef(mem_vec_11, pl.const(0, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1934__ssa_v0, 64])
                ] = pl.tile.mul(kv_chunk_tail_v1_inline1907__ssa_v0_1, kv_chunk_tail_v1_inline1907__ssa_v0_1)
                t__tmp_v90_1: pl.Tile[[32, 1], pl.FP32, pl.MemRef(mem_vec_64, pl.const(65536, pl.INT64), 128), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1934__ssa_v0, 1])] = (
                    pl.tile.row_sum(kv_sq_tail_inline1906__ssa_v0_1, kv_reduce_tmp_inline1959__ssa_v0)
                )
                kv_row_sum_tail_inline1905__ssa_v0_1: pl.Tile[
                    [1, 32], pl.FP32, pl.MemRef(mem_vec_64, pl.const(65536, pl.INT64), 128), pl.Mem.Vec, pl.TileView(valid_shape=[1, valid_rows_inline1934__ssa_v0])
                ] = pl.tile.reshape(t__tmp_v90_1, [1, 32])
                kv_sq_sum_tail_inline1908__ssa_v3_1: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_145, pl.const(24576, pl.INT64), 128), pl.Mem.Vec] = pl.tile.add(
                    kv_sq_sum_tail_inline1908__ssa_v3, kv_row_sum_tail_inline1905__ssa_v0_1
                )
                kv_sq_sum_tail_inline1908__rv_v2: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_145, pl.const(24576, pl.INT64), 128), pl.Mem.Vec] = pl.yield_(kv_sq_sum_tail_inline1908__ssa_v3_1)
            t__tmp_v91: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_12, pl.const(8192, pl.INT64), 128), pl.Mem.Vec] = pl.tile.muls(kv_sq_sum_tail_inline1908__rv_v2, 0.001953125)
            t__tmp_v92: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_12, pl.const(8192, pl.INT64), 128), pl.Mem.Vec] = pl.tile.adds(t__tmp_v91, 9.9999999999999995e-07)
            t__tmp_v93: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_12, pl.const(8192, pl.INT64), 128), pl.Mem.Vec] = pl.tile.sqrt(t__tmp_v92)
            kv_inv_rms_tail_inline1951__ssa_v0: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_64, pl.const(65536, pl.INT64), 128), pl.Mem.Vec] = pl.tile.recip(t__tmp_v93)
            kv_inv_rms_t_tail_inline1904__ssa_v0: pl.Tile[[32, 1], pl.FP32, pl.MemRef(mem_vec_64, pl.const(65536, pl.INT64), 128), pl.Mem.Vec] = pl.tile.reshape(
                kv_inv_rms_tail_inline1951__ssa_v0, [32, 1]
            )
            for n0_tail_inline1975__idx_v0 in pl.range(0, 384, 128):
                kv_chunk_tail_inline1909__ssa_v1: pl.Tile[
                    [32, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(8192, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1934__ssa_v0, 64])
                ] = pl.tile.load(kv_fp32_inline1938__rv_v10, [tg_inline1968__ssa_v0, n0_tail_inline1975__idx_v0], [32, 64], [valid_rows_inline1934__ssa_v0, 64], target_memory=pl.Mem.Vec)
                gamma_kv_input_tail_inline1903__ssa_v0: pl.Tile[[64], pl.BF16, pl.MemRef(mem_vec_145, pl.const(24576, pl.INT64), 128), pl.Mem.Vec] = pl.tile.load(
                    gamma_ckv__ssa_v0, [n0_tail_inline1975__idx_v0], [64], [64], target_memory=pl.Mem.Vec
                )
                kv_chunk_tail_inline1909__ssa_v1_1: pl.Tile[
                    [32, 64], pl.FP32, pl.MemRef(mem_vec_18, pl.const(49152, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1934__ssa_v0, 64])
                ] = pl.tile.load(kv_fp32_inline1938__rv_v10, [tg_inline1968__ssa_v0, n0_tail_inline1975__idx_v0 + 64], [32, 64], [valid_rows_inline1934__ssa_v0, 64], target_memory=pl.Mem.Vec)
                gamma_kv_input_tail_inline1903__ssa_v0_1: pl.Tile[[64], pl.BF16, pl.MemRef(mem_vec_146, pl.const(32768, pl.INT64), 128), pl.Mem.Vec] = pl.tile.load(
                    gamma_ckv__ssa_v0, [n0_tail_inline1975__idx_v0 + 64], [64], [64], target_memory=pl.Mem.Vec
                )
                t__tmp_v94: pl.Tile[[32, 64], pl.BF16, pl.MemRef(mem_vec_11, pl.const(0, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1934__ssa_v0, 64])] = pl.tile.cast(
                    kv_chunk_tail_inline1909__ssa_v1, target_type=pl.BF16, mode="rint"
                )
                kv_chunk_tail_v2_inline1932__ssa_v0: pl.Tile[
                    [32, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(8192, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1934__ssa_v0, 64])
                ] = pl.tile.cast(t__tmp_v94, target_type=pl.FP32, mode="round")
                gamma_kv_cast_tail_inline1902__ssa_v0: pl.Tile[[64], pl.FP32, pl.MemRef(mem_vec_11, pl.const(0, pl.INT64), 256), pl.Mem.Vec] = pl.tile.cast(
                    gamma_kv_input_tail_inline1903__ssa_v0, target_type=pl.FP32, mode="round"
                )
                gamma_kv_chunk_tail_inline1901__ssa_v0: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_11, pl.const(0, pl.INT64), 256), pl.Mem.Vec] = gamma_kv_cast_tail_inline1902__ssa_v0
                t__tmp_v95: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(8192, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1934__ssa_v0, 64])] = (
                    pl.tile.row_expand_mul(kv_chunk_tail_v2_inline1932__ssa_v0, kv_inv_rms_t_tail_inline1904__ssa_v0)
                )
                kv_normed_tail_inline1900__ssa_v0: pl.Tile[
                    [32, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(8192, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1934__ssa_v0, 64])
                ] = pl.tile.col_expand_mul(t__tmp_v95, gamma_kv_chunk_tail_inline1901__ssa_v0)
                kv_normed_bf16_tail_inline1944__ssa_v0: pl.Tile[
                    [32, 64], pl.BF16, pl.MemRef(mem_vec_12, pl.const(8192, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1934__ssa_v0, 64])
                ] = pl.tile.cast(kv_normed_tail_inline1900__ssa_v0, target_type=pl.BF16, mode="rint")
                kv_normed_valid_inline1899__ssa_v0: pl.Tile[
                    [32, 64], pl.BF16, pl.MemRef(mem_vec_12, pl.const(8192, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1934__ssa_v0, 64])
                ] = pl.tile.set_validshape(kv_normed_bf16_tail_inline1944__ssa_v0, valid_rows_inline1934__ssa_v0, 64)
                kv_view_inline1962__store: pl.Tensor[[t_dim_inline1931__ssa_v0, 512], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                    kv_normed_valid_inline1899__ssa_v0, [out_tg_inline1969__ssa_v0, n0_tail_inline1975__idx_v0], kv_view_inline1962__ssa_v0
                )
                t__tmp_v94_1: pl.Tile[[32, 64], pl.BF16, pl.MemRef(mem_vec_17, pl.const(40960, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1934__ssa_v0, 64])] = (
                    pl.tile.cast(kv_chunk_tail_inline1909__ssa_v1_1, target_type=pl.BF16, mode="rint")
                )
                kv_chunk_tail_v2_inline1932__ssa_v0_1: pl.Tile[
                    [32, 64], pl.FP32, pl.MemRef(mem_vec_18, pl.const(49152, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1934__ssa_v0, 64])
                ] = pl.tile.cast(t__tmp_v94_1, target_type=pl.FP32, mode="round")
                gamma_kv_cast_tail_inline1902__ssa_v0_1: pl.Tile[[64], pl.FP32, pl.MemRef(mem_vec_17, pl.const(40960, pl.INT64), 256), pl.Mem.Vec] = pl.tile.cast(
                    gamma_kv_input_tail_inline1903__ssa_v0_1, target_type=pl.FP32, mode="round"
                )
                gamma_kv_chunk_tail_inline1901__ssa_v0_1: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_17, pl.const(40960, pl.INT64), 256), pl.Mem.Vec] = gamma_kv_cast_tail_inline1902__ssa_v0_1
                t__tmp_v95_1: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_18, pl.const(49152, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1934__ssa_v0, 64])] = (
                    pl.tile.row_expand_mul(kv_chunk_tail_v2_inline1932__ssa_v0_1, kv_inv_rms_t_tail_inline1904__ssa_v0)
                )
                kv_normed_tail_inline1900__ssa_v0_1: pl.Tile[
                    [32, 64], pl.FP32, pl.MemRef(mem_vec_18, pl.const(49152, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1934__ssa_v0, 64])
                ] = pl.tile.col_expand_mul(t__tmp_v95_1, gamma_kv_chunk_tail_inline1901__ssa_v0_1)
                kv_normed_bf16_tail_inline1944__ssa_v0_1: pl.Tile[
                    [32, 64], pl.BF16, pl.MemRef(mem_vec_18, pl.const(49152, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1934__ssa_v0, 64])
                ] = pl.tile.cast(kv_normed_tail_inline1900__ssa_v0_1, target_type=pl.BF16, mode="rint")
                kv_normed_valid_inline1899__ssa_v0_1: pl.Tile[
                    [32, 64], pl.BF16, pl.MemRef(mem_vec_18, pl.const(49152, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1934__ssa_v0, 64])
                ] = pl.tile.set_validshape(kv_normed_bf16_tail_inline1944__ssa_v0_1, valid_rows_inline1934__ssa_v0, 64)
                kv_view_inline1962__store_1: pl.Tensor[[t_dim_inline1931__ssa_v0, 512], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                    kv_normed_valid_inline1899__ssa_v0_1, [out_tg_inline1969__ssa_v0, n0_tail_inline1975__idx_v0 + 64], kv_view_inline1962__ssa_v0
                )
            kv_chunk_tail_inline1909__ssa_v1_2: pl.Tile[
                [32, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(8192, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1934__ssa_v0, 64])
            ] = pl.tile.load(kv_fp32_inline1938__rv_v10, [tg_inline1968__ssa_v0, 384], [32, 64], [valid_rows_inline1934__ssa_v0, 64], target_memory=pl.Mem.Vec)
            t__tmp_v94_2: pl.Tile[[32, 64], pl.BF16, pl.MemRef(mem_vec_18, pl.const(49152, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1934__ssa_v0, 64])] = pl.tile.cast(
                kv_chunk_tail_inline1909__ssa_v1_2, target_type=pl.BF16, mode="rint"
            )
            kv_chunk_tail_v2_inline1932__ssa_v0_2: pl.Tile[
                [32, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(8192, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1934__ssa_v0, 64])
            ] = pl.tile.cast(t__tmp_v94_2, target_type=pl.FP32, mode="round")
            gamma_kv_input_tail_inline1903__ssa_v0_2: pl.Tile[[64], pl.BF16, pl.MemRef(mem_vec_11, pl.const(0, pl.INT64), 128), pl.Mem.Vec] = pl.tile.load(
                gamma_ckv__ssa_v0, [384], [64], [64], target_memory=pl.Mem.Vec
            )
            gamma_kv_cast_tail_inline1902__ssa_v0_2: pl.Tile[[64], pl.FP32, pl.MemRef(mem_vec_18, pl.const(49152, pl.INT64), 256), pl.Mem.Vec] = pl.tile.cast(
                gamma_kv_input_tail_inline1903__ssa_v0_2, target_type=pl.FP32, mode="round"
            )
            gamma_kv_chunk_tail_inline1901__ssa_v0_2: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_18, pl.const(49152, pl.INT64), 256), pl.Mem.Vec] = gamma_kv_cast_tail_inline1902__ssa_v0_2
            t__tmp_v95_2: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(8192, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1934__ssa_v0, 64])] = (
                pl.tile.row_expand_mul(kv_chunk_tail_v2_inline1932__ssa_v0_2, kv_inv_rms_t_tail_inline1904__ssa_v0)
            )
            kv_normed_tail_inline1900__ssa_v0_2: pl.Tile[
                [32, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(8192, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1934__ssa_v0, 64])
            ] = pl.tile.col_expand_mul(t__tmp_v95_2, gamma_kv_chunk_tail_inline1901__ssa_v0_2)
            kv_normed_bf16_tail_inline1944__ssa_v0_2: pl.Tile[
                [32, 64], pl.BF16, pl.MemRef(mem_vec_12, pl.const(8192, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1934__ssa_v0, 64])
            ] = pl.tile.cast(kv_normed_tail_inline1900__ssa_v0_2, target_type=pl.BF16, mode="rint")
            kv_normed_valid_inline1899__ssa_v0_2: pl.Tile[
                [32, 64], pl.BF16, pl.MemRef(mem_vec_12, pl.const(8192, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1934__ssa_v0, 64])
            ] = pl.tile.set_validshape(kv_normed_bf16_tail_inline1944__ssa_v0_2, valid_rows_inline1934__ssa_v0, 64)
            kv_view_inline1962__store_2: pl.Tensor[[t_dim_inline1931__ssa_v0, 512], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                kv_normed_valid_inline1899__ssa_v0_2, [out_tg_inline1969__ssa_v0, 384], kv_view_inline1962__ssa_v0
            )
            gamma_rope_input_tail_inline1926__ssa_v0: pl.Tile[[64], pl.BF16, pl.MemRef(mem_vec_12, pl.const(8192, pl.INT64), 128), pl.Mem.Vec] = pl.tile.load(
                gamma_ckv__ssa_v0, [448], [64], [64], target_memory=pl.Mem.Vec
            )
            gamma_rope_cast_tail_inline1971__ssa_v0: pl.Tile[[64], pl.FP32, pl.MemRef(mem_vec_11, pl.const(0, pl.INT64), 256), pl.Mem.Vec] = pl.tile.cast(
                gamma_rope_input_tail_inline1926__ssa_v0, target_type=pl.FP32, mode="round"
            )
            gamma_rope_tail_inline1898__ssa_v0: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_11, pl.const(0, pl.INT64), 256), pl.Mem.Vec] = gamma_rope_cast_tail_inline1971__ssa_v0
            kv_rope_chunk_tail_inline1897__ssa_v0: pl.Tile[
                [32, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(8192, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1934__ssa_v0, 64])
            ] = pl.tile.load(kv_fp32_inline1938__rv_v10, [tg_inline1968__ssa_v0, 448], [32, 64], [valid_rows_inline1934__ssa_v0, 64], target_memory=pl.Mem.Vec)
            t__tmp_v96: pl.Tile[[32, 64], pl.BF16, pl.MemRef(mem_vec_18, pl.const(49152, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1934__ssa_v0, 64])] = pl.tile.cast(
                kv_rope_chunk_tail_inline1897__ssa_v0, target_type=pl.BF16, mode="rint"
            )
            kv_rope_chunk_tail_v1_inline1935__ssa_v0: pl.Tile[
                [32, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(8192, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1934__ssa_v0, 64])
            ] = pl.tile.cast(t__tmp_v96, target_type=pl.FP32, mode="round")
            t__tmp_v97: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(8192, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1934__ssa_v0, 64])] = (
                pl.tile.row_expand_mul(kv_rope_chunk_tail_v1_inline1935__ssa_v0, kv_inv_rms_t_tail_inline1904__ssa_v0)
            )
            kv_rope_norm_tail_inline1896__ssa_v0: pl.Tile[
                [32, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(8192, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1934__ssa_v0, 64])
            ] = pl.tile.col_expand_mul(t__tmp_v97, gamma_rope_tail_inline1898__ssa_v0)
            t__tmp_v98: pl.Tile[[32, 64], pl.BF16, pl.MemRef(mem_vec_18, pl.const(49152, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1934__ssa_v0, 64])] = pl.tile.cast(
                kv_rope_norm_tail_inline1896__ssa_v0, target_type=pl.BF16, mode="rint"
            )
            kv_rope_norm_tail_v1_inline1917__ssa_v0: pl.Tile[
                [32, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(8192, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1934__ssa_v0, 64])
            ] = pl.tile.cast(t__tmp_v98, target_type=pl.FP32, mode="round")
            kv_cos_il_tail_inline1937__ssa_v0: pl.Tile[
                [32, 64], pl.FP32, pl.MemRef(mem_vec_18, pl.const(49152, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1934__ssa_v0, 64])
            ] = pl.tile.load(q_rope_cos_il_inline545__ssa_v0, [out_tg_inline1969__ssa_v0, 0], [32, 64], [valid_rows_inline1934__ssa_v0, 64], target_memory=pl.Mem.Vec)
            kv_sin_signed_tail_inline1942__ssa_v0: pl.Tile[
                [32, 64], pl.FP32, pl.MemRef(mem_vec_11, pl.const(0, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1934__ssa_v0, 64])
            ] = pl.tile.load(q_rope_sin_signed_inline544__ssa_v0, [out_tg_inline1969__ssa_v0, 0], [32, 64], [valid_rows_inline1934__ssa_v0, 64], target_memory=pl.Mem.Vec)
            t__tmp_v99: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_17, pl.const(40960, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.full([32, 64], dtype=pl.FP32, value=1.0)
            t__ci_tmp_v0: pl.Tile[[1, 192], pl.FP32, pl.MemRef(mem_vec_64, pl.const(65536, pl.INT64), 768), pl.Mem.Vec] = pl.tile.create([1, 192], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            t__tmp_v100: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_145, pl.const(24576, pl.INT64), 256), pl.Mem.Vec] = pl.tile.ci(
                pl.const(0, pl.INT32), [1, 64], tmp=t__ci_tmp_v0, dtype=pl.INT32, descending=False
            )
            t__tmp_v101: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_64, pl.const(65536, pl.INT64), 256), pl.Mem.Vec] = pl.tile.cast(t__tmp_v100, target_type=pl.FP32, mode="round")
            kv_col_inline1912__ssa_v0: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_17, pl.const(40960, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.col_expand_mul(t__tmp_v99, t__tmp_v101)
            t__tmp_v102: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_64, pl.const(65536, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.muls(kv_col_inline1912__ssa_v0, 0.5)
            t__tmp_v103: pl.Tile[[32, 64], pl.INT32, pl.MemRef(mem_vec_64, pl.const(65536, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.cast(t__tmp_v102, target_type=pl.INT32, mode="trunc")
            kv_dup_f_inline1895__ssa_v0: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_64, pl.const(65536, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.cast(t__tmp_v103, target_type=pl.FP32, mode="round")
            t__tmp_v104: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_64, pl.const(65536, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.muls(kv_dup_f_inline1895__ssa_v0, 2.0)
            kv_lane_inline1979__ssa_v0: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_64, pl.const(65536, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.sub(kv_col_inline1912__ssa_v0, t__tmp_v104)
            t__tmp_v105: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_17, pl.const(40960, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.adds(kv_col_inline1912__ssa_v0, 1.0)
            t__tmp_v106: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_64, pl.const(65536, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.muls(kv_lane_inline1979__ssa_v0, 2.0)
            kv_swap_f_inline1894__ssa_v0: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_17, pl.const(40960, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.sub(t__tmp_v105, t__tmp_v106)
            t__ci_tmp_v1: pl.Tile[[1, 192], pl.FP32, pl.MemRef(mem_vec_64, pl.const(65536, pl.INT64), 768), pl.Mem.Vec] = pl.tile.create([1, 192], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            t__tmp_v107: pl.Tile[[1, 32], pl.INT32, pl.MemRef(mem_vec_145, pl.const(24576, pl.INT64), 128), pl.Mem.Vec] = pl.tile.ci(
                pl.const(0, pl.INT32), [1, 32], tmp=t__ci_tmp_v1, dtype=pl.INT32, descending=False
            )
            t__tmp_v108: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_64, pl.const(65536, pl.INT64), 128), pl.Mem.Vec] = pl.tile.cast(t__tmp_v107, target_type=pl.FP32, mode="round")
            kv_row_seed_inline1893__ssa_v0: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_145, pl.const(24576, pl.INT64), 128), pl.Mem.Vec] = pl.tile.muls(t__tmp_v108, 64.0)
            t__tmp_v109: pl.Tile[[64, 32], pl.FP32, pl.MemRef(mem_vec_64, pl.const(65536, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.full([64, 32], dtype=pl.FP32, value=1.0)
            kv_row_grid_inline1892__ssa_v0: pl.Tile[[64, 32], pl.FP32, pl.MemRef(mem_vec_64, pl.const(65536, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.col_expand_mul(
                t__tmp_v109, kv_row_seed_inline1893__ssa_v0
            )
            transpose_tmp: pl.Tile[[64, 32], pl.FP32, pl.MemRef(mem_vec_145, pl.const(24576, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.create([64, 32], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            kv_row_offset_inline1891__ssa_v0: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_146, pl.const(32768, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.transpose(
                kv_row_grid_inline1892__ssa_v0, 0, 1, transpose_tmp
            )
            t__tmp_v110: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_17, pl.const(40960, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.add(kv_swap_f_inline1894__ssa_v0, kv_row_offset_inline1891__ssa_v0)
            kv_swap_idx_tail_inline1890__ssa_v0: pl.Tile[[32, 64], pl.INT32, pl.MemRef(mem_vec_17, pl.const(40960, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.cast(
                t__tmp_v110, target_type=pl.INT32, mode="round"
            )
            kv_gather_tmp_inline1889__ssa_v0: pl.Tile[[32, 64], pl.INT32, pl.MemRef(mem_vec_64, pl.const(65536, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.create(
                [32, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec
            )
            kv_swapped_tail_inline1888__ssa_v0: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_145, pl.const(24576, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.gather(
                kv_rope_norm_tail_v1_inline1917__ssa_v0, kv_swap_idx_tail_inline1890__ssa_v0, kv_gather_tmp_inline1889__ssa_v0
            )
            t__tmp_v111: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(8192, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1934__ssa_v0, 64])] = pl.tile.mul(
                kv_rope_norm_tail_v1_inline1917__ssa_v0, kv_cos_il_tail_inline1937__ssa_v0
            )
            t__tmp_v112: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_18, pl.const(49152, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.mul(
                kv_swapped_tail_inline1888__ssa_v0, kv_sin_signed_tail_inline1942__ssa_v0
            )
            kv_rope_rot_tail_inline1947__ssa_v0: pl.Tile[
                [32, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(8192, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1934__ssa_v0, 64])
            ] = pl.tile.add(t__tmp_v111, t__tmp_v112)
            kv_rope_i16_tail_inline1887__ssa_v0: pl.Tile[
                [32, 64], pl.BF16, pl.MemRef(mem_vec_12, pl.const(8192, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1934__ssa_v0, 64])
            ] = pl.tile.cast(kv_rope_rot_tail_inline1947__ssa_v0, target_type=pl.BF16, mode="rint")
            kv_rope_valid_inline1966__ssa_v0: pl.Tile[
                [32, 64], pl.BF16, pl.MemRef(mem_vec_12, pl.const(8192, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1934__ssa_v0, 64])
            ] = pl.tile.set_validshape(kv_rope_i16_tail_inline1887__ssa_v0, valid_rows_inline1934__ssa_v0, 64)
            kv_view_inline1962__store_v0: pl.Tensor[[t_dim_inline1931__ssa_v0, 512], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                kv_rope_valid_inline1966__ssa_v0, [out_tg_inline1969__ssa_v0, 448], kv_view_inline1962__ssa_v0
            )
        return kv_view_inline1962__ssa_v0

    @pl.function(type=pl.FunctionType.Spmd)
    def kv_rms_norm_rope_spmd(
        self,
        tile_rows_inline1923__ssa_v0: pl.Scalar[pl.INDEX],
        tile_base_inline1950__idx_v0: pl.Scalar[pl.INDEX],
        kv_fp32_inline1938__rv_v10: pl.Tensor[[t_matmul_inline1930__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)],
        kv_view_inline1962__ssa_v0: pl.InOut[pl.Tensor[[t_dim_inline1931__ssa_v0, 512], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)]],
        gamma_ckv__ssa_v0: pl.Tensor[[512], pl.BF16, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 1024)],
        q_rope_cos_il_inline545__ssa_v0: pl.Tensor[[t_dim_inline546__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)],
        q_rope_sin_signed_inline544__ssa_v0: pl.Tensor[[t_dim_inline546__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 0)],
        q_rope_swap_idx_inline543__ssa_v0: pl.Tensor[[t_dim_inline546__ssa_v0, 64], pl.INT32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 0)],
    ):
        pl.func_attr({"mx_tensor_views_blocked": True})
        kv_view_inline1962__ssa_v1: pl.Tensor[[t_dim_inline1931__ssa_v0, 512], pl.BF16, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 0)] = self.kv_rms_norm_rope(
            tile_rows_inline1923__ssa_v0,
            tile_base_inline1950__idx_v0,
            kv_fp32_inline1938__rv_v10,
            kv_view_inline1962__ssa_v0,
            gamma_ckv__ssa_v0,
            q_rope_cos_il_inline545__ssa_v0,
            q_rope_sin_signed_inline544__ssa_v0,
            q_rope_swap_idx_inline543__ssa_v0,
            attrs={"arg_directions": [pl.adir.scalar, pl.adir.scalar, pl.adir.input, pl.adir.inout, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input]},
        )

    @pl.function(type=pl.FunctionType.AIC, level=pl.Level.AIC, role=pl.Role.SubWorker)
    def kv_score_proj(
        cmp4_kv_proj_pad_inline722_inline2006__ssa_v0: pl.Out[pl.Tensor[[384, 1024], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 1572864)]],
        cmp4_score_proj_pad_inline729_inline2003__ssa_v0: pl.Out[pl.Tensor[[384, 1024], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 1572864)]],
        t_matmul_inline720_inline2020__ssa_v0: pl.Scalar[pl.INDEX],
        bs_inline719_inline2040__ssa_v0: pl.Scalar[pl.INDEX],
        x_flat_inline721_inline2001__ssa_v0: pl.Tensor[[T_DYN, 4096], pl.BF16, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)],
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
        kv_worker_inline728_inline2012__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        for idx_inline727_inline1994__idx_v0, (cmp4_kv_proj_pad_inline722_inline2006__iter_v1, cmp4_score_proj_pad_inline729_inline2003__iter_v1) in pl.range(
            kv_worker_inline728_inline2012__ssa_v0,
            t_matmul_inline720_inline2020__ssa_v0 // 4,
            24,
            init_values=(cmp4_kv_proj_pad_inline722_inline2006__ssa_v0, cmp4_score_proj_pad_inline729_inline2003__ssa_v0),
        ):
            global_row0_inline723_inline2014__ssa_v0: pl.Scalar[pl.INDEX] = idx_inline727_inline1994__idx_v0 // 16 * 64
            o0_inline732_inline2018__ssa_v0: pl.Scalar[pl.INDEX] = idx_inline727_inline1994__idx_v0 % 16 * 64
            x_rows_inline725_inline2038__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(bs_inline719_inline2040__ssa_v0 - global_row0_inline723_inline2014__ssa_v0, 64)
            kv_acc_inline730_inline2007__tile: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_5, pl.const(0, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.create(
                [64, 64], dtype=pl.FP32, target_memory=pl.Mem.Vec
            )
            score_acc_inline726_inline2021__tile: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_5, pl.const(0, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.create(
                [64, 64], dtype=pl.FP32, target_memory=pl.Mem.Vec
            )
            kv_acc_inline730_inline2007__tile_narrowed_storage: pl.Tile[
                [64, 64], pl.FP32, pl.MemRef(mem_acc_7, pl.const(0, pl.INT64), 16384), pl.Mem.Acc, pl.TileView(compact=pl.CompactMode.normal)
            ] = pl.tile.create([64, 64], dtype=pl.FP32, target_memory=pl.Mem.Acc, compact=True)
            kv_acc_inline730_inline2007__tile_narrowed: pl.Tile[
                [64, 64], pl.FP32, pl.MemRef(mem_acc_7, pl.const(0, pl.INT64), 16384), pl.Mem.Acc, pl.TileView(valid_shape=[x_rows_inline725_inline2038__ssa_v0, 64], compact=pl.CompactMode.normal)
            ] = pl.tile.set_validshape(kv_acc_inline730_inline2007__tile_narrowed_storage, x_rows_inline725_inline2038__ssa_v0, 64)
            score_acc_inline726_inline2021__tile_narrowed_storage: pl.Tile[
                [64, 64], pl.FP32, pl.MemRef(mem_acc_8, pl.const(16384, pl.INT64), 16384), pl.Mem.Acc, pl.TileView(compact=pl.CompactMode.normal)
            ] = pl.tile.create([64, 64], dtype=pl.FP32, target_memory=pl.Mem.Acc, compact=True)
            score_acc_inline726_inline2021__tile_narrowed: pl.Tile[
                [64, 64], pl.FP32, pl.MemRef(mem_acc_8, pl.const(16384, pl.INT64), 16384), pl.Mem.Acc, pl.TileView(valid_shape=[x_rows_inline725_inline2038__ssa_v0, 64], compact=pl.CompactMode.normal)
            ] = pl.tile.set_validshape(score_acc_inline726_inline2021__tile_narrowed_storage, x_rows_inline725_inline2038__ssa_v0, 64)
            for kb_inline717_inline2002__idx_v0, (kv_acc_inline730_inline2007__iter_v1, score_acc_inline726_inline2021__iter_v1) in pl.range(
                0, 8, 2, init_values=(kv_acc_inline730_inline2007__tile_narrowed, score_acc_inline726_inline2021__tile_narrowed)
            ):
                k0_inline716_inline2024__ssa_v0: pl.Scalar[pl.INDEX] = kb_inline717_inline2002__idx_v0 * 512
                k0_inline716_inline2024__ssa_v0_1: pl.Scalar[pl.INDEX] = kb_inline717_inline2002__idx_v0 * 512 + 512
                x_tile_inline731_inline1999__tile: pl.Tile[
                    [64, 512], pl.BF16, pl.MemRef(mem_mat_9, pl.const(327680, pl.INT64), 65536), pl.Mem.Mat, pl.TileView(valid_shape=[x_rows_inline725_inline2038__ssa_v0, 512])
                ] = pl.tile.load(
                    x_flat_inline721_inline2001__ssa_v0,
                    [global_row0_inline723_inline2014__ssa_v0, k0_inline716_inline2024__ssa_v0],
                    [64, 512],
                    [x_rows_inline725_inline2038__ssa_v0, 512],
                    target_memory=pl.Mem.Mat,
                )
                wkv_tile_inline718_inline2004__tile: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_mat_10, pl.const(0, pl.INT64), 65536), pl.Mem.Mat] = pl.tile.load(
                    cmp_wkv__ssa_v0, [o0_inline732_inline2018__ssa_v0, k0_inline716_inline2024__ssa_v0], [64, 512], [64, 512], target_memory=pl.Mem.Mat
                )
                wgate_tile_inline715_inline2016__tile: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_mat_11, pl.const(65536, pl.INT64), 65536), pl.Mem.Mat] = pl.tile.load(
                    cmp_wgate__ssa_v0, [o0_inline732_inline2018__ssa_v0, k0_inline716_inline2024__ssa_v0], [64, 512], [64, 512], target_memory=pl.Mem.Mat
                )
                x_tile_inline731_inline1999__tile_1: pl.Tile[
                    [64, 512], pl.BF16, pl.MemRef(mem_mat_12, pl.const(131072, pl.INT64), 65536), pl.Mem.Mat, pl.TileView(valid_shape=[x_rows_inline725_inline2038__ssa_v0, 512])
                ] = pl.tile.load(
                    x_flat_inline721_inline2001__ssa_v0,
                    [global_row0_inline723_inline2014__ssa_v0, k0_inline716_inline2024__ssa_v0_1],
                    [64, 512],
                    [x_rows_inline725_inline2038__ssa_v0, 512],
                    target_memory=pl.Mem.Mat,
                )
                wkv_tile_inline718_inline2004__tile_1: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_mat_13, pl.const(196608, pl.INT64), 65536), pl.Mem.Mat] = pl.tile.load(
                    cmp_wkv__ssa_v0, [o0_inline732_inline2018__ssa_v0, k0_inline716_inline2024__ssa_v0_1], [64, 512], [64, 512], target_memory=pl.Mem.Mat
                )
                wgate_tile_inline715_inline2016__tile_1: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_mat_14, pl.const(262144, pl.INT64), 65536), pl.Mem.Mat] = pl.tile.load(
                    cmp_wgate__ssa_v0, [o0_inline732_inline2018__ssa_v0, k0_inline716_inline2024__ssa_v0_1], [64, 512], [64, 512], target_memory=pl.Mem.Mat
                )
                if k0_inline716_inline2024__ssa_v0 == 0:
                    wkv_tile_inline718_inline2004__tile_t: pl.Tile[
                        [512, 64], pl.BF16, pl.MemRef(mem_mat_10, pl.const(0, pl.INT64), 65536), pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)
                    ] = pl.tile.transpose_view(wkv_tile_inline718_inline2004__tile)
                    kv_acc_inline730_inline2007__tile_l0_init_storage: pl.Tile[
                        [64, 64], pl.FP32, pl.MemRef(mem_acc_7, pl.const(0, pl.INT64), 16384), pl.Mem.Acc, pl.TileView(compact=pl.CompactMode.normal)
                    ] = pl.tile.create([64, 64], dtype=pl.FP32, target_memory=pl.Mem.Acc, compact=True)
                    kv_acc_inline730_inline2007__tile_l0_init: pl.Tile[
                        [64, 64],
                        pl.FP32,
                        pl.MemRef(mem_acc_7, pl.const(0, pl.INT64), 16384),
                        pl.Mem.Acc,
                        pl.TileView(valid_shape=[x_rows_inline725_inline2038__ssa_v0, 64], compact=pl.CompactMode.normal),
                    ] = pl.tile.set_validshape(kv_acc_inline730_inline2007__tile_l0_init_storage, x_rows_inline725_inline2038__ssa_v0, 64)
                    kv_acc_inline730_inline2007__tile_l0_a: pl.Tile[
                        [64, 256],
                        pl.BF16,
                        pl.MemRef(mem_left_16, pl.const(0, pl.INT64), 32768),
                        pl.Mem.Left,
                        pl.TileView(valid_shape=[x_rows_inline725_inline2038__ssa_v0, 256], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal),
                    ] = pl.tile.extract(x_tile_inline731_inline1999__tile, 0, 0, [64, 256], target_memory=pl.Mem.Left)
                    kv_acc_inline730_inline2007__tile_l0_b: pl.Tile[[256, 64], pl.BF16, pl.MemRef(mem_right_17, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                        wkv_tile_inline718_inline2004__tile_t, 0, 0, [256, 64], target_memory=pl.Mem.Right
                    )
                    kv_acc_inline730_inline2007__tile_l0_a_1: pl.Tile[
                        [64, 256],
                        pl.BF16,
                        pl.MemRef(mem_left_18, pl.const(32768, pl.INT64), 32768),
                        pl.Mem.Left,
                        pl.TileView(valid_shape=[x_rows_inline725_inline2038__ssa_v0, 256], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal),
                    ] = pl.tile.extract(x_tile_inline731_inline1999__tile, 0, 256, [64, 256], target_memory=pl.Mem.Left)
                    kv_acc_inline730_inline2007__tile_l0_b_1: pl.Tile[[256, 64], pl.BF16, pl.MemRef(mem_right_19, pl.const(32768, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                        wkv_tile_inline718_inline2004__tile_t, 256, 0, [256, 64], target_memory=pl.Mem.Right
                    )
                    kv_acc_inline730_inline2007__tile_l0_c_acc: pl.Tile[
                        [64, 64],
                        pl.FP32,
                        pl.MemRef(mem_acc_7, pl.const(0, pl.INT64), 16384),
                        pl.Mem.Acc,
                        pl.TileView(valid_shape=[x_rows_inline725_inline2038__ssa_v0, 64], compact=pl.CompactMode.normal),
                    ] = pl.tile.matmul_acc(kv_acc_inline730_inline2007__tile_l0_init, kv_acc_inline730_inline2007__tile_l0_a, kv_acc_inline730_inline2007__tile_l0_b, True)
                    kv_acc_inline730_inline2007__tile_l0_c_acc_1: pl.Tile[
                        [64, 64],
                        pl.FP32,
                        pl.MemRef(mem_acc_7, pl.const(0, pl.INT64), 16384),
                        pl.Mem.Acc,
                        pl.TileView(valid_shape=[x_rows_inline725_inline2038__ssa_v0, 64], compact=pl.CompactMode.normal),
                    ] = pl.tile.matmul_acc(kv_acc_inline730_inline2007__tile_l0_c_acc, kv_acc_inline730_inline2007__tile_l0_a_1, kv_acc_inline730_inline2007__tile_l0_b_1, False)
                    wgate_tile_inline715_inline2016__tile_t: pl.Tile[
                        [512, 64], pl.BF16, pl.MemRef(mem_mat_11, pl.const(65536, pl.INT64), 65536), pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)
                    ] = pl.tile.transpose_view(wgate_tile_inline715_inline2016__tile)
                    score_acc_inline726_inline2021__tile_l0_init_storage: pl.Tile[
                        [64, 64], pl.FP32, pl.MemRef(mem_acc_8, pl.const(16384, pl.INT64), 16384), pl.Mem.Acc, pl.TileView(compact=pl.CompactMode.normal)
                    ] = pl.tile.create([64, 64], dtype=pl.FP32, target_memory=pl.Mem.Acc, compact=True)
                    score_acc_inline726_inline2021__tile_l0_init: pl.Tile[
                        [64, 64],
                        pl.FP32,
                        pl.MemRef(mem_acc_8, pl.const(16384, pl.INT64), 16384),
                        pl.Mem.Acc,
                        pl.TileView(valid_shape=[x_rows_inline725_inline2038__ssa_v0, 64], compact=pl.CompactMode.normal),
                    ] = pl.tile.set_validshape(score_acc_inline726_inline2021__tile_l0_init_storage, x_rows_inline725_inline2038__ssa_v0, 64)
                    score_acc_inline726_inline2021__tile_l0_a: pl.Tile[
                        [64, 256],
                        pl.BF16,
                        pl.MemRef(mem_left_16, pl.const(0, pl.INT64), 32768),
                        pl.Mem.Left,
                        pl.TileView(valid_shape=[x_rows_inline725_inline2038__ssa_v0, 256], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal),
                    ] = pl.tile.extract(x_tile_inline731_inline1999__tile, 0, 0, [64, 256], target_memory=pl.Mem.Left)
                    score_acc_inline726_inline2021__tile_l0_b: pl.Tile[[256, 64], pl.BF16, pl.MemRef(mem_right_17, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                        wgate_tile_inline715_inline2016__tile_t, 0, 0, [256, 64], target_memory=pl.Mem.Right
                    )
                    score_acc_inline726_inline2021__tile_l0_a_1: pl.Tile[
                        [64, 256],
                        pl.BF16,
                        pl.MemRef(mem_left_18, pl.const(32768, pl.INT64), 32768),
                        pl.Mem.Left,
                        pl.TileView(valid_shape=[x_rows_inline725_inline2038__ssa_v0, 256], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal),
                    ] = pl.tile.extract(x_tile_inline731_inline1999__tile, 0, 256, [64, 256], target_memory=pl.Mem.Left)
                    score_acc_inline726_inline2021__tile_l0_b_1: pl.Tile[[256, 64], pl.BF16, pl.MemRef(mem_right_19, pl.const(32768, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                        wgate_tile_inline715_inline2016__tile_t, 256, 0, [256, 64], target_memory=pl.Mem.Right
                    )
                    score_acc_inline726_inline2021__tile_l0_c_acc: pl.Tile[
                        [64, 64],
                        pl.FP32,
                        pl.MemRef(mem_acc_8, pl.const(16384, pl.INT64), 16384),
                        pl.Mem.Acc,
                        pl.TileView(valid_shape=[x_rows_inline725_inline2038__ssa_v0, 64], compact=pl.CompactMode.normal),
                    ] = pl.tile.matmul_acc(score_acc_inline726_inline2021__tile_l0_init, score_acc_inline726_inline2021__tile_l0_a, score_acc_inline726_inline2021__tile_l0_b, True)
                    score_acc_inline726_inline2021__tile_l0_c_acc_1: pl.Tile[
                        [64, 64],
                        pl.FP32,
                        pl.MemRef(mem_acc_8, pl.const(16384, pl.INT64), 16384),
                        pl.Mem.Acc,
                        pl.TileView(valid_shape=[x_rows_inline725_inline2038__ssa_v0, 64], compact=pl.CompactMode.normal),
                    ] = pl.tile.matmul_acc(score_acc_inline726_inline2021__tile_l0_c_acc, score_acc_inline726_inline2021__tile_l0_a_1, score_acc_inline726_inline2021__tile_l0_b_1, False)
                    kv_acc_inline730_inline2007__phi_v5, score_acc_inline726_inline2021__phi_v5 = pl.yield_(
                        kv_acc_inline730_inline2007__tile_l0_c_acc_1, score_acc_inline726_inline2021__tile_l0_c_acc_1
                    )
                else:
                    wkv_tile_inline718_inline2004__tile_t_1: pl.Tile[
                        [512, 64], pl.BF16, pl.MemRef(mem_mat_10, pl.const(0, pl.INT64), 65536), pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)
                    ] = pl.tile.transpose_view(wkv_tile_inline718_inline2004__tile)
                    kv_acc_inline730_inline2007__tile_l0_a_2: pl.Tile[
                        [64, 256],
                        pl.BF16,
                        pl.MemRef(mem_left_16, pl.const(0, pl.INT64), 32768),
                        pl.Mem.Left,
                        pl.TileView(valid_shape=[x_rows_inline725_inline2038__ssa_v0, 256], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal),
                    ] = pl.tile.extract(x_tile_inline731_inline1999__tile, 0, 0, [64, 256], target_memory=pl.Mem.Left)
                    kv_acc_inline730_inline2007__tile_l0_b_2: pl.Tile[[256, 64], pl.BF16, pl.MemRef(mem_right_17, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                        wkv_tile_inline718_inline2004__tile_t_1, 0, 0, [256, 64], target_memory=pl.Mem.Right
                    )
                    kv_acc_inline730_inline2007__tile_l0_a_3: pl.Tile[
                        [64, 256],
                        pl.BF16,
                        pl.MemRef(mem_left_18, pl.const(32768, pl.INT64), 32768),
                        pl.Mem.Left,
                        pl.TileView(valid_shape=[x_rows_inline725_inline2038__ssa_v0, 256], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal),
                    ] = pl.tile.extract(x_tile_inline731_inline1999__tile, 0, 256, [64, 256], target_memory=pl.Mem.Left)
                    kv_acc_inline730_inline2007__tile_l0_b_3: pl.Tile[[256, 64], pl.BF16, pl.MemRef(mem_right_19, pl.const(32768, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                        wkv_tile_inline718_inline2004__tile_t_1, 256, 0, [256, 64], target_memory=pl.Mem.Right
                    )
                    kv_acc_inline730_inline2007__tile_l0_c_acc_2: pl.Tile[
                        [64, 64],
                        pl.FP32,
                        pl.MemRef(mem_acc_7, pl.const(0, pl.INT64), 16384),
                        pl.Mem.Acc,
                        pl.TileView(valid_shape=[x_rows_inline725_inline2038__ssa_v0, 64], compact=pl.CompactMode.normal),
                    ] = pl.tile.matmul_acc(kv_acc_inline730_inline2007__iter_v1, kv_acc_inline730_inline2007__tile_l0_a_2, kv_acc_inline730_inline2007__tile_l0_b_2)
                    kv_acc_inline730_inline2007__tile_l0_c_acc_3: pl.Tile[
                        [64, 64],
                        pl.FP32,
                        pl.MemRef(mem_acc_7, pl.const(0, pl.INT64), 16384),
                        pl.Mem.Acc,
                        pl.TileView(valid_shape=[x_rows_inline725_inline2038__ssa_v0, 64], compact=pl.CompactMode.normal),
                    ] = pl.tile.matmul_acc(kv_acc_inline730_inline2007__tile_l0_c_acc_2, kv_acc_inline730_inline2007__tile_l0_a_3, kv_acc_inline730_inline2007__tile_l0_b_3)
                    wgate_tile_inline715_inline2016__tile_t_1: pl.Tile[
                        [512, 64], pl.BF16, pl.MemRef(mem_mat_11, pl.const(65536, pl.INT64), 65536), pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)
                    ] = pl.tile.transpose_view(wgate_tile_inline715_inline2016__tile)
                    score_acc_inline726_inline2021__tile_l0_a_2: pl.Tile[
                        [64, 256],
                        pl.BF16,
                        pl.MemRef(mem_left_16, pl.const(0, pl.INT64), 32768),
                        pl.Mem.Left,
                        pl.TileView(valid_shape=[x_rows_inline725_inline2038__ssa_v0, 256], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal),
                    ] = pl.tile.extract(x_tile_inline731_inline1999__tile, 0, 0, [64, 256], target_memory=pl.Mem.Left)
                    score_acc_inline726_inline2021__tile_l0_b_2: pl.Tile[[256, 64], pl.BF16, pl.MemRef(mem_right_17, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                        wgate_tile_inline715_inline2016__tile_t_1, 0, 0, [256, 64], target_memory=pl.Mem.Right
                    )
                    score_acc_inline726_inline2021__tile_l0_a_3: pl.Tile[
                        [64, 256],
                        pl.BF16,
                        pl.MemRef(mem_left_18, pl.const(32768, pl.INT64), 32768),
                        pl.Mem.Left,
                        pl.TileView(valid_shape=[x_rows_inline725_inline2038__ssa_v0, 256], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal),
                    ] = pl.tile.extract(x_tile_inline731_inline1999__tile, 0, 256, [64, 256], target_memory=pl.Mem.Left)
                    score_acc_inline726_inline2021__tile_l0_b_3: pl.Tile[[256, 64], pl.BF16, pl.MemRef(mem_right_19, pl.const(32768, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                        wgate_tile_inline715_inline2016__tile_t_1, 256, 0, [256, 64], target_memory=pl.Mem.Right
                    )
                    score_acc_inline726_inline2021__tile_l0_c_acc_2: pl.Tile[
                        [64, 64],
                        pl.FP32,
                        pl.MemRef(mem_acc_8, pl.const(16384, pl.INT64), 16384),
                        pl.Mem.Acc,
                        pl.TileView(valid_shape=[x_rows_inline725_inline2038__ssa_v0, 64], compact=pl.CompactMode.normal),
                    ] = pl.tile.matmul_acc(score_acc_inline726_inline2021__iter_v1, score_acc_inline726_inline2021__tile_l0_a_2, score_acc_inline726_inline2021__tile_l0_b_2)
                    score_acc_inline726_inline2021__tile_l0_c_acc_3: pl.Tile[
                        [64, 64],
                        pl.FP32,
                        pl.MemRef(mem_acc_8, pl.const(16384, pl.INT64), 16384),
                        pl.Mem.Acc,
                        pl.TileView(valid_shape=[x_rows_inline725_inline2038__ssa_v0, 64], compact=pl.CompactMode.normal),
                    ] = pl.tile.matmul_acc(score_acc_inline726_inline2021__tile_l0_c_acc_2, score_acc_inline726_inline2021__tile_l0_a_3, score_acc_inline726_inline2021__tile_l0_b_3)
                    kv_acc_inline730_inline2007__phi_v5, score_acc_inline726_inline2021__phi_v5 = pl.yield_(
                        kv_acc_inline730_inline2007__tile_l0_c_acc_3, score_acc_inline726_inline2021__tile_l0_c_acc_3
                    )
                wkv_tile_inline718_inline2004__tile_t_2: pl.Tile[
                    [512, 64], pl.BF16, pl.MemRef(mem_mat_13, pl.const(196608, pl.INT64), 65536), pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)
                ] = pl.tile.transpose_view(wkv_tile_inline718_inline2004__tile_1)
                kv_acc_inline730_inline2007__tile_l0_a_4: pl.Tile[
                    [64, 256],
                    pl.BF16,
                    pl.MemRef(mem_left_16, pl.const(0, pl.INT64), 32768),
                    pl.Mem.Left,
                    pl.TileView(valid_shape=[x_rows_inline725_inline2038__ssa_v0, 256], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal),
                ] = pl.tile.extract(x_tile_inline731_inline1999__tile_1, 0, 0, [64, 256], target_memory=pl.Mem.Left)
                kv_acc_inline730_inline2007__tile_l0_b_4: pl.Tile[[256, 64], pl.BF16, pl.MemRef(mem_right_17, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                    wkv_tile_inline718_inline2004__tile_t_2, 0, 0, [256, 64], target_memory=pl.Mem.Right
                )
                kv_acc_inline730_inline2007__tile_l0_a_5: pl.Tile[
                    [64, 256],
                    pl.BF16,
                    pl.MemRef(mem_left_18, pl.const(32768, pl.INT64), 32768),
                    pl.Mem.Left,
                    pl.TileView(valid_shape=[x_rows_inline725_inline2038__ssa_v0, 256], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal),
                ] = pl.tile.extract(x_tile_inline731_inline1999__tile_1, 0, 256, [64, 256], target_memory=pl.Mem.Left)
                kv_acc_inline730_inline2007__tile_l0_b_5: pl.Tile[[256, 64], pl.BF16, pl.MemRef(mem_right_19, pl.const(32768, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                    wkv_tile_inline718_inline2004__tile_t_2, 256, 0, [256, 64], target_memory=pl.Mem.Right
                )
                kv_acc_inline730_inline2007__tile_l0_c_acc_4: pl.Tile[
                    [64, 64], pl.FP32, pl.MemRef(mem_acc_7, pl.const(0, pl.INT64), 16384), pl.Mem.Acc, pl.TileView(valid_shape=[x_rows_inline725_inline2038__ssa_v0, 64], compact=pl.CompactMode.normal)
                ] = pl.tile.matmul_acc(kv_acc_inline730_inline2007__phi_v5, kv_acc_inline730_inline2007__tile_l0_a_4, kv_acc_inline730_inline2007__tile_l0_b_4)
                kv_acc_inline730_inline2007__tile_l0_c_acc_5: pl.Tile[
                    [64, 64], pl.FP32, pl.MemRef(mem_acc_7, pl.const(0, pl.INT64), 16384), pl.Mem.Acc, pl.TileView(valid_shape=[x_rows_inline725_inline2038__ssa_v0, 64], compact=pl.CompactMode.normal)
                ] = pl.tile.matmul_acc(kv_acc_inline730_inline2007__tile_l0_c_acc_4, kv_acc_inline730_inline2007__tile_l0_a_5, kv_acc_inline730_inline2007__tile_l0_b_5)
                wgate_tile_inline715_inline2016__tile_t_2: pl.Tile[
                    [512, 64], pl.BF16, pl.MemRef(mem_mat_14, pl.const(262144, pl.INT64), 65536), pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)
                ] = pl.tile.transpose_view(wgate_tile_inline715_inline2016__tile_1)
                score_acc_inline726_inline2021__tile_l0_a_4: pl.Tile[
                    [64, 256],
                    pl.BF16,
                    pl.MemRef(mem_left_16, pl.const(0, pl.INT64), 32768),
                    pl.Mem.Left,
                    pl.TileView(valid_shape=[x_rows_inline725_inline2038__ssa_v0, 256], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal),
                ] = pl.tile.extract(x_tile_inline731_inline1999__tile_1, 0, 0, [64, 256], target_memory=pl.Mem.Left)
                score_acc_inline726_inline2021__tile_l0_b_4: pl.Tile[[256, 64], pl.BF16, pl.MemRef(mem_right_17, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                    wgate_tile_inline715_inline2016__tile_t_2, 0, 0, [256, 64], target_memory=pl.Mem.Right
                )
                score_acc_inline726_inline2021__tile_l0_a_5: pl.Tile[
                    [64, 256],
                    pl.BF16,
                    pl.MemRef(mem_left_18, pl.const(32768, pl.INT64), 32768),
                    pl.Mem.Left,
                    pl.TileView(valid_shape=[x_rows_inline725_inline2038__ssa_v0, 256], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal),
                ] = pl.tile.extract(x_tile_inline731_inline1999__tile_1, 0, 256, [64, 256], target_memory=pl.Mem.Left)
                score_acc_inline726_inline2021__tile_l0_b_5: pl.Tile[[256, 64], pl.BF16, pl.MemRef(mem_right_19, pl.const(32768, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                    wgate_tile_inline715_inline2016__tile_t_2, 256, 0, [256, 64], target_memory=pl.Mem.Right
                )
                score_acc_inline726_inline2021__tile_l0_c_acc_4: pl.Tile[
                    [64, 64],
                    pl.FP32,
                    pl.MemRef(mem_acc_8, pl.const(16384, pl.INT64), 16384),
                    pl.Mem.Acc,
                    pl.TileView(valid_shape=[x_rows_inline725_inline2038__ssa_v0, 64], compact=pl.CompactMode.normal),
                ] = pl.tile.matmul_acc(score_acc_inline726_inline2021__phi_v5, score_acc_inline726_inline2021__tile_l0_a_4, score_acc_inline726_inline2021__tile_l0_b_4)
                score_acc_inline726_inline2021__tile_l0_c_acc_5: pl.Tile[
                    [64, 64],
                    pl.FP32,
                    pl.MemRef(mem_acc_8, pl.const(16384, pl.INT64), 16384),
                    pl.Mem.Acc,
                    pl.TileView(valid_shape=[x_rows_inline725_inline2038__ssa_v0, 64], compact=pl.CompactMode.normal),
                ] = pl.tile.matmul_acc(score_acc_inline726_inline2021__tile_l0_c_acc_4, score_acc_inline726_inline2021__tile_l0_a_5, score_acc_inline726_inline2021__tile_l0_b_5)
                kv_acc_inline730_inline2007__rv_v2, score_acc_inline726_inline2021__rv_v2 = pl.yield_(kv_acc_inline730_inline2007__tile_l0_c_acc_5, score_acc_inline726_inline2021__tile_l0_c_acc_5)
            cmp4_kv_proj_pad_inline722_inline2006__tile: pl.Tensor[[384, 1024], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 1572864)] = pl.tile.store(
                kv_acc_inline730_inline2007__rv_v2, [global_row0_inline723_inline2014__ssa_v0, o0_inline732_inline2018__ssa_v0], cmp4_kv_proj_pad_inline722_inline2006__iter_v1
            )
            cmp4_score_proj_pad_inline729_inline2003__tile: pl.Tensor[[384, 1024], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 1572864)] = pl.tile.store(
                score_acc_inline726_inline2021__rv_v2, [global_row0_inline723_inline2014__ssa_v0, o0_inline732_inline2018__ssa_v0], cmp4_score_proj_pad_inline729_inline2003__iter_v1
            )
            cmp4_kv_proj_pad_inline722_inline2006__rv_v2, cmp4_score_proj_pad_inline729_inline2003__rv_v2 = pl.yield_(
                cmp4_kv_proj_pad_inline722_inline2006__tile, cmp4_score_proj_pad_inline729_inline2003__tile
            )
        return

    @pl.function(type=pl.FunctionType.Spmd)
    def kv_score_proj_spmd(
        self,
        cmp4_kv_proj_pad_inline722_inline2006__ssa_v0: pl.Out[pl.Tensor[[384, 1024], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 1572864)]],
        cmp4_score_proj_pad_inline729_inline2003__ssa_v0: pl.Out[pl.Tensor[[384, 1024], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 1572864)]],
        t_matmul_inline720_inline2020__ssa_v0: pl.Scalar[pl.INDEX],
        bs_inline719_inline2040__ssa_v0: pl.Scalar[pl.INDEX],
        x_flat_inline721_inline2001__ssa_v0: pl.Tensor[[T_DYN, 4096], pl.BF16, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)],
        cmp_wkv__ssa_v0: pl.Tensor[[1024, 4096], pl.BF16, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 8388608)],
        cmp_wgate__ssa_v0: pl.Tensor[[1024, 4096], pl.BF16, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 8388608)],
    ):
        pl.func_attr({"mx_tensor_views_blocked": True})
        self.kv_score_proj(
            cmp4_kv_proj_pad_inline722_inline2006__ssa_v0,
            cmp4_score_proj_pad_inline729_inline2003__ssa_v0,
            t_matmul_inline720_inline2020__ssa_v0,
            bs_inline719_inline2040__ssa_v0,
            x_flat_inline721_inline2001__ssa_v0,
            cmp_wkv__ssa_v0,
            cmp_wgate__ssa_v0,
            attrs={"arg_directions": [pl.adir.output_existing, pl.adir.output_existing, pl.adir.scalar, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.input]},
        )

    @pl.function(type=pl.FunctionType.AIC, level=pl.Level.AIC, role=pl.Role.SubWorker)
    def kv_score_proj_0(
        kv_proj_pad_inline2152__ssa_v0: pl.Out[pl.Tensor[[384, 256], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 393216)]],
        score_proj_pad_inline2156__ssa_v0: pl.Out[pl.Tensor[[384, 256], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 393216)]],
        t_matmul_inline415_inline2162__ssa_v0: pl.Scalar[pl.INDEX],
        bs_inline414_inline2160__ssa_v0: pl.Scalar[pl.INDEX],
        x_flat_inline416_inline2157__ssa_v0: pl.Tensor[[T_DYN, 4096], pl.BF16, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)],
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
        kv_worker_inline423_inline2168__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        for idx_inline419_inline2171__idx_v0, (kv_proj_pad_inline2152__iter_v1, score_proj_pad_inline2156__iter_v1) in pl.range(
            kv_worker_inline423_inline2168__ssa_v0, t_matmul_inline415_inline2162__ssa_v0 // 2, 24, init_values=(kv_proj_pad_inline2152__ssa_v0, score_proj_pad_inline2156__ssa_v0)
        ):
            global_row0_inline418_inline2169__ssa_v0: pl.Scalar[pl.INDEX] = idx_inline419_inline2171__idx_v0 // 8 * 16
            o0_inline421_inline2154__ssa_v0: pl.Scalar[pl.INDEX] = idx_inline419_inline2171__idx_v0 % 8 * 32
            kv_acc_inline422_inline2139__tile: pl.Tile[[16, 32], pl.FP32, pl.MemRef(mem_acc_5, pl.const(0, pl.INT64), 2048), pl.Mem.Acc] = pl.tile.create(
                [16, 32], dtype=pl.FP32, target_memory=pl.Mem.Acc
            )
            score_acc_inline424_inline2161__tile: pl.Tile[[16, 32], pl.FP32, pl.MemRef(mem_acc_6, pl.const(2048, pl.INT64), 2048), pl.Mem.Acc] = pl.tile.create(
                [16, 32], dtype=pl.FP32, target_memory=pl.Mem.Acc
            )
            for kb_inline425_inline2174__idx_v0, (kv_acc_inline422_inline2139__iter_v1, score_acc_inline424_inline2161__iter_v1) in pl.range(
                0, 8, 2, init_values=(kv_acc_inline422_inline2139__tile, score_acc_inline424_inline2161__tile)
            ):
                k0_inline420_inline2148__ssa_v0: pl.Scalar[pl.INDEX] = kb_inline425_inline2174__idx_v0 * 512
                x_rows_inline426_inline2158__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(bs_inline414_inline2160__ssa_v0 - global_row0_inline418_inline2169__ssa_v0, 16)
                k0_inline420_inline2148__ssa_v0_1: pl.Scalar[pl.INDEX] = kb_inline425_inline2174__idx_v0 * 512 + 512
                x_rows_inline426_inline2158__ssa_v0_1: pl.Scalar[pl.INDEX] = pl.min(bs_inline414_inline2160__ssa_v0 - global_row0_inline418_inline2169__ssa_v0, 16)
                x_tile_inline413_inline2170__tile: pl.Tile[
                    [16, 512], pl.BF16, pl.MemRef(mem_mat_7, pl.const(81920, pl.INT64), 16384), pl.Mem.Mat, pl.TileView(valid_shape=[x_rows_inline426_inline2158__ssa_v0, 512])
                ] = pl.tile.load(
                    x_flat_inline416_inline2157__ssa_v0,
                    [global_row0_inline418_inline2169__ssa_v0, k0_inline420_inline2148__ssa_v0],
                    [16, 512],
                    [x_rows_inline426_inline2158__ssa_v0, 512],
                    target_memory=pl.Mem.Mat,
                )
                wkv_tile_inline412_inline2167__tile: pl.Tile[[32, 512], pl.BF16, pl.MemRef(mem_mat_8, pl.const(98304, pl.INT64), 32768), pl.Mem.Mat] = pl.tile.load(
                    inner_wkv__ssa_v0, [o0_inline421_inline2154__ssa_v0, k0_inline420_inline2148__ssa_v0], [32, 512], [32, 512], target_memory=pl.Mem.Mat
                )
                wgate_tile_inline411_inline2151__tile: pl.Tile[[32, 512], pl.BF16, pl.MemRef(mem_mat_9, pl.const(131072, pl.INT64), 32768), pl.Mem.Mat] = pl.tile.load(
                    inner_wgate__ssa_v0, [o0_inline421_inline2154__ssa_v0, k0_inline420_inline2148__ssa_v0], [32, 512], [32, 512], target_memory=pl.Mem.Mat
                )
                x_tile_inline413_inline2170__tile_1: pl.Tile[
                    [16, 512], pl.BF16, pl.MemRef(mem_mat_10, pl.const(0, pl.INT64), 16384), pl.Mem.Mat, pl.TileView(valid_shape=[x_rows_inline426_inline2158__ssa_v0_1, 512])
                ] = pl.tile.load(
                    x_flat_inline416_inline2157__ssa_v0,
                    [global_row0_inline418_inline2169__ssa_v0, k0_inline420_inline2148__ssa_v0_1],
                    [16, 512],
                    [x_rows_inline426_inline2158__ssa_v0_1, 512],
                    target_memory=pl.Mem.Mat,
                )
                wkv_tile_inline412_inline2167__tile_1: pl.Tile[[32, 512], pl.BF16, pl.MemRef(mem_mat_11, pl.const(16384, pl.INT64), 32768), pl.Mem.Mat] = pl.tile.load(
                    inner_wkv__ssa_v0, [o0_inline421_inline2154__ssa_v0, k0_inline420_inline2148__ssa_v0_1], [32, 512], [32, 512], target_memory=pl.Mem.Mat
                )
                wgate_tile_inline411_inline2151__tile_1: pl.Tile[[32, 512], pl.BF16, pl.MemRef(mem_mat_12, pl.const(49152, pl.INT64), 32768), pl.Mem.Mat] = pl.tile.load(
                    inner_wgate__ssa_v0, [o0_inline421_inline2154__ssa_v0, k0_inline420_inline2148__ssa_v0_1], [32, 512], [32, 512], target_memory=pl.Mem.Mat
                )
                wkv_tile_inline412_inline2167__tile_t: pl.Tile[
                    [512, 32], pl.BF16, pl.MemRef(mem_mat_8, pl.const(98304, pl.INT64), 32768), pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)
                ] = pl.tile.transpose_view(wkv_tile_inline412_inline2167__tile)
                x_tile_inline413_inline2170__tile_Left: pl.Tile[
                    [16, 512], pl.BF16, pl.MemRef(mem_left_13, pl.const(0, pl.INT64), 16384), pl.Mem.Left, pl.TileView(valid_shape=[x_rows_inline426_inline2158__ssa_v0, 512])
                ] = pl.tile.move(x_tile_inline413_inline2170__tile, target_memory=pl.Mem.Left)
                wkv_tile_inline412_inline2167__tile_t_Right: pl.Tile[[512, 32], pl.BF16, pl.MemRef(mem_right_14, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.move(
                    wkv_tile_inline412_inline2167__tile_t, target_memory=pl.Mem.Right
                )
                kv_acc_inline422_inline2139__tile_1: pl.Tile[[16, 32], pl.FP32, pl.MemRef(mem_acc_5, pl.const(0, pl.INT64), 2048), pl.Mem.Acc] = pl.tile.matmul_acc(
                    kv_acc_inline422_inline2139__iter_v1, x_tile_inline413_inline2170__tile_Left, wkv_tile_inline412_inline2167__tile_t_Right, k0_inline420_inline2148__ssa_v0 == 0
                )
                wgate_tile_inline411_inline2151__tile_t: pl.Tile[
                    [512, 32], pl.BF16, pl.MemRef(mem_mat_9, pl.const(131072, pl.INT64), 32768), pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)
                ] = pl.tile.transpose_view(wgate_tile_inline411_inline2151__tile)
                wgate_tile_inline411_inline2151__tile_t_Right: pl.Tile[[512, 32], pl.BF16, pl.MemRef(mem_right_14, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.move(
                    wgate_tile_inline411_inline2151__tile_t, target_memory=pl.Mem.Right
                )
                score_acc_inline424_inline2161__tile_1: pl.Tile[[16, 32], pl.FP32, pl.MemRef(mem_acc_6, pl.const(2048, pl.INT64), 2048), pl.Mem.Acc] = pl.tile.matmul_acc(
                    score_acc_inline424_inline2161__iter_v1, x_tile_inline413_inline2170__tile_Left, wgate_tile_inline411_inline2151__tile_t_Right, k0_inline420_inline2148__ssa_v0 == 0
                )
                wkv_tile_inline412_inline2167__tile_t_1: pl.Tile[
                    [512, 32], pl.BF16, pl.MemRef(mem_mat_11, pl.const(16384, pl.INT64), 32768), pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)
                ] = pl.tile.transpose_view(wkv_tile_inline412_inline2167__tile_1)
                x_tile_inline413_inline2170__tile_Left_1: pl.Tile[
                    [16, 512], pl.BF16, pl.MemRef(mem_left_16, pl.const(16384, pl.INT64), 16384), pl.Mem.Left, pl.TileView(valid_shape=[x_rows_inline426_inline2158__ssa_v0_1, 512])
                ] = pl.tile.move(x_tile_inline413_inline2170__tile_1, target_memory=pl.Mem.Left)
                wkv_tile_inline412_inline2167__tile_t_Right_1: pl.Tile[[512, 32], pl.BF16, pl.MemRef(mem_right_17, pl.const(32768, pl.INT64), 32768), pl.Mem.Right] = pl.tile.move(
                    wkv_tile_inline412_inline2167__tile_t_1, target_memory=pl.Mem.Right
                )
                kv_acc_inline422_inline2139__tile_2: pl.Tile[[16, 32], pl.FP32, pl.MemRef(mem_acc_5, pl.const(0, pl.INT64), 2048), pl.Mem.Acc] = pl.tile.matmul_acc(
                    kv_acc_inline422_inline2139__tile_1, x_tile_inline413_inline2170__tile_Left_1, wkv_tile_inline412_inline2167__tile_t_Right_1, k0_inline420_inline2148__ssa_v0_1 == 0
                )
                wgate_tile_inline411_inline2151__tile_t_1: pl.Tile[
                    [512, 32], pl.BF16, pl.MemRef(mem_mat_12, pl.const(49152, pl.INT64), 32768), pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)
                ] = pl.tile.transpose_view(wgate_tile_inline411_inline2151__tile_1)
                wgate_tile_inline411_inline2151__tile_t_Right_1: pl.Tile[[512, 32], pl.BF16, pl.MemRef(mem_right_17, pl.const(32768, pl.INT64), 32768), pl.Mem.Right] = pl.tile.move(
                    wgate_tile_inline411_inline2151__tile_t_1, target_memory=pl.Mem.Right
                )
                score_acc_inline424_inline2161__tile_2: pl.Tile[[16, 32], pl.FP32, pl.MemRef(mem_acc_6, pl.const(2048, pl.INT64), 2048), pl.Mem.Acc] = pl.tile.matmul_acc(
                    score_acc_inline424_inline2161__tile_1, x_tile_inline413_inline2170__tile_Left_1, wgate_tile_inline411_inline2151__tile_t_Right_1, k0_inline420_inline2148__ssa_v0_1 == 0
                )
                kv_acc_inline422_inline2139__rv_v2, score_acc_inline424_inline2161__rv_v2 = pl.yield_(kv_acc_inline422_inline2139__tile_2, score_acc_inline424_inline2161__tile_2)
            kv_proj_pad_inline2152__tile: pl.Tensor[[384, 256], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 393216)] = pl.tile.store(
                kv_acc_inline422_inline2139__rv_v2, [global_row0_inline418_inline2169__ssa_v0, o0_inline421_inline2154__ssa_v0], kv_proj_pad_inline2152__iter_v1
            )
            score_proj_pad_inline2156__tile: pl.Tensor[[384, 256], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 393216)] = pl.tile.store(
                score_acc_inline424_inline2161__rv_v2, [global_row0_inline418_inline2169__ssa_v0, o0_inline421_inline2154__ssa_v0], score_proj_pad_inline2156__iter_v1
            )
            kv_proj_pad_inline2152__rv_v2, score_proj_pad_inline2156__rv_v2 = pl.yield_(kv_proj_pad_inline2152__tile, score_proj_pad_inline2156__tile)
        return kv_proj_pad_inline2152__ssa_v0, score_proj_pad_inline2156__ssa_v0

    @pl.function(type=pl.FunctionType.Spmd)
    def kv_score_proj_spmd_0(
        self,
        kv_proj_pad_inline2152__ssa_v0: pl.Out[pl.Tensor[[384, 256], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 393216)]],
        score_proj_pad_inline2156__ssa_v0: pl.Out[pl.Tensor[[384, 256], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 393216)]],
        t_matmul_inline415_inline2162__ssa_v0: pl.Scalar[pl.INDEX],
        bs_inline414_inline2160__ssa_v0: pl.Scalar[pl.INDEX],
        x_flat_inline416_inline2157__ssa_v0: pl.Tensor[[T_DYN, 4096], pl.BF16, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)],
        inner_wkv__ssa_v0: pl.Tensor[[256, 4096], pl.BF16, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 2097152)],
        inner_wgate__ssa_v0: pl.Tensor[[256, 4096], pl.BF16, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 2097152)],
    ) -> tuple[pl.Tensor[[384, 256], pl.FP32], pl.Tensor[[384, 256], pl.FP32]]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        ret__tmp_v0: pl.Tuple[pl.Tensor[[384, 256], pl.FP32], pl.Tensor[[384, 256], pl.FP32]] = self.kv_score_proj_0(
            kv_proj_pad_inline2152__ssa_v0,
            score_proj_pad_inline2156__ssa_v0,
            t_matmul_inline415_inline2162__ssa_v0,
            bs_inline414_inline2160__ssa_v0,
            x_flat_inline416_inline2157__ssa_v0,
            inner_wkv__ssa_v0,
            inner_wgate__ssa_v0,
            attrs={"arg_directions": [pl.adir.output_existing, pl.adir.output_existing, pl.adir.scalar, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.input]},
        )
        kv_proj_pad_inline2152__rv_v2: pl.Tensor[[384, 256], pl.FP32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 393216)] = ret__tmp_v0[0]
        score_proj_pad_inline2156__rv_v2: pl.Tensor[[384, 256], pl.FP32, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 393216)] = ret__tmp_v0[1]
        return kv_proj_pad_inline2152__ssa_v0, score_proj_pad_inline2156__ssa_v0

    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def kv_touch(
        ori_kv_flat_inline2519__ssa_v0: pl.InOut[pl.Tensor[[ori_block_num_inline2509__ssa_v0 * pl.const(32, pl.INDEX), 512], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)]],
    ) -> pl.Tensor[[ori_block_num_inline2509__ssa_v0 * pl.const(32, pl.INDEX), 512], pl.BF16]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        mem_vec_1: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 1024)
        t__tile: pl.Tile[[1, 512], pl.BF16, pl.MemRef(mem_vec_1, pl.const(0, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.load(
            ori_kv_flat_inline2519__ssa_v0, [0, 0], [1, 512], [1, 512], target_memory=pl.Mem.Vec
        )
        ori_kv_flat_inline2519__tile: pl.Tensor[[ori_block_num_inline2509__ssa_v0 * pl.const(32, pl.INDEX), 512], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)] = pl.tile.store(
            t__tile, [0, 0], ori_kv_flat_inline2519__ssa_v0
        )
        return ori_kv_flat_inline2519__ssa_v0

    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def merge_norm(
        rope_swap_idx_inline597__ssa_v0: pl.Tensor[[16, 64], pl.INT32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 4096)],
        t_dim_inline586__ssa_v0: pl.Scalar[pl.INDEX],
        attn_mi_inline600__ssa_v0: pl.Tensor[[t_heads_inline2521__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)],
        attn_li_inline594__ssa_v0: pl.Tensor[[t_heads_inline2521__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)],
        attn_oi_inline592__ssa_v0: pl.Tensor[[t_heads_inline2521__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)],
        merge_sink_inline599__ssa_v0: pl.Tensor[[64, 1], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 256)],
        rope_cos_il_inline591__ssa_v0: pl.Tensor[[384, 64], pl.FP32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 98304)],
        rope_sin_signed_inline589__ssa_v0: pl.Tensor[[384, 64], pl.FP32, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 98304)],
        o_packed_heads__ssa_v0: pl.Out[pl.Tensor[[3072, 4096], pl.BF16, pl.MemRef("mem_ddr_7", pl.const(0, pl.INT64), 25165824)]],
    ) -> pl.Tensor[[3072, 4096], pl.BF16]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        mem_vec_16: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 4096)
        mem_vec_17: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 4096)
        mem_vec_20: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32768)
        mem_vec_28: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
        mem_vec_30: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        mem_vec_31: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        mem_vec_32: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 4096)
        mem_vec_33: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 4096)
        m_worker_inline583__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        m_swap_inline598__ssa_v0: pl.Tile[[16, 64], pl.INT32, pl.MemRef(mem_vec_20, pl.const(8192, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.load(
            rope_swap_idx_inline597__ssa_v0, [0, 0], [16, 64], [16, 64], target_memory=pl.Mem.Vec
        )
        m_swap_f_inline605__ssa_v0: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_20, pl.const(8192, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(
            m_swap_inline598__ssa_v0, target_type=pl.FP32, mode="round"
        )
        m_swap_source_inline606__ssa_v0: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_20, pl.const(8192, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.adds(m_swap_f_inline605__ssa_v0, 448.0)
        m_row_ids_inline609__ci_tmp_v0: pl.Tile[[1, 192], pl.FP32, pl.MemRef(mem_vec_28, pl.const(40960, pl.INT64), 768), pl.Mem.Vec] = pl.tile.create(
            [1, 192], dtype=pl.FP32, target_memory=pl.Mem.Vec
        )
        m_row_ids_inline609__ssa_v0: pl.Tile[[1, 16], pl.INT32, pl.MemRef(mem_vec_16, pl.const(0, pl.INT64), 64), pl.Mem.Vec] = pl.tile.ci(
            pl.const(0, pl.INT32), [1, 16], tmp=m_row_ids_inline609__ci_tmp_v0, dtype=pl.INT32, descending=False
        )
        m_row_ids_f_inline595__ssa_v0: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_28, pl.const(40960, pl.INT64), 64), pl.Mem.Vec] = pl.tile.cast(
            m_row_ids_inline609__ssa_v0, target_type=pl.FP32, mode="round"
        )
        m_row_offsets_inline610__ssa_v0: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_28, pl.const(40960, pl.INT64), 64), pl.Mem.Vec] = pl.tile.muls(m_row_ids_f_inline595__ssa_v0, 512.0)
        m_row_offsets_col_inline593__ssa_v0: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_28, pl.const(40960, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(m_row_offsets_inline610__ssa_v0, [16, 1])
        m_swap_flat_inline585__ssa_v0: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_20, pl.const(8192, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.row_expand_add(
            m_swap_source_inline606__ssa_v0, m_row_offsets_col_inline593__ssa_v0
        )
        m_swap_idx_inline608__ssa_v0: pl.Tile[[16, 64], pl.INT32, pl.MemRef(mem_vec_16, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(
            m_swap_flat_inline585__ssa_v0, target_type=pl.INT32, mode="round"
        )
        m_gather_tmp_inline590__ssa_v0: pl.Tile[[16, 64], pl.INT32, pl.MemRef(mem_vec_17, pl.const(4096, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.create(
            [16, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec
        )
        for m_idx_inline588__idx_v0 in pl.range(m_worker_inline583__ssa_v0, t_dim_inline586__ssa_v0 * 4, 48):
            m_t_inline613__ssa_v0: pl.Scalar[pl.INDEX] = m_idx_inline588__idx_v0 // 4
            m_h_idx_inline614__ssa_v0: pl.Scalar[pl.INDEX] = m_idx_inline588__idx_v0 - m_t_inline613__ssa_v0 * 4
            m_h0_inline607__ssa_v0: pl.Scalar[pl.INDEX] = m_h_idx_inline614__ssa_v0 * 16
            m_row_inline618__ssa_v0: pl.Scalar[pl.INDEX] = m_idx_inline588__idx_v0 * 16
            m_mi_inline616__ssa_v0: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_28, pl.const(40960, pl.INT64), 64), pl.Mem.Vec] = pl.tile.load(
                attn_mi_inline600__ssa_v0, [m_row_inline618__ssa_v0, 0], [16, 1], [16, 1], target_memory=pl.Mem.Vec
            )
            m_li_inline584__ssa_v0: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_32, pl.const(57856, pl.INT64), 64), pl.Mem.Vec] = pl.tile.load(
                attn_li_inline594__ssa_v0, [m_row_inline618__ssa_v0, 0], [16, 1], [16, 1], target_memory=pl.Mem.Vec
            )
            m_oi_inline582__ssa_v0: pl.Tile[[16, 512], pl.FP32, pl.MemRef(mem_vec_20, pl.const(8192, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.load(
                attn_oi_inline592__ssa_v0, [m_row_inline618__ssa_v0, 0], [16, 512], [16, 512], target_memory=pl.Mem.Vec
            )
            n_sink_bias_inline580__ssa_v0: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_33, pl.const(61952, pl.INT64), 64), pl.Mem.Vec] = pl.tile.load(
                merge_sink_inline599__ssa_v0, [m_h0_inline607__ssa_v0, 0], [16, 1], [16, 1], target_memory=pl.Mem.Vec
            )
            t__rm_a0_tmp_v0: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_28, pl.const(40960, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(m_mi_inline616__ssa_v0, [1, 16])
            t__rm_a1_tmp_v1: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_28, pl.const(40960, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(m_mi_inline616__ssa_v0, [1, 16])
            t__row_major_tmp_v2: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_30, pl.const(57344, pl.INT64), 64), pl.Mem.Vec] = pl.tile.sub(t__rm_a0_tmp_v0, t__rm_a1_tmp_v1)
            t__tmp_v306: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_30, pl.const(57344, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(t__row_major_tmp_v2, [16, 1])
            n_sink_tile_inline577__rm_a0_tmp_v3: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_30, pl.const(57344, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(t__tmp_v306, [1, 16])
            n_sink_tile_inline577__rm_a1_tmp_v4: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_33, pl.const(61952, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(n_sink_bias_inline580__ssa_v0, [1, 16])
            n_sink_tile_inline577__row_major_tmp_v5: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_33, pl.const(61952, pl.INT64), 64), pl.Mem.Vec] = pl.tile.add(
                n_sink_tile_inline577__rm_a0_tmp_v3, n_sink_tile_inline577__rm_a1_tmp_v4
            )
            n_sink_tile_inline577__ssa_v0: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_33, pl.const(61952, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(
                n_sink_tile_inline577__row_major_tmp_v5, [16, 1]
            )
            t__rm_a0_tmp_v6: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_33, pl.const(61952, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(n_sink_tile_inline577__ssa_v0, [1, 16])
            t__rm_a1_tmp_v7: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_28, pl.const(40960, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(m_mi_inline616__ssa_v0, [1, 16])
            t__row_major_tmp_v8: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_28, pl.const(40960, pl.INT64), 64), pl.Mem.Vec] = pl.tile.sub(t__rm_a0_tmp_v6, t__rm_a1_tmp_v7)
            t__tmp_v307: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_28, pl.const(40960, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(t__row_major_tmp_v8, [16, 1])
            t__rm_a0_tmp_v9: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_28, pl.const(40960, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(t__tmp_v307, [1, 16])
            t__row_major_tmp_v10: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_28, pl.const(40960, pl.INT64), 64), pl.Mem.Vec] = pl.tile.exp(t__rm_a0_tmp_v9)
            t__tmp_v308: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_28, pl.const(40960, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(t__row_major_tmp_v10, [16, 1])
            n_denom_inline617__rm_a0_tmp_v11: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_32, pl.const(57856, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(m_li_inline584__ssa_v0, [1, 16])
            n_denom_inline617__rm_a1_tmp_v12: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_28, pl.const(40960, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(t__tmp_v308, [1, 16])
            n_denom_inline617__row_major_tmp_v13: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_28, pl.const(40960, pl.INT64), 64), pl.Mem.Vec] = pl.tile.add(
                n_denom_inline617__rm_a0_tmp_v11, n_denom_inline617__rm_a1_tmp_v12
            )
            n_denom_inline617__ssa_v0: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_28, pl.const(40960, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(n_denom_inline617__row_major_tmp_v13, [16, 1])
            n_full_inline576__ssa_v0: pl.Tile[[16, 512], pl.FP32, pl.MemRef(mem_vec_20, pl.const(8192, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.row_expand_div(
                m_oi_inline582__ssa_v0, n_denom_inline617__ssa_v0
            )
            n_bf16_inline604__ssa_v0: pl.Tile[[16, 512], pl.BF16, pl.MemRef(mem_vec_28, pl.const(40960, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.cast(
                n_full_inline576__ssa_v0, target_type=pl.BF16, mode="rint"
            )
            n_rounded_inline579__ssa_v0: pl.Tile[[16, 512], pl.FP32, pl.MemRef(mem_vec_20, pl.const(8192, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.cast(
                n_bf16_inline604__ssa_v0, target_type=pl.FP32, mode="round"
            )
            m_cos_il_inline574__ssa_v0: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_30, pl.const(57344, pl.INT64), 256), pl.Mem.Vec] = pl.tile.load(
                rope_cos_il_inline591__ssa_v0, [m_t_inline613__ssa_v0, 0], [1, 64], [1, 64], target_memory=pl.Mem.Vec
            )
            m_sin_signed_inline573__ssa_v0: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_31, pl.const(57600, pl.INT64), 256), pl.Mem.Vec] = pl.tile.load(
                rope_sin_signed_inline589__ssa_v0, [m_t_inline613__ssa_v0, 0], [1, 64], [1, 64], target_memory=pl.Mem.Vec
            )
            m_swapped_inline587__ssa_v0: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_32, pl.const(57856, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.gather(
                n_rounded_inline579__ssa_v0, m_swap_idx_inline608__ssa_v0, m_gather_tmp_inline590__ssa_v0
            )
            m_rope_inline575__ssa_v0_textract: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_33, pl.const(61952, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.extract(
                n_rounded_inline579__ssa_v0, 0, 448, [16, 64], target_memory=pl.Mem.Vec
            )
            t__tmp_v309: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_20, pl.const(8192, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.col_expand_mul(
                m_rope_inline575__ssa_v0_textract, m_cos_il_inline574__ssa_v0
            )
            t__tmp_v310: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_32, pl.const(57856, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.col_expand_mul(
                m_swapped_inline587__ssa_v0, m_sin_signed_inline573__ssa_v0
            )
            m_rot_inline612__ssa_v0: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_20, pl.const(8192, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.add(t__tmp_v309, t__tmp_v310)
            n_rope_bf16_inline581__ssa_v0: pl.Tile[[16, 64], pl.BF16, pl.MemRef(mem_vec_32, pl.const(57856, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(
                m_rot_inline612__ssa_v0, target_type=pl.BF16, mode="rint"
            )
            t__tmp_v311: pl.Tile[[16, 448], pl.BF16, pl.MemRef(mem_vec_28, pl.const(40960, pl.INT64), 16256), pl.Mem.Vec] = pl.tile.slice(n_bf16_inline604__ssa_v0, [16, 448], [0, 0])
            n_full_bf16_inline611__ssa_v0: pl.Tile[[16, 512], pl.BF16, pl.MemRef(mem_vec_20, pl.const(8192, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.concat(t__tmp_v311, n_rope_bf16_inline581__ssa_v0)
            n_group_bf16_inline572__ssa_v0: pl.Tile[[2, 4096], pl.BF16, pl.MemRef(mem_vec_20, pl.const(8192, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.reshape(n_full_bf16_inline611__ssa_v0, [2, 4096])
            n_pack_first_inline571__ssa_v0: pl.Tile[[1, 4096], pl.BF16, pl.MemRef(mem_vec_20, pl.const(8192, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.slice(
                n_group_bf16_inline572__ssa_v0, [1, 4096], [0, 0]
            )
            n_pack_second_inline570__ssa_v0: pl.Tile[[1, 4096], pl.BF16, pl.MemRef(mem_vec_20, pl.const(16384, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.slice(
                n_group_bf16_inline572__ssa_v0, [1, 4096], [1, 0]
            )
            n_pack_row_inline578__ssa_v0: pl.Scalar[pl.INDEX] = m_h0_inline607__ssa_v0 // 8 * 384 + m_t_inline613__ssa_v0
            n_pack_row_second_inline596__ssa_v0: pl.Scalar[pl.INDEX] = n_pack_row_inline578__ssa_v0 + 384
            o_packed_heads__store: pl.Tensor[[3072, 4096], pl.BF16, pl.MemRef("mem_ddr_7", pl.const(0, pl.INT64), 25165824)] = pl.tile.store(
                n_pack_first_inline571__ssa_v0, [n_pack_row_inline578__ssa_v0, 0], o_packed_heads__ssa_v0
            )
            o_packed_heads__store_v0: pl.Tensor[[3072, 4096], pl.BF16, pl.MemRef("mem_ddr_7", pl.const(0, pl.INT64), 25165824)] = pl.tile.store(
                n_pack_second_inline570__ssa_v0, [n_pack_row_second_inline596__ssa_v0, 0], o_packed_heads__ssa_v0
            )
        return o_packed_heads__ssa_v0

    @pl.function(type=pl.FunctionType.Spmd)
    def merge_norm_spmd(
        self,
        rope_swap_idx_inline597__ssa_v0: pl.Tensor[[16, 64], pl.INT32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 4096)],
        t_dim_inline586__ssa_v0: pl.Scalar[pl.INDEX],
        attn_mi_inline600__ssa_v0: pl.Tensor[[t_heads_inline2521__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)],
        attn_li_inline594__ssa_v0: pl.Tensor[[t_heads_inline2521__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)],
        attn_oi_inline592__ssa_v0: pl.Tensor[[t_heads_inline2521__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)],
        merge_sink_inline599__ssa_v0: pl.Tensor[[64, 1], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 256)],
        rope_cos_il_inline591__ssa_v0: pl.Tensor[[384, 64], pl.FP32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 98304)],
        rope_sin_signed_inline589__ssa_v0: pl.Tensor[[384, 64], pl.FP32, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 98304)],
        o_packed_heads__ssa_v0: pl.Out[pl.Tensor[[3072, 4096], pl.BF16, pl.MemRef("mem_ddr_7", pl.const(0, pl.INT64), 25165824)]],
    ) -> pl.Tensor[[3072, 4096], pl.BF16]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        o_packed_heads__ssa_v2: pl.Tensor[[3072, 4096], pl.BF16, pl.MemRef("mem_ddr_8", pl.const(0, pl.INT64), 25165824)] = self.merge_norm(
            rope_swap_idx_inline597__ssa_v0,
            t_dim_inline586__ssa_v0,
            attn_mi_inline600__ssa_v0,
            attn_li_inline594__ssa_v0,
            attn_oi_inline592__ssa_v0,
            merge_sink_inline599__ssa_v0,
            rope_cos_il_inline591__ssa_v0,
            rope_sin_signed_inline589__ssa_v0,
            o_packed_heads__ssa_v0,
            attrs={"arg_directions": [pl.adir.input, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.output_existing]},
        )
        return o_packed_heads__ssa_v0

    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def oproj_token_scale(
        act_scale_dq_inline694__ssa_v0: pl.Out[pl.Tensor[[1, 384], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 1536)]],
        t_dim_inline652__ssa_v0: pl.Scalar[pl.INDEX],
        o_r_pad_inline675__rv_v2: pl.Tensor[[384, 8192], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 12582912)],
    ) -> pl.Tensor[[1, 384], pl.FP32]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        mem_vec_2: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32)
        mem_vec_3: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32768)
        mem_vec_7: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32768)
        mem_vec_8: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32)
        mem_vec_12: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32)
        mem_vec_13: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32768)
        mem_vec_17: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32768)
        mem_vec_18: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32)
        unroll_main_end: pl.Scalar[pl.INDEX] = (t_dim_inline652__ssa_v0 + 7) // 16 * 16
        for qt_inline704__idx_v0, (act_scale_dq_inline694__iter_v1,) in pl.range(0, unroll_main_end, 16, init_values=(act_scale_dq_inline694__ssa_v0,)):
            token_amax_inline692__tile: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_2, pl.const(65600, pl.INT64), 32), pl.Mem.Vec] = pl.tile.full([1, 8], dtype=pl.FP32, value=0.0001)
            for scale_group_inline651__idx_v0, (token_amax_inline692__iter_v1,) in pl.range(8, init_values=(token_amax_inline692__tile,)):
                scale_col_inline648__ssa_v0: pl.Scalar[pl.INDEX] = scale_group_inline651__idx_v0 * 1024
                projected_inline655__tile: pl.Tile[[8, 1024], pl.FP32, pl.MemRef(mem_vec_3, pl.const(65632, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.load(
                    o_r_pad_inline675__rv_v2, [qt_inline704__idx_v0, scale_col_inline648__ssa_v0], [8, 1024], [8, 1024], target_memory=pl.Mem.Vec
                )
                t__tile: pl.Tile[[8, 1024], pl.BF16, pl.MemRef(mem_vec_7, pl.const(98400, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.cast(projected_inline655__tile, target_type=pl.BF16, mode="rint")
                projected_v1_inline653__tile: pl.Tile[[8, 1024], pl.FP32, pl.MemRef(mem_vec_3, pl.const(65632, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.cast(t__tile, target_type=pl.FP32, mode="round")
                t__tile_1: pl.Tile[[8, 1024], pl.FP32, pl.MemRef(mem_vec_3, pl.const(65632, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.abs(projected_v1_inline653__tile)
                tmp_tile: pl.Tile[[8, 1024], pl.FP32, pl.MemRef(mem_vec_7, pl.const(98400, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.create([8, 1024], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                t__tile_2: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_8, pl.const(131168, pl.INT64), 32), pl.Mem.Vec] = pl.tile.row_max(t__tile_1, tmp_tile)
                group_amax_inline707__tile: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_8, pl.const(131168, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(t__tile_2, [1, 8])
                token_amax_inline692__tile_1: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_2, pl.const(65600, pl.INT64), 32), pl.Mem.Vec] = pl.tile.maximum(
                    token_amax_inline692__iter_v1, group_amax_inline707__tile
                )
                token_amax_inline692__rv_v2: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_2, pl.const(65600, pl.INT64), 32), pl.Mem.Vec] = pl.yield_(token_amax_inline692__tile_1)
            t__tile_3: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_3, pl.const(65632, pl.INT64), 32), pl.Mem.Vec] = pl.tile.muls(token_amax_inline692__rv_v2, 0.007874015748031496)
            act_scale_dq_inline694__tile: pl.Tensor[[1, 384], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 1536)] = pl.tile.store(
                t__tile_3, [0, qt_inline704__idx_v0], act_scale_dq_inline694__iter_v1
            )
            token_amax_inline692__tile_2: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_12, pl.const(0, pl.INT64), 32), pl.Mem.Vec] = pl.tile.full([1, 8], dtype=pl.FP32, value=0.0001)
            for scale_group_inline651__idx_v0_1, (token_amax_inline692__iter_v1_1,) in pl.range(8, init_values=(token_amax_inline692__tile_2,)):
                scale_col_inline648__ssa_v0_1: pl.Scalar[pl.INDEX] = scale_group_inline651__idx_v0_1 * 1024
                projected_inline655__tile_1: pl.Tile[[8, 1024], pl.FP32, pl.MemRef(mem_vec_13, pl.const(32, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.load(
                    o_r_pad_inline675__rv_v2, [qt_inline704__idx_v0 + 8, scale_col_inline648__ssa_v0_1], [8, 1024], [8, 1024], target_memory=pl.Mem.Vec
                )
                t__tile_4: pl.Tile[[8, 1024], pl.BF16, pl.MemRef(mem_vec_17, pl.const(32800, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.cast(
                    projected_inline655__tile_1, target_type=pl.BF16, mode="rint"
                )
                projected_v1_inline653__tile_1: pl.Tile[[8, 1024], pl.FP32, pl.MemRef(mem_vec_13, pl.const(32, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.cast(
                    t__tile_4, target_type=pl.FP32, mode="round"
                )
                t__tile_5: pl.Tile[[8, 1024], pl.FP32, pl.MemRef(mem_vec_13, pl.const(32, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.abs(projected_v1_inline653__tile_1)
                tmp_tile_1: pl.Tile[[8, 1024], pl.FP32, pl.MemRef(mem_vec_17, pl.const(32800, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.create([8, 1024], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                t__tile_6: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_18, pl.const(65568, pl.INT64), 32), pl.Mem.Vec] = pl.tile.row_max(t__tile_5, tmp_tile_1)
                group_amax_inline707__tile_1: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_18, pl.const(65568, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(t__tile_6, [1, 8])
                token_amax_inline692__tile_3: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_12, pl.const(0, pl.INT64), 32), pl.Mem.Vec] = pl.tile.maximum(
                    token_amax_inline692__iter_v1_1, group_amax_inline707__tile_1
                )
                token_amax_inline692__rv_v2_1: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_12, pl.const(0, pl.INT64), 32), pl.Mem.Vec] = pl.yield_(token_amax_inline692__tile_3)
            t__tile_7: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_13, pl.const(32, pl.INT64), 32), pl.Mem.Vec] = pl.tile.muls(token_amax_inline692__rv_v2_1, 0.007874015748031496)
            act_scale_dq_inline694__tile_1: pl.Tensor[[1, 384], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 1536)] = pl.tile.store(
                t__tile_7, [0, qt_inline704__idx_v0 + 8], act_scale_dq_inline694__tile
            )
            act_scale_dq_inline694__rv_v2_main: pl.Tensor[[1, 384], pl.FP32, pl.MemRef("mem_ddr_22", pl.const(0, pl.INT64), 1536)] = pl.yield_(act_scale_dq_inline694__tile_1)
        unroll_rem: pl.Scalar[pl.INDEX] = (t_dim_inline652__ssa_v0 + 7) // 8 - (t_dim_inline652__ssa_v0 + 7) // 16 * 2
        if unroll_rem == 1:
            token_amax_inline692__tile_4: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_13, pl.const(32, pl.INT64), 32), pl.Mem.Vec] = pl.tile.full([1, 8], dtype=pl.FP32, value=0.0001)
            for scale_group_inline651__idx_v0_2, (token_amax_inline692__iter_v1_2,) in pl.range(8, init_values=(token_amax_inline692__tile_4,)):
                scale_col_inline648__ssa_v0_2: pl.Scalar[pl.INDEX] = scale_group_inline651__idx_v0_2 * 1024
                projected_inline655__tile_2: pl.Tile[[8, 1024], pl.FP32, pl.MemRef(mem_vec_3, pl.const(65632, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.load(
                    o_r_pad_inline675__rv_v2, [unroll_main_end, scale_col_inline648__ssa_v0_2], [8, 1024], [8, 1024], target_memory=pl.Mem.Vec
                )
                t__tile_8: pl.Tile[[8, 1024], pl.BF16, pl.MemRef(mem_vec_7, pl.const(98400, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.cast(projected_inline655__tile_2, target_type=pl.BF16, mode="rint")
                projected_v1_inline653__tile_2: pl.Tile[[8, 1024], pl.FP32, pl.MemRef(mem_vec_3, pl.const(65632, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.cast(
                    t__tile_8, target_type=pl.FP32, mode="round"
                )
                t__tile_9: pl.Tile[[8, 1024], pl.FP32, pl.MemRef(mem_vec_3, pl.const(65632, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.abs(projected_v1_inline653__tile_2)
                tmp_tile_2: pl.Tile[[8, 1024], pl.FP32, pl.MemRef(mem_vec_7, pl.const(98400, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.create([8, 1024], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                t__tile_10: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_17, pl.const(32800, pl.INT64), 32), pl.Mem.Vec] = pl.tile.row_max(t__tile_9, tmp_tile_2)
                group_amax_inline707__tile_2: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_17, pl.const(32800, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(t__tile_10, [1, 8])
                token_amax_inline692__tile_5: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_13, pl.const(32, pl.INT64), 32), pl.Mem.Vec] = pl.tile.maximum(
                    token_amax_inline692__iter_v1_2, group_amax_inline707__tile_2
                )
                token_amax_inline692__rv_v2_2: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_13, pl.const(32, pl.INT64), 32), pl.Mem.Vec] = pl.yield_(token_amax_inline692__tile_5)
            t__tile_11: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_3, pl.const(65632, pl.INT64), 32), pl.Mem.Vec] = pl.tile.muls(token_amax_inline692__rv_v2_2, 0.007874015748031496)
            act_scale_dq_inline694__tile_2: pl.Tensor[[1, 384], pl.FP32, pl.MemRef("mem_ddr_22", pl.const(0, pl.INT64), 1536)] = pl.tile.store(
                t__tile_11, [0, unroll_main_end], act_scale_dq_inline694__rv_v2_main
            )
        return act_scale_dq_inline694__ssa_v0

    @pl.function(type=pl.FunctionType.AIC, level=pl.Level.AIC, role=pl.Role.SubWorker)
    def proj_a_mm(
        t_dim_inline652__ssa_v0: pl.Scalar[pl.INDEX],
        row_base_o_inline663__ssa_v0: pl.Scalar[pl.INDEX],
        o_packed_heads__ssa_v1: pl.Tensor[[3072, 4096], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 25165824)],
        wo_a__ssa_v0: pl.Tensor[[8, 1024, 4096], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 67108864)],
        g_inline695__idx_v0: pl.Scalar[pl.INDEX],
        o_r_pad_inline675__iter_v1: pl.Out[pl.Tensor[[384, 8192], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 12582912)]],
        out_col_g_inline650__ssa_v0: pl.Scalar[pl.INDEX],
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
        pa_unit_inline647__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        pa_rb_inline645__ssa_v0: pl.Scalar[pl.INDEX] = pa_unit_inline647__ssa_v0 // 8
        nf_inline641__ssa_v0: pl.Scalar[pl.INDEX] = pa_unit_inline647__ssa_v0 - pa_rb_inline645__ssa_v0 * 8
        pa_r0_inline660__ssa_v0: pl.Scalar[pl.INDEX] = pa_rb_inline645__ssa_v0 * 128
        pa_rows_inline657__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(t_dim_inline652__ssa_v0 - pa_r0_inline660__ssa_v0, 128)
        pa_src0_inline681__ssa_v0: pl.Scalar[pl.INDEX] = row_base_o_inline663__ssa_v0 + pa_r0_inline660__ssa_v0
        n0_inline671__ssa_v0: pl.Scalar[pl.INDEX] = nf_inline641__ssa_v0 * 128
        xa_first_inline666__tile: pl.Tile[[128, 256], pl.BF16, pl.MemRef(mem_mat_3, pl.const(131072, pl.INT64), 65536), pl.Mem.Mat, pl.TileView(valid_shape=[pa_rows_inline657__ssa_v0, 256])] = (
            pl.tile.load(o_packed_heads__ssa_v1, [pa_src0_inline681__ssa_v0, 0], [128, 256], [pa_rows_inline657__ssa_v0, 256], target_memory=pl.Mem.Mat)
        )
        wa_first_inline710__tile_view2d: pl.Tensor[[8192, 4096], pl.BF16, pl.TensorView(stride=[4096, 1], layout=pl.TensorLayout.ND), pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 67108864)] = (
            pl.tensor.view(wo_a__ssa_v0, [8192, 4096])
        )
        wa_first_inline710__tile: pl.Tile[[128, 256], pl.BF16, pl.MemRef(mem_mat_4, pl.const(196608, pl.INT64), 65536), pl.Mem.Mat] = pl.tile.load(
            wa_first_inline710__tile_view2d, [g_inline695__idx_v0 * 1024 + n0_inline671__ssa_v0, 0], [128, 256], [128, 256], target_memory=pl.Mem.Mat
        )
        wa_first_inline710__tile_t: pl.Tile[
            [256, 128], pl.BF16, pl.MemRef(mem_mat_4, pl.const(196608, pl.INT64), 65536), pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)
        ] = pl.tile.transpose_view(wa_first_inline710__tile)
        acc_a_inline699__tile_l0_init_storage: pl.Tile[[128, 128], pl.FP32, pl.MemRef(mem_acc_5, pl.const(0, pl.INT64), 65536), pl.Mem.Acc, pl.TileView(compact=pl.CompactMode.normal)] = (
            pl.tile.create([128, 128], dtype=pl.FP32, target_memory=pl.Mem.Acc, compact=True)
        )
        acc_a_inline699__tile_l0_init: pl.Tile[
            [128, 128], pl.FP32, pl.MemRef(mem_acc_5, pl.const(0, pl.INT64), 65536), pl.Mem.Acc, pl.TileView(valid_shape=[pa_rows_inline657__ssa_v0, 128], compact=pl.CompactMode.normal)
        ] = pl.tile.set_validshape(acc_a_inline699__tile_l0_init_storage, pa_rows_inline657__ssa_v0, 128)
        acc_a_inline699__tile_l0_a: pl.Tile[
            [128, 128],
            pl.BF16,
            pl.MemRef(mem_left_6, pl.const(0, pl.INT64), 32768),
            pl.Mem.Left,
            pl.TileView(valid_shape=[pa_rows_inline657__ssa_v0, 128], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal),
        ] = pl.tile.extract(xa_first_inline666__tile, 0, 0, [128, 128], target_memory=pl.Mem.Left)
        acc_a_inline699__tile_l0_b: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_7, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
            wa_first_inline710__tile_t, 0, 0, [128, 128], target_memory=pl.Mem.Right
        )
        acc_a_inline699__tile_l0_a_1: pl.Tile[
            [128, 128],
            pl.BF16,
            pl.MemRef(mem_left_8, pl.const(32768, pl.INT64), 32768),
            pl.Mem.Left,
            pl.TileView(valid_shape=[pa_rows_inline657__ssa_v0, 128], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal),
        ] = pl.tile.extract(xa_first_inline666__tile, 0, 128, [128, 128], target_memory=pl.Mem.Left)
        acc_a_inline699__tile_l0_b_1: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_9, pl.const(32768, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
            wa_first_inline710__tile_t, 128, 0, [128, 128], target_memory=pl.Mem.Right
        )
        acc_a_inline699__tile_l0_c_acc: pl.Tile[
            [128, 128], pl.FP32, pl.MemRef(mem_acc_5, pl.const(0, pl.INT64), 65536), pl.Mem.Acc, pl.TileView(valid_shape=[pa_rows_inline657__ssa_v0, 128], compact=pl.CompactMode.normal)
        ] = pl.tile.matmul_acc(acc_a_inline699__tile_l0_init, acc_a_inline699__tile_l0_a, acc_a_inline699__tile_l0_b, True)
        acc_a_inline699__tile_l0_c_acc_1: pl.Tile[
            [128, 128], pl.FP32, pl.MemRef(mem_acc_5, pl.const(0, pl.INT64), 65536), pl.Mem.Acc, pl.TileView(valid_shape=[pa_rows_inline657__ssa_v0, 128], compact=pl.CompactMode.normal)
        ] = pl.tile.matmul_acc(acc_a_inline699__tile_l0_c_acc, acc_a_inline699__tile_l0_a_1, acc_a_inline699__tile_l0_b_1, False)
        for kb_inline677__idx_v0, (acc_a_inline699__iter_v1,) in pl.range(1, 15, 2, init_values=(acc_a_inline699__tile_l0_c_acc_1,)):
            k0_inline662__ssa_v0: pl.Scalar[pl.INDEX] = kb_inline677__idx_v0 * 256
            k0_inline662__ssa_v0_1: pl.Scalar[pl.INDEX] = kb_inline677__idx_v0 * 256 + 256
            xa_k_chunk_inline644__tile: pl.Tile[[128, 256], pl.BF16, pl.MemRef(mem_mat_3, pl.const(131072, pl.INT64), 65536), pl.Mem.Mat, pl.TileView(valid_shape=[pa_rows_inline657__ssa_v0, 256])] = (
                pl.tile.load(o_packed_heads__ssa_v1, [pa_src0_inline681__ssa_v0, k0_inline662__ssa_v0], [128, 256], [pa_rows_inline657__ssa_v0, 256], target_memory=pl.Mem.Mat)
            )
            xa_k_chunk_inline644__tile_1: pl.Tile[
                [128, 256], pl.BF16, pl.MemRef(mem_mat_4, pl.const(196608, pl.INT64), 65536), pl.Mem.Mat, pl.TileView(valid_shape=[pa_rows_inline657__ssa_v0, 256])
            ] = pl.tile.load(o_packed_heads__ssa_v1, [pa_src0_inline681__ssa_v0, k0_inline662__ssa_v0_1], [128, 256], [pa_rows_inline657__ssa_v0, 256], target_memory=pl.Mem.Mat)
            wa_k_chunk_inline642__tile_view2d: pl.Tensor[[8192, 4096], pl.BF16, pl.TensorView(stride=[4096, 1], layout=pl.TensorLayout.ND), pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 67108864)] = (
                pl.tensor.view(wo_a__ssa_v0, [8192, 4096])
            )
            wa_k_chunk_inline642__tile: pl.Tile[[128, 256], pl.BF16, pl.MemRef(mem_mat_13, pl.const(0, pl.INT64), 65536), pl.Mem.Mat] = pl.tile.load(
                wa_k_chunk_inline642__tile_view2d, [g_inline695__idx_v0 * 1024 + n0_inline671__ssa_v0, k0_inline662__ssa_v0], [128, 256], [128, 256], target_memory=pl.Mem.Mat
            )
            wa_k_chunk_inline642__tile_view2d_1: pl.Tensor[
                [8192, 4096], pl.BF16, pl.TensorView(stride=[4096, 1], layout=pl.TensorLayout.ND), pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 67108864)
            ] = pl.tensor.view(wo_a__ssa_v0, [8192, 4096])
            wa_k_chunk_inline642__tile_1: pl.Tile[[128, 256], pl.BF16, pl.MemRef(mem_mat_14, pl.const(65536, pl.INT64), 65536), pl.Mem.Mat] = pl.tile.load(
                wa_k_chunk_inline642__tile_view2d_1, [g_inline695__idx_v0 * 1024 + n0_inline671__ssa_v0, k0_inline662__ssa_v0_1], [128, 256], [128, 256], target_memory=pl.Mem.Mat
            )
            wa_k_chunk_inline642__tile_t: pl.Tile[
                [256, 128], pl.BF16, pl.MemRef(mem_mat_13, pl.const(0, pl.INT64), 65536), pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)
            ] = pl.tile.transpose_view(wa_k_chunk_inline642__tile)
            acc_a_inline699__iter_v1_l0_a: pl.Tile[
                [128, 128],
                pl.BF16,
                pl.MemRef(mem_left_6, pl.const(0, pl.INT64), 32768),
                pl.Mem.Left,
                pl.TileView(valid_shape=[pa_rows_inline657__ssa_v0, 128], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal),
            ] = pl.tile.extract(xa_k_chunk_inline644__tile, 0, 0, [128, 128], target_memory=pl.Mem.Left)
            acc_a_inline699__iter_v1_l0_b: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_7, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                wa_k_chunk_inline642__tile_t, 0, 0, [128, 128], target_memory=pl.Mem.Right
            )
            acc_a_inline699__iter_v1_l0_a_1: pl.Tile[
                [128, 128],
                pl.BF16,
                pl.MemRef(mem_left_8, pl.const(32768, pl.INT64), 32768),
                pl.Mem.Left,
                pl.TileView(valid_shape=[pa_rows_inline657__ssa_v0, 128], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal),
            ] = pl.tile.extract(xa_k_chunk_inline644__tile, 0, 128, [128, 128], target_memory=pl.Mem.Left)
            acc_a_inline699__iter_v1_l0_b_1: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_9, pl.const(32768, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                wa_k_chunk_inline642__tile_t, 128, 0, [128, 128], target_memory=pl.Mem.Right
            )
            acc_a_inline699__iter_v1_l0_c_acc: pl.Tile[
                [128, 128], pl.FP32, pl.MemRef(mem_acc_5, pl.const(0, pl.INT64), 65536), pl.Mem.Acc, pl.TileView(valid_shape=[pa_rows_inline657__ssa_v0, 128], compact=pl.CompactMode.normal)
            ] = pl.tile.matmul_acc(acc_a_inline699__iter_v1, acc_a_inline699__iter_v1_l0_a, acc_a_inline699__iter_v1_l0_b)
            acc_a_inline699__iter_v1_l0_c_acc_1: pl.Tile[
                [128, 128], pl.FP32, pl.MemRef(mem_acc_5, pl.const(0, pl.INT64), 65536), pl.Mem.Acc, pl.TileView(valid_shape=[pa_rows_inline657__ssa_v0, 128], compact=pl.CompactMode.normal)
            ] = pl.tile.matmul_acc(acc_a_inline699__iter_v1_l0_c_acc, acc_a_inline699__iter_v1_l0_a_1, acc_a_inline699__iter_v1_l0_b_1)
            wa_k_chunk_inline642__tile_t_1: pl.Tile[
                [256, 128], pl.BF16, pl.MemRef(mem_mat_14, pl.const(65536, pl.INT64), 65536), pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)
            ] = pl.tile.transpose_view(wa_k_chunk_inline642__tile_1)
            acc_a_inline699__iter_v1_l0_a_2: pl.Tile[
                [128, 128],
                pl.BF16,
                pl.MemRef(mem_left_6, pl.const(0, pl.INT64), 32768),
                pl.Mem.Left,
                pl.TileView(valid_shape=[pa_rows_inline657__ssa_v0, 128], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal),
            ] = pl.tile.extract(xa_k_chunk_inline644__tile_1, 0, 0, [128, 128], target_memory=pl.Mem.Left)
            acc_a_inline699__iter_v1_l0_b_2: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_7, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                wa_k_chunk_inline642__tile_t_1, 0, 0, [128, 128], target_memory=pl.Mem.Right
            )
            acc_a_inline699__iter_v1_l0_a_3: pl.Tile[
                [128, 128],
                pl.BF16,
                pl.MemRef(mem_left_8, pl.const(32768, pl.INT64), 32768),
                pl.Mem.Left,
                pl.TileView(valid_shape=[pa_rows_inline657__ssa_v0, 128], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal),
            ] = pl.tile.extract(xa_k_chunk_inline644__tile_1, 0, 128, [128, 128], target_memory=pl.Mem.Left)
            acc_a_inline699__iter_v1_l0_b_3: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_9, pl.const(32768, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                wa_k_chunk_inline642__tile_t_1, 128, 0, [128, 128], target_memory=pl.Mem.Right
            )
            acc_a_inline699__iter_v1_l0_c_acc_2: pl.Tile[
                [128, 128], pl.FP32, pl.MemRef(mem_acc_5, pl.const(0, pl.INT64), 65536), pl.Mem.Acc, pl.TileView(valid_shape=[pa_rows_inline657__ssa_v0, 128], compact=pl.CompactMode.normal)
            ] = pl.tile.matmul_acc(acc_a_inline699__iter_v1_l0_c_acc_1, acc_a_inline699__iter_v1_l0_a_2, acc_a_inline699__iter_v1_l0_b_2)
            acc_a_inline699__iter_v1_l0_c_acc_3: pl.Tile[
                [128, 128], pl.FP32, pl.MemRef(mem_acc_5, pl.const(0, pl.INT64), 65536), pl.Mem.Acc, pl.TileView(valid_shape=[pa_rows_inline657__ssa_v0, 128], compact=pl.CompactMode.normal)
            ] = pl.tile.matmul_acc(acc_a_inline699__iter_v1_l0_c_acc_2, acc_a_inline699__iter_v1_l0_a_3, acc_a_inline699__iter_v1_l0_b_3)
            acc_a_inline699__rv_v2_main: pl.Tile[
                [128, 128], pl.FP32, pl.MemRef(mem_acc_5, pl.const(0, pl.INT64), 65536), pl.Mem.Acc, pl.TileView(valid_shape=[pa_rows_inline657__ssa_v0, 128], compact=pl.CompactMode.normal)
            ] = pl.yield_(acc_a_inline699__iter_v1_l0_c_acc_3)
        xa_k_chunk_inline644__tile_2: pl.Tile[[128, 256], pl.BF16, pl.MemRef(mem_mat_3, pl.const(131072, pl.INT64), 65536), pl.Mem.Mat, pl.TileView(valid_shape=[pa_rows_inline657__ssa_v0, 256])] = (
            pl.tile.load(o_packed_heads__ssa_v1, [pa_src0_inline681__ssa_v0, 3840], [128, 256], [pa_rows_inline657__ssa_v0, 256], target_memory=pl.Mem.Mat)
        )
        wa_k_chunk_inline642__tile_view2d_2: pl.Tensor[[8192, 4096], pl.BF16, pl.TensorView(stride=[4096, 1], layout=pl.TensorLayout.ND), pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 67108864)] = (
            pl.tensor.view(wo_a__ssa_v0, [8192, 4096])
        )
        wa_k_chunk_inline642__tile_2: pl.Tile[[128, 256], pl.BF16, pl.MemRef(mem_mat_4, pl.const(196608, pl.INT64), 65536), pl.Mem.Mat] = pl.tile.load(
            wa_k_chunk_inline642__tile_view2d_2, [g_inline695__idx_v0 * 1024 + n0_inline671__ssa_v0, 3840], [128, 256], [128, 256], target_memory=pl.Mem.Mat
        )
        wa_k_chunk_inline642__tile_t_2: pl.Tile[
            [256, 128], pl.BF16, pl.MemRef(mem_mat_4, pl.const(196608, pl.INT64), 65536), pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)
        ] = pl.tile.transpose_view(wa_k_chunk_inline642__tile_2)
        acc_a_inline699__iter_v1_l0_a_4: pl.Tile[
            [128, 128],
            pl.BF16,
            pl.MemRef(mem_left_6, pl.const(0, pl.INT64), 32768),
            pl.Mem.Left,
            pl.TileView(valid_shape=[pa_rows_inline657__ssa_v0, 128], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal),
        ] = pl.tile.extract(xa_k_chunk_inline644__tile_2, 0, 0, [128, 128], target_memory=pl.Mem.Left)
        acc_a_inline699__iter_v1_l0_b_4: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_7, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
            wa_k_chunk_inline642__tile_t_2, 0, 0, [128, 128], target_memory=pl.Mem.Right
        )
        acc_a_inline699__iter_v1_l0_a_5: pl.Tile[
            [128, 128],
            pl.BF16,
            pl.MemRef(mem_left_8, pl.const(32768, pl.INT64), 32768),
            pl.Mem.Left,
            pl.TileView(valid_shape=[pa_rows_inline657__ssa_v0, 128], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal),
        ] = pl.tile.extract(xa_k_chunk_inline644__tile_2, 0, 128, [128, 128], target_memory=pl.Mem.Left)
        acc_a_inline699__iter_v1_l0_b_5: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_9, pl.const(32768, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
            wa_k_chunk_inline642__tile_t_2, 128, 0, [128, 128], target_memory=pl.Mem.Right
        )
        acc_a_inline699__iter_v1_l0_c_acc_4: pl.Tile[
            [128, 128], pl.FP32, pl.MemRef(mem_acc_5, pl.const(0, pl.INT64), 65536), pl.Mem.Acc, pl.TileView(valid_shape=[pa_rows_inline657__ssa_v0, 128], compact=pl.CompactMode.normal)
        ] = pl.tile.matmul_acc(acc_a_inline699__rv_v2_main, acc_a_inline699__iter_v1_l0_a_4, acc_a_inline699__iter_v1_l0_b_4)
        acc_a_inline699__iter_v1_l0_c_acc_5: pl.Tile[
            [128, 128], pl.FP32, pl.MemRef(mem_acc_5, pl.const(0, pl.INT64), 65536), pl.Mem.Acc, pl.TileView(valid_shape=[pa_rows_inline657__ssa_v0, 128], compact=pl.CompactMode.normal)
        ] = pl.tile.matmul_acc(acc_a_inline699__iter_v1_l0_c_acc_4, acc_a_inline699__iter_v1_l0_a_5, acc_a_inline699__iter_v1_l0_b_5)
        acc_a_inline699__rv_v2: pl.Tile[
            [128, 128], pl.FP32, pl.MemRef(mem_acc_5, pl.const(0, pl.INT64), 65536), pl.Mem.Acc, pl.TileView(valid_shape=[pa_rows_inline657__ssa_v0, 128], compact=pl.CompactMode.normal)
        ] = acc_a_inline699__iter_v1_l0_c_acc_5
        o_r_pad_inline675__tile: pl.Tensor[[384, 8192], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 12582912)] = pl.tile.store(
            acc_a_inline699__rv_v2, [pa_r0_inline660__ssa_v0, out_col_g_inline650__ssa_v0 + n0_inline671__ssa_v0], o_r_pad_inline675__iter_v1
        )
        return o_r_pad_inline675__iter_v1

    @pl.function(type=pl.FunctionType.Spmd)
    def proj_a_mm_spmd(
        self,
        t_dim_inline652__ssa_v0: pl.Scalar[pl.INDEX],
        row_base_o_inline663__ssa_v0: pl.Scalar[pl.INDEX],
        o_packed_heads__ssa_v1: pl.Tensor[[3072, 4096], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 25165824)],
        wo_a__ssa_v0: pl.Tensor[[8, 1024, 4096], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 67108864)],
        g_inline695__idx_v0: pl.Scalar[pl.INDEX],
        o_r_pad_inline675__iter_v1: pl.Out[pl.Tensor[[384, 8192], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 12582912)]],
        out_col_g_inline650__ssa_v0: pl.Scalar[pl.INDEX],
    ) -> pl.Tensor[[384, 8192], pl.FP32]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        o_r_pad_inline675__ssa_v3: pl.Tensor[[384, 8192], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 12582912)] = self.proj_a_mm(
            t_dim_inline652__ssa_v0,
            row_base_o_inline663__ssa_v0,
            o_packed_heads__ssa_v1,
            wo_a__ssa_v0,
            g_inline695__idx_v0,
            o_r_pad_inline675__iter_v1,
            out_col_g_inline650__ssa_v0,
            attrs={"arg_directions": [pl.adir.scalar, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.scalar, pl.adir.output_existing, pl.adir.scalar]},
        )
        return o_r_pad_inline675__iter_v1

    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def proj_b_act(
        wo_b_scale__ssa_v0: pl.Tensor[[4096], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 16384)],
        attn_out__ssa_v0: pl.Out[pl.Tensor[[T_DYN, 4096], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)]],
        t_dim_inline652__ssa_v0: pl.Scalar[pl.INDEX],
        partials_inline684__rv_v2: pl.Tensor[[384, 32768], pl.INT32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 50331648)],
        act_scale_dq_inline694__rv_v2: pl.Tensor[[1, 384], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 1536)],
    ) -> pl.Tensor[[T_DYN, 4096], pl.BF16]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        mem_vec_4: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 2048)
        mem_vec_5: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
        mem_vec_6: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
        mem_vec_7: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
        act_idx_inline625__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        tblk_inline624__ssa_v0: pl.Scalar[pl.INDEX] = act_idx_inline625__ssa_v0 // 8
        nreg_inline690__ssa_v0: pl.Scalar[pl.INDEX] = act_idx_inline625__ssa_v0 - tblk_inline624__ssa_v0 * 8
        ob_n0_inline703__ssa_v0: pl.Scalar[pl.INDEX] = nreg_inline690__ssa_v0 * 512
        t0_inline635__ssa_v1: pl.Scalar[pl.INDEX] = tblk_inline624__ssa_v0 * 32
        wb_scale_inline630__tile: pl.Tile[[512], pl.FP32, pl.MemRef(mem_vec_4, pl.const(0, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.load(
            wo_b_scale__ssa_v0, [ob_n0_inline703__ssa_v0], [512], [512], target_memory=pl.Mem.Vec
        )
        wb_scale_chunk_inline678__tile: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_4, pl.const(0, pl.INT64), 2048), pl.Mem.Vec] = wb_scale_inline630__tile
        for b_tb_inline701__idx_v0, (attn_out__iter_v1,) in pl.range(t0_inline635__ssa_v1, pl.min(t0_inline635__ssa_v1 + 32, t_dim_inline652__ssa_v0), 8, init_values=(attn_out__ssa_v0,)):
            acc_i32_inline623__tile: pl.Tile[[8, 512], pl.INT32, pl.MemRef(mem_vec_5, pl.const(2048, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.full([8, 512], dtype=pl.INT32, value=0)
            for act_g_inline622__idx_v0, (acc_i32_inline623__iter_v1,) in pl.range(0, 8, 2, init_values=(acc_i32_inline623__tile,)):
                p_col0_inline665__ssa_v0: pl.Scalar[pl.INDEX] = act_g_inline622__idx_v0 * 4096 + ob_n0_inline703__ssa_v0
                p_col0_inline665__ssa_v0_1: pl.Scalar[pl.INDEX] = act_g_inline622__idx_v0 * 4096 + ob_n0_inline703__ssa_v0 + 4096
                p_g_inline621__tile: pl.Tile[[8, 512], pl.INT32, pl.MemRef(mem_vec_6, pl.const(18432, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.load(
                    partials_inline684__rv_v2, [b_tb_inline701__idx_v0, p_col0_inline665__ssa_v0], [8, 512], [8, 512], target_memory=pl.Mem.Vec
                )
                p_g_inline621__tile_1: pl.Tile[[8, 512], pl.INT32, pl.MemRef(mem_vec_7, pl.const(34816, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.load(
                    partials_inline684__rv_v2, [b_tb_inline701__idx_v0, p_col0_inline665__ssa_v0_1], [8, 512], [8, 512], target_memory=pl.Mem.Vec
                )
                acc_i32_inline623__tile_1: pl.Tile[[8, 512], pl.INT32, pl.MemRef(mem_vec_6, pl.const(18432, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.add(
                    acc_i32_inline623__iter_v1, p_g_inline621__tile
                )
                acc_i32_inline623__tile_2: pl.Tile[[8, 512], pl.INT32, pl.MemRef(mem_vec_5, pl.const(2048, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.add(
                    acc_i32_inline623__tile_1, p_g_inline621__tile_1
                )
                acc_i32_inline623__rv_v2: pl.Tile[[8, 512], pl.INT32, pl.MemRef(mem_vec_5, pl.const(2048, pl.INT64), 16384), pl.Mem.Vec] = pl.yield_(acc_i32_inline623__tile_2)
            t__tile: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_6, pl.const(18432, pl.INT64), 32), pl.Mem.Vec] = pl.tile.load(
                act_scale_dq_inline694__rv_v2, [0, b_tb_inline701__idx_v0], [1, 8], [1, 8], target_memory=pl.Mem.Vec
            )
            output_token_scale_inline620__tile: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_6, pl.const(18432, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(t__tile, [8, 1])
            t__tile_1: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_5, pl.const(2048, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.cast(acc_i32_inline623__rv_v2, target_type=pl.FP32, mode="round")
            acc_inline674__tile: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_5, pl.const(2048, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.row_expand_mul(t__tile_1, output_token_scale_inline620__tile)
            out_t_inline705__tile: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_5, pl.const(2048, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.col_expand_mul(
                acc_inline674__tile, wb_scale_chunk_inline678__tile
            )
            out_bf16_inline619__tile: pl.Tile[[8, 512], pl.BF16, pl.MemRef(mem_vec_5, pl.const(2048, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.cast(
                out_t_inline705__tile, target_type=pl.BF16, mode="rint"
            )
            attn_out__tile: pl.Tensor[[T_DYN, 4096], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                out_bf16_inline619__tile, [b_tb_inline701__idx_v0, ob_n0_inline703__ssa_v0], attn_out__iter_v1
            )
            attn_out__rv_v2: pl.Tensor[[T_DYN, 4096], pl.BF16, pl.MemRef("mem_ddr_16", pl.const(0, pl.INT64), 0)] = pl.yield_(attn_out__tile)
        return attn_out__ssa_v0

    @pl.function(type=pl.FunctionType.Spmd)
    def proj_b_act_spmd(
        self,
        wo_b_scale__ssa_v0: pl.Tensor[[4096], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 16384)],
        attn_out__ssa_v0: pl.Out[pl.Tensor[[T_DYN, 4096], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)]],
        t_dim_inline652__ssa_v0: pl.Scalar[pl.INDEX],
        partials_inline684__rv_v2: pl.Tensor[[384, 32768], pl.INT32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 50331648)],
        act_scale_dq_inline694__rv_v2: pl.Tensor[[1, 384], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 1536)],
    ) -> pl.Tensor[[T_DYN, 4096], pl.BF16]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        attn_out__rv_v2: pl.Tensor[[T_DYN, 4096], pl.BF16, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 0)] = self.proj_b_act(
            wo_b_scale__ssa_v0,
            attn_out__ssa_v0,
            t_dim_inline652__ssa_v0,
            partials_inline684__rv_v2,
            act_scale_dq_inline694__rv_v2,
            attrs={"arg_directions": [pl.adir.input, pl.adir.output_existing, pl.adir.scalar, pl.adir.input, pl.adir.input]},
        )
        return attn_out__ssa_v0

    @pl.function(type=pl.FunctionType.AIC, level=pl.Level.AIC, role=pl.Role.SubWorker)
    def proj_b_mm(
        partials_inline684__iter_v1: pl.Out[pl.Tensor[[384, 32768], pl.INT32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 50331648)]],
        col_g_inline712__ssa_v0: pl.Scalar[pl.INDEX],
        o_r_i8_pad_inline667__rv_v7: pl.Tensor[[384, 8192], pl.INT8, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 3145728)],
        wo_b__ssa_v0: pl.Tensor[[4096, 8192], pl.INT8, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 33554432)],
        g_inline708__idx_v0: pl.Scalar[pl.INDEX],
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
        pb_unit_inline638__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        tb_inline637__ssa_v0: pl.Scalar[pl.INDEX] = pb_unit_inline638__ssa_v0 // 8
        dc_inline636__ssa_v0: pl.Scalar[pl.INDEX] = pb_unit_inline638__ssa_v0 - tb_inline637__ssa_v0 * 8
        t0_inline635__ssa_v0: pl.Scalar[pl.INDEX] = tb_inline637__ssa_v0 * 128
        d0_inline634__ssa_v0: pl.Scalar[pl.INDEX] = dc_inline636__ssa_v0 * 512
        for nf_inline633__idx_v0, (partials_inline684__iter_v3,) in pl.range(2, init_values=(partials_inline684__iter_v1,)):
            n0_inline671__ssa_v1: pl.Scalar[pl.INDEX] = d0_inline634__ssa_v0 + nf_inline633__idx_v0 * 256
            acc_b_inline656__tile: pl.Tile[[128, 256], pl.INT32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 131072), pl.Mem.Acc] = pl.tile.create([128, 256], dtype=pl.INT32, target_memory=pl.Mem.Acc)
            for kb_inline646__idx_v0, (acc_b_inline656__iter_v1,) in pl.range(0, 4, 2, init_values=(acc_b_inline656__tile,)):
                k0_inline662__ssa_v1: pl.Scalar[pl.INDEX] = col_g_inline712__ssa_v0 + kb_inline646__idx_v0 * 256
                k0_inline662__ssa_v1_1: pl.Scalar[pl.INDEX] = col_g_inline712__ssa_v0 + (kb_inline646__idx_v0 * 256 + 256)
                b_act_inline691__tile: pl.Tile[[128, 256], pl.INT8, pl.MemRef(mem_mat_4, pl.const(0, pl.INT64), 32768), pl.Mem.Mat] = pl.tile.load(
                    o_r_i8_pad_inline667__rv_v7, [t0_inline635__ssa_v0, k0_inline662__ssa_v1], [128, 256], [128, 256], target_memory=pl.Mem.Mat
                )
                b_weight_inline632__tile: pl.Tile[[256, 256], pl.INT8, pl.MemRef(mem_mat_5, pl.const(32768, pl.INT64), 65536), pl.Mem.Mat] = pl.tile.load(
                    wo_b__ssa_v0, [n0_inline671__ssa_v1, k0_inline662__ssa_v1], [256, 256], [256, 256], target_memory=pl.Mem.Mat
                )
                b_act_inline691__tile_1: pl.Tile[[128, 256], pl.INT8, pl.MemRef(mem_mat_6, pl.const(98304, pl.INT64), 32768), pl.Mem.Mat] = pl.tile.load(
                    o_r_i8_pad_inline667__rv_v7, [t0_inline635__ssa_v0, k0_inline662__ssa_v1_1], [128, 256], [128, 256], target_memory=pl.Mem.Mat
                )
                b_weight_inline632__tile_1: pl.Tile[[256, 256], pl.INT8, pl.MemRef(mem_mat_7, pl.const(131072, pl.INT64), 65536), pl.Mem.Mat] = pl.tile.load(
                    wo_b__ssa_v0, [n0_inline671__ssa_v1, k0_inline662__ssa_v1_1], [256, 256], [256, 256], target_memory=pl.Mem.Mat
                )
                b_weight_inline632__tile_t: pl.Tile[
                    [256, 256], pl.INT8, pl.MemRef(mem_mat_5, pl.const(32768, pl.INT64), 65536), pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)
                ] = pl.tile.transpose_view(b_weight_inline632__tile)
                b_act_inline691__tile_Left: pl.Tile[[128, 256], pl.INT8, pl.MemRef(mem_left_8, pl.const(32768, pl.INT64), 32768), pl.Mem.Left] = pl.tile.move(
                    b_act_inline691__tile, target_memory=pl.Mem.Left
                )
                b_weight_inline632__tile_t_Right: pl.Tile[[256, 256], pl.INT8, pl.MemRef(mem_right_9, pl.const(0, pl.INT64), 65536), pl.Mem.Right] = pl.tile.move(
                    b_weight_inline632__tile_t, target_memory=pl.Mem.Right
                )
                acc_b_inline656__tile_1: pl.Tile[[128, 256], pl.INT32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 131072), pl.Mem.Acc] = pl.tile.matmul_acc(
                    acc_b_inline656__iter_v1, b_act_inline691__tile_Left, b_weight_inline632__tile_t_Right, kb_inline646__idx_v0 == 0
                )
                b_weight_inline632__tile_t_1: pl.Tile[
                    [256, 256], pl.INT8, pl.MemRef(mem_mat_7, pl.const(131072, pl.INT64), 65536), pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)
                ] = pl.tile.transpose_view(b_weight_inline632__tile_1)
                b_act_inline691__tile_Left_1: pl.Tile[[128, 256], pl.INT8, pl.MemRef(mem_left_10, pl.const(0, pl.INT64), 32768), pl.Mem.Left] = pl.tile.move(
                    b_act_inline691__tile_1, target_memory=pl.Mem.Left
                )
                b_weight_inline632__tile_t_Right_1: pl.Tile[[256, 256], pl.INT8, pl.MemRef(mem_right_9, pl.const(0, pl.INT64), 65536), pl.Mem.Right] = pl.tile.move(
                    b_weight_inline632__tile_t_1, target_memory=pl.Mem.Right
                )
                acc_b_inline656__tile_2: pl.Tile[[128, 256], pl.INT32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 131072), pl.Mem.Acc] = pl.tile.matmul_acc(
                    acc_b_inline656__tile_1, b_act_inline691__tile_Left_1, b_weight_inline632__tile_t_Right_1, kb_inline646__idx_v0 == -1
                )
                acc_b_inline656__rv_v2: pl.Tile[[128, 256], pl.INT32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 131072), pl.Mem.Acc] = pl.yield_(acc_b_inline656__tile_2)
            partials_inline684__tile: pl.Tensor[[384, 32768], pl.INT32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 50331648)] = pl.tile.store(
                acc_b_inline656__rv_v2, [t0_inline635__ssa_v0, g_inline708__idx_v0 * 4096 + n0_inline671__ssa_v1], partials_inline684__iter_v3
            )
            partials_inline684__rv_v4: pl.Tensor[[384, 32768], pl.INT32, pl.MemRef("mem_ddr_13", pl.const(0, pl.INT64), 50331648)] = pl.yield_(partials_inline684__tile)
        return partials_inline684__iter_v1

    @pl.function(type=pl.FunctionType.Spmd)
    def proj_b_mm_spmd(
        self,
        partials_inline684__iter_v1: pl.Out[pl.Tensor[[384, 32768], pl.INT32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 50331648)]],
        col_g_inline712__ssa_v0: pl.Scalar[pl.INDEX],
        o_r_i8_pad_inline667__rv_v7: pl.Tensor[[384, 8192], pl.INT8, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 3145728)],
        wo_b__ssa_v0: pl.Tensor[[4096, 8192], pl.INT8, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 33554432)],
        g_inline708__idx_v0: pl.Scalar[pl.INDEX],
    ) -> pl.Tensor[[384, 32768], pl.INT32]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        partials_inline684__rv_v4: pl.Tensor[[384, 32768], pl.INT32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 50331648)] = self.proj_b_mm(
            partials_inline684__iter_v1,
            col_g_inline712__ssa_v0,
            o_r_i8_pad_inline667__rv_v7,
            wo_b__ssa_v0,
            g_inline708__idx_v0,
            attrs={"arg_directions": [pl.adir.output_existing, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.scalar]},
        )
        return partials_inline684__iter_v1

    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def q_rope_prepare(
        rope_cos_il_view_inline1852__ssa_v0: pl.Out[pl.Tensor[[t_dim_inline1855__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)]],
        rope_sin_signed_view_inline1847__ssa_v0: pl.Out[pl.Tensor[[t_dim_inline1855__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)]],
        rope_swap_idx_view_inline1848__ssa_v0: pl.Out[pl.Tensor[[t_dim_inline1855__ssa_v0, 64], pl.INT32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)]],
        token_tiles_inline1841__ssa_v0: pl.Scalar[pl.INDEX],
        t_dim_inline1855__ssa_v0: pl.Scalar[pl.INDEX],
        rope_cos_view_inline1854__ssa_v0: pl.Tensor[[t_dim_inline1855__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)],
        rope_sin_view_inline1842__ssa_v0: pl.Tensor[[t_dim_inline1855__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 0)],
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
        qrp_worker_inline1846__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        qrp_ones_inline1845__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_5, pl.const(16384, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.full([8, 64], dtype=pl.FP32, value=1.0)
        qrp_idx_i32_inline1843__ci_tmp_v0: pl.Tile[[1, 192], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 768), pl.Mem.Vec] = pl.tile.create([1, 192], dtype=pl.FP32, target_memory=pl.Mem.Vec)
        qrp_idx_i32_inline1843__tile: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_13, pl.const(2048, pl.INT64), 256), pl.Mem.Vec] = pl.tile.ci(
            pl.const(0, pl.INT32), [1, 64], tmp=qrp_idx_i32_inline1843__ci_tmp_v0, dtype=pl.INT32, descending=False
        )
        qrp_idx_fp32_inline1838__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 256), pl.Mem.Vec] = pl.tile.cast(
            qrp_idx_i32_inline1843__tile, target_type=pl.FP32, mode="round"
        )
        qrp_col_inline1853__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_5, pl.const(16384, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.col_expand_mul(
            qrp_ones_inline1845__tile, qrp_idx_fp32_inline1838__tile
        )
        qrp_half_inline1840__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.muls(qrp_col_inline1853__tile, 0.5)
        qrp_dup_i32_inline1856__tile: pl.Tile[[8, 64], pl.INT32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(
            qrp_half_inline1840__tile, target_type=pl.INT32, mode="trunc"
        )
        qrp_dup_f_inline1850__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(
            qrp_dup_i32_inline1856__tile, target_type=pl.FP32, mode="round"
        )
        qrp_dup_idx_inline1857__tile: pl.Tile[[8, 64], pl.INT32, pl.MemRef(mem_vec_13, pl.const(2048, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(
            qrp_dup_f_inline1850__tile, target_type=pl.INT32, mode="round"
        )
        t__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.muls(qrp_dup_f_inline1850__tile, 2.0)
        qrp_lane_inline1861__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.sub(qrp_col_inline1853__tile, t__tile)
        qrp_next_col_inline1859__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_5, pl.const(16384, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.adds(qrp_col_inline1853__tile, 1.0)
        qrp_lane_offset_inline1862__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_17, pl.const(4096, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.muls(qrp_lane_inline1861__tile, 2.0)
        qrp_swap_f_inline1844__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_5, pl.const(16384, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.sub(
            qrp_next_col_inline1859__tile, qrp_lane_offset_inline1862__tile
        )
        qrp_swap_idx_inline1870__tile: pl.Tile[[8, 64], pl.INT32, pl.MemRef(mem_vec_5, pl.const(16384, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(
            qrp_swap_f_inline1844__tile, target_type=pl.INT32, mode="round"
        )
        t__tile_1: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.muls(qrp_lane_inline1861__tile, 2.0)
        qrp_sign_inline1869__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.subs(t__tile_1, 1.0)
        for qrp_idx_inline1860__idx_v0, (rope_cos_il_view_inline1852__iter_v1, rope_sin_signed_view_inline1847__iter_v1, rope_swap_idx_view_inline1848__iter_v1) in pl.range(
            qrp_worker_inline1846__ssa_v0,
            token_tiles_inline1841__ssa_v0,
            pl.min(token_tiles_inline1841__ssa_v0, 48),
            init_values=(rope_cos_il_view_inline1852__ssa_v0, rope_sin_signed_view_inline1847__ssa_v0, rope_swap_idx_view_inline1848__ssa_v0),
        ):
            qrp_t0_inline1863__ssa_v0: pl.Scalar[pl.INDEX] = qrp_idx_inline1860__idx_v0 * 8
            qrp_valid_rows_inline1865__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(t_dim_inline1855__ssa_v0 - qrp_t0_inline1863__ssa_v0, 8)
            if qrp_valid_rows_inline1865__ssa_v0 == 8:
                qrp_cos_rows_full_inline1864__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_17, pl.const(4096, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.load(
                    rope_cos_view_inline1854__ssa_v0, [qrp_t0_inline1863__ssa_v0, 0], [8, 64], [8, 64], target_memory=pl.Mem.Vec
                )
                qrp_sin_rows_full_inline1867__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_23, pl.const(6144, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.load(
                    rope_sin_view_inline1842__ssa_v0, [qrp_t0_inline1863__ssa_v0, 0], [8, 64], [8, 64], target_memory=pl.Mem.Vec
                )
                qrp_cos_full_inline1871__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_17, pl.const(4096, pl.INT64), 2048), pl.Mem.Vec] = qrp_cos_rows_full_inline1864__tile
                qrp_sin_full_inline1866__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_23, pl.const(6144, pl.INT64), 2048), pl.Mem.Vec] = qrp_sin_rows_full_inline1867__tile
                gather_acc_init: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_24, pl.const(8192, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.create([8, 64], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                for gather_lv, (gather_ia,) in pl.range(8, init_values=(gather_acc_init,)):
                    gather_inp_row: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_17, pl.const(4096, pl.INT64), 256), pl.Mem.Vec] = pl.tile.slice(
                        qrp_cos_full_inline1871__ssa_v0, [1, 64], [gather_lv, 0], [1, 64]
                    )
                    gather_idx_row: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_13, pl.const(2048, pl.INT64), 256), pl.Mem.Vec] = pl.tile.slice(
                        qrp_dup_idx_inline1857__tile, [1, 64], [gather_lv, 0], [1, 64]
                    )
                    gather_row_tmp: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_40, pl.const(10240, pl.INT64), 256), pl.Mem.Vec] = pl.tile.create([1, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec)
                    gather_row: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_43, pl.const(12288, pl.INT64), 256), pl.Mem.Vec] = pl.tile.gather(gather_inp_row, gather_idx_row, gather_row_tmp)
                    gather_asmbl: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_24, pl.const(8192, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.assemble(gather_ia, gather_row, [gather_lv, 0])
                    gather_rv: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_24, pl.const(8192, pl.INT64), 2048), pl.Mem.Vec] = pl.yield_(gather_asmbl)
                qrp_cos_il_full_inline1858__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_24, pl.const(8192, pl.INT64), 2048), pl.Mem.Vec] = gather_rv
                gather_acc_init_1: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_17, pl.const(4096, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.create([8, 64], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                for gather_lv_1, (gather_ia_1,) in pl.range(8, init_values=(gather_acc_init_1,)):
                    gather_inp_row_1: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_23, pl.const(6144, pl.INT64), 256), pl.Mem.Vec] = pl.tile.slice(
                        qrp_sin_full_inline1866__ssa_v0, [1, 64], [gather_lv_1, 0], [1, 64]
                    )
                    gather_idx_row_1: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_13, pl.const(2048, pl.INT64), 256), pl.Mem.Vec] = pl.tile.slice(
                        qrp_dup_idx_inline1857__tile, [1, 64], [gather_lv_1, 0], [1, 64]
                    )
                    gather_row_tmp_1: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_40, pl.const(10240, pl.INT64), 256), pl.Mem.Vec] = pl.tile.create([1, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec)
                    gather_row_1: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_43, pl.const(12288, pl.INT64), 256), pl.Mem.Vec] = pl.tile.gather(gather_inp_row_1, gather_idx_row_1, gather_row_tmp_1)
                    gather_asmbl_1: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_17, pl.const(4096, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.assemble(gather_ia_1, gather_row_1, [gather_lv_1, 0])
                    gather_rv_1: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_17, pl.const(4096, pl.INT64), 2048), pl.Mem.Vec] = pl.yield_(gather_asmbl_1)
                qrp_sin_il_full_inline1837__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_17, pl.const(4096, pl.INT64), 2048), pl.Mem.Vec] = gather_rv_1
                qrp_sin_signed_full_inline1836__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_17, pl.const(4096, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.mul(
                    qrp_sin_il_full_inline1837__tile, qrp_sign_inline1869__tile
                )
                rope_cos_il_view_inline1852__tile: pl.Tensor[[t_dim_inline1855__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                    qrp_cos_il_full_inline1858__tile, [qrp_t0_inline1863__ssa_v0, 0], rope_cos_il_view_inline1852__iter_v1
                )
                rope_sin_signed_view_inline1847__tile: pl.Tensor[[t_dim_inline1855__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                    qrp_sin_signed_full_inline1836__tile, [qrp_t0_inline1863__ssa_v0, 0], rope_sin_signed_view_inline1847__iter_v1
                )
                rope_swap_idx_view_inline1848__tile: pl.Tensor[[t_dim_inline1855__ssa_v0, 64], pl.INT32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                    qrp_swap_idx_inline1870__tile, [qrp_t0_inline1863__ssa_v0, 0], rope_swap_idx_view_inline1848__iter_v1
                )
                rope_cos_il_view_inline1852__phi_v4, rope_sin_signed_view_inline1847__phi_v4, rope_swap_idx_view_inline1848__phi_v4 = pl.yield_(
                    rope_cos_il_view_inline1852__tile, rope_sin_signed_view_inline1847__tile, rope_swap_idx_view_inline1848__tile
                )
            else:
                qrp_cos_rows_tail_inline1834__ssa_v0: pl.Tile[
                    [8, 64], pl.FP32, pl.MemRef(mem_vec_17, pl.const(4096, pl.INT64), 2048), pl.Mem.Vec, pl.TileView(valid_shape=[qrp_valid_rows_inline1865__ssa_v0, 64])
                ] = pl.tile.load(rope_cos_view_inline1854__ssa_v0, [qrp_t0_inline1863__ssa_v0, 0], [8, 64], [qrp_valid_rows_inline1865__ssa_v0, 64], target_memory=pl.Mem.Vec)
                qrp_sin_rows_tail_inline1832__ssa_v0: pl.Tile[
                    [8, 64], pl.FP32, pl.MemRef(mem_vec_23, pl.const(6144, pl.INT64), 2048), pl.Mem.Vec, pl.TileView(valid_shape=[qrp_valid_rows_inline1865__ssa_v0, 64])
                ] = pl.tile.load(rope_sin_view_inline1842__ssa_v0, [qrp_t0_inline1863__ssa_v0, 0], [8, 64], [qrp_valid_rows_inline1865__ssa_v0, 64], target_memory=pl.Mem.Vec)
                t__tmp_v13: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_24, pl.const(8192, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.full([8, 64], dtype=pl.FP32, value=1.0)
                t__ci_tmp_v1: pl.Tile[[1, 192], pl.FP32, pl.MemRef(mem_vec_40, pl.const(10240, pl.INT64), 768), pl.Mem.Vec] = pl.tile.create([1, 192], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                t__tmp_v14: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_43, pl.const(12288, pl.INT64), 256), pl.Mem.Vec] = pl.tile.ci(
                    pl.const(0, pl.INT32), [1, 64], tmp=t__ci_tmp_v1, dtype=pl.INT32, descending=False
                )
                t__tmp_v15: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_40, pl.const(10240, pl.INT64), 256), pl.Mem.Vec] = pl.tile.cast(t__tmp_v14, target_type=pl.FP32, mode="round")
                qrp_tail_col_inline1839__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_24, pl.const(8192, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.col_expand_mul(t__tmp_v13, t__tmp_v15)
                t__tmp_v16: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_40, pl.const(10240, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.muls(qrp_tail_col_inline1839__ssa_v0, 0.5)
                t__tmp_v17: pl.Tile[[8, 64], pl.INT32, pl.MemRef(mem_vec_40, pl.const(10240, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(t__tmp_v16, target_type=pl.INT32, mode="trunc")
                qrp_tail_dup_f_inline1835__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_40, pl.const(10240, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(
                    t__tmp_v17, target_type=pl.FP32, mode="round"
                )
                t__tmp_v18: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_43, pl.const(12288, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.muls(qrp_tail_dup_f_inline1835__ssa_v0, 2.0)
                qrp_tail_lane_inline1831__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_43, pl.const(12288, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.sub(
                    qrp_tail_col_inline1839__ssa_v0, t__tmp_v18
                )
                t__tmp_v19: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_24, pl.const(8192, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.adds(qrp_tail_col_inline1839__ssa_v0, 1.0)
                t__tmp_v20: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_46, pl.const(14336, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.muls(qrp_tail_lane_inline1831__ssa_v0, 2.0)
                qrp_tail_swap_f_inline1851__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_24, pl.const(8192, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.sub(t__tmp_v19, t__tmp_v20)
                t__ci_tmp_v2: pl.Tile[[1, 192], pl.FP32, pl.MemRef(mem_vec_46, pl.const(14336, pl.INT64), 768), pl.Mem.Vec] = pl.tile.create([1, 192], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                t__tmp_v21: pl.Tile[[1, 8], pl.INT32, pl.MemRef(mem_vec_54, pl.const(18432, pl.INT64), 32), pl.Mem.Vec] = pl.tile.ci(
                    pl.const(0, pl.INT32), [1, 8], tmp=t__ci_tmp_v2, dtype=pl.INT32, descending=False
                )
                t__tmp_v22: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_46, pl.const(14336, pl.INT64), 32), pl.Mem.Vec] = pl.tile.cast(t__tmp_v21, target_type=pl.FP32, mode="round")
                qrp_row_seed_inline1830__ssa_v0: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_54, pl.const(18432, pl.INT64), 32), pl.Mem.Vec] = pl.tile.muls(t__tmp_v22, 64.0)
                t__tmp_v23: pl.Tile[[64, 8], pl.FP32, pl.MemRef(mem_vec_46, pl.const(14336, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.full([64, 8], dtype=pl.FP32, value=1.0)
                qrp_row_grid_inline1829__ssa_v0: pl.Tile[[64, 8], pl.FP32, pl.MemRef(mem_vec_46, pl.const(14336, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.col_expand_mul(
                    t__tmp_v23, qrp_row_seed_inline1830__ssa_v0
                )
                transpose_tmp: pl.Tile[[64, 8], pl.FP32, pl.MemRef(mem_vec_54, pl.const(18432, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.create([64, 8], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                qrp_row_offset_inline1833__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_55, pl.const(20480, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.transpose(
                    qrp_row_grid_inline1829__ssa_v0, 0, 1, transpose_tmp
                )
                t__tmp_v24: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_40, pl.const(10240, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.add(
                    qrp_tail_dup_f_inline1835__ssa_v0, qrp_row_offset_inline1833__ssa_v0
                )
                qrp_dup_idx_tail_inline1828__ssa_v0: pl.Tile[[8, 64], pl.INT32, pl.MemRef(mem_vec_40, pl.const(10240, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(
                    t__tmp_v24, target_type=pl.INT32, mode="round"
                )
                qrp_gather_tmp_inline1827__ssa_v0: pl.Tile[[8, 64], pl.INT32, pl.MemRef(mem_vec_46, pl.const(14336, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.create(
                    [8, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec
                )
                qrp_cos_il_tail_inline1868__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_54, pl.const(18432, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.gather(
                    qrp_cos_rows_tail_inline1834__ssa_v0, qrp_dup_idx_tail_inline1828__ssa_v0, qrp_gather_tmp_inline1827__ssa_v0
                )
                qrp_sin_il_tail_inline1849__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_17, pl.const(4096, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.gather(
                    qrp_sin_rows_tail_inline1832__ssa_v0, qrp_dup_idx_tail_inline1828__ssa_v0, qrp_gather_tmp_inline1827__ssa_v0
                )
                t__tmp_v25: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_23, pl.const(6144, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.muls(qrp_tail_lane_inline1831__ssa_v0, 2.0)
                qrp_tail_sign_inline1826__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_23, pl.const(6144, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.subs(t__tmp_v25, 1.0)
                qrp_sin_signed_tail_inline1825__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_17, pl.const(4096, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.mul(
                    qrp_sin_il_tail_inline1849__ssa_v0, qrp_tail_sign_inline1826__ssa_v0
                )
                t__tmp_v26: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_54, pl.const(18432, pl.INT64), 2048), pl.Mem.Vec, pl.TileView(valid_shape=[qrp_valid_rows_inline1865__ssa_v0, 64])] = (
                    pl.tile.set_validshape(qrp_cos_il_tail_inline1868__ssa_v0, qrp_valid_rows_inline1865__ssa_v0, 64)
                )
                pl.tile.store(t__tmp_v26, [qrp_t0_inline1863__ssa_v0, 0], rope_cos_il_view_inline1852__iter_v1)
                t__tmp_v27: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_17, pl.const(4096, pl.INT64), 2048), pl.Mem.Vec, pl.TileView(valid_shape=[qrp_valid_rows_inline1865__ssa_v0, 64])] = (
                    pl.tile.set_validshape(qrp_sin_signed_tail_inline1825__ssa_v0, qrp_valid_rows_inline1865__ssa_v0, 64)
                )
                pl.tile.store(t__tmp_v27, [qrp_t0_inline1863__ssa_v0, 0], rope_sin_signed_view_inline1847__iter_v1)
                t__tmp_v28: pl.Tile[[8, 64], pl.INT32, pl.MemRef(mem_vec_17, pl.const(4096, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(
                    qrp_tail_swap_f_inline1851__ssa_v0, target_type=pl.INT32, mode="round"
                )
                t__tmp_v29: pl.Tile[[8, 64], pl.INT32, pl.MemRef(mem_vec_17, pl.const(4096, pl.INT64), 2048), pl.Mem.Vec, pl.TileView(valid_shape=[qrp_valid_rows_inline1865__ssa_v0, 64])] = (
                    pl.tile.set_validshape(t__tmp_v28, qrp_valid_rows_inline1865__ssa_v0, 64)
                )
                pl.tile.store(t__tmp_v29, [qrp_t0_inline1863__ssa_v0, 0], rope_swap_idx_view_inline1848__iter_v1)
                rope_cos_il_view_inline1852__phi_v4, rope_sin_signed_view_inline1847__phi_v4, rope_swap_idx_view_inline1848__phi_v4 = pl.yield_(
                    rope_cos_il_view_inline1852__iter_v1, rope_sin_signed_view_inline1847__iter_v1, rope_swap_idx_view_inline1848__iter_v1
                )
            rope_cos_il_view_inline1852__rv_v2, rope_sin_signed_view_inline1847__rv_v2, rope_swap_idx_view_inline1848__rv_v2 = pl.yield_(
                rope_cos_il_view_inline1852__phi_v4, rope_sin_signed_view_inline1847__phi_v4, rope_swap_idx_view_inline1848__phi_v4
            )
        return

    @pl.function(type=pl.FunctionType.Spmd)
    def q_rope_prepare_spmd(
        self,
        rope_cos_il_view_inline1852__ssa_v0: pl.Out[pl.Tensor[[t_dim_inline1855__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)]],
        rope_sin_signed_view_inline1847__ssa_v0: pl.Out[pl.Tensor[[t_dim_inline1855__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)]],
        rope_swap_idx_view_inline1848__ssa_v0: pl.Out[pl.Tensor[[t_dim_inline1855__ssa_v0, 64], pl.INT32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)]],
        token_tiles_inline1841__ssa_v0: pl.Scalar[pl.INDEX],
        t_dim_inline1855__ssa_v0: pl.Scalar[pl.INDEX],
        rope_cos_view_inline1854__ssa_v0: pl.Tensor[[t_dim_inline1855__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)],
        rope_sin_view_inline1842__ssa_v0: pl.Tensor[[t_dim_inline1855__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 0)],
    ):
        pl.func_attr({"mx_tensor_views_blocked": True})
        self.q_rope_prepare(
            rope_cos_il_view_inline1852__ssa_v0,
            rope_sin_signed_view_inline1847__ssa_v0,
            rope_swap_idx_view_inline1848__ssa_v0,
            token_tiles_inline1841__ssa_v0,
            t_dim_inline1855__ssa_v0,
            rope_cos_view_inline1854__ssa_v0,
            rope_sin_view_inline1842__ssa_v0,
            attrs={"arg_directions": [pl.adir.output_existing, pl.adir.output_existing, pl.adir.output_existing, pl.adir.scalar, pl.adir.scalar, pl.adir.input, pl.adir.input]},
        )

    @pl.function(type=pl.FunctionType.AIC, level=pl.Level.AIC, role=pl.Role.SubWorker)
    def qk_pv_aic(
        ffts_workspace_inline2458__ssa_v0: pl.Tensor[[256], pl.INT64, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 2048)],
        t_dim_inline2515__ssa_v0: pl.Scalar[pl.INDEX],
        q_flat_inline2539__ssa_v0: pl.Tensor[[t_heads_inline2521__ssa_v0, 512], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)],
        valid_block_mask_inline2517__ssa_v0: pl.Tensor[[t_dim_inline2515__ssa_v0, 16], pl.INT32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)],
        kv_transfer_inline2476__ssa_v0: pl.InOut[pl.Tensor[[9216, 512], pl.BF16, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 9437184)]],
        score_transfer_inline2473__ssa_v0: pl.InOut[pl.Tensor[[4608, 128], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 2359296)]],
        probability_transfer_inline2461__ssa_v0: pl.InOut[pl.Tensor[[4608, 128], pl.BF16, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 1179648)]],
        pv_transfer_inline2471__ssa_v0: pl.InOut[pl.Tensor[[4608, 512], pl.FP32, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 9437184)]],
        attn_sink_col_inline2487__ssa_v0: pl.Tensor[[64, 1], pl.FP32, pl.MemRef("mem_ddr_7", pl.const(0, pl.INT64), 256)],
        position_ids_t1__ssa_v0: pl.Tensor[[T_DYN, 1], pl.INT32, pl.MemRef("mem_ddr_8", pl.const(0, pl.INT64), 0)],
        window_swa_indices__ssa_v0: pl.Tensor[[T_DYN, 128], pl.INT32, pl.MemRef("mem_ddr_9", pl.const(0, pl.INT64), 0)],
        ori_kv_flat_inline2519__ssa_v1: pl.Tensor[[ori_block_num_inline2509__ssa_v0 * pl.const(32, pl.INDEX), 512], pl.BF16, pl.MemRef("mem_ddr_10", pl.const(0, pl.INT64), 0)],
        cmp_sparse_indices_inline2527__rv_v2: pl.Tensor[[t_dim_inline2515__ssa_v0, 512], pl.INT32, pl.MemRef("mem_ddr_11", pl.const(0, pl.INT64), 0)],
        cmp_block_table__ssa_v0: pl.Tensor[[B_DYN, COMPRESSED_TABLE_COLUMNS_DYN], pl.INT32, pl.MemRef("mem_ddr_12", pl.const(0, pl.INT64), 0)],
        cmp_kv_flat_inline2564__ssa_v0: pl.Tensor[[cmp_block_num_inline2493__ssa_v0 * pl.const(32, pl.INDEX), 512], pl.BF16, pl.MemRef("mem_ddr_13", pl.const(0, pl.INT64), 0)],
        sparse_bias_inline2513__rv_v2: pl.Tensor[[t_dim_inline2515__ssa_v0, 640], pl.FP32, pl.MemRef("mem_ddr_14", pl.const(0, pl.INT64), 0)],
        mi_transfer_inline2470__ssa_v0: pl.InOut[pl.Tensor[[4608, 1], pl.FP32, pl.MemRef("mem_ddr_15", pl.const(0, pl.INT64), 18432)]],
        li_transfer_inline2478__ssa_v0: pl.InOut[pl.Tensor[[4608, 1], pl.FP32, pl.MemRef("mem_ddr_16", pl.const(0, pl.INT64), 18432)]],
        attn_mi_inline2536__ssa_v0: pl.Out[pl.Tensor[[t_heads_inline2521__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_17", pl.const(0, pl.INT64), 0)]],
        attn_li_inline2559__ssa_v0: pl.Out[pl.Tensor[[t_heads_inline2521__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_18", pl.const(0, pl.INT64), 0)]],
        attn_oi_inline2479__ssa_v0: pl.Out[pl.Tensor[[t_heads_inline2521__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_19", pl.const(0, pl.INT64), 0)]],
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
        qk_core_inline2560__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        pl.system.set_ffts(ffts_workspace_inline2458__ssa_v0)
        for qk_t_inline2491__idx_v0 in pl.range(qk_core_inline2560__ssa_v0, t_dim_inline2515__ssa_v0, 24):
            qk_q_inline2489__ssa_v0: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_mat_20, pl.const(0, pl.INT64), 65536), pl.Mem.Mat] = pl.tile.load(
                q_flat_inline2539__ssa_v0, [qk_t_inline2491__idx_v0 * 64, 0], [64, 512], [64, 512], target_memory=pl.Mem.Mat
            )
            qk_l1_inline2463__ssa_v0: pl.Tile[[384, 512], pl.BF16, pl.MemRef(mem_mat_21, pl.const(65536, pl.INT64), 393216), pl.Mem.Mat] = pl.tile.create(
                [384, 512], dtype=pl.BF16, target_memory=pl.Mem.Mat
            )
            for qk_tick_inline2516__idx_v0, (qk_l1_inline2463__iter_v1,) in pl.range(7, init_values=(qk_l1_inline2463__ssa_v0,)):
                if qk_tick_inline2516__idx_v0 < 5:
                    qk_sb_inline2542__ssa_v0: pl.Scalar[pl.INDEX] = qk_tick_inline2516__idx_v0
                    t__tile: pl.Scalar[pl.INT32] = pl.tensor.read(valid_block_mask_inline2517__ssa_v0, [qk_t_inline2491__idx_v0, qk_sb_inline2542__ssa_v0])
                    if 0 < pl.cast(t__tile, pl.INDEX):
                        qk_slot_inline2512__ssa_v0: pl.Scalar[pl.INDEX] = qk_core_inline2560__ssa_v0 * 3 + qk_sb_inline2542__ssa_v0 % 3
                        qk_kv_row_inline2457__ssa_v0: pl.Scalar[pl.INDEX] = qk_slot_inline2512__ssa_v0 * 128
                        qk_transfer_row_inline2514__ssa_v0: pl.Scalar[pl.INDEX] = qk_slot_inline2512__ssa_v0 * 64
                        pl.system.sync_wait(0, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIC)
                        qk_l1_row_inline2538__ssa_v0: pl.Scalar[pl.INDEX] = qk_sb_inline2542__ssa_v0 % 3 * 128
                        qk_l1_inline2463__ssa_v3: pl.Tile[[384, 512], pl.BF16, pl.MemRef(mem_mat_21, pl.const(65536, pl.INT64), 393216), pl.Mem.Mat] = pl.tile.gather_row(
                            qk_l1_inline2463__iter_v1, kv_transfer_inline2476__ssa_v0, [qk_l1_row_inline2538__ssa_v0, 0], [qk_kv_row_inline2457__ssa_v0, 0], [128, 512], transpose=False
                        )
                        qk_l1_t_inline2499__ssa_v0: pl.Tile[
                            [512, 384], pl.BF16, pl.MemRef(mem_mat_21, pl.const(65536, pl.INT64), 393216), pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)
                        ] = pl.tile.transpose_view(qk_l1_inline2463__ssa_v3)
                        qk_scores_inline2550__ssa_v0_l0_init: pl.Tile[[64, 128], pl.FP32, pl.MemRef(mem_acc_31, pl.const(0, pl.INT64), 32768), pl.Mem.Acc] = pl.tile.create(
                            [64, 128], dtype=pl.FP32, target_memory=pl.Mem.Acc
                        )
                        for qk_scores_inline2550__ssa_v0_l0_ko, (qk_scores_inline2550__ssa_v0_l0_c,) in pl.range(0, 512, 256, init_values=(qk_scores_inline2550__ssa_v0_l0_init,)):
                            qk_scores_inline2550__ssa_v0_l0_a: pl.Tile[
                                [64, 128], pl.BF16, pl.MemRef(mem_left_23, pl.const(0, pl.INT64), 16384), pl.Mem.Left, pl.TileView(blayout=pl.TileLayout.row_major)
                            ] = pl.tile.extract(qk_q_inline2489__ssa_v0, 0, qk_scores_inline2550__ssa_v0_l0_ko, [64, 128], target_memory=pl.Mem.Left)
                            qk_scores_inline2550__ssa_v0_l0_b: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_24, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                                qk_l1_t_inline2499__ssa_v0, qk_scores_inline2550__ssa_v0_l0_ko, qk_l1_row_inline2538__ssa_v0, [128, 128], target_memory=pl.Mem.Right
                            )
                            qk_scores_inline2550__ssa_v0_l0_a_1: pl.Tile[
                                [64, 128], pl.BF16, pl.MemRef(mem_left_25, pl.const(16384, pl.INT64), 16384), pl.Mem.Left, pl.TileView(blayout=pl.TileLayout.row_major)
                            ] = pl.tile.extract(qk_q_inline2489__ssa_v0, 0, qk_scores_inline2550__ssa_v0_l0_ko + 128, [64, 128], target_memory=pl.Mem.Left)
                            qk_scores_inline2550__ssa_v0_l0_b_1: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_26, pl.const(32768, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                                qk_l1_t_inline2499__ssa_v0, qk_scores_inline2550__ssa_v0_l0_ko + 128, qk_l1_row_inline2538__ssa_v0, [128, 128], target_memory=pl.Mem.Right
                            )
                            qk_scores_inline2550__ssa_v0_l0_c_acc: pl.Tile[[64, 128], pl.FP32, pl.MemRef(mem_acc_31, pl.const(0, pl.INT64), 32768), pl.Mem.Acc] = pl.tile.matmul_acc(
                                qk_scores_inline2550__ssa_v0_l0_c, qk_scores_inline2550__ssa_v0_l0_a, qk_scores_inline2550__ssa_v0_l0_b, qk_scores_inline2550__ssa_v0_l0_ko == 0
                            )
                            qk_scores_inline2550__ssa_v0_l0_c_acc_1: pl.Tile[[64, 128], pl.FP32, pl.MemRef(mem_acc_31, pl.const(0, pl.INT64), 32768), pl.Mem.Acc] = pl.tile.matmul_acc(
                                qk_scores_inline2550__ssa_v0_l0_c_acc, qk_scores_inline2550__ssa_v0_l0_a_1, qk_scores_inline2550__ssa_v0_l0_b_1, qk_scores_inline2550__ssa_v0_l0_ko == -128
                            )
                            qk_scores_inline2550__ssa_v0: pl.Tile[[64, 128], pl.FP32, pl.MemRef(mem_acc_31, pl.const(0, pl.INT64), 32768), pl.Mem.Acc] = pl.yield_(
                                qk_scores_inline2550__ssa_v0_l0_c_acc_1
                            )
                        score_transfer_inline2473__store: pl.Tensor[[4608, 128], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 2359296)] = pl.tile.store(
                            qk_scores_inline2550__ssa_v0, [qk_transfer_row_inline2514__ssa_v0, 0], score_transfer_inline2473__ssa_v0
                        )
                        pl.system.sync_set(1, pipe=pl.PipeType.FIX, ffts_mode=2, core_type=pl.KernelType.AIC)
                        qk_l1_inline2463__phi_v4: pl.Tile[[384, 512], pl.BF16, pl.MemRef(mem_mat_21, pl.const(65536, pl.INT64), 393216), pl.Mem.Mat] = pl.yield_(qk_l1_inline2463__ssa_v3)
                    else:
                        qk_l1_inline2463__phi_v4: pl.Tile[[384, 512], pl.BF16, pl.MemRef(mem_mat_21, pl.const(65536, pl.INT64), 393216), pl.Mem.Mat] = pl.yield_(qk_l1_inline2463__iter_v1)
                    qk_l1_inline2463__phi_v5: pl.Tile[[384, 512], pl.BF16, pl.MemRef(mem_mat_21, pl.const(65536, pl.INT64), 393216), pl.Mem.Mat] = pl.yield_(qk_l1_inline2463__phi_v4)
                else:
                    qk_l1_inline2463__phi_v5: pl.Tile[[384, 512], pl.BF16, pl.MemRef(mem_mat_21, pl.const(65536, pl.INT64), 393216), pl.Mem.Mat] = pl.yield_(qk_l1_inline2463__iter_v1)
                if 2 <= qk_tick_inline2516__idx_v0:
                    pv_sb_inline2552__ssa_v0: pl.Scalar[pl.INDEX] = qk_tick_inline2516__idx_v0 - 2
                    t__tile_1: pl.Scalar[pl.INT32] = pl.tensor.read(valid_block_mask_inline2517__ssa_v0, [qk_t_inline2491__idx_v0, pv_sb_inline2552__ssa_v0])
                    if 0 < pl.cast(t__tile_1, pl.INDEX):
                        pv_slot_inline2592__ssa_v0: pl.Scalar[pl.INDEX] = qk_core_inline2560__ssa_v0 * 3 + pv_sb_inline2552__ssa_v0 % 3
                        pv_transfer_row_inline2556__ssa_v0: pl.Scalar[pl.INDEX] = pv_slot_inline2592__ssa_v0 * 64
                        pl.system.sync_wait(2, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIC)
                        pv_probability_inline2558__ssa_v0: pl.Tile[[64, 128], pl.BF16, pl.MemRef(mem_mat_30, pl.const(458752, pl.INT64), 16384), pl.Mem.Mat] = pl.tile.load(
                            probability_transfer_inline2461__ssa_v0, [pv_transfer_row_inline2556__ssa_v0, 0], [64, 128], [64, 128], target_memory=pl.Mem.Mat
                        )
                        pv_l1_row_inline2562__ssa_v0: pl.Scalar[pl.INDEX] = pv_sb_inline2552__ssa_v0 % 3 * 128
                        pv_output_inline2482__ssa_v0_l0_init: pl.Tile[[64, 512], pl.FP32, pl.MemRef(mem_acc_31, pl.const(0, pl.INT64), 131072), pl.Mem.Acc] = pl.tile.create(
                            [64, 512], dtype=pl.FP32, target_memory=pl.Mem.Acc
                        )
                        for pv_output_inline2482__ssa_v0_l0_ko, (pv_output_inline2482__ssa_v0_l0_c,) in pl.range(0, 128, 64, init_values=(pv_output_inline2482__ssa_v0_l0_init,)):
                            pv_output_inline2482__ssa_v0_l0_a: pl.Tile[
                                [64, 32], pl.BF16, pl.MemRef(mem_left_23, pl.const(0, pl.INT64), 4096), pl.Mem.Left, pl.TileView(blayout=pl.TileLayout.row_major)
                            ] = pl.tile.extract(pv_probability_inline2558__ssa_v0, 0, pv_output_inline2482__ssa_v0_l0_ko, [64, 32], target_memory=pl.Mem.Left)
                            pv_output_inline2482__ssa_v0_l0_b: pl.Tile[[32, 512], pl.BF16, pl.MemRef(mem_right_24, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                                qk_l1_inline2463__phi_v5, pv_output_inline2482__ssa_v0_l0_ko + pv_l1_row_inline2562__ssa_v0, 0, [32, 512], target_memory=pl.Mem.Right
                            )
                            pv_output_inline2482__ssa_v0_l0_a_1: pl.Tile[
                                [64, 32], pl.BF16, pl.MemRef(mem_left_25, pl.const(16384, pl.INT64), 4096), pl.Mem.Left, pl.TileView(blayout=pl.TileLayout.row_major)
                            ] = pl.tile.extract(pv_probability_inline2558__ssa_v0, 0, pv_output_inline2482__ssa_v0_l0_ko + 32, [64, 32], target_memory=pl.Mem.Left)
                            pv_output_inline2482__ssa_v0_l0_b_1: pl.Tile[[32, 512], pl.BF16, pl.MemRef(mem_right_26, pl.const(32768, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                                qk_l1_inline2463__phi_v5, pv_output_inline2482__ssa_v0_l0_ko + pv_l1_row_inline2562__ssa_v0 + 32, 0, [32, 512], target_memory=pl.Mem.Right
                            )
                            pv_output_inline2482__ssa_v0_l0_c_acc: pl.Tile[[64, 512], pl.FP32, pl.MemRef(mem_acc_31, pl.const(0, pl.INT64), 131072), pl.Mem.Acc] = pl.tile.matmul_acc(
                                pv_output_inline2482__ssa_v0_l0_c, pv_output_inline2482__ssa_v0_l0_a, pv_output_inline2482__ssa_v0_l0_b, pv_output_inline2482__ssa_v0_l0_ko == 0
                            )
                            pv_output_inline2482__ssa_v0_l0_c_acc_1: pl.Tile[[64, 512], pl.FP32, pl.MemRef(mem_acc_31, pl.const(0, pl.INT64), 131072), pl.Mem.Acc] = pl.tile.matmul_acc(
                                pv_output_inline2482__ssa_v0_l0_c_acc, pv_output_inline2482__ssa_v0_l0_a_1, pv_output_inline2482__ssa_v0_l0_b_1, pv_output_inline2482__ssa_v0_l0_ko == -32
                            )
                            pv_output_inline2482__ssa_v0: pl.Tile[[64, 512], pl.FP32, pl.MemRef(mem_acc_31, pl.const(0, pl.INT64), 131072), pl.Mem.Acc] = pl.yield_(
                                pv_output_inline2482__ssa_v0_l0_c_acc_1
                            )
                        pv_transfer_inline2471__store: pl.Tensor[[4608, 512], pl.FP32, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 9437184)] = pl.tile.store(
                            pv_output_inline2482__ssa_v0, [pv_transfer_row_inline2556__ssa_v0, 0], pv_transfer_inline2471__ssa_v0
                        )
                        pl.system.sync_set(3, pipe=pl.PipeType.FIX, ffts_mode=2, core_type=pl.KernelType.AIC)
                qk_l1_inline2463__rv_v2: pl.Tile[[384, 512], pl.BF16, pl.MemRef(mem_mat_21, pl.const(65536, pl.INT64), 393216), pl.Mem.Mat] = pl.yield_(qk_l1_inline2463__phi_v5)
            pl.system.set_ffts(ffts_workspace_inline2458__ssa_v0)

    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def qk_pv_aiv(
        ffts_workspace_inline2458__ssa_v0: pl.Tensor[[256], pl.INT64, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 2048)],
        t_dim_inline2515__ssa_v0: pl.Scalar[pl.INDEX],
        q_flat_inline2539__ssa_v0: pl.Tensor[[t_heads_inline2521__ssa_v0, 512], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)],
        valid_block_mask_inline2517__ssa_v0: pl.Tensor[[t_dim_inline2515__ssa_v0, 16], pl.INT32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)],
        kv_transfer_inline2476__ssa_v0: pl.InOut[pl.Tensor[[9216, 512], pl.BF16, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 9437184)]],
        score_transfer_inline2473__ssa_v0: pl.InOut[pl.Tensor[[4608, 128], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 2359296)]],
        probability_transfer_inline2461__ssa_v0: pl.InOut[pl.Tensor[[4608, 128], pl.BF16, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 1179648)]],
        pv_transfer_inline2471__ssa_v0: pl.InOut[pl.Tensor[[4608, 512], pl.FP32, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 9437184)]],
        attn_sink_col_inline2487__ssa_v0: pl.Tensor[[64, 1], pl.FP32, pl.MemRef("mem_ddr_7", pl.const(0, pl.INT64), 256)],
        position_ids_t1__ssa_v0: pl.Tensor[[T_DYN, 1], pl.INT32, pl.MemRef("mem_ddr_8", pl.const(0, pl.INT64), 0)],
        window_swa_indices__ssa_v0: pl.Tensor[[T_DYN, 128], pl.INT32, pl.MemRef("mem_ddr_9", pl.const(0, pl.INT64), 0)],
        ori_kv_flat_inline2519__ssa_v1: pl.Tensor[[ori_block_num_inline2509__ssa_v0 * pl.const(32, pl.INDEX), 512], pl.BF16, pl.MemRef("mem_ddr_10", pl.const(0, pl.INT64), 0)],
        cmp_sparse_indices_inline2527__rv_v2: pl.Tensor[[t_dim_inline2515__ssa_v0, 512], pl.INT32, pl.MemRef("mem_ddr_11", pl.const(0, pl.INT64), 0)],
        cmp_block_table__ssa_v0: pl.Tensor[[B_DYN, COMPRESSED_TABLE_COLUMNS_DYN], pl.INT32, pl.MemRef("mem_ddr_12", pl.const(0, pl.INT64), 0)],
        cmp_kv_flat_inline2564__ssa_v0: pl.Tensor[[cmp_block_num_inline2493__ssa_v0 * pl.const(32, pl.INDEX), 512], pl.BF16, pl.MemRef("mem_ddr_13", pl.const(0, pl.INT64), 0)],
        sparse_bias_inline2513__rv_v2: pl.Tensor[[t_dim_inline2515__ssa_v0, 640], pl.FP32, pl.MemRef("mem_ddr_14", pl.const(0, pl.INT64), 0)],
        mi_transfer_inline2470__ssa_v0: pl.InOut[pl.Tensor[[4608, 1], pl.FP32, pl.MemRef("mem_ddr_15", pl.const(0, pl.INT64), 18432)]],
        li_transfer_inline2478__ssa_v0: pl.InOut[pl.Tensor[[4608, 1], pl.FP32, pl.MemRef("mem_ddr_16", pl.const(0, pl.INT64), 18432)]],
        attn_mi_inline2536__ssa_v0: pl.Out[pl.Tensor[[t_heads_inline2521__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_17", pl.const(0, pl.INT64), 0)]],
        attn_li_inline2559__ssa_v0: pl.Out[pl.Tensor[[t_heads_inline2521__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_18", pl.const(0, pl.INT64), 0)]],
        attn_oi_inline2479__ssa_v0: pl.Out[pl.Tensor[[t_heads_inline2521__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_19", pl.const(0, pl.INT64), 0)]],
    ) -> tuple[
        pl.Tensor[[9216, 512], pl.BF16],
        pl.Tensor[[4608, 128], pl.FP32],
        pl.Tensor[[4608, 128], pl.BF16],
        pl.Tensor[[4608, 512], pl.FP32],
        pl.Tensor[[4608, 1], pl.FP32],
        pl.Tensor[[4608, 1], pl.FP32],
        pl.Tensor[[t_heads_inline2521__ssa_v0, 1], pl.FP32],
        pl.Tensor[[t_heads_inline2521__ssa_v0, 1], pl.FP32],
        pl.Tensor[[t_heads_inline2521__ssa_v0, 512], pl.FP32],
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
        qk_core_inline2560__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        pl.system.set_ffts(ffts_workspace_inline2458__ssa_v0)
        for qk_t_inline2491__idx_v0 in pl.range(qk_core_inline2560__ssa_v0, t_dim_inline2515__ssa_v0, 24):
            qk_b_inline2460__ssa_v0: pl.Scalar[pl.INDEX] = qk_t_inline2491__idx_v0 // 6
            qk_aiv_inline2549__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_subblock_idx()
            pl.system.set_ffts(ffts_workspace_inline2458__ssa_v0)
            qk_lane_head_inline2563__ssa_v0: pl.Scalar[pl.INDEX] = qk_aiv_inline2549__ssa_v0 * 32
            qk_lane_kv_inline2497__ssa_v0: pl.Scalar[pl.INDEX] = qk_aiv_inline2549__ssa_v0 * 64
            qk_reduce_tmp_inline2548__ssa_v0: pl.Tile[[32, 128], pl.FP32, pl.MemRef(mem_vec_20, pl.const(0, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.create(
                [32, 128], dtype=pl.FP32, target_memory=pl.Mem.Vec
            )
            running_m_inline2565__ssa_v0: pl.Tile[[32, 1], pl.FP32, pl.MemRef(mem_vec_21, pl.const(16384, pl.INT64), 128), pl.Mem.Vec] = pl.tile.load(
                attn_sink_col_inline2487__ssa_v0, [qk_lane_head_inline2563__ssa_v0, 0], [32, 1], [32, 1], target_memory=pl.Mem.Vec
            )
            running_l_inline2567__rm_a0_tmp_v0: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_21, pl.const(16384, pl.INT64), 128), pl.Mem.Vec] = pl.tile.reshape(running_m_inline2565__ssa_v0, [1, 32])
            running_l_inline2567__row_major_tmp_v1: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_22, pl.const(16512, pl.INT64), 128), pl.Mem.Vec] = pl.tile.muls(running_l_inline2567__rm_a0_tmp_v0, 0.0)
            running_l_inline2567__ssa_v0: pl.Tile[[32, 1], pl.FP32, pl.MemRef(mem_vec_22, pl.const(16512, pl.INT64), 128), pl.Mem.Vec] = pl.tile.reshape(
                running_l_inline2567__row_major_tmp_v1, [32, 1]
            )
            running_left_inline2569__ssa_v0: pl.Tile[[32, 256], pl.FP32, pl.MemRef(mem_vec_23, pl.const(16640, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.full([32, 256], dtype=pl.FP32, value=0.0)
            running_right_inline2477__ssa_v0: pl.Tile[[32, 256], pl.FP32, pl.MemRef(mem_vec_24, pl.const(49408, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.full([32, 256], dtype=pl.FP32, value=0.0)
            for qk_tick_inline2465__idx_v0, (m_iter, l_iter, left_iter, right_iter) in pl.range(
                8, init_values=(running_m_inline2565__ssa_v0, running_l_inline2567__ssa_v0, running_left_inline2569__ssa_v0, running_right_inline2477__ssa_v0)
            ):
                if qk_tick_inline2465__idx_v0 < 5:
                    qk_sb_inline2542__ssa_v1: pl.Scalar[pl.INDEX] = qk_tick_inline2465__idx_v0
                    t__tile: pl.Scalar[pl.INT32] = pl.tensor.read(valid_block_mask_inline2517__ssa_v0, [qk_t_inline2491__idx_v0, qk_sb_inline2542__ssa_v1])
                    if 0 < pl.cast(t__tile, pl.INDEX):
                        qk_slot_inline2512__ssa_v1: pl.Scalar[pl.INDEX] = qk_core_inline2560__ssa_v0 * 3 + qk_sb_inline2542__ssa_v1 % 3
                        qk_kv_row_inline2457__ssa_v1: pl.Scalar[pl.INDEX] = qk_slot_inline2512__ssa_v1 * 128
                        qk_s0_inline2495__ssa_v0: pl.Scalar[pl.INDEX] = qk_sb_inline2542__ssa_v1 * 128
                        qk_kv_half_inline2577__ssa_v0: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.tile.full(
                            [64, 512], dtype=pl.BF16, value=0.0
                        )
                        if qk_s0_inline2495__ssa_v0 < 128:
                            t__tile_1: pl.Scalar[pl.INT32] = pl.tensor.read(position_ids_t1__ssa_v0, [qk_t_inline2491__idx_v0, 0])
                            qk_pos_inline2557__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(t__tile_1, pl.INDEX)
                            qk_win_len_inline2580__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(qk_pos_inline2557__ssa_v0 + 1, 128)
                            qk_win_start_inline2582__ssa_v0: pl.Scalar[pl.INDEX] = qk_pos_inline2557__ssa_v0 - qk_win_len_inline2580__ssa_v0 + 1
                            qk_head_inline2534__ssa_v0: pl.Scalar[pl.INDEX] = (qk_win_start_inline2582__ssa_v0 + qk_s0_inline2495__ssa_v0) % 32
                            qk_rows_inline2583__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(pl.max(qk_win_len_inline2580__ssa_v0 - qk_s0_inline2495__ssa_v0 - qk_lane_kv_inline2497__ssa_v0, 0), 64)
                            qk_lo_inline2585__ssa_v0: pl.Scalar[pl.INDEX] = 0
                            qk_hi_inline2468__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(32 - qk_head_inline2534__ssa_v0 - qk_lane_kv_inline2497__ssa_v0, qk_rows_inline2583__ssa_v0)
                            if qk_lo_inline2585__ssa_v0 < qk_hi_inline2468__ssa_v0:
                                qk_raw_row_inline2483__tile: pl.Scalar[pl.INT32] = pl.tensor.read(
                                    window_swa_indices__ssa_v0, [qk_t_inline2491__idx_v0, qk_s0_inline2495__ssa_v0 + qk_lane_kv_inline2497__ssa_v0 + qk_lo_inline2585__ssa_v0]
                                )
                                if 0 <= pl.cast(qk_raw_row_inline2483__tile, pl.INDEX):
                                    qk_kv_half_inline2577__ssa_v1: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.tile.gather_row(
                                        qk_kv_half_inline2577__ssa_v0,
                                        ori_kv_flat_inline2519__ssa_v1,
                                        [qk_lo_inline2585__ssa_v0, 0],
                                        [qk_raw_row_inline2483__tile, 0],
                                        [64, 512],
                                        valid_shape=[qk_hi_inline2468__ssa_v0 - qk_lo_inline2585__ssa_v0, 512],
                                        transpose=False,
                                    )
                                    qk_kv_half_inline2577__phi_v2: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(
                                        qk_kv_half_inline2577__ssa_v1
                                    )
                                else:
                                    qk_kv_half_inline2577__phi_v2: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(
                                        qk_kv_half_inline2577__ssa_v0
                                    )
                                qk_kv_half_inline2577__phi_v3: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(
                                    qk_kv_half_inline2577__phi_v2
                                )
                            else:
                                qk_kv_half_inline2577__phi_v3: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(
                                    qk_kv_half_inline2577__ssa_v0
                                )
                            qk_lo_inline2585__ssa_v1: pl.Scalar[pl.INDEX] = pl.max(32 - qk_head_inline2534__ssa_v0 - qk_lane_kv_inline2497__ssa_v0, 0)
                            qk_hi_inline2468__ssa_v1: pl.Scalar[pl.INDEX] = pl.min(64 - qk_head_inline2534__ssa_v0 - qk_lane_kv_inline2497__ssa_v0, qk_rows_inline2583__ssa_v0)
                            if qk_lo_inline2585__ssa_v1 < qk_hi_inline2468__ssa_v1:
                                qk_raw_row_inline2483__tile_1: pl.Scalar[pl.INT32] = pl.tensor.read(
                                    window_swa_indices__ssa_v0, [qk_t_inline2491__idx_v0, qk_s0_inline2495__ssa_v0 + qk_lane_kv_inline2497__ssa_v0 + qk_lo_inline2585__ssa_v1]
                                )
                                if 0 <= pl.cast(qk_raw_row_inline2483__tile_1, pl.INDEX):
                                    qk_kv_half_inline2577__ssa_v4: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.tile.gather_row(
                                        qk_kv_half_inline2577__phi_v3,
                                        ori_kv_flat_inline2519__ssa_v1,
                                        [qk_lo_inline2585__ssa_v1, 0],
                                        [qk_raw_row_inline2483__tile_1, 0],
                                        [64, 512],
                                        valid_shape=[qk_hi_inline2468__ssa_v1 - qk_lo_inline2585__ssa_v1, 512],
                                        transpose=False,
                                    )
                                    qk_kv_half_inline2577__phi_v5: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(
                                        qk_kv_half_inline2577__ssa_v4
                                    )
                                else:
                                    qk_kv_half_inline2577__phi_v5: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(
                                        qk_kv_half_inline2577__phi_v3
                                    )
                                qk_kv_half_inline2577__phi_v6: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(
                                    qk_kv_half_inline2577__phi_v5
                                )
                            else:
                                qk_kv_half_inline2577__phi_v6: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(
                                    qk_kv_half_inline2577__phi_v3
                                )
                            qk_lo_inline2585__ssa_v2: pl.Scalar[pl.INDEX] = pl.max(64 - qk_head_inline2534__ssa_v0 - qk_lane_kv_inline2497__ssa_v0, 0)
                            qk_hi_inline2468__ssa_v2: pl.Scalar[pl.INDEX] = pl.min(96 - qk_head_inline2534__ssa_v0 - qk_lane_kv_inline2497__ssa_v0, qk_rows_inline2583__ssa_v0)
                            if qk_lo_inline2585__ssa_v2 < qk_hi_inline2468__ssa_v2:
                                qk_raw_row_inline2483__tile_2: pl.Scalar[pl.INT32] = pl.tensor.read(
                                    window_swa_indices__ssa_v0, [qk_t_inline2491__idx_v0, qk_s0_inline2495__ssa_v0 + qk_lane_kv_inline2497__ssa_v0 + qk_lo_inline2585__ssa_v2]
                                )
                                if 0 <= pl.cast(qk_raw_row_inline2483__tile_2, pl.INDEX):
                                    qk_kv_half_inline2577__ssa_v7: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.tile.gather_row(
                                        qk_kv_half_inline2577__phi_v6,
                                        ori_kv_flat_inline2519__ssa_v1,
                                        [qk_lo_inline2585__ssa_v2, 0],
                                        [qk_raw_row_inline2483__tile_2, 0],
                                        [64, 512],
                                        valid_shape=[qk_hi_inline2468__ssa_v2 - qk_lo_inline2585__ssa_v2, 512],
                                        transpose=False,
                                    )
                                    qk_kv_half_inline2577__phi_v8: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(
                                        qk_kv_half_inline2577__ssa_v7
                                    )
                                else:
                                    qk_kv_half_inline2577__phi_v8: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(
                                        qk_kv_half_inline2577__phi_v6
                                    )
                                qk_kv_half_inline2577__phi_v9: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(
                                    qk_kv_half_inline2577__phi_v8
                                )
                            else:
                                qk_kv_half_inline2577__phi_v9: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(
                                    qk_kv_half_inline2577__phi_v6
                                )
                            qk_lo_inline2585__ssa_v3: pl.Scalar[pl.INDEX] = pl.max(96 - qk_head_inline2534__ssa_v0 - qk_lane_kv_inline2497__ssa_v0, 0)
                            qk_hi_inline2468__ssa_v3: pl.Scalar[pl.INDEX] = pl.min(128 - qk_head_inline2534__ssa_v0 - qk_lane_kv_inline2497__ssa_v0, qk_rows_inline2583__ssa_v0)
                            if qk_lo_inline2585__ssa_v3 < qk_hi_inline2468__ssa_v3:
                                qk_raw_row_inline2483__tile_3: pl.Scalar[pl.INT32] = pl.tensor.read(
                                    window_swa_indices__ssa_v0, [qk_t_inline2491__idx_v0, qk_s0_inline2495__ssa_v0 + qk_lane_kv_inline2497__ssa_v0 + qk_lo_inline2585__ssa_v3]
                                )
                                if 0 <= pl.cast(qk_raw_row_inline2483__tile_3, pl.INDEX):
                                    qk_kv_half_inline2577__ssa_v10: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.tile.gather_row(
                                        qk_kv_half_inline2577__phi_v9,
                                        ori_kv_flat_inline2519__ssa_v1,
                                        [qk_lo_inline2585__ssa_v3, 0],
                                        [qk_raw_row_inline2483__tile_3, 0],
                                        [64, 512],
                                        valid_shape=[qk_hi_inline2468__ssa_v3 - qk_lo_inline2585__ssa_v3, 512],
                                        transpose=False,
                                    )
                                    qk_kv_half_inline2577__phi_v11: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(
                                        qk_kv_half_inline2577__ssa_v10
                                    )
                                else:
                                    qk_kv_half_inline2577__phi_v11: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(
                                        qk_kv_half_inline2577__phi_v9
                                    )
                                qk_kv_half_inline2577__phi_v12: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(
                                    qk_kv_half_inline2577__phi_v11
                                )
                            else:
                                qk_kv_half_inline2577__phi_v12: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(
                                    qk_kv_half_inline2577__phi_v9
                                )
                            qk_lo_inline2585__ssa_v4: pl.Scalar[pl.INDEX] = pl.max(128 - qk_head_inline2534__ssa_v0 - qk_lane_kv_inline2497__ssa_v0, 0)
                            qk_hi_inline2468__ssa_v4: pl.Scalar[pl.INDEX] = pl.min(160 - qk_head_inline2534__ssa_v0 - qk_lane_kv_inline2497__ssa_v0, qk_rows_inline2583__ssa_v0)
                            if qk_lo_inline2585__ssa_v4 < qk_hi_inline2468__ssa_v4:
                                qk_raw_row_inline2483__tile_4: pl.Scalar[pl.INT32] = pl.tensor.read(
                                    window_swa_indices__ssa_v0, [qk_t_inline2491__idx_v0, qk_s0_inline2495__ssa_v0 + qk_lane_kv_inline2497__ssa_v0 + qk_lo_inline2585__ssa_v4]
                                )
                                if 0 <= pl.cast(qk_raw_row_inline2483__tile_4, pl.INDEX):
                                    qk_kv_half_inline2577__ssa_v13: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.tile.gather_row(
                                        qk_kv_half_inline2577__phi_v12,
                                        ori_kv_flat_inline2519__ssa_v1,
                                        [qk_lo_inline2585__ssa_v4, 0],
                                        [qk_raw_row_inline2483__tile_4, 0],
                                        [64, 512],
                                        valid_shape=[qk_hi_inline2468__ssa_v4 - qk_lo_inline2585__ssa_v4, 512],
                                        transpose=False,
                                    )
                                    qk_kv_half_inline2577__phi_v14: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(
                                        qk_kv_half_inline2577__ssa_v13
                                    )
                                else:
                                    qk_kv_half_inline2577__phi_v14: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(
                                        qk_kv_half_inline2577__phi_v12
                                    )
                                qk_kv_half_inline2577__phi_v15: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(
                                    qk_kv_half_inline2577__phi_v14
                                )
                            else:
                                qk_kv_half_inline2577__phi_v15: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(
                                    qk_kv_half_inline2577__phi_v12
                                )
                            qk_kv_half_inline2577__phi_v21: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(qk_kv_half_inline2577__phi_v15)
                        else:
                            qk_kv_half_inline2577__ssa_v0: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.tile.full(
                                [64, 512], dtype=pl.BF16, value=0.0
                            )
                            for qk_row_inline2480__idx_v0, (qk_kv_half_inline2577__iter_v16,) in pl.range(64, init_values=(qk_kv_half_inline2577__ssa_v0,)):
                                qk_cmp_k_inline2587__ssa_v0: pl.Scalar[pl.INDEX] = qk_s0_inline2495__ssa_v0 + qk_lane_kv_inline2497__ssa_v0 + qk_row_inline2480__idx_v0 - 128
                                if qk_cmp_k_inline2587__ssa_v0 < 512:
                                    qk_ridx_inline2589__tile: pl.Scalar[pl.INT32] = pl.tensor.read(cmp_sparse_indices_inline2527__rv_v2, [qk_t_inline2491__idx_v0, qk_cmp_k_inline2587__ssa_v0])
                                    if 0 <= pl.cast(qk_ridx_inline2589__tile, pl.INDEX):
                                        t__tile_2: pl.Scalar[pl.INT32] = pl.tensor.read(cmp_block_table__ssa_v0, [qk_b_inline2460__ssa_v0, pl.cast(qk_ridx_inline2589__tile, pl.INDEX) // 32])
                                        qk_page_inline2488__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(t__tile_2, pl.INDEX)
                                        qk_src_inline2506__ssa_v0: pl.Scalar[pl.INDEX] = qk_page_inline2488__ssa_v0 * 32 + pl.cast(qk_ridx_inline2589__tile, pl.INDEX) % 32
                                        qk_kv_half_inline2577__ssa_v18: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.tile.gather_row(
                                            qk_kv_half_inline2577__iter_v16, cmp_kv_flat_inline2564__ssa_v0, [qk_row_inline2480__idx_v0, 0], [qk_src_inline2506__ssa_v0, 0], [1, 512], transpose=False
                                        )
                                        qk_kv_half_inline2577__phi_v19: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(
                                            qk_kv_half_inline2577__ssa_v18
                                        )
                                    else:
                                        qk_kv_half_inline2577__phi_v19: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(
                                            qk_kv_half_inline2577__iter_v16
                                        )
                                    qk_kv_half_inline2577__phi_v20: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(
                                        qk_kv_half_inline2577__phi_v19
                                    )
                                else:
                                    qk_kv_half_inline2577__phi_v20: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(
                                        qk_kv_half_inline2577__iter_v16
                                    )
                                qk_kv_half_inline2577__rv_v17: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(
                                    qk_kv_half_inline2577__phi_v20
                                )
                            qk_kv_half_inline2577__phi_v21: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 65536), pl.Mem.Vec] = pl.yield_(qk_kv_half_inline2577__rv_v17)
                        kv_transfer_inline2476__store: pl.Tensor[[9216, 512], pl.BF16, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 9437184)] = pl.tile.store(
                            qk_kv_half_inline2577__phi_v21, [qk_kv_row_inline2457__ssa_v1 + qk_lane_kv_inline2497__ssa_v0, 0], kv_transfer_inline2476__ssa_v0
                        )
                if 0 < qk_tick_inline2465__idx_v0 and qk_tick_inline2465__idx_v0 <= 5:
                    softmax_sb_inline2572__ssa_v0: pl.Scalar[pl.INDEX] = qk_tick_inline2465__idx_v0 - 1
                    t__tile_3: pl.Scalar[pl.INT32] = pl.tensor.read(valid_block_mask_inline2517__ssa_v0, [qk_t_inline2491__idx_v0, softmax_sb_inline2572__ssa_v0])
                    if 0 < pl.cast(t__tile_3, pl.INDEX):
                        qk_slot_inline2512__ssa_v2: pl.Scalar[pl.INDEX] = qk_core_inline2560__ssa_v0 * 3 + softmax_sb_inline2572__ssa_v0 % 3
                        qk_transfer_row_inline2514__ssa_v2: pl.Scalar[pl.INDEX] = qk_slot_inline2512__ssa_v2 * 64
                        qk_s0_v1_inline2469__ssa_v0: pl.Scalar[pl.INDEX] = softmax_sb_inline2572__ssa_v0 * 128
                        pl.system.sync_wait(1, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIV)
                        qk_scores_half_inline2467__ssa_v0: pl.Tile[[32, 128], pl.FP32, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.load(
                            score_transfer_inline2473__ssa_v0, [qk_transfer_row_inline2514__ssa_v2 + qk_lane_head_inline2563__ssa_v0, 0], [32, 128], [32, 128], target_memory=pl.Mem.Vec
                        )
                        qk_bias_inline2554__ssa_v0: pl.Tile[[1, 128], pl.FP32, pl.MemRef(mem_vec_60, pl.const(148224, pl.INT64), 512), pl.Mem.Vec] = pl.tile.load(
                            sparse_bias_inline2513__rv_v2, [qk_t_inline2491__idx_v0, qk_s0_v1_inline2469__ssa_v0], [1, 128], [1, 128], target_memory=pl.Mem.Vec
                        )
                        qk_scaled_inline2590__ssa_v0: pl.Tile[[32, 128], pl.FP32, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.muls(
                            qk_scores_half_inline2467__ssa_v0, 0.044194173824159223
                        )
                        qk_masked_inline2522__ssa_v0: pl.Tile[[32, 128], pl.FP32, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.col_expand_add(
                            qk_scaled_inline2590__ssa_v0, qk_bias_inline2554__ssa_v0
                        )
                        qk_mi_inline2588__ssa_v0: pl.Tile[[32, 1], pl.FP32, pl.MemRef(mem_vec_60, pl.const(148224, pl.INT64), 128), pl.Mem.Vec] = pl.tile.row_max(
                            qk_masked_inline2522__ssa_v0, qk_reduce_tmp_inline2548__ssa_v0
                        )
                        t__tmp_v282: pl.Tile[[32, 128], pl.FP32, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.row_expand_sub(
                            qk_masked_inline2522__ssa_v0, qk_mi_inline2588__ssa_v0
                        )
                        qk_exp_inline2593__ssa_v0: pl.Tile[[32, 128], pl.FP32, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.exp(t__tmp_v282)
                        qk_li_inline2485__ssa_v0: pl.Tile[[32, 1], pl.FP32, pl.MemRef(mem_vec_47, pl.const(147712, pl.INT64), 128), pl.Mem.Vec] = pl.tile.row_sum(
                            qk_exp_inline2593__ssa_v0, qk_reduce_tmp_inline2548__ssa_v0
                        )
                        qk_probability_inline2492__ssa_v0: pl.Tile[[32, 128], pl.BF16, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.cast(
                            qk_exp_inline2593__ssa_v0, target_type=pl.BF16, mode="rint"
                        )
                        probability_transfer_inline2461__store: pl.Tensor[[4608, 128], pl.BF16, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 1179648)] = pl.tile.store(
                            qk_probability_inline2492__ssa_v0, [qk_transfer_row_inline2514__ssa_v2 + qk_lane_head_inline2563__ssa_v0, 0], probability_transfer_inline2461__ssa_v0
                        )
                        mi_transfer_inline2470__store: pl.Tensor[[4608, 1], pl.FP32, pl.MemRef("mem_ddr_15", pl.const(0, pl.INT64), 18432)] = pl.tile.store(
                            qk_mi_inline2588__ssa_v0, [qk_transfer_row_inline2514__ssa_v2 + qk_lane_head_inline2563__ssa_v0, 0], mi_transfer_inline2470__ssa_v0
                        )
                        li_transfer_inline2478__store: pl.Tensor[[4608, 1], pl.FP32, pl.MemRef("mem_ddr_16", pl.const(0, pl.INT64), 18432)] = pl.tile.store(
                            qk_li_inline2485__ssa_v0, [qk_transfer_row_inline2514__ssa_v2 + qk_lane_head_inline2563__ssa_v0, 0], li_transfer_inline2478__ssa_v0
                        )
                        pl.system.sync_set(2, pipe=pl.PipeType.MTE3, ffts_mode=2, core_type=pl.KernelType.AIV)
                if qk_tick_inline2465__idx_v0 < 5:
                    t__tile_4: pl.Scalar[pl.INT32] = pl.tensor.read(valid_block_mask_inline2517__ssa_v0, [qk_t_inline2491__idx_v0, qk_tick_inline2465__idx_v0])
                    if 0 < pl.cast(t__tile_4, pl.INDEX):
                        pl.system.sync_set(0, pipe=pl.PipeType.MTE3, ffts_mode=2, core_type=pl.KernelType.AIV)
                if 3 <= qk_tick_inline2465__idx_v0:
                    pv_sb_inline2552__ssa_v1: pl.Scalar[pl.INDEX] = qk_tick_inline2465__idx_v0 - 2 - 1
                    t__tile_5: pl.Scalar[pl.INT32] = pl.tensor.read(valid_block_mask_inline2517__ssa_v0, [qk_t_inline2491__idx_v0, pv_sb_inline2552__ssa_v1])
                    if 0 < pl.cast(t__tile_5, pl.INDEX):
                        pv_slot_inline2592__ssa_v1: pl.Scalar[pl.INDEX] = qk_core_inline2560__ssa_v0 * 3 + pv_sb_inline2552__ssa_v1 % 3
                        pv_transfer_row_inline2556__ssa_v1: pl.Scalar[pl.INDEX] = pv_slot_inline2592__ssa_v1 * 64
                        pl.system.sync_wait(3, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIV)
                        pv_m_inline2464__ssa_v0: pl.Tile[[32, 1], pl.FP32, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 128), pl.Mem.Vec] = pl.tile.load(
                            mi_transfer_inline2470__ssa_v0, [pv_transfer_row_inline2556__ssa_v1 + qk_lane_head_inline2563__ssa_v0, 0], [32, 1], [32, 1], target_memory=pl.Mem.Vec
                        )
                        pv_l_inline2532__ssa_v0: pl.Tile[[32, 1], pl.FP32, pl.MemRef(mem_vec_60, pl.const(148224, pl.INT64), 128), pl.Mem.Vec] = pl.tile.load(
                            li_transfer_inline2478__ssa_v0, [pv_transfer_row_inline2556__ssa_v1 + qk_lane_head_inline2563__ssa_v0, 0], [32, 1], [32, 1], target_memory=pl.Mem.Vec
                        )
                        next_m_inline2543__rm_a0_tmp_v2: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_21, pl.const(16384, pl.INT64), 128), pl.Mem.Vec] = pl.tile.reshape(m_iter, [1, 32])
                        next_m_inline2543__rm_a1_tmp_v3: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 128), pl.Mem.Vec] = pl.tile.reshape(
                            pv_m_inline2464__ssa_v0, [1, 32]
                        )
                        next_m_inline2543__row_major_tmp_v4: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_47, pl.const(147712, pl.INT64), 128), pl.Mem.Vec] = pl.tile.maximum(
                            next_m_inline2543__rm_a0_tmp_v2, next_m_inline2543__rm_a1_tmp_v3
                        )
                        next_m_inline2543__ssa_v0: pl.Tile[[32, 1], pl.FP32, pl.MemRef(mem_vec_47, pl.const(147712, pl.INT64), 128), pl.Mem.Vec] = pl.tile.reshape(
                            next_m_inline2543__row_major_tmp_v4, [32, 1]
                        )
                        t__rm_a0_tmp_v5: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_21, pl.const(16384, pl.INT64), 128), pl.Mem.Vec] = pl.tile.reshape(m_iter, [1, 32])
                        t__rm_a1_tmp_v6: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_47, pl.const(147712, pl.INT64), 128), pl.Mem.Vec] = pl.tile.reshape(next_m_inline2543__ssa_v0, [1, 32])
                        t__row_major_tmp_v7: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_52, pl.const(147840, pl.INT64), 128), pl.Mem.Vec] = pl.tile.sub(t__rm_a0_tmp_v5, t__rm_a1_tmp_v6)
                        t__tmp_v285: pl.Tile[[32, 1], pl.FP32, pl.MemRef(mem_vec_52, pl.const(147840, pl.INT64), 128), pl.Mem.Vec] = pl.tile.reshape(t__row_major_tmp_v7, [32, 1])
                        alpha_inline2566__rm_a0_tmp_v8: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_52, pl.const(147840, pl.INT64), 128), pl.Mem.Vec] = pl.tile.reshape(t__tmp_v285, [1, 32])
                        alpha_inline2566__row_major_tmp_v9: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_52, pl.const(147840, pl.INT64), 128), pl.Mem.Vec] = pl.tile.exp(alpha_inline2566__rm_a0_tmp_v8)
                        alpha_inline2566__ssa_v0: pl.Tile[[32, 1], pl.FP32, pl.MemRef(mem_vec_52, pl.const(147840, pl.INT64), 128), pl.Mem.Vec] = pl.tile.reshape(
                            alpha_inline2566__row_major_tmp_v9, [32, 1]
                        )
                        t__rm_a0_tmp_v10: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 128), pl.Mem.Vec] = pl.tile.reshape(pv_m_inline2464__ssa_v0, [1, 32])
                        t__rm_a1_tmp_v11: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_47, pl.const(147712, pl.INT64), 128), pl.Mem.Vec] = pl.tile.reshape(next_m_inline2543__ssa_v0, [1, 32])
                        t__row_major_tmp_v12: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 128), pl.Mem.Vec] = pl.tile.sub(t__rm_a0_tmp_v10, t__rm_a1_tmp_v11)
                        t__tmp_v286: pl.Tile[[32, 1], pl.FP32, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 128), pl.Mem.Vec] = pl.tile.reshape(t__row_major_tmp_v12, [32, 1])
                        beta_inline2535__rm_a0_tmp_v13: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 128), pl.Mem.Vec] = pl.tile.reshape(t__tmp_v286, [1, 32])
                        beta_inline2535__row_major_tmp_v14: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_55, pl.const(147968, pl.INT64), 128), pl.Mem.Vec] = pl.tile.exp(beta_inline2535__rm_a0_tmp_v13)
                        beta_inline2535__ssa_v0: pl.Tile[[32, 1], pl.FP32, pl.MemRef(mem_vec_55, pl.const(147968, pl.INT64), 128), pl.Mem.Vec] = pl.tile.reshape(
                            beta_inline2535__row_major_tmp_v14, [32, 1]
                        )
                        t__rm_a0_tmp_v15: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_52, pl.const(147840, pl.INT64), 128), pl.Mem.Vec] = pl.tile.reshape(alpha_inline2566__ssa_v0, [1, 32])
                        t__rm_a1_tmp_v16: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_22, pl.const(16512, pl.INT64), 128), pl.Mem.Vec] = pl.tile.reshape(l_iter, [1, 32])
                        t__row_major_tmp_v17: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 128), pl.Mem.Vec] = pl.tile.mul(t__rm_a0_tmp_v15, t__rm_a1_tmp_v16)
                        t__tmp_v287: pl.Tile[[32, 1], pl.FP32, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 128), pl.Mem.Vec] = pl.tile.reshape(t__row_major_tmp_v17, [32, 1])
                        t__rm_a0_tmp_v18: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_55, pl.const(147968, pl.INT64), 128), pl.Mem.Vec] = pl.tile.reshape(beta_inline2535__ssa_v0, [1, 32])
                        t__rm_a1_tmp_v19: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_60, pl.const(148224, pl.INT64), 128), pl.Mem.Vec] = pl.tile.reshape(pv_l_inline2532__ssa_v0, [1, 32])
                        t__row_major_tmp_v20: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_60, pl.const(148224, pl.INT64), 128), pl.Mem.Vec] = pl.tile.mul(t__rm_a0_tmp_v18, t__rm_a1_tmp_v19)
                        t__tmp_v288: pl.Tile[[32, 1], pl.FP32, pl.MemRef(mem_vec_60, pl.const(148224, pl.INT64), 128), pl.Mem.Vec] = pl.tile.reshape(t__row_major_tmp_v20, [32, 1])
                        next_l_inline2596__rm_a0_tmp_v21: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 128), pl.Mem.Vec] = pl.tile.reshape(t__tmp_v287, [1, 32])
                        next_l_inline2596__rm_a1_tmp_v22: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_60, pl.const(148224, pl.INT64), 128), pl.Mem.Vec] = pl.tile.reshape(t__tmp_v288, [1, 32])
                        next_l_inline2596__row_major_tmp_v23: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_58, pl.const(148096, pl.INT64), 128), pl.Mem.Vec] = pl.tile.add(
                            next_l_inline2596__rm_a0_tmp_v21, next_l_inline2596__rm_a1_tmp_v22
                        )
                        next_l_inline2596__ssa_v0: pl.Tile[[32, 1], pl.FP32, pl.MemRef(mem_vec_58, pl.const(148096, pl.INT64), 128), pl.Mem.Vec] = pl.tile.reshape(
                            next_l_inline2596__row_major_tmp_v23, [32, 1]
                        )
                        pv_left_inline2586__ssa_v0: pl.Tile[[32, 256], pl.FP32, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.load(
                            pv_transfer_inline2471__ssa_v0, [pv_transfer_row_inline2556__ssa_v1 + qk_lane_head_inline2563__ssa_v0, 0], [32, 256], [32, 256], target_memory=pl.Mem.Vec
                        )
                        t__tmp_v289: pl.Tile[[32, 256], pl.FP32, pl.MemRef(mem_vec_60, pl.const(148224, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.row_expand_mul(left_iter, alpha_inline2566__ssa_v0)
                        t__tmp_v290: pl.Tile[[32, 256], pl.FP32, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.row_expand_mul(
                            pv_left_inline2586__ssa_v0, beta_inline2535__ssa_v0
                        )
                        next_left_inline2459__ssa_v0: pl.Tile[[32, 256], pl.FP32, pl.MemRef(mem_vec_23, pl.const(16640, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.add(t__tmp_v289, t__tmp_v290)
                        pv_right_inline2540__ssa_v0: pl.Tile[[32, 256], pl.FP32, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.load(
                            pv_transfer_inline2471__ssa_v0, [pv_transfer_row_inline2556__ssa_v1 + qk_lane_head_inline2563__ssa_v0, 256], [32, 256], [32, 256], target_memory=pl.Mem.Vec
                        )
                        t__tmp_v291: pl.Tile[[32, 256], pl.FP32, pl.MemRef(mem_vec_60, pl.const(148224, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.row_expand_mul(right_iter, alpha_inline2566__ssa_v0)
                        t__tmp_v292: pl.Tile[[32, 256], pl.FP32, pl.MemRef(mem_vec_25, pl.const(82176, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.row_expand_mul(
                            pv_right_inline2540__ssa_v0, beta_inline2535__ssa_v0
                        )
                        next_right_inline2597__ssa_v0: pl.Tile[[32, 256], pl.FP32, pl.MemRef(mem_vec_24, pl.const(49408, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.add(t__tmp_v291, t__tmp_v292)
                        m_valid_inline2594__rv_v0, l_valid_inline2502__rv_v0, left_valid_inline2555__rv_v0, right_valid_inline2595__rv_v0 = pl.yield_(
                            next_m_inline2543__ssa_v0, next_l_inline2596__ssa_v0, next_left_inline2459__ssa_v0, next_right_inline2597__ssa_v0
                        )
                    else:
                        m_iter_mv: pl.Tile[[32, 1], pl.FP32, pl.MemRef(mem_vec_47, pl.const(147712, pl.INT64), 128), pl.Mem.Vec] = pl.tile.move(m_iter, target_memory=pl.Mem.Vec)
                        l_iter_mv: pl.Tile[[32, 1], pl.FP32, pl.MemRef(mem_vec_58, pl.const(148096, pl.INT64), 128), pl.Mem.Vec] = pl.tile.move(l_iter, target_memory=pl.Mem.Vec)
                        m_valid_inline2594__rv_v0, l_valid_inline2502__rv_v0, left_valid_inline2555__rv_v0, right_valid_inline2595__rv_v0 = pl.yield_(m_iter_mv, l_iter_mv, left_iter, right_iter)
                    m_after_inline2579__rv_v0, l_after_inline2525__rv_v0, left_after_inline2503__rv_v0, right_after_inline2561__rv_v0 = pl.yield_(
                        m_valid_inline2594__rv_v0, l_valid_inline2502__rv_v0, left_valid_inline2555__rv_v0, right_valid_inline2595__rv_v0
                    )
                else:
                    m_iter_mv_1: pl.Tile[[32, 1], pl.FP32, pl.MemRef(mem_vec_47, pl.const(147712, pl.INT64), 128), pl.Mem.Vec] = pl.tile.move(m_iter, target_memory=pl.Mem.Vec)
                    l_iter_mv_1: pl.Tile[[32, 1], pl.FP32, pl.MemRef(mem_vec_58, pl.const(148096, pl.INT64), 128), pl.Mem.Vec] = pl.tile.move(l_iter, target_memory=pl.Mem.Vec)
                    m_after_inline2579__rv_v0, l_after_inline2525__rv_v0, left_after_inline2503__rv_v0, right_after_inline2561__rv_v0 = pl.yield_(m_iter_mv_1, l_iter_mv_1, left_iter, right_iter)
                l_after_inline2525__rv_v0_mv: pl.Tile[[32, 1], pl.FP32, pl.MemRef(mem_vec_22, pl.const(16512, pl.INT64), 128), pl.Mem.Vec] = pl.tile.move(
                    l_after_inline2525__rv_v0, target_memory=pl.Mem.Vec
                )
                m_after_inline2579__rv_v0_mv: pl.Tile[[32, 1], pl.FP32, pl.MemRef(mem_vec_21, pl.const(16384, pl.INT64), 128), pl.Mem.Vec] = pl.tile.move(
                    m_after_inline2579__rv_v0, target_memory=pl.Mem.Vec
                )
                running_m_inline2570, running_l_inline2573, running_left_inline2575, running_right_inline2518 = pl.yield_(
                    m_after_inline2579__rv_v0_mv, l_after_inline2525__rv_v0_mv, left_after_inline2503__rv_v0, right_after_inline2561__rv_v0
                )
            qk_output_row_inline2520__ssa_v0: pl.Scalar[pl.INDEX] = qk_t_inline2491__idx_v0 * 64 + qk_lane_head_inline2563__ssa_v0
            attn_mi_inline2536__store: pl.Tensor[[t_heads_inline2521__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_17", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                running_m_inline2570, [qk_output_row_inline2520__ssa_v0, 0], attn_mi_inline2536__ssa_v0
            )
            attn_li_inline2559__store: pl.Tensor[[t_heads_inline2521__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_18", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                running_l_inline2573, [qk_output_row_inline2520__ssa_v0, 0], attn_li_inline2559__ssa_v0
            )
            attn_oi_inline2479__store: pl.Tensor[[t_heads_inline2521__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_19", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                running_left_inline2575, [qk_output_row_inline2520__ssa_v0, 0], attn_oi_inline2479__ssa_v0
            )
            attn_oi_inline2479__store_v0: pl.Tensor[[t_heads_inline2521__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_19", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                running_right_inline2518, [qk_output_row_inline2520__ssa_v0, 256], attn_oi_inline2479__ssa_v0
            )
        return (
            kv_transfer_inline2476__ssa_v0,
            score_transfer_inline2473__ssa_v0,
            probability_transfer_inline2461__ssa_v0,
            pv_transfer_inline2471__ssa_v0,
            mi_transfer_inline2470__ssa_v0,
            li_transfer_inline2478__ssa_v0,
            attn_mi_inline2536__ssa_v0,
            attn_li_inline2559__ssa_v0,
            attn_oi_inline2479__ssa_v0,
        )

    @pl.function(type=pl.FunctionType.Group, level=pl.Level.CORE_GROUP, role=pl.Role.SubWorker)
    def qk_pv(
        ffts_workspace_inline2458__ssa_v0: pl.Tensor[[256], pl.INT64, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 2048)],
        t_dim_inline2515__ssa_v0: pl.Scalar[pl.INDEX],
        q_flat_inline2539__ssa_v0: pl.Tensor[[t_heads_inline2521__ssa_v0, 512], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)],
        valid_block_mask_inline2517__ssa_v0: pl.Tensor[[t_dim_inline2515__ssa_v0, 16], pl.INT32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)],
        kv_transfer_inline2476__ssa_v0: pl.InOut[pl.Tensor[[9216, 512], pl.BF16, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 9437184)]],
        score_transfer_inline2473__ssa_v0: pl.InOut[pl.Tensor[[4608, 128], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 2359296)]],
        probability_transfer_inline2461__ssa_v0: pl.InOut[pl.Tensor[[4608, 128], pl.BF16, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 1179648)]],
        pv_transfer_inline2471__ssa_v0: pl.InOut[pl.Tensor[[4608, 512], pl.FP32, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 9437184)]],
        attn_sink_col_inline2487__ssa_v0: pl.Tensor[[64, 1], pl.FP32, pl.MemRef("mem_ddr_7", pl.const(0, pl.INT64), 256)],
        position_ids_t1__ssa_v0: pl.Tensor[[T_DYN, 1], pl.INT32, pl.MemRef("mem_ddr_8", pl.const(0, pl.INT64), 0)],
        window_swa_indices__ssa_v0: pl.Tensor[[T_DYN, 128], pl.INT32, pl.MemRef("mem_ddr_9", pl.const(0, pl.INT64), 0)],
        ori_kv_flat_inline2519__ssa_v1: pl.Tensor[[ori_block_num_inline2509__ssa_v0 * pl.const(32, pl.INDEX), 512], pl.BF16, pl.MemRef("mem_ddr_10", pl.const(0, pl.INT64), 0)],
        cmp_sparse_indices_inline2527__rv_v2: pl.Tensor[[t_dim_inline2515__ssa_v0, 512], pl.INT32, pl.MemRef("mem_ddr_11", pl.const(0, pl.INT64), 0)],
        cmp_block_table__ssa_v0: pl.Tensor[[B_DYN, COMPRESSED_TABLE_COLUMNS_DYN], pl.INT32, pl.MemRef("mem_ddr_12", pl.const(0, pl.INT64), 0)],
        cmp_kv_flat_inline2564__ssa_v0: pl.Tensor[[cmp_block_num_inline2493__ssa_v0 * pl.const(32, pl.INDEX), 512], pl.BF16, pl.MemRef("mem_ddr_13", pl.const(0, pl.INT64), 0)],
        sparse_bias_inline2513__rv_v2: pl.Tensor[[t_dim_inline2515__ssa_v0, 640], pl.FP32, pl.MemRef("mem_ddr_14", pl.const(0, pl.INT64), 0)],
        mi_transfer_inline2470__ssa_v0: pl.InOut[pl.Tensor[[4608, 1], pl.FP32, pl.MemRef("mem_ddr_15", pl.const(0, pl.INT64), 18432)]],
        li_transfer_inline2478__ssa_v0: pl.InOut[pl.Tensor[[4608, 1], pl.FP32, pl.MemRef("mem_ddr_16", pl.const(0, pl.INT64), 18432)]],
        attn_mi_inline2536__ssa_v0: pl.Out[pl.Tensor[[t_heads_inline2521__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_17", pl.const(0, pl.INT64), 0)]],
        attn_li_inline2559__ssa_v0: pl.Out[pl.Tensor[[t_heads_inline2521__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_18", pl.const(0, pl.INT64), 0)]],
        attn_oi_inline2479__ssa_v0: pl.Out[pl.Tensor[[t_heads_inline2521__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_19", pl.const(0, pl.INT64), 0)]],
    ) -> tuple[
        pl.Tensor[[9216, 512], pl.BF16],
        pl.Tensor[[4608, 128], pl.FP32],
        pl.Tensor[[4608, 128], pl.BF16],
        pl.Tensor[[4608, 512], pl.FP32],
        pl.Tensor[[4608, 1], pl.FP32],
        pl.Tensor[[4608, 1], pl.FP32],
        pl.Tensor[[t_heads_inline2521__ssa_v0, 1], pl.FP32],
        pl.Tensor[[t_heads_inline2521__ssa_v0, 1], pl.FP32],
        pl.Tensor[[t_heads_inline2521__ssa_v0, 512], pl.FP32],
    ]:
        pl.func_attr({"mx_tensor_views_blocked": True, "split_aiv": True})
        self.qk_pv_aic(
            ffts_workspace_inline2458__ssa_v0,
            t_dim_inline2515__ssa_v0,
            q_flat_inline2539__ssa_v0,
            valid_block_mask_inline2517__ssa_v0,
            kv_transfer_inline2476__ssa_v0,
            score_transfer_inline2473__ssa_v0,
            probability_transfer_inline2461__ssa_v0,
            pv_transfer_inline2471__ssa_v0,
            attn_sink_col_inline2487__ssa_v0,
            position_ids_t1__ssa_v0,
            window_swa_indices__ssa_v0,
            ori_kv_flat_inline2519__ssa_v1,
            cmp_sparse_indices_inline2527__rv_v2,
            cmp_block_table__ssa_v0,
            cmp_kv_flat_inline2564__ssa_v0,
            sparse_bias_inline2513__rv_v2,
            mi_transfer_inline2470__ssa_v0,
            li_transfer_inline2478__ssa_v0,
            attn_mi_inline2536__ssa_v0,
            attn_li_inline2559__ssa_v0,
            attn_oi_inline2479__ssa_v0,
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
            ffts_workspace_inline2458__ssa_v0,
            t_dim_inline2515__ssa_v0,
            q_flat_inline2539__ssa_v0,
            valid_block_mask_inline2517__ssa_v0,
            kv_transfer_inline2476__ssa_v0,
            score_transfer_inline2473__ssa_v0,
            probability_transfer_inline2461__ssa_v0,
            pv_transfer_inline2471__ssa_v0,
            attn_sink_col_inline2487__ssa_v0,
            position_ids_t1__ssa_v0,
            window_swa_indices__ssa_v0,
            ori_kv_flat_inline2519__ssa_v1,
            cmp_sparse_indices_inline2527__rv_v2,
            cmp_block_table__ssa_v0,
            cmp_kv_flat_inline2564__ssa_v0,
            sparse_bias_inline2513__rv_v2,
            mi_transfer_inline2470__ssa_v0,
            li_transfer_inline2478__ssa_v0,
            attn_mi_inline2536__ssa_v0,
            attn_li_inline2559__ssa_v0,
            attn_oi_inline2479__ssa_v0,
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
            kv_transfer_inline2476__ssa_v0,
            score_transfer_inline2473__ssa_v0,
            probability_transfer_inline2461__ssa_v0,
            pv_transfer_inline2471__ssa_v0,
            mi_transfer_inline2470__ssa_v0,
            li_transfer_inline2478__ssa_v0,
            attn_mi_inline2536__ssa_v0,
            attn_li_inline2559__ssa_v0,
            attn_oi_inline2479__ssa_v0,
        )

    @pl.function(type=pl.FunctionType.Spmd)
    def qk_pv_spmd(
        self,
        ffts_workspace_inline2458__ssa_v0: pl.Tensor[[256], pl.INT64, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 2048)],
        t_dim_inline2515__ssa_v0: pl.Scalar[pl.INDEX],
        q_flat_inline2539__ssa_v0: pl.Tensor[[t_heads_inline2521__ssa_v0, 512], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)],
        valid_block_mask_inline2517__ssa_v0: pl.Tensor[[t_dim_inline2515__ssa_v0, 16], pl.INT32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)],
        kv_transfer_inline2476__ssa_v0: pl.InOut[pl.Tensor[[9216, 512], pl.BF16, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 9437184)]],
        score_transfer_inline2473__ssa_v0: pl.InOut[pl.Tensor[[4608, 128], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 2359296)]],
        probability_transfer_inline2461__ssa_v0: pl.InOut[pl.Tensor[[4608, 128], pl.BF16, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 1179648)]],
        pv_transfer_inline2471__ssa_v0: pl.InOut[pl.Tensor[[4608, 512], pl.FP32, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 9437184)]],
        attn_sink_col_inline2487__ssa_v0: pl.Tensor[[64, 1], pl.FP32, pl.MemRef("mem_ddr_7", pl.const(0, pl.INT64), 256)],
        position_ids_t1__ssa_v0: pl.Tensor[[T_DYN, 1], pl.INT32, pl.MemRef("mem_ddr_8", pl.const(0, pl.INT64), 0)],
        window_swa_indices__ssa_v0: pl.Tensor[[T_DYN, 128], pl.INT32, pl.MemRef("mem_ddr_9", pl.const(0, pl.INT64), 0)],
        ori_kv_flat_inline2519__ssa_v1: pl.Tensor[[ori_block_num_inline2509__ssa_v0 * pl.const(32, pl.INDEX), 512], pl.BF16, pl.MemRef("mem_ddr_10", pl.const(0, pl.INT64), 0)],
        cmp_sparse_indices_inline2527__rv_v2: pl.Tensor[[t_dim_inline2515__ssa_v0, 512], pl.INT32, pl.MemRef("mem_ddr_11", pl.const(0, pl.INT64), 0)],
        cmp_block_table__ssa_v0: pl.Tensor[[B_DYN, COMPRESSED_TABLE_COLUMNS_DYN], pl.INT32, pl.MemRef("mem_ddr_12", pl.const(0, pl.INT64), 0)],
        cmp_kv_flat_inline2564__ssa_v0: pl.Tensor[[cmp_block_num_inline2493__ssa_v0 * pl.const(32, pl.INDEX), 512], pl.BF16, pl.MemRef("mem_ddr_13", pl.const(0, pl.INT64), 0)],
        sparse_bias_inline2513__rv_v2: pl.Tensor[[t_dim_inline2515__ssa_v0, 640], pl.FP32, pl.MemRef("mem_ddr_14", pl.const(0, pl.INT64), 0)],
        mi_transfer_inline2470__ssa_v0: pl.InOut[pl.Tensor[[4608, 1], pl.FP32, pl.MemRef("mem_ddr_15", pl.const(0, pl.INT64), 18432)]],
        li_transfer_inline2478__ssa_v0: pl.InOut[pl.Tensor[[4608, 1], pl.FP32, pl.MemRef("mem_ddr_16", pl.const(0, pl.INT64), 18432)]],
        attn_mi_inline2536__ssa_v0: pl.Out[pl.Tensor[[t_heads_inline2521__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_17", pl.const(0, pl.INT64), 0)]],
        attn_li_inline2559__ssa_v0: pl.Out[pl.Tensor[[t_heads_inline2521__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_18", pl.const(0, pl.INT64), 0)]],
        attn_oi_inline2479__ssa_v0: pl.Out[pl.Tensor[[t_heads_inline2521__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_19", pl.const(0, pl.INT64), 0)]],
    ) -> tuple[pl.Tensor[[t_heads_inline2521__ssa_v0, 1], pl.FP32], pl.Tensor[[t_heads_inline2521__ssa_v0, 1], pl.FP32], pl.Tensor[[t_heads_inline2521__ssa_v0, 512], pl.FP32]]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        ret__tmp_v0: pl.Tuple[
            pl.Tensor[[9216, 512], pl.BF16],
            pl.Tensor[[4608, 128], pl.FP32],
            pl.Tensor[[4608, 128], pl.BF16],
            pl.Tensor[[4608, 512], pl.FP32],
            pl.Tensor[[4608, 1], pl.FP32],
            pl.Tensor[[4608, 1], pl.FP32],
            pl.Tensor[[t_heads_inline2521__ssa_v0, 1], pl.FP32],
            pl.Tensor[[t_heads_inline2521__ssa_v0, 1], pl.FP32],
            pl.Tensor[[t_heads_inline2521__ssa_v0, 512], pl.FP32],
        ] = self.qk_pv(
            ffts_workspace_inline2458__ssa_v0,
            t_dim_inline2515__ssa_v0,
            q_flat_inline2539__ssa_v0,
            valid_block_mask_inline2517__ssa_v0,
            kv_transfer_inline2476__ssa_v0,
            score_transfer_inline2473__ssa_v0,
            probability_transfer_inline2461__ssa_v0,
            pv_transfer_inline2471__ssa_v0,
            attn_sink_col_inline2487__ssa_v0,
            position_ids_t1__ssa_v0,
            window_swa_indices__ssa_v0,
            ori_kv_flat_inline2519__ssa_v1,
            cmp_sparse_indices_inline2527__rv_v2,
            cmp_block_table__ssa_v0,
            cmp_kv_flat_inline2564__ssa_v0,
            sparse_bias_inline2513__rv_v2,
            mi_transfer_inline2470__ssa_v0,
            li_transfer_inline2478__ssa_v0,
            attn_mi_inline2536__ssa_v0,
            attn_li_inline2559__ssa_v0,
            attn_oi_inline2479__ssa_v0,
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
        kv_transfer_inline2476__ssa_v1: pl.Tensor[[9216, 512], pl.BF16, pl.MemRef("mem_ddr_20", pl.const(0, pl.INT64), 9437184)] = ret__tmp_v0[0]
        score_transfer_inline2473__ssa_v1: pl.Tensor[[4608, 128], pl.FP32, pl.MemRef("mem_ddr_21", pl.const(0, pl.INT64), 2359296)] = ret__tmp_v0[1]
        probability_transfer_inline2461__ssa_v1: pl.Tensor[[4608, 128], pl.BF16, pl.MemRef("mem_ddr_22", pl.const(0, pl.INT64), 1179648)] = ret__tmp_v0[2]
        pv_transfer_inline2471__ssa_v1: pl.Tensor[[4608, 512], pl.FP32, pl.MemRef("mem_ddr_23", pl.const(0, pl.INT64), 9437184)] = ret__tmp_v0[3]
        mi_transfer_inline2470__ssa_v1: pl.Tensor[[4608, 1], pl.FP32, pl.MemRef("mem_ddr_24", pl.const(0, pl.INT64), 18432)] = ret__tmp_v0[4]
        li_transfer_inline2478__ssa_v1: pl.Tensor[[4608, 1], pl.FP32, pl.MemRef("mem_ddr_25", pl.const(0, pl.INT64), 18432)] = ret__tmp_v0[5]
        attn_mi_inline2536__ssa_v1: pl.Tensor[[t_heads_inline2521__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_26", pl.const(0, pl.INT64), 0)] = ret__tmp_v0[6]
        attn_li_inline2559__ssa_v1: pl.Tensor[[t_heads_inline2521__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_27", pl.const(0, pl.INT64), 0)] = ret__tmp_v0[7]
        attn_oi_inline2479__ssa_v1: pl.Tensor[[t_heads_inline2521__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_28", pl.const(0, pl.INT64), 0)] = ret__tmp_v0[8]
        return attn_mi_inline2536__ssa_v0, attn_li_inline2559__ssa_v0, attn_oi_inline2479__ssa_v0

    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def qproj_dequant_rms_nope_rope(
        q_flat_inline2971__ssa_v0: pl.Out[pl.Tensor[[t_dim_inline3002__ssa_v0, 32768], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)]],
        tile_rows_inline311_inline1886__ssa_v0: pl.Scalar[pl.INDEX],
        tile_base_inline314_inline1882__idx_v0: pl.Scalar[pl.INDEX],
        qr_scale_pad_store_inline1879__rv_v2: pl.Tensor[[512, 1], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 2048)],
        q_rope_cos_il_inline545__ssa_v0: pl.Tensor[[t_dim_inline546__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)],
        q_rope_sin_signed_inline544__ssa_v0: pl.Tensor[[t_dim_inline546__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)],
        q_rope_swap_idx_inline543__ssa_v0: pl.Tensor[[t_dim_inline546__ssa_v0, 64], pl.INT32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 0)],
        q_proj_i32_inline310_inline1876__ssa_v9: pl.Tensor[[qproj_t_matmul_inline313_inline1884__ssa_v0, 32768], pl.INT32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 0)],
        wq_b_scale__ssa_v0: pl.Tensor[[32768], pl.FP32, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 131072)],
    ):
        pl.func_attr({"mx_tensor_views_blocked": True})
        mem_vec_7: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32)
        mem_vec_8: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 2048)
        mem_vec_9: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 2048)
        mem_vec_10: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 2048)
        mem_vec_11: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
        mem_vec_13: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
        mem_vec_20: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
        mem_vec_21: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
        mem_vec_33: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        mem_vec_34: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        mem_vec_45: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
        mem_vec_46: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
        mem_vec_58: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        mem_vec_59: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        dq_worker_inline2980__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        for dq_work_inline2984__idx_v0, (q_flat_inline2971__iter_v1,) in pl.range(
            dq_worker_inline2980__ssa_v0, (tile_rows_inline311_inline1886__ssa_v0 + 7) // 8 * 16, 48, init_values=(q_flat_inline2971__ssa_v0,)
        ):
            hg_inline3016__ssa_v0: pl.Scalar[pl.INDEX] = dq_work_inline2984__idx_v0 % 16 * 4
            tg_inline2983__ssa_v0: pl.Scalar[pl.INDEX] = dq_work_inline2984__idx_v0 // 16 * 8
            out_tg_inline2978__ssa_v0: pl.Scalar[pl.INDEX] = tile_base_inline314_inline1882__idx_v0 + tg_inline2983__ssa_v0
            if tg_inline2983__ssa_v0 + 8 <= tile_rows_inline311_inline1886__ssa_v0:
                qr_scale_dq_t_inline2982__tile: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_7, pl.const(101376, pl.INT64), 32), pl.Mem.Vec] = pl.tile.load(
                    qr_scale_pad_store_inline1879__rv_v2, [tg_inline2983__ssa_v0, 0], [8, 1], [8, 1], target_memory=pl.Mem.Vec
                )
                q_cos_il_inline2974__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_8, pl.const(101408, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.load(
                    q_rope_cos_il_inline545__ssa_v0, [out_tg_inline2978__ssa_v0, 0], [8, 64], [8, 64], target_memory=pl.Mem.Vec
                )
                q_sin_signed_inline2991__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_9, pl.const(103456, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.load(
                    q_rope_sin_signed_inline544__ssa_v0, [out_tg_inline2978__ssa_v0, 0], [8, 64], [8, 64], target_memory=pl.Mem.Vec
                )
                q_swap_idx_inline2972__tile: pl.Tile[[8, 64], pl.INT32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.load(
                    q_rope_swap_idx_inline543__ssa_v0, [out_tg_inline2978__ssa_v0, 0], [8, 64], [8, 64], target_memory=pl.Mem.Vec
                )
                for h_inner_inline2956__idx_v0, (q_flat_inline2971__iter_v3,) in pl.range(0, 4, 2, init_values=(q_flat_inline2971__iter_v1,)):
                    h_inline2989__ssa_v0: pl.Scalar[pl.INDEX] = hg_inline3016__ssa_v0 + h_inner_inline2956__idx_v0
                    h0_inline2970__ssa_v0: pl.Scalar[pl.INDEX] = h_inline2989__ssa_v0 * 512
                    h_inline2989__ssa_v0_1: pl.Scalar[pl.INDEX] = hg_inline3016__ssa_v0 + (h_inner_inline2956__idx_v0 + 1)
                    h0_inline2970__ssa_v0_1: pl.Scalar[pl.INDEX] = h_inline2989__ssa_v0_1 * 512
                    q_head_acc_inline2969__tile: pl.Tile[[8, 512], pl.INT32, pl.MemRef(mem_vec_11, pl.const(2048, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.load(
                        q_proj_i32_inline310_inline1876__ssa_v9, [tg_inline2983__ssa_v0, h0_inline2970__ssa_v0], [8, 512], [8, 512], target_memory=pl.Mem.Vec
                    )
                    t__tile: pl.Tile[[512], pl.FP32, pl.MemRef(mem_vec_20, pl.const(34816, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.load(
                        wq_b_scale__ssa_v0, [h0_inline2970__ssa_v0], [512], [512], target_memory=pl.Mem.Vec
                    )
                    q_head_acc_inline2969__tile_1: pl.Tile[[8, 512], pl.INT32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.load(
                        q_proj_i32_inline310_inline1876__ssa_v9, [tg_inline2983__ssa_v0, h0_inline2970__ssa_v0_1], [8, 512], [8, 512], target_memory=pl.Mem.Vec
                    )
                    t__tile_1: pl.Tile[[512], pl.FP32, pl.MemRef(mem_vec_45, pl.const(68096, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.load(
                        wq_b_scale__ssa_v0, [h0_inline2970__ssa_v0_1], [512], [512], target_memory=pl.Mem.Vec
                    )
                    q_head_scale_inline2981__tile: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_20, pl.const(34816, pl.INT64), 2048), pl.Mem.Vec] = t__tile
                    q_head_acc_fp32_inline2967__tile: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_11, pl.const(2048, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.cast(
                        q_head_acc_inline2969__tile, target_type=pl.FP32, mode="none"
                    )
                    q_head_row_scaled_inline2965__tile: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_11, pl.const(2048, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.row_expand_mul(
                        q_head_acc_fp32_inline2967__tile, qr_scale_dq_t_inline2982__tile
                    )
                    q_head_dq_inline2962__tile: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_11, pl.const(2048, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.col_expand_mul(
                        q_head_row_scaled_inline2965__tile, q_head_scale_inline2981__tile
                    )
                    t__tile_2: pl.Tile[[8, 512], pl.BF16, pl.MemRef(mem_vec_20, pl.const(34816, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.cast(
                        q_head_dq_inline2962__tile, target_type=pl.BF16, mode="rint"
                    )
                    q_head_dq_v1_inline2998__tile: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_11, pl.const(2048, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.cast(
                        t__tile_2, target_type=pl.FP32, mode="round"
                    )
                    q_head_sq_inline2977__tile: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_20, pl.const(34816, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.mul(
                        q_head_dq_v1_inline2998__tile, q_head_dq_v1_inline2998__tile
                    )
                    tmp_tile: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_21, pl.const(51200, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.create([8, 512], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                    q_head_sq_row_inline2961__tile: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_33, pl.const(67584, pl.INT64), 32), pl.Mem.Vec] = pl.tile.row_sum(q_head_sq_inline2977__tile, tmp_tile)
                    q_head_sq_sum_inline2960__tile: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_33, pl.const(67584, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(q_head_sq_row_inline2961__tile, [1, 8])
                    q_head_sq_mean_inline2955__tile: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_20, pl.const(34816, pl.INT64), 32), pl.Mem.Vec] = pl.tile.muls(
                        q_head_sq_sum_inline2960__tile, 0.001953125
                    )
                    q_head_var_inline2959__tile: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_20, pl.const(34816, pl.INT64), 32), pl.Mem.Vec] = pl.tile.adds(
                        q_head_sq_mean_inline2955__tile, 9.9999999999999995e-07
                    )
                    rsqrt_tmp: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_21, pl.const(51200, pl.INT64), 32), pl.Mem.Vec] = pl.tile.create([1, 8], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                    q_head_inv_rms_inline2988__tile: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_33, pl.const(67584, pl.INT64), 32), pl.Mem.Vec] = pl.tile.rsqrt(q_head_var_inline2959__tile, rsqrt_tmp)
                    q_head_inv_rms_t_inline3013__tile: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_33, pl.const(67584, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(
                        q_head_inv_rms_inline2988__tile, [8, 1]
                    )
                    t__tile_3: pl.Tile[[8, 448], pl.FP32, pl.MemRef(mem_vec_11, pl.const(2048, pl.INT64), 16128), pl.Mem.Vec] = pl.tile.slice(q_head_dq_v1_inline2998__tile, [8, 448], [0, 0])
                    q_nope_normed_inline2994__tile: pl.Tile[[8, 448], pl.FP32, pl.MemRef(mem_vec_20, pl.const(34816, pl.INT64), 14336), pl.Mem.Vec] = pl.tile.row_expand_mul(
                        t__tile_3, q_head_inv_rms_t_inline3013__tile
                    )
                    q_nope_bf16_inline2968__tile: pl.Tile[[8, 448], pl.BF16, pl.MemRef(mem_vec_20, pl.const(34816, pl.INT64), 7168), pl.Mem.Vec] = pl.tile.cast(
                        q_nope_normed_inline2994__tile, target_type=pl.BF16, mode="rint"
                    )
                    q_rope_chunk_raw_inline2992__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_11, pl.const(3840, pl.INT64), 14592), pl.Mem.Vec] = pl.tile.slice(
                        q_head_dq_v1_inline2998__tile, [8, 64], [0, 448]
                    )
                    q_rope_chunk_inline2985__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_11, pl.const(2048, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.row_expand_mul(
                        q_rope_chunk_raw_inline2992__tile, q_head_inv_rms_t_inline3013__tile
                    )
                    t__tile_4: pl.Tile[[8, 64], pl.BF16, pl.MemRef(mem_vec_21, pl.const(51200, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.cast(
                        q_rope_chunk_inline2985__tile, target_type=pl.BF16, mode="rint"
                    )
                    q_rope_chunk_v1_inline2993__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_11, pl.const(2048, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(
                        t__tile_4, target_type=pl.FP32, mode="round"
                    )
                    gather_acc_init: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_21, pl.const(51200, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.create([8, 64], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                    for gather_lv, (gather_ia,) in pl.range(8, init_values=(gather_acc_init,)):
                        gather_inp_row: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_11, pl.const(2048, pl.INT64), 256), pl.Mem.Vec] = pl.tile.slice(
                            q_rope_chunk_v1_inline2993__tile, [1, 64], [gather_lv, 0], [1, 64]
                        )
                        gather_idx_row: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 256), pl.Mem.Vec] = pl.tile.slice(
                            q_swap_idx_inline2972__tile, [1, 64], [gather_lv, 0], [1, 64]
                        )
                        gather_row_tmp: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_33, pl.const(67584, pl.INT64), 256), pl.Mem.Vec] = pl.tile.create(
                            [1, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec
                        )
                        gather_row: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_34, pl.const(67840, pl.INT64), 256), pl.Mem.Vec] = pl.tile.gather(gather_inp_row, gather_idx_row, gather_row_tmp)
                        gather_asmbl: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_21, pl.const(51200, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.assemble(gather_ia, gather_row, [gather_lv, 0])
                        gather_rv: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_21, pl.const(51200, pl.INT64), 2048), pl.Mem.Vec] = pl.yield_(gather_asmbl)
                    q_rope_swapped_inline3005__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_21, pl.const(51200, pl.INT64), 2048), pl.Mem.Vec] = gather_rv
                    q_rope_base_inline2995__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_11, pl.const(2048, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.mul(
                        q_rope_chunk_v1_inline2993__tile, q_cos_il_inline2974__tile
                    )
                    q_rope_delta_inline2996__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_21, pl.const(51200, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.mul(
                        q_rope_swapped_inline3005__tile, q_sin_signed_inline2991__tile
                    )
                    q_rope_rot_inline3006__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_11, pl.const(2048, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.add(
                        q_rope_base_inline2995__tile, q_rope_delta_inline2996__tile
                    )
                    q_rope_bf16_inline2979__tile: pl.Tile[[8, 64], pl.BF16, pl.MemRef(mem_vec_11, pl.const(2048, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.cast(
                        q_rope_rot_inline3006__tile, target_type=pl.BF16, mode="rint"
                    )
                    q_flat_inline2971__tile: pl.Tensor[[t_dim_inline3002__ssa_v0, 32768], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                        q_nope_bf16_inline2968__tile, [out_tg_inline2978__ssa_v0, h0_inline2970__ssa_v0], q_flat_inline2971__iter_v3
                    )
                    q_flat_inline2971__tile_1: pl.Tensor[[t_dim_inline3002__ssa_v0, 32768], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                        q_rope_bf16_inline2979__tile, [out_tg_inline2978__ssa_v0, h0_inline2970__ssa_v0 + 448], q_flat_inline2971__tile
                    )
                    q_head_scale_inline2981__tile_1: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_45, pl.const(68096, pl.INT64), 2048), pl.Mem.Vec] = t__tile_1
                    q_head_acc_fp32_inline2967__tile_1: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.cast(
                        q_head_acc_inline2969__tile_1, target_type=pl.FP32, mode="none"
                    )
                    q_head_row_scaled_inline2965__tile_1: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.row_expand_mul(
                        q_head_acc_fp32_inline2967__tile_1, qr_scale_dq_t_inline2982__tile
                    )
                    q_head_dq_inline2962__tile_1: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.col_expand_mul(
                        q_head_row_scaled_inline2965__tile_1, q_head_scale_inline2981__tile_1
                    )
                    t__tile_5: pl.Tile[[8, 512], pl.BF16, pl.MemRef(mem_vec_45, pl.const(68096, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.cast(
                        q_head_dq_inline2962__tile_1, target_type=pl.BF16, mode="rint"
                    )
                    q_head_dq_v1_inline2998__tile_1: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.cast(
                        t__tile_5, target_type=pl.FP32, mode="round"
                    )
                    q_head_sq_inline2977__tile_1: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_45, pl.const(68096, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.mul(
                        q_head_dq_v1_inline2998__tile_1, q_head_dq_v1_inline2998__tile_1
                    )
                    tmp_tile_1: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_46, pl.const(84480, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.create([8, 512], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                    q_head_sq_row_inline2961__tile_1: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_58, pl.const(100864, pl.INT64), 32), pl.Mem.Vec] = pl.tile.row_sum(
                        q_head_sq_inline2977__tile_1, tmp_tile_1
                    )
                    q_head_sq_sum_inline2960__tile_1: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_58, pl.const(100864, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(
                        q_head_sq_row_inline2961__tile_1, [1, 8]
                    )
                    q_head_sq_mean_inline2955__tile_1: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_45, pl.const(68096, pl.INT64), 32), pl.Mem.Vec] = pl.tile.muls(
                        q_head_sq_sum_inline2960__tile_1, 0.001953125
                    )
                    q_head_var_inline2959__tile_1: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_45, pl.const(68096, pl.INT64), 32), pl.Mem.Vec] = pl.tile.adds(
                        q_head_sq_mean_inline2955__tile_1, 9.9999999999999995e-07
                    )
                    rsqrt_tmp_1: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_46, pl.const(84480, pl.INT64), 32), pl.Mem.Vec] = pl.tile.create([1, 8], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                    q_head_inv_rms_inline2988__tile_1: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_58, pl.const(100864, pl.INT64), 32), pl.Mem.Vec] = pl.tile.rsqrt(
                        q_head_var_inline2959__tile_1, rsqrt_tmp_1
                    )
                    q_head_inv_rms_t_inline3013__tile_1: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_58, pl.const(100864, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(
                        q_head_inv_rms_inline2988__tile_1, [8, 1]
                    )
                    t__tile_6: pl.Tile[[8, 448], pl.FP32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 16128), pl.Mem.Vec] = pl.tile.slice(q_head_dq_v1_inline2998__tile_1, [8, 448], [0, 0])
                    q_nope_normed_inline2994__tile_1: pl.Tile[[8, 448], pl.FP32, pl.MemRef(mem_vec_45, pl.const(68096, pl.INT64), 14336), pl.Mem.Vec] = pl.tile.row_expand_mul(
                        t__tile_6, q_head_inv_rms_t_inline3013__tile_1
                    )
                    q_nope_bf16_inline2968__tile_1: pl.Tile[[8, 448], pl.BF16, pl.MemRef(mem_vec_45, pl.const(68096, pl.INT64), 7168), pl.Mem.Vec] = pl.tile.cast(
                        q_nope_normed_inline2994__tile_1, target_type=pl.BF16, mode="rint"
                    )
                    q_rope_chunk_raw_inline2992__tile_1: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(20224, pl.INT64), 14592), pl.Mem.Vec] = pl.tile.slice(
                        q_head_dq_v1_inline2998__tile_1, [8, 64], [0, 448]
                    )
                    q_rope_chunk_inline2985__tile_1: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.row_expand_mul(
                        q_rope_chunk_raw_inline2992__tile_1, q_head_inv_rms_t_inline3013__tile_1
                    )
                    t__tile_7: pl.Tile[[8, 64], pl.BF16, pl.MemRef(mem_vec_46, pl.const(84480, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.cast(
                        q_rope_chunk_inline2985__tile_1, target_type=pl.BF16, mode="rint"
                    )
                    q_rope_chunk_v1_inline2993__tile_1: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(
                        t__tile_7, target_type=pl.FP32, mode="round"
                    )
                    gather_acc_init_1: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_46, pl.const(84480, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.create([8, 64], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                    for gather_lv_1, (gather_ia_1,) in pl.range(8, init_values=(gather_acc_init_1,)):
                        gather_inp_row_1: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 256), pl.Mem.Vec] = pl.tile.slice(
                            q_rope_chunk_v1_inline2993__tile_1, [1, 64], [gather_lv_1, 0], [1, 64]
                        )
                        gather_idx_row_1: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 256), pl.Mem.Vec] = pl.tile.slice(
                            q_swap_idx_inline2972__tile, [1, 64], [gather_lv_1, 0], [1, 64]
                        )
                        gather_row_tmp_1: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_58, pl.const(100864, pl.INT64), 256), pl.Mem.Vec] = pl.tile.create(
                            [1, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec
                        )
                        gather_row_1: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_59, pl.const(101120, pl.INT64), 256), pl.Mem.Vec] = pl.tile.gather(
                            gather_inp_row_1, gather_idx_row_1, gather_row_tmp_1
                        )
                        gather_asmbl_1: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_46, pl.const(84480, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.assemble(gather_ia_1, gather_row_1, [gather_lv_1, 0])
                        gather_rv_1: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_46, pl.const(84480, pl.INT64), 2048), pl.Mem.Vec] = pl.yield_(gather_asmbl_1)
                    q_rope_swapped_inline3005__tile_1: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_46, pl.const(84480, pl.INT64), 2048), pl.Mem.Vec] = gather_rv_1
                    q_rope_base_inline2995__tile_1: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.mul(
                        q_rope_chunk_v1_inline2993__tile_1, q_cos_il_inline2974__tile
                    )
                    q_rope_delta_inline2996__tile_1: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_46, pl.const(84480, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.mul(
                        q_rope_swapped_inline3005__tile_1, q_sin_signed_inline2991__tile
                    )
                    q_rope_rot_inline3006__tile_1: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.add(
                        q_rope_base_inline2995__tile_1, q_rope_delta_inline2996__tile_1
                    )
                    q_rope_bf16_inline2979__tile_1: pl.Tile[[8, 64], pl.BF16, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.cast(
                        q_rope_rot_inline3006__tile_1, target_type=pl.BF16, mode="rint"
                    )
                    q_flat_inline2971__tile_2: pl.Tensor[[t_dim_inline3002__ssa_v0, 32768], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                        q_nope_bf16_inline2968__tile_1, [out_tg_inline2978__ssa_v0, h0_inline2970__ssa_v0_1], q_flat_inline2971__tile_1
                    )
                    q_flat_inline2971__tile_3: pl.Tensor[[t_dim_inline3002__ssa_v0, 32768], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                        q_rope_bf16_inline2979__tile_1, [out_tg_inline2978__ssa_v0, h0_inline2970__ssa_v0_1 + 448], q_flat_inline2971__tile_2
                    )
                    q_flat_inline2971__rv_v4: pl.Tensor[[t_dim_inline3002__ssa_v0, 32768], pl.BF16, pl.MemRef("mem_ddr_65", pl.const(0, pl.INT64), 0)] = pl.yield_(q_flat_inline2971__tile_3)
                q_flat_inline2971__phi_v7: pl.Tensor[[t_dim_inline3002__ssa_v0, 32768], pl.BF16, pl.MemRef("mem_ddr_117", pl.const(0, pl.INT64), 0)] = pl.yield_(q_flat_inline2971__rv_v4)
            else:
                valid_tail_rows_inline2990__ssa_v0: pl.Scalar[pl.INDEX] = tile_rows_inline311_inline1886__ssa_v0 - tg_inline2983__ssa_v0
                qr_scale_dq_tail_inline3009__ssa_v0: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_9, pl.const(103456, pl.INT64), 32), pl.Mem.Vec] = pl.tile.load(
                    qr_scale_pad_store_inline1879__rv_v2, [tg_inline2983__ssa_v0, 0], [8, 1], [8, 1], target_memory=pl.Mem.Vec
                )
                q_cos_il_tail_inline2986__ssa_v0: pl.Tile[
                    [8, 64], pl.FP32, pl.MemRef(mem_vec_21, pl.const(51200, pl.INT64), 2048), pl.Mem.Vec, pl.TileView(valid_shape=[valid_tail_rows_inline2990__ssa_v0, 64])
                ] = pl.tile.load(q_rope_cos_il_inline545__ssa_v0, [out_tg_inline2978__ssa_v0, 0], [8, 64], [valid_tail_rows_inline2990__ssa_v0, 64], target_memory=pl.Mem.Vec)
                q_sin_signed_tail_inline2999__ssa_v0: pl.Tile[
                    [8, 64], pl.FP32, pl.MemRef(mem_vec_45, pl.const(68096, pl.INT64), 2048), pl.Mem.Vec, pl.TileView(valid_shape=[valid_tail_rows_inline2990__ssa_v0, 64])
                ] = pl.tile.load(q_rope_sin_signed_inline544__ssa_v0, [out_tg_inline2978__ssa_v0, 0], [8, 64], [valid_tail_rows_inline2990__ssa_v0, 64], target_memory=pl.Mem.Vec)
                t__tmp_v58: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_11, pl.const(2048, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.full([8, 64], dtype=pl.FP32, value=1.0)
                t__ci_tmp_v0: pl.Tile[[1, 192], pl.FP32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 768), pl.Mem.Vec] = pl.tile.create([1, 192], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                t__tmp_v59: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_20, pl.const(34816, pl.INT64), 256), pl.Mem.Vec] = pl.tile.ci(
                    pl.const(0, pl.INT32), [1, 64], tmp=t__ci_tmp_v0, dtype=pl.INT32, descending=False
                )
                t__tmp_v60: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 256), pl.Mem.Vec] = pl.tile.cast(t__tmp_v59, target_type=pl.FP32, mode="round")
                q_col_inline3000__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_11, pl.const(2048, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.col_expand_mul(t__tmp_v58, t__tmp_v60)
                t__tmp_v61: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.muls(q_col_inline3000__ssa_v0, 0.5)
                t__tmp_v62: pl.Tile[[8, 64], pl.INT32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(t__tmp_v61, target_type=pl.INT32, mode="trunc")
                q_dup_f_inline3003__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(t__tmp_v62, target_type=pl.FP32, mode="round")
                t__tmp_v63: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.muls(q_dup_f_inline3003__ssa_v0, 2.0)
                q_lane_inline2987__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.sub(q_col_inline3000__ssa_v0, t__tmp_v63)
                t__tmp_v64: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_11, pl.const(2048, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.adds(q_col_inline3000__ssa_v0, 1.0)
                t__tmp_v65: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.muls(q_lane_inline2987__ssa_v0, 2.0)
                q_swap_f_inline3004__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_11, pl.const(2048, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.sub(t__tmp_v64, t__tmp_v65)
                t__ci_tmp_v1: pl.Tile[[1, 192], pl.FP32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 768), pl.Mem.Vec] = pl.tile.create([1, 192], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                t__tmp_v66: pl.Tile[[1, 8], pl.INT32, pl.MemRef(mem_vec_20, pl.const(34816, pl.INT64), 32), pl.Mem.Vec] = pl.tile.ci(
                    pl.const(0, pl.INT32), [1, 8], tmp=t__ci_tmp_v1, dtype=pl.INT32, descending=False
                )
                t__tmp_v67: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 32), pl.Mem.Vec] = pl.tile.cast(t__tmp_v66, target_type=pl.FP32, mode="round")
                q_row_seed_inline2975__ssa_v0: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_20, pl.const(34816, pl.INT64), 32), pl.Mem.Vec] = pl.tile.muls(t__tmp_v67, 64.0)
                t__tmp_v68: pl.Tile[[64, 8], pl.FP32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.full([64, 8], dtype=pl.FP32, value=1.0)
                q_row_grid_inline3010__ssa_v0: pl.Tile[[64, 8], pl.FP32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.col_expand_mul(
                    t__tmp_v68, q_row_seed_inline2975__ssa_v0
                )
                transpose_tmp: pl.Tile[[64, 8], pl.FP32, pl.MemRef(mem_vec_20, pl.const(34816, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.create([64, 8], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                q_row_offset_inline3012__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_46, pl.const(84480, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.transpose(
                    q_row_grid_inline3010__ssa_v0, 0, 1, transpose_tmp
                )
                t__tmp_v69: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_11, pl.const(2048, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.add(q_swap_f_inline3004__ssa_v0, q_row_offset_inline3012__ssa_v0)
                q_swap_idx_tail_inline3007__ssa_v0: pl.Tile[[8, 64], pl.INT32, pl.MemRef(mem_vec_46, pl.const(84480, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(
                    t__tmp_v69, target_type=pl.INT32, mode="round"
                )
                q_head_reduce_tmp_inline3014__ssa_v0: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_11, pl.const(2048, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.create(
                    [8, 512], dtype=pl.FP32, target_memory=pl.Mem.Vec
                )
                q_gather_tmp_inline3017__ssa_v0: pl.Tile[[8, 64], pl.INT32, pl.MemRef(mem_vec_8, pl.const(101408, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.create(
                    [8, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec
                )
                for h_inner_tail_inline3018__idx_v0 in pl.range(4):
                    h_tail_inline3020__ssa_v0: pl.Scalar[pl.INDEX] = hg_inline3016__ssa_v0 + h_inner_tail_inline3018__idx_v0
                    h0_tail_inline2964__ssa_v0: pl.Scalar[pl.INDEX] = h_tail_inline3020__ssa_v0 * 512
                    q_head_acc_tail_inline2997__ssa_v0: pl.Tile[[8, 512], pl.INT32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.load(
                        q_proj_i32_inline310_inline1876__ssa_v9, [tg_inline2983__ssa_v0, h0_tail_inline2964__ssa_v0], [8, 512], [8, 512], target_memory=pl.Mem.Vec
                    )
                    q_head_scale_input_tail_inline2973__ssa_v0: pl.Tile[[512], pl.FP32, pl.MemRef(mem_vec_20, pl.const(34816, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.load(
                        wq_b_scale__ssa_v0, [h0_tail_inline2964__ssa_v0], [512], [512], target_memory=pl.Mem.Vec
                    )
                    q_head_scale_tail_inline3021__ssa_v0: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_20, pl.const(34816, pl.INT64), 2048), pl.Mem.Vec] = q_head_scale_input_tail_inline2973__ssa_v0
                    q_head_acc_fp32_tail_inline2958__ssa_v0: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.cast(
                        q_head_acc_tail_inline2997__ssa_v0, target_type=pl.FP32, mode="none"
                    )
                    q_head_row_scaled_tail_inline3019__ssa_v0: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.row_expand_mul(
                        q_head_acc_fp32_tail_inline2958__ssa_v0, qr_scale_dq_tail_inline3009__ssa_v0
                    )
                    q_head_dq_tail_inline2954__ssa_v0: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.col_expand_mul(
                        q_head_row_scaled_tail_inline3019__ssa_v0, q_head_scale_tail_inline3021__ssa_v0
                    )
                    t__tmp_v70: pl.Tile[[8, 512], pl.BF16, pl.MemRef(mem_vec_20, pl.const(34816, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.cast(
                        q_head_dq_tail_inline2954__ssa_v0, target_type=pl.BF16, mode="rint"
                    )
                    q_head_dq_tail_v1_inline3001__ssa_v0: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.cast(
                        t__tmp_v70, target_type=pl.FP32, mode="round"
                    )
                    q_head_sq_tail_inline2966__ssa_v0: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_20, pl.const(34816, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.mul(
                        q_head_dq_tail_v1_inline3001__ssa_v0, q_head_dq_tail_v1_inline3001__ssa_v0
                    )
                    q_head_sq_sum_tail_inline2953__ssa_v0: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 32), pl.Mem.Vec] = pl.tile.row_sum(
                        q_head_sq_tail_inline2966__ssa_v0, q_head_reduce_tmp_inline3014__ssa_v0
                    )
                    t__rm_a0_tmp_v0: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(q_head_sq_sum_tail_inline2953__ssa_v0, [1, 8])
                    t__row_major_tmp_v1: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_20, pl.const(34816, pl.INT64), 32), pl.Mem.Vec] = pl.tile.muls(t__rm_a0_tmp_v0, 0.001953125)
                    t__tmp_v71: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_20, pl.const(34816, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(t__row_major_tmp_v1, [8, 1])
                    t__rm_a0_tmp_v2: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_20, pl.const(34816, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(t__tmp_v71, [1, 8])
                    t__row_major_tmp_v3: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_20, pl.const(34816, pl.INT64), 32), pl.Mem.Vec] = pl.tile.adds(t__rm_a0_tmp_v2, 9.9999999999999995e-07)
                    t__tmp_v72: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_20, pl.const(34816, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(t__row_major_tmp_v3, [8, 1])
                    t__rm_a0_tmp_v4: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_20, pl.const(34816, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(t__tmp_v72, [1, 8])
                    t__row_major_tmp_v5: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_20, pl.const(34816, pl.INT64), 32), pl.Mem.Vec] = pl.tile.sqrt(t__rm_a0_tmp_v4)
                    t__tmp_v73: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_20, pl.const(34816, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(t__row_major_tmp_v5, [8, 1])
                    q_head_inv_rms_tail_inline2952__rm_a0_tmp_v6: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_20, pl.const(34816, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(t__tmp_v73, [1, 8])
                    q_head_inv_rms_tail_inline2952__row_major_tmp_v7: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 32), pl.Mem.Vec] = pl.tile.recip(
                        q_head_inv_rms_tail_inline2952__rm_a0_tmp_v6
                    )
                    q_head_inv_rms_tail_inline2952__ssa_v0: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(
                        q_head_inv_rms_tail_inline2952__row_major_tmp_v7, [8, 1]
                    )
                    t__tmp_v74: pl.Tile[[8, 448], pl.FP32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 16128), pl.Mem.Vec] = pl.tile.slice(q_head_dq_tail_v1_inline3001__ssa_v0, [8, 448], [0, 0])
                    q_nope_normed_tail_inline2951__ssa_v0: pl.Tile[[8, 448], pl.FP32, pl.MemRef(mem_vec_20, pl.const(34816, pl.INT64), 14336), pl.Mem.Vec] = pl.tile.row_expand_mul(
                        t__tmp_v74, q_head_inv_rms_tail_inline2952__ssa_v0
                    )
                    q_nope_bf16_tail_inline3011__ssa_v0: pl.Tile[[8, 448], pl.BF16, pl.MemRef(mem_vec_20, pl.const(34816, pl.INT64), 7168), pl.Mem.Vec] = pl.tile.cast(
                        q_nope_normed_tail_inline2951__ssa_v0, target_type=pl.BF16, mode="rint"
                    )
                    q_nope_valid_inline2950__ssa_v0: pl.Tile[
                        [8, 448], pl.BF16, pl.MemRef(mem_vec_20, pl.const(34816, pl.INT64), 7168), pl.Mem.Vec, pl.TileView(valid_shape=[valid_tail_rows_inline2990__ssa_v0, 448])
                    ] = pl.tile.set_validshape(q_nope_bf16_tail_inline3011__ssa_v0, valid_tail_rows_inline2990__ssa_v0, 448)
                    pl.tile.store(q_nope_valid_inline2950__ssa_v0, [out_tg_inline2978__ssa_v0, h0_tail_inline2964__ssa_v0], q_flat_inline2971__iter_v1)
                    q_rope_chunk_raw_tail_inline2976__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(20224, pl.INT64), 14592), pl.Mem.Vec] = pl.tile.slice(
                        q_head_dq_tail_v1_inline3001__ssa_v0, [8, 64], [0, 448]
                    )
                    q_rope_chunk_tail_inline3008__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.row_expand_mul(
                        q_rope_chunk_raw_tail_inline2976__ssa_v0, q_head_inv_rms_tail_inline2952__ssa_v0
                    )
                    t__tmp_v75: pl.Tile[[8, 64], pl.BF16, pl.MemRef(mem_vec_20, pl.const(34816, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.cast(
                        q_rope_chunk_tail_inline3008__ssa_v0, target_type=pl.BF16, mode="rint"
                    )
                    q_rope_chunk_tail_v1_inline2963__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(
                        t__tmp_v75, target_type=pl.FP32, mode="round"
                    )
                    q_rope_swapped_tail_inline2957__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_20, pl.const(34816, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.gather(
                        q_rope_chunk_tail_v1_inline2963__ssa_v0, q_swap_idx_tail_inline3007__ssa_v0, q_gather_tmp_inline3017__ssa_v0
                    )
                    q_rope_base_tail_inline3015__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.mul(
                        q_rope_chunk_tail_v1_inline2963__ssa_v0, q_cos_il_tail_inline2986__ssa_v0
                    )
                    q_rope_delta_tail_inline2949__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_20, pl.const(34816, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.mul(
                        q_rope_swapped_tail_inline2957__ssa_v0, q_sin_signed_tail_inline2999__ssa_v0
                    )
                    q_rope_rot_tail_inline2948__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.add(
                        q_rope_base_tail_inline3015__ssa_v0, q_rope_delta_tail_inline2949__ssa_v0
                    )
                    q_rope_bf16_tail_inline2947__ssa_v0: pl.Tile[[8, 64], pl.BF16, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.cast(
                        q_rope_rot_tail_inline2948__ssa_v0, target_type=pl.BF16, mode="rint"
                    )
                    q_rope_valid_inline2946__ssa_v0: pl.Tile[
                        [8, 64], pl.BF16, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 1024), pl.Mem.Vec, pl.TileView(valid_shape=[valid_tail_rows_inline2990__ssa_v0, 64])
                    ] = pl.tile.set_validshape(q_rope_bf16_tail_inline2947__ssa_v0, valid_tail_rows_inline2990__ssa_v0, 64)
                    pl.tile.store(q_rope_valid_inline2946__ssa_v0, [out_tg_inline2978__ssa_v0, h0_tail_inline2964__ssa_v0 + 448], q_flat_inline2971__iter_v1)
                q_flat_inline2971__phi_v7: pl.Tensor[[t_dim_inline3002__ssa_v0, 32768], pl.BF16, pl.MemRef("mem_ddr_117", pl.const(0, pl.INT64), 0)] = pl.yield_(q_flat_inline2971__iter_v1)
            q_flat_inline2971__rv_v2: pl.Tensor[[t_dim_inline3002__ssa_v0, 32768], pl.BF16, pl.MemRef("mem_ddr_118", pl.const(0, pl.INT64), 0)] = pl.yield_(q_flat_inline2971__phi_v7)
        return

    @pl.function(type=pl.FunctionType.Spmd)
    def qproj_dequant_rms_nope_rope_spmd(
        self,
        q_flat_inline2971__ssa_v0: pl.Out[pl.Tensor[[t_dim_inline3002__ssa_v0, 32768], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)]],
        tile_rows_inline311_inline1886__ssa_v0: pl.Scalar[pl.INDEX],
        tile_base_inline314_inline1882__idx_v0: pl.Scalar[pl.INDEX],
        qr_scale_pad_store_inline1879__rv_v2: pl.Tensor[[512, 1], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 2048)],
        q_rope_cos_il_inline545__ssa_v0: pl.Tensor[[t_dim_inline546__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)],
        q_rope_sin_signed_inline544__ssa_v0: pl.Tensor[[t_dim_inline546__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)],
        q_rope_swap_idx_inline543__ssa_v0: pl.Tensor[[t_dim_inline546__ssa_v0, 64], pl.INT32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 0)],
        q_proj_i32_inline310_inline1876__ssa_v9: pl.Tensor[[qproj_t_matmul_inline313_inline1884__ssa_v0, 32768], pl.INT32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 0)],
        wq_b_scale__ssa_v0: pl.Tensor[[32768], pl.FP32, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 131072)],
    ):
        pl.func_attr({"mx_tensor_views_blocked": True})
        self.qproj_dequant_rms_nope_rope(
            q_flat_inline2971__ssa_v0,
            tile_rows_inline311_inline1886__ssa_v0,
            tile_base_inline314_inline1882__idx_v0,
            qr_scale_pad_store_inline1879__rv_v2,
            q_rope_cos_il_inline545__ssa_v0,
            q_rope_sin_signed_inline544__ssa_v0,
            q_rope_swap_idx_inline543__ssa_v0,
            q_proj_i32_inline310_inline1876__ssa_v9,
            wq_b_scale__ssa_v0,
            attrs={"arg_directions": [pl.adir.output_existing, pl.adir.scalar, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input]},
        )

    @pl.function(type=pl.FunctionType.AIC, level=pl.Level.AIC, role=pl.Role.SubWorker)
    def qproj_matmul(
        q_proj_i32_inline310_inline1876__ssa_v0: pl.Out[pl.Tensor[[qproj_t_matmul_inline313_inline1884__ssa_v0, 32768], pl.INT32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)]],
        qproj_full_rows_inline2934__ssa_v0: pl.Scalar[pl.INDEX],
        qr_i8_matmul_inline1874__rv_v2: pl.Tensor[[512, 1024], pl.INT8, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 524288)],
        wq_b__ssa_v0: pl.Tensor[[1024, 32768], pl.INT8, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 33554432)],
        qproj_t_matmul_inline2932__ssa_v0: pl.Scalar[pl.INDEX],
        tile_rows_inline311_inline1886__ssa_v0: pl.Scalar[pl.INDEX],
    ) -> pl.Tensor[[qproj_t_matmul_inline313_inline1884__ssa_v0, 32768], pl.INT32]:
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
        qproj_worker_inline2937__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        for qproj_n_idx_inline2938__idx_v0, (q_proj_i32_inline310_inline1876__iter_v1,) in pl.range(qproj_worker_inline2937__ssa_v0, 64, 24, init_values=(q_proj_i32_inline310_inline1876__ssa_v0,)):
            w_col0_inline2939__ssa_v0: pl.Scalar[pl.INDEX] = qproj_n_idx_inline2938__idx_v0 * 512
            for t0_inline2941__idx_v0, (q_proj_i32_inline310_inline1876__iter_v3,) in pl.range(0, qproj_full_rows_inline2934__ssa_v0, 64, init_values=(q_proj_i32_inline310_inline1876__iter_v1,)):
                col_acc_inline2942__tile: pl.Tile[[64, 512], pl.INT32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 131072), pl.Mem.Acc] = pl.tile.create(
                    [64, 512], dtype=pl.INT32, target_memory=pl.Mem.Acc
                )
                for qr_proj_col0_inline2943__idx_v0, (col_acc_inline2942__iter_v1,) in pl.range(0, 1024, 512, init_values=(col_acc_inline2942__tile,)):
                    qr_i8_chunk_inline2933__tile: pl.Tile[[64, 256], pl.INT8, pl.MemRef(mem_mat_4, pl.const(0, pl.INT64), 16384), pl.Mem.Mat] = pl.tile.load(
                        qr_i8_matmul_inline1874__rv_v2, [t0_inline2941__idx_v0, qr_proj_col0_inline2943__idx_v0], [64, 256], [64, 256], target_memory=pl.Mem.Mat
                    )
                    wq_chunk_inline2944__tile: pl.Tile[[256, 512], pl.INT8, pl.MemRef(mem_mat_5, pl.const(16384, pl.INT64), 131072), pl.Mem.Mat] = pl.tile.load(
                        wq_b__ssa_v0, [qr_proj_col0_inline2943__idx_v0, w_col0_inline2939__ssa_v0], [256, 512], [256, 512], target_memory=pl.Mem.Mat
                    )
                    qr_i8_chunk_inline2933__tile_1: pl.Tile[[64, 256], pl.INT8, pl.MemRef(mem_mat_6, pl.const(147456, pl.INT64), 16384), pl.Mem.Mat] = pl.tile.load(
                        qr_i8_matmul_inline1874__rv_v2, [t0_inline2941__idx_v0, qr_proj_col0_inline2943__idx_v0 + 256], [64, 256], [64, 256], target_memory=pl.Mem.Mat
                    )
                    wq_chunk_inline2944__tile_1: pl.Tile[[256, 512], pl.INT8, pl.MemRef(mem_mat_7, pl.const(163840, pl.INT64), 131072), pl.Mem.Mat] = pl.tile.load(
                        wq_b__ssa_v0, [qr_proj_col0_inline2943__idx_v0 + 256, w_col0_inline2939__ssa_v0], [256, 512], [256, 512], target_memory=pl.Mem.Mat
                    )
                    for col_acc_inline2942__tile_l0_ko, (col_acc_inline2942__tile_l0_c,) in pl.range(0, 256, 128, init_values=(col_acc_inline2942__iter_v1,)):
                        col_acc_inline2942__tile_l0_a: pl.Tile[[64, 64], pl.INT8, pl.MemRef(mem_left_8, pl.const(12288, pl.INT64), 4096), pl.Mem.Left, pl.TileView(blayout=pl.TileLayout.row_major)] = (
                            pl.tile.extract(qr_i8_chunk_inline2933__tile, 0, col_acc_inline2942__tile_l0_ko, [64, 64], target_memory=pl.Mem.Left)
                        )
                        col_acc_inline2942__tile_l0_b: pl.Tile[[64, 512], pl.INT8, pl.MemRef(mem_right_9, pl.const(32768, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                            wq_chunk_inline2944__tile, col_acc_inline2942__tile_l0_ko, 0, [64, 512], target_memory=pl.Mem.Right
                        )
                        col_acc_inline2942__tile_l0_a_1: pl.Tile[[64, 64], pl.INT8, pl.MemRef(mem_left_10, pl.const(0, pl.INT64), 4096), pl.Mem.Left, pl.TileView(blayout=pl.TileLayout.row_major)] = (
                            pl.tile.extract(qr_i8_chunk_inline2933__tile, 0, col_acc_inline2942__tile_l0_ko + 64, [64, 64], target_memory=pl.Mem.Left)
                        )
                        col_acc_inline2942__tile_l0_b_1: pl.Tile[[64, 512], pl.INT8, pl.MemRef(mem_right_11, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                            wq_chunk_inline2944__tile, col_acc_inline2942__tile_l0_ko + 64, 0, [64, 512], target_memory=pl.Mem.Right
                        )
                        col_acc_inline2942__tile_l0_c_acc: pl.Tile[[64, 512], pl.INT32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 131072), pl.Mem.Acc] = pl.tile.matmul_acc(
                            col_acc_inline2942__tile_l0_c, col_acc_inline2942__tile_l0_a, col_acc_inline2942__tile_l0_b, qr_proj_col0_inline2943__idx_v0 == 0 and col_acc_inline2942__tile_l0_ko == 0
                        )
                        col_acc_inline2942__tile_l0_c_acc_1: pl.Tile[[64, 512], pl.INT32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 131072), pl.Mem.Acc] = pl.tile.matmul_acc(
                            col_acc_inline2942__tile_l0_c_acc,
                            col_acc_inline2942__tile_l0_a_1,
                            col_acc_inline2942__tile_l0_b_1,
                            qr_proj_col0_inline2943__idx_v0 == 0 and col_acc_inline2942__tile_l0_ko == -64,
                        )
                        col_acc_inline2942__tile_1: pl.Tile[[64, 512], pl.INT32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 131072), pl.Mem.Acc] = pl.yield_(col_acc_inline2942__tile_l0_c_acc_1)
                    for col_acc_inline2942__tile_l0_ko_1, (col_acc_inline2942__tile_l0_c_1,) in pl.range(0, 256, 128, init_values=(col_acc_inline2942__tile_1,)):
                        col_acc_inline2942__tile_l0_a_2: pl.Tile[
                            [64, 64], pl.INT8, pl.MemRef(mem_left_13, pl.const(4096, pl.INT64), 4096), pl.Mem.Left, pl.TileView(blayout=pl.TileLayout.row_major)
                        ] = pl.tile.extract(qr_i8_chunk_inline2933__tile_1, 0, col_acc_inline2942__tile_l0_ko_1, [64, 64], target_memory=pl.Mem.Left)
                        col_acc_inline2942__tile_l0_b_2: pl.Tile[[64, 512], pl.INT8, pl.MemRef(mem_right_9, pl.const(32768, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                            wq_chunk_inline2944__tile_1, col_acc_inline2942__tile_l0_ko_1, 0, [64, 512], target_memory=pl.Mem.Right
                        )
                        col_acc_inline2942__tile_l0_a_3: pl.Tile[
                            [64, 64], pl.INT8, pl.MemRef(mem_left_15, pl.const(8192, pl.INT64), 4096), pl.Mem.Left, pl.TileView(blayout=pl.TileLayout.row_major)
                        ] = pl.tile.extract(qr_i8_chunk_inline2933__tile_1, 0, col_acc_inline2942__tile_l0_ko_1 + 64, [64, 64], target_memory=pl.Mem.Left)
                        col_acc_inline2942__tile_l0_b_3: pl.Tile[[64, 512], pl.INT8, pl.MemRef(mem_right_11, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                            wq_chunk_inline2944__tile_1, col_acc_inline2942__tile_l0_ko_1 + 64, 0, [64, 512], target_memory=pl.Mem.Right
                        )
                        col_acc_inline2942__tile_l0_c_acc_2: pl.Tile[[64, 512], pl.INT32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 131072), pl.Mem.Acc] = pl.tile.matmul_acc(
                            col_acc_inline2942__tile_l0_c_1,
                            col_acc_inline2942__tile_l0_a_2,
                            col_acc_inline2942__tile_l0_b_2,
                            qr_proj_col0_inline2943__idx_v0 == -256 and col_acc_inline2942__tile_l0_ko_1 == 0,
                        )
                        col_acc_inline2942__tile_l0_c_acc_3: pl.Tile[[64, 512], pl.INT32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 131072), pl.Mem.Acc] = pl.tile.matmul_acc(
                            col_acc_inline2942__tile_l0_c_acc_2,
                            col_acc_inline2942__tile_l0_a_3,
                            col_acc_inline2942__tile_l0_b_3,
                            qr_proj_col0_inline2943__idx_v0 == -256 and col_acc_inline2942__tile_l0_ko_1 == -64,
                        )
                        col_acc_inline2942__tile_2: pl.Tile[[64, 512], pl.INT32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 131072), pl.Mem.Acc] = pl.yield_(col_acc_inline2942__tile_l0_c_acc_3)
                    col_acc_inline2942__rv_v2: pl.Tile[[64, 512], pl.INT32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 131072), pl.Mem.Acc] = pl.yield_(col_acc_inline2942__tile_2)
                q_proj_i32_inline310_inline1876__tile: pl.Tensor[[qproj_t_matmul_inline313_inline1884__ssa_v0, 32768], pl.INT32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                    col_acc_inline2942__rv_v2, [t0_inline2941__idx_v0, w_col0_inline2939__ssa_v0], q_proj_i32_inline310_inline1876__iter_v3
                )
                q_proj_i32_inline310_inline1876__rv_v4: pl.Tensor[[qproj_t_matmul_inline313_inline1884__ssa_v0, 32768], pl.INT32, pl.MemRef("mem_ddr_19", pl.const(0, pl.INT64), 0)] = pl.yield_(
                    q_proj_i32_inline310_inline1876__tile
                )
            tail_w_col0_inline2945__ssa_v0: pl.Scalar[pl.INDEX] = w_col0_inline2939__ssa_v0
            for tail_t0_inline2935__idx_v0, (q_proj_i32_inline310_inline1876__iter_v6,) in pl.range(
                qproj_full_rows_inline2934__ssa_v0, qproj_t_matmul_inline2932__ssa_v0, 16, init_values=(q_proj_i32_inline310_inline1876__rv_v4,)
            ):
                qproj_tail_rows_inline2931__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(tile_rows_inline311_inline1886__ssa_v0 - tail_t0_inline2935__idx_v0, 16)
                tail_acc_inline2940__tile: pl.Tile[[16, 512], pl.INT32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 65536), pl.Mem.Acc] = pl.tile.create(
                    [16, 512], dtype=pl.INT32, target_memory=pl.Mem.Acc
                )
                for tail_qr_col0_inline2930__idx_v0, (tail_acc_inline2940__iter_v1,) in pl.range(0, 1024, 512, init_values=(tail_acc_inline2940__tile,)):
                    qr_i8_tail_inline2929__tile: pl.Tile[
                        [16, 256], pl.INT8, pl.MemRef(mem_mat_4, pl.const(0, pl.INT64), 4096), pl.Mem.Mat, pl.TileView(valid_shape=[qproj_tail_rows_inline2931__ssa_v0, 256])
                    ] = pl.tile.load(
                        qr_i8_matmul_inline1874__rv_v2, [tail_t0_inline2935__idx_v0, tail_qr_col0_inline2930__idx_v0], [16, 256], [qproj_tail_rows_inline2931__ssa_v0, 256], target_memory=pl.Mem.Mat
                    )
                    wq_tail_inline2928__tile: pl.Tile[[256, 512], pl.INT8, pl.MemRef(mem_mat_5, pl.const(16384, pl.INT64), 131072), pl.Mem.Mat] = pl.tile.load(
                        wq_b__ssa_v0, [tail_qr_col0_inline2930__idx_v0, tail_w_col0_inline2945__ssa_v0], [256, 512], [256, 512], target_memory=pl.Mem.Mat
                    )
                    qr_i8_tail_inline2929__tile_1: pl.Tile[
                        [16, 256], pl.INT8, pl.MemRef(mem_mat_6, pl.const(147456, pl.INT64), 4096), pl.Mem.Mat, pl.TileView(valid_shape=[qproj_tail_rows_inline2931__ssa_v0, 256])
                    ] = pl.tile.load(
                        qr_i8_matmul_inline1874__rv_v2,
                        [tail_t0_inline2935__idx_v0, tail_qr_col0_inline2930__idx_v0 + 256],
                        [16, 256],
                        [qproj_tail_rows_inline2931__ssa_v0, 256],
                        target_memory=pl.Mem.Mat,
                    )
                    wq_tail_inline2928__tile_1: pl.Tile[[256, 512], pl.INT8, pl.MemRef(mem_mat_7, pl.const(163840, pl.INT64), 131072), pl.Mem.Mat] = pl.tile.load(
                        wq_b__ssa_v0, [tail_qr_col0_inline2930__idx_v0 + 256, tail_w_col0_inline2945__ssa_v0], [256, 512], [256, 512], target_memory=pl.Mem.Mat
                    )
                    for tail_acc_inline2940__tile_l0_ko, (tail_acc_inline2940__tile_l0_c,) in pl.range(0, 256, 128, init_values=(tail_acc_inline2940__iter_v1,)):
                        tail_acc_inline2940__tile_l0_a: pl.Tile[
                            [16, 64],
                            pl.INT8,
                            pl.MemRef(mem_left_8, pl.const(12288, pl.INT64), 1024),
                            pl.Mem.Left,
                            pl.TileView(valid_shape=[qproj_tail_rows_inline2931__ssa_v0, 64], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal),
                        ] = pl.tile.extract(qr_i8_tail_inline2929__tile, 0, tail_acc_inline2940__tile_l0_ko, [16, 64], target_memory=pl.Mem.Left)
                        tail_acc_inline2940__tile_l0_b: pl.Tile[[64, 512], pl.INT8, pl.MemRef(mem_right_9, pl.const(32768, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                            wq_tail_inline2928__tile, tail_acc_inline2940__tile_l0_ko, 0, [64, 512], target_memory=pl.Mem.Right
                        )
                        tail_acc_inline2940__tile_l0_a_1: pl.Tile[
                            [16, 64],
                            pl.INT8,
                            pl.MemRef(mem_left_10, pl.const(0, pl.INT64), 1024),
                            pl.Mem.Left,
                            pl.TileView(valid_shape=[qproj_tail_rows_inline2931__ssa_v0, 64], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal),
                        ] = pl.tile.extract(qr_i8_tail_inline2929__tile, 0, tail_acc_inline2940__tile_l0_ko + 64, [16, 64], target_memory=pl.Mem.Left)
                        tail_acc_inline2940__tile_l0_b_1: pl.Tile[[64, 512], pl.INT8, pl.MemRef(mem_right_11, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                            wq_tail_inline2928__tile, tail_acc_inline2940__tile_l0_ko + 64, 0, [64, 512], target_memory=pl.Mem.Right
                        )
                        tail_acc_inline2940__tile_l0_c_acc: pl.Tile[[16, 512], pl.INT32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 65536), pl.Mem.Acc] = pl.tile.matmul_acc(
                            tail_acc_inline2940__tile_l0_c,
                            tail_acc_inline2940__tile_l0_a,
                            tail_acc_inline2940__tile_l0_b,
                            tail_qr_col0_inline2930__idx_v0 == 0 and tail_acc_inline2940__tile_l0_ko == 0,
                        )
                        tail_acc_inline2940__tile_l0_c_acc_1: pl.Tile[[16, 512], pl.INT32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 65536), pl.Mem.Acc] = pl.tile.matmul_acc(
                            tail_acc_inline2940__tile_l0_c_acc,
                            tail_acc_inline2940__tile_l0_a_1,
                            tail_acc_inline2940__tile_l0_b_1,
                            tail_qr_col0_inline2930__idx_v0 == 0 and tail_acc_inline2940__tile_l0_ko == -64,
                        )
                        tail_acc_inline2940__tile_1: pl.Tile[[16, 512], pl.INT32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 65536), pl.Mem.Acc] = pl.yield_(tail_acc_inline2940__tile_l0_c_acc_1)
                    for tail_acc_inline2940__tile_l0_ko_1, (tail_acc_inline2940__tile_l0_c_1,) in pl.range(0, 256, 128, init_values=(tail_acc_inline2940__tile_1,)):
                        tail_acc_inline2940__tile_l0_a_2: pl.Tile[
                            [16, 64],
                            pl.INT8,
                            pl.MemRef(mem_left_13, pl.const(4096, pl.INT64), 1024),
                            pl.Mem.Left,
                            pl.TileView(valid_shape=[qproj_tail_rows_inline2931__ssa_v0, 64], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal),
                        ] = pl.tile.extract(qr_i8_tail_inline2929__tile_1, 0, tail_acc_inline2940__tile_l0_ko_1, [16, 64], target_memory=pl.Mem.Left)
                        tail_acc_inline2940__tile_l0_b_2: pl.Tile[[64, 512], pl.INT8, pl.MemRef(mem_right_9, pl.const(32768, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                            wq_tail_inline2928__tile_1, tail_acc_inline2940__tile_l0_ko_1, 0, [64, 512], target_memory=pl.Mem.Right
                        )
                        tail_acc_inline2940__tile_l0_a_3: pl.Tile[
                            [16, 64],
                            pl.INT8,
                            pl.MemRef(mem_left_15, pl.const(8192, pl.INT64), 1024),
                            pl.Mem.Left,
                            pl.TileView(valid_shape=[qproj_tail_rows_inline2931__ssa_v0, 64], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal),
                        ] = pl.tile.extract(qr_i8_tail_inline2929__tile_1, 0, tail_acc_inline2940__tile_l0_ko_1 + 64, [16, 64], target_memory=pl.Mem.Left)
                        tail_acc_inline2940__tile_l0_b_3: pl.Tile[[64, 512], pl.INT8, pl.MemRef(mem_right_11, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                            wq_tail_inline2928__tile_1, tail_acc_inline2940__tile_l0_ko_1 + 64, 0, [64, 512], target_memory=pl.Mem.Right
                        )
                        tail_acc_inline2940__tile_l0_c_acc_2: pl.Tile[[16, 512], pl.INT32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 65536), pl.Mem.Acc] = pl.tile.matmul_acc(
                            tail_acc_inline2940__tile_l0_c_1,
                            tail_acc_inline2940__tile_l0_a_2,
                            tail_acc_inline2940__tile_l0_b_2,
                            tail_qr_col0_inline2930__idx_v0 == -256 and tail_acc_inline2940__tile_l0_ko_1 == 0,
                        )
                        tail_acc_inline2940__tile_l0_c_acc_3: pl.Tile[[16, 512], pl.INT32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 65536), pl.Mem.Acc] = pl.tile.matmul_acc(
                            tail_acc_inline2940__tile_l0_c_acc_2,
                            tail_acc_inline2940__tile_l0_a_3,
                            tail_acc_inline2940__tile_l0_b_3,
                            tail_qr_col0_inline2930__idx_v0 == -256 and tail_acc_inline2940__tile_l0_ko_1 == -64,
                        )
                        tail_acc_inline2940__tile_2: pl.Tile[[16, 512], pl.INT32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 65536), pl.Mem.Acc] = pl.yield_(tail_acc_inline2940__tile_l0_c_acc_3)
                    tail_acc_inline2940__rv_v2: pl.Tile[[16, 512], pl.INT32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 65536), pl.Mem.Acc] = pl.yield_(tail_acc_inline2940__tile_2)
                q_proj_i32_inline310_inline1876__tile_1: pl.Tensor[[qproj_t_matmul_inline313_inline1884__ssa_v0, 32768], pl.INT32, pl.MemRef("mem_ddr_19", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                    tail_acc_inline2940__rv_v2, [tail_t0_inline2935__idx_v0, tail_w_col0_inline2945__ssa_v0], q_proj_i32_inline310_inline1876__iter_v6
                )
                q_proj_i32_inline310_inline1876__rv_v7: pl.Tensor[[qproj_t_matmul_inline313_inline1884__ssa_v0, 32768], pl.INT32, pl.MemRef("mem_ddr_36", pl.const(0, pl.INT64), 0)] = pl.yield_(
                    q_proj_i32_inline310_inline1876__tile_1
                )
            q_proj_i32_inline310_inline1876__rv_v2: pl.Tensor[[qproj_t_matmul_inline313_inline1884__ssa_v0, 32768], pl.INT32, pl.MemRef("mem_ddr_37", pl.const(0, pl.INT64), 0)] = pl.yield_(
                q_proj_i32_inline310_inline1876__rv_v7
            )
        return q_proj_i32_inline310_inline1876__ssa_v0

    @pl.function(type=pl.FunctionType.Spmd)
    def qproj_matmul_spmd(
        self,
        q_proj_i32_inline310_inline1876__ssa_v0: pl.Out[pl.Tensor[[qproj_t_matmul_inline313_inline1884__ssa_v0, 32768], pl.INT32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)]],
        qproj_full_rows_inline2934__ssa_v0: pl.Scalar[pl.INDEX],
        qr_i8_matmul_inline1874__rv_v2: pl.Tensor[[512, 1024], pl.INT8, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 524288)],
        wq_b__ssa_v0: pl.Tensor[[1024, 32768], pl.INT8, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 33554432)],
        qproj_t_matmul_inline2932__ssa_v0: pl.Scalar[pl.INDEX],
        tile_rows_inline311_inline1886__ssa_v0: pl.Scalar[pl.INDEX],
    ) -> pl.Tensor[[qproj_t_matmul_inline313_inline1884__ssa_v0, 32768], pl.INT32]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        q_proj_i32_inline310_inline1876__rv_v2: pl.Tensor[[qproj_t_matmul_inline313_inline1884__ssa_v0, 32768], pl.INT32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)] = self.qproj_matmul(
            q_proj_i32_inline310_inline1876__ssa_v0,
            qproj_full_rows_inline2934__ssa_v0,
            qr_i8_matmul_inline1874__rv_v2,
            wq_b__ssa_v0,
            qproj_t_matmul_inline2932__ssa_v0,
            tile_rows_inline311_inline1886__ssa_v0,
            attrs={"arg_directions": [pl.adir.output_existing, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.scalar, pl.adir.scalar]},
        )
        return q_proj_i32_inline310_inline1876__ssa_v0

    @pl.function(type=pl.FunctionType.AIC, level=pl.Level.AIC, role=pl.Role.SubWorker)
    def qr_hadamard_matmul(
        hadamard_idx__ssa_v0: pl.Tensor[[128, 128], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 32768)],
        qh_acc_gm_inline990_inline2335__ssa_v0: pl.Out[pl.Tensor[[bs_heads_inline1000_inline2314__ssa_v0, 128], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)]],
        bs_heads_inline1000_inline2314__ssa_v0: pl.Scalar[pl.INDEX],
        qr_bf16_inline2281__ssa_v0: pl.Tensor[[24576, 128], pl.BF16, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 6291456)],
    ) -> pl.Tensor[[bs_heads_inline1000_inline2314__ssa_v0, 128], pl.FP32]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        mem_mat_3: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 32768)
        mem_mat_4: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 16384)
        mem_left_5: pl.Ptr = pl.tile.alloc(pl.Mem.Left, 16384)
        mem_right_6: pl.Ptr = pl.tile.alloc(pl.Mem.Right, 32768)
        mem_acc_7: pl.Ptr = pl.tile.alloc(pl.Mem.Acc, 32768)
        qh_worker_inline996_inline2284__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        qh_hadamard_inline1009_inline2336__tile: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_mat_3, pl.const(0, pl.INT64), 32768), pl.Mem.Mat] = pl.tile.load(
            hadamard_idx__ssa_v0, [0, 0], [128, 128], [128, 128], target_memory=pl.Mem.Mat
        )
        for idx_inline992_inline2337__idx_v0, (qh_acc_gm_inline990_inline2335__iter_v1,) in pl.range(
            qh_worker_inline996_inline2284__ssa_v0, bs_heads_inline1000_inline2314__ssa_v0 // 64, 24, init_values=(qh_acc_gm_inline990_inline2335__ssa_v0,)
        ):
            o0_inline994_inline2325__ssa_v0: pl.Scalar[pl.INDEX] = idx_inline992_inline2337__idx_v0 * 64
            t__tile: pl.Tile[[64, 128], pl.BF16, pl.MemRef(mem_mat_4, pl.const(32768, pl.INT64), 16384), pl.Mem.Mat] = pl.tile.load(
                qr_bf16_inline2281__ssa_v0, [o0_inline994_inline2325__ssa_v0, 0], [64, 128], [64, 128], target_memory=pl.Mem.Mat
            )
            t__tile_Left: pl.Tile[[64, 128], pl.BF16, pl.MemRef(mem_left_5, pl.const(0, pl.INT64), 16384), pl.Mem.Left] = pl.tile.move(t__tile, target_memory=pl.Mem.Left)
            qh_hadamard_inline1009_inline2336__tile_Right: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_6, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.move(
                qh_hadamard_inline1009_inline2336__tile, target_memory=pl.Mem.Right
            )
            qh_acc_inline995_inline2334__tile: pl.Tile[[64, 128], pl.FP32, pl.MemRef(mem_acc_7, pl.const(0, pl.INT64), 32768), pl.Mem.Acc] = pl.tile.matmul(
                t__tile_Left, qh_hadamard_inline1009_inline2336__tile_Right
            )
            qh_acc_gm_inline990_inline2335__tile: pl.Tensor[[bs_heads_inline1000_inline2314__ssa_v0, 128], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                qh_acc_inline995_inline2334__tile, [o0_inline994_inline2325__ssa_v0, 0], qh_acc_gm_inline990_inline2335__iter_v1
            )
            qh_acc_gm_inline990_inline2335__rv_v2: pl.Tensor[[bs_heads_inline1000_inline2314__ssa_v0, 128], pl.FP32, pl.MemRef("mem_ddr_8", pl.const(0, pl.INT64), 0)] = pl.yield_(
                qh_acc_gm_inline990_inline2335__tile
            )
        return qh_acc_gm_inline990_inline2335__ssa_v0

    @pl.function(type=pl.FunctionType.Spmd)
    def qr_hadamard_matmul_spmd(
        self,
        hadamard_idx__ssa_v0: pl.Tensor[[128, 128], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 32768)],
        qh_acc_gm_inline990_inline2335__ssa_v0: pl.Out[pl.Tensor[[bs_heads_inline1000_inline2314__ssa_v0, 128], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)]],
        bs_heads_inline1000_inline2314__ssa_v0: pl.Scalar[pl.INDEX],
        qr_bf16_inline2281__ssa_v0: pl.Tensor[[24576, 128], pl.BF16, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 6291456)],
    ) -> pl.Tensor[[bs_heads_inline1000_inline2314__ssa_v0, 128], pl.FP32]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        qh_acc_gm_inline990_inline2335__rv_v2: pl.Tensor[[bs_heads_inline1000_inline2314__ssa_v0, 128], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)] = self.qr_hadamard_matmul(
            hadamard_idx__ssa_v0,
            qh_acc_gm_inline990_inline2335__ssa_v0,
            bs_heads_inline1000_inline2314__ssa_v0,
            qr_bf16_inline2281__ssa_v0,
            attrs={"arg_directions": [pl.adir.input, pl.adir.output_existing, pl.adir.scalar, pl.adir.input]},
        )
        return qh_acc_gm_inline990_inline2335__ssa_v0

    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def qr_hadamard_quant(
        qr_hadamard_i8_inline569__ssa_v0: pl.Out[pl.Tensor[[24576, 128], pl.INT8, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 3145728)]],
        qr_hadamard_scale_dq_inline567__ssa_v0: pl.Out[pl.Tensor[[24576, 1], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 98304)]],
        bs_heads_inline1000_inline2314__ssa_v0: pl.Scalar[pl.INDEX],
        qh_acc_gm_inline990_inline2335__rv_v2: pl.Tensor[[bs_heads_inline1000_inline2314__ssa_v0, 128], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)],
    ) -> tuple[pl.Tensor[[24576, 128], pl.INT8], pl.Tensor[[24576, 1], pl.FP32]]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        mem_vec_3: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32768)
        mem_vec_9: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        mem_vec_12: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
        mem_vec_14: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32768)
        mem_vec_15: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        qh_quant_worker_inline998_inline2323__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        for idx_inline991_inline2338__idx_v0, (qr_hadamard_i8_inline569__iter_v1, qr_hadamard_scale_dq_inline567__iter_v1) in pl.range(
            qh_quant_worker_inline998_inline2323__ssa_v0, bs_heads_inline1000_inline2314__ssa_v0 // 64, 48, init_values=(qr_hadamard_i8_inline569__ssa_v0, qr_hadamard_scale_dq_inline567__ssa_v0)
        ):
            o0_inline994_inline2325__ssa_v1: pl.Scalar[pl.INDEX] = idx_inline991_inline2338__idx_v0 * 64
            qh_full_f32_inline1011_inline2266__tile: pl.Tile[[64, 128], pl.FP32, pl.MemRef(mem_vec_3, pl.const(49408, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.load(
                qh_acc_gm_inline990_inline2335__rv_v2, [o0_inline994_inline2325__ssa_v1, 0], [64, 128], [64, 128], target_memory=pl.Mem.Vec
            )
            t__tile: pl.Tile[[64, 128], pl.BF16, pl.MemRef(mem_vec_14, pl.const(16384, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.cast(
                qh_full_f32_inline1011_inline2266__tile, target_type=pl.BF16, mode="rint"
            )
            qh_full_f32_v1_inline1002_inline2294__tile: pl.Tile[[64, 128], pl.FP32, pl.MemRef(mem_vec_3, pl.const(49408, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.cast(
                t__tile, target_type=pl.FP32, mode="round"
            )
            t__tile_1: pl.Tile[[64, 128], pl.FP32, pl.MemRef(mem_vec_3, pl.const(49408, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.muls(qh_full_f32_v1_inline1002_inline2294__tile, 0.088388347648318447)
            t__tile_2: pl.Tile[[64, 128], pl.BF16, pl.MemRef(mem_vec_14, pl.const(16384, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.cast(t__tile_1, target_type=pl.BF16, mode="rint")
            qh_full_f32_v2_inline1005_inline2296__tile: pl.Tile[[64, 128], pl.FP32, pl.MemRef(mem_vec_3, pl.const(49408, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.cast(
                t__tile_2, target_type=pl.FP32, mode="round"
            )
            qh_amax_inline1003_inline2270__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_9, pl.const(82176, pl.INT64), 256), pl.Mem.Vec] = pl.tile.full([1, 64], dtype=pl.FP32, value=0.0001)
            for h0_inline1006_inline2276__idx_v0, (qh_amax_inline1003_inline2270__iter_v1,) in pl.range(0, 128, 64, init_values=(qh_amax_inline1003_inline2270__tile,)):
                qh_a_f32_inline1007_inline2279__tile_textract: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_14, pl.const(16384, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.extract(
                    qh_full_f32_v2_inline1005_inline2296__tile, 0, h0_inline1006_inline2276__idx_v0, [64, 64], target_memory=pl.Mem.Vec
                )
                qh_a_neg_inline1010_inline2265__tile: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_14, pl.const(16384, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.neg(
                    qh_a_f32_inline1007_inline2279__tile_textract
                )
                qh_a_f32_inline1007_inline2279__tile_textract_1: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(0, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.extract(
                    qh_full_f32_v2_inline1005_inline2296__tile, 0, h0_inline1006_inline2276__idx_v0, [64, 64], target_memory=pl.Mem.Vec
                )
                qh_a_abs_inline1012_inline2264__tile: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(0, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.maximum(
                    qh_a_f32_inline1007_inline2279__tile_textract_1, qh_a_neg_inline1010_inline2265__tile
                )
                tmp_tile: pl.Tile[[64, 128], pl.FP32, pl.MemRef(mem_vec_14, pl.const(16384, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.create([64, 128], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                qh_a_max_col_inline1013_inline2272__tile: pl.Tile[[64, 1], pl.FP32, pl.MemRef(mem_vec_15, pl.const(49152, pl.INT64), 256), pl.Mem.Vec] = pl.tile.row_max(
                    qh_a_abs_inline1012_inline2264__tile, tmp_tile
                )
                qh_a_max_inline993_inline2263__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_15, pl.const(49152, pl.INT64), 256), pl.Mem.Vec] = pl.tile.reshape(
                    qh_a_max_col_inline1013_inline2272__tile, [1, 64]
                )
                qh_amax_inline1003_inline2270__tile_1: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_9, pl.const(82176, pl.INT64), 256), pl.Mem.Vec] = pl.tile.maximum(
                    qh_amax_inline1003_inline2270__iter_v1, qh_a_max_inline993_inline2263__tile
                )
                qh_amax_inline1003_inline2270__rv_v2: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_9, pl.const(82176, pl.INT64), 256), pl.Mem.Vec] = pl.yield_(qh_amax_inline1003_inline2270__tile_1)
            qh_scale_numerator_inline1014_inline2306__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_14, pl.const(16384, pl.INT64), 256), pl.Mem.Vec] = pl.tile.full(
                [1, 64], dtype=pl.FP32, value=127.0
            )
            qh_scale_quant_row_inline1016_inline2274__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_14, pl.const(16384, pl.INT64), 256), pl.Mem.Vec] = pl.tile.div(
                qh_scale_numerator_inline1014_inline2306__tile, qh_amax_inline1003_inline2270__rv_v2
            )
            qh_scale_recip_inline1008_inline2286__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(0, pl.INT64), 256), pl.Mem.Vec] = pl.tile.recip(
                qh_scale_quant_row_inline1016_inline2274__tile
            )
            qh_scale_dq_inline1004_inline2308__tile: pl.Tile[[64, 1], pl.FP32, pl.MemRef(mem_vec_12, pl.const(0, pl.INT64), 256), pl.Mem.Vec] = pl.tile.reshape(
                qh_scale_recip_inline1008_inline2286__tile, [64, 1]
            )
            t__rm_a0_tmp_v0: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(0, pl.INT64), 256), pl.Mem.Vec] = pl.tile.reshape(qh_scale_dq_inline1004_inline2308__tile, [1, 64])
            t__row_major_tmp_v1: pl.Tile[[1, 64], pl.FP16, pl.MemRef(mem_vec_9, pl.const(82176, pl.INT64), 128), pl.Mem.Vec] = pl.tile.cast(t__rm_a0_tmp_v0, target_type=pl.FP16, mode="rint")
            t__tile_3: pl.Tile[[64, 1], pl.FP16, pl.MemRef(mem_vec_9, pl.const(82176, pl.INT64), 128), pl.Mem.Vec] = pl.tile.reshape(t__row_major_tmp_v1, [64, 1])
            t__rm_a0_tmp_v2: pl.Tile[[1, 64], pl.FP16, pl.MemRef(mem_vec_9, pl.const(82176, pl.INT64), 128), pl.Mem.Vec] = pl.tile.reshape(t__tile_3, [1, 64])
            t__row_major_tmp_v3: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(0, pl.INT64), 256), pl.Mem.Vec] = pl.tile.cast(t__rm_a0_tmp_v2, target_type=pl.FP32, mode="round")
            t__tile_4: pl.Tile[[64, 1], pl.FP32, pl.MemRef(mem_vec_12, pl.const(0, pl.INT64), 256), pl.Mem.Vec] = pl.tile.reshape(t__row_major_tmp_v3, [64, 1])
            qr_hadamard_scale_dq_inline567__tile: pl.Tensor[[24576, 1], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 98304)] = pl.tile.store(
                t__tile_4, [o0_inline994_inline2325__ssa_v1, 0], qr_hadamard_scale_dq_inline567__iter_v1
            )
            qh_scale_quant_inline999_inline2302__tile: pl.Tile[[64, 1], pl.FP32, pl.MemRef(mem_vec_14, pl.const(16384, pl.INT64), 256), pl.Mem.Vec] = pl.tile.reshape(
                qh_scale_quant_row_inline1016_inline2274__tile, [64, 1]
            )
            qh_q_scaled_inline1015_inline2298__tile: pl.Tile[[64, 128], pl.FP32, pl.MemRef(mem_vec_3, pl.const(49408, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.row_expand_mul(
                qh_full_f32_v2_inline1005_inline2296__tile, qh_scale_quant_inline999_inline2302__tile
            )
            qh_q_i32_inline988_inline2262__tile: pl.Tile[[64, 128], pl.INT32, pl.MemRef(mem_vec_3, pl.const(49408, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.cast(
                qh_q_scaled_inline1015_inline2298__tile, target_type=pl.INT32, mode="rint"
            )
            qh_q_half_inline987_inline2261__tile: pl.Tile[[64, 128], pl.FP16, pl.MemRef(mem_vec_3, pl.const(49408, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.cast(
                qh_q_i32_inline988_inline2262__tile, target_type=pl.FP16, mode="round"
            )
            qh_i8_inline986_inline2312__tile: pl.Tile[[64, 128], pl.INT8, pl.MemRef(mem_vec_3, pl.const(49408, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.cast(
                qh_q_half_inline987_inline2261__tile, target_type=pl.INT8, mode="trunc"
            )
            qr_hadamard_i8_inline569__tile: pl.Tensor[[24576, 128], pl.INT8, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 3145728)] = pl.tile.store(
                qh_i8_inline986_inline2312__tile, [o0_inline994_inline2325__ssa_v1, 0], qr_hadamard_i8_inline569__iter_v1
            )
            qr_hadamard_i8_inline569__rv_v2, qr_hadamard_scale_dq_inline567__rv_v2 = pl.yield_(qr_hadamard_i8_inline569__tile, qr_hadamard_scale_dq_inline567__tile)
        return qr_hadamard_i8_inline569__ssa_v0, qr_hadamard_scale_dq_inline567__ssa_v0

    @pl.function(type=pl.FunctionType.Spmd)
    def qr_hadamard_quant_spmd(
        self,
        qr_hadamard_i8_inline569__ssa_v0: pl.Out[pl.Tensor[[24576, 128], pl.INT8, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 3145728)]],
        qr_hadamard_scale_dq_inline567__ssa_v0: pl.Out[pl.Tensor[[24576, 1], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 98304)]],
        bs_heads_inline1000_inline2314__ssa_v0: pl.Scalar[pl.INDEX],
        qh_acc_gm_inline990_inline2335__rv_v2: pl.Tensor[[bs_heads_inline1000_inline2314__ssa_v0, 128], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)],
    ) -> tuple[pl.Tensor[[24576, 128], pl.INT8], pl.Tensor[[24576, 1], pl.FP32]]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        ret__tmp_v0: pl.Tuple[pl.Tensor[[24576, 128], pl.INT8], pl.Tensor[[24576, 1], pl.FP32]] = self.qr_hadamard_quant(
            qr_hadamard_i8_inline569__ssa_v0,
            qr_hadamard_scale_dq_inline567__ssa_v0,
            bs_heads_inline1000_inline2314__ssa_v0,
            qh_acc_gm_inline990_inline2335__rv_v2,
            attrs={"arg_directions": [pl.adir.output_existing, pl.adir.output_existing, pl.adir.scalar, pl.adir.input]},
        )
        qr_hadamard_i8_inline569__rv_v2: pl.Tensor[[24576, 128], pl.INT8, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 3145728)] = ret__tmp_v0[0]
        qr_hadamard_scale_dq_inline567__rv_v2: pl.Tensor[[24576, 1], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 98304)] = ret__tmp_v0[1]
        return qr_hadamard_i8_inline569__ssa_v0, qr_hadamard_scale_dq_inline567__ssa_v0

    @pl.function(type=pl.FunctionType.AIC, level=pl.Level.AIC, role=pl.Role.SubWorker)
    def qr_proj_matmul(
        qr_fp32_inline304_inline1875__rv_v2: pl.InOut[pl.Tensor[[qr_t_matmul_inline305_inline1881__ssa_v0, 1024], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)]],
        qr_full_rows_inline2846__ssa_v0: pl.Scalar[pl.INDEX],
        tile_base_inline307_inline1878__idx_v0: pl.Scalar[pl.INDEX],
        x_view_inline2843__ssa_v0: pl.Tensor[[pl.tensor.dim(x_normed_t__ssa_v0, pl.const(0, pl.INDEX)), 4096], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)],
        wq_a__ssa_v0: pl.Tensor[[4096, 1024], pl.BF16, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 8388608)],
        qr_t_matmul_inline2845__ssa_v0: pl.Scalar[pl.INDEX],
        tile_rows_inline306_inline1880__ssa_v0: pl.Scalar[pl.INDEX],
    ) -> pl.Tensor[[qr_t_matmul_inline305_inline1881__ssa_v0, 1024], pl.FP32]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        mem_acc_4: pl.Ptr = pl.tile.alloc(pl.Mem.Acc, 32768)
        mem_mat_5: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 32768)
        mem_mat_6: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 65536)
        mem_mat_7: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 32768)
        mem_mat_8: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 65536)
        mem_left_9: pl.Ptr = pl.tile.alloc(pl.Mem.Left, 16384)
        mem_right_10: pl.Ptr = pl.tile.alloc(pl.Mem.Right, 32768)
        mem_left_11: pl.Ptr = pl.tile.alloc(pl.Mem.Left, 16384)
        mem_right_12: pl.Ptr = pl.tile.alloc(pl.Mem.Right, 32768)
        mem_left_14: pl.Ptr = pl.tile.alloc(pl.Mem.Left, 16384)
        mem_left_16: pl.Ptr = pl.tile.alloc(pl.Mem.Left, 16384)
        qbg_idx_inline2855__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        q_a_col0_inline2848__ssa_v0: pl.Scalar[pl.INDEX] = qbg_idx_inline2855__ssa_v0 * 128
        for dense_t0_inline2857__idx_v0, (qr_fp32_inline304_inline1875__iter_v6,) in pl.range(0, qr_full_rows_inline2846__ssa_v0, 64, init_values=(qr_fp32_inline304_inline1875__rv_v2,)):
            dense_x0_inline2850__ssa_v0: pl.Scalar[pl.INDEX] = tile_base_inline307_inline1878__idx_v0 + dense_t0_inline2857__idx_v0
            dense_acc_inline2844__tile: pl.Tile[[64, 128], pl.FP32, pl.MemRef(mem_acc_4, pl.const(0, pl.INT64), 32768), pl.Mem.Acc] = pl.tile.create([64, 128], dtype=pl.FP32, target_memory=pl.Mem.Acc)
            for dense_k_inline2842__idx_v0, (dense_acc_inline2844__iter_v1,) in pl.range(0, 16, 2, init_values=(dense_acc_inline2844__tile,)):
                dense_d0_inline2852__ssa_v0: pl.Scalar[pl.INDEX] = dense_k_inline2842__idx_v0 * 256
                dense_d0_inline2852__ssa_v0_1: pl.Scalar[pl.INDEX] = dense_k_inline2842__idx_v0 * 256 + 256
                dense_x_inline2847__tile: pl.Tile[[64, 256], pl.BF16, pl.MemRef(mem_mat_5, pl.const(0, pl.INT64), 32768), pl.Mem.Mat] = pl.tile.load(
                    x_view_inline2843__ssa_v0, [dense_x0_inline2850__ssa_v0, dense_d0_inline2852__ssa_v0], [64, 256], [64, 256], target_memory=pl.Mem.Mat
                )
                dense_w_inline2854__tile: pl.Tile[[256, 128], pl.BF16, pl.MemRef(mem_mat_6, pl.const(32768, pl.INT64), 65536), pl.Mem.Mat] = pl.tile.load(
                    wq_a__ssa_v0, [dense_d0_inline2852__ssa_v0, q_a_col0_inline2848__ssa_v0], [256, 128], [256, 128], target_memory=pl.Mem.Mat
                )
                dense_x_inline2847__tile_1: pl.Tile[[64, 256], pl.BF16, pl.MemRef(mem_mat_7, pl.const(98304, pl.INT64), 32768), pl.Mem.Mat] = pl.tile.load(
                    x_view_inline2843__ssa_v0, [dense_x0_inline2850__ssa_v0, dense_d0_inline2852__ssa_v0_1], [64, 256], [64, 256], target_memory=pl.Mem.Mat
                )
                dense_w_inline2854__tile_1: pl.Tile[[256, 128], pl.BF16, pl.MemRef(mem_mat_8, pl.const(131072, pl.INT64), 65536), pl.Mem.Mat] = pl.tile.load(
                    wq_a__ssa_v0, [dense_d0_inline2852__ssa_v0_1, q_a_col0_inline2848__ssa_v0], [256, 128], [256, 128], target_memory=pl.Mem.Mat
                )
                dense_acc_inline2844__tile_l0_a: pl.Tile[[64, 128], pl.BF16, pl.MemRef(mem_left_9, pl.const(49152, pl.INT64), 16384), pl.Mem.Left, pl.TileView(blayout=pl.TileLayout.row_major)] = (
                    pl.tile.extract(dense_x_inline2847__tile, 0, 0, [64, 128], target_memory=pl.Mem.Left)
                )
                dense_acc_inline2844__tile_l0_b: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_10, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                    dense_w_inline2854__tile, 0, 0, [128, 128], target_memory=pl.Mem.Right
                )
                dense_acc_inline2844__tile_l0_a_1: pl.Tile[[64, 128], pl.BF16, pl.MemRef(mem_left_11, pl.const(0, pl.INT64), 16384), pl.Mem.Left, pl.TileView(blayout=pl.TileLayout.row_major)] = (
                    pl.tile.extract(dense_x_inline2847__tile, 0, 128, [64, 128], target_memory=pl.Mem.Left)
                )
                dense_acc_inline2844__tile_l0_b_1: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_12, pl.const(32768, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                    dense_w_inline2854__tile, 128, 0, [128, 128], target_memory=pl.Mem.Right
                )
                dense_acc_inline2844__tile_l0_c_acc: pl.Tile[[64, 128], pl.FP32, pl.MemRef(mem_acc_4, pl.const(0, pl.INT64), 32768), pl.Mem.Acc] = pl.tile.matmul_acc(
                    dense_acc_inline2844__iter_v1, dense_acc_inline2844__tile_l0_a, dense_acc_inline2844__tile_l0_b, dense_k_inline2842__idx_v0 == 0
                )
                dense_acc_inline2844__tile_l0_c_acc_1: pl.Tile[[64, 128], pl.FP32, pl.MemRef(mem_acc_4, pl.const(0, pl.INT64), 32768), pl.Mem.Acc] = pl.tile.matmul_acc(
                    dense_acc_inline2844__tile_l0_c_acc, dense_acc_inline2844__tile_l0_a_1, dense_acc_inline2844__tile_l0_b_1, False
                )
                dense_acc_inline2844__tile_l0_a_2: pl.Tile[[64, 128], pl.BF16, pl.MemRef(mem_left_14, pl.const(16384, pl.INT64), 16384), pl.Mem.Left, pl.TileView(blayout=pl.TileLayout.row_major)] = (
                    pl.tile.extract(dense_x_inline2847__tile_1, 0, 0, [64, 128], target_memory=pl.Mem.Left)
                )
                dense_acc_inline2844__tile_l0_b_2: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_10, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                    dense_w_inline2854__tile_1, 0, 0, [128, 128], target_memory=pl.Mem.Right
                )
                dense_acc_inline2844__tile_l0_a_3: pl.Tile[[64, 128], pl.BF16, pl.MemRef(mem_left_16, pl.const(32768, pl.INT64), 16384), pl.Mem.Left, pl.TileView(blayout=pl.TileLayout.row_major)] = (
                    pl.tile.extract(dense_x_inline2847__tile_1, 0, 128, [64, 128], target_memory=pl.Mem.Left)
                )
                dense_acc_inline2844__tile_l0_b_3: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_12, pl.const(32768, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                    dense_w_inline2854__tile_1, 128, 0, [128, 128], target_memory=pl.Mem.Right
                )
                dense_acc_inline2844__tile_l0_c_acc_2: pl.Tile[[64, 128], pl.FP32, pl.MemRef(mem_acc_4, pl.const(0, pl.INT64), 32768), pl.Mem.Acc] = pl.tile.matmul_acc(
                    dense_acc_inline2844__tile_l0_c_acc_1, dense_acc_inline2844__tile_l0_a_2, dense_acc_inline2844__tile_l0_b_2, dense_k_inline2842__idx_v0 == -1
                )
                dense_acc_inline2844__tile_l0_c_acc_3: pl.Tile[[64, 128], pl.FP32, pl.MemRef(mem_acc_4, pl.const(0, pl.INT64), 32768), pl.Mem.Acc] = pl.tile.matmul_acc(
                    dense_acc_inline2844__tile_l0_c_acc_2, dense_acc_inline2844__tile_l0_a_3, dense_acc_inline2844__tile_l0_b_3, False
                )
                dense_acc_inline2844__rv_v2: pl.Tile[[64, 128], pl.FP32, pl.MemRef(mem_acc_4, pl.const(0, pl.INT64), 32768), pl.Mem.Acc] = pl.yield_(dense_acc_inline2844__tile_l0_c_acc_3)
            qr_fp32_inline304_inline1875__tile: pl.Tensor[[qr_t_matmul_inline305_inline1881__ssa_v0, 1024], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                dense_acc_inline2844__rv_v2, [dense_t0_inline2857__idx_v0, q_a_col0_inline2848__ssa_v0], qr_fp32_inline304_inline1875__iter_v6, atomic=pl.AtomicType.Add
            )
            qr_fp32_inline304_inline1875__rv_v7: pl.Tensor[[qr_t_matmul_inline305_inline1881__ssa_v0, 1024], pl.FP32, pl.MemRef("mem_ddr_20", pl.const(0, pl.INT64), 0)] = pl.yield_(
                qr_fp32_inline304_inline1875__tile
            )
        for t0_inline2840__idx_v0, (qr_fp32_inline304_inline1875__iter_v9,) in pl.range(
            qr_full_rows_inline2846__ssa_v0, qr_t_matmul_inline2845__ssa_v0, 16, init_values=(qr_fp32_inline304_inline1875__rv_v7,)
        ):
            q_acc_inline2841__tile: pl.Tile[[16, 128], pl.FP32, pl.MemRef(mem_acc_4, pl.const(0, pl.INT64), 8192), pl.Mem.Acc] = pl.tile.create([16, 128], dtype=pl.FP32, target_memory=pl.Mem.Acc)
            for db_inline2838__idx_v0, (q_acc_inline2841__iter_v1,) in pl.range(0, 16, 2, init_values=(q_acc_inline2841__tile,)):
                qr_d0_inline2839__ssa_v0: pl.Scalar[pl.INDEX] = db_inline2838__idx_v0 * 256
                qr_rows_inline2837__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(tile_rows_inline306_inline1880__ssa_v0 - t0_inline2840__idx_v0, 16)
                x_t0_inline2836__ssa_v0: pl.Scalar[pl.INDEX] = tile_base_inline307_inline1878__idx_v0 + t0_inline2840__idx_v0
                qr_d0_inline2839__ssa_v0_1: pl.Scalar[pl.INDEX] = db_inline2838__idx_v0 * 256 + 256
                qr_rows_inline2837__ssa_v0_1: pl.Scalar[pl.INDEX] = pl.min(tile_rows_inline306_inline1880__ssa_v0 - t0_inline2840__idx_v0, 16)
                x_t0_inline2836__ssa_v0_1: pl.Scalar[pl.INDEX] = tile_base_inline307_inline1878__idx_v0 + t0_inline2840__idx_v0
                q_x_chunk_bf16_inline2835__tile: pl.Tile[
                    [16, 256], pl.BF16, pl.MemRef(mem_mat_5, pl.const(0, pl.INT64), 8192), pl.Mem.Mat, pl.TileView(valid_shape=[qr_rows_inline2837__ssa_v0, 256])
                ] = pl.tile.load(x_view_inline2843__ssa_v0, [x_t0_inline2836__ssa_v0, qr_d0_inline2839__ssa_v0], [16, 256], [qr_rows_inline2837__ssa_v0, 256], target_memory=pl.Mem.Mat)
                w_chunk_inline2834__tile: pl.Tile[[256, 128], pl.BF16, pl.MemRef(mem_mat_6, pl.const(32768, pl.INT64), 65536), pl.Mem.Mat] = pl.tile.load(
                    wq_a__ssa_v0, [qr_d0_inline2839__ssa_v0, q_a_col0_inline2848__ssa_v0], [256, 128], [256, 128], target_memory=pl.Mem.Mat
                )
                q_x_chunk_bf16_inline2835__tile_1: pl.Tile[
                    [16, 256], pl.BF16, pl.MemRef(mem_mat_7, pl.const(98304, pl.INT64), 8192), pl.Mem.Mat, pl.TileView(valid_shape=[qr_rows_inline2837__ssa_v0_1, 256])
                ] = pl.tile.load(x_view_inline2843__ssa_v0, [x_t0_inline2836__ssa_v0_1, qr_d0_inline2839__ssa_v0_1], [16, 256], [qr_rows_inline2837__ssa_v0_1, 256], target_memory=pl.Mem.Mat)
                w_chunk_inline2834__tile_1: pl.Tile[[256, 128], pl.BF16, pl.MemRef(mem_mat_8, pl.const(131072, pl.INT64), 65536), pl.Mem.Mat] = pl.tile.load(
                    wq_a__ssa_v0, [qr_d0_inline2839__ssa_v0_1, q_a_col0_inline2848__ssa_v0], [256, 128], [256, 128], target_memory=pl.Mem.Mat
                )
                q_acc_inline2841__tile_l0_a: pl.Tile[
                    [16, 128],
                    pl.BF16,
                    pl.MemRef(mem_left_9, pl.const(49152, pl.INT64), 4096),
                    pl.Mem.Left,
                    pl.TileView(valid_shape=[qr_rows_inline2837__ssa_v0, 128], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal),
                ] = pl.tile.extract(q_x_chunk_bf16_inline2835__tile, 0, 0, [16, 128], target_memory=pl.Mem.Left)
                q_acc_inline2841__tile_l0_b: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_10, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                    w_chunk_inline2834__tile, 0, 0, [128, 128], target_memory=pl.Mem.Right
                )
                q_acc_inline2841__tile_l0_a_1: pl.Tile[
                    [16, 128],
                    pl.BF16,
                    pl.MemRef(mem_left_11, pl.const(0, pl.INT64), 4096),
                    pl.Mem.Left,
                    pl.TileView(valid_shape=[qr_rows_inline2837__ssa_v0, 128], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal),
                ] = pl.tile.extract(q_x_chunk_bf16_inline2835__tile, 0, 128, [16, 128], target_memory=pl.Mem.Left)
                q_acc_inline2841__tile_l0_b_1: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_12, pl.const(32768, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                    w_chunk_inline2834__tile, 128, 0, [128, 128], target_memory=pl.Mem.Right
                )
                q_acc_inline2841__tile_l0_c_acc: pl.Tile[[16, 128], pl.FP32, pl.MemRef(mem_acc_4, pl.const(0, pl.INT64), 8192), pl.Mem.Acc] = pl.tile.matmul_acc(
                    q_acc_inline2841__iter_v1, q_acc_inline2841__tile_l0_a, q_acc_inline2841__tile_l0_b, db_inline2838__idx_v0 == 0
                )
                q_acc_inline2841__tile_l0_c_acc_1: pl.Tile[[16, 128], pl.FP32, pl.MemRef(mem_acc_4, pl.const(0, pl.INT64), 8192), pl.Mem.Acc] = pl.tile.matmul_acc(
                    q_acc_inline2841__tile_l0_c_acc, q_acc_inline2841__tile_l0_a_1, q_acc_inline2841__tile_l0_b_1, False
                )
                q_acc_inline2841__tile_l0_a_2: pl.Tile[
                    [16, 128],
                    pl.BF16,
                    pl.MemRef(mem_left_14, pl.const(16384, pl.INT64), 4096),
                    pl.Mem.Left,
                    pl.TileView(valid_shape=[qr_rows_inline2837__ssa_v0_1, 128], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal),
                ] = pl.tile.extract(q_x_chunk_bf16_inline2835__tile_1, 0, 0, [16, 128], target_memory=pl.Mem.Left)
                q_acc_inline2841__tile_l0_b_2: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_10, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                    w_chunk_inline2834__tile_1, 0, 0, [128, 128], target_memory=pl.Mem.Right
                )
                q_acc_inline2841__tile_l0_a_3: pl.Tile[
                    [16, 128],
                    pl.BF16,
                    pl.MemRef(mem_left_16, pl.const(32768, pl.INT64), 4096),
                    pl.Mem.Left,
                    pl.TileView(valid_shape=[qr_rows_inline2837__ssa_v0_1, 128], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal),
                ] = pl.tile.extract(q_x_chunk_bf16_inline2835__tile_1, 0, 128, [16, 128], target_memory=pl.Mem.Left)
                q_acc_inline2841__tile_l0_b_3: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_12, pl.const(32768, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                    w_chunk_inline2834__tile_1, 128, 0, [128, 128], target_memory=pl.Mem.Right
                )
                q_acc_inline2841__tile_l0_c_acc_2: pl.Tile[[16, 128], pl.FP32, pl.MemRef(mem_acc_4, pl.const(0, pl.INT64), 8192), pl.Mem.Acc] = pl.tile.matmul_acc(
                    q_acc_inline2841__tile_l0_c_acc_1, q_acc_inline2841__tile_l0_a_2, q_acc_inline2841__tile_l0_b_2, db_inline2838__idx_v0 == -1
                )
                q_acc_inline2841__tile_l0_c_acc_3: pl.Tile[[16, 128], pl.FP32, pl.MemRef(mem_acc_4, pl.const(0, pl.INT64), 8192), pl.Mem.Acc] = pl.tile.matmul_acc(
                    q_acc_inline2841__tile_l0_c_acc_2, q_acc_inline2841__tile_l0_a_3, q_acc_inline2841__tile_l0_b_3, False
                )
                q_acc_inline2841__rv_v2: pl.Tile[[16, 128], pl.FP32, pl.MemRef(mem_acc_4, pl.const(0, pl.INT64), 8192), pl.Mem.Acc] = pl.yield_(q_acc_inline2841__tile_l0_c_acc_3)
            qr_fp32_inline304_inline1875__tile_1: pl.Tensor[[qr_t_matmul_inline305_inline1881__ssa_v0, 1024], pl.FP32, pl.MemRef("mem_ddr_20", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                q_acc_inline2841__rv_v2, [t0_inline2840__idx_v0, q_a_col0_inline2848__ssa_v0], qr_fp32_inline304_inline1875__iter_v9, atomic=pl.AtomicType.Add
            )
            qr_fp32_inline304_inline1875__rv_v10: pl.Tensor[[qr_t_matmul_inline305_inline1881__ssa_v0, 1024], pl.FP32, pl.MemRef("mem_ddr_37", pl.const(0, pl.INT64), 0)] = pl.yield_(
                qr_fp32_inline304_inline1875__tile_1
            )
        return qr_fp32_inline304_inline1875__rv_v2

    @pl.function(type=pl.FunctionType.Spmd)
    def qr_proj_matmul_spmd(
        self,
        qr_fp32_inline304_inline1875__rv_v2: pl.InOut[pl.Tensor[[qr_t_matmul_inline305_inline1881__ssa_v0, 1024], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)]],
        qr_full_rows_inline2846__ssa_v0: pl.Scalar[pl.INDEX],
        tile_base_inline307_inline1878__idx_v0: pl.Scalar[pl.INDEX],
        x_view_inline2843__ssa_v0: pl.Tensor[[pl.tensor.dim(x_normed_t__ssa_v0, pl.const(0, pl.INDEX)), 4096], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)],
        wq_a__ssa_v0: pl.Tensor[[4096, 1024], pl.BF16, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 8388608)],
        qr_t_matmul_inline2845__ssa_v0: pl.Scalar[pl.INDEX],
        tile_rows_inline306_inline1880__ssa_v0: pl.Scalar[pl.INDEX],
    ) -> pl.Tensor[[qr_t_matmul_inline305_inline1881__ssa_v0, 1024], pl.FP32]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        qr_fp32_inline304_inline1875__rv_v10: pl.Tensor[[qr_t_matmul_inline305_inline1881__ssa_v0, 1024], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 0)] = self.qr_proj_matmul(
            qr_fp32_inline304_inline1875__rv_v2,
            qr_full_rows_inline2846__ssa_v0,
            tile_base_inline307_inline1878__idx_v0,
            x_view_inline2843__ssa_v0,
            wq_a__ssa_v0,
            qr_t_matmul_inline2845__ssa_v0,
            tile_rows_inline306_inline1880__ssa_v0,
            attrs={"arg_directions": [pl.adir.inout, pl.adir.scalar, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.scalar, pl.adir.scalar]},
        )
        return qr_fp32_inline304_inline1875__rv_v2

    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def qr_proj_seed(
        qr_fp32_inline304_inline1875__ssa_v0: pl.Out[pl.Tensor[[qr_t_matmul_inline305_inline1881__ssa_v0, 1024], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)]],
        qr_t_matmul_inline2845__ssa_v0: pl.Scalar[pl.INDEX],
    ) -> pl.Tensor[[qr_t_matmul_inline305_inline1881__ssa_v0, 1024], pl.FP32]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        mem_vec_1: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 8192)
        for ts0_inline2849__idx_v0, (qr_fp32_inline304_inline1875__iter_v1,) in pl.range(0, qr_t_matmul_inline2845__ssa_v0, 16, init_values=(qr_fp32_inline304_inline1875__ssa_v0,)):
            for nseed0_inline2853__idx_v0, (qr_fp32_inline304_inline1875__iter_v3,) in pl.range(0, 1024, 128, init_values=(qr_fp32_inline304_inline1875__iter_v1,)):
                qr_seed_inline2851__tile: pl.Tile[[16, 128], pl.FP32, pl.MemRef(mem_vec_1, pl.const(0, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.full([16, 128], dtype=pl.FP32, value=0.0)
                qr_fp32_inline304_inline1875__tile: pl.Tensor[[qr_t_matmul_inline305_inline1881__ssa_v0, 1024], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                    qr_seed_inline2851__tile, [ts0_inline2849__idx_v0, nseed0_inline2853__idx_v0], qr_fp32_inline304_inline1875__iter_v3
                )
                qr_fp32_inline304_inline1875__rv_v4: pl.Tensor[[qr_t_matmul_inline305_inline1881__ssa_v0, 1024], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)] = pl.yield_(
                    qr_fp32_inline304_inline1875__tile
                )
            qr_fp32_inline304_inline1875__rv_v2: pl.Tensor[[qr_t_matmul_inline305_inline1881__ssa_v0, 1024], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)] = pl.yield_(
                qr_fp32_inline304_inline1875__rv_v4
            )
        return qr_fp32_inline304_inline1875__ssa_v0

    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def qr_rms_norm_quant(
        tile_rows_inline306_inline1880__ssa_v0: pl.Scalar[pl.INDEX],
        tile_base_inline307_inline1878__idx_v0: pl.Scalar[pl.INDEX],
        qr_fp32_inline304_inline1875__rv_v10: pl.Tensor[[qr_t_matmul_inline305_inline1881__ssa_v0, 1024], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)],
        gamma_cq__ssa_v0: pl.Tensor[[1024], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 2048)],
        qr_scale_pad_store_inline1879__iter_v1: pl.InOut[pl.Tensor[[512, 1], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 2048)]],
        qr_scale_view_inline2896__ssa_v0: pl.Out[pl.Tensor[[t_dim_inline2888__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)]],
        qr_i8_matmul_inline1874__iter_v1: pl.InOut[pl.Tensor[[512, 1024], pl.INT8, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 524288)]],
        qr_view_inline2890__ssa_v0: pl.Out[pl.Tensor[[t_dim_inline2888__ssa_v0, 1024], pl.INT8, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 0)]],
    ) -> tuple[pl.Tensor[[512, 1], pl.FP32], pl.Tensor[[512, 1024], pl.INT8], pl.Tensor[[t_dim_inline2888__ssa_v0, 1], pl.FP32]]:
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
        tg_idx_inline2891__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        tg_inline2894__ssa_v0: pl.Scalar[pl.INDEX] = tg_idx_inline2891__ssa_v0 * 8
        valid_rows_inline2898__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(tile_rows_inline306_inline1880__ssa_v0 - tg_inline2894__ssa_v0, 8)
        out_tg_inline2902__ssa_v0: pl.Scalar[pl.INDEX] = tile_base_inline307_inline1878__idx_v0 + tg_inline2894__ssa_v0
        t__tile: pl.Tile[[8, 1024], pl.FP32, pl.MemRef(mem_vec_6, pl.const(17504, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.load(
            qr_fp32_inline304_inline1875__rv_v10, [tg_inline2894__ssa_v0, 0], [8, 1024], [8, 1024], target_memory=pl.Mem.Vec
        )
        t__tile_1: pl.Tile[[8, 1024], pl.BF16, pl.MemRef(mem_vec_7, pl.const(50272, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.cast(t__tile, target_type=pl.BF16, mode="rint")
        qr_rms_full_inline338_inline2866__tile: pl.Tile[[8, 1024], pl.FP32, pl.MemRef(mem_vec_6, pl.const(17504, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.cast(
            t__tile_1, target_type=pl.FP32, mode="round"
        )
        qr_square_full_inline334_inline2886__tile: pl.Tile[[8, 1024], pl.FP32, pl.MemRef(mem_vec_6, pl.const(17504, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.mul(
            qr_rms_full_inline338_inline2866__tile, qr_rms_full_inline338_inline2866__tile
        )
        t__tile_2: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_6, pl.const(17504, pl.INT64), 30720), pl.Mem.Vec] = pl.tile.slice(qr_square_full_inline334_inline2886__tile, [8, 512], [0, 0])
        t__tile_3: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_6, pl.const(19552, pl.INT64), 30720), pl.Mem.Vec] = pl.tile.slice(qr_square_full_inline334_inline2886__tile, [8, 512], [0, 512])
        qr_square_half_inline332_inline2903__tile: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_6, pl.const(17504, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.add(t__tile_2, t__tile_3)
        t__tile_4: pl.Tile[[8, 256], pl.FP32, pl.MemRef(mem_vec_6, pl.const(17504, pl.INT64), 15360), pl.Mem.Vec] = pl.tile.slice(qr_square_half_inline332_inline2903__tile, [8, 256], [0, 0])
        t__tile_5: pl.Tile[[8, 256], pl.FP32, pl.MemRef(mem_vec_6, pl.const(18528, pl.INT64), 15360), pl.Mem.Vec] = pl.tile.slice(qr_square_half_inline332_inline2903__tile, [8, 256], [0, 256])
        qr_square_quarter_inline329_inline2906__tile: pl.Tile[[8, 256], pl.FP32, pl.MemRef(mem_vec_6, pl.const(17504, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.add(t__tile_4, t__tile_5)
        t__tile_6: pl.Tile[[8, 128], pl.FP32, pl.MemRef(mem_vec_6, pl.const(17504, pl.INT64), 7680), pl.Mem.Vec] = pl.tile.slice(qr_square_quarter_inline329_inline2906__tile, [8, 128], [0, 0])
        t__tile_7: pl.Tile[[8, 128], pl.FP32, pl.MemRef(mem_vec_6, pl.const(18016, pl.INT64), 7680), pl.Mem.Vec] = pl.tile.slice(qr_square_quarter_inline329_inline2906__tile, [8, 128], [0, 128])
        qr_square_eighth_inline324_inline2867__tile: pl.Tile[[8, 128], pl.FP32, pl.MemRef(mem_vec_6, pl.const(17504, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.add(t__tile_6, t__tile_7)
        t__tile_8: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_6, pl.const(17504, pl.INT64), 3840), pl.Mem.Vec] = pl.tile.slice(qr_square_eighth_inline324_inline2867__tile, [8, 64], [0, 0])
        t__tile_9: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_6, pl.const(17760, pl.INT64), 3840), pl.Mem.Vec] = pl.tile.slice(qr_square_eighth_inline324_inline2867__tile, [8, 64], [0, 64])
        qr_square_sixteenth_inline320_inline2885__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_7, pl.const(50272, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.add(t__tile_8, t__tile_9)
        tmp_tile: pl.Tile[[8, 128], pl.FP32, pl.MemRef(mem_vec_6, pl.const(17504, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.create([8, 128], dtype=pl.FP32, target_memory=pl.Mem.Vec)
        t__tile_10: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_32, pl.const(1088, pl.INT64), 32), pl.Mem.Vec] = pl.tile.row_sum(qr_square_sixteenth_inline320_inline2885__tile, tmp_tile)
        qr_sq_sum_inline330_inline2878__tile: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_32, pl.const(1088, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(t__tile_10, [1, 8])
        t__tile_11: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_6, pl.const(17504, pl.INT64), 32), pl.Mem.Vec] = pl.tile.muls(qr_sq_sum_inline330_inline2878__tile, 0.0009765625)
        qr_variance_inline323_inline2881__tile: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_6, pl.const(17504, pl.INT64), 32), pl.Mem.Vec] = pl.tile.adds(t__tile_11, 9.9999999999999995e-07)
        qr_rms_approx_inline319_inline2887__tile: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_7, pl.const(50272, pl.INT64), 32), pl.Mem.Vec] = pl.tile.sqrt(qr_variance_inline323_inline2881__tile)
        qr_variance_bits_inline327_inline2899__tile: pl.Tile[[1, 8], pl.INT32, pl.MemRef(mem_vec_6, pl.const(17504, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reinterpret_view(
            qr_variance_inline323_inline2881__tile, dtype=pl.INT32
        )
        qr_approx_bits_inline318_inline2892__tile: pl.Tile[[1, 8], pl.INT32, pl.MemRef(mem_vec_7, pl.const(50272, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reinterpret_view(
            qr_rms_approx_inline319_inline2887__tile, dtype=pl.INT32
        )
        qr_rounded_bits_inline331_inline2870__tile: pl.Tile[[1, 8], pl.INT32, pl.MemRef(mem_vec_41, pl.const(9280, pl.INT64), 32), pl.Mem.Vec] = pl.tile.full([1, 8], dtype=pl.INT32, value=0)
        for qr_sqrt_row_inline325_inline2876__idx_v0 in pl.range(8):
            t__tile_12: pl.Scalar[pl.INT32] = pl.tile.read(qr_variance_bits_inline327_inline2899__tile, [0, qr_sqrt_row_inline325_inline2876__idx_v0])
            qr_x_bits_inline340_inline2889__ssa_v0: pl.Scalar[pl.INT64] = pl.cast(t__tile_12, pl.INT64)
            t__tile_13: pl.Scalar[pl.INT32] = pl.tile.read(qr_approx_bits_inline318_inline2892__tile, [0, qr_sqrt_row_inline325_inline2876__idx_v0])
            qr_y_bits_inline336_inline2872__ssa_v0: pl.Scalar[pl.INT64] = pl.cast(t__tile_13, pl.INT64)
            if 8388608 <= qr_x_bits_inline340_inline2889__ssa_v0 and qr_x_bits_inline340_inline2889__ssa_v0 < 2139095040:
                qr_x_exp_inline344_inline2915__ssa_v0: pl.Scalar[pl.INT64] = qr_x_bits_inline340_inline2889__ssa_v0 // 8388608
                qr_x_sig_inline341_inline2869__ssa_v0: pl.Scalar[pl.INT64] = qr_x_bits_inline340_inline2889__ssa_v0 % 8388608 + 8388608
                for _qr_sqrt_step_inline335_inline2893__idx_v0, (qr_y_bits_inline336_inline2872__iter_v1,) in pl.range(2, init_values=(qr_y_bits_inline336_inline2872__ssa_v0,)):
                    qr_y_exp_inline328_inline2865__ssa_v0: pl.Scalar[pl.INT64] = qr_y_bits_inline336_inline2872__iter_v1 // 8388608
                    qr_y_sig_inline345_inline2877__ssa_v0: pl.Scalar[pl.INT64] = qr_y_bits_inline336_inline2872__iter_v1 % 8388608 + 8388608
                    qr_upper_mid_inline326_inline2874__ssa_v0: pl.Scalar[pl.INDEX] = qr_y_sig_inline345_inline2877__ssa_v0 * 2 + 1
                    qr_upper_square_inline346_inline2875__ssa_v0: pl.Scalar[pl.INDEX] = qr_upper_mid_inline326_inline2874__ssa_v0 * qr_upper_mid_inline326_inline2874__ssa_v0
                    qr_x_upper_inline347_inline2900__ssa_v0: pl.Scalar[pl.INT64] = (
                        qr_x_sig_inline341_inline2869__ssa_v0 << qr_x_exp_inline344_inline2915__ssa_v0 - qr_y_exp_inline328_inline2865__ssa_v0 * 2 + 152
                    )
                    qr_previous_bits_inline343_inline2882__ssa_v0: pl.Scalar[pl.INT64] = qr_y_bits_inline336_inline2872__iter_v1 - 1
                    qr_previous_exp_inline337_inline2901__ssa_v0: pl.Scalar[pl.INT64] = qr_previous_bits_inline343_inline2882__ssa_v0 // 8388608
                    qr_previous_sig_inline322_inline2920__ssa_v0: pl.Scalar[pl.INT64] = qr_previous_bits_inline343_inline2882__ssa_v0 % 8388608 + 8388608
                    qr_lower_mid_inline342_inline2911__ssa_v0: pl.Scalar[pl.INDEX] = qr_previous_sig_inline322_inline2920__ssa_v0 * 2 + 1
                    qr_lower_square_inline333_inline2883__ssa_v0: pl.Scalar[pl.INDEX] = qr_lower_mid_inline342_inline2911__ssa_v0 * qr_lower_mid_inline342_inline2911__ssa_v0
                    qr_x_lower_inline348_inline2909__ssa_v0: pl.Scalar[pl.INT64] = (
                        qr_x_sig_inline341_inline2869__ssa_v0 << qr_x_exp_inline344_inline2915__ssa_v0 - qr_previous_exp_inline337_inline2901__ssa_v0 * 2 + 152
                    )
                    if (
                        qr_upper_square_inline346_inline2875__ssa_v0 < qr_x_upper_inline347_inline2900__ssa_v0
                        or qr_x_upper_inline347_inline2900__ssa_v0 == qr_upper_square_inline346_inline2875__ssa_v0
                        and qr_y_bits_inline336_inline2872__iter_v1 % 2 == 1
                    ):
                        qr_y_bits_inline336_inline2872__ssa_v3: pl.Scalar[pl.INT64] = qr_y_bits_inline336_inline2872__iter_v1 + 1
                        qr_y_bits_inline336_inline2872__phi_v6: pl.Scalar[pl.INT64] = pl.yield_(qr_y_bits_inline336_inline2872__ssa_v3)
                    else:
                        if (
                            qr_x_lower_inline348_inline2909__ssa_v0 < qr_lower_square_inline333_inline2883__ssa_v0
                            or qr_x_lower_inline348_inline2909__ssa_v0 == qr_lower_square_inline333_inline2883__ssa_v0
                            and qr_y_bits_inline336_inline2872__iter_v1 % 2 == 1
                        ):
                            qr_y_bits_inline336_inline2872__ssa_v4: pl.Scalar[pl.INT64] = qr_y_bits_inline336_inline2872__iter_v1 - 1
                            qr_y_bits_inline336_inline2872__phi_v5: pl.Scalar[pl.INT64] = pl.yield_(qr_y_bits_inline336_inline2872__ssa_v4)
                        else:
                            qr_y_bits_inline336_inline2872__phi_v5: pl.Scalar[pl.INT64] = pl.yield_(qr_y_bits_inline336_inline2872__iter_v1)
                        qr_y_bits_inline336_inline2872__phi_v6: pl.Scalar[pl.INT64] = pl.yield_(qr_y_bits_inline336_inline2872__phi_v5)
                    qr_y_bits_inline336_inline2872__rv_v2: pl.Scalar[pl.INT64] = pl.yield_(qr_y_bits_inline336_inline2872__phi_v6)
                pl.tile.write(qr_rounded_bits_inline331_inline2870__tile, [0, qr_sqrt_row_inline325_inline2876__idx_v0], pl.cast(qr_y_bits_inline336_inline2872__rv_v2, pl.INT32))
            else:
                pl.tile.write(qr_rounded_bits_inline331_inline2870__tile, [0, qr_sqrt_row_inline325_inline2876__idx_v0], pl.cast(qr_y_bits_inline336_inline2872__ssa_v0, pl.INT32))
        qr_rms_inline321_inline2871__tile: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_41, pl.const(9280, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reinterpret_view(
            qr_rounded_bits_inline331_inline2870__tile, dtype=pl.FP32
        )
        qr_inv_rms_inline317_inline2910__tile: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_20, pl.const(0, pl.INT64), 32), pl.Mem.Vec] = pl.tile.full([1, 8], dtype=pl.FP32, value=0.0)
        for qr_rms_row_inline316_inline2914__idx_v0 in pl.range(8):
            qr_rms_value_inline339_inline2907__tile: pl.Scalar[pl.FP32] = pl.tile.read(qr_rms_inline321_inline2871__tile, [0, qr_rms_row_inline316_inline2914__idx_v0])
            pl.tile.write(qr_inv_rms_inline317_inline2910__tile, [0, qr_rms_row_inline316_inline2914__idx_v0], 1.0 / qr_rms_value_inline339_inline2907__tile)
        _qr_sq_sum_inline2868__ssa_v0: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_32, pl.const(1088, pl.INT64), 32), pl.Mem.Vec] = qr_sq_sum_inline330_inline2878__tile
        qr_inv_rms_inline2895__ssa_v0: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_20, pl.const(0, pl.INT64), 32), pl.Mem.Vec] = qr_inv_rms_inline317_inline2910__tile
        _qr_rms_inline2926__ssa_v0: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_41, pl.const(9280, pl.INT64), 32), pl.Mem.Vec] = qr_rms_inline321_inline2871__tile
        qr_inv_rms_t_inline2904__tile: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_20, pl.const(0, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(qr_inv_rms_inline2895__ssa_v0, [8, 1])
        qr_tile_amax_inline2916__tile: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_21, pl.const(32, pl.INT64), 32), pl.Mem.Vec] = pl.tile.full([1, 8], dtype=pl.FP32, value=0.0001)
        for qr_max_col0_inline2912__idx_v0, (qr_tile_amax_inline2916__iter_v1,) in pl.range(0, 1024, 512, init_values=(qr_tile_amax_inline2916__tile,)):
            t__tile_14: pl.Tile[[8, 256], pl.FP32, pl.MemRef(mem_vec_6, pl.const(17504, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.load(
                qr_fp32_inline304_inline1875__rv_v10, [tg_inline2894__ssa_v0, qr_max_col0_inline2912__idx_v0], [8, 256], [8, 256], target_memory=pl.Mem.Vec
            )
            t__tile_15: pl.Tile[[256], pl.BF16, pl.MemRef(mem_vec_23, pl.const(64, pl.INT64), 512), pl.Mem.Vec] = pl.tile.load(
                gamma_cq__ssa_v0, [qr_max_col0_inline2912__idx_v0], [256], [256], target_memory=pl.Mem.Vec
            )
            t__tile_16: pl.Tile[[8, 256], pl.FP32, pl.MemRef(mem_vec_7, pl.const(50272, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.load(
                qr_fp32_inline304_inline1875__rv_v10, [tg_inline2894__ssa_v0, qr_max_col0_inline2912__idx_v0 + 256], [8, 256], [8, 256], target_memory=pl.Mem.Vec
            )
            t__tile_17: pl.Tile[[256], pl.BF16, pl.MemRef(mem_vec_25, pl.const(576, pl.INT64), 512), pl.Mem.Vec] = pl.tile.load(
                gamma_cq__ssa_v0, [qr_max_col0_inline2912__idx_v0 + 256], [256], [256], target_memory=pl.Mem.Vec
            )
            t__tile_18: pl.Tile[[8, 256], pl.BF16, pl.MemRef(mem_vec_32, pl.const(1088, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(t__tile_14, target_type=pl.BF16, mode="rint")
            qr_max_chunk_inline2897__tile: pl.Tile[[8, 256], pl.FP32, pl.MemRef(mem_vec_6, pl.const(17504, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.cast(t__tile_18, target_type=pl.FP32, mode="round")
            gamma_max_cast_inline2873__tile: pl.Tile[[256], pl.FP32, pl.MemRef(mem_vec_32, pl.const(1088, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.cast(t__tile_15, target_type=pl.FP32, mode="round")
            gamma_max_chunk_inline2917__tile: pl.Tile[[1, 256], pl.FP32, pl.MemRef(mem_vec_32, pl.const(1088, pl.INT64), 1024), pl.Mem.Vec] = gamma_max_cast_inline2873__tile
            t__tile_19: pl.Tile[[8, 256], pl.FP32, pl.MemRef(mem_vec_6, pl.const(17504, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.row_expand_mul(
                qr_max_chunk_inline2897__tile, qr_inv_rms_t_inline2904__tile
            )
            qr_normalized_inline2918__tile: pl.Tile[[8, 256], pl.FP32, pl.MemRef(mem_vec_6, pl.const(17504, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.col_expand_mul(
                t__tile_19, gamma_max_chunk_inline2917__tile
            )
            t__tile_20: pl.Tile[[8, 256], pl.FP32, pl.MemRef(mem_vec_6, pl.const(17504, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.abs(qr_normalized_inline2918__tile)
            tmp_tile_1: pl.Tile[[8, 256], pl.FP32, pl.MemRef(mem_vec_32, pl.const(1088, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.create([8, 256], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            t__tile_21: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_23, pl.const(64, pl.INT64), 32), pl.Mem.Vec] = pl.tile.row_max(t__tile_20, tmp_tile_1)
            qr_normalized_max_inline2919__tile: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_23, pl.const(64, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(t__tile_21, [1, 8])
            qr_tile_amax_inline2916__tile_1: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_6, pl.const(17504, pl.INT64), 32), pl.Mem.Vec] = pl.tile.maximum(
                qr_tile_amax_inline2916__iter_v1, qr_normalized_max_inline2919__tile
            )
            t__tile_22: pl.Tile[[8, 256], pl.BF16, pl.MemRef(mem_vec_41, pl.const(9280, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(t__tile_16, target_type=pl.BF16, mode="rint")
            qr_max_chunk_inline2897__tile_1: pl.Tile[[8, 256], pl.FP32, pl.MemRef(mem_vec_7, pl.const(50272, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.cast(t__tile_22, target_type=pl.FP32, mode="round")
            gamma_max_cast_inline2873__tile_1: pl.Tile[[256], pl.FP32, pl.MemRef(mem_vec_41, pl.const(9280, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.cast(t__tile_17, target_type=pl.FP32, mode="round")
            gamma_max_chunk_inline2917__tile_1: pl.Tile[[1, 256], pl.FP32, pl.MemRef(mem_vec_41, pl.const(9280, pl.INT64), 1024), pl.Mem.Vec] = gamma_max_cast_inline2873__tile_1
            t__tile_23: pl.Tile[[8, 256], pl.FP32, pl.MemRef(mem_vec_7, pl.const(50272, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.row_expand_mul(
                qr_max_chunk_inline2897__tile_1, qr_inv_rms_t_inline2904__tile
            )
            qr_normalized_inline2918__tile_1: pl.Tile[[8, 256], pl.FP32, pl.MemRef(mem_vec_7, pl.const(50272, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.col_expand_mul(
                t__tile_23, gamma_max_chunk_inline2917__tile_1
            )
            t__tile_24: pl.Tile[[8, 256], pl.FP32, pl.MemRef(mem_vec_7, pl.const(50272, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.abs(qr_normalized_inline2918__tile_1)
            tmp_tile_2: pl.Tile[[8, 256], pl.FP32, pl.MemRef(mem_vec_41, pl.const(9280, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.create([8, 256], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            t__tile_25: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_25, pl.const(576, pl.INT64), 32), pl.Mem.Vec] = pl.tile.row_max(t__tile_24, tmp_tile_2)
            qr_normalized_max_inline2919__tile_1: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_25, pl.const(576, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(t__tile_25, [1, 8])
            qr_tile_amax_inline2916__tile_2: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_21, pl.const(32, pl.INT64), 32), pl.Mem.Vec] = pl.tile.maximum(
                qr_tile_amax_inline2916__tile_1, qr_normalized_max_inline2919__tile_1
            )
            qr_tile_amax_inline2916__rv_v2: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_21, pl.const(32, pl.INT64), 32), pl.Mem.Vec] = pl.yield_(qr_tile_amax_inline2916__tile_2)
        qr_scale_quant_row_inline2921__tile: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_45, pl.const(17472, pl.INT64), 32), pl.Mem.Vec] = pl.tile.full([1, 8], dtype=pl.FP32, value=0.0)
        qr_scale_dq_row_inline2923__tile: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_6, pl.const(17504, pl.INT64), 32), pl.Mem.Vec] = pl.tile.full([1, 8], dtype=pl.FP32, value=0.0)
        for qr_scale_row_inline2924__idx_v0 in pl.range(8):
            qr_amax_value_inline2880__tile: pl.Scalar[pl.FP32] = pl.tile.read(qr_tile_amax_inline2916__rv_v2, [0, qr_scale_row_inline2924__idx_v0])
            qr_quant_value_inline2913__ssa_v0: pl.Scalar[pl.FP32] = 127.0 / qr_amax_value_inline2880__tile
            pl.tile.write(qr_scale_quant_row_inline2921__tile, [0, qr_scale_row_inline2924__idx_v0], qr_quant_value_inline2913__ssa_v0)
            pl.tile.write(qr_scale_dq_row_inline2923__tile, [0, qr_scale_row_inline2924__idx_v0], 1.0 / qr_quant_value_inline2913__ssa_v0)
        qr_scale_quant_t_inline2927__tile: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_45, pl.const(17472, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(qr_scale_quant_row_inline2921__tile, [8, 1])
        qr_tile_scale_dq_inline2922__tile: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_6, pl.const(17504, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(qr_scale_dq_row_inline2923__tile, [8, 1])
        qr_scale_pad_store_inline1879__tile: pl.Tensor[[512, 1], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 2048)] = pl.tile.store(
            qr_tile_scale_dq_inline2922__tile, [tg_inline2894__ssa_v0, 0], qr_scale_pad_store_inline1879__iter_v1
        )
        if valid_rows_inline2898__ssa_v0 == 8:
            qr_scale_view_inline2896__tile: pl.Tensor[[t_dim_inline2888__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                qr_tile_scale_dq_inline2922__tile, [out_tg_inline2902__ssa_v0, 0], qr_scale_view_inline2896__ssa_v0
            )
        else:
            qr_scale_tail_inline2864__ssa_v0: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_6, pl.const(17504, pl.INT64), 32), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline2898__ssa_v0, 1])] = (
                pl.tile.load(qr_scale_pad_store_inline1879__tile, [tg_inline2894__ssa_v0, 0], [8, 1], [valid_rows_inline2898__ssa_v0, 1], target_memory=pl.Mem.Vec)
            )
            qr_scale_view_inline2896__store: pl.Tensor[[t_dim_inline2888__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                qr_scale_tail_inline2864__ssa_v0, [out_tg_inline2902__ssa_v0, 0], qr_scale_view_inline2896__ssa_v0
            )
        for qa_inline2863__idx_v0, (qr_i8_matmul_inline1874__iter_v3, qr_view_inline2890__iter_v1) in pl.range(
            0, 1024, 512, init_values=(qr_i8_matmul_inline1874__iter_v1, qr_view_inline2890__ssa_v0)
        ):
            t__tile_26: pl.Tile[[8, 256], pl.FP32, pl.MemRef(mem_vec_6, pl.const(17504, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.load(
                qr_fp32_inline304_inline1875__rv_v10, [tg_inline2894__ssa_v0, qa_inline2863__idx_v0], [8, 256], [8, 256], target_memory=pl.Mem.Vec
            )
            t__tile_27: pl.Tile[[256], pl.BF16, pl.MemRef(mem_vec_23, pl.const(64, pl.INT64), 512), pl.Mem.Vec] = pl.tile.load(
                gamma_cq__ssa_v0, [qa_inline2863__idx_v0], [256], [256], target_memory=pl.Mem.Vec
            )
            t__tile_28: pl.Tile[[8, 256], pl.FP32, pl.MemRef(mem_vec_7, pl.const(50272, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.load(
                qr_fp32_inline304_inline1875__rv_v10, [tg_inline2894__ssa_v0, qa_inline2863__idx_v0 + 256], [8, 256], [8, 256], target_memory=pl.Mem.Vec
            )
            t__tile_29: pl.Tile[[256], pl.BF16, pl.MemRef(mem_vec_25, pl.const(576, pl.INT64), 512), pl.Mem.Vec] = pl.tile.load(
                gamma_cq__ssa_v0, [qa_inline2863__idx_v0 + 256], [256], [256], target_memory=pl.Mem.Vec
            )
            t__tile_30: pl.Tile[[8, 256], pl.BF16, pl.MemRef(mem_vec_32, pl.const(1088, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(t__tile_26, target_type=pl.BF16, mode="rint")
            qr_chunk_inline2862__tile: pl.Tile[[8, 256], pl.FP32, pl.MemRef(mem_vec_6, pl.const(17504, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.cast(t__tile_30, target_type=pl.FP32, mode="round")
            gamma_q_cast_inline2884__tile: pl.Tile[[256], pl.FP32, pl.MemRef(mem_vec_32, pl.const(1088, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.cast(t__tile_27, target_type=pl.FP32, mode="round")
            gamma_q_chunk_inline2861__tile: pl.Tile[[1, 256], pl.FP32, pl.MemRef(mem_vec_32, pl.const(1088, pl.INT64), 1024), pl.Mem.Vec] = gamma_q_cast_inline2884__tile
            t__tile_31: pl.Tile[[8, 256], pl.FP32, pl.MemRef(mem_vec_6, pl.const(17504, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.row_expand_mul(qr_chunk_inline2862__tile, qr_inv_rms_t_inline2904__tile)
            qr_q_normed_inline2860__tile: pl.Tile[[8, 256], pl.FP32, pl.MemRef(mem_vec_6, pl.const(17504, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.col_expand_mul(
                t__tile_31, gamma_q_chunk_inline2861__tile
            )
            qr_q_scaled_inline2859__tile: pl.Tile[[8, 256], pl.FP32, pl.MemRef(mem_vec_6, pl.const(17504, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.row_expand_mul(
                qr_q_normed_inline2860__tile, qr_scale_quant_t_inline2927__tile
            )
            qr_q_i32_inline2925__tile: pl.Tile[[8, 256], pl.INT32, pl.MemRef(mem_vec_6, pl.const(17504, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.cast(
                qr_q_scaled_inline2859__tile, target_type=pl.INT32, mode="rint"
            )
            qr_q_half_inline2858__tile: pl.Tile[[8, 256], pl.FP16, pl.MemRef(mem_vec_6, pl.const(17504, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(
                qr_q_i32_inline2925__tile, target_type=pl.FP16, mode="round"
            )
            qr_q_i8_inline2905__tile: pl.Tile[[8, 256], pl.INT8, pl.MemRef(mem_vec_6, pl.const(17504, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(
                qr_q_half_inline2858__tile, target_type=pl.INT8, mode="trunc"
            )
            qr_i8_matmul_inline1874__tile: pl.Tensor[[512, 1024], pl.INT8, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 524288)] = pl.tile.store(
                qr_q_i8_inline2905__tile, [tg_inline2894__ssa_v0, qa_inline2863__idx_v0], qr_i8_matmul_inline1874__iter_v3
            )
            if valid_rows_inline2898__ssa_v0 == 8:
                qr_view_inline2890__tile: pl.Tensor[[t_dim_inline2888__ssa_v0, 1024], pl.INT8, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                    qr_q_i8_inline2905__tile, [out_tg_inline2902__ssa_v0, qa_inline2863__idx_v0], qr_view_inline2890__iter_v1
                )
                qr_view_inline2890__phi_v4: pl.Tensor[[t_dim_inline2888__ssa_v0, 1024], pl.INT8, pl.MemRef("mem_ddr_62", pl.const(0, pl.INT64), 0)] = pl.yield_(qr_view_inline2890__tile)
            else:
                qr_q_tail_inline2908__ssa_v0: pl.Tile[
                    [8, 256], pl.INT8, pl.MemRef(mem_vec_6, pl.const(17504, pl.INT64), 2048), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline2898__ssa_v0, 256])
                ] = pl.tile.load(qr_i8_matmul_inline1874__tile, [tg_inline2894__ssa_v0, qa_inline2863__idx_v0], [8, 256], [valid_rows_inline2898__ssa_v0, 256], target_memory=pl.Mem.Vec)
                pl.tile.store(qr_q_tail_inline2908__ssa_v0, [out_tg_inline2902__ssa_v0, qa_inline2863__idx_v0], qr_view_inline2890__iter_v1)
                qr_view_inline2890__phi_v4: pl.Tensor[[t_dim_inline2888__ssa_v0, 1024], pl.INT8, pl.MemRef("mem_ddr_62", pl.const(0, pl.INT64), 0)] = pl.yield_(qr_view_inline2890__iter_v1)
            t__tile_32: pl.Tile[[8, 256], pl.BF16, pl.MemRef(mem_vec_41, pl.const(9280, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(t__tile_28, target_type=pl.BF16, mode="rint")
            qr_chunk_inline2862__tile_1: pl.Tile[[8, 256], pl.FP32, pl.MemRef(mem_vec_7, pl.const(50272, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.cast(t__tile_32, target_type=pl.FP32, mode="round")
            gamma_q_cast_inline2884__tile_1: pl.Tile[[256], pl.FP32, pl.MemRef(mem_vec_41, pl.const(9280, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.cast(t__tile_29, target_type=pl.FP32, mode="round")
            gamma_q_chunk_inline2861__tile_1: pl.Tile[[1, 256], pl.FP32, pl.MemRef(mem_vec_41, pl.const(9280, pl.INT64), 1024), pl.Mem.Vec] = gamma_q_cast_inline2884__tile_1
            t__tile_33: pl.Tile[[8, 256], pl.FP32, pl.MemRef(mem_vec_7, pl.const(50272, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.row_expand_mul(
                qr_chunk_inline2862__tile_1, qr_inv_rms_t_inline2904__tile
            )
            qr_q_normed_inline2860__tile_1: pl.Tile[[8, 256], pl.FP32, pl.MemRef(mem_vec_7, pl.const(50272, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.col_expand_mul(
                t__tile_33, gamma_q_chunk_inline2861__tile_1
            )
            qr_q_scaled_inline2859__tile_1: pl.Tile[[8, 256], pl.FP32, pl.MemRef(mem_vec_7, pl.const(50272, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.row_expand_mul(
                qr_q_normed_inline2860__tile_1, qr_scale_quant_t_inline2927__tile
            )
            qr_q_i32_inline2925__tile_1: pl.Tile[[8, 256], pl.INT32, pl.MemRef(mem_vec_7, pl.const(50272, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.cast(
                qr_q_scaled_inline2859__tile_1, target_type=pl.INT32, mode="rint"
            )
            qr_q_half_inline2858__tile_1: pl.Tile[[8, 256], pl.FP16, pl.MemRef(mem_vec_7, pl.const(50272, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(
                qr_q_i32_inline2925__tile_1, target_type=pl.FP16, mode="round"
            )
            qr_q_i8_inline2905__tile_1: pl.Tile[[8, 256], pl.INT8, pl.MemRef(mem_vec_7, pl.const(50272, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(
                qr_q_half_inline2858__tile_1, target_type=pl.INT8, mode="trunc"
            )
            qr_i8_matmul_inline1874__tile_1: pl.Tensor[[512, 1024], pl.INT8, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 524288)] = pl.tile.store(
                qr_q_i8_inline2905__tile_1, [tg_inline2894__ssa_v0, qa_inline2863__idx_v0 + 256], qr_i8_matmul_inline1874__tile
            )
            if valid_rows_inline2898__ssa_v0 == 8:
                qr_view_inline2890__tile_1: pl.Tensor[[t_dim_inline2888__ssa_v0, 1024], pl.INT8, pl.MemRef("mem_ddr_62", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                    qr_q_i8_inline2905__tile_1, [out_tg_inline2902__ssa_v0, qa_inline2863__idx_v0 + 256], qr_view_inline2890__phi_v4
                )
                qr_view_inline2890__phi_v4_1: pl.Tensor[[t_dim_inline2888__ssa_v0, 1024], pl.INT8, pl.MemRef("mem_ddr_73", pl.const(0, pl.INT64), 0)] = pl.yield_(qr_view_inline2890__tile_1)
            else:
                qr_q_tail_inline2908__ssa_v0_1: pl.Tile[
                    [8, 256], pl.INT8, pl.MemRef(mem_vec_7, pl.const(50272, pl.INT64), 2048), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline2898__ssa_v0, 256])
                ] = pl.tile.load(qr_i8_matmul_inline1874__tile_1, [tg_inline2894__ssa_v0, qa_inline2863__idx_v0 + 256], [8, 256], [valid_rows_inline2898__ssa_v0, 256], target_memory=pl.Mem.Vec)
                pl.tile.store(qr_q_tail_inline2908__ssa_v0_1, [out_tg_inline2902__ssa_v0, qa_inline2863__idx_v0 + 256], qr_view_inline2890__phi_v4)
                qr_view_inline2890__phi_v4_1: pl.Tensor[[t_dim_inline2888__ssa_v0, 1024], pl.INT8, pl.MemRef("mem_ddr_73", pl.const(0, pl.INT64), 0)] = pl.yield_(qr_view_inline2890__phi_v4)
            qr_i8_matmul_inline1874__rv_v4, qr_view_inline2890__rv_v2 = pl.yield_(qr_i8_matmul_inline1874__tile_1, qr_view_inline2890__phi_v4_1)
        return qr_scale_pad_store_inline1879__iter_v1, qr_i8_matmul_inline1874__iter_v1, qr_scale_view_inline2896__ssa_v0

    @pl.function(type=pl.FunctionType.Spmd)
    def qr_rms_norm_quant_spmd(
        self,
        tile_rows_inline306_inline1880__ssa_v0: pl.Scalar[pl.INDEX],
        tile_base_inline307_inline1878__idx_v0: pl.Scalar[pl.INDEX],
        qr_fp32_inline304_inline1875__rv_v10: pl.Tensor[[qr_t_matmul_inline305_inline1881__ssa_v0, 1024], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)],
        gamma_cq__ssa_v0: pl.Tensor[[1024], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 2048)],
        qr_scale_pad_store_inline1879__iter_v1: pl.InOut[pl.Tensor[[512, 1], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 2048)]],
        qr_scale_view_inline2896__ssa_v0: pl.Out[pl.Tensor[[t_dim_inline2888__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)]],
        qr_i8_matmul_inline1874__iter_v1: pl.InOut[pl.Tensor[[512, 1024], pl.INT8, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 524288)]],
        qr_view_inline2890__ssa_v0: pl.Out[pl.Tensor[[t_dim_inline2888__ssa_v0, 1024], pl.INT8, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 0)]],
    ) -> tuple[pl.Tensor[[512, 1], pl.FP32], pl.Tensor[[512, 1024], pl.INT8]]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        ret__tmp_v0: pl.Tuple[pl.Tensor[[512, 1], pl.FP32], pl.Tensor[[512, 1024], pl.INT8], pl.Tensor[[t_dim_inline2888__ssa_v0, 1], pl.FP32]] = self.qr_rms_norm_quant(
            tile_rows_inline306_inline1880__ssa_v0,
            tile_base_inline307_inline1878__idx_v0,
            qr_fp32_inline304_inline1875__rv_v10,
            gamma_cq__ssa_v0,
            qr_scale_pad_store_inline1879__iter_v1,
            qr_scale_view_inline2896__ssa_v0,
            qr_i8_matmul_inline1874__iter_v1,
            qr_view_inline2890__ssa_v0,
            attrs={"arg_directions": [pl.adir.scalar, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.inout, pl.adir.output_existing, pl.adir.inout, pl.adir.output_existing]},
        )
        qr_scale_pad_store_inline1879__ssa_v3: pl.Tensor[[512, 1], pl.FP32, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 2048)] = ret__tmp_v0[0]
        qr_i8_matmul_inline1874__rv_v4: pl.Tensor[[512, 1024], pl.INT8, pl.MemRef("mem_ddr_7", pl.const(0, pl.INT64), 524288)] = ret__tmp_v0[1]
        qr_scale_view_inline2896__ssa_v2: pl.Tensor[[t_dim_inline2888__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_8", pl.const(0, pl.INT64), 0)] = ret__tmp_v0[2]
        return qr_scale_pad_store_inline1879__iter_v1, qr_i8_matmul_inline1874__iter_v1

    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def quant(
        o_r_i8_pad_inline667__iter_v1: pl.Out[pl.Tensor[[384, 8192], pl.INT8, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 3145728)]],
        t_dim_inline652__ssa_v0: pl.Scalar[pl.INDEX],
        act_scale_dq_inline694__rv_v2: pl.Tensor[[1, 384], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 1536)],
        o_r_pad_inline675__rv_v2: pl.Tensor[[384, 8192], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 12582912)],
        col_g_inline712__ssa_v0: pl.Scalar[pl.INDEX],
        proj_b_padded_rows_inline669__ssa_v0: pl.Scalar[pl.INDEX],
    ) -> pl.Tensor[[384, 8192], pl.INT8]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        mem_vec_4: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32768)
        mem_vec_6: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32768)
        mem_vec_7: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32)
        mem_vec_8: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
        mem_vec_14: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32)
        mem_vec_15: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
        unroll_main_end: pl.Scalar[pl.INDEX] = (t_dim_inline652__ssa_v0 + 7) // 16 * 16
        for qt_inline713__idx_v0, (o_r_i8_pad_inline667__iter_v3,) in pl.range(0, unroll_main_end, 16, init_values=(o_r_i8_pad_inline667__iter_v1,)):
            token_scale_inline668__tile: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_8, pl.const(81984, pl.INT64), 32), pl.Mem.Vec] = pl.tile.load(
                act_scale_dq_inline694__rv_v2, [0, qt_inline713__idx_v0], [1, 8], [1, 8], target_memory=pl.Mem.Vec
            )
            oc_q_inline670__tile: pl.Tile[[8, 1024], pl.FP32, pl.MemRef(mem_vec_4, pl.const(16416, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.load(
                o_r_pad_inline675__rv_v2, [qt_inline713__idx_v0, col_g_inline712__ssa_v0], [8, 1024], [8, 1024], target_memory=pl.Mem.Vec
            )
            token_scale_inline668__tile_1: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_15, pl.const(32, pl.INT64), 32), pl.Mem.Vec] = pl.tile.load(
                act_scale_dq_inline694__rv_v2, [0, qt_inline713__idx_v0 + 8], [1, 8], [1, 8], target_memory=pl.Mem.Vec
            )
            oc_q_inline670__tile_1: pl.Tile[[8, 1024], pl.FP32, pl.MemRef(mem_vec_6, pl.const(49184, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.load(
                o_r_pad_inline675__rv_v2, [qt_inline713__idx_v0 + 8, col_g_inline712__ssa_v0], [8, 1024], [8, 1024], target_memory=pl.Mem.Vec
            )
            t__tile: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_7, pl.const(81952, pl.INT64), 32), pl.Mem.Vec] = pl.tile.recip(token_scale_inline668__tile)
            g_sq_col_inline697__tile: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_7, pl.const(81952, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(t__tile, [8, 1])
            t__tile_1: pl.Tile[[8, 1024], pl.BF16, pl.MemRef(mem_vec_8, pl.const(81984, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.cast(oc_q_inline670__tile, target_type=pl.BF16, mode="rint")
            oc_q_v1_inline685__tile: pl.Tile[[8, 1024], pl.FP32, pl.MemRef(mem_vec_4, pl.const(16416, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.cast(t__tile_1, target_type=pl.FP32, mode="round")
            oq_scaled_inline688__tile: pl.Tile[[8, 1024], pl.FP32, pl.MemRef(mem_vec_4, pl.const(16416, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.row_expand_mul(
                oc_q_v1_inline685__tile, g_sq_col_inline697__tile
            )
            oq_i32_inline706__tile: pl.Tile[[8, 1024], pl.INT32, pl.MemRef(mem_vec_4, pl.const(16416, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.cast(
                oq_scaled_inline688__tile, target_type=pl.INT32, mode="rint"
            )
            oq_half_inline683__tile: pl.Tile[[8, 1024], pl.FP16, pl.MemRef(mem_vec_4, pl.const(16416, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.cast(
                oq_i32_inline706__tile, target_type=pl.FP16, mode="round"
            )
            oq_i8_inline661__tile: pl.Tile[[8, 1024], pl.INT8, pl.MemRef(mem_vec_4, pl.const(16416, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.cast(
                oq_half_inline683__tile, target_type=pl.INT8, mode="trunc"
            )
            o_r_i8_pad_inline667__tile: pl.Tensor[[384, 8192], pl.INT8, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 3145728)] = pl.tile.store(
                oq_i8_inline661__tile, [qt_inline713__idx_v0, col_g_inline712__ssa_v0], o_r_i8_pad_inline667__iter_v3
            )
            t__tile_2: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_14, pl.const(0, pl.INT64), 32), pl.Mem.Vec] = pl.tile.recip(token_scale_inline668__tile_1)
            g_sq_col_inline697__tile_1: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_14, pl.const(0, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(t__tile_2, [8, 1])
            t__tile_3: pl.Tile[[8, 1024], pl.BF16, pl.MemRef(mem_vec_15, pl.const(32, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.cast(oc_q_inline670__tile_1, target_type=pl.BF16, mode="rint")
            oc_q_v1_inline685__tile_1: pl.Tile[[8, 1024], pl.FP32, pl.MemRef(mem_vec_6, pl.const(49184, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.cast(t__tile_3, target_type=pl.FP32, mode="round")
            oq_scaled_inline688__tile_1: pl.Tile[[8, 1024], pl.FP32, pl.MemRef(mem_vec_6, pl.const(49184, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.row_expand_mul(
                oc_q_v1_inline685__tile_1, g_sq_col_inline697__tile_1
            )
            oq_i32_inline706__tile_1: pl.Tile[[8, 1024], pl.INT32, pl.MemRef(mem_vec_6, pl.const(49184, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.cast(
                oq_scaled_inline688__tile_1, target_type=pl.INT32, mode="rint"
            )
            oq_half_inline683__tile_1: pl.Tile[[8, 1024], pl.FP16, pl.MemRef(mem_vec_6, pl.const(49184, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.cast(
                oq_i32_inline706__tile_1, target_type=pl.FP16, mode="round"
            )
            oq_i8_inline661__tile_1: pl.Tile[[8, 1024], pl.INT8, pl.MemRef(mem_vec_6, pl.const(49184, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.cast(
                oq_half_inline683__tile_1, target_type=pl.INT8, mode="trunc"
            )
            o_r_i8_pad_inline667__tile_1: pl.Tensor[[384, 8192], pl.INT8, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 3145728)] = pl.tile.store(
                oq_i8_inline661__tile_1, [qt_inline713__idx_v0 + 8, col_g_inline712__ssa_v0], o_r_i8_pad_inline667__tile
            )
            o_r_i8_pad_inline667__rv_v4_main: pl.Tensor[[384, 8192], pl.INT8, pl.MemRef("mem_ddr_21", pl.const(0, pl.INT64), 3145728)] = pl.yield_(o_r_i8_pad_inline667__tile_1)
        unroll_rem: pl.Scalar[pl.INDEX] = (t_dim_inline652__ssa_v0 + 7) // 8 - (t_dim_inline652__ssa_v0 + 7) // 16 * 2
        if unroll_rem == 1:
            token_scale_inline668__tile_2: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_4, pl.const(16416, pl.INT64), 32), pl.Mem.Vec] = pl.tile.load(
                act_scale_dq_inline694__rv_v2, [0, unroll_main_end], [1, 8], [1, 8], target_memory=pl.Mem.Vec
            )
            t__tile_4: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_8, pl.const(81984, pl.INT64), 32), pl.Mem.Vec] = pl.tile.recip(token_scale_inline668__tile_2)
            g_sq_col_inline697__tile_2: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_8, pl.const(81984, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(t__tile_4, [8, 1])
            oc_q_inline670__tile_2: pl.Tile[[8, 1024], pl.FP32, pl.MemRef(mem_vec_4, pl.const(16416, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.load(
                o_r_pad_inline675__rv_v2, [unroll_main_end, col_g_inline712__ssa_v0], [8, 1024], [8, 1024], target_memory=pl.Mem.Vec
            )
            t__tile_5: pl.Tile[[8, 1024], pl.BF16, pl.MemRef(mem_vec_6, pl.const(49184, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.cast(oc_q_inline670__tile_2, target_type=pl.BF16, mode="rint")
            oc_q_v1_inline685__tile_2: pl.Tile[[8, 1024], pl.FP32, pl.MemRef(mem_vec_4, pl.const(16416, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.cast(t__tile_5, target_type=pl.FP32, mode="round")
            oq_scaled_inline688__tile_2: pl.Tile[[8, 1024], pl.FP32, pl.MemRef(mem_vec_4, pl.const(16416, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.row_expand_mul(
                oc_q_v1_inline685__tile_2, g_sq_col_inline697__tile_2
            )
            oq_i32_inline706__tile_2: pl.Tile[[8, 1024], pl.INT32, pl.MemRef(mem_vec_4, pl.const(16416, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.cast(
                oq_scaled_inline688__tile_2, target_type=pl.INT32, mode="rint"
            )
            oq_half_inline683__tile_2: pl.Tile[[8, 1024], pl.FP16, pl.MemRef(mem_vec_4, pl.const(16416, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.cast(
                oq_i32_inline706__tile_2, target_type=pl.FP16, mode="round"
            )
            oq_i8_inline661__tile_2: pl.Tile[[8, 1024], pl.INT8, pl.MemRef(mem_vec_4, pl.const(16416, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.cast(
                oq_half_inline683__tile_2, target_type=pl.INT8, mode="trunc"
            )
            o_r_i8_pad_inline667__tile_2: pl.Tensor[[384, 8192], pl.INT8, pl.MemRef("mem_ddr_21", pl.const(0, pl.INT64), 3145728)] = pl.tile.store(
                oq_i8_inline661__tile_2, [unroll_main_end, col_g_inline712__ssa_v0], o_r_i8_pad_inline667__rv_v4_main
            )
            o_r_i8_pad_inline667__rv_v4: pl.Tensor[[384, 8192], pl.INT8, pl.MemRef("mem_ddr_31", pl.const(0, pl.INT64), 3145728)] = pl.yield_(o_r_i8_pad_inline667__tile_2)
        else:
            o_r_i8_pad_inline667__rv_v4: pl.Tensor[[384, 8192], pl.INT8, pl.MemRef("mem_ddr_31", pl.const(0, pl.INT64), 3145728)] = pl.yield_(o_r_i8_pad_inline667__rv_v4_main)
        for zt_inline714__idx_v0, (o_r_i8_pad_inline667__iter_v6,) in pl.range(t_dim_inline652__ssa_v0, proj_b_padded_rows_inline669__ssa_v0, 8, init_values=(o_r_i8_pad_inline667__rv_v4,)):
            zero_half_inline709__tile: pl.Tile[[8, 1024], pl.FP16, pl.MemRef(mem_vec_4, pl.const(16416, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.full([8, 1024], dtype=pl.FP16, value=0.0)
            t__tile_6: pl.Tile[[8, 1024], pl.INT8, pl.MemRef(mem_vec_4, pl.const(16416, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.cast(zero_half_inline709__tile, target_type=pl.INT8, mode="trunc")
            o_r_i8_pad_inline667__tile_3: pl.Tensor[[384, 8192], pl.INT8, pl.MemRef("mem_ddr_31", pl.const(0, pl.INT64), 3145728)] = pl.tile.store(
                t__tile_6, [zt_inline714__idx_v0, col_g_inline712__ssa_v0], o_r_i8_pad_inline667__iter_v6
            )
            o_r_i8_pad_inline667__rv_v7: pl.Tensor[[384, 8192], pl.INT8, pl.MemRef("mem_ddr_34", pl.const(0, pl.INT64), 3145728)] = pl.yield_(o_r_i8_pad_inline667__tile_3)
        return o_r_i8_pad_inline667__iter_v1

    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def rmsnorm_rope_cache_write(
        bs_inline2088__ssa_v0: pl.Scalar[pl.INDEX],
        cmp_cos_il__rv_v2: pl.Tensor[[T_DYN, 64], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)],
        cmp_sin_signed__rv_v2: pl.Tensor[[T_DYN, 64], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)],
        pooled_kv_inline553__rv_v2: pl.Tensor[[384, 512], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 786432)],
        normed_kv_inline2080__ssa_v0: pl.InOut[pl.Tensor[[384, 512], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 786432)]],
        norm_w_2d_inline2100__ssa_v0: pl.Tensor[[1, 512], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 2048)],
        cmp_kv_cache_flat_inline2074__ssa_v0: pl.Out[pl.Tensor[[cmp_block_num_inline2079__ssa_v0 * pl.const(32, pl.INDEX), 512], pl.BF16, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 0)]],
        kv_flat_inline2091__ssa_v0: pl.Out[pl.Tensor[[T_DYN, 512], pl.FP32, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 0)]],
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
        rms_blk_inline2078__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        b0_inline2102__ssa_v0: pl.Scalar[pl.INDEX] = rms_blk_inline2078__ssa_v0 * 16
        rms_blk_rows_inline2084__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(bs_inline2088__ssa_v0 - b0_inline2102__ssa_v0, 16)
        cos_b_inline2103__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_8, pl.const(16896, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(valid_shape=[rms_blk_rows_inline2084__ssa_v0, 64])] = (
            pl.tile.load(cmp_cos_il__rv_v2, [b0_inline2102__ssa_v0, 0], [16, 64], [rms_blk_rows_inline2084__ssa_v0, 64], target_memory=pl.Mem.Vec)
        )
        sin_b_inline2104__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_9, pl.const(20992, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(valid_shape=[rms_blk_rows_inline2084__ssa_v0, 64])] = (
            pl.tile.load(cmp_sin_signed__rv_v2, [b0_inline2102__ssa_v0, 0], [16, 64], [rms_blk_rows_inline2084__ssa_v0, 64], target_memory=pl.Mem.Vec)
        )
        partial_sq_inline2063__tile: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_35, pl.const(12544, pl.INT64), 64), pl.Mem.Vec] = pl.tile.full([1, 16], dtype=pl.FP32, value=0.0)
        for k0_inline2073__idx_v0, (partial_sq_inline2063__iter_v1,) in pl.range(0, 512, 64, init_values=(partial_sq_inline2063__tile,)):
            kv_rms_chunk_inline2060__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(4096, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.load(
                pooled_kv_inline553__rv_v2, [b0_inline2102__ssa_v0, k0_inline2073__idx_v0], [16, 64], [16, 64], target_memory=pl.Mem.Vec
            )
            kv_rms_sq_inline2085__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.mul(
                kv_rms_chunk_inline2060__tile, kv_rms_chunk_inline2060__tile
            )
            tmp_tile: pl.Tile[[16, 128], pl.FP32, pl.MemRef(mem_vec_13, pl.const(4096, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.create([16, 128], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            t__tile: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_32, pl.const(12288, pl.INT64), 64), pl.Mem.Vec] = pl.tile.row_sum(kv_rms_sq_inline2085__tile, tmp_tile)
            kv_rms_rowsum_inline2089__tile: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_32, pl.const(12288, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(t__tile, [1, 16])
            partial_sq_inline2063__tile_1: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_35, pl.const(12544, pl.INT64), 64), pl.Mem.Vec] = pl.tile.add(
                partial_sq_inline2063__iter_v1, kv_rms_rowsum_inline2089__tile
            )
            partial_sq_inline2063__rv_v2: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_35, pl.const(12544, pl.INT64), 64), pl.Mem.Vec] = pl.yield_(partial_sq_inline2063__tile_1)
        t__tile_1: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_13, pl.const(4096, pl.INT64), 64), pl.Mem.Vec] = pl.tile.muls(partial_sq_inline2063__rv_v2, 0.001953125)
        t__tile_2: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_13, pl.const(4096, pl.INT64), 64), pl.Mem.Vec] = pl.tile.adds(t__tile_1, 9.9999999999999995e-07)
        variance_inline2064__tile: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_13, pl.const(4096, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(t__tile_2, [16, 1])
        t__rm_a0_tmp_v0: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_13, pl.const(4096, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(variance_inline2064__tile, [1, 16])
        t__row_major_tmp_v1: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_13, pl.const(4096, pl.INT64), 64), pl.Mem.Vec] = pl.tile.sqrt(t__rm_a0_tmp_v0)
        t__tile_3: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_13, pl.const(4096, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(t__row_major_tmp_v1, [16, 1])
        inv_rms_inline2067__rm_a0_tmp_v2: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_13, pl.const(4096, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(t__tile_3, [1, 16])
        inv_rms_inline2067__row_major_tmp_v3: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_35, pl.const(12544, pl.INT64), 64), pl.Mem.Vec] = pl.tile.recip(inv_rms_inline2067__rm_a0_tmp_v2)
        inv_rms_inline2067__tile: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_35, pl.const(12544, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(inv_rms_inline2067__row_major_tmp_v3, [16, 1])
        for k0_inline2059__idx_v0, (normed_kv_inline2080__iter_v1,) in pl.range(0, 448, 64, init_values=(normed_kv_inline2080__ssa_v0,)):
            kv_norm_chunk_inline2062__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(4096, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.load(
                pooled_kv_inline553__rv_v2, [b0_inline2102__ssa_v0, k0_inline2059__idx_v0], [16, 64], [16, 64], target_memory=pl.Mem.Vec
            )
            gamma_inline2058__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(0, pl.INT64), 256), pl.Mem.Vec] = pl.tile.load(
                norm_w_2d_inline2100__ssa_v0, [0, k0_inline2059__idx_v0], [1, 64], [1, 64], target_memory=pl.Mem.Vec
            )
            t__tile_4: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(4096, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.row_expand_mul(kv_norm_chunk_inline2062__tile, inv_rms_inline2067__tile)
            normed_chunk_inline2097__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(4096, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.col_expand_mul(t__tile_4, gamma_inline2058__tile)
            normed_kv_inline2080__tile: pl.Tensor[[384, 512], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 786432)] = pl.tile.store(
                normed_chunk_inline2097__tile, [b0_inline2102__ssa_v0, k0_inline2059__idx_v0], normed_kv_inline2080__iter_v1
            )
            normed_kv_inline2080__rv_v2: pl.Tensor[[384, 512], pl.FP32, pl.MemRef("mem_ddr_25", pl.const(0, pl.INT64), 786432)] = pl.yield_(normed_kv_inline2080__tile)
        kv_rope_norm_inline2057__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(4096, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.load(
            pooled_kv_inline553__rv_v2, [b0_inline2102__ssa_v0, 448], [16, 64], [16, 64], target_memory=pl.Mem.Vec
        )
        gamma_rope_inline2056__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(0, pl.INT64), 256), pl.Mem.Vec] = pl.tile.load(
            norm_w_2d_inline2100__ssa_v0, [0, 448], [1, 64], [1, 64], target_memory=pl.Mem.Vec
        )
        t__tile_5: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(4096, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.row_expand_mul(kv_rope_norm_inline2057__tile, inv_rms_inline2067__tile)
        rope_normed_inline2055__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(4096, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.col_expand_mul(t__tile_5, gamma_rope_inline2056__tile)
        rope_ones_inline2099__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.full([16, 64], dtype=pl.FP32, value=1.0)
        rope_index_inline2095__ci_tmp_v0: pl.Tile[[1, 192], pl.FP32, pl.MemRef(mem_vec_35, pl.const(12544, pl.INT64), 768), pl.Mem.Vec] = pl.tile.create(
            [1, 192], dtype=pl.FP32, target_memory=pl.Mem.Vec
        )
        rope_index_inline2095__tile: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_32, pl.const(12288, pl.INT64), 256), pl.Mem.Vec] = pl.tile.ci(
            pl.const(0, pl.INT32), [1, 64], tmp=rope_index_inline2095__ci_tmp_v0, dtype=pl.INT32, descending=False
        )
        rope_index_f_inline2054__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_35, pl.const(12544, pl.INT64), 256), pl.Mem.Vec] = pl.tile.cast(
            rope_index_inline2095__tile, target_type=pl.FP32, mode="round"
        )
        rope_col_inline2069__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.col_expand_mul(
            rope_ones_inline2099__tile, rope_index_f_inline2054__tile
        )
        t__tile_6: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_35, pl.const(12544, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.muls(rope_col_inline2069__tile, 0.5)
        t__tile_7: pl.Tile[[16, 64], pl.INT32, pl.MemRef(mem_vec_35, pl.const(12544, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(t__tile_6, target_type=pl.INT32, mode="trunc")
        rope_dup_f_inline2061__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_35, pl.const(12544, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(t__tile_7, target_type=pl.FP32, mode="round")
        t__tile_8: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_35, pl.const(12544, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.muls(rope_dup_f_inline2061__tile, 2.0)
        rope_lane_inline2101__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_35, pl.const(12544, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.sub(rope_col_inline2069__tile, t__tile_8)
        t__tile_9: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.adds(rope_col_inline2069__tile, 1.0)
        t__tile_10: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_35, pl.const(12544, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.muls(rope_lane_inline2101__tile, 2.0)
        t__tile_11: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.sub(t__tile_9, t__tile_10)
        rope_swap_idx_inline2053__tile: pl.Tile[[16, 64], pl.INT32, pl.MemRef(mem_vec_12, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(t__tile_11, target_type=pl.INT32, mode="round")
        gather_acc_init: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_35, pl.const(12544, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.create([16, 64], dtype=pl.FP32, target_memory=pl.Mem.Vec)
        for gather_lv, (gather_ia,) in pl.range(16, init_values=(gather_acc_init,)):
            gather_inp_row: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(4096, pl.INT64), 256), pl.Mem.Vec] = pl.tile.slice(rope_normed_inline2055__tile, [1, 64], [gather_lv, 0], [1, 64])
            gather_idx_row: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_12, pl.const(0, pl.INT64), 256), pl.Mem.Vec] = pl.tile.slice(rope_swap_idx_inline2053__tile, [1, 64], [gather_lv, 0], [1, 64])
            gather_row_tmp: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_32, pl.const(12288, pl.INT64), 256), pl.Mem.Vec] = pl.tile.create([1, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec)
            gather_row: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_46, pl.const(16640, pl.INT64), 256), pl.Mem.Vec] = pl.tile.gather(gather_inp_row, gather_idx_row, gather_row_tmp)
            gather_asmbl: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_35, pl.const(12544, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.assemble(gather_ia, gather_row, [gather_lv, 0])
            gather_rv: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_35, pl.const(12544, pl.INT64), 4096), pl.Mem.Vec] = pl.yield_(gather_asmbl)
        swapped_inline2083__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_35, pl.const(12544, pl.INT64), 4096), pl.Mem.Vec] = gather_rv
        t__tile_12: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(4096, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.mul(rope_normed_inline2055__tile, cos_b_inline2103__tile)
        t__tile_13: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_8, pl.const(16896, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.mul(swapped_inline2083__tile, sin_b_inline2104__tile)
        rope_rot_inline2052__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(4096, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.add(t__tile_12, t__tile_13)
        normed_kv_inline2080__tile_1: pl.Tensor[[384, 512], pl.FP32, pl.MemRef("mem_ddr_25", pl.const(0, pl.INT64), 786432)] = pl.tile.store(
            rope_rot_inline2052__tile, [b0_inline2102__ssa_v0, 448], normed_kv_inline2080__rv_v2
        )
        for inner_inline2051__idx_v0, (cmp_kv_cache_flat_inline2074__iter_v1, kv_flat_inline2091__iter_v1) in pl.range(
            rms_blk_rows_inline2084__ssa_v0, init_values=(cmp_kv_cache_flat_inline2074__ssa_v0, kv_flat_inline2091__ssa_v0)
        ):
            token_inline2094__ssa_v1: pl.Scalar[pl.INDEX] = b0_inline2102__ssa_v0 + inner_inline2051__idx_v0
            cache_row_i64_inline2066__tile: pl.Scalar[pl.INT64] = pl.tensor.read(cmp_slot_mapping__ssa_v0, [token_inline2094__ssa_v1])
            if 0 <= cache_row_i64_inline2066__tile:
                cache_row_inline2050__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(cache_row_i64_inline2066__tile, pl.INDEX)
                kv_row_fp32_inline2049__tile: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_13, pl.const(4096, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.load(
                    normed_kv_inline2080__tile_1, [token_inline2094__ssa_v1, 0], [1, 512], [1, 512], target_memory=pl.Mem.Vec
                )
                kv_flat_inline2091__tile: pl.Tensor[[T_DYN, 512], pl.FP32, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 0)] = pl.tile.store(
                    kv_row_fp32_inline2049__tile, [token_inline2094__ssa_v1, 0], kv_flat_inline2091__iter_v1
                )
                t__tile_14: pl.Tile[[1, 512], pl.BF16, pl.MemRef(mem_vec_13, pl.const(4096, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.cast(kv_row_fp32_inline2049__tile, target_type=pl.BF16, mode="rint")
                cmp_kv_cache_flat_inline2074__tile: pl.Tensor[[cmp_block_num_inline2079__ssa_v0 * pl.const(32, pl.INDEX), 512], pl.BF16, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 0)] = (
                    pl.tile.store(t__tile_14, [cache_row_inline2050__ssa_v0, 0], cmp_kv_cache_flat_inline2074__iter_v1)
                )
                cmp_kv_cache_flat_inline2074__phi_v4, kv_flat_inline2091__phi_v4 = pl.yield_(cmp_kv_cache_flat_inline2074__tile, kv_flat_inline2091__tile)
            else:
                cmp_kv_cache_flat_inline2074__phi_v4, kv_flat_inline2091__phi_v4 = pl.yield_(cmp_kv_cache_flat_inline2074__iter_v1, kv_flat_inline2091__iter_v1)
            cmp_kv_cache_flat_inline2074__rv_v2, kv_flat_inline2091__rv_v2 = pl.yield_(cmp_kv_cache_flat_inline2074__phi_v4, kv_flat_inline2091__phi_v4)
        return

    @pl.function(type=pl.FunctionType.Spmd)
    def rmsnorm_rope_cache_write_spmd(
        self,
        bs_inline2088__ssa_v0: pl.Scalar[pl.INDEX],
        cmp_cos_il__rv_v2: pl.Tensor[[T_DYN, 64], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)],
        cmp_sin_signed__rv_v2: pl.Tensor[[T_DYN, 64], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)],
        pooled_kv_inline553__rv_v2: pl.Tensor[[384, 512], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 786432)],
        normed_kv_inline2080__ssa_v0: pl.InOut[pl.Tensor[[384, 512], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 786432)]],
        norm_w_2d_inline2100__ssa_v0: pl.Tensor[[1, 512], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 2048)],
        cmp_kv_cache_flat_inline2074__ssa_v0: pl.Out[pl.Tensor[[cmp_block_num_inline2079__ssa_v0 * pl.const(32, pl.INDEX), 512], pl.BF16, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 0)]],
        kv_flat_inline2091__ssa_v0: pl.Out[pl.Tensor[[T_DYN, 512], pl.FP32, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 0)]],
        cmp_slot_mapping__ssa_v0: pl.Tensor[[T_DYN], pl.INT64, pl.MemRef("mem_ddr_7", pl.const(0, pl.INT64), 0)],
    ):
        pl.func_attr({"mx_tensor_views_blocked": True})
        self.rmsnorm_rope_cache_write(
            bs_inline2088__ssa_v0,
            cmp_cos_il__rv_v2,
            cmp_sin_signed__rv_v2,
            pooled_kv_inline553__rv_v2,
            normed_kv_inline2080__ssa_v0,
            norm_w_2d_inline2100__ssa_v0,
            cmp_kv_cache_flat_inline2074__ssa_v0,
            kv_flat_inline2091__ssa_v0,
            cmp_slot_mapping__ssa_v0,
            attrs={"arg_directions": [pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.inout, pl.adir.input, pl.adir.output_existing, pl.adir.output_existing, pl.adir.input]},
        )

    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def rmsnorm_rope(
        normed_kv_inline560__ssa_v1: pl.Out[pl.Tensor[[384, 128], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 98304)]],
        rms_blocks_inline507_inline2176__ssa_v0: pl.Scalar[pl.INDEX],
        rms_workers_inline505_inline2128__ssa_v0: pl.Scalar[pl.INDEX],
        bs_inline492_inline2146__ssa_v0: pl.Scalar[pl.INDEX],
        inner_cos_il__rv_v2: pl.Tensor[[T_DYN, 64], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)],
        inner_sin_signed__rv_v2: pl.Tensor[[T_DYN, 64], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)],
        pooled_kv_inline471_inline2185__rv_v2: pl.Tensor[[384, 128], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 196608)],
        norm_w_2d_inline497_inline2131__ssa_v0: pl.Tensor[[1, 128], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 512)],
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
        rms_worker_inline451_inline2125__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        rope_ones_inline485_inline2123__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_26, pl.const(16384, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.full([16, 64], dtype=pl.FP32, value=1.0)
        rope_index_inline501_inline2124__ci_tmp_v0: pl.Tile[[1, 192], pl.FP32, pl.MemRef(mem_vec_30, pl.const(28736, pl.INT64), 768), pl.Mem.Vec] = pl.tile.create(
            [1, 192], dtype=pl.FP32, target_memory=pl.Mem.Vec
        )
        rope_index_inline501_inline2124__tile: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_19, pl.const(0, pl.INT64), 256), pl.Mem.Vec] = pl.tile.ci(
            pl.const(0, pl.INT32), [1, 64], tmp=rope_index_inline501_inline2124__ci_tmp_v0, dtype=pl.INT32, descending=False
        )
        rope_index_f_inline506_inline2122__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_30, pl.const(28736, pl.INT64), 256), pl.Mem.Vec] = pl.tile.cast(
            rope_index_inline501_inline2124__tile, target_type=pl.FP32, mode="round"
        )
        rope_col_inline508_inline2121__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_26, pl.const(16384, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.col_expand_mul(
            rope_ones_inline485_inline2123__tile, rope_index_f_inline506_inline2122__tile
        )
        t__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_30, pl.const(28736, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.muls(rope_col_inline508_inline2121__tile, 0.5)
        t__tile_1: pl.Tile[[16, 64], pl.INT32, pl.MemRef(mem_vec_30, pl.const(28736, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(t__tile, target_type=pl.INT32, mode="trunc")
        rope_dup_f_inline458_inline2120__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_30, pl.const(28736, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(
            t__tile_1, target_type=pl.FP32, mode="round"
        )
        t__tile_2: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_30, pl.const(28736, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.muls(rope_dup_f_inline458_inline2120__tile, 2.0)
        rope_lane_inline511_inline2138__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_30, pl.const(28736, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.sub(
            rope_col_inline508_inline2121__tile, t__tile_2
        )
        t__tile_3: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_26, pl.const(16384, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.adds(rope_col_inline508_inline2121__tile, 1.0)
        t__tile_4: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_30, pl.const(28736, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.muls(rope_lane_inline511_inline2138__tile, 2.0)
        t__tile_5: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_26, pl.const(16384, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.sub(t__tile_3, t__tile_4)
        rope_swap_idx_inline461_inline2135__tile: pl.Tile[[16, 64], pl.INT32, pl.MemRef(mem_vec_19, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(
            t__tile_5, target_type=pl.INT32, mode="round"
        )
        for rms_blk_inline442_inline2119__idx_v0, (normed_kv_inline560__iter_v2,) in pl.range(
            rms_worker_inline451_inline2125__ssa_v0, rms_blocks_inline507_inline2176__ssa_v0, rms_workers_inline505_inline2128__ssa_v0, init_values=(normed_kv_inline560__ssa_v1,)
        ):
            b0_inline441_inline2209__ssa_v0: pl.Scalar[pl.INDEX] = rms_blk_inline442_inline2119__idx_v0 * 16
            rms_blk_rows_inline502_inline2200__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(bs_inline492_inline2146__ssa_v0 - b0_inline441_inline2209__ssa_v0, 16)
            cos_b_inline463_inline2179__tile: pl.Tile[
                [16, 64], pl.FP32, pl.MemRef(mem_vec_20, pl.const(4096, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(valid_shape=[rms_blk_rows_inline502_inline2200__ssa_v0, 64])
            ] = pl.tile.load(inner_cos_il__rv_v2, [b0_inline441_inline2209__ssa_v0, 0], [16, 64], [rms_blk_rows_inline502_inline2200__ssa_v0, 64], target_memory=pl.Mem.Vec)
            sin_b_inline460_inline2208__tile: pl.Tile[
                [16, 64], pl.FP32, pl.MemRef(mem_vec_21, pl.const(8192, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(valid_shape=[rms_blk_rows_inline502_inline2200__ssa_v0, 64])
            ] = pl.tile.load(inner_sin_signed__rv_v2, [b0_inline441_inline2209__ssa_v0, 0], [16, 64], [rms_blk_rows_inline502_inline2200__ssa_v0, 64], target_memory=pl.Mem.Vec)
            partial_sq_inline486_inline2118__tile: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_49, pl.const(36992, pl.INT64), 64), pl.Mem.Vec] = pl.tile.full([1, 16], dtype=pl.FP32, value=0.0)
            kv_rms_chunk_inline439_inline2155__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_26, pl.const(16384, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.load(
                pooled_kv_inline471_inline2185__rv_v2, [b0_inline441_inline2209__ssa_v0, 0], [16, 64], [16, 64], target_memory=pl.Mem.Vec
            )
            kv_rms_chunk_inline439_inline2155__tile_1: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_30, pl.const(28736, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.load(
                pooled_kv_inline471_inline2185__rv_v2, [b0_inline441_inline2209__ssa_v0, 64], [16, 64], [16, 64], target_memory=pl.Mem.Vec
            )
            kv_rms_sq_inline438_inline2150__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_25, pl.const(12288, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.mul(
                kv_rms_chunk_inline439_inline2155__tile, kv_rms_chunk_inline439_inline2155__tile
            )
            tmp_tile: pl.Tile[[16, 128], pl.FP32, pl.MemRef(mem_vec_26, pl.const(16384, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.create([16, 128], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            t__tile_6: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_27, pl.const(24576, pl.INT64), 64), pl.Mem.Vec] = pl.tile.row_sum(kv_rms_sq_inline438_inline2150__tile, tmp_tile)
            kv_rms_rowsum_inline437_inline2114__tile: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_27, pl.const(24576, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(t__tile_6, [1, 16])
            partial_sq_inline486_inline2118__tile_1: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_26, pl.const(16384, pl.INT64), 64), pl.Mem.Vec] = pl.tile.add(
                partial_sq_inline486_inline2118__tile, kv_rms_rowsum_inline437_inline2114__tile
            )
            kv_rms_sq_inline438_inline2150__tile_1: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_29, pl.const(24640, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.mul(
                kv_rms_chunk_inline439_inline2155__tile_1, kv_rms_chunk_inline439_inline2155__tile_1
            )
            tmp_tile_1: pl.Tile[[16, 128], pl.FP32, pl.MemRef(mem_vec_30, pl.const(28736, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.create([16, 128], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            t__tile_7: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_31, pl.const(36928, pl.INT64), 64), pl.Mem.Vec] = pl.tile.row_sum(kv_rms_sq_inline438_inline2150__tile_1, tmp_tile_1)
            kv_rms_rowsum_inline437_inline2114__tile_1: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_31, pl.const(36928, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(t__tile_7, [1, 16])
            partial_sq_inline486_inline2118__tile_2: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_49, pl.const(36992, pl.INT64), 64), pl.Mem.Vec] = pl.tile.add(
                partial_sq_inline486_inline2118__tile_1, kv_rms_rowsum_inline437_inline2114__tile_1
            )
            t__tile_8: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_26, pl.const(16384, pl.INT64), 64), pl.Mem.Vec] = pl.tile.muls(partial_sq_inline486_inline2118__tile_2, 0.0078125)
            t__tile_9: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_26, pl.const(16384, pl.INT64), 64), pl.Mem.Vec] = pl.tile.adds(t__tile_8, 9.9999999999999995e-07)
            variance_inline436_inline2142__tile: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_26, pl.const(16384, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(t__tile_9, [16, 1])
            t__rm_a0_tmp_v0: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_26, pl.const(16384, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(variance_inline436_inline2142__tile, [1, 16])
            t__row_major_tmp_v1: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_26, pl.const(16384, pl.INT64), 64), pl.Mem.Vec] = pl.tile.sqrt(t__rm_a0_tmp_v0)
            t__tile_10: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_26, pl.const(16384, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(t__row_major_tmp_v1, [16, 1])
            inv_rms_inline435_inline2126__rm_a0_tmp_v2: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_26, pl.const(16384, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(t__tile_10, [1, 16])
            inv_rms_inline435_inline2126__row_major_tmp_v3: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_29, pl.const(24640, pl.INT64), 64), pl.Mem.Vec] = pl.tile.recip(
                inv_rms_inline435_inline2126__rm_a0_tmp_v2
            )
            inv_rms_inline435_inline2126__tile: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_29, pl.const(24640, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(
                inv_rms_inline435_inline2126__row_major_tmp_v3, [16, 1]
            )
            kv_norm_chunk_inline498_inline2181__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_26, pl.const(16384, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.load(
                pooled_kv_inline471_inline2185__rv_v2, [b0_inline441_inline2209__ssa_v0, 0], [16, 64], [16, 64], target_memory=pl.Mem.Vec
            )
            gamma_inline509_inline2116__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_30, pl.const(28736, pl.INT64), 256), pl.Mem.Vec] = pl.tile.load(
                norm_w_2d_inline497_inline2131__ssa_v0, [0, 0], [1, 64], [1, 64], target_memory=pl.Mem.Vec
            )
            t__tile_11: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_26, pl.const(16384, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.row_expand_mul(
                kv_norm_chunk_inline498_inline2181__tile, inv_rms_inline435_inline2126__tile
            )
            normed_chunk_inline433_inline2113__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_26, pl.const(16384, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.col_expand_mul(
                t__tile_11, gamma_inline509_inline2116__tile
            )
            normed_nope_inline432_inline2112__tile: pl.Tile[[16, 64], pl.BF16, pl.MemRef(mem_vec_25, pl.const(12288, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(
                normed_chunk_inline433_inline2113__tile, target_type=pl.BF16, mode="rint"
            )
            kv_rope_norm_inline472_inline2111__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_26, pl.const(16384, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.load(
                pooled_kv_inline471_inline2185__rv_v2, [b0_inline441_inline2209__ssa_v0, 64], [16, 64], [16, 64], target_memory=pl.Mem.Vec
            )
            gamma_rope_inline434_inline2110__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_30, pl.const(28736, pl.INT64), 256), pl.Mem.Vec] = pl.tile.load(
                norm_w_2d_inline497_inline2131__ssa_v0, [0, 64], [1, 64], [1, 64], target_memory=pl.Mem.Vec
            )
            t__tile_12: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_26, pl.const(16384, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.row_expand_mul(
                kv_rope_norm_inline472_inline2111__tile, inv_rms_inline435_inline2126__tile
            )
            rope_normed_inline449_inline2109__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_26, pl.const(16384, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.col_expand_mul(
                t__tile_12, gamma_rope_inline434_inline2110__tile
            )
            gather_acc_init: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_30, pl.const(28736, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.create([16, 64], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            for gather_lv, (gather_ia,) in pl.range(16, init_values=(gather_acc_init,)):
                gather_inp_row: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_26, pl.const(16384, pl.INT64), 256), pl.Mem.Vec] = pl.tile.slice(
                    rope_normed_inline449_inline2109__tile, [1, 64], [gather_lv, 0], [1, 64]
                )
                gather_idx_row: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_19, pl.const(0, pl.INT64), 256), pl.Mem.Vec] = pl.tile.slice(
                    rope_swap_idx_inline461_inline2135__tile, [1, 64], [gather_lv, 0], [1, 64]
                )
                gather_row_tmp: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_29, pl.const(24640, pl.INT64), 256), pl.Mem.Vec] = pl.tile.create([1, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec)
                gather_row: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_49, pl.const(36992, pl.INT64), 256), pl.Mem.Vec] = pl.tile.gather(gather_inp_row, gather_idx_row, gather_row_tmp)
                gather_asmbl: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_30, pl.const(28736, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.assemble(gather_ia, gather_row, [gather_lv, 0])
                gather_rv: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_30, pl.const(28736, pl.INT64), 4096), pl.Mem.Vec] = pl.yield_(gather_asmbl)
            swapped_inline431_inline2108__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_30, pl.const(28736, pl.INT64), 4096), pl.Mem.Vec] = gather_rv
            t__tile_13: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_26, pl.const(16384, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.mul(
                rope_normed_inline449_inline2109__tile, cos_b_inline463_inline2179__tile
            )
            t__tile_14: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_30, pl.const(28736, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.mul(
                swapped_inline431_inline2108__tile, sin_b_inline460_inline2208__tile
            )
            rope_rot_inline452_inline2199__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_26, pl.const(16384, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.add(t__tile_13, t__tile_14)
            normed_rope_inline430_inline2107__tile: pl.Tile[[16, 64], pl.BF16, pl.MemRef(mem_vec_26, pl.const(16384, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(
                rope_rot_inline452_inline2199__tile, target_type=pl.BF16, mode="rint"
            )
            for inner_inline429_inline2166__idx_v0, (normed_kv_inline560__iter_v4,) in pl.range(rms_blk_rows_inline502_inline2200__ssa_v0, init_values=(normed_kv_inline560__iter_v2,)):
                token_inline454_inline2186__ssa_v2: pl.Scalar[pl.INDEX] = b0_inline441_inline2209__ssa_v0 + inner_inline429_inline2166__idx_v0
                token_pos_inline478_inline2184__tile: pl.Scalar[pl.INT32] = pl.tensor.read(position_ids__ssa_v0, [token_inline454_inline2186__ssa_v2])
                if (pl.cast(token_pos_inline478_inline2184__tile, pl.INDEX) + 1) % 4 == 0:
                    request_inline428_inline2106__ssa_v0: pl.Scalar[pl.INDEX] = token_inline454_inline2186__ssa_v2 // 6
                    first_pos_inline462_inline2133__tile: pl.Scalar[pl.INT32] = pl.tensor.read(position_ids__ssa_v0, [request_inline428_inline2106__ssa_v0 * 6])
                    first_boundary_inline445_inline2173__ssa_v0: pl.Scalar[pl.INDEX] = 3 - pl.cast(first_pos_inline462_inline2133__tile, pl.INDEX) % 4
                    compact_token_inline427_inline2105__ssa_v0: pl.Scalar[pl.INDEX] = (
                        request_inline428_inline2106__ssa_v0 * 2 + (token_inline454_inline2186__ssa_v2 % 6 - first_boundary_inline445_inline2173__ssa_v0) // 4
                    )
                    t__tile_15: pl.Tile[[1, 64], pl.BF16, pl.MemRef(mem_vec_25, pl.const(12288, pl.INT64), 128), pl.Mem.Vec] = pl.tile.slice(
                        normed_nope_inline432_inline2112__tile, [1, 64], [inner_inline429_inline2166__idx_v0, 0]
                    )
                    normed_kv_inline560__tile: pl.Tensor[[384, 128], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 98304)] = pl.tile.store(
                        t__tile_15, [compact_token_inline427_inline2105__ssa_v0, 0], normed_kv_inline560__iter_v4
                    )
                    t__tile_16: pl.Tile[[1, 64], pl.BF16, pl.MemRef(mem_vec_26, pl.const(16384, pl.INT64), 128), pl.Mem.Vec] = pl.tile.slice(
                        normed_rope_inline430_inline2107__tile, [1, 64], [inner_inline429_inline2166__idx_v0, 0]
                    )
                    normed_kv_inline560__tile_1: pl.Tensor[[384, 128], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 98304)] = pl.tile.store(
                        t__tile_16, [compact_token_inline427_inline2105__ssa_v0, 64], normed_kv_inline560__tile
                    )
                    normed_kv_inline560__phi_v8: pl.Tensor[[384, 128], pl.BF16, pl.MemRef("mem_ddr_55", pl.const(0, pl.INT64), 98304)] = pl.yield_(normed_kv_inline560__tile_1)
                else:
                    normed_kv_inline560__phi_v8: pl.Tensor[[384, 128], pl.BF16, pl.MemRef("mem_ddr_55", pl.const(0, pl.INT64), 98304)] = pl.yield_(normed_kv_inline560__iter_v4)
                normed_kv_inline560__rv_v5: pl.Tensor[[384, 128], pl.BF16, pl.MemRef("mem_ddr_56", pl.const(0, pl.INT64), 98304)] = pl.yield_(normed_kv_inline560__phi_v8)
            normed_kv_inline560__rv_v3: pl.Tensor[[384, 128], pl.BF16, pl.MemRef("mem_ddr_57", pl.const(0, pl.INT64), 98304)] = pl.yield_(normed_kv_inline560__rv_v5)
        return normed_kv_inline560__ssa_v1

    @pl.function(type=pl.FunctionType.Spmd)
    def rmsnorm_rope_spmd(
        self,
        normed_kv_inline560__ssa_v1: pl.Out[pl.Tensor[[384, 128], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 98304)]],
        rms_blocks_inline507_inline2176__ssa_v0: pl.Scalar[pl.INDEX],
        rms_workers_inline505_inline2128__ssa_v0: pl.Scalar[pl.INDEX],
        bs_inline492_inline2146__ssa_v0: pl.Scalar[pl.INDEX],
        inner_cos_il__rv_v2: pl.Tensor[[T_DYN, 64], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)],
        inner_sin_signed__rv_v2: pl.Tensor[[T_DYN, 64], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)],
        pooled_kv_inline471_inline2185__rv_v2: pl.Tensor[[384, 128], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 196608)],
        norm_w_2d_inline497_inline2131__ssa_v0: pl.Tensor[[1, 128], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 512)],
        position_ids__ssa_v0: pl.Tensor[[T_DYN], pl.INT32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 0)],
    ) -> pl.Tensor[[384, 128], pl.BF16]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        normed_kv_inline560__rv_v3: pl.Tensor[[384, 128], pl.BF16, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 98304)] = self.rmsnorm_rope(
            normed_kv_inline560__ssa_v1,
            rms_blocks_inline507_inline2176__ssa_v0,
            rms_workers_inline505_inline2128__ssa_v0,
            bs_inline492_inline2146__ssa_v0,
            inner_cos_il__rv_v2,
            inner_sin_signed__rv_v2,
            pooled_kv_inline471_inline2185__rv_v2,
            norm_w_2d_inline497_inline2131__ssa_v0,
            position_ids__ssa_v0,
            attrs={"arg_directions": [pl.adir.output_existing, pl.adir.scalar, pl.adir.scalar, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input]},
        )
        return normed_kv_inline560__ssa_v1

    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def rope_cs(
        rope_swap_idx_inline2571__ssa_v0: pl.Out[pl.Tensor[[16, 64], pl.INT32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 4096)]],
        rope_cos_il_inline2484__ssa_v0: pl.Out[pl.Tensor[[384, 64], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 98304)]],
        rope_sin_signed_inline2456__ssa_v0: pl.Out[pl.Tensor[[384, 64], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 98304)]],
        rope_cs_blocks_inline2504__ssa_v0: pl.Scalar[pl.INDEX],
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
        sw_ones_inline2455__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_5, pl.const(8704, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.full([16, 64], dtype=pl.FP32, value=1.0)
        t__ci_tmp_v0: pl.Tile[[1, 192], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 768), pl.Mem.Vec] = pl.tile.create([1, 192], dtype=pl.FP32, target_memory=pl.Mem.Vec)
        t__tile: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_27, pl.const(4096, pl.INT64), 256), pl.Mem.Vec] = pl.tile.ci(
            pl.const(0, pl.INT32), [1, 64], tmp=t__ci_tmp_v0, dtype=pl.INT32, descending=False
        )
        sw_idx_f_inline2530__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 256), pl.Mem.Vec] = pl.tile.cast(t__tile, target_type=pl.FP32, mode="round")
        sw_col_inline2466__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_5, pl.const(8704, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.col_expand_mul(
            sw_ones_inline2455__tile, sw_idx_f_inline2530__tile
        )
        t__tile_1: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.muls(sw_col_inline2466__tile, 0.5)
        sw_dup_i32_inline2526__tile: pl.Tile[[16, 64], pl.INT32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(t__tile_1, target_type=pl.INT32, mode="trunc")
        sw_dup_f_inline2454__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(
            sw_dup_i32_inline2526__tile, target_type=pl.FP32, mode="round"
        )
        t__tile_2: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.muls(sw_dup_f_inline2454__tile, 2.0)
        sw_lane_inline2505__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.sub(sw_col_inline2466__tile, t__tile_2)
        t__tile_3: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_5, pl.const(8704, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.adds(sw_col_inline2466__tile, 1.0)
        t__tile_4: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.muls(sw_lane_inline2505__tile, 2.0)
        sw_swap_f_inline2453__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_5, pl.const(8704, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.sub(t__tile_3, t__tile_4)
        t__tile_5: pl.Tile[[16, 64], pl.INT32, pl.MemRef(mem_vec_5, pl.const(8704, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(sw_swap_f_inline2453__tile, target_type=pl.INT32, mode="round")
        rope_swap_idx_inline2571__tile: pl.Tensor[[16, 64], pl.INT32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 4096)] = pl.tile.store(t__tile_5, [0, 0], rope_swap_idx_inline2571__ssa_v0)
        cs_ones_inline2568__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_5, pl.const(8704, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.full([8, 64], dtype=pl.FP32, value=1.0)
        t__ci_tmp_v1: pl.Tile[[1, 192], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 768), pl.Mem.Vec] = pl.tile.create([1, 192], dtype=pl.FP32, target_memory=pl.Mem.Vec)
        t__tile_6: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_27, pl.const(4096, pl.INT64), 256), pl.Mem.Vec] = pl.tile.ci(
            pl.const(0, pl.INT32), [1, 64], tmp=t__ci_tmp_v1, dtype=pl.INT32, descending=False
        )
        cs_idx_f_inline2486__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 256), pl.Mem.Vec] = pl.tile.cast(t__tile_6, target_type=pl.FP32, mode="round")
        cs_col_inline2551__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_5, pl.const(8704, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.col_expand_mul(
            cs_ones_inline2568__tile, cs_idx_f_inline2486__tile
        )
        t__tile_7: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.muls(cs_col_inline2551__tile, 0.5)
        cs_dup_i32_inline2452__tile: pl.Tile[[8, 64], pl.INT32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(t__tile_7, target_type=pl.INT32, mode="trunc")
        cs_dup_f_inline2451__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(
            cs_dup_i32_inline2452__tile, target_type=pl.FP32, mode="round"
        )
        cs_dup_idx_inline2510__tile: pl.Tile[[8, 64], pl.INT32, pl.MemRef(mem_vec_27, pl.const(4096, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(
            cs_dup_f_inline2451__tile, target_type=pl.INT32, mode="round"
        )
        t__tile_8: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.muls(cs_dup_f_inline2451__tile, 2.0)
        cs_lane_inline2450__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_5, pl.const(8704, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.sub(cs_col_inline2551__tile, t__tile_8)
        t__tile_9: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_5, pl.const(8704, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.muls(cs_lane_inline2450__tile, 2.0)
        t__tile_10: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_5, pl.const(8704, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.subs(t__tile_9, 1.0)
        cs_sign_inline2574__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_5, pl.const(8704, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.neg(t__tile_10)
        for cs_rb_inline2449__idx_v0, (rope_cos_il_inline2484__iter_v1, rope_sin_signed_inline2456__iter_v1) in pl.range(
            rope_cs_blocks_inline2504__ssa_v0, init_values=(rope_cos_il_inline2484__ssa_v0, rope_sin_signed_inline2456__ssa_v0)
        ):
            cs_t0_inline2448__ssa_v0: pl.Scalar[pl.INDEX] = cs_rb_inline2449__idx_v0 * 8
            cs_cos_inline2462__tile: pl.Tile[[8, 32], pl.FP32, pl.MemRef(mem_vec_33, pl.const(6144, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.load(
                freqs_cos__ssa_v0, [cs_t0_inline2448__ssa_v0, 0], [8, 32], [8, 32], target_memory=pl.Mem.Vec
            )
            cs_sin_inline2447__tile: pl.Tile[[8, 32], pl.FP32, pl.MemRef(mem_vec_34, pl.const(7168, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.load(
                freqs_sin__ssa_v0, [cs_t0_inline2448__ssa_v0, 0], [8, 32], [8, 32], target_memory=pl.Mem.Vec
            )
            gather_acc_init: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.create([8, 64], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            for gather_lv, (gather_ia,) in pl.range(8, init_values=(gather_acc_init,)):
                gather_inp_row: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_33, pl.const(6144, pl.INT64), 128), pl.Mem.Vec] = pl.tile.slice(cs_cos_inline2462__tile, [1, 32], [gather_lv, 0], [1, 32])
                gather_idx_row: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_27, pl.const(4096, pl.INT64), 256), pl.Mem.Vec] = pl.tile.slice(
                    cs_dup_idx_inline2510__tile, [1, 64], [gather_lv, 0], [1, 64]
                )
                gather_row_tmp: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_36, pl.const(8192, pl.INT64), 256), pl.Mem.Vec] = pl.tile.create([1, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec)
                gather_row: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_37, pl.const(8448, pl.INT64), 256), pl.Mem.Vec] = pl.tile.gather(gather_inp_row, gather_idx_row, gather_row_tmp)
                gather_asmbl: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.assemble(gather_ia, gather_row, [gather_lv, 0])
                gather_rv: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 2048), pl.Mem.Vec] = pl.yield_(gather_asmbl)
            t__tile_11: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 2048), pl.Mem.Vec] = gather_rv
            rope_cos_il_inline2484__tile: pl.Tensor[[384, 64], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 98304)] = pl.tile.store(
                t__tile_11, [cs_t0_inline2448__ssa_v0, 0], rope_cos_il_inline2484__iter_v1
            )
            gather_acc_init_1: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.create([8, 64], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            for gather_lv_1, (gather_ia_1,) in pl.range(8, init_values=(gather_acc_init_1,)):
                gather_inp_row_1: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_34, pl.const(7168, pl.INT64), 128), pl.Mem.Vec] = pl.tile.slice(
                    cs_sin_inline2447__tile, [1, 32], [gather_lv_1, 0], [1, 32]
                )
                gather_idx_row_1: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_27, pl.const(4096, pl.INT64), 256), pl.Mem.Vec] = pl.tile.slice(
                    cs_dup_idx_inline2510__tile, [1, 64], [gather_lv_1, 0], [1, 64]
                )
                gather_row_tmp_1: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_33, pl.const(6144, pl.INT64), 256), pl.Mem.Vec] = pl.tile.create([1, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec)
                gather_row_1: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_36, pl.const(8192, pl.INT64), 256), pl.Mem.Vec] = pl.tile.gather(gather_inp_row_1, gather_idx_row_1, gather_row_tmp_1)
                gather_asmbl_1: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.assemble(gather_ia_1, gather_row_1, [gather_lv_1, 0])
                gather_rv_1: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 2048), pl.Mem.Vec] = pl.yield_(gather_asmbl_1)
            cs_sin_il_inline2446__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 2048), pl.Mem.Vec] = gather_rv_1
            t__tile_12: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.mul(cs_sin_il_inline2446__tile, cs_sign_inline2574__tile)
            rope_sin_signed_inline2456__tile: pl.Tensor[[384, 64], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 98304)] = pl.tile.store(
                t__tile_12, [cs_t0_inline2448__ssa_v0, 0], rope_sin_signed_inline2456__iter_v1
            )
            rope_cos_il_inline2484__rv_v2, rope_sin_signed_inline2456__rv_v2 = pl.yield_(rope_cos_il_inline2484__tile, rope_sin_signed_inline2456__tile)
        return rope_swap_idx_inline2571__ssa_v0, rope_cos_il_inline2484__ssa_v0, rope_sin_signed_inline2456__ssa_v0

    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def scatter_softmax_pool(
        pooled_kv_inline553__ssa_v0: pl.Out[pl.Tensor[[384, 512], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 786432)]],
        b_dim_inline760_inline1995__ssa_v0: pl.Scalar[pl.INDEX],
        pool_workers_inline756_inline1990__ssa_v0: pl.Scalar[pl.INDEX],
        position_ids__ssa_v0: pl.Tensor[[T_DYN], pl.INT32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)],
        s_dim_inline769_inline2026__ssa_v0: pl.Scalar[pl.INDEX],
        cmp4_score_proj_pad_inline758_inline1992__ssa_v0: pl.Tensor[[384, 1024], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 1572864)],
        cmp_ape__ssa_v0: pl.Tensor[[4, 1024], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 16384)],
        cmp4_kv_proj_pad_inline761_inline2009__ssa_v0: pl.Tensor[[384, 1024], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 1572864)],
        compress_state_block_table__ssa_v0: pl.Tensor[[B_DYN, 7], pl.INT32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 0)],
        compress_state_flat_inline751_inline1991__ssa_v0: pl.Tensor[[compress_state_rows_inline745_inline2013__ssa_v0, 2048], pl.FP32, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 0)],
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
        pool_worker_inline742_inline2025__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        for c_idx_inline750_inline2027__idx_v0, (pooled_kv_inline553__iter_v1,) in pl.range(
            pool_worker_inline742_inline2025__ssa_v0, b_dim_inline760_inline1995__ssa_v0, pool_workers_inline756_inline1990__ssa_v0, init_values=(pooled_kv_inline553__ssa_v0,)
        ):
            first_pos_b_inline762_inline2015__tile: pl.Scalar[pl.INT32] = pl.tensor.read(position_ids__ssa_v0, [c_idx_inline750_inline2027__idx_v0 * s_dim_inline769_inline2026__ssa_v0])
            for s_idx_inline764_inline2000__idx_v0, (pooled_kv_inline553__iter_v3,) in pl.range(s_dim_inline769_inline2026__ssa_v0, init_values=(pooled_kv_inline553__iter_v1,)):
                token_inline755_inline2028__ssa_v0: pl.Scalar[pl.INDEX] = c_idx_inline750_inline2027__idx_v0 * s_dim_inline769_inline2026__ssa_v0 + s_idx_inline764_inline2000__idx_v0
                token_pos_inline765_inline2011__tile: pl.Scalar[pl.INT32] = pl.tensor.read(position_ids__ssa_v0, [token_inline755_inline2028__ssa_v0])
                t__tile: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_7, pl.const(12288, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.full([1, 512], dtype=pl.FP32, value=0.0)
                pooled_kv_inline553__tile: pl.Tensor[[384, 512], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 786432)] = pl.tile.store(
                    t__tile, [token_inline755_inline2028__ssa_v0, 0], pooled_kv_inline553__iter_v3
                )
                if (pl.cast(token_pos_inline765_inline2011__tile, pl.INDEX) + 1) % 4 == 0:
                    window_start_inline763_inline2029__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(token_pos_inline765_inline2011__tile, pl.INDEX) - 8 + 1
                    last_ape_row_inline766_inline2030__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(pl.cast(token_pos_inline765_inline2011__tile, pl.INDEX) % 4, pl.INDEX)
                    t__tile_1: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_7, pl.const(12288, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.load(
                        cmp4_score_proj_pad_inline758_inline1992__ssa_v0, [token_inline755_inline2028__ssa_v0, 512], [1, 512], [1, 512], target_memory=pl.Mem.Vec
                    )
                    t__tile_2: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_9, pl.const(14336, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.load(
                        cmp_ape__ssa_v0, [last_ape_row_inline766_inline2030__ssa_v0, 512], [1, 512], [1, 512], target_memory=pl.Mem.Vec
                    )
                    mi_inline768_inline2017__tile: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_7, pl.const(12288, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.add(t__tile_1, t__tile_2)
                    t__tile_3: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_9, pl.const(14336, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.sub(mi_inline768_inline2017__tile, mi_inline768_inline2017__tile)
                    li_inline767_inline2031__tile: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_9, pl.const(14336, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.exp(t__tile_3)
                    oi_inline770_inline2005__tile: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_13, pl.const(0, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.load(
                        cmp4_kv_proj_pad_inline761_inline2009__ssa_v0, [token_inline755_inline2028__ssa_v0, 512], [1, 512], [1, 512], target_memory=pl.Mem.Vec
                    )
                    for state_idx_inline743_inline2033__idx_v0, (li_inline767_inline2031__iter_v1, mi_inline768_inline2017__iter_v1, oi_inline770_inline2005__iter_v1) in pl.range(
                        7, init_values=(li_inline767_inline2031__tile, mi_inline768_inline2017__tile, oi_inline770_inline2005__tile)
                    ):
                        logical_pos_inline759_inline2035__ssa_v0: pl.Scalar[pl.INDEX] = window_start_inline763_inline2029__ssa_v0 + state_idx_inline743_inline2033__idx_v0
                        value_inline752_inline2039__tile: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_14, pl.const(2048, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.full(
                            [1, 512], dtype=pl.FP32, value=0.0
                        )
                        score_inline771_inline1998__tile: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_15, pl.const(4096, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.full(
                            [1, 512], dtype=pl.FP32, value=-3.4028234663852886e38
                        )
                        state_half_inline748_inline1993__ssa_v0: pl.Scalar[pl.INDEX] = 0
                        if 4 <= state_idx_inline743_inline2033__idx_v0:
                            state_half_inline748_inline1993__ssa_v1: pl.Scalar[pl.INDEX] = 512
                            state_half_inline748_inline1993__phi_v2: pl.Scalar[pl.INDEX] = pl.yield_(state_half_inline748_inline1993__ssa_v1)
                        else:
                            state_half_inline748_inline1993__phi_v2: pl.Scalar[pl.INDEX] = pl.yield_(state_half_inline748_inline1993__ssa_v0)
                        if 0 <= logical_pos_inline759_inline2035__ssa_v0 and logical_pos_inline759_inline2035__ssa_v0 < pl.cast(first_pos_b_inline762_inline2015__tile, pl.INDEX):
                            ring_row_inline741_inline2041__ssa_v0: pl.Scalar[pl.INDEX] = logical_pos_inline759_inline2035__ssa_v0 % 14
                            state_page_off_inline740_inline2042__ssa_v0: pl.Scalar[pl.INDEX] = ring_row_inline741_inline2041__ssa_v0 // 2
                            state_blk_id_i32_inline739_inline2043__tile: pl.Scalar[pl.INT32] = pl.tensor.read(
                                compress_state_block_table__ssa_v0, [c_idx_inline750_inline2027__idx_v0, state_page_off_inline740_inline2042__ssa_v0]
                            )
                            if 0 <= pl.cast(state_blk_id_i32_inline739_inline2043__tile, pl.INDEX):
                                state_blk_id_inline738_inline2044__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(state_blk_id_i32_inline739_inline2043__tile, pl.INDEX)
                                state_intra_row_inline749_inline2036__ssa_v0: pl.Scalar[pl.INDEX] = ring_row_inline741_inline2041__ssa_v0 % 2
                                state_row_inline737_inline2045__ssa_v0: pl.Scalar[pl.INDEX] = state_blk_id_inline738_inline2044__ssa_v0 * 2 + state_intra_row_inline749_inline2036__ssa_v0
                                value_inline752_inline2039__tile_1: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_16, pl.const(6144, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.load(
                                    compress_state_flat_inline751_inline1991__ssa_v0,
                                    [state_row_inline737_inline2045__ssa_v0, state_half_inline748_inline1993__phi_v2],
                                    [1, 512],
                                    [1, 512],
                                    target_memory=pl.Mem.Vec,
                                )
                                score_inline771_inline1998__tile_1: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_17, pl.const(8192, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.load(
                                    compress_state_flat_inline751_inline1991__ssa_v0,
                                    [state_row_inline737_inline2045__ssa_v0, state_half_inline748_inline1993__phi_v2 + 1024],
                                    [1, 512],
                                    [1, 512],
                                    target_memory=pl.Mem.Vec,
                                )
                                score_inline771_inline1998__phi_v2, value_inline752_inline2039__phi_v2 = pl.yield_(score_inline771_inline1998__tile_1, value_inline752_inline2039__tile_1)
                            else:
                                score_inline771_inline1998__tile_mv: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_17, pl.const(8192, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.move(
                                    score_inline771_inline1998__tile, target_memory=pl.Mem.Vec
                                )
                                value_inline752_inline2039__tile_mv: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_16, pl.const(6144, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.move(
                                    value_inline752_inline2039__tile, target_memory=pl.Mem.Vec
                                )
                                score_inline771_inline1998__phi_v2, value_inline752_inline2039__phi_v2 = pl.yield_(score_inline771_inline1998__tile_mv, value_inline752_inline2039__tile_mv)
                            score_inline771_inline1998__phi_v3, value_inline752_inline2039__phi_v3 = pl.yield_(score_inline771_inline1998__phi_v2, value_inline752_inline2039__phi_v2)
                        else:
                            score_inline771_inline1998__tile_mv_1: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_17, pl.const(8192, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.move(
                                score_inline771_inline1998__tile, target_memory=pl.Mem.Vec
                            )
                            value_inline752_inline2039__tile_mv_1: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_16, pl.const(6144, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.move(
                                value_inline752_inline2039__tile, target_memory=pl.Mem.Vec
                            )
                            score_inline771_inline1998__phi_v3, value_inline752_inline2039__phi_v3 = pl.yield_(score_inline771_inline1998__tile_mv_1, value_inline752_inline2039__tile_mv_1)
                        if pl.cast(first_pos_b_inline762_inline2015__tile, pl.INDEX) <= logical_pos_inline759_inline2035__ssa_v0:
                            if logical_pos_inline759_inline2035__ssa_v0 <= pl.cast(token_pos_inline765_inline2011__tile, pl.INDEX):
                                overlay_token_inline736_inline2046__ssa_v0: pl.Scalar[pl.INDEX] = (
                                    c_idx_inline750_inline2027__idx_v0 * s_dim_inline769_inline2026__ssa_v0
                                    + logical_pos_inline759_inline2035__ssa_v0
                                    - pl.cast(first_pos_b_inline762_inline2015__tile, pl.INDEX)
                                )
                                ape_row_inline753_inline2034__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(logical_pos_inline759_inline2035__ssa_v0 % 4, pl.INDEX)
                                value_inline752_inline2039__tile_2: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_14, pl.const(2048, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.load(
                                    cmp4_kv_proj_pad_inline761_inline2009__ssa_v0,
                                    [overlay_token_inline736_inline2046__ssa_v0, state_half_inline748_inline1993__phi_v2],
                                    [1, 512],
                                    [1, 512],
                                    target_memory=pl.Mem.Vec,
                                )
                                t__tile_4: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_15, pl.const(4096, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.load(
                                    cmp4_score_proj_pad_inline758_inline1992__ssa_v0,
                                    [overlay_token_inline736_inline2046__ssa_v0, state_half_inline748_inline1993__phi_v2],
                                    [1, 512],
                                    [1, 512],
                                    target_memory=pl.Mem.Vec,
                                )
                                t__tile_5: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_24, pl.const(10240, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.load(
                                    cmp_ape__ssa_v0, [ape_row_inline753_inline2034__ssa_v0, state_half_inline748_inline1993__phi_v2], [1, 512], [1, 512], target_memory=pl.Mem.Vec
                                )
                                score_inline771_inline1998__tile_2: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_15, pl.const(4096, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.add(t__tile_4, t__tile_5)
                                score_inline771_inline1998__phi_v5, value_inline752_inline2039__phi_v5 = pl.yield_(score_inline771_inline1998__tile_2, value_inline752_inline2039__tile_2)
                            else:
                                score_inline771_inline1998__phi_v3_mv: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_15, pl.const(4096, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.move(
                                    score_inline771_inline1998__phi_v3, target_memory=pl.Mem.Vec
                                )
                                value_inline752_inline2039__phi_v3_mv: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_14, pl.const(2048, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.move(
                                    value_inline752_inline2039__phi_v3, target_memory=pl.Mem.Vec
                                )
                                score_inline771_inline1998__phi_v5, value_inline752_inline2039__phi_v5 = pl.yield_(score_inline771_inline1998__phi_v3_mv, value_inline752_inline2039__phi_v3_mv)
                            score_inline771_inline1998__phi_v6, value_inline752_inline2039__phi_v6 = pl.yield_(score_inline771_inline1998__phi_v5, value_inline752_inline2039__phi_v5)
                        else:
                            score_inline771_inline1998__phi_v3_mv_1: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_15, pl.const(4096, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.move(
                                score_inline771_inline1998__phi_v3, target_memory=pl.Mem.Vec
                            )
                            value_inline752_inline2039__phi_v3_mv_1: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_14, pl.const(2048, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.move(
                                value_inline752_inline2039__phi_v3, target_memory=pl.Mem.Vec
                            )
                            score_inline771_inline1998__phi_v6, value_inline752_inline2039__phi_v6 = pl.yield_(score_inline771_inline1998__phi_v3_mv_1, value_inline752_inline2039__phi_v3_mv_1)
                        mi_next_inline735_inline2032__tile: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_16, pl.const(6144, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.maximum(
                            mi_inline768_inline2017__iter_v1, score_inline771_inline1998__phi_v6
                        )
                        t__tile_6: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_17, pl.const(8192, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.sub(
                            mi_inline768_inline2017__iter_v1, mi_next_inline735_inline2032__tile
                        )
                        alpha_inline734_inline2047__tile: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_17, pl.const(8192, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.exp(t__tile_6)
                        t__tile_7: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_15, pl.const(4096, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.sub(
                            score_inline771_inline1998__phi_v6, mi_next_inline735_inline2032__tile
                        )
                        beta_inline733_inline2048__tile: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_15, pl.const(4096, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.exp(t__tile_7)
                        t__tile_8: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_24, pl.const(10240, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.mul(
                            alpha_inline734_inline2047__tile, li_inline767_inline2031__iter_v1
                        )
                        li_inline767_inline2031__tile_1: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_9, pl.const(14336, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.add(
                            t__tile_8, beta_inline733_inline2048__tile
                        )
                        t__tile_9: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_17, pl.const(8192, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.mul(
                            oi_inline770_inline2005__iter_v1, alpha_inline734_inline2047__tile
                        )
                        t__tile_10: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_14, pl.const(2048, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.mul(
                            value_inline752_inline2039__phi_v6, beta_inline733_inline2048__tile
                        )
                        oi_inline770_inline2005__tile_1: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_13, pl.const(0, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.add(t__tile_9, t__tile_10)
                        mi_inline768_inline2017__ssa_v3: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_16, pl.const(6144, pl.INT64), 2048), pl.Mem.Vec] = mi_next_inline735_inline2032__tile
                        mi_inline768_inline2017__ssa_v3_mv: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_7, pl.const(12288, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.move(
                            mi_inline768_inline2017__ssa_v3, target_memory=pl.Mem.Vec
                        )
                        li_inline767_inline2031__rv_v2, mi_inline768_inline2017__rv_v2, oi_inline770_inline2005__rv_v2 = pl.yield_(
                            li_inline767_inline2031__tile_1, mi_inline768_inline2017__ssa_v3_mv, oi_inline770_inline2005__tile_1
                        )
                    t__tile_11: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_7, pl.const(12288, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.div(
                        oi_inline770_inline2005__rv_v2, li_inline767_inline2031__rv_v2
                    )
                    pooled_kv_inline553__tile_1: pl.Tensor[[384, 512], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 786432)] = pl.tile.store(
                        t__tile_11, [token_inline755_inline2028__ssa_v0, 0], pooled_kv_inline553__tile
                    )
                    pooled_kv_inline553__phi_v9: pl.Tensor[[384, 512], pl.FP32, pl.MemRef("mem_ddr_44", pl.const(0, pl.INT64), 786432)] = pl.yield_(pooled_kv_inline553__tile_1)
                else:
                    pooled_kv_inline553__phi_v9: pl.Tensor[[384, 512], pl.FP32, pl.MemRef("mem_ddr_44", pl.const(0, pl.INT64), 786432)] = pl.yield_(pooled_kv_inline553__tile)
                pooled_kv_inline553__rv_v4: pl.Tensor[[384, 512], pl.FP32, pl.MemRef("mem_ddr_45", pl.const(0, pl.INT64), 786432)] = pl.yield_(pooled_kv_inline553__phi_v9)
            pooled_kv_inline553__rv_v2: pl.Tensor[[384, 512], pl.FP32, pl.MemRef("mem_ddr_46", pl.const(0, pl.INT64), 786432)] = pl.yield_(pooled_kv_inline553__rv_v4)
        return pooled_kv_inline553__ssa_v0

    @pl.function(type=pl.FunctionType.Spmd)
    def scatter_softmax_pool_spmd(
        self,
        pooled_kv_inline553__ssa_v0: pl.Out[pl.Tensor[[384, 512], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 786432)]],
        b_dim_inline760_inline1995__ssa_v0: pl.Scalar[pl.INDEX],
        pool_workers_inline756_inline1990__ssa_v0: pl.Scalar[pl.INDEX],
        position_ids__ssa_v0: pl.Tensor[[T_DYN], pl.INT32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)],
        s_dim_inline769_inline2026__ssa_v0: pl.Scalar[pl.INDEX],
        cmp4_score_proj_pad_inline758_inline1992__ssa_v0: pl.Tensor[[384, 1024], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 1572864)],
        cmp_ape__ssa_v0: pl.Tensor[[4, 1024], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 16384)],
        cmp4_kv_proj_pad_inline761_inline2009__ssa_v0: pl.Tensor[[384, 1024], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 1572864)],
        compress_state_block_table__ssa_v0: pl.Tensor[[B_DYN, 7], pl.INT32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 0)],
        compress_state_flat_inline751_inline1991__ssa_v0: pl.Tensor[[compress_state_rows_inline745_inline2013__ssa_v0, 2048], pl.FP32, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 0)],
    ) -> pl.Tensor[[384, 512], pl.FP32]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        pooled_kv_inline553__rv_v2: pl.Tensor[[384, 512], pl.FP32, pl.MemRef("mem_ddr_7", pl.const(0, pl.INT64), 786432)] = self.scatter_softmax_pool(
            pooled_kv_inline553__ssa_v0,
            b_dim_inline760_inline1995__ssa_v0,
            pool_workers_inline756_inline1990__ssa_v0,
            position_ids__ssa_v0,
            s_dim_inline769_inline2026__ssa_v0,
            cmp4_score_proj_pad_inline758_inline1992__ssa_v0,
            cmp_ape__ssa_v0,
            cmp4_kv_proj_pad_inline761_inline2009__ssa_v0,
            compress_state_block_table__ssa_v0,
            compress_state_flat_inline751_inline1991__ssa_v0,
            attrs={
                "arg_directions": [pl.adir.output_existing, pl.adir.scalar, pl.adir.scalar, pl.adir.input, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input]
            },
        )
        return pooled_kv_inline553__ssa_v0

    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def scatter_softmax_pool_0(
        pooled_kv_inline471_inline2185__ssa_v0: pl.Out[pl.Tensor[[384, 128], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 196608)]],
        b_dim_inline459_inline2149__ssa_v0: pl.Scalar[pl.INDEX],
        pool_workers_inline477_inline2147__ssa_v0: pl.Scalar[pl.INDEX],
        position_ids__ssa_v0: pl.Tensor[[T_DYN], pl.INT32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)],
        s_dim_inline464_inline2145__ssa_v0: pl.Scalar[pl.INDEX],
        score_proj_pad_inline2156__rv_v2: pl.Tensor[[384, 256], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 393216)],
        inner_ape__ssa_v0: pl.Tensor[[4, 256], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 4096)],
        kv_proj_pad_inline2152__rv_v2: pl.Tensor[[384, 256], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 393216)],
        inner_compress_state_block_table__ssa_v0: pl.Tensor[[B_DYN, 7], pl.INT32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 0)],
        compress_state_flat_inline474_inline2141__ssa_v0: pl.Tensor[[compress_state_rows_inline470_inline2190__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 0)],
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
        pool_worker_inline457_inline2178__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        for c_idx_inline476_inline2172__idx_v0, (pooled_kv_inline471_inline2185__iter_v1,) in pl.range(
            pool_worker_inline457_inline2178__ssa_v0, b_dim_inline459_inline2149__ssa_v0, pool_workers_inline477_inline2147__ssa_v0, init_values=(pooled_kv_inline471_inline2185__ssa_v0,)
        ):
            first_pos_b_inline479_inline2180__tile: pl.Scalar[pl.INT32] = pl.tensor.read(position_ids__ssa_v0, [c_idx_inline476_inline2172__idx_v0 * s_dim_inline464_inline2145__ssa_v0])
            for s_idx_inline456_inline2163__idx_v0, (pooled_kv_inline471_inline2185__iter_v3,) in pl.range(s_dim_inline464_inline2145__ssa_v0, init_values=(pooled_kv_inline471_inline2185__iter_v1,)):
                token_inline454_inline2186__ssa_v0: pl.Scalar[pl.INDEX] = c_idx_inline476_inline2172__idx_v0 * s_dim_inline464_inline2145__ssa_v0 + s_idx_inline456_inline2163__idx_v0
                token_pos_inline478_inline2184__tile: pl.Scalar[pl.INT32] = pl.tensor.read(position_ids__ssa_v0, [token_inline454_inline2186__ssa_v0])
                t__tile: pl.Tile[[1, 128], pl.FP32, pl.MemRef(mem_vec_7, pl.const(1536, pl.INT64), 512), pl.Mem.Vec] = pl.tile.full([1, 128], dtype=pl.FP32, value=0.0)
                pooled_kv_inline471_inline2185__tile: pl.Tensor[[384, 128], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 196608)] = pl.tile.store(
                    t__tile, [token_inline454_inline2186__ssa_v0, 0], pooled_kv_inline471_inline2185__iter_v3
                )
                if (pl.cast(token_pos_inline478_inline2184__tile, pl.INDEX) + 1) % 4 == 0:
                    window_start_inline504_inline2187__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(token_pos_inline478_inline2184__tile, pl.INDEX) - 8 + 1
                    for h0_inline503_inline2191__idx_v0, (pooled_kv_inline471_inline2185__iter_v6,) in pl.range(0, 128, 64, init_values=(pooled_kv_inline471_inline2185__tile,)):
                        last_ape_row_inline475_inline2189__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(pl.cast(token_pos_inline478_inline2184__tile, pl.INDEX) % 4, pl.INDEX)
                        t__tile_1: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_7, pl.const(1536, pl.INT64), 256), pl.Mem.Vec] = pl.tile.load(
                            score_proj_pad_inline2156__rv_v2, [token_inline454_inline2186__ssa_v0, h0_inline503_inline2191__idx_v0 + 128], [1, 64], [1, 64], target_memory=pl.Mem.Vec
                        )
                        t__tile_2: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_9, pl.const(2048, pl.INT64), 256), pl.Mem.Vec] = pl.tile.load(
                            inner_ape__ssa_v0, [last_ape_row_inline475_inline2189__ssa_v0, h0_inline503_inline2191__idx_v0 + 128], [1, 64], [1, 64], target_memory=pl.Mem.Vec
                        )
                        mi_inline448_inline2182__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_7, pl.const(1536, pl.INT64), 256), pl.Mem.Vec] = pl.tile.add(t__tile_1, t__tile_2)
                        t__tile_3: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_9, pl.const(2048, pl.INT64), 256), pl.Mem.Vec] = pl.tile.sub(
                            mi_inline448_inline2182__tile, mi_inline448_inline2182__tile
                        )
                        li_inline453_inline2192__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_9, pl.const(2048, pl.INT64), 256), pl.Mem.Vec] = pl.tile.exp(t__tile_3)
                        oi_inline446_inline2195__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(0, pl.INT64), 256), pl.Mem.Vec] = pl.tile.load(
                            kv_proj_pad_inline2152__rv_v2, [token_inline454_inline2186__ssa_v0, h0_inline503_inline2191__idx_v0 + 128], [1, 64], [1, 64], target_memory=pl.Mem.Vec
                        )
                        for state_idx_inline465_inline2196__idx_v0, (li_inline453_inline2192__iter_v1, mi_inline448_inline2182__iter_v1, oi_inline446_inline2195__iter_v1) in pl.range(
                            7, init_values=(li_inline453_inline2192__tile, mi_inline448_inline2182__tile, oi_inline446_inline2195__tile)
                        ):
                            logical_pos_inline473_inline2164__ssa_v0: pl.Scalar[pl.INDEX] = window_start_inline504_inline2187__ssa_v0 + state_idx_inline465_inline2196__idx_v0
                            value_inline447_inline2197__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_14, pl.const(256, pl.INT64), 256), pl.Mem.Vec] = pl.tile.full(
                                [1, 64], dtype=pl.FP32, value=0.0
                            )
                            score_inline444_inline2198__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_15, pl.const(512, pl.INT64), 256), pl.Mem.Vec] = pl.tile.full(
                                [1, 64], dtype=pl.FP32, value=-3.4028234663852886e38
                            )
                            state_half_inline450_inline2203__ssa_v0: pl.Scalar[pl.INDEX] = 0
                            if 4 <= state_idx_inline465_inline2196__idx_v0:
                                state_half_inline450_inline2203__ssa_v1: pl.Scalar[pl.INDEX] = 128
                                state_half_inline450_inline2203__phi_v2: pl.Scalar[pl.INDEX] = pl.yield_(state_half_inline450_inline2203__ssa_v1)
                            else:
                                state_half_inline450_inline2203__phi_v2: pl.Scalar[pl.INDEX] = pl.yield_(state_half_inline450_inline2203__ssa_v0)
                            if 0 <= logical_pos_inline473_inline2164__ssa_v0 and logical_pos_inline473_inline2164__ssa_v0 < pl.cast(first_pos_b_inline479_inline2180__tile, pl.INDEX):
                                ring_row_inline466_inline2159__ssa_v0: pl.Scalar[pl.INDEX] = logical_pos_inline473_inline2164__ssa_v0 % 14
                                state_page_off_inline483_inline2204__ssa_v0: pl.Scalar[pl.INDEX] = ring_row_inline466_inline2159__ssa_v0 // 2
                                state_blk_id_i32_inline484_inline2194__tile: pl.Scalar[pl.INT32] = pl.tensor.read(
                                    inner_compress_state_block_table__ssa_v0, [c_idx_inline476_inline2172__idx_v0, state_page_off_inline483_inline2204__ssa_v0]
                                )
                                if 0 <= pl.cast(state_blk_id_i32_inline484_inline2194__tile, pl.INDEX):
                                    state_blk_id_inline488_inline2165__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(state_blk_id_i32_inline484_inline2194__tile, pl.INDEX)
                                    state_intra_row_inline469_inline2201__ssa_v0: pl.Scalar[pl.INDEX] = ring_row_inline466_inline2159__ssa_v0 % 2
                                    state_row_inline481_inline2188__ssa_v0: pl.Scalar[pl.INDEX] = state_blk_id_inline488_inline2165__ssa_v0 * 2 + state_intra_row_inline469_inline2201__ssa_v0
                                    value_inline447_inline2197__tile_1: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_16, pl.const(768, pl.INT64), 256), pl.Mem.Vec] = pl.tile.load(
                                        compress_state_flat_inline474_inline2141__ssa_v0,
                                        [state_row_inline481_inline2188__ssa_v0, state_half_inline450_inline2203__phi_v2 + h0_inline503_inline2191__idx_v0],
                                        [1, 64],
                                        [1, 64],
                                        target_memory=pl.Mem.Vec,
                                    )
                                    score_inline444_inline2198__tile_1: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_17, pl.const(1024, pl.INT64), 256), pl.Mem.Vec] = pl.tile.load(
                                        compress_state_flat_inline474_inline2141__ssa_v0,
                                        [state_row_inline481_inline2188__ssa_v0, state_half_inline450_inline2203__phi_v2 + h0_inline503_inline2191__idx_v0 + 256],
                                        [1, 64],
                                        [1, 64],
                                        target_memory=pl.Mem.Vec,
                                    )
                                    score_inline444_inline2198__phi_v2, value_inline447_inline2197__phi_v2 = pl.yield_(score_inline444_inline2198__tile_1, value_inline447_inline2197__tile_1)
                                else:
                                    score_inline444_inline2198__tile_mv: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_17, pl.const(1024, pl.INT64), 256), pl.Mem.Vec] = pl.tile.move(
                                        score_inline444_inline2198__tile, target_memory=pl.Mem.Vec
                                    )
                                    value_inline447_inline2197__tile_mv: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_16, pl.const(768, pl.INT64), 256), pl.Mem.Vec] = pl.tile.move(
                                        value_inline447_inline2197__tile, target_memory=pl.Mem.Vec
                                    )
                                    score_inline444_inline2198__phi_v2, value_inline447_inline2197__phi_v2 = pl.yield_(score_inline444_inline2198__tile_mv, value_inline447_inline2197__tile_mv)
                                score_inline444_inline2198__phi_v3, value_inline447_inline2197__phi_v3 = pl.yield_(score_inline444_inline2198__phi_v2, value_inline447_inline2197__phi_v2)
                            else:
                                score_inline444_inline2198__tile_mv_1: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_17, pl.const(1024, pl.INT64), 256), pl.Mem.Vec] = pl.tile.move(
                                    score_inline444_inline2198__tile, target_memory=pl.Mem.Vec
                                )
                                value_inline447_inline2197__tile_mv_1: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_16, pl.const(768, pl.INT64), 256), pl.Mem.Vec] = pl.tile.move(
                                    value_inline447_inline2197__tile, target_memory=pl.Mem.Vec
                                )
                                score_inline444_inline2198__phi_v3, value_inline447_inline2197__phi_v3 = pl.yield_(score_inline444_inline2198__tile_mv_1, value_inline447_inline2197__tile_mv_1)
                            if pl.cast(first_pos_b_inline479_inline2180__tile, pl.INDEX) <= logical_pos_inline473_inline2164__ssa_v0:
                                if logical_pos_inline473_inline2164__ssa_v0 <= pl.cast(token_pos_inline478_inline2184__tile, pl.INDEX):
                                    overlay_token_inline467_inline2202__ssa_v0: pl.Scalar[pl.INDEX] = (
                                        c_idx_inline476_inline2172__idx_v0 * s_dim_inline464_inline2145__ssa_v0
                                        + logical_pos_inline473_inline2164__ssa_v0
                                        - pl.cast(first_pos_b_inline479_inline2180__tile, pl.INDEX)
                                    )
                                    ape_row_inline491_inline2206__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(logical_pos_inline473_inline2164__ssa_v0 % 4, pl.INDEX)
                                    value_inline447_inline2197__tile_2: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_14, pl.const(256, pl.INT64), 256), pl.Mem.Vec] = pl.tile.load(
                                        kv_proj_pad_inline2152__rv_v2,
                                        [overlay_token_inline467_inline2202__ssa_v0, state_half_inline450_inline2203__phi_v2 + h0_inline503_inline2191__idx_v0],
                                        [1, 64],
                                        [1, 64],
                                        target_memory=pl.Mem.Vec,
                                    )
                                    t__tile_4: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_15, pl.const(512, pl.INT64), 256), pl.Mem.Vec] = pl.tile.load(
                                        score_proj_pad_inline2156__rv_v2,
                                        [overlay_token_inline467_inline2202__ssa_v0, state_half_inline450_inline2203__phi_v2 + h0_inline503_inline2191__idx_v0],
                                        [1, 64],
                                        [1, 64],
                                        target_memory=pl.Mem.Vec,
                                    )
                                    t__tile_5: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_24, pl.const(1280, pl.INT64), 256), pl.Mem.Vec] = pl.tile.load(
                                        inner_ape__ssa_v0,
                                        [ape_row_inline491_inline2206__ssa_v0, state_half_inline450_inline2203__phi_v2 + h0_inline503_inline2191__idx_v0],
                                        [1, 64],
                                        [1, 64],
                                        target_memory=pl.Mem.Vec,
                                    )
                                    score_inline444_inline2198__tile_2: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_15, pl.const(512, pl.INT64), 256), pl.Mem.Vec] = pl.tile.add(t__tile_4, t__tile_5)
                                    score_inline444_inline2198__phi_v5, value_inline447_inline2197__phi_v5 = pl.yield_(score_inline444_inline2198__tile_2, value_inline447_inline2197__tile_2)
                                else:
                                    score_inline444_inline2198__phi_v3_mv: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_15, pl.const(512, pl.INT64), 256), pl.Mem.Vec] = pl.tile.move(
                                        score_inline444_inline2198__phi_v3, target_memory=pl.Mem.Vec
                                    )
                                    value_inline447_inline2197__phi_v3_mv: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_14, pl.const(256, pl.INT64), 256), pl.Mem.Vec] = pl.tile.move(
                                        value_inline447_inline2197__phi_v3, target_memory=pl.Mem.Vec
                                    )
                                    score_inline444_inline2198__phi_v5, value_inline447_inline2197__phi_v5 = pl.yield_(score_inline444_inline2198__phi_v3_mv, value_inline447_inline2197__phi_v3_mv)
                                score_inline444_inline2198__phi_v6, value_inline447_inline2197__phi_v6 = pl.yield_(score_inline444_inline2198__phi_v5, value_inline447_inline2197__phi_v5)
                            else:
                                score_inline444_inline2198__phi_v3_mv_1: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_15, pl.const(512, pl.INT64), 256), pl.Mem.Vec] = pl.tile.move(
                                    score_inline444_inline2198__phi_v3, target_memory=pl.Mem.Vec
                                )
                                value_inline447_inline2197__phi_v3_mv_1: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_14, pl.const(256, pl.INT64), 256), pl.Mem.Vec] = pl.tile.move(
                                    value_inline447_inline2197__phi_v3, target_memory=pl.Mem.Vec
                                )
                                score_inline444_inline2198__phi_v6, value_inline447_inline2197__phi_v6 = pl.yield_(score_inline444_inline2198__phi_v3_mv_1, value_inline447_inline2197__phi_v3_mv_1)
                            mi_next_inline489_inline2143__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_16, pl.const(768, pl.INT64), 256), pl.Mem.Vec] = pl.tile.maximum(
                                mi_inline448_inline2182__iter_v1, score_inline444_inline2198__phi_v6
                            )
                            t__tile_6: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_17, pl.const(1024, pl.INT64), 256), pl.Mem.Vec] = pl.tile.sub(
                                mi_inline448_inline2182__iter_v1, mi_next_inline489_inline2143__tile
                            )
                            alpha_inline487_inline2183__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_17, pl.const(1024, pl.INT64), 256), pl.Mem.Vec] = pl.tile.exp(t__tile_6)
                            t__tile_7: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_15, pl.const(512, pl.INT64), 256), pl.Mem.Vec] = pl.tile.sub(
                                score_inline444_inline2198__phi_v6, mi_next_inline489_inline2143__tile
                            )
                            beta_inline490_inline2207__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_15, pl.const(512, pl.INT64), 256), pl.Mem.Vec] = pl.tile.exp(t__tile_7)
                            t__tile_8: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_24, pl.const(1280, pl.INT64), 256), pl.Mem.Vec] = pl.tile.mul(
                                alpha_inline487_inline2183__tile, li_inline453_inline2192__iter_v1
                            )
                            li_inline453_inline2192__tile_1: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_9, pl.const(2048, pl.INT64), 256), pl.Mem.Vec] = pl.tile.add(
                                t__tile_8, beta_inline490_inline2207__tile
                            )
                            t__tile_9: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_17, pl.const(1024, pl.INT64), 256), pl.Mem.Vec] = pl.tile.mul(
                                oi_inline446_inline2195__iter_v1, alpha_inline487_inline2183__tile
                            )
                            t__tile_10: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_14, pl.const(256, pl.INT64), 256), pl.Mem.Vec] = pl.tile.mul(
                                value_inline447_inline2197__phi_v6, beta_inline490_inline2207__tile
                            )
                            oi_inline446_inline2195__tile_1: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(0, pl.INT64), 256), pl.Mem.Vec] = pl.tile.add(t__tile_9, t__tile_10)
                            mi_inline448_inline2182__ssa_v3: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_16, pl.const(768, pl.INT64), 256), pl.Mem.Vec] = mi_next_inline489_inline2143__tile
                            mi_inline448_inline2182__ssa_v3_mv: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_7, pl.const(1536, pl.INT64), 256), pl.Mem.Vec] = pl.tile.move(
                                mi_inline448_inline2182__ssa_v3, target_memory=pl.Mem.Vec
                            )
                            li_inline453_inline2192__rv_v2, mi_inline448_inline2182__rv_v2, oi_inline446_inline2195__rv_v2 = pl.yield_(
                                li_inline453_inline2192__tile_1, mi_inline448_inline2182__ssa_v3_mv, oi_inline446_inline2195__tile_1
                            )
                        t__tile_11: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_7, pl.const(1536, pl.INT64), 256), pl.Mem.Vec] = pl.tile.div(
                            oi_inline446_inline2195__rv_v2, li_inline453_inline2192__rv_v2
                        )
                        pooled_kv_inline471_inline2185__tile_1: pl.Tensor[[384, 128], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 196608)] = pl.tile.store(
                            t__tile_11, [token_inline454_inline2186__ssa_v0, h0_inline503_inline2191__idx_v0], pooled_kv_inline471_inline2185__iter_v6
                        )
                        pooled_kv_inline471_inline2185__rv_v7: pl.Tensor[[384, 128], pl.FP32, pl.MemRef("mem_ddr_44", pl.const(0, pl.INT64), 196608)] = pl.yield_(
                            pooled_kv_inline471_inline2185__tile_1
                        )
                    pooled_kv_inline471_inline2185__phi_v9: pl.Tensor[[384, 128], pl.FP32, pl.MemRef("mem_ddr_45", pl.const(0, pl.INT64), 196608)] = pl.yield_(pooled_kv_inline471_inline2185__rv_v7)
                else:
                    pooled_kv_inline471_inline2185__phi_v9: pl.Tensor[[384, 128], pl.FP32, pl.MemRef("mem_ddr_45", pl.const(0, pl.INT64), 196608)] = pl.yield_(pooled_kv_inline471_inline2185__tile)
                pooled_kv_inline471_inline2185__rv_v4: pl.Tensor[[384, 128], pl.FP32, pl.MemRef("mem_ddr_46", pl.const(0, pl.INT64), 196608)] = pl.yield_(pooled_kv_inline471_inline2185__phi_v9)
            pooled_kv_inline471_inline2185__rv_v2: pl.Tensor[[384, 128], pl.FP32, pl.MemRef("mem_ddr_47", pl.const(0, pl.INT64), 196608)] = pl.yield_(pooled_kv_inline471_inline2185__rv_v4)
        return pooled_kv_inline471_inline2185__ssa_v0

    @pl.function(type=pl.FunctionType.Spmd)
    def scatter_softmax_pool_spmd_0(
        self,
        pooled_kv_inline471_inline2185__ssa_v0: pl.Out[pl.Tensor[[384, 128], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 196608)]],
        b_dim_inline459_inline2149__ssa_v0: pl.Scalar[pl.INDEX],
        pool_workers_inline477_inline2147__ssa_v0: pl.Scalar[pl.INDEX],
        position_ids__ssa_v0: pl.Tensor[[T_DYN], pl.INT32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)],
        s_dim_inline464_inline2145__ssa_v0: pl.Scalar[pl.INDEX],
        score_proj_pad_inline2156__rv_v2: pl.Tensor[[384, 256], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 393216)],
        inner_ape__ssa_v0: pl.Tensor[[4, 256], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 4096)],
        kv_proj_pad_inline2152__rv_v2: pl.Tensor[[384, 256], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 393216)],
        inner_compress_state_block_table__ssa_v0: pl.Tensor[[B_DYN, 7], pl.INT32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 0)],
        compress_state_flat_inline474_inline2141__ssa_v0: pl.Tensor[[compress_state_rows_inline470_inline2190__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 0)],
    ) -> pl.Tensor[[384, 128], pl.FP32]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        pooled_kv_inline471_inline2185__rv_v2: pl.Tensor[[384, 128], pl.FP32, pl.MemRef("mem_ddr_7", pl.const(0, pl.INT64), 196608)] = self.scatter_softmax_pool_0(
            pooled_kv_inline471_inline2185__ssa_v0,
            b_dim_inline459_inline2149__ssa_v0,
            pool_workers_inline477_inline2147__ssa_v0,
            position_ids__ssa_v0,
            s_dim_inline464_inline2145__ssa_v0,
            score_proj_pad_inline2156__rv_v2,
            inner_ape__ssa_v0,
            kv_proj_pad_inline2152__rv_v2,
            inner_compress_state_block_table__ssa_v0,
            compress_state_flat_inline474_inline2141__ssa_v0,
            attrs={
                "arg_directions": [pl.adir.output_existing, pl.adir.scalar, pl.adir.scalar, pl.adir.input, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input]
            },
        )
        return pooled_kv_inline471_inline2185__ssa_v0

    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def weights_proj_reduce(
        weights_partial_inline2406__rv_v2: pl.Tensor[[384, 64], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 98304)],
        weights_inline2424__ssa_v0: pl.Out[pl.Tensor[[384, 64], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 98304)]],
    ) -> pl.Tensor[[384, 64], pl.FP32]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        mem_vec_2: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 4096)
        mem_vec_3: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 2048)
        w_rb_inline2377__ssa_v1: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        w_r0_inline2401__ssa_v1: pl.Scalar[pl.INDEX] = w_rb_inline2377__ssa_v1 * 16
        w_sum_inline2387__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_2, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.load(
            weights_partial_inline2406__rv_v2, [w_r0_inline2401__ssa_v1, 0], [16, 64], [16, 64], target_memory=pl.Mem.Vec
        )
        t__tile: pl.Tile[[16, 64], pl.BF16, pl.MemRef(mem_vec_3, pl.const(4096, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(w_sum_inline2387__tile, target_type=pl.BF16, mode="rint")
        w_sum_v1_inline2410__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_2, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(t__tile, target_type=pl.FP32, mode="round")
        t__tile_1: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_2, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.muls(w_sum_v1_inline2410__tile, 0.011048543456039806)
        w_scaled_inline2393__tile: pl.Tile[[16, 64], pl.BF16, pl.MemRef(mem_vec_3, pl.const(4096, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(t__tile_1, target_type=pl.BF16, mode="rint")
        t__cast_fp32_tmp_v0: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_2, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(w_scaled_inline2393__tile, target_type=pl.FP32, mode="round")
        t__tile_2: pl.Tile[[16, 64], pl.FP16, pl.MemRef(mem_vec_3, pl.const(4096, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(t__cast_fp32_tmp_v0, target_type=pl.FP16, mode="round")
        t__tile_3: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_2, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(t__tile_2, target_type=pl.FP32, mode="round")
        weights_inline2424__tile: pl.Tensor[[384, 64], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 98304)] = pl.tile.store(
            t__tile_3, [w_r0_inline2401__ssa_v1, 0], weights_inline2424__ssa_v0
        )
        return weights_inline2424__ssa_v0

    @pl.function(type=pl.FunctionType.Spmd)
    def weights_proj_reduce_spmd(
        self,
        weights_partial_inline2406__rv_v2: pl.Tensor[[384, 64], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 98304)],
        weights_inline2424__ssa_v0: pl.Out[pl.Tensor[[384, 64], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 98304)]],
    ) -> pl.Tensor[[384, 64], pl.FP32]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        weights_inline2424__ssa_v1: pl.Tensor[[384, 64], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 98304)] = self.weights_proj_reduce(
            weights_partial_inline2406__rv_v2, weights_inline2424__ssa_v0, attrs={"arg_directions": [pl.adir.input, pl.adir.output_existing]}
        )
        return weights_inline2424__ssa_v0

    @pl.function(type=pl.FunctionType.AIC, level=pl.Level.AIC, role=pl.Role.SubWorker)
    def weights_proj(
        weights_partial_inline2406__ssa_v0: pl.Out[pl.Tensor[[384, 64], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 98304)]],
        row_blocks_inline2444__ssa_v0: pl.Scalar[pl.INDEX],
        bs_inline2408__ssa_v0: pl.Scalar[pl.INDEX],
        x_flat_inline2399__ssa_v0: pl.Tensor[[T_DYN, 4096], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)],
        weights_proj__ssa_v0: pl.Tensor[[4096, 64], pl.BF16, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 524288)],
    ) -> pl.Tensor[[384, 64], pl.FP32]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        mem_acc_3: pl.Ptr = pl.tile.alloc(pl.Mem.Acc, 4096)
        mem_mat_4: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 16384)
        mem_mat_5: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 65536)
        mem_left_6: pl.Ptr = pl.tile.alloc(pl.Mem.Left, 8192)
        mem_right_7: pl.Ptr = pl.tile.alloc(pl.Mem.Right, 32768)
        mem_left_8: pl.Ptr = pl.tile.alloc(pl.Mem.Left, 8192)
        mem_right_9: pl.Ptr = pl.tile.alloc(pl.Mem.Right, 32768)
        w_worker_inline2390__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        for w_unit_inline2382__idx_v0, (weights_partial_inline2406__iter_v1,) in pl.range(
            w_worker_inline2390__ssa_v0, row_blocks_inline2444__ssa_v0, 8, init_values=(weights_partial_inline2406__ssa_v0,)
        ):
            w_rb_inline2377__ssa_v0: pl.Scalar[pl.INDEX] = w_unit_inline2382__idx_v0
            kb_inline2413__ssa_v0: pl.Scalar[pl.INDEX] = w_unit_inline2382__idx_v0 - w_rb_inline2377__ssa_v0
            w_r0_inline2401__ssa_v0: pl.Scalar[pl.INDEX] = w_rb_inline2377__ssa_v0 * 16
            w_rows_inline2402__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(bs_inline2408__ssa_v0 - w_r0_inline2401__ssa_v0, 16)
            k_base_inline2375__ssa_v0: pl.Scalar[pl.INDEX] = kb_inline2413__ssa_v0 * 4096
            weights_acc_inline2396__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 4096), pl.Mem.Acc] = pl.tile.create([16, 64], dtype=pl.FP32, target_memory=pl.Mem.Acc)
            for db_inline2430__idx_v0, (weights_acc_inline2396__iter_v1,) in pl.range(8, init_values=(weights_acc_inline2396__tile,)):
                d0_inline2386__ssa_v0: pl.Scalar[pl.INDEX] = k_base_inline2375__ssa_v0 + db_inline2430__idx_v0 * 512
                x_tile_inline2443__tile: pl.Tile[[16, 512], pl.BF16, pl.MemRef(mem_mat_4, pl.const(0, pl.INT64), 16384), pl.Mem.Mat, pl.TileView(valid_shape=[w_rows_inline2402__ssa_v0, 512])] = (
                    pl.tile.load(x_flat_inline2399__ssa_v0, [w_r0_inline2401__ssa_v0, d0_inline2386__ssa_v0], [16, 512], [w_rows_inline2402__ssa_v0, 512], target_memory=pl.Mem.Mat)
                )
                weights_proj_tile_inline2388__tile: pl.Tile[[512, 64], pl.BF16, pl.MemRef(mem_mat_5, pl.const(16384, pl.INT64), 65536), pl.Mem.Mat] = pl.tile.load(
                    weights_proj__ssa_v0, [d0_inline2386__ssa_v0, 0], [512, 64], [512, 64], target_memory=pl.Mem.Mat
                )
                weights_acc_inline2396__tile_l0_a: pl.Tile[
                    [16, 256],
                    pl.BF16,
                    pl.MemRef(mem_left_6, pl.const(0, pl.INT64), 8192),
                    pl.Mem.Left,
                    pl.TileView(valid_shape=[w_rows_inline2402__ssa_v0, 256], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal),
                ] = pl.tile.extract(x_tile_inline2443__tile, 0, 0, [16, 256], target_memory=pl.Mem.Left)
                weights_acc_inline2396__tile_l0_b: pl.Tile[[256, 64], pl.BF16, pl.MemRef(mem_right_7, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                    weights_proj_tile_inline2388__tile, 0, 0, [256, 64], target_memory=pl.Mem.Right
                )
                weights_acc_inline2396__tile_l0_a_1: pl.Tile[
                    [16, 256],
                    pl.BF16,
                    pl.MemRef(mem_left_8, pl.const(8192, pl.INT64), 8192),
                    pl.Mem.Left,
                    pl.TileView(valid_shape=[w_rows_inline2402__ssa_v0, 256], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal),
                ] = pl.tile.extract(x_tile_inline2443__tile, 0, 256, [16, 256], target_memory=pl.Mem.Left)
                weights_acc_inline2396__tile_l0_b_1: pl.Tile[[256, 64], pl.BF16, pl.MemRef(mem_right_9, pl.const(32768, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                    weights_proj_tile_inline2388__tile, 256, 0, [256, 64], target_memory=pl.Mem.Right
                )
                weights_acc_inline2396__tile_l0_c_acc: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 4096), pl.Mem.Acc] = pl.tile.matmul_acc(
                    weights_acc_inline2396__iter_v1, weights_acc_inline2396__tile_l0_a, weights_acc_inline2396__tile_l0_b, db_inline2430__idx_v0 == 0
                )
                weights_acc_inline2396__tile_l0_c_acc_1: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 4096), pl.Mem.Acc] = pl.tile.matmul_acc(
                    weights_acc_inline2396__tile_l0_c_acc, weights_acc_inline2396__tile_l0_a_1, weights_acc_inline2396__tile_l0_b_1, False
                )
                weights_acc_inline2396__rv_v2: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 4096), pl.Mem.Acc] = pl.yield_(weights_acc_inline2396__tile_l0_c_acc_1)
            weights_partial_inline2406__tile: pl.Tensor[[384, 64], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 98304)] = pl.tile.store(
                weights_acc_inline2396__rv_v2, [kb_inline2413__ssa_v0 * 384 + w_r0_inline2401__ssa_v0, 0], weights_partial_inline2406__iter_v1
            )
            weights_partial_inline2406__rv_v2: pl.Tensor[[384, 64], pl.FP32, pl.MemRef("mem_ddr_12", pl.const(0, pl.INT64), 98304)] = pl.yield_(weights_partial_inline2406__tile)
        return weights_partial_inline2406__ssa_v0

    @pl.function(type=pl.FunctionType.Spmd)
    def weights_proj_spmd(
        self,
        weights_partial_inline2406__ssa_v0: pl.Out[pl.Tensor[[384, 64], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 98304)]],
        row_blocks_inline2444__ssa_v0: pl.Scalar[pl.INDEX],
        bs_inline2408__ssa_v0: pl.Scalar[pl.INDEX],
        x_flat_inline2399__ssa_v0: pl.Tensor[[T_DYN, 4096], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)],
        weights_proj__ssa_v0: pl.Tensor[[4096, 64], pl.BF16, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 524288)],
    ) -> pl.Tensor[[384, 64], pl.FP32]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        weights_partial_inline2406__rv_v2: pl.Tensor[[384, 64], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 98304)] = self.weights_proj(
            weights_partial_inline2406__ssa_v0,
            row_blocks_inline2444__ssa_v0,
            bs_inline2408__ssa_v0,
            x_flat_inline2399__ssa_v0,
            weights_proj__ssa_v0,
            attrs={"arg_directions": [pl.adir.output_existing, pl.adir.scalar, pl.adir.scalar, pl.adir.input, pl.adir.input]},
        )
        return weights_partial_inline2406__ssa_v0

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
            t_dim_inline546__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(x_normed_t__ssa_v0, 0)
            q_rope_cos_il_inline545__ssa_v0: pl.Tensor[[t_dim_inline546__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_66", pl.const(0, pl.INT64), 0)] = pl.tensor.create(
                [t_dim_inline546__ssa_v0, 64], dtype=pl.FP32, layout=pl.TensorLayout.ND
            )
            q_rope_sin_signed_inline544__ssa_v0: pl.Tensor[[t_dim_inline546__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_67", pl.const(0, pl.INT64), 0)] = pl.tensor.create(
                [t_dim_inline546__ssa_v0, 64], dtype=pl.FP32, layout=pl.TensorLayout.ND
            )
            q_rope_swap_idx_inline543__ssa_v0: pl.Tensor[[t_dim_inline546__ssa_v0, 64], pl.INT32, pl.MemRef("mem_ddr_68", pl.const(0, pl.INT64), 0)] = pl.tensor.create(
                [t_dim_inline546__ssa_v0, 64], dtype=pl.INT32, layout=pl.TensorLayout.ND
            )
            t_dim_inline1855__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(freqs_cos__ssa_v0, 0)
            rope_cos_view_inline1854__ssa_v0: pl.Tensor[[t_dim_inline1855__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_7", pl.const(0, pl.INT64), 0)] = pl.tensor.reshape(
                freqs_cos__ssa_v0, [t_dim_inline1855__ssa_v0, 64]
            )
            rope_sin_view_inline1842__ssa_v0: pl.Tensor[[t_dim_inline1855__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_8", pl.const(0, pl.INT64), 0)] = pl.tensor.reshape(
                freqs_sin__ssa_v0, [t_dim_inline1855__ssa_v0, 64]
            )
            rope_cos_il_view_inline1852__ssa_v0: pl.Tensor[[t_dim_inline1855__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_66", pl.const(0, pl.INT64), 0)] = pl.tensor.reshape(
                q_rope_cos_il_inline545__ssa_v0, [t_dim_inline1855__ssa_v0, 64]
            )
            rope_sin_signed_view_inline1847__ssa_v0: pl.Tensor[[t_dim_inline1855__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_67", pl.const(0, pl.INT64), 0)] = pl.tensor.reshape(
                q_rope_sin_signed_inline544__ssa_v0, [t_dim_inline1855__ssa_v0, 64]
            )
            rope_swap_idx_view_inline1848__ssa_v0: pl.Tensor[[t_dim_inline1855__ssa_v0, 64], pl.INT32, pl.MemRef("mem_ddr_68", pl.const(0, pl.INT64), 0)] = pl.tensor.reshape(
                q_rope_swap_idx_inline543__ssa_v0, [t_dim_inline1855__ssa_v0, 64]
            )
            token_tiles_inline1841__ssa_v0: pl.Scalar[pl.INDEX] = (t_dim_inline1855__ssa_v0 + 7) // 8
            ret__tmp_v0_1: pl.Tuple[pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(
                self.q_rope_prepare_spmd,
                rope_cos_il_view_inline1852__ssa_v0,
                rope_sin_signed_view_inline1847__ssa_v0,
                rope_swap_idx_view_inline1848__ssa_v0,
                token_tiles_inline1841__ssa_v0,
                t_dim_inline1855__ssa_v0,
                rope_cos_view_inline1854__ssa_v0,
                rope_sin_view_inline1842__ssa_v0,
                core_num=pl.min(token_tiles_inline1841__ssa_v0, 48),
                allow_early_resolve=True,
                attrs={"arg_directions": [pl.adir.output_existing, pl.adir.output_existing, pl.adir.output_existing, pl.adir.scalar, pl.adir.scalar, pl.adir.input, pl.adir.input]},
            )
            tid__ssa_v1: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_1[0]
            qr_i8_matmul_inline1874__ssa_v0: pl.Tensor[[512, 1024], pl.INT8, pl.MemRef("mem_ddr_69", pl.const(0, pl.INT64), 524288)] = pl.tensor.create(
                [512, 1024], dtype=pl.INT8, layout=pl.TensorLayout.ND
            )
            qr_scale_pad_store_inline1879__ssa_v0: pl.Tensor[[512, 1], pl.FP32, pl.MemRef("mem_ddr_70", pl.const(0, pl.INT64), 2048)] = pl.tensor.create(
                [512, 1], dtype=pl.FP32, layout=pl.TensorLayout.ND, manual_dep=True
            )
            t_dim_inline308_inline1877__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(x_normed_t__ssa_v0, 0)
            for tile_base_inline307_inline1878__idx_v0, (qr_i8_matmul_inline1874__iter_v1, qr_scale_pad_store_inline1879__iter_v1) in pl.range(
                0,
                t_dim_inline308_inline1877__ssa_v0,
                512,
                init_values=(qr_i8_matmul_inline1874__ssa_v0, qr_scale_pad_store_inline1879__ssa_v0),
                attrs={"iter_arg_rebind_0": False, "iter_arg_rebind_1": False},
            ):
                tile_rows_inline306_inline1880__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(t_dim_inline308_inline1877__ssa_v0 - tile_base_inline307_inline1878__idx_v0, 512)
                with pl.scope():
                    qr_t_matmul_inline305_inline1881__ssa_v0: pl.Scalar[pl.INDEX] = (tile_rows_inline306_inline1880__ssa_v0 + 15) // 16 * 16
                    qr_fp32_inline304_inline1875__ssa_v0: pl.Tensor[[qr_t_matmul_inline305_inline1881__ssa_v0, 1024], pl.FP32, pl.MemRef("mem_ddr_71", pl.const(0, pl.INT64), 0)] = pl.tensor.create(
                        [qr_t_matmul_inline305_inline1881__ssa_v0, 1024], dtype=pl.FP32, layout=pl.TensorLayout.ND
                    )
                    x_view_inline2843__ssa_v0: pl.Tensor[[pl.tensor.dim(x_normed_t__ssa_v0, pl.const(0, pl.INDEX)), 4096], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)] = (
                        pl.tensor.reshape(x_normed_t__ssa_v0, [pl.tensor.dim(x_normed_t__ssa_v0, 0), 4096])
                    )
                    qr_t_matmul_inline2845__ssa_v0: pl.Scalar[pl.INDEX] = (tile_rows_inline306_inline1880__ssa_v0 + 15) // 16 * 16
                    qr_full_rows_inline2846__ssa_v0: pl.Scalar[pl.INDEX] = tile_rows_inline306_inline1880__ssa_v0 // 64 * 64
                    qr_fp32_inline304_inline1875__rv_v2: pl.Tensor[[qr_t_matmul_inline305_inline1881__ssa_v0, 1024], pl.FP32, pl.MemRef("mem_ddr_72", pl.const(0, pl.INT64), 0)] = self.qr_proj_seed(
                        qr_fp32_inline304_inline1875__ssa_v0, qr_t_matmul_inline2845__ssa_v0, attrs={"arg_directions": [pl.adir.inout, pl.adir.scalar]}
                    )
                    ret__tmp_v0_2: pl.Tuple[pl.Tensor[[qr_t_matmul_inline305_inline1881__ssa_v0, 1024], pl.FP32], pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(
                        self.qr_proj_matmul_spmd,
                        qr_fp32_inline304_inline1875__rv_v2,
                        qr_full_rows_inline2846__ssa_v0,
                        tile_base_inline307_inline1878__idx_v0,
                        x_view_inline2843__ssa_v0,
                        wq_a__ssa_v0,
                        qr_t_matmul_inline2845__ssa_v0,
                        tile_rows_inline306_inline1880__ssa_v0,
                        core_num=8,
                        allow_early_resolve=True,
                        attrs={"arg_directions": [pl.adir.inout, pl.adir.scalar, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.scalar, pl.adir.scalar]},
                    )
                    qr_fp32_inline304_inline1875__rv_v10: pl.Tensor[[qr_t_matmul_inline305_inline1881__ssa_v0, 1024], pl.FP32, pl.MemRef("mem_ddr_73", pl.const(0, pl.INT64), 0)] = ret__tmp_v0_2[0]
                    tid__ssa_v2: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_2[1]
                    t_dim_inline2888__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(qr__ssa_v0, 0)
                    qr_view_inline2890__ssa_v0: pl.Tensor[[t_dim_inline2888__ssa_v0, 1024], pl.INT8, pl.MemRef("mem_ddr_64", pl.const(0, pl.INT64), 0)] = pl.tensor.reshape(
                        qr__ssa_v0, [t_dim_inline2888__ssa_v0, 1024]
                    )
                    qr_scale_view_inline2896__ssa_v0: pl.Tensor[[t_dim_inline2888__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_65", pl.const(0, pl.INT64), 0)] = pl.tensor.reshape(
                        qr_scale__ssa_v0, [t_dim_inline2888__ssa_v0, 1]
                    )
                    qr_token_tiles_inline2879__ssa_v0: pl.Scalar[pl.INDEX] = (tile_rows_inline306_inline1880__ssa_v0 + 7) // 8
                    ret__tmp_v0_3: pl.Tuple[pl.Tensor[[512, 1], pl.FP32], pl.Tensor[[512, 1024], pl.INT8], pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(
                        self.qr_rms_norm_quant_spmd,
                        tile_rows_inline306_inline1880__ssa_v0,
                        tile_base_inline307_inline1878__idx_v0,
                        qr_fp32_inline304_inline1875__rv_v10,
                        gamma_cq__ssa_v0,
                        qr_scale_pad_store_inline1879__iter_v1,
                        qr_scale_view_inline2896__ssa_v0,
                        qr_i8_matmul_inline1874__iter_v1,
                        qr_view_inline2890__ssa_v0,
                        core_num=qr_token_tiles_inline2879__ssa_v0,
                        allow_early_resolve=True,
                        attrs={"arg_directions": [pl.adir.scalar, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.inout, pl.adir.inout, pl.adir.inout, pl.adir.inout]},
                    )
                    qr_scale_pad_store_inline1879__ssa_v3: pl.Tensor[[512, 1], pl.FP32, pl.MemRef("mem_ddr_74", pl.const(0, pl.INT64), 2048)] = ret__tmp_v0_3[0]
                    qr_i8_matmul_inline1874__rv_v4: pl.Tensor[[512, 1024], pl.INT8, pl.MemRef("mem_ddr_75", pl.const(0, pl.INT64), 524288)] = ret__tmp_v0_3[1]
                    tid__ssa_v3: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_3[2]
                qr_i8_matmul_inline1874__rv_v2, qr_scale_pad_store_inline1879__rv_v2 = pl.yield_(qr_i8_matmul_inline1874__rv_v4, qr_scale_pad_store_inline1879__ssa_v3)
            q_seq_dep_inline1883__ssa_v0: pl.Scalar[pl.TASK_ID] = pl.system.task_dummy(deps=[])
            t_dim_inline315_inline1885__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(x_normed_t__ssa_v0, 0)
            for tile_base_inline314_inline1882__idx_v0 in pl.range(0, t_dim_inline315_inline1885__ssa_v0, 512):
                tile_rows_inline311_inline1886__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(t_dim_inline315_inline1885__ssa_v0 - tile_base_inline314_inline1882__idx_v0, 512)
                with pl.scope():
                    qproj_t_matmul_inline313_inline1884__ssa_v0: pl.Scalar[pl.INDEX] = (tile_rows_inline311_inline1886__ssa_v0 + 15) // 16 * 16
                    q_proj_i32_inline310_inline1876__ssa_v0: pl.Tensor[[qproj_t_matmul_inline313_inline1884__ssa_v0, 32768], pl.INT32, pl.MemRef("mem_ddr_78", pl.const(0, pl.INT64), 0)] = (
                        pl.tensor.create([qproj_t_matmul_inline313_inline1884__ssa_v0, 32768], dtype=pl.INT32, layout=pl.TensorLayout.ND)
                    )
                    qproj_t_matmul_inline2932__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(q_proj_i32_inline310_inline1876__ssa_v0, 0)
                    qproj_full_rows_inline2934__ssa_v0: pl.Scalar[pl.INDEX] = tile_rows_inline311_inline1886__ssa_v0 // 64 * 64
                    ret__tmp_v0_4: pl.Tuple[pl.Tensor[[qproj_t_matmul_inline313_inline1884__ssa_v0, 32768], pl.INT32], pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(
                        self.qproj_matmul_spmd,
                        q_proj_i32_inline310_inline1876__ssa_v0,
                        qproj_full_rows_inline2934__ssa_v0,
                        qr_i8_matmul_inline1874__rv_v2,
                        wq_b__ssa_v0,
                        qproj_t_matmul_inline2932__ssa_v0,
                        tile_rows_inline311_inline1886__ssa_v0,
                        deps=[q_seq_dep_inline1883__ssa_v0],
                        core_num=24,
                        attrs={"arg_directions": [pl.adir.inout, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.scalar, pl.adir.scalar]},
                    )
                    q_proj_i32_inline310_inline1876__rv_v2: pl.Tensor[[qproj_t_matmul_inline313_inline1884__ssa_v0, 32768], pl.INT32, pl.MemRef("mem_ddr_79", pl.const(0, pl.INT64), 0)] = (
                        ret__tmp_v0_4[0]
                    )
                    qproj_tid_inline2936__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_4[1]
                    q_proj_i32_inline310_inline1876__ssa_v9: pl.Tensor[[qproj_t_matmul_inline313_inline1884__ssa_v0, 32768], pl.INT32, pl.MemRef("mem_ddr_80", pl.const(0, pl.INT64), 0)] = (
                        q_proj_i32_inline310_inline1876__rv_v2
                    )
                    t_dim_inline3002__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(q__ssa_v0, 0)
                    q_flat_inline2971__ssa_v0: pl.Tensor[[t_dim_inline3002__ssa_v0, 32768], pl.BF16, pl.MemRef("mem_ddr_62", pl.const(0, pl.INT64), 0)] = pl.tensor.reshape(
                        q__ssa_v0, [t_dim_inline3002__ssa_v0, 32768]
                    )
                    ret__tmp_v0_5: pl.Tuple[pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(
                        self.qproj_dequant_rms_nope_rope_spmd,
                        q_flat_inline2971__ssa_v0,
                        tile_rows_inline311_inline1886__ssa_v0,
                        tile_base_inline314_inline1882__idx_v0,
                        qr_scale_pad_store_inline1879__rv_v2,
                        q_rope_cos_il_inline545__ssa_v0,
                        q_rope_sin_signed_inline544__ssa_v0,
                        q_rope_swap_idx_inline543__ssa_v0,
                        q_proj_i32_inline310_inline1876__ssa_v9,
                        wq_b_scale__ssa_v0,
                        core_num=48,
                        allow_early_resolve=True,
                        attrs={"arg_directions": [pl.adir.inout, pl.adir.scalar, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input]},
                    )
                    tid__ssa_v4: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_5[0]
            t_dim_inline1931__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(x_normed_t__ssa_v0, 0)
            for tile_base_inline1950__idx_v0 in pl.range(0, t_dim_inline1931__ssa_v0, 512):
                tile_rows_inline1923__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(t_dim_inline1931__ssa_v0 - tile_base_inline1950__idx_v0, 512)
                with pl.scope():
                    x_view_inline1933__ssa_v0: pl.Tensor[[t_dim_inline1931__ssa_v0, 4096], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)] = pl.tensor.reshape(
                        x_normed_t__ssa_v0, [t_dim_inline1931__ssa_v0, 4096]
                    )
                    t_matmul_inline1930__ssa_v0: pl.Scalar[pl.INDEX] = (tile_rows_inline1923__ssa_v0 + 15) // 16 * 16
                    kv_full_rows_inline1946__ssa_v0: pl.Scalar[pl.INDEX] = tile_rows_inline1923__ssa_v0 // 64 * 64
                    kv_m_groups_inline1949__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(pl.max(tile_rows_inline1923__ssa_v0 // 128, 1), 3)
                    kv_fp32_inline1938__ssa_v0: pl.Tensor[[t_matmul_inline1930__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_81", pl.const(0, pl.INT64), 0)] = pl.tensor.create(
                        [t_matmul_inline1930__ssa_v0, 512], dtype=pl.FP32, layout=pl.TensorLayout.ND
                    )
                    kv_fp32_inline1938__rv_v2: pl.Tensor[[t_matmul_inline1930__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_82", pl.const(0, pl.INT64), 0)] = self.kv_proj_seed(
                        kv_fp32_inline1938__ssa_v0, t_matmul_inline1930__ssa_v0, attrs={"arg_directions": [pl.adir.inout, pl.adir.scalar]}
                    )
                    ret__tmp_v0_6: pl.Tuple[pl.Tensor[[t_matmul_inline1930__ssa_v0, 512], pl.FP32], pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(
                        self.kv_proj_matmul_spmd,
                        kv_m_groups_inline1949__ssa_v0,
                        kv_fp32_inline1938__rv_v2,
                        kv_full_rows_inline1946__ssa_v0,
                        tile_base_inline1950__idx_v0,
                        x_view_inline1933__ssa_v0,
                        wkv__ssa_v0,
                        t_matmul_inline1930__ssa_v0,
                        tile_rows_inline1923__ssa_v0,
                        deps=[late_dep__ssa_v0],
                        core_num=kv_m_groups_inline1949__ssa_v0 * 8,
                        attrs={"arg_directions": [pl.adir.scalar, pl.adir.inout, pl.adir.scalar, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.scalar, pl.adir.scalar]},
                    )
                    kv_fp32_inline1938__rv_v10: pl.Tensor[[t_matmul_inline1930__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_83", pl.const(0, pl.INT64), 0)] = ret__tmp_v0_6[0]
                    _kv_tid_inline1957__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_6[1]
                    kv_view_inline1962__ssa_v0: pl.Tensor[[t_dim_inline1931__ssa_v0, 512], pl.BF16, pl.MemRef("mem_ddr_63", pl.const(0, pl.INT64), 0)] = pl.tensor.reshape(
                        kv__ssa_v0, [t_dim_inline1931__ssa_v0, 512]
                    )
                    kv_token_tiles_inline1963__ssa_v0: pl.Scalar[pl.INDEX] = (tile_rows_inline1923__ssa_v0 + 31) // 32
                    self.kv_rms_norm_rope_spmd(
                        tile_rows_inline1923__ssa_v0,
                        tile_base_inline1950__idx_v0,
                        kv_fp32_inline1938__rv_v10,
                        kv_view_inline1962__ssa_v0,
                        gamma_ckv__ssa_v0,
                        q_rope_cos_il_inline545__ssa_v0,
                        q_rope_sin_signed_inline544__ssa_v0,
                        q_rope_swap_idx_inline543__ssa_v0,
                        attrs={
                            "arg_directions": [pl.adir.scalar, pl.adir.scalar, pl.adir.input, pl.adir.inout, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input],
                            "core_num": kv_token_tiles_inline1963__ssa_v0,
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
            pooled_kv_inline553__ssa_v0: pl.Tensor[[384, 512], pl.FP32, pl.MemRef("mem_ddr_85", pl.const(0, pl.INT64), 786432)] = pl.tensor.create([384, 512], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            kv_proj_pad_inline552__ssa_v0: pl.Tensor[[384, 1024], pl.FP32, pl.MemRef("mem_ddr_86", pl.const(0, pl.INT64), 1572864)] = pl.tensor.create(
                [384, 1024], dtype=pl.FP32, layout=pl.TensorLayout.ND
            )
            score_proj_pad_inline551__ssa_v0: pl.Tensor[[384, 1024], pl.FP32, pl.MemRef("mem_ddr_87", pl.const(0, pl.INT64), 1572864)] = pl.tensor.create(
                [384, 1024], dtype=pl.FP32, layout=pl.TensorLayout.ND
            )
            bs_inline719_inline2040__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(x_normed_t__ssa_v0, 0)
            t_matmul_inline720_inline2020__ssa_v0: pl.Scalar[pl.INDEX] = (bs_inline719_inline2040__ssa_v0 + 63) // 64 * 64
            x_flat_inline721_inline2001__ssa_v0: pl.Tensor[[T_DYN, 4096], pl.BF16, pl.MemRef("mem_ddr_88", pl.const(0, pl.INT64), 0)] = x_normed_t__ssa_v0
            cmp4_kv_proj_pad_inline722_inline2006__ssa_v0: pl.Tensor[[384, 1024], pl.FP32, pl.MemRef("mem_ddr_89", pl.const(0, pl.INT64), 1572864)] = kv_proj_pad_inline552__ssa_v0
            cmp4_score_proj_pad_inline729_inline2003__ssa_v0: pl.Tensor[[384, 1024], pl.FP32, pl.MemRef("mem_ddr_90", pl.const(0, pl.INT64), 1572864)] = score_proj_pad_inline551__ssa_v0
            ret__tmp_v0_7: pl.Tuple[pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(
                self.kv_score_proj_spmd,
                cmp4_kv_proj_pad_inline722_inline2006__ssa_v0,
                cmp4_score_proj_pad_inline729_inline2003__ssa_v0,
                t_matmul_inline720_inline2020__ssa_v0,
                bs_inline719_inline2040__ssa_v0,
                x_flat_inline721_inline2001__ssa_v0,
                cmp_wkv__ssa_v0,
                cmp_wgate__ssa_v0,
                deps=[late_dep__ssa_v0],
                core_num=24,
                attrs={"arg_directions": [pl.adir.output_existing, pl.adir.output_existing, pl.adir.scalar, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.input]},
            )
            _kv_score_tid_inline724_inline2008__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_7[0]
            projection_tid_inline1996__ssa_v0: pl.Scalar[pl.TASK_ID] = _kv_score_tid_inline724_inline2008__ssa_v0
            b_dim_inline760_inline1995__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(compress_state_block_table__ssa_v0, 0)
            bs_inline754_inline1997__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(position_ids__ssa_v0, 0)
            s_dim_inline769_inline2026__ssa_v0: pl.Scalar[pl.INDEX] = bs_inline754_inline1997__ssa_v0 // b_dim_inline760_inline1995__ssa_v0
            cmp4_kv_proj_pad_inline761_inline2009__ssa_v0: pl.Tensor[[384, 1024], pl.FP32, pl.MemRef("mem_ddr_91", pl.const(0, pl.INT64), 1572864)] = kv_proj_pad_inline552__ssa_v0
            cmp4_score_proj_pad_inline758_inline1992__ssa_v0: pl.Tensor[[384, 1024], pl.FP32, pl.MemRef("mem_ddr_92", pl.const(0, pl.INT64), 1572864)] = score_proj_pad_inline551__ssa_v0
            compress_state_block_num_inline746_inline2010__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(compress_state__ssa_v0, 0)
            compress_state_rows_inline745_inline2013__ssa_v0: pl.Scalar[pl.INDEX] = compress_state_block_num_inline746_inline2010__ssa_v0 * 2
            compress_state_flat_inline751_inline1991__ssa_v0: pl.Tensor[[compress_state_rows_inline745_inline2013__ssa_v0, 2048], pl.FP32, pl.MemRef("mem_ddr_17", pl.const(0, pl.INT64), 0)] = (
                pl.tensor.reshape(compress_state__ssa_v0, [compress_state_rows_inline745_inline2013__ssa_v0, 2048])
            )
            _kv_score_tid_inline744_inline2019__ssa_v0: pl.Scalar[pl.TASK_ID] = projection_tid_inline1996__ssa_v0
            pool_workers_inline756_inline1990__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(b_dim_inline760_inline1995__ssa_v0, 48)
            ret__tmp_v0_8: pl.Tuple[pl.Tensor[[384, 512], pl.FP32], pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(
                self.scatter_softmax_pool_spmd,
                pooled_kv_inline553__ssa_v0,
                b_dim_inline760_inline1995__ssa_v0,
                pool_workers_inline756_inline1990__ssa_v0,
                position_ids__ssa_v0,
                s_dim_inline769_inline2026__ssa_v0,
                cmp4_score_proj_pad_inline758_inline1992__ssa_v0,
                cmp_ape__ssa_v0,
                cmp4_kv_proj_pad_inline761_inline2009__ssa_v0,
                compress_state_block_table__ssa_v0,
                compress_state_flat_inline751_inline1991__ssa_v0,
                deps=[_kv_score_tid_inline744_inline2019__ssa_v0],
                core_num=pool_workers_inline756_inline1990__ssa_v0,
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
            pooled_kv_inline553__rv_v2: pl.Tensor[[384, 512], pl.FP32, pl.MemRef("mem_ddr_93", pl.const(0, pl.INT64), 786432)] = ret__tmp_v0_8[0]
            pool_tid_inline747_inline2022__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_8[1]
            pool_tid_inline2037__ssa_v0: pl.Scalar[pl.TASK_ID] = pool_tid_inline747_inline2022__ssa_v0
            projection_ready_tid_inline1989__ssa_v0: pl.Scalar[pl.TASK_ID] = _kv_score_tid_inline744_inline2019__ssa_v0
            pool_tid_inline549__ssa_v0: pl.Scalar[pl.TASK_ID] = pool_tid_inline2037__ssa_v0
            kv_score_tid_inline548__ssa_v0: pl.Scalar[pl.TASK_ID] = projection_ready_tid_inline1989__ssa_v0
            b_dim_inline2092__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(compress_state_block_table__ssa_v0, 0)
            bs_inline2088__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(position_ids__ssa_v0, 0)
            s_dim_inline2071__ssa_v0: pl.Scalar[pl.INDEX] = bs_inline2088__ssa_v0 // b_dim_inline2092__ssa_v0
            rms_blocks_inline2082__ssa_v0: pl.Scalar[pl.INDEX] = (bs_inline2088__ssa_v0 + 15) // 16
            cmp_block_num_inline2079__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(cmp_kv__ssa_v0, 0)
            kv_flat_inline2091__ssa_v0: pl.Tensor[[T_DYN, 512], pl.FP32, pl.MemRef("mem_ddr_94", pl.const(0, pl.INT64), 0)] = cmp_out__ssa_v0
            cmp_kv_cache_flat_inline2074__ssa_v0: pl.Tensor[[cmp_block_num_inline2079__ssa_v0 * pl.const(32, pl.INDEX), 512], pl.BF16, pl.MemRef("mem_ddr_30", pl.const(0, pl.INT64), 0)] = (
                pl.tensor.reshape(cmp_kv__ssa_v0, [cmp_block_num_inline2079__ssa_v0 * 32, 512])
            )
            compress_state_block_num_inline2075__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(compress_state__ssa_v0, 0)
            compress_state_rows_inline2065__ssa_v0: pl.Scalar[pl.INDEX] = compress_state_block_num_inline2075__ssa_v0 * 2
            compress_state_flat_inline2090__ssa_v0: pl.Tensor[[compress_state_rows_inline2065__ssa_v0, 2048], pl.FP32, pl.MemRef("mem_ddr_17", pl.const(0, pl.INT64), 0)] = pl.tensor.reshape(
                compress_state__ssa_v0, [compress_state_rows_inline2065__ssa_v0, 2048]
            )
            cmp4_kv_proj_pad_inline2070__ssa_v0: pl.Tensor[[384, 1024], pl.FP32, pl.MemRef("mem_ddr_95", pl.const(0, pl.INT64), 1572864)] = kv_proj_pad_inline552__ssa_v0
            cmp4_score_proj_pad_inline2087__ssa_v0: pl.Tensor[[384, 1024], pl.FP32, pl.MemRef("mem_ddr_96", pl.const(0, pl.INT64), 1572864)] = score_proj_pad_inline551__ssa_v0
            commit_workers_inline2086__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(b_dim_inline2092__ssa_v0, 48)
            ret__tmp_v0_9: pl.Tuple[pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(
                self.compress_state_commit_spmd,
                compress_state_flat_inline2090__ssa_v0,
                b_dim_inline2092__ssa_v0,
                commit_workers_inline2086__ssa_v0,
                s_dim_inline2071__ssa_v0,
                state_slot_mapping__ssa_v0,
                position_ids__ssa_v0,
                cmp4_kv_proj_pad_inline2070__ssa_v0,
                cmp4_score_proj_pad_inline2087__ssa_v0,
                cmp_ape__ssa_v0,
                deps=[pool_tid_inline549__ssa_v0, pool_tid_inline549__ssa_v0],
                core_num=commit_workers_inline2086__ssa_v0,
                attrs={"arg_directions": [pl.adir.output_existing, pl.adir.scalar, pl.adir.scalar, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input]},
            )
            tid__ssa_v5: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_9[0]
            normed_kv_inline2080__ssa_v0: pl.Tensor[[384, 512], pl.FP32, pl.MemRef("mem_ddr_97", pl.const(0, pl.INT64), 786432)] = pl.tensor.create(
                [384, 512], dtype=pl.FP32, layout=pl.TensorLayout.ND
            )
            norm_w_2d_inline2100__ssa_v0: pl.Tensor[[1, 512], pl.FP32, pl.MemRef("mem_ddr_16", pl.const(0, pl.INT64), 2048)] = pl.tensor.reshape(cmp_norm_w__ssa_v0, [1, 512])
            ret__tmp_v0_10: pl.Tuple[pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(
                self.rmsnorm_rope_cache_write_spmd,
                bs_inline2088__ssa_v0,
                cmp_cos_il__rv_v2,
                cmp_sin_signed__rv_v2,
                pooled_kv_inline553__rv_v2,
                normed_kv_inline2080__ssa_v0,
                norm_w_2d_inline2100__ssa_v0,
                cmp_kv_cache_flat_inline2074__ssa_v0,
                kv_flat_inline2091__ssa_v0,
                cmp_slot_mapping__ssa_v0,
                deps=[pool_tid_inline549__ssa_v0, pool_tid_inline549__ssa_v0],
                core_num=rms_blocks_inline2082__ssa_v0,
                allow_early_resolve=True,
                attrs={"arg_directions": [pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.inout, pl.adir.input, pl.adir.output_existing, pl.adir.output_existing, pl.adir.input]},
            )
            cache_write_tid_inline2072__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_10[0]
            cmp_out__ssa_v1: pl.Tensor[[T_DYN, 512], pl.FP32, pl.MemRef("mem_ddr_98", pl.const(0, pl.INT64), 0)] = cmp_out__ssa_v0
            cmp_kv_score_tid__ssa_v0: pl.Scalar[pl.TASK_ID] = kv_score_tid_inline548__ssa_v0
            idx_kv_unused__ssa_v0: pl.Tensor[[T_DYN, 128], pl.FP32, pl.MemRef("mem_ddr_99", pl.const(0, pl.INT64), 0)] = pl.tensor.create([T_DYN, 128], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            normed_kv_inline560__ssa_v0: pl.Tensor[[384, 128], pl.BF16, pl.MemRef("mem_ddr_100", pl.const(0, pl.INT64), 98304)] = pl.tensor.create([384, 128], dtype=pl.BF16, layout=pl.TensorLayout.ND)
            kv_proj_pad_inline2152__ssa_v0: pl.Tensor[[384, 256], pl.FP32, pl.MemRef("mem_ddr_101", pl.const(0, pl.INT64), 393216)] = pl.tensor.create(
                [384, 256], dtype=pl.FP32, layout=pl.TensorLayout.ND
            )
            score_proj_pad_inline2156__ssa_v0: pl.Tensor[[384, 256], pl.FP32, pl.MemRef("mem_ddr_102", pl.const(0, pl.INT64), 393216)] = pl.tensor.create(
                [384, 256], dtype=pl.FP32, layout=pl.TensorLayout.ND
            )
            bs_inline414_inline2160__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(x_normed_t__ssa_v0, 0)
            t_matmul_inline415_inline2162__ssa_v0: pl.Scalar[pl.INDEX] = (bs_inline414_inline2160__ssa_v0 + 15) // 16 * 16
            x_flat_inline416_inline2157__ssa_v0: pl.Tensor[[T_DYN, 4096], pl.BF16, pl.MemRef("mem_ddr_103", pl.const(0, pl.INT64), 0)] = x_normed_t__ssa_v0
            ret__tmp_v0_11: pl.Tuple[pl.Tensor[[384, 256], pl.FP32], pl.Tensor[[384, 256], pl.FP32], pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(
                self.kv_score_proj_spmd_0,
                kv_proj_pad_inline2152__ssa_v0,
                score_proj_pad_inline2156__ssa_v0,
                t_matmul_inline415_inline2162__ssa_v0,
                bs_inline414_inline2160__ssa_v0,
                x_flat_inline416_inline2157__ssa_v0,
                inner_wkv__ssa_v0,
                inner_wgate__ssa_v0,
                deps=[late_dep__ssa_v0, late_dep__ssa_v0],
                core_num=24,
                attrs={"arg_directions": [pl.adir.output_existing, pl.adir.output_existing, pl.adir.scalar, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.input]},
            )
            kv_proj_pad_inline2152__rv_v2: pl.Tensor[[384, 256], pl.FP32, pl.MemRef("mem_ddr_104", pl.const(0, pl.INT64), 393216)] = ret__tmp_v0_11[0]
            score_proj_pad_inline2156__rv_v2: pl.Tensor[[384, 256], pl.FP32, pl.MemRef("mem_ddr_105", pl.const(0, pl.INT64), 393216)] = ret__tmp_v0_11[1]
            _kv_score_tid_inline417_inline2177__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_11[2]
            projection_tid_inline2205__ssa_v0: pl.Scalar[pl.TASK_ID] = _kv_score_tid_inline417_inline2177__ssa_v0
            b_dim_inline459_inline2149__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(inner_compress_state_block_table__ssa_v0, 0)
            bs_inline492_inline2146__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(position_ids__ssa_v0, 0)
            s_dim_inline464_inline2145__ssa_v0: pl.Scalar[pl.INDEX] = bs_inline492_inline2146__ssa_v0 // b_dim_inline459_inline2149__ssa_v0
            rms_blocks_inline507_inline2176__ssa_v0: pl.Scalar[pl.INDEX] = (bs_inline492_inline2146__ssa_v0 + 15) // 16
            compress_state_block_num_inline468_inline2144__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(inner_compress_state__ssa_v0, 0)
            compress_state_rows_inline470_inline2190__ssa_v0: pl.Scalar[pl.INDEX] = compress_state_block_num_inline468_inline2144__ssa_v0 * 2
            compress_state_flat_inline474_inline2141__ssa_v0: pl.Tensor[[compress_state_rows_inline470_inline2190__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_27", pl.const(0, pl.INT64), 0)] = (
                pl.tensor.reshape(inner_compress_state__ssa_v0, [compress_state_rows_inline470_inline2190__ssa_v0, 512])
            )
            _kv_score_tid_inline482_inline2140__ssa_v0: pl.Scalar[pl.TASK_ID] = projection_tid_inline2205__ssa_v0
            pooled_kv_inline471_inline2185__ssa_v0: pl.Tensor[[384, 128], pl.FP32, pl.MemRef("mem_ddr_106", pl.const(0, pl.INT64), 196608)] = pl.tensor.create(
                [384, 128], dtype=pl.FP32, layout=pl.TensorLayout.ND
            )
            pool_workers_inline477_inline2147__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(b_dim_inline459_inline2149__ssa_v0, 48)
            ret__tmp_v0_12: pl.Tuple[pl.Tensor[[384, 128], pl.FP32], pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(
                self.scatter_softmax_pool_spmd_0,
                pooled_kv_inline471_inline2185__ssa_v0,
                b_dim_inline459_inline2149__ssa_v0,
                pool_workers_inline477_inline2147__ssa_v0,
                position_ids__ssa_v0,
                s_dim_inline464_inline2145__ssa_v0,
                score_proj_pad_inline2156__rv_v2,
                inner_ape__ssa_v0,
                kv_proj_pad_inline2152__rv_v2,
                inner_compress_state_block_table__ssa_v0,
                compress_state_flat_inline474_inline2141__ssa_v0,
                deps=[_kv_score_tid_inline482_inline2140__ssa_v0],
                core_num=pool_workers_inline477_inline2147__ssa_v0,
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
            pooled_kv_inline471_inline2185__rv_v2: pl.Tensor[[384, 128], pl.FP32, pl.MemRef("mem_ddr_107", pl.const(0, pl.INT64), 196608)] = ret__tmp_v0_12[0]
            pool_tid_inline480_inline2175__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_12[1]
            commit_workers_inline493_inline2210__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(b_dim_inline459_inline2149__ssa_v0, 48)
            ret__tmp_v0_13: pl.Tuple[pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(
                self.compress_state_commit_spmd_0,
                compress_state_flat_inline474_inline2141__ssa_v0,
                b_dim_inline459_inline2149__ssa_v0,
                commit_workers_inline493_inline2210__ssa_v0,
                s_dim_inline464_inline2145__ssa_v0,
                inner_state_slot_mapping__ssa_v0,
                position_ids__ssa_v0,
                kv_proj_pad_inline2152__rv_v2,
                score_proj_pad_inline2156__rv_v2,
                inner_ape__ssa_v0,
                deps=[pool_tid_inline480_inline2175__ssa_v0],
                core_num=commit_workers_inline493_inline2210__ssa_v0,
                attrs={"arg_directions": [pl.adir.output_existing, pl.adir.scalar, pl.adir.scalar, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input]},
            )
            tid__ssa_v6: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_13[0]
            norm_w_2d_inline497_inline2131__ssa_v0: pl.Tensor[[1, 128], pl.FP32, pl.MemRef("mem_ddr_26", pl.const(0, pl.INT64), 512)] = pl.tensor.reshape(inner_norm_w__ssa_v0, [1, 128])
            ret__tmp_v0_14: pl.Tuple[pl.Tensor[[384, 128], pl.BF16], pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(
                self.indexer_boundary_init_spmd,
                normed_kv_inline560__ssa_v0,
                deps=[pool_tid_inline480_inline2175__ssa_v0],
                core_num=b_dim_inline459_inline2149__ssa_v0,
                attrs={"arg_directions": [pl.adir.output_existing]},
            )
            normed_kv_inline560__ssa_v1: pl.Tensor[[384, 128], pl.BF16, pl.MemRef("mem_ddr_108", pl.const(0, pl.INT64), 98304)] = ret__tmp_v0_14[0]
            boundary_init_tid_inline510_inline2153__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_14[1]
            rms_workers_inline505_inline2128__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(rms_blocks_inline507_inline2176__ssa_v0, 2)
            ret__tmp_v0_15: pl.Tuple[pl.Tensor[[384, 128], pl.BF16], pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(
                self.rmsnorm_rope_spmd,
                normed_kv_inline560__ssa_v1,
                rms_blocks_inline507_inline2176__ssa_v0,
                rms_workers_inline505_inline2128__ssa_v0,
                bs_inline492_inline2146__ssa_v0,
                inner_cos_il__rv_v2,
                inner_sin_signed__rv_v2,
                pooled_kv_inline471_inline2185__rv_v2,
                norm_w_2d_inline497_inline2131__ssa_v0,
                position_ids__ssa_v0,
                deps=[pool_tid_inline480_inline2175__ssa_v0, boundary_init_tid_inline510_inline2153__ssa_v0],
                core_num=rms_workers_inline505_inline2128__ssa_v0,
                attrs={"arg_directions": [pl.adir.inout, pl.adir.scalar, pl.adir.scalar, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input]},
            )
            normed_kv_inline560__rv_v3: pl.Tensor[[384, 128], pl.BF16, pl.MemRef("mem_ddr_109", pl.const(0, pl.INT64), 98304)] = ret__tmp_v0_15[0]
            rms_tid_inline455_inline2127__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_15[1]
            rms_tid_inline2115__ssa_v0: pl.Scalar[pl.TASK_ID] = rms_tid_inline455_inline2127__ssa_v0
            rms_tid_inline557__ssa_v0: pl.Scalar[pl.TASK_ID] = rms_tid_inline2115__ssa_v0
            bs_inline2238__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(position_ids__ssa_v0, 0)
            compact_rows_inline2241__ssa_v0: pl.Scalar[pl.INDEX] = bs_inline2238__ssa_v0 // 6 * 2
            rms_blocks_inline2239__ssa_v0: pl.Scalar[pl.INDEX] = (compact_rows_inline2241__ssa_v0 + 15) // 16
            kv_flat_inline2250__ssa_v0: pl.Tensor[[T_DYN, 128], pl.FP32, pl.MemRef("mem_ddr_110", pl.const(0, pl.INT64), 0)] = idx_kv_unused__ssa_v0
            idx_kv_scale_values_inline2235__ssa_v0: pl.Tensor[[384, 1], pl.FP32, pl.MemRef("mem_ddr_111", pl.const(0, pl.INT64), 1536)] = pl.tensor.create(
                [384, 1], dtype=pl.FP32, layout=pl.TensorLayout.ND
            )
            kv_final_inline2230__ssa_v0: pl.Tensor[[384, 128], pl.FP32, pl.MemRef("mem_ddr_112", pl.const(0, pl.INT64), 196608)] = pl.tensor.create(
                [384, 128], dtype=pl.FP32, layout=pl.TensorLayout.ND
            )
            ret__tmp_v0_16: pl.Tuple[pl.Tensor[[384, 128], pl.FP32], pl.Scalar[pl.TASK_ID]] = pl.submit(
                self.kv_hadamard,
                kv_final_inline2230__ssa_v0,
                hadamard_idx__ssa_v0,
                rms_blocks_inline2239__ssa_v0,
                compact_rows_inline2241__ssa_v0,
                normed_kv_inline560__rv_v3,
                deps=[rms_tid_inline557__ssa_v0, cmp_kv_score_tid__ssa_v0],
                attrs={"arg_directions": [pl.adir.output_existing, pl.adir.input, pl.adir.scalar, pl.adir.scalar, pl.adir.input]},
            )
            kv_final_inline2230__rv_v2: pl.Tensor[[384, 128], pl.FP32, pl.MemRef("mem_ddr_113", pl.const(0, pl.INT64), 196608)] = ret__tmp_v0_16[0]
            hadamard_tid_inline2226__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_16[1]
            ret__tmp_v0_17: pl.Tuple[pl.Tensor[[384, 1], pl.FP32], pl.Tensor[[IDX_CACHE_BLOCK_NUM_DYN, INDEXER_PAGE_BYTES_DYN], pl.INT8], pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(
                self.kv_and_cache_write_spmd,
                compact_rows_inline2241__ssa_v0,
                kv_final_inline2230__rv_v2,
                idx_kv_scale_values_inline2235__ssa_v0,
                idx_kv_cache__ssa_v0,
                kv_flat_inline2250__ssa_v0,
                position_ids__ssa_v0,
                idx_slot_mapping__ssa_v0,
                deps=[hadamard_tid_inline2226__ssa_v0],
                core_num=rms_blocks_inline2239__ssa_v0,
                allow_early_resolve=True,
                attrs={"arg_directions": [pl.adir.scalar, pl.adir.input, pl.adir.output_existing, pl.adir.inout, pl.adir.output_existing, pl.adir.input, pl.adir.input]},
            )
            idx_kv_scale_values_inline2235__ssa_v1: pl.Tensor[[384, 1], pl.FP32, pl.MemRef("mem_ddr_114", pl.const(0, pl.INT64), 1536)] = ret__tmp_v0_17[0]
            idx_kv_cache__rv_v2: pl.Tensor[[IDX_CACHE_BLOCK_NUM_DYN, INDEXER_PAGE_BYTES_DYN], pl.INT8, pl.MemRef("mem_ddr_115", pl.const(0, pl.INT64), 0)] = ret__tmp_v0_17[1]
            _write_tid_inline2228__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_17[2]
            ret__tmp_v0_18: pl.Tuple[pl.Tensor[[IDX_CACHE_BLOCK_NUM_DYN, INDEXER_PAGE_BYTES_DYN], pl.INT8], pl.Scalar[pl.TASK_ID]] = pl.submit(
                self.idx_kv_scale_commit,
                compact_rows_inline2241__ssa_v0,
                position_ids__ssa_v0,
                idx_slot_mapping__ssa_v0,
                idx_kv_cache__rv_v2,
                idx_kv_scale_values_inline2235__ssa_v1,
                deps=[_write_tid_inline2228__ssa_v0],
                allow_early_resolve=True,
                attrs={"arg_directions": [pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.inout, pl.adir.input]},
            )
            idx_kv_cache__rv_v3: pl.Tensor[[IDX_CACHE_BLOCK_NUM_DYN, INDEXER_PAGE_BYTES_DYN], pl.INT8, pl.MemRef("mem_ddr_116", pl.const(0, pl.INT64), 0)] = ret__tmp_v0_18[0]
            scale_commit_tid_inline2214__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_18[1]
            write_tid_inline554__ssa_v0: pl.Scalar[pl.TASK_ID] = scale_commit_tid_inline2214__ssa_v0
            idx_cache_write_tid__ssa_v0: pl.Scalar[pl.TASK_ID] = write_tid_inline554__ssa_v0
            with pl.scope():
                qr_hadamard_i8_inline569__ssa_v0: pl.Tensor[[24576, 128], pl.INT8, pl.MemRef("mem_ddr_117", pl.const(0, pl.INT64), 3145728)] = pl.tensor.create(
                    [24576, 128], dtype=pl.INT8, layout=pl.TensorLayout.ND
                )
                qr_hadamard_scale_dq_inline567__ssa_v0: pl.Tensor[[24576, 1], pl.FP32, pl.MemRef("mem_ddr_118", pl.const(0, pl.INT64), 98304)] = pl.tensor.create(
                    [24576, 1], dtype=pl.FP32, layout=pl.TensorLayout.ND
                )
                qr_bf16_inline2281__ssa_v0: pl.Tensor[[24576, 128], pl.BF16, pl.MemRef("mem_ddr_119", pl.const(0, pl.INT64), 6291456)] = pl.tensor.create(
                    [24576, 128], dtype=pl.BF16, layout=pl.TensorLayout.ND
                )
                bs_inline970_inline2307__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(x_normed_t__ssa_v0, 0)
                row_blocks_inline966_inline2326__ssa_v0: pl.Scalar[pl.INDEX] = (bs_inline970_inline2307__ssa_v0 + 15) // 16
                qr_acc_pad_inline964_inline2287__ssa_v0: pl.Tensor[[384, 8192], pl.INT32, pl.MemRef("mem_ddr_120", pl.const(0, pl.INT64), 12582912)] = pl.tensor.create(
                    [384, 8192], dtype=pl.INT32, layout=pl.TensorLayout.ND
                )
                ret__tmp_v0_19: pl.Tuple[pl.Tensor[[384, 8192], pl.INT32], pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(
                    self.idx_qr_proj_matmul_spmd,
                    qr_acc_pad_inline964_inline2287__ssa_v0,
                    row_blocks_inline966_inline2326__ssa_v0,
                    bs_inline970_inline2307__ssa_v0,
                    qr__ssa_v0,
                    idx_wq_b__ssa_v0,
                    core_num=24,
                    allow_early_resolve=True,
                    attrs={"arg_directions": [pl.adir.output_existing, pl.adir.scalar, pl.adir.scalar, pl.adir.input, pl.adir.input]},
                )
                qr_acc_pad_inline964_inline2287__rv_v2: pl.Tensor[[384, 8192], pl.INT32, pl.MemRef("mem_ddr_121", pl.const(0, pl.INT64), 12582912)] = ret__tmp_v0_19[0]
                idx_qr_mm_tid_inline978_inline2293__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_19[1]
                qr_bf16_2d_inline975_inline2321__ssa_v0: pl.Tensor[[384, 8192], pl.BF16, pl.MemRef("mem_ddr_119", pl.const(0, pl.INT64), 6291456)] = pl.tensor.reshape(
                    qr_bf16_inline2281__ssa_v0, [384, 8192]
                )
                dq_rope_units_inline957_inline2290__ssa_v0: pl.Scalar[pl.INDEX] = bs_inline970_inline2307__ssa_v0 // 8 * 16
                dq_rope_workers_inline969_inline2273__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(dq_rope_units_inline957_inline2290__ssa_v0, 48)
                ret__tmp_v0_20: pl.Tuple[pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(
                    self.idx_qr_dequant_rope_spmd,
                    qr_bf16_2d_inline975_inline2321__ssa_v0,
                    dq_rope_units_inline957_inline2290__ssa_v0,
                    dq_rope_workers_inline969_inline2273__ssa_v0,
                    qr_scale__ssa_v0,
                    idx_cos_il__rv_v2,
                    idx_sin_signed__rv_v2,
                    idx_wq_b_scale__ssa_v0,
                    qr_acc_pad_inline964_inline2287__rv_v2,
                    core_num=dq_rope_workers_inline969_inline2273__ssa_v0,
                    allow_early_resolve=True,
                    attrs={"arg_directions": [pl.adir.output_existing, pl.adir.scalar, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input]},
                )
                tid__ssa_v7: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_20[0]
                idx_qr_mm_tid_inline2269__ssa_v0: pl.Scalar[pl.TASK_ID] = idx_qr_mm_tid_inline978_inline2293__ssa_v0
                bs_inline1001_inline2292__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(x_normed_t__ssa_v0, 0)
                bs_heads_inline1000_inline2314__ssa_v0: pl.Scalar[pl.INDEX] = bs_inline1001_inline2292__ssa_v0 * 64
                qh_acc_gm_inline990_inline2335__ssa_v0: pl.Tensor[[bs_heads_inline1000_inline2314__ssa_v0, 128], pl.FP32, pl.MemRef("mem_ddr_122", pl.const(0, pl.INT64), 0)] = pl.tensor.create(
                    [bs_heads_inline1000_inline2314__ssa_v0, 128], dtype=pl.FP32, layout=pl.TensorLayout.ND
                )
                ret__tmp_v0_21: pl.Tuple[pl.Tensor[[bs_heads_inline1000_inline2314__ssa_v0, 128], pl.FP32], pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(
                    self.qr_hadamard_matmul_spmd,
                    hadamard_idx__ssa_v0,
                    qh_acc_gm_inline990_inline2335__ssa_v0,
                    bs_heads_inline1000_inline2314__ssa_v0,
                    qr_bf16_inline2281__ssa_v0,
                    deps=[idx_qr_mm_tid_inline2269__ssa_v0],
                    core_num=24,
                    allow_early_resolve=True,
                    attrs={"arg_directions": [pl.adir.input, pl.adir.output_existing, pl.adir.scalar, pl.adir.input]},
                )
                qh_acc_gm_inline990_inline2335__rv_v2: pl.Tensor[[bs_heads_inline1000_inline2314__ssa_v0, 128], pl.FP32, pl.MemRef("mem_ddr_123", pl.const(0, pl.INT64), 0)] = ret__tmp_v0_21[0]
                qh_mm_tid_inline997_inline2317__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_21[1]
                ret__tmp_v0_22: pl.Tuple[pl.Tensor[[24576, 128], pl.INT8], pl.Tensor[[24576, 1], pl.FP32], pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(
                    self.qr_hadamard_quant_spmd,
                    qr_hadamard_i8_inline569__ssa_v0,
                    qr_hadamard_scale_dq_inline567__ssa_v0,
                    bs_heads_inline1000_inline2314__ssa_v0,
                    qh_acc_gm_inline990_inline2335__rv_v2,
                    core_num=48,
                    allow_early_resolve=True,
                    attrs={"arg_directions": [pl.adir.output_existing, pl.adir.output_existing, pl.adir.scalar, pl.adir.input]},
                )
                qr_hadamard_i8_inline569__rv_v2: pl.Tensor[[24576, 128], pl.INT8, pl.MemRef("mem_ddr_124", pl.const(0, pl.INT64), 3145728)] = ret__tmp_v0_22[0]
                qr_hadamard_scale_dq_inline567__rv_v2: pl.Tensor[[24576, 1], pl.FP32, pl.MemRef("mem_ddr_125", pl.const(0, pl.INT64), 98304)] = ret__tmp_v0_22[1]
                qh_quant_tid_inline989_inline2318__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_22[2]
                qh_quant_tid_inline2259__ssa_v0: pl.Scalar[pl.TASK_ID] = qh_quant_tid_inline989_inline2318__ssa_v0
                qh_quant_tid_inline564__ssa_v0: pl.Scalar[pl.TASK_ID] = qh_quant_tid_inline2259__ssa_v0
                weights_gate_dep_inline562__ssa_v0: pl.Scalar[pl.TASK_ID] = pl.system.task_dummy(deps=[])
                bs_inline2408__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(x_normed_t__ssa_v0, 0)
                row_blocks_inline2444__ssa_v0: pl.Scalar[pl.INDEX] = (bs_inline2408__ssa_v0 + 15) // 16
                x_flat_inline2399__ssa_v0: pl.Tensor[[T_DYN, 4096], pl.BF16, pl.MemRef("mem_ddr_126", pl.const(0, pl.INT64), 0)] = x_normed_t__ssa_v0
                weights_inline2424__ssa_v0: pl.Tensor[[384, 64], pl.FP32, pl.MemRef("mem_ddr_127", pl.const(0, pl.INT64), 98304)] = pl.tensor.create(
                    [384, 64], dtype=pl.FP32, layout=pl.TensorLayout.ND
                )
                weights_partial_inline2406__ssa_v0: pl.Tensor[[384, 64], pl.FP32, pl.MemRef("mem_ddr_128", pl.const(0, pl.INT64), 98304)] = pl.tensor.create(
                    [384, 64], dtype=pl.FP32, layout=pl.TensorLayout.ND
                )
                ret__tmp_v0_23: pl.Tuple[pl.Tensor[[384, 64], pl.FP32], pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(
                    self.weights_proj_spmd,
                    weights_partial_inline2406__ssa_v0,
                    row_blocks_inline2444__ssa_v0,
                    bs_inline2408__ssa_v0,
                    x_flat_inline2399__ssa_v0,
                    weights_proj__ssa_v0,
                    deps=[weights_gate_dep_inline562__ssa_v0],
                    core_num=8,
                    allow_early_resolve=True,
                    attrs={"arg_directions": [pl.adir.output_existing, pl.adir.scalar, pl.adir.scalar, pl.adir.input, pl.adir.input]},
                )
                weights_partial_inline2406__rv_v2: pl.Tensor[[384, 64], pl.FP32, pl.MemRef("mem_ddr_129", pl.const(0, pl.INT64), 98304)] = ret__tmp_v0_23[0]
                _weights_tid_inline2411__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_23[1]
                ret__tmp_v0_24: pl.Tuple[pl.Tensor[[384, 64], pl.FP32], pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(
                    self.weights_proj_reduce_spmd,
                    weights_partial_inline2406__rv_v2,
                    weights_inline2424__ssa_v0,
                    core_num=row_blocks_inline2444__ssa_v0,
                    allow_early_resolve=True,
                    attrs={"arg_directions": [pl.adir.input, pl.adir.output_existing]},
                )
                weights_inline2424__ssa_v1: pl.Tensor[[384, 64], pl.FP32, pl.MemRef("mem_ddr_130", pl.const(0, pl.INT64), 98304)] = ret__tmp_v0_24[0]
                weights_tid_inline2419__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_24[1]
                b_dim_inline1083_inline2380__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(idx_block_table__ssa_v0, 0)
                table_columns_inline1070_inline2373__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(idx_block_table__ssa_v0, 1)
                idx_table_len_inline1095_inline2409__ssa_v0: pl.Scalar[pl.INDEX] = b_dim_inline1083_inline2380__ssa_v0 * table_columns_inline1070_inline2373__ssa_v0
                idx_block_table_flat_inline1138_inline2374__ssa_v0: pl.Tensor[[idx_table_len_inline1095_inline2409__ssa_v0], pl.INT32, pl.MemRef("mem_ddr_33", pl.const(0, pl.INT64), 0)] = (
                    pl.tensor.reshape(idx_block_table__ssa_v0, [idx_table_len_inline1095_inline2409__ssa_v0])
                )
                pair_arena_inline1088_inline2392__ssa_v0: pl.Tensor[[24576, 1024], pl.FP32, pl.MemRef("mem_ddr_131", pl.const(0, pl.INT64), 100663296)] = pl.tensor.create(
                    [24576, 1024], dtype=pl.FP32, layout=pl.TensorLayout.ND
                )
                score_arena_inline1090_inline2379__ssa_v0: pl.Tensor[[384, 8192], pl.FP32, pl.MemRef("mem_ddr_132", pl.const(0, pl.INT64), 12582912)] = pl.tensor.create(
                    [384, 8192], dtype=pl.FP32, layout=pl.TensorLayout.ND
                )
                coefficients_inline1085_inline2414__ssa_v0: pl.Tensor[[6144, 64], pl.FP16, pl.MemRef("mem_ddr_133", pl.const(0, pl.INT64), 786432)] = pl.tensor.create(
                    [6144, 64], dtype=pl.FP16, layout=pl.TensorLayout.ND
                )
                ret__tmp_v0_25: pl.Tuple[pl.Tensor[[6144, 64], pl.FP16], pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(
                    self.indexer_head_coefficients_spmd,
                    position_ids__ssa_v0,
                    coefficients_inline1085_inline2414__ssa_v0,
                    qr_hadamard_scale_dq_inline567__rv_v2,
                    weights_inline2424__ssa_v1,
                    deps=[qh_quant_tid_inline564__ssa_v0, weights_tid_inline2419__ssa_v0],
                    core_num=48,
                    allow_early_resolve=True,
                    attrs={"arg_directions": [pl.adir.input, pl.adir.output_existing, pl.adir.input, pl.adir.input]},
                )
                coefficients_inline1085_inline2414__rv_v2: pl.Tensor[[6144, 64], pl.FP16, pl.MemRef("mem_ddr_134", pl.const(0, pl.INT64), 786432)] = ret__tmp_v0_25[0]
                coefficients_tid_inline1091_inline2415__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_25[1]
                gm_pipe_buffer_0: pl.Tensor[[1], pl.FP32, pl.MemRef("mem_ddr_135", pl.const(0, pl.INT64), 4)] = pl.tensor.create([1], dtype=pl.FP32, layout=pl.TensorLayout.ND, manual_dep=True)
                ret__tmp_v0_26: pl.Tuple[pl.Tensor[[384, 8192], pl.FP32], pl.Tensor[[24576, 1024], pl.FP32], pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(
                    self.indexer_score_topk_leaf_spmd,
                    position_ids__ssa_v0,
                    kv_seq_lens__ssa_v0,
                    score_arena_inline1090_inline2379__ssa_v0,
                    qr_hadamard_i8_inline569__rv_v2,
                    coefficients_inline1085_inline2414__rv_v2,
                    idx_block_table_flat_inline1138_inline2374__ssa_v0,
                    table_columns_inline1070_inline2373__ssa_v0,
                    idx_kv_cache__rv_v3,
                    pair_arena_inline1088_inline2392__ssa_v0,
                    gm_pipe_buffer_0,
                    deps=[coefficients_tid_inline1091_inline2415__ssa_v0, idx_cache_write_tid__ssa_v0],
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
                score_arena_inline1090_inline2379__rv_v2: pl.Tensor[[384, 8192], pl.FP32, pl.MemRef("mem_ddr_136", pl.const(0, pl.INT64), 12582912)] = ret__tmp_v0_26[0]
                pair_arena_inline1088_inline2392__ssa_v1: pl.Tensor[[24576, 1024], pl.FP32, pl.MemRef("mem_ddr_137", pl.const(0, pl.INT64), 100663296)] = ret__tmp_v0_26[1]
                score_tid_inline1129_inline2404__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_26[2]
                for topk_batch_inline1113_inline2339__idx_v0, (max_topk_cache_len_inline1099_inline2340__iter_v1,) in pl.range(
                    b_dim_inline1083_inline2380__ssa_v0, init_values=(0,), attrs={"iter_arg_rebind_0": True}
                ):
                    t__tmp_v253: pl.Scalar[pl.INT32] = pl.tensor.read(kv_seq_lens__ssa_v0, [topk_batch_inline1113_inline2339__idx_v0])
                    topk_cache_len_inline1058_inline2345__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(t__tmp_v253, pl.INDEX) // 4
                    max_topk_cache_len_inline1099_inline2340__ssa_v3: pl.Scalar[pl.INDEX] = pl.max(max_topk_cache_len_inline1099_inline2340__iter_v1, topk_cache_len_inline1058_inline2345__ssa_v0)
                    max_topk_cache_len_inline1099_inline2340__rv_v2: pl.Scalar[pl.INDEX] = pl.yield_(max_topk_cache_len_inline1099_inline2340__ssa_v3)
                with pl.scope():
                    if max_topk_cache_len_inline1099_inline2340__rv_v2 <= 8192:
                        ret__tmp_v0_27: pl.Tuple[pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(
                            self._decode_csa_tp1_attention_indexer_topk_single_leaf_publish,
                            position_ids__ssa_v0,
                            kv_seq_lens__ssa_v0,
                            score_arena_inline1090_inline2379__rv_v2,
                            idx_topk_scores__ssa_v0,
                            idx_topk__ssa_v0,
                            deps=[score_tid_inline1129_inline2404__ssa_v0],
                            core_num=48,
                            allow_early_resolve=True,
                            attrs={"arg_directions": [pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.output_existing, pl.adir.output_existing]},
                        )
                        tid__ssa_v8: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_27[0]
                    else:
                        ret__tmp_v0_28: pl.Tuple[pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(
                            self._decode_csa_tp1_attention_indexer_topk_query_merge,
                            position_ids__ssa_v0,
                            kv_seq_lens__ssa_v0,
                            pair_arena_inline1088_inline2392__ssa_v1,
                            idx_topk_scores__ssa_v0,
                            idx_topk__ssa_v0,
                            deps=[score_tid_inline1129_inline2404__ssa_v0],
                            core_num=48,
                            allow_early_resolve=True,
                            attrs={"arg_directions": [pl.adir.input, pl.adir.input, pl.adir.inout, pl.adir.output_existing, pl.adir.output_existing]},
                        )
                        tid__ssa_v9: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_28[0]
                idx_topk_scores__ssa_v1: pl.Tensor[[T_DYN, 512], pl.FP32, pl.MemRef("mem_ddr_138", pl.const(0, pl.INT64), 0)] = idx_topk_scores__ssa_v0
                idx_topk__ssa_v1: pl.Tensor[[T_DYN, 512], pl.INT32, pl.MemRef("mem_ddr_139", pl.const(0, pl.INT64), 0)] = idx_topk__ssa_v0
                idx_topk_scores__ssa_v2: pl.Tensor[[T_DYN, 512], pl.FP32, pl.MemRef("mem_ddr_140", pl.const(0, pl.INT64), 0)] = idx_topk_scores__ssa_v1
                idx_topk__ssa_v2: pl.Tensor[[T_DYN, 512], pl.INT32, pl.MemRef("mem_ddr_141", pl.const(0, pl.INT64), 0)] = idx_topk__ssa_v1
                idx_topk_scores__ssa_v3: pl.Tensor[[T_DYN, 512], pl.FP32, pl.MemRef("mem_ddr_142", pl.const(0, pl.INT64), 0)] = idx_topk_scores__ssa_v2
                idx_topk__ssa_v3: pl.Tensor[[T_DYN, 512], pl.INT32, pl.MemRef("mem_ddr_143", pl.const(0, pl.INT64), 0)] = idx_topk__ssa_v2
            o_packed_heads__ssa_v0: pl.Tensor[[3072, 4096], pl.BF16, pl.MemRef("mem_ddr_144", pl.const(0, pl.INT64), 25165824)] = pl.tensor.create(
                [3072, 4096], dtype=pl.BF16, layout=pl.TensorLayout.ND
            )
            ori_block_num_inline2509__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(kv_cache__ssa_v0, 0)
            t_dim_inline2515__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(q__ssa_v0, 0)
            t_heads_inline2521__ssa_v0: pl.Scalar[pl.INDEX] = t_dim_inline2515__ssa_v0 * 64
            rope_cs_blocks_inline2504__ssa_v0: pl.Scalar[pl.INDEX] = t_dim_inline2515__ssa_v0 // 8
            ori_kv_flat_inline2519__ssa_v0: pl.Tensor[[ori_block_num_inline2509__ssa_v0 * pl.const(32, pl.INDEX), 512], pl.BF16, pl.MemRef("mem_ddr_29", pl.const(0, pl.INT64), 0)] = pl.tensor.reshape(
                kv_cache__ssa_v0, [ori_block_num_inline2509__ssa_v0 * 32, 512]
            )
            ret__tmp_v0_29: pl.Tuple[pl.Tensor[[ori_block_num_inline2509__ssa_v0 * pl.const(32, pl.INDEX), 512], pl.BF16], pl.Scalar[pl.TASK_ID]] = pl.submit(
                self.kv_touch, ori_kv_flat_inline2519__ssa_v0, allow_early_resolve=True, attrs={"arg_directions": [pl.adir.inout]}
            )
            ori_kv_flat_inline2519__ssa_v1: pl.Tensor[[ori_block_num_inline2509__ssa_v0 * pl.const(32, pl.INDEX), 512], pl.BF16, pl.MemRef("mem_ddr_145", pl.const(0, pl.INT64), 0)] = ret__tmp_v0_29[0]
            tid__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_29[1]
            sparse_bias_inline2513__ssa_v0: pl.Tensor[[t_dim_inline2515__ssa_v0, 640], pl.FP32, pl.MemRef("mem_ddr_146", pl.const(0, pl.INT64), 0)] = pl.tensor.create(
                [t_dim_inline2515__ssa_v0, 640], dtype=pl.FP32, layout=pl.TensorLayout.ND
            )
            cmp_sparse_indices_inline2527__ssa_v0: pl.Tensor[[t_dim_inline2515__ssa_v0, 512], pl.INT32, pl.MemRef("mem_ddr_147", pl.const(0, pl.INT64), 0)] = pl.tensor.create(
                [t_dim_inline2515__ssa_v0, 512], dtype=pl.INT32, layout=pl.TensorLayout.ND
            )
            valid_block_mask_inline2517__ssa_v0: pl.Tensor[[t_dim_inline2515__ssa_v0, 16], pl.INT32, pl.MemRef("mem_ddr_148", pl.const(0, pl.INT64), 0)] = pl.tensor.create(
                [t_dim_inline2515__ssa_v0, 16], dtype=pl.INT32, layout=pl.TensorLayout.ND
            )
            ret__tmp_v0_30: pl.Tuple[pl.Tensor[[t_dim_inline2515__ssa_v0, 512], pl.INT32], pl.Tensor[[t_dim_inline2515__ssa_v0, 640], pl.FP32], pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(
                self.csa_slots_build_valid_qk_plan_spmd,
                cmp_sparse_indices_inline2527__ssa_v0,
                sparse_bias_inline2513__ssa_v0,
                t_dim_inline2515__ssa_v0,
                idx_topk__ssa_v3,
                position_ids_t1__ssa_v0,
                window_swa_indices__ssa_v0,
                valid_block_mask_inline2517__ssa_v0,
                core_num=16,
                allow_early_resolve=True,
                attrs={"arg_directions": [pl.adir.output_existing, pl.adir.output_existing, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.output_existing]},
            )
            cmp_sparse_indices_inline2527__rv_v2: pl.Tensor[[t_dim_inline2515__ssa_v0, 512], pl.INT32, pl.MemRef("mem_ddr_149", pl.const(0, pl.INT64), 0)] = ret__tmp_v0_30[0]
            sparse_bias_inline2513__rv_v2: pl.Tensor[[t_dim_inline2515__ssa_v0, 640], pl.FP32, pl.MemRef("mem_ddr_150", pl.const(0, pl.INT64), 0)] = ret__tmp_v0_30[1]
            qk_plan_tid_inline2533__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_30[2]
            cmp_block_num_inline2493__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(cmp_kv__ssa_v0, 0)
            cmp_kv_flat_inline2564__ssa_v0: pl.Tensor[[cmp_block_num_inline2493__ssa_v0 * pl.const(32, pl.INDEX), 512], pl.BF16, pl.MemRef("mem_ddr_30", pl.const(0, pl.INT64), 0)] = pl.tensor.reshape(
                cmp_kv__ssa_v0, [cmp_block_num_inline2493__ssa_v0 * 32, 512]
            )
            q_flat_inline2539__ssa_v0: pl.Tensor[[t_heads_inline2521__ssa_v0, 512], pl.BF16, pl.MemRef("mem_ddr_62", pl.const(0, pl.INT64), 0)] = pl.tensor.reshape(
                q__ssa_v0, [t_heads_inline2521__ssa_v0, 512]
            )
            attn_sink_col_inline2487__ssa_v0: pl.Tensor[[64, 1], pl.FP32, pl.MemRef("mem_ddr_43", pl.const(0, pl.INT64), 256)] = pl.tensor.reshape(attn_sink__ssa_v0, [64, 1])
            attn_mi_inline2536__ssa_v0: pl.Tensor[[t_heads_inline2521__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_151", pl.const(0, pl.INT64), 0)] = pl.tensor.create(
                [t_heads_inline2521__ssa_v0, 1], dtype=pl.FP32, layout=pl.TensorLayout.ND
            )
            attn_li_inline2559__ssa_v0: pl.Tensor[[t_heads_inline2521__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_152", pl.const(0, pl.INT64), 0)] = pl.tensor.create(
                [t_heads_inline2521__ssa_v0, 1], dtype=pl.FP32, layout=pl.TensorLayout.ND
            )
            attn_oi_inline2479__ssa_v0: pl.Tensor[[t_heads_inline2521__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_153", pl.const(0, pl.INT64), 0)] = pl.tensor.create(
                [t_heads_inline2521__ssa_v0, 512], dtype=pl.FP32, layout=pl.TensorLayout.ND
            )
            kv_transfer_inline2476__ssa_v0: pl.Tensor[[9216, 512], pl.BF16, pl.MemRef("mem_ddr_154", pl.const(0, pl.INT64), 9437184)] = pl.tensor.create(
                [9216, 512], dtype=pl.BF16, layout=pl.TensorLayout.ND
            )
            score_transfer_inline2473__ssa_v0: pl.Tensor[[4608, 128], pl.FP32, pl.MemRef("mem_ddr_155", pl.const(0, pl.INT64), 2359296)] = pl.tensor.create(
                [4608, 128], dtype=pl.FP32, layout=pl.TensorLayout.ND
            )
            probability_transfer_inline2461__ssa_v0: pl.Tensor[[4608, 128], pl.BF16, pl.MemRef("mem_ddr_156", pl.const(0, pl.INT64), 1179648)] = pl.tensor.create(
                [4608, 128], dtype=pl.BF16, layout=pl.TensorLayout.ND
            )
            pv_transfer_inline2471__ssa_v0: pl.Tensor[[4608, 512], pl.FP32, pl.MemRef("mem_ddr_157", pl.const(0, pl.INT64), 9437184)] = pl.tensor.create(
                [4608, 512], dtype=pl.FP32, layout=pl.TensorLayout.ND
            )
            mi_transfer_inline2470__ssa_v0: pl.Tensor[[4608, 1], pl.FP32, pl.MemRef("mem_ddr_158", pl.const(0, pl.INT64), 18432)] = pl.tensor.create(
                [4608, 1], dtype=pl.FP32, layout=pl.TensorLayout.ND
            )
            li_transfer_inline2478__ssa_v0: pl.Tensor[[4608, 1], pl.FP32, pl.MemRef("mem_ddr_159", pl.const(0, pl.INT64), 18432)] = pl.tensor.create(
                [4608, 1], dtype=pl.FP32, layout=pl.TensorLayout.ND
            )
            ffts_workspace_inline2458__ssa_v0: pl.Tensor[[256], pl.INT64, pl.MemRef("mem_ddr_160", pl.const(0, pl.INT64), 2048)] = pl.tensor.create([256], dtype=pl.INT64, layout=pl.TensorLayout.ND)
            ret__tmp_v0_31: pl.Tuple[
                pl.Tensor[[t_heads_inline2521__ssa_v0, 1], pl.FP32], pl.Tensor[[t_heads_inline2521__ssa_v0, 1], pl.FP32], pl.Tensor[[t_heads_inline2521__ssa_v0, 512], pl.FP32], pl.Scalar[pl.TASK_ID]
            ] = pl.spmd_submit(
                self.qk_pv_spmd,
                ffts_workspace_inline2458__ssa_v0,
                t_dim_inline2515__ssa_v0,
                q_flat_inline2539__ssa_v0,
                valid_block_mask_inline2517__ssa_v0,
                kv_transfer_inline2476__ssa_v0,
                score_transfer_inline2473__ssa_v0,
                probability_transfer_inline2461__ssa_v0,
                pv_transfer_inline2471__ssa_v0,
                attn_sink_col_inline2487__ssa_v0,
                position_ids_t1__ssa_v0,
                window_swa_indices__ssa_v0,
                ori_kv_flat_inline2519__ssa_v1,
                cmp_sparse_indices_inline2527__rv_v2,
                cmp_block_table__ssa_v0,
                cmp_kv_flat_inline2564__ssa_v0,
                sparse_bias_inline2513__rv_v2,
                mi_transfer_inline2470__ssa_v0,
                li_transfer_inline2478__ssa_v0,
                attn_mi_inline2536__ssa_v0,
                attn_li_inline2559__ssa_v0,
                attn_oi_inline2479__ssa_v0,
                deps=[qk_plan_tid_inline2533__ssa_v0],
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
            attn_mi_inline2536__ssa_v1: pl.Tensor[[t_heads_inline2521__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_161", pl.const(0, pl.INT64), 0)] = ret__tmp_v0_31[0]
            attn_li_inline2559__ssa_v1: pl.Tensor[[t_heads_inline2521__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_162", pl.const(0, pl.INT64), 0)] = ret__tmp_v0_31[1]
            attn_oi_inline2479__ssa_v1: pl.Tensor[[t_heads_inline2521__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_163", pl.const(0, pl.INT64), 0)] = ret__tmp_v0_31[2]
            qk_tid_inline2472__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_31[3]
            rope_cos_il_inline2484__ssa_v0: pl.Tensor[[384, 64], pl.FP32, pl.MemRef("mem_ddr_164", pl.const(0, pl.INT64), 98304)] = pl.tensor.create(
                [384, 64], dtype=pl.FP32, layout=pl.TensorLayout.ND
            )
            rope_sin_signed_inline2456__ssa_v0: pl.Tensor[[384, 64], pl.FP32, pl.MemRef("mem_ddr_165", pl.const(0, pl.INT64), 98304)] = pl.tensor.create(
                [384, 64], dtype=pl.FP32, layout=pl.TensorLayout.ND
            )
            rope_swap_idx_inline2571__ssa_v0: pl.Tensor[[16, 64], pl.INT32, pl.MemRef("mem_ddr_166", pl.const(0, pl.INT64), 4096)] = pl.tensor.create(
                [16, 64], dtype=pl.INT32, layout=pl.TensorLayout.ND
            )
            ret__tmp_v0_32: pl.Tuple[pl.Tensor[[16, 64], pl.INT32], pl.Tensor[[384, 64], pl.FP32], pl.Tensor[[384, 64], pl.FP32], pl.Scalar[pl.TASK_ID]] = pl.submit(
                self.rope_cs,
                rope_swap_idx_inline2571__ssa_v0,
                rope_cos_il_inline2484__ssa_v0,
                rope_sin_signed_inline2456__ssa_v0,
                rope_cs_blocks_inline2504__ssa_v0,
                freqs_cos__ssa_v0,
                freqs_sin__ssa_v0,
                allow_early_resolve=True,
                attrs={"arg_directions": [pl.adir.output_existing, pl.adir.output_existing, pl.adir.output_existing, pl.adir.scalar, pl.adir.input, pl.adir.input]},
            )
            rope_swap_idx_inline2571__ssa_v1: pl.Tensor[[16, 64], pl.INT32, pl.MemRef("mem_ddr_167", pl.const(0, pl.INT64), 4096)] = ret__tmp_v0_32[0]
            rope_cos_il_inline2484__rv_v2: pl.Tensor[[384, 64], pl.FP32, pl.MemRef("mem_ddr_168", pl.const(0, pl.INT64), 98304)] = ret__tmp_v0_32[1]
            rope_sin_signed_inline2456__rv_v2: pl.Tensor[[384, 64], pl.FP32, pl.MemRef("mem_ddr_169", pl.const(0, pl.INT64), 98304)] = ret__tmp_v0_32[2]
            rope_tid_inline2475__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_32[3]
            attn_mi_inline600__ssa_v0: pl.Tensor[[t_heads_inline2521__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_170", pl.const(0, pl.INT64), 0)] = attn_mi_inline2536__ssa_v1
            attn_li_inline594__ssa_v0: pl.Tensor[[t_heads_inline2521__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_171", pl.const(0, pl.INT64), 0)] = attn_li_inline2559__ssa_v1
            attn_oi_inline592__ssa_v0: pl.Tensor[[t_heads_inline2521__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_172", pl.const(0, pl.INT64), 0)] = attn_oi_inline2479__ssa_v1
            rope_cos_il_inline591__ssa_v0: pl.Tensor[[384, 64], pl.FP32, pl.MemRef("mem_ddr_173", pl.const(0, pl.INT64), 98304)] = rope_cos_il_inline2484__rv_v2
            rope_sin_signed_inline589__ssa_v0: pl.Tensor[[384, 64], pl.FP32, pl.MemRef("mem_ddr_174", pl.const(0, pl.INT64), 98304)] = rope_sin_signed_inline2456__rv_v2
            rope_swap_idx_inline597__ssa_v0: pl.Tensor[[16, 64], pl.INT32, pl.MemRef("mem_ddr_175", pl.const(0, pl.INT64), 4096)] = rope_swap_idx_inline2571__ssa_v1
            qk_tid_inline602__ssa_v0: pl.Scalar[pl.TASK_ID] = qk_tid_inline2472__ssa_v0
            rope_tid_inline601__ssa_v0: pl.Scalar[pl.TASK_ID] = rope_tid_inline2475__ssa_v0
            t_dim_inline586__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(q__ssa_v0, 0)
            merge_sink_inline599__ssa_v0: pl.Tensor[[64, 1], pl.FP32, pl.MemRef("mem_ddr_43", pl.const(0, pl.INT64), 256)] = pl.tensor.reshape(attn_sink__ssa_v0, [64, 1])
            ret__tmp_v0_33: pl.Tuple[pl.Tensor[[3072, 4096], pl.BF16], pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(
                self.merge_norm_spmd,
                rope_swap_idx_inline597__ssa_v0,
                t_dim_inline586__ssa_v0,
                attn_mi_inline600__ssa_v0,
                attn_li_inline594__ssa_v0,
                attn_oi_inline592__ssa_v0,
                merge_sink_inline599__ssa_v0,
                rope_cos_il_inline591__ssa_v0,
                rope_sin_signed_inline589__ssa_v0,
                o_packed_heads__ssa_v0,
                deps=[qk_tid_inline602__ssa_v0, rope_tid_inline601__ssa_v0],
                core_num=48,
                attrs={"arg_directions": [pl.adir.input, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.output_existing]},
            )
            o_packed_heads__ssa_v2: pl.Tensor[[3072, 4096], pl.BF16, pl.MemRef("mem_ddr_176", pl.const(0, pl.INT64), 25165824)] = ret__tmp_v0_33[0]
            merge_tid_inline615__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_33[1]
            o_packed_heads__ssa_v1: pl.Tensor[[3072, 4096], pl.BF16, pl.MemRef("mem_ddr_177", pl.const(0, pl.INT64), 25165824)] = o_packed_heads__ssa_v2
            heads_dep__ssa_v0: pl.Scalar[pl.TASK_ID] = merge_tid_inline615__ssa_v0
            t_dim_inline652__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(attn_out__ssa_v0, 0)
            act_t_blks_inline654__ssa_v0: pl.Scalar[pl.INDEX] = (t_dim_inline652__ssa_v0 + 31) // 32
            proj_a_rows_inline659__ssa_v0: pl.Scalar[pl.INDEX] = (t_dim_inline652__ssa_v0 + 127) // 128
            proj_b_t_rows_inline640__ssa_v0: pl.Scalar[pl.INDEX] = (t_dim_inline652__ssa_v0 + 127) // 128
            proj_b_padded_rows_inline669__ssa_v0: pl.Scalar[pl.INDEX] = proj_b_t_rows_inline640__ssa_v0 * 128
            o_r_pad_inline675__ssa_v0: pl.Tensor[[384, 8192], pl.FP32, pl.MemRef("mem_ddr_178", pl.const(0, pl.INT64), 12582912)] = pl.tensor.create(
                [384, 8192], dtype=pl.FP32, layout=pl.TensorLayout.ND
            )
            o_r_i8_pad_inline667__ssa_v0: pl.Tensor[[384, 8192], pl.INT8, pl.MemRef("mem_ddr_179", pl.const(0, pl.INT64), 3145728)] = pl.tensor.create(
                [384, 8192], dtype=pl.INT8, layout=pl.TensorLayout.ND
            )
            act_scale_dq_inline694__ssa_v0: pl.Tensor[[1, 384], pl.FP32, pl.MemRef("mem_ddr_180", pl.const(0, pl.INT64), 1536)] = pl.tensor.create([1, 384], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            partials_inline684__ssa_v0: pl.Tensor[[384, 32768], pl.INT32, pl.MemRef("mem_ddr_181", pl.const(0, pl.INT64), 50331648)] = pl.tensor.create(
                [384, 32768], dtype=pl.INT32, layout=pl.TensorLayout.ND
            )
            proj_a_tids_inline687__ssa_v0: pl.Array[8, pl.TASK_ID] = pl.array.create(8, dtype=pl.TASK_ID)
            proj_b_tids_inline689__ssa_v0: pl.Array[8, pl.TASK_ID] = pl.array.create(8, dtype=pl.TASK_ID)
            with pl.scope(mode=pl.ScopeMode.MANUAL):
                for g_inline695__idx_v0, (o_r_pad_inline675__iter_v1, proj_a_tids_inline687__iter_v1) in pl.parallel(
                    8, init_values=(o_r_pad_inline675__ssa_v0, proj_a_tids_inline687__ssa_v0), attrs={"iter_arg_rebind_0": False, "iter_arg_rebind_1": True}
                ):
                    row_base_o_inline663__ssa_v0: pl.Scalar[pl.INDEX] = g_inline695__idx_v0 * 384
                    out_col_g_inline650__ssa_v0: pl.Scalar[pl.INDEX] = g_inline695__idx_v0 * 1024
                    ret__tmp_v0_34: pl.Tuple[pl.Tensor[[384, 8192], pl.FP32], pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(
                        self.proj_a_mm_spmd,
                        t_dim_inline652__ssa_v0,
                        row_base_o_inline663__ssa_v0,
                        o_packed_heads__ssa_v1,
                        wo_a__ssa_v0,
                        g_inline695__idx_v0,
                        o_r_pad_inline675__iter_v1,
                        out_col_g_inline650__ssa_v0,
                        deps=[heads_dep__ssa_v0],
                        core_num=proj_a_rows_inline659__ssa_v0 * 8,
                        allow_early_resolve=True,
                        attrs={"arg_directions": [pl.adir.scalar, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.scalar, pl.adir.output_existing, pl.adir.scalar]},
                    )
                    o_r_pad_inline675__ssa_v3: pl.Tensor[[384, 8192], pl.FP32, pl.MemRef("mem_ddr_182", pl.const(0, pl.INT64), 12582912)] = ret__tmp_v0_34[0]
                    pa_tid_inline664__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_34[1]
                    proj_a_tids_inline687__ssa_v3: pl.Array[8, pl.TASK_ID] = pl.array.update_element(proj_a_tids_inline687__iter_v1, g_inline695__idx_v0, pa_tid_inline664__ssa_v0)
                    o_r_pad_inline675__rv_v2, proj_a_tids_inline687__rv_v2 = pl.yield_(o_r_pad_inline675__ssa_v3, proj_a_tids_inline687__ssa_v3)
            _submit_deps_buf_inline696__ssa_v0: pl.Array[8, pl.TASK_ID] = pl.array.create(8, dtype=pl.TASK_ID)
            t__tmp_v312: pl.Scalar[pl.TASK_ID] = pl.array.get_element(proj_a_tids_inline687__rv_v2, 0)
            _submit_deps_buf_inline676__ssa_v0: pl.Array[8, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline696__ssa_v0, 0, t__tmp_v312)
            t__tmp_v313: pl.Scalar[pl.TASK_ID] = pl.array.get_element(proj_a_tids_inline687__rv_v2, 1)
            _submit_deps_buf_inline679__ssa_v0: pl.Array[8, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline676__ssa_v0, 1, t__tmp_v313)
            t__tmp_v314: pl.Scalar[pl.TASK_ID] = pl.array.get_element(proj_a_tids_inline687__rv_v2, 2)
            _submit_deps_buf_inline698__ssa_v0: pl.Array[8, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline679__ssa_v0, 2, t__tmp_v314)
            t__tmp_v315: pl.Scalar[pl.TASK_ID] = pl.array.get_element(proj_a_tids_inline687__rv_v2, 3)
            _submit_deps_buf_inline711__ssa_v0: pl.Array[8, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline698__ssa_v0, 3, t__tmp_v315)
            t__tmp_v316: pl.Scalar[pl.TASK_ID] = pl.array.get_element(proj_a_tids_inline687__rv_v2, 4)
            _submit_deps_buf_inline682__ssa_v0: pl.Array[8, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline711__ssa_v0, 4, t__tmp_v316)
            t__tmp_v317: pl.Scalar[pl.TASK_ID] = pl.array.get_element(proj_a_tids_inline687__rv_v2, 5)
            _submit_deps_buf_inline700__ssa_v0: pl.Array[8, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline682__ssa_v0, 5, t__tmp_v317)
            t__tmp_v318: pl.Scalar[pl.TASK_ID] = pl.array.get_element(proj_a_tids_inline687__rv_v2, 6)
            _submit_deps_buf_inline702__ssa_v0: pl.Array[8, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline700__ssa_v0, 6, t__tmp_v318)
            t__tmp_v319: pl.Scalar[pl.TASK_ID] = pl.array.get_element(proj_a_tids_inline687__rv_v2, 7)
            _submit_deps_buf_inline686__ssa_v0: pl.Array[8, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline702__ssa_v0, 7, t__tmp_v319)
            ret__tmp_v0_35: pl.Tuple[pl.Tensor[[1, 384], pl.FP32], pl.Scalar[pl.TASK_ID]] = pl.submit(
                self.oproj_token_scale,
                act_scale_dq_inline694__ssa_v0,
                t_dim_inline652__ssa_v0,
                o_r_pad_inline675__rv_v2,
                deps=[_submit_deps_buf_inline686__ssa_v0],
                allow_early_resolve=True,
                attrs={"arg_directions": [pl.adir.output_existing, pl.adir.scalar, pl.adir.input]},
            )
            act_scale_dq_inline694__rv_v2: pl.Tensor[[1, 384], pl.FP32, pl.MemRef("mem_ddr_184", pl.const(0, pl.INT64), 1536)] = ret__tmp_v0_35[0]
            scale_tid_inline673__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_35[1]
            with pl.scope(mode=pl.ScopeMode.MANUAL):
                for g_inline708__idx_v0, (o_r_i8_pad_inline667__iter_v1, partials_inline684__iter_v1, proj_b_tids_inline689__iter_v1) in pl.parallel(
                    8,
                    init_values=(o_r_i8_pad_inline667__ssa_v0, partials_inline684__ssa_v0, proj_b_tids_inline689__ssa_v0),
                    attrs={"iter_arg_rebind_0": False, "iter_arg_rebind_1": False, "iter_arg_rebind_2": True},
                ):
                    col_g_inline712__ssa_v0: pl.Scalar[pl.INDEX] = g_inline708__idx_v0 * 1024
                    ret__tmp_v0_36: pl.Tuple[pl.Tensor[[384, 8192], pl.INT8], pl.Scalar[pl.TASK_ID]] = pl.submit(
                        self.quant,
                        o_r_i8_pad_inline667__iter_v1,
                        t_dim_inline652__ssa_v0,
                        act_scale_dq_inline694__rv_v2,
                        o_r_pad_inline675__rv_v2,
                        col_g_inline712__ssa_v0,
                        proj_b_padded_rows_inline669__ssa_v0,
                        deps=[scale_tid_inline673__ssa_v0],
                        allow_early_resolve=True,
                        attrs={"arg_directions": [pl.adir.output_existing, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.scalar, pl.adir.scalar]},
                    )
                    o_r_i8_pad_inline667__rv_v7: pl.Tensor[[384, 8192], pl.INT8, pl.MemRef("mem_ddr_185", pl.const(0, pl.INT64), 3145728)] = ret__tmp_v0_36[0]
                    q_tid_inline693__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_36[1]
                    ret__tmp_v0_37: pl.Tuple[pl.Tensor[[384, 32768], pl.INT32], pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(
                        self.proj_b_mm_spmd,
                        partials_inline684__iter_v1,
                        col_g_inline712__ssa_v0,
                        o_r_i8_pad_inline667__rv_v7,
                        wo_b__ssa_v0,
                        g_inline708__idx_v0,
                        deps=[q_tid_inline693__ssa_v0],
                        core_num=proj_b_t_rows_inline640__ssa_v0 * 8,
                        allow_early_resolve=True,
                        attrs={"arg_directions": [pl.adir.output_existing, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.scalar]},
                    )
                    partials_inline684__rv_v4: pl.Tensor[[384, 32768], pl.INT32, pl.MemRef("mem_ddr_186", pl.const(0, pl.INT64), 50331648)] = ret__tmp_v0_37[0]
                    pb_tid_inline639__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_37[1]
                    proj_b_tids_inline689__ssa_v3: pl.Array[8, pl.TASK_ID] = pl.array.update_element(proj_b_tids_inline689__iter_v1, g_inline708__idx_v0, pb_tid_inline639__ssa_v0)
                    o_r_i8_pad_inline667__rv_v2, partials_inline684__rv_v2, proj_b_tids_inline689__rv_v2 = pl.yield_(
                        o_r_i8_pad_inline667__rv_v7, partials_inline684__rv_v4, proj_b_tids_inline689__ssa_v3
                    )
            _submit_deps_buf_inline649__ssa_v0: pl.Array[8, pl.TASK_ID] = pl.array.create(8, dtype=pl.TASK_ID)
            t__tmp_v327: pl.Scalar[pl.TASK_ID] = pl.array.get_element(proj_b_tids_inline689__rv_v2, 0)
            _submit_deps_buf_inline631__ssa_v0: pl.Array[8, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline649__ssa_v0, 0, t__tmp_v327)
            t__tmp_v328: pl.Scalar[pl.TASK_ID] = pl.array.get_element(proj_b_tids_inline689__rv_v2, 1)
            _submit_deps_buf_inline680__ssa_v0: pl.Array[8, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline631__ssa_v0, 1, t__tmp_v328)
            t__tmp_v329: pl.Scalar[pl.TASK_ID] = pl.array.get_element(proj_b_tids_inline689__rv_v2, 2)
            _submit_deps_buf_inline643__ssa_v0: pl.Array[8, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline680__ssa_v0, 2, t__tmp_v329)
            t__tmp_v330: pl.Scalar[pl.TASK_ID] = pl.array.get_element(proj_b_tids_inline689__rv_v2, 3)
            _submit_deps_buf_inline629__ssa_v0: pl.Array[8, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline643__ssa_v0, 3, t__tmp_v330)
            t__tmp_v331: pl.Scalar[pl.TASK_ID] = pl.array.get_element(proj_b_tids_inline689__rv_v2, 4)
            _submit_deps_buf_inline672__ssa_v0: pl.Array[8, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline629__ssa_v0, 4, t__tmp_v331)
            t__tmp_v332: pl.Scalar[pl.TASK_ID] = pl.array.get_element(proj_b_tids_inline689__rv_v2, 5)
            _submit_deps_buf_inline628__ssa_v0: pl.Array[8, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline672__ssa_v0, 5, t__tmp_v332)
            t__tmp_v333: pl.Scalar[pl.TASK_ID] = pl.array.get_element(proj_b_tids_inline689__rv_v2, 6)
            _submit_deps_buf_inline627__ssa_v0: pl.Array[8, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline628__ssa_v0, 6, t__tmp_v333)
            t__tmp_v334: pl.Scalar[pl.TASK_ID] = pl.array.get_element(proj_b_tids_inline689__rv_v2, 7)
            _submit_deps_buf_inline626__ssa_v0: pl.Array[8, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline627__ssa_v0, 7, t__tmp_v334)
            ret__tmp_v0_38: pl.Tuple[pl.Tensor[[T_DYN, 4096], pl.BF16], pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(
                self.proj_b_act_spmd,
                wo_b_scale__ssa_v0,
                attn_out__ssa_v0,
                t_dim_inline652__ssa_v0,
                partials_inline684__rv_v2,
                act_scale_dq_inline694__rv_v2,
                deps=[_submit_deps_buf_inline626__ssa_v0],
                core_num=act_t_blks_inline654__ssa_v0 * 8,
                allow_early_resolve=True,
                attrs={"arg_directions": [pl.adir.input, pl.adir.output_existing, pl.adir.scalar, pl.adir.input, pl.adir.input]},
            )
            attn_out__rv_v2: pl.Tensor[[T_DYN, 4096], pl.BF16, pl.MemRef("mem_ddr_189", pl.const(0, pl.INT64), 0)] = ret__tmp_v0_38[0]
            _act_tid_inline658__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_38[1]
        return attn_out__rv_v2