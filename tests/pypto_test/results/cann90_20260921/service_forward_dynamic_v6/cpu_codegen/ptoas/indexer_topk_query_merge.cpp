#include "pto/pto-inst.hpp"
using namespace pto;

template <typename Tensor>
static AICORE inline auto PTOAS__GLOBAL_TENSOR_DATA(Tensor &tensor)
    -> decltype(tensor.data()) {
  return tensor.data();
}


enum class PTOAutoSyncTailMode : int {
  kBarrierAll = 0,
  kSetWaitMte3ToSEvent0 = 1,
};

static AICORE inline void ptoas_auto_sync_tail(
    PTOAutoSyncTailMode mode = PTOAutoSyncTailMode::kBarrierAll) {
  switch (mode) {
  case PTOAutoSyncTailMode::kSetWaitMte3ToSEvent0:
    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    break;
  case PTOAutoSyncTailMode::kBarrierAll:
  default:
    pipe_barrier(PIPE_ALL);
    break;
  }
}

template <typename Ptr>
static AICORE inline void PTOAS__DCCI_SINGLE_CACHE_LINE(Ptr ptr) {
  dcci((__gm__ void*)ptr, cache_line_t::SINGLE_CACHE_LINE);
}

AICORE void indexer_topk_query_merge(__gm__ int64_t* v1, __gm__ int32_t* v2, __gm__ float* v3, __gm__ float* v4, __gm__ int32_t* v5, int64_t v6, int64_t v7, int32_t v8, int32_t v9) {
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  pto::MrgSortExecutedNumList v10 = pto::MrgSortExecutedNumList{0, 0, 0, 0};
  // pto: %c0_i64
  const int64_t v11 = 0;
  // pto: %c4096_i64
  const int64_t v12 = 4096;
  // pto: %c8192_i64
  const int64_t v13 = 8192;
  // pto: %c16384_i64
  const int64_t v14 = 16384;
  // pto: %c1_index
  const int64_t v15 = 1;
  // pto: %c24576_index
  const int64_t v16 = 24576;
  // pto: %c1024_index
  const int64_t v17 = 1024;
  // pto: %c512_index
  const int64_t v18 = 512;
  // pto: %c48_index
  const int64_t v19 = 48;
  // pto: %c6_index
  const int64_t v20 = 6;
  // pto: %c4_index
  const int64_t v21 = 4;
  // pto: %c1_i64
  const int64_t v22 = 1;
  // pto: %c4_i64
  const int64_t v23 = 4;
  // pto: %c262144_index
  const int64_t v24 = 262144;
  // pto: %c0_index
  const int64_t v25 = 0;
  // pto: %c8191_index
  const int64_t v26 = 8191;
  // pto: %c8192_index
  const int64_t v27 = 8192;
  // pto: %c2_index
  const int64_t v28 = 2;
  // pto: %c64_index
  const int64_t v29 = 64;
  // pto: %c2048_index
  const int64_t v30 = 2048;
  // pto: %cn1_i32
  const int32_t v31 = -1;
  // pto: %cst_21
  const float v32 = -3.40282347E+38f;
  // pto: %pair_arena__ssa_v0_view
  const int64_t v33 = 1;
  // pto: %pair_arena__ssa_v0_view
  const int64_t v34 = 1;
  // pto: %pair_arena__ssa_v0_view
  const int64_t v35 = 1;
  // pto: %pair_arena__ssa_v0_view
  int64_t v36 = v16 * v17;
  // pto: %pair_arena__ssa_v0_view
  int64_t v37 = v35 * v36;
  // pto: %pair_arena__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v38 = pto::Shape<1, 1, 1, -1, -1>(v33, v34, v35, v16, v17);
  // pto: %pair_arena__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v39 = pto::Stride<-1, -1, -1, -1, -1>(v34 * v37, v37, v36, v17, v15);
  // pto: %pair_arena__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v40 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v3, v38, v39);
  // pto: %topk_scores__ssa_v0_view
  const int64_t v41 = 1;
  // pto: %topk_scores__ssa_v0_view
  const int64_t v42 = 1;
  // pto: %topk_scores__ssa_v0_view
  const int64_t v43 = 1;
  // pto: %topk_scores__ssa_v0_view
  int64_t v44 = (int64_t) v6;
  // pto: %topk_scores__ssa_v0_view
  int64_t v45 = v44 * v18;
  // pto: %topk_scores__ssa_v0_view
  int64_t v46 = v43 * v45;
  // pto: %topk_scores__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v47 = pto::Shape<1, 1, 1, -1, -1>(v41, v42, v43, v44, v18);
  // pto: %topk_scores__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v48 = pto::Stride<-1, -1, -1, -1, -1>(v42 * v46, v46, v45, v18, v15);
  // pto: %topk_scores__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v49 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v4, v47, v48);
  // pto: %topk_indices__ssa_v0_view
  const int64_t v50 = 1;
  // pto: %topk_indices__ssa_v0_view
  const int64_t v51 = 1;
  // pto: %topk_indices__ssa_v0_view
  const int64_t v52 = 1;
  // pto: %topk_indices__ssa_v0_view
  int64_t v53 = (int64_t) v6;
  // pto: %topk_indices__ssa_v0_view
  int64_t v54 = v53 * v18;
  // pto: %topk_indices__ssa_v0_view
  int64_t v55 = v52 * v54;
  // pto: %topk_indices__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v56 = pto::Shape<1, 1, 1, -1, -1>(v50, v51, v52, v53, v18);
  // pto: %topk_indices__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v57 = pto::Stride<-1, -1, -1, -1, -1>(v51 * v55, v55, v54, v18, v15);
  // pto: %topk_indices__ssa_v0_view
  GlobalTensor<int32_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v58 = GlobalTensor<int32_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v5, v56, v57);
  // pto: %worker__ssa_v0
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID3);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID5);
  for (int64_t v59 = (int64_t) v8; v59 < v6; v59 += v19) {
    // pto: %position_inline543__tile
    ;
    int64_t v60 = (v1)[v59];
    // pto: %0, %t__tile
    ;
    int32_t v61 = (v2)[v59 / v20];
    // pto: %1, %2
    ;
    int64_t v62 = (int64_t) v61 / v21;
    // pto: %3, %4
    ;
    int64_t v63 = (int64_t) ((uint64_t) v60 + (uint64_t) v22) / v23;
    // pto: %6
    ;
    int64_t v64 = v62 < v63 ? v62 : v63;
    // pto: %7
    ;
    int64_t v65 = v64 < v24 ? v64 : v24;
    // pto: %8
    ;
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
    if (v65 > v25) {
      // pto: %9, %10, %11
      ;
      // pto: %12
      ;
      int64_t v66 = (int64_t) ((uint64_t) v59 * (uint64_t) v29);
      wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
      wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID3);
      for (int64_t v67 = v15; v67 < ((int64_t) ((uint64_t) ((int64_t) ((uint64_t) v65 + (uint64_t) v26) / v27) * (uint64_t) v28)); v67 += v15) {
        // pto: %left_inline2636__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v68 = Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v15, v17);
        // pto: %left_inline2636__ssa_v0
        ;
        uint64_t v69 = (uint64_t) v11;
        TASSIGN(v68, v69);
        // pto: %13
        ;
        int64_t v70 = v66 < v25 ? v25 : v66;
        // pto: %pair_arena__ssa_v0_pview
        ;
        __gm__ float* v71 = PTOAS__GLOBAL_TENSOR_DATA(v40);
        // pto: %pair_arena__ssa_v0_pview
        ;
        const int64_t v72 = 0;
        // pto: %pair_arena__ssa_v0_pview
        ;
        const int64_t v73 = 1024;
        // pto: %pair_arena__ssa_v0_pview
        ;
        pto::Shape<1, 1, 1, 1, 1024> v74 = pto::Shape<1, 1, 1, 1, 1024>();
        // pto: %pair_arena__ssa_v0_pview
        ;
        pto::Stride<1024, 1024, 1024, 1024, 1> v75 = pto::Stride<1024, 1024, 1024, 1024, 1>();
        // pto: %pair_arena__ssa_v0_pview
        ;
        GlobalTensor<float, pto::Shape<1, 1, 1, 1, 1024>, pto::Stride<1024, 1024, 1024, 1024, 1>, pto::Layout::ND> v76 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 1024>, pto::Stride<1024, 1024, 1024, 1024, 1>, pto::Layout::ND>(v71 + (v72 + v70 * v73), v74, v75);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
        TLOAD(v68, v76);
        // pto: %right_inline2635__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v77 = Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v15, v17);
        // pto: %right_inline2635__ssa_v0
        ;
        uint64_t v78 = (uint64_t) v12;
        TASSIGN(v77, v78);
        // pto: %14
        ;
        int64_t v79 = (int64_t) ((uint64_t) v66 + (uint64_t) v67);
        // pto: %15
        ;
        int64_t v80 = v79 < v25 ? v25 : v79;
        // pto: %16
        ;
        __gm__ float* v81 = PTOAS__GLOBAL_TENSOR_DATA(v40);
        // pto: %16
        ;
        const int64_t v82 = 0;
        // pto: %16
        ;
        const int64_t v83 = 1024;
        // pto: %16
        ;
        pto::Shape<1, 1, 1, 1, 1024> v84 = pto::Shape<1, 1, 1, 1, 1024>();
        // pto: %16
        ;
        pto::Stride<1024, 1024, 1024, 1024, 1> v85 = pto::Stride<1024, 1024, 1024, 1024, 1>();
        // pto: %16
        ;
        GlobalTensor<float, pto::Shape<1, 1, 1, 1, 1024>, pto::Stride<1024, 1024, 1024, 1024, 1>, pto::Layout::ND> v86 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 1024>, pto::Stride<1024, 1024, 1024, 1024, 1>, pto::Layout::ND>(v81 + (v82 + v80 * v83), v84, v85);
        TLOAD(v77, v86);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        // pto: %merge_tmp_inline2634__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 2048, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v87 = Tile<TileType::Vec, float, 1, 2048, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v15, v30);
        // pto: %merge_tmp_inline2634__ssa_v0
        ;
        uint64_t v88 = (uint64_t) v13;
        TASSIGN(v87, v88);
        // pto: %merged_all_inline2633__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 2048, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v89 = Tile<TileType::Vec, float, 1, 2048, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v15, v30);
        // pto: %merged_all_inline2633__ssa_v0
        ;
        uint64_t v90 = (uint64_t) v14;
        TASSIGN(v89, v90);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        pipe_barrier(PIPE_V);
        TMRGSORT<Tile<TileType::Vec, float, 1, 2048, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, Tile<TileType::Vec, float, 1, 2048, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, false>(v89, v10, v87, v77, v68);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        // pto: %merged_inline2632__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v91 = Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v15, v17);
        // pto: %merged_inline2632__ssa_v0
        ;
        uint64_t v92 = (uint64_t) v14;
        TASSIGN(v91, v92);
        // pto: %slice_view
        ;
        Tile<TileType::Vec, float, 1, 2048, BLayout::RowMajor, 1, 1024, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v93;
        // pto: %slice_view
        ;
        uint64_t v94 = (uint64_t) v14;
        TASSIGN(v93, v94);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        TSTORE(v76, v93);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
      };
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
      // pto: %root_pairs_inline554__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v95 = Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v15, v17);
      // pto: %root_pairs_inline554__ssa_v0
      ;
      uint64_t v96 = (uint64_t) v13;
      TASSIGN(v95, v96);
      // pto: %19
      ;
      int64_t v97 = v66 < v25 ? v25 : v66;
      // pto: %20
      ;
      __gm__ float* v98 = PTOAS__GLOBAL_TENSOR_DATA(v40);
      // pto: %20
      ;
      const int64_t v99 = 0;
      // pto: %20
      ;
      const int64_t v100 = 1024;
      // pto: %20
      ;
      pto::Shape<1, 1, 1, 1, 1024> v101 = pto::Shape<1, 1, 1, 1, 1024>();
      // pto: %20
      ;
      pto::Stride<1024, 1024, 1024, 1024, 1> v102 = pto::Stride<1024, 1024, 1024, 1024, 1>();
      // pto: %20
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 1, 1024>, pto::Stride<1024, 1024, 1024, 1024, 1>, pto::Layout::ND> v103 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 1024>, pto::Stride<1024, 1024, 1024, 1024, 1>, pto::Layout::ND>(v98 + (v99 + v97 * v100), v101, v102);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      TLOAD(v95, v103);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
      // pto: %root_scores_inline551__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v104 = Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v15, v18);
      // pto: %root_scores_inline551__ssa_v0
      ;
      uint64_t v105 = (uint64_t) v14;
      TASSIGN(v104, v105);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
      TGATHER<Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, MaskPattern::P0101>(v104, v95);
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
      // pto: %21
      ;
      int64_t v106 = v59 < v25 ? v25 : v59;
      // pto: %topk_scores__ssa_v0_pview
      ;
      __gm__ float* v107 = PTOAS__GLOBAL_TENSOR_DATA(v49);
      // pto: %topk_scores__ssa_v0_pview
      ;
      const int64_t v108 = 0;
      // pto: %topk_scores__ssa_v0_pview
      ;
      const int64_t v109 = 512;
      // pto: %topk_scores__ssa_v0_pview
      ;
      pto::Shape<1, 1, 1, 1, 512> v110 = pto::Shape<1, 1, 1, 1, 512>();
      // pto: %topk_scores__ssa_v0_pview
      ;
      pto::Stride<512, 512, 512, 512, 1> v111 = pto::Stride<512, 512, 512, 512, 1>();
      // pto: %topk_scores__ssa_v0_pview
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<512, 512, 512, 512, 1>, pto::Layout::ND> v112 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<512, 512, 512, 512, 1>, pto::Layout::ND>(v107 + (v108 + v106 * v109), v110, v111);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
      TSTORE(v112, v104);
      set_flag(PIPE_MTE3, PIPE_V, EVENT_ID4);
      // pto: %root_indices_inline555__ssa_v0
      ;
      Tile<TileType::Vec, int32_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v113 = Tile<TileType::Vec, int32_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v15, v18);
      // pto: %root_indices_inline555__ssa_v0
      ;
      uint64_t v114 = (uint64_t) v14;
      TASSIGN(v113, v114);
      wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID4);
      TGATHER<Tile<TileType::Vec, int32_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, MaskPattern::P1010>(v113, v95);
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
      set_flag(PIPE_V, PIPE_S, EVENT_ID0);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      // pto: %22
      ;
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
      wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
      if (v65 >= v18) {
        // pto: %topk_indices__ssa_v0_pview
        ;
        __gm__ int32_t* v115 = PTOAS__GLOBAL_TENSOR_DATA(v58);
        // pto: %topk_indices__ssa_v0_pview
        ;
        const int64_t v116 = 0;
        // pto: %topk_indices__ssa_v0_pview
        ;
        const int64_t v117 = 512;
        // pto: %topk_indices__ssa_v0_pview
        ;
        pto::Shape<1, 1, 1, 1, 512> v118 = pto::Shape<1, 1, 1, 1, 512>();
        // pto: %topk_indices__ssa_v0_pview
        ;
        pto::Stride<512, 512, 512, 512, 1> v119 = pto::Stride<512, 512, 512, 512, 1>();
        // pto: %topk_indices__ssa_v0_pview
        ;
        GlobalTensor<int32_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<512, 512, 512, 512, 1>, pto::Layout::ND> v120 = GlobalTensor<int32_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<512, 512, 512, 512, 1>, pto::Layout::ND>(v115 + (v116 + v106 * v117), v118, v119);
        TSTORE(v120, v113);
      } else {
        // pto: %output_indices_inline547__ssa_v0
        ;
        Tile<TileType::Vec, int32_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v121 = Tile<TileType::Vec, int32_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v15, v18);
        // pto: %output_indices_inline547__ssa_v0
        ;
        uint64_t v122 = (uint64_t) v13;
        TASSIGN(v121, v122);
        pipe_barrier(PIPE_V);
        TEXPANDS(v121, v31);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID3);
        set_flag(PIPE_V, PIPE_S, EVENT_ID1);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID1);
        for (int64_t v123 = v25; v123 < v65; v123 += v15) {
          // pto: %t__tmp_v1
          ;
          int32_t v124 = v113.GetValue(v123);
          v121.SetValue(v123, v124);
        };
        set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
        // pto: %27
        ;
        __gm__ int32_t* v125 = PTOAS__GLOBAL_TENSOR_DATA(v58);
        // pto: %27
        ;
        const int64_t v126 = 0;
        // pto: %27
        ;
        const int64_t v127 = 512;
        // pto: %27
        ;
        pto::Shape<1, 1, 1, 1, 512> v128 = pto::Shape<1, 1, 1, 1, 512>();
        // pto: %27
        ;
        pto::Stride<512, 512, 512, 512, 1> v129 = pto::Stride<512, 512, 512, 512, 1>();
        // pto: %27
        ;
        GlobalTensor<int32_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<512, 512, 512, 512, 1>, pto::Layout::ND> v130 = GlobalTensor<int32_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<512, 512, 512, 512, 1>, pto::Layout::ND>(v125 + (v126 + v106 * v127), v128, v129);
        wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID3);
        TSTORE(v130, v121);
      };
      set_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
      set_flag(PIPE_MTE3, PIPE_V, EVENT_ID3);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
    } else {
      // pto: %t__tmp_v2
      ;
      Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v131 = Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v15, v18);
      // pto: %t__tmp_v2
      ;
      uint64_t v132 = (uint64_t) v13;
      TASSIGN(v131, v132);
      pipe_barrier(PIPE_V);
      wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID5);
      TEXPANDS(v131, v32);
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID4);
      // pto: %28
      ;
      int64_t v133 = v59 < v25 ? v25 : v59;
      // pto: %29
      ;
      __gm__ float* v134 = PTOAS__GLOBAL_TENSOR_DATA(v49);
      // pto: %29
      ;
      const int64_t v135 = 0;
      // pto: %29
      ;
      const int64_t v136 = 512;
      // pto: %29
      ;
      pto::Shape<1, 1, 1, 1, 512> v137 = pto::Shape<1, 1, 1, 1, 512>();
      // pto: %29
      ;
      pto::Stride<512, 512, 512, 512, 1> v138 = pto::Stride<512, 512, 512, 512, 1>();
      // pto: %29
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<512, 512, 512, 512, 1>, pto::Layout::ND> v139 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<512, 512, 512, 512, 1>, pto::Layout::ND>(v134 + (v135 + v133 * v136), v137, v138);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID4);
      TSTORE(v139, v131);
      set_flag(PIPE_MTE3, PIPE_V, EVENT_ID6);
      // pto: %t__tmp_v3
      ;
      Tile<TileType::Vec, int32_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v140 = Tile<TileType::Vec, int32_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v15, v18);
      // pto: %t__tmp_v3
      ;
      uint64_t v141 = (uint64_t) v13;
      TASSIGN(v140, v141);
      wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID6);
      TEXPANDS(v140, v31);
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID5);
      // pto: %31
      ;
      __gm__ int32_t* v142 = PTOAS__GLOBAL_TENSOR_DATA(v58);
      // pto: %31
      ;
      const int64_t v143 = 0;
      // pto: %31
      ;
      const int64_t v144 = 512;
      // pto: %31
      ;
      pto::Shape<1, 1, 1, 1, 512> v145 = pto::Shape<1, 1, 1, 1, 512>();
      // pto: %31
      ;
      pto::Stride<512, 512, 512, 512, 1> v146 = pto::Stride<512, 512, 512, 512, 1>();
      // pto: %31
      ;
      GlobalTensor<int32_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<512, 512, 512, 512, 1>, pto::Layout::ND> v147 = GlobalTensor<int32_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<512, 512, 512, 512, 1>, pto::Layout::ND>(v142 + (v143 + v133 * v144), v145, v146);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID5);
      TSTORE(v147, v140);
      set_flag(PIPE_MTE3, PIPE_V, EVENT_ID5);
    };
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  }
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID3);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID5);
  #endif // __DAV_VEC__

  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}