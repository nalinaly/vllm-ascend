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

AICORE void compress_state_commit(__gm__ float* v1, __gm__ int32_t* v2, __gm__ int64_t* v3, __gm__ float* v4, __gm__ float* v5, __gm__ float* v6, int64_t v7, int64_t v8, int64_t v9, int64_t v10, int64_t v11, int64_t v12, int32_t v13, int32_t v14) {
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  // pto: %c0_i64
  const int64_t v15 = 0;
  // pto: %c4096_i64
  const int64_t v16 = 4096;
  // pto: %c1_index
  const int64_t v17 = 1;
  // pto: %c2_index
  const int64_t v18 = 2;
  // pto: %c384_index
  const int64_t v19 = 384;
  // pto: %c1024_index
  const int64_t v20 = 1024;
  // pto: %c4_index
  const int64_t v21 = 4;
  // pto: %c0_index
  const int64_t v22 = 0;
  // pto: %c2048_index
  const int64_t v23 = 2048;
  // pto: %c4_i64
  const int64_t v24 = 4;
  // pto: %compress_state__ssa_v0_view
  const int64_t v25 = 1;
  // pto: %compress_state__ssa_v0_view
  const int64_t v26 = 1;
  // pto: %compress_state__ssa_v0_view
  const int64_t v27 = 1;
  // pto: %compress_state__ssa_v0_view
  int64_t v28 = (int64_t) v10;
  // pto: %compress_state__ssa_v0_view
  int64_t v29 = (int64_t) v11;
  // pto: %compress_state__ssa_v0_view
  int64_t v30 = v28 * v29;
  // pto: %compress_state__ssa_v0_view
  int64_t v31 = v27 * v30;
  // pto: %compress_state__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v32 = pto::Shape<1, 1, 1, -1, -1>(v25, v26, v27, v28, (int64_t) v11);
  // pto: %compress_state__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v33 = pto::Stride<-1, -1, -1, -1, -1>(v26 * v31, v31, v30, v29, v17);
  // pto: %compress_state__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v34 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v1, v32, v33);
  // pto: %cmp4_kv_proj_pad_inline2063__ssa_v0_view
  const int64_t v35 = 1;
  // pto: %cmp4_kv_proj_pad_inline2063__ssa_v0_view
  const int64_t v36 = 1;
  // pto: %cmp4_kv_proj_pad_inline2063__ssa_v0_view
  const int64_t v37 = 1;
  // pto: %cmp4_kv_proj_pad_inline2063__ssa_v0_view
  int64_t v38 = v19 * v20;
  // pto: %cmp4_kv_proj_pad_inline2063__ssa_v0_view
  int64_t v39 = v37 * v38;
  // pto: %cmp4_kv_proj_pad_inline2063__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v40 = pto::Shape<1, 1, 1, -1, -1>(v35, v36, v37, v19, v20);
  // pto: %cmp4_kv_proj_pad_inline2063__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v41 = pto::Stride<-1, -1, -1, -1, -1>(v36 * v39, v39, v38, v20, v17);
  // pto: %cmp4_kv_proj_pad_inline2063__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v42 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v4, v40, v41);
  // pto: %cmp4_score_proj_pad_inline2058__ssa_v0_view
  const int64_t v43 = 1;
  // pto: %cmp4_score_proj_pad_inline2058__ssa_v0_view
  const int64_t v44 = 1;
  // pto: %cmp4_score_proj_pad_inline2058__ssa_v0_view
  const int64_t v45 = 1;
  // pto: %cmp4_score_proj_pad_inline2058__ssa_v0_view
  int64_t v46 = v19 * v20;
  // pto: %cmp4_score_proj_pad_inline2058__ssa_v0_view
  int64_t v47 = v45 * v46;
  // pto: %cmp4_score_proj_pad_inline2058__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v48 = pto::Shape<1, 1, 1, -1, -1>(v43, v44, v45, v19, v20);
  // pto: %cmp4_score_proj_pad_inline2058__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v49 = pto::Stride<-1, -1, -1, -1, -1>(v44 * v47, v47, v46, v20, v17);
  // pto: %cmp4_score_proj_pad_inline2058__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v50 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v5, v48, v49);
  // pto: %cmp_ape__ssa_v0_view
  const int64_t v51 = 1;
  // pto: %cmp_ape__ssa_v0_view
  const int64_t v52 = 1;
  // pto: %cmp_ape__ssa_v0_view
  const int64_t v53 = 1;
  // pto: %cmp_ape__ssa_v0_view
  int64_t v54 = v21 * v20;
  // pto: %cmp_ape__ssa_v0_view
  int64_t v55 = v53 * v54;
  // pto: %cmp_ape__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v56 = pto::Shape<1, 1, 1, -1, -1>(v51, v52, v53, v21, v20);
  // pto: %cmp_ape__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v57 = pto::Stride<-1, -1, -1, -1, -1>(v52 * v55, v55, v54, v20, v17);
  // pto: %cmp_ape__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v58 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v6, v56, v57);
  // pto: %commit_worker_inline2053__ssa_v0
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  for (int64_t v59 = (int64_t) v13; v59 < v7; v59 += v8) {
    for (int64_t v60 = v22; v60 < v9; v60 += v17) {
      // pto: %3, %4
      ;
      int64_t v61 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) v59 * (uint64_t) v9) + (uint64_t) v60);
      // pto: %flat_offset_mul
      ;
      int64_t v62 = (int64_t) ((uint64_t) v61 * (uint64_t) v18);
      // pto: %state_page_inline2090__tile
      ;
      int32_t v63 = (v2)[v62];
      // pto: %6, %state_offset_inline2047__tile
      ;
      int32_t v64 = (v2)[(int64_t) ((uint64_t) v62 + (uint64_t) v17)];
      // pto: %7
      ;
      int64_t v65 = (int64_t) v63;
      // pto: %9
      ;
      int64_t v66 = (int64_t) v64;
      // pto: %8, %10, %11
      ;
      if (v65 >= v22 & v66 >= v22) {
        // pto: %token_pos_inline2046__tile
        ;
        int64_t v67 = (v3)[v61];
        // pto: %14
        ;
        int64_t v68 = (int64_t) ((uint64_t) v66 * (uint64_t) v23);
        // pto: %15
        ;
        int64_t v69 = v67 % v24;
        // pto: %t__tile
        ;
        Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v70 = Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v17, v20);
        // pto: %t__tile
        ;
        uint64_t v71 = (uint64_t) v15;
        TASSIGN(v70, v71);
        // pto: %17
        ;
        int64_t v72 = v61 < v22 ? v22 : v61;
        // pto: %cmp4_kv_proj_pad_inline2063__ssa_v0_pview
        ;
        __gm__ float* v73 = PTOAS__GLOBAL_TENSOR_DATA(v42);
        // pto: %cmp4_kv_proj_pad_inline2063__ssa_v0_pview
        ;
        const int64_t v74 = 0;
        // pto: %cmp4_kv_proj_pad_inline2063__ssa_v0_pview
        ;
        const int64_t v75 = 1024;
        // pto: %cmp4_kv_proj_pad_inline2063__ssa_v0_pview
        ;
        pto::Shape<1, 1, 1, 1, 1024> v76 = pto::Shape<1, 1, 1, 1, 1024>();
        // pto: %cmp4_kv_proj_pad_inline2063__ssa_v0_pview
        ;
        pto::Stride<1024, 1024, 1024, 1024, 1> v77 = pto::Stride<1024, 1024, 1024, 1024, 1>();
        // pto: %cmp4_kv_proj_pad_inline2063__ssa_v0_pview
        ;
        GlobalTensor<float, pto::Shape<1, 1, 1, 1, 1024>, pto::Stride<1024, 1024, 1024, 1024, 1>, pto::Layout::ND> v78 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 1024>, pto::Stride<1024, 1024, 1024, 1024, 1>, pto::Layout::ND>(v73 + (v74 + v72 * v75), v76, v77);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        TLOAD(v70, v78);
        set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        // pto: %18
        ;
        int64_t v79 = v65 < v22 ? v22 : v65;
        // pto: %19
        ;
        int64_t v80 = v68 < v22 ? v22 : v68;
        // pto: %compress_state__iter_v3_pview
        ;
        int64_t v81 = (int64_t) v11;
        // pto: %compress_state__iter_v3_pview
        ;
        const int64_t v82 = 0;
        // pto: %compress_state__iter_v3_pview
        ;
        __gm__ float* v83 = PTOAS__GLOBAL_TENSOR_DATA(v34);
        // pto: %compress_state__iter_v3_pview
        ;
        const int64_t v84 = 1;
        // pto: %compress_state__iter_v3_pview
        ;
        const int64_t v85 = 1;
        // pto: %compress_state__iter_v3_pview
        ;
        const int64_t v86 = 1;
        // pto: %compress_state__iter_v3_pview
        ;
        int64_t v87 = v17 * v81;
        // pto: %compress_state__iter_v3_pview
        ;
        int64_t v88 = v86 * v87;
        // pto: %compress_state__iter_v3_pview
        ;
        pto::Shape<1, 1, 1, 1, 1024> v89 = pto::Shape<1, 1, 1, 1, 1024>(v84, v85, v86, v17, v20);
        // pto: %compress_state__iter_v3_pview
        ;
        pto::Stride<-1, -1, -1, -1, -1> v90 = pto::Stride<-1, -1, -1, -1, -1>(v85 * v88, v88, v87, v81, v17);
        // pto: %compress_state__iter_v3_pview
        ;
        GlobalTensor<float, pto::Shape<1, 1, 1, 1, 1024>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v91 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 1024>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v83 + (v82 + v79 * v81 + v80 * v17), v89, v90);
        wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        TSTORE(v91, v70);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
        // pto: %0
        ;
        Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v92 = Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v17, v20);
        // pto: %0
        ;
        uint64_t v93 = (uint64_t) v15;
        TASSIGN(v92, v93);
        // pto: %cmp4_score_proj_pad_inline2058__ssa_v0_pview
        ;
        __gm__ float* v94 = PTOAS__GLOBAL_TENSOR_DATA(v50);
        // pto: %cmp4_score_proj_pad_inline2058__ssa_v0_pview
        ;
        const int64_t v95 = 0;
        // pto: %cmp4_score_proj_pad_inline2058__ssa_v0_pview
        ;
        const int64_t v96 = 1024;
        // pto: %cmp4_score_proj_pad_inline2058__ssa_v0_pview
        ;
        pto::Shape<1, 1, 1, 1, 1024> v97 = pto::Shape<1, 1, 1, 1, 1024>();
        // pto: %cmp4_score_proj_pad_inline2058__ssa_v0_pview
        ;
        pto::Stride<1024, 1024, 1024, 1024, 1> v98 = pto::Stride<1024, 1024, 1024, 1024, 1>();
        // pto: %cmp4_score_proj_pad_inline2058__ssa_v0_pview
        ;
        GlobalTensor<float, pto::Shape<1, 1, 1, 1, 1024>, pto::Stride<1024, 1024, 1024, 1024, 1>, pto::Layout::ND> v99 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 1024>, pto::Stride<1024, 1024, 1024, 1024, 1>, pto::Layout::ND>(v94 + (v95 + v72 * v96), v97, v98);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
        TLOAD(v92, v99);
        // pto: %1
        ;
        Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v100 = Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v17, v20);
        // pto: %1
        ;
        uint64_t v101 = (uint64_t) v16;
        TASSIGN(v100, v101);
        // pto: %21
        ;
        int64_t v102 = v69 < v22 ? v22 : v69;
        // pto: %cmp_ape__ssa_v0_pview
        ;
        __gm__ float* v103 = PTOAS__GLOBAL_TENSOR_DATA(v58);
        // pto: %cmp_ape__ssa_v0_pview
        ;
        const int64_t v104 = 0;
        // pto: %cmp_ape__ssa_v0_pview
        ;
        const int64_t v105 = 1024;
        // pto: %cmp_ape__ssa_v0_pview
        ;
        pto::Shape<1, 1, 1, 1, 1024> v106 = pto::Shape<1, 1, 1, 1, 1024>();
        // pto: %cmp_ape__ssa_v0_pview
        ;
        pto::Stride<1024, 1024, 1024, 1024, 1> v107 = pto::Stride<1024, 1024, 1024, 1024, 1>();
        // pto: %cmp_ape__ssa_v0_pview
        ;
        GlobalTensor<float, pto::Shape<1, 1, 1, 1, 1024>, pto::Stride<1024, 1024, 1024, 1024, 1>, pto::Layout::ND> v108 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 1024>, pto::Stride<1024, 1024, 1024, 1024, 1>, pto::Layout::ND>(v103 + (v104 + v102 * v105), v106, v107);
        TLOAD(v100, v108);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        // pto: %2
        ;
        Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v109 = Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v17, v20);
        // pto: %2
        ;
        uint64_t v110 = (uint64_t) v15;
        TASSIGN(v109, v110);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        TADD(v109, v92, v100);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        // pto: %24
        ;
        int64_t v111 = (int64_t) ((uint64_t) v68 + (uint64_t) v20);
        // pto: %25
        ;
        int64_t v112 = v111 < v22 ? v22 : v111;
        // pto: %compress_state__tile_pview
        ;
        int64_t v113 = (int64_t) v11;
        // pto: %compress_state__tile_pview
        ;
        const int64_t v114 = 0;
        // pto: %compress_state__tile_pview
        ;
        __gm__ float* v115 = PTOAS__GLOBAL_TENSOR_DATA(v34);
        // pto: %compress_state__tile_pview
        ;
        const int64_t v116 = 1;
        // pto: %compress_state__tile_pview
        ;
        const int64_t v117 = 1;
        // pto: %compress_state__tile_pview
        ;
        const int64_t v118 = 1;
        // pto: %compress_state__tile_pview
        ;
        int64_t v119 = v17 * v113;
        // pto: %compress_state__tile_pview
        ;
        int64_t v120 = v118 * v119;
        // pto: %compress_state__tile_pview
        ;
        pto::Shape<1, 1, 1, 1, 1024> v121 = pto::Shape<1, 1, 1, 1, 1024>(v116, v117, v118, v17, v20);
        // pto: %compress_state__tile_pview
        ;
        pto::Stride<-1, -1, -1, -1, -1> v122 = pto::Stride<-1, -1, -1, -1, -1>(v117 * v120, v120, v119, v113, v17);
        // pto: %compress_state__tile_pview
        ;
        GlobalTensor<float, pto::Shape<1, 1, 1, 1, 1024>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v123 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 1024>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v115 + (v114 + v79 * v113 + v112 * v17), v121, v122);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        TSTORE(v123, v109);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
      };
    };
  }
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  #endif // __DAV_VEC__

  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}