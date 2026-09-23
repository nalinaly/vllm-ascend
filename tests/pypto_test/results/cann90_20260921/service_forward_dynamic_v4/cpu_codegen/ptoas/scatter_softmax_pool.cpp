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

AICORE void scatter_softmax_pool(__gm__ float* v1, __gm__ int64_t* v2, __gm__ int32_t* v3, __gm__ float* v4, __gm__ float* v5, __gm__ float* v6, __gm__ float* v7, int64_t v8, int64_t v9, int64_t v10, int64_t v11, int64_t v12, int64_t v13, int64_t v14, int64_t v15, int32_t v16, int32_t v17) {
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  const int64_t v18 = -7;
  // pto: %c16384_i64
  const int64_t v19 = 16384;
  // pto: %c32768_i64
  const int64_t v20 = 32768;
  // pto: %c8192_i64
  const int64_t v21 = 8192;
  // pto: %c0_i64
  const int64_t v22 = 0;
  const int64_t v23 = 2048;
  // pto: %c4096_i64
  const int64_t v24 = 4096;
  // pto: %c6144_i64
  const int64_t v25 = 6144;
  const int64_t v26 = 40960;
  const int64_t v27 = 12288;
  const int64_t v28 = 10240;
  const int64_t v29 = 24576;
  const int64_t v30 = 20480;
  const int64_t v31 = 18432;
  // pto: %c384_index
  const int64_t v32 = 384;
  // pto: %c512_index
  const int64_t v33 = 512;
  // pto: %c1_index
  const int64_t v34 = 1;
  // pto: %c1024_index
  const int64_t v35 = 1024;
  // pto: %c4_index
  const int64_t v36 = 4;
  // pto: %c0_index
  const int64_t v37 = 0;
  // pto: %cst_19
  const float v38 = 0.0f;
  // pto: %c1_i64
  const int64_t v39 = 1;
  // pto: %c4_i64
  const int64_t v40 = 4;
  // pto: %c8_index
  const int64_t v41 = 8;
  // pto: %cst_24
  const float v42 = -3.40282347E+38f;
  // pto: %c2_i64
  const int64_t v43 = 2;
  // pto: %c2048_index
  const int64_t v44 = 2048;
  // pto: %c2_index
  const int64_t v45 = 2;
  // pto: %pooled_kv_inline205__ssa_v0_view
  const int64_t v46 = 1;
  // pto: %pooled_kv_inline205__ssa_v0_view
  const int64_t v47 = 1;
  // pto: %pooled_kv_inline205__ssa_v0_view
  const int64_t v48 = 1;
  // pto: %pooled_kv_inline205__ssa_v0_view
  int64_t v49 = v32 * v33;
  // pto: %pooled_kv_inline205__ssa_v0_view
  int64_t v50 = v48 * v49;
  // pto: %pooled_kv_inline205__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v51 = pto::Shape<1, 1, 1, -1, -1>(v46, v47, v48, v32, v33);
  // pto: %pooled_kv_inline205__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v52 = pto::Stride<-1, -1, -1, -1, -1>(v47 * v50, v50, v49, v33, v34);
  // pto: %pooled_kv_inline205__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v53 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v1, v51, v52);
  // pto: %compress_state__ssa_v0_view
  const int64_t v54 = 1;
  // pto: %compress_state__ssa_v0_view
  const int64_t v55 = 1;
  // pto: %compress_state__ssa_v0_view
  const int64_t v56 = 1;
  // pto: %compress_state__ssa_v0_view
  int64_t v57 = (int64_t) v14;
  // pto: %compress_state__ssa_v0_view
  int64_t v58 = (int64_t) v15;
  // pto: %compress_state__ssa_v0_view
  int64_t v59 = v57 * v58;
  // pto: %compress_state__ssa_v0_view
  int64_t v60 = v56 * v59;
  // pto: %compress_state__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v61 = pto::Shape<1, 1, 1, -1, -1>(v54, v55, v56, v57, (int64_t) v15);
  // pto: %compress_state__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v62 = pto::Stride<-1, -1, -1, -1, -1>(v55 * v60, v60, v59, v58, v34);
  // pto: %compress_state__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v63 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v4, v61, v62);
  // pto: %cmp4_kv_proj_pad_inline530_inline1991__ssa_v0_view
  const int64_t v64 = 1;
  // pto: %cmp4_kv_proj_pad_inline530_inline1991__ssa_v0_view
  const int64_t v65 = 1;
  // pto: %cmp4_kv_proj_pad_inline530_inline1991__ssa_v0_view
  const int64_t v66 = 1;
  // pto: %cmp4_kv_proj_pad_inline530_inline1991__ssa_v0_view
  int64_t v67 = v32 * v35;
  // pto: %cmp4_kv_proj_pad_inline530_inline1991__ssa_v0_view
  int64_t v68 = v66 * v67;
  // pto: %cmp4_kv_proj_pad_inline530_inline1991__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v69 = pto::Shape<1, 1, 1, -1, -1>(v64, v65, v66, v32, v35);
  // pto: %cmp4_kv_proj_pad_inline530_inline1991__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v70 = pto::Stride<-1, -1, -1, -1, -1>(v65 * v68, v68, v67, v35, v34);
  // pto: %cmp4_kv_proj_pad_inline530_inline1991__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v71 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v5, v69, v70);
  // pto: %cmp4_score_proj_pad_inline510_inline1964__ssa_v0_view
  const int64_t v72 = 1;
  // pto: %cmp4_score_proj_pad_inline510_inline1964__ssa_v0_view
  const int64_t v73 = 1;
  // pto: %cmp4_score_proj_pad_inline510_inline1964__ssa_v0_view
  const int64_t v74 = 1;
  // pto: %cmp4_score_proj_pad_inline510_inline1964__ssa_v0_view
  int64_t v75 = v32 * v35;
  // pto: %cmp4_score_proj_pad_inline510_inline1964__ssa_v0_view
  int64_t v76 = v74 * v75;
  // pto: %cmp4_score_proj_pad_inline510_inline1964__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v77 = pto::Shape<1, 1, 1, -1, -1>(v72, v73, v74, v32, v35);
  // pto: %cmp4_score_proj_pad_inline510_inline1964__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v78 = pto::Stride<-1, -1, -1, -1, -1>(v73 * v76, v76, v75, v35, v34);
  // pto: %cmp4_score_proj_pad_inline510_inline1964__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v79 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v6, v77, v78);
  // pto: %cmp_ape__ssa_v0_view
  const int64_t v80 = 1;
  // pto: %cmp_ape__ssa_v0_view
  const int64_t v81 = 1;
  // pto: %cmp_ape__ssa_v0_view
  const int64_t v82 = 1;
  // pto: %cmp_ape__ssa_v0_view
  int64_t v83 = v36 * v35;
  // pto: %cmp_ape__ssa_v0_view
  int64_t v84 = v82 * v83;
  // pto: %cmp_ape__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v85 = pto::Shape<1, 1, 1, -1, -1>(v80, v81, v82, v36, v35);
  // pto: %cmp_ape__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v86 = pto::Stride<-1, -1, -1, -1, -1>(v81 * v84, v84, v83, v35, v34);
  // pto: %cmp_ape__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v87 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v7, v85, v86);
  // pto: %score_inline521_inline2011__phi_v4
  Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v88 = Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v33);
  // pto: %score_inline521_inline2011__phi_v4
  uint64_t v89 = (uint64_t) v25;
  TASSIGN(v88, v89);
  // pto: %value_inline533_inline2013__phi_v3
  Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v90 = Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v33);
  // pto: %value_inline533_inline2013__phi_v3
  uint64_t v91 = (uint64_t) v24;
  TASSIGN(v90, v91);
  // pto: %score_inline521_inline2011__phi_v3
  Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v92 = Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v33);
  // pto: %score_inline521_inline2011__phi_v3
  uint64_t v93 = (uint64_t) v25;
  TASSIGN(v92, v93);
  // pto: %value_inline533_inline2013__phi_v2
  Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v94 = Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v33);
  // pto: %value_inline533_inline2013__phi_v2
  uint64_t v95 = (uint64_t) v24;
  TASSIGN(v94, v95);
  // pto: %score_inline521_inline2011__phi_v7
  Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v96 = Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v33);
  // pto: %score_inline521_inline2011__phi_v7
  uint64_t v97 = (uint64_t) v22;
  TASSIGN(v96, v97);
  // pto: %value_inline533_inline2013__phi_v6
  Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v98 = Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v33);
  // pto: %value_inline533_inline2013__phi_v6
  uint64_t v99 = (uint64_t) v21;
  TASSIGN(v98, v99);
  // pto: %score_inline521_inline2011__phi_v6
  Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v100 = Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v33);
  // pto: %score_inline521_inline2011__phi_v6
  uint64_t v101 = (uint64_t) v22;
  TASSIGN(v100, v101);
  // pto: %value_inline533_inline2013__phi_v5
  Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v102 = Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v33);
  // pto: %value_inline533_inline2013__phi_v5
  uint64_t v103 = (uint64_t) v21;
  TASSIGN(v102, v103);
  // pto: %pool_worker_inline520_inline2005__ssa_v0
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  for (int64_t v104 = (int64_t) v16; v104 < v8; v104 += v9) {
    // pto: %3
    ;
    int64_t v105 = (int64_t) ((uint64_t) v104 * (uint64_t) v10);
    // pto: %first_pos_b_inline529_inline2000__tile
    ;
    int64_t v106 = (v2)[v105];
    for (int64_t v107 = v37; v107 < v10; v107 += v34) {
      // pto: %5
      ;
      int64_t v108 = (int64_t) ((uint64_t) v105 + (uint64_t) v107);
      // pto: %token_pos_inline514_inline1994__tile
      ;
      int64_t v109 = (v2)[v108];
      // pto: %t__tile
      ;
      Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v110 = Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v33);
      // pto: %t__tile
      ;
      uint64_t v111 = (uint64_t) v19;
      TASSIGN(v110, v111);
      wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
      TEXPANDS(v110, v38);
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      // pto: %6
      ;
      int64_t v112 = v108 < v37 ? v37 : v108;
      // pto: %pooled_kv_inline205__iter_v3_pview
      ;
      __gm__ float* v113 = PTOAS__GLOBAL_TENSOR_DATA(v53);
      // pto: %pooled_kv_inline205__iter_v3_pview
      ;
      const int64_t v114 = 0;
      // pto: %pooled_kv_inline205__iter_v3_pview
      ;
      const int64_t v115 = 512;
      // pto: %pooled_kv_inline205__iter_v3_pview
      ;
      pto::Shape<1, 1, 1, 1, 512> v116 = pto::Shape<1, 1, 1, 1, 512>();
      // pto: %pooled_kv_inline205__iter_v3_pview
      ;
      pto::Stride<512, 512, 512, 512, 1> v117 = pto::Stride<512, 512, 512, 512, 1>();
      // pto: %pooled_kv_inline205__iter_v3_pview
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<512, 512, 512, 512, 1>, pto::Layout::ND> v118 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<512, 512, 512, 512, 1>, pto::Layout::ND>(v113 + (v114 + v112 * v115), v116, v117);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      TSTORE(v118, v110);
      set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
      // pto: %7, %8, %9
      ;
      wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
      if ((int64_t) ((uint64_t) v109 + (uint64_t) v39) % v40 == v22) {
        // pto: %window_values_inline525_inline1987__ssa_v0
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v119 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v41, v33);
        // pto: %window_values_inline525_inline1987__ssa_v0
        ;
        uint64_t v120 = (uint64_t) v19;
        TASSIGN(v119, v120);
        TEXPANDS(v119, v38);
        // pto: %window_scores_inline526_inline2004__ssa_v0
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v121 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v41, v33);
        // pto: %window_scores_inline526_inline2004__ssa_v0
        ;
        uint64_t v122 = (uint64_t) v20;
        TASSIGN(v121, v122);
        TEXPANDS(v121, v42);
        for (int64_t v123 = v37; v123 < v41; v123 += v34) {
          // pto: %13
          ;
          int64_t v124 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) v109 + (uint64_t) v18) + (uint64_t) v123);
          // pto: %value_inline533_inline2013__ssa_v0
          ;
          Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v125 = Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v33);
          // pto: %value_inline533_inline2013__ssa_v0
          ;
          uint64_t v126 = (uint64_t) v21;
          TASSIGN(v125, v126);
          pipe_barrier(PIPE_V);
          TEXPANDS(v125, v38);
          // pto: %score_inline521_inline2011__ssa_v0
          ;
          Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v127 = Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v33);
          // pto: %score_inline521_inline2011__ssa_v0
          ;
          uint64_t v128 = (uint64_t) v22;
          TASSIGN(v127, v128);
          TEXPANDS(v127, v42);
          // pto: %14, %state_half_inline511_inline2014__phi_v2
          ;
          int64_t v129 = v123 >= v36 ? v33 : v37;
          // pto: %16, %17, %18
          ;
          wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
          if (v124 >= v37 & v124 < v106) {
            // pto: %score_inline521_inline2011__ssa_v1
            ;
            Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v130 = Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v33);
            // pto: %score_inline521_inline2011__ssa_v1
            ;
            uint64_t v131 = (uint64_t) v23;
            TASSIGN(v130, v131);
            TEXPANDS(v130, v38);
            // pto: %flat_offset_mul, %flat_offset, %20, %19
            ;
            int32_t v132 = (v3)[(int64_t) ((uint64_t) ((int64_t) (uint64_t) v104 * (uint64_t) v13) + (uint64_t) (v124 / v43))];
            // pto: %22
            ;
            int64_t v133 = (int64_t) v132;
            // pto: %23
            ;
            if (v133 >= v37) {
              // pto: %24, %26, %27
              ;
              int64_t v134 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) (v124 % v43) * (uint64_t) v44) + (uint64_t) v129);
              // pto: %value_inline533_inline2013__ssa_v1
              ;
              Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v135 = Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v33);
              // pto: %value_inline533_inline2013__ssa_v1
              ;
              uint64_t v136 = (uint64_t) v24;
              TASSIGN(v135, v136);
              // pto: %28
              ;
              int64_t v137 = v133 < v37 ? v37 : v133;
              // pto: %29
              ;
              int64_t v138 = v134 < v37 ? v37 : v134;
              // pto: %compress_state__ssa_v0_pview
              ;
              int64_t v139 = (int64_t) v15;
              // pto: %compress_state__ssa_v0_pview
              ;
              const int64_t v140 = 0;
              // pto: %compress_state__ssa_v0_pview
              ;
              __gm__ float* v141 = PTOAS__GLOBAL_TENSOR_DATA(v63);
              // pto: %compress_state__ssa_v0_pview
              ;
              const int64_t v142 = 1;
              // pto: %compress_state__ssa_v0_pview
              ;
              const int64_t v143 = 1;
              // pto: %compress_state__ssa_v0_pview
              ;
              const int64_t v144 = 1;
              // pto: %compress_state__ssa_v0_pview
              ;
              int64_t v145 = v34 * v139;
              // pto: %compress_state__ssa_v0_pview
              ;
              int64_t v146 = v144 * v145;
              // pto: %compress_state__ssa_v0_pview
              ;
              pto::Shape<1, 1, 1, 1, 512> v147 = pto::Shape<1, 1, 1, 1, 512>(v142, v143, v144, v34, v33);
              // pto: %compress_state__ssa_v0_pview
              ;
              pto::Stride<-1, -1, -1, -1, -1> v148 = pto::Stride<-1, -1, -1, -1, -1>(v143 * v146, v146, v145, v139, v34);
              // pto: %compress_state__ssa_v0_pview
              ;
              GlobalTensor<float, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v149 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v141 + (v140 + v137 * v139 + v138 * v34), v147, v148);
              TLOAD(v135, v149);
              // pto: %score_inline521_inline2011__ssa_v2
              ;
              Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v150 = Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v33);
              // pto: %score_inline521_inline2011__ssa_v2
              ;
              uint64_t v151 = (uint64_t) v25;
              TASSIGN(v150, v151);
              // pto: %31
              ;
              int64_t v152 = (int64_t) ((uint64_t) v134 + (uint64_t) v35);
              // pto: %32
              ;
              int64_t v153 = v152 < v37 ? v37 : v152;
              // pto: %33
              ;
              int64_t v154 = (int64_t) v15;
              // pto: %33
              ;
              const int64_t v155 = 0;
              // pto: %33
              ;
              __gm__ float* v156 = PTOAS__GLOBAL_TENSOR_DATA(v63);
              // pto: %33
              ;
              const int64_t v157 = 1;
              // pto: %33
              ;
              const int64_t v158 = 1;
              // pto: %33
              ;
              const int64_t v159 = 1;
              // pto: %33
              ;
              int64_t v160 = v34 * v154;
              // pto: %33
              ;
              int64_t v161 = v159 * v160;
              // pto: %33
              ;
              pto::Shape<1, 1, 1, 1, 512> v162 = pto::Shape<1, 1, 1, 1, 512>(v157, v158, v159, v34, v33);
              // pto: %33
              ;
              pto::Stride<-1, -1, -1, -1, -1> v163 = pto::Stride<-1, -1, -1, -1, -1>(v158 * v161, v161, v160, v154, v34);
              // pto: %33
              ;
              GlobalTensor<float, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v164 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v156 + (v155 + v137 * v154 + v153 * v34), v162, v163);
              TLOAD(v150, v164);
            } else {
              // pto: %score_inline521_inline2011__ssa_v1_mv
              ;
              Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v165 = Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v33);
              // pto: %score_inline521_inline2011__ssa_v1_mv
              ;
              uint64_t v166 = (uint64_t) v25;
              TASSIGN(v165, v166);
              pipe_barrier(PIPE_V);
              TMOV(v165, v130);
              // pto: %value_inline533_inline2013__ssa_v0_mv
              ;
              Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v167 = Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v33);
              // pto: %value_inline533_inline2013__ssa_v0_mv
              ;
              uint64_t v168 = (uint64_t) v24;
              TASSIGN(v167, v168);
              TMOV(v167, v125);
            };
          } else {
            // pto: %score_inline521_inline2011__ssa_v0_mv
            ;
            Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v169 = Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v33);
            // pto: %score_inline521_inline2011__ssa_v0_mv
            ;
            uint64_t v170 = (uint64_t) v25;
            TASSIGN(v169, v170);
            pipe_barrier(PIPE_V);
            TMOV(v169, v127);
            // pto: %0
            ;
            Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v171 = Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v33);
            // pto: %0
            ;
            uint64_t v172 = (uint64_t) v24;
            TASSIGN(v171, v172);
            TMOV(v171, v125);
          };
          set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
          set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
          // pto: %34
          ;
          wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
          wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
          if (v106 <= v124) {
            // pto: %35
            ;
            if (v124 <= v109) {
              // pto: %38, %40
              ;
              int64_t v173 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) v105 + (uint64_t) v124) - (uint64_t) v106);
              // pto: %41
              ;
              int64_t v174 = v124 % v40;
              // pto: %value_inline533_inline2013__ssa_v4
              ;
              Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v175 = Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v33);
              // pto: %value_inline533_inline2013__ssa_v4
              ;
              uint64_t v176 = (uint64_t) v21;
              TASSIGN(v175, v176);
              // pto: %43
              ;
              int64_t v177 = v173 < v37 ? v37 : v173;
              // pto: %44
              ;
              int64_t v178 = v129 < v37 ? v37 : v129;
              // pto: %cmp4_kv_proj_pad_inline530_inline1991__ssa_v0_pview
              ;
              __gm__ float* v179 = PTOAS__GLOBAL_TENSOR_DATA(v71);
              // pto: %cmp4_kv_proj_pad_inline530_inline1991__ssa_v0_pview
              ;
              const int64_t v180 = 0;
              // pto: %cmp4_kv_proj_pad_inline530_inline1991__ssa_v0_pview
              ;
              const int64_t v181 = 1024;
              // pto: %cmp4_kv_proj_pad_inline530_inline1991__ssa_v0_pview
              ;
              pto::Shape<1, 1, 1, 1, 512> v182 = pto::Shape<1, 1, 1, 1, 512>();
              // pto: %cmp4_kv_proj_pad_inline530_inline1991__ssa_v0_pview
              ;
              pto::Stride<1024, 1024, 1024, 1024, 1> v183 = pto::Stride<1024, 1024, 1024, 1024, 1>();
              // pto: %cmp4_kv_proj_pad_inline530_inline1991__ssa_v0_pview
              ;
              GlobalTensor<float, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<1024, 1024, 1024, 1024, 1>, pto::Layout::ND> v184 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<1024, 1024, 1024, 1024, 1>, pto::Layout::ND>(v179 + (v180 + v177 * v181 + v178), v182, v183);
              TLOAD(v175, v184);
              // pto: %t__tmp_v115
              ;
              Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v185 = Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v33);
              // pto: %t__tmp_v115
              ;
              uint64_t v186 = (uint64_t) v22;
              TASSIGN(v185, v186);
              // pto: %cmp4_score_proj_pad_inline510_inline1964__ssa_v0_pview
              ;
              __gm__ float* v187 = PTOAS__GLOBAL_TENSOR_DATA(v79);
              // pto: %cmp4_score_proj_pad_inline510_inline1964__ssa_v0_pview
              ;
              const int64_t v188 = 0;
              // pto: %cmp4_score_proj_pad_inline510_inline1964__ssa_v0_pview
              ;
              const int64_t v189 = 1024;
              // pto: %cmp4_score_proj_pad_inline510_inline1964__ssa_v0_pview
              ;
              pto::Shape<1, 1, 1, 1, 512> v190 = pto::Shape<1, 1, 1, 1, 512>();
              // pto: %cmp4_score_proj_pad_inline510_inline1964__ssa_v0_pview
              ;
              pto::Stride<1024, 1024, 1024, 1024, 1> v191 = pto::Stride<1024, 1024, 1024, 1024, 1>();
              // pto: %cmp4_score_proj_pad_inline510_inline1964__ssa_v0_pview
              ;
              GlobalTensor<float, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<1024, 1024, 1024, 1024, 1>, pto::Layout::ND> v192 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<1024, 1024, 1024, 1024, 1>, pto::Layout::ND>(v187 + (v188 + v177 * v189 + v178), v190, v191);
              TLOAD(v185, v192);
              // pto: %t__tmp_v116
              ;
              Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v193 = Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v33);
              // pto: %t__tmp_v116
              ;
              uint64_t v194 = (uint64_t) v23;
              TASSIGN(v193, v194);
              // pto: %47
              ;
              int64_t v195 = v174 < v37 ? v37 : v174;
              // pto: %cmp_ape__ssa_v0_pview
              ;
              __gm__ float* v196 = PTOAS__GLOBAL_TENSOR_DATA(v87);
              // pto: %cmp_ape__ssa_v0_pview
              ;
              const int64_t v197 = 0;
              // pto: %cmp_ape__ssa_v0_pview
              ;
              const int64_t v198 = 1024;
              // pto: %cmp_ape__ssa_v0_pview
              ;
              pto::Shape<1, 1, 1, 1, 512> v199 = pto::Shape<1, 1, 1, 1, 512>();
              // pto: %cmp_ape__ssa_v0_pview
              ;
              pto::Stride<1024, 1024, 1024, 1024, 1> v200 = pto::Stride<1024, 1024, 1024, 1024, 1>();
              // pto: %cmp_ape__ssa_v0_pview
              ;
              GlobalTensor<float, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<1024, 1024, 1024, 1024, 1>, pto::Layout::ND> v201 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<1024, 1024, 1024, 1024, 1>, pto::Layout::ND>(v196 + (v197 + v195 * v198 + v178), v199, v200);
              TLOAD(v193, v201);
              set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
              // pto: %score_inline521_inline2011__ssa_v5
              ;
              Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v202 = Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v33);
              // pto: %score_inline521_inline2011__ssa_v5
              ;
              uint64_t v203 = (uint64_t) v22;
              TASSIGN(v202, v203);
              wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
              TADD(v202, v185, v193);
            } else {
              // pto: %score_inline521_inline2011__phi_v4_mv
              ;
              Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v204 = Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v33);
              // pto: %score_inline521_inline2011__phi_v4_mv
              ;
              uint64_t v205 = (uint64_t) v22;
              TASSIGN(v204, v205);
              pipe_barrier(PIPE_V);
              TMOV(v204, v88);
              // pto: %value_inline533_inline2013__phi_v3_mv
              ;
              Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v206 = Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v33);
              // pto: %value_inline533_inline2013__phi_v3_mv
              ;
              uint64_t v207 = (uint64_t) v21;
              TASSIGN(v206, v207);
              TMOV(v206, v90);
            };
          } else {
            // pto: %1
            ;
            Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v208 = Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v33);
            // pto: %1
            ;
            uint64_t v209 = (uint64_t) v22;
            TASSIGN(v208, v209);
            pipe_barrier(PIPE_V);
            TMOV(v208, v88);
            // pto: %2
            ;
            Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v210 = Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v33);
            // pto: %2
            ;
            uint64_t v211 = (uint64_t) v21;
            TASSIGN(v210, v211);
            TMOV(v210, v90);
          };
          set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
          // pto: %49, %50, %52, %51, %assemble_view
          ;
          int64_t v212 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) ((int64_t) (uint64_t) (v123 % v36) * (uint64_t) v45) + (uint64_t) (v123 / v36)) * (uint64_t) v23);
          // pto: %assemble_view
          ;
          Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, 1, 512, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v213;
          // pto: %assemble_view
          ;
          uint64_t v214 = (uint64_t) ((int64_t) (uint64_t) v212 + (uint64_t) v19);
          TASSIGN(v213, v214);
          pipe_barrier(PIPE_V);
          TMOV(v213, v98);
          // pto: %53
          ;
          Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, 1, 512, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v215;
          // pto: %53
          ;
          uint64_t v216 = (uint64_t) ((int64_t) (uint64_t) v212 + (uint64_t) v20);
          TASSIGN(v215, v216);
          TMOV(v215, v96);
        };
        // pto: %t__tmp_v117
        ;
        Tile<TileType::Vec, float, 4, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v217 = Tile<TileType::Vec, float, 4, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v36, v33);
        // pto: %t__tmp_v117
        ;
        uint64_t v218 = (uint64_t) v20;
        TASSIGN(v217, v218);
        // pto: %slice_view
        ;
        Tile<TileType::Vec, float, 4, 512, BLayout::RowMajor, 4, 512, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v219;
        // pto: %slice_view
        ;
        uint64_t v220 = (uint64_t) v20;
        TASSIGN(v219, v220);
        // pto: %t__tmp_v118
        ;
        Tile<TileType::Vec, float, 4, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v221 = Tile<TileType::Vec, float, 4, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v36, v33);
        // pto: %t__tmp_v118
        ;
        uint64_t v222 = (uint64_t) v26;
        TASSIGN(v221, v222);
        // pto: %54
        ;
        Tile<TileType::Vec, float, 4, 512, BLayout::RowMajor, 4, 512, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v223;
        // pto: %54
        ;
        uint64_t v224 = (uint64_t) v26;
        TASSIGN(v223, v224);
        // pto: %max4_inline517_inline2017__ssa_v0
        ;
        Tile<TileType::Vec, float, 4, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v225 = Tile<TileType::Vec, float, 4, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v36, v33);
        // pto: %max4_inline517_inline2017__ssa_v0
        ;
        uint64_t v226 = (uint64_t) v21;
        TASSIGN(v225, v226);
        pipe_barrier(PIPE_V);
        TMAX(v225, v219, v223);
        // pto: %t__tmp_v119
        ;
        Tile<TileType::Vec, float, 2, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v227 = Tile<TileType::Vec, float, 2, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v45, v33);
        // pto: %t__tmp_v119
        ;
        uint64_t v228 = (uint64_t) v21;
        TASSIGN(v227, v228);
        // pto: %55
        ;
        Tile<TileType::Vec, float, 2, 512, BLayout::RowMajor, 2, 512, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v229;
        // pto: %55
        ;
        uint64_t v230 = (uint64_t) v21;
        TASSIGN(v229, v230);
        // pto: %t__tmp_v120
        ;
        Tile<TileType::Vec, float, 2, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v231 = Tile<TileType::Vec, float, 2, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v45, v33);
        // pto: %t__tmp_v120
        ;
        uint64_t v232 = (uint64_t) v27;
        TASSIGN(v231, v232);
        // pto: %56
        ;
        Tile<TileType::Vec, float, 2, 512, BLayout::RowMajor, 2, 512, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v233;
        // pto: %56
        ;
        uint64_t v234 = (uint64_t) v27;
        TASSIGN(v233, v234);
        // pto: %max2_inline509_inline1998__ssa_v0
        ;
        Tile<TileType::Vec, float, 2, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v235 = Tile<TileType::Vec, float, 2, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v45, v33);
        // pto: %max2_inline509_inline1998__ssa_v0
        ;
        uint64_t v236 = (uint64_t) v21;
        TASSIGN(v235, v236);
        pipe_barrier(PIPE_V);
        TMAX(v235, v229, v233);
        // pto: %t__tmp_v121
        ;
        Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v237 = Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v33);
        // pto: %t__tmp_v121
        ;
        uint64_t v238 = (uint64_t) v21;
        TASSIGN(v237, v238);
        // pto: %57
        ;
        Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, 1, 512, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v239;
        // pto: %57
        ;
        uint64_t v240 = (uint64_t) v21;
        TASSIGN(v239, v240);
        // pto: %t__tmp_v122
        ;
        Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v241 = Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v33);
        // pto: %t__tmp_v122
        ;
        uint64_t v242 = (uint64_t) v28;
        TASSIGN(v241, v242);
        // pto: %58
        ;
        Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, 1, 512, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v243;
        // pto: %58
        ;
        uint64_t v244 = (uint64_t) v28;
        TASSIGN(v243, v244);
        // pto: %maximum_inline502_inline1982__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v245 = Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v33);
        // pto: %maximum_inline502_inline1982__ssa_v0
        ;
        uint64_t v246 = (uint64_t) v21;
        TASSIGN(v245, v246);
        pipe_barrier(PIPE_V);
        TMAX(v245, v239, v243);
        // pto: %t__tmp_v123
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v247 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v41, v33);
        // pto: %t__tmp_v123
        ;
        uint64_t v248 = (uint64_t) v20;
        TASSIGN(v247, v248);
        pipe_barrier(PIPE_V);
        TCOLEXPANDSUB(v247, v121, v245);
        // pto: %probability_inline501_inline2012__ssa_v0
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v249 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v41, v33);
        // pto: %probability_inline501_inline2012__ssa_v0
        ;
        uint64_t v250 = (uint64_t) v20;
        TASSIGN(v249, v250);
        pipe_barrier(PIPE_V);
        TEXP(v249, v247);
        // pto: %t__tmp_v124
        ;
        Tile<TileType::Vec, float, 4, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v251 = Tile<TileType::Vec, float, 4, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v36, v33);
        // pto: %t__tmp_v124
        ;
        uint64_t v252 = (uint64_t) v20;
        TASSIGN(v251, v252);
        // pto: %59
        ;
        Tile<TileType::Vec, float, 4, 512, BLayout::RowMajor, 4, 512, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v253;
        // pto: %59
        ;
        uint64_t v254 = (uint64_t) v20;
        TASSIGN(v253, v254);
        // pto: %t__tmp_v125
        ;
        Tile<TileType::Vec, float, 4, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v255 = Tile<TileType::Vec, float, 4, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v36, v33);
        // pto: %t__tmp_v125
        ;
        uint64_t v256 = (uint64_t) v26;
        TASSIGN(v255, v256);
        // pto: %60
        ;
        Tile<TileType::Vec, float, 4, 512, BLayout::RowMajor, 4, 512, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v257;
        // pto: %60
        ;
        uint64_t v258 = (uint64_t) v26;
        TASSIGN(v257, v258);
        // pto: %sum4_inline519_inline2001__ssa_v0
        ;
        Tile<TileType::Vec, float, 4, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v259 = Tile<TileType::Vec, float, 4, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v36, v33);
        // pto: %sum4_inline519_inline2001__ssa_v0
        ;
        uint64_t v260 = (uint64_t) v21;
        TASSIGN(v259, v260);
        pipe_barrier(PIPE_V);
        TADD(v259, v253, v257);
        // pto: %t__tmp_v126
        ;
        Tile<TileType::Vec, float, 2, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v261 = Tile<TileType::Vec, float, 2, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v45, v33);
        // pto: %t__tmp_v126
        ;
        uint64_t v262 = (uint64_t) v21;
        TASSIGN(v261, v262);
        // pto: %61
        ;
        Tile<TileType::Vec, float, 2, 512, BLayout::RowMajor, 2, 512, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v263;
        // pto: %61
        ;
        uint64_t v264 = (uint64_t) v21;
        TASSIGN(v263, v264);
        // pto: %t__tmp_v127
        ;
        Tile<TileType::Vec, float, 2, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v265 = Tile<TileType::Vec, float, 2, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v45, v33);
        // pto: %t__tmp_v127
        ;
        uint64_t v266 = (uint64_t) v27;
        TASSIGN(v265, v266);
        // pto: %62
        ;
        Tile<TileType::Vec, float, 2, 512, BLayout::RowMajor, 2, 512, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v267;
        // pto: %62
        ;
        uint64_t v268 = (uint64_t) v27;
        TASSIGN(v267, v268);
        // pto: %sum2_inline500_inline1965__ssa_v0
        ;
        Tile<TileType::Vec, float, 2, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v269 = Tile<TileType::Vec, float, 2, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v45, v33);
        // pto: %sum2_inline500_inline1965__ssa_v0
        ;
        uint64_t v270 = (uint64_t) v21;
        TASSIGN(v269, v270);
        pipe_barrier(PIPE_V);
        TADD(v269, v263, v267);
        // pto: %t__tmp_v128
        ;
        Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v271 = Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v33);
        // pto: %t__tmp_v128
        ;
        uint64_t v272 = (uint64_t) v21;
        TASSIGN(v271, v272);
        // pto: %63
        ;
        Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, 1, 512, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v273;
        // pto: %63
        ;
        uint64_t v274 = (uint64_t) v21;
        TASSIGN(v273, v274);
        // pto: %t__tmp_v129
        ;
        Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v275 = Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v33);
        // pto: %t__tmp_v129
        ;
        uint64_t v276 = (uint64_t) v28;
        TASSIGN(v275, v276);
        // pto: %64
        ;
        Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, 1, 512, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v277;
        // pto: %64
        ;
        uint64_t v278 = (uint64_t) v28;
        TASSIGN(v277, v278);
        // pto: %total_inline499_inline2018__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v279 = Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v33);
        // pto: %total_inline499_inline2018__ssa_v0
        ;
        uint64_t v280 = (uint64_t) v21;
        TASSIGN(v279, v280);
        pipe_barrier(PIPE_V);
        TADD(v279, v273, v277);
        // pto: %probability_v1_inline507_inline2019__ssa_v0
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v281 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v41, v33);
        // pto: %probability_v1_inline507_inline2019__ssa_v0
        ;
        uint64_t v282 = (uint64_t) v20;
        TASSIGN(v281, v282);
        pipe_barrier(PIPE_V);
        TCOLEXPANDDIV(v281, v249, v279);
        // pto: %weighted_inline498_inline2020__ssa_v0
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v283 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v41, v33);
        // pto: %weighted_inline498_inline2020__ssa_v0
        ;
        uint64_t v284 = (uint64_t) v19;
        TASSIGN(v283, v284);
        pipe_barrier(PIPE_V);
        TMUL(v283, v119, v281);
        // pto: %t__tmp_v130
        ;
        Tile<TileType::Vec, float, 4, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v285 = Tile<TileType::Vec, float, 4, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v36, v33);
        // pto: %t__tmp_v130
        ;
        uint64_t v286 = (uint64_t) v19;
        TASSIGN(v285, v286);
        // pto: %65
        ;
        Tile<TileType::Vec, float, 4, 512, BLayout::RowMajor, 4, 512, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v287;
        // pto: %65
        ;
        uint64_t v288 = (uint64_t) v19;
        TASSIGN(v287, v288);
        // pto: %t__tmp_v131
        ;
        Tile<TileType::Vec, float, 4, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v289 = Tile<TileType::Vec, float, 4, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v36, v33);
        // pto: %t__tmp_v131
        ;
        uint64_t v290 = (uint64_t) v29;
        TASSIGN(v289, v290);
        // pto: %66
        ;
        Tile<TileType::Vec, float, 4, 512, BLayout::RowMajor, 4, 512, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v291;
        // pto: %66
        ;
        uint64_t v292 = (uint64_t) v29;
        TASSIGN(v291, v292);
        // pto: %weighted4_inline497_inline1985__ssa_v0
        ;
        Tile<TileType::Vec, float, 4, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v293 = Tile<TileType::Vec, float, 4, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v36, v33);
        // pto: %weighted4_inline497_inline1985__ssa_v0
        ;
        uint64_t v294 = (uint64_t) v19;
        TASSIGN(v293, v294);
        pipe_barrier(PIPE_V);
        TADD(v293, v287, v291);
        // pto: %t__tmp_v132
        ;
        Tile<TileType::Vec, float, 2, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v295 = Tile<TileType::Vec, float, 2, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v45, v33);
        // pto: %t__tmp_v132
        ;
        uint64_t v296 = (uint64_t) v19;
        TASSIGN(v295, v296);
        // pto: %67
        ;
        Tile<TileType::Vec, float, 2, 512, BLayout::RowMajor, 2, 512, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v297;
        // pto: %67
        ;
        uint64_t v298 = (uint64_t) v19;
        TASSIGN(v297, v298);
        // pto: %t__tmp_v133
        ;
        Tile<TileType::Vec, float, 2, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v299 = Tile<TileType::Vec, float, 2, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v45, v33);
        // pto: %t__tmp_v133
        ;
        uint64_t v300 = (uint64_t) v30;
        TASSIGN(v299, v300);
        // pto: %68
        ;
        Tile<TileType::Vec, float, 2, 512, BLayout::RowMajor, 2, 512, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v301;
        // pto: %68
        ;
        uint64_t v302 = (uint64_t) v30;
        TASSIGN(v301, v302);
        // pto: %weighted2_inline496_inline2003__ssa_v0
        ;
        Tile<TileType::Vec, float, 2, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v303 = Tile<TileType::Vec, float, 2, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v45, v33);
        // pto: %weighted2_inline496_inline2003__ssa_v0
        ;
        uint64_t v304 = (uint64_t) v19;
        TASSIGN(v303, v304);
        pipe_barrier(PIPE_V);
        TADD(v303, v297, v301);
        // pto: %t__tmp_v134
        ;
        Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v305 = Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v33);
        // pto: %t__tmp_v134
        ;
        uint64_t v306 = (uint64_t) v19;
        TASSIGN(v305, v306);
        // pto: %69
        ;
        Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, 1, 512, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v307;
        // pto: %69
        ;
        uint64_t v308 = (uint64_t) v19;
        TASSIGN(v307, v308);
        // pto: %t__tmp_v135
        ;
        Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v309 = Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v33);
        // pto: %t__tmp_v135
        ;
        uint64_t v310 = (uint64_t) v31;
        TASSIGN(v309, v310);
        // pto: %70
        ;
        Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, 1, 512, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v311;
        // pto: %70
        ;
        uint64_t v312 = (uint64_t) v31;
        TASSIGN(v311, v312);
        // pto: %pooled_row_inline516_inline1962__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v313 = Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v33);
        // pto: %pooled_row_inline516_inline1962__ssa_v0
        ;
        uint64_t v314 = (uint64_t) v19;
        TASSIGN(v313, v314);
        pipe_barrier(PIPE_V);
        TADD(v313, v307, v311);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
        TSTORE(v118, v313);
      };
      set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    };
  }
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  #endif // __DAV_VEC__

  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}