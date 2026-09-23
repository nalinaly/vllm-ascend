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

AICORE void scatter_softmax_pool_0(__gm__ float* v1, __gm__ float* v2, __gm__ float* v3, __gm__ int64_t* v4, __gm__ int32_t* v5, __gm__ float* v6, __gm__ float* v7, __gm__ float* v8, __gm__ float* v9, int64_t v10, int64_t v11, int64_t v12, int64_t v13, int64_t v14, int64_t v15, int64_t v16, int64_t v17, int32_t v18, int32_t v19) {
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  const int64_t v20 = -7;
  // pto: %c768_i64
  const int64_t v21 = 768;
  // pto: %c2816_i64
  const int64_t v22 = 2816;
  // pto: %c0_i64
  const int64_t v23 = 0;
  // pto: %c256_i64
  const int64_t v24 = 256;
  // pto: %c512_i64
  const int64_t v25 = 512;
  const int64_t v26 = 1792;
  const int64_t v27 = 3328;
  const int64_t v28 = 3072;
  const int64_t v29 = 1280;
  const int64_t v30 = 1024;
  // pto: %c384_index
  const int64_t v31 = 384;
  // pto: %c128_index
  const int64_t v32 = 128;
  // pto: %c1_index
  const int64_t v33 = 1;
  // pto: %c256_index
  const int64_t v34 = 256;
  // pto: %c4_index
  const int64_t v35 = 4;
  // pto: %c0_index
  const int64_t v36 = 0;
  // pto: %cst_16
  const float v37 = 0.0f;
  // pto: %c1_i64
  const int64_t v38 = 1;
  // pto: %c4_i64
  const int64_t v39 = 4;
  // pto: %c64_index
  const int64_t v40 = 64;
  // pto: %c8_index
  const int64_t v41 = 8;
  // pto: %cst_22
  const float v42 = -3.40282347E+38f;
  // pto: %c2_i64
  const int64_t v43 = 2;
  // pto: %c512_index
  const int64_t v44 = 512;
  // pto: %c2_index
  const int64_t v45 = 2;
  // pto: %pooled_kv_inline2148__ssa_v0_view
  const int64_t v46 = 1;
  // pto: %pooled_kv_inline2148__ssa_v0_view
  const int64_t v47 = 1;
  // pto: %pooled_kv_inline2148__ssa_v0_view
  const int64_t v48 = 1;
  // pto: %pooled_kv_inline2148__ssa_v0_view
  int64_t v49 = v31 * v32;
  // pto: %pooled_kv_inline2148__ssa_v0_view
  int64_t v50 = v48 * v49;
  // pto: %pooled_kv_inline2148__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v51 = pto::Shape<1, 1, 1, -1, -1>(v46, v47, v48, v31, v32);
  // pto: %pooled_kv_inline2148__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v52 = pto::Stride<-1, -1, -1, -1, -1>(v47 * v50, v50, v49, v32, v33);
  // pto: %pooled_kv_inline2148__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v53 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v1, v51, v52);
  // pto: %window_scores_inline454_inline2131__ssa_v0_view
  const int64_t v54 = 1;
  // pto: %window_scores_inline454_inline2131__ssa_v0_view
  const int64_t v55 = 1;
  // pto: %window_scores_inline454_inline2131__ssa_v0_view
  const int64_t v56 = 1;
  // pto: %window_scores_inline454_inline2131__ssa_v0_view
  int64_t v57 = v31 * v32;
  // pto: %window_scores_inline454_inline2131__ssa_v0_view
  int64_t v58 = v56 * v57;
  // pto: %window_scores_inline454_inline2131__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v59 = pto::Shape<1, 1, 1, -1, -1>(v54, v55, v56, v31, v32);
  // pto: %window_scores_inline454_inline2131__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v60 = pto::Stride<-1, -1, -1, -1, -1>(v55 * v58, v58, v57, v32, v33);
  // pto: %window_scores_inline454_inline2131__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v61 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v2, v59, v60);
  // pto: %window_values_inline439_inline2141__ssa_v0_view
  const int64_t v62 = 1;
  // pto: %window_values_inline439_inline2141__ssa_v0_view
  const int64_t v63 = 1;
  // pto: %window_values_inline439_inline2141__ssa_v0_view
  const int64_t v64 = 1;
  // pto: %window_values_inline439_inline2141__ssa_v0_view
  int64_t v65 = v31 * v32;
  // pto: %window_values_inline439_inline2141__ssa_v0_view
  int64_t v66 = v64 * v65;
  // pto: %window_values_inline439_inline2141__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v67 = pto::Shape<1, 1, 1, -1, -1>(v62, v63, v64, v31, v32);
  // pto: %window_values_inline439_inline2141__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v68 = pto::Stride<-1, -1, -1, -1, -1>(v63 * v66, v66, v65, v32, v33);
  // pto: %window_values_inline439_inline2141__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v69 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v3, v67, v68);
  // pto: %inner_compress_state__ssa_v0_view
  const int64_t v70 = 1;
  // pto: %inner_compress_state__ssa_v0_view
  const int64_t v71 = 1;
  // pto: %inner_compress_state__ssa_v0_view
  const int64_t v72 = 1;
  // pto: %inner_compress_state__ssa_v0_view
  int64_t v73 = (int64_t) v16;
  // pto: %inner_compress_state__ssa_v0_view
  int64_t v74 = (int64_t) v17;
  // pto: %inner_compress_state__ssa_v0_view
  int64_t v75 = v73 * v74;
  // pto: %inner_compress_state__ssa_v0_view
  int64_t v76 = v72 * v75;
  // pto: %inner_compress_state__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v77 = pto::Shape<1, 1, 1, -1, -1>(v70, v71, v72, v73, (int64_t) v17);
  // pto: %inner_compress_state__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v78 = pto::Stride<-1, -1, -1, -1, -1>(v71 * v76, v76, v75, v74, v33);
  // pto: %inner_compress_state__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v79 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v6, v77, v78);
  // pto: %kv_proj_pad_inline2147__rv_v2_view
  const int64_t v80 = 1;
  // pto: %kv_proj_pad_inline2147__rv_v2_view
  const int64_t v81 = 1;
  // pto: %kv_proj_pad_inline2147__rv_v2_view
  const int64_t v82 = 1;
  // pto: %kv_proj_pad_inline2147__rv_v2_view
  int64_t v83 = v31 * v34;
  // pto: %kv_proj_pad_inline2147__rv_v2_view
  int64_t v84 = v82 * v83;
  // pto: %kv_proj_pad_inline2147__rv_v2_view
  pto::Shape<1, 1, 1, -1, -1> v85 = pto::Shape<1, 1, 1, -1, -1>(v80, v81, v82, v31, v34);
  // pto: %kv_proj_pad_inline2147__rv_v2_view
  pto::Stride<-1, -1, -1, -1, -1> v86 = pto::Stride<-1, -1, -1, -1, -1>(v81 * v84, v84, v83, v34, v33);
  // pto: %kv_proj_pad_inline2147__rv_v2_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v87 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v7, v85, v86);
  // pto: %score_proj_pad_inline2182__rv_v2_view
  const int64_t v88 = 1;
  // pto: %score_proj_pad_inline2182__rv_v2_view
  const int64_t v89 = 1;
  // pto: %score_proj_pad_inline2182__rv_v2_view
  const int64_t v90 = 1;
  // pto: %score_proj_pad_inline2182__rv_v2_view
  int64_t v91 = v31 * v34;
  // pto: %score_proj_pad_inline2182__rv_v2_view
  int64_t v92 = v90 * v91;
  // pto: %score_proj_pad_inline2182__rv_v2_view
  pto::Shape<1, 1, 1, -1, -1> v93 = pto::Shape<1, 1, 1, -1, -1>(v88, v89, v90, v31, v34);
  // pto: %score_proj_pad_inline2182__rv_v2_view
  pto::Stride<-1, -1, -1, -1, -1> v94 = pto::Stride<-1, -1, -1, -1, -1>(v89 * v92, v92, v91, v34, v33);
  // pto: %score_proj_pad_inline2182__rv_v2_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v95 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v8, v93, v94);
  // pto: %inner_ape__ssa_v0_view
  const int64_t v96 = 1;
  // pto: %inner_ape__ssa_v0_view
  const int64_t v97 = 1;
  // pto: %inner_ape__ssa_v0_view
  const int64_t v98 = 1;
  // pto: %inner_ape__ssa_v0_view
  int64_t v99 = v35 * v34;
  // pto: %inner_ape__ssa_v0_view
  int64_t v100 = v98 * v99;
  // pto: %inner_ape__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v101 = pto::Shape<1, 1, 1, -1, -1>(v96, v97, v98, v35, v34);
  // pto: %inner_ape__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v102 = pto::Stride<-1, -1, -1, -1, -1>(v97 * v100, v100, v99, v34, v33);
  // pto: %inner_ape__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v103 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v9, v101, v102);
  // pto: %score_inline406_inline2174__phi_v4
  Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v104 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v33, v40);
  // pto: %score_inline406_inline2174__phi_v4
  uint64_t v105 = (uint64_t) v25;
  TASSIGN(v104, v105);
  // pto: %value_inline409_inline2193__phi_v3
  Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v106 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v33, v40);
  // pto: %value_inline409_inline2193__phi_v3
  uint64_t v107 = (uint64_t) v24;
  TASSIGN(v106, v107);
  // pto: %score_inline406_inline2174__phi_v3
  Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v108 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v33, v40);
  // pto: %score_inline406_inline2174__phi_v3
  uint64_t v109 = (uint64_t) v25;
  TASSIGN(v108, v109);
  // pto: %value_inline409_inline2193__phi_v2
  Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v110 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v33, v40);
  // pto: %value_inline409_inline2193__phi_v2
  uint64_t v111 = (uint64_t) v24;
  TASSIGN(v110, v111);
  // pto: %score_inline406_inline2174__phi_v7
  Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v112 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v33, v40);
  // pto: %score_inline406_inline2174__phi_v7
  uint64_t v113 = (uint64_t) v22;
  TASSIGN(v112, v113);
  // pto: %value_inline409_inline2193__phi_v6
  Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v114 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v33, v40);
  // pto: %value_inline409_inline2193__phi_v6
  uint64_t v115 = (uint64_t) v21;
  TASSIGN(v114, v115);
  // pto: %score_inline406_inline2174__phi_v6
  Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v116 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v33, v40);
  // pto: %score_inline406_inline2174__phi_v6
  uint64_t v117 = (uint64_t) v22;
  TASSIGN(v116, v117);
  // pto: %value_inline409_inline2193__phi_v5
  Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v118 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v33, v40);
  // pto: %value_inline409_inline2193__phi_v5
  uint64_t v119 = (uint64_t) v21;
  TASSIGN(v118, v119);
  // pto: %pool_worker_inline451_inline2134__ssa_v0
  int64_t v120 = (int64_t) v18;
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID3);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID4);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  for (int64_t v121 = v120; v121 < v10; v121 += v11) {
    // pto: %31
    ;
    int64_t v122 = (int64_t) ((uint64_t) v121 * (uint64_t) v12);
    // pto: %first_pos_b_inline436_inline2155__tile
    ;
    int64_t v123 = (v4)[v122];
    for (int64_t v124 = v36; v124 < v12; v124 += v33) {
      // pto: %33
      ;
      int64_t v125 = (int64_t) ((uint64_t) v122 + (uint64_t) v124);
      // pto: %token_pos_inline440_inline2175__tile
      ;
      int64_t v126 = (v4)[v125];
      // pto: %t__tile
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v127 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v33, v32);
      // pto: %t__tile
      ;
      uint64_t v128 = (uint64_t) v21;
      TASSIGN(v127, v128);
      wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
      TEXPANDS(v127, v37);
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      // pto: %34
      ;
      int64_t v129 = v125 < v36 ? v36 : v125;
      // pto: %pooled_kv_inline2148__iter_v3_pview
      ;
      __gm__ float* v130 = PTOAS__GLOBAL_TENSOR_DATA(v53);
      // pto: %pooled_kv_inline2148__iter_v3_pview
      ;
      const int64_t v131 = 0;
      // pto: %pooled_kv_inline2148__iter_v3_pview
      ;
      const int64_t v132 = 128;
      // pto: %pooled_kv_inline2148__iter_v3_pview
      ;
      pto::Shape<1, 1, 1, 1, 128> v133 = pto::Shape<1, 1, 1, 1, 128>();
      // pto: %pooled_kv_inline2148__iter_v3_pview
      ;
      pto::Stride<128, 128, 128, 128, 1> v134 = pto::Stride<128, 128, 128, 128, 1>();
      // pto: %pooled_kv_inline2148__iter_v3_pview
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<128, 128, 128, 128, 1>, pto::Layout::ND> v135 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<128, 128, 128, 128, 1>, pto::Layout::ND>(v130 + (v131 + v129 * v132), v133, v134);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      TSTORE(v135, v127);
      set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
      // pto: %35, %36, %37
      ;
      wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
      if ((int64_t) ((uint64_t) v126 + (uint64_t) v38) % v39 == v23) {
        for (int64_t v136 = v36; v136 < v32; v136 += v40) {
          // pto: %40
          ;
          int64_t v137 = (int64_t) ((uint64_t) v120 * (uint64_t) v41);
          wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
          for (int64_t v138 = v36; v138 < v41; v138 += v33) {
            // pto: %42
            ;
            int64_t v139 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) v126 + (uint64_t) v20) + (uint64_t) v138);
            // pto: %value_inline409_inline2193__tile
            ;
            Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v140 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v33, v40);
            // pto: %value_inline409_inline2193__tile
            ;
            uint64_t v141 = (uint64_t) v21;
            TASSIGN(v140, v141);
            wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID3);
            pipe_barrier(PIPE_V);
            TEXPANDS(v140, v37);
            // pto: %score_inline406_inline2174__tile
            ;
            Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v142 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v33, v40);
            // pto: %score_inline406_inline2174__tile
            ;
            uint64_t v143 = (uint64_t) v22;
            TASSIGN(v142, v143);
            wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID4);
            TEXPANDS(v142, v42);
            // pto: %43, %state_half_inline405_inline2145__phi_v2
            ;
            int64_t v144 = v138 >= v35 ? v32 : v36;
            // pto: %45, %46, %47
            ;
            wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
            if (v139 >= v36 & v139 < v123) {
              // pto: %0
              ;
              Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v145 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v33, v40);
              // pto: %0
              ;
              uint64_t v146 = (uint64_t) v23;
              TASSIGN(v145, v146);
              TEXPANDS(v145, v37);
              // pto: %flat_offset_mul, %flat_offset, %49, %48
              ;
              int32_t v147 = (v5)[(int64_t) ((uint64_t) ((int64_t) (uint64_t) v121 * (uint64_t) v15) + (uint64_t) (v139 / v43))];
              // pto: %51
              ;
              int64_t v148 = (int64_t) v147;
              // pto: %52
              ;
              if (v148 >= v36) {
                // pto: %53, %55, %56, %57
                ;
                int64_t v149 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) ((int64_t) (uint64_t) (v139 % v43) * (uint64_t) v44) + (uint64_t) v144) + (uint64_t) v136);
                // pto: %1
                ;
                Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v150 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v33, v40);
                // pto: %1
                ;
                uint64_t v151 = (uint64_t) v24;
                TASSIGN(v150, v151);
                // pto: %58
                ;
                int64_t v152 = v148 < v36 ? v36 : v148;
                // pto: %59
                ;
                int64_t v153 = v149 < v36 ? v36 : v149;
                // pto: %inner_compress_state__ssa_v0_pview
                ;
                int64_t v154 = (int64_t) v17;
                // pto: %inner_compress_state__ssa_v0_pview
                ;
                const int64_t v155 = 0;
                // pto: %inner_compress_state__ssa_v0_pview
                ;
                __gm__ float* v156 = PTOAS__GLOBAL_TENSOR_DATA(v79);
                // pto: %inner_compress_state__ssa_v0_pview
                ;
                const int64_t v157 = 1;
                // pto: %inner_compress_state__ssa_v0_pview
                ;
                const int64_t v158 = 1;
                // pto: %inner_compress_state__ssa_v0_pview
                ;
                const int64_t v159 = 1;
                // pto: %inner_compress_state__ssa_v0_pview
                ;
                int64_t v160 = v33 * v154;
                // pto: %inner_compress_state__ssa_v0_pview
                ;
                int64_t v161 = v159 * v160;
                // pto: %inner_compress_state__ssa_v0_pview
                ;
                pto::Shape<1, 1, 1, 1, 64> v162 = pto::Shape<1, 1, 1, 1, 64>(v157, v158, v159, v33, v40);
                // pto: %inner_compress_state__ssa_v0_pview
                ;
                pto::Stride<-1, -1, -1, -1, -1> v163 = pto::Stride<-1, -1, -1, -1, -1>(v158 * v161, v161, v160, v154, v33);
                // pto: %inner_compress_state__ssa_v0_pview
                ;
                GlobalTensor<float, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v164 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v156 + (v155 + v152 * v154 + v153 * v33), v162, v163);
                TLOAD(v150, v164);
                // pto: %2
                ;
                Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v165 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v33, v40);
                // pto: %2
                ;
                uint64_t v166 = (uint64_t) v25;
                TASSIGN(v165, v166);
                // pto: %61
                ;
                int64_t v167 = (int64_t) ((uint64_t) v149 + (uint64_t) v34);
                // pto: %62
                ;
                int64_t v168 = v167 < v36 ? v36 : v167;
                // pto: %63
                ;
                int64_t v169 = (int64_t) v17;
                // pto: %63
                ;
                const int64_t v170 = 0;
                // pto: %63
                ;
                __gm__ float* v171 = PTOAS__GLOBAL_TENSOR_DATA(v79);
                // pto: %63
                ;
                const int64_t v172 = 1;
                // pto: %63
                ;
                const int64_t v173 = 1;
                // pto: %63
                ;
                const int64_t v174 = 1;
                // pto: %63
                ;
                int64_t v175 = v33 * v169;
                // pto: %63
                ;
                int64_t v176 = v174 * v175;
                // pto: %63
                ;
                pto::Shape<1, 1, 1, 1, 64> v177 = pto::Shape<1, 1, 1, 1, 64>(v172, v173, v174, v33, v40);
                // pto: %63
                ;
                pto::Stride<-1, -1, -1, -1, -1> v178 = pto::Stride<-1, -1, -1, -1, -1>(v173 * v176, v176, v175, v169, v33);
                // pto: %63
                ;
                GlobalTensor<float, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v179 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v171 + (v170 + v152 * v169 + v168 * v33), v177, v178);
                TLOAD(v165, v179);
              } else {
                // pto: %score_inline406_inline2174__tile_mv
                ;
                Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v180 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v33, v40);
                // pto: %score_inline406_inline2174__tile_mv
                ;
                uint64_t v181 = (uint64_t) v25;
                TASSIGN(v180, v181);
                pipe_barrier(PIPE_V);
                TMOV(v180, v145);
                // pto: %value_inline409_inline2193__tile_mv
                ;
                Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v182 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v33, v40);
                // pto: %value_inline409_inline2193__tile_mv
                ;
                uint64_t v183 = (uint64_t) v24;
                TASSIGN(v182, v183);
                TMOV(v182, v140);
              };
            } else {
              // pto: %3
              ;
              Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v184 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v33, v40);
              // pto: %3
              ;
              uint64_t v185 = (uint64_t) v25;
              TASSIGN(v184, v185);
              pipe_barrier(PIPE_V);
              TMOV(v184, v142);
              // pto: %4
              ;
              Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v186 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v33, v40);
              // pto: %4
              ;
              uint64_t v187 = (uint64_t) v24;
              TASSIGN(v186, v187);
              TMOV(v186, v140);
            };
            set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            // pto: %64
            ;
            wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            if (v123 <= v139) {
              // pto: %65
              ;
              if (v139 <= v126) {
                // pto: %68, %70
                ;
                int64_t v188 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) v122 + (uint64_t) v139) - (uint64_t) v123);
                // pto: %71
                ;
                int64_t v189 = v139 % v39;
                // pto: %5
                ;
                Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v190 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v33, v40);
                // pto: %5
                ;
                uint64_t v191 = (uint64_t) v21;
                TASSIGN(v190, v191);
                // pto: %73
                ;
                int64_t v192 = v188 < v36 ? v36 : v188;
                // pto: %74
                ;
                int64_t v193 = (int64_t) ((uint64_t) v144 + (uint64_t) v136);
                // pto: %75
                ;
                int64_t v194 = v193 < v36 ? v36 : v193;
                // pto: %kv_proj_pad_inline2147__rv_v2_pview
                ;
                __gm__ float* v195 = PTOAS__GLOBAL_TENSOR_DATA(v87);
                // pto: %kv_proj_pad_inline2147__rv_v2_pview
                ;
                const int64_t v196 = 0;
                // pto: %kv_proj_pad_inline2147__rv_v2_pview
                ;
                const int64_t v197 = 256;
                // pto: %kv_proj_pad_inline2147__rv_v2_pview
                ;
                pto::Shape<1, 1, 1, 1, 64> v198 = pto::Shape<1, 1, 1, 1, 64>();
                // pto: %kv_proj_pad_inline2147__rv_v2_pview
                ;
                pto::Stride<256, 256, 256, 256, 1> v199 = pto::Stride<256, 256, 256, 256, 1>();
                // pto: %kv_proj_pad_inline2147__rv_v2_pview
                ;
                GlobalTensor<float, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<256, 256, 256, 256, 1>, pto::Layout::ND> v200 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<256, 256, 256, 256, 1>, pto::Layout::ND>(v195 + (v196 + v192 * v197 + v194), v198, v199);
                TLOAD(v190, v200);
                // pto: %6
                ;
                Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v201 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v33, v40);
                // pto: %6
                ;
                uint64_t v202 = (uint64_t) v22;
                TASSIGN(v201, v202);
                // pto: %score_proj_pad_inline2182__rv_v2_pview
                ;
                __gm__ float* v203 = PTOAS__GLOBAL_TENSOR_DATA(v95);
                // pto: %score_proj_pad_inline2182__rv_v2_pview
                ;
                const int64_t v204 = 0;
                // pto: %score_proj_pad_inline2182__rv_v2_pview
                ;
                const int64_t v205 = 256;
                // pto: %score_proj_pad_inline2182__rv_v2_pview
                ;
                pto::Shape<1, 1, 1, 1, 64> v206 = pto::Shape<1, 1, 1, 1, 64>();
                // pto: %score_proj_pad_inline2182__rv_v2_pview
                ;
                pto::Stride<256, 256, 256, 256, 1> v207 = pto::Stride<256, 256, 256, 256, 1>();
                // pto: %score_proj_pad_inline2182__rv_v2_pview
                ;
                GlobalTensor<float, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<256, 256, 256, 256, 1>, pto::Layout::ND> v208 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<256, 256, 256, 256, 1>, pto::Layout::ND>(v203 + (v204 + v192 * v205 + v194), v206, v207);
                TLOAD(v201, v208);
                // pto: %7
                ;
                Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v209 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v33, v40);
                // pto: %7
                ;
                uint64_t v210 = (uint64_t) v23;
                TASSIGN(v209, v210);
                // pto: %79
                ;
                int64_t v211 = v189 < v36 ? v36 : v189;
                // pto: %inner_ape__ssa_v0_pview
                ;
                __gm__ float* v212 = PTOAS__GLOBAL_TENSOR_DATA(v103);
                // pto: %inner_ape__ssa_v0_pview
                ;
                const int64_t v213 = 0;
                // pto: %inner_ape__ssa_v0_pview
                ;
                const int64_t v214 = 256;
                // pto: %inner_ape__ssa_v0_pview
                ;
                pto::Shape<1, 1, 1, 1, 64> v215 = pto::Shape<1, 1, 1, 1, 64>();
                // pto: %inner_ape__ssa_v0_pview
                ;
                pto::Stride<256, 256, 256, 256, 1> v216 = pto::Stride<256, 256, 256, 256, 1>();
                // pto: %inner_ape__ssa_v0_pview
                ;
                GlobalTensor<float, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<256, 256, 256, 256, 1>, pto::Layout::ND> v217 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<256, 256, 256, 256, 1>, pto::Layout::ND>(v212 + (v213 + v211 * v214 + v194), v215, v216);
                TLOAD(v209, v217);
                set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
                // pto: %8
                ;
                Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v218 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v33, v40);
                // pto: %8
                ;
                uint64_t v219 = (uint64_t) v22;
                TASSIGN(v218, v219);
                wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
                TADD(v218, v201, v209);
              } else {
                // pto: %score_inline406_inline2174__phi_v4_mv
                ;
                Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v220 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v33, v40);
                // pto: %score_inline406_inline2174__phi_v4_mv
                ;
                uint64_t v221 = (uint64_t) v22;
                TASSIGN(v220, v221);
                pipe_barrier(PIPE_V);
                TMOV(v220, v104);
                // pto: %value_inline409_inline2193__phi_v3_mv
                ;
                Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v222 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v33, v40);
                // pto: %value_inline409_inline2193__phi_v3_mv
                ;
                uint64_t v223 = (uint64_t) v21;
                TASSIGN(v222, v223);
                TMOV(v222, v106);
              };
            } else {
              // pto: %9
              ;
              Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v224 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v33, v40);
              // pto: %9
              ;
              uint64_t v225 = (uint64_t) v22;
              TASSIGN(v224, v225);
              pipe_barrier(PIPE_V);
              TMOV(v224, v104);
              // pto: %10
              ;
              Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v226 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v33, v40);
              // pto: %10
              ;
              uint64_t v227 = (uint64_t) v21;
              TASSIGN(v226, v227);
              TMOV(v226, v106);
            };
            set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
            set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            // pto: %86, %82, %83, %85, %84
            ;
            int64_t v228 = (int64_t) ((uint64_t) v137 + (uint64_t) ((int64_t) (uint64_t) ((int64_t) (uint64_t) (v138 % v35) * (uint64_t) v45) + (uint64_t) (v138 / v35)));
            // pto: %87
            ;
            int64_t v229 = v228 < v36 ? v36 : v228;
            // pto: %88
            ;
            int64_t v230 = v136 < v36 ? v36 : v136;
            // pto: %window_values_inline439_inline2141__iter_v7_pview
            ;
            __gm__ float* v231 = PTOAS__GLOBAL_TENSOR_DATA(v69);
            // pto: %window_values_inline439_inline2141__iter_v7_pview
            ;
            const int64_t v232 = 0;
            // pto: %window_values_inline439_inline2141__iter_v7_pview
            ;
            const int64_t v233 = 128;
            // pto: %window_values_inline439_inline2141__iter_v7_pview
            ;
            pto::Shape<1, 1, 1, 1, 64> v234 = pto::Shape<1, 1, 1, 1, 64>();
            // pto: %window_values_inline439_inline2141__iter_v7_pview
            ;
            pto::Stride<128, 128, 128, 128, 1> v235 = pto::Stride<128, 128, 128, 128, 1>();
            // pto: %window_values_inline439_inline2141__iter_v7_pview
            ;
            GlobalTensor<float, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<128, 128, 128, 128, 1>, pto::Layout::ND> v236 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<128, 128, 128, 128, 1>, pto::Layout::ND>(v231 + (v232 + v229 * v233 + v230), v234, v235);
            wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
            TSTORE(v236, v114);
            set_flag(PIPE_MTE3, PIPE_V, EVENT_ID3);
            // pto: %window_scores_inline454_inline2131__iter_v7_pview
            ;
            __gm__ float* v237 = PTOAS__GLOBAL_TENSOR_DATA(v61);
            // pto: %window_scores_inline454_inline2131__iter_v7_pview
            ;
            const int64_t v238 = 0;
            // pto: %window_scores_inline454_inline2131__iter_v7_pview
            ;
            const int64_t v239 = 128;
            // pto: %window_scores_inline454_inline2131__iter_v7_pview
            ;
            pto::Shape<1, 1, 1, 1, 64> v240 = pto::Shape<1, 1, 1, 1, 64>();
            // pto: %window_scores_inline454_inline2131__iter_v7_pview
            ;
            pto::Stride<128, 128, 128, 128, 1> v241 = pto::Stride<128, 128, 128, 128, 1>();
            // pto: %window_scores_inline454_inline2131__iter_v7_pview
            ;
            GlobalTensor<float, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<128, 128, 128, 128, 1>, pto::Layout::ND> v242 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<128, 128, 128, 128, 1>, pto::Layout::ND>(v237 + (v238 + v229 * v239 + v230), v240, v241);
            TSTORE(v242, v112);
            set_flag(PIPE_MTE3, PIPE_V, EVENT_ID4);
          };
          set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
          // pto: %score_rows_inline460_inline2149__tile
          ;
          Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v243 = Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v41, v40);
          // pto: %score_rows_inline460_inline2149__tile
          ;
          uint64_t v244 = (uint64_t) v21;
          TASSIGN(v243, v244);
          // pto: %91
          ;
          int64_t v245 = v137 < v36 ? v36 : v137;
          // pto: %92
          ;
          int64_t v246 = v136 < v36 ? v36 : v136;
          // pto: %window_scores_inline454_inline2131__rv_v8_pview
          ;
          __gm__ float* v247 = PTOAS__GLOBAL_TENSOR_DATA(v61);
          // pto: %window_scores_inline454_inline2131__rv_v8_pview
          ;
          const int64_t v248 = 0;
          // pto: %window_scores_inline454_inline2131__rv_v8_pview
          ;
          const int64_t v249 = 128;
          // pto: %window_scores_inline454_inline2131__rv_v8_pview
          ;
          pto::Shape<1, 1, 1, 8, 64> v250 = pto::Shape<1, 1, 1, 8, 64>();
          // pto: %window_scores_inline454_inline2131__rv_v8_pview
          ;
          pto::Stride<1024, 1024, 1024, 128, 1> v251 = pto::Stride<1024, 1024, 1024, 128, 1>();
          // pto: %window_scores_inline454_inline2131__rv_v8_pview
          ;
          GlobalTensor<float, pto::Shape<1, 1, 1, 8, 64>, pto::Stride<1024, 1024, 1024, 128, 1>, pto::Layout::ND> v252 = GlobalTensor<float, pto::Shape<1, 1, 1, 8, 64>, pto::Stride<1024, 1024, 1024, 128, 1>, pto::Layout::ND>(v247 + (v248 + v245 * v249 + v246), v250, v251);
          wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
          TLOAD(v243, v252);
          set_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
          // pto: %11
          ;
          Tile<TileType::Vec, float, 4, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v253 = Tile<TileType::Vec, float, 4, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v35, v40);
          // pto: %11
          ;
          uint64_t v254 = (uint64_t) v21;
          TASSIGN(v253, v254);
          // pto: %slice_view
          ;
          Tile<TileType::Vec, float, 4, 64, BLayout::RowMajor, 4, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v255;
          // pto: %slice_view
          ;
          uint64_t v256 = (uint64_t) v21;
          TASSIGN(v255, v256);
          // pto: %12
          ;
          Tile<TileType::Vec, float, 4, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v257 = Tile<TileType::Vec, float, 4, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v35, v40);
          // pto: %12
          ;
          uint64_t v258 = (uint64_t) v26;
          TASSIGN(v257, v258);
          // pto: %93
          ;
          Tile<TileType::Vec, float, 4, 64, BLayout::RowMajor, 4, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v259;
          // pto: %93
          ;
          uint64_t v260 = (uint64_t) v26;
          TASSIGN(v259, v260);
          // pto: %max4_inline461_inline2183__tile
          ;
          Tile<TileType::Vec, float, 4, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v261 = Tile<TileType::Vec, float, 4, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v35, v40);
          // pto: %max4_inline461_inline2183__tile
          ;
          uint64_t v262 = (uint64_t) v22;
          TASSIGN(v261, v262);
          wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
          TMAX(v261, v255, v259);
          // pto: %13
          ;
          Tile<TileType::Vec, float, 2, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v263 = Tile<TileType::Vec, float, 2, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v45, v40);
          // pto: %13
          ;
          uint64_t v264 = (uint64_t) v22;
          TASSIGN(v263, v264);
          // pto: %94
          ;
          Tile<TileType::Vec, float, 2, 64, BLayout::RowMajor, 2, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v265;
          // pto: %94
          ;
          uint64_t v266 = (uint64_t) v22;
          TASSIGN(v265, v266);
          // pto: %14
          ;
          Tile<TileType::Vec, float, 2, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v267 = Tile<TileType::Vec, float, 2, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v45, v40);
          // pto: %14
          ;
          uint64_t v268 = (uint64_t) v27;
          TASSIGN(v267, v268);
          // pto: %95
          ;
          Tile<TileType::Vec, float, 2, 64, BLayout::RowMajor, 2, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v269;
          // pto: %95
          ;
          uint64_t v270 = (uint64_t) v27;
          TASSIGN(v269, v270);
          // pto: %max2_inline463_inline2178__tile
          ;
          Tile<TileType::Vec, float, 2, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v271 = Tile<TileType::Vec, float, 2, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v45, v40);
          // pto: %max2_inline463_inline2178__tile
          ;
          uint64_t v272 = (uint64_t) v22;
          TASSIGN(v271, v272);
          pipe_barrier(PIPE_V);
          TMAX(v271, v265, v269);
          // pto: %15
          ;
          Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v273 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v33, v40);
          // pto: %15
          ;
          uint64_t v274 = (uint64_t) v22;
          TASSIGN(v273, v274);
          // pto: %96
          ;
          Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v275;
          // pto: %96
          ;
          uint64_t v276 = (uint64_t) v22;
          TASSIGN(v275, v276);
          // pto: %16
          ;
          Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v277 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v33, v40);
          // pto: %16
          ;
          uint64_t v278 = (uint64_t) v28;
          TASSIGN(v277, v278);
          // pto: %97
          ;
          Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v279;
          // pto: %97
          ;
          uint64_t v280 = (uint64_t) v28;
          TASSIGN(v279, v280);
          // pto: %maximum_inline425_inline2199__tile
          ;
          Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v281 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v33, v40);
          // pto: %maximum_inline425_inline2199__tile
          ;
          uint64_t v282 = (uint64_t) v22;
          TASSIGN(v281, v282);
          pipe_barrier(PIPE_V);
          TMAX(v281, v275, v279);
          // pto: %17
          ;
          Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v283 = Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v41, v40);
          // pto: %17
          ;
          uint64_t v284 = (uint64_t) v21;
          TASSIGN(v283, v284);
          pipe_barrier(PIPE_V);
          TCOLEXPANDSUB(v283, v243, v281);
          // pto: %probability_inline464_inline2201__tile
          ;
          Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v285 = Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v41, v40);
          // pto: %probability_inline464_inline2201__tile
          ;
          uint64_t v286 = (uint64_t) v21;
          TASSIGN(v285, v286);
          pipe_barrier(PIPE_V);
          TEXP(v285, v283);
          // pto: %18
          ;
          Tile<TileType::Vec, float, 4, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v287 = Tile<TileType::Vec, float, 4, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v35, v40);
          // pto: %18
          ;
          uint64_t v288 = (uint64_t) v21;
          TASSIGN(v287, v288);
          // pto: %98
          ;
          Tile<TileType::Vec, float, 4, 64, BLayout::RowMajor, 4, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v289;
          // pto: %98
          ;
          uint64_t v290 = (uint64_t) v21;
          TASSIGN(v289, v290);
          // pto: %19
          ;
          Tile<TileType::Vec, float, 4, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v291 = Tile<TileType::Vec, float, 4, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v35, v40);
          // pto: %19
          ;
          uint64_t v292 = (uint64_t) v26;
          TASSIGN(v291, v292);
          // pto: %99
          ;
          Tile<TileType::Vec, float, 4, 64, BLayout::RowMajor, 4, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v293;
          // pto: %99
          ;
          uint64_t v294 = (uint64_t) v26;
          TASSIGN(v293, v294);
          // pto: %sum4_inline465_inline2203__tile
          ;
          Tile<TileType::Vec, float, 4, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v295 = Tile<TileType::Vec, float, 4, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v35, v40);
          // pto: %sum4_inline465_inline2203__tile
          ;
          uint64_t v296 = (uint64_t) v22;
          TASSIGN(v295, v296);
          pipe_barrier(PIPE_V);
          TADD(v295, v289, v293);
          // pto: %20
          ;
          Tile<TileType::Vec, float, 2, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v297 = Tile<TileType::Vec, float, 2, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v45, v40);
          // pto: %20
          ;
          uint64_t v298 = (uint64_t) v22;
          TASSIGN(v297, v298);
          // pto: %100
          ;
          Tile<TileType::Vec, float, 2, 64, BLayout::RowMajor, 2, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v299;
          // pto: %100
          ;
          uint64_t v300 = (uint64_t) v22;
          TASSIGN(v299, v300);
          // pto: %21
          ;
          Tile<TileType::Vec, float, 2, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v301 = Tile<TileType::Vec, float, 2, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v45, v40);
          // pto: %21
          ;
          uint64_t v302 = (uint64_t) v27;
          TASSIGN(v301, v302);
          // pto: %101
          ;
          Tile<TileType::Vec, float, 2, 64, BLayout::RowMajor, 2, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v303;
          // pto: %101
          ;
          uint64_t v304 = (uint64_t) v27;
          TASSIGN(v303, v304);
          // pto: %sum2_inline468_inline2204__tile
          ;
          Tile<TileType::Vec, float, 2, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v305 = Tile<TileType::Vec, float, 2, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v45, v40);
          // pto: %sum2_inline468_inline2204__tile
          ;
          uint64_t v306 = (uint64_t) v22;
          TASSIGN(v305, v306);
          pipe_barrier(PIPE_V);
          TADD(v305, v299, v303);
          // pto: %22
          ;
          Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v307 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v33, v40);
          // pto: %22
          ;
          uint64_t v308 = (uint64_t) v22;
          TASSIGN(v307, v308);
          // pto: %102
          ;
          Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v309;
          // pto: %102
          ;
          uint64_t v310 = (uint64_t) v22;
          TASSIGN(v309, v310);
          // pto: %23
          ;
          Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v311 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v33, v40);
          // pto: %23
          ;
          uint64_t v312 = (uint64_t) v28;
          TASSIGN(v311, v312);
          // pto: %103
          ;
          Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v313;
          // pto: %103
          ;
          uint64_t v314 = (uint64_t) v28;
          TASSIGN(v313, v314);
          // pto: %total_inline469_inline2163__tile
          ;
          Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v315 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v33, v40);
          // pto: %total_inline469_inline2163__tile
          ;
          uint64_t v316 = (uint64_t) v22;
          TASSIGN(v315, v316);
          pipe_barrier(PIPE_V);
          TADD(v315, v309, v313);
          // pto: %probability_v1_inline470_inline2205__tile
          ;
          Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v317 = Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v41, v40);
          // pto: %probability_v1_inline470_inline2205__tile
          ;
          uint64_t v318 = (uint64_t) v21;
          TASSIGN(v317, v318);
          pipe_barrier(PIPE_V);
          TCOLEXPANDDIV(v317, v285, v315);
          set_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
          // pto: %value_rows_inline467_inline2139__tile
          ;
          Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v319 = Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v41, v40);
          // pto: %value_rows_inline467_inline2139__tile
          ;
          uint64_t v320 = (uint64_t) v22;
          TASSIGN(v319, v320);
          // pto: %window_values_inline439_inline2141__rv_v8_pview
          ;
          __gm__ float* v321 = PTOAS__GLOBAL_TENSOR_DATA(v69);
          // pto: %window_values_inline439_inline2141__rv_v8_pview
          ;
          const int64_t v322 = 0;
          // pto: %window_values_inline439_inline2141__rv_v8_pview
          ;
          const int64_t v323 = 128;
          // pto: %window_values_inline439_inline2141__rv_v8_pview
          ;
          pto::Shape<1, 1, 1, 8, 64> v324 = pto::Shape<1, 1, 1, 8, 64>();
          // pto: %window_values_inline439_inline2141__rv_v8_pview
          ;
          pto::Stride<1024, 1024, 1024, 128, 1> v325 = pto::Stride<1024, 1024, 1024, 128, 1>();
          // pto: %window_values_inline439_inline2141__rv_v8_pview
          ;
          GlobalTensor<float, pto::Shape<1, 1, 1, 8, 64>, pto::Stride<1024, 1024, 1024, 128, 1>, pto::Layout::ND> v326 = GlobalTensor<float, pto::Shape<1, 1, 1, 8, 64>, pto::Stride<1024, 1024, 1024, 128, 1>, pto::Layout::ND>(v321 + (v322 + v245 * v323 + v246), v324, v325);
          wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
          TLOAD(v319, v326);
          set_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
          // pto: %weighted_inline445_inline2158__tile
          ;
          Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v327 = Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v41, v40);
          // pto: %weighted_inline445_inline2158__tile
          ;
          uint64_t v328 = (uint64_t) v21;
          TASSIGN(v327, v328);
          wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
          TMUL(v327, v319, v317);
          // pto: %24
          ;
          Tile<TileType::Vec, float, 4, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v329 = Tile<TileType::Vec, float, 4, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v35, v40);
          // pto: %24
          ;
          uint64_t v330 = (uint64_t) v21;
          TASSIGN(v329, v330);
          // pto: %106
          ;
          Tile<TileType::Vec, float, 4, 64, BLayout::RowMajor, 4, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v331;
          // pto: %106
          ;
          uint64_t v332 = (uint64_t) v21;
          TASSIGN(v331, v332);
          // pto: %25
          ;
          Tile<TileType::Vec, float, 4, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v333 = Tile<TileType::Vec, float, 4, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v35, v40);
          // pto: %25
          ;
          uint64_t v334 = (uint64_t) v26;
          TASSIGN(v333, v334);
          // pto: %107
          ;
          Tile<TileType::Vec, float, 4, 64, BLayout::RowMajor, 4, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v335;
          // pto: %107
          ;
          uint64_t v336 = (uint64_t) v26;
          TASSIGN(v335, v336);
          // pto: %weighted4_inline448_inline2202__tile
          ;
          Tile<TileType::Vec, float, 4, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v337 = Tile<TileType::Vec, float, 4, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v35, v40);
          // pto: %weighted4_inline448_inline2202__tile
          ;
          uint64_t v338 = (uint64_t) v21;
          TASSIGN(v337, v338);
          pipe_barrier(PIPE_V);
          TADD(v337, v331, v335);
          // pto: %26
          ;
          Tile<TileType::Vec, float, 2, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v339 = Tile<TileType::Vec, float, 2, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v45, v40);
          // pto: %26
          ;
          uint64_t v340 = (uint64_t) v21;
          TASSIGN(v339, v340);
          // pto: %108
          ;
          Tile<TileType::Vec, float, 2, 64, BLayout::RowMajor, 2, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v341;
          // pto: %108
          ;
          uint64_t v342 = (uint64_t) v21;
          TASSIGN(v341, v342);
          // pto: %27
          ;
          Tile<TileType::Vec, float, 2, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v343 = Tile<TileType::Vec, float, 2, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v45, v40);
          // pto: %27
          ;
          uint64_t v344 = (uint64_t) v29;
          TASSIGN(v343, v344);
          // pto: %109
          ;
          Tile<TileType::Vec, float, 2, 64, BLayout::RowMajor, 2, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v345;
          // pto: %109
          ;
          uint64_t v346 = (uint64_t) v29;
          TASSIGN(v345, v346);
          // pto: %weighted2_inline430_inline2146__tile
          ;
          Tile<TileType::Vec, float, 2, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v347 = Tile<TileType::Vec, float, 2, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v45, v40);
          // pto: %weighted2_inline430_inline2146__tile
          ;
          uint64_t v348 = (uint64_t) v21;
          TASSIGN(v347, v348);
          pipe_barrier(PIPE_V);
          TADD(v347, v341, v345);
          // pto: %28
          ;
          Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v349 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v33, v40);
          // pto: %28
          ;
          uint64_t v350 = (uint64_t) v21;
          TASSIGN(v349, v350);
          // pto: %110
          ;
          Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v351;
          // pto: %110
          ;
          uint64_t v352 = (uint64_t) v21;
          TASSIGN(v351, v352);
          // pto: %29
          ;
          Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v353 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v33, v40);
          // pto: %29
          ;
          uint64_t v354 = (uint64_t) v30;
          TASSIGN(v353, v354);
          // pto: %111
          ;
          Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v355;
          // pto: %111
          ;
          uint64_t v356 = (uint64_t) v30;
          TASSIGN(v355, v356);
          // pto: %30
          ;
          Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v357 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v33, v40);
          // pto: %30
          ;
          uint64_t v358 = (uint64_t) v21;
          TASSIGN(v357, v358);
          pipe_barrier(PIPE_V);
          TADD(v357, v351, v355);
          set_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
          // pto: %pooled_kv_inline2148__iter_v6_pview
          ;
          __gm__ float* v359 = PTOAS__GLOBAL_TENSOR_DATA(v53);
          // pto: %pooled_kv_inline2148__iter_v6_pview
          ;
          const int64_t v360 = 0;
          // pto: %pooled_kv_inline2148__iter_v6_pview
          ;
          const int64_t v361 = 128;
          // pto: %pooled_kv_inline2148__iter_v6_pview
          ;
          pto::Shape<1, 1, 1, 1, 64> v362 = pto::Shape<1, 1, 1, 1, 64>();
          // pto: %pooled_kv_inline2148__iter_v6_pview
          ;
          pto::Stride<128, 128, 128, 128, 1> v363 = pto::Stride<128, 128, 128, 128, 1>();
          // pto: %pooled_kv_inline2148__iter_v6_pview
          ;
          GlobalTensor<float, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<128, 128, 128, 128, 1>, pto::Layout::ND> v364 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<128, 128, 128, 128, 1>, pto::Layout::ND>(v359 + (v360 + v129 * v361 + v246), v362, v363);
          wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
          TSTORE(v364, v357);
          set_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
        };
      };
      set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    };
  }
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID3);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID4);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  #endif // __DAV_VEC__

  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}