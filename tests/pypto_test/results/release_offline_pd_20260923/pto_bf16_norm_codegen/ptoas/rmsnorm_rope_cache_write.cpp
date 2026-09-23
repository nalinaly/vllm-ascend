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

AICORE void rmsnorm_rope_cache_write(__gm__ int64_t* v1, __gm__ int32_t* v2, __gm__ float* v3, __gm__ float* v4, __gm__ float* v5, __gm__ float* v6, __gm__ bfloat16_t* v7, __gm__ bfloat16_t* v8, __gm__ float* v9, __gm__ int32_t* v10, int64_t v11, int64_t v12, int64_t v13, int64_t v14, int64_t v15, int32_t v16, int32_t v17) {
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  const int64_t v18 = 256;
  // pto: %c0_i64
  const int64_t v19 = 0;
  // pto: %c4096_i64
  const int64_t v20 = 4096;
  // pto: %c8192_i64
  const int64_t v21 = 8192;
  // pto: %c24576_i64
  const int64_t v22 = 24576;
  const int64_t v23 = 8704;
  const int64_t v24 = 8448;
  // pto: %c40960_i64
  const int64_t v25 = 40960;
  // pto: %c45056_i64
  const int64_t v26 = 45056;
  // pto: %c49152_i64
  const int64_t v27 = 49152;
  // pto: %c49408_i64
  const int64_t v28 = 49408;
  // pto: %c1_index
  const int64_t v29 = 1;
  // pto: %c64_index
  const int64_t v30 = 64;
  // pto: %c384_index
  const int64_t v31 = 384;
  // pto: %c512_index
  const int64_t v32 = 512;
  // pto: %c2_index
  const int64_t v33 = 2;
  // pto: %c16_index
  const int64_t v34 = 16;
  // pto: %cst_16
  const float v35 = 1.0f;
  // pto: %cst_17
  const float v36 = 0.0f;
  // pto: %c0_index
  const int64_t v37 = 0;
  // pto: %c1_i64
  const int64_t v38 = 1;
  // pto: %c4_i64
  const int64_t v39 = 4;
  // pto: %c6_index
  const int64_t v40 = 6;
  // pto: %c256_index
  const int64_t v41 = 256;
  // pto: %c128_index
  const int64_t v42 = 128;
  // pto: %cst_24
  const float v43 = 0.001953125f;
  // pto: %cst_25
  const float v44 = 9.99999997E-7f;
  // pto: %c448_index
  const int64_t v45 = 448;
  // pto: %c192_index
  const int64_t v46 = 192;
  // pto: %c0_i32
  const int32_t v47 = 0;
  // pto: %cst_29
  const float v48 = 0.5f;
  // pto: %cst_30
  const float v49 = 2.0f;
  // pto: %c32_index
  const int64_t v50 = 32;
  // pto: %cmp_freqs_cos__ssa_v0_view
  const int64_t v51 = 1;
  // pto: %cmp_freqs_cos__ssa_v0_view
  const int64_t v52 = 1;
  // pto: %cmp_freqs_cos__ssa_v0_view
  const int64_t v53 = 1;
  // pto: %cmp_freqs_cos__ssa_v0_view
  int64_t v54 = (int64_t) v14;
  // pto: %cmp_freqs_cos__ssa_v0_view
  int64_t v55 = v54 * v30;
  // pto: %cmp_freqs_cos__ssa_v0_view
  int64_t v56 = v53 * v55;
  // pto: %cmp_freqs_cos__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v57 = pto::Shape<1, 1, 1, -1, -1>(v51, v52, v53, v54, v30);
  // pto: %cmp_freqs_cos__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v58 = pto::Stride<-1, -1, -1, -1, -1>(v52 * v56, v56, v55, v30, v29);
  // pto: %cmp_freqs_cos__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v59 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v3, v57, v58);
  // pto: %cmp_freqs_sin__ssa_v0_view
  const int64_t v60 = 1;
  // pto: %cmp_freqs_sin__ssa_v0_view
  const int64_t v61 = 1;
  // pto: %cmp_freqs_sin__ssa_v0_view
  const int64_t v62 = 1;
  // pto: %cmp_freqs_sin__ssa_v0_view
  int64_t v63 = (int64_t) v14;
  // pto: %cmp_freqs_sin__ssa_v0_view
  int64_t v64 = v63 * v30;
  // pto: %cmp_freqs_sin__ssa_v0_view
  int64_t v65 = v62 * v64;
  // pto: %cmp_freqs_sin__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v66 = pto::Shape<1, 1, 1, -1, -1>(v60, v61, v62, v63, v30);
  // pto: %cmp_freqs_sin__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v67 = pto::Stride<-1, -1, -1, -1, -1>(v61 * v65, v65, v64, v30, v29);
  // pto: %cmp_freqs_sin__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v68 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v4, v66, v67);
  // pto: %pooled_kv_inline207__rv_v2_view
  const int64_t v69 = 1;
  // pto: %pooled_kv_inline207__rv_v2_view
  const int64_t v70 = 1;
  // pto: %pooled_kv_inline207__rv_v2_view
  const int64_t v71 = 1;
  // pto: %pooled_kv_inline207__rv_v2_view
  int64_t v72 = v31 * v32;
  // pto: %pooled_kv_inline207__rv_v2_view
  int64_t v73 = v71 * v72;
  // pto: %pooled_kv_inline207__rv_v2_view
  pto::Shape<1, 1, 1, -1, -1> v74 = pto::Shape<1, 1, 1, -1, -1>(v69, v70, v71, v31, v32);
  // pto: %pooled_kv_inline207__rv_v2_view
  pto::Stride<-1, -1, -1, -1, -1> v75 = pto::Stride<-1, -1, -1, -1, -1>(v70 * v73, v73, v72, v32, v29);
  // pto: %pooled_kv_inline207__rv_v2_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v76 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v5, v74, v75);
  // pto: %normed_kv_inline2045__ssa_v0_view
  const int64_t v77 = 1;
  // pto: %normed_kv_inline2045__ssa_v0_view
  const int64_t v78 = 1;
  // pto: %normed_kv_inline2045__ssa_v0_view
  const int64_t v79 = 1;
  // pto: %normed_kv_inline2045__ssa_v0_view
  int64_t v80 = v31 * v32;
  // pto: %normed_kv_inline2045__ssa_v0_view
  int64_t v81 = v79 * v80;
  // pto: %normed_kv_inline2045__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v82 = pto::Shape<1, 1, 1, -1, -1>(v77, v78, v79, v31, v32);
  // pto: %normed_kv_inline2045__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v83 = pto::Stride<-1, -1, -1, -1, -1>(v78 * v81, v81, v80, v32, v29);
  // pto: %normed_kv_inline2045__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v84 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v6, v82, v83);
  // pto: %norm_w_2d_inline2058__ssa_v0_view
  const int64_t v85 = 1;
  // pto: %norm_w_2d_inline2058__ssa_v0_view
  const int64_t v86 = 1;
  // pto: %norm_w_2d_inline2058__ssa_v0_view
  const int64_t v87 = 1;
  // pto: %norm_w_2d_inline2058__ssa_v0_view
  int64_t v88 = v29 * v32;
  // pto: %norm_w_2d_inline2058__ssa_v0_view
  int64_t v89 = v87 * v88;
  // pto: %norm_w_2d_inline2058__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v90 = pto::Shape<1, 1, 1, -1, -1>(v85, v86, v87, v29, v32);
  // pto: %norm_w_2d_inline2058__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v91 = pto::Stride<-1, -1, -1, -1, -1>(v86 * v89, v89, v88, v32, v29);
  // pto: %norm_w_2d_inline2058__ssa_v0_view
  GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v92 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v7, v90, v91);
  // pto: %84
  int64_t v93 = (int64_t) ((uint64_t) v15 * (uint64_t) v50);
  // pto: %cmp_kv_cache_flat_inline2090__ssa_v0_view
  const int64_t v94 = 1;
  // pto: %cmp_kv_cache_flat_inline2090__ssa_v0_view
  const int64_t v95 = 1;
  // pto: %cmp_kv_cache_flat_inline2090__ssa_v0_view
  const int64_t v96 = 1;
  // pto: %cmp_kv_cache_flat_inline2090__ssa_v0_view
  int64_t v97 = v93 * v32;
  // pto: %cmp_kv_cache_flat_inline2090__ssa_v0_view
  int64_t v98 = v96 * v97;
  // pto: %cmp_kv_cache_flat_inline2090__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v99 = pto::Shape<1, 1, 1, -1, -1>(v94, v95, v96, v93, v32);
  // pto: %cmp_kv_cache_flat_inline2090__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v100 = pto::Stride<-1, -1, -1, -1, -1>(v95 * v98, v98, v97, v32, v29);
  // pto: %cmp_kv_cache_flat_inline2090__ssa_v0_view
  GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v101 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v8, v99, v100);
  // pto: %kv_flat_inline2068__ssa_v0_view
  const int64_t v102 = 1;
  // pto: %kv_flat_inline2068__ssa_v0_view
  const int64_t v103 = 1;
  // pto: %kv_flat_inline2068__ssa_v0_view
  const int64_t v104 = 1;
  // pto: %kv_flat_inline2068__ssa_v0_view
  int64_t v105 = (int64_t) v12;
  // pto: %kv_flat_inline2068__ssa_v0_view
  int64_t v106 = v105 * v32;
  // pto: %kv_flat_inline2068__ssa_v0_view
  int64_t v107 = v104 * v106;
  // pto: %kv_flat_inline2068__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v108 = pto::Shape<1, 1, 1, -1, -1>(v102, v103, v104, v105, v32);
  // pto: %kv_flat_inline2068__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v109 = pto::Stride<-1, -1, -1, -1, -1>(v103 * v107, v107, v106, v32, v29);
  // pto: %kv_flat_inline2068__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v110 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v9, v108, v109);
  // pto: %cosine_inline374_inline2041__phi_v4
  Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v111 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v30);
  // pto: %cosine_inline374_inline2041__phi_v4
  uint64_t v112 = (uint64_t) v19;
  TASSIGN(v111, v112);
  // pto: %sine_inline373_inline2039__phi_v4
  Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v113 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v30);
  // pto: %sine_inline373_inline2039__phi_v4
  uint64_t v114 = (uint64_t) v20;
  TASSIGN(v113, v114);
  // pto: %rms_blk_inline2043__ssa_v0, %23
  int64_t v115 = (int64_t) ((uint64_t) ((int64_t) v16) * (uint64_t) v34);
  // pto: %24
  int64_t v116 = (int64_t) ((uint64_t) v11 - (uint64_t) v115);
  // pto: %25
  int64_t v117 = v116 < v34 ? v116 : v34;
  // pto: %cosine_inline374_inline2041__tile
  Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v118 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v30);
  // pto: %cosine_inline374_inline2041__tile
  uint64_t v119 = (uint64_t) v19;
  TASSIGN(v118, v119);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
  TEXPANDS(v118, v35);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  // pto: %sine_inline373_inline2039__tile
  Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v120 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v30);
  // pto: %sine_inline373_inline2039__tile
  uint64_t v121 = (uint64_t) v20;
  TASSIGN(v120, v121);
  TEXPANDS(v120, v36);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  for (int64_t v122 = v37; v122 < v117; v122 += v29) {
    // pto: %26
    ;
    int64_t v123 = (int64_t) ((uint64_t) v115 + (uint64_t) v122);
    // pto: %position_inline370_inline2077__tile
    ;
    int64_t v124 = (v1)[v123];
    // pto: %27
    ;
    int64_t v125 = (int64_t) ((uint64_t) v124 + (uint64_t) v38);
    // pto: %28, %29
    ;
    if (v125 % v39 == v19) {
      // pto: %31, %30
      ;
      int32_t v126 = (v2)[v123 / v40];
      // pto: %32, %36, %34
      ;
      int64_t v127 = (int64_t) ((uint64_t) ((int64_t) v126) + (uint64_t) (v125 / v39));
      // pto: %gather_row_view
      ;
      int64_t v128 = (int64_t) ((uint64_t) v122 * (uint64_t) v18);
      // pto: %gather_row_view
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v129;
      // pto: %gather_row_view
      ;
      uint64_t v130 = (uint64_t) v128;
      TASSIGN(v129, v130);
      // pto: %37
      ;
      int64_t v131 = v127 < v37 ? v37 : v127;
      // pto: %cmp_freqs_cos__ssa_v0_pview
      ;
      __gm__ float* v132 = PTOAS__GLOBAL_TENSOR_DATA(v59);
      // pto: %cmp_freqs_cos__ssa_v0_pview
      ;
      const int64_t v133 = 0;
      // pto: %cmp_freqs_cos__ssa_v0_pview
      ;
      const int64_t v134 = 64;
      // pto: %cmp_freqs_cos__ssa_v0_pview
      ;
      pto::Shape<1, 1, 1, 1, 64> v135 = pto::Shape<1, 1, 1, 1, 64>();
      // pto: %cmp_freqs_cos__ssa_v0_pview
      ;
      pto::Stride<64, 64, 64, 64, 1> v136 = pto::Stride<64, 64, 64, 64, 1>();
      // pto: %cmp_freqs_cos__ssa_v0_pview
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<64, 64, 64, 64, 1>, pto::Layout::ND> v137 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<64, 64, 64, 64, 1>, pto::Layout::ND>(v132 + (v133 + v131 * v134), v135, v136);
      TLOAD(v129, v137);
      // pto: %38
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v138;
      // pto: %38
      ;
      uint64_t v139 = (uint64_t) ((int64_t) (uint64_t) v128 + (uint64_t) v20);
      TASSIGN(v138, v139);
      // pto: %cmp_freqs_sin__ssa_v0_pview
      ;
      __gm__ float* v140 = PTOAS__GLOBAL_TENSOR_DATA(v68);
      // pto: %cmp_freqs_sin__ssa_v0_pview
      ;
      const int64_t v141 = 0;
      // pto: %cmp_freqs_sin__ssa_v0_pview
      ;
      const int64_t v142 = 64;
      // pto: %cmp_freqs_sin__ssa_v0_pview
      ;
      pto::Shape<1, 1, 1, 1, 64> v143 = pto::Shape<1, 1, 1, 1, 64>();
      // pto: %cmp_freqs_sin__ssa_v0_pview
      ;
      pto::Stride<64, 64, 64, 64, 1> v144 = pto::Stride<64, 64, 64, 64, 1>();
      // pto: %cmp_freqs_sin__ssa_v0_pview
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<64, 64, 64, 64, 1>, pto::Layout::ND> v145 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<64, 64, 64, 64, 1>, pto::Layout::ND>(v140 + (v141 + v131 * v142), v143, v144);
      TLOAD(v138, v145);
    };
  }
  // pto: %cos_b_inline2078__ssa_v0
  Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v146 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v30);
  // pto: %cos_b_inline2078__ssa_v0
  uint64_t v147 = (uint64_t) v19;
  TASSIGN(v146, v147);
  // pto: %sin_b_inline2079__ssa_v0
  Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v148 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v30);
  // pto: %sin_b_inline2079__ssa_v0
  uint64_t v149 = (uint64_t) v20;
  TASSIGN(v148, v149);
  // pto: %rms_low_inline2067__tile
  Tile<TileType::Vec, float, 16, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v150 = Tile<TileType::Vec, float, 16, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v41);
  // pto: %rms_low_inline2067__tile
  uint64_t v151 = (uint64_t) v21;
  TASSIGN(v150, v151);
  // pto: %40
  int64_t v152 = v115 < v37 ? v37 : v115;
  // pto: %pooled_kv_inline207__rv_v2_pview
  __gm__ float* v153 = PTOAS__GLOBAL_TENSOR_DATA(v76);
  // pto: %pooled_kv_inline207__rv_v2_pview
  const int64_t v154 = 0;
  // pto: %pooled_kv_inline207__rv_v2_pview
  const int64_t v155 = 512;
  // pto: %pooled_kv_inline207__rv_v2_pview
  pto::Shape<1, 1, 1, 16, 256> v156 = pto::Shape<1, 1, 1, 16, 256>();
  // pto: %pooled_kv_inline207__rv_v2_pview
  pto::Stride<8192, 8192, 8192, 512, 1> v157 = pto::Stride<8192, 8192, 8192, 512, 1>();
  // pto: %pooled_kv_inline207__rv_v2_pview
  GlobalTensor<float, pto::Shape<1, 1, 1, 16, 256>, pto::Stride<8192, 8192, 8192, 512, 1>, pto::Layout::ND> v158 = GlobalTensor<float, pto::Shape<1, 1, 1, 16, 256>, pto::Stride<8192, 8192, 8192, 512, 1>, pto::Layout::ND>(v153 + (v154 + v152 * v155), v156, v157);
  TLOAD(v150, v158);
  set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  // pto: %rms_high_inline2081__tile
  Tile<TileType::Vec, float, 16, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v159 = Tile<TileType::Vec, float, 16, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v41);
  // pto: %rms_high_inline2081__tile
  uint64_t v160 = (uint64_t) v22;
  TASSIGN(v159, v160);
  // pto: %42
  __gm__ float* v161 = PTOAS__GLOBAL_TENSOR_DATA(v76);
  // pto: %42
  const int64_t v162 = 256;
  // pto: %42
  const int64_t v163 = 512;
  // pto: %42
  pto::Shape<1, 1, 1, 16, 256> v164 = pto::Shape<1, 1, 1, 16, 256>();
  // pto: %42
  pto::Stride<8192, 8192, 8192, 512, 1> v165 = pto::Stride<8192, 8192, 8192, 512, 1>();
  // pto: %42
  GlobalTensor<float, pto::Shape<1, 1, 1, 16, 256>, pto::Stride<8192, 8192, 8192, 512, 1>, pto::Layout::ND> v166 = GlobalTensor<float, pto::Shape<1, 1, 1, 16, 256>, pto::Stride<8192, 8192, 8192, 512, 1>, pto::Layout::ND>(v161 + (v162 + v152 * v163), v164, v165);
  TLOAD(v159, v166);
  set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
  // pto: %t__tile
  Tile<TileType::Vec, float, 16, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v167 = Tile<TileType::Vec, float, 16, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v41);
  // pto: %t__tile
  uint64_t v168 = (uint64_t) v21;
  TASSIGN(v167, v168);
  wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  TMUL(v167, v150, v150);
  // pto: %2
  Tile<TileType::Vec, float, 16, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v169 = Tile<TileType::Vec, float, 16, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v41);
  // pto: %2
  uint64_t v170 = (uint64_t) v22;
  TASSIGN(v169, v170);
  wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
  TMUL(v169, v159, v159);
  // pto: %folded4_inline2084__tile
  Tile<TileType::Vec, float, 16, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v171 = Tile<TileType::Vec, float, 16, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v41);
  // pto: %folded4_inline2084__tile
  uint64_t v172 = (uint64_t) v21;
  TASSIGN(v171, v172);
  pipe_barrier(PIPE_V);
  TADD(v171, v167, v169);
  // pto: %3
  Tile<TileType::Vec, float, 16, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v173 = Tile<TileType::Vec, float, 16, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v42);
  // pto: %3
  uint64_t v174 = (uint64_t) v21;
  TASSIGN(v173, v174);
  // pto: %slice_view
  Tile<TileType::Vec, float, 16, 256, BLayout::RowMajor, 16, 128, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v175;
  // pto: %slice_view
  uint64_t v176 = (uint64_t) v21;
  TASSIGN(v175, v176);
  // pto: %4
  Tile<TileType::Vec, float, 16, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v177 = Tile<TileType::Vec, float, 16, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v42);
  // pto: %4
  uint64_t v178 = (uint64_t) v23;
  TASSIGN(v177, v178);
  // pto: %43
  Tile<TileType::Vec, float, 16, 256, BLayout::RowMajor, 16, 128, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v179;
  // pto: %43
  uint64_t v180 = (uint64_t) v23;
  TASSIGN(v179, v180);
  // pto: %folded2_inline2083__tile
  Tile<TileType::Vec, float, 16, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v181 = Tile<TileType::Vec, float, 16, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v42);
  // pto: %folded2_inline2083__tile
  uint64_t v182 = (uint64_t) v21;
  TASSIGN(v181, v182);
  pipe_barrier(PIPE_V);
  TADD(v181, v175, v179);
  // pto: %5
  Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v183 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v30);
  // pto: %5
  uint64_t v184 = (uint64_t) v21;
  TASSIGN(v183, v184);
  // pto: %44
  Tile<TileType::Vec, float, 16, 128, BLayout::RowMajor, 16, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v185;
  // pto: %44
  uint64_t v186 = (uint64_t) v21;
  TASSIGN(v185, v186);
  // pto: %6
  Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v187 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v30);
  // pto: %6
  uint64_t v188 = (uint64_t) v24;
  TASSIGN(v187, v188);
  // pto: %45
  Tile<TileType::Vec, float, 16, 128, BLayout::RowMajor, 16, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v189;
  // pto: %45
  uint64_t v190 = (uint64_t) v24;
  TASSIGN(v189, v190);
  // pto: %folded1_inline2071__tile
  Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v191 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v30);
  // pto: %folded1_inline2071__tile
  uint64_t v192 = (uint64_t) v22;
  TASSIGN(v191, v192);
  pipe_barrier(PIPE_V);
  TADD(v191, v185, v189);
  // pto: %tmp_tile
  Tile<TileType::Vec, float, 16, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v193 = Tile<TileType::Vec, float, 16, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v42);
  // pto: %tmp_tile
  uint64_t v194 = (uint64_t) v21;
  TASSIGN(v193, v194);
  // pto: %square_sum_inline2085__tile
  Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v195 = Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v29);
  // pto: %square_sum_inline2085__tile
  uint64_t v196 = (uint64_t) v25;
  TASSIGN(v195, v196);
  pipe_barrier(PIPE_V);
  TROWSUM(v195, v191, v193);
  // pto: %t__rm_a0_tmp_v0
  Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v197 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v29, v34);
  // pto: %t__rm_a0_tmp_v0
  uint64_t v198 = (uint64_t) v25;
  TASSIGN(v197, v198);
  // pto: %t__row_major_tmp_v1
  Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v199 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v29, v34);
  // pto: %t__row_major_tmp_v1
  uint64_t v200 = (uint64_t) v21;
  TASSIGN(v199, v200);
  pipe_barrier(PIPE_V);
  TMULS(v199, v197, v43);
  // pto: %7
  Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v201 = Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v29);
  // pto: %7
  uint64_t v202 = (uint64_t) v21;
  TASSIGN(v201, v202);
  // pto: %variance_inline2057__rm_a0_tmp_v2
  Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v203 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v29, v34);
  // pto: %variance_inline2057__rm_a0_tmp_v2
  uint64_t v204 = (uint64_t) v21;
  TASSIGN(v203, v204);
  // pto: %variance_inline2057__row_major_tmp_v3
  Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v205 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v29, v34);
  // pto: %variance_inline2057__row_major_tmp_v3
  uint64_t v206 = (uint64_t) v21;
  TASSIGN(v205, v206);
  pipe_barrier(PIPE_V);
  TADDS(v205, v203, v44);
  // pto: %variance_inline2057__tile
  Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v207 = Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v29);
  // pto: %variance_inline2057__tile
  uint64_t v208 = (uint64_t) v21;
  TASSIGN(v207, v208);
  // pto: %rms_inline2080__rm_a0_tmp_v4
  Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v209 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v29, v34);
  // pto: %rms_inline2080__rm_a0_tmp_v4
  uint64_t v210 = (uint64_t) v21;
  TASSIGN(v209, v210);
  // pto: %rms_inline2080__row_major_tmp_v5
  Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v211 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v29, v34);
  // pto: %rms_inline2080__row_major_tmp_v5
  uint64_t v212 = (uint64_t) v26;
  TASSIGN(v211, v212);
  pipe_barrier(PIPE_V);
  TSQRT(v211, v209);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
  // pto: %rms_inline2080__tile
  Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v213 = Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v29);
  // pto: %rms_inline2080__tile
  uint64_t v214 = (uint64_t) v26;
  TASSIGN(v213, v214);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
  for (int64_t v215 = v37; v215 < v45; v215 += v30) {
    // pto: %kv_norm_chunk_inline2087__tile
    ;
    Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v216 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v30);
    // pto: %kv_norm_chunk_inline2087__tile
    ;
    uint64_t v217 = (uint64_t) v21;
    TASSIGN(v216, v217);
    // pto: %47
    ;
    int64_t v218 = v215 < v37 ? v37 : v215;
    // pto: %48
    ;
    __gm__ float* v219 = PTOAS__GLOBAL_TENSOR_DATA(v76);
    // pto: %48
    ;
    const int64_t v220 = 0;
    // pto: %48
    ;
    const int64_t v221 = 512;
    // pto: %48
    ;
    pto::Shape<1, 1, 1, 16, 64> v222 = pto::Shape<1, 1, 1, 16, 64>();
    // pto: %48
    ;
    pto::Stride<8192, 8192, 8192, 512, 1> v223 = pto::Stride<8192, 8192, 8192, 512, 1>();
    // pto: %48
    ;
    GlobalTensor<float, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<8192, 8192, 8192, 512, 1>, pto::Layout::ND> v224 = GlobalTensor<float, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<8192, 8192, 8192, 512, 1>, pto::Layout::ND>(v219 + (v220 + v152 * v221 + v218), v222, v223);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    TLOAD(v216, v224);
    // pto: %8
    ;
    Tile<TileType::Vec, bfloat16_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v225 = Tile<TileType::Vec, bfloat16_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v29, v30);
    // pto: %8
    ;
    uint64_t v226 = (uint64_t) v25;
    TASSIGN(v225, v226);
    // pto: %norm_w_2d_inline2058__ssa_v0_pview
    ;
    __gm__ bfloat16_t* v227 = PTOAS__GLOBAL_TENSOR_DATA(v92);
    // pto: %norm_w_2d_inline2058__ssa_v0_pview
    ;
    const int64_t v228 = 0;
    // pto: %norm_w_2d_inline2058__ssa_v0_pview
    ;
    pto::Shape<1, 1, 1, 1, 64> v229 = pto::Shape<1, 1, 1, 1, 64>();
    // pto: %norm_w_2d_inline2058__ssa_v0_pview
    ;
    pto::Stride<512, 512, 512, 512, 1> v230 = pto::Stride<512, 512, 512, 512, 1>();
    // pto: %norm_w_2d_inline2058__ssa_v0_pview
    ;
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<512, 512, 512, 512, 1>, pto::Layout::ND> v231 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<512, 512, 512, 512, 1>, pto::Layout::ND>(v227 + (v228 + v218), v229, v230);
    TLOAD(v225, v231);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
    // pto: %gamma_inline2048__tile
    ;
    Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v232 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v29, v30);
    // pto: %gamma_inline2048__tile
    ;
    uint64_t v233 = (uint64_t) v22;
    TASSIGN(v232, v233);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
    RoundMode v234 = RoundMode::CAST_ROUND;
    SaturationMode v235 = SaturationMode::OFF;
    TCVT(v232, v225, v234, v235);
    // pto: %9
    ;
    Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v236 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v30);
    // pto: %9
    ;
    uint64_t v237 = (uint64_t) v21;
    TASSIGN(v236, v237);
    TROWEXPANDDIV(v236, v216, v213);
    // pto: %normed_chunk_inline2044__tile
    ;
    Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v238 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v30);
    // pto: %normed_chunk_inline2044__tile
    ;
    uint64_t v239 = (uint64_t) v21;
    TASSIGN(v238, v239);
    pipe_barrier(PIPE_V);
    TCOLEXPANDMUL(v238, v236, v232);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    // pto: %normed_kv_inline2045__iter_v1_pview
    ;
    __gm__ float* v240 = PTOAS__GLOBAL_TENSOR_DATA(v84);
    // pto: %normed_kv_inline2045__iter_v1_pview
    ;
    const int64_t v241 = 0;
    // pto: %normed_kv_inline2045__iter_v1_pview
    ;
    const int64_t v242 = 512;
    // pto: %normed_kv_inline2045__iter_v1_pview
    ;
    pto::Shape<1, 1, 1, 16, 64> v243 = pto::Shape<1, 1, 1, 16, 64>();
    // pto: %normed_kv_inline2045__iter_v1_pview
    ;
    pto::Stride<8192, 8192, 8192, 512, 1> v244 = pto::Stride<8192, 8192, 8192, 512, 1>();
    // pto: %normed_kv_inline2045__iter_v1_pview
    ;
    GlobalTensor<float, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<8192, 8192, 8192, 512, 1>, pto::Layout::ND> v245 = GlobalTensor<float, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<8192, 8192, 8192, 512, 1>, pto::Layout::ND>(v240 + (v241 + v152 * v242 + v218), v243, v244);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(v245, v238);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  }
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
  // pto: %kv_rope_norm_inline2073__tile
  Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v246 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v30);
  // pto: %kv_rope_norm_inline2073__tile
  uint64_t v247 = (uint64_t) v21;
  TASSIGN(v246, v247);
  // pto: %53
  __gm__ float* v248 = PTOAS__GLOBAL_TENSOR_DATA(v76);
  // pto: %53
  const int64_t v249 = 448;
  // pto: %53
  const int64_t v250 = 512;
  // pto: %53
  pto::Shape<1, 1, 1, 16, 64> v251 = pto::Shape<1, 1, 1, 16, 64>();
  // pto: %53
  pto::Stride<8192, 8192, 8192, 512, 1> v252 = pto::Stride<8192, 8192, 8192, 512, 1>();
  // pto: %53
  GlobalTensor<float, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<8192, 8192, 8192, 512, 1>, pto::Layout::ND> v253 = GlobalTensor<float, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<8192, 8192, 8192, 512, 1>, pto::Layout::ND>(v248 + (v249 + v152 * v250), v251, v252);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
  TLOAD(v246, v253);
  // pto: %10
  Tile<TileType::Vec, bfloat16_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v254 = Tile<TileType::Vec, bfloat16_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v29, v30);
  // pto: %10
  uint64_t v255 = (uint64_t) v25;
  TASSIGN(v254, v255);
  // pto: %54
  __gm__ bfloat16_t* v256 = PTOAS__GLOBAL_TENSOR_DATA(v92);
  // pto: %54
  unsigned v257 = 448;
  // pto: %54
  pto::Shape<1, 1, 1, 1, 64> v258 = pto::Shape<1, 1, 1, 1, 64>();
  // pto: %54
  pto::Stride<512, 512, 512, 512, 1> v259 = pto::Stride<512, 512, 512, 512, 1>();
  // pto: %54
  GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<512, 512, 512, 512, 1>, pto::Layout::ND> v260 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<512, 512, 512, 512, 1>, pto::Layout::ND>(v256 + v257, v258, v259);
  TLOAD(v254, v260);
  set_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
  // pto: %gamma_rope_inline2091__tile
  Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v261 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v29, v30);
  // pto: %gamma_rope_inline2091__tile
  uint64_t v262 = (uint64_t) v22;
  TASSIGN(v261, v262);
  wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
  RoundMode v263 = RoundMode::CAST_ROUND;
  SaturationMode v264 = SaturationMode::OFF;
  TCVT(v261, v254, v263, v264);
  // pto: %11
  Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v265 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v30);
  // pto: %11
  uint64_t v266 = (uint64_t) v21;
  TASSIGN(v265, v266);
  TROWEXPANDDIV(v265, v246, v213);
  set_flag(PIPE_V, PIPE_S, EVENT_ID0);
  // pto: %rope_normed_inline2055__tile
  Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v267 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v30);
  // pto: %rope_normed_inline2055__tile
  uint64_t v268 = (uint64_t) v21;
  TASSIGN(v267, v268);
  pipe_barrier(PIPE_V);
  TCOLEXPANDMUL(v267, v265, v261);
  // pto: %rope_ones_inline2092__tile
  Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v269 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v30);
  // pto: %rope_ones_inline2092__tile
  uint64_t v270 = (uint64_t) v22;
  TASSIGN(v269, v270);
  pipe_barrier(PIPE_V);
  TEXPANDS(v269, v35);
  // pto: %rope_index_inline2093__ci_tmp_v0
  Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v271 = Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v29, v46);
  // pto: %rope_index_inline2093__ci_tmp_v0
  uint64_t v272 = (uint64_t) v25;
  TASSIGN(v271, v272);
  // pto: %rope_index_inline2093__tile
  Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v273 = Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v29, v30);
  // pto: %rope_index_inline2093__tile
  uint64_t v274 = (uint64_t) v26;
  TASSIGN(v273, v274);
  // pto: %ci_tmp_view
  Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, 1, 192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v275;
  TRESHAPE(v275, v271);
  // pto: %ci_dst_view
  Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v276;
  TRESHAPE(v276, v273);
  wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
  TCI<Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, 1, 192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, int32_t, 0>(v276, v47, v275);
  set_flag(PIPE_S, PIPE_V, EVENT_ID0);
  // pto: %rope_index_f_inline2066__tile
  Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v277 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v29, v30);
  // pto: %rope_index_f_inline2066__tile
  uint64_t v278 = (uint64_t) v25;
  TASSIGN(v277, v278);
  wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
  RoundMode v279 = RoundMode::CAST_ROUND;
  SaturationMode v280 = SaturationMode::OFF;
  TCVT(v277, v273, v279, v280);
  // pto: %rope_col_inline2094__tile
  Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v281 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v30);
  // pto: %rope_col_inline2094__tile
  uint64_t v282 = (uint64_t) v22;
  TASSIGN(v281, v282);
  pipe_barrier(PIPE_V);
  TCOLEXPANDMUL(v281, v269, v277);
  // pto: %12
  Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v283 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v30);
  // pto: %12
  uint64_t v284 = (uint64_t) v25;
  TASSIGN(v283, v284);
  pipe_barrier(PIPE_V);
  TMULS(v283, v281, v48);
  // pto: %13
  Tile<TileType::Vec, int32_t, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v285 = Tile<TileType::Vec, int32_t, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v30);
  // pto: %13
  uint64_t v286 = (uint64_t) v25;
  TASSIGN(v285, v286);
  pipe_barrier(PIPE_V);
  RoundMode v287 = RoundMode::CAST_TRUNC;
  SaturationMode v288 = SaturationMode::ON;
  TCVT(v285, v283, v287, v288);
  // pto: %rope_dup_f_inline2096__tile
  Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v289 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v30);
  // pto: %rope_dup_f_inline2096__tile
  uint64_t v290 = (uint64_t) v25;
  TASSIGN(v289, v290);
  pipe_barrier(PIPE_V);
  RoundMode v291 = RoundMode::CAST_ROUND;
  SaturationMode v292 = SaturationMode::OFF;
  TCVT(v289, v285, v291, v292);
  // pto: %14
  Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v293 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v30);
  // pto: %14
  uint64_t v294 = (uint64_t) v25;
  TASSIGN(v293, v294);
  pipe_barrier(PIPE_V);
  TMULS(v293, v289, v49);
  // pto: %rope_lane_inline2098__tile
  Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v295 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v30);
  // pto: %rope_lane_inline2098__tile
  uint64_t v296 = (uint64_t) v25;
  TASSIGN(v295, v296);
  pipe_barrier(PIPE_V);
  TSUB(v295, v281, v293);
  // pto: %15
  Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v297 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v30);
  // pto: %15
  uint64_t v298 = (uint64_t) v22;
  TASSIGN(v297, v298);
  pipe_barrier(PIPE_V);
  TADDS(v297, v281, v35);
  // pto: %16
  Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v299 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v30);
  // pto: %16
  uint64_t v300 = (uint64_t) v26;
  TASSIGN(v299, v300);
  TMULS(v299, v295, v49);
  // pto: %17
  Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v301 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v30);
  // pto: %17
  uint64_t v302 = (uint64_t) v22;
  TASSIGN(v301, v302);
  pipe_barrier(PIPE_V);
  TSUB(v301, v297, v299);
  // pto: %rope_swap_idx_inline2053__tile
  Tile<TileType::Vec, int32_t, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v303 = Tile<TileType::Vec, int32_t, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v30);
  // pto: %rope_swap_idx_inline2053__tile
  uint64_t v304 = (uint64_t) v22;
  TASSIGN(v303, v304);
  pipe_barrier(PIPE_V);
  RoundMode v305 = RoundMode::CAST_ROUND;
  SaturationMode v306 = SaturationMode::ON;
  TCVT(v303, v301, v305, v306);
  // pto: %gather_acc_init
  Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v307 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v30);
  // pto: %gather_acc_init
  uint64_t v308 = (uint64_t) v26;
  TASSIGN(v307, v308);
  for (int64_t v309 = v37; v309 < v34; v309 += v29) {
    // pto: %gather_inp_row
    ;
    Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v310 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v29, v30);
    // pto: %gather_inp_row
    ;
    uint64_t v311 = (uint64_t) v21;
    TASSIGN(v310, v311);
    // pto: %55
    ;
    int64_t v312 = (int64_t) ((uint64_t) v309 * (uint64_t) v18);
    // pto: %55
    ;
    Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v313;
    // pto: %55
    ;
    uint64_t v314 = (uint64_t) ((int64_t) (uint64_t) v312 + (uint64_t) v21);
    TASSIGN(v313, v314);
    // pto: %gather_idx_row
    ;
    Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v315 = Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v29, v30);
    // pto: %gather_idx_row
    ;
    uint64_t v316 = (uint64_t) v22;
    TASSIGN(v315, v316);
    // pto: %56
    ;
    Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v317;
    // pto: %56
    ;
    uint64_t v318 = (uint64_t) ((int64_t) (uint64_t) v312 + (uint64_t) v22);
    TASSIGN(v317, v318);
    // pto: %gather_row_tmp
    ;
    Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v319 = Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v29, v30);
    // pto: %gather_row_tmp
    ;
    uint64_t v320 = (uint64_t) v27;
    TASSIGN(v319, v320);
    // pto: %gather_row
    ;
    Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v321 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v29, v30);
    // pto: %gather_row
    ;
    uint64_t v322 = (uint64_t) v28;
    TASSIGN(v321, v322);
    pipe_barrier(PIPE_V);
    TGATHER(v321, v313, v317, v319);
    // pto: %assemble_view
    ;
    Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v323;
    // pto: %assemble_view
    ;
    uint64_t v324 = (uint64_t) ((int64_t) (uint64_t) v312 + (uint64_t) v26);
    TASSIGN(v323, v324);
    pipe_barrier(PIPE_V);
    TMOV(v323, v321);
  }
  // pto: %swapped_inline2059__tile
  Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v325 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v30);
  // pto: %swapped_inline2059__tile
  uint64_t v326 = (uint64_t) v26;
  TASSIGN(v325, v326);
  // pto: %18
  Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v327 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v30);
  // pto: %18
  uint64_t v328 = (uint64_t) v22;
  TASSIGN(v327, v328);
  pipe_barrier(PIPE_V);
  TMULS(v327, v295, v49);
  // pto: %19
  Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v329 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v30);
  // pto: %19
  uint64_t v330 = (uint64_t) v22;
  TASSIGN(v329, v330);
  pipe_barrier(PIPE_V);
  TSUBS(v329, v327, v35);
  // pto: %sin_signed_inline2088__tile
  Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v331 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v30);
  // pto: %sin_signed_inline2088__tile
  uint64_t v332 = (uint64_t) v22;
  TASSIGN(v331, v332);
  pipe_barrier(PIPE_V);
  TMUL(v331, v148, v329);
  // pto: %20
  Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v333 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v30);
  // pto: %20
  uint64_t v334 = (uint64_t) v21;
  TASSIGN(v333, v334);
  TMUL(v333, v267, v146);
  // pto: %21
  Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v335 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v30);
  // pto: %21
  uint64_t v336 = (uint64_t) v22;
  TASSIGN(v335, v336);
  pipe_barrier(PIPE_V);
  TMUL(v335, v325, v331);
  // pto: %rope_rot_inline2038__tile
  Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v337 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v30);
  // pto: %rope_rot_inline2038__tile
  uint64_t v338 = (uint64_t) v21;
  TASSIGN(v337, v338);
  pipe_barrier(PIPE_V);
  TADD(v337, v333, v335);
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
  // pto: %normed_kv_inline2045__rv_v2_pview
  __gm__ float* v339 = PTOAS__GLOBAL_TENSOR_DATA(v84);
  // pto: %normed_kv_inline2045__rv_v2_pview
  const int64_t v340 = 448;
  // pto: %normed_kv_inline2045__rv_v2_pview
  const int64_t v341 = 512;
  // pto: %normed_kv_inline2045__rv_v2_pview
  pto::Shape<1, 1, 1, 16, 64> v342 = pto::Shape<1, 1, 1, 16, 64>();
  // pto: %normed_kv_inline2045__rv_v2_pview
  pto::Stride<8192, 8192, 8192, 512, 1> v343 = pto::Stride<8192, 8192, 8192, 512, 1>();
  // pto: %normed_kv_inline2045__rv_v2_pview
  GlobalTensor<float, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<8192, 8192, 8192, 512, 1>, pto::Layout::ND> v344 = GlobalTensor<float, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<8192, 8192, 8192, 512, 1>, pto::Layout::ND>(v339 + (v340 + v152 * v341), v342, v343);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
  TSTORE(v344, v337);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
  for (int64_t v345 = v37; v345 < v117; v345 += v29) {
    // pto: %59
    ;
    int64_t v346 = (int64_t) ((uint64_t) v115 + (uint64_t) v345);
    // pto: %token_pos_inline2082__tile
    ;
    int64_t v347 = (v1)[v346];
    // pto: %60
    ;
    int64_t v348 = (int64_t) ((uint64_t) v347 + (uint64_t) v38);
    // pto: %61, %62
    ;
    if (v348 % v39 == v19) {
      // pto: %64, %63
      ;
      int32_t v349 = (v2)[v346 / v40];
      // pto: %65, %69, %67, %flat_offset_mul
      ;
      int64_t v350 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) ((int64_t) v349) + (uint64_t) (v348 / v39)) * (uint64_t) v33);
      // pto: %cache_page_inline2035__tile
      ;
      int32_t v351 = (v10)[v350];
      // pto: %71, %cache_offset_inline2047__tile
      ;
      int32_t v352 = (v10)[(int64_t) ((uint64_t) v350 + (uint64_t) v29)];
      // pto: %72
      ;
      int64_t v353 = (int64_t) v351;
      // pto: %74
      ;
      int64_t v354 = (int64_t) v352;
      // pto: %73, %75, %76
      ;
      if (v353 >= v37 & v354 >= v37) {
        // pto: %78, %80
        ;
        int64_t v355 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) v353 * (uint64_t) v50) + (uint64_t) v354);
        // pto: %kv_row_fp32_inline2034__tile
        ;
        Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v356 = Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v29, v32);
        // pto: %kv_row_fp32_inline2034__tile
        ;
        uint64_t v357 = (uint64_t) v21;
        TASSIGN(v356, v357);
        // pto: %81
        ;
        int64_t v358 = v346 < v37 ? v37 : v346;
        // pto: %normed_kv_inline2045__tile_pview
        ;
        __gm__ float* v359 = PTOAS__GLOBAL_TENSOR_DATA(v84);
        // pto: %normed_kv_inline2045__tile_pview
        ;
        const int64_t v360 = 0;
        // pto: %normed_kv_inline2045__tile_pview
        ;
        const int64_t v361 = 512;
        // pto: %normed_kv_inline2045__tile_pview
        ;
        pto::Shape<1, 1, 1, 1, 512> v362 = pto::Shape<1, 1, 1, 1, 512>();
        // pto: %normed_kv_inline2045__tile_pview
        ;
        pto::Stride<512, 512, 512, 512, 1> v363 = pto::Stride<512, 512, 512, 512, 1>();
        // pto: %normed_kv_inline2045__tile_pview
        ;
        GlobalTensor<float, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<512, 512, 512, 512, 1>, pto::Layout::ND> v364 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<512, 512, 512, 512, 1>, pto::Layout::ND>(v359 + (v360 + v358 * v361), v362, v363);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
        TLOAD(v356, v364);
        set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        // pto: %kv_flat_inline2068__iter_v1_pview
        ;
        __gm__ float* v365 = PTOAS__GLOBAL_TENSOR_DATA(v110);
        // pto: %kv_flat_inline2068__iter_v1_pview
        ;
        const int64_t v366 = 0;
        // pto: %kv_flat_inline2068__iter_v1_pview
        ;
        const int64_t v367 = 512;
        // pto: %kv_flat_inline2068__iter_v1_pview
        ;
        pto::Shape<1, 1, 1, 1, 512> v368 = pto::Shape<1, 1, 1, 1, 512>();
        // pto: %kv_flat_inline2068__iter_v1_pview
        ;
        pto::Stride<512, 512, 512, 512, 1> v369 = pto::Stride<512, 512, 512, 512, 1>();
        // pto: %kv_flat_inline2068__iter_v1_pview
        ;
        GlobalTensor<float, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<512, 512, 512, 512, 1>, pto::Layout::ND> v370 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<512, 512, 512, 512, 1>, pto::Layout::ND>(v365 + (v366 + v358 * v367), v368, v369);
        wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        TSTORE(v370, v356);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        // pto: %22
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v371 = Tile<TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v29, v32);
        // pto: %22
        ;
        uint64_t v372 = (uint64_t) v21;
        TASSIGN(v371, v372);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        RoundMode v373 = RoundMode::CAST_RINT;
        SaturationMode v374 = SaturationMode::OFF;
        TCVT(v371, v356, v373, v374);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
        // pto: %83
        ;
        int64_t v375 = v355 < v37 ? v37 : v355;
        // pto: %cmp_kv_cache_flat_inline2090__iter_v1_pview
        ;
        __gm__ bfloat16_t* v376 = PTOAS__GLOBAL_TENSOR_DATA(v101);
        // pto: %cmp_kv_cache_flat_inline2090__iter_v1_pview
        ;
        const int64_t v377 = 0;
        // pto: %cmp_kv_cache_flat_inline2090__iter_v1_pview
        ;
        const int64_t v378 = 512;
        // pto: %cmp_kv_cache_flat_inline2090__iter_v1_pview
        ;
        pto::Shape<1, 1, 1, 1, 512> v379 = pto::Shape<1, 1, 1, 1, 512>();
        // pto: %cmp_kv_cache_flat_inline2090__iter_v1_pview
        ;
        pto::Stride<512, 512, 512, 512, 1> v380 = pto::Stride<512, 512, 512, 512, 1>();
        // pto: %cmp_kv_cache_flat_inline2090__iter_v1_pview
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<512, 512, 512, 512, 1>, pto::Layout::ND> v381 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<512, 512, 512, 512, 1>, pto::Layout::ND>(v376 + (v377 + v375 * v378), v379, v380);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
        TSTORE(v381, v371);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
      };
    };
  }
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
  #endif // __DAV_VEC__

  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}