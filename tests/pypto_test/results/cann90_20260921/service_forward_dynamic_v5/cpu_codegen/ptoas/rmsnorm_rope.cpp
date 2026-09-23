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

AICORE void rmsnorm_rope(__gm__ bfloat16_t* v1, __gm__ int64_t* v2, __gm__ int32_t* v3, __gm__ float* v4, __gm__ float* v5, __gm__ float* v6, __gm__ float* v7, int64_t v8, int64_t v9, int64_t v10, int64_t v11, int64_t v12, int64_t v13, int32_t v14, int32_t v15) {
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  const int64_t v16 = 128;
  const int64_t v17 = 256;
  // pto: %c20480_i64
  const int64_t v18 = 20480;
  // pto: %c0_i64
  const int64_t v19 = 0;
  // pto: %c4096_i64
  const int64_t v20 = 4096;
  // pto: %c8192_i64
  const int64_t v21 = 8192;
  // pto: %c12288_i64
  const int64_t v22 = 12288;
  // pto: %c16384_i64
  const int64_t v23 = 16384;
  // pto: %c30976_i64
  const int64_t v24 = 30976;
  // pto: %c28672_i64
  const int64_t v25 = 28672;
  // pto: %c30720_i64
  const int64_t v26 = 30720;
  // pto: %c384_index
  const int64_t v27 = 384;
  // pto: %c128_index
  const int64_t v28 = 128;
  // pto: %c1_index
  const int64_t v29 = 1;
  // pto: %c64_index
  const int64_t v30 = 64;
  // pto: %c16_index
  const int64_t v31 = 16;
  // pto: %cst_14
  const float v32 = 1.0f;
  // pto: %c192_index
  const int64_t v33 = 192;
  // pto: %c0_i32
  const int32_t v34 = 0;
  // pto: %cst_17
  const float v35 = 0.5f;
  // pto: %cst_18
  const float v36 = 2.0f;
  // pto: %cst_19
  const float v37 = 0.0f;
  // pto: %c0_index
  const int64_t v38 = 0;
  // pto: %c1_i64
  const int64_t v39 = 1;
  // pto: %c4_i64
  const int64_t v40 = 4;
  // pto: %c6_index
  const int64_t v41 = 6;
  // pto: %cst_24
  const float v42 = 0.0078125f;
  // pto: %cst_25
  const float v43 = 9.99999997E-7f;
  // pto: %c3_index
  const int64_t v44 = 3;
  // pto: %c2_index
  const int64_t v45 = 2;
  // pto: %c4_index
  const int64_t v46 = 4;
  // pto: %normed_kv_inline214__ssa_v1_view
  const int64_t v47 = 1;
  // pto: %normed_kv_inline214__ssa_v1_view
  const int64_t v48 = 1;
  // pto: %normed_kv_inline214__ssa_v1_view
  const int64_t v49 = 1;
  // pto: %normed_kv_inline214__ssa_v1_view
  int64_t v50 = v27 * v28;
  // pto: %normed_kv_inline214__ssa_v1_view
  int64_t v51 = v49 * v50;
  // pto: %normed_kv_inline214__ssa_v1_view
  pto::Shape<1, 1, 1, -1, -1> v52 = pto::Shape<1, 1, 1, -1, -1>(v47, v48, v49, v27, v28);
  // pto: %normed_kv_inline214__ssa_v1_view
  pto::Stride<-1, -1, -1, -1, -1> v53 = pto::Stride<-1, -1, -1, -1, -1>(v48 * v51, v51, v50, v28, v29);
  // pto: %normed_kv_inline214__ssa_v1_view
  GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v54 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v1, v52, v53);
  // pto: %inner_freqs_cos__ssa_v0_view
  const int64_t v55 = 1;
  // pto: %inner_freqs_cos__ssa_v0_view
  const int64_t v56 = 1;
  // pto: %inner_freqs_cos__ssa_v0_view
  const int64_t v57 = 1;
  // pto: %inner_freqs_cos__ssa_v0_view
  int64_t v58 = (int64_t) v13;
  // pto: %inner_freqs_cos__ssa_v0_view
  int64_t v59 = v58 * v30;
  // pto: %inner_freqs_cos__ssa_v0_view
  int64_t v60 = v57 * v59;
  // pto: %inner_freqs_cos__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v61 = pto::Shape<1, 1, 1, -1, -1>(v55, v56, v57, v58, v30);
  // pto: %inner_freqs_cos__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v62 = pto::Stride<-1, -1, -1, -1, -1>(v56 * v60, v60, v59, v30, v29);
  // pto: %inner_freqs_cos__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v63 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v4, v61, v62);
  // pto: %inner_freqs_sin__ssa_v0_view
  const int64_t v64 = 1;
  // pto: %inner_freqs_sin__ssa_v0_view
  const int64_t v65 = 1;
  // pto: %inner_freqs_sin__ssa_v0_view
  const int64_t v66 = 1;
  // pto: %inner_freqs_sin__ssa_v0_view
  int64_t v67 = (int64_t) v13;
  // pto: %inner_freqs_sin__ssa_v0_view
  int64_t v68 = v67 * v30;
  // pto: %inner_freqs_sin__ssa_v0_view
  int64_t v69 = v66 * v68;
  // pto: %inner_freqs_sin__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v70 = pto::Shape<1, 1, 1, -1, -1>(v64, v65, v66, v67, v30);
  // pto: %inner_freqs_sin__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v71 = pto::Stride<-1, -1, -1, -1, -1>(v65 * v69, v69, v68, v30, v29);
  // pto: %inner_freqs_sin__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v72 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v5, v70, v71);
  // pto: %pooled_kv_inline2148__rv_v2_view
  const int64_t v73 = 1;
  // pto: %pooled_kv_inline2148__rv_v2_view
  const int64_t v74 = 1;
  // pto: %pooled_kv_inline2148__rv_v2_view
  const int64_t v75 = 1;
  // pto: %pooled_kv_inline2148__rv_v2_view
  int64_t v76 = v27 * v28;
  // pto: %pooled_kv_inline2148__rv_v2_view
  int64_t v77 = v75 * v76;
  // pto: %pooled_kv_inline2148__rv_v2_view
  pto::Shape<1, 1, 1, -1, -1> v78 = pto::Shape<1, 1, 1, -1, -1>(v73, v74, v75, v27, v28);
  // pto: %pooled_kv_inline2148__rv_v2_view
  pto::Stride<-1, -1, -1, -1, -1> v79 = pto::Stride<-1, -1, -1, -1, -1>(v74 * v77, v77, v76, v28, v29);
  // pto: %pooled_kv_inline2148__rv_v2_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v80 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v6, v78, v79);
  // pto: %norm_w_2d_inline434_inline2120__ssa_v0_view
  const int64_t v81 = 1;
  // pto: %norm_w_2d_inline434_inline2120__ssa_v0_view
  const int64_t v82 = 1;
  // pto: %norm_w_2d_inline434_inline2120__ssa_v0_view
  const int64_t v83 = 1;
  // pto: %norm_w_2d_inline434_inline2120__ssa_v0_view
  int64_t v84 = v29 * v28;
  // pto: %norm_w_2d_inline434_inline2120__ssa_v0_view
  int64_t v85 = v83 * v84;
  // pto: %norm_w_2d_inline434_inline2120__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v86 = pto::Shape<1, 1, 1, -1, -1>(v81, v82, v83, v29, v28);
  // pto: %norm_w_2d_inline434_inline2120__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v87 = pto::Stride<-1, -1, -1, -1, -1>(v82 * v85, v85, v84, v28, v29);
  // pto: %norm_w_2d_inline434_inline2120__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v88 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v7, v86, v87);
  // pto: %cosine_inline3248__phi_v4
  Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v89 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v30);
  // pto: %cosine_inline3248__phi_v4
  uint64_t v90 = (uint64_t) v21;
  TASSIGN(v89, v90);
  // pto: %sine_inline3247__phi_v4
  Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v91 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v30);
  // pto: %sine_inline3247__phi_v4
  uint64_t v92 = (uint64_t) v22;
  TASSIGN(v91, v92);
  // pto: %rms_worker_inline422_inline2140__ssa_v0
  // pto: %rope_ones_inline449_inline2114__tile
  Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v93 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v30);
  // pto: %rope_ones_inline449_inline2114__tile
  uint64_t v94 = (uint64_t) v18;
  TASSIGN(v93, v94);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
  TEXPANDS(v93, v32);
  // pto: %rope_index_inline456_inline2170__ci_tmp_v0
  Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v95 = Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v29, v33);
  // pto: %rope_index_inline456_inline2170__ci_tmp_v0
  uint64_t v96 = (uint64_t) v19;
  TASSIGN(v95, v96);
  // pto: %rope_index_inline456_inline2170__tile
  Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v97 = Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v29, v30);
  // pto: %rope_index_inline456_inline2170__tile
  uint64_t v98 = (uint64_t) v20;
  TASSIGN(v97, v98);
  // pto: %ci_tmp_view
  Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, 1, 192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v99;
  TRESHAPE(v99, v95);
  // pto: %ci_dst_view
  Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v100;
  TRESHAPE(v100, v97);
  TCI<Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, 1, 192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, int32_t, 0>(v100, v34, v99);
  set_flag(PIPE_S, PIPE_V, EVENT_ID0);
  // pto: %rope_index_f_inline401_inline2126__tile
  Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v101 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v29, v30);
  // pto: %rope_index_f_inline401_inline2126__tile
  uint64_t v102 = (uint64_t) v19;
  TASSIGN(v101, v102);
  wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
  RoundMode v103 = RoundMode::CAST_ROUND;
  SaturationMode v104 = SaturationMode::OFF;
  TCVT(v101, v97, v103, v104);
  // pto: %rope_col_inline400_inline2129__tile
  Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v105 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v30);
  // pto: %rope_col_inline400_inline2129__tile
  uint64_t v106 = (uint64_t) v18;
  TASSIGN(v105, v106);
  pipe_barrier(PIPE_V);
  TCOLEXPANDMUL(v105, v93, v101);
  // pto: %t__tile
  Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v107 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v30);
  // pto: %t__tile
  uint64_t v108 = (uint64_t) v19;
  TASSIGN(v107, v108);
  pipe_barrier(PIPE_V);
  TMULS(v107, v105, v35);
  // pto: %0
  Tile<TileType::Vec, int32_t, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v109 = Tile<TileType::Vec, int32_t, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v30);
  // pto: %0
  uint64_t v110 = (uint64_t) v19;
  TASSIGN(v109, v110);
  pipe_barrier(PIPE_V);
  RoundMode v111 = RoundMode::CAST_TRUNC;
  SaturationMode v112 = SaturationMode::ON;
  TCVT(v109, v107, v111, v112);
  // pto: %rope_dup_f_inline438_inline2169__tile
  Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v113 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v30);
  // pto: %rope_dup_f_inline438_inline2169__tile
  uint64_t v114 = (uint64_t) v19;
  TASSIGN(v113, v114);
  pipe_barrier(PIPE_V);
  RoundMode v115 = RoundMode::CAST_ROUND;
  SaturationMode v116 = SaturationMode::OFF;
  TCVT(v113, v109, v115, v116);
  // pto: %1
  Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v117 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v30);
  // pto: %1
  uint64_t v118 = (uint64_t) v19;
  TASSIGN(v117, v118);
  pipe_barrier(PIPE_V);
  TMULS(v117, v113, v36);
  // pto: %rope_lane_inline418_inline2112__tile
  Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v119 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v30);
  // pto: %rope_lane_inline418_inline2112__tile
  uint64_t v120 = (uint64_t) v19;
  TASSIGN(v119, v120);
  pipe_barrier(PIPE_V);
  TSUB(v119, v105, v117);
  // pto: %2
  Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v121 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v30);
  // pto: %2
  uint64_t v122 = (uint64_t) v18;
  TASSIGN(v121, v122);
  pipe_barrier(PIPE_V);
  TADDS(v121, v105, v32);
  // pto: %3
  Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v123 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v30);
  // pto: %3
  uint64_t v124 = (uint64_t) v20;
  TASSIGN(v123, v124);
  TMULS(v123, v119, v36);
  // pto: %4
  Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v125 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v30);
  // pto: %4
  uint64_t v126 = (uint64_t) v18;
  TASSIGN(v125, v126);
  pipe_barrier(PIPE_V);
  TSUB(v125, v121, v123);
  // pto: %rope_swap_idx_inline399_inline2111__tile
  Tile<TileType::Vec, int32_t, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v127 = Tile<TileType::Vec, int32_t, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v30);
  // pto: %rope_swap_idx_inline399_inline2111__tile
  uint64_t v128 = (uint64_t) v20;
  TASSIGN(v127, v128);
  pipe_barrier(PIPE_V);
  RoundMode v129 = RoundMode::CAST_ROUND;
  SaturationMode v130 = SaturationMode::ON;
  TCVT(v127, v125, v129, v130);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  for (int64_t v131 = (int64_t) v14; v131 < v8; v131 += v9) {
    // pto: %18
    ;
    int64_t v132 = (int64_t) ((uint64_t) v131 * (uint64_t) v31);
    // pto: %19
    ;
    int64_t v133 = (int64_t) ((uint64_t) v10 - (uint64_t) v132);
    // pto: %20
    ;
    int64_t v134 = v133 < v31 ? v133 : v31;
    // pto: %cosine_inline3248__tile
    ;
    Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v135 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v30);
    // pto: %cosine_inline3248__tile
    ;
    uint64_t v136 = (uint64_t) v21;
    TASSIGN(v135, v136);
    TEXPANDS(v135, v32);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    // pto: %sine_inline3247__tile
    ;
    Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v137 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v30);
    // pto: %sine_inline3247__tile
    ;
    uint64_t v138 = (uint64_t) v22;
    TASSIGN(v137, v138);
    TEXPANDS(v137, v37);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
    for (int64_t v139 = v38; v139 < v134; v139 += v29) {
      // pto: %21
      ;
      int64_t v140 = (int64_t) ((uint64_t) v132 + (uint64_t) v139);
      // pto: %position_inline3244__tile
      ;
      int64_t v141 = (v2)[v140];
      // pto: %22
      ;
      int64_t v142 = (int64_t) ((uint64_t) v141 + (uint64_t) v39);
      // pto: %23, %24
      ;
      if (v142 % v40 == v19) {
        // pto: %26, %25
        ;
        int32_t v143 = (v3)[v140 / v41];
        // pto: %27, %31, %29
        ;
        int64_t v144 = (int64_t) ((uint64_t) ((int64_t) v143) + (uint64_t) (v142 / v40));
        // pto: %gather_row_view
        ;
        int64_t v145 = (int64_t) ((uint64_t) v139 * (uint64_t) v17);
        // pto: %gather_row_view
        ;
        Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v146;
        // pto: %gather_row_view
        ;
        uint64_t v147 = (uint64_t) ((int64_t) (uint64_t) v145 + (uint64_t) v21);
        TASSIGN(v146, v147);
        // pto: %32
        ;
        int64_t v148 = v144 < v38 ? v38 : v144;
        // pto: %inner_freqs_cos__ssa_v0_pview
        ;
        __gm__ float* v149 = PTOAS__GLOBAL_TENSOR_DATA(v63);
        // pto: %inner_freqs_cos__ssa_v0_pview
        ;
        const int64_t v150 = 0;
        // pto: %inner_freqs_cos__ssa_v0_pview
        ;
        const int64_t v151 = 64;
        // pto: %inner_freqs_cos__ssa_v0_pview
        ;
        pto::Shape<1, 1, 1, 1, 64> v152 = pto::Shape<1, 1, 1, 1, 64>();
        // pto: %inner_freqs_cos__ssa_v0_pview
        ;
        pto::Stride<64, 64, 64, 64, 1> v153 = pto::Stride<64, 64, 64, 64, 1>();
        // pto: %inner_freqs_cos__ssa_v0_pview
        ;
        GlobalTensor<float, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<64, 64, 64, 64, 1>, pto::Layout::ND> v154 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<64, 64, 64, 64, 1>, pto::Layout::ND>(v149 + (v150 + v148 * v151), v152, v153);
        TLOAD(v146, v154);
        // pto: %33
        ;
        Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v155;
        // pto: %33
        ;
        uint64_t v156 = (uint64_t) ((int64_t) (uint64_t) v145 + (uint64_t) v22);
        TASSIGN(v155, v156);
        // pto: %inner_freqs_sin__ssa_v0_pview
        ;
        __gm__ float* v157 = PTOAS__GLOBAL_TENSOR_DATA(v72);
        // pto: %inner_freqs_sin__ssa_v0_pview
        ;
        const int64_t v158 = 0;
        // pto: %inner_freqs_sin__ssa_v0_pview
        ;
        const int64_t v159 = 64;
        // pto: %inner_freqs_sin__ssa_v0_pview
        ;
        pto::Shape<1, 1, 1, 1, 64> v160 = pto::Shape<1, 1, 1, 1, 64>();
        // pto: %inner_freqs_sin__ssa_v0_pview
        ;
        pto::Stride<64, 64, 64, 64, 1> v161 = pto::Stride<64, 64, 64, 64, 1>();
        // pto: %inner_freqs_sin__ssa_v0_pview
        ;
        GlobalTensor<float, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<64, 64, 64, 64, 1>, pto::Layout::ND> v162 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<64, 64, 64, 64, 1>, pto::Layout::ND>(v157 + (v158 + v148 * v159), v160, v161);
        TLOAD(v155, v162);
      };
    };
    // pto: %cos_b_inline407_inline2107__ssa_v0
    ;
    Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v163 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v30);
    // pto: %cos_b_inline407_inline2107__ssa_v0
    ;
    uint64_t v164 = (uint64_t) v21;
    TASSIGN(v163, v164);
    // pto: %sin_b_inline396_inline2137__ssa_v0
    ;
    Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v165 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v30);
    // pto: %sin_b_inline396_inline2137__ssa_v0
    ;
    uint64_t v166 = (uint64_t) v22;
    TASSIGN(v165, v166);
    // pto: %kv_rms_low_inline466_inline2106__tile
    ;
    Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v167 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v30);
    // pto: %kv_rms_low_inline466_inline2106__tile
    ;
    uint64_t v168 = (uint64_t) v18;
    TASSIGN(v167, v168);
    // pto: %35
    ;
    int64_t v169 = v132 < v38 ? v38 : v132;
    // pto: %pooled_kv_inline2148__rv_v2_pview
    ;
    __gm__ float* v170 = PTOAS__GLOBAL_TENSOR_DATA(v80);
    // pto: %pooled_kv_inline2148__rv_v2_pview
    ;
    const int64_t v171 = 0;
    // pto: %pooled_kv_inline2148__rv_v2_pview
    ;
    const int64_t v172 = 128;
    // pto: %pooled_kv_inline2148__rv_v2_pview
    ;
    pto::Shape<1, 1, 1, 16, 64> v173 = pto::Shape<1, 1, 1, 16, 64>();
    // pto: %pooled_kv_inline2148__rv_v2_pview
    ;
    pto::Stride<2048, 2048, 2048, 128, 1> v174 = pto::Stride<2048, 2048, 2048, 128, 1>();
    // pto: %pooled_kv_inline2148__rv_v2_pview
    ;
    GlobalTensor<float, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<2048, 2048, 2048, 128, 1>, pto::Layout::ND> v175 = GlobalTensor<float, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<2048, 2048, 2048, 128, 1>, pto::Layout::ND>(v170 + (v171 + v169 * v172), v173, v174);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
    TLOAD(v167, v175);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    // pto: %kv_rms_high_inline437_inline2104__tile
    ;
    Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v176 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v30);
    // pto: %kv_rms_high_inline437_inline2104__tile
    ;
    uint64_t v177 = (uint64_t) v23;
    TASSIGN(v176, v177);
    // pto: %37
    ;
    __gm__ float* v178 = PTOAS__GLOBAL_TENSOR_DATA(v80);
    // pto: %37
    ;
    const int64_t v179 = 64;
    // pto: %37
    ;
    const int64_t v180 = 128;
    // pto: %37
    ;
    pto::Shape<1, 1, 1, 16, 64> v181 = pto::Shape<1, 1, 1, 16, 64>();
    // pto: %37
    ;
    pto::Stride<2048, 2048, 2048, 128, 1> v182 = pto::Stride<2048, 2048, 2048, 128, 1>();
    // pto: %37
    ;
    GlobalTensor<float, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<2048, 2048, 2048, 128, 1>, pto::Layout::ND> v183 = GlobalTensor<float, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<2048, 2048, 2048, 128, 1>, pto::Layout::ND>(v178 + (v179 + v169 * v180), v181, v182);
    TLOAD(v176, v183);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
    // pto: %7
    ;
    Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v184 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v30);
    // pto: %7
    ;
    uint64_t v185 = (uint64_t) v18;
    TASSIGN(v184, v185);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TMUL(v184, v167, v167);
    // pto: %8
    ;
    Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v186 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v30);
    // pto: %8
    ;
    uint64_t v187 = (uint64_t) v23;
    TASSIGN(v186, v187);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
    TMUL(v186, v176, v176);
    // pto: %folded_sq_inline455_inline2103__tile
    ;
    Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v188 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v30);
    // pto: %folded_sq_inline455_inline2103__tile
    ;
    uint64_t v189 = (uint64_t) v23;
    TASSIGN(v188, v189);
    pipe_barrier(PIPE_V);
    TADD(v188, v184, v186);
    // pto: %tmp_tile
    ;
    Tile<TileType::Vec, float, 16, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v190 = Tile<TileType::Vec, float, 16, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v28);
    // pto: %tmp_tile
    ;
    uint64_t v191 = (uint64_t) v18;
    TASSIGN(v190, v191);
    // pto: %square_sum_inline395_inline2102__tile
    ;
    Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v192 = Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v29);
    // pto: %square_sum_inline395_inline2102__tile
    ;
    uint64_t v193 = (uint64_t) v24;
    TASSIGN(v192, v193);
    pipe_barrier(PIPE_V);
    TROWSUM(v192, v188, v190);
    // pto: %t__rm_a0_tmp_v0
    ;
    Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v194 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v29, v31);
    // pto: %t__rm_a0_tmp_v0
    ;
    uint64_t v195 = (uint64_t) v24;
    TASSIGN(v194, v195);
    // pto: %t__row_major_tmp_v1
    ;
    Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v196 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v29, v31);
    // pto: %t__row_major_tmp_v1
    ;
    uint64_t v197 = (uint64_t) v18;
    TASSIGN(v196, v197);
    pipe_barrier(PIPE_V);
    TMULS(v196, v194, v42);
    // pto: %9
    ;
    Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v198 = Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v29);
    // pto: %9
    ;
    uint64_t v199 = (uint64_t) v18;
    TASSIGN(v198, v199);
    // pto: %variance_inline420_inline2161__rm_a0_tmp_v2
    ;
    Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v200 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v29, v31);
    // pto: %variance_inline420_inline2161__rm_a0_tmp_v2
    ;
    uint64_t v201 = (uint64_t) v18;
    TASSIGN(v200, v201);
    // pto: %variance_inline420_inline2161__row_major_tmp_v3
    ;
    Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v202 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v29, v31);
    // pto: %variance_inline420_inline2161__row_major_tmp_v3
    ;
    uint64_t v203 = (uint64_t) v18;
    TASSIGN(v202, v203);
    pipe_barrier(PIPE_V);
    TADDS(v202, v200, v43);
    // pto: %variance_inline420_inline2161__tile
    ;
    Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v204 = Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v29);
    // pto: %variance_inline420_inline2161__tile
    ;
    uint64_t v205 = (uint64_t) v18;
    TASSIGN(v204, v205);
    // pto: %rms_inline412_inline2101__rm_a0_tmp_v4
    ;
    Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v206 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v29, v31);
    // pto: %rms_inline412_inline2101__rm_a0_tmp_v4
    ;
    uint64_t v207 = (uint64_t) v18;
    TASSIGN(v206, v207);
    // pto: %rms_inline412_inline2101__row_major_tmp_v5
    ;
    Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v208 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v29, v31);
    // pto: %rms_inline412_inline2101__row_major_tmp_v5
    ;
    uint64_t v209 = (uint64_t) v24;
    TASSIGN(v208, v209);
    pipe_barrier(PIPE_V);
    TSQRT(v208, v206);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID4);
    // pto: %rms_inline412_inline2101__tile
    ;
    Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v210 = Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v29);
    // pto: %rms_inline412_inline2101__tile
    ;
    uint64_t v211 = (uint64_t) v24;
    TASSIGN(v210, v211);
    // pto: %kv_norm_chunk_inline394_inline2100__tile
    ;
    Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v212 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v30);
    // pto: %kv_norm_chunk_inline394_inline2100__tile
    ;
    uint64_t v213 = (uint64_t) v18;
    TASSIGN(v212, v213);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID4);
    TLOAD(v212, v175);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
    // pto: %gamma_inline433_inline2099__tile
    ;
    Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v214 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v29, v30);
    // pto: %gamma_inline433_inline2099__tile
    ;
    uint64_t v215 = (uint64_t) v23;
    TASSIGN(v214, v215);
    // pto: %norm_w_2d_inline434_inline2120__ssa_v0_pview
    ;
    __gm__ float* v216 = PTOAS__GLOBAL_TENSOR_DATA(v88);
    // pto: %norm_w_2d_inline434_inline2120__ssa_v0_pview
    ;
    pto::Shape<1, 1, 1, 1, 64> v217 = pto::Shape<1, 1, 1, 1, 64>();
    // pto: %norm_w_2d_inline434_inline2120__ssa_v0_pview
    ;
    pto::Stride<128, 128, 128, 128, 1> v218 = pto::Stride<128, 128, 128, 128, 1>();
    // pto: %norm_w_2d_inline434_inline2120__ssa_v0_pview
    ;
    GlobalTensor<float, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<128, 128, 128, 128, 1>, pto::Layout::ND> v219 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<128, 128, 128, 128, 1>, pto::Layout::ND>(v216, v217, v218);
    TLOAD(v214, v219);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
    // pto: %10
    ;
    Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v220 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v30);
    // pto: %10
    ;
    uint64_t v221 = (uint64_t) v18;
    TASSIGN(v220, v221);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
    TROWEXPANDDIV(v220, v212, v210);
    // pto: %normed_chunk_inline431_inline2105__tile
    ;
    Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v222 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v30);
    // pto: %normed_chunk_inline431_inline2105__tile
    ;
    uint64_t v223 = (uint64_t) v18;
    TASSIGN(v222, v223);
    pipe_barrier(PIPE_V);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
    TCOLEXPANDMUL(v222, v220, v214);
    // pto: %normed_nope_inline442_inline2098__tile
    ;
    Tile<TileType::Vec, bfloat16_t, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v224 = Tile<TileType::Vec, bfloat16_t, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v30);
    // pto: %normed_nope_inline442_inline2098__tile
    ;
    uint64_t v225 = (uint64_t) v25;
    TASSIGN(v224, v225);
    pipe_barrier(PIPE_V);
    RoundMode v226 = RoundMode::CAST_RINT;
    SaturationMode v227 = SaturationMode::OFF;
    TCVT(v224, v222, v226, v227);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID5);
    // pto: %kv_rope_norm_inline393_inline2097__tile
    ;
    Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v228 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v30);
    // pto: %kv_rope_norm_inline393_inline2097__tile
    ;
    uint64_t v229 = (uint64_t) v18;
    TASSIGN(v228, v229);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID5);
    TLOAD(v228, v183);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);
    // pto: %gamma_rope_inline472_inline2160__tile
    ;
    Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v230 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v29, v30);
    // pto: %gamma_rope_inline472_inline2160__tile
    ;
    uint64_t v231 = (uint64_t) v23;
    TASSIGN(v230, v231);
    // pto: %42
    ;
    __gm__ float* v232 = PTOAS__GLOBAL_TENSOR_DATA(v88);
    // pto: %42
    ;
    unsigned v233 = 64;
    // pto: %42
    ;
    pto::Shape<1, 1, 1, 1, 64> v234 = pto::Shape<1, 1, 1, 1, 64>();
    // pto: %42
    ;
    pto::Stride<128, 128, 128, 128, 1> v235 = pto::Stride<128, 128, 128, 128, 1>();
    // pto: %42
    ;
    GlobalTensor<float, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<128, 128, 128, 128, 1>, pto::Layout::ND> v236 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<128, 128, 128, 128, 1>, pto::Layout::ND>(v232 + v233, v234, v235);
    TLOAD(v230, v236);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID5);
    // pto: %11
    ;
    Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v237 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v30);
    // pto: %11
    ;
    uint64_t v238 = (uint64_t) v18;
    TASSIGN(v237, v238);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);
    TROWEXPANDDIV(v237, v228, v210);
    // pto: %rope_normed_inline417_inline2096__tile
    ;
    Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v239 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v30);
    // pto: %rope_normed_inline417_inline2096__tile
    ;
    uint64_t v240 = (uint64_t) v18;
    TASSIGN(v239, v240);
    pipe_barrier(PIPE_V);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID5);
    TCOLEXPANDMUL(v239, v237, v230);
    // pto: %gather_acc_init
    ;
    Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v241 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v30);
    // pto: %gather_acc_init
    ;
    uint64_t v242 = (uint64_t) v23;
    TASSIGN(v241, v242);
    for (int64_t v243 = v38; v243 < v31; v243 += v29) {
      // pto: %gather_inp_row
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v244 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v29, v30);
      // pto: %gather_inp_row
      ;
      uint64_t v245 = (uint64_t) v18;
      TASSIGN(v244, v245);
      // pto: %slice_view
      ;
      int64_t v246 = (int64_t) ((uint64_t) v243 * (uint64_t) v17);
      // pto: %slice_view
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v247;
      // pto: %slice_view
      ;
      uint64_t v248 = (uint64_t) ((int64_t) (uint64_t) v246 + (uint64_t) v18);
      TASSIGN(v247, v248);
      // pto: %gather_idx_row
      ;
      Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v249 = Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v29, v30);
      // pto: %gather_idx_row
      ;
      uint64_t v250 = (uint64_t) v20;
      TASSIGN(v249, v250);
      // pto: %43
      ;
      Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v251;
      // pto: %43
      ;
      uint64_t v252 = (uint64_t) ((int64_t) (uint64_t) v246 + (uint64_t) v20);
      TASSIGN(v251, v252);
      // pto: %gather_row_tmp
      ;
      Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v253 = Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v29, v30);
      // pto: %gather_row_tmp
      ;
      uint64_t v254 = (uint64_t) v24;
      TASSIGN(v253, v254);
      // pto: %gather_row
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v255 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v29, v30);
      // pto: %gather_row
      ;
      uint64_t v256 = (uint64_t) v26;
      TASSIGN(v255, v256);
      pipe_barrier(PIPE_V);
      TGATHER(v255, v247, v251, v253);
      // pto: %assemble_view
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v257;
      // pto: %assemble_view
      ;
      uint64_t v258 = (uint64_t) ((int64_t) (uint64_t) v246 + (uint64_t) v23);
      TASSIGN(v257, v258);
      pipe_barrier(PIPE_V);
      TMOV(v257, v255);
    };
    // pto: %swapped_inline392_inline2191__tile
    ;
    Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v259 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v30);
    // pto: %swapped_inline392_inline2191__tile
    ;
    uint64_t v260 = (uint64_t) v23;
    TASSIGN(v259, v260);
    // pto: %12
    ;
    Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v261 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v30);
    // pto: %12
    ;
    uint64_t v262 = (uint64_t) v24;
    TASSIGN(v261, v262);
    TMULS(v261, v119, v36);
    // pto: %13
    ;
    Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v263 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v30);
    // pto: %13
    ;
    uint64_t v264 = (uint64_t) v24;
    TASSIGN(v263, v264);
    pipe_barrier(PIPE_V);
    TSUBS(v263, v261, v32);
    // pto: %sin_signed_inline390_inline2168__tile
    ;
    Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v265 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v30);
    // pto: %sin_signed_inline390_inline2168__tile
    ;
    uint64_t v266 = (uint64_t) v22;
    TASSIGN(v265, v266);
    pipe_barrier(PIPE_V);
    TMUL(v265, v165, v263);
    // pto: %14
    ;
    Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v267 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v30);
    // pto: %14
    ;
    uint64_t v268 = (uint64_t) v18;
    TASSIGN(v267, v268);
    TMUL(v267, v239, v163);
    // pto: %15
    ;
    Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v269 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v30);
    // pto: %15
    ;
    uint64_t v270 = (uint64_t) v21;
    TASSIGN(v269, v270);
    pipe_barrier(PIPE_V);
    TMUL(v269, v259, v265);
    // pto: %rope_rot_inline391_inline2095__tile
    ;
    Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v271 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v30);
    // pto: %rope_rot_inline391_inline2095__tile
    ;
    uint64_t v272 = (uint64_t) v18;
    TASSIGN(v271, v272);
    pipe_barrier(PIPE_V);
    TADD(v271, v267, v269);
    // pto: %normed_rope_inline447_inline2200__tile
    ;
    Tile<TileType::Vec, bfloat16_t, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v273 = Tile<TileType::Vec, bfloat16_t, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v30);
    // pto: %normed_rope_inline447_inline2200__tile
    ;
    uint64_t v274 = (uint64_t) v18;
    TASSIGN(v273, v274);
    pipe_barrier(PIPE_V);
    RoundMode v275 = RoundMode::CAST_RINT;
    SaturationMode v276 = SaturationMode::OFF;
    TCVT(v273, v271, v275, v276);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
    for (int64_t v277 = v38; v277 < v134; v277 += v29) {
      // pto: %44
      ;
      int64_t v278 = (int64_t) ((uint64_t) v132 + (uint64_t) v277);
      // pto: %token_pos_inline440_inline2175__tile
      ;
      int64_t v279 = (v2)[v278];
      // pto: %45, %46, %47
      ;
      if ((int64_t) ((uint64_t) v279 + (uint64_t) v39) % v40 == v19) {
        // pto: %48
        ;
        int64_t v280 = v278 / v41;
        // pto: %49, %first_pos_inline389_inline2113__tile
        ;
        int64_t v281 = (v2)[(int64_t) ((uint64_t) v280 * (uint64_t) v41)];
        // pto: %53, %57, %54, %55, %52, %50, %56
        ;
        int64_t v282 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) v280 * (uint64_t) v45) + (uint64_t) ((int64_t) ((uint64_t) (v278 % v41) - (uint64_t) ((int64_t) (uint64_t) v44 - (uint64_t) (v281 % v40))) / v46));
        // pto: %16
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v283 = Tile<TileType::Vec, bfloat16_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v29, v30);
        // pto: %16
        ;
        uint64_t v284 = (uint64_t) v25;
        TASSIGN(v283, v284);
        // pto: %58
        ;
        int64_t v285 = (int64_t) ((uint64_t) v277 * (uint64_t) v16);
        // pto: %58
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v286;
        // pto: %58
        ;
        uint64_t v287 = (uint64_t) ((int64_t) (uint64_t) v285 + (uint64_t) v25);
        TASSIGN(v286, v287);
        // pto: %59
        ;
        int64_t v288 = v282 < v38 ? v38 : v282;
        // pto: %normed_kv_inline214__iter_v4_pview
        ;
        __gm__ bfloat16_t* v289 = PTOAS__GLOBAL_TENSOR_DATA(v54);
        // pto: %normed_kv_inline214__iter_v4_pview
        ;
        const int64_t v290 = 0;
        // pto: %normed_kv_inline214__iter_v4_pview
        ;
        const int64_t v291 = 128;
        // pto: %normed_kv_inline214__iter_v4_pview
        ;
        pto::Shape<1, 1, 1, 1, 64> v292 = pto::Shape<1, 1, 1, 1, 64>();
        // pto: %normed_kv_inline214__iter_v4_pview
        ;
        pto::Stride<128, 128, 128, 128, 1> v293 = pto::Stride<128, 128, 128, 128, 1>();
        // pto: %normed_kv_inline214__iter_v4_pview
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<128, 128, 128, 128, 1>, pto::Layout::ND> v294 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<128, 128, 128, 128, 1>, pto::Layout::ND>(v289 + (v290 + v288 * v291), v292, v293);
        pipe_barrier(PIPE_MTE3);
        TSTORE(v294, v286);
        // pto: %17
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v295 = Tile<TileType::Vec, bfloat16_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v29, v30);
        // pto: %17
        ;
        uint64_t v296 = (uint64_t) v18;
        TASSIGN(v295, v296);
        // pto: %60
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v297;
        // pto: %60
        ;
        uint64_t v298 = (uint64_t) ((int64_t) (uint64_t) v285 + (uint64_t) v18);
        TASSIGN(v297, v298);
        // pto: %normed_kv_inline214__tile_pview
        ;
        __gm__ bfloat16_t* v299 = PTOAS__GLOBAL_TENSOR_DATA(v54);
        // pto: %normed_kv_inline214__tile_pview
        ;
        const int64_t v300 = 64;
        // pto: %normed_kv_inline214__tile_pview
        ;
        const int64_t v301 = 128;
        // pto: %normed_kv_inline214__tile_pview
        ;
        pto::Shape<1, 1, 1, 1, 64> v302 = pto::Shape<1, 1, 1, 1, 64>();
        // pto: %normed_kv_inline214__tile_pview
        ;
        pto::Stride<128, 128, 128, 128, 1> v303 = pto::Stride<128, 128, 128, 128, 1>();
        // pto: %normed_kv_inline214__tile_pview
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<128, 128, 128, 128, 1>, pto::Layout::ND> v304 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<128, 128, 128, 128, 1>, pto::Layout::ND>(v299 + (v300 + v288 * v301), v302, v303);
        pipe_barrier(PIPE_MTE3);
        TSTORE(v304, v297);
      };
    };
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  }
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
  #endif // __DAV_VEC__

  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}