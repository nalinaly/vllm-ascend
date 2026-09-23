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

AICORE void kv_rms_norm_rope(__gm__ float* v1, __gm__ bfloat16_t* v2, __gm__ bfloat16_t* v3, __gm__ float* v4, __gm__ float* v5, __gm__ int32_t* v6, int64_t v7, int64_t v8, int64_t v9, int64_t v10, int64_t v11, int64_t v12, int32_t v13, int32_t v14) {
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  const int64_t v15 = 256;
  // pto: %c32768_i64
  const int64_t v16 = 32768;
  // pto: %c16384_i64
  const int64_t v17 = 16384;
  // pto: %c57344_i64
  const int64_t v18 = 57344;
  // pto: %c40960_i64
  const int64_t v19 = 40960;
  // pto: %c49152_i64
  const int64_t v20 = 49152;
  // pto: %c0_i64
  const int64_t v21 = 0;
  // pto: %c8192_i64
  const int64_t v22 = 8192;
  // pto: %c512_index
  const int64_t v23 = 512;
  // pto: %c1_index
  const int64_t v24 = 1;
  // pto: %c64_index
  const int64_t v25 = 64;
  // pto: %c32_index
  const int64_t v26 = 32;
  // pto: %cst_11
  const float v27 = 0.0f;
  // pto: %c0_index
  const int64_t v28 = 0;
  // pto: %c128_index
  const int64_t v29 = 128;
  // pto: %cst_14
  const float v30 = 0.001953125f;
  // pto: %cst_15
  const float v31 = 9.99999997E-7f;
  // pto: %cst_16
  const float v32 = 1.0f;
  // pto: %c384_index
  const int64_t v33 = 384;
  // pto: %c448_index
  const int64_t v34 = 448;
  // pto: %c192_index
  const int64_t v35 = 192;
  // pto: %c0_i32
  const int32_t v36 = 0;
  // pto: %cst_21
  const float v37 = 0.5f;
  // pto: %cst_22
  const float v38 = 2.0f;
  // pto: %cst_23
  const float v39 = 64.0f;
  // pto: %kv_fp32_inline1890__phi_v18_view
  const int64_t v40 = 1;
  // pto: %kv_fp32_inline1890__phi_v18_view
  const int64_t v41 = 1;
  // pto: %kv_fp32_inline1890__phi_v18_view
  const int64_t v42 = 1;
  // pto: %kv_fp32_inline1890__phi_v18_view
  int64_t v43 = (int64_t) v9;
  // pto: %kv_fp32_inline1890__phi_v18_view
  int64_t v44 = v43 * v23;
  // pto: %kv_fp32_inline1890__phi_v18_view
  int64_t v45 = v42 * v44;
  // pto: %kv_fp32_inline1890__phi_v18_view
  pto::Shape<1, 1, 1, -1, -1> v46 = pto::Shape<1, 1, 1, -1, -1>(v40, v41, v42, v43, v23);
  // pto: %kv_fp32_inline1890__phi_v18_view
  pto::Stride<-1, -1, -1, -1, -1> v47 = pto::Stride<-1, -1, -1, -1, -1>(v41 * v45, v45, v44, v23, v24);
  // pto: %kv_fp32_inline1890__phi_v18_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v48 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v1, v46, v47);
  // pto: %kv_view_inline1911__ssa_v0_view
  const int64_t v49 = 1;
  // pto: %kv_view_inline1911__ssa_v0_view
  const int64_t v50 = 1;
  // pto: %kv_view_inline1911__ssa_v0_view
  const int64_t v51 = 1;
  // pto: %kv_view_inline1911__ssa_v0_view
  int64_t v52 = (int64_t) v10;
  // pto: %kv_view_inline1911__ssa_v0_view
  int64_t v53 = v52 * v23;
  // pto: %kv_view_inline1911__ssa_v0_view
  int64_t v54 = v51 * v53;
  // pto: %kv_view_inline1911__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v55 = pto::Shape<1, 1, 1, -1, -1>(v49, v50, v51, v52, v23);
  // pto: %kv_view_inline1911__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v56 = pto::Stride<-1, -1, -1, -1, -1>(v50 * v54, v54, v53, v23, v24);
  // pto: %kv_view_inline1911__ssa_v0_view
  GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v57 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v2, v55, v56);
  // pto: %gamma_ckv__ssa_v0_view
  const int64_t v58 = 1;
  // pto: %gamma_ckv__ssa_v0_view
  const int64_t v59 = 1;
  // pto: %gamma_ckv__ssa_v0_view
  const int64_t v60 = 1;
  // pto: %gamma_ckv__ssa_v0_view
  const int64_t v61 = 1;
  // pto: %gamma_ckv__ssa_v0_view
  int64_t v62 = v23 * v24;
  // pto: %gamma_ckv__ssa_v0_view
  int64_t v63 = v61 * v62;
  // pto: %gamma_ckv__ssa_v0_view
  int64_t v64 = v60 * v63;
  // pto: %gamma_ckv__ssa_v0_view
  pto::Shape<1, 1, 1, 1, -1> v65 = pto::Shape<1, 1, 1, 1, -1>(v58, v59, v60, v61, v23);
  // pto: %gamma_ckv__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v66 = pto::Stride<-1, -1, -1, -1, -1>(v59 * v64, v64, v63, v62, v24);
  // pto: %gamma_ckv__ssa_v0_view
  GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v67 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v3, v65, v66);
  // pto: %freqs_cos__ssa_v0_view
  const int64_t v68 = 1;
  // pto: %freqs_cos__ssa_v0_view
  const int64_t v69 = 1;
  // pto: %freqs_cos__ssa_v0_view
  const int64_t v70 = 1;
  // pto: %freqs_cos__ssa_v0_view
  int64_t v71 = (int64_t) v11;
  // pto: %freqs_cos__ssa_v0_view
  int64_t v72 = v71 * v25;
  // pto: %freqs_cos__ssa_v0_view
  int64_t v73 = v70 * v72;
  // pto: %freqs_cos__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v74 = pto::Shape<1, 1, 1, -1, -1>(v68, v69, v70, v71, v25);
  // pto: %freqs_cos__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v75 = pto::Stride<-1, -1, -1, -1, -1>(v69 * v73, v73, v72, v25, v24);
  // pto: %freqs_cos__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v76 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v4, v74, v75);
  // pto: %q_rope_sin_signed_inline197__ssa_v0_view
  const int64_t v77 = 1;
  // pto: %q_rope_sin_signed_inline197__ssa_v0_view
  const int64_t v78 = 1;
  // pto: %q_rope_sin_signed_inline197__ssa_v0_view
  const int64_t v79 = 1;
  // pto: %q_rope_sin_signed_inline197__ssa_v0_view
  int64_t v80 = (int64_t) v12;
  // pto: %q_rope_sin_signed_inline197__ssa_v0_view
  int64_t v81 = v80 * v25;
  // pto: %q_rope_sin_signed_inline197__ssa_v0_view
  int64_t v82 = v79 * v81;
  // pto: %q_rope_sin_signed_inline197__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v83 = pto::Shape<1, 1, 1, -1, -1>(v77, v78, v79, v80, v25);
  // pto: %q_rope_sin_signed_inline197__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v84 = pto::Stride<-1, -1, -1, -1, -1>(v78 * v82, v82, v81, v25, v24);
  // pto: %q_rope_sin_signed_inline197__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v85 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v5, v83, v84);
  // pto: %q_rope_swap_idx_inline196__ssa_v0_view
  const int64_t v86 = 1;
  // pto: %q_rope_swap_idx_inline196__ssa_v0_view
  const int64_t v87 = 1;
  // pto: %q_rope_swap_idx_inline196__ssa_v0_view
  const int64_t v88 = 1;
  // pto: %q_rope_swap_idx_inline196__ssa_v0_view
  int64_t v89 = (int64_t) v12;
  // pto: %q_rope_swap_idx_inline196__ssa_v0_view
  int64_t v90 = v89 * v25;
  // pto: %q_rope_swap_idx_inline196__ssa_v0_view
  int64_t v91 = v88 * v90;
  // pto: %q_rope_swap_idx_inline196__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v92 = pto::Shape<1, 1, 1, -1, -1>(v86, v87, v88, v89, v25);
  // pto: %q_rope_swap_idx_inline196__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v93 = pto::Stride<-1, -1, -1, -1, -1>(v87 * v91, v91, v90, v25, v24);
  // pto: %q_rope_swap_idx_inline196__ssa_v0_view
  GlobalTensor<int32_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v94 = GlobalTensor<int32_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v6, v92, v93);
  // pto: %tg_idx_inline1931__ssa_v0, %63
  int64_t v95 = (int64_t) ((uint64_t) ((int64_t) v13) * (uint64_t) v26);
  // pto: %64
  int64_t v96 = (int64_t) ((uint64_t) v7 - (uint64_t) v95);
  // pto: %65
  int64_t v97 = v96 < v26 ? v96 : v26;
  // pto: %66
  int64_t v98 = (int64_t) ((uint64_t) v8 + (uint64_t) v95);
  // pto: %67
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID4);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID4);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID5);
  if (v97 == v26) {
    // pto: %kv_sq_lanes_inline1878__tile
    ;
    Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v99 = Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v26, v25);
    // pto: %kv_sq_lanes_inline1878__tile
    ;
    uint64_t v100 = (uint64_t) v16;
    TASSIGN(v99, v100);
    TEXPANDS(v99, v27);
    for (int64_t v101 = v28; v101 < v23; v101 += v29) {
      // pto: %kv_chunk_inline1959__tile
      ;
      Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v102 = Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v26, v25);
      // pto: %kv_chunk_inline1959__tile
      ;
      uint64_t v103 = (uint64_t) v17;
      TASSIGN(v102, v103);
      // pto: %68
      ;
      int64_t v104 = v95 < v28 ? v28 : v95;
      // pto: %69
      ;
      int64_t v105 = v101 < v28 ? v28 : v101;
      // pto: %kv_fp32_inline1890__phi_v18_pview
      ;
      __gm__ float* v106 = PTOAS__GLOBAL_TENSOR_DATA(v48);
      // pto: %kv_fp32_inline1890__phi_v18_pview
      ;
      const int64_t v107 = 0;
      // pto: %kv_fp32_inline1890__phi_v18_pview
      ;
      const int64_t v108 = 512;
      // pto: %kv_fp32_inline1890__phi_v18_pview
      ;
      pto::Shape<1, 1, 1, 32, 64> v109 = pto::Shape<1, 1, 1, 32, 64>();
      // pto: %kv_fp32_inline1890__phi_v18_pview
      ;
      pto::Stride<16384, 16384, 16384, 512, 1> v110 = pto::Stride<16384, 16384, 16384, 512, 1>();
      // pto: %kv_fp32_inline1890__phi_v18_pview
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 32, 64>, pto::Stride<16384, 16384, 16384, 512, 1>, pto::Layout::ND> v111 = GlobalTensor<float, pto::Shape<1, 1, 1, 32, 64>, pto::Stride<16384, 16384, 16384, 512, 1>, pto::Layout::ND>(v106 + (v107 + v104 * v108 + v105), v109, v110);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      TLOAD(v102, v111);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      // pto: %0
      ;
      Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v112 = Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v26, v25);
      // pto: %0
      ;
      uint64_t v113 = (uint64_t) v18;
      TASSIGN(v112, v113);
      // pto: %71
      ;
      int64_t v114 = (int64_t) ((uint64_t) v101 + (uint64_t) v25);
      // pto: %72
      ;
      int64_t v115 = v114 < v28 ? v28 : v114;
      // pto: %73
      ;
      __gm__ float* v116 = PTOAS__GLOBAL_TENSOR_DATA(v48);
      // pto: %73
      ;
      const int64_t v117 = 0;
      // pto: %73
      ;
      const int64_t v118 = 512;
      // pto: %73
      ;
      pto::Shape<1, 1, 1, 32, 64> v119 = pto::Shape<1, 1, 1, 32, 64>();
      // pto: %73
      ;
      pto::Stride<16384, 16384, 16384, 512, 1> v120 = pto::Stride<16384, 16384, 16384, 512, 1>();
      // pto: %73
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 32, 64>, pto::Stride<16384, 16384, 16384, 512, 1>, pto::Layout::ND> v121 = GlobalTensor<float, pto::Shape<1, 1, 1, 32, 64>, pto::Stride<16384, 16384, 16384, 512, 1>, pto::Layout::ND>(v116 + (v117 + v104 * v118 + v115), v119, v120);
      TLOAD(v112, v121);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
      // pto: %t__tile
      ;
      Tile<TileType::Vec, bfloat16_t, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v122 = Tile<TileType::Vec, bfloat16_t, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v26, v25);
      // pto: %t__tile
      ;
      uint64_t v123 = (uint64_t) v19;
      TASSIGN(v122, v123);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      RoundMode v124 = RoundMode::CAST_RINT;
      SaturationMode v125 = SaturationMode::OFF;
      TCVT(v122, v102, v124, v125);
      // pto: %kv_chunk_v1_inline1939__tile
      ;
      Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v126 = Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v26, v25);
      // pto: %kv_chunk_v1_inline1939__tile
      ;
      uint64_t v127 = (uint64_t) v17;
      TASSIGN(v126, v127);
      pipe_barrier(PIPE_V);
      RoundMode v128 = RoundMode::CAST_ROUND;
      SaturationMode v129 = SaturationMode::OFF;
      TCVT(v126, v122, v128, v129);
      // pto: %kv_sq_inline1876__tile
      ;
      Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v130 = Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v26, v25);
      // pto: %kv_sq_inline1876__tile
      ;
      uint64_t v131 = (uint64_t) v17;
      TASSIGN(v130, v131);
      pipe_barrier(PIPE_V);
      TMUL(v130, v126, v126);
      // pto: %1
      ;
      Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v132 = Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v26, v25);
      // pto: %1
      ;
      uint64_t v133 = (uint64_t) v17;
      TASSIGN(v132, v133);
      pipe_barrier(PIPE_V);
      TADD(v132, v99, v130);
      // pto: %2
      ;
      Tile<TileType::Vec, bfloat16_t, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v134 = Tile<TileType::Vec, bfloat16_t, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v26, v25);
      // pto: %2
      ;
      uint64_t v135 = (uint64_t) v20;
      TASSIGN(v134, v135);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
      RoundMode v136 = RoundMode::CAST_RINT;
      SaturationMode v137 = SaturationMode::OFF;
      TCVT(v134, v112, v136, v137);
      // pto: %3
      ;
      Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v138 = Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v26, v25);
      // pto: %3
      ;
      uint64_t v139 = (uint64_t) v18;
      TASSIGN(v138, v139);
      pipe_barrier(PIPE_V);
      RoundMode v140 = RoundMode::CAST_ROUND;
      SaturationMode v141 = SaturationMode::OFF;
      TCVT(v138, v134, v140, v141);
      // pto: %4
      ;
      Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v142 = Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v26, v25);
      // pto: %4
      ;
      uint64_t v143 = (uint64_t) v18;
      TASSIGN(v142, v143);
      pipe_barrier(PIPE_V);
      TMUL(v142, v138, v138);
      // pto: %5
      ;
      Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v144 = Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v26, v25);
      // pto: %5
      ;
      uint64_t v145 = (uint64_t) v16;
      TASSIGN(v144, v145);
      pipe_barrier(PIPE_V);
      TADD(v144, v132, v142);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    };
    // pto: %tmp_tile
    ;
    Tile<TileType::Vec, float, 32, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v146 = Tile<TileType::Vec, float, 32, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v26, v29);
    // pto: %tmp_tile
    ;
    uint64_t v147 = (uint64_t) v17;
    TASSIGN(v146, v147);
    // pto: %6
    ;
    Tile<TileType::Vec, float, 32, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v148 = Tile<TileType::Vec, float, 32, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v26, v24);
    // pto: %6
    ;
    uint64_t v149 = (uint64_t) v18;
    TASSIGN(v148, v149);
    pipe_barrier(PIPE_V);
    TROWSUM(v148, v99, v146);
    // pto: %kv_sq_sum_inline1886__tile
    ;
    Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v150 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v24, v26);
    // pto: %kv_sq_sum_inline1886__tile
    ;
    uint64_t v151 = (uint64_t) v18;
    TASSIGN(v150, v151);
    // pto: %7
    ;
    Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v152 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v24, v26);
    // pto: %7
    ;
    uint64_t v153 = (uint64_t) v17;
    TASSIGN(v152, v153);
    pipe_barrier(PIPE_V);
    TMULS(v152, v150, v30);
    // pto: %8
    ;
    Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v154 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v24, v26);
    // pto: %8
    ;
    uint64_t v155 = (uint64_t) v17;
    TASSIGN(v154, v155);
    pipe_barrier(PIPE_V);
    TADDS(v154, v152, v31);
    // pto: %kv_rms_inline1875__tile
    ;
    Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v156 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v24, v26);
    // pto: %kv_rms_inline1875__tile
    ;
    uint64_t v157 = (uint64_t) v17;
    TASSIGN(v156, v157);
    pipe_barrier(PIPE_V);
    TSQRT(v156, v154);
    // pto: %9
    ;
    Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v158 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v24, v26);
    // pto: %9
    ;
    uint64_t v159 = (uint64_t) v16;
    TASSIGN(v158, v159);
    TEXPANDS(v158, v32);
    // pto: %kv_inv_rms_inline1894__tile
    ;
    Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v160 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v24, v26);
    // pto: %kv_inv_rms_inline1894__tile
    ;
    uint64_t v161 = (uint64_t) v20;
    TASSIGN(v160, v161);
    pipe_barrier(PIPE_V);
    TDIV(v160, v158, v156);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    // pto: %kv_inv_rms_t_inline1910__tile
    ;
    Tile<TileType::Vec, float, 32, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v162 = Tile<TileType::Vec, float, 32, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v26, v24);
    // pto: %kv_inv_rms_t_inline1910__tile
    ;
    uint64_t v163 = (uint64_t) v20;
    TASSIGN(v162, v163);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    for (int64_t v164 = v28; v164 < v33; v164 += v29) {
      // pto: %10
      ;
      Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v165 = Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v26, v25);
      // pto: %10
      ;
      uint64_t v166 = (uint64_t) v17;
      TASSIGN(v165, v166);
      // pto: %74
      ;
      int64_t v167 = v95 < v28 ? v28 : v95;
      // pto: %75
      ;
      int64_t v168 = v164 < v28 ? v28 : v164;
      // pto: %76
      ;
      __gm__ float* v169 = PTOAS__GLOBAL_TENSOR_DATA(v48);
      // pto: %76
      ;
      const int64_t v170 = 0;
      // pto: %76
      ;
      const int64_t v171 = 512;
      // pto: %76
      ;
      pto::Shape<1, 1, 1, 32, 64> v172 = pto::Shape<1, 1, 1, 32, 64>();
      // pto: %76
      ;
      pto::Stride<16384, 16384, 16384, 512, 1> v173 = pto::Stride<16384, 16384, 16384, 512, 1>();
      // pto: %76
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 32, 64>, pto::Stride<16384, 16384, 16384, 512, 1>, pto::Layout::ND> v174 = GlobalTensor<float, pto::Shape<1, 1, 1, 32, 64>, pto::Stride<16384, 16384, 16384, 512, 1>, pto::Layout::ND>(v169 + (v170 + v167 * v171 + v168), v172, v173);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
      TLOAD(v165, v174);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
      // pto: %11
      ;
      Tile<TileType::Vec, bfloat16_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v175 = Tile<TileType::Vec, bfloat16_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v24, v25);
      // pto: %11
      ;
      uint64_t v176 = (uint64_t) v21;
      TASSIGN(v175, v176);
      // pto: %gamma_ckv__ssa_v0_pview
      ;
      __gm__ bfloat16_t* v177 = PTOAS__GLOBAL_TENSOR_DATA(v67);
      // pto: %gamma_ckv__ssa_v0_pview
      ;
      const int64_t v178 = 0;
      // pto: %gamma_ckv__ssa_v0_pview
      ;
      pto::Shape<1, 1, 1, 1, 64> v179 = pto::Shape<1, 1, 1, 1, 64>();
      // pto: %gamma_ckv__ssa_v0_pview
      ;
      pto::Stride<64, 64, 64, 64, 1> v180 = pto::Stride<64, 64, 64, 64, 1>();
      // pto: %gamma_ckv__ssa_v0_pview
      ;
      GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<64, 64, 64, 64, 1>, pto::Layout::ND> v181 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<64, 64, 64, 64, 1>, pto::Layout::ND>(v177 + (v178 + v168), v179, v180);
      TLOAD(v175, v181);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
      // pto: %12
      ;
      Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v182 = Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v26, v25);
      // pto: %12
      ;
      uint64_t v183 = (uint64_t) v16;
      TASSIGN(v182, v183);
      // pto: %79
      ;
      int64_t v184 = (int64_t) ((uint64_t) v164 + (uint64_t) v25);
      // pto: %80
      ;
      int64_t v185 = v184 < v28 ? v28 : v184;
      // pto: %81
      ;
      __gm__ float* v186 = PTOAS__GLOBAL_TENSOR_DATA(v48);
      // pto: %81
      ;
      const int64_t v187 = 0;
      // pto: %81
      ;
      const int64_t v188 = 512;
      // pto: %81
      ;
      pto::Shape<1, 1, 1, 32, 64> v189 = pto::Shape<1, 1, 1, 32, 64>();
      // pto: %81
      ;
      pto::Stride<16384, 16384, 16384, 512, 1> v190 = pto::Stride<16384, 16384, 16384, 512, 1>();
      // pto: %81
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 32, 64>, pto::Stride<16384, 16384, 16384, 512, 1>, pto::Layout::ND> v191 = GlobalTensor<float, pto::Shape<1, 1, 1, 32, 64>, pto::Stride<16384, 16384, 16384, 512, 1>, pto::Layout::ND>(v186 + (v187 + v167 * v188 + v185), v189, v190);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
      TLOAD(v182, v191);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);
      // pto: %13
      ;
      Tile<TileType::Vec, bfloat16_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v192 = Tile<TileType::Vec, bfloat16_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v24, v25);
      // pto: %13
      ;
      uint64_t v193 = (uint64_t) v22;
      TASSIGN(v192, v193);
      // pto: %84
      ;
      __gm__ bfloat16_t* v194 = PTOAS__GLOBAL_TENSOR_DATA(v67);
      // pto: %84
      ;
      const int64_t v195 = 0;
      // pto: %84
      ;
      pto::Shape<1, 1, 1, 1, 64> v196 = pto::Shape<1, 1, 1, 1, 64>();
      // pto: %84
      ;
      pto::Stride<64, 64, 64, 64, 1> v197 = pto::Stride<64, 64, 64, 64, 1>();
      // pto: %84
      ;
      GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<64, 64, 64, 64, 1>, pto::Layout::ND> v198 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<64, 64, 64, 64, 1>, pto::Layout::ND>(v194 + (v195 + v185), v196, v197);
      TLOAD(v192, v198);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID5);
      // pto: %14
      ;
      Tile<TileType::Vec, bfloat16_t, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v199 = Tile<TileType::Vec, bfloat16_t, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v26, v25);
      // pto: %14
      ;
      uint64_t v200 = (uint64_t) v18;
      TASSIGN(v199, v200);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
      RoundMode v201 = RoundMode::CAST_RINT;
      SaturationMode v202 = SaturationMode::OFF;
      TCVT(v199, v165, v201, v202);
      // pto: %kv_chunk_v2_inline1873__tile
      ;
      Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v203 = Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v26, v25);
      // pto: %kv_chunk_v2_inline1873__tile
      ;
      uint64_t v204 = (uint64_t) v17;
      TASSIGN(v203, v204);
      pipe_barrier(PIPE_V);
      RoundMode v205 = RoundMode::CAST_ROUND;
      SaturationMode v206 = SaturationMode::OFF;
      TCVT(v203, v199, v205, v206);
      // pto: %gamma_kv_cast_inline1937__tile
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v207 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v24, v25);
      // pto: %gamma_kv_cast_inline1937__tile
      ;
      uint64_t v208 = (uint64_t) v18;
      TASSIGN(v207, v208);
      pipe_barrier(PIPE_V);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
      RoundMode v209 = RoundMode::CAST_ROUND;
      SaturationMode v210 = SaturationMode::OFF;
      TCVT(v207, v175, v209, v210);
      // pto: %gamma_kv_chunk_inline1872__tile
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v211 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v24, v25);
      // pto: %gamma_kv_chunk_inline1872__tile
      ;
      uint64_t v212 = (uint64_t) v18;
      TASSIGN(v211, v212);
      // pto: %15
      ;
      Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v213 = Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v26, v25);
      // pto: %15
      ;
      uint64_t v214 = (uint64_t) v17;
      TASSIGN(v213, v214);
      TROWEXPANDMUL(v213, v203, v162);
      // pto: %kv_normed_inline1951__tile
      ;
      Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v215 = Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v26, v25);
      // pto: %kv_normed_inline1951__tile
      ;
      uint64_t v216 = (uint64_t) v17;
      TASSIGN(v215, v216);
      pipe_barrier(PIPE_V);
      TCOLEXPANDMUL(v215, v213, v211);
      // pto: %kv_normed_bf16_inline1871__tile
      ;
      Tile<TileType::Vec, bfloat16_t, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v217 = Tile<TileType::Vec, bfloat16_t, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v26, v25);
      // pto: %kv_normed_bf16_inline1871__tile
      ;
      uint64_t v218 = (uint64_t) v17;
      TASSIGN(v217, v218);
      pipe_barrier(PIPE_V);
      RoundMode v219 = RoundMode::CAST_RINT;
      SaturationMode v220 = SaturationMode::OFF;
      TCVT(v217, v215, v219, v220);
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      // pto: %85
      ;
      int64_t v221 = v98 < v28 ? v28 : v98;
      // pto: %kv_view_inline1911__iter_v1_pview
      ;
      __gm__ bfloat16_t* v222 = PTOAS__GLOBAL_TENSOR_DATA(v57);
      // pto: %kv_view_inline1911__iter_v1_pview
      ;
      const int64_t v223 = 0;
      // pto: %kv_view_inline1911__iter_v1_pview
      ;
      const int64_t v224 = 512;
      // pto: %kv_view_inline1911__iter_v1_pview
      ;
      pto::Shape<1, 1, 1, 32, 64> v225 = pto::Shape<1, 1, 1, 32, 64>();
      // pto: %kv_view_inline1911__iter_v1_pview
      ;
      pto::Stride<16384, 16384, 16384, 512, 1> v226 = pto::Stride<16384, 16384, 16384, 512, 1>();
      // pto: %kv_view_inline1911__iter_v1_pview
      ;
      GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 32, 64>, pto::Stride<16384, 16384, 16384, 512, 1>, pto::Layout::ND> v227 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 32, 64>, pto::Stride<16384, 16384, 16384, 512, 1>, pto::Layout::ND>(v222 + (v223 + v221 * v224 + v168), v225, v226);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      pipe_barrier(PIPE_MTE3);
      TSTORE(v227, v217);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
      // pto: %16
      ;
      Tile<TileType::Vec, bfloat16_t, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v228 = Tile<TileType::Vec, bfloat16_t, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v26, v25);
      // pto: %16
      ;
      uint64_t v229 = (uint64_t) v19;
      TASSIGN(v228, v229);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);
      RoundMode v230 = RoundMode::CAST_RINT;
      SaturationMode v231 = SaturationMode::OFF;
      TCVT(v228, v182, v230, v231);
      // pto: %17
      ;
      Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v232 = Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v26, v25);
      // pto: %17
      ;
      uint64_t v233 = (uint64_t) v16;
      TASSIGN(v232, v233);
      pipe_barrier(PIPE_V);
      RoundMode v234 = RoundMode::CAST_ROUND;
      SaturationMode v235 = SaturationMode::OFF;
      TCVT(v232, v228, v234, v235);
      // pto: %18
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v236 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v24, v25);
      // pto: %18
      ;
      uint64_t v237 = (uint64_t) v19;
      TASSIGN(v236, v237);
      pipe_barrier(PIPE_V);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID5);
      RoundMode v238 = RoundMode::CAST_ROUND;
      SaturationMode v239 = SaturationMode::OFF;
      TCVT(v236, v192, v238, v239);
      // pto: %19
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v240 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v24, v25);
      // pto: %19
      ;
      uint64_t v241 = (uint64_t) v19;
      TASSIGN(v240, v241);
      // pto: %20
      ;
      Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v242 = Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v26, v25);
      // pto: %20
      ;
      uint64_t v243 = (uint64_t) v16;
      TASSIGN(v242, v243);
      TROWEXPANDMUL(v242, v232, v162);
      // pto: %21
      ;
      Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v244 = Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v26, v25);
      // pto: %21
      ;
      uint64_t v245 = (uint64_t) v16;
      TASSIGN(v244, v245);
      pipe_barrier(PIPE_V);
      TCOLEXPANDMUL(v244, v242, v240);
      // pto: %22
      ;
      Tile<TileType::Vec, bfloat16_t, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v246 = Tile<TileType::Vec, bfloat16_t, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v26, v25);
      // pto: %22
      ;
      uint64_t v247 = (uint64_t) v16;
      TASSIGN(v246, v247);
      pipe_barrier(PIPE_V);
      RoundMode v248 = RoundMode::CAST_RINT;
      SaturationMode v249 = SaturationMode::OFF;
      TCVT(v246, v244, v248, v249);
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
      // pto: %kv_view_inline1911__tile_pview
      ;
      __gm__ bfloat16_t* v250 = PTOAS__GLOBAL_TENSOR_DATA(v57);
      // pto: %kv_view_inline1911__tile_pview
      ;
      const int64_t v251 = 0;
      // pto: %kv_view_inline1911__tile_pview
      ;
      const int64_t v252 = 512;
      // pto: %kv_view_inline1911__tile_pview
      ;
      pto::Shape<1, 1, 1, 32, 64> v253 = pto::Shape<1, 1, 1, 32, 64>();
      // pto: %kv_view_inline1911__tile_pview
      ;
      pto::Stride<16384, 16384, 16384, 512, 1> v254 = pto::Stride<16384, 16384, 16384, 512, 1>();
      // pto: %kv_view_inline1911__tile_pview
      ;
      GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 32, 64>, pto::Stride<16384, 16384, 16384, 512, 1>, pto::Layout::ND> v255 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 32, 64>, pto::Stride<16384, 16384, 16384, 512, 1>, pto::Layout::ND>(v250 + (v251 + v221 * v252 + v185), v253, v254);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
      pipe_barrier(PIPE_MTE3);
      TSTORE(v255, v246);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    };
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
    // pto: %23
    ;
    Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v256 = Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v26, v25);
    // pto: %23
    ;
    uint64_t v257 = (uint64_t) v17;
    TASSIGN(v256, v257);
    // pto: %91
    ;
    int64_t v258 = v95 < v28 ? v28 : v95;
    // pto: %92
    ;
    __gm__ float* v259 = PTOAS__GLOBAL_TENSOR_DATA(v48);
    // pto: %92
    ;
    const int64_t v260 = 384;
    // pto: %92
    ;
    const int64_t v261 = 512;
    // pto: %92
    ;
    pto::Shape<1, 1, 1, 32, 64> v262 = pto::Shape<1, 1, 1, 32, 64>();
    // pto: %92
    ;
    pto::Stride<16384, 16384, 16384, 512, 1> v263 = pto::Stride<16384, 16384, 16384, 512, 1>();
    // pto: %92
    ;
    GlobalTensor<float, pto::Shape<1, 1, 1, 32, 64>, pto::Stride<16384, 16384, 16384, 512, 1>, pto::Layout::ND> v264 = GlobalTensor<float, pto::Shape<1, 1, 1, 32, 64>, pto::Stride<16384, 16384, 16384, 512, 1>, pto::Layout::ND>(v259 + (v260 + v258 * v261), v262, v263);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
    TLOAD(v256, v264);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID6);
    // pto: %24
    ;
    Tile<TileType::Vec, bfloat16_t, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v265 = Tile<TileType::Vec, bfloat16_t, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v26, v25);
    // pto: %24
    ;
    uint64_t v266 = (uint64_t) v16;
    TASSIGN(v265, v266);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID6);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    RoundMode v267 = RoundMode::CAST_RINT;
    SaturationMode v268 = SaturationMode::OFF;
    TCVT(v265, v256, v267, v268);
    // pto: %25
    ;
    Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v269 = Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v26, v25);
    // pto: %25
    ;
    uint64_t v270 = (uint64_t) v17;
    TASSIGN(v269, v270);
    pipe_barrier(PIPE_V);
    RoundMode v271 = RoundMode::CAST_ROUND;
    SaturationMode v272 = SaturationMode::OFF;
    TCVT(v269, v265, v271, v272);
    // pto: %26
    ;
    Tile<TileType::Vec, bfloat16_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v273 = Tile<TileType::Vec, bfloat16_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v24, v25);
    // pto: %26
    ;
    uint64_t v274 = (uint64_t) v18;
    TASSIGN(v273, v274);
    // pto: %93
    ;
    __gm__ bfloat16_t* v275 = PTOAS__GLOBAL_TENSOR_DATA(v67);
    // pto: %93
    ;
    unsigned v276 = 384;
    // pto: %93
    ;
    pto::Shape<1, 1, 1, 1, 64> v277 = pto::Shape<1, 1, 1, 1, 64>();
    // pto: %93
    ;
    pto::Stride<64, 64, 64, 64, 1> v278 = pto::Stride<64, 64, 64, 64, 1>();
    // pto: %93
    ;
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<64, 64, 64, 64, 1>, pto::Layout::ND> v279 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<64, 64, 64, 64, 1>, pto::Layout::ND>(v275 + v276, v277, v278);
    TLOAD(v273, v279);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID7);
    // pto: %27
    ;
    Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v280 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v24, v25);
    // pto: %27
    ;
    uint64_t v281 = (uint64_t) v16;
    TASSIGN(v280, v281);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID7);
    pipe_barrier(PIPE_V);
    RoundMode v282 = RoundMode::CAST_ROUND;
    SaturationMode v283 = SaturationMode::OFF;
    TCVT(v280, v273, v282, v283);
    // pto: %28
    ;
    Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v284 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v24, v25);
    // pto: %28
    ;
    uint64_t v285 = (uint64_t) v16;
    TASSIGN(v284, v285);
    // pto: %29
    ;
    Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v286 = Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v26, v25);
    // pto: %29
    ;
    uint64_t v287 = (uint64_t) v17;
    TASSIGN(v286, v287);
    TROWEXPANDMUL(v286, v269, v162);
    // pto: %30
    ;
    Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v288 = Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v26, v25);
    // pto: %30
    ;
    uint64_t v289 = (uint64_t) v17;
    TASSIGN(v288, v289);
    pipe_barrier(PIPE_V);
    TCOLEXPANDMUL(v288, v286, v284);
    // pto: %31
    ;
    Tile<TileType::Vec, bfloat16_t, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v290 = Tile<TileType::Vec, bfloat16_t, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v26, v25);
    // pto: %31
    ;
    uint64_t v291 = (uint64_t) v17;
    TASSIGN(v290, v291);
    pipe_barrier(PIPE_V);
    RoundMode v292 = RoundMode::CAST_RINT;
    SaturationMode v293 = SaturationMode::OFF;
    TCVT(v290, v288, v292, v293);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
    // pto: %95
    ;
    int64_t v294 = v98 < v28 ? v28 : v98;
    // pto: %kv_view_inline1911__rv_v2_main_pview
    ;
    __gm__ bfloat16_t* v295 = PTOAS__GLOBAL_TENSOR_DATA(v57);
    // pto: %kv_view_inline1911__rv_v2_main_pview
    ;
    const int64_t v296 = 384;
    // pto: %kv_view_inline1911__rv_v2_main_pview
    ;
    const int64_t v297 = 512;
    // pto: %kv_view_inline1911__rv_v2_main_pview
    ;
    pto::Shape<1, 1, 1, 32, 64> v298 = pto::Shape<1, 1, 1, 32, 64>();
    // pto: %kv_view_inline1911__rv_v2_main_pview
    ;
    pto::Stride<16384, 16384, 16384, 512, 1> v299 = pto::Stride<16384, 16384, 16384, 512, 1>();
    // pto: %kv_view_inline1911__rv_v2_main_pview
    ;
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 32, 64>, pto::Stride<16384, 16384, 16384, 512, 1>, pto::Layout::ND> v300 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 32, 64>, pto::Stride<16384, 16384, 16384, 512, 1>, pto::Layout::ND>(v295 + (v296 + v294 * v297), v298, v299);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
    TSTORE(v300, v290);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
    // pto: %32
    ;
    Tile<TileType::Vec, bfloat16_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v301 = Tile<TileType::Vec, bfloat16_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v24, v25);
    // pto: %32
    ;
    uint64_t v302 = (uint64_t) v17;
    TASSIGN(v301, v302);
    // pto: %96
    ;
    __gm__ bfloat16_t* v303 = PTOAS__GLOBAL_TENSOR_DATA(v67);
    // pto: %96
    ;
    unsigned v304 = 448;
    // pto: %96
    ;
    pto::Shape<1, 1, 1, 1, 64> v305 = pto::Shape<1, 1, 1, 1, 64>();
    // pto: %96
    ;
    pto::Stride<64, 64, 64, 64, 1> v306 = pto::Stride<64, 64, 64, 64, 1>();
    // pto: %96
    ;
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<64, 64, 64, 64, 1>, pto::Layout::ND> v307 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<64, 64, 64, 64, 1>, pto::Layout::ND>(v303 + v304, v305, v306);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
    TLOAD(v301, v307);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    // pto: %gamma_rope_cast_inline1869__tile
    ;
    Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v308 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v24, v25);
    // pto: %gamma_rope_cast_inline1869__tile
    ;
    uint64_t v309 = (uint64_t) v18;
    TASSIGN(v308, v309);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    RoundMode v310 = RoundMode::CAST_ROUND;
    SaturationMode v311 = SaturationMode::OFF;
    TCVT(v308, v301, v310, v311);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
    // pto: %gamma_rope_inline1868__tile
    ;
    Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v312 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v24, v25);
    // pto: %gamma_rope_inline1868__tile
    ;
    uint64_t v313 = (uint64_t) v18;
    TASSIGN(v312, v313);
    // pto: %kv_rope_chunk_inline1940__tile
    ;
    Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v314 = Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v26, v25);
    // pto: %kv_rope_chunk_inline1940__tile
    ;
    uint64_t v315 = (uint64_t) v17;
    TASSIGN(v314, v315);
    // pto: %98
    ;
    __gm__ float* v316 = PTOAS__GLOBAL_TENSOR_DATA(v48);
    // pto: %98
    ;
    const int64_t v317 = 448;
    // pto: %98
    ;
    const int64_t v318 = 512;
    // pto: %98
    ;
    pto::Shape<1, 1, 1, 32, 64> v319 = pto::Shape<1, 1, 1, 32, 64>();
    // pto: %98
    ;
    pto::Stride<16384, 16384, 16384, 512, 1> v320 = pto::Stride<16384, 16384, 16384, 512, 1>();
    // pto: %98
    ;
    GlobalTensor<float, pto::Shape<1, 1, 1, 32, 64>, pto::Stride<16384, 16384, 16384, 512, 1>, pto::Layout::ND> v321 = GlobalTensor<float, pto::Shape<1, 1, 1, 32, 64>, pto::Stride<16384, 16384, 16384, 512, 1>, pto::Layout::ND>(v316 + (v317 + v258 * v318), v319, v320);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
    TLOAD(v314, v321);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    // pto: %33
    ;
    Tile<TileType::Vec, bfloat16_t, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v322 = Tile<TileType::Vec, bfloat16_t, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v26, v25);
    // pto: %33
    ;
    uint64_t v323 = (uint64_t) v16;
    TASSIGN(v322, v323);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    RoundMode v324 = RoundMode::CAST_RINT;
    SaturationMode v325 = SaturationMode::OFF;
    TCVT(v322, v314, v324, v325);
    // pto: %kv_rope_chunk_v1_inline1866__tile
    ;
    Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v326 = Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v26, v25);
    // pto: %kv_rope_chunk_v1_inline1866__tile
    ;
    uint64_t v327 = (uint64_t) v17;
    TASSIGN(v326, v327);
    pipe_barrier(PIPE_V);
    RoundMode v328 = RoundMode::CAST_ROUND;
    SaturationMode v329 = SaturationMode::OFF;
    TCVT(v326, v322, v328, v329);
    // pto: %34
    ;
    Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v330 = Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v26, v25);
    // pto: %34
    ;
    uint64_t v331 = (uint64_t) v17;
    TASSIGN(v330, v331);
    pipe_barrier(PIPE_V);
    TROWEXPANDMUL(v330, v326, v162);
    // pto: %kv_rope_norm_chunk_inline1936__tile
    ;
    Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v332 = Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v26, v25);
    // pto: %kv_rope_norm_chunk_inline1936__tile
    ;
    uint64_t v333 = (uint64_t) v17;
    TASSIGN(v332, v333);
    pipe_barrier(PIPE_V);
    TCOLEXPANDMUL(v332, v330, v312);
    // pto: %35
    ;
    Tile<TileType::Vec, bfloat16_t, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v334 = Tile<TileType::Vec, bfloat16_t, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v26, v25);
    // pto: %35
    ;
    uint64_t v335 = (uint64_t) v16;
    TASSIGN(v334, v335);
    pipe_barrier(PIPE_V);
    RoundMode v336 = RoundMode::CAST_RINT;
    SaturationMode v337 = SaturationMode::OFF;
    TCVT(v334, v332, v336, v337);
    // pto: %kv_rope_norm_chunk_v1_inline1865__tile
    ;
    Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v338 = Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v26, v25);
    // pto: %kv_rope_norm_chunk_v1_inline1865__tile
    ;
    uint64_t v339 = (uint64_t) v17;
    TASSIGN(v338, v339);
    pipe_barrier(PIPE_V);
    RoundMode v340 = RoundMode::CAST_ROUND;
    SaturationMode v341 = SaturationMode::OFF;
    TCVT(v338, v334, v340, v341);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
    // pto: %kv_cos_il_full_inline1863__tile
    ;
    Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v342 = Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v26, v25);
    // pto: %kv_cos_il_full_inline1863__tile
    ;
    uint64_t v343 = (uint64_t) v16;
    TASSIGN(v342, v343);
    // pto: %freqs_cos__ssa_v0_pview
    ;
    __gm__ float* v344 = PTOAS__GLOBAL_TENSOR_DATA(v76);
    // pto: %freqs_cos__ssa_v0_pview
    ;
    const int64_t v345 = 0;
    // pto: %freqs_cos__ssa_v0_pview
    ;
    const int64_t v346 = 64;
    // pto: %freqs_cos__ssa_v0_pview
    ;
    pto::Shape<1, 1, 1, 32, 64> v347 = pto::Shape<1, 1, 1, 32, 64>();
    // pto: %freqs_cos__ssa_v0_pview
    ;
    pto::Stride<2048, 2048, 2048, 64, 1> v348 = pto::Stride<2048, 2048, 2048, 64, 1>();
    // pto: %freqs_cos__ssa_v0_pview
    ;
    GlobalTensor<float, pto::Shape<1, 1, 1, 32, 64>, pto::Stride<2048, 2048, 2048, 64, 1>, pto::Layout::ND> v349 = GlobalTensor<float, pto::Shape<1, 1, 1, 32, 64>, pto::Stride<2048, 2048, 2048, 64, 1>, pto::Layout::ND>(v344 + (v345 + v294 * v346), v347, v348);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
    TLOAD(v342, v349);
    // pto: %kv_sin_signed_full_inline1930__tile
    ;
    Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v350 = Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v26, v25);
    // pto: %kv_sin_signed_full_inline1930__tile
    ;
    uint64_t v351 = (uint64_t) v18;
    TASSIGN(v350, v351);
    // pto: %q_rope_sin_signed_inline197__ssa_v0_pview
    ;
    __gm__ float* v352 = PTOAS__GLOBAL_TENSOR_DATA(v85);
    // pto: %q_rope_sin_signed_inline197__ssa_v0_pview
    ;
    const int64_t v353 = 0;
    // pto: %q_rope_sin_signed_inline197__ssa_v0_pview
    ;
    const int64_t v354 = 64;
    // pto: %q_rope_sin_signed_inline197__ssa_v0_pview
    ;
    pto::Shape<1, 1, 1, 32, 64> v355 = pto::Shape<1, 1, 1, 32, 64>();
    // pto: %q_rope_sin_signed_inline197__ssa_v0_pview
    ;
    pto::Stride<2048, 2048, 2048, 64, 1> v356 = pto::Stride<2048, 2048, 2048, 64, 1>();
    // pto: %q_rope_sin_signed_inline197__ssa_v0_pview
    ;
    GlobalTensor<float, pto::Shape<1, 1, 1, 32, 64>, pto::Stride<2048, 2048, 2048, 64, 1>, pto::Layout::ND> v357 = GlobalTensor<float, pto::Shape<1, 1, 1, 32, 64>, pto::Stride<2048, 2048, 2048, 64, 1>, pto::Layout::ND>(v352 + (v353 + v294 * v354), v355, v356);
    TLOAD(v350, v357);
    // pto: %kv_swap_idx_full_inline1862__tile
    ;
    Tile<TileType::Vec, int32_t, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v358 = Tile<TileType::Vec, int32_t, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v26, v25);
    // pto: %kv_swap_idx_full_inline1862__tile
    ;
    uint64_t v359 = (uint64_t) v19;
    TASSIGN(v358, v359);
    // pto: %q_rope_swap_idx_inline196__ssa_v0_pview
    ;
    __gm__ int32_t* v360 = PTOAS__GLOBAL_TENSOR_DATA(v94);
    // pto: %q_rope_swap_idx_inline196__ssa_v0_pview
    ;
    const int64_t v361 = 0;
    // pto: %q_rope_swap_idx_inline196__ssa_v0_pview
    ;
    const int64_t v362 = 64;
    // pto: %q_rope_swap_idx_inline196__ssa_v0_pview
    ;
    pto::Shape<1, 1, 1, 32, 64> v363 = pto::Shape<1, 1, 1, 32, 64>();
    // pto: %q_rope_swap_idx_inline196__ssa_v0_pview
    ;
    pto::Stride<2048, 2048, 2048, 64, 1> v364 = pto::Stride<2048, 2048, 2048, 64, 1>();
    // pto: %q_rope_swap_idx_inline196__ssa_v0_pview
    ;
    GlobalTensor<int32_t, pto::Shape<1, 1, 1, 32, 64>, pto::Stride<2048, 2048, 2048, 64, 1>, pto::Layout::ND> v365 = GlobalTensor<int32_t, pto::Shape<1, 1, 1, 32, 64>, pto::Stride<2048, 2048, 2048, 64, 1>, pto::Layout::ND>(v360 + (v361 + v294 * v362), v363, v364);
    TLOAD(v358, v365);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    // pto: %gather_acc_init
    ;
    Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v366 = Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v26, v25);
    // pto: %gather_acc_init
    ;
    uint64_t v367 = (uint64_t) v20;
    TASSIGN(v366, v367);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    for (int64_t v368 = v28; v368 < v26; v368 += v24) {
      // pto: %gather_inp_row
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v369 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v24, v25);
      // pto: %gather_inp_row
      ;
      uint64_t v370 = (uint64_t) v17;
      TASSIGN(v369, v370);
      // pto: %slice_view
      ;
      int64_t v371 = (int64_t) ((uint64_t) v368 * (uint64_t) v15);
      // pto: %slice_view
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v372;
      // pto: %slice_view
      ;
      uint64_t v373 = (uint64_t) ((int64_t) (uint64_t) v371 + (uint64_t) v17);
      TASSIGN(v372, v373);
      // pto: %gather_idx_row
      ;
      Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v374 = Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v24, v25);
      // pto: %gather_idx_row
      ;
      uint64_t v375 = (uint64_t) v19;
      TASSIGN(v374, v375);
      // pto: %102
      ;
      Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v376;
      // pto: %102
      ;
      uint64_t v377 = (uint64_t) ((int64_t) (uint64_t) v371 + (uint64_t) v19);
      TASSIGN(v376, v377);
      // pto: %gather_row_tmp
      ;
      Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v378 = Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v24, v25);
      // pto: %gather_row_tmp
      ;
      uint64_t v379 = (uint64_t) v21;
      TASSIGN(v378, v379);
      // pto: %gather_row
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v380 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v24, v25);
      // pto: %gather_row
      ;
      uint64_t v381 = (uint64_t) v22;
      TASSIGN(v380, v381);
      pipe_barrier(PIPE_V);
      TGATHER(v380, v372, v376, v378);
      // pto: %assemble_view
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v382;
      // pto: %assemble_view
      ;
      uint64_t v383 = (uint64_t) ((int64_t) (uint64_t) v371 + (uint64_t) v20);
      TASSIGN(v382, v383);
      pipe_barrier(PIPE_V);
      TMOV(v382, v380);
    };
    // pto: %kv_swapped_full_inline1860__tile
    ;
    Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v384 = Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v26, v25);
    // pto: %kv_swapped_full_inline1860__tile
    ;
    uint64_t v385 = (uint64_t) v20;
    TASSIGN(v384, v385);
    // pto: %36
    ;
    Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v386 = Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v26, v25);
    // pto: %36
    ;
    uint64_t v387 = (uint64_t) v17;
    TASSIGN(v386, v387);
    TMUL(v386, v338, v342);
    // pto: %37
    ;
    Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v388 = Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v26, v25);
    // pto: %37
    ;
    uint64_t v389 = (uint64_t) v16;
    TASSIGN(v388, v389);
    pipe_barrier(PIPE_V);
    TMUL(v388, v384, v350);
    // pto: %kv_rope_rot_full_inline1859__tile
    ;
    Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v390 = Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v26, v25);
    // pto: %kv_rope_rot_full_inline1859__tile
    ;
    uint64_t v391 = (uint64_t) v17;
    TASSIGN(v390, v391);
    pipe_barrier(PIPE_V);
    TADD(v390, v386, v388);
    // pto: %kv_rope_i16_full_inline1861__tile
    ;
    Tile<TileType::Vec, bfloat16_t, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v392 = Tile<TileType::Vec, bfloat16_t, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v26, v25);
    // pto: %kv_rope_i16_full_inline1861__tile
    ;
    uint64_t v393 = (uint64_t) v17;
    TASSIGN(v392, v393);
    pipe_barrier(PIPE_V);
    RoundMode v394 = RoundMode::CAST_RINT;
    SaturationMode v395 = SaturationMode::OFF;
    TCVT(v392, v390, v394, v395);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID3);
    // pto: %kv_view_inline1911__rv_v2_pview
    ;
    __gm__ bfloat16_t* v396 = PTOAS__GLOBAL_TENSOR_DATA(v57);
    // pto: %kv_view_inline1911__rv_v2_pview
    ;
    const int64_t v397 = 448;
    // pto: %kv_view_inline1911__rv_v2_pview
    ;
    const int64_t v398 = 512;
    // pto: %kv_view_inline1911__rv_v2_pview
    ;
    pto::Shape<1, 1, 1, 32, 64> v399 = pto::Shape<1, 1, 1, 32, 64>();
    // pto: %kv_view_inline1911__rv_v2_pview
    ;
    pto::Stride<16384, 16384, 16384, 512, 1> v400 = pto::Stride<16384, 16384, 16384, 512, 1>();
    // pto: %kv_view_inline1911__rv_v2_pview
    ;
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 32, 64>, pto::Stride<16384, 16384, 16384, 512, 1>, pto::Layout::ND> v401 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 32, 64>, pto::Stride<16384, 16384, 16384, 512, 1>, pto::Layout::ND>(v396 + (v397 + v294 * v398), v399, v400);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID3);
    TSTORE(v401, v392);
  } else {
    // pto: %kv_reduce_tmp_inline1857__ssa_v0
    ;
    Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v402 = Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v26, v25);
    // pto: %kv_reduce_tmp_inline1857__ssa_v0
    ;
    uint64_t v403 = (uint64_t) v17;
    TASSIGN(v402, v403);
    // pto: %kv_sq_lanes_tail_inline1856__ssa_v0
    ;
    Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v404 = Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v26, v25);
    // pto: %kv_sq_lanes_tail_inline1856__ssa_v0
    ;
    uint64_t v405 = (uint64_t) v16;
    TASSIGN(v404, v405);
    TEXPANDS(v404, v27);
    for (int64_t v406 = v28; v406 < v23; v406 += v29) {
      // pto: %kv_chunk_tail_inline1853__ssa_v0
      ;
      Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v407 = Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v97, v25);
      // pto: %kv_chunk_tail_inline1853__ssa_v0
      ;
      uint64_t v408 = (uint64_t) v18;
      TASSIGN(v407, v408);
      // pto: %105
      ;
      int64_t v409 = v95 < v28 ? v28 : v95;
      // pto: %106
      ;
      int64_t v410 = v406 < v28 ? v28 : v406;
      // pto: %107
      ;
      const int64_t v411 = 0;
      // pto: %107
      ;
      __gm__ float* v412 = PTOAS__GLOBAL_TENSOR_DATA(v48);
      // pto: %107
      ;
      const int64_t v413 = 1;
      // pto: %107
      ;
      const int64_t v414 = 1;
      // pto: %107
      ;
      const int64_t v415 = 1;
      // pto: %107
      ;
      int64_t v416 = v97 * v23;
      // pto: %107
      ;
      int64_t v417 = v415 * v416;
      // pto: %107
      ;
      pto::Shape<1, 1, 1, -1, 64> v418 = pto::Shape<1, 1, 1, -1, 64>(v413, v414, v415, v97, v25);
      // pto: %107
      ;
      pto::Stride<-1, -1, -1, -1, -1> v419 = pto::Stride<-1, -1, -1, -1, -1>(v414 * v417, v417, v416, v23, v24);
      // pto: %107
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, -1, 64>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v420 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, 64>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v412 + (v411 + v409 * v23 + v410 * v24), v418, v419);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID4);
      TLOAD(v407, v420);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      // pto: %38
      ;
      Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v421 = Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v97, v25);
      // pto: %38
      ;
      uint64_t v422 = (uint64_t) v19;
      TASSIGN(v421, v422);
      // pto: %109
      ;
      int64_t v423 = (int64_t) ((uint64_t) v406 + (uint64_t) v25);
      // pto: %110
      ;
      int64_t v424 = v423 < v28 ? v28 : v423;
      // pto: %111
      ;
      const int64_t v425 = 0;
      // pto: %111
      ;
      __gm__ float* v426 = PTOAS__GLOBAL_TENSOR_DATA(v48);
      // pto: %111
      ;
      const int64_t v427 = 1;
      // pto: %111
      ;
      const int64_t v428 = 1;
      // pto: %111
      ;
      const int64_t v429 = 1;
      // pto: %111
      ;
      int64_t v430 = v97 * v23;
      // pto: %111
      ;
      int64_t v431 = v429 * v430;
      // pto: %111
      ;
      pto::Shape<1, 1, 1, -1, 64> v432 = pto::Shape<1, 1, 1, -1, 64>(v427, v428, v429, v97, v25);
      // pto: %111
      ;
      pto::Stride<-1, -1, -1, -1, -1> v433 = pto::Stride<-1, -1, -1, -1, -1>(v428 * v431, v431, v430, v23, v24);
      // pto: %111
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, -1, 64>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v434 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, 64>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v426 + (v425 + v409 * v23 + v424 * v24), v432, v433);
      TLOAD(v421, v434);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
      // pto: %t__tmp_v88
      ;
      Tile<TileType::Vec, bfloat16_t, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v435 = Tile<TileType::Vec, bfloat16_t, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v97, v25);
      // pto: %t__tmp_v88
      ;
      uint64_t v436 = (uint64_t) v20;
      TASSIGN(v435, v436);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      RoundMode v437 = RoundMode::CAST_RINT;
      SaturationMode v438 = SaturationMode::OFF;
      TCVT(v435, v407, v437, v438);
      // pto: %kv_chunk_tail_v1_inline1849__ssa_v0
      ;
      Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v439 = Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v97, v25);
      // pto: %kv_chunk_tail_v1_inline1849__ssa_v0
      ;
      uint64_t v440 = (uint64_t) v18;
      TASSIGN(v439, v440);
      pipe_barrier(PIPE_V);
      RoundMode v441 = RoundMode::CAST_ROUND;
      SaturationMode v442 = SaturationMode::OFF;
      TCVT(v439, v435, v441, v442);
      // pto: %kv_sq_tail_inline1848__ssa_v0
      ;
      Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v443 = Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v97, v25);
      // pto: %kv_sq_tail_inline1848__ssa_v0
      ;
      uint64_t v444 = (uint64_t) v18;
      TASSIGN(v443, v444);
      pipe_barrier(PIPE_V);
      TMUL(v443, v439, v439);
      // pto: %kv_sq_lanes_tail_inline1856__ssa_v3
      ;
      Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v445 = Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v26, v25);
      // pto: %kv_sq_lanes_tail_inline1856__ssa_v3
      ;
      uint64_t v446 = (uint64_t) v18;
      TASSIGN(v445, v446);
      pipe_barrier(PIPE_V);
      TADD(v445, v404, v443);
      // pto: %39
      ;
      Tile<TileType::Vec, bfloat16_t, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v447 = Tile<TileType::Vec, bfloat16_t, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v97, v25);
      // pto: %39
      ;
      uint64_t v448 = (uint64_t) v21;
      TASSIGN(v447, v448);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
      RoundMode v449 = RoundMode::CAST_RINT;
      SaturationMode v450 = SaturationMode::OFF;
      TCVT(v447, v421, v449, v450);
      // pto: %40
      ;
      Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v451 = Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v97, v25);
      // pto: %40
      ;
      uint64_t v452 = (uint64_t) v19;
      TASSIGN(v451, v452);
      pipe_barrier(PIPE_V);
      RoundMode v453 = RoundMode::CAST_ROUND;
      SaturationMode v454 = SaturationMode::OFF;
      TCVT(v451, v447, v453, v454);
      // pto: %41
      ;
      Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v455 = Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v97, v25);
      // pto: %41
      ;
      uint64_t v456 = (uint64_t) v19;
      TASSIGN(v455, v456);
      pipe_barrier(PIPE_V);
      TMUL(v455, v451, v451);
      // pto: %42
      ;
      Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v457 = Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v26, v25);
      // pto: %42
      ;
      uint64_t v458 = (uint64_t) v16;
      TASSIGN(v457, v458);
      pipe_barrier(PIPE_V);
      TADD(v457, v445, v455);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID4);
    };
    // pto: %t__tmp_v89
    ;
    Tile<TileType::Vec, float, 32, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v459 = Tile<TileType::Vec, float, 32, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v26, v24);
    // pto: %t__tmp_v89
    ;
    uint64_t v460 = (uint64_t) v18;
    TASSIGN(v459, v460);
    pipe_barrier(PIPE_V);
    TROWSUM(v459, v404, v402);
    // pto: %kv_sq_sum_tail_inline1846__ssa_v0
    ;
    Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v461 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v24, v26);
    // pto: %kv_sq_sum_tail_inline1846__ssa_v0
    ;
    uint64_t v462 = (uint64_t) v18;
    TASSIGN(v461, v462);
    // pto: %t__tmp_v90
    ;
    Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v463 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v24, v26);
    // pto: %t__tmp_v90
    ;
    uint64_t v464 = (uint64_t) v17;
    TASSIGN(v463, v464);
    pipe_barrier(PIPE_V);
    TMULS(v463, v461, v30);
    // pto: %t__tmp_v91
    ;
    Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v465 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v24, v26);
    // pto: %t__tmp_v91
    ;
    uint64_t v466 = (uint64_t) v17;
    TASSIGN(v465, v466);
    pipe_barrier(PIPE_V);
    TADDS(v465, v463, v31);
    // pto: %kv_rms_tail_inline1870__ssa_v0
    ;
    Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v467 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v24, v26);
    // pto: %kv_rms_tail_inline1870__ssa_v0
    ;
    uint64_t v468 = (uint64_t) v17;
    TASSIGN(v467, v468);
    pipe_barrier(PIPE_V);
    TSQRT(v467, v465);
    // pto: %t__tmp_v92
    ;
    Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v469 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v24, v26);
    // pto: %t__tmp_v92
    ;
    uint64_t v470 = (uint64_t) v16;
    TASSIGN(v469, v470);
    TEXPANDS(v469, v32);
    // pto: %kv_inv_rms_tail_inline1845__ssa_v0
    ;
    Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v471 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v24, v26);
    // pto: %kv_inv_rms_tail_inline1845__ssa_v0
    ;
    uint64_t v472 = (uint64_t) v20;
    TASSIGN(v471, v472);
    pipe_barrier(PIPE_V);
    TDIV(v471, v469, v467);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID5);
    // pto: %kv_inv_rms_t_tail_inline1844__ssa_v0
    ;
    Tile<TileType::Vec, float, 32, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v473 = Tile<TileType::Vec, float, 32, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v26, v24);
    // pto: %kv_inv_rms_t_tail_inline1844__ssa_v0
    ;
    uint64_t v474 = (uint64_t) v20;
    TASSIGN(v473, v474);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID5);
    for (int64_t v475 = v28; v475 < v33; v475 += v29) {
      // pto: %kv_chunk_tail_inline1853__ssa_v1
      ;
      Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v476 = Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v97, v25);
      // pto: %kv_chunk_tail_inline1853__ssa_v1
      ;
      uint64_t v477 = (uint64_t) v17;
      TASSIGN(v476, v477);
      // pto: %112
      ;
      int64_t v478 = v95 < v28 ? v28 : v95;
      // pto: %113
      ;
      int64_t v479 = v475 < v28 ? v28 : v475;
      // pto: %114
      ;
      const int64_t v480 = 0;
      // pto: %114
      ;
      __gm__ float* v481 = PTOAS__GLOBAL_TENSOR_DATA(v48);
      // pto: %114
      ;
      const int64_t v482 = 1;
      // pto: %114
      ;
      const int64_t v483 = 1;
      // pto: %114
      ;
      const int64_t v484 = 1;
      // pto: %114
      ;
      int64_t v485 = v97 * v23;
      // pto: %114
      ;
      int64_t v486 = v484 * v485;
      // pto: %114
      ;
      pto::Shape<1, 1, 1, -1, 64> v487 = pto::Shape<1, 1, 1, -1, 64>(v482, v483, v484, v97, v25);
      // pto: %114
      ;
      pto::Stride<-1, -1, -1, -1, -1> v488 = pto::Stride<-1, -1, -1, -1, -1>(v483 * v486, v486, v485, v23, v24);
      // pto: %114
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, -1, 64>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v489 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, 64>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v481 + (v480 + v478 * v23 + v479 * v24), v487, v488);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID4);
      TLOAD(v476, v489);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      // pto: %gamma_kv_input_tail_inline1881__ssa_v0
      ;
      Tile<TileType::Vec, bfloat16_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v490 = Tile<TileType::Vec, bfloat16_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v24, v25);
      // pto: %gamma_kv_input_tail_inline1881__ssa_v0
      ;
      uint64_t v491 = (uint64_t) v21;
      TASSIGN(v490, v491);
      // pto: %116
      ;
      __gm__ bfloat16_t* v492 = PTOAS__GLOBAL_TENSOR_DATA(v67);
      // pto: %116
      ;
      const int64_t v493 = 0;
      // pto: %116
      ;
      pto::Shape<1, 1, 1, 1, 64> v494 = pto::Shape<1, 1, 1, 1, 64>();
      // pto: %116
      ;
      pto::Stride<64, 64, 64, 64, 1> v495 = pto::Stride<64, 64, 64, 64, 1>();
      // pto: %116
      ;
      GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<64, 64, 64, 64, 1>, pto::Layout::ND> v496 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<64, 64, 64, 64, 1>, pto::Layout::ND>(v492 + (v493 + v479), v494, v495);
      TLOAD(v490, v496);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
      // pto: %43
      ;
      Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v497 = Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v97, v25);
      // pto: %43
      ;
      uint64_t v498 = (uint64_t) v16;
      TASSIGN(v497, v498);
      // pto: %118
      ;
      int64_t v499 = (int64_t) ((uint64_t) v475 + (uint64_t) v25);
      // pto: %119
      ;
      int64_t v500 = v499 < v28 ? v28 : v499;
      // pto: %120
      ;
      const int64_t v501 = 0;
      // pto: %120
      ;
      __gm__ float* v502 = PTOAS__GLOBAL_TENSOR_DATA(v48);
      // pto: %120
      ;
      const int64_t v503 = 1;
      // pto: %120
      ;
      const int64_t v504 = 1;
      // pto: %120
      ;
      const int64_t v505 = 1;
      // pto: %120
      ;
      int64_t v506 = v97 * v23;
      // pto: %120
      ;
      int64_t v507 = v505 * v506;
      // pto: %120
      ;
      pto::Shape<1, 1, 1, -1, 64> v508 = pto::Shape<1, 1, 1, -1, 64>(v503, v504, v505, v97, v25);
      // pto: %120
      ;
      pto::Stride<-1, -1, -1, -1, -1> v509 = pto::Stride<-1, -1, -1, -1, -1>(v504 * v507, v507, v506, v23, v24);
      // pto: %120
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, -1, 64>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v510 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, 64>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v502 + (v501 + v478 * v23 + v500 * v24), v508, v509);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID5);
      TLOAD(v497, v510);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
      // pto: %44
      ;
      Tile<TileType::Vec, bfloat16_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v511 = Tile<TileType::Vec, bfloat16_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v24, v25);
      // pto: %44
      ;
      uint64_t v512 = (uint64_t) v22;
      TASSIGN(v511, v512);
      // pto: %123
      ;
      __gm__ bfloat16_t* v513 = PTOAS__GLOBAL_TENSOR_DATA(v67);
      // pto: %123
      ;
      const int64_t v514 = 0;
      // pto: %123
      ;
      pto::Shape<1, 1, 1, 1, 64> v515 = pto::Shape<1, 1, 1, 1, 64>();
      // pto: %123
      ;
      pto::Stride<64, 64, 64, 64, 1> v516 = pto::Stride<64, 64, 64, 64, 1>();
      // pto: %123
      ;
      GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<64, 64, 64, 64, 1>, pto::Layout::ND> v517 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<64, 64, 64, 64, 1>, pto::Layout::ND>(v513 + (v514 + v500), v515, v516);
      TLOAD(v511, v517);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
      // pto: %t__tmp_v93
      ;
      Tile<TileType::Vec, bfloat16_t, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v518 = Tile<TileType::Vec, bfloat16_t, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v97, v25);
      // pto: %t__tmp_v93
      ;
      uint64_t v519 = (uint64_t) v18;
      TASSIGN(v518, v519);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      RoundMode v520 = RoundMode::CAST_RINT;
      SaturationMode v521 = SaturationMode::OFF;
      TCVT(v518, v476, v520, v521);
      // pto: %kv_chunk_tail_v2_inline1950__ssa_v0
      ;
      Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v522 = Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v97, v25);
      // pto: %kv_chunk_tail_v2_inline1950__ssa_v0
      ;
      uint64_t v523 = (uint64_t) v17;
      TASSIGN(v522, v523);
      pipe_barrier(PIPE_V);
      RoundMode v524 = RoundMode::CAST_ROUND;
      SaturationMode v525 = SaturationMode::OFF;
      TCVT(v522, v518, v524, v525);
      // pto: %gamma_kv_cast_tail_inline1904__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v526 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v24, v25);
      // pto: %gamma_kv_cast_tail_inline1904__ssa_v0
      ;
      uint64_t v527 = (uint64_t) v18;
      TASSIGN(v526, v527);
      pipe_barrier(PIPE_V);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
      RoundMode v528 = RoundMode::CAST_ROUND;
      SaturationMode v529 = SaturationMode::OFF;
      TCVT(v526, v490, v528, v529);
      // pto: %gamma_kv_chunk_tail_inline1852__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v530 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v24, v25);
      // pto: %gamma_kv_chunk_tail_inline1852__ssa_v0
      ;
      uint64_t v531 = (uint64_t) v18;
      TASSIGN(v530, v531);
      // pto: %t__tmp_v94
      ;
      Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v532 = Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v97, v25);
      // pto: %t__tmp_v94
      ;
      uint64_t v533 = (uint64_t) v17;
      TASSIGN(v532, v533);
      TROWEXPANDMUL(v532, v522, v473);
      // pto: %kv_normed_tail_inline1843__ssa_v0
      ;
      Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v534 = Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v97, v25);
      // pto: %kv_normed_tail_inline1843__ssa_v0
      ;
      uint64_t v535 = (uint64_t) v17;
      TASSIGN(v534, v535);
      pipe_barrier(PIPE_V);
      TCOLEXPANDMUL(v534, v532, v530);
      // pto: %kv_normed_bf16_tail_inline1842__ssa_v0
      ;
      Tile<TileType::Vec, bfloat16_t, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v536 = Tile<TileType::Vec, bfloat16_t, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v97, v25);
      // pto: %kv_normed_bf16_tail_inline1842__ssa_v0
      ;
      uint64_t v537 = (uint64_t) v17;
      TASSIGN(v536, v537);
      pipe_barrier(PIPE_V);
      RoundMode v538 = RoundMode::CAST_RINT;
      SaturationMode v539 = SaturationMode::OFF;
      TCVT(v536, v534, v538, v539);
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID4);
      v536.SetValidShape(v97, v25);
      // pto: %124
      ;
      int64_t v540 = v98 < v28 ? v28 : v98;
      // pto: %kv_view_inline1911__ssa_v0_pview
      ;
      const int64_t v541 = 0;
      // pto: %kv_view_inline1911__ssa_v0_pview
      ;
      __gm__ bfloat16_t* v542 = PTOAS__GLOBAL_TENSOR_DATA(v57);
      // pto: %kv_view_inline1911__ssa_v0_pview
      ;
      const int64_t v543 = 1;
      // pto: %kv_view_inline1911__ssa_v0_pview
      ;
      const int64_t v544 = 1;
      // pto: %kv_view_inline1911__ssa_v0_pview
      ;
      const int64_t v545 = 1;
      // pto: %kv_view_inline1911__ssa_v0_pview
      ;
      int64_t v546 = v97 * v23;
      // pto: %kv_view_inline1911__ssa_v0_pview
      ;
      int64_t v547 = v545 * v546;
      // pto: %kv_view_inline1911__ssa_v0_pview
      ;
      pto::Shape<1, 1, 1, -1, 64> v548 = pto::Shape<1, 1, 1, -1, 64>(v543, v544, v545, v97, v25);
      // pto: %kv_view_inline1911__ssa_v0_pview
      ;
      pto::Stride<-1, -1, -1, -1, -1> v549 = pto::Stride<-1, -1, -1, -1, -1>(v544 * v547, v547, v546, v23, v24);
      // pto: %kv_view_inline1911__ssa_v0_pview
      ;
      GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 64>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v550 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 64>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v542 + (v541 + v540 * v23 + v479 * v24), v548, v549);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID4);
      pipe_barrier(PIPE_MTE3);
      TSTORE(v550, v536);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID4);
      // pto: %45
      ;
      Tile<TileType::Vec, bfloat16_t, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v551 = Tile<TileType::Vec, bfloat16_t, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v97, v25);
      // pto: %45
      ;
      uint64_t v552 = (uint64_t) v19;
      TASSIGN(v551, v552);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
      RoundMode v553 = RoundMode::CAST_RINT;
      SaturationMode v554 = SaturationMode::OFF;
      TCVT(v551, v497, v553, v554);
      // pto: %46
      ;
      Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v555 = Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v97, v25);
      // pto: %46
      ;
      uint64_t v556 = (uint64_t) v16;
      TASSIGN(v555, v556);
      pipe_barrier(PIPE_V);
      RoundMode v557 = RoundMode::CAST_ROUND;
      SaturationMode v558 = SaturationMode::OFF;
      TCVT(v555, v551, v557, v558);
      // pto: %47
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v559 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v24, v25);
      // pto: %47
      ;
      uint64_t v560 = (uint64_t) v19;
      TASSIGN(v559, v560);
      pipe_barrier(PIPE_V);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
      RoundMode v561 = RoundMode::CAST_ROUND;
      SaturationMode v562 = SaturationMode::OFF;
      TCVT(v559, v511, v561, v562);
      // pto: %48
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v563 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v24, v25);
      // pto: %48
      ;
      uint64_t v564 = (uint64_t) v19;
      TASSIGN(v563, v564);
      // pto: %49
      ;
      Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v565 = Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v97, v25);
      // pto: %49
      ;
      uint64_t v566 = (uint64_t) v16;
      TASSIGN(v565, v566);
      TROWEXPANDMUL(v565, v555, v473);
      // pto: %50
      ;
      Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v567 = Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v97, v25);
      // pto: %50
      ;
      uint64_t v568 = (uint64_t) v16;
      TASSIGN(v567, v568);
      pipe_barrier(PIPE_V);
      TCOLEXPANDMUL(v567, v565, v563);
      // pto: %51
      ;
      Tile<TileType::Vec, bfloat16_t, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v569 = Tile<TileType::Vec, bfloat16_t, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v97, v25);
      // pto: %51
      ;
      uint64_t v570 = (uint64_t) v16;
      TASSIGN(v569, v570);
      pipe_barrier(PIPE_V);
      RoundMode v571 = RoundMode::CAST_RINT;
      SaturationMode v572 = SaturationMode::OFF;
      TCVT(v569, v567, v571, v572);
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID5);
      v569.SetValidShape(v97, v25);
      // pto: %130
      ;
      const int64_t v573 = 0;
      // pto: %130
      ;
      __gm__ bfloat16_t* v574 = PTOAS__GLOBAL_TENSOR_DATA(v57);
      // pto: %130
      ;
      const int64_t v575 = 1;
      // pto: %130
      ;
      const int64_t v576 = 1;
      // pto: %130
      ;
      const int64_t v577 = 1;
      // pto: %130
      ;
      int64_t v578 = v97 * v23;
      // pto: %130
      ;
      int64_t v579 = v577 * v578;
      // pto: %130
      ;
      pto::Shape<1, 1, 1, -1, 64> v580 = pto::Shape<1, 1, 1, -1, 64>(v575, v576, v577, v97, v25);
      // pto: %130
      ;
      pto::Stride<-1, -1, -1, -1, -1> v581 = pto::Stride<-1, -1, -1, -1, -1>(v576 * v579, v579, v578, v23, v24);
      // pto: %130
      ;
      GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 64>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v582 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 64>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v574 + (v573 + v540 * v23 + v500 * v24), v580, v581);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID5);
      pipe_barrier(PIPE_MTE3);
      TSTORE(v582, v569);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID5);
    };
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID6);
    // pto: %53
    ;
    Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v583 = Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v97, v25);
    // pto: %53
    ;
    uint64_t v584 = (uint64_t) v17;
    TASSIGN(v583, v584);
    // pto: %131
    ;
    int64_t v585 = v95 < v28 ? v28 : v95;
    // pto: %132
    ;
    const int64_t v586 = 0;
    // pto: %132
    ;
    __gm__ float* v587 = PTOAS__GLOBAL_TENSOR_DATA(v48);
    // pto: %132
    ;
    const int64_t v588 = 1;
    // pto: %132
    ;
    const int64_t v589 = 1;
    // pto: %132
    ;
    const int64_t v590 = 1;
    // pto: %132
    ;
    int64_t v591 = v97 * v23;
    // pto: %132
    ;
    int64_t v592 = v590 * v591;
    // pto: %132
    ;
    pto::Shape<1, 1, 1, -1, 64> v593 = pto::Shape<1, 1, 1, -1, 64>(v588, v589, v590, v97, v25);
    // pto: %132
    ;
    pto::Stride<-1, -1, -1, -1, -1> v594 = pto::Stride<-1, -1, -1, -1, -1>(v589 * v592, v592, v591, v23, v24);
    // pto: %132
    ;
    GlobalTensor<float, pto::Shape<1, 1, 1, -1, 64>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v595 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, 64>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v587 + (v586 + v585 * v23 + v33 * v24), v593, v594);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID6);
    TLOAD(v583, v595);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    // pto: %54
    ;
    Tile<TileType::Vec, bfloat16_t, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v596 = Tile<TileType::Vec, bfloat16_t, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v97, v25);
    // pto: %54
    ;
    uint64_t v597 = (uint64_t) v16;
    TASSIGN(v596, v597);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
    RoundMode v598 = RoundMode::CAST_RINT;
    SaturationMode v599 = SaturationMode::OFF;
    TCVT(v596, v583, v598, v599);
    // pto: %55
    ;
    Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v600 = Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v97, v25);
    // pto: %55
    ;
    uint64_t v601 = (uint64_t) v17;
    TASSIGN(v600, v601);
    pipe_barrier(PIPE_V);
    RoundMode v602 = RoundMode::CAST_ROUND;
    SaturationMode v603 = SaturationMode::OFF;
    TCVT(v600, v596, v602, v603);
    // pto: %56
    ;
    Tile<TileType::Vec, bfloat16_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v604 = Tile<TileType::Vec, bfloat16_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v24, v25);
    // pto: %56
    ;
    uint64_t v605 = (uint64_t) v18;
    TASSIGN(v604, v605);
    // pto: %133
    ;
    __gm__ bfloat16_t* v606 = PTOAS__GLOBAL_TENSOR_DATA(v67);
    // pto: %133
    ;
    unsigned v607 = 384;
    // pto: %133
    ;
    pto::Shape<1, 1, 1, 1, 64> v608 = pto::Shape<1, 1, 1, 1, 64>();
    // pto: %133
    ;
    pto::Stride<64, 64, 64, 64, 1> v609 = pto::Stride<64, 64, 64, 64, 1>();
    // pto: %133
    ;
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<64, 64, 64, 64, 1>, pto::Layout::ND> v610 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<64, 64, 64, 64, 1>, pto::Layout::ND>(v606 + v607, v608, v609);
    TLOAD(v604, v610);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    // pto: %57
    ;
    Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v611 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v24, v25);
    // pto: %57
    ;
    uint64_t v612 = (uint64_t) v16;
    TASSIGN(v611, v612);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    pipe_barrier(PIPE_V);
    RoundMode v613 = RoundMode::CAST_ROUND;
    SaturationMode v614 = SaturationMode::OFF;
    TCVT(v611, v604, v613, v614);
    // pto: %58
    ;
    Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v615 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v24, v25);
    // pto: %58
    ;
    uint64_t v616 = (uint64_t) v16;
    TASSIGN(v615, v616);
    // pto: %59
    ;
    Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v617 = Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v97, v25);
    // pto: %59
    ;
    uint64_t v618 = (uint64_t) v17;
    TASSIGN(v617, v618);
    TROWEXPANDMUL(v617, v600, v473);
    // pto: %60
    ;
    Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v619 = Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v97, v25);
    // pto: %60
    ;
    uint64_t v620 = (uint64_t) v17;
    TASSIGN(v619, v620);
    pipe_barrier(PIPE_V);
    TCOLEXPANDMUL(v619, v617, v615);
    // pto: %61
    ;
    Tile<TileType::Vec, bfloat16_t, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v621 = Tile<TileType::Vec, bfloat16_t, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v97, v25);
    // pto: %61
    ;
    uint64_t v622 = (uint64_t) v17;
    TASSIGN(v621, v622);
    pipe_barrier(PIPE_V);
    RoundMode v623 = RoundMode::CAST_RINT;
    SaturationMode v624 = SaturationMode::OFF;
    TCVT(v621, v619, v623, v624);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID6);
    v621.SetValidShape(v97, v25);
    // pto: %135
    ;
    int64_t v625 = v98 < v28 ? v28 : v98;
    // pto: %136
    ;
    const int64_t v626 = 0;
    // pto: %136
    ;
    __gm__ bfloat16_t* v627 = PTOAS__GLOBAL_TENSOR_DATA(v57);
    // pto: %136
    ;
    const int64_t v628 = 1;
    // pto: %136
    ;
    const int64_t v629 = 1;
    // pto: %136
    ;
    const int64_t v630 = 1;
    // pto: %136
    ;
    int64_t v631 = v97 * v23;
    // pto: %136
    ;
    int64_t v632 = v630 * v631;
    // pto: %136
    ;
    pto::Shape<1, 1, 1, -1, 64> v633 = pto::Shape<1, 1, 1, -1, 64>(v628, v629, v630, v97, v25);
    // pto: %136
    ;
    pto::Stride<-1, -1, -1, -1, -1> v634 = pto::Stride<-1, -1, -1, -1, -1>(v629 * v632, v632, v631, v23, v24);
    // pto: %136
    ;
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 64>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v635 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 64>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v627 + (v626 + v625 * v23 + v33 * v24), v633, v634);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID6);
    TSTORE(v635, v621);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID7);
    // pto: %gamma_rope_input_tail_inline1884__ssa_v0
    ;
    Tile<TileType::Vec, bfloat16_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v636 = Tile<TileType::Vec, bfloat16_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v24, v25);
    // pto: %gamma_rope_input_tail_inline1884__ssa_v0
    ;
    uint64_t v637 = (uint64_t) v17;
    TASSIGN(v636, v637);
    // pto: %137
    ;
    __gm__ bfloat16_t* v638 = PTOAS__GLOBAL_TENSOR_DATA(v67);
    // pto: %137
    ;
    unsigned v639 = 448;
    // pto: %137
    ;
    pto::Shape<1, 1, 1, 1, 64> v640 = pto::Shape<1, 1, 1, 1, 64>();
    // pto: %137
    ;
    pto::Stride<64, 64, 64, 64, 1> v641 = pto::Stride<64, 64, 64, 64, 1>();
    // pto: %137
    ;
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<64, 64, 64, 64, 1>, pto::Layout::ND> v642 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<64, 64, 64, 64, 1>, pto::Layout::ND>(v638 + v639, v640, v641);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID7);
    TLOAD(v636, v642);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    // pto: %gamma_rope_cast_tail_inline1929__ssa_v0
    ;
    Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v643 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v24, v25);
    // pto: %gamma_rope_cast_tail_inline1929__ssa_v0
    ;
    uint64_t v644 = (uint64_t) v18;
    TASSIGN(v643, v644);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    RoundMode v645 = RoundMode::CAST_ROUND;
    SaturationMode v646 = SaturationMode::OFF;
    TCVT(v643, v636, v645, v646);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID6);
    // pto: %gamma_rope_tail_inline1858__ssa_v0
    ;
    Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v647 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v24, v25);
    // pto: %gamma_rope_tail_inline1858__ssa_v0
    ;
    uint64_t v648 = (uint64_t) v18;
    TASSIGN(v647, v648);
    // pto: %kv_rope_chunk_tail_inline1880__ssa_v0
    ;
    Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v649 = Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v97, v25);
    // pto: %kv_rope_chunk_tail_inline1880__ssa_v0
    ;
    uint64_t v650 = (uint64_t) v17;
    TASSIGN(v649, v650);
    // pto: %139
    ;
    const int64_t v651 = 0;
    // pto: %139
    ;
    __gm__ float* v652 = PTOAS__GLOBAL_TENSOR_DATA(v48);
    // pto: %139
    ;
    const int64_t v653 = 1;
    // pto: %139
    ;
    const int64_t v654 = 1;
    // pto: %139
    ;
    const int64_t v655 = 1;
    // pto: %139
    ;
    int64_t v656 = v97 * v23;
    // pto: %139
    ;
    int64_t v657 = v655 * v656;
    // pto: %139
    ;
    pto::Shape<1, 1, 1, -1, 64> v658 = pto::Shape<1, 1, 1, -1, 64>(v653, v654, v655, v97, v25);
    // pto: %139
    ;
    pto::Stride<-1, -1, -1, -1, -1> v659 = pto::Stride<-1, -1, -1, -1, -1>(v654 * v657, v657, v656, v23, v24);
    // pto: %139
    ;
    GlobalTensor<float, pto::Shape<1, 1, 1, -1, 64>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v660 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, 64>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v652 + (v651 + v585 * v23 + v34 * v24), v658, v659);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID6);
    TLOAD(v649, v660);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    // pto: %t__tmp_v95
    ;
    Tile<TileType::Vec, bfloat16_t, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v661 = Tile<TileType::Vec, bfloat16_t, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v97, v25);
    // pto: %t__tmp_v95
    ;
    uint64_t v662 = (uint64_t) v16;
    TASSIGN(v661, v662);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    RoundMode v663 = RoundMode::CAST_RINT;
    SaturationMode v664 = SaturationMode::OFF;
    TCVT(v661, v649, v663, v664);
    // pto: %kv_rope_chunk_tail_v1_inline1840__ssa_v0
    ;
    Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v665 = Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v97, v25);
    // pto: %kv_rope_chunk_tail_v1_inline1840__ssa_v0
    ;
    uint64_t v666 = (uint64_t) v17;
    TASSIGN(v665, v666);
    pipe_barrier(PIPE_V);
    RoundMode v667 = RoundMode::CAST_ROUND;
    SaturationMode v668 = SaturationMode::OFF;
    TCVT(v665, v661, v667, v668);
    // pto: %t__tmp_v96
    ;
    Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v669 = Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v97, v25);
    // pto: %t__tmp_v96
    ;
    uint64_t v670 = (uint64_t) v17;
    TASSIGN(v669, v670);
    pipe_barrier(PIPE_V);
    TROWEXPANDMUL(v669, v665, v473);
    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    // pto: %kv_rope_norm_tail_inline1899__ssa_v0
    ;
    Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v671 = Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v97, v25);
    // pto: %kv_rope_norm_tail_inline1899__ssa_v0
    ;
    uint64_t v672 = (uint64_t) v17;
    TASSIGN(v671, v672);
    pipe_barrier(PIPE_V);
    TCOLEXPANDMUL(v671, v669, v647);
    // pto: %t__tmp_v97
    ;
    Tile<TileType::Vec, bfloat16_t, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v673 = Tile<TileType::Vec, bfloat16_t, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v97, v25);
    // pto: %t__tmp_v97
    ;
    uint64_t v674 = (uint64_t) v16;
    TASSIGN(v673, v674);
    pipe_barrier(PIPE_V);
    RoundMode v675 = RoundMode::CAST_RINT;
    SaturationMode v676 = SaturationMode::OFF;
    TCVT(v673, v671, v675, v676);
    // pto: %kv_rope_norm_tail_v1_inline1854__ssa_v0
    ;
    Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v677 = Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v97, v25);
    // pto: %kv_rope_norm_tail_v1_inline1854__ssa_v0
    ;
    uint64_t v678 = (uint64_t) v17;
    TASSIGN(v677, v678);
    pipe_barrier(PIPE_V);
    RoundMode v679 = RoundMode::CAST_ROUND;
    SaturationMode v680 = SaturationMode::OFF;
    TCVT(v677, v673, v679, v680);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID7);
    // pto: %kv_cos_il_tail_inline1955__ssa_v0
    ;
    Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v681 = Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v97, v25);
    // pto: %kv_cos_il_tail_inline1955__ssa_v0
    ;
    uint64_t v682 = (uint64_t) v16;
    TASSIGN(v681, v682);
    // pto: %141
    ;
    const int64_t v683 = 0;
    // pto: %141
    ;
    __gm__ float* v684 = PTOAS__GLOBAL_TENSOR_DATA(v76);
    // pto: %141
    ;
    const int64_t v685 = 1;
    // pto: %141
    ;
    const int64_t v686 = 1;
    // pto: %141
    ;
    const int64_t v687 = 1;
    // pto: %141
    ;
    int64_t v688 = v97 * v25;
    // pto: %141
    ;
    int64_t v689 = v687 * v688;
    // pto: %141
    ;
    pto::Shape<1, 1, 1, -1, 64> v690 = pto::Shape<1, 1, 1, -1, 64>(v685, v686, v687, v97, v25);
    // pto: %141
    ;
    pto::Stride<-1, -1, -1, -1, -1> v691 = pto::Stride<-1, -1, -1, -1, -1>(v686 * v689, v689, v688, v25, v24);
    // pto: %141
    ;
    GlobalTensor<float, pto::Shape<1, 1, 1, -1, 64>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v692 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, 64>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v684 + (v683 + v625 * v25 + v28 * v24), v690, v691);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID7);
    TLOAD(v681, v692);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    // pto: %kv_sin_signed_tail_inline1851__ssa_v0
    ;
    Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v693 = Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v97, v25);
    // pto: %kv_sin_signed_tail_inline1851__ssa_v0
    ;
    uint64_t v694 = (uint64_t) v18;
    TASSIGN(v693, v694);
    // pto: %143
    ;
    const int64_t v695 = 0;
    // pto: %143
    ;
    __gm__ float* v696 = PTOAS__GLOBAL_TENSOR_DATA(v85);
    // pto: %143
    ;
    const int64_t v697 = 1;
    // pto: %143
    ;
    const int64_t v698 = 1;
    // pto: %143
    ;
    const int64_t v699 = 1;
    // pto: %143
    ;
    int64_t v700 = v97 * v25;
    // pto: %143
    ;
    int64_t v701 = v699 * v700;
    // pto: %143
    ;
    pto::Shape<1, 1, 1, -1, 64> v702 = pto::Shape<1, 1, 1, -1, 64>(v697, v698, v699, v97, v25);
    // pto: %143
    ;
    pto::Stride<-1, -1, -1, -1, -1> v703 = pto::Stride<-1, -1, -1, -1, -1>(v698 * v701, v701, v700, v25, v24);
    // pto: %143
    ;
    GlobalTensor<float, pto::Shape<1, 1, 1, -1, 64>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v704 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, 64>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v696 + (v695 + v625 * v25 + v28 * v24), v702, v703);
    TLOAD(v693, v704);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
    // pto: %t__tmp_v98
    ;
    Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v705 = Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v26, v25);
    // pto: %t__tmp_v98
    ;
    uint64_t v706 = (uint64_t) v19;
    TASSIGN(v705, v706);
    TEXPANDS(v705, v32);
    // pto: %t__ci_tmp_v0
    ;
    Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v707 = Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v24, v35);
    // pto: %t__ci_tmp_v0
    ;
    uint64_t v708 = (uint64_t) v20;
    TASSIGN(v707, v708);
    // pto: %t__tmp_v99
    ;
    Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v709 = Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v24, v25);
    // pto: %t__tmp_v99
    ;
    uint64_t v710 = (uint64_t) v21;
    TASSIGN(v709, v710);
    // pto: %ci_tmp_view
    ;
    Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, 1, 192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v711;
    TRESHAPE(v711, v707);
    // pto: %ci_dst_view
    ;
    Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v712;
    TRESHAPE(v712, v709);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
    TCI<Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, 1, 192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, int32_t, 0>(v712, v36, v711);
    set_flag(PIPE_S, PIPE_V, EVENT_ID0);
    // pto: %t__tmp_v100
    ;
    Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v713 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v24, v25);
    // pto: %t__tmp_v100
    ;
    uint64_t v714 = (uint64_t) v20;
    TASSIGN(v713, v714);
    wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
    RoundMode v715 = RoundMode::CAST_ROUND;
    SaturationMode v716 = SaturationMode::OFF;
    TCVT(v713, v709, v715, v716);
    // pto: %kv_col_inline1850__ssa_v0
    ;
    Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v717 = Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v26, v25);
    // pto: %kv_col_inline1850__ssa_v0
    ;
    uint64_t v718 = (uint64_t) v19;
    TASSIGN(v717, v718);
    pipe_barrier(PIPE_V);
    TCOLEXPANDMUL(v717, v705, v713);
    // pto: %t__tmp_v101
    ;
    Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v719 = Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v26, v25);
    // pto: %t__tmp_v101
    ;
    uint64_t v720 = (uint64_t) v20;
    TASSIGN(v719, v720);
    pipe_barrier(PIPE_V);
    TMULS(v719, v717, v37);
    // pto: %t__tmp_v102
    ;
    Tile<TileType::Vec, int32_t, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v721 = Tile<TileType::Vec, int32_t, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v26, v25);
    // pto: %t__tmp_v102
    ;
    uint64_t v722 = (uint64_t) v20;
    TASSIGN(v721, v722);
    pipe_barrier(PIPE_V);
    RoundMode v723 = RoundMode::CAST_TRUNC;
    SaturationMode v724 = SaturationMode::ON;
    TCVT(v721, v719, v723, v724);
    // pto: %kv_dup_f_inline1847__ssa_v0
    ;
    Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v725 = Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v26, v25);
    // pto: %kv_dup_f_inline1847__ssa_v0
    ;
    uint64_t v726 = (uint64_t) v20;
    TASSIGN(v725, v726);
    pipe_barrier(PIPE_V);
    RoundMode v727 = RoundMode::CAST_ROUND;
    SaturationMode v728 = SaturationMode::OFF;
    TCVT(v725, v721, v727, v728);
    // pto: %t__tmp_v103
    ;
    Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v729 = Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v26, v25);
    // pto: %t__tmp_v103
    ;
    uint64_t v730 = (uint64_t) v20;
    TASSIGN(v729, v730);
    pipe_barrier(PIPE_V);
    TMULS(v729, v725, v38);
    // pto: %kv_lane_inline1891__ssa_v0
    ;
    Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v731 = Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v26, v25);
    // pto: %kv_lane_inline1891__ssa_v0
    ;
    uint64_t v732 = (uint64_t) v20;
    TASSIGN(v731, v732);
    pipe_barrier(PIPE_V);
    TSUB(v731, v717, v729);
    // pto: %t__tmp_v104
    ;
    Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v733 = Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v26, v25);
    // pto: %t__tmp_v104
    ;
    uint64_t v734 = (uint64_t) v19;
    TASSIGN(v733, v734);
    pipe_barrier(PIPE_V);
    TADDS(v733, v717, v32);
    // pto: %t__tmp_v105
    ;
    Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v735 = Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v26, v25);
    // pto: %t__tmp_v105
    ;
    uint64_t v736 = (uint64_t) v20;
    TASSIGN(v735, v736);
    TMULS(v735, v731, v38);
    // pto: %kv_swap_f_inline1867__ssa_v0
    ;
    Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v737 = Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v26, v25);
    // pto: %kv_swap_f_inline1867__ssa_v0
    ;
    uint64_t v738 = (uint64_t) v19;
    TASSIGN(v737, v738);
    pipe_barrier(PIPE_V);
    TSUB(v737, v733, v735);
    set_flag(PIPE_V, PIPE_S, EVENT_ID1);
    // pto: %t__ci_tmp_v1
    ;
    Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v739 = Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v24, v35);
    // pto: %t__ci_tmp_v1
    ;
    uint64_t v740 = (uint64_t) v20;
    TASSIGN(v739, v740);
    // pto: %t__tmp_v106
    ;
    Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v741 = Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v24, v26);
    // pto: %t__tmp_v106
    ;
    uint64_t v742 = (uint64_t) v21;
    TASSIGN(v741, v742);
    // pto: %144
    ;
    Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, 1, 192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v743;
    TRESHAPE(v743, v739);
    // pto: %145
    ;
    Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, 1, 32, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v744;
    TRESHAPE(v744, v741);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID1);
    TCI<Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, 1, 32, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, 1, 192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, int32_t, 0>(v744, v36, v743);
    set_flag(PIPE_S, PIPE_V, EVENT_ID1);
    // pto: %t__tmp_v107
    ;
    Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v745 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v24, v26);
    // pto: %t__tmp_v107
    ;
    uint64_t v746 = (uint64_t) v20;
    TASSIGN(v745, v746);
    wait_flag(PIPE_S, PIPE_V, EVENT_ID1);
    RoundMode v747 = RoundMode::CAST_ROUND;
    SaturationMode v748 = SaturationMode::OFF;
    TCVT(v745, v741, v747, v748);
    // pto: %kv_row_seed_inline1912__ssa_v0
    ;
    Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v749 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v24, v26);
    // pto: %kv_row_seed_inline1912__ssa_v0
    ;
    uint64_t v750 = (uint64_t) v21;
    TASSIGN(v749, v750);
    pipe_barrier(PIPE_V);
    TMULS(v749, v745, v39);
    // pto: %t__tmp_v108
    ;
    Tile<TileType::Vec, float, 64, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v751 = Tile<TileType::Vec, float, 64, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v25, v26);
    // pto: %t__tmp_v108
    ;
    uint64_t v752 = (uint64_t) v20;
    TASSIGN(v751, v752);
    pipe_barrier(PIPE_V);
    TEXPANDS(v751, v32);
    // pto: %kv_row_grid_inline1864__ssa_v0
    ;
    Tile<TileType::Vec, float, 64, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v753 = Tile<TileType::Vec, float, 64, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v25, v26);
    // pto: %kv_row_grid_inline1864__ssa_v0
    ;
    uint64_t v754 = (uint64_t) v20;
    TASSIGN(v753, v754);
    pipe_barrier(PIPE_V);
    TCOLEXPANDMUL(v753, v751, v749);
    // pto: %transpose_tmp
    ;
    Tile<TileType::Vec, float, 64, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v755 = Tile<TileType::Vec, float, 64, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v25, v26);
    // pto: %transpose_tmp
    ;
    uint64_t v756 = (uint64_t) v21;
    TASSIGN(v755, v756);
    // pto: %kv_row_offset_inline1922__ssa_v0
    ;
    Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v757 = Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v26, v25);
    // pto: %kv_row_offset_inline1922__ssa_v0
    ;
    uint64_t v758 = (uint64_t) v22;
    TASSIGN(v757, v758);
    pipe_barrier(PIPE_V);
    TTRANS(v757, v753, v755);
    // pto: %t__tmp_v109
    ;
    Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v759 = Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v26, v25);
    // pto: %t__tmp_v109
    ;
    uint64_t v760 = (uint64_t) v19;
    TASSIGN(v759, v760);
    pipe_barrier(PIPE_V);
    TADD(v759, v737, v757);
    // pto: %kv_swap_idx_tail_inline1896__ssa_v0
    ;
    Tile<TileType::Vec, int32_t, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v761 = Tile<TileType::Vec, int32_t, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v26, v25);
    // pto: %kv_swap_idx_tail_inline1896__ssa_v0
    ;
    uint64_t v762 = (uint64_t) v19;
    TASSIGN(v761, v762);
    pipe_barrier(PIPE_V);
    RoundMode v763 = RoundMode::CAST_ROUND;
    SaturationMode v764 = SaturationMode::ON;
    TCVT(v761, v759, v763, v764);
    // pto: %kv_gather_tmp_inline1920__ssa_v0
    ;
    Tile<TileType::Vec, int32_t, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v765 = Tile<TileType::Vec, int32_t, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v26, v25);
    // pto: %kv_gather_tmp_inline1920__ssa_v0
    ;
    uint64_t v766 = (uint64_t) v20;
    TASSIGN(v765, v766);
    // pto: %kv_swapped_tail_inline1889__ssa_v0
    ;
    Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v767 = Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v26, v25);
    // pto: %kv_swapped_tail_inline1889__ssa_v0
    ;
    uint64_t v768 = (uint64_t) v21;
    TASSIGN(v767, v768);
    pipe_barrier(PIPE_V);
    TGATHER(v767, v677, v761, v765);
    // pto: %t__tmp_v110
    ;
    Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v769 = Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v97, v25);
    // pto: %t__tmp_v110
    ;
    uint64_t v770 = (uint64_t) v17;
    TASSIGN(v769, v770);
    pipe_barrier(PIPE_V);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TMUL(v769, v677, v681);
    // pto: %t__tmp_v111
    ;
    Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v771 = Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v26, v25);
    // pto: %t__tmp_v111
    ;
    uint64_t v772 = (uint64_t) v16;
    TASSIGN(v771, v772);
    pipe_barrier(PIPE_V);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
    TMUL(v771, v767, v693);
    // pto: %kv_rope_rot_tail_inline1839__ssa_v0
    ;
    Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v773 = Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v97, v25);
    // pto: %kv_rope_rot_tail_inline1839__ssa_v0
    ;
    uint64_t v774 = (uint64_t) v17;
    TASSIGN(v773, v774);
    pipe_barrier(PIPE_V);
    TADD(v773, v769, v771);
    // pto: %kv_rope_i16_tail_inline1917__ssa_v0
    ;
    Tile<TileType::Vec, bfloat16_t, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v775 = Tile<TileType::Vec, bfloat16_t, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v97, v25);
    // pto: %kv_rope_i16_tail_inline1917__ssa_v0
    ;
    uint64_t v776 = (uint64_t) v17;
    TASSIGN(v775, v776);
    pipe_barrier(PIPE_V);
    RoundMode v777 = RoundMode::CAST_RINT;
    SaturationMode v778 = SaturationMode::OFF;
    TCVT(v775, v773, v777, v778);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID7);
    v775.SetValidShape(v97, v25);
    // pto: %147
    ;
    const int64_t v779 = 0;
    // pto: %147
    ;
    __gm__ bfloat16_t* v780 = PTOAS__GLOBAL_TENSOR_DATA(v57);
    // pto: %147
    ;
    const int64_t v781 = 1;
    // pto: %147
    ;
    const int64_t v782 = 1;
    // pto: %147
    ;
    const int64_t v783 = 1;
    // pto: %147
    ;
    int64_t v784 = v97 * v23;
    // pto: %147
    ;
    int64_t v785 = v783 * v784;
    // pto: %147
    ;
    pto::Shape<1, 1, 1, -1, 64> v786 = pto::Shape<1, 1, 1, -1, 64>(v781, v782, v783, v97, v25);
    // pto: %147
    ;
    pto::Stride<-1, -1, -1, -1, -1> v787 = pto::Stride<-1, -1, -1, -1, -1>(v782 * v785, v785, v784, v23, v24);
    // pto: %147
    ;
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 64>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v788 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 64>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v780 + (v779 + v625 * v23 + v34 * v24), v786, v787);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID7);
    TSTORE(v788, v775);
  }
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID4);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID4);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID5);
  #endif // __DAV_VEC__

  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}