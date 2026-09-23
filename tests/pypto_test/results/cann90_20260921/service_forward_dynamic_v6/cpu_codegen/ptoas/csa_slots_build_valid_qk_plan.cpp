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

AICORE void csa_slots_build_valid_qk_plan(__gm__ int32_t* v1, __gm__ float* v2, __gm__ int64_t* v3, __gm__ int32_t* v4, __gm__ int32_t* v5, __gm__ int32_t* v6, int64_t v7, int64_t v8, int64_t v9, int64_t v10, int64_t v11, int32_t v12, int32_t v13) {
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  const int32_t v14 = 1;
  // pto: %c3072_i64
  const int64_t v15 = 3072;
  // pto: %c19456_i64
  const int64_t v16 = 19456;
  // pto: %c0_i64
  const int64_t v17 = 0;
  // pto: %c2048_i64
  const int64_t v18 = 2048;
  // pto: %c2560_i64
  const int64_t v19 = 2560;
  // pto: %c512_index
  const int64_t v20 = 512;
  // pto: %c1_index
  const int64_t v21 = 1;
  // pto: %c1024_index
  const int64_t v22 = 1024;
  // pto: %c16_index
  const int64_t v23 = 16;
  // pto: %c8_index
  const int64_t v24 = 8;
  // pto: %c128_index
  const int64_t v25 = 128;
  // pto: %c0_index
  const int64_t v26 = 0;
  // pto: %c6_index
  const int64_t v27 = 6;
  // pto: %c4_index
  const int64_t v28 = 4;
  // pto: %cst_14
  const float v29 = 1.0f;
  // pto: %cst_15
  const float v30 = 0.0f;
  // pto: %c32_index
  const int64_t v31 = 32;
  // pto: %c192_index
  const int64_t v32 = 192;
  const int32_t v33 = 0;
  // pto: %c5_index
  const int64_t v34 = 5;
  // pto: %cst_20
  const float v35 = 1.00000002E+20f;
  // pto: %c384_index
  const int64_t v36 = 384;
  // pto: %cst_22
  const float v37 = -1.00000002E+20f;
  // pto: %cmp_sparse_indices_inline2524__ssa_v0_view
  const int64_t v38 = 1;
  // pto: %cmp_sparse_indices_inline2524__ssa_v0_view
  const int64_t v39 = 1;
  // pto: %cmp_sparse_indices_inline2524__ssa_v0_view
  const int64_t v40 = 1;
  // pto: %cmp_sparse_indices_inline2524__ssa_v0_view
  int64_t v41 = (int64_t) v8;
  // pto: %cmp_sparse_indices_inline2524__ssa_v0_view
  int64_t v42 = v41 * v20;
  // pto: %cmp_sparse_indices_inline2524__ssa_v0_view
  int64_t v43 = v40 * v42;
  // pto: %cmp_sparse_indices_inline2524__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v44 = pto::Shape<1, 1, 1, -1, -1>(v38, v39, v40, v41, v20);
  // pto: %cmp_sparse_indices_inline2524__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v45 = pto::Stride<-1, -1, -1, -1, -1>(v39 * v43, v43, v42, v20, v21);
  // pto: %cmp_sparse_indices_inline2524__ssa_v0_view
  GlobalTensor<int32_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v46 = GlobalTensor<int32_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v1, v44, v45);
  // pto: %sparse_bias_inline2510__ssa_v0_view
  const int64_t v47 = 1;
  // pto: %sparse_bias_inline2510__ssa_v0_view
  const int64_t v48 = 1;
  // pto: %sparse_bias_inline2510__ssa_v0_view
  const int64_t v49 = 1;
  // pto: %sparse_bias_inline2510__ssa_v0_view
  int64_t v50 = (int64_t) v8;
  // pto: %sparse_bias_inline2510__ssa_v0_view
  int64_t v51 = v50 * v22;
  // pto: %sparse_bias_inline2510__ssa_v0_view
  int64_t v52 = v49 * v51;
  // pto: %sparse_bias_inline2510__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v53 = pto::Shape<1, 1, 1, -1, -1>(v47, v48, v49, v50, v22);
  // pto: %sparse_bias_inline2510__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v54 = pto::Stride<-1, -1, -1, -1, -1>(v48 * v52, v52, v51, v22, v21);
  // pto: %sparse_bias_inline2510__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v55 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v2, v53, v54);
  // pto: %idx_topk__ssa_v3_view
  const int64_t v56 = 1;
  // pto: %idx_topk__ssa_v3_view
  const int64_t v57 = 1;
  // pto: %idx_topk__ssa_v3_view
  const int64_t v58 = 1;
  // pto: %idx_topk__ssa_v3_view
  int64_t v59 = (int64_t) v9;
  // pto: %idx_topk__ssa_v3_view
  int64_t v60 = v59 * v20;
  // pto: %idx_topk__ssa_v3_view
  int64_t v61 = v58 * v60;
  // pto: %idx_topk__ssa_v3_view
  pto::Shape<1, 1, 1, -1, -1> v62 = pto::Shape<1, 1, 1, -1, -1>(v56, v57, v58, v59, v20);
  // pto: %idx_topk__ssa_v3_view
  pto::Stride<-1, -1, -1, -1, -1> v63 = pto::Stride<-1, -1, -1, -1, -1>(v57 * v61, v61, v60, v20, v21);
  // pto: %idx_topk__ssa_v3_view
  GlobalTensor<int32_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v64 = GlobalTensor<int32_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v4, v62, v63);
  // pto: %v_valid_inline2571__phi_v5
  Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v65 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v21, v25);
  // pto: %v_valid_inline2571__phi_v5
  uint64_t v66 = (uint64_t) v17;
  TASSIGN(v65, v66);
  // pto: %v_valid_inline2571__phi_v4
  Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v67 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v21, v25);
  // pto: %v_valid_inline2571__phi_v4
  uint64_t v68 = (uint64_t) v17;
  TASSIGN(v67, v68);
  // pto: %plan_worker_inline2561__ssa_v0, %22
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_S, PIPE_V, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  for (int64_t v69 = (int64_t) ((uint64_t) ((int64_t) v12) * (uint64_t) v24); v69 < v8; v69 += v25) {
    // pto: %23
    ;
    int64_t v70 = (int64_t) ((uint64_t) v8 - (uint64_t) v69);
    // pto: %24
    ;
    int64_t v71 = v70 < v24 ? v70 : v24;
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
    for (int64_t v72 = v26; v72 < v71; v72 += v21) {
      // pto: %25
      ;
      int64_t v73 = (int64_t) ((uint64_t) v69 + (uint64_t) v72);
      // pto: %27
      ;
      int64_t v74 = (v3)[v73];
      // pto: %t__tile
      ;
      Tile<TileType::Vec, int32_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v75 = Tile<TileType::Vec, int32_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v21, v20);
      // pto: %t__tile
      ;
      uint64_t v76 = (uint64_t) v15;
      TASSIGN(v75, v76);
      // pto: %29
      ;
      int64_t v77 = v73 < v26 ? v26 : v73;
      // pto: %idx_topk__ssa_v3_pview
      ;
      __gm__ int32_t* v78 = PTOAS__GLOBAL_TENSOR_DATA(v64);
      // pto: %idx_topk__ssa_v3_pview
      ;
      const int64_t v79 = 0;
      // pto: %idx_topk__ssa_v3_pview
      ;
      const int64_t v80 = 512;
      // pto: %idx_topk__ssa_v3_pview
      ;
      pto::Shape<1, 1, 1, 1, 512> v81 = pto::Shape<1, 1, 1, 1, 512>();
      // pto: %idx_topk__ssa_v3_pview
      ;
      pto::Stride<512, 512, 512, 512, 1> v82 = pto::Stride<512, 512, 512, 512, 1>();
      // pto: %idx_topk__ssa_v3_pview
      ;
      GlobalTensor<int32_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<512, 512, 512, 512, 1>, pto::Layout::ND> v83 = GlobalTensor<int32_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<512, 512, 512, 512, 1>, pto::Layout::ND>(v78 + (v79 + v77 * v80), v81, v82);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
      TLOAD(v75, v83);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      // pto: %c_raw_inline2530__tile
      ;
      Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v84 = Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v21, v20);
      // pto: %c_raw_inline2530__tile
      ;
      uint64_t v85 = (uint64_t) v15;
      TASSIGN(v84, v85);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      RoundMode v86 = RoundMode::CAST_ROUND;
      SaturationMode v87 = SaturationMode::OFF;
      TCVT(v84, v75, v86, v87);
      // pto: %30
      ;
      int64_t v88 = (int64_t) ((uint64_t) v74 + (uint64_t) v21);
      // pto: %31, %32, %33
      ;
      float v89 = (float) ((int32_t) v88 / v28);
      // pto: %0
      ;
      Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v90 = Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v21, v20);
      // pto: %0
      ;
      uint64_t v91 = (uint64_t) v16;
      TASSIGN(v90, v91);
      pipe_barrier(PIPE_V);
      TADDS(v90, v84, v29);
      // pto: %1
      ;
      Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v92 = Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v21, v20);
      // pto: %1
      ;
      uint64_t v93 = (uint64_t) v16;
      TASSIGN(v92, v93);
      pipe_barrier(PIPE_V);
      TMAXS(v92, v90, v30);
      // pto: %c_ge_inline2560__tile
      ;
      Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v94 = Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v21, v20);
      // pto: %c_ge_inline2560__tile
      ;
      uint64_t v95 = (uint64_t) v16;
      TASSIGN(v94, v95);
      pipe_barrier(PIPE_V);
      TMINS(v94, v92, v29);
      // pto: %2
      ;
      Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v96 = Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v21, v20);
      // pto: %2
      ;
      uint64_t v97 = (uint64_t) v17;
      TASSIGN(v96, v97);
      TNEG(v96, v84);
      // pto: %3
      ;
      Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v98 = Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v21, v20);
      // pto: %3
      ;
      uint64_t v99 = (uint64_t) v17;
      TASSIGN(v98, v99);
      pipe_barrier(PIPE_V);
      TADDS(v98, v96, v89);
      // pto: %4
      ;
      Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v100 = Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v21, v20);
      // pto: %4
      ;
      uint64_t v101 = (uint64_t) v17;
      TASSIGN(v100, v101);
      pipe_barrier(PIPE_V);
      TMAXS(v100, v98, v30);
      // pto: %c_lt_inline2563__tile
      ;
      Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v102 = Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v21, v20);
      // pto: %c_lt_inline2563__tile
      ;
      uint64_t v103 = (uint64_t) v17;
      TASSIGN(v102, v103);
      pipe_barrier(PIPE_V);
      TMINS(v102, v100, v29);
      // pto: %c_mask_inline2562__tile
      ;
      Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v104 = Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v21, v20);
      // pto: %c_mask_inline2562__tile
      ;
      uint64_t v105 = (uint64_t) v16;
      TASSIGN(v104, v105);
      pipe_barrier(PIPE_V);
      TMUL(v104, v94, v102);
      // pto: %5
      ;
      Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v106 = Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v21, v20);
      // pto: %5
      ;
      uint64_t v107 = (uint64_t) v15;
      TASSIGN(v106, v107);
      TADDS(v106, v84, v29);
      // pto: %6
      ;
      Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v108 = Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v21, v20);
      // pto: %6
      ;
      uint64_t v109 = (uint64_t) v15;
      TASSIGN(v108, v109);
      pipe_barrier(PIPE_V);
      TMUL(v108, v104, v106);
      // pto: %c_out_inline2535__tile
      ;
      Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v110 = Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v21, v20);
      // pto: %c_out_inline2535__tile
      ;
      uint64_t v111 = (uint64_t) v15;
      TASSIGN(v110, v111);
      pipe_barrier(PIPE_V);
      TSUBS(v110, v108, v29);
      // pto: %7
      ;
      Tile<TileType::Vec, int32_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v112 = Tile<TileType::Vec, int32_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v21, v20);
      // pto: %7
      ;
      uint64_t v113 = (uint64_t) v16;
      TASSIGN(v112, v113);
      pipe_barrier(PIPE_V);
      RoundMode v114 = RoundMode::CAST_ROUND;
      SaturationMode v115 = SaturationMode::ON;
      TCVT(v112, v110, v114, v115);
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      // pto: %cmp_sparse_indices_inline2524__iter_v3_pview
      ;
      __gm__ int32_t* v116 = PTOAS__GLOBAL_TENSOR_DATA(v46);
      // pto: %cmp_sparse_indices_inline2524__iter_v3_pview
      ;
      const int64_t v117 = 0;
      // pto: %cmp_sparse_indices_inline2524__iter_v3_pview
      ;
      const int64_t v118 = 512;
      // pto: %cmp_sparse_indices_inline2524__iter_v3_pview
      ;
      pto::Shape<1, 1, 1, 1, 512> v119 = pto::Shape<1, 1, 1, 1, 512>();
      // pto: %cmp_sparse_indices_inline2524__iter_v3_pview
      ;
      pto::Stride<512, 512, 512, 512, 1> v120 = pto::Stride<512, 512, 512, 512, 1>();
      // pto: %cmp_sparse_indices_inline2524__iter_v3_pview
      ;
      GlobalTensor<int32_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<512, 512, 512, 512, 1>, pto::Layout::ND> v121 = GlobalTensor<int32_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<512, 512, 512, 512, 1>, pto::Layout::ND>(v116 + (v117 + v77 * v118), v119, v120);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      TSTORE(v121, v112);
      set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
      // pto: %36
      ;
      int64_t v122 = v88 < v25 ? v88 : v25;
      // pto: %37, %38
      ;
      int64_t v123 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) v74 - (uint64_t) v122) + (uint64_t) v21);
      // pto: %t__ci_tmp_v0
      ;
      Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v124 = Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v21, v32);
      // pto: %t__ci_tmp_v0
      ;
      uint64_t v125 = (uint64_t) v16;
      TASSIGN(v124, v125);
      // pto: %8
      ;
      Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v126 = Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v21, v25);
      // pto: %8
      ;
      uint64_t v127 = (uint64_t) v17;
      TASSIGN(v126, v127);
      // pto: %ci_tmp_view
      ;
      Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, 1, 192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v128;
      TRESHAPE(v128, v124);
      // pto: %ci_dst_view
      ;
      Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, 1, 128, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v129;
      TRESHAPE(v129, v126);
      wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
      TCI<Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, 1, 128, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, 1, 192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, int32_t, 0>(v129, v33, v128);
      set_flag(PIPE_S, PIPE_V, EVENT_ID1);
      // pto: %v_columns_inline2567__tile
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v130 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v21, v25);
      // pto: %v_columns_inline2567__tile
      ;
      uint64_t v131 = (uint64_t) v16;
      TASSIGN(v130, v131);
      wait_flag(PIPE_S, PIPE_V, EVENT_ID1);
      RoundMode v132 = RoundMode::CAST_ROUND;
      SaturationMode v133 = SaturationMode::OFF;
      TCVT(v130, v126, v132, v133);
      // pto: %v_valid_inline2571__tile
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v134 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v21, v25);
      // pto: %v_valid_inline2571__tile
      ;
      uint64_t v135 = (uint64_t) v17;
      TASSIGN(v134, v135);
      pipe_barrier(PIPE_V);
      TEXPANDS(v134, v30);
      // pto: %v_block_valid_inline2542__rv_v2
      ;
      int32_t v136;
      v136 = v33;
      for (int64_t v137 = v26; v137 < v34; v137 += v21) {
        // pto: %41, %42, %39
        ;
        int64_t v138 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) v137 * (uint64_t) v31) - (uint64_t) (v123 % v31));
        // pto: %43
        ;
        int64_t v139 = v138 < v26 ? v26 : v138;
        // pto: %46
        ;
        int64_t v140 = (int64_t) ((uint64_t) v138 + (uint64_t) v31);
        // pto: %47
        ;
        int64_t v141 = v140 < v122 ? v140 : v122;
        // pto: %48
        ;
        // pto: %v_block_valid_inline2542__phi_v5
        ;
        int32_t v142;
        if (v139 < v141) {
          // pto: %26, %51, %52, %49, %50, %v_page_inline2593__tile
          ;
          int32_t v143 = (v5)[(int64_t) ((uint64_t) ((int64_t) (uint64_t) (v73 / v27) * (uint64_t) v11) + (uint64_t) ((int64_t) ((uint64_t) v123 + (uint64_t) v139) / v31))];
          // pto: %53, %54
          ;
          bool v144 = (int64_t) v143 >= v26;
          // pto: %v_block_valid_inline2542__phi_v4
          ;
          int32_t v145 = v144 ? v14 : v136;
          if (v144) {
            // pto: %56, %57
            ;
            float v146 = (float) ((int32_t) v139);
            // pto: %58, %59
            ;
            float v147 = (float) ((int32_t) v141);
            // pto: %9
            ;
            Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v148 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v21, v25);
            // pto: %9
            ;
            uint64_t v149 = (uint64_t) v18;
            TASSIGN(v148, v149);
            pipe_barrier(PIPE_V);
            TSUBS(v148, v130, v146);
            // pto: %10
            ;
            Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v150 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v21, v25);
            // pto: %10
            ;
            uint64_t v151 = (uint64_t) v18;
            TASSIGN(v150, v151);
            pipe_barrier(PIPE_V);
            TADDS(v150, v148, v29);
            // pto: %11
            ;
            Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v152 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v21, v25);
            // pto: %11
            ;
            uint64_t v153 = (uint64_t) v18;
            TASSIGN(v152, v153);
            pipe_barrier(PIPE_V);
            TMAXS(v152, v150, v30);
            // pto: %v_ge_inline2499__tile
            ;
            Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v154 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v21, v25);
            // pto: %v_ge_inline2499__tile
            ;
            uint64_t v155 = (uint64_t) v18;
            TASSIGN(v154, v155);
            pipe_barrier(PIPE_V);
            TMINS(v154, v152, v29);
            // pto: %12
            ;
            Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v156 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v21, v25);
            // pto: %12
            ;
            uint64_t v157 = (uint64_t) v19;
            TASSIGN(v156, v157);
            TNEG(v156, v130);
            // pto: %13
            ;
            Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v158 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v21, v25);
            // pto: %13
            ;
            uint64_t v159 = (uint64_t) v19;
            TASSIGN(v158, v159);
            pipe_barrier(PIPE_V);
            TADDS(v158, v156, v147);
            // pto: %14
            ;
            Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v160 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v21, v25);
            // pto: %14
            ;
            uint64_t v161 = (uint64_t) v19;
            TASSIGN(v160, v161);
            pipe_barrier(PIPE_V);
            TMAXS(v160, v158, v30);
            // pto: %v_lt_inline2508__tile
            ;
            Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v162 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v21, v25);
            // pto: %v_lt_inline2508__tile
            ;
            uint64_t v163 = (uint64_t) v19;
            TASSIGN(v162, v163);
            pipe_barrier(PIPE_V);
            TMINS(v162, v160, v29);
            // pto: %15
            ;
            Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v164 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v21, v25);
            // pto: %15
            ;
            uint64_t v165 = (uint64_t) v18;
            TASSIGN(v164, v165);
            pipe_barrier(PIPE_V);
            TMUL(v164, v154, v162);
            // pto: %16
            ;
            Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v166 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v21, v25);
            // pto: %16
            ;
            uint64_t v167 = (uint64_t) v17;
            TASSIGN(v166, v167);
            pipe_barrier(PIPE_V);
            TADD(v166, v134, v164);
          };
          v142 = v145;
        } else {
          v142 = v136;
        };
        v136 = v142;
      };
      // pto: %60
      ;
      int64_t v168 = (int64_t) ((uint64_t) v73 * (uint64_t) v23);
      (v6)[v168] = v136;
      // pto: %17
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v169 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v21, v25);
      // pto: %17
      ;
      uint64_t v170 = (uint64_t) v16;
      TASSIGN(v169, v170);
      pipe_barrier(PIPE_V);
      TSUBS(v169, v134, v29);
      // pto: %18
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v171 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v21, v25);
      // pto: %18
      ;
      uint64_t v172 = (uint64_t) v16;
      TASSIGN(v171, v172);
      pipe_barrier(PIPE_V);
      TMULS(v171, v169, v35);
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
      // pto: %sparse_bias_inline2510__iter_v3_pview
      ;
      __gm__ float* v173 = PTOAS__GLOBAL_TENSOR_DATA(v55);
      // pto: %sparse_bias_inline2510__iter_v3_pview
      ;
      const int64_t v174 = 0;
      // pto: %sparse_bias_inline2510__iter_v3_pview
      ;
      const int64_t v175 = 1024;
      // pto: %sparse_bias_inline2510__iter_v3_pview
      ;
      pto::Shape<1, 1, 1, 1, 128> v176 = pto::Shape<1, 1, 1, 1, 128>();
      // pto: %sparse_bias_inline2510__iter_v3_pview
      ;
      pto::Stride<1024, 1024, 1024, 1024, 1> v177 = pto::Stride<1024, 1024, 1024, 1024, 1>();
      // pto: %sparse_bias_inline2510__iter_v3_pview
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<1024, 1024, 1024, 1024, 1>, pto::Layout::ND> v178 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<1024, 1024, 1024, 1024, 1>, pto::Layout::ND>(v173 + (v174 + v77 * v175), v176, v177);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
      TSTORE(v178, v171);
      set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
      // pto: %19
      ;
      Tile<TileType::Vec, float, 1, 384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v179 = Tile<TileType::Vec, float, 1, 384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v21, v36);
      // pto: %19
      ;
      uint64_t v180 = (uint64_t) v16;
      TASSIGN(v179, v180);
      wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
      TEXPANDS(v179, v37);
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
      // pto: %sparse_bias_inline2510__tile_pview
      ;
      __gm__ float* v181 = PTOAS__GLOBAL_TENSOR_DATA(v55);
      // pto: %sparse_bias_inline2510__tile_pview
      ;
      const int64_t v182 = 128;
      // pto: %sparse_bias_inline2510__tile_pview
      ;
      const int64_t v183 = 1024;
      // pto: %sparse_bias_inline2510__tile_pview
      ;
      pto::Shape<1, 1, 1, 1, 384> v184 = pto::Shape<1, 1, 1, 1, 384>();
      // pto: %sparse_bias_inline2510__tile_pview
      ;
      pto::Stride<1024, 1024, 1024, 1024, 1> v185 = pto::Stride<1024, 1024, 1024, 1024, 1>();
      // pto: %sparse_bias_inline2510__tile_pview
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 1, 384>, pto::Stride<1024, 1024, 1024, 1024, 1>, pto::Layout::ND> v186 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 384>, pto::Stride<1024, 1024, 1024, 1024, 1>, pto::Layout::ND>(v181 + (v182 + v77 * v183), v184, v185);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
      TSTORE(v186, v179);
      // pto: %20
      ;
      Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v187 = Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v21, v20);
      // pto: %20
      ;
      uint64_t v188 = (uint64_t) v15;
      TASSIGN(v187, v188);
      TMINS(v187, v110, v30);
      // pto: %21
      ;
      Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v189 = Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v21, v20);
      // pto: %21
      ;
      uint64_t v190 = (uint64_t) v15;
      TASSIGN(v189, v190);
      pipe_barrier(PIPE_V);
      TMULS(v189, v187, v35);
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID3);
      // pto: %67
      ;
      __gm__ float* v191 = PTOAS__GLOBAL_TENSOR_DATA(v55);
      // pto: %67
      ;
      const int64_t v192 = 512;
      // pto: %67
      ;
      const int64_t v193 = 1024;
      // pto: %67
      ;
      pto::Shape<1, 1, 1, 1, 512> v194 = pto::Shape<1, 1, 1, 1, 512>();
      // pto: %67
      ;
      pto::Stride<1024, 1024, 1024, 1024, 1> v195 = pto::Stride<1024, 1024, 1024, 1024, 1>();
      // pto: %67
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<1024, 1024, 1024, 1024, 1>, pto::Layout::ND> v196 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<1024, 1024, 1024, 1024, 1>, pto::Layout::ND>(v191 + (v192 + v77 * v193), v194, v195);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID3);
      pipe_barrier(PIPE_MTE3);
      TSTORE(v196, v189);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    };
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
    // pto: %c_blk_tmp_inline2522__ssa_v0
    ;
    Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v197 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v24, v20);
    // pto: %c_blk_tmp_inline2522__ssa_v0
    ;
    uint64_t v198 = (uint64_t) v15;
    TASSIGN(v197, v198);
    // pto: %t__tmp_v310
    ;
    Tile<TileType::Vec, int32_t, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v199 = Tile<TileType::Vec, int32_t, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v20);
    // pto: %t__tmp_v310
    ;
    uint64_t v200 = (uint64_t) v16;
    TASSIGN(v199, v200);
    // pto: %68
    ;
    int64_t v201 = v69 < v26 ? v26 : v69;
    // pto: %cmp_sparse_indices_inline2524__rv_v4_pview
    ;
    const int64_t v202 = 0;
    // pto: %cmp_sparse_indices_inline2524__rv_v4_pview
    ;
    __gm__ int32_t* v203 = PTOAS__GLOBAL_TENSOR_DATA(v46);
    // pto: %cmp_sparse_indices_inline2524__rv_v4_pview
    ;
    const int64_t v204 = 1;
    // pto: %cmp_sparse_indices_inline2524__rv_v4_pview
    ;
    const int64_t v205 = 1;
    // pto: %cmp_sparse_indices_inline2524__rv_v4_pview
    ;
    const int64_t v206 = 1;
    // pto: %cmp_sparse_indices_inline2524__rv_v4_pview
    ;
    int64_t v207 = v71 * v20;
    // pto: %cmp_sparse_indices_inline2524__rv_v4_pview
    ;
    int64_t v208 = v206 * v207;
    // pto: %cmp_sparse_indices_inline2524__rv_v4_pview
    ;
    pto::Shape<1, 1, 1, -1, 512> v209 = pto::Shape<1, 1, 1, -1, 512>(v204, v205, v206, v71, v20);
    // pto: %cmp_sparse_indices_inline2524__rv_v4_pview
    ;
    pto::Stride<-1, -1, -1, -1, -1> v210 = pto::Stride<-1, -1, -1, -1, -1>(v205 * v208, v208, v207, v20, v21);
    // pto: %cmp_sparse_indices_inline2524__rv_v4_pview
    ;
    GlobalTensor<int32_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v211 = GlobalTensor<int32_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v203 + (v202 + v201 * v20 + v26 * v21), v209, v210);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    TLOAD(v199, v211);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
    // pto: %c_block_slots_inline2586__ssa_v0
    ;
    Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v212 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v20);
    // pto: %c_block_slots_inline2586__ssa_v0
    ;
    uint64_t v213 = (uint64_t) v16;
    TASSIGN(v212, v213);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
    RoundMode v214 = RoundMode::CAST_ROUND;
    SaturationMode v215 = SaturationMode::OFF;
    TCVT(v212, v199, v214, v215);
    // pto: %t__tmp_v311
    ;
    Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v216 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v20);
    // pto: %t__tmp_v311
    ;
    uint64_t v217 = (uint64_t) v16;
    TASSIGN(v216, v217);
    pipe_barrier(PIPE_V);
    TADDS(v216, v212, v29);
    // pto: %t__tmp_v312
    ;
    Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v218 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v20);
    // pto: %t__tmp_v312
    ;
    uint64_t v219 = (uint64_t) v16;
    TASSIGN(v218, v219);
    pipe_barrier(PIPE_V);
    TMAXS(v218, v216, v30);
    // pto: %t__tmp_v313
    ;
    Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v220 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v20);
    // pto: %t__tmp_v313
    ;
    uint64_t v221 = (uint64_t) v16;
    TASSIGN(v220, v221);
    pipe_barrier(PIPE_V);
    TMINS(v220, v218, v29);
    // pto: %c_blk_valid_inline2529__ssa_v0
    ;
    Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v222 = Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v21);
    // pto: %c_blk_valid_inline2529__ssa_v0
    ;
    uint64_t v223 = (uint64_t) v17;
    TASSIGN(v222, v223);
    pipe_barrier(PIPE_V);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
    TROWMAX(v222, v220, v197);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
    for (int64_t v224 = v26; v224 < v71; v224 += v21) {
      // pto: %t__tmp_v314
      ;
      float v225 = v222.GetValue(v224);
      // pto: %72
      ;
      int32_t v226 = (int32_t) v225;
      // pto: %73, %74, %75
      ;
      int64_t v227 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) ((int64_t) (uint64_t) v69 + (uint64_t) v224) * (uint64_t) v23) + (uint64_t) v21);
      (v6)[v227] = v226;
    };
    set_flag(PIPE_S, PIPE_V, EVENT_ID0);
  }
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  pipe_barrier(PIPE_ALL);
  dcci((__gm__ void*)0, cache_line_t::ENTIRE_DATA_CACHE);
  dsb((mem_dsb_t)0);
  #endif // __DAV_VEC__

  pipe_barrier(PIPE_ALL);
  dcci((__gm__ void*)0, cache_line_t::ENTIRE_DATA_CACHE);
  dsb((mem_dsb_t)0);
  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}