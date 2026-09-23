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
  // pto: %cmp_sparse_indices_inline2528__ssa_v0_view
  const int64_t v38 = 1;
  // pto: %cmp_sparse_indices_inline2528__ssa_v0_view
  const int64_t v39 = 1;
  // pto: %cmp_sparse_indices_inline2528__ssa_v0_view
  const int64_t v40 = 1;
  // pto: %cmp_sparse_indices_inline2528__ssa_v0_view
  int64_t v41 = (int64_t) v8;
  // pto: %cmp_sparse_indices_inline2528__ssa_v0_view
  int64_t v42 = v41 * v20;
  // pto: %cmp_sparse_indices_inline2528__ssa_v0_view
  int64_t v43 = v40 * v42;
  // pto: %cmp_sparse_indices_inline2528__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v44 = pto::Shape<1, 1, 1, -1, -1>(v38, v39, v40, v41, v20);
  // pto: %cmp_sparse_indices_inline2528__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v45 = pto::Stride<-1, -1, -1, -1, -1>(v39 * v43, v43, v42, v20, v21);
  // pto: %cmp_sparse_indices_inline2528__ssa_v0_view
  GlobalTensor<int32_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v46 = GlobalTensor<int32_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v1, v44, v45);
  // pto: %sparse_bias_inline2514__ssa_v0_view
  const int64_t v47 = 1;
  // pto: %sparse_bias_inline2514__ssa_v0_view
  const int64_t v48 = 1;
  // pto: %sparse_bias_inline2514__ssa_v0_view
  const int64_t v49 = 1;
  // pto: %sparse_bias_inline2514__ssa_v0_view
  int64_t v50 = (int64_t) v8;
  // pto: %sparse_bias_inline2514__ssa_v0_view
  int64_t v51 = v50 * v22;
  // pto: %sparse_bias_inline2514__ssa_v0_view
  int64_t v52 = v49 * v51;
  // pto: %sparse_bias_inline2514__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v53 = pto::Shape<1, 1, 1, -1, -1>(v47, v48, v49, v50, v22);
  // pto: %sparse_bias_inline2514__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v54 = pto::Stride<-1, -1, -1, -1, -1>(v48 * v52, v52, v51, v22, v21);
  // pto: %sparse_bias_inline2514__ssa_v0_view
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
  // pto: %v_valid_inline2547__phi_v5
  Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v65 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v21, v25);
  // pto: %v_valid_inline2547__phi_v5
  uint64_t v66 = (uint64_t) v17;
  TASSIGN(v65, v66);
  // pto: %v_valid_inline2547__phi_v4
  Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v67 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v21, v25);
  // pto: %v_valid_inline2547__phi_v4
  uint64_t v68 = (uint64_t) v17;
  TASSIGN(v67, v68);
  // pto: %plan_worker_inline2520__ssa_v0, %26
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_S, PIPE_V, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  for (int64_t v69 = (int64_t) ((uint64_t) ((int64_t) v12) * (uint64_t) v24); v69 < v8; v69 += v25) {
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
    for (int64_t v70 = v26; v70 < v24; v70 += v21) {
      // pto: %27
      ;
      int64_t v71 = (int64_t) ((uint64_t) v69 + (uint64_t) v70);
      // pto: %29
      ;
      int64_t v72 = (v3)[v71];
      // pto: %t__tile
      ;
      Tile<TileType::Vec, int32_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v73 = Tile<TileType::Vec, int32_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v21, v20);
      // pto: %t__tile
      ;
      uint64_t v74 = (uint64_t) v15;
      TASSIGN(v73, v74);
      // pto: %31
      ;
      int64_t v75 = v71 < v26 ? v26 : v71;
      // pto: %idx_topk__ssa_v3_pview
      ;
      __gm__ int32_t* v76 = PTOAS__GLOBAL_TENSOR_DATA(v64);
      // pto: %idx_topk__ssa_v3_pview
      ;
      const int64_t v77 = 0;
      // pto: %idx_topk__ssa_v3_pview
      ;
      const int64_t v78 = 512;
      // pto: %idx_topk__ssa_v3_pview
      ;
      pto::Shape<1, 1, 1, 1, 512> v79 = pto::Shape<1, 1, 1, 1, 512>();
      // pto: %idx_topk__ssa_v3_pview
      ;
      pto::Stride<512, 512, 512, 512, 1> v80 = pto::Stride<512, 512, 512, 512, 1>();
      // pto: %idx_topk__ssa_v3_pview
      ;
      GlobalTensor<int32_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<512, 512, 512, 512, 1>, pto::Layout::ND> v81 = GlobalTensor<int32_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<512, 512, 512, 512, 1>, pto::Layout::ND>(v76 + (v77 + v75 * v78), v79, v80);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
      TLOAD(v73, v81);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      // pto: %c_raw_inline2587__tile
      ;
      Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v82 = Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v21, v20);
      // pto: %c_raw_inline2587__tile
      ;
      uint64_t v83 = (uint64_t) v15;
      TASSIGN(v82, v83);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      RoundMode v84 = RoundMode::CAST_ROUND;
      SaturationMode v85 = SaturationMode::OFF;
      TCVT(v82, v73, v84, v85);
      // pto: %32
      ;
      int64_t v86 = (int64_t) ((uint64_t) v72 + (uint64_t) v21);
      // pto: %33, %34, %35
      ;
      float v87 = (float) ((int32_t) v86 / v28);
      // pto: %0
      ;
      Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v88 = Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v21, v20);
      // pto: %0
      ;
      uint64_t v89 = (uint64_t) v16;
      TASSIGN(v88, v89);
      pipe_barrier(PIPE_V);
      TADDS(v88, v82, v29);
      // pto: %1
      ;
      Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v90 = Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v21, v20);
      // pto: %1
      ;
      uint64_t v91 = (uint64_t) v16;
      TASSIGN(v90, v91);
      pipe_barrier(PIPE_V);
      TMAXS(v90, v88, v30);
      // pto: %c_ge_inline2532__tile
      ;
      Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v92 = Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v21, v20);
      // pto: %c_ge_inline2532__tile
      ;
      uint64_t v93 = (uint64_t) v16;
      TASSIGN(v92, v93);
      pipe_barrier(PIPE_V);
      TMINS(v92, v90, v29);
      // pto: %2
      ;
      Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v94 = Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v21, v20);
      // pto: %2
      ;
      uint64_t v95 = (uint64_t) v17;
      TASSIGN(v94, v95);
      TNEG(v94, v82);
      // pto: %3
      ;
      Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v96 = Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v21, v20);
      // pto: %3
      ;
      uint64_t v97 = (uint64_t) v17;
      TASSIGN(v96, v97);
      pipe_barrier(PIPE_V);
      TADDS(v96, v94, v87);
      // pto: %4
      ;
      Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v98 = Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v21, v20);
      // pto: %4
      ;
      uint64_t v99 = (uint64_t) v17;
      TASSIGN(v98, v99);
      pipe_barrier(PIPE_V);
      TMAXS(v98, v96, v30);
      // pto: %c_lt_inline2569__tile
      ;
      Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v100 = Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v21, v20);
      // pto: %c_lt_inline2569__tile
      ;
      uint64_t v101 = (uint64_t) v17;
      TASSIGN(v100, v101);
      pipe_barrier(PIPE_V);
      TMINS(v100, v98, v29);
      // pto: %c_mask_inline2540__tile
      ;
      Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v102 = Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v21, v20);
      // pto: %c_mask_inline2540__tile
      ;
      uint64_t v103 = (uint64_t) v16;
      TASSIGN(v102, v103);
      pipe_barrier(PIPE_V);
      TMUL(v102, v92, v100);
      // pto: %5
      ;
      Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v104 = Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v21, v20);
      // pto: %5
      ;
      uint64_t v105 = (uint64_t) v15;
      TASSIGN(v104, v105);
      TADDS(v104, v82, v29);
      // pto: %6
      ;
      Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v106 = Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v21, v20);
      // pto: %6
      ;
      uint64_t v107 = (uint64_t) v15;
      TASSIGN(v106, v107);
      pipe_barrier(PIPE_V);
      TMUL(v106, v102, v104);
      // pto: %c_out_inline2516__tile
      ;
      Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v108 = Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v21, v20);
      // pto: %c_out_inline2516__tile
      ;
      uint64_t v109 = (uint64_t) v15;
      TASSIGN(v108, v109);
      pipe_barrier(PIPE_V);
      TSUBS(v108, v106, v29);
      // pto: %7
      ;
      Tile<TileType::Vec, int32_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v110 = Tile<TileType::Vec, int32_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v21, v20);
      // pto: %7
      ;
      uint64_t v111 = (uint64_t) v16;
      TASSIGN(v110, v111);
      pipe_barrier(PIPE_V);
      RoundMode v112 = RoundMode::CAST_ROUND;
      SaturationMode v113 = SaturationMode::ON;
      TCVT(v110, v108, v112, v113);
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      // pto: %cmp_sparse_indices_inline2528__iter_v3_pview
      ;
      __gm__ int32_t* v114 = PTOAS__GLOBAL_TENSOR_DATA(v46);
      // pto: %cmp_sparse_indices_inline2528__iter_v3_pview
      ;
      const int64_t v115 = 0;
      // pto: %cmp_sparse_indices_inline2528__iter_v3_pview
      ;
      const int64_t v116 = 512;
      // pto: %cmp_sparse_indices_inline2528__iter_v3_pview
      ;
      pto::Shape<1, 1, 1, 1, 512> v117 = pto::Shape<1, 1, 1, 1, 512>();
      // pto: %cmp_sparse_indices_inline2528__iter_v3_pview
      ;
      pto::Stride<512, 512, 512, 512, 1> v118 = pto::Stride<512, 512, 512, 512, 1>();
      // pto: %cmp_sparse_indices_inline2528__iter_v3_pview
      ;
      GlobalTensor<int32_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<512, 512, 512, 512, 1>, pto::Layout::ND> v119 = GlobalTensor<int32_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<512, 512, 512, 512, 1>, pto::Layout::ND>(v114 + (v115 + v75 * v116), v117, v118);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      TSTORE(v119, v110);
      set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
      // pto: %38
      ;
      int64_t v120 = v86 < v25 ? v86 : v25;
      // pto: %39, %40
      ;
      int64_t v121 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) v72 - (uint64_t) v120) + (uint64_t) v21);
      // pto: %t__ci_tmp_v0
      ;
      Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v122 = Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v21, v32);
      // pto: %t__ci_tmp_v0
      ;
      uint64_t v123 = (uint64_t) v16;
      TASSIGN(v122, v123);
      // pto: %8
      ;
      Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v124 = Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v21, v25);
      // pto: %8
      ;
      uint64_t v125 = (uint64_t) v17;
      TASSIGN(v124, v125);
      // pto: %ci_tmp_view
      ;
      Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, 1, 192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v126;
      TRESHAPE(v126, v122);
      // pto: %ci_dst_view
      ;
      Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, 1, 128, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v127;
      TRESHAPE(v127, v124);
      wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
      TCI<Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, 1, 128, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, 1, 192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, int32_t, 0>(v127, v33, v126);
      set_flag(PIPE_S, PIPE_V, EVENT_ID1);
      // pto: %v_columns_inline2546__tile
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v128 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v21, v25);
      // pto: %v_columns_inline2546__tile
      ;
      uint64_t v129 = (uint64_t) v16;
      TASSIGN(v128, v129);
      wait_flag(PIPE_S, PIPE_V, EVENT_ID1);
      RoundMode v130 = RoundMode::CAST_ROUND;
      SaturationMode v131 = SaturationMode::OFF;
      TCVT(v128, v124, v130, v131);
      // pto: %v_valid_inline2547__tile
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v132 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v21, v25);
      // pto: %v_valid_inline2547__tile
      ;
      uint64_t v133 = (uint64_t) v17;
      TASSIGN(v132, v133);
      pipe_barrier(PIPE_V);
      TEXPANDS(v132, v30);
      // pto: %v_block_valid_inline2525__rv_v2
      ;
      int32_t v134;
      v134 = v33;
      for (int64_t v135 = v26; v135 < v34; v135 += v21) {
        // pto: %43, %44, %41
        ;
        int64_t v136 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) v135 * (uint64_t) v31) - (uint64_t) (v121 % v31));
        // pto: %45
        ;
        int64_t v137 = v136 < v26 ? v26 : v136;
        // pto: %48
        ;
        int64_t v138 = (int64_t) ((uint64_t) v136 + (uint64_t) v31);
        // pto: %49
        ;
        int64_t v139 = v138 < v120 ? v138 : v120;
        // pto: %50
        ;
        // pto: %v_block_valid_inline2525__phi_v5
        ;
        int32_t v140;
        if (v137 < v139) {
          // pto: %28, %53, %54, %51, %52, %v_page_inline2533__tile
          ;
          int32_t v141 = (v5)[(int64_t) ((uint64_t) ((int64_t) (uint64_t) (v71 / v27) * (uint64_t) v11) + (uint64_t) ((int64_t) ((uint64_t) v121 + (uint64_t) v137) / v31))];
          // pto: %55, %56
          ;
          bool v142 = (int64_t) v141 >= v26;
          // pto: %v_block_valid_inline2525__phi_v4
          ;
          int32_t v143 = v142 ? v14 : v134;
          if (v142) {
            // pto: %58, %59
            ;
            float v144 = (float) ((int32_t) v137);
            // pto: %60, %61
            ;
            float v145 = (float) ((int32_t) v139);
            // pto: %9
            ;
            Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v146 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v21, v25);
            // pto: %9
            ;
            uint64_t v147 = (uint64_t) v18;
            TASSIGN(v146, v147);
            pipe_barrier(PIPE_V);
            TSUBS(v146, v128, v144);
            // pto: %10
            ;
            Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v148 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v21, v25);
            // pto: %10
            ;
            uint64_t v149 = (uint64_t) v18;
            TASSIGN(v148, v149);
            pipe_barrier(PIPE_V);
            TADDS(v148, v146, v29);
            // pto: %11
            ;
            Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v150 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v21, v25);
            // pto: %11
            ;
            uint64_t v151 = (uint64_t) v18;
            TASSIGN(v150, v151);
            pipe_barrier(PIPE_V);
            TMAXS(v150, v148, v30);
            // pto: %v_ge_inline2517__tile
            ;
            Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v152 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v21, v25);
            // pto: %v_ge_inline2517__tile
            ;
            uint64_t v153 = (uint64_t) v18;
            TASSIGN(v152, v153);
            pipe_barrier(PIPE_V);
            TMINS(v152, v150, v29);
            // pto: %12
            ;
            Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v154 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v21, v25);
            // pto: %12
            ;
            uint64_t v155 = (uint64_t) v19;
            TASSIGN(v154, v155);
            TNEG(v154, v128);
            // pto: %13
            ;
            Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v156 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v21, v25);
            // pto: %13
            ;
            uint64_t v157 = (uint64_t) v19;
            TASSIGN(v156, v157);
            pipe_barrier(PIPE_V);
            TADDS(v156, v154, v145);
            // pto: %14
            ;
            Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v158 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v21, v25);
            // pto: %14
            ;
            uint64_t v159 = (uint64_t) v19;
            TASSIGN(v158, v159);
            pipe_barrier(PIPE_V);
            TMAXS(v158, v156, v30);
            // pto: %v_lt_inline2496__tile
            ;
            Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v160 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v21, v25);
            // pto: %v_lt_inline2496__tile
            ;
            uint64_t v161 = (uint64_t) v19;
            TASSIGN(v160, v161);
            pipe_barrier(PIPE_V);
            TMINS(v160, v158, v29);
            // pto: %15
            ;
            Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v162 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v21, v25);
            // pto: %15
            ;
            uint64_t v163 = (uint64_t) v18;
            TASSIGN(v162, v163);
            pipe_barrier(PIPE_V);
            TMUL(v162, v152, v160);
            // pto: %16
            ;
            Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v164 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v21, v25);
            // pto: %16
            ;
            uint64_t v165 = (uint64_t) v17;
            TASSIGN(v164, v165);
            pipe_barrier(PIPE_V);
            TADD(v164, v132, v162);
          };
          v140 = v143;
        } else {
          v140 = v134;
        };
        v134 = v140;
      };
      // pto: %62
      ;
      int64_t v166 = (int64_t) ((uint64_t) v71 * (uint64_t) v23);
      (v6)[v166] = v134;
      // pto: %17
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v167 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v21, v25);
      // pto: %17
      ;
      uint64_t v168 = (uint64_t) v16;
      TASSIGN(v167, v168);
      pipe_barrier(PIPE_V);
      TSUBS(v167, v132, v29);
      // pto: %18
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v169 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v21, v25);
      // pto: %18
      ;
      uint64_t v170 = (uint64_t) v16;
      TASSIGN(v169, v170);
      pipe_barrier(PIPE_V);
      TMULS(v169, v167, v35);
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
      // pto: %sparse_bias_inline2514__iter_v3_pview
      ;
      __gm__ float* v171 = PTOAS__GLOBAL_TENSOR_DATA(v55);
      // pto: %sparse_bias_inline2514__iter_v3_pview
      ;
      const int64_t v172 = 0;
      // pto: %sparse_bias_inline2514__iter_v3_pview
      ;
      const int64_t v173 = 1024;
      // pto: %sparse_bias_inline2514__iter_v3_pview
      ;
      pto::Shape<1, 1, 1, 1, 128> v174 = pto::Shape<1, 1, 1, 1, 128>();
      // pto: %sparse_bias_inline2514__iter_v3_pview
      ;
      pto::Stride<1024, 1024, 1024, 1024, 1> v175 = pto::Stride<1024, 1024, 1024, 1024, 1>();
      // pto: %sparse_bias_inline2514__iter_v3_pview
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<1024, 1024, 1024, 1024, 1>, pto::Layout::ND> v176 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<1024, 1024, 1024, 1024, 1>, pto::Layout::ND>(v171 + (v172 + v75 * v173), v174, v175);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
      TSTORE(v176, v169);
      set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
      // pto: %19
      ;
      Tile<TileType::Vec, float, 1, 384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v177 = Tile<TileType::Vec, float, 1, 384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v21, v36);
      // pto: %19
      ;
      uint64_t v178 = (uint64_t) v16;
      TASSIGN(v177, v178);
      wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
      TEXPANDS(v177, v37);
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
      // pto: %sparse_bias_inline2514__tile_pview
      ;
      __gm__ float* v179 = PTOAS__GLOBAL_TENSOR_DATA(v55);
      // pto: %sparse_bias_inline2514__tile_pview
      ;
      const int64_t v180 = 128;
      // pto: %sparse_bias_inline2514__tile_pview
      ;
      const int64_t v181 = 1024;
      // pto: %sparse_bias_inline2514__tile_pview
      ;
      pto::Shape<1, 1, 1, 1, 384> v182 = pto::Shape<1, 1, 1, 1, 384>();
      // pto: %sparse_bias_inline2514__tile_pview
      ;
      pto::Stride<1024, 1024, 1024, 1024, 1> v183 = pto::Stride<1024, 1024, 1024, 1024, 1>();
      // pto: %sparse_bias_inline2514__tile_pview
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 1, 384>, pto::Stride<1024, 1024, 1024, 1024, 1>, pto::Layout::ND> v184 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 384>, pto::Stride<1024, 1024, 1024, 1024, 1>, pto::Layout::ND>(v179 + (v180 + v75 * v181), v182, v183);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
      TSTORE(v184, v177);
      // pto: %20
      ;
      Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v185 = Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v21, v20);
      // pto: %20
      ;
      uint64_t v186 = (uint64_t) v15;
      TASSIGN(v185, v186);
      TMINS(v185, v108, v30);
      // pto: %21
      ;
      Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v187 = Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v21, v20);
      // pto: %21
      ;
      uint64_t v188 = (uint64_t) v15;
      TASSIGN(v187, v188);
      pipe_barrier(PIPE_V);
      TMULS(v187, v185, v35);
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID3);
      // pto: %69
      ;
      __gm__ float* v189 = PTOAS__GLOBAL_TENSOR_DATA(v55);
      // pto: %69
      ;
      const int64_t v190 = 512;
      // pto: %69
      ;
      const int64_t v191 = 1024;
      // pto: %69
      ;
      pto::Shape<1, 1, 1, 1, 512> v192 = pto::Shape<1, 1, 1, 1, 512>();
      // pto: %69
      ;
      pto::Stride<1024, 1024, 1024, 1024, 1> v193 = pto::Stride<1024, 1024, 1024, 1024, 1>();
      // pto: %69
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<1024, 1024, 1024, 1024, 1>, pto::Layout::ND> v194 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<1024, 1024, 1024, 1024, 1>, pto::Layout::ND>(v189 + (v190 + v75 * v191), v192, v193);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID3);
      pipe_barrier(PIPE_MTE3);
      TSTORE(v194, v187);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    };
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    // pto: %22
    ;
    Tile<TileType::Vec, int32_t, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v195 = Tile<TileType::Vec, int32_t, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v24, v20);
    // pto: %22
    ;
    uint64_t v196 = (uint64_t) v15;
    TASSIGN(v195, v196);
    // pto: %70
    ;
    int64_t v197 = v69 < v26 ? v26 : v69;
    // pto: %cmp_sparse_indices_inline2528__rv_v4_pview
    ;
    __gm__ int32_t* v198 = PTOAS__GLOBAL_TENSOR_DATA(v46);
    // pto: %cmp_sparse_indices_inline2528__rv_v4_pview
    ;
    const int64_t v199 = 0;
    // pto: %cmp_sparse_indices_inline2528__rv_v4_pview
    ;
    const int64_t v200 = 512;
    // pto: %cmp_sparse_indices_inline2528__rv_v4_pview
    ;
    pto::Shape<1, 1, 1, 8, 512> v201 = pto::Shape<1, 1, 1, 8, 512>();
    // pto: %cmp_sparse_indices_inline2528__rv_v4_pview
    ;
    pto::Stride<4096, 4096, 4096, 512, 1> v202 = pto::Stride<4096, 4096, 4096, 512, 1>();
    // pto: %cmp_sparse_indices_inline2528__rv_v4_pview
    ;
    GlobalTensor<int32_t, pto::Shape<1, 1, 1, 8, 512>, pto::Stride<4096, 4096, 4096, 512, 1>, pto::Layout::ND> v203 = GlobalTensor<int32_t, pto::Shape<1, 1, 1, 8, 512>, pto::Stride<4096, 4096, 4096, 512, 1>, pto::Layout::ND>(v198 + (v199 + v197 * v200), v201, v202);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    TLOAD(v195, v203);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
    // pto: %c_block_slots_inline2599__tile
    ;
    Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v204 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v24, v20);
    // pto: %c_block_slots_inline2599__tile
    ;
    uint64_t v205 = (uint64_t) v15;
    TASSIGN(v204, v205);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
    RoundMode v206 = RoundMode::CAST_ROUND;
    SaturationMode v207 = SaturationMode::OFF;
    TCVT(v204, v195, v206, v207);
    // pto: %23
    ;
    Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v208 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v24, v20);
    // pto: %23
    ;
    uint64_t v209 = (uint64_t) v15;
    TASSIGN(v208, v209);
    pipe_barrier(PIPE_V);
    TADDS(v208, v204, v29);
    // pto: %24
    ;
    Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v210 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v24, v20);
    // pto: %24
    ;
    uint64_t v211 = (uint64_t) v15;
    TASSIGN(v210, v211);
    pipe_barrier(PIPE_V);
    TMAXS(v210, v208, v30);
    // pto: %25
    ;
    Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v212 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v24, v20);
    // pto: %25
    ;
    uint64_t v213 = (uint64_t) v15;
    TASSIGN(v212, v213);
    pipe_barrier(PIPE_V);
    TMINS(v212, v210, v29);
    // pto: %tmp_tile
    ;
    Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v214 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v24, v20);
    // pto: %tmp_tile
    ;
    uint64_t v215 = (uint64_t) v16;
    TASSIGN(v214, v215);
    // pto: %c_blk_valid_inline2483__tile
    ;
    Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v216 = Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v24, v21);
    // pto: %c_blk_valid_inline2483__tile
    ;
    uint64_t v217 = (uint64_t) v17;
    TASSIGN(v216, v217);
    pipe_barrier(PIPE_V);
    TROWMAX(v216, v212, v214);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
    for (int64_t v218 = v26; v218 < v24; v218 += v21) {
      // pto: %72
      ;
      float v219 = v216.GetValue(v218);
      // pto: %75
      ;
      int32_t v220 = (int32_t) v219;
      // pto: %76, %77, %78
      ;
      int64_t v221 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) ((int64_t) (uint64_t) v69 + (uint64_t) v218) * (uint64_t) v23) + (uint64_t) v21);
      (v6)[v221] = v220;
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