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

AICORE void rope_cs(__gm__ int32_t* v1, __gm__ float* v2, __gm__ float* v3, int64_t v4, int64_t v5, int64_t v6) {
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  // pto: %c0_i64
  const int64_t v7 = 0;
  // pto: %c4352_i64
  const int64_t v8 = 4352;
  // pto: %c4096_i64
  const int64_t v9 = 4096;
  // pto: %c16_index
  const int64_t v10 = 16;
  // pto: %c64_index
  const int64_t v11 = 64;
  // pto: %c1_index
  const int64_t v12 = 1;
  // pto: %c384_index
  const int64_t v13 = 384;
  // pto: %cst_7
  const float v14 = 1.0f;
  // pto: %c192_index
  const int64_t v15 = 192;
  // pto: %c0_i32
  const int32_t v16 = 0;
  // pto: %cst_10
  const float v17 = 0.5f;
  // pto: %cst_11
  const float v18 = 2.0f;
  // pto: %c0_index
  const int64_t v19 = 0;
  // pto: %c8_index
  const int64_t v20 = 8;
  // pto: %rope_swap_idx_inline2487__ssa_v0_view
  const int64_t v21 = 1;
  // pto: %rope_swap_idx_inline2487__ssa_v0_view
  const int64_t v22 = 1;
  // pto: %rope_swap_idx_inline2487__ssa_v0_view
  const int64_t v23 = 1;
  // pto: %rope_swap_idx_inline2487__ssa_v0_view
  int64_t v24 = v10 * v11;
  // pto: %rope_swap_idx_inline2487__ssa_v0_view
  int64_t v25 = v23 * v24;
  // pto: %rope_swap_idx_inline2487__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v26 = pto::Shape<1, 1, 1, -1, -1>(v21, v22, v23, v10, v11);
  // pto: %rope_swap_idx_inline2487__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v27 = pto::Stride<-1, -1, -1, -1, -1>(v22 * v25, v25, v24, v11, v12);
  // pto: %rope_swap_idx_inline2487__ssa_v0_view
  GlobalTensor<int32_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v28 = GlobalTensor<int32_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v1, v26, v27);
  // pto: %freqs_sin__ssa_v0_view
  const int64_t v29 = 1;
  // pto: %freqs_sin__ssa_v0_view
  const int64_t v30 = 1;
  // pto: %freqs_sin__ssa_v0_view
  const int64_t v31 = 1;
  // pto: %freqs_sin__ssa_v0_view
  int64_t v32 = (int64_t) v6;
  // pto: %freqs_sin__ssa_v0_view
  int64_t v33 = v32 * v11;
  // pto: %freqs_sin__ssa_v0_view
  int64_t v34 = v31 * v33;
  // pto: %freqs_sin__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v35 = pto::Shape<1, 1, 1, -1, -1>(v29, v30, v31, v32, v11);
  // pto: %freqs_sin__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v36 = pto::Stride<-1, -1, -1, -1, -1>(v30 * v34, v34, v33, v11, v12);
  // pto: %freqs_sin__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v37 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v2, v35, v36);
  // pto: %rope_sin_signed_inline2481__ssa_v0_view
  const int64_t v38 = 1;
  // pto: %rope_sin_signed_inline2481__ssa_v0_view
  const int64_t v39 = 1;
  // pto: %rope_sin_signed_inline2481__ssa_v0_view
  const int64_t v40 = 1;
  // pto: %rope_sin_signed_inline2481__ssa_v0_view
  int64_t v41 = v13 * v11;
  // pto: %rope_sin_signed_inline2481__ssa_v0_view
  int64_t v42 = v40 * v41;
  // pto: %rope_sin_signed_inline2481__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v43 = pto::Shape<1, 1, 1, -1, -1>(v38, v39, v40, v13, v11);
  // pto: %rope_sin_signed_inline2481__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v44 = pto::Stride<-1, -1, -1, -1, -1>(v39 * v42, v42, v41, v11, v12);
  // pto: %rope_sin_signed_inline2481__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v45 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v3, v43, v44);
  // pto: %sw_ones_inline2557__tile
  Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v46 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v10, v11);
  // pto: %sw_ones_inline2557__tile
  uint64_t v47 = (uint64_t) v7;
  TASSIGN(v46, v47);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  TEXPANDS(v46, v14);
  // pto: %t__ci_tmp_v0
  Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v48 = Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v12, v15);
  // pto: %t__ci_tmp_v0
  uint64_t v49 = (uint64_t) v8;
  TASSIGN(v48, v49);
  // pto: %t__tile
  Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v50 = Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v12, v11);
  // pto: %t__tile
  uint64_t v51 = (uint64_t) v9;
  TASSIGN(v50, v51);
  // pto: %ci_tmp_view
  Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, 1, 192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v52;
  TRESHAPE(v52, v48);
  // pto: %ci_dst_view
  Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v53;
  TRESHAPE(v53, v50);
  TCI<Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, 1, 192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, int32_t, 0>(v53, v16, v52);
  set_flag(PIPE_S, PIPE_V, EVENT_ID0);
  // pto: %sw_idx_f_inline2528__tile
  Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v54 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v12, v11);
  // pto: %sw_idx_f_inline2528__tile
  uint64_t v55 = (uint64_t) v8;
  TASSIGN(v54, v55);
  wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
  RoundMode v56 = RoundMode::CAST_ROUND;
  SaturationMode v57 = SaturationMode::OFF;
  TCVT(v54, v50, v56, v57);
  // pto: %sw_col_inline2478__tile
  Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v58 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v10, v11);
  // pto: %sw_col_inline2478__tile
  uint64_t v59 = (uint64_t) v7;
  TASSIGN(v58, v59);
  pipe_barrier(PIPE_V);
  TCOLEXPANDMUL(v58, v46, v54);
  // pto: %0
  Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v60 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v10, v11);
  // pto: %0
  uint64_t v61 = (uint64_t) v8;
  TASSIGN(v60, v61);
  pipe_barrier(PIPE_V);
  TMULS(v60, v58, v17);
  // pto: %sw_dup_i32_inline2477__tile
  Tile<TileType::Vec, int32_t, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v62 = Tile<TileType::Vec, int32_t, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v10, v11);
  // pto: %sw_dup_i32_inline2477__tile
  uint64_t v63 = (uint64_t) v8;
  TASSIGN(v62, v63);
  pipe_barrier(PIPE_V);
  RoundMode v64 = RoundMode::CAST_TRUNC;
  SaturationMode v65 = SaturationMode::ON;
  TCVT(v62, v60, v64, v65);
  // pto: %sw_dup_f_inline2476__tile
  Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v66 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v10, v11);
  // pto: %sw_dup_f_inline2476__tile
  uint64_t v67 = (uint64_t) v8;
  TASSIGN(v66, v67);
  pipe_barrier(PIPE_V);
  RoundMode v68 = RoundMode::CAST_ROUND;
  SaturationMode v69 = SaturationMode::OFF;
  TCVT(v66, v62, v68, v69);
  // pto: %1
  Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v70 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v10, v11);
  // pto: %1
  uint64_t v71 = (uint64_t) v8;
  TASSIGN(v70, v71);
  pipe_barrier(PIPE_V);
  TMULS(v70, v66, v18);
  // pto: %sw_lane_inline2480__tile
  Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v72 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v10, v11);
  // pto: %sw_lane_inline2480__tile
  uint64_t v73 = (uint64_t) v8;
  TASSIGN(v72, v73);
  pipe_barrier(PIPE_V);
  TSUB(v72, v58, v70);
  // pto: %2
  Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v74 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v10, v11);
  // pto: %2
  uint64_t v75 = (uint64_t) v7;
  TASSIGN(v74, v75);
  pipe_barrier(PIPE_V);
  TADDS(v74, v58, v14);
  // pto: %3
  Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v76 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v10, v11);
  // pto: %3
  uint64_t v77 = (uint64_t) v8;
  TASSIGN(v76, v77);
  TMULS(v76, v72, v18);
  // pto: %sw_swap_f_inline2475__tile
  Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v78 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v10, v11);
  // pto: %sw_swap_f_inline2475__tile
  uint64_t v79 = (uint64_t) v7;
  TASSIGN(v78, v79);
  pipe_barrier(PIPE_V);
  TSUB(v78, v74, v76);
  set_flag(PIPE_V, PIPE_S, EVENT_ID0);
  // pto: %4
  Tile<TileType::Vec, int32_t, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v80 = Tile<TileType::Vec, int32_t, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v10, v11);
  // pto: %4
  uint64_t v81 = (uint64_t) v7;
  TASSIGN(v80, v81);
  pipe_barrier(PIPE_V);
  RoundMode v82 = RoundMode::CAST_ROUND;
  SaturationMode v83 = SaturationMode::ON;
  TCVT(v80, v78, v82, v83);
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  // pto: %rope_swap_idx_inline2487__ssa_v0_pview
  __gm__ int32_t* v84 = PTOAS__GLOBAL_TENSOR_DATA(v28);
  // pto: %rope_swap_idx_inline2487__ssa_v0_pview
  pto::Shape<1, 1, 1, 16, 64> v85 = pto::Shape<1, 1, 1, 16, 64>();
  // pto: %rope_swap_idx_inline2487__ssa_v0_pview
  pto::Stride<1024, 1024, 1024, 64, 1> v86 = pto::Stride<1024, 1024, 1024, 64, 1>();
  // pto: %rope_swap_idx_inline2487__ssa_v0_pview
  GlobalTensor<int32_t, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<1024, 1024, 1024, 64, 1>, pto::Layout::ND> v87 = GlobalTensor<int32_t, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<1024, 1024, 1024, 64, 1>, pto::Layout::ND>(v84, v85, v86);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  TSTORE(v87, v80);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  // pto: %cs_ones_inline2474__ssa_v0
  Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v88 = Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v20, v11);
  // pto: %cs_ones_inline2474__ssa_v0
  uint64_t v89 = (uint64_t) v7;
  TASSIGN(v88, v89);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  TEXPANDS(v88, v14);
  // pto: %t__ci_tmp_v1
  Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v90 = Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v12, v15);
  // pto: %t__ci_tmp_v1
  uint64_t v91 = (uint64_t) v8;
  TASSIGN(v90, v91);
  // pto: %t__tmp_v334
  Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v92 = Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v12, v11);
  // pto: %t__tmp_v334
  uint64_t v93 = (uint64_t) v9;
  TASSIGN(v92, v93);
  // pto: %5
  Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, 1, 192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v94;
  TRESHAPE(v94, v90);
  // pto: %6
  Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v95;
  TRESHAPE(v95, v92);
  wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
  TCI<Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, 1, 192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, int32_t, 0>(v95, v16, v94);
  set_flag(PIPE_S, PIPE_V, EVENT_ID1);
  // pto: %cs_idx_f_inline2604__ssa_v0
  Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v96 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v12, v11);
  // pto: %cs_idx_f_inline2604__ssa_v0
  uint64_t v97 = (uint64_t) v8;
  TASSIGN(v96, v97);
  wait_flag(PIPE_S, PIPE_V, EVENT_ID1);
  RoundMode v98 = RoundMode::CAST_ROUND;
  SaturationMode v99 = SaturationMode::OFF;
  TCVT(v96, v92, v98, v99);
  // pto: %cs_col_inline2619__ssa_v0
  Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v100 = Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v20, v11);
  // pto: %cs_col_inline2619__ssa_v0
  uint64_t v101 = (uint64_t) v7;
  TASSIGN(v100, v101);
  pipe_barrier(PIPE_V);
  TCOLEXPANDMUL(v100, v88, v96);
  // pto: %t__tmp_v335
  Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v102 = Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v20, v11);
  // pto: %t__tmp_v335
  uint64_t v103 = (uint64_t) v8;
  TASSIGN(v102, v103);
  pipe_barrier(PIPE_V);
  TMULS(v102, v100, v17);
  // pto: %cs_dup_i32_inline2566__ssa_v0
  Tile<TileType::Vec, int32_t, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v104 = Tile<TileType::Vec, int32_t, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v20, v11);
  // pto: %cs_dup_i32_inline2566__ssa_v0
  uint64_t v105 = (uint64_t) v8;
  TASSIGN(v104, v105);
  pipe_barrier(PIPE_V);
  RoundMode v106 = RoundMode::CAST_TRUNC;
  SaturationMode v107 = SaturationMode::ON;
  TCVT(v104, v102, v106, v107);
  // pto: %cs_dup_f_inline2472__ssa_v0
  Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v108 = Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v20, v11);
  // pto: %cs_dup_f_inline2472__ssa_v0
  uint64_t v109 = (uint64_t) v8;
  TASSIGN(v108, v109);
  pipe_barrier(PIPE_V);
  RoundMode v110 = RoundMode::CAST_ROUND;
  SaturationMode v111 = SaturationMode::OFF;
  TCVT(v108, v104, v110, v111);
  // pto: %t__tmp_v336
  Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v112 = Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v20, v11);
  // pto: %t__tmp_v336
  uint64_t v113 = (uint64_t) v8;
  TASSIGN(v112, v113);
  pipe_barrier(PIPE_V);
  TMULS(v112, v108, v18);
  // pto: %cs_lane_inline2532__ssa_v0
  Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v114 = Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v20, v11);
  // pto: %cs_lane_inline2532__ssa_v0
  uint64_t v115 = (uint64_t) v7;
  TASSIGN(v114, v115);
  pipe_barrier(PIPE_V);
  TSUB(v114, v100, v112);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  // pto: %t__tmp_v337
  Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v116 = Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v20, v11);
  // pto: %t__tmp_v337
  uint64_t v117 = (uint64_t) v7;
  TASSIGN(v116, v117);
  pipe_barrier(PIPE_V);
  TMULS(v116, v114, v18);
  // pto: %t__tmp_v338
  Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v118 = Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v20, v11);
  // pto: %t__tmp_v338
  uint64_t v119 = (uint64_t) v7;
  TASSIGN(v118, v119);
  pipe_barrier(PIPE_V);
  TSUBS(v118, v116, v14);
  // pto: %cs_sign_inline2537__ssa_v0
  Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v120 = Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v20, v11);
  // pto: %cs_sign_inline2537__ssa_v0
  uint64_t v121 = (uint64_t) v7;
  TASSIGN(v120, v121);
  pipe_barrier(PIPE_V);
  TNEG(v120, v118);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  for (int64_t v122 = v19; v122 < v4; v122 += v12) {
    // pto: %7
    ;
    int64_t v123 = (int64_t) ((uint64_t) v122 * (uint64_t) v20);
    // pto: %8
    ;
    int64_t v124 = (int64_t) ((uint64_t) v5 - (uint64_t) v123);
    // pto: %9
    ;
    int64_t v125 = v124 < v20 ? v124 : v20;
    // pto: %cs_sin_inline2468__ssa_v0
    ;
    Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v126 = Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v125, v11);
    // pto: %cs_sin_inline2468__ssa_v0
    ;
    uint64_t v127 = (uint64_t) v8;
    TASSIGN(v126, v127);
    // pto: %10
    ;
    int64_t v128 = v123 < v19 ? v19 : v123;
    // pto: %freqs_sin__ssa_v0_pview
    ;
    const int64_t v129 = 0;
    // pto: %freqs_sin__ssa_v0_pview
    ;
    __gm__ float* v130 = PTOAS__GLOBAL_TENSOR_DATA(v37);
    // pto: %freqs_sin__ssa_v0_pview
    ;
    const int64_t v131 = 1;
    // pto: %freqs_sin__ssa_v0_pview
    ;
    const int64_t v132 = 1;
    // pto: %freqs_sin__ssa_v0_pview
    ;
    const int64_t v133 = 1;
    // pto: %freqs_sin__ssa_v0_pview
    ;
    int64_t v134 = v125 * v11;
    // pto: %freqs_sin__ssa_v0_pview
    ;
    int64_t v135 = v133 * v134;
    // pto: %freqs_sin__ssa_v0_pview
    ;
    pto::Shape<1, 1, 1, -1, 64> v136 = pto::Shape<1, 1, 1, -1, 64>(v131, v132, v133, v125, v11);
    // pto: %freqs_sin__ssa_v0_pview
    ;
    pto::Stride<-1, -1, -1, -1, -1> v137 = pto::Stride<-1, -1, -1, -1, -1>(v132 * v135, v135, v134, v11, v12);
    // pto: %freqs_sin__ssa_v0_pview
    ;
    GlobalTensor<float, pto::Shape<1, 1, 1, -1, 64>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v138 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, 64>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v130 + (v129 + v128 * v11 + v19 * v12), v136, v137);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    TLOAD(v126, v138);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    v120.SetValidShape(v125, v11);
    // pto: %t__tmp_v339
    ;
    Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v139 = Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v125, v11);
    // pto: %t__tmp_v339
    ;
    uint64_t v140 = (uint64_t) v8;
    TASSIGN(v139, v140);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    pipe_barrier(PIPE_V);
    TMUL(v139, v126, v120);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
    // pto: %rope_sin_signed_inline2481__ssa_v0_pview
    ;
    const int64_t v141 = 0;
    // pto: %rope_sin_signed_inline2481__ssa_v0_pview
    ;
    __gm__ float* v142 = PTOAS__GLOBAL_TENSOR_DATA(v45);
    // pto: %rope_sin_signed_inline2481__ssa_v0_pview
    ;
    const int64_t v143 = 1;
    // pto: %rope_sin_signed_inline2481__ssa_v0_pview
    ;
    const int64_t v144 = 1;
    // pto: %rope_sin_signed_inline2481__ssa_v0_pview
    ;
    const int64_t v145 = 1;
    // pto: %rope_sin_signed_inline2481__ssa_v0_pview
    ;
    int64_t v146 = v125 * v11;
    // pto: %rope_sin_signed_inline2481__ssa_v0_pview
    ;
    int64_t v147 = v145 * v146;
    // pto: %rope_sin_signed_inline2481__ssa_v0_pview
    ;
    pto::Shape<1, 1, 1, -1, 64> v148 = pto::Shape<1, 1, 1, -1, 64>(v143, v144, v145, v125, v11);
    // pto: %rope_sin_signed_inline2481__ssa_v0_pview
    ;
    pto::Stride<-1, -1, -1, -1, -1> v149 = pto::Stride<-1, -1, -1, -1, -1>(v144 * v147, v147, v146, v11, v12);
    // pto: %rope_sin_signed_inline2481__ssa_v0_pview
    ;
    GlobalTensor<float, pto::Shape<1, 1, 1, -1, 64>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v150 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, 64>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v142 + (v141 + v128 * v11 + v19 * v12), v148, v149);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
    TSTORE(v150, v139);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  }
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  #endif // __DAV_VEC__

  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}