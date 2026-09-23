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

AICORE void indexer_head_coefficients(__gm__ int64_t* v1, __gm__ half* v2, __gm__ float* v3, __gm__ float* v4, int64_t v5, int32_t v6, int32_t v7) {
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  // pto: %c384_i64
  const int64_t v8 = 384;
  // pto: %c0_i64
  const int64_t v9 = 0;
  // pto: %c256_i64
  const int64_t v10 = 256;
  // pto: %c1_index
  const int64_t v11 = 1;
  // pto: %c6144_index
  const int64_t v12 = 6144;
  // pto: %c64_index
  const int64_t v13 = 64;
  // pto: %c24576_index
  const int64_t v14 = 24576;
  // pto: %c384_index
  const int64_t v15 = 384;
  // pto: %c48_index
  const int64_t v16 = 48;
  // pto: %c0_index
  const int64_t v17 = 0;
  // pto: %c16_index
  const int64_t v18 = 16;
  // pto: %cst_11
  const float v19 = 1.0f;
  // pto: %coefficients_inline3259__ssa_v0_view
  const int64_t v20 = 1;
  // pto: %coefficients_inline3259__ssa_v0_view
  const int64_t v21 = 1;
  // pto: %coefficients_inline3259__ssa_v0_view
  const int64_t v22 = 1;
  // pto: %coefficients_inline3259__ssa_v0_view
  int64_t v23 = v12 * v13;
  // pto: %coefficients_inline3259__ssa_v0_view
  int64_t v24 = v22 * v23;
  // pto: %coefficients_inline3259__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v25 = pto::Shape<1, 1, 1, -1, -1>(v20, v21, v22, v12, v13);
  // pto: %coefficients_inline3259__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v26 = pto::Stride<-1, -1, -1, -1, -1>(v21 * v24, v24, v23, v13, v11);
  // pto: %coefficients_inline3259__ssa_v0_view
  GlobalTensor<half, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v27 = GlobalTensor<half, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v2, v25, v26);
  // pto: %qr_hadamard_scale_dq_inline220__rv_v2_view
  const int64_t v28 = 1;
  // pto: %qr_hadamard_scale_dq_inline220__rv_v2_view
  const int64_t v29 = 1;
  // pto: %qr_hadamard_scale_dq_inline220__rv_v2_view
  const int64_t v30 = 1;
  // pto: %qr_hadamard_scale_dq_inline220__rv_v2_view
  int64_t v31 = v14 * v11;
  // pto: %qr_hadamard_scale_dq_inline220__rv_v2_view
  int64_t v32 = v30 * v31;
  // pto: %qr_hadamard_scale_dq_inline220__rv_v2_view
  pto::Shape<1, 1, 1, -1, -1> v33 = pto::Shape<1, 1, 1, -1, -1>(v28, v29, v30, v14, v11);
  // pto: %qr_hadamard_scale_dq_inline220__rv_v2_view
  pto::Stride<-1, -1, -1, -1, -1> v34 = pto::Stride<-1, -1, -1, -1, -1>(v29 * v32, v32, v31, v11, v14);
  // pto: %qr_hadamard_scale_dq_inline220__rv_v2_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN> v35 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN>(v3, v33, v34);
  // pto: %weights_inline2409__ssa_v0_view
  const int64_t v36 = 1;
  // pto: %weights_inline2409__ssa_v0_view
  const int64_t v37 = 1;
  // pto: %weights_inline2409__ssa_v0_view
  const int64_t v38 = 1;
  // pto: %weights_inline2409__ssa_v0_view
  int64_t v39 = v15 * v13;
  // pto: %weights_inline2409__ssa_v0_view
  int64_t v40 = v38 * v39;
  // pto: %weights_inline2409__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v41 = pto::Shape<1, 1, 1, -1, -1>(v36, v37, v38, v15, v13);
  // pto: %weights_inline2409__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v42 = pto::Stride<-1, -1, -1, -1, -1>(v37 * v40, v40, v39, v13, v11);
  // pto: %weights_inline2409__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v43 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v4, v41, v42);
  // pto: %coefficient_worker_inline3256__ssa_v0
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  for (int64_t v44 = (int64_t) v6; v44 < v5; v44 += v16) {
    // pto: %4
    ;
    int64_t v45 = (int64_t) ((uint64_t) v44 * (uint64_t) v13);
    // pto: %t__tile
    ;
    Tile<TileType::Vec, float, 64, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v46 = Tile<TileType::Vec, float, 64, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v13, v11);
    // pto: %t__tile
    ;
    uint64_t v47 = (uint64_t) v8;
    TASSIGN(v46, v47);
    // pto: %5
    ;
    int64_t v48 = v45 < v17 ? v17 : v45;
    // pto: %qr_hadamard_scale_dq_inline220__rv_v2_pview
    ;
    __gm__ float* v49 = PTOAS__GLOBAL_TENSOR_DATA(v35);
    // pto: %qr_hadamard_scale_dq_inline220__rv_v2_pview
    ;
    const int64_t v50 = 0;
    // pto: %qr_hadamard_scale_dq_inline220__rv_v2_pview
    ;
    pto::Shape<1, 1, 1, 64, 1> v51 = pto::Shape<1, 1, 1, 64, 1>();
    // pto: %qr_hadamard_scale_dq_inline220__rv_v2_pview
    ;
    pto::Stride<64, 64, 64, 1, 24576> v52 = pto::Stride<64, 64, 64, 1, 24576>();
    // pto: %qr_hadamard_scale_dq_inline220__rv_v2_pview
    ;
    GlobalTensor<float, pto::Shape<1, 1, 1, 64, 1>, pto::Stride<64, 64, 64, 1, 24576>, pto::Layout::DN> v53 = GlobalTensor<float, pto::Shape<1, 1, 1, 64, 1>, pto::Stride<64, 64, 64, 1, 24576>, pto::Layout::DN>(v49 + (v50 + v48), v51, v52);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    TLOAD(v46, v53);
    // pto: %query_scale_inline3252__tile
    ;
    Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v54 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v11, v13);
    // pto: %query_scale_inline3252__tile
    ;
    uint64_t v55 = (uint64_t) v8;
    TASSIGN(v54, v55);
    // pto: %query_weight_inline3251__tile
    ;
    Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v56 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v11, v13);
    // pto: %query_weight_inline3251__tile
    ;
    uint64_t v57 = (uint64_t) v9;
    TASSIGN(v56, v57);
    // pto: %6
    ;
    int64_t v58 = v44 < v17 ? v17 : v44;
    // pto: %weights_inline2409__ssa_v0_pview
    ;
    __gm__ float* v59 = PTOAS__GLOBAL_TENSOR_DATA(v43);
    // pto: %weights_inline2409__ssa_v0_pview
    ;
    const int64_t v60 = 0;
    // pto: %weights_inline2409__ssa_v0_pview
    ;
    const int64_t v61 = 64;
    // pto: %weights_inline2409__ssa_v0_pview
    ;
    pto::Shape<1, 1, 1, 1, 64> v62 = pto::Shape<1, 1, 1, 1, 64>();
    // pto: %weights_inline2409__ssa_v0_pview
    ;
    pto::Stride<64, 64, 64, 64, 1> v63 = pto::Stride<64, 64, 64, 64, 1>();
    // pto: %weights_inline2409__ssa_v0_pview
    ;
    GlobalTensor<float, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<64, 64, 64, 64, 1>, pto::Layout::ND> v64 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<64, 64, 64, 64, 1>, pto::Layout::ND>(v59 + (v60 + v58 * v61), v62, v63);
    TLOAD(v56, v64);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    // pto: %0
    ;
    Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v65 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v11, v13);
    // pto: %0
    ;
    uint64_t v66 = (uint64_t) v8;
    TASSIGN(v65, v66);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TMUL(v65, v54, v56);
    // pto: %head_coefficient_inline3250__tile
    ;
    Tile<TileType::Vec, half, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v67 = Tile<TileType::Vec, half, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v11, v13);
    // pto: %head_coefficient_inline3250__tile
    ;
    uint64_t v68 = (uint64_t) v10;
    TASSIGN(v67, v68);
    pipe_barrier(PIPE_V);
    RoundMode v69 = RoundMode::CAST_RINT;
    SaturationMode v70 = SaturationMode::OFF;
    TCVT(v67, v65, v69, v70);
    // pto: %1
    ;
    Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v71 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v18, v13);
    // pto: %1
    ;
    uint64_t v72 = (uint64_t) v8;
    TASSIGN(v71, v72);
    pipe_barrier(PIPE_V);
    TEXPANDS(v71, v19);
    // pto: %2
    ;
    Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v73 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v11, v13);
    // pto: %2
    ;
    uint64_t v74 = (uint64_t) v9;
    TASSIGN(v73, v74);
    RoundMode v75 = RoundMode::CAST_ROUND;
    SaturationMode v76 = SaturationMode::OFF;
    TCVT(v73, v67, v75, v76);
    // pto: %coefficient_rows_inline3249__tile
    ;
    Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v77 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v18, v13);
    // pto: %coefficient_rows_inline3249__tile
    ;
    uint64_t v78 = (uint64_t) v8;
    TASSIGN(v77, v78);
    pipe_barrier(PIPE_V);
    TCOLEXPANDMUL(v77, v71, v73);
    // pto: %7
    ;
    int64_t v79 = (int64_t) ((uint64_t) v44 * (uint64_t) v18);
    // pto: %3
    ;
    Tile<TileType::Vec, half, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v80 = Tile<TileType::Vec, half, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v18, v13);
    // pto: %3
    ;
    uint64_t v81 = (uint64_t) v8;
    TASSIGN(v80, v81);
    pipe_barrier(PIPE_V);
    RoundMode v82 = RoundMode::CAST_ROUND;
    SaturationMode v83 = SaturationMode::OFF;
    TCVT(v80, v77, v82, v83);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    // pto: %8
    ;
    int64_t v84 = v79 < v17 ? v17 : v79;
    // pto: %coefficients_inline3259__iter_v1_pview
    ;
    __gm__ half* v85 = PTOAS__GLOBAL_TENSOR_DATA(v27);
    // pto: %coefficients_inline3259__iter_v1_pview
    ;
    const int64_t v86 = 0;
    // pto: %coefficients_inline3259__iter_v1_pview
    ;
    const int64_t v87 = 64;
    // pto: %coefficients_inline3259__iter_v1_pview
    ;
    pto::Shape<1, 1, 1, 16, 64> v88 = pto::Shape<1, 1, 1, 16, 64>();
    // pto: %coefficients_inline3259__iter_v1_pview
    ;
    pto::Stride<1024, 1024, 1024, 64, 1> v89 = pto::Stride<1024, 1024, 1024, 64, 1>();
    // pto: %coefficients_inline3259__iter_v1_pview
    ;
    GlobalTensor<half, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<1024, 1024, 1024, 64, 1>, pto::Layout::ND> v90 = GlobalTensor<half, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<1024, 1024, 1024, 64, 1>, pto::Layout::ND>(v85 + (v86 + v84 * v87), v88, v89);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(v90, v80);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  }
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  #endif // __DAV_VEC__

  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}