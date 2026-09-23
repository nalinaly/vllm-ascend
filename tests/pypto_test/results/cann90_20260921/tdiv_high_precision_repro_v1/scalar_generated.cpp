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

AICORE void scalar_division_incore_0(__gm__ float* v1, __gm__ float* v2, __gm__ float* v3, int32_t v4, int32_t v5) {
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  // pto: %c0_i64
  const int64_t v6 = 0;
  // pto: %c16_index
  const int64_t v7 = 16;
  // pto: %c256_index
  const int64_t v8 = 256;
  // pto: %c1_index
  const int64_t v9 = 1;
  // pto: %cst_4
  const float v10 = 0.0f;
  // pto: %c0_index
  const int64_t v11 = 0;
  // pto: %result__ssa_v0_view
  const int64_t v12 = 1;
  // pto: %result__ssa_v0_view
  const int64_t v13 = 1;
  // pto: %result__ssa_v0_view
  const int64_t v14 = 1;
  // pto: %result__ssa_v0_view
  int64_t v15 = v7 * v8;
  // pto: %result__ssa_v0_view
  int64_t v16 = v14 * v15;
  // pto: %result__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v17 = pto::Shape<1, 1, 1, -1, -1>(v12, v13, v14, v7, v8);
  // pto: %result__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v18 = pto::Stride<-1, -1, -1, -1, -1>(v13 * v16, v16, v15, v8, v9);
  // pto: %result__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v19 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v3, v17, v18);
  // pto: %row__ssa_v0
  int64_t v20 = (int64_t) v4;
  // pto: %values__tile
  Tile<TileType::Vec, float, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v21 = Tile<TileType::Vec, float, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v9, v8);
  // pto: %values__tile
  uint64_t v22 = (uint64_t) v6;
  TASSIGN(v21, v22);
  TEXPANDS(v21, v10);
  set_flag(PIPE_V, PIPE_S, EVENT_ID0);
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
  for (int64_t v23 = v11; v23 < v8; v23 += v9) {
    // pto: %flat_offset_mul, %flat_offset
    ;
    int64_t v24 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) v20 * (uint64_t) v8) + (uint64_t) v23);
    // pto: %left__tile
    ;
    float v25 = (v1)[v24];
    // pto: %right__tile
    ;
    float v26 = (v2)[v24];
    // pto: %2
    ;
    float v27 = v25 / v26;
    v21.SetValue(v23, v27);
  }
  set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
  // pto: %5
  int64_t v28 = v20 < v11 ? v11 : v20;
  // pto: %result__ssa_v0_pview
  __gm__ float* v29 = PTOAS__GLOBAL_TENSOR_DATA(v19);
  // pto: %result__ssa_v0_pview
  const int64_t v30 = 0;
  // pto: %result__ssa_v0_pview
  const int64_t v31 = 256;
  // pto: %result__ssa_v0_pview
  pto::Shape<1, 1, 1, 1, 256> v32 = pto::Shape<1, 1, 1, 1, 256>();
  // pto: %result__ssa_v0_pview
  pto::Stride<256, 256, 256, 256, 1> v33 = pto::Stride<256, 256, 256, 256, 1>();
  // pto: %result__ssa_v0_pview
  GlobalTensor<float, pto::Shape<1, 1, 1, 1, 256>, pto::Stride<256, 256, 256, 256, 1>, pto::Layout::ND> v34 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 256>, pto::Stride<256, 256, 256, 256, 1>, pto::Layout::ND>(v29 + (v30 + v28 * v31), v32, v33);
  wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  TSTORE(v34, v21);
  #endif // __DAV_VEC__

  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}