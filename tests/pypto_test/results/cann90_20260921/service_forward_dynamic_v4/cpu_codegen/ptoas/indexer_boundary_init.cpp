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

AICORE void indexer_boundary_init(__gm__ bfloat16_t* v1, int32_t v2, int32_t v3) {
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  // pto: %c0_i64
  const int64_t v4 = 0;
  // pto: %c384_index
  const int64_t v5 = 384;
  // pto: %c128_index
  const int64_t v6 = 128;
  // pto: %c1_index
  const int64_t v7 = 1;
  // pto: %c2_index
  const int64_t v8 = 2;
  // pto: %cst_5
  const bfloat16_t v9 = 0.0f;
  // pto: %c0_index
  const int64_t v10 = 0;
  // pto: %normed_kv_inline212__ssa_v0_view
  const int64_t v11 = 1;
  // pto: %normed_kv_inline212__ssa_v0_view
  const int64_t v12 = 1;
  // pto: %normed_kv_inline212__ssa_v0_view
  const int64_t v13 = 1;
  // pto: %normed_kv_inline212__ssa_v0_view
  int64_t v14 = v5 * v6;
  // pto: %normed_kv_inline212__ssa_v0_view
  int64_t v15 = v13 * v14;
  // pto: %normed_kv_inline212__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v16 = pto::Shape<1, 1, 1, -1, -1>(v11, v12, v13, v5, v6);
  // pto: %normed_kv_inline212__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v17 = pto::Stride<-1, -1, -1, -1, -1>(v12 * v15, v15, v14, v6, v7);
  // pto: %normed_kv_inline212__ssa_v0_view
  GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v18 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v1, v16, v17);
  // pto: %init_request_inline434_inline2104__ssa_v0, %0
  int64_t v19 = (int64_t) ((uint64_t) ((int64_t) v2) * (uint64_t) v8);
  // pto: %t__tile
  Tile<TileType::Vec, bfloat16_t, 2, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v20 = Tile<TileType::Vec, bfloat16_t, 2, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v8, v6);
  // pto: %t__tile
  uint64_t v21 = (uint64_t) v4;
  TASSIGN(v20, v21);
  TEXPANDS(v20, v9);
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  // pto: %1
  int64_t v22 = v19 < v10 ? v10 : v19;
  // pto: %normed_kv_inline212__ssa_v0_pview
  __gm__ bfloat16_t* v23 = PTOAS__GLOBAL_TENSOR_DATA(v18);
  // pto: %normed_kv_inline212__ssa_v0_pview
  const int64_t v24 = 0;
  // pto: %normed_kv_inline212__ssa_v0_pview
  const int64_t v25 = 128;
  // pto: %normed_kv_inline212__ssa_v0_pview
  pto::Shape<1, 1, 1, 2, 128> v26 = pto::Shape<1, 1, 1, 2, 128>();
  // pto: %normed_kv_inline212__ssa_v0_pview
  pto::Stride<256, 256, 256, 128, 1> v27 = pto::Stride<256, 256, 256, 128, 1>();
  // pto: %normed_kv_inline212__ssa_v0_pview
  GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 2, 128>, pto::Stride<256, 256, 256, 128, 1>, pto::Layout::ND> v28 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 2, 128>, pto::Stride<256, 256, 256, 128, 1>, pto::Layout::ND>(v23 + (v24 + v22 * v25), v26, v27);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  TSTORE(v28, v20);
  #endif // __DAV_VEC__

  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}