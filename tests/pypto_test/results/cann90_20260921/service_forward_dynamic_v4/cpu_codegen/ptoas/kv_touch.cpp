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

AICORE void kv_touch(__gm__ bfloat16_t* v1, int64_t v2) {
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  // pto: %c0_i64
  const int64_t v3 = 0;
  // pto: %c512_index
  const int64_t v4 = 512;
  // pto: %c1_index
  const int64_t v5 = 1;
  // pto: %c32_index
  const int64_t v6 = 32;
  // pto: %1
  int64_t v7 = (int64_t) ((uint64_t) v2 * (uint64_t) v6);
  // pto: %ori_kv_flat_inline2541__ssa_v0_view
  const int64_t v8 = 1;
  // pto: %ori_kv_flat_inline2541__ssa_v0_view
  const int64_t v9 = 1;
  // pto: %ori_kv_flat_inline2541__ssa_v0_view
  const int64_t v10 = 1;
  // pto: %ori_kv_flat_inline2541__ssa_v0_view
  int64_t v11 = v7 * v4;
  // pto: %ori_kv_flat_inline2541__ssa_v0_view
  int64_t v12 = v10 * v11;
  // pto: %ori_kv_flat_inline2541__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v13 = pto::Shape<1, 1, 1, -1, -1>(v8, v9, v10, v7, v4);
  // pto: %ori_kv_flat_inline2541__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v14 = pto::Stride<-1, -1, -1, -1, -1>(v9 * v12, v12, v11, v4, v5);
  // pto: %ori_kv_flat_inline2541__ssa_v0_view
  GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v15 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v1, v13, v14);
  // pto: %t__tile
  Tile<TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v16 = Tile<TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v5, v4);
  // pto: %t__tile
  uint64_t v17 = (uint64_t) v3;
  TASSIGN(v16, v17);
  // pto: %ori_kv_flat_inline2541__ssa_v0_pview
  __gm__ bfloat16_t* v18 = PTOAS__GLOBAL_TENSOR_DATA(v15);
  // pto: %ori_kv_flat_inline2541__ssa_v0_pview
  pto::Shape<1, 1, 1, 1, 512> v19 = pto::Shape<1, 1, 1, 1, 512>();
  // pto: %ori_kv_flat_inline2541__ssa_v0_pview
  pto::Stride<512, 512, 512, 512, 1> v20 = pto::Stride<512, 512, 512, 512, 1>();
  // pto: %ori_kv_flat_inline2541__ssa_v0_pview
  GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<512, 512, 512, 512, 1>, pto::Layout::ND> v21 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<512, 512, 512, 512, 1>, pto::Layout::ND>(v18, v19, v20);
  TLOAD(v16, v21);
  set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
  wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
  TSTORE(v21, v16);
  #endif // __DAV_VEC__

  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}