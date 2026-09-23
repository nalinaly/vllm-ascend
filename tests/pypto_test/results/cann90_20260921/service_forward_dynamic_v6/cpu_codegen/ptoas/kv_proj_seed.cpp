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

AICORE void kv_proj_seed(__gm__ float* v1, int64_t v2, int64_t v3) {
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  // pto: %c0_i64
  const int64_t v4 = 0;
  // pto: %c512_index
  const int64_t v5 = 512;
  // pto: %c1_index
  const int64_t v6 = 1;
  // pto: %c0_index
  const int64_t v7 = 0;
  // pto: %c16_index
  const int64_t v8 = 16;
  // pto: %c128_index
  const int64_t v9 = 128;
  // pto: %cst_6
  const float v10 = 0.0f;
  // pto: %kv_fp32_inline1920__ssa_v0_view
  const int64_t v11 = 1;
  // pto: %kv_fp32_inline1920__ssa_v0_view
  const int64_t v12 = 1;
  // pto: %kv_fp32_inline1920__ssa_v0_view
  const int64_t v13 = 1;
  // pto: %kv_fp32_inline1920__ssa_v0_view
  int64_t v14 = (int64_t) v3;
  // pto: %kv_fp32_inline1920__ssa_v0_view
  int64_t v15 = v14 * v5;
  // pto: %kv_fp32_inline1920__ssa_v0_view
  int64_t v16 = v13 * v15;
  // pto: %kv_fp32_inline1920__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v17 = pto::Shape<1, 1, 1, -1, -1>(v11, v12, v13, v14, v5);
  // pto: %kv_fp32_inline1920__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v18 = pto::Stride<-1, -1, -1, -1, -1>(v12 * v16, v16, v15, v5, v6);
  // pto: %kv_fp32_inline1920__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v19 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v1, v17, v18);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  for (int64_t v20 = v7; v20 < v3; v20 += v8) {
    for (int64_t v21 = v7; v21 < v5; v21 += v9) {
      // pto: %kv_seed_inline1933__tile
      ;
      Tile<TileType::Vec, float, 16, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v22 = Tile<TileType::Vec, float, 16, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v8, v9);
      // pto: %kv_seed_inline1933__tile
      ;
      uint64_t v23 = (uint64_t) v4;
      TASSIGN(v22, v23);
      wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
      TEXPANDS(v22, v10);
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      // pto: %0
      ;
      int64_t v24 = v20 < v7 ? v7 : v20;
      // pto: %1
      ;
      int64_t v25 = v21 < v7 ? v7 : v21;
      // pto: %kv_fp32_inline1920__iter_v3_pview
      ;
      __gm__ float* v26 = PTOAS__GLOBAL_TENSOR_DATA(v19);
      // pto: %kv_fp32_inline1920__iter_v3_pview
      ;
      const int64_t v27 = 0;
      // pto: %kv_fp32_inline1920__iter_v3_pview
      ;
      const int64_t v28 = 512;
      // pto: %kv_fp32_inline1920__iter_v3_pview
      ;
      pto::Shape<1, 1, 1, 16, 128> v29 = pto::Shape<1, 1, 1, 16, 128>();
      // pto: %kv_fp32_inline1920__iter_v3_pview
      ;
      pto::Stride<8192, 8192, 8192, 512, 1> v30 = pto::Stride<8192, 8192, 8192, 512, 1>();
      // pto: %kv_fp32_inline1920__iter_v3_pview
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 16, 128>, pto::Stride<8192, 8192, 8192, 512, 1>, pto::Layout::ND> v31 = GlobalTensor<float, pto::Shape<1, 1, 1, 16, 128>, pto::Stride<8192, 8192, 8192, 512, 1>, pto::Layout::ND>(v26 + (v27 + v24 * v28 + v25), v29, v30);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      TSTORE(v31, v22);
      set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    };
  }
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  #endif // __DAV_VEC__

  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}