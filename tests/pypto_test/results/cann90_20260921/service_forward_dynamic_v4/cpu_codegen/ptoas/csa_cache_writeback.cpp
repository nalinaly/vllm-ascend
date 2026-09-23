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

AICORE void csa_cache_writeback(__gm__ bfloat16_t* v1, __gm__ int32_t* v2, __gm__ bfloat16_t* v3, int64_t v4, int64_t v5, int64_t v6, int32_t v7, int32_t v8) {
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  // pto: %c0_i64
  const int64_t v9 = 0;
  // pto: %c512_index
  const int64_t v10 = 512;
  // pto: %c1_index
  const int64_t v11 = 1;
  // pto: %c2_index
  const int64_t v12 = 2;
  // pto: %c8_index
  const int64_t v13 = 8;
  // pto: %c0_index
  const int64_t v14 = 0;
  // pto: %c32_index
  const int64_t v15 = 32;
  // pto: %15
  int64_t v16 = (int64_t) ((uint64_t) v5 * (uint64_t) v15);
  // pto: %kv_cache_flat__ssa_v0_view
  const int64_t v17 = 1;
  // pto: %kv_cache_flat__ssa_v0_view
  const int64_t v18 = 1;
  // pto: %kv_cache_flat__ssa_v0_view
  const int64_t v19 = 1;
  // pto: %kv_cache_flat__ssa_v0_view
  int64_t v20 = v16 * v10;
  // pto: %kv_cache_flat__ssa_v0_view
  int64_t v21 = v19 * v20;
  // pto: %kv_cache_flat__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v22 = pto::Shape<1, 1, 1, -1, -1>(v17, v18, v19, v16, v10);
  // pto: %kv_cache_flat__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v23 = pto::Stride<-1, -1, -1, -1, -1>(v18 * v21, v21, v20, v10, v11);
  // pto: %kv_cache_flat__ssa_v0_view
  GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v24 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v1, v22, v23);
  // pto: %kv__ssa_v0_view
  const int64_t v25 = 1;
  // pto: %kv__ssa_v0_view
  const int64_t v26 = 1;
  // pto: %kv__ssa_v0_view
  const int64_t v27 = 1;
  // pto: %kv__ssa_v0_view
  int64_t v28 = (int64_t) v6;
  // pto: %kv__ssa_v0_view
  int64_t v29 = v28 * v10;
  // pto: %kv__ssa_v0_view
  int64_t v30 = v27 * v29;
  // pto: %kv__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v31 = pto::Shape<1, 1, 1, -1, -1>(v25, v26, v27, v28, v10);
  // pto: %kv__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v32 = pto::Stride<-1, -1, -1, -1, -1>(v26 * v30, v30, v29, v10, v11);
  // pto: %kv__ssa_v0_view
  GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v33 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v3, v31, v32);
  // pto: %wb_worker__ssa_v0
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  for (int64_t v34 = (int64_t) v7; v34 < v4; v34 += v13) {
    for (int64_t v35 = v14; v35 < v13; v35 += v11) {
      // pto: %0, %1
      ;
      int64_t v36 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) v34 * (uint64_t) v13) + (uint64_t) v35);
      // pto: %flat_offset_mul
      ;
      int64_t v37 = (int64_t) ((uint64_t) v36 * (uint64_t) v12);
      // pto: %write_page__tile
      ;
      int32_t v38 = (v2)[v37];
      // pto: %3, %write_offset__tile
      ;
      int32_t v39 = (v2)[(int64_t) ((uint64_t) v37 + (uint64_t) v11)];
      // pto: %4
      ;
      int64_t v40 = (int64_t) v38;
      // pto: %6
      ;
      int64_t v41 = (int64_t) v39;
      // pto: %5, %7, %8
      ;
      if (v40 >= v14 & v41 >= v14) {
        // pto: %10, %12
        ;
        int64_t v42 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) v40 * (uint64_t) v15) + (uint64_t) v41);
        // pto: %t__tile
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v43 = Tile<TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v11, v10);
        // pto: %t__tile
        ;
        uint64_t v44 = (uint64_t) v9;
        TASSIGN(v43, v44);
        // pto: %13
        ;
        int64_t v45 = v36 < v14 ? v14 : v36;
        // pto: %kv__ssa_v0_pview
        ;
        __gm__ bfloat16_t* v46 = PTOAS__GLOBAL_TENSOR_DATA(v33);
        // pto: %kv__ssa_v0_pview
        ;
        const int64_t v47 = 0;
        // pto: %kv__ssa_v0_pview
        ;
        const int64_t v48 = 512;
        // pto: %kv__ssa_v0_pview
        ;
        pto::Shape<1, 1, 1, 1, 512> v49 = pto::Shape<1, 1, 1, 1, 512>();
        // pto: %kv__ssa_v0_pview
        ;
        pto::Stride<512, 512, 512, 512, 1> v50 = pto::Stride<512, 512, 512, 512, 1>();
        // pto: %kv__ssa_v0_pview
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<512, 512, 512, 512, 1>, pto::Layout::ND> v51 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<512, 512, 512, 512, 1>, pto::Layout::ND>(v46 + (v47 + v45 * v48), v49, v50);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        TLOAD(v43, v51);
        set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        // pto: %14
        ;
        int64_t v52 = v42 < v14 ? v14 : v42;
        // pto: %kv_cache_flat__iter_v3_pview
        ;
        __gm__ bfloat16_t* v53 = PTOAS__GLOBAL_TENSOR_DATA(v24);
        // pto: %kv_cache_flat__iter_v3_pview
        ;
        const int64_t v54 = 0;
        // pto: %kv_cache_flat__iter_v3_pview
        ;
        const int64_t v55 = 512;
        // pto: %kv_cache_flat__iter_v3_pview
        ;
        pto::Shape<1, 1, 1, 1, 512> v56 = pto::Shape<1, 1, 1, 1, 512>();
        // pto: %kv_cache_flat__iter_v3_pview
        ;
        pto::Stride<512, 512, 512, 512, 1> v57 = pto::Stride<512, 512, 512, 512, 1>();
        // pto: %kv_cache_flat__iter_v3_pview
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<512, 512, 512, 512, 1>, pto::Layout::ND> v58 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<512, 512, 512, 512, 1>, pto::Layout::ND>(v53 + (v54 + v52 * v55), v56, v57);
        wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        TSTORE(v58, v43);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
      };
    };
  }
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  #endif // __DAV_VEC__

  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}