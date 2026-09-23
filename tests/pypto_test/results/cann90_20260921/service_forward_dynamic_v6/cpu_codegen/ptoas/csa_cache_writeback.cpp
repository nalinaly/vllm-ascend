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

AICORE void csa_cache_writeback(__gm__ bfloat16_t* v1, __gm__ int32_t* v2, __gm__ bfloat16_t* v3, int64_t v4, int64_t v5, int64_t v6, int64_t v7, int32_t v8, int32_t v9) {
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  // pto: %c0_i64
  const int64_t v10 = 0;
  // pto: %c512_index
  const int64_t v11 = 512;
  // pto: %c1_index
  const int64_t v12 = 1;
  // pto: %c2_index
  const int64_t v13 = 2;
  // pto: %c8_index
  const int64_t v14 = 8;
  // pto: %c0_index
  const int64_t v15 = 0;
  // pto: %c32_index
  const int64_t v16 = 32;
  // pto: %17
  int64_t v17 = (int64_t) ((uint64_t) v6 * (uint64_t) v16);
  // pto: %kv_cache_flat__ssa_v0_view
  const int64_t v18 = 1;
  // pto: %kv_cache_flat__ssa_v0_view
  const int64_t v19 = 1;
  // pto: %kv_cache_flat__ssa_v0_view
  const int64_t v20 = 1;
  // pto: %kv_cache_flat__ssa_v0_view
  int64_t v21 = v17 * v11;
  // pto: %kv_cache_flat__ssa_v0_view
  int64_t v22 = v20 * v21;
  // pto: %kv_cache_flat__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v23 = pto::Shape<1, 1, 1, -1, -1>(v18, v19, v20, v17, v11);
  // pto: %kv_cache_flat__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v24 = pto::Stride<-1, -1, -1, -1, -1>(v19 * v22, v22, v21, v11, v12);
  // pto: %kv_cache_flat__ssa_v0_view
  GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v25 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v1, v23, v24);
  // pto: %kv__ssa_v0_view
  const int64_t v26 = 1;
  // pto: %kv__ssa_v0_view
  const int64_t v27 = 1;
  // pto: %kv__ssa_v0_view
  const int64_t v28 = 1;
  // pto: %kv__ssa_v0_view
  int64_t v29 = (int64_t) v7;
  // pto: %kv__ssa_v0_view
  int64_t v30 = v29 * v11;
  // pto: %kv__ssa_v0_view
  int64_t v31 = v28 * v30;
  // pto: %kv__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v32 = pto::Shape<1, 1, 1, -1, -1>(v26, v27, v28, v29, v11);
  // pto: %kv__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v33 = pto::Stride<-1, -1, -1, -1, -1>(v27 * v31, v31, v30, v11, v12);
  // pto: %kv__ssa_v0_view
  GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v34 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v3, v32, v33);
  // pto: %wb_worker__ssa_v0
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  for (int64_t v35 = (int64_t) v8; v35 < v4; v35 += v14) {
    // pto: %0
    ;
    int64_t v36 = (int64_t) ((uint64_t) v35 * (uint64_t) v14);
    // pto: %1
    ;
    int64_t v37 = (int64_t) ((uint64_t) v7 - (uint64_t) v36);
    // pto: %2
    ;
    for (int64_t v38 = v15; v38 < (v37 < v14 ? v37 : v14); v38 += v12) {
      // pto: %3
      ;
      int64_t v39 = (int64_t) ((uint64_t) v36 + (uint64_t) v38);
      // pto: %flat_offset_mul
      ;
      int64_t v40 = (int64_t) ((uint64_t) v39 * (uint64_t) v13);
      // pto: %write_page__tile
      ;
      int32_t v41 = (v2)[v40];
      // pto: %5, %write_offset__tile
      ;
      int32_t v42 = (v2)[(int64_t) ((uint64_t) v40 + (uint64_t) v12)];
      // pto: %6
      ;
      int64_t v43 = (int64_t) v41;
      // pto: %8
      ;
      int64_t v44 = (int64_t) v42;
      // pto: %7, %9, %10
      ;
      if (v43 >= v15 & v44 >= v15) {
        // pto: %12, %14
        ;
        int64_t v45 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) v43 * (uint64_t) v16) + (uint64_t) v44);
        // pto: %t__tile
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v46 = Tile<TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v12, v11);
        // pto: %t__tile
        ;
        uint64_t v47 = (uint64_t) v10;
        TASSIGN(v46, v47);
        // pto: %15
        ;
        int64_t v48 = v39 < v15 ? v15 : v39;
        // pto: %kv__ssa_v0_pview
        ;
        __gm__ bfloat16_t* v49 = PTOAS__GLOBAL_TENSOR_DATA(v34);
        // pto: %kv__ssa_v0_pview
        ;
        const int64_t v50 = 0;
        // pto: %kv__ssa_v0_pview
        ;
        const int64_t v51 = 512;
        // pto: %kv__ssa_v0_pview
        ;
        pto::Shape<1, 1, 1, 1, 512> v52 = pto::Shape<1, 1, 1, 1, 512>();
        // pto: %kv__ssa_v0_pview
        ;
        pto::Stride<512, 512, 512, 512, 1> v53 = pto::Stride<512, 512, 512, 512, 1>();
        // pto: %kv__ssa_v0_pview
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<512, 512, 512, 512, 1>, pto::Layout::ND> v54 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<512, 512, 512, 512, 1>, pto::Layout::ND>(v49 + (v50 + v48 * v51), v52, v53);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        TLOAD(v46, v54);
        set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        // pto: %16
        ;
        int64_t v55 = v45 < v15 ? v15 : v45;
        // pto: %kv_cache_flat__iter_v3_pview
        ;
        __gm__ bfloat16_t* v56 = PTOAS__GLOBAL_TENSOR_DATA(v25);
        // pto: %kv_cache_flat__iter_v3_pview
        ;
        const int64_t v57 = 0;
        // pto: %kv_cache_flat__iter_v3_pview
        ;
        const int64_t v58 = 512;
        // pto: %kv_cache_flat__iter_v3_pview
        ;
        pto::Shape<1, 1, 1, 1, 512> v59 = pto::Shape<1, 1, 1, 1, 512>();
        // pto: %kv_cache_flat__iter_v3_pview
        ;
        pto::Stride<512, 512, 512, 512, 1> v60 = pto::Stride<512, 512, 512, 512, 1>();
        // pto: %kv_cache_flat__iter_v3_pview
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<512, 512, 512, 512, 1>, pto::Layout::ND> v61 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<512, 512, 512, 512, 1>, pto::Layout::ND>(v56 + (v57 + v55 * v58), v59, v60);
        wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        TSTORE(v61, v46);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
      };
    };
  }
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  #endif // __DAV_VEC__

  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}