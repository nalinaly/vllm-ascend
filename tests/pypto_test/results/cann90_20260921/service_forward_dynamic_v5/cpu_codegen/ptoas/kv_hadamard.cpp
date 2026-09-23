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

AICORE void kv_hadamard(__gm__ float* v1, __gm__ bfloat16_t* v2, __gm__ bfloat16_t* v3, int64_t v4, int64_t v5) {
  using T = float;

  #if defined(__DAV_CUBE__)
  // pto: %c0_i64
  const int64_t v6 = 0;
  // pto: %c16384_i64
  const int64_t v7 = 16384;
  // pto: %c384_index
  const int64_t v8 = 384;
  // pto: %c128_index
  const int64_t v9 = 128;
  // pto: %c1_index
  const int64_t v10 = 1;
  // pto: %c0_index
  const int64_t v11 = 0;
  // pto: %c64_index
  const int64_t v12 = 64;
  // pto: %c16_index
  const int64_t v13 = 16;
  // pto: %kv_final_inline2230__ssa_v0_view
  const int64_t v14 = 1;
  // pto: %kv_final_inline2230__ssa_v0_view
  const int64_t v15 = 1;
  // pto: %kv_final_inline2230__ssa_v0_view
  const int64_t v16 = 1;
  // pto: %kv_final_inline2230__ssa_v0_view
  int64_t v17 = v8 * v9;
  // pto: %kv_final_inline2230__ssa_v0_view
  int64_t v18 = v16 * v17;
  // pto: %kv_final_inline2230__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v19 = pto::Shape<1, 1, 1, -1, -1>(v14, v15, v16, v8, v9);
  // pto: %kv_final_inline2230__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v20 = pto::Stride<-1, -1, -1, -1, -1>(v15 * v18, v18, v17, v9, v10);
  // pto: %kv_final_inline2230__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v21 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v1, v19, v20);
  // pto: %hadamard_idx__ssa_v0_view
  const int64_t v22 = 1;
  // pto: %hadamard_idx__ssa_v0_view
  const int64_t v23 = 1;
  // pto: %hadamard_idx__ssa_v0_view
  const int64_t v24 = 1;
  // pto: %hadamard_idx__ssa_v0_view
  int64_t v25 = v9 * v9;
  // pto: %hadamard_idx__ssa_v0_view
  int64_t v26 = v24 * v25;
  // pto: %hadamard_idx__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v27 = pto::Shape<1, 1, 1, -1, -1>(v22, v23, v24, v9, v9);
  // pto: %hadamard_idx__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v28 = pto::Stride<-1, -1, -1, -1, -1>(v23 * v26, v26, v25, v9, v10);
  // pto: %hadamard_idx__ssa_v0_view
  GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v29 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v2, v27, v28);
  // pto: %normed_kv_inline214__rv_v3_view
  const int64_t v30 = 1;
  // pto: %normed_kv_inline214__rv_v3_view
  const int64_t v31 = 1;
  // pto: %normed_kv_inline214__rv_v3_view
  const int64_t v32 = 1;
  // pto: %normed_kv_inline214__rv_v3_view
  int64_t v33 = v8 * v9;
  // pto: %normed_kv_inline214__rv_v3_view
  int64_t v34 = v32 * v33;
  // pto: %normed_kv_inline214__rv_v3_view
  pto::Shape<1, 1, 1, -1, -1> v35 = pto::Shape<1, 1, 1, -1, -1>(v30, v31, v32, v8, v9);
  // pto: %normed_kv_inline214__rv_v3_view
  pto::Stride<-1, -1, -1, -1, -1> v36 = pto::Stride<-1, -1, -1, -1, -1>(v31 * v34, v34, v33, v9, v10);
  // pto: %normed_kv_inline214__rv_v3_view
  GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v37 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v3, v35, v36);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  for (int64_t v38 = v11; v38 < v9; v38 += v12) {
    // pto: %hadamard_tile_inline2233__tile
    ;
    Tile<TileType::Mat, bfloat16_t, 128, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v39 = Tile<TileType::Mat, bfloat16_t, 128, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v9, v12);
    // pto: %hadamard_tile_inline2233__tile
    ;
    uint64_t v40 = (uint64_t) v6;
    TASSIGN(v39, v40);
    // pto: %0
    ;
    int64_t v41 = v38 < v11 ? v11 : v38;
    // pto: %hadamard_idx__ssa_v0_pview
    ;
    __gm__ bfloat16_t* v42 = PTOAS__GLOBAL_TENSOR_DATA(v29);
    // pto: %hadamard_idx__ssa_v0_pview
    ;
    const int64_t v43 = 0;
    // pto: %hadamard_idx__ssa_v0_pview
    ;
    pto::Shape<1, 1, 1, 128, 64> v44 = pto::Shape<1, 1, 1, 128, 64>();
    // pto: %hadamard_idx__ssa_v0_pview
    ;
    pto::Stride<16384, 16384, 16384, 128, 1> v45 = pto::Stride<16384, 16384, 16384, 128, 1>();
    // pto: %hadamard_idx__ssa_v0_pview
    ;
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 128, 64>, pto::Stride<16384, 16384, 16384, 128, 1>, pto::Layout::ND> v46 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 128, 64>, pto::Stride<16384, 16384, 16384, 128, 1>, pto::Layout::ND>(v42 + (v43 + v41), v44, v45);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    TLOAD(v39, v46);
    for (int64_t v47 = v11; v47 < v4; v47 += v10) {
      // pto: %1
      ;
      int64_t v48 = (int64_t) ((uint64_t) v47 * (uint64_t) v13);
      // pto: %2
      ;
      int64_t v49 = (int64_t) ((uint64_t) v5 - (uint64_t) v48);
      // pto: %3
      ;
      int64_t v50 = v49 < v13 ? v49 : v13;
      // pto: %kv_proj_tile_inline2223__tile
      ;
      Tile<TileType::Mat, bfloat16_t, 16, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v51 = Tile<TileType::Mat, bfloat16_t, 16, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v50, v9);
      // pto: %kv_proj_tile_inline2223__tile
      ;
      uint64_t v52 = (uint64_t) v7;
      TASSIGN(v51, v52);
      // pto: %4
      ;
      int64_t v53 = v48 < v11 ? v11 : v48;
      // pto: %normed_kv_inline214__rv_v3_pview
      ;
      const int64_t v54 = 0;
      // pto: %normed_kv_inline214__rv_v3_pview
      ;
      __gm__ bfloat16_t* v55 = PTOAS__GLOBAL_TENSOR_DATA(v37);
      // pto: %normed_kv_inline214__rv_v3_pview
      ;
      const int64_t v56 = 1;
      // pto: %normed_kv_inline214__rv_v3_pview
      ;
      const int64_t v57 = 1;
      // pto: %normed_kv_inline214__rv_v3_pview
      ;
      const int64_t v58 = 1;
      // pto: %normed_kv_inline214__rv_v3_pview
      ;
      int64_t v59 = v50 * v9;
      // pto: %normed_kv_inline214__rv_v3_pview
      ;
      int64_t v60 = v58 * v59;
      // pto: %normed_kv_inline214__rv_v3_pview
      ;
      pto::Shape<1, 1, 1, -1, 128> v61 = pto::Shape<1, 1, 1, -1, 128>(v56, v57, v58, v50, v9);
      // pto: %normed_kv_inline214__rv_v3_pview
      ;
      pto::Stride<-1, -1, -1, -1, -1> v62 = pto::Stride<-1, -1, -1, -1, -1>(v57 * v60, v60, v59, v9, v10);
      // pto: %normed_kv_inline214__rv_v3_pview
      ;
      GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 128>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v63 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 128>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v55 + (v54 + v53 * v9 + v11 * v10), v61, v62);
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
      TLOAD(v51, v63);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      // pto: %kv_proj_tile_inline2223__tile_Left
      ;
      Tile<TileType::Left, bfloat16_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v64 = Tile<TileType::Left, bfloat16_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v50, v9);
      // pto: %kv_proj_tile_inline2223__tile_Left
      ;
      uint64_t v65 = (uint64_t) v6;
      TASSIGN(v64, v65);
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
      TMOV(v64, v51);
      set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
      // pto: %hadamard_tile_inline2233__tile_Right
      ;
      Tile<TileType::Right, bfloat16_t, 128, 64, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v66 = Tile<TileType::Right, bfloat16_t, 128, 64, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v9, v12);
      // pto: %hadamard_tile_inline2233__tile_Right
      ;
      uint64_t v67 = (uint64_t) v6;
      TASSIGN(v66, v67);
      TMOV(v66, v39);
      set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      // pto: %kv_hadamard_acc_inline2244__tile
      ;
      Tile<TileType::Acc, float, 16, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Normal> v68 = Tile<TileType::Acc, float, 16, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Normal>(v50, v12);
      // pto: %kv_hadamard_acc_inline2244__tile
      ;
      uint64_t v69 = (uint64_t) v6;
      TASSIGN(v68, v69);
      wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
      TMATMUL(v68, v64, v66);
      set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
      set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
      // pto: %kv_final_inline2230__iter_v3_pview
      ;
      const int64_t v70 = 0;
      // pto: %kv_final_inline2230__iter_v3_pview
      ;
      __gm__ float* v71 = PTOAS__GLOBAL_TENSOR_DATA(v21);
      // pto: %kv_final_inline2230__iter_v3_pview
      ;
      const int64_t v72 = 1;
      // pto: %kv_final_inline2230__iter_v3_pview
      ;
      const int64_t v73 = 1;
      // pto: %kv_final_inline2230__iter_v3_pview
      ;
      const int64_t v74 = 1;
      // pto: %kv_final_inline2230__iter_v3_pview
      ;
      int64_t v75 = v50 * v9;
      // pto: %kv_final_inline2230__iter_v3_pview
      ;
      int64_t v76 = v74 * v75;
      // pto: %kv_final_inline2230__iter_v3_pview
      ;
      pto::Shape<1, 1, 1, -1, 64> v77 = pto::Shape<1, 1, 1, -1, 64>(v72, v73, v74, v50, v12);
      // pto: %kv_final_inline2230__iter_v3_pview
      ;
      pto::Stride<-1, -1, -1, -1, -1> v78 = pto::Stride<-1, -1, -1, -1, -1>(v73 * v76, v76, v75, v9, v10);
      // pto: %kv_final_inline2230__iter_v3_pview
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, -1, 64>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v79 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, 64>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v71 + (v70 + v53 * v9 + v41 * v10), v77, v78);
      wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
      TSTORE(v79, v68);
      set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    };
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  }
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  #endif // __DAV_CUBE__

  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}