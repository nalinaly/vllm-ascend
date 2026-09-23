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

AICORE void vector_division_incore_0(__gm__ float* v1, __gm__ float* v2, __gm__ float* v3, __gm__ float* v4, int32_t v5, int32_t v6) {
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  // pto: %c0_i64
  const int64_t v7 = 0;
  // pto: %c1024_i64
  const int64_t v8 = 1024;
  // pto: %c2048_i64
  const int64_t v9 = 2048;
  // pto: %c16_index
  const int64_t v10 = 16;
  // pto: %c256_index
  const int64_t v11 = 256;
  // pto: %c1_index
  const int64_t v12 = 1;
  // pto: %c0_index
  const int64_t v13 = 0;
  // pto: %numerator__ssa_v0_view
  const int64_t v14 = 1;
  // pto: %numerator__ssa_v0_view
  const int64_t v15 = 1;
  // pto: %numerator__ssa_v0_view
  const int64_t v16 = 1;
  // pto: %numerator__ssa_v0_view
  int64_t v17 = v10 * v11;
  // pto: %numerator__ssa_v0_view
  int64_t v18 = v16 * v17;
  // pto: %numerator__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v19 = pto::Shape<1, 1, 1, -1, -1>(v14, v15, v16, v10, v11);
  // pto: %numerator__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v20 = pto::Stride<-1, -1, -1, -1, -1>(v15 * v18, v18, v17, v11, v12);
  // pto: %numerator__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v21 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v1, v19, v20);
  // pto: %denominator__ssa_v0_view
  const int64_t v22 = 1;
  // pto: %denominator__ssa_v0_view
  const int64_t v23 = 1;
  // pto: %denominator__ssa_v0_view
  const int64_t v24 = 1;
  // pto: %denominator__ssa_v0_view
  int64_t v25 = v10 * v11;
  // pto: %denominator__ssa_v0_view
  int64_t v26 = v24 * v25;
  // pto: %denominator__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v27 = pto::Shape<1, 1, 1, -1, -1>(v22, v23, v24, v10, v11);
  // pto: %denominator__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v28 = pto::Stride<-1, -1, -1, -1, -1>(v23 * v26, v26, v25, v11, v12);
  // pto: %denominator__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v29 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v2, v27, v28);
  // pto: %normal__ssa_v0_view
  const int64_t v30 = 1;
  // pto: %normal__ssa_v0_view
  const int64_t v31 = 1;
  // pto: %normal__ssa_v0_view
  const int64_t v32 = 1;
  // pto: %normal__ssa_v0_view
  int64_t v33 = v10 * v11;
  // pto: %normal__ssa_v0_view
  int64_t v34 = v32 * v33;
  // pto: %normal__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v35 = pto::Shape<1, 1, 1, -1, -1>(v30, v31, v32, v10, v11);
  // pto: %normal__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v36 = pto::Stride<-1, -1, -1, -1, -1>(v31 * v34, v34, v33, v11, v12);
  // pto: %normal__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v37 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v3, v35, v36);
  // pto: %precise__ssa_v0_view
  const int64_t v38 = 1;
  // pto: %precise__ssa_v0_view
  const int64_t v39 = 1;
  // pto: %precise__ssa_v0_view
  const int64_t v40 = 1;
  // pto: %precise__ssa_v0_view
  int64_t v41 = v10 * v11;
  // pto: %precise__ssa_v0_view
  int64_t v42 = v40 * v41;
  // pto: %precise__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v43 = pto::Shape<1, 1, 1, -1, -1>(v38, v39, v40, v10, v11);
  // pto: %precise__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v44 = pto::Stride<-1, -1, -1, -1, -1>(v39 * v42, v42, v41, v11, v12);
  // pto: %precise__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v45 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v4, v43, v44);
  // pto: %row__ssa_v0
  int64_t v46 = (int64_t) v5;
  // pto: %left__tile
  Tile<TileType::Vec, float, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v47 = Tile<TileType::Vec, float, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v12, v11);
  // pto: %left__tile
  uint64_t v48 = (uint64_t) v7;
  TASSIGN(v47, v48);
  // pto: %1
  int64_t v49 = v46 < v13 ? v13 : v46;
  // pto: %numerator__ssa_v0_pview
  __gm__ float* v50 = PTOAS__GLOBAL_TENSOR_DATA(v21);
  // pto: %numerator__ssa_v0_pview
  const int64_t v51 = 0;
  // pto: %numerator__ssa_v0_pview
  const int64_t v52 = 256;
  // pto: %numerator__ssa_v0_pview
  pto::Shape<1, 1, 1, 1, 256> v53 = pto::Shape<1, 1, 1, 1, 256>();
  // pto: %numerator__ssa_v0_pview
  pto::Stride<256, 256, 256, 256, 1> v54 = pto::Stride<256, 256, 256, 256, 1>();
  // pto: %numerator__ssa_v0_pview
  GlobalTensor<float, pto::Shape<1, 1, 1, 1, 256>, pto::Stride<256, 256, 256, 256, 1>, pto::Layout::ND> v55 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 256>, pto::Stride<256, 256, 256, 256, 1>, pto::Layout::ND>(v50 + (v51 + v49 * v52), v53, v54);
  TLOAD(v47, v55);
  // pto: %right__tile
  Tile<TileType::Vec, float, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v56 = Tile<TileType::Vec, float, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v12, v11);
  // pto: %right__tile
  uint64_t v57 = (uint64_t) v8;
  TASSIGN(v56, v57);
  // pto: %denominator__ssa_v0_pview
  __gm__ float* v58 = PTOAS__GLOBAL_TENSOR_DATA(v29);
  // pto: %denominator__ssa_v0_pview
  const int64_t v59 = 0;
  // pto: %denominator__ssa_v0_pview
  const int64_t v60 = 256;
  // pto: %denominator__ssa_v0_pview
  pto::Shape<1, 1, 1, 1, 256> v61 = pto::Shape<1, 1, 1, 1, 256>();
  // pto: %denominator__ssa_v0_pview
  pto::Stride<256, 256, 256, 256, 1> v62 = pto::Stride<256, 256, 256, 256, 1>();
  // pto: %denominator__ssa_v0_pview
  GlobalTensor<float, pto::Shape<1, 1, 1, 1, 256>, pto::Stride<256, 256, 256, 256, 1>, pto::Layout::ND> v63 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 256>, pto::Stride<256, 256, 256, 256, 1>, pto::Layout::ND>(v58 + (v59 + v49 * v60), v61, v62);
  TLOAD(v56, v63);
  set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  // pto: %t__tile
  Tile<TileType::Vec, float, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v64 = Tile<TileType::Vec, float, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v12, v11);
  // pto: %t__tile
  uint64_t v65 = (uint64_t) v9;
  TASSIGN(v64, v65);
  wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  TDIV(v64, v47, v56);
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  // pto: %normal__ssa_v0_pview
  __gm__ float* v66 = PTOAS__GLOBAL_TENSOR_DATA(v37);
  // pto: %normal__ssa_v0_pview
  const int64_t v67 = 0;
  // pto: %normal__ssa_v0_pview
  const int64_t v68 = 256;
  // pto: %normal__ssa_v0_pview
  pto::Shape<1, 1, 1, 1, 256> v69 = pto::Shape<1, 1, 1, 1, 256>();
  // pto: %normal__ssa_v0_pview
  pto::Stride<256, 256, 256, 256, 1> v70 = pto::Stride<256, 256, 256, 256, 1>();
  // pto: %normal__ssa_v0_pview
  GlobalTensor<float, pto::Shape<1, 1, 1, 1, 256>, pto::Stride<256, 256, 256, 256, 1>, pto::Layout::ND> v71 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 256>, pto::Stride<256, 256, 256, 256, 1>, pto::Layout::ND>(v66 + (v67 + v49 * v68), v69, v70);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  TSTORE(v71, v64);
  // pto: %0
  Tile<TileType::Vec, float, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v72 = Tile<TileType::Vec, float, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v12, v11);
  // pto: %0
  uint64_t v73 = (uint64_t) v7;
  TASSIGN(v72, v73);
  pipe_barrier(PIPE_V);
  TDIV<pto::DivAlgorithm::HIGH_PRECISION>(v72, v47, v56);
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
  // pto: %precise__ssa_v0_pview
  __gm__ float* v74 = PTOAS__GLOBAL_TENSOR_DATA(v45);
  // pto: %precise__ssa_v0_pview
  const int64_t v75 = 0;
  // pto: %precise__ssa_v0_pview
  const int64_t v76 = 256;
  // pto: %precise__ssa_v0_pview
  pto::Shape<1, 1, 1, 1, 256> v77 = pto::Shape<1, 1, 1, 1, 256>();
  // pto: %precise__ssa_v0_pview
  pto::Stride<256, 256, 256, 256, 1> v78 = pto::Stride<256, 256, 256, 256, 1>();
  // pto: %precise__ssa_v0_pview
  GlobalTensor<float, pto::Shape<1, 1, 1, 1, 256>, pto::Stride<256, 256, 256, 256, 1>, pto::Layout::ND> v79 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 256>, pto::Stride<256, 256, 256, 256, 1>, pto::Layout::ND>(v74 + (v75 + v49 * v76), v77, v78);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
  TSTORE(v79, v72);
  #endif // __DAV_VEC__

  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}