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

AICORE void csa_rope_sign(__gm__ int32_t* v1, __gm__ int32_t* v2, __gm__ int32_t* v3, __gm__ int32_t* v4, __gm__ int32_t* v5, __gm__ int32_t* v6, __gm__ float* v7, __gm__ float* v8, int64_t v9, int64_t v10, int64_t v11, int64_t v12) {
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  // pto: %c1280_i64
  const int64_t v13 = 1280;
  // pto: %c256_i64
  const int64_t v14 = 256;
  // pto: %c0_i64
  const int64_t v15 = 0;
  // pto: %c1_index
  const int64_t v16 = 1;
  // pto: %c64_index
  const int64_t v17 = 64;
  // pto: %c0_index
  const int64_t v18 = 0;
  // pto: %c4_index
  const int64_t v19 = 4;
  // pto: %cst_7
  const float v20 = 1.0f;
  // pto: %c192_index
  const int64_t v21 = 192;
  // pto: %c0_i32
  const int32_t v22 = 0;
  // pto: %cst_10
  const float v23 = 0.5f;
  // pto: %cst_11
  const float v24 = 2.0f;
  // pto: %idx_sin_signed__ssa_v0_view
  const int64_t v25 = 1;
  // pto: %idx_sin_signed__ssa_v0_view
  const int64_t v26 = 1;
  // pto: %idx_sin_signed__ssa_v0_view
  const int64_t v27 = 1;
  // pto: %idx_sin_signed__ssa_v0_view
  int64_t v28 = (int64_t) v12;
  // pto: %idx_sin_signed__ssa_v0_view
  int64_t v29 = v28 * v17;
  // pto: %idx_sin_signed__ssa_v0_view
  int64_t v30 = v27 * v29;
  // pto: %idx_sin_signed__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v31 = pto::Shape<1, 1, 1, -1, -1>(v25, v26, v27, v28, v17);
  // pto: %idx_sin_signed__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v32 = pto::Stride<-1, -1, -1, -1, -1>(v26 * v30, v30, v29, v17, v16);
  // pto: %idx_sin_signed__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v33 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v7, v31, v32);
  // pto: %freqs_sin__ssa_v0_view
  const int64_t v34 = 1;
  // pto: %freqs_sin__ssa_v0_view
  const int64_t v35 = 1;
  // pto: %freqs_sin__ssa_v0_view
  const int64_t v36 = 1;
  // pto: %freqs_sin__ssa_v0_view
  int64_t v37 = (int64_t) v12;
  // pto: %freqs_sin__ssa_v0_view
  int64_t v38 = v37 * v17;
  // pto: %freqs_sin__ssa_v0_view
  int64_t v39 = v36 * v38;
  // pto: %freqs_sin__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v40 = pto::Shape<1, 1, 1, -1, -1>(v34, v35, v36, v37, v17);
  // pto: %freqs_sin__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v41 = pto::Stride<-1, -1, -1, -1, -1>(v35 * v39, v39, v38, v17, v16);
  // pto: %freqs_sin__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v42 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v8, v40, v41);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  // pto: %prefix_inline189__rv_v2
  int64_t v43;
  v43 = v18;
  for (int64_t v44 = v18; v44 < v10; v44 += v16) {
    // pto: %begin_inline186__tile
    ;
    int32_t v45 = (v2)[v44];
    // pto: %7, %end_inline185__tile
    ;
    int32_t v46 = (v2)[(int64_t) ((uint64_t) v44 + (uint64_t) v16)];
    // pto: %length_inline184__tile
    ;
    int32_t v47 = (v1)[v44];
    // pto: %8, %9, %10, %11
    ;
    int64_t v48 = (int64_t) ((int32_t) (uint32_t) ((int32_t) (uint32_t) v47 + (uint32_t) v45) - (uint32_t) v46) / v19;
    // pto: %12
    ;
    uint64_t v49 = (uint64_t) v43;
    // pto: %12, %13, %14
    ;
    int32_t v50 = (int32_t) ((int64_t) (uint64_t) ((int64_t) v49 - (uint64_t) v48) - (uint64_t) v16);
    (v3)[v44] = v50;
    // pto: %17
    ;
    uint64_t v51 = (uint64_t) v43;
    // pto: %15, %16, %17, %20
    ;
    v43 = (int64_t) ((uint64_t) ((int64_t) v51 + (uint64_t) ((int64_t) v47 / v19)) - (uint64_t) v48);
  }
  // pto: %prefix_inline195__rv_v2
  int64_t v52;
  v52 = v18;
  for (int64_t v53 = v18; v53 < v10; v53 += v16) {
    // pto: %begin_inline192__tile
    ;
    int32_t v54 = (v5)[v53];
    // pto: %22, %end_inline191__tile
    ;
    int32_t v55 = (v5)[(int64_t) ((uint64_t) v53 + (uint64_t) v16)];
    // pto: %length_inline190__tile
    ;
    int32_t v56 = (v4)[v53];
    // pto: %23, %24, %25, %26
    ;
    int64_t v57 = (int64_t) ((int32_t) (uint32_t) ((int32_t) (uint32_t) v56 + (uint32_t) v54) - (uint32_t) v55) / v19;
    // pto: %27
    ;
    uint64_t v58 = (uint64_t) v52;
    // pto: %27, %28, %29
    ;
    int32_t v59 = (int32_t) ((int64_t) (uint64_t) ((int64_t) v58 - (uint64_t) v57) - (uint64_t) v16);
    (v6)[v53] = v59;
    // pto: %32
    ;
    uint64_t v60 = (uint64_t) v52;
    // pto: %30, %31, %32, %35
    ;
    v52 = (int64_t) ((uint64_t) ((int64_t) v60 + (uint64_t) ((int64_t) v56 / v19)) - (uint64_t) v57);
  }
  // pto: %il_ones__tile
  Tile<TileType::Vec, float, 4, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v61 = Tile<TileType::Vec, float, 4, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v19, v17);
  // pto: %il_ones__tile
  uint64_t v62 = (uint64_t) v13;
  TASSIGN(v61, v62);
  TEXPANDS(v61, v20);
  // pto: %t__ci_tmp_v0
  Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v63 = Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v16, v21);
  // pto: %t__ci_tmp_v0
  uint64_t v64 = (uint64_t) v14;
  TASSIGN(v63, v64);
  // pto: %t__tile
  Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v65 = Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v16, v17);
  // pto: %t__tile
  uint64_t v66 = (uint64_t) v15;
  TASSIGN(v65, v66);
  // pto: %ci_tmp_view
  Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, 1, 192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v67;
  TRESHAPE(v67, v63);
  // pto: %ci_dst_view
  Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v68;
  TRESHAPE(v68, v65);
  TCI<Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, 1, 192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, int32_t, 0>(v68, v22, v67);
  set_flag(PIPE_S, PIPE_V, EVENT_ID0);
  // pto: %il_lane_ids__tile
  Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v69 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v16, v17);
  // pto: %il_lane_ids__tile
  uint64_t v70 = (uint64_t) v14;
  TASSIGN(v69, v70);
  wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
  RoundMode v71 = RoundMode::CAST_ROUND;
  SaturationMode v72 = SaturationMode::OFF;
  TCVT(v69, v65, v71, v72);
  // pto: %il_col__tile
  Tile<TileType::Vec, float, 4, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v73 = Tile<TileType::Vec, float, 4, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v19, v17);
  // pto: %il_col__tile
  uint64_t v74 = (uint64_t) v13;
  TASSIGN(v73, v74);
  pipe_barrier(PIPE_V);
  TCOLEXPANDMUL(v73, v61, v69);
  // pto: %0
  Tile<TileType::Vec, float, 4, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v75 = Tile<TileType::Vec, float, 4, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v19, v17);
  // pto: %0
  uint64_t v76 = (uint64_t) v14;
  TASSIGN(v75, v76);
  pipe_barrier(PIPE_V);
  TMULS(v75, v73, v23);
  // pto: %1
  Tile<TileType::Vec, int32_t, 4, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v77 = Tile<TileType::Vec, int32_t, 4, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v19, v17);
  // pto: %1
  uint64_t v78 = (uint64_t) v14;
  TASSIGN(v77, v78);
  pipe_barrier(PIPE_V);
  RoundMode v79 = RoundMode::CAST_TRUNC;
  SaturationMode v80 = SaturationMode::ON;
  TCVT(v77, v75, v79, v80);
  // pto: %il_dup_f__tile
  Tile<TileType::Vec, float, 4, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v81 = Tile<TileType::Vec, float, 4, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v19, v17);
  // pto: %il_dup_f__tile
  uint64_t v82 = (uint64_t) v14;
  TASSIGN(v81, v82);
  pipe_barrier(PIPE_V);
  RoundMode v83 = RoundMode::CAST_ROUND;
  SaturationMode v84 = SaturationMode::OFF;
  TCVT(v81, v77, v83, v84);
  // pto: %2
  Tile<TileType::Vec, float, 4, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v85 = Tile<TileType::Vec, float, 4, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v19, v17);
  // pto: %2
  uint64_t v86 = (uint64_t) v14;
  TASSIGN(v85, v86);
  pipe_barrier(PIPE_V);
  TMULS(v85, v81, v24);
  // pto: %il_lane__tile
  Tile<TileType::Vec, float, 4, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v87 = Tile<TileType::Vec, float, 4, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v19, v17);
  // pto: %il_lane__tile
  uint64_t v88 = (uint64_t) v13;
  TASSIGN(v87, v88);
  pipe_barrier(PIPE_V);
  TSUB(v87, v73, v85);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  // pto: %3
  Tile<TileType::Vec, float, 4, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v89 = Tile<TileType::Vec, float, 4, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v19, v17);
  // pto: %3
  uint64_t v90 = (uint64_t) v13;
  TASSIGN(v89, v90);
  pipe_barrier(PIPE_V);
  TMULS(v89, v87, v24);
  // pto: %il_sign__tile
  Tile<TileType::Vec, float, 4, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v91 = Tile<TileType::Vec, float, 4, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v19, v17);
  // pto: %il_sign__tile
  uint64_t v92 = (uint64_t) v13;
  TASSIGN(v91, v92);
  pipe_barrier(PIPE_V);
  TSUBS(v91, v89, v20);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  for (int64_t v93 = v18; v93 < v12; v93 += v19) {
    // pto: %4
    ;
    Tile<TileType::Vec, float, 4, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v94 = Tile<TileType::Vec, float, 4, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v19, v17);
    // pto: %4
    ;
    uint64_t v95 = (uint64_t) v14;
    TASSIGN(v94, v95);
    // pto: %36
    ;
    int64_t v96 = v93 < v18 ? v18 : v93;
    // pto: %freqs_sin__ssa_v0_pview
    ;
    __gm__ float* v97 = PTOAS__GLOBAL_TENSOR_DATA(v42);
    // pto: %freqs_sin__ssa_v0_pview
    ;
    const int64_t v98 = 0;
    // pto: %freqs_sin__ssa_v0_pview
    ;
    const int64_t v99 = 64;
    // pto: %freqs_sin__ssa_v0_pview
    ;
    pto::Shape<1, 1, 1, 4, 64> v100 = pto::Shape<1, 1, 1, 4, 64>();
    // pto: %freqs_sin__ssa_v0_pview
    ;
    pto::Stride<256, 256, 256, 64, 1> v101 = pto::Stride<256, 256, 256, 64, 1>();
    // pto: %freqs_sin__ssa_v0_pview
    ;
    GlobalTensor<float, pto::Shape<1, 1, 1, 4, 64>, pto::Stride<256, 256, 256, 64, 1>, pto::Layout::ND> v102 = GlobalTensor<float, pto::Shape<1, 1, 1, 4, 64>, pto::Stride<256, 256, 256, 64, 1>, pto::Layout::ND>(v97 + (v98 + v96 * v99), v100, v101);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    TLOAD(v94, v102);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    // pto: %5
    ;
    Tile<TileType::Vec, float, 4, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v103 = Tile<TileType::Vec, float, 4, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v19, v17);
    // pto: %5
    ;
    uint64_t v104 = (uint64_t) v14;
    TASSIGN(v103, v104);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    pipe_barrier(PIPE_V);
    TMUL(v103, v94, v91);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    // pto: %idx_sin_signed__iter_v1_pview
    ;
    __gm__ float* v105 = PTOAS__GLOBAL_TENSOR_DATA(v33);
    // pto: %idx_sin_signed__iter_v1_pview
    ;
    const int64_t v106 = 0;
    // pto: %idx_sin_signed__iter_v1_pview
    ;
    const int64_t v107 = 64;
    // pto: %idx_sin_signed__iter_v1_pview
    ;
    pto::Shape<1, 1, 1, 4, 64> v108 = pto::Shape<1, 1, 1, 4, 64>();
    // pto: %idx_sin_signed__iter_v1_pview
    ;
    pto::Stride<256, 256, 256, 64, 1> v109 = pto::Stride<256, 256, 256, 64, 1>();
    // pto: %idx_sin_signed__iter_v1_pview
    ;
    GlobalTensor<float, pto::Shape<1, 1, 1, 4, 64>, pto::Stride<256, 256, 256, 64, 1>, pto::Layout::ND> v110 = GlobalTensor<float, pto::Shape<1, 1, 1, 4, 64>, pto::Stride<256, 256, 256, 64, 1>, pto::Layout::ND>(v105 + (v106 + v96 * v107), v108, v109);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(v110, v103);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  }
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  pipe_barrier(PIPE_ALL);
  dcci((__gm__ void*)0, cache_line_t::ENTIRE_DATA_CACHE);
  dsb((mem_dsb_t)0);
  #endif // __DAV_VEC__

  pipe_barrier(PIPE_ALL);
  dcci((__gm__ void*)0, cache_line_t::ENTIRE_DATA_CACHE);
  dsb((mem_dsb_t)0);
  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}