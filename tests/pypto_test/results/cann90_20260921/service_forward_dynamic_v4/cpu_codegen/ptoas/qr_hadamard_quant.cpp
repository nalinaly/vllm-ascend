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

AICORE void qr_hadamard_quant(__gm__ int8_t* v1, __gm__ float* v2, __gm__ float* v3, int64_t v4, int64_t v5, int32_t v6, int32_t v7) {
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  // pto: %c49408_i64
  const int64_t v8 = 49408;
  // pto: %c16384_i64
  const int64_t v9 = 16384;
  // pto: %c82176_i64
  const int64_t v10 = 82176;
  // pto: %c0_i64
  const int64_t v11 = 0;
  // pto: %c49152_i64
  const int64_t v12 = 49152;
  // pto: %c24576_index
  const int64_t v13 = 24576;
  // pto: %c128_index
  const int64_t v14 = 128;
  // pto: %c1_index
  const int64_t v15 = 1;
  // pto: %c64_index
  const int64_t v16 = 64;
  // pto: %c48_index
  const int64_t v17 = 48;
  // pto: %c0_index
  const int64_t v18 = 0;
  // pto: %cst_11
  const float v19 = 0.0883883461f;
  // pto: %cst_12
  const float v20 = 9.99999974E-5f;
  // pto: %cst_13
  const float v21 = 127.0f;
  // pto: %qr_hadamard_i8_inline221__ssa_v0_view
  const int64_t v22 = 1;
  // pto: %qr_hadamard_i8_inline221__ssa_v0_view
  const int64_t v23 = 1;
  // pto: %qr_hadamard_i8_inline221__ssa_v0_view
  const int64_t v24 = 1;
  // pto: %qr_hadamard_i8_inline221__ssa_v0_view
  int64_t v25 = v13 * v14;
  // pto: %qr_hadamard_i8_inline221__ssa_v0_view
  int64_t v26 = v24 * v25;
  // pto: %qr_hadamard_i8_inline221__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v27 = pto::Shape<1, 1, 1, -1, -1>(v22, v23, v24, v13, v14);
  // pto: %qr_hadamard_i8_inline221__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v28 = pto::Stride<-1, -1, -1, -1, -1>(v23 * v26, v26, v25, v14, v15);
  // pto: %qr_hadamard_i8_inline221__ssa_v0_view
  GlobalTensor<int8_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v29 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v1, v27, v28);
  // pto: %qr_hadamard_scale_dq_inline219__ssa_v0_view
  const int64_t v30 = 1;
  // pto: %qr_hadamard_scale_dq_inline219__ssa_v0_view
  const int64_t v31 = 1;
  // pto: %qr_hadamard_scale_dq_inline219__ssa_v0_view
  const int64_t v32 = 1;
  // pto: %qr_hadamard_scale_dq_inline219__ssa_v0_view
  int64_t v33 = v13 * v15;
  // pto: %qr_hadamard_scale_dq_inline219__ssa_v0_view
  int64_t v34 = v32 * v33;
  // pto: %qr_hadamard_scale_dq_inline219__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v35 = pto::Shape<1, 1, 1, -1, -1>(v30, v31, v32, v13, v15);
  // pto: %qr_hadamard_scale_dq_inline219__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v36 = pto::Stride<-1, -1, -1, -1, -1>(v31 * v34, v34, v33, v15, v13);
  // pto: %qr_hadamard_scale_dq_inline219__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN> v37 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN>(v2, v35, v36);
  // pto: %qh_acc_gm_inline956_inline2274__rv_v2_view
  const int64_t v38 = 1;
  // pto: %qh_acc_gm_inline956_inline2274__rv_v2_view
  const int64_t v39 = 1;
  // pto: %qh_acc_gm_inline956_inline2274__rv_v2_view
  const int64_t v40 = 1;
  // pto: %qh_acc_gm_inline956_inline2274__rv_v2_view
  int64_t v41 = (int64_t) v5;
  // pto: %qh_acc_gm_inline956_inline2274__rv_v2_view
  int64_t v42 = v41 * v14;
  // pto: %qh_acc_gm_inline956_inline2274__rv_v2_view
  int64_t v43 = v40 * v42;
  // pto: %qh_acc_gm_inline956_inline2274__rv_v2_view
  pto::Shape<1, 1, 1, -1, -1> v44 = pto::Shape<1, 1, 1, -1, -1>(v38, v39, v40, v41, v14);
  // pto: %qh_acc_gm_inline956_inline2274__rv_v2_view
  pto::Stride<-1, -1, -1, -1, -1> v45 = pto::Stride<-1, -1, -1, -1, -1>(v39 * v43, v43, v42, v14, v15);
  // pto: %qh_acc_gm_inline956_inline2274__rv_v2_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v46 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v3, v44, v45);
  // pto: %qh_quant_worker_inline946_inline2316__ssa_v0
  // pto: %6
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  for (int64_t v47 = (int64_t) v6; v47 < (v5 / v16); v47 += v17) {
    // pto: %7
    ;
    int64_t v48 = (int64_t) ((uint64_t) v47 * (uint64_t) v16);
    // pto: %qh_full_f32_inline947_inline2267__tile
    ;
    Tile<TileType::Vec, float, 64, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v49 = Tile<TileType::Vec, float, 64, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v16, v14);
    // pto: %qh_full_f32_inline947_inline2267__tile
    ;
    uint64_t v50 = (uint64_t) v8;
    TASSIGN(v49, v50);
    // pto: %8
    ;
    int64_t v51 = v48 < v18 ? v18 : v48;
    // pto: %qh_acc_gm_inline956_inline2274__rv_v2_pview
    ;
    __gm__ float* v52 = PTOAS__GLOBAL_TENSOR_DATA(v46);
    // pto: %qh_acc_gm_inline956_inline2274__rv_v2_pview
    ;
    const int64_t v53 = 0;
    // pto: %qh_acc_gm_inline956_inline2274__rv_v2_pview
    ;
    const int64_t v54 = 128;
    // pto: %qh_acc_gm_inline956_inline2274__rv_v2_pview
    ;
    pto::Shape<1, 1, 1, 64, 128> v55 = pto::Shape<1, 1, 1, 64, 128>();
    // pto: %qh_acc_gm_inline956_inline2274__rv_v2_pview
    ;
    pto::Stride<8192, 8192, 8192, 128, 1> v56 = pto::Stride<8192, 8192, 8192, 128, 1>();
    // pto: %qh_acc_gm_inline956_inline2274__rv_v2_pview
    ;
    GlobalTensor<float, pto::Shape<1, 1, 1, 64, 128>, pto::Stride<8192, 8192, 8192, 128, 1>, pto::Layout::ND> v57 = GlobalTensor<float, pto::Shape<1, 1, 1, 64, 128>, pto::Stride<8192, 8192, 8192, 128, 1>, pto::Layout::ND>(v52 + (v53 + v51 * v54), v55, v56);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    TLOAD(v49, v57);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    // pto: %t__tile
    ;
    Tile<TileType::Vec, bfloat16_t, 64, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v58 = Tile<TileType::Vec, bfloat16_t, 64, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v16, v14);
    // pto: %t__tile
    ;
    uint64_t v59 = (uint64_t) v9;
    TASSIGN(v58, v59);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    RoundMode v60 = RoundMode::CAST_RINT;
    SaturationMode v61 = SaturationMode::OFF;
    TCVT(v58, v49, v60, v61);
    // pto: %qh_full_f32_v1_inline959_inline2266__tile
    ;
    Tile<TileType::Vec, float, 64, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v62 = Tile<TileType::Vec, float, 64, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v16, v14);
    // pto: %qh_full_f32_v1_inline959_inline2266__tile
    ;
    uint64_t v63 = (uint64_t) v8;
    TASSIGN(v62, v63);
    pipe_barrier(PIPE_V);
    RoundMode v64 = RoundMode::CAST_ROUND;
    SaturationMode v65 = SaturationMode::OFF;
    TCVT(v62, v58, v64, v65);
    // pto: %0
    ;
    Tile<TileType::Vec, float, 64, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v66 = Tile<TileType::Vec, float, 64, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v16, v14);
    // pto: %0
    ;
    uint64_t v67 = (uint64_t) v8;
    TASSIGN(v66, v67);
    pipe_barrier(PIPE_V);
    TMULS(v66, v62, v19);
    // pto: %1
    ;
    Tile<TileType::Vec, bfloat16_t, 64, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v68 = Tile<TileType::Vec, bfloat16_t, 64, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v16, v14);
    // pto: %1
    ;
    uint64_t v69 = (uint64_t) v9;
    TASSIGN(v68, v69);
    pipe_barrier(PIPE_V);
    RoundMode v70 = RoundMode::CAST_RINT;
    SaturationMode v71 = SaturationMode::OFF;
    TCVT(v68, v66, v70, v71);
    // pto: %qh_full_f32_v2_inline952_inline2265__tile
    ;
    Tile<TileType::Vec, float, 64, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v72 = Tile<TileType::Vec, float, 64, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v16, v14);
    // pto: %qh_full_f32_v2_inline952_inline2265__tile
    ;
    uint64_t v73 = (uint64_t) v8;
    TASSIGN(v72, v73);
    pipe_barrier(PIPE_V);
    RoundMode v74 = RoundMode::CAST_ROUND;
    SaturationMode v75 = SaturationMode::OFF;
    TCVT(v72, v68, v74, v75);
    // pto: %qh_amax_inline960_inline2264__tile
    ;
    Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v76 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v15, v16);
    // pto: %qh_amax_inline960_inline2264__tile
    ;
    uint64_t v77 = (uint64_t) v10;
    TASSIGN(v76, v77);
    TEXPANDS(v76, v20);
    for (int64_t v78 = v18; v78 < v14; v78 += v16) {
      // pto: %qh_a_f32_inline948_inline2263__tile_textract
      ;
      Tile<TileType::Vec, float, 64, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v79 = Tile<TileType::Vec, float, 64, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v16, v16);
      // pto: %qh_a_f32_inline948_inline2263__tile_textract
      ;
      uint64_t v80 = (uint64_t) v9;
      TASSIGN(v79, v80);
      pipe_barrier(PIPE_V);
      TEXTRACT(v79, v72, v18, v78);
      // pto: %qh_a_neg_inline964_inline2262__tile
      ;
      Tile<TileType::Vec, float, 64, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v81 = Tile<TileType::Vec, float, 64, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v16, v16);
      // pto: %qh_a_neg_inline964_inline2262__tile
      ;
      uint64_t v82 = (uint64_t) v9;
      TASSIGN(v81, v82);
      pipe_barrier(PIPE_V);
      TNEG(v81, v79);
      // pto: %2
      ;
      Tile<TileType::Vec, float, 64, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v83 = Tile<TileType::Vec, float, 64, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v16, v16);
      // pto: %2
      ;
      uint64_t v84 = (uint64_t) v11;
      TASSIGN(v83, v84);
      TEXTRACT(v83, v72, v18, v78);
      // pto: %qh_a_abs_inline962_inline2285__tile
      ;
      Tile<TileType::Vec, float, 64, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v85 = Tile<TileType::Vec, float, 64, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v16, v16);
      // pto: %qh_a_abs_inline962_inline2285__tile
      ;
      uint64_t v86 = (uint64_t) v11;
      TASSIGN(v85, v86);
      pipe_barrier(PIPE_V);
      TMAX(v85, v83, v81);
      // pto: %tmp_tile
      ;
      Tile<TileType::Vec, float, 64, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v87 = Tile<TileType::Vec, float, 64, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v16, v14);
      // pto: %tmp_tile
      ;
      uint64_t v88 = (uint64_t) v9;
      TASSIGN(v87, v88);
      // pto: %qh_a_max_col_inline965_inline2261__tile
      ;
      Tile<TileType::Vec, float, 64, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v89 = Tile<TileType::Vec, float, 64, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v16, v15);
      // pto: %qh_a_max_col_inline965_inline2261__tile
      ;
      uint64_t v90 = (uint64_t) v12;
      TASSIGN(v89, v90);
      pipe_barrier(PIPE_V);
      TROWMAX(v89, v85, v87);
      // pto: %qh_a_max_inline967_inline2279__tile
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v91 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v15, v16);
      // pto: %qh_a_max_inline967_inline2279__tile
      ;
      uint64_t v92 = (uint64_t) v12;
      TASSIGN(v91, v92);
      // pto: %3
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v93 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v15, v16);
      // pto: %3
      ;
      uint64_t v94 = (uint64_t) v10;
      TASSIGN(v93, v94);
      pipe_barrier(PIPE_V);
      TMAX(v93, v76, v91);
    };
    // pto: %qh_scale_numerator_inline970_inline2260__tile
    ;
    Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v95 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v15, v16);
    // pto: %qh_scale_numerator_inline970_inline2260__tile
    ;
    uint64_t v96 = (uint64_t) v9;
    TASSIGN(v95, v96);
    pipe_barrier(PIPE_V);
    TEXPANDS(v95, v21);
    // pto: %qh_scale_quant_row_inline957_inline2259__tile
    ;
    Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v97 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v15, v16);
    // pto: %qh_scale_quant_row_inline957_inline2259__tile
    ;
    uint64_t v98 = (uint64_t) v9;
    TASSIGN(v97, v98);
    pipe_barrier(PIPE_V);
    TDIV(v97, v95, v76);
    // pto: %qh_scale_recip_inline949_inline2334__tile
    ;
    Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v99 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v15, v16);
    // pto: %qh_scale_recip_inline949_inline2334__tile
    ;
    uint64_t v100 = (uint64_t) v11;
    TASSIGN(v99, v100);
    pipe_barrier(PIPE_V);
    TRECIP(v99, v97);
    // pto: %qh_scale_dq_inline971_inline2258__tile
    ;
    Tile<TileType::Vec, float, 64, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v101 = Tile<TileType::Vec, float, 64, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v16, v15);
    // pto: %qh_scale_dq_inline971_inline2258__tile
    ;
    uint64_t v102 = (uint64_t) v11;
    TASSIGN(v101, v102);
    // pto: %t__rm_a0_tmp_v0
    ;
    Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v103 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v15, v16);
    // pto: %t__rm_a0_tmp_v0
    ;
    uint64_t v104 = (uint64_t) v11;
    TASSIGN(v103, v104);
    // pto: %t__row_major_tmp_v1
    ;
    Tile<TileType::Vec, half, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v105 = Tile<TileType::Vec, half, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v15, v16);
    // pto: %t__row_major_tmp_v1
    ;
    uint64_t v106 = (uint64_t) v10;
    TASSIGN(v105, v106);
    pipe_barrier(PIPE_V);
    RoundMode v107 = RoundMode::CAST_RINT;
    SaturationMode v108 = SaturationMode::OFF;
    TCVT(v105, v103, v107, v108);
    // pto: %4
    ;
    Tile<TileType::Vec, half, 64, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v109 = Tile<TileType::Vec, half, 64, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v16, v15);
    // pto: %4
    ;
    uint64_t v110 = (uint64_t) v10;
    TASSIGN(v109, v110);
    // pto: %t__rm_a0_tmp_v2
    ;
    Tile<TileType::Vec, half, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v111 = Tile<TileType::Vec, half, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v15, v16);
    // pto: %t__rm_a0_tmp_v2
    ;
    uint64_t v112 = (uint64_t) v10;
    TASSIGN(v111, v112);
    // pto: %t__row_major_tmp_v3
    ;
    Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v113 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v15, v16);
    // pto: %t__row_major_tmp_v3
    ;
    uint64_t v114 = (uint64_t) v11;
    TASSIGN(v113, v114);
    pipe_barrier(PIPE_V);
    RoundMode v115 = RoundMode::CAST_ROUND;
    SaturationMode v116 = SaturationMode::OFF;
    TCVT(v113, v111, v115, v116);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    // pto: %5
    ;
    Tile<TileType::Vec, float, 64, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v117 = Tile<TileType::Vec, float, 64, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v16, v15);
    // pto: %5
    ;
    uint64_t v118 = (uint64_t) v11;
    TASSIGN(v117, v118);
    // pto: %qr_hadamard_scale_dq_inline219__iter_v1_pview
    ;
    __gm__ float* v119 = PTOAS__GLOBAL_TENSOR_DATA(v37);
    // pto: %qr_hadamard_scale_dq_inline219__iter_v1_pview
    ;
    const int64_t v120 = 0;
    // pto: %qr_hadamard_scale_dq_inline219__iter_v1_pview
    ;
    pto::Shape<1, 1, 1, 64, 1> v121 = pto::Shape<1, 1, 1, 64, 1>();
    // pto: %qr_hadamard_scale_dq_inline219__iter_v1_pview
    ;
    pto::Stride<64, 64, 64, 1, 24576> v122 = pto::Stride<64, 64, 64, 1, 24576>();
    // pto: %qr_hadamard_scale_dq_inline219__iter_v1_pview
    ;
    GlobalTensor<float, pto::Shape<1, 1, 1, 64, 1>, pto::Stride<64, 64, 64, 1, 24576>, pto::Layout::DN> v123 = GlobalTensor<float, pto::Shape<1, 1, 1, 64, 1>, pto::Stride<64, 64, 64, 1, 24576>, pto::Layout::DN>(v119 + (v120 + v51), v121, v122);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(v123, v117);
    // pto: %qh_scale_quant_inline973_inline2257__tile
    ;
    Tile<TileType::Vec, float, 64, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v124 = Tile<TileType::Vec, float, 64, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v16, v15);
    // pto: %qh_scale_quant_inline973_inline2257__tile
    ;
    uint64_t v125 = (uint64_t) v9;
    TASSIGN(v124, v125);
    // pto: %qh_q_scaled_inline966_inline2310__tile
    ;
    Tile<TileType::Vec, float, 64, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v126 = Tile<TileType::Vec, float, 64, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v16, v14);
    // pto: %qh_q_scaled_inline966_inline2310__tile
    ;
    uint64_t v127 = (uint64_t) v8;
    TASSIGN(v126, v127);
    TROWEXPANDMUL(v126, v72, v124);
    // pto: %qh_q_i32_inline945_inline2256__tile
    ;
    Tile<TileType::Vec, int32_t, 64, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v128 = Tile<TileType::Vec, int32_t, 64, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v16, v14);
    // pto: %qh_q_i32_inline945_inline2256__tile
    ;
    uint64_t v129 = (uint64_t) v8;
    TASSIGN(v128, v129);
    pipe_barrier(PIPE_V);
    RoundMode v130 = RoundMode::CAST_RINT;
    SaturationMode v131 = SaturationMode::ON;
    TCVT(v128, v126, v130, v131);
    // pto: %qh_q_half_inline944_inline2255__tile
    ;
    Tile<TileType::Vec, half, 64, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v132 = Tile<TileType::Vec, half, 64, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v16, v14);
    // pto: %qh_q_half_inline944_inline2255__tile
    ;
    uint64_t v133 = (uint64_t) v8;
    TASSIGN(v132, v133);
    pipe_barrier(PIPE_V);
    RoundMode v134 = RoundMode::CAST_ROUND;
    SaturationMode v135 = SaturationMode::OFF;
    TCVT(v132, v128, v134, v135);
    // pto: %qh_i8_inline943_inline2254__tile
    ;
    Tile<TileType::Vec, int8_t, 64, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v136 = Tile<TileType::Vec, int8_t, 64, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v16, v14);
    // pto: %qh_i8_inline943_inline2254__tile
    ;
    uint64_t v137 = (uint64_t) v8;
    TASSIGN(v136, v137);
    pipe_barrier(PIPE_V);
    RoundMode v138 = RoundMode::CAST_TRUNC;
    SaturationMode v139 = SaturationMode::ON;
    TCVT(v136, v132, v138, v139);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
    // pto: %qr_hadamard_i8_inline221__iter_v1_pview
    ;
    __gm__ int8_t* v140 = PTOAS__GLOBAL_TENSOR_DATA(v29);
    // pto: %qr_hadamard_i8_inline221__iter_v1_pview
    ;
    const int64_t v141 = 0;
    // pto: %qr_hadamard_i8_inline221__iter_v1_pview
    ;
    const int64_t v142 = 128;
    // pto: %qr_hadamard_i8_inline221__iter_v1_pview
    ;
    pto::Shape<1, 1, 1, 64, 128> v143 = pto::Shape<1, 1, 1, 64, 128>();
    // pto: %qr_hadamard_i8_inline221__iter_v1_pview
    ;
    pto::Stride<8192, 8192, 8192, 128, 1> v144 = pto::Stride<8192, 8192, 8192, 128, 1>();
    // pto: %qr_hadamard_i8_inline221__iter_v1_pview
    ;
    GlobalTensor<int8_t, pto::Shape<1, 1, 1, 64, 128>, pto::Stride<8192, 8192, 8192, 128, 1>, pto::Layout::ND> v145 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 64, 128>, pto::Stride<8192, 8192, 8192, 128, 1>, pto::Layout::ND>(v140 + (v141 + v51 * v142), v143, v144);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
    TSTORE(v145, v136);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  }
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  #endif // __DAV_VEC__

  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}