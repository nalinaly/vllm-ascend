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

AICORE void idx_qr_dequant_rope(__gm__ bfloat16_t* v1, __gm__ float* v2, __gm__ float* v3, __gm__ float* v4, __gm__ float* v5, __gm__ int32_t* v6, int64_t v7, int64_t v8, int64_t v9, int64_t v10, int32_t v11, int32_t v12) {
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  const int64_t v13 = 256;
  const int64_t v14 = 512;
  // pto: %c6176_i64
  const int64_t v15 = 6176;
  // pto: %c10272_i64
  const int64_t v16 = 10272;
  // pto: %c14368_i64
  const int64_t v17 = 14368;
  // pto: %c0_i64
  const int64_t v18 = 0;
  // pto: %c2048_i64
  const int64_t v19 = 2048;
  // pto: %c2080_i64
  const int64_t v20 = 2080;
  // pto: %c4128_i64
  const int64_t v21 = 4128;
  // pto: %c18464_i64
  const int64_t v22 = 18464;
  // pto: %c24096_i64
  const int64_t v23 = 24096;
  const int64_t v24 = 6432;
  // pto: %c19488_i64
  const int64_t v25 = 19488;
  // pto: %c19744_i64
  const int64_t v26 = 19744;
  // pto: %c20000_i64
  const int64_t v27 = 20000;
  const int64_t v28 = 10528;
  // pto: %c25120_i64
  const int64_t v29 = 25120;
  // pto: %c25376_i64
  const int64_t v30 = 25376;
  const int64_t v31 = 14624;
  // pto: %c384_index
  const int64_t v32 = 384;
  // pto: %c8192_index
  const int64_t v33 = 8192;
  // pto: %c1_index
  const int64_t v34 = 1;
  // pto: %c64_index
  const int64_t v35 = 64;
  // pto: %c8_index
  const int64_t v36 = 8;
  // pto: %cst_22
  const float v37 = 1.0f;
  // pto: %c192_index
  const int64_t v38 = 192;
  // pto: %c0_i32
  const int32_t v39 = 0;
  // pto: %cst_25
  const float v40 = 0.5f;
  // pto: %cst_26
  const float v41 = 2.0f;
  // pto: %c16_index
  const int64_t v42 = 16;
  // pto: %c4_index
  const int64_t v43 = 4;
  // pto: %c0_index
  const int64_t v44 = 0;
  // pto: %c2_index
  const int64_t v45 = 2;
  // pto: %c128_index
  const int64_t v46 = 128;
  // pto: %qr_bf16_2d_inline897_inline2328__ssa_v0_view
  const int64_t v47 = 1;
  // pto: %qr_bf16_2d_inline897_inline2328__ssa_v0_view
  const int64_t v48 = 1;
  // pto: %qr_bf16_2d_inline897_inline2328__ssa_v0_view
  const int64_t v49 = 1;
  // pto: %qr_bf16_2d_inline897_inline2328__ssa_v0_view
  int64_t v50 = v32 * v33;
  // pto: %qr_bf16_2d_inline897_inline2328__ssa_v0_view
  int64_t v51 = v49 * v50;
  // pto: %qr_bf16_2d_inline897_inline2328__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v52 = pto::Shape<1, 1, 1, -1, -1>(v47, v48, v49, v32, v33);
  // pto: %qr_bf16_2d_inline897_inline2328__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v53 = pto::Stride<-1, -1, -1, -1, -1>(v48 * v51, v51, v50, v33, v34);
  // pto: %qr_bf16_2d_inline897_inline2328__ssa_v0_view
  GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v54 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v1, v52, v53);
  // pto: %qr_scale__ssa_v0_view
  const int64_t v55 = 1;
  // pto: %qr_scale__ssa_v0_view
  const int64_t v56 = 1;
  // pto: %qr_scale__ssa_v0_view
  const int64_t v57 = 1;
  // pto: %qr_scale__ssa_v0_view
  int64_t v58 = (int64_t) v10;
  // pto: %qr_scale__ssa_v0_view
  int64_t v59 = v58 * v34;
  // pto: %qr_scale__ssa_v0_view
  int64_t v60 = v57 * v59;
  // pto: %qr_scale__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v61 = pto::Shape<1, 1, 1, -1, -1>(v55, v56, v57, v58, v34);
  // pto: %qr_scale__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v62 = pto::Stride<-1, -1, -1, -1, -1>(v56 * v60, v60, v59, v34, (int64_t) v10);
  // pto: %qr_scale__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN> v63 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN>(v2, v61, v62);
  // pto: %freqs_cos__ssa_v0_view
  const int64_t v64 = 1;
  // pto: %freqs_cos__ssa_v0_view
  const int64_t v65 = 1;
  // pto: %freqs_cos__ssa_v0_view
  const int64_t v66 = 1;
  // pto: %freqs_cos__ssa_v0_view
  int64_t v67 = (int64_t) v10;
  // pto: %freqs_cos__ssa_v0_view
  int64_t v68 = v67 * v35;
  // pto: %freqs_cos__ssa_v0_view
  int64_t v69 = v66 * v68;
  // pto: %freqs_cos__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v70 = pto::Shape<1, 1, 1, -1, -1>(v64, v65, v66, v67, v35);
  // pto: %freqs_cos__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v71 = pto::Stride<-1, -1, -1, -1, -1>(v65 * v69, v69, v68, v35, v34);
  // pto: %freqs_cos__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v72 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v3, v70, v71);
  // pto: %idx_sin_signed__ssa_v0_view
  const int64_t v73 = 1;
  // pto: %idx_sin_signed__ssa_v0_view
  const int64_t v74 = 1;
  // pto: %idx_sin_signed__ssa_v0_view
  const int64_t v75 = 1;
  // pto: %idx_sin_signed__ssa_v0_view
  int64_t v76 = (int64_t) v10;
  // pto: %idx_sin_signed__ssa_v0_view
  int64_t v77 = v76 * v35;
  // pto: %idx_sin_signed__ssa_v0_view
  int64_t v78 = v75 * v77;
  // pto: %idx_sin_signed__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v79 = pto::Shape<1, 1, 1, -1, -1>(v73, v74, v75, v76, v35);
  // pto: %idx_sin_signed__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v80 = pto::Stride<-1, -1, -1, -1, -1>(v74 * v78, v78, v77, v35, v34);
  // pto: %idx_sin_signed__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v81 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v4, v79, v80);
  // pto: %idx_wq_b_scale__ssa_v0_view
  const int64_t v82 = 1;
  // pto: %idx_wq_b_scale__ssa_v0_view
  const int64_t v83 = 1;
  // pto: %idx_wq_b_scale__ssa_v0_view
  const int64_t v84 = 1;
  // pto: %idx_wq_b_scale__ssa_v0_view
  const int64_t v85 = 1;
  // pto: %idx_wq_b_scale__ssa_v0_view
  int64_t v86 = v33 * v34;
  // pto: %idx_wq_b_scale__ssa_v0_view
  int64_t v87 = v85 * v86;
  // pto: %idx_wq_b_scale__ssa_v0_view
  int64_t v88 = v84 * v87;
  // pto: %idx_wq_b_scale__ssa_v0_view
  pto::Shape<1, 1, 1, 1, -1> v89 = pto::Shape<1, 1, 1, 1, -1>(v82, v83, v84, v85, v33);
  // pto: %idx_wq_b_scale__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v90 = pto::Stride<-1, -1, -1, -1, -1>(v83 * v88, v88, v87, v86, v34);
  // pto: %idx_wq_b_scale__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v91 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v5, v89, v90);
  // pto: %qr_acc_pad_inline922_inline2319__rv_v2_view
  const int64_t v92 = 1;
  // pto: %qr_acc_pad_inline922_inline2319__rv_v2_view
  const int64_t v93 = 1;
  // pto: %qr_acc_pad_inline922_inline2319__rv_v2_view
  const int64_t v94 = 1;
  // pto: %qr_acc_pad_inline922_inline2319__rv_v2_view
  int64_t v95 = v32 * v33;
  // pto: %qr_acc_pad_inline922_inline2319__rv_v2_view
  int64_t v96 = v94 * v95;
  // pto: %qr_acc_pad_inline922_inline2319__rv_v2_view
  pto::Shape<1, 1, 1, -1, -1> v97 = pto::Shape<1, 1, 1, -1, -1>(v92, v93, v94, v32, v33);
  // pto: %qr_acc_pad_inline922_inline2319__rv_v2_view
  pto::Stride<-1, -1, -1, -1, -1> v98 = pto::Stride<-1, -1, -1, -1, -1>(v93 * v96, v96, v95, v33, v34);
  // pto: %qr_acc_pad_inline922_inline2319__rv_v2_view
  GlobalTensor<int32_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v99 = GlobalTensor<int32_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v6, v97, v98);
  // pto: %dq_rope_worker_inline907_inline2293__ssa_v0
  // pto: %sw_ones_inline920_inline2337__tile
  Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v100 = Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v36, v35);
  // pto: %sw_ones_inline920_inline2337__tile
  uint64_t v101 = (uint64_t) v15;
  TASSIGN(v100, v101);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  TEXPANDS(v100, v37);
  // pto: %t__ci_tmp_v0
  Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v102 = Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v38);
  // pto: %t__ci_tmp_v0
  uint64_t v103 = (uint64_t) v16;
  TASSIGN(v102, v103);
  // pto: %t__tile
  Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v104 = Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v35);
  // pto: %t__tile
  uint64_t v105 = (uint64_t) v17;
  TASSIGN(v104, v105);
  // pto: %ci_tmp_view
  Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, 1, 192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v106;
  TRESHAPE(v106, v102);
  // pto: %ci_dst_view
  Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v107;
  TRESHAPE(v107, v104);
  TCI<Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, 1, 192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, int32_t, 0>(v107, v39, v106);
  set_flag(PIPE_S, PIPE_V, EVENT_ID0);
  // pto: %sw_index_inline893_inline2348__tile
  Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v108 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v35);
  // pto: %sw_index_inline893_inline2348__tile
  uint64_t v109 = (uint64_t) v16;
  TASSIGN(v108, v109);
  wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
  RoundMode v110 = RoundMode::CAST_ROUND;
  SaturationMode v111 = SaturationMode::OFF;
  TCVT(v108, v104, v110, v111);
  // pto: %sw_col_inline905_inline2317__tile
  Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v112 = Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v36, v35);
  // pto: %sw_col_inline905_inline2317__tile
  uint64_t v113 = (uint64_t) v15;
  TASSIGN(v112, v113);
  pipe_barrier(PIPE_V);
  TCOLEXPANDMUL(v112, v100, v108);
  // pto: %0
  Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v114 = Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v36, v35);
  // pto: %0
  uint64_t v115 = (uint64_t) v16;
  TASSIGN(v114, v115);
  pipe_barrier(PIPE_V);
  TMULS(v114, v112, v40);
  // pto: %1
  Tile<TileType::Vec, int32_t, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v116 = Tile<TileType::Vec, int32_t, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v36, v35);
  // pto: %1
  uint64_t v117 = (uint64_t) v16;
  TASSIGN(v116, v117);
  pipe_barrier(PIPE_V);
  RoundMode v118 = RoundMode::CAST_TRUNC;
  SaturationMode v119 = SaturationMode::ON;
  TCVT(v116, v114, v118, v119);
  // pto: %sw_dup_f_inline923_inline2289__tile
  Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v120 = Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v36, v35);
  // pto: %sw_dup_f_inline923_inline2289__tile
  uint64_t v121 = (uint64_t) v16;
  TASSIGN(v120, v121);
  pipe_barrier(PIPE_V);
  RoundMode v122 = RoundMode::CAST_ROUND;
  SaturationMode v123 = SaturationMode::OFF;
  TCVT(v120, v116, v122, v123);
  // pto: %2
  Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v124 = Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v36, v35);
  // pto: %2
  uint64_t v125 = (uint64_t) v16;
  TASSIGN(v124, v125);
  pipe_barrier(PIPE_V);
  TMULS(v124, v120, v41);
  // pto: %sw_lane_inline944_inline2296__tile
  Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v126 = Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v36, v35);
  // pto: %sw_lane_inline944_inline2296__tile
  uint64_t v127 = (uint64_t) v16;
  TASSIGN(v126, v127);
  pipe_barrier(PIPE_V);
  TSUB(v126, v112, v124);
  // pto: %3
  Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v128 = Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v36, v35);
  // pto: %3
  uint64_t v129 = (uint64_t) v15;
  TASSIGN(v128, v129);
  pipe_barrier(PIPE_V);
  TADDS(v128, v112, v37);
  // pto: %4
  Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v130 = Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v36, v35);
  // pto: %4
  uint64_t v131 = (uint64_t) v16;
  TASSIGN(v130, v131);
  TMULS(v130, v126, v41);
  // pto: %5
  Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v132 = Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v36, v35);
  // pto: %5
  uint64_t v133 = (uint64_t) v15;
  TASSIGN(v132, v133);
  pipe_barrier(PIPE_V);
  TSUB(v132, v128, v130);
  // pto: %rope_swap_idx_inline894_inline2286__tile
  Tile<TileType::Vec, int32_t, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v134 = Tile<TileType::Vec, int32_t, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v36, v35);
  // pto: %rope_swap_idx_inline894_inline2286__tile
  uint64_t v135 = (uint64_t) v18;
  TASSIGN(v134, v135);
  pipe_barrier(PIPE_V);
  RoundMode v136 = RoundMode::CAST_ROUND;
  SaturationMode v137 = SaturationMode::ON;
  TCVT(v134, v132, v136, v137);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID7);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID6);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID5);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID4);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
  for (int64_t v138 = (int64_t) v11; v138 < v7; v138 += v8) {
    // pto: %72, %73
    ;
    int64_t v139 = (int64_t) ((uint64_t) (v138 % v42) * (uint64_t) v43);
    // pto: %74, %75
    ;
    int64_t v140 = (int64_t) ((uint64_t) (v138 / v42) * (uint64_t) v36);
    // pto: %76, %77
    ;
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID4);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID5);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
    if ((int64_t) ((uint64_t) v140 + (uint64_t) v36) <= v9) {
      // pto: %qr_scale_tile_inline931_inline2298__tile
      ;
      Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v141 = Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v36, v34);
      // pto: %qr_scale_tile_inline931_inline2298__tile
      ;
      uint64_t v142 = (uint64_t) v19;
      TASSIGN(v141, v142);
      // pto: %78
      ;
      int64_t v143 = v140 < v44 ? v44 : v140;
      // pto: %qr_scale__ssa_v0_pview
      ;
      int64_t v144 = (int64_t) v10;
      // pto: %qr_scale__ssa_v0_pview
      ;
      const int64_t v145 = 0;
      // pto: %qr_scale__ssa_v0_pview
      ;
      __gm__ float* v146 = PTOAS__GLOBAL_TENSOR_DATA(v63);
      // pto: %qr_scale__ssa_v0_pview
      ;
      const int64_t v147 = 1;
      // pto: %qr_scale__ssa_v0_pview
      ;
      const int64_t v148 = 1;
      // pto: %qr_scale__ssa_v0_pview
      ;
      const int64_t v149 = 1;
      // pto: %qr_scale__ssa_v0_pview
      ;
      int64_t v150 = v36 * v34;
      // pto: %qr_scale__ssa_v0_pview
      ;
      int64_t v151 = v149 * v150;
      // pto: %qr_scale__ssa_v0_pview
      ;
      pto::Shape<1, 1, 1, 8, 1> v152 = pto::Shape<1, 1, 1, 8, 1>(v147, v148, v149, v36, v34);
      // pto: %qr_scale__ssa_v0_pview
      ;
      pto::Stride<-1, -1, -1, -1, -1> v153 = pto::Stride<-1, -1, -1, -1, -1>(v148 * v151, v151, v150, v34, v144);
      // pto: %qr_scale__ssa_v0_pview
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 8, 1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN> v154 = GlobalTensor<float, pto::Shape<1, 1, 1, 8, 1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN>(v146 + (v145 + v143 * v34 + v44 * v144), v152, v153);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID6);
      TLOAD(v141, v154);
      // pto: %cos_tile_inline932_inline2331__tile
      ;
      Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v155 = Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v36, v35);
      // pto: %cos_tile_inline932_inline2331__tile
      ;
      uint64_t v156 = (uint64_t) v20;
      TASSIGN(v155, v156);
      // pto: %freqs_cos__ssa_v0_pview
      ;
      __gm__ float* v157 = PTOAS__GLOBAL_TENSOR_DATA(v72);
      // pto: %freqs_cos__ssa_v0_pview
      ;
      const int64_t v158 = 0;
      // pto: %freqs_cos__ssa_v0_pview
      ;
      const int64_t v159 = 64;
      // pto: %freqs_cos__ssa_v0_pview
      ;
      pto::Shape<1, 1, 1, 8, 64> v160 = pto::Shape<1, 1, 1, 8, 64>();
      // pto: %freqs_cos__ssa_v0_pview
      ;
      pto::Stride<512, 512, 512, 64, 1> v161 = pto::Stride<512, 512, 512, 64, 1>();
      // pto: %freqs_cos__ssa_v0_pview
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 8, 64>, pto::Stride<512, 512, 512, 64, 1>, pto::Layout::ND> v162 = GlobalTensor<float, pto::Shape<1, 1, 1, 8, 64>, pto::Stride<512, 512, 512, 64, 1>, pto::Layout::ND>(v157 + (v158 + v143 * v159), v160, v161);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID7);
      TLOAD(v155, v162);
      // pto: %sin_tile_inline933_inline2306__tile
      ;
      Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v163 = Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v36, v35);
      // pto: %sin_tile_inline933_inline2306__tile
      ;
      uint64_t v164 = (uint64_t) v21;
      TASSIGN(v163, v164);
      // pto: %idx_sin_signed__ssa_v0_pview
      ;
      __gm__ float* v165 = PTOAS__GLOBAL_TENSOR_DATA(v81);
      // pto: %idx_sin_signed__ssa_v0_pview
      ;
      const int64_t v166 = 0;
      // pto: %idx_sin_signed__ssa_v0_pview
      ;
      const int64_t v167 = 64;
      // pto: %idx_sin_signed__ssa_v0_pview
      ;
      pto::Shape<1, 1, 1, 8, 64> v168 = pto::Shape<1, 1, 1, 8, 64>();
      // pto: %idx_sin_signed__ssa_v0_pview
      ;
      pto::Stride<512, 512, 512, 64, 1> v169 = pto::Stride<512, 512, 512, 64, 1>();
      // pto: %idx_sin_signed__ssa_v0_pview
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 8, 64>, pto::Stride<512, 512, 512, 64, 1>, pto::Layout::ND> v170 = GlobalTensor<float, pto::Shape<1, 1, 1, 8, 64>, pto::Stride<512, 512, 512, 64, 1>, pto::Layout::ND>(v165 + (v166 + v143 * v167), v168, v169);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      TLOAD(v163, v170);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID6);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID5);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID4);
      for (int64_t v171 = v44; v171 < v43; v171 += v45) {
        // pto: %81, %82
        ;
        int64_t v172 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) v139 + (uint64_t) v171) * (uint64_t) v46);
        // pto: %84, %83, %85
        ;
        int64_t v173 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) v139 + (uint64_t) ((int64_t) (uint64_t) v171 + (uint64_t) v34)) * (uint64_t) v46);
        // pto: %6
        ;
        Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v174 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v46);
        // pto: %6
        ;
        uint64_t v175 = (uint64_t) v22;
        TASSIGN(v174, v175);
        // pto: %86
        ;
        int64_t v176 = v172 < v44 ? v44 : v172;
        // pto: %idx_wq_b_scale__ssa_v0_pview
        ;
        __gm__ float* v177 = PTOAS__GLOBAL_TENSOR_DATA(v91);
        // pto: %idx_wq_b_scale__ssa_v0_pview
        ;
        const int64_t v178 = 0;
        // pto: %idx_wq_b_scale__ssa_v0_pview
        ;
        pto::Shape<1, 1, 1, 1, 128> v179 = pto::Shape<1, 1, 1, 1, 128>();
        // pto: %idx_wq_b_scale__ssa_v0_pview
        ;
        pto::Stride<128, 128, 128, 128, 1> v180 = pto::Stride<128, 128, 128, 128, 1>();
        // pto: %idx_wq_b_scale__ssa_v0_pview
        ;
        GlobalTensor<float, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<128, 128, 128, 128, 1>, pto::Layout::ND> v181 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<128, 128, 128, 128, 1>, pto::Layout::ND>(v177 + (v178 + v176), v179, v180);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
        TLOAD(v174, v181);
        // pto: %7
        ;
        Tile<TileType::Vec, int32_t, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v182 = Tile<TileType::Vec, int32_t, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v36, v46);
        // pto: %7
        ;
        uint64_t v183 = (uint64_t) v15;
        TASSIGN(v182, v183);
        // pto: %qr_acc_pad_inline922_inline2319__rv_v2_pview
        ;
        __gm__ int32_t* v184 = PTOAS__GLOBAL_TENSOR_DATA(v99);
        // pto: %qr_acc_pad_inline922_inline2319__rv_v2_pview
        ;
        const int64_t v185 = 0;
        // pto: %qr_acc_pad_inline922_inline2319__rv_v2_pview
        ;
        const int64_t v186 = 8192;
        // pto: %qr_acc_pad_inline922_inline2319__rv_v2_pview
        ;
        pto::Shape<1, 1, 1, 8, 128> v187 = pto::Shape<1, 1, 1, 8, 128>();
        // pto: %qr_acc_pad_inline922_inline2319__rv_v2_pview
        ;
        pto::Stride<65536, 65536, 65536, 8192, 1> v188 = pto::Stride<65536, 65536, 65536, 8192, 1>();
        // pto: %qr_acc_pad_inline922_inline2319__rv_v2_pview
        ;
        GlobalTensor<int32_t, pto::Shape<1, 1, 1, 8, 128>, pto::Stride<65536, 65536, 65536, 8192, 1>, pto::Layout::ND> v189 = GlobalTensor<int32_t, pto::Shape<1, 1, 1, 8, 128>, pto::Stride<65536, 65536, 65536, 8192, 1>, pto::Layout::ND>(v184 + (v185 + v143 * v186 + v176), v187, v188);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID4);
        TLOAD(v182, v189);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        // pto: %8
        ;
        Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v190 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v46);
        // pto: %8
        ;
        uint64_t v191 = (uint64_t) v23;
        TASSIGN(v190, v191);
        // pto: %89
        ;
        int64_t v192 = v173 < v44 ? v44 : v173;
        // pto: %90
        ;
        __gm__ float* v193 = PTOAS__GLOBAL_TENSOR_DATA(v91);
        // pto: %90
        ;
        const int64_t v194 = 0;
        // pto: %90
        ;
        pto::Shape<1, 1, 1, 1, 128> v195 = pto::Shape<1, 1, 1, 1, 128>();
        // pto: %90
        ;
        pto::Stride<128, 128, 128, 128, 1> v196 = pto::Stride<128, 128, 128, 128, 1>();
        // pto: %90
        ;
        GlobalTensor<float, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<128, 128, 128, 128, 1>, pto::Layout::ND> v197 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<128, 128, 128, 128, 1>, pto::Layout::ND>(v193 + (v194 + v192), v195, v196);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID5);
        TLOAD(v190, v197);
        // pto: %9
        ;
        Tile<TileType::Vec, int32_t, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v198 = Tile<TileType::Vec, int32_t, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v36, v46);
        // pto: %9
        ;
        uint64_t v199 = (uint64_t) v16;
        TASSIGN(v198, v199);
        // pto: %93
        ;
        __gm__ int32_t* v200 = PTOAS__GLOBAL_TENSOR_DATA(v99);
        // pto: %93
        ;
        const int64_t v201 = 0;
        // pto: %93
        ;
        const int64_t v202 = 8192;
        // pto: %93
        ;
        pto::Shape<1, 1, 1, 8, 128> v203 = pto::Shape<1, 1, 1, 8, 128>();
        // pto: %93
        ;
        pto::Stride<65536, 65536, 65536, 8192, 1> v204 = pto::Stride<65536, 65536, 65536, 8192, 1>();
        // pto: %93
        ;
        GlobalTensor<int32_t, pto::Shape<1, 1, 1, 8, 128>, pto::Stride<65536, 65536, 65536, 8192, 1>, pto::Layout::ND> v205 = GlobalTensor<int32_t, pto::Shape<1, 1, 1, 8, 128>, pto::Stride<65536, 65536, 65536, 8192, 1>, pto::Layout::ND>(v200 + (v201 + v143 * v202 + v192), v203, v204);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID6);
        TLOAD(v198, v205);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        // pto: %wq_scale_inline936_inline2349__tile
        ;
        Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v206 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v46);
        // pto: %wq_scale_inline936_inline2349__tile
        ;
        uint64_t v207 = (uint64_t) v22;
        TASSIGN(v206, v207);
        // pto: %acc_fp32_inline924_inline2327__tile
        ;
        Tile<TileType::Vec, float, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v208 = Tile<TileType::Vec, float, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v36, v46);
        // pto: %acc_fp32_inline924_inline2327__tile
        ;
        uint64_t v209 = (uint64_t) v15;
        TASSIGN(v208, v209);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        RoundMode v210 = RoundMode::CAST_NONE;
        SaturationMode v211 = SaturationMode::OFF;
        TCVT(v208, v182, v210, v211);
        // pto: %10
        ;
        Tile<TileType::Vec, float, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v212 = Tile<TileType::Vec, float, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v36, v46);
        // pto: %10
        ;
        uint64_t v213 = (uint64_t) v17;
        TASSIGN(v212, v213);
        TEXPANDS(v212, v37);
        // pto: %11
        ;
        Tile<TileType::Vec, float, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v214 = Tile<TileType::Vec, float, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v36, v46);
        // pto: %11
        ;
        uint64_t v215 = (uint64_t) v17;
        TASSIGN(v214, v215);
        pipe_barrier(PIPE_V);
        TROWEXPANDMUL(v214, v212, v141);
        // pto: %qr_dequant_scale_inline939_inline2314__tile
        ;
        Tile<TileType::Vec, float, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v216 = Tile<TileType::Vec, float, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v36, v46);
        // pto: %qr_dequant_scale_inline939_inline2314__tile
        ;
        uint64_t v217 = (uint64_t) v17;
        TASSIGN(v216, v217);
        pipe_barrier(PIPE_V);
        TCOLEXPANDMUL(v216, v214, v206);
        // pto: %qr_dequant_inline940_inline2355__tile
        ;
        Tile<TileType::Vec, float, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v218 = Tile<TileType::Vec, float, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v36, v46);
        // pto: %qr_dequant_inline940_inline2355__tile
        ;
        uint64_t v219 = (uint64_t) v15;
        TASSIGN(v218, v219);
        pipe_barrier(PIPE_V);
        TMUL(v218, v208, v216);
        // pto: %12
        ;
        Tile<TileType::Vec, bfloat16_t, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v220 = Tile<TileType::Vec, bfloat16_t, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v36, v46);
        // pto: %12
        ;
        uint64_t v221 = (uint64_t) v17;
        TASSIGN(v220, v221);
        pipe_barrier(PIPE_V);
        RoundMode v222 = RoundMode::CAST_RINT;
        SaturationMode v223 = SaturationMode::OFF;
        TCVT(v220, v218, v222, v223);
        // pto: %qr_dequant_v1_inline921_inline2305__tile
        ;
        Tile<TileType::Vec, float, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v224 = Tile<TileType::Vec, float, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v36, v46);
        // pto: %qr_dequant_v1_inline921_inline2305__tile
        ;
        uint64_t v225 = (uint64_t) v15;
        TASSIGN(v224, v225);
        pipe_barrier(PIPE_V);
        RoundMode v226 = RoundMode::CAST_ROUND;
        SaturationMode v227 = SaturationMode::OFF;
        TCVT(v224, v220, v226, v227);
        // pto: %13
        ;
        Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v228 = Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v36, v35);
        // pto: %13
        ;
        uint64_t v229 = (uint64_t) v15;
        TASSIGN(v228, v229);
        // pto: %slice_view
        ;
        Tile<TileType::Vec, float, 8, 128, BLayout::RowMajor, 8, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v230;
        // pto: %slice_view
        ;
        uint64_t v231 = (uint64_t) v15;
        TASSIGN(v230, v231);
        // pto: %qr_nope_bf16_inline916_inline2361__tile
        ;
        Tile<TileType::Vec, bfloat16_t, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v232 = Tile<TileType::Vec, bfloat16_t, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v36, v35);
        // pto: %qr_nope_bf16_inline916_inline2361__tile
        ;
        uint64_t v233 = (uint64_t) v22;
        TASSIGN(v232, v233);
        pipe_barrier(PIPE_V);
        RoundMode v234 = RoundMode::CAST_RINT;
        SaturationMode v235 = SaturationMode::OFF;
        TCVT(v232, v230, v234, v235);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        // pto: %qr_rope_slice_inline941_inline2301__tile
        ;
        Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v236 = Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v36, v35);
        // pto: %qr_rope_slice_inline941_inline2301__tile
        ;
        uint64_t v237 = (uint64_t) v24;
        TASSIGN(v236, v237);
        // pto: %94
        ;
        Tile<TileType::Vec, float, 8, 128, BLayout::RowMajor, 8, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v238;
        // pto: %94
        ;
        uint64_t v239 = (uint64_t) v24;
        TASSIGN(v238, v239);
        // pto: %gather_acc_init
        ;
        Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v240 = Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v36, v35);
        // pto: %gather_acc_init
        ;
        uint64_t v241 = (uint64_t) v17;
        TASSIGN(v240, v241);
        for (int64_t v242 = v44; v242 < v36; v242 += v34) {
          // pto: %gather_inp_row
          ;
          Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v243 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v35);
          // pto: %gather_inp_row
          ;
          uint64_t v244 = (uint64_t) v15;
          TASSIGN(v243, v244);
          // pto: %95
          ;
          Tile<TileType::Vec, float, 8, 128, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v245;
          // pto: %95
          ;
          uint64_t v246 = (uint64_t) ((int64_t) (uint64_t) ((int64_t) (uint64_t) v242 * (uint64_t) v14) + (uint64_t) v24);
          TASSIGN(v245, v246);
          // pto: %gather_idx_row
          ;
          Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v247 = Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v35);
          // pto: %gather_idx_row
          ;
          uint64_t v248 = (uint64_t) v18;
          TASSIGN(v247, v248);
          // pto: %96
          ;
          int64_t v249 = (int64_t) ((uint64_t) v242 * (uint64_t) v13);
          // pto: %96
          ;
          Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v250;
          // pto: %96
          ;
          uint64_t v251 = (uint64_t) v249;
          TASSIGN(v250, v251);
          // pto: %gather_row_tmp
          ;
          Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v252 = Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v35);
          // pto: %gather_row_tmp
          ;
          uint64_t v253 = (uint64_t) v25;
          TASSIGN(v252, v253);
          // pto: %gather_row
          ;
          Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v254 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v35);
          // pto: %gather_row
          ;
          uint64_t v255 = (uint64_t) v26;
          TASSIGN(v254, v255);
          pipe_barrier(PIPE_V);
          TGATHER(v254, v245, v250, v252);
          // pto: %assemble_view
          ;
          Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v256;
          // pto: %assemble_view
          ;
          uint64_t v257 = (uint64_t) ((int64_t) (uint64_t) v249 + (uint64_t) v17);
          TASSIGN(v256, v257);
          pipe_barrier(PIPE_V);
          TMOV(v256, v254);
        };
        // pto: %qr_swapped_inline928_inline2336__tile
        ;
        Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v258 = Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v36, v35);
        // pto: %qr_swapped_inline928_inline2336__tile
        ;
        uint64_t v259 = (uint64_t) v17;
        TASSIGN(v258, v259);
        // pto: %14
        ;
        Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v260 = Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v36, v35);
        // pto: %14
        ;
        uint64_t v261 = (uint64_t) v15;
        TASSIGN(v260, v261);
        pipe_barrier(PIPE_V);
        TMUL(v260, v238, v155);
        // pto: %15
        ;
        Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v262 = Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v36, v35);
        // pto: %15
        ;
        uint64_t v263 = (uint64_t) v17;
        TASSIGN(v262, v263);
        TMUL(v262, v258, v163);
        // pto: %rope_rot_inline900_inline2358__tile
        ;
        Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v264 = Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v36, v35);
        // pto: %rope_rot_inline900_inline2358__tile
        ;
        uint64_t v265 = (uint64_t) v15;
        TASSIGN(v264, v265);
        pipe_barrier(PIPE_V);
        TADD(v264, v260, v262);
        // pto: %rope_bf16_inline942_inline2333__tile
        ;
        Tile<TileType::Vec, bfloat16_t, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v266 = Tile<TileType::Vec, bfloat16_t, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v36, v35);
        // pto: %rope_bf16_inline942_inline2333__tile
        ;
        uint64_t v267 = (uint64_t) v15;
        TASSIGN(v266, v267);
        pipe_barrier(PIPE_V);
        RoundMode v268 = RoundMode::CAST_RINT;
        SaturationMode v269 = SaturationMode::OFF;
        TCVT(v266, v264, v268, v269);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
        // pto: %qr_bf16_2d_inline897_inline2328__iter_v3_pview
        ;
        __gm__ bfloat16_t* v270 = PTOAS__GLOBAL_TENSOR_DATA(v54);
        // pto: %qr_bf16_2d_inline897_inline2328__iter_v3_pview
        ;
        const int64_t v271 = 0;
        // pto: %qr_bf16_2d_inline897_inline2328__iter_v3_pview
        ;
        const int64_t v272 = 8192;
        // pto: %qr_bf16_2d_inline897_inline2328__iter_v3_pview
        ;
        pto::Shape<1, 1, 1, 8, 64> v273 = pto::Shape<1, 1, 1, 8, 64>();
        // pto: %qr_bf16_2d_inline897_inline2328__iter_v3_pview
        ;
        pto::Stride<65536, 65536, 65536, 8192, 1> v274 = pto::Stride<65536, 65536, 65536, 8192, 1>();
        // pto: %qr_bf16_2d_inline897_inline2328__iter_v3_pview
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 8, 64>, pto::Stride<65536, 65536, 65536, 8192, 1>, pto::Layout::ND> v275 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 8, 64>, pto::Stride<65536, 65536, 65536, 8192, 1>, pto::Layout::ND>(v270 + (v271 + v143 * v272 + v176), v273, v274);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        pipe_barrier(PIPE_MTE3);
        TSTORE(v275, v232);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
        // pto: %101
        ;
        int64_t v276 = (int64_t) ((uint64_t) v172 + (uint64_t) v35);
        // pto: %102
        ;
        int64_t v277 = v276 < v44 ? v44 : v276;
        // pto: %qr_bf16_2d_inline897_inline2328__tile_pview
        ;
        __gm__ bfloat16_t* v278 = PTOAS__GLOBAL_TENSOR_DATA(v54);
        // pto: %qr_bf16_2d_inline897_inline2328__tile_pview
        ;
        const int64_t v279 = 0;
        // pto: %qr_bf16_2d_inline897_inline2328__tile_pview
        ;
        const int64_t v280 = 8192;
        // pto: %qr_bf16_2d_inline897_inline2328__tile_pview
        ;
        pto::Shape<1, 1, 1, 8, 64> v281 = pto::Shape<1, 1, 1, 8, 64>();
        // pto: %qr_bf16_2d_inline897_inline2328__tile_pview
        ;
        pto::Stride<65536, 65536, 65536, 8192, 1> v282 = pto::Stride<65536, 65536, 65536, 8192, 1>();
        // pto: %qr_bf16_2d_inline897_inline2328__tile_pview
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 8, 64>, pto::Stride<65536, 65536, 65536, 8192, 1>, pto::Layout::ND> v283 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 8, 64>, pto::Stride<65536, 65536, 65536, 8192, 1>, pto::Layout::ND>(v278 + (v279 + v143 * v280 + v277), v281, v282);
        pipe_barrier(PIPE_MTE3);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
        TSTORE(v283, v266);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID4);
        // pto: %16
        ;
        Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v284 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v46);
        // pto: %16
        ;
        uint64_t v285 = (uint64_t) v23;
        TASSIGN(v284, v285);
        // pto: %17
        ;
        Tile<TileType::Vec, float, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v286 = Tile<TileType::Vec, float, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v36, v46);
        // pto: %17
        ;
        uint64_t v287 = (uint64_t) v16;
        TASSIGN(v286, v287);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        RoundMode v288 = RoundMode::CAST_NONE;
        SaturationMode v289 = SaturationMode::OFF;
        TCVT(v286, v198, v288, v289);
        // pto: %18
        ;
        Tile<TileType::Vec, float, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v290 = Tile<TileType::Vec, float, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v36, v46);
        // pto: %18
        ;
        uint64_t v291 = (uint64_t) v27;
        TASSIGN(v290, v291);
        TEXPANDS(v290, v37);
        // pto: %19
        ;
        Tile<TileType::Vec, float, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v292 = Tile<TileType::Vec, float, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v36, v46);
        // pto: %19
        ;
        uint64_t v293 = (uint64_t) v27;
        TASSIGN(v292, v293);
        pipe_barrier(PIPE_V);
        TROWEXPANDMUL(v292, v290, v141);
        // pto: %20
        ;
        Tile<TileType::Vec, float, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v294 = Tile<TileType::Vec, float, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v36, v46);
        // pto: %20
        ;
        uint64_t v295 = (uint64_t) v27;
        TASSIGN(v294, v295);
        pipe_barrier(PIPE_V);
        TCOLEXPANDMUL(v294, v292, v284);
        // pto: %21
        ;
        Tile<TileType::Vec, float, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v296 = Tile<TileType::Vec, float, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v36, v46);
        // pto: %21
        ;
        uint64_t v297 = (uint64_t) v16;
        TASSIGN(v296, v297);
        pipe_barrier(PIPE_V);
        TMUL(v296, v286, v294);
        // pto: %22
        ;
        Tile<TileType::Vec, bfloat16_t, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v298 = Tile<TileType::Vec, bfloat16_t, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v36, v46);
        // pto: %22
        ;
        uint64_t v299 = (uint64_t) v27;
        TASSIGN(v298, v299);
        pipe_barrier(PIPE_V);
        RoundMode v300 = RoundMode::CAST_RINT;
        SaturationMode v301 = SaturationMode::OFF;
        TCVT(v298, v296, v300, v301);
        // pto: %23
        ;
        Tile<TileType::Vec, float, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v302 = Tile<TileType::Vec, float, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v36, v46);
        // pto: %23
        ;
        uint64_t v303 = (uint64_t) v16;
        TASSIGN(v302, v303);
        pipe_barrier(PIPE_V);
        RoundMode v304 = RoundMode::CAST_ROUND;
        SaturationMode v305 = SaturationMode::OFF;
        TCVT(v302, v298, v304, v305);
        // pto: %24
        ;
        Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v306 = Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v36, v35);
        // pto: %24
        ;
        uint64_t v307 = (uint64_t) v16;
        TASSIGN(v306, v307);
        // pto: %103
        ;
        Tile<TileType::Vec, float, 8, 128, BLayout::RowMajor, 8, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v308;
        // pto: %103
        ;
        uint64_t v309 = (uint64_t) v16;
        TASSIGN(v308, v309);
        // pto: %25
        ;
        Tile<TileType::Vec, bfloat16_t, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v310 = Tile<TileType::Vec, bfloat16_t, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v36, v35);
        // pto: %25
        ;
        uint64_t v311 = (uint64_t) v23;
        TASSIGN(v310, v311);
        pipe_barrier(PIPE_V);
        RoundMode v312 = RoundMode::CAST_RINT;
        SaturationMode v313 = SaturationMode::OFF;
        TCVT(v310, v308, v312, v313);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
        // pto: %26
        ;
        Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v314 = Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v36, v35);
        // pto: %26
        ;
        uint64_t v315 = (uint64_t) v28;
        TASSIGN(v314, v315);
        // pto: %104
        ;
        Tile<TileType::Vec, float, 8, 128, BLayout::RowMajor, 8, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v316;
        // pto: %104
        ;
        uint64_t v317 = (uint64_t) v28;
        TASSIGN(v316, v317);
        // pto: %27
        ;
        Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v318 = Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v36, v35);
        // pto: %27
        ;
        uint64_t v319 = (uint64_t) v27;
        TASSIGN(v318, v319);
        for (int64_t v320 = v44; v320 < v36; v320 += v34) {
          // pto: %28
          ;
          Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v321 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v35);
          // pto: %28
          ;
          uint64_t v322 = (uint64_t) v16;
          TASSIGN(v321, v322);
          // pto: %106
          ;
          Tile<TileType::Vec, float, 8, 128, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v323;
          // pto: %106
          ;
          uint64_t v324 = (uint64_t) ((int64_t) (uint64_t) ((int64_t) (uint64_t) v320 * (uint64_t) v14) + (uint64_t) v28);
          TASSIGN(v323, v324);
          // pto: %29
          ;
          Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v325 = Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v35);
          // pto: %29
          ;
          uint64_t v326 = (uint64_t) v18;
          TASSIGN(v325, v326);
          // pto: %107
          ;
          int64_t v327 = (int64_t) ((uint64_t) v320 * (uint64_t) v13);
          // pto: %107
          ;
          Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v328;
          // pto: %107
          ;
          uint64_t v329 = (uint64_t) v327;
          TASSIGN(v328, v329);
          // pto: %30
          ;
          Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v330 = Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v35);
          // pto: %30
          ;
          uint64_t v331 = (uint64_t) v29;
          TASSIGN(v330, v331);
          // pto: %31
          ;
          Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v332 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v35);
          // pto: %31
          ;
          uint64_t v333 = (uint64_t) v30;
          TASSIGN(v332, v333);
          pipe_barrier(PIPE_V);
          TGATHER(v332, v323, v328, v330);
          // pto: %108
          ;
          Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v334;
          // pto: %108
          ;
          uint64_t v335 = (uint64_t) ((int64_t) (uint64_t) v327 + (uint64_t) v27);
          TASSIGN(v334, v335);
          pipe_barrier(PIPE_V);
          TMOV(v334, v332);
        };
        // pto: %33
        ;
        Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v336 = Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v36, v35);
        // pto: %33
        ;
        uint64_t v337 = (uint64_t) v27;
        TASSIGN(v336, v337);
        // pto: %34
        ;
        Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v338 = Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v36, v35);
        // pto: %34
        ;
        uint64_t v339 = (uint64_t) v16;
        TASSIGN(v338, v339);
        pipe_barrier(PIPE_V);
        TMUL(v338, v316, v155);
        // pto: %35
        ;
        Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v340 = Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v36, v35);
        // pto: %35
        ;
        uint64_t v341 = (uint64_t) v27;
        TASSIGN(v340, v341);
        TMUL(v340, v336, v163);
        // pto: %36
        ;
        Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v342 = Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v36, v35);
        // pto: %36
        ;
        uint64_t v343 = (uint64_t) v16;
        TASSIGN(v342, v343);
        pipe_barrier(PIPE_V);
        TADD(v342, v338, v340);
        // pto: %37
        ;
        Tile<TileType::Vec, bfloat16_t, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v344 = Tile<TileType::Vec, bfloat16_t, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v36, v35);
        // pto: %37
        ;
        uint64_t v345 = (uint64_t) v16;
        TASSIGN(v344, v345);
        pipe_barrier(PIPE_V);
        RoundMode v346 = RoundMode::CAST_RINT;
        SaturationMode v347 = SaturationMode::OFF;
        TCVT(v344, v342, v346, v347);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID3);
        // pto: %112
        ;
        __gm__ bfloat16_t* v348 = PTOAS__GLOBAL_TENSOR_DATA(v54);
        // pto: %112
        ;
        const int64_t v349 = 0;
        // pto: %112
        ;
        const int64_t v350 = 8192;
        // pto: %112
        ;
        pto::Shape<1, 1, 1, 8, 64> v351 = pto::Shape<1, 1, 1, 8, 64>();
        // pto: %112
        ;
        pto::Stride<65536, 65536, 65536, 8192, 1> v352 = pto::Stride<65536, 65536, 65536, 8192, 1>();
        // pto: %112
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 8, 64>, pto::Stride<65536, 65536, 65536, 8192, 1>, pto::Layout::ND> v353 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 8, 64>, pto::Stride<65536, 65536, 65536, 8192, 1>, pto::Layout::ND>(v348 + (v349 + v143 * v350 + v192), v351, v352);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
        pipe_barrier(PIPE_MTE3);
        TSTORE(v353, v310);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID5);
        // pto: %115
        ;
        int64_t v354 = (int64_t) ((uint64_t) v173 + (uint64_t) v35);
        // pto: %116
        ;
        int64_t v355 = v354 < v44 ? v44 : v354;
        // pto: %117
        ;
        __gm__ bfloat16_t* v356 = PTOAS__GLOBAL_TENSOR_DATA(v54);
        // pto: %117
        ;
        const int64_t v357 = 0;
        // pto: %117
        ;
        const int64_t v358 = 8192;
        // pto: %117
        ;
        pto::Shape<1, 1, 1, 8, 64> v359 = pto::Shape<1, 1, 1, 8, 64>();
        // pto: %117
        ;
        pto::Stride<65536, 65536, 65536, 8192, 1> v360 = pto::Stride<65536, 65536, 65536, 8192, 1>();
        // pto: %117
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 8, 64>, pto::Stride<65536, 65536, 65536, 8192, 1>, pto::Layout::ND> v361 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 8, 64>, pto::Stride<65536, 65536, 65536, 8192, 1>, pto::Layout::ND>(v356 + (v357 + v143 * v358 + v355), v359, v360);
        pipe_barrier(PIPE_MTE3);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID3);
        TSTORE(v361, v344);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID6);
      };
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID4);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID5);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID6);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID6);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID7);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    } else {
      // pto: %tail_swap_idx_inline943_inline2357__tile
      ;
      Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v362 = Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v35);
      // pto: %tail_swap_idx_inline943_inline2357__tile
      ;
      uint64_t v363 = (uint64_t) v18;
      TASSIGN(v362, v363);
      // pto: %118
      ;
      uint64_t v364 = (uint64_t) v18;
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
      for (int64_t v365 = v140; v365 < v9; v365 += v34) {
        // pto: %tail_qr_scale_value_inline946_inline2352__tile
        ;
        float v366 = (v2)[v365];
        // pto: %tail_cos_tile_inline896_inline2353__tile
        ;
        Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v367 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v35);
        // pto: %tail_cos_tile_inline896_inline2353__tile
        ;
        uint64_t v368 = (uint64_t) v20;
        TASSIGN(v367, v368);
        // pto: %119
        ;
        int64_t v369 = v365 < v44 ? v44 : v365;
        // pto: %120
        ;
        __gm__ float* v370 = PTOAS__GLOBAL_TENSOR_DATA(v72);
        // pto: %120
        ;
        const int64_t v371 = 0;
        // pto: %120
        ;
        const int64_t v372 = 64;
        // pto: %120
        ;
        pto::Shape<1, 1, 1, 1, 64> v373 = pto::Shape<1, 1, 1, 1, 64>();
        // pto: %120
        ;
        pto::Stride<64, 64, 64, 64, 1> v374 = pto::Stride<64, 64, 64, 64, 1>();
        // pto: %120
        ;
        GlobalTensor<float, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<64, 64, 64, 64, 1>, pto::Layout::ND> v375 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<64, 64, 64, 64, 1>, pto::Layout::ND>(v370 + (v371 + v369 * v372), v373, v374);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
        TLOAD(v367, v375);
        // pto: %tail_sin_tile_inline930_inline2359__tile
        ;
        Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v376 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v35);
        // pto: %tail_sin_tile_inline930_inline2359__tile
        ;
        uint64_t v377 = (uint64_t) v21;
        TASSIGN(v376, v377);
        // pto: %122
        ;
        __gm__ float* v378 = PTOAS__GLOBAL_TENSOR_DATA(v81);
        // pto: %122
        ;
        const int64_t v379 = 0;
        // pto: %122
        ;
        const int64_t v380 = 64;
        // pto: %122
        ;
        pto::Shape<1, 1, 1, 1, 64> v381 = pto::Shape<1, 1, 1, 1, 64>();
        // pto: %122
        ;
        pto::Stride<64, 64, 64, 64, 1> v382 = pto::Stride<64, 64, 64, 64, 1>();
        // pto: %122
        ;
        GlobalTensor<float, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<64, 64, 64, 64, 1>, pto::Layout::ND> v383 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<64, 64, 64, 64, 1>, pto::Layout::ND>(v378 + (v379 + v369 * v380), v381, v382);
        pipe_barrier(PIPE_ALL);
        TLOAD(v376, v383);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID7);
        for (int64_t v384 = v44; v384 < v43; v384 += v45) {
          // pto: %123, %124
          ;
          int64_t v385 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) v139 + (uint64_t) v384) * (uint64_t) v46);
          // pto: %126, %125, %127
          ;
          int64_t v386 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) v139 + (uint64_t) ((int64_t) (uint64_t) v384 + (uint64_t) v34)) * (uint64_t) v46);
          // pto: %38
          ;
          Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v387 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v46);
          // pto: %38
          ;
          uint64_t v388 = (uint64_t) v15;
          TASSIGN(v387, v388);
          // pto: %128
          ;
          int64_t v389 = v385 < v44 ? v44 : v385;
          // pto: %129
          ;
          __gm__ float* v390 = PTOAS__GLOBAL_TENSOR_DATA(v91);
          // pto: %129
          ;
          const int64_t v391 = 0;
          // pto: %129
          ;
          pto::Shape<1, 1, 1, 1, 128> v392 = pto::Shape<1, 1, 1, 1, 128>();
          // pto: %129
          ;
          pto::Stride<128, 128, 128, 128, 1> v393 = pto::Stride<128, 128, 128, 128, 1>();
          // pto: %129
          ;
          GlobalTensor<float, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<128, 128, 128, 128, 1>, pto::Layout::ND> v394 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<128, 128, 128, 128, 1>, pto::Layout::ND>(v390 + (v391 + v389), v392, v393);
          wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID7);
          TLOAD(v387, v394);
          // pto: %39
          ;
          Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v395 = Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v46);
          // pto: %39
          ;
          uint64_t v396 = (uint64_t) v16;
          TASSIGN(v395, v396);
          // pto: %132
          ;
          __gm__ int32_t* v397 = PTOAS__GLOBAL_TENSOR_DATA(v99);
          // pto: %132
          ;
          const int64_t v398 = 0;
          // pto: %132
          ;
          const int64_t v399 = 8192;
          // pto: %132
          ;
          pto::Shape<1, 1, 1, 1, 128> v400 = pto::Shape<1, 1, 1, 1, 128>();
          // pto: %132
          ;
          pto::Stride<8192, 8192, 8192, 8192, 1> v401 = pto::Stride<8192, 8192, 8192, 8192, 1>();
          // pto: %132
          ;
          GlobalTensor<int32_t, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<8192, 8192, 8192, 8192, 1>, pto::Layout::ND> v402 = GlobalTensor<int32_t, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<8192, 8192, 8192, 8192, 1>, pto::Layout::ND>(v397 + (v398 + v369 * v399 + v389), v400, v401);
          TLOAD(v395, v402);
          set_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
          // pto: %40
          ;
          Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v403 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v46);
          // pto: %40
          ;
          uint64_t v404 = (uint64_t) v17;
          TASSIGN(v403, v404);
          // pto: %133
          ;
          int64_t v405 = v386 < v44 ? v44 : v386;
          // pto: %134
          ;
          __gm__ float* v406 = PTOAS__GLOBAL_TENSOR_DATA(v91);
          // pto: %134
          ;
          const int64_t v407 = 0;
          // pto: %134
          ;
          pto::Shape<1, 1, 1, 1, 128> v408 = pto::Shape<1, 1, 1, 1, 128>();
          // pto: %134
          ;
          pto::Stride<128, 128, 128, 128, 1> v409 = pto::Stride<128, 128, 128, 128, 1>();
          // pto: %134
          ;
          GlobalTensor<float, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<128, 128, 128, 128, 1>, pto::Layout::ND> v410 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<128, 128, 128, 128, 1>, pto::Layout::ND>(v406 + (v407 + v405), v408, v409);
          wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
          TLOAD(v403, v410);
          // pto: %41
          ;
          Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v411 = Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v46);
          // pto: %41
          ;
          uint64_t v412 = (uint64_t) v27;
          TASSIGN(v411, v412);
          // pto: %137
          ;
          __gm__ int32_t* v413 = PTOAS__GLOBAL_TENSOR_DATA(v99);
          // pto: %137
          ;
          const int64_t v414 = 0;
          // pto: %137
          ;
          const int64_t v415 = 8192;
          // pto: %137
          ;
          pto::Shape<1, 1, 1, 1, 128> v416 = pto::Shape<1, 1, 1, 1, 128>();
          // pto: %137
          ;
          pto::Stride<8192, 8192, 8192, 8192, 1> v417 = pto::Stride<8192, 8192, 8192, 8192, 1>();
          // pto: %137
          ;
          GlobalTensor<int32_t, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<8192, 8192, 8192, 8192, 1>, pto::Layout::ND> v418 = GlobalTensor<int32_t, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<8192, 8192, 8192, 8192, 1>, pto::Layout::ND>(v413 + (v414 + v369 * v415 + v405), v416, v417);
          TLOAD(v411, v418);
          set_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
          // pto: %tail_wq_scale_inline947_inline2330__tile
          ;
          Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v419 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v46);
          // pto: %tail_wq_scale_inline947_inline2330__tile
          ;
          uint64_t v420 = (uint64_t) v15;
          TASSIGN(v419, v420);
          // pto: %tail_acc_fp32_inline949_inline2315__tile
          ;
          Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v421 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v46);
          // pto: %tail_acc_fp32_inline949_inline2315__tile
          ;
          uint64_t v422 = (uint64_t) v16;
          TASSIGN(v421, v422);
          wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
          RoundMode v423 = RoundMode::CAST_NONE;
          SaturationMode v424 = SaturationMode::OFF;
          TCVT(v421, v395, v423, v424);
          // pto: %tail_qr_dequant_scale_inline950_inline2360__tile
          ;
          Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v425 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v46);
          // pto: %tail_qr_dequant_scale_inline950_inline2360__tile
          ;
          uint64_t v426 = (uint64_t) v15;
          TASSIGN(v425, v426);
          TMULS(v425, v419, v366);
          // pto: %tail_qr_dequant_inline909_inline2325__tile
          ;
          Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v427 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v46);
          // pto: %tail_qr_dequant_inline909_inline2325__tile
          ;
          uint64_t v428 = (uint64_t) v15;
          TASSIGN(v427, v428);
          pipe_barrier(PIPE_V);
          TMUL(v427, v421, v425);
          // pto: %42
          ;
          Tile<TileType::Vec, bfloat16_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v429 = Tile<TileType::Vec, bfloat16_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v46);
          // pto: %42
          ;
          uint64_t v430 = (uint64_t) v16;
          TASSIGN(v429, v430);
          pipe_barrier(PIPE_V);
          RoundMode v431 = RoundMode::CAST_RINT;
          SaturationMode v432 = SaturationMode::OFF;
          TCVT(v429, v427, v431, v432);
          // pto: %tail_qr_dequant_v1_inline918_inline2308__tile
          ;
          Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v433 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v46);
          // pto: %tail_qr_dequant_v1_inline918_inline2308__tile
          ;
          uint64_t v434 = (uint64_t) v15;
          TASSIGN(v433, v434);
          pipe_barrier(PIPE_V);
          RoundMode v435 = RoundMode::CAST_ROUND;
          SaturationMode v436 = SaturationMode::OFF;
          TCVT(v433, v429, v435, v436);
          // pto: %43
          ;
          Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v437 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v35);
          // pto: %43
          ;
          uint64_t v438 = (uint64_t) v15;
          TASSIGN(v437, v438);
          // pto: %138
          ;
          Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v439;
          // pto: %138
          ;
          uint64_t v440 = (uint64_t) v15;
          TASSIGN(v439, v440);
          // pto: %tail_qr_nope_bf16_inline937_inline2363__tile
          ;
          Tile<TileType::Vec, bfloat16_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v441 = Tile<TileType::Vec, bfloat16_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v35);
          // pto: %tail_qr_nope_bf16_inline937_inline2363__tile
          ;
          uint64_t v442 = (uint64_t) v29;
          TASSIGN(v441, v442);
          pipe_barrier(PIPE_V);
          RoundMode v443 = RoundMode::CAST_RINT;
          SaturationMode v444 = SaturationMode::OFF;
          TCVT(v441, v439, v443, v444);
          set_flag(PIPE_V, PIPE_MTE3, EVENT_ID4);
          // pto: %tail_qr_rope_slice_inline951_inline2285__tile
          ;
          Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v445 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v35);
          // pto: %tail_qr_rope_slice_inline951_inline2285__tile
          ;
          uint64_t v446 = (uint64_t) v24;
          TASSIGN(v445, v446);
          // pto: %139
          ;
          Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v447;
          // pto: %139
          ;
          uint64_t v448 = (uint64_t) v24;
          TASSIGN(v447, v448);
          // pto: %44
          ;
          Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v449 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v35);
          // pto: %44
          ;
          uint64_t v450 = (uint64_t) v16;
          TASSIGN(v449, v450);
          // pto: %45
          ;
          Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v451 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v35);
          // pto: %45
          ;
          uint64_t v452 = (uint64_t) v15;
          TASSIGN(v451, v452);
          // pto: %140
          ;
          Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v453;
          // pto: %140
          ;
          uint64_t v454 = (uint64_t) v24;
          TASSIGN(v453, v454);
          // pto: %46
          ;
          Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v455 = Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v35);
          // pto: %46
          ;
          uint64_t v456 = (uint64_t) v18;
          TASSIGN(v455, v456);
          // pto: %141
          ;
          Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v457;
          // pto: %141
          ;
          uint64_t v458 = (uint64_t) v18;
          TASSIGN(v457, v458);
          // pto: %47
          ;
          Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v459 = Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v35);
          // pto: %47
          ;
          uint64_t v460 = (uint64_t) v22;
          TASSIGN(v459, v460);
          // pto: %48
          ;
          Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v461 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v35);
          // pto: %48
          ;
          uint64_t v462 = (uint64_t) v23;
          TASSIGN(v461, v462);
          TGATHER(v461, v453, v457, v459);
          // pto: %142
          ;
          Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v463;
          // pto: %142
          ;
          uint64_t v464 = (uint64_t) v16;
          TASSIGN(v463, v464);
          pipe_barrier(PIPE_V);
          TMOV(v463, v461);
          // pto: %tail_qr_swapped_inline892_inline2300__tile
          ;
          Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v465 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v35);
          // pto: %tail_qr_swapped_inline892_inline2300__tile
          ;
          uint64_t v466 = (uint64_t) v16;
          TASSIGN(v465, v466);
          // pto: %50
          ;
          Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v467 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v35);
          // pto: %50
          ;
          uint64_t v468 = (uint64_t) v15;
          TASSIGN(v467, v468);
          TMUL(v467, v447, v367);
          // pto: %51
          ;
          Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v469 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v35);
          // pto: %51
          ;
          uint64_t v470 = (uint64_t) v16;
          TASSIGN(v469, v470);
          pipe_barrier(PIPE_V);
          TMUL(v469, v465, v376);
          // pto: %tail_rope_rot_inline891_inline2292__tile
          ;
          Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v471 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v35);
          // pto: %tail_rope_rot_inline891_inline2292__tile
          ;
          uint64_t v472 = (uint64_t) v15;
          TASSIGN(v471, v472);
          pipe_barrier(PIPE_V);
          TADD(v471, v467, v469);
          // pto: %tail_rope_bf16_inline890_inline2284__tile
          ;
          Tile<TileType::Vec, bfloat16_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v473 = Tile<TileType::Vec, bfloat16_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v35);
          // pto: %tail_rope_bf16_inline890_inline2284__tile
          ;
          uint64_t v474 = (uint64_t) v15;
          TASSIGN(v473, v474);
          pipe_barrier(PIPE_V);
          RoundMode v475 = RoundMode::CAST_RINT;
          SaturationMode v476 = SaturationMode::OFF;
          TCVT(v473, v471, v475, v476);
          set_flag(PIPE_V, PIPE_MTE3, EVENT_ID5);
          // pto: %qr_bf16_2d_inline897_inline2328__iter_v9_pview
          ;
          __gm__ bfloat16_t* v477 = PTOAS__GLOBAL_TENSOR_DATA(v54);
          // pto: %qr_bf16_2d_inline897_inline2328__iter_v9_pview
          ;
          const int64_t v478 = 0;
          // pto: %qr_bf16_2d_inline897_inline2328__iter_v9_pview
          ;
          const int64_t v479 = 8192;
          // pto: %qr_bf16_2d_inline897_inline2328__iter_v9_pview
          ;
          pto::Shape<1, 1, 1, 1, 64> v480 = pto::Shape<1, 1, 1, 1, 64>();
          // pto: %qr_bf16_2d_inline897_inline2328__iter_v9_pview
          ;
          pto::Stride<8192, 8192, 8192, 8192, 1> v481 = pto::Stride<8192, 8192, 8192, 8192, 1>();
          // pto: %qr_bf16_2d_inline897_inline2328__iter_v9_pview
          ;
          GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<8192, 8192, 8192, 8192, 1>, pto::Layout::ND> v482 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<8192, 8192, 8192, 8192, 1>, pto::Layout::ND>(v477 + (v478 + v369 * v479 + v389), v480, v481);
          wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID4);
          pipe_barrier(PIPE_MTE3);
          TSTORE(v482, v441);
          // pto: %148
          ;
          int64_t v483 = (int64_t) ((uint64_t) v385 + (uint64_t) v35);
          // pto: %149
          ;
          int64_t v484 = v483 < v44 ? v44 : v483;
          // pto: %150
          ;
          __gm__ bfloat16_t* v485 = PTOAS__GLOBAL_TENSOR_DATA(v54);
          // pto: %150
          ;
          const int64_t v486 = 0;
          // pto: %150
          ;
          const int64_t v487 = 8192;
          // pto: %150
          ;
          pto::Shape<1, 1, 1, 1, 64> v488 = pto::Shape<1, 1, 1, 1, 64>();
          // pto: %150
          ;
          pto::Stride<8192, 8192, 8192, 8192, 1> v489 = pto::Stride<8192, 8192, 8192, 8192, 1>();
          // pto: %150
          ;
          GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<8192, 8192, 8192, 8192, 1>, pto::Layout::ND> v490 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<8192, 8192, 8192, 8192, 1>, pto::Layout::ND>(v485 + (v486 + v369 * v487 + v484), v488, v489);
          pipe_barrier(PIPE_MTE3);
          wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID5);
          TSTORE(v490, v473);
          set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID7);
          // pto: %52
          ;
          Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v491 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v46);
          // pto: %52
          ;
          uint64_t v492 = (uint64_t) v17;
          TASSIGN(v491, v492);
          // pto: %53
          ;
          Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v493 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v46);
          // pto: %53
          ;
          uint64_t v494 = (uint64_t) v27;
          TASSIGN(v493, v494);
          wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
          RoundMode v495 = RoundMode::CAST_NONE;
          SaturationMode v496 = SaturationMode::OFF;
          TCVT(v493, v411, v495, v496);
          // pto: %54
          ;
          Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v497 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v46);
          // pto: %54
          ;
          uint64_t v498 = (uint64_t) v17;
          TASSIGN(v497, v498);
          TMULS(v497, v491, v366);
          // pto: %55
          ;
          Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v499 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v46);
          // pto: %55
          ;
          uint64_t v500 = (uint64_t) v17;
          TASSIGN(v499, v500);
          pipe_barrier(PIPE_V);
          TMUL(v499, v493, v497);
          // pto: %56
          ;
          Tile<TileType::Vec, bfloat16_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v501 = Tile<TileType::Vec, bfloat16_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v46);
          // pto: %56
          ;
          uint64_t v502 = (uint64_t) v27;
          TASSIGN(v501, v502);
          pipe_barrier(PIPE_V);
          RoundMode v503 = RoundMode::CAST_RINT;
          SaturationMode v504 = SaturationMode::OFF;
          TCVT(v501, v499, v503, v504);
          // pto: %57
          ;
          Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v505 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v46);
          // pto: %57
          ;
          uint64_t v506 = (uint64_t) v17;
          TASSIGN(v505, v506);
          pipe_barrier(PIPE_V);
          RoundMode v507 = RoundMode::CAST_ROUND;
          SaturationMode v508 = SaturationMode::OFF;
          TCVT(v505, v501, v507, v508);
          // pto: %58
          ;
          Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v509 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v35);
          // pto: %58
          ;
          uint64_t v510 = (uint64_t) v17;
          TASSIGN(v509, v510);
          // pto: %151
          ;
          Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v511;
          // pto: %151
          ;
          uint64_t v512 = (uint64_t) v17;
          TASSIGN(v511, v512);
          // pto: %59
          ;
          Tile<TileType::Vec, bfloat16_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v513 = Tile<TileType::Vec, bfloat16_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v35);
          // pto: %59
          ;
          uint64_t v514 = (uint64_t) v30;
          TASSIGN(v513, v514);
          pipe_barrier(PIPE_V);
          RoundMode v515 = RoundMode::CAST_RINT;
          SaturationMode v516 = SaturationMode::OFF;
          TCVT(v513, v511, v515, v516);
          set_flag(PIPE_V, PIPE_MTE3, EVENT_ID6);
          // pto: %60
          ;
          Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v517 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v35);
          // pto: %60
          ;
          uint64_t v518 = (uint64_t) v31;
          TASSIGN(v517, v518);
          // pto: %152
          ;
          Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v519;
          // pto: %152
          ;
          uint64_t v520 = (uint64_t) v31;
          TASSIGN(v519, v520);
          // pto: %61
          ;
          Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v521 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v35);
          // pto: %61
          ;
          uint64_t v522 = (uint64_t) v27;
          TASSIGN(v521, v522);
          // pto: %62
          ;
          Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v523 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v35);
          // pto: %62
          ;
          uint64_t v524 = (uint64_t) v17;
          TASSIGN(v523, v524);
          // pto: %153
          ;
          Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v525;
          // pto: %153
          ;
          uint64_t v526 = (uint64_t) v31;
          TASSIGN(v525, v526);
          // pto: %63
          ;
          Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v527 = Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v35);
          // pto: %63
          ;
          uint64_t v528 = (uint64_t) v18;
          TASSIGN(v527, v528);
          // pto: %154
          ;
          Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v529;
          // pto: %154
          ;
          uint64_t v530 = (uint64_t) v18;
          TASSIGN(v529, v530);
          // pto: %64
          ;
          Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v531 = Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v35);
          // pto: %64
          ;
          uint64_t v532 = (uint64_t) v25;
          TASSIGN(v531, v532);
          // pto: %65
          ;
          Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v533 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v35);
          // pto: %65
          ;
          uint64_t v534 = (uint64_t) v26;
          TASSIGN(v533, v534);
          TGATHER(v533, v525, v529, v531);
          // pto: %155
          ;
          Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v535;
          // pto: %155
          ;
          uint64_t v536 = (uint64_t) v27;
          TASSIGN(v535, v536);
          pipe_barrier(PIPE_V);
          TMOV(v535, v533);
          // pto: %67
          ;
          Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v537 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v35);
          // pto: %67
          ;
          uint64_t v538 = (uint64_t) v27;
          TASSIGN(v537, v538);
          // pto: %68
          ;
          Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v539 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v35);
          // pto: %68
          ;
          uint64_t v540 = (uint64_t) v17;
          TASSIGN(v539, v540);
          TMUL(v539, v519, v367);
          // pto: %69
          ;
          Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v541 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v35);
          // pto: %69
          ;
          uint64_t v542 = (uint64_t) v27;
          TASSIGN(v541, v542);
          pipe_barrier(PIPE_V);
          TMUL(v541, v537, v376);
          // pto: %70
          ;
          Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v543 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v35);
          // pto: %70
          ;
          uint64_t v544 = (uint64_t) v17;
          TASSIGN(v543, v544);
          pipe_barrier(PIPE_V);
          TADD(v543, v539, v541);
          // pto: %71
          ;
          Tile<TileType::Vec, bfloat16_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v545 = Tile<TileType::Vec, bfloat16_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v35);
          // pto: %71
          ;
          uint64_t v546 = (uint64_t) v17;
          TASSIGN(v545, v546);
          pipe_barrier(PIPE_V);
          RoundMode v547 = RoundMode::CAST_RINT;
          SaturationMode v548 = SaturationMode::OFF;
          TCVT(v545, v543, v547, v548);
          set_flag(PIPE_V, PIPE_MTE3, EVENT_ID7);
          // pto: %159
          ;
          __gm__ bfloat16_t* v549 = PTOAS__GLOBAL_TENSOR_DATA(v54);
          // pto: %159
          ;
          const int64_t v550 = 0;
          // pto: %159
          ;
          const int64_t v551 = 8192;
          // pto: %159
          ;
          pto::Shape<1, 1, 1, 1, 64> v552 = pto::Shape<1, 1, 1, 1, 64>();
          // pto: %159
          ;
          pto::Stride<8192, 8192, 8192, 8192, 1> v553 = pto::Stride<8192, 8192, 8192, 8192, 1>();
          // pto: %159
          ;
          GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<8192, 8192, 8192, 8192, 1>, pto::Layout::ND> v554 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<8192, 8192, 8192, 8192, 1>, pto::Layout::ND>(v549 + (v550 + v369 * v551 + v405), v552, v553);
          wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID6);
          pipe_barrier(PIPE_MTE3);
          TSTORE(v554, v513);
          // pto: %162
          ;
          int64_t v555 = (int64_t) ((uint64_t) v386 + (uint64_t) v35);
          // pto: %163
          ;
          int64_t v556 = v555 < v44 ? v44 : v555;
          // pto: %164
          ;
          __gm__ bfloat16_t* v557 = PTOAS__GLOBAL_TENSOR_DATA(v54);
          // pto: %164
          ;
          const int64_t v558 = 0;
          // pto: %164
          ;
          const int64_t v559 = 8192;
          // pto: %164
          ;
          pto::Shape<1, 1, 1, 1, 64> v560 = pto::Shape<1, 1, 1, 1, 64>();
          // pto: %164
          ;
          pto::Stride<8192, 8192, 8192, 8192, 1> v561 = pto::Stride<8192, 8192, 8192, 8192, 1>();
          // pto: %164
          ;
          GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<8192, 8192, 8192, 8192, 1>, pto::Layout::ND> v562 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<8192, 8192, 8192, 8192, 1>, pto::Layout::ND>(v557 + (v558 + v369 * v559 + v556), v560, v561);
          pipe_barrier(PIPE_MTE3);
          wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID7);
          TSTORE(v562, v545);
          set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
        };
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID7);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
      };
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    };
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID4);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID5);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  }
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID4);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID5);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID6);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID7);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  #endif // __DAV_VEC__

  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}