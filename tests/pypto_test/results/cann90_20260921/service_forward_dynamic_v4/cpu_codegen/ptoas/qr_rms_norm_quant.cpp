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

AICORE void qr_rms_norm_quant(__gm__ float* v1, __gm__ bfloat16_t* v2, __gm__ float* v3, __gm__ float* v4, __gm__ int8_t* v5, __gm__ int8_t* v6, int64_t v7, int64_t v8, int64_t v9, int64_t v10, int32_t v11, int32_t v12) {
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  // pto: %c17504_i64
  const int64_t v13 = 17504;
  // pto: %c50272_i64
  const int64_t v14 = 50272;
  const int64_t v15 = 19552;
  const int64_t v16 = 18528;
  const int64_t v17 = 18016;
  const int64_t v18 = 17760;
  // pto: %c1088_i64
  const int64_t v19 = 1088;
  // pto: %c9280_i64
  const int64_t v20 = 9280;
  // pto: %c0_i64
  const int64_t v21 = 0;
  // pto: %c32_i64
  const int64_t v22 = 32;
  // pto: %c64_i64
  const int64_t v23 = 64;
  // pto: %c576_i64
  const int64_t v24 = 576;
  // pto: %c17472_i64
  const int64_t v25 = 17472;
  // pto: %c1024_index
  const int64_t v26 = 1024;
  // pto: %c1_index
  const int64_t v27 = 1;
  // pto: %c512_index
  const int64_t v28 = 512;
  // pto: %c8_index
  const int64_t v29 = 8;
  // pto: %c0_index
  const int64_t v30 = 0;
  // pto: %c256_index
  const int64_t v31 = 256;
  // pto: %c128_index
  const int64_t v32 = 128;
  // pto: %c64_index
  const int64_t v33 = 64;
  // pto: %cst_21
  const float v34 = 9.765625E-4f;
  // pto: %cst_22
  const float v35 = 9.99999997E-7f;
  // pto: %c0_i32
  const int32_t v36 = 0;
  // pto: %c8388608_index
  const int64_t v37 = 8388608;
  // pto: %c2139095040_i64
  const int64_t v38 = 2139095040;
  // pto: %c8388608_i64
  const int64_t v39 = 8388608;
  // pto: %c2_index
  const int64_t v40 = 2;
  // pto: %c2_i64
  const int64_t v41 = 2;
  // pto: %c152_i64
  const int64_t v42 = 152;
  // pto: %c1_i64
  const int64_t v43 = 1;
  // pto: %cst_31
  const float v44 = 0.0f;
  // pto: %cst_32
  const float v45 = 1.0f;
  // pto: %cst_33
  const float v46 = 9.99999974E-5f;
  // pto: %cst_34
  const float v47 = 127.0f;
  // pto: %qr_fp32_inline0_inline1836__rv_v10_view
  const int64_t v48 = 1;
  // pto: %qr_fp32_inline0_inline1836__rv_v10_view
  const int64_t v49 = 1;
  // pto: %qr_fp32_inline0_inline1836__rv_v10_view
  const int64_t v50 = 1;
  // pto: %qr_fp32_inline0_inline1836__rv_v10_view
  int64_t v51 = (int64_t) v9;
  // pto: %qr_fp32_inline0_inline1836__rv_v10_view
  int64_t v52 = v51 * v26;
  // pto: %qr_fp32_inline0_inline1836__rv_v10_view
  int64_t v53 = v50 * v52;
  // pto: %qr_fp32_inline0_inline1836__rv_v10_view
  pto::Shape<1, 1, 1, -1, -1> v54 = pto::Shape<1, 1, 1, -1, -1>(v48, v49, v50, v51, v26);
  // pto: %qr_fp32_inline0_inline1836__rv_v10_view
  pto::Stride<-1, -1, -1, -1, -1> v55 = pto::Stride<-1, -1, -1, -1, -1>(v49 * v53, v53, v52, v26, v27);
  // pto: %qr_fp32_inline0_inline1836__rv_v10_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v56 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v1, v54, v55);
  // pto: %gamma_cq__ssa_v0_view
  const int64_t v57 = 1;
  // pto: %gamma_cq__ssa_v0_view
  const int64_t v58 = 1;
  // pto: %gamma_cq__ssa_v0_view
  const int64_t v59 = 1;
  // pto: %gamma_cq__ssa_v0_view
  const int64_t v60 = 1;
  // pto: %gamma_cq__ssa_v0_view
  int64_t v61 = v26 * v27;
  // pto: %gamma_cq__ssa_v0_view
  int64_t v62 = v60 * v61;
  // pto: %gamma_cq__ssa_v0_view
  int64_t v63 = v59 * v62;
  // pto: %gamma_cq__ssa_v0_view
  pto::Shape<1, 1, 1, 1, -1> v64 = pto::Shape<1, 1, 1, 1, -1>(v57, v58, v59, v60, v26);
  // pto: %gamma_cq__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v65 = pto::Stride<-1, -1, -1, -1, -1>(v58 * v63, v63, v62, v61, v27);
  // pto: %gamma_cq__ssa_v0_view
  GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v66 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v2, v64, v65);
  // pto: %qr_scale_pad_store_inline1827__iter_v1_view
  const int64_t v67 = 1;
  // pto: %qr_scale_pad_store_inline1827__iter_v1_view
  const int64_t v68 = 1;
  // pto: %qr_scale_pad_store_inline1827__iter_v1_view
  const int64_t v69 = 1;
  // pto: %qr_scale_pad_store_inline1827__iter_v1_view
  int64_t v70 = v28 * v27;
  // pto: %qr_scale_pad_store_inline1827__iter_v1_view
  int64_t v71 = v69 * v70;
  // pto: %qr_scale_pad_store_inline1827__iter_v1_view
  pto::Shape<1, 1, 1, -1, -1> v72 = pto::Shape<1, 1, 1, -1, -1>(v67, v68, v69, v28, v27);
  // pto: %qr_scale_pad_store_inline1827__iter_v1_view
  pto::Stride<-1, -1, -1, -1, -1> v73 = pto::Stride<-1, -1, -1, -1, -1>(v68 * v71, v71, v70, v27, v28);
  // pto: %qr_scale_pad_store_inline1827__iter_v1_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN> v74 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN>(v3, v72, v73);
  // pto: %qr_scale_view_inline3100__ssa_v0_view
  const int64_t v75 = 1;
  // pto: %qr_scale_view_inline3100__ssa_v0_view
  const int64_t v76 = 1;
  // pto: %qr_scale_view_inline3100__ssa_v0_view
  const int64_t v77 = 1;
  // pto: %qr_scale_view_inline3100__ssa_v0_view
  int64_t v78 = (int64_t) v10;
  // pto: %qr_scale_view_inline3100__ssa_v0_view
  int64_t v79 = v78 * v27;
  // pto: %qr_scale_view_inline3100__ssa_v0_view
  int64_t v80 = v77 * v79;
  // pto: %qr_scale_view_inline3100__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v81 = pto::Shape<1, 1, 1, -1, -1>(v75, v76, v77, v78, v27);
  // pto: %qr_scale_view_inline3100__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v82 = pto::Stride<-1, -1, -1, -1, -1>(v76 * v80, v80, v79, v27, (int64_t) v10);
  // pto: %qr_scale_view_inline3100__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN> v83 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN>(v4, v81, v82);
  // pto: %qr_i8_matmul_inline1825__iter_v1_view
  const int64_t v84 = 1;
  // pto: %qr_i8_matmul_inline1825__iter_v1_view
  const int64_t v85 = 1;
  // pto: %qr_i8_matmul_inline1825__iter_v1_view
  const int64_t v86 = 1;
  // pto: %qr_i8_matmul_inline1825__iter_v1_view
  int64_t v87 = v28 * v26;
  // pto: %qr_i8_matmul_inline1825__iter_v1_view
  int64_t v88 = v86 * v87;
  // pto: %qr_i8_matmul_inline1825__iter_v1_view
  pto::Shape<1, 1, 1, -1, -1> v89 = pto::Shape<1, 1, 1, -1, -1>(v84, v85, v86, v28, v26);
  // pto: %qr_i8_matmul_inline1825__iter_v1_view
  pto::Stride<-1, -1, -1, -1, -1> v90 = pto::Stride<-1, -1, -1, -1, -1>(v85 * v88, v88, v87, v26, v27);
  // pto: %qr_i8_matmul_inline1825__iter_v1_view
  GlobalTensor<int8_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v91 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v5, v89, v90);
  // pto: %qr_view_inline3085__ssa_v0_view
  const int64_t v92 = 1;
  // pto: %qr_view_inline3085__ssa_v0_view
  const int64_t v93 = 1;
  // pto: %qr_view_inline3085__ssa_v0_view
  const int64_t v94 = 1;
  // pto: %qr_view_inline3085__ssa_v0_view
  int64_t v95 = (int64_t) v10;
  // pto: %qr_view_inline3085__ssa_v0_view
  int64_t v96 = v95 * v26;
  // pto: %qr_view_inline3085__ssa_v0_view
  int64_t v97 = v94 * v96;
  // pto: %qr_view_inline3085__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v98 = pto::Shape<1, 1, 1, -1, -1>(v92, v93, v94, v95, v26);
  // pto: %qr_view_inline3085__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v99 = pto::Stride<-1, -1, -1, -1, -1>(v93 * v97, v97, v96, v26, v27);
  // pto: %qr_view_inline3085__ssa_v0_view
  GlobalTensor<int8_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v100 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v6, v98, v99);
  // pto: %tg_idx_inline3113__ssa_v0, %49
  int64_t v101 = (int64_t) ((uint64_t) ((int64_t) v11) * (uint64_t) v29);
  // pto: %50
  int64_t v102 = (int64_t) ((uint64_t) v7 - (uint64_t) v101);
  // pto: %51
  int64_t v103 = v102 < v29 ? v102 : v29;
  // pto: %52
  int64_t v104 = (int64_t) ((uint64_t) v8 + (uint64_t) v101);
  // pto: %t__tile
  Tile<TileType::Vec, float, 8, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v105 = Tile<TileType::Vec, float, 8, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v29, v26);
  // pto: %t__tile
  uint64_t v106 = (uint64_t) v13;
  TASSIGN(v105, v106);
  // pto: %53
  int64_t v107 = v101 < v30 ? v30 : v101;
  // pto: %qr_fp32_inline0_inline1836__rv_v10_pview
  __gm__ float* v108 = PTOAS__GLOBAL_TENSOR_DATA(v56);
  // pto: %qr_fp32_inline0_inline1836__rv_v10_pview
  const int64_t v109 = 0;
  // pto: %qr_fp32_inline0_inline1836__rv_v10_pview
  const int64_t v110 = 1024;
  // pto: %qr_fp32_inline0_inline1836__rv_v10_pview
  pto::Shape<1, 1, 1, 8, 1024> v111 = pto::Shape<1, 1, 1, 8, 1024>();
  // pto: %qr_fp32_inline0_inline1836__rv_v10_pview
  pto::Stride<8192, 8192, 8192, 1024, 1> v112 = pto::Stride<8192, 8192, 8192, 1024, 1>();
  // pto: %qr_fp32_inline0_inline1836__rv_v10_pview
  GlobalTensor<float, pto::Shape<1, 1, 1, 8, 1024>, pto::Stride<8192, 8192, 8192, 1024, 1>, pto::Layout::ND> v113 = GlobalTensor<float, pto::Shape<1, 1, 1, 8, 1024>, pto::Stride<8192, 8192, 8192, 1024, 1>, pto::Layout::ND>(v108 + (v109 + v107 * v110), v111, v112);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
  TLOAD(v105, v113);
  set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  // pto: %0
  Tile<TileType::Vec, bfloat16_t, 8, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v114 = Tile<TileType::Vec, bfloat16_t, 8, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v29, v26);
  // pto: %0
  uint64_t v115 = (uint64_t) v14;
  TASSIGN(v114, v115);
  wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  RoundMode v116 = RoundMode::CAST_RINT;
  SaturationMode v117 = SaturationMode::OFF;
  TCVT(v114, v105, v116, v117);
  // pto: %qr_rms_full_inline1306_inline3116__tile
  Tile<TileType::Vec, float, 8, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v118 = Tile<TileType::Vec, float, 8, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v29, v26);
  // pto: %qr_rms_full_inline1306_inline3116__tile
  uint64_t v119 = (uint64_t) v13;
  TASSIGN(v118, v119);
  pipe_barrier(PIPE_V);
  RoundMode v120 = RoundMode::CAST_ROUND;
  SaturationMode v121 = SaturationMode::OFF;
  TCVT(v118, v114, v120, v121);
  // pto: %qr_square_full_inline1303_inline3087__tile
  Tile<TileType::Vec, float, 8, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v122 = Tile<TileType::Vec, float, 8, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v29, v26);
  // pto: %qr_square_full_inline1303_inline3087__tile
  uint64_t v123 = (uint64_t) v13;
  TASSIGN(v122, v123);
  pipe_barrier(PIPE_V);
  TMUL(v122, v118, v118);
  // pto: %1
  Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v124 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v29, v28);
  // pto: %1
  uint64_t v125 = (uint64_t) v13;
  TASSIGN(v124, v125);
  // pto: %slice_view
  Tile<TileType::Vec, float, 8, 1024, BLayout::RowMajor, 8, 512, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v126;
  // pto: %slice_view
  uint64_t v127 = (uint64_t) v13;
  TASSIGN(v126, v127);
  // pto: %2
  Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v128 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v29, v28);
  // pto: %2
  uint64_t v129 = (uint64_t) v15;
  TASSIGN(v128, v129);
  // pto: %54
  Tile<TileType::Vec, float, 8, 1024, BLayout::RowMajor, 8, 512, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v130;
  // pto: %54
  uint64_t v131 = (uint64_t) v15;
  TASSIGN(v130, v131);
  // pto: %qr_square_half_inline1304_inline3112__tile
  Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v132 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v29, v28);
  // pto: %qr_square_half_inline1304_inline3112__tile
  uint64_t v133 = (uint64_t) v13;
  TASSIGN(v132, v133);
  pipe_barrier(PIPE_V);
  TADD(v132, v126, v130);
  // pto: %3
  Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v134 = Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v29, v31);
  // pto: %3
  uint64_t v135 = (uint64_t) v13;
  TASSIGN(v134, v135);
  // pto: %55
  Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, 8, 256, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v136;
  // pto: %55
  uint64_t v137 = (uint64_t) v13;
  TASSIGN(v136, v137);
  // pto: %4
  Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v138 = Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v29, v31);
  // pto: %4
  uint64_t v139 = (uint64_t) v16;
  TASSIGN(v138, v139);
  // pto: %56
  Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, 8, 256, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v140;
  // pto: %56
  uint64_t v141 = (uint64_t) v16;
  TASSIGN(v140, v141);
  // pto: %qr_square_quarter_inline1300_inline3096__tile
  Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v142 = Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v29, v31);
  // pto: %qr_square_quarter_inline1300_inline3096__tile
  uint64_t v143 = (uint64_t) v13;
  TASSIGN(v142, v143);
  pipe_barrier(PIPE_V);
  TADD(v142, v136, v140);
  // pto: %5
  Tile<TileType::Vec, float, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v144 = Tile<TileType::Vec, float, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v29, v32);
  // pto: %5
  uint64_t v145 = (uint64_t) v13;
  TASSIGN(v144, v145);
  // pto: %57
  Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, 8, 128, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v146;
  // pto: %57
  uint64_t v147 = (uint64_t) v13;
  TASSIGN(v146, v147);
  // pto: %6
  Tile<TileType::Vec, float, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v148 = Tile<TileType::Vec, float, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v29, v32);
  // pto: %6
  uint64_t v149 = (uint64_t) v17;
  TASSIGN(v148, v149);
  // pto: %58
  Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, 8, 128, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v150;
  // pto: %58
  uint64_t v151 = (uint64_t) v17;
  TASSIGN(v150, v151);
  // pto: %qr_square_eighth_inline1305_inline3110__tile
  Tile<TileType::Vec, float, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v152 = Tile<TileType::Vec, float, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v29, v32);
  // pto: %qr_square_eighth_inline1305_inline3110__tile
  uint64_t v153 = (uint64_t) v13;
  TASSIGN(v152, v153);
  pipe_barrier(PIPE_V);
  TADD(v152, v146, v150);
  // pto: %7
  Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v154 = Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v29, v33);
  // pto: %7
  uint64_t v155 = (uint64_t) v13;
  TASSIGN(v154, v155);
  // pto: %59
  Tile<TileType::Vec, float, 8, 128, BLayout::RowMajor, 8, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v156;
  // pto: %59
  uint64_t v157 = (uint64_t) v13;
  TASSIGN(v156, v157);
  // pto: %8
  Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v158 = Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v29, v33);
  // pto: %8
  uint64_t v159 = (uint64_t) v18;
  TASSIGN(v158, v159);
  // pto: %60
  Tile<TileType::Vec, float, 8, 128, BLayout::RowMajor, 8, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v160;
  // pto: %60
  uint64_t v161 = (uint64_t) v18;
  TASSIGN(v160, v161);
  // pto: %qr_square_sixteenth_inline1299_inline3095__tile
  Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v162 = Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v29, v33);
  // pto: %qr_square_sixteenth_inline1299_inline3095__tile
  uint64_t v163 = (uint64_t) v14;
  TASSIGN(v162, v163);
  pipe_barrier(PIPE_V);
  TADD(v162, v156, v160);
  // pto: %tmp_tile
  Tile<TileType::Vec, float, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v164 = Tile<TileType::Vec, float, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v29, v32);
  // pto: %tmp_tile
  uint64_t v165 = (uint64_t) v13;
  TASSIGN(v164, v165);
  // pto: %9
  Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v166 = Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v29, v27);
  // pto: %9
  uint64_t v167 = (uint64_t) v19;
  TASSIGN(v166, v167);
  pipe_barrier(PIPE_V);
  TROWSUM(v166, v162, v164);
  // pto: %qr_sq_sum_inline1298_inline3088__tile
  Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v168 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v27, v29);
  // pto: %qr_sq_sum_inline1298_inline3088__tile
  uint64_t v169 = (uint64_t) v19;
  TASSIGN(v168, v169);
  // pto: %10
  Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v170 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v27, v29);
  // pto: %10
  uint64_t v171 = (uint64_t) v13;
  TASSIGN(v170, v171);
  pipe_barrier(PIPE_V);
  TMULS(v170, v168, v34);
  // pto: %qr_variance_inline1293_inline3086__tile
  Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v172 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v27, v29);
  // pto: %qr_variance_inline1293_inline3086__tile
  uint64_t v173 = (uint64_t) v13;
  TASSIGN(v172, v173);
  pipe_barrier(PIPE_V);
  TADDS(v172, v170, v35);
  // pto: %qr_rms_approx_inline1295_inline3083__tile
  Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v174 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v27, v29);
  // pto: %qr_rms_approx_inline1295_inline3083__tile
  uint64_t v175 = (uint64_t) v14;
  TASSIGN(v174, v175);
  pipe_barrier(PIPE_V);
  TSQRT(v174, v172);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  // pto: %qr_variance_bits_inline1314_inline3092__tile
  Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v176 = Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v27, v29);
  // pto: %qr_variance_bits_inline1314_inline3092__tile
  uint64_t v177 = (uint64_t) v13;
  TASSIGN(v176, v177);
  // pto: %qr_approx_bits_inline1318_inline3073__tile
  Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v178 = Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v27, v29);
  // pto: %qr_approx_bits_inline1318_inline3073__tile
  uint64_t v179 = (uint64_t) v14;
  TASSIGN(v178, v179);
  // pto: %qr_rounded_bits_inline1296_inline3076__tile
  Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v180 = Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v27, v29);
  // pto: %qr_rounded_bits_inline1296_inline3076__tile
  uint64_t v181 = (uint64_t) v20;
  TASSIGN(v180, v181);
  TEXPANDS(v180, v36);
  set_flag(PIPE_V, PIPE_S, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
  for (int64_t v182 = v30; v182 < v29; v182 += v27) {
    // pto: %61
    ;
    int32_t v183 = v176.GetValue(v182);
    // pto: %62
    ;
    int64_t v184 = (int64_t) v183;
    // pto: %63
    ;
    int32_t v185 = v178.GetValue(v182);
    // pto: %66
    ;
    // pto: %68, %69, %70
    ;
    if ((int64_t) v183 >= v37 & v184 < v38) {
      // pto: %71
      ;
      int64_t v186 = v184 / v39;
      // pto: %72, %73
      ;
      int64_t v187 = (int64_t) ((uint64_t) (v184 % v39) + (uint64_t) v39);
      // pto: %qr_y_bits_inline1297_inline3137__rv_v2
      ;
      int64_t v188;
      v188 = (int64_t) v185;
      for (int64_t v189 = v30; v189 < v40; v189 += v27) {
        // pto: %74
        ;
        int64_t v190 = v188 / v39;
        // pto: %75
        ;
        int64_t v191 = v188 % v39;
        // pto: %76, %77, %79
        ;
        int64_t v192 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) ((int64_t) (uint64_t) v191 + (uint64_t) v39) * (uint64_t) v41) + (uint64_t) v27);
        // pto: %80
        ;
        int64_t v193 = (int64_t) ((uint64_t) v192 * (uint64_t) v192);
        // pto: %84, %82, %81, %83
        ;
        int64_t v194 = (int64_t) ((uint64_t) v187 << (uint64_t) ((int64_t) (uint64_t) ((int64_t) (uint64_t) v186 - (uint64_t) ((int64_t) (uint64_t) v190 * (uint64_t) v41)) + (uint64_t) v42));
        // pto: %85
        ;
        uint64_t v195 = (uint64_t) v188;
        // pto: %85
        ;
        int64_t v196 = (int64_t) (v195 - (uint64_t) v43);
        // pto: %87, %88, %89, %91
        ;
        int64_t v197 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) ((int64_t) (uint64_t) (v196 % v39) + (uint64_t) v39) * (uint64_t) v41) + (uint64_t) v27);
        // pto: %92
        ;
        int64_t v198 = (int64_t) ((uint64_t) v197 * (uint64_t) v197);
        // pto: %96, %94, %86, %93, %95
        ;
        int64_t v199 = (int64_t) ((uint64_t) v187 << (uint64_t) ((int64_t) (uint64_t) ((int64_t) (uint64_t) v186 - (uint64_t) ((int64_t) (uint64_t) (v196 / v39) * (uint64_t) v41)) + (uint64_t) v42));
        // pto: %101
        ;
        int64_t v200 = v188 % v41;
        // pto: %102
        ;
        bool v201 = v200 == v43;
        // pto: %98, %100, %103, %104
        ;
        // pto: %qr_y_bits_inline1297_inline3137__phi_v6
        ;
        int64_t v202;
        if (v193 < v194 | v194 == v193 & v201) {
          // pto: %105
          ;
          uint64_t v203 = (uint64_t) v188;
          // pto: %105
          ;
          v202 = (int64_t) (v203 + (uint64_t) v43);
        } else {
          // pto: %107, %109, %112, %113, %qr_y_bits_inline1297_inline3137__phi_v5
          ;
          int64_t v204 = v199 < v198 | v199 == v198 & v201 ? v196 : v188;
          v202 = v204;
        };
        v188 = v202;
      };
      // pto: %115
      ;
      int32_t v205 = (int32_t) v188;
      v180.SetValue(v182, v205);
    } else {
      v180.SetValue(v182, v185);
    };
  }
  set_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
  // pto: %qr_rms_inline1291_inline3115__tile
  Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v206 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v27, v29);
  // pto: %qr_rms_inline1291_inline3115__tile
  uint64_t v207 = (uint64_t) v20;
  TASSIGN(v206, v207);
  // pto: %qr_inv_rms_inline1290_inline3121__tile
  Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v208 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v27, v29);
  // pto: %qr_inv_rms_inline1290_inline3121__tile
  uint64_t v209 = (uint64_t) v21;
  TASSIGN(v208, v209);
  TEXPANDS(v208, v44);
  set_flag(PIPE_V, PIPE_S, EVENT_ID1);
  wait_flag(PIPE_V, PIPE_S, EVENT_ID1);
  for (int64_t v210 = v30; v210 < v29; v210 += v27) {
    // pto: %qr_rms_value_inline1288_inline3124__tile
    ;
    float v211 = v206.GetValue(v210);
    // pto: %123
    ;
    float v212 = v45 / v211;
    v208.SetValue(v210, v212);
  }
  set_flag(PIPE_S, PIPE_V, EVENT_ID0);
  // pto: %_qr_sq_sum_inline3126__ssa_v0
  Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v213 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v27, v29);
  // pto: %_qr_sq_sum_inline3126__ssa_v0
  uint64_t v214 = (uint64_t) v19;
  TASSIGN(v213, v214);
  // pto: %qr_inv_rms_inline3089__ssa_v0
  Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v215 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v27, v29);
  // pto: %qr_inv_rms_inline3089__ssa_v0
  uint64_t v216 = (uint64_t) v21;
  TASSIGN(v215, v216);
  // pto: %_qr_rms_inline3127__ssa_v0
  Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v217 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v27, v29);
  // pto: %_qr_rms_inline3127__ssa_v0
  uint64_t v218 = (uint64_t) v20;
  TASSIGN(v217, v218);
  // pto: %qr_inv_rms_t_inline3128__tile
  Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v219 = Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v29, v27);
  // pto: %qr_inv_rms_t_inline3128__tile
  uint64_t v220 = (uint64_t) v21;
  TASSIGN(v219, v220);
  // pto: %qr_tile_amax_inline3129__tile
  Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v221 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v27, v29);
  // pto: %qr_tile_amax_inline3129__tile
  uint64_t v222 = (uint64_t) v22;
  TASSIGN(v221, v222);
  TEXPANDS(v221, v46);
  wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
  for (int64_t v223 = v30; v223 < v26; v223 += v28) {
    // pto: %11
    ;
    Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v224 = Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v29, v31);
    // pto: %11
    ;
    uint64_t v225 = (uint64_t) v13;
    TASSIGN(v224, v225);
    // pto: %127
    ;
    int64_t v226 = v223 < v30 ? v30 : v223;
    // pto: %128
    ;
    __gm__ float* v227 = PTOAS__GLOBAL_TENSOR_DATA(v56);
    // pto: %128
    ;
    const int64_t v228 = 0;
    // pto: %128
    ;
    const int64_t v229 = 1024;
    // pto: %128
    ;
    pto::Shape<1, 1, 1, 8, 256> v230 = pto::Shape<1, 1, 1, 8, 256>();
    // pto: %128
    ;
    pto::Stride<8192, 8192, 8192, 1024, 1> v231 = pto::Stride<8192, 8192, 8192, 1024, 1>();
    // pto: %128
    ;
    GlobalTensor<float, pto::Shape<1, 1, 1, 8, 256>, pto::Stride<8192, 8192, 8192, 1024, 1>, pto::Layout::ND> v232 = GlobalTensor<float, pto::Shape<1, 1, 1, 8, 256>, pto::Stride<8192, 8192, 8192, 1024, 1>, pto::Layout::ND>(v227 + (v228 + v107 * v229 + v226), v230, v231);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    TLOAD(v224, v232);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
    // pto: %12
    ;
    Tile<TileType::Vec, bfloat16_t, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v233 = Tile<TileType::Vec, bfloat16_t, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v27, v31);
    // pto: %12
    ;
    uint64_t v234 = (uint64_t) v23;
    TASSIGN(v233, v234);
    // pto: %gamma_cq__ssa_v0_pview
    ;
    __gm__ bfloat16_t* v235 = PTOAS__GLOBAL_TENSOR_DATA(v66);
    // pto: %gamma_cq__ssa_v0_pview
    ;
    const int64_t v236 = 0;
    // pto: %gamma_cq__ssa_v0_pview
    ;
    pto::Shape<1, 1, 1, 1, 256> v237 = pto::Shape<1, 1, 1, 1, 256>();
    // pto: %gamma_cq__ssa_v0_pview
    ;
    pto::Stride<256, 256, 256, 256, 1> v238 = pto::Stride<256, 256, 256, 256, 1>();
    // pto: %gamma_cq__ssa_v0_pview
    ;
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 256>, pto::Stride<256, 256, 256, 256, 1>, pto::Layout::ND> v239 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 256>, pto::Stride<256, 256, 256, 256, 1>, pto::Layout::ND>(v235 + (v236 + v226), v237, v238);
    TLOAD(v233, v239);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
    // pto: %13
    ;
    Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v240 = Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v29, v31);
    // pto: %13
    ;
    uint64_t v241 = (uint64_t) v14;
    TASSIGN(v240, v241);
    // pto: %131
    ;
    int64_t v242 = (int64_t) ((uint64_t) v223 + (uint64_t) v31);
    // pto: %132
    ;
    int64_t v243 = v242 < v30 ? v30 : v242;
    // pto: %133
    ;
    __gm__ float* v244 = PTOAS__GLOBAL_TENSOR_DATA(v56);
    // pto: %133
    ;
    const int64_t v245 = 0;
    // pto: %133
    ;
    const int64_t v246 = 1024;
    // pto: %133
    ;
    pto::Shape<1, 1, 1, 8, 256> v247 = pto::Shape<1, 1, 1, 8, 256>();
    // pto: %133
    ;
    pto::Stride<8192, 8192, 8192, 1024, 1> v248 = pto::Stride<8192, 8192, 8192, 1024, 1>();
    // pto: %133
    ;
    GlobalTensor<float, pto::Shape<1, 1, 1, 8, 256>, pto::Stride<8192, 8192, 8192, 1024, 1>, pto::Layout::ND> v249 = GlobalTensor<float, pto::Shape<1, 1, 1, 8, 256>, pto::Stride<8192, 8192, 8192, 1024, 1>, pto::Layout::ND>(v244 + (v245 + v107 * v246 + v243), v247, v248);
    TLOAD(v240, v249);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
    // pto: %14
    ;
    Tile<TileType::Vec, bfloat16_t, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v250 = Tile<TileType::Vec, bfloat16_t, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v27, v31);
    // pto: %14
    ;
    uint64_t v251 = (uint64_t) v24;
    TASSIGN(v250, v251);
    // pto: %136
    ;
    __gm__ bfloat16_t* v252 = PTOAS__GLOBAL_TENSOR_DATA(v66);
    // pto: %136
    ;
    const int64_t v253 = 0;
    // pto: %136
    ;
    pto::Shape<1, 1, 1, 1, 256> v254 = pto::Shape<1, 1, 1, 1, 256>();
    // pto: %136
    ;
    pto::Stride<256, 256, 256, 256, 1> v255 = pto::Stride<256, 256, 256, 256, 1>();
    // pto: %136
    ;
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 256>, pto::Stride<256, 256, 256, 256, 1>, pto::Layout::ND> v256 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 256>, pto::Stride<256, 256, 256, 256, 1>, pto::Layout::ND>(v252 + (v253 + v243), v254, v255);
    TLOAD(v250, v256);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);
    // pto: %15
    ;
    Tile<TileType::Vec, bfloat16_t, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v257 = Tile<TileType::Vec, bfloat16_t, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v29, v31);
    // pto: %15
    ;
    uint64_t v258 = (uint64_t) v19;
    TASSIGN(v257, v258);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
    RoundMode v259 = RoundMode::CAST_RINT;
    SaturationMode v260 = SaturationMode::OFF;
    TCVT(v257, v224, v259, v260);
    // pto: %qr_max_chunk_inline3078__tile
    ;
    Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v261 = Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v29, v31);
    // pto: %qr_max_chunk_inline3078__tile
    ;
    uint64_t v262 = (uint64_t) v13;
    TASSIGN(v261, v262);
    pipe_barrier(PIPE_V);
    RoundMode v263 = RoundMode::CAST_ROUND;
    SaturationMode v264 = SaturationMode::OFF;
    TCVT(v261, v257, v263, v264);
    // pto: %gamma_max_cast_inline3132__tile
    ;
    Tile<TileType::Vec, float, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v265 = Tile<TileType::Vec, float, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v27, v31);
    // pto: %gamma_max_cast_inline3132__tile
    ;
    uint64_t v266 = (uint64_t) v19;
    TASSIGN(v265, v266);
    pipe_barrier(PIPE_V);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
    RoundMode v267 = RoundMode::CAST_ROUND;
    SaturationMode v268 = SaturationMode::OFF;
    TCVT(v265, v233, v267, v268);
    // pto: %gamma_max_chunk_inline3134__tile
    ;
    Tile<TileType::Vec, float, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v269 = Tile<TileType::Vec, float, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v27, v31);
    // pto: %gamma_max_chunk_inline3134__tile
    ;
    uint64_t v270 = (uint64_t) v19;
    TASSIGN(v269, v270);
    // pto: %16
    ;
    Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v271 = Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v29, v31);
    // pto: %16
    ;
    uint64_t v272 = (uint64_t) v13;
    TASSIGN(v271, v272);
    TROWEXPANDMUL(v271, v261, v219);
    // pto: %qr_normalized_inline3136__tile
    ;
    Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v273 = Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v29, v31);
    // pto: %qr_normalized_inline3136__tile
    ;
    uint64_t v274 = (uint64_t) v13;
    TASSIGN(v273, v274);
    pipe_barrier(PIPE_V);
    TCOLEXPANDMUL(v273, v271, v269);
    // pto: %17
    ;
    Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v275 = Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v29, v31);
    // pto: %17
    ;
    uint64_t v276 = (uint64_t) v13;
    TASSIGN(v275, v276);
    pipe_barrier(PIPE_V);
    TABS(v275, v273);
    // pto: %18
    ;
    Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v277 = Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v29, v31);
    // pto: %18
    ;
    uint64_t v278 = (uint64_t) v19;
    TASSIGN(v277, v278);
    // pto: %19
    ;
    Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v279 = Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v29, v27);
    // pto: %19
    ;
    uint64_t v280 = (uint64_t) v23;
    TASSIGN(v279, v280);
    pipe_barrier(PIPE_V);
    TROWMAX(v279, v275, v277);
    // pto: %qr_normalized_max_inline3122__tile
    ;
    Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v281 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v27, v29);
    // pto: %qr_normalized_max_inline3122__tile
    ;
    uint64_t v282 = (uint64_t) v23;
    TASSIGN(v281, v282);
    // pto: %20
    ;
    Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v283 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v27, v29);
    // pto: %20
    ;
    uint64_t v284 = (uint64_t) v13;
    TASSIGN(v283, v284);
    pipe_barrier(PIPE_V);
    TMAX(v283, v221, v281);
    // pto: %21
    ;
    Tile<TileType::Vec, bfloat16_t, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v285 = Tile<TileType::Vec, bfloat16_t, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v29, v31);
    // pto: %21
    ;
    uint64_t v286 = (uint64_t) v20;
    TASSIGN(v285, v286);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
    RoundMode v287 = RoundMode::CAST_RINT;
    SaturationMode v288 = SaturationMode::OFF;
    TCVT(v285, v240, v287, v288);
    // pto: %22
    ;
    Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v289 = Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v29, v31);
    // pto: %22
    ;
    uint64_t v290 = (uint64_t) v14;
    TASSIGN(v289, v290);
    pipe_barrier(PIPE_V);
    RoundMode v291 = RoundMode::CAST_ROUND;
    SaturationMode v292 = SaturationMode::OFF;
    TCVT(v289, v285, v291, v292);
    // pto: %23
    ;
    Tile<TileType::Vec, float, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v293 = Tile<TileType::Vec, float, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v27, v31);
    // pto: %23
    ;
    uint64_t v294 = (uint64_t) v20;
    TASSIGN(v293, v294);
    pipe_barrier(PIPE_V);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);
    RoundMode v295 = RoundMode::CAST_ROUND;
    SaturationMode v296 = SaturationMode::OFF;
    TCVT(v293, v250, v295, v296);
    // pto: %24
    ;
    Tile<TileType::Vec, float, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v297 = Tile<TileType::Vec, float, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v27, v31);
    // pto: %24
    ;
    uint64_t v298 = (uint64_t) v20;
    TASSIGN(v297, v298);
    // pto: %25
    ;
    Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v299 = Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v29, v31);
    // pto: %25
    ;
    uint64_t v300 = (uint64_t) v14;
    TASSIGN(v299, v300);
    TROWEXPANDMUL(v299, v289, v219);
    // pto: %26
    ;
    Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v301 = Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v29, v31);
    // pto: %26
    ;
    uint64_t v302 = (uint64_t) v14;
    TASSIGN(v301, v302);
    pipe_barrier(PIPE_V);
    TCOLEXPANDMUL(v301, v299, v297);
    // pto: %27
    ;
    Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v303 = Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v29, v31);
    // pto: %27
    ;
    uint64_t v304 = (uint64_t) v14;
    TASSIGN(v303, v304);
    pipe_barrier(PIPE_V);
    TABS(v303, v301);
    // pto: %28
    ;
    Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v305 = Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v29, v31);
    // pto: %28
    ;
    uint64_t v306 = (uint64_t) v20;
    TASSIGN(v305, v306);
    // pto: %29
    ;
    Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v307 = Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v29, v27);
    // pto: %29
    ;
    uint64_t v308 = (uint64_t) v24;
    TASSIGN(v307, v308);
    pipe_barrier(PIPE_V);
    TROWMAX(v307, v303, v305);
    // pto: %30
    ;
    Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v309 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v27, v29);
    // pto: %30
    ;
    uint64_t v310 = (uint64_t) v24;
    TASSIGN(v309, v310);
    // pto: %31
    ;
    Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v311 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v27, v29);
    // pto: %31
    ;
    uint64_t v312 = (uint64_t) v22;
    TASSIGN(v311, v312);
    pipe_barrier(PIPE_V);
    TMAX(v311, v283, v309);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  }
  // pto: %qr_scale_quant_row_inline3094__tile
  Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v313 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v27, v29);
  // pto: %qr_scale_quant_row_inline3094__tile
  uint64_t v314 = (uint64_t) v25;
  TASSIGN(v313, v314);
  TEXPANDS(v313, v44);
  // pto: %qr_scale_dq_row_inline3098__tile
  Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v315 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v27, v29);
  // pto: %qr_scale_dq_row_inline3098__tile
  uint64_t v316 = (uint64_t) v13;
  TASSIGN(v315, v316);
  pipe_barrier(PIPE_V);
  TEXPANDS(v315, v44);
  set_flag(PIPE_V, PIPE_S, EVENT_ID2);
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_S, EVENT_ID2);
  for (int64_t v317 = v30; v317 < v29; v317 += v27) {
    // pto: %qr_amax_value_inline3138__tile
    ;
    float v318 = v221.GetValue(v317);
    // pto: %139
    ;
    float v319 = v47 / v318;
    v313.SetValue(v317, v319);
    // pto: %142
    ;
    float v320 = v45 / v319;
    v315.SetValue(v317, v320);
  }
  set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
  // pto: %qr_scale_quant_t_inline3109__tile
  Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v321 = Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v29, v27);
  // pto: %qr_scale_quant_t_inline3109__tile
  uint64_t v322 = (uint64_t) v25;
  TASSIGN(v321, v322);
  // pto: %qr_tile_scale_dq_inline3074__tile
  Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v323 = Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v29, v27);
  // pto: %qr_tile_scale_dq_inline3074__tile
  uint64_t v324 = (uint64_t) v13;
  TASSIGN(v323, v324);
  // pto: %qr_scale_pad_store_inline1827__iter_v1_pview
  __gm__ float* v325 = PTOAS__GLOBAL_TENSOR_DATA(v74);
  // pto: %qr_scale_pad_store_inline1827__iter_v1_pview
  const int64_t v326 = 0;
  // pto: %qr_scale_pad_store_inline1827__iter_v1_pview
  pto::Shape<1, 1, 1, 8, 1> v327 = pto::Shape<1, 1, 1, 8, 1>();
  // pto: %qr_scale_pad_store_inline1827__iter_v1_pview
  pto::Stride<8, 8, 8, 1, 512> v328 = pto::Stride<8, 8, 8, 1, 512>();
  // pto: %qr_scale_pad_store_inline1827__iter_v1_pview
  GlobalTensor<float, pto::Shape<1, 1, 1, 8, 1>, pto::Stride<8, 8, 8, 1, 512>, pto::Layout::DN> v329 = GlobalTensor<float, pto::Shape<1, 1, 1, 8, 1>, pto::Stride<8, 8, 8, 1, 512>, pto::Layout::DN>(v325 + (v326 + v107), v327, v328);
  wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  TSTORE(v329, v323);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  // pto: %146
  bool v330 = v103 == v29;
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  if (v330) {
    // pto: %147
    ;
    int64_t v331 = v104 < v30 ? v30 : v104;
    // pto: %qr_scale_view_inline3100__ssa_v0_pview
    ;
    int64_t v332 = (int64_t) v10;
    // pto: %qr_scale_view_inline3100__ssa_v0_pview
    ;
    const int64_t v333 = 0;
    // pto: %qr_scale_view_inline3100__ssa_v0_pview
    ;
    __gm__ float* v334 = PTOAS__GLOBAL_TENSOR_DATA(v83);
    // pto: %qr_scale_view_inline3100__ssa_v0_pview
    ;
    const int64_t v335 = 1;
    // pto: %qr_scale_view_inline3100__ssa_v0_pview
    ;
    const int64_t v336 = 1;
    // pto: %qr_scale_view_inline3100__ssa_v0_pview
    ;
    const int64_t v337 = 1;
    // pto: %qr_scale_view_inline3100__ssa_v0_pview
    ;
    int64_t v338 = v29 * v27;
    // pto: %qr_scale_view_inline3100__ssa_v0_pview
    ;
    int64_t v339 = v337 * v338;
    // pto: %qr_scale_view_inline3100__ssa_v0_pview
    ;
    pto::Shape<1, 1, 1, 8, 1> v340 = pto::Shape<1, 1, 1, 8, 1>(v335, v336, v337, v29, v27);
    // pto: %qr_scale_view_inline3100__ssa_v0_pview
    ;
    pto::Stride<-1, -1, -1, -1, -1> v341 = pto::Stride<-1, -1, -1, -1, -1>(v336 * v339, v339, v338, v27, v332);
    // pto: %qr_scale_view_inline3100__ssa_v0_pview
    ;
    GlobalTensor<float, pto::Shape<1, 1, 1, 8, 1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN> v342 = GlobalTensor<float, pto::Shape<1, 1, 1, 8, 1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN>(v334 + (v333 + v331 * v27 + v30 * v332), v340, v341);
    TSTORE(v342, v323);
  } else {
    // pto: %qr_scale_tail_inline3123__ssa_v0
    ;
    Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v343 = Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v103, v27);
    // pto: %qr_scale_tail_inline3123__ssa_v0
    ;
    uint64_t v344 = (uint64_t) v13;
    TASSIGN(v343, v344);
    // pto: %qr_scale_pad_store_inline1827__tile_pview
    ;
    const int64_t v345 = 0;
    // pto: %qr_scale_pad_store_inline1827__tile_pview
    ;
    __gm__ float* v346 = PTOAS__GLOBAL_TENSOR_DATA(v74);
    // pto: %qr_scale_pad_store_inline1827__tile_pview
    ;
    const int64_t v347 = 1;
    // pto: %qr_scale_pad_store_inline1827__tile_pview
    ;
    const int64_t v348 = 1;
    // pto: %qr_scale_pad_store_inline1827__tile_pview
    ;
    const int64_t v349 = 1;
    // pto: %qr_scale_pad_store_inline1827__tile_pview
    ;
    int64_t v350 = v103 * v27;
    // pto: %qr_scale_pad_store_inline1827__tile_pview
    ;
    int64_t v351 = v349 * v350;
    // pto: %qr_scale_pad_store_inline1827__tile_pview
    ;
    pto::Shape<1, 1, 1, -1, 1> v352 = pto::Shape<1, 1, 1, -1, 1>(v347, v348, v349, v103, v27);
    // pto: %qr_scale_pad_store_inline1827__tile_pview
    ;
    pto::Stride<-1, -1, -1, -1, -1> v353 = pto::Stride<-1, -1, -1, -1, -1>(v348 * v351, v351, v350, v27, v28);
    // pto: %qr_scale_pad_store_inline1827__tile_pview
    ;
    GlobalTensor<float, pto::Shape<1, 1, 1, -1, 1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN> v354 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, 1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN>(v346 + (v345 + v107 * v27 + v30 * v28), v352, v353);
    TLOAD(v343, v354);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    // pto: %149
    ;
    int64_t v355 = v104 < v30 ? v30 : v104;
    // pto: %150
    ;
    int64_t v356 = (int64_t) v10;
    // pto: %150
    ;
    const int64_t v357 = 0;
    // pto: %150
    ;
    __gm__ float* v358 = PTOAS__GLOBAL_TENSOR_DATA(v83);
    // pto: %150
    ;
    const int64_t v359 = 1;
    // pto: %150
    ;
    const int64_t v360 = 1;
    // pto: %150
    ;
    const int64_t v361 = 1;
    // pto: %150
    ;
    int64_t v362 = v103 * v27;
    // pto: %150
    ;
    int64_t v363 = v361 * v362;
    // pto: %150
    ;
    pto::Shape<1, 1, 1, -1, 1> v364 = pto::Shape<1, 1, 1, -1, 1>(v359, v360, v361, v103, v27);
    // pto: %150
    ;
    pto::Stride<-1, -1, -1, -1, -1> v365 = pto::Stride<-1, -1, -1, -1, -1>(v360 * v363, v363, v362, v27, v356);
    // pto: %150
    ;
    GlobalTensor<float, pto::Shape<1, 1, 1, -1, 1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN> v366 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, 1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN>(v358 + (v357 + v355 * v27 + v30 * v356), v364, v365);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    TSTORE(v366, v343);
  }
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
  for (int64_t v367 = v30; v367 < v26; v367 += v28) {
    // pto: %32
    ;
    Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v368 = Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v29, v31);
    // pto: %32
    ;
    uint64_t v369 = (uint64_t) v13;
    TASSIGN(v368, v369);
    // pto: %152
    ;
    int64_t v370 = v367 < v30 ? v30 : v367;
    // pto: %153
    ;
    __gm__ float* v371 = PTOAS__GLOBAL_TENSOR_DATA(v56);
    // pto: %153
    ;
    const int64_t v372 = 0;
    // pto: %153
    ;
    const int64_t v373 = 1024;
    // pto: %153
    ;
    pto::Shape<1, 1, 1, 8, 256> v374 = pto::Shape<1, 1, 1, 8, 256>();
    // pto: %153
    ;
    pto::Stride<8192, 8192, 8192, 1024, 1> v375 = pto::Stride<8192, 8192, 8192, 1024, 1>();
    // pto: %153
    ;
    GlobalTensor<float, pto::Shape<1, 1, 1, 8, 256>, pto::Stride<8192, 8192, 8192, 1024, 1>, pto::Layout::ND> v376 = GlobalTensor<float, pto::Shape<1, 1, 1, 8, 256>, pto::Stride<8192, 8192, 8192, 1024, 1>, pto::Layout::ND>(v371 + (v372 + v107 * v373 + v370), v374, v375);
    TLOAD(v368, v376);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID5);
    // pto: %33
    ;
    Tile<TileType::Vec, bfloat16_t, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v377 = Tile<TileType::Vec, bfloat16_t, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v27, v31);
    // pto: %33
    ;
    uint64_t v378 = (uint64_t) v23;
    TASSIGN(v377, v378);
    // pto: %155
    ;
    __gm__ bfloat16_t* v379 = PTOAS__GLOBAL_TENSOR_DATA(v66);
    // pto: %155
    ;
    const int64_t v380 = 0;
    // pto: %155
    ;
    pto::Shape<1, 1, 1, 1, 256> v381 = pto::Shape<1, 1, 1, 1, 256>();
    // pto: %155
    ;
    pto::Stride<256, 256, 256, 256, 1> v382 = pto::Stride<256, 256, 256, 256, 1>();
    // pto: %155
    ;
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 256>, pto::Stride<256, 256, 256, 256, 1>, pto::Layout::ND> v383 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 256>, pto::Stride<256, 256, 256, 256, 1>, pto::Layout::ND>(v379 + (v380 + v370), v381, v382);
    TLOAD(v377, v383);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID6);
    // pto: %34
    ;
    Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v384 = Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v29, v31);
    // pto: %34
    ;
    uint64_t v385 = (uint64_t) v14;
    TASSIGN(v384, v385);
    // pto: %157
    ;
    int64_t v386 = (int64_t) ((uint64_t) v367 + (uint64_t) v31);
    // pto: %158
    ;
    int64_t v387 = v386 < v30 ? v30 : v386;
    // pto: %159
    ;
    __gm__ float* v388 = PTOAS__GLOBAL_TENSOR_DATA(v56);
    // pto: %159
    ;
    const int64_t v389 = 0;
    // pto: %159
    ;
    const int64_t v390 = 1024;
    // pto: %159
    ;
    pto::Shape<1, 1, 1, 8, 256> v391 = pto::Shape<1, 1, 1, 8, 256>();
    // pto: %159
    ;
    pto::Stride<8192, 8192, 8192, 1024, 1> v392 = pto::Stride<8192, 8192, 8192, 1024, 1>();
    // pto: %159
    ;
    GlobalTensor<float, pto::Shape<1, 1, 1, 8, 256>, pto::Stride<8192, 8192, 8192, 1024, 1>, pto::Layout::ND> v393 = GlobalTensor<float, pto::Shape<1, 1, 1, 8, 256>, pto::Stride<8192, 8192, 8192, 1024, 1>, pto::Layout::ND>(v388 + (v389 + v107 * v390 + v387), v391, v392);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
    TLOAD(v384, v393);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID7);
    // pto: %35
    ;
    Tile<TileType::Vec, bfloat16_t, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v394 = Tile<TileType::Vec, bfloat16_t, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v27, v31);
    // pto: %35
    ;
    uint64_t v395 = (uint64_t) v24;
    TASSIGN(v394, v395);
    // pto: %162
    ;
    __gm__ bfloat16_t* v396 = PTOAS__GLOBAL_TENSOR_DATA(v66);
    // pto: %162
    ;
    const int64_t v397 = 0;
    // pto: %162
    ;
    pto::Shape<1, 1, 1, 1, 256> v398 = pto::Shape<1, 1, 1, 1, 256>();
    // pto: %162
    ;
    pto::Stride<256, 256, 256, 256, 1> v399 = pto::Stride<256, 256, 256, 256, 1>();
    // pto: %162
    ;
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 256>, pto::Stride<256, 256, 256, 256, 1>, pto::Layout::ND> v400 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 256>, pto::Stride<256, 256, 256, 256, 1>, pto::Layout::ND>(v396 + (v397 + v387), v398, v399);
    TLOAD(v394, v400);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    // pto: %36
    ;
    Tile<TileType::Vec, bfloat16_t, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v401 = Tile<TileType::Vec, bfloat16_t, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v29, v31);
    // pto: %36
    ;
    uint64_t v402 = (uint64_t) v19;
    TASSIGN(v401, v402);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID5);
    RoundMode v403 = RoundMode::CAST_RINT;
    SaturationMode v404 = SaturationMode::OFF;
    TCVT(v401, v368, v403, v404);
    // pto: %qr_chunk_inline3072__tile
    ;
    Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v405 = Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v29, v31);
    // pto: %qr_chunk_inline3072__tile
    ;
    uint64_t v406 = (uint64_t) v13;
    TASSIGN(v405, v406);
    pipe_barrier(PIPE_V);
    RoundMode v407 = RoundMode::CAST_ROUND;
    SaturationMode v408 = SaturationMode::OFF;
    TCVT(v405, v401, v407, v408);
    // pto: %gamma_q_cast_inline3130__tile
    ;
    Tile<TileType::Vec, float, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v409 = Tile<TileType::Vec, float, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v27, v31);
    // pto: %gamma_q_cast_inline3130__tile
    ;
    uint64_t v410 = (uint64_t) v19;
    TASSIGN(v409, v410);
    pipe_barrier(PIPE_V);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID6);
    RoundMode v411 = RoundMode::CAST_ROUND;
    SaturationMode v412 = SaturationMode::OFF;
    TCVT(v409, v377, v411, v412);
    // pto: %gamma_q_chunk_inline3071__tile
    ;
    Tile<TileType::Vec, float, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v413 = Tile<TileType::Vec, float, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v27, v31);
    // pto: %gamma_q_chunk_inline3071__tile
    ;
    uint64_t v414 = (uint64_t) v19;
    TASSIGN(v413, v414);
    // pto: %37
    ;
    Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v415 = Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v29, v31);
    // pto: %37
    ;
    uint64_t v416 = (uint64_t) v13;
    TASSIGN(v415, v416);
    TROWEXPANDMUL(v415, v405, v219);
    // pto: %qr_q_normed_inline3070__tile
    ;
    Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v417 = Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v29, v31);
    // pto: %qr_q_normed_inline3070__tile
    ;
    uint64_t v418 = (uint64_t) v13;
    TASSIGN(v417, v418);
    pipe_barrier(PIPE_V);
    TCOLEXPANDMUL(v417, v415, v413);
    // pto: %qr_q_scaled_inline3081__tile
    ;
    Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v419 = Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v29, v31);
    // pto: %qr_q_scaled_inline3081__tile
    ;
    uint64_t v420 = (uint64_t) v13;
    TASSIGN(v419, v420);
    pipe_barrier(PIPE_V);
    TROWEXPANDMUL(v419, v417, v321);
    // pto: %qr_q_i32_inline3069__tile
    ;
    Tile<TileType::Vec, int32_t, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v421 = Tile<TileType::Vec, int32_t, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v29, v31);
    // pto: %qr_q_i32_inline3069__tile
    ;
    uint64_t v422 = (uint64_t) v13;
    TASSIGN(v421, v422);
    pipe_barrier(PIPE_V);
    RoundMode v423 = RoundMode::CAST_RINT;
    SaturationMode v424 = SaturationMode::ON;
    TCVT(v421, v419, v423, v424);
    // pto: %qr_q_half_inline3084__tile
    ;
    Tile<TileType::Vec, half, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v425 = Tile<TileType::Vec, half, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v29, v31);
    // pto: %qr_q_half_inline3084__tile
    ;
    uint64_t v426 = (uint64_t) v13;
    TASSIGN(v425, v426);
    pipe_barrier(PIPE_V);
    RoundMode v427 = RoundMode::CAST_ROUND;
    SaturationMode v428 = SaturationMode::OFF;
    TCVT(v425, v421, v427, v428);
    // pto: %qr_q_i8_inline3080__tile
    ;
    Tile<TileType::Vec, int8_t, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v429 = Tile<TileType::Vec, int8_t, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v29, v31);
    // pto: %qr_q_i8_inline3080__tile
    ;
    uint64_t v430 = (uint64_t) v13;
    TASSIGN(v429, v430);
    pipe_barrier(PIPE_V);
    RoundMode v431 = RoundMode::CAST_TRUNC;
    SaturationMode v432 = SaturationMode::ON;
    TCVT(v429, v425, v431, v432);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
    // pto: %qr_i8_matmul_inline1825__iter_v3_pview
    ;
    __gm__ int8_t* v433 = PTOAS__GLOBAL_TENSOR_DATA(v91);
    // pto: %qr_i8_matmul_inline1825__iter_v3_pview
    ;
    const int64_t v434 = 0;
    // pto: %qr_i8_matmul_inline1825__iter_v3_pview
    ;
    const int64_t v435 = 1024;
    // pto: %qr_i8_matmul_inline1825__iter_v3_pview
    ;
    pto::Shape<1, 1, 1, 8, 256> v436 = pto::Shape<1, 1, 1, 8, 256>();
    // pto: %qr_i8_matmul_inline1825__iter_v3_pview
    ;
    pto::Stride<8192, 8192, 8192, 1024, 1> v437 = pto::Stride<8192, 8192, 8192, 1024, 1>();
    // pto: %qr_i8_matmul_inline1825__iter_v3_pview
    ;
    GlobalTensor<int8_t, pto::Shape<1, 1, 1, 8, 256>, pto::Stride<8192, 8192, 8192, 1024, 1>, pto::Layout::ND> v438 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 8, 256>, pto::Stride<8192, 8192, 8192, 1024, 1>, pto::Layout::ND>(v433 + (v434 + v107 * v435 + v370), v436, v437);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
    pipe_barrier(PIPE_MTE3);
    TSTORE(v438, v429);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
    if (v330) {
      // pto: %166
      ;
      int64_t v439 = v104 < v30 ? v30 : v104;
      // pto: %qr_view_inline3085__iter_v1_pview
      ;
      __gm__ int8_t* v440 = PTOAS__GLOBAL_TENSOR_DATA(v100);
      // pto: %qr_view_inline3085__iter_v1_pview
      ;
      const int64_t v441 = 0;
      // pto: %qr_view_inline3085__iter_v1_pview
      ;
      const int64_t v442 = 1024;
      // pto: %qr_view_inline3085__iter_v1_pview
      ;
      pto::Shape<1, 1, 1, 8, 256> v443 = pto::Shape<1, 1, 1, 8, 256>();
      // pto: %qr_view_inline3085__iter_v1_pview
      ;
      pto::Stride<8192, 8192, 8192, 1024, 1> v444 = pto::Stride<8192, 8192, 8192, 1024, 1>();
      // pto: %qr_view_inline3085__iter_v1_pview
      ;
      GlobalTensor<int8_t, pto::Shape<1, 1, 1, 8, 256>, pto::Stride<8192, 8192, 8192, 1024, 1>, pto::Layout::ND> v445 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 8, 256>, pto::Stride<8192, 8192, 8192, 1024, 1>, pto::Layout::ND>(v440 + (v441 + v439 * v442 + v370), v443, v444);
      TSTORE(v445, v429);
    } else {
      // pto: %qr_q_tail_inline3133__ssa_v0
      ;
      Tile<TileType::Vec, int8_t, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v446 = Tile<TileType::Vec, int8_t, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v103, v31);
      // pto: %qr_q_tail_inline3133__ssa_v0
      ;
      uint64_t v447 = (uint64_t) v13;
      TASSIGN(v446, v447);
      // pto: %qr_i8_matmul_inline1825__tile_pview
      ;
      const int64_t v448 = 0;
      // pto: %qr_i8_matmul_inline1825__tile_pview
      ;
      __gm__ int8_t* v449 = PTOAS__GLOBAL_TENSOR_DATA(v91);
      // pto: %qr_i8_matmul_inline1825__tile_pview
      ;
      const int64_t v450 = 1;
      // pto: %qr_i8_matmul_inline1825__tile_pview
      ;
      const int64_t v451 = 1;
      // pto: %qr_i8_matmul_inline1825__tile_pview
      ;
      const int64_t v452 = 1;
      // pto: %qr_i8_matmul_inline1825__tile_pview
      ;
      int64_t v453 = v103 * v26;
      // pto: %qr_i8_matmul_inline1825__tile_pview
      ;
      int64_t v454 = v452 * v453;
      // pto: %qr_i8_matmul_inline1825__tile_pview
      ;
      pto::Shape<1, 1, 1, -1, 256> v455 = pto::Shape<1, 1, 1, -1, 256>(v450, v451, v452, v103, v31);
      // pto: %qr_i8_matmul_inline1825__tile_pview
      ;
      pto::Stride<-1, -1, -1, -1, -1> v456 = pto::Stride<-1, -1, -1, -1, -1>(v451 * v454, v454, v453, v26, v27);
      // pto: %qr_i8_matmul_inline1825__tile_pview
      ;
      GlobalTensor<int8_t, pto::Shape<1, 1, 1, -1, 256>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v457 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, -1, 256>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v449 + (v448 + v107 * v26 + v370 * v27), v455, v456);
      TLOAD(v446, v457);
      set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID1);
      // pto: %170
      ;
      int64_t v458 = v104 < v30 ? v30 : v104;
      // pto: %172
      ;
      const int64_t v459 = 0;
      // pto: %172
      ;
      __gm__ int8_t* v460 = PTOAS__GLOBAL_TENSOR_DATA(v100);
      // pto: %172
      ;
      const int64_t v461 = 1;
      // pto: %172
      ;
      const int64_t v462 = 1;
      // pto: %172
      ;
      const int64_t v463 = 1;
      // pto: %172
      ;
      int64_t v464 = v103 * v26;
      // pto: %172
      ;
      int64_t v465 = v463 * v464;
      // pto: %172
      ;
      pto::Shape<1, 1, 1, -1, 256> v466 = pto::Shape<1, 1, 1, -1, 256>(v461, v462, v463, v103, v31);
      // pto: %172
      ;
      pto::Stride<-1, -1, -1, -1, -1> v467 = pto::Stride<-1, -1, -1, -1, -1>(v462 * v465, v465, v464, v26, v27);
      // pto: %172
      ;
      GlobalTensor<int8_t, pto::Shape<1, 1, 1, -1, 256>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v468 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, -1, 256>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v460 + (v459 + v458 * v26 + v370 * v27), v466, v467);
      wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID1);
      TSTORE(v468, v446);
    };
    // pto: %38
    ;
    Tile<TileType::Vec, bfloat16_t, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v469 = Tile<TileType::Vec, bfloat16_t, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v29, v31);
    // pto: %38
    ;
    uint64_t v470 = (uint64_t) v20;
    TASSIGN(v469, v470);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID7);
    RoundMode v471 = RoundMode::CAST_RINT;
    SaturationMode v472 = SaturationMode::OFF;
    TCVT(v469, v384, v471, v472);
    // pto: %39
    ;
    Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v473 = Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v29, v31);
    // pto: %39
    ;
    uint64_t v474 = (uint64_t) v14;
    TASSIGN(v473, v474);
    pipe_barrier(PIPE_V);
    RoundMode v475 = RoundMode::CAST_ROUND;
    SaturationMode v476 = SaturationMode::OFF;
    TCVT(v473, v469, v475, v476);
    // pto: %40
    ;
    Tile<TileType::Vec, float, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v477 = Tile<TileType::Vec, float, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v27, v31);
    // pto: %40
    ;
    uint64_t v478 = (uint64_t) v20;
    TASSIGN(v477, v478);
    pipe_barrier(PIPE_V);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    RoundMode v479 = RoundMode::CAST_ROUND;
    SaturationMode v480 = SaturationMode::OFF;
    TCVT(v477, v394, v479, v480);
    // pto: %41
    ;
    Tile<TileType::Vec, float, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v481 = Tile<TileType::Vec, float, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v27, v31);
    // pto: %41
    ;
    uint64_t v482 = (uint64_t) v20;
    TASSIGN(v481, v482);
    // pto: %42
    ;
    Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v483 = Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v29, v31);
    // pto: %42
    ;
    uint64_t v484 = (uint64_t) v14;
    TASSIGN(v483, v484);
    TROWEXPANDMUL(v483, v473, v219);
    // pto: %43
    ;
    Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v485 = Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v29, v31);
    // pto: %43
    ;
    uint64_t v486 = (uint64_t) v14;
    TASSIGN(v485, v486);
    pipe_barrier(PIPE_V);
    TCOLEXPANDMUL(v485, v483, v481);
    // pto: %44
    ;
    Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v487 = Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v29, v31);
    // pto: %44
    ;
    uint64_t v488 = (uint64_t) v14;
    TASSIGN(v487, v488);
    pipe_barrier(PIPE_V);
    TROWEXPANDMUL(v487, v485, v321);
    // pto: %45
    ;
    Tile<TileType::Vec, int32_t, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v489 = Tile<TileType::Vec, int32_t, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v29, v31);
    // pto: %45
    ;
    uint64_t v490 = (uint64_t) v14;
    TASSIGN(v489, v490);
    pipe_barrier(PIPE_V);
    RoundMode v491 = RoundMode::CAST_RINT;
    SaturationMode v492 = SaturationMode::ON;
    TCVT(v489, v487, v491, v492);
    // pto: %46
    ;
    Tile<TileType::Vec, half, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v493 = Tile<TileType::Vec, half, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v29, v31);
    // pto: %46
    ;
    uint64_t v494 = (uint64_t) v14;
    TASSIGN(v493, v494);
    pipe_barrier(PIPE_V);
    RoundMode v495 = RoundMode::CAST_ROUND;
    SaturationMode v496 = SaturationMode::OFF;
    TCVT(v493, v489, v495, v496);
    // pto: %47
    ;
    Tile<TileType::Vec, int8_t, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v497 = Tile<TileType::Vec, int8_t, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v29, v31);
    // pto: %47
    ;
    uint64_t v498 = (uint64_t) v14;
    TASSIGN(v497, v498);
    pipe_barrier(PIPE_V);
    RoundMode v499 = RoundMode::CAST_TRUNC;
    SaturationMode v500 = SaturationMode::ON;
    TCVT(v497, v493, v499, v500);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
    // pto: %177
    ;
    __gm__ int8_t* v501 = PTOAS__GLOBAL_TENSOR_DATA(v91);
    // pto: %177
    ;
    const int64_t v502 = 0;
    // pto: %177
    ;
    const int64_t v503 = 1024;
    // pto: %177
    ;
    pto::Shape<1, 1, 1, 8, 256> v504 = pto::Shape<1, 1, 1, 8, 256>();
    // pto: %177
    ;
    pto::Stride<8192, 8192, 8192, 1024, 1> v505 = pto::Stride<8192, 8192, 8192, 1024, 1>();
    // pto: %177
    ;
    GlobalTensor<int8_t, pto::Shape<1, 1, 1, 8, 256>, pto::Stride<8192, 8192, 8192, 1024, 1>, pto::Layout::ND> v506 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 8, 256>, pto::Stride<8192, 8192, 8192, 1024, 1>, pto::Layout::ND>(v501 + (v502 + v107 * v503 + v387), v504, v505);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
    pipe_barrier(PIPE_MTE3);
    TSTORE(v506, v497);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID4);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID4);
    if (v330) {
      // pto: %180
      ;
      int64_t v507 = v104 < v30 ? v30 : v104;
      // pto: %qr_view_inline3085__phi_v4_pview
      ;
      __gm__ int8_t* v508 = PTOAS__GLOBAL_TENSOR_DATA(v100);
      // pto: %qr_view_inline3085__phi_v4_pview
      ;
      const int64_t v509 = 0;
      // pto: %qr_view_inline3085__phi_v4_pview
      ;
      const int64_t v510 = 1024;
      // pto: %qr_view_inline3085__phi_v4_pview
      ;
      pto::Shape<1, 1, 1, 8, 256> v511 = pto::Shape<1, 1, 1, 8, 256>();
      // pto: %qr_view_inline3085__phi_v4_pview
      ;
      pto::Stride<8192, 8192, 8192, 1024, 1> v512 = pto::Stride<8192, 8192, 8192, 1024, 1>();
      // pto: %qr_view_inline3085__phi_v4_pview
      ;
      GlobalTensor<int8_t, pto::Shape<1, 1, 1, 8, 256>, pto::Stride<8192, 8192, 8192, 1024, 1>, pto::Layout::ND> v513 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 8, 256>, pto::Stride<8192, 8192, 8192, 1024, 1>, pto::Layout::ND>(v508 + (v509 + v507 * v510 + v387), v511, v512);
      TSTORE(v513, v497);
    } else {
      // pto: %48
      ;
      Tile<TileType::Vec, int8_t, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v514 = Tile<TileType::Vec, int8_t, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v103, v31);
      // pto: %48
      ;
      uint64_t v515 = (uint64_t) v14;
      TASSIGN(v514, v515);
      // pto: %186
      ;
      const int64_t v516 = 0;
      // pto: %186
      ;
      __gm__ int8_t* v517 = PTOAS__GLOBAL_TENSOR_DATA(v91);
      // pto: %186
      ;
      const int64_t v518 = 1;
      // pto: %186
      ;
      const int64_t v519 = 1;
      // pto: %186
      ;
      const int64_t v520 = 1;
      // pto: %186
      ;
      int64_t v521 = v103 * v26;
      // pto: %186
      ;
      int64_t v522 = v520 * v521;
      // pto: %186
      ;
      pto::Shape<1, 1, 1, -1, 256> v523 = pto::Shape<1, 1, 1, -1, 256>(v518, v519, v520, v103, v31);
      // pto: %186
      ;
      pto::Stride<-1, -1, -1, -1, -1> v524 = pto::Stride<-1, -1, -1, -1, -1>(v519 * v522, v522, v521, v26, v27);
      // pto: %186
      ;
      GlobalTensor<int8_t, pto::Shape<1, 1, 1, -1, 256>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v525 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, -1, 256>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v517 + (v516 + v107 * v26 + v387 * v27), v523, v524);
      TLOAD(v514, v525);
      set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID2);
      // pto: %187
      ;
      int64_t v526 = v104 < v30 ? v30 : v104;
      // pto: %190
      ;
      const int64_t v527 = 0;
      // pto: %190
      ;
      __gm__ int8_t* v528 = PTOAS__GLOBAL_TENSOR_DATA(v100);
      // pto: %190
      ;
      const int64_t v529 = 1;
      // pto: %190
      ;
      const int64_t v530 = 1;
      // pto: %190
      ;
      const int64_t v531 = 1;
      // pto: %190
      ;
      int64_t v532 = v103 * v26;
      // pto: %190
      ;
      int64_t v533 = v531 * v532;
      // pto: %190
      ;
      pto::Shape<1, 1, 1, -1, 256> v534 = pto::Shape<1, 1, 1, -1, 256>(v529, v530, v531, v103, v31);
      // pto: %190
      ;
      pto::Stride<-1, -1, -1, -1, -1> v535 = pto::Stride<-1, -1, -1, -1, -1>(v530 * v533, v533, v532, v26, v27);
      // pto: %190
      ;
      GlobalTensor<int8_t, pto::Shape<1, 1, 1, -1, 256>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v536 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, -1, 256>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v528 + (v527 + v526 * v26 + v387 * v27), v534, v535);
      wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID2);
      TSTORE(v536, v514);
    };
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
  }
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
  #endif // __DAV_VEC__

  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}