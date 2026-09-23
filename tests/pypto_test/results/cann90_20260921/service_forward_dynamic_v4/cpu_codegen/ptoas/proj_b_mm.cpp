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

AICORE void proj_b_mm(__gm__ int32_t* v1, __gm__ int8_t* v2, __gm__ int8_t* v3, int64_t v4, int64_t v5, int32_t v6, int32_t v7) {
  using T = float;

  #if defined(__DAV_CUBE__)
  // pto: %c0_i64
  const int64_t v8 = 0;
  // pto: %c32768_i64
  const int64_t v9 = 32768;
  // pto: %c98304_i64
  const int64_t v10 = 98304;
  // pto: %c131072_i64
  const int64_t v11 = 131072;
  // pto: %c384_index
  const int64_t v12 = 384;
  // pto: %c32768_index
  const int64_t v13 = 32768;
  // pto: %c1_index
  const int64_t v14 = 1;
  // pto: %c8192_index
  const int64_t v15 = 8192;
  // pto: %c4096_index
  const int64_t v16 = 4096;
  // pto: %c8_index
  const int64_t v17 = 8;
  // pto: %c128_index
  const int64_t v18 = 128;
  // pto: %c512_index
  const int64_t v19 = 512;
  // pto: %c0_index
  const int64_t v20 = 0;
  // pto: %c2_index
  const int64_t v21 = 2;
  // pto: %c256_index
  const int64_t v22 = 256;
  // pto: %c4_index
  const int64_t v23 = 4;
  // pto: %cn1_index
  const int64_t v24 = -1;
  // pto: %partials_inline342__iter_v1_view
  const int64_t v25 = 1;
  // pto: %partials_inline342__iter_v1_view
  const int64_t v26 = 1;
  // pto: %partials_inline342__iter_v1_view
  const int64_t v27 = 1;
  // pto: %partials_inline342__iter_v1_view
  int64_t v28 = v12 * v13;
  // pto: %partials_inline342__iter_v1_view
  int64_t v29 = v27 * v28;
  // pto: %partials_inline342__iter_v1_view
  pto::Shape<1, 1, 1, -1, -1> v30 = pto::Shape<1, 1, 1, -1, -1>(v25, v26, v27, v12, v13);
  // pto: %partials_inline342__iter_v1_view
  pto::Stride<-1, -1, -1, -1, -1> v31 = pto::Stride<-1, -1, -1, -1, -1>(v26 * v29, v29, v28, v13, v14);
  // pto: %partials_inline342__iter_v1_view
  GlobalTensor<int32_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v32 = GlobalTensor<int32_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v1, v30, v31);
  // pto: %o_r_i8_pad_inline319__rv_v7_view
  const int64_t v33 = 1;
  // pto: %o_r_i8_pad_inline319__rv_v7_view
  const int64_t v34 = 1;
  // pto: %o_r_i8_pad_inline319__rv_v7_view
  const int64_t v35 = 1;
  // pto: %o_r_i8_pad_inline319__rv_v7_view
  int64_t v36 = v12 * v15;
  // pto: %o_r_i8_pad_inline319__rv_v7_view
  int64_t v37 = v35 * v36;
  // pto: %o_r_i8_pad_inline319__rv_v7_view
  pto::Shape<1, 1, 1, -1, -1> v38 = pto::Shape<1, 1, 1, -1, -1>(v33, v34, v35, v12, v15);
  // pto: %o_r_i8_pad_inline319__rv_v7_view
  pto::Stride<-1, -1, -1, -1, -1> v39 = pto::Stride<-1, -1, -1, -1, -1>(v34 * v37, v37, v36, v15, v14);
  // pto: %o_r_i8_pad_inline319__rv_v7_view
  GlobalTensor<int8_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v40 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v2, v38, v39);
  // pto: %wo_b__ssa_v0_view
  const int64_t v41 = 1;
  // pto: %wo_b__ssa_v0_view
  const int64_t v42 = 1;
  // pto: %wo_b__ssa_v0_view
  const int64_t v43 = 1;
  // pto: %wo_b__ssa_v0_view
  int64_t v44 = v16 * v15;
  // pto: %wo_b__ssa_v0_view
  int64_t v45 = v43 * v44;
  // pto: %wo_b__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v46 = pto::Shape<1, 1, 1, -1, -1>(v41, v42, v43, v16, v15);
  // pto: %wo_b__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v47 = pto::Stride<-1, -1, -1, -1, -1>(v42 * v45, v45, v44, v15, v14);
  // pto: %wo_b__ssa_v0_view
  GlobalTensor<int8_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v48 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v3, v46, v47);
  // pto: %pb_unit_inline305__ssa_v0
  int64_t v49 = (int64_t) v6;
  // pto: %7
  int64_t v50 = v49 / v17;
  // pto: %10
  int64_t v51 = (int64_t) ((uint64_t) v50 * (uint64_t) v18);
  set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  for (int64_t v52 = v20; v52 < v21; v52 += v14) {
    // pto: %9, %8, %11, %13, %12
    ;
    int64_t v53 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) ((int64_t) (uint64_t) v49 - (uint64_t) ((int64_t) (uint64_t) v50 * (uint64_t) v17)) * (uint64_t) v19) + (uint64_t) ((int64_t) (uint64_t) v52 * (uint64_t) v22));
    // pto: %acc_b_inline283__tile
    ;
    Tile<TileType::Acc, int32_t, 128, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null> v54 = Tile<TileType::Acc, int32_t, 128, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null>(v18, v22);
    // pto: %acc_b_inline283__tile
    ;
    uint64_t v55 = (uint64_t) v8;
    TASSIGN(v54, v55);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    for (int64_t v56 = v20; v56 < v23; v56 += v21) {
      // pto: %14
      ;
      int64_t v57 = (int64_t) ((uint64_t) v56 * (uint64_t) v22);
      // pto: %15
      ;
      int64_t v58 = (int64_t) ((uint64_t) v4 + (uint64_t) v57);
      // pto: %18, %17
      ;
      int64_t v59 = (int64_t) ((uint64_t) v4 + (uint64_t) ((int64_t) (uint64_t) v57 + (uint64_t) v22));
      // pto: %b_act_inline281__tile
      ;
      Tile<TileType::Mat, int8_t, 128, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v60 = Tile<TileType::Mat, int8_t, 128, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v18, v22);
      // pto: %b_act_inline281__tile
      ;
      uint64_t v61 = (uint64_t) v8;
      TASSIGN(v60, v61);
      // pto: %19
      ;
      int64_t v62 = v51 < v20 ? v20 : v51;
      // pto: %20
      ;
      int64_t v63 = v58 < v20 ? v20 : v58;
      // pto: %o_r_i8_pad_inline319__rv_v7_pview
      ;
      __gm__ int8_t* v64 = PTOAS__GLOBAL_TENSOR_DATA(v40);
      // pto: %o_r_i8_pad_inline319__rv_v7_pview
      ;
      const int64_t v65 = 0;
      // pto: %o_r_i8_pad_inline319__rv_v7_pview
      ;
      const int64_t v66 = 8192;
      // pto: %o_r_i8_pad_inline319__rv_v7_pview
      ;
      pto::Shape<1, 1, 1, 128, 256> v67 = pto::Shape<1, 1, 1, 128, 256>();
      // pto: %o_r_i8_pad_inline319__rv_v7_pview
      ;
      pto::Stride<1048576, 1048576, 1048576, 8192, 1> v68 = pto::Stride<1048576, 1048576, 1048576, 8192, 1>();
      // pto: %o_r_i8_pad_inline319__rv_v7_pview
      ;
      GlobalTensor<int8_t, pto::Shape<1, 1, 1, 128, 256>, pto::Stride<1048576, 1048576, 1048576, 8192, 1>, pto::Layout::ND> v69 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 128, 256>, pto::Stride<1048576, 1048576, 1048576, 8192, 1>, pto::Layout::ND>(v64 + (v65 + v62 * v66 + v63), v67, v68);
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
      TLOAD(v60, v69);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      // pto: %b_weight_inline280__tile
      ;
      Tile<TileType::Mat, int8_t, 256, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v70 = Tile<TileType::Mat, int8_t, 256, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v22, v22);
      // pto: %b_weight_inline280__tile
      ;
      uint64_t v71 = (uint64_t) v9;
      TASSIGN(v70, v71);
      // pto: %21
      ;
      int64_t v72 = v53 < v20 ? v20 : v53;
      // pto: %wo_b__ssa_v0_pview
      ;
      __gm__ int8_t* v73 = PTOAS__GLOBAL_TENSOR_DATA(v48);
      // pto: %wo_b__ssa_v0_pview
      ;
      const int64_t v74 = 0;
      // pto: %wo_b__ssa_v0_pview
      ;
      const int64_t v75 = 8192;
      // pto: %wo_b__ssa_v0_pview
      ;
      pto::Shape<1, 1, 1, 256, 256> v76 = pto::Shape<1, 1, 1, 256, 256>();
      // pto: %wo_b__ssa_v0_pview
      ;
      pto::Stride<2097152, 2097152, 2097152, 8192, 1> v77 = pto::Stride<2097152, 2097152, 2097152, 8192, 1>();
      // pto: %wo_b__ssa_v0_pview
      ;
      GlobalTensor<int8_t, pto::Shape<1, 1, 1, 256, 256>, pto::Stride<2097152, 2097152, 2097152, 8192, 1>, pto::Layout::ND> v78 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 256, 256>, pto::Stride<2097152, 2097152, 2097152, 8192, 1>, pto::Layout::ND>(v73 + (v74 + v72 * v75 + v63), v76, v77);
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
      TLOAD(v70, v78);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
      // pto: %0
      ;
      Tile<TileType::Mat, int8_t, 128, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v79 = Tile<TileType::Mat, int8_t, 128, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v18, v22);
      // pto: %0
      ;
      uint64_t v80 = (uint64_t) v10;
      TASSIGN(v79, v80);
      // pto: %24
      ;
      int64_t v81 = v59 < v20 ? v20 : v59;
      // pto: %25
      ;
      __gm__ int8_t* v82 = PTOAS__GLOBAL_TENSOR_DATA(v40);
      // pto: %25
      ;
      const int64_t v83 = 0;
      // pto: %25
      ;
      const int64_t v84 = 8192;
      // pto: %25
      ;
      pto::Shape<1, 1, 1, 128, 256> v85 = pto::Shape<1, 1, 1, 128, 256>();
      // pto: %25
      ;
      pto::Stride<1048576, 1048576, 1048576, 8192, 1> v86 = pto::Stride<1048576, 1048576, 1048576, 8192, 1>();
      // pto: %25
      ;
      GlobalTensor<int8_t, pto::Shape<1, 1, 1, 128, 256>, pto::Stride<1048576, 1048576, 1048576, 8192, 1>, pto::Layout::ND> v87 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 128, 256>, pto::Stride<1048576, 1048576, 1048576, 8192, 1>, pto::Layout::ND>(v82 + (v83 + v62 * v84 + v81), v85, v86);
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
      TLOAD(v79, v87);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID2);
      // pto: %1
      ;
      Tile<TileType::Mat, int8_t, 256, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v88 = Tile<TileType::Mat, int8_t, 256, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v22, v22);
      // pto: %1
      ;
      uint64_t v89 = (uint64_t) v11;
      TASSIGN(v88, v89);
      // pto: %28
      ;
      __gm__ int8_t* v90 = PTOAS__GLOBAL_TENSOR_DATA(v48);
      // pto: %28
      ;
      const int64_t v91 = 0;
      // pto: %28
      ;
      const int64_t v92 = 8192;
      // pto: %28
      ;
      pto::Shape<1, 1, 1, 256, 256> v93 = pto::Shape<1, 1, 1, 256, 256>();
      // pto: %28
      ;
      pto::Stride<2097152, 2097152, 2097152, 8192, 1> v94 = pto::Stride<2097152, 2097152, 2097152, 8192, 1>();
      // pto: %28
      ;
      GlobalTensor<int8_t, pto::Shape<1, 1, 1, 256, 256>, pto::Stride<2097152, 2097152, 2097152, 8192, 1>, pto::Layout::ND> v95 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 256, 256>, pto::Stride<2097152, 2097152, 2097152, 8192, 1>, pto::Layout::ND>(v90 + (v91 + v72 * v92 + v81), v93, v94);
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
      TLOAD(v88, v95);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID3);
      // pto: %b_weight_inline280__tile_t
      ;
      Tile<TileType::Mat, int8_t, 256, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v96 = Tile<TileType::Mat, int8_t, 256, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v22, v22);
      // pto: %b_weight_inline280__tile_t
      ;
      uint64_t v97 = (uint64_t) v9;
      TASSIGN(v96, v97);
      // pto: %b_act_inline281__tile_Left
      ;
      Tile<TileType::Left, int8_t, 128, 256, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v98 = Tile<TileType::Left, int8_t, 128, 256, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v18, v22);
      // pto: %b_act_inline281__tile_Left
      ;
      uint64_t v99 = (uint64_t) v9;
      TASSIGN(v98, v99);
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      TMOV(v98, v60);
      set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
      // pto: %b_weight_inline280__tile_t_Right
      ;
      Tile<TileType::Right, int8_t, 256, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v100 = Tile<TileType::Right, int8_t, 256, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v22, v22);
      // pto: %b_weight_inline280__tile_t_Right
      ;
      uint64_t v101 = (uint64_t) v8;
      TASSIGN(v100, v101);
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
      wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
      pipe_barrier(PIPE_MTE1);
      TMOV(v100, v96);
      set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
      // pto: %29
      ;
      wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      if (v56 == v20) {
        TMATMUL(v54, v98, v100);
      } else {
        TMATMUL_ACC(v54, v54, v98, v100);
      };
      set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
      // pto: %3
      ;
      Tile<TileType::Mat, int8_t, 256, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v102 = Tile<TileType::Mat, int8_t, 256, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v22, v22);
      // pto: %3
      ;
      uint64_t v103 = (uint64_t) v11;
      TASSIGN(v102, v103);
      // pto: %4
      ;
      Tile<TileType::Left, int8_t, 128, 256, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v104 = Tile<TileType::Left, int8_t, 128, 256, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v18, v22);
      // pto: %4
      ;
      uint64_t v105 = (uint64_t) v8;
      TASSIGN(v104, v105);
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID2);
      TMOV(v104, v79);
      set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
      // pto: %5
      ;
      Tile<TileType::Right, int8_t, 256, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v106 = Tile<TileType::Right, int8_t, 256, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v22, v22);
      // pto: %5
      ;
      uint64_t v107 = (uint64_t) v8;
      TASSIGN(v106, v107);
      wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
      pipe_barrier(PIPE_MTE1);
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID3);
      TMOV(v106, v102);
      set_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
      set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
      // pto: %30
      ;
      wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
      if (v56 == v24) {
        TMATMUL(v54, v104, v106);
      } else {
        TMATMUL_ACC(v54, v54, v104, v106);
      };
      set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    };
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    // pto: %31
    ;
    int64_t v108 = v51 < v20 ? v20 : v51;
    // pto: %32, %33
    ;
    int64_t v109 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) v5 * (uint64_t) v16) + (uint64_t) v53);
    // pto: %34
    ;
    int64_t v110 = v109 < v20 ? v20 : v109;
    // pto: %partials_inline342__iter_v3_pview
    ;
    __gm__ int32_t* v111 = PTOAS__GLOBAL_TENSOR_DATA(v32);
    // pto: %partials_inline342__iter_v3_pview
    ;
    const int64_t v112 = 0;
    // pto: %partials_inline342__iter_v3_pview
    ;
    const int64_t v113 = 32768;
    // pto: %partials_inline342__iter_v3_pview
    ;
    pto::Shape<1, 1, 1, 128, 256> v114 = pto::Shape<1, 1, 1, 128, 256>();
    // pto: %partials_inline342__iter_v3_pview
    ;
    pto::Stride<4194304, 4194304, 4194304, 32768, 1> v115 = pto::Stride<4194304, 4194304, 4194304, 32768, 1>();
    // pto: %partials_inline342__iter_v3_pview
    ;
    GlobalTensor<int32_t, pto::Shape<1, 1, 1, 128, 256>, pto::Stride<4194304, 4194304, 4194304, 32768, 1>, pto::Layout::ND> v116 = GlobalTensor<int32_t, pto::Shape<1, 1, 1, 128, 256>, pto::Stride<4194304, 4194304, 4194304, 32768, 1>, pto::Layout::ND>(v111 + (v112 + v108 * v113 + v110), v114, v115);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    pipe_barrier(PIPE_FIX);
    TSTORE(v116, v54);
    set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  }
  wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  #endif // __DAV_CUBE__

  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}