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

AICORE void hc_small_aic(__gm__ float* v1, __gm__ float* v2, __gm__ float* v3, __gm__ bfloat16_t* v4, __gm__ float* v5, __gm__ float* v6, __gm__ float* v7, __gm__ float* v8, __gm__ float* v9, __gm__ float* v10, __gm__ float* v11, __gm__ bfloat16_t* v12, __gm__ float* v13, int64_t v14, int32_t v15, int64_t v16, int64_t v17, int64_t v18, int64_t v19, int64_t v20, int64_t v21, int64_t v22, int64_t v23, int64_t v24, int64_t v25, int64_t v26, int64_t v27, int64_t v28, int64_t v29, int64_t v30, int32_t v31, int32_t v32) {
  using T = float;

  #if defined(__DAV_CUBE__)
  const int32_t v33 = 0;
  // pto: %c0_i64
  const int64_t v34 = 0;
  // pto: %c131072_i64
  const int64_t v35 = 131072;
  // pto: %c196608_i64
  const int64_t v36 = 196608;
  // pto: %c32768_i64
  const int64_t v37 = 32768;
  // pto: %c32_index
  const int64_t v38 = 32;
  // pto: %c1_index
  const int64_t v39 = 1;
  // pto: %c24_index
  const int64_t v40 = 24;
  // pto: %c16384_index
  const int64_t v41 = 16384;
  // pto: %c11_index
  const int64_t v42 = 11;
  // pto: %c1536_index
  const int64_t v43 = 1536;
  // pto: %c511_index
  const int64_t v44 = 511;
  const int64_t v45 = 1024;
  // pto: %c2_index
  const int64_t v46 = 2;
  // pto: %c512_index
  const int64_t v47 = 512;
  // pto: %c0_index
  const int64_t v48 = 0;
  // pto: %c256_index
  const int64_t v49 = 256;
  // pto: %cn256_index
  const int64_t v50 = -256;
  // pto: %49
  int64_t v51 = (int64_t) ((uint64_t) v29 * (uint64_t) v42);
  // pto: %mixes_partials_inline131__ssa_v0_view
  const int64_t v52 = 1;
  // pto: %mixes_partials_inline131__ssa_v0_view
  const int64_t v53 = 1;
  // pto: %mixes_partials_inline131__ssa_v0_view
  const int64_t v54 = 1;
  // pto: %mixes_partials_inline131__ssa_v0_view
  int64_t v55 = v51 * v38;
  // pto: %mixes_partials_inline131__ssa_v0_view
  int64_t v56 = v54 * v55;
  // pto: %mixes_partials_inline131__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v57 = pto::Shape<1, 1, 1, -1, -1>(v52, v53, v54, v51, v38);
  // pto: %mixes_partials_inline131__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v58 = pto::Stride<-1, -1, -1, -1, -1>(v53 * v56, v56, v55, v38, v39);
  // pto: %mixes_partials_inline131__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v59 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v1, v57, v58);
  // pto: %hc_fn__ssa_v0_view
  const int64_t v60 = 1;
  // pto: %hc_fn__ssa_v0_view
  const int64_t v61 = 1;
  // pto: %hc_fn__ssa_v0_view
  const int64_t v62 = 1;
  // pto: %hc_fn__ssa_v0_view
  int64_t v63 = v40 * v41;
  // pto: %hc_fn__ssa_v0_view
  int64_t v64 = v62 * v63;
  // pto: %hc_fn__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v65 = pto::Shape<1, 1, 1, -1, -1>(v60, v61, v62, v40, v41);
  // pto: %hc_fn__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v66 = pto::Stride<-1, -1, -1, -1, -1>(v61 * v64, v64, v63, v41, v39);
  // pto: %hc_fn__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v67 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v3, v65, v66);
  // pto: %small_acc_inline240__rv_v2
  Tile<TileType::Acc, float, 32, 32, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null> v68 = Tile<TileType::Acc, float, 32, 32, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null>(v38, v38);
  // pto: %small_acc_inline240__rv_v2
  uint64_t v69 = (uint64_t) v34;
  TASSIGN(v68, v69);
  const int32_t v70 = 0;
  auto v71 = TPipe<0, Direction::DIR_V2C, 65536, 2, 2, false>(v13, v70, v33);
  // pto: %worker_inline104__ssa_v0
  // pto: %linear_workers_inline199__ssa_v0_idx
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  for (int64_t v72 = (int64_t) v31; v72 < v14; v72 += (int64_t) v15) {
    // pto: %21
    ;
    int64_t v73 = v72 % v42;
    // pto: %22
    ;
    int64_t v74 = (int64_t) ((uint64_t) v73 * (uint64_t) v43);
    // pto: %23
    ;
    int64_t v75 = (int64_t) ((uint64_t) v41 - (uint64_t) v74);
    // pto: %24
    ;
    int64_t v76 = v75 < v43 ? v75 : v43;
    // pto: %25, %26, %19, %20
    ;
    int64_t v77 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) v73 * (uint64_t) v16) + (uint64_t) ((int64_t) (uint64_t) (v72 / v42) * (uint64_t) v38));
    // pto: %small_acc_inline240__ssa_v0
    ;
    Tile<TileType::Acc, float, 32, 32, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null> v78 = Tile<TileType::Acc, float, 32, 32, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null>(v38, v38);
    // pto: %small_acc_inline240__ssa_v0
    ;
    uint64_t v79 = (uint64_t) v34;
    TASSIGN(v78, v79);
    // pto: %27
    ;
    int64_t v80 = (int64_t) ((uint64_t) v76 + (uint64_t) v44);
    // pto: %28
    ;
    int64_t v81 = v80 / v45;
    // pto: %30
    ;
    int64_t v82 = (int64_t) ((uint64_t) v74 + (uint64_t) v47);
    // pto: %small_weight_inline163__ssa_v0
    ;
    Tile<TileType::Mat, float, 32, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v83 = Tile<TileType::Mat, float, 32, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v40, v47);
    // pto: %small_weight_inline163__ssa_v0
    ;
    uint64_t v84 = (uint64_t) v35;
    TASSIGN(v83, v84);
    // pto: %31
    ;
    int64_t v85 = v74 < v48 ? v48 : v74;
    // pto: %hc_fn__ssa_v0_pview
    ;
    __gm__ float* v86 = PTOAS__GLOBAL_TENSOR_DATA(v67);
    // pto: %hc_fn__ssa_v0_pview
    ;
    const int64_t v87 = 0;
    // pto: %hc_fn__ssa_v0_pview
    ;
    pto::Shape<1, 1, 1, 24, 512> v88 = pto::Shape<1, 1, 1, 24, 512>();
    // pto: %hc_fn__ssa_v0_pview
    ;
    pto::Stride<393216, 393216, 393216, 16384, 1> v89 = pto::Stride<393216, 393216, 393216, 16384, 1>();
    // pto: %hc_fn__ssa_v0_pview
    ;
    GlobalTensor<float, pto::Shape<1, 1, 1, 24, 512>, pto::Stride<393216, 393216, 393216, 16384, 1>, pto::Layout::ND> v90 = GlobalTensor<float, pto::Shape<1, 1, 1, 24, 512>, pto::Stride<393216, 393216, 393216, 16384, 1>, pto::Layout::ND>(v86 + (v87 + v85), v88, v89);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    TLOAD(v83, v90);
    // pto: %0
    ;
    Tile<TileType::Mat, float, 32, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v91 = Tile<TileType::Mat, float, 32, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v40, v47);
    // pto: %0
    ;
    uint64_t v92 = (uint64_t) v36;
    TASSIGN(v91, v92);
    // pto: %32
    ;
    int64_t v93 = v82 < v48 ? v48 : v82;
    // pto: %33
    ;
    __gm__ float* v94 = PTOAS__GLOBAL_TENSOR_DATA(v67);
    // pto: %33
    ;
    const int64_t v95 = 0;
    // pto: %33
    ;
    pto::Shape<1, 1, 1, 24, 512> v96 = pto::Shape<1, 1, 1, 24, 512>();
    // pto: %33
    ;
    pto::Stride<393216, 393216, 393216, 16384, 1> v97 = pto::Stride<393216, 393216, 393216, 16384, 1>();
    // pto: %33
    ;
    GlobalTensor<float, pto::Shape<1, 1, 1, 24, 512>, pto::Stride<393216, 393216, 393216, 16384, 1>, pto::Layout::ND> v98 = GlobalTensor<float, pto::Shape<1, 1, 1, 24, 512>, pto::Stride<393216, 393216, 393216, 16384, 1>, pto::Layout::ND>(v94 + (v95 + v93), v96, v97);
    TLOAD(v91, v98);
    // pto: %small_weight_t_inline180__ssa_v0
    ;
    Tile<TileType::Mat, float, 512, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v99 = Tile<TileType::Mat, float, 512, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v47, v40);
    // pto: %small_weight_t_inline180__ssa_v0
    ;
    uint64_t v100 = (uint64_t) v35;
    TASSIGN(v99, v100);
    // pto: %small_gathered_inline98__ssa_v0
    ;
    const int32_t v101 = 32;
    // pto: %small_gathered_inline98__ssa_v0
    ;
    const int32_t v102 = 512;
    // pto: %small_gathered_inline98__ssa_v0
    ;
    Tile<TileType::Mat, float, 32, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v103 = Tile<TileType::Mat, float, 32, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v101, v102);
    TPOP<TPipe<0, Direction::DIR_V2C, 65536, 2, 2, false>, Tile<TileType::Mat, float, 32, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>, TileSplitAxis::TILE_UP_DOWN>(v71, v103);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    // pto: %small_left_inline117__ssa_v0
    ;
    Tile<TileType::Left, float, 32, 256, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v104 = Tile<TileType::Left, float, 32, 256, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v38, v49);
    // pto: %small_left_inline117__ssa_v0
    ;
    uint64_t v105 = (uint64_t) v34;
    TASSIGN(v104, v105);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    uint16_t v106 = static_cast<uint16_t>(v48);
    uint16_t v107 = static_cast<uint16_t>(v48);
    TEXTRACT(v104, v103, v106, v107);
    // pto: %small_right_inline121__ssa_v0
    ;
    Tile<TileType::Right, float, 256, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Normal> v108 = Tile<TileType::Right, float, 256, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Normal>(v49, v40);
    // pto: %small_right_inline121__ssa_v0
    ;
    uint64_t v109 = (uint64_t) v34;
    TASSIGN(v108, v109);
    uint16_t v110 = static_cast<uint16_t>(v48);
    uint16_t v111 = static_cast<uint16_t>(v48);
    TEXTRACT(v108, v99, v110, v111);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    // pto: %1
    ;
    Tile<TileType::Left, float, 32, 256, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v112 = Tile<TileType::Left, float, 32, 256, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v38, v49);
    // pto: %1
    ;
    uint64_t v113 = (uint64_t) v37;
    TASSIGN(v112, v113);
    uint16_t v114 = static_cast<uint16_t>(v48);
    uint16_t v115 = static_cast<uint16_t>(v49);
    TEXTRACT(v112, v103, v114, v115);
    // pto: %2
    ;
    Tile<TileType::Right, float, 256, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Normal> v116 = Tile<TileType::Right, float, 256, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Normal>(v49, v40);
    // pto: %2
    ;
    uint64_t v117 = (uint64_t) v37;
    TASSIGN(v116, v117);
    uint16_t v118 = static_cast<uint16_t>(v49);
    uint16_t v119 = static_cast<uint16_t>(v48);
    TEXTRACT(v116, v99, v118, v119);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    TMATMUL(v78, v104, v108);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
    pipe_barrier(PIPE_M);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
    TMATMUL_ACC(v78, v78, v112, v116);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
    TFREE<TPipe<0, Direction::DIR_V2C, 65536, 2, 2, false>, TileSplitAxis::TILE_UP_DOWN>(v71);
    // pto: %4
    ;
    Tile<TileType::Mat, float, 512, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v120 = Tile<TileType::Mat, float, 512, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v47, v40);
    // pto: %4
    ;
    uint64_t v121 = (uint64_t) v36;
    TASSIGN(v120, v121);
    // pto: %34
    ;
    const int32_t v122 = 32;
    // pto: %34
    ;
    const int32_t v123 = 512;
    // pto: %34
    ;
    Tile<TileType::Mat, float, 32, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v124 = Tile<TileType::Mat, float, 32, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v122, v123);
    TPOP<TPipe<0, Direction::DIR_V2C, 65536, 2, 2, false>, Tile<TileType::Mat, float, 32, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>, TileSplitAxis::TILE_UP_DOWN>(v71, v124);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
    // pto: %5
    ;
    Tile<TileType::Left, float, 32, 256, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v125 = Tile<TileType::Left, float, 32, 256, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v38, v49);
    // pto: %5
    ;
    uint64_t v126 = (uint64_t) v34;
    TASSIGN(v125, v126);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
    uint16_t v127 = static_cast<uint16_t>(v48);
    uint16_t v128 = static_cast<uint16_t>(v48);
    TEXTRACT(v125, v124, v127, v128);
    // pto: %6
    ;
    Tile<TileType::Right, float, 256, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Normal> v129 = Tile<TileType::Right, float, 256, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Normal>(v49, v40);
    // pto: %6
    ;
    uint64_t v130 = (uint64_t) v34;
    TASSIGN(v129, v130);
    uint16_t v131 = static_cast<uint16_t>(v48);
    uint16_t v132 = static_cast<uint16_t>(v48);
    TEXTRACT(v129, v120, v131, v132);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID2);
    // pto: %7
    ;
    Tile<TileType::Left, float, 32, 256, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v133 = Tile<TileType::Left, float, 32, 256, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v38, v49);
    // pto: %7
    ;
    uint64_t v134 = (uint64_t) v37;
    TASSIGN(v133, v134);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
    uint16_t v135 = static_cast<uint16_t>(v48);
    uint16_t v136 = static_cast<uint16_t>(v49);
    TEXTRACT(v133, v124, v135, v136);
    // pto: %8
    ;
    Tile<TileType::Right, float, 256, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Normal> v137 = Tile<TileType::Right, float, 256, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Normal>(v49, v40);
    // pto: %8
    ;
    uint64_t v138 = (uint64_t) v37;
    TASSIGN(v137, v138);
    uint16_t v139 = static_cast<uint16_t>(v49);
    uint16_t v140 = static_cast<uint16_t>(v48);
    TEXTRACT(v137, v120, v139, v140);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID3);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID2);
    pipe_barrier(PIPE_M);
    TMATMUL_ACC(v78, v78, v125, v129);
    pipe_barrier(PIPE_M);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID3);
    TMATMUL_ACC(v78, v78, v133, v137);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
    TFREE<TPipe<0, Direction::DIR_V2C, 65536, 2, 2, false>, TileSplitAxis::TILE_UP_DOWN>(v71);
    // pto: %36, %37, %29, %38
    ;
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
    if ((int64_t) ((uint64_t) (v80 / v47) - (uint64_t) ((int64_t) (uint64_t) v81 * (uint64_t) v46)) == v39) {
      int64_t v141 = (int64_t) ((uint64_t) v81 * (uint64_t) v45);
      // pto: %40
      ;
      int64_t v142 = (int64_t) ((uint64_t) v74 + (uint64_t) v141);
      // pto: %11
      ;
      Tile<TileType::Mat, float, 32, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v143 = Tile<TileType::Mat, float, 32, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v40, v47);
      // pto: %11
      ;
      uint64_t v144 = (uint64_t) v35;
      TASSIGN(v143, v144);
      // pto: %41
      ;
      int64_t v145 = v142 < v48 ? v48 : v142;
      // pto: %42
      ;
      __gm__ float* v146 = PTOAS__GLOBAL_TENSOR_DATA(v67);
      // pto: %42
      ;
      const int64_t v147 = 0;
      // pto: %42
      ;
      pto::Shape<1, 1, 1, 24, 512> v148 = pto::Shape<1, 1, 1, 24, 512>();
      // pto: %42
      ;
      pto::Stride<393216, 393216, 393216, 16384, 1> v149 = pto::Stride<393216, 393216, 393216, 16384, 1>();
      // pto: %42
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 24, 512>, pto::Stride<393216, 393216, 393216, 16384, 1>, pto::Layout::ND> v150 = GlobalTensor<float, pto::Shape<1, 1, 1, 24, 512>, pto::Stride<393216, 393216, 393216, 16384, 1>, pto::Layout::ND>(v146 + (v147 + v145), v148, v149);
      TLOAD(v143, v150);
      // pto: %12
      ;
      Tile<TileType::Mat, float, 512, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v151 = Tile<TileType::Mat, float, 512, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v47, v40);
      // pto: %12
      ;
      uint64_t v152 = (uint64_t) v35;
      TASSIGN(v151, v152);
      // pto: %43
      ;
      const int32_t v153 = 32;
      // pto: %43
      ;
      const int32_t v154 = 512;
      // pto: %43
      ;
      Tile<TileType::Mat, float, 32, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v155 = Tile<TileType::Mat, float, 32, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v153, v154);
      TPOP<TPipe<0, Direction::DIR_V2C, 65536, 2, 2, false>, Tile<TileType::Mat, float, 32, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>, TileSplitAxis::TILE_UP_DOWN>(v71, v155);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID2);
      // pto: %13
      ;
      Tile<TileType::Left, float, 32, 256, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v156 = Tile<TileType::Left, float, 32, 256, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v38, v49);
      // pto: %13
      ;
      uint64_t v157 = (uint64_t) v34;
      TASSIGN(v156, v157);
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID2);
      uint16_t v158 = static_cast<uint16_t>(v48);
      uint16_t v159 = static_cast<uint16_t>(v48);
      TEXTRACT(v156, v155, v158, v159);
      // pto: %14
      ;
      Tile<TileType::Right, float, 256, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Normal> v160 = Tile<TileType::Right, float, 256, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Normal>(v49, v40);
      // pto: %14
      ;
      uint64_t v161 = (uint64_t) v34;
      TASSIGN(v160, v161);
      uint16_t v162 = static_cast<uint16_t>(v48);
      uint16_t v163 = static_cast<uint16_t>(v48);
      TEXTRACT(v160, v151, v162, v163);
      set_flag(PIPE_MTE1, PIPE_M, EVENT_ID4);
      // pto: %15
      ;
      Tile<TileType::Left, float, 32, 256, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v164 = Tile<TileType::Left, float, 32, 256, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v38, v49);
      // pto: %15
      ;
      uint64_t v165 = (uint64_t) v37;
      TASSIGN(v164, v165);
      uint16_t v166 = static_cast<uint16_t>(v48);
      uint16_t v167 = static_cast<uint16_t>(v49);
      TEXTRACT(v164, v155, v166, v167);
      // pto: %16
      ;
      Tile<TileType::Right, float, 256, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Normal> v168 = Tile<TileType::Right, float, 256, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Normal>(v49, v40);
      // pto: %16
      ;
      uint64_t v169 = (uint64_t) v37;
      TASSIGN(v168, v169);
      uint16_t v170 = static_cast<uint16_t>(v49);
      uint16_t v171 = static_cast<uint16_t>(v48);
      TEXTRACT(v168, v151, v170, v171);
      set_flag(PIPE_MTE1, PIPE_M, EVENT_ID5);
      // pto: %45
      ;
      wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID4);
      if (v141 == v48) {
        pipe_barrier(PIPE_M);
        TMATMUL(v78, v156, v160);
      } else {
        pipe_barrier(PIPE_M);
        TMATMUL_ACC(v78, v78, v156, v160);
      };
      // pto: %47
      ;
      wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID5);
      if (v141 == v50) {
        pipe_barrier(PIPE_M);
        TMATMUL(v78, v164, v168);
      } else {
        pipe_barrier(PIPE_M);
        TMATMUL_ACC(v78, v78, v164, v168);
      };
      TFREE<TPipe<0, Direction::DIR_V2C, 65536, 2, 2, false>, TileSplitAxis::TILE_UP_DOWN>(v71);
    };
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    // pto: %48
    ;
    int64_t v172 = v77 < v48 ? v48 : v77;
    // pto: %mixes_partials_inline131__iter_v1_pview
    ;
    __gm__ float* v173 = PTOAS__GLOBAL_TENSOR_DATA(v59);
    // pto: %mixes_partials_inline131__iter_v1_pview
    ;
    const int64_t v174 = 0;
    // pto: %mixes_partials_inline131__iter_v1_pview
    ;
    const int64_t v175 = 32;
    // pto: %mixes_partials_inline131__iter_v1_pview
    ;
    pto::Shape<1, 1, 1, 32, 32> v176 = pto::Shape<1, 1, 1, 32, 32>();
    // pto: %mixes_partials_inline131__iter_v1_pview
    ;
    pto::Stride<1024, 1024, 1024, 32, 1> v177 = pto::Stride<1024, 1024, 1024, 32, 1>();
    // pto: %mixes_partials_inline131__iter_v1_pview
    ;
    GlobalTensor<float, pto::Shape<1, 1, 1, 32, 32>, pto::Stride<1024, 1024, 1024, 32, 1>, pto::Layout::ND> v178 = GlobalTensor<float, pto::Shape<1, 1, 1, 32, 32>, pto::Stride<1024, 1024, 1024, 32, 1>, pto::Layout::ND>(v173 + (v174 + v172 * v175), v176, v177);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    TSTORE(v178, v68);
    set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  }
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  dcci((__gm__ void*)0, cache_line_t::ENTIRE_DATA_CACHE);
  pipe_barrier(PIPE_MTE2);
  pipe_barrier(PIPE_MTE3);
  pipe_barrier(PIPE_FIX);
  dsb(DSB_DDR);
  SYNCALL<SyncCoreType::Mix>();
  dcci((__gm__ void*)0, cache_line_t::ENTIRE_DATA_CACHE);
  #endif // __DAV_CUBE__

  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}

AICORE void hc_small_aiv(__gm__ float* v1, __gm__ float* v2, __gm__ float* v3, __gm__ bfloat16_t* v4, __gm__ float* v5, __gm__ float* v6, __gm__ float* v7, __gm__ float* v8, __gm__ float* v9, __gm__ float* v10, __gm__ float* v11, __gm__ bfloat16_t* v12, __gm__ float* v13, int64_t v14, int32_t v15, int64_t v16, int64_t v17, int64_t v18, int64_t v19, int64_t v20, int64_t v21, int64_t v22, int64_t v23, int64_t v24, int64_t v25, int64_t v26, int64_t v27, int64_t v28, int64_t v29, int64_t v30, int32_t v31, int32_t v32, int32_t v33) {
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  const int64_t v34 = 12800;
  const int64_t v35 = 8704;
  const int64_t v36 = 4608;
  // pto: %c32896_i64
  const int64_t v37 = 32896;
  // pto: %c41088_i64
  const int64_t v38 = 41088;
  // pto: %c114816_i64
  const int64_t v39 = 114816;
  // pto: %c0_i64
  const int64_t v40 = 0;
  // pto: %c32960_i64
  const int64_t v41 = 32960;
  // pto: %c82048_i64
  const int64_t v42 = 82048;
  // pto: %c33024_i64
  const int64_t v43 = 33024;
  // pto: %c11264_i64
  const int64_t v44 = 11264;
  // pto: %c1024_i64
  const int64_t v45 = 1024;
  // pto: %c1376_i64
  const int64_t v46 = 1376;
  // pto: %c1728_i64
  const int64_t v47 = 1728;
  const int64_t v48 = 1088;
  const int64_t v49 = 32;
  // pto: %c512_i64
  const int64_t v50 = 512;
  // pto: %c768_i64
  const int64_t v51 = 768;
  // pto: %c1280_i64
  const int64_t v52 = 1280;
  // pto: %c128_i64
  const int64_t v53 = 128;
  // pto: %c256_i64
  const int64_t v54 = 256;
  // pto: %c384_i64
  const int64_t v55 = 384;
  // pto: %c640_i64
  const int64_t v56 = 640;
  // pto: %c1792_i64
  const int64_t v57 = 1792;
  // pto: %c1824_i64
  const int64_t v58 = 1824;
  // pto: %c1856_i64
  const int64_t v59 = 1856;
  // pto: %c1888_i64
  const int64_t v60 = 1888;
  const int64_t v61 = 1056;
  // pto: %c32768_i64
  const int64_t v62 = 32768;
  // pto: %c32800_i64
  const int64_t v63 = 32800;
  // pto: %c32832_i64
  const int64_t v64 = 32832;
  const int64_t v65 = 1120;
  // pto: %c32864_i64
  const int64_t v66 = 32864;
  // pto: %c16384_i64
  const int64_t v67 = 16384;
  // pto: %c57472_i64
  const int64_t v68 = 57472;
  // pto: %c98432_i64
  const int64_t v69 = 98432;
  // pto: %c131200_i64
  const int64_t v70 = 131200;
  // pto: %c73856_i64
  const int64_t v71 = 73856;
  // pto: %c32_index
  const int64_t v72 = 32;
  // pto: %c1_index
  const int64_t v73 = 1;
  // pto: %c24_index
  const int64_t v74 = 24;
  // pto: %c16384_index
  const int64_t v75 = 16384;
  // pto: %c3_index
  const int64_t v76 = 3;
  // pto: %c11_index
  const int64_t v77 = 11;
  // pto: %c16_index
  const int64_t v78 = 16;
  // pto: %c4096_index
  const int64_t v79 = 4096;
  // pto: %c0_i32
  const int32_t v80 = 0;
  // pto: %c1536_index
  const int64_t v81 = 1536;
  // pto: %cst_45
  const float v82 = 0.0f;
  // pto: %c511_index
  const int64_t v83 = 511;
  const int64_t v84 = 1024;
  // pto: %c2_index
  const int64_t v85 = 2;
  // pto: %c0_index
  const int64_t v86 = 0;
  // pto: %c512_index
  const int64_t v87 = 512;
  // pto: %c256_index
  const int64_t v88 = 256;
  // pto: %c8_index
  const int64_t v89 = 8;
  // pto: %cst_53
  const float v90 = 6.10351563E-5f;
  // pto: %cst_54
  const float v91 = 9.99999997E-7f;
  // pto: %c192_index
  const int64_t v92 = 192;
  // pto: %cst_56
  const float v93 = 1.0f;
  // pto: %cst_57
  const float v94 = 0.5f;
  // pto: %cst_58
  const float v95 = 2.0f;
  // pto: %cst_59
  const float v96 = 0.25f;
  // pto: %cst_60
  const float v97 = 4.0f;
  // pto: %cst_61
  const float v98 = 0.125f;
  // pto: %cst_62
  const float v99 = 8.0f;
  // pto: %cst_63
  const float v100 = 0.0625f;
  // pto: %cst_64
  const float v101 = 16.0f;
  // pto: %c4_index
  const int64_t v102 = 4;
  // pto: %c5_index
  const int64_t v103 = 5;
  // pto: %c6_index
  const int64_t v104 = 6;
  // pto: %c7_index
  const int64_t v105 = 7;
  // pto: %c9_index
  const int64_t v106 = 9;
  // pto: %c10_index
  const int64_t v107 = 10;
  // pto: %c12_index
  const int64_t v108 = 12;
  // pto: %c13_index
  const int64_t v109 = 13;
  // pto: %c14_index
  const int64_t v110 = 14;
  // pto: %c15_index
  const int64_t v111 = 15;
  // pto: %c19_index
  const int64_t v112 = 19;
  // pto: %c64_index
  const int64_t v113 = 64;
  // pto: %c8192_index
  const int64_t v114 = 8192;
  // pto: %c12288_index
  const int64_t v115 = 12288;
  // pto: %455
  int64_t v116 = (int64_t) ((uint64_t) v29 * (uint64_t) v77);
  // pto: %rms_partials_inline139__ssa_v0_view
  const int64_t v117 = 1;
  // pto: %rms_partials_inline139__ssa_v0_view
  const int64_t v118 = 1;
  // pto: %rms_partials_inline139__ssa_v0_view
  const int64_t v119 = 1;
  // pto: %rms_partials_inline139__ssa_v0_view
  int64_t v120 = v116 * v73;
  // pto: %rms_partials_inline139__ssa_v0_view
  int64_t v121 = v119 * v120;
  // pto: %rms_partials_inline139__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v122 = pto::Shape<1, 1, 1, -1, -1>(v117, v118, v119, v116, v73);
  // pto: %rms_partials_inline139__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v123 = pto::Stride<-1, -1, -1, -1, -1>(v118 * v121, v121, v120, v73, v116);
  // pto: %rms_partials_inline139__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN> v124 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN>(v2, v122, v123);
  // pto: %x_flat_inline108__ssa_v0_view
  const int64_t v125 = 1;
  // pto: %x_flat_inline108__ssa_v0_view
  const int64_t v126 = 1;
  // pto: %x_flat_inline108__ssa_v0_view
  const int64_t v127 = 1;
  // pto: %x_flat_inline108__ssa_v0_view
  int64_t v128 = (int64_t) v30;
  // pto: %x_flat_inline108__ssa_v0_view
  int64_t v129 = v128 * v75;
  // pto: %x_flat_inline108__ssa_v0_view
  int64_t v130 = v127 * v129;
  // pto: %x_flat_inline108__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v131 = pto::Shape<1, 1, 1, -1, -1>(v125, v126, v127, v128, v75);
  // pto: %x_flat_inline108__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v132 = pto::Stride<-1, -1, -1, -1, -1>(v126 * v130, v130, v129, v75, v73);
  // pto: %x_flat_inline108__ssa_v0_view
  GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v133 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v4, v131, v132);
  // pto: %456
  int64_t v134 = (int64_t) ((uint64_t) v29 * (uint64_t) v72);
  // pto: %parts_2d_inline94__ssa_v0_view
  const int64_t v135 = 1;
  // pto: %parts_2d_inline94__ssa_v0_view
  const int64_t v136 = 1;
  // pto: %parts_2d_inline94__ssa_v0_view
  const int64_t v137 = 1;
  // pto: %parts_2d_inline94__ssa_v0_view
  int64_t v138 = v77 * v134;
  // pto: %parts_2d_inline94__ssa_v0_view
  int64_t v139 = v137 * v138;
  // pto: %parts_2d_inline94__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v140 = pto::Shape<1, 1, 1, -1, -1>(v135, v136, v137, v77, v134);
  // pto: %parts_2d_inline94__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v141 = pto::Stride<-1, -1, -1, -1, -1>(v136 * v139, v139, v138, v134, v73);
  // pto: %parts_2d_inline94__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v142 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v6, v140, v141);
  // pto: %rms_2d_inline107__ssa_v0_view
  const int64_t v143 = 1;
  // pto: %rms_2d_inline107__ssa_v0_view
  const int64_t v144 = 1;
  // pto: %rms_2d_inline107__ssa_v0_view
  const int64_t v145 = 1;
  // pto: %rms_2d_inline107__ssa_v0_view
  int64_t v146 = (int64_t) v29;
  // pto: %rms_2d_inline107__ssa_v0_view
  int64_t v147 = v77 * v146;
  // pto: %rms_2d_inline107__ssa_v0_view
  int64_t v148 = v145 * v147;
  // pto: %rms_2d_inline107__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v149 = pto::Shape<1, 1, 1, -1, -1>(v143, v144, v145, v77, (int64_t) v29);
  // pto: %rms_2d_inline107__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v150 = pto::Stride<-1, -1, -1, -1, -1>(v144 * v148, v148, v147, v146, v73);
  // pto: %rms_2d_inline107__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v151 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v7, v149, v150);
  // pto: %hc_base_2d_inline203__ssa_v0_view
  const int64_t v152 = 1;
  // pto: %hc_base_2d_inline203__ssa_v0_view
  const int64_t v153 = 1;
  // pto: %hc_base_2d_inline203__ssa_v0_view
  const int64_t v154 = 1;
  // pto: %hc_base_2d_inline203__ssa_v0_view
  int64_t v155 = v73 * v74;
  // pto: %hc_base_2d_inline203__ssa_v0_view
  int64_t v156 = v154 * v155;
  // pto: %hc_base_2d_inline203__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v157 = pto::Shape<1, 1, 1, -1, -1>(v152, v153, v154, v73, v74);
  // pto: %hc_base_2d_inline203__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v158 = pto::Stride<-1, -1, -1, -1, -1>(v153 * v156, v156, v155, v74, v73);
  // pto: %hc_base_2d_inline203__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v159 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v8, v157, v158);
  // pto: %comb_inline133__ssa_v0_view
  const int64_t v160 = 1;
  // pto: %comb_inline133__ssa_v0_view
  const int64_t v161 = 1;
  // pto: %comb_inline133__ssa_v0_view
  const int64_t v162 = 1;
  // pto: %comb_inline133__ssa_v0_view
  int64_t v163 = (int64_t) v30;
  // pto: %comb_inline133__ssa_v0_view
  int64_t v164 = v163 * v78;
  // pto: %comb_inline133__ssa_v0_view
  int64_t v165 = v162 * v164;
  // pto: %comb_inline133__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v166 = pto::Shape<1, 1, 1, -1, -1>(v160, v161, v162, v163, v78);
  // pto: %comb_inline133__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v167 = pto::Stride<-1, -1, -1, -1, -1>(v161 * v165, v165, v164, v78, v73);
  // pto: %comb_inline133__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v168 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v9, v166, v167);
  // pto: %hc_scale_2d_inline304__ssa_v0_view
  const int64_t v169 = 1;
  // pto: %hc_scale_2d_inline304__ssa_v0_view
  const int64_t v170 = 1;
  // pto: %hc_scale_2d_inline304__ssa_v0_view
  const int64_t v171 = 1;
  // pto: %hc_scale_2d_inline304__ssa_v0_view
  int64_t v172 = v73 * v76;
  // pto: %hc_scale_2d_inline304__ssa_v0_view
  int64_t v173 = v171 * v172;
  // pto: %hc_scale_2d_inline304__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v174 = pto::Shape<1, 1, 1, -1, -1>(v169, v170, v171, v73, v76);
  // pto: %hc_scale_2d_inline304__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v175 = pto::Stride<-1, -1, -1, -1, -1>(v170 * v173, v173, v172, v76, v73);
  // pto: %hc_scale_2d_inline304__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v176 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v10, v174, v175);
  // pto: %457
  int64_t v177 = (int64_t) ((uint64_t) v30 * (uint64_t) v102);
  // pto: %post_flat_inline84__ssa_v0_view
  const int64_t v178 = 1;
  // pto: %post_flat_inline84__ssa_v0_view
  const int64_t v179 = 1;
  // pto: %post_flat_inline84__ssa_v0_view
  const int64_t v180 = 1;
  // pto: %post_flat_inline84__ssa_v0_view
  int64_t v181 = v73 * v177;
  // pto: %post_flat_inline84__ssa_v0_view
  int64_t v182 = v180 * v181;
  // pto: %post_flat_inline84__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v183 = pto::Shape<1, 1, 1, -1, -1>(v178, v179, v180, v73, v177);
  // pto: %post_flat_inline84__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v184 = pto::Stride<-1, -1, -1, -1, -1>(v179 * v182, v182, v181, v177, v73);
  // pto: %post_flat_inline84__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v185 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v11, v183, v184);
  // pto: %mixed_view_inline92__ssa_v0_view
  const int64_t v186 = 1;
  // pto: %mixed_view_inline92__ssa_v0_view
  const int64_t v187 = 1;
  // pto: %mixed_view_inline92__ssa_v0_view
  const int64_t v188 = 1;
  // pto: %mixed_view_inline92__ssa_v0_view
  int64_t v189 = (int64_t) v30;
  // pto: %mixed_view_inline92__ssa_v0_view
  int64_t v190 = v189 * v79;
  // pto: %mixed_view_inline92__ssa_v0_view
  int64_t v191 = v188 * v190;
  // pto: %mixed_view_inline92__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v192 = pto::Shape<1, 1, 1, -1, -1>(v186, v187, v188, v189, v79);
  // pto: %mixed_view_inline92__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v193 = pto::Stride<-1, -1, -1, -1, -1>(v187 * v191, v191, v190, v79, v73);
  // pto: %mixed_view_inline92__ssa_v0_view
  GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v194 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v12, v192, v193);
  // pto: %small_rms_total_inline170__rv_v2
  Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v195 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v78);
  // pto: %small_rms_total_inline170__rv_v2
  uint64_t v196 = (uint64_t) v40;
  TASSIGN(v195, v196);
  const int32_t v197 = 0;
  auto v198 = TPipe<0, Direction::DIR_V2C, 65536, 2, 2, false>(v13, v197, v80);
  // pto: %worker_inline104__ssa_v0
  int64_t v199 = (int64_t) v31;
  // pto: %linear_workers_inline199__ssa_v0_idx
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID4);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID5);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID6);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
  for (int64_t v200 = v199; v200 < v14; v200 += (int64_t) v15) {
    // pto: %54, %55
    ;
    int64_t v201 = (int64_t) ((uint64_t) (v200 / v77) * (uint64_t) v72);
    // pto: %56
    ;
    int64_t v202 = v200 % v77;
    // pto: %57
    ;
    int64_t v203 = (int64_t) ((uint64_t) v202 * (uint64_t) v81);
    // pto: %58
    ;
    int64_t v204 = (int64_t) ((uint64_t) v75 - (uint64_t) v203);
    // pto: %59
    ;
    int64_t v205 = v204 < v81 ? v204 : v81;
    // pto: %small_rms_total_inline170__ssa_v0
    ;
    Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v206 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v78);
    // pto: %small_rms_total_inline170__ssa_v0
    ;
    uint64_t v207 = (uint64_t) v37;
    TASSIGN(v206, v207);
    pipe_barrier(PIPE_V);
    TEXPANDS(v206, v82);
    // pto: %62
    ;
    int64_t v208 = (int64_t) ((uint64_t) v205 + (uint64_t) v83);
    // pto: %63
    ;
    int64_t v209 = v208 / v84;
    // pto: %small_lane_inline57__ssa_v0, %65
    ;
    int64_t v210 = (int64_t) ((uint64_t) ((int64_t) v33) * (uint64_t) v78);
    // pto: %66
    ;
    int64_t v211 = (int64_t) ((uint64_t) v201 + (uint64_t) v210);
    // pto: %67
    ;
    int64_t v212 = (int64_t) ((uint64_t) v17 - (uint64_t) v211);
    // pto: %68
    ;
    int64_t v213 = v212 < v78 ? v212 : v78;
    // pto: %69
    ;
    int64_t v214 = v213 < v86 ? v86 : v213;
    // pto: %70
    ;
    int64_t v215 = (int64_t) ((uint64_t) v203 + (uint64_t) v87);
    // pto: %small_input_inline181__ssa_v0
    ;
    Tile<TileType::Vec, bfloat16_t, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v216 = Tile<TileType::Vec, bfloat16_t, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v214, v87);
    // pto: %small_input_inline181__ssa_v0
    ;
    uint64_t v217 = (uint64_t) v38;
    TASSIGN(v216, v217);
    // pto: %77
    ;
    int64_t v218 = v211 < v86 ? v86 : v211;
    // pto: %78
    ;
    int64_t v219 = v203 < v86 ? v86 : v203;
    // pto: %x_flat_inline108__ssa_v0_pview
    ;
    const int64_t v220 = 0;
    // pto: %x_flat_inline108__ssa_v0_pview
    ;
    __gm__ bfloat16_t* v221 = PTOAS__GLOBAL_TENSOR_DATA(v133);
    // pto: %x_flat_inline108__ssa_v0_pview
    ;
    const int64_t v222 = 1;
    // pto: %x_flat_inline108__ssa_v0_pview
    ;
    const int64_t v223 = 1;
    // pto: %x_flat_inline108__ssa_v0_pview
    ;
    const int64_t v224 = 1;
    // pto: %x_flat_inline108__ssa_v0_pview
    ;
    int64_t v225 = v214 * v75;
    // pto: %x_flat_inline108__ssa_v0_pview
    ;
    int64_t v226 = v224 * v225;
    // pto: %x_flat_inline108__ssa_v0_pview
    ;
    pto::Shape<1, 1, 1, -1, 512> v227 = pto::Shape<1, 1, 1, -1, 512>(v222, v223, v224, v214, v87);
    // pto: %x_flat_inline108__ssa_v0_pview
    ;
    pto::Stride<-1, -1, -1, -1, -1> v228 = pto::Stride<-1, -1, -1, -1, -1>(v223 * v226, v226, v225, v75, v73);
    // pto: %x_flat_inline108__ssa_v0_pview
    ;
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v229 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v221 + (v220 + v218 * v75 + v219 * v73), v227, v228);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    TLOAD(v216, v229);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    // pto: %0
    ;
    Tile<TileType::Vec, bfloat16_t, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v230 = Tile<TileType::Vec, bfloat16_t, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v214, v87);
    // pto: %0
    ;
    uint64_t v231 = (uint64_t) v39;
    TASSIGN(v230, v231);
    // pto: %80
    ;
    int64_t v232 = v215 < v86 ? v86 : v215;
    // pto: %81
    ;
    const int64_t v233 = 0;
    // pto: %81
    ;
    __gm__ bfloat16_t* v234 = PTOAS__GLOBAL_TENSOR_DATA(v133);
    // pto: %81
    ;
    const int64_t v235 = 1;
    // pto: %81
    ;
    const int64_t v236 = 1;
    // pto: %81
    ;
    const int64_t v237 = 1;
    // pto: %81
    ;
    int64_t v238 = v214 * v75;
    // pto: %81
    ;
    int64_t v239 = v237 * v238;
    // pto: %81
    ;
    pto::Shape<1, 1, 1, -1, 512> v240 = pto::Shape<1, 1, 1, -1, 512>(v235, v236, v237, v214, v87);
    // pto: %81
    ;
    pto::Stride<-1, -1, -1, -1, -1> v241 = pto::Stride<-1, -1, -1, -1, -1>(v236 * v239, v239, v238, v75, v73);
    // pto: %81
    ;
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v242 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v234 + (v233 + v218 * v75 + v232 * v73), v240, v241);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
    TLOAD(v230, v242);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
    // pto: %small_fp32_inline309__ssa_v0
    ;
    Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v243 = Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v214, v87);
    // pto: %small_fp32_inline309__ssa_v0
    ;
    uint64_t v244 = (uint64_t) v40;
    TASSIGN(v243, v244);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    RoundMode v245 = RoundMode::CAST_ROUND;
    SaturationMode v246 = SaturationMode::ON;
    TCVT(v243, v216, v245, v246);
    v243.SetValidShape(v214, v87);
    // pto: %small_clean_inline102__ssa_v0
    ;
    Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null> v247 = Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>(v78, v87);
    // pto: %small_clean_inline102__ssa_v0
    ;
    uint64_t v248 = (uint64_t) v40;
    TASSIGN(v247, v248);
    pipe_barrier(PIPE_V);
    TFILLPAD<pto::TFillPadMode::InPlace>(v247, v243);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    v247.SetValidShape(v78, v87);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TPUSH<TPipe<0, Direction::DIR_V2C, 65536, 2, 2, false>, Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>, TileSplitAxis::TILE_UP_DOWN>(v198, v247);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
    // pto: %small_rms_tmp_inline130__ssa_v0
    ;
    Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v249 = Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v78, v87);
    // pto: %small_rms_tmp_inline130__ssa_v0
    ;
    uint64_t v250 = (uint64_t) v38;
    TASSIGN(v249, v250);
    // pto: %t__tmp_v1
    ;
    Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null> v251 = Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>(v78, v87);
    // pto: %t__tmp_v1
    ;
    uint64_t v252 = (uint64_t) v40;
    TASSIGN(v251, v252);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
    TMUL(v251, v247, v247);
    // pto: %small_sq_inline197__ssa_v0
    ;
    Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v253 = Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v78, v73);
    // pto: %small_sq_inline197__ssa_v0
    ;
    uint64_t v254 = (uint64_t) v41;
    TASSIGN(v253, v254);
    pipe_barrier(PIPE_V);
    TROWSUM(v253, v251, v249);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
    // pto: %t__tmp_v2
    ;
    Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v255 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v78);
    // pto: %t__tmp_v2
    ;
    uint64_t v256 = (uint64_t) v41;
    TASSIGN(v255, v256);
    // pto: %small_rms_total_inline170__ssa_v3
    ;
    Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v257 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v78);
    // pto: %small_rms_total_inline170__ssa_v3
    ;
    uint64_t v258 = (uint64_t) v40;
    TASSIGN(v257, v258);
    pipe_barrier(PIPE_V);
    TADD(v257, v206, v255);
    // pto: %1
    ;
    Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v259 = Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v214, v87);
    // pto: %1
    ;
    uint64_t v260 = (uint64_t) v42;
    TASSIGN(v259, v260);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
    RoundMode v261 = RoundMode::CAST_ROUND;
    SaturationMode v262 = SaturationMode::ON;
    TCVT(v259, v230, v261, v262);
    v259.SetValidShape(v214, v87);
    // pto: %3
    ;
    Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null> v263 = Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>(v78, v87);
    // pto: %3
    ;
    uint64_t v264 = (uint64_t) v42;
    TASSIGN(v263, v264);
    pipe_barrier(PIPE_V);
    TFILLPAD<pto::TFillPadMode::InPlace>(v263, v259);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
    v263.SetValidShape(v78, v87);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
    TPUSH<TPipe<0, Direction::DIR_V2C, 65536, 2, 2, false>, Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>, TileSplitAxis::TILE_UP_DOWN>(v198, v263);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
    // pto: %5
    ;
    Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v265 = Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v78, v87);
    // pto: %5
    ;
    uint64_t v266 = (uint64_t) v39;
    TASSIGN(v265, v266);
    // pto: %6
    ;
    Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null> v267 = Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>(v78, v87);
    // pto: %6
    ;
    uint64_t v268 = (uint64_t) v42;
    TASSIGN(v267, v268);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
    TMUL(v267, v263, v263);
    // pto: %7
    ;
    Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v269 = Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v78, v73);
    // pto: %7
    ;
    uint64_t v270 = (uint64_t) v43;
    TASSIGN(v269, v270);
    pipe_barrier(PIPE_V);
    TROWSUM(v269, v267, v265);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
    // pto: %8
    ;
    Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v271 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v78);
    // pto: %8
    ;
    uint64_t v272 = (uint64_t) v43;
    TASSIGN(v271, v272);
    // pto: %9
    ;
    Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v273 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v78);
    // pto: %9
    ;
    uint64_t v274 = (uint64_t) v37;
    TASSIGN(v273, v274);
    pipe_barrier(PIPE_V);
    TADD(v273, v257, v271);
    // pto: %83, %84, %64, %85
    ;
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
    if ((int64_t) ((uint64_t) (v208 / v87) - (uint64_t) ((int64_t) (uint64_t) v209 * (uint64_t) v85)) == v73) {
      // pto: %87
      ;
      int64_t v275 = (int64_t) ((uint64_t) v203 + (uint64_t) ((int64_t) (uint64_t) v209 * (uint64_t) v84));
      // pto: %10
      ;
      Tile<TileType::Vec, bfloat16_t, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v276 = Tile<TileType::Vec, bfloat16_t, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v214, v87);
      // pto: %10
      ;
      uint64_t v277 = (uint64_t) v38;
      TASSIGN(v276, v277);
      // pto: %95
      ;
      int64_t v278 = v275 < v86 ? v86 : v275;
      // pto: %96
      ;
      const int64_t v279 = 0;
      // pto: %96
      ;
      __gm__ bfloat16_t* v280 = PTOAS__GLOBAL_TENSOR_DATA(v133);
      // pto: %96
      ;
      const int64_t v281 = 1;
      // pto: %96
      ;
      const int64_t v282 = 1;
      // pto: %96
      ;
      const int64_t v283 = 1;
      // pto: %96
      ;
      int64_t v284 = v214 * v75;
      // pto: %96
      ;
      int64_t v285 = v283 * v284;
      // pto: %96
      ;
      pto::Shape<1, 1, 1, -1, 512> v286 = pto::Shape<1, 1, 1, -1, 512>(v281, v282, v283, v214, v87);
      // pto: %96
      ;
      pto::Stride<-1, -1, -1, -1, -1> v287 = pto::Stride<-1, -1, -1, -1, -1>(v282 * v285, v285, v284, v75, v73);
      // pto: %96
      ;
      GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v288 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v280 + (v279 + v218 * v75 + v278 * v73), v286, v287);
      TLOAD(v276, v288);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
      // pto: %11
      ;
      Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v289 = Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v214, v87);
      // pto: %11
      ;
      uint64_t v290 = (uint64_t) v40;
      TASSIGN(v289, v290);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
      pipe_barrier(PIPE_V);
      RoundMode v291 = RoundMode::CAST_ROUND;
      SaturationMode v292 = SaturationMode::ON;
      TCVT(v289, v276, v291, v292);
      v289.SetValidShape(v214, v87);
      // pto: %13
      ;
      Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null> v293 = Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>(v78, v87);
      // pto: %13
      ;
      uint64_t v294 = (uint64_t) v40;
      TASSIGN(v293, v294);
      pipe_barrier(PIPE_V);
      TFILLPAD<pto::TFillPadMode::InPlace>(v293, v289);
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
      v293.SetValidShape(v78, v87);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
      TPUSH<TPipe<0, Direction::DIR_V2C, 65536, 2, 2, false>, Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>, TileSplitAxis::TILE_UP_DOWN>(v198, v293);
      set_flag(PIPE_MTE3, PIPE_V, EVENT_ID3);
      // pto: %15
      ;
      Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v295 = Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v78, v87);
      // pto: %15
      ;
      uint64_t v296 = (uint64_t) v38;
      TASSIGN(v295, v296);
      // pto: %16
      ;
      Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null> v297 = Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>(v78, v87);
      // pto: %16
      ;
      uint64_t v298 = (uint64_t) v40;
      TASSIGN(v297, v298);
      wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID3);
      TMUL(v297, v293, v293);
      // pto: %17
      ;
      Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v299 = Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v78, v73);
      // pto: %17
      ;
      uint64_t v300 = (uint64_t) v42;
      TASSIGN(v299, v300);
      pipe_barrier(PIPE_V);
      TROWSUM(v299, v297, v295);
      // pto: %18
      ;
      Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v301 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v78);
      // pto: %18
      ;
      uint64_t v302 = (uint64_t) v42;
      TASSIGN(v301, v302);
      // pto: %19
      ;
      Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v303 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v78);
      // pto: %19
      ;
      uint64_t v304 = (uint64_t) v40;
      TASSIGN(v303, v304);
      pipe_barrier(PIPE_V);
      TADD(v303, v273, v301);
    } else {
      // pto: %small_rms_total_inline170__rv_v2_main_mv
      ;
      Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v305 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v78);
      // pto: %small_rms_total_inline170__rv_v2_main_mv
      ;
      uint64_t v306 = (uint64_t) v40;
      TASSIGN(v305, v306);
      pipe_barrier(PIPE_V);
      TMOV(v305, v273);
    };
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID3);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID4);
    // pto: %60, %61, %98
    ;
    int64_t v307 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) ((int64_t) (uint64_t) v202 * (uint64_t) v16) + (uint64_t) v201) + (uint64_t) v210);
    // pto: %t__tmp_v3
    ;
    Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v308 = Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v78, v73);
    // pto: %t__tmp_v3
    ;
    uint64_t v309 = (uint64_t) v40;
    TASSIGN(v308, v309);
    // pto: %99
    ;
    int64_t v310 = v307 < v86 ? v86 : v307;
    // pto: %rms_partials_inline139__iter_v1_pview
    ;
    const int64_t v311 = 0;
    // pto: %rms_partials_inline139__iter_v1_pview
    ;
    __gm__ float* v312 = PTOAS__GLOBAL_TENSOR_DATA(v124);
    // pto: %rms_partials_inline139__iter_v1_pview
    ;
    const int64_t v313 = 1;
    // pto: %rms_partials_inline139__iter_v1_pview
    ;
    const int64_t v314 = 1;
    // pto: %rms_partials_inline139__iter_v1_pview
    ;
    const int64_t v315 = 1;
    // pto: %rms_partials_inline139__iter_v1_pview
    ;
    int64_t v316 = v78 * v73;
    // pto: %rms_partials_inline139__iter_v1_pview
    ;
    int64_t v317 = v315 * v316;
    // pto: %rms_partials_inline139__iter_v1_pview
    ;
    pto::Shape<1, 1, 1, 16, 1> v318 = pto::Shape<1, 1, 1, 16, 1>(v313, v314, v315, v78, v73);
    // pto: %rms_partials_inline139__iter_v1_pview
    ;
    pto::Stride<-1, -1, -1, -1, -1> v319 = pto::Stride<-1, -1, -1, -1, -1>(v314 * v317, v317, v316, v73, v116);
    // pto: %rms_partials_inline139__iter_v1_pview
    ;
    GlobalTensor<float, pto::Shape<1, 1, 1, 16, 1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN> v320 = GlobalTensor<float, pto::Shape<1, 1, 1, 16, 1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN>(v312 + (v311 + v310 * v73 + v86 * v116), v318, v319);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID3);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID4);
    TSTORE(v320, v308);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  }
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
  dcci((__gm__ void*)0, cache_line_t::ENTIRE_DATA_CACHE);
  pipe_barrier(PIPE_ALL);
  dsb(DSB_DDR);
  SYNCALL<SyncCoreType::Mix>();
  dcci((__gm__ void*)0, cache_line_t::ENTIRE_DATA_CACHE);
  // pto: %100, %101, %tail_lane_inline125__ssa_v0
  int64_t v321 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) v199 * (uint64_t) v85) + (uint64_t) ((int64_t) v33));
  // pto: %102
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
  if (v321 < v18) {
    // pto: %104
    ;
    int64_t v322 = v321 < v20 ? v321 : v20;
    // pto: %103, %105
    ;
    int64_t v323 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) v321 * (uint64_t) v19) + (uint64_t) v322);
    // pto: %107
    ;
    int64_t v324 = (int64_t) ((uint64_t) v20 - (uint64_t) v321);
    // pto: %108
    ;
    int64_t v325 = v324 < v86 ? v86 : v324;
    // pto: %109
    ;
    int64_t v326 = v325 < v73 ? v325 : v73;
    // pto: %106, %110
    ;
    int64_t v327 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) v323 + (uint64_t) v19) + (uint64_t) v326);
    for (int64_t v328 = v323; v328 < v327; v328 += v85) {
      // pto: %scale2_inline47__tile
      ;
      float v329 = (v5)[v85];
      // pto: %111
      ;
      int64_t v330 = (int64_t) ((uint64_t) v327 - (uint64_t) v328);
      // pto: %112
      ;
      int64_t v331 = v330 < v85 ? v330 : v85;
      // pto: %113
      ;
      int64_t v332 = (int64_t) ((uint64_t) v331 * (uint64_t) v72);
      // pto: %c_parts_load_inline219__ssa_v0
      ;
      Tile<TileType::Vec, float, 11, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v333 = Tile<TileType::Vec, float, 11, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v77, v332);
      // pto: %c_parts_load_inline219__ssa_v0
      ;
      uint64_t v334 = (uint64_t) v40;
      TASSIGN(v333, v334);
      // pto: %114
      ;
      int64_t v335 = (int64_t) ((uint64_t) v328 * (uint64_t) v72);
      // pto: %115
      ;
      int64_t v336 = v335 < v86 ? v86 : v335;
      // pto: %parts_2d_inline94__ssa_v0_pview
      ;
      const int64_t v337 = 0;
      // pto: %parts_2d_inline94__ssa_v0_pview
      ;
      __gm__ float* v338 = PTOAS__GLOBAL_TENSOR_DATA(v142);
      // pto: %parts_2d_inline94__ssa_v0_pview
      ;
      const int64_t v339 = 1;
      // pto: %parts_2d_inline94__ssa_v0_pview
      ;
      const int64_t v340 = 1;
      // pto: %parts_2d_inline94__ssa_v0_pview
      ;
      const int64_t v341 = 1;
      // pto: %parts_2d_inline94__ssa_v0_pview
      ;
      int64_t v342 = v77 * v134;
      // pto: %parts_2d_inline94__ssa_v0_pview
      ;
      int64_t v343 = v341 * v342;
      // pto: %parts_2d_inline94__ssa_v0_pview
      ;
      pto::Shape<1, 1, 1, 11, -1> v344 = pto::Shape<1, 1, 1, 11, -1>(v339, v340, v341, v77, v332);
      // pto: %parts_2d_inline94__ssa_v0_pview
      ;
      pto::Stride<-1, -1, -1, -1, -1> v345 = pto::Stride<-1, -1, -1, -1, -1>(v340 * v343, v343, v342, v134, v73);
      // pto: %parts_2d_inline94__ssa_v0_pview
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 11, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v346 = GlobalTensor<float, pto::Shape<1, 1, 1, 11, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v338 + (v337 + v86 * v134 + v336 * v73), v344, v345);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
      TLOAD(v333, v346);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
      // pto: %t__tmp_v4
      ;
      Tile<TileType::Vec, float, 11, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null> v347 = Tile<TileType::Vec, float, 11, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>(v77, v88);
      // pto: %t__tmp_v4
      ;
      uint64_t v348 = (uint64_t) v40;
      TASSIGN(v347, v348);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
      TFILLPAD<pto::TFillPadMode::InPlace>(v347, v333);
      v347.SetValidShape(v77, v88);
      // pto: %c_parts_tmp_inline280__ssa_v0
      ;
      Tile<TileType::Vec, float, 11, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v349 = Tile<TileType::Vec, float, 11, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v77, v88);
      // pto: %c_parts_tmp_inline280__ssa_v0
      ;
      uint64_t v350 = (uint64_t) v44;
      TASSIGN(v349, v350);
      // pto: %t__tmp_v5
      ;
      Tile<TileType::Vec, float, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v351 = Tile<TileType::Vec, float, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v88);
      // pto: %t__tmp_v5
      ;
      uint64_t v352 = (uint64_t) v40;
      TASSIGN(v351, v352);
      // pto: %colsum_src_view
      ;
      Tile<TileType::Vec, float, 11, 256, BLayout::RowMajor, 11, 256, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null> v353;
      TRESHAPE(v353, v347);
      // pto: %colsum_tmp_view
      ;
      Tile<TileType::Vec, float, 11, 256, BLayout::RowMajor, 11, 256, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v354;
      TRESHAPE(v354, v349);
      // pto: %colsum_dst_view
      ;
      Tile<TileType::Vec, float, 1, 256, BLayout::RowMajor, 1, 256, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v355;
      TRESHAPE(v355, v351);
      pipe_barrier(PIPE_V);
      const bool v356 = true;
      TCOLSUM(v355, v353, v354, v356);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID4);
      // pto: %c_total_inline45__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v357 = Tile<TileType::Vec, float, 8, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v89, v72);
      // pto: %c_total_inline45__ssa_v0
      ;
      uint64_t v358 = (uint64_t) v40;
      TASSIGN(v357, v358);
      // pto: %c_sq_load_inline140__ssa_v0
      ;
      Tile<TileType::Vec, float, 11, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v359 = Tile<TileType::Vec, float, 11, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v77, v331);
      // pto: %c_sq_load_inline140__ssa_v0
      ;
      uint64_t v360 = (uint64_t) v45;
      TASSIGN(v359, v360);
      // pto: %117
      ;
      int64_t v361 = v328 < v86 ? v86 : v328;
      // pto: %rms_2d_inline107__ssa_v0_pview
      ;
      int64_t v362 = (int64_t) v29;
      // pto: %rms_2d_inline107__ssa_v0_pview
      ;
      const int64_t v363 = 0;
      // pto: %rms_2d_inline107__ssa_v0_pview
      ;
      __gm__ float* v364 = PTOAS__GLOBAL_TENSOR_DATA(v151);
      // pto: %rms_2d_inline107__ssa_v0_pview
      ;
      const int64_t v365 = 1;
      // pto: %rms_2d_inline107__ssa_v0_pview
      ;
      const int64_t v366 = 1;
      // pto: %rms_2d_inline107__ssa_v0_pview
      ;
      const int64_t v367 = 1;
      // pto: %rms_2d_inline107__ssa_v0_pview
      ;
      int64_t v368 = v77 * v362;
      // pto: %rms_2d_inline107__ssa_v0_pview
      ;
      int64_t v369 = v367 * v368;
      // pto: %rms_2d_inline107__ssa_v0_pview
      ;
      pto::Shape<1, 1, 1, 11, -1> v370 = pto::Shape<1, 1, 1, 11, -1>(v365, v366, v367, v77, v331);
      // pto: %rms_2d_inline107__ssa_v0_pview
      ;
      pto::Stride<-1, -1, -1, -1, -1> v371 = pto::Stride<-1, -1, -1, -1, -1>(v366 * v369, v369, v368, v362, v73);
      // pto: %rms_2d_inline107__ssa_v0_pview
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 11, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v372 = GlobalTensor<float, pto::Shape<1, 1, 1, 11, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v364 + (v363 + v86 * v362 + v361 * v73), v370, v371);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID4);
      TLOAD(v359, v372);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);
      // pto: %t__tmp_v6
      ;
      Tile<TileType::Vec, float, 11, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null> v373 = Tile<TileType::Vec, float, 11, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>(v77, v89);
      // pto: %t__tmp_v6
      ;
      uint64_t v374 = (uint64_t) v45;
      TASSIGN(v373, v374);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);
      TFILLPAD<pto::TFillPadMode::InPlace>(v373, v359);
      v373.SetValidShape(v77, v89);
      // pto: %c_sq_tmp_inline86__ssa_v0
      ;
      Tile<TileType::Vec, float, 11, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v375 = Tile<TileType::Vec, float, 11, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v77, v89);
      // pto: %c_sq_tmp_inline86__ssa_v0
      ;
      uint64_t v376 = (uint64_t) v46;
      TASSIGN(v375, v376);
      // pto: %c_rms_tmp_inline55__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v377 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v89);
      // pto: %c_rms_tmp_inline55__ssa_v0
      ;
      uint64_t v378 = (uint64_t) v47;
      TASSIGN(v377, v378);
      // pto: %t__tmp_v7
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v379 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v89);
      // pto: %t__tmp_v7
      ;
      uint64_t v380 = (uint64_t) v45;
      TASSIGN(v379, v380);
      // pto: %118
      ;
      Tile<TileType::Vec, float, 11, 8, BLayout::RowMajor, 11, 8, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null> v381;
      TRESHAPE(v381, v373);
      // pto: %119
      ;
      Tile<TileType::Vec, float, 11, 8, BLayout::RowMajor, 11, 8, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v382;
      TRESHAPE(v382, v375);
      // pto: %120
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, 1, 8, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v383;
      TRESHAPE(v383, v379);
      pipe_barrier(PIPE_V);
      const bool v384 = true;
      TCOLSUM(v383, v381, v382, v384);
      // pto: %t__tmp_v8
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v385 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v89);
      // pto: %t__tmp_v8
      ;
      uint64_t v386 = (uint64_t) v45;
      TASSIGN(v385, v386);
      pipe_barrier(PIPE_V);
      TMULS(v385, v379, v90);
      // pto: %t__tmp_v9
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v387 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v89);
      // pto: %t__tmp_v9
      ;
      uint64_t v388 = (uint64_t) v45;
      TASSIGN(v387, v388);
      pipe_barrier(PIPE_V);
      TADDS(v387, v385, v91);
      // pto: %t__tmp_v10
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v389 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v89);
      // pto: %t__tmp_v10
      ;
      uint64_t v390 = (uint64_t) v48;
      TASSIGN(v389, v390);
      pipe_barrier(PIPE_V);
      TRSQRT(v389, v387, v377);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID5);
      // pto: %c_inv_inline76__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v391 = Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v89, v73);
      // pto: %c_inv_inline76__ssa_v0
      ;
      uint64_t v392 = (uint64_t) v48;
      TASSIGN(v391, v392);
      // pto: %comb_inv_inline196__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v393 = Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v89, v73);
      // pto: %comb_inv_inline196__ssa_v0
      ;
      uint64_t v394 = (uint64_t) v48;
      TASSIGN(v393, v394);
      // pto: %t__tmp_v11
      ;
      Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v395 = Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v89, v78);
      // pto: %t__tmp_v11
      ;
      uint64_t v396 = (uint64_t) v49;
      TASSIGN(v395, v396);
      // pto: %slice_view
      ;
      Tile<TileType::Vec, float, 8, 32, BLayout::RowMajor, 8, 16, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v397;
      // pto: %slice_view
      ;
      uint64_t v398 = (uint64_t) v49;
      TASSIGN(v397, v398);
      // pto: %raw_inline210__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v399 = Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v89, v78);
      // pto: %raw_inline210__ssa_v0
      ;
      uint64_t v400 = (uint64_t) v40;
      TASSIGN(v399, v400);
      TADDS(v399, v397, v82);
      // pto: %base_inline153__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v401 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v78);
      // pto: %base_inline153__ssa_v0
      ;
      uint64_t v402 = (uint64_t) v45;
      TASSIGN(v401, v402);
      // pto: %hc_base_2d_inline203__ssa_v0_pview
      ;
      __gm__ float* v403 = PTOAS__GLOBAL_TENSOR_DATA(v159);
      // pto: %hc_base_2d_inline203__ssa_v0_pview
      ;
      unsigned v404 = 8;
      // pto: %hc_base_2d_inline203__ssa_v0_pview
      ;
      pto::Shape<1, 1, 1, 1, 16> v405 = pto::Shape<1, 1, 1, 1, 16>();
      // pto: %hc_base_2d_inline203__ssa_v0_pview
      ;
      pto::Stride<24, 24, 24, 24, 1> v406 = pto::Stride<24, 24, 24, 24, 1>();
      // pto: %hc_base_2d_inline203__ssa_v0_pview
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<24, 24, 24, 24, 1>, pto::Layout::ND> v407 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<24, 24, 24, 24, 1>, pto::Layout::ND>(v403 + v404, v405, v406);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID5);
      TLOAD(v401, v407);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID5);
      // pto: %t__tmp_v12
      ;
      Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v408 = Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v89, v78);
      // pto: %t__tmp_v12
      ;
      uint64_t v409 = (uint64_t) v50;
      TASSIGN(v408, v409);
      pipe_barrier(PIPE_V);
      TROWEXPANDMUL(v408, v399, v393);
      // pto: %t__tmp_v13
      ;
      Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v410 = Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v89, v78);
      // pto: %t__tmp_v13
      ;
      uint64_t v411 = (uint64_t) v50;
      TASSIGN(v410, v411);
      pipe_barrier(PIPE_V);
      TMULS(v410, v408, v329);
      // pto: %t__tmp_v14
      ;
      Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v412 = Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v89, v78);
      // pto: %t__tmp_v14
      ;
      uint64_t v413 = (uint64_t) v40;
      TASSIGN(v412, v413);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID5);
      TCOLEXPAND(v412, v401);
      // pto: %logits_inline200__ssa_v1
      ;
      Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v414 = Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v89, v78);
      // pto: %logits_inline200__ssa_v1
      ;
      uint64_t v415 = (uint64_t) v51;
      TASSIGN(v414, v415);
      pipe_barrier(PIPE_V);
      TADD(v414, v410, v412);
      // pto: %t__tmp_v15
      ;
      Tile<TileType::Vec, float, 2, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v416 = Tile<TileType::Vec, float, 2, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v85, v78);
      // pto: %t__tmp_v15
      ;
      uint64_t v417 = (uint64_t) v51;
      TASSIGN(v416, v417);
      // pto: %regular_logits_inline251__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v418 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v72);
      // pto: %regular_logits_inline251__ssa_v0
      ;
      uint64_t v419 = (uint64_t) v51;
      TASSIGN(v418, v419);
      // pto: %ids_inline201__ci_tmp_v0
      ;
      Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v420 = Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v92);
      // pto: %ids_inline201__ci_tmp_v0
      ;
      uint64_t v421 = (uint64_t) v40;
      TASSIGN(v420, v421);
      // pto: %ids_inline201__ssa_v1
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v422 = Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v72);
      // pto: %ids_inline201__ssa_v1
      ;
      uint64_t v423 = (uint64_t) v52;
      TASSIGN(v422, v423);
      // pto: %ci_tmp_view
      ;
      Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, 1, 192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v424;
      TRESHAPE(v424, v420);
      // pto: %ci_dst_view
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, 1, 32, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v425;
      TRESHAPE(v425, v422);
      pipe_barrier(PIPE_V);
      TCI<Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, 1, 32, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, 1, 192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, int32_t, 0>(v425, v80, v424);
      // pto: %tmp_inline204__ssa_v1
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v426 = Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v72);
      // pto: %tmp_inline204__ssa_v1
      ;
      uint64_t v427 = (uint64_t) v40;
      TASSIGN(v426, v427);
      // pto: %idx_float_inline257__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v428 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v72);
      // pto: %idx_float_inline257__ssa_v0
      ;
      uint64_t v429 = (uint64_t) v53;
      TASSIGN(v428, v429);
      pipe_barrier(PIPE_V);
      RoundMode v430 = RoundMode::CAST_ROUND;
      SaturationMode v431 = SaturationMode::ON;
      TCVT(v428, v422, v430, v431);
      // pto: %t__tmp_v16
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v432 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v72);
      // pto: %t__tmp_v16
      ;
      uint64_t v433 = (uint64_t) v54;
      TASSIGN(v432, v433);
      pipe_barrier(PIPE_V);
      TMULS(v432, v428, v93);
      // pto: %t__tmp_v17
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v434 = Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v72);
      // pto: %t__tmp_v17
      ;
      uint64_t v435 = (uint64_t) v54;
      TASSIGN(v434, v435);
      pipe_barrier(PIPE_V);
      RoundMode v436 = RoundMode::CAST_FLOOR;
      SaturationMode v437 = SaturationMode::ON;
      TCVT(v434, v432, v436, v437);
      // pto: %ri1_low_inline258__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v438 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v72);
      // pto: %ri1_low_inline258__ssa_v0
      ;
      uint64_t v439 = (uint64_t) v54;
      TASSIGN(v438, v439);
      pipe_barrier(PIPE_V);
      RoundMode v440 = RoundMode::CAST_ROUND;
      SaturationMode v441 = SaturationMode::ON;
      TCVT(v438, v434, v440, v441);
      // pto: %t__tmp_v18
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v442 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v72);
      // pto: %t__tmp_v18
      ;
      uint64_t v443 = (uint64_t) v55;
      TASSIGN(v442, v443);
      TMULS(v442, v428, v94);
      // pto: %t__tmp_v19
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v444 = Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v72);
      // pto: %t__tmp_v19
      ;
      uint64_t v445 = (uint64_t) v55;
      TASSIGN(v444, v445);
      pipe_barrier(PIPE_V);
      RoundMode v446 = RoundMode::CAST_FLOOR;
      SaturationMode v447 = SaturationMode::ON;
      TCVT(v444, v442, v446, v447);
      // pto: %ri1_high_inline226__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v448 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v72);
      // pto: %ri1_high_inline226__ssa_v0
      ;
      uint64_t v449 = (uint64_t) v55;
      TASSIGN(v448, v449);
      pipe_barrier(PIPE_V);
      RoundMode v450 = RoundMode::CAST_ROUND;
      SaturationMode v451 = SaturationMode::ON;
      TCVT(v448, v444, v450, v451);
      // pto: %t__tmp_v20
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v452 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v72);
      // pto: %t__tmp_v20
      ;
      uint64_t v453 = (uint64_t) v55;
      TASSIGN(v452, v453);
      pipe_barrier(PIPE_V);
      TMULS(v452, v448, v95);
      // pto: %ri1_bit_inline259__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v454 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v72);
      // pto: %ri1_bit_inline259__ssa_v0
      ;
      uint64_t v455 = (uint64_t) v54;
      TASSIGN(v454, v455);
      pipe_barrier(PIPE_V);
      TSUB(v454, v438, v452);
      // pto: %t__tmp_v21
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v456 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v72);
      // pto: %t__tmp_v21
      ;
      uint64_t v457 = (uint64_t) v54;
      TASSIGN(v456, v457);
      pipe_barrier(PIPE_V);
      TMULS(v456, v454, v95);
      // pto: %t__tmp_v22
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v458 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v72);
      // pto: %t__tmp_v22
      ;
      uint64_t v459 = (uint64_t) v54;
      TASSIGN(v458, v459);
      pipe_barrier(PIPE_V);
      TSUB(v458, v428, v456);
      // pto: %t__tmp_v23
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v460 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v72);
      // pto: %t__tmp_v23
      ;
      uint64_t v461 = (uint64_t) v54;
      TASSIGN(v460, v461);
      pipe_barrier(PIPE_V);
      TADDS(v460, v458, v93);
      // pto: %ri1_inline186__ssa_v2
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v462 = Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v72);
      // pto: %ri1_inline186__ssa_v2
      ;
      uint64_t v463 = (uint64_t) v54;
      TASSIGN(v462, v463);
      pipe_barrier(PIPE_V);
      RoundMode v464 = RoundMode::CAST_ROUND;
      SaturationMode v465 = SaturationMode::ON;
      TCVT(v462, v460, v464, v465);
      // pto: %t__tmp_v24
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v466 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v72);
      // pto: %t__tmp_v24
      ;
      uint64_t v467 = (uint64_t) v55;
      TASSIGN(v466, v467);
      TMULS(v466, v428, v94);
      // pto: %t__tmp_v25
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v468 = Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v72);
      // pto: %t__tmp_v25
      ;
      uint64_t v469 = (uint64_t) v55;
      TASSIGN(v468, v469);
      pipe_barrier(PIPE_V);
      RoundMode v470 = RoundMode::CAST_FLOOR;
      SaturationMode v471 = SaturationMode::ON;
      TCVT(v468, v466, v470, v471);
      // pto: %ri2_low_inline261__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v472 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v72);
      // pto: %ri2_low_inline261__ssa_v0
      ;
      uint64_t v473 = (uint64_t) v55;
      TASSIGN(v472, v473);
      pipe_barrier(PIPE_V);
      RoundMode v474 = RoundMode::CAST_ROUND;
      SaturationMode v475 = SaturationMode::ON;
      TCVT(v472, v468, v474, v475);
      // pto: %t__tmp_v26
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v476 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v72);
      // pto: %t__tmp_v26
      ;
      uint64_t v477 = (uint64_t) v50;
      TASSIGN(v476, v477);
      TMULS(v476, v428, v96);
      // pto: %t__tmp_v27
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v478 = Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v72);
      // pto: %t__tmp_v27
      ;
      uint64_t v479 = (uint64_t) v50;
      TASSIGN(v478, v479);
      pipe_barrier(PIPE_V);
      RoundMode v480 = RoundMode::CAST_FLOOR;
      SaturationMode v481 = SaturationMode::ON;
      TCVT(v478, v476, v480, v481);
      // pto: %ri2_high_inline264__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v482 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v72);
      // pto: %ri2_high_inline264__ssa_v0
      ;
      uint64_t v483 = (uint64_t) v50;
      TASSIGN(v482, v483);
      pipe_barrier(PIPE_V);
      RoundMode v484 = RoundMode::CAST_ROUND;
      SaturationMode v485 = SaturationMode::ON;
      TCVT(v482, v478, v484, v485);
      // pto: %t__tmp_v28
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v486 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v72);
      // pto: %t__tmp_v28
      ;
      uint64_t v487 = (uint64_t) v50;
      TASSIGN(v486, v487);
      pipe_barrier(PIPE_V);
      TMULS(v486, v482, v95);
      // pto: %ri2_bit_inline265__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v488 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v72);
      // pto: %ri2_bit_inline265__ssa_v0
      ;
      uint64_t v489 = (uint64_t) v55;
      TASSIGN(v488, v489);
      pipe_barrier(PIPE_V);
      TSUB(v488, v472, v486);
      // pto: %t__tmp_v29
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v490 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v72);
      // pto: %t__tmp_v29
      ;
      uint64_t v491 = (uint64_t) v55;
      TASSIGN(v490, v491);
      pipe_barrier(PIPE_V);
      TMULS(v490, v488, v97);
      // pto: %t__tmp_v30
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v492 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v72);
      // pto: %t__tmp_v30
      ;
      uint64_t v493 = (uint64_t) v55;
      TASSIGN(v492, v493);
      pipe_barrier(PIPE_V);
      TSUB(v492, v428, v490);
      // pto: %t__tmp_v31
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v494 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v72);
      // pto: %t__tmp_v31
      ;
      uint64_t v495 = (uint64_t) v55;
      TASSIGN(v494, v495);
      pipe_barrier(PIPE_V);
      TADDS(v494, v492, v95);
      // pto: %ri2_inline252__ssa_v1
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v496 = Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v72);
      // pto: %ri2_inline252__ssa_v1
      ;
      uint64_t v497 = (uint64_t) v55;
      TASSIGN(v496, v497);
      pipe_barrier(PIPE_V);
      RoundMode v498 = RoundMode::CAST_ROUND;
      SaturationMode v499 = SaturationMode::ON;
      TCVT(v496, v494, v498, v499);
      // pto: %t__tmp_v32
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v500 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v72);
      // pto: %t__tmp_v32
      ;
      uint64_t v501 = (uint64_t) v50;
      TASSIGN(v500, v501);
      TMULS(v500, v428, v96);
      // pto: %t__tmp_v33
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v502 = Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v72);
      // pto: %t__tmp_v33
      ;
      uint64_t v503 = (uint64_t) v50;
      TASSIGN(v502, v503);
      pipe_barrier(PIPE_V);
      RoundMode v504 = RoundMode::CAST_FLOOR;
      SaturationMode v505 = SaturationMode::ON;
      TCVT(v502, v500, v504, v505);
      // pto: %ci1_low_inline295__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v506 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v72);
      // pto: %ci1_low_inline295__ssa_v0
      ;
      uint64_t v507 = (uint64_t) v50;
      TASSIGN(v506, v507);
      pipe_barrier(PIPE_V);
      RoundMode v508 = RoundMode::CAST_ROUND;
      SaturationMode v509 = SaturationMode::ON;
      TCVT(v506, v502, v508, v509);
      // pto: %t__tmp_v34
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v510 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v72);
      // pto: %t__tmp_v34
      ;
      uint64_t v511 = (uint64_t) v56;
      TASSIGN(v510, v511);
      TMULS(v510, v428, v98);
      // pto: %t__tmp_v35
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v512 = Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v72);
      // pto: %t__tmp_v35
      ;
      uint64_t v513 = (uint64_t) v56;
      TASSIGN(v512, v513);
      pipe_barrier(PIPE_V);
      RoundMode v514 = RoundMode::CAST_FLOOR;
      SaturationMode v515 = SaturationMode::ON;
      TCVT(v512, v510, v514, v515);
      // pto: %ci1_high_inline270__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v516 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v72);
      // pto: %ci1_high_inline270__ssa_v0
      ;
      uint64_t v517 = (uint64_t) v56;
      TASSIGN(v516, v517);
      pipe_barrier(PIPE_V);
      RoundMode v518 = RoundMode::CAST_ROUND;
      SaturationMode v519 = SaturationMode::ON;
      TCVT(v516, v512, v518, v519);
      // pto: %t__tmp_v36
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v520 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v72);
      // pto: %t__tmp_v36
      ;
      uint64_t v521 = (uint64_t) v56;
      TASSIGN(v520, v521);
      pipe_barrier(PIPE_V);
      TMULS(v520, v516, v95);
      // pto: %ci1_bit_inline194__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v522 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v72);
      // pto: %ci1_bit_inline194__ssa_v0
      ;
      uint64_t v523 = (uint64_t) v50;
      TASSIGN(v522, v523);
      pipe_barrier(PIPE_V);
      TSUB(v522, v506, v520);
      // pto: %t__tmp_v37
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v524 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v72);
      // pto: %t__tmp_v37
      ;
      uint64_t v525 = (uint64_t) v50;
      TASSIGN(v524, v525);
      pipe_barrier(PIPE_V);
      TMULS(v524, v522, v99);
      // pto: %t__tmp_v38
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v526 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v72);
      // pto: %t__tmp_v38
      ;
      uint64_t v527 = (uint64_t) v50;
      TASSIGN(v526, v527);
      pipe_barrier(PIPE_V);
      TSUB(v526, v428, v524);
      // pto: %t__tmp_v39
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v528 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v72);
      // pto: %t__tmp_v39
      ;
      uint64_t v529 = (uint64_t) v50;
      TASSIGN(v528, v529);
      pipe_barrier(PIPE_V);
      TADDS(v528, v526, v97);
      // pto: %ci1_inline183__ssa_v1
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v530 = Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v72);
      // pto: %ci1_inline183__ssa_v1
      ;
      uint64_t v531 = (uint64_t) v50;
      TASSIGN(v530, v531);
      pipe_barrier(PIPE_V);
      RoundMode v532 = RoundMode::CAST_ROUND;
      SaturationMode v533 = SaturationMode::ON;
      TCVT(v530, v528, v532, v533);
      // pto: %t__tmp_v40
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v534 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v72);
      // pto: %t__tmp_v40
      ;
      uint64_t v535 = (uint64_t) v56;
      TASSIGN(v534, v535);
      TMULS(v534, v428, v98);
      // pto: %t__tmp_v41
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v536 = Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v72);
      // pto: %t__tmp_v41
      ;
      uint64_t v537 = (uint64_t) v56;
      TASSIGN(v536, v537);
      pipe_barrier(PIPE_V);
      RoundMode v538 = RoundMode::CAST_FLOOR;
      SaturationMode v539 = SaturationMode::ON;
      TCVT(v536, v534, v538, v539);
      // pto: %ci2_low_inline59__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v540 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v72);
      // pto: %ci2_low_inline59__ssa_v0
      ;
      uint64_t v541 = (uint64_t) v56;
      TASSIGN(v540, v541);
      pipe_barrier(PIPE_V);
      RoundMode v542 = RoundMode::CAST_ROUND;
      SaturationMode v543 = SaturationMode::ON;
      TCVT(v540, v536, v542, v543);
      // pto: %t__tmp_v42
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v544 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v72);
      // pto: %t__tmp_v42
      ;
      uint64_t v545 = (uint64_t) v52;
      TASSIGN(v544, v545);
      TMULS(v544, v428, v100);
      // pto: %t__tmp_v43
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v546 = Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v72);
      // pto: %t__tmp_v43
      ;
      uint64_t v547 = (uint64_t) v52;
      TASSIGN(v546, v547);
      pipe_barrier(PIPE_V);
      RoundMode v548 = RoundMode::CAST_FLOOR;
      SaturationMode v549 = SaturationMode::ON;
      TCVT(v546, v544, v548, v549);
      // pto: %ci2_high_inline256__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v550 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v72);
      // pto: %ci2_high_inline256__ssa_v0
      ;
      uint64_t v551 = (uint64_t) v52;
      TASSIGN(v550, v551);
      pipe_barrier(PIPE_V);
      RoundMode v552 = RoundMode::CAST_ROUND;
      SaturationMode v553 = SaturationMode::ON;
      TCVT(v550, v546, v552, v553);
      // pto: %t__tmp_v44
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v554 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v72);
      // pto: %t__tmp_v44
      ;
      uint64_t v555 = (uint64_t) v52;
      TASSIGN(v554, v555);
      pipe_barrier(PIPE_V);
      TMULS(v554, v550, v95);
      // pto: %ci2_bit_inline166__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v556 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v72);
      // pto: %ci2_bit_inline166__ssa_v0
      ;
      uint64_t v557 = (uint64_t) v56;
      TASSIGN(v556, v557);
      pipe_barrier(PIPE_V);
      TSUB(v556, v540, v554);
      // pto: %t__tmp_v45
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v558 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v72);
      // pto: %t__tmp_v45
      ;
      uint64_t v559 = (uint64_t) v56;
      TASSIGN(v558, v559);
      pipe_barrier(PIPE_V);
      TMULS(v558, v556, v101);
      // pto: %t__tmp_v46
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v560 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v72);
      // pto: %t__tmp_v46
      ;
      uint64_t v561 = (uint64_t) v53;
      TASSIGN(v560, v561);
      pipe_barrier(PIPE_V);
      TSUB(v560, v428, v558);
      // pto: %t__tmp_v47
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v562 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v72);
      // pto: %t__tmp_v47
      ;
      uint64_t v563 = (uint64_t) v53;
      TASSIGN(v562, v563);
      pipe_barrier(PIPE_V);
      TADDS(v562, v560, v99);
      // pto: %ci2_inline255__ssa_v1
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v564 = Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v72);
      // pto: %ci2_inline255__ssa_v1
      ;
      uint64_t v565 = (uint64_t) v53;
      TASSIGN(v564, v565);
      pipe_barrier(PIPE_V);
      RoundMode v566 = RoundMode::CAST_ROUND;
      SaturationMode v567 = SaturationMode::ON;
      TCVT(v564, v562, v566, v567);
      // pto: %t__tmp_v48
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v568 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v72);
      // pto: %t__tmp_v48
      ;
      uint64_t v569 = (uint64_t) v56;
      TASSIGN(v568, v569);
      TGATHER(v568, v418, v462, v426);
      // pto: %max2_inline227__ssa_v1
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v570 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v72);
      // pto: %max2_inline227__ssa_v1
      ;
      uint64_t v571 = (uint64_t) v56;
      TASSIGN(v570, v571);
      pipe_barrier(PIPE_V);
      TMAX(v570, v418, v568);
      // pto: %t__tmp_v49
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v572 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v72);
      // pto: %t__tmp_v49
      ;
      uint64_t v573 = (uint64_t) v52;
      TASSIGN(v572, v573);
      pipe_barrier(PIPE_V);
      TGATHER(v572, v570, v496, v426);
      // pto: %max4_inline137__ssa_v1
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v574 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v72);
      // pto: %max4_inline137__ssa_v1
      ;
      uint64_t v575 = (uint64_t) v56;
      TASSIGN(v574, v575);
      pipe_barrier(PIPE_V);
      TMAX(v574, v570, v572);
      // pto: %t__tmp_v50
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v576 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v72);
      // pto: %t__tmp_v50
      ;
      uint64_t v577 = (uint64_t) v56;
      TASSIGN(v576, v577);
      pipe_barrier(PIPE_V);
      TSUB(v576, v418, v574);
      // pto: %exps_inline233__ssa_v1
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v578 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v72);
      // pto: %exps_inline233__ssa_v1
      ;
      uint64_t v579 = (uint64_t) v56;
      TASSIGN(v578, v579);
      pipe_barrier(PIPE_V);
      TEXP(v578, v576);
      // pto: %t__tmp_v51
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v580 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v72);
      // pto: %t__tmp_v51
      ;
      uint64_t v581 = (uint64_t) v51;
      TASSIGN(v580, v581);
      pipe_barrier(PIPE_V);
      TGATHER(v580, v578, v462, v426);
      // pto: %sum2_inline229__ssa_v1
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v582 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v72);
      // pto: %sum2_inline229__ssa_v1
      ;
      uint64_t v583 = (uint64_t) v54;
      TASSIGN(v582, v583);
      pipe_barrier(PIPE_V);
      TADD(v582, v578, v580);
      // pto: %t__tmp_v52
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v584 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v72);
      // pto: %t__tmp_v52
      ;
      uint64_t v585 = (uint64_t) v51;
      TASSIGN(v584, v585);
      pipe_barrier(PIPE_V);
      TGATHER(v584, v582, v496, v426);
      // pto: %sum4_inline231__ssa_v1
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v586 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v72);
      // pto: %sum4_inline231__ssa_v1
      ;
      uint64_t v587 = (uint64_t) v54;
      TASSIGN(v586, v587);
      pipe_barrier(PIPE_V);
      TADD(v586, v582, v584);
      // pto: %t__tmp_v53
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v588 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v72);
      // pto: %t__tmp_v53
      ;
      uint64_t v589 = (uint64_t) v54;
      TASSIGN(v588, v589);
      pipe_barrier(PIPE_V);
      TDIV(v588, v578, v586);
      // pto: %probs_inline232__ssa_v1
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v590 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v72);
      // pto: %probs_inline232__ssa_v1
      ;
      uint64_t v591 = (uint64_t) v54;
      TASSIGN(v590, v591);
      pipe_barrier(PIPE_V);
      TADDS(v590, v588, v91);
      // pto: %t__tmp_v54
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v592 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v72);
      // pto: %t__tmp_v54
      ;
      uint64_t v593 = (uint64_t) v55;
      TASSIGN(v592, v593);
      pipe_barrier(PIPE_V);
      TGATHER(v592, v590, v530, v426);
      // pto: %col2_inline239__ssa_v4
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v594 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v72);
      // pto: %col2_inline239__ssa_v4
      ;
      uint64_t v595 = (uint64_t) v55;
      TASSIGN(v594, v595);
      pipe_barrier(PIPE_V);
      TADD(v594, v590, v592);
      // pto: %t__tmp_v55
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v596 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v72);
      // pto: %t__tmp_v55
      ;
      uint64_t v597 = (uint64_t) v50;
      TASSIGN(v596, v597);
      pipe_barrier(PIPE_V);
      TGATHER(v596, v594, v564, v426);
      // pto: %t__tmp_v56
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v598 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v72);
      // pto: %t__tmp_v56
      ;
      uint64_t v599 = (uint64_t) v40;
      TASSIGN(v598, v599);
      pipe_barrier(PIPE_V);
      TADD(v598, v594, v596);
      // pto: %col4_inline243__ssa_v4
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v600 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v72);
      // pto: %col4_inline243__ssa_v4
      ;
      uint64_t v601 = (uint64_t) v40;
      TASSIGN(v600, v601);
      pipe_barrier(PIPE_V);
      TADDS(v600, v598, v91);
      // pto: %cur_inline244__ssa_v4
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v602 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v72);
      // pto: %cur_inline244__ssa_v4
      ;
      uint64_t v603 = (uint64_t) v40;
      TASSIGN(v602, v603);
      pipe_barrier(PIPE_V);
      TDIV(v602, v590, v600);
      set_flag(PIPE_V, PIPE_S, EVENT_ID0);
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID5);
      wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
      for (int64_t v604 = v86; v604 < v331; v604 += v73) {
        // pto: %122
        ;
        int64_t v605 = (int64_t) ((uint64_t) v604 * (uint64_t) v78);
        // pto: %sv00_inline267__ssa_v0
        ;
        float v606 = v602.GetValue(v605);
        // pto: %124
        ;
        int64_t v607 = (int64_t) ((uint64_t) v605 + (uint64_t) v73);
        // pto: %sv01_inline276__ssa_v0
        ;
        float v608 = v602.GetValue(v607);
        // pto: %128
        ;
        int64_t v609 = (int64_t) ((uint64_t) v605 + (uint64_t) v85);
        // pto: %sv02_inline281__ssa_v0
        ;
        float v610 = v602.GetValue(v609);
        // pto: %132
        ;
        int64_t v611 = (int64_t) ((uint64_t) v605 + (uint64_t) v76);
        // pto: %sv03_inline263__ssa_v0
        ;
        float v612 = v602.GetValue(v611);
        // pto: %136
        ;
        int64_t v613 = (int64_t) ((uint64_t) v605 + (uint64_t) v102);
        // pto: %sv04_inline311__ssa_v0
        ;
        float v614 = v602.GetValue(v613);
        // pto: %140
        ;
        int64_t v615 = (int64_t) ((uint64_t) v605 + (uint64_t) v103);
        // pto: %sv05_inline284__ssa_v0
        ;
        float v616 = v602.GetValue(v615);
        // pto: %144
        ;
        int64_t v617 = (int64_t) ((uint64_t) v605 + (uint64_t) v104);
        // pto: %sv06_inline100__ssa_v0
        ;
        float v618 = v602.GetValue(v617);
        // pto: %148
        ;
        int64_t v619 = (int64_t) ((uint64_t) v605 + (uint64_t) v105);
        // pto: %sv07_inline198__ssa_v0
        ;
        float v620 = v602.GetValue(v619);
        // pto: %152
        ;
        int64_t v621 = (int64_t) ((uint64_t) v605 + (uint64_t) v89);
        // pto: %sv08_inline172__ssa_v0
        ;
        float v622 = v602.GetValue(v621);
        // pto: %156
        ;
        int64_t v623 = (int64_t) ((uint64_t) v605 + (uint64_t) v106);
        // pto: %sv09_inline221__ssa_v0
        ;
        float v624 = v602.GetValue(v623);
        // pto: %160
        ;
        int64_t v625 = (int64_t) ((uint64_t) v605 + (uint64_t) v107);
        // pto: %sv10_inline286__ssa_v0
        ;
        float v626 = v602.GetValue(v625);
        // pto: %164
        ;
        int64_t v627 = (int64_t) ((uint64_t) v605 + (uint64_t) v77);
        // pto: %sv11_inline260__ssa_v0
        ;
        float v628 = v602.GetValue(v627);
        // pto: %168
        ;
        int64_t v629 = (int64_t) ((uint64_t) v605 + (uint64_t) v108);
        // pto: %sv12_inline287__ssa_v0
        ;
        float v630 = v602.GetValue(v629);
        // pto: %172
        ;
        int64_t v631 = (int64_t) ((uint64_t) v605 + (uint64_t) v109);
        // pto: %sv13_inline288__ssa_v0
        ;
        float v632 = v602.GetValue(v631);
        // pto: %176
        ;
        int64_t v633 = (int64_t) ((uint64_t) v605 + (uint64_t) v110);
        // pto: %sv14_inline290__ssa_v0
        ;
        float v634 = v602.GetValue(v633);
        // pto: %180
        ;
        int64_t v635 = (int64_t) ((uint64_t) v605 + (uint64_t) v111);
        // pto: %sv15_inline291__ssa_v0
        ;
        float v636 = v602.GetValue(v635);
        // pto: %sv00_inline267__rv_v2
        ;
        float v637;
        // pto: %sv01_inline276__rv_v2
        ;
        float v638;
        // pto: %sv02_inline281__rv_v2
        ;
        float v639;
        // pto: %sv03_inline263__rv_v2
        ;
        float v640;
        // pto: %sv04_inline311__rv_v2
        ;
        float v641;
        // pto: %sv05_inline284__rv_v2
        ;
        float v642;
        // pto: %sv06_inline100__rv_v2
        ;
        float v643;
        // pto: %sv07_inline198__rv_v2
        ;
        float v644;
        // pto: %sv08_inline172__rv_v2
        ;
        float v645;
        // pto: %sv09_inline221__rv_v2
        ;
        float v646;
        // pto: %sv10_inline286__rv_v2
        ;
        float v647;
        // pto: %sv11_inline260__rv_v2
        ;
        float v648;
        // pto: %sv12_inline287__rv_v2
        ;
        float v649;
        // pto: %sv13_inline288__rv_v2
        ;
        float v650;
        // pto: %sv14_inline290__rv_v2
        ;
        float v651;
        // pto: %sv15_inline291__rv_v2
        ;
        float v652;
        v637 = v606;
        v638 = v608;
        v639 = v610;
        v640 = v612;
        v641 = v614;
        v642 = v616;
        v643 = v618;
        v644 = v620;
        v645 = v622;
        v646 = v624;
        v647 = v626;
        v648 = v628;
        v649 = v630;
        v650 = v632;
        v651 = v634;
        v652 = v636;
        for (int64_t v653 = v86; v653 < v112; v653 += v73) {
          // pto: %183
          ;
          float v654 = v637 + v638;
          // pto: %184
          ;
          float v655 = v639 + v640;
          // pto: %185, %186, %187
          ;
          float v656 = v93 / (v654 + v655 + v91);
          // pto: %188
          ;
          float v657 = v637 * v656;
          // pto: %189
          ;
          float v658 = v638 * v656;
          // pto: %190
          ;
          float v659 = v639 * v656;
          // pto: %191
          ;
          float v660 = v640 * v656;
          // pto: %192
          ;
          float v661 = v641 + v642;
          // pto: %193
          ;
          float v662 = v643 + v644;
          // pto: %194, %195, %196
          ;
          float v663 = v93 / (v661 + v662 + v91);
          // pto: %197
          ;
          float v664 = v641 * v663;
          // pto: %198
          ;
          float v665 = v642 * v663;
          // pto: %199
          ;
          float v666 = v643 * v663;
          // pto: %200
          ;
          float v667 = v644 * v663;
          // pto: %201
          ;
          float v668 = v645 + v646;
          // pto: %202
          ;
          float v669 = v647 + v648;
          // pto: %203, %204, %205
          ;
          float v670 = v93 / (v668 + v669 + v91);
          // pto: %206
          ;
          float v671 = v645 * v670;
          // pto: %207
          ;
          float v672 = v646 * v670;
          // pto: %208
          ;
          float v673 = v647 * v670;
          // pto: %209
          ;
          float v674 = v648 * v670;
          // pto: %210
          ;
          float v675 = v649 + v650;
          // pto: %211
          ;
          float v676 = v651 + v652;
          // pto: %212, %213, %214
          ;
          float v677 = v93 / (v675 + v676 + v91);
          // pto: %215
          ;
          float v678 = v649 * v677;
          // pto: %216
          ;
          float v679 = v650 * v677;
          // pto: %217
          ;
          float v680 = v651 * v677;
          // pto: %218
          ;
          float v681 = v652 * v677;
          // pto: %219, %220, %221, %222, %223
          ;
          float v682 = v93 / ((v657 + v664) + (v671 + v678) + v91);
          // pto: %224
          ;
          // pto: %225
          ;
          // pto: %226
          ;
          // pto: %227
          ;
          // pto: %228, %229, %230, %231, %232
          ;
          float v683 = v93 / ((v658 + v665) + (v672 + v679) + v91);
          // pto: %233
          ;
          // pto: %234
          ;
          // pto: %235
          ;
          // pto: %236
          ;
          // pto: %237, %238, %239, %240, %241
          ;
          float v684 = v93 / ((v659 + v666) + (v673 + v680) + v91);
          // pto: %242
          ;
          // pto: %243
          ;
          // pto: %244
          ;
          // pto: %245
          ;
          // pto: %246, %247, %248, %249, %250
          ;
          float v685 = v93 / ((v660 + v667) + (v674 + v681) + v91);
          // pto: %251
          ;
          // pto: %252
          ;
          // pto: %253
          ;
          // pto: %254
          ;
          v637 = v657 * v682;
          v638 = v658 * v683;
          v639 = v659 * v684;
          v640 = v660 * v685;
          v641 = v664 * v682;
          v642 = v665 * v683;
          v643 = v666 * v684;
          v644 = v667 * v685;
          v645 = v671 * v682;
          v646 = v672 * v683;
          v647 = v673 * v684;
          v648 = v674 * v685;
          v649 = v678 * v682;
          v650 = v679 * v683;
          v651 = v680 * v684;
          v652 = v681 * v685;
        };
        v602.SetValue(v605, v637);
        v602.SetValue(v607, v638);
        v602.SetValue(v609, v639);
        v602.SetValue(v611, v640);
        v602.SetValue(v613, v641);
        v602.SetValue(v615, v642);
        v602.SetValue(v617, v643);
        v602.SetValue(v619, v644);
        v602.SetValue(v621, v645);
        v602.SetValue(v623, v646);
        v602.SetValue(v625, v647);
        v602.SetValue(v627, v648);
        v602.SetValue(v629, v649);
        v602.SetValue(v631, v650);
        v602.SetValue(v633, v651);
        v602.SetValue(v635, v652);
      };
      set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
      // pto: %t__tmp_v57
      ;
      Tile<TileType::Vec, float, 2, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v686 = Tile<TileType::Vec, float, 2, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v85, v78);
      // pto: %t__tmp_v57
      ;
      uint64_t v687 = (uint64_t) v40;
      TASSIGN(v686, v687);
      v686.SetValidShape(v331, v78);
      // pto: %comb_inline133__ssa_v0_pview
      ;
      const int64_t v688 = 0;
      // pto: %comb_inline133__ssa_v0_pview
      ;
      __gm__ float* v689 = PTOAS__GLOBAL_TENSOR_DATA(v168);
      // pto: %comb_inline133__ssa_v0_pview
      ;
      const int64_t v690 = 1;
      // pto: %comb_inline133__ssa_v0_pview
      ;
      const int64_t v691 = 1;
      // pto: %comb_inline133__ssa_v0_pview
      ;
      const int64_t v692 = 1;
      // pto: %comb_inline133__ssa_v0_pview
      ;
      int64_t v693 = v331 * v78;
      // pto: %comb_inline133__ssa_v0_pview
      ;
      int64_t v694 = v692 * v693;
      // pto: %comb_inline133__ssa_v0_pview
      ;
      pto::Shape<1, 1, 1, -1, 16> v695 = pto::Shape<1, 1, 1, -1, 16>(v690, v691, v692, v331, v78);
      // pto: %comb_inline133__ssa_v0_pview
      ;
      pto::Stride<-1, -1, -1, -1, -1> v696 = pto::Stride<-1, -1, -1, -1, -1>(v691 * v694, v694, v693, v78, v73);
      // pto: %comb_inline133__ssa_v0_pview
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, -1, 16>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v697 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, 16>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v689 + (v688 + v361 * v78 + v86 * v73), v695, v696);
      wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID5);
      TSTORE(v697, v686);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
    };
  } else {
    // pto: %319
    ;
    int64_t v698 = (int64_t) ((uint64_t) v321 - (uint64_t) v18);
    // pto: %321
    ;
    int64_t v699 = v698 < v22 ? v698 : v22;
    // pto: %320, %322, %323
    ;
    int64_t v700 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) ((int64_t) (uint64_t) v698 * (uint64_t) v21) + (uint64_t) v699) * (uint64_t) v85);
    // pto: %324
    ;
    int64_t v701 = (int64_t) ((uint64_t) v22 - (uint64_t) v698);
    // pto: %325
    ;
    int64_t v702 = v701 < v86 ? v86 : v701;
    // pto: %326
    ;
    int64_t v703 = v702 < v73 ? v702 : v73;
    // pto: %329, %327, %328
    ;
    int64_t v704 = (int64_t) ((uint64_t) v700 + (uint64_t) ((int64_t) (uint64_t) ((int64_t) (uint64_t) v21 + (uint64_t) v703) * (uint64_t) v85));
    // pto: %330
    ;
    // pto: %331
    ;
    // pto: %d_begin_inline224__phi_v2
    ;
    int64_t v705;
    // pto: %d_end_inline301__phi_v2
    ;
    int64_t v706;
    // pto: %work_begin_inline50__phi_v3
    ;
    int64_t v707;
    // pto: %work_end_inline48__phi_v3
    ;
    int64_t v708;
    if (v24 < v25) {
      // pto: %332
      ;
      // pto: %d_worker_inline306__phi_v2
      ;
      int64_t v709;
      // pto: %d_workers_inline266__phi_v2
      ;
      int64_t v710;
      // pto: %pair_id_inline305__phi_v2
      ;
      int64_t v711;
      if (v698 < v26) {
        // pto: %333
        ;
        int64_t v712 = (int64_t) ((uint64_t) v27 + (uint64_t) v73);
        // pto: %334
        ;
        // pto: %336
        ;
        v709 = v698 % v712;
        v710 = v712;
        v711 = v698 / v712;
      } else {
        // pto: %338
        ;
        int64_t v713 = (int64_t) ((uint64_t) v698 - (uint64_t) v26);
        // pto: %340, %339
        ;
        // pto: %341
        ;
        v709 = v713 % v27;
        v710 = v27;
        v711 = (int64_t) ((uint64_t) v28 + (uint64_t) (v713 / v27));
      };
      // pto: %342
      ;
      uint64_t v714 = (uint64_t) v711;
      // pto: %342
      ;
      int64_t v715 = (int64_t) (v714 * (uint64_t) v85);
      // pto: %343
      ;
      int64_t v716 = v23 < v715 ? v23 : v715;
      // pto: %344
      ;
      int64_t v717 = (int64_t) ((uint64_t) v716 + (uint64_t) v85);
      // pto: %345
      ;
      // pto: %346
      ;
      int64_t v718 = v89 / v710;
      // pto: %347
      ;
      int64_t v719 = v89 % v710;
      // pto: %348
      ;
      uint64_t v720 = (uint64_t) v709;
      // pto: %349
      ;
      bool v721 = v709 < v719;
      // pto: %349
      ;
      int64_t v722 = v721 ? v709 : v719;
      // pto: %348, %350
      ;
      int64_t v723 = (int64_t) ((uint64_t) ((int64_t) v720 * (uint64_t) v718) + (uint64_t) v722);
      // pto: %352
      ;
      uint64_t v724 = (uint64_t) v709;
      // pto: %352
      ;
      int64_t v725 = (int64_t) ((uint64_t) v719 - v724);
      // pto: %353
      ;
      int64_t v726 = v725 < v86 ? v86 : v725;
      // pto: %354
      ;
      int64_t v727 = v726 < v73 ? v726 : v73;
      // pto: %351, %355
      ;
      v705 = v723;
      v706 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) v723 + (uint64_t) v718) + (uint64_t) v727);
      v707 = v716;
      v708 = v23 < v717 ? v23 : v717;
    } else {
      v705 = v86;
      v706 = v89;
      v707 = v700;
      v708 = v23 < v704 ? v23 : v704;
    };
    // pto: %356
    ;
    bool v728 = v25 <= v698;
    // pto: %work_begin_inline50__phi_v5
    ;
    int64_t v729 = v728 ? v86 : v707;
    // pto: %work_end_inline48__phi_v5
    ;
    int64_t v730 = v728 ? v86 : v708;
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID6);
    for (int64_t v731 = v729; v731 < v730; v731 += v85) {
      // pto: %357
      ;
      int64_t v732 = (int64_t) ((uint64_t) v730 - (uint64_t) v731);
      // pto: %358
      ;
      int64_t v733 = v732 < v85 ? v732 : v85;
      // pto: %359
      ;
      int64_t v734 = (int64_t) ((uint64_t) v733 * (uint64_t) v72);
      // pto: %m_parts_load_inline312__ssa_v0
      ;
      Tile<TileType::Vec, float, 11, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v735 = Tile<TileType::Vec, float, 11, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v77, v734);
      // pto: %m_parts_load_inline312__ssa_v0
      ;
      uint64_t v736 = (uint64_t) v40;
      TASSIGN(v735, v736);
      // pto: %360
      ;
      int64_t v737 = (int64_t) ((uint64_t) v731 * (uint64_t) v72);
      // pto: %361
      ;
      int64_t v738 = v737 < v86 ? v86 : v737;
      // pto: %363
      ;
      const int64_t v739 = 0;
      // pto: %363
      ;
      __gm__ float* v740 = PTOAS__GLOBAL_TENSOR_DATA(v142);
      // pto: %363
      ;
      const int64_t v741 = 1;
      // pto: %363
      ;
      const int64_t v742 = 1;
      // pto: %363
      ;
      const int64_t v743 = 1;
      // pto: %363
      ;
      int64_t v744 = v77 * v134;
      // pto: %363
      ;
      int64_t v745 = v743 * v744;
      // pto: %363
      ;
      pto::Shape<1, 1, 1, 11, -1> v746 = pto::Shape<1, 1, 1, 11, -1>(v741, v742, v743, v77, v734);
      // pto: %363
      ;
      pto::Stride<-1, -1, -1, -1, -1> v747 = pto::Stride<-1, -1, -1, -1, -1>(v742 * v745, v745, v744, v134, v73);
      // pto: %363
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 11, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v748 = GlobalTensor<float, pto::Shape<1, 1, 1, 11, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v740 + (v739 + v86 * v134 + v738 * v73), v746, v747);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID4);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID6);
      TLOAD(v735, v748);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID6);
      // pto: %t__tmp_v58
      ;
      Tile<TileType::Vec, float, 11, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null> v749 = Tile<TileType::Vec, float, 11, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>(v77, v88);
      // pto: %t__tmp_v58
      ;
      uint64_t v750 = (uint64_t) v40;
      TASSIGN(v749, v750);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID6);
      TFILLPAD<pto::TFillPadMode::InPlace>(v749, v735);
      v749.SetValidShape(v77, v88);
      // pto: %m_parts_tmp_inline313__ssa_v0
      ;
      Tile<TileType::Vec, float, 11, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v751 = Tile<TileType::Vec, float, 11, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v77, v88);
      // pto: %m_parts_tmp_inline313__ssa_v0
      ;
      uint64_t v752 = (uint64_t) v44;
      TASSIGN(v751, v752);
      // pto: %t__tmp_v59
      ;
      Tile<TileType::Vec, float, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v753 = Tile<TileType::Vec, float, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v88);
      // pto: %t__tmp_v59
      ;
      uint64_t v754 = (uint64_t) v40;
      TASSIGN(v753, v754);
      // pto: %364
      ;
      Tile<TileType::Vec, float, 11, 256, BLayout::RowMajor, 11, 256, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null> v755;
      TRESHAPE(v755, v749);
      // pto: %365
      ;
      Tile<TileType::Vec, float, 11, 256, BLayout::RowMajor, 11, 256, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v756;
      TRESHAPE(v756, v751);
      // pto: %366
      ;
      Tile<TileType::Vec, float, 1, 256, BLayout::RowMajor, 1, 256, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v757;
      TRESHAPE(v757, v753);
      pipe_barrier(PIPE_V);
      const bool v758 = true;
      TCOLSUM(v757, v755, v756, v758);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID7);
      // pto: %m_total_inline315__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v759 = Tile<TileType::Vec, float, 8, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v89, v72);
      // pto: %m_total_inline315__ssa_v0
      ;
      uint64_t v760 = (uint64_t) v40;
      TASSIGN(v759, v760);
      // pto: %m_sq_load_inline175__ssa_v0
      ;
      Tile<TileType::Vec, float, 11, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v761 = Tile<TileType::Vec, float, 11, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v77, v733);
      // pto: %m_sq_load_inline175__ssa_v0
      ;
      uint64_t v762 = (uint64_t) v45;
      TASSIGN(v761, v762);
      // pto: %367
      ;
      int64_t v763 = v731 < v86 ? v86 : v731;
      // pto: %368
      ;
      int64_t v764 = (int64_t) v29;
      // pto: %368
      ;
      const int64_t v765 = 0;
      // pto: %368
      ;
      __gm__ float* v766 = PTOAS__GLOBAL_TENSOR_DATA(v151);
      // pto: %368
      ;
      const int64_t v767 = 1;
      // pto: %368
      ;
      const int64_t v768 = 1;
      // pto: %368
      ;
      const int64_t v769 = 1;
      // pto: %368
      ;
      int64_t v770 = v77 * v764;
      // pto: %368
      ;
      int64_t v771 = v769 * v770;
      // pto: %368
      ;
      pto::Shape<1, 1, 1, 11, -1> v772 = pto::Shape<1, 1, 1, 11, -1>(v767, v768, v769, v77, v733);
      // pto: %368
      ;
      pto::Stride<-1, -1, -1, -1, -1> v773 = pto::Stride<-1, -1, -1, -1, -1>(v768 * v771, v771, v770, v764, v73);
      // pto: %368
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 11, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v774 = GlobalTensor<float, pto::Shape<1, 1, 1, 11, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v766 + (v765 + v86 * v764 + v763 * v73), v772, v773);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID7);
      TLOAD(v761, v774);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID7);
      // pto: %t__tmp_v60
      ;
      Tile<TileType::Vec, float, 11, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null> v775 = Tile<TileType::Vec, float, 11, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>(v77, v89);
      // pto: %t__tmp_v60
      ;
      uint64_t v776 = (uint64_t) v45;
      TASSIGN(v775, v776);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID7);
      TFILLPAD<pto::TFillPadMode::InPlace>(v775, v761);
      v775.SetValidShape(v77, v89);
      // pto: %m_sq_tmp_inline307__ssa_v0
      ;
      Tile<TileType::Vec, float, 11, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v777 = Tile<TileType::Vec, float, 11, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v77, v89);
      // pto: %m_sq_tmp_inline307__ssa_v0
      ;
      uint64_t v778 = (uint64_t) v46;
      TASSIGN(v777, v778);
      // pto: %m_rms_tmp_inline316__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v779 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v89);
      // pto: %m_rms_tmp_inline316__ssa_v0
      ;
      uint64_t v780 = (uint64_t) v47;
      TASSIGN(v779, v780);
      // pto: %t__tmp_v61
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v781 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v89);
      // pto: %t__tmp_v61
      ;
      uint64_t v782 = (uint64_t) v45;
      TASSIGN(v781, v782);
      // pto: %369
      ;
      Tile<TileType::Vec, float, 11, 8, BLayout::RowMajor, 11, 8, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null> v783;
      TRESHAPE(v783, v775);
      // pto: %370
      ;
      Tile<TileType::Vec, float, 11, 8, BLayout::RowMajor, 11, 8, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v784;
      TRESHAPE(v784, v777);
      // pto: %371
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, 1, 8, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v785;
      TRESHAPE(v785, v781);
      pipe_barrier(PIPE_V);
      const bool v786 = true;
      TCOLSUM(v785, v783, v784, v786);
      // pto: %t__tmp_v62
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v787 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v89);
      // pto: %t__tmp_v62
      ;
      uint64_t v788 = (uint64_t) v45;
      TASSIGN(v787, v788);
      pipe_barrier(PIPE_V);
      TMULS(v787, v781, v90);
      // pto: %t__tmp_v63
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v789 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v89);
      // pto: %t__tmp_v63
      ;
      uint64_t v790 = (uint64_t) v45;
      TASSIGN(v789, v790);
      pipe_barrier(PIPE_V);
      TADDS(v789, v787, v91);
      // pto: %t__tmp_v64
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v791 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v89);
      // pto: %t__tmp_v64
      ;
      uint64_t v792 = (uint64_t) v57;
      TASSIGN(v791, v792);
      pipe_barrier(PIPE_V);
      TRSQRT(v791, v789, v779);
      // pto: %m_inv_inline56__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v793 = Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v89, v73);
      // pto: %m_inv_inline56__ssa_v0
      ;
      uint64_t v794 = (uint64_t) v57;
      TASSIGN(v793, v794);
      // pto: %gate_base_inline282__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v795 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v89);
      // pto: %gate_base_inline282__ssa_v0
      ;
      uint64_t v796 = (uint64_t) v58;
      TASSIGN(v795, v796);
      // pto: %372
      ;
      __gm__ float* v797 = PTOAS__GLOBAL_TENSOR_DATA(v159);
      // pto: %372
      ;
      pto::Shape<1, 1, 1, 1, 8> v798 = pto::Shape<1, 1, 1, 1, 8>();
      // pto: %372
      ;
      pto::Stride<24, 24, 24, 24, 1> v799 = pto::Stride<24, 24, 24, 24, 1>();
      // pto: %372
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 1, 8>, pto::Stride<24, 24, 24, 24, 1>, pto::Layout::ND> v800 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 8>, pto::Stride<24, 24, 24, 24, 1>, pto::Layout::ND>(v797, v798, v799);
      TLOAD(v795, v800);
      // pto: %gate_scales_inline317__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v801 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v76);
      // pto: %gate_scales_inline317__ssa_v0
      ;
      uint64_t v802 = (uint64_t) v59;
      TASSIGN(v801, v802);
      // pto: %hc_scale_2d_inline304__ssa_v0_pview
      ;
      __gm__ float* v803 = PTOAS__GLOBAL_TENSOR_DATA(v176);
      // pto: %hc_scale_2d_inline304__ssa_v0_pview
      ;
      pto::Shape<1, 1, 1, 1, 3> v804 = pto::Shape<1, 1, 1, 1, 3>();
      // pto: %hc_scale_2d_inline304__ssa_v0_pview
      ;
      pto::Stride<3, 3, 3, 3, 1> v805 = pto::Stride<3, 3, 3, 3, 1>();
      // pto: %hc_scale_2d_inline304__ssa_v0_pview
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 1, 3>, pto::Stride<3, 3, 3, 3, 1>, pto::Layout::ND> v806 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 3>, pto::Stride<3, 3, 3, 3, 1>, pto::Layout::ND>(v803, v804, v805);
      TLOAD(v801, v806);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      // pto: %t__ci_tmp_v1
      ;
      Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v807 = Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v92);
      // pto: %t__ci_tmp_v1
      ;
      uint64_t v808 = (uint64_t) v45;
      TASSIGN(v807, v808);
      // pto: %t__tmp_v65
      ;
      Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v809 = Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v89);
      // pto: %t__tmp_v65
      ;
      uint64_t v810 = (uint64_t) v60;
      TASSIGN(v809, v810);
      // pto: %373
      ;
      Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, 1, 192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v811;
      TRESHAPE(v811, v807);
      // pto: %374
      ;
      Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, 1, 8, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v812;
      TRESHAPE(v812, v809);
      pipe_barrier(PIPE_V);
      TCI<Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, 1, 8, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, 1, 192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, int32_t, 0>(v812, v80, v811);
      // pto: %gate_ids_inline292__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v813 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v89);
      // pto: %gate_ids_inline292__ssa_v0
      ;
      uint64_t v814 = (uint64_t) v45;
      TASSIGN(v813, v814);
      pipe_barrier(PIPE_V);
      RoundMode v815 = RoundMode::CAST_ROUND;
      SaturationMode v816 = SaturationMode::ON;
      TCVT(v813, v809, v815, v816);
      // pto: %t__tmp_v66
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v817 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v89);
      // pto: %t__tmp_v66
      ;
      uint64_t v818 = (uint64_t) v45;
      TASSIGN(v817, v818);
      pipe_barrier(PIPE_V);
      TMULS(v817, v813, v96);
      // pto: %gate_scale_ids_inline318__ssa_v0
      ;
      Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v819 = Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v89);
      // pto: %gate_scale_ids_inline318__ssa_v0
      ;
      uint64_t v820 = (uint64_t) v45;
      TASSIGN(v819, v820);
      pipe_barrier(PIPE_V);
      RoundMode v821 = RoundMode::CAST_FLOOR;
      SaturationMode v822 = SaturationMode::ON;
      TCVT(v819, v817, v821, v822);
      // pto: %gate_tmp_inline319__ssa_v0
      ;
      Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v823 = Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v89);
      // pto: %gate_tmp_inline319__ssa_v0
      ;
      uint64_t v824 = (uint64_t) v61;
      TASSIGN(v823, v824);
      // pto: %gate_scale_vec_inline320__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v825 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v89);
      // pto: %gate_scale_vec_inline320__ssa_v0
      ;
      uint64_t v826 = (uint64_t) v48;
      TASSIGN(v825, v826);
      pipe_barrier(PIPE_V);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      TGATHER(v825, v801, v819, v823);
      // pto: %t__tmp_v67
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v827 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v89, v89);
      // pto: %t__tmp_v67
      ;
      uint64_t v828 = (uint64_t) v40;
      TASSIGN(v827, v828);
      // pto: %375
      ;
      Tile<TileType::Vec, float, 8, 32, BLayout::RowMajor, 8, 8, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v829;
      // pto: %375
      ;
      uint64_t v830 = (uint64_t) v40;
      TASSIGN(v829, v830);
      // pto: %gate_raw_inline62__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v831 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v89, v89);
      // pto: %gate_raw_inline62__ssa_v0
      ;
      uint64_t v832 = (uint64_t) v40;
      TASSIGN(v831, v832);
      TADDS(v831, v829, v82);
      // pto: %gate_norm_inline269__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v833 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v89, v89);
      // pto: %gate_norm_inline269__ssa_v0
      ;
      uint64_t v834 = (uint64_t) v40;
      TASSIGN(v833, v834);
      pipe_barrier(PIPE_V);
      TROWEXPANDMUL(v833, v831, v793);
      // pto: %gate_scaled_inline321__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v835 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v89, v89);
      // pto: %gate_scaled_inline321__ssa_v0
      ;
      uint64_t v836 = (uint64_t) v40;
      TASSIGN(v835, v836);
      pipe_barrier(PIPE_V);
      TCOLEXPANDMUL(v835, v833, v825);
      // pto: %t__tmp_v68
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v837 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v89, v89);
      // pto: %t__tmp_v68
      ;
      uint64_t v838 = (uint64_t) v54;
      TASSIGN(v837, v838);
      TCOLEXPAND(v837, v795);
      // pto: %gate_logits_inline119__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v839 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v89, v89);
      // pto: %gate_logits_inline119__ssa_v0
      ;
      uint64_t v840 = (uint64_t) v40;
      TASSIGN(v839, v840);
      pipe_barrier(PIPE_V);
      TADD(v839, v835, v837);
      // pto: %t__tmp_v69
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v841 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v89, v89);
      // pto: %t__tmp_v69
      ;
      uint64_t v842 = (uint64_t) v40;
      TASSIGN(v841, v842);
      pipe_barrier(PIPE_V);
      TNEG(v841, v839);
      // pto: %t__tmp_v70
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v843 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v89, v89);
      // pto: %t__tmp_v70
      ;
      uint64_t v844 = (uint64_t) v40;
      TASSIGN(v843, v844);
      pipe_barrier(PIPE_V);
      TEXP(v843, v841);
      // pto: %t__tmp_v71
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v845 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v89, v89);
      // pto: %t__tmp_v71
      ;
      uint64_t v846 = (uint64_t) v40;
      TASSIGN(v845, v846);
      pipe_barrier(PIPE_V);
      TADDS(v845, v843, v93);
      // pto: %gate_sigmoid_inline90__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v847 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v89, v89);
      // pto: %gate_sigmoid_inline90__ssa_v0
      ;
      uint64_t v848 = (uint64_t) v51;
      TASSIGN(v847, v848);
      pipe_barrier(PIPE_V);
      TRECIP(v847, v845);
      // pto: %transpose_tmp
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v849 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v89, v89);
      // pto: %transpose_tmp
      ;
      uint64_t v850 = (uint64_t) v40;
      TASSIGN(v849, v850);
      // pto: %gate_transposed_inline44__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v851 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v89, v89);
      // pto: %gate_transposed_inline44__ssa_v0
      ;
      uint64_t v852 = (uint64_t) v45;
      TASSIGN(v851, v852);
      pipe_barrier(PIPE_V);
      TTRANS(v851, v847, v849);
      // pto: %t__ci_tmp_v2
      ;
      Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v853 = Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v92);
      // pto: %t__ci_tmp_v2
      ;
      uint64_t v854 = (uint64_t) v40;
      TASSIGN(v853, v854);
      // pto: %t__tmp_v72
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v855 = Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v72);
      // pto: %t__tmp_v72
      ;
      uint64_t v856 = (uint64_t) v52;
      TASSIGN(v855, v856);
      // pto: %376
      ;
      Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, 1, 192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v857;
      TRESHAPE(v857, v853);
      // pto: %377
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, 1, 32, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v858;
      TRESHAPE(v858, v855);
      TCI<Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, 1, 32, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, 1, 192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, int32_t, 0>(v858, v80, v857);
      // pto: %post_ids_float_inline43__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v859 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v72);
      // pto: %post_ids_float_inline43__ssa_v0
      ;
      uint64_t v860 = (uint64_t) v40;
      TASSIGN(v859, v860);
      pipe_barrier(PIPE_V);
      RoundMode v861 = RoundMode::CAST_ROUND;
      SaturationMode v862 = SaturationMode::ON;
      TCVT(v859, v855, v861, v862);
      // pto: %t__tmp_v73
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v863 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v72);
      // pto: %t__tmp_v73
      ;
      uint64_t v864 = (uint64_t) v53;
      TASSIGN(v863, v864);
      pipe_barrier(PIPE_V);
      TMULS(v863, v859, v96);
      // pto: %t__tmp_v74
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v865 = Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v72);
      // pto: %t__tmp_v74
      ;
      uint64_t v866 = (uint64_t) v53;
      TASSIGN(v865, v866);
      pipe_barrier(PIPE_V);
      RoundMode v867 = RoundMode::CAST_FLOOR;
      SaturationMode v868 = SaturationMode::ON;
      TCVT(v865, v863, v867, v868);
      // pto: %post_rows_inline42__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v869 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v72);
      // pto: %post_rows_inline42__ssa_v0
      ;
      uint64_t v870 = (uint64_t) v53;
      TASSIGN(v869, v870);
      pipe_barrier(PIPE_V);
      RoundMode v871 = RoundMode::CAST_ROUND;
      SaturationMode v872 = SaturationMode::ON;
      TCVT(v869, v865, v871, v872);
      // pto: %t__tmp_v75
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v873 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v72);
      // pto: %t__tmp_v75
      ;
      uint64_t v874 = (uint64_t) v53;
      TASSIGN(v873, v874);
      pipe_barrier(PIPE_V);
      TMULS(v873, v869, v97);
      // pto: %t__tmp_v76
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v875 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v72);
      // pto: %t__tmp_v76
      ;
      uint64_t v876 = (uint64_t) v40;
      TASSIGN(v875, v876);
      pipe_barrier(PIPE_V);
      TADD(v875, v859, v873);
      // pto: %t__tmp_v77
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v877 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v72);
      // pto: %t__tmp_v77
      ;
      uint64_t v878 = (uint64_t) v40;
      TASSIGN(v877, v878);
      pipe_barrier(PIPE_V);
      TADDS(v877, v875, v97);
      // pto: %post_ids_inline274__ssa_v0
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v879 = Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v72);
      // pto: %post_ids_inline274__ssa_v0
      ;
      uint64_t v880 = (uint64_t) v40;
      TASSIGN(v879, v880);
      pipe_barrier(PIPE_V);
      RoundMode v881 = RoundMode::CAST_ROUND;
      SaturationMode v882 = SaturationMode::ON;
      TCVT(v879, v877, v881, v882);
      // pto: %post_tmp_inline41__ssa_v0
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v883 = Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v72);
      // pto: %post_tmp_inline41__ssa_v0
      ;
      uint64_t v884 = (uint64_t) v53;
      TASSIGN(v883, v884);
      // pto: %t__tmp_v78
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v885 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v113);
      // pto: %t__tmp_v78
      ;
      uint64_t v886 = (uint64_t) v51;
      TASSIGN(v885, v886);
      // pto: %post_values_inline40__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v887 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v72);
      // pto: %post_values_inline40__ssa_v0
      ;
      uint64_t v888 = (uint64_t) v54;
      TASSIGN(v887, v888);
      pipe_barrier(PIPE_V);
      TGATHER(v887, v885, v879, v883);
      // pto: %t__tmp_v79
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v889 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v72);
      // pto: %t__tmp_v79
      ;
      uint64_t v890 = (uint64_t) v40;
      TASSIGN(v889, v890);
      pipe_barrier(PIPE_V);
      TMULS(v889, v887, v95);
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID6);
      // pto: %378
      ;
      int64_t v891 = (int64_t) ((uint64_t) v733 * (uint64_t) v102);
      v889.SetValidShape(v73, v891);
      // pto: %379
      ;
      bool v892 = v705 == v86;
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID6);
      if (v892) {
        // pto: %381
        ;
        int64_t v893 = (int64_t) ((uint64_t) v731 * (uint64_t) v102);
        // pto: %382
        ;
        int64_t v894 = v893 < v86 ? v86 : v893;
        // pto: %post_flat_inline84__ssa_v0_pview
        ;
        const int64_t v895 = 0;
        // pto: %post_flat_inline84__ssa_v0_pview
        ;
        __gm__ float* v896 = PTOAS__GLOBAL_TENSOR_DATA(v185);
        // pto: %post_flat_inline84__ssa_v0_pview
        ;
        const int64_t v897 = 1;
        // pto: %post_flat_inline84__ssa_v0_pview
        ;
        const int64_t v898 = 1;
        // pto: %post_flat_inline84__ssa_v0_pview
        ;
        const int64_t v899 = 1;
        // pto: %post_flat_inline84__ssa_v0_pview
        ;
        int64_t v900 = v73 * v177;
        // pto: %post_flat_inline84__ssa_v0_pview
        ;
        int64_t v901 = v899 * v900;
        // pto: %post_flat_inline84__ssa_v0_pview
        ;
        pto::Shape<1, 1, 1, 1, -1> v902 = pto::Shape<1, 1, 1, 1, -1>(v897, v898, v899, v73, v891);
        // pto: %post_flat_inline84__ssa_v0_pview
        ;
        pto::Stride<-1, -1, -1, -1, -1> v903 = pto::Stride<-1, -1, -1, -1, -1>(v898 * v901, v901, v900, v177, v73);
        // pto: %post_flat_inline84__ssa_v0_pview
        ;
        GlobalTensor<float, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v904 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v896 + (v895 + v86 * v177 + v894 * v73), v902, v903);
        TSTORE(v904, v889);
      };
      set_flag(PIPE_MTE3, PIPE_V, EVENT_ID4);
      // pto: %t__tmp_v80
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v905 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v89);
      // pto: %t__tmp_v80
      ;
      uint64_t v906 = (uint64_t) v45;
      TASSIGN(v905, v906);
      // pto: %383
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, 1, 8, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v907;
      // pto: %383
      ;
      uint64_t v908 = (uint64_t) v45;
      TASSIGN(v907, v908);
      // pto: %t__tmp_v81
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v909 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v89);
      // pto: %t__tmp_v81
      ;
      uint64_t v910 = (uint64_t) v62;
      TASSIGN(v909, v910);
      TADDS(v909, v907, v91);
      // pto: %pre0_inline9__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v911 = Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v89, v73);
      // pto: %pre0_inline9__ssa_v0
      ;
      uint64_t v912 = (uint64_t) v62;
      TASSIGN(v911, v912);
      // pto: %t__tmp_v82
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v913 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v89);
      // pto: %t__tmp_v82
      ;
      uint64_t v914 = (uint64_t) v61;
      TASSIGN(v913, v914);
      // pto: %384
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, 1, 8, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v915;
      // pto: %384
      ;
      uint64_t v916 = (uint64_t) v61;
      TASSIGN(v915, v916);
      // pto: %t__tmp_v83
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v917 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v89);
      // pto: %t__tmp_v83
      ;
      uint64_t v918 = (uint64_t) v63;
      TASSIGN(v917, v918);
      TADDS(v917, v915, v91);
      // pto: %pre1_inline8__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v919 = Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v89, v73);
      // pto: %pre1_inline8__ssa_v0
      ;
      uint64_t v920 = (uint64_t) v63;
      TASSIGN(v919, v920);
      // pto: %t__tmp_v84
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v921 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v89);
      // pto: %t__tmp_v84
      ;
      uint64_t v922 = (uint64_t) v48;
      TASSIGN(v921, v922);
      // pto: %385
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, 1, 8, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v923;
      // pto: %385
      ;
      uint64_t v924 = (uint64_t) v48;
      TASSIGN(v923, v924);
      // pto: %t__tmp_v85
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v925 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v89);
      // pto: %t__tmp_v85
      ;
      uint64_t v926 = (uint64_t) v64;
      TASSIGN(v925, v926);
      TADDS(v925, v923, v91);
      // pto: %pre2_inline147__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v927 = Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v89, v73);
      // pto: %pre2_inline147__ssa_v0
      ;
      uint64_t v928 = (uint64_t) v64;
      TASSIGN(v927, v928);
      // pto: %t__tmp_v86
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v929 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v89);
      // pto: %t__tmp_v86
      ;
      uint64_t v930 = (uint64_t) v65;
      TASSIGN(v929, v930);
      // pto: %386
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, 1, 8, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v931;
      // pto: %386
      ;
      uint64_t v932 = (uint64_t) v65;
      TASSIGN(v931, v932);
      // pto: %t__tmp_v87
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v933 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v89);
      // pto: %t__tmp_v87
      ;
      uint64_t v934 = (uint64_t) v66;
      TASSIGN(v933, v934);
      TADDS(v933, v931, v91);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID6);
      // pto: %pre3_inline7__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v935 = Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v89, v73);
      // pto: %pre3_inline7__ssa_v0
      ;
      uint64_t v936 = (uint64_t) v66;
      TASSIGN(v935, v936);
      // pto: %387
      ;
      uint64_t v937 = (uint64_t) v706;
      // pto: %387
      ;
      uint64_t v938 = (uint64_t) v705;
      // pto: %390
      ;
      uint64_t v939 = (uint64_t) v705;
      // pto: %387, %388, %389, %390
      ;
      int64_t v940 = (int64_t) (v939 + (uint64_t) ((int64_t) (uint64_t) ((int64_t) (v937 - v938) / v85) * (uint64_t) v85));
      wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID4);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
      for (int64_t v941 = v705; v941 < v940; v941 += v85) {
        // pto: %391
        ;
        int64_t v942 = (int64_t) ((uint64_t) v941 * (uint64_t) v87);
        // pto: %394
        ;
        int64_t v943 = (int64_t) ((uint64_t) v942 + (uint64_t) v87);
        // pto: %x0_bf16_inline34__ssa_v0
        ;
        Tile<TileType::Vec, bfloat16_t, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v944 = Tile<TileType::Vec, bfloat16_t, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v733, v87);
        // pto: %x0_bf16_inline34__ssa_v0
        ;
        uint64_t v945 = (uint64_t) v67;
        TASSIGN(v944, v945);
        // pto: %397
        ;
        int64_t v946 = v942 < v86 ? v86 : v942;
        // pto: %398
        ;
        const int64_t v947 = 0;
        // pto: %398
        ;
        __gm__ bfloat16_t* v948 = PTOAS__GLOBAL_TENSOR_DATA(v133);
        // pto: %398
        ;
        const int64_t v949 = 1;
        // pto: %398
        ;
        const int64_t v950 = 1;
        // pto: %398
        ;
        const int64_t v951 = 1;
        // pto: %398
        ;
        int64_t v952 = v733 * v75;
        // pto: %398
        ;
        int64_t v953 = v951 * v952;
        // pto: %398
        ;
        pto::Shape<1, 1, 1, -1, 512> v954 = pto::Shape<1, 1, 1, -1, 512>(v949, v950, v951, v733, v87);
        // pto: %398
        ;
        pto::Stride<-1, -1, -1, -1, -1> v955 = pto::Stride<-1, -1, -1, -1, -1>(v950 * v953, v953, v952, v75, v73);
        // pto: %398
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v956 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v948 + (v947 + v763 * v75 + v946 * v73), v954, v955);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        TLOAD(v944, v956);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        // pto: %x1_bf16_inline14__ssa_v0
        ;
        Tile<TileType::Vec, bfloat16_t, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v957 = Tile<TileType::Vec, bfloat16_t, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v733, v87);
        // pto: %x1_bf16_inline14__ssa_v0
        ;
        uint64_t v958 = (uint64_t) v38;
        TASSIGN(v957, v958);
        // pto: %400
        ;
        int64_t v959 = (int64_t) ((uint64_t) v942 + (uint64_t) v79);
        // pto: %401
        ;
        int64_t v960 = v959 < v86 ? v86 : v959;
        // pto: %402
        ;
        const int64_t v961 = 0;
        // pto: %402
        ;
        __gm__ bfloat16_t* v962 = PTOAS__GLOBAL_TENSOR_DATA(v133);
        // pto: %402
        ;
        const int64_t v963 = 1;
        // pto: %402
        ;
        const int64_t v964 = 1;
        // pto: %402
        ;
        const int64_t v965 = 1;
        // pto: %402
        ;
        int64_t v966 = v733 * v75;
        // pto: %402
        ;
        int64_t v967 = v965 * v966;
        // pto: %402
        ;
        pto::Shape<1, 1, 1, -1, 512> v968 = pto::Shape<1, 1, 1, -1, 512>(v963, v964, v965, v733, v87);
        // pto: %402
        ;
        pto::Stride<-1, -1, -1, -1, -1> v969 = pto::Stride<-1, -1, -1, -1, -1>(v964 * v967, v967, v966, v75, v73);
        // pto: %402
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v970 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v962 + (v961 + v763 * v75 + v960 * v73), v968, v969);
        TLOAD(v957, v970);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        // pto: %x2_bf16_inline145__ssa_v0
        ;
        Tile<TileType::Vec, bfloat16_t, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v971 = Tile<TileType::Vec, bfloat16_t, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v733, v87);
        // pto: %x2_bf16_inline145__ssa_v0
        ;
        uint64_t v972 = (uint64_t) v68;
        TASSIGN(v971, v972);
        // pto: %404
        ;
        int64_t v973 = (int64_t) ((uint64_t) v942 + (uint64_t) v114);
        // pto: %405
        ;
        int64_t v974 = v973 < v86 ? v86 : v973;
        // pto: %406
        ;
        const int64_t v975 = 0;
        // pto: %406
        ;
        __gm__ bfloat16_t* v976 = PTOAS__GLOBAL_TENSOR_DATA(v133);
        // pto: %406
        ;
        const int64_t v977 = 1;
        // pto: %406
        ;
        const int64_t v978 = 1;
        // pto: %406
        ;
        const int64_t v979 = 1;
        // pto: %406
        ;
        int64_t v980 = v733 * v75;
        // pto: %406
        ;
        int64_t v981 = v979 * v980;
        // pto: %406
        ;
        pto::Shape<1, 1, 1, -1, 512> v982 = pto::Shape<1, 1, 1, -1, 512>(v977, v978, v979, v733, v87);
        // pto: %406
        ;
        pto::Stride<-1, -1, -1, -1, -1> v983 = pto::Stride<-1, -1, -1, -1, -1>(v978 * v981, v981, v980, v75, v73);
        // pto: %406
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v984 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v976 + (v975 + v763 * v75 + v974 * v73), v982, v983);
        TLOAD(v971, v984);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
        // pto: %x3_bf16_inline5__ssa_v0
        ;
        Tile<TileType::Vec, bfloat16_t, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v985 = Tile<TileType::Vec, bfloat16_t, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v733, v87);
        // pto: %x3_bf16_inline5__ssa_v0
        ;
        uint64_t v986 = (uint64_t) v37;
        TASSIGN(v985, v986);
        // pto: %408
        ;
        int64_t v987 = (int64_t) ((uint64_t) v942 + (uint64_t) v115);
        // pto: %409
        ;
        int64_t v988 = v987 < v86 ? v86 : v987;
        // pto: %410
        ;
        const int64_t v989 = 0;
        // pto: %410
        ;
        __gm__ bfloat16_t* v990 = PTOAS__GLOBAL_TENSOR_DATA(v133);
        // pto: %410
        ;
        const int64_t v991 = 1;
        // pto: %410
        ;
        const int64_t v992 = 1;
        // pto: %410
        ;
        const int64_t v993 = 1;
        // pto: %410
        ;
        int64_t v994 = v733 * v75;
        // pto: %410
        ;
        int64_t v995 = v993 * v994;
        // pto: %410
        ;
        pto::Shape<1, 1, 1, -1, 512> v996 = pto::Shape<1, 1, 1, -1, 512>(v991, v992, v993, v733, v87);
        // pto: %410
        ;
        pto::Stride<-1, -1, -1, -1, -1> v997 = pto::Stride<-1, -1, -1, -1, -1>(v992 * v995, v995, v994, v75, v73);
        // pto: %410
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v998 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v990 + (v989 + v763 * v75 + v988 * v73), v996, v997);
        TLOAD(v985, v998);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
        // pto: %20
        ;
        Tile<TileType::Vec, bfloat16_t, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v999 = Tile<TileType::Vec, bfloat16_t, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v733, v87);
        // pto: %20
        ;
        uint64_t v1000 = (uint64_t) v69;
        TASSIGN(v999, v1000);
        // pto: %412
        ;
        int64_t v1001 = v943 < v86 ? v86 : v943;
        // pto: %413
        ;
        const int64_t v1002 = 0;
        // pto: %413
        ;
        __gm__ bfloat16_t* v1003 = PTOAS__GLOBAL_TENSOR_DATA(v133);
        // pto: %413
        ;
        const int64_t v1004 = 1;
        // pto: %413
        ;
        const int64_t v1005 = 1;
        // pto: %413
        ;
        const int64_t v1006 = 1;
        // pto: %413
        ;
        int64_t v1007 = v733 * v75;
        // pto: %413
        ;
        int64_t v1008 = v1006 * v1007;
        // pto: %413
        ;
        pto::Shape<1, 1, 1, -1, 512> v1009 = pto::Shape<1, 1, 1, -1, 512>(v1004, v1005, v1006, v733, v87);
        // pto: %413
        ;
        pto::Stride<-1, -1, -1, -1, -1> v1010 = pto::Stride<-1, -1, -1, -1, -1>(v1005 * v1008, v1008, v1007, v75, v73);
        // pto: %413
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v1011 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v1003 + (v1002 + v763 * v75 + v1001 * v73), v1009, v1010);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
        TLOAD(v999, v1011);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);
        // pto: %21
        ;
        Tile<TileType::Vec, bfloat16_t, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v1012 = Tile<TileType::Vec, bfloat16_t, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v733, v87);
        // pto: %21
        ;
        uint64_t v1013 = (uint64_t) v39;
        TASSIGN(v1012, v1013);
        int64_t v1014 = (int64_t) ((uint64_t) v942 + (uint64_t) v36);
        // pto: %416
        ;
        int64_t v1015 = v1014 < v86 ? v86 : v1014;
        // pto: %417
        ;
        const int64_t v1016 = 0;
        // pto: %417
        ;
        __gm__ bfloat16_t* v1017 = PTOAS__GLOBAL_TENSOR_DATA(v133);
        // pto: %417
        ;
        const int64_t v1018 = 1;
        // pto: %417
        ;
        const int64_t v1019 = 1;
        // pto: %417
        ;
        const int64_t v1020 = 1;
        // pto: %417
        ;
        int64_t v1021 = v733 * v75;
        // pto: %417
        ;
        int64_t v1022 = v1020 * v1021;
        // pto: %417
        ;
        pto::Shape<1, 1, 1, -1, 512> v1023 = pto::Shape<1, 1, 1, -1, 512>(v1018, v1019, v1020, v733, v87);
        // pto: %417
        ;
        pto::Stride<-1, -1, -1, -1, -1> v1024 = pto::Stride<-1, -1, -1, -1, -1>(v1019 * v1022, v1022, v1021, v75, v73);
        // pto: %417
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v1025 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v1017 + (v1016 + v763 * v75 + v1015 * v73), v1023, v1024);
        TLOAD(v1012, v1025);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID5);
        // pto: %22
        ;
        Tile<TileType::Vec, bfloat16_t, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v1026 = Tile<TileType::Vec, bfloat16_t, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v733, v87);
        // pto: %22
        ;
        uint64_t v1027 = (uint64_t) v70;
        TASSIGN(v1026, v1027);
        int64_t v1028 = (int64_t) ((uint64_t) v942 + (uint64_t) v35);
        // pto: %420
        ;
        int64_t v1029 = v1028 < v86 ? v86 : v1028;
        // pto: %421
        ;
        const int64_t v1030 = 0;
        // pto: %421
        ;
        __gm__ bfloat16_t* v1031 = PTOAS__GLOBAL_TENSOR_DATA(v133);
        // pto: %421
        ;
        const int64_t v1032 = 1;
        // pto: %421
        ;
        const int64_t v1033 = 1;
        // pto: %421
        ;
        const int64_t v1034 = 1;
        // pto: %421
        ;
        int64_t v1035 = v733 * v75;
        // pto: %421
        ;
        int64_t v1036 = v1034 * v1035;
        // pto: %421
        ;
        pto::Shape<1, 1, 1, -1, 512> v1037 = pto::Shape<1, 1, 1, -1, 512>(v1032, v1033, v1034, v733, v87);
        // pto: %421
        ;
        pto::Stride<-1, -1, -1, -1, -1> v1038 = pto::Stride<-1, -1, -1, -1, -1>(v1033 * v1036, v1036, v1035, v75, v73);
        // pto: %421
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v1039 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v1031 + (v1030 + v763 * v75 + v1029 * v73), v1037, v1038);
        TLOAD(v1026, v1039);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID6);
        // pto: %23
        ;
        Tile<TileType::Vec, bfloat16_t, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v1040 = Tile<TileType::Vec, bfloat16_t, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v733, v87);
        // pto: %23
        ;
        uint64_t v1041 = (uint64_t) v71;
        TASSIGN(v1040, v1041);
        int64_t v1042 = (int64_t) ((uint64_t) v942 + (uint64_t) v34);
        // pto: %424
        ;
        int64_t v1043 = v1042 < v86 ? v86 : v1042;
        // pto: %425
        ;
        const int64_t v1044 = 0;
        // pto: %425
        ;
        __gm__ bfloat16_t* v1045 = PTOAS__GLOBAL_TENSOR_DATA(v133);
        // pto: %425
        ;
        const int64_t v1046 = 1;
        // pto: %425
        ;
        const int64_t v1047 = 1;
        // pto: %425
        ;
        const int64_t v1048 = 1;
        // pto: %425
        ;
        int64_t v1049 = v733 * v75;
        // pto: %425
        ;
        int64_t v1050 = v1048 * v1049;
        // pto: %425
        ;
        pto::Shape<1, 1, 1, -1, 512> v1051 = pto::Shape<1, 1, 1, -1, 512>(v1046, v1047, v1048, v733, v87);
        // pto: %425
        ;
        pto::Stride<-1, -1, -1, -1, -1> v1052 = pto::Stride<-1, -1, -1, -1, -1>(v1047 * v1050, v1050, v1049, v75, v73);
        // pto: %425
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v1053 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v1045 + (v1044 + v763 * v75 + v1043 * v73), v1051, v1052);
        TLOAD(v1040, v1053);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID7);
        // pto: %x0_inline36__ssa_v0
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v1054 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v733, v87);
        // pto: %x0_inline36__ssa_v0
        ;
        uint64_t v1055 = (uint64_t) v40;
        TASSIGN(v1054, v1055);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        pipe_barrier(PIPE_V);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID5);
        RoundMode v1056 = RoundMode::CAST_ROUND;
        SaturationMode v1057 = SaturationMode::ON;
        TCVT(v1054, v944, v1056, v1057);
        // pto: %x1_inline182__ssa_v0
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v1058 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v733, v87);
        // pto: %x1_inline182__ssa_v0
        ;
        uint64_t v1059 = (uint64_t) v67;
        TASSIGN(v1058, v1059);
        pipe_barrier(PIPE_V);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        RoundMode v1060 = RoundMode::CAST_ROUND;
        SaturationMode v1061 = SaturationMode::ON;
        TCVT(v1058, v957, v1060, v1061);
        // pto: %x2_inline64__ssa_v0
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v1062 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v733, v87);
        // pto: %x2_inline64__ssa_v0
        ;
        uint64_t v1063 = (uint64_t) v38;
        TASSIGN(v1062, v1063);
        pipe_barrier(PIPE_V);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
        RoundMode v1064 = RoundMode::CAST_ROUND;
        SaturationMode v1065 = SaturationMode::ON;
        TCVT(v1062, v971, v1064, v1065);
        // pto: %x3_inline4__ssa_v0
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v1066 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v733, v87);
        // pto: %x3_inline4__ssa_v0
        ;
        uint64_t v1067 = (uint64_t) v68;
        TASSIGN(v1066, v1067);
        pipe_barrier(PIPE_V);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
        RoundMode v1068 = RoundMode::CAST_ROUND;
        SaturationMode v1069 = SaturationMode::ON;
        TCVT(v1066, v985, v1068, v1069);
        // pto: %y0_inline242__ssa_v0
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v1070 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v733, v87);
        // pto: %y0_inline242__ssa_v0
        ;
        uint64_t v1071 = (uint64_t) v40;
        TASSIGN(v1070, v1071);
        TROWEXPANDMUL(v1070, v1054, v911);
        // pto: %y1_inline69__ssa_v0
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v1072 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v733, v87);
        // pto: %y1_inline69__ssa_v0
        ;
        uint64_t v1073 = (uint64_t) v67;
        TASSIGN(v1072, v1073);
        TROWEXPANDMUL(v1072, v1058, v919);
        // pto: %y2_inline97__ssa_v0
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v1074 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v733, v87);
        // pto: %y2_inline97__ssa_v0
        ;
        uint64_t v1075 = (uint64_t) v38;
        TASSIGN(v1074, v1075);
        TROWEXPANDMUL(v1074, v1062, v927);
        // pto: %y3_inline3__ssa_v0
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v1076 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v733, v87);
        // pto: %y3_inline3__ssa_v0
        ;
        uint64_t v1077 = (uint64_t) v68;
        TASSIGN(v1076, v1077);
        pipe_barrier(PIPE_V);
        TROWEXPANDMUL(v1076, v1066, v935);
        // pto: %t__tmp_v88
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v1078 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v733, v87);
        // pto: %t__tmp_v88
        ;
        uint64_t v1079 = (uint64_t) v40;
        TASSIGN(v1078, v1079);
        TADD(v1078, v1070, v1072);
        // pto: %t__tmp_v89
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v1080 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v733, v87);
        // pto: %t__tmp_v89
        ;
        uint64_t v1081 = (uint64_t) v67;
        TASSIGN(v1080, v1081);
        pipe_barrier(PIPE_V);
        TADD(v1080, v1074, v1076);
        // pto: %y_tile_inline2__ssa_v0
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v1082 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v733, v87);
        // pto: %y_tile_inline2__ssa_v0
        ;
        uint64_t v1083 = (uint64_t) v40;
        TASSIGN(v1082, v1083);
        pipe_barrier(PIPE_V);
        TADD(v1082, v1078, v1080);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        // pto: %y_bf16_inline1__ssa_v0
        ;
        Tile<TileType::Vec, bfloat16_t, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v1084 = Tile<TileType::Vec, bfloat16_t, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v733, v87);
        // pto: %y_bf16_inline1__ssa_v0
        ;
        uint64_t v1085 = (uint64_t) v40;
        TASSIGN(v1084, v1085);
        pipe_barrier(PIPE_V);
        RoundMode v1086 = RoundMode::CAST_RINT;
        SaturationMode v1087 = SaturationMode::ON;
        TCVT(v1084, v1082, v1086, v1087);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID7);
        v1084.SetValidShape(v733, v87);
        // pto: %mixed_view_inline92__ssa_v0_pview
        ;
        const int64_t v1088 = 0;
        // pto: %mixed_view_inline92__ssa_v0_pview
        ;
        __gm__ bfloat16_t* v1089 = PTOAS__GLOBAL_TENSOR_DATA(v194);
        // pto: %mixed_view_inline92__ssa_v0_pview
        ;
        const int64_t v1090 = 1;
        // pto: %mixed_view_inline92__ssa_v0_pview
        ;
        const int64_t v1091 = 1;
        // pto: %mixed_view_inline92__ssa_v0_pview
        ;
        const int64_t v1092 = 1;
        // pto: %mixed_view_inline92__ssa_v0_pview
        ;
        int64_t v1093 = v733 * v79;
        // pto: %mixed_view_inline92__ssa_v0_pview
        ;
        int64_t v1094 = v1092 * v1093;
        // pto: %mixed_view_inline92__ssa_v0_pview
        ;
        pto::Shape<1, 1, 1, -1, 512> v1095 = pto::Shape<1, 1, 1, -1, 512>(v1090, v1091, v1092, v733, v87);
        // pto: %mixed_view_inline92__ssa_v0_pview
        ;
        pto::Stride<-1, -1, -1, -1, -1> v1096 = pto::Stride<-1, -1, -1, -1, -1>(v1091 * v1094, v1094, v1093, v79, v73);
        // pto: %mixed_view_inline92__ssa_v0_pview
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v1097 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v1089 + (v1088 + v763 * v79 + v946 * v73), v1095, v1096);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID7);
        pipe_barrier(PIPE_MTE3);
        TSTORE(v1097, v1084);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID5);
        // pto: %24
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v1098 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v733, v87);
        // pto: %24
        ;
        uint64_t v1099 = (uint64_t) v42;
        TASSIGN(v1098, v1099);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID6);
        RoundMode v1100 = RoundMode::CAST_ROUND;
        SaturationMode v1101 = SaturationMode::ON;
        TCVT(v1098, v999, v1100, v1101);
        // pto: %25
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v1102 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v733, v87);
        // pto: %25
        ;
        uint64_t v1103 = (uint64_t) v69;
        TASSIGN(v1102, v1103);
        pipe_barrier(PIPE_V);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID5);
        RoundMode v1104 = RoundMode::CAST_ROUND;
        SaturationMode v1105 = SaturationMode::ON;
        TCVT(v1102, v1012, v1104, v1105);
        // pto: %26
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v1106 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v733, v87);
        // pto: %26
        ;
        uint64_t v1107 = (uint64_t) v39;
        TASSIGN(v1106, v1107);
        pipe_barrier(PIPE_V);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID6);
        RoundMode v1108 = RoundMode::CAST_ROUND;
        SaturationMode v1109 = SaturationMode::ON;
        TCVT(v1106, v1026, v1108, v1109);
        // pto: %27
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v1110 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v733, v87);
        // pto: %27
        ;
        uint64_t v1111 = (uint64_t) v70;
        TASSIGN(v1110, v1111);
        pipe_barrier(PIPE_V);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID7);
        RoundMode v1112 = RoundMode::CAST_ROUND;
        SaturationMode v1113 = SaturationMode::ON;
        TCVT(v1110, v1040, v1112, v1113);
        // pto: %28
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v1114 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v733, v87);
        // pto: %28
        ;
        uint64_t v1115 = (uint64_t) v42;
        TASSIGN(v1114, v1115);
        TROWEXPANDMUL(v1114, v1098, v911);
        // pto: %29
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v1116 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v733, v87);
        // pto: %29
        ;
        uint64_t v1117 = (uint64_t) v69;
        TASSIGN(v1116, v1117);
        TROWEXPANDMUL(v1116, v1102, v919);
        // pto: %30
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v1118 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v733, v87);
        // pto: %30
        ;
        uint64_t v1119 = (uint64_t) v39;
        TASSIGN(v1118, v1119);
        TROWEXPANDMUL(v1118, v1106, v927);
        // pto: %31
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v1120 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v733, v87);
        // pto: %31
        ;
        uint64_t v1121 = (uint64_t) v70;
        TASSIGN(v1120, v1121);
        pipe_barrier(PIPE_V);
        TROWEXPANDMUL(v1120, v1110, v935);
        // pto: %32
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v1122 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v733, v87);
        // pto: %32
        ;
        uint64_t v1123 = (uint64_t) v42;
        TASSIGN(v1122, v1123);
        TADD(v1122, v1114, v1116);
        // pto: %33
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v1124 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v733, v87);
        // pto: %33
        ;
        uint64_t v1125 = (uint64_t) v69;
        TASSIGN(v1124, v1125);
        pipe_barrier(PIPE_V);
        TADD(v1124, v1118, v1120);
        // pto: %34
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v1126 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v733, v87);
        // pto: %34
        ;
        uint64_t v1127 = (uint64_t) v42;
        TASSIGN(v1126, v1127);
        pipe_barrier(PIPE_V);
        TADD(v1126, v1122, v1124);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
        // pto: %35
        ;
        Tile<TileType::Vec, bfloat16_t, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v1128 = Tile<TileType::Vec, bfloat16_t, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v733, v87);
        // pto: %35
        ;
        uint64_t v1129 = (uint64_t) v42;
        TASSIGN(v1128, v1129);
        pipe_barrier(PIPE_V);
        RoundMode v1130 = RoundMode::CAST_RINT;
        SaturationMode v1131 = SaturationMode::ON;
        TCVT(v1128, v1126, v1130, v1131);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        v1128.SetValidShape(v733, v87);
        // pto: %431
        ;
        const int64_t v1132 = 0;
        // pto: %431
        ;
        __gm__ bfloat16_t* v1133 = PTOAS__GLOBAL_TENSOR_DATA(v194);
        // pto: %431
        ;
        const int64_t v1134 = 1;
        // pto: %431
        ;
        const int64_t v1135 = 1;
        // pto: %431
        ;
        const int64_t v1136 = 1;
        // pto: %431
        ;
        int64_t v1137 = v733 * v79;
        // pto: %431
        ;
        int64_t v1138 = v1136 * v1137;
        // pto: %431
        ;
        pto::Shape<1, 1, 1, -1, 512> v1139 = pto::Shape<1, 1, 1, -1, 512>(v1134, v1135, v1136, v733, v87);
        // pto: %431
        ;
        pto::Stride<-1, -1, -1, -1, -1> v1140 = pto::Stride<-1, -1, -1, -1, -1>(v1135 * v1138, v1138, v1137, v79, v73);
        // pto: %431
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v1141 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v1133 + (v1132 + v763 * v79 + v1001 * v73), v1139, v1140);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        pipe_barrier(PIPE_MTE3);
        TSTORE(v1141, v1128);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID6);
      };
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      set_flag(PIPE_MTE3, PIPE_V, EVENT_ID7);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID4);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID5);
      // pto: %432
      ;
      uint64_t v1142 = (uint64_t) v706;
      // pto: %432, %433
      ;
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID7);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID5);
      if ((int64_t) (v1142 - (uint64_t) v940) == v73) {
        // pto: %434
        ;
        int64_t v1143 = (int64_t) ((uint64_t) v940 * (uint64_t) v87);
        // pto: %37
        ;
        Tile<TileType::Vec, bfloat16_t, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v1144 = Tile<TileType::Vec, bfloat16_t, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v733, v87);
        // pto: %37
        ;
        uint64_t v1145 = (uint64_t) v67;
        TASSIGN(v1144, v1145);
        // pto: %437
        ;
        int64_t v1146 = v1143 < v86 ? v86 : v1143;
        // pto: %438
        ;
        const int64_t v1147 = 0;
        // pto: %438
        ;
        __gm__ bfloat16_t* v1148 = PTOAS__GLOBAL_TENSOR_DATA(v133);
        // pto: %438
        ;
        const int64_t v1149 = 1;
        // pto: %438
        ;
        const int64_t v1150 = 1;
        // pto: %438
        ;
        const int64_t v1151 = 1;
        // pto: %438
        ;
        int64_t v1152 = v733 * v75;
        // pto: %438
        ;
        int64_t v1153 = v1151 * v1152;
        // pto: %438
        ;
        pto::Shape<1, 1, 1, -1, 512> v1154 = pto::Shape<1, 1, 1, -1, 512>(v1149, v1150, v1151, v733, v87);
        // pto: %438
        ;
        pto::Stride<-1, -1, -1, -1, -1> v1155 = pto::Stride<-1, -1, -1, -1, -1>(v1150 * v1153, v1153, v1152, v75, v73);
        // pto: %438
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v1156 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v1148 + (v1147 + v763 * v75 + v1146 * v73), v1154, v1155);
        TLOAD(v1144, v1156);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        // pto: %38
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v1157 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v733, v87);
        // pto: %38
        ;
        uint64_t v1158 = (uint64_t) v40;
        TASSIGN(v1157, v1158);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        pipe_barrier(PIPE_V);
        RoundMode v1159 = RoundMode::CAST_ROUND;
        SaturationMode v1160 = SaturationMode::ON;
        TCVT(v1157, v1144, v1159, v1160);
        // pto: %39
        ;
        Tile<TileType::Vec, bfloat16_t, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v1161 = Tile<TileType::Vec, bfloat16_t, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v733, v87);
        // pto: %39
        ;
        uint64_t v1162 = (uint64_t) v38;
        TASSIGN(v1161, v1162);
        // pto: %440
        ;
        int64_t v1163 = (int64_t) ((uint64_t) v1143 + (uint64_t) v79);
        // pto: %441
        ;
        int64_t v1164 = v1163 < v86 ? v86 : v1163;
        // pto: %442
        ;
        const int64_t v1165 = 0;
        // pto: %442
        ;
        __gm__ bfloat16_t* v1166 = PTOAS__GLOBAL_TENSOR_DATA(v133);
        // pto: %442
        ;
        const int64_t v1167 = 1;
        // pto: %442
        ;
        const int64_t v1168 = 1;
        // pto: %442
        ;
        const int64_t v1169 = 1;
        // pto: %442
        ;
        int64_t v1170 = v733 * v75;
        // pto: %442
        ;
        int64_t v1171 = v1169 * v1170;
        // pto: %442
        ;
        pto::Shape<1, 1, 1, -1, 512> v1172 = pto::Shape<1, 1, 1, -1, 512>(v1167, v1168, v1169, v733, v87);
        // pto: %442
        ;
        pto::Stride<-1, -1, -1, -1, -1> v1173 = pto::Stride<-1, -1, -1, -1, -1>(v1168 * v1171, v1171, v1170, v75, v73);
        // pto: %442
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v1174 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v1166 + (v1165 + v763 * v75 + v1164 * v73), v1172, v1173);
        TLOAD(v1161, v1174);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        // pto: %40
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v1175 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v733, v87);
        // pto: %40
        ;
        uint64_t v1176 = (uint64_t) v67;
        TASSIGN(v1175, v1176);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        pipe_barrier(PIPE_V);
        RoundMode v1177 = RoundMode::CAST_ROUND;
        SaturationMode v1178 = SaturationMode::ON;
        TCVT(v1175, v1161, v1177, v1178);
        // pto: %41
        ;
        Tile<TileType::Vec, bfloat16_t, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v1179 = Tile<TileType::Vec, bfloat16_t, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v733, v87);
        // pto: %41
        ;
        uint64_t v1180 = (uint64_t) v68;
        TASSIGN(v1179, v1180);
        // pto: %444
        ;
        int64_t v1181 = (int64_t) ((uint64_t) v1143 + (uint64_t) v114);
        // pto: %445
        ;
        int64_t v1182 = v1181 < v86 ? v86 : v1181;
        // pto: %446
        ;
        const int64_t v1183 = 0;
        // pto: %446
        ;
        __gm__ bfloat16_t* v1184 = PTOAS__GLOBAL_TENSOR_DATA(v133);
        // pto: %446
        ;
        const int64_t v1185 = 1;
        // pto: %446
        ;
        const int64_t v1186 = 1;
        // pto: %446
        ;
        const int64_t v1187 = 1;
        // pto: %446
        ;
        int64_t v1188 = v733 * v75;
        // pto: %446
        ;
        int64_t v1189 = v1187 * v1188;
        // pto: %446
        ;
        pto::Shape<1, 1, 1, -1, 512> v1190 = pto::Shape<1, 1, 1, -1, 512>(v1185, v1186, v1187, v733, v87);
        // pto: %446
        ;
        pto::Stride<-1, -1, -1, -1, -1> v1191 = pto::Stride<-1, -1, -1, -1, -1>(v1186 * v1189, v1189, v1188, v75, v73);
        // pto: %446
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v1192 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v1184 + (v1183 + v763 * v75 + v1182 * v73), v1190, v1191);
        TLOAD(v1179, v1192);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        // pto: %42
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v1193 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v733, v87);
        // pto: %42
        ;
        uint64_t v1194 = (uint64_t) v38;
        TASSIGN(v1193, v1194);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        pipe_barrier(PIPE_V);
        RoundMode v1195 = RoundMode::CAST_ROUND;
        SaturationMode v1196 = SaturationMode::ON;
        TCVT(v1193, v1179, v1195, v1196);
        // pto: %43
        ;
        Tile<TileType::Vec, bfloat16_t, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v1197 = Tile<TileType::Vec, bfloat16_t, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v733, v87);
        // pto: %43
        ;
        uint64_t v1198 = (uint64_t) v42;
        TASSIGN(v1197, v1198);
        // pto: %448
        ;
        int64_t v1199 = (int64_t) ((uint64_t) v1143 + (uint64_t) v115);
        // pto: %449
        ;
        int64_t v1200 = v1199 < v86 ? v86 : v1199;
        // pto: %450
        ;
        const int64_t v1201 = 0;
        // pto: %450
        ;
        __gm__ bfloat16_t* v1202 = PTOAS__GLOBAL_TENSOR_DATA(v133);
        // pto: %450
        ;
        const int64_t v1203 = 1;
        // pto: %450
        ;
        const int64_t v1204 = 1;
        // pto: %450
        ;
        const int64_t v1205 = 1;
        // pto: %450
        ;
        int64_t v1206 = v733 * v75;
        // pto: %450
        ;
        int64_t v1207 = v1205 * v1206;
        // pto: %450
        ;
        pto::Shape<1, 1, 1, -1, 512> v1208 = pto::Shape<1, 1, 1, -1, 512>(v1203, v1204, v1205, v733, v87);
        // pto: %450
        ;
        pto::Stride<-1, -1, -1, -1, -1> v1209 = pto::Stride<-1, -1, -1, -1, -1>(v1204 * v1207, v1207, v1206, v75, v73);
        // pto: %450
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v1210 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v1202 + (v1201 + v763 * v75 + v1200 * v73), v1208, v1209);
        TLOAD(v1197, v1210);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        // pto: %44
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v1211 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v733, v87);
        // pto: %44
        ;
        uint64_t v1212 = (uint64_t) v68;
        TASSIGN(v1211, v1212);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        pipe_barrier(PIPE_V);
        RoundMode v1213 = RoundMode::CAST_ROUND;
        SaturationMode v1214 = SaturationMode::ON;
        TCVT(v1211, v1197, v1213, v1214);
        // pto: %45
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v1215 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v733, v87);
        // pto: %45
        ;
        uint64_t v1216 = (uint64_t) v40;
        TASSIGN(v1215, v1216);
        TROWEXPANDMUL(v1215, v1157, v911);
        // pto: %46
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v1217 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v733, v87);
        // pto: %46
        ;
        uint64_t v1218 = (uint64_t) v67;
        TASSIGN(v1217, v1218);
        TROWEXPANDMUL(v1217, v1175, v919);
        // pto: %47
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v1219 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v733, v87);
        // pto: %47
        ;
        uint64_t v1220 = (uint64_t) v38;
        TASSIGN(v1219, v1220);
        TROWEXPANDMUL(v1219, v1193, v927);
        // pto: %48
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v1221 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v733, v87);
        // pto: %48
        ;
        uint64_t v1222 = (uint64_t) v68;
        TASSIGN(v1221, v1222);
        pipe_barrier(PIPE_V);
        TROWEXPANDMUL(v1221, v1211, v935);
        // pto: %49
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v1223 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v733, v87);
        // pto: %49
        ;
        uint64_t v1224 = (uint64_t) v40;
        TASSIGN(v1223, v1224);
        TADD(v1223, v1215, v1217);
        // pto: %50
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v1225 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v733, v87);
        // pto: %50
        ;
        uint64_t v1226 = (uint64_t) v67;
        TASSIGN(v1225, v1226);
        pipe_barrier(PIPE_V);
        TADD(v1225, v1219, v1221);
        // pto: %51
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v1227 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v733, v87);
        // pto: %51
        ;
        uint64_t v1228 = (uint64_t) v40;
        TASSIGN(v1227, v1228);
        pipe_barrier(PIPE_V);
        TADD(v1227, v1223, v1225);
        // pto: %52
        ;
        Tile<TileType::Vec, bfloat16_t, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v1229 = Tile<TileType::Vec, bfloat16_t, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v733, v87);
        // pto: %52
        ;
        uint64_t v1230 = (uint64_t) v40;
        TASSIGN(v1229, v1230);
        pipe_barrier(PIPE_V);
        RoundMode v1231 = RoundMode::CAST_RINT;
        SaturationMode v1232 = SaturationMode::ON;
        TCVT(v1229, v1227, v1231, v1232);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        v1229.SetValidShape(v733, v87);
        // pto: %454
        ;
        const int64_t v1233 = 0;
        // pto: %454
        ;
        __gm__ bfloat16_t* v1234 = PTOAS__GLOBAL_TENSOR_DATA(v194);
        // pto: %454
        ;
        const int64_t v1235 = 1;
        // pto: %454
        ;
        const int64_t v1236 = 1;
        // pto: %454
        ;
        const int64_t v1237 = 1;
        // pto: %454
        ;
        int64_t v1238 = v733 * v79;
        // pto: %454
        ;
        int64_t v1239 = v1237 * v1238;
        // pto: %454
        ;
        pto::Shape<1, 1, 1, -1, 512> v1240 = pto::Shape<1, 1, 1, -1, 512>(v1235, v1236, v1237, v733, v87);
        // pto: %454
        ;
        pto::Stride<-1, -1, -1, -1, -1> v1241 = pto::Stride<-1, -1, -1, -1, -1>(v1236 * v1239, v1239, v1238, v79, v73);
        // pto: %454
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v1242 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v1234 + (v1233 + v763 * v79 + v1146 * v73), v1240, v1241);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        TSTORE(v1242, v1229);
      };
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
    };
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID6);
  }
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID4);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID5);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID6);
  #endif // __DAV_VEC__

  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}